//! The order book: price levels in two `BTreeMap<tick, Level>`, each level a FIFO intrusive doubly-linked list of
//! order nodes held in a slab (O(1) cancel from anywhere in the queue, no allocation per order after warm-up).
//! See the crate docs for the semantics and the assumptions behind them.

use std::collections::BTreeMap;

use at_core::types::InstrumentRules;
use at_risk::FeeRates;
use serde::Serialize;

use crate::grid::Grid;
use crate::hash::IntMap;
use crate::types::*;

const NIL: u32 = u32::MAX;

/// Engine parameters that are not instrument rules.
#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
pub struct EngineConfig {
    /// Default market-order protection when an order does not carry its own band.
    pub market_band: Band,
    /// Reduce-only orders are exempt from the minimum notional (they may close a position smaller than it).
    pub reduce_only_exempt_min_notional: bool,
}

impl Default for EngineConfig {
    fn default() -> Self {
        Self { market_band: Band::Percent(1.0), reduce_only_exempt_min_notional: true }
    }
}

/// Instrument limits in grid units.
#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
pub struct Limits {
    pub min_lots: i64,
    pub max_lots: i64,
    pub max_mkt_lots: i64,
    pub min_notional: f64,
}

#[derive(Clone, Debug)]
struct Node {
    id: OrderId,
    owner: OwnerId,
    side: Side,
    tif: Tif,
    ro: bool,
    price: i64,
    qty: i64,
    cum: i64,
    leaves: i64,
    prio: u64,
    prev: u32,
    next: u32,
}

#[derive(Clone, Copy, Debug)]
struct Level {
    head: u32,
    tail: u32,
    total: i64,
    count: u32,
}

/// Per-owner ledger (one-way position mode). Quantities in lots; cash and fees in quote currency.
/// `cash` is the signed traded notional (sells add, buys subtract), so PnL = cash + position * mark - fees.
#[derive(Clone, Debug, Default, Serialize)]
pub struct OwnerState {
    pub position: i64,
    pub cash: f64,
    pub fees: f64,
    pub maker_lots: i64,
    pub taker_lots: i64,
    pub trades: u64,
    #[serde(skip)]
    ro: Vec<OrderId>,
}

/// Read-only view of a live order.
#[derive(Clone, Debug, PartialEq)]
pub struct OrderView {
    pub id: OrderId,
    pub owner: OwnerId,
    pub side: Side,
    pub tif: Tif,
    pub reduce_only: bool,
    pub price: i64,
    pub qty: i64,
    pub cum: i64,
    pub leaves: i64,
    /// Open qty ahead of this order at its price level (queue position).
    pub queue_ahead: i64,
}

/// L2 view: (price ticks, lots), bids best (highest) first, asks best (lowest) first.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct L2 {
    pub bids: Vec<(i64, i64)>,
    pub asks: Vec<(i64, i64)>,
}

pub struct OrderBook {
    grid: Grid,
    limits: Limits,
    fees: FeeRates,
    cfg: EngineConfig,
    bids: BTreeMap<i64, Level>,
    asks: BTreeMap<i64, Level>,
    nodes: Vec<Node>,
    free: Vec<u32>,
    index: IntMap<OrderId, u32>,
    owners: Vec<OwnerState>,
    trade_seq: u64,
    prio_seq: u64,
    version: u64,
    touched: Vec<OwnerId>,
}

#[inline]
fn reducible(pos: i64, side: Side) -> i64 {
    match side {
        Side::Buy => (-pos).max(0),
        Side::Sell => pos.max(0),
    }
}

/// Does a `side` order at `px` cross the opposite best?
#[inline]
fn crosses(bids: &BTreeMap<i64, Level>, asks: &BTreeMap<i64, Level>, side: Side, px: i64) -> bool {
    match side {
        Side::Buy => asks.first_key_value().is_some_and(|(a, _)| px >= *a),
        Side::Sell => bids.last_key_value().is_some_and(|(b, _)| px <= *b),
    }
}

fn remove_id(list: &mut Vec<OrderId>, id: OrderId) {
    if let Some(i) = list.iter().position(|x| *x == id) {
        list.swap_remove(i);
    }
}

impl OrderBook {
    /// `rules` gives tick size, qty step, min / max qty (limit and market) and min notional; `fees` the maker / taker
    /// rates (bps of notional; at_risk::FeeSchedule::rates resolves them from config/fees.yaml).
    pub fn new(rules: &InstrumentRules, fees: FeeRates) -> Result<Self, String> {
        Self::with_config(rules, fees, EngineConfig::default())
    }

    pub fn with_config(rules: &InstrumentRules, fees: FeeRates, cfg: EngineConfig) -> Result<Self, String> {
        let grid = Grid::new(rules.tick_size, rules.qty_step)?;
        let lim = |x: f64, dflt: i64| if x > 0.0 && x.is_finite() { grid.lots_nearest(x) } else { dflt };
        let max_lots = lim(rules.max_qty, i64::MAX);
        let limits = Limits {
            min_lots: lim(rules.min_qty, 1).max(1),
            max_lots,
            max_mkt_lots: lim(rules.max_mkt_qty, max_lots),
            min_notional: if rules.min_notional.is_finite() { rules.min_notional.max(0.0) } else { 0.0 },
        };
        match cfg.market_band {
            Band::Percent(p) if !(p.is_finite() && p > 0.0) => return Err(format!("market band percent must be > 0, got {p}")),
            Band::Ticks(n) if n < 1 => return Err(format!("market band ticks must be >= 1, got {n}")),
            _ => {}
        }
        Ok(Self {
            grid,
            limits,
            fees,
            cfg,
            bids: BTreeMap::new(),
            asks: BTreeMap::new(),
            nodes: Vec::new(),
            free: Vec::new(),
            index: IntMap::default(),
            owners: vec![OwnerState::default()],
            trade_seq: 0,
            prio_seq: 0,
            version: 0,
            touched: Vec::new(),
        })
    }

    pub fn grid(&self) -> &Grid {
        &self.grid
    }

    pub fn limits(&self) -> &Limits {
        &self.limits
    }

    pub fn fees(&self) -> FeeRates {
        self.fees
    }

    pub fn config(&self) -> &EngineConfig {
        &self.cfg
    }

    /// Book version: increases on every change of resting liquidity (used as the ob200 `seq`).
    pub fn version(&self) -> u64 {
        self.version
    }

    /// Number of trades so far (the last trade's seq).
    pub fn trade_count(&self) -> u64 {
        self.trade_seq
    }

    pub fn best_bid(&self) -> Option<(i64, i64)> {
        self.bids.last_key_value().map(|(p, l)| (*p, l.total))
    }

    pub fn best_ask(&self) -> Option<(i64, i64)> {
        self.asks.first_key_value().map(|(p, l)| (*p, l.total))
    }

    pub fn n_orders(&self) -> usize {
        self.index.len()
    }

    pub fn n_levels(&self) -> (usize, usize) {
        (self.bids.len(), self.asks.len())
    }

    pub fn owner(&self, owner: OwnerId) -> Option<&OwnerState> {
        self.owners.get(owner as usize)
    }

    /// All owner ledgers, indexed by owner id.
    pub fn owners(&self) -> &[OwnerState] {
        &self.owners
    }

    pub fn position(&self, owner: OwnerId) -> i64 {
        self.owners.get(owner as usize).map_or(0, |o| o.position)
    }

    pub fn contains(&self, id: OrderId) -> bool {
        self.index.contains_key(&id)
    }

    pub fn order(&self, id: OrderId) -> Option<OrderView> {
        let slot = *self.index.get(&id)?;
        let n = &self.nodes[slot as usize];
        let mut ahead = 0;
        let mut s = n.prev;
        while s != NIL {
            ahead += self.nodes[s as usize].leaves;
            s = self.nodes[s as usize].prev;
        }
        Some(OrderView { id, owner: n.owner, side: n.side, tif: n.tif, reduce_only: n.ro, price: n.price, qty: n.qty, cum: n.cum, leaves: n.leaves, queue_ahead: ahead })
    }

    /// Order ids at one price level in queue (time priority) order.
    pub fn queue(&self, side: Side, price: i64) -> Vec<OrderId> {
        let map = match side {
            Side::Buy => &self.bids,
            Side::Sell => &self.asks,
        };
        let mut out = vec![];
        if let Some(l) = map.get(&price) {
            let mut s = l.head;
            while s != NIL {
                out.push(self.nodes[s as usize].id);
                s = self.nodes[s as usize].next;
            }
        }
        out
    }

    /// Open qty (lots) resting at one price level; 0 when the level is empty (used by `atx mm-replay`).
    pub fn level_qty(&self, side: Side, price: i64) -> i64 {
        let map = match side {
            Side::Buy => &self.bids,
            Side::Sell => &self.asks,
        };
        map.get(&price).map_or(0, |l| l.total)
    }

    /// Deepest level on a side (lowest bid / highest ask): the edge of a depth-limited replayed view.
    pub fn worst_price(&self, side: Side) -> Option<i64> {
        match side {
            Side::Buy => self.bids.first_key_value().map(|(p, _)| *p),
            Side::Sell => self.asks.last_key_value().map(|(p, _)| *p),
        }
    }

    /// Top `depth` levels per side.
    pub fn snapshot(&self, depth: usize) -> L2 {
        let mut l2 = L2::default();
        self.snapshot_into(depth, &mut l2);
        l2
    }

    pub fn snapshot_into(&self, depth: usize, l2: &mut L2) {
        l2.bids.clear();
        l2.asks.clear();
        l2.bids.extend(self.bids.iter().rev().take(depth).map(|(p, l)| (*p, l.total)));
        l2.asks.extend(self.asks.iter().take(depth).map(|(p, l)| (*p, l.total)));
    }

    // ------------------------------------------------------------------ public operations

    pub fn submit(&mut self, o: &NewOrder) -> Vec<Event> {
        let mut out = Vec::new();
        self.submit_into(o, &mut out);
        out
    }

    pub fn cancel(&mut self, ts: i64, id: OrderId) -> Vec<Event> {
        let mut out = Vec::new();
        self.cancel_into(ts, id, &mut out);
        out
    }

    /// Amend price and / or total qty (Bybit `qty` = order quantity after modification, including the filled part).
    pub fn amend(&mut self, ts: i64, id: OrderId, price: Option<f64>, qty: Option<f64>) -> Vec<Event> {
        let mut out = Vec::new();
        self.amend_into(ts, id, price, qty, &mut out);
        out
    }

    /// Validate, match and (GTC / PostOnly) rest a new order; events are appended to `out`.
    pub fn submit_into(&mut self, o: &NewOrder, out: &mut Vec<Event>) {
        let reject = |out: &mut Vec<Event>, reason| out.push(Event::Reject(Reject { ts: o.ts, id: o.id, owner: o.owner, op: Op::New, reason }));
        if o.owner > MAX_OWNER {
            return reject(out, Reason::BadInput);
        }
        if self.index.contains_key(&o.id) {
            return reject(out, Reason::DuplicatedId);
        }
        if o.qty == 0.0 {
            return reject(out, Reason::QtyCannotBeZero);
        }
        let mut lots = match self.grid.lots(o.qty) {
            Some(l) if l > 0 => l,
            Some(0) => return reject(out, Reason::QtyCannotBeZero),
            _ => return reject(out, Reason::InvalidQty),
        };
        let market = o.ord_type == OrdType::Market;
        if market && o.tif == Tif::PostOnly {
            return reject(out, Reason::MarketOrderCannotBePostOnly);
        }
        // Limit price; for a market order the protected IOC price from the opposite best (None: empty side).
        let (limit, ref_px) = if market {
            let best = match o.side {
                Side::Buy => self.asks.first_key_value().map(|(p, _)| *p),
                Side::Sell => self.bids.last_key_value().map(|(p, _)| *p),
            };
            (best.map(|b| self.band_price(o.side, b, o.band.unwrap_or(self.cfg.market_band))), best)
        } else {
            let t = match o.price {
                Some(p) if p.is_finite() && p > 0.0 => match self.grid.ticks(p) {
                    Some(t) if t > 0 => t,
                    _ => return reject(out, Reason::InvalidPriceScale),
                },
                _ => return reject(out, Reason::LimitOrderInvalidPrice),
            };
            (Some(t), Some(t))
        };
        if lots < self.limits.min_lots {
            return reject(out, Reason::QtyBelowMin);
        }
        if !o.reduce_only && lots > if market { self.limits.max_mkt_lots } else { self.limits.max_lots } {
            return reject(out, Reason::QtyAboveMax);
        }
        self.ensure_owner(o.owner);
        let mut adjusted = false;
        if o.reduce_only {
            let r = reducible(self.owners[o.owner as usize].position, o.side);
            if r == 0 {
                return reject(out, Reason::ReduceOnlyRuleNotSatisfied);
            }
            if lots > r {
                lots = r;
                adjusted = true;
            }
        }
        if !(o.reduce_only && self.cfg.reduce_only_exempt_min_notional)
            && let Some(px) = ref_px
            && self.grid.price(px) * self.grid.qty(lots) < self.limits.min_notional * (1.0 - 1e-12)
        {
            return reject(out, Reason::NotionalBelowMin);
        }
        let Some(limit) = limit else {
            // Market order with an empty opposite side: accepted and cancelled unfilled.
            out.push(Event::Order(OrderUpdate { ts: o.ts, id: o.id, owner: o.owner, side: o.side, price: 0, qty: lots, leaves: 0, cum: 0, status: Status::Cancelled, reason: Some(Reason::NoImmediateQtyToFill) }));
            return;
        };
        if o.tif == Tif::PostOnly && crosses(&self.bids, &self.asks, o.side, limit) {
            return reject(out, Reason::PostOnlyWillTakeLiquidity);
        }
        if o.tif == Tif::FOK && !market && self.fillable(o.side, o.owner, limit, lots) < lots {
            return reject(out, Reason::CancelForNoFullFill);
        }

        let cum = self.take(o.ts, o.id, o.owner, o.side, limit, lots, out);
        let leaves = lots - cum;
        let rests = !market && matches!(o.tif, Tif::GTC | Tif::PostOnly);
        let (status, reason) = if leaves == 0 {
            (Status::Filled, None)
        } else if rests {
            self.rest(o.id, o.owner, o.side, o.tif, o.reduce_only, limit, lots, cum, leaves);
            if o.reduce_only {
                self.owners[o.owner as usize].ro.push(o.id);
                self.touched.push(o.owner);
            }
            (if cum > 0 { Status::PartiallyFilled } else { Status::New }, None)
        } else {
            (Status::Cancelled, Some(if cum == 0 { Reason::NoImmediateQtyToFill } else { Reason::IocRemainderCancelled }))
        };
        let reason = reason.or(if adjusted { Some(Reason::ReduceOnlyQtyAdjusted) } else { None });
        let qty = if status == Status::Cancelled { lots } else { cum + leaves };
        out.push(Event::Order(OrderUpdate { ts: o.ts, id: o.id, owner: o.owner, side: o.side, price: limit, qty, leaves: if rests { leaves } else { 0 }, cum, status, reason }));
        self.trim_touched(o.ts, out);
    }

    pub fn cancel_into(&mut self, ts: i64, id: OrderId, out: &mut Vec<Event>) {
        let Some(&slot) = self.index.get(&id) else {
            out.push(Event::Reject(Reject { ts, id, owner: 0, op: Op::Cancel, reason: Reason::OrderNotExist }));
            return;
        };
        let n = self.remove(slot);
        if n.ro {
            remove_id(&mut self.owners[n.owner as usize].ro, id);
        }
        out.push(Event::Order(OrderUpdate { ts, id, owner: n.owner, side: n.side, price: n.price, qty: n.qty, leaves: 0, cum: n.cum, status: Status::Cancelled, reason: Some(Reason::CancelByUser) }));
    }

    pub fn amend_into(&mut self, ts: i64, id: OrderId, price: Option<f64>, qty: Option<f64>, out: &mut Vec<Event>) {
        let Some(&slot) = self.index.get(&id) else {
            out.push(Event::Reject(Reject { ts, id, owner: 0, op: Op::Amend, reason: Reason::OrderNotExist }));
            return;
        };
        let n = self.nodes[slot as usize].clone();
        let reject = |out: &mut Vec<Event>, reason| out.push(Event::Reject(Reject { ts, id, owner: n.owner, op: Op::Amend, reason }));
        let new_px = match price {
            None => n.price,
            Some(p) if p.is_finite() && p > 0.0 => match self.grid.ticks(p) {
                Some(t) if t > 0 => t,
                _ => return reject(out, Reason::InvalidPriceScale),
            },
            Some(_) => return reject(out, Reason::LimitOrderInvalidPrice),
        };
        let new_qty = match qty {
            None => n.qty,
            Some(0.0) => return reject(out, Reason::QtyCannotBeZero),
            Some(q) => match self.grid.lots(q) {
                Some(l) if l > 0 => l,
                _ => return reject(out, Reason::InvalidQty),
            },
        };
        if new_px == n.price && new_qty == n.qty {
            return reject(out, Reason::OrderNotModified);
        }
        if new_qty <= n.cum {
            return reject(out, Reason::AmendQtyNotAboveFilled);
        }
        if new_qty < self.limits.min_lots {
            return reject(out, Reason::QtyBelowMin);
        }
        if !n.ro && new_qty > self.limits.max_lots {
            return reject(out, Reason::QtyAboveMax);
        }
        let mut new_leaves = new_qty - n.cum;
        let mut adjusted = false;
        if n.ro {
            let r = reducible(self.owners[n.owner as usize].position, n.side);
            if r == 0 {
                return reject(out, Reason::ReduceOnlyRuleNotSatisfied);
            }
            if new_leaves > r {
                new_leaves = r;
                adjusted = true;
            }
        }
        if !(n.ro && self.cfg.reduce_only_exempt_min_notional) && self.grid.price(new_px) * self.grid.qty(new_qty) < self.limits.min_notional * (1.0 - 1e-12) {
            return reject(out, Reason::NotionalBelowMin);
        }
        let price_changed = new_px != n.price;
        if n.tif == Tif::PostOnly && price_changed && crosses(&self.bids, &self.asks, n.side, new_px) {
            return reject(out, Reason::PostOnlyWillTakeLiquidity);
        }
        let reason = Some(if adjusted { Reason::ReduceOnlyQtyAdjusted } else { Reason::Amended });
        let eff_qty = n.cum + new_leaves;
        if !price_changed && eff_qty <= n.qty {
            // Pure qty decrease: in place, queue position kept.
            self.cut(slot, n.leaves - new_leaves);
            let status = if n.cum > 0 { Status::PartiallyFilled } else { Status::New };
            out.push(Event::Order(OrderUpdate { ts, id, owner: n.owner, side: n.side, price: n.price, qty: eff_qty, leaves: new_leaves, cum: n.cum, status, reason }));
        } else {
            // Price change or qty increase: leaves the queue, may cross (as taker), re-enters at the back.
            self.remove(slot);
            let mut cum = n.cum;
            let mut leaves = new_leaves;
            if price_changed && crosses(&self.bids, &self.asks, n.side, new_px) {
                let f = self.take(ts, id, n.owner, n.side, new_px, leaves, out);
                cum += f;
                leaves -= f;
            }
            let status = if leaves > 0 {
                self.rest(id, n.owner, n.side, n.tif, n.ro, new_px, cum + leaves, cum, leaves);
                if cum > 0 { Status::PartiallyFilled } else { Status::New }
            } else {
                if n.ro {
                    remove_id(&mut self.owners[n.owner as usize].ro, id);
                }
                Status::Filled
            };
            out.push(Event::Order(OrderUpdate { ts, id, owner: n.owner, side: n.side, price: new_px, qty: cum + leaves, leaves, cum, status, reason }));
        }
        if n.ro {
            self.touched.push(n.owner);
        }
        self.trim_touched(ts, out);
    }

    // ------------------------------------------------------------------ internals

    fn ensure_owner(&mut self, owner: OwnerId) {
        if self.owners.len() <= owner as usize {
            self.owners.resize_with(owner as usize + 1, OwnerState::default);
        }
    }

    fn band_price(&self, side: Side, best: i64, band: Band) -> i64 {
        match (side, band) {
            (Side::Buy, Band::Ticks(n)) => best + n.max(1),
            (Side::Sell, Band::Ticks(n)) => (best - n.max(1)).max(1),
            (Side::Buy, Band::Percent(p)) => ((best as f64) * (1.0 + p / 100.0) + 1e-9).floor() as i64,
            (Side::Sell, Band::Percent(p)) => (((best as f64) * (1.0 - p / 100.0) - 1e-9).ceil() as i64).max(1),
        }
    }

    fn side_map(&mut self, side: Side) -> &mut BTreeMap<i64, Level> {
        match side {
            Side::Buy => &mut self.bids,
            Side::Sell => &mut self.asks,
        }
    }

    /// Put a new node at the back of its level (the caller keeps the owner's reduce-only list).
    #[allow(clippy::too_many_arguments)]
    fn rest(&mut self, id: OrderId, owner: OwnerId, side: Side, tif: Tif, ro: bool, price: i64, qty: i64, cum: i64, leaves: i64) {
        self.prio_seq += 1;
        self.version += 1;
        let node = Node { id, owner, side, tif, ro, price, qty, cum, leaves, prio: self.prio_seq, prev: NIL, next: NIL };
        let slot = match self.free.pop() {
            Some(s) => {
                self.nodes[s as usize] = node;
                s
            }
            None => {
                self.nodes.push(node);
                (self.nodes.len() - 1) as u32
            }
        };
        let Self { bids, asks, nodes, .. } = self;
        let map = match side {
            Side::Buy => bids,
            Side::Sell => asks,
        };
        let lvl = map.entry(price).or_insert(Level { head: NIL, tail: NIL, total: 0, count: 0 });
        if lvl.tail == NIL {
            lvl.head = slot;
        } else {
            nodes[lvl.tail as usize].next = slot;
            nodes[slot as usize].prev = lvl.tail;
        }
        lvl.tail = slot;
        lvl.total += leaves;
        lvl.count += 1;
        self.index.insert(id, slot);
    }

    /// Unlink a node from its level, free the slot, drop the id from the index. RO lists are the caller's job.
    fn remove(&mut self, slot: u32) -> Node {
        self.version += 1;
        let n = self.nodes[slot as usize].clone();
        let Self { bids, asks, nodes, .. } = self;
        let map = match n.side {
            Side::Buy => bids,
            Side::Sell => asks,
        };
        let lvl = map.get_mut(&n.price).expect("resting order has a level");
        if n.prev == NIL {
            lvl.head = n.next;
        } else {
            nodes[n.prev as usize].next = n.next;
        }
        if n.next == NIL {
            lvl.tail = n.prev;
        } else {
            nodes[n.next as usize].prev = n.prev;
        }
        lvl.total -= n.leaves;
        lvl.count -= 1;
        if lvl.count == 0 {
            map.remove(&n.price);
        }
        self.free.push(slot);
        self.index.remove(&n.id);
        n
    }

    /// Reduce a resting order's open qty in place (priority kept).
    fn cut(&mut self, slot: u32, by: i64) {
        if by <= 0 {
            return;
        }
        self.version += 1;
        let (side, price) = {
            let n = &mut self.nodes[slot as usize];
            n.leaves -= by;
            n.qty -= by;
            (n.side, n.price)
        };
        self.side_map(side).get_mut(&price).expect("level").total -= by;
    }

    /// Match an aggressor against the opposite side up to `limit`; returns the filled lots.
    #[allow(clippy::too_many_arguments)]
    fn take(&mut self, ts: i64, id: OrderId, owner: OwnerId, side: Side, limit: i64, mut want: i64, out: &mut Vec<Event>) -> i64 {
        let Self { bids, asks, nodes, free, index, owners, trade_seq, touched, grid, fees, version, .. } = self;
        let opp = match side {
            Side::Buy => asks,
            Side::Sell => bids,
        };
        let (taker_rate, maker_rate) = (fees.taker_bps / 1e4, fees.maker_bps / 1e4);
        let mut filled = 0;
        while want > 0 {
            let entry = match side {
                Side::Buy => opp.first_entry(),
                Side::Sell => opp.last_entry(),
            };
            let Some(mut entry) = entry else { break };
            let px = *entry.key();
            if (side == Side::Buy && px > limit) || (side == Side::Sell && px < limit) {
                break;
            }
            *version += 1;
            let price_f = grid.price(px);
            let lvl = entry.get_mut();
            while want > 0 && lvl.head != NIL {
                let slot = lvl.head;
                let (m_id, m_owner, m_side, m_ro, m_leaves) = {
                    let m = &nodes[slot as usize];
                    (m.id, m.owner, m.side, m.ro, m.leaves)
                };
                let mut avail = m_leaves;
                if m_ro {
                    let cap = reducible(owners[m_owner as usize].position, m_side);
                    if cap < avail {
                        let cut = avail - cap;
                        let m = &mut nodes[slot as usize];
                        m.leaves = cap;
                        m.qty -= cut;
                        lvl.total -= cut;
                        avail = cap;
                        let st = if m.cum > 0 { Status::PartiallyFilled } else { Status::New };
                        if cap == 0 {
                            out.push(Event::Order(OrderUpdate { ts, id: m_id, owner: m_owner, side: m_side, price: px, qty: m.qty, leaves: 0, cum: m.cum, status: Status::Cancelled, reason: Some(Reason::CancelByReduceOnly) }));
                            let next = m.next;
                            lvl.head = next;
                            if next == NIL {
                                lvl.tail = NIL;
                            } else {
                                nodes[next as usize].prev = NIL;
                            }
                            lvl.count -= 1;
                            free.push(slot);
                            index.remove(&m_id);
                            remove_id(&mut owners[m_owner as usize].ro, m_id);
                            continue;
                        }
                        out.push(Event::Order(OrderUpdate { ts, id: m_id, owner: m_owner, side: m_side, price: px, qty: m.qty, leaves: cap, cum: m.cum, status: st, reason: Some(Reason::ReduceOnlyQtyAdjusted) }));
                    }
                }
                let f = want.min(avail);
                *trade_seq += 1;
                let notional = price_f * grid.qty(f);
                let (tf, mf) = (notional * taker_rate, notional * maker_rate);
                {
                    let t = &mut owners[owner as usize];
                    t.position += side.sign() * f;
                    t.cash -= side.sign() as f64 * notional;
                    t.fees += tf;
                    t.taker_lots += f;
                    t.trades += 1;
                }
                {
                    let mk = &mut owners[m_owner as usize];
                    mk.position += m_side.sign() * f;
                    mk.cash -= m_side.sign() as f64 * notional;
                    mk.fees += mf;
                    mk.maker_lots += f;
                    mk.trades += 1;
                }
                touched.push(owner);
                touched.push(m_owner);
                out.push(Event::Trade(Trade { seq: *trade_seq, ts, price: px, qty: f, taker_side: side, taker_id: id, maker_id: m_id, taker_owner: owner, maker_owner: m_owner, taker_fee: tf, maker_fee: mf }));
                want -= f;
                filled += f;
                lvl.total -= f;
                let m = &mut nodes[slot as usize];
                m.leaves -= f;
                m.cum += f;
                if m.leaves == 0 {
                    out.push(Event::Order(OrderUpdate { ts, id: m_id, owner: m_owner, side: m_side, price: px, qty: m.qty, leaves: 0, cum: m.cum, status: Status::Filled, reason: None }));
                    let next = m.next;
                    lvl.head = next;
                    if next == NIL {
                        lvl.tail = NIL;
                    } else {
                        nodes[next as usize].prev = NIL;
                    }
                    lvl.count -= 1;
                    free.push(slot);
                    index.remove(&m_id);
                    if m_ro {
                        remove_id(&mut owners[m_owner as usize].ro, m_id);
                    }
                } else {
                    out.push(Event::Order(OrderUpdate { ts, id: m_id, owner: m_owner, side: m_side, price: px, qty: m.qty, leaves: m.leaves, cum: m.cum, status: Status::PartiallyFilled, reason: None }));
                }
            }
            if lvl.head == NIL {
                entry.remove();
            }
        }
        filled
    }

    /// Dry run of `take` for FOK: the lots an aggressor could fill up to `limit`, honouring reduce-only caps of the
    /// makers as positions change along the way.
    fn fillable(&self, side: Side, owner: OwnerId, limit: i64, want: i64) -> i64 {
        fn walk<'a>(levels: impl Iterator<Item = (&'a i64, &'a Level)>, b: &OrderBook, side: Side, owner: OwnerId, limit: i64, want: i64) -> i64 {
            let mut delta: Vec<(OwnerId, i64)> = Vec::new();
            let mut acc = 0;
            for (px, lvl) in levels {
                if acc >= want || (side == Side::Buy && *px > limit) || (side == Side::Sell && *px < limit) {
                    break;
                }
                let mut s = lvl.head;
                while s != NIL && acc < want {
                    let m = &b.nodes[s as usize];
                    let mut avail = m.leaves;
                    if m.ro {
                        let d = delta.iter().find(|(o, _)| *o == m.owner).map_or(0, |x| x.1);
                        avail = avail.min(reducible(b.owners[m.owner as usize].position + d, m.side));
                    }
                    let f = avail.min(want - acc);
                    if f > 0 && m.owner != owner {
                        for (o, q) in [(m.owner, m.side.sign() * f), (owner, side.sign() * f)] {
                            match delta.iter_mut().find(|(x, _)| *x == o) {
                                Some(e) => e.1 += q,
                                None => delta.push((o, q)),
                            }
                        }
                    }
                    acc += f;
                    s = m.next;
                }
            }
            acc
        }
        match side {
            Side::Buy => walk(self.asks.iter(), self, side, owner, limit, want),
            Side::Sell => walk(self.bids.iter().rev(), self, side, owner, limit, want),
        }
    }

    /// Enforce the reduce-only rule for every owner whose position changed: resting reduce-only orders may not add
    /// up to more than the position; the order furthest from the market (then the newest) is reduced or cancelled.
    fn trim_touched(&mut self, ts: i64, out: &mut Vec<Event>) {
        if self.touched.is_empty() {
            return;
        }
        let mut touched = std::mem::take(&mut self.touched);
        touched.sort_unstable();
        touched.dedup();
        for &o in &touched {
            if !self.owners[o as usize].ro.is_empty() {
                self.trim_ro(ts, o, out);
            }
        }
        touched.clear();
        self.touched = touched;
    }

    fn trim_ro(&mut self, ts: i64, owner: OwnerId, out: &mut Vec<Event>) {
        let pos = self.owners[owner as usize].position;
        let ids = std::mem::take(&mut self.owners[owner as usize].ro);
        let mut keep: Vec<(i64, u64, u32)> = Vec::with_capacity(ids.len());
        let mut kept_ids = Vec::with_capacity(ids.len());
        for id in ids {
            let slot = self.index[&id];
            let n = &self.nodes[slot as usize];
            if reducible(pos, n.side) == 0 {
                let n = self.remove(slot);
                out.push(Event::Order(OrderUpdate { ts, id, owner, side: n.side, price: n.price, qty: n.qty, leaves: 0, cum: n.cum, status: Status::Cancelled, reason: Some(Reason::CancelByReduceOnly) }));
            } else {
                // Priority key: nearest to the market first (asks ascending, bids descending), then oldest.
                let k = if n.side == Side::Sell { n.price } else { -n.price };
                keep.push((k, n.prio, slot));
            }
        }
        keep.sort_unstable();
        let mut room = pos.abs();
        for (_, _, slot) in keep {
            let (id, leaves) = (self.nodes[slot as usize].id, self.nodes[slot as usize].leaves);
            if leaves <= room {
                room -= leaves;
                kept_ids.push(id);
            } else if room > 0 {
                self.cut(slot, leaves - room);
                let n = &self.nodes[slot as usize];
                out.push(Event::Order(OrderUpdate { ts, id, owner, side: n.side, price: n.price, qty: n.qty, leaves: n.leaves, cum: n.cum, status: if n.cum > 0 { Status::PartiallyFilled } else { Status::New }, reason: Some(Reason::ReduceOnlyQtyAdjusted) }));
                room = 0;
                kept_ids.push(id);
            } else {
                let n = self.remove(slot);
                out.push(Event::Order(OrderUpdate { ts, id, owner, side: n.side, price: n.price, qty: n.qty, leaves: 0, cum: n.cum, status: Status::Cancelled, reason: Some(Reason::CancelByReduceOnly) }));
            }
        }
        self.owners[owner as usize].ro = kept_ids;
    }

    /// Full structural check (O(orders)); for tests and debugging.
    pub fn check_invariants(&self) -> Result<(), String> {
        let mut seen = 0usize;
        for (side, map) in [(Side::Buy, &self.bids), (Side::Sell, &self.asks)] {
            for (px, l) in map {
                let (mut s, mut prev, mut total, mut count, mut last_prio) = (l.head, NIL, 0i64, 0u32, 0u64);
                if l.head == NIL {
                    return Err(format!("empty level {px} kept"));
                }
                while s != NIL {
                    let n = &self.nodes[s as usize];
                    if n.side != side || n.price != *px || n.prev != prev {
                        return Err(format!("node {} misplaced at level {px}", n.id));
                    }
                    if n.leaves <= 0 || n.qty != n.cum + n.leaves {
                        return Err(format!("order {}: qty {} != cum {} + leaves {}", n.id, n.qty, n.cum, n.leaves));
                    }
                    if n.prio <= last_prio {
                        return Err(format!("level {px}: time priority broken at {}", n.id));
                    }
                    if self.index.get(&n.id) != Some(&s) {
                        return Err(format!("index does not point at order {}", n.id));
                    }
                    last_prio = n.prio;
                    total += n.leaves;
                    count += 1;
                    prev = s;
                    s = n.next;
                }
                if l.tail != prev || total != l.total || count != l.count {
                    return Err(format!("level {px}: total {} / count {} / tail mismatch (sum {total}, n {count})", l.total, l.count));
                }
                seen += count as usize;
            }
        }
        if seen != self.index.len() {
            return Err(format!("{} indexed orders but {seen} in the book", self.index.len()));
        }
        if let (Some((b, _)), Some((a, _))) = (self.best_bid(), self.best_ask())
            && b >= a
        {
            return Err(format!("crossed book: bid {b} >= ask {a}"));
        }
        let net: i64 = self.owners.iter().map(|o| o.position).sum();
        if net != 0 {
            return Err(format!("positions do not net to zero: {net}"));
        }
        for (o, st) in self.owners.iter().enumerate() {
            let mut sum = 0;
            for id in &st.ro {
                let slot = *self.index.get(id).ok_or(format!("owner {o}: reduce-only order {id} not live"))?;
                let n = &self.nodes[slot as usize];
                if !n.ro || n.owner as usize != o || reducible(st.position, n.side) == 0 {
                    return Err(format!("owner {o}: reduce-only order {id} invalid for position {}", st.position));
                }
                sum += n.leaves;
            }
            if sum > st.position.abs() {
                return Err(format!("owner {o}: reduce-only qty {sum} exceeds position {}", st.position));
            }
        }
        let n_ro: usize = self.owners.iter().map(|o| o.ro.len()).sum();
        let n_ro_nodes = self.index.values().filter(|s| self.nodes[**s as usize].ro).count();
        if n_ro != n_ro_nodes {
            return Err(format!("reduce-only lists hold {n_ro} ids, book has {n_ro_nodes} reduce-only orders"));
        }
        Ok(())
    }
}
