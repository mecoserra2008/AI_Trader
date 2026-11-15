//! Bar-driven paper broker. Conservative by construction:
//! - if a bar touches both stop and target, the **stop** is assumed to fill first;
//! - gaps through a stop fill at the bar open (worse than the stop);
//! - a limit entry filled in a bar is immediately exposed to that same bar's stop;
//! - exits pay taker fee + slippage, exactly as the risk engine's cost model assumes.

use at_core::config::Costs;
use at_core::types::{EntryType, Position, Side};
use at_core::time::Ms;
use at_risk::{FeeRates, SizedOrder};
use serde::{Deserialize, Serialize};

use crate::Broker;

#[derive(Clone, Debug, Serialize)]
pub struct ClosedTrade {
    pub symbol: String,
    pub side: Side,
    pub qty: f64,
    pub entry_price: f64,
    pub exit_price: f64,
    pub exit_reason: &'static str,
    pub pnl: f64,
    pub r_multiple: f64,
    pub opened_ts: Ms,
    pub closed_ts: Ms,
}

#[derive(Clone, Debug)]
struct Open {
    pos: Position,
    risk_amount: f64,
}

#[derive(Clone, Debug)]
struct Pending {
    order: SizedOrder,
}

pub struct PaperBroker {
    label: String,
    cash_equity: f64,
    costs: Costs,
    open: Vec<Open>,
    pending: Vec<Pending>,
    closed: Vec<ClosedTrade>,
    last_price: std::collections::HashMap<String, (f64, Ms)>,
    seq: u64,
}

impl PaperBroker {
    pub fn new(label: &str, equity: f64, costs: Costs) -> Self {
        Self { label: label.into(), cash_equity: equity, costs, open: vec![], pending: vec![], closed: vec![], last_price: Default::default(), seq: 0 }
    }

    pub fn closed_trades(&self) -> &[ClosedTrade] {
        &self.closed
    }

    fn fee(&self, notional: f64, taker: bool) -> f64 {
        notional * if taker { self.costs.taker_bps } else { self.costs.maker_bps } / 1e4
    }

    fn slip(&self, price: f64, side: Side, opening: bool) -> f64 {
        // Slippage always moves the fill against us.
        let dir = if opening { side.sign() } else { -side.sign() };
        price * (1.0 + dir * self.costs.slippage_bps / 1e4)
    }

    fn open_position(&mut self, o: &SizedOrder, fill: f64, taker: bool, ts: Ms) {
        self.cash_equity -= self.fee(o.qty * fill, taker);
        self.open.push(Open {
            pos: Position {
                symbol: o.proposal.symbol.clone(),
                side: o.proposal.side,
                qty: o.qty,
                entry_price: fill,
                stop: o.stop,
                take_profit: o.take_profit,
                cluster: o.proposal.cluster.clone(),
                opened_ts: ts,
            },
            risk_amount: o.risk_amount,
        });
    }

    fn close_at(&mut self, i: usize, raw_price: f64, reason: &'static str, ts: Ms) {
        let o = self.open.remove(i);
        let p = &o.pos;
        let exit = self.slip(raw_price, p.side, false);
        let gross = p.qty * (exit - p.entry_price) * p.side.sign();
        let pnl = gross - self.fee(p.qty * exit, true);
        self.cash_equity += pnl;
        self.closed.push(ClosedTrade {
            symbol: p.symbol.clone(),
            side: p.side,
            qty: p.qty,
            entry_price: p.entry_price,
            exit_price: exit,
            exit_reason: reason,
            pnl,
            r_multiple: pnl / o.risk_amount,
            opened_ts: p.opened_ts,
            closed_ts: ts,
        });
    }

    /// Advance one completed bar for `symbol`.
    pub fn on_bar(&mut self, symbol: &str, ts: Ms, open: f64, high: f64, low: f64, close: f64) {
        self.last_price.insert(symbol.to_string(), (close, ts));
        // 1) pending limit entries
        let mut k = 0;
        while k < self.pending.len() {
            let pd = &self.pending[k];
            if pd.order.proposal.symbol != symbol {
                k += 1;
                continue;
            }
            let px = pd.order.entry;
            let fill = match pd.order.proposal.side {
                Side::Long if low <= px => Some(open.min(px)),
                Side::Short if high >= px => Some(open.max(px)),
                _ => None,
            };
            match fill {
                Some(f) => {
                    let o = self.pending.remove(k).order;
                    self.open_position(&o, f, false, ts);
                }
                None => k += 1,
            }
        }
        // 2) exits
        let mut i = 0;
        while i < self.open.len() {
            let p = &self.open[i].pos;
            if p.symbol != symbol {
                i += 1;
                continue;
            }
            let s = p.side.sign();
            let stop_hit = match p.side {
                Side::Long => low <= p.stop,
                Side::Short => high >= p.stop,
            };
            let tp_hit = match p.side {
                Side::Long => high >= p.take_profit,
                Side::Short => low <= p.take_profit,
            };
            if stop_hit {
                // gap through the stop fills at the open
                let px = if (open - p.stop) * s < 0.0 { open } else { p.stop };
                self.close_at(i, px, "stop", ts);
            } else if tp_hit {
                let px = if (open - p.take_profit) * s > 0.0 { open } else { p.take_profit };
                self.close_at(i, px, "take_profit", ts);
            } else {
                i += 1;
            }
        }
    }

    /// Mark-to-market equity using the last seen close of each symbol.
    pub fn marked_equity(&self) -> f64 {
        self.cash_equity
            + self
                .open
                .iter()
                .map(|o| {
                    let px = self.last_price.get(&o.pos.symbol).map(|x| x.0).unwrap_or(o.pos.entry_price);
                    o.pos.qty * (px - o.pos.entry_price) * o.pos.side.sign()
                })
                .sum::<f64>()
    }
}

impl Broker for PaperBroker {
    fn name(&self) -> &str {
        &self.label
    }

    async fn equity(&self) -> anyhow::Result<f64> {
        Ok(self.marked_equity())
    }

    async fn positions(&self) -> anyhow::Result<Vec<Position>> {
        Ok(self.open.iter().map(|o| o.pos.clone()).collect())
    }

    async fn submit(&mut self, o: &SizedOrder, link_id: &str) -> anyhow::Result<String> {
        self.seq += 1;
        let id = format!("{}-{}-{link_id}", self.label, self.seq);
        match o.proposal.entry_type {
            EntryType::Market => {
                let (px, ts) = self.last_price.get(&o.proposal.symbol).copied().unwrap_or((o.entry, 0));
                let fill = self.slip(px, o.proposal.side, true);
                self.open_position(o, fill, true, ts);
            }
            EntryType::Limit => self.pending.push(Pending { order: o.clone() }),
        }
        Ok(id)
    }

    async fn move_stop(&mut self, symbol: &str, stop: f64) -> anyhow::Result<()> {
        for o in self.open.iter_mut().filter(|o| o.pos.symbol == symbol) {
            o.pos.stop = stop;
        }
        Ok(())
    }

    async fn close(&mut self, symbol: &str) -> anyhow::Result<()> {
        self.pending.retain(|p| p.order.proposal.symbol != symbol);
        if let Some(i) = self.open.iter().position(|o| o.pos.symbol == symbol) {
            let (px, ts) = self.last_price.get(symbol).copied().unwrap_or((self.open[i].pos.entry_price, 0));
            self.close_at(i, px, "manual", ts);
        }
        Ok(())
    }
}

// ================================================================ PaperBook: intents -> paper fills

/// `data/exec/KILL` (or any flag path): present means cancel everything and flatten (paper only here).
pub fn kill_switch_active(path: &std::path::Path) -> bool {
    path.exists()
}

/// A resting or not-yet-executed paper entry, sized by the risk engine.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BookOrder {
    pub id: String,
    pub symbol: String,
    pub strategy: String,
    pub family: String,
    pub side: Side,
    pub entry_type: EntryType,
    /// Limit price, or the reference price a market entry was sized at.
    pub price: f64,
    pub qty: f64,
    pub stop: f64,
    pub target: Option<f64>,
    pub risk_usd: f64,
    pub fees: FeeRates,
    pub submitted_ts: Ms,
    /// Limit orders are cancelled after this many bars without a fill.
    pub ttl_bars: u32,
    pub bars_seen: u32,
    /// Limit pegged to the touch (H-EXE1 / R-EXE-2, config/risk.yaml paper_entry): fills at the first bar open after
    /// submission moved `peg_slip_bps` against us, as maker. None: an ordinary limit / market order.
    #[serde(default)]
    pub peg_slip_bps: Option<f64>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PaperPosition {
    pub id: String,
    pub symbol: String,
    pub strategy: String,
    pub family: String,
    pub side: Side,
    pub qty: f64,
    pub entry_price: f64,
    pub stop: f64,
    pub target: Option<f64>,
    pub risk_usd: f64,
    pub fees: FeeRates,
    pub opened_ts: Ms,
    pub entry_fee: f64,
    /// A flat intent asked to close: executes at the next bar open.
    pub exit_requested: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PaperFill {
    pub ts: Ms,
    pub id: String,
    pub symbol: String,
    pub strategy: String,
    /// buy | sell
    pub side: String,
    pub qty: f64,
    pub price: f64,
    /// maker | taker
    pub liquidity: String,
    pub fee: f64,
    /// entry | exit
    pub kind: String,
    /// entry | stop | target | exit_signal | kill
    pub reason: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PaperTrade {
    pub id: String,
    pub symbol: String,
    pub strategy: String,
    pub family: String,
    pub side: Side,
    pub qty: f64,
    pub entry_price: f64,
    pub exit_price: f64,
    pub opened_ts: Ms,
    pub closed_ts: Ms,
    pub exit_reason: String,
    pub gross_pnl: f64,
    pub fees: f64,
    pub pnl: f64,
    pub risk_usd: f64,
    pub r_multiple: f64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PaperCancel {
    pub ts: Ms,
    pub id: String,
    pub symbol: String,
    /// expired | exit_signal | kill | stop_through_before_fill | target_through_before_fill
    pub reason: String,
}

/// Bar-driven paper book for risk-sized intents. Conservative by construction:
/// - market entries fill at the **next bar open** plus slippage (taker); if that open is already
///   through the stop or the target the entry is cancelled instead;
/// - a resting limit fills at its price (maker) only when the bar **trades through** it (strictly);
///   a limit that is marketable at its first open fills there as taker (never worse than the limit);
/// - stops rest on the venue (R-EXEC-1) and trigger on the bar low/high (stand-in for MarkPrice);
///   a gap through the stop fills at the open; stop and target in one bar -> stop first;
/// - targets fill at the target (never at a better gapped open), taker + slippage (TP triggers a market order);
/// - a limit filled inside a bar is exposed to that bar's stop but not to its target (order unknown);
/// - every exit pays taker + slippage, the same cost model the risk engine sized with.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PaperBook {
    pub label: String,
    pub cash: f64,
    pub orders: Vec<BookOrder>,
    pub positions: Vec<PaperPosition>,
    /// Latest close and its bar open time, per symbol.
    pub last_close: std::collections::BTreeMap<String, (f64, Ms)>,
    #[serde(skip)]
    pub fills: Vec<PaperFill>,
    #[serde(skip)]
    pub trades: Vec<PaperTrade>,
    #[serde(skip)]
    pub cancels: Vec<PaperCancel>,
}

fn adverse(price: f64, side: Side, opening: bool, slippage_bps: f64) -> f64 {
    let dir = if opening { side.sign() } else { -side.sign() };
    price * (1.0 + dir * slippage_bps / 1e4)
}

fn order_side(side: Side, opening: bool) -> String {
    match (side, opening) {
        (Side::Long, true) | (Side::Short, false) => "buy".into(),
        _ => "sell".into(),
    }
}

impl PaperBook {
    pub fn new(label: &str, equity: f64) -> Self {
        Self { label: label.into(), cash: equity, orders: vec![], positions: vec![], last_close: Default::default(), fills: vec![], trades: vec![], cancels: vec![] }
    }

    pub fn submit(&mut self, order: BookOrder) {
        self.orders.push(order);
    }

    /// Flat intent: cancel this strategy's resting entries in `symbol` and close its position at the
    /// next open. Returns whether anything was found.
    pub fn request_exit(&mut self, symbol: &str, strategy: &str, ts: Ms) -> bool {
        let mut found = false;
        let mut keep = vec![];
        for o in std::mem::take(&mut self.orders) {
            if o.symbol == symbol && o.strategy == strategy {
                self.cancels.push(PaperCancel { ts, id: o.id, symbol: o.symbol, reason: "exit_signal".into() });
                found = true;
            } else {
                keep.push(o);
            }
        }
        self.orders = keep;
        for p in self.positions.iter_mut().filter(|p| p.symbol == symbol && p.strategy == strategy) {
            p.exit_requested = true;
            found = true;
        }
        found
    }

    /// Kill switch: cancel every resting order and close every position at its last close (taker + slippage).
    pub fn kill(&mut self, ts: Ms) {
        for o in std::mem::take(&mut self.orders) {
            self.cancels.push(PaperCancel { ts, id: o.id, symbol: o.symbol, reason: "kill".into() });
        }
        while !self.positions.is_empty() {
            let px = self.last_close.get(&self.positions[0].symbol).map(|x| x.0).unwrap_or(self.positions[0].entry_price);
            self.close(0, px, "kill", ts);
        }
    }

    fn open(&mut self, o: BookOrder, px: f64, taker: bool, ts: Ms) {
        let fee = o.qty * px * if taker { o.fees.taker_bps } else { o.fees.maker_bps } / 1e4;
        self.cash -= fee;
        self.fills.push(PaperFill {
            ts,
            id: o.id.clone(),
            symbol: o.symbol.clone(),
            strategy: o.strategy.clone(),
            side: order_side(o.side, true),
            qty: o.qty,
            price: px,
            liquidity: if taker { "taker" } else { "maker" }.into(),
            fee,
            kind: "entry".into(),
            reason: "entry".into(),
        });
        self.positions.push(PaperPosition {
            id: o.id,
            symbol: o.symbol,
            strategy: o.strategy,
            family: o.family,
            side: o.side,
            qty: o.qty,
            entry_price: px,
            stop: o.stop,
            target: o.target,
            risk_usd: o.risk_usd,
            fees: o.fees,
            opened_ts: ts,
            entry_fee: fee,
            exit_requested: false,
        });
    }

    /// Exit at `raw_px` moved against us by slippage, paying the taker fee.
    fn close(&mut self, i: usize, raw_px: f64, reason: &str, ts: Ms) {
        let p = self.positions.remove(i);
        let px = adverse(raw_px, p.side, false, p.fees.slippage_bps);
        let fee = p.qty * px * p.fees.taker_bps / 1e4;
        let gross = p.qty * (px - p.entry_price) * p.side.sign();
        self.cash += gross - fee;
        self.fills.push(PaperFill {
            ts,
            id: p.id.clone(),
            symbol: p.symbol.clone(),
            strategy: p.strategy.clone(),
            side: order_side(p.side, false),
            qty: p.qty,
            price: px,
            liquidity: "taker".into(),
            fee,
            kind: "exit".into(),
            reason: reason.into(),
        });
        let pnl = gross - fee - p.entry_fee;
        self.trades.push(PaperTrade {
            id: p.id,
            symbol: p.symbol,
            strategy: p.strategy,
            family: p.family,
            side: p.side,
            qty: p.qty,
            entry_price: p.entry_price,
            exit_price: px,
            opened_ts: p.opened_ts,
            closed_ts: ts,
            exit_reason: reason.into(),
            gross_pnl: gross,
            fees: fee + p.entry_fee,
            pnl,
            risk_usd: p.risk_usd,
            r_multiple: if p.risk_usd > 0.0 { pnl / p.risk_usd } else { 0.0 },
        });
    }

    /// Advance one completed bar (`ts` = bar open time) for `symbol`. Orders submitted at or before
    /// the bar open take part in it.
    pub fn on_bar(&mut self, symbol: &str, ts: Ms, open: f64, high: f64, low: f64, close: f64) {
        // 1) exits requested by flat intents: at this open
        while let Some(i) = self.positions.iter().position(|p| p.symbol == symbol && p.exit_requested) {
            self.close(i, open, "exit_signal", ts);
        }
        // 2) entries
        let mut filled_in_bar: Vec<String> = vec![];
        let mut keep = vec![];
        for mut o in std::mem::take(&mut self.orders) {
            if o.symbol != symbol || o.submitted_ts > ts {
                keep.push(o);
                continue;
            }
            let s = o.side.sign();
            match o.entry_type {
                EntryType::Market => {
                    if (open - o.stop) * s <= 0.0 {
                        self.cancels.push(PaperCancel { ts, id: o.id, symbol: o.symbol, reason: "stop_through_before_fill".into() });
                    } else if o.target.is_some_and(|t| (open - t) * s >= 0.0) {
                        self.cancels.push(PaperCancel { ts, id: o.id, symbol: o.symbol, reason: "target_through_before_fill".into() });
                    } else {
                        let px = adverse(open, o.side, true, o.fees.slippage_bps);
                        self.open(o, px, true, ts);
                    }
                }
                EntryType::Limit if o.peg_slip_bps.is_some() => {
                    // Pegged at the touch: the order joins the bid (buys) / ask (sells) at the first open after
                    // submission and is re-pegged as the touch moves away; H-EXE1 filled 275 of 275 replayed
                    // entries within 37 s. On 1h bars (no 1m data) the fill is the open moved peg_slip_bps against
                    // us (R-EXE-2: 1.1-1.6 bp beyond the decision-time mid), as maker. The fill is taken whether or
                    // not the bar range reaches that price: a bar that never reaches it moved toward the resting
                    // order, which would fill at least as well, and dropping those fills would keep only entries
                    // that went our way first. Same pre-fill cancels as a market order (stop or target gapped through).
                    let slip = o.peg_slip_bps.unwrap_or(0.0);
                    if (open - o.stop) * s <= 0.0 {
                        self.cancels.push(PaperCancel { ts, id: o.id, symbol: o.symbol, reason: "stop_through_before_fill".into() });
                    } else if o.target.is_some_and(|t| (open - t) * s >= 0.0) {
                        self.cancels.push(PaperCancel { ts, id: o.id, symbol: o.symbol, reason: "target_through_before_fill".into() });
                    } else {
                        let px = open * (1.0 + s * slip / 1e4);
                        self.open(o, px, false, ts);
                    }
                }
                EntryType::Limit => {
                    let marketable = (o.price - open) * s >= 0.0; // long: open <= limit
                    let traded_through = match o.side {
                        Side::Long => low < o.price,
                        Side::Short => high > o.price,
                    };
                    if o.bars_seen == 0 && marketable {
                        // crosses the book at its first open: taker, never worse than the limit
                        let px = adverse(open, o.side, true, o.fees.slippage_bps);
                        let px = if (px - o.price) * s > 0.0 { o.price } else { px };
                        self.open(o, px, true, ts);
                    } else if traded_through {
                        filled_in_bar.push(o.id.clone());
                        let px = o.price;
                        self.open(o, px, false, ts);
                    } else {
                        o.bars_seen += 1;
                        if o.bars_seen >= o.ttl_bars.max(1) {
                            self.cancels.push(PaperCancel { ts, id: o.id, symbol: o.symbol, reason: "expired".into() });
                        } else {
                            keep.push(o);
                        }
                    }
                }
            }
        }
        self.orders = keep;
        // 3) resting stops and targets
        let mut i = 0;
        while i < self.positions.len() {
            let p = &self.positions[i];
            if p.symbol != symbol {
                i += 1;
                continue;
            }
            let s = p.side.sign();
            let intrabar = filled_in_bar.contains(&p.id);
            let stop_hit = match p.side {
                Side::Long => low <= p.stop,
                Side::Short => high >= p.stop,
            };
            let tp_hit = !intrabar
                && p.target.is_some_and(|t| match p.side {
                    Side::Long => high >= t,
                    Side::Short => low <= t,
                });
            if stop_hit {
                // a gap through the stop fills at the open (only for positions held at the open)
                let px = if !intrabar && (open - p.stop) * s < 0.0 { open } else { p.stop };
                self.close(i, px, "stop", ts);
            } else if tp_hit {
                let t = p.target.unwrap_or(close);
                self.close(i, t, "target", ts);
            } else {
                i += 1;
            }
        }
        self.last_close.insert(symbol.to_string(), (close, ts));
    }

    pub fn unrealised(&self, p: &PaperPosition) -> f64 {
        let px = self.last_close.get(&p.symbol).map(|x| x.0).unwrap_or(p.entry_price);
        p.qty * (px - p.entry_price) * p.side.sign()
    }

    /// Cash plus unrealised P&L at the last closes (exit costs not deducted).
    pub fn marked_equity(&self) -> f64 {
        self.cash + self.positions.iter().map(|p| self.unrealised(p)).sum::<f64>()
    }

    /// Positions and resting entries in the risk engine's portfolio form.
    pub fn book_entries(&self) -> Vec<at_risk::BookEntry> {
        let pos = self.positions.iter().map(|p| at_risk::BookEntry { symbol: p.symbol.clone(), strategy: p.strategy.clone(), family: p.family.clone(), side: p.side, qty: p.qty, entry_price: p.entry_price, pending: false });
        let ord = self.orders.iter().map(|o| at_risk::BookEntry { symbol: o.symbol.clone(), strategy: o.strategy.clone(), family: o.family.clone(), side: o.side, qty: o.qty, entry_price: o.price, pending: true });
        pos.chain(ord).collect()
    }

    pub fn marks(&self) -> std::collections::HashMap<String, f64> {
        self.last_close.iter().map(|(k, v)| (k.clone(), v.0)).collect()
    }

    /// Take the fills, trades and cancels produced since the last drain.
    pub fn drain(&mut self) -> (Vec<PaperFill>, Vec<PaperTrade>, Vec<PaperCancel>) {
        (std::mem::take(&mut self.fills), std::mem::take(&mut self.trades), std::mem::take(&mut self.cancels))
    }
}

#[cfg(test)]
mod book_tests {
    use super::*;

    const CRYPTO: FeeRates = FeeRates { maker_bps: 2.0, taker_bps: 5.5, slippage_bps: 2.0 };
    const ZERO: FeeRates = FeeRates { maker_bps: 0.0, taker_bps: 0.0, slippage_bps: 0.0 };

    fn order(entry_type: EntryType, side: Side, price: f64, stop: f64, target: Option<f64>, fees: FeeRates) -> BookOrder {
        let (id, symbol, strategy, family) = ("o1".into(), "ETHUSDT".into(), "tsmom".into(), "trend".into());
        BookOrder { id, symbol, strategy, family, side, entry_type, price, qty: 1.0, stop, target, risk_usd: 5.0, fees, submitted_ts: 0, ttl_bars: 2, bars_seen: 0, peg_slip_bps: None }
    }

    #[test]
    fn pegged_limit_fills_at_the_open_plus_slip_as_maker() {
        for (side, sign) in [(Side::Long, 1.0), (Side::Short, -1.0)] {
            let mut b = PaperBook::new("t", 1000.0);
            let stop = 100.0 - sign * 5.0;
            let mut o = order(EntryType::Limit, side, 100.0, stop, Some(100.0 + sign * 10.0), CRYPTO);
            o.peg_slip_bps = Some(1.3);
            b.submit(o);
            // the bar never trades at open + 1.3 bp in the trade direction: still a fill (see on_bar)
            b.on_bar("ETHUSDT", 0, 100.0, 100.0 + if sign > 0.0 { 0.001 } else { 1.0 }, 100.0 - if sign > 0.0 { 1.0 } else { 0.001 }, 100.0);
            let f = &b.fills[0];
            assert!((f.price - 100.0 * (1.0 + sign * 1.3e-4)).abs() < 1e-9, "{side:?} {}", f.price);
            assert_eq!(f.liquidity, "maker");
            assert!((f.fee - f.price * 2.0e-4).abs() < 1e-12, "maker fee from the order's fee rates");
            assert_eq!(b.positions.len(), 1);
        }
        // a gap through the stop before the fill cancels, as for a market order
        let mut b = PaperBook::new("t", 1000.0);
        let mut o = order(EntryType::Limit, Side::Long, 100.0, 95.0, Some(110.0), CRYPTO);
        o.peg_slip_bps = Some(1.3);
        b.submit(o);
        b.on_bar("ETHUSDT", 0, 94.0, 96.0, 93.0, 95.5);
        assert!(b.positions.is_empty() && b.cancels[0].reason == "stop_through_before_fill");
        // an old state file without the field still loads (serde default)
        let mut v = serde_json::to_value(order(EntryType::Limit, Side::Long, 100.0, 95.0, None, CRYPTO)).unwrap();
        v.as_object_mut().unwrap().remove("peg_slip_bps");
        assert_eq!(serde_json::from_value::<BookOrder>(v).unwrap().peg_slip_bps, None);
    }

    #[test]
    fn market_fills_at_next_open_with_slippage_and_taker_fee() {
        let mut b = PaperBook::new("t", 1000.0);
        b.submit(order(EntryType::Market, Side::Long, 100.0, 95.0, Some(110.0), CRYPTO));
        b.on_bar("ETHUSDT", 0, 100.0, 101.0, 99.0, 100.5);
        let f = &b.fills[0];
        assert!((f.price - 100.02).abs() < 1e-9, "open + 2 bp");
        assert!((f.fee - 100.02 * 5.5e-4).abs() < 1e-12);
        assert_eq!(f.liquidity, "taker");
        assert_eq!(b.positions.len(), 1);
    }

    #[test]
    fn market_entry_cancelled_when_open_is_through_the_stop() {
        let mut b = PaperBook::new("t", 1000.0);
        b.submit(order(EntryType::Market, Side::Long, 100.0, 95.0, Some(110.0), CRYPTO));
        b.on_bar("ETHUSDT", 0, 94.0, 96.0, 93.0, 95.5);
        assert!(b.positions.is_empty() && b.fills.is_empty());
        assert_eq!(b.cancels[0].reason, "stop_through_before_fill");
    }

    #[test]
    fn limit_needs_trade_through_and_pays_maker() {
        let mut b = PaperBook::new("t", 1000.0);
        b.submit(order(EntryType::Limit, Side::Long, 99.0, 95.0, Some(110.0), CRYPTO));
        b.on_bar("ETHUSDT", 0, 100.0, 101.0, 99.0, 100.0); // touch only
        assert!(b.positions.is_empty());
        b.on_bar("ETHUSDT", 1, 100.0, 101.0, 98.9, 100.0); // through
        let f = &b.fills[0];
        assert_eq!((f.price, f.liquidity.as_str()), (99.0, "maker"));
        assert!((f.fee - 99.0 * 2e-4).abs() < 1e-12);
    }

    #[test]
    fn limit_expires_after_ttl() {
        let mut b = PaperBook::new("t", 1000.0);
        b.submit(order(EntryType::Limit, Side::Long, 99.0, 95.0, None, CRYPTO));
        b.on_bar("ETHUSDT", 0, 100.0, 101.0, 99.5, 100.0);
        b.on_bar("ETHUSDT", 1, 100.0, 101.0, 99.5, 100.0);
        assert!(b.orders.is_empty());
        assert_eq!(b.cancels[0].reason, "expired");
    }

    #[test]
    fn marketable_limit_is_taker_capped_at_limit() {
        let mut b = PaperBook::new("t", 1000.0);
        b.submit(order(EntryType::Limit, Side::Short, 100.0, 105.0, None, CRYPTO));
        b.on_bar("ETHUSDT", 0, 100.01, 100.5, 99.0, 99.5);
        let f = &b.fills[0];
        assert_eq!(f.liquidity, "taker");
        assert_eq!(f.price, 100.0, "open - 2 bp would be worse than the limit: capped");
    }

    #[test]
    fn limit_filled_in_bar_sees_stop_but_not_target() {
        let mut b = PaperBook::new("t", 1000.0);
        b.submit(order(EntryType::Limit, Side::Long, 99.0, 95.0, Some(101.0), ZERO));
        b.on_bar("ETHUSDT", 0, 100.0, 102.0, 98.0, 100.0);
        assert_eq!(b.positions.len(), 1, "target in the fill bar is ignored");
        let mut b = PaperBook::new("t", 1000.0);
        b.submit(order(EntryType::Limit, Side::Long, 99.0, 95.0, Some(101.0), ZERO));
        b.on_bar("ETHUSDT", 0, 100.0, 102.0, 94.0, 100.0);
        assert_eq!(b.trades[0].exit_reason, "stop");
        assert_eq!(b.trades[0].exit_price, 95.0, "intrabar fill: stop price, not the open");
    }

    #[test]
    fn resting_stop_first_and_gap_fills_at_open() {
        let mut b = PaperBook::new("t", 1000.0);
        b.submit(order(EntryType::Market, Side::Long, 100.0, 95.0, Some(110.0), ZERO));
        b.on_bar("ETHUSDT", 0, 100.0, 100.0, 100.0, 100.0);
        b.on_bar("ETHUSDT", 1, 100.0, 111.0, 94.0, 105.0);
        assert_eq!(b.trades[0].exit_reason, "stop");
        assert!((b.trades[0].r_multiple + 1.0).abs() < 1e-9, "1 unit x 5 = risk 5");
        let mut b = PaperBook::new("t", 1000.0);
        b.submit(order(EntryType::Market, Side::Short, 100.0, 105.0, Some(90.0), ZERO));
        b.on_bar("ETHUSDT", 0, 100.0, 100.0, 100.0, 100.0);
        b.on_bar("ETHUSDT", 1, 107.0, 108.0, 106.0, 107.0);
        assert_eq!(b.trades[0].exit_price, 107.0);
        assert!(b.trades[0].r_multiple < -1.0);
    }

    #[test]
    fn target_fills_at_target_with_exit_costs() {
        let mut b = PaperBook::new("t", 1000.0);
        b.submit(order(EntryType::Market, Side::Long, 100.0, 95.0, Some(110.0), CRYPTO));
        b.on_bar("ETHUSDT", 0, 100.0, 100.0, 100.0, 100.0);
        b.on_bar("ETHUSDT", 1, 112.0, 113.0, 111.0, 112.0); // gapped above: still the target price
        let t = &b.trades[0];
        assert_eq!(t.exit_reason, "target");
        assert!((t.exit_price - 110.0 * (1.0 - 2e-4)).abs() < 1e-9);
        let expect = (t.exit_price - 100.02) - 100.02 * 5.5e-4 - t.exit_price * 5.5e-4;
        assert!((t.pnl - expect).abs() < 1e-9);
        assert!((b.cash - (1000.0 + t.pnl)).abs() < 1e-9);
        assert!((b.marked_equity() - b.cash).abs() < 1e-12);
    }

    #[test]
    fn exit_signal_closes_at_next_open() {
        let mut b = PaperBook::new("t", 1000.0);
        b.submit(order(EntryType::Market, Side::Long, 100.0, 95.0, None, ZERO));
        b.on_bar("ETHUSDT", 0, 100.0, 100.0, 100.0, 101.0);
        assert!((b.marked_equity() - 1001.0).abs() < 1e-9);
        assert!(b.request_exit("ETHUSDT", "tsmom", 1));
        assert!(!b.request_exit("BTCUSDT", "tsmom", 1));
        b.on_bar("ETHUSDT", 1, 102.0, 103.0, 101.0, 102.5);
        assert_eq!(b.trades[0].exit_reason, "exit_signal");
        assert_eq!(b.trades[0].exit_price, 102.0);
    }

    #[test]
    fn kill_cancels_and_flattens() {
        let mut b = PaperBook::new("t", 1000.0);
        b.submit(order(EntryType::Market, Side::Long, 100.0, 95.0, None, CRYPTO));
        b.on_bar("ETHUSDT", 0, 100.0, 100.0, 100.0, 99.0);
        let mut o = order(EntryType::Limit, Side::Long, 50.0, 45.0, None, CRYPTO);
        o.id = "o2".into();
        o.symbol = "BTCUSDT".into();
        b.submit(o);
        b.kill(5);
        assert!(b.orders.is_empty() && b.positions.is_empty());
        assert_eq!(b.cancels[0].reason, "kill");
        assert_eq!(b.trades[0].exit_reason, "kill");
        assert!((b.trades[0].exit_price - 99.0 * (1.0 - 2e-4)).abs() < 1e-9);
        let dir = std::env::temp_dir().join(format!("atx-kill-{}", std::process::id()));
        assert!(!kill_switch_active(&dir));
        std::fs::write(&dir, "").unwrap();
        assert!(kill_switch_active(&dir));
        std::fs::remove_file(&dir).unwrap();
    }

    #[test]
    fn book_state_round_trips_through_json() {
        let mut b = PaperBook::new("t", 1000.0);
        b.submit(order(EntryType::Limit, Side::Long, 99.0, 95.0, Some(110.0), CRYPTO));
        b.on_bar("ETHUSDT", 0, 100.0, 101.0, 99.5, 100.0);
        let s = serde_json::to_string(&b).unwrap();
        let c: PaperBook = serde_json::from_str(&s).unwrap();
        assert_eq!(c.orders.len(), 1);
        assert_eq!(c.orders[0].bars_seen, 1);
        assert_eq!(c.book_entries().len(), 1);
        assert!(c.book_entries()[0].pending);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use at_core::types::TradeProposal;

    fn order(entry_type: EntryType) -> SizedOrder {
        SizedOrder {
            proposal: TradeProposal { id: "t".into(), symbol: "BTCUSDT".into(), side: Side::Long, entry_type, entry: 100.0, stop: 95.0, take_profit: 110.0, p_tp: 0.5, cluster: "L1".into(), channel: "C-MAIN".into() },
            entry: 100.0,
            stop: 95.0,
            take_profit: 110.0,
            qty: 10.0,
            notional: 1000.0,
            risk_amount: 50.0,
            reward_risk: 2.0,
            cost_r: 0.0,
            net_expected_r: 0.4,
        }
    }

    fn costs() -> Costs {
        Costs { taker_bps: 0.0, maker_bps: 0.0, slippage_bps: 0.0 }
    }

    #[tokio::test]
    async fn ambiguous_bar_assumes_stop_first() {
        let mut b = PaperBroker::new("paper", 10_000.0, costs());
        b.on_bar("BTCUSDT", 0, 100.0, 100.0, 100.0, 100.0);
        b.submit(&order(EntryType::Market), "x").await.unwrap();
        b.on_bar("BTCUSDT", 1, 100.0, 111.0, 94.0, 105.0);
        let t = &b.closed_trades()[0];
        assert_eq!(t.exit_reason, "stop");
        assert!((t.r_multiple + 1.0).abs() < 1e-9);
    }

    #[tokio::test]
    async fn gap_through_stop_fills_at_open() {
        let mut b = PaperBroker::new("paper", 10_000.0, costs());
        b.on_bar("BTCUSDT", 0, 100.0, 100.0, 100.0, 100.0);
        b.submit(&order(EntryType::Market), "x").await.unwrap();
        b.on_bar("BTCUSDT", 1, 90.0, 92.0, 89.0, 91.0);
        let t = &b.closed_trades()[0];
        assert_eq!(t.exit_price, 90.0);
        assert!(t.r_multiple < -1.0, "gaps can lose more than 1R");
    }

    #[tokio::test]
    async fn limit_fills_then_hits_target() {
        let mut b = PaperBroker::new("paper", 10_000.0, costs());
        b.submit(&order(EntryType::Limit), "x").await.unwrap();
        b.on_bar("BTCUSDT", 0, 102.0, 103.0, 99.5, 101.0);
        assert_eq!(b.positions().await.unwrap().len(), 1);
        b.on_bar("BTCUSDT", 1, 101.0, 112.0, 100.5, 111.0);
        assert_eq!(b.closed_trades()[0].exit_reason, "take_profit");
        assert!((b.equity().await.unwrap() - 10_100.0).abs() < 1e-9);
    }
}
