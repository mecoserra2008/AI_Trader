//! Property tests over random flows (deterministic seeds):
//! 1. book invariants after every operation (never crossed, level totals = queue sums, FIFO priority, index,
//!    positions net to zero, reduce-only orders never exceed the position);
//! 2. quantity conservation per order (filled + open + cancelled = order qty) and taker lots = maker lots;
//! 3. the engine's trades and final book equal those of a naive reference matcher (sorted scans, no linked lists).

use std::collections::HashMap;

use at_core::types::InstrumentRules;
use at_match::flow::{FlowParams, random_flow};
use at_match::*;
use at_risk::FeeRates;

fn rules() -> InstrumentRules {
    InstrumentRules { symbol: "BTCUSDT".into(), tick_size: 0.1, qty_step: 0.001, min_qty: 0.001, max_qty: 1000.0, max_mkt_qty: 100.0, min_notional: 0.0, max_leverage: 100.0 }
}

fn fees() -> FeeRates {
    FeeRates { maker_bps: 2.0, taker_bps: 5.5, slippage_bps: 0.0 }
}

fn apply(b: &mut OrderBook, e: &InputEvent, out: &mut Vec<Event>) {
    match e {
        InputEvent::New { ts, id, owner, side, ord_type, tif, price, qty, reduce_only } => b.submit_into(
            &NewOrder { ts: *ts, id: *id, owner: *owner as u32, side: *side, ord_type: *ord_type, tif: *tif, price: *price, qty: *qty, reduce_only: *reduce_only, band: None },
            out,
        ),
        InputEvent::Cancel { ts, id } => b.cancel_into(*ts, *id, out),
        InputEvent::Amend { ts, id, price, qty } => b.amend_into(*ts, *id, *price, *qty, out),
    }
}

#[test]
fn invariants_and_conservation_hold_on_random_flows() {
    for seed in 1..=12u64 {
        let flow = random_flow(seed, 6_000, &FlowParams { depth: 30, max_lots: 8, owners: 4, ..Default::default() });
        let mut b = OrderBook::new(&rules(), fees()).unwrap();
        let mut ev = Vec::new();
        let mut last: HashMap<u64, OrderUpdate> = HashMap::new();
        let mut filled: HashMap<u64, i64> = HashMap::new();
        let (mut taker, mut maker) = (0i64, 0i64);
        for (k, e) in flow.iter().enumerate() {
            ev.clear();
            apply(&mut b, e, &mut ev);
            if let Err(msg) = b.check_invariants() {
                panic!("seed {seed} event {k} {e:?}: {msg}");
            }
            for x in &ev {
                match x {
                    Event::Trade(t) => {
                        taker += t.qty;
                        maker += t.qty;
                        *filled.entry(t.taker_id).or_default() += t.qty;
                        *filled.entry(t.maker_id).or_default() += t.qty;
                        assert!(t.qty > 0);
                    }
                    Event::Order(u) => {
                        assert_eq!(u.qty, u.cum + u.leaves + if u.status == Status::Cancelled { u.qty - u.cum } else { 0 }, "{u:?}");
                        last.insert(u.id, u.clone());
                    }
                    Event::Reject(_) => {}
                }
            }
        }
        assert_eq!(taker, maker);
        let mut open = 0;
        for (id, u) in &last {
            assert_eq!(filled.get(id).copied().unwrap_or(0), u.cum, "seed {seed}: order {id} cum vs trades");
            match u.status {
                Status::New | Status::PartiallyFilled => {
                    open += u.leaves;
                    let v = b.order(*id).expect("open order is in the book");
                    assert_eq!((v.leaves, v.cum), (u.leaves, u.cum));
                }
                _ => assert!(!b.contains(*id), "closed order {id} still in the book"),
            }
        }
        let l2 = b.snapshot(usize::MAX);
        assert_eq!(open, l2.bids.iter().chain(&l2.asks).map(|x| x.1).sum::<i64>(), "seed {seed}: resting qty");
        assert!(b.trade_count() > 100, "seed {seed}: flow should trade");
    }
}

// ------------------------------------------------------------------ naive reference matcher

#[derive(Clone, Debug)]
struct ROrd {
    id: u64,
    side: Side,
    tif: Tif,
    price: i64,
    qty: i64,
    cum: i64,
    prio: u64,
}

#[derive(Default)]
struct Ref {
    book: Vec<ROrd>,
    prio: u64,
    trades: Vec<(u64, u64, i64, i64)>,
}

impl Ref {
    fn best(&self, side: Side) -> Option<i64> {
        let it = self.book.iter().filter(|o| o.side == side).map(|o| o.price);
        if side == Side::Buy { it.max() } else { it.min() }
    }

    fn crosses(&self, side: Side, px: i64) -> bool {
        match side {
            Side::Buy => self.best(Side::Sell).is_some_and(|a| px >= a),
            Side::Sell => self.best(Side::Buy).is_some_and(|b| px <= b),
        }
    }

    fn eligible(&self, side: Side, limit: i64) -> Vec<usize> {
        let mut idx: Vec<usize> = (0..self.book.len()).filter(|&i| self.book[i].side != side && if side == Side::Buy { self.book[i].price <= limit } else { self.book[i].price >= limit }).collect();
        idx.sort_by_key(|&i| (if side == Side::Buy { self.book[i].price } else { -self.book[i].price }, self.book[i].prio));
        idx
    }

    fn take(&mut self, id: u64, side: Side, limit: i64, mut want: i64) -> i64 {
        let mut filled = 0;
        while want > 0 {
            let Some(&i) = self.eligible(side, limit).first() else { break };
            let m = &mut self.book[i];
            let f = want.min(m.qty - m.cum);
            m.cum += f;
            self.trades.push((id, m.id, m.price, f));
            want -= f;
            filled += f;
            if m.cum == m.qty {
                self.book.remove(i);
            }
        }
        filled
    }

    fn rest(&mut self, o: ROrd) {
        self.prio += 1;
        self.book.push(ROrd { prio: self.prio, ..o });
    }

    fn submit(&mut self, g: &Grid, e: &InputEvent) {
        let InputEvent::New { id, side, ord_type, tif, price, qty, .. } = e else { unreachable!() };
        if self.book.iter().any(|o| o.id == *id) {
            return;
        }
        let lots = g.lots(*qty).unwrap();
        let limit = if *ord_type == OrdType::Market {
            let Some(b) = self.best(side.opposite()) else { return };
            match side {
                Side::Buy => ((b as f64) * 1.01 + 1e-9).floor() as i64,
                Side::Sell => ((b as f64) * 0.99 - 1e-9).ceil() as i64,
            }
        } else {
            g.ticks(price.unwrap()).unwrap()
        };
        if *tif == Tif::PostOnly && self.crosses(*side, limit) {
            return;
        }
        if *tif == Tif::FOK && *ord_type == OrdType::Limit {
            let avail: i64 = self.eligible(*side, limit).iter().map(|&i| self.book[i].qty - self.book[i].cum).sum();
            if avail < lots {
                return;
            }
        }
        let cum = self.take(*id, *side, limit, lots);
        if cum < lots && *ord_type == OrdType::Limit && matches!(tif, Tif::GTC | Tif::PostOnly) {
            self.rest(ROrd { id: *id, side: *side, tif: *tif, price: limit, qty: lots, cum, prio: 0 });
        }
    }

    fn cancel(&mut self, id: u64) {
        self.book.retain(|o| o.id != id);
    }

    fn amend(&mut self, g: &Grid, id: u64, price: Option<f64>, qty: Option<f64>) {
        let Some(i) = self.book.iter().position(|o| o.id == id) else { return };
        let o = self.book[i].clone();
        let px = price.map_or(o.price, |p| g.ticks(p).unwrap());
        let q = qty.map_or(o.qty, |q| g.lots(q).unwrap());
        if (px == o.price && q == o.qty) || q <= o.cum {
            return;
        }
        if o.tif == Tif::PostOnly && px != o.price && self.crosses(o.side, px) {
            return;
        }
        if px == o.price && q <= o.qty {
            self.book[i].qty = q;
            return;
        }
        self.book.remove(i);
        let mut cum = o.cum;
        if px != o.price && self.crosses(o.side, px) {
            cum += self.take(id, o.side, px, q - o.cum);
        }
        if cum < q {
            self.rest(ROrd { price: px, qty: q, cum, ..o });
        }
    }

    fn l2(&self) -> L2 {
        let mut m: std::collections::BTreeMap<(bool, i64), i64> = Default::default();
        for o in &self.book {
            *m.entry((o.side == Side::Buy, o.price)).or_default() += o.qty - o.cum;
        }
        L2 {
            bids: m.iter().rev().filter(|((b, _), _)| *b).map(|((_, p), q)| (*p, *q)).collect(),
            asks: m.iter().filter(|((b, _), _)| !*b).map(|((_, p), q)| (*p, *q)).collect(),
        }
    }
}

#[test]
fn engine_matches_a_naive_reference_matcher() {
    for seed in 100..106u64 {
        let flow = random_flow(seed, 4_000, &FlowParams { depth: 25, max_lots: 6, owners: 5, reduce_only: false, ..Default::default() });
        let mut b = OrderBook::new(&rules(), fees()).unwrap();
        let g = *b.grid();
        let mut r = Ref::default();
        let mut ev = Vec::new();
        let mut trades = Vec::new();
        for e in &flow {
            ev.clear();
            apply(&mut b, e, &mut ev);
            trades.extend(ev.iter().filter_map(|x| if let Event::Trade(t) = x { Some((t.taker_id, t.maker_id, t.price, t.qty)) } else { None }));
            match e {
                InputEvent::New { .. } => r.submit(&g, e),
                InputEvent::Cancel { id, .. } => r.cancel(*id),
                InputEvent::Amend { id, price, qty, .. } => r.amend(&g, *id, *price, *qty),
            }
        }
        assert!(trades.len() > 100);
        assert_eq!(trades, r.trades, "seed {seed}: trade sequences differ");
        assert_eq!(b.snapshot(usize::MAX), r.l2(), "seed {seed}: final books differ");
    }
}
