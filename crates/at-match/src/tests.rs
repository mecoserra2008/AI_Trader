//! Unit tests: one or more per rule in docs/MATCHING_ENGINE.md.

use at_core::types::InstrumentRules;
use at_risk::FeeRates;

use crate::*;

fn rules() -> InstrumentRules {
    InstrumentRules { symbol: "BTCUSDT".into(), tick_size: 0.1, qty_step: 0.001, min_qty: 0.001, max_qty: 100.0, max_mkt_qty: 10.0, min_notional: 5.0, max_leverage: 100.0 }
}

fn fees() -> FeeRates {
    FeeRates { maker_bps: 2.0, taker_bps: 5.5, slippage_bps: 0.0 }
}

fn book() -> OrderBook {
    OrderBook::new(&rules(), fees()).unwrap()
}

fn lim(id: u64, owner: u32, side: Side, tif: Tif, price: f64, qty: f64) -> NewOrder {
    NewOrder::limit(id as i64, id, owner, side, tif, price, qty)
}

fn trades(ev: &[Event]) -> Vec<&Trade> {
    ev.iter().filter_map(|e| if let Event::Trade(t) = e { Some(t) } else { None }).collect()
}

fn rejects(ev: &[Event]) -> Vec<Reason> {
    ev.iter().filter_map(|e| if let Event::Reject(r) = e { Some(r.reason) } else { None }).collect()
}

fn update(ev: &[Event], id: u64) -> &OrderUpdate {
    ev.iter().rev().find_map(|e| if let Event::Order(u) = e { (u.id == id).then_some(u) } else { None }).expect("order update")
}

fn ok(b: &OrderBook) {
    b.check_invariants().unwrap();
}

#[test]
fn limit_orders_rest_and_define_the_touch() {
    let mut b = book();
    let ev = b.submit(&lim(1, 1, Side::Buy, Tif::GTC, 100.0, 1.0));
    assert_eq!(update(&ev, 1).status, Status::New);
    b.submit(&lim(2, 1, Side::Sell, Tif::GTC, 100.5, 2.0));
    b.submit(&lim(3, 1, Side::Buy, Tif::GTC, 99.9, 0.5));
    assert_eq!(b.best_bid(), Some((1000, 1000)));
    assert_eq!(b.best_ask(), Some((1005, 2000)));
    assert_eq!(b.snapshot(10).bids, vec![(1000, 1000), (999, 500)]);
    ok(&b);
}

#[test]
fn price_then_time_priority() {
    let mut b = book();
    b.submit(&lim(1, 1, Side::Sell, Tif::GTC, 101.0, 1.0));
    b.submit(&lim(2, 2, Side::Sell, Tif::GTC, 100.0, 1.0));
    b.submit(&lim(3, 3, Side::Sell, Tif::GTC, 100.0, 1.0));
    let ev = b.submit(&lim(4, 4, Side::Buy, Tif::GTC, 101.0, 2.5));
    let t = trades(&ev);
    assert_eq!(t.iter().map(|t| (t.maker_id, t.price, t.qty)).collect::<Vec<_>>(), vec![(2, 1000, 1000), (3, 1000, 1000), (1, 1010, 500)]);
    assert!(t.iter().all(|t| t.taker_side == Side::Buy && t.taker_id == 4));
    assert_eq!(update(&ev, 4).status, Status::Filled);
    assert_eq!(b.order(1).unwrap().leaves, 500);
    ok(&b);
}

#[test]
fn partial_fill_keeps_queue_position() {
    let mut b = book();
    b.submit(&lim(1, 1, Side::Sell, Tif::GTC, 100.0, 2.0));
    b.submit(&lim(2, 2, Side::Sell, Tif::GTC, 100.0, 2.0));
    let ev = b.submit(&lim(3, 3, Side::Buy, Tif::IOC, 100.0, 0.5));
    assert_eq!(update(&ev, 1).status, Status::PartiallyFilled);
    assert_eq!(b.queue(Side::Sell, 1000), vec![1, 2]);
    assert_eq!(b.order(2).unwrap().queue_ahead, 1500);
    let ev = b.submit(&lim(4, 3, Side::Buy, Tif::IOC, 100.0, 1.6));
    assert_eq!(trades(&ev).iter().map(|t| (t.maker_id, t.qty)).collect::<Vec<_>>(), vec![(1, 1500), (2, 100)]);
    ok(&b);
}

#[test]
fn gtc_taker_rests_its_remainder_at_its_limit() {
    let mut b = book();
    b.submit(&lim(1, 1, Side::Sell, Tif::GTC, 100.0, 1.0));
    let ev = b.submit(&lim(2, 2, Side::Buy, Tif::GTC, 100.2, 3.0));
    let u = update(&ev, 2);
    assert_eq!((u.status, u.cum, u.leaves, u.price), (Status::PartiallyFilled, 1000, 2000, 1002));
    assert_eq!(trades(&ev)[0].price, 1000, "trades at the maker's price");
    assert_eq!(b.best_bid(), Some((1002, 2000)));
    assert_eq!(b.best_ask(), None);
    ok(&b);
}

#[test]
fn ioc_fills_what_crosses_and_cancels_the_rest() {
    let mut b = book();
    b.submit(&lim(1, 1, Side::Sell, Tif::GTC, 100.0, 1.0));
    b.submit(&lim(2, 1, Side::Sell, Tif::GTC, 100.5, 1.0));
    let ev = b.submit(&lim(3, 2, Side::Buy, Tif::IOC, 100.2, 3.0));
    let u = update(&ev, 3);
    assert_eq!((u.status, u.cum, u.leaves, u.reason), (Status::Cancelled, 1000, 0, Some(Reason::IocRemainderCancelled)));
    assert_eq!(b.best_bid(), None, "IOC never rests");
    let ev = b.submit(&lim(4, 2, Side::Buy, Tif::IOC, 100.2, 1.0));
    assert_eq!(update(&ev, 4).reason, Some(Reason::NoImmediateQtyToFill));
    assert!(trades(&ev).is_empty());
    ok(&b);
}

#[test]
fn fok_is_all_or_nothing() {
    let mut b = book();
    b.submit(&lim(1, 1, Side::Sell, Tif::GTC, 100.0, 1.0));
    b.submit(&lim(2, 1, Side::Sell, Tif::GTC, 100.1, 1.0));
    let v = b.version();
    let ev = b.submit(&lim(3, 2, Side::Buy, Tif::FOK, 100.1, 2.5));
    assert_eq!(rejects(&ev), vec![Reason::CancelForNoFullFill]);
    assert_eq!(b.version(), v, "book untouched");
    let ev = b.submit(&lim(4, 2, Side::Buy, Tif::FOK, 100.1, 2.0));
    assert_eq!(update(&ev, 4).status, Status::Filled);
    assert_eq!(trades(&ev).len(), 2);
    ok(&b);
}

#[test]
fn fok_precheck_honours_reduce_only_caps() {
    let mut b = book();
    // Owner 1 gets long 1.0, then rests a plain sell and a reduce-only sell at the same price (plain first).
    b.submit(&lim(1, 2, Side::Sell, Tif::GTC, 100.0, 1.0));
    b.submit(&lim(2, 1, Side::Buy, Tif::IOC, 100.0, 1.0));
    assert_eq!(b.position(1), 1000);
    b.submit(&lim(3, 1, Side::Sell, Tif::GTC, 101.0, 1.0));
    b.submit(&lim(4, 1, Side::Sell, Tif::GTC, 101.0, 1.0).reduce_only());
    assert_eq!(b.snapshot(1).asks, vec![(1010, 2000)]);
    // After the plain sell fills, owner 1 is flat, so the reduce-only sell cannot fill: only 1.0 is available.
    let ev = b.submit(&lim(5, 3, Side::Buy, Tif::FOK, 101.0, 2.0));
    assert_eq!(rejects(&ev), vec![Reason::CancelForNoFullFill]);
    // A GTC buy takes the plain sell; the reduce-only sell is cancelled at fill time instead of opening a short.
    let ev = b.submit(&lim(6, 3, Side::Buy, Tif::GTC, 101.0, 2.0));
    assert_eq!(trades(&ev).len(), 1);
    assert_eq!(update(&ev, 4).reason, Some(Reason::CancelByReduceOnly));
    assert_eq!(b.position(1), 0);
    assert_eq!(b.best_bid(), Some((1010, 1000)));
    ok(&b);
}

#[test]
fn post_only_rejects_if_it_would_take() {
    let mut b = book();
    b.submit(&lim(1, 1, Side::Sell, Tif::GTC, 100.0, 1.0));
    let ev = b.submit(&lim(2, 2, Side::Buy, Tif::PostOnly, 100.0, 1.0));
    assert_eq!(rejects(&ev), vec![Reason::PostOnlyWillTakeLiquidity]);
    let ev = b.submit(&lim(3, 2, Side::Buy, Tif::PostOnly, 99.9, 1.0));
    assert_eq!(update(&ev, 3).status, Status::New);
    let ev = b.submit(&lim(4, 2, Side::Sell, Tif::PostOnly, 99.9, 1.0));
    assert_eq!(rejects(&ev), vec![Reason::PostOnlyWillTakeLiquidity]);
    assert_eq!(b.trade_count(), 0);
    ok(&b);
}

#[test]
fn market_orders_are_protected_iocs() {
    let mut b = OrderBook::with_config(&rules(), fees(), EngineConfig { market_band: Band::Percent(0.5), ..Default::default() }).unwrap();
    b.submit(&lim(1, 1, Side::Sell, Tif::GTC, 100.0, 1.0));
    b.submit(&lim(2, 1, Side::Sell, Tif::GTC, 100.5, 1.0));
    b.submit(&lim(3, 1, Side::Sell, Tif::GTC, 100.6, 1.0));
    // band: 100.0 * 1.005 = 100.5 -> takes 100.0 and 100.5, not 100.6
    let ev = b.submit(&NewOrder::market(4, 4, 2, Side::Buy, 3.0));
    let u = update(&ev, 4);
    assert_eq!((u.status, u.cum, u.price, u.reason), (Status::Cancelled, 2000, 1005, Some(Reason::IocRemainderCancelled)));
    assert_eq!(b.best_ask(), Some((1006, 1000)));
    // per-order tick band
    let mut o = NewOrder::market(5, 5, 2, Side::Buy, 1.0);
    o.band = Some(Band::Ticks(1));
    b.submit(&lim(6, 1, Side::Sell, Tif::GTC, 100.8, 1.0));
    b.submit(&o);
    assert_eq!(b.best_ask(), Some((1008, 1000)));
    // empty opposite side
    let ev = b.submit(&NewOrder::market(7, 7, 2, Side::Sell, 1.0));
    assert_eq!(update(&ev, 7).reason, Some(Reason::NoImmediateQtyToFill));
    // PostOnly market and max market qty
    let mut o = NewOrder::market(8, 8, 2, Side::Buy, 1.0);
    o.tif = Tif::PostOnly;
    assert_eq!(rejects(&b.submit(&o)), vec![Reason::MarketOrderCannotBePostOnly]);
    assert_eq!(rejects(&b.submit(&NewOrder::market(9, 9, 2, Side::Buy, 10.001))), vec![Reason::QtyAboveMax]);
    // market FOK flag is ignored (market = IOC)
    let mut o = NewOrder::market(10, 10, 2, Side::Buy, 5.0);
    o.tif = Tif::FOK;
    let ev = b.submit(&o);
    assert_eq!(update(&ev, 10).cum, 1000);
    ok(&b);
}

#[test]
fn instrument_filters_reject_with_names() {
    let mut b = book();
    let r = |b: &mut OrderBook, o: NewOrder| rejects(&b.submit(&o));
    assert_eq!(r(&mut b, lim(1, 1, Side::Buy, Tif::GTC, 100.05, 1.0)), vec![Reason::InvalidPriceScale]);
    assert_eq!(r(&mut b, lim(2, 1, Side::Buy, Tif::GTC, 100.0, 1.0005)), vec![Reason::InvalidQty]);
    assert_eq!(r(&mut b, lim(3, 1, Side::Buy, Tif::GTC, 100.0, 0.0)), vec![Reason::QtyCannotBeZero]);
    assert_eq!(r(&mut b, lim(4, 1, Side::Buy, Tif::GTC, 100.0, -1.0)), vec![Reason::InvalidQty]);
    assert_eq!(r(&mut b, lim(5, 1, Side::Buy, Tif::GTC, -1.0, 1.0)), vec![Reason::LimitOrderInvalidPrice]);
    assert_eq!(r(&mut b, lim(6, 1, Side::Buy, Tif::GTC, 100.0, 100.001)), vec![Reason::QtyAboveMax]);
    assert_eq!(r(&mut b, lim(7, 1, Side::Buy, Tif::GTC, 100.0, 0.049)), vec![Reason::NotionalBelowMin], "4.9 USDT < 5");
    assert!(r(&mut b, lim(8, 1, Side::Buy, Tif::GTC, 100.0, 0.05)).is_empty(), "5 USDT is allowed");
    assert_eq!(r(&mut b, lim(8, 1, Side::Buy, Tif::GTC, 100.0, 1.0)), vec![Reason::DuplicatedId]);
    let mut o = lim(9, 1, Side::Buy, Tif::GTC, 1.0, 1.0);
    o.price = None;
    assert_eq!(r(&mut b, o), vec![Reason::LimitOrderInvalidPrice]);
    let mut c = rules();
    c.min_qty = 0.01;
    let mut b2 = OrderBook::new(&c, fees()).unwrap();
    assert_eq!(rejects(&b2.submit(&lim(1, 1, Side::Buy, Tif::GTC, 1000.0, 0.009))), vec![Reason::QtyBelowMin]);
    ok(&b);
}

#[test]
fn cancel_removes_only_that_order() {
    let mut b = book();
    for id in 1..=3 {
        b.submit(&lim(id, 1, Side::Buy, Tif::GTC, 100.0, 1.0));
    }
    let ev = b.cancel(4, 2);
    assert_eq!((update(&ev, 2).status, update(&ev, 2).reason), (Status::Cancelled, Some(Reason::CancelByUser)));
    assert_eq!(b.queue(Side::Buy, 1000), vec![1, 3]);
    assert_eq!(rejects(&b.cancel(5, 2)), vec![Reason::OrderNotExist]);
    assert_eq!(rejects(&b.cancel(5, 99)), vec![Reason::OrderNotExist]);
    b.cancel(6, 1);
    b.cancel(7, 3);
    assert_eq!(b.n_levels(), (0, 0));
    ok(&b);
}

#[test]
fn amend_qty_decrease_keeps_priority_increase_loses_it() {
    let mut b = book();
    for id in 1..=3 {
        b.submit(&lim(id, 1, Side::Sell, Tif::GTC, 100.0, 1.0));
    }
    let ev = b.amend(4, 1, None, Some(0.4));
    assert_eq!(update(&ev, 1).reason, Some(Reason::Amended));
    assert_eq!(b.queue(Side::Sell, 1000), vec![1, 2, 3]);
    assert_eq!(b.best_ask(), Some((1000, 2400)));
    b.amend(5, 1, None, Some(0.9));
    assert_eq!(b.queue(Side::Sell, 1000), vec![2, 3, 1], "qty increase goes to the back");
    b.amend(6, 2, Some(100.1), None);
    assert_eq!(b.queue(Side::Sell, 1000), vec![3, 1]);
    assert_eq!(b.queue(Side::Sell, 1001), vec![2]);
    b.amend(7, 2, Some(100.0), None);
    assert_eq!(b.queue(Side::Sell, 1000), vec![3, 1, 2], "price change back still loses priority");
    ok(&b);
}

#[test]
fn amend_after_partial_fill_counts_total_qty() {
    let mut b = book();
    b.submit(&lim(1, 1, Side::Sell, Tif::GTC, 100.0, 1.0));
    b.submit(&lim(2, 2, Side::Sell, Tif::GTC, 100.0, 1.0));
    b.submit(&lim(3, 3, Side::Buy, Tif::IOC, 100.0, 0.6));
    assert_eq!(rejects(&b.amend(4, 1, None, Some(0.6))), vec![Reason::AmendQtyNotAboveFilled]);
    assert_eq!(rejects(&b.amend(4, 1, None, Some(1.0))), vec![Reason::OrderNotModified]);
    let ev = b.amend(4, 1, None, Some(0.8));
    let u = update(&ev, 1);
    assert_eq!((u.qty, u.cum, u.leaves, u.status), (800, 600, 200, Status::PartiallyFilled));
    assert_eq!(b.queue(Side::Sell, 1000), vec![1, 2]);
    assert_eq!(rejects(&b.amend(5, 9, None, Some(1.0))), vec![Reason::OrderNotExist]);
    assert_eq!(rejects(&b.amend(5, 1, Some(100.03), None)), vec![Reason::InvalidPriceScale]);
    assert_eq!(rejects(&b.amend(5, 1, None, Some(0.0))), vec![Reason::QtyCannotBeZero]);
    ok(&b);
}

#[test]
fn amend_that_crosses_trades_as_taker() {
    let mut b = book();
    b.submit(&lim(1, 1, Side::Sell, Tif::GTC, 100.0, 1.0));
    b.submit(&lim(2, 2, Side::Buy, Tif::GTC, 99.0, 2.0));
    let ev = b.amend(3, 2, Some(100.0), None);
    let t = trades(&ev);
    assert_eq!((t[0].taker_id, t[0].maker_id, t[0].taker_side, t[0].qty), (2, 1, Side::Buy, 1000));
    assert!((t[0].taker_fee - 100.0 * 5.5e-4).abs() < 1e-12);
    let u = update(&ev, 2);
    assert_eq!((u.status, u.cum, u.leaves), (Status::PartiallyFilled, 1000, 1000));
    assert_eq!(b.best_bid(), Some((1000, 1000)));
    // PostOnly amend that would cross is refused and leaves the order alone
    b.submit(&lim(4, 3, Side::Sell, Tif::PostOnly, 101.0, 1.0));
    assert_eq!(rejects(&b.amend(5, 4, Some(100.0), None)), vec![Reason::PostOnlyWillTakeLiquidity]);
    assert_eq!(b.order(4).unwrap().price, 1010);
    ok(&b);
}

#[test]
fn reduce_only_rules() {
    let mut b = book();
    assert_eq!(rejects(&b.submit(&lim(1, 1, Side::Sell, Tif::GTC, 100.0, 1.0).reduce_only())), vec![Reason::ReduceOnlyRuleNotSatisfied]);
    // owner 1 buys 1.0 from owner 9
    b.submit(&lim(2, 9, Side::Sell, Tif::GTC, 100.0, 1.0));
    b.submit(&lim(3, 1, Side::Buy, Tif::IOC, 100.0, 1.0));
    assert_eq!((b.position(1), b.position(9)), (1000, -1000));
    assert_eq!(rejects(&b.submit(&lim(4, 1, Side::Buy, Tif::GTC, 99.0, 0.5).reduce_only())), vec![Reason::ReduceOnlyRuleNotSatisfied], "a buy does not reduce a long");
    // qty above the position is cut to it
    let ev = b.submit(&lim(5, 1, Side::Sell, Tif::GTC, 102.0, 3.0).reduce_only());
    let u = update(&ev, 5);
    assert_eq!((u.leaves, u.reason), (1000, Some(Reason::ReduceOnlyQtyAdjusted)));
    // a nearer reduce-only order trims the furthest one
    let ev = b.submit(&lim(6, 1, Side::Sell, Tif::GTC, 101.0, 0.4).reduce_only());
    let u5 = update(&ev, 5);
    assert_eq!((u5.leaves, u5.reason), (600, Some(Reason::ReduceOnlyQtyAdjusted)));
    ok(&b);
    // min notional does not apply to reduce-only (assumption 5)
    let ev = b.submit(&lim(7, 1, Side::Sell, Tif::GTC, 100.5, 0.001).reduce_only());
    assert!(rejects(&ev).is_empty());
    assert_eq!(b.order(5).unwrap().leaves, 599);
    // position closed elsewhere: every reduce-only order is cancelled
    b.submit(&lim(8, 7, Side::Buy, Tif::GTC, 99.0, 1.0));
    let ev = b.submit(&NewOrder::market(9, 9, 1, Side::Sell, 1.0));
    assert_eq!(b.position(1), 0);
    for id in [5, 6, 7] {
        assert_eq!(update(&ev, id).reason, Some(Reason::CancelByReduceOnly));
        assert!(!b.contains(id));
    }
    ok(&b);
}

#[test]
fn reduce_only_market_closes_and_never_flips() {
    let mut b = book();
    b.submit(&lim(1, 9, Side::Buy, Tif::GTC, 100.0, 5.0));
    b.submit(&NewOrder::market(2, 2, 1, Side::Sell, 2.0));
    assert_eq!(b.position(1), -2000);
    b.submit(&lim(3, 8, Side::Sell, Tif::GTC, 100.1, 5.0));
    let ev = b.submit(&NewOrder::market(4, 4, 1, Side::Buy, 5.0).reduce_only());
    assert_eq!(b.position(1), 0);
    assert_eq!(update(&ev, 4).reason, Some(Reason::ReduceOnlyQtyAdjusted));
    assert_eq!(trades(&ev).iter().map(|t| t.qty).sum::<i64>(), 2000);
    ok(&b);
}

#[test]
fn fees_and_ledger() {
    let mut b = book();
    b.submit(&lim(1, 1, Side::Sell, Tif::GTC, 50000.0, 0.2));
    let ev = b.submit(&lim(2, 2, Side::Buy, Tif::GTC, 50000.0, 0.2));
    let t = trades(&ev)[0];
    assert!((t.taker_fee - 10000.0 * 5.5e-4).abs() < 1e-9);
    assert!((t.maker_fee - 10000.0 * 2.0e-4).abs() < 1e-9);
    let (s, bu) = (b.owner(1).unwrap(), b.owner(2).unwrap());
    assert_eq!((s.position, bu.position), (-200, 200));
    assert!((s.cash - 10000.0).abs() < 1e-9 && (bu.cash + 10000.0).abs() < 1e-9);
    assert_eq!((s.maker_lots, bu.taker_lots), (200, 200));
    ok(&b);
}

#[test]
fn self_match_is_allowed_and_nets_out() {
    let mut b = book();
    b.submit(&lim(1, 1, Side::Sell, Tif::GTC, 100.0, 1.0));
    let ev = b.submit(&lim(2, 1, Side::Buy, Tif::GTC, 100.0, 1.0));
    assert_eq!(trades(&ev).len(), 1);
    let o = b.owner(1).unwrap();
    assert_eq!(o.position, 0);
    assert!(o.cash.abs() < 1e-12 && o.fees > 0.0);
    ok(&b);
}

#[test]
fn engine_is_deterministic() {
    let flow = flow::random_flow(7, 20_000, &flow::FlowParams::default());
    let run = || {
        let r = InstrumentRules { min_notional: 0.0, ..rules() };
        let mut b = OrderBook::new(&r, fees()).unwrap();
        let mut all = vec![];
        let mut s = String::new();
        for e in &flow {
            s.clear();
            e.write_json(&mut s);
            let ev = match e {
                InputEvent::New { ts, id, owner, side, ord_type, tif, price, qty, reduce_only } => {
                    b.submit(&NewOrder { ts: *ts, id: *id, owner: *owner as u32, side: *side, ord_type: *ord_type, tif: *tif, price: *price, qty: *qty, reduce_only: *reduce_only, band: None })
                }
                InputEvent::Cancel { ts, id } => b.cancel(*ts, *id),
                InputEvent::Amend { ts, id, price, qty } => b.amend(*ts, *id, *price, *qty),
            };
            all.extend(ev);
        }
        b.check_invariants().unwrap();
        (all, b.snapshot(200))
    };
    let (a, sa) = run();
    let (b2, sb) = run();
    assert!(a.iter().any(|e| matches!(e, Event::Trade(_))));
    assert_eq!(a, b2);
    assert_eq!(sa, sb);
}

#[test]
fn publisher_snapshot_then_windowed_deltas() {
    let mut b = book();
    let mut p = Publisher::new("BTCUSDT", 2);
    let mut out = Vec::new();
    b.submit(&lim(1, 1, Side::Buy, Tif::GTC, 100.0, 1.0));
    b.submit(&lim(2, 1, Side::Buy, Tif::GTC, 99.9, 1.0));
    b.submit(&lim(3, 1, Side::Buy, Tif::GTC, 99.8, 1.0));
    b.submit(&lim(4, 1, Side::Sell, Tif::GTC, 100.1, 0.5));
    assert!(p.publish(&b, 100, 4, &mut out).unwrap());
    assert!(!p.publish(&b, 200, 4, &mut out).unwrap(), "nothing changed, nothing sent");
    b.cancel(5, 1); // 99.8 enters the 2-level window
    assert!(p.publish(&b, 300, 5, &mut out).unwrap());
    let lines: Vec<serde_json::Value> = String::from_utf8(out).unwrap().lines().map(|l| serde_json::from_str(l).unwrap()).collect();
    assert_eq!(lines[0]["type"], "snapshot");
    assert_eq!(lines[0]["topic"], "orderbook.2.BTCUSDT");
    assert_eq!(lines[0]["data"]["b"], serde_json::json!([["100.0", "1.000"], ["99.9", "1.000"]]));
    assert_eq!(lines[0]["data"]["a"], serde_json::json!([["100.1", "0.500"]]));
    assert_eq!(lines[0]["data"]["u"], 1);
    assert_eq!(lines[1]["type"], "delta");
    assert_eq!(lines[1]["data"]["b"], serde_json::json!([["100.0", "0"], ["99.8", "1.000"]]));
    assert_eq!(lines[1]["data"]["a"], serde_json::json!([]));
    assert_eq!(lines[1]["data"]["u"], 2);
    assert_eq!((lines[1]["ts"].as_i64(), lines[1]["cts"].as_i64()), (Some(300), Some(5)));
    assert_eq!(p.published().bids, b.snapshot(2).bids);
}

#[test]
fn sim_driver_bins_ids_and_rejects() {
    let r = rules();
    let mut sim = Sim::new(OrderBook::new(&r, fees()).unwrap(), SimConfig { symbol: "BTCUSDT".into(), snapshot_every_ms: 100, depth: 200 });
    let (mut t, mut bk, mut rj, mut od) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    let mut sinks = Sinks { trades: &mut t, book: &mut bk, rejects: &mut rj, orders: Some(&mut od) };
    let input = [
        r#"{"ts_ms": 1000, "kind": "new", "id": "a1", "side": "Sell", "type": "Limit", "tif": "GTC", "price": "100.0", "qty": "1", "owner": "mm"}"#,
        r#"{"ts_ms": 1050, "kind": "new", "id": 7, "side": "Buy", "type": "Market", "qty": 0.4, "owner": 3}"#,
        r#"{"ts_ms": 1060, "kind": "new", "id": "a1", "side": "Sell", "price": 101, "qty": 1}"#,
        r#"{"ts_ms": 1150, "kind": "cancel", "id": "zz"}"#,
        r#"{"ts_ms": 1100, "kind": "cancel", "id": "a1"}"#,
        r#"not json"#,
        r#"{"ts_ms": 1250, "kind": "amend", "id": "a1", "qty": 0.5}"#,
        r#"{"ts_ms": 1260, "kind": "cancel", "id": "a1"}"#,
        r#"{"ts_ms": 1270, "kind": "cancel", "id": "a1"}"#,
    ];
    for l in input {
        sim.process_line(l, &mut sinks).unwrap();
    }
    let st = sim.finish(&mut sinks).unwrap();
    let s = |v: &Vec<u8>| String::from_utf8(v.clone()).unwrap();
    let tl: Vec<serde_json::Value> = s(&t).lines().map(|l| serde_json::from_str(l).unwrap()).collect();
    assert_eq!(tl.len(), 1);
    assert_eq!(tl[0]["maker"], "a1");
    assert_eq!(tl[0]["taker"], 7);
    assert_eq!(tl[0]["maker_owner"], "mm");
    assert_eq!(tl[0]["taker_owner"], 3);
    assert_eq!(tl[0]["price"], 100.0);
    assert_eq!(tl[0]["size"], 0.4);
    assert_eq!(tl[0]["side"], "Buy");
    let rl: Vec<serde_json::Value> = s(&rj).lines().map(|l| serde_json::from_str(l).unwrap()).collect();
    let reasons: Vec<&str> = rl.iter().map(|r| r["reason"].as_str().unwrap()).collect();
    assert_eq!(reasons, vec!["EC_DuplicatedClOrdID", "EC_OrderNotExist", "BadInput", "BadInput", "EC_OrderNotExist"]);
    let bl: Vec<serde_json::Value> = s(&bk).lines().map(|l| serde_json::from_str(l).unwrap()).collect();
    assert_eq!(bl.iter().map(|m| m["ts"].as_i64().unwrap()).collect::<Vec<_>>(), vec![1100, 1300]);
    assert_eq!(bl[0]["data"]["a"], serde_json::json!([["100.0", "0.600"]]));
    assert_eq!(bl[1]["data"]["a"], serde_json::json!([["100.0", "0"]]));
    assert_eq!(st.trades, 1);
    assert_eq!(st.events.new, 3);
    assert!(s(&od).lines().count() >= 4);
}
