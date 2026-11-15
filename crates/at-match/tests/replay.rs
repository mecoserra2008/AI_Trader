//! Replay test: an ob200 snapshot plus a minute of deltas, replayed as add / cancel / amend events through the
//! JSONL driver, must leave the engine's book equal to the book the messages describe, after every message, with no
//! trades and no rejects.
//!
//! The synthetic stream always runs. A real Bybit stream runs too when `AT_MATCH_OB200` points at an unzipped
//! `<date>_<SYMBOL>_ob200.data` JSON-lines file (BTCUSDT grid: tick 0.1, step 0.001; override with
//! `AT_MATCH_TICK` / `AT_MATCH_STEP`); it replays the first snapshot and the following 60 s.

use std::collections::BTreeMap;

use at_core::types::InstrumentRules;
use at_match::flow::Rng;
use at_match::*;
use at_risk::FeeRates;
use serde_json::{Value, json};

fn engine(tick: f64, step: f64) -> OrderBook {
    let r = InstrumentRules { symbol: "BTCUSDT".into(), tick_size: tick, qty_step: step, min_qty: step, max_qty: 1e12, max_mkt_qty: 1e12, min_notional: 0.0, max_leverage: 100.0 };
    OrderBook::new(&r, FeeRates { maker_bps: 2.0, taker_bps: 5.5, slippage_bps: 0.0 }).unwrap()
}

/// Reference L2 from the messages themselves (what book.py does).
#[derive(Default)]
struct RefBook {
    b: BTreeMap<i64, i64>,
    a: BTreeMap<i64, i64>,
}

impl RefBook {
    fn apply(&mut self, g: &Grid, m: &Value) {
        if m["type"] == "snapshot" {
            self.b.clear();
            self.a.clear();
        }
        for (side, key) in [(&mut self.b, "b"), (&mut self.a, "a")] {
            for lv in m["data"][key].as_array().unwrap() {
                let p = g.ticks(lv[0].as_str().unwrap().parse().unwrap()).unwrap();
                let s: f64 = lv[1].as_str().unwrap().parse().unwrap();
                if s == 0.0 {
                    side.remove(&p);
                } else {
                    side.insert(p, g.lots(s).unwrap());
                }
            }
        }
    }

    fn l2(&self) -> L2 {
        L2 { bids: self.b.iter().rev().map(|(p, q)| (*p, *q)).collect(), asks: self.a.iter().map(|(p, q)| (*p, *q)).collect() }
    }
}

/// Replays `msgs` and checks the engine after every message. Returns (messages, events) replayed.
fn replay(msgs: &[(i64, Value)], tick: f64, step: f64) -> (usize, usize) {
    let mut sim = Sim::new(engine(tick, step), SimConfig { symbol: "BTCUSDT".into(), snapshot_every_ms: 0, depth: 200 });
    let g = *sim.book().grid();
    let mut rp = Ob200Replayer::new(g, 1, 1);
    let mut rf = RefBook::default();
    let (mut t, mut bk, mut rj) = (Vec::new(), Vec::new(), Vec::new());
    let mut sinks = Sinks { trades: &mut t, book: &mut bk, rejects: &mut rj, orders: None };
    let (mut evs, mut line) = (Vec::new(), String::new());
    let mut n_events = 0;
    for (k, (ts, m)) in msgs.iter().enumerate() {
        evs.clear();
        rp.apply(m, *ts, &mut evs).unwrap();
        rf.apply(&g, m);
        for e in &evs {
            line.clear();
            e.write_json(&mut line);
            sim.process_line(&line, &mut sinks).unwrap();
        }
        n_events += evs.len();
        assert_eq!(rp.book(), rf.l2(), "replayer state after message {k}");
        assert_eq!(sim.book().snapshot(usize::MAX), rf.l2(), "engine book after message {k}");
        sim.book().check_invariants().unwrap();
    }
    assert!(t.is_empty(), "replay must not trade: {}", String::from_utf8_lossy(&t[..t.len().min(500)]));
    assert!(rj.is_empty(), "replay must not reject: {}", String::from_utf8_lossy(&rj[..rj.len().min(500)]));
    (msgs.len(), n_events)
}

fn lv(g: &Grid, p: i64, q: i64) -> Value {
    json!([g.fmt_price(p), if q == 0 { "0".to_string() } else { g.fmt_qty(q) }])
}

/// A 200-level snapshot and 600 deltas (one minute at 100 ms): size changes, removals, insertions, and touch moves
/// in which a level changes side inside one message (removal and insertion at the same price).
fn synthetic_stream(seed: u64) -> Vec<(i64, Value)> {
    let g = Grid::new(0.1, 0.001).unwrap();
    let mut r = Rng::new(seed);
    let (mut bids, mut asks) = (BTreeMap::<i64, i64>::new(), BTreeMap::<i64, i64>::new());
    let mid = 750_000i64;
    for k in 0..200 {
        bids.insert(mid - 1 - k, 1 + r.below(5000) as i64);
        asks.insert(mid + k, 1 + r.below(5000) as i64);
    }
    let msg = |kind: &str, ts: i64, b: Vec<Value>, a: Vec<Value>, u: u64| json!({"topic": "orderbook.200.BTCUSDT", "type": kind, "ts": ts, "data": {"s": "BTCUSDT", "b": b, "a": a, "u": u, "seq": u}, "cts": ts});
    let mut out = vec![];
    let ts0 = 1_731_456_000_000i64;
    out.push((ts0, msg("snapshot", ts0, bids.iter().rev().map(|(p, q)| lv(&g, *p, *q)).collect(), asks.iter().map(|(p, q)| lv(&g, *p, *q)).collect(), 1)));
    for k in 1..=600u64 {
        let ts = ts0 + 100 * k as i64;
        let (mut db, mut da) = (BTreeMap::<i64, i64>::new(), BTreeMap::<i64, i64>::new());
        for _ in 0..1 + r.below(8) {
            let buy = r.below(2) == 0;
            let (side, other, d) = if buy { (&mut bids, &asks, &mut db) } else { (&mut asks, &bids, &mut da) };
            match r.below(10) {
                0..=4 if !side.is_empty() => {
                    // resize an existing level
                    let keys: Vec<i64> = side.keys().copied().collect();
                    let p = keys[r.below(keys.len() as u64) as usize];
                    let q = 1 + r.below(8000) as i64;
                    side.insert(p, q);
                    d.insert(p, q);
                }
                5..=6 if side.len() > 20 => {
                    let keys: Vec<i64> = side.keys().copied().collect();
                    let p = keys[r.below(keys.len() as u64) as usize];
                    side.remove(&p);
                    d.insert(p, 0);
                }
                _ => {
                    // insert a new level on the correct side of the opposite touch
                    let p = if buy { other.keys().next().copied().unwrap_or(mid) - 1 - r.below(40) as i64 } else { other.keys().next_back().copied().unwrap_or(mid) + 1 + r.below(40) as i64 };
                    let q = 1 + r.below(3000) as i64;
                    *side.entry(p).or_default() += q;
                    d.insert(p, side[&p]);
                }
            }
        }
        if r.below(6) == 0 && asks.len() > 1 {
            // the best ask is lifted and a bid appears at that price, in one message
            let (&p, _) = asks.iter().next().unwrap();
            asks.remove(&p);
            da.insert(p, 0);
            let q = 1 + r.below(2000) as i64;
            bids.insert(p, q);
            db.insert(p, q);
        }
        out.push((ts, msg("delta", ts, db.iter().rev().map(|(p, q)| lv(&g, *p, *q)).collect(), da.iter().map(|(p, q)| lv(&g, *p, *q)).collect(), k + 1)));
    }
    out
}

#[test]
fn synthetic_ob200_minute_replays_exactly() {
    for seed in [1, 2, 3] {
        let msgs = synthetic_stream(seed);
        let (m, e) = replay(&msgs, 0.1, 0.001);
        assert_eq!(m, 601);
        assert!(e > 2000, "seed {seed}: {e} events");
    }
}

#[test]
fn real_ob200_minute_replays_exactly_if_available() {
    let Ok(path) = std::env::var("AT_MATCH_OB200") else {
        eprintln!("AT_MATCH_OB200 not set: real-stream replay skipped");
        return;
    };
    let tick: f64 = std::env::var("AT_MATCH_TICK").ok().and_then(|s| s.parse().ok()).unwrap_or(0.1);
    let step: f64 = std::env::var("AT_MATCH_STEP").ok().and_then(|s| s.parse().ok()).unwrap_or(0.001);
    use std::io::BufRead;
    let f = std::io::BufReader::new(std::fs::File::open(&path).unwrap());
    let mut msgs = vec![];
    let mut t0 = None;
    for line in f.lines() {
        let line = line.unwrap();
        if line.trim().is_empty() {
            continue;
        }
        let m: Value = serde_json::from_str(&line).unwrap();
        let ts = m["ts"].as_i64().unwrap();
        if t0.is_none() {
            if m["type"] != "snapshot" {
                continue;
            }
            t0 = Some(ts);
        }
        if ts > t0.unwrap() + 60_000 {
            break;
        }
        msgs.push((ts, m));
    }
    assert!(!msgs.is_empty(), "no snapshot in {path}");
    let (m, e) = replay(&msgs, tick, step);
    eprintln!("real ob200 replay: {m} messages, {e} engine events, book equal after every message");
}
