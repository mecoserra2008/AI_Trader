//! Throughput benchmark: `cargo run --release -p at-match --example bench [-- N_EVENTS]`.
//!
//! 1. engine only: pre-built orders through `submit_into` / `cancel_into` / `amend_into` (one core, no I/O);
//! 2. JSONL driver: the same flow as text lines, parsed and matched, outputs (trades, book at 100 ms, rejects)
//!    rendered into memory sinks.

use std::time::Instant;

use at_core::types::InstrumentRules;
use at_match::flow::{FlowParams, random_flow};
use at_match::*;
use at_risk::FeeRates;

enum Op {
    New(NewOrder),
    Cancel(i64, u64),
    Amend(i64, u64, Option<f64>, Option<f64>),
}

fn main() {
    let n: usize = std::env::args().nth(1).and_then(|s| s.parse().ok()).unwrap_or(1_000_000);
    let rules = InstrumentRules { symbol: "BTCUSDT".into(), tick_size: 0.1, qty_step: 0.001, min_qty: 0.001, max_qty: 1190.0, max_mkt_qty: 119.0, min_notional: 5.0, max_leverage: 100.0 };
    let fees = FeeRates { maker_bps: 2.0, taker_bps: 5.5, slippage_bps: 0.0 };
    let flow = random_flow(1, n, &FlowParams::default());
    let ops: Vec<Op> = flow
        .iter()
        .map(|e| match e {
            InputEvent::New { ts, id, owner, side, ord_type, tif, price, qty, reduce_only } => {
                Op::New(NewOrder { ts: *ts, id: *id, owner: *owner as u32, side: *side, ord_type: *ord_type, tif: *tif, price: *price, qty: *qty, reduce_only: *reduce_only, band: None })
            }
            InputEvent::Cancel { ts, id } => Op::Cancel(*ts, *id),
            InputEvent::Amend { ts, id, price, qty } => Op::Amend(*ts, *id, *price, *qty),
        })
        .collect();

    let mut book = OrderBook::new(&rules, fees).unwrap();
    let mut out = Vec::with_capacity(256);
    let (mut n_events, mut levels_sum, mut samples) = (0usize, 0usize, 0usize);
    let t0 = Instant::now();
    for (k, op) in ops.iter().enumerate() {
        out.clear();
        match op {
            Op::New(o) => book.submit_into(o, &mut out),
            Op::Cancel(ts, id) => book.cancel_into(*ts, *id, &mut out),
            Op::Amend(ts, id, p, q) => book.amend_into(*ts, *id, *p, *q, &mut out),
        }
        n_events += out.len();
        if k % 1024 == 0 {
            let (b, a) = book.n_levels();
            levels_sum += b + a;
            samples += 1;
        }
    }
    let el = t0.elapsed().as_secs_f64();
    book.check_invariants().unwrap();
    let (b, a) = book.n_levels();
    println!(
        "engine only: {n} order events in {el:.3} s = {:.2} M events/s ({:.0} ns/event); {} output events, {} trades; avg levels (both sides) {:.0}; final levels bid {b} ask {a}, resting orders {}",
        n as f64 / el / 1e6,
        el * 1e9 / n as f64,
        n_events,
        book.trade_count(),
        levels_sum as f64 / samples.max(1) as f64,
        book.n_orders()
    );

    drop(ops);
    let mut lines = String::new();
    for e in &flow {
        e.write_json(&mut lines);
        lines.push('\n');
    }
    let mut sim = Sim::new(OrderBook::new(&rules, fees).unwrap(), SimConfig { symbol: "BTCUSDT".into(), snapshot_every_ms: 100, depth: 200 });
    let (mut tw, mut bw, mut rw) = (Vec::with_capacity(1 << 26), Vec::with_capacity(1 << 26), Vec::with_capacity(1 << 24));
    let mut sinks = Sinks { trades: &mut tw, book: &mut bw, rejects: &mut rw, orders: None };
    let t0 = Instant::now();
    for l in lines.lines() {
        sim.process_line(l, &mut sinks).unwrap();
    }
    let st = sim.finish(&mut sinks).unwrap();
    let el = t0.elapsed().as_secs_f64();
    println!(
        "JSONL driver: {} lines in {el:.3} s = {:.2} M lines/s; {} trades, {} book messages ({:.1} MB), {} rejects",
        st.lines,
        st.lines as f64 / el / 1e6,
        st.trades,
        st.book_snapshots + st.book_deltas,
        bw.len() as f64 / 1e6,
        st.rejects.values().sum::<u64>()
    );
}
