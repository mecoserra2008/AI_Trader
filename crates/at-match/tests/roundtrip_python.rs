//! Round trip: `book.jsonl` written by the driver is read, unchanged, by the project's Python parser
//! (`src/ai_trader/sim/book.py`: `iter_messages`, `Book.apply`, `build_from_messages`), and the Python book equals
//! the engine's final top-200 window. Needs `uv` (skipped with a message when it is not installed).

use std::path::PathBuf;
use std::process::Command;

use at_core::types::InstrumentRules;
use at_match::flow::{FlowParams, random_flow};
use at_match::*;
use at_risk::FeeRates;

const CHECK: &str = r#"
import json, sys
from ai_trader.sim.book import Book, build_from_messages, iter_messages
path, expect_path = sys.argv[1], sys.argv[2]
exp = json.load(open(expect_path))
book = Book()
n = 0
for m in iter_messages(path):
    assert m["topic"] == "orderbook.200.BTCUSDT" and m["type"] in ("snapshot", "delta"), m
    assert set(m["data"]) == {"s", "b", "a", "u", "seq"}, m["data"].keys()
    assert book.apply(m["type"], m["data"]["b"], m["data"]["a"])
    n += 1
got_b = sorted(([p, s] for p, s in book.bids.items()), reverse=True)
got_a = sorted([p, s] for p, s in book.asks.items())
assert got_b == [[float(p), float(s)] for p, s in exp["bids"]], "bids differ"
assert got_a == [[float(p), float(s)] for p, s in exp["asks"]], "asks differ"
df, qa = build_from_messages(iter_messages(path))
assert qa["u_gaps"] == 0 and qa["ignored_before_snapshot"] == 0 and qa["snapshots"] == 1, qa
assert qa["crossed_bins"] == 0 and df.height > 0, qa
assert abs(df["bid1"][-1] - float(exp["bids"][0][0])) < 1e-9 and abs(df["ask1"][-1] - float(exp["asks"][0][0])) < 1e-9
print("OK", n, "messages", df.height, "bins", qa)
"#;

#[test]
fn book_jsonl_is_read_by_sim_book_py() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
    if Command::new("uv").arg("--version").output().is_err() {
        eprintln!("uv not found: Python round-trip skipped");
        return;
    }
    let dir = std::env::temp_dir().join(format!("at-match-roundtrip-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let input = dir.join("events.jsonl");
    let mut text = String::new();
    for e in random_flow(42, 30_000, &FlowParams::default()) {
        e.write_json(&mut text);
        text.push('\n');
    }
    std::fs::write(&input, text).unwrap();
    let rules = InstrumentRules { symbol: "BTCUSDT".into(), tick_size: 0.1, qty_step: 0.001, min_qty: 0.001, max_qty: 1190.0, max_mkt_qty: 119.0, min_notional: 5.0, max_leverage: 100.0 };
    let book = OrderBook::new(&rules, FeeRates { maker_bps: 2.0, taker_bps: 5.5, slippage_bps: 0.0 }).unwrap();
    let out = dir.join("out");
    let stats = run_files(book, SimConfig { symbol: "BTCUSDT".into(), snapshot_every_ms: 100, depth: 200 }, &input, &out, false).unwrap();
    assert!(stats.trades > 100 && stats.book_deltas > 100, "{stats:?}");

    // Expected final window: replay the same input through a fresh engine.
    let book = OrderBook::new(&rules, FeeRates { maker_bps: 2.0, taker_bps: 5.5, slippage_bps: 0.0 }).unwrap();
    let mut sim = Sim::new(book, SimConfig { symbol: "BTCUSDT".into(), snapshot_every_ms: 100, depth: 200 });
    let (mut a, mut b, mut c) = (std::io::sink(), std::io::sink(), std::io::sink());
    let mut sinks = Sinks { trades: &mut a, book: &mut b, rejects: &mut c, orders: None };
    for l in std::fs::read_to_string(&input).unwrap().lines() {
        sim.process_line(l, &mut sinks).unwrap();
    }
    let g = *sim.book().grid();
    let snap = sim.book().snapshot(200);
    let fmt = |v: &[(i64, i64)]| v.iter().map(|(p, q)| vec![g.fmt_price(*p), g.fmt_qty(*q)]).collect::<Vec<_>>();
    let expect = serde_json::json!({"bids": fmt(&snap.bids), "asks": fmt(&snap.asks)});
    let expect_path = dir.join("expect.json");
    std::fs::write(&expect_path, expect.to_string()).unwrap();

    let res = Command::new("uv")
        .args(["run", "--no-sync", "python", "-c", CHECK])
        .arg(out.join("book.jsonl"))
        .arg(&expect_path)
        .current_dir(&root)
        .env("PYTHONIOENCODING", "utf-8")
        .output()
        .expect("run uv");
    let so = String::from_utf8_lossy(&res.stdout);
    let se = String::from_utf8_lossy(&res.stderr);
    assert!(res.status.success() && so.contains("OK"), "python check failed:\nstdout: {so}\nstderr: {se}");
    eprintln!("{so}");
    let _ = std::fs::remove_dir_all(&dir);
}
