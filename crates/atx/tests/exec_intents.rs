//! End-to-end: `atx exec-intents` on the sample intents (tests/data/exec_intents), replay mode, CSV bars.
//! Writes only under a temp directory (no journal, private kill-file path).

use std::path::{Path, PathBuf};
use std::process::Command;

use serde_json::Value;

fn data() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/data/exec_intents")
}

fn tmp(tag: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("atx-exec-intents-{tag}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).unwrap();
    d
}

fn atx(out: &Path, mode: &str, env: &str) -> std::process::Output {
    Command::new(env!("CARGO_BIN_EXE_atx"))
        .current_dir(env!("CARGO_MANIFEST_DIR"))
        .env("AI_TRADER_ENVIRONMENT", env)
        .args(["exec-intents", "--file"])
        .arg(data().join("intents.jsonl"))
        .args(["--mode", mode, "--equity", "1000", "--run", "sample", "--no-journal", "--bars-dir"])
        .arg(data().join("bars"))
        .arg("--out-dir")
        .arg(out)
        .arg("--kill-file")
        .arg(out.join("KILL"))
        .output()
        .expect("atx runs")
}

fn jsonl(p: &Path) -> Vec<Value> {
    std::fs::read_to_string(p).unwrap().lines().map(|l| serde_json::from_str(l).unwrap()).collect()
}

#[test]
fn replay_sample_intents_end_to_end() {
    let out = tmp("replay");
    let r = atx(&out, "replay", "demo");
    assert!(r.status.success(), "stderr: {}", String::from_utf8_lossy(&r.stderr));

    let d = jsonl(&out.join("decisions/sample.jsonl"));
    assert_eq!(d.len(), 7);
    let by_id = |id: &str| d.iter().find(|x| x["intent_id"] == id).unwrap().clone();
    let rej = |id: &str| by_id(id)["rejections"].as_array().unwrap().iter().map(|x| x.as_str().unwrap().to_string()).collect::<Vec<_>>();
    for id in ["A-eth-trend", "C-sol-mr", "D-nvda-carry", "E-nvda-exit"] {
        assert_eq!(by_id(id)["decision"], "accept", "{id}: {:?}", rej(id));
    }
    assert_eq!(rej("B-btc-trend"), vec!["max_positions_per_cluster"], "R-PD-1: one trend bet at a time");
    assert!(rej("F-btc-shock").contains(&"gate_blocked".to_string()));
    assert!(rej("G-btc-pos").contains(&"stop_beyond_max_sigma".to_string()));
    assert!(rej("G-btc-pos").contains(&"max_new_trades_per_day".to_string()));
    assert_eq!(by_id("D-nvda-carry")["fee_class"], "tradfi");
    assert!(by_id("D-nvda-carry")["risk_r"].as_f64().unwrap() <= 0.5, "gate reduce halves the risk");
    for x in &d {
        for k in ["intent_id", "decision", "qty", "notional_usd", "risk_usd", "risk_r", "rejections", "cluster", "portfolio"] {
            assert!(x.get(k).is_some(), "contract field {k}");
        }
        if x["decision"] == "accept" && x["action"] == "open" {
            assert!(x["risk_r"].as_f64().unwrap() <= 1.0 + 1e-9, "never more than the fixed risk");
        }
    }

    let paper = out.join("paper/sample");
    let trades = jsonl(&paper.join("trades.jsonl"));
    let reason = |id: &str| trades.iter().find(|t| t["id"] == id).map(|t| t["exit_reason"].as_str().unwrap().to_string());
    assert_eq!(reason("A-eth-trend").as_deref(), Some("target"));
    assert_eq!(reason("C-sol-mr").as_deref(), Some("stop"));
    assert_eq!(reason("D-nvda-carry").as_deref(), Some("exit_signal"));
    let c = trades.iter().find(|t| t["id"] == "C-sol-mr").unwrap();
    assert!((c["r_multiple"].as_f64().unwrap() + 1.0).abs() < 0.01, "a clean stop loses ~1R including costs");

    let fills = jsonl(&paper.join("fills.jsonl"));
    let liq = |id: &str| fills.iter().find(|f| f["id"] == id && f["kind"] == "entry").unwrap()["liquidity"].as_str().unwrap().to_string();
    assert_eq!(liq("C-sol-mr"), "maker");
    assert_eq!(liq("A-eth-trend"), "taker");

    let s: Value = serde_json::from_str(&std::fs::read_to_string(paper.join("summary.json")).unwrap()).unwrap();
    assert_eq!(s["paper_only"], true);
    assert_eq!((s["accepted"].as_u64(), s["trades"].as_u64(), s["open_positions"].as_u64()), (Some(4), Some(3), Some(0)));
    let pnl: f64 = trades.iter().map(|t| t["pnl"].as_f64().unwrap()).sum();
    assert!((s["end_equity_marked"].as_f64().unwrap() - (1000.0 + pnl)).abs() < 1e-6);
    std::fs::remove_dir_all(&out).ok();
}

#[test]
fn kill_file_blocks_every_entry() {
    let out = tmp("kill");
    std::fs::write(out.join("KILL"), "").unwrap();
    let r = atx(&out, "replay", "demo");
    assert!(r.status.success(), "stderr: {}", String::from_utf8_lossy(&r.stderr));
    let d = jsonl(&out.join("decisions/sample.jsonl"));
    for x in d.iter().filter(|x| x["side"] != "flat") {
        assert!(x["rejections"].as_array().unwrap().iter().any(|r| r == "kill_switch"), "{}", x["intent_id"]);
    }
    assert!(jsonl(&out.join("paper/sample/fills.jsonl")).is_empty());
    std::fs::remove_dir_all(&out).ok();
}

#[test]
fn live_mode_and_mainnet_are_refused() {
    let out = tmp("refuse");
    let r = atx(&out, "live", "demo");
    assert!(!r.status.success());
    assert!(String::from_utf8_lossy(&r.stderr).contains("refused"));
    let r = atx(&out, "replay", "mainnet");
    assert!(!r.status.success());
    assert!(String::from_utf8_lossy(&r.stderr).contains("mainnet"));
    assert!(!out.join("decisions").exists(), "nothing written when refused");
    std::fs::remove_dir_all(&out).ok();
}
