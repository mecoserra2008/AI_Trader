//! `atx match-sim`: replay an order-event JSONL file through the at-match engine (Bybit semantics, research only:
//! a simulator with no network or exchange access). Contract: docs/MATCHING_ENGINE.md.
//!
//! Instrument rules come from `--rules <json>` (one InstrumentRules object or a list), else from the lake's
//! instruments table (point in time at the first event's ts when it is a real epoch-ms time, else the latest).
//! Fees come from config/fees.yaml (or `--fees`), class from `--fee-class`, else the lake symbol type.

use std::io::BufRead;
use std::path::PathBuf;

use anyhow::{Context, bail};
use at_core::time::now_ms;
use at_core::types::InstrumentRules;
use at_match::{Band, EngineConfig, OrderBook, SimConfig, run_files};
use at_risk::FeeSchedule;

use crate::cmd::Ctx;
use crate::lake_read::read_instruments;

#[derive(clap::Args)]
pub struct MatchSimArgs {
    /// Input events, one JSON object per line: {"ts_ms","kind":"new|cancel|amend","id","side","type","tif","price","qty","reduce_only","owner"}.
    #[arg(long)]
    events: PathBuf,
    #[arg(long, default_value = "BTCUSDT")]
    symbol: String,
    /// Output directory: trades.jsonl, book.jsonl (orderbook.<depth> shape), rejects.jsonl, stats.json [, orders.jsonl].
    #[arg(long)]
    out: PathBuf,
    /// Book publication cadence in ms (Bybit ob200: 100); 0 = a message after every input event.
    #[arg(long, default_value_t = 100)]
    snapshot_every_ms: i64,
    /// Levels per side in book.jsonl.
    #[arg(long, default_value_t = 200)]
    depth: usize,
    /// Instrument rules JSON instead of the lake.
    #[arg(long)]
    rules: Option<PathBuf>,
    /// Fee schedule instead of config/fees.yaml.
    #[arg(long)]
    fees: Option<PathBuf>,
    /// Fee class (crypto | tradfi) instead of the lake symbol type.
    #[arg(long)]
    fee_class: Option<String>,
    /// Market-order protection band in percent of the opposite best (default 1).
    #[arg(long, conflicts_with = "market_band_ticks")]
    market_band_pct: Option<f64>,
    /// Market-order protection band in ticks from the opposite best.
    #[arg(long)]
    market_band_ticks: Option<i64>,
    /// Also write orders.jsonl (every order update).
    #[arg(long)]
    orders: bool,
}

fn first_ts(path: &PathBuf) -> anyhow::Result<Option<i64>> {
    let f = std::io::BufReader::new(std::fs::File::open(path).with_context(|| path.display().to_string())?);
    for line in f.lines() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        return Ok(serde_json::from_str::<serde_json::Value>(&line).ok().and_then(|v| v["ts_ms"].as_i64()));
    }
    Ok(None)
}

pub fn run(ctx: &Ctx, a: MatchSimArgs) -> anyhow::Result<()> {
    let symbol = a.symbol.to_uppercase();
    let (rules, symbol_type): (InstrumentRules, Option<String>) = match &a.rules {
        Some(p) => {
            let v: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(p).with_context(|| p.display().to_string())?)?;
            let list: Vec<InstrumentRules> = if v.is_array() { serde_json::from_value(v)? } else { vec![serde_json::from_value(v)?] };
            let r = list.into_iter().find(|r| r.symbol.eq_ignore_ascii_case(&symbol)).with_context(|| format!("{}: no rules for {symbol}", p.display()))?;
            (r, None)
        }
        None => {
            // Point in time at the first event when it carries a real epoch-ms timestamp (>= 2001-09-09).
            let as_of = first_ts(&a.events)?.filter(|t| *t >= 1_000_000_000_000).unwrap_or_else(now_ms);
            let (mut r, mut types) = read_instruments(ctx.lake.root(), "bybit", ctx.category(), as_of)?;
            if !r.contains_key(&symbol) {
                tracing::warn!(%symbol, as_of, "no instrument rules known at the first event; using the latest snapshot");
                (r, types) = read_instruments(ctx.lake.root(), "bybit", ctx.category(), now_ms())?;
            }
            match r.remove(&symbol) {
                Some(x) => (x, types.remove(&symbol)),
                None => bail!("no instrument rules for {symbol} in the lake: run `atx instruments` or pass --rules"),
            }
        }
    };
    let fees_path = a.fees.clone().unwrap_or_else(|| ctx.s.root.join("config/fees.yaml"));
    let schedule = FeeSchedule::load(&fees_path)?;
    let hint = a.fee_class.clone().or_else(|| symbol_type.map(|t| if t.is_empty() { "crypto".into() } else { "tradfi".into() }));
    let fees = schedule.rates(&symbol, hint.as_deref());
    let mut cfg = EngineConfig::default();
    if let Some(p) = a.market_band_pct {
        cfg.market_band = Band::Percent(p);
    }
    if let Some(n) = a.market_band_ticks {
        cfg.market_band = Band::Ticks(n);
    }
    let book = OrderBook::with_config(&rules, fees, cfg).map_err(|e| anyhow::anyhow!("{symbol}: {e}"))?;
    tracing::info!(%symbol, tick = rules.tick_size, step = rules.qty_step, min_qty = rules.min_qty, min_notional = rules.min_notional, maker_bps = fees.maker_bps, taker_bps = fees.taker_bps, "match-sim");
    let st = run_files(book, SimConfig { symbol: symbol.clone(), snapshot_every_ms: a.snapshot_every_ms, depth: a.depth }, &a.events, &a.out, a.orders)?;
    let perf = st.perf.as_ref().map_or(0.0, |p| p.lines_per_s);
    println!(
        "{symbol}: {} lines ({} new, {} cancel, {} amend, {} bad) -> {} trades, {} book messages, {} rejects; {:.0} lines/s; out {}",
        st.lines,
        st.events.new,
        st.events.cancel,
        st.events.amend,
        st.bad_input,
        st.trades,
        st.book_snapshots + st.book_deltas,
        st.rejects.values().sum::<u64>(),
        perf,
        a.out.display()
    );
    Ok(())
}
