//! Websocket fan-in → per-symbol aggregators → periodic lake flushes.
//!
//! One websocket connection per chunk of symbols. A reconnect invalidates only that chunk's books
//! (they're rebuilt from the snapshot Bybit sends on resubscribe). Candles close on exchange time
//! when a later event arrives, or on wall clock + grace for quiet symbols. Rows are buffered in
//! memory and flushed to `flow_3s` every `flush_secs` (and on shutdown).

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use at_bybit::ws::{self, WsMessage};
use at_core::time::{Ms, now_ms};
use at_store::Lake;
use serde_json::Value;
use tokio::sync::mpsc;

use crate::candle::{FlowRow, SymbolFlow, rows_to_batch};

#[derive(Clone, Debug)]
pub struct StreamCfg {
    pub category: String,
    pub symbols: Vec<String>,
    pub depth: u32,
    pub ws_url: String,
}

#[derive(Clone, Debug)]
pub struct CollectorCfg {
    pub venue: String,
    pub streams: Vec<StreamCfg>,
    pub symbols_per_conn: usize,
    pub grace_ms: Ms,
    pub flush_secs: u64,
    /// Stop after this long (None = run forever).
    pub run_for: Option<Duration>,
}

enum Event {
    Msg(usize, WsMessage),
    Reset(usize, Vec<String>),
}

#[derive(Default, Debug, Clone)]
pub struct Stats {
    pub msgs: u64,
    pub trades: u64,
    pub book_msgs: u64,
    pub rows_written: u64,
    pub resets: u64,
    pub files: u64,
    pub liquidations: u64,
}

/// One forced liquidation from `allLiquidation.{symbol}` (linear only). `side` is the side of the
/// LIQUIDATED position: "long" (Bybit S = "Buy") means a long was force-sold; "short" means forced buying.
#[derive(Clone, Debug, PartialEq)]
pub struct LiqRow {
    pub symbol: String,
    pub ts: i64,
    pub side: &'static str,
    pub price: f64,
    pub size: f64,
    pub known_ts: i64,
}

/// Parse an allLiquidation message. known_ts is never before the event (Bybit's clock runs ~0.3 s ahead).
pub fn parse_liquidations(data: &Value, received_ts: i64) -> Vec<LiqRow> {
    data.as_array()
        .into_iter()
        .flatten()
        .filter_map(|x| {
            let side = match x["S"].as_str()? {
                "Buy" => "long",
                "Sell" => "short",
                _ => return None,
            };
            let ts = x["T"].as_i64()?;
            Some(LiqRow { symbol: x["s"].as_str()?.to_string(), ts, side, price: num(&x["p"])?, size: num(&x["v"])?, known_ts: received_ts.max(ts) })
        })
        .collect()
}

pub fn liq_batch(venue: &str, rows: &[LiqRow]) -> at_store::Batch {
    let n = rows.len();
    at_store::Batch::new()
        .utf8_const("venue", venue, n)
        .utf8("symbol", rows.iter().map(|r| r.symbol.clone()).collect())
        .ts("event_ts", rows.iter().map(|r| r.ts).collect())
        .utf8("side", rows.iter().map(|r| r.side.to_string()).collect())
        .f64("price", rows.iter().map(|r| r.price).collect())
        .f64("size", rows.iter().map(|r| r.size).collect())
        .ts("known_ts", rows.iter().map(|r| r.known_ts).collect())
}

fn key(stream: usize, symbol: &str) -> (usize, String) {
    (stream, symbol.to_string())
}

pub async fn run(cfg: CollectorCfg, lake: Arc<Lake>) -> anyhow::Result<Stats> {
    let (tx, mut rx) = mpsc::channel::<Event>(100_000);
    let mut handles = Vec::new();
    for (si, st) in cfg.streams.iter().enumerate() {
        for chunk in st.symbols.chunks(cfg.symbols_per_conn.max(1)) {
            let mut topics: Vec<String> = chunk.iter().flat_map(|s| [format!("publicTrade.{s}"), format!("orderbook.{}.{s}", st.depth)]).collect();
            if st.category == "linear" {
                topics.extend(chunk.iter().map(|s| format!("allLiquidation.{s}")));
            }
            let (ctx, mut crx) = mpsc::channel::<WsMessage>(50_000);
            handles.push(tokio::spawn(ws::run(st.ws_url.clone(), topics, None, ctx)));
            let fwd = tx.clone();
            let syms = chunk.to_vec();
            handles.push(tokio::spawn(async move {
                while let Some(m) = crx.recv().await {
                    let ev = if m.topic == "__reconnect__" { Event::Reset(si, syms.clone()) } else { Event::Msg(si, m) };
                    if fwd.send(ev).await.is_err() {
                        break;
                    }
                }
                Ok(())
            }));
        }
    }
    drop(tx);

    let mut flows: HashMap<(usize, String), SymbolFlow> = HashMap::new();
    for (si, st) in cfg.streams.iter().enumerate() {
        for s in &st.symbols {
            flows.insert(key(si, s), SymbolFlow::new(s));
        }
    }
    let mut buffers: Vec<Vec<FlowRow>> = vec![Vec::new(); cfg.streams.len()];
    let mut liqs: Vec<LiqRow> = Vec::new();
    let mut stats = Stats::default();
    let mut tick = tokio::time::interval(Duration::from_millis(500));
    let mut next_flush = now_ms() + cfg.flush_secs as i64 * 1000;
    let deadline = cfg.run_for.map(|d| now_ms() + d.as_millis() as i64);
    let ctrl_c = tokio::signal::ctrl_c();
    tokio::pin!(ctrl_c);

    loop {
        tokio::select! {
            ev = rx.recv() => {
                let Some(ev) = ev else { break };
                match ev {
                    Event::Reset(si, syms) => {
                        stats.resets += 1;
                        let t = now_ms();
                        for s in syms {
                            if let Some(f) = flows.get_mut(&key(si, &s)) { f.reset_book(t); }
                        }
                    }
                    Event::Msg(si, m) => {
                        stats.msgs += 1;
                        if m.topic.starts_with("allLiquidation.") {
                            let rows = parse_liquidations(&m.data, m.received_ts);
                            stats.liquidations += rows.len() as u64;
                            liqs.extend(rows);
                        } else {
                            handle(si, &m, &mut flows, &mut buffers[si], &mut stats);
                        }
                    }
                }
            }
            _ = tick.tick() => {
                let now = now_ms();
                // Close quiet candles once the wall clock is past their end + grace.
                let horizon = now - cfg.grace_ms;
                for ((si, _), f) in flows.iter_mut() {
                    f.roll_to(horizon, now, &mut buffers[*si]);
                }
                if now >= next_flush {
                    flush(&cfg, &lake, &mut buffers, &mut liqs, &mut stats).await?;
                    tracing::info!(msgs = stats.msgs, trades = stats.trades, book = stats.book_msgs, rows = stats.rows_written, liquidations = stats.liquidations, resets = stats.resets, "flushed");
                    next_flush = now + cfg.flush_secs as i64 * 1000;
                }
                if deadline.is_some_and(|d| now >= d) { break; }
            }
            _ = &mut ctrl_c => { tracing::info!("ctrl-c: flushing and stopping"); break; }
        }
    }
    flush(&cfg, &lake, &mut buffers, &mut liqs, &mut stats).await?;
    for h in handles {
        h.abort();
    }
    Ok(stats)
}

fn handle(si: usize, m: &WsMessage, flows: &mut HashMap<(usize, String), SymbolFlow>, out: &mut Vec<FlowRow>, stats: &mut Stats) {
    let mut parts = m.topic.splitn(3, '.');
    let kind = parts.next().unwrap_or_default();
    let known = m.received_ts;
    match kind {
        "publicTrade" => {
            let sym = parts.next().unwrap_or_default();
            let Some(f) = flows.get_mut(&key(si, sym)) else { return };
            for t in m.data.as_array().into_iter().flatten() {
                let (Some(ts), Some(p), Some(v)) = (t["T"].as_i64(), num(&t["p"]), num(&t["v"])) else { continue };
                f.on_trade(ts, p, v, t["S"].as_str() == Some("Buy"), known, out);
                stats.trades += 1;
            }
        }
        "orderbook" => {
            let _depth = parts.next();
            let sym = parts.next().unwrap_or_default();
            let Some(f) = flows.get_mut(&key(si, sym)) else { return };
            let ts = m.ts.unwrap_or(known);
            f.on_book(ts, &m.kind, &m.data, known, out);
            stats.book_msgs += 1;
        }
        _ => {}
    }
}

fn num(v: &Value) -> Option<f64> {
    v.as_str().and_then(|s| s.parse().ok()).or_else(|| v.as_f64())
}

async fn flush(cfg: &CollectorCfg, lake: &Arc<Lake>, buffers: &mut [Vec<FlowRow>], liqs: &mut Vec<LiqRow>, stats: &mut Stats) -> anyhow::Result<()> {
    if !liqs.is_empty() {
        let batch = liq_batch(&cfg.venue, &std::mem::take(liqs));
        let lake = lake.clone();
        tokio::task::spawn_blocking(move || lake.write("liquidations", &batch)).await??;
    }
    for (si, buf) in buffers.iter_mut().enumerate() {
        if buf.is_empty() {
            continue;
        }
        let rows = std::mem::take(buf);
        let batch = rows_to_batch(&cfg.venue, &cfg.streams[si].category, &rows);
        let lake = lake.clone();
        let rep = tokio::task::spawn_blocking(move || lake.write("flow_3s", &batch)).await??;
        stats.rows_written += rep.rows as u64;
        stats.files += rep.files.len() as u64;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn parses_liquidations_and_clamps_known_ts() {
        let data = json!([
            {"T": 1_000_500, "s": "BTCUSDT", "S": "Buy", "v": "0.5", "p": "75000.1"},
            {"T": 1_000_100, "s": "ETHUSDT", "S": "Sell", "v": "3", "p": "2230"},
            {"T": 1_000_200, "s": "XRPUSDT", "S": "?", "v": "1", "p": "1"}
        ]);
        let rows = parse_liquidations(&data, 1_000_300);
        assert_eq!(rows.len(), 2);
        assert_eq!(rows[0].side, "long");
        assert_eq!(rows[0].known_ts, 1_000_500, "never known before it happened");
        assert_eq!(rows[1].side, "short");
        assert_eq!(rows[1].known_ts, 1_000_300);
    }

    #[test]
    fn liquidation_batch_matches_schema() {
        let root = std::env::temp_dir().join(format!("liq-test-{}", now_ms()));
        let schema = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../config/schema.yaml");
        let lake = Lake::open(&root, &schema).unwrap();
        let rows = parse_liquidations(&json!([{"T": 1_790_000_000_000i64, "s": "BTCUSDT", "S": "Buy", "v": "0.5", "p": "75000"}]), 1_790_000_000_100);
        let rep = lake.write("liquidations", &liq_batch("bybit", &rows)).unwrap();
        assert_eq!(rep.rows, 1);
        std::fs::remove_dir_all(root).ok();
    }
}
