//! Websocket fan-in → per-symbol aggregators → periodic lake flushes.
//!
//! One websocket connection per chunk of symbols, each run by a supervisor task. A reconnect
//! invalidates only that chunk's books (they're rebuilt from the snapshot Bybit sends on resubscribe).
//! A connection that delivers no data message for `silence_ms` is torn down and reconnected: this
//! catches websocket stalls and dead TCP sessions after a resume from standby, which the websocket
//! client alone only notices minutes later. Repeated silence on a session that delivered nothing
//! but its subscription snapshots is a quiet market (e.g. TradFi perps off hours), so the threshold
//! doubles (up to 16x) instead of reconnecting every `silence_ms`.
//!
//! Candles close on a trade + book watermark (see [`crate::candle`]). Rows are buffered in memory and
//! flushed to `flow_3s` every `flush_secs` (and on shutdown). Every `heartbeat_secs` each connection
//! logs one `heartbeat` line (messages, trades, book updates, trade lag, late trades, silent symbols)
//! and warns when its trade rate falls below 20% of its trailing rate while book messages keep
//! flowing (the `trade_deficit` pattern WS02 found only after the fact).

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use at_bybit::ws::{self, WsMessage};
use at_core::time::{Ms, now_ms};
use at_store::Lake;
use serde_json::Value;
use tokio::sync::mpsc;

use crate::candle::{ClosePolicy, FlowRow, SymbolFlow, rows_to_batch};

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
    /// Wall-clock grace before a candle of an idle symbol is closed.
    pub grace_ms: Ms,
    /// Hard cap on how long a candle waits for a lagging stream.
    pub max_hold_ms: Ms,
    /// Reconnect a connection that delivered no data message for this long (0 = never).
    pub silence_ms: Ms,
    /// Per-connection heartbeat log period (0 = off).
    pub heartbeat_secs: u64,
    pub flush_secs: u64,
    /// Stop after this long (None = run forever).
    pub run_for: Option<Duration>,
}

enum Event {
    Msg(usize, WsMessage),
    Reset(usize, &'static str),
}

#[derive(Default, Debug, Clone)]
pub struct Stats {
    pub msgs: u64,
    pub trades: u64,
    pub book_msgs: u64,
    pub rows_written: u64,
    pub resets: u64,
    pub silent_reconnects: u64,
    pub late_trades: u64,
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

/// Silence threshold for the next session: back to the base after a healthy session, doubled
/// (capped at 16x) after a session that delivered no more than its subscription snapshots.
pub fn next_silence_limit(base: Ms, current: Ms, session_msgs: u64, n_topics: usize) -> Ms {
    if session_msgs > 4 * n_topics as u64 { base } else { (current * 2).min(base * 16) }
}

/// Run one connection forever: (re)spawn the websocket, forward its messages, tear it down when silent.
async fn supervise(ci: usize, url: String, topics: Vec<String>, silence_ms: Ms, fwd: mpsc::Sender<Event>) {
    let mut limit = silence_ms;
    loop {
        let (ctx, mut crx) = mpsc::channel::<WsMessage>(50_000);
        let ws_task = tokio::spawn(ws::run(url.clone(), topics.clone(), None, ctx));
        let started = now_ms();
        let mut last = started;
        let mut session_msgs = 0u64;
        let mut check = tokio::time::interval(Duration::from_secs(1));
        let silent = loop {
            tokio::select! {
                m = crx.recv() => match m {
                    None => break None,
                    Some(m) => {
                        let ev = if m.topic == "__reconnect__" {
                            Event::Reset(ci, "ws_reconnect")
                        } else {
                            last = m.received_ts;
                            session_msgs += 1;
                            Event::Msg(ci, m)
                        };
                        if fwd.send(ev).await.is_err() {
                            ws_task.abort();
                            return;
                        }
                    }
                },
                _ = check.tick() => {
                    // Wall clock on purpose: after a resume from standby it jumps, so a dead session is caught at once.
                    let quiet = now_ms() - last;
                    if silence_ms > 0 && quiet > limit {
                        break Some(quiet);
                    }
                }
            }
        };
        ws_task.abort();
        let next = next_silence_limit(silence_ms, limit, session_msgs, topics.len());
        match silent {
            Some(quiet) => tracing::warn!(conn = ci, url = %url, silent_s = quiet / 1000, session_msgs, next_limit_s = next / 1000, "stream silent: reconnecting"),
            None => tracing::warn!(conn = ci, url = %url, "websocket task ended: restarting"),
        }
        limit = next;
        if fwd.send(Event::Reset(ci, if silent.is_some() { "silent" } else { "ws_ended" })).await.is_err() {
            return;
        }
        if silent.is_none() {
            tokio::time::sleep(Duration::from_secs(1)).await;
        }
    }
}

struct Conn {
    si: usize,
    syms: Vec<String>,
    label: String,
}

/// Per-connection counters for one heartbeat window, plus trailing rates.
#[derive(Default)]
struct Beat {
    msgs: u64,
    trades: u64,
    book: u64,
    liqs: u64,
    resets: u64,
    silent: u64,
    lag_sum: i64,
    lag_n: u64,
    lag_max: i64,
    last_rx: Ms,
    // trailing (EMA over windows) trade and book rates, and the number of windows seen
    ema_trades: f64,
    ema_book: f64,
    windows: u64,
    // symbol counters at the start of the window
    sym_trades: Vec<u64>,
    late0: u64,
}

const EMA_ALPHA: f64 = 1.0 / 30.0;

/// The trade feed is trickling: trades far below their trailing rate while the book keeps flowing.
pub fn trade_deficit(trades: u64, book: u64, ema_trades: f64, ema_book: f64, windows: u64) -> bool {
    windows >= 10 && ema_trades >= 50.0 && (trades as f64) < 0.2 * ema_trades && (book as f64) >= 0.5 * ema_book
}

pub async fn run(cfg: CollectorCfg, lake: Arc<Lake>) -> anyhow::Result<Stats> {
    let (tx, mut rx) = mpsc::channel::<Event>(100_000);
    let mut handles = Vec::new();
    let mut conns: Vec<Conn> = Vec::new();
    for (si, st) in cfg.streams.iter().enumerate() {
        for chunk in st.symbols.chunks(cfg.symbols_per_conn.max(1)) {
            let mut topics: Vec<String> = chunk.iter().flat_map(|s| [format!("publicTrade.{s}"), format!("orderbook.{}.{s}", st.depth)]).collect();
            if st.category == "linear" {
                topics.extend(chunk.iter().map(|s| format!("allLiquidation.{s}")));
            }
            let ci = conns.len();
            conns.push(Conn { si, syms: chunk.to_vec(), label: format!("{}/{} {}+{}", st.category, st.depth, chunk[0], chunk.len() - 1) });
            handles.push(tokio::spawn(supervise(ci, st.ws_url.clone(), topics, cfg.silence_ms, tx.clone())));
        }
    }
    drop(tx);

    let policy = ClosePolicy { grace_ms: cfg.grace_ms, max_hold_ms: cfg.max_hold_ms.max(cfg.grace_ms) };
    let mut flows: HashMap<(usize, String), SymbolFlow> = HashMap::new();
    for (si, st) in cfg.streams.iter().enumerate() {
        for s in &st.symbols {
            flows.insert(key(si, s), SymbolFlow::with_policy(s, policy));
        }
    }
    let mut beats: Vec<Beat> = conns.iter().map(|c| Beat { sym_trades: vec![0; c.syms.len()], last_rx: now_ms(), ..Default::default() }).collect();
    let mut buffers: Vec<Vec<FlowRow>> = vec![Vec::new(); cfg.streams.len()];
    let mut liqs: Vec<LiqRow> = Vec::new();
    let mut stats = Stats::default();
    let mut tick = tokio::time::interval(Duration::from_millis(500));
    let mut next_flush = now_ms() + cfg.flush_secs as i64 * 1000;
    let hb_ms = cfg.heartbeat_secs as i64 * 1000;
    let mut next_beat = now_ms() + hb_ms;
    let deadline = cfg.run_for.map(|d| now_ms() + d.as_millis() as i64);
    let ctrl_c = tokio::signal::ctrl_c();
    tokio::pin!(ctrl_c);

    loop {
        tokio::select! {
            ev = rx.recv() => {
                let Some(ev) = ev else { break };
                match ev {
                    Event::Reset(ci, reason) => {
                        stats.resets += 1;
                        beats[ci].resets += 1;
                        if reason == "silent" {
                            stats.silent_reconnects += 1;
                            beats[ci].silent += 1;
                        }
                        let t = now_ms();
                        let c = &conns[ci];
                        for s in &c.syms {
                            if let Some(f) = flows.get_mut(&key(c.si, s)) { f.reset_book(t); }
                        }
                    }
                    Event::Msg(ci, m) => {
                        stats.msgs += 1;
                        let si = conns[ci].si;
                        let b = &mut beats[ci];
                        b.msgs += 1;
                        b.last_rx = m.received_ts;
                        if m.topic.starts_with("allLiquidation.") {
                            let rows = parse_liquidations(&m.data, m.received_ts);
                            stats.liquidations += rows.len() as u64;
                            b.liqs += rows.len() as u64;
                            liqs.extend(rows);
                        } else {
                            handle(si, &m, &mut flows, &mut buffers[si], &mut stats, b);
                        }
                    }
                }
            }
            _ = tick.tick() => {
                let now = now_ms();
                // Close candles whose streams are idle past the grace, or held past the cap.
                for ((si, _), f) in flows.iter_mut() {
                    f.advance(now, &mut buffers[*si]);
                }
                if hb_ms > 0 && now >= next_beat {
                    heartbeat(&conns, &mut beats, &flows, now, &mut stats);
                    next_beat = now + hb_ms;
                }
                if now >= next_flush {
                    flush(&cfg, &lake, &mut buffers, &mut liqs, &mut stats).await?;
                    tracing::info!(msgs = stats.msgs, trades = stats.trades, book = stats.book_msgs, rows = stats.rows_written, liquidations = stats.liquidations, resets = stats.resets, silent_reconnects = stats.silent_reconnects, late_trades = stats.late_trades, "flushed");
                    next_flush = now + cfg.flush_secs as i64 * 1000;
                }
                if deadline.is_some_and(|d| now >= d) { break; }
            }
            _ = &mut ctrl_c => { tracing::info!("ctrl-c: flushing and stopping"); break; }
        }
    }
    // Close every candle that has ended (the open one is dropped, as before), then write.
    let now = now_ms();
    for ((si, _), f) in flows.iter_mut() {
        f.close_until(now, now, &mut buffers[*si]);
    }
    flush(&cfg, &lake, &mut buffers, &mut liqs, &mut stats).await?;
    for h in handles {
        h.abort();
    }
    Ok(stats)
}

fn heartbeat(conns: &[Conn], beats: &mut [Beat], flows: &HashMap<(usize, String), SymbolFlow>, now: Ms, stats: &mut Stats) {
    for (ci, c) in conns.iter().enumerate() {
        let b = &mut beats[ci];
        let mut quiet_syms = 0;
        let mut late = 0u64;
        let mut open_max = 0usize;
        for (k, s) in c.syms.iter().enumerate() {
            if let Some(f) = flows.get(&key(c.si, s)) {
                if f.trades == b.sym_trades[k] {
                    quiet_syms += 1;
                }
                b.sym_trades[k] = f.trades;
                late += f.late_trades;
                open_max = open_max.max(f.open_candles());
            }
        }
        let late_win = late - b.late0;
        b.late0 = late;
        stats.late_trades += late_win;
        let lag_mean = if b.lag_n > 0 { b.lag_sum / b.lag_n as i64 } else { 0 };
        tracing::info!(
            conn = ci,
            stream = %c.label,
            msgs = b.msgs,
            trades = b.trades,
            book = b.book,
            liq = b.liqs,
            quiet_syms,
            silent_s = (now - b.last_rx) / 1000,
            trade_lag_ms_mean = lag_mean,
            trade_lag_ms_max = b.lag_max,
            late_trades = late_win,
            open_candles_max = open_max,
            resets = b.resets,
            silent_reconnects = b.silent,
            "heartbeat"
        );
        if trade_deficit(b.trades, b.book, b.ema_trades, b.ema_book, b.windows) {
            tracing::warn!(conn = ci, stream = %c.label, trades = b.trades, trailing = b.ema_trades as u64, book = b.book, "trade feed trickling while the book flows (trade_deficit)");
        }
        if b.windows == 0 {
            (b.ema_trades, b.ema_book) = (b.trades as f64, b.book as f64);
        } else {
            b.ema_trades += EMA_ALPHA * (b.trades as f64 - b.ema_trades);
            b.ema_book += EMA_ALPHA * (b.book as f64 - b.ema_book);
        }
        b.windows += 1;
        (b.msgs, b.trades, b.book, b.liqs, b.resets, b.silent, b.lag_sum, b.lag_n, b.lag_max) = (0, 0, 0, 0, 0, 0, 0, 0, 0);
    }
}

fn handle(si: usize, m: &WsMessage, flows: &mut HashMap<(usize, String), SymbolFlow>, out: &mut Vec<FlowRow>, stats: &mut Stats, beat: &mut Beat) {
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
                beat.trades += 1;
                let lag = known - ts;
                beat.lag_sum += lag;
                beat.lag_n += 1;
                beat.lag_max = beat.lag_max.max(lag);
            }
            f.advance(known, out);
        }
        "orderbook" => {
            let _depth = parts.next();
            let sym = parts.next().unwrap_or_default();
            let Some(f) = flows.get_mut(&key(si, sym)) else { return };
            let ts = m.ts.unwrap_or(known);
            f.on_book(ts, &m.kind, &m.data, known, out);
            f.advance(known, out);
            stats.book_msgs += 1;
            beat.book += 1;
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

    #[test]
    fn silence_threshold_backs_off_on_quiet_sessions_only() {
        let base = 30_000;
        // healthy session (data beyond the snapshots) -> back to the base
        assert_eq!(next_silence_limit(base, 120_000, 1_000, 75), base);
        // only snapshots, then silence again -> double, capped at 16x
        assert_eq!(next_silence_limit(base, base, 80, 75), 60_000);
        assert_eq!(next_silence_limit(base, 400_000, 0, 75), 480_000);
        assert_eq!(next_silence_limit(base, 480_000, 0, 75), 480_000);
    }

    #[test]
    fn trade_deficit_needs_history_and_a_live_book() {
        assert!(!trade_deficit(10, 1_000, 500.0, 1_000.0, 3), "warm-up");
        assert!(trade_deficit(10, 1_000, 500.0, 1_000.0, 30));
        assert!(!trade_deficit(10, 100, 500.0, 1_000.0, 30), "book also dropped: an outage, not a trickle");
        assert!(!trade_deficit(5, 1_000, 20.0, 1_000.0, 30), "symbols that barely trade");
        assert!(!trade_deficit(200, 1_000, 500.0, 1_000.0, 30));
    }
}
