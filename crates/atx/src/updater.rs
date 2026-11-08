//! Live market-data updater (`atx update`): keeps bars, funding, open interest, DVOL and
//! Fear & Greed current in the lake between manual backfills. Config: `live:` in settings.yaml.
//!
//! Each stream (e.g. `bars/last/1h/BTCUSDT`) has a cursor in `data/state/updater_cursor.json`:
//! the event_ts of the newest row written. A stream is polled when its next item should exist
//! (cursor + step + availability lag); only rows strictly newer than the cursor are written, and
//! the cursor advances only after the lake write succeeded. New rows of one cycle are merged into
//! one write per table (one Parquet file per partition, not one per stream). Without a cursor a stream starts at
//! now - `backfill_days` (overlapping existing data is harmless: the reader de-duplicates).
//! Per-stream errors are logged and retried with capped exponential backoff; the loop never stops.

use std::collections::{BTreeMap, HashMap};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use anyhow::{Context, bail};
use at_alt as alt;
use at_bybit::BybitRest;
use at_bybit::lake as bl;
use at_bybit::market::{MarketData, PriceType, period_ms};
use at_core::time::{DAY, Freq, Ms, SECOND, now_ms, to_datetime};
use serde_yaml::Value as Yaml;
use tokio::sync::Notify;

use crate::cmd::Ctx;

/// 23 long-history research symbols + 13 TradFi perps (flow.streams).
pub const DEFAULT_SYMBOLS: [&str; 36] = [
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "DOGEUSDT", "ADAUSDT", "LINKUSDT", "LTCUSDT", "BCHUSDT", "AVAXUSDT", "DOTUSDT", "BNBUSDT", "TRXUSDT",
    "NEARUSDT", "ATOMUSDT", "ETCUSDT", "FILUSDT", "UNIUSDT", "AAVEUSDT", "XLMUSDT", "SUIUSDT", "ENAUSDT", "HYPEUSDT", "SOXLUSDT", "QQQUSDT", "NVDAUSDT",
    "TSLAUSDT", "MSTRUSDT", "MUUSDT", "GOOGLUSDT", "AAPLUSDT", "INTCUSDT", "SPCXUSDT", "XAUUSDT", "XAGUSDT", "CLUSDT",
];

const MAX_BACKOFF_MS: Ms = 600 * SECOND;

// ---------------------------------------------------------------- config

#[derive(Clone, Debug, PartialEq)]
pub struct LiveCfg {
    pub symbols: Vec<String>,
    pub bar_freqs: Vec<Freq>,
    pub price_types: Vec<PriceType>,
    pub funding: bool,
    pub funding_poll_secs: u64,
    /// Open-interest interval (Bybit `intervalTime`), `None` = disabled.
    pub oi_interval: Option<String>,
    pub dvol: Vec<String>,
    /// Deribit resolution in seconds.
    pub dvol_resolution: String,
    pub fng: bool,
    pub poll_secs: u64,
    pub bar_grace_secs: u64,
    pub backfill_days: i64,
    pub cursor_file: PathBuf,
}

fn str_list(v: &Yaml, what: &str) -> anyhow::Result<Vec<String>> {
    match v {
        Yaml::Null => Ok(Vec::new()),
        Yaml::Bool(false) => Ok(Vec::new()),
        Yaml::Sequence(xs) => xs.iter().map(|x| x.as_str().map(|s| s.trim().to_string()).with_context(|| format!("live.{what}: expected strings"))).collect(),
        Yaml::String(s) => Ok(s.split(',').map(|x| x.trim().to_string()).filter(|x| !x.is_empty()).collect()),
        _ => bail!("live.{what}: expected a list"),
    }
}

/// Symbols from a universe file: whitespace/comma separated, `#` starts a comment.
pub fn parse_symbol_file(text: &str) -> Vec<String> {
    text.lines().map(|l| l.split('#').next().unwrap_or("")).flat_map(|l| l.split(|c: char| c == ',' || c.is_whitespace())).filter(|s| !s.is_empty()).map(str::to_string).collect()
}

fn dedup_upper(xs: Vec<String>) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    for s in xs.into_iter().map(|s| s.trim().to_uppercase()).filter(|s| !s.is_empty()) {
        if !out.contains(&s) {
            out.push(s);
        }
    }
    out
}

impl LiveCfg {
    /// Parse the `live:` section. Relative paths resolve against `root`.
    pub fn from_yaml(live: &Yaml, root: &Path) -> anyhow::Result<Self> {
        let get = |k: &str| live.get(k).unwrap_or(&Yaml::Null);
        let u64_or = |k: &str, d: u64| -> anyhow::Result<u64> {
            match get(k) {
                Yaml::Null => Ok(d),
                v => v.as_u64().with_context(|| format!("live.{k}: expected a non-negative integer")),
            }
        };
        let bool_or = |k: &str, d: bool| -> anyhow::Result<bool> {
            match get(k) {
                Yaml::Null => Ok(d),
                v => v.as_bool().with_context(|| format!("live.{k}: expected true/false")),
            }
        };
        let symbols = match get("symbols") {
            Yaml::Null => DEFAULT_SYMBOLS.iter().map(|s| s.to_string()).collect(),
            Yaml::String(s) if !s.contains(',') => {
                let p = root.join(s);
                parse_symbol_file(&std::fs::read_to_string(&p).with_context(|| format!("live.symbols: reading universe file {}", p.display()))?)
            }
            v => str_list(v, "symbols")?,
        };
        let symbols = dedup_upper(symbols);
        anyhow::ensure!(!symbols.is_empty(), "live.symbols is empty");
        let bar_freqs = match get("bar_freqs") {
            Yaml::Null => vec![Freq::H1, Freq::H4, Freq::D1],
            v => str_list(v, "bar_freqs")?.iter().map(|s| Freq::parse(s)).collect::<Result<_, _>>()?,
        };
        let price_types = match get("price_types") {
            Yaml::Null => vec![PriceType::Last, PriceType::Mark],
            v => str_list(v, "price_types")?.iter().map(|s| PriceType::parse(s)).collect::<Result<_, _>>()?,
        };
        let oi_interval = match get("open_interest") {
            Yaml::Null | Yaml::Bool(true) => Some("4h".to_string()),
            Yaml::Bool(false) => None,
            Yaml::String(s) => Some(s.clone()),
            m @ Yaml::Mapping(_) => Some(m.get("interval").and_then(Yaml::as_str).unwrap_or("4h").to_string()),
            _ => bail!("live.open_interest: expected false or {{interval: 4h}}"),
        };
        if let Some(i) = &oi_interval {
            period_ms(i).context("live.open_interest.interval")?;
        }
        let dvol = match get("dvol") {
            Yaml::Null => vec!["BTC".to_string(), "ETH".to_string()],
            v => str_list(v, "dvol")?.into_iter().map(|s| s.to_uppercase()).collect(),
        };
        let dvol_resolution = match get("dvol_resolution") {
            Yaml::Null => "3600".to_string(),
            Yaml::Number(n) => n.to_string(),
            Yaml::String(s) => s.clone(),
            _ => bail!("live.dvol_resolution: expected seconds"),
        };
        dvol_resolution.parse::<u64>().ok().filter(|&x| x > 0).context("live.dvol_resolution: expected seconds, e.g. 3600")?;
        let cursor_file = root.join(get("cursor_file").as_str().unwrap_or("data/state/updater_cursor.json"));
        let cfg = Self {
            symbols,
            bar_freqs,
            price_types,
            funding: bool_or("funding", true)?,
            funding_poll_secs: u64_or("funding_poll_secs", 600)?,
            oi_interval,
            dvol,
            dvol_resolution,
            fng: bool_or("fng", true)?,
            poll_secs: u64_or("poll_secs", 20)?.max(1),
            bar_grace_secs: u64_or("bar_grace_secs", 5)?,
            backfill_days: u64_or("backfill_days", 3)? as i64,
            cursor_file,
        };
        Ok(cfg)
    }
}

// ---------------------------------------------------------------- cursor state (pure parts)

/// Persisted cursors: stream key -> event_ts of the newest row written to the lake.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Cursors(BTreeMap<String, Ms>);

impl Cursors {
    pub fn get(&self, key: &str) -> Option<Ms> {
        self.0.get(key).copied()
    }

    /// Move a cursor forward. Never moves backwards; returns whether it changed.
    pub fn advance(&mut self, key: &str, ts: Ms) -> bool {
        match self.0.get(key) {
            Some(&c) if c >= ts => false,
            _ => {
                self.0.insert(key.to_string(), ts);
                true
            }
        }
    }

    pub fn len(&self) -> usize {
        self.0.len()
    }

    pub fn to_json(&self) -> String {
        let streams: serde_json::Map<String, serde_json::Value> = self.0.iter().map(|(k, v)| (k.clone(), (*v).into())).collect();
        let doc = serde_json::json!({ "version": 1, "updated_utc": to_datetime(now_ms()).to_rfc3339(), "streams": streams });
        serde_json::to_string_pretty(&doc).unwrap_or_default()
    }

    pub fn from_json(text: &str) -> anyhow::Result<Self> {
        let v: serde_json::Value = serde_json::from_str(text)?;
        let streams = v["streams"].as_object().context("cursor file: no `streams` object")?;
        let mut out = BTreeMap::new();
        for (k, x) in streams {
            out.insert(k.clone(), x.as_i64().with_context(|| format!("cursor {k}: not an integer"))?);
        }
        Ok(Self(out))
    }

    /// Missing file = fresh start. A corrupt file is logged and ignored (re-fetching is harmless).
    pub fn load(path: &Path) -> Self {
        match std::fs::read_to_string(path) {
            Ok(t) => Self::from_json(&t).unwrap_or_else(|e| {
                tracing::error!("ignoring unreadable cursor file {}: {e:#}", path.display());
                Self::default()
            }),
            Err(_) => Self::default(),
        }
    }

    /// Atomic replace: write a temp file next to the target, then rename over it.
    pub fn save(&self, path: &Path) -> anyhow::Result<()> {
        if let Some(d) = path.parent() {
            std::fs::create_dir_all(d)?;
        }
        let tmp = path.with_extension("json.tmp");
        std::fs::write(&tmp, self.to_json()).with_context(|| format!("writing {}", tmp.display()))?;
        std::fs::rename(&tmp, path).with_context(|| format!("renaming {} -> {}", tmp.display(), path.display()))?;
        Ok(())
    }
}

/// Rows to write and the cursor to commit once the write succeeded: only rows with
/// `ts > cursor` (and `ts >= floor`, the start used when there is no cursor yet).
/// Nothing new -> `None` (the cursor must not move).
pub fn plan_write<T>(cursor: Option<Ms>, floor: Ms, rows: Vec<T>, ts: impl Fn(&T) -> Ms) -> (Vec<T>, Option<Ms>) {
    let keep: Vec<T> = rows.into_iter().filter(|r| ts(r) >= floor && cursor.is_none_or(|c| ts(r) > c)).collect();
    let newest = keep.iter().map(&ts).max();
    (keep, newest)
}

/// Advance the cursors of one write group, only if that write succeeded. Returns whether any moved.
pub fn commit<T, E>(cursors: &mut Cursors, keys: &[(String, Ms)], write: &Result<T, E>) -> bool {
    write.is_ok() && keys.iter().fold(false, |moved, (k, ts)| cursors.advance(k, *ts) | moved)
}

/// When a stream's next item becomes available: next event_ts is `cursor + step`, which can be
/// fetched `lag` later (for bars: one bar duration + grace). No cursor -> due now.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Cadence {
    pub step: Ms,
    pub lag: Ms,
}

impl Cadence {
    pub fn next_due(self, cursor: Option<Ms>) -> Ms {
        cursor.map_or(Ms::MIN, |c| c + self.step + self.lag)
    }
}

/// Backoff after an empty or failed poll: poll interval doubling per consecutive miss, capped.
pub fn backoff_ms(poll_ms: Ms, streak: u32) -> Ms {
    let shift = streak.saturating_sub(1).min(16);
    poll_ms.saturating_mul(1 << shift).min(MAX_BACKOFF_MS.max(poll_ms))
}

// ---------------------------------------------------------------- streams

#[derive(Clone, Debug)]
enum Stream {
    Bars { sym: String, freq: Freq, pt: PriceType },
    Funding { sym: String },
    Oi { sym: String, interval: String, period: Ms },
    Dvol { cur: String },
    Fng,
}

impl Stream {
    fn key(&self) -> String {
        match self {
            Stream::Bars { sym, freq, pt } => format!("bars/{}/{freq}/{sym}", pt.as_str()),
            Stream::Funding { sym } => format!("funding/{sym}"),
            Stream::Oi { sym, interval, .. } => format!("open_interest/{interval}/{sym}"),
            Stream::Dvol { cur } => format!("dvol/{cur}"),
            Stream::Fng => "fng".to_string(),
        }
    }
}

/// Rows staged in one cycle: table -> (merged batch, (stream key, newest event_ts) per stream).
type Pending = BTreeMap<&'static str, (at_store::Batch, Vec<(String, Ms)>)>;

struct Staged {
    table: &'static str,
    batch: at_store::Batch,
    key: String,
    newest: Ms,
}

#[derive(Default)]
struct Retry {
    streak: u32,
    not_before: Ms,
}

#[derive(Default, Debug)]
struct CycleStats {
    due: usize,
    written: usize,
    rows: usize,
    empty: usize,
    errors: usize,
}

struct Updater<'a> {
    ctx: &'a Ctx,
    cfg: LiveCfg,
    md: MarketData<'a>,
    http: reqwest::Client,
    cursors: Cursors,
    retry: HashMap<String, Retry>,
    last_poll: HashMap<String, Ms>,
    streams: Vec<Stream>,
    dvol_res_ms: Ms,
}

impl<'a> Updater<'a> {
    fn cadence(&self, s: &Stream) -> Option<Cadence> {
        let grace = self.cfg.bar_grace_secs as Ms * SECOND;
        match s {
            Stream::Bars { freq, .. } => Some(Cadence { step: freq.duration_ms(), lag: freq.duration_ms() + grace }),
            Stream::Oi { period, .. } => Some(Cadence { step: *period, lag: *period + grace }),
            Stream::Dvol { .. } => Some(Cadence { step: self.dvol_res_ms, lag: self.dvol_res_ms + self.ctx.alt_latency("deribit_dvol") + grace }),
            Stream::Fng => Some(Cadence { step: DAY, lag: self.ctx.alt_latency("alternative_me_fng") }),
            Stream::Funding { .. } => None,
        }
    }

    fn is_due(&self, s: &Stream, key: &str, now: Ms) -> bool {
        if self.retry.get(key).is_some_and(|r| now < r.not_before) {
            return false;
        }
        match self.cadence(s) {
            Some(c) => now >= c.next_due(self.cursors.get(key)),
            None => self.last_poll.get(key).is_none_or(|&t| now - t >= self.cfg.funding_poll_secs as Ms * SECOND),
        }
    }

    /// Fetch the rows newer than the cursor. Nothing is written here (see [`Updater::flush`]).
    async fn fetch(&self, s: &Stream, key: &str, now: Ms) -> anyhow::Result<Option<Staged>> {
        let cursor = self.cursors.get(key);
        let start = cursor.map_or(now - self.cfg.backfill_days * DAY, |c| c + 1);
        let grace = self.cfg.bar_grace_secs as Ms * SECOND;
        let ctx = self.ctx;
        let (table, batch, newest) = match s {
            Stream::Bars { sym, freq, pt } => {
                let start = cursor.map_or(freq.floor(start), |_| start);
                // Only bars closed at least `grace` ago: guards against a small local clock lead.
                let end = freq.floor(now - grace);
                if start >= end {
                    return Ok(None);
                }
                let xs = self.md.klines(sym, *freq, *pt, start, end).await?;
                let (rows, newest) = plan_write(cursor, start, xs, |k| k.open_ts);
                ("bars", bl::klines(sym, *freq, *pt, &rows, ctx.latency("bars")), newest)
            }
            Stream::Funding { sym } => {
                let xs = self.md.funding_history(sym, start, now).await?;
                let (rows, newest) = plan_write(cursor, start, xs, |f| f.ts);
                ("funding", bl::funding(sym, &rows, ctx.latency("funding")), newest)
            }
            Stream::Oi { sym, interval, period } => {
                // Complete periods only: the venue may still update the current bucket.
                let end = (now - grace) - (now - grace).rem_euclid(*period);
                if start >= end {
                    return Ok(None);
                }
                let xs = self.md.open_interest(sym, interval, start, end).await?;
                let (rows, newest) = plan_write(cursor, start, xs, |x| x.ts);
                ("open_interest", bl::open_interest(sym, interval, *period, &rows, ctx.latency("open_interest")), newest)
            }
            Stream::Dvol { cur } => {
                let xs = alt::deribit::dvol(&self.http, cur, &self.cfg.dvol_resolution, start, now, ctx.alt_latency("deribit_dvol")).await?;
                let (rows, newest) = plan_write(cursor, start, xs, |o| o.event_ts);
                ("alt_series", alt::to_batch(alt::deribit::SOURCE, alt::deribit::PIT, &rows), newest)
            }
            Stream::Fng => {
                let xs = alt::fng::fetch_all(&self.http, ctx.alt_latency("alternative_me_fng")).await?;
                let (rows, newest) = plan_write(cursor, start, xs, |o| o.event_ts);
                ("alt_series", alt::to_batch(alt::fng::SOURCE, alt::fng::PIT, &rows), newest)
            }
        };
        Ok(newest.map(|newest| Staged { table, batch, key: key.to_string(), newest }))
    }

    /// One lake write per table; cursors of the streams in a write advance only if it succeeded.
    /// Returns (streams committed, rows committed, streams whose write failed).
    fn flush(&mut self, pending: Pending) -> (usize, usize, Vec<String>) {
        let (mut streams, mut rows, mut failed) = (0, 0, Vec::new());
        let mut moved = false;
        for (table, (batch, keys)) in pending {
            let res = self.ctx.lake.write(table, &batch);
            moved |= commit(&mut self.cursors, &keys, &res);
            match res {
                Ok(r) => {
                    streams += keys.len();
                    rows += r.rows;
                    tracing::debug!("{table}: {} rows from {} streams committed", r.rows, keys.len());
                }
                Err(e) => {
                    tracing::warn!("lake write {table} failed ({} streams, will retry): {e:#}", keys.len());
                    failed.extend(keys.into_iter().map(|(k, _)| k));
                }
            }
        }
        if moved && let Err(e) = self.cursors.save(&self.cfg.cursor_file) {
            // In-memory cursors stay advanced; after a restart the rows are fetched again (harmless).
            tracing::error!("saving cursors failed: {e:#}");
        }
        (streams, rows, failed)
    }

    fn note_miss(&mut self, key: String, now: Ms) {
        let r = self.retry.entry(key).or_default();
        r.streak += 1;
        r.not_before = now + backoff_ms(self.cfg.poll_secs as Ms * SECOND, r.streak);
    }

    async fn cycle(&mut self, stop: &AtomicBool) -> CycleStats {
        let mut st = CycleStats::default();
        let mut pending = Pending::new();
        let streams = std::mem::take(&mut self.streams);
        for s in &streams {
            if stop.load(Ordering::Relaxed) {
                break;
            }
            let key = s.key();
            let now = now_ms();
            if !self.is_due(s, &key, now) {
                continue;
            }
            st.due += 1;
            let res = self.fetch(s, &key, now).await;
            let is_funding = matches!(s, Stream::Funding { .. });
            if is_funding {
                self.last_poll.insert(key.clone(), now);
            }
            match res {
                Ok(Some(x)) => {
                    let slot = pending.entry(x.table).or_default();
                    match slot.0.append(x.batch) {
                        Ok(()) => slot.1.push((x.key, x.newest)),
                        Err(e) => {
                            st.errors += 1;
                            tracing::warn!(stream = %key, "staging failed: {e:#}");
                            self.note_miss(key, now);
                        }
                    }
                }
                Ok(None) => {
                    st.empty += 1;
                    // An empty funding poll is normal (settlements are hours apart).
                    if !is_funding {
                        self.note_miss(key, now);
                    }
                }
                Err(e) => {
                    st.errors += 1;
                    tracing::warn!(stream = %key, "update failed (will retry): {e:#}");
                    self.note_miss(key, now);
                }
            }
        }
        self.streams = streams;
        let staged: Vec<String> = pending.values().flat_map(|(_, ks)| ks.iter().map(|(k, _)| k.clone())).collect();
        let (written, rows, failed) = self.flush(pending);
        (st.written, st.rows) = (written, rows);
        st.errors += failed.len();
        let now = now_ms();
        for k in staged {
            if failed.contains(&k) {
                self.note_miss(k, now);
            } else {
                self.retry.remove(&k);
            }
        }
        st
    }

    /// Most overdue stream: how long its next item has been expected but not written.
    fn max_staleness(&self, now: Ms) -> (Ms, String, usize) {
        let mut worst = (0, "-".to_string());
        let mut uninit = 0;
        for s in &self.streams {
            let Some(c) = self.cadence(s) else { continue };
            let key = s.key();
            match self.cursors.get(&key) {
                None => uninit += 1,
                Some(cur) => {
                    let over = now - c.next_due(Some(cur));
                    if over > worst.0 {
                        worst = (over, key);
                    }
                }
            }
        }
        (worst.0, worst.1, uninit)
    }
}

fn fmt_dur(ms: Ms) -> String {
    let s = ms / 1000;
    match s {
        s if s < 120 => format!("{s}s"),
        s if s < 7200 => format!("{}m", s / 60),
        s => format!("{:.1}h", s as f64 / 3600.0),
    }
}

pub async fn run(ctx: &Ctx, once: bool) -> anyhow::Result<()> {
    let cfg = LiveCfg::from_yaml(&ctx.raw()["live"], &ctx.s.root)?;
    let rest: BybitRest = ctx.market_rest()?;
    let md = MarketData::new(&rest, ctx.category());
    match md.server_time_ms().await {
        Ok(server) => {
            let skew = now_ms() - server;
            if skew.abs() * 2 > cfg.bar_grace_secs as Ms * SECOND {
                tracing::warn!("local clock is {skew:+} ms off Bybit; consider raising live.bar_grace_secs");
            }
        }
        Err(e) => tracing::warn!("could not read Bybit server time: {e:#}"),
    }

    let mut streams = Vec::new();
    for sym in &cfg.symbols {
        for &pt in &cfg.price_types {
            for &freq in &cfg.bar_freqs {
                streams.push(Stream::Bars { sym: sym.clone(), freq, pt });
            }
        }
        if cfg.funding {
            streams.push(Stream::Funding { sym: sym.clone() });
        }
        if let Some(i) = &cfg.oi_interval {
            streams.push(Stream::Oi { sym: sym.clone(), interval: i.clone(), period: period_ms(i)? });
        }
    }
    streams.extend(cfg.dvol.iter().map(|c| Stream::Dvol { cur: c.clone() }));
    if cfg.fng {
        streams.push(Stream::Fng);
    }
    let cursors = Cursors::load(&cfg.cursor_file);
    tracing::info!(
        "updater: {} symbols, {} streams, {} cursors loaded from {}, poll {}s{}",
        cfg.symbols.len(),
        streams.len(),
        cursors.len(),
        cfg.cursor_file.display(),
        cfg.poll_secs,
        if once { " (single cycle)" } else { "" }
    );

    let stop = Arc::new(AtomicBool::new(false));
    let wake = Arc::new(Notify::new());
    {
        let (stop, wake) = (stop.clone(), wake.clone());
        tokio::spawn(async move {
            if tokio::signal::ctrl_c().await.is_ok() {
                tracing::info!("Ctrl-C: finishing the current stream, then stopping");
                stop.store(true, Ordering::Relaxed);
                wake.notify_waiters();
            }
        });
    }

    let dvol_res_ms = cfg.dvol_resolution.parse::<Ms>().unwrap_or(3600) * SECOND;
    let poll = Duration::from_secs(cfg.poll_secs);
    let mut up = Updater { ctx, cfg, md, http: alt::http()?, cursors, retry: HashMap::new(), last_poll: HashMap::new(), streams, dvol_res_ms };
    let mut n = 0u64;
    loop {
        n += 1;
        let t0 = Instant::now();
        let st = up.cycle(&stop).await;
        let (stale, worst, uninit) = up.max_staleness(now_ms());
        tracing::info!(
            "heartbeat cycle {n}: due {} written {} (+{} rows) empty {} errors {} | max staleness {} ({worst}) | {uninit} streams without data | {:.1}s",
            st.due,
            st.written,
            st.rows,
            st.empty,
            st.errors,
            fmt_dur(stale),
            t0.elapsed().as_secs_f64()
        );
        if once || stop.load(Ordering::Relaxed) {
            break;
        }
        tokio::select! {
            _ = tokio::time::sleep(poll.saturating_sub(t0.elapsed()).max(Duration::from_secs(1))) => {}
            _ = wake.notified() => {}
        }
        if stop.load(Ordering::Relaxed) {
            break;
        }
    }
    tracing::info!("updater stopped after {n} cycles");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn plan_write_only_newer_rows() {
        let (rows, c) = plan_write(Some(20), 0, vec![10, 20, 30, 40], |x| *x);
        assert_eq!(rows, vec![30, 40]);
        assert_eq!(c, Some(40));
    }

    #[test]
    fn plan_write_nothing_new_keeps_cursor() {
        let (rows, c) = plan_write(Some(40), 0, vec![10, 20, 40], |x| *x);
        assert!(rows.is_empty());
        assert_eq!(c, None);
        let (rows, c) = plan_write::<i64>(Some(40), 0, vec![], |x| *x);
        assert!(rows.is_empty() && c.is_none());
    }

    #[test]
    fn plan_write_without_cursor_respects_floor() {
        let (rows, c) = plan_write(None, 15, vec![30, 10, 20], |x| *x);
        assert_eq!(rows, vec![30, 20]);
        assert_eq!(c, Some(30));
    }

    #[test]
    fn cursor_never_moves_backwards() {
        let mut cs = Cursors::default();
        assert!(cs.advance("k", 100));
        assert!(!cs.advance("k", 50));
        assert!(!cs.advance("k", 100));
        assert_eq!(cs.get("k"), Some(100));
        assert!(cs.advance("k", 101));
        assert_eq!(cs.get("k"), Some(101));
    }

    #[test]
    fn no_write_no_advance_flow() {
        // The updater's path: plan -> lake write -> commit (cursor moves only if the write succeeded).
        fn step(cs: &mut Cursors, rows: Vec<i64>, write_ok: bool) -> bool {
            let (rows, newest) = plan_write(cs.get("k"), 0, rows, |x| *x);
            let Some(n) = newest else { return false };
            assert!(!rows.is_empty());
            let write: Result<usize, &str> = if write_ok { Ok(rows.len()) } else { Err("disk full") };
            commit(cs, &[("k".to_string(), n)], &write)
        }
        let mut cs = Cursors::default();
        assert!(!step(&mut cs, vec![1, 2, 3], false));
        assert_eq!(cs.get("k"), None);
        assert!(step(&mut cs, vec![1, 2, 3], true));
        assert_eq!(cs.get("k"), Some(3));
        assert!(!step(&mut cs, vec![2, 3], true));
        assert_eq!(cs.get("k"), Some(3));
        assert!(!step(&mut cs, vec![4, 5], false));
        assert_eq!(cs.get("k"), Some(3));
        assert!(step(&mut cs, vec![4, 5], true));
        assert_eq!(cs.get("k"), Some(5));
    }

    #[test]
    fn commit_group_all_or_nothing() {
        let mut cs = Cursors::default();
        cs.advance("a", 10);
        let keys = vec![("a".to_string(), 20), ("b".to_string(), 5)];
        assert!(!commit(&mut cs, &keys, &Err::<(), _>("io")));
        assert_eq!((cs.get("a"), cs.get("b")), (Some(10), None));
        assert!(commit(&mut cs, &keys, &Ok::<_, ()>(())));
        assert_eq!((cs.get("a"), cs.get("b")), (Some(20), Some(5)));
        // a stale group never rewinds
        assert!(!commit(&mut cs, &[("a".to_string(), 15)], &Ok::<_, ()>(())));
        assert_eq!(cs.get("a"), Some(20));
    }

    #[test]
    fn cursor_file_roundtrip_atomic() {
        let dir = std::env::temp_dir().join(format!("atx_cursor_test_{}", std::process::id()));
        let path = dir.join("state").join("updater_cursor.json");
        let mut cs = Cursors::default();
        cs.advance("bars/last/1h/BTCUSDT", 1_762_480_800_000);
        cs.advance("fng", 1_762_473_600_000);
        cs.save(&path).unwrap();
        assert!(!path.with_extension("json.tmp").exists());
        assert_eq!(Cursors::load(&path), cs);
        cs.advance("fng", 1_762_560_000_000);
        cs.save(&path).unwrap();
        assert_eq!(Cursors::load(&path).get("fng"), Some(1_762_560_000_000));
        std::fs::write(&path, "{not json").unwrap();
        assert_eq!(Cursors::load(&path), Cursors::default());
        assert_eq!(Cursors::load(&dir.join("missing.json")), Cursors::default());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn cadence_and_backoff() {
        let c = Cadence { step: 3_600_000, lag: 3_605_000 };
        assert_eq!(c.next_due(None), Ms::MIN);
        // last written bar opened 10:00 -> 11:00 bar closes 12:00, fetchable at 12:00:05
        assert_eq!(c.next_due(Some(36_000_000)), 36_000_000 + 7_205_000);
        assert_eq!(backoff_ms(20_000, 1), 20_000);
        assert_eq!(backoff_ms(20_000, 2), 40_000);
        assert_eq!(backoff_ms(20_000, 30), MAX_BACKOFF_MS);
    }

    #[test]
    fn config_defaults() {
        let cfg = LiveCfg::from_yaml(&Yaml::Null, Path::new("/repo")).unwrap();
        assert_eq!(cfg.symbols.len(), 36);
        assert_eq!(cfg.bar_freqs, vec![Freq::H1, Freq::H4, Freq::D1]);
        assert_eq!(cfg.price_types, vec![PriceType::Last, PriceType::Mark]);
        assert_eq!(cfg.oi_interval.as_deref(), Some("4h"));
        assert_eq!(cfg.dvol, vec!["BTC", "ETH"]);
        assert!(cfg.funding && cfg.fng);
        assert_eq!((cfg.poll_secs, cfg.bar_grace_secs, cfg.backfill_days, cfg.funding_poll_secs), (20, 5, 3, 600));
        assert_eq!(cfg.cursor_file, Path::new("/repo").join("data/state/updater_cursor.json"));
    }

    #[test]
    fn config_parses_live_section() {
        let y: Yaml = serde_yaml::from_str(
            "symbols: [btcusdt, ETHUSDT, BTCUSDT]\nbar_freqs: [1h, 1d]\nprice_types: [last]\nfunding: false\nopen_interest: {interval: 1h}\ndvol: [BTC]\nfng: false\npoll_secs: 30\nbar_grace_secs: 8\n",
        )
        .unwrap();
        let cfg = LiveCfg::from_yaml(&y, Path::new(".")).unwrap();
        assert_eq!(cfg.symbols, vec!["BTCUSDT", "ETHUSDT"]);
        assert_eq!(cfg.bar_freqs, vec![Freq::H1, Freq::D1]);
        assert_eq!(cfg.price_types, vec![PriceType::Last]);
        assert!(!cfg.funding && !cfg.fng);
        assert_eq!(cfg.oi_interval.as_deref(), Some("1h"));
        assert_eq!(cfg.dvol, vec!["BTC"]);
        assert_eq!((cfg.poll_secs, cfg.bar_grace_secs), (30, 8));

        let off: Yaml = serde_yaml::from_str("open_interest: false\ndvol: []\n").unwrap();
        let cfg = LiveCfg::from_yaml(&off, Path::new(".")).unwrap();
        assert!(cfg.oi_interval.is_none() && cfg.dvol.is_empty());

        for bad in ["bar_freqs: [2h]", "price_types: [bid]", "open_interest: {interval: 3h}", "poll_secs: -1", "symbols: []", "funding: maybe"] {
            let y: Yaml = serde_yaml::from_str(bad).unwrap();
            assert!(LiveCfg::from_yaml(&y, Path::new(".")).is_err(), "{bad} should fail");
        }
    }

    #[test]
    fn config_symbols_from_universe_file() {
        let dir = std::env::temp_dir().join(format!("atx_universe_test_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("u.txt"), "# research set\nBTCUSDT ETHUSDT\nsolusdt, XRPUSDT  # comment\n").unwrap();
        let y: Yaml = serde_yaml::from_str("symbols: u.txt").unwrap();
        let cfg = LiveCfg::from_yaml(&y, &dir).unwrap();
        assert_eq!(cfg.symbols, vec!["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT"]);
        let y: Yaml = serde_yaml::from_str("symbols: missing.txt").unwrap();
        assert!(LiveCfg::from_yaml(&y, &dir).is_err());
        let _ = std::fs::remove_dir_all(&dir);
    }
}
