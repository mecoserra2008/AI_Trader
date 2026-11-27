//! Live market-data updater (`atx update`): keeps bars, funding, open interest, DVOL and
//! Fear & Greed current in the lake between manual backfills. Config: `live:` in settings.yaml.
//! Since 2026-11-27 (D8) it also runs the feeds that used to be one-off backfills and went stale on
//! 2026-11-07..10: the daily instrument snapshot, DefiLlama stablecoins, Deribit perp funding and the
//! `live.extra` streams (open interest at other intervals, long/short ratio, index bars).
//!
//! Each stream (e.g. `bars/last/1h/BTCUSDT`) has a cursor in `data/state/updater_cursor.json`:
//! the event_ts of the newest row written. A stream is polled when its next item should exist
//! (cursor + step + availability lag); only rows strictly newer than the cursor are written, and
//! the cursor advances only after the lake write succeeded. New rows of one cycle are merged into
//! one write per table (one Parquet file per partition, not one per stream). Without a cursor a stream resumes
//! from the newest row already in the lake for that stream (no duplicate rows, no hole after a long stop), else
//! it starts at now - `backfill_days`.
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

/// The daily instrument snapshot is taken at 00:05 UTC.
const INSTRUMENTS_AT: Ms = 5 * 60 * SECOND;
const HOUR: Ms = 3_600 * SECOND;

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
    /// Daily linear instrument snapshot (survivorship, WS09).
    pub instruments: bool,
    /// DefiLlama total stablecoin supply, daily.
    pub stablecoins: bool,
    /// Deribit perpetual funding (hourly) for these currencies.
    pub deribit_funding: Vec<String>,
    /// Extra per-symbol streams beyond the per-symbol defaults.
    pub extra: Vec<Extra>,
}

/// One `live.extra` entry: a stream kind at a given resolution for a list of symbols.
#[derive(Clone, Debug, PartialEq)]
pub enum Extra {
    Oi { interval: String, symbols: Vec<String> },
    Lsr { period: String, symbols: Vec<String> },
    Bars { pt: PriceType, freq: Freq, symbols: Vec<String> },
}

fn parse_extra(v: &Yaml) -> anyhow::Result<Vec<Extra>> {
    let xs = match v {
        Yaml::Null => return Ok(Vec::new()),
        Yaml::Sequence(xs) => xs,
        _ => bail!("live.extra: expected a list of {{kind: ..., symbols: [...]}}"),
    };
    let mut out = Vec::new();
    for x in xs {
        let symbols = dedup_upper(str_list(x.get("symbols").unwrap_or(&Yaml::Null), "extra.symbols")?);
        anyhow::ensure!(!symbols.is_empty(), "live.extra: every entry needs symbols");
        let field = |k: &str| x.get(k).and_then(Yaml::as_str).map(str::to_string);
        out.push(match field("kind").as_deref() {
            Some("open_interest") => {
                let interval = field("interval").context("live.extra open_interest: interval missing")?;
                period_ms(&interval).context("live.extra open_interest.interval")?;
                Extra::Oi { interval, symbols }
            }
            Some("ls_ratio") => {
                let period = field("period").context("live.extra ls_ratio: period missing")?;
                period_ms(&period).context("live.extra ls_ratio.period")?;
                Extra::Lsr { period, symbols }
            }
            Some("bars") => Extra::Bars {
                pt: PriceType::parse(&field("price_type").unwrap_or_else(|| "last".into()))?,
                freq: Freq::parse(&field("freq").context("live.extra bars: freq missing")?)?,
                symbols,
            },
            k => bail!("live.extra: unknown kind {k:?} (open_interest | ls_ratio | bars)"),
        });
    }
    Ok(out)
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
            instruments: bool_or("instruments", false)?,
            stablecoins: bool_or("stablecoins", false)?,
            deribit_funding: str_list(get("deribit_funding"), "deribit_funding")?.into_iter().map(|s| s.to_uppercase()).collect(),
            extra: parse_extra(get("extra"))?,
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
    Lsr { sym: String, period: String, period_ms: Ms },
    Dvol { cur: String },
    Fng,
    Instruments,
    Stablecoins,
    DeribitFunding { cur: String },
}

impl Stream {
    fn key(&self) -> String {
        match self {
            Stream::Bars { sym, freq, pt } => format!("bars/{}/{freq}/{sym}", pt.as_str()),
            Stream::Funding { sym } => format!("funding/{sym}"),
            Stream::Oi { sym, interval, .. } => format!("open_interest/{interval}/{sym}"),
            Stream::Lsr { sym, period, .. } => format!("ls_ratio/{period}/{sym}"),
            Stream::Dvol { cur } => format!("dvol/{cur}"),
            Stream::Fng => "fng".to_string(),
            Stream::Instruments => "instruments".to_string(),
            Stream::Stablecoins => "stablecoins".to_string(),
            Stream::DeribitFunding { cur } => format!("deribit_funding/{cur}"),
        }
    }

    /// Where this stream's rows live in the lake: partition path under the lake root and the row
    /// filters (file columns) that select it. Used to resume a stream that has no cursor.
    fn lake_location(&self) -> (Vec<String>, Vec<(&'static str, String)>) {
        let v = |s: &str| s.to_string();
        match self {
            Stream::Bars { sym, freq, pt } => (vec![v("bars"), v("venue=bybit"), format!("price_type={}", pt.as_str()), format!("freq={freq}")], vec![("symbol", sym.clone())]),
            Stream::Funding { sym } => (vec![v("funding"), v("venue=bybit")], vec![("symbol", sym.clone())]),
            Stream::Oi { sym, interval, .. } => (vec![v("open_interest"), v("venue=bybit"), format!("interval={interval}")], vec![("symbol", sym.clone())]),
            Stream::Lsr { sym, period, .. } => (vec![v("ls_ratio"), v("venue=bybit"), format!("period={period}")], vec![("symbol", sym.clone())]),
            Stream::Dvol { cur } => (vec![v("alt_series"), format!("source={}", alt::deribit::SOURCE)], vec![("series_id", v("dvol_close")), ("entity", cur.clone())]),
            Stream::Fng => (vec![v("alt_series"), format!("source={}", alt::fng::SOURCE)], vec![]),
            Stream::Instruments => (vec![v("instruments"), v("venue=bybit")], vec![]),
            Stream::Stablecoins => (vec![v("alt_series"), format!("source={}", alt::defillama::SOURCE)], vec![("series_id", v("stablecoins_total_usd"))]),
            Stream::DeribitFunding { cur } => (vec![v("alt_series"), format!("source={}", alt::deribit::SOURCE)], vec![("series_id", v("perp_funding_1h")), ("entity", cur.clone())]),
        }
    }
}

// ---------------------------------------------------------------- resume from the lake

fn col_as(rb: &arrow::record_batch::RecordBatch, name: &str, ty: &arrow::datatypes::DataType) -> Option<arrow::array::ArrayRef> {
    rb.column_by_name(name).and_then(|a| arrow::compute::cast(a, ty).ok())
}

/// Newest `event_ts` among the rows of one Parquet file that match `filters` (compared as strings).
fn file_newest(path: &Path, filters: &[(&str, String)]) -> anyhow::Result<Option<Ms>> {
    use arrow::array::{Array, Int64Array, StringArray};
    use arrow::datatypes::DataType;
    let f = std::fs::File::open(path)?;
    let reader = parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder::try_new(f)?.build()?;
    let mut best: Option<Ms> = None;
    for rb in reader {
        let rb = rb?;
        let ev = col_as(&rb, "event_ts", &DataType::Int64).context("event_ts")?;
        let ev = ev.as_any().downcast_ref::<Int64Array>().context("event_ts as i64")?;
        let mut cols = Vec::new();
        for (c, want) in filters {
            let Some(a) = col_as(&rb, c, &DataType::Utf8) else { return Ok(None) };
            cols.push((a, want));
        }
        for i in 0..rb.num_rows() {
            let keep = cols.iter().all(|(a, want)| {
                a.as_any().downcast_ref::<StringArray>().is_some_and(|c| !c.is_null(i) && c.value(i) == want.as_str())
            });
            if keep && !ev.is_null(i) {
                best = Some(best.map_or(ev.value(i), |b: Ms| b.max(ev.value(i))));
            }
        }
    }
    Ok(best)
}

/// Newest `event_ts` of a stream already in the lake. Year partitions are scanned newest first; the
/// first one with a matching row decides. `None` when the stream has no rows.
pub fn lake_newest(root: &Path, parts: &[String], filters: &[(&str, String)]) -> anyhow::Result<Option<Ms>> {
    let dir = parts.iter().fold(root.to_path_buf(), |d, p| d.join(p));
    let Ok(rd) = std::fs::read_dir(&dir) else { return Ok(None) };
    let parquet_in = |d: &Path| -> Vec<PathBuf> {
        std::fs::read_dir(d).map(|rd| rd.filter_map(|e| e.ok()).map(|e| e.path()).filter(|p| p.extension().is_some_and(|x| x == "parquet")).collect()).unwrap_or_default()
    };
    let mut years: Vec<PathBuf> = rd.filter_map(|e| e.ok()).map(|e| e.path()).filter(|p| p.is_dir() && p.file_name().is_some_and(|n| n.to_string_lossy().starts_with("year="))).collect();
    years.sort();
    let groups: Vec<Vec<PathBuf>> = if years.is_empty() { vec![parquet_in(&dir)] } else { years.iter().rev().map(|y| parquet_in(y)).collect() };
    for g in groups {
        let mut best: Option<Ms> = None;
        for f in g {
            if let Some(t) = file_newest(&f, filters).with_context(|| f.display().to_string())? {
                best = Some(best.map_or(t, |b| b.max(t)));
            }
        }
        if best.is_some() {
            return Ok(best);
        }
    }
    Ok(None)
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
            Stream::Lsr { period_ms, .. } => Some(Cadence { step: *period_ms, lag: *period_ms + grace }),
            Stream::Stablecoins => Some(Cadence { step: DAY, lag: self.ctx.alt_latency("defillama_stablecoins") + grace }),
            Stream::DeribitFunding { .. } => Some(Cadence { step: HOUR, lag: HOUR + self.ctx.alt_latency("deribit_funding") + grace }),
            // Daily snapshot at 00:05 UTC: see `due_at`.
            Stream::Instruments => Some(Cadence { step: DAY, lag: 0 }),
            Stream::Funding { .. } => None,
        }
    }

    /// When the stream's next item is due (None = polled on a timer, i.e. funding).
    fn due_at(&self, s: &Stream, cursor: Option<Ms>) -> Option<Ms> {
        match s {
            Stream::Instruments => Some(instruments_due(cursor)),
            _ => self.cadence(s).map(|c| c.next_due(cursor)),
        }
    }

    fn is_due(&self, s: &Stream, key: &str, now: Ms) -> bool {
        if self.retry.get(key).is_some_and(|r| now < r.not_before) {
            return false;
        }
        match self.due_at(s, self.cursors.get(key)) {
            Some(t) => now >= t,
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
            Stream::Lsr { sym, period, period_ms } => {
                // Complete periods only, as for open interest.
                let end = (now - grace) - (now - grace).rem_euclid(*period_ms);
                if start >= end {
                    return Ok(None);
                }
                let xs = self.md.long_short_ratio(sym, period, start, end).await?;
                let (rows, newest) = plan_write(cursor, start, xs, |x| x.ts);
                ("ls_ratio", bl::ls_ratio(sym, period, *period_ms, &rows, ctx.latency("ls_ratio")), newest)
            }
            Stream::Instruments => {
                let xs = self.md.instruments().await?;
                let (rows, newest) = plan_write(cursor, start, xs, |_| now);
                ("instruments", bl::instruments(ctx.category(), &rows, now), newest)
            }
            Stream::Stablecoins => {
                let xs = alt::defillama::stablecoins_total(&self.http, ctx.alt_latency("defillama_stablecoins")).await?;
                let (rows, newest) = plan_write(cursor, start, xs, |o| o.event_ts);
                ("alt_series", alt::to_batch(alt::defillama::SOURCE, alt::defillama::PIT, &rows), newest)
            }
            Stream::DeribitFunding { cur } => {
                let xs = alt::deribit::perp_funding(&self.http, cur, start, now, ctx.alt_latency("deribit_funding")).await?;
                let (rows, newest) = plan_write(cursor, start, xs, |o| o.event_ts);
                ("alt_series", alt::to_batch(alt::deribit::SOURCE, alt::deribit::PIT, &rows), newest)
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
            let key = s.key();
            let Some(due) = self.due_at(s, self.cursors.get(&key)) else { continue };
            match self.cursors.get(&key) {
                None => uninit += 1,
                Some(_) => {
                    let over = now - due;
                    if over > worst.0 {
                        worst = (over, key);
                    }
                }
            }
        }
        (worst.0, worst.1, uninit)
    }
}

/// 00:05 UTC of the day after the last snapshot (due at once without one).
pub fn instruments_due(cursor: Option<Ms>) -> Ms {
    cursor.map_or(Ms::MIN, |c| c - c.rem_euclid(DAY) + DAY + INSTRUMENTS_AT)
}

/// Streams from `live.instruments`, `live.stablecoins`, `live.deribit_funding` and `live.extra`, without
/// duplicating a per-symbol default stream (same key).
fn extra_streams(cfg: &LiveCfg) -> anyhow::Result<Vec<Stream>> {
    let mut out = Vec::new();
    if cfg.instruments {
        out.push(Stream::Instruments);
    }
    if cfg.stablecoins {
        out.push(Stream::Stablecoins);
    }
    out.extend(cfg.deribit_funding.iter().map(|c| Stream::DeribitFunding { cur: c.clone() }));
    for x in &cfg.extra {
        match x {
            Extra::Oi { interval, symbols } => {
                let period = period_ms(interval)?;
                for s in symbols {
                    if cfg.oi_interval.as_deref() == Some(interval.as_str()) && cfg.symbols.contains(s) {
                        continue;
                    }
                    out.push(Stream::Oi { sym: s.clone(), interval: interval.clone(), period });
                }
            }
            Extra::Lsr { period, symbols } => {
                let pm = period_ms(period)?;
                out.extend(symbols.iter().map(|s| Stream::Lsr { sym: s.clone(), period: period.clone(), period_ms: pm }));
            }
            Extra::Bars { pt, freq, symbols } => {
                for s in symbols {
                    if cfg.symbols.contains(s) && cfg.price_types.contains(pt) && cfg.bar_freqs.contains(freq) {
                        continue;
                    }
                    out.push(Stream::Bars { sym: s.clone(), freq: *freq, pt: *pt });
                }
            }
        }
    }
    Ok(out)
}

/// Give every stream without a cursor the newest event_ts already in the lake. Returns how many were seeded.
fn seed_cursors(cursors: &mut Cursors, streams: &[Stream], root: &Path) -> usize {
    let mut n = 0;
    for s in streams {
        let key = s.key();
        if cursors.get(&key).is_some() {
            continue;
        }
        let (parts, filters) = s.lake_location();
        match lake_newest(root, &parts, &filters) {
            Ok(Some(t)) => {
                cursors.advance(&key, t);
                tracing::info!(stream = %key, newest = %to_datetime(t).to_rfc3339(), "resuming from the lake");
                n += 1;
            }
            Ok(None) => {}
            Err(e) => tracing::warn!(stream = %key, "could not read the lake to resume ({e:#}); starting from backfill_days"),
        }
    }
    n
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
    streams.extend(extra_streams(&cfg)?);
    let mut cursors = Cursors::load(&cfg.cursor_file);
    let seeded = seed_cursors(&mut cursors, &streams, ctx.lake.root());
    if seeded > 0 {
        tracing::info!("{seeded} streams without a cursor resume from the newest lake row");
        if let Err(e) = cursors.save(&cfg.cursor_file) {
            tracing::error!("saving seeded cursors failed: {e:#}");
        }
    }
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
    fn config_new_feeds_and_extra_streams() {
        let cfg = LiveCfg::from_yaml(&Yaml::Null, Path::new(".")).unwrap();
        assert!(!cfg.instruments && !cfg.stablecoins && cfg.deribit_funding.is_empty() && cfg.extra.is_empty(), "off unless configured");
        let y: Yaml = serde_yaml::from_str(
            "symbols: [BTCUSDT, ETHUSDT]\ninstruments: true\nstablecoins: true\nderibit_funding: [btc, ETH]\nextra:\n  - {kind: open_interest, interval: 1h, symbols: [BTCUSDT]}\n  - {kind: open_interest, interval: 4h, symbols: [BTCUSDT, SOLUSDT]}\n  - {kind: ls_ratio, period: 4h, symbols: [btcusdt, ETHUSDT]}\n  - {kind: bars, price_type: index, freq: 1h, symbols: [BTCUSDT]}\n  - {kind: bars, price_type: last, freq: 1h, symbols: [BTCUSDT]}\n",
        )
        .unwrap();
        let cfg = LiveCfg::from_yaml(&y, Path::new(".")).unwrap();
        assert!(cfg.instruments && cfg.stablecoins);
        assert_eq!(cfg.deribit_funding, vec!["BTC", "ETH"]);
        assert_eq!(cfg.extra[2], Extra::Lsr { period: "4h".into(), symbols: vec!["BTCUSDT".into(), "ETHUSDT".into()] });
        let keys: Vec<String> = extra_streams(&cfg).unwrap().iter().map(Stream::key).collect();
        assert_eq!(
            keys,
            vec![
                "instruments",
                "stablecoins",
                "deribit_funding/BTC",
                "deribit_funding/ETH",
                "open_interest/1h/BTCUSDT",
                // BTCUSDT 4h OI is already a default stream; SOLUSDT is not in live.symbols
                "open_interest/4h/SOLUSDT",
                "ls_ratio/4h/BTCUSDT",
                "ls_ratio/4h/ETHUSDT",
                // last 1h BTCUSDT is a default stream
                "bars/index/1h/BTCUSDT",
            ]
        );
        for bad in ["extra: [{kind: oi, symbols: [BTCUSDT]}]", "extra: [{kind: ls_ratio, period: 3h, symbols: [BTCUSDT]}]", "extra: [{kind: bars, freq: 1h}]", "extra: {kind: bars}"] {
            let y: Yaml = serde_yaml::from_str(bad).unwrap();
            assert!(LiveCfg::from_yaml(&y, Path::new(".")).is_err(), "{bad} should fail");
        }
    }

    #[test]
    fn instruments_snapshot_is_daily_at_0005() {
        assert_eq!(instruments_due(None), Ms::MIN);
        let day = 1_795_737_600_000; // 2026-11-27 00:00 UTC
        assert_eq!(instruments_due(Some(day + 21 * 3_600_000)), day + DAY + INSTRUMENTS_AT);
        assert_eq!(instruments_due(Some(day + INSTRUMENTS_AT + 1_234)), day + DAY + INSTRUMENTS_AT);
    }

    #[test]
    fn lake_newest_resumes_each_stream_from_its_own_rows() {
        use at_bybit::market::LongShortRatio;
        let dir = std::env::temp_dir().join(format!("atx_lake_newest_{}", std::process::id()));
        let schema = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../config/schema.yaml");
        let lake = at_store::Lake::open(&dir, &schema).unwrap();
        let h = 3_600_000;
        let t0 = 1_762_473_600_000; // 2025-11-07
        let t1 = 1_794_009_600_000; // 2026-11-07
        let rows = |t: Ms| vec![LongShortRatio { ts: t, buy_ratio: 0.6, sell_ratio: 0.4 }, LongShortRatio { ts: t + 4 * h, buy_ratio: 0.5, sell_ratio: 0.5 }];
        lake.write("ls_ratio", &bl::ls_ratio("BTCUSDT", "4h", 4 * h, &rows(t0), 2_000)).unwrap();
        lake.write("ls_ratio", &bl::ls_ratio("BTCUSDT", "4h", 4 * h, &rows(t1), 2_000)).unwrap();
        lake.write("ls_ratio", &bl::ls_ratio("ETHUSDT", "4h", 4 * h, &rows(t0), 2_000)).unwrap();
        let obs = |series: &str, ent: &str, t: Ms| alt::Obs { series_id: series.into(), entity: ent.into(), event_ts: t, value: Some(1.0), vintage: None, known_ts: t + h };
        lake.write("alt_series", &alt::to_batch("deribit", alt::PitQuality::Native, &[obs("perp_funding_1h", "BTC", t1), obs("perp_funding_8h", "BTC", t1 + h), obs("dvol_close", "BTC", t1 + 9 * h)])).unwrap();
        let s = |sym: &str| Stream::Lsr { sym: sym.into(), period: "4h".into(), period_ms: 4 * h };
        let newest = |st: &Stream| {
            let (p, f) = st.lake_location();
            lake_newest(&dir, &p, &f).unwrap()
        };
        assert_eq!(newest(&s("BTCUSDT")), Some(t1 + 4 * h), "newest year partition wins");
        assert_eq!(newest(&s("ETHUSDT")), Some(t0 + 4 * h), "falls back to older years");
        assert_eq!(newest(&s("SOLUSDT")), None);
        assert_eq!(newest(&Stream::DeribitFunding { cur: "BTC".into() }), Some(t1), "series and entity filters");
        assert_eq!(newest(&Stream::DeribitFunding { cur: "ETH".into() }), None);
        assert_eq!(newest(&Stream::Dvol { cur: "BTC".into() }), Some(t1 + 9 * h));
        let mut cs = Cursors::default();
        cs.advance("ls_ratio/4h/ETHUSDT", t1);
        assert_eq!(seed_cursors(&mut cs, &[s("BTCUSDT"), s("ETHUSDT"), s("SOLUSDT")], &dir), 1);
        assert_eq!(cs.get("ls_ratio/4h/BTCUSDT"), Some(t1 + 4 * h));
        assert_eq!(cs.get("ls_ratio/4h/ETHUSDT"), Some(t1), "an existing cursor is never replaced");
        assert_eq!(cs.get("ls_ratio/4h/SOLUSDT"), None);
        let _ = std::fs::remove_dir_all(&dir);
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
