//! Historical trend bars (`ProtoOAGetTrendbarsReq`): decoding, monthly windows and paging.
//!
//! A trend bar carries `low` (1/100000) and unsigned deltas to open, high and close, a tick volume, and its open time
//! in whole minutes. The server caps both the time range of one request (per period, not documented exactly; about
//! five weeks for M1..M5) and the number of bars per response (`hasMore` = the response was truncated, and it is not
//! documented which end is kept). The pager therefore never relies on either: it fetches one calendar month at a time
//! (shorter than every cap), and a window that comes back with `hasMore`, or with a range / boundary error, is split
//! in half and both halves are fetched again, whichever end the server kept. Bars are filtered to `[from, to)` by open
//! time and deduplicated by open time, so pages that overlap at a boundary cannot duplicate a bar.
//!
//! History start: walking backwards month by month, the first month with data is the newest; once some data has been
//! seen, `max_empty` consecutive empty months mean the series has started (the server keeps nothing older).

use chrono::{Datelike, NaiveDate, TimeZone, Utc};

use crate::codec::OaError;
use crate::pb;
use crate::session::Session;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Period {
    M1,
    M5,
    M15,
    M30,
    H1,
    H4,
    D1,
    W1,
    MN1,
}

impl Period {
    pub fn parse(s: &str) -> anyhow::Result<Self> {
        Ok(match s.to_ascii_uppercase().as_str() {
            "M1" => Self::M1,
            "M5" => Self::M5,
            "M15" => Self::M15,
            "M30" => Self::M30,
            "H1" => Self::H1,
            "H4" => Self::H4,
            "D1" => Self::D1,
            "W1" => Self::W1,
            "MN1" => Self::MN1,
            o => anyhow::bail!("period must be one of M1 M5 M15 M30 H1 H4 D1 W1 MN1, got {o:?}"),
        })
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::M1 => "M1",
            Self::M5 => "M5",
            Self::M15 => "M15",
            Self::M30 => "M30",
            Self::H1 => "H1",
            Self::H4 => "H4",
            Self::D1 => "D1",
            Self::W1 => "W1",
            Self::MN1 => "MN1",
        }
    }

    pub fn pb(self) -> i32 {
        use pb::ProtoOaTrendbarPeriod as T;
        (match self {
            Self::M1 => T::M1,
            Self::M5 => T::M5,
            Self::M15 => T::M15,
            Self::M30 => T::M30,
            Self::H1 => T::H1,
            Self::H4 => T::H4,
            Self::D1 => T::D1,
            Self::W1 => T::W1,
            Self::MN1 => T::Mn1,
        }) as i32
    }

    /// Nominal bar length in ms (a month counts as 28 days; only used to stop bisecting).
    pub fn nominal_ms(self) -> i64 {
        const MIN: i64 = 60_000;
        match self {
            Self::M1 => MIN,
            Self::M5 => 5 * MIN,
            Self::M15 => 15 * MIN,
            Self::M30 => 30 * MIN,
            Self::H1 => 60 * MIN,
            Self::H4 => 240 * MIN,
            Self::D1 => 1_440 * MIN,
            Self::W1 => 7 * 1_440 * MIN,
            Self::MN1 => 28 * 1_440 * MIN,
        }
    }
}

/// One bar: open time (ms UTC) and raw prices (1/100000), tick volume.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Bar {
    pub ts_open: i64,
    pub o: i64,
    pub h: i64,
    pub l: i64,
    pub c: i64,
    pub volume: i64,
}

/// Decode one trend bar; `None` if it has no low or no timestamp (never seen; skipped rather than guessed).
pub fn decode(t: &pb::ProtoOaTrendbar) -> Option<Bar> {
    let l = t.low?;
    let ts = t.utc_timestamp_in_minutes? as i64 * 60_000;
    let d = |x: Option<u64>| l + x.unwrap_or(0) as i64;
    Some(Bar { ts_open: ts, o: d(t.delta_open), h: d(t.delta_high), l, c: d(t.delta_close), volume: t.volume })
}

/// Encode a bar the way the server does (used by tests and the fake server).
pub fn encode(b: &Bar, period: Period) -> pb::ProtoOaTrendbar {
    pb::ProtoOaTrendbar {
        volume: b.volume,
        period: Some(period.pb()),
        low: Some(b.l),
        delta_open: Some((b.o - b.l) as u64),
        delta_close: Some((b.c - b.l) as u64),
        delta_high: Some((b.h - b.l) as u64),
        utc_timestamp_in_minutes: Some((b.ts_open / 60_000) as u32),
    }
}

/// One calendar month (UTC) `[start, end)`, and the part `[a, b)` of it inside the requested range.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Month {
    pub name: String,
    pub start: i64,
    pub end: i64,
    pub a: i64,
    pub b: i64,
}

impl Month {
    /// Clipped by the requested range, or not over yet at `now_ms`: its file is provisional and refetched next run.
    pub fn is_partial(&self, now_ms: i64) -> bool {
        self.a > self.start || self.b < self.end || self.end > now_ms
    }
}

/// Calendar months (UTC) overlapping `[from_ms, to_ms)`, newest first.
pub fn months_desc(from_ms: i64, to_ms: i64) -> Vec<Month> {
    let mut out = Vec::new();
    if from_ms >= to_ms {
        return out;
    }
    let ms = |y: i32, m: u32| NaiveDate::from_ymd_opt(y, m, 1).expect("valid date").and_hms_opt(0, 0, 0).expect("midnight").and_utc().timestamp_millis();
    let last = Utc.timestamp_millis_opt(to_ms - 1).single().expect("valid ms").date_naive();
    let (mut y, mut m) = (last.year(), last.month());
    loop {
        let start = ms(y, m);
        let end = if m == 12 { ms(y + 1, 1) } else { ms(y, m + 1) };
        out.push(Month { name: format!("{y:04}-{m:02}"), start, end, a: start.max(from_ms), b: end.min(to_ms) });
        if start <= from_ms {
            return out;
        }
        (y, m) = if m == 1 { (y - 1, 12) } else { (y, m - 1) };
    }
}

/// Error codes that mean "this window is too long / malformed for the period": split the window and retry.
fn is_range_error(e: &anyhow::Error) -> bool {
    e.downcast_ref::<OaError>().is_some_and(|oe| {
        let c = oe.code.to_ascii_uppercase();
        let d = oe.description.to_ascii_uppercase();
        c.contains("BOUNDAR") || c.contains("INVALID_REQUEST") || c.contains("INCORRECT") || d.contains("PERIOD") || d.contains("RANGE")
    })
}

/// All bars with open time in `[from_ms, to_ms)`, chronological, unique by open time. Splits a window in half on
/// `hasMore` or on a range error until it is a single bar long; a single-bar window that is still truncated is an error.
pub async fn fetch_window(sess: &mut Session, account: i64, symbol_id: i64, period: Period, from_ms: i64, to_ms: i64, requests: &mut usize) -> anyhow::Result<Vec<Bar>> {
    let mut out: Vec<Bar> = Vec::new();
    let mut stack = vec![(from_ms, to_ms)];
    while let Some((a, b)) = stack.pop() {
        if a >= b {
            continue;
        }
        let req = pb::ProtoOaGetTrendbarsReq {
            payload_type: None,
            ctid_trader_account_id: account,
            from_timestamp: Some(a),
            to_timestamp: Some(b),
            period: period.pb(),
            symbol_id,
            count: None,
        };
        *requests += 1;
        anyhow::ensure!(*requests < 200_000, "trend bar paging did not terminate");
        let splittable = b - a > period.nominal_ms();
        let mid = a + (b - a) / 2;
        let res = match sess.request(&req).await {
            Ok(r) => r,
            Err(e) if splittable && is_range_error(&e) => {
                tracing::debug!("trend bars [{a}, {b}): {e}; splitting the window");
                stack.push((a, mid));
                stack.push((mid, b));
                continue;
            }
            Err(e) => return Err(e),
        };
        if res.has_more.unwrap_or(false) {
            anyhow::ensure!(splittable, "trend bars [{a}, {b}) are still truncated (hasMore) at one bar per window");
            stack.push((a, mid));
            stack.push((mid, b));
            continue;
        }
        out.extend(res.trendbar.iter().filter_map(decode).filter(|x| x.ts_open >= a && x.ts_open < b));
    }
    out.sort_by_key(|x| x.ts_open);
    out.dedup_by_key(|x| x.ts_open);
    Ok(out)
}

/// What `fetch_history` did.
#[derive(Clone, Debug, Default, serde::Serialize)]
pub struct HistoryReport {
    pub months_fetched: usize,
    pub months_skipped: usize,
    pub requests: usize,
    pub n_bars_fetched: usize,
    /// Oldest / newest bar open time seen (fetched or already on disk), ms UTC.
    pub earliest_ts: Option<i64>,
    pub latest_ts: Option<i64>,
    /// `history_start` (max_empty empty months after data), or `from_reached`.
    pub stopped: String,
    pub oldest_month_checked: Option<String>,
}

/// A month already on disk: its row count and oldest / newest bar (None when empty).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct OnDisk {
    pub n: usize,
    pub first_ts: Option<i64>,
    pub last_ts: Option<i64>,
}

/// Walk calendar months backwards from `to_ms` to `from_ms`. A month whose final (non-partial) file exists
/// (`have(month)` returns it) is skipped; every other month is fetched with [`fetch_window`] and handed to `sink`
/// with its partial flag. Stops early once data has been seen and `max_empty` consecutive months are empty.
#[allow(clippy::too_many_arguments)]
pub async fn fetch_history(
    sess: &mut Session,
    account: i64,
    symbol_id: i64,
    period: Period,
    from_ms: i64,
    to_ms: i64,
    now_ms: i64,
    max_empty: usize,
    mut have: impl FnMut(&Month) -> Option<OnDisk>,
    mut sink: impl FnMut(&Month, &[Bar], bool) -> anyhow::Result<()>,
) -> anyhow::Result<HistoryReport> {
    let mut r = HistoryReport { stopped: "from_reached".into(), ..Default::default() };
    let (mut seen, mut empty_run) = (false, 0usize);
    let note = |r: &mut HistoryReport, first: Option<i64>, last: Option<i64>| {
        if let Some(f) = first {
            r.earliest_ts = Some(r.earliest_ts.map_or(f, |e| e.min(f)));
        }
        if let Some(l) = last {
            r.latest_ts = Some(r.latest_ts.map_or(l, |e| e.max(l)));
        }
    };
    for m in months_desc(from_ms, to_ms) {
        r.oldest_month_checked = Some(m.name.clone());
        let partial = m.is_partial(now_ms);
        let n = match (partial, have(&m)) {
            (false, Some(d)) => {
                r.months_skipped += 1;
                note(&mut r, d.first_ts, d.last_ts);
                d.n
            }
            _ => {
                let bars = fetch_window(sess, account, symbol_id, period, m.a, m.b, &mut r.requests).await?;
                r.months_fetched += 1;
                r.n_bars_fetched += bars.len();
                note(&mut r, bars.first().map(|b| b.ts_open), bars.last().map(|b| b.ts_open));
                sink(&m, &bars, partial)?;
                bars.len()
            }
        };
        if n > 0 {
            seen = true;
            empty_run = 0;
        } else if seen {
            empty_run += 1;
            if empty_run >= max_empty.max(1) {
                r.stopped = "history_start".into();
                break;
            }
        }
    }
    Ok(r)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decode_roundtrip() {
        let b = Bar { ts_open: 28_333_335 * 60_000, o: 108_123, h: 108_200, l: 108_000, c: 108_150, volume: 42 };
        assert_eq!(decode(&encode(&b, Period::M5)), Some(b));
        let mut t = encode(&b, Period::M5);
        t.low = None;
        assert_eq!(decode(&t), None);
    }

    #[test]
    fn period_parse() {
        for p in ["M1", "m5", "M15", "M30", "H1", "H4", "D1", "W1", "MN1"] {
            assert_eq!(Period::parse(p).unwrap().as_str(), p.to_ascii_uppercase());
        }
        assert!(Period::parse("M2").is_err());
        assert_eq!(Period::MN1.pb(), pb::ProtoOaTrendbarPeriod::Mn1 as i32);
    }

    #[test]
    fn months_newest_first_and_clipped() {
        let ms = |s: &str| chrono::DateTime::parse_from_rfc3339(s).unwrap().timestamp_millis();
        let m = months_desc(ms("2024-11-15T00:00:00Z"), ms("2025-01-10T00:00:00Z"));
        let names: Vec<&str> = m.iter().map(|x| x.name.as_str()).collect();
        assert_eq!(names, ["2025-01", "2024-12", "2024-11"]);
        assert_eq!((m[0].a, m[0].b), (ms("2025-01-01T00:00:00Z"), ms("2025-01-10T00:00:00Z")));
        assert_eq!(m[0].end, ms("2025-02-01T00:00:00Z"));
        assert_eq!((m[2].a, m[2].b), (ms("2024-11-15T00:00:00Z"), ms("2024-12-01T00:00:00Z")));
        let now = ms("2026-01-01T00:00:00Z");
        assert!(m[0].is_partial(now) && !m[1].is_partial(now) && m[2].is_partial(now));
        assert!(m[1].is_partial(ms("2024-12-31T23:00:00Z")), "a month not over yet is partial");
        assert_eq!(months_desc(ms("2025-01-01T00:00:00Z"), ms("2025-02-01T00:00:00Z")).len(), 1);
        assert!(months_desc(5, 5).is_empty());
    }
}
