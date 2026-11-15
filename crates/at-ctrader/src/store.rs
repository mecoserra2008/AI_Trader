//! Compact research files (never the lake: config/schema.yaml has no cTrader tables).
//!
//! `data/research/xvenue/ctrader_ticks/<symbol>/<YYYY-MM-DD>_<bid|ask>.parquet`, one file per UTC day and side,
//! columns `event_ts` (timestamp ms UTC, the broker's tick time), `price` (f64), `known_ts` (when we fetched
//! it; constant per file), zstd level 3 as at-store.
//!
//! Trend bars: `<out>/<symbol>/<period>/<YYYY-MM>.parquet` (a month still in progress at fetch time is written as
//! `<YYYY-MM>.partial.parquet` and refetched on the next run), columns `ts_open` (timestamp ms UTC, the bar's open),
//! `o, h, l, c` (f64), `tick_volume` (i64, ticks, not traded volume), `known_ts`.

use std::fs::File;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use arrow::array::{ArrayRef, Float64Array, Int64Array, TimestampMillisecondArray};
use arrow::datatypes::{DataType, Field, Schema, TimeUnit};
use arrow::record_batch::RecordBatch;
use parquet::arrow::ArrowWriter;
use parquet::basic::{Compression, ZstdLevel};
use parquet::file::properties::WriterProperties;

use crate::ticks::{Side, Tick};
use crate::trendbars::{Bar, Period};

pub const DAY_MS: i64 = 86_400_000;

/// File-system-safe symbol directory name (cTrader names can contain '/', ' ' or '#').
pub fn safe_name(sym: &str) -> String {
    sym.chars().map(|c| if c.is_ascii_alphanumeric() || matches!(c, '.' | '-' | '_') { c } else { '_' }).collect()
}

pub fn tick_path(root: &Path, symbol: &str, date: &str, side: Side) -> PathBuf {
    root.join(safe_name(symbol)).join(format!("{date}_{}.parquet", side.as_str()))
}

/// Write one day's ticks (already filtered to the day). Atomic via a .tmp rename.
pub fn write_ticks(path: &Path, ticks: &[Tick], known_ts: i64) -> anyhow::Result<()> {
    if let Some(d) = path.parent() {
        std::fs::create_dir_all(d)?;
    }
    let tz: Arc<str> = "UTC".into();
    let schema = Arc::new(Schema::new(vec![
        Field::new("event_ts", DataType::Timestamp(TimeUnit::Millisecond, Some(tz.clone())), false),
        Field::new("price", DataType::Float64, false),
        Field::new("known_ts", DataType::Timestamp(TimeUnit::Millisecond, Some(tz)), false),
    ]));
    let cols: Vec<ArrayRef> = vec![
        Arc::new(TimestampMillisecondArray::from(ticks.iter().map(|t| t.ts).collect::<Vec<_>>()).with_timezone("UTC")),
        Arc::new(Float64Array::from(ticks.iter().map(|t| crate::price(t.raw)).collect::<Vec<_>>())),
        Arc::new(TimestampMillisecondArray::from(vec![known_ts; ticks.len()]).with_timezone("UTC")),
    ];
    let rb = RecordBatch::try_new(schema.clone(), cols)?;
    let props = WriterProperties::builder().set_compression(Compression::ZSTD(ZstdLevel::try_new(3)?)).build();
    let tmp = path.with_extension("parquet.tmp");
    {
        let mut w = ArrowWriter::try_new(File::create(&tmp)?, schema, Some(props))?;
        w.write(&rb)?;
        w.close()?;
    }
    std::fs::rename(&tmp, path)?;
    Ok(())
}

pub fn trendbar_path(root: &Path, symbol: &str, period: Period, month: &str, partial: bool) -> PathBuf {
    root.join(safe_name(symbol)).join(period.as_str()).join(if partial { format!("{month}.partial.parquet") } else { format!("{month}.parquet") })
}

/// Write one month of trend bars (chronological). Atomic via a .tmp rename. An empty month writes a file with no rows,
/// so a rerun knows the month was fetched.
pub fn write_trendbars(path: &Path, bars: &[Bar], known_ts: i64) -> anyhow::Result<()> {
    if let Some(d) = path.parent() {
        std::fs::create_dir_all(d)?;
    }
    let tz: Arc<str> = "UTC".into();
    let ts = |name: &str| Field::new(name, DataType::Timestamp(TimeUnit::Millisecond, Some(tz.clone())), false);
    let f64f = |name: &str| Field::new(name, DataType::Float64, false);
    let schema = Arc::new(Schema::new(vec![ts("ts_open"), f64f("o"), f64f("h"), f64f("l"), f64f("c"), Field::new("tick_volume", DataType::Int64, false), ts("known_ts")]));
    let px = |f: fn(&Bar) -> i64| -> ArrayRef { Arc::new(Float64Array::from(bars.iter().map(|b| crate::price(f(b))).collect::<Vec<_>>())) };
    let cols: Vec<ArrayRef> = vec![
        Arc::new(TimestampMillisecondArray::from(bars.iter().map(|b| b.ts_open).collect::<Vec<_>>()).with_timezone("UTC")),
        px(|b| b.o),
        px(|b| b.h),
        px(|b| b.l),
        px(|b| b.c),
        Arc::new(Int64Array::from(bars.iter().map(|b| b.volume).collect::<Vec<_>>())),
        Arc::new(TimestampMillisecondArray::from(vec![known_ts; bars.len()]).with_timezone("UTC")),
    ];
    let rb = RecordBatch::try_new(schema.clone(), cols)?;
    let props = WriterProperties::builder().set_compression(Compression::ZSTD(ZstdLevel::try_new(3)?)).build();
    let tmp = path.with_extension("parquet.tmp");
    {
        let mut w = ArrowWriter::try_new(File::create(&tmp)?, schema, Some(props))?;
        w.write(&rb)?;
        w.close()?;
    }
    std::fs::rename(&tmp, path)?;
    Ok(())
}

/// UTC day starts `[d0, d0 + DAY)` covering `[from_ms, to_ms)`.
pub fn days(from_ms: i64, to_ms: i64) -> Vec<i64> {
    let mut d = from_ms.div_euclid(DAY_MS) * DAY_MS;
    let mut v = Vec::new();
    while d < to_ms {
        v.push(d);
        d += DAY_MS;
    }
    v
}

#[cfg(test)]
mod tests {
    use super::*;
    use parquet::file::reader::{FileReader, SerializedFileReader};

    #[test]
    fn writes_readable_parquet() {
        let dir = std::env::temp_dir().join(format!("at-ctrader-store-{}", std::process::id()));
        let p = tick_path(&dir, "XAU/USD", "2026-11-10", Side::Bid);
        assert!(p.to_string_lossy().contains("XAU_USD"));
        write_ticks(&p, &[Tick { ts: 1, raw: 123_000 }, Tick { ts: 2, raw: 124_000 }], 10).unwrap();
        let r = SerializedFileReader::new(File::open(&p).unwrap()).unwrap();
        assert_eq!(r.metadata().file_metadata().num_rows(), 2);
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn writes_trendbars() {
        let dir = std::env::temp_dir().join(format!("at-ctrader-bars-{}", std::process::id()));
        let p = trendbar_path(&dir, "EUR/USD", Period::M5, "2024-01", false);
        assert!(p.ends_with("EUR_USD/M5/2024-01.parquet"));
        assert!(trendbar_path(&dir, "X", Period::M5, "2024-01", true).ends_with("X/M5/2024-01.partial.parquet"));
        write_trendbars(&p, &[Bar { ts_open: 60_000, o: 110_000, h: 110_050, l: 109_990, c: 110_010, volume: 7 }], 10).unwrap();
        let r = SerializedFileReader::new(File::open(&p).unwrap()).unwrap();
        assert_eq!(r.metadata().file_metadata().num_rows(), 1);
        assert_eq!(r.metadata().file_metadata().schema_descr().num_columns(), 7);
        write_trendbars(&p, &[], 10).unwrap();
        assert_eq!(SerializedFileReader::new(File::open(&p).unwrap()).unwrap().metadata().file_metadata().num_rows(), 0);
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn day_split() {
        assert_eq!(days(DAY_MS + 5, 3 * DAY_MS), vec![DAY_MS, 2 * DAY_MS]);
        assert_eq!(days(0, 1), vec![0]);
    }
}
