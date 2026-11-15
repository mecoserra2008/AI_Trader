//! Read-only access to the Parquet lake for the paper/replay executor: bars and instrument rules.
//! Revisions are rows with a later `known_ts`; the latest known version of each key wins, and only
//! rows with `known_ts <= as_of` are used (point in time).

use std::collections::{BTreeMap, HashMap, HashSet};
use std::fs::File;
use std::path::{Path, PathBuf};

use anyhow::Context;
use arrow::array::{Array, ArrayRef, Float64Array, Int64Array, StringArray};
use arrow::compute::cast;
use arrow::datatypes::DataType;
use arrow::record_batch::RecordBatch;
use at_core::time::Ms;
use at_core::types::InstrumentRules;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Bar {
    /// Bar open time.
    pub ts: Ms,
    pub close_ts: Ms,
    pub known_ts: Ms,
    pub open: f64,
    pub high: f64,
    pub low: f64,
    pub close: f64,
}

fn parquet_files(dir: &Path) -> Vec<PathBuf> {
    let Ok(rd) = std::fs::read_dir(dir) else { return vec![] };
    let mut v: Vec<PathBuf> = rd.filter_map(|e| e.ok()).map(|e| e.path()).filter(|p| p.extension().is_some_and(|x| x == "parquet")).collect();
    v.sort();
    v
}

fn read_file(path: &Path) -> anyhow::Result<Vec<RecordBatch>> {
    let f = File::open(path).with_context(|| path.display().to_string())?;
    let reader = ParquetRecordBatchReaderBuilder::try_new(f).with_context(|| path.display().to_string())?.build()?;
    reader.collect::<Result<Vec<_>, _>>().with_context(|| path.display().to_string())
}

fn col(rb: &RecordBatch, name: &str, ty: &DataType) -> anyhow::Result<ArrayRef> {
    let a = rb.column_by_name(name).with_context(|| format!("column {name} missing"))?;
    Ok(cast(a, ty)?)
}

fn strings(rb: &RecordBatch, name: &str) -> anyhow::Result<Vec<Option<String>>> {
    let a = col(rb, name, &DataType::Utf8)?;
    let a = a.as_any().downcast_ref::<StringArray>().context("utf8")?;
    Ok((0..a.len()).map(|i| (!a.is_null(i)).then(|| a.value(i).to_string())).collect())
}

fn floats(rb: &RecordBatch, name: &str) -> anyhow::Result<Vec<f64>> {
    let a = col(rb, name, &DataType::Float64)?;
    let a = a.as_any().downcast_ref::<Float64Array>().context("f64")?;
    Ok((0..a.len()).map(|i| if a.is_null(i) { f64::NAN } else { a.value(i) }).collect())
}

fn ints(rb: &RecordBatch, name: &str) -> anyhow::Result<Vec<i64>> {
    let a = col(rb, name, &DataType::Int64)?;
    let a = a.as_any().downcast_ref::<Int64Array>().context("i64")?;
    Ok((0..a.len()).map(|i| if a.is_null(i) { i64::MIN } else { a.value(i) }).collect())
}

/// Bars `bars/venue=V/price_type=P/freq=F/year=Y/*.parquet` for `symbols` with `ts >= from` and
/// `known_ts <= as_of`, deduplicated (latest known revision), sorted by time.
pub fn read_bars(lake: &Path, venue: &str, price_type: &str, freq: &str, symbols: &HashSet<String>, from: Ms, as_of: Ms) -> anyhow::Result<HashMap<String, Vec<Bar>>> {
    let base = lake.join("bars").join(format!("venue={venue}")).join(format!("price_type={price_type}")).join(format!("freq={freq}"));
    let from_year = at_core::time::to_datetime(from).format("%Y").to_string().parse::<i32>().unwrap_or(0);
    let mut best: HashMap<String, BTreeMap<Ms, Bar>> = HashMap::new();
    let Ok(rd) = std::fs::read_dir(&base) else { return Ok(HashMap::new()) };
    let mut years: Vec<(i32, PathBuf)> = rd
        .filter_map(|e| e.ok())
        .filter_map(|e| {
            let n = e.file_name().to_string_lossy().to_string();
            n.strip_prefix("year=").and_then(|y| y.parse::<i32>().ok()).map(|y| (y, e.path()))
        })
        .filter(|(y, _)| *y >= from_year)
        .collect();
    years.sort();
    for (_, dir) in years {
        for f in parquet_files(&dir) {
            for rb in read_file(&f)? {
                let sym = strings(&rb, "symbol")?;
                let ts = ints(&rb, "event_ts")?;
                let cts = ints(&rb, "close_ts")?;
                let kts = ints(&rb, "known_ts")?;
                let (o, h, l, c) = (floats(&rb, "open")?, floats(&rb, "high")?, floats(&rb, "low")?, floats(&rb, "close")?);
                for i in 0..rb.num_rows() {
                    let Some(s) = sym[i].as_ref() else { continue };
                    if !symbols.contains(s) || ts[i] < from || kts[i] > as_of {
                        continue;
                    }
                    let bar = Bar { ts: ts[i], close_ts: cts[i], known_ts: kts[i], open: o[i], high: h[i], low: l[i], close: c[i] };
                    if ![bar.open, bar.high, bar.low, bar.close].iter().all(|x| x.is_finite() && *x > 0.0) {
                        continue;
                    }
                    let m = best.entry(s.clone()).or_default();
                    match m.get(&bar.ts) {
                        Some(prev) if prev.known_ts >= bar.known_ts => {}
                        _ => {
                            m.insert(bar.ts, bar);
                        }
                    }
                }
            }
        }
    }
    Ok(best.into_iter().map(|(k, v)| (k, v.into_values().collect())).collect())
}

/// Latest instrument snapshot per symbol known at `as_of`, for `category`. Also returns the venue
/// `symbol_type` ('' = crypto; ETF | stock | commodity = TradFi).
pub fn read_instruments(lake: &Path, venue: &str, category: &str, as_of: Ms) -> anyhow::Result<(HashMap<String, InstrumentRules>, HashMap<String, String>)> {
    let dir = lake.join("instruments").join(format!("venue={venue}"));
    let mut best: HashMap<String, (Ms, InstrumentRules, String)> = HashMap::new();
    for f in parquet_files(&dir) {
        for rb in read_file(&f)? {
            let sym = strings(&rb, "symbol")?;
            let cat = strings(&rb, "category")?;
            let st = strings(&rb, "symbol_type").unwrap_or_else(|_| vec![None; rb.num_rows()]);
            let kts = ints(&rb, "known_ts")?;
            let (tick, step, minq, maxq, maxm, minn, lev) =
                (floats(&rb, "tick_size")?, floats(&rb, "qty_step")?, floats(&rb, "min_qty")?, floats(&rb, "max_qty")?, floats(&rb, "max_mkt_qty")?, floats(&rb, "min_notional")?, floats(&rb, "max_leverage")?);
            for i in 0..rb.num_rows() {
                let (Some(s), Some(c)) = (sym[i].as_ref(), cat[i].as_ref()) else { continue };
                if c != category || kts[i] > as_of {
                    continue;
                }
                if best.get(s).is_some_and(|(k, _, _)| *k >= kts[i]) {
                    continue;
                }
                let r = InstrumentRules { symbol: s.clone(), tick_size: tick[i], qty_step: step[i], min_qty: minq[i], max_qty: maxq[i], max_mkt_qty: maxm[i], min_notional: minn[i], max_leverage: lev[i] };
                best.insert(s.clone(), (kts[i], r, st[i].clone().unwrap_or_default()));
            }
        }
    }
    let types = best.iter().map(|(k, v)| (k.clone(), v.2.clone())).collect();
    Ok((best.into_iter().map(|(k, v)| (k, v.1)).collect(), types))
}

/// `<dir>/<SYMBOL>.csv` with header `ts_ms,open,high,low,close[,...]` (bar open time in ms).
pub fn read_bars_csv(dir: &Path, symbols: &HashSet<String>, bar_ms: Ms) -> anyhow::Result<HashMap<String, Vec<Bar>>> {
    let mut out = HashMap::new();
    for s in symbols {
        let p = dir.join(format!("{s}.csv"));
        let Ok(text) = std::fs::read_to_string(&p) else { continue };
        let mut lines = text.lines().filter(|l| !l.trim().is_empty());
        let header: Vec<String> = lines.next().unwrap_or_default().split(',').map(|h| h.trim().to_ascii_lowercase()).collect();
        let idx = |n: &str| header.iter().position(|h| h == n).with_context(|| format!("{}: no column {n}", p.display()));
        let (it, io, ih, il, ic) = (idx("ts_ms")?, idx("open")?, idx("high")?, idx("low")?, idx("close")?);
        let mut bars = vec![];
        for (n, l) in lines.enumerate() {
            let f: Vec<&str> = l.split(',').map(str::trim).collect();
            let num = |i: usize| f.get(i).and_then(|x| x.parse::<f64>().ok()).with_context(|| format!("{} line {}", p.display(), n + 2));
            let ts = num(it)? as Ms;
            bars.push(Bar { ts, close_ts: ts + bar_ms, known_ts: ts + bar_ms, open: num(io)?, high: num(ih)?, low: num(il)?, close: num(ic)? });
        }
        bars.sort_by_key(|b| b.ts);
        bars.dedup_by_key(|b| b.ts);
        out.insert(s.clone(), bars);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn csv_bars_parse_sort_and_dedup() {
        let dir = std::env::temp_dir().join(format!("atx-bars-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("ETHUSDT.csv"), "ts_ms,open,high,low,close\n3600000,2,3,1,2.5\n0,1,2,0.5,1.5\n0,9,9,9,9\n").unwrap();
        let m = read_bars_csv(&dir, &HashSet::from(["ETHUSDT".to_string(), "NONE".to_string()]), 3_600_000).unwrap();
        let b = &m["ETHUSDT"];
        assert_eq!(b.len(), 2);
        assert_eq!((b[0].ts, b[0].close_ts, b[1].open), (0, 3_600_000, 2.0));
        assert!(!m.contains_key("NONE"));
        std::fs::remove_dir_all(&dir).ok();
    }

    /// Reads the real lake if it is present (skips silently on a fresh checkout).
    #[test]
    fn lake_instruments_and_bars_are_readable() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../data/lake");
        if !root.join("instruments").is_dir() {
            return;
        }
        let now = at_core::time::now_ms();
        let (rules, _) = read_instruments(&root, "bybit", "linear", now).unwrap();
        if let Some(r) = rules.get("BTCUSDT") {
            assert!(r.tick_size > 0.0 && r.qty_step > 0.0);
        }
        let bars = read_bars(&root, "bybit", "last", "1d", &HashSet::from(["BTCUSDT".to_string()]), now - 30 * at_core::time::DAY, now).unwrap();
        if let Some(b) = bars.get("BTCUSDT") {
            assert!(b.windows(2).all(|w| w[0].ts < w[1].ts));
            assert!(b.iter().all(|x| x.known_ts <= now));
        }
    }
}
