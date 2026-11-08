//! Historical trades from `public.bybit.com/trading/{SYMBOL}/{SYMBOL}{YYYY-MM-DD}.csv.gz`.
//! Bybit's `side` is the taker (aggressor) side, so signed volume / CVD is exact without a tick rule.

use std::collections::BTreeMap;
use std::io::Read;
use std::path::{Path, PathBuf};

use anyhow::{Context, bail};
use at_core::time::{Freq, Ms};
use flate2::read::GzDecoder;

#[derive(Clone, Debug, PartialEq)]
pub struct Trade {
    pub ts: Ms,
    pub trade_id: String,
    pub is_buy: bool,
    pub price: f64,
    pub size: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct FlowBar {
    pub open_ts: Ms,
    pub buy_volume: f64,
    pub sell_volume: f64,
    pub buy_turnover: f64,
    pub sell_turnover: f64,
    pub n_trades: i64,
    pub n_buy: i64,
    pub n_sell: i64,
    pub vwap: Option<f64>,
    pub max_trade_size: f64,
}

pub fn archive_url(base: &str, symbol: &str, date: &str) -> String {
    format!("{}/{symbol}/{symbol}{date}.csv.gz", base.trim_end_matches('/'))
}

/// Download (or read from `cache_dir`) one day of trades. Returns `Ok(None)` if the file doesn't exist.
pub async fn fetch_day(http: &reqwest::Client, base: &str, symbol: &str, date: &str, cache_dir: Option<&Path>) -> anyhow::Result<Option<Vec<Trade>>> {
    let cache_path: Option<PathBuf> = cache_dir.map(|d| d.join(symbol).join(format!("{symbol}{date}.csv.gz")));
    let gz = match cache_path.as_ref().filter(|p| p.is_file()) {
        Some(p) => std::fs::read(p)?,
        None => {
            // Daily files are 10-50 MB: generous timeout, and retry transient failures.
            let mut attempt = 0;
            let bytes = loop {
                attempt += 1;
                let res = async {
                    let resp = http.get(archive_url(base, symbol, date)).timeout(std::time::Duration::from_secs(300)).send().await?;
                    if resp.status().as_u16() == 404 {
                        return Ok::<_, anyhow::Error>(None);
                    }
                    Ok(Some(resp.error_for_status()?.bytes().await?.to_vec()))
                }
                .await;
                match res {
                    Ok(None) => return Ok(None),
                    Ok(Some(b)) => break b,
                    Err(e) if attempt < 4 => {
                        tracing::warn!(symbol, date, attempt, error = %e, "trade archive download failed; retrying");
                        tokio::time::sleep(std::time::Duration::from_secs(5 * attempt)).await;
                    }
                    Err(e) => return Err(e),
                }
            };
            if let Some(p) = &cache_path {
                std::fs::create_dir_all(p.parent().unwrap())?;
                std::fs::write(p, &bytes)?;
            }
            bytes
        }
    };
    let mut text = String::new();
    GzDecoder::new(gz.as_slice()).read_to_string(&mut text).context("decompressing trade archive")?;
    parse_csv(&text).map(Some)
}

pub fn parse_csv(text: &str) -> anyhow::Result<Vec<Trade>> {
    let mut rdr = csv::Reader::from_reader(text.as_bytes());
    let headers = rdr.headers()?.clone();
    let col = |name: &str| headers.iter().position(|h| h == name).with_context(|| format!("trade archive missing column {name}"));
    let (c_ts, c_side, c_size, c_price, c_id) = (col("timestamp")?, col("side")?, col("size")?, col("price")?, col("trdMatchID")?);
    let mut out = Vec::new();
    for rec in rdr.records() {
        let rec = rec?;
        let secs: f64 = rec[c_ts].parse()?;
        let is_buy = match &rec[c_side] {
            "Buy" => true,
            "Sell" => false,
            other => bail!("unknown trade side {other:?}"),
        };
        out.push(Trade {
            ts: (secs * 1000.0).round() as Ms,
            trade_id: rec[c_id].to_string(),
            is_buy,
            price: rec[c_price].parse()?,
            size: rec[c_size].parse()?,
        });
    }
    out.sort_by_key(|t| t.ts);
    Ok(out)
}

/// Aggregate trades into fixed-frequency order-flow bars (only bars containing trades are emitted).
pub fn aggregate(trades: &[Trade], freq: Freq) -> Vec<FlowBar> {
    let mut bars: BTreeMap<Ms, FlowBar> = BTreeMap::new();
    for t in trades {
        let open_ts = freq.floor(t.ts);
        let b = bars.entry(open_ts).or_insert_with(|| FlowBar {
            open_ts,
            buy_volume: 0.0,
            sell_volume: 0.0,
            buy_turnover: 0.0,
            sell_turnover: 0.0,
            n_trades: 0,
            n_buy: 0,
            n_sell: 0,
            vwap: None,
            max_trade_size: 0.0,
        });
        let notional = t.price * t.size;
        if t.is_buy {
            b.buy_volume += t.size;
            b.buy_turnover += notional;
            b.n_buy += 1;
        } else {
            b.sell_volume += t.size;
            b.sell_turnover += notional;
            b.n_sell += 1;
        }
        b.n_trades += 1;
        b.max_trade_size = b.max_trade_size.max(t.size);
    }
    bars.into_values()
        .map(|mut b| {
            let vol = b.buy_volume + b.sell_volume;
            b.vwap = (vol > 0.0).then(|| (b.buy_turnover + b.sell_turnover) / vol);
            b
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE: &str = "timestamp,symbol,side,size,price,tickDirection,trdMatchID,grossValue,homeNotional,foreignNotional,RPI
1793491200.7997,BTCUSDT,Sell,0.048,77511.10,MinusTick,a,1,0.048,1,0
1793491201.1000,BTCUSDT,Buy,0.1,77512.00,PlusTick,b,1,0.1,1,0
1793494800.0000,BTCUSDT,Buy,0.2,77600.00,PlusTick,c,1,0.2,1,0
";

    #[test]
    fn parses_and_aggregates() {
        let t = parse_csv(SAMPLE).unwrap();
        assert_eq!(t.len(), 3);
        assert_eq!(t[0].ts, 1_793_491_200_800);
        let bars = aggregate(&t, Freq::H1);
        assert_eq!(bars.len(), 2);
        assert_eq!(bars[0].n_trades, 2);
        assert!((bars[0].sell_volume - 0.048).abs() < 1e-12);
        assert!((bars[0].buy_volume - 0.1).abs() < 1e-12);
        assert_eq!(bars[1].open_ts, 1_793_494_800_000);
    }
}
