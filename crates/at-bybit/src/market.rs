//! Public market data with pagination. Every fetch returns records sorted by time ascending,
//! de-duplicated, and restricted to periods that are complete at fetch time.

use std::collections::BTreeMap;

use anyhow::{Context, bail};
use at_core::time::{Freq, Ms, now_ms};
use serde_json::Value;

use crate::rest::BybitRest;

#[derive(Clone, Debug)]
pub struct Instrument {
    pub symbol: String,
    pub base: String,
    pub quote: String,
    pub settle: String,
    pub contract_type: String,
    /// "" for crypto; "ETF" | "stock" | "commodity" for TradFi perpetuals.
    pub symbol_type: String,
    pub status: String,
    pub launch_ts: Option<Ms>,
    pub tick_size: f64,
    pub qty_step: f64,
    pub min_qty: f64,
    pub max_qty: f64,
    pub max_mkt_qty: f64,
    pub min_notional: f64,
    pub max_leverage: f64,
    pub funding_interval_min: i32,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Kline {
    pub open_ts: Ms,
    pub open: f64,
    pub high: f64,
    pub low: f64,
    pub close: f64,
    pub volume: Option<f64>,
    pub turnover: Option<f64>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum PriceType {
    Last,
    Mark,
    Index,
    Premium,
}

impl PriceType {
    pub fn parse(s: &str) -> anyhow::Result<Self> {
        Ok(match s {
            "last" => PriceType::Last,
            "mark" => PriceType::Mark,
            "index" => PriceType::Index,
            "premium" => PriceType::Premium,
            other => bail!("unknown price type {other:?}"),
        })
    }

    pub fn as_str(self) -> &'static str {
        match self {
            PriceType::Last => "last",
            PriceType::Mark => "mark",
            PriceType::Index => "index",
            PriceType::Premium => "premium",
        }
    }

    fn path(self) -> &'static str {
        match self {
            PriceType::Last => "/v5/market/kline",
            PriceType::Mark => "/v5/market/mark-price-kline",
            PriceType::Index => "/v5/market/index-price-kline",
            PriceType::Premium => "/v5/market/premium-index-price-kline",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Funding {
    pub ts: Ms,
    pub rate: f64,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct OpenInterest {
    pub ts: Ms,
    pub open_interest: f64,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LongShortRatio {
    pub ts: Ms,
    pub buy_ratio: f64,
    pub sell_ratio: f64,
}

#[derive(Clone, Debug)]
pub struct Ticker {
    pub symbol: String,
    pub last_price: f64,
    pub mark_price: f64,
    pub turnover_24h: f64,
    pub open_interest_value: f64,
    pub funding_rate: f64,
    /// Best bid / ask (0 when the venue omits them).
    pub bid1: f64,
    pub ask1: f64,
}

/// Open-interest / long-short periods accepted by Bybit, with their durations.
pub fn period_ms(p: &str) -> anyhow::Result<Ms> {
    Ok(match p {
        "5min" => 5 * 60_000,
        "15min" => 15 * 60_000,
        "30min" => 30 * 60_000,
        "1h" => 3_600_000,
        "4h" => 4 * 3_600_000,
        "1d" => 86_400_000,
        other => bail!("unsupported period {other:?} (5min,15min,30min,1h,4h,1d)"),
    })
}

fn f(v: &Value) -> f64 {
    v.as_str().and_then(|s| s.parse().ok()).or_else(|| v.as_f64()).unwrap_or(f64::NAN)
}

fn i(v: &Value) -> i64 {
    v.as_str().and_then(|s| s.parse().ok()).or_else(|| v.as_i64()).unwrap_or(0)
}

pub struct MarketData<'a> {
    rest: &'a BybitRest,
    category: String,
}

impl<'a> MarketData<'a> {
    pub fn new(rest: &'a BybitRest, category: &str) -> Self {
        Self { rest, category: category.to_string() }
    }

    pub async fn server_time_ms(&self) -> anyhow::Result<Ms> {
        let r = self.rest.get_public("/v5/market/time", &[]).await?;
        Ok(i(&r["timeNano"]) / 1_000_000)
    }

    pub async fn instruments(&self) -> anyhow::Result<Vec<Instrument>> {
        let mut out = Vec::new();
        let mut cursor = String::new();
        loop {
            let mut params = vec![("category", self.category.clone()), ("limit", "1000".to_string())];
            if !cursor.is_empty() {
                params.push(("cursor", cursor.clone()));
            }
            let r = self.rest.get_public("/v5/market/instruments-info", &params).await?;
            for x in r["list"].as_array().context("instruments: no list")? {
                let launch = i(&x["launchTime"]);
                out.push(Instrument {
                    symbol: x["symbol"].as_str().unwrap_or_default().to_string(),
                    base: x["baseCoin"].as_str().unwrap_or_default().to_string(),
                    quote: x["quoteCoin"].as_str().unwrap_or_default().to_string(),
                    settle: x["settleCoin"].as_str().unwrap_or_default().to_string(),
                    contract_type: x["contractType"].as_str().unwrap_or_default().to_string(),
                    symbol_type: x["symbolType"].as_str().unwrap_or_default().to_string(),
                    status: x["status"].as_str().unwrap_or_default().to_string(),
                    launch_ts: (launch > 0).then_some(launch),
                    tick_size: f(&x["priceFilter"]["tickSize"]),
                    qty_step: f(&x["lotSizeFilter"]["qtyStep"]),
                    min_qty: f(&x["lotSizeFilter"]["minOrderQty"]),
                    max_qty: f(&x["lotSizeFilter"]["maxOrderQty"]),
                    max_mkt_qty: f(&x["lotSizeFilter"]["maxMktOrderQty"]),
                    min_notional: f(&x["lotSizeFilter"]["minNotionalValue"]),
                    max_leverage: f(&x["leverageFilter"]["maxLeverage"]),
                    funding_interval_min: i(&x["fundingInterval"]) as i32,
                });
            }
            cursor = r["nextPageCursor"].as_str().unwrap_or_default().to_string();
            if cursor.is_empty() {
                break;
            }
        }
        Ok(out)
    }

    /// Klines in `[start, end)`. Only bars whose close time is <= now are returned (no partial bar).
    pub async fn klines(&self, symbol: &str, freq: Freq, price_type: PriceType, start: Ms, end: Ms) -> anyhow::Result<Vec<Kline>> {
        const LIMIT: usize = 1000;
        let dur = freq.duration_ms();
        let mut by_ts: BTreeMap<Ms, Kline> = BTreeMap::new();
        let mut cur_end = end - 1;
        while cur_end >= start {
            let params = [
                ("category", self.category.clone()),
                ("symbol", symbol.to_string()),
                ("interval", freq.bybit_interval().to_string()),
                ("start", start.to_string()),
                ("end", cur_end.to_string()),
                ("limit", LIMIT.to_string()),
            ];
            let r = self.rest.get_public(price_type.path(), &params).await?;
            let rows = r["list"].as_array().context("kline: no list")?;
            if rows.is_empty() {
                break;
            }
            let mut oldest = Ms::MAX;
            for row in rows {
                let a = row.as_array().context("kline row not an array")?;
                let ts = i(&a[0]);
                oldest = oldest.min(ts);
                by_ts.insert(ts, Kline {
                    open_ts: ts,
                    open: f(&a[1]),
                    high: f(&a[2]),
                    low: f(&a[3]),
                    close: f(&a[4]),
                    volume: a.get(5).map(f),
                    turnover: a.get(6).map(f),
                });
            }
            if rows.len() < LIMIT || oldest <= start {
                break;
            }
            cur_end = oldest - 1;
        }
        let now = now_ms();
        Ok(by_ts
            .into_values()
            .filter(|k| k.open_ts >= start && k.open_ts < end && k.open_ts + dur <= now)
            .filter(|k| {
                // Venue data contains occasional impossible bars (e.g. BCHUSDT 1d 2020-12-14 with open = low = 0).
                let ok = k.open > 0.0 && k.low > 0.0 && k.high >= k.open.max(k.close) && k.low <= k.open.min(k.close);
                if !ok {
                    tracing::warn!(symbol, ts = k.open_ts, "dropping invalid kline {:?}", k);
                }
                ok
            })
            .collect())
    }

    /// Realised funding in `[start, end)`.
    pub async fn funding_history(&self, symbol: &str, start: Ms, end: Ms) -> anyhow::Result<Vec<Funding>> {
        const LIMIT: usize = 200;
        let mut by_ts: BTreeMap<Ms, Funding> = BTreeMap::new();
        let mut cur_end = end - 1;
        while cur_end >= start {
            let params = [
                ("category", self.category.clone()),
                ("symbol", symbol.to_string()),
                ("startTime", start.to_string()),
                ("endTime", cur_end.to_string()),
                ("limit", LIMIT.to_string()),
            ];
            let r = self.rest.get_public("/v5/market/funding/history", &params).await?;
            let rows = r["list"].as_array().context("funding: no list")?;
            if rows.is_empty() {
                break;
            }
            let mut oldest = Ms::MAX;
            for x in rows {
                let ts = i(&x["fundingRateTimestamp"]);
                oldest = oldest.min(ts);
                by_ts.insert(ts, Funding { ts, rate: f(&x["fundingRate"]) });
            }
            if rows.len() < LIMIT || oldest <= start {
                break;
            }
            cur_end = oldest - 1;
        }
        Ok(by_ts.into_values().filter(|x| x.ts >= start && x.ts < end).collect())
    }

    #[allow(clippy::too_many_arguments)]
    /// Cursor-paginated series (open interest, long/short ratio). Bybit pages backwards from `end`.
    async fn cursor_series(&self, path: &str, symbol: &str, period_key: &str, period: &str, limit: usize, start: Ms, end: Ms) -> anyhow::Result<Vec<Value>> {
        let mut out = Vec::new();
        let mut cursor = String::new();
        loop {
            let mut params = vec![
                ("category", self.category.clone()),
                ("symbol", symbol.to_string()),
                (period_key, period.to_string()),
                ("startTime", start.to_string()),
                ("endTime", (end - 1).to_string()),
                ("limit", limit.to_string()),
            ];
            if !cursor.is_empty() {
                params.push(("cursor", cursor.clone()));
            }
            let r = self.rest.get_public(path, &params).await?;
            let rows = r["list"].as_array().cloned().unwrap_or_default();
            let n = rows.len();
            let oldest = rows.iter().map(|x| i(&x["timestamp"])).min().unwrap_or(Ms::MIN);
            out.extend(rows);
            cursor = r["nextPageCursor"].as_str().unwrap_or_default().to_string();
            if n == 0 || cursor.is_empty() || oldest <= start {
                break;
            }
        }
        Ok(out)
    }

    pub async fn open_interest(&self, symbol: &str, interval: &str, start: Ms, end: Ms) -> anyhow::Result<Vec<OpenInterest>> {
        period_ms(interval)?;
        let rows = self.cursor_series("/v5/market/open-interest", symbol, "intervalTime", interval, 200, start, end).await?;
        let mut by_ts = BTreeMap::new();
        for x in rows {
            let ts = i(&x["timestamp"]);
            by_ts.insert(ts, OpenInterest { ts, open_interest: f(&x["openInterest"]) });
        }
        Ok(by_ts.into_values().filter(|x| x.ts >= start && x.ts < end).collect())
    }

    pub async fn long_short_ratio(&self, symbol: &str, period: &str, start: Ms, end: Ms) -> anyhow::Result<Vec<LongShortRatio>> {
        period_ms(period)?;
        let rows = self.cursor_series("/v5/market/account-ratio", symbol, "period", period, 500, start, end).await?;
        let mut by_ts = BTreeMap::new();
        for x in rows {
            let ts = i(&x["timestamp"]);
            by_ts.insert(ts, LongShortRatio { ts, buy_ratio: f(&x["buyRatio"]), sell_ratio: f(&x["sellRatio"]) });
        }
        Ok(by_ts.into_values().filter(|x| x.ts >= start && x.ts < end).collect())
    }

    pub async fn tickers(&self) -> anyhow::Result<Vec<Ticker>> {
        let r = self.rest.get_public("/v5/market/tickers", &[("category", self.category.clone())]).await?;
        Ok(r["list"]
            .as_array()
            .context("tickers: no list")?
            .iter()
            .map(|x| Ticker {
                symbol: x["symbol"].as_str().unwrap_or_default().to_string(),
                last_price: f(&x["lastPrice"]),
                mark_price: f(&x["markPrice"]),
                turnover_24h: f(&x["turnover24h"]),
                open_interest_value: f(&x["openInterestValue"]),
                funding_rate: f(&x["fundingRate"]),
                bid1: f(&x["bid1Price"]),
                ask1: f(&x["ask1Price"]),
            })
            .collect())
    }
}
