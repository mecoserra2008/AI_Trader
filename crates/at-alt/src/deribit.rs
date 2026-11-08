//! Deribit DVOL implied-volatility index (BTC, ETH). Candles are [ts, open, high, low, close];
//! a candle is known only after it closes.

use anyhow::{Context, bail};
use at_core::time::{Ms, now_ms};
use serde_json::Value;

use crate::{Obs, PitQuality};

pub const SOURCE: &str = "deribit";
pub const PIT: PitQuality = PitQuality::Native;

fn resolution_ms(resolution: &str) -> anyhow::Result<Ms> {
    Ok(match resolution {
        "1D" => 86_400_000,
        s => s.parse::<i64>().map(|x| x * 1000).with_context(|| format!("bad DVOL resolution {s:?}"))?,
    })
}

/// DVOL close series in `[start, end)`. Paginates backwards via Deribit's `continuation`.
pub async fn dvol(http: &reqwest::Client, currency: &str, resolution: &str, start: Ms, end: Ms, latency_ms: Ms) -> anyhow::Result<Vec<Obs>> {
    let res_ms = resolution_ms(resolution)?;
    let now = now_ms();
    let mut out = std::collections::BTreeMap::new();
    let mut cur_end = end;
    loop {
        let url = format!(
            "https://www.deribit.com/api/v2/public/get_volatility_index_data?currency={currency}&start_timestamp={start}&end_timestamp={cur_end}&resolution={resolution}"
        );
        let v: Value = http.get(&url).send().await?.error_for_status()?.json().await?;
        if let Some(e) = v.get("error") {
            bail!("deribit error: {e}");
        }
        let rows = v["result"]["data"].as_array().cloned().unwrap_or_default();
        for r in &rows {
            let ts = r[0].as_i64().unwrap_or(0);
            let known = ts + res_ms + latency_ms;
            if ts >= start && ts < end && known <= now {
                out.insert(ts, Obs {
                    series_id: "dvol_close".into(),
                    entity: currency.to_string(),
                    event_ts: ts,
                    value: r[4].as_f64(),
                    vintage: None,
                    known_ts: known,
                });
            }
        }
        match v["result"]["continuation"].as_i64() {
            Some(c) if !rows.is_empty() && c > start && c < cur_end => cur_end = c,
            _ => break,
        }
    }
    Ok(out.into_values().collect())
}
