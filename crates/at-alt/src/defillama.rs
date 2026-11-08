//! DefiLlama total stablecoin supply (USD-pegged), daily. Revisable: DefiLlama back-fills
//! newly tracked stablecoins, so history before our own archive start is not point-in-time.

use anyhow::Context;
use at_core::time::{Ms, now_ms};
use serde_json::Value;

use crate::{Obs, PitQuality};

pub const SOURCE: &str = "defillama";
pub const PIT: PitQuality = PitQuality::Revisable;

pub async fn stablecoins_total(http: &reqwest::Client, latency_ms: Ms) -> anyhow::Result<Vec<Obs>> {
    let v: Value = http.get("https://stablecoins.llama.fi/stablecoincharts/all").send().await?.error_for_status()?.json().await?;
    let now = now_ms();
    Ok(v.as_array()
        .context("defillama: expected array")?
        .iter()
        .filter_map(|x| {
            let ts = x["date"].as_str()?.parse::<i64>().ok()? * 1000;
            let value = x["totalCirculatingUSD"]["peggedUSD"].as_f64();
            Some(Obs { series_id: "stablecoins_total_usd".into(), entity: "all".into(), event_ts: ts, value, vintage: None, known_ts: ts + latency_ms })
        })
        .filter(|o| o.known_ts <= now)
        .collect())
}
