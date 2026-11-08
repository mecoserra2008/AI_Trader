//! Crypto Fear & Greed Index (alternative.me), daily since 2018-02.

use anyhow::Context;
use at_core::time::{Ms, now_ms};
use serde_json::Value;

use crate::{Obs, PitQuality};

pub const SOURCE: &str = "alternative_me";
pub const PIT: PitQuality = PitQuality::Native;

pub async fn fetch_all(http: &reqwest::Client, latency_ms: Ms) -> anyhow::Result<Vec<Obs>> {
    let v: Value = http.get("https://api.alternative.me/fng/?limit=0&format=json").send().await?.error_for_status()?.json().await?;
    let now = now_ms();
    let mut out: Vec<Obs> = v["data"]
        .as_array()
        .context("fng: no data array")?
        .iter()
        .filter_map(|x| {
            let ts = x["timestamp"].as_str()?.parse::<i64>().ok()? * 1000;
            let value = x["value"].as_str()?.parse::<f64>().ok();
            Some(Obs { series_id: "fear_greed".into(), entity: "crypto".into(), event_ts: ts, value, vintage: None, known_ts: ts + latency_ms })
        })
        .filter(|o| o.known_ts <= now)
        .collect();
    out.sort_by_key(|o| o.event_ts);
    Ok(out)
}
