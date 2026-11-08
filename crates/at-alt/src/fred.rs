//! FRED observations with **ALFRED vintages**: every revision is its own row, with known_ts at
//! the vintage date (+ latency), so a read as of date t sees exactly what was published by t.

use anyhow::{Context, bail};
use at_core::time::{Ms, parse_ts};
use serde_json::Value;

use crate::{Obs, PitQuality};

pub const SOURCE: &str = "fred";
pub const PIT: PitQuality = PitQuality::Vintage;

pub async fn observations_all_vintages(http: &reqwest::Client, api_key: &str, series_id: &str, latency_ms: Ms) -> anyhow::Result<Vec<Obs>> {
    const LIMIT: usize = 100_000;
    let mut out = Vec::new();
    let mut offset = 0usize;
    loop {
        let url = format!(
            "https://api.stlouisfed.org/fred/series/observations?series_id={series_id}&api_key={api_key}&file_type=json\
             &realtime_start=1776-07-04&realtime_end=9999-12-31&limit={LIMIT}&offset={offset}"
        );
        let v: Value = http.get(&url).send().await?.json().await?;
        if let Some(msg) = v.get("error_message") {
            bail!("FRED {series_id}: {msg}");
        }
        let rows = v["observations"].as_array().context("FRED: no observations")?;
        for r in rows {
            let date = r["date"].as_str().unwrap_or_default();
            let vintage = r["realtime_start"].as_str().unwrap_or_default();
            let value = r["value"].as_str().and_then(|s| s.parse::<f64>().ok()); // "." = missing
            out.push(Obs {
                series_id: series_id.to_string(),
                entity: "us".into(),
                event_ts: parse_ts(date)?,
                value,
                vintage: Some(vintage.to_string()),
                // Never before the period itself: a vintage dated earlier than the observation can't be known earlier.
                known_ts: parse_ts(vintage)?.max(parse_ts(date)?) + latency_ms,
            });
        }
        if rows.len() < LIMIT {
            break;
        }
        offset += LIMIT;
    }
    Ok(out)
}
