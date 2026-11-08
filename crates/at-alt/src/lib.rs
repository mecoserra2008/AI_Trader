//! Alternative-data connectors. Each produces `alt_series` rows with an explicit point-in-time
//! quality tag and a conservative `known_ts` (config/datasources.yaml → defaults.known_ts_latency_ms).

pub mod defillama;
pub mod deribit;
pub mod fng;
pub mod fred;

use at_core::time::Ms;
use at_store::Batch;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PitQuality {
    /// Publication-time stamps, values never revised.
    Native,
    /// Source keeps revision history; each vintage is its own row.
    Vintage,
    /// Values can be silently revised: trust only self-archived live snapshots.
    Revisable,
}

impl PitQuality {
    pub fn as_str(self) -> &'static str {
        match self {
            PitQuality::Native => "native",
            PitQuality::Vintage => "vintage",
            PitQuality::Revisable => "revisable",
        }
    }
}

#[derive(Clone, Debug)]
pub struct Obs {
    pub series_id: String,
    pub entity: String,
    pub event_ts: Ms,
    pub value: Option<f64>,
    pub vintage: Option<String>,
    pub known_ts: Ms,
}

pub fn to_batch(source: &str, pit: PitQuality, xs: &[Obs]) -> Batch {
    let n = xs.len();
    Batch::new()
        .utf8_const("source", source, n)
        .utf8("series_id", xs.iter().map(|o| o.series_id.clone()).collect())
        .utf8("entity", xs.iter().map(|o| o.entity.clone()).collect())
        .ts("event_ts", xs.iter().map(|o| o.event_ts).collect())
        .f64_opt("value", xs.iter().map(|o| o.value).collect())
        .utf8_opt("vintage", xs.iter().map(|o| o.vintage.clone()).collect())
        .utf8_const("pit_quality", pit.as_str(), n)
        .ts("known_ts", xs.iter().map(|o| o.known_ts).collect())
}

pub fn http() -> anyhow::Result<reqwest::Client> {
    Ok(reqwest::Client::builder().timeout(std::time::Duration::from_secs(30)).user_agent("ai-trader/0.1").build()?)
}
