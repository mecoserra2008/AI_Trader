//! UTC millisecond timestamps, bar frequencies and the look-ahead guard.

use chrono::{DateTime, NaiveDate, TimeZone, Utc};
use serde::{Deserialize, Serialize};

use crate::error::{Error, Result};

/// Milliseconds since the Unix epoch, UTC. The only timestamp representation in the system.
pub type Ms = i64;

pub const SECOND: Ms = 1_000;
pub const MINUTE: Ms = 60 * SECOND;
pub const HOUR: Ms = 60 * MINUTE;
pub const DAY: Ms = 24 * HOUR;

pub fn now_ms() -> Ms {
    Utc::now().timestamp_millis()
}

pub fn to_datetime(ms: Ms) -> DateTime<Utc> {
    Utc.timestamp_millis_opt(ms).single().unwrap_or_default()
}

/// Accepts `YYYY-MM-DD` (midnight UTC) or RFC 3339.
pub fn parse_ts(s: &str) -> anyhow::Result<Ms> {
    if let Ok(d) = NaiveDate::parse_from_str(s, "%Y-%m-%d") {
        return Ok(d.and_hms_opt(0, 0, 0).unwrap().and_utc().timestamp_millis());
    }
    Ok(DateTime::parse_from_rfc3339(s)?.timestamp_millis())
}

pub fn format_date(ms: Ms) -> String {
    to_datetime(ms).format("%Y-%m-%d").to_string()
}

/// Refuse any output computed from data that was not knowable at `as_of`.
pub fn assert_no_lookahead(data_max_ts: Ms, as_of: Ms) -> Result<()> {
    if data_max_ts > as_of {
        return Err(Error::LookAhead { data_max_ts, as_of });
    }
    Ok(())
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub enum Freq {
    #[serde(rename = "1m")]
    M1,
    #[serde(rename = "5m")]
    M5,
    #[serde(rename = "15m")]
    M15,
    #[serde(rename = "1h")]
    H1,
    #[serde(rename = "4h")]
    H4,
    #[serde(rename = "1d")]
    D1,
}

impl Freq {
    pub const ALL: [Freq; 6] = [Freq::M1, Freq::M5, Freq::M15, Freq::H1, Freq::H4, Freq::D1];

    pub fn as_str(self) -> &'static str {
        match self {
            Freq::M1 => "1m",
            Freq::M5 => "5m",
            Freq::M15 => "15m",
            Freq::H1 => "1h",
            Freq::H4 => "4h",
            Freq::D1 => "1d",
        }
    }

    pub fn parse(s: &str) -> anyhow::Result<Freq> {
        Freq::ALL
            .into_iter()
            .find(|f| f.as_str() == s)
            .ok_or_else(|| anyhow::anyhow!("unknown frequency {s:?} (expected one of 1m,5m,15m,1h,4h,1d)"))
    }

    pub fn duration_ms(self) -> Ms {
        match self {
            Freq::M1 => MINUTE,
            Freq::M5 => 5 * MINUTE,
            Freq::M15 => 15 * MINUTE,
            Freq::H1 => HOUR,
            Freq::H4 => 4 * HOUR,
            Freq::D1 => DAY,
        }
    }

    /// Bybit v5 kline `interval` parameter.
    pub fn bybit_interval(self) -> &'static str {
        match self {
            Freq::M1 => "1",
            Freq::M5 => "5",
            Freq::M15 => "15",
            Freq::H1 => "60",
            Freq::H4 => "240",
            Freq::D1 => "D",
        }
    }

    /// Start of the bar containing `ms` (bars are aligned to the Unix epoch, i.e. 00:00 UTC).
    pub fn floor(self, ms: Ms) -> Ms {
        ms - ms.rem_euclid(self.duration_ms())
    }
}

impl std::fmt::Display for Freq {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn freq_roundtrip_and_floor() {
        for f in Freq::ALL {
            assert_eq!(Freq::parse(f.as_str()).unwrap(), f);
        }
        let t = parse_ts("2026-11-07T13:47:12Z").unwrap();
        assert_eq!(Freq::H1.floor(t), parse_ts("2026-11-07T13:00:00Z").unwrap());
        assert_eq!(Freq::D1.floor(t), parse_ts("2026-11-07").unwrap());
        assert_eq!(Freq::H4.floor(t), parse_ts("2026-11-07T12:00:00Z").unwrap());
    }

    #[test]
    fn lookahead_guard() {
        assert!(assert_no_lookahead(10, 10).is_ok());
        assert!(matches!(assert_no_lookahead(11, 10), Err(Error::LookAhead { .. })));
    }
}
