//! Credentials from the environment (`.env` is loaded by `at_core::config::load`). Values are never printed:
//! only "present / missing" per key, and `Debug` is redacted.

use at_core::config::secret;

pub const KEYS: &[&str] = &["CTRADER_CLIENT_ID", "CTRADER_CLIENT_SECRET", "CTRADER_ACCESS_TOKEN", "CTRADER_REFRESH_TOKEN", "CTRADER_ENV", "CTRADER_ACCOUNT_ID"];
pub const REQUIRED: &[&str] = &["CTRADER_CLIENT_ID", "CTRADER_CLIENT_SECRET", "CTRADER_ACCESS_TOKEN", "CTRADER_ENV"];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CtraderEnv {
    Live,
    Demo,
}

impl CtraderEnv {
    pub fn parse(s: &str) -> anyhow::Result<Self> {
        match s.trim().to_ascii_lowercase().as_str() {
            "live" => Ok(Self::Live),
            "demo" => Ok(Self::Demo),
            other => anyhow::bail!("CTRADER_ENV must be live or demo, got {other:?}"),
        }
    }

    pub fn host(self) -> &'static str {
        match self {
            Self::Live => "live.ctraderapi.com",
            Self::Demo => "demo.ctraderapi.com",
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Live => "live",
            Self::Demo => "demo",
        }
    }

    pub fn is_live(self) -> bool {
        self == Self::Live
    }
}

pub const PORT: u16 = 5035;

#[derive(Clone)]
pub struct CtraderCreds {
    pub client_id: String,
    pub client_secret: String,
    pub access_token: String,
    pub refresh_token: Option<String>,
    pub env: CtraderEnv,
    /// ctidTraderAccountId or the trader login shown in the cTrader UI.
    pub account_id: Option<i64>,
}

impl std::fmt::Debug for CtraderCreds {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CtraderCreds").field("env", &self.env).field("account_id", &self.account_id.map(|_| "<set>")).finish_non_exhaustive()
    }
}

/// `(key, present)` for every key; the value itself is never returned.
pub fn presence() -> Vec<(&'static str, bool)> {
    KEYS.iter().map(|k| (*k, secret(k).is_some())).collect()
}

impl CtraderCreds {
    pub fn from_env() -> anyhow::Result<Self> {
        let missing: Vec<&str> = REQUIRED.iter().copied().filter(|k| secret(k).is_none()).collect();
        anyhow::ensure!(missing.is_empty(), "missing in .env: {}", missing.join(", "));
        let account_id = match secret("CTRADER_ACCOUNT_ID") {
            Some(v) => Some(v.trim().parse::<i64>().map_err(|_| anyhow::anyhow!("CTRADER_ACCOUNT_ID must be an integer"))?),
            None => None,
        };
        Ok(Self {
            client_id: secret("CTRADER_CLIENT_ID").expect("checked"),
            client_secret: secret("CTRADER_CLIENT_SECRET").expect("checked"),
            access_token: secret("CTRADER_ACCESS_TOKEN").expect("checked"),
            refresh_token: secret("CTRADER_REFRESH_TOKEN"),
            env: CtraderEnv::parse(&secret("CTRADER_ENV").expect("checked"))?,
            account_id,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn env_parse_and_redacted_debug() {
        assert_eq!(CtraderEnv::parse(" LIVE ").unwrap().host(), "live.ctraderapi.com");
        assert_eq!(CtraderEnv::parse("demo").unwrap().host(), "demo.ctraderapi.com");
        assert!(CtraderEnv::parse("mainnet").is_err());
        let c =
            CtraderCreds { client_id: "id-x".into(), client_secret: "sec-x".into(), access_token: "tok-x".into(), refresh_token: None, env: CtraderEnv::Demo, account_id: Some(1) };
        let s = format!("{c:?}");
        assert!(!s.contains("sec-x") && !s.contains("tok-x") && !s.contains("id-x"), "{s}");
    }
}
