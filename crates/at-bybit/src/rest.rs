//! HTTP client for Bybit v5 with request pacing, retries and signing.

use std::time::Duration;

use anyhow::{Context, anyhow};
use at_core::Error;
use at_core::time::now_ms;
use serde_json::Value;
use tokio::sync::Mutex;
use tokio::time::{Instant, sleep, sleep_until};

use crate::sign::rest_signature;

/// retCodes that are worth retrying (rate limit, server busy, timestamp drift).
const RETRYABLE: [i64; 4] = [10006, 10016, 10002, 10000];
const MAX_ATTEMPTS: u32 = 5;

#[derive(Clone)]
pub struct Credentials {
    pub api_key: String,
    pub api_secret: String,
}

impl std::fmt::Debug for Credentials {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("Credentials { api_key: \"***\", api_secret: \"***\" }")
    }
}

pub struct BybitRest {
    http: reqwest::Client,
    base: String,
    creds: Option<Credentials>,
    recv_window_ms: u64,
    min_interval: Duration,
    next_slot: Mutex<Instant>,
}

impl BybitRest {
    pub fn new(base: &str, creds: Option<Credentials>, recv_window_ms: u64, rate_per_s: f64) -> anyhow::Result<Self> {
        let http = reqwest::Client::builder()
            .timeout(Duration::from_secs(20))
            .user_agent("ai-trader/0.1")
            .build()?;
        Ok(Self {
            http,
            base: base.trim_end_matches('/').to_string(),
            creds,
            recv_window_ms,
            min_interval: Duration::from_secs_f64(1.0 / rate_per_s.max(0.1)),
            next_slot: Mutex::new(Instant::now()),
        })
    }

    pub fn base(&self) -> &str {
        &self.base
    }

    pub fn has_credentials(&self) -> bool {
        self.creds.is_some()
    }

    pub async fn get_public(&self, path: &str, params: &[(&str, String)]) -> anyhow::Result<Value> {
        self.request(reqwest::Method::GET, path, &encode(params), None, false).await
    }

    pub async fn get_private(&self, path: &str, params: &[(&str, String)]) -> anyhow::Result<Value> {
        self.request(reqwest::Method::GET, path, &encode(params), None, true).await
    }

    pub async fn post_private(&self, path: &str, body: &Value) -> anyhow::Result<Value> {
        let body = serde_json::to_string(body)?;
        self.request(reqwest::Method::POST, path, "", Some(body), true).await
    }

    /// Pace requests: one slot every `min_interval`, shared by all callers of this client.
    async fn pace(&self) {
        let mut next = self.next_slot.lock().await;
        let now = Instant::now();
        if *next > now {
            sleep_until(*next).await;
        }
        *next = Instant::now().max(*next) + self.min_interval;
    }

    async fn request(&self, method: reqwest::Method, path: &str, query: &str, body: Option<String>, signed: bool) -> anyhow::Result<Value> {
        let url = if query.is_empty() { format!("{}{path}", self.base) } else { format!("{}{path}?{query}", self.base) };
        let mut last_err = anyhow!("no attempt made");
        for attempt in 0..MAX_ATTEMPTS {
            if attempt > 0 {
                sleep(Duration::from_millis(500 * 2u64.pow(attempt - 1))).await;
            }
            self.pace().await;
            let mut req = self.http.request(method.clone(), &url);
            if signed {
                let creds = self.creds.as_ref().context("private endpoint called without credentials")?;
                let ts = now_ms();
                let payload = body.as_deref().unwrap_or(query);
                let sig = rest_signature(&creds.api_secret, ts, &creds.api_key, self.recv_window_ms, payload);
                req = req
                    .header("X-BAPI-API-KEY", &creds.api_key)
                    .header("X-BAPI-TIMESTAMP", ts.to_string())
                    .header("X-BAPI-RECV-WINDOW", self.recv_window_ms.to_string())
                    .header("X-BAPI-SIGN", sig);
            }
            if let Some(b) = &body {
                req = req.header("Content-Type", "application/json").body(b.clone());
            }
            let resp = match req.send().await {
                Ok(r) => r,
                Err(e) => {
                    last_err = anyhow!("{method} {path}: {e}");
                    continue;
                }
            };
            let status = resp.status();
            if status.as_u16() == 429 || status.is_server_error() {
                last_err = anyhow!("{method} {path}: HTTP {status}");
                continue;
            }
            let v: Value = resp.json().await.with_context(|| format!("{method} {path}: HTTP {status}, non-JSON body"))?;
            let code = v["retCode"].as_i64().unwrap_or(-1);
            if code == 0 {
                return Ok(v["result"].clone());
            }
            let err = Error::Venue { code, msg: v["retMsg"].as_str().unwrap_or("").to_string() };
            if RETRYABLE.contains(&code) {
                last_err = anyhow!("{method} {path}: {err}");
                continue;
            }
            return Err(anyhow!(err).context(format!("{method} {path}")));
        }
        Err(last_err.context(format!("giving up after {MAX_ATTEMPTS} attempts")))
    }
}

/// Query string in insertion order. The exact same string is sent and signed.
fn encode(params: &[(&str, String)]) -> String {
    let mut s = form_urlencoded::Serializer::new(String::new());
    for (k, v) in params {
        s.append_pair(k, v);
    }
    s.finish()
}

/// Venue error code of an `anyhow::Error` produced by this client, if any.
pub fn venue_code(e: &anyhow::Error) -> Option<i64> {
    e.chain().find_map(|c| match c.downcast_ref::<Error>() {
        Some(Error::Venue { code, .. }) => Some(*code),
        _ => None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn encode_preserves_order() {
        assert_eq!(encode(&[("symbol", "BTCUSDT".into()), ("category", "linear".into())]), "symbol=BTCUSDT&category=linear");
    }
}
