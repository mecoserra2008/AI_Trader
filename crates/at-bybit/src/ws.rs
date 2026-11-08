//! WebSocket streams with heartbeat and automatic reconnect. Messages carrying a `topic`
//! are forwarded to the channel together with the local receive time (the live `known_ts`).

use std::time::Duration;

use anyhow::{Context, bail};
use at_core::time::{Ms, now_ms};
use futures_util::{SinkExt, StreamExt};
use serde_json::{Value, json};
use tokio::sync::mpsc;
use tokio_tungstenite::connect_async;
use tokio_tungstenite::tungstenite::Message;

use crate::rest::Credentials;
use crate::sign::ws_signature;

const PING_EVERY: Duration = Duration::from_secs(20);
/// Bybit public linear streams accept at most 10 args per subscribe request.
const SUB_CHUNK: usize = 10;

#[derive(Debug, Clone)]
pub struct WsMessage {
    pub received_ts: Ms,
    pub topic: String,
    /// "snapshot" | "delta" for order books; empty otherwise.
    pub kind: String,
    /// Exchange timestamp of the message (ms), if present.
    pub ts: Option<Ms>,
    pub data: Value,
}

/// Run forever, reconnecting with exponential backoff (capped at 30 s). Returns only if the receiver is dropped.
/// A synthetic message with topic "__reconnect__" is sent before every reconnect so consumers can
/// invalidate state (e.g. order books, which are rebuilt from the fresh snapshot).
pub async fn run(url: String, topics: Vec<String>, creds: Option<Credentials>, tx: mpsc::Sender<WsMessage>) -> anyhow::Result<()> {
    let mut backoff = Duration::from_secs(1);
    loop {
        if backoff > Duration::from_secs(1) {
            let m = WsMessage { received_ts: now_ms(), topic: "__reconnect__".into(), kind: String::new(), ts: None, data: Value::Null };
            if tx.send(m).await.is_err() {
                return Ok(());
            }
        }
        let started = std::time::Instant::now();
        match session(&url, &topics, creds.as_ref(), &tx).await {
            Ok(()) => return Ok(()),
            Err(e) => {
                // A long healthy session resets the backoff; any session end invalidates downstream state.
                if started.elapsed() > Duration::from_secs(60) {
                    backoff = Duration::from_secs(1);
                }
                backoff = backoff.max(Duration::from_millis(1001));
                tracing::warn!(url = %url, error = %e, "websocket session ended; reconnecting in {:?}", backoff);
                tokio::time::sleep(backoff).await;
                backoff = (backoff * 2).min(Duration::from_secs(30));
            }
        }
    }
}

async fn session(url: &str, topics: &[String], creds: Option<&Credentials>, tx: &mpsc::Sender<WsMessage>) -> anyhow::Result<()> {
    let (ws, _) = connect_async(url).await.with_context(|| format!("connecting {url}"))?;
    let (mut write, mut read) = ws.split();

    if let Some(c) = creds {
        let expires = now_ms() + 10_000;
        let auth = json!({"op": "auth", "args": [c.api_key, expires, ws_signature(&c.api_secret, expires)]});
        write.send(Message::Text(auth.to_string())).await?;
    }
    for chunk in topics.chunks(SUB_CHUNK) {
        write.send(Message::Text(json!({"op": "subscribe", "args": chunk}).to_string())).await?;
    }
    tracing::info!(url = %url, n_topics = topics.len(), "websocket subscribed");

    let mut ping = tokio::time::interval(PING_EVERY);
    ping.tick().await;
    loop {
        tokio::select! {
            _ = ping.tick() => {
                write.send(Message::Text(json!({"op": "ping"}).to_string())).await?;
            }
            msg = read.next() => {
                let msg = match msg {
                    Some(m) => m?,
                    None => bail!("stream closed by server"),
                };
                match msg {
                    Message::Text(t) => {
                        let v: Value = serde_json::from_str(&t)?;
                        if let Some(topic) = v["topic"].as_str() {
                            let m = WsMessage {
                                received_ts: now_ms(),
                                topic: topic.to_string(),
                                kind: v["type"].as_str().unwrap_or_default().to_string(),
                                ts: v["ts"].as_i64(),
                                data: v["data"].clone(),
                            };
                            if tx.send(m).await.is_err() {
                                return Ok(());
                            }
                        } else if v["success"] == json!(false) {
                            bail!("websocket op failed: {t}");
                        }
                    }
                    Message::Ping(p) => write.send(Message::Pong(p)).await?,
                    Message::Close(f) => bail!("server closed: {f:?}"),
                    _ => {}
                }
            }
        }
    }
}
