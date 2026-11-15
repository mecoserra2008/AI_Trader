//! One Open API connection: a reader task turns the byte stream into frames; requests are matched to
//! responses by `clientMsgId`; unrelated frames (spot events) are queued for `next_event`.

use std::collections::VecDeque;
use std::time::{Duration, Instant};

use serde::Serialize;
use tokio::io::{AsyncRead, AsyncReadExt, AsyncWrite, AsyncWriteExt};
use tokio::sync::mpsc;

use crate::codec::{self, Frame, OaError, OaMsg, OaRequest};
use crate::creds::{CtraderCreds, CtraderEnv, PORT};
use crate::pb;

pub const DEFAULT_TIMEOUT: Duration = Duration::from_secs(20);
/// Spotware allows 5 historical requests per second per connection; keep a margin.
pub const HIST_MIN_INTERVAL: Duration = Duration::from_millis(220);
/// The server drops connections silent for 30 s; send a heartbeat at least this often.
pub const HEARTBEAT_EVERY: Duration = Duration::from_secs(10);

type Writer = Box<dyn AsyncWrite + Send + Unpin>;

pub struct Session {
    wr: Writer,
    rx: mpsc::Receiver<anyhow::Result<Frame>>,
    events: VecDeque<Frame>,
    next_id: u64,
    last_hist: Option<Instant>,
    last_send: Instant,
    pub hist_interval: Duration,
    pub timeout: Duration,
    reader: tokio::task::JoinHandle<()>,
}

#[derive(Clone, Copy, Debug, Default, Serialize)]
pub struct ConnectTiming {
    pub tcp_ms: f64,
    pub tls_ms: f64,
}

impl Drop for Session {
    fn drop(&mut self) {
        self.reader.abort();
    }
}

impl Session {
    /// TCP + TLS to `<env>.ctraderapi.com:5035`.
    pub async fn connect(env: CtraderEnv) -> anyhow::Result<(Self, ConnectTiming)> {
        let t0 = Instant::now();
        let tcp = tokio::time::timeout(DEFAULT_TIMEOUT, tokio::net::TcpStream::connect((env.host(), PORT))).await??;
        tcp.set_nodelay(true)?;
        let tcp_ms = t0.elapsed().as_secs_f64() * 1e3;
        let t1 = Instant::now();
        let cx = tokio_native_tls::TlsConnector::from(native_tls::TlsConnector::new()?);
        let tls = tokio::time::timeout(DEFAULT_TIMEOUT, cx.connect(env.host(), tcp)).await??;
        let tls_ms = t1.elapsed().as_secs_f64() * 1e3;
        Ok((Self::from_stream(tls), ConnectTiming { tcp_ms, tls_ms }))
    }

    /// Wrap any byte stream (TLS socket, or `tokio::io::duplex` in tests).
    pub fn from_stream<S: AsyncRead + AsyncWrite + Send + 'static>(s: S) -> Self {
        let (mut rd, wr) = tokio::io::split(s);
        let (tx, rx) = mpsc::channel(4096);
        let reader = tokio::spawn(async move {
            let mut len = [0u8; 4];
            loop {
                if let Err(e) = rd.read_exact(&mut len).await {
                    let _ = tx.send(Err(anyhow::anyhow!("connection closed: {e}"))).await;
                    return;
                }
                let n = u32::from_be_bytes(len) as usize;
                if n > codec::MAX_FRAME {
                    let _ = tx.send(Err(anyhow::anyhow!("frame of {n} bytes exceeds limit"))).await;
                    return;
                }
                let mut body = vec![0u8; n];
                if let Err(e) = rd.read_exact(&mut body).await {
                    let _ = tx.send(Err(anyhow::anyhow!("connection closed mid-frame: {e}"))).await;
                    return;
                }
                if tx.send(Frame::decode_body(&body)).await.is_err() {
                    return;
                }
            }
        });
        Self {
            wr: Box::new(wr),
            rx,
            events: VecDeque::new(),
            next_id: 1,
            last_hist: None,
            last_send: Instant::now(),
            hist_interval: HIST_MIN_INTERVAL,
            timeout: DEFAULT_TIMEOUT,
            reader,
        }
    }

    /// Send a frame. Refuses any payload type outside the read-only allowlist.
    pub async fn send_frame(&mut self, f: &Frame) -> anyhow::Result<()> {
        anyhow::ensure!(codec::SENDABLE.contains(&(f.payload_type as i32)), "payload type {} is not on the read-only allowlist; refusing to send", f.payload_type);
        if codec::is_historical(f.payload_type) {
            if let Some(t) = self.last_hist {
                let wait = self.hist_interval.saturating_sub(t.elapsed());
                if !wait.is_zero() {
                    tokio::time::sleep(wait).await;
                }
            }
            self.last_hist = Some(Instant::now());
        }
        self.wr.write_all(&f.encode()).await?;
        self.wr.flush().await?;
        self.last_send = Instant::now();
        Ok(())
    }

    pub async fn heartbeat(&mut self) -> anyhow::Result<()> {
        self.send_frame(&Frame::of(&pb::ProtoHeartbeatEvent { payload_type: None }, None)).await
    }

    /// Heartbeat if nothing was sent for `HEARTBEAT_EVERY`.
    pub async fn keepalive(&mut self) -> anyhow::Result<()> {
        if self.last_send.elapsed() >= HEARTBEAT_EVERY {
            self.heartbeat().await?;
        }
        Ok(())
    }

    async fn recv(&mut self, timeout: Duration) -> anyhow::Result<Option<Frame>> {
        match tokio::time::timeout(timeout, self.rx.recv()).await {
            Err(_) => Ok(None),
            Ok(None) => anyhow::bail!("connection reader stopped"),
            Ok(Some(r)) => r.map(Some),
        }
    }

    /// Send `req` and wait for its response. Rate-limit errors (`BLOCKED_PAYLOAD_TYPE`) are retried after
    /// the server's `retryAfter` (default 1 s), up to 5 times.
    pub async fn request<R: OaRequest>(&mut self, req: &R) -> anyhow::Result<R::Response> {
        let mut attempt = 0;
        loop {
            match self.request_once(req).await {
                Err(e) if attempt < 5 => match e.downcast_ref::<OaError>() {
                    Some(oe) if oe.code == "BLOCKED_PAYLOAD_TYPE" => {
                        attempt += 1;
                        let s = oe.retry_after_s.unwrap_or(1).clamp(1, 60);
                        tracing::warn!("cTrader rate limit on payload {}; retrying in {s} s", R::PAYLOAD);
                        tokio::time::sleep(Duration::from_secs(s)).await;
                    }
                    _ => return Err(e),
                },
                r => return r,
            }
        }
    }

    async fn request_once<R: OaRequest>(&mut self, req: &R) -> anyhow::Result<R::Response> {
        let id = self.next_id.to_string();
        self.next_id += 1;
        self.send_frame(&Frame::of(req, Some(id.clone()))).await?;
        let deadline = Instant::now() + self.timeout;
        loop {
            let left = deadline.saturating_duration_since(Instant::now());
            let Some(f) = self.recv(left).await? else {
                anyhow::bail!("timeout waiting for response to payload type {}", R::PAYLOAD);
            };
            let mine = f.client_msg_id.as_deref() == Some(id.as_str());
            if let Some(e) = f.error()
                && (mine || f.client_msg_id.is_none())
            {
                return Err(e.into());
            }
            if mine {
                return f.parse::<R::Response>();
            }
            if f.is::<pb::ProtoHeartbeatEvent>() {
                continue;
            }
            self.events.push_back(f);
        }
    }

    /// Next unsolicited frame (spot event, disconnect notice...), heartbeats filtered; `None` on timeout.
    pub async fn next_event(&mut self, timeout: Duration) -> anyhow::Result<Option<Frame>> {
        if let Some(f) = self.events.pop_front() {
            return Ok(Some(f));
        }
        let deadline = Instant::now() + timeout;
        loop {
            let left = deadline.saturating_duration_since(Instant::now());
            match self.recv(left).await? {
                None => return Ok(None),
                Some(f) if f.is::<pb::ProtoHeartbeatEvent>() => continue,
                Some(f) => return Ok(Some(f)),
            }
        }
    }

    // ---- typed helpers (all read-only) ----

    pub async fn app_auth(&mut self, c: &CtraderCreds) -> anyhow::Result<()> {
        self.request(&pb::ProtoOaApplicationAuthReq { payload_type: None, client_id: c.client_id.clone(), client_secret: c.client_secret.clone() }).await?;
        Ok(())
    }

    pub async fn version(&mut self) -> anyhow::Result<String> {
        Ok(self.request(&pb::ProtoOaVersionReq { payload_type: None }).await?.version)
    }

    pub async fn accounts(&mut self, c: &CtraderCreds) -> anyhow::Result<pb::ProtoOaGetAccountListByAccessTokenRes> {
        self.request(&pb::ProtoOaGetAccountListByAccessTokenReq { payload_type: None, access_token: c.access_token.clone() }).await
    }

    pub async fn account_auth(&mut self, account: i64, c: &CtraderCreds) -> anyhow::Result<()> {
        self.request(&pb::ProtoOaAccountAuthReq { payload_type: None, ctid_trader_account_id: account, access_token: c.access_token.clone() }).await?;
        Ok(())
    }

    pub async fn trader(&mut self, account: i64) -> anyhow::Result<pb::ProtoOaTrader> {
        Ok(self.request(&pb::ProtoOaTraderReq { payload_type: None, ctid_trader_account_id: account }).await?.trader)
    }

    pub async fn assets(&mut self, account: i64) -> anyhow::Result<Vec<pb::ProtoOaAsset>> {
        Ok(self.request(&pb::ProtoOaAssetListReq { payload_type: None, ctid_trader_account_id: account }).await?.asset)
    }

    pub async fn symbols(&mut self, account: i64) -> anyhow::Result<Vec<pb::ProtoOaLightSymbol>> {
        Ok(self.request(&pb::ProtoOaSymbolsListReq { payload_type: None, ctid_trader_account_id: account, include_archived_symbols: Some(false) }).await?.symbol)
    }

    pub async fn categories(&mut self, account: i64) -> anyhow::Result<Vec<pb::ProtoOaSymbolCategory>> {
        Ok(self.request(&pb::ProtoOaSymbolCategoryListReq { payload_type: None, ctid_trader_account_id: account }).await?.symbol_category)
    }

    pub async fn symbol_details(&mut self, account: i64, ids: &[i64]) -> anyhow::Result<Vec<pb::ProtoOaSymbol>> {
        let mut out = Vec::new();
        for chunk in ids.chunks(50) {
            out.extend(self.request(&pb::ProtoOaSymbolByIdReq { payload_type: None, ctid_trader_account_id: account, symbol_id: chunk.to_vec() }).await?.symbol);
        }
        Ok(out)
    }

    pub async fn subscribe_spots(&mut self, account: i64, ids: &[i64]) -> anyhow::Result<()> {
        self.request(&pb::ProtoOaSubscribeSpotsReq { payload_type: None, ctid_trader_account_id: account, symbol_id: ids.to_vec(), subscribe_to_spot_timestamp: Some(true) })
            .await?;
        Ok(())
    }
}

/// Scope of the access token: `SCOPE_VIEW` (0) is read-only, `SCOPE_TRADE` (1) allows trading.
pub fn scope_name(scope: Option<i32>) -> &'static str {
    match scope {
        Some(0) => "SCOPE_VIEW (read-only)",
        Some(1) => "SCOPE_TRADE (can trade)",
        Some(_) => "unknown",
        None => "not reported",
    }
}

/// Refuse anything but a read-only token. A missing scope is treated as unsafe too.
pub fn ensure_read_only(scope: Option<i32>) -> anyhow::Result<()> {
    if scope == Some(pb::ProtoOaClientPermissionScope::ScopeView as i32) {
        return Ok(());
    }
    anyhow::bail!(
        "the access token has scope {}; this project is data-only. Issue a read-only token: in the Open API \
         application's Playground / OAuth flow choose scope 'accounts' (not 'trading'), then put the new \
         CTRADER_ACCESS_TOKEN and CTRADER_REFRESH_TOKEN in .env",
        scope_name(scope)
    )
}

/// Pick the account: `CTRADER_ACCOUNT_ID` (ctid id or trader login) if set, else the only/first account whose
/// `isLive` matches the environment. Errors if the chosen account belongs to the other environment.
pub fn pick_account(accts: &[pb::ProtoOaCtidTraderAccount], wanted: Option<i64>, env: CtraderEnv) -> anyhow::Result<pb::ProtoOaCtidTraderAccount> {
    let a = match wanted {
        Some(w) => accts
            .iter()
            .find(|a| a.ctid_trader_account_id as i64 == w || a.trader_login == Some(w))
            .ok_or_else(|| anyhow::anyhow!("CTRADER_ACCOUNT_ID is not among the {} accounts of this token", accts.len()))?,
        None => accts
            .iter()
            .find(|a| a.is_live.unwrap_or(false) == env.is_live())
            .ok_or_else(|| anyhow::anyhow!("no {} account on this token ({} accounts in total)", env.as_str(), accts.len()))?,
    };
    anyhow::ensure!(
        a.is_live.unwrap_or(false) == env.is_live(),
        "the selected account is a {} account but CTRADER_ENV={}",
        if a.is_live.unwrap_or(false) { "live" } else { "demo" },
        env.as_str()
    );
    Ok(a.clone())
}

/// Decode a frame we expect to be an event, for callers that match on payload type.
pub fn parse_event<M: OaMsg>(f: &Frame) -> Option<M> {
    f.parse::<M>().ok()
}
