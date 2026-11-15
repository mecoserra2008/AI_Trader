//! Framing and typed messages. A frame = u32 big-endian length + `ProtoMessage` bytes.

use prost::Message;

use crate::pb::{self, ProtoOaPayloadType as P};

/// Largest frame we accept (symbol lists of large brokers are a few MB).
pub const MAX_FRAME: usize = 64 * 1024 * 1024;

/// Payload types this client may send. Everything else (orders, amends, cancels, closes, margin-call
/// updates, execution-event subscriptions) is refused before it reaches the socket.
pub const SENDABLE: &[i32] = &[
    pb::ProtoPayloadType::HeartbeatEvent as i32,
    P::ProtoOaApplicationAuthReq as i32,
    P::ProtoOaAccountAuthReq as i32,
    P::ProtoOaVersionReq as i32,
    P::ProtoOaAssetListReq as i32,
    P::ProtoOaSymbolsListReq as i32,
    P::ProtoOaSymbolByIdReq as i32,
    P::ProtoOaTraderReq as i32,
    P::ProtoOaSubscribeSpotsReq as i32,
    P::ProtoOaUnsubscribeSpotsReq as i32,
    P::ProtoOaGetTickdataReq as i32,
    P::ProtoOaGetTrendbarsReq as i32,
    P::ProtoOaGetAccountsByAccessTokenReq as i32,
    P::ProtoOaSymbolCategoryReq as i32,
    P::ProtoOaAccountLogoutReq as i32,
];

/// Historical-data requests share the ~5 req/s limit (tick data and trend bars).
pub fn is_historical(payload_type: u32) -> bool {
    payload_type == P::ProtoOaGetTickdataReq as u32 || payload_type == P::ProtoOaGetTrendbarsReq as u32
}

#[derive(Clone, Debug, PartialEq)]
pub struct Frame {
    pub payload_type: u32,
    pub payload: Vec<u8>,
    pub client_msg_id: Option<String>,
}

impl Frame {
    pub fn encode(&self) -> Vec<u8> {
        let m = pb::ProtoMessage { payload_type: self.payload_type, payload: Some(self.payload.clone()), client_msg_id: self.client_msg_id.clone() };
        let body = m.encode_to_vec();
        let mut out = Vec::with_capacity(4 + body.len());
        out.extend_from_slice(&(body.len() as u32).to_be_bytes());
        out.extend_from_slice(&body);
        out
    }

    /// Decode the `ProtoMessage` body of one frame (without the length prefix).
    pub fn decode_body(body: &[u8]) -> anyhow::Result<Self> {
        let m = pb::ProtoMessage::decode(body)?;
        Ok(Self { payload_type: m.payload_type, payload: m.payload.unwrap_or_default(), client_msg_id: m.client_msg_id })
    }

    pub fn of<M: OaMsg>(m: &M, client_msg_id: Option<String>) -> Self {
        Self { payload_type: M::PAYLOAD, payload: m.encode_to_vec(), client_msg_id }
    }

    pub fn parse<M: OaMsg>(&self) -> anyhow::Result<M> {
        anyhow::ensure!(self.payload_type == M::PAYLOAD, "expected payload type {}, got {}", M::PAYLOAD, self.payload_type);
        Ok(M::decode(self.payload.as_slice())?)
    }

    pub fn is<M: OaMsg>(&self) -> bool {
        self.payload_type == M::PAYLOAD
    }

    /// The error carried by this frame, if it is an Open API or common error response.
    pub fn error(&self) -> Option<OaError> {
        if self.payload_type == pb::ProtoOaErrorRes::PAYLOAD {
            let e = pb::ProtoOaErrorRes::decode(self.payload.as_slice()).ok()?;
            return Some(OaError { code: e.error_code, description: e.description.unwrap_or_default(), retry_after_s: e.retry_after });
        }
        if self.payload_type == pb::ProtoErrorRes::PAYLOAD {
            let e = pb::ProtoErrorRes::decode(self.payload.as_slice()).ok()?;
            return Some(OaError { code: e.error_code, description: e.description.unwrap_or_default(), retry_after_s: None });
        }
        None
    }
}

/// Split complete frames off the front of `buf`; returns the frames and leaves the partial tail in `buf`.
pub fn split_frames(buf: &mut Vec<u8>) -> anyhow::Result<Vec<Frame>> {
    let mut out = Vec::new();
    let mut pos = 0;
    while buf.len() - pos >= 4 {
        let n = u32::from_be_bytes(buf[pos..pos + 4].try_into().expect("4 bytes")) as usize;
        anyhow::ensure!(n <= MAX_FRAME, "frame of {n} bytes exceeds the {MAX_FRAME} limit");
        if buf.len() - pos - 4 < n {
            break;
        }
        out.push(Frame::decode_body(&buf[pos + 4..pos + 4 + n])?);
        pos += 4 + n;
    }
    buf.drain(..pos);
    Ok(out)
}

#[derive(Clone, Debug, PartialEq, thiserror::Error)]
#[error("cTrader error {code}: {description}")]
pub struct OaError {
    pub code: String,
    pub description: String,
    pub retry_after_s: Option<u64>,
}

/// A protobuf message with a fixed payload type.
pub trait OaMsg: Message + Default {
    const PAYLOAD: u32;
}

/// A request the client may send, with its response type.
pub trait OaRequest: OaMsg {
    type Response: OaMsg;
}

macro_rules! oa_msg {
    ($($t:ident = $p:expr),* $(,)?) => { $(impl OaMsg for pb::$t { const PAYLOAD: u32 = $p as u32; })* };
}
macro_rules! oa_req {
    ($($req:ident => $res:ident),* $(,)?) => { $(impl OaRequest for pb::$req { type Response = pb::$res; })* };
}

oa_msg! {
    ProtoHeartbeatEvent = pb::ProtoPayloadType::HeartbeatEvent,
    ProtoErrorRes = pb::ProtoPayloadType::ErrorRes,
    ProtoOaErrorRes = P::ProtoOaErrorRes,
    ProtoOaApplicationAuthReq = P::ProtoOaApplicationAuthReq,
    ProtoOaApplicationAuthRes = P::ProtoOaApplicationAuthRes,
    ProtoOaAccountAuthReq = P::ProtoOaAccountAuthReq,
    ProtoOaAccountAuthRes = P::ProtoOaAccountAuthRes,
    ProtoOaVersionReq = P::ProtoOaVersionReq,
    ProtoOaVersionRes = P::ProtoOaVersionRes,
    ProtoOaAssetListReq = P::ProtoOaAssetListReq,
    ProtoOaAssetListRes = P::ProtoOaAssetListRes,
    ProtoOaSymbolsListReq = P::ProtoOaSymbolsListReq,
    ProtoOaSymbolsListRes = P::ProtoOaSymbolsListRes,
    ProtoOaSymbolByIdReq = P::ProtoOaSymbolByIdReq,
    ProtoOaSymbolByIdRes = P::ProtoOaSymbolByIdRes,
    ProtoOaTraderReq = P::ProtoOaTraderReq,
    ProtoOaTraderRes = P::ProtoOaTraderRes,
    ProtoOaSubscribeSpotsReq = P::ProtoOaSubscribeSpotsReq,
    ProtoOaSubscribeSpotsRes = P::ProtoOaSubscribeSpotsRes,
    ProtoOaUnsubscribeSpotsReq = P::ProtoOaUnsubscribeSpotsReq,
    ProtoOaUnsubscribeSpotsRes = P::ProtoOaUnsubscribeSpotsRes,
    ProtoOaSpotEvent = P::ProtoOaSpotEvent,
    ProtoOaGetTickDataReq = P::ProtoOaGetTickdataReq,
    ProtoOaGetTickDataRes = P::ProtoOaGetTickdataRes,
    ProtoOaGetTrendbarsReq = P::ProtoOaGetTrendbarsReq,
    ProtoOaGetTrendbarsRes = P::ProtoOaGetTrendbarsRes,
    ProtoOaGetAccountListByAccessTokenReq = P::ProtoOaGetAccountsByAccessTokenReq,
    ProtoOaGetAccountListByAccessTokenRes = P::ProtoOaGetAccountsByAccessTokenRes,
    ProtoOaSymbolCategoryListReq = P::ProtoOaSymbolCategoryReq,
    ProtoOaSymbolCategoryListRes = P::ProtoOaSymbolCategoryRes,
    ProtoOaAccountsTokenInvalidatedEvent = P::ProtoOaAccountsTokenInvalidatedEvent,
    ProtoOaAccountDisconnectEvent = P::ProtoOaAccountDisconnectEvent,
    ProtoOaClientDisconnectEvent = P::ProtoOaClientDisconnectEvent,
}

oa_req! {
    ProtoOaApplicationAuthReq => ProtoOaApplicationAuthRes,
    ProtoOaAccountAuthReq => ProtoOaAccountAuthRes,
    ProtoOaVersionReq => ProtoOaVersionRes,
    ProtoOaAssetListReq => ProtoOaAssetListRes,
    ProtoOaSymbolsListReq => ProtoOaSymbolsListRes,
    ProtoOaSymbolByIdReq => ProtoOaSymbolByIdRes,
    ProtoOaTraderReq => ProtoOaTraderRes,
    ProtoOaSubscribeSpotsReq => ProtoOaSubscribeSpotsRes,
    ProtoOaUnsubscribeSpotsReq => ProtoOaUnsubscribeSpotsRes,
    ProtoOaGetTickDataReq => ProtoOaGetTickDataRes,
    ProtoOaGetTrendbarsReq => ProtoOaGetTrendbarsRes,
    ProtoOaGetAccountListByAccessTokenReq => ProtoOaGetAccountListByAccessTokenRes,
    ProtoOaSymbolCategoryListReq => ProtoOaSymbolCategoryListRes,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frame_roundtrip_and_split() {
        let req = pb::ProtoOaVersionReq { payload_type: None };
        let f = Frame::of(&req, Some("7".into()));
        let mut buf = f.encode();
        let tail = Frame::of(&pb::ProtoHeartbeatEvent { payload_type: None }, None).encode();
        buf.extend_from_slice(&tail[..3]); // partial second frame
        let got = split_frames(&mut buf).unwrap();
        assert_eq!(got, vec![f]);
        assert_eq!(buf.len(), 3);
        buf.extend_from_slice(&tail[3..]);
        let got = split_frames(&mut buf).unwrap();
        assert!(got[0].is::<pb::ProtoHeartbeatEvent>());
        assert!(buf.is_empty());
    }

    #[test]
    fn trading_payloads_are_not_sendable() {
        for p in
            [P::ProtoOaNewOrderReq, P::ProtoOaCancelOrderReq, P::ProtoOaAmendOrderReq, P::ProtoOaAmendPositionSltpReq, P::ProtoOaClosePositionReq, P::ProtoOaMarginCallUpdateReq]
        {
            assert!(!SENDABLE.contains(&(p as i32)), "{p:?} must not be sendable");
        }
        assert!(SENDABLE.contains(&(P::ProtoOaGetTickdataReq as i32)));
        assert!(SENDABLE.contains(&(pb::ProtoOaGetTrendbarsReq::PAYLOAD as i32)));
        assert!(is_historical(pb::ProtoOaGetTrendbarsReq::PAYLOAD), "trend bars share the historical rate limit");
    }

    #[test]
    fn error_frames_decode() {
        let e = pb::ProtoOaErrorRes {
            payload_type: None,
            ctid_trader_account_id: None,
            error_code: "BLOCKED_PAYLOAD_TYPE".into(),
            description: Some("rate".into()),
            maintenance_end_timestamp: None,
            retry_after: Some(2),
        };
        let f = Frame::of(&e, Some("1".into()));
        let err = f.error().unwrap();
        assert_eq!(err.code, "BLOCKED_PAYLOAD_TYPE");
        assert_eq!(err.retry_after_s, Some(2));
        assert!(Frame::of(&pb::ProtoHeartbeatEvent { payload_type: None }, None).error().is_none());
    }
}
