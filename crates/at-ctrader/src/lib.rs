//! Read-only client for the Spotware cTrader Open API (protobuf over TLS, port 5035): auth, symbols, historical ticks
//! and trend bars, live spots.
//!
//! Wire format: every frame is a 4-byte big-endian length followed by a serialized `ProtoMessage`
//! `{payloadType, payload, clientMsgId}`; `payload` is the serialized inner message.
//!
//! Safety: this crate is for market data only. Only the request types in [`codec::SENDABLE`] can be sent
//! (no order, amend, cancel or close messages exist in the typed API, and the raw sender refuses them), and
//! [`session::Session::check`] refuses a token whose scope is `SCOPE_TRADE`.
//!
//! Message definitions: `proto/*.proto` from github.com/spotware/openapi-proto-messages (commit a9676f2,
//! 2026-11-04); `src/pb/openapi.rs` is the prost-build 0.13 output, checked in so builds need no protoc.

#[allow(clippy::all, clippy::pedantic, clippy::nursery, missing_docs, rustdoc::all)]
#[rustfmt::skip]
pub mod pb {
    include!("pb/openapi.rs");
}

pub mod codec;
pub mod creds;
pub mod flow;
pub mod session;
pub mod store;
pub mod ticks;
pub mod trendbars;

pub use codec::{Frame, OaError};
pub use creds::{CtraderCreds, CtraderEnv};
pub use session::Session;

/// cTrader prices on the wire are integers in 1/100000 of a unit.
pub const PRICE_SCALE: f64 = 100_000.0;

pub fn price(raw: i64) -> f64 {
    raw as f64 / PRICE_SCALE
}
