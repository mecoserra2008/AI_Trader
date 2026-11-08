//! Bybit v5 connectivity.
//!
//! - [`rest`]: signed/unsigned HTTP client with rate limiting and retries
//! - [`market`]: public market data (instruments, klines, funding, OI, long/short ratio, tickers)
//! - [`archive`]: historical trade files from public.bybit.com, aggregated into order-flow bars
//! - [`trading`]: private account / order / position endpoints
//! - [`ws`]: public and private WebSocket streams with heartbeat and reconnect
//! - [`lake`]: conversion of venue records into lake batches (known_ts rules live here)

pub mod archive;
pub mod lake;
pub mod market;
pub mod rest;
pub mod sign;
pub mod trading;
pub mod ws;

pub use rest::{BybitRest, Credentials};

pub const VENUE: &str = "bybit";
