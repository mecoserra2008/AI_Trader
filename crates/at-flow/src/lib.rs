//! Order-flow collection: live L2 books + trades aggregated into 3-second candles.
//!
//! Storage principle: only the 3-second base candles are stored (`flow_3s`). Every other view
//! (1-minute rollups, SMA/EMA stacks over many windows, z-scores) is *derived* from them, so
//! storage grows with the base layer only. See docs/ORDER_FLOW.md.
//!
//! - [`book`]: L2 order book from snapshot + delta messages; banded depth / imbalance features
//! - [`candle`]: per-symbol 3 s aggregator (trades, time-weighted book features, L1 OFI)
//! - [`stack`]: O(1) SMA / EMA filters used to derive multi-window feature stacks
//! - [`collector`]: websocket fan-in, candle finalisation on exchange time, periodic lake flushes

pub mod book;
pub mod candle;
pub mod collector;
pub mod stack;

/// Depth bands (bps from mid) for banded imbalance / depth. Fixed: they are schema columns.
pub const BANDS_BPS: [f64; 5] = [5.0, 10.0, 25.0, 50.0, 100.0];
