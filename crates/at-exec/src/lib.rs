//! Execution: one [`Broker`] interface, two implementations.
//!
//! - [`bybit::BybitBroker`]: Bybit v5 (demo / testnet / mainnet), bracket orders with position-level TP/SL
//! - [`paper::PaperBroker`]: deterministic bar-driven simulator, used for the paired baselines
//!   (RANDOM_N, RULE_N) that run alongside the AI on the same candidate list.

#![allow(async_fn_in_trait)]

pub mod bybit;
pub mod paper;

use at_core::types::Position;
use at_risk::SizedOrder;

pub trait Broker {
    fn name(&self) -> &str;
    async fn equity(&self) -> anyhow::Result<f64>;
    async fn positions(&self) -> anyhow::Result<Vec<Position>>;
    /// Submit an entry with attached stop and target. Returns the venue order id.
    async fn submit(&mut self, order: &SizedOrder, link_id: &str) -> anyhow::Result<String>;
    /// Move the stop of an open position (structural trailing rule).
    async fn move_stop(&mut self, symbol: &str, stop: f64) -> anyhow::Result<()>;
    /// Close the whole position at market.
    async fn close(&mut self, symbol: &str) -> anyhow::Result<()>;
}
