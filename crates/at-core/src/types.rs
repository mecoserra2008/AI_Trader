//! Domain types shared by the risk engine, execution and connectors.

use serde::{Deserialize, Serialize};

use crate::time::Ms;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Side {
    Long,
    Short,
}

impl Side {
    /// +1 for long, -1 for short.
    pub fn sign(self) -> f64 {
        match self {
            Side::Long => 1.0,
            Side::Short => -1.0,
        }
    }

    /// Bybit order side that opens this position.
    pub fn bybit_open(self) -> &'static str {
        match self {
            Side::Long => "Buy",
            Side::Short => "Sell",
        }
    }

    /// Bybit order side that closes this position.
    pub fn bybit_close(self) -> &'static str {
        match self {
            Side::Long => "Sell",
            Side::Short => "Buy",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum EntryType {
    Market,
    Limit,
}

/// Venue trading constraints for one instrument.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct InstrumentRules {
    pub symbol: String,
    pub tick_size: f64,
    pub qty_step: f64,
    pub min_qty: f64,
    pub max_qty: f64,
    pub max_mkt_qty: f64,
    pub min_notional: f64,
    pub max_leverage: f64,
}

/// A trade the decision layer wants. Prices come from structural placement (ARCHITECTURE §3.11);
/// `p_tp` is the final posterior P(TP before SL). Size is never part of a proposal.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TradeProposal {
    pub id: String,
    pub symbol: String,
    pub side: Side,
    pub entry_type: EntryType,
    pub entry: f64,
    pub stop: f64,
    pub take_profit: f64,
    pub p_tp: f64,
    pub cluster: String,
    pub channel: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Position {
    pub symbol: String,
    pub side: Side,
    pub qty: f64,
    pub entry_price: f64,
    pub stop: f64,
    pub take_profit: f64,
    pub cluster: String,
    pub opened_ts: Ms,
}

impl Position {
    pub fn notional(&self, mark: f64) -> f64 {
        self.qty * mark
    }
}
