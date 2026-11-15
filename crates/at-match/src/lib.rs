//! # at-match: a deterministic Bybit-semantics limit order book (research simulator)
//!
//! A single-instrument matching engine with the *documented* semantics of Bybit v5 USDT linear perpetuals
//! (docs/MATCHING_ENGINE.md is the contract). It is a simulator: no clocks (all times come from the input), no
//! network, no exchange code. The same input always gives the same output.
//!
//! ```
//! use at_core::types::InstrumentRules;
//! use at_match::{NewOrder, OrderBook, Side, Tif};
//! use at_risk::FeeRates;
//! let rules = InstrumentRules { symbol: "BTCUSDT".into(), tick_size: 0.1, qty_step: 0.001, min_qty: 0.001,
//!     max_qty: 1190.0, max_mkt_qty: 119.0, min_notional: 5.0, max_leverage: 100.0 };
//! let fees = FeeRates { maker_bps: 2.0, taker_bps: 5.5, slippage_bps: 0.0 };
//! let mut book = OrderBook::new(&rules, fees).unwrap();
//! book.submit(&NewOrder::limit(0, 1, 1, Side::Sell, Tif::GTC, 75000.0, 0.01));
//! let ev = book.submit(&NewOrder::market(1, 2, 2, Side::Buy, 0.004));
//! assert_eq!(book.best_ask(), Some((750000, 6)));            // ticks, lots
//! # let _ = ev;
//! ```
//!
//! ## Semantics implemented
//!
//! - **Price-time priority.** Levels are FIFO queues. A partial fill keeps the order's place. An amendment that
//!   changes the price or increases the qty sends the order to the back of the (new) level's queue; a qty
//!   decrease keeps its place.
//! - **Limit orders**, time in force:
//!   - GTC: matches what crosses, rests the remainder;
//!   - IOC: matches what crosses, cancels the remainder (`EC_NoImmediateQtyToFill` if nothing filled, else
//!     `IocRemainderCancelled`);
//!   - FOK: fills completely or is rejected untouched (`EC_CancelForNoFullFill`); the pre-check walks the queue
//!     with the same reduce-only caps the match would apply, so it is exact;
//!   - PostOnly: rejected untouched with `EC_PostOnlyWillTakeLiquidity` if it would cross, else rests.
//! - **Market orders** are IOC limits at a protected price (Bybit market-order price protection): Buy limit =
//!   ask1 x (1 + pct/100) floored to the tick, or ask1 + n ticks; Sell from bid1 symmetrically. The band is an
//!   engine parameter ([`EngineConfig::market_band`]) and can be set per order (Bybit `slippageToleranceType` /
//!   `slippageTolerance`). Liquidity beyond the band is not taken; the remainder is cancelled.
//!   A market order with PostOnly is rejected (`EC_MarketOrderCannotBePostOnly`); any other `tif` is ignored
//!   (Bybit: market orders always use IOC).
//! - **Reduce-only** against a per-owner one-way position ledger (filled from the engine's own trades):
//!   - rejected (`ReduceOnlyRuleNotSatisfied`) when the owner has no position on the opposite side;
//!   - a qty above the position is cut to the position (`ReduceOnlyQtyAdjusted` on the order update);
//!   - after every change of position, the owner's resting reduce-only orders may not add up to more than the
//!     position: the order furthest from the market (ties: the newest) is reduced, or cancelled
//!     (`CancelByReduceOnly`). Orders that no longer reduce (position flat or flipped) are cancelled;
//!   - a resting reduce-only order is capped at fill time too, so it can never open or flip a position.
//! - **Instrument filters** (named rejects): `EC_QtyCannotBeZero`, `EC_EcInvalidQty` (negative / not a multiple of
//!   the qty step), `QtyBelowMinOrderQty`, `QtyAboveMaxOrderQty` (`max_qty` for limits, `max_mkt_qty` for
//!   market orders), `EC_LimitOrderInvalidPrice` (missing / non-positive price), `EC_InvalidPriceScale` (not a
//!   multiple of the tick), `OrderValueBelowMinNotional` (price x qty < min notional; market orders use the
//!   opposite best as the price), `EC_DuplicatedClOrdID` (id already live). Cancel / amend of an unknown or
//!   closed order: `EC_OrderNotExist`; an amend that changes nothing: `OrderNotModified`; an amend to a qty at
//!   or below the filled qty: `AmendQtyNotAboveFilled`.
//! - **Fees**: maker / taker bps of notional from `config/fees.yaml` (`at_risk::FeeSchedule`), taker = aggressor.
//!   An amended order that crosses is the taker.
//! - **Outputs**: [`Event`]s (trades with the aggressor side, order updates, rejects); L2 snapshot / delta
//!   messages in Bybit `orderbook.200` shape ([`l2`]); the JSONL driver ([`sim`]) behind `atx match-sim`.
//!
//! ## Assumptions (not confirmed from the Bybit v5 docs)
//!
//! 1. Amend priority: the v5 docs do not state queue priority on amend; we use the usual rule (price change or qty
//!    increase loses priority, qty decrease keeps it), as the contract asks.
//! 2. An amend whose new price crosses executes immediately as a taker (GTC); for a PostOnly order such an amend
//!    is rejected with `EC_PostOnlyWillTakeLiquidity` and the order stays unchanged.
//! 3. Off-grid price / qty are rejected rather than rounded; reject names without a confirmed Bybit code
//!    (`QtyBelowMinOrderQty`, `QtyAboveMaxOrderQty`, `OrderValueBelowMinNotional`, `ReduceOnlyRuleNotSatisfied`,
//!    `OrderNotModified`, `AmendQtyNotAboveFilled`, `IocRemainderCancelled`, `ReduceOnlyQtyAdjusted`) are ours.
//! 4. The default market band is 1% (Bybit's own default protection for linear perps is not published); market
//!    orders with an empty opposite side are cancelled unfilled (`EC_NoImmediateQtyToFill`).
//! 5. Reduce-only orders are exempt from the minimum notional and from `max_qty` (Bybit splits oversized
//!    reduce-only orders; we keep one order). The reduce-only trimming rule (furthest price first) follows the
//!    Bybit help-centre description of reduce-only orders.
//! 6. Self-matching is allowed (Bybit `smpType` default `None`); both legs pay fees and the position nets out.
//! 7. Ids must be unique among live orders; an id may be reused after its order closes.
//! 8. One-way position mode; no margin, leverage, liquidation, ADL, mark / index price, funding, price limits
//!    relative to the mark price, conditional / TP-SL orders, RPI orders, iceberg or hidden liquidity, or
//!    internal latency. These are Bybit internals or account features outside the documented matching
//!    semantics and are out of scope.
//! 9. Fees are not rounded (Bybit rounds fees to the settlement coin's precision); maker rebates (negative maker
//!    fees) are not modelled, as in `config/fees.yaml`.

pub mod book;
pub mod flow;
pub mod grid;
mod hash;
pub mod l2;
pub mod replay;
pub mod sim;
pub mod types;

pub use book::{EngineConfig, L2, Limits, OrderBook, OrderView, OwnerState};
pub use grid::Grid;
pub use l2::Publisher;
pub use replay::Ob200Replayer;
pub use sim::{InputEvent, Sim, SimConfig, SimStats, Sinks, run_files};
pub use types::*;

#[cfg(test)]
mod tests;
