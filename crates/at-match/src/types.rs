//! Order, event and reason types. Prices are integer ticks and quantities integer lots (qty steps) inside the engine;
//! the [`crate::Grid`] converts to and from venue decimals.

use serde::{Deserialize, Serialize};

/// Engine order id. The JSONL driver maps external ids onto it.
pub type OrderId = u64;
/// Engine owner id (dense small integers; the JSONL driver interns owner names). Owner 0 is the anonymous owner.
pub type OwnerId = u32;

/// Largest owner id the engine accepts (owner state is a dense vector).
pub const MAX_OWNER: OwnerId = 1 << 20;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Side {
    Buy,
    Sell,
}

impl Side {
    /// +1 for Buy, -1 for Sell.
    #[inline]
    pub fn sign(self) -> i64 {
        match self {
            Side::Buy => 1,
            Side::Sell => -1,
        }
    }

    #[inline]
    pub fn opposite(self) -> Side {
        match self {
            Side::Buy => Side::Sell,
            Side::Sell => Side::Buy,
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Side::Buy => "Buy",
            Side::Sell => "Sell",
        }
    }

    pub fn parse(s: &str) -> Option<Side> {
        match s {
            "Buy" | "buy" | "BUY" | "b" | "B" => Some(Side::Buy),
            "Sell" | "sell" | "SELL" | "s" | "S" => Some(Side::Sell),
            _ => None,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum OrdType {
    Limit,
    Market,
}

impl OrdType {
    pub fn parse(s: &str) -> Option<OrdType> {
        match s {
            "Limit" | "limit" | "LIMIT" => Some(OrdType::Limit),
            "Market" | "market" | "MARKET" => Some(OrdType::Market),
            _ => None,
        }
    }
}

/// Time in force (Bybit v5 `timeInForce`; `RPI` is not modelled).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Tif {
    GTC,
    IOC,
    FOK,
    PostOnly,
}

impl Tif {
    pub fn parse(s: &str) -> Option<Tif> {
        match s {
            "GTC" | "gtc" | "GoodTillCancel" => Some(Tif::GTC),
            "IOC" | "ioc" | "ImmediateOrCancel" => Some(Tif::IOC),
            "FOK" | "fok" | "FillOrKill" => Some(Tif::FOK),
            "PostOnly" | "postonly" | "post_only" | "POSTONLY" => Some(Tif::PostOnly),
            _ => None,
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Tif::GTC => "GTC",
            Tif::IOC => "IOC",
            Tif::FOK => "FOK",
            Tif::PostOnly => "PostOnly",
        }
    }
}

/// Market-order price protection (Bybit `slippageToleranceType` / `slippageTolerance`): a market Buy executes as an
/// IOC limit at `ask1 * (1 + pct/100)` (Percent) or `ask1 + n * tick` (TickSize); a Sell symmetrically from bid1.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub enum Band {
    Percent(f64),
    Ticks(i64),
}

/// A new order as submitted (venue units; the engine validates and converts them).
#[derive(Clone, Debug, PartialEq)]
pub struct NewOrder {
    pub ts: i64,
    pub id: OrderId,
    pub owner: OwnerId,
    pub side: Side,
    pub ord_type: OrdType,
    pub tif: Tif,
    /// Limit price; ignored for market orders.
    pub price: Option<f64>,
    pub qty: f64,
    pub reduce_only: bool,
    /// Per-order market protection; `None` uses the engine default.
    pub band: Option<Band>,
}

impl NewOrder {
    pub fn limit(ts: i64, id: OrderId, owner: OwnerId, side: Side, tif: Tif, price: f64, qty: f64) -> Self {
        Self { ts, id, owner, side, ord_type: OrdType::Limit, tif, price: Some(price), qty, reduce_only: false, band: None }
    }

    pub fn market(ts: i64, id: OrderId, owner: OwnerId, side: Side, qty: f64) -> Self {
        Self { ts, id, owner, side, ord_type: OrdType::Market, tif: Tif::IOC, price: None, qty, reduce_only: false, band: None }
    }

    pub fn reduce_only(mut self) -> Self {
        self.reduce_only = true;
        self
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub enum Status {
    New,
    PartiallyFilled,
    Filled,
    Cancelled,
}

impl Status {
    pub fn as_str(self) -> &'static str {
        match self {
            Status::New => "New",
            Status::PartiallyFilled => "PartiallyFilled",
            Status::Filled => "Filled",
            Status::Cancelled => "Cancelled",
        }
    }

    pub fn is_closed(self) -> bool {
        matches!(self, Status::Filled | Status::Cancelled)
    }
}

/// Operation a reject refers to.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub enum Op {
    New,
    Cancel,
    Amend,
}

impl Op {
    pub fn as_str(self) -> &'static str {
        match self {
            Op::New => "new",
            Op::Cancel => "cancel",
            Op::Amend => "amend",
        }
    }
}

/// Named reasons. `EC_*` names are Bybit v5 `rejectReason` enum values and `CancelBy*` are Bybit `cancelType`
/// values; the others are engine names for rules whose Bybit code we did not confirm (see the crate docs).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize)]
pub enum Reason {
    // rejects of a new order
    PostOnlyWillTakeLiquidity,
    CancelForNoFullFill,
    MarketOrderCannotBePostOnly,
    LimitOrderInvalidPrice,
    InvalidPriceScale,
    QtyCannotBeZero,
    InvalidQty,
    QtyBelowMin,
    QtyAboveMax,
    NotionalBelowMin,
    ReduceOnlyRuleNotSatisfied,
    DuplicatedId,
    // rejects of cancel / amend
    OrderNotExist,
    OrderNotModified,
    AmendQtyNotAboveFilled,
    // malformed input (driver only)
    BadInput,
    // order updates
    NoImmediateQtyToFill,
    IocRemainderCancelled,
    CancelByUser,
    CancelByReduceOnly,
    ReduceOnlyQtyAdjusted,
    Amended,
}

impl Reason {
    pub fn as_str(self) -> &'static str {
        match self {
            Reason::PostOnlyWillTakeLiquidity => "EC_PostOnlyWillTakeLiquidity",
            Reason::CancelForNoFullFill => "EC_CancelForNoFullFill",
            Reason::MarketOrderCannotBePostOnly => "EC_MarketOrderCannotBePostOnly",
            Reason::LimitOrderInvalidPrice => "EC_LimitOrderInvalidPrice",
            Reason::InvalidPriceScale => "EC_InvalidPriceScale",
            Reason::QtyCannotBeZero => "EC_QtyCannotBeZero",
            Reason::InvalidQty => "EC_EcInvalidQty",
            Reason::QtyBelowMin => "QtyBelowMinOrderQty",
            Reason::QtyAboveMax => "QtyAboveMaxOrderQty",
            Reason::NotionalBelowMin => "OrderValueBelowMinNotional",
            Reason::ReduceOnlyRuleNotSatisfied => "ReduceOnlyRuleNotSatisfied",
            Reason::DuplicatedId => "EC_DuplicatedClOrdID",
            Reason::OrderNotExist => "EC_OrderNotExist",
            Reason::OrderNotModified => "OrderNotModified",
            Reason::AmendQtyNotAboveFilled => "AmendQtyNotAboveFilled",
            Reason::BadInput => "BadInput",
            Reason::NoImmediateQtyToFill => "EC_NoImmediateQtyToFill",
            Reason::IocRemainderCancelled => "IocRemainderCancelled",
            Reason::CancelByUser => "CancelByUser",
            Reason::CancelByReduceOnly => "CancelByReduceOnly",
            Reason::ReduceOnlyQtyAdjusted => "ReduceOnlyQtyAdjusted",
            Reason::Amended => "Amended",
        }
    }
}

/// One execution. `taker_side` is the aggressor's side (Bybit publicTrade `S`). Price in ticks, qty in lots, fees in
/// quote currency (positive = paid).
#[derive(Clone, Debug, PartialEq)]
pub struct Trade {
    pub seq: u64,
    pub ts: i64,
    pub price: i64,
    pub qty: i64,
    pub taker_side: Side,
    pub taker_id: OrderId,
    pub maker_id: OrderId,
    pub taker_owner: OwnerId,
    pub maker_owner: OwnerId,
    pub taker_fee: f64,
    pub maker_fee: f64,
}

/// Order state after an operation. `qty` is the total order qty (including filled), `leaves` the open remainder,
/// `cum` the filled qty; price is the (effective) limit price in ticks, 0 for a market order that found no book.
#[derive(Clone, Debug, PartialEq)]
pub struct OrderUpdate {
    pub ts: i64,
    pub id: OrderId,
    pub owner: OwnerId,
    pub side: Side,
    pub price: i64,
    pub qty: i64,
    pub leaves: i64,
    pub cum: i64,
    pub status: Status,
    pub reason: Option<Reason>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Reject {
    pub ts: i64,
    pub id: OrderId,
    pub owner: OwnerId,
    pub op: Op,
    pub reason: Reason,
}

#[derive(Clone, Debug, PartialEq)]
pub enum Event {
    Trade(Trade),
    Order(OrderUpdate),
    Reject(Reject),
}
