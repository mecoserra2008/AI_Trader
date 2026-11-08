//! Private endpoints: account, positions, orders, stops. One-way position mode (positionIdx = 0).

use anyhow::Context;
use serde_json::{Value, json};

use crate::rest::{BybitRest, venue_code};

/// "leverage not modified": setting the current value again is not an error for us.
const LEVERAGE_NOT_MODIFIED: i64 = 110043;

#[derive(Clone, Debug)]
pub struct VenuePosition {
    pub symbol: String,
    pub side: String,
    pub size: f64,
    pub avg_price: f64,
    pub mark_price: f64,
    pub stop_loss: f64,
    pub take_profit: f64,
    pub unrealised_pnl: f64,
    pub created_ts: i64,
}

#[derive(Clone, Debug)]
pub struct WalletSummary {
    pub total_equity: f64,
    pub total_available: f64,
    pub total_margin_balance: f64,
}

/// Order request with prices/quantities already rounded to the instrument grid and formatted.
#[derive(Clone, Debug)]
pub struct OrderRequest {
    pub symbol: String,
    pub side: &'static str,
    pub order_type: &'static str,
    pub qty: String,
    pub price: Option<String>,
    pub time_in_force: String,
    pub take_profit: Option<String>,
    pub stop_loss: Option<String>,
    pub tp_trigger_by: String,
    pub sl_trigger_by: String,
    pub reduce_only: bool,
    pub order_link_id: String,
}

impl OrderRequest {
    pub fn to_body(&self, category: &str) -> Value {
        let mut b = json!({
            "category": category,
            "symbol": self.symbol,
            "side": self.side,
            "orderType": self.order_type,
            "qty": self.qty,
            "positionIdx": 0,
            "reduceOnly": self.reduce_only,
            "orderLinkId": self.order_link_id,
        });
        if let Some(p) = &self.price {
            b["price"] = json!(p);
            b["timeInForce"] = json!(self.time_in_force);
        }
        if self.take_profit.is_some() || self.stop_loss.is_some() {
            b["tpslMode"] = json!("Full");
        }
        if let Some(tp) = &self.take_profit {
            b["takeProfit"] = json!(tp);
            b["tpTriggerBy"] = json!(self.tp_trigger_by);
        }
        if let Some(sl) = &self.stop_loss {
            b["stopLoss"] = json!(sl);
            b["slTriggerBy"] = json!(self.sl_trigger_by);
        }
        b
    }
}

#[derive(Clone, Debug)]
pub struct OrderAck {
    pub order_id: String,
    pub order_link_id: String,
}

fn f(v: &Value) -> f64 {
    v.as_str().and_then(|s| s.parse().ok()).unwrap_or(0.0)
}

pub struct Trading<'a> {
    rest: &'a BybitRest,
    category: String,
    settle_coin: String,
}

impl<'a> Trading<'a> {
    pub fn new(rest: &'a BybitRest, category: &str, settle_coin: &str) -> Self {
        Self { rest, category: category.to_string(), settle_coin: settle_coin.to_string() }
    }

    pub async fn wallet(&self) -> anyhow::Result<WalletSummary> {
        let r = self.rest.get_private("/v5/account/wallet-balance", &[("accountType", "UNIFIED".to_string())]).await?;
        let a = r["list"].get(0).context("wallet: empty list")?;
        Ok(WalletSummary {
            total_equity: f(&a["totalEquity"]),
            total_available: f(&a["totalAvailableBalance"]),
            total_margin_balance: f(&a["totalMarginBalance"]),
        })
    }

    pub async fn positions(&self) -> anyhow::Result<Vec<VenuePosition>> {
        let mut out = Vec::new();
        let mut cursor = String::new();
        loop {
            let mut p = vec![("category", self.category.clone()), ("settleCoin", self.settle_coin.clone()), ("limit", "200".to_string())];
            if !cursor.is_empty() {
                p.push(("cursor", cursor.clone()));
            }
            let r = self.rest.get_private("/v5/position/list", &p).await?;
            for x in r["list"].as_array().cloned().unwrap_or_default() {
                let size = f(&x["size"]);
                if size == 0.0 {
                    continue;
                }
                out.push(VenuePosition {
                    symbol: x["symbol"].as_str().unwrap_or_default().to_string(),
                    side: x["side"].as_str().unwrap_or_default().to_string(),
                    size,
                    avg_price: f(&x["avgPrice"]),
                    mark_price: f(&x["markPrice"]),
                    stop_loss: f(&x["stopLoss"]),
                    take_profit: f(&x["takeProfit"]),
                    unrealised_pnl: f(&x["unrealisedPnl"]),
                    created_ts: x["createdTime"].as_str().and_then(|s| s.parse().ok()).unwrap_or(0),
                });
            }
            cursor = r["nextPageCursor"].as_str().unwrap_or_default().to_string();
            if cursor.is_empty() {
                break;
            }
        }
        Ok(out)
    }

    pub async fn set_leverage(&self, symbol: &str, leverage: &str) -> anyhow::Result<()> {
        let body = json!({"category": self.category, "symbol": symbol, "buyLeverage": leverage, "sellLeverage": leverage});
        match self.rest.post_private("/v5/position/set-leverage", &body).await {
            Ok(_) => Ok(()),
            Err(e) if venue_code(&e) == Some(LEVERAGE_NOT_MODIFIED) => Ok(()),
            Err(e) => Err(e),
        }
    }

    pub async fn create_order(&self, req: &OrderRequest) -> anyhow::Result<OrderAck> {
        let r = self.rest.post_private("/v5/order/create", &req.to_body(&self.category)).await?;
        Ok(OrderAck {
            order_id: r["orderId"].as_str().unwrap_or_default().to_string(),
            order_link_id: r["orderLinkId"].as_str().unwrap_or_default().to_string(),
        })
    }

    /// Move the position-level stop and/or target (used by the structural trailing rule).
    pub async fn set_trading_stop(&self, symbol: &str, stop_loss: Option<&str>, take_profit: Option<&str>, sl_trigger_by: &str) -> anyhow::Result<()> {
        let mut body = json!({"category": self.category, "symbol": symbol, "tpslMode": "Full", "positionIdx": 0});
        if let Some(sl) = stop_loss {
            body["stopLoss"] = json!(sl);
            body["slTriggerBy"] = json!(sl_trigger_by);
        }
        if let Some(tp) = take_profit {
            body["takeProfit"] = json!(tp);
        }
        self.rest.post_private("/v5/position/trading-stop", &body).await.map(|_| ())
    }

    /// Change price and/or quantity of a resting order in place (keeps its id; loses queue priority on price).
    pub async fn amend_order(&self, symbol: &str, order_id: &str, price: Option<&str>, qty: Option<&str>) -> anyhow::Result<()> {
        let mut body = json!({"category": self.category, "symbol": symbol, "orderId": order_id});
        if let Some(p) = price {
            body["price"] = json!(p);
        }
        if let Some(q) = qty {
            body["qty"] = json!(q);
        }
        self.rest.post_private("/v5/order/amend", &body).await.map(|_| ())
    }

    pub async fn cancel_order(&self, symbol: &str, order_id: &str) -> anyhow::Result<()> {
        let body = json!({"category": self.category, "symbol": symbol, "orderId": order_id});
        self.rest.post_private("/v5/order/cancel", &body).await.map(|_| ())
    }

    pub async fn cancel_all(&self, symbol: &str) -> anyhow::Result<()> {
        let body = json!({"category": self.category, "symbol": symbol});
        self.rest.post_private("/v5/order/cancel-all", &body).await.map(|_| ())
    }

    pub async fn open_orders(&self, symbol: Option<&str>) -> anyhow::Result<Vec<Value>> {
        let mut p = vec![("category", self.category.clone())];
        match symbol {
            Some(s) => p.push(("symbol", s.to_string())),
            None => p.push(("settleCoin", self.settle_coin.clone())),
        }
        let r = self.rest.get_private("/v5/order/realtime", &p).await?;
        Ok(r["list"].as_array().cloned().unwrap_or_default())
    }

    pub async fn executions(&self, start_ms: i64, end_ms: i64) -> anyhow::Result<Vec<Value>> {
        let mut out = Vec::new();
        let mut cursor = String::new();
        loop {
            let mut p = vec![
                ("category", self.category.clone()),
                ("startTime", start_ms.to_string()),
                ("endTime", end_ms.to_string()),
                ("limit", "100".to_string()),
            ];
            if !cursor.is_empty() {
                p.push(("cursor", cursor.clone()));
            }
            let r = self.rest.get_private("/v5/execution/list", &p).await?;
            out.extend(r["list"].as_array().cloned().unwrap_or_default());
            cursor = r["nextPageCursor"].as_str().unwrap_or_default().to_string();
            if cursor.is_empty() {
                break;
            }
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bracket_body_shape() {
        let req = OrderRequest {
            symbol: "BTCUSDT".into(),
            side: "Buy",
            order_type: "Limit",
            qty: "0.010".into(),
            price: Some("75000.0".into()),
            time_in_force: "GTC".into(),
            take_profit: Some("79000.0".into()),
            stop_loss: Some("73500.0".into()),
            tp_trigger_by: "LastPrice".into(),
            sl_trigger_by: "MarkPrice".into(),
            reduce_only: false,
            order_link_id: "s1-BTCUSDT".into(),
        };
        let b = req.to_body("linear");
        assert_eq!(b["tpslMode"], "Full");
        assert_eq!(b["slTriggerBy"], "MarkPrice");
        assert_eq!(b["positionIdx"], 0);
        assert_eq!(b["timeInForce"], "GTC");
        assert_eq!(b["qty"], "0.010");
    }
}
