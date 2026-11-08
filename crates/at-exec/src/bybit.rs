use std::collections::{HashMap, HashSet};

use anyhow::Context;
use at_bybit::BybitRest;
use at_bybit::trading::{OrderRequest, Trading};
use at_core::config::Settings;
use at_core::num::format_step;
use at_core::types::{EntryType, InstrumentRules, Position, Side};
use at_risk::SizedOrder;

use crate::Broker;

pub struct BybitBroker {
    rest: BybitRest,
    category: String,
    settle_coin: String,
    rules: HashMap<String, InstrumentRules>,
    stop_trigger_by: String,
    tp_trigger_by: String,
    time_in_force: String,
    leverage: String,
    leverage_set: HashSet<String>,
    label: String,
}

impl BybitBroker {
    /// Refuses to construct a mainnet broker unless both mainnet keys are turned (config + env var).
    pub fn new(settings: &Settings, rest: BybitRest, rules: HashMap<String, InstrumentRules>) -> anyhow::Result<Self> {
        settings.ensure_execution_allowed()?;
        anyhow::ensure!(rest.has_credentials(), "Bybit broker needs API credentials for {}", settings.environment.as_str());
        let e = &settings.execution;
        Ok(Self {
            rest,
            category: settings.venues.bybit.category.clone(),
            settle_coin: settings.venues.bybit.settle_coin.clone(),
            rules,
            stop_trigger_by: e.stop_trigger_by.clone(),
            tp_trigger_by: e.tp_trigger_by.clone(),
            time_in_force: e.limit_time_in_force.clone(),
            leverage: format!("{}", e.default_leverage),
            leverage_set: HashSet::new(),
            label: format!("bybit-{}", settings.environment.as_str()),
        })
    }

    fn trading(&self) -> Trading<'_> {
        Trading::new(&self.rest, &self.category, &self.settle_coin)
    }

    fn rule(&self, symbol: &str) -> anyhow::Result<&InstrumentRules> {
        self.rules.get(symbol).with_context(|| format!("no instrument rules for {symbol}"))
    }
}

impl Broker for BybitBroker {
    fn name(&self) -> &str {
        &self.label
    }

    async fn equity(&self) -> anyhow::Result<f64> {
        Ok(self.trading().wallet().await?.total_equity)
    }

    async fn positions(&self) -> anyhow::Result<Vec<Position>> {
        Ok(self
            .trading()
            .positions()
            .await?
            .into_iter()
            .map(|p| Position {
                side: if p.side == "Sell" { Side::Short } else { Side::Long },
                symbol: p.symbol,
                qty: p.size,
                entry_price: p.avg_price,
                stop: p.stop_loss,
                take_profit: p.take_profit,
                cluster: String::new(),
                opened_ts: p.created_ts,
            })
            .collect())
    }

    async fn submit(&mut self, o: &SizedOrder, link_id: &str) -> anyhow::Result<String> {
        let sym = o.proposal.symbol.clone();
        let rule = self.rule(&sym)?.clone();
        if !self.leverage_set.contains(&sym) {
            let lev = rule.max_leverage.min(self.leverage.parse().unwrap_or(1.0));
            self.trading().set_leverage(&sym, &format!("{lev}")).await?;
            self.leverage_set.insert(sym.clone());
        }
        let (order_type, price) = match o.proposal.entry_type {
            EntryType::Market => ("Market", None),
            EntryType::Limit => ("Limit", Some(format_step(o.entry, rule.tick_size))),
        };
        let req = OrderRequest {
            symbol: sym,
            side: o.proposal.side.bybit_open(),
            order_type,
            qty: format_step(o.qty, rule.qty_step),
            price,
            time_in_force: self.time_in_force.clone(),
            take_profit: Some(format_step(o.take_profit, rule.tick_size)),
            stop_loss: Some(format_step(o.stop, rule.tick_size)),
            tp_trigger_by: self.tp_trigger_by.clone(),
            sl_trigger_by: self.stop_trigger_by.clone(),
            reduce_only: false,
            order_link_id: link_id.chars().take(36).collect(),
        };
        Ok(self.trading().create_order(&req).await?.order_id)
    }

    async fn move_stop(&mut self, symbol: &str, stop: f64) -> anyhow::Result<()> {
        let tick = self.rule(symbol)?.tick_size;
        self.trading().set_trading_stop(symbol, Some(&format_step(stop, tick)), None, &self.stop_trigger_by).await
    }

    async fn close(&mut self, symbol: &str) -> anyhow::Result<()> {
        let step = self.rule(symbol)?.qty_step;
        let positions = self.trading().positions().await?;
        let Some(p) = positions.iter().find(|p| p.symbol == symbol) else { return Ok(()) };
        let side = if p.side == "Sell" { Side::Short } else { Side::Long };
        let req = OrderRequest {
            symbol: symbol.to_string(),
            side: side.bybit_close(),
            order_type: "Market",
            qty: format_step(p.size, step),
            price: None,
            time_in_force: "IOC".into(),
            take_profit: None,
            stop_loss: None,
            tp_trigger_by: self.tp_trigger_by.clone(),
            sl_trigger_by: self.stop_trigger_by.clone(),
            reduce_only: true,
            order_link_id: format!("close-{symbol}-{}", at_core::time::now_ms()).chars().take(36).collect(),
        };
        self.trading().create_order(&req).await.map(|_| ())
    }
}
