//! Deterministic risk engine. The LLM proposes *what* to trade; this decides *whether* and *how much*.
//!
//! Sizing: every position loses exactly `risk_per_trade_pct` of equity if the stop is hit,
//! **including** round-trip fees and slippage:
//!     qty = R / (|entry − stop| + entry · cost_frac)
//! Expected value is recomputed net of the same costs from the posterior P(TP first):
//!     E[R]_net = p · (|tp − entry| − entry · cost_frac) / (|entry − stop| + entry · cost_frac) − (1 − p)

use std::collections::HashMap;

use at_core::config::RiskConfig;
use at_core::num::{Round, round_to_step};
use at_core::types::{EntryType, InstrumentRules, Position, Side, TradeProposal};
use serde::Serialize;

#[derive(Clone, Debug, Serialize)]
pub struct AccountState {
    pub equity: f64,
    pub day_start_equity: f64,
    pub trades_opened_today: u32,
    pub open_positions: Vec<Position>,
}

#[derive(Clone, Debug, Serialize)]
pub struct SizedOrder {
    pub proposal: TradeProposal,
    /// Prices rounded to the tick grid (stop away from entry, target toward entry: conservative).
    pub entry: f64,
    pub stop: f64,
    pub take_profit: f64,
    pub qty: f64,
    pub notional: f64,
    pub risk_amount: f64,
    pub reward_risk: f64,
    pub cost_r: f64,
    pub net_expected_r: f64,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
#[serde(tag = "reason", rename_all = "snake_case")]
pub enum Rejection {
    DailyLossHalt { drawdown_pct: f64 },
    NoSlots,
    DuplicateSymbol,
    ClusterCap,
    UnknownInstrument,
    InvalidGeometry { detail: String },
    StopTooFar { pct: f64 },
    BelowMinExpectedR { net_expected_r: f64 },
    BelowMinQty { qty: f64 },
    BelowMinNotional { notional: f64 },
    ExposureCap { gross_x: f64 },
}

#[derive(Clone, Debug, Default, Serialize)]
pub struct RiskDecision {
    pub accepted: Vec<SizedOrder>,
    pub rejected: Vec<(String, Rejection)>,
    pub halted: bool,
}

pub struct RiskEngine {
    cfg: RiskConfig,
}

impl RiskEngine {
    pub fn new(cfg: RiskConfig) -> Self {
        Self { cfg }
    }

    pub fn config(&self) -> &RiskConfig {
        &self.cfg
    }

    /// Round-trip cost as a fraction of price: entry fee (maker for limit, taker for market),
    /// exit fee (taker: stops are market orders) and slippage on every market leg.
    pub fn cost_frac(&self, entry_type: EntryType) -> f64 {
        let c = &self.cfg.costs;
        let entry = match entry_type {
            EntryType::Market => c.taker_bps + c.slippage_bps,
            EntryType::Limit => c.maker_bps,
        };
        (entry + c.taker_bps + c.slippage_bps) / 1e4
    }

    pub fn drawdown_pct(&self, account: &AccountState) -> f64 {
        if account.day_start_equity <= 0.0 {
            return 0.0;
        }
        (account.equity / account.day_start_equity - 1.0) * 100.0
    }

    /// Evaluate proposals **in rank order** (best first). Earlier proposals consume slots first.
    pub fn evaluate(&self, proposals: &[TradeProposal], account: &AccountState, rules: &HashMap<String, InstrumentRules>, marks: &HashMap<String, f64>) -> RiskDecision {
        let mut d = RiskDecision::default();
        let dd = self.drawdown_pct(account);
        if dd <= -self.cfg.daily_loss_limit_pct {
            d.halted = true;
            d.rejected = proposals.iter().map(|p| (p.id.clone(), Rejection::DailyLossHalt { drawdown_pct: dd })).collect();
            return d;
        }

        let mut slots = self
            .cfg
            .max_new_trades_per_day
            .saturating_sub(account.trades_opened_today)
            .min(self.cfg.max_open_positions.saturating_sub(account.open_positions.len() as u32));
        let mut gross: f64 = account
            .open_positions
            .iter()
            .map(|p| p.notional(marks.get(&p.symbol).copied().unwrap_or(p.entry_price)))
            .sum();
        let mut cluster_count: HashMap<String, u32> = HashMap::new();
        for p in &account.open_positions {
            *cluster_count.entry(p.cluster.clone()).or_default() += 1;
        }
        let mut symbols: Vec<String> = account.open_positions.iter().map(|p| p.symbol.clone()).collect();
        let risk_amount = account.equity * self.cfg.risk_per_trade_pct / 100.0;

        for p in proposals {
            let reject = |d: &mut RiskDecision, r: Rejection| d.rejected.push((p.id.clone(), r));
            if slots == 0 {
                reject(&mut d, Rejection::NoSlots);
                continue;
            }
            if symbols.contains(&p.symbol) {
                reject(&mut d, Rejection::DuplicateSymbol);
                continue;
            }
            if cluster_count.get(&p.cluster).copied().unwrap_or(0) >= self.cfg.max_positions_per_cluster {
                reject(&mut d, Rejection::ClusterCap);
                continue;
            }
            let Some(rule) = rules.get(&p.symbol) else {
                reject(&mut d, Rejection::UnknownInstrument);
                continue;
            };
            match self.size(p, rule, risk_amount) {
                Err(r) => reject(&mut d, r),
                Ok(order) => {
                    let new_gross = gross + order.notional;
                    let gross_x = new_gross / account.equity;
                    if gross_x > self.cfg.max_gross_exposure_x {
                        reject(&mut d, Rejection::ExposureCap { gross_x });
                        continue;
                    }
                    gross = new_gross;
                    slots -= 1;
                    *cluster_count.entry(p.cluster.clone()).or_default() += 1;
                    symbols.push(p.symbol.clone());
                    d.accepted.push(order);
                }
            }
        }
        d
    }

    fn size(&self, p: &TradeProposal, rule: &InstrumentRules, risk_amount: f64) -> Result<SizedOrder, Rejection> {
        let geometry = |detail: &str| Rejection::InvalidGeometry { detail: detail.to_string() };
        // Written as negated positives on purpose: NaN fails every comparison, so NaN inputs are rejected.
        #[allow(clippy::nonminimal_bool)]
        let invalid = !(p.entry > 0.0 && p.stop > 0.0 && p.take_profit > 0.0) || !(0.0..=1.0).contains(&p.p_tp);
        if invalid {
            return Err(geometry("non-positive price or p_tp outside [0, 1]"));
        }
        let t = rule.tick_size;
        let (entry, stop, tp) = match p.side {
            Side::Long => (round_to_step(p.entry, t, Round::Nearest), round_to_step(p.stop, t, Round::Down), round_to_step(p.take_profit, t, Round::Down)),
            Side::Short => (round_to_step(p.entry, t, Round::Nearest), round_to_step(p.stop, t, Round::Up), round_to_step(p.take_profit, t, Round::Up)),
        };
        let s = p.side.sign();
        if (entry - stop) * s <= 0.0 {
            return Err(geometry("stop is not on the losing side of entry"));
        }
        if (tp - entry) * s <= 0.0 {
            return Err(geometry("take profit is not on the winning side of entry"));
        }
        let stop_dist = (entry - stop).abs();
        let stop_pct = stop_dist / entry * 100.0;
        if stop_pct > self.cfg.max_stop_distance_pct {
            return Err(Rejection::StopTooFar { pct: stop_pct });
        }

        let cost = entry * self.cost_frac(p.entry_type);
        let tp_dist = (tp - entry).abs();
        let net_expected_r = p.p_tp * (tp_dist - cost) / (stop_dist + cost) - (1.0 - p.p_tp);
        if net_expected_r < self.cfg.min_net_expected_r {
            return Err(Rejection::BelowMinExpectedR { net_expected_r });
        }

        let max_qty = match p.entry_type {
            EntryType::Market => rule.max_mkt_qty.min(rule.max_qty),
            EntryType::Limit => rule.max_qty,
        };
        let qty = round_to_step((risk_amount / (stop_dist + cost)).min(max_qty), rule.qty_step, Round::Down);
        if qty < rule.min_qty || qty <= 0.0 {
            return Err(Rejection::BelowMinQty { qty });
        }
        let notional = qty * entry;
        if notional < rule.min_notional {
            return Err(Rejection::BelowMinNotional { notional });
        }
        Ok(SizedOrder {
            proposal: p.clone(),
            entry,
            stop,
            take_profit: tp,
            qty,
            notional,
            risk_amount: qty * (stop_dist + cost),
            reward_risk: tp_dist / stop_dist,
            cost_r: cost / (stop_dist + cost),
            net_expected_r,
        })
    }

    /// Slot replacement (ARCHITECTURE §3.12): swap the weakest open position for a new candidate
    /// only if the expected-value gain clears the hurdle. Age never enters.
    pub fn should_replace(&self, open_remaining_r: f64, candidate_net_r: f64) -> bool {
        candidate_net_r - open_remaining_r > self.cfg.slot_replacement_hurdle_r
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use at_core::config::Costs;

    fn cfg() -> RiskConfig {
        RiskConfig {
            risk_per_trade_pct: 0.5,
            max_new_trades_per_day: 3,
            max_open_positions: 5,
            max_positions_per_cluster: 1,
            daily_loss_limit_pct: 1.5,
            max_gross_exposure_x: 3.0,
            max_stop_distance_pct: 15.0,
            min_net_expected_r: 0.10,
            costs: Costs { taker_bps: 5.5, maker_bps: 2.0, slippage_bps: 2.0 },
            slot_replacement_hurdle_r: 0.15,
        }
    }

    fn rules() -> HashMap<String, InstrumentRules> {
        let r = |s: &str, tick, step| InstrumentRules {
            symbol: s.into(),
            tick_size: tick,
            qty_step: step,
            min_qty: step,
            max_qty: 1e6,
            max_mkt_qty: 1e5,
            min_notional: 5.0,
            max_leverage: 100.0,
        };
        HashMap::from([("BTCUSDT".to_string(), r("BTCUSDT", 0.1, 0.001)), ("ETHUSDT".to_string(), r("ETHUSDT", 0.01, 0.01)), ("SOLUSDT".to_string(), r("SOLUSDT", 0.001, 0.1))])
    }

    #[allow(clippy::too_many_arguments)]
    fn prop(id: &str, sym: &str, side: Side, entry: f64, stop: f64, tp: f64, p: f64, cluster: &str) -> TradeProposal {
        TradeProposal { id: id.into(), symbol: sym.into(), side, entry_type: EntryType::Limit, entry, stop, take_profit: tp, p_tp: p, cluster: cluster.into(), channel: "C-MAIN".into() }
    }

    fn acct(equity: f64) -> AccountState {
        AccountState { equity, day_start_equity: equity, trades_opened_today: 0, open_positions: vec![] }
    }

    #[test]
    fn loss_at_stop_equals_fixed_risk_including_costs() {
        let e = RiskEngine::new(cfg());
        let d = e.evaluate(&[prop("a", "BTCUSDT", Side::Long, 75000.0, 73500.0, 79000.0, 0.45, "L1")], &acct(10_000.0), &rules(), &HashMap::new());
        let o = &d.accepted[0];
        // effective stop distance = 1500 + 75000 * (2 maker + 5.5 taker + 2 slip) bps = 1571.25
        let eff = 1500.0 + 75000.0 * 9.5e-4;
        assert!(o.risk_amount <= 50.0 + 1e-9, "never risks more than 0.5% of equity");
        assert!(o.risk_amount > 50.0 - 0.001 * eff, "qty rounding loses less than one step");
        assert_eq!(o.qty, 0.031);
    }

    #[test]
    fn short_geometry_and_rounding_direction() {
        let e = RiskEngine::new(cfg());
        let d = e.evaluate(&[prop("s", "ETHUSDT", Side::Short, 3000.004, 3100.001, 2800.009, 0.45, "L1")], &acct(10_000.0), &rules(), &HashMap::new());
        let o = &d.accepted[0];
        assert_eq!(o.stop, 3100.01, "short stop rounds up (away from entry)");
        assert_eq!(o.take_profit, 2800.01, "short target rounds up (toward entry)");
        let bad = e.evaluate(&[prop("x", "ETHUSDT", Side::Short, 3000.0, 2900.0, 2800.0, 0.5, "L1")], &acct(10_000.0), &rules(), &HashMap::new());
        assert!(matches!(bad.rejected[0].1, Rejection::InvalidGeometry { .. }));
    }

    #[test]
    fn caps_slots_clusters_and_halts() {
        let e = RiskEngine::new(cfg());
        let ps = [
            prop("1", "BTCUSDT", Side::Long, 75000.0, 73500.0, 79000.0, 0.45, "L1"),
            prop("2", "ETHUSDT", Side::Long, 3000.0, 2940.0, 3150.0, 0.45, "L1"),
            prop("3", "SOLUSDT", Side::Long, 150.0, 146.0, 160.0, 0.45, "ALT"),
        ];
        let d = e.evaluate(&ps, &acct(10_000.0), &rules(), &HashMap::new());
        assert_eq!(d.accepted.len(), 2);
        assert_eq!(d.rejected, vec![("2".to_string(), Rejection::ClusterCap)]);

        let mut a = acct(10_000.0);
        a.trades_opened_today = 3;
        assert!(matches!(e.evaluate(&ps, &a, &rules(), &HashMap::new()).rejected[0].1, Rejection::NoSlots));

        let mut a = acct(9_800.0);
        a.day_start_equity = 10_000.0;
        let d = e.evaluate(&ps, &a, &rules(), &HashMap::new());
        assert!(d.halted && d.accepted.is_empty());
    }

    #[test]
    fn expected_value_after_costs_gates_trades() {
        let e = RiskEngine::new(cfg());
        // Driftless coin flip at 1:1 has negative EV after costs.
        let d = e.evaluate(&[prop("c", "BTCUSDT", Side::Long, 75000.0, 74000.0, 76000.0, 0.5, "L1")], &acct(10_000.0), &rules(), &HashMap::new());
        assert!(matches!(d.rejected[0].1, Rejection::BelowMinExpectedR { .. }));
    }
}
