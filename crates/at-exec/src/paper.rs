//! Bar-driven paper broker. Conservative by construction:
//! - if a bar touches both stop and target, the **stop** is assumed to fill first;
//! - gaps through a stop fill at the bar open (worse than the stop);
//! - a limit entry filled in a bar is immediately exposed to that same bar's stop;
//! - exits pay taker fee + slippage, exactly as the risk engine's cost model assumes.

use at_core::config::Costs;
use at_core::types::{EntryType, Position, Side};
use at_core::time::Ms;
use at_risk::SizedOrder;
use serde::Serialize;

use crate::Broker;

#[derive(Clone, Debug, Serialize)]
pub struct ClosedTrade {
    pub symbol: String,
    pub side: Side,
    pub qty: f64,
    pub entry_price: f64,
    pub exit_price: f64,
    pub exit_reason: &'static str,
    pub pnl: f64,
    pub r_multiple: f64,
    pub opened_ts: Ms,
    pub closed_ts: Ms,
}

#[derive(Clone, Debug)]
struct Open {
    pos: Position,
    risk_amount: f64,
}

#[derive(Clone, Debug)]
struct Pending {
    order: SizedOrder,
}

pub struct PaperBroker {
    label: String,
    cash_equity: f64,
    costs: Costs,
    open: Vec<Open>,
    pending: Vec<Pending>,
    closed: Vec<ClosedTrade>,
    last_price: std::collections::HashMap<String, (f64, Ms)>,
    seq: u64,
}

impl PaperBroker {
    pub fn new(label: &str, equity: f64, costs: Costs) -> Self {
        Self { label: label.into(), cash_equity: equity, costs, open: vec![], pending: vec![], closed: vec![], last_price: Default::default(), seq: 0 }
    }

    pub fn closed_trades(&self) -> &[ClosedTrade] {
        &self.closed
    }

    fn fee(&self, notional: f64, taker: bool) -> f64 {
        notional * if taker { self.costs.taker_bps } else { self.costs.maker_bps } / 1e4
    }

    fn slip(&self, price: f64, side: Side, opening: bool) -> f64 {
        // Slippage always moves the fill against us.
        let dir = if opening { side.sign() } else { -side.sign() };
        price * (1.0 + dir * self.costs.slippage_bps / 1e4)
    }

    fn open_position(&mut self, o: &SizedOrder, fill: f64, taker: bool, ts: Ms) {
        self.cash_equity -= self.fee(o.qty * fill, taker);
        self.open.push(Open {
            pos: Position {
                symbol: o.proposal.symbol.clone(),
                side: o.proposal.side,
                qty: o.qty,
                entry_price: fill,
                stop: o.stop,
                take_profit: o.take_profit,
                cluster: o.proposal.cluster.clone(),
                opened_ts: ts,
            },
            risk_amount: o.risk_amount,
        });
    }

    fn close_at(&mut self, i: usize, raw_price: f64, reason: &'static str, ts: Ms) {
        let o = self.open.remove(i);
        let p = &o.pos;
        let exit = self.slip(raw_price, p.side, false);
        let gross = p.qty * (exit - p.entry_price) * p.side.sign();
        let pnl = gross - self.fee(p.qty * exit, true);
        self.cash_equity += pnl;
        self.closed.push(ClosedTrade {
            symbol: p.symbol.clone(),
            side: p.side,
            qty: p.qty,
            entry_price: p.entry_price,
            exit_price: exit,
            exit_reason: reason,
            pnl,
            r_multiple: pnl / o.risk_amount,
            opened_ts: p.opened_ts,
            closed_ts: ts,
        });
    }

    /// Advance one completed bar for `symbol`.
    pub fn on_bar(&mut self, symbol: &str, ts: Ms, open: f64, high: f64, low: f64, close: f64) {
        self.last_price.insert(symbol.to_string(), (close, ts));
        // 1) pending limit entries
        let mut k = 0;
        while k < self.pending.len() {
            let pd = &self.pending[k];
            if pd.order.proposal.symbol != symbol {
                k += 1;
                continue;
            }
            let px = pd.order.entry;
            let fill = match pd.order.proposal.side {
                Side::Long if low <= px => Some(open.min(px)),
                Side::Short if high >= px => Some(open.max(px)),
                _ => None,
            };
            match fill {
                Some(f) => {
                    let o = self.pending.remove(k).order;
                    self.open_position(&o, f, false, ts);
                }
                None => k += 1,
            }
        }
        // 2) exits
        let mut i = 0;
        while i < self.open.len() {
            let p = &self.open[i].pos;
            if p.symbol != symbol {
                i += 1;
                continue;
            }
            let s = p.side.sign();
            let stop_hit = match p.side {
                Side::Long => low <= p.stop,
                Side::Short => high >= p.stop,
            };
            let tp_hit = match p.side {
                Side::Long => high >= p.take_profit,
                Side::Short => low <= p.take_profit,
            };
            if stop_hit {
                // gap through the stop fills at the open
                let px = if (open - p.stop) * s < 0.0 { open } else { p.stop };
                self.close_at(i, px, "stop", ts);
            } else if tp_hit {
                let px = if (open - p.take_profit) * s > 0.0 { open } else { p.take_profit };
                self.close_at(i, px, "take_profit", ts);
            } else {
                i += 1;
            }
        }
    }

    /// Mark-to-market equity using the last seen close of each symbol.
    pub fn marked_equity(&self) -> f64 {
        self.cash_equity
            + self
                .open
                .iter()
                .map(|o| {
                    let px = self.last_price.get(&o.pos.symbol).map(|x| x.0).unwrap_or(o.pos.entry_price);
                    o.pos.qty * (px - o.pos.entry_price) * o.pos.side.sign()
                })
                .sum::<f64>()
    }
}

impl Broker for PaperBroker {
    fn name(&self) -> &str {
        &self.label
    }

    async fn equity(&self) -> anyhow::Result<f64> {
        Ok(self.marked_equity())
    }

    async fn positions(&self) -> anyhow::Result<Vec<Position>> {
        Ok(self.open.iter().map(|o| o.pos.clone()).collect())
    }

    async fn submit(&mut self, o: &SizedOrder, link_id: &str) -> anyhow::Result<String> {
        self.seq += 1;
        let id = format!("{}-{}-{link_id}", self.label, self.seq);
        match o.proposal.entry_type {
            EntryType::Market => {
                let (px, ts) = self.last_price.get(&o.proposal.symbol).copied().unwrap_or((o.entry, 0));
                let fill = self.slip(px, o.proposal.side, true);
                self.open_position(o, fill, true, ts);
            }
            EntryType::Limit => self.pending.push(Pending { order: o.clone() }),
        }
        Ok(id)
    }

    async fn move_stop(&mut self, symbol: &str, stop: f64) -> anyhow::Result<()> {
        for o in self.open.iter_mut().filter(|o| o.pos.symbol == symbol) {
            o.pos.stop = stop;
        }
        Ok(())
    }

    async fn close(&mut self, symbol: &str) -> anyhow::Result<()> {
        self.pending.retain(|p| p.order.proposal.symbol != symbol);
        if let Some(i) = self.open.iter().position(|o| o.pos.symbol == symbol) {
            let (px, ts) = self.last_price.get(symbol).copied().unwrap_or((self.open[i].pos.entry_price, 0));
            self.close_at(i, px, "manual", ts);
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use at_core::types::TradeProposal;

    fn order(entry_type: EntryType) -> SizedOrder {
        SizedOrder {
            proposal: TradeProposal { id: "t".into(), symbol: "BTCUSDT".into(), side: Side::Long, entry_type, entry: 100.0, stop: 95.0, take_profit: 110.0, p_tp: 0.5, cluster: "L1".into(), channel: "C-MAIN".into() },
            entry: 100.0,
            stop: 95.0,
            take_profit: 110.0,
            qty: 10.0,
            notional: 1000.0,
            risk_amount: 50.0,
            reward_risk: 2.0,
            cost_r: 0.0,
            net_expected_r: 0.4,
        }
    }

    fn costs() -> Costs {
        Costs { taker_bps: 0.0, maker_bps: 0.0, slippage_bps: 0.0 }
    }

    #[tokio::test]
    async fn ambiguous_bar_assumes_stop_first() {
        let mut b = PaperBroker::new("paper", 10_000.0, costs());
        b.on_bar("BTCUSDT", 0, 100.0, 100.0, 100.0, 100.0);
        b.submit(&order(EntryType::Market), "x").await.unwrap();
        b.on_bar("BTCUSDT", 1, 100.0, 111.0, 94.0, 105.0);
        let t = &b.closed_trades()[0];
        assert_eq!(t.exit_reason, "stop");
        assert!((t.r_multiple + 1.0).abs() < 1e-9);
    }

    #[tokio::test]
    async fn gap_through_stop_fills_at_open() {
        let mut b = PaperBroker::new("paper", 10_000.0, costs());
        b.on_bar("BTCUSDT", 0, 100.0, 100.0, 100.0, 100.0);
        b.submit(&order(EntryType::Market), "x").await.unwrap();
        b.on_bar("BTCUSDT", 1, 90.0, 92.0, 89.0, 91.0);
        let t = &b.closed_trades()[0];
        assert_eq!(t.exit_price, 90.0);
        assert!(t.r_multiple < -1.0, "gaps can lose more than 1R");
    }

    #[tokio::test]
    async fn limit_fills_then_hits_target() {
        let mut b = PaperBroker::new("paper", 10_000.0, costs());
        b.submit(&order(EntryType::Limit), "x").await.unwrap();
        b.on_bar("BTCUSDT", 0, 102.0, 103.0, 99.5, 101.0);
        assert_eq!(b.positions().await.unwrap().len(), 1);
        b.on_bar("BTCUSDT", 1, 101.0, 112.0, 100.5, 111.0);
        assert_eq!(b.closed_trades()[0].exit_reason, "take_profit");
        assert!((b.equity().await.unwrap() - 10_100.0).abs() < 1e-9);
    }
}
