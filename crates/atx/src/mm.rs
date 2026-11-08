//! `atx mm`: weekend skewed passive quoting (H-MM1, forward-only). Config: `config/mm.yaml`.
//!
//!   atx mm --until 2026-11-08T12:00:00Z [--dry-run] [--yes]
//!
//! Every `refresh_s`, per symbol: read the book (public mainnet data), recent public trades, our position and
//! resting quotes. One post-only bid and one post-only ask, each the venue minimum size, around a reservation
//! price that leans with inventory (back toward flat) and with order flow (trade + book imbalance). Distances are
//! in units of recent 1h sigma; the ask is closer than the bid (bearish skew); our own fill toxicity (markouts)
//! widens the quotes and can pause adding; swing levels push a quote beyond the level (fill on the sweep).
//! Resting quotes are re-priced only when their target drifted > `requote_sigma` after `min_rest_s` (or > 1
//! sigma), so they keep their queue place instead of chasing the price. Inventory in profit is cleared one lot at
//! a time with a passive reduce-only limit at the touch. In the last `exit_window_min` minutes no new risk is
//! added and inventory is cleared passively; at `--until`, on the loss limit, after repeated errors or on Ctrl-C,
//! every quote is cancelled and any remainder closed with reduce-only market orders.
//!
//! Loss limit on strategy P&L from a fill ledger (cash of our executions + ledger inventory at mid; wallet equity
//! is not used because it moves with any non-USDT balances). Demo / testnet freely; mainnet behind the two-key
//! guard and `--yes`. Positions or foreign orders already open on a configured symbol refuse the start.
//! Journal: `<journal>[_mainnet]/YYYY-MM-DD/*.jsonl`; run summary: `data/mm/<env>-<run_id>/`.

use std::collections::{HashMap, HashSet};
use std::time::Duration;

use anyhow::{Context, bail};
use at_bybit::market::{MarketData, PriceType};
use at_bybit::trading::OrderRequest;
use at_core::config::Environment;
use at_core::num::{Round, format_step, round_to_step};
use at_core::time::{Freq, now_ms, parse_ts};
use at_core::types::InstrumentRules;
use at_exec::Broker;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use crate::cmd::Ctx;
use crate::exec::{Journal, Venue, journal_root};

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct MmCfg {
    pub symbols: Vec<String>,
    pub refresh_s: u64,
    pub sigma_lookback_min: i64,
    pub sigma_floor_bps: f64,
    pub ask_offset_sigma: f64,
    pub bid_offset_sigma: f64,
    pub requote_sigma: f64,
    pub min_rest_s: i64,
    pub inventory_shift_sigma: f64,
    pub take_profit_sigma: f64,
    pub max_short_usd: f64,
    pub max_long_usd: f64,
    pub max_gross_usd: f64,
    pub flow_window_s: i64,
    pub flow_depth_bps: f64,
    pub flow_shift_sigma: f64,
    pub tox_horizon_s: i64,
    pub tox_alpha: f64,
    pub tox_widen: f64,
    pub tox_pause_sigma: f64,
    pub level_lookback_h: i64,
    pub level_pivot_bars: usize,
    pub level_snap_sigma: f64,
    pub level_buffer_sigma: f64,
    pub loss_limit_usd: f64,
    pub exit_window_min: i64,
    pub leverage: f64,
}

// ---------------------------------------------------------------- levels

/// Swing highs / lows (fractal pivots on 15m bars) plus the lookback high and low.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Levels {
    pub res: Vec<f64>,
    pub sup: Vec<f64>,
}

pub fn levels(highs: &[f64], lows: &[f64], k: usize) -> Levels {
    let n = highs.len().min(lows.len());
    let mut lv = Levels::default();
    if n == 0 {
        return lv;
    }
    for i in k..n.saturating_sub(k) {
        let w = (i - k)..=(i + k);
        if w.clone().all(|j| j == i || highs[j] < highs[i]) {
            lv.res.push(highs[i]);
        }
        if w.clone().all(|j| j == i || lows[j] > lows[i]) {
            lv.sup.push(lows[i]);
        }
    }
    lv.res.push(highs[..n].iter().cloned().fold(f64::MIN, f64::max));
    lv.sup.push(lows[..n].iter().cloned().fold(f64::MAX, f64::min));
    lv
}

/// Liquidity-grab refinement, one-directional on purpose: a quote that sits just inside a level (within `snap`
/// of it) moves beyond the level by `buffer`, so we fill on the sweep rather than ahead of it. Quotes only ever
/// move away from the mid (H-VZ / H-GT / H-AU found no directional value in levels).
pub fn snap_beyond(ask: f64, bid: f64, mid: f64, sigma: f64, lv: &Levels, snap: f64, buffer: f64) -> (f64, f64) {
    let (sn, bf) = (snap * sigma * mid, buffer * sigma * mid);
    let ask = lv.res.iter().filter(|&&r| r >= ask && r <= ask + sn).cloned().fold(ask, |a, r| a.max(r + bf));
    let bid = lv.sup.iter().filter(|&&x| x <= bid && x >= bid - sn).cloned().fold(bid, |b, x| b.min(x - bf));
    (ask, bid)
}

// ---------------------------------------------------------------- order flow

/// Signed notional imbalance of public trades `(time_ms, +1 buy / -1 sell, notional)` since `since_ms`.
pub fn trade_imbalance(trades: &[(i64, f64, f64)], since_ms: i64) -> f64 {
    let (mut s, mut a) = (0.0, 0.0);
    for &(_, sign, notional) in trades.iter().filter(|x| x.0 >= since_ms) {
        s += sign * notional;
        a += notional;
    }
    if a > 0.0 { s / a } else { 0.0 }
}

/// Book imbalance within `depth_bps` of mid: (bid notional - ask notional) / total.
pub fn book_imbalance(bids: &[(f64, f64)], asks: &[(f64, f64)], mid: f64, depth_bps: f64) -> f64 {
    let d = depth_bps / 1e4;
    let b: f64 = bids.iter().filter(|x| x.0 >= mid * (1.0 - d)).map(|x| x.0 * x.1).sum();
    let a: f64 = asks.iter().filter(|x| x.0 <= mid * (1.0 + d)).map(|x| x.0 * x.1).sum();
    if a + b > 0.0 { (b - a) / (a + b) } else { 0.0 }
}

// ---------------------------------------------------------------- quotes

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Quote {
    pub price: f64,
    pub qty: f64,
    pub reduce_only: bool,
}

/// Everything the quote rule looks at for one symbol.
#[derive(Clone, Debug, Default)]
pub struct Inputs {
    pub bid1: f64,
    pub ask1: f64,
    /// 1h sigma as a fraction.
    pub sigma: f64,
    /// Signed position (short < 0) and its average entry.
    pub inv_qty: f64,
    pub avg_entry: f64,
    /// Absolute notional held in the other symbols.
    pub gross_others_usd: f64,
    /// Order flow in [-1, 1] (+ = buying pressure).
    pub flow: f64,
    /// Own-fill markout EWMA in sigma units (negative = picked off).
    pub tox: f64,
    /// Wind-down: only reduce inventory.
    pub exit_only: bool,
}

/// Smallest order the venue accepts at this price, on the qty grid (5% buffer over min notional).
pub fn min_lot(rule: &InstrumentRules, px: f64) -> f64 {
    let by_notional = if rule.min_notional > 0.0 { rule.min_notional * 1.05 / px } else { 0.0 };
    round_to_step(rule.min_qty.max(by_notional), rule.qty_step, Round::Up)
}

/// Target (bid, ask). Pure, so it is unit-tested.
pub fn target_quotes(cfg: &MmCfg, rule: &InstrumentRules, x: &Inputs, lv: &Levels) -> (Option<Quote>, Option<Quote>) {
    #[allow(clippy::nonminimal_bool)]
    if !(x.bid1 > 0.0 && x.ask1 > x.bid1 && x.sigma > 0.0) {
        return (None, None);
    }
    let (tick, sig) = (rule.tick_size, x.sigma);
    let mid = 0.5 * (x.bid1 + x.ask1);
    let inv_usd = x.inv_qty * mid;
    let inv_frac = if inv_usd < 0.0 { inv_usd / cfg.max_short_usd } else { inv_usd / cfg.max_long_usd }.clamp(-1.0, 1.0);
    let widen = 1.0 + cfg.tox_widen * (-x.tox).max(0.0);
    let pause_adding = x.tox < -cfg.tox_pause_sigma || x.exit_only;
    // Short inventory (inv_frac < 0) lifts the reservation price (bid closer = cover, ask further); buying flow lifts it too.
    let r = mid * (1.0 + (-cfg.inventory_shift_sigma * inv_frac + cfg.flow_shift_sigma * x.flow.clamp(-1.0, 1.0)) * sig);
    let (mut ask, mut bid) = snap_beyond(r * (1.0 + cfg.ask_offset_sigma * sig * widen), r * (1.0 - cfg.bid_offset_sigma * sig * widen), mid, sig, lv, cfg.level_snap_sigma, cfg.level_buffer_sigma);
    // Clear inventory passively at the touch when it is in profit, or during the wind-down.
    let tp = cfg.take_profit_sigma * sig;
    if x.inv_qty < 0.0 && (x.exit_only || (x.avg_entry > 0.0 && mid <= x.avg_entry * (1.0 - tp))) {
        bid = bid.max(x.bid1);
    }
    if x.inv_qty > 0.0 && (x.exit_only || (x.avg_entry > 0.0 && mid >= x.avg_entry * (1.0 + tp))) {
        ask = ask.min(x.ask1);
    }
    let ask_px = round_to_step(ask, tick, Round::Up).max(x.bid1 + tick);
    let bid_px = round_to_step(bid, tick, Round::Down).min(x.ask1 - tick);
    let lot = min_lot(rule, mid);
    let adds_ok = |after: f64| (if after < 0.0 { -after <= cfg.max_short_usd } else { after <= cfg.max_long_usd }) && x.gross_others_usd + after.abs() <= cfg.max_gross_usd;
    let reduce_qty = |q: f64| round_to_step(q.abs().min(lot), rule.qty_step, Round::Down);

    let ask = if x.inv_qty > 0.0 {
        Some(Quote { price: ask_px, qty: reduce_qty(x.inv_qty), reduce_only: true })
    } else if !pause_adding && adds_ok(inv_usd - lot * mid) {
        Some(Quote { price: ask_px, qty: lot, reduce_only: false })
    } else {
        None
    };
    let bid = if x.inv_qty < 0.0 {
        Some(Quote { price: bid_px, qty: reduce_qty(x.inv_qty), reduce_only: true })
    } else if !pause_adding && adds_ok(inv_usd + lot * mid) {
        Some(Quote { price: bid_px, qty: lot, reduce_only: false })
    } else {
        None
    };
    (bid.filter(|q| q.qty > 0.0), ask.filter(|q| q.qty > 0.0))
}

/// Robust 1h sigma (fraction) from 1m closes: mean |r| * sqrt(pi/2) * sqrt(60), floored.
pub fn sigma_1h(closes: &[f64], floor_bps: f64) -> f64 {
    let r: Vec<f64> = closes.windows(2).filter(|w| w[0] > 0.0 && w[1] > 0.0).map(|w| (w[1] / w[0]).ln().abs()).collect();
    let floor = floor_bps / 1e4;
    if r.len() < 30 {
        return floor;
    }
    let s = r.iter().sum::<f64>() / r.len() as f64 * (std::f64::consts::PI / 2.0).sqrt() * 60f64.sqrt();
    s.max(floor)
}

// ---------------------------------------------------------------- fill ledger

/// Strategy cash from our own executions (trades and funding) on the configured symbols, deduplicated by execId.
#[derive(Default)]
pub struct Ledger {
    seen: HashSet<String>,
    pub cash: f64,
    pub fees: f64,
    pub funding: f64,
    pub fills: Vec<Value>,
    /// Signed inventory implied by the same executions (the execution feed lags positions by seconds; marking the
    /// venue position against lagged cash would show phantom losses or gains).
    pub inv: HashMap<String, f64>,
    cursor_ms: i64,
}

impl Ledger {
    /// Returns true for a new trade execution.
    pub fn add(&mut self, x: &Value) -> bool {
        let id = x["execId"].as_str().unwrap_or_default().to_string();
        if id.is_empty() || !self.seen.insert(id) {
            return false;
        }
        self.cursor_ms = self.cursor_ms.max(num(&x["execTime"]) as i64);
        let fee = num(&x["execFee"]);
        if x["execType"].as_str().unwrap_or("Trade") == "Funding" {
            self.cash -= fee;
            self.funding -= fee;
            return false;
        }
        let sign = if x["side"] == "Sell" { 1.0 } else { -1.0 };
        self.cash += sign * num(&x["execValue"]) - fee;
        self.fees += fee;
        *self.inv.entry(x["symbol"].as_str().unwrap_or_default().to_string()).or_default() -= sign * num(&x["execQty"]);
        self.fills.push(x.clone());
        true
    }

    /// P&L = cash + ledger inventory marked at mid.
    pub fn pnl(&self, mids: &HashMap<String, f64>) -> f64 {
        self.cash + self.inv.iter().map(|(s, q)| q * mids.get(s).copied().unwrap_or(0.0)).sum::<f64>()
    }

    pub fn is_flat(&self) -> bool {
        self.inv.values().all(|q| q.abs() < 1e-9)
    }
}

fn num(v: &Value) -> f64 {
    v.as_str().and_then(|s| s.parse().ok()).or_else(|| v.as_f64()).unwrap_or(0.0)
}

fn pairs(v: &Value) -> Vec<(f64, f64)> {
    v.as_array().map(|a| a.iter().map(|x| (num(&x[0]), num(&x[1]))).collect()).unwrap_or_default()
}

// ---------------------------------------------------------------- run

pub async fn run(ctx: &Ctx, until: &str, dry_run: bool, yes: bool) -> anyhow::Result<()> {
    let cfg: MmCfg = serde_yaml::from_str(&std::fs::read_to_string(ctx.s.root.join("config/mm.yaml"))?).context("config/mm.yaml")?;
    let env = ctx.s.environment;
    let deadline = parse_ts(until)?;
    anyhow::ensure!(deadline > now_ms(), "--until {until} is in the past");
    if env == Environment::Mainnet && !dry_run {
        anyhow::ensure!(yes, "atx mm on mainnet places real orders; re-run with --yes");
    }
    if !dry_run {
        ctx.s.ensure_execution_allowed()?;
    }
    let symbols: Vec<String> = cfg.symbols.iter().map(|s| s.trim().to_uppercase()).collect();
    let venue = Venue::new(ctx)?;
    let t = venue.trading();
    let mrest = ctx.market_rest()?;
    let market = MarketData::new(&mrest, ctx.category());
    let cat = ctx.category().to_string();
    let rules = venue.rules(&symbols.iter().cloned().collect::<HashSet<_>>()).await?;
    for s in &symbols {
        anyhow::ensure!(rules.contains_key(s), "{s}: not a trading instrument");
    }

    // Refuse to start on top of anything we did not open.
    let positions = t.positions().await?;
    if let Some(p) = positions.iter().find(|p| symbols.contains(&p.symbol)) {
        bail!("{} already has a {} {} position: flatten it first (atx flatten) or remove it from config/mm.yaml", p.symbol, p.side, p.size);
    }
    let orders = t.open_orders(None).await?;
    for o in orders.iter().filter(|o| symbols.iter().any(|s| o["symbol"] == s.as_str())) {
        let link = o["orderLinkId"].as_str().unwrap_or_default();
        anyhow::ensure!(link.starts_with("mm"), "{}: foreign open order {link:?}; cancel it first", o["symbol"]);
        if !dry_run {
            t.cancel_order(o["symbol"].as_str().unwrap_or_default(), o["orderId"].as_str().unwrap_or_default()).await?;
        }
    }

    let start_ms = now_ms();
    let run_id = format!("mm{}", chrono::Utc::now().format("%m%d%H%M"));
    let out_dir = ctx.s.abs(&ctx.s.paths.data_root).join("mm").join(format!("{}-{run_id}", env.as_str()));
    std::fs::create_dir_all(&out_dir)?;
    let journal = Journal::new(journal_root(&ctx.s), &run_id);
    let equity0 = t.wallet().await?.total_equity;
    if !dry_run {
        for s in &symbols {
            let lev = rules[s].max_leverage.min(cfg.leverage);
            t.set_leverage(s, &format!("{lev}")).await.with_context(|| format!("{s}: set leverage"))?;
        }
    }
    journal.write("session_start", json!({"source": "atx mm", "hypothesis": "H-MM1", "environment": env.as_str(), "dry_run": dry_run, "until": until, "wallet_equity": equity0, "config": cfg}))?;
    println!("atx mm {run_id} on {} ({}) until {until}  wallet {equity0:.2}  symbols {}", env.as_str(), if dry_run { "DRY RUN" } else { "LIVE QUOTES" }, symbols.join(","));

    let mut sigmas: HashMap<String, f64> = HashMap::new();
    let mut lvls: HashMap<String, Levels> = HashMap::new();
    let mut tox: HashMap<String, f64> = symbols.iter().map(|s| (s.clone(), 0.0)).collect();
    let mut pending_markouts: Vec<(String, f64, f64, i64)> = vec![]; // (symbol, +1 buy / -1 sell, price, time)
    let mut sigma_at = 0i64;
    let mut ledger = Ledger { cursor_ms: start_ms, ..Default::default() };
    let mut errors = 0u32;
    let mut tick_n = 0u64;
    let mut seq = 0u64;
    let ctrl_c = tokio::signal::ctrl_c();
    tokio::pin!(ctrl_c);
    let reason: &str = loop {
        let now = now_ms();
        if now >= deadline {
            break "deadline";
        }
        let exit_only = now >= deadline - cfg.exit_window_min * 60_000;
        let step: anyhow::Result<Option<&'static str>> = async {
            if now_ms() - sigma_at > 15 * 60_000 {
                for s in &symbols {
                    let end = now_ms();
                    let k = market.klines(s, Freq::M1, PriceType::Last, end - cfg.sigma_lookback_min * 60_000, end).await?;
                    let closes: Vec<f64> = k.iter().map(|x| x.close).collect();
                    sigmas.insert(s.clone(), sigma_1h(&closes, cfg.sigma_floor_bps));
                    let k15 = market.klines(s, Freq::M15, PriceType::Last, end - cfg.level_lookback_h * 3_600_000, end).await?;
                    let (h, l): (Vec<f64>, Vec<f64>) = k15.iter().map(|x| (x.high, x.low)).unzip();
                    let lv = levels(&h, &l, cfg.level_pivot_bars);
                    journal.write("snapshot", json!({"source": "atx mm", "symbol": s, "sigma_1h": sigmas[s], "resistance": lv.res, "support": lv.sup}))?;
                    lvls.insert(s.clone(), lv);
                }
                sigma_at = now_ms();
            }
            // Book + public trades per symbol.
            let mut books: HashMap<String, (f64, f64, f64)> = HashMap::new(); // bid1, ask1, flow
            for s in &symbols {
                let ob = mrest.get_public("/v5/market/orderbook", &[("category", cat.clone()), ("symbol", s.clone()), ("limit", "50".into())]).await?;
                let (bids, asks) = (pairs(&ob["b"]), pairs(&ob["a"]));
                let (Some(b1), Some(a1)) = (bids.first(), asks.first()) else { continue };
                let mid = 0.5 * (b1.0 + a1.0);
                let tr = mrest.get_public("/v5/market/recent-trade", &[("category", cat.clone()), ("symbol", s.clone()), ("limit", "1000".into())]).await?;
                let trades: Vec<(i64, f64, f64)> = tr["list"]
                    .as_array()
                    .map(|a| a.iter().map(|x| (num(&x["time"]) as i64, if x["side"] == "Buy" { 1.0 } else { -1.0 }, num(&x["price"]) * num(&x["size"]))).collect())
                    .unwrap_or_default();
                let flow = 0.5 * trade_imbalance(&trades, now_ms() - cfg.flow_window_s * 1000) + 0.5 * book_imbalance(&bids, &asks, mid, cfg.flow_depth_bps);
                books.insert(s.clone(), (b1.0, a1.0, flow));
            }
            let mids: HashMap<String, f64> = books.iter().map(|(s, b)| (s.clone(), 0.5 * (b.0 + b.1))).collect();
            let pos = t.positions().await?;
            let ours: Vec<Value> = t.open_orders(None).await?.into_iter().filter(|o| o["orderLinkId"].as_str().unwrap_or_default().starts_with("mm")).collect();
            if !dry_run {
                for x in t.executions((ledger.cursor_ms - 60_000).max(start_ms), now_ms()).await? {
                    if symbols.iter().any(|s| x["symbol"] == s.as_str()) && ledger.add(&x) {
                        let s = x["symbol"].as_str().unwrap_or_default().to_string();
                        let sign = if x["side"] == "Buy" { 1.0 } else { -1.0 };
                        pending_markouts.push((s, sign, num(&x["execPrice"]), num(&x["execTime"]) as i64));
                    }
                }
            }
            // Toxicity: markout of each fill tox_horizon_s later, in sigma units.
            let due: Vec<_> = pending_markouts.iter().filter(|m| now_ms() - m.3 >= cfg.tox_horizon_s * 1000).cloned().collect();
            pending_markouts.retain(|m| now_ms() - m.3 < cfg.tox_horizon_s * 1000);
            for (s, sign, px, ts) in due {
                if let (Some(mid), Some(sg)) = (mids.get(&s), sigmas.get(&s)) {
                    let m = sign * (mid - px) / px / sg;
                    let e = tox.entry(s.clone()).or_default();
                    *e = (1.0 - cfg.tox_alpha) * *e + cfg.tox_alpha * m;
                    journal.write("fill", json!({"source": "atx mm", "symbol": s, "side": if sign > 0.0 { "Buy" } else { "Sell" }, "price": px, "exec_ms": ts, "markout_sigma": m, "horizon_s": cfg.tox_horizon_s, "tox_ewma": *e}))?;
                }
            }
            let pnl = ledger.pnl(&mids);
            if pnl <= -cfg.loss_limit_usd {
                return Ok(Some("loss_limit"));
            }
            let inv: HashMap<String, (f64, f64)> = symbols
                .iter()
                .map(|s| (s.clone(), pos.iter().find(|p| &p.symbol == s).map(|p| (if p.side == "Sell" { -p.size } else { p.size }, p.avg_price)).unwrap_or((0.0, 0.0))))
                .collect();
            let notional = |s: &String| inv[s].0.abs() * mids.get(s).copied().unwrap_or(0.0);
            let mut line = vec![];
            for s in &symbols {
                let Some(&(bid1, ask1, flow)) = books.get(s) else { continue };
                let rule = &rules[s];
                let x = Inputs {
                    bid1,
                    ask1,
                    sigma: sigmas[s],
                    inv_qty: inv[s].0,
                    avg_entry: inv[s].1,
                    gross_others_usd: symbols.iter().filter(|o| *o != s).map(notional).sum(),
                    flow,
                    tox: tox[s],
                    exit_only,
                };
                let (bid, ask) = target_quotes(&cfg, rule, &x, &lvls[s]);
                let mid = 0.5 * (bid1 + ask1);
                for (side, want) in [("Buy", bid), ("Sell", ask)] {
                    let resting: Vec<&Value> = ours.iter().filter(|o| o["symbol"] == s.as_str() && o["side"] == side).collect();
                    for o in resting.iter().skip(1) {
                        if !dry_run {
                            t.cancel_order(s, o["orderId"].as_str().unwrap_or_default()).await?;
                        }
                    }
                    let cur = resting.first();
                    // Keep a resting quote unless it is wrong in kind or drifted far enough for long enough.
                    let replace = match (want, cur) {
                        (Some(q), Some(o)) => {
                            let drift = (num(&o["price"]) - q.price).abs() / mid / x.sigma;
                            let age_s = (now_ms() - num(&o["updatedTime"]) as i64) / 1000;
                            let kind = (o["reduceOnly"] == true) != q.reduce_only || (num(&o["leavesQty"]) - q.qty).abs() > rule.qty_step * 0.5;
                            kind || drift > 1.0 || (drift > cfg.requote_sigma && age_s >= cfg.min_rest_s)
                        }
                        (None, Some(_)) => true,
                        _ => false,
                    };
                    if let (true, Some(o)) = (replace, cur) {
                        if !dry_run {
                            t.cancel_order(s, o["orderId"].as_str().unwrap_or_default()).await?;
                        }
                        journal.write("order", json!({"source": "atx mm", "environment": env.as_str(), "dry_run": dry_run, "symbol": s, "side": side, "action": "cancel", "order_id": o["orderId"], "price": o["price"], "inputs": format!("{x:?}")}))?;
                    }
                    if let (Some(q), true) = (want, cur.is_none() || replace) {
                        seq += 1;
                        let link = format!("{run_id}-{}-{}{seq}", s.trim_end_matches("USDT"), if side == "Buy" { "b" } else { "a" });
                        let req = OrderRequest {
                            symbol: s.clone(),
                            side,
                            order_type: "Limit",
                            qty: format_step(q.qty, rule.qty_step),
                            price: Some(format_step(q.price, rule.tick_size)),
                            time_in_force: "PostOnly".into(),
                            take_profit: None,
                            stop_loss: None,
                            tp_trigger_by: "LastPrice".into(),
                            sl_trigger_by: "MarkPrice".into(),
                            reduce_only: q.reduce_only,
                            order_link_id: link.chars().take(36).collect(),
                        };
                        let id = if dry_run { "dry".to_string() } else { t.create_order(&req).await?.order_id };
                        journal.write("order", json!({"source": "atx mm", "environment": env.as_str(), "dry_run": dry_run, "symbol": s, "side": side, "action": "place", "order_id": id, "link_id": req.order_link_id, "price": q.price, "qty": q.qty, "reduce_only": q.reduce_only, "bid1": bid1, "ask1": ask1, "sigma_1h": x.sigma, "flow": flow, "tox": x.tox, "inventory": x.inv_qty, "exit_only": exit_only}))?;
                    }
                }
                let fmt = |q: Option<Quote>| q.map(|q| format!("{}{}", q.price, if q.reduce_only { "R" } else { "" })).unwrap_or_else(|| "-".into());
                line.push(format!("{} inv ${:+.1} b {} a {} fl {:+.2} tx {:+.2} σ{:.0}", s.trim_end_matches("USDT"), x.inv_qty * mid, fmt(bid), fmt(ask), flow, x.tox, x.sigma * 1e4));
            }
            if tick_n % 6 == 0 {
                println!("{} pnl {pnl:+.3} fills {} fees {:.4}{} | {}", chrono::Utc::now().format("%H:%M:%S"), ledger.fills.len(), ledger.fees, if exit_only { " EXIT" } else { "" }, line.join(" | "));
            }
            Ok(None)
        }
        .await;
        match step {
            Ok(Some(r)) => break r,
            Ok(None) => errors = 0,
            Err(e) => {
                errors += 1;
                println!("tick error {errors}: {e:#}");
                let _ = journal.write("order", json!({"source": "atx mm", "error": format!("{e:#}")}));
                if errors >= 10 {
                    break "errors";
                }
            }
        }
        tick_n += 1;
        tokio::select! {
            _ = tokio::time::sleep(Duration::from_secs(cfg.refresh_s)) => {}
            _ = &mut ctrl_c => break "ctrl_c",
        }
    };

    // ---- wind down: cancel every quote, close every position, verify.
    println!("stopping ({reason}): cancelling quotes and closing positions");
    let mut flat = true;
    if !dry_run {
        let mut broker = Venue::broker(ctx, rules.clone())?;
        for attempt in 0..3 {
            for s in &symbols {
                if let Err(e) = t.cancel_all(s).await {
                    println!("{s}: cancel_all failed: {e:#}");
                }
                if let Err(e) = broker.close(s).await {
                    println!("{s}: close failed: {e:#}");
                }
            }
            tokio::time::sleep(Duration::from_secs(3)).await;
            let pos = t.positions().await?;
            let left_orders = t.open_orders(None).await?.into_iter().filter(|o| symbols.iter().any(|s| o["symbol"] == s.as_str())).count();
            flat = !pos.iter().any(|p| symbols.contains(&p.symbol)) && left_orders == 0;
            if flat {
                break;
            }
            println!("not flat after attempt {}", attempt + 1);
        }
        // Wait (up to ~60 s) for the lagging execution feed to include the closing trades.
        for _ in 0..12 {
            for x in t.executions(start_ms, now_ms()).await.unwrap_or_default() {
                if symbols.iter().any(|s| x["symbol"] == s.as_str()) {
                    ledger.add(&x);
                }
            }
            if !flat || ledger.is_flat() {
                break;
            }
            tokio::time::sleep(Duration::from_secs(5)).await;
        }
    }
    let equity1 = t.wallet().await?.total_equity;
    let fills = &ledger.fills;
    let maker = fills.iter().filter(|x| x["isMaker"] == true).count();
    let traded: f64 = fills.iter().map(|x| num(&x["execValue"])).sum();
    let summary = json!({
        "run_id": run_id, "environment": env.as_str(), "dry_run": dry_run, "reason": reason, "flat": flat,
        "start_ms": start_ms, "end_ms": now_ms(),
        "pnl": ledger.cash, "ledger_flat": ledger.is_flat(), "pnl_note": "fill ledger: cash of our trades and funding (valid as total P&L when ledger_flat)",
        "fees": ledger.fees, "funding": ledger.funding, "fills": fills.len(), "maker_fills": maker, "traded_notional": traded,
        "pnl_bps_of_traded": if traded > 0.0 { ledger.cash / traded * 1e4 } else { 0.0 },
        "tox_final": tox, "wallet_equity_start": equity0, "wallet_equity_end": equity1, "config": cfg,
    });
    std::fs::write(out_dir.join("summary.json"), serde_json::to_string_pretty(&summary)?)?;
    std::fs::write(out_dir.join("executions.json"), serde_json::to_string_pretty(fills)?)?;
    journal.write("session_end", summary.clone())?;
    println!("{}", serde_json::to_string_pretty(&summary)?);
    anyhow::ensure!(flat, "NOT FLAT after wind-down: check the account and run atx flatten");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg() -> MmCfg {
        serde_yaml::from_str(&std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/../../config/mm.yaml")).unwrap()).unwrap()
    }

    fn rule() -> InstrumentRules {
        InstrumentRules { symbol: "SOLUSDT".into(), tick_size: 0.01, qty_step: 0.1, min_qty: 0.1, max_qty: 1e6, max_mkt_qty: 1e5, min_notional: 5.0, max_leverage: 50.0 }
    }

    fn inp(inv: f64, avg: f64) -> Inputs {
        Inputs { bid1: 89.99, ask1: 90.00, sigma: 0.004, inv_qty: inv, avg_entry: avg, ..Default::default() }
    }

    #[test]
    fn flat_book_is_skewed_bearish_and_minimum_size() {
        let c = cfg();
        let (b, a) = target_quotes(&c, &rule(), &inp(0.0, 0.0), &Levels::default());
        let (b, a) = (b.unwrap(), a.unwrap());
        let mid = 89.995;
        assert!(a.price - mid < mid - b.price, "ask closer than bid");
        assert_eq!((a.qty, b.qty), (0.1, 0.1), "always the minimum lot");
        assert!(!a.reduce_only && !b.reduce_only);
        assert!(b.price < 89.99 && a.price > 90.0);
    }

    #[test]
    fn caps_stop_adding_but_allow_unwinding() {
        let c = cfg();
        // short 0.3 SOL ~ $27: one more lot ($9) breaches the $30 short cap; the bid (cover) stays, reduce-only.
        let (b, a) = target_quotes(&c, &rule(), &inp(-0.3, 91.0), &Levels::default());
        assert!(a.is_none());
        let b = b.unwrap();
        assert!(b.reduce_only && b.qty == 0.1);
        // short inventory pulls the bid closer than when flat
        let (b0, _) = target_quotes(&c, &rule(), &inp(0.0, 0.0), &Levels::default());
        assert!(b.price > b0.unwrap().price);
        // gross cap: other symbols already hold $55 -> no new risk in either direction from flat
        let x = Inputs { gross_others_usd: 55.0, ..inp(0.0, 0.0) };
        assert_eq!(target_quotes(&c, &rule(), &x, &Levels::default()), (None, None));
    }

    #[test]
    fn profit_and_exit_clear_passively_at_the_touch() {
        let c = cfg();
        // short from 91.0, mid 89.995 is > 0.6 sigma in profit -> cover bid joins the best bid
        let (b, a) = target_quotes(&c, &rule(), &inp(-0.1, 91.0), &Levels::default());
        assert_eq!(b.unwrap().price, 89.99);
        assert!(a.is_some(), "can still add short within caps");
        // short at a loss: cover bid stays away from the touch
        let (b, _) = target_quotes(&c, &rule(), &inp(-0.1, 89.5), &Levels::default());
        assert!(b.unwrap().price < 89.99);
        // wind-down: nothing new, cover at the touch
        let x = Inputs { exit_only: true, ..inp(-0.1, 89.5) };
        let (b, a) = target_quotes(&c, &rule(), &x, &Levels::default());
        assert!(a.is_none());
        assert_eq!(b.unwrap().price, 89.99);
        assert_eq!(target_quotes(&c, &rule(), &Inputs { exit_only: true, ..inp(0.0, 0.0) }, &Levels::default()), (None, None));
    }

    #[test]
    fn flow_and_toxicity() {
        let c = cfg();
        let (b0, a0) = target_quotes(&c, &rule(), &inp(0.0, 0.0), &Levels::default());
        let (b1, a1) = target_quotes(&c, &rule(), &Inputs { flow: 1.0, ..inp(0.0, 0.0) }, &Levels::default());
        assert!(a1.unwrap().price > a0.unwrap().price && b1.unwrap().price > b0.unwrap().price, "buying flow lifts both quotes");
        let (b2, a2) = target_quotes(&c, &rule(), &Inputs { tox: -0.5, ..inp(0.0, 0.0) }, &Levels::default());
        assert!(a2.unwrap().price > a0.unwrap().price && b2.unwrap().price < b0.unwrap().price, "toxic fills widen");
        assert_eq!(target_quotes(&c, &rule(), &Inputs { tox: -1.0, ..inp(0.0, 0.0) }, &Levels::default()), (None, None), "very toxic pauses adding");
        assert!((trade_imbalance(&[(10, 1.0, 3.0), (11, -1.0, 1.0), (1, -1.0, 100.0)], 5) - 0.5).abs() < 1e-12);
        assert!((book_imbalance(&[(99.9, 3.0), (90.0, 100.0)], &[(100.1, 1.0)], 100.0, 25.0) - (299.7 - 100.1) / (299.7 + 100.1)).abs() < 1e-9);
    }

    #[test]
    fn never_crosses_and_handles_bad_books() {
        let c = cfg();
        let x = Inputs { sigma: 1e-6, ..inp(0.0, 0.0) };
        let (b, a) = target_quotes(&c, &rule(), &x, &Levels::default());
        assert!(b.unwrap().price < 90.00 && a.unwrap().price > 89.99);
        assert_eq!(target_quotes(&c, &rule(), &Inputs { bid1: 0.0, ..inp(0.0, 0.0) }, &Levels::default()), (None, None));
        assert_eq!(target_quotes(&c, &rule(), &Inputs { bid1: 90.0, ask1: 90.0, ..inp(0.0, 0.0) }, &Levels::default()), (None, None));
    }

    #[test]
    fn levels_and_snap() {
        let h = [10.0, 11.0, 12.0, 11.0, 10.5, 11.5, 13.0, 11.0, 10.0];
        let l = [9.0, 8.0, 7.0, 8.0, 8.5, 7.5, 9.0, 8.0, 7.9];
        let lv = levels(&h, &l, 2);
        assert!(lv.res.contains(&12.0) && lv.res.contains(&13.0));
        assert!(lv.sup.contains(&7.0));
        let lv = Levels { res: vec![100.5], sup: vec![99.4] };
        let (a, b) = snap_beyond(100.4, 99.5, 100.0, 0.01, &lv, 0.5, 0.05);
        assert!((a - 100.55).abs() < 1e-9 && (b - 99.35).abs() < 1e-9);
        let lv = Levels { res: vec![101.5, 100.2], sup: vec![98.0, 99.8] };
        assert_eq!(snap_beyond(100.4, 99.5, 100.0, 0.01, &lv, 0.5, 0.05), (100.4, 99.5));
    }

    #[test]
    fn ledger_round_trip_and_dedupe() {
        let mut l = Ledger::default();
        let sell = json!({"execId": "1", "symbol": "SOLUSDT", "side": "Sell", "execValue": "18.0", "execQty": "0.2", "execFee": "0.0036", "execType": "Trade", "execTime": "1"});
        let buy = json!({"execId": "2", "symbol": "SOLUSDT", "side": "Buy", "execValue": "17.9", "execQty": "0.2", "execFee": "0.00358", "execType": "Trade", "execTime": "2"});
        assert!(l.add(&sell));
        assert!(!l.add(&sell));
        let mids = HashMap::from([("SOLUSDT".to_string(), 89.0)]);
        assert!((l.inv["SOLUSDT"] + 0.2).abs() < 1e-12);
        assert!((l.pnl(&mids) - (18.0 - 0.0036 - 17.8)).abs() < 1e-9);
        l.add(&buy);
        l.add(&json!({"execId": "3", "symbol": "SOLUSDT", "side": "Sell", "execValue": "0", "execFee": "-0.001", "execType": "Funding", "execTime": "3"}));
        assert_eq!(l.fills.len(), 2);
        assert!(l.is_flat());
        assert!((l.cash - (18.0 - 17.9 - 0.0036 - 0.00358 + 0.001)).abs() < 1e-9);
        assert!((l.funding - 0.001).abs() < 1e-12);
    }

    #[test]
    fn sigma_floor_and_scale() {
        assert_eq!(sigma_1h(&[1.0; 10], 15.0), 0.0015);
        let closes: Vec<f64> = (0..241).map(|i| if i % 2 == 0 { 100.0 } else { 100.1 }).collect();
        let s = sigma_1h(&closes, 1.0);
        assert!((s - 0.001 * (std::f64::consts::PI / 2.0).sqrt() * 60f64.sqrt()).abs() < 1e-4);
    }
}
