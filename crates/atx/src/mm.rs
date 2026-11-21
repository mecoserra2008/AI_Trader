//! `atx mm`: skewed passive quoting across many contracts (H-MM1 weekend, H-MM2 multi-asset). Config: `config/mm*.yaml`.
//!
//!   atx mm --until 2026-11-09T12:00:00Z [--config config/mm_multi.yaml] [--skip-busy] [--dry-run] [--yes]
//!
//! Every `refresh_s`: one tickers call gives every book top; public order flow (trade + book imbalance) is refreshed
//! for a rotating batch of `flow_batch` symbols; our positions, resting quotes and executions come from three private
//! calls. Per symbol, one post-only bid and one post-only ask around a reservation price that leans with inventory
//! (back toward flat) and with order flow. Distances are in units of 1h sigma, the larger of a slow (4 h) and a fast
//! (30 min) estimate from 1m klines, so quotes widen as soon as volatility bursts. The skew is in DISTANCE (the ask
//! sits closer than the bid); quotes never cross. Size: the side that adds inventory is always the venue minimum lot;
//! the side that reduces inventory scales from 1 to `reduce_lots_max` lots with |inventory| / cap. Own-fill markouts
//! (toxicity) widen quotes and can pause adding; a mid move larger than `shock_pause_sigma` per tick pauses adding on
//! that symbol for `shock_cooldown_s`; swing levels push a quote beyond the level. Resting quotes move only when their
//! target drifted > `requote_sigma` after `min_rest_s` (or > 1 sigma). Profitable inventory is cleared passively at
//! the touch. Order operations are throttled (`order_pause_ms`).
//!
//! Risk: per-symbol notional caps after a fill (one lot is always allowed from flat), a gross cap, a per-symbol loss
//! stop (cancel + flatten that symbol only) and a global loss limit, all on a fill ledger (cash of our executions +
//! ledger inventory at mid). The last `exit_window_min` minutes add no risk; at `--until`, on the global loss limit,
//! after repeated errors or on Ctrl-C every quote is cancelled and every position closed (reduce-only market).
//! A leaderboard (P&L, fills, markouts, mean-reversion statistics per symbol) is written every heartbeat.
//! Positions or foreign orders on a configured symbol refuse the start, or drop that symbol with `--skip-busy`.
//! Journal: `<journal>[_mainnet]/YYYY-MM-DD/*.jsonl`; run files: `data/mm/<env>-<run_id>/`.

use std::collections::{HashMap, HashSet};
use std::time::Duration;

use anyhow::Context;
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
    #[serde(default)]
    pub sigma_fast_min: i64,
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
    #[serde(default)]
    pub reduce_lots_max: u32,
    pub flow_window_s: i64,
    pub flow_depth_bps: f64,
    pub flow_shift_sigma: f64,
    #[serde(default)]
    pub flow_batch: usize,
    pub tox_horizon_s: i64,
    pub tox_alpha: f64,
    pub tox_widen: f64,
    pub tox_pause_sigma: f64,
    #[serde(default)]
    pub shock_pause_sigma: f64,
    #[serde(default)]
    pub shock_cooldown_s: i64,
    pub level_lookback_h: i64,
    pub level_pivot_bars: usize,
    pub level_snap_sigma: f64,
    pub level_buffer_sigma: f64,
    pub loss_limit_usd: f64,
    #[serde(default)]
    pub symbol_loss_limit_usd: f64,
    pub exit_window_min: i64,
    pub leverage: f64,
    #[serde(default)]
    pub order_pause_ms: u64,
    /// Portfolio net exposure cap (signed sum of all symbols' inventory, USD; 0 = off): adds that push |net| above it pause.
    #[serde(default)]
    pub max_net_usd: f64,
    /// Every reservation price leans against the portfolio's net inventory by this many sigma at the net cap.
    #[serde(default)]
    pub portfolio_shift_sigma: f64,
    /// Discretionary per-symbol views (logged, not tested signals): reservation shift in sigma units and optional caps.
    #[serde(default)]
    pub bias: HashMap<String, Bias>,
    /// Optional JSON of structural zones per symbol (scripts/levels.py): {"CLUSDT": [{"lo": .., "hi": ..}, ..]}.
    #[serde(default)]
    pub levels_file: Option<String>,
    /// Slow-link settings. Up to this many symbols, book tops come from one small ticker call per symbol instead of
    /// the whole-category call (~700 KB); 0 = always the whole category.
    #[serde(default)]
    pub per_symbol_tickers_max: usize,
    /// Public trades fetched per order-flow refresh (0 = 500, the venue maximum).
    #[serde(default)]
    pub flow_trades_limit: u32,
    /// A symbol whose book top is older than this when its quotes are decided adds no inventory that tick (its
    /// resting adding quotes are pulled; reducing quotes stay). 0 = off.
    #[serde(default)]
    pub max_quote_age_s: f64,
    /// Micro-volatility quoting (0 = off): quote distances, skews, take-profit and requotes use the expected move over
    /// `micro_horizon_s`, from `micro_interval_s` returns of recent public trades (last `micro_lookback` intervals),
    /// floored at `micro_floor_bps`. The 1h sigma still drives shock pauses, toxicity and levels. Meant for 0% maker fees.
    #[serde(default)]
    pub micro_interval_s: u64,
    #[serde(default)]
    pub micro_lookback: usize,
    #[serde(default)]
    pub micro_horizon_s: f64,
    #[serde(default)]
    pub micro_floor_bps: f64,
    /// Pivot-level tilt: reservation shift (quote-sigma units) toward the nearer level side (near support: up). Levels
    /// further than `level_reach_sigma` (1h sigma) count as that far. Evidence (P4 / R-PX-4): levels break about as often
    /// as they hold, so keep this small and A/B it.
    #[serde(default)]
    pub level_shift_sigma: f64,
    #[serde(default)]
    pub level_reach_sigma: f64,
    /// Quantity as an exposure factor: the adding side quotes 1..add_lots_max minimum lots, scaled by the shortfall to the
    /// inventory target (`add_inventory_weight`) and by the level tilt (`add_level_weight`); caps still apply. 0/1 = one lot.
    #[serde(default)]
    pub add_lots_max: u32,
    #[serde(default)]
    pub add_inventory_weight: f64,
    #[serde(default)]
    pub add_level_weight: f64,
    /// Break-even floor (bps over the average entry; 0 = off): reducing quotes never sell a long below entry x (1 + edge)
    /// or buy back a short above entry x (1 - edge), except in the wind-down. Stops the inventory skew from realising
    /// losses on lots bought earlier (2026-11-12 NVDA run).
    #[serde(default)]
    pub reduce_min_edge_bps: f64,
    /// Trend guard (0 = off): when the return over the recent-trade window is beyond `trend_guard_sigma` x the 1h sigma
    /// scaled to that window and points against the discretionary view (bias / inventory target), the view and the
    /// target are switched off for that tick, so the quoter stops leaning into a falling (rising) market.
    #[serde(default)]
    pub trend_guard_sigma: f64,
}

#[derive(Clone, Debug, Default, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Bias {
    /// Reservation-price shift in units of 1h sigma (+ = bullish: both quotes up, bid closer).
    pub sigma: f64,
    /// The bias switches off once the mid trades below / above these levels (view target reached).
    #[serde(default)]
    pub off_below: Option<f64>,
    #[serde(default)]
    pub off_above: Option<f64>,
    #[serde(default)]
    pub max_short_usd: Option<f64>,
    #[serde(default)]
    pub max_long_usd: Option<f64>,
    /// Inventory the quoter steers toward (USD, signed; + = hold a small long most of the time). The inventory skew
    /// works on the distance from this target: below it the reservation price rises (bid closer, ask further) in
    /// proportion to the shortfall; take-profit only sells the excess above it. 0 / absent = steer to flat.
    #[serde(default)]
    pub target_usd: Option<f64>,
    #[serde(default)]
    pub note: String,
}

impl Bias {
    pub fn active_sigma(&self, mid: f64) -> f64 {
        let off = self.off_below.is_some_and(|l| mid <= l) || self.off_above.is_some_and(|h| mid >= h);
        if off { 0.0 } else { self.sigma }
    }
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

// ---------------------------------------------------------------- order flow and statistics

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

/// 1h sigma from the last `minutes` 1m returns (needs minutes + 1 closes; 0 when there are not enough or minutes <= 0).
/// Fixed 2026-11-11: the fast window used to pass `minutes` closes (one return short of sigma_1h's 30-return minimum),
/// so the 30-minute sigma always fell back to the floor and quotes never widened on 30-minute bursts.
pub fn sigma_last(closes: &[f64], minutes: i64, floor_bps: f64) -> f64 {
    if minutes <= 0 || closes.len() as i64 <= minutes {
        return 0.0;
    }
    sigma_1h(&closes[closes.len() - minutes as usize - 1..], floor_bps)
}

/// Per-interval sigma (fraction) from recent public trades `(time_ms, price)` (any order): last price per
/// `interval_s` bucket counted back from `now_ms`, robust sigma of the bucket-to-bucket log returns over the last
/// `lookback` buckets (a gap of k empty buckets scales the return by 1/sqrt(k+1)). None with fewer than 8 returns.
pub fn micro_sigma(trades: &[(i64, f64)], now_ms: i64, interval_s: u64, lookback: usize) -> Option<f64> {
    if interval_s == 0 || lookback < 2 {
        return None;
    }
    let iv = interval_s as i64 * 1000;
    let mut last: Vec<Option<(i64, f64)>> = vec![None; lookback + 1];
    for &(t, px) in trades {
        if px <= 0.0 || t > now_ms {
            continue;
        }
        let k = ((now_ms - t) / iv) as usize;
        if k <= lookback && last[k].is_none_or(|(t0, _)| t > t0) {
            last[k] = Some((t, px));
        }
    }
    let pts: Vec<(usize, f64)> = (0..=lookback).rev().filter_map(|k| last[k].map(|(_, p)| (k, p))).collect();
    let r: Vec<f64> = pts.windows(2).map(|w| (w[1].1 / w[0].1).ln().abs() / ((w[0].0 - w[1].0) as f64).sqrt()).collect();
    if r.len() < 8 {
        return None;
    }
    Some(r.iter().sum::<f64>() / r.len() as f64 * (std::f64::consts::PI / 2.0).sqrt())
}

/// Where the mid sits between the nearest support below and resistance above, in [-1, 1]: +1 at support, -1 at
/// resistance, 0 halfway or with no level on either side. Distances are in 1h sigma and capped at `reach` (default 3).
pub fn level_tilt(mid: f64, sigma: f64, lv: &Levels, reach: f64) -> f64 {
    if !(mid > 0.0 && sigma > 0.0) {
        return 0.0;
    }
    let reach = if reach > 0.0 { reach } else { 3.0 };
    let sup = lv.sup.iter().filter(|&&x| x < mid).cloned().fold(f64::NAN, f64::max);
    let res = lv.res.iter().filter(|&&x| x > mid).cloned().fold(f64::NAN, f64::min);
    let ds = if sup.is_finite() { ((mid - sup) / mid / sigma).min(reach) } else { reach };
    let dr = if res.is_finite() { ((res - mid) / mid / sigma).min(reach) } else { reach };
    if ds + dr <= 0.0 { 0.0 } else { (dr - ds) / (dr + ds) }
}

/// Mean-reversion statistics from 1m closes: lag-1 autocorrelation and the variance ratio VR(15m / 1m).
pub fn mr_stats(closes: &[f64]) -> (f64, f64) {
    let r: Vec<f64> = closes.windows(2).filter(|w| w[0] > 0.0 && w[1] > 0.0).map(|w| (w[1] / w[0]).ln()).collect();
    if r.len() < 60 {
        return (f64::NAN, f64::NAN);
    }
    let n = r.len() as f64;
    let m = r.iter().sum::<f64>() / n;
    let v = r.iter().map(|x| (x - m).powi(2)).sum::<f64>() / n;
    let c1 = r.windows(2).map(|w| (w[0] - m) * (w[1] - m)).sum::<f64>() / n;
    let r15: Vec<f64> = r.chunks(15).filter(|c| c.len() == 15).map(|c| c.iter().sum::<f64>()).collect();
    let m15 = r15.iter().sum::<f64>() / r15.len().max(1) as f64;
    let v15 = r15.iter().map(|x| (x - m15).powi(2)).sum::<f64>() / r15.len().max(1) as f64;
    let ac1 = if v > 0.0 { c1 / v } else { f64::NAN };
    let vr = if v > 0.0 { v15 / (15.0 * v) } else { f64::NAN };
    (ac1, vr)
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
    /// Shock cooldown: don't add inventory.
    pub paused: bool,
    /// Discretionary reservation shift (sigma units) and per-symbol caps (0 = use the config defaults).
    pub bias_sigma: f64,
    pub cap_short_usd: f64,
    pub cap_long_usd: f64,
    /// Signed inventory (USD) held in all OTHER symbols (portfolio net exposure).
    pub net_others_usd: f64,
    /// Resting inventory-adding orders in all OTHER symbols (USD notional): counted as filled for the net cap.
    pub pend_sell_others_usd: f64,
    pub pend_buy_others_usd: f64,
    /// Inventory target in USD (Bias::target_usd); 0 = flat.
    pub target_usd: f64,
    /// Pivot-level tilt in [-1, 1] (+ = nearer support than resistance), from `level_tilt`.
    pub level_tilt: f64,
}

/// Trend guard: true when the recent move `ret` (log return over `window_s`) runs against the view by more than `k`
/// standard deviations of a `sigma_1h` random walk over that window. `view` > 0 = long lean (bias or target), < 0 = short.
pub fn trend_against_view(ret: f64, window_s: f64, sigma_1h: f64, k: f64, view: f64) -> bool {
    if k <= 0.0 || view == 0.0 || window_s <= 0.0 || sigma_1h <= 0.0 {
        return false;
    }
    let sd = sigma_1h * (window_s / 3600.0).sqrt();
    if view > 0.0 { ret < -k * sd } else { ret > k * sd }
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
    let cap_s = if x.cap_short_usd > 0.0 { x.cap_short_usd } else { cfg.max_short_usd };
    let cap_l = if x.cap_long_usd > 0.0 { x.cap_long_usd } else { cfg.max_long_usd };
    let inv_frac = if inv_usd < 0.0 { inv_usd / cap_s } else { inv_usd / cap_l }.clamp(-1.0, 1.0);
    // Skew on the distance from the inventory target: -1 at the short cap, 0 at the target, +1 at the long cap.
    let target = x.target_usd.clamp(-cap_s, cap_l);
    let dev = inv_usd - target;
    let dev_frac = if dev < 0.0 { dev / (target + cap_s).max(1e-9) } else { dev / (cap_l - target).max(1e-9) }.clamp(-1.0, 1.0);
    let widen = 1.0 + cfg.tox_widen * (-x.tox).max(0.0);
    let pause_adding = x.tox < -cfg.tox_pause_sigma || x.exit_only || x.paused;
    // Short inventory (inv_frac < 0) lifts the reservation price (bid closer = cover, ask further); buying flow lifts it too.
    let net_frac = if cfg.max_net_usd > 0.0 { ((x.net_others_usd + inv_usd) / cfg.max_net_usd).clamp(-1.0, 1.0) } else { 0.0 };
    let r = mid * (1.0 + (cfg.level_shift_sigma * x.level_tilt.clamp(-1.0, 1.0) - cfg.inventory_shift_sigma * dev_frac - cfg.portfolio_shift_sigma * net_frac + cfg.flow_shift_sigma * x.flow.clamp(-1.0, 1.0) + x.bias_sigma) * sig);
    let (mut ask, mut bid) =
        snap_beyond(r * (1.0 + cfg.ask_offset_sigma * sig * widen), r * (1.0 - cfg.bid_offset_sigma * sig * widen), mid, sig, lv, cfg.level_snap_sigma, cfg.level_buffer_sigma);
    // Clear inventory passively at the touch when it is in profit, or during the wind-down.
    let tp = cfg.take_profit_sigma * sig;
    if x.inv_qty < 0.0 && (x.exit_only || (x.avg_entry > 0.0 && mid <= x.avg_entry * (1.0 - tp))) {
        bid = bid.max(x.bid1);
    }
    // A long below its target is the position we want: no take-profit at the touch until it exceeds the target.
    if x.inv_qty > 0.0 && (x.exit_only || (x.avg_entry > 0.0 && mid >= x.avg_entry * (1.0 + tp) && inv_usd > target)) {
        ask = ask.min(x.ask1);
    }
    let mut ask_px = round_to_step(ask, tick, Round::Up).max(x.bid1 + tick);
    let mut bid_px = round_to_step(bid, tick, Round::Down).min(x.ask1 - tick);
    if cfg.reduce_min_edge_bps > 0.0 && !x.exit_only && x.avg_entry > 0.0 {
        let e = cfg.reduce_min_edge_bps / 1e4;
        if x.inv_qty > 0.0 {
            ask_px = ask_px.max(round_to_step(x.avg_entry * (1.0 + e), tick, Round::Up));
        } else if x.inv_qty < 0.0 {
            bid_px = bid_px.min(round_to_step(x.avg_entry * (1.0 - e), tick, Round::Down));
        }
    }
    let lot = min_lot(rule, mid);
    let flat = x.inv_qty.abs() < rule.qty_step * 0.5;
    // Caps apply after a full fill; one lot from flat is always allowed (some minimum lots exceed a small cap).
    // Net cap counts resting adding orders elsewhere as if filled (they can all fill in one move).
    let adds_ok = |after: f64| {
        let within = if after < 0.0 { -after <= cap_s } else { after <= cap_l };
        let pending = if after < inv_usd { -x.pend_sell_others_usd } else { x.pend_buy_others_usd };
        let net_after = x.net_others_usd + pending + after;
        let net_before = x.net_others_usd + inv_usd;
        let net_ok = cfg.max_net_usd <= 0.0 || net_after.abs() <= cfg.max_net_usd || net_after.abs() < net_before.abs();
        (within || flat) && net_ok && x.gross_others_usd + after.abs() <= cfg.max_gross_usd
    };
    // Reducing side scales with inventory: 1 lot near flat, up to reduce_lots_max lots at the cap.
    let k = cfg.reduce_lots_max.max(1) as f64;
    let reduce_qty = |q: f64| {
        let lots = 1.0 + ((k - 1.0) * inv_frac.abs()).round();
        let want = round_to_step((lots * lot).min(q.abs()), rule.qty_step, Round::Down);
        if want <= 0.0 { round_to_step(q.abs(), rule.qty_step, Round::Down) } else { want }
    };

    // Quantity as an exposure factor on the adding side: 1..add_lots_max lots, more when far from the target or near a level.
    let n_max = cfg.add_lots_max.max(1) as f64;
    let add_qty = |score: f64, sell: bool| {
        let n = 1.0 + ((n_max - 1.0) * score.clamp(0.0, 1.0)).round();
        let mut q = n * lot;
        // more than one lot must fit inside the caps (the from-flat exception covers a single lot only)
        let fits = |q: f64| {
            let after = if sell { inv_usd - q * mid } else { inv_usd + q * mid };
            adds_ok(after) && (if after < 0.0 { -after <= cap_s } else { after <= cap_l })
        };
        while q > lot + 1e-12 && !fits(q) {
            q -= lot;
        }
        round_to_step(q, rule.qty_step, Round::Down).max(lot)
    };
    let tilt = x.level_tilt.clamp(-1.0, 1.0);
    let score_bid = cfg.add_inventory_weight * (-dev_frac).max(0.0) + cfg.add_level_weight * tilt.max(0.0);
    let score_ask = cfg.add_inventory_weight * dev_frac.max(0.0) + cfg.add_level_weight * (-tilt).max(0.0);
    let ask = if x.inv_qty > 0.0 && !flat {
        // Selling a long: outside the wind-down only the excess over the target scales up; at or below it, one lot.
        let q = if x.exit_only || target <= 0.0 { reduce_qty(x.inv_qty) } else if inv_usd > target { reduce_qty(x.inv_qty).min(round_to_step(dev / mid, rule.qty_step, Round::Up).max(lot)) } else { lot.min(x.inv_qty) };
        Some(Quote { price: ask_px, qty: round_to_step(q, rule.qty_step, Round::Down).max(rule.qty_step), reduce_only: true })
    } else if !pause_adding && adds_ok(inv_usd - lot * mid) {
        Some(Quote { price: ask_px, qty: add_qty(score_ask, true), reduce_only: false })
    } else {
        None
    };
    let bid = if x.inv_qty < 0.0 && !flat {
        Some(Quote { price: bid_px, qty: reduce_qty(x.inv_qty), reduce_only: true })
    } else if !pause_adding && adds_ok(inv_usd + lot * mid) {
        Some(Quote { price: bid_px, qty: add_qty(score_bid, false), reduce_only: false })
    } else {
        None
    };
    (bid.filter(|q| q.qty > 0.0), ask.filter(|q| q.qty > 0.0))
}

/// Log the first few stale-top events and then every 20th, so a slow link is visible without flooding the log.
fn stale_log_due(n: u64) -> bool {
    n < 5 || n % 20 == 0
}

// ---------------------------------------------------------------- fill ledger

/// Strategy cash from our own executions (trades and funding) on the configured symbols, deduplicated by execId,
/// in total and per symbol.
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
    pub cash_by: HashMap<String, f64>,
    pub fees_by: HashMap<String, f64>,
    pub fills_by: HashMap<String, u32>,
    pub traded_by: HashMap<String, f64>,
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
        let sym = x["symbol"].as_str().unwrap_or_default().to_string();
        let fee = num(&x["execFee"]);
        if x["execType"].as_str().unwrap_or("Trade") == "Funding" {
            self.cash -= fee;
            self.funding -= fee;
            *self.cash_by.entry(sym).or_default() -= fee;
            return false;
        }
        let sign = if x["side"] == "Sell" { 1.0 } else { -1.0 };
        let flow = sign * num(&x["execValue"]) - fee;
        self.cash += flow;
        self.fees += fee;
        *self.cash_by.entry(sym.clone()).or_default() += flow;
        *self.fees_by.entry(sym.clone()).or_default() += fee;
        *self.fills_by.entry(sym.clone()).or_default() += 1;
        *self.traded_by.entry(sym.clone()).or_default() += num(&x["execValue"]);
        *self.inv.entry(sym).or_default() -= sign * num(&x["execQty"]);
        self.fills.push(x.clone());
        true
    }

    /// An adopted position (restart after a crash) enters the ledger at its venue average entry, as if we had
    /// traded it: without this the cover shows up as phantom inventory and the stops see P&L that does not exist.
    pub fn seed(&mut self, sym: &str, signed_qty: f64, avg_price: f64) {
        if signed_qty == 0.0 || avg_price.is_nan() || avg_price <= 0.0 {
            return;
        }
        let flow = -signed_qty * avg_price;
        self.cash += flow;
        *self.cash_by.entry(sym.to_string()).or_default() += flow;
        *self.inv.entry(sym.to_string()).or_default() += signed_qty;
    }

    /// P&L = cash + ledger inventory marked at mid.
    pub fn pnl(&self, mids: &HashMap<String, f64>) -> f64 {
        self.cash + self.inv.iter().map(|(s, q)| q * mids.get(s).copied().unwrap_or(0.0)).sum::<f64>()
    }

    pub fn pnl_of(&self, sym: &str, mid: f64) -> f64 {
        self.cash_by.get(sym).copied().unwrap_or(0.0) + self.inv.get(sym).copied().unwrap_or(0.0) * mid
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

/// Per-symbol state carried between ticks (public so `atx mm-replay` runs the same state machine).
#[derive(Default, Clone, Debug)]
pub struct SymState {
    pub sigma: f64,
    pub ac1: f64,
    pub vr15: f64,
    pub flow: f64,
    pub tox: f64,
    pub last_mid: f64,
    pub last_mid_ms: i64,
    pub paused_until_ms: i64,
    pub stopped: bool,
    pub levels: Levels,
    /// Per-interval micro sigma from recent trades (0 = not available yet).
    pub micro: f64,
    /// Log return and span (s) of the recent-trade window (oldest to newest trade), for the trend guard.
    pub trend_ret: f64,
    pub trend_span_s: f64,
}

// ---------------------------------------------------------------- per-tick decision steps (pure; shared by the live loop and `atx mm-replay`)

/// Slow (`sigma_lookback_min`) and fast (`sigma_fast_min`) 1h sigma from 1m closes; the quoter uses the larger.
pub fn sigma_pair(cfg: &MmCfg, closes: &[f64]) -> (f64, f64) {
    (sigma_1h(closes, cfg.sigma_floor_bps), sigma_last(closes, cfg.sigma_fast_min, cfg.sigma_floor_bps))
}

/// Pivot levels from 15m highs / lows plus the structural zones (a zone above the mid adds its top as resistance, one
/// below adds its bottom as support).
pub fn levels_with_zones(cfg: &MmCfg, highs: &[f64], lows: &[f64], zones: &[(f64, f64)], mid_now: f64) -> Levels {
    let mut lv = levels(highs, lows, cfg.level_pivot_bars);
    for (lo_z, hi_z) in zones {
        if *lo_z > mid_now {
            lv.res.push(*hi_z); // ask just beyond the zone's top: fill on the sweep through it
        } else if *hi_z < mid_now {
            lv.sup.push(*lo_z);
        }
    }
    lv
}

/// Lookback (intervals) of the micro sigma: `micro_lookback`, 40 when unset.
pub fn micro_lookback(cfg: &MmCfg) -> usize {
    if cfg.micro_lookback == 0 { 40 } else { cfg.micro_lookback }
}

/// Recent trades fetched per order-flow refresh: `flow_trades_limit` (max 1000), 500 when unset.
pub fn flow_trades_limit(cfg: &MmCfg) -> u32 {
    if cfg.flow_trades_limit == 0 { 500 } else { cfg.flow_trades_limit.min(1000) }
}

/// Log return and span (s) from the oldest to the newest of `(time_ms, price)` points (first oldest, last newest on ties).
pub fn trend_window(pts: &[(i64, f64)]) -> Option<(f64, f64)> {
    let (old, new) = (pts.iter().min_by_key(|p| p.0)?, pts.iter().max_by_key(|p| p.0)?);
    Some(((new.1 / old.1).ln(), (new.0 - old.0) as f64 / 1000.0))
}

/// Order-flow signal in [-1, 1]: half trade imbalance over `flow_window_s`, half book imbalance within `flow_depth_bps`.
pub fn flow_signal(cfg: &MmCfg, trades: &[(i64, f64, f64)], now_ms: i64, bids: &[(f64, f64)], asks: &[(f64, f64)], mid: f64) -> f64 {
    0.5 * trade_imbalance(trades, now_ms - cfg.flow_window_s * 1000) + 0.5 * book_imbalance(bids, asks, mid, cfg.flow_depth_bps)
}

/// Own-fill markout in 1h-sigma units and the updated toxicity EWMA.
pub fn tox_update(cfg: &MmCfg, tox: f64, sign: f64, px: f64, mid: f64, sigma: f64) -> (f64, f64) {
    let m = sign * (mid - px) / px / sigma.max(1e-6);
    (m, (1.0 - cfg.tox_alpha) * tox + cfg.tox_alpha * m)
}

/// Shock pause: Some(|log move|) when the mid moved more than `shock_pause_sigma` 1h sigma scaled to the real time since
/// the last observation (at least `refresh_s`).
pub fn shock_jump(cfg: &MmCfg, e: &SymState, mid: f64, now_ms: i64) -> Option<f64> {
    if e.last_mid > 0.0 && cfg.shock_pause_sigma > 0.0 && e.sigma > 0.0 {
        // scale by the real time since the last observation (a 50-symbol tick takes much longer than refresh_s)
        let dt_s = ((now_ms - e.last_mid_ms) as f64 / 1000.0).max(cfg.refresh_s as f64);
        let jump = (mid / e.last_mid).ln().abs();
        if jump > cfg.shock_pause_sigma * e.sigma * (dt_s / 3600.0).sqrt() {
            return Some(jump);
        }
    }
    None
}

/// The sigma quotes are expressed in: the micro sigma scaled to `micro_horizon_s` (floored) when micro quoting is on and
/// a micro estimate exists, else the 1h sigma.
pub fn quote_sigma(cfg: &MmCfg, micro: f64, sigma_1h: f64) -> f64 {
    if cfg.micro_interval_s > 0 && micro > 0.0 {
        let h = if cfg.micro_horizon_s > 0.0 { cfg.micro_horizon_s } else { cfg.micro_interval_s as f64 };
        (micro * (h / cfg.micro_interval_s as f64).sqrt()).max(cfg.micro_floor_bps / 1e4)
    } else {
        sigma_1h
    }
}

/// Signed discretionary view of a symbol (bias, or the sign of its inventory target), for the trend guard.
pub fn view_of(cfg: &MmCfg, sym: &str, mid: f64) -> f64 {
    cfg.bias.get(sym).map(|b| b.active_sigma(mid) + b.target_usd.unwrap_or(0.0).signum() * 1e-9).unwrap_or(0.0)
}

/// Portfolio context from the OTHER symbols (all zero in a single-symbol replay).
#[derive(Clone, Copy, Debug, Default)]
pub struct Others {
    pub gross_usd: f64,
    pub net_usd: f64,
    pub pend_sell_usd: f64,
    pub pend_buy_usd: f64,
}

/// The quote rule's inputs for one symbol from its state, book top, position `(signed qty, avg entry)` and portfolio.
#[allow(clippy::too_many_arguments)]
pub fn symbol_inputs(cfg: &MmCfg, sym: &str, e: &SymState, bid1: f64, ask1: f64, pos: (f64, f64), others: Others, exit_only: bool, paused: bool, guard: bool) -> Inputs {
    let mid = 0.5 * (bid1 + ask1);
    Inputs {
        bid1,
        ask1,
        sigma: quote_sigma(cfg, e.micro, e.sigma),
        level_tilt: if cfg.level_shift_sigma != 0.0 || cfg.add_level_weight != 0.0 { level_tilt(mid, e.sigma, &e.levels, cfg.level_reach_sigma) } else { 0.0 },
        inv_qty: pos.0,
        avg_entry: pos.1,
        gross_others_usd: others.gross_usd,
        flow: e.flow,
        tox: e.tox,
        exit_only,
        paused,
        bias_sigma: if guard { 0.0 } else { cfg.bias.get(sym).map(|b| b.active_sigma(mid)).unwrap_or(0.0) },
        cap_short_usd: cfg.bias.get(sym).and_then(|b| b.max_short_usd).unwrap_or(0.0),
        cap_long_usd: cfg.bias.get(sym).and_then(|b| b.max_long_usd).unwrap_or(0.0),
        target_usd: if guard { 0.0 } else { cfg.bias.get(sym).and_then(|b| b.target_usd).unwrap_or(0.0) },
        net_others_usd: others.net_usd,
        pend_sell_others_usd: others.pend_sell_usd,
        pend_buy_others_usd: others.pend_buy_usd,
    }
}

/// A resting quote as the replace rule sees it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Resting {
    pub price: f64,
    pub leaves_qty: f64,
    pub reduce_only: bool,
    pub updated_ms: i64,
}

/// Keep a resting quote unless it is wrong in kind or drifted far enough (in quote-sigma units) for long enough.
pub fn needs_replace(cfg: &MmCfg, rule: &InstrumentRules, want: Option<Quote>, cur: Option<&Resting>, mid: f64, sigma_quote: f64, now_ms: i64) -> bool {
    match (want, cur) {
        (Some(q), Some(o)) => {
            let drift = (o.price - q.price).abs() / mid / sigma_quote;
            let age_s = (now_ms - o.updated_ms) / 1000;
            let kind = o.reduce_only != q.reduce_only || (o.leaves_qty - q.qty).abs() > rule.qty_step * 0.5;
            kind || drift > 1.0 || (drift > cfg.requote_sigma && age_s >= cfg.min_rest_s)
        }
        (None, Some(_)) => true,
        _ => false,
    }
}

// ---------------------------------------------------------------- weekend alt quoting rules (H-MMW1-4; pure, shared by `atx mm-replay`)

/// H-MMW1 level ladder: quote prices at structural zones (`levels_file` / scripts/levels.py, `(lo, hi)`) lying
/// `band.0`..`band.1` x `sigma_1h` from the mid: a support zone below the mid quotes a bid at its top edge (the first
/// price of the zone a falling market touches), a resistance zone above quotes an ask at its bottom edge. Distances are
/// log distances of that edge from the mid. Returns (bids, asks), nearest to the mid first, duplicates removed.
pub fn ladder_levels(zones: &[(f64, f64)], mid: f64, sigma_1h: f64, band: (f64, f64)) -> (Vec<f64>, Vec<f64>) {
    let (mut bids, mut asks) = (vec![], vec![]);
    if !(mid > 0.0 && sigma_1h > 0.0) {
        return (bids, asks);
    }
    let inside = |px: f64| {
        let d = (px / mid).ln().abs() / sigma_1h;
        px > 0.0 && d >= band.0 && d <= band.1
    };
    for &(lo, hi) in zones {
        if hi < mid && inside(hi) {
            bids.push(hi);
        } else if lo > mid && inside(lo) {
            asks.push(lo);
        }
    }
    bids.sort_by(|a, b| b.total_cmp(a));
    asks.sort_by(|a, b| a.total_cmp(b));
    bids.dedup();
    asks.dedup();
    (bids, asks)
}

/// Stop of a level-ladder position: `k` x `sigma_risk` (log units, the unshrunk risk sigma, R-VOL-2) beyond the level.
pub fn level_stop(level: f64, long: bool, k: f64, sigma_risk: f64) -> f64 {
    if long { level * (-k * sigma_risk).exp() } else { level * (k * sigma_risk).exp() }
}

/// Take-profit of a level-ladder fill: the mid at fill time on the tick grid, rounded away from the entry, and never less
/// than one tick beyond the entry (a take-profit is never a loss).
pub fn ladder_tp(entry: f64, mid_fill: f64, long: bool, tick: f64) -> f64 {
    if long {
        round_to_step(mid_fill, tick, Round::Up).max(round_to_step(entry + tick, tick, Round::Nearest))
    } else {
        round_to_step(mid_fill, tick, Round::Down).min(round_to_step(entry - tick, tick, Round::Nearest))
    }
}

/// H-MMW2: the larger of the BTC 1 s and 5 s log returns in units of the BTC 1h sigma scaled to that horizon.
pub fn btc_move_z(r1: f64, r5: f64, sigma_1h: f64) -> f64 {
    if sigma_1h <= 0.0 {
        return 0.0;
    }
    (r1.abs() / (sigma_1h * (1.0f64 / 3600.0).sqrt())).max(r5.abs() / (sigma_1h * (5.0f64 / 3600.0).sqrt()))
}

/// BTC tape state for the H-MMW2 pull: recent prints for the 1 s / 5 s returns and 1m closes for the 1h sigma (the robust
/// `sigma_1h` of the last 240 closed minutes, refreshed once a minute; `floor_bps` floor).
#[derive(Clone, Debug, Default)]
pub struct BtcTape {
    recent: std::collections::VecDeque<(i64, f64)>,
    closes: std::collections::BTreeMap<i64, f64>,
    pub sigma: f64,
    sigma_min: i64,
    pub floor_bps: f64,
}

impl BtcTape {
    pub fn new(floor_bps: f64) -> Self {
        Self { floor_bps, sigma_min: i64::MIN, ..Default::default() }
    }

    /// Seed 1m closes (warm-up bars of earlier days): minute start ms -> close.
    pub fn seed_close(&mut self, minute_ms: i64, close: f64) {
        self.closes.insert(minute_ms, close);
    }

    fn px_at(&self, t: i64) -> Option<f64> {
        let i = self.recent.partition_point(|x| x.0 <= t);
        if i == 0 { None } else { Some(self.recent[i - 1].1) }
    }

    /// One BTC print: returns its move z (`btc_move_z`; 0 while the sigma or the 5 s history is not available).
    pub fn on_print(&mut self, ts: i64, px: f64) -> f64 {
        if px <= 0.0 {
            return 0.0;
        }
        let minute = ts.div_euclid(60_000) * 60_000;
        if minute != self.sigma_min {
            self.sigma_min = minute;
            let c: Vec<f64> = self.closes.range(minute - 240 * 60_000..minute).map(|x| *x.1).collect();
            self.sigma = if c.len() >= 31 { sigma_1h(&c, self.floor_bps) } else { 0.0 };
        }
        self.closes.insert(minute, px);
        while self.recent.len() > 1 && self.recent[1].0 < ts - 6_000 {
            self.recent.pop_front();
        }
        let (p1, p5) = (self.px_at(ts - 1_000), self.px_at(ts - 5_000));
        self.recent.push_back((ts, px));
        match (p1, p5) {
            (Some(a), Some(b)) if self.sigma > 0.0 => btc_move_z((px / a).ln(), (px / b).ln(), self.sigma),
            _ => 0.0,
        }
    }
}

/// Top-`n` book imbalance in quantity: (bid qty - ask qty) / total over the first `n` levels of each side.
pub fn top_imbalance(bids: &[(f64, f64)], asks: &[(f64, f64)], n: usize) -> f64 {
    let b: f64 = bids.iter().take(n).map(|x| x.1).sum();
    let a: f64 = asks.iter().take(n).map(|x| x.1).sum();
    if a + b > 0.0 { (b - a) / (a + b) } else { 0.0 }
}

/// H-MMW3: the side an order-flow signal threatens. Buying pressure beyond the threshold (> +t) threatens our ask (we would
/// sell before a rise): Some(true) = ask; selling pressure (< -t) threatens our bid: Some(false). None inside.
pub fn threatened_side(signal: f64, threshold: f64) -> Option<bool> {
    if threshold <= 0.0 {
        None
    } else if signal > threshold {
        Some(true)
    } else if signal < -threshold {
        Some(false)
    } else {
        None
    }
}

/// H-MMW3 side pull on target quotes: the threatened side is withdrawn (`widen_sigma` <= 0) or moved `widen_sigma` x
/// `sigma` further from the mid (ask up, bid down, on the tick grid).
pub fn side_pull(bid: Option<Quote>, ask: Option<Quote>, threatened_ask: Option<bool>, widen_sigma: f64, sigma: f64, tick: f64) -> (Option<Quote>, Option<Quote>) {
    match threatened_ask {
        None => (bid, ask),
        Some(true) if widen_sigma <= 0.0 => (bid, None),
        Some(false) if widen_sigma <= 0.0 => (None, ask),
        Some(true) => (bid, ask.map(|q| Quote { price: round_to_step(q.price * (1.0 + widen_sigma * sigma), tick, Round::Up), ..q })),
        Some(false) => (bid.map(|q| Quote { price: round_to_step(q.price * (1.0 - widen_sigma * sigma), tick, Round::Down), ..q }), ask),
    }
}

/// H-MMW4: join the touch only when the others' size ahead of us is below `q` x the median touch size.
pub fn short_queue(ahead: f64, median_level: f64, q: f64) -> bool {
    q > 0.0 && median_level > 0.0 && ahead < q * median_level
}

/// H-MMW4 leave rule: the size behind us now is below (1 - `frac`) x its maximum over the last `window_ms` (it fell by
/// more than `frac`). `hist` = (time, behind) observations, oldest first, the last one current.
pub fn behind_thinned(hist: &[(i64, f64)], now: i64, window_ms: i64, frac: f64) -> bool {
    let Some(&(_, cur)) = hist.last() else { return false };
    let mx = hist.iter().filter(|x| x.0 >= now - window_ms).map(|x| x.1).fold(0.0, f64::max);
    mx > 0.0 && cur < (1.0 - frac) * mx
}

/// Median of a sample (NaN when empty).
pub fn median(v: &[f64]) -> f64 {
    if v.is_empty() {
        return f64::NAN;
    }
    let mut s = v.to_vec();
    s.sort_by(|a, b| a.total_cmp(b));
    let n = s.len();
    if n % 2 == 1 { s[n / 2] } else { 0.5 * (s[n / 2 - 1] + s[n / 2]) }
}

/// H-MMW1 arm A2 liquidation-cluster proxy from a public tape that carries no liquidation flag (pre-declared
/// 2026-11-21): at least `min_prints` same-side (same aggressor) prints within `window_ms`, spanning at least
/// `min_levels` distinct price levels, with summed notional >= `min_notional` (10 x the previous day's median print
/// notional). A trigger empties that side's window, so the next cluster needs fresh prints.
#[derive(Clone, Debug)]
pub struct LiqProxy {
    pub window_ms: i64,
    pub min_prints: usize,
    pub min_levels: usize,
    pub min_notional: f64,
    buf: [std::collections::VecDeque<(i64, i64, f64)>; 2],
}

impl LiqProxy {
    pub fn new(window_ms: i64, min_prints: usize, min_levels: usize, min_notional: f64) -> Self {
        Self { window_ms, min_prints, min_levels, min_notional, buf: Default::default() }
    }

    /// One print (time, price tick, notional USD, buy aggressor): true when it completes a cluster.
    pub fn on_print(&mut self, ts: i64, tick: i64, notional: f64, buy: bool) -> bool {
        if self.min_notional <= 0.0 {
            return false;
        }
        let b = &mut self.buf[buy as usize];
        while b.front().is_some_and(|x| x.0 < ts - self.window_ms) {
            b.pop_front();
        }
        b.push_back((ts, tick, notional));
        if b.len() < self.min_prints {
            return false;
        }
        let total: f64 = b.iter().map(|x| x.2).sum();
        if total < self.min_notional {
            return false;
        }
        let mut ticks: Vec<i64> = b.iter().map(|x| x.1).collect();
        ticks.sort_unstable();
        ticks.dedup();
        if ticks.len() < self.min_levels {
            return false;
        }
        b.clear();
        true
    }
}

// ---------------------------------------------------------------- run

pub async fn run(ctx: &Ctx, until: &str, config: Option<&str>, skip_busy: bool, adopt: bool, dry_run: bool, yes: bool) -> anyhow::Result<()> {
    let cfg_path = ctx.s.root.join(config.unwrap_or("config/mm.yaml"));
    let cfg: MmCfg = serde_yaml::from_str(&std::fs::read_to_string(&cfg_path).with_context(|| cfg_path.display().to_string())?).with_context(|| cfg_path.display().to_string())?;
    let env = ctx.s.environment;
    let deadline = parse_ts(until)?;
    anyhow::ensure!(deadline > now_ms(), "--until {until} is in the past");
    if env == Environment::Mainnet && !dry_run {
        anyhow::ensure!(yes, "atx mm on mainnet places real orders; re-run with --yes");
    }
    if !dry_run {
        ctx.s.ensure_execution_allowed()?;
    }
    let mut symbols: Vec<String> = cfg.symbols.iter().map(|s| s.trim().to_uppercase()).collect();
    let venue = Venue::new(ctx)?;
    let t = venue.trading();
    let mrest = ctx.market_rest()?;
    let market = MarketData::new(&mrest, ctx.category());
    let cat = ctx.category().to_string();
    let rules = venue.rules(&symbols.iter().cloned().collect::<HashSet<_>>()).await?;
    let unknown: Vec<String> = symbols.iter().filter(|s| !rules.contains_key(*s)).cloned().collect();
    anyhow::ensure!(unknown.is_empty() || skip_busy, "not trading instruments: {unknown:?}");
    symbols.retain(|s| rules.contains_key(s));

    // Refuse (or with --skip-busy, drop) symbols that carry anything we did not open.
    let positions = t.positions().await?;
    let orders = t.open_orders(None).await?;
    // --adopt: positions on configured symbols are ours (a restart after a crash) and are managed passively.
    let mut busy: Vec<String> = if adopt { vec![] } else { positions.iter().filter(|p| symbols.contains(&p.symbol)).map(|p| p.symbol.clone()).collect() };
    for o in orders.iter().filter(|o| symbols.iter().any(|s| o["symbol"] == s.as_str())) {
        let link = o["orderLinkId"].as_str().unwrap_or_default();
        let sym = o["symbol"].as_str().unwrap_or_default().to_string();
        if link.starts_with("mm") {
            if !dry_run {
                t.cancel_order(&sym, o["orderId"].as_str().unwrap_or_default()).await?;
            }
        } else {
            busy.push(sym);
        }
    }
    busy.sort();
    busy.dedup();
    if !busy.is_empty() {
        anyhow::ensure!(skip_busy, "symbols with positions or foreign orders: {busy:?} (flatten them, or pass --skip-busy)");
        println!("skipping busy symbols: {}", busy.join(","));
        symbols.retain(|s| !busy.contains(s));
    }
    anyhow::ensure!(!symbols.is_empty(), "no symbols left to quote");

    let start_ms = now_ms();
    let run_id = format!("mm{}", chrono::Utc::now().format("%m%d%H%M"));
    let out_dir = ctx.s.abs(&ctx.s.paths.data_root).join("mm").join(format!("{}-{run_id}", env.as_str()));
    std::fs::create_dir_all(&out_dir)?;
    // The safety wrapper (scripts/mm_run.sh) flattens exactly these symbols if this process dies.
    std::fs::write(
        out_dir.join("symbols.txt"),
        symbols.join("
"),
    )?;
    let journal = Journal::new(journal_root(&ctx.s), &run_id);
    let equity0 = t.wallet().await?.total_equity;
    let pause = Duration::from_millis(cfg.order_pause_ms);
    if !dry_run {
        for s in &symbols {
            let lev = rules[s].max_leverage.min(cfg.leverage);
            if let Err(e) = t.set_leverage(s, &format!("{lev}")).await {
                println!("{s}: set leverage {lev} failed: {e:#}");
            }
            tokio::time::sleep(pause).await;
        }
    }
    journal.write("session_start", json!({"source": "atx mm", "config_file": cfg_path.display().to_string(), "environment": env.as_str(), "dry_run": dry_run, "until": until, "wallet_equity": equity0, "symbols": symbols, "skipped": busy, "config": cfg}))?;
    println!("atx mm {run_id} on {} ({}) until {until}  wallet {equity0:.2}  {} symbols", env.as_str(), if dry_run { "DRY RUN" } else { "LIVE QUOTES" }, symbols.len());

    let mut st: HashMap<String, SymState> = symbols.iter().map(|s| (s.clone(), SymState::default())).collect();
    // structural zones (optional): symbol -> [(lo, hi)]
    let zones: HashMap<String, Vec<(f64, f64)>> = match &cfg.levels_file {
        Some(f) => {
            let v: Value = serde_json::from_str(&std::fs::read_to_string(ctx.s.root.join(f)).with_context(|| f.clone())?)?;
            v.as_object()
                .map(|o| o.iter().map(|(k, zs)| (k.clone(), zs.as_array().map(|a| a.iter().map(|z| (num(&z["lo"]), num(&z["hi"]))).collect()).unwrap_or_default())).collect())
                .unwrap_or_default()
        }
        None => HashMap::new(),
    };
    for (s, b) in &cfg.bias {
        println!(
            "view {s}: bias {:+.2} sigma{}{} ({})",
            b.sigma,
            b.off_below.map(|l| format!(", off below {l}")).unwrap_or_default(),
            b.off_above.map(|h| format!(", off above {h}")).unwrap_or_default(),
            b.note
        );
    }
    let mut pending_markouts: Vec<(String, f64, f64, i64)> = vec![]; // (symbol, +1 buy / -1 sell, price, time)
    let mut sigma_at = 0i64;
    let mut ledger = Ledger { cursor_ms: start_ms, ..Default::default() };
    if adopt {
        for p in positions.iter().filter(|p| symbols.contains(&p.symbol) && p.size > 0.0) {
            let q = if p.side == "Sell" { -p.size } else { p.size };
            ledger.seed(&p.symbol, q, p.avg_price);
            println!("{}: adopted {q:+} @ {} into the ledger", p.symbol, p.avg_price);
        }
    }
    let mut broker = if dry_run { None } else { Some(Venue::broker(ctx, rules.clone())?) };
    let mut errors = 0u32;
    let mut tick_n = 0u64;
    let mut seq = 0u64;
    let mut flow_cursor = 0usize;
    let mut stale_n = 0u64;
    let mut guard_n = 0u64;
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
                    let (slow, fast) = sigma_pair(&cfg, &closes);
                    let (ac1, vr15) = mr_stats(&closes);
                    let k15 = market.klines(s, Freq::M15, PriceType::Last, end - cfg.level_lookback_h * 3_600_000, end).await?;
                    let (h, l): (Vec<f64>, Vec<f64>) = k15.iter().map(|x| (x.high, x.low)).unzip();
                    let e = st.get_mut(s).expect("state");
                    e.sigma = slow.max(fast);
                    e.ac1 = ac1;
                    e.vr15 = vr15;
                    e.levels = levels_with_zones(&cfg, &h, &l, zones.get(s).map(|z| z.as_slice()).unwrap_or(&[]), closes.last().copied().unwrap_or(0.0));
                    journal.write("snapshot", json!({"source": "atx mm", "symbol": s, "sigma_1h": e.sigma, "sigma_slow": slow, "sigma_fast": fast, "ac1_1m": ac1, "vr15": vr15}))?;
                }
                sigma_at = now_ms();
            }
            // Book tops: one whole-category call, or one small call per symbol on a short list (slow links); each top
            // keeps the time it was read so stale prices never add inventory. Order flow for a rotating batch.
            let mut tops: HashMap<String, (f64, f64)> = HashMap::new();
            let mut top_ms: HashMap<String, i64> = HashMap::new();
            let per_symbol = cfg.per_symbol_tickers_max > 0 && symbols.len() <= cfg.per_symbol_tickers_max;
            if per_symbol {
                for s in &symbols {
                    let t0 = now_ms();
                    if let Some(x) = market.ticker(s).await? {
                        tops.insert(x.symbol.clone(), (x.bid1, x.ask1));
                        top_ms.insert(x.symbol, t0);
                    }
                }
            } else {
                let all = market.tickers().await?;
                let at = now_ms();
                for x in all.into_iter().filter(|x| symbols.contains(&x.symbol)) {
                    top_ms.insert(x.symbol.clone(), at);
                    tops.insert(x.symbol, (x.bid1, x.ask1));
                }
            }
            let trades_limit = flow_trades_limit(&cfg);
            let batch = if cfg.flow_batch == 0 { symbols.len() } else { cfg.flow_batch.min(symbols.len()) };
            for k in 0..batch {
                let s = &symbols[(flow_cursor + k) % symbols.len()];
                let ob = mrest.get_public("/v5/market/orderbook", &[("category", cat.clone()), ("symbol", s.clone()), ("limit", "50".into())]).await?;
                let (bids, asks) = (pairs(&ob["b"]), pairs(&ob["a"]));
                let (Some(b1), Some(a1)) = (bids.first(), asks.first()) else { continue };
                let mid = 0.5 * (b1.0 + a1.0);
                let tr = mrest.get_public("/v5/market/recent-trade", &[("category", cat.clone()), ("symbol", s.clone()), ("limit", trades_limit.to_string())]).await?;
                let trades: Vec<(i64, f64, f64)> = tr["list"]
                    .as_array()
                    .map(|a| a.iter().map(|x| (num(&x["time"]) as i64, if x["side"] == "Buy" { 1.0 } else { -1.0 }, num(&x["price"]) * num(&x["size"]))).collect())
                    .unwrap_or_default();
                if cfg.micro_interval_s > 0 {
                    let px: Vec<(i64, f64)> = tr["list"].as_array().map(|a| a.iter().map(|x| (num(&x["time"]) as i64, num(&x["price"]))).collect()).unwrap_or_default();
                    st.get_mut(s).expect("state").micro = micro_sigma(&px, now_ms(), cfg.micro_interval_s, micro_lookback(&cfg)).unwrap_or(0.0);
                }
                if cfg.trend_guard_sigma > 0.0 {
                    let pts: Vec<(i64, f64)> = tr["list"].as_array().map(|a| a.iter().map(|x| (num(&x["time"]) as i64, num(&x["price"]))).filter(|p| p.1 > 0.0).collect()).unwrap_or_default();
                    if let Some((ret, span_s)) = trend_window(&pts) {
                        let e = st.get_mut(s).expect("state");
                        e.trend_ret = ret;
                        e.trend_span_s = span_s;
                    }
                }
                st.get_mut(s).expect("state").flow = flow_signal(&cfg, &trades, now_ms(), &bids, &asks, mid);
            }
            flow_cursor = (flow_cursor + batch) % symbols.len();
            let mids: HashMap<String, f64> = tops.iter().filter(|(_, b)| b.0 > 0.0 && b.1 > b.0).map(|(s, b)| (s.clone(), 0.5 * (b.0 + b.1))).collect();
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
                if let (Some(mid), Some(e)) = (mids.get(&s), st.get_mut(&s)) {
                    let m;
                    (m, e.tox) = tox_update(&cfg, e.tox, sign, px, *mid, e.sigma);
                    journal.write("fill", json!({"source": "atx mm", "symbol": s, "side": if sign > 0.0 { "Buy" } else { "Sell" }, "price": px, "exec_ms": ts, "markout_sigma": m, "horizon_s": cfg.tox_horizon_s, "tox_ewma": e.tox}))?;
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
            let gross_all: f64 = symbols.iter().map(notional).sum();
            let signed = |s: &String| inv[s].0 * mids.get(s).copied().unwrap_or(0.0);
            let net_all: f64 = symbols.iter().map(signed).sum();
            // resting inventory-adding (non reduce-only) quotes per symbol, USD notional
            let mut pend: HashMap<String, (f64, f64)> = HashMap::new(); // (sell, buy)
            for o in ours.iter().filter(|o| o["reduceOnly"] != true) {
                let e = pend.entry(o["symbol"].as_str().unwrap_or_default().to_string()).or_default();
                let usd = num(&o["leavesQty"]) * num(&o["price"]);
                if o["side"] == "Sell" { e.0 += usd } else { e.1 += usd }
            }
            let mut pend_sell_all: f64 = pend.values().map(|v| v.0).sum();
            let mut pend_buy_all: f64 = pend.values().map(|v| v.1).sum();
            for s in &symbols {
                let Some(&(mut bid1, mut ask1)) = tops.get(s) else { continue };
                let Some(&(mut mid)) = mids.get(s) else { continue };
                // Short list: re-read this symbol's top right before deciding its quotes; its age counts from the request
                // start, so a slow response is itself treated as stale.
                if per_symbol {
                    let t0 = now_ms();
                    if let Ok(Some(x)) = market.ticker(s).await
                        && x.bid1 > 0.0
                        && x.ask1 > x.bid1
                    {
                        (bid1, ask1, mid) = (x.bid1, x.ask1, 0.5 * (x.bid1 + x.ask1));
                        top_ms.insert(s.clone(), t0);
                    }
                }
                let rule = &rules[s];
                // per-symbol loss stop: cancel + flatten this symbol, then leave it alone
                if cfg.symbol_loss_limit_usd > 0.0 && !st[s].stopped && ledger.pnl_of(s, mid) <= -cfg.symbol_loss_limit_usd {
                    st.get_mut(s).expect("state").stopped = true;
                    if let Some(b) = broker.as_mut() {
                        let _ = t.cancel_all(s).await;
                        let _ = b.close(s).await;
                    }
                    journal.write("position_update", json!({"source": "atx mm", "symbol": s, "action": "symbol_stop", "pnl": ledger.pnl_of(s, mid)}))?;
                    println!("{s}: symbol loss stop at {:+.3} USD; flattened", ledger.pnl_of(s, mid));
                    continue;
                }
                if st[s].stopped {
                    continue;
                }
                // shock pause on a large mid move between ticks
                {
                    let e = st.get_mut(s).expect("state");
                    if let Some(jump) = shock_jump(&cfg, e, mid, now_ms()) {
                        e.paused_until_ms = now_ms() + cfg.shock_cooldown_s * 1000;
                        journal.write("snapshot", json!({"source": "atx mm", "symbol": s, "event": "shock_pause", "move_bps": jump * 1e4, "sigma_1h": e.sigma}))?;
                    }
                    e.last_mid = mid;
                    e.last_mid_ms = now_ms();
                }
                let age_s = (now_ms() - top_ms.get(s).copied().unwrap_or(0)) as f64 / 1000.0;
                let stale = cfg.max_quote_age_s > 0.0 && age_s > cfg.max_quote_age_s;
                if stale && stale_log_due(stale_n) {
                    println!("{s}: book top {age_s:.1}s old > {}s; not adding this tick", cfg.max_quote_age_s);
                }
                if stale {
                    stale_n += 1;
                    journal.write("snapshot", json!({"source": "atx mm", "symbol": s, "event": "stale_top", "age_s": age_s}))?;
                }
                let e = st[s].clone();
                let guard = trend_against_view(e.trend_ret, e.trend_span_s, e.sigma, cfg.trend_guard_sigma, view_of(&cfg, s, mid));
                if guard && stale_log_due(guard_n) {
                    println!("{s}: trend guard ({:+.2}% over {:.0}s against the view): bias and target off", e.trend_ret * 100.0, e.trend_span_s);
                }
                if guard {
                    guard_n += 1;
                }
                let others = Others {
                    gross_usd: gross_all - notional(s),
                    net_usd: net_all - signed(s),
                    pend_sell_usd: pend_sell_all - pend.get(s).map(|v| v.0).unwrap_or(0.0),
                    pend_buy_usd: pend_buy_all - pend.get(s).map(|v| v.1).unwrap_or(0.0),
                };
                let x = symbol_inputs(&cfg, s, &e, bid1, ask1, inv[s], others, exit_only, now_ms() < e.paused_until_ms || stale, guard);
                let (bid, ask) = target_quotes(&cfg, rule, &x, &e.levels);
                // running totals: later symbols in this tick see the adding quotes just decided here
                {
                    let old = pend.get(s).copied().unwrap_or((0.0, 0.0));
                    let new_s = ask.filter(|q| !q.reduce_only).map(|q| q.qty * q.price).unwrap_or(0.0);
                    let new_b = bid.filter(|q| !q.reduce_only).map(|q| q.qty * q.price).unwrap_or(0.0);
                    pend_sell_all += new_s - old.0;
                    pend_buy_all += new_b - old.1;
                    pend.insert(s.clone(), (new_s, new_b));
                }
                for (side, want) in [("Buy", bid), ("Sell", ask)] {
                    let resting: Vec<&Value> = ours.iter().filter(|o| o["symbol"] == s.as_str() && o["side"] == side).collect();
                    for o in resting.iter().skip(1) {
                        if !dry_run {
                            if let Err(e) = t.cancel_order(s, o["orderId"].as_str().unwrap_or_default()).await {
                                println!("{s}: cancel skipped ({e:#}); likely filled or already gone");
                            }
                            tokio::time::sleep(pause).await;
                        }
                    }
                    let cur = resting.first();
                    // Keep a resting quote unless it is wrong in kind or drifted far enough for long enough.
                    let rest = cur.map(|o| Resting { price: num(&o["price"]), leaves_qty: num(&o["leavesQty"]), reduce_only: o["reduceOnly"] == true, updated_ms: num(&o["updatedTime"]) as i64 });
                    let replace = needs_replace(&cfg, rule, want, rest.as_ref(), mid, x.sigma, now_ms());
                    if let (true, Some(o)) = (replace, cur) {
                        if !dry_run {
                            if let Err(e) = t.cancel_order(s, o["orderId"].as_str().unwrap_or_default()).await {
                                println!("{s}: cancel skipped ({e:#}); likely filled or already gone");
                            }
                            tokio::time::sleep(pause).await;
                        }
                        journal.write("order", json!({"source": "atx mm", "environment": env.as_str(), "dry_run": dry_run, "symbol": s, "side": side, "action": "cancel", "order_id": o["orderId"], "price": o["price"]}))?;
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
                        let id = if dry_run {
                            "dry".to_string()
                        } else {
                            let r = t.create_order(&req).await;
                            tokio::time::sleep(pause).await;
                            match r {
                                Ok(a) => a.order_id,
                                Err(e) => {
                                    println!("{s} {side}: order rejected: {e:#}");
                                    continue;
                                }
                            }
                        };
                        journal.write("order", json!({"source": "atx mm", "environment": env.as_str(), "dry_run": dry_run, "symbol": s, "side": side, "action": "place", "order_id": id, "link_id": req.order_link_id, "price": q.price, "qty": q.qty, "reduce_only": q.reduce_only, "bid1": bid1, "ask1": ask1, "sigma_quote": x.sigma, "sigma_1h": e.sigma, "level_tilt": x.level_tilt, "flow": x.flow, "tox": x.tox, "inventory": x.inv_qty, "exit_only": exit_only, "paused": x.paused}))?;
                    }
                }
            }
            if tick_n % 6 == 0 {
                let mut board: Vec<Value> = symbols
                    .iter()
                    .map(|s| {
                        let e = &st[s];
                        let mid = mids.get(s).copied().unwrap_or(0.0);
                        let traded = ledger.traded_by.get(s).copied().unwrap_or(0.0);
                        let p = ledger.pnl_of(s, mid);
                        let spread = tops.get(s).map(|b| if b.0 > 0.0 { (b.1 - b.0) / mid * 1e4 } else { f64::NAN }).unwrap_or(f64::NAN);
                        json!({"symbol": s, "pnl_usd": p, "fills": ledger.fills_by.get(s).copied().unwrap_or(0), "traded_usd": traded,
                               "pnl_bps_traded": if traded > 0.0 { p / traded * 1e4 } else { 0.0 }, "fees": ledger.fees_by.get(s).copied().unwrap_or(0.0),
                               "inventory_usd": inv[s].0 * mid, "tox": e.tox, "ac1_1m": e.ac1, "vr15": e.vr15,
                               "mr_score": 0.5 * (-e.ac1) + 0.5 * (1.0 - e.vr15), "sigma_1h_bp": e.sigma * 1e4, "spread_bp": spread,
                               "status": if e.stopped { "stopped" } else if now_ms() < e.paused_until_ms { "paused" } else { "active" }})
                    })
                    .collect();
                board.sort_by(|a, b| b["pnl_usd"].as_f64().unwrap_or(0.0).partial_cmp(&a["pnl_usd"].as_f64().unwrap_or(0.0)).unwrap_or(std::cmp::Ordering::Equal));
                std::fs::write(out_dir.join("leaderboard.json"), serde_json::to_string_pretty(&json!({"ts_ms": now_ms(), "pnl": pnl, "fills": ledger.fills.len(), "fees": ledger.fees, "gross_usd": gross_all, "rows": board}))?)?;
                let fmt = |r: &Value| format!("{} {:+.3}({})", r["symbol"].as_str().unwrap_or("").trim_end_matches("USDT"), r["pnl_usd"].as_f64().unwrap_or(0.0), r["fills"]);
                let top: Vec<String> = board.iter().take(3).map(fmt).collect();
                let bot: Vec<String> = board.iter().rev().take(3).map(fmt).collect();
                let active = board.iter().filter(|r| r["status"] == "active").count();
                println!("{} pnl {pnl:+.3} fills {} fees {:.3} gross ${gross_all:.0} net ${net_all:+.0} active {active}/{}{} | tick {:.1}s stale {stale_n} | best {} | worst {}", chrono::Utc::now().format("%H:%M:%S"),
                         ledger.fills.len(), ledger.fees, symbols.len(), if exit_only { " EXIT" } else { "" }, (now_ms() - now) as f64 / 1000.0, top.join(" "), bot.join(" "));
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
    if let Some(b) = broker.as_mut() {
        for attempt in 0..3 {
            for s in &symbols {
                if let Err(e) = t.cancel_all(s).await {
                    println!("{s}: cancel_all failed: {e:#}");
                }
                if let Err(e) = b.close(s).await {
                    println!("{s}: close failed: {e:#}");
                }
                tokio::time::sleep(pause).await;
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
    let per_symbol: Value = symbols
        .iter()
        .map(|s| (s.clone(), json!({"pnl": ledger.cash_by.get(s).copied().unwrap_or(0.0), "fills": ledger.fills_by.get(s).copied().unwrap_or(0), "traded": ledger.traded_by.get(s).copied().unwrap_or(0.0), "tox": st[s].tox, "stopped": st[s].stopped})))
        .collect::<serde_json::Map<String, Value>>()
        .into();
    let summary = json!({
        "run_id": run_id, "environment": env.as_str(), "dry_run": dry_run, "reason": reason, "flat": flat,
        "start_ms": start_ms, "end_ms": now_ms(),
        "pnl": ledger.cash, "ledger_flat": ledger.is_flat(), "pnl_note": "fill ledger: cash of our trades and funding (valid as total P&L when ledger_flat)",
        "fees": ledger.fees, "funding": ledger.funding, "fills": fills.len(), "maker_fills": maker, "traded_notional": traded,
        "pnl_bps_of_traded": if traded > 0.0 { ledger.cash / traded * 1e4 } else { 0.0 },
        "per_symbol": per_symbol, "wallet_equity_start": equity0, "wallet_equity_end": equity1, "config_file": cfg_path.display().to_string(), "config": cfg,
    });
    std::fs::write(out_dir.join("summary.json"), serde_json::to_string_pretty(&summary)?)?;
    std::fs::write(out_dir.join("executions.json"), serde_json::to_string_pretty(fills)?)?;
    journal.write("session_end", summary.clone())?;
    println!("{}", serde_json::to_string_pretty(&json!({"reason": reason, "flat": flat, "pnl": ledger.cash, "fills": fills.len(), "fees": ledger.fees, "traded": traded}))?);
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
    fn multi_config_parses() {
        let c: MmCfg = serde_yaml::from_str(&std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/../../config/mm_multi.yaml")).unwrap()).unwrap();
        assert!(c.symbols.len() >= 40 && c.reduce_lots_max >= 2 && c.shock_pause_sigma > 0.0 && c.symbol_loss_limit_usd > 0.0);
        assert!(!c.symbols.iter().any(|s| s == "SPYUSDT"), "never quote the user's own SPY position");
    }

    #[test]
    fn illiquid_config_parses_with_slow_link_settings() {
        let c: MmCfg = serde_yaml::from_str(&std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/../../config/mm_illiq20.yaml")).unwrap()).unwrap();
        assert_eq!(c.symbols.len(), 20);
        assert!(c.symbols.len() <= c.per_symbol_tickers_max && c.max_quote_age_s > 0.0 && c.flow_trades_limit > 0 && c.bias.is_empty());
        assert!(!c.symbols.iter().any(|s| s == "SPYUSDT"), "never quote the user's own SPY position");
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
        let (b, a) = target_quotes(&c, &rule(), &inp(-0.3, 91.0), &Levels::default());
        assert!(a.is_none());
        let b = b.unwrap();
        assert!(b.reduce_only && b.qty >= 0.1);
        let (b0, _) = target_quotes(&c, &rule(), &inp(0.0, 0.0), &Levels::default());
        assert!(b.price > b0.unwrap().price);
        let x = Inputs { gross_others_usd: 55.0, ..inp(0.0, 0.0) };
        assert_eq!(target_quotes(&c, &rule(), &x, &Levels::default()), (None, None));
    }

    #[test]
    fn reducing_side_scales_with_inventory_adding_side_stays_minimum() {
        let mut c = cfg();
        c.reduce_lots_max = 3;
        // short 0.3 SOL (~$27 of a $30 cap): cover bid scales up to 3 lots; flat: 1 lot both sides
        let (b, _) = target_quotes(&c, &rule(), &inp(-0.3, 91.0), &Levels::default());
        assert!((b.unwrap().qty - 0.3).abs() < 1e-9);
        let (b, _) = target_quotes(&c, &rule(), &inp(-0.1, 91.0), &Levels::default());
        assert!((b.unwrap().qty - 0.1).abs() < 1e-9);
        // one lot is allowed from flat even when it exceeds the per-symbol cap
        c.max_long_usd = 2.0;
        let (b, _) = target_quotes(&c, &rule(), &inp(0.0, 0.0), &Levels::default());
        assert!(b.is_some());
    }

    #[test]
    fn shock_pause_stops_adding_only() {
        let c = cfg();
        let x = Inputs { paused: true, ..inp(0.0, 0.0) };
        assert_eq!(target_quotes(&c, &rule(), &x, &Levels::default()), (None, None));
        let x = Inputs { paused: true, ..inp(-0.1, 91.0) };
        let (b, a) = target_quotes(&c, &rule(), &x, &Levels::default());
        assert!(b.unwrap().reduce_only && a.is_none());
    }

    #[test]
    fn profit_and_exit_clear_passively_at_the_touch() {
        let c = cfg();
        let (b, a) = target_quotes(&c, &rule(), &inp(-0.1, 91.0), &Levels::default());
        assert_eq!(b.unwrap().price, 89.99);
        assert!(a.is_some(), "can still add short within caps");
        let (b, _) = target_quotes(&c, &rule(), &inp(-0.1, 89.5), &Levels::default());
        assert!(b.unwrap().price < 89.99);
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
    fn bias_shifts_both_quotes_and_switches_off_at_target() {
        let c = cfg();
        let (b0, a0) = target_quotes(&c, &rule(), &inp(0.0, 0.0), &Levels::default());
        let (b1, a1) = target_quotes(&c, &rule(), &Inputs { bias_sigma: -0.5, ..inp(0.0, 0.0) }, &Levels::default());
        assert!(a1.unwrap().price < a0.unwrap().price && b1.unwrap().price < b0.unwrap().price, "bearish view lowers both quotes");
        let b = Bias { sigma: -0.4, off_below: Some(3500.0), ..Default::default() };
        assert_eq!(b.active_sigma(4000.0), -0.4);
        assert_eq!(b.active_sigma(3499.0), 0.0);
    }

    #[test]
    fn portfolio_net_cap_pauses_the_crowded_side_and_skews_back() {
        let mut c = cfg();
        c.max_net_usd = 60.0;
        c.portfolio_shift_sigma = 0.6;
        // other symbols already net short $58: another short lot would breach the net cap; the bid still quotes
        let x = Inputs { net_others_usd: -58.0, ..inp(0.0, 0.0) };
        let (b, a) = target_quotes(&c, &rule(), &x, &Levels::default());
        assert!(a.is_none() && b.is_some());
        // resting asks elsewhere count too: $50 of pending sells + $10 net short already block another short lot
        let x2 = Inputs { net_others_usd: -10.0, pend_sell_others_usd: 50.0, ..inp(0.0, 0.0) };
        assert!(target_quotes(&c, &rule(), &x2, &Levels::default()).1.is_none());
        // the net-short portfolio lifts the reservation price: bid closer than with a flat portfolio
        let (b0, _) = target_quotes(&c, &rule(), &inp(0.0, 0.0), &Levels::default());
        assert!(b.unwrap().price > b0.unwrap().price);
    }

    #[test]
    fn mr_stats_detects_bounce() {
        let closes: Vec<f64> = (0..241).map(|i| if i % 2 == 0 { 100.0 } else { 100.1 }).collect();
        let (ac1, vr) = mr_stats(&closes);
        assert!(ac1 < -0.9 && vr < 0.1);
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
    fn adopted_position_is_seeded_at_entry() {
        let mut l = Ledger::default();
        l.seed("XRPUSDT", -10.0, 1.20);
        let mids = HashMap::from([("XRPUSDT".to_string(), 1.20)]);
        assert!(l.pnl(&mids).abs() < 1e-12, "adopted at entry: zero P&L at the entry price");
        // cover the short at 1.10: +1.0 realised minus the fee, and the ledger is flat again (no phantom long)
        l.add(&json!({"execId": "a", "symbol": "XRPUSDT", "side": "Buy", "execValue": "11.0", "execQty": "10", "execFee": "0.002", "execTime": "1", "execType": "Trade"}));
        assert!(l.is_flat());
        let mids = HashMap::from([("XRPUSDT".to_string(), 1.50)]);
        assert!((l.pnl(&mids) - 0.998).abs() < 1e-9 && (l.pnl_of("XRPUSDT", 1.5) - 0.998).abs() < 1e-9);
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
        assert!((l.pnl_of("SOLUSDT", 89.0) - l.pnl(&mids)).abs() < 1e-12);
        l.add(&buy);
        l.add(&json!({"execId": "3", "symbol": "SOLUSDT", "side": "Sell", "execValue": "0", "execFee": "-0.001", "execType": "Funding", "execTime": "3"}));
        assert_eq!(l.fills.len(), 2);
        assert_eq!(l.fills_by["SOLUSDT"], 2);
        assert!(l.is_flat());
        assert!((l.cash - (18.0 - 17.9 - 0.0036 - 0.00358 + 0.001)).abs() < 1e-9);
        assert!((l.funding - 0.001).abs() < 1e-12);
    }

    #[test]
    fn positive_inventory_target_pulls_the_bid_closer_and_keeps_the_core_long() {
        let mut c = cfg();
        c.max_long_usd = 30.0;
        c.max_short_usd = 7.0;
        let rule = rule();
        let flat = inp(0.0, 0.0);
        let (b0, _) = target_quotes(&c, &rule, &flat, &Levels::default());
        let (b1, a1) = target_quotes(&c, &rule, &Inputs { target_usd: 13.0, ..flat.clone() }, &Levels::default());
        assert!(b1.unwrap().price > b0.unwrap().price, "below target: bid closer");
        // the further below the target, the closer the bid (inversely proportional to inventory)
        let mid = 0.5 * (flat.bid1 + flat.ask1);
        let small_long = Inputs { inv_qty: 6.0 / mid, avg_entry: mid, target_usd: 13.0, ..flat.clone() };
        let (b2, a2) = target_quotes(&c, &rule, &small_long, &Levels::default());
        assert!(b2.unwrap().price < b1.unwrap().price && a2.unwrap().price < a1.unwrap().price, "closer to target: both quotes move down");
        // in profit but below the target: no take-profit at the touch
        let up = Inputs { bid1: mid * 1.05, ask1: mid * 1.05 + 0.01, ..small_long.clone() };
        let (_, a3) = target_quotes(&c, &rule, &up, &Levels::default());
        assert!(a3.unwrap().price > up.ask1, "core long is not sold at the touch");
        // above the target and in profit: sell the excess at the touch
        let big = Inputs { inv_qty: 25.0 / mid, ..up.clone() };
        let (_, a4) = target_quotes(&c, &rule, &big, &Levels::default());
        assert!((a4.unwrap().price - up.ask1).abs() < 1e-9 && a4.unwrap().reduce_only);
    }

    #[test]
    fn break_even_floor_stops_selling_below_entry() {
        let mut c = cfg();
        c.reduce_min_edge_bps = 1.0;
        c.max_long_usd = 60.0;
        // long bought at 91, market now 89.99/90.00: without the floor the reducing ask sits near the market
        let x = Inputs { inv_qty: 0.3, avg_entry: 91.0, ..inp(0.0, 0.0) };
        let (_, a) = target_quotes(&c, &rule(), &x, &Levels::default());
        assert!(a.unwrap().reduce_only && a.unwrap().price >= 91.0 * 1.0001 - 1e-9, "{:?}", a);
        c.reduce_min_edge_bps = 0.0;
        let (_, a0) = target_quotes(&c, &rule(), &x, &Levels::default());
        assert!(a0.unwrap().price < 91.0, "without the floor it would sell below entry");
        // the wind-down still clears at the touch
        c.reduce_min_edge_bps = 1.0;
        let (_, a2) = target_quotes(&c, &rule(), &Inputs { exit_only: true, ..x }, &Levels::default());
        assert!((a2.unwrap().price - 90.0).abs() < 1e-9);
    }

    #[test]
    fn trend_guard_switches_off_against_the_view() {
        // 1h sigma 2%, 10-minute window: sd = 0.02 * sqrt(600/3600) = 0.82%
        assert!(trend_against_view(-0.015, 600.0, 0.02, 1.5, 0.25), "falling 1.5% vs long view");
        assert!(!trend_against_view(-0.005, 600.0, 0.02, 1.5, 0.25), "small dip: keep the view");
        assert!(!trend_against_view(0.02, 600.0, 0.02, 1.5, 0.25), "rising: no guard for a long view");
        assert!(trend_against_view(0.015, 600.0, 0.02, 1.5, -0.4), "rising vs short view");
        assert!(!trend_against_view(-0.05, 600.0, 0.02, 0.0, 0.25), "off when k = 0");
    }

    #[test]
    fn micro_sigma_from_trades() {
        // a price that alternates +/-10 bp every 15 s: per-interval sigma ~ 10 bp * sqrt(pi/2)
        let now = 1_000_000_000i64;
        let mut tr = vec![];
        let mut px = 100.0;
        for k in (0..41).rev() {
            px *= if k % 2 == 0 { 1.001 } else { 1.0 / 1.001 };
            tr.push((now - k * 15_000 - 100, px));
            tr.push((now - k * 15_000 - 7_000, px * 0.9)); // an earlier print in the same bucket is ignored
        }
        let s = micro_sigma(&tr, now, 15, 40).unwrap();
        assert!((s - 0.001 * (std::f64::consts::PI / 2.0).sqrt()).abs() < 2e-5, "{s}");
        assert!(micro_sigma(&tr[..6], now, 15, 40).is_none(), "too few buckets");
        assert!(micro_sigma(&tr, now, 0, 40).is_none());
    }

    #[test]
    fn level_tilt_points_to_the_nearer_level() {
        let lv = Levels { res: vec![110.0], sup: vec![99.0] };
        let t = level_tilt(100.0, 0.01, &lv, 20.0);
        assert!(t > 0.7, "near support -> + ({t})");
        let t2 = level_tilt(109.0, 0.01, &lv, 20.0);
        assert!(t2 < -0.7, "near resistance -> - ({t2})");
        assert_eq!(level_tilt(100.0, 0.01, &Levels::default(), 3.0), 0.0, "no levels -> 0");
        // capped reach: far levels on both sides -> 0
        assert!(level_tilt(100.0, 0.001, &lv, 3.0).abs() < 1e-12);
    }

    #[test]
    fn quantity_factor_scales_the_adding_side_within_caps() {
        let mut c = cfg();
        c.max_long_usd = 60.0;
        c.max_short_usd = 7.0;
        c.max_gross_usd = 100.0;
        c.add_lots_max = 3;
        c.add_inventory_weight = 1.0;
        c.add_level_weight = 0.5;
        let flat = inp(0.0, 0.0);
        let (b, _) = target_quotes(&c, &rule(), &Inputs { target_usd: 40.0, ..flat.clone() }, &Levels::default());
        assert!((b.unwrap().qty - 0.3).abs() < 1e-9, "far below target: 3 lots ({:?})", b);
        let (b2, _) = target_quotes(&c, &rule(), &flat, &Levels::default());
        assert!((b2.unwrap().qty - 0.1).abs() < 1e-9, "no target, no tilt: one lot");
        let (b3, _) = target_quotes(&c, &rule(), &Inputs { level_tilt: 1.0, ..flat.clone() }, &Levels::default());
        assert!((b3.unwrap().qty - 0.2).abs() < 1e-9, "at support: more lots ({:?})", b3);
        // caps still bind: long cap 15 USD at 90 -> at most 1 lot (9 USD) from flat
        c.max_long_usd = 15.0;
        let (b4, _) = target_quotes(&c, &rule(), &Inputs { target_usd: 14.0, ..flat }, &Levels::default());
        assert!((b4.unwrap().qty - 0.1).abs() < 1e-9, "cap binds ({:?})", b4);
    }

    #[test]
    fn fast_sigma_window_is_live_on_a_burst() {
        // calm 4 h, then a 30-minute burst of 40 bp 1m moves: the 30-minute sigma must see it (it used to return the floor)
        let mut c = vec![100.0];
        for i in 0..240 {
            c.push(c[c.len() - 1] * if i % 2 == 0 { 1.0002 } else { 0.9998 });
        }
        for i in 0..30 {
            c.push(c[c.len() - 1] * if i % 2 == 0 { 1.004 } else { 0.996 });
        }
        let fast = sigma_last(&c, 30, 20.0);
        let slow = sigma_1h(&c[c.len() - 241..], 20.0);
        assert!(fast > 0.02 && fast > 2.0 * slow, "fast {fast} slow {slow}");
        assert_eq!(sigma_last(&c[..30], 30, 20.0), 0.0, "not enough closes");
    }

    #[test]
    fn shared_decision_steps_match_the_live_rules() {
        let mut c = cfg();
        // quote sigma: 1h sigma unless micro quoting is on and a micro estimate exists (then scaled and floored)
        assert_eq!(quote_sigma(&c, 0.0005, 0.004), 0.004);
        (c.micro_interval_s, c.micro_horizon_s, c.micro_floor_bps) = (15, 60.0, 3.0);
        assert!((quote_sigma(&c, 0.0005, 0.004) - 0.001).abs() < 1e-12);
        assert_eq!(quote_sigma(&c, 0.0001, 0.004), 0.0003);
        assert_eq!(quote_sigma(&c, 0.0, 0.004), 0.004);
        // replace rule: kind change or > 1 sigma drift at once; > requote_sigma only after min_rest_s
        let r = rule();
        let q = Quote { price: 90.0, qty: 0.1, reduce_only: false };
        let rest = |px: f64, ro: bool, t: i64| Resting { price: px, leaves_qty: 0.1, reduce_only: ro, updated_ms: t };
        let s = 0.004;
        assert!(!needs_replace(&c, &r, Some(q), Some(&rest(90.0, false, 0)), 90.0, s, 1_000));
        assert!(needs_replace(&c, &r, Some(q), Some(&rest(90.0, true, 0)), 90.0, s, 1_000));
        assert!(needs_replace(&c, &r, Some(q), Some(&rest(90.0 * (1.0 + 1.1 * s), false, 0)), 90.0, s, 1_000));
        let mid_drift = rest(90.0 * (1.0 + 0.5 * s), false, 0);
        assert!(!needs_replace(&c, &r, Some(q), Some(&mid_drift), 90.0, s, (c.min_rest_s - 1) * 1000));
        assert!(needs_replace(&c, &r, Some(q), Some(&mid_drift), 90.0, s, c.min_rest_s * 1000));
        assert!(needs_replace(&c, &r, None, Some(&mid_drift), 90.0, s, 0) && !needs_replace(&c, &r, Some(q), None, 90.0, s, 0));
        // symbol_inputs carries state and position through; the guard switches bias and target off
        let e = SymState { sigma: 0.004, flow: 0.3, tox: -0.2, ..Default::default() };
        let x = symbol_inputs(&c, "SOLUSDT", &e, 89.99, 90.0, (0.5, 88.0), Others { gross_usd: 10.0, ..Default::default() }, false, true, false);
        assert_eq!((x.inv_qty, x.avg_entry, x.flow, x.tox, x.gross_others_usd, x.paused), (0.5, 88.0, 0.3, -0.2, 10.0, true));
        // shock: a 3-sigma move per tick pauses; the first observation never does
        c.shock_pause_sigma = 3.0;
        let mut e2 = SymState { sigma: 0.01, ..Default::default() };
        assert!(shock_jump(&c, &e2, 100.0, 0).is_none());
        (e2.last_mid, e2.last_mid_ms) = (100.0, 0);
        let lim = c.shock_pause_sigma * 0.01 * (c.refresh_s as f64 / 3600.0).sqrt();
        assert!(shock_jump(&c, &e2, 100.0 * (1.0 + 1.2 * lim), 1).is_some() && shock_jump(&c, &e2, 100.0 * (1.0 + 0.5 * lim), 1).is_none());
        assert_eq!(trend_window(&[(2000, 101.0), (1000, 100.0)]).map(|x| x.1), Some(1.0));
    }

    #[test]
    fn sigma_floor_and_scale() {
        assert_eq!(sigma_1h(&[1.0; 10], 15.0), 0.0015);
        let closes: Vec<f64> = (0..241).map(|i| if i % 2 == 0 { 100.0 } else { 100.1 }).collect();
        let s = sigma_1h(&closes, 1.0);
        assert!((s - 0.001 * (std::f64::consts::PI / 2.0).sqrt() * 60f64.sqrt()).abs() < 1e-4);
    }

    // ------------------------------------------------------------ weekend alt quoting rules (H-MMW1-4)

    #[test]
    fn ladder_levels_keep_zones_inside_the_sigma_band() {
        // mid 100, sigma 1%: band 1.5-3.5% from the mid
        let zones = [(97.0, 98.0), (95.0, 96.0), (99.5, 99.8), (101.6, 102.5), (103.0, 104.0), (90.0, 91.0)];
        let (b, a) = ladder_levels(&zones, 100.0, 0.01, (1.5, 3.5));
        // support tops 98.0 (2.0%) and 96.0 (4.1%: out), 99.8 (0.2%: too close), 91 (out)
        assert_eq!(b, vec![98.0]);
        // resistance bottoms 101.6 (1.59%) and 103.0 (2.96%), nearest first
        assert_eq!(a, vec![101.6, 103.0]);
        // a zone straddling the mid quotes nothing; no sigma, no levels
        assert_eq!(ladder_levels(&[(99.0, 101.0)], 100.0, 0.01, (0.0, 9.0)), (vec![], vec![]));
        assert_eq!(ladder_levels(&zones, 100.0, 0.0, (1.5, 3.5)), (vec![], vec![]));
    }

    #[test]
    fn ladder_stop_and_take_profit() {
        let s = level_stop(100.0, true, 1.5, 0.01);
        assert!((s - 100.0 * (-0.015f64).exp()).abs() < 1e-12 && s < 100.0);
        assert!(level_stop(100.0, false, 1.5, 0.01) > 100.0);
        // long filled at 100.00 with the mid at 100.025 (tick 0.01): TP at 100.03 (mid rounded up)
        assert!((ladder_tp(100.0, 100.025, true, 0.01) - 100.03).abs() < 1e-9);
        // a fill through the price (mid already below the entry): never below entry + 1 tick
        assert!((ladder_tp(100.0, 99.90, true, 0.01) - 100.01).abs() < 1e-9);
        // short: mid rounded down, at most entry - 1 tick
        assert!((ladder_tp(100.0, 99.975, false, 0.01) - 99.97).abs() < 1e-9);
        assert!((ladder_tp(100.0, 100.2, false, 0.01) - 99.99).abs() < 1e-9);
    }

    #[test]
    fn btc_move_z_scales_to_the_horizon() {
        let s = 0.006; // 60 bp per hour -> 1 s sd = 1 bp, 5 s sd = 2.236 bp
        assert!((btc_move_z(0.0001, 0.0, s) - 1.0).abs() < 1e-9);
        assert!((btc_move_z(0.0, -0.0002236068, s) - 1.0).abs() < 1e-6);
        assert!((btc_move_z(0.0003, 0.0001, s) - 3.0).abs() < 1e-9);
        assert_eq!(btc_move_z(0.01, 0.01, 0.0), 0.0);
    }

    #[test]
    fn btc_tape_needs_sigma_and_history_then_flags_jumps() {
        let mut t = BtcTape::new(1.0);
        // no warm-up: z = 0
        assert_eq!(t.on_print(0, 100.0), 0.0);
        let mut t = BtcTape::new(1.0);
        // 240 warm-up minutes alternating +-10 bp: sigma ~ 10 bp x sqrt(pi/2) x sqrt(60)
        for i in 0..240 {
            t.seed_close(-(240 - i) * 60_000, if i % 2 == 0 { 100.0 } else { 100.1 });
        }
        assert_eq!(t.on_print(0, 100.0), 0.0, "no 5 s history yet");
        assert!(t.sigma > 0.0);
        assert!(t.on_print(3_000, 100.0) == 0.0, "still no print 5 s back");
        assert!(t.on_print(6_000, 100.0).abs() < 1e-12, "flat: z 0");
        // +30 bp in 1 s: z = 0.003 / (sigma / 60) >> 4
        let z = t.on_print(7_000, 100.3);
        assert!(z > 4.0, "{z} sigma {}", t.sigma);
        assert!((z - 0.003f64.max((100.3f64 / 100.0).ln()) / (t.sigma / 60.0)).abs() < 0.2);
    }

    #[test]
    fn imbalance_and_side_pull() {
        let bids = [(9.9, 10.0), (9.8, 10.0), (9.7, 5.0), (9.6, 5.0), (9.5, 5.0), (9.4, 1000.0)];
        let asks = [(10.0, 5.0), (10.1, 5.0), (10.2, 5.0), (10.3, 5.0), (10.4, 5.0)];
        assert!((top_imbalance(&bids, &asks, 5) - (35.0 - 25.0) / 60.0).abs() < 1e-12, "only the top 5 levels count");
        assert_eq!(top_imbalance(&[], &[], 5), 0.0);
        assert_eq!(threatened_side(0.6, 0.5), Some(true));
        assert_eq!(threatened_side(-0.6, 0.5), Some(false));
        assert_eq!(threatened_side(0.4, 0.5), None);
        assert_eq!(threatened_side(0.9, 0.0), None, "threshold 0 = off");
        let b = Some(Quote { price: 99.0, qty: 1.0, reduce_only: false });
        let a = Some(Quote { price: 101.0, qty: 1.0, reduce_only: true });
        assert_eq!(side_pull(b, a, Some(true), 0.0, 0.01, 0.01), (b, None), "buying pressure withdraws the ask");
        assert_eq!(side_pull(b, a, Some(false), 0.0, 0.01, 0.01), (None, a));
        assert_eq!(side_pull(b, a, None, 0.0, 0.01, 0.01), (b, a));
        let (b2, a2) = side_pull(b, a, Some(true), 1.0, 0.01, 0.01);
        assert_eq!(b2, b);
        let a2 = a2.unwrap();
        assert!((a2.price - 102.01).abs() < 1e-9 && a2.reduce_only, "widened by 1 sigma, rounded up: {}", a2.price);
        let (b3, _) = side_pull(b, a, Some(false), 1.0, 0.01, 0.01);
        assert!((b3.unwrap().price - 98.01).abs() < 1e-9);
    }

    #[test]
    fn short_queue_and_leave_rules() {
        assert!(short_queue(40.0, 100.0, 0.5));
        assert!(!short_queue(50.0, 100.0, 0.5));
        assert!(!short_queue(0.0, 0.0, 1.0), "no median yet: never join");
        assert!(!short_queue(0.0, 100.0, 0.0));
        let h = [(0, 10.0), (1_000, 100.0), (2_500, 49.0)];
        assert!(behind_thinned(&h, 2_500, 2_000, 0.5), "100 -> 49 within 2 s");
        assert!(!behind_thinned(&[(0, 100.0), (1_000, 60.0), (2_500, 49.0)], 2_500, 2_000, 0.5), "the 100 is older than 2 s");
        assert!(!behind_thinned(&[(0, 0.0)], 0, 2_000, 0.5));
        assert!(!behind_thinned(&[], 0, 2_000, 0.5));
        assert_eq!(median(&[3.0, 1.0, 2.0]), 2.0);
        assert_eq!(median(&[4.0, 1.0, 2.0, 3.0]), 2.5);
        assert!(median(&[]).is_nan());
    }

    #[test]
    fn liquidation_proxy_needs_count_levels_and_notional() {
        // 5 sells within 30 s over 3 levels with >= 1000 USD in total
        let mut p = LiqProxy::new(30_000, 5, 3, 1_000.0);
        let prints = [(0, 100, 300.0), (1_000, 100, 300.0), (2_000, 99, 100.0), (3_000, 99, 100.0)];
        for (t, k, n) in prints {
            assert!(!p.on_print(t, k, n, false));
        }
        // a buy print does not count for the sell side
        assert!(!p.on_print(3_500, 101, 5_000.0, true));
        assert!(p.on_print(4_000, 98, 300.0, false), "5th sell, 3 levels, 1100 USD");
        assert!(!p.on_print(4_500, 97, 300.0, false), "a trigger empties the window");
        // too few levels: 5 prints at 2 prices
        let mut q = LiqProxy::new(30_000, 5, 3, 10.0);
        for i in 0..6 {
            assert!(!q.on_print(i * 100, 100 + i % 2, 50.0, true));
        }
        // the window slides: prints older than 30 s drop out
        let mut r = LiqProxy::new(30_000, 5, 3, 10.0);
        for (i, t) in [0, 1_000, 2_000, 3_000].into_iter().enumerate() {
            assert!(!r.on_print(t, 100 + i as i64, 10.0, false));
        }
        assert!(!r.on_print(40_000, 110, 10.0, false), "the first four are gone");
        // notional below the floor; a zero floor switches the proxy off
        let mut s = LiqProxy::new(30_000, 5, 3, 1e9);
        let mut z = LiqProxy::new(30_000, 1, 1, 0.0);
        for i in 0..6 {
            assert!(!s.on_print(i, 100 + i, 10.0, false));
            assert!(!z.on_print(i, 100 + i, 10.0, false));
        }
    }
}
