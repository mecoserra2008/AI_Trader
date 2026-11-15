//! `atx scalp`: deterministic order-flow scalper for one illiquid perp (KAVAUSDT), pre-registered as H-SCALP1
//! (research/hypotheses.yaml). Config: `config/scalp_kava.yaml`.
//!
//!   atx scalp --mode shadow [--config config/scalp_kava.yaml] [--minutes N]
//!   atx scalp --mode live      refused in this build (see `live_preflight`)
//!
//! Inputs: Bybit public WS `orderbook.<depth>`, `publicTrade` and `tickers` for the symbol (mainnet market data). The
//! book is maintained by `at_flow::book::Book`. Every book update adds its L1 Cont-Kukanov-Stoikov OFI to per-second
//! buckets; every trade adds its signed volume. Once per second the engine computes the features (OFI and taker delta
//! over 10/30/60 s, z(OFI_w) against its own last 1800 one-second samples, book imbalance within N ticks, microprice
//! vs mid, sweeps, absorption, spread in ticks, queue at the touch) and applies the entry rule:
//!
//!   LONG  if spread <= 2 ticks AND z(OFI_30) >= k AND delta_30 > 0 AND microprice > mid AND no sell sweep in 10 s
//!   SHORT if spread <= 2 ticks AND z(OFI_30) <= -k AND delta_30 < 0 AND microprice < mid AND no buy sweep in 10 s
//!
//! Entry: PostOnly limit joining the touch on our side, cancelled after `entry_timeout_s` or when z(OFI_30) crosses 0
//! against the trade / the opposite signal fires. Stop (frozen at the signal): 1 tick beyond the `swing_lookback_s` trade
//! extreme, at least `min_stop_ticks`; > `max_stop_ticks` = skip. Take profit: reduce-only PostOnly at exactly
//! `reward_r` x the stop distance (integer ticks). Time stop: passive exit at the far touch, then market. Funding: shorts
//! flat by T-5 min (and no short entry within `short_entry_block_min`); longs too unless >= `long_hold_min_r` in profit.
//!
//! Shadow fill model (conservative): a passive order fills only when a trade prints THROUGH its price, or when trades AT
//! its price exceed the displayed queue ahead at placement plus our qty (the queue ahead is depleted by trades only).
//! LastPrice stops fill at min(stop - slip ticks, the triggering print) (long; short symmetric), taker fee. TPs fill only on
//! a trade-through, maker fee. Journal: `data/journal/scalp/<date>.jsonl` (every signal, order, cancel, fill, exit, status).
//! A STATUS line goes to stdout every `status_every_s`.
//!
//! All price logic runs on integer ticks; prices are converted once, at the boundary.

use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::io::Write as _;
use std::path::{Path, PathBuf};
use std::time::Duration;

use anyhow::{Context, bail, ensure};
use at_bybit::market::MarketData;
use at_bybit::ws::{self, WsMessage};
use at_core::time::{format_date, now_ms};
use at_core::types::Side;
use at_flow::book::{Book, L1, ofi};
use at_risk::live::LiveSmallCfg;
use chrono::{SecondsFormat, Utc};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use crate::cmd::Ctx;

/// Engine version: a change to the mechanics or the fill model starts a new track record.
pub const SCALP_VERSION: &str = "scalp-v1";

// ---------------------------------------------------------------- config

/// Scalper sub-limits. They can only be TIGHTER than `live_small` and than these compiled ceilings.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ScalpLimits {
    pub notional_cap_usd: f64,
    pub day_loss_stop_usd: f64,
    pub max_entries_per_run: u32,
    pub max_consecutive_losses: u32,
}

/// The user's scalper numbers (2026-11-15), compiled in: the yaml may be stricter, never looser.
pub const SCALP_CEILINGS: ScalpLimits = ScalpLimits { notional_cap_usd: 20.0, day_loss_stop_usd: 3.0, max_entries_per_run: 20, max_consecutive_losses: 4 };

impl ScalpLimits {
    pub fn validate(&self) -> anyhow::Result<()> {
        let k = SCALP_CEILINGS;
        let pos = |x: f64| x.is_finite() && x > 0.0;
        ensure!(pos(self.notional_cap_usd) && self.notional_cap_usd <= k.notional_cap_usd, "limits.notional_cap_usd must be in (0, {}]", k.notional_cap_usd);
        ensure!(pos(self.day_loss_stop_usd) && self.day_loss_stop_usd <= k.day_loss_stop_usd, "limits.day_loss_stop_usd must be in (0, {}]", k.day_loss_stop_usd);
        ensure!((1..=k.max_entries_per_run).contains(&self.max_entries_per_run), "limits.max_entries_per_run must be in [1, {}]", k.max_entries_per_run);
        ensure!((1..=k.max_consecutive_losses).contains(&self.max_consecutive_losses), "limits.max_consecutive_losses must be in [1, {}]", k.max_consecutive_losses);
        Ok(())
    }

    /// Tighter-only against the live desk's profile: the scalper shares its daily counters, so it may never be looser.
    pub fn validate_against_live(&self, live: &LiveSmallCfg) -> anyhow::Result<()> {
        self.validate()?;
        let min_tier = live.tiers.iter().map(|t| t.notional_cap_usd).fold(f64::INFINITY, f64::min);
        ensure!(self.notional_cap_usd <= min_tier, "notional_cap_usd {} exceeds the smallest live tier cap {min_tier}", self.notional_cap_usd);
        ensure!(self.notional_cap_usd <= live.max_gross_notional_usd, "notional_cap_usd exceeds live max_gross_notional_usd");
        ensure!(self.day_loss_stop_usd <= live.daily_loss_stop_usd, "day_loss_stop_usd {} exceeds live daily_loss_stop_usd {}", self.day_loss_stop_usd, live.daily_loss_stop_usd);
        ensure!(self.max_entries_per_run <= live.max_entries_per_day, "max_entries_per_run {} exceeds live max_entries_per_day {}", self.max_entries_per_run, live.max_entries_per_day);
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ScalpCfg {
    pub symbol: String,
    pub hypothesis: String,
    pub depth: u32,
    pub eval_interval_ms: u64,
    pub ofi_windows_s: Vec<i64>,
    pub z_window_s: usize,
    pub z_min_obs: usize,
    pub imbalance_ticks: i64,
    pub sweep_min_levels: usize,
    pub absorption_window_s: i64,
    pub absorption_mult: f64,
    pub signal_window_s: i64,
    pub ofi_z_entry: f64,
    pub max_spread_ticks: i64,
    pub sweep_lookback_s: i64,
    pub entry_timeout_s: i64,
    pub reward_r: i64,
    pub min_stop_ticks: i64,
    pub swing_lookback_s: i64,
    pub max_stop_ticks: i64,
    pub stop_trigger_by: String,
    pub stop_slip_ticks: i64,
    pub time_stop_min: f64,
    pub passive_exit_s: i64,
    pub cooldown_s: i64,
    pub funding_flat_min: f64,
    pub short_entry_block_min: f64,
    pub long_hold_min_r: f64,
    pub maker_bps: f64,
    pub taker_bps: f64,
    pub limits: ScalpLimits,
    pub status_every_s: u64,
    /// Live-only settings. Not part of the strategy: excluded from `hash()`, so the H-SCALP1 track record (whose
    /// shadow trades carry the strategy hash) is the same with or without this section.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub live: Option<ScalpLiveCfg>,
}

/// `live:` section of the scalper config (used only by `--mode live`).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ScalpLiveCfg {
    /// Where the EV gate's p_tp comes from. Only "h_scalp1_shadow": the 95 % CI LOWER bound of the TP-hit rate in the
    /// frozen H-SCALP1 evaluation (`verdict_file`, written by `scripts/scalp_eval.py --write`).
    pub p_tp_source: String,
    pub verdict_file: String,
    /// Horizon of the sigma handed to the risk engine's placement check: sd of 1 s mid log returns x sqrt(horizon).
    pub sigma_horizon_s: i64,
    /// REST fallback poll of the venue state (the private WS is primary).
    pub poll_s: u64,
    /// Scalper tag inside the bot's link prefix: orderLinkId `<link_prefix>-<tag>-...` (lsb-sc-...).
    pub link_tag: String,
    /// After a refused entry decision, no new decision for this long.
    pub refusal_backoff_s: i64,
    /// A position without its exchange stop longer than this (after a re-attach attempt) halts the run.
    pub stop_grace_s: i64,
}

impl ScalpLiveCfg {
    pub fn validate(&self) -> anyhow::Result<()> {
        ensure!(self.p_tp_source == "h_scalp1_shadow", "live.p_tp_source must be h_scalp1_shadow");
        ensure!(!self.verdict_file.is_empty(), "live.verdict_file must be set");
        ensure!((10..=3600).contains(&self.sigma_horizon_s), "live.sigma_horizon_s in [10, 3600]");
        ensure!((1..=5).contains(&self.poll_s), "live.poll_s in [1, 5] (REST fallback every <= 5 s)");
        ensure!(!self.link_tag.is_empty() && self.link_tag.len() <= 4 && self.link_tag.chars().all(|c| c.is_ascii_alphanumeric()), "live.link_tag: 1-4 ASCII letters/digits");
        ensure!(self.refusal_backoff_s >= 1 && self.stop_grace_s >= 1, "live.refusal_backoff_s and stop_grace_s >= 1");
        Ok(())
    }
}

impl ScalpCfg {
    pub fn load(path: &Path) -> anyhow::Result<Self> {
        let text = std::fs::read_to_string(path).with_context(|| path.display().to_string())?;
        let c: Self = serde_yaml::from_str(&text).with_context(|| path.display().to_string())?;
        c.validate().with_context(|| path.display().to_string())?;
        Ok(c)
    }

    pub fn validate(&self) -> anyhow::Result<()> {
        self.limits.validate()?;
        ensure!(!self.symbol.is_empty() && self.symbol != "SPYUSDT", "symbol must be set and is never SPYUSDT");
        ensure!(!self.ofi_windows_s.is_empty() && self.ofi_windows_s.iter().all(|w| *w > 0), "ofi_windows_s must be positive");
        ensure!(self.ofi_windows_s.contains(&self.signal_window_s), "signal_window_s must be one of ofi_windows_s");
        ensure!(self.z_window_s >= self.z_min_obs && self.z_min_obs >= 30, "z_window_s >= z_min_obs >= 30");
        ensure!(matches!(self.reward_r, 2 | 3), "reward_r must be 2 or 3");
        ensure!(self.min_stop_ticks >= 1 && self.max_stop_ticks >= self.min_stop_ticks, "1 <= min_stop_ticks <= max_stop_ticks");
        ensure!(matches!(self.stop_trigger_by.as_str(), "LastPrice" | "MarkPrice"), "stop_trigger_by: LastPrice | MarkPrice");
        ensure!(self.stop_slip_ticks >= 0 && self.max_spread_ticks >= 1, "stop_slip_ticks >= 0, max_spread_ticks >= 1");
        ensure!(self.entry_timeout_s > 0 && self.passive_exit_s > 0 && self.cooldown_s >= 0 && self.time_stop_min > 0.0, "timeouts must be positive");
        ensure!(self.funding_flat_min > 0.0 && self.short_entry_block_min >= self.funding_flat_min + self.time_stop_min, "short_entry_block_min >= funding_flat_min + time_stop_min");
        ensure!(self.long_hold_min_r >= 1.0, "long_hold_min_r must be >= 1 (user rule)");
        ensure!(self.maker_bps >= 0.0 && self.taker_bps >= self.maker_bps, "fees: 0 <= maker <= taker");
        ensure!(self.ofi_z_entry > 0.0 && self.sweep_min_levels >= 2 && self.imbalance_ticks >= 1, "ofi_z_entry > 0, sweep_min_levels >= 2, imbalance_ticks >= 1");
        if let Some(l) = &self.live {
            l.validate()?;
        }
        Ok(())
    }

    /// FNV-1a 64 of the config JSON: logged with every journal line; a change starts a new record.
    pub fn hash(&self) -> String {
        let mut strategy = self.clone();
        strategy.live = None;
        let s = serde_json::to_string(&strategy).unwrap_or_default();
        let h = s.bytes().fold(0xcbf2_9ce4_8422_2325_u64, |h, b| (h ^ b as u64).wrapping_mul(0x0100_0000_01b3));
        format!("{h:016x}")
    }

    fn max_window_s(&self) -> i64 {
        self.ofi_windows_s.iter().copied().max().unwrap_or(60)
    }

    pub fn signal_idx(&self) -> usize {
        self.ofi_windows_s.iter().position(|w| *w == self.signal_window_s).unwrap_or(0)
    }
}

// ---------------------------------------------------------------- feature maths (pure)

/// Price to integer ticks.
pub fn to_ticks(px: f64, tick: f64) -> i64 {
    (px / tick).round() as i64
}

/// Microprice = (bid * ask_qty + ask * bid_qty) / (bid_qty + ask_qty): leans toward the thinner side's price.
pub fn microprice(l1: &L1) -> f64 {
    let t = l1.bid_qty + l1.ask_qty;
    if t <= 0.0 { 0.5 * (l1.bid + l1.ask) } else { (l1.bid * l1.ask_qty + l1.ask * l1.bid_qty) / t }
}

/// (bid - ask) / (bid + ask) of the displayed sizes within `n` ticks of each touch (levels as (price ticks, size), best first).
pub fn imbalance_within(bids: &[(i64, f64)], asks: &[(i64, f64)], n: i64) -> f64 {
    let (Some(bb), Some(ba)) = (bids.first().map(|x| x.0), asks.first().map(|x| x.0)) else { return 0.0 };
    let b: f64 = bids.iter().take_while(|(p, _)| bb - p <= n).map(|x| x.1).sum();
    let a: f64 = asks.iter().take_while(|(p, _)| p - ba <= n).map(|x| x.1).sum();
    if a + b > 0.0 { (b - a) / (a + b) } else { 0.0 }
}

/// One public trade: local receive ms, exchange ms, price in ticks, aggressor buy?, size.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Tr {
    pub ms: i64,
    pub ex_ms: i64,
    pub px: i64,
    pub buy: bool,
    pub vol: f64,
}

/// Sweeps in a batch of trades: same aggressor side and exchange timestamp, at >= `min_levels` distinct prices.
/// Returns (exchange ms, aggressor buy?, distinct levels).
pub fn sweeps(trades: &[Tr], min_levels: usize) -> Vec<(i64, bool, usize)> {
    let mut g: BTreeMap<(i64, bool), BTreeSet<i64>> = BTreeMap::new();
    for t in trades {
        g.entry((t.ex_ms, t.buy)).or_default().insert(t.px);
    }
    g.into_iter().filter(|(_, s)| s.len() >= min_levels).map(|((ms, buy), s)| (ms, buy, s.len())).collect()
}

/// Per-second sums over a sliding horizon (OFI, signed taker volume).
#[derive(Clone, Debug, Default)]
pub struct SecSums {
    q: VecDeque<(i64, f64)>,
    horizon_s: i64,
}

impl SecSums {
    pub fn new(horizon_s: i64) -> Self {
        Self { q: VecDeque::new(), horizon_s }
    }

    pub fn add(&mut self, sec: i64, x: f64) {
        match self.q.back_mut() {
            Some(b) if b.0 == sec => b.1 += x,
            _ => self.q.push_back((sec, x)),
        }
        while self.q.front().is_some_and(|f| f.0 <= sec - self.horizon_s) {
            self.q.pop_front();
        }
    }

    /// Sum over the seconds in (now_sec - w, now_sec].
    pub fn sum(&self, now_sec: i64, w: i64) -> f64 {
        self.q.iter().rev().filter(|(s, _)| *s <= now_sec).take_while(|(s, _)| *s > now_sec - w).map(|x| x.1).sum()
    }
}

/// Rolling mean / sd of the last `n` samples; z of a new value against the PAST window (then pushed).
#[derive(Clone, Debug)]
pub struct RollingZ {
    q: VecDeque<f64>,
    n: usize,
    min_obs: usize,
    sum: f64,
    sumsq: f64,
    pushes: u64,
}

impl RollingZ {
    pub fn new(n: usize, min_obs: usize) -> Self {
        Self { q: VecDeque::with_capacity(n + 1), n, min_obs, sum: 0.0, sumsq: 0.0, pushes: 0 }
    }

    pub fn len(&self) -> usize {
        self.q.len()
    }

    pub fn z(&self, x: f64) -> Option<f64> {
        let m = self.q.len();
        if m < self.min_obs.max(2) {
            return None;
        }
        let mean = self.sum / m as f64;
        let var = ((self.sumsq - m as f64 * mean * mean) / (m as f64 - 1.0)).max(0.0);
        let sd = var.sqrt();
        (sd > 0.0 && sd.is_finite()).then(|| (x - mean) / sd)
    }

    /// Sample sd of the window (None before min_obs).
    pub fn sd(&self) -> Option<f64> {
        let m = self.q.len();
        if m < self.min_obs.max(2) {
            return None;
        }
        let mean = self.sum / m as f64;
        let sd = ((self.sumsq - m as f64 * mean * mean) / (m as f64 - 1.0)).max(0.0).sqrt();
        (sd > 0.0 && sd.is_finite()).then_some(sd)
    }

    pub fn z_then_push(&mut self, x: f64) -> Option<f64> {
        let z = self.z(x);
        self.q.push_back(x);
        self.sum += x;
        self.sumsq += x * x;
        if self.q.len() > self.n
            && let Some(o) = self.q.pop_front()
        {
            self.sum -= o;
            self.sumsq -= o * o;
        }
        self.pushes += 1;
        if self.pushes.is_multiple_of(600) {
            // exact recompute: no drift from the running sums
            self.sum = self.q.iter().sum();
            self.sumsq = self.q.iter().map(|v| v * v).sum();
        }
        z
    }
}

/// Absorption: aggressive volume executed AT an unchanged touch over the window, relative to the displayed touch size.
#[derive(Clone, Debug, Default)]
pub struct Absorb {
    bid_px: i64,
    bid_since: i64,
    ask_px: i64,
    ask_since: i64,
    hits: VecDeque<(i64, i64, bool, f64)>,
}

impl Absorb {
    pub fn on_l1(&mut self, now: i64, bid: i64, ask: i64) {
        if bid != self.bid_px {
            self.bid_px = bid;
            self.bid_since = now;
        }
        if ask != self.ask_px {
            self.ask_px = ask;
            self.ask_since = now;
        }
    }

    pub fn on_trade(&mut self, t: &Tr, keep_ms: i64) {
        self.hits.push_back((t.ms, t.px, t.buy, t.vol));
        while self.hits.front().is_some_and(|h| h.0 < t.ms - keep_ms) {
            self.hits.pop_front();
        }
    }

    pub fn reset(&mut self) {
        *self = Self::default();
    }

    /// Ratio of aggressive volume at the touch (sells into the bid for `bid = true`) over the last `window_ms`, to the
    /// displayed touch size, provided the touch price has not changed during the window; 0 otherwise.
    pub fn ratio(&self, now: i64, window_ms: i64, bid: bool, touch_qty: f64) -> f64 {
        let (px, since) = if bid { (self.bid_px, self.bid_since) } else { (self.ask_px, self.ask_since) };
        if now - since < window_ms || touch_qty <= 0.0 {
            return 0.0;
        }
        let v: f64 = self.hits.iter().filter(|h| h.0 > now - window_ms && h.1 == px && h.2 != bid).map(|h| h.3).sum();
        v / touch_qty
    }
}

/// Features at one evaluation instant (prices in ticks; flows in contracts).
#[derive(Clone, Debug, Default, Serialize)]
pub struct Feat {
    pub ms: i64,
    pub bid: i64,
    pub ask: i64,
    pub bid_qty: f64,
    pub ask_qty: f64,
    pub spread_ticks: i64,
    /// microprice - mid, in ticks.
    pub micro_ticks: f64,
    pub imbalance: f64,
    pub ofi: Vec<f64>,
    pub ofi_z: Vec<Option<f64>>,
    pub delta: Vec<f64>,
    pub buy_sweep_age_s: Option<f64>,
    pub sell_sweep_age_s: Option<f64>,
    pub absorb_bid: f64,
    pub absorb_ask: f64,
    pub swing_lo: i64,
    pub swing_hi: i64,
    /// Log return of the mid over the last 60 s (house rule: log returns).
    pub ret_60s: f64,
}

/// The H-SCALP1 entry rule.
pub fn signal(cfg: &ScalpCfg, f: &Feat) -> Option<Side> {
    let i = cfg.signal_idx();
    let z = f.ofi_z.get(i).copied().flatten()?;
    let d = f.delta.get(i).copied().unwrap_or(0.0);
    if f.spread_ticks > cfg.max_spread_ticks || f.spread_ticks < 1 {
        return None;
    }
    let recent = |age: Option<f64>| age.is_some_and(|a| a <= cfg.sweep_lookback_s as f64);
    if z >= cfg.ofi_z_entry && d > 0.0 && f.micro_ticks > 0.0 && !recent(f.sell_sweep_age_s) {
        return Some(Side::Long);
    }
    if z <= -cfg.ofi_z_entry && d < 0.0 && f.micro_ticks < 0.0 && !recent(f.buy_sweep_age_s) {
        return Some(Side::Short);
    }
    None
}

// ---------------------------------------------------------------- bracket, funding, costs (pure)

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub struct Bracket {
    pub entry: i64,
    pub stop: i64,
    pub tp: i64,
    /// Stop distance in ticks (1 R in price).
    pub d: i64,
}

/// Structural stop 1 tick beyond the swing extreme (long: the low), at least `min_stop_ticks`; TP at exactly
/// `reward_r` x the distance. Err when the stop would be farther than `max_stop_ticks`.
pub fn bracket(cfg: &ScalpCfg, side: Side, entry: i64, swing_extreme: i64) -> Result<Bracket, String> {
    let stop = match side {
        Side::Long => (entry - cfg.min_stop_ticks).min(swing_extreme - 1),
        Side::Short => (entry + cfg.min_stop_ticks).max(swing_extreme + 1),
    };
    let d = (entry - stop).abs();
    if d > cfg.max_stop_ticks {
        return Err(format!("stop_too_far: {d} ticks > {}", cfg.max_stop_ticks));
    }
    let tp = entry + side.sign() as i64 * cfg.reward_r * d;
    Ok(Bracket { entry, stop, tp, d })
}

/// 1 R in USD: the loss of a plain stop-out under the shadow model (stop slip and both fees included).
pub fn r_unit_usd(cfg: &ScalpCfg, qty: f64, b: &Bracket, tick: f64) -> f64 {
    let (e, s) = (b.entry as f64 * tick, b.stop as f64 * tick);
    qty * ((e - s).abs() + cfg.stop_slip_ticks as f64 * tick + e * cfg.maker_bps / 1e4 + s * cfg.taker_bps / 1e4)
}

/// Size: floor(notional cap / price) on the qty step; None below the venue minimum qty or notional.
pub fn size_qty(cap_usd: f64, px: f64, qty_step: f64, min_qty: f64, min_notional: f64) -> Option<f64> {
    if px <= 0.0 || qty_step <= 0.0 {
        return None;
    }
    let q = ((cap_usd / px) / qty_step + 1e-9).floor() * qty_step;
    let q = (q / qty_step).round() * qty_step;
    (q >= min_qty && q * px >= min_notional && q * px <= cap_usd + 1e-9).then_some(q)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum FundingAct {
    None,
    Flatten,
    HoldLong,
}

/// Inside the last `funding_flat_min` before a funding time: shorts always flatten; longs flatten unless already
/// >= `long_hold_min_r` in profit (then held across and logged).
pub fn funding_action(cfg: &ScalpCfg, side: Side, now: i64, next_funding_ms: i64, open_r: f64) -> FundingAct {
    let to = next_funding_ms - now;
    if next_funding_ms <= 0 || to <= 0 || to as f64 > cfg.funding_flat_min * 60_000.0 {
        return FundingAct::None;
    }
    match side {
        Side::Short => FundingAct::Flatten,
        Side::Long if open_r >= cfg.long_hold_min_r => FundingAct::HoldLong,
        Side::Long => FundingAct::Flatten,
    }
}

/// No new short within `short_entry_block_min` of funding; no new long inside the flat window (+1 min).
pub fn entry_blocked_by_funding(cfg: &ScalpCfg, side: Side, now: i64, next_funding_ms: i64) -> bool {
    if next_funding_ms <= now {
        return false;
    }
    let mins = (next_funding_ms - now) as f64 / 60_000.0;
    match side {
        Side::Short => mins < cfg.short_entry_block_min,
        Side::Long => mins < cfg.funding_flat_min + 1.0,
    }
}

/// Next funding time on a UTC-aligned grid of `interval_ms` (Bybit 8 h: 00, 08, 16 UTC). Fallback when the ticker
/// stream has not sent nextFundingTime.
pub fn next_funding_on_grid(now: i64, interval_ms: i64) -> i64 {
    if interval_ms <= 0 {
        return 0;
    }
    (now.div_euclid(interval_ms) + 1) * interval_ms
}

// ---------------------------------------------------------------- shadow fill model (pure)

/// A resting passive order in the conservative queue model. `buy`: a bid (filled by sellers).
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct Passive {
    pub buy: bool,
    pub px: i64,
    pub qty: f64,
    /// Displayed size at our price when we joined (we queue behind it).
    pub ahead: f64,
    /// Volume traded at our price since we joined.
    pub traded_at: f64,
}

impl Passive {
    /// True when this print fills the order: a print through our price (any aggressor), or prints at our price by the
    /// other side whose cumulative volume exceeds the queue ahead plus our qty.
    pub fn on_trade(&mut self, px: i64, vol: f64, aggressor_buy: bool) -> bool {
        let through = if self.buy { px < self.px } else { px > self.px };
        if through {
            return true;
        }
        if px == self.px && aggressor_buy != self.buy {
            self.traded_at += vol;
            return self.traded_at > self.ahead + self.qty;
        }
        false
    }
}

/// Stop check on a print (LastPrice trigger): Some(fill px in ticks) = min(stop - slip, print) for a long.
pub fn stop_fill(side: Side, stop: i64, slip: i64, print: i64) -> Option<i64> {
    match side {
        Side::Long if print <= stop => Some((stop - slip).min(print)),
        Side::Short if print >= stop => Some((stop + slip).max(print)),
        _ => None,
    }
}

/// Take profit fills only on a trade THROUGH the target.
pub fn tp_hit(side: Side, tp: i64, print: i64) -> bool {
    match side {
        Side::Long => print > tp,
        Side::Short => print < tp,
    }
}

// ---------------------------------------------------------------- engine (no I/O: returns journal events)

#[derive(Clone, Debug)]
pub struct Inst {
    pub tick: f64,
    pub qty_step: f64,
    pub min_qty: f64,
    pub min_notional: f64,
    pub funding_interval_ms: i64,
}

#[derive(Clone, Debug, Serialize)]
struct Entering {
    id: String,
    side: Side,
    order: Passive,
    placed_ms: i64,
    br: Bracket,
    r_usd: f64,
}

#[derive(Clone, Debug, Serialize)]
struct Open {
    id: String,
    side: Side,
    qty: f64,
    br: Bracket,
    r_usd: f64,
    filled_ms: i64,
    exit: Option<(Passive, i64)>,
    funding_usd: f64,
    held_across: bool,
}

#[derive(Clone, Debug, Serialize)]
enum Phase {
    Flat,
    Entering(Entering),
    Open(Open),
}

#[derive(Clone, Debug, Default, Serialize)]
pub struct Stats {
    pub signals: u64,
    pub orders: u64,
    pub fills: u64,
    pub cancels: u64,
    pub skips: u64,
    pub trades: u64,
    pub wins: u64,
    pub tp_hits: u64,
    pub net_r: f64,
    pub net_usd: f64,
    pub consec_losses: u32,
    pub day: String,
    pub day_net_usd: f64,
    pub by_exit: BTreeMap<String, u32>,
    /// Live sub-limits that WOULD have halted a live run (shadow keeps trading to collect the sample).
    pub would_halt: Vec<String>,
}

pub struct Engine {
    pub cfg: ScalpCfg,
    pub inst: Inst,
    pub mode: String,
    book: Book,
    prev_l1: Option<L1>,
    ofi: SecSums,
    delta: SecSums,
    z: Vec<RollingZ>,
    trades: VecDeque<Tr>,
    last_sweep: [Option<i64>; 2],
    absorb: Absorb,
    mids: VecDeque<(i64, f64)>,
    pub mark: f64,
    pub next_funding_ms: i64,
    pub funding_rate: f64,
    phase: Phase,
    cooldown_until: i64,
    seq: u64,
    pub stats: Stats,
    pub last_feat: Option<Feat>,
    pub resets: u64,
    /// false (live mode): features and the signal only; positions are managed by the live runner, nothing simulated.
    simulate: bool,
    /// The entry rule's output at the last evaluation (side and the features it fired on).
    pub last_signal: Option<(Side, Feat)>,
    ret1s: RollingZ,
    last_mid_sec: Option<(i64, f64)>,
}

fn ev(kind: &str, ms: i64, mut payload: Value) -> Value {
    payload["type"] = json!(kind);
    payload["ms"] = json!(ms);
    payload
}

fn num(v: &Value) -> Option<f64> {
    v.as_str().and_then(|s| s.parse().ok()).or_else(|| v.as_f64())
}

impl Engine {
    pub fn new(cfg: ScalpCfg, inst: Inst, mode: &str) -> Self {
        let zs = cfg.ofi_windows_s.iter().map(|_| RollingZ::new(cfg.z_window_s, cfg.z_min_obs)).collect();
        let h = cfg.max_window_s() + 1;
        Self {
            ofi: SecSums::new(h),
            delta: SecSums::new(h),
            z: zs,
            inst,
            mode: mode.to_string(),
            book: Book::default(),
            prev_l1: None,
            trades: VecDeque::new(),
            last_sweep: [None, None],
            absorb: Absorb::default(),
            mids: VecDeque::new(),
            mark: 0.0,
            next_funding_ms: 0,
            funding_rate: 0.0,
            phase: Phase::Flat,
            cooldown_until: 0,
            seq: 0,
            stats: Stats::default(),
            last_feat: None,
            resets: 0,
            simulate: mode == "shadow",
            last_signal: None,
            ret1s: RollingZ::new(cfg.z_window_s, cfg.z_min_obs),
            last_mid_sec: None,
            cfg,
        }
    }

    fn t(&self, px: f64) -> i64 {
        to_ticks(px, self.inst.tick)
    }

    fn px(&self, t: i64) -> f64 {
        t as f64 * self.inst.tick
    }

    fn keep_ms(&self) -> i64 {
        1000 * self.cfg.swing_lookback_s.max(self.cfg.sweep_lookback_s).max(self.cfg.absorption_window_s).max(self.cfg.max_window_s())
    }

    pub fn position_label(&self) -> String {
        match &self.phase {
            Phase::Flat => "flat".into(),
            Phase::Entering(e) => format!("entering {:?}@{:.5}", e.side, self.px(e.order.px)),
            Phase::Open(o) => format!("{:?}@{:.5} sl {:.5} tp {:.5}", o.side, self.px(o.br.entry), self.px(o.br.stop), self.px(o.br.tp)),
        }
    }

    // ------------------------------------------------------------ inputs

    pub fn on_reset(&mut self, now: i64) -> Vec<Value> {
        self.book.invalidate();
        self.prev_l1 = None;
        self.absorb.reset();
        self.resets += 1;
        vec![ev("reconnect", now, json!({"phase": self.position_label()}))]
    }

    pub fn on_book(&mut self, now: i64, kind: &str, data: &Value) {
        if !self.book.apply(kind, data) {
            return;
        }
        let Some(l1) = self.book.l1() else { return };
        if l1.bid >= l1.ask {
            return;
        }
        // OFI between consecutive L1 states of one book (a snapshot restarts the chain)
        if kind != "snapshot"
            && let Some(p) = self.prev_l1
            && p != l1
        {
            self.ofi.add(now.div_euclid(1000), ofi(&p, &l1));
        }
        self.prev_l1 = Some(l1);
        let (b, a) = (self.t(l1.bid), self.t(l1.ask));
        self.absorb.on_l1(now, b, a);
    }

    pub fn on_ticker(&mut self, data: &Value) {
        if let Some(m) = num(&data["markPrice"]).filter(|x| *x > 0.0) {
            self.mark = m;
        }
        if let Some(t) = num(&data["nextFundingTime"]).filter(|x| *x > 0.0) {
            self.next_funding_ms = t as i64;
        }
        if let Some(r) = num(&data["fundingRate"]) {
            self.funding_rate = r;
        }
    }

    /// One publicTrade message: features first, then fills / exits print by print.
    pub fn on_trades(&mut self, now: i64, data: &Value) -> Vec<Value> {
        let batch: Vec<Tr> = data
            .as_array()
            .into_iter()
            .flatten()
            .filter_map(|t| {
                let (ex, p, v) = (t["T"].as_i64()?, num(&t["p"])?, num(&t["v"])?);
                Some(Tr { ms: now, ex_ms: ex, px: self.t(p), buy: t["S"].as_str() == Some("Buy"), vol: v })
            })
            .collect();
        let keep = self.keep_ms();
        for (_, buy, _) in sweeps(&batch, self.cfg.sweep_min_levels) {
            self.last_sweep[if buy { 0 } else { 1 }] = Some(now);
        }
        let mut out = Vec::new();
        for t in &batch {
            self.delta.add(now.div_euclid(1000), if t.buy { t.vol } else { -t.vol });
            self.absorb.on_trade(t, keep);
            self.trades.push_back(*t);
            if self.simulate {
                out.extend(self.on_print(now, t));
            }
        }
        while self.trades.front().is_some_and(|x| x.ms < now - keep) {
            self.trades.pop_front();
        }
        out
    }

    fn on_print(&mut self, now: i64, t: &Tr) -> Vec<Value> {
        let mut out = Vec::new();
        if let Phase::Entering(e) = &mut self.phase
            && e.order.on_trade(t.px, t.vol, t.buy)
        {
            let e = e.clone();
            self.stats.fills += 1;
            out.push(ev("fill", now, json!({"id": e.id, "leg": "entry", "side": e.side, "px": self.px(e.order.px), "qty": e.order.qty, "queue_ahead": e.order.ahead, "traded_at": e.order.traded_at, "print_px": self.px(t.px), "wait_s": (now - e.placed_ms) as f64 / 1000.0, "fee_kind": "maker"})));
            self.phase = Phase::Open(Open { id: e.id, side: e.side, qty: e.order.qty, br: e.br, r_usd: e.r_usd, filled_ms: now, exit: None, funding_usd: 0.0, held_across: false });
        }
        let (side, stop, tp) = match &self.phase {
            Phase::Open(o) => (o.side, o.br.stop, o.br.tp),
            _ => return out,
        };
        // worst-case ordering inside a print: the stop first, then the target, then the passive time exit
        if self.cfg.stop_trigger_by == "LastPrice"
            && let Some(px) = stop_fill(side, stop, self.cfg.stop_slip_ticks, t.px)
        {
            out.extend(self.close(now, px, "stop", false));
            return out;
        }
        if tp_hit(side, tp, t.px) {
            out.extend(self.close(now, tp, "tp", true));
            return out;
        }
        let passive_px = match &mut self.phase {
            Phase::Open(Open { exit: Some((p, _)), .. }) => p.on_trade(t.px, t.vol, t.buy).then_some(p.px),
            _ => None,
        };
        if let Some(px) = passive_px {
            out.extend(self.close(now, px, "time_passive", true));
        }
        out
    }

    // ------------------------------------------------------------ the per-second step

    fn features(&mut self, now: i64) -> Option<Feat> {
        let l1 = self.book.l1().filter(|l| self.book.valid && l.bid < l.ask)?;
        let (bid, ask) = (self.t(l1.bid), self.t(l1.ask));
        let mid = 0.5 * (l1.bid + l1.ask);
        let sec = now.div_euclid(1000);
        let mut f = Feat { ms: now, bid, ask, bid_qty: l1.bid_qty, ask_qty: l1.ask_qty, spread_ticks: ask - bid, ..Default::default() };
        f.micro_ticks = (microprice(&l1) - mid) / self.inst.tick;
        let tick = self.inst.tick;
        let bids: Vec<(i64, f64)> = self.book.bids_desc().take(64).map(|(p, q)| (to_ticks(p, tick), q)).collect();
        let asks: Vec<(i64, f64)> = self.book.asks_asc().take(64).map(|(p, q)| (to_ticks(p, tick), q)).collect();
        f.imbalance = imbalance_within(&bids, &asks, self.cfg.imbalance_ticks);
        for (k, w) in self.cfg.ofi_windows_s.clone().iter().enumerate() {
            let o = self.ofi.sum(sec, *w);
            f.ofi.push(o);
            f.ofi_z.push(self.z[k].z_then_push(o));
            f.delta.push(self.delta.sum(sec, *w));
        }
        let age = |x: Option<i64>| x.map(|m| (now - m) as f64 / 1000.0);
        f.buy_sweep_age_s = age(self.last_sweep[0]);
        f.sell_sweep_age_s = age(self.last_sweep[1]);
        let wa = self.cfg.absorption_window_s * 1000;
        f.absorb_bid = self.absorb.ratio(now, wa, true, l1.bid_qty);
        f.absorb_ask = self.absorb.ratio(now, wa, false, l1.ask_qty);
        let since = now - self.cfg.swing_lookback_s * 1000;
        let recent = self.trades.iter().filter(|t| t.ms >= since);
        f.swing_lo = recent.clone().map(|t| t.px).min().unwrap_or(bid).min(bid);
        f.swing_hi = recent.map(|t| t.px).max().unwrap_or(ask).max(ask);
        self.mids.push_back((now, mid));
        while self.mids.front().is_some_and(|m| m.0 < now - 60_000) {
            self.mids.pop_front();
        }
        f.ret_60s = self.mids.front().map(|m| (mid / m.1).ln()).unwrap_or(0.0);
        // 1 s mid log returns (one sample per evaluated second) for the live sigma
        if let Some((ps, pm)) = self.last_mid_sec
            && sec > ps
            && pm > 0.0
        {
            self.ret1s.z_then_push((mid / pm).ln() / ((sec - ps) as f64).sqrt());
        }
        self.last_mid_sec = Some((sec, mid));
        Some(f)
    }

    fn roll_day(&mut self, now: i64) {
        let d = format_date(now);
        if self.stats.day != d {
            self.stats.day = d;
            self.stats.day_net_usd = 0.0;
        }
    }

    pub fn on_second(&mut self, now: i64) -> Vec<Value> {
        let mut out = Vec::new();
        self.roll_day(now);
        if self.next_funding_ms <= 0 {
            self.next_funding_ms = next_funding_on_grid(now, self.inst.funding_interval_ms);
        }
        // funding settlement passed while holding: book it, move to the next grid time
        if now >= self.next_funding_ms {
            if let Phase::Open(o) = &mut self.phase {
                let ref_px = if self.mark > 0.0 { self.mark } else { o.br.entry as f64 * self.inst.tick };
                let pay = -o.side.sign() * self.funding_rate * o.qty * ref_px;
                o.funding_usd += pay;
                out.push(ev("funding", now, json!({"id": o.id, "side": o.side, "rate": self.funding_rate, "ref_px": ref_px, "usd": pay, "funding_ms": self.next_funding_ms})));
            }
            self.next_funding_ms = next_funding_on_grid(now, self.inst.funding_interval_ms).max(self.next_funding_ms + self.inst.funding_interval_ms);
        }
        let Some(f) = self.features(now) else { return out };
        self.last_feat = Some(f.clone());
        let sig = signal(&self.cfg, &f);
        self.last_signal = sig.map(|s| (s, f.clone()));
        if !self.simulate {
            return out;
        }
        out.extend(self.manage(now, &f, sig));
        if let Some(side) = sig {
            out.extend(self.on_signal(now, &f, side));
        }
        out
    }

    fn manage(&mut self, now: i64, f: &Feat, sig: Option<Side>) -> Vec<Value> {
        let mut out = Vec::new();
        match self.phase.clone() {
            Phase::Flat => {}
            Phase::Entering(e) => {
                let z = f.ofi_z.get(self.cfg.signal_idx()).copied().flatten().unwrap_or(0.0);
                let reason = if now - e.placed_ms >= self.cfg.entry_timeout_s * 1000 {
                    Some("timeout")
                } else if z * e.side.sign() < 0.0 {
                    Some("z_crossed_zero")
                } else if sig.is_some_and(|s| s != e.side) {
                    Some("opposite_signal")
                } else {
                    None
                };
                if let Some(r) = reason {
                    self.stats.cancels += 1;
                    out.push(ev("cancel", now, json!({"id": e.id, "side": e.side, "px": self.px(e.order.px), "reason": r, "queue_ahead": e.order.ahead, "traded_at": e.order.traded_at, "age_s": (now - e.placed_ms) as f64 / 1000.0})));
                    self.phase = Phase::Flat;
                }
            }
            Phase::Open(o) => {
                let mid_t = 0.5 * (f.bid + f.ask) as f64;
                let open_usd = o.side.sign() * (mid_t - o.br.entry as f64) * self.inst.tick * o.qty;
                let open_r = if o.r_usd > 0.0 { open_usd / o.r_usd } else { 0.0 };
                // MarkPrice trigger (if configured): exit at min(stop - slip, bid) for a long
                if self.cfg.stop_trigger_by == "MarkPrice" && self.mark > 0.0 {
                    let m = self.t(self.mark);
                    if stop_fill(o.side, o.br.stop, 0, m).is_some() {
                        let touch = if o.side == Side::Long { f.bid } else { f.ask };
                        let px = stop_fill(o.side, o.br.stop, self.cfg.stop_slip_ticks, touch).unwrap_or(o.br.stop - o.side.sign() as i64 * self.cfg.stop_slip_ticks);
                        return self.close(now, px, "stop_mark", false);
                    }
                }
                match funding_action(&self.cfg, o.side, now, self.next_funding_ms, open_r) {
                    FundingAct::Flatten => {
                        let px = self.market_px(o.side, o.qty, f);
                        return self.close(now, px, "funding_flat", false);
                    }
                    FundingAct::HoldLong if !o.held_across => {
                        if let Phase::Open(x) = &mut self.phase {
                            x.held_across = true;
                        }
                        out.push(ev("funding_hold", now, json!({"id": o.id, "open_r": open_r, "rate": self.funding_rate, "funding_ms": self.next_funding_ms})));
                    }
                    _ => {}
                }
                let held = now - o.filled_ms;
                let ts_ms = (self.cfg.time_stop_min * 60_000.0) as i64;
                match &o.exit {
                    None if held >= ts_ms && !o.held_across => {
                        // passive at the far touch (long: sell at the ask), behind the displayed queue
                        let (px, ahead) = if o.side == Side::Long { (f.ask, f.ask_qty) } else { (f.bid, f.bid_qty) };
                        let p = Passive { buy: o.side == Side::Short, px, qty: o.qty, ahead, traded_at: 0.0 };
                        out.push(ev("order", now, json!({"id": o.id, "leg": "time_exit_passive", "side": o.side, "px": self.px(px), "qty": o.qty, "queue_ahead": ahead, "open_r": open_r})));
                        if let Phase::Open(x) = &mut self.phase {
                            x.exit = Some((p, now));
                        }
                    }
                    Some((_, started)) if now - started >= self.cfg.passive_exit_s * 1000 => {
                        let px = self.market_px(o.side, o.qty, f);
                        return self.close(now, px, "time_market", false);
                    }
                    _ => {}
                }
            }
        }
        out
    }

    /// Market exit price: the near touch, one tick worse when our qty exceeds the touch size.
    fn market_px(&self, side: Side, qty: f64, f: &Feat) -> i64 {
        match side {
            Side::Long => f.bid - i64::from(qty > f.bid_qty),
            Side::Short => f.ask + i64::from(qty > f.ask_qty),
        }
    }

    fn on_signal(&mut self, now: i64, f: &Feat, side: Side) -> Vec<Value> {
        self.stats.signals += 1;
        let mut out = Vec::new();
        let base = json!({"side": side, "feat": f});
        let blocked = match &self.phase {
            Phase::Flat if now < self.cooldown_until => Some("cooldown".to_string()),
            Phase::Flat if entry_blocked_by_funding(&self.cfg, side, now, self.next_funding_ms) => Some("funding_block".to_string()),
            Phase::Flat => None,
            Phase::Entering(_) => Some("entering".to_string()),
            Phase::Open(_) => Some("in_position".to_string()),
        };
        if let Some(r) = blocked {
            let mut s = base;
            s["action"] = json!("ignored");
            s["reason"] = json!(r);
            out.push(ev("signal", now, s));
            return out;
        }
        let (entry, extreme, ahead) = match side {
            Side::Long => (f.bid, f.swing_lo, f.bid_qty),
            Side::Short => (f.ask, f.swing_hi, f.ask_qty),
        };
        let br = match bracket(&self.cfg, side, entry, extreme) {
            Ok(b) => b,
            Err(e) => {
                self.stats.skips += 1;
                let mut s = base;
                s["action"] = json!("skip");
                s["reason"] = json!(e);
                out.push(ev("signal", now, s));
                return out;
            }
        };
        let px = self.px(entry);
        let Some(qty) = size_qty(self.cfg.limits.notional_cap_usd, px, self.inst.qty_step, self.inst.min_qty, self.inst.min_notional) else {
            self.stats.skips += 1;
            let mut s = base;
            s["action"] = json!("skip");
            s["reason"] = json!("size_below_venue_minimum");
            out.push(ev("signal", now, s));
            return out;
        };
        let r_usd = r_unit_usd(&self.cfg, qty, &br, self.inst.tick);
        self.seq += 1;
        let id = format!("{}-{}", now / 1000, self.seq);
        self.stats.orders += 1;
        if self.stats.orders > self.cfg.limits.max_entries_per_run as u64 {
            self.would_halt("max_entries_per_run");
        }
        let mut s = base;
        s["action"] = json!("order");
        s["id"] = json!(id);
        out.push(ev("signal", now, s));
        out.push(ev(
            "order",
            now,
            json!({"id": id, "leg": "entry", "side": side, "tif": "PostOnly", "px": px, "qty": qty, "notional_usd": qty * px, "queue_ahead": ahead,
                   "stop": self.px(br.stop), "tp": self.px(br.tp), "stop_ticks": br.d, "reward_r": self.cfg.reward_r, "r_usd": r_usd,
                   "stop_bps": br.d as f64 * self.inst.tick / px * 1e4, "stop_trigger_by": self.cfg.stop_trigger_by, "next_funding_ms": self.next_funding_ms}),
        ));
        self.phase = Phase::Entering(Entering { id, side, order: Passive { buy: side == Side::Long, px: entry, qty, ahead, traded_at: 0.0 }, placed_ms: now, br, r_usd });
        out
    }

    fn would_halt(&mut self, k: &str) {
        if !self.stats.would_halt.iter().any(|x| x == k) {
            self.stats.would_halt.push(k.to_string());
        }
    }

    /// Close the open position at `px` (ticks); maker or taker exit fee.
    fn close(&mut self, now: i64, px: i64, kind: &str, maker: bool) -> Vec<Value> {
        let Phase::Open(o) = self.phase.clone() else { return vec![] };
        let tick = self.inst.tick;
        let (e, x) = (o.br.entry as f64 * tick, px as f64 * tick);
        let gross = o.side.sign() * (x - e) * o.qty;
        let fee_in = e * o.qty * self.cfg.maker_bps / 1e4;
        let fee_out = x * o.qty * if maker { self.cfg.maker_bps } else { self.cfg.taker_bps } / 1e4;
        let net = gross - fee_in - fee_out + o.funding_usd;
        let r = if o.r_usd > 0.0 { net / o.r_usd } else { 0.0 };
        let s = &mut self.stats;
        s.trades += 1;
        s.net_r += r;
        s.net_usd += net;
        s.day_net_usd += net;
        *s.by_exit.entry(kind.to_string()).or_default() += 1;
        if kind == "tp" {
            s.tp_hits += 1;
        }
        if net > 0.0 {
            s.wins += 1;
            s.consec_losses = 0;
        } else {
            s.consec_losses += 1;
        }
        if s.consec_losses >= self.cfg.limits.max_consecutive_losses {
            self.would_halt("max_consecutive_losses");
        }
        if self.stats.day_net_usd <= -self.cfg.limits.day_loss_stop_usd {
            self.would_halt("day_loss_stop");
        }
        self.phase = Phase::Flat;
        self.cooldown_until = now + self.cfg.cooldown_s * 1000;
        vec![ev(
            "exit",
            now,
            json!({"id": o.id, "side": o.side, "kind": kind, "fee_kind": if maker { "maker" } else { "taker" }, "entry": e, "exit": x, "stop": o.br.stop as f64 * tick, "tp": o.br.tp as f64 * tick,
                   "stop_ticks": o.br.d, "qty": o.qty, "gross_usd": gross, "fees_usd": fee_in + fee_out, "funding_usd": o.funding_usd, "net_usd": net, "r_usd": o.r_usd, "net_r": r,
                   "held_s": (now - o.filled_ms) as f64 / 1000.0, "held_across_funding": o.held_across, "mark": self.mark,
                   "ret_log": o.side.sign() * (x / e).ln()}),
        )]
    }

    /// Shutdown: a shadow position is closed at the market touch (journaled as "shutdown"); a resting entry is cancelled.
    pub fn shutdown(&mut self, now: i64) -> Vec<Value> {
        match self.phase.clone() {
            Phase::Open(o) => match self.last_feat.clone() {
                Some(f) => {
                    let px = self.market_px(o.side, o.qty, &f);
                    self.close(now, px, "shutdown", false)
                }
                None => vec![],
            },
            Phase::Entering(e) => {
                self.phase = Phase::Flat;
                self.stats.cancels += 1;
                vec![ev("cancel", now, json!({"id": e.id, "reason": "shutdown"}))]
            }
            Phase::Flat => vec![],
        }
    }

    /// Volatility over `horizon_s`: sd of 1 s mid log returns x sqrt(horizon). None during warm-up.
    pub fn sigma_over(&self, horizon_s: i64) -> Option<f64> {
        self.ret1s.sd().map(|s| s * (horizon_s.max(1) as f64).sqrt())
    }

    pub fn status_line(&self, now: i64) -> String {
        let s = &self.stats;
        let ts = chrono::DateTime::from_timestamp_millis(now).unwrap_or_default().format("%Y-%m-%dT%H:%M:%SZ");
        let f = self.last_feat.as_ref();
        let z = f.and_then(|f| f.ofi_z.get(self.cfg.signal_idx()).copied().flatten());
        let warm = self.z.get(self.cfg.signal_idx()).map(|z| z.len()).unwrap_or(0);
        format!(
            "STATUS {ts} {} {} | signals {} orders {} fills {} cancels {} skips {} | trades {} hits {} tp {} | net R {:+.3} (mean {:+.3}) | net USD {:+.4} day {:+.4} | {} | z{} {} spr {} warm {}/{} | book {} resets {} | halt {:?}",
            self.cfg.symbol,
            self.mode,
            s.signals,
            s.orders,
            s.fills,
            s.cancels,
            s.skips,
            s.trades,
            s.wins,
            s.tp_hits,
            s.net_r,
            if s.trades > 0 { s.net_r / s.trades as f64 } else { 0.0 },
            s.net_usd,
            s.day_net_usd,
            self.position_label(),
            self.cfg.signal_window_s,
            z.map(|z| format!("{z:+.2}")).unwrap_or_else(|| "-".into()),
            f.map(|f| f.spread_ticks.to_string()).unwrap_or_else(|| "-".into()),
            warm,
            self.cfg.z_min_obs,
            if self.book.valid { "ok" } else { "INVALID" },
            self.resets,
            s.would_halt
        )
    }
}

// ---------------------------------------------------------------- journal

struct ScalpJournal {
    dir: PathBuf,
    run_id: String,
    cfg_hash: String,
    mode: String,
}

impl ScalpJournal {
    fn write(&self, mut v: Value) -> anyhow::Result<()> {
        let now = Utc::now();
        v["ts"] = json!(now.to_rfc3339_opts(SecondsFormat::Millis, true));
        v["run_id"] = json!(self.run_id);
        v["version"] = json!(SCALP_VERSION);
        v["cfg_hash"] = json!(self.cfg_hash);
        v["mode"] = json!(self.mode);
        std::fs::create_dir_all(&self.dir)?;
        let p = self.dir.join(format!("{}.jsonl", now.format("%Y-%m-%d")));
        let mut f = std::fs::OpenOptions::new().create(true).append(true).open(&p).with_context(|| p.display().to_string())?;
        writeln!(f, "{}", serde_json::to_string(&v)?)?;
        Ok(())
    }
}

// ---------------------------------------------------------------- CLI

#[derive(clap::Args, Debug)]
pub struct ScalpArgs {
    /// shadow (no orders, simulated fills) | live (refused in this build)
    #[arg(long, default_value = "shadow")]
    pub mode: String,
    /// Config file relative to the repo.
    #[arg(long, default_value = "config/scalp_kava.yaml")]
    pub config: String,
    /// Stop after this many minutes (default: run until killed / Ctrl-C).
    #[arg(long)]
    pub minutes: Option<u64>,
    /// Live only: run id; with the approval gate MANUAL the run needs data/live/approvals/scalp-<run_id>.ok.
    #[arg(long)]
    pub run_id: Option<String>,
    /// Live only: most entries this run (<= limits.max_entries_per_run).
    #[arg(long, default_value_t = 12)]
    pub max_entries: u32,
}

pub async fn run(ctx: &Ctx, a: ScalpArgs) -> anyhow::Result<()> {
    let cfg = ScalpCfg::load(&ctx.s.root.join(&a.config))?;
    match a.mode.as_str() {
        "shadow" => {}
        "live" => {
            let run_id = a.run_id.clone().context("--mode live needs --run-id (the run approval is data/live/approvals/scalp-<run_id>.ok)")?;
            return crate::scalp_live::run_live(ctx, cfg, &a.config, &run_id, a.max_entries, a.minutes).await;
        }
        m => bail!("--mode must be shadow or live, got {m:?}"),
    }

    // instrument rules (public REST, mainnet market data)
    let rest = ctx.market_rest()?;
    let md = MarketData::new(&rest, "linear");
    let ins = md.instruments().await?.into_iter().find(|i| i.symbol == cfg.symbol).with_context(|| format!("{} not listed", cfg.symbol))?;
    ensure!(ins.status == "Trading", "{} status {}", cfg.symbol, ins.status);
    let inst = Inst {
        tick: ins.tick_size,
        qty_step: ins.qty_step,
        min_qty: ins.min_qty,
        min_notional: if ins.min_notional.is_finite() { ins.min_notional } else { 5.0 },
        funding_interval_ms: ins.funding_interval_min.max(1) as i64 * 60_000,
    };
    let start = now_ms();
    let run_id = format!("{}-{}", chrono::DateTime::from_timestamp_millis(start).unwrap_or_default().format("%Y%m%dT%H%M%S"), std::process::id());
    let journal = ScalpJournal { dir: ctx.s.abs(&ctx.s.paths.journal).join("scalp"), run_id: run_id.clone(), cfg_hash: cfg.hash(), mode: a.mode.clone() };
    journal.write(ev(
        "run_start",
        start,
        json!({"symbol": cfg.symbol, "hypothesis": cfg.hypothesis, "config": cfg, "config_file": a.config, "tick": inst.tick, "qty_step": inst.qty_step, "min_qty": inst.min_qty,
               "min_notional": inst.min_notional, "funding_interval_ms": inst.funding_interval_ms, "pid": std::process::id(), "atx_version": env!("CARGO_PKG_VERSION")}),
    ))?;
    println!("atx scalp {} {} run {} cfg {} tick {} step {} (journal {})", a.mode, cfg.symbol, run_id, cfg.hash(), inst.tick, inst.qty_step, journal.dir.display());

    let url = ctx.s.bybit_endpoints(ctx.s.venues.bybit.market_data_env)?.ws_public.clone();
    let topics = vec![format!("orderbook.{}.{}", cfg.depth, cfg.symbol), format!("publicTrade.{}", cfg.symbol), format!("tickers.{}", cfg.symbol)];
    let (tx, mut rx) = tokio::sync::mpsc::channel::<WsMessage>(50_000);
    let ws_task = tokio::spawn(ws::run(url, topics, None, tx));

    let status_every = cfg.status_every_s.max(5) as i64 * 1000;
    let eval_ms = cfg.eval_interval_ms.max(200);
    let mut eng = Engine::new(cfg, inst, &a.mode);
    let mut tick = tokio::time::interval(Duration::from_millis(eval_ms));
    let mut next_status = start + status_every;
    let deadline = a.minutes.map(|m| start + m as i64 * 60_000);
    let ctrl_c = tokio::signal::ctrl_c();
    tokio::pin!(ctrl_c);
    let write_all = |evs: Vec<Value>| -> anyhow::Result<()> {
        for e in evs {
            journal.write(e)?;
        }
        Ok(())
    };
    loop {
        tokio::select! {
            m = rx.recv() => {
                let Some(m) = m else { bail!("websocket task ended") };
                let now = m.received_ts;
                if m.topic == "__reconnect__" {
                    write_all(eng.on_reset(now))?;
                } else if m.topic.starts_with("orderbook.") {
                    eng.on_book(now, &m.kind, &m.data);
                } else if m.topic.starts_with("publicTrade.") {
                    write_all(eng.on_trades(now, &m.data))?;
                } else if m.topic.starts_with("tickers.") {
                    eng.on_ticker(&m.data);
                }
            }
            _ = tick.tick() => {
                let now = now_ms();
                write_all(eng.on_second(now))?;
                if now >= next_status {
                    let line = eng.status_line(now);
                    println!("{line}");
                    let _ = std::io::stdout().flush();
                    write_all(vec![ev("status", now, json!({"line": line, "stats": eng.stats, "feat": eng.last_feat, "mark": eng.mark, "funding_rate": eng.funding_rate, "next_funding_ms": eng.next_funding_ms}))])?;
                    next_status = now + status_every;
                }
                if deadline.is_some_and(|d| now >= d) { break; }
            }
            _ = &mut ctrl_c => { println!("ctrl-c: stopping"); break; }
        }
    }
    let now = now_ms();
    write_all(eng.shutdown(now))?;
    let line = eng.status_line(now);
    println!("{line}");
    write_all(vec![ev("run_end", now, json!({"line": line, "stats": eng.stats}))])?;
    ws_task.abort();
    Ok(())
}

// ---------------------------------------------------------------- tests

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg() -> ScalpCfg {
        let p = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../config/scalp_kava.yaml");
        ScalpCfg::load(&p).expect("config/scalp_kava.yaml loads and validates")
    }

    fn live() -> LiveSmallCfg {
        LiveSmallCfg::load(&Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")).expect("live_small loads")
    }

    fn inst() -> Inst {
        Inst { tick: 0.00001, qty_step: 0.1, min_qty: 0.1, min_notional: 5.0, funding_interval_ms: 8 * 3_600_000 }
    }

    #[test]
    fn microprice_and_imbalance() {
        let l1 = L1 { bid: 0.02889, bid_qty: 3000.0, ask: 0.02891, ask_qty: 1000.0 };
        let m = microprice(&l1);
        assert!((m - (0.02889 * 1000.0 + 0.02891 * 3000.0) / 4000.0).abs() < 1e-12);
        assert!(m > 0.5 * (l1.bid + l1.ask), "heavier bid: microprice above mid (toward the thin ask)");
        let bids = [(2889, 3000.0), (2888, 1000.0), (2880, 9e9)];
        let asks = [(2891, 1000.0), (2893, 1000.0)];
        // within 5 ticks: bids 4000 (2880 is 9 ticks away), asks 2000
        assert!((imbalance_within(&bids, &asks, 5) - (4000.0 - 2000.0) / 6000.0).abs() < 1e-12);
        assert_eq!(imbalance_within(&[], &asks, 5), 0.0);
    }

    #[test]
    fn ofi_buckets_windows_and_z() {
        let mut s = SecSums::new(61);
        for sec in 0..100 {
            s.add(sec, 1.0);
        }
        s.add(99, 2.0);
        assert_eq!(s.sum(99, 10), 12.0, "seconds 90..=99 plus the extra 2");
        assert_eq!(s.sum(99, 60), 62.0);
        // L1 OFI sign: a bid step-up is buy pressure (at-flow ofi)
        let p = L1 { bid: 1.0, bid_qty: 5.0, ask: 1.1, ask_qty: 5.0 };
        assert!(ofi(&p, &L1 { bid: 1.01, bid_qty: 2.0, ..p }) > 0.0);
        let mut z = RollingZ::new(100, 30);
        for i in 0..29 {
            assert!(z.z_then_push((i % 5) as f64).is_none(), "no z before min_obs");
        }
        z.z_then_push(4.0);
        let zz = z.z(10.0).unwrap();
        assert!(zz > 3.0, "an outlier is a large z, got {zz}");
        // the window drops old samples and the running sums stay exact
        for _ in 0..2000 {
            z.z_then_push(1.0);
        }
        z.z_then_push(3.0);
        assert_eq!(z.len(), 100);
        assert!(z.z(3.0).unwrap() > 5.0);
    }

    #[test]
    fn sweep_detection() {
        let t = |ex: i64, px: i64, buy: bool| Tr { ms: 0, ex_ms: ex, px, buy, vol: 10.0 };
        let batch = [t(1, 100, true), t(1, 101, true), t(1, 102, true), t(2, 100, false), t(2, 100, false), t(3, 99, false), t(3, 98, false)];
        let s = sweeps(&batch, 2);
        assert_eq!(s, vec![(1, true, 3), (3, false, 2)], "same ts + side at >= 2 prices; two prints at one price is not a sweep");
        assert!(sweeps(&batch, 4).is_empty());
    }

    #[test]
    fn absorption_needs_an_unchanged_touch() {
        let mut a = Absorb::default();
        a.on_l1(0, 2889, 2891);
        for k in 0..5 {
            a.on_trade(&Tr { ms: 11_000 + k, ex_ms: 0, px: 2889, buy: false, vol: 500.0 }, 60_000);
        }
        // 2500 sold into an unchanged bid of 1000 over the last 10 s
        assert!((a.ratio(12_000, 10_000, true, 1000.0) - 2.5).abs() < 1e-12);
        assert_eq!(a.ratio(12_000, 10_000, false, 1000.0), 0.0, "no buying at the ask");
        a.on_l1(12_500, 2888, 2891);
        assert_eq!(a.ratio(13_000, 10_000, true, 1000.0), 0.0, "the bid moved: not absorption");
    }

    #[test]
    fn entry_rule_is_symmetric() {
        let c = cfg();
        let i = c.signal_idx();
        let mut f = Feat { spread_ticks: 2, micro_ticks: 0.3, ofi: vec![0.0; 3], ofi_z: vec![None; 3], delta: vec![0.0; 3], ..Default::default() };
        f.ofi_z[i] = Some(1.6);
        f.delta[i] = 100.0;
        assert_eq!(signal(&c, &f), Some(Side::Long));
        f.sell_sweep_age_s = Some(5.0);
        assert_eq!(signal(&c, &f), None, "sell sweep in the last 10 s");
        f.sell_sweep_age_s = Some(11.0);
        f.spread_ticks = 3;
        assert_eq!(signal(&c, &f), None, "spread too wide");
        let s = Feat { spread_ticks: 1, micro_ticks: -0.3, ofi_z: vec![Some(-1.6); 3], delta: vec![-5.0; 3], ..f.clone() };
        assert_eq!(signal(&c, &s), Some(Side::Short));
        let s2 = Feat { micro_ticks: 0.1, ..s.clone() };
        assert_eq!(signal(&c, &s2), None, "microprice must agree");
        let s3 = Feat { ofi_z: vec![None; 3], ..s };
        assert_eq!(signal(&c, &s3), None, "no z during warm-up");
    }

    #[test]
    fn bracket_math_2r_3r_and_ticks() {
        let mut c = cfg();
        // long at 2889, swing low 2887: stop = min(2889 - 4, 2886) = 2885, d = 4, tp = 2889 + 8
        let b = bracket(&c, Side::Long, 2889, 2887).unwrap();
        assert_eq!(b, Bracket { entry: 2889, stop: 2885, tp: 2897, d: 4 });
        // structural stop farther than the minimum: 1 tick beyond the swing low
        let b = bracket(&c, Side::Long, 2889, 2882).unwrap();
        assert_eq!((b.stop, b.d, b.tp), (2881, 8, 2905));
        let b = bracket(&c, Side::Short, 2891, 2893).unwrap();
        assert_eq!(b, Bracket { entry: 2891, stop: 2895, tp: 2883, d: 4 });
        assert!(bracket(&c, Side::Long, 2889, 2870).is_err(), "19 ticks > max 12: skip");
        c.reward_r = 3;
        let b = bracket(&c, Side::Short, 2891, 2896).unwrap();
        assert_eq!((b.stop, b.d, b.tp), (2897, 6, 2891 - 18));
        // tick rounding at the boundary and exact prices
        assert_eq!(to_ticks(0.02889, 0.00001), 2889);
        assert_eq!(to_ticks(0.028889999999, 0.00001), 2889);
        // R unit: a plain stop-out is exactly -1 R under the shadow model
        let c = cfg();
        let b = bracket(&c, Side::Long, 2889, 2887).unwrap();
        let q = 692.0;
        let r = r_unit_usd(&c, q, &b, 0.00001);
        let (e, x) = (0.02889, (b.stop - c.stop_slip_ticks) as f64 * 0.00001);
        let net = (x - e) * q - e * q * 2e-4 - x * q * 5.5e-4;
        assert!((net / r + 1.0).abs() < 2e-3, "stop-out = -1 R, got {}", net / r);
    }

    #[test]
    fn sizing_respects_the_cap_and_venue_minimums() {
        assert_eq!(size_qty(20.0, 0.0289, 0.1, 0.1, 5.0), Some(692.0));
        let q = size_qty(20.0, 0.0289, 0.1, 0.1, 5.0).unwrap();
        assert!(q * 0.0289 <= 20.0);
        assert_eq!(size_qty(4.0, 0.0289, 0.1, 0.1, 5.0), None, "below the 5 USD minimum notional");
    }

    #[test]
    fn funding_rule() {
        let c = cfg();
        let t = 1_000 * 3_600_000;
        assert_eq!(funding_action(&c, Side::Short, t - 6 * 60_000, t, 5.0), FundingAct::None);
        assert_eq!(funding_action(&c, Side::Short, t - 4 * 60_000, t, 5.0), FundingAct::Flatten, "shorts never cross");
        assert_eq!(funding_action(&c, Side::Long, t - 4 * 60_000, t, 0.5), FundingAct::Flatten);
        assert_eq!(funding_action(&c, Side::Long, t - 4 * 60_000, t, 1.0), FundingAct::HoldLong, ">= 1 R in profit may cross");
        assert!(entry_blocked_by_funding(&c, Side::Short, t - 10 * 60_000, t));
        assert!(!entry_blocked_by_funding(&c, Side::Short, t - 12 * 60_000, t));
        assert!(entry_blocked_by_funding(&c, Side::Long, t - 5 * 60_000, t));
        assert!(!entry_blocked_by_funding(&c, Side::Long, t - 7 * 60_000, t));
        let h8 = 8 * 3_600_000;
        assert_eq!(next_funding_on_grid(h8 * 3 + 1, h8), h8 * 4);
        assert_eq!(next_funding_on_grid(h8 * 3, h8), h8 * 4, "at the funding instant the next one is 8 h later");
    }

    #[test]
    fn queue_fill_model() {
        let mut p = Passive { buy: true, px: 2889, qty: 692.0, ahead: 3000.0, traded_at: 0.0 };
        assert!(!p.on_trade(2889, 3000.0, true), "buyers lifting the ask never fill our bid");
        assert!(!p.on_trade(2889, 3000.0, false), "queue ahead consumed, not yet our qty");
        assert!(!p.on_trade(2889, 692.0, false), "exactly ahead + qty is not MORE than it");
        assert!(p.on_trade(2889, 1.0, false));
        let mut p = Passive { buy: true, px: 2889, qty: 692.0, ahead: 1e9, traded_at: 0.0 };
        assert!(p.on_trade(2888, 0.1, false), "a print through our price fills");
        let mut s = Passive { buy: false, px: 2891, qty: 1.0, ahead: 0.0, traded_at: 0.0 };
        assert!(s.on_trade(2892, 0.1, true));
        assert_eq!(stop_fill(Side::Long, 2885, 1, 2885), Some(2884));
        assert_eq!(stop_fill(Side::Long, 2885, 1, 2880), Some(2880), "gap: the print, not the stop");
        assert_eq!(stop_fill(Side::Long, 2885, 1, 2886), None);
        assert_eq!(stop_fill(Side::Short, 2895, 1, 2895), Some(2896));
        assert!(!tp_hit(Side::Long, 2897, 2897) && tp_hit(Side::Long, 2897, 2898), "TP needs a trade-through");
        assert!(tp_hit(Side::Short, 2883, 2882));
    }

    #[test]
    fn sub_limits_tighter_only() {
        let c = cfg();
        assert!(c.limits.notional_cap_usd <= 20.0);
        c.limits.validate_against_live(&live()).unwrap();
        let mut l = c.limits.clone();
        l.notional_cap_usd = 20.01;
        assert!(l.validate().is_err(), "notional cap above 20 is refused");
        let mut l = c.limits.clone();
        l.day_loss_stop_usd = 3.5;
        assert!(l.validate().is_err());
        let mut l = c.limits.clone();
        l.max_consecutive_losses = 5;
        assert!(l.validate().is_err());
        let mut l = c.limits.clone();
        l.max_entries_per_run = 21;
        assert!(l.validate().is_err());
        let mut lv = live();
        lv.max_entries_per_day = 10;
        assert!(c.limits.validate_against_live(&lv).is_err(), "the scalper may never be looser than live_small");
        let mut lv = live();
        lv.daily_loss_stop_usd = 2.0;
        assert!(c.limits.validate_against_live(&lv).is_err());
    }

    #[test]
    fn strategy_hash_ignores_the_live_section() {
        let c = cfg();
        assert!(c.live.is_some(), "config/scalp_kava.yaml carries the live section");
        let mut bare = c.clone();
        bare.live = None;
        assert_eq!(c.hash(), bare.hash());
        assert_eq!(c.hash(), "17416371082833e8", "H-SCALP1 strategy hash as registered and logged by the shadow run");
    }

    fn snapshot(e: &mut Engine, now: i64, bid: &str, bq: &str, ask: &str, aq: &str) {
        e.on_book(now, "snapshot", &json!({"b": [[bid, bq], ["0.02880", "5000"]], "a": [[ask, aq], ["0.02900", "5000"]], "u": 1}));
    }

    #[test]
    fn engine_end_to_end_long_tp() {
        let mut c = cfg();
        c.z_min_obs = 30;
        c.z_window_s = 60;
        let mut e = Engine::new(c, inst(), "shadow");
        let t0 = 1_800_000_000_000_i64 + 3_600_000; // well away from a funding time
        snapshot(&mut e, t0, "0.02889", "3000", "0.02891", "1000");
        // warm-up: alternating small OFI so the sd is positive
        for s in 0..40 {
            let now = t0 + s * 1000;
            let q = if s % 2 == 0 { "3100" } else { "3000" };
            e.on_book(now + 1, "delta", &json!({"b": [["0.02889", q]], "a": [], "u": 2}));
            e.on_second(now + 999);
        }
        // a burst of bid adds and taker buying -> long signal
        let now = t0 + 41_000;
        for k in 0..20 {
            e.on_book(now + k, "delta", &json!({"b": [["0.02889", format!("{}", 3000 + 500 * (k + 1))]], "a": [], "u": 3}));
        }
        e.on_trades(now + 50, &json!([{"T": now, "S": "Buy", "p": "0.02891", "v": "100"}]));
        let evs = e.on_second(now + 999);
        assert!(evs.iter().any(|v| v["type"] == "order" && v["leg"] == "entry"), "entry order placed: {evs:?}");
        // fill: a print through our bid
        let evs = e.on_trades(now + 2000, &json!([{"T": now + 2000, "S": "Sell", "p": "0.02888", "v": "1"}]));
        assert!(evs.iter().any(|v| v["type"] == "fill"));
        // a print AT the tp does nothing; through it -> tp exit, maker
        let Phase::Open(o) = e.phase.clone() else { panic!("open") };
        let tp_px = format!("{:.5}", o.br.tp as f64 * 0.00001);
        assert!(e.on_trades(now + 3000, &json!([{"T": now + 3000, "S": "Buy", "p": tp_px, "v": "1"}])).is_empty());
        let through = format!("{:.5}", (o.br.tp + 1) as f64 * 0.00001);
        let evs = e.on_trades(now + 4000, &json!([{"T": now + 4000, "S": "Buy", "p": through, "v": "1"}]));
        let x = evs.iter().find(|v| v["type"] == "exit").expect("tp exit");
        assert_eq!(x["kind"], "tp");
        // 2 R in price nets < 1 R once the R unit carries the stop slip and both fees (4-tick stop: about +0.96 R)
        let r = x["net_r"].as_f64().unwrap();
        assert!(r > 0.9 && r < 1.0, "2R gross minus fees: {x}");
        assert_eq!(e.stats.trades, 1);
        assert_eq!(e.stats.tp_hits, 1);
    }
}
