//! `atx tape`: deterministic, READ-ONLY participant labeller on Bybit public market data (no keys, no orders,
//! no balances). Config: `config/tape.yaml` (every threshold lives there; its hash is logged).
//!
//!   atx tape --symbols BTCUSDT,ETHUSDT --window-s 60 --log logs/tape.log --journal data/journal/tape/{date}.jsonl
//!
//! One websocket carries, per symbol: `orderbook.1` (L1 flicker, iceberg episodes), `orderbook.50` (cancels near the
//! touch), `orderbook.<deep>` (walls, spoofs, liquidity pools), `publicTrade` and `allLiquidation`. Footprints:
//!
//! 1. HFT_MM: L1 update rate and flicker share (L1 levels that leave within `flicker_ms` without a print at their
//!    price), cancel-to-trade near the touch; labelled when rate AND flicker > their rolling 30-min percentile.
//! 2. SPOOF: a wall (>= k x median level size within the band) pulled, not traded, while price was within N ticks and
//!    closer than at its birth; follow-up: did mid move away from the wall's side.
//! 3. ALGO_BUY / ALGO_SELL: >= n same-side clips of one size (within tol) with inter-arrival CV < max.
//! 4. ICEBERG_BID / ICEBERG_ASK: executed at a touch price > m x its largest displayed size, with refills.
//! 5. SWEEP (multi-level / burst), LIQ_CASCADE (allLiquidation clusters).
//! 6. Taker delta by clip-notional bucket (small/medium/large/whale from the rolling clip distribution).
//! 7. PUSH: votes of the large+whale delta, algo side, defence (icebergs + consumed walls) and spoofs (a pulled wall
//!    points AGAINST the intended direction), with the next liquidity pool, defended level and swept level.
//!
//! These are descriptive labels, NOT tested signals: every output line says so.

use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::io::Write as _;
use std::path::{Path, PathBuf};
use std::time::Duration;

use anyhow::{Context, bail, ensure};
use at_bybit::market::MarketData;
use at_bybit::ws::{self, WsMessage};
use at_core::time::now_ms;
use at_flow::book::Book;
use at_flow::collector::parse_liquidations;
use chrono::{SecondsFormat, Utc};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use crate::cmd::Ctx;
use crate::scalp::{RollingZ, SecSums, to_ticks};

pub const TAPE_VERSION: &str = "tape-v1";
pub const NOTE: &str = "descriptive label, not a tested signal";

// ---------------------------------------------------------------- config

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct HftCfg {
    pub flicker_ms: i64,
    pub trade_pad_ms: i64,
    pub near_touch_ticks: i64,
    pub sample_every_s: i64,
    pub history_min: i64,
    pub pctile: f64,
    pub min_history: usize,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SpoofCfg {
    pub size_mult: f64,
    pub band_bps: f64,
    pub end_frac: f64,
    pub traded_frac: f64,
    pub approach_ticks: i64,
    pub min_life_ms: i64,
    pub follow_s: i64,
    pub max_walls: usize,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AlgoCfg {
    pub qty_tol: f64,
    pub min_clips: usize,
    pub max_cv: f64,
    pub min_cadence_ms: f64,
    /// Clips below this many x the rolling p50 clip notional are ignored (round-lot retail is not slicing).
    pub min_clip_p50_mult: f64,
    pub eval_every_s: i64,
    pub stop_after_s: i64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct IceCfg {
    pub exec_mult: f64,
    pub min_refills: u32,
    pub alert_usd_default: f64,
    pub alert_usd: BTreeMap<String, f64>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SweepCfg {
    pub min_levels: usize,
    pub burst_ms: i64,
    pub burst_min_ticks: i64,
    pub burst_p99_mult: f64,
    pub alert_median_mult: f64,
    /// ... and the sweep notional must also exceed this many x the rolling p99 clip notional.
    pub alert_min_p99_mult: f64,
    pub alert_min_history: usize,
    pub history: usize,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LiqCfg {
    pub cascade_n: usize,
    pub cascade_s: i64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BucketCfg {
    pub history: usize,
    pub min_obs: usize,
    pub recompute_every_s: i64,
    pub p_small: f64,
    pub p_large: f64,
    pub p_whale: f64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PoolCfg {
    pub min_ticks: i64,
    pub max_pct: f64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PushCfg {
    pub big_min_imbalance: f64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TapeCfg {
    pub window_s: i64,
    pub l1_depth: u32,
    pub near_depth: u32,
    pub deep_depth_default: u32,
    pub deep_depth: BTreeMap<String, u32>,
    pub alert_cooldown_ms: i64,
    pub hft: HftCfg,
    pub spoof: SpoofCfg,
    pub algo: AlgoCfg,
    pub iceberg: IceCfg,
    pub sweep: SweepCfg,
    pub liq: LiqCfg,
    pub buckets: BucketCfg,
    pub pool: PoolCfg,
    pub push: PushCfg,
}

impl TapeCfg {
    pub fn load(path: &Path) -> anyhow::Result<Self> {
        let text = std::fs::read_to_string(path).with_context(|| path.display().to_string())?;
        let c: Self = serde_yaml::from_str(&text).with_context(|| path.display().to_string())?;
        c.validate().with_context(|| path.display().to_string())?;
        Ok(c)
    }

    pub fn validate(&self) -> anyhow::Result<()> {
        ensure!((10..=3600).contains(&self.window_s), "window_s in [10, 3600]");
        ensure!(self.l1_depth == 1 && self.near_depth > 1 && self.deep_depth_default >= self.near_depth, "depths: l1 = 1 < near <= deep");
        ensure!(self.hft.flicker_ms > 0 && self.hft.sample_every_s > 0 && self.hft.history_min > 0 && (0.5..1.0).contains(&self.hft.pctile), "hft settings");
        ensure!(self.spoof.size_mult > 1.0 && self.spoof.band_bps > 0.0 && (0.0..1.0).contains(&self.spoof.end_frac) && self.spoof.approach_ticks >= 0, "spoof settings");
        ensure!(self.algo.min_clips >= 3 && self.algo.max_cv > 0.0 && self.algo.qty_tol >= 0.0 && self.algo.eval_every_s > 0, "algo settings");
        ensure!(self.iceberg.exec_mult > 1.0, "iceberg.exec_mult > 1");
        ensure!(self.buckets.recompute_every_s > 0 && self.alert_cooldown_ms >= 0, "buckets.recompute_every_s > 0, alert_cooldown_ms >= 0");
        ensure!(self.sweep.min_levels >= 2 && self.sweep.burst_ms > 0 && self.sweep.history >= self.sweep.alert_min_history, "sweep settings");
        ensure!(self.liq.cascade_n >= 2 && self.liq.cascade_s > 0, "liq settings");
        let b = &self.buckets;
        ensure!(b.p_small < b.p_large && b.p_large < b.p_whale && b.p_whale < 1.0 && b.history >= b.min_obs, "bucket percentiles");
        Ok(())
    }

    pub fn deep_for(&self, sym: &str) -> u32 {
        self.deep_depth.get(sym).copied().unwrap_or(self.deep_depth_default)
    }

    pub fn hash(&self) -> String {
        let s = serde_json::to_string(self).unwrap_or_default();
        let h = s.bytes().fold(0xcbf2_9ce4_8422_2325_u64, |h, b| (h ^ b as u64).wrapping_mul(0x0100_0000_01b3));
        format!("{h:016x}")
    }
}

// ---------------------------------------------------------------- small pure helpers

/// Nearest-rank quantile of an unsorted sample (copy + sort). None below `min_obs`.
pub fn quantile(xs: impl Iterator<Item = f64>, p: f64, min_obs: usize) -> Option<f64> {
    let mut v: Vec<f64> = xs.filter(|x| x.is_finite()).collect();
    if v.len() < min_obs.max(1) {
        return None;
    }
    v.sort_by(f64::total_cmp);
    Some(v[((v.len() - 1) as f64 * p).round() as usize])
}

/// Bounded FIFO of samples.
#[derive(Clone, Debug, Default)]
pub struct Roll {
    pub q: VecDeque<f64>,
    cap: usize,
}

impl Roll {
    pub fn new(cap: usize) -> Self {
        Self { q: VecDeque::new(), cap }
    }

    pub fn push(&mut self, x: f64) {
        self.q.push_back(x);
        while self.q.len() > self.cap {
            self.q.pop_front();
        }
    }

    pub fn quantile(&self, p: f64, min_obs: usize) -> Option<f64> {
        quantile(self.q.iter().copied(), p, min_obs)
    }
}

/// Bucket of a clip notional against [p_small, p_large, p_whale] cuts: 0 small, 1 medium, 2 large, 3 whale.
pub fn bucket(cuts: &[f64; 3], usd: f64) -> usize {
    if usd < cuts[0] {
        0
    } else if usd <= cuts[1] {
        1
    } else if usd <= cuts[2] {
        2
    } else {
        3
    }
}

pub fn fmt_usd(x: f64) -> String {
    let (s, a) = (if x < 0.0 { "-" } else { "" }, x.abs());
    if a >= 1e6 {
        format!("{s}${:.2}M", a / 1e6)
    } else if a >= 1e3 {
        format!("{s}${:.0}k", a / 1e3)
    } else {
        format!("{s}${a:.0}")
    }
}

fn num(v: &Value) -> Option<f64> {
    v.as_str().and_then(|s| s.parse().ok()).or_else(|| v.as_f64())
}

fn levels(v: &Value) -> Vec<(f64, f64)> {
    v.as_array()
        .into_iter()
        .flatten()
        .filter_map(|lv| Some((lv.get(0)?.as_str()?.parse().ok()?, lv.get(1)?.as_str()?.parse().ok()?)))
        .collect()
}

// ---------------------------------------------------------------- 1. L1 flicker

/// One L1 episode: a (side, price) that was the touch from `born` to `end` (local ms).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Ep {
    pub bid: bool,
    pub px: i64,
    pub born: i64,
    pub end: i64,
}

#[derive(Clone, Debug)]
pub struct L1Track {
    prev: Option<(i64, f64, i64, f64)>,
    bid: Option<(i64, i64)>,
    ask: Option<(i64, i64)>,
    pending: VecDeque<Ep>,
    pub upd: SecSums,
    pub ends: SecSums,
    pub flicks: SecSums,
}

impl L1Track {
    pub fn new(horizon_s: i64) -> Self {
        Self { prev: None, bid: None, ask: None, pending: VecDeque::new(), upd: SecSums::new(horizon_s), ends: SecSums::new(horizon_s), flicks: SecSums::new(horizon_s) }
    }

    pub fn reset(&mut self) {
        self.prev = None;
        self.bid = None;
        self.ask = None;
        self.pending.clear();
    }

    /// One L1 state (ticks, sizes). Counts a change of price or size on either side; closes episodes whose price left.
    pub fn on_l1(&mut self, now: i64, b: i64, bq: f64, a: i64, aq: f64) {
        let cur = (b, bq, a, aq);
        if self.prev.is_some_and(|p| p != cur) {
            self.upd.add(now.div_euclid(1000), 1.0);
        }
        self.prev = Some(cur);
        for (side, px) in [(true, b), (false, a)] {
            let slot = if side { &mut self.bid } else { &mut self.ask };
            match *slot {
                Some((p, _)) if p == px => {}
                Some((p, born)) => {
                    self.pending.push_back(Ep { bid: side, px: p, born, end: now });
                    *slot = Some((px, now));
                }
                None => *slot = Some((px, now)),
            }
        }
        while self.pending.len() > 20_000 {
            self.pending.pop_front();
        }
    }

    /// Classify episodes that ended >= `settle_ms` ago: flicker = lived < `flicker_ms` with no print at its price
    /// between born - 200 ms and end + `pad_ms`. `trades` = (local ms, price ticks).
    pub fn classify(&mut self, now: i64, trades: &VecDeque<(i64, i64)>, flicker_ms: i64, pad_ms: i64) {
        let settle = pad_ms + 200;
        let sec = now.div_euclid(1000);
        while let Some(e) = self.pending.front().copied()
            && e.end <= now - settle
        {
            self.pending.pop_front();
            let traded = trades.iter().any(|&(ms, px)| px == e.px && ms >= e.born - 200 && ms <= e.end + pad_ms);
            self.ends.add(sec, 1.0);
            if e.end - e.born < flicker_ms && !traded {
                self.flicks.add(sec, 1.0);
            }
        }
    }
}

// ---------------------------------------------------------------- 2. walls and spoofs

#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
pub struct Wall {
    pub bid: bool,
    pub px: i64,
    pub born: i64,
    pub peak: f64,
    pub traded: f64,
    pub born_dist: i64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum WallEndKind {
    Spoof,
    Pulled,
    Consumed,
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
pub struct WallEnd {
    pub kind: WallEndKind,
    pub bid: bool,
    pub px: i64,
    pub peak: f64,
    pub traded: f64,
    pub life_ms: i64,
    pub dist_ticks: i64,
    pub born_dist: i64,
    pub ms: i64,
}

/// Touch and mid in ticks.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Touch {
    pub bid: i64,
    pub ask: i64,
    pub mid: f64,
}

/// One book level update (ticks, new size).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Lvl {
    pub bid: bool,
    pub px: i64,
    pub q: f64,
}

#[derive(Clone, Debug, Default)]
pub struct Walls {
    pub map: BTreeMap<(bool, i64), Wall>,
}

impl Walls {
    pub fn on_level(&mut self, c: &SpoofCfg, now: i64, lv: Lvl, median: f64, t: Touch) -> Option<WallEnd> {
        let dist = if lv.bid { t.bid - lv.px } else { lv.px - t.ask };
        let key = (lv.bid, lv.px);
        if let Some(w) = self.map.get_mut(&key) {
            if lv.q > w.peak {
                w.peak = lv.q;
            }
            if lv.q > 0.0 && lv.q >= c.end_frac * w.peak {
                return None;
            }
            let w = self.map.remove(&key)?;
            let removed = w.peak - lv.q;
            let life_ms = now - w.born;
            let kind = if w.traded >= c.traded_frac * removed {
                WallEndKind::Consumed
            } else if (0..=c.approach_ticks).contains(&dist) && dist < w.born_dist && life_ms >= c.min_life_ms {
                WallEndKind::Spoof
            } else {
                WallEndKind::Pulled
            };
            return Some(WallEnd { kind, bid: lv.bid, px: lv.px, peak: w.peak, traded: w.traded, life_ms, dist_ticks: dist, born_dist: w.born_dist, ms: now });
        }
        let in_band = t.mid > 0.0 && (lv.px as f64 - t.mid).abs() / t.mid * 1e4 <= c.band_bps;
        if median > 0.0 && lv.q >= c.size_mult * median && in_band && dist >= 0 && self.map.len() < c.max_walls {
            self.map.insert(key, Wall { bid: lv.bid, px: lv.px, born: now, peak: lv.q, traded: 0.0, born_dist: dist });
        }
        None
    }

    /// An aggressive print: sells hit bid walls, buys lift ask walls at that price.
    pub fn on_trade(&mut self, px: i64, vol: f64, aggressor_buy: bool) {
        if let Some(w) = self.map.get_mut(&(!aggressor_buy, px)) {
            w.traded += vol;
        }
    }

    /// Forget walls that drifted far outside the band (price ran away from them).
    pub fn prune(&mut self, mid: f64, band_bps: f64) {
        self.map.retain(|_, w| mid <= 0.0 || (w.px as f64 - mid).abs() / mid * 1e4 <= 2.0 * band_bps);
    }
}

// ---------------------------------------------------------------- 3. execution algos

/// A taker clip: trades with one aggressor side and one exchange time T, summed.
#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
pub struct Clip {
    pub ms: i64,
    pub ex_ms: i64,
    pub buy: bool,
    pub qty: f64,
    pub usd: f64,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct Algo {
    pub buy: bool,
    pub clip_qty: f64,
    pub clips: usize,
    pub cadence_s: f64,
    pub cv: f64,
    pub notional: f64,
    pub first_ms: i64,
    pub last_ms: i64,
}

impl Algo {
    pub fn key(&self) -> String {
        format!("{}:{:.3e}", if self.buy { "B" } else { "S" }, self.clip_qty)
    }
}

/// Clusters of same-side clips whose size is within `tol` (relative) of the cluster's smallest; a cluster with
/// >= `min_clips` clips whose exchange-time gaps have CV < `max_cv` and mean >= `min_cadence_ms` is an algo.
pub fn detect_algos(clips: &[Clip], tol: f64, min_clips: usize, max_cv: f64, min_cadence_ms: f64) -> Vec<Algo> {
    let mut out = Vec::new();
    for side in [true, false] {
        let mut v: Vec<&Clip> = clips.iter().filter(|c| c.buy == side && c.qty > 0.0).collect();
        v.sort_by(|a, b| a.qty.total_cmp(&b.qty));
        let mut i = 0;
        while i < v.len() {
            let q0 = v[i].qty;
            let mut j = i;
            while j < v.len() && v[j].qty <= q0 * (1.0 + tol) + 1e-12 {
                j += 1;
            }
            let g = &v[i..j];
            if g.len() >= min_clips {
                let mut ts: Vec<i64> = g.iter().map(|c| c.ex_ms).collect();
                ts.sort_unstable();
                let gaps: Vec<f64> = ts.windows(2).map(|w| (w[1] - w[0]) as f64).collect();
                let n = gaps.len() as f64;
                let mean = gaps.iter().sum::<f64>() / n;
                let sd = (gaps.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1.0).max(1.0)).sqrt();
                let cv = if mean > 0.0 { sd / mean } else { f64::INFINITY };
                if mean >= min_cadence_ms && cv < max_cv {
                    let mut qs: Vec<f64> = g.iter().map(|c| c.qty).collect();
                    qs.sort_by(f64::total_cmp);
                    out.push(Algo {
                        buy: side,
                        clip_qty: qs[qs.len() / 2],
                        clips: g.len(),
                        cadence_s: mean / 1000.0,
                        cv,
                        notional: g.iter().map(|c| c.usd).sum(),
                        first_ms: ts[0],
                        last_ms: *ts.last().unwrap_or(&ts[0]),
                    });
                }
            }
            i = j;
        }
    }
    out
}

// ---------------------------------------------------------------- 4. icebergs

/// One L1 episode on one side, followed for hidden size: executed volume vs displayed, refills after hits.
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize)]
pub struct Ice {
    pub bid: bool,
    pub active: bool,
    pub px: i64,
    pub born: i64,
    pub disp: f64,
    pub max_disp: f64,
    pub exec: f64,
    pub refills: u32,
    hit: bool,
    pub recorded: bool,
    pub alerted: bool,
}

impl Ice {
    pub fn new(bid: bool) -> Self {
        Self { bid, ..Default::default() }
    }

    /// L1 state on this side. Returns the finished episode when the touch price changed.
    pub fn on_l1(&mut self, now: i64, px: i64, qty: f64) -> Option<Ice> {
        if !self.active || px != self.px {
            let bid = self.bid;
            let old = std::mem::replace(self, Ice { bid, active: true, px, born: now, disp: qty, max_disp: qty, ..Default::default() });
            return old.active.then_some(old);
        }
        if qty > self.disp && self.hit {
            self.refills += 1;
            self.hit = false;
        }
        self.disp = qty;
        self.max_disp = self.max_disp.max(qty);
        None
    }

    /// An aggressive print: sells hit the bid episode, buys lift the ask episode, at its price.
    pub fn on_trade(&mut self, px: i64, vol: f64, aggressor_buy: bool) -> bool {
        if self.active && px == self.px && aggressor_buy != self.bid {
            self.exec += vol;
            self.hit = true;
            return true;
        }
        false
    }

    pub fn qualifies(&self, mult: f64, min_refills: u32) -> bool {
        self.max_disp > 0.0 && self.exec > mult * self.max_disp && self.refills >= min_refills
    }
}

// ---------------------------------------------------------------- 5. sweeps and liquidations

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct Sweep {
    pub kind: &'static str,
    pub ms: i64,
    pub ex_ms: i64,
    pub buy: bool,
    pub levels: usize,
    pub from_px: i64,
    /// Where the aggressor stopped (buy: highest print, sell: lowest): the stops taken up to here.
    pub to_px: i64,
    pub qty: f64,
    pub usd: f64,
}

/// One public trade (local ms, exchange ms, price ticks, aggressor buy?, size, notional USD).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TTr {
    pub ms: i64,
    pub ex_ms: i64,
    pub px: i64,
    pub buy: bool,
    pub qty: f64,
    pub usd: f64,
}

/// Multi-level sweeps in a batch: same aggressor side and exchange time at >= `min_levels` distinct prices.
pub fn multi_level_sweeps(trades: &[TTr], min_levels: usize) -> Vec<Sweep> {
    let mut g: BTreeMap<(i64, bool), Vec<&TTr>> = BTreeMap::new();
    for t in trades {
        g.entry((t.ex_ms, t.buy)).or_default().push(t);
    }
    g.into_iter()
        .filter_map(|((ex_ms, buy), v)| {
            let prices: BTreeSet<i64> = v.iter().map(|t| t.px).collect();
            if prices.len() < min_levels {
                return None;
            }
            let to_px = if buy { *prices.last()? } else { *prices.first()? };
            Some(Sweep { kind: "multi_level", ms: v[0].ms, ex_ms, buy, levels: prices.len(), from_px: v[0].px, to_px, qty: v.iter().map(|t| t.qty).sum(), usd: v.iter().map(|t| t.usd).sum() })
        })
        .collect()
}

/// Bursts: same-side notional over `burst_ms` (exchange time) above `min_usd` while price moved >= `min_ticks` the
/// aggressor's way. One burst per side per `burst_ms`.
#[derive(Clone, Debug, Default)]
pub struct Burst {
    q: [VecDeque<TTr>; 2],
    cool: [i64; 2],
}

impl Burst {
    pub fn reset(&mut self) {
        *self = Self::default();
    }

    pub fn on_trade(&mut self, t: &TTr, burst_ms: i64, min_ticks: i64, min_usd: f64) -> Option<Sweep> {
        let s = usize::from(!t.buy);
        let q = &mut self.q[s];
        q.push_back(*t);
        while q.front().is_some_and(|f| f.ex_ms < t.ex_ms - burst_ms) {
            q.pop_front();
        }
        let first = q.front()?.px;
        let moved = if t.buy { t.px - first } else { first - t.px };
        let usd: f64 = q.iter().map(|x| x.usd).sum();
        if min_usd > 0.0 && usd > min_usd && moved >= min_ticks && t.ex_ms >= self.cool[s] {
            self.cool[s] = t.ex_ms + burst_ms;
            let levels = q.iter().map(|x| x.px).collect::<BTreeSet<_>>().len();
            let qty = q.iter().map(|x| x.qty).sum();
            q.clear();
            return Some(Sweep { kind: "burst", ms: t.ms, ex_ms: t.ex_ms, buy: t.buy, levels, from_px: first, to_px: t.px, qty, usd });
        }
        None
    }
}

/// Liquidations of one side (index 0 = longs liquidated = forced sells, 1 = shorts = forced buys) over `win_ms`.
#[derive(Clone, Debug, Default)]
pub struct LiqClusters {
    q: [VecDeque<(i64, f64, f64)>; 2],
    on: [bool; 2],
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct Cascade {
    pub long_liquidated: bool,
    pub n: usize,
    pub usd: f64,
    pub lo_px: f64,
    pub hi_px: f64,
}

impl LiqClusters {
    /// Returns a cascade the first time the count reaches `n` (re-armed when it falls below `n`).
    pub fn on_liq(&mut self, now: i64, long: bool, px: f64, usd: f64, n: usize, win_ms: i64) -> Option<Cascade> {
        let s = usize::from(!long);
        let q = &mut self.q[s];
        q.push_back((now, px, usd));
        while q.front().is_some_and(|f| f.0 < now - win_ms) {
            q.pop_front();
        }
        if q.len() < n {
            self.on[s] = false;
            return None;
        }
        if self.on[s] {
            return None;
        }
        self.on[s] = true;
        let lo = q.iter().map(|x| x.1).fold(f64::INFINITY, f64::min);
        let hi = q.iter().map(|x| x.1).fold(f64::NEG_INFINITY, f64::max);
        Some(Cascade { long_liquidated: long, n: q.len(), usd: q.iter().map(|x| x.2).sum(), lo_px: lo, hi_px: hi })
    }

    pub fn tick(&mut self, now: i64, n: usize, win_ms: i64) {
        for s in [0usize, 1] {
            while self.q[s].front().is_some_and(|f| f.0 < now - win_ms) {
                self.q[s].pop_front();
            }
            if self.q[s].len() < n {
                self.on[s] = false;
            }
        }
    }
}

// ---------------------------------------------------------------- 7. push summary and pools

/// Largest resting level more than `min_ticks` from the touch and within `max_pct` % of mid. Levels best first.
pub fn pool(levels: &[(i64, f64)], bid: bool, t: Touch, min_ticks: i64, max_pct: f64) -> Option<(i64, f64)> {
    levels
        .iter()
        .filter(|(p, _)| {
            let d = if bid { t.bid - p } else { p - t.ask };
            d > min_ticks && t.mid > 0.0 && (*p as f64 - t.mid).abs() / t.mid * 100.0 <= max_pct
        })
        .max_by(|a, b| a.1.total_cmp(&b.1))
        .copied()
}

/// Votes (signed strengths; 0 / non-finite = footprint inactive) -> (side -1/0/+1, agreeing, active).
pub fn push_vote(votes: &[(&str, f64)]) -> (i8, usize, usize) {
    let active: Vec<f64> = votes.iter().map(|v| v.1).filter(|x| x.is_finite() && *x != 0.0).collect();
    let sum: i64 = active.iter().map(|x| x.signum() as i64).sum();
    let side = sum.signum() as i8;
    let agree = if side == 0 { 0 } else { active.iter().filter(|x| x.signum() as i8 == side).count() };
    (side, agree, active.len())
}

/// Large + whale taker delta vote: counts only when imbalanced by >= `min_imb` of its gross.
pub fn big_vote(delta: &[f64; 4], gross: &[f64; 4], min_imb: f64) -> f64 {
    let d = delta[2] + delta[3];
    let g = gross[2] + gross[3];
    if g > 0.0 && d.abs() >= min_imb * g { d } else { 0.0 }
}

// ---------------------------------------------------------------- per-symbol engine

#[derive(Clone, Debug, Serialize)]
pub struct Alert {
    pub sym: String,
    pub kind: String,
    pub side: String,
    pub text: String,
    pub data: Value,
}

#[derive(Clone, Debug, Serialize)]
struct SpoofRec {
    end: WallEnd,
    px: f64,
    usd: f64,
    mid_at: f64,
    due: i64,
    follow_bps: Option<f64>,
    moved_away: Option<bool>,
}

#[derive(Clone, Debug, Serialize)]
struct IceRec {
    ms: i64,
    born: i64,
    bid: bool,
    px: i64,
    exec: f64,
    max_disp: f64,
    refills: u32,
    usd: f64,
    active: bool,
}

pub struct Sym {
    pub sym: String,
    pub tick: f64,
    dec: usize,
    deep_depth: u32,
    l1book: Book,
    near: Book,
    deep: Book,
    l1: L1Track,
    canc: SecSums,
    tvol: SecSums,
    recent: VecDeque<(i64, i64)>,
    clips: VecDeque<Clip>,
    dist: Roll,
    cuts: Option<[f64; 3]>,
    bdelta: [SecSums; 4],
    bgross: [SecSums; 4],
    unbucketed: SecSums,
    walls: Walls,
    median: f64,
    median_ms: i64,
    ice: [Ice; 2],
    burst: Burst,
    sweep_hist: Roll,
    liq: LiqClusters,
    liq_usd: [SecSums; 2],
    liq_n: [SecSums; 2],
    spoofs: VecDeque<SpoofRec>,
    consumed: VecDeque<WallEnd>,
    pulled_n: SecSums,
    icebergs: VecDeque<IceRec>,
    sweeps: VecDeque<Sweep>,
    algos: Vec<Algo>,
    algo_on: BTreeMap<String, (i64, Algo)>,
    rate_hist: Roll,
    flick_hist: Roll,
    mids: VecDeque<(i64, f64)>,
    ret1s: RollingZ,
    last_mid_sec: Option<(i64, f64)>,
    pub resets: u64,
    pub msgs: u64,
}

fn alert(sym: &str, kind: &str, side: &str, text: String, data: Value) -> Alert {
    Alert { sym: sym.to_string(), kind: kind.to_string(), side: side.to_string(), text, data }
}

impl Sym {
    pub fn new(cfg: &TapeCfg, sym: &str, tick: f64) -> Self {
        let h = cfg.window_s + 5;
        let dec = if tick > 0.0 { (-tick.log10()).ceil().max(0.0) as usize } else { 4 };
        let samples = (cfg.hft.history_min * 60 / cfg.hft.sample_every_s).max(1) as usize;
        let ss = || SecSums::new(h);
        Self {
            sym: sym.to_string(),
            tick,
            dec,
            deep_depth: cfg.deep_for(sym),
            l1book: Book::default(),
            near: Book::default(),
            deep: Book::default(),
            l1: L1Track::new(h),
            canc: ss(),
            tvol: ss(),
            recent: VecDeque::new(),
            clips: VecDeque::new(),
            dist: Roll::new(cfg.buckets.history),
            cuts: None,
            bdelta: [ss(), ss(), ss(), ss()],
            bgross: [ss(), ss(), ss(), ss()],
            unbucketed: ss(),
            walls: Walls::default(),
            median: 0.0,
            median_ms: 0,
            ice: [Ice::new(true), Ice::new(false)],
            burst: Burst::default(),
            sweep_hist: Roll::new(cfg.sweep.history),
            liq: LiqClusters::default(),
            liq_usd: [ss(), ss()],
            liq_n: [ss(), ss()],
            spoofs: VecDeque::new(),
            consumed: VecDeque::new(),
            pulled_n: ss(),
            icebergs: VecDeque::new(),
            sweeps: VecDeque::new(),
            algos: Vec::new(),
            algo_on: BTreeMap::new(),
            rate_hist: Roll::new(samples),
            flick_hist: Roll::new(samples),
            mids: VecDeque::new(),
            ret1s: RollingZ::new(1800, 60),
            last_mid_sec: None,
            resets: 0,
            msgs: 0,
        }
    }

    fn t(&self, px: f64) -> i64 {
        to_ticks(px, self.tick)
    }

    fn p(&self, t: i64) -> String {
        format!("{:.*}", self.dec, t as f64 * self.tick)
    }

    fn usd_t(&self, t: i64, q: f64) -> f64 {
        t as f64 * self.tick * q
    }

    fn touch(&self) -> Option<Touch> {
        let l = self.l1book.l1().or_else(|| self.deep.l1())?;
        let (b, a) = (self.t(l.bid), self.t(l.ask));
        (b < a).then(|| Touch { bid: b, ask: a, mid: 0.5 * (b + a) as f64 })
    }

    pub fn on_reset(&mut self) {
        self.l1book.invalidate();
        self.near.invalidate();
        self.deep.invalidate();
        self.l1.reset();
        self.walls.map.clear();
        self.ice = [Ice::new(true), Ice::new(false)];
        self.burst.reset();
        self.resets += 1;
    }

    pub fn on_book(&mut self, cfg: &TapeCfg, depth: u32, now: i64, kind: &str, data: &Value) -> Vec<Alert> {
        self.msgs += 1;
        let mut out = Vec::new();
        if depth == cfg.l1_depth {
            if !self.l1book.apply(kind, data) {
                return out;
            }
            let Some(l) = self.l1book.l1() else { return out };
            let (b, a) = (self.t(l.bid), self.t(l.ask));
            if b >= a {
                return out;
            }
            self.l1.on_l1(now, b, l.bid_qty, a, l.ask_qty);
            for (k, (px, q)) in [(b, l.bid_qty), (a, l.ask_qty)].into_iter().enumerate() {
                if let Some(old) = self.ice[k].on_l1(now, px, q)
                    && old.recorded
                    && let Some(r) = self.icebergs.iter_mut().rev().find(|r| r.born == old.born && r.bid == old.bid && r.px == old.px)
                {
                    r.active = false;
                    r.ms = now;
                }
            }
        }
        if depth == cfg.near_depth {
            if kind == "delta"
                && self.near.valid
                && let Some(l) = self.near.l1()
            {
                let n = cfg.hft.near_touch_ticks as f64 * self.tick + 1e-12;
                let mut dec = 0.0;
                for (p, q) in levels(&data["b"]) {
                    let old = self.near.qty_at(true, p);
                    if p >= l.bid - n && q < old {
                        dec += old - q;
                    }
                }
                for (p, q) in levels(&data["a"]) {
                    let old = self.near.qty_at(false, p);
                    if p <= l.ask + n && q < old {
                        dec += old - q;
                    }
                }
                if dec > 0.0 {
                    self.canc.add(now.div_euclid(1000), dec);
                }
            }
            self.near.apply(kind, data);
        }
        if depth == self.deep_depth {
            if kind == "snapshot" {
                self.walls.map.clear();
            }
            if !self.deep.apply(kind, data) {
                return out;
            }
            let Some(t) = self.touch() else { return out };
            if now - self.median_ms >= 1000 {
                let band = cfg.spoof.band_bps;
                let tick = self.tick;
                let within = |p: f64| (p / tick - t.mid).abs() / t.mid * 1e4 <= band;
                let sizes = self.deep.bids_desc().take_while(|(p, _)| within(*p)).chain(self.deep.asks_asc().take_while(|(p, _)| within(*p))).map(|x| x.1);
                self.median = quantile(sizes, 0.5, 5).unwrap_or(0.0);
                self.median_ms = now;
                self.walls.prune(t.mid, band);
            }
            let lv: Vec<Lvl> = levels(&data["b"]).into_iter().map(|(p, q)| Lvl { bid: true, px: self.t(p), q }).chain(levels(&data["a"]).into_iter().map(|(p, q)| Lvl { bid: false, px: self.t(p), q })).collect();
            for l in lv {
                let Some(e) = self.walls.on_level(&cfg.spoof, now, l, self.median, t) else { continue };
                match e.kind {
                    WallEndKind::Spoof => {
                        let usd = self.usd_t(e.px, e.peak);
                        let side = if e.bid { "bid" } else { "ask" };
                        let text = format!(
                            "SPOOF_PULL {side} wall {} size {:.4} ({}) pulled after {:.1}s at {} ticks from touch (born {} ticks away), traded {:.4}; implies push {}",
                            self.p(e.px),
                            e.peak,
                            fmt_usd(usd),
                            e.life_ms as f64 / 1000.0,
                            e.dist_ticks,
                            e.born_dist,
                            e.traded,
                            if e.bid { "DOWN" } else { "UP" }
                        );
                        out.push(alert(&self.sym, "SPOOF_PULL", side, text, json!({"wall": e, "px": e.px as f64 * self.tick, "usd": usd, "median_level": self.median})));
                        self.spoofs.push_back(SpoofRec { end: e, px: e.px as f64 * self.tick, usd, mid_at: t.mid, due: now + cfg.spoof.follow_s * 1000, follow_bps: None, moved_away: None });
                    }
                    WallEndKind::Consumed => self.consumed.push_back(e),
                    WallEndKind::Pulled => self.pulled_n.add(now.div_euclid(1000), 1.0),
                }
            }
        }
        out
    }

    pub fn on_trades(&mut self, cfg: &TapeCfg, now: i64, data: &Value) -> Vec<Alert> {
        self.msgs += 1;
        let mut out = Vec::new();
        let tick = self.tick;
        let batch: Vec<TTr> = data
            .as_array()
            .into_iter()
            .flatten()
            .filter_map(|t| {
                let (ex, p, v) = (t["T"].as_i64()?, num(&t["p"])?, num(&t["v"])?);
                Some(TTr { ms: now, ex_ms: ex, px: to_ticks(p, tick), buy: t["S"].as_str() == Some("Buy"), qty: v, usd: p * v })
            })
            .collect();
        let sec = now.div_euclid(1000);
        let p99 = self.cuts.map(|c| c[2]).unwrap_or(0.0);
        let ice_alert = cfg.iceberg.alert_usd.get(&self.sym).copied().unwrap_or(cfg.iceberg.alert_usd_default);
        for t in &batch {
            self.recent.push_back((now, t.px));
            self.tvol.add(sec, t.qty);
            self.walls.on_trade(t.px, t.qty, t.buy);
            for k in [0usize, 1] {
                if !self.ice[k].on_trade(t.px, t.qty, t.buy) || !self.ice[k].qualifies(cfg.iceberg.exec_mult, cfg.iceberg.min_refills) {
                    continue;
                }
                let ic = self.ice[k];
                let usd = self.usd_t(ic.px, ic.exec);
                if ic.recorded {
                    if let Some(r) = self.icebergs.iter_mut().rev().find(|r| r.born == ic.born && r.bid == ic.bid && r.px == ic.px) {
                        r.exec = ic.exec;
                        r.max_disp = ic.max_disp;
                        r.refills = ic.refills;
                        r.usd = usd;
                        r.ms = now;
                    }
                } else {
                    self.ice[k].recorded = true;
                    self.icebergs.push_back(IceRec { ms: now, born: ic.born, bid: ic.bid, px: ic.px, exec: ic.exec, max_disp: ic.max_disp, refills: ic.refills, usd, active: true });
                }
                if !ic.alerted && usd >= ice_alert {
                    self.ice[k].alerted = true;
                    let (kind, side) = if ic.bid { ("ICEBERG_BID", "bid") } else { ("ICEBERG_ASK", "ask") };
                    let text = format!(
                        "{kind} {} absorbed {:.4} ({}) vs max displayed {:.4}, {} refills over {:.1}s: a passive {} defending {}",
                        self.p(ic.px),
                        ic.exec,
                        fmt_usd(usd),
                        ic.max_disp,
                        ic.refills,
                        (now - ic.born) as f64 / 1000.0,
                        if ic.bid { "buyer" } else { "seller" },
                        self.p(ic.px)
                    );
                    out.push(alert(&self.sym, kind, side, text, json!({"px": ic.px as f64 * tick, "exec": ic.exec, "usd": usd, "max_disp": ic.max_disp, "refills": ic.refills})));
                }
            }
            if let Some(s) = self.burst.on_trade(t, cfg.sweep.burst_ms, cfg.sweep.burst_min_ticks, cfg.sweep.burst_p99_mult * p99) {
                out.extend(self.on_sweep(cfg, s));
            }
        }
        // clips: one per (T, side) of the batch
        let mut g: BTreeMap<(i64, bool), Clip> = BTreeMap::new();
        for t in &batch {
            let c = g.entry((t.ex_ms, t.buy)).or_insert(Clip { ms: now, ex_ms: t.ex_ms, buy: t.buy, qty: 0.0, usd: 0.0 });
            c.qty += t.qty;
            c.usd += t.usd;
        }
        for c in g.into_values() {
            let signed = if c.buy { c.usd } else { -c.usd };
            match self.cuts {
                Some(cuts) => {
                    let b = bucket(&cuts, c.usd);
                    self.bdelta[b].add(sec, signed);
                    self.bgross[b].add(sec, c.usd);
                }
                None => self.unbucketed.add(sec, signed),
            }
            self.dist.push(c.usd);
            self.clips.push_back(c);
        }
        for s in multi_level_sweeps(&batch, cfg.sweep.min_levels) {
            out.extend(self.on_sweep(cfg, s));
        }
        while self.recent.front().is_some_and(|x| x.0 < now - 10_000) {
            self.recent.pop_front();
        }
        while self.clips.front().is_some_and(|c| c.ms < now - cfg.window_s * 1000) {
            self.clips.pop_front();
        }
        out
    }

    fn on_sweep(&mut self, cfg: &TapeCfg, s: Sweep) -> Vec<Alert> {
        let mut out = Vec::new();
        let med = self.sweep_hist.quantile(0.5, cfg.sweep.alert_min_history);
        let p99 = self.cuts.map(|c| c[2]);
        if let Some(m) = med
            && let Some(p99) = p99
            && m > 0.0
            && s.usd > cfg.sweep.alert_median_mult * m
            && s.usd > cfg.sweep.alert_min_p99_mult * p99
        {
            let side = if s.buy { "buy" } else { "sell" };
            let text = format!(
                "SWEEP {side} ({}) {} -> {} across {} levels, {:.4} ({} = {:.1}x median sweep): stops taken up to {}",
                s.kind,
                self.p(s.from_px),
                self.p(s.to_px),
                s.levels,
                s.qty,
                fmt_usd(s.usd),
                s.usd / m,
                self.p(s.to_px)
            );
            out.push(alert(&self.sym, "SWEEP", side, text, json!({"sweep": s, "from": s.from_px as f64 * self.tick, "to": s.to_px as f64 * self.tick, "median_sweep_usd": m})));
        }
        self.sweep_hist.push(s.usd);
        self.sweeps.push_back(s);
        out
    }

    pub fn on_liq(&mut self, cfg: &TapeCfg, now: i64, data: &Value) -> Vec<Alert> {
        self.msgs += 1;
        let mut out = Vec::new();
        for r in parse_liquidations(data, now) {
            let long = r.side == "long";
            let usd = r.price * r.size;
            let s = usize::from(!long);
            self.liq_usd[s].add(now.div_euclid(1000), usd);
            self.liq_n[s].add(now.div_euclid(1000), 1.0);
            if let Some(c) = self.liq.on_liq(now, long, r.price, usd, cfg.liq.cascade_n, cfg.liq.cascade_s * 1000) {
                let side = if long { "longs" } else { "shorts" };
                let text = format!(
                    "LIQ_CASCADE {side} liquidated x{} in {}s, {} between {:.*} and {:.*} (forced {})",
                    c.n,
                    cfg.liq.cascade_s,
                    fmt_usd(c.usd),
                    self.dec,
                    c.lo_px,
                    self.dec,
                    c.hi_px,
                    if long { "selling" } else { "buying" }
                );
                out.push(alert(&self.sym, "LIQ_CASCADE", side, text, json!({"cascade": c})));
            }
        }
        out
    }

    /// The once-per-second step: flicker classification, percentiles, algo start/stop, spoof follow-ups, mid samples.
    pub fn on_second(&mut self, cfg: &TapeCfg, now: i64) -> Vec<Alert> {
        let mut out = Vec::new();
        let sec = now.div_euclid(1000);
        self.l1.classify(now, &self.recent, cfg.hft.flicker_ms, cfg.hft.trade_pad_ms);
        self.liq.tick(now, cfg.liq.cascade_n, cfg.liq.cascade_s * 1000);
        let t = self.touch();
        if let Some(t) = t {
            let mid = t.mid * self.tick;
            if let Some((ps, pm)) = self.last_mid_sec
                && sec > ps
                && pm > 0.0
            {
                self.ret1s.z_then_push((mid / pm).ln() / ((sec - ps) as f64).sqrt());
            }
            self.last_mid_sec = Some((sec, mid));
            self.mids.push_back((now, mid));
            while self.mids.front().is_some_and(|m| m.0 < now - cfg.window_s * 1000 - 1000) {
                self.mids.pop_front();
            }
        }
        // spoof follow-ups: did mid move away from the pulled wall (bid wall: up; ask wall: down)?
        if let Some(t) = t {
            for s in self.spoofs.iter_mut().filter(|s| s.follow_bps.is_none() && now >= s.due) {
                let away = if s.end.bid { t.mid - s.mid_at } else { s.mid_at - t.mid };
                s.follow_bps = Some(away / s.mid_at * 1e4);
                s.moved_away = Some(away >= 1.0);
            }
        }
        if sec % cfg.buckets.recompute_every_s == 0 {
            let b = &cfg.buckets;
            if let (Some(a), Some(m), Some(w)) = (self.dist.quantile(b.p_small, b.min_obs), self.dist.quantile(b.p_large, b.min_obs), self.dist.quantile(b.p_whale, b.min_obs)) {
                self.cuts = Some([a, m, w]);
            }
        }
        if sec % cfg.hft.sample_every_s == 0 {
            let (rate, flick, _) = self.hft_now(cfg, sec);
            self.rate_hist.push(rate);
            if let Some(f) = flick {
                self.flick_hist.push(f);
            }
        }
        if sec % cfg.algo.eval_every_s == 0 {
            let a = &cfg.algo;
            self.algos = match self.cuts {
                Some(c) => {
                    let clips: Vec<Clip> = self.clips.iter().filter(|x| x.usd >= a.min_clip_p50_mult * c[0]).copied().collect();
                    detect_algos(&clips, a.qty_tol, a.min_clips, a.max_cv, a.min_cadence_ms)
                }
                None => Vec::new(),
            };
            for al in self.algos.clone() {
                let k = al.key();
                if !self.algo_on.contains_key(&k) {
                    let (kind, side) = if al.buy { ("ALGO_BUY", "buy") } else { ("ALGO_SELL", "sell") };
                    let text = format!(
                        "{kind} start: {} clips of {:.4} every {:.1}s (cv {:.2}), {} in {}s",
                        al.clips,
                        al.clip_qty,
                        al.cadence_s,
                        al.cv,
                        fmt_usd(al.notional),
                        cfg.window_s
                    );
                    out.push(alert(&self.sym, kind, side, text, json!({"algo": al, "event": "start"})));
                }
                self.algo_on.insert(k, (now, al));
            }
            let stale: Vec<String> = self.algo_on.iter().filter(|(_, (seen, _))| now - seen > cfg.algo.stop_after_s * 1000).map(|(k, _)| k.clone()).collect();
            for k in stale {
                if let Some((_, al)) = self.algo_on.remove(&k) {
                    let (kind, side) = if al.buy { ("ALGO_BUY", "buy") } else { ("ALGO_SELL", "sell") };
                    let text = format!("{kind} stop: clip {:.4} not seen for {}s (last run {} clips, {})", al.clip_qty, cfg.algo.stop_after_s, al.clips, fmt_usd(al.notional));
                    out.push(alert(&self.sym, kind, side, text, json!({"algo": al, "event": "stop"})));
                }
            }
        }
        let keep = now - 2 * cfg.window_s * 1000;
        while self.spoofs.front().is_some_and(|s| s.end.ms < keep) {
            self.spoofs.pop_front();
        }
        while self.consumed.front().is_some_and(|s| s.ms < keep) {
            self.consumed.pop_front();
        }
        while self.icebergs.front().is_some_and(|s| !s.active && s.ms < keep) {
            self.icebergs.pop_front();
        }
        while self.sweeps.front().is_some_and(|s| s.ms < keep) {
            self.sweeps.pop_front();
        }
        out
    }

    /// (L1 updates / s, flicker share, cancel-to-trade near the touch) over the trailing window.
    fn hft_now(&self, cfg: &TapeCfg, sec: i64) -> (f64, Option<f64>, Option<f64>) {
        let w = cfg.window_s;
        let rate = self.l1.upd.sum(sec, w) / w as f64 + 0.0;
        let ends = self.l1.ends.sum(sec, w);
        let flick = (ends > 0.0).then(|| self.l1.flicks.sum(sec, w) / ends + 0.0);
        let tv = self.tvol.sum(sec, w);
        let ct = (tv > 0.0).then(|| (self.canc.sum(sec, w) - tv).max(0.0) / tv);
        (rate, flick, ct)
    }

    /// One summary per window: the console line and the JSON row.
    pub fn window(&self, cfg: &TapeCfg, now: i64) -> (String, Value) {
        let sec = now.div_euclid(1000);
        let w = cfg.window_s;
        let since = now - w * 1000;
        let ts = chrono::DateTime::from_timestamp_millis(now).unwrap_or_default().format("%H:%M:%SZ");
        let t = self.touch();
        let mid = t.map(|t| t.mid * self.tick).unwrap_or(0.0);
        let ret = self.mids.front().filter(|m| m.1 > 0.0 && mid > 0.0).map(|m| (mid / m.1).ln()).unwrap_or(0.0);
        let ret_z = self.ret1s.sd().map(|sd| ret / (sd * (w as f64).sqrt()));
        // 1. HFT
        let (rate, flick, ct) = self.hft_now(cfg, sec);
        let p = cfg.hft.pctile;
        let (rp, fp) = (self.rate_hist.quantile(p, cfg.hft.min_history), self.flick_hist.quantile(p, cfg.hft.min_history));
        let hft = match (rp, fp, flick) {
            (Some(rp), Some(fp), Some(f)) => {
                if rate > rp && f > fp {
                    "ON"
                } else {
                    "off"
                }
            }
            _ => "warmup",
        };
        // 6. buckets
        let delta: [f64; 4] = std::array::from_fn(|k| self.bdelta[k].sum(sec, w) + 0.0);
        let gross: [f64; 4] = std::array::from_fn(|k| self.bgross[k].sum(sec, w) + 0.0);
        let total_gross: f64 = gross.iter().sum();
        // 3. algos (latest evaluation)
        let algo_net: f64 = self.algos.iter().map(|a| if a.buy { a.notional } else { -a.notional }).sum();
        // 4. defence: icebergs + consumed walls in the window
        let ices: Vec<&IceRec> = self.icebergs.iter().filter(|r| r.active || r.ms >= since).collect();
        let cons: Vec<&WallEnd> = self.consumed.iter().filter(|e| e.ms >= since).collect();
        let defence: f64 = ices.iter().map(|r| if r.bid { r.usd } else { -r.usd }).sum::<f64>() + cons.iter().map(|e| (if e.bid { 1.0 } else { -1.0 }) * self.usd_t(e.px, e.traded)).sum::<f64>();
        // 2. spoofs: a pulled bid wall implies a seller (push down), a pulled ask wall a buyer (push up)
        let sp: Vec<&SpoofRec> = self.spoofs.iter().filter(|s| s.end.ms >= since).collect();
        let spoof_vote: f64 = sp.iter().map(|s| if s.end.bid { -s.usd } else { s.usd }).sum();
        // 5. sweeps, liquidations
        let sw: Vec<&Sweep> = self.sweeps.iter().filter(|s| s.ms >= since).collect();
        let liq_n: [f64; 2] = std::array::from_fn(|k| self.liq_n[k].sum(sec, w) + 0.0);
        let liq_usd: [f64; 2] = std::array::from_fn(|k| self.liq_usd[k].sum(sec, w) + 0.0);
        // 7. push
        let big = big_vote(&delta, &gross, cfg.push.big_min_imbalance);
        let votes = [("large_whale_delta", big), ("algo", algo_net), ("defence", defence), ("spoof", spoof_vote)];
        let (side, agree, active) = push_vote(&votes);
        let push_side = match side {
            1 => "up",
            -1 => "down",
            _ => "none",
        };
        let bids: Vec<(i64, f64)> = self.deep.bids_desc().take(1000).map(|(p, q)| (self.t(p), q)).collect();
        let asks: Vec<(i64, f64)> = self.deep.asks_asc().take(1000).map(|(p, q)| (self.t(p), q)).collect();
        let pool_j = |bid: bool| -> Option<Value> {
            let t = t?;
            let (px, q) = pool(if bid { &bids } else { &asks }, bid, t, cfg.pool.min_ticks, cfg.pool.max_pct)?;
            Some(json!({"px": px as f64 * self.tick, "qty": q, "usd": self.usd_t(px, q), "dist_bps": (px as f64 - t.mid) / t.mid * 1e4}))
        };
        let (pool_up, pool_dn) = (pool_j(false), pool_j(true));
        let target = match side {
            1 => pool_up.clone(),
            -1 => pool_dn.clone(),
            _ => None,
        };
        let defended = t.and_then(|t| ices.iter().min_by(|a, b| (a.px as f64 - t.mid).abs().total_cmp(&(b.px as f64 - t.mid).abs())).map(|r| json!({"px": r.px as f64 * self.tick, "side": if r.bid { "bid" } else { "ask" }, "usd": r.usd, "active": r.active})));
        let swept = sw.last().map(|s| json!({"px": s.to_px as f64 * self.tick, "side": if s.buy { "buy" } else { "sell" }, "kind": s.kind, "usd": s.usd}));
        let pj = |v: &Option<Value>| v.as_ref().and_then(|x| x["px"].as_f64()).map(|x| format!("{:.*}", self.dec, x)).unwrap_or_else(|| "-".into());
        let algo_s = if self.algos.is_empty() {
            "-".to_string()
        } else {
            self.algos.iter().map(|a| format!("{} {:.4}x{} @{:.1}s {}", if a.buy { "ALGO_BUY" } else { "ALGO_SELL" }, a.clip_qty, a.clips, a.cadence_s, fmt_usd(a.notional))).collect::<Vec<_>>().join(", ")
        };
        let ice_s = if ices.is_empty() {
            "-".to_string()
        } else {
            ices.iter().map(|r| format!("{} {} {}", if r.bid { "BID" } else { "ASK" }, self.p(r.px), fmt_usd(r.usd))).collect::<Vec<_>>().join(", ")
        };
        let spoof_s = if sp.is_empty() {
            "-".to_string()
        } else {
            let b = sp.iter().filter(|s| s.end.bid).count();
            format!("{} (bid {b}, ask {}) {}", sp.len(), sp.len() - b, fmt_usd(sp.iter().map(|s| s.usd).sum()))
        };
        let sweep_s = match sw.last() {
            None => "0".to_string(),
            Some(l) => format!("{} (buy {} sell {}) last {} to {}", sw.len(), sw.iter().filter(|s| s.buy).count(), sw.iter().filter(|s| !s.buy).count(), if l.buy { "buy" } else { "sell" }, self.p(l.to_px)),
        };
        let line = format!(
            "TAPE {ts} {} mid {:.*} ret{}s {:+.1}bp z {} | HFT_MM {hft} L1 {rate:.1}/s flick {} c/t {} | delta{} S {} M {} L {} W {} (gross {}) | {} | ICE {} | SPOOF {} | SWEEP {} | LIQ long {}/{} short {}/{} | PUSH {} {agree}/{active} target {} defended {} swept {} [{NOTE}]",
            self.sym,
            self.dec,
            mid,
            w,
            ret * 1e4,
            ret_z.map(|z| format!("{z:+.1}")).unwrap_or_else(|| "-".into()),
            flick.map(|f| format!("{:.0}%", f * 100.0)).unwrap_or_else(|| "-".into()),
            ct.map(|c| format!("{c:.1}")).unwrap_or_else(|| "-".into()),
            if self.cuts.is_none() { "(warmup)" } else { "" },
            fmt_usd(delta[0]),
            fmt_usd(delta[1]),
            fmt_usd(delta[2]),
            fmt_usd(delta[3]),
            fmt_usd(total_gross),
            algo_s,
            ice_s,
            spoof_s,
            sweep_s,
            liq_n[0],
            fmt_usd(liq_usd[0]),
            liq_n[1],
            fmt_usd(liq_usd[1]),
            push_side.to_uppercase(),
            pj(&target),
            defended.as_ref().and_then(|d| d["px"].as_f64()).map(|x| format!("{:.*}", self.dec, x)).unwrap_or_else(|| "-".into()),
            swept.as_ref().and_then(|d| d["px"].as_f64()).map(|x| format!("{:.*}", self.dec, x)).unwrap_or_else(|| "-".into()),
        );
        let row = json!({
            "type": "window", "symbol": self.sym, "ms": now, "window_s": w, "mid": mid, "ret_log": ret, "ret_z": ret_z,
            "hft": {"label": hft, "l1_updates_per_s": rate, "flicker_share": flick, "cancel_to_trade": ct, "rate_p": rp, "flicker_p": fp,
                    "l1_episodes_ended": self.l1.ends.sum(sec, w), "flickers": self.l1.flicks.sum(sec, w)},
            "delta_usd": {"small": delta[0], "medium": delta[1], "large": delta[2], "whale": delta[3], "unbucketed": self.unbucketed.sum(sec, w)},
            "gross_usd": {"small": gross[0], "medium": gross[1], "large": gross[2], "whale": gross[3]},
            "delta_share_of_gross": if total_gross > 0.0 { json!(delta.iter().map(|d| d / total_gross).collect::<Vec<_>>()) } else { Value::Null },
            "bucket_cuts_usd": self.cuts,
            "algos": self.algos,
            "icebergs": ices,
            "walls_consumed": cons,
            "walls_tracked": self.walls.map.len(),
            "walls_pulled_not_spoof": self.pulled_n.sum(sec, w),
            "median_level_qty": self.median,
            "spoofs": sp,
            "sweeps": sw,
            "liquidations": {"long_n": liq_n[0], "long_usd": liq_usd[0], "short_n": liq_n[1], "short_usd": liq_usd[1]},
            "push": {"push_side": push_side, "agree": agree, "active": active, "confidence": format!("{agree}/{active}"),
                     "votes": votes.iter().map(|(k, v)| (k.to_string(), json!(v))).collect::<serde_json::Map<_, _>>(),
                     "target_zone": target, "pool_up": pool_up, "pool_down": pool_dn, "defended_level": defended, "swept_level": swept, "note": NOTE},
            "book_ok": {"l1": self.l1book.valid, "near": self.near.valid, "deep": self.deep.valid, "deep_depth": self.deep_depth},
            "resets": self.resets, "msgs": self.msgs,
        });
        (line, row)
    }
}

// ---------------------------------------------------------------- output

struct Out {
    root: PathBuf,
    log: PathBuf,
    journal: String,
    run_id: String,
    cfg_hash: String,
}

impl Out {
    fn line(&self, s: &str) {
        println!("{s}");
        let _ = std::io::stdout().flush();
        if let Some(d) = self.log.parent() {
            let _ = std::fs::create_dir_all(d);
        }
        if let Ok(mut f) = std::fs::OpenOptions::new().create(true).append(true).open(&self.log) {
            let _ = writeln!(f, "{s}");
        }
    }

    fn journal(&self, mut v: Value) -> anyhow::Result<()> {
        let now = Utc::now();
        v["ts"] = json!(now.to_rfc3339_opts(SecondsFormat::Millis, true));
        v["run_id"] = json!(self.run_id);
        v["version"] = json!(TAPE_VERSION);
        v["cfg_hash"] = json!(self.cfg_hash);
        let date = now.format("%Y-%m-%d").to_string();
        let p = self.root.join(self.journal.replace("{date}", &date).replace("<date>", &date));
        if let Some(d) = p.parent() {
            std::fs::create_dir_all(d)?;
        }
        let mut f = std::fs::OpenOptions::new().create(true).append(true).open(&p).with_context(|| p.display().to_string())?;
        writeln!(f, "{}", serde_json::to_string(&v)?)?;
        Ok(())
    }
}

// ---------------------------------------------------------------- CLI

#[derive(clap::Args, Debug)]
pub struct TapeArgs {
    /// Comma-separated linear USDT perps (SPY symbols are refused).
    #[arg(long, default_value = "BTCUSDT,ETHUSDT,SOLUSDT,ZECUSDT,XRPUSDT,DOGEUSDT,HYPEUSDT,SUIUSDT")]
    pub symbols: String,
    /// Summary window in seconds (overrides config window_s).
    #[arg(long)]
    pub window_s: Option<i64>,
    #[arg(long, default_value = "config/tape.yaml")]
    pub config: String,
    /// Console log (appended), relative to the repo.
    #[arg(long, default_value = "logs/tape.log")]
    pub log: String,
    /// Journal path; {date} or <date> is replaced by the UTC date.
    #[arg(long, default_value = "data/journal/tape/{date}.jsonl")]
    pub journal: String,
    /// Stop after N minutes (default: run until killed).
    #[arg(long)]
    pub minutes: Option<u64>,
}

pub async fn run(ctx: &Ctx, a: TapeArgs) -> anyhow::Result<()> {
    let mut cfg = TapeCfg::load(&ctx.s.root.join(&a.config))?;
    if let Some(w) = a.window_s {
        cfg.window_s = w;
        cfg.validate()?;
    }
    let syms: Vec<String> = a.symbols.split(',').map(|s| s.trim().to_uppercase()).filter(|s| !s.is_empty()).collect();
    ensure!(!syms.is_empty(), "no symbols");
    ensure!(syms.iter().all(|s| !s.starts_with("SPY")), "SPY symbols are never used");
    let rest = ctx.market_rest()?;
    let md = MarketData::new(&rest, "linear");
    let ins = md.instruments().await?;
    let mut book: BTreeMap<String, Sym> = BTreeMap::new();
    for s in &syms {
        let i = ins.iter().find(|i| &i.symbol == s).with_context(|| format!("{s} is not a listed linear instrument"))?;
        ensure!(i.status == "Trading", "{s} status {}", i.status);
        book.insert(s.clone(), Sym::new(&cfg, s, i.tick_size));
    }
    let mut topics: Vec<String> = Vec::new();
    for s in &syms {
        let deep = cfg.deep_for(s);
        let mut depths = vec![cfg.l1_depth, cfg.near_depth, deep];
        depths.dedup();
        topics.extend(depths.iter().map(|d| format!("orderbook.{d}.{s}")));
        topics.push(format!("publicTrade.{s}"));
        topics.push(format!("allLiquidation.{s}"));
    }
    let start = now_ms();
    let run_id = format!("{}-{}", chrono::DateTime::from_timestamp_millis(start).unwrap_or_default().format("%Y%m%dT%H%M%S"), std::process::id());
    let out = Out { root: ctx.s.root.clone(), log: ctx.s.root.join(&a.log), journal: a.journal.clone(), run_id: run_id.clone(), cfg_hash: cfg.hash() };
    out.journal(json!({"type": "run_start", "ms": start, "symbols": syms, "topics": topics, "config": cfg, "pid": std::process::id(), "note": NOTE}))?;
    out.line(&format!(
        "atx tape {TAPE_VERSION} run {run_id} pid {} cfg {} | {} symbols, {} topics, window {}s | public data only, no orders | {NOTE}",
        std::process::id(),
        cfg.hash(),
        syms.len(),
        topics.len(),
        cfg.window_s
    ));

    let url = ctx.s.bybit_endpoints(ctx.s.venues.bybit.market_data_env)?.ws_public.clone();
    let (tx, mut rx) = tokio::sync::mpsc::channel::<WsMessage>(100_000);
    let ws_task = tokio::spawn(ws::run(url, topics, None, tx));
    let mut tick = tokio::time::interval(Duration::from_millis(1000));
    let wms = cfg.window_s * 1000;
    let mut next_window = (start.div_euclid(wms) + 1) * wms;
    let deadline = a.minutes.map(|m| start + m as i64 * 60_000);
    let mut last_alert: BTreeMap<(String, String, String), i64> = BTreeMap::new();
    let ctrl_c = tokio::signal::ctrl_c();
    tokio::pin!(ctrl_c);
    let mut emit = |alerts: Vec<Alert>, now: i64| -> anyhow::Result<()> {
        for al in alerts {
            let key = (al.sym.clone(), al.kind.clone(), al.side.clone());
            let shown = last_alert.get(&key).is_none_or(|t| now - t >= cfg.alert_cooldown_ms);
            if shown {
                last_alert.insert(key, now);
                let ts = chrono::DateTime::from_timestamp_millis(now).unwrap_or_default().format("%H:%M:%S%.3fZ");
                out.line(&format!("ALERT {ts} {} {} [{NOTE}]", al.sym, al.text));
            }
            out.journal(json!({"type": "alert", "ms": now, "symbol": al.sym, "kind": al.kind, "side": al.side, "text": al.text, "data": al.data, "console": shown, "note": NOTE}))?;
        }
        Ok(())
    };
    loop {
        tokio::select! {
            m = rx.recv() => {
                let Some(m) = m else { bail!("websocket task ended") };
                let now = m.received_ts;
                if m.topic == "__reconnect__" {
                    for s in book.values_mut() {
                        s.on_reset();
                    }
                    out.line(&format!("RECONNECT {} books reset", chrono::DateTime::from_timestamp_millis(now).unwrap_or_default().format("%H:%M:%SZ")));
                    continue;
                }
                let parts: Vec<&str> = m.topic.split('.').collect();
                let alerts = match parts.as_slice() {
                    ["orderbook", d, s] => match (book.get_mut(*s), d.parse::<u32>()) {
                        (Some(x), Ok(d)) => x.on_book(&cfg, d, now, &m.kind, &m.data),
                        _ => vec![],
                    },
                    ["publicTrade", s] => book.get_mut(*s).map(|x| x.on_trades(&cfg, now, &m.data)).unwrap_or_default(),
                    ["allLiquidation", s] => book.get_mut(*s).map(|x| x.on_liq(&cfg, now, &m.data)).unwrap_or_default(),
                    _ => vec![],
                };
                emit(alerts, now)?;
            }
            _ = tick.tick() => {
                let now = now_ms();
                for s in book.values_mut() {
                    let al = s.on_second(&cfg, now);
                    emit(al, now)?;
                }
                if now >= next_window {
                    for s in book.values() {
                        let (line, row) = s.window(&cfg, now);
                        out.line(&line);
                        out.journal(row)?;
                    }
                    next_window = (now.div_euclid(wms) + 1) * wms;
                }
                if deadline.is_some_and(|d| now >= d) { break; }
            }
            _ = &mut ctrl_c => { out.line("ctrl-c: stopping"); break; }
        }
    }
    out.journal(json!({"type": "run_end", "ms": now_ms()}))?;
    ws_task.abort();
    Ok(())
}

// ---------------------------------------------------------------- tests

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg() -> TapeCfg {
        TapeCfg::load(&Path::new(env!("CARGO_MANIFEST_DIR")).join("../../config/tape.yaml")).expect("config/tape.yaml loads")
    }

    #[test]
    fn flicker_counts_quotes_that_leave_fast_without_a_print() {
        let mut l = L1Track::new(120);
        let trades: VecDeque<(i64, i64)> = VecDeque::from(vec![(10_150, 101)]);
        l.on_l1(10_000, 100, 1.0, 110, 1.0);
        l.on_l1(10_100, 101, 1.0, 110, 1.0); // bid 100 ends after 100 ms, no print at 100: flicker
        l.on_l1(10_200, 100, 2.0, 110, 1.0); // bid 101 ends after 100 ms but printed at 101: not a flicker
        l.on_l1(12_000, 99, 2.0, 110, 1.0); // bid 100 lived 1.8 s: not a flicker
        l.on_l1(12_000, 99, 2.0, 110, 3.0); // size change only: an update, no episode
        assert_eq!(l.upd.sum(12, 60), 4.0);
        l.classify(14_000, &trades, 500, 500);
        assert_eq!(l.ends.sum(14, 60), 3.0);
        assert_eq!(l.flicks.sum(14, 60), 1.0);
    }

    #[test]
    fn spoof_is_a_wall_pulled_on_approach_and_traded_walls_are_defence() {
        let c = cfg().spoof;
        let mut w = Walls::default();
        let t0 = Touch { bid: 100_000, ask: 100_001, mid: 100_000.5 };
        // median level 1, wall of 10 on the bid 8 ticks below the touch
        assert!(w.on_level(&c, 0, Lvl { bid: true, px: 99_992, q: 10.0 }, 1.0, t0).is_none());
        assert_eq!(w.map.len(), 1);
        // price approaches to 2 ticks; the wall is cancelled without trades
        let t1 = Touch { bid: 99_994, ask: 99_995, mid: 99_994.5 };
        let e = w.on_level(&c, 3_000, Lvl { bid: true, px: 99_992, q: 0.0 }, 1.0, t1).unwrap();
        assert_eq!(e.kind, WallEndKind::Spoof);
        assert_eq!((e.dist_ticks, e.born_dist, e.life_ms), (2, 8, 3_000));
        // same wall pulled while price is far away: pulled, not a spoof
        w.on_level(&c, 0, Lvl { bid: false, px: 100_010, q: 10.0 }, 1.0, t0);
        assert_eq!(w.on_level(&c, 3_000, Lvl { bid: false, px: 100_010, q: 1.0 }, 1.0, t0).unwrap().kind, WallEndKind::Pulled);
        // a wall that trades away: consumed (defence)
        w.on_level(&c, 0, Lvl { bid: true, px: 99_999, q: 10.0 }, 1.0, t0);
        w.on_trade(99_999, 8.0, false);
        let e = w.on_level(&c, 2_000, Lvl { bid: true, px: 99_999, q: 2.0 }, 1.0, Touch { bid: 99_999, ask: 100_000, mid: 99_999.5 }).unwrap();
        assert_eq!(e.kind, WallEndKind::Consumed);
        // small size is not a wall; partial reduction above end_frac keeps it
        assert!(w.on_level(&c, 0, Lvl { bid: true, px: 99_990, q: 4.0 }, 1.0, t0).is_none());
        assert!(w.map.is_empty());
        w.on_level(&c, 0, Lvl { bid: true, px: 99_990, q: 10.0 }, 1.0, t0);
        assert!(w.on_level(&c, 10, Lvl { bid: true, px: 99_990, q: 6.0 }, 1.0, t0).is_none());
        // a flicker-short wall is not a spoof even on approach
        let e = w.on_level(&c, 100, Lvl { bid: true, px: 99_990, q: 0.0 }, 1.0, Touch { bid: 99_991, ask: 99_992, mid: 99_991.5 }).unwrap();
        assert_eq!(e.kind, WallEndKind::Pulled);
    }

    #[test]
    fn algo_cadence_detects_regular_identical_clips_only() {
        let clip = |ex: i64, buy: bool, qty: f64| Clip { ms: ex, ex_ms: ex, buy, qty, usd: qty * 100.0 };
        let mut v: Vec<Clip> = (0..8).map(|k| clip(k * 5_000 + (k % 2) * 300, false, 0.5)).collect();
        // random-sized and irregular buys
        v.extend([clip(100, true, 0.5), clip(150, true, 0.5), clip(9_000, true, 0.5), clip(9_100, true, 0.5), clip(30_000, true, 0.5)]);
        v.push(clip(1_000, false, 0.9));
        let a = detect_algos(&v, 0.01, 5, 0.5, 200.0);
        assert_eq!(a.len(), 1, "{a:?}");
        let a = &a[0];
        assert!(!a.buy && a.clips == 8 && (a.clip_qty - 0.5).abs() < 1e-12);
        assert!((a.cadence_s - 5.0).abs() < 0.1 && a.cv < 0.5);
        assert!((a.notional - 400.0).abs() < 1e-9);
        assert!(detect_algos(&v[..4], 0.01, 5, 0.5, 200.0).is_empty(), "fewer than 5 clips");
    }

    #[test]
    fn iceberg_needs_execution_beyond_display_and_a_refill() {
        let mut ice = Ice::new(true);
        assert!(ice.on_l1(0, 100, 2.0).is_none());
        for k in 0..3 {
            assert!(ice.on_trade(100, 1.5, false), "sell hits the bid");
            ice.on_l1(10 + k, 100, 0.5);
            ice.on_l1(20 + k, 100, 2.0); // refilled
        }
        assert!(!ice.on_trade(100, 1.0, true), "buys do not hit the bid");
        assert!(!ice.on_trade(99, 1.0, false), "another price");
        assert_eq!((ice.exec, ice.refills), (4.5, 3));
        assert!(ice.qualifies(1.5, 1));
        let done = ice.on_l1(100, 99, 1.0).unwrap();
        assert!(done.qualifies(1.5, 1) && done.px == 100);
        // a level eaten without refills is not an iceberg
        let mut x = Ice::new(false);
        x.on_l1(0, 200, 4.0);
        x.on_trade(200, 4.0, true);
        x.on_l1(1, 200, 0.5);
        assert!(!x.qualifies(1.5, 1));
    }

    #[test]
    fn sweeps_multi_level_and_bursts() {
        let t = |ex: i64, px: i64, buy: bool, usd: f64| TTr { ms: 0, ex_ms: ex, px, buy, qty: usd / 100.0, usd };
        let batch = [t(1, 100, false, 100.0), t(1, 99, false, 200.0), t(1, 97, false, 300.0), t(2, 100, true, 50.0), t(2, 100, true, 50.0)];
        let s = multi_level_sweeps(&batch, 2);
        assert_eq!(s.len(), 1);
        assert!(!s[0].buy && s[0].levels == 3 && s[0].to_px == 97 && s[0].from_px == 100 && (s[0].usd - 600.0).abs() < 1e-9);
        let mut b = Burst::default();
        assert!(b.on_trade(&t(0, 100, true, 400.0), 1000, 2, 1000.0).is_none());
        assert!(b.on_trade(&t(500, 101, true, 400.0), 1000, 2, 1000.0).is_none(), "800 < 1000");
        let s = b.on_trade(&t(900, 102, true, 400.0), 1000, 2, 1000.0).unwrap();
        assert!(s.buy && s.to_px == 102 && s.from_px == 100 && s.kind == "burst");
        assert!(b.on_trade(&t(2_500, 100, true, 5_000.0), 1000, 2, 1000.0).is_none(), "no price move");
    }

    #[test]
    fn liquidation_cascade_fires_once_per_cluster() {
        let mut l = LiqClusters::default();
        let mut fired = 0;
        for k in 0..8 {
            if l.on_liq(k * 1000, true, 100.0 - k as f64, 1000.0, 5, 30_000).is_some() {
                fired += 1;
                assert_eq!(k, 4);
            }
        }
        assert_eq!(fired, 1);
        assert!(l.on_liq(1000, false, 100.0, 1.0, 5, 30_000).is_none(), "the other side is separate");
        l.tick(100_000, 5, 30_000);
        assert!((0..5).map(|k| l.on_liq(100_000 + k, true, 90.0, 1.0, 5, 30_000)).last().flatten().is_some(), "re-armed");
    }

    #[test]
    fn size_buckets_split_by_rolling_percentiles() {
        let mut r = Roll::new(1000);
        for k in 1..=100 {
            r.push(k as f64);
        }
        let cuts = [r.quantile(0.5, 50).unwrap(), r.quantile(0.95, 50).unwrap(), r.quantile(0.99, 50).unwrap()];
        assert_eq!(cuts, [51.0, 95.0, 99.0]);
        assert_eq!([bucket(&cuts, 10.0), bucket(&cuts, 60.0), bucket(&cuts, 97.0), bucket(&cuts, 500.0)], [0, 1, 2, 3]);
        assert!(Roll::new(10).quantile(0.5, 5).is_none(), "no cuts before min_obs");
    }

    #[test]
    fn push_summary_votes_pools_and_big_delta() {
        assert_eq!(push_vote(&[("big", -5.0), ("algo", -1.0), ("defence", 2.0), ("spoof", 0.0)]), (-1, 2, 3));
        assert_eq!(push_vote(&[("big", 1.0), ("algo", -1.0)]), (0, 0, 2));
        assert_eq!(push_vote(&[("big", 0.0)]), (0, 0, 0));
        // large+whale: -900 of 1000 gross counts; +50 of 1000 does not
        assert_eq!(big_vote(&[10.0, 0.0, -400.0, -500.0], &[0.0, 0.0, 500.0, 500.0], 0.2), -900.0);
        assert_eq!(big_vote(&[0.0, 0.0, 25.0, 25.0], &[0.0, 0.0, 500.0, 500.0], 0.2), 0.0);
        let t = Touch { bid: 1000, ask: 1001, mid: 1000.5 };
        let bids = [(1000, 50.0), (999, 3.0), (997, 9.0), (995, 7.0), (900, 100.0)];
        assert_eq!(pool(&bids, true, t, 2, 1.0), Some((997, 9.0)), "beyond 2 ticks, within 1 %: 900 is 10 % away");
        let asks = [(1001, 80.0), (1002, 5.0), (1004, 6.0)];
        assert_eq!(pool(&asks, false, t, 2, 1.0), Some((1004, 6.0)));
    }

    #[test]
    fn engine_end_to_end_on_synthetic_messages() {
        let c = cfg();
        let mut s = Sym::new(&c, "BTCUSDT", 0.1);
        let snap = json!({"b": [["100.0", "1"]], "a": [["100.1", "1"]], "u": 1});
        s.on_book(&c, 1, 1_000, "snapshot", &snap);
        let tr = json!([{"T": 1_000, "S": "Sell", "p": "100.0", "v": "0.5"}, {"T": 1_000, "S": "Sell", "p": "99.9", "v": "0.5"}]);
        s.on_trades(&c, 1_100, &tr);
        assert_eq!(s.sweeps.len(), 1);
        let liq = json!([{"T": 1_000, "s": "BTCUSDT", "S": "Buy", "v": "1", "p": "100"}]);
        s.on_liq(&c, 1_200, &liq);
        s.on_second(&c, 2_000);
        let (line, row) = s.window(&c, 60_000);
        assert!(line.contains("BTCUSDT") && line.contains(NOTE), "{line}");
        assert_eq!(row["liquidations"]["long_n"], json!(1.0));
        assert_eq!(row["push"]["note"], json!(NOTE));
    }
}
