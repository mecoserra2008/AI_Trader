//! `live_small`: the risk profile for the user's LIVE Bybit account (user authorisation 2026-11-15 ~16:10 UTC).
//!
//! Pure: no I/O and no clock. `atx live` gathers the inputs (account, bot positions, books, tiers, correlations) and
//! calls [`decide_live`] before every venue write. The limits come from `config/risk.yaml` `live_small`, and the yaml
//! can only TIGHTEN them: [`LiveCeilings`] holds the user's numbers compiled into the engine, and a profile that is
//! looser than them is refused at load (fail closed). The LLM never sets any of it; an intent carries no size.
//!
//! Sizing (USD, fees and slippage included):
//!     risk_to_stop(q) = q * (|e - s| + e * entry_bps + s * (taker + slippage)) <= risk_to_stop_usd   (1.70)
//!     max_loss(q)     = q * (|e - s| + e * entry_bps + s * (taker + tier.stop_slip_bps)) <= max_loss_per_trade_usd (2.00)
//!     q = floor_step(min(risk_to_stop_usd / per-unit risk, notional_cap / e, max_qty))
//! with entry_bps = maker + peg slip (PostOnly at the touch) and notional_cap = min(tier cap, depth share of the book).
//! If that rounds below the venue minimum, the minimum lot is used only when its max_loss still fits the cap and its
//! notional the tier cap; otherwise the trade is refused (`min_lot_over_max_loss`).

use std::collections::{BTreeMap, HashMap};
use std::path::Path;

use anyhow::{Context, ensure};
use at_core::num::{Round, round_to_step};
use at_core::types::{EntryType, InstrumentRules, Side};
use serde::{Deserialize, Serialize};

use crate::fees::FeeRates;
use crate::intent::{IntentSide, OrderIntent, PlacementCfg};

// ---------------------------------------------------------------- config

/// The user's limits (2026-11-15). The yaml may be stricter, never looser.
#[derive(Clone, Copy, Debug)]
pub struct LiveCeilings {
    pub max_loss_per_trade_usd: f64,
    pub max_entries_per_day: u32,
    pub daily_loss_stop_usd: f64,
    pub max_open_positions: u32,
    pub max_same_direction: u32,
    pub max_gross_notional_usd: f64,
    pub max_tier_notional_usd: f64,
    pub min_tier_notional_cap_usd: f64,
    pub max_entry_window_min: f64,
    pub min_manual_trades: u32,
    pub max_btc_corr_threshold: f64,
}

pub const LIVE_CEILINGS: LiveCeilings = LiveCeilings {
    max_loss_per_trade_usd: 2.00,
    max_entries_per_day: 20,
    daily_loss_stop_usd: 10.0,
    max_open_positions: 3,
    max_same_direction: 2,
    max_gross_notional_usd: 400.0,
    max_tier_notional_usd: 200.0,
    min_tier_notional_cap_usd: 0.0,
    max_entry_window_min: 15.0,
    min_manual_trades: 5,
    max_btc_corr_threshold: 0.7,
};

/// Symbols the bot never touches, whatever the yaml says.
pub const ALWAYS_EXCLUDED: &[&str] = &["SPYUSDT"];

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LiveTier {
    pub name: String,
    /// Rank by 24h turnover among Trading USDT linear perpetuals (1 = largest). None: no rank limit.
    #[serde(default)]
    pub max_rank: Option<u32>,
    pub min_turnover_usd: f64,
    /// Widest quoted spread for this tier (bp of the mid); a wider book drops to the next tier.
    pub max_spread_bps: f64,
    pub notional_cap_usd: f64,
    /// Adverse slip of a stop exit beyond the stop price, budgeted in the $2 cap (bp of the stop price).
    pub stop_slip_bps: f64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LiveDepthCfg {
    /// Book depth is summed within this distance of the touch, each side (bp).
    pub band_bps: f64,
    /// A position's notional is at most this share of min(bid depth, ask depth) within the band.
    pub max_share: f64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LiveCorrCfg {
    pub btc_symbol: String,
    /// Rolling correlation of bar log returns to BTC above this = the BTC-beta cluster (one bet).
    pub threshold: f64,
    pub freq: String,
    pub window_hours: u32,
    /// Fewer aligned returns than this: the correlation is unknown and the symbol counts as BTC-correlated.
    pub min_obs: u32,
    pub cache_hours: f64,
    /// Sector clusters (`config/sectors.yaml`, overrides > sectors).
    pub sector_file: String,
    /// Sectors that are not a cluster (catch-all labels).
    #[serde(default)]
    pub ignore_sectors: Vec<String>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LiveEntryCfg {
    /// PostOnly (maker only) or GTC.
    pub time_in_force: String,
    /// Re-peg window; the entry is cancelled after it (R-EXE-1: 15 min).
    pub window_min: f64,
    pub poll_ms: u64,
    /// Peg slip beyond the touch in the cost model (paper_entry.slip_bps, H-EXE1).
    pub slip_bps: f64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LiveApprovalCfg {
    /// The first N live trades need the user's approval; then the gate switches to auto.
    pub manual_trades: u32,
    /// A pending proposal expires after this many minutes.
    pub ttl_min: f64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LiveSmallCfg {
    pub max_loss_per_trade_usd: f64,
    pub risk_to_stop_usd: f64,
    pub max_entries_per_day: u32,
    pub daily_loss_stop_usd: f64,
    /// Also refuse an entry whose worst case (realised + every open trade's max loss + this one) passes the stop.
    pub daily_loss_budget: bool,
    pub max_open_positions: u32,
    pub max_same_direction: u32,
    pub max_gross_notional_usd: f64,
    pub max_stop_distance_pct: f64,
    /// Inclusive EV gate (reject net E[R] < min), live costs.
    pub min_net_expected_r: f64,
    pub require_sigma: bool,
    pub excluded_symbols: Vec<String>,
    /// Bybit symbolType values never traded (e.g. "innovation"); "" is crypto.
    #[serde(default)]
    pub excluded_symbol_types: Vec<String>,
    pub tiers: Vec<LiveTier>,
    pub depth: LiveDepthCfg,
    pub correlation: LiveCorrCfg,
    pub entry: LiveEntryCfg,
    pub stop_trigger_by: String,
    pub leverage: f64,
    /// Required free margin = notional / leverage * this.
    pub margin_buffer: f64,
    pub approval: LiveApprovalCfg,
    /// orderLinkId prefix of every bot order (identifies the bot's orders on the venue).
    pub link_prefix: String,
}

#[derive(Deserialize)]
struct RiskYamlLive {
    live_small: LiveSmallCfg,
}

impl LiveSmallCfg {
    /// `config/risk.yaml` `live_small`, validated against [`LIVE_CEILINGS`].
    pub fn load(root: &Path) -> anyhow::Result<Self> {
        let p = root.join("config").join("risk.yaml");
        let text = std::fs::read_to_string(&p).with_context(|| p.display().to_string())?;
        Self::from_risk_yaml(&text).with_context(|| format!("{}: live_small", p.display()))
    }

    pub fn from_risk_yaml(text: &str) -> anyhow::Result<Self> {
        let c = serde_yaml::from_str::<RiskYamlLive>(text)?.live_small;
        c.validate()?;
        Ok(c)
    }

    pub fn validate(&self) -> anyhow::Result<()> {
        let k = LIVE_CEILINGS;
        let pos = |x: f64| x.is_finite() && x > 0.0;
        ensure!(pos(self.max_loss_per_trade_usd) && self.max_loss_per_trade_usd <= k.max_loss_per_trade_usd, "max_loss_per_trade_usd must be in (0, {}]", k.max_loss_per_trade_usd);
        ensure!(pos(self.risk_to_stop_usd) && self.risk_to_stop_usd <= self.max_loss_per_trade_usd, "risk_to_stop_usd must be in (0, max_loss_per_trade_usd]");
        ensure!(self.max_entries_per_day <= k.max_entries_per_day, "max_entries_per_day must be <= {}", k.max_entries_per_day);
        ensure!(pos(self.daily_loss_stop_usd) && self.daily_loss_stop_usd <= k.daily_loss_stop_usd, "daily_loss_stop_usd must be in (0, {}]", k.daily_loss_stop_usd);
        ensure!(self.max_open_positions <= k.max_open_positions, "max_open_positions must be <= {}", k.max_open_positions);
        ensure!(self.max_same_direction <= k.max_same_direction, "max_same_direction must be <= {}", k.max_same_direction);
        ensure!(pos(self.max_gross_notional_usd) && self.max_gross_notional_usd <= k.max_gross_notional_usd, "max_gross_notional_usd must be in (0, {}]", k.max_gross_notional_usd);
        ensure!(pos(self.max_stop_distance_pct) && self.max_stop_distance_pct <= 15.0, "max_stop_distance_pct must be in (0, 15]");
        ensure!(self.min_net_expected_r.is_finite() && self.min_net_expected_r >= 0.0, "min_net_expected_r must be >= 0");
        ensure!(!self.tiers.is_empty(), "tiers must not be empty");
        for t in &self.tiers {
            ensure!(pos(t.notional_cap_usd) && t.notional_cap_usd <= k.max_tier_notional_usd, "tier {}: notional_cap_usd must be in (0, {}]", t.name, k.max_tier_notional_usd);
            ensure!(t.notional_cap_usd >= k.min_tier_notional_cap_usd, "tier {}: notional cap too small", t.name);
            ensure!(t.stop_slip_bps.is_finite() && t.stop_slip_bps >= 0.0, "tier {}: stop_slip_bps must be >= 0", t.name);
            ensure!(t.min_turnover_usd.is_finite() && t.min_turnover_usd >= 0.0, "tier {}: min_turnover_usd must be >= 0", t.name);
            ensure!(t.max_spread_bps.is_finite() && t.max_spread_bps > 0.0, "tier {}: max_spread_bps must be > 0", t.name);
        }
        ensure!(self.tiers.windows(2).all(|w| w[0].notional_cap_usd >= w[1].notional_cap_usd), "tiers must be ordered from the most liquid (largest cap) down");
        ensure!(pos(self.depth.band_bps) && pos(self.depth.max_share) && self.depth.max_share <= 1.0, "depth: band_bps > 0 and max_share in (0, 1]");
        let c = &self.correlation;
        ensure!(c.threshold > 0.0 && c.threshold <= k.max_btc_corr_threshold, "correlation.threshold must be in (0, {}]", k.max_btc_corr_threshold);
        ensure!(c.window_hours >= 24 && c.min_obs >= 20, "correlation: window_hours >= 24 and min_obs >= 20");
        let e = &self.entry;
        ensure!(matches!(e.time_in_force.as_str(), "PostOnly" | "GTC"), "entry.time_in_force: PostOnly | GTC");
        ensure!(pos(e.window_min) && e.window_min <= k.max_entry_window_min, "entry.window_min must be in (0, {}]", k.max_entry_window_min);
        ensure!(e.poll_ms >= 250, "entry.poll_ms must be >= 250");
        ensure!(e.slip_bps.is_finite() && e.slip_bps >= 0.0, "entry.slip_bps must be >= 0");
        ensure!(matches!(self.stop_trigger_by.as_str(), "MarkPrice" | "LastPrice" | "IndexPrice"), "stop_trigger_by: MarkPrice | LastPrice | IndexPrice");
        ensure!(self.leverage >= 1.0 && self.leverage <= 10.0, "leverage must be in [1, 10]");
        ensure!(self.margin_buffer >= 1.0, "margin_buffer must be >= 1");
        ensure!(self.approval.manual_trades >= k.min_manual_trades, "approval.manual_trades must be >= {}", k.min_manual_trades);
        ensure!(pos(self.approval.ttl_min), "approval.ttl_min must be > 0");
        let p = &self.link_prefix;
        ensure!(!p.is_empty() && p.len() <= 8 && p.chars().all(|c| c.is_ascii_alphanumeric()), "link_prefix: 1-8 ASCII letters/digits");
        for s in ALWAYS_EXCLUDED {
            ensure!(self.excluded_symbols.iter().any(|x| x == s), "excluded_symbols must contain {s}");
        }
        Ok(())
    }

    pub fn is_excluded(&self, symbol: &str) -> bool {
        ALWAYS_EXCLUDED.contains(&symbol) || self.excluded_symbols.iter().any(|x| x == symbol)
    }

    /// Stable hash of the profile (FNV-1a 64 over its JSON): logged with every live event; a change starts a new record.
    pub fn profile_hash(&self) -> String {
        let s = serde_json::to_string(self).unwrap_or_default();
        let h = s.bytes().fold(0xcbf2_9ce4_8422_2325_u64, |h, b| (h ^ b as u64).wrapping_mul(0x0100_0000_01b3));
        format!("{h:016x}")
    }

    /// Tier of a symbol from its 24h-turnover rank (1 = largest), turnover and quoted spread (bp). The first (most liquid)
    /// tier whose every condition holds; None: below every tier (too illiquid).
    pub fn tier_for(&self, rank: u32, turnover_usd: f64, spread_bps: f64) -> Option<&LiveTier> {
        self.tiers.iter().find(|t| t.max_rank.is_none_or(|m| rank <= m) && turnover_usd >= t.min_turnover_usd && spread_bps <= t.max_spread_bps)
    }

    /// The bot's orderLinkId prefix including the separator.
    pub fn link_tag(&self) -> String {
        format!("{}-", self.link_prefix)
    }
}

// ---------------------------------------------------------------- inputs

/// Per-symbol market facts gathered by `atx live` (venue rules, books, turnover rank, sector, BTC correlation).
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct SymbolFacts {
    pub symbol: String,
    pub rules: Option<InstrumentRules>,
    /// Trading, LinearPerpetual, quote and settle USDT.
    pub usdt_linear_perp: bool,
    /// Bybit symbolType ("" = crypto; ETF, stock, commodity, innovation...).
    #[serde(default)]
    pub symbol_type: String,
    pub turnover_24h: f64,
    pub rank: u32,
    pub best_bid: f64,
    pub best_ask: f64,
    /// USD depth within `depth.band_bps` of the touch.
    pub bid_depth_usd: f64,
    pub ask_depth_usd: f64,
    pub sector: Option<String>,
    /// Rolling correlation to BTC; None = unknown (counts as correlated).
    pub btc_corr: Option<f64>,
    /// A position or open order on this symbol that the bot did not open: never trade it.
    pub foreign_activity: bool,
}

/// A bot position or resting bot entry.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LivePos {
    pub symbol: String,
    pub side: Side,
    pub qty: f64,
    pub entry_price: f64,
    pub mark: f64,
    /// Unrealised P&L plus the position's realised part (fees, funding) in USD. 0 for a pending entry.
    pub open_pnl_usd: f64,
    /// Planned worst case of this trade (max_loss at entry).
    pub max_loss_usd: f64,
    pub pending: bool,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct LiveAccount {
    /// Free USDT margin.
    pub available_usd: f64,
    /// Bot entries submitted since the UTC day start (filled or not).
    pub entries_today: u32,
    /// Net realised P&L of bot trades closed since the UTC day start (fees included).
    pub realised_today_usd: f64,
    pub kill: bool,
}

// ---------------------------------------------------------------- output

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LiveRejection {
    KillSwitch,
    NotLiveMode,
    MissingVersions,
    InvalidIntent,
    NotPeggedEntry,
    ExcludedSymbol,
    NotUsdtLinearPerp,
    ForeignActivityOnSymbol,
    UnknownInstrument,
    NoBook,
    TooIlliquid,
    DuplicateSymbol,
    MaxEntriesPerDay,
    DailyLossStop,
    DailyLossBudget,
    MaxOpenPositions,
    MaxSameDirection,
    MaxGrossNotional,
    CorrelatedHedge,
    StopTooFar,
    StopInsideMinSigma,
    StopBeyondMaxSigma,
    MissingSigma,
    NetErBelowMin,
    NetErMissing,
    MinLotOverMaxLoss,
    MaxLossPerTrade,
    InsufficientMargin,
}

impl LiveRejection {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::KillSwitch => "kill_switch",
            Self::NotLiveMode => "not_live_mode",
            Self::MissingVersions => "missing_versions",
            Self::InvalidIntent => "invalid_intent",
            Self::NotPeggedEntry => "not_pegged_entry",
            Self::ExcludedSymbol => "excluded_symbol",
            Self::NotUsdtLinearPerp => "not_usdt_linear_perp",
            Self::ForeignActivityOnSymbol => "foreign_activity_on_symbol",
            Self::UnknownInstrument => "unknown_instrument",
            Self::NoBook => "no_book",
            Self::TooIlliquid => "too_illiquid",
            Self::DuplicateSymbol => "duplicate_symbol",
            Self::MaxEntriesPerDay => "max_entries_per_day",
            Self::DailyLossStop => "daily_loss_stop",
            Self::DailyLossBudget => "daily_loss_budget",
            Self::MaxOpenPositions => "max_open_positions",
            Self::MaxSameDirection => "max_same_direction",
            Self::MaxGrossNotional => "max_gross_notional",
            Self::CorrelatedHedge => "correlated_hedge",
            Self::StopTooFar => "stop_too_far",
            Self::StopInsideMinSigma => "stop_inside_min_sigma",
            Self::StopBeyondMaxSigma => "stop_beyond_max_sigma",
            Self::MissingSigma => "missing_sigma",
            Self::NetErBelowMin => "net_er_below_min",
            Self::NetErMissing => "net_er_missing",
            Self::MinLotOverMaxLoss => "min_lot_over_max_loss",
            Self::MaxLossPerTrade => "max_loss_per_trade",
            Self::InsufficientMargin => "insufficient_margin",
        }
    }
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct LiveDecision {
    pub intent_id: String,
    /// accept | reject
    pub decision: String,
    pub rejections: Vec<String>,
    pub details: BTreeMap<String, String>,
    pub symbol: String,
    pub side: Option<Side>,
    pub tier: Option<String>,
    pub notional_cap_usd: f64,
    pub qty: f64,
    pub entry: f64,
    pub stop: f64,
    pub target: Option<f64>,
    pub notional_usd: f64,
    pub risk_to_stop_usd: f64,
    pub max_loss_usd: f64,
    pub stop_slip_bps: f64,
    pub stop_pct: Option<f64>,
    pub stop_sigma: Option<f64>,
    pub net_er_r: Option<f64>,
    pub min_lot_used: bool,
    /// Correlation cluster labels of the symbol (sector, "btc_beta").
    pub clusters: Vec<String>,
    /// Day P&L used by the daily stop: realised + open losses.
    pub day_pnl_usd: f64,
    pub open_after: u32,
    pub same_direction_after: u32,
    pub gross_after_usd: f64,
}

impl LiveDecision {
    pub fn accepted(&self) -> bool {
        self.decision == "accept"
    }

    fn reject(&mut self, r: LiveRejection, detail: String) {
        let k = r.as_str().to_string();
        if !self.rejections.contains(&k) {
            self.rejections.push(k.clone());
            self.details.insert(k, detail);
        }
    }

    pub fn has(&self, r: LiveRejection) -> bool {
        self.rejections.iter().any(|x| x == r.as_str())
    }
}

// ---------------------------------------------------------------- maths

/// Per-unit loss at the stop, fees included: |e - s| + e * entry_bps + s * exit_bps.
pub fn unit_loss(entry: f64, stop: f64, entry_bps: f64, exit_bps: f64) -> f64 {
    (entry - stop).abs() + entry * entry_bps / 1e4 + stop * exit_bps / 1e4
}

/// Entry leg of a pegged PostOnly entry: maker fee plus the peg slip.
pub fn entry_bps(fees: &FeeRates, cfg: &LiveSmallCfg) -> f64 {
    fees.maker_bps + cfg.entry.slip_bps
}

/// Worst-case loss of `qty` at the stop with the tier's stop slip (the $2 check).
pub fn max_loss_usd(qty: f64, entry: f64, stop: f64, fees: &FeeRates, cfg: &LiveSmallCfg, stop_slip_bps: f64) -> f64 {
    qty * unit_loss(entry, stop, entry_bps(fees, cfg), fees.taker_bps + stop_slip_bps)
}

/// Pearson correlation of two equally long series; None when too short or degenerate.
pub fn pearson(x: &[f64], y: &[f64]) -> Option<f64> {
    let n = x.len().min(y.len());
    if n < 3 {
        return None;
    }
    let (mx, my) = (x[..n].iter().sum::<f64>() / n as f64, y[..n].iter().sum::<f64>() / n as f64);
    let (mut sxy, mut sxx, mut syy) = (0.0, 0.0, 0.0);
    for i in 0..n {
        let (a, b) = (x[i] - mx, y[i] - my);
        sxy += a * b;
        sxx += a * a;
        syy += b * b;
    }
    let r = sxy / (sxx * syy).sqrt();
    r.is_finite().then_some(r)
}

/// Log returns of two close series aligned on timestamp: (ts, close) pairs, ascending.
pub fn aligned_log_returns(a: &[(i64, f64)], b: &[(i64, f64)]) -> (Vec<f64>, Vec<f64>) {
    let bm: HashMap<i64, f64> = b.iter().copied().collect();
    let common: Vec<(f64, f64)> = a.iter().filter_map(|(t, ca)| bm.get(t).map(|cb| (*ca, *cb))).filter(|(x, y)| *x > 0.0 && *y > 0.0).collect();
    let mut ra = Vec::with_capacity(common.len());
    let mut rb = Vec::with_capacity(common.len());
    for w in common.windows(2) {
        ra.push((w[1].0 / w[0].0).ln());
        rb.push((w[1].1 / w[0].1).ln());
    }
    (ra, rb)
}

/// Correlation cluster labels of a symbol: its sector (unless ignored) and "btc_beta" when correlated to BTC above the
/// threshold, when it IS BTC, or when the correlation is unknown (fail closed).
pub fn clusters_of(f: &SymbolFacts, cfg: &LiveSmallCfg) -> Vec<String> {
    let mut out = Vec::new();
    if let Some(s) = f.sector.as_deref().filter(|s| !s.is_empty() && !cfg.correlation.ignore_sectors.iter().any(|x| x == s)) {
        out.push(format!("sector:{s}"));
    }
    if f.symbol == cfg.correlation.btc_symbol || f.btc_corr.is_none_or(|c| c > cfg.correlation.threshold) {
        out.push("btc_beta".into());
    }
    out
}

/// Two symbols are the same bet when they are the same symbol or share a cluster label.
pub fn correlated(a: &SymbolFacts, b: &SymbolFacts, cfg: &LiveSmallCfg) -> bool {
    if a.symbol == b.symbol {
        return true;
    }
    let cb = clusters_of(b, cfg);
    clusters_of(a, cfg).iter().any(|c| cb.contains(c))
}

// ---------------------------------------------------------------- decide

fn finite_pos(x: Option<f64>) -> Option<f64> {
    x.filter(|v| v.is_finite() && *v > 0.0)
}

/// Decide one live entry intent. `entry_px` overrides the intent's reference price (the current touch at submission).
/// `facts` must hold the intent's symbol and every symbol in `positions` (unknown ones count as BTC-correlated).
#[allow(clippy::too_many_arguments)]
pub fn decide_live(
    intent: &OrderIntent,
    entry_px: Option<f64>,
    account: &LiveAccount,
    positions: &[LivePos],
    facts: &HashMap<String, SymbolFacts>,
    fees: &FeeRates,
    placement: &PlacementCfg,
    cfg: &LiveSmallCfg,
) -> LiveDecision {
    use LiveRejection as R;
    let mut d = LiveDecision { intent_id: intent.intent_id.clone(), decision: "reject".into(), symbol: intent.symbol.clone(), ..Default::default() };

    // ---- what the intent is
    if account.kill {
        d.reject(R::KillSwitch, "data/live/KILL present: no new entries".into());
    }
    if intent.mode.as_deref() != Some("live") {
        d.reject(R::NotLiveMode, format!("intent mode {:?}: the live desk takes only mode \"live\"", intent.mode));
    }
    let ctx = intent.context.as_ref();
    let ver = |k: &str| ctx.and_then(|c| c.get(k)).and_then(|v| v.as_str()).is_some_and(|s| !s.trim().is_empty());
    if !(ver("model_id") && ver("prompt_version")) {
        d.reject(R::MissingVersions, "context.model_id and context.prompt_version are required (principle 8)".into());
    }
    let Some(side) = intent.side.side() else {
        d.reject(R::InvalidIntent, "side flat: exits go through reduce-only closes, not propose".into());
        return d;
    };
    d.side = Some(side);
    match &intent.entry {
        Some(e) if e.kind == EntryType::Limit && e.peg.as_deref() == Some("touch") => {}
        _ => d.reject(R::NotPeggedEntry, "entries must be {type: limit, peg: touch} (R-EXE-1)".into()),
    }
    let sym = intent.symbol.as_str();
    if cfg.is_excluded(sym) {
        d.reject(R::ExcludedSymbol, format!("{sym} is excluded: never touched"));
    }
    let fact = facts.get(sym);
    match fact {
        None => d.reject(R::UnknownInstrument, format!("no market facts for {sym}")),
        Some(f) => {
            if !f.usdt_linear_perp {
                d.reject(R::NotUsdtLinearPerp, format!("{sym} is not a Trading USDT linear perpetual"));
            }
            if cfg.excluded_symbol_types.contains(&f.symbol_type) {
                d.reject(R::ExcludedSymbol, format!("{sym}: symbol type {:?} is excluded", f.symbol_type));
            }
            if f.foreign_activity {
                d.reject(R::ForeignActivityOnSymbol, format!("{sym} has a position or order the bot did not open: left alone"));
            }
            if f.rules.is_none() {
                d.reject(R::UnknownInstrument, format!("no instrument rules for {sym}"));
            }
        }
    }

    // ---- account and book limits
    let bot: Vec<&LivePos> = positions.iter().collect();
    if bot.iter().any(|p| p.symbol == sym) {
        d.reject(R::DuplicateSymbol, format!("{sym} already has a bot position or resting entry"));
    }
    if account.entries_today >= cfg.max_entries_per_day {
        d.reject(R::MaxEntriesPerDay, format!("{} entries today, max {}", account.entries_today, cfg.max_entries_per_day));
    }
    let open_loss: f64 = bot.iter().filter(|p| !p.pending).map(|p| p.open_pnl_usd.min(0.0)).sum();
    d.day_pnl_usd = account.realised_today_usd + open_loss;
    if d.day_pnl_usd <= -cfg.daily_loss_stop_usd {
        d.reject(
            R::DailyLossStop,
            format!("day P&L {:.2} (realised {:.2} + open losses {:.2}) <= -{}", d.day_pnl_usd, account.realised_today_usd, open_loss, cfg.daily_loss_stop_usd),
        );
    }
    let n_open = bot.len() as u32;
    if n_open >= cfg.max_open_positions {
        d.reject(R::MaxOpenPositions, format!("{n_open} bot positions or resting entries, max {}", cfg.max_open_positions));
    }
    let same = bot.iter().filter(|p| p.side == side).count() as u32;
    if same >= cfg.max_same_direction {
        d.reject(R::MaxSameDirection, format!("{same} bot positions already {side:?}, max {}", cfg.max_same_direction));
    }
    if let Some(f) = fact {
        for p in bot.iter().filter(|p| !p.pending && p.side != side && p.open_pnl_usd < 0.0) {
            let pf = facts.get(&p.symbol).cloned().unwrap_or_else(|| SymbolFacts { symbol: p.symbol.clone(), ..Default::default() });
            if correlated(f, &pf, cfg) {
                d.reject(
                    R::CorrelatedHedge,
                    format!(
                        "{sym} {side:?} would hedge the losing {} {:?} ({:+.2} USD; clusters {:?} / {:?}): no hedging to escape",
                        p.symbol,
                        p.side,
                        p.open_pnl_usd,
                        clusters_of(f, cfg),
                        clusters_of(&pf, cfg)
                    ),
                );
            }
        }
        d.clusters = clusters_of(f, cfg);
    }

    // ---- geometry
    let ref_px = entry_px.or_else(|| intent.entry.as_ref().and_then(|e| e.price));
    let (Some(f), Some(re), Some(rs)) = (fact, finite_pos(ref_px), finite_pos(intent.stop)) else {
        d.reject(R::InvalidIntent, "entry reference price and stop must be positive numbers".into());
        return d;
    };
    let Some(rule) = f.rules.clone() else { return d };
    if let Some(p) = intent.p_tp
        && !(0.0..=1.0).contains(&p)
    {
        d.reject(R::InvalidIntent, format!("p_tp={p} outside [0, 1]"));
    }
    let s = side.sign();
    let t = rule.tick_size;
    let rnd = if side == Side::Long { Round::Down } else { Round::Up };
    let entry = round_to_step(re, t, Round::Nearest);
    let stop = round_to_step(rs, t, rnd);
    let target = intent.targets.first().copied().filter(|x| x.is_finite() && *x > 0.0).map(|x| round_to_step(x, t, rnd));
    (d.entry, d.stop, d.target) = (entry, stop, target);
    if (entry - stop) * s <= 0.0 {
        d.reject(R::InvalidIntent, format!("stop {stop} is not on the losing side of entry {entry} for a {side:?}"));
        return d;
    }
    if let Some(tp) = target
        && (tp - entry) * s <= 0.0
    {
        d.reject(R::InvalidIntent, format!("target {tp} is not on the winning side of entry {entry}"));
        return d;
    }
    let stop_pct = (entry - stop).abs() / entry * 100.0;
    d.stop_pct = Some(stop_pct);
    if stop_pct > cfg.max_stop_distance_pct {
        d.reject(R::StopTooFar, format!("stop {stop_pct:.3}% > max {}%", cfg.max_stop_distance_pct));
    }
    match finite_pos(intent.sigma.as_ref().and_then(|x| x.per_bar)) {
        None if cfg.require_sigma => d.reject(R::MissingSigma, "sigma.per_bar missing or not positive".into()),
        None => {}
        Some(sig) => {
            let k = (entry / stop).ln().abs() / sig;
            d.stop_sigma = Some(k);
            if k < placement.min_stop_distance_sigma {
                d.reject(R::StopInsideMinSigma, format!("stop {k:.3} sigma < min {}", placement.min_stop_distance_sigma));
            }
            if k > placement.max_stop_distance_sigma {
                d.reject(R::StopBeyondMaxSigma, format!("stop {k:.3} sigma > max {}", placement.max_stop_distance_sigma));
            }
        }
    }

    // ---- tier and notional cap
    let depth = f.bid_depth_usd.min(f.ask_depth_usd);
    if !(f.best_bid > 0.0 && f.best_ask >= f.best_bid && depth.is_finite() && depth > 0.0) {
        d.reject(R::NoBook, format!("{sym}: no usable order book"));
        return d;
    }
    let spread_bps = (f.best_ask - f.best_bid) / ((f.best_ask + f.best_bid) / 2.0) * 1e4;
    let tier = cfg.tier_for(f.rank, f.turnover_24h, spread_bps).cloned();
    let Some(tier) = tier else {
        d.reject(R::TooIlliquid, format!("{sym}: rank {} turnover {:.0} USD spread {spread_bps:.2} bp is below every tier", f.rank, f.turnover_24h));
        return d;
    };
    d.tier = Some(tier.name.clone());
    d.stop_slip_bps = tier.stop_slip_bps;
    let cap = tier.notional_cap_usd.min(cfg.depth.max_share * depth);
    d.notional_cap_usd = cap;

    // ---- costs, EV
    let ebps = entry_bps(fees, cfg);
    let plan_unit = unit_loss(entry, stop, ebps, fees.taker_bps + fees.slippage_bps);
    let rt_cost = entry * (ebps + fees.taker_bps + fees.slippage_bps) / 1e4;
    let engine_er = match (intent.p_tp, target) {
        (Some(p), Some(tp)) => Some(p * ((tp - entry).abs() - rt_cost) / plan_unit - (1.0 - p)),
        _ => None,
    };
    let er = match (engine_er, intent.net_er_r.filter(|x| x.is_finite())) {
        (Some(a), Some(b)) => Some(a.min(b)),
        (a, b) => a.or(b),
    };
    d.net_er_r = er;
    match er {
        None => d.reject(R::NetErMissing, "need p_tp with a target, or net_er_r".into()),
        Some(x) if x < cfg.min_net_expected_r => d.reject(R::NetErBelowMin, format!("net E[R] {x:.4} < min {}", cfg.min_net_expected_r)),
        _ => {}
    }

    // ---- size
    let max_qty = rule.max_qty;
    let mut qty = round_to_step((cfg.risk_to_stop_usd / plan_unit).min(cap / entry).min(max_qty), rule.qty_step, Round::Down);
    if !(qty >= rule.min_qty && qty > 0.0 && qty * entry >= rule.min_notional) {
        let min_lot = round_to_step(rule.min_qty.max(rule.min_notional / entry), rule.qty_step, Round::Up);
        let ml = max_loss_usd(min_lot, entry, stop, fees, cfg, tier.stop_slip_bps);
        if ml > cfg.max_loss_per_trade_usd || min_lot * entry > cap {
            d.reject(
                R::MinLotOverMaxLoss,
                format!(
                    "minimum lot {min_lot} ({:.2} USD) would lose {ml:.2} USD at the stop (cap {}) or exceed the notional cap {cap:.2}",
                    min_lot * entry,
                    cfg.max_loss_per_trade_usd
                ),
            );
            return d;
        }
        qty = min_lot;
        d.min_lot_used = true;
    }
    d.qty = qty;
    d.notional_usd = qty * entry;
    d.risk_to_stop_usd = qty * plan_unit;
    d.max_loss_usd = max_loss_usd(qty, entry, stop, fees, cfg, tier.stop_slip_bps);
    if d.max_loss_usd > cfg.max_loss_per_trade_usd + 1e-9 {
        d.reject(R::MaxLossPerTrade, format!("worst case {:.4} USD > {}", d.max_loss_usd, cfg.max_loss_per_trade_usd));
    }
    if cfg.daily_loss_budget {
        let open_worst: f64 = bot.iter().map(|p| p.max_loss_usd.max(0.0)).sum();
        let worst_day = account.realised_today_usd - open_worst - d.max_loss_usd;
        if worst_day < -cfg.daily_loss_stop_usd {
            d.reject(
                R::DailyLossBudget,
                format!(
                    "realised {:.2} - open worst cases {open_worst:.2} - this trade {:.2} = {worst_day:.2} < -{}",
                    account.realised_today_usd, d.max_loss_usd, cfg.daily_loss_stop_usd
                ),
            );
        }
    }
    let gross_now: f64 = bot.iter().map(|p| p.qty * if p.pending || p.mark <= 0.0 { p.entry_price } else { p.mark }).sum();
    d.gross_after_usd = gross_now + d.notional_usd;
    if d.gross_after_usd > cfg.max_gross_notional_usd + 1e-9 {
        d.reject(R::MaxGrossNotional, format!("gross {:.2} + {:.2} = {:.2} USD > max {}", gross_now, d.notional_usd, d.gross_after_usd, cfg.max_gross_notional_usd));
    }
    let margin = d.notional_usd / cfg.leverage * cfg.margin_buffer;
    if account.available_usd < margin {
        d.reject(R::InsufficientMargin, format!("available {:.2} USD < required {margin:.2}", account.available_usd));
    }
    d.open_after = n_open + 1;
    d.same_direction_after = same + 1;
    if d.rejections.is_empty() {
        d.decision = "accept".into();
    }
    d
}

/// While re-pegging: the most a pegged entry may chase before it is cancelled. True when `qty` at `price` still keeps the
/// worst case under the cap and the stop on the losing side.
pub fn peg_price_ok(side: Side, qty: f64, price: f64, stop: f64, fees: &FeeRates, cfg: &LiveSmallCfg, stop_slip_bps: f64) -> bool {
    (price - stop) * side.sign() > 0.0 && max_loss_usd(qty, price, stop, fees, cfg, stop_slip_bps) <= cfg.max_loss_per_trade_usd + 1e-9
}

/// Quantity allowed when a pegged entry chases to `price`: never more than the decided quantity, never a risk to the
/// stop above `risk_to_stop_usd`, never a notional above `notional0` (the caller passes min(tier / depth cap, decided
/// notional + gross headroom), so the tier and gross caps keep holding while the peg moves).
/// Rounded down to the lot step; 0 when the stop is no longer on the losing side.
#[allow(clippy::too_many_arguments)]
pub fn chase_qty(side: Side, qty0: f64, notional0: f64, price: f64, stop: f64, fees: &FeeRates, cfg: &LiveSmallCfg, qty_step: f64) -> f64 {
    if (price - stop) * side.sign() <= 0.0 || price <= 0.0 {
        return 0.0;
    }
    let unit = unit_loss(price, stop, entry_bps(fees, cfg), fees.taker_bps + fees.slippage_bps);
    round_to_step(qty0.min(cfg.risk_to_stop_usd / unit).min(notional0 / price), qty_step, Round::Down)
}

/// Intent side as a venue side (None for flat).
pub fn intent_side(s: IntentSide) -> Option<Side> {
    s.side()
}

// ---------------------------------------------------------------- tests

#[cfg(test)]
mod tests {
    use super::*;

    fn root() -> std::path::PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
    }

    fn cfg() -> LiveSmallCfg {
        LiveSmallCfg::load(&root()).expect("project live_small profile loads")
    }

    fn fees() -> FeeRates {
        FeeRates { maker_bps: 2.0, taker_bps: 5.5, slippage_bps: 2.0 }
    }

    fn placement() -> PlacementCfg {
        PlacementCfg { min_stop_distance_sigma: 0.5, max_stop_distance_sigma: 4.0, stop_cap_bounds_sigma: Some([2.5, 6.0]) }
    }

    fn rules(sym: &str, tick: f64, step: f64) -> InstrumentRules {
        InstrumentRules { symbol: sym.into(), tick_size: tick, qty_step: step, min_qty: step, max_qty: 1e6, max_mkt_qty: 1e5, min_notional: 5.0, max_leverage: 50.0 }
    }

    fn fact(sym: &str, px: f64, tick: f64, step: f64, rank: u32, sector: &str, corr: Option<f64>) -> SymbolFacts {
        SymbolFacts {
            symbol: sym.into(),
            rules: Some(rules(sym, tick, step)),
            usdt_linear_perp: true,
            symbol_type: String::new(),
            turnover_24h: if rank <= 10 {
                1e9
            } else if rank <= 40 {
                1e8
            } else {
                5e6
            },
            rank,
            best_bid: px,
            best_ask: px + tick,
            bid_depth_usd: 5e6,
            ask_depth_usd: 5e6,
            sector: Some(sector.into()),
            btc_corr: corr,
            foreign_activity: false,
        }
    }

    fn facts() -> HashMap<String, SymbolFacts> {
        [
            fact("BTCUSDT", 64000.0, 0.1, 0.001, 1, "btc", Some(1.0)),
            fact("ETHUSDT", 3000.0, 0.01, 0.01, 2, "l1", Some(0.85)),
            fact("SOLUSDT", 150.0, 0.01, 0.1, 3, "l1", Some(0.8)),
            fact("XMRUSDT", 300.0, 0.01, 0.01, 30, "privacy", Some(0.3)),
            fact("ZECUSDT", 50.0, 0.01, 0.1, 45, "privacy", Some(0.35)),
            fact("TAOUSDT", 400.0, 0.01, 0.01, 35, "ai", Some(0.4)),
            fact("SPYUSDT", 600.0, 0.01, 0.01, 60, "", Some(0.2)),
        ]
        .into_iter()
        .map(|f| (f.symbol.clone(), f))
        .collect()
    }

    /// Long at `px`, stop `stop_pct`% below, target 3x the stop distance, sigma such that the stop is 2 sigma.
    fn intent(sym: &str, side: &str, px: f64, stop_pct: f64) -> OrderIntent {
        let sgn = if side == "long" { 1.0 } else { -1.0 };
        let stop = px * (1.0 - sgn * stop_pct / 100.0);
        let tp = px * (1.0 + sgn * 3.0 * stop_pct / 100.0);
        let sig = (px / stop).ln().abs() / 2.0;
        serde_json::from_value(serde_json::json!({
            "intent_id": format!("{sym}-{side}"), "ts_ms": 0, "mode": "live", "strategy": "shock_regime", "family": "shock",
            "symbol": sym, "side": side, "entry": {"type": "limit", "price": px, "peg": "touch"}, "stop": stop, "targets": [tp],
            "sigma": {"per_bar": sig}, "p_tp": 0.4, "net_er_r": 0.5,
            "context": {"model_id": "claude-opus-5-5", "prompt_version": "v0.1"}
        }))
        .unwrap()
    }

    fn acct() -> LiveAccount {
        LiveAccount { available_usd: 1000.0, entries_today: 0, realised_today_usd: 0.0, kill: false }
    }

    fn pos(sym: &str, side: Side, qty: f64, px: f64, pnl: f64) -> LivePos {
        LivePos { symbol: sym.into(), side, qty, entry_price: px, mark: px, open_pnl_usd: pnl, max_loss_usd: 2.0, pending: false }
    }

    fn run(i: &OrderIntent, a: &LiveAccount, p: &[LivePos]) -> LiveDecision {
        decide_live(i, None, a, p, &facts(), &fees(), &placement(), &cfg())
    }

    #[test]
    fn project_profile_matches_the_users_limits() {
        let c = cfg();
        assert_eq!(c.max_loss_per_trade_usd, 2.0);
        assert_eq!(c.risk_to_stop_usd, 1.7);
        assert_eq!((c.max_entries_per_day, c.max_open_positions, c.max_same_direction), (20, 3, 2));
        assert_eq!((c.daily_loss_stop_usd, c.max_gross_notional_usd), (10.0, 400.0));
        assert!(c.is_excluded("SPYUSDT"));
        assert_eq!(c.approval.manual_trades, 5);
        assert_eq!(c.entry.window_min, 15.0);
        assert_eq!(c.tiers.first().map(|t| t.notional_cap_usd), Some(200.0));
        assert_eq!(c.tiers.last().map(|t| t.notional_cap_usd), Some(50.0));
        assert_eq!(c.profile_hash(), cfg().profile_hash(), "hash is stable");
    }

    #[test]
    fn looser_yaml_than_the_ceilings_is_refused() {
        let text = std::fs::read_to_string(root().join("config/risk.yaml")).unwrap();
        for (from, to) in [
            ("max_loss_per_trade_usd: 2.00", "max_loss_per_trade_usd: 2.50"),
            ("max_entries_per_day: 20", "max_entries_per_day: 21"),
            ("daily_loss_stop_usd: 10.0", "daily_loss_stop_usd: 12.0"),
            ("max_open_positions: 3", "max_open_positions: 4"),
            ("max_same_direction: 2", "max_same_direction: 3"),
            ("max_gross_notional_usd: 400.0", "max_gross_notional_usd: 500.0"),
            ("notional_cap_usd: 200", "notional_cap_usd: 250"),
            ("manual_trades: 5", "manual_trades: 4"),
            ("excluded_symbols: [SPYUSDT]", "excluded_symbols: []"),
            ("risk_to_stop_usd: 1.70", "risk_to_stop_usd: 2.10"),
            ("window_min: 15", "window_min: 30"),
        ] {
            assert!(text.contains(from), "risk.yaml should contain {from:?}");
            assert!(LiveSmallCfg::from_risk_yaml(&text.replace(from, to)).is_err(), "{to} must be refused");
        }
    }

    #[test]
    fn sizing_keeps_risk_to_stop_and_worst_case_under_the_caps() {
        // BTC long 64000, stop 1% -> risk-limited (1.70 / ~650 per BTC = 0.0026 -> 0.002)
        let d = run(&intent("BTCUSDT", "long", 64000.0, 1.0), &acct(), &[]);
        assert!(d.accepted(), "{:?}", d.details);
        assert_eq!(d.tier.as_deref(), Some("liquid"));
        assert!(d.risk_to_stop_usd <= 1.70 + 1e-9, "{}", d.risk_to_stop_usd);
        assert!(d.max_loss_usd <= 2.0, "{}", d.max_loss_usd);
        assert_eq!(d.qty, 0.002);
        // ETH long, stop 0.3%: notional-limited at the liquid cap 200
        let d = run(&intent("ETHUSDT", "long", 3000.0, 0.3), &acct(), &[]);
        assert!(d.accepted(), "{:?}", d.details);
        assert!(d.notional_usd <= 200.0 + 1e-9 && d.notional_usd >= 180.0 - 1e-9, "{}", d.notional_usd);
        // XMR (rank 30 -> mid, 100 cap) and ZEC (rank 45 -> illiquid, 50 cap)
        let d = run(&intent("XMRUSDT", "short", 300.0, 0.3), &acct(), &[]);
        assert_eq!(d.tier.as_deref(), Some("mid"));
        assert!(d.notional_usd <= 100.0 + 1e-9 && d.notional_usd > 95.0, "{}", d.notional_usd);
        let d = run(&intent("ZECUSDT", "long", 50.0, 0.3), &acct(), &[]);
        assert_eq!(d.tier.as_deref(), Some("illiquid"));
        assert!(d.notional_usd <= 50.0 + 1e-9 && d.notional_usd > 45.0, "{}", d.notional_usd);
        assert!(d.max_loss_usd <= 2.0);
    }

    #[test]
    fn depth_share_caps_the_notional() {
        let mut fs = facts();
        fs.get_mut("XMRUSDT").unwrap().bid_depth_usd = 200.0; // 20% of 200 = 40 USD cap
        let d = decide_live(&intent("XMRUSDT", "long", 300.0, 0.3), None, &acct(), &[], &fs, &fees(), &placement(), &cfg());
        assert!(d.notional_cap_usd <= 40.0 + 1e-9, "{}", d.notional_cap_usd);
        assert!(d.notional_usd <= 40.0 + 1e-9);
    }

    #[test]
    fn minimum_lot_over_the_two_dollar_cap_is_skipped() {
        // BTC min lot 0.001 = 64 USD; stop 4% -> ~2.6 USD + costs > 2
        let d = run(&intent("BTCUSDT", "long", 64000.0, 4.0), &acct(), &[]);
        assert!(d.has(LiveRejection::MinLotOverMaxLoss), "{:?}", d.details);
        // stop 2.8%: 1.70 budget rounds to 0 lots but the min lot loses ~1.92 <= 2 -> taken at the min lot
        let d = run(&intent("BTCUSDT", "long", 64000.0, 2.8), &acct(), &[]);
        assert!(d.accepted(), "{:?}", d.details);
        assert!(d.min_lot_used && d.qty == 0.001 && d.max_loss_usd <= 2.0, "{d:?}");
    }

    #[test]
    fn entries_per_day_limit() {
        let mut a = acct();
        a.entries_today = 20;
        assert!(run(&intent("ETHUSDT", "long", 3000.0, 1.0), &a, &[]).has(LiveRejection::MaxEntriesPerDay));
        a.entries_today = 19;
        assert!(run(&intent("ETHUSDT", "long", 3000.0, 1.0), &a, &[]).accepted());
    }

    #[test]
    fn daily_loss_stop_counts_realised_and_open_losses() {
        let mut a = acct();
        a.realised_today_usd = -8.0;
        let p = [pos("XMRUSDT", Side::Long, 0.3, 300.0, -2.0)];
        let d = run(&intent("ETHUSDT", "short", 3000.0, 1.0), &a, &p);
        assert!(d.has(LiveRejection::DailyLossStop), "{:?}", d.details);
        // open profits do not offset: realised -10 with a winner open is still stopped
        a.realised_today_usd = -10.0;
        let p = [pos("XMRUSDT", Side::Long, 0.3, 300.0, 5.0)];
        assert!(run(&intent("ETHUSDT", "short", 3000.0, 1.0), &a, &p).has(LiveRejection::DailyLossStop));
        a.realised_today_usd = -3.0;
        assert!(!run(&intent("ETHUSDT", "short", 3000.0, 1.0), &a, &p).has(LiveRejection::DailyLossStop));
    }

    #[test]
    fn daily_loss_budget_refuses_an_entry_that_could_pass_the_stop() {
        let mut a = acct();
        a.realised_today_usd = -7.0;
        let p = [pos("XMRUSDT", Side::Long, 0.3, 300.0, 0.0)]; // worst case 2.0 open
        let d = run(&intent("ETHUSDT", "short", 3000.0, 1.0), &a, &p);
        assert!(d.has(LiveRejection::DailyLossBudget), "{:?}", d.details);
        assert!(!d.has(LiveRejection::DailyLossStop));
        a.realised_today_usd = -5.0;
        assert!(run(&intent("ETHUSDT", "short", 3000.0, 1.0), &a, &p).accepted());
    }

    #[test]
    fn max_three_open_including_resting_entries() {
        let mut p = vec![pos("XMRUSDT", Side::Long, 0.1, 300.0, 1.0), pos("ETHUSDT", Side::Short, 0.01, 3000.0, 1.0)];
        p.push(LivePos { pending: true, ..pos("TAOUSDT", Side::Short, 0.1, 400.0, 0.0) });
        let d = run(&intent("ZECUSDT", "long", 50.0, 1.0), &acct(), &p);
        assert!(d.has(LiveRejection::MaxOpenPositions), "{:?}", d.details);
        p.pop();
        assert!(run(&intent("ZECUSDT", "long", 50.0, 1.0), &acct(), &p).accepted());
    }

    #[test]
    fn max_two_in_the_same_direction() {
        let p = [pos("XMRUSDT", Side::Long, 0.1, 300.0, 1.0), pos("ETHUSDT", Side::Long, 0.01, 3000.0, 1.0)];
        let d = run(&intent("ZECUSDT", "long", 50.0, 1.0), &acct(), &p);
        assert!(d.has(LiveRejection::MaxSameDirection), "{:?}", d.details);
        assert!(!run(&intent("ZECUSDT", "short", 50.0, 1.0), &acct(), &p).has(LiveRejection::MaxSameDirection));
    }

    #[test]
    fn gross_notional_cap() {
        let p = [pos("BTCUSDT", Side::Long, 0.002, 64000.0, 1.0), pos("XMRUSDT", Side::Short, 0.6, 300.0, 1.0)]; // 128 + 180
        let d = run(&intent("ETHUSDT", "short", 3000.0, 0.3), &acct(), &p);
        assert!(d.has(LiveRejection::MaxGrossNotional), "{:?}", d.details);
        assert!(d.gross_after_usd > 400.0);
        // a smaller trade that fits is fine
        let d = run(&intent("ZECUSDT", "long", 50.0, 1.0), &acct(), &p);
        assert!(!d.has(LiveRejection::MaxGrossNotional), "{:?}", d.details);
    }

    #[test]
    fn correlated_hedge_of_a_loser_is_refused() {
        // long BTC losing; short ETH (btc_beta) is a hedge -> refused
        let p = [pos("BTCUSDT", Side::Long, 0.002, 64000.0, -0.5)];
        let d = run(&intent("ETHUSDT", "short", 3000.0, 1.0), &acct(), &p);
        assert!(d.has(LiveRejection::CorrelatedHedge), "{:?}", d.details);
        // same position winning -> allowed
        let p2 = [pos("BTCUSDT", Side::Long, 0.002, 64000.0, 0.5)];
        assert!(run(&intent("ETHUSDT", "short", 3000.0, 1.0), &acct(), &p2).accepted());
        // uncorrelated short (XMR, corr 0.3, other sector) while BTC loses -> allowed
        assert!(!run(&intent("XMRUSDT", "short", 300.0, 1.0), &acct(), &p).has(LiveRejection::CorrelatedHedge));
        // same sector (privacy) is correlated even with low BTC correlation
        let p3 = [pos("XMRUSDT", Side::Long, 0.1, 300.0, -0.2)];
        assert!(run(&intent("ZECUSDT", "short", 50.0, 1.0), &acct(), &p3).has(LiveRejection::CorrelatedHedge));
        // unknown correlation counts as btc_beta (fail closed)
        let mut fs = facts();
        fs.get_mut("TAOUSDT").unwrap().btc_corr = None;
        let d = decide_live(&intent("TAOUSDT", "short", 400.0, 1.0), None, &acct(), &p, &fs, &fees(), &placement(), &cfg());
        assert!(d.has(LiveRejection::CorrelatedHedge), "{:?}", d.details);
    }

    #[test]
    fn excluded_foreign_and_non_usdt_symbols_are_never_touched() {
        let d = run(&intent("SPYUSDT", "long", 600.0, 1.0), &acct(), &[]);
        assert!(d.has(LiveRejection::ExcludedSymbol));
        let mut fs = facts();
        fs.get_mut("SOLUSDT").unwrap().foreign_activity = true;
        fs.get_mut("XMRUSDT").unwrap().usdt_linear_perp = false;
        fs.get_mut("TAOUSDT").unwrap().symbol_type = "innovation".into();
        let mut c = cfg();
        c.excluded_symbol_types = vec!["innovation".into()];
        let d = decide_live(&intent("TAOUSDT", "long", 400.0, 1.0), None, &acct(), &[], &fs, &fees(), &placement(), &c);
        assert!(d.has(LiveRejection::ExcludedSymbol));
        let d = decide_live(&intent("SOLUSDT", "long", 150.0, 1.0), None, &acct(), &[], &fs, &fees(), &placement(), &cfg());
        assert!(d.has(LiveRejection::ForeignActivityOnSymbol));
        let d = decide_live(&intent("XMRUSDT", "long", 300.0, 1.0), None, &acct(), &[], &fs, &fees(), &placement(), &cfg());
        assert!(d.has(LiveRejection::NotUsdtLinearPerp));
        let d = run(&intent("DOGEUSDT", "long", 0.2, 1.0), &acct(), &[]);
        assert!(d.has(LiveRejection::UnknownInstrument));
    }

    #[test]
    fn kill_mode_versions_and_entry_type() {
        let mut a = acct();
        a.kill = true;
        assert!(run(&intent("ETHUSDT", "long", 3000.0, 1.0), &a, &[]).has(LiveRejection::KillSwitch));
        let mut i = intent("ETHUSDT", "long", 3000.0, 1.0);
        i.mode = Some("paper".into());
        assert!(run(&i, &acct(), &[]).has(LiveRejection::NotLiveMode));
        let mut i = intent("ETHUSDT", "long", 3000.0, 1.0);
        i.context = Some(serde_json::json!({"model_id": "x"}));
        assert!(run(&i, &acct(), &[]).has(LiveRejection::MissingVersions));
        let mut i = intent("ETHUSDT", "long", 3000.0, 1.0);
        i.entry.as_mut().unwrap().peg = None;
        assert!(run(&i, &acct(), &[]).has(LiveRejection::NotPeggedEntry));
        let mut i = intent("ETHUSDT", "long", 3000.0, 1.0);
        i.entry.as_mut().unwrap().kind = EntryType::Market;
        assert!(run(&i, &acct(), &[]).has(LiveRejection::NotPeggedEntry));
    }

    #[test]
    fn placement_ev_duplicate_and_margin() {
        let d = run(&intent("ETHUSDT", "long", 3000.0, 12.0), &acct(), &[]);
        assert!(d.has(LiveRejection::StopTooFar));
        let mut i = intent("ETHUSDT", "long", 3000.0, 1.0);
        i.sigma = None;
        assert!(run(&i, &acct(), &[]).has(LiveRejection::MissingSigma));
        let mut i = intent("ETHUSDT", "long", 3000.0, 1.0);
        i.net_er_r = Some(0.05);
        assert!(run(&i, &acct(), &[]).has(LiveRejection::NetErBelowMin));
        let p = [pos("ETHUSDT", Side::Long, 0.01, 3000.0, 0.0)];
        assert!(run(&intent("ETHUSDT", "long", 3000.0, 1.0), &acct(), &p).has(LiveRejection::DuplicateSymbol));
        let mut a = acct();
        a.available_usd = 10.0;
        assert!(run(&intent("ETHUSDT", "long", 3000.0, 0.3), &a, &[]).has(LiveRejection::InsufficientMargin));
    }

    #[test]
    fn tiers_from_rank_and_turnover() {
        let c = cfg();
        let name = |r, t, s| c.tier_for(r, t, s).map(|t| t.name.clone());
        assert_eq!(name(1, 5e9, 0.1).as_deref(), Some("liquid"));
        assert_eq!(name(10, 5e8, 1.0).as_deref(), Some("liquid"));
        assert_eq!(name(11, 5e8, 1.0).as_deref(), Some("mid"));
        assert_eq!(name(3, 5e9, 8.0).as_deref(), Some("mid"), "a wide spread drops a top-10 symbol to mid");
        assert_eq!(name(5, 1e6, 0.5), None, "rank alone is not enough");
        assert_eq!(name(200, 4e6, 20.0).as_deref(), Some("illiquid"));
        assert_eq!(name(200, 1e5, 1.0), None);
        assert_eq!(name(200, 4e6, 80.0), None, "spread wider than every tier");
    }

    #[test]
    fn peg_chase_stops_at_the_cap_or_the_stop() {
        let c = cfg();
        // 0.002 BTC, stop 63360 (1%): fine at 64000, not beyond the stop, and not once the worst case passes 2 USD
        assert!(peg_price_ok(Side::Long, 0.002, 64000.0, 63360.0, &fees(), &c, 10.0));
        assert!(!peg_price_ok(Side::Long, 0.002, 63300.0, 63360.0, &fees(), &c, 10.0));
        assert!(!peg_price_ok(Side::Long, 0.002, 64300.0, 63360.0, &fees(), &c, 10.0));
        assert!(peg_price_ok(Side::Short, 0.002, 64000.0, 64640.0, &fees(), &c, 10.0));
    }

    #[test]
    fn chasing_shrinks_the_order_instead_of_the_caps() {
        let c = cfg();
        // risk-limited BTC long: 0.002 at 64000, stop 63360 (risk ~1.42); chasing up keeps qty until a cap binds
        assert_eq!(chase_qty(Side::Long, 0.002, 128.0, 64000.0, 63360.0, &fees(), &c, 0.001), 0.002);
        assert_eq!(chase_qty(Side::Long, 0.002, 128.0, 64100.0, 63360.0, &fees(), &c, 0.001), 0.001, "notional cap: 0.002 x 64100 > 128");
        // a 100 USD token, stop 99.85: chasing up raises the risk per unit -> fewer units, risk to the stop stays <= 1.70
        let q = chase_qty(Side::Long, 10.0, 1000.0, 100.0, 99.85, &fees(), &c, 0.1);
        let q2 = chase_qty(Side::Long, 10.0, 1000.0, 100.06, 99.85, &fees(), &c, 0.1);
        assert!(q2 < q, "{q} -> {q2}");
        let unit = unit_loss(100.06, 99.85, entry_bps(&fees(), &c), 7.5);
        assert!(q2 * unit <= 1.70 + 1e-9);
        assert_eq!(chase_qty(Side::Short, 1.0, 100.0, 99.0, 98.0, &fees(), &c, 0.1), 0.0, "stop on the wrong side");
    }

    #[test]
    fn correlation_maths() {
        let x = [1.0, 2.0, 3.0, 4.0];
        assert!((pearson(&x, &[2.0, 4.0, 6.0, 8.0]).unwrap() - 1.0).abs() < 1e-12);
        assert!((pearson(&x, &[8.0, 6.0, 4.0, 2.0]).unwrap() + 1.0).abs() < 1e-12);
        assert_eq!(pearson(&x, &[1.0, 1.0, 1.0, 1.0]), None);
        let a = [(0, 100.0), (1, 101.0), (2, 102.0), (4, 99.0)];
        let b = [(0, 10.0), (1, 10.1), (3, 9.0), (4, 9.9)];
        let (ra, rb) = aligned_log_returns(&a, &b);
        assert_eq!(ra.len(), 2, "aligned on common timestamps 0, 1, 4");
        assert!((ra[0] - (1.01f64).ln()).abs() < 1e-12 && (rb[0] - (1.01f64).ln()).abs() < 1e-12);
    }
}
