//! OrderIntent -> IntentDecision (docs/EXECUTION_PIPELINE.md). Pure: no I/O and no clock.
//!
//! Sizing is fixed risk per trade, fees and slippage included:
//!     risk_usd = equity * risk_per_trade_pct / 100 * (reduce_risk_fraction if the gate said "reduce")
//!     qty      = risk_usd / (|entry - stop| + entry * cost_frac)       (rounded down to qty_step)
//! with `cost_frac` the round trip from `config/fees.yaml` for the symbol's class.
//!
//! Every rule is evaluated and every violation is reported (not just the first), so a rejected intent
//! says everything that was wrong with it. Exits (`side: flat`) are never blocked by entry limits.

use std::collections::{BTreeMap, HashMap};
use std::path::Path;

use anyhow::Context;
use at_core::config::RiskConfig;
use at_core::num::{Round, round_to_step};
use at_core::types::{EntryType, InstrumentRules, Side};
use serde::{Deserialize, Deserializer, Serialize};
use serde_json::Value;

use crate::AccountState;
use crate::fees::FeeSchedule;

// ---------------------------------------------------------------- intent (input, written by K)

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum IntentSide {
    Long,
    Short,
    Flat,
}

impl IntentSide {
    pub fn side(self) -> Option<Side> {
        match self {
            IntentSide::Long => Some(Side::Long),
            IntentSide::Short => Some(Side::Short),
            IntentSide::Flat => None,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct IntentEntry {
    #[serde(rename = "type")]
    pub kind: EntryType,
    /// Limit price, or the as-of reference price of a market entry. A market entry without a price is
    /// sized at the latest known mark.
    #[serde(default)]
    pub price: Option<f64>,
    /// "touch": a limit pegged to the best bid (buys) / ask (sells), filled by the paper fill model of
    /// `paper_entry` in config/risk.yaml (H-EXE1, R-EXE-2). `price` is then the decision-time reference.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub peg: Option<String>,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct IntentSigma {
    /// Risk sigma of one bar of the strategy's frequency, as a log-return fraction. R-VOL-2: this is the
    /// unshrunk sigma; the weekday factor is for expectations only and is never applied to placement.
    #[serde(default)]
    pub per_bar: Option<f64>,
    #[serde(default, rename = "daily_logHAR")]
    pub daily_log_har: Option<f64>,
    #[serde(default)]
    pub weekday_factor: Option<f64>,
    #[serde(default)]
    pub source: Option<String>,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct IntentGate {
    /// pass | block | reduce
    #[serde(default)]
    pub decision: Option<String>,
    #[serde(default, deserialize_with = "null_default")]
    pub rules_applied: Vec<String>,
    #[serde(default, deserialize_with = "null_default")]
    pub reasons: Vec<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct OrderIntent {
    pub intent_id: String,
    pub ts_ms: i64,
    #[serde(default)]
    pub as_of: Option<String>,
    /// replay | paper. "live" is refused by every consumer.
    #[serde(default)]
    pub mode: Option<String>,
    pub strategy: String,
    /// Strategy family = risk cluster (R-PD-1).
    pub family: String,
    pub symbol: String,
    #[serde(default)]
    pub asset_class: Option<String>,
    pub side: IntentSide,
    #[serde(default)]
    pub signal: Option<f64>,
    #[serde(default)]
    pub entry: Option<IntentEntry>,
    #[serde(default)]
    pub stop: Option<f64>,
    #[serde(default, deserialize_with = "null_default")]
    pub targets: Vec<f64>,
    #[serde(default)]
    pub sigma: Option<IntentSigma>,
    #[serde(default)]
    pub p_tp: Option<f64>,
    #[serde(default)]
    pub net_er_r: Option<f64>,
    #[serde(default)]
    pub gate: Option<IntentGate>,
    #[serde(default)]
    pub context: Option<Value>,
}

fn null_default<'de, D: Deserializer<'de>, T: Default + Deserialize<'de>>(d: D) -> Result<T, D::Error> {
    Ok(Option::<T>::deserialize(d)?.unwrap_or_default())
}

impl OrderIntent {
    pub fn gate_decision(&self) -> &str {
        self.gate.as_ref().and_then(|g| g.decision.as_deref()).unwrap_or("pass")
    }
}

// ---------------------------------------------------------------- config

#[derive(Clone, Debug, Deserialize)]
pub struct PlacementCfg {
    pub min_stop_distance_sigma: f64,
    /// The cap actually enforced: the yaml prior, replaced by `data/models/risk/stop_cap.json` when present (H-STOP1).
    pub max_stop_distance_sigma: f64,
    /// Hard bounds [lo, hi] any stop-cap file must lie in (config/risk.yaml).
    #[serde(default)]
    pub stop_cap_bounds_sigma: Option<[f64; 2]>,
}

#[derive(Deserialize)]
struct RiskYamlPlacement {
    placement: PlacementCfg,
    /// Paper / replay EV gate (user decision 2026-11-14): exclusive lower bound on net E[R].
    #[serde(default)]
    min_net_expected_r_paper: Option<f64>,
    #[serde(default)]
    paper_entry: Option<PaperEntryCfg>,
}

/// `paper_entry` in config/risk.yaml: how entries marked `peg: "touch"` are costed and filled (H-EXE1, R-EXE-2).
#[derive(Clone, Debug, PartialEq, Deserialize)]
pub struct PaperEntryCfg {
    /// pegged_limit | as_signal
    pub mode: String,
    /// Adverse slip of the pegged fill beyond the decision-time mid, in the trade direction (bp).
    pub slip_bps: f64,
    /// Re-peg window in minutes (documentation for the 1h paper book: the window lies inside the first bar).
    #[serde(default)]
    pub window_min: Option<f64>,
}

/// Fraction of the fixed risk used when the bias gate says "reduce" (contract note 2026-11-13).
pub const GATE_REDUCE_RISK_FRACTION: f64 = 0.5;

#[derive(Clone, Debug)]
pub struct DecideConfig {
    pub risk: RiskConfig,
    pub placement: PlacementCfg,
    pub fees: FeeSchedule,
    pub reduce_risk_fraction: f64,
    /// Where `placement.max_stop_distance_sigma` came from (yaml prior or the stop-cap file).
    pub stop_cap_source: String,
    /// `min_net_expected_r_paper` from config/risk.yaml; applied only by [`DecideConfig::for_mode`] in paper / replay.
    pub min_net_expected_r_paper: Option<f64>,
    /// The EV gate is exclusive (reject net E[R] <= min) instead of inclusive (reject < min): the paper gate "> 0".
    pub net_er_exclusive: bool,
    /// Pegged-entry model (config/risk.yaml `paper_entry`).
    pub paper_entry: Option<PaperEntryCfg>,
}

impl DecideConfig {
    /// `config/risk.yaml` (placement section) + `config/fees.yaml` + the stop cap (`data/models/risk/stop_cap.json`
    /// if present, validated against `placement.stop_cap_bounds_sigma`; otherwise the yaml value) under `root`.
    pub fn load(root: &Path, risk: RiskConfig) -> anyhow::Result<Self> {
        Self::load_with_cap_file(root, risk, &root.join(crate::stop_cap::STOP_CAP_FILE))
    }

    pub fn load_with_cap_file(root: &Path, risk: RiskConfig, cap_file: &Path) -> anyhow::Result<Self> {
        let p = root.join("config").join("risk.yaml");
        let text = std::fs::read_to_string(&p).with_context(|| p.display().to_string())?;
        let y = serde_yaml::from_str::<RiskYamlPlacement>(&text).with_context(|| format!("{}: placement", p.display()))?;
        let (mut placement, paper_min, paper_entry) = (y.placement, y.min_net_expected_r_paper, y.paper_entry);
        if let Some(pe) = &paper_entry {
            anyhow::ensure!(pe.slip_bps.is_finite() && pe.slip_bps >= 0.0, "paper_entry.slip_bps {} must be >= 0", pe.slip_bps);
            anyhow::ensure!(matches!(pe.mode.as_str(), "pegged_limit" | "as_signal"), "paper_entry.mode {:?}: pegged_limit | as_signal", pe.mode);
        }
        if let Some(v) = paper_min {
            anyhow::ensure!(v.is_finite() && v >= 0.0, "min_net_expected_r_paper {v} must be a finite number >= 0");
        }
        let cap = crate::stop_cap::load(cap_file, placement.max_stop_distance_sigma, placement.stop_cap_bounds_sigma)?;
        anyhow::ensure!(placement.min_stop_distance_sigma < cap.cap_sigma, "stop cap {} <= min_stop_distance_sigma {}", cap.cap_sigma, placement.min_stop_distance_sigma);
        placement.max_stop_distance_sigma = cap.cap_sigma;
        let fees = FeeSchedule::load(&root.join("config").join("fees.yaml"))?;
        Ok(Self {
            risk,
            placement,
            fees,
            reduce_risk_fraction: GATE_REDUCE_RISK_FRACTION,
            stop_cap_source: cap.source,
            min_net_expected_r_paper: paper_min,
            net_er_exclusive: false,
            paper_entry,
        })
    }

    /// The EV gate for an execution mode. Paper and replay (never mainnet, never any other mode) use the exclusive
    /// `min_net_expected_r_paper` when it is configured; everything else keeps `min_net_expected_r` (inclusive).
    pub fn for_mode(mut self, mode: &str, mainnet: bool) -> Self {
        let paper = matches!(mode.trim().to_ascii_lowercase().as_str(), "paper" | "replay");
        if paper
            && !mainnet
            && let Some(v) = self.min_net_expected_r_paper
        {
            self.risk.min_net_expected_r = v;
            self.net_er_exclusive = true;
        }
        self
    }
}

// ---------------------------------------------------------------- portfolio (input)

/// An open position or a resting entry order. Resting entries count as positions for slots, symbols,
/// clusters and gross exposure, so a re-run cannot double up.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BookEntry {
    pub symbol: String,
    pub strategy: String,
    pub family: String,
    pub side: Side,
    pub qty: f64,
    /// Fill price (open position) or limit / reference price (pending entry).
    pub entry_price: f64,
    pub pending: bool,
}

#[derive(Clone, Debug, Default)]
pub struct Portfolio {
    pub entries: Vec<BookEntry>,
    /// Latest known price per symbol (as of the decision time).
    pub marks: HashMap<String, f64>,
    /// `data/exec/KILL` present: no new entries.
    pub kill_switch: bool,
}

impl Portfolio {
    pub fn gross_notional(&self) -> f64 {
        self.entries.iter().map(|e| e.qty * if e.pending { e.entry_price } else { self.marks.get(&e.symbol).copied().unwrap_or(e.entry_price) }).sum()
    }

    pub fn family_counts(&self) -> BTreeMap<String, u32> {
        let mut m = BTreeMap::new();
        for e in &self.entries {
            *m.entry(e.family.clone()).or_default() += 1;
        }
        m
    }
}

// ---------------------------------------------------------------- decision (output)

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum IntentRejection {
    /// `data/exec/KILL` is present.
    KillSwitch,
    /// The bias gate decided "block".
    GateBlocked,
    /// Missing or inconsistent fields (no stop, stop on the wrong side, non-finite numbers...).
    InvalidIntent,
    /// Account equity is not positive.
    NoEquity,
    /// No venue rules (tick, lot, min notional) for the symbol.
    UnknownInstrument,
    /// The symbol already has a position or resting entry (one-way, no stacking, no flips).
    DuplicateSymbol,
    DailyLossLimit,
    MaxNewTradesPerDay,
    MaxOpenPositions,
    /// Family cap (R-PD-1: the trend variants are one bet).
    MaxPositionsPerCluster,
    MaxGrossExposure,
    /// Position notional above the instrument's max leverage times equity.
    ExceedsMaxLeverage,
    /// |entry - stop| / entry above max_stop_distance_pct.
    StopTooFar,
    /// Stop closer than min_stop_distance_sigma bar sigmas.
    StopInsideMinSigma,
    /// Stop farther than max_stop_distance_sigma bar sigmas.
    StopBeyondMaxSigma,
    /// No sigma.per_bar: the placement check cannot run.
    MissingSigma,
    NetErBelowMin,
    /// Neither (p_tp, target) nor net_er_r: expected value cannot be checked.
    NetErMissing,
    BelowMinQty,
    BelowMinNotional,
    /// side = flat but this strategy holds nothing in the symbol.
    NoPositionToClose,
    /// Paper mode: the intent is older than the allowed age.
    StaleIntent,
    /// intent_id already decided.
    DuplicateIntent,
}

impl IntentRejection {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::KillSwitch => "kill_switch",
            Self::GateBlocked => "gate_blocked",
            Self::InvalidIntent => "invalid_intent",
            Self::NoEquity => "no_equity",
            Self::UnknownInstrument => "unknown_instrument",
            Self::DuplicateSymbol => "duplicate_symbol",
            Self::DailyLossLimit => "daily_loss_limit",
            Self::MaxNewTradesPerDay => "max_new_trades_per_day",
            Self::MaxOpenPositions => "max_open_positions",
            Self::MaxPositionsPerCluster => "max_positions_per_cluster",
            Self::MaxGrossExposure => "max_gross_exposure",
            Self::ExceedsMaxLeverage => "exceeds_max_leverage",
            Self::StopTooFar => "stop_too_far",
            Self::StopInsideMinSigma => "stop_inside_min_sigma",
            Self::StopBeyondMaxSigma => "stop_beyond_max_sigma",
            Self::MissingSigma => "missing_sigma",
            Self::NetErBelowMin => "net_er_below_min",
            Self::NetErMissing => "net_er_missing",
            Self::BelowMinQty => "below_min_qty",
            Self::BelowMinNotional => "below_min_notional",
            Self::NoPositionToClose => "no_position_to_close",
            Self::StaleIntent => "stale_intent",
            Self::DuplicateIntent => "duplicate_intent",
        }
    }
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct PortfolioSummary {
    pub gross_x: f64,
    pub open_positions: u32,
    pub family_counts: BTreeMap<String, u32>,
}

/// One line of `data/exec/decisions/<run>.jsonl`. The first nine fields are the contract; the rest are
/// additive detail (contract note 2026-11-13).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct IntentDecision {
    pub intent_id: String,
    /// accept | reject
    pub decision: String,
    pub qty: f64,
    pub notional_usd: f64,
    pub risk_usd: f64,
    /// risk_usd in units of the fixed per-trade risk (1.0 normally, 0.5 after a gate "reduce").
    pub risk_r: f64,
    pub rejections: Vec<String>,
    pub cluster: String,
    /// Portfolio after this decision (including this order if accepted).
    pub portfolio: PortfolioSummary,
    /// open | close | none
    pub action: String,
    pub symbol: String,
    pub strategy: String,
    pub side: IntentSide,
    pub entry_type: Option<EntryType>,
    /// Tick-rounded prices (stop away from entry, target toward entry).
    pub entry: Option<f64>,
    pub stop: Option<f64>,
    pub target: Option<f64>,
    pub stop_pct: Option<f64>,
    pub stop_sigma: Option<f64>,
    /// Round-trip cost in R.
    pub cost_r: Option<f64>,
    /// Engine-checked net expected R: the lower of the engine's recomputation and the intent's net_er_r.
    pub net_er_r: Option<f64>,
    pub fee_class: String,
    /// Rejection name -> human detail with the numbers involved.
    pub details: BTreeMap<String, String>,
    /// Pegged entry (`entry.peg: "touch"`): the adverse slip the paper fill applies (paper_entry.slip_bps).
    #[serde(default)]
    pub peg_slip_bps: Option<f64>,
}

impl IntentDecision {
    pub fn accepted(&self) -> bool {
        self.decision == "accept"
    }

    fn base(intent: &OrderIntent) -> Self {
        Self {
            intent_id: intent.intent_id.clone(),
            decision: "reject".into(),
            qty: 0.0,
            notional_usd: 0.0,
            risk_usd: 0.0,
            risk_r: 0.0,
            rejections: vec![],
            cluster: intent.family.clone(),
            portfolio: PortfolioSummary::default(),
            action: "none".into(),
            symbol: intent.symbol.clone(),
            strategy: intent.strategy.clone(),
            side: intent.side,
            entry_type: intent.entry.as_ref().map(|e| e.kind),
            entry: None,
            stop: None,
            target: None,
            stop_pct: None,
            stop_sigma: None,
            cost_r: None,
            net_er_r: None,
            fee_class: String::new(),
            details: BTreeMap::new(),
            peg_slip_bps: None,
        }
    }

    /// A rejection decided outside `decide` (duplicate id, stale intent).
    pub fn reject_only(intent: &OrderIntent, r: IntentRejection, detail: &str, account: &AccountState, portfolio: &Portfolio) -> Self {
        let mut d = Self::base(intent);
        d.rejections.push(r.as_str().into());
        d.details.insert(r.as_str().into(), detail.into());
        d.portfolio = summary(portfolio, account.equity);
        d
    }

    fn reject(&mut self, r: IntentRejection, detail: String) {
        let k = r.as_str().to_string();
        if !self.rejections.contains(&k) {
            self.rejections.push(k.clone());
            self.details.insert(k, detail);
        }
    }
}

fn summary(p: &Portfolio, equity: f64) -> PortfolioSummary {
    PortfolioSummary {
        gross_x: if equity > 0.0 { p.gross_notional() / equity } else { 0.0 },
        open_positions: p.entries.len() as u32,
        family_counts: p.family_counts(),
    }
}

fn finite_pos(x: Option<f64>) -> Option<f64> {
    x.filter(|v| v.is_finite() && *v > 0.0)
}

// ---------------------------------------------------------------- decide

/// Decide one intent against the account (equity, UTC-day start equity, entries opened today; its
/// `open_positions` is not consulted) and the portfolio (open positions + resting entries, marks, kill flag).
pub fn decide(intent: &OrderIntent, account: &AccountState, portfolio: &Portfolio, rules: &HashMap<String, InstrumentRules>, cfg: &DecideConfig) -> IntentDecision {
    use IntentRejection as R;
    let mut d = IntentDecision::base(intent);
    d.fee_class = cfg.fees.class_of(&intent.symbol, intent.asset_class.as_deref());
    let rc = &cfg.risk;

    let Some(side) = intent.side.side() else {
        return decide_exit(d, intent, account, portfolio);
    };

    // ---- account and portfolio limits
    if portfolio.kill_switch {
        d.reject(R::KillSwitch, "data/exec/KILL present: no new entries".into());
    }
    let gate = intent.gate_decision();
    if gate == "block" {
        d.reject(R::GateBlocked, intent.gate.as_ref().map(|g| g.reasons.join("; ")).unwrap_or_default());
    }
    let equity = account.equity;
    if !(equity.is_finite() && equity > 0.0) {
        d.reject(R::NoEquity, format!("equity={equity}"));
    }
    if account.day_start_equity > 0.0 {
        let dd = (equity / account.day_start_equity - 1.0) * 100.0;
        if dd <= -rc.daily_loss_limit_pct {
            d.reject(R::DailyLossLimit, format!("day drawdown {dd:.3}% <= -{}%", rc.daily_loss_limit_pct));
        }
    }
    if account.trades_opened_today >= rc.max_new_trades_per_day {
        d.reject(R::MaxNewTradesPerDay, format!("{} opened today, max {}", account.trades_opened_today, rc.max_new_trades_per_day));
    }
    let n_open = portfolio.entries.len() as u32;
    if n_open >= rc.max_open_positions {
        d.reject(R::MaxOpenPositions, format!("{n_open} open or pending, max {}", rc.max_open_positions));
    }
    if portfolio.entries.iter().any(|e| e.symbol == intent.symbol) {
        d.reject(R::DuplicateSymbol, format!("{} already has a position or resting entry", intent.symbol));
    }
    let fam = portfolio.entries.iter().filter(|e| e.family == intent.family).count() as u32;
    if fam >= rc.max_positions_per_cluster {
        d.reject(R::MaxPositionsPerCluster, format!("family {} has {fam}, max {}", intent.family, rc.max_positions_per_cluster));
    }
    let rule = rules.get(&intent.symbol);
    if rule.is_none() {
        d.reject(R::UnknownInstrument, format!("no instrument rules for {}", intent.symbol));
    }

    // ---- geometry
    let entry_type = intent.entry.as_ref().map(|e| e.kind);
    let raw_entry = match &intent.entry {
        Some(IntentEntry { kind: EntryType::Limit, price, .. }) => finite_pos(*price),
        Some(IntentEntry { kind: EntryType::Market, price, .. }) => finite_pos(*price).or_else(|| finite_pos(portfolio.marks.get(&intent.symbol).copied())),
        None => None,
    };
    let raw_stop = finite_pos(intent.stop);
    let mut geometry_ok = true;
    if entry_type.is_none() || raw_entry.is_none() {
        d.reject(R::InvalidIntent, "entry missing, or no price (limit) / no price and no mark (market)".into());
        geometry_ok = false;
    }
    if raw_stop.is_none() {
        d.reject(R::InvalidIntent, "stop missing or not a positive number".into());
        geometry_ok = false;
    }
    if let Some(p) = intent.p_tp
        && !(0.0..=1.0).contains(&p)
    {
        d.reject(R::InvalidIntent, format!("p_tp={p} outside [0, 1]"));
        geometry_ok = false;
    }
    let (Some(rule), Some(et), Some(re), Some(rs), true) = (rule, entry_type, raw_entry, raw_stop, geometry_ok) else {
        d.portfolio = summary(portfolio, equity);
        return d;
    };

    let s = side.sign();
    let t = rule.tick_size;
    // Long: stop below rounds down (away), target above rounds down (toward). Short: both round up.
    let rnd = if side == Side::Long { Round::Down } else { Round::Up };
    let entry = round_to_step(re, t, Round::Nearest);
    let stop = round_to_step(rs, t, rnd);
    let target = intent.targets.first().copied().filter(|x| x.is_finite() && *x > 0.0).map(|x| round_to_step(x, t, rnd));
    d.entry = Some(entry);
    d.stop = Some(stop);
    d.target = target;
    if (entry - stop) * s <= 0.0 {
        d.reject(R::InvalidIntent, format!("stop {stop} is not on the losing side of entry {entry} for a {side:?}"));
        d.portfolio = summary(portfolio, equity);
        return d;
    }
    if let Some(tp) = target
        && (tp - entry) * s <= 0.0
    {
        d.reject(R::InvalidIntent, format!("target {tp} is not on the winning side of entry {entry}"));
        d.portfolio = summary(portfolio, equity);
        return d;
    }

    // ---- placement
    let stop_dist = (entry - stop).abs();
    let stop_pct = stop_dist / entry * 100.0;
    d.stop_pct = Some(stop_pct);
    if stop_pct > rc.max_stop_distance_pct {
        d.reject(R::StopTooFar, format!("stop {stop_pct:.3}% > max {}%", rc.max_stop_distance_pct));
    }
    match finite_pos(intent.sigma.as_ref().and_then(|x| x.per_bar)) {
        None => d.reject(R::MissingSigma, "sigma.per_bar missing or not positive".into()),
        Some(sig) => {
            let k = (entry / stop).ln().abs() / sig;
            d.stop_sigma = Some(k);
            let pl = &cfg.placement;
            if k < pl.min_stop_distance_sigma {
                d.reject(R::StopInsideMinSigma, format!("stop {k:.3} sigma < min {}", pl.min_stop_distance_sigma));
            }
            if k > pl.max_stop_distance_sigma {
                d.reject(R::StopBeyondMaxSigma, format!("stop {k:.3} sigma > max {}", pl.max_stop_distance_sigma));
            }
        }
    }

    // ---- costs and expected value
    let fees = cfg.fees.rates(&intent.symbol, Some(&d.fee_class));
    let peg = intent.entry.as_ref().and_then(|e| e.peg.as_deref());
    let cost = match peg {
        None => entry * fees.round_trip_frac(et),
        Some(p) => match (&cfg.paper_entry, et) {
            (Some(pe), EntryType::Limit) if p == "touch" && pe.mode == "pegged_limit" => {
                d.peg_slip_bps = Some(pe.slip_bps);
                entry * fees.round_trip_frac_pegged(pe.slip_bps)
            }
            _ => {
                d.reject(R::InvalidIntent, format!("entry.peg {p:?} needs a limit entry, peg \"touch\" and paper_entry.mode pegged_limit in config/risk.yaml"));
                entry * fees.round_trip_frac(et)
            }
        },
    };
    let eff = stop_dist + cost;
    d.cost_r = Some(cost / eff);
    let engine_er = match (intent.p_tp, target) {
        (Some(p), Some(tp)) => Some(p * ((tp - entry).abs() - cost) / eff - (1.0 - p)),
        _ => None,
    };
    let er = match (engine_er, intent.net_er_r.filter(|x| x.is_finite())) {
        (Some(a), Some(b)) => Some(a.min(b)),
        (a, b) => a.or(b),
    };
    d.net_er_r = er;
    match er {
        None => d.reject(R::NetErMissing, "need p_tp with a target, or net_er_r".into()),
        Some(x) if x < rc.min_net_expected_r || (cfg.net_er_exclusive && x <= rc.min_net_expected_r) => {
            d.reject(R::NetErBelowMin, format!("net E[R] {x:.4} {} min {}", if cfg.net_er_exclusive { "<=" } else { "<" }, rc.min_net_expected_r))
        }
        _ => {}
    }

    // ---- size
    let risk_frac = if gate == "reduce" { cfg.reduce_risk_fraction } else { 1.0 };
    let budget = equity.max(0.0) * rc.risk_per_trade_pct / 100.0 * risk_frac;
    let max_qty = match et {
        EntryType::Market => rule.max_mkt_qty.min(rule.max_qty),
        EntryType::Limit => rule.max_qty,
    };
    let qty = round_to_step((budget / eff).min(max_qty), rule.qty_step, Round::Down);
    let notional = qty * entry;
    if !(qty > 0.0 && qty >= rule.min_qty) {
        d.reject(R::BelowMinQty, format!("qty {qty} < min {} (risk budget {budget:.4} / effective stop {eff:.6})", rule.min_qty));
    } else if notional < rule.min_notional {
        d.reject(R::BelowMinNotional, format!("notional {notional:.4} < min {}", rule.min_notional));
    }
    if equity > 0.0 {
        let gross_x = (portfolio.gross_notional() + notional) / equity;
        if gross_x > rc.max_gross_exposure_x {
            d.reject(R::MaxGrossExposure, format!("gross {gross_x:.3}x > max {}x", rc.max_gross_exposure_x));
        }
        let lev = notional / equity;
        if rule.max_leverage > 0.0 && lev > rule.max_leverage {
            d.reject(R::ExceedsMaxLeverage, format!("position {lev:.3}x > instrument max {}x", rule.max_leverage));
        }
    }

    if d.rejections.is_empty() {
        let risk_usd = qty * eff;
        let unit = equity * rc.risk_per_trade_pct / 100.0;
        d.decision = "accept".into();
        d.action = "open".into();
        d.qty = qty;
        d.notional_usd = notional;
        d.risk_usd = risk_usd;
        d.risk_r = if unit > 0.0 { risk_usd / unit } else { 0.0 };
        let mut after = portfolio.clone();
        after.entries.push(BookEntry { symbol: intent.symbol.clone(), strategy: intent.strategy.clone(), family: intent.family.clone(), side, qty, entry_price: entry, pending: true });
        d.portfolio = summary(&after, equity);
    } else {
        d.portfolio = summary(portfolio, equity);
    }
    d
}

/// side = flat: close what this strategy holds in the symbol (position and resting entries).
fn decide_exit(mut d: IntentDecision, intent: &OrderIntent, account: &AccountState, portfolio: &Portfolio) -> IntentDecision {
    let held: Vec<&BookEntry> = portfolio.entries.iter().filter(|e| e.symbol == intent.symbol && e.strategy == intent.strategy).collect();
    if held.is_empty() {
        d.reject(IntentRejection::NoPositionToClose, format!("{} holds nothing in {}", intent.strategy, intent.symbol));
        d.portfolio = summary(portfolio, account.equity);
        return d;
    }
    d.decision = "accept".into();
    d.action = "close".into();
    d.qty = held.iter().map(|e| e.qty).sum();
    d.notional_usd = held.iter().map(|e| e.qty * portfolio.marks.get(&e.symbol).copied().unwrap_or(e.entry_price)).sum();
    let mut after = portfolio.clone();
    after.entries.retain(|e| !(e.symbol == intent.symbol && e.strategy == intent.strategy));
    d.portfolio = summary(&after, account.equity);
    d
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fees::FeeRates;
    use at_core::config::Costs;

    fn cfg() -> DecideConfig {
        let risk = RiskConfig {
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
        };
        let fees = FeeSchedule {
            default_class: "crypto".into(),
            classes: BTreeMap::from([
                ("crypto".into(), FeeRates { maker_bps: 2.0, taker_bps: 5.5, slippage_bps: 2.0 }),
                ("tradfi".into(), FeeRates { maker_bps: 0.0, taker_bps: 2.75, slippage_bps: 2.0 }),
            ]),
            symbols: BTreeMap::new(),
        };
        let placement = PlacementCfg { min_stop_distance_sigma: 0.5, max_stop_distance_sigma: 4.0, stop_cap_bounds_sigma: Some([2.5, 6.0]) };
        let stop_cap_source = "test".into();
        let paper_entry = Some(PaperEntryCfg { mode: "pegged_limit".into(), slip_bps: 1.3, window_min: Some(15.0) });
        let reduce_risk_fraction = GATE_REDUCE_RISK_FRACTION;
        DecideConfig { risk, placement, fees, reduce_risk_fraction, stop_cap_source, min_net_expected_r_paper: Some(0.0), net_er_exclusive: false, paper_entry }
    }

    fn rules() -> HashMap<String, InstrumentRules> {
        let r = |s: &str, tick, step, min_notional| InstrumentRules { symbol: s.into(), tick_size: tick, qty_step: step, min_qty: step, max_qty: 1e6, max_mkt_qty: 1e5, min_notional, max_leverage: 50.0 };
        HashMap::from([
            ("BTCUSDT".to_string(), r("BTCUSDT", 0.1, 0.001, 5.0)),
            ("ETHUSDT".to_string(), r("ETHUSDT", 0.01, 0.01, 5.0)),
            ("SOLUSDT".to_string(), r("SOLUSDT", 0.001, 0.1, 5.0)),
            ("NVDAUSDT".to_string(), r("NVDAUSDT", 0.01, 0.01, 5.0)),
        ])
    }

    /// ETH long, limit 3000, stop 2940 (2%), target 3150, sigma 1%/bar -> stop = 2.02 sigma.
    fn intent() -> OrderIntent {
        serde_json::from_value(serde_json::json!({
            "intent_id": "i1", "ts_ms": 0, "as_of": "2026-11-13T00:00:00Z", "mode": "replay",
            "strategy": "tsmom", "family": "trend", "symbol": "ETHUSDT", "asset_class": "crypto", "side": "long",
            "signal": 0.7, "entry": {"type": "limit", "price": 3000.0}, "stop": 2940.0, "targets": [3150.0],
            "sigma": {"per_bar": 0.01, "daily_logHAR": 0.03, "weekday_factor": 1.0, "source": "vol.har"},
            "p_tp": 0.45, "net_er_r": 0.3, "gate": {"decision": "pass", "rules_applied": ["R-PX-2"], "reasons": []},
            "context": {"regime_cell": "R4"}
        }))
        .unwrap()
    }

    fn acct(equity: f64) -> AccountState {
        AccountState { equity, day_start_equity: equity, trades_opened_today: 0, open_positions: vec![] }
    }

    fn held(symbol: &str, strategy: &str, family: &str, qty: f64, px: f64) -> BookEntry {
        BookEntry { symbol: symbol.into(), strategy: strategy.into(), family: family.into(), side: Side::Long, qty, entry_price: px, pending: false }
    }

    fn run(i: &OrderIntent, a: &AccountState, p: &Portfolio) -> IntentDecision {
        decide(i, a, p, &rules(), &cfg())
    }

    fn has(d: &IntentDecision, r: IntentRejection) -> bool {
        d.rejections.iter().any(|x| x == r.as_str()) && d.details.contains_key(r.as_str())
    }

    #[test]
    fn accepted_intent_risks_exactly_the_fixed_fraction_including_costs() {
        let d = run(&intent(), &acct(10_000.0), &Portfolio::default());
        assert!(d.accepted(), "{:?}", d.rejections);
        // effective stop = 60 + 3000 * (2 maker + 5.5 taker + 2 slip) bps = 62.85; budget 50 -> 0.795... -> 0.79
        assert_eq!(d.qty, 0.79);
        assert!(d.risk_usd <= 50.0 + 1e-9 && d.risk_usd > 50.0 - 0.01 * 62.85);
        assert!((d.risk_r - d.risk_usd / 50.0).abs() < 1e-12);
        assert!((d.notional_usd - 0.79 * 3000.0).abs() < 1e-9);
        assert_eq!(d.action, "open");
        assert_eq!(d.portfolio.open_positions, 1);
        assert_eq!(d.portfolio.family_counts["trend"], 1);
        assert!((d.stop_sigma.unwrap() - (3000.0f64 / 2940.0).ln() / 0.01).abs() < 1e-9);
        let json = serde_json::to_value(&d).unwrap();
        for k in ["intent_id", "decision", "qty", "notional_usd", "risk_usd", "risk_r", "rejections", "cluster", "portfolio"] {
            assert!(json.get(k).is_some(), "contract field {k}");
        }
        assert!(json["portfolio"].get("gross_x").is_some() && json["portfolio"].get("family_counts").is_some());
    }

    #[test]
    fn market_entry_costs_more_and_tradfi_costs_less() {
        let mut m = intent();
        m.entry = Some(IntentEntry { kind: EntryType::Market, price: Some(3000.0), peg: None });
        let dm = run(&m, &acct(10_000.0), &Portfolio::default());
        let dl = run(&intent(), &acct(10_000.0), &Portfolio::default());
        assert!(dm.cost_r.unwrap() > dl.cost_r.unwrap());
        let mut t = intent();
        t.symbol = "NVDAUSDT".into();
        t.asset_class = Some("tradfi".into());
        let dt = run(&t, &acct(10_000.0), &Portfolio::default());
        assert_eq!(dt.fee_class, "tradfi");
        assert!(dt.cost_r.unwrap() < dl.cost_r.unwrap());
        // market entry without a price is sized at the mark
        m.entry = Some(IntentEntry { kind: EntryType::Market, price: None, peg: None });
        let p = Portfolio { marks: HashMap::from([("ETHUSDT".to_string(), 3000.0)]), ..Default::default() };
        assert!(run(&m, &acct(10_000.0), &p).accepted());
        assert!(has(&run(&m, &acct(10_000.0), &Portfolio::default()), IntentRejection::InvalidIntent));
    }

    #[test]
    fn gate_reduce_halves_risk_and_block_rejects() {
        let mut i = intent();
        i.gate.as_mut().unwrap().decision = Some("reduce".into());
        let d = run(&i, &acct(10_000.0), &Portfolio::default());
        assert!(d.accepted());
        assert!(d.risk_r <= 0.5 + 1e-9 && d.risk_r > 0.45);
        i.gate.as_mut().unwrap().decision = Some("block".into());
        assert!(has(&run(&i, &acct(10_000.0), &Portfolio::default()), IntentRejection::GateBlocked));
    }

    #[test]
    fn kill_switch_rejects_entries_but_not_exits() {
        let p = Portfolio { kill_switch: true, ..Default::default() };
        assert!(has(&run(&intent(), &acct(10_000.0), &p), IntentRejection::KillSwitch));
        let mut x = intent();
        x.side = IntentSide::Flat;
        let p = Portfolio { kill_switch: true, entries: vec![held("ETHUSDT", "tsmom", "trend", 1.0, 3000.0)], ..Default::default() };
        let d = run(&x, &acct(10_000.0), &p);
        assert!(d.accepted() && d.action == "close" && d.qty == 1.0);
        assert_eq!(d.portfolio.open_positions, 0);
    }

    #[test]
    fn exit_without_position_is_rejected() {
        let mut x = intent();
        x.side = IntentSide::Flat;
        let p = Portfolio { entries: vec![held("ETHUSDT", "other_strategy", "trend", 1.0, 3000.0)], ..Default::default() };
        assert!(has(&run(&x, &acct(10_000.0), &p), IntentRejection::NoPositionToClose));
    }

    #[test]
    fn daily_loss_limit() {
        let mut a = acct(9_840.0);
        a.day_start_equity = 10_000.0;
        assert!(has(&run(&intent(), &a, &Portfolio::default()), IntentRejection::DailyLossLimit));
        a.equity = 9_860.0;
        assert!(run(&intent(), &a, &Portfolio::default()).accepted());
    }

    #[test]
    fn no_equity() {
        assert!(has(&run(&intent(), &acct(0.0), &Portfolio::default()), IntentRejection::NoEquity));
    }

    #[test]
    fn max_new_trades_per_day() {
        let mut a = acct(10_000.0);
        a.trades_opened_today = 3;
        assert!(has(&run(&intent(), &a, &Portfolio::default()), IntentRejection::MaxNewTradesPerDay));
    }

    #[test]
    fn max_open_positions_counts_pending_entries() {
        let mut entries: Vec<BookEntry> = (0..5).map(|k| held(&format!("X{k}USDT"), "s", &format!("f{k}"), 0.001, 1.0)).collect();
        entries[0].pending = true;
        let p = Portfolio { entries, ..Default::default() };
        let d = run(&intent(), &acct(10_000.0), &p);
        assert!(has(&d, IntentRejection::MaxOpenPositions));
        assert!(!has(&d, IntentRejection::MaxPositionsPerCluster));
    }

    #[test]
    fn cluster_cap_by_family() {
        let p = Portfolio { entries: vec![held("BTCUSDT", "donchian_breakout", "trend", 0.001, 75_000.0)], ..Default::default() };
        let d = run(&intent(), &acct(10_000.0), &p);
        assert!(has(&d, IntentRejection::MaxPositionsPerCluster));
        assert_eq!(d.rejections.len(), 1, "only the family cap: {:?}", d.rejections);
        let p = Portfolio { entries: vec![held("BTCUSDT", "zscore_reversion", "mean_reversion", 0.001, 75_000.0)], ..Default::default() };
        assert!(run(&intent(), &acct(10_000.0), &p).accepted());
    }

    #[test]
    fn duplicate_symbol() {
        let p = Portfolio { entries: vec![held("ETHUSDT", "zscore_reversion", "mean_reversion", 0.1, 3000.0)], ..Default::default() };
        assert!(has(&run(&intent(), &acct(10_000.0), &p), IntentRejection::DuplicateSymbol));
    }

    #[test]
    fn max_gross_exposure_marks_open_positions() {
        // 2.4x already open, plus ~0.24x new -> under 3x
        let open = held("SOLUSDT", "carry_x", "carry", 8.0, 3000.0);
        let mut p = Portfolio { entries: vec![open], ..Default::default() };
        assert!(run(&intent(), &acct(10_000.0), &p).accepted());
        p.marks.insert("SOLUSDT".into(), 3600.0); // 2.88x + 0.24x > 3x
        assert!(has(&run(&intent(), &acct(10_000.0), &p), IntentRejection::MaxGrossExposure));
    }

    #[test]
    fn exceeds_max_leverage() {
        let mut r = rules();
        r.get_mut("ETHUSDT").unwrap().max_leverage = 0.1;
        let d = decide(&intent(), &acct(10_000.0), &Portfolio::default(), &r, &cfg());
        assert!(has(&d, IntentRejection::ExceedsMaxLeverage));
    }

    #[test]
    fn unknown_instrument() {
        let mut i = intent();
        i.symbol = "NOPEUSDT".into();
        let d = run(&i, &acct(10_000.0), &Portfolio::default());
        assert!(has(&d, IntentRejection::UnknownInstrument));
        assert_eq!(d.qty, 0.0);
    }

    #[test]
    fn invalid_geometry() {
        let mut i = intent();
        i.stop = Some(3060.0);
        assert!(has(&run(&i, &acct(10_000.0), &Portfolio::default()), IntentRejection::InvalidIntent));
        let mut i = intent();
        i.stop = None;
        assert!(has(&run(&i, &acct(10_000.0), &Portfolio::default()), IntentRejection::InvalidIntent));
        let mut i = intent();
        i.targets = vec![2900.0];
        assert!(has(&run(&i, &acct(10_000.0), &Portfolio::default()), IntentRejection::InvalidIntent));
        let mut i = intent();
        i.p_tp = Some(1.5);
        assert!(has(&run(&i, &acct(10_000.0), &Portfolio::default()), IntentRejection::InvalidIntent));
        let mut i = intent();
        i.entry = Some(IntentEntry { kind: EntryType::Limit, price: Some(f64::NAN), peg: None });
        assert!(has(&run(&i, &acct(10_000.0), &Portfolio::default()), IntentRejection::InvalidIntent));
    }

    #[test]
    fn short_rounding_is_conservative() {
        let mut i = intent();
        i.side = IntentSide::Short;
        i.entry = Some(IntentEntry { kind: EntryType::Limit, price: Some(3000.004), peg: None });
        i.stop = Some(3060.001);
        i.targets = vec![2850.009];
        let d = run(&i, &acct(10_000.0), &Portfolio::default());
        assert!(d.accepted(), "{:?}", d.details);
        assert_eq!(d.stop, Some(3060.01), "short stop rounds up (away)");
        assert_eq!(d.target, Some(2850.01), "short target rounds up (toward entry)");
    }

    #[test]
    fn stop_too_far_pct() {
        let mut i = intent();
        i.stop = Some(2400.0); // 20%
        i.sigma.as_mut().unwrap().per_bar = Some(0.1);
        i.targets = vec![4500.0];
        let d = run(&i, &acct(10_000.0), &Portfolio::default());
        assert!(has(&d, IntentRejection::StopTooFar));
        assert!(!has(&d, IntentRejection::StopBeyondMaxSigma));
    }

    #[test]
    fn stop_sigma_bounds() {
        let mut i = intent();
        i.sigma.as_mut().unwrap().per_bar = Some(0.05); // 2% stop = 0.4 sigma
        assert!(has(&run(&i, &acct(10_000.0), &Portfolio::default()), IntentRejection::StopInsideMinSigma));
        i.sigma.as_mut().unwrap().per_bar = Some(0.004); // 5 sigma
        assert!(has(&run(&i, &acct(10_000.0), &Portfolio::default()), IntentRejection::StopBeyondMaxSigma));
        i.sigma = None;
        assert!(has(&run(&i, &acct(10_000.0), &Portfolio::default()), IntentRejection::MissingSigma));
    }

    #[test]
    fn net_er_gate_uses_the_lower_estimate() {
        let mut i = intent();
        i.net_er_r = Some(0.05); // engine says ~0.6, intent says 0.05 -> conservative
        let d = run(&i, &acct(10_000.0), &Portfolio::default());
        assert!(has(&d, IntentRejection::NetErBelowMin));
        assert_eq!(d.net_er_r, Some(0.05));
        let mut i = intent();
        i.p_tp = Some(0.30); // 0.3 * (150 - 2.85) / 62.85 - 0.7 = 0.002 < 0.1
        i.net_er_r = None;
        assert!(has(&run(&i, &acct(10_000.0), &Portfolio::default()), IntentRejection::NetErBelowMin));
        let mut i = intent();
        i.targets = vec![];
        i.net_er_r = None;
        assert!(has(&run(&i, &acct(10_000.0), &Portfolio::default()), IntentRejection::NetErMissing));
        // own-exit strategy (no target): K's net_er_r is used
        i.net_er_r = Some(0.2);
        assert!(run(&i, &acct(10_000.0), &Portfolio::default()).accepted());
    }

    #[test]
    fn small_accounts_hit_lot_and_notional_minimums() {
        let mut i = intent();
        i.symbol = "BTCUSDT".into();
        i.entry = Some(IntentEntry { kind: EntryType::Limit, price: Some(75_000.0), peg: None });
        i.stop = Some(73_500.0);
        i.targets = vec![79_000.0];
        assert!(has(&run(&i, &acct(100.0), &Portfolio::default()), IntentRejection::BelowMinQty));
        let mut r = rules();
        let eth = r.get_mut("ETHUSDT").unwrap();
        (eth.qty_step, eth.min_qty, eth.min_notional) = (0.001, 0.001, 100.0);
        let d = decide(&intent(), &acct(100.0), &Portfolio::default(), &r, &cfg()); // qty 0.007 -> $21
        assert!(has(&d, IntentRejection::BelowMinNotional), "{:?}", d.details);
    }

    #[test]
    fn all_violations_are_reported() {
        let mut a = acct(10_000.0);
        a.trades_opened_today = 9;
        let mut i = intent();
        i.sigma = None;
        i.gate.as_mut().unwrap().decision = Some("block".into());
        let p = Portfolio { kill_switch: true, ..Default::default() };
        let d = run(&i, &a, &p);
        for r in [IntentRejection::KillSwitch, IntentRejection::GateBlocked, IntentRejection::MaxNewTradesPerDay, IntentRejection::MissingSigma] {
            assert!(has(&d, r), "{r:?} in {:?}", d.rejections);
        }
        assert_eq!(d.decision, "reject");
        assert_eq!(d.qty, 0.0);
    }

    #[test]
    fn intent_json_tolerates_nulls_and_missing_optionals() {
        let i: OrderIntent = serde_json::from_str(r#"{"intent_id":"x","ts_ms":1,"strategy":"s","family":"trend","symbol":"ETHUSDT","side":"flat","targets":null,"gate":{"decision":"pass","rules_applied":null}}"#).unwrap();
        assert_eq!(i.side, IntentSide::Flat);
        assert!(i.targets.is_empty() && i.entry.is_none());
        let r = IntentDecision::reject_only(&i, IntentRejection::StaleIntent, "old", &acct(1.0), &Portfolio::default());
        assert_eq!(r.rejections, vec!["stale_intent"]);
    }

    #[test]
    fn project_config_loads() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
        let (_, risk) = at_core::config::load(&root).unwrap();
        let c = DecideConfig::load(&root, risk).unwrap();
        assert!(c.placement.min_stop_distance_sigma < c.placement.max_stop_distance_sigma);
        let [lo, hi] = c.placement.stop_cap_bounds_sigma.expect("risk.yaml has placement.stop_cap_bounds_sigma");
        assert!(lo <= c.placement.max_stop_distance_sigma && c.placement.max_stop_distance_sigma <= hi);
    }

    #[test]
    fn paper_ev_gate_applies_only_in_paper_or_replay_and_never_on_mainnet() {
        let base = cfg();
        let mut i = intent();
        i.net_er_r = Some(0.05); // between the paper gate (> 0) and the live gate (>= 0.10)
        let cases = [("paper", false, true), ("replay", false, true), ("PAPER", false, true), ("paper", true, false), ("replay", true, false)];
        for (mode, mainnet, accept) in cases.into_iter().chain([("live", false, false), ("demo", false, false), ("", false, false)]) {
            let c = base.clone().for_mode(mode, mainnet);
            let d = decide(&i, &acct(10_000.0), &Portfolio::default(), &rules(), &c);
            assert_eq!(d.accepted(), accept, "mode {mode:?} mainnet {mainnet}: {:?}", d.rejections);
            if !accept {
                assert!(has(&d, IntentRejection::NetErBelowMin));
                assert_eq!(c.risk.min_net_expected_r, 0.10, "the 0.10 gate is kept outside paper / replay");
            }
        }
        // the paper gate is exclusive: exactly zero and negative expectations are rejected
        let c = base.clone().for_mode("paper", false);
        for er in [0.0, -0.01] {
            i.net_er_r = Some(er);
            assert!(has(&decide(&i, &acct(10_000.0), &Portfolio::default(), &rules(), &c), IntentRejection::NetErBelowMin), "net E[R] {er}");
        }
        // no paper value configured: paper keeps the base gate
        let c = DecideConfig { min_net_expected_r_paper: None, ..base }.for_mode("paper", false);
        i.net_er_r = Some(0.05);
        assert!(has(&decide(&i, &acct(10_000.0), &Portfolio::default(), &rules(), &c), IntentRejection::NetErBelowMin));
    }

    #[test]
    fn pegged_entry_costs_maker_plus_slip_and_needs_the_config() {
        let mut p = intent();
        p.entry = Some(IntentEntry { kind: EntryType::Limit, price: Some(3000.0), peg: Some("touch".into()) });
        let dp = run(&p, &acct(10_000.0), &Portfolio::default());
        let dl = run(&intent(), &acct(10_000.0), &Portfolio::default());
        assert!(dp.accepted(), "{:?}", dp.rejections);
        assert_eq!(dp.peg_slip_bps, Some(1.3));
        assert_eq!(dl.peg_slip_bps, None);
        // cost = 3000 * (2.0 maker + 1.3 slip + 5.5 taker + 2.0 exit slip) bp; one plain limit costs 1.3 bp less
        let eff = 60.0 + 3000.0 * 10.8e-4;
        assert!((dp.cost_r.unwrap() - 3000.0 * 10.8e-4 / eff).abs() < 1e-12);
        assert!(dp.cost_r.unwrap() > dl.cost_r.unwrap());
        // a peg on a market entry, an unknown peg or a missing config is refused
        let mut m = p.clone();
        m.entry = Some(IntentEntry { kind: EntryType::Market, price: Some(3000.0), peg: Some("touch".into()) });
        assert!(has(&run(&m, &acct(10_000.0), &Portfolio::default()), IntentRejection::InvalidIntent));
        let mut u = p.clone();
        u.entry = Some(IntentEntry { kind: EntryType::Limit, price: Some(3000.0), peg: Some("mid".into()) });
        assert!(has(&run(&u, &acct(10_000.0), &Portfolio::default()), IntentRejection::InvalidIntent));
        let c = DecideConfig { paper_entry: None, ..cfg() };
        assert!(has(&decide(&p, &acct(10_000.0), &Portfolio::default(), &rules(), &c), IntentRejection::InvalidIntent));
        let j: OrderIntent = serde_json::from_value(serde_json::json!({"intent_id": "x", "ts_ms": 1, "strategy": "s", "family": "shock", "symbol": "ETHUSDT",
            "side": "long", "entry": {"type": "limit", "price": 3000.0, "peg": "touch"}})).unwrap();
        assert_eq!(j.entry.unwrap().peg.as_deref(), Some("touch"));
    }

    #[test]
    fn project_risk_yaml_has_the_paper_gate() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
        let (_, risk) = at_core::config::load(&root).unwrap();
        let c = DecideConfig::load(&root, risk).unwrap();
        assert_eq!(c.min_net_expected_r_paper, Some(0.0));
        let pe = c.paper_entry.clone().expect("risk.yaml paper_entry");
        assert_eq!((pe.mode.as_str(), pe.slip_bps), ("pegged_limit", 1.3));
        assert_eq!(c.risk.min_net_expected_r, 0.10);
        assert_eq!(c.clone().for_mode("paper", true).risk.min_net_expected_r, 0.10);
        assert_eq!(c.for_mode("paper", false).risk.min_net_expected_r, 0.0);
    }

    #[test]
    fn stop_cap_file_overrides_the_yaml_prior_and_is_enforced() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
        let (_, risk) = at_core::config::load(&root).unwrap();
        let dir = std::env::temp_dir().join(format!("at_risk_decide_cap_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let missing = dir.join("absent.json");
        let c = DecideConfig::load_with_cap_file(&root, risk.clone(), &missing).unwrap();
        assert_eq!(c.placement.max_stop_distance_sigma, 4.0, "no file: the yaml prior");
        let f = dir.join("stop_cap.json");
        std::fs::write(&f, r#"{"applied_cap_sigma": 2.6, "version": "t", "n": 40}"#).unwrap();
        let c = DecideConfig::load_with_cap_file(&root, risk.clone(), &f).unwrap();
        assert_eq!(c.placement.max_stop_distance_sigma, 2.6);
        assert!(c.stop_cap_source.contains("stop_cap.json"));
        // the test intent's stop is 2.02 sigma: accepted under 2.6; a 3 sigma stop is now beyond the cap
        assert!(decide(&intent(), &acct(10_000.0), &Portfolio::default(), &rules(), &c).accepted());
        let mut wide = intent();
        wide.stop = Some(3000.0 * (-0.03f64).exp());
        let d = decide(&wide, &acct(10_000.0), &Portfolio::default(), &rules(), &c);
        assert!(has(&d, IntentRejection::StopBeyondMaxSigma), "{:?}", d.rejections);
        std::fs::write(&f, r#"{"applied_cap_sigma": 8.0}"#).unwrap();
        assert!(DecideConfig::load_with_cap_file(&root, risk, &f).is_err(), "a cap outside the hard bounds is refused");
        let _ = std::fs::remove_dir_all(&dir);
    }
}
