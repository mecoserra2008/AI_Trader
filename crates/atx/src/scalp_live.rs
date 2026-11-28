//! `atx scalp --mode live`: the H-SCALP1 scalper on the user's live account, inside the live desk's limits. DISABLED
//! until H-SCALP1 passes: without a frozen passing evaluation (`live.verdict_file`) the run refuses to start.
//!
//!   atx scalp --mode live --run-id <ID> [--max-entries 12] [--minutes N]
//!
//! Shared state: every scalper trade is a `TradeRec` (source "scalp", id `sc-<run>-<n>`, orderLinkId `lsb-sc-...`) in
//! `data/live/state.json`, written under the desk's `StateLock` with the desk's load / save. So `entries_today`,
//! realised P&L, the daily stop, open slots, same side and gross are one set of counters for the desk and the scalper,
//! and `atx live status` / `reconcile` show and reconcile scalper trades like any bot trade.
//!
//! Every entry: `at_risk::live::decide_live` with the live_small profile (tier notional caps tightened to the scalper's
//! 20 USD; nothing loosened; min_net_expected_r unchanged, never the learning minimum), p_tp = the CI lower bound of the
//! shadow TP-hit rate, then the scalper sub-limits (run entries, scalper day loss, consecutive losses, no same-side
//! entry while a desk bot position in a correlated cluster is open). Start: mainnet + two-key guard + no KILL +
//! sub-limits + passing verdict + (gate MANUAL) the run approval data/live/approvals/scalp-<run_id>.ok, written by the
//! orchestrator under the user's delegation, never by this code. The guard and KILL are re-checked every loop.
//!
//! Orders: PostOnly entry at the touch with the exchange stopLoss attached (LastPrice, tpslMode Full); on the fill a
//! reduce-only PostOnly take profit at exactly reward_r x the stop distance. Time stop and funding as in shadow (passive
//! reduce-only exit at the far touch `-tx`, then a reduce-only market close). Venue state: private WS (order, execution,
//! position) with a REST poll every `live.poll_s`. Any inconsistency (unknown scalper order, unexpected position or size,
//! position without stop after a re-attach attempt) halts the run: its own orders are cancelled, the position keeps its
//! exchange stop, the halt is journaled.

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::time::Duration;

use anyhow::{Context, bail, ensure};
use at_bybit::Credentials;
use at_bybit::market::MarketData;
use at_bybit::trading::OrderRequest;
use at_bybit::ws::{self, WsMessage};
use at_core::config::{self, Environment};
use at_core::num::format_step;
use at_core::time::{DAY, now_ms};
use at_core::types::{EntryType, Side};
use at_risk::fees::FeeRates;
use at_risk::intent::{IntentEntry, IntentSide, IntentSigma, OrderIntent, PlacementCfg};
use at_risk::live::{self as lv, LiveAccount, LiveDecision, LivePos, LiveSmallCfg, SymbolFacts};
use at_risk::DecideConfig;
use serde_json::{Value, json};

use crate::cmd::Ctx;
use crate::live::{Desk, LiveState, StateLock, TradeRec, entry_link, positions_of};
use crate::scalp::{self, Bracket, Engine, FundingAct, Inst, SCALP_VERSION, ScalpCfg, ScalpLimits};

pub const SCALP_SOURCE: &str = "scalp";
const HYPOTHESIS: &str = "H-SCALP1";
const MIN_FILLED_TRADES: u64 = 30;

// ---------------------------------------------------------------- start gate (pure)

/// p_tp for the EV gate from the frozen H-SCALP1 evaluation: the 95 % CI LOWER bound of the TP-hit rate. Refused unless
/// the file is H-SCALP1, for this strategy hash, verdict "supported", on >= 30 filled trades, with a lower bound in (0, 1].
pub fn p_tp_from_verdict(v: &Value, strategy_hash: &str) -> anyhow::Result<f64> {
    ensure!(v["hypothesis"] == HYPOTHESIS, "verdict file is not {HYPOTHESIS} (got {})", v["hypothesis"]);
    ensure!(v["cfg_hash"] == strategy_hash, "verdict cfg_hash {} != strategy hash {strategy_hash}: the evaluated strategy is not this one", v["cfg_hash"]);
    ensure!(v["verdict"] == "supported", "H-SCALP1 verdict is {}: live stays disabled until it passes", v["verdict"]);
    let n = v["n"].as_u64().unwrap_or(0);
    ensure!(n >= MIN_FILLED_TRADES, "verdict on {n} trades < {MIN_FILLED_TRADES}");
    let lo = v["tp_hit_ci95"].get(0).and_then(Value::as_f64).context("verdict has no tp_hit_ci95 [lo, hi]")?;
    let rate = v["tp_hit_rate"].as_f64().context("verdict has no tp_hit_rate")?;
    ensure!(lo.is_finite() && lo > 0.0 && lo <= 1.0 && lo <= rate + 1e-12, "tp_hit_ci95 lower bound {lo} must be in (0, tp_hit_rate]");
    Ok(lo)
}

/// The run approval file (written by the orchestrator under the user's delegation; never by this code).
pub fn run_approval_path(live_root: &Path, run_id: &str) -> PathBuf {
    live_root.join("approvals").join(format!("scalp-{run_id}.ok"))
}

/// Inputs of the start gate, gathered by the caller.
pub struct StartInputs<'a> {
    pub env: Environment,
    pub guard: Result<(), String>,
    pub venue_writable: bool,
    pub kill: bool,
    pub verdict: Option<&'a Value>,
    pub gate_manual: bool,
    pub run_approval: bool,
    pub max_entries: u32,
}

/// Start gate, in order: mainnet, the two-key guard, KILL, the sub-limits (tighter than live_small only), the run's
/// --max-entries, the H-SCALP1 verdict (missing or not passing: refuse), the approval. Returns p_tp.
pub fn start_preflight(x: &StartInputs, cfg: &ScalpCfg, live: &LiveSmallCfg) -> anyhow::Result<f64> {
    ensure!(x.env == Environment::Mainnet, "live mode needs environment mainnet (AI_TRADER_ENVIRONMENT=mainnet); got {}", x.env.as_str());
    if let Err(e) = &x.guard {
        bail!("live mode refused: two-key mainnet guard not satisfied ({e})");
    }
    ensure!(x.venue_writable, "live mode refused: the venue client is read-only (two-key guard)");
    ensure!(!x.kill, "live mode refused: data/live/KILL is present");
    cfg.limits.validate_against_live(live).context("live mode refused: scalper sub-limits")?;
    ensure!(x.max_entries >= 1 && x.max_entries <= cfg.limits.max_entries_per_run, "--max-entries must be in [1, {}]", cfg.limits.max_entries_per_run);
    let lc = cfg.live.as_ref().context("live mode refused: config has no live section")?;
    ensure!(lc.p_tp_source == "h_scalp1_shadow", "live.p_tp_source must be h_scalp1_shadow");
    let v = x.verdict.with_context(|| format!("live mode refused: no H-SCALP1 verdict file ({}): live stays disabled until H-SCALP1 passes", lc.verdict_file))?;
    let p = p_tp_from_verdict(v, &cfg.hash()).context("live mode refused")?;
    ensure!(!x.gate_manual || x.run_approval, "live mode refused: the live_small approval gate is MANUAL and there is no run approval file data/live/approvals/scalp-<run_id>.ok");
    Ok(p)
}

/// live_small with the scalper's sub-limits on top: every tier's notional cap at most the scalper cap. Tighter only.
pub fn tightened(live: &LiveSmallCfg, lim: &ScalpLimits) -> LiveSmallCfg {
    let mut c = live.clone();
    for t in &mut c.tiers {
        t.notional_cap_usd = t.notional_cap_usd.min(lim.notional_cap_usd);
    }
    c
}

/// Run id: 1-16 ASCII letters / digits (it goes into orderLinkIds).
pub fn valid_run_id(r: &str) -> bool {
    !r.is_empty() && r.len() <= 16 && r.chars().all(|c| c.is_ascii_alphanumeric())
}

pub fn trade_id(tag: &str, run_id: &str, n: u32) -> String {
    format!("{tag}-{run_id}-{n}")
}

// ---------------------------------------------------------------- per-entry checks (pure)

/// Net realised P&L of scalper trades closed since the UTC day start.
pub fn scalp_realised_today(st: &LiveState, now: i64) -> f64 {
    let start = now - now.rem_euclid(DAY);
    st.trades.iter().filter(|t| t.is_scalp() && t.closed_ms.is_some_and(|c| c >= start)).filter_map(|t| t.realised_usd).sum()
}

/// What the run has done so far (for the sub-limits).
#[derive(Clone, Debug, Default)]
pub struct RunCounters {
    pub entries: u32,
    pub consec_losses: u32,
}

/// The scalper's own refusals, on top of decide_live (each can only add a refusal).
#[allow(clippy::too_many_arguments)]
pub fn scalp_refusals(
    st: &LiveState,
    venue_pos: &[LivePos],
    facts: &HashMap<String, SymbolFacts>,
    live: &LiveSmallCfg,
    lim: &ScalpLimits,
    symbol: &str,
    side: Side,
    run: &RunCounters,
    max_entries: u32,
    now: i64,
) -> Vec<String> {
    let mut r = Vec::new();
    if run.entries >= max_entries.min(lim.max_entries_per_run) {
        r.push(format!("scalp_max_entries_per_run: {} of {}", run.entries, max_entries.min(lim.max_entries_per_run)));
    }
    if run.consec_losses >= lim.max_consecutive_losses {
        r.push(format!("scalp_consecutive_losses: {} >= {}", run.consec_losses, lim.max_consecutive_losses));
    }
    let open_scalp: f64 = st
        .trades
        .iter()
        .filter(|t| t.active() && t.is_scalp())
        .filter_map(|t| venue_pos.iter().find(|p| p.symbol == t.symbol && Some(p.side) == t.side && !p.pending))
        .map(|p| p.open_pnl_usd.min(0.0))
        .sum();
    let day = scalp_realised_today(st, now) + open_scalp;
    if day <= -lim.day_loss_stop_usd {
        r.push(format!("scalp_day_loss_stop: scalper day P&L {day:.4} <= -{}", lim.day_loss_stop_usd));
    }
    // a desk bot position in a correlated cluster on the same side is the same bet: no scalper entry beside it
    let me = facts.get(symbol).cloned().unwrap_or_else(|| SymbolFacts { symbol: symbol.into(), ..Default::default() });
    for t in st.trades.iter().filter(|t| t.active() && !t.is_scalp() && t.side == Some(side)) {
        let open = venue_pos.iter().any(|p| p.symbol == t.symbol && Some(p.side) == t.side && !p.pending);
        let theirs = facts.get(&t.symbol).cloned().unwrap_or_else(|| SymbolFacts { symbol: t.symbol.clone(), ..Default::default() });
        if open && lv::correlated(&me, &theirs, live) {
            r.push(format!("scalp_correlated_same_side: desk trade {} {} {side:?} shares a cluster ({:?} / {:?})", t.id, t.symbol, lv::clusters_of(&me, live), lv::clusters_of(&theirs, live)));
        }
    }
    r
}

/// The OrderIntent of one scalper entry (mode live, pegged at the touch, stop and the 2R target, p_tp from the verdict).
#[allow(clippy::too_many_arguments)]
pub fn build_intent(cfg: &ScalpCfg, id: &str, side: Side, br: &Bracket, tick: f64, sigma: f64, p_tp: f64, now: i64, run_id: &str) -> OrderIntent {
    let px = |t: i64| t as f64 * tick;
    let horizon = cfg.live.as_ref().map(|l| l.sigma_horizon_s).unwrap_or(60);
    OrderIntent {
        intent_id: id.to_string(),
        ts_ms: now,
        as_of: None,
        mode: Some("live".into()),
        strategy: "scalp_flow".into(),
        family: "scalp_flow".into(),
        symbol: cfg.symbol.clone(),
        asset_class: Some("crypto".into()),
        side: if side == Side::Long { IntentSide::Long } else { IntentSide::Short },
        signal: None,
        entry: Some(IntentEntry { kind: EntryType::Limit, price: Some(px(br.entry)), peg: Some("touch".into()) }),
        stop: Some(px(br.stop)),
        targets: vec![px(br.tp)],
        sigma: Some(IntentSigma { per_bar: Some(sigma), daily_log_har: None, weekday_factor: None, source: Some(format!("scalp: sd(1 s mid log returns) x sqrt({horizon} s)")) }),
        p_tp: Some(p_tp),
        net_er_r: None,
        gate: None,
        context: Some(json!({
            "model_id": format!("none: deterministic engine {SCALP_VERSION}"),
            "prompt_version": format!("{HYPOTHESIS} strategy {}", cfg.hash()),
            "session_id": format!("scalp-{run_id}"),
            "p_tp_source": "h_scalp1_shadow (95% CI lower bound of the TP-hit rate)",
            "reason": "H-SCALP1 entry rule",
        })),
    }
}

/// Reserve one scalper entry in the shared state: the UTC day rolls, entries_today counts it, the trade is recorded.
pub fn reserve(st: &mut LiveState, tr: TradeRec, now: i64) {
    st.roll_day(now);
    st.entries_today += 1;
    st.trades.push(tr);
}

/// Load -> reserve -> save under the desk's StateLock (the same lock and schema as `atx live`). The runner does the same
/// inside the lock it already holds for the decision (`try_entry`); this standalone form is the tested round trip.
#[cfg(test)]
pub async fn reserve_in_state(root: &Path, state: &Path, tr: TradeRec, now: i64) -> anyhow::Result<LiveState> {
    let _lock = StateLock::acquire(root).await?;
    let mut st = LiveState::load(state)?;
    reserve(&mut st, tr, now);
    st.save(state)?;
    Ok(st)
}

/// Apply `f` to one trade of the shared state under the lock.
pub async fn update_trade(root: &Path, state: &Path, id: &str, f: impl FnOnce(&mut TradeRec)) -> anyhow::Result<Option<TradeRec>> {
    let _lock = StateLock::acquire(root).await?;
    let mut st = LiveState::load(state)?;
    let out = st.trades.iter_mut().find(|t| t.id == id).map(|t| {
        f(t);
        t.clone()
    });
    st.save(state)?;
    Ok(out)
}

/// The full per-entry decision: decide_live on the tightened profile (with the D2 / D3 entry gates), then the scalper
/// refusals (merged).
#[allow(clippy::too_many_arguments)]
pub fn decide_entry(
    intent: &OrderIntent,
    touch: f64,
    account: &LiveAccount,
    positions: &[LivePos],
    st: &LiveState,
    facts: &HashMap<String, SymbolFacts>,
    fees: &FeeRates,
    placement: &PlacementCfg,
    live: &LiveSmallCfg,
    lim: &ScalpLimits,
    run: &RunCounters,
    max_entries: u32,
    now: i64,
    gates: &lv::LiveGates,
) -> LiveDecision {
    let tight = tightened(live, lim);
    let mut d = lv::decide_live(intent, Some(touch), account, positions, facts, fees, placement, &tight, gates);
    let side = intent.side.side().unwrap_or(Side::Long);
    for r in scalp_refusals(st, positions, facts, live, lim, &intent.symbol, side, run, max_entries, now) {
        let k = r.split(':').next().unwrap_or("scalp").to_string();
        if !d.rejections.contains(&k) {
            d.rejections.push(k.clone());
            d.details.insert(k, r);
        }
    }
    if !d.rejections.is_empty() {
        d.decision = "reject".into();
    }
    d
}

// ---------------------------------------------------------------- venue state (pure parsing + consistency)

#[derive(Clone, Debug, Default, PartialEq)]
pub struct VOrder {
    pub id: String,
    pub link: String,
    pub status: String,
    pub cum: f64,
    pub avg: f64,
}

fn num(v: &Value) -> f64 {
    v.as_str().and_then(|s| s.parse().ok()).or_else(|| v.as_f64()).unwrap_or(0.0)
}

pub fn parse_order(o: &Value) -> VOrder {
    VOrder {
        id: o["orderId"].as_str().unwrap_or_default().into(),
        link: o["orderLinkId"].as_str().unwrap_or_default().into(),
        status: o["orderStatus"].as_str().unwrap_or_default().into(),
        cum: num(&o["cumExecQty"]),
        avg: num(&o["avgPrice"]),
    }
}

fn order_done(status: &str) -> bool {
    matches!(status, "Filled" | "Cancelled" | "Rejected" | "Deactivated" | "PartiallyFilledCanceled")
}

/// Venue position on the symbol: (side, size, avg, stop). Size 0 = flat.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct VPos {
    pub side: Option<Side>,
    pub size: f64,
    pub stop: f64,
}

pub fn parse_position(p: &Value) -> VPos {
    let size = num(&p["size"]);
    let side = match p["side"].as_str() {
        Some("Buy") if size > 0.0 => Some(Side::Long),
        Some("Sell") if size > 0.0 => Some(Side::Short),
        _ => None,
    };
    VPos { side, size, stop: num(&p["stopLoss"]) }
}

/// Phase of the live position.
#[derive(Clone, Debug, PartialEq)]
pub enum LivePhase {
    Flat,
    Entering { id: String, order_id: String, side: Side, br: Bracket, qty: f64, placed_ms: i64, cum: f64, cancel_sent: bool },
    Open { id: String, side: Side, qty: f64, br: Bracket, r_usd: f64, filled_ms: i64, tp: Option<String>, tx: Option<(String, i64)>, market_sent: bool, held_across: bool, stop_missing_since: Option<i64>, reattached: bool },
    /// The position is gone; waiting for the closed P&L (reconcile) to book the trade.
    Closing { id: String, since_ms: i64 },
}

/// What the venue shows that the run must react to.
#[derive(Clone, Debug, PartialEq)]
pub enum Check {
    Ok,
    ReattachStop,
    Halt(String),
}

/// Consistency of the venue with the run: scalper orders the run does not know, a position the run does not hold (or of
/// another size / side), a position without its exchange stop (re-attach once, then halt after the grace).
pub fn consistency(phase: &LivePhase, pos: &VPos, own_orders: &[VOrder], known: &HashSet<String>, qty_step: f64, now: i64, stop_grace_ms: i64) -> Check {
    if let Some(o) = own_orders.iter().find(|o| !known.contains(&o.id) && !order_done(&o.status)) {
        return Check::Halt(format!("unknown scalper order {} ({})", o.id, o.link));
    }
    match phase {
        LivePhase::Flat | LivePhase::Closing { .. } if pos.size > 0.0 => Check::Halt(format!("position {:?} {} on the symbol while the run holds none", pos.side, pos.size)),
        LivePhase::Entering { side, qty, .. } if pos.size > 0.0 && (pos.side != Some(*side) || pos.size > qty + qty_step * 0.5 + 1e-9) => {
            Check::Halt(format!("position {:?} {} does not match the resting {side:?} entry of {qty}", pos.side, pos.size))
        }
        LivePhase::Open { side, qty, stop_missing_since, reattached, .. } if pos.size > 0.0 => {
            if pos.side != Some(*side) || (pos.size - qty).abs() > qty_step * 0.5 + 1e-9 {
                return Check::Halt(format!("position {:?} {} != the run's {side:?} {qty}", pos.side, pos.size));
            }
            if pos.stop <= 0.0 {
                return match stop_missing_since {
                    Some(t) if *reattached && now - t > stop_grace_ms => Check::Halt("position without exchange stop after the re-attach attempt".into()),
                    Some(t) if now - t > stop_grace_ms => Check::ReattachStop,
                    _ => Check::Ok,
                };
            }
            Check::Ok
        }
        _ => Check::Ok,
    }
}

// ---------------------------------------------------------------- the runner

struct Runner<'a> {
    desk: Desk<'a>,
    cfg: ScalpCfg,
    live_cfg: crate::scalp::ScalpLiveCfg,
    inst: Inst,
    rules: at_core::types::InstrumentRules,
    run_id: String,
    approved_by: String,
    p_tp: f64,
    max_entries: u32,
    placement: PlacementCfg,
    phase: LivePhase,
    run: RunCounters,
    seq: u32,
    known: HashSet<String>,
    cooldown_until: i64,
    backoff_until: i64,
    journal_dir: PathBuf,
}

impl Runner<'_> {
    fn state_path(&self) -> PathBuf {
        self.desk.paths.state()
    }
    fn root(&self) -> PathBuf {
        self.desk.paths.root.clone()
    }
    fn tag(&self) -> String {
        format!("{}{}-", self.desk.cfg.link_tag(), self.live_cfg.link_tag)
    }
    fn px(&self, t: i64) -> String {
        format_step(t as f64 * self.inst.tick, self.inst.tick)
    }

    fn j(&self, kind: &str, mut v: Value) {
        v["type"] = json!(kind);
        v["ms"] = json!(now_ms());
        v["ts"] = json!(chrono::Utc::now().to_rfc3339());
        v["run_id"] = json!(self.run_id);
        v["version"] = json!(SCALP_VERSION);
        v["cfg_hash"] = json!(self.cfg.hash());
        v["mode"] = json!("live");
        v["approved_by"] = json!(self.approved_by);
        let _ = std::fs::create_dir_all(&self.journal_dir);
        let p = self.journal_dir.join(format!("{}.jsonl", chrono::Utc::now().format("%Y-%m-%d")));
        if let Ok(mut f) = std::fs::OpenOptions::new().create(true).append(true).open(p) {
            use std::io::Write as _;
            let _ = writeln!(f, "{v}");
        }
        let _ = self.desk.jw(kind_for_desk(kind), json!({"source": "atx scalp", "scalp": v}), &self.versions());
    }

    fn versions(&self) -> Value {
        let mut v = self.desk.versions(None);
        v["scalp_version"] = json!(SCALP_VERSION);
        v["scalp_strategy_hash"] = json!(self.cfg.hash());
        v["model_id"] = json!(format!("none: deterministic engine {SCALP_VERSION}"));
        v["prompt_version"] = json!(format!("{HYPOTHESIS} strategy {}", self.cfg.hash()));
        v
    }

    fn halt_checks(&self) -> Option<String> {
        if self.desk.paths.kill_active() {
            return Some("kill_switch".into());
        }
        if let Err(e) = self.desk.ctx.s.ensure_execution_allowed() {
            return Some(format!("two-key guard lost: {e}"));
        }
        None
    }

    /// Cancel every open scalper order of this run on the symbol (the position keeps its exchange stop), journal, stop.
    async fn halt(&mut self, why: &str) -> anyhow::Result<()> {
        let tag = self.tag();
        let orders = self.desk.venue.trading().open_orders(Some(&self.cfg.symbol)).await.unwrap_or_default();
        let mut cancelled = Vec::new();
        for o in orders.iter().filter(|o| o["orderLinkId"].as_str().is_some_and(|l| l.starts_with(&tag))) {
            let id = o["orderId"].as_str().unwrap_or_default();
            let r = self.desk.venue.cancel(&self.cfg.symbol, id).await;
            cancelled.push(json!({"order_id": id, "link": o["orderLinkId"], "ok": r.is_ok()}));
        }
        let id = match &self.phase {
            LivePhase::Entering { id, .. } | LivePhase::Open { id, .. } | LivePhase::Closing { id, .. } => Some(id.clone()),
            LivePhase::Flat => None,
        };
        if let Some(id) = id {
            let why = why.to_string();
            let _ = update_trade(&self.root(), &self.state_path(), &id, |t| t.notes.push(format!("scalper HALT: {why} (own orders cancelled, exchange stop kept)"))).await;
        }
        self.j("halt", json!({"reason": why, "cancelled": cancelled, "phase": format!("{:?}", self.phase)}));
        println!("HALT: {why} (own orders cancelled: {}; any position keeps its exchange stop)", cancelled.len());
        Ok(())
    }

    // ------------------------------------------------------------ entry

    async fn try_entry(&mut self, eng: &Engine, side: Side, f: &scalp::Feat, now: i64) -> anyhow::Result<()> {
        let extreme = if side == Side::Long { f.swing_lo } else { f.swing_hi };
        let entry = if side == Side::Long { f.bid } else { f.ask };
        let br = match scalp::bracket(&self.cfg, side, entry, extreme) {
            Ok(b) => b,
            Err(e) => {
                self.j("signal", json!({"side": side, "action": "skip", "reason": e}));
                return Ok(());
            }
        };
        let Some(sigma) = eng.sigma_over(self.live_cfg.sigma_horizon_s) else { return Ok(()) };
        let next = (self.seq + 1).max(1);
        let id = trade_id(&self.live_cfg.link_tag, &self.run_id, next);
        let intent = build_intent(&self.cfg, &id, side, &br, self.inst.tick, sigma, self.p_tp, now, &self.run_id);
        let touch = br.entry as f64 * self.inst.tick;

        // decide and reserve under the desk lock, against the venue snapshot and the shared counters
        let lock = StateLock::acquire(&self.root()).await?;
        let mut st = LiveState::load(&self.state_path())?;
        let snap = self.desk.snapshot().await?;
        let notes = self.desk.reconcile(&mut st, &snap, true).await?;
        let positions = positions_of(&st, &snap.positions);
        let mut syms: Vec<String> = positions.iter().map(|p| p.symbol.clone()).collect();
        syms.push(self.cfg.symbol.clone());
        syms.sort();
        syms.dedup();
        let facts = self.desk.facts(&syms, &snap, &st).await?;
        let stype = facts.get(&self.cfg.symbol).map(|f| f.symbol_type.clone()).unwrap_or_default();
        let fees = self.desk.fee_rates(&self.cfg.symbol, &stype).await;
        let account = LiveAccount { available_usd: snap.available, entries_today: st.entries_today, realised_today_usd: st.realised_today(now), kill: self.desk.paths.kill_active() };
        let gates = self.desk.paths.gates_at(now);
        let d = decide_entry(&intent, touch, &account, &positions, &st, &facts, &fees, &self.placement, &self.desk.cfg, &self.cfg.limits, &self.run, self.max_entries, now, &gates);
        self.j("risk_decision", json!({"intent": intent, "decision": d, "fees": fees, "entries_today": st.entries_today, "reconcile_notes": notes, "covered_by": self.approved_by}));
        if !d.accepted() {
            st.save(&self.state_path())?;
            drop(lock);
            self.backoff_until = now + self.live_cfg.refusal_backoff_s * 1000;
            println!("entry refused: {}", d.rejections.join(", "));
            return Ok(());
        }
        let rule = &self.rules;
        let tr = TradeRec {
            id: id.clone(),
            intent_id: id.clone(),
            symbol: self.cfg.symbol.clone(),
            side: Some(side),
            tier: d.tier.clone().unwrap_or_default(),
            status: "submitting".into(),
            qty: d.qty,
            entry_ref: d.entry,
            mid_at_submit: (f.bid + f.ask) as f64 * 0.5 * self.inst.tick,
            first_peg_price: d.entry,
            stop: d.stop,
            target: d.target,
            risk_to_stop_usd: d.risk_to_stop_usd,
            max_loss_usd: d.max_loss_usd,
            stop_slip_bps: d.stop_slip_bps,
            notional_usd: d.notional_usd,
            submitted_ms: now,
            lease_until_ms: now + self.cfg.entry_timeout_s * 1000 + 120_000,
            approved_by: self.approved_by.clone(),
            versions: self.versions(),
            source: SCALP_SOURCE.into(),
            stop_trigger_by: self.cfg.stop_trigger_by.clone(),
            ..Default::default()
        };
        reserve(&mut st, tr, now);
        st.save(&self.state_path())?;
        drop(lock);
        self.seq = next;
        self.run.entries += 1;

        let lev = self.desk.cfg.leverage.min(rule.max_leverage.max(1.0));
        if let Err(e) = self.desk.apply_leverage(&self.cfg.symbol, lev).await {
            let msg = format!("leverage {lev}: {e:#}");
            update_trade(&self.root(), &self.state_path(), &id, |t| {
                t.status = "cancelled".into();
                t.entry_end_reason = Some("leverage_not_set".into());
                t.notes.push(msg.clone());
            })
            .await?;
            self.j("order", json!({"id": id, "action": "not_sent", "reason": msg}));
            return Ok(());
        }
        let link = entry_link(&self.desk.cfg, &id, 1);
        let req = OrderRequest {
            symbol: self.cfg.symbol.clone(),
            side: side.bybit_open(),
            order_type: "Limit",
            qty: format_step(d.qty, rule.qty_step),
            price: Some(format_step(d.entry, rule.tick_size)),
            time_in_force: "PostOnly".into(),
            take_profit: None,
            stop_loss: Some(format_step(d.stop, rule.tick_size)),
            tp_trigger_by: "LastPrice".into(),
            sl_trigger_by: self.cfg.stop_trigger_by.clone(),
            reduce_only: false,
            order_link_id: link.clone(),
        };
        match self.desk.venue.place(&req).await {
            Ok(oid) => {
                self.known.insert(oid.clone());
                let (o2, l2) = (oid.clone(), link.clone());
                update_trade(&self.root(), &self.state_path(), &id, |t| {
                    t.status = "pending".into();
                    t.order_ids.push(o2);
                    t.link_ids.push(l2);
                })
                .await?;
                self.j("order", json!({"id": id, "leg": "entry", "action": "place", "order_id": oid, "link": link, "side": side, "px": d.entry, "qty": d.qty, "stop": d.stop, "tp": d.target, "tif": "PostOnly", "sl_trigger_by": self.cfg.stop_trigger_by}));
                self.phase = LivePhase::Entering { id, order_id: oid, side, br, qty: d.qty, placed_ms: now, cum: 0.0, cancel_sent: false };
            }
            Err(e) => {
                let msg = format!("{e:#}");
                update_trade(&self.root(), &self.state_path(), &id, |t| {
                    t.status = "cancelled".into();
                    t.entry_end_reason = Some("place_error".into());
                    t.notes.push(msg.clone());
                })
                .await?;
                self.j("order", json!({"id": id, "leg": "entry", "action": "place_error", "error": msg}));
                self.backoff_until = now + self.live_cfg.refusal_backoff_s * 1000;
            }
        }
        Ok(())
    }

    // ------------------------------------------------------------ venue events

    async fn on_order(&mut self, o: &VOrder, now: i64) -> anyhow::Result<()> {
        match self.phase.clone() {
            LivePhase::Entering { id, order_id, side, br, cum, .. } if o.id == order_id => {
                if let LivePhase::Entering { cum: c, .. } = &mut self.phase {
                    *c = o.cum.max(cum);
                }
                if o.status == "Filled" || (order_done(&o.status) && o.cum > 0.0) {
                    let avg = if o.avg > 0.0 { o.avg } else { br.entry as f64 * self.inst.tick };
                    self.opened(&id, side, o.cum, avg, br, now).await?;
                } else if order_done(&o.status) {
                    self.j("cancel", json!({"id": id, "order_id": o.id, "status": o.status}));
                    let st = o.status.clone();
                    update_trade(&self.root(), &self.state_path(), &id, |t| {
                        t.status = "cancelled".into();
                        t.entry_done_ms = Some(now);
                        t.entry_end_reason = Some(st);
                    })
                    .await?;
                    self.phase = LivePhase::Flat;
                }
            }
            LivePhase::Open { id, tp: Some(tp), .. } if o.id == tp && o.status == "Cancelled" => {
                // PostOnly TP rejected (would have crossed: price already beyond the target) -> close the profit at market
                self.j("order", json!({"id": id, "leg": "tp", "action": "venue_cancelled", "order_id": o.id}));
                if let LivePhase::Open { tp, .. } = &mut self.phase {
                    *tp = None;
                }
                self.market_close("tp_postonly_cross").await?;
            }
            LivePhase::Open { id, tx: Some((tx, _)), .. } if o.id == tx && o.status == "Cancelled" && o.cum <= 0.0 => {
                self.j("order", json!({"id": id, "leg": "time_exit", "action": "venue_cancelled", "order_id": o.id}));
                if let LivePhase::Open { tx, .. } = &mut self.phase {
                    *tx = None; // re-placed at the new far touch next second
                }
            }
            _ => {}
        }
        Ok(())
    }

    async fn opened(&mut self, id: &str, side: Side, qty: f64, avg: f64, br: Bracket, now: i64) -> anyhow::Result<()> {
        let r_usd = scalp::r_unit_usd(&self.cfg, qty, &br, self.inst.tick);
        update_trade(&self.root(), &self.state_path(), id, |t| {
            t.status = "open".into();
            t.filled_qty = qty;
            t.avg_entry = avg;
            t.first_fill_ms.get_or_insert(now);
            t.entry_done_ms = Some(now);
            t.entry_end_reason = Some("filled".into());
        })
        .await?;
        self.j("fill", json!({"id": id, "leg": "entry", "side": side, "qty": qty, "avg": avg}));
        // reduce-only PostOnly take profit at exactly reward_r x the stop distance
        let link = format!("{}{id}-tp", self.desk.cfg.link_tag());
        let req = OrderRequest {
            symbol: self.cfg.symbol.clone(),
            side: side.bybit_close(),
            order_type: "Limit",
            qty: format_step(qty, self.rules.qty_step),
            price: Some(self.px(br.tp)),
            time_in_force: "PostOnly".into(),
            take_profit: None,
            stop_loss: None,
            tp_trigger_by: "LastPrice".into(),
            sl_trigger_by: self.cfg.stop_trigger_by.clone(),
            reduce_only: true,
            order_link_id: link.clone(),
        };
        let tp = match self.desk.venue.place(&req).await {
            Ok(oid) => {
                self.known.insert(oid.clone());
                let o2 = oid.clone();
                update_trade(&self.root(), &self.state_path(), id, |t| t.tp_order_id = Some(o2)).await?;
                self.j("order", json!({"id": id, "leg": "tp", "action": "place", "order_id": oid, "link": link, "px": self.px(br.tp)}));
                Some(oid)
            }
            Err(e) => {
                self.j("order", json!({"id": id, "leg": "tp", "action": "place_error", "error": format!("{e:#}")}));
                None
            }
        };
        self.phase = LivePhase::Open {
            id: id.to_string(),
            side,
            qty,
            br,
            r_usd,
            filled_ms: now,
            tp,
            tx: None,
            market_sent: false,
            held_across: false,
            stop_missing_since: None,
            reattached: false,
        };
        Ok(())
    }

    async fn cancel_exits(&mut self) {
        if let LivePhase::Open { tp, tx, .. } = self.phase.clone() {
            for oid in tp.into_iter().chain(tx.map(|x| x.0)) {
                let r = self.desk.venue.cancel(&self.cfg.symbol, &oid).await;
                self.j("cancel", json!({"order_id": oid, "leg": "exit", "ok": r.is_ok()}));
            }
            if let LivePhase::Open { tp, tx, .. } = &mut self.phase {
                *tp = None;
                *tx = None;
            }
        }
    }

    async fn market_close(&mut self, why: &str) -> anyhow::Result<()> {
        let LivePhase::Open { id, qty, market_sent: false, .. } = self.phase.clone() else { return Ok(()) };
        self.cancel_exits().await;
        let st = LiveState::load(&self.state_path())?;
        let tr = st.trades.iter().find(|t| t.id == id).cloned().context("scalper trade missing from the live state")?;
        let r = self.desk.close_qty(&tr, qty, &self.rules, why).await;
        self.j("order", json!({"id": id, "leg": "market_close", "why": why, "ok": r.is_ok(), "order_id": r.as_ref().ok()}));
        if let LivePhase::Open { market_sent, .. } = &mut self.phase {
            *market_sent = r.is_ok();
        }
        Ok(())
    }

    /// Position gone: book the trade through the desk's reconcile (closed P&L), update the run counters.
    async fn try_book_close(&mut self, now: i64) -> anyhow::Result<()> {
        let LivePhase::Closing { id, since_ms } = self.phase.clone() else { return Ok(()) };
        let lock = StateLock::acquire(&self.root()).await?;
        let mut st = LiveState::load(&self.state_path())?;
        let snap = self.desk.snapshot().await?;
        let _ = self.desk.reconcile(&mut st, &snap, true).await?;
        st.save(&self.state_path())?;
        drop(lock);
        let Some(t) = st.trades.iter().find(|t| t.id == id) else { return Ok(()) };
        if t.status != "closed" {
            return Ok(());
        }
        let pnl = t.realised_usd.unwrap_or(0.0);
        self.run.consec_losses = if pnl > 0.0 { 0 } else { self.run.consec_losses + 1 };
        self.j("exit", json!({"id": id, "booked_after_s": (now - since_ms) as f64 / 1000.0, "kind": t.exit_kind, "exit": t.exit_avg, "entry": t.avg_entry, "realised_usd": pnl, "net_r": if t.risk_to_stop_usd > 0.0 { pnl / t.risk_to_stop_usd } else { 0.0 }, "consec_losses": self.run.consec_losses}));
        self.phase = LivePhase::Flat;
        self.cooldown_until = now + self.cfg.cooldown_s * 1000;
        Ok(())
    }

    /// Per-second management of the open position: time stop, funding, entry cancel conditions.
    async fn manage(&mut self, eng: &Engine, now: i64) -> anyhow::Result<()> {
        let Some(f) = eng.last_feat.clone() else { return Ok(()) };
        match self.phase.clone() {
            LivePhase::Entering { id, order_id, side, placed_ms, cancel_sent: false, .. } => {
                let z = f.ofi_z.get(self.cfg.signal_idx()).copied().flatten().unwrap_or(0.0);
                let opposite = eng.last_signal.as_ref().is_some_and(|(s, _)| *s != side);
                let why = if now - placed_ms >= self.cfg.entry_timeout_s * 1000 {
                    Some("timeout")
                } else if z * side.sign() < 0.0 {
                    Some("z_crossed_zero")
                } else if opposite {
                    Some("opposite_signal")
                } else {
                    None
                };
                if let Some(w) = why {
                    let r = self.desk.venue.cancel(&self.cfg.symbol, &order_id).await;
                    self.j("cancel", json!({"id": id, "order_id": order_id, "reason": w, "ok": r.is_ok()}));
                    if let LivePhase::Entering { cancel_sent, .. } = &mut self.phase {
                        *cancel_sent = true; // the order update (WS / REST) finishes the transition
                    }
                }
            }
            LivePhase::Open { id, side, qty, br, r_usd, filled_ms, tx, held_across, market_sent: false, .. } => {
                let mid = 0.5 * (f.bid + f.ask) as f64 * self.inst.tick;
                let open_r = if r_usd > 0.0 { side.sign() * (mid - br.entry as f64 * self.inst.tick) * qty / r_usd } else { 0.0 };
                match scalp::funding_action(&self.cfg, side, now, eng.next_funding_ms, open_r) {
                    FundingAct::Flatten => return self.market_close("funding_flat").await,
                    FundingAct::HoldLong if !held_across => {
                        self.j("funding_hold", json!({"id": id, "open_r": open_r, "rate": eng.funding_rate, "funding_ms": eng.next_funding_ms}));
                        if let LivePhase::Open { held_across, .. } = &mut self.phase {
                            *held_across = true;
                        }
                    }
                    _ => {}
                }
                let ts_ms = (self.cfg.time_stop_min * 60_000.0) as i64;
                match tx {
                    None if now - filled_ms >= ts_ms && !held_across => {
                        // time stop: TP out, passive reduce-only exit at the far touch (`-tx`)
                        if let LivePhase::Open { tp: Some(tp), .. } = self.phase.clone() {
                            let r = self.desk.venue.cancel(&self.cfg.symbol, &tp).await;
                            self.j("cancel", json!({"id": id, "leg": "tp", "order_id": tp, "reason": "time_stop", "ok": r.is_ok()}));
                            if let LivePhase::Open { tp, .. } = &mut self.phase {
                                *tp = None;
                            }
                        }
                        let px = if side == Side::Long { f.ask } else { f.bid };
                        let link = format!("{}{id}-tx", self.desk.cfg.link_tag());
                        let req = OrderRequest {
                            symbol: self.cfg.symbol.clone(),
                            side: side.bybit_close(),
                            order_type: "Limit",
                            qty: format_step(qty, self.rules.qty_step),
                            price: Some(self.px(px)),
                            time_in_force: "PostOnly".into(),
                            take_profit: None,
                            stop_loss: None,
                            tp_trigger_by: "LastPrice".into(),
                            sl_trigger_by: self.cfg.stop_trigger_by.clone(),
                            reduce_only: true,
                            order_link_id: link.clone(),
                        };
                        if let Ok(oid) = self.desk.venue.place(&req).await {
                            self.known.insert(oid.clone());
                            self.j("order", json!({"id": id, "leg": "time_exit_passive", "order_id": oid, "link": link, "px": self.px(px)}));
                            if let LivePhase::Open { tx, .. } = &mut self.phase {
                                *tx = Some((oid, now));
                            }
                        }
                    }
                    Some((_, started)) if now - started >= self.cfg.passive_exit_s * 1000 => return self.market_close("time_market").await,
                    _ => {}
                }
            }
            _ => {}
        }
        Ok(())
    }

    /// REST fallback: entry / exit order status and the venue position, then the consistency check.
    async fn poll(&mut self, now: i64) -> anyhow::Result<Option<String>> {
        let watch: Vec<String> = match &self.phase {
            LivePhase::Entering { order_id, .. } => vec![order_id.clone()],
            LivePhase::Open { tp, tx, .. } => tp.iter().cloned().chain(tx.iter().map(|x| x.0.clone())).collect(),
            _ => vec![],
        };
        for oid in watch {
            let o = self.desk.venue.trading().order_status(&self.cfg.symbol, &oid).await?;
            if let Some(o) = o {
                self.on_order(&parse_order(&o), now).await?;
            }
        }
        let pos = self.desk.venue.trading().positions().await?.into_iter().find(|p| p.symbol == self.cfg.symbol);
        let vp = pos
            .map(|p| VPos { side: Some(if p.side == "Sell" { Side::Short } else { Side::Long }), size: p.size, stop: p.stop_loss })
            .unwrap_or_default();
        self.on_position(&vp, now).await?;
        let tag = self.tag();
        let own: Vec<VOrder> = self.desk.venue.trading().open_orders(Some(&self.cfg.symbol)).await?.iter().map(parse_order).filter(|o| o.link.starts_with(&tag)).collect();
        match consistency(&self.phase, &vp, &own, &self.known, self.rules.qty_step, now, self.live_cfg.stop_grace_s * 1000) {
            Check::Ok => Ok(None),
            Check::ReattachStop => {
                let LivePhase::Open { br, .. } = self.phase.clone() else { return Ok(None) };
                let r = self.desk.venue.set_stop(&self.cfg.symbol, &self.px(br.stop), &self.cfg.stop_trigger_by).await;
                self.j("position_update", json!({"action": "reattach_stop", "stop": self.px(br.stop), "ok": r.is_ok()}));
                if let LivePhase::Open { reattached, .. } = &mut self.phase {
                    *reattached = true;
                }
                Ok(None)
            }
            Check::Halt(why) => Ok(Some(why)),
        }
    }

    async fn on_position(&mut self, vp: &VPos, now: i64) -> anyhow::Result<()> {
        match self.phase.clone() {
            LivePhase::Open { id, .. } if vp.size <= 0.0 => {
                self.cancel_exits().await;
                self.j("position_update", json!({"id": id, "action": "position_gone"}));
                self.phase = LivePhase::Closing { id, since_ms: now };
            }
            LivePhase::Open { .. } => {
                if let LivePhase::Open { stop_missing_since, .. } = &mut self.phase {
                    *stop_missing_since = if vp.stop > 0.0 { None } else { stop_missing_since.or(Some(now)) };
                }
            }
            _ => {}
        }
        Ok(())
    }
}

fn kind_for_desk(kind: &str) -> &'static str {
    match kind {
        "order" | "cancel" => "order",
        "fill" => "fill",
        "risk_decision" => "risk_decision",
        "position_update" | "exit" | "funding_hold" => "position_update",
        _ => "decision",
    }
}

/// `atx scalp --mode live`.
pub async fn run_live(ctx: &Ctx, cfg: ScalpCfg, config_file: &str, run_id: &str, max_entries: u32, minutes: Option<u64>) -> anyhow::Result<()> {
    ensure!(valid_run_id(run_id), "--run-id must be 1-16 ASCII letters / digits");
    let desk = Desk::new(ctx)?;
    let live_cfg = cfg.live.clone().context("config has no live section")?;
    let st = LiveState::load(&desk.paths.state())?;
    let verdict: Option<Value> = std::fs::read_to_string(ctx.s.root.join(&live_cfg.verdict_file)).ok().and_then(|t| serde_json::from_str(&t).ok());
    let approval = run_approval_path(&desk.paths.root, run_id);
    let inputs = StartInputs {
        env: ctx.s.environment,
        guard: ctx.s.ensure_execution_allowed().map_err(|e| e.to_string()),
        venue_writable: desk.venue.writable,
        kill: desk.paths.kill_active(),
        verdict: verdict.as_ref(),
        gate_manual: st.gate_manual(&desk.cfg),
        run_approval: approval.exists(),
        max_entries,
    };
    let p_tp = start_preflight(&inputs, &cfg, &desk.cfg)?;
    let approved_by = if inputs.run_approval { format!("scalp-run:{run_id}") } else { "auto".to_string() };

    let md = MarketData::new(&desk.venue.exec, "linear");
    let ins = md.instruments().await?.into_iter().find(|i| i.symbol == cfg.symbol).with_context(|| format!("{} not listed", cfg.symbol))?;
    ensure!(ins.status == "Trading", "{} status {}", cfg.symbol, ins.status);
    let inst = Inst { tick: ins.tick_size, qty_step: ins.qty_step, min_qty: ins.min_qty, min_notional: if ins.min_notional.is_finite() { ins.min_notional } else { 5.0 }, funding_interval_ms: ins.funding_interval_min.max(1) as i64 * 60_000 };
    let rules = at_core::types::InstrumentRules {
        symbol: ins.symbol.clone(),
        tick_size: ins.tick_size,
        qty_step: ins.qty_step,
        min_qty: ins.min_qty,
        max_qty: ins.max_qty,
        max_mkt_qty: ins.max_mkt_qty,
        min_notional: inst.min_notional,
        max_leverage: ins.max_leverage,
    };
    let placement = DecideConfig::load(&ctx.s.root, ctx.risk.clone())?.placement;
    let journal_dir = ctx.s.abs(&ctx.s.paths.journal).join("scalp");
    let mut rn = Runner {
        desk,
        cfg: cfg.clone(),
        live_cfg: live_cfg.clone(),
        inst: inst.clone(),
        rules,
        run_id: run_id.to_string(),
        approved_by,
        p_tp,
        max_entries,
        placement,
        phase: LivePhase::Flat,
        run: RunCounters::default(),
        seq: 0,
        known: HashSet::new(),
        cooldown_until: 0,
        backoff_until: 0,
        journal_dir,
    };
    rn.j("run_start", json!({"symbol": cfg.symbol, "config_file": config_file, "config": cfg, "p_tp": p_tp, "verdict": verdict, "max_entries": max_entries, "run_approval": approval.display().to_string(), "pid": std::process::id()}));
    println!("atx scalp LIVE {} run {run_id} p_tp {p_tp:.4} (CI lower bound) max entries {max_entries} covered by {}", cfg.symbol, rn.approved_by);

    // streams: public book / trades / ticker, private order / execution / position
    let ws_private = ctx.s.bybit_endpoints(ctx.s.environment)?.ws_private.clone();
    let public = ctx.s.bybit_endpoints(ctx.s.venues.bybit.market_data_env)?.ws_public.clone();
    let creds = config::bybit_credentials(ctx.s.environment).map(|(k, s)| Credentials { api_key: k, api_secret: s }).context("no API credentials")?;
    let (tx, mut rx) = tokio::sync::mpsc::channel::<WsMessage>(50_000);
    let pub_task = tokio::spawn(ws::run(public, vec![format!("orderbook.{}.{}", cfg.depth, cfg.symbol), format!("publicTrade.{}", cfg.symbol), format!("tickers.{}", cfg.symbol)], None, tx.clone()));
    let (ptx, mut prx) = tokio::sync::mpsc::channel::<WsMessage>(10_000);
    let priv_task = tokio::spawn(ws::run(ws_private, vec!["order".into(), "execution".into(), "position".into()], Some(creds), ptx));
    drop(tx);

    let mut eng = Engine::new(cfg.clone(), inst, "live");
    let mut tick = tokio::time::interval(Duration::from_millis(cfg.eval_interval_ms.max(200)));
    let start = now_ms();
    let deadline = minutes.map(|m| start + m as i64 * 60_000);
    let mut next_poll = start;
    let mut next_status = start + cfg.status_every_s.max(5) as i64 * 1000;
    let ctrl_c = tokio::signal::ctrl_c();
    tokio::pin!(ctrl_c);
    let mut halted: Option<String> = None;
    loop {
        tokio::select! {
            m = rx.recv() => {
                let Some(m) = m else { halted = Some("public websocket ended".into()); break };
                let now = m.received_ts;
                if m.topic == "__reconnect__" { eng.on_reset(now); }
                else if m.topic.starts_with("orderbook.") { eng.on_book(now, &m.kind, &m.data); }
                else if m.topic.starts_with("publicTrade.") { eng.on_trades(now, &m.data); }
                else if m.topic.starts_with("tickers.") { eng.on_ticker(&m.data); }
            }
            m = prx.recv() => {
                let Some(m) = m else { halted = Some("private websocket ended".into()); break };
                let now = m.received_ts;
                if m.topic == "__reconnect__" { next_poll = now; continue; }
                for x in m.data.as_array().into_iter().flatten().filter(|x| x["symbol"] == cfg.symbol.as_str()) {
                    match m.topic.as_str() {
                        "order" => rn.on_order(&parse_order(x), now).await?,
                        "position" => { let p = parse_position(x); rn.on_position(&p, now).await?; }
                        "execution" => rn.j("execution", json!({"exec_id": x["execId"], "order_id": x["orderId"], "link": x["orderLinkId"], "px": x["execPrice"], "qty": x["execQty"], "fee": x["execFee"], "is_maker": x["isMaker"], "stop_order_type": x["stopOrderType"]})),
                        _ => {}
                    }
                }
            }
            _ = tick.tick() => {
                let now = now_ms();
                if let Some(h) = rn.halt_checks() { halted = Some(h); break; }
                eng.on_second(now);
                rn.manage(&eng, now).await?;
                if now >= next_poll {
                    next_poll = now + live_cfg.poll_s as i64 * 1000;
                    match rn.poll(now).await {
                        Ok(Some(h)) => { halted = Some(h); break; }
                        Ok(None) => {}
                        Err(e) => println!("poll error: {e:#}"),
                    }
                    rn.try_book_close(now).await?;
                }
                let stopping = deadline.is_some_and(|d| now >= d);
                if stopping && rn.phase == LivePhase::Flat { break; }
                if !stopping && rn.phase == LivePhase::Flat && now >= rn.cooldown_until && now >= rn.backoff_until
                    && let Some((side, f)) = eng.last_signal.clone()
                    && !scalp::entry_blocked_by_funding(&cfg, side, now, eng.next_funding_ms)
                {
                    rn.try_entry(&eng, side, &f, now).await?;
                }
                if now >= next_status {
                    next_status = now + cfg.status_every_s.max(5) as i64 * 1000;
                    println!("STATUS LIVE {} | entries {} consec losses {} | {:?}", chrono::Utc::now().format("%H:%M:%SZ"), rn.run.entries, rn.run.consec_losses, rn.phase);
                }
            }
            _ = &mut ctrl_c => {
                // a resting entry is cancelled; an open position keeps its exchange stop and take profit
                if let LivePhase::Entering { order_id, .. } = rn.phase.clone() { let _ = rn.desk.venue.cancel(&cfg.symbol, &order_id).await; }
                rn.j("run_end", json!({"reason": "ctrl-c", "phase": format!("{:?}", rn.phase)}));
                break;
            }
        }
    }
    if let Some(h) = &halted {
        rn.halt(h).await?;
    } else {
        rn.j("run_end", json!({"reason": "done", "entries": rn.run.entries}));
    }
    pub_task.abort();
    priv_task.abort();
    if let Some(h) = halted {
        bail!("scalper halted: {h}");
    }
    Ok(())
}

// ---------------------------------------------------------------- tests

#[cfg(test)]
mod tests {
    use super::*;
    use at_core::types::InstrumentRules;

    fn cfg() -> ScalpCfg {
        ScalpCfg::load(&Path::new(env!("CARGO_MANIFEST_DIR")).join("../../config/scalp_kava.yaml")).unwrap()
    }
    fn live() -> LiveSmallCfg {
        LiveSmallCfg::load(&Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")).unwrap()
    }
    fn verdict(c: &ScalpCfg, v: &str, lo: f64) -> Value {
        json!({"hypothesis": "H-SCALP1", "cfg_hash": c.hash(), "verdict": v, "n": 30, "tp_hit_rate": 0.8, "tp_hit_ci95": [lo, 0.93]})
    }
    fn inputs(v: Option<&Value>) -> StartInputs<'_> {
        StartInputs { env: Environment::Mainnet, guard: Ok(()), venue_writable: true, kill: false, verdict: v, gate_manual: true, run_approval: true, max_entries: 12 }
    }

    #[test]
    fn p_tp_is_the_ci_lower_bound_of_the_tp_hit_rate() {
        let c = cfg();
        assert_eq!(p_tp_from_verdict(&verdict(&c, "supported", 0.62), &c.hash()).unwrap(), 0.62);
        assert!(p_tp_from_verdict(&verdict(&c, "rejected", 0.62), &c.hash()).is_err(), "failing verdict");
        assert!(p_tp_from_verdict(&verdict(&c, "supported", 0.62), "0000").is_err(), "another strategy hash");
        assert!(p_tp_from_verdict(&verdict(&c, "supported", 0.0), &c.hash()).is_err(), "lower bound 0");
        assert!(p_tp_from_verdict(&verdict(&c, "supported", 0.85), &c.hash()).is_err(), "lower bound above the rate");
        let mut v = verdict(&c, "supported", 0.6);
        v["n"] = json!(29);
        assert!(p_tp_from_verdict(&v, &c.hash()).is_err(), "fewer than 30 trades");
        // the EV gate with that p: net E[R] = p (|tp - e| - round trip) / unit - (1 - p), as in decide_live
        let (e, s, tp) = (0.02900, 0.02892, 0.02916);
        let fees = FeeRates { maker_bps: 2.0, taker_bps: 5.5, slippage_bps: 2.0 };
        let l = live();
        let ebps = lv::entry_bps(&fees, &l);
        let unit = lv::unit_loss(e, s, ebps, fees.taker_bps + fees.slippage_bps);
        let rt = e * (ebps + fees.taker_bps + fees.slippage_bps) / 1e4;
        let er = |p: f64| p * ((tp - e) - rt) / unit - (1.0 - p);
        assert!(er(0.62) > 0.10, "p_tp 0.62 passes the 0.10 gate: {}", er(0.62));
        assert!(er(0.45) < 0.10, "p_tp 0.45 (the lower bound of a weaker sample) does not: {}", er(0.45));
    }

    #[test]
    fn start_refusal_paths() {
        let (c, l) = (cfg(), live());
        let ok = verdict(&c, "supported", 0.62);
        assert_eq!(start_preflight(&inputs(Some(&ok)), &c, &l).unwrap(), 0.62);
        let e = start_preflight(&inputs(None), &c, &l).unwrap_err();
        assert!(format!("{e:#}").contains("no H-SCALP1 verdict"), "{e:#}");
        let bad = verdict(&c, "rejected", 0.62);
        assert!(format!("{:#}", start_preflight(&inputs(Some(&bad)), &c, &l).unwrap_err()).contains("verdict is"));
        let x = StartInputs { run_approval: false, ..inputs(Some(&ok)) };
        assert!(format!("{:#}", start_preflight(&x, &c, &l).unwrap_err()).contains("run approval"));
        let x = StartInputs { run_approval: false, gate_manual: false, ..inputs(Some(&ok)) };
        assert!(start_preflight(&x, &c, &l).is_ok(), "gate auto: no run approval needed");
        let x = StartInputs { guard: Err("no key".into()), ..inputs(Some(&ok)) };
        assert!(format!("{:#}", start_preflight(&x, &c, &l).unwrap_err()).contains("guard"));
        let x = StartInputs { kill: true, ..inputs(Some(&ok)) };
        assert!(format!("{:#}", start_preflight(&x, &c, &l).unwrap_err()).contains("KILL"));
        let x = StartInputs { env: Environment::Demo, ..inputs(Some(&ok)) };
        assert!(start_preflight(&x, &c, &l).is_err());
        let x = StartInputs { max_entries: 13, ..inputs(Some(&ok)) };
        assert!(start_preflight(&x, &c, &l).is_err(), "--max-entries above limits.max_entries_per_run");
        assert!(valid_run_id("r1115a") && !valid_run_id("a-b") && !valid_run_id(""));
    }

    fn facts() -> HashMap<String, SymbolFacts> {
        let rule = |s: &str, tick: f64, step: f64| InstrumentRules { symbol: s.into(), tick_size: tick, qty_step: step, min_qty: step, max_qty: 1e9, max_mkt_qty: 1e9, min_notional: 5.0, max_leverage: 25.0 };
        let mut m = HashMap::new();
        m.insert(
            "KAVAUSDT".into(),
            SymbolFacts { symbol: "KAVAUSDT".into(), rules: Some(rule("KAVAUSDT", 0.00001, 0.1)), usdt_linear_perp: true, turnover_24h: 18.6e6, rank: 200, best_bid: 0.02899, best_ask: 0.02901, bid_depth_usd: 5000.0, ask_depth_usd: 5000.0, btc_corr: Some(0.9), ..Default::default() },
        );
        m.insert(
            "ETHUSDT".into(),
            SymbolFacts { symbol: "ETHUSDT".into(), rules: Some(rule("ETHUSDT", 0.01, 0.01)), usdt_linear_perp: true, turnover_24h: 5e9, rank: 2, best_bid: 3000.0, best_ask: 3000.01, bid_depth_usd: 1e6, ask_depth_usd: 1e6, btc_corr: Some(0.9), ..Default::default() },
        );
        m
    }

    fn desk_trade(id: &str, sym: &str, side: Side, status: &str, realised: Option<f64>, closed: Option<i64>) -> TradeRec {
        TradeRec { id: id.into(), symbol: sym.into(), side: Some(side), status: status.into(), qty: 0.01, first_peg_price: 3000.0, max_loss_usd: 1.5, realised_usd: realised, closed_ms: closed, ..Default::default() }
    }

    #[tokio::test]
    async fn shared_counters_with_the_desk_state() {
        let root = std::env::temp_dir().join(format!("scalp-live-test-{}-{}", std::process::id(), now_ms()));
        let state = root.join("state.json");
        let now = 1_800_000_000_000_i64 + 3 * 3_600_000;
        let (c, l) = (cfg(), live());
        let fees = FeeRates { maker_bps: 2.0, taker_bps: 5.5, slippage_bps: 2.0 };
        let placement = PlacementCfg { min_stop_distance_sigma: 0.5, max_stop_distance_sigma: 4.0, stop_cap_bounds_sigma: None };
        // the desk already has 18 entries today, a closed loss and an open ETH short
        let mut st = LiveState { day: at_core::time::format_date(now), entries_today: 18, ..Default::default() };
        st.trades.push(desk_trade("d1", "BTCUSDT", Side::Long, "closed", Some(-2.0), Some(now - 60_000)));
        st.trades.push(desk_trade("d2", "ETHUSDT", Side::Short, "open", None, None));
        st.save(&state).unwrap();
        let venue = vec![at_bybit::trading::VenuePosition {
            symbol: "ETHUSDT".into(),
            side: "Sell".into(),
            size: 0.01,
            avg_price: 3000.0,
            mark_price: 3000.0,
            stop_loss: 3050.0,
            take_profit: 0.0,
            unrealised_pnl: 0.1,
            cur_realised_pnl: 0.0,
            created_ts: now,
        }];
        let br = Bracket { entry: 2899, stop: 2891, tp: 2915, d: 8 };
        let run = RunCounters::default();
        let decide = |st: &LiveState, side: Side, id: &str, run: &RunCounters| {
            let intent = build_intent(&c, id, side, &br, 0.00001, 0.0016, 0.62, now, "t1");
            let positions = positions_of(st, &venue);
            let acct = LiveAccount { available_usd: 100.0, entries_today: st.entries_today, realised_today_usd: st.realised_today(now), kill: false };
            let gates = lv::LiveGates {
                now_ms: now,
                file: Some(lv::GatesFile {
                    written_ms: now - 1000,
                    calendar: lv::GateCalendar { ok: true, reason: None },
                    jump: [("BTCUSDT".to_string(), lv::JumpGate { ratio: Some(1.0), ..Default::default() })].into_iter().collect(),
                    ..Default::default()
                }),
                error: None,
            };
            decide_entry(&intent, 0.02899, &acct, &positions, st, &facts(), &fees, &placement, &l, &c.limits, run, 12, now, &gates)
        };
        let st0 = LiveState::load(&state).unwrap();
        let d = decide(&st0, Side::Long, "sc-t1-1", &run);
        assert!(d.accepted(), "{:?}", d.details);
        assert!(d.notional_usd <= 20.0 + 1e-9, "tier cap tightened to the scalper's 20 USD: {}", d.notional_usd);
        assert_eq!(d.day_pnl_usd, -2.0, "the desk's realised loss is in the day P&L the scalper entry is checked against");
        // a SHORT next to the open desk ETH short: same side, both btc_beta -> refused by the scalper rule
        let d = decide(&st0, Side::Short, "sc-t1-1", &run);
        assert!(d.rejections.iter().any(|r| r == "scalp_correlated_same_side"), "{:?}", d.rejections);

        // reserve through the desk's lock and schema
        let id = trade_id("sc", "t1", 1);
        let tr = TradeRec { id: id.clone(), symbol: "KAVAUSDT".into(), side: Some(Side::Long), status: "submitting".into(), qty: 600.0, first_peg_price: 0.02899, max_loss_usd: 0.2, source: SCALP_SOURCE.into(), ..Default::default() };
        let st1 = reserve_in_state(&root, &state, tr, now).await.unwrap();
        assert_eq!(st1.entries_today, 19, "one entries_today for the desk and the scalper");
        let back = LiveState::load(&state).unwrap();
        assert_eq!(back.entries_today, 19);
        assert!(back.trades.iter().any(|t| t.id == id && t.source == "scalp" && t.active()));
        let link = entry_link(&l, &id, 1);
        assert!(link.starts_with("lsb-sc-"), "{link}: a bot order (lsb-) tagged as the scalper's");
        assert_eq!(positions_of(&back, &venue).len(), 2, "open slots / same side / gross see the desk ETH short and the scalper entry");

        // the 20th entry of the day is the last one: decide_live counts the scalper's entries too
        let mut st2 = back.clone();
        st2.entries_today = 20;
        let d = decide(&st2, Side::Long, "sc-t1-2", &run);
        assert!(d.rejections.iter().any(|r| r == "max_entries_per_day"), "{:?}", d.rejections);
        assert!(d.rejections.iter().any(|r| r == "duplicate_symbol"), "one KAVA trade at a time across desk and scalper");

        // scalper sub-limits on top
        let mut st3 = back.clone();
        st3.trades.retain(|t| t.id != id);
        st3.trades.push(TradeRec { id: "sc-t1-0".into(), symbol: "KAVAUSDT".into(), side: Some(Side::Long), status: "closed".into(), realised_usd: Some(-3.0), closed_ms: Some(now - 1000), source: SCALP_SOURCE.into(), ..Default::default() });
        let d = decide(&st3, Side::Long, "sc-t1-2", &run);
        assert!(d.rejections.iter().any(|r| r == "scalp_day_loss_stop"), "{:?}", d.rejections);
        let d = decide(&st0, Side::Long, "sc-t1-2", &RunCounters { entries: 12, consec_losses: 4 });
        assert!(d.rejections.iter().any(|r| r == "scalp_max_entries_per_run"));
        assert!(d.rejections.iter().any(|r| r == "scalp_consecutive_losses"));

        // updates go through the same lock
        update_trade(&root, &state, &id, |t| t.status = "open".into()).await.unwrap();
        assert_eq!(LiveState::load(&state).unwrap().trades.iter().find(|t| t.id == id).unwrap().status, "open");
        std::fs::remove_dir_all(&root).ok();
    }

    #[test]
    fn inconsistencies_halt() {
        let known: HashSet<String> = ["e1".to_string()].into();
        let br = Bracket { entry: 2899, stop: 2891, tp: 2915, d: 8 };
        let open = LivePhase::Open { id: "x".into(), side: Side::Long, qty: 600.0, br, r_usd: 0.1, filled_ms: 0, tp: None, tx: None, market_sent: false, held_across: false, stop_missing_since: None, reattached: false };
        let pos = VPos { side: Some(Side::Long), size: 600.0, stop: 0.02891 };
        assert_eq!(consistency(&open, &pos, &[], &known, 0.1, 0, 10_000), Check::Ok);
        let unknown = VOrder { id: "zz".into(), link: "lsb-sc-x-9".into(), status: "New".into(), ..Default::default() };
        assert!(matches!(consistency(&open, &pos, &[unknown], &known, 0.1, 0, 10_000), Check::Halt(_)), "unknown scalper order");
        assert!(matches!(consistency(&LivePhase::Flat, &pos, &[], &known, 0.1, 0, 10_000), Check::Halt(_)), "position while flat");
        let bigger = VPos { size: 900.0, ..pos };
        assert!(matches!(consistency(&open, &bigger, &[], &known, 0.1, 0, 10_000), Check::Halt(_)), "size mismatch");
        let nostop = VPos { stop: 0.0, ..pos };
        let missing = |since: i64, re: bool| {
            let mut p = open.clone();
            if let LivePhase::Open { stop_missing_since, reattached, .. } = &mut p {
                *stop_missing_since = Some(since);
                *reattached = re;
            }
            p
        };
        assert_eq!(consistency(&missing(0, false), &nostop, &[], &known, 0.1, 5_000, 10_000), Check::Ok, "inside the grace");
        assert_eq!(consistency(&missing(0, false), &nostop, &[], &known, 0.1, 11_000, 10_000), Check::ReattachStop);
        assert!(matches!(consistency(&missing(0, true), &nostop, &[], &known, 0.1, 11_000, 10_000), Check::Halt(_)), "still no stop after the re-attach");
        // parsing
        let o = parse_order(&json!({"orderId": "e1", "orderLinkId": "lsb-sc-t1-1-1", "orderStatus": "PartiallyFilled", "cumExecQty": "100.5", "avgPrice": "0.02899", "reduceOnly": false}));
        assert_eq!((o.cum, o.status.as_str()), (100.5, "PartiallyFilled"));
        let p = parse_position(&json!({"symbol": "KAVAUSDT", "side": "Sell", "size": "600", "avgPrice": "0.029", "stopLoss": "0.0291"}));
        assert_eq!((p.side, p.size, p.stop), (Some(Side::Short), 600.0, 0.0291));
        assert_eq!(parse_position(&json!({"side": "", "size": "0"})).side, None);
    }

    #[test]
    fn tightened_profile_is_never_looser() {
        let (c, l) = (cfg(), live());
        let t = tightened(&l, &c.limits);
        for (a, b) in t.tiers.iter().zip(&l.tiers) {
            assert!(a.notional_cap_usd <= b.notional_cap_usd && a.notional_cap_usd <= 20.0);
        }
        assert_eq!((t.min_net_expected_r, t.max_entries_per_day, t.daily_loss_stop_usd), (l.min_net_expected_r, l.max_entries_per_day, l.daily_loss_stop_usd));
        t.validate().unwrap();
    }
}
