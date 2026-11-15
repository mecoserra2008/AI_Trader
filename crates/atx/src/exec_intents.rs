//! `atx exec-intents`: OrderIntents (JSONL, agent K) -> risk engine -> paper broker. Paper / replay ONLY.
//!
//!   atx exec-intents --file intents.jsonl --mode replay|paper [--equity 100] [--bars-dir DIR] [--freq 1h] [--as-of ISO]
//!
//! - replay: a fresh paper book walks the bars forward from the first intent; each intent is decided at
//!   its timestamp rounded **up** to the bar grid (only bars closed by then are known), and its orders
//!   fill from the next bar on.
//! - paper: one pass at now on a persistent book (`data/exec/paper/<run>/state.json`): advance it over
//!   the bars completed since the last pass, then decide the new intents; their orders fill next pass.
//!   `--as-of` replaces "now" (point in time: only bars known by then are read); it may not lie in the
//!   future, and a paper book is never moved backwards past bars it has already processed.
//!
//! Outputs: `data/exec/decisions/<run>.jsonl` (one IntentDecision per intent) and
//! `data/exec/paper/<run>/{fills,trades,cancels,equity}.jsonl`, `positions.json`, `summary.json`,
//! `state.json`, plus journal events (session_start, risk_decision, order, fill, session_end).
//! `--mode live` and a mainnet environment are refused: nothing here can reach a venue.
//! The kill switch `data/exec/KILL` cancels and flattens the paper book and rejects new entries.

use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::io::Write as _;
use std::path::{Path, PathBuf};

use anyhow::{Context, bail, ensure};
use at_core::config::Environment;
use at_core::time::{Freq, Ms, format_date, now_ms, to_datetime};
use at_core::types::InstrumentRules;
use at_exec::paper::{BookOrder, PaperBook, kill_switch_active};
use at_risk::{AccountState, DecideConfig, IntentDecision, IntentRejection, OrderIntent, Portfolio, decide};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use crate::cmd::Ctx;
use crate::exec::{Journal, journal_root};
use crate::lake_read::{Bar, read_bars, read_bars_csv, read_instruments};

#[derive(Clone, Debug)]
pub struct ExecArgs {
    pub file: PathBuf,
    pub mode: String,
    pub equity: f64,
    pub bars_dir: Option<PathBuf>,
    pub freq: String,
    pub price_type: String,
    pub run: Option<String>,
    pub out_dir: Option<PathBuf>,
    pub kill_file: Option<PathBuf>,
    pub limit_ttl_bars: u32,
    pub max_intent_age_bars: u32,
    pub no_journal: bool,
    /// Clock override (RFC 3339 or YYYY-MM-DD): the pass runs as of this time instead of now.
    pub as_of: Option<String>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Mode {
    Replay,
    Paper,
}

impl Mode {
    pub fn as_str(self) -> &'static str {
        match self {
            Mode::Replay => "replay",
            Mode::Paper => "paper",
        }
    }
}

/// Only replay and paper exist. "live" (or anything else) is refused.
pub fn parse_mode(s: &str) -> anyhow::Result<Mode> {
    match s.trim().to_ascii_lowercase().as_str() {
        "replay" => Ok(Mode::Replay),
        "paper" => Ok(Mode::Paper),
        "live" => bail!("--mode live is refused: exec-intents is paper/replay only and never sends orders to a venue"),
        other => bail!("unknown --mode {other:?} (replay | paper)"),
    }
}

/// Defensive: even a paper run refuses to start while the configured environment is mainnet.
pub fn ensure_not_mainnet(env: Environment) -> anyhow::Result<()> {
    ensure!(env != Environment::Mainnet, "exec-intents is refused while environment = mainnet (paper/replay only; set environment to demo)");
    Ok(())
}

/// One OrderIntent per non-empty line. Symbols are upper-cased; an intent with mode "live" refuses the file.
pub fn parse_intents(text: &str) -> anyhow::Result<Vec<OrderIntent>> {
    let mut out = vec![];
    for (n, line) in text.lines().enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        let mut i: OrderIntent = serde_json::from_str(line).with_context(|| format!("intents line {}: not an OrderIntent (docs/EXECUTION_PIPELINE.md)", n + 1))?;
        i.symbol = i.symbol.trim().to_uppercase();
        ensure!(!i.intent_id.trim().is_empty(), "intents line {}: empty intent_id", n + 1);
        ensure!(!i.symbol.is_empty(), "intents line {}: empty symbol", n + 1);
        ensure!(i.ts_ms >= 0, "intents line {}: negative ts_ms", n + 1);
        if i.mode.as_deref().is_some_and(|m| m.eq_ignore_ascii_case("live")) {
            bail!("intents line {}: mode \"live\" is refused (paper/replay only)", n + 1);
        }
        out.push(i);
    }
    Ok(out)
}

/// Decision time: the intent timestamp rounded up to the bar grid, so only bars closed by then are known.
pub fn effective_ts(ts: Ms, bar_ms: Ms) -> Ms {
    ts.div_euclid(bar_ms) * bar_ms + if ts.rem_euclid(bar_ms) == 0 { 0 } else { bar_ms }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct DayCounters {
    pub date: String,
    pub start_equity: f64,
    pub trades_opened: u32,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RunState {
    pub run: String,
    pub freq: String,
    pub start_equity: f64,
    pub book: PaperBook,
    pub last_bar_ts: BTreeMap<String, Ms>,
    pub day: Option<DayCounters>,
    pub seen: BTreeSet<String>,
}

impl RunState {
    pub fn new(run: &str, freq: &str, equity: f64) -> Self {
        Self { run: run.into(), freq: freq.into(), start_equity: equity, book: PaperBook::new(run, equity), last_bar_ts: BTreeMap::new(), day: None, seen: BTreeSet::new() }
    }

    fn roll_day(&mut self, ts: Ms, equity_log: &mut Vec<Value>) {
        let date = format_date(ts);
        if self.day.as_ref().is_some_and(|d| d.date == date) {
            return;
        }
        let eq = self.book.marked_equity();
        equity_log.push(json!({"date": date, "ts": ts, "equity": eq, "cash": self.book.cash, "open_positions": self.book.positions.len(), "pending_orders": self.book.orders.len()}));
        self.day = Some(DayCounters { date, start_equity: eq, trades_opened: 0 });
    }
}

/// Everything one pass produced (pure engine output; the caller writes files and the journal).
#[derive(Default)]
pub struct PassOutput {
    pub decisions: Vec<IntentDecision>,
    /// Decision time per decision (same order).
    pub decided_at: Vec<Ms>,
    pub orders: Vec<BookOrder>,
    pub fills: Vec<at_exec::paper::PaperFill>,
    pub trades: Vec<at_exec::paper::PaperTrade>,
    pub cancels: Vec<at_exec::paper::PaperCancel>,
    pub equity: Vec<Value>,
    pub skipped_seen: usize,
    pub bars: usize,
    pub kill_fired: bool,
}

pub struct EngineCfg<'a> {
    pub mode: Mode,
    pub bar_ms: Ms,
    pub now: Ms,
    pub limit_ttl_bars: u32,
    pub max_intent_age_bars: u32,
    pub rules: &'a HashMap<String, InstrumentRules>,
    pub decide: &'a DecideConfig,
    /// Polled at the start and at every new bar timestamp.
    pub kill: &'a dyn Fn() -> bool,
}

/// Merge intents and bars in time order and run them through the risk engine and the paper book.
pub fn run_pass(state: &mut RunState, mut intents: Vec<OrderIntent>, bars: &HashMap<String, Vec<Bar>>, cfg: &EngineCfg) -> PassOutput {
    let mut out = PassOutput::default();
    intents.retain(|i| {
        let fresh = cfg.mode == Mode::Replay || !state.seen.contains(&i.intent_id);
        if !fresh {
            out.skipped_seen += 1;
        }
        fresh
    });
    // Paper: one pass at now. The book first advances over the completed bars, then every new intent is
    // decided at now and its orders fill from the next bar on (processing latency counts against us).
    let floor = if cfg.mode == Mode::Paper { effective_ts(cfg.now, cfg.bar_ms) } else { Ms::MIN };
    let mut queue: Vec<(Ms, OrderIntent)> = intents.into_iter().map(|i| (effective_ts(i.ts_ms, cfg.bar_ms).max(floor), i)).collect();
    queue.sort_by_key(|(e, _)| *e); // stable: file order within a decision time

    let mut timeline: Vec<(Ms, &str, &Bar)> = bars
        .iter()
        .flat_map(|(s, v)| {
            let last = state.last_bar_ts.get(s).copied().unwrap_or(Ms::MIN);
            v.iter().filter(move |b| b.ts > last && b.close_ts <= cfg.now).map(move |b| (b.ts, s.as_str(), b))
        })
        .collect();
    timeline.sort_by(|a, b| (a.0, a.1).cmp(&(b.0, b.1)));

    let mut killed = (cfg.kill)();
    let mut qi = 0;
    let mut last_ts = None;
    let step = |state: &mut RunState, out: &mut PassOutput, t: Ms, killed: &mut bool, qi: &mut usize, upto: Option<Ms>| {
        state.roll_day(t, &mut out.equity);
        let k = (cfg.kill)();
        if k && (!state.book.orders.is_empty() || !state.book.positions.is_empty()) {
            state.book.kill(t);
            out.kill_fired = true;
        }
        *killed = k;
        while *qi < queue.len() && upto.is_none_or(|u| queue[*qi].0 <= u) {
            let (e, intent) = &queue[*qi];
            process_intent(state, out, intent, (*e).max(t), *killed, cfg);
            *qi += 1;
        }
    };
    if killed && (!state.book.orders.is_empty() || !state.book.positions.is_empty()) {
        let t = timeline.first().map(|x| x.0).unwrap_or(cfg.now);
        state.book.kill(t);
        out.kill_fired = true;
    }
    for &(t, sym, b) in &timeline {
        if last_ts != Some(t) {
            step(state, &mut out, t, &mut killed, &mut qi, Some(t));
            last_ts = Some(t);
        }
        state.book.on_bar(sym, b.ts, b.open, b.high, b.low, b.close);
        state.last_bar_ts.insert(sym.to_string(), b.ts);
        out.bars += 1;
    }
    // Intents after the last bar are decided at their own time; their orders wait for the next bars.
    while qi < queue.len() {
        let e = queue[qi].0;
        step(state, &mut out, e, &mut killed, &mut qi, Some(e));
    }
    let (f, tr, c) = state.book.drain();
    out.fills = f;
    out.trades = tr;
    out.cancels = c;
    let eq = state.book.marked_equity();
    out.equity.push(json!({"date": "end", "ts": last_ts.unwrap_or(cfg.now), "equity": eq, "cash": state.book.cash, "open_positions": state.book.positions.len(), "pending_orders": state.book.orders.len()}));
    out
}

fn process_intent(state: &mut RunState, out: &mut PassOutput, intent: &OrderIntent, at: Ms, killed: bool, cfg: &EngineCfg) {
    let portfolio = Portfolio { entries: state.book.book_entries(), marks: state.book.marks(), kill_switch: killed };
    let day = state.day.clone().unwrap_or(DayCounters { date: format_date(at), start_equity: state.book.marked_equity(), trades_opened: 0 });
    let account = AccountState { equity: state.book.marked_equity(), day_start_equity: day.start_equity, trades_opened_today: day.trades_opened, open_positions: vec![] };

    let d = if state.seen.contains(&intent.intent_id) {
        IntentDecision::reject_only(intent, IntentRejection::DuplicateIntent, "intent_id already decided", &account, &portfolio)
    } else if cfg.mode == Mode::Paper && cfg.now - effective_ts(intent.ts_ms, cfg.bar_ms) > cfg.max_intent_age_bars as Ms * cfg.bar_ms {
        IntentDecision::reject_only(intent, IntentRejection::StaleIntent, &format!("older than {} bars at now", cfg.max_intent_age_bars), &account, &portfolio)
    } else {
        decide(intent, &account, &portfolio, cfg.rules, cfg.decide)
    };
    state.seen.insert(intent.intent_id.clone());

    if d.accepted() {
        match d.action.as_str() {
            "open" => {
                let side = intent.side.side().expect("open decisions have a side");
                let order = BookOrder {
                    id: intent.intent_id.clone(),
                    symbol: intent.symbol.clone(),
                    strategy: intent.strategy.clone(),
                    family: intent.family.clone(),
                    side,
                    entry_type: d.entry_type.expect("open decisions have an entry"),
                    price: d.entry.unwrap_or_default(),
                    qty: d.qty,
                    stop: d.stop.unwrap_or_default(),
                    target: d.target,
                    risk_usd: d.risk_usd,
                    fees: cfg.decide.fees.rates(&intent.symbol, Some(&d.fee_class)),
                    submitted_ts: at,
                    ttl_bars: cfg.limit_ttl_bars,
                    bars_seen: 0,
                    peg_slip_bps: d.peg_slip_bps,
                };
                state.book.submit(order.clone());
                out.orders.push(order);
                if let Some(day) = state.day.as_mut() {
                    day.trades_opened += 1;
                }
            }
            "close" => {
                state.book.request_exit(&intent.symbol, &intent.strategy, at);
            }
            _ => {}
        }
    }
    out.decisions.push(d);
    out.decided_at.push(at);
}

// ---------------------------------------------------------------- I/O

fn iso(ms: Ms) -> String {
    to_datetime(ms).to_rfc3339_opts(chrono::SecondsFormat::Secs, true)
}

fn write_jsonl<T: Serialize>(path: &Path, rows: &[T], append: bool) -> anyhow::Result<()> {
    if let Some(d) = path.parent() {
        std::fs::create_dir_all(d)?;
    }
    let mut f = std::fs::OpenOptions::new().create(true).write(true).append(append).truncate(!append).open(path).with_context(|| path.display().to_string())?;
    for r in rows {
        writeln!(f, "{}", serde_json::to_string(r)?)?;
    }
    Ok(())
}

fn write_json<T: Serialize>(path: &Path, v: &T) -> anyhow::Result<()> {
    if let Some(d) = path.parent() {
        std::fs::create_dir_all(d)?;
    }
    let tmp = path.with_extension("json.tmp");
    std::fs::write(&tmp, serde_json::to_string_pretty(v)?)?;
    std::fs::rename(&tmp, path).with_context(|| path.display().to_string())
}

fn valid_run_id(s: &str) -> bool {
    !s.is_empty() && s.len() <= 80 && s.chars().all(|c| c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '.')) && !s.starts_with('.')
}

pub async fn run(ctx: &Ctx, a: ExecArgs) -> anyhow::Result<()> {
    let mode = parse_mode(&a.mode)?;
    ensure_not_mainnet(ctx.s.environment)?;
    ensure!(a.equity.is_finite() && a.equity > 0.0, "--equity must be positive");
    let freq = Freq::parse(&a.freq)?;
    let bar_ms = freq.duration_ms();
    let wall = now_ms();
    let now = match a.as_of.as_deref() {
        Some(s) => {
            let t = at_core::time::parse_ts(s).with_context(|| format!("--as-of {s:?}"))?;
            ensure!(t <= wall + 60_000, "--as-of {s} lies in the future (now {})", iso(wall));
            t
        }
        None => wall,
    };
    let intents = parse_intents(&std::fs::read_to_string(&a.file).with_context(|| a.file.display().to_string())?)?;

    let data_root = ctx.s.abs(&ctx.s.paths.data_root);
    let exec_root = a.out_dir.clone().unwrap_or_else(|| data_root.join("exec"));
    let kill_path = a.kill_file.clone().unwrap_or_else(|| data_root.join("exec").join("KILL"));
    let run_id = a.run.clone().unwrap_or_else(|| match mode {
        Mode::Replay => format!("replay-{}", to_datetime(now).format("%Y%m%dT%H%M%SZ")),
        Mode::Paper => "paper".into(),
    });
    ensure!(valid_run_id(&run_id), "--run {run_id:?}: use letters, digits, '-', '_', '.'");
    let paper_dir = exec_root.join("paper").join(&run_id);
    let decisions_path = exec_root.join("decisions").join(format!("{run_id}.jsonl"));
    let state_path = paper_dir.join("state.json");

    // EV gate by mode: paper / replay use min_net_expected_r_paper (exclusive), never on mainnet (user decision 2026-11-14)
    let mut decide_cfg = DecideConfig::load(&ctx.s.root, ctx.risk.clone())?.for_mode(mode.as_str(), ctx.s.environment == Environment::Mainnet);
    let mut state = match (mode, std::fs::read_to_string(&state_path)) {
        (Mode::Paper, Ok(t)) => {
            let s: RunState = serde_json::from_str(&t).with_context(|| format!("{}: corrupt paper state (fix or move it)", state_path.display()))?;
            ensure!(s.freq == a.freq, "paper run {run_id} uses freq {} (asked {})", s.freq, a.freq);
            s
        }
        _ => RunState::new(&run_id, &a.freq, a.equity),
    };
    if let Some((s, t)) = state.last_bar_ts.iter().find(|(_, t)| **t + bar_ms > now) {
        bail!("paper run {run_id} already processed the {s} bar opening {} (closes after {}): a book is never moved backwards", iso(*t), iso(now));
    }

    let mut symbols: HashSet<String> = intents.iter().map(|i| i.symbol.clone()).collect();
    symbols.extend(state.book.positions.iter().map(|p| p.symbol.clone()));
    symbols.extend(state.book.orders.iter().map(|o| o.symbol.clone()));

    // Instrument rules and fee classes
    let instruments_json = a.bars_dir.as_ref().map(|d| d.join("instruments.json")).filter(|p| p.is_file());
    let rules: HashMap<String, InstrumentRules> = match &instruments_json {
        Some(p) => {
            let v: Vec<InstrumentRules> = serde_json::from_str(&std::fs::read_to_string(p)?).with_context(|| p.display().to_string())?;
            v.into_iter().map(|r| (r.symbol.to_uppercase(), r)).collect()
        }
        None => {
            let (r, types) = read_instruments(ctx.lake.root(), "bybit", ctx.category(), now)?;
            for s in &symbols {
                if let Some(t) = types.get(s) {
                    decide_cfg.fees.set_class_if_absent(s, if t.is_empty() { "crypto" } else { "tradfi" });
                }
            }
            r
        }
    };

    // Bars
    let first_e = intents.iter().map(|i| effective_ts(i.ts_ms, bar_ms)).min().unwrap_or(now);
    let fresh_from = match mode {
        Mode::Replay => first_e - 2 * bar_ms,
        Mode::Paper => now - (a.max_intent_age_bars as Ms + 2) * bar_ms,
    };
    let from = state.last_bar_ts.values().min().map(|t| t + 1).unwrap_or(fresh_from);
    let bars = match &a.bars_dir {
        Some(d) => read_bars_csv(d, &symbols, bar_ms)?,
        None => read_bars(ctx.lake.root(), "bybit", &a.price_type, freq.as_str(), &symbols, from, now)?,
    };
    let missing: Vec<&String> = symbols.iter().filter(|s| bars.get(*s).is_none_or(|v| v.is_empty())).collect();
    if !missing.is_empty() {
        tracing::warn!("no {} bars for {:?}: their orders cannot fill", a.freq, missing);
    }

    let kill = || kill_switch_active(&kill_path);
    let ecfg = EngineCfg { mode, bar_ms, now, limit_ttl_bars: a.limit_ttl_bars, max_intent_age_bars: a.max_intent_age_bars, rules: &rules, decide: &decide_cfg, kill: &kill };
    let n_intents = intents.len();
    let out = run_pass(&mut state, intents, &bars, &ecfg);

    // ---- files
    let append = mode == Mode::Paper;
    let decisions: Vec<Value> = out
        .decisions
        .iter()
        .zip(&out.decided_at)
        .map(|(d, t)| {
            let mut v = serde_json::to_value(d).unwrap_or_default();
            v["decided_at_ms"] = json!(t);
            v["decided_at"] = json!(iso(*t));
            v["mode"] = json!(mode.as_str());
            v["run"] = json!(run_id);
            v
        })
        .collect();
    write_jsonl(&decisions_path, &decisions, append)?;
    write_jsonl(&paper_dir.join("fills.jsonl"), &out.fills, append)?;
    write_jsonl(&paper_dir.join("trades.jsonl"), &out.trades, append)?;
    write_jsonl(&paper_dir.join("cancels.jsonl"), &out.cancels, append)?;
    write_jsonl(&paper_dir.join("equity.jsonl"), &out.equity, append)?;
    let positions: Vec<Value> = state
        .book
        .positions
        .iter()
        .map(|p| {
            let mut v = serde_json::to_value(p).unwrap_or_default();
            v["mark"] = json!(state.book.last_close.get(&p.symbol).map(|x| x.0));
            v["unrealised"] = json!(state.book.unrealised(p));
            v["unrealised_r"] = json!(if p.risk_usd > 0.0 { state.book.unrealised(p) / p.risk_usd } else { 0.0 });
            v
        })
        .collect();
    write_json(&paper_dir.join("positions.json"), &json!({"positions": positions, "pending_orders": state.book.orders}))?;
    write_json(&state_path, &state)?;

    let mut by_reason: BTreeMap<String, u32> = BTreeMap::new();
    for d in &out.decisions {
        for r in &d.rejections {
            *by_reason.entry(r.clone()).or_default() += 1;
        }
    }
    let accepted = out.decisions.iter().filter(|d| d.accepted()).count();
    let rs: Vec<f64> = out.trades.iter().map(|t| t.r_multiple).collect();
    let pnl: f64 = out.trades.iter().map(|t| t.pnl).sum();
    let end_equity = state.book.marked_equity();
    let mut summary = json!({
        "run": run_id, "mode": mode.as_str(), "paper_only": true, "file": a.file.display().to_string(), "as_of": iso(now), "as_of_override": a.as_of.is_some(),
        "stop_cap_source": decide_cfg.stop_cap_source,
        "freq": a.freq, "bars_source": a.bars_dir.as_ref().map(|d| d.display().to_string()).unwrap_or_else(|| format!("lake bybit/{}/{}", a.price_type, a.freq)),
        "intents_in_file": n_intents, "already_decided": out.skipped_seen, "decided": out.decisions.len(),
        "accepted": accepted, "rejected": out.decisions.len() - accepted, "rejections_by_reason": by_reason,
        "orders": out.orders.len(), "fills": out.fills.len(), "cancels": out.cancels.len(), "trades": out.trades.len(),
        "wins": rs.iter().filter(|r| **r > 0.0).count(), "sum_r": rs.iter().sum::<f64>(),
        "mean_r": if rs.is_empty() { Value::Null } else { json!(rs.iter().sum::<f64>() / rs.len() as f64) },
        "realised_pnl": pnl, "start_equity": state.start_equity, "end_equity_marked": end_equity,
        "open_positions": state.book.positions.len(), "pending_orders": state.book.orders.len(),
        "bars_processed": out.bars, "kill_switch_fired": out.kill_fired, "kill_file": kill_path.display().to_string(),
        "missing_bars": missing, "risk": {"risk_per_trade_pct": ctx.risk.risk_per_trade_pct, "max_open_positions": ctx.risk.max_open_positions,
            "max_positions_per_cluster": ctx.risk.max_positions_per_cluster, "max_new_trades_per_day": ctx.risk.max_new_trades_per_day,
            "daily_loss_limit_pct": ctx.risk.daily_loss_limit_pct, "max_gross_exposure_x": ctx.risk.max_gross_exposure_x,
            "min_stop_distance_sigma": decide_cfg.placement.min_stop_distance_sigma, "max_stop_distance_sigma": decide_cfg.placement.max_stop_distance_sigma},
        "fees": decide_cfg.fees.classes,
    });
    summary["risk"]["min_net_expected_r"] = json!(decide_cfg.risk.min_net_expected_r);
    summary["risk"]["net_er_exclusive"] = json!(decide_cfg.net_er_exclusive);
    write_json(&paper_dir.join("summary.json"), &summary)?;

    // ---- journal
    if !a.no_journal {
        let j = Journal::new(journal_root(&ctx.s), &format!("exec-{}-{run_id}", mode.as_str()));
        j.write("session_start", json!({"component": "exec-intents", "mode": mode.as_str(), "paper_only": true, "file": a.file.display().to_string(), "environment": ctx.s.environment.as_str(), "start_equity": state.start_equity, "risk": summary["risk"], "fees": summary["fees"]}))?;
        for d in &decisions {
            j.write("risk_decision", json!({"source": "exec-intents", "decision": d}))?;
        }
        for o in &out.orders {
            j.write("order", json!({"status": "paper", "mode": mode.as_str(), "order": o}))?;
        }
        for f in &out.fills {
            j.write("fill", json!({"status": "paper", "mode": mode.as_str(), "fill": f}))?;
        }
        j.write("session_end", json!({"summary": summary}))?;
    }

    println!("exec-intents {} run={run_id}: {} intents, {} decided ({} accepted, {} rejected), {} already decided", mode.as_str(), n_intents, out.decisions.len(), accepted, out.decisions.len() - accepted, out.skipped_seen);
    for (r, n) in &by_reason {
        println!("  {r:<28} {n}");
    }
    println!(
        "paper book: {} fills, {} trades (sum R {:+.2}), realised {:+.4}, equity {:.4} -> {:.4}, {} open, {} pending{}",
        out.fills.len(),
        out.trades.len(),
        rs.iter().sum::<f64>(),
        pnl,
        state.start_equity,
        end_equity,
        state.book.positions.len(),
        state.book.orders.len(),
        if out.kill_fired { ", KILL fired" } else { "" }
    );
    println!("decisions -> {}\npaper     -> {}", decisions_path.display(), paper_dir.display());
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn live_mode_and_mainnet_are_refused() {
        assert!(parse_mode("live").unwrap_err().to_string().contains("refused"));
        assert!(parse_mode("LIVE").is_err());
        assert!(parse_mode("bogus").is_err());
        assert_eq!(parse_mode("Replay").unwrap(), Mode::Replay);
        assert!(ensure_not_mainnet(Environment::Mainnet).is_err());
        assert!(ensure_not_mainnet(Environment::Demo).is_ok());
        let line = r#"{"intent_id":"x","ts_ms":0,"mode":"live","strategy":"s","family":"trend","symbol":"ethusdt","side":"flat"}"#;
        assert!(parse_intents(line).is_err());
        let ok = parse_intents(&line.replace("live", "paper")).unwrap();
        assert_eq!(ok[0].symbol, "ETHUSDT");
    }

    #[test]
    fn decision_time_rounds_up_to_the_bar_grid() {
        let h = 3_600_000;
        assert_eq!(effective_ts(0, h), 0);
        assert_eq!(effective_ts(h, h), h);
        assert_eq!(effective_ts(h + 1, h), 2 * h);
        assert_eq!(effective_ts(2 * h - 1, h), 2 * h);
    }

    fn cfg_parts() -> (HashMap<String, InstrumentRules>, DecideConfig) {
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
        let (_, risk) = at_core::config::load(&root).unwrap();
        let rules = HashMap::from([("ETHUSDT".to_string(), InstrumentRules { symbol: "ETHUSDT".into(), tick_size: 0.01, qty_step: 0.01, min_qty: 0.01, max_qty: 1e4, max_mkt_qty: 1e3, min_notional: 5.0, max_leverage: 50.0 })]);
        (rules, DecideConfig::load(&root, risk).unwrap())
    }

    fn intent(id: &str, ts: Ms, side: &str) -> OrderIntent {
        serde_json::from_value(json!({"intent_id": id, "ts_ms": ts, "strategy": "tsmom", "family": "trend", "symbol": "ETHUSDT", "asset_class": "crypto", "side": side,
            "entry": {"type": "market", "price": 3000.0}, "stop": 2940.0, "targets": [3150.0], "sigma": {"per_bar": 0.01}, "p_tp": 0.45, "net_er_r": 0.3}))
        .unwrap()
    }

    fn bars(n: i64, px: f64) -> HashMap<String, Vec<Bar>> {
        let h = 3_600_000;
        HashMap::from([("ETHUSDT".to_string(), (0..n).map(|k| Bar { ts: k * h, close_ts: (k + 1) * h, known_ts: (k + 1) * h, open: px, high: px + 5.0, low: px - 5.0, close: px }).collect())])
    }

    #[test]
    fn pass_is_point_in_time_and_counts_entries_per_day() {
        let (rules, dc) = cfg_parts();
        let no_kill = || false;
        let h = 3_600_000;
        let cfg = EngineCfg { mode: Mode::Replay, bar_ms: h, now: 100 * h, limit_ttl_bars: 3, max_intent_age_bars: 2, rules: &rules, decide: &dc, kill: &no_kill };
        let mut st = RunState::new("t", "1h", 1000.0);
        // decided at 2h (rounded up from 1h+1ms): fills at the 2h open, never in the 1h bar
        let out = run_pass(&mut st, vec![intent("a", h + 1, "long"), intent("a", h + 2, "long")], &bars(6, 3000.0), &cfg);
        assert_eq!(out.decided_at[0], 2 * h);
        assert!(out.decisions[0].accepted());
        assert_eq!(out.decisions[1].rejections, vec!["duplicate_intent"]);
        assert_eq!(out.fills[0].ts, 2 * h);
        assert_eq!(st.day.as_ref().unwrap().trades_opened, 1);
        assert_eq!(st.book.positions.len(), 1);
    }

    #[test]
    fn kill_flattens_and_blocks_new_entries() {
        let (rules, dc) = cfg_parts();
        let h = 3_600_000;
        let mut st = RunState::new("t", "1h", 1000.0);
        let no_kill = || false;
        let cfg = EngineCfg { mode: Mode::Replay, bar_ms: h, now: 100 * h, limit_ttl_bars: 3, max_intent_age_bars: 2, rules: &rules, decide: &dc, kill: &no_kill };
        run_pass(&mut st, vec![intent("a", 0, "long")], &bars(3, 3000.0), &cfg);
        assert_eq!(st.book.positions.len(), 1);
        let kill = || true;
        let cfg = EngineCfg { kill: &kill, ..cfg };
        let more = HashMap::from([("ETHUSDT".to_string(), vec![Bar { ts: 3 * h, close_ts: 4 * h, known_ts: 4 * h, open: 3000.0, high: 3001.0, low: 2999.0, close: 3000.0 }])]);
        let out = run_pass(&mut st, vec![intent("b", 3 * h, "long")], &more, &cfg);
        assert!(out.kill_fired);
        assert_eq!(out.trades[0].exit_reason, "kill");
        assert!(st.book.positions.is_empty());
        assert!(out.decisions[0].rejections.contains(&"kill_switch".to_string()));
    }

    #[test]
    fn paper_mode_skips_seen_and_rejects_stale() {
        let (rules, dc) = cfg_parts();
        let h = 3_600_000;
        let no_kill = || false;
        let cfg = EngineCfg { mode: Mode::Paper, bar_ms: h, now: 10 * h, limit_ttl_bars: 3, max_intent_age_bars: 2, rules: &rules, decide: &dc, kill: &no_kill };
        let mut st = RunState::new("p", "1h", 1000.0);
        let out = run_pass(&mut st, vec![intent("old", 2 * h, "long"), intent("new", 9 * h, "long")], &bars(10, 3000.0), &cfg);
        assert_eq!(out.decisions[0].rejections, vec!["stale_intent"]);
        assert!(out.decisions[1].accepted());
        assert_eq!(out.decided_at[1], 10 * h, "decided at now, after the book advanced");
        assert_eq!(st.book.orders.len(), 1, "the new order waits for the next completed bar");
        let out = run_pass(&mut st, vec![intent("new", 9 * h, "long")], &bars(10, 3000.0), &cfg);
        assert_eq!(out.skipped_seen, 1);
        assert!(out.decisions.is_empty());
        assert_eq!(out.bars, 0, "bars are processed once");
    }
}
