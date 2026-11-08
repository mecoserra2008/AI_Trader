//! Execution bridge: decision proposals -> risk engine -> Bybit (demo / testnet).
//!
//!   atx propose --file proposals.json [--submit] [--session-id ID]
//!   atx positions [--manage [--breakeven]]
//!   atx flatten --symbol SYM --yes
//!
//! Everything written to the venue is journaled with the same JSONL envelope as the Python
//! `Journal` (`{ts, session_id, type, ...payload}` under `<journal>/YYYY-MM-DD/<type>.jsonl`).
//! Schema and rules: docs/EXECUTION.md.

use std::collections::{HashMap, HashSet};
use std::io::Write as _;
use std::path::{Path, PathBuf};
use std::time::Duration;

use anyhow::{Context, bail};
use at_bybit::market::MarketData;
use at_bybit::trading::{Trading, VenuePosition};
use at_core::config::{Environment, Settings};
use at_core::num::{Round, round_to_step};
use at_core::time::{format_date, now_ms};
use at_core::types::{InstrumentRules, Position, Side, TradeProposal};
use at_exec::Broker;
use at_exec::bybit::BybitBroker;
use at_risk::{AccountState, Rejection, RiskDecision, RiskEngine};
use chrono::{DateTime, SecondsFormat, Utc};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use crate::cmd::Ctx;

/// Bybit `orderLinkId` limit.
pub const MAX_LINK_ID: usize = 36;
/// How many journal days are scanned for the orders behind open positions.
const JOURNAL_LOOKBACK_DAYS: usize = 60;

// ---------------------------------------------------------------- proposal file

/// Input of `atx propose`. Proposals are ranked best-first; earlier ones consume slots first.
#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProposalFile {
    #[serde(default)]
    pub session_id: Option<String>,
    pub proposals: Vec<TradeProposal>,
    /// Optional symbol -> cluster map for positions already open (cluster cap).
    #[serde(default)]
    pub clusters: HashMap<String, String>,
    /// Free-form provenance (model, prompt version, as_of...). Journaled, never interpreted.
    #[serde(default)]
    pub meta: Option<Value>,
}

pub fn parse_proposal_file(text: &str) -> anyhow::Result<ProposalFile> {
    let mut f: ProposalFile = serde_json::from_str(text).context("proposal file is not valid JSON for the schema in docs/EXECUTION.md")?;
    anyhow::ensure!(!f.proposals.is_empty(), "proposal file has no proposals");
    let mut ids = HashSet::new();
    for p in &mut f.proposals {
        p.symbol = p.symbol.trim().to_uppercase();
        anyhow::ensure!(!p.id.trim().is_empty(), "proposal with empty id");
        anyhow::ensure!(!p.symbol.is_empty(), "proposal {} has an empty symbol", p.id);
        anyhow::ensure!(ids.insert(p.id.clone()), "duplicate proposal id {:?}", p.id);
    }
    f.clusters = f.clusters.into_iter().map(|(k, v)| (k.trim().to_uppercase(), v)).collect();
    Ok(f)
}

/// `--session-id` wins over the file's `session_id`; one of them is required.
pub fn resolve_session(cli: Option<String>, file: Option<String>) -> anyhow::Result<String> {
    let s = cli.or(file).map(|s| s.trim().to_string()).filter(|s| !s.is_empty());
    s.context("no session id: set \"session_id\" in the file or pass --session-id")
}

// ---------------------------------------------------------------- order link id

/// `"<session>-<proposal id>"`, restricted to Bybit's alphabet (letters, digits, `-`, `_`) and at most
/// 36 chars. Longer ids keep a 27-char prefix plus an 8-hex hash of the full id, so two long ids that
/// share a prefix still map to different link ids (Bybit rejects a reused link id: re-submits are idempotent).
pub fn link_id(session: &str, proposal_id: &str) -> String {
    let full: String = format!("{session}-{proposal_id}").chars().map(|c| if c.is_ascii_alphanumeric() || c == '-' || c == '_' { c } else { '_' }).collect();
    if full.len() <= MAX_LINK_ID {
        return full;
    }
    // FNV-1a 32-bit: stable across runs and platforms.
    let h = full.bytes().fold(0x811c_9dc5_u32, |h, b| (h ^ b as u32).wrapping_mul(0x0100_0193));
    format!("{}-{h:08x}", &full[..MAX_LINK_ID - 9])
}

// ---------------------------------------------------------------- environment guard

/// `--submit` and `flatten` only run on demo / testnet. Mainnet is refused here regardless of the
/// two-key guard; that guard (`Settings::ensure_execution_allowed`) is still applied on top.
pub fn ensure_paper_venue(env: Environment, what: &str) -> anyhow::Result<()> {
    match env {
        Environment::Demo | Environment::Testnet => Ok(()),
        Environment::Mainnet => bail!("{what} is refused on mainnet (only demo / testnet); environment = {}", env.as_str()),
    }
}

// ---------------------------------------------------------------- day state

/// `data/state/exec_day.json`: inputs of the daily caps that the venue cannot tell us.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct DayState {
    /// UTC date, YYYY-MM-DD.
    pub date: String,
    /// Equity at the first call of the UTC day.
    pub day_start_equity: f64,
    /// Entry orders submitted today (counted at submission, filled or not).
    pub trades_opened_today: u32,
}

impl DayState {
    /// Keep `prev` if it is for `today`, otherwise start a new day at `equity`.
    pub fn roll(prev: Option<DayState>, today: &str, equity: f64) -> DayState {
        match prev {
            Some(p) if p.date == today => p,
            _ => DayState { date: today.to_string(), day_start_equity: equity, trades_opened_today: 0 },
        }
    }

    pub fn load(path: &Path) -> anyhow::Result<Option<DayState>> {
        match std::fs::read_to_string(path) {
            Ok(t) => Ok(Some(serde_json::from_str(&t).with_context(|| format!("{}: corrupt day state (fix or delete it)", path.display()))?)),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
            Err(e) => Err(e).with_context(|| path.display().to_string()),
        }
    }

    /// Atomic write (temp file + rename).
    pub fn save(&self, path: &Path) -> anyhow::Result<()> {
        if let Some(d) = path.parent() {
            std::fs::create_dir_all(d)?;
        }
        let tmp = path.with_extension("json.tmp");
        std::fs::write(&tmp, serde_json::to_string_pretty(self)?)?;
        std::fs::rename(&tmp, path).with_context(|| path.display().to_string())
    }
}

/// Real money keeps its own state and journal so demo runs can never mix into it.
fn day_state_path(s: &Settings) -> PathBuf {
    let name = if s.environment == Environment::Mainnet { "exec_day_mainnet.json" } else { "exec_day.json" };
    s.abs(&s.paths.data_root).join("state").join(name)
}

// ---------------------------------------------------------------- journal

/// Mirrors `ai_trader.journal.writer.EVENT_TYPES`.
pub const EVENT_TYPES: &[&str] = &[
    "session_start",
    "snapshot",
    "llm_message",
    "tool_call",
    "view",
    "decision",
    "risk_decision",
    "order",
    "fill",
    "position_update",
    "baseline",
    "shadow_outcome",
    "manual_action",
    "session_end",
];

/// Append-only JSONL journal, same layout and envelope as the Python `Journal`.
pub struct Journal {
    root: PathBuf,
    session_id: String,
}

impl Journal {
    pub fn new(root: PathBuf, session_id: &str) -> Self {
        Self { root, session_id: session_id.to_string() }
    }

    pub fn write(&self, event_type: &str, payload: Value) -> anyhow::Result<()> {
        self.write_at(event_type, payload, Utc::now())
    }

    pub fn write_at(&self, event_type: &str, payload: Value, ts: DateTime<Utc>) -> anyhow::Result<()> {
        anyhow::ensure!(EVENT_TYPES.contains(&event_type), "unknown journal event type {event_type:?}");
        let Value::Object(fields) = payload else { bail!("journal payload must be a JSON object") };
        let mut line = serde_json::Map::new();
        // Python: datetime.now(UTC).isoformat() -> 2026-11-07T12:00:00.123456+00:00
        line.insert("ts".into(), json!(ts.to_rfc3339_opts(SecondsFormat::Micros, false)));
        line.insert("session_id".into(), json!(self.session_id));
        line.insert("type".into(), json!(event_type));
        line.extend(fields);
        let dir = self.root.join(ts.format("%Y-%m-%d").to_string());
        std::fs::create_dir_all(&dir)?;
        let mut f = std::fs::OpenOptions::new().create(true).append(true).open(dir.join(format!("{event_type}.jsonl")))?;
        writeln!(f, "{}", Value::Object(line))?;
        Ok(())
    }
}

pub(crate) fn journal_root(s: &Settings) -> PathBuf {
    let j = s.abs(&s.paths.journal);
    if s.environment == Environment::Mainnet { j.with_file_name(format!("{}_mainnet", j.file_name().and_then(|n| n.to_str()).unwrap_or("journal"))) } else { j }
}

/// Events of one type from the last `max_days` day directories, oldest first. Unparseable lines are skipped.
pub fn read_events(root: &Path, event_type: &str, max_days: usize) -> Vec<Value> {
    let Ok(rd) = std::fs::read_dir(root) else { return vec![] };
    let mut days: Vec<PathBuf> = rd.filter_map(|e| e.ok()).map(|e| e.path()).filter(|p| p.is_dir()).collect();
    days.sort();
    let skip = days.len().saturating_sub(max_days);
    days.into_iter()
        .skip(skip)
        .filter_map(|d| std::fs::read_to_string(d.join(format!("{event_type}.jsonl"))).ok())
        .flat_map(|t| t.lines().filter_map(|l| serde_json::from_str::<Value>(l).ok()).collect::<Vec<_>>())
        .collect()
}

/// Latest successfully submitted order journaled for `symbol`.
fn last_order_for<'a>(orders: &'a [Value], symbol: &str) -> Option<&'a Value> {
    orders.iter().rev().find(|o| o["status"] == "submitted" && o["symbol"] == symbol)
}

fn order_by_link<'a>(orders: &'a [Value], link: &str) -> Option<&'a Value> {
    orders.iter().rev().find(|o| o["link_id"] == link)
}

// ---------------------------------------------------------------- position maths

/// Unrealised P&L in R, R = stop distance at entry (price units).
pub fn unrealised_r(side: Side, entry: f64, mark: f64, r_dist: f64) -> Option<f64> {
    (r_dist > 0.0 && entry > 0.0 && mark > 0.0).then(|| (mark - entry) * side.sign() / r_dist)
}

/// Breakeven rule: once unrealised >= 1R, move the stop to the entry price (rounded to the tick on the
/// protective side). Returns the new stop only if it tightens the current one and stays on the right side of mark.
pub fn breakeven_stop(side: Side, entry: f64, mark: f64, stop: f64, r_dist: f64, tick: f64) -> Option<f64> {
    let r = unrealised_r(side, entry, mark, r_dist)?;
    if r < 1.0 {
        return None;
    }
    let s = side.sign();
    let be = round_to_step(entry, tick, if side == Side::Long { Round::Up } else { Round::Down });
    let tightens = stop <= 0.0 || (be - stop) * s > 0.0;
    let below_mark = (mark - be) * s > 0.0;
    (tightens && below_mark).then_some(be)
}

fn side_of(venue_side: &str) -> Side {
    if venue_side == "Sell" { Side::Short } else { Side::Long }
}

fn num(v: &Value) -> f64 {
    v.as_str().and_then(|s| s.parse().ok()).or_else(|| v.as_f64()).unwrap_or(0.0)
}

/// Stop distance at entry: journal (planned entry/stop of the order that opened it), else the position's own stop.
fn r_distance(p: &VenuePosition, orders: &[Value]) -> (f64, &'static str) {
    if let Some(o) = last_order_for(orders, &p.symbol) {
        let d = (num(&o["entry"]) - num(&o["stop"])).abs();
        if d > 0.0 {
            return (d, "journal");
        }
    }
    if p.stop_loss > 0.0 {
        return ((p.avg_price - p.stop_loss).abs(), "position");
    }
    (0.0, "none")
}

/// Open entry orders (not reduce-only) for symbols without a position: they count as positions for the
/// risk engine so a re-run cannot double up on slots, symbols, clusters or gross exposure.
fn pending_entries<'a>(orders: &'a [Value], positions: &[VenuePosition]) -> Vec<&'a Value> {
    orders.iter().filter(|o| o["reduceOnly"] != true && !positions.iter().any(|p| o["symbol"] == p.symbol.as_str())).collect()
}

// ---------------------------------------------------------------- shared venue plumbing

pub(crate) struct Venue {
    rest: at_bybit::BybitRest,
    category: String,
    settle: String,
}

impl Venue {
    pub(crate) fn new(ctx: &Ctx) -> anyhow::Result<Self> {
        let rest = ctx.exec_rest()?;
        anyhow::ensure!(rest.has_credentials(), "no API credentials for {} in .env", ctx.s.environment.as_str());
        Ok(Self { rest, category: ctx.category().to_string(), settle: ctx.s.venues.bybit.settle_coin.clone() })
    }

    pub(crate) fn trading(&self) -> Trading<'_> {
        Trading::new(&self.rest, &self.category, &self.settle)
    }

    fn market(&self) -> MarketData<'_> {
        MarketData::new(&self.rest, &self.category)
    }

    pub(crate) async fn rules(&self, symbols: &HashSet<String>) -> anyhow::Result<HashMap<String, InstrumentRules>> {
        Ok(self
            .market()
            .instruments()
            .await?
            .into_iter()
            .filter(|i| symbols.contains(&i.symbol) && i.status == "Trading")
            .map(|i| {
                let r = InstrumentRules {
                    symbol: i.symbol.clone(),
                    tick_size: i.tick_size,
                    qty_step: i.qty_step,
                    min_qty: i.min_qty,
                    max_qty: i.max_qty,
                    max_mkt_qty: i.max_mkt_qty,
                    min_notional: i.min_notional,
                    max_leverage: i.max_leverage,
                };
                (i.symbol, r)
            })
            .collect())
    }

    async fn marks(&self) -> anyhow::Result<HashMap<String, f64>> {
        Ok(self.market().tickers().await?.into_iter().map(|t| (t.symbol, if t.mark_price > 0.0 { t.mark_price } else { t.last_price })).collect())
    }

    /// Write path: a broker on its own client (BybitBroker::new applies the mainnet two-key guard).
    pub(crate) fn broker(ctx: &Ctx, rules: HashMap<String, InstrumentRules>) -> anyhow::Result<BybitBroker> {
        BybitBroker::new(&ctx.s, ctx.exec_rest()?, rules)
    }
}

fn describe(r: &Rejection) -> String {
    let v = serde_json::to_value(r).unwrap_or_default();
    let mut s = v["reason"].as_str().unwrap_or("?").to_string();
    if let Value::Object(m) = &v {
        for (k, x) in m.iter().filter(|(k, _)| *k != "reason") {
            match x.as_f64() {
                Some(f) => s += &format!(" {k}={f:.4}"),
                None => s += &format!(" {k}={}", x.as_str().map(String::from).unwrap_or_else(|| x.to_string())),
            }
        }
    }
    s
}

fn print_decision(d: &RiskDecision, proposals: &[TradeProposal]) {
    if d.halted {
        println!("HALTED: daily loss limit reached, no new entries until the next UTC day");
    }
    println!("\naccepted ({})", d.accepted.len());
    if !d.accepted.is_empty() {
        println!("  {:<14} {:<12} {:<5} {:<6} {:>12} {:>12} {:>12} {:>10} {:>12} {:>9} {:>6} {:>7}", "id", "symbol", "side", "type", "entry", "stop", "tp", "qty", "notional", "risk", "R:R", "netE[R]");
    }
    for o in &d.accepted {
        let p = &o.proposal;
        println!(
            "  {:<14} {:<12} {:<5} {:<6} {:>12} {:>12} {:>12} {:>10} {:>12.2} {:>9.2} {:>6.2} {:>7.3}",
            p.id,
            p.symbol,
            format!("{:?}", p.side).to_lowercase(),
            format!("{:?}", p.entry_type).to_lowercase(),
            o.entry,
            o.stop,
            o.take_profit,
            o.qty,
            o.notional,
            o.risk_amount,
            o.reward_risk,
            o.net_expected_r
        );
    }
    println!("rejected ({})", d.rejected.len());
    for (id, r) in &d.rejected {
        let sym = proposals.iter().find(|p| &p.id == id).map(|p| p.symbol.as_str()).unwrap_or("?");
        println!("  {id:<14} {sym:<12} {}", describe(r));
    }
}

// ---------------------------------------------------------------- atx propose

pub async fn propose(ctx: &Ctx, file: &Path, submit: bool, session: Option<String>) -> anyhow::Result<()> {
    let pf = parse_proposal_file(&std::fs::read_to_string(file).with_context(|| file.display().to_string())?)?;
    let session = resolve_session(session, pf.session_id.clone())?;
    let env = ctx.s.environment;
    if submit {
        ensure_paper_venue(env, "--submit")?;
        ctx.s.ensure_execution_allowed()?;
    }
    let venue = Venue::new(ctx)?;
    let t = venue.trading();
    let journal = Journal::new(journal_root(&ctx.s), &session);
    let past_orders = read_events(&journal_root(&ctx.s), "order", JOURNAL_LOOKBACK_DAYS);

    let equity = t.wallet().await?.total_equity;
    let venue_positions = t.positions().await?;
    let open_orders = t.open_orders(None).await?;
    let state_path = day_state_path(&ctx.s);
    let mut day = DayState::roll(DayState::load(&state_path)?, &format_date(now_ms()), equity);
    day.save(&state_path)?;

    let cluster_of = |symbol: &str, link: Option<&str>| -> String {
        if let Some(c) = pf.clusters.get(symbol) {
            return c.clone();
        }
        let o = link.and_then(|l| order_by_link(&past_orders, l)).or_else(|| last_order_for(&past_orders, symbol));
        o.and_then(|o| o["cluster"].as_str()).unwrap_or_default().to_string()
    };
    let mut positions: Vec<Position> = venue_positions
        .iter()
        .map(|p| Position {
            symbol: p.symbol.clone(),
            side: side_of(&p.side),
            qty: p.size,
            entry_price: p.avg_price,
            stop: p.stop_loss,
            take_profit: p.take_profit,
            cluster: cluster_of(&p.symbol, None),
            opened_ts: p.created_ts,
        })
        .collect();
    let pending = pending_entries(&open_orders, &venue_positions);
    for o in &pending {
        let sym = o["symbol"].as_str().unwrap_or_default();
        positions.push(Position {
            symbol: sym.to_string(),
            side: side_of(o["side"].as_str().unwrap_or_default()),
            qty: num(&o["qty"]),
            entry_price: num(&o["price"]),
            stop: num(&o["stopLoss"]),
            take_profit: num(&o["takeProfit"]),
            cluster: cluster_of(sym, o["orderLinkId"].as_str()),
            opened_ts: num(&o["createdTime"]) as i64,
        });
    }

    let symbols: HashSet<String> = pf.proposals.iter().map(|p| p.symbol.clone()).collect();
    let rules = venue.rules(&symbols).await?;
    let marks = venue.marks().await?;
    let account = AccountState { equity, day_start_equity: day.day_start_equity, trades_opened_today: day.trades_opened_today, open_positions: positions };
    let engine = RiskEngine::new(ctx.risk.clone());
    let d = engine.evaluate(&pf.proposals, &account, &rules, &marks);

    println!(
        "{} session {session}  equity {equity:.2}  day start {:.2} ({:+.2}%)  trades today {}  open positions {} (+{} pending entries)",
        env.as_str(),
        day.day_start_equity,
        engine.drawdown_pct(&account),
        day.trades_opened_today,
        venue_positions.len(),
        pending.len()
    );
    print_decision(&d, &pf.proposals);

    let rejected: Vec<Value> = d.rejected.iter().map(|(id, r)| json!({"id": id, "rejection": r, "detail": describe(r)})).collect();
    journal.write(
        "risk_decision",
        json!({
            "source": "atx propose",
            "environment": env.as_str(),
            "submit": submit,
            "proposals_file": file.display().to_string(),
            "account": {"equity": equity, "day_start_equity": day.day_start_equity, "trades_opened_today": day.trades_opened_today, "open_positions": account.open_positions},
            "proposals": pf.proposals,
            "meta": pf.meta,
            "accepted": d.accepted,
            "rejected": rejected,
            "halted": d.halted,
            "risk_config": {"risk_per_trade_pct": ctx.risk.risk_per_trade_pct, "min_net_expected_r": ctx.risk.min_net_expected_r},
        }),
    )?;

    if !submit {
        println!("\ndry run: nothing sent (re-run with --submit to place the accepted orders on {})", env.as_str());
        return Ok(());
    }
    if d.accepted.is_empty() {
        println!("\nnothing to submit");
        return Ok(());
    }
    let mut broker = Venue::broker(ctx, rules)?;
    println!("\nsubmitting {} order(s) via {}", d.accepted.len(), broker.name());
    let mut failures = 0;
    for o in &d.accepted {
        let p = &o.proposal;
        let link = link_id(&session, &p.id);
        let base = json!({
            "broker": broker.name(),
            "link_id": link,
            "proposal_id": p.id,
            "symbol": p.symbol,
            "side": p.side,
            "entry_type": p.entry_type,
            "qty": o.qty,
            "entry": o.entry,
            "stop": o.stop,
            "take_profit": o.take_profit,
            "notional": o.notional,
            "risk_amount": o.risk_amount,
            "net_expected_r": o.net_expected_r,
            "p_tp": p.p_tp,
            "cluster": p.cluster,
            "channel": p.channel,
        });
        let mut rec = base.clone();
        match broker.submit(o, &link).await {
            Ok(venue_id) => {
                day.trades_opened_today += 1;
                day.save(&state_path)?;
                println!("  OK    {:<14} {:<12} link {link:<36} venue order {venue_id}", p.id, p.symbol);
                rec["status"] = json!("submitted");
                rec["venue_order_id"] = json!(venue_id);
            }
            Err(e) => {
                failures += 1;
                println!("  FAIL  {:<14} {:<12} link {link:<36} {e:#}", p.id, p.symbol);
                rec["status"] = json!("error");
                rec["error"] = json!(format!("{e:#}"));
            }
        }
        journal.write("order", rec)?;
    }
    println!("trades opened today: {}", day.trades_opened_today);
    anyhow::ensure!(failures == 0, "{failures} order(s) failed (journaled)");
    Ok(())
}

// ---------------------------------------------------------------- atx positions

pub async fn positions(ctx: &Ctx, manage: bool, breakeven: bool, session: Option<String>) -> anyhow::Result<()> {
    let venue = Venue::new(ctx)?;
    let t = venue.trading();
    let orders_journal = read_events(&journal_root(&ctx.s), "order", JOURNAL_LOOKBACK_DAYS);
    let ps = t.positions().await?;
    let open = t.open_orders(None).await?;

    println!("{} positions ({})", ctx.s.environment.as_str(), ps.len());
    if !ps.is_empty() {
        println!("  {:<12} {:<5} {:>10} {:>12} {:>12} {:>12} {:>12} {:>10} {:>7}  R from", "symbol", "side", "qty", "entry", "mark", "stop", "tp", "uPnL", "R");
    }
    let mut rows = Vec::new();
    for p in &ps {
        let side = side_of(&p.side);
        let (r_dist, src) = r_distance(p, &orders_journal);
        let r = unrealised_r(side, p.avg_price, p.mark_price, r_dist);
        let r_txt = r.map(|r| format!("{r:+.2}")).unwrap_or_else(|| "n/a".into());
        println!(
            "  {:<12} {:<5} {:>10} {:>12} {:>12} {:>12} {:>12} {:>10.2} {:>7}  {src}",
            p.symbol,
            format!("{side:?}").to_lowercase(),
            p.size,
            p.avg_price,
            p.mark_price,
            p.stop_loss,
            p.take_profit,
            p.unrealised_pnl,
            r_txt
        );
        rows.push((p, side, r_dist, src, r));
    }
    println!("open orders ({})", open.len());
    for o in &open {
        println!(
            "  {:<12} {:<4} {:<6} price {:<12} qty {:<10} SL {:<12} TP {:<12} {:<12} reduceOnly {:<5} link {}  id {}",
            o["symbol"].as_str().unwrap_or_default(),
            o["side"].as_str().unwrap_or_default(),
            o["orderType"].as_str().unwrap_or_default(),
            o["price"].as_str().unwrap_or_default(),
            o["qty"].as_str().unwrap_or_default(),
            o["stopLoss"].as_str().unwrap_or_default(),
            o["takeProfit"].as_str().unwrap_or_default(),
            o["orderStatus"].as_str().unwrap_or_default(),
            o["reduceOnly"],
            o["orderLinkId"].as_str().unwrap_or_default(),
            o["orderId"].as_str().unwrap_or_default(),
        );
    }

    if !manage {
        return Ok(());
    }
    // Exits are price/state driven only (ARCHITECTURE §3.12): nothing here closes on time.
    if !breakeven {
        println!("\n--manage: no rule enabled (structural trail not implemented; enable --breakeven to move stops to entry at >= 1R)");
        return Ok(());
    }
    let symbols: HashSet<String> = ps.iter().map(|p| p.symbol.clone()).collect();
    let rules = venue.rules(&symbols).await?;
    let mut changes = Vec::new();
    for (p, side, r_dist, src, r) in &rows {
        let Some(rule) = rules.get(&p.symbol) else { continue };
        if let Some(new_stop) = breakeven_stop(*side, p.avg_price, p.mark_price, p.stop_loss, *r_dist, rule.tick_size) {
            changes.push((*p, *side, *r_dist, *src, r.unwrap_or_default(), new_stop));
        }
    }
    println!("\n--manage --breakeven: {} stop move(s)", changes.len());
    if changes.is_empty() {
        return Ok(());
    }
    let mut broker = Venue::broker(ctx, rules)?;
    let journal = Journal::new(journal_root(&ctx.s), &session.unwrap_or_else(|| "atx-manage".into()));
    for (p, side, r_dist, src, r, new_stop) in changes {
        let res = broker.move_stop(&p.symbol, new_stop).await;
        let ok = res.is_ok();
        match &res {
            Ok(()) => println!("  {:<12} stop {} -> {new_stop} (uR {r:+.2})", p.symbol, p.stop_loss),
            Err(e) => println!("  {:<12} FAILED to move stop to {new_stop}: {e:#}", p.symbol),
        }
        journal.write(
            "position_update",
            json!({
                "broker": broker.name(),
                "action": "move_stop",
                "rule": "breakeven_at_1r",
                "symbol": p.symbol,
                "side": side,
                "qty": p.size,
                "entry": p.avg_price,
                "mark": p.mark_price,
                "old_stop": p.stop_loss,
                "new_stop": new_stop,
                "take_profit": p.take_profit,
                "unrealised_r": r,
                "r_distance": r_dist,
                "r_source": src,
                "status": if ok { "applied" } else { "error" },
                "error": res.err().map(|e| format!("{e:#}")),
            }),
        )?;
    }
    Ok(())
}

// ---------------------------------------------------------------- atx flatten

pub async fn flatten(ctx: &Ctx, symbol: &str, yes: bool, session: Option<String>) -> anyhow::Result<()> {
    // Allowed on mainnet (it only ever reduces risk); the two-key guard still applies.
    ctx.s.ensure_execution_allowed()?;
    let symbol = symbol.trim().to_uppercase();
    anyhow::ensure!(yes, "flatten cancels every open order and closes the {symbol} position on {}; re-run with --yes", ctx.s.environment.as_str());
    let venue = Venue::new(ctx)?;
    let t = venue.trading();
    let orders = t.open_orders(Some(&symbol)).await?;
    let pos = t.positions().await?.into_iter().find(|p| p.symbol == symbol);
    let cancelled: Vec<Value> = orders.iter().map(|o| json!({"order_id": o["orderId"], "link_id": o["orderLinkId"], "side": o["side"], "price": o["price"], "qty": o["qty"]})).collect();

    if !orders.is_empty() {
        t.cancel_all(&symbol).await?;
    }
    println!("{symbol}: cancelled {} open order(s)", orders.len());
    for c in &cancelled {
        println!("  {} {} {} @ {}  link {}", c["order_id"].as_str().unwrap_or_default(), c["side"].as_str().unwrap_or_default(), c["qty"].as_str().unwrap_or_default(), c["price"].as_str().unwrap_or_default(), c["link_id"].as_str().unwrap_or_default());
    }
    let mut still_open = false;
    if let Some(p) = &pos {
        let rules = venue.rules(&HashSet::from([symbol.clone()])).await?;
        let mut broker = Venue::broker(ctx, rules)?;
        broker.close(&symbol).await?;
        tokio::time::sleep(Duration::from_secs(2)).await;
        still_open = t.positions().await?.iter().any(|q| q.symbol == symbol);
        println!("{symbol}: closed {} {} @ mark {} (uPnL {:.2}): {}", p.side, p.size, p.mark_price, p.unrealised_pnl, if still_open { "STILL OPEN, check the account" } else { "flat" });
    } else {
        println!("{symbol}: no open position");
    }
    let remaining = t.open_orders(Some(&symbol)).await?.len();
    Journal::new(journal_root(&ctx.s), &session.unwrap_or_else(|| "atx-manual".into())).write(
        "manual_action",
        json!({
            "action": "flatten",
            "environment": ctx.s.environment.as_str(),
            "symbol": symbol,
            "cancelled_orders": cancelled,
            "closed_position": pos.as_ref().map(|p| json!({"side": p.side, "qty": p.size, "entry": p.avg_price, "mark": p.mark_price, "stop": p.stop_loss, "take_profit": p.take_profit, "unrealised_pnl": p.unrealised_pnl})),
            "position_still_open": still_open,
            "orders_remaining": remaining,
        }),
    )?;
    anyhow::ensure!(!still_open && remaining == 0, "{symbol} not fully flat: position open = {still_open}, orders remaining = {remaining}");
    Ok(())
}

// ---------------------------------------------------------------- tests

#[cfg(test)]
mod tests {
    use super::*;

    fn tmp(name: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("atx-exec-test-{name}-{}-{}", std::process::id(), now_ms()));
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    const FILE: &str = r#"{
        "session_id": "s20261107",
        "clusters": {"ethusdt": "L1"},
        "proposals": [
            {"id": "p1", "symbol": "btcusdt", "side": "long", "entry_type": "limit", "entry": 85000, "stop": 83300,
             "take_profit": 88400, "p_tp": 0.45, "cluster": "L1", "channel": "C-MAIN"},
            {"id": "p2", "symbol": "SOLUSDT", "side": "short", "entry_type": "market", "entry": 150, "stop": 155,
             "take_profit": 140, "p_tp": 0.4, "cluster": "ALT", "channel": "C-MAIN"}
        ]
    }"#;

    #[test]
    fn parses_proposal_file() {
        let f = parse_proposal_file(FILE).unwrap();
        assert_eq!(f.session_id.as_deref(), Some("s20261107"));
        assert_eq!(f.proposals.len(), 2);
        assert_eq!(f.proposals[0].symbol, "BTCUSDT", "symbols are upper-cased");
        assert_eq!(f.proposals[1].side, Side::Short);
        assert_eq!(f.clusters.get("ETHUSDT").map(String::as_str), Some("L1"));
        // clusters and session_id are optional
        let f = parse_proposal_file(r#"{"proposals": [{"id": "a", "symbol": "BTCUSDT", "side": "long", "entry_type": "market", "entry": 1, "stop": 0.9, "take_profit": 1.2, "p_tp": 0.5, "cluster": "", "channel": "x"}]}"#).unwrap();
        assert!(f.session_id.is_none() && f.clusters.is_empty());
    }

    #[test]
    fn rejects_bad_proposal_files() {
        assert!(parse_proposal_file(r#"{"session_id": "s", "proposals": []}"#).is_err(), "empty");
        let dup = FILE.replace("\"p2\"", "\"p1\"");
        assert!(parse_proposal_file(&dup).is_err(), "duplicate ids");
        assert!(parse_proposal_file(&FILE.replace("\"short\"", "\"sideways\"")).is_err(), "bad side");
        assert!(parse_proposal_file(&FILE.replace("\"p_tp\": 0.4,", "")).is_err(), "missing field");
        assert!(parse_proposal_file(&FILE.replace("\"clusters\"", "\"qty\": 1, \"clusters\"")).is_err(), "unknown top-level field (size is never part of a proposal)");
    }

    #[test]
    fn session_resolution() {
        assert_eq!(resolve_session(Some("cli".into()), Some("file".into())).unwrap(), "cli");
        assert_eq!(resolve_session(None, Some("file".into())).unwrap(), "file");
        assert!(resolve_session(None, None).is_err());
        assert!(resolve_session(Some("  ".into()), None).is_err());
    }

    #[test]
    fn link_ids_fit_bybit() {
        assert_eq!(link_id("s1", "p1"), "s1-p1");
        assert_eq!(link_id("sess 1", "a/b"), "sess_1-a_b", "only [A-Za-z0-9_-]");
        let exact = format!("{}-{}", "s".repeat(17), "p".repeat(18));
        assert_eq!(link_id(&"s".repeat(17), &"p".repeat(18)), exact, "36 chars pass through");
        let a = link_id("session-2026-11-07T12-00-00", "BTCUSDT-long-breakout-1");
        let b = link_id("session-2026-11-07T12-00-00", "BTCUSDT-long-breakout-2");
        assert_eq!(a.len(), MAX_LINK_ID);
        assert_eq!(b.len(), MAX_LINK_ID);
        assert_ne!(a, b, "truncation keeps long ids distinct");
        assert!(a.starts_with("session-2026-11-07T12-00-00"));
        assert_eq!(a, link_id("session-2026-11-07T12-00-00", "BTCUSDT-long-breakout-1"), "deterministic");
    }

    #[test]
    fn submit_refuses_mainnet() {
        assert!(ensure_paper_venue(Environment::Demo, "--submit").is_ok());
        assert!(ensure_paper_venue(Environment::Testnet, "--submit").is_ok());
        let e = ensure_paper_venue(Environment::Mainnet, "--submit").unwrap_err().to_string();
        assert!(e.contains("mainnet"));
    }

    #[test]
    fn day_state_resets_at_utc_day_change() {
        let fresh = DayState::roll(None, "2026-11-07", 1000.0);
        assert_eq!(fresh, DayState { date: "2026-11-07".into(), day_start_equity: 1000.0, trades_opened_today: 0 });
        let mut same = fresh.clone();
        same.trades_opened_today = 2;
        let kept = DayState::roll(Some(same.clone()), "2026-11-07", 950.0);
        assert_eq!(kept, same, "same day keeps the day-start equity and the trade count");
        let next = DayState::roll(Some(same), "2026-11-08", 950.0);
        assert_eq!(next, DayState { date: "2026-11-08".into(), day_start_equity: 950.0, trades_opened_today: 0 });

        let dir = tmp("day");
        let path = dir.join("state").join("exec_day.json");
        assert_eq!(DayState::load(&path).unwrap(), None);
        next.save(&path).unwrap();
        assert_eq!(DayState::load(&path).unwrap(), Some(next));
        std::fs::write(&path, "{not json").unwrap();
        assert!(DayState::load(&path).is_err(), "corrupt state is an error, not a silent reset");
        std::fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn journal_envelope_matches_python() {
        let dir = tmp("journal");
        let j = Journal::new(dir.clone(), "s1");
        let ts = DateTime::parse_from_rfc3339("2026-11-07T10:11:12.345678Z").unwrap().with_timezone(&Utc);
        j.write_at("order", json!({"symbol": "BTCUSDT", "status": "submitted", "entry": 85000.0, "stop": 83300.0}), ts).unwrap();
        j.write_at("order", json!({"symbol": "BTCUSDT", "status": "error", "entry": 1.0, "stop": 0.5}), ts).unwrap();
        assert!(j.write_at("not_a_type", json!({}), ts).is_err());
        let text = std::fs::read_to_string(dir.join("2026-11-07").join("order.jsonl")).unwrap();
        let first: Value = serde_json::from_str(text.lines().next().unwrap()).unwrap();
        assert_eq!(first["ts"], "2026-11-07T10:11:12.345678+00:00");
        assert_eq!(first["session_id"], "s1");
        assert_eq!(first["type"], "order");
        let evs = read_events(&dir, "order", 10);
        assert_eq!(evs.len(), 2);
        let o = last_order_for(&evs, "BTCUSDT").unwrap();
        assert_eq!(o["stop"], 83300.0, "errored orders are ignored for the R distance");
        std::fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn breakeven_only_at_one_r_and_only_tightens() {
        // long 100, stop 95 (R = 5)
        assert_eq!(unrealised_r(Side::Long, 100.0, 105.0, 5.0), Some(1.0));
        assert_eq!(unrealised_r(Side::Short, 100.0, 105.0, 5.0), Some(-1.0));
        assert_eq!(unrealised_r(Side::Long, 100.0, 105.0, 0.0), None);
        assert_eq!(breakeven_stop(Side::Long, 100.0, 104.9, 95.0, 5.0, 0.1), None, "below 1R");
        assert_eq!(breakeven_stop(Side::Long, 100.04, 106.0, 95.0, 5.0, 0.1), Some(100.1), "long rounds up");
        assert_eq!(breakeven_stop(Side::Long, 100.0, 106.0, 100.0, 5.0, 0.1), None, "already at breakeven");
        assert_eq!(breakeven_stop(Side::Short, 100.06, 94.0, 105.0, 5.0, 0.1), Some(100.0), "short rounds down");
        assert_eq!(breakeven_stop(Side::Short, 100.0, 94.0, 99.0, 5.0, 0.1), None, "stop already better than entry");
    }
}
