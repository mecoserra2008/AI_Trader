//! `atx live`: the small live desk on Bybit USDT linear perpetuals (config/risk.yaml `live_small`, user authorisation
//! 2026-11-15 ~16:10 UTC). Every venue write passes `at_risk::live::decide_live` first; the limits live there.
//!
//!   atx live status                 balance, bot positions with their exchange stops, today's entries, P&L, limits left
//!   atx live tiers [--top N]        notional tier of every Trading USDT linear perp (mainnet turnover rank, spread)
//!   atx live propose --file I.json  risk-checked; writes data/live/pending/<id>.json while the approval gate is on, else submits
//!   atx live approve <id>           THE USER ONLY: interactive terminal + confirmation code, then submits
//!   atx live submit --id <id>       internal: submits an approved (or auto-gated) proposal
//!   atx live reconcile              venue -> bot state: fills, closes and P&L, missing stops repaired, stale bot entries cancelled
//!   atx live close --id <trade>     early exit of one bot trade (reduce-only market close)
//!   atx live kill                   data/live/KILL + cancel every bot order (positions keep their exchange stops)
//!   atx live flatten-bot --yes      cancel bot orders and close BOT positions with reduce-only market orders
//!
//! Safety:
//! - Mainnet writes need the two-key guard (execution.allow_mainnet AND AI_TRADER_ALLOW_MAINNET=1). Without it, status,
//!   tiers and reconcile run read-only (the venue client refuses every write), and everything else is refused.
//! - The bot only ever touches orders whose orderLinkId starts with `<link_prefix>-` and positions it opened itself (state
//!   file). Symbols with a foreign position or order are never traded. SPYUSDT is never touched.
//! - Every entry carries an exchange-side stop (stopLoss, tpslMode Full) from the first order, verified after the fill.
//! - Entries: PostOnly limit at the touch, re-pegged until the window ends, then cancelled (R-EXE-1).
//!
//! Files (mainnet root data/live, other environments data/live/<env>): state.json, pending/<id>.json,
//! approvals/<id>.ok (the user's file approval: the confirmation code), done/<id>.json, KILL (data/live/KILL stops every
//! environment), corr_cache.json (shared). Journal: data/journal/live (mainnet) or data/journal/live_<env>.

use std::collections::{HashMap, HashSet};
use std::io::{BufRead, IsTerminal, Write as _};
use std::path::{Path, PathBuf};
use std::time::Duration;

use anyhow::{Context, bail, ensure};
use at_bybit::BybitRest;
use at_bybit::market::{Book, Instrument, MarketData, PriceType};
use at_bybit::trading::{OrderRequest, Trading, VenuePosition};
use at_core::config::Environment;
use at_core::num::{Round, format_step, round_to_step};
use at_core::time::{DAY, Freq, HOUR, now_ms};
use at_core::types::{InstrumentRules, Side};
use at_risk::fees::{FeeRates, FeeSchedule};
use at_risk::live::{self as lv, LiveAccount, LiveDecision, LivePos, LiveSmallCfg, SymbolFacts};
use at_risk::{DecideConfig, OrderIntent};
use chrono::Utc;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use crate::cmd::Ctx;
use crate::exec::Journal;

/// Execution tool version: changing the live entry / exit mechanics starts a new live track record.
pub const LIVE_EXEC_VERSION: &str = "live-pegged-v1";
const BOOK_LEVELS: u32 = 200;
const LOCK_STALE_MS: i64 = 120_000;

// ---------------------------------------------------------------- CLI

#[derive(clap::Subcommand)]
pub enum LiveCmd {
    /// Balance, bot positions with their exchange stops, foreign positions (never touched), entries today, P&L, limits left.
    Status,
    /// Notional tier of every Trading USDT linear perpetual (mainnet 24h turnover rank and quoted spread).
    Tiers {
        #[arg(long, default_value_t = 40)]
        top: usize,
    },
    /// Risk-check an OrderIntent (mode "live", entry {type: limit, peg: touch}); pending approval while the gate is on.
    Propose {
        #[arg(long)]
        file: PathBuf,
        #[command(flatten)]
        test: TestOpts,
    },
    /// For the USER only: approve a pending proposal (interactive terminal and its confirmation code), then submit it.
    Approve { id: String },
    /// Internal: submit a pending proposal that the user approved (approvals/<id>.ok) or that the auto gate allows.
    Submit {
        #[arg(long)]
        id: String,
        #[command(flatten)]
        test: TestOpts,
    },
    /// Venue -> bot state (fills, closes, P&L), repair missing stops, cancel stale bot entries. Read-only on mainnet without the guard.
    Reconcile {
        #[arg(long)]
        read_only: bool,
    },
    /// Early exit of ONE bot trade: cancel its resting entry and close its position with a reduce-only market order.
    Close {
        /// Trade id (= proposal id, see `atx live status`).
        #[arg(long)]
        id: String,
        #[arg(long, default_value = "")]
        reason: String,
    },
    /// Write the KILL flag and cancel every bot order. Bot positions keep their exchange-side stops.
    Kill,
    /// Cancel bot orders and close bot positions with reduce-only market orders. Never touches anything else.
    FlattenBot {
        #[arg(long)]
        yes: bool,
    },
}

/// Test hooks, refused on mainnet.
#[derive(clap::Args, Clone, Default, Debug)]
pub struct TestOpts {
    /// DEMO / TESTNET ONLY: peg this many bp behind the touch (exercises re-pegging and the window cancel without a fill).
    #[arg(long)]
    pub test_behind_bps: Option<f64>,
    /// DEMO / TESTNET ONLY: re-peg window in seconds instead of entry.window_min.
    #[arg(long)]
    pub test_window_secs: Option<u64>,
}

impl TestOpts {
    fn any(&self) -> bool {
        self.test_behind_bps.is_some() || self.test_window_secs.is_some()
    }
}

pub async fn run(ctx: &Ctx, cmd: LiveCmd) -> anyhow::Result<()> {
    let desk = Desk::new(ctx)?;
    match cmd {
        LiveCmd::Status => desk.status().await,
        LiveCmd::Tiers { top } => desk.tiers(top).await,
        LiveCmd::Propose { file, test } => desk.propose(&file, &test).await,
        LiveCmd::Approve { id } => desk.approve(&id).await,
        LiveCmd::Submit { id, test } => desk.submit(&id, &test).await,
        LiveCmd::Reconcile { read_only } => desk.reconcile_cmd(read_only).await,
        LiveCmd::Close { id, reason } => desk.close(&id, &reason).await,
        LiveCmd::Kill => desk.kill().await,
        LiveCmd::FlattenBot { yes } => desk.flatten_bot(yes).await,
    }
}

// ---------------------------------------------------------------- paths and state

pub struct LivePaths {
    pub root: PathBuf,
    pub global_kill: PathBuf,
    pub journal: PathBuf,
    pub corr_cache: PathBuf,
}

impl LivePaths {
    pub fn new(data_root: &Path, journal_base: &Path, env: Environment) -> Self {
        let base = data_root.join("live");
        let root = if env == Environment::Mainnet { base.clone() } else { base.join(env.as_str()) };
        let journal = if env == Environment::Mainnet { journal_base.join("live") } else { journal_base.join(format!("live_{}", env.as_str())) };
        Self { global_kill: base.join("KILL"), corr_cache: base.join("corr_cache.json"), root, journal }
    }
    fn state(&self) -> PathBuf {
        self.root.join("state.json")
    }
    fn pending(&self, id: &str) -> PathBuf {
        self.root.join("pending").join(format!("{id}.json"))
    }
    fn approval(&self, id: &str) -> PathBuf {
        self.root.join("approvals").join(format!("{id}.ok"))
    }
    fn done(&self, id: &str) -> PathBuf {
        self.root.join("done").join(format!("{id}.json"))
    }
    fn local_kill(&self) -> PathBuf {
        self.root.join("KILL")
    }
    pub fn kill_active(&self) -> bool {
        self.global_kill.exists() || self.local_kill().exists()
    }
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct TradeRec {
    pub id: String,
    pub intent_id: String,
    pub symbol: String,
    pub side: Option<Side>,
    pub tier: String,
    /// submitting | pending | open | closed | cancelled | error
    pub status: String,
    pub qty: f64,
    pub filled_qty: f64,
    pub entry_ref: f64,
    pub mid_at_submit: f64,
    pub first_peg_price: f64,
    pub avg_entry: f64,
    pub stop: f64,
    pub target: Option<f64>,
    pub risk_to_stop_usd: f64,
    pub max_loss_usd: f64,
    pub stop_slip_bps: f64,
    /// Most notional a chasing peg may reach: min(tier / depth cap, decided notional + gross headroom).
    #[serde(default)]
    pub notional_usd: f64,
    pub order_ids: Vec<String>,
    pub link_ids: Vec<String>,
    pub repegs: u32,
    pub submitted_ms: i64,
    pub first_fill_ms: Option<i64>,
    pub entry_done_ms: Option<i64>,
    pub entry_end_reason: Option<String>,
    pub lease_until_ms: i64,
    pub closed_ms: Option<i64>,
    pub exit_avg: Option<f64>,
    pub exit_kind: Option<String>,
    pub realised_usd: Option<f64>,
    pub approved_by: String,
    pub versions: Value,
    pub notes: Vec<String>,
}

impl TradeRec {
    fn active(&self) -> bool {
        matches!(self.status.as_str(), "submitting" | "pending" | "open")
    }
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct LiveState {
    pub env: String,
    /// UTC date of `entries_today`.
    pub day: String,
    pub entries_today: u32,
    /// Approved (non-auto) proposals whose entry filled: the approval gate is on while this is < approval.manual_trades.
    pub confirmed_trades: u32,
    pub trades: Vec<TradeRec>,
    /// Positions and orders found at the first reconcile (never touched).
    #[serde(default)]
    pub foreign_at_start: Option<Value>,
    #[serde(default)]
    pub created_ms: i64,
}

impl LiveState {
    pub fn load(path: &Path) -> anyhow::Result<Self> {
        match std::fs::read_to_string(path) {
            Ok(t) => serde_json::from_str(&t).with_context(|| format!("{}: corrupt live state (fix it; never delete it blindly)", path.display())),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(Self::default()),
            Err(e) => Err(e).with_context(|| path.display().to_string()),
        }
    }

    pub fn save(&self, path: &Path) -> anyhow::Result<()> {
        write_json(path, &serde_json::to_value(self)?)
    }

    pub fn roll_day(&mut self, now: i64) {
        let today = day_of(now);
        if self.day != today {
            self.day = today;
            self.entries_today = 0;
        }
    }

    pub fn realised_today(&self, now: i64) -> f64 {
        let start = now - now.rem_euclid(DAY);
        self.trades.iter().filter(|t| t.closed_ms.is_some_and(|c| c >= start)).filter_map(|t| t.realised_usd).sum()
    }

    pub fn gate_manual(&self, cfg: &LiveSmallCfg) -> bool {
        self.confirmed_trades < cfg.approval.manual_trades
    }
}

fn day_of(ms: i64) -> String {
    at_core::time::format_date(ms)
}

fn write_json(path: &Path, v: &Value) -> anyhow::Result<()> {
    if let Some(d) = path.parent() {
        std::fs::create_dir_all(d)?;
    }
    let tmp = path.with_extension("tmp");
    std::fs::write(&tmp, serde_json::to_string_pretty(v)?)?;
    std::fs::rename(&tmp, path).with_context(|| path.display().to_string())
}

/// Exclusive lock on the live state for short read-decide-write sections.
struct StateLock(PathBuf);

impl StateLock {
    async fn acquire(root: &Path) -> anyhow::Result<Self> {
        std::fs::create_dir_all(root)?;
        let p = root.join("LOCK");
        for _ in 0..300 {
            match std::fs::OpenOptions::new().write(true).create_new(true).open(&p) {
                Ok(mut f) => {
                    let _ = write!(f, "{} {}", std::process::id(), now_ms());
                    return Ok(Self(p));
                }
                Err(_) => {
                    let stale = std::fs::read_to_string(&p)
                        .ok()
                        .and_then(|t| t.split_whitespace().nth(1).and_then(|x| x.parse::<i64>().ok()))
                        .is_some_and(|t| now_ms() - t > LOCK_STALE_MS);
                    if stale {
                        let _ = std::fs::remove_file(&p);
                    }
                    tokio::time::sleep(Duration::from_millis(100)).await;
                }
            }
        }
        bail!("{} is held by another atx live process", p.display())
    }
}

impl Drop for StateLock {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.0);
    }
}

// ---------------------------------------------------------------- proposals

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Proposal {
    pub id: String,
    pub env: String,
    pub created_ms: i64,
    pub expires_ms: i64,
    pub confirm_code: String,
    pub intent: OrderIntent,
    /// The decision at proposal time (sized at the then touch; re-checked at submission).
    pub decision: LiveDecision,
    pub needs_approval: bool,
    pub versions: Value,
}

fn fnv(s: &str) -> u64 {
    s.bytes().fold(0xcbf2_9ce4_8422_2325_u64, |h, b| (h ^ b as u64).wrapping_mul(0x0100_0000_01b3))
}

pub fn proposal_id(now: i64, intent_id: &str) -> String {
    let t = chrono::DateTime::from_timestamp_millis(now).unwrap_or_default();
    format!("{}{:04x}", t.format("%m%d%H%M%S"), fnv(&format!("{intent_id}{now}")) & 0xffff)
}

pub fn confirm_code(id: &str, intent: &OrderIntent) -> String {
    format!("{:06X}", fnv(&format!("{id}|{}", serde_json::to_string(intent).unwrap_or_default())) & 0xff_ffff)
}

/// Bybit orderLinkId of the n-th entry order of a proposal (<= 36 chars, [A-Za-z0-9-]).
pub fn entry_link(cfg: &LiveSmallCfg, id: &str, n: u32) -> String {
    format!("{}{id}-{n}", cfg.link_tag()).chars().take(36).collect()
}

fn close_link(cfg: &LiveSmallCfg, id: &str) -> String {
    format!("{}x-{id}-{}", cfg.link_tag(), now_ms() % 100_000).chars().take(36).collect()
}

// ---------------------------------------------------------------- venue

struct Venue {
    exec: BybitRest,
    mkt: BybitRest,
    category: String,
    settle: String,
    writable: bool,
}

impl Venue {
    fn trading(&self) -> Trading<'_> {
        Trading::new(&self.exec, &self.category, &self.settle)
    }
    fn exec_market(&self) -> MarketData<'_> {
        MarketData::new(&self.exec, &self.category)
    }
    fn main_market(&self) -> MarketData<'_> {
        MarketData::new(&self.mkt, &self.category)
    }
    fn w(&self) -> anyhow::Result<()> {
        ensure!(self.writable, "read-only: venue writes are blocked (mainnet needs execution.allow_mainnet=true AND AI_TRADER_ALLOW_MAINNET=1)");
        Ok(())
    }
    async fn place(&self, req: &OrderRequest) -> anyhow::Result<String> {
        self.w()?;
        Ok(self.trading().create_order(req).await?.order_id)
    }
    async fn amend(&self, symbol: &str, order_id: &str, price: &str, qty: Option<&str>) -> anyhow::Result<()> {
        self.w()?;
        self.trading().amend_order(symbol, order_id, Some(price), qty).await
    }
    async fn cancel(&self, symbol: &str, order_id: &str) -> anyhow::Result<()> {
        self.w()?;
        self.trading().cancel_order(symbol, order_id).await
    }
    async fn set_stop(&self, symbol: &str, stop: &str, trigger: &str) -> anyhow::Result<()> {
        self.w()?;
        self.trading().set_trading_stop(symbol, Some(stop), None, trigger).await
    }
    async fn set_leverage(&self, symbol: &str, lev: &str) -> anyhow::Result<()> {
        self.w()?;
        self.trading().set_leverage(symbol, lev).await
    }
}

struct Snapshot {
    equity: f64,
    available: f64,
    coin: Option<Value>,
    positions: Vec<VenuePosition>,
    orders: Vec<Value>,
}

impl Snapshot {
    fn bot_order(o: &Value, tag: &str) -> bool {
        o["orderLinkId"].as_str().is_some_and(|l| l.starts_with(tag))
    }

    /// The conditional orders Bybit creates for a position-level stop / target (no link id).
    fn position_tpsl(o: &Value) -> bool {
        matches!(o["stopOrderType"].as_str(), Some("StopLoss" | "TakeProfit" | "TrailingStop" | "PartialStopLoss" | "PartialTakeProfit"))
            && o["orderLinkId"].as_str().unwrap_or("").is_empty()
    }

    /// An order the bot did not place: neither bot-tagged nor the TP/SL of a bot position.
    fn foreign_order(o: &Value, tag: &str, bot_syms: &HashSet<&str>) -> bool {
        let bot_tpsl = Self::position_tpsl(o) && o["symbol"].as_str().is_some_and(|s| bot_syms.contains(s));
        !(Self::bot_order(o, tag) || bot_tpsl)
    }
}

// ---------------------------------------------------------------- the desk

pub struct Desk<'a> {
    ctx: &'a Ctx,
    cfg: LiveSmallCfg,
    paths: LivePaths,
    env: Environment,
    venue: Venue,
    journal: Journal,
    fees: FeeSchedule,
}

fn side_of(s: &str) -> Side {
    if s == "Sell" { Side::Short } else { Side::Long }
}

fn num(v: &Value) -> f64 {
    v.as_str().and_then(|s| s.parse().ok()).or_else(|| v.as_f64()).unwrap_or(0.0)
}

fn rules_of(i: &Instrument) -> InstrumentRules {
    InstrumentRules {
        symbol: i.symbol.clone(),
        tick_size: i.tick_size,
        qty_step: i.qty_step,
        min_qty: i.min_qty,
        max_qty: i.max_qty,
        max_mkt_qty: i.max_mkt_qty,
        min_notional: if i.min_notional.is_finite() { i.min_notional } else { 5.0 },
        max_leverage: i.max_leverage,
    }
}

fn usdt_perp(i: &Instrument) -> bool {
    i.status == "Trading" && i.contract_type == "LinearPerpetual" && i.quote == "USDT" && i.settle == "USDT"
}

impl<'a> Desk<'a> {
    pub fn new(ctx: &'a Ctx) -> anyhow::Result<Self> {
        let cfg = LiveSmallCfg::load(&ctx.s.root)?;
        let env = ctx.s.environment;
        let paths = LivePaths::new(&ctx.s.abs(&ctx.s.paths.data_root), &ctx.s.abs(&ctx.s.paths.journal), env);
        let exec = ctx.exec_rest()?;
        ensure!(exec.has_credentials(), "no API credentials for {} in .env", env.as_str());
        let writable = ctx.s.ensure_execution_allowed().is_ok();
        let venue = Venue { exec, mkt: ctx.market_rest()?, category: ctx.category().to_string(), settle: ctx.s.venues.bybit.settle_coin.clone(), writable };
        let journal = Journal::new(paths.journal.clone(), &format!("live-{}", env.as_str()));
        let fees = FeeSchedule::load(&ctx.s.root.join("config").join("fees.yaml"))?;
        Ok(Self { ctx, cfg, paths, env, venue, journal, fees })
    }

    fn versions(&self, intent: Option<&OrderIntent>) -> Value {
        let c = intent.and_then(|i| i.context.as_ref());
        json!({
            "env": self.env.as_str(),
            "atx_version": env!("CARGO_PKG_VERSION"),
            "exec_version": LIVE_EXEC_VERSION,
            "live_profile_hash": self.cfg.profile_hash(),
            "model_id": c.and_then(|c| c.get("model_id")).cloned().unwrap_or(Value::Null),
            "prompt_version": c.and_then(|c| c.get("prompt_version")).cloned().unwrap_or(Value::Null),
            "packet_version": c.and_then(|c| c.get("packet_version")).cloned().unwrap_or(Value::Null),
            "session_id": c.and_then(|c| c.get("session_id")).cloned().unwrap_or(Value::Null),
        })
    }

    fn jw(&self, kind: &str, mut payload: Value, versions: &Value) -> anyhow::Result<()> {
        payload["versions"] = versions.clone();
        payload["environment"] = json!(self.env.as_str());
        self.journal.write(kind, payload)
    }

    /// Writes need the two-key guard on mainnet; test hooks are refused there.
    fn ensure_writes(&self, test: &TestOpts) -> anyhow::Result<()> {
        self.ctx.s.ensure_execution_allowed()?;
        ensure!(self.venue.writable, "venue client is read-only");
        if test.any() {
            ensure!(self.env != Environment::Mainnet, "test hooks (--test-*) are refused on mainnet");
        }
        Ok(())
    }

    async fn snapshot(&self) -> anyhow::Result<Snapshot> {
        let t = self.venue.trading();
        let w = t.wallet().await?;
        let coin = t.settle_coin_row().await.ok().flatten();
        let positions = t.positions().await?;
        let orders = t.open_orders(None).await?;
        Ok(Snapshot { equity: w.total_equity, available: w.total_available, coin, positions, orders })
    }

    fn bot_position<'s>(&self, snap: &'s Snapshot, t: &TradeRec) -> Option<&'s VenuePosition> {
        snap.positions.iter().find(|p| p.symbol == t.symbol && Some(side_of(&p.side)) == t.side)
    }

    /// Venue positions / orders the bot did not open.
    fn foreign(&self, snap: &Snapshot, st: &LiveState) -> (Vec<Value>, Vec<Value>) {
        let tag = self.cfg.link_tag();
        let bot_syms: HashSet<&str> = st.trades.iter().filter(|t| t.active()).map(|t| t.symbol.as_str()).collect();
        let pos = snap
            .positions
            .iter()
            .filter(|p| !bot_syms.contains(p.symbol.as_str()))
            .map(|p| json!({"symbol": p.symbol, "side": p.side, "size": p.size, "avg_price": p.avg_price, "mark": p.mark_price, "stop_loss": p.stop_loss, "take_profit": p.take_profit, "upnl": p.unrealised_pnl}))
            .collect();
        let ord = snap
            .orders
            .iter()
            .filter(|o| Snapshot::foreign_order(o, &tag, &bot_syms))
            .map(|o| json!({"symbol": o["symbol"], "side": o["side"], "type": o["orderType"], "price": o["price"], "qty": o["qty"], "status": o["orderStatus"], "link": o["orderLinkId"], "stop_order_type": o["stopOrderType"], "reduce_only": o["reduceOnly"]}))
            .collect();
        (pos, ord)
    }

    // ------------------------------------------------------------ reconcile

    /// Venue -> state. With `write`, missing stops are re-attached, stale / orphan bot entry orders cancelled.
    async fn reconcile(&self, st: &mut LiveState, snap: &Snapshot, write: bool) -> anyhow::Result<Vec<String>> {
        let now = now_ms();
        let mut notes = Vec::new();
        st.env = self.env.as_str().into();
        st.roll_day(now);
        if st.created_ms == 0 {
            st.created_ms = now;
            let (p, o) = self.foreign(snap, st);
            st.foreign_at_start = Some(json!({"ts_ms": now, "positions": p, "orders": o}));
            notes.push(format!(
                "first reconcile: recorded {} foreign position(s) and {} foreign order(s); they are never touched",
                st.foreign_at_start.as_ref().unwrap()["positions"].as_array().map_or(0, |a| a.len()),
                st.foreign_at_start.as_ref().unwrap()["orders"].as_array().map_or(0, |a| a.len())
            ));
        }
        let tag = self.cfg.link_tag();
        let window_ms = (self.cfg.entry.window_min * 60_000.0) as i64;
        let t = self.venue.trading();
        // a recently "cancelled" entry whose own orders did fill (status lag) is revived, never left as a foreign position
        for i in 0..st.trades.len() {
            let tr = st.trades[i].clone();
            let recent = tr.entry_done_ms.is_some_and(|d| now - d < 30 * 60_000);
            if tr.status != "cancelled" || !recent || self.bot_position(snap, &tr).is_none() || st.trades.iter().any(|x| x.active() && x.symbol == tr.symbol) {
                continue;
            }
            let (mut q, mut v) = (0.0, 0.0);
            for oid in &tr.order_ids {
                if let Some(o) = t.order_status(&tr.symbol, oid).await? {
                    q += num(&o["cumExecQty"]);
                    v += num(&o["cumExecQty"]) * num(&o["avgPrice"]);
                }
            }
            if q > 0.0 {
                let x = &mut st.trades[i];
                x.status = "open".into();
                x.filled_qty = q;
                x.avg_entry = v / q;
                x.notes.push("revived: the entry orders filled (cumExecQty) after the trade was marked cancelled".into());
                notes.push(format!("{} {}: REVIVED as open (its orders filled {q})", x.id, x.symbol));
                let confirmed = x.approved_by != "auto";
                self.jw("position_update", json!({"action": "revived", "trade_id": x.id, "symbol": x.symbol, "filled_qty": q, "avg_entry": x.avg_entry}), &x.versions)?;
                if confirmed {
                    st.confirmed_trades += 1;
                }
            }
        }
        for i in 0..st.trades.len() {
            if !st.trades[i].active() {
                continue;
            }
            let tr = st.trades[i].clone();
            if tr.status == "submitting" && tr.lease_until_ms > now {
                continue;
            }
            let pos = self.bot_position(snap, &tr).cloned();
            // by order id, or by the trade's deterministic link prefix (a submit that crashed before saving its order ids)
            let own = format!("{tag}{}-", tr.id);
            let live_orders: Vec<&Value> = snap
                .orders
                .iter()
                .filter(|o| {
                    o["symbol"] == tr.symbol.as_str()
                        && (tr.order_ids.iter().any(|id| o["orderId"] == id.as_str()) || o["orderLinkId"].as_str().is_some_and(|l| l.starts_with(&own)))
                })
                .collect();
            match pos {
                Some(p) => {
                    let x = &mut st.trades[i];
                    if x.status != "open" {
                        x.status = "open".into();
                        if x.filled_qty <= 0.0 {
                            x.filled_qty = p.size;
                        }
                        if x.avg_entry <= 0.0 {
                            x.avg_entry = p.avg_price;
                        }
                        notes.push(format!("{} {}: position found, trade open", x.id, x.symbol));
                    }
                    if p.stop_loss <= 0.0 {
                        let msg = format!("{} {}: EXCHANGE STOP MISSING (stop {} expected)", x.id, x.symbol, x.stop);
                        notes.push(msg.clone());
                        if write {
                            let tick = self.rules_for(&x.symbol).await.map(|r| r.tick_size).unwrap_or(0.0);
                            let res = self.venue.set_stop(&x.symbol, &format_step(x.stop, tick), &self.cfg.stop_trigger_by).await;
                            notes.push(format!("  re-attach stop: {}", res.as_ref().map(|_| "ok".to_string()).unwrap_or_else(|e| format!("FAILED {e:#}"))));
                            self.jw("position_update", json!({"action": "reattach_stop", "trade_id": x.id, "symbol": x.symbol, "stop": x.stop, "ok": res.is_ok()}), &x.versions)?;
                        }
                    }
                    for o in &live_orders {
                        let age = now - num(&o["createdTime"]) as i64;
                        if age > window_ms + 60_000 && o["reduceOnly"] != true {
                            notes.push(format!("{} {}: stale bot entry order {} ({}s old)", tr.id, tr.symbol, o["orderId"], age / 1000));
                            if write {
                                let r = self.venue.cancel(&tr.symbol, o["orderId"].as_str().unwrap_or_default()).await;
                                self.jw(
                                    "order",
                                    json!({"action": "cancel_stale", "trade_id": tr.id, "symbol": tr.symbol, "order_id": o["orderId"], "ok": r.is_ok()}),
                                    &tr.versions,
                                )?;
                            }
                        }
                    }
                }
                None if tr.status == "open" || tr.filled_qty > 0.0 => {
                    // position gone: stop, target or a reduce-only close filled
                    let since = tr.first_fill_ms.unwrap_or(tr.submitted_ms) - 5_000;
                    let recs: Vec<Value> = t.closed_pnl(&tr.symbol, since).await?.into_iter().filter(|r| num(&r["createdTime"]) as i64 >= since).collect();
                    if recs.is_empty() {
                        notes.push(format!("{} {}: position gone, closed P&L not yet published (kept open)", tr.id, tr.symbol));
                        continue;
                    }
                    let pnl: f64 = recs.iter().map(|r| num(&r["closedPnl"])).sum();
                    let qty: f64 = recs.iter().map(|r| num(&r["closedSize"])).sum();
                    let exit_avg = if qty > 0.0 { recs.iter().map(|r| num(&r["avgExitPrice"]) * num(&r["closedSize"])).sum::<f64>() / qty } else { 0.0 };
                    let closed_ms = recs.iter().map(|r| num(&r["updatedTime"]) as i64).max().unwrap_or(now);
                    let execs = t.executions_for(&tr.symbol, None, Some(since)).await.unwrap_or_default();
                    let exits: Vec<&Value> = execs
                        .iter()
                        .filter(|e| !tr.order_ids.iter().any(|id| e["orderId"] == id.as_str()) && e["side"] == tr.side.map(|s| s.bybit_close()).unwrap_or_default())
                        .collect();
                    let close_tag = format!("{tag}x-");
                    let kind = exits
                        .iter()
                        .find_map(|e| {
                            if e["orderLinkId"].as_str().is_some_and(|l| l.starts_with(&close_tag)) {
                                return Some("bot_close".to_string());
                            }
                            e["stopOrderType"].as_str().filter(|s| !s.is_empty() && *s != "UNKNOWN").map(String::from).or_else(|| e["createType"].as_str().map(String::from))
                        })
                        .unwrap_or_else(|| "unknown".into());
                    for e in &exits {
                        let px = num(&e["execPrice"]);
                        let s = tr.side.map(|s| s.sign()).unwrap_or(1.0);
                        let slip_vs_stop = if tr.stop > 0.0 { (tr.stop - px) * s / tr.stop * 1e4 } else { 0.0 };
                        self.jw(
                            "fill",
                            json!({"leg": "exit", "trade_id": tr.id, "symbol": tr.symbol, "side": tr.side, "tier": tr.tier, "order_id": e["orderId"], "exec_id": e["execId"],
                                   "exec_price": px, "exec_qty": num(&e["execQty"]), "exec_fee": num(&e["execFee"]), "is_maker": e["isMaker"], "exec_ms": num(&e["execTime"]) as i64,
                                   "exit_kind": kind, "stop": tr.stop, "target": tr.target, "slip_bps_vs_stop": slip_vs_stop, "create_type": e["createType"], "stop_order_type": e["stopOrderType"]}),
                            &tr.versions,
                        )?;
                    }
                    let x = &mut st.trades[i];
                    x.status = "closed".into();
                    x.closed_ms = Some(closed_ms);
                    x.exit_avg = Some(exit_avg);
                    x.exit_kind = Some(kind.clone());
                    x.realised_usd = Some(pnl);
                    let r_net = if x.risk_to_stop_usd > 0.0 { pnl / x.risk_to_stop_usd } else { 0.0 };
                    notes.push(format!("{} {}: CLOSED by {kind} at {exit_avg:.6} -> {pnl:+.4} USD ({r_net:+.3} R)", x.id, x.symbol));
                    self.jw(
                        "position_update",
                        json!({"action": "closed", "trade_id": x.id, "intent_id": x.intent_id, "symbol": x.symbol, "side": x.side, "tier": x.tier, "qty": x.filled_qty, "avg_entry": x.avg_entry,
                               "exit_avg": exit_avg, "exit_kind": kind, "stop": x.stop, "target": x.target, "realised_usd": pnl, "risk_to_stop_usd": x.risk_to_stop_usd,
                               "max_loss_usd": x.max_loss_usd, "r_net": r_net, "opened_ms": x.first_fill_ms, "closed_ms": closed_ms, "closed_pnl_records": recs.len()}),
                        &x.versions,
                    )?;
                }
                None => {
                    // never filled: entry still resting, or cancelled / expired
                    if live_orders.is_empty() {
                        let x = &mut st.trades[i];
                        x.status = "cancelled".into();
                        x.entry_end_reason.get_or_insert_with(|| "not_on_venue".into());
                        notes.push(format!("{} {}: entry no longer on the venue, never filled -> cancelled", x.id, x.symbol));
                    } else {
                        for o in &live_orders {
                            let age = now - num(&o["createdTime"]) as i64;
                            if age > window_ms + 60_000 {
                                notes.push(format!("{} {}: stale bot entry {} ({}s old, window {} min)", tr.id, tr.symbol, o["orderId"], age / 1000, self.cfg.entry.window_min));
                                if write {
                                    let r = self.venue.cancel(&tr.symbol, o["orderId"].as_str().unwrap_or_default()).await;
                                    self.jw(
                                        "order",
                                        json!({"action": "cancel_stale", "trade_id": tr.id, "symbol": tr.symbol, "order_id": o["orderId"], "ok": r.is_ok()}),
                                        &tr.versions,
                                    )?;
                                }
                            }
                        }
                    }
                }
            }
        }
        // bot-tagged orders that belong to no active trade (a crashed submit): ours, so cancelled when writing
        let known: HashSet<String> = st.trades.iter().filter(|t| t.active()).flat_map(|t| t.order_ids.clone()).collect();
        let own_prefixes: Vec<String> = st.trades.iter().filter(|t| t.active()).map(|t| format!("{tag}{}-", t.id)).collect();
        for o in snap.orders.iter().filter(|o| Snapshot::bot_order(o, &tag)) {
            let id = o["orderId"].as_str().unwrap_or_default();
            let leased = st.trades.iter().any(|t| t.status == "submitting" && t.lease_until_ms > now && t.symbol == o["symbol"].as_str().unwrap_or_default());
            let owned = o["orderLinkId"].as_str().is_some_and(|l| own_prefixes.iter().any(|p| l.starts_with(p.as_str())));
            if !known.contains(id) && !owned && !leased {
                notes.push(format!("orphan bot order {} {} {}", o["symbol"], o["orderLinkId"], id));
                if write {
                    let r = self.venue.cancel(o["symbol"].as_str().unwrap_or_default(), id).await;
                    self.jw("order", json!({"action": "cancel_orphan", "symbol": o["symbol"], "order_id": id, "link": o["orderLinkId"], "ok": r.is_ok()}), &self.versions(None))?;
                }
            }
        }
        Ok(notes)
    }

    async fn rules_for(&self, symbol: &str) -> anyhow::Result<InstrumentRules> {
        let all = self.venue.exec_market().instruments().await?;
        all.iter().find(|i| i.symbol == symbol).map(rules_of).with_context(|| format!("no instrument {symbol}"))
    }

    // ------------------------------------------------------------ facts

    fn sectors(&self) -> HashMap<String, String> {
        let p = self.ctx.s.root.join(&self.cfg.correlation.sector_file);
        let Ok(text) = std::fs::read_to_string(&p) else { return HashMap::new() };
        let Ok(y) = serde_yaml::from_str::<serde_yaml::Value>(&text) else { return HashMap::new() };
        let mut m = HashMap::new();
        for key in ["sectors", "overrides"] {
            if let Some(map) = y[key].as_mapping() {
                for (k, v) in map {
                    if let (Some(k), Some(v)) = (k.as_str(), v.as_str()) {
                        m.insert(k.to_string(), v.to_string());
                    }
                }
            }
        }
        m
    }

    /// Rolling correlation of 1h log returns to BTC (mainnet bars), cached `cache_hours`.
    async fn btc_corr(&self, symbols: &[String]) -> HashMap<String, Option<f64>> {
        let c = &self.cfg.correlation;
        let mut cache: serde_json::Map<String, Value> = std::fs::read_to_string(&self.paths.corr_cache).ok().and_then(|t| serde_json::from_str(&t).ok()).unwrap_or_default();
        let now = now_ms();
        let ttl = (c.cache_hours * HOUR as f64) as i64;
        let mut out = HashMap::new();
        let freq = Freq::parse(&c.freq).unwrap_or(Freq::H1);
        let start = now - c.window_hours as i64 * HOUR;
        let mut btc: Option<Vec<(i64, f64)>> = None;
        let mut dirty = false;
        for s in symbols {
            if *s == c.btc_symbol {
                out.insert(s.clone(), Some(1.0));
                continue;
            }
            if let Some(v) = cache.get(s)
                && now - v["ts_ms"].as_i64().unwrap_or(0) < ttl
                && v["window_hours"].as_u64() == Some(c.window_hours as u64)
            {
                out.insert(s.clone(), v["corr"].as_f64());
                continue;
            }
            let md = self.venue.main_market();
            if btc.is_none() {
                btc = md.klines(&c.btc_symbol, freq, PriceType::Last, start, now).await.ok().map(|k| k.iter().map(|b| (b.open_ts, b.close)).collect());
            }
            let corr = match (&btc, md.klines(s, freq, PriceType::Last, start, now).await) {
                (Some(b), Ok(k)) => {
                    let a: Vec<(i64, f64)> = k.iter().map(|x| (x.open_ts, x.close)).collect();
                    let (ra, rb) = lv::aligned_log_returns(&a, b);
                    if ra.len() >= c.min_obs as usize { lv::pearson(&ra, &rb) } else { None }
                }
                _ => None,
            };
            cache.insert(s.clone(), json!({"corr": corr, "ts_ms": now, "window_hours": c.window_hours}));
            dirty = true;
            out.insert(s.clone(), corr);
        }
        if dirty {
            let _ = write_json(&self.paths.corr_cache, &Value::Object(cache));
        }
        out
    }

    /// Mainnet turnover rank among Trading USDT linear perps, with spread: symbol -> (rank, turnover, bid, ask, symbolType).
    async fn ranks(&self) -> anyhow::Result<HashMap<String, (u32, f64, f64, f64, String)>> {
        let md = self.venue.main_market();
        let ins: HashMap<String, Instrument> = md.instruments().await?.into_iter().filter(usdt_perp).map(|i| (i.symbol.clone(), i)).collect();
        let mut rows: Vec<_> = md.tickers().await?.into_iter().filter(|t| ins.contains_key(&t.symbol)).collect();
        rows.sort_by(|a, b| b.turnover_24h.partial_cmp(&a.turnover_24h).unwrap_or(std::cmp::Ordering::Equal));
        Ok(rows.into_iter().enumerate().map(|(k, t)| (t.symbol.clone(), ((k + 1) as u32, t.turnover_24h, t.bid1, t.ask1, ins[&t.symbol].symbol_type.clone()))).collect())
    }

    async fn facts(&self, symbols: &[String], snap: &Snapshot, st: &LiveState) -> anyhow::Result<HashMap<String, SymbolFacts>> {
        let ins: HashMap<String, Instrument> = self.venue.exec_market().instruments().await?.into_iter().map(|i| (i.symbol.clone(), i)).collect();
        let ranks = self.ranks().await?;
        let sectors = self.sectors();
        let corr = self.btc_corr(symbols).await;
        let tag = self.cfg.link_tag();
        let bot_syms: HashSet<&str> = st.trades.iter().filter(|t| t.active()).map(|t| t.symbol.as_str()).collect();
        let mut out = HashMap::new();
        for s in symbols {
            let i = ins.get(s);
            let book = self.venue.exec_market().orderbook(s, BOOK_LEVELS).await.unwrap_or_default();
            let (bd, ad) = book.depth_usd(self.cfg.depth.band_bps);
            let (rank, turnover, _, _, stype) = ranks.get(s).cloned().unwrap_or((u32::MAX, 0.0, 0.0, 0.0, String::new()));
            let foreign_pos = snap.positions.iter().any(|p| p.symbol == *s && !bot_syms.contains(s.as_str()));
            let foreign_ord = snap.orders.iter().any(|o| o["symbol"] == s.as_str() && Snapshot::foreign_order(o, &tag, &bot_syms));
            out.insert(
                s.clone(),
                SymbolFacts {
                    symbol: s.clone(),
                    rules: i.map(rules_of),
                    usdt_linear_perp: i.is_some_and(usdt_perp),
                    symbol_type: i.map(|i| i.symbol_type.clone()).unwrap_or(stype),
                    turnover_24h: turnover,
                    rank,
                    best_bid: book.best_bid().unwrap_or(0.0),
                    best_ask: book.best_ask().unwrap_or(0.0),
                    bid_depth_usd: bd,
                    ask_depth_usd: ad,
                    sector: sectors.get(s).cloned(),
                    btc_corr: corr.get(s).cloned().flatten(),
                    foreign_activity: foreign_pos || foreign_ord,
                },
            );
        }
        Ok(out)
    }

    /// Fee rates for sizing: config/fees.yaml by class, never below the account's own rates when the venue reports them.
    async fn fee_rates(&self, symbol: &str, symbol_type: &str) -> FeeRates {
        let class = if symbol_type.is_empty() || symbol_type == "innovation" { "crypto" } else { "tradfi" };
        let mut r = self.fees.rates(symbol, Some(class));
        if let Ok(Some((m, t))) = self.venue.trading().fee_rate_bps(symbol).await {
            r.maker_bps = r.maker_bps.max(m);
            r.taker_bps = r.taker_bps.max(t);
        }
        r
    }

    fn live_positions(&self, st: &LiveState, snap: &Snapshot) -> Vec<LivePos> {
        st.trades
            .iter()
            .filter(|t| t.active())
            .filter_map(|t| {
                let side = t.side?;
                Some(match self.bot_position(snap, t) {
                    Some(p) => LivePos {
                        symbol: t.symbol.clone(),
                        side,
                        qty: p.size,
                        entry_price: p.avg_price,
                        mark: p.mark_price,
                        open_pnl_usd: p.unrealised_pnl + p.cur_realised_pnl,
                        max_loss_usd: t.max_loss_usd,
                        pending: false,
                    },
                    None => LivePos {
                        symbol: t.symbol.clone(),
                        side,
                        qty: t.qty,
                        entry_price: t.first_peg_price.max(t.entry_ref),
                        mark: 0.0,
                        open_pnl_usd: 0.0,
                        max_loss_usd: t.max_loss_usd,
                        pending: true,
                    },
                })
            })
            .collect()
    }

    /// Full decision for `intent` against the current venue state; sized at the current touch.
    async fn decide(&self, intent: &OrderIntent, st: &LiveState, snap: &Snapshot) -> anyhow::Result<(LiveDecision, HashMap<String, SymbolFacts>, FeeRates)> {
        let positions = self.live_positions(st, snap);
        let mut syms: Vec<String> = positions.iter().map(|p| p.symbol.clone()).collect();
        syms.push(intent.symbol.clone());
        syms.sort();
        syms.dedup();
        let facts = self.facts(&syms, snap, st).await?;
        let f = facts.get(&intent.symbol);
        let touch = f.and_then(|f| match intent.side.side() {
            Some(Side::Long) => (f.best_bid > 0.0).then_some(f.best_bid),
            Some(Side::Short) => (f.best_ask > 0.0).then_some(f.best_ask),
            None => None,
        });
        let fees = self.fee_rates(&intent.symbol, f.map(|f| f.symbol_type.as_str()).unwrap_or("")).await;
        let placement = DecideConfig::load(&self.ctx.s.root, self.ctx.risk.clone())?.placement;
        let account =
            LiveAccount { available_usd: snap.available, entries_today: st.entries_today, realised_today_usd: st.realised_today(now_ms()), kill: self.paths.kill_active() };
        // Learning trades: only while the approval gate is manual (each entry approved by the user) may the EV gate use the
        // lower learning minimum; once the gate is auto the configured min_net_expected_r applies again.
        let cfg = match self.cfg.learning_min_net_expected_r {
            Some(l) if st.gate_manual(&self.cfg) => {
                let mut c = self.cfg.clone();
                c.min_net_expected_r = l;
                c
            }
            _ => self.cfg.clone(),
        };
        let d = lv::decide_live(intent, touch, &account, &positions, &facts, &fees, &placement, &cfg);
        Ok((d, facts, fees))
    }

    fn print_decision(&self, d: &LiveDecision) {
        println!(
            "{} {} {:?} tier {} | qty {} @ {} stop {} tp {:?} | notional {:.2} (cap {:.2}) risk-to-stop {:.4} max-loss {:.4} USD | stop {:.3}% {:?} sigma | netE[R] {:?} | clusters {:?}{}",
            d.decision.to_uppercase(),
            d.symbol,
            d.side,
            d.tier.as_deref().unwrap_or("-"),
            d.qty,
            d.entry,
            d.stop,
            d.target,
            d.notional_usd,
            d.notional_cap_usd,
            d.risk_to_stop_usd,
            d.max_loss_usd,
            d.stop_pct.unwrap_or(0.0),
            d.stop_sigma.map(|x| (x * 100.0).round() / 100.0),
            d.net_er_r.map(|x| (x * 1000.0).round() / 1000.0),
            d.clusters,
            if d.min_lot_used { " | MIN LOT" } else { "" }
        );
        for (k, v) in &d.details {
            println!("  REFUSED {k}: {v}");
        }
    }

    // ------------------------------------------------------------ status / tiers / reconcile

    pub async fn status(&self) -> anyhow::Result<()> {
        // load -> reconcile -> save under one lock: a concurrent submit's reservation must never be overwritten
        let lock = if self.venue.writable { Some(StateLock::acquire(&self.paths.root).await?) } else { None };
        let mut st = LiveState::load(&self.paths.state())?;
        let snap = self.snapshot().await?;
        let notes = self.reconcile(&mut st, &snap, false).await?;
        if lock.is_some() {
            // bookkeeping only (fills, closes); a read-only run (mainnet without the guard) writes no local state
            st.save(&self.paths.state())?;
        }
        drop(lock);
        self.print_status(&st, &snap, &notes).await;
        let (fp, fo) = self.foreign(&snap, &st);
        self.jw("snapshot", json!({"source": "atx live status", "equity": snap.equity, "available": snap.available, "usdt": snap.coin, "bot_trades": st.trades.iter().filter(|t| t.active()).collect::<Vec<_>>(), "foreign_positions": fp, "foreign_orders": fo, "notes": notes}), &self.versions(None))?;
        Ok(())
    }

    async fn print_status(&self, st: &LiveState, snap: &Snapshot, notes: &[String]) {
        let now = now_ms();
        let c = &self.cfg;
        println!(
            "atx live status  env {}  {}  profile {}  kill {}",
            self.env.as_str(),
            if self.venue.writable { "WRITES ENABLED" } else { "READ-ONLY" },
            c.profile_hash(),
            if self.paths.kill_active() { "ON" } else { "off" }
        );
        println!(
            "approval gate: {} ({} of {} confirmed trades)",
            if st.gate_manual(c) { "MANUAL (user approves each entry)" } else { "auto" },
            st.confirmed_trades,
            c.approval.manual_trades
        );
        let coin = snap.coin.as_ref();
        println!(
            "USDT: wallet balance {} equity {} available-to-withdraw {} | account equity {:.2} USD, available {:.2} USD",
            coin.map(|c| c["walletBalance"].to_string()).unwrap_or_else(|| "?".into()),
            coin.map(|c| c["equity"].to_string()).unwrap_or_else(|| "?".into()),
            coin.map(|c| c["availableToWithdraw"].to_string()).unwrap_or_else(|| "?".into()),
            snap.equity,
            snap.available
        );
        if let Ok(Some((m, t))) = self.venue.trading().fee_rate_bps("BTCUSDT").await {
            println!("account fee rate BTCUSDT: maker {m:.2} bp, taker {t:.2} bp (sizing uses max(config, account))");
        }
        let positions = self.live_positions(st, snap);
        println!("\nbot positions / resting entries ({})", positions.len());
        for t in st.trades.iter().filter(|t| t.active()) {
            match self.bot_position(snap, t) {
                Some(p) => println!(
                    "  {} {:<12} {:<5} size {} avg {} mark {} EXCHANGE STOP {} TP {} uPnL {:+.4} (+realised {:+.4}) | planned stop {} max-loss {:.2} tier {}",
                    t.id, p.symbol, p.side, p.size, p.avg_price, p.mark_price, p.stop_loss, p.take_profit, p.unrealised_pnl, p.cur_realised_pnl, t.stop, t.max_loss_usd, t.tier
                ),
                None => println!("  {} {:<12} {:?} {} pending entry qty {} stop {} (orders {:?})", t.id, t.symbol, t.side, t.status, t.qty, t.stop, t.order_ids),
            }
        }
        let tag = c.link_tag();
        let bot_orders: Vec<&Value> = snap.orders.iter().filter(|o| Snapshot::bot_order(o, &tag)).collect();
        println!("bot orders ({})", bot_orders.len());
        for o in bot_orders {
            println!(
                "  {} {} {} {} @ {} qty {} SL {} {} link {}",
                o["symbol"], o["side"], o["orderType"], o["timeInForce"], o["price"], o["qty"], o["stopLoss"], o["orderStatus"], o["orderLinkId"]
            );
        }
        let (fp, fo) = self.foreign(snap, st);
        println!("\nforeign positions ({}) and orders ({}): NOT the bot's, never touched", fp.len(), fo.len());
        for p in &fp {
            println!(
                "  POSITION {} {} size {} avg {} mark {} SL {} TP {} uPnL {}",
                p["symbol"], p["side"], p["size"], p["avg_price"], p["mark"], p["stop_loss"], p["take_profit"], p["upnl"]
            );
        }
        for o in &fo {
            println!(
                "  ORDER    {} {} {} @ {} qty {} {} stopOrderType {} link {}",
                o["symbol"], o["side"], o["type"], o["price"], o["qty"], o["status"], o["stop_order_type"], o["link"]
            );
        }
        let realised = st.realised_today(now);
        let open_loss: f64 = positions.iter().filter(|p| !p.pending).map(|p| p.open_pnl_usd.min(0.0)).sum();
        let open_pnl: f64 = positions.iter().filter(|p| !p.pending).map(|p| p.open_pnl_usd).sum();
        let gross: f64 = positions.iter().map(|p| p.qty * if p.pending || p.mark <= 0.0 { p.entry_price } else { p.mark }).sum();
        let longs = positions.iter().filter(|p| p.side == Side::Long).count() as u32;
        let shorts = positions.iter().filter(|p| p.side == Side::Short).count() as u32;
        let day_pnl = realised + open_loss;
        println!(
            "\ntoday ({} UTC): entries {} / {} | realised {:+.4} USD | open P&L {:+.4} | day P&L for the stop {:+.4} (stop at -{})",
            st.day, st.entries_today, c.max_entries_per_day, realised, open_pnl, day_pnl, c.daily_loss_stop_usd
        );
        println!(
            "limits left: entries {} | open slots {} | long {} short {} | gross {:.2} of {} USD | loss to the day stop {:.2} USD{}",
            c.max_entries_per_day.saturating_sub(st.entries_today),
            c.max_open_positions.saturating_sub(positions.len() as u32),
            c.max_same_direction.saturating_sub(longs),
            c.max_same_direction.saturating_sub(shorts),
            c.max_gross_notional_usd - gross,
            c.max_gross_notional_usd,
            c.daily_loss_stop_usd + day_pnl,
            if day_pnl <= -c.daily_loss_stop_usd { "  DAILY STOP HIT: no new entries until the next UTC day" } else { "" }
        );
        let closed: Vec<&TradeRec> = st.trades.iter().filter(|t| t.status == "closed").collect();
        if !closed.is_empty() {
            let tot: f64 = closed.iter().filter_map(|t| t.realised_usd).sum();
            println!("closed bot trades: {} (net {:+.4} USD)", closed.len(), tot);
        }
        for n in notes {
            println!("note: {n}");
        }
    }

    pub async fn tiers(&self, top: usize) -> anyhow::Result<()> {
        let ranks = self.ranks().await?;
        let mut rows: Vec<_> = ranks.into_iter().collect();
        rows.sort_by_key(|(_, r)| r.0);
        let mut counts: Vec<(String, usize, Vec<String>)> = self.cfg.tiers.iter().map(|t| (t.name.clone(), 0, vec![])).collect();
        let mut none = 0;
        println!("{:>4} {:<16} {:<11} {:>12} {:>8} {:<9} {:>6}", "rank", "symbol", "type", "turnover24h", "spread", "tier", "cap");
        for (k, (s, (rank, tv, bid, ask, stype))) in rows.iter().enumerate() {
            let spread = if *bid > 0.0 && *ask >= *bid { (ask - bid) / ((ask + bid) / 2.0) * 1e4 } else { f64::INFINITY };
            let excluded = self.cfg.is_excluded(s) || self.cfg.excluded_symbol_types.iter().any(|t| t == stype);
            let tier = if excluded { None } else { self.cfg.tier_for(*rank, *tv, spread) };
            match tier {
                Some(t) => {
                    let c = counts.iter_mut().find(|c| c.0 == t.name).unwrap();
                    c.1 += 1;
                    if c.2.len() < 60 {
                        c.2.push(s.clone());
                    }
                }
                None => none += 1,
            }
            if k < top {
                println!(
                    "{rank:>4} {s:<16} {:<11} {:>10.1}M {:>6.2}bp {:<9} {:>6}",
                    if stype.is_empty() { "crypto" } else { stype },
                    tv / 1e6,
                    spread,
                    if excluded { "EXCLUDED" } else { tier.map(|t| t.name.as_str()).unwrap_or("none") },
                    tier.map(|t| format!("{:.0}", t.notional_cap_usd)).unwrap_or_else(|| "-".into())
                );
            }
        }
        println!("\nper tier (all {} Trading USDT linear perps):", rows.len());
        for (name, n, syms) in &counts {
            println!("  {name:<9} {n:>4}  {}{}", syms.join(" "), if *n > syms.len() { " ..." } else { "" });
        }
        println!("  none      {none:>4}  (excluded or below every tier: not tradable)");
        Ok(())
    }

    pub async fn reconcile_cmd(&self, read_only: bool) -> anyhow::Result<()> {
        let write = !read_only && self.venue.writable;
        let _l = StateLock::acquire(&self.paths.root).await?;
        let mut st = LiveState::load(&self.paths.state())?;
        let snap = self.snapshot().await?;
        let notes = self.reconcile(&mut st, &snap, write).await?;
        if self.venue.writable {
            st.save(&self.paths.state())?;
        }
        println!("reconcile {} ({}): {} note(s)", self.env.as_str(), if write { "repairs enabled" } else { "READ-ONLY" }, notes.len());
        for n in &notes {
            println!("  {n}");
        }
        let (fp, fo) = self.foreign(&snap, &st);
        self.jw(
            "snapshot",
            json!({"source": "atx live reconcile", "write": write, "notes": notes, "foreign_positions": fp, "foreign_orders": fo, "equity": snap.equity}),
            &self.versions(None),
        )?;
        drop(_l);
        self.print_status(&st, &snap, &[]).await;
        Ok(())
    }

    // ------------------------------------------------------------ propose / approve / submit

    fn read_intent(path: &Path) -> anyhow::Result<OrderIntent> {
        let text = std::fs::read_to_string(path).with_context(|| path.display().to_string())?;
        let t = text.trim();
        // one JSON object, or a JSONL file with exactly one intent line
        let lines: Vec<&str> = t.lines().map(str::trim).filter(|l| !l.is_empty()).collect();
        let body = if serde_json::from_str::<Value>(t).is_ok() {
            t
        } else if lines.len() == 1 {
            lines[0]
        } else {
            bail!("intent file: one OrderIntent per proposal ({} lines)", lines.len())
        };
        let mut i: OrderIntent = serde_json::from_str(body).context("intent file: one OrderIntent JSON object (docs/EXECUTION_PIPELINE.md)")?;
        i.symbol = i.symbol.trim().to_uppercase();
        Ok(i)
    }

    pub async fn propose(&self, file: &Path, test: &TestOpts) -> anyhow::Result<()> {
        self.ensure_writes(test)?;
        let intent = Self::read_intent(file)?;
        let versions = self.versions(Some(&intent));
        let lock = StateLock::acquire(&self.paths.root).await?;
        let mut st = LiveState::load(&self.paths.state())?;
        let snap = self.snapshot().await?;
        let notes = self.reconcile(&mut st, &snap, true).await?;
        st.save(&self.paths.state())?;
        let (d, _, fees) = self.decide(&intent, &st, &snap).await?;
        self.print_decision(&d);
        self.jw("risk_decision", json!({"source": "atx live propose", "intent": intent, "decision": d, "fees": fees, "reconcile_notes": notes, "entries_today": st.entries_today, "confirmed_trades": st.confirmed_trades}), &versions)?;
        if !d.accepted() {
            bail!("REFUSED by the live risk engine: {}", d.rejections.join(", "));
        }
        let now = now_ms();
        let id = proposal_id(now, &intent.intent_id);
        let needs_approval = st.gate_manual(&self.cfg);
        let prop = Proposal {
            id: id.clone(),
            env: self.env.as_str().into(),
            created_ms: now,
            expires_ms: now + (self.cfg.approval.ttl_min * 60_000.0) as i64,
            confirm_code: confirm_code(&id, &intent),
            intent,
            decision: d,
            needs_approval,
            versions: versions.clone(),
        };
        write_json(&self.paths.pending(&id), &serde_json::to_value(&prop)?)?;
        self.jw("decision", json!({"action": "proposal", "proposal_id": id, "needs_approval": needs_approval, "intent_id": prop.intent.intent_id, "symbol": prop.intent.symbol, "expires_ms": prop.expires_ms}), &versions)?;
        drop(lock);
        if needs_approval {
            println!("\nPENDING APPROVAL ({} of {} confirmed trades): {}", st.confirmed_trades, self.cfg.approval.manual_trades, self.paths.pending(&id).display());
            println!("nothing was sent. Only the USER approves, in an interactive terminal:  atx live approve {id}");
            println!("(or the user creates {} containing the confirm_code of the pending file). Expires in {} min.", self.paths.approval(&id).display(), self.cfg.approval.ttl_min);
            return Ok(());
        }
        self.submit_proposal(prop, "auto", test).await
    }

    pub async fn approve(&self, id: &str) -> anyhow::Result<()> {
        ensure!(
            std::io::stdin().is_terminal() && std::io::stdout().is_terminal(),
            "approve is for the user: run it yourself in an interactive terminal (stdin/stdout must be a TTY)"
        );
        self.ensure_writes(&TestOpts::default())?;
        let prop: Proposal = serde_json::from_str(&std::fs::read_to_string(self.paths.pending(id)).with_context(|| format!("no pending proposal {id}"))?)?;
        println!("proposal {id} ({}): {} {:?}", prop.env, prop.intent.symbol, prop.intent.side);
        self.print_decision(&prop.decision);
        println!("reason: {}", prop.intent.context.as_ref().and_then(|c| c.get("reason")).and_then(|v| v.as_str()).unwrap_or("-"));
        print!("type {} to approve and SUBMIT this order on {}: ", prop.confirm_code, self.env.as_str().to_uppercase());
        std::io::stdout().flush()?;
        let mut line = String::new();
        std::io::stdin().lock().read_line(&mut line)?;
        ensure!(line.trim().eq_ignore_ascii_case(&prop.confirm_code), "wrong code: not approved");
        write_json(&self.paths.approval(id), &json!({"code": prop.confirm_code, "by": "atx live approve (interactive)", "ts_ms": now_ms()}))?;
        self.jw("manual_action", json!({"action": "approve", "proposal_id": id, "via": "cli"}), &prop.versions)?;
        self.submit_proposal(prop, "user-cli", &TestOpts::default()).await
    }

    pub async fn submit(&self, id: &str, test: &TestOpts) -> anyhow::Result<()> {
        self.ensure_writes(test)?;
        let prop: Proposal = serde_json::from_str(&std::fs::read_to_string(self.paths.pending(id)).with_context(|| format!("no pending proposal {id}"))?)?;
        let st = LiveState::load(&self.paths.state())?;
        let approved_file = std::fs::read_to_string(self.paths.approval(id)).ok();
        let by = match approved_file {
            Some(t) if t.contains(&prop.confirm_code) => "user-file",
            Some(_) => bail!("{}: does not contain the confirmation code", self.paths.approval(id).display()),
            None if !st.gate_manual(&self.cfg) => "auto",
            None => bail!("proposal {id} needs the user's approval ({} of {} confirmed trades): atx live approve {id}", st.confirmed_trades, self.cfg.approval.manual_trades),
        };
        self.submit_proposal(prop, by, test).await
    }

    async fn submit_proposal(&self, prop: Proposal, approved_by: &str, test: &TestOpts) -> anyhow::Result<()> {
        self.ensure_writes(test)?;
        let now = now_ms();
        let id = prop.id.clone();
        let versions = prop.versions.clone();
        let finish = |status: &str, extra: Value| -> anyhow::Result<()> {
            let mut v = serde_json::to_value(&prop)?;
            v["final_status"] = json!(status);
            v["final"] = extra;
            write_json(&self.paths.done(&id), &v)?;
            let _ = std::fs::remove_file(self.paths.pending(&id));
            Ok(())
        };
        if now > prop.expires_ms {
            finish("expired", json!({"now_ms": now}))?;
            self.jw("decision", json!({"action": "expired", "proposal_id": id}), &versions)?;
            bail!("proposal {id} expired");
        }
        // ---- re-check and reserve under the lock
        let lock = StateLock::acquire(&self.paths.root).await?;
        let mut st = LiveState::load(&self.paths.state())?;
        let snap = self.snapshot().await?;
        self.reconcile(&mut st, &snap, true).await?;
        let (d, facts, fees) = self.decide(&prop.intent, &st, &snap).await?;
        self.print_decision(&d);
        self.jw("risk_decision", json!({"source": "atx live submit", "proposal_id": id, "approved_by": approved_by, "decision": d, "fees": fees}), &versions)?;
        if !d.accepted() {
            st.save(&self.paths.state())?;
            finish("refused_at_submit", serde_json::to_value(&d)?)?;
            bail!("REFUSED at submission: {}", d.rejections.join(", "));
        }
        let f = facts.get(&d.symbol).cloned().context("facts")?;
        let rule = f.rules.clone().context("rules")?;
        let side = d.side.context("side")?;
        let window_ms = test.test_window_secs.map(|s| s as i64 * 1000).unwrap_or((self.cfg.entry.window_min * 60_000.0) as i64);
        let mid = (f.best_bid + f.best_ask) / 2.0;
        let mut tr = TradeRec {
            id: id.clone(),
            intent_id: prop.intent.intent_id.clone(),
            symbol: d.symbol.clone(),
            side: Some(side),
            tier: d.tier.clone().unwrap_or_default(),
            status: "submitting".into(),
            qty: d.qty,
            entry_ref: prop.intent.entry.as_ref().and_then(|e| e.price).unwrap_or(0.0),
            mid_at_submit: mid,
            first_peg_price: d.entry,
            stop: d.stop,
            target: d.target,
            risk_to_stop_usd: d.risk_to_stop_usd,
            max_loss_usd: d.max_loss_usd,
            stop_slip_bps: d.stop_slip_bps,
            // the chase may grow the notional only up to the tier / depth cap and the gross headroom left at the decision
            notional_usd: d.notional_cap_usd.min(d.notional_usd + (self.cfg.max_gross_notional_usd - d.gross_after_usd).max(0.0)),
            submitted_ms: now,
            lease_until_ms: now + window_ms + 120_000,
            approved_by: approved_by.into(),
            versions: versions.clone(),
            ..Default::default()
        };
        st.entries_today += 1;
        st.trades.push(tr.clone());
        st.save(&self.paths.state())?;
        drop(lock);

        // ---- leverage: set on the symbol before every entry (idempotent) and read back; no entry order unless the venue
        // holds it (fail closed). The symbol has no foreign activity: decide_live checked.
        let lev = self.cfg.leverage.min(rule.max_leverage.max(1.0));
        let lev_ok = match self.apply_leverage(&d.symbol, lev).await {
            Ok(()) => true,
            Err(e) => {
                tr.notes.push(format!("leverage {lev}: {e:#}: entry not sent"));
                false
            }
        };

        // ---- pegged entry
        let out = if lev_ok { self.peg_entry(&mut tr, &rule, &fees, window_ms, test).await } else { Ok(PegEnd { reason: "leverage_not_set".into() }) };
        let out = match out {
            Ok(o) => o,
            Err(e) => {
                tr.notes.push(format!("peg loop error: {e:#}"));
                PegEnd { reason: "error".into() }
            }
        };
        // ---- fills, stop verification
        let fills = self.journal_entry_fills(&mut tr).await.unwrap_or_else(|e| {
            tr.notes.push(format!("fills: {e:#}"));
            0
        });
        if tr.filled_qty <= 0.0 {
            // never mark a filled entry as cancelled: the venue position is the last word
            tokio::time::sleep(Duration::from_millis(800)).await;
            if let Some(p) = self.venue.trading().positions().await?.into_iter().find(|p| p.symbol == tr.symbol && Some(side_of(&p.side)) == tr.side) {
                tr.filled_qty = p.size;
                tr.avg_entry = p.avg_price;
                tr.notes.push("fill found on the position (order status / executions lagged)".into());
            }
        }
        if tr.filled_qty > 0.0 {
            tokio::time::sleep(Duration::from_millis(800)).await;
            self.verify_stop(&mut tr, &rule).await?;
        }
        tr.entry_done_ms = Some(now_ms());
        tr.entry_end_reason = Some(out.reason.clone());
        tr.status = if tr.filled_qty > 0.0 { "open".into() } else { "cancelled".into() };
        let lock = StateLock::acquire(&self.paths.root).await?;
        let mut st = LiveState::load(&self.paths.state())?;
        if tr.filled_qty > 0.0 && approved_by != "auto" {
            st.confirmed_trades += 1;
        }
        tr.lease_until_ms = 0;
        match st.trades.iter_mut().find(|x| x.id == tr.id) {
            Some(x) => *x = tr.clone(),
            None => st.trades.push(tr.clone()),
        }
        st.save(&self.paths.state())?;
        drop(lock);
        let s = side.sign();
        let slip_mid = if tr.avg_entry > 0.0 && mid > 0.0 { (tr.avg_entry - mid) * s / mid * 1e4 } else { 0.0 };
        self.jw(
            "order",
            json!({"action": "entry_done", "trade_id": tr.id, "intent_id": tr.intent_id, "symbol": tr.symbol, "side": side, "tier": tr.tier, "reason": out.reason, "qty": tr.qty,
                   "filled_qty": tr.filled_qty, "avg_entry": tr.avg_entry, "mid_at_submit": mid, "entry_ref": tr.entry_ref, "slip_bps_vs_mid": slip_mid,
                   "paper_slip_bps": self.cfg.entry.slip_bps, "repegs": tr.repegs, "orders": tr.order_ids, "fills": fills, "submitted_ms": tr.submitted_ms,
                   "first_fill_ms": tr.first_fill_ms, "time_to_first_fill_ms": tr.first_fill_ms.map(|t| t - tr.submitted_ms), "window_ms": window_ms,
                   "stop": tr.stop, "target": tr.target, "risk_to_stop_usd": tr.risk_to_stop_usd, "max_loss_usd": tr.max_loss_usd, "approved_by": approved_by, "notes": tr.notes,
                   "test_hooks": if test.any() { json!({"behind_bps": test.test_behind_bps, "window_secs": test.test_window_secs}) } else { Value::Null }}),
            &versions,
        )?;
        finish(&tr.status, json!({"trade": tr}))?;
        println!(
            "\nentry {}: {} {} filled {} of {} @ {:.6} (mid at submit {mid:.6}, {slip_mid:+.2} bp) | {} re-pegs | {} | reason {}",
            tr.id,
            tr.symbol,
            tr.status,
            tr.filled_qty,
            tr.qty,
            tr.avg_entry,
            tr.repegs,
            if tr.filled_qty > 0.0 { format!("exchange stop {}", tr.stop) } else { "no position".to_string() },
            out.reason
        );
        if approved_by != "auto" && tr.filled_qty > 0.0 {
            println!("confirmed trades: {} of {} (gate {})", st.confirmed_trades, self.cfg.approval.manual_trades, if st.gate_manual(&self.cfg) { "manual" } else { "now AUTO" });
        }
        Ok(())
    }

    fn peg_target(&self, side: Side, book: &Book, tick: f64, test: &TestOpts) -> Option<f64> {
        let px = match side {
            Side::Long => book.best_bid()?,
            Side::Short => book.best_ask()?,
        };
        Some(match test.test_behind_bps {
            Some(b) if side == Side::Long => round_to_step(px * (1.0 - b / 1e4), tick, Round::Down),
            Some(b) => round_to_step(px * (1.0 + b / 1e4), tick, Round::Up),
            None => px,
        })
    }

    /// PostOnly limit at the touch, re-pegged every poll until filled, the window ends, the stop is breached, the chase
    /// would break the $2 cap, or KILL appears. The stop is attached to every entry order.
    /// Set `lev` on the symbol (Bybit's "leverage not modified" counts as success), then read it back.
    async fn apply_leverage(&self, symbol: &str, lev: f64) -> anyhow::Result<()> {
        self.venue.set_leverage(symbol, &format!("{lev}")).await.context("set-leverage")?;
        let got = self.venue.trading().leverage(symbol).await.context("leverage read-back")?;
        leverage_matches(lev, got)
    }

    async fn peg_entry(&self, tr: &mut TradeRec, rule: &InstrumentRules, fees: &FeeRates, window_ms: i64, test: &TestOpts) -> anyhow::Result<PegEnd> {
        let side = tr.side.context("side")?;
        let deadline = tr.submitted_ms + window_ms;
        let poll = Duration::from_millis(self.cfg.entry.poll_ms);
        let t = self.venue.trading();
        let mut done_qty = 0.0; // filled on finished orders
        let mut cur: Option<(String, f64)> = None; // (order id, price)
        let mut cur_qty = 0.0; // quantity of the resting order
        let mut n = 0u32;
        let reason;
        loop {
            // 1. the resting order first: a fill always wins over a later chase decision
            let mut cum = 0.0;
            if let Some((oid, _)) = cur.clone() {
                let o = t.order_status(&tr.symbol, &oid).await.ok().flatten();
                let status = o.as_ref().and_then(|o| o["orderStatus"].as_str()).unwrap_or("").to_string();
                cum = o.as_ref().map(|o| num(&o["cumExecQty"])).unwrap_or(0.0);
                if cum > 0.0 && tr.first_fill_ms.is_none() {
                    tr.first_fill_ms = Some(o.as_ref().map(|o| num(&o["updatedTime"]) as i64).filter(|x| *x > 0).unwrap_or_else(now_ms));
                }
                match status.as_str() {
                    "Filled" => {
                        cur = None;
                        reason = "filled".to_string();
                        break;
                    }
                    "Cancelled" | "Rejected" | "Deactivated" | "PartiallyFilledCanceled" => {
                        done_qty += cum;
                        self.jw(
                            "order",
                            json!({"action": "venue_cancelled", "trade_id": tr.id, "symbol": tr.symbol, "order_id": oid, "status": status, "cum_exec_qty": cum,
                                   "reject_reason": o.as_ref().map(|o| o["rejectReason"].clone()), "cancel_type": o.as_ref().map(|o| o["cancelType"].clone())}),
                            &tr.versions,
                        )?;
                        cur = None; // PostOnly would have crossed: re-place at the new touch
                        cum = 0.0;
                    }
                    _ => {}
                }
            }
            if self.paths.kill_active() {
                reason = "kill".to_string();
                break;
            }
            if now_ms() > deadline {
                reason = "window".to_string();
                break;
            }
            let book = self.venue.exec_market().orderbook(&tr.symbol, 1).await.unwrap_or_default();
            let Some(px) = self.peg_target(side, &book, rule.tick_size, test) else {
                tokio::time::sleep(poll).await;
                continue;
            };
            if (px - tr.stop) * side.sign() <= 0.0 {
                reason = "stop_breached".to_string();
                break;
            }
            // 2. the size allowed at this price: risk to the stop <= 1.70, notional <= cap / gross headroom, worst case <= 2 USD
            let allowed = lv::chase_qty(side, tr.qty, tr.notional_usd, px, tr.stop, fees, &self.cfg, rule.qty_step);
            let filled_now = done_qty + cum;
            let stop_reason = if filled_now > 0.0 { "partial_chase_limit" } else { "chase_limit" };
            if allowed <= 0.0 || !lv::peg_price_ok(side, allowed.max(filled_now), px, tr.stop, fees, &self.cfg, tr.stop_slip_bps) {
                reason = stop_reason.to_string();
                break;
            }
            let order_qty = round_to_step(allowed - done_qty, rule.qty_step, Round::Down);
            match cur.clone() {
                None => {
                    if order_qty < rule.min_qty || order_qty * px < rule.min_notional {
                        reason = if done_qty > 0.0 && allowed >= tr.qty { "filled" } else { stop_reason }.to_string();
                        break;
                    }
                    n += 1;
                    let link = entry_link(&self.cfg, &tr.id, n);
                    let req = OrderRequest {
                        symbol: tr.symbol.clone(),
                        side: side.bybit_open(),
                        order_type: "Limit",
                        qty: format_step(order_qty, rule.qty_step),
                        price: Some(format_step(px, rule.tick_size)),
                        time_in_force: self.cfg.entry.time_in_force.clone(),
                        take_profit: tr.target.map(|x| format_step(x, rule.tick_size)),
                        stop_loss: Some(format_step(tr.stop, rule.tick_size)),
                        tp_trigger_by: self.ctx.s.execution.tp_trigger_by.clone(),
                        sl_trigger_by: self.cfg.stop_trigger_by.clone(),
                        reduce_only: false,
                        order_link_id: link.clone(),
                    };
                    match self.venue.place(&req).await {
                        Ok(oid) => {
                            if n == 1 {
                                tr.first_peg_price = px;
                            }
                            tr.order_ids.push(oid.clone());
                            tr.link_ids.push(link.clone());
                            self.jw(
                                "order",
                                json!({"action": "place", "trade_id": tr.id, "symbol": tr.symbol, "side": side, "order_id": oid, "link": link, "price": px, "qty": order_qty,
                                       "stop": tr.stop, "target": tr.target, "tif": req.time_in_force, "bid": book.best_bid(), "ask": book.best_ask()}),
                                &tr.versions,
                            )?;
                            cur = Some((oid, px));
                            cur_qty = order_qty;
                        }
                        Err(e) => {
                            self.jw(
                                "order",
                                json!({"action": "place_error", "trade_id": tr.id, "symbol": tr.symbol, "link": link, "price": px, "error": format!("{e:#}")}),
                                &tr.versions,
                            )?;
                            reason = "place_error".to_string();
                            break;
                        }
                    }
                }
                Some((oid, opx)) if (px - opx).abs() > rule.tick_size * 0.5 => {
                    // the chase may only shrink the order
                    if order_qty <= cum + 1e-12 || order_qty - cum < rule.min_qty {
                        reason = stop_reason.to_string();
                        break;
                    }
                    let new_qty = (order_qty + 1e-12 < cur_qty).then(|| format_step(order_qty, rule.qty_step));
                    match self.venue.amend(&tr.symbol, &oid, &format_step(px, rule.tick_size), new_qty.as_deref()).await {
                        Ok(()) => {
                            tr.repegs += 1;
                            if new_qty.is_some() {
                                cur_qty = order_qty;
                            }
                            self.jw(
                                "order",
                                json!({"action": "repeg", "trade_id": tr.id, "symbol": tr.symbol, "order_id": oid, "from": opx, "to": px, "qty": cur_qty, "cum_exec_qty": cum,
                                       "bid": book.best_bid(), "ask": book.best_ask()}),
                                &tr.versions,
                            )?;
                            cur = Some((oid, px));
                        }
                        Err(e) => {
                            self.jw(
                                "order",
                                json!({"action": "repeg_error", "trade_id": tr.id, "symbol": tr.symbol, "order_id": oid, "to": px, "error": format!("{e:#}")}),
                                &tr.versions,
                            )?;
                        }
                    }
                }
                Some(_) => {}
            }
            tokio::time::sleep(poll).await;
        }
        if let Some((oid, _)) = cur {
            let r = self.venue.cancel(&tr.symbol, &oid).await;
            self.jw(
                "order",
                json!({"action": "cancel", "reason": reason, "trade_id": tr.id, "symbol": tr.symbol, "order_id": oid, "ok": r.is_ok(), "error": r.err().map(|e| format!("{e:#}"))}),
                &tr.versions,
            )?;
        }
        // the venue's final order states are the truth for the filled quantity and price
        tokio::time::sleep(Duration::from_millis(400)).await;
        let (mut q, mut v) = (0.0, 0.0);
        for oid in tr.order_ids.clone() {
            let o = t.order_status(&tr.symbol, &oid).await.ok().flatten();
            let (cum, avg) = o.as_ref().map(|o| (num(&o["cumExecQty"]), num(&o["avgPrice"]))).unwrap_or((0.0, 0.0));
            q += cum;
            v += cum * avg;
        }
        if q > 0.0 {
            tr.filled_qty = q;
            tr.avg_entry = if v > 0.0 { v / q } else { tr.first_peg_price };
        }
        Ok(PegEnd { reason })
    }

    /// Journal the entry executions of every order of the trade; sets filled qty, VWAP and the first fill time.
    async fn journal_entry_fills(&self, tr: &mut TradeRec) -> anyhow::Result<usize> {
        let t = self.venue.trading();
        let side = tr.side.context("side")?;
        // executions are published with a lag: wait until they cover the filled quantity (order status)
        let mut execs: Vec<(String, Value)> = Vec::new();
        for attempt in 0..12 {
            tokio::time::sleep(Duration::from_millis(if attempt == 0 { 500 } else { 1000 })).await;
            execs.clear();
            for oid in tr.order_ids.clone() {
                for e in t.executions_for(&tr.symbol, Some(&oid), None).await? {
                    if e["execType"] == "Trade" {
                        execs.push((oid.clone(), e));
                    }
                }
            }
            let got: f64 = execs.iter().map(|(_, e)| num(&e["execQty"])).sum();
            if got + 1e-12 >= tr.filled_qty {
                break;
            }
        }
        let (mut q, mut v, mut n) = (0.0, 0.0, 0usize);
        let mut first: Option<i64> = None;
        for (oid, e) in execs {
            let (px, qty, ts) = (num(&e["execPrice"]), num(&e["execQty"]), num(&e["execTime"]) as i64);
            q += qty;
            v += px * qty;
            n += 1;
            first = Some(first.map_or(ts, |f: i64| f.min(ts)));
            let slip = if tr.mid_at_submit > 0.0 { (px - tr.mid_at_submit) * side.sign() / tr.mid_at_submit * 1e4 } else { 0.0 };
            self.jw(
                "fill",
                json!({"leg": "entry", "trade_id": tr.id, "intent_id": tr.intent_id, "symbol": tr.symbol, "side": side, "tier": tr.tier, "order_id": oid, "exec_id": e["execId"],
                       "exec_price": px, "exec_qty": qty, "exec_fee": num(&e["execFee"]), "fee_rate": num(&e["feeRate"]), "is_maker": e["isMaker"], "exec_ms": ts,
                       "submitted_ms": tr.submitted_ms, "time_to_fill_ms": ts - tr.submitted_ms, "mid_at_submit": tr.mid_at_submit, "entry_ref": tr.entry_ref,
                       "slip_bps_vs_mid": slip, "paper_slip_bps": self.cfg.entry.slip_bps, "stop": tr.stop}),
                &tr.versions,
            )?;
        }
        if q > 0.0 && q + 1e-12 >= tr.filled_qty {
            tr.filled_qty = q;
            tr.avg_entry = v / q;
            tr.first_fill_ms = first.or(tr.first_fill_ms);
        } else if q + 1e-12 < tr.filled_qty {
            tr.notes.push(format!("executions cover {q} of the filled {} (journal incomplete)", tr.filled_qty));
        }
        Ok(n)
    }

    /// The position must carry the exchange stop. If it does not, attach it; if that fails, close the bot quantity.
    async fn verify_stop(&self, tr: &mut TradeRec, rule: &InstrumentRules) -> anyhow::Result<()> {
        let t = self.venue.trading();
        let pos = t.positions().await?.into_iter().find(|p| p.symbol == tr.symbol && Some(side_of(&p.side)) == tr.side);
        let Some(p) = pos else {
            tr.notes.push("no position after the fill (closed already?)".into());
            return Ok(());
        };
        if (p.stop_loss - tr.stop).abs() <= rule.tick_size * 0.5 {
            self.jw(
                "position_update",
                json!({"action": "stop_verified", "trade_id": tr.id, "symbol": tr.symbol, "exchange_stop": p.stop_loss, "take_profit": p.take_profit, "size": p.size}),
                &tr.versions,
            )?;
            return Ok(());
        }
        let r = self.venue.set_stop(&tr.symbol, &format_step(tr.stop, rule.tick_size), &self.cfg.stop_trigger_by).await;
        self.jw("position_update", json!({"action": "stop_attach", "trade_id": tr.id, "symbol": tr.symbol, "found_stop": p.stop_loss, "stop": tr.stop, "ok": r.is_ok(), "error": r.as_ref().err().map(|e| format!("{e:#}"))}), &tr.versions)?;
        if r.is_err() {
            let qty = round_to_step(tr.filled_qty.min(p.size), rule.qty_step, Round::Down);
            let res = self.close_qty(tr, qty, rule, "stop_attach_failed").await;
            tr.notes.push(format!(
                "STOP COULD NOT BE ATTACHED: emergency reduce-only close -> {}",
                res.as_ref().map(|_| "sent".to_string()).unwrap_or_else(|e| format!("FAILED {e:#}"))
            ));
        }
        Ok(())
    }

    async fn close_qty(&self, tr: &TradeRec, qty: f64, rule: &InstrumentRules, why: &str) -> anyhow::Result<String> {
        let side = tr.side.context("side")?;
        ensure!(qty > 0.0, "nothing to close");
        let req = OrderRequest {
            symbol: tr.symbol.clone(),
            side: side.bybit_close(),
            order_type: "Market",
            qty: format_step(qty, rule.qty_step),
            price: None,
            time_in_force: "IOC".into(),
            take_profit: None,
            stop_loss: None,
            tp_trigger_by: self.ctx.s.execution.tp_trigger_by.clone(),
            sl_trigger_by: self.cfg.stop_trigger_by.clone(),
            reduce_only: true,
            order_link_id: close_link(&self.cfg, &tr.id),
        };
        let r = self.venue.place(&req).await;
        self.jw("order", json!({"action": "reduce_only_close", "why": why, "trade_id": tr.id, "symbol": tr.symbol, "qty": qty, "link": req.order_link_id, "ok": r.is_ok(), "order_id": r.as_ref().ok(), "error": r.as_ref().err().map(|e| format!("{e:#}"))}), &tr.versions)?;
        r
    }

    // ------------------------------------------------------------ kill / flatten

    async fn cancel_bot_orders(&self, snap: &Snapshot) -> Vec<Value> {
        let tag = self.cfg.link_tag();
        let mut out = Vec::new();
        for o in snap.orders.iter().filter(|o| Snapshot::bot_order(o, &tag)) {
            let (sym, id) = (o["symbol"].as_str().unwrap_or_default(), o["orderId"].as_str().unwrap_or_default());
            let r = self.venue.cancel(sym, id).await;
            out.push(json!({"symbol": sym, "order_id": id, "link": o["orderLinkId"], "ok": r.is_ok(), "error": r.err().map(|e| format!("{e:#}"))}));
        }
        out
    }

    pub async fn kill(&self) -> anyhow::Result<()> {
        let flag = if self.env == Environment::Mainnet { self.paths.global_kill.clone() } else { self.paths.local_kill() };
        if let Some(d) = flag.parent() {
            std::fs::create_dir_all(d)?;
        }
        std::fs::write(&flag, format!("killed at {} by atx live kill\n", Utc::now().to_rfc3339()))?;
        println!("KILL flag written: {} (new entries refused until the user deletes it)", flag.display());
        let snap = self.snapshot().await?;
        let cancelled = if self.venue.writable {
            self.cancel_bot_orders(&snap).await
        } else {
            println!("read-only venue (no two-key guard): no orders could be cancelled; there is no live bot running without the guard");
            vec![]
        };
        for c in &cancelled {
            println!("  cancelled bot order {} {} ok={}", c["symbol"], c["link"], c["ok"]);
        }
        println!("bot positions keep their exchange-side stops; use `atx live flatten-bot --yes` to close them");
        self.jw("manual_action", json!({"action": "kill", "flag": flag.display().to_string(), "cancelled": cancelled}), &self.versions(None))?;
        Ok(())
    }

    pub async fn close(&self, id: &str, reason: &str) -> anyhow::Result<()> {
        self.ensure_writes(&TestOpts::default())?;
        let _l = StateLock::acquire(&self.paths.root).await?;
        let mut st = LiveState::load(&self.paths.state())?;
        let snap = self.snapshot().await?;
        self.reconcile(&mut st, &snap, true).await?;
        let tr = st.trades.iter().find(|t| t.id == id && t.active()).cloned().with_context(|| format!("no active bot trade {id}"))?;
        let mut cancelled = Vec::new();
        for o in snap.orders.iter().filter(|o| tr.order_ids.iter().any(|x| o["orderId"] == x.as_str())) {
            let oid = o["orderId"].as_str().unwrap_or_default();
            let r = self.venue.cancel(&tr.symbol, oid).await;
            cancelled.push(json!({"order_id": oid, "ok": r.is_ok()}));
        }
        let mut sent = None;
        if let Some(p) = self.bot_position(&snap, &tr) {
            let rule = self.rules_for(&tr.symbol).await?;
            let qty = round_to_step(if tr.filled_qty > 0.0 { tr.filled_qty.min(p.size) } else { p.size }, rule.qty_step, Round::Down);
            sent = Some(self.close_qty(&tr, qty, &rule, &format!("close: {reason}")).await?);
            println!("{} {}: reduce-only close of {qty} sent", tr.id, tr.symbol);
        }
        tokio::time::sleep(Duration::from_secs(2)).await;
        let snap2 = self.snapshot().await?;
        let notes = self.reconcile(&mut st, &snap2, true).await?;
        st.save(&self.paths.state())?;
        self.jw("manual_action", json!({"action": "close", "trade_id": id, "reason": reason, "cancelled": cancelled, "close_order": sent, "notes": notes}), &tr.versions)?;
        for n in notes {
            println!("  {n}");
        }
        Ok(())
    }

    pub async fn flatten_bot(&self, yes: bool) -> anyhow::Result<()> {
        self.ensure_writes(&TestOpts::default())?;
        ensure!(yes, "flatten-bot cancels every bot order and closes every BOT position at market on {}; re-run with --yes", self.env.as_str());
        let _l = StateLock::acquire(&self.paths.root).await?;
        let mut st = LiveState::load(&self.paths.state())?;
        let snap = self.snapshot().await?;
        self.reconcile(&mut st, &snap, true).await?;
        let cancelled = self.cancel_bot_orders(&snap).await;
        let mut closes = Vec::new();
        for tr in st.trades.iter().filter(|t| t.active()) {
            let Some(p) = self.bot_position(&snap, tr) else { continue };
            let rule = self.rules_for(&tr.symbol).await?;
            let qty = round_to_step(if tr.filled_qty > 0.0 { tr.filled_qty.min(p.size) } else { p.size }, rule.qty_step, Round::Down);
            let r = self.close_qty(tr, qty, &rule, "flatten-bot").await;
            println!("  close {} {} {} -> {}", tr.symbol, p.side, qty, r.as_ref().map(|s| s.clone()).unwrap_or_else(|e| format!("FAILED {e:#}")));
            closes.push(json!({"trade_id": tr.id, "symbol": tr.symbol, "qty": qty, "ok": r.is_ok()}));
        }
        tokio::time::sleep(Duration::from_secs(2)).await;
        let snap2 = self.snapshot().await?;
        let notes = self.reconcile(&mut st, &snap2, true).await?;
        st.save(&self.paths.state())?;
        self.jw("manual_action", json!({"action": "flatten_bot", "cancelled": cancelled, "closes": closes, "notes": notes}), &self.versions(None))?;
        println!("flatten-bot: {} order(s) cancelled, {} close(s) sent", cancelled.len(), closes.len());
        for n in notes {
            println!("  {n}");
        }
        Ok(())
    }
}

struct PegEnd {
    reason: String,
}

/// The venue's leverage on the symbol must equal the one the desk set (None: the venue reports none, refused).
fn leverage_matches(want: f64, got: Option<f64>) -> anyhow::Result<()> {
    match got {
        Some(g) if (g - want).abs() < 1e-6 => Ok(()),
        Some(g) => bail!("venue leverage {g} != {want}"),
        None => bail!("venue reports no leverage for the symbol"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn paths_separate_mainnet_from_demo() {
        let m = LivePaths::new(Path::new("data"), Path::new("data/journal"), Environment::Mainnet);
        let d = LivePaths::new(Path::new("data"), Path::new("data/journal"), Environment::Demo);
        assert_eq!(m.root, Path::new("data/live"));
        assert_eq!(m.pending("x"), Path::new("data/live/pending/x.json"));
        assert_eq!(m.global_kill, Path::new("data/live/KILL"));
        assert_eq!(m.journal, Path::new("data/journal/live"));
        assert_eq!(d.root, Path::new("data/live/demo"));
        assert_eq!(d.journal, Path::new("data/journal/live_demo"));
        assert_eq!(d.global_kill, m.global_kill, "data/live/KILL stops every environment");
    }

    #[test]
    fn leverage_must_be_held_by_the_venue() {
        assert!(leverage_matches(10.0, Some(10.0)).is_ok());
        assert!(leverage_matches(10.0, Some(5.0)).is_err());
        assert!(leverage_matches(10.0, Some(25.0)).is_err());
        assert!(leverage_matches(10.0, None).is_err());
    }

    #[test]
    fn link_ids_and_codes() {
        let cfg = LiveSmallCfg::load(&Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")).unwrap();
        let id = proposal_id(1_794_758_362_159, "intent-1");
        assert_eq!(id.len(), 14);
        let l = entry_link(&cfg, &id, 12);
        assert!(l.starts_with("lsb-") && l.len() <= 36 && l.chars().all(|c| c.is_ascii_alphanumeric() || c == '-'), "{l}");
        assert!(close_link(&cfg, &id).starts_with("lsb-x-"));
        let i: OrderIntent = serde_json::from_value(json!({"intent_id": "a", "ts_ms": 0, "strategy": "s", "family": "f", "symbol": "BTCUSDT", "side": "long"})).unwrap();
        let c = confirm_code(&id, &i);
        assert_eq!(c.len(), 6);
        assert_eq!(c, confirm_code(&id, &i));
        assert_ne!(c, confirm_code("other", &i));
    }

    #[test]
    fn state_day_roll_and_gate() {
        let cfg = LiveSmallCfg::load(&Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")).unwrap();
        let mut s = LiveState { day: "2026-11-14".into(), entries_today: 7, ..Default::default() };
        let now = 1_794_758_362_159; // 2026-11-15
        s.roll_day(now);
        assert_eq!((s.day.as_str(), s.entries_today), ("2026-11-15", 0));
        s.entries_today = 3;
        s.roll_day(now + 1000);
        assert_eq!(s.entries_today, 3, "same day keeps the count");
        assert!(s.gate_manual(&cfg));
        s.confirmed_trades = 5;
        assert!(!s.gate_manual(&cfg));
        let start = now - now.rem_euclid(DAY);
        s.trades.push(TradeRec { status: "closed".into(), closed_ms: Some(start + 10), realised_usd: Some(-1.5), ..Default::default() });
        s.trades.push(TradeRec { status: "closed".into(), closed_ms: Some(start - 10), realised_usd: Some(-4.0), ..Default::default() });
        assert_eq!(s.realised_today(now), -1.5, "only trades closed since the UTC day start");
    }
}
