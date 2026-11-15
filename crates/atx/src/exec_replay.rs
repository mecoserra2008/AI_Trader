//! `atx exec-replay`: entry execution of trade signals on a replayed real Bybit ob200 book + trade tape (research only:
//! no network, no orders; H-EXE1).
//!
//!   atx exec-replay --symbol BTCUSDT --date 2026-11-10 --ob200 D_BTCUSDT_ob200.data.zip --trades BTCUSDT2026-11-10.csv.gz \
//!       --signals signals.json --out DIR [--latency-ms 150]
//!
//! signals.json: {"windows_min": [1, 5, 15], "queues": ["pessimistic", "prorata", "optimistic"],
//!                "signals": [{"id", "ts_ms", "side": "Buy" | "Sell", "qty", "stop", "take_profit"}]}
//!
//! Market: the same replay as `atx mm-replay` (`mm_replay::Market`: ob200 snapshot + deltas through
//! `at_match::Ob200Replayer` into an `at_match::OrderBook`, the public trade tape merged by time). Our orders are virtual
//! (they never change the book), so every arm of every signal sees the same market.
//!
//! Arms per signal (decision time t0 = `ts_ms`; the qty is floored to the lot step, at least one lot):
//!   market   a taker order sent at t0 and executed `latency_ms` later against the replayed book: it walks the opposite
//!            side level by level (VWAP), like a Bybit market order inside its price band.
//!   limit_N  a PostOnly limit for the whole qty joining the touch (best bid for a buy, best ask for a sell), sent at t0,
//!            live `latency_ms` later at the back of the queue (queue accounting of `mm_replay::Queue`, one run per
//!            cancel model). Re-pegged: when the touch moves away from our price (a higher best bid for a buy), an amend
//!            to the new touch is sent (one in flight at a time); it takes effect `latency_ms` later at the back of the
//!            new level (Bybit: a price amend loses priority); an amend or placement that would cross is rejected
//!            (PostOnly) and the order stays where it was (a placement is re-sent at the touch). A market moving through
//!            our price is not chased down: we are then the best price and the next print on our side fills us. The
//!            cancel is sent at t0 + N min; fills until it takes effect count. Partial fills are kept.
//! Fills of a resting order: a print at our price on our side consumes the queue from the front; a print through our
//! price fills us (up to its size); a replayed resting order on the other side at or through our price fills us (up to
//! its size). RPI prints do not fill (as in mm-replay).
//!
//! Exit (the same rule for every arm, so the arms differ only by their entry): from the last entry fill, the first print
//! at or through the stop or the target in the rest of the day's tape; stops and targets trigger market orders on
//! Bybit's last price, so the exit price is that print's price. None by the end of the tape = "open" at the last print
//! (the study continues the trade on 1h bars).
//!
//! Outputs in `--out`: `results.jsonl` (one row per signal x arm x queue model; the market arm once per signal) and
//! `stats.json` (market replay checks). Deterministic: the same inputs give the same files (`perf` excepted).

use std::fs::File;
use std::io::{BufRead, BufWriter, Write};
use std::path::{Path, PathBuf};

use anyhow::{Context, bail};
use at_match::{Grid, Side};
use at_risk::FeeSchedule;
use serde::Deserialize;
use serde_json::{Value, json};

use crate::cmd::Ctx;
use crate::lake_read::read_instruments;
use crate::mm_replay::{Market, Queue, QueueModel, Tr, open_lines, read_trades};

const DAY_MS: i64 = 86_400_000;
const MIN_MS: i64 = 60_000;

#[derive(Clone, Debug, Deserialize)]
pub struct Signal {
    pub id: String,
    pub ts_ms: i64,
    pub side: String,
    pub qty: f64,
    pub stop: f64,
    pub take_profit: f64,
}

fn default_windows() -> Vec<i64> {
    vec![1, 5, 15]
}

fn all_queues() -> Vec<QueueModel> {
    vec![QueueModel::Pessimistic, QueueModel::Prorata, QueueModel::Optimistic]
}

#[derive(Clone, Debug, Deserialize)]
pub struct SignalsFile {
    #[serde(default = "default_windows")]
    pub windows_min: Vec<i64>,
    #[serde(default = "all_queues")]
    pub queues: Vec<QueueModel>,
    pub signals: Vec<Signal>,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct EFill {
    pub ts: i64,
    pub price: f64,
    pub lots: i64,
    pub kind: &'static str,
}

/// One entry arm of one signal: the market order (window 0) or a pegged limit (window N minutes).
#[derive(Clone, Debug)]
pub struct Peg {
    pub sig: usize,
    pub side: Side,
    pub window_min: i64,
    pub model: QueueModel,
    pub t0: i64,
    pub lots: i64,
    pub left: i64,
    latency: i64,
    tick: i64,
    sent: bool,
    active: bool,
    active_ms: i64,
    q: Queue,
    amend: Option<(i64, i64)>,
    cancel_ms: i64,
    pub done: bool,
    pub fills: Vec<EFill>,
    pub amends: u32,
    pub amend_rejects: u32,
    pub postonly_rejects: u32,
    pub touch0: Option<(f64, f64)>,
    pub walked_levels: u32,
}

impl Peg {
    pub fn new(sig: usize, side: Side, window_min: i64, model: QueueModel, t0: i64, lots: i64, latency: i64) -> Self {
        Self {
            sig,
            side,
            window_min,
            model,
            t0,
            lots,
            left: lots,
            latency,
            tick: 0,
            sent: false,
            active: false,
            active_ms: 0,
            q: Queue::default(),
            amend: None,
            cancel_ms: t0 + window_min * MIN_MS + latency,
            done: lots <= 0,
            fills: vec![],
            amends: 0,
            amend_rejects: 0,
            postonly_rejects: 0,
            touch0: None,
            walked_levels: 0,
        }
    }

    pub fn is_market(&self) -> bool {
        self.window_min == 0
    }

    /// Our side's touch (best bid for a buy, best ask for a sell), in ticks.
    fn touch(&self, m: &Market) -> Option<i64> {
        match self.side {
            Side::Buy => m.book.best_bid().map(|b| b.0),
            Side::Sell => m.book.best_ask().map(|a| a.0),
        }
    }

    /// Would a PostOnly order at `tick` cross the replayed book?
    fn crosses(&self, m: &Market, tick: i64) -> bool {
        match self.side {
            Side::Buy => m.book.best_ask().is_some_and(|a| tick >= a.0),
            Side::Sell => m.book.best_bid().is_some_and(|b| tick <= b.0),
        }
    }

    fn fill(&mut self, ts: i64, price: f64, lots: i64, kind: &'static str) -> i64 {
        let f = lots.min(self.left);
        if f <= 0 {
            return 0;
        }
        self.left -= f;
        self.fills.push(EFill { ts, price, lots: f, kind });
        if self.left == 0 {
            self.done = true;
        }
        f
    }

    /// The market order: walk the opposite side of the replayed book (levels best first).
    fn execute_market(&mut self, ts: i64, m: &Market) {
        let (bids, asks) = m.levels(usize::MAX);
        let book = if self.side == Side::Buy { asks } else { bids };
        let g = *m.grid();
        for (p, q) in book {
            if self.left == 0 {
                break;
            }
            let lots = g.lots_nearest(q);
            self.walked_levels += 1;
            self.fill(ts, p, lots, "taker");
        }
        self.done = true;
    }

    /// Fire every timer strictly before `t` (send, market execution, activation, amend, cancel), in time order.
    pub fn advance(&mut self, t: i64, m: &Market) {
        loop {
            if self.done {
                return;
            }
            let mut next = i64::MAX;
            if !self.sent {
                next = self.t0;
            } else if self.is_market() {
                next = self.t0 + self.latency;
            } else {
                if !self.active {
                    next = next.min(self.active_ms);
                }
                if let Some((e, _)) = self.amend {
                    next = next.min(e);
                }
                next = next.min(self.cancel_ms);
            }
            if next >= t {
                return;
            }
            if !self.sent {
                self.sent = true;
                let (b, a) = (m.book.best_bid(), m.book.best_ask());
                let g = m.grid();
                self.touch0 = b.zip(a).map(|(b, a)| (g.price(b.0), g.price(a.0)));
                match self.touch(m) {
                    Some(tk) => self.tick = tk,
                    None => self.done = true,
                }
                self.active_ms = self.t0 + self.latency;
                continue;
            }
            if self.is_market() {
                self.execute_market(next, m);
                continue;
            }
            if next == self.cancel_ms {
                self.done = true;
                continue;
            }
            if !self.active && next == self.active_ms {
                if self.crosses(m, self.tick) {
                    // PostOnly reject at activation: re-send at the touch now
                    self.postonly_rejects += 1;
                    match self.touch(m) {
                        Some(tk) => self.tick = tk,
                        None => self.done = true,
                    }
                    self.active_ms = next + self.latency;
                    continue;
                }
                self.active = true;
                let vis = m.visible(self.side, self.tick);
                self.q = if vis { Queue { ahead: m.book.level_qty(self.side, self.tick) as f64, behind: 0.0, known: true } } else { Queue::default() };
                continue;
            }
            if let Some((e, tk)) = self.amend
                && e == next
            {
                self.amend = None;
                if self.crosses(m, tk) {
                    self.amend_rejects += 1;
                } else {
                    self.tick = tk;
                    let vis = m.visible(self.side, tk);
                    self.q = if vis { Queue { ahead: m.book.level_qty(self.side, tk) as f64, behind: 0.0, known: true } } else { Queue::default() };
                }
                continue;
            }
            // a timer equal to none of the above cannot happen; stop rather than spin
            return;
        }
    }

    /// A real trade: queue consumption at our price, fills through it.
    pub fn on_trade(&mut self, tr: &Tr, g: &Grid) {
        if self.done || !self.active || self.is_market() || tr.rpi {
            return;
        }
        let ours = match self.side {
            Side::Buy => !tr.buy,
            Side::Sell => tr.buy,
        };
        if !ours {
            return;
        }
        let through = match self.side {
            Side::Buy => tr.tick < self.tick,
            Side::Sell => tr.tick > self.tick,
        };
        let px = g.price(self.tick);
        if through {
            self.q = Queue { ahead: 0.0, behind: 0.0, known: true };
            self.fill(tr.ts, px, tr.lots, "through");
        } else if tr.tick == self.tick {
            if self.q.known {
                let f = self.q.on_trade_at(tr.lots as f64, self.left);
                if f > 0 {
                    self.fill(tr.ts, px, f, "queue");
                }
            } else {
                self.q = Queue { ahead: 0.0, behind: 0.0, known: true };
            }
        }
    }

    /// A book message: fills by a crossing resting order, queue reconciliation, and the re-peg decision.
    pub fn on_book(&mut self, m: &Market, cts: i64) {
        if self.done || !self.active || self.is_market() {
            return;
        }
        let g = *m.grid();
        let opp = self.side.opposite();
        let crossing = match (self.side, m.book.best_ask(), m.book.best_bid()) {
            (Side::Buy, Some(a), _) if a.0 <= self.tick => Some(a.0),
            (Side::Sell, _, Some(b)) if b.0 >= self.tick => Some(b.0),
            _ => None,
        };
        if let Some(best) = crossing {
            let (lo, hi) = if self.side == Side::Buy { (best, self.tick) } else { (self.tick, best) };
            let mut lots = 0;
            for t in lo..=hi {
                lots += m.book.level_qty(opp, t);
                if lots >= self.left {
                    break;
                }
            }
            self.q = Queue { ahead: 0.0, behind: 0.0, known: true };
            self.fill(cts, g.price(self.tick), lots, "cross");
            if self.done {
                return;
            }
        } else {
            let touched = m.snapshot_msg || m.changed.iter().any(|c| c.0 == self.side && c.1 == self.tick);
            if touched && m.visible(self.side, self.tick) {
                let lvl = m.book.level_qty(self.side, self.tick) as f64;
                self.q.on_level(lvl, self.model);
            }
        }
        // re-peg: the touch moved away from us (and the cancel has not been sent)
        if self.amend.is_none()
            && cts < self.cancel_ms - self.latency
            && let Some(tk) = self.touch(m)
        {
            let away = match self.side {
                Side::Buy => tk > self.tick,
                Side::Sell => tk < self.tick,
            };
            if away {
                self.amend = Some((cts + self.latency, tk));
                self.amends += 1;
            }
        }
    }

    pub fn filled_lots(&self) -> i64 {
        self.lots - self.left
    }

    pub fn vwap(&self) -> Option<f64> {
        let n: i64 = self.fills.iter().map(|f| f.lots).sum();
        (n > 0).then(|| self.fills.iter().map(|f| f.price * f.lots as f64).sum::<f64>() / n as f64)
    }
}

/// The first print at or through the stop or the target after `from` (exclusive): (kind, ts, price); "open" at the last
/// print when neither is touched.
pub fn first_exit(trades: &[Tr], from: i64, buy: bool, stop: f64, tp: f64) -> (&'static str, i64, f64) {
    let i = trades.partition_point(|t| t.ts <= from);
    for t in &trades[i..] {
        let (hit_stop, hit_tp) = if buy { (t.px <= stop, t.px >= tp) } else { (t.px >= stop, t.px <= tp) };
        if hit_stop {
            return ("stop", t.ts, t.px);
        }
        if hit_tp {
            return ("tp", t.ts, t.px);
        }
    }
    trades.last().map_or(("open", from, f64::NAN), |t| ("open", t.ts, t.px))
}

/// Runs every peg over one symbol-day: book messages from `book`, trades `trades` (time-sorted). Returns the replay checks.
pub fn run_pegs(m: &mut Market, pegs: &mut [Peg], book: &mut dyn BufRead, trades: &[Tr], end: i64) -> anyhow::Result<()> {
    let mut ti = 0usize;
    let mut line = String::new();
    let mut last_cts = i64::MIN;
    let step = |m: &mut Market, pegs: &mut [Peg], tr: &Tr| {
        for p in pegs.iter_mut() {
            p.advance(tr.ts, m);
        }
        m.on_trade(tr);
        let g = *m.grid();
        for p in pegs.iter_mut() {
            p.on_trade(tr, &g);
        }
    };
    loop {
        line.clear();
        if book.read_line(&mut line)? == 0 {
            break;
        }
        let s = line.trim();
        if s.is_empty() {
            continue;
        }
        let msg: Value = serde_json::from_str(s).with_context(|| format!("ob200 line {}", m.n_msgs + 1))?;
        let cts = msg["cts"].as_i64().or_else(|| msg["ts"].as_i64()).unwrap_or(last_cts).max(last_cts);
        last_cts = cts;
        while ti < trades.len() && trades[ti].ts <= cts {
            step(m, pegs, &trades[ti]);
            ti += 1;
        }
        for p in pegs.iter_mut() {
            p.advance(cts, m);
        }
        m.apply_book(&msg, cts)?;
        for p in pegs.iter_mut() {
            p.on_book(m, cts);
        }
    }
    while ti < trades.len() {
        step(m, pegs, &trades[ti]);
        ti += 1;
    }
    for p in pegs.iter_mut() {
        p.advance(end, m);
    }
    Ok(())
}

#[derive(clap::Args)]
pub struct ExecReplayArgs {
    #[arg(long)]
    symbol: String,
    /// UTC day (YYYY-MM-DD) of the archives.
    #[arg(long)]
    date: String,
    /// quote-saver ob200 archive (.zip, or unzipped JSON lines).
    #[arg(long)]
    ob200: PathBuf,
    /// public.bybit.com trade archive (.csv.gz or .csv).
    #[arg(long)]
    trades: PathBuf,
    /// Signals JSON (see the module docs).
    #[arg(long)]
    signals: PathBuf,
    #[arg(long)]
    out: PathBuf,
    /// Delay from sending an order (amend, cancel) to it taking effect.
    #[arg(long, default_value_t = 150)]
    latency_ms: i64,
    #[arg(long, default_value_t = 200)]
    depth: usize,
    #[arg(long)]
    fees: Option<PathBuf>,
    #[arg(long)]
    fee_class: Option<String>,
    /// Override the tick size / qty step (historical days whose grid differs from the lake's latest rules).
    #[arg(long)]
    tick: Option<f64>,
    #[arg(long)]
    qty_step: Option<f64>,
}

fn side_of(s: &str) -> anyhow::Result<Side> {
    match s {
        "Buy" | "buy" | "long" => Ok(Side::Buy),
        "Sell" | "sell" | "short" => Ok(Side::Sell),
        x => bail!("unknown side {x:?}"),
    }
}

pub fn run(ctx: &Ctx, a: ExecReplayArgs) -> anyhow::Result<()> {
    let t0 = std::time::Instant::now();
    let symbol = a.symbol.to_uppercase();
    let nd = chrono::NaiveDate::parse_from_str(&a.date, "%Y-%m-%d")?;
    let day0 = nd.and_hms_opt(0, 0, 0).context("date")?.and_utc().timestamp_millis();
    let (mut r, mut types) = read_instruments(ctx.lake.root(), "bybit", ctx.category(), day0)?;
    if !r.contains_key(&symbol) {
        (r, types) = read_instruments(ctx.lake.root(), "bybit", ctx.category(), at_core::time::now_ms())?;
    }
    let mut rules = r.remove(&symbol).with_context(|| format!("no instrument rules for {symbol}"))?;
    if let Some(t) = a.tick {
        rules.tick_size = t;
    }
    if let Some(q) = a.qty_step {
        rules.qty_step = q;
    }
    let schedule = FeeSchedule::load(&a.fees.clone().unwrap_or_else(|| ctx.s.root.join("config/fees.yaml")))?;
    let hint = a.fee_class.clone().or_else(|| types.remove(&symbol).map(|t| if t.is_empty() { "crypto".into() } else { "tradfi".into() }));
    let fees = schedule.rates(&symbol, hint.as_deref());
    let sf: SignalsFile = serde_json::from_str(&std::fs::read_to_string(&a.signals)?)?;
    std::fs::create_dir_all(&a.out)?;
    let mut m = Market::new(&rules, a.depth)?;
    let trades = read_trades(&a.trades, m.grid())?;
    let g = *m.grid();
    let mut pegs = vec![];
    for (i, s) in sf.signals.iter().enumerate() {
        anyhow::ensure!(
            s.ts_ms >= day0 && s.ts_ms + sf.windows_min.iter().max().copied().unwrap_or(0) * MIN_MS + 2 * a.latency_ms < day0 + DAY_MS,
            "signal {} outside the day",
            s.id
        );
        let side = side_of(&s.side)?;
        let lots = ((s.qty / g.qty_step() + 1e-9).floor() as i64).max(1);
        pegs.push(Peg::new(i, side, 0, QueueModel::Prorata, s.ts_ms, lots, a.latency_ms));
        for &w in &sf.windows_min {
            anyhow::ensure!(w > 0, "windows_min must be > 0");
            for &qm in &sf.queues {
                pegs.push(Peg::new(i, side, w, qm, s.ts_ms, lots, a.latency_ms));
            }
        }
    }
    let mut book = open_lines(&a.ob200)?;
    run_pegs(&mut m, &mut pegs, &mut *book, &trades, day0 + DAY_MS)?;
    write_results(&a.out.join("results.jsonl"), &m, &pegs, &sf, &trades, &g, fees.maker_bps, fees.taker_bps)?;
    let stats = json!({
        "symbol": symbol, "date": a.date, "latency_ms": a.latency_ms, "depth": a.depth, "rules": rules,
        "fees": {"maker_bps": fees.maker_bps, "taker_bps": fees.taker_bps}, "signals": sf.signals.len(), "pegs": pegs.len(),
        "market": {"book_messages": m.n_msgs, "trades": m.n_trades, "rpi_trades": m.n_rpi, "engine_trades": m.engine_trades, "engine_rejects": m.engine_rejects,
                   "crossed_messages": m.crossed_msgs, "trades_outside_touch": m.trades_outside_touch, "replayer_checks": m.checks,
                   "replayer_check_mismatches": m.check_mismatch, "mid_changes": m.mids.len()},
        "perf": {"seconds": t0.elapsed().as_secs_f64()},
    });
    std::fs::write(a.out.join("stats.json"), serde_json::to_string_pretty(&stats)?)?;
    println!(
        "{symbol} {}: {} book messages, {} trades, {} signals, {} pegs, {} filled; engine trades {} rejects {} check mismatches {}/{}; {:.1}s",
        a.date,
        m.n_msgs,
        m.n_trades,
        sf.signals.len(),
        pegs.len(),
        pegs.iter().filter(|p| p.filled_lots() > 0).count(),
        m.engine_trades,
        m.engine_rejects,
        m.check_mismatch,
        m.checks,
        t0.elapsed().as_secs_f64()
    );
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn write_results(path: &Path, m: &Market, pegs: &[Peg], sf: &SignalsFile, trades: &[Tr], g: &Grid, maker_bps: f64, taker_bps: f64) -> anyhow::Result<()> {
    let mut w = BufWriter::new(File::create(path)?);
    for p in pegs {
        let s = &sf.signals[p.sig];
        let buy = p.side == Side::Buy;
        let vwap = p.vwap();
        let first = p.fills.first().map(|f| f.ts);
        let last = p.fills.last().map(|f| f.ts);
        let exit = last.map(|t| first_exit(trades, t, buy, s.stop, s.take_profit));
        let kinds = |k: &str| p.fills.iter().filter(|f| f.kind == k).map(|f| f.lots).sum::<i64>();
        let fee_bps = if p.is_market() { taker_bps } else { maker_bps };
        let win_end = p.t0 + p.window_min.max(1) * MIN_MS;
        let row = json!({
            "signal": s.id, "arm": if p.is_market() { "market" } else { "limit" }, "window_min": p.window_min,
            "queue": if p.is_market() { "-" } else { p.model.as_str() }, "side": if buy { "Buy" } else { "Sell" }, "t0": p.t0,
            "lots": p.lots, "filled_lots": p.filled_lots(), "qty": g.qty(p.lots), "filled_qty": g.qty(p.filled_lots()), "vwap": vwap,
            "entry_fee_bps": fee_bps, "first_fill_ms": first, "last_fill_ms": last, "n_fills": p.fills.len(),
            "lots_queue": kinds("queue"), "lots_through": kinds("through"), "lots_cross": kinds("cross"), "lots_taker": kinds("taker"),
            "walked_levels": p.walked_levels, "amends": p.amends, "amend_rejects": p.amend_rejects, "postonly_rejects": p.postonly_rejects,
            "bid0": p.touch0.map(|x| x.0), "ask0": p.touch0.map(|x| x.1), "mid0": m.mid_at(p.t0), "mid_window_end": m.mid_at(win_end),
            "mid_fill_1s": first.and_then(|t| m.mid_at(t + 1_000)), "mid_fill_60s": first.and_then(|t| m.mid_at(t + 60_000)), "mid_fill": first.and_then(|t| m.mid_at(t)),
            "exit_kind": exit.map(|e| e.0), "exit_ts": exit.map(|e| e.1), "exit_px": exit.map(|e| e.2),
            "stop": s.stop, "take_profit": s.take_profit,
        });
        writeln!(w, "{}", serde_json::to_string(&row)?)?;
    }
    w.flush()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use at_core::types::InstrumentRules;
    use std::io::BufReader;

    fn rules() -> InstrumentRules {
        InstrumentRules { symbol: "TRXUSDT".into(), tick_size: 0.0001, qty_step: 1.0, min_qty: 1.0, max_qty: 1e6, max_mkt_qty: 1e5, min_notional: 5.0, max_leverage: 50.0 }
    }

    fn lv(p: f64, q: f64) -> Value {
        json!([format!("{p:.4}"), format!("{q}")])
    }

    fn msg(kind: &str, cts: i64, b: Vec<Value>, a: Vec<Value>) -> String {
        json!({"topic": "orderbook.200.TRXUSDT", "type": kind, "ts": cts, "cts": cts, "data": {"s": "TRXUSDT", "b": b, "a": a}}).to_string()
    }

    fn trade(ts: i64, px: f64, lots: i64, buy: bool) -> Tr {
        Tr { ts, px, tick: (px / 0.0001).round() as i64, lots, buy, rpi: false }
    }

    fn run(lines: &[String], trades: &[Tr], pegs: &mut [Peg]) -> Market {
        let mut m = Market::new(&rules(), 200).unwrap();
        let text = lines.join("\n");
        let mut rd = BufReader::new(text.as_bytes());
        run_pegs(&mut m, pegs, &mut rd, trades, DAY_MS).unwrap();
        m
    }

    fn snap() -> String {
        msg("snapshot", 0, vec![lv(0.2999, 100.0), lv(0.2998, 100.0)], vec![lv(0.3001, 30.0), lv(0.3002, 100.0)])
    }

    #[test]
    fn market_walks_the_book_after_latency() {
        let lines = vec![snap(), msg("delta", 1_100, vec![], vec![lv(0.3001, 10.0)])];
        let mut pegs = vec![Peg::new(0, Side::Buy, 0, QueueModel::Prorata, 1_000, 50, 150)];
        run(&lines, &[], &mut pegs);
        // at 1_150 the ask level 0.3001 holds 10 (the delta at 1_100), then 40 from 0.3002
        assert_eq!(pegs[0].filled_lots(), 50);
        assert_eq!(pegs[0].fills.len(), 2);
        assert_eq!(pegs[0].fills[0].ts, 1_150);
        assert!((pegs[0].vwap().unwrap() - (10.0 * 0.3001 + 40.0 * 0.3002) / 50.0).abs() < 1e-12);
    }

    #[test]
    fn limit_joins_the_back_and_fills_from_the_front() {
        let lines = vec![snap(), msg("delta", 5_000, vec![], vec![lv(0.3001, 31.0)])];
        // sells at our bid 0.2999: 100 ahead of us (joined at activation), then 20 for us
        let trades = vec![trade(2_000, 0.2999, 60, false), trade(3_000, 0.2999, 60, false), trade(4_000, 0.2999, 30, false)];
        let mut pegs: Vec<Peg> = [QueueModel::Pessimistic, QueueModel::Prorata, QueueModel::Optimistic].iter().map(|&qm| Peg::new(0, Side::Buy, 1, qm, 1_000, 20, 150)).collect();
        run(&lines, &trades, &mut pegs);
        for p in &pegs {
            assert_eq!(p.filled_lots(), 20, "{:?}", p.model);
            assert_eq!(p.fills[0].kind, "queue");
            assert_eq!(p.fills[0].ts, 3_000);
            assert!((p.vwap().unwrap() - 0.2999).abs() < 1e-12);
        }
    }

    #[test]
    fn repeg_follows_the_touch_and_cancel_ends_the_window() {
        // the bid moves up at 2_000 to 0.3000; our amend takes effect at 2_150 at the back of 0.3000 (50 ahead)
        let lines = vec![snap(), msg("delta", 2_000, vec![lv(0.3000, 50.0)], vec![]), msg("delta", 70_000, vec![lv(0.3000, 10.0)], vec![])];
        let trades = vec![trade(3_000, 0.3000, 60, false), trade(62_000, 0.3000, 100, false)];
        let mut pegs = vec![Peg::new(0, Side::Buy, 1, QueueModel::Prorata, 1_000, 20, 150)];
        run(&lines, &trades, &mut pegs);
        let p = &pegs[0];
        assert_eq!(p.amends, 1);
        // 60 sold at 0.3000: 50 ahead, then 10 of our 20; the print at 62_000 is after the cancel (61_150)
        assert_eq!(p.filled_lots(), 10);
        assert_eq!(p.fills[0].ts, 3_000);
        assert!((p.fills[0].price - 0.3000).abs() < 1e-12);
        assert!(p.done);
    }

    #[test]
    fn through_print_fills_and_pessimistic_never_beats_optimistic() {
        // others cancel at our level (100 -> 40), then a sell prints below our bid
        let lines = vec![snap(), msg("delta", 2_000, vec![lv(0.2999, 40.0)], vec![])];
        let trades = vec![trade(3_000, 0.2999, 45, false), trade(4_000, 0.2998, 5, false)];
        let mut pegs: Vec<Peg> = [QueueModel::Pessimistic, QueueModel::Prorata, QueueModel::Optimistic].iter().map(|&qm| Peg::new(0, Side::Buy, 1, qm, 1_000, 10, 150)).collect();
        run(&lines, &trades, &mut pegs);
        let f: Vec<i64> = pegs.iter().map(|p| p.filled_lots()).collect();
        // pessimistic: 60 cancels from behind (none) then ahead -> 40 ahead, 45 at our price: 5 for us, + 5 through
        assert_eq!(f[0], 10);
        // optimistic: cancels from ahead first -> 40 ahead -> same here (no lots behind us)
        assert_eq!(f[2], 10);
        assert!(f[0] <= f[1] && f[1] <= f[2]);
        assert!(pegs[0].fills.iter().any(|x| x.kind == "through"));
    }

    #[test]
    fn exit_is_the_first_print_through_stop_or_target() {
        let tr = vec![trade(1, 0.3000, 1, true), trade(2, 0.3050, 1, true), trade(3, 0.2900, 1, false), trade(4, 0.3200, 1, true)];
        assert_eq!(first_exit(&tr, 1, true, 0.2950, 0.3100), ("stop", 3, 0.2900));
        assert_eq!(first_exit(&tr, 3, true, 0.2950, 0.3100), ("tp", 4, 0.3200));
        assert_eq!(first_exit(&tr, 1, false, 0.3100, 0.2950).0, "tp");
        assert_eq!(first_exit(&tr, 4, true, 0.2, 0.4), ("open", 4, 0.3200));
    }
}
