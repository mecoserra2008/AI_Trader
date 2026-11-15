//! `atx mm-replay`: a market-making backtest on replayed real Bybit order books (research only: no network, no orders).
//!
//!   atx mm-replay --symbol TRXUSDT --date 2026-11-10 --ob200 D_TRXUSDT_ob200.data.zip --trades TRXUSDT2026-11-10.csv.gz \
//!       --variants variants.json --out DIR [--warmup-bars prev_bars_1m.csv,..] [--latency-ms 150] [--forecast sigma.csv]
//!   atx mm-replay --symbol TRXUSDT --trades TRXUSDT2026-11-09.csv.gz --out DIR --bars-only      (1m bars for warm-up)
//!
//! Market: the ob200 snapshot + deltas go through `at_match::Ob200Replayer` into an `at_match::OrderBook` (the real
//! book, every level), and the public trade tape (public.bybit.com) is merged by time: a trade with time <= a book
//! message's matching-engine time `cts` is applied before that message.
//!
//! Quoters: each variant (config/mm*.yaml + overrides) x queue model runs the live quoting code of `atx mm`
//! (`mm::symbol_inputs`, `mm::target_quotes`, `mm::needs_replace`, the same sigma, flow, micro-sigma, toxicity, shock
//! and trend-guard steps) on its own clock: one decision every `refresh_s` (or `tick_ms`). Our orders are virtual and
//! never change the book (minimum lots: our impact on others is ignored), so every quoter sees the same market.
//!
//! Queue accounting of one resting order (others' lots ahead of us and behind us at our price):
//!   - an order becomes live `latency_ms` after it is sent (its PostOnly check runs then, against the real book); it joins
//!     the back of the queue: ahead = the level's size at that moment, behind = 0. A cancel takes effect `latency_ms`
//!     after it is sent; fills before then count.
//!   - a real trade at our price, on our side, consumes the queue from the front: ahead first, then us, then behind.
//!     A trade through our price (a sell print below our bid, a buy print above our ask) fills us (up to its size).
//!   - when a book message changes our level, growth joins behind us; a decrease the tape did not explain is cancels:
//!     FIFO-pessimistic takes them from behind us first (ahead last), pro-rata (primary) in proportion to ahead / behind,
//!     LIFO-optimistic from ahead of us first.
//!   - a real order resting on the other side at or through our price (the replayed book crosses our virtual quote)
//!     would have matched us: we are filled at our price, up to the crossing size.
//!   - a level outside the 200-level window is unknown: when it enters the window, all of its size counts as ahead.
//!   - Retail-price-improvement prints (RPI = 1) and prints off the book's tick grid trade against hidden quotes, not the
//!     visible queue: they feed the flow and bars but do not move or fill our queue unless `--fill-rpi`.
//!
//! Not modelled (stated in the report): portfolio coupling across symbols (gross / net caps see only this symbol),
//! structural zones (`levels_file`), the live link's stale tops and per-tick work time (use `tick_ms`), funding,
//! Bybit internals (hidden priority, internal latency).
//!
//! Outputs in `--out`: `fills.jsonl` (every fill with mids 1 / 10 / 60 s later), `quotes.json` (order funnel per
//! quoter), `stats.json` (per quoter P&L decomposition: spread captured, adverse selection at 1 / 10 / 60 s,
//! inventory P&L, fees; plus market replay checks), `bars_1m.csv` (the day's 1m last-price bars from the tape).
//! Deterministic: the same inputs give the same files (`perf` in stats.json is the only exception).

use std::collections::{BTreeMap, VecDeque};
use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};

use anyhow::{Context, bail};
use at_core::types::InstrumentRules;
use at_match::{Grid, InputEvent, NewOrder, Ob200Replayer, OrderBook, Side, Tif};
use at_risk::{FeeRates, FeeSchedule};
use flate2::read::{DeflateDecoder, MultiGzDecoder};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use crate::cmd::Ctx;
use crate::lake_read::read_instruments;
use crate::mm::{self, MmCfg, Others, Quote, Resting, SymState};

const DAY_MS: i64 = 86_400_000;
const MIN_MS: i64 = 60_000;
const MARKOUTS_S: [i64; 3] = [1, 10, 60];

/// (price, qty) levels, best first.
type Levels2 = Vec<(f64, f64)>;

// ---------------------------------------------------------------- queue accounting

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum QueueModel {
    /// FIFO-pessimistic: unexplained decreases come from behind us first (from ahead of us last).
    Pessimistic,
    /// Pro-rata (primary): in proportion to the lots ahead of and behind us.
    Prorata,
    /// LIFO-optimistic: from ahead of us first.
    Optimistic,
}

impl QueueModel {
    pub fn as_str(self) -> &'static str {
        match self {
            QueueModel::Pessimistic => "pessimistic",
            QueueModel::Prorata => "prorata",
            QueueModel::Optimistic => "optimistic",
        }
    }
}

/// Others' lots ahead of and behind one of our resting orders at its price. `known` = false while the level is outside
/// the replayed depth window.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Queue {
    pub ahead: f64,
    pub behind: f64,
    pub known: bool,
}

impl Queue {
    /// A trade of `lots` at our price on our side: consumes ahead, then us (returns our filled lots, at most `ours`, whole
    /// lots), then behind.
    pub fn on_trade_at(&mut self, lots: f64, ours: i64) -> i64 {
        let a = self.ahead.min(lots);
        self.ahead -= a;
        let mut x = lots - a;
        let f = ((x + 1e-9).floor() as i64).clamp(0, ours.max(0));
        x -= f as f64;
        self.behind = (self.behind - x).max(0.0);
        f
    }

    /// The level now shows `level` lots of others: growth joins behind us; a decrease is cancels split by `model`.
    pub fn on_level(&mut self, level: f64, model: QueueModel) {
        if !self.known {
            *self = Queue { ahead: level.max(0.0), behind: 0.0, known: true };
            return;
        }
        let cur = self.ahead + self.behind;
        let d = level - cur;
        if d >= 0.0 {
            self.behind += d;
            return;
        }
        let c = -d;
        let from_ahead = match model {
            QueueModel::Pessimistic => (c - self.behind).max(0.0),
            QueueModel::Optimistic => c.min(self.ahead),
            QueueModel::Prorata => {
                if cur > 0.0 {
                    c * self.ahead / cur
                } else {
                    0.0
                }
            }
        };
        self.ahead = (self.ahead - from_ahead).max(0.0);
        self.behind = (self.behind - (c - from_ahead)).max(0.0);
    }
}

// ---------------------------------------------------------------- inputs

/// One public trade (public.bybit.com archive): ms, price, lots, aggressor side, RPI flag.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Tr {
    pub ts: i64,
    pub px: f64,
    pub tick: i64,
    pub lots: i64,
    pub buy: bool,
    pub rpi: bool,
}

/// Streams the first member of a .zip (deflate or stored), or a plain file.
pub(crate) fn open_lines(path: &Path) -> anyhow::Result<Box<dyn BufRead>> {
    let f = File::open(path).with_context(|| path.display().to_string())?;
    let mut r = BufReader::with_capacity(1 << 20, f);
    if path.extension().is_some_and(|e| e.eq_ignore_ascii_case("zip")) {
        let mut h = [0u8; 30];
        r.read_exact(&mut h)?;
        anyhow::ensure!(&h[0..4] == b"PK\x03\x04", "{}: not a zip local file header", path.display());
        let method = u16::from_le_bytes([h[8], h[9]]);
        let csize = u32::from_le_bytes([h[18], h[19], h[20], h[21]]) as u64;
        let skip = u16::from_le_bytes([h[26], h[27]]) as usize + u16::from_le_bytes([h[28], h[29]]) as usize;
        let mut buf = vec![0u8; skip];
        r.read_exact(&mut buf)?;
        return match method {
            8 => Ok(Box::new(BufReader::with_capacity(1 << 20, DeflateDecoder::new(r)))),
            0 if csize != 0xFFFF_FFFF && csize > 0 => Ok(Box::new(BufReader::new(r.take(csize)))),
            m => bail!("{}: unsupported zip member (method {m})", path.display()),
        };
    }
    if path.extension().is_some_and(|e| e.eq_ignore_ascii_case("gz")) {
        return Ok(Box::new(BufReader::with_capacity(1 << 20, MultiGzDecoder::new(r))));
    }
    Ok(Box::new(r))
}

/// Reads a public.bybit.com trade archive (csv.gz): ms = round(timestamp s x 1000), stable-sorted by ms.
pub fn read_trades(path: &Path, grid: &Grid) -> anyhow::Result<Vec<Tr>> {
    let mut rdr = csv::Reader::from_reader(open_lines(path)?);
    let h = rdr.headers()?.clone();
    let col = |n: &str| h.iter().position(|x| x == n);
    let (c_ts, c_side, c_size, c_px) =
        (col("timestamp").context("no timestamp")?, col("side").context("no side")?, col("size").context("no size")?, col("price").context("no price")?);
    let c_rpi = col("RPI");
    let mut out = Vec::new();
    let mut rec = csv::StringRecord::new();
    while rdr.read_record(&mut rec)? {
        let secs: f64 = rec[c_ts].parse()?;
        let px: f64 = rec[c_px].parse()?;
        let size: f64 = rec[c_size].parse()?;
        let buy = match &rec[c_side] {
            "Buy" => true,
            "Sell" => false,
            s => bail!("unknown side {s:?}"),
        };
        // A print off the book's tick grid (sub-tick prices, seen on TRXUSDT during bursts while every ob200 level stays
        // on the grid) traded against hidden liquidity: it is handled like an RPI print (no effect on the visible queue).
        let (tick, off_grid) = match grid.ticks(px) {
            Some(t) => (t, false),
            None => ((px / grid.tick_size()).round() as i64, true),
        };
        let lots = grid.lots_nearest(size);
        let rpi = off_grid || c_rpi.is_some_and(|c| rec.get(c).is_some_and(|v| v.trim() == "1" || v.trim().eq_ignore_ascii_case("true")));
        out.push(Tr { ts: (secs * 1000.0).round() as i64, px, tick, lots, buy, rpi });
    }
    out.sort_by_key(|t| t.ts);
    Ok(out)
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
pub struct Ohlc {
    pub o: f64,
    pub h: f64,
    pub l: f64,
    pub c: f64,
    pub v: f64,
    pub n: u64,
}

fn read_bars(path: &Path, bars: &mut BTreeMap<i64, Ohlc>) -> anyhow::Result<()> {
    let mut rdr = csv::Reader::from_reader(open_lines(path)?);
    let mut rec = csv::StringRecord::new();
    while rdr.read_record(&mut rec)? {
        let p = |i: usize| -> anyhow::Result<f64> { Ok(rec[i].parse::<f64>()?) };
        bars.insert(rec[0].parse()?, Ohlc { o: p(1)?, h: p(2)?, l: p(3)?, c: p(4)?, v: p(5)?, n: rec[6].parse()? });
    }
    Ok(())
}

fn write_bars(path: &Path, bars: &BTreeMap<i64, Ohlc>, from: i64, to: i64) -> anyhow::Result<()> {
    let mut w = BufWriter::new(File::create(path)?);
    writeln!(w, "minute_ms,open,high,low,close,volume,n")?;
    for (m, b) in bars.range(from..to) {
        writeln!(w, "{m},{},{},{},{},{},{}", b.o, b.h, b.l, b.c, b.v, b.n)?;
    }
    Ok(())
}

/// Piecewise-constant 1h sigma forecast (fraction): `valid_from_ms,sigma_1h` rows.
fn read_forecast(path: &Path) -> anyhow::Result<Vec<(i64, f64)>> {
    let mut rdr = csv::Reader::from_reader(open_lines(path)?);
    let mut v = vec![];
    for r in rdr.records() {
        let r = r?;
        v.push((r[0].parse::<i64>()?, r[1].parse::<f64>()?));
    }
    v.sort_by_key(|x| x.0);
    Ok(v)
}

/// Regime gate rows (`valid_from_ms,on`; on = 1 / true), sorted by time.
fn read_gate(path: &Path) -> anyhow::Result<Vec<(i64, bool)>> {
    let mut rdr = csv::Reader::from_reader(open_lines(path)?);
    let mut v = vec![];
    for r in rdr.records() {
        let r = r?;
        let on = r[1].trim();
        v.push((r[0].parse::<i64>()?, on == "1" || on.eq_ignore_ascii_case("true")));
    }
    v.sort_by_key(|x| x.0);
    Ok(v)
}

// ---------------------------------------------------------------- market

pub struct Market {
    pub book: OrderBook,
    grid: Grid,
    rp: Ob200Replayer,
    evs: Vec<InputEvent>,
    out: Vec<at_match::Event>,
    pub depth: usize,
    recent: VecDeque<Tr>,
    bars: BTreeMap<i64, Ohlc>,
    /// (time, mid) at every change of the touch.
    pub mids: Vec<(i64, f64)>,
    last_top: (i64, i64),
    pub n_msgs: u64,
    pub n_trades: u64,
    pub n_rpi: u64,
    pub engine_trades: u64,
    pub engine_rejects: u64,
    pub crossed_msgs: u64,
    pub trades_outside_touch: u64,
    pub checks: u64,
    pub check_mismatch: u64,
    pub changed: Vec<(Side, i64)>,
    pub snapshot_msg: bool,
}

impl Market {
    pub fn new(rules: &InstrumentRules, depth: usize) -> anyhow::Result<Self> {
        // The market book holds others' liquidity only: no min qty / notional / max qty filters.
        let mut r = rules.clone();
        r.min_qty = r.qty_step;
        r.min_notional = 0.0;
        r.max_qty = 0.0;
        r.max_mkt_qty = 0.0;
        let book = OrderBook::new(&r, FeeRates { maker_bps: 0.0, taker_bps: 0.0, slippage_bps: 0.0 }).map_err(|e| anyhow::anyhow!(e))?;
        let grid = *book.grid();
        Ok(Self {
            book,
            grid,
            rp: Ob200Replayer::new(grid, 1, 1),
            evs: vec![],
            out: vec![],
            depth,
            recent: VecDeque::with_capacity(1001),
            bars: BTreeMap::new(),
            mids: vec![],
            last_top: (i64::MIN, i64::MIN),
            n_msgs: 0,
            n_trades: 0,
            n_rpi: 0,
            engine_trades: 0,
            engine_rejects: 0,
            crossed_msgs: 0,
            trades_outside_touch: 0,
            checks: 0,
            check_mismatch: 0,
            changed: vec![],
            snapshot_msg: false,
        })
    }

    pub fn grid(&self) -> &Grid {
        &self.grid
    }

    pub fn top(&self) -> Option<(i64, i64)> {
        match (self.book.best_bid(), self.book.best_ask()) {
            (Some(b), Some(a)) if a.0 > b.0 => Some((b.0, a.0)),
            _ => None,
        }
    }

    pub fn mid(&self) -> Option<f64> {
        self.top().map(|(b, a)| 0.5 * (self.grid.price(b) + self.grid.price(a)))
    }

    /// Is `tick` on `side` inside the replayed depth window (or is the whole side visible)?
    pub fn visible(&self, side: Side, tick: i64) -> bool {
        let n = match side {
            Side::Buy => self.book.n_levels().0,
            Side::Sell => self.book.n_levels().1,
        };
        if n < self.depth {
            return true;
        }
        match (side, self.book.worst_price(side)) {
            (Side::Buy, Some(w)) => tick >= w,
            (Side::Sell, Some(w)) => tick <= w,
            _ => true,
        }
    }

    /// Apply one ob200 message at `cts`; `changed` lists the levels it touched.
    pub fn apply_book(&mut self, msg: &Value, cts: i64) -> anyhow::Result<()> {
        self.n_msgs += 1;
        self.changed.clear();
        self.snapshot_msg = msg["type"] == "snapshot";
        for (side, key) in [(Side::Buy, "b"), (Side::Sell, "a")] {
            for lv in msg["data"][key].as_array().map(|a| a.as_slice()).unwrap_or(&[]) {
                let p = match &lv[0] {
                    Value::String(s) => s.parse::<f64>().ok(),
                    v => v.as_f64(),
                };
                if let Some(t) = p.and_then(|p| self.grid.ticks(p)) {
                    self.changed.push((side, t));
                }
            }
        }
        self.evs.clear();
        self.rp.apply(msg, cts, &mut self.evs).map_err(|e| anyhow::anyhow!("ob200 message {}: {e}", self.n_msgs))?;
        for e in &self.evs {
            self.out.clear();
            match e {
                InputEvent::New { ts, id, owner, side, price, qty, .. } => {
                    self.book.submit_into(&NewOrder::limit(*ts, *id, *owner as u32, *side, Tif::GTC, price.unwrap_or(0.0), *qty), &mut self.out)
                }
                InputEvent::Cancel { ts, id } => self.book.cancel_into(*ts, *id, &mut self.out),
                InputEvent::Amend { ts, id, price, qty } => self.book.amend_into(*ts, *id, *price, *qty, &mut self.out),
            }
            for o in &self.out {
                match o {
                    at_match::Event::Trade(_) => self.engine_trades += 1,
                    at_match::Event::Reject(_) => self.engine_rejects += 1,
                    _ => {}
                }
            }
        }
        if let (Some(b), Some(a)) = (self.book.best_bid(), self.book.best_ask())
            && a.0 <= b.0
        {
            self.crossed_msgs += 1;
        }
        if let Some((b, a)) = self.top()
            && (b, a) != self.last_top
        {
            self.last_top = (b, a);
            self.mids.push((cts, 0.5 * (self.grid.price(b) + self.grid.price(a))));
        }
        if self.n_msgs % 20_000 == 1 {
            self.checks += 1;
            if self.rp.book() != self.book.snapshot(usize::MAX) {
                self.check_mismatch += 1;
            }
        }
        Ok(())
    }

    pub fn on_trade(&mut self, t: &Tr) {
        self.n_trades += 1;
        self.n_rpi += t.rpi as u64;
        if !t.rpi
            && let Some((b, a)) = self.top()
            && (t.tick < b || t.tick > a)
        {
            self.trades_outside_touch += 1;
        }
        if self.recent.len() == 1000 {
            self.recent.pop_front();
        }
        self.recent.push_back(*t);
        let m = t.ts.div_euclid(MIN_MS) * MIN_MS;
        let q = self.grid.qty(t.lots);
        let b = self.bars.entry(m).or_insert(Ohlc { o: t.px, h: t.px, l: t.px, c: t.px, v: 0.0, n: 0 });
        b.h = b.h.max(t.px);
        b.l = b.l.min(t.px);
        b.c = t.px;
        b.v += q;
        b.n += 1;
    }

    /// Closes of the CLOSED 1m bars opening in [from, now - 1 min] (Bybit klines as `atx mm` reads them); minutes
    /// without trades repeat the previous close (venue bars do).
    pub fn closes(&self, from: i64, now: i64) -> Vec<f64> {
        let (a, b) = ((from + MIN_MS - 1).div_euclid(MIN_MS) * MIN_MS, (now - MIN_MS).div_euclid(MIN_MS) * MIN_MS);
        let mut last = self.bars.range(..a).next_back().map(|x| x.1.c);
        let mut out = vec![];
        let mut it = self.bars.range(a..=b).peekable();
        let mut m = a;
        while m <= b {
            if let Some((k, bar)) = it.peek()
                && **k == m
            {
                last = Some(bar.c);
                it.next();
            }
            if let Some(c) = last {
                out.push(c);
            }
            m += MIN_MS;
        }
        out
    }

    /// Highs and lows of the CLOSED 15m bars opening in [from, now - 15 min].
    pub fn bars15(&self, from: i64, now: i64) -> (Vec<f64>, Vec<f64>) {
        let q = 15 * MIN_MS;
        let (a, b) = ((from + q - 1).div_euclid(q) * q, (now - q).div_euclid(q) * q);
        let (mut h, mut l) = (vec![], vec![]);
        let mut s = a;
        while s <= b {
            let mut hh = f64::MIN;
            let mut ll = f64::MAX;
            for (_, x) in self.bars.range(s..s + q) {
                hh = hh.max(x.h);
                ll = ll.min(x.l);
            }
            if hh > f64::MIN {
                h.push(hh);
                l.push(ll);
            }
            s += q;
        }
        (h, l)
    }

    /// The last `n` public trades at or before `now` (as the recent-trade endpoint returns them).
    pub fn recent(&self, n: usize, now: i64) -> impl Iterator<Item = &Tr> {
        let k = self.recent.iter().rposition(|t| t.ts <= now).map_or(0, |i| i + 1);
        self.recent.range(k.saturating_sub(n)..k)
    }

    pub fn levels(&self, n: usize) -> (Levels2, Levels2) {
        let l2 = self.book.snapshot(n);
        let g = &self.grid;
        (l2.bids.iter().map(|(p, q)| (g.price(*p), g.qty(*q))).collect(), l2.asks.iter().map(|(p, q)| (g.price(*p), g.qty(*q))).collect())
    }

    /// Mid at or before `t` (the first mid when `t` is earlier).
    pub fn mid_at(&self, t: i64) -> Option<f64> {
        let i = self.mids.partition_point(|x| x.0 <= t);
        if i == 0 { self.mids.first().map(|x| x.1) } else { Some(self.mids[i - 1].1) }
    }
}

// ---------------------------------------------------------------- variants

#[derive(Clone, Debug, Deserialize, Serialize, Default, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum SigmaMode {
    /// The live rule: micro sigma when `micro_interval_s` > 0, else max(slow, fast) 1h sigma.
    #[default]
    Config,
    /// A constant quote sigma of `fixed_bps`.
    Fixed,
    /// `forecast_scale` x the external 1h forecast (`--forecast`); the config rule while none is known.
    Forecast,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct VariantSpec {
    pub name: String,
    /// config/mm*.yaml (relative to the repo root).
    pub config: String,
    /// MmCfg fields to override (merged into the YAML).
    #[serde(default)]
    pub overrides: serde_json::Map<String, Value>,
    #[serde(default)]
    pub sigma_mode: SigmaMode,
    #[serde(default)]
    pub fixed_bps: f64,
    #[serde(default = "one")]
    pub forecast_scale: f64,
    /// Decision period (ms); 0 = refresh_s.
    #[serde(default)]
    pub tick_ms: i64,
    /// Emulate the live flow-refresh rotation: this symbol's flow is refreshed only on ticks where it is inside the
    /// rotating batch of `flow_batch` of `cfg.symbols` (true), or every tick (false).
    #[serde(default)]
    pub flow_rotation: bool,
    #[serde(default = "all_queues")]
    pub queues: Vec<QueueModel>,
    /// Wind-down deadline (ms): the last `exit_window_min` add no inventory. None = no wind-down.
    #[serde(default)]
    pub deadline_ms: Option<i64>,
    /// Regime gate (H-MMR3): CSV `valid_from_ms,on` (piecewise constant, 1 = quote; before the first row the gate is
    /// off), relative to the repo root. While the gate is off the quoter adds no inventory (exit-only quotes, the live
    /// wind-down path). None = always on.
    #[serde(default)]
    pub gate: Option<String>,
    /// With a gate: when it turns off with inventory, cancel and close at market (taker) instead of exit-only quotes.
    #[serde(default)]
    pub gate_flatten: bool,
}

fn one() -> f64 {
    1.0
}

fn all_queues() -> Vec<QueueModel> {
    vec![QueueModel::Pessimistic, QueueModel::Prorata, QueueModel::Optimistic]
}

#[derive(Clone, Debug, Deserialize)]
pub struct VariantsFile {
    pub variants: Vec<VariantSpec>,
}

pub fn load_cfg(root: &Path, spec: &VariantSpec) -> anyhow::Result<MmCfg> {
    let p = root.join(&spec.config);
    let mut v: Value = serde_yaml::from_str(&std::fs::read_to_string(&p).with_context(|| p.display().to_string())?)?;
    let m = v.as_object_mut().context("config is not a mapping")?;
    for (k, x) in &spec.overrides {
        m.insert(k.clone(), x.clone());
    }
    serde_json::from_value(v).with_context(|| format!("{}: {} with overrides", spec.name, p.display()))
}

// ---------------------------------------------------------------- quoter

#[derive(Clone, Debug)]
struct SimOrder {
    side: Side,
    price: f64,
    tick: i64,
    leaves: i64,
    reduce_only: bool,
    active_ms: i64,
    active: bool,
    cancel_ms: Option<i64>,
    updated_ms: i64,
    q: Queue,
    ahead0: f64,
    closed: bool,
    filled: i64,
}

#[derive(Clone, Debug, Serialize)]
pub struct Fill {
    pub ts: i64,
    pub side: &'static str,
    pub price: f64,
    pub qty: f64,
    pub maker: bool,
    pub kind: &'static str,
    pub reduce_only: bool,
    pub fee: f64,
    pub mid: f64,
    pub ahead0_lots: f64,
    pub pos_after: f64,
}

#[derive(Clone, Debug, Default, Serialize)]
pub struct Funnel {
    pub ticks: u64,
    pub placed: u64,
    pub activated: u64,
    pub postonly_rejects: u64,
    pub cancels: u64,
    pub ro_trimmed: u64,
    pub fills_queue: u64,
    pub fills_through: u64,
    pub fills_cross: u64,
    pub fills_stop: u64,
    pub filled_orders: u64,
    pub ahead0_lots_sum: f64,
    pub ahead0_unknown: u64,
    pub dist_bps_sum: f64,
    pub live_ms_bid: i64,
    pub live_ms_ask: i64,
    pub shock_pauses: u64,
    pub guard_ticks: u64,
    pub forecast_missing_ticks: u64,
    pub gate_off_ticks: u64,
    pub gate_flattens: u64,
    pub fills_gate: u64,
}

pub struct Quoter {
    pub name: String,
    pub model: QueueModel,
    cfg: std::rc::Rc<MmCfg>,
    spec: VariantSpec,
    symbol: String,
    rules: InstrumentRules,
    fees: FeeRates,
    latency: i64,
    tick_ms: i64,
    start: i64,
    end: i64,
    fill_rpi: bool,
    st: SymState,
    pos: i64,
    avg: f64,
    pub cash: f64,
    pub fee_paid: f64,
    orders: Vec<SimOrder>,
    next_tick: i64,
    sigma_at: i64,
    tick_n: u64,
    markouts: Vec<(f64, f64, i64)>,
    pub stopped: Option<(i64, &'static str)>,
    flatten_ms: Option<i64>,
    pub fills: Vec<Fill>,
    pub funnel: Funnel,
    max_inv_usd: f64,
    sym_index: Option<usize>,
    gate: Option<std::rc::Rc<Vec<(i64, bool)>>>,
    gate_was_on: bool,
}

fn side_str(s: Side) -> &'static str {
    match s {
        Side::Buy => "Buy",
        Side::Sell => "Sell",
    }
}

impl Quoter {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        cfg: MmCfg,
        spec: VariantSpec,
        model: QueueModel,
        symbol: &str,
        rules: &InstrumentRules,
        fees: FeeRates,
        latency: i64,
        start: i64,
        end: i64,
        fill_rpi: bool,
    ) -> Self {
        let tick_ms = if spec.tick_ms > 0 { spec.tick_ms } else { cfg.refresh_s.max(1) as i64 * 1000 };
        let sym_index = cfg.symbols.iter().position(|s| s.eq_ignore_ascii_case(symbol));
        Self {
            name: spec.name.clone(),
            model,
            cfg: std::rc::Rc::new(cfg),
            spec,
            symbol: symbol.to_string(),
            rules: rules.clone(),
            fees,
            latency,
            tick_ms,
            start,
            end,
            fill_rpi,
            st: SymState::default(),
            pos: 0,
            avg: 0.0,
            cash: 0.0,
            fee_paid: 0.0,
            orders: vec![],
            next_tick: start,
            sigma_at: 0,
            tick_n: 0,
            markouts: vec![],
            stopped: None,
            flatten_ms: None,
            fills: vec![],
            funnel: Funnel::default(),
            max_inv_usd: 0.0,
            sym_index,
            gate: None,
            gate_was_on: false,
        }
    }

    /// Attach the regime gate rows (`VariantSpec::gate`, read once per file by the driver).
    pub fn set_gate(&mut self, rows: std::rc::Rc<Vec<(i64, bool)>>) {
        self.gate = Some(rows);
    }

    /// Is the regime gate off at `now`? (No gate: never.)
    fn gate_off(&self, now: i64) -> bool {
        let Some(g) = &self.gate else { return false };
        let i = g.partition_point(|x| x.0 <= now);
        i == 0 || !g[i - 1].1
    }

    fn step(&self) -> f64 {
        self.rules.qty_step
    }

    pub fn position_qty(&self) -> f64 {
        self.pos as f64 * self.step()
    }

    /// Place a PostOnly limit order (sent now, live after the latency).
    pub fn place(&mut self, side: Side, price: f64, qty: f64, reduce_only: bool, now: i64, grid: &Grid) -> bool {
        let (Some(tick), Some(lots)) = (grid.ticks(price), grid.lots(qty)) else { return false };
        if lots <= 0 {
            return false;
        }
        self.funnel.placed += 1;
        let price = grid.price(tick);
        self.orders.push(SimOrder {
            side,
            price,
            tick,
            leaves: lots,
            reduce_only,
            active_ms: now + self.latency,
            active: false,
            cancel_ms: None,
            updated_ms: now,
            q: Queue::default(),
            ahead0: 0.0,
            closed: false,
            filled: 0,
        });
        true
    }

    fn cancel_idx(&mut self, i: usize, now: i64) {
        if self.orders[i].cancel_ms.is_none() && !self.orders[i].closed {
            self.orders[i].cancel_ms = Some(now + self.latency);
            self.funnel.cancels += 1;
        }
    }

    fn close_order(&mut self, i: usize, t: i64) {
        let o = &mut self.orders[i];
        if o.closed {
            return;
        }
        if o.active {
            let live = (t.min(self.end) - o.active_ms.max(self.start)).max(0);
            match o.side {
                Side::Buy => self.funnel.live_ms_bid += live,
                Side::Sell => self.funnel.live_ms_ask += live,
            }
        }
        o.closed = true;
    }

    fn reducible(&self, side: Side) -> i64 {
        match side {
            Side::Buy => (-self.pos).max(0),
            Side::Sell => self.pos.max(0),
        }
    }

    /// Book a fill of `lots` at `price` on `side`.
    #[allow(clippy::too_many_arguments)]
    fn book_fill(&mut self, ts: i64, side: Side, price: f64, lots: i64, maker: bool, kind: &'static str, reduce_only: bool, ahead0: f64, mid: f64) {
        if lots <= 0 {
            return;
        }
        let s = side.sign();
        let notional = self.rules.qty_step * lots as f64 * price;
        let fee = notional * if maker { self.fees.maker_bps } else { self.fees.taker_bps } / 1e4;
        self.cash -= s as f64 * notional + fee;
        self.fee_paid += fee;
        let ap = self.pos.abs();
        if self.pos == 0 || self.pos.signum() == s {
            self.avg = (ap as f64 * self.avg + lots as f64 * price) / (ap + lots) as f64;
            self.pos += s * lots;
        } else if lots <= ap {
            self.pos += s * lots;
            if self.pos == 0 {
                self.avg = 0.0;
            }
        } else {
            self.pos = s * (lots - ap);
            self.avg = price;
        }
        self.max_inv_usd = self.max_inv_usd.max((self.pos as f64 * self.step() * mid).abs());
        match kind {
            "queue" => self.funnel.fills_queue += 1,
            "through" => self.funnel.fills_through += 1,
            "cross" => self.funnel.fills_cross += 1,
            "gate" => self.funnel.fills_gate += 1,
            _ => self.funnel.fills_stop += 1,
        }
        self.markouts.push((s as f64, price, ts));
        let q = self.rules.qty_step * lots as f64;
        self.fills.push(Fill { ts, side: side_str(side), price, qty: q, maker, kind, reduce_only, fee, mid, ahead0_lots: ahead0, pos_after: self.position_qty() });
        // Venue reduce-only rule: resting reduce-only orders never exceed the position; orders that no longer reduce go.
        for i in 0..self.orders.len() {
            if self.orders[i].closed || !self.orders[i].reduce_only {
                continue;
            }
            let allowed = self.reducible(self.orders[i].side);
            if allowed == 0 {
                self.close_order(i, ts);
                self.funnel.ro_trimmed += 1;
            } else if self.orders[i].leaves > allowed {
                self.orders[i].leaves = allowed;
                self.funnel.ro_trimmed += 1;
            }
        }
    }

    /// Fill order `i` for `lots` (capped by reduce-only).
    fn fill_order(&mut self, i: usize, lots: i64, ts: i64, kind: &'static str, mid: f64) -> i64 {
        let o = &self.orders[i];
        let mut f = lots.min(o.leaves);
        if o.reduce_only {
            f = f.min(self.reducible(o.side));
        }
        if f <= 0 {
            return 0;
        }
        let (side, price, ro, a0, first) = (o.side, o.price, o.reduce_only, o.ahead0, o.filled == 0);
        let o = &mut self.orders[i];
        o.leaves -= f;
        o.filled += f;
        o.updated_ms = ts;
        let done = o.leaves == 0;
        if first {
            self.funnel.filled_orders += 1;
        }
        if done {
            self.close_order(i, ts);
        }
        self.book_fill(ts, side, price, f, true, kind, ro, a0, mid);
        f
    }

    /// Fire every timer strictly before `t` (activations, effective cancels, the stop flatten, decision ticks), in order.
    pub fn advance(&mut self, t: i64, m: &Market) {
        loop {
            let mut next = i64::MAX;
            for o in self.orders.iter().filter(|o| !o.closed) {
                if !o.active {
                    next = next.min(o.active_ms);
                }
                if let Some(c) = o.cancel_ms {
                    next = next.min(c);
                }
            }
            if let Some(f) = self.flatten_ms {
                next = next.min(f);
            }
            if self.stopped.is_none() && self.next_tick < self.end {
                next = next.min(self.next_tick);
            }
            if next >= t || next == i64::MAX {
                break;
            }
            // activations and cancels due at `next`, then the flatten, then the tick
            for i in 0..self.orders.len() {
                if self.orders[i].closed {
                    continue;
                }
                if self.orders[i].cancel_ms.is_some_and(|c| c <= next) {
                    let c = self.orders[i].cancel_ms.unwrap_or(next);
                    self.close_order(i, c);
                    continue;
                }
                if !self.orders[i].active && self.orders[i].active_ms <= next {
                    self.activate(i, m);
                }
            }
            if self.flatten_ms.is_some_and(|f| f <= next) {
                self.flatten_ms = None;
                self.flatten(next, m, if self.stopped.is_some() { "stop" } else { "gate" });
            }
            if self.stopped.is_none() && self.next_tick <= next && self.next_tick < self.end {
                let now = self.next_tick;
                self.next_tick += self.tick_ms;
                self.tick(now, m);
            }
        }
    }

    fn activate(&mut self, i: usize, m: &Market) {
        let o = &self.orders[i];
        let (side, tick, t, price) = (o.side, o.tick, o.active_ms, o.price);
        let crosses = match side {
            Side::Buy => m.book.best_ask().is_some_and(|a| tick >= a.0),
            Side::Sell => m.book.best_bid().is_some_and(|b| tick <= b.0),
        };
        if crosses {
            self.funnel.postonly_rejects += 1;
            self.orders[i].closed = true;
            return;
        }
        let vis = m.visible(side, tick);
        let lvl = m.book.level_qty(side, tick) as f64;
        let mid = m.mid().unwrap_or(price);
        let o = &mut self.orders[i];
        o.active = true;
        o.updated_ms = t;
        o.q = if vis { Queue { ahead: lvl, behind: 0.0, known: true } } else { Queue::default() };
        o.ahead0 = if vis { lvl } else { f64::NAN };
        self.funnel.activated += 1;
        if vis {
            self.funnel.ahead0_lots_sum += lvl;
        } else {
            self.funnel.ahead0_unknown += 1;
        }
        self.funnel.dist_bps_sum += (price - mid).abs() / mid * 1e4;
    }

    /// A real trade: queue consumption at our price, fills through it.
    pub fn on_trade(&mut self, tr: &Tr, m: &Market) {
        if tr.rpi && !self.fill_rpi {
            return;
        }
        let mid = m.mid().unwrap_or(tr.px);
        for i in 0..self.orders.len() {
            let o = &self.orders[i];
            if o.closed || !o.active || o.leaves <= 0 {
                continue;
            }
            let (side, tick, leaves) = (o.side, o.tick, o.leaves);
            // our bid is hit by sell aggressors, our ask lifted by buy aggressors
            let ours = match side {
                Side::Buy => !tr.buy,
                Side::Sell => tr.buy,
            };
            if !ours {
                continue;
            }
            let through = match side {
                Side::Buy => tr.tick < tick,
                Side::Sell => tr.tick > tick,
            };
            if through {
                self.orders[i].q = Queue { ahead: 0.0, behind: 0.0, known: true };
                self.fill_order(i, tr.lots, tr.ts, "through", mid);
            } else if tr.tick == tick {
                let f = if self.orders[i].q.known {
                    self.orders[i].q.on_trade_at(tr.lots as f64, leaves)
                } else {
                    // a print at our price while the level is outside the window: it is at the front now
                    self.orders[i].q = Queue { ahead: 0.0, behind: 0.0, known: true };
                    0
                };
                if f > 0 {
                    self.fill_order(i, f, tr.ts, "queue", mid);
                }
            }
        }
    }

    /// A book message: reconcile the queue of every order whose level changed; fill orders the book crossed.
    pub fn on_book(&mut self, m: &Market, cts: i64) {
        if self.orders.iter().all(|o| o.closed || !o.active) {
            return;
        }
        let mid = m.mid().unwrap_or(0.0);
        for i in 0..self.orders.len() {
            let o = &self.orders[i];
            if o.closed || !o.active || o.leaves <= 0 {
                continue;
            }
            let (side, tick, leaves, price) = (o.side, o.tick, o.leaves, o.price);
            // crossed by a real resting order on the other side
            let opp = side.opposite();
            let crossing = match (side, m.book.best_ask(), m.book.best_bid()) {
                (Side::Buy, Some(a), _) if a.0 <= tick => Some(a.0),
                (Side::Sell, _, Some(b)) if b.0 >= tick => Some(b.0),
                _ => None,
            };
            if let Some(best) = crossing {
                let (lo, hi) = if side == Side::Buy { (best, tick) } else { (tick, best) };
                let mut lots = 0;
                for t in lo..=hi {
                    lots += m.book.level_qty(opp, t);
                    if lots >= leaves {
                        break;
                    }
                }
                self.orders[i].q = Queue { ahead: 0.0, behind: 0.0, known: true };
                self.fill_order(i, lots, cts, "cross", if mid > 0.0 { mid } else { price });
                continue;
            }
            let touched = m.snapshot_msg || m.changed.iter().any(|c| c.0 == side && c.1 == tick);
            if touched && m.visible(side, tick) {
                let lvl = m.book.level_qty(side, tick) as f64;
                let model = self.model;
                self.orders[i].q.on_level(lvl, model);
            }
        }
    }

    fn flatten(&mut self, now: i64, m: &Market, kind: &'static str) {
        if self.pos == 0 {
            return;
        }
        let side = if self.pos > 0 { Side::Sell } else { Side::Buy };
        let mut left = self.pos.abs();
        let (bids, asks) = m.levels(usize::MAX);
        let book = if side == Side::Sell { bids } else { asks };
        let mid = m.mid().unwrap_or(self.avg);
        let mut fills = vec![];
        for (p, q) in book {
            if left == 0 {
                break;
            }
            let lots = m.grid().lots_nearest(q).min(left);
            fills.push((p, lots));
            left -= lots;
        }
        if left > 0 {
            // empty book side: close at the mid (reported as a stop fill)
            fills.push((mid, left));
        }
        for (p, lots) in fills {
            self.book_fill(now, side, p, lots, false, kind, true, f64::NAN, mid);
        }
    }

    fn stop(&mut self, now: i64, why: &'static str) {
        self.stopped = Some((now, why));
        for i in 0..self.orders.len() {
            self.cancel_idx(i, now);
        }
        // cancel_all then a reduce-only market close, one order pause later
        self.flatten_ms = Some(now + self.cfg.order_pause_ms as i64 + self.latency);
    }

    fn flow_due(&self) -> bool {
        if !self.spec.flow_rotation {
            return true;
        }
        let (Some(i), n) = (self.sym_index, self.cfg.symbols.len()) else { return true };
        let batch = if self.cfg.flow_batch == 0 { n } else { self.cfg.flow_batch.min(n) };
        let start = (self.tick_n as usize * batch) % n.max(1);
        (i + n - start) % n < batch
    }

    /// One decision, as one symbol inside the `atx mm` loop.
    fn tick(&mut self, now: i64, m: &Market) {
        self.funnel.ticks += 1;
        self.orders.retain(|o| !o.closed);
        let cfg = self.cfg.clone();
        if now - self.sigma_at > 15 * MIN_MS {
            let closes = m.closes(now - cfg.sigma_lookback_min * MIN_MS, now);
            let (slow, fast) = mm::sigma_pair(&cfg, &closes);
            let (h, l) = m.bars15(now - cfg.level_lookback_h * 3_600_000, now);
            self.st.sigma = slow.max(fast);
            self.st.levels = mm::levels_with_zones(&cfg, &h, &l, &[], closes.last().copied().unwrap_or(0.0));
            self.sigma_at = now;
        }
        let Some((bt, at)) = m.top() else {
            self.tick_n += 1;
            return;
        };
        let g = *m.grid();
        let (bid1, ask1) = (g.price(bt), g.price(at));
        let mid = 0.5 * (bid1 + ask1);
        if self.flow_due() {
            let limit = mm::flow_trades_limit(&cfg) as usize;
            let rec: Vec<&Tr> = m.recent(limit, now).collect();
            let (bids, asks) = m.levels(50);
            let trades: Vec<(i64, f64, f64)> = rec.iter().map(|t| (t.ts, if t.buy { 1.0 } else { -1.0 }, t.px * g.qty(t.lots))).collect();
            if cfg.micro_interval_s > 0 {
                let px: Vec<(i64, f64)> = rec.iter().map(|t| (t.ts, t.px)).collect();
                self.st.micro = mm::micro_sigma(&px, now, cfg.micro_interval_s, mm::micro_lookback(&cfg)).unwrap_or(0.0);
            }
            if cfg.trend_guard_sigma > 0.0 {
                let pts: Vec<(i64, f64)> = rec.iter().map(|t| (t.ts, t.px)).filter(|p| p.1 > 0.0).collect();
                if let Some((r, s)) = mm::trend_window(&pts) {
                    self.st.trend_ret = r;
                    self.st.trend_span_s = s;
                }
            }
            if let (Some(b1), Some(a1)) = (bids.first(), asks.first()) {
                self.st.flow = mm::flow_signal(&cfg, &trades, now, &bids, &asks, 0.5 * (b1.0 + a1.0));
            }
        }
        // toxicity: markouts due
        let due: Vec<(f64, f64, i64)> = self.markouts.iter().filter(|x| now - x.2 >= cfg.tox_horizon_s * 1000).cloned().collect();
        self.markouts.retain(|x| now - x.2 < cfg.tox_horizon_s * 1000);
        for (s, px, _) in due {
            (_, self.st.tox) = mm::tox_update(&cfg, self.st.tox, s, px, mid, self.st.sigma);
        }
        // loss limits on the fill ledger (cash + inventory at mid)
        let pnl = self.cash + self.position_qty() * mid;
        if pnl <= -cfg.loss_limit_usd {
            self.stop(now, "loss_limit");
            self.tick_n += 1;
            return;
        }
        if cfg.symbol_loss_limit_usd > 0.0 && pnl <= -cfg.symbol_loss_limit_usd {
            self.stop(now, "symbol_stop");
            self.tick_n += 1;
            return;
        }
        if let Some(_jump) = mm::shock_jump(&cfg, &self.st, mid, now) {
            self.st.paused_until_ms = now + cfg.shock_cooldown_s * 1000;
            self.funnel.shock_pauses += 1;
        }
        self.st.last_mid = mid;
        self.st.last_mid_ms = now;
        let e = self.st.clone();
        let guard = mm::trend_against_view(e.trend_ret, e.trend_span_s, e.sigma, cfg.trend_guard_sigma, mm::view_of(&cfg, &self.symbol, mid));
        self.funnel.guard_ticks += guard as u64;
        let gate_off = self.gate_off(now);
        self.funnel.gate_off_ticks += gate_off as u64;
        if gate_off && self.spec.gate_flatten && self.gate_was_on && self.pos != 0 && self.flatten_ms.is_none() {
            for i in 0..self.orders.len() {
                self.cancel_idx(i, now);
            }
            self.flatten_ms = Some(now + cfg.order_pause_ms as i64 + self.latency);
            self.funnel.gate_flattens += 1;
        }
        self.gate_was_on = !gate_off;
        let exit_only = gate_off || self.spec.deadline_ms.is_some_and(|d| now >= d - cfg.exit_window_min * MIN_MS);
        let pos = (self.position_qty(), if self.pos == 0 { 0.0 } else { self.avg });
        let mut x = mm::symbol_inputs(&cfg, &self.symbol, &e, bid1, ask1, pos, Others::default(), exit_only, now < e.paused_until_ms, guard);
        match self.spec.sigma_mode {
            SigmaMode::Config => {}
            SigmaMode::Fixed => x.sigma = self.spec.fixed_bps / 1e4,
            SigmaMode::Forecast => match self.forecast_now(now) {
                Some(s) => x.sigma = self.spec.forecast_scale * s,
                None => self.funnel.forecast_missing_ticks += 1,
            },
        }
        let (bid, ask) = mm::target_quotes(&cfg, &self.rules, &x, &e.levels);
        let mut t_op = now;
        let pause = cfg.order_pause_ms as i64;
        for (side, want) in [(Side::Buy, bid), (Side::Sell, ask)] {
            let resting: Vec<usize> = (0..self.orders.len()).filter(|&i| !self.orders[i].closed && self.orders[i].cancel_ms.is_none() && self.orders[i].side == side).collect();
            for &i in resting.iter().skip(1) {
                self.cancel_idx(i, t_op);
                t_op += pause;
            }
            let cur = resting.first().copied();
            let rest = cur.map(|i| {
                let o = &self.orders[i];
                Resting { price: o.price, leaves_qty: g.qty(o.leaves), reduce_only: o.reduce_only, updated_ms: o.updated_ms }
            });
            let replace = mm::needs_replace(&cfg, &self.rules, want, rest.as_ref(), mid, x.sigma, now);
            if let (true, Some(i)) = (replace, cur) {
                self.cancel_idx(i, t_op);
                t_op += pause;
            }
            if let (Some(q), true) = (want, cur.is_none() || replace) {
                let Quote { price, qty, reduce_only } = q;
                self.place(side, price, qty, reduce_only, t_op, &g);
                t_op += pause;
            }
        }
        self.tick_n += 1;
    }

    fn forecast_now(&self, now: i64) -> Option<f64> {
        let f = FORECAST.with(|f| f.borrow().clone());
        let i = f.partition_point(|x| x.0 <= now);
        if i == 0 { None } else { Some(f[i - 1].1).filter(|s| *s > 0.0 && s.is_finite()) }
    }

    /// Close the books at `end`: open orders' live time, inventory marked at the mid.
    pub fn finish(&mut self, end: i64) {
        for i in 0..self.orders.len() {
            self.close_order(i, end);
        }
    }

    pub fn summary(&self, m: &Market, end: i64) -> Value {
        let mid_end = m.mid_at(end).unwrap_or(0.0);
        let (mut spread, mut fees) = (0.0, 0.0);
        let mut adv = [0.0f64; 3];
        let mut inv = 0.0;
        let mut vol = 0.0;
        let mut maker = 0u64;
        for f in &self.fills {
            let s = if f.side == "Buy" { 1.0 } else { -1.0 };
            let m0 = f.mid;
            spread += s * f.qty * (m0 - f.price);
            let mut prev = m0;
            for (k, h) in MARKOUTS_S.iter().enumerate() {
                let mh = m.mid_at((f.ts + h * 1000).min(end)).unwrap_or(m0);
                adv[k] += s * f.qty * (mh - m0);
                prev = mh;
            }
            inv += s * f.qty * (mid_end - prev);
            fees += f.fee;
            vol += f.qty * f.price;
            maker += f.maker as u64;
        }
        let gross = spread + adv[2] + inv;
        let net = gross - fees;
        let ledger = self.cash + self.position_qty() * mid_end;
        let fu = &self.funnel;
        let span = (end - self.start).max(1) as f64;
        json!({
            "name": self.name, "queue": self.model.as_str(), "symbol": self.symbol,
            "sigma_mode": self.spec.sigma_mode, "fixed_bps": self.spec.fixed_bps, "forecast_scale": self.spec.forecast_scale, "tick_ms": self.tick_ms,
            "pnl": {"net": net, "gross": gross, "spread": spread, "adverse_1s": adv[0], "adverse_10s": adv[1], "adverse_60s": adv[2],
                    "inventory": inv, "fees": fees, "ledger_check": ledger - net},
            "fills": self.fills.len(), "maker_fills": maker, "volume_usd": vol, "end_position": self.position_qty(), "end_mid": mid_end,
            "end_inventory_usd": self.position_qty() * mid_end, "max_inventory_usd": self.max_inv_usd,
            "stopped": self.stopped.map(|(t, w)| json!({"ts": t, "reason": w})),
            "funnel": {
                "ticks": fu.ticks, "placed": fu.placed, "activated": fu.activated, "postonly_rejects": fu.postonly_rejects, "cancels": fu.cancels,
                "ro_trimmed": fu.ro_trimmed, "fills_queue": fu.fills_queue, "fills_through": fu.fills_through, "fills_cross": fu.fills_cross,
                "fills_stop": fu.fills_stop, "filled_orders": fu.filled_orders,
                "ahead0_lots_mean": if fu.activated > fu.ahead0_unknown { fu.ahead0_lots_sum / (fu.activated - fu.ahead0_unknown) as f64 } else { f64::NAN },
                "ahead0_unknown": fu.ahead0_unknown,
                "dist_bps_mean": if fu.activated > 0 { fu.dist_bps_sum / fu.activated as f64 } else { f64::NAN },
                "live_share_bid": fu.live_ms_bid as f64 / span, "live_share_ask": fu.live_ms_ask as f64 / span,
                "shock_pauses": fu.shock_pauses, "guard_ticks": fu.guard_ticks, "forecast_missing_ticks": fu.forecast_missing_ticks,
                "gate_off_ticks": fu.gate_off_ticks, "gate_flattens": fu.gate_flattens, "fills_gate": fu.fills_gate,
            },
            "gate": self.spec.gate, "gate_flatten": self.spec.gate_flatten,
        })
    }
}

thread_local! {
    /// The external 1h sigma forecast (single-threaded replay; set once per run).
    static FORECAST: std::cell::RefCell<std::rc::Rc<Vec<(i64, f64)>>> = std::cell::RefCell::new(std::rc::Rc::new(vec![]));
}

// ---------------------------------------------------------------- driver

pub struct ReplayParams {
    pub start: i64,
    pub end: i64,
    pub latency_ms: i64,
    pub fill_rpi: bool,
    pub depth: usize,
}

/// Runs every quoter over one symbol-day: book messages from `book`, trades `trades` (time-sorted).
pub fn run_replay(m: &mut Market, quoters: &mut [Quoter], book: &mut dyn BufRead, trades: &[Tr], p: &ReplayParams) -> anyhow::Result<()> {
    let mut ti = 0usize;
    let mut line = String::new();
    let mut last_cts = i64::MIN;
    let step = |m: &mut Market, quoters: &mut [Quoter], tr: &Tr| {
        if tr.ts < p.end {
            for q in quoters.iter_mut() {
                q.advance(tr.ts, m);
            }
        }
        m.on_trade(tr);
        if tr.ts >= p.start && tr.ts < p.end {
            for q in quoters.iter_mut() {
                q.on_trade(tr, m);
            }
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
            step(m, quoters, &trades[ti]);
            ti += 1;
        }
        if cts < p.end {
            for q in quoters.iter_mut() {
                q.advance(cts, m);
            }
        }
        m.apply_book(&msg, cts)?;
        if cts >= p.start && cts < p.end {
            for q in quoters.iter_mut() {
                q.on_book(m, cts);
            }
        }
    }
    while ti < trades.len() {
        step(m, quoters, &trades[ti]);
        ti += 1;
    }
    for q in quoters.iter_mut() {
        q.advance(p.end, m);
        q.finish(p.end);
    }
    Ok(())
}

#[derive(clap::Args)]
pub struct MmReplayArgs {
    #[arg(long)]
    symbol: String,
    /// UTC day (YYYY-MM-DD): the replay window is the whole day unless --start-ms / --end-ms narrow it.
    #[arg(long)]
    date: Option<String>,
    #[arg(long)]
    start_ms: Option<i64>,
    #[arg(long)]
    end_ms: Option<i64>,
    /// quote-saver ob200 archive (.zip, or unzipped JSON lines).
    #[arg(long)]
    ob200: Option<PathBuf>,
    /// public.bybit.com trade archive (.csv.gz or .csv).
    #[arg(long)]
    trades: PathBuf,
    /// 1m bars of earlier days (bars_1m.csv from previous runs or --bars-only) for the sigma / level warm-up.
    #[arg(long, value_delimiter = ',')]
    warmup_bars: Vec<PathBuf>,
    /// Variants JSON: {"variants": [{"name", "config", "overrides", "sigma_mode", "fixed_bps", "forecast_scale", "tick_ms", "flow_rotation", "queues", "deadline_ms", "gate", "gate_flatten"}]}.
    #[arg(long)]
    variants: Option<PathBuf>,
    #[arg(long)]
    out: PathBuf,
    /// Delay from sending an order (or a cancel) to it taking effect.
    #[arg(long, default_value_t = 150)]
    latency_ms: i64,
    /// 1h sigma forecast CSV (valid_from_ms,sigma_1h) for sigma_mode "forecast".
    #[arg(long)]
    forecast: Option<PathBuf>,
    /// Let RPI prints fill our queue (default: they don't).
    #[arg(long)]
    fill_rpi: bool,
    /// Levels per side in the replayed feed.
    #[arg(long, default_value_t = 200)]
    depth: usize,
    #[arg(long)]
    rules: Option<PathBuf>,
    #[arg(long)]
    fees: Option<PathBuf>,
    #[arg(long)]
    fee_class: Option<String>,
    /// Override the tick size / qty step (historical days whose grid differs from the lake's latest rules).
    #[arg(long)]
    tick: Option<f64>,
    #[arg(long)]
    qty_step: Option<f64>,
    /// Only write bars_1m.csv from the trades.
    #[arg(long)]
    bars_only: bool,
}

fn parse_day(d: &str) -> anyhow::Result<i64> {
    let nd = chrono::NaiveDate::parse_from_str(d, "%Y-%m-%d")?;
    Ok(nd.and_hms_opt(0, 0, 0).context("date")?.and_utc().timestamp_millis())
}

pub fn run(ctx: &Ctx, a: MmReplayArgs) -> anyhow::Result<()> {
    let t0 = std::time::Instant::now();
    let symbol = a.symbol.to_uppercase();
    let day0 = a.date.as_deref().map(parse_day).transpose()?;
    let start = match a.start_ms.or(day0) {
        Some(s) => s,
        None if a.bars_only => at_core::time::now_ms(),
        None => bail!("--date or --start-ms"),
    };
    let end = a.end_ms.unwrap_or_else(|| day0.map_or(start + DAY_MS, |d| d + DAY_MS));
    let (mut rules, symbol_type) = match &a.rules {
        Some(p) => {
            let v: Value = serde_json::from_str(&std::fs::read_to_string(p)?)?;
            let list: Vec<InstrumentRules> = if v.is_array() { serde_json::from_value(v)? } else { vec![serde_json::from_value(v)?] };
            (list.into_iter().find(|r| r.symbol.eq_ignore_ascii_case(&symbol)).context("no rules for the symbol")?, None)
        }
        None => {
            let (mut r, mut types) = read_instruments(ctx.lake.root(), "bybit", ctx.category(), start)?;
            if !r.contains_key(&symbol) {
                (r, types) = read_instruments(ctx.lake.root(), "bybit", ctx.category(), at_core::time::now_ms())?;
            }
            (r.remove(&symbol).with_context(|| format!("no instrument rules for {symbol}"))?, types.remove(&symbol))
        }
    };
    if let Some(t) = a.tick {
        rules.tick_size = t;
    }
    if let Some(q) = a.qty_step {
        rules.qty_step = q;
    }
    let schedule = FeeSchedule::load(&a.fees.clone().unwrap_or_else(|| ctx.s.root.join("config/fees.yaml")))?;
    let hint = a.fee_class.clone().or_else(|| symbol_type.map(|t| if t.is_empty() { "crypto".into() } else { "tradfi".into() }));
    let fees = schedule.rates(&symbol, hint.as_deref());
    std::fs::create_dir_all(&a.out)?;
    let mut m = Market::new(&rules, a.depth)?;
    let trades = read_trades(&a.trades, m.grid())?;
    if a.bars_only {
        for t in &trades {
            m.on_trade(t);
        }
        let (lo, hi) = (trades.first().map_or(0, |t| t.ts.div_euclid(MIN_MS) * MIN_MS), trades.last().map_or(0, |t| t.ts + 1));
        write_bars(&a.out.join("bars_1m.csv"), &m.bars, lo, hi)?;
        println!("{symbol}: {} trades -> {} 1m bars in {}", trades.len(), m.bars.len(), a.out.display());
        return Ok(());
    }
    for p in &a.warmup_bars {
        read_bars(p, &mut m.bars)?;
    }
    if let Some(f) = &a.forecast {
        let v = read_forecast(f)?;
        FORECAST.with(|x| *x.borrow_mut() = std::rc::Rc::new(v));
    }
    let vf: VariantsFile = serde_json::from_str(&std::fs::read_to_string(a.variants.as_ref().context("--variants")?)?)?;
    let mut quoters = vec![];
    let mut gates: BTreeMap<String, std::rc::Rc<Vec<(i64, bool)>>> = BTreeMap::new();
    for v in &vf.variants {
        let cfg = load_cfg(&ctx.s.root, v)?;
        let gate = match &v.gate {
            Some(g) if !gates.contains_key(g) => {
                let rows = std::rc::Rc::new(read_gate(&ctx.s.root.join(g))?);
                gates.insert(g.clone(), rows.clone());
                Some(rows)
            }
            Some(g) => gates.get(g).cloned(),
            None => None,
        };
        for qm in &v.queues {
            let mut q = Quoter::new(cfg.clone(), v.clone(), *qm, &symbol, &rules, fees, a.latency_ms, start, end, a.fill_rpi);
            if let Some(g) = &gate {
                q.set_gate(g.clone());
            }
            quoters.push(q);
        }
    }
    let p = ReplayParams { start, end, latency_ms: a.latency_ms, fill_rpi: a.fill_rpi, depth: a.depth };
    let mut book = open_lines(a.ob200.as_ref().context("--ob200")?)?;
    run_replay(&mut m, &mut quoters, &mut *book, &trades, &p)?;
    let day_lo = start.div_euclid(DAY_MS) * DAY_MS;
    write_bars(&a.out.join("bars_1m.csv"), &m.bars, day_lo, day_lo + DAY_MS)?;
    write_outputs(&a.out, &m, &quoters, &p, &symbol, &rules, fees, t0.elapsed().as_secs_f64())?;
    println!(
        "{symbol}: {} book messages, {} trades ({} RPI), {} quoters, {} fills; engine trades {} rejects {} check mismatches {}/{}; {:.1}s",
        m.n_msgs,
        m.n_trades,
        m.n_rpi,
        quoters.len(),
        quoters.iter().map(|q| q.fills.len()).sum::<usize>(),
        m.engine_trades,
        m.engine_rejects,
        m.check_mismatch,
        m.checks,
        t0.elapsed().as_secs_f64()
    );
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn write_outputs(out: &Path, m: &Market, quoters: &[Quoter], p: &ReplayParams, symbol: &str, rules: &InstrumentRules, fees: FeeRates, secs: f64) -> anyhow::Result<()> {
    let mut w = BufWriter::new(File::create(out.join("fills.jsonl"))?);
    for q in quoters {
        for f in &q.fills {
            let mut v = serde_json::to_value(f)?;
            let mk: Vec<Value> = MARKOUTS_S.iter().map(|h| json!(m.mid_at((f.ts + h * 1000).min(p.end)))).collect();
            v["variant"] = json!(q.name);
            v["queue"] = json!(q.model.as_str());
            v["mid_1s"] = mk[0].clone();
            v["mid_10s"] = mk[1].clone();
            v["mid_60s"] = mk[2].clone();
            writeln!(w, "{}", serde_json::to_string(&v)?)?;
        }
    }
    w.flush()?;
    let rows: Vec<Value> = quoters.iter().map(|q| q.summary(m, p.end)).collect();
    let quotes: Vec<Value> = rows.iter().map(|r| json!({"name": r["name"], "queue": r["queue"], "funnel": r["funnel"]})).collect();
    std::fs::write(out.join("quotes.json"), serde_json::to_string_pretty(&quotes)?)?;
    let first_mid = m.mids.iter().find(|x| x.0 >= p.start).map(|x| x.1);
    let stats = json!({
        "symbol": symbol, "start_ms": p.start, "end_ms": p.end, "latency_ms": p.latency_ms, "fill_rpi": p.fill_rpi, "depth": p.depth,
        "rules": rules, "fees": {"maker_bps": fees.maker_bps, "taker_bps": fees.taker_bps},
        "market": {"book_messages": m.n_msgs, "trades": m.n_trades, "rpi_trades": m.n_rpi, "engine_trades": m.engine_trades, "engine_rejects": m.engine_rejects,
                   "crossed_messages": m.crossed_msgs, "trades_outside_touch": m.trades_outside_touch, "replayer_checks": m.checks,
                   "replayer_check_mismatches": m.check_mismatch, "mid_changes": m.mids.len(), "mid_start": first_mid, "mid_end": m.mid_at(p.end)},
        "quoters": rows,
        "perf": {"seconds": secs},
    });
    std::fs::write(out.join("stats.json"), serde_json::to_string_pretty(&stats)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rules() -> InstrumentRules {
        InstrumentRules { symbol: "TRXUSDT".into(), tick_size: 0.0001, qty_step: 1.0, min_qty: 1.0, max_qty: 1e6, max_mkt_qty: 1e5, min_notional: 5.0, max_leverage: 50.0 }
    }

    fn spec(name: &str) -> VariantSpec {
        VariantSpec {
            name: name.into(),
            config: "config/mm_illiq20.yaml".into(),
            overrides: serde_json::Map::new(),
            sigma_mode: SigmaMode::Fixed,
            fixed_bps: 10.0,
            forecast_scale: 1.0,
            tick_ms: 0,
            flow_rotation: false,
            queues: all_queues(),
            deadline_ms: None,
            gate: None,
            gate_flatten: false,
        }
    }

    fn cfg() -> MmCfg {
        load_cfg(Path::new(concat!(env!("CARGO_MANIFEST_DIR"), "/../..")), &spec("t")).unwrap()
    }

    fn quoter(model: QueueModel) -> Quoter {
        Quoter::new(cfg(), spec("t"), model, "TRXUSDT", &rules(), FeeRates { maker_bps: 2.0, taker_bps: 5.5, slippage_bps: 0.0 }, 150, 0, 10_000_000, false)
    }

    fn lv(p: f64, q: f64) -> Value {
        json!([format!("{p:.4}"), format!("{q}")])
    }

    fn msg(kind: &str, cts: i64, b: Vec<Value>, a: Vec<Value>) -> Value {
        json!({"topic": "orderbook.200.TRXUSDT", "type": kind, "ts": cts, "data": {"s": "TRXUSDT", "b": b, "a": a, "u": 1, "seq": 1}, "cts": cts})
    }

    fn trade(ts: i64, px: f64, lots: i64, buy: bool) -> Tr {
        Tr { ts, px, tick: (px / 0.0001).round() as i64, lots, buy, rpi: false }
    }

    #[test]
    fn queue_cancels_under_the_three_models() {
        // 10 lots ahead, 5 behind; the level drops to 9 with no trade: 6 lots cancelled
        let q0 = Queue { ahead: 10.0, behind: 5.0, known: true };
        let mut p = q0;
        p.on_level(9.0, QueueModel::Pessimistic);
        assert_eq!((p.ahead, p.behind), (9.0, 0.0), "pessimistic: behind first, then 1 from ahead");
        let mut r = q0;
        r.on_level(9.0, QueueModel::Prorata);
        assert!((r.ahead - 6.0).abs() < 1e-12 && (r.behind - 3.0).abs() < 1e-12, "pro-rata 10/15 of 6 from ahead: {r:?}");
        let mut o = q0;
        o.on_level(9.0, QueueModel::Optimistic);
        assert_eq!((o.ahead, o.behind), (4.0, 5.0), "optimistic: ahead first");
        // growth always joins behind
        for m in [QueueModel::Pessimistic, QueueModel::Prorata, QueueModel::Optimistic] {
            let mut g = q0;
            g.on_level(20.0, m);
            assert_eq!((g.ahead, g.behind), (10.0, 10.0));
        }
        // more cancels than queue: both sides empty, never negative
        let mut z = q0;
        z.on_level(0.0, QueueModel::Prorata);
        assert_eq!((z.ahead, z.behind), (0.0, 0.0));
        // an unknown level (outside the window) counts entirely ahead when it appears
        let mut u = Queue::default();
        u.on_level(7.0, QueueModel::Optimistic);
        assert_eq!((u.ahead, u.behind, u.known), (7.0, 0.0, true));
    }

    #[test]
    fn trades_consume_from_the_front() {
        let mut q = Queue { ahead: 3.0, behind: 4.0, known: true };
        assert_eq!(q.on_trade_at(2.0, 5), 0);
        assert_eq!((q.ahead, q.behind), (1.0, 4.0));
        // 1 ahead, then 5 of ours (we hold 5), the remaining 2 eat behind us
        assert_eq!(q.on_trade_at(8.0, 5), 5);
        assert_eq!((q.ahead, q.behind), (0.0, 2.0));
        // a fractional pro-rata queue fills whole lots only (rounded down)
        let mut f = Queue { ahead: 0.4, behind: 0.0, known: true };
        assert_eq!(f.on_trade_at(1.0, 1), 0);
        assert_eq!(f.on_trade_at(1.0, 1), 1);
    }

    /// Book at 0.3000 / 0.3002 with 100 lots on each touch; our bid joins at 0.3000 behind 100 lots.
    fn setup(model: QueueModel) -> (Market, Quoter) {
        let mut m = Market::new(&rules(), 200).unwrap();
        m.apply_book(&msg("snapshot", 1_000, vec![lv(0.3000, 100.0), lv(0.2999, 50.0)], vec![lv(0.3002, 100.0), lv(0.3003, 50.0)]), 1_000).unwrap();
        let mut q = quoter(model);
        q.next_tick = i64::MAX; // no decisions: orders are placed by hand
        let g = *m.grid();
        assert!(q.place(Side::Buy, 0.3000, 20.0, false, 2_000, &g));
        (m, q)
    }

    #[test]
    fn latency_on_placement_and_cancel() {
        let (m, mut q) = setup(QueueModel::Prorata);
        // a sell sweep through our price 100 ms after sending: the order is not live yet
        let t = trade(2_100, 0.2999, 500, false);
        q.advance(t.ts, &m);
        q.on_trade(&t, &m);
        assert!(q.fills.is_empty(), "not live before the latency");
        // 150 ms after sending it is live and joins behind the 100 lots
        let t = trade(2_200, 0.2999, 30, false);
        q.advance(t.ts, &m);
        assert_eq!(q.orders[0].q.ahead, 100.0);
        q.on_trade(&t, &m);
        assert_eq!(q.fills.len(), 1, "a print through our price fills us");
        assert_eq!(q.fills[0].qty, 20.0);
        assert_eq!(q.fills[0].kind, "through");
        // a cancel sent at 3000 takes effect at 3150: a fill at 3100 still counts, at 3200 it does not
        let g = *m.grid();
        q.place(Side::Sell, 0.3002, 20.0, true, 2_500, &g);
        q.advance(2_700, &m);
        q.cancel_idx(1, 3_000);
        let t = trade(3_100, 0.3003, 10, true);
        q.advance(t.ts, &m);
        q.on_trade(&t, &m);
        assert_eq!(q.fills.len(), 2, "fills before the cancel is effective count");
        let t = trade(3_200, 0.3003, 10, true);
        q.advance(t.ts, &m);
        q.on_trade(&t, &m);
        assert_eq!(q.fills.len(), 2, "no fill after the cancel took effect");
    }

    #[test]
    fn queue_position_under_the_three_models_end_to_end() {
        let mut got = vec![];
        for model in [QueueModel::Pessimistic, QueueModel::Prorata, QueueModel::Optimistic] {
            let (mut m, mut q) = setup(model);
            q.advance(2_200, &m);
            // 30 lots join behind us, then the level drops by 60 with no print (cancels): 130 -> 70
            m.apply_book(&msg("delta", 2_300, vec![lv(0.3000, 130.0)], vec![]), 2_300).unwrap();
            q.advance(2_300, &m);
            q.on_book(&m, 2_300);
            m.apply_book(&msg("delta", 2_400, vec![lv(0.3000, 70.0)], vec![]), 2_400).unwrap();
            q.on_book(&m, 2_400);
            let ahead = q.orders[0].q.ahead;
            // then a 60-lot sell print at our price
            let t = trade(2_500, 0.3000, 60, false);
            q.advance(t.ts, &m);
            q.on_trade(&t, &m);
            got.push((ahead, q.fills.iter().map(|f| f.qty).sum::<f64>()));
        }
        // pessimistic: 30 from behind, 30 from ahead -> 70 ahead, no fill; pro-rata: 60 x 100/130 from ahead -> ~53.8,
        // 6 lots filled; optimistic: 60 from ahead -> 40 ahead, 20 (all) filled
        assert_eq!(got[0], (70.0, 0.0));
        assert!((got[1].0 - (100.0 - 60.0 * 100.0 / 130.0)).abs() < 1e-9 && got[1].1 == 6.0, "{:?}", got[1]);
        assert_eq!(got[2], (40.0, 20.0));
    }

    #[test]
    fn crossing_book_fills_and_postonly_rejects() {
        let (mut m, mut q) = setup(QueueModel::Prorata);
        q.advance(2_200, &m);
        // asks move down to our bid price (a resting seller at 0.3000): we are filled there
        m.apply_book(&msg("delta", 2_300, vec![lv(0.3000, 0.0)], vec![lv(0.3000, 5.0)]), 2_300).unwrap();
        q.on_book(&m, 2_300);
        assert_eq!(q.fills.len(), 1);
        assert_eq!((q.fills[0].kind, q.fills[0].qty), ("cross", 5.0));
        // a bid sent at or above the ask is rejected PostOnly at activation
        let g = *m.grid();
        q.place(Side::Buy, 0.3001, 20.0, false, 3_000, &g);
        m.apply_book(&msg("delta", 3_100, vec![], vec![lv(0.3001, 9.0)]), 3_100).unwrap();
        q.advance(3_200, &m);
        assert_eq!(q.funnel.postonly_rejects, 1);
    }

    fn synthetic_day() -> (Vec<String>, Vec<Tr>) {
        let mut r = at_match::flow::Rng::new(7);
        let mut lines = vec![];
        let mut trades = vec![];
        let mut mid = 3000i64; // ticks of 0.0001 -> 0.3000
        lines.push(msg("snapshot", 0, (1..=20).map(|k| lv((mid - k) as f64 * 1e-4, 100.0)).collect(), (1..=20).map(|k| lv((mid + k) as f64 * 1e-4, 100.0)).collect()).to_string());
        let mut t = 0;
        while t < 3 * 3_600_000 {
            t += 500;
            let mv = r.below(3) as i64 - 1;
            if mv != 0 && r.below(10) == 0 {
                let old = mid;
                mid += mv;
                let (b, a): (Vec<Value>, Vec<Value>) = if mv > 0 {
                    (vec![lv((mid - 1) as f64 * 1e-4, 80.0)], vec![lv((old + 1) as f64 * 1e-4, 0.0), lv((mid + 20) as f64 * 1e-4, 100.0)])
                } else {
                    (vec![lv((old - 1) as f64 * 1e-4, 0.0), lv((mid - 20) as f64 * 1e-4, 100.0)], vec![lv((mid + 1) as f64 * 1e-4, 80.0)])
                };
                lines.push(msg("delta", t, b, a).to_string());
            } else {
                let sz = 50.0 + r.below(100) as f64;
                lines.push(msg("delta", t, vec![lv((mid - 1) as f64 * 1e-4, sz)], vec![lv((mid + 1) as f64 * 1e-4, 150.0 - sz)]).to_string());
            }
            if r.below(4) == 0 {
                let buy = r.below(2) == 0;
                let px = if buy { mid + 1 } else { mid - 1 };
                trades.push(trade(t - 1, px as f64 * 1e-4, 1 + r.below(60) as i64, buy));
            }
        }
        (lines, trades)
    }

    fn run_synthetic() -> (String, String) {
        let (lines, trades) = synthetic_day();
        let mut m = Market::new(&rules(), 200).unwrap();
        let mut s = spec("fixed10");
        s.overrides.insert("refresh_s".into(), json!(5));
        let mut quoters: Vec<Quoter> = all_queues()
            .into_iter()
            .map(|qm| Quoter::new(cfg_with(&s), s.clone(), qm, "TRXUSDT", &rules(), FeeRates { maker_bps: 2.0, taker_bps: 5.5, slippage_bps: 0.0 }, 150, 0, 3 * 3_600_000, false))
            .collect();
        let text = lines.join("\n");
        let mut rd = BufReader::new(text.as_bytes());
        let p = ReplayParams { start: 0, end: 3 * 3_600_000, latency_ms: 150, fill_rpi: false, depth: 200 };
        run_replay(&mut m, &mut quoters, &mut rd, &trades, &p).unwrap();
        let dir = std::env::temp_dir().join(format!("mm_replay_test_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        write_outputs(&dir, &m, &quoters, &p, "TRXUSDT", &rules(), FeeRates { maker_bps: 2.0, taker_bps: 5.5, slippage_bps: 0.0 }, 0.0).unwrap();
        let mut st: Value = serde_json::from_str(&std::fs::read_to_string(dir.join("stats.json")).unwrap()).unwrap();
        st["perf"] = json!(null);
        let fills = std::fs::read_to_string(dir.join("fills.jsonl")).unwrap();
        let _ = std::fs::remove_dir_all(&dir);
        (st.to_string(), fills)
    }

    fn cfg_with(s: &VariantSpec) -> MmCfg {
        load_cfg(Path::new(concat!(env!("CARGO_MANIFEST_DIR"), "/../..")), s).unwrap()
    }

    #[test]
    fn replay_is_deterministic_and_consistent() {
        let (a, fa) = run_synthetic();
        let (b, fb) = run_synthetic();
        assert_eq!(a, b, "same inputs, same stats");
        assert_eq!(fa, fb, "same inputs, same fills");
        let st: Value = serde_json::from_str(&a).unwrap();
        assert_eq!(st["market"]["engine_trades"], 0);
        assert_eq!(st["market"]["engine_rejects"], 0);
        assert_eq!(st["market"]["replayer_check_mismatches"], 0);
        let qs = st["quoters"].as_array().unwrap();
        assert_eq!(qs.len(), 3);
        for q in qs {
            // the decomposition adds up to the cash ledger marked at the end mid
            assert!(q["pnl"]["ledger_check"].as_f64().unwrap().abs() < 1e-9, "{q}");
            assert!(q["funnel"]["placed"].as_u64().unwrap() > 0);
        }
        // optimistic fills at least as often as pro-rata, pro-rata at least as often as pessimistic
        let n = |i: usize| qs[i]["funnel"]["fills_queue"].as_u64().unwrap();
        assert!(n(0) <= n(1) && n(1) <= n(2), "{} {} {}", n(0), n(1), n(2));
    }

    #[test]
    fn regime_gate_blocks_new_inventory_and_flattens() {
        let (lines, trades) = synthetic_day();
        let gate = std::rc::Rc::new(vec![(0, true), (3_600_000, false), (7_200_000, true)]);
        let fee = FeeRates { maker_bps: 2.0, taker_bps: 5.5, slippage_bps: 0.0 };
        let mk = |s: &VariantSpec| Quoter::new(cfg_with(s), s.clone(), QueueModel::Optimistic, "TRXUSDT", &rules(), fee, 150, 0, 3 * 3_600_000, false);
        let mut quoters = vec![];
        for flat in [false, true] {
            let mut s = spec(if flat { "gated_flat" } else { "gated" });
            s.overrides.insert("refresh_s".into(), json!(5));
            s.gate_flatten = flat;
            s.fixed_bps = 2.0;
            let mut q = mk(&s);
            q.set_gate(gate.clone());
            quoters.push(q);
        }
        let mut s = spec("ungated");
        s.overrides.insert("refresh_s".into(), json!(5));
        s.fixed_bps = 2.0;
        quoters.push(mk(&s));
        let text = lines.join("\n");
        let mut rd = BufReader::new(text.as_bytes());
        let p = ReplayParams { start: 0, end: 3 * 3_600_000, latency_ms: 150, fill_rpi: false, depth: 200 };
        run_replay(&mut m_new(), &mut quoters, &mut rd, &trades, &p).unwrap();
        for q in &quoters[..2] {
            // 720 decision ticks (5 s) inside the off hour
            assert_eq!(q.funnel.gate_off_ticks, 720, "{}", q.name);
            // while the gate is off (plus one latency for orders already sent) no fill adds inventory
            let off = |t: i64| t > 3_600_000 + 5_000 + 150 && t < 7_200_000;
            assert!(q.fills.iter().filter(|f| off(f.ts)).all(|f| f.reduce_only), "{}", q.name);
            assert!(q.fills.iter().any(|f| f.ts >= 7_200_000), "{} quotes again after the gate reopens", q.name);
        }
        assert_eq!(quoters[2].funnel.gate_off_ticks, 0);
        assert!(quoters[2].fills.iter().any(|f| f.ts > 3_600_000 + 5_000 && f.ts < 7_200_000 && !f.reduce_only), "the ungated quoter adds inventory in that hour");
        // the flatten variant is flat within a pause + latency of the first off tick when it held inventory
        let fq = &quoters[1];
        if fq.funnel.gate_flattens > 0 {
            assert!(fq.fills.iter().any(|f| f.kind == "gate" && f.pos_after == 0.0));
        }
    }

    fn m_new() -> Market {
        Market::new(&rules(), 200).unwrap()
    }
}
