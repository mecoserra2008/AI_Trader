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
//! Weekend alt quoting rules (H-MMW1-4, the pure rules in `mm`): `mode` "ladder" (minimum lots at structural zones
//! `--zones` 1.5-3.5 sigma from the mid, a passive take-profit at the mid at fill time, a taker stop 1.5 sigma_risk
//! beyond the level), "touch" (join the touch only behind a short queue, leave when the queue behind us thins); overlays
//! on any mode: a BTC-move pull (`btc_k`, `--btc-trades`: cancel every quote when |BTC 1 s or 5 s return| > k x the BTC
//! 1h sigma scaled to the horizon, re-quote after `btc_cooldown_s`), a liquidation-cluster pull (`liq_pull_min`, the
//! public-tape proxy `mm::LiqProxy` with `--liq-min-notional`) and a side pull on top-5 imbalance or 30 s taker flow
//! (`side_signal`). `latencies_ms` runs one quoter per latency in the same pass.
//!
//! WMQ quoting families (program WMM2, replay-only; every field defaults to the behaviour above): `mode` "probe" (an
//! intensity probe: one minimum-lot order per side at each distance `probe_x` x the quoter's 1h sigma, 0 = join the
//! touch, re-pegged with the live replace rule; per-distance live time and fills give the fill-intensity curve
//! lambda(delta)), "model" (`price_rule` "depth" = symmetric `depth_sigma` x sigma; "glft" = the Gueant-Lehalle-Fernandez
//! -Tapia closed-form asymptotic quotes; "as" = Avellaneda-Stoikov with a rolling horizon `as_tau_s`; plus `add_sigma` /
//! `add_bps`; sizes `lots` x the minimum lot with `size_rule` "flat" or "linear" (size skew); a symmetric `cap_usd`;
//! inventory unwinds `unwind_passive_s` (the reducing quote joins the touch), `unwind_aggr_s` (time limit: taker close)
//! and `unwind_stop_sigma` (taker close on an adverse move from the entry); a trend side pull `trend_k`; `gate_widen_sigma`
//! widens instead of going exit-only while the gate is off). Compact inputs: `--write-compact DIR` writes the book and
//! trades the replay read as gzip binary (lossless on the day's grid), `--compact DIR` replays them.
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
use at_core::num::{Round, round_to_step};
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
/// WMQ: a longer markout, reported next to the three above (the net decomposition still ends at 60 s).
const MARKOUT_LONG_S: i64 = 300;

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

/// Reads a public.bybit.com trade archive as a price tape (ms, price), stable-sorted by ms (the BTC tape of H-MMW2).
pub fn read_px_tape(path: &Path) -> anyhow::Result<Vec<(i64, f64)>> {
    let mut rdr = csv::Reader::from_reader(open_lines(path)?);
    let h = rdr.headers()?.clone();
    let col = |n: &str| h.iter().position(|x| x == n);
    let (c_ts, c_px) = (col("timestamp").context("no timestamp")?, col("price").context("no price")?);
    let mut out = Vec::new();
    let mut rec = csv::StringRecord::new();
    while rdr.read_record(&mut rec)? {
        let secs: f64 = rec[c_ts].parse()?;
        out.push(((secs * 1000.0).round() as i64, rec[c_px].parse::<f64>()?));
    }
    out.sort_by_key(|t| t.0);
    Ok(out)
}

/// Structural zones of one symbol from a `levels_file`-format JSON ({"SYM": [{"lo", "hi"}, ..]}, or a bare list).
pub fn read_zones(path: &Path, symbol: &str) -> anyhow::Result<Vec<(f64, f64)>> {
    let v: Value = serde_json::from_str(&std::fs::read_to_string(path).with_context(|| path.display().to_string())?)?;
    let list = if v.is_array() { &v } else { v.get(symbol).unwrap_or(&Value::Null) };
    let num = |x: &Value| x.as_f64().or_else(|| x.as_str().and_then(|s| s.parse().ok())).unwrap_or(f64::NAN);
    Ok(list.as_array().map(|a| a.iter().map(|z| (num(&z["lo"]), num(&z["hi"]))).filter(|z| z.0.is_finite() && z.1.is_finite()).collect()).unwrap_or_default())
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
/// Widen-factor rows (`valid_from_ms,factor`), sorted by time.
fn read_widen(path: &Path) -> anyhow::Result<Vec<(i64, f64)>> {
    let mut v = read_forecast(path)?;
    v.retain(|x| x.1.is_finite() && x.1 > 0.0);
    Ok(v)
}

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
    /// H-MMW2: the BTC tape (1 s / 5 s returns, 1h sigma) and the counts of BTC prints with z > 2 / 3 / 4.
    pub btc: mm::BtcTape,
    pub btc_prints: u64,
    pub btc_z_counts: [u64; 3],
    /// H-MMW1 A2: the liquidation-cluster proxy and its trigger times.
    pub liq: mm::LiqProxy,
    pub liq_events: Vec<i64>,
    /// Every print's notional (USD), for the day's median print (the next day's proxy threshold).
    pub print_notional: Vec<f32>,
    /// H-MMW4: touch sizes (lots) sampled every 10 s over the last 60 min, and their median (refreshed once a minute).
    touch_samples: VecDeque<(i64, f64)>,
    next_touch_sample: i64,
    touch_med: f64,
    touch_med_at: i64,
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
            btc: mm::BtcTape::new(1.0),
            btc_prints: 0,
            btc_z_counts: [0; 3],
            liq: mm::LiqProxy::new(30_000, 5, 3, 0.0),
            liq_events: vec![],
            print_notional: vec![],
            touch_samples: VecDeque::new(),
            next_touch_sample: i64::MIN,
            touch_med: f64::NAN,
            touch_med_at: i64::MIN / 4,
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
        if cts >= self.next_touch_sample
            && let (Some(b), Some(a)) = (self.book.best_bid(), self.book.best_ask())
        {
            self.touch_samples.push_back((cts, b.1 as f64));
            self.touch_samples.push_back((cts, a.1 as f64));
            while self.touch_samples.front().is_some_and(|x| x.0 < cts - 3_600_000) {
                self.touch_samples.pop_front();
            }
            self.next_touch_sample = cts.div_euclid(10_000) * 10_000 + 10_000;
        }
        if self.n_msgs % 20_000 == 1 {
            self.checks += 1;
            if self.rp.book() != self.book.snapshot(usize::MAX) {
                self.check_mismatch += 1;
            }
        }
        Ok(())
    }

    /// Median touch size (lots) of the samples before `now` (last 60 min), refreshed once a minute; NaN before any.
    pub fn touch_median(&mut self, now: i64) -> f64 {
        if now - self.touch_med_at >= MIN_MS {
            let v: Vec<f64> = self.touch_samples.iter().filter(|x| x.0 <= now && x.0 >= now - 3_600_000).map(|x| x.1).collect();
            self.touch_med = mm::median(&v);
            self.touch_med_at = now;
        }
        self.touch_med
    }

    /// One BTC print (H-MMW2): its move z.
    pub fn on_btc(&mut self, ts: i64, px: f64) -> f64 {
        let z = self.btc.on_print(ts, px);
        self.btc_prints += 1;
        for (k, c) in [2.0, 3.0, 4.0].iter().zip(self.btc_z_counts.iter_mut()) {
            *c += (z > *k) as u64;
        }
        z
    }

    /// A public trade: bars, flow history, the print-notional sample and the liquidation-cluster proxy (true = a cluster
    /// completed on this print).
    pub fn on_trade(&mut self, t: &Tr) -> bool {
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
        self.print_notional.push((q * t.px) as f32);
        let liq = self.liq.on_print(t.ts, t.tick, q * t.px, t.buy);
        if liq {
            self.liq_events.push(t.ts);
        }
        liq
    }

    /// Median print notional (USD) of the trades seen.
    pub fn median_print_notional(&self) -> f64 {
        let v: Vec<f64> = self.print_notional.iter().map(|x| *x as f64).collect();
        mm::median(&v)
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

/// How a quoter prices its orders.
#[derive(Clone, Copy, Debug, Deserialize, Serialize, Default, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum QuoteMode {
    /// The live `atx mm` rule (`mm::target_quotes`).
    #[default]
    Spread,
    /// H-MMW1 level ladder (`mm::ladder_levels`): minimum lots at structural zones `ladder_band` x 1h sigma from the mid,
    /// a passive reduce-only take-profit at the mid at fill time (`mm::ladder_tp`), a taker stop `ladder_stop_sigma` x
    /// the 1h sigma beyond the level (`mm::level_stop`).
    Ladder,
    /// H-MMW4 touch join: the live rule's sides, sizes and reduce-only flags, priced at the touch, joined only when the
    /// size ahead is < `touch_q` x the median touch size (`mm::short_queue`); cancelled when the size behind us falls by
    /// more than `touch_leave_frac` within `touch_leave_ms` (`mm::behind_thinned`).
    Touch,
    /// WMQ intensity probe: one minimum-lot order per side at each `probe_x` x sigma from the mid (0 = the touch);
    /// fills are independent events (no inventory logic, no loss stops, no pauses).
    Probe,
    /// WMQ quote rule (`model_quotes`): `price_rule`, `size_rule`, `cap_usd`, unwinds and the trend pull.
    Model,
}

/// WMQ price rule of mode "model" (half-spreads in units of the quoter's 1h sigma).
#[derive(Clone, Copy, Debug, Deserialize, Serialize, Default, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum ModelPrice {
    /// Symmetric `depth_sigma` on both sides, no price skew.
    #[default]
    Depth,
    /// Gueant, Lehalle & Fernandez-Tapia (2013) closed-form asymptotic quotes with lambda(x) = A exp(-k x).
    Glft,
    /// Avellaneda & Stoikov (2008) with a rolling horizon `as_tau_s`.
    As,
}

/// WMQ size rule of mode "model".
#[derive(Clone, Copy, Debug, Deserialize, Serialize, Default, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum SizeRule {
    /// `lots` on both sides (the reducing side capped at the position).
    #[default]
    Flat,
    /// Size skew: adding side round(lots x (1 - |q| / cap)), reducing side round(lots x (1 + |q| / cap)) (<= |q|).
    Linear,
}

/// H-MMW3 signal.
#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum SideSignal {
    /// Top-5 book imbalance in quantity (`mm::top_imbalance`).
    Imb5,
    /// Signed taker notional imbalance of the last 30 s (`mm::trade_imbalance`).
    Flow30,
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
    /// One quoter per latency (and queue model) in the same pass; empty = `--latency-ms`.
    #[serde(default)]
    pub latencies_ms: Vec<i64>,
    #[serde(default)]
    pub mode: QuoteMode,
    #[serde(default = "ladder_band")]
    pub ladder_band: (f64, f64),
    #[serde(default = "ladder_stop")]
    pub ladder_stop_sigma: f64,
    #[serde(default = "ladder_max")]
    pub ladder_max_per_side: usize,
    /// H-MMW1 A2: minutes without adding quotes after a liquidation-cluster proxy trigger (0 = off).
    #[serde(default)]
    pub liq_pull_min: f64,
    /// H-MMW2: cancel every quote when the BTC move z (`mm::btc_move_z`) exceeds `btc_k` (0 = off); no quotes for
    /// `btc_cooldown_s` after the latest trigger.
    #[serde(default)]
    pub btc_k: f64,
    #[serde(default)]
    pub btc_cooldown_s: f64,
    /// H-MMW3: withdraw (`side_widen_sigma` = 0) or widen the side the signal threatens beyond `side_threshold`.
    #[serde(default)]
    pub side_signal: Option<SideSignal>,
    #[serde(default)]
    pub side_threshold: f64,
    #[serde(default)]
    pub side_widen_sigma: f64,
    /// H-MMW4 (mode touch).
    #[serde(default)]
    pub touch_q: f64,
    #[serde(default = "half")]
    pub touch_leave_frac: f64,
    #[serde(default = "two_s")]
    pub touch_leave_ms: i64,
    /// WMQ mode "probe": distances in sigma units (0 = join the touch).
    #[serde(default)]
    pub probe_x: Vec<f64>,
    /// WMQ mode "model".
    #[serde(default)]
    pub price_rule: ModelPrice,
    #[serde(default = "one")]
    pub depth_sigma: f64,
    /// GLFT / AS risk aversion (per sigma unit and base order), intensity A (fills per second at the mid) and k (per sigma).
    #[serde(default)]
    pub glft_gamma: f64,
    #[serde(default)]
    pub glft_a: f64,
    #[serde(default)]
    pub glft_k: f64,
    #[serde(default = "hour_s")]
    pub as_tau_s: f64,
    /// Added to both half-spreads: sigma units and basis points (e.g. adverse selection and the maker fee).
    #[serde(default)]
    pub add_sigma: f64,
    #[serde(default)]
    pub add_bps: f64,
    /// Base order size in minimum lots.
    #[serde(default = "one_u")]
    pub lots: u32,
    #[serde(default)]
    pub size_rule: SizeRule,
    /// Symmetric inventory cap (USD after a fill; one base order from flat always allowed); 0 = the config's long / short caps.
    #[serde(default)]
    pub cap_usd: f64,
    /// After holding inventory this long (s, since it left flat or flipped) the reducing quote joins the touch; 0 = off.
    #[serde(default)]
    pub unwind_passive_s: f64,
    /// Time limit (s): close the position at market (taker); 0 = off.
    #[serde(default)]
    pub unwind_aggr_s: f64,
    /// Close at market when the mid moved this many sigma against the average entry; 0 = off.
    #[serde(default)]
    pub unwind_stop_sigma: f64,
    /// Trend pull: no adding bid when the `trend_window_min` log return < -k x sigma x sqrt(window / 1 h), no adding ask when
    /// > +k; 0 = off.
    #[serde(default)]
    pub trend_k: f64,
    #[serde(default = "sixty")]
    pub trend_window_min: i64,
    /// With a gate (mode "model"): while it is off, widen both half-spreads by this many sigma instead of exit-only; 0 = exit-only.
    #[serde(default)]
    pub gate_widen_sigma: f64,
    /// Widen series (CSV `valid_from_ms,factor`, piecewise constant, relative to the repo root; 1 before the first row):
    /// mode "model" multiplies both half-spreads by the factor, the live rule multiplies its quote sigma. None = off.
    #[serde(default)]
    pub widen_file: Option<String>,
}

impl Default for VariantSpec {
    fn default() -> Self {
        serde_json::from_value(json!({"name": "", "config": ""})).expect("defaults")
    }
}

fn ladder_band() -> (f64, f64) {
    (1.5, 3.5)
}

fn ladder_stop() -> f64 {
    1.5
}

fn ladder_max() -> usize {
    3
}

fn half() -> f64 {
    0.5
}

fn two_s() -> i64 {
    2_000
}

fn one() -> f64 {
    1.0
}

fn one_u() -> u32 {
    1
}

fn hour_s() -> f64 {
    3600.0
}

fn sixty() -> i64 {
    60
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

// ---------------------------------------------------------------- WMQ quote rule (pure)

/// Gueant, Lehalle & Fernandez-Tapia (2013) closed-form asymptotic half-spreads, in units of the 1h sigma, at inventory
/// `q` (base orders, long > 0), for fill intensity lambda(x) = `a` exp(-`k` x) (x = distance from the mid in sigma, `a`
/// per second) and a per-second price variance of 1/3600 sigma^2: bid = c1 + (2q + 1) / 2 c2, ask = c1 - (2q - 1) / 2 c2,
/// c1 = ln(1 + g / k) / g, c2 = sqrt(g / (2 k a) / 3600 x (1 + g / k)^(1 + k / g)). None unless g, a, k > 0.
pub fn glft_half_spreads(g: f64, a: f64, k: f64, q: f64) -> Option<(f64, f64)> {
    if !(g > 0.0 && a > 0.0 && k > 0.0) {
        return None;
    }
    let c1 = (1.0 + g / k).ln() / g;
    let c2 = (g / (2.0 * k * a) / 3600.0 * (1.0 + g / k).powf(1.0 + k / g)).sqrt();
    Some((c1 + (2.0 * q + 1.0) / 2.0 * c2, c1 - (2.0 * q - 1.0) / 2.0 * c2))
}

/// Avellaneda & Stoikov (2008) half-spreads (sigma units) with a rolling horizon `tau_s`: reservation shift -q g t and
/// total spread g t + 2 ln(1 + g / k) / g, t = tau_s / 3600 (the horizon's variance in sigma^2). None unless g, k, tau > 0.
pub fn as_half_spreads(g: f64, k: f64, tau_s: f64, q: f64) -> Option<(f64, f64)> {
    if !(g > 0.0 && k > 0.0 && tau_s > 0.0) {
        return None;
    }
    let t = tau_s / 3600.0;
    let h = (1.0 + g / k).ln() / g + 0.5 * g * t;
    Some((h + q * g * t, h - q * g * t))
}

/// What the WMQ quote rule sees (quantities in qty steps).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct ModelIn {
    pub bid1: f64,
    pub ask1: f64,
    /// 1h sigma (fraction).
    pub sigma: f64,
    /// Signed position, the minimum lot and the inventory cap.
    pub pos: i64,
    pub lot: i64,
    pub cap: i64,
    /// Seconds since the position left flat (or flipped).
    pub hold_s: f64,
    pub no_add_bid: bool,
    pub no_add_ask: bool,
    /// Extra half-spread (sigma units), e.g. a news gate.
    pub widen: f64,
    /// Multiplier of the half-spreads (0 or 1 = none), e.g. a news-intensity widen.
    pub mult: f64,
}

/// WMQ quotes (bid, ask): half-spreads from `price_rule` at inventory q = pos / (lots x lot), plus `add_sigma`,
/// `add_bps` and `widen`; never crossing (PostOnly: a skewed quote may sit inside the spread); the reducing side joins the touch once the
/// position is older than `unwind_passive_s`; sizes from `size_rule`; adding orders stay within `cap` after a full fill
/// (one base order from flat is always allowed); the reducing order is reduce-only and never exceeds the position.
pub fn model_quotes(spec: &VariantSpec, rule: &InstrumentRules, x: &ModelIn) -> (Option<Quote>, Option<Quote>) {
    #[allow(clippy::nonminimal_bool)]
    if !(x.bid1 > 0.0 && x.ask1 > x.bid1 && x.sigma > 0.0 && x.lot > 0) {
        return (None, None);
    }
    let tick = rule.tick_size;
    let mid = 0.5 * (x.bid1 + x.ask1);
    let l = spec.lots.max(1) as i64;
    let order = l * x.lot;
    let q = x.pos as f64 / order as f64;
    let hs = match spec.price_rule {
        ModelPrice::Depth => Some((spec.depth_sigma, spec.depth_sigma)),
        ModelPrice::Glft => glft_half_spreads(spec.glft_gamma, spec.glft_a, spec.glft_k, q),
        ModelPrice::As => as_half_spreads(spec.glft_gamma, spec.glft_k, spec.as_tau_s, q),
    };
    let Some((db, da)) = hs.filter(|h| h.0.is_finite() && h.1.is_finite()) else { return (None, None) };
    let add = spec.add_sigma + spec.add_bps / 1e4 / x.sigma + x.widen;
    let f = if x.mult > 0.0 { x.mult } else { 1.0 };
    let mut bid_px = round_to_step(mid * (1.0 - (db + add) * f * x.sigma), tick, Round::Down).min(x.ask1 - tick);
    let mut ask_px = round_to_step(mid * (1.0 + (da + add) * f * x.sigma), tick, Round::Up).max(x.bid1 + tick);
    if spec.unwind_passive_s > 0.0 && x.hold_s >= spec.unwind_passive_s {
        if x.pos > 0 {
            ask_px = ask_px.min(x.ask1);
        } else if x.pos < 0 {
            bid_px = bid_px.max(x.bid1);
        }
    }
    let abs = x.pos.abs();
    let frac = if x.cap > 0 { (abs as f64 / x.cap as f64).min(1.0) } else { 1.0 };
    let (n_add, n_red) = match spec.size_rule {
        SizeRule::Flat => (l, l),
        SizeRule::Linear => ((l as f64 * (1.0 - frac)).round() as i64, (l as f64 * (1.0 + frac)).round() as i64),
    };
    let mut add_q = n_add.max(0) * x.lot;
    while add_q > 0 && abs + add_q > x.cap && (abs != 0 || add_q > order) {
        add_q -= x.lot;
    }
    let red_q = (n_red.max(1) * x.lot).min(abs);
    let mk = |px: f64, n: i64, ro: bool| (n > 0).then_some(Quote { price: px, qty: n as f64 * rule.qty_step, reduce_only: ro });
    let bid = if x.pos < 0 { mk(bid_px, red_q, true) } else if x.no_add_bid { None } else { mk(bid_px, add_q, false) };
    let ask = if x.pos > 0 { mk(ask_px, red_q, true) } else if x.no_add_ask { None } else { mk(ask_px, add_q, false) };
    (bid, ask)
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
    /// 0 = a plain quote; TAG_ADD = a ladder adding order; > 0 = the take-profit of ladder lot `tag`.
    tag: i64,
    /// Touch mode: (time, lots behind us) observations of the last `touch_leave_ms`.
    bh: Vec<(i64, f64)>,
    /// The mid and the quoter's 1h sigma when the order was sent (probe distances).
    place_mid: f64,
    place_sigma: f64,
}

const TAG_ADD: i64 = -1;

/// One ladder position lot: filled at `entry`, stopped (taker) beyond `stop`, cleared passively at `tp`.
#[derive(Clone, Debug)]
struct Lot {
    id: i64,
    long: bool,
    lots: i64,
    tp: f64,
    stop: f64,
    closing: bool,
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
    /// Mode "probe": which distance filled, and the mid / sigma when the order was sent.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub probe: Option<ProbeTag>,
}

#[derive(Clone, Copy, Debug, Serialize, PartialEq)]
pub struct ProbeTag {
    pub j: usize,
    pub x: f64,
    pub place_mid: f64,
    pub place_sigma: f64,
}

/// Mode "probe": exposure of one distance and side.
#[derive(Clone, Copy, Debug, Default, Serialize)]
pub struct ProbeLeg {
    pub live_ms: i64,
    pub placed: u64,
    /// Sum of live ms x distance from the mid at sending (bp, and sigma units).
    pub live_bps_ms: f64,
    pub live_x_ms: f64,
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
    pub btc_pulls: u64,
    pub btc_pull_cancels: u64,
    pub liq_pulls: u64,
    pub side_pull_ticks: u64,
    pub touch_skips: u64,
    pub touch_leaves: u64,
    pub ladder_adds_placed: u64,
    pub ladder_lots: u64,
    pub ladder_tp_fills: u64,
    pub ladder_stops: u64,
    pub model_ticks: u64,
    pub model_passive_ticks: u64,
    pub model_unwind_time: u64,
    pub model_unwind_stop: u64,
    pub model_trend_bid_ticks: u64,
    pub model_trend_ask_ticks: u64,
    pub model_widen_ticks: u64,
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
    grid: Grid,
    zones: std::rc::Rc<Vec<(f64, f64)>>,
    lots: Vec<Lot>,
    next_lot: i64,
    /// Pending taker orders (due time, side, lots, kind, ladder lot id or 0).
    mkt: Vec<(i64, Side, i64, &'static str, i64)>,
    pull_until: i64,
    liq_until: i64,
    /// Mode "probe": exposure per distance and side (0 bid, 1 ask).
    probe: Vec<[ProbeLeg; 2]>,
    /// Mode "model": when the position left flat or flipped.
    hold_since: Option<i64>,
    /// (mid, sigma) stamped on orders sent this tick.
    place_ctx: (f64, f64),
    /// `VariantSpec::widen_file` rows (read once per file by the driver).
    widen: Option<std::rc::Rc<Vec<(i64, f64)>>>,
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
        let n_probe = spec.probe_x.len();
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
            sigma_at: i64::MIN / 4,
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
            grid: Grid::new(rules.tick_size, rules.qty_step).expect("instrument grid"),
            zones: std::rc::Rc::new(vec![]),
            lots: vec![],
            next_lot: 1,
            mkt: vec![],
            pull_until: i64::MIN,
            liq_until: i64::MIN,
            probe: vec![[ProbeLeg::default(); 2]; n_probe],
            hold_since: None,
            place_ctx: (f64::NAN, f64::NAN),
            widen: None,
        }
    }

    /// Attach the widen series (`VariantSpec::widen_file`).
    pub fn set_widen(&mut self, rows: std::rc::Rc<Vec<(i64, f64)>>) {
        self.widen = Some(rows);
    }

    /// The widen factor at `now` (1 without a series or before its first row).
    fn widen_at(&self, now: i64) -> f64 {
        let Some(w) = &self.widen else { return 1.0 };
        let i = w.partition_point(|x| x.0 <= now);
        if i == 0 { 1.0 } else { w[i - 1].1.max(1e-9) }
    }

    /// Attach the structural zones of the day (`--zones`, read once by the driver).
    pub fn set_zones(&mut self, z: std::rc::Rc<Vec<(f64, f64)>>) {
        self.zones = z;
    }

    pub fn latency(&self) -> i64 {
        self.latency
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
        self.place_tagged(side, price, qty, reduce_only, now, grid, 0)
    }

    #[allow(clippy::too_many_arguments)]
    fn place_tagged(&mut self, side: Side, price: f64, qty: f64, reduce_only: bool, now: i64, grid: &Grid, tag: i64) -> bool {
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
            tag,
            bh: vec![],
            place_mid: self.place_ctx.0,
            place_sigma: self.place_ctx.1,
        });
        if self.spec.mode == QuoteMode::Probe
            && let Some(leg) = usize::try_from(tag - 1).ok().and_then(|j| self.probe.get_mut(j))
        {
            leg[(side == Side::Sell) as usize].placed += 1;
        }
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
            if self.spec.mode == QuoteMode::Probe
                && let Some(leg) = usize::try_from(o.tag - 1).ok().and_then(|j| self.probe.get_mut(j))
            {
                let l = &mut leg[(o.side == Side::Sell) as usize];
                let d = (o.price - o.place_mid).abs() / o.place_mid;
                l.live_ms += live;
                if d.is_finite() {
                    l.live_bps_ms += live as f64 * d * 1e4;
                    l.live_x_ms += live as f64 * d / o.place_sigma.max(1e-12);
                }
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
    fn book_fill(&mut self, ts: i64, side: Side, price: f64, lots: i64, maker: bool, kind: &'static str, reduce_only: bool, ahead0: f64, mid: f64, tag: i64) {
        if lots <= 0 {
            return;
        }
        let s = side.sign();
        if self.spec.mode == QuoteMode::Ladder {
            self.ladder_on_fill(ts, side, price, lots, mid, tag);
        }
        let notional = self.rules.qty_step * lots as f64 * price;
        let fee = notional * if maker { self.fees.maker_bps } else { self.fees.taker_bps } / 1e4;
        self.cash -= s as f64 * notional + fee;
        self.fee_paid += fee;
        let before = self.pos;
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
        if self.pos == 0 {
            self.hold_since = None;
        } else if before == 0 || before.signum() != self.pos.signum() {
            self.hold_since = Some(ts);
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
        self.fills.push(Fill { ts, side: side_str(side), price, qty: q, maker, kind, reduce_only, fee, mid, ahead0_lots: ahead0, pos_after: self.position_qty(), probe: None });
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
        let tag = self.orders[i].tag;
        self.book_fill(ts, side, price, f, true, kind, ro, a0, mid, tag);
        if self.spec.mode == QuoteMode::Probe {
            let o = &self.orders[i];
            let j = usize::try_from(tag - 1).unwrap_or(usize::MAX);
            let x = self.spec.probe_x.get(j).copied().unwrap_or(f64::NAN);
            let pt = ProbeTag { j, x, place_mid: o.place_mid, place_sigma: o.place_sigma };
            if let Some(last) = self.fills.last_mut() {
                last.probe = Some(pt);
            }
            // one fill event per probe order: the rest (if any) goes, a new order is sent next tick
            if !done {
                self.close_order(i, ts);
            }
        }
        f
    }

    /// Ladder lot bookkeeping of one fill (before the position update): the reducing part closes lot `tag` first, then the
    /// oldest lots; the adding part opens a lot with its stop and places its take-profit (sent now).
    fn ladder_on_fill(&mut self, ts: i64, side: Side, price: f64, lots: i64, mid: f64, tag: i64) {
        let s = side.sign();
        let reduce = if self.pos != 0 && self.pos.signum() != s { lots.min(self.pos.abs()) } else { 0 };
        let mut left = reduce;
        if tag > 0
            && let Some(l) = self.lots.iter_mut().find(|l| l.id == tag)
        {
            let x = left.min(l.lots);
            l.lots -= x;
            left -= x;
            self.funnel.ladder_tp_fills += (x > 0) as u64;
        }
        for l in self.lots.iter_mut() {
            let x = left.min(l.lots);
            l.lots -= x;
            left -= x;
        }
        self.lots.retain(|l| l.lots > 0);
        let add = lots - reduce;
        if add <= 0 {
            return;
        }
        let long = s > 0;
        let tick = self.rules.tick_size;
        let id = self.next_lot;
        self.next_lot += 1;
        let tp = mm::ladder_tp(price, mid, long, tick);
        let stop = mm::level_stop(price, long, self.spec.ladder_stop_sigma, self.st.sigma);
        self.lots.push(Lot { id, long, lots: add, tp, stop, closing: false });
        self.funnel.ladder_lots += 1;
        // holding inventory: the other side's adding orders go (they would net against the lot)
        for i in 0..self.orders.len() {
            if self.orders[i].tag == TAG_ADD && self.orders[i].side != side {
                self.cancel_idx(i, ts);
            }
        }
        let g = self.grid;
        let q = g.qty(add);
        self.place_tagged(side.opposite(), tp, q, true, ts, &g, id);
    }

    /// Ladder stops on a print: a lot whose stop the print reached is closed at market (taker) one latency later; its
    /// take-profit is cancelled.
    fn ladder_stops(&mut self, px: f64, ts: i64) {
        for k in 0..self.lots.len() {
            let l = &self.lots[k];
            if l.closing || !(if l.long { px <= l.stop } else { px >= l.stop }) {
                continue;
            }
            let (id, long, n) = (l.id, l.long, l.lots);
            self.lots[k].closing = true;
            for i in 0..self.orders.len() {
                if self.orders[i].tag == id {
                    self.cancel_idx(i, ts);
                }
            }
            self.mkt.push((ts + self.latency, if long { Side::Sell } else { Side::Buy }, n, "stop", id));
            self.funnel.ladder_stops += 1;
        }
    }

    /// H-MMW2: a BTC print with move z: beyond `btc_k`, cancel every quote now and quote nothing for the cooldown.
    pub fn on_btc(&mut self, ts: i64, z: f64) {
        if self.spec.btc_k <= 0.0 || z <= self.spec.btc_k || self.stopped.is_some() {
            return;
        }
        if ts >= self.pull_until {
            self.funnel.btc_pulls += 1;
        }
        self.pull_until = self.pull_until.max(ts + (self.spec.btc_cooldown_s * 1000.0) as i64);
        for i in 0..self.orders.len() {
            if !self.orders[i].closed && self.orders[i].cancel_ms.is_none() {
                self.funnel.btc_pull_cancels += 1;
                self.cancel_idx(i, ts);
            }
        }
    }

    /// H-MMW1 A2: a liquidation cluster: cancel the adding quotes now and add nothing for `liq_pull_min`.
    pub fn on_liq(&mut self, ts: i64) {
        if self.spec.liq_pull_min <= 0.0 || self.stopped.is_some() {
            return;
        }
        if ts >= self.liq_until {
            self.funnel.liq_pulls += 1;
        }
        self.liq_until = self.liq_until.max(ts + (self.spec.liq_pull_min * 60_000.0) as i64);
        for i in 0..self.orders.len() {
            if !self.orders[i].reduce_only {
                self.cancel_idx(i, ts);
            }
        }
    }

    /// A taker order of `lots` on `side` against the book, reduce-only (capped at the position; a ladder lot `tag` is
    /// capped at that lot); an empty book side closes at the mid.
    fn market_order(&mut self, now: i64, side: Side, lots: i64, kind: &'static str, tag: i64, m: &Market) {
        let mut left = lots.min(self.reducible(side));
        if tag > 0 {
            left = left.min(self.lots.iter().find(|l| l.id == tag).map_or(0, |l| l.lots));
        }
        if left <= 0 {
            return;
        }
        let (bids, asks) = m.levels(usize::MAX);
        let book = if side == Side::Sell { bids } else { asks };
        let mid = m.mid().unwrap_or(self.avg);
        let mut fills = vec![];
        for (p, q) in book {
            if left == 0 {
                break;
            }
            let n = m.grid().lots_nearest(q).min(left);
            fills.push((p, n));
            left -= n;
        }
        if left > 0 {
            fills.push((mid, left));
        }
        for (p, n) in fills {
            self.book_fill(now, side, p, n, false, kind, true, f64::NAN, mid, tag);
        }
    }

    /// Fire every timer strictly before `t` (activations, effective cancels, the stop flatten, decision ticks), in order.
    pub fn advance(&mut self, t: i64, m: &mut Market) {
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
            for x in &self.mkt {
                next = next.min(x.0);
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
            if self.mkt.iter().any(|x| x.0 <= next) {
                let due: Vec<_> = self.mkt.iter().filter(|x| x.0 <= next).cloned().collect();
                self.mkt.retain(|x| x.0 > next);
                for (t, side, n, kind, tag) in due {
                    self.market_order(t, side, n, kind, tag, m);
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
        if !self.lots.is_empty() {
            self.ladder_stops(tr.px, tr.ts);
        }
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
                self.touch_leave(i, tr.ts);
            }
        }
    }

    /// Touch mode: record the size behind order `i` and cancel it when that queue thinned (`mm::behind_thinned`).
    fn touch_leave(&mut self, i: usize, now: i64) {
        if self.spec.mode != QuoteMode::Touch {
            return;
        }
        let (w, frac) = (self.spec.touch_leave_ms, self.spec.touch_leave_frac);
        let o = &mut self.orders[i];
        if o.closed || !o.active || o.cancel_ms.is_some() || !o.q.known {
            return;
        }
        o.bh.retain(|x| x.0 >= now - w);
        o.bh.push((now, o.q.behind));
        if mm::behind_thinned(&o.bh, now, w, frac) {
            self.funnel.touch_leaves += 1;
            self.cancel_idx(i, now);
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
                self.touch_leave(i, cts);
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
            self.book_fill(now, side, p, lots, false, kind, true, f64::NAN, mid, 0);
        }
        self.lots.clear();
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
    fn tick(&mut self, now: i64, m: &mut Market) {
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
        if self.spec.mode == QuoteMode::Probe {
            self.tick_probe(now, &g, (bt, at), mid);
            self.tick_n += 1;
            return;
        }
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
        if now < self.pull_until {
            // H-MMW2 pull: every quote stays cancelled until the cooldown ends
            for i in 0..self.orders.len() {
                self.cancel_idx(i, now);
            }
            self.gate_was_on = !gate_off;
            self.tick_n += 1;
            return;
        }
        if gate_off && self.spec.gate_flatten && self.gate_was_on && self.pos != 0 && self.flatten_ms.is_none() {
            for i in 0..self.orders.len() {
                self.cancel_idx(i, now);
            }
            self.flatten_ms = Some(now + cfg.order_pause_ms as i64 + self.latency);
            self.funnel.gate_flattens += 1;
        }
        self.gate_was_on = !gate_off;
        let exit_only = gate_off || self.spec.deadline_ms.is_some_and(|d| now >= d - cfg.exit_window_min * MIN_MS);
        let liq_paused = now < self.liq_until;
        if self.spec.mode == QuoteMode::Ladder {
            self.tick_ladder(now, bid1, ask1, exit_only || liq_paused || now < e.paused_until_ms);
            self.tick_n += 1;
            return;
        }
        if self.spec.mode == QuoteMode::Model {
            let deadline = self.spec.deadline_ms.is_some_and(|d| now >= d - cfg.exit_window_min * MIN_MS);
            self.tick_model(now, m, &g, (bid1, ask1), gate_off, deadline, liq_paused || now < e.paused_until_ms);
            self.tick_n += 1;
            return;
        }
        let pos = (self.position_qty(), if self.pos == 0 { 0.0 } else { self.avg });
        let mut x = mm::symbol_inputs(&cfg, &self.symbol, &e, bid1, ask1, pos, Others::default(), exit_only, now < e.paused_until_ms || liq_paused, guard);
        match self.spec.sigma_mode {
            SigmaMode::Config => {}
            SigmaMode::Fixed => x.sigma = self.spec.fixed_bps / 1e4,
            SigmaMode::Forecast => match self.forecast_now(now) {
                Some(s) => x.sigma = self.spec.forecast_scale * s,
                None => self.funnel.forecast_missing_ticks += 1,
            },
        }
        let wf = self.widen_at(now);
        if wf != 1.0 {
            x.sigma *= wf;
            self.funnel.model_widen_ticks += 1;
        }
        let (mut bid, mut ask) = mm::target_quotes(&cfg, &self.rules, &x, &e.levels);
        if let Some(sig) = self.spec.side_signal {
            let s = match sig {
                SideSignal::Imb5 => {
                    let (b5, a5) = m.levels(5);
                    mm::top_imbalance(&b5, &a5, 5)
                }
                SideSignal::Flow30 => {
                    let tr: Vec<(i64, f64, f64)> = m.recent(1000, now).filter(|t| t.ts >= now - 30_000).map(|t| (t.ts, if t.buy { 1.0 } else { -1.0 }, t.px * g.qty(t.lots))).collect();
                    mm::trade_imbalance(&tr, now - 30_000)
                }
            };
            let th = mm::threatened_side(s, self.spec.side_threshold);
            self.funnel.side_pull_ticks += th.is_some() as u64;
            (bid, ask) = mm::side_pull(bid, ask, th, self.spec.side_widen_sigma, x.sigma, self.rules.tick_size);
        }
        if self.spec.mode == QuoteMode::Touch {
            let med = m.touch_median(now);
            self.tick_touch(now, &g, (bt, at), bid, ask, med, m);
            self.tick_n += 1;
            return;
        }
        self.manage_quotes(now, &g, mid, x.sigma, bid, ask);
        self.tick_n += 1;
    }

    /// One resting quote per side: extra ones go, the first is replaced when `mm::needs_replace` says so, a missing one
    /// is sent (order operations throttled by `order_pause_ms`, as the live loop).
    fn manage_quotes(&mut self, now: i64, g: &Grid, mid: f64, sigma: f64, bid: Option<Quote>, ask: Option<Quote>) {
        let cfg = self.cfg.clone();
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
            let replace = mm::needs_replace(&cfg, &self.rules, want, rest.as_ref(), mid, sigma, now);
            if let (true, Some(i)) = (replace, cur) {
                self.cancel_idx(i, t_op);
                t_op += pause;
            }
            if let (Some(q), true) = (want, cur.is_none() || replace) {
                let Quote { price, qty, reduce_only } = q;
                self.place(side, price, qty, reduce_only, t_op, g);
                t_op += pause;
            }
        }
    }

    /// WMQ probe decision: for every distance x and side, one minimum-lot order at x sigma from the mid (x = 0: at the
    /// touch, moved as soon as the touch moves), kept while `mm::needs_replace` allows (the live replace rule).
    fn tick_probe(&mut self, now: i64, g: &Grid, top: (i64, i64), mid: f64) {
        let sig = self.st.sigma;
        if sig.is_nan() || sig <= 0.0 {
            return;
        }
        let cfg = self.cfg.clone();
        let tick = self.rules.tick_size;
        let (bid1, ask1) = (g.price(top.0), g.price(top.1));
        let lot = mm::min_lot(&self.rules, mid);
        self.place_ctx = (mid, sig);
        let xs = self.spec.probe_x.clone();
        for (j, &x) in xs.iter().enumerate() {
            let tag = j as i64 + 1;
            for side in [Side::Buy, Side::Sell] {
                let px = match (x > 0.0, side) {
                    (false, Side::Buy) => bid1,
                    (false, Side::Sell) => ask1,
                    (true, Side::Buy) => round_to_step(mid * (1.0 - x * sig), tick, Round::Down).min(ask1 - tick),
                    (true, Side::Sell) => round_to_step(mid * (1.0 + x * sig), tick, Round::Up).max(bid1 + tick),
                };
                let want = Quote { price: px, qty: lot, reduce_only: false };
                let cur = (0..self.orders.len()).find(|&i| {
                    let o = &self.orders[i];
                    !o.closed && o.cancel_ms.is_none() && o.tag == tag && o.side == side
                });
                let replace = cur.is_some_and(|i| {
                    let o = &self.orders[i];
                    if x > 0.0 {
                        let rest = Resting { price: o.price, leaves_qty: g.qty(o.leaves), reduce_only: false, updated_ms: o.updated_ms };
                        mm::needs_replace(&cfg, &self.rules, Some(want), Some(&rest), mid, sig, now)
                    } else {
                        o.tick != if side == Side::Buy { top.0 } else { top.1 } || (g.qty(o.leaves) - lot).abs() > self.rules.qty_step * 0.5
                    }
                });
                if let (true, Some(i)) = (replace, cur) {
                    self.cancel_idx(i, now);
                }
                if cur.is_none() || replace {
                    self.place_tagged(side, px, lot, false, now, g, tag);
                }
            }
        }
    }

    /// WMQ model decision: aggressive unwinds (time limit, adverse-move stop: cancel everything, close at market one
    /// latency later), the trend pull, then `model_quotes` through the live replace rule.
    #[allow(clippy::too_many_arguments)]
    fn tick_model(&mut self, now: i64, m: &Market, g: &Grid, top: (f64, f64), gate_off: bool, deadline: bool, paused: bool) {
        let (bid1, ask1) = top;
        let mid = 0.5 * (bid1 + ask1);
        let sig = self.st.sigma;
        if sig.is_nan() || sig <= 0.0 {
            return;
        }
        self.funnel.model_ticks += 1;
        let lot = g.lots_nearest(mm::min_lot(&self.rules, mid)).max(1);
        let cap_usd = if self.spec.cap_usd > 0.0 { self.spec.cap_usd } else { self.cfg.max_long_usd.min(self.cfg.max_short_usd) };
        let cap = (cap_usd / (mid * self.step())).floor() as i64;
        let hold_s = self.hold_since.map_or(0.0, |t| (now - t) as f64 / 1000.0);
        if !self.mkt.is_empty() {
            return; // an unwind is on its way
        }
        if self.pos != 0 {
            let long = self.pos > 0;
            let timed = self.spec.unwind_aggr_s > 0.0 && hold_s >= self.spec.unwind_aggr_s;
            let k = self.spec.unwind_stop_sigma;
            let stop = k > 0.0 && self.avg > 0.0 && if long { mid <= self.avg * (1.0 - k * sig) } else { mid >= self.avg * (1.0 + k * sig) };
            if timed || stop {
                for i in 0..self.orders.len() {
                    self.cancel_idx(i, now);
                }
                self.mkt.push((now + self.latency, if long { Side::Sell } else { Side::Buy }, self.pos.abs(), "unwind", 0));
                if stop {
                    self.funnel.model_unwind_stop += 1;
                } else {
                    self.funnel.model_unwind_time += 1;
                }
                return;
            }
        }
        let (mut no_bid, mut no_ask) = (false, false);
        if self.spec.trend_k > 0.0 {
            let w = self.spec.trend_window_min.max(1);
            let c = m.closes(now - (w + 1) * MIN_MS, now);
            if c.len() >= 2 && c[0] > 0.0 {
                let r = (c[c.len() - 1] / c[0]).ln();
                let sd = sig * (w as f64 / 60.0).sqrt();
                no_bid = r < -self.spec.trend_k * sd;
                no_ask = r > self.spec.trend_k * sd;
                self.funnel.model_trend_bid_ticks += no_bid as u64;
                self.funnel.model_trend_ask_ticks += no_ask as u64;
            }
        }
        let widen = if gate_off && self.spec.gate_widen_sigma > 0.0 { self.spec.gate_widen_sigma } else { 0.0 };
        self.funnel.model_widen_ticks += (widen > 0.0) as u64;
        let no_add = deadline || paused || (gate_off && self.spec.gate_widen_sigma <= 0.0);
        let mult = self.widen_at(now);
        self.funnel.model_widen_ticks += (mult != 1.0) as u64;
        let x = ModelIn { bid1, ask1, sigma: sig, pos: self.pos, lot, cap, hold_s, no_add_bid: no_add || no_bid, no_add_ask: no_add || no_ask, widen, mult };
        self.funnel.model_passive_ticks += (self.pos != 0 && self.spec.unwind_passive_s > 0.0 && hold_s >= self.spec.unwind_passive_s) as u64;
        let (bid, ask) = model_quotes(&self.spec, &self.rules, &x);
        self.place_ctx = (mid, sig);
        self.manage_quotes(now, g, mid, sig, bid, ask);
    }

    /// H-MMW1 ladder decision: take-profits of open lots re-sent when missing (post-only safe at the touch), adding
    /// orders at the qualifying levels within the caps (resting adds counted as filled; one lot from flat), stale adds
    /// cancelled.
    fn tick_ladder(&mut self, now: i64, bid1: f64, ask1: f64, no_adds: bool) {
        let cfg = self.cfg.clone();
        let g = self.grid;
        let tick = self.rules.tick_size;
        let pause = cfg.order_pause_ms as i64;
        let mut t_op = now;
        let mid = 0.5 * (bid1 + ask1);
        // take-profits
        let lots: Vec<(i64, bool, i64, f64)> = self.lots.iter().filter(|l| !l.closing).map(|l| (l.id, l.long, l.lots, l.tp)).collect();
        for (id, long, n, tp) in lots {
            if self.orders.iter().any(|o| !o.closed && o.cancel_ms.is_none() && o.tag == id) {
                continue;
            }
            let px = if long { tp.max(bid1 + tick) } else { tp.min(ask1 - tick) };
            self.place_tagged(if long { Side::Sell } else { Side::Buy }, px, g.qty(n), true, t_op, &g, id);
            t_op += pause;
        }
        // adding orders
        let (sup, res) = if no_adds { (vec![], vec![]) } else { mm::ladder_levels(&self.zones, mid, self.st.sigma, self.spec.ladder_band) };
        let lot = mm::min_lot(&self.rules, mid);
        let inv = self.position_qty() * mid;
        for (side, levels) in [(Side::Buy, sup), (Side::Sell, res)] {
            let adding = match side {
                Side::Buy => self.pos >= 0,
                Side::Sell => self.pos <= 0,
            };
            let cap = if side == Side::Buy { cfg.max_long_usd } else { cfg.max_short_usd };
            let mut want: Vec<i64> = vec![];
            if adding {
                let mut used = inv.abs();
                for px in levels.into_iter().take(self.spec.ladder_max_per_side) {
                    let p = round_to_step(px, tick, if side == Side::Buy { Round::Down } else { Round::Up });
                    let ok_px = if side == Side::Buy { p < ask1 } else { p > bid1 };
                    let n = lot * p;
                    if !ok_px || !(used + n <= cap || (self.pos == 0 && want.is_empty())) {
                        break;
                    }
                    used += n;
                    if let Some(t) = g.ticks(p) {
                        want.push(t);
                    }
                }
            }
            let resting: Vec<usize> = (0..self.orders.len()).filter(|&i| !self.orders[i].closed && self.orders[i].cancel_ms.is_none() && self.orders[i].tag == TAG_ADD && self.orders[i].side == side).collect();
            for &i in &resting {
                if !want.contains(&self.orders[i].tick) {
                    self.cancel_idx(i, t_op);
                    t_op += pause;
                }
            }
            for t in want {
                if resting.iter().any(|&i| self.orders[i].tick == t) {
                    continue;
                }
                if self.place_tagged(side, g.price(t), lot, false, t_op, &g, TAG_ADD) {
                    self.funnel.ladder_adds_placed += 1;
                }
                t_op += pause;
            }
        }
    }

    /// H-MMW4 touch decision: each wanted quote (side, size, reduce-only from the live rule) priced at the touch; a resting
    /// quote stays while it is at the touch and of the right kind, otherwise it goes; a new one joins only a short queue.
    #[allow(clippy::too_many_arguments)]
    fn tick_touch(&mut self, now: i64, g: &Grid, top: (i64, i64), bid: Option<Quote>, ask: Option<Quote>, med: f64, m: &Market) {
        let pause = self.cfg.order_pause_ms as i64;
        let mut t_op = now;
        for (side, want, touch) in [(Side::Buy, bid, top.0), (Side::Sell, ask, top.1)] {
            let resting: Vec<usize> = (0..self.orders.len()).filter(|&i| !self.orders[i].closed && self.orders[i].cancel_ms.is_none() && self.orders[i].side == side).collect();
            let mut keep = None;
            for &i in &resting {
                let o = &self.orders[i];
                let ok = keep.is_none() && want.is_some_and(|q| o.tick == touch && o.reduce_only == q.reduce_only && (g.qty(o.leaves) - q.qty).abs() <= self.rules.qty_step * 0.5);
                if ok {
                    keep = Some(i);
                } else {
                    self.cancel_idx(i, t_op);
                    t_op += pause;
                }
            }
            let Some(q) = want else { continue };
            if keep.is_some() {
                continue;
            }
            let ahead = m.book.level_qty(side, touch) as f64;
            if !mm::short_queue(ahead, med, self.spec.touch_q) {
                self.funnel.touch_skips += 1;
                continue;
            }
            self.place(side, g.price(touch), q.qty, q.reduce_only, t_op, g);
            t_op += pause;
        }
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
        let mut adv_long = 0.0;
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
            adv_long += s * f.qty * (m.mid_at((f.ts + MARKOUT_LONG_S * 1000).min(end)).unwrap_or(m0) - m0);
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
        let mut out = json!({
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
        });
        out["latency_ms"] = json!(self.latency);
        out["mode"] = json!(self.spec.mode);
        out["pnl"]["adverse_300s"] = json!(adv_long);
        if self.spec.mode == QuoteMode::Model {
            out["model"] = json!({"ticks": fu.model_ticks, "passive_ticks": fu.model_passive_ticks, "unwind_time": fu.model_unwind_time,
                                  "unwind_stop": fu.model_unwind_stop, "trend_bid_ticks": fu.model_trend_bid_ticks,
                                  "trend_ask_ticks": fu.model_trend_ask_ticks, "widen_ticks": fu.model_widen_ticks});
        }
        if self.spec.mode == QuoteMode::Probe {
            let mut nf = vec![[0u64; 2]; self.probe.len()];
            for f in &self.fills {
                if let Some(p) = f.probe
                    && p.j < nf.len()
                {
                    nf[p.j][(f.side == "Sell") as usize] += 1;
                }
            }
            let legs: Vec<Value> = self
                .probe
                .iter()
                .enumerate()
                .flat_map(|(j, l)| {
                    let x = self.spec.probe_x[j];
                    let nf = &nf;
                    (0..2).map(move |s| {
                        let g = &l[s];
                        json!({"j": j, "x": x, "side": if s == 0 { "Buy" } else { "Sell" }, "live_ms": g.live_ms, "placed": g.placed, "fills": nf[j][s],
                               "dist_bps_mean": if g.live_ms > 0 { g.live_bps_ms / g.live_ms as f64 } else { f64::NAN },
                               "x_eff_mean": if g.live_ms > 0 { g.live_x_ms / g.live_ms as f64 } else { f64::NAN }})
                    })
                })
                .collect();
            out["probe"] = json!(legs);
        }
        let mmw = json!({"btc_pulls": fu.btc_pulls, "btc_pull_cancels": fu.btc_pull_cancels, "liq_pulls": fu.liq_pulls, "side_pull_ticks": fu.side_pull_ticks,
                         "touch_skips": fu.touch_skips, "touch_leaves": fu.touch_leaves, "ladder_adds_placed": fu.ladder_adds_placed,
                         "ladder_lots": fu.ladder_lots, "ladder_tp_fills": fu.ladder_tp_fills, "ladder_stops": fu.ladder_stops,
                         "widen_ticks": fu.model_widen_ticks});
        if let (Some(f), Some(x)) = (out["funnel"].as_object_mut(), mmw.as_object()) {
            f.extend(x.clone());
        }
        out
    }
}

thread_local! {
    /// The external 1h sigma forecast (single-threaded replay; set once per run).
    static FORECAST: std::cell::RefCell<std::rc::Rc<Vec<(i64, f64)>>> = std::cell::RefCell::new(std::rc::Rc::new(vec![]));
}

// ---------------------------------------------------------------- compact inputs (WMQ)

const COMPACT_FORMAT: &str = "wmq-compact-1";

fn put_uv(w: &mut impl Write, mut x: u64) -> std::io::Result<()> {
    let mut b = [0u8; 10];
    let mut n = 0;
    loop {
        let c = (x & 0x7f) as u8;
        x >>= 7;
        if x == 0 {
            b[n] = c;
            n += 1;
            break;
        }
        b[n] = c | 0x80;
        n += 1;
    }
    w.write_all(&b[..n])
}

fn zz(x: i64) -> u64 {
    ((x << 1) ^ (x >> 63)) as u64
}

fn unzz(u: u64) -> i64 {
    ((u >> 1) as i64) ^ -((u & 1) as i64)
}

/// A varint; None at a clean end of the stream (no byte read).
fn get_uv(r: &mut impl Read) -> anyhow::Result<Option<u64>> {
    let (mut x, mut shift) = (0u64, 0u32);
    let mut b = [0u8; 1];
    loop {
        if r.read(&mut b)? == 0 {
            anyhow::ensure!(shift == 0, "compact: truncated varint");
            return Ok(None);
        }
        x |= ((b[0] & 0x7f) as u64) << shift;
        if b[0] & 0x80 == 0 {
            return Ok(Some(x));
        }
        shift += 7;
        anyhow::ensure!(shift < 64, "compact: bad varint");
    }
}

fn need_uv(r: &mut impl Read) -> anyhow::Result<u64> {
    get_uv(r)?.context("compact: truncated record")
}

type GzOut = flate2::write::GzEncoder<BufWriter<File>>;
type GzIn = BufReader<MultiGzDecoder<BufReader<File>>>;

fn compact_create(path: &Path, kind: &str, symbol: &str, grid: Option<&Grid>) -> anyhow::Result<GzOut> {
    if let Some(d) = path.parent() {
        std::fs::create_dir_all(d)?;
    }
    let f = File::create(path).with_context(|| path.display().to_string())?;
    let mut w = flate2::write::GzEncoder::new(BufWriter::with_capacity(1 << 20, f), flate2::Compression::new(6));
    let h = json!({"format": COMPACT_FORMAT, "kind": kind, "symbol": symbol, "tick_size": grid.map(|g| g.tick_size()), "qty_step": grid.map(|g| g.qty_step())});
    writeln!(w, "{h}")?;
    Ok(w)
}

/// Opens a compact file and checks its header (kind, and the grid it was written on).
fn compact_open(path: &Path, kind: &str, grid: Option<&Grid>) -> anyhow::Result<GzIn> {
    let f = File::open(path).with_context(|| path.display().to_string())?;
    let mut r = BufReader::with_capacity(1 << 20, MultiGzDecoder::new(BufReader::with_capacity(1 << 20, f)));
    let mut line = String::new();
    r.read_line(&mut line)?;
    let h: Value = serde_json::from_str(line.trim()).with_context(|| format!("{}: compact header", path.display()))?;
    anyhow::ensure!(h["format"] == COMPACT_FORMAT && h["kind"] == kind, "{}: not a {kind} file of {COMPACT_FORMAT}", path.display());
    if let Some(g) = grid {
        let same = |a: &Value, b: f64| a.as_f64().is_some_and(|a| (a - b).abs() <= 1e-12 * b.abs());
        anyhow::ensure!(same(&h["tick_size"], g.tick_size()) && same(&h["qty_step"], g.qty_step()), "{}: written on grid {} / {}, the day's grid is {} / {}", path.display(), h["tick_size"], h["qty_step"], g.tick_size(), g.qty_step());
    }
    Ok(r)
}

/// Writes book messages as they are replayed: per message zigzag(cts - previous cts), kind (0 delta, 1 snapshot), then
/// per side the level count and per level zigzag(tick - previous level's tick) and the size in lots (0 = remove).
pub struct CompactBookWriter {
    w: GzOut,
    grid: Grid,
    last: i64,
}

impl CompactBookWriter {
    pub fn create(path: &Path, symbol: &str, grid: &Grid) -> anyhow::Result<Self> {
        Ok(Self { w: compact_create(path, "book", symbol, Some(grid))?, grid: *grid, last: 0 })
    }

    pub fn put(&mut self, cts: i64, msg: &Value) -> anyhow::Result<()> {
        let kind = match msg["type"].as_str() {
            Some("snapshot") => 1u8,
            Some("delta") => 0u8,
            other => bail!("compact: unknown message type {other:?}"),
        };
        put_uv(&mut self.w, zz(cts.wrapping_sub(self.last)))?;
        self.last = cts;
        self.w.write_all(&[kind])?;
        let num = |v: &Value| match v {
            Value::String(s) => s.parse::<f64>().ok(),
            v => v.as_f64(),
        };
        for key in ["b", "a"] {
            let lv = msg["data"][key].as_array().map(|a| a.as_slice()).unwrap_or(&[]);
            put_uv(&mut self.w, lv.len() as u64)?;
            let mut prev = 0i64;
            for x in lv {
                let t = num(&x[0]).and_then(|p| self.grid.ticks(p)).with_context(|| format!("compact: price {} off the tick grid", x[0]))?;
                let s = num(&x[1]).with_context(|| format!("compact: bad size {}", x[1]))?;
                let l = if s == 0.0 { 0 } else { self.grid.lots(s).with_context(|| format!("compact: size {s} off the qty grid"))? };
                put_uv(&mut self.w, zz(t.wrapping_sub(prev)))?;
                put_uv(&mut self.w, zz(l))?;
                prev = t;
            }
        }
        Ok(())
    }

    pub fn finish(self) -> anyhow::Result<()> {
        self.w.finish()?.flush()?;
        Ok(())
    }
}

/// Where book messages come from: (matching-engine time, message) in replay order.
pub trait BookSource {
    fn next_msg(&mut self) -> anyhow::Result<Option<(i64, Value)>>;
}

/// The quote-saver JSON lines (time = `cts`, else `ts`, never decreasing), optionally written to a compact file as read.
pub struct JsonBook<'a> {
    rd: &'a mut dyn BufRead,
    line: String,
    last_cts: i64,
    n: u64,
    pub tee: Option<CompactBookWriter>,
}

impl<'a> JsonBook<'a> {
    pub fn new(rd: &'a mut dyn BufRead, tee: Option<CompactBookWriter>) -> Self {
        Self { rd, line: String::new(), last_cts: i64::MIN, n: 0, tee }
    }
}

impl BookSource for JsonBook<'_> {
    fn next_msg(&mut self) -> anyhow::Result<Option<(i64, Value)>> {
        loop {
            self.line.clear();
            if self.rd.read_line(&mut self.line)? == 0 {
                return Ok(None);
            }
            let s = self.line.trim();
            if s.is_empty() {
                continue;
            }
            self.n += 1;
            let msg: Value = serde_json::from_str(s).with_context(|| format!("ob200 line {}", self.n))?;
            let cts = msg["cts"].as_i64().or_else(|| msg["ts"].as_i64()).unwrap_or(self.last_cts).max(self.last_cts);
            self.last_cts = cts;
            if let Some(w) = self.tee.as_mut() {
                w.put(cts, &msg)?;
            }
            return Ok(Some((cts, msg)));
        }
    }
}

/// A compact book file (`CompactBookWriter`), decoded into the same messages (prices and sizes as numbers on the grid).
pub struct CompactBook {
    r: GzIn,
    grid: Grid,
    last: i64,
}

impl CompactBook {
    pub fn open(path: &Path, grid: &Grid) -> anyhow::Result<Self> {
        Ok(Self { r: compact_open(path, "book", Some(grid))?, grid: *grid, last: 0 })
    }
}

impl BookSource for CompactBook {
    fn next_msg(&mut self) -> anyhow::Result<Option<(i64, Value)>> {
        let Some(d) = get_uv(&mut self.r)? else { return Ok(None) };
        let cts = self.last.wrapping_add(unzz(d));
        self.last = cts;
        let mut k = [0u8; 1];
        self.r.read_exact(&mut k)?;
        let mut sides: [Vec<Value>; 2] = [vec![], vec![]];
        for side in sides.iter_mut() {
            let n = need_uv(&mut self.r)? as usize;
            side.reserve(n);
            let mut prev = 0i64;
            for _ in 0..n {
                let t = prev.wrapping_add(unzz(need_uv(&mut self.r)?));
                let l = unzz(need_uv(&mut self.r)?);
                prev = t;
                side.push(json!([self.grid.price(t), self.grid.qty(l)]));
            }
        }
        let [b, a] = sides;
        Ok(Some((cts, json!({"type": if k[0] == 1 { "snapshot" } else { "delta" }, "data": {"b": b, "a": a}}))))
    }
}

/// Trades as the replay parsed them: zigzag(ts delta), zigzag(tick delta), zigzag(lots), flags (1 buy, 2 RPI / off-grid,
/// 4 = the price follows as f64 bits because it is not exactly the grid price of its tick).
pub fn write_compact_trades(path: &Path, symbol: &str, grid: &Grid, trades: &[Tr]) -> anyhow::Result<()> {
    let mut w = compact_create(path, "trades", symbol, Some(grid))?;
    let (mut lt, mut lk) = (0i64, 0i64);
    for t in trades {
        put_uv(&mut w, zz(t.ts.wrapping_sub(lt)))?;
        put_uv(&mut w, zz(t.tick.wrapping_sub(lk)))?;
        put_uv(&mut w, zz(t.lots))?;
        let exact = grid.price(t.tick).to_bits() == t.px.to_bits();
        w.write_all(&[(t.buy as u8) | ((t.rpi as u8) << 1) | ((!exact as u8) << 2)])?;
        if !exact {
            w.write_all(&t.px.to_le_bytes())?;
        }
        (lt, lk) = (t.ts, t.tick);
    }
    w.finish()?.flush()?;
    Ok(())
}

pub fn read_compact_trades(path: &Path, grid: &Grid) -> anyhow::Result<Vec<Tr>> {
    let mut r = compact_open(path, "trades", Some(grid))?;
    let (mut lt, mut lk) = (0i64, 0i64);
    let mut out = vec![];
    while let Some(d) = get_uv(&mut r)? {
        let ts = lt.wrapping_add(unzz(d));
        let tick = lk.wrapping_add(unzz(need_uv(&mut r)?));
        let lots = unzz(need_uv(&mut r)?);
        let mut f = [0u8; 1];
        r.read_exact(&mut f)?;
        let px = if f[0] & 4 != 0 {
            let mut b = [0u8; 8];
            r.read_exact(&mut b)?;
            f64::from_le_bytes(b)
        } else {
            grid.price(tick)
        };
        out.push(Tr { ts, px, tick, lots, buy: f[0] & 1 != 0, rpi: f[0] & 2 != 0 });
        (lt, lk) = (ts, tick);
    }
    Ok(out)
}

/// A price tape (the BTC tape of the pull): zigzag(ts delta), zigzag(delta of price x 100); prices must be exact in cents.
pub fn write_compact_px(path: &Path, tape: &[(i64, f64)]) -> anyhow::Result<()> {
    let mut w = compact_create(path, "px", "", None)?;
    let (mut lt, mut lc) = (0i64, 0i64);
    for &(t, px) in tape {
        let c = (px * 100.0).round() as i64;
        anyhow::ensure!((c as f64 / 100.0).to_bits() == px.to_bits(), "compact px: {px} is not exact in cents");
        put_uv(&mut w, zz(t.wrapping_sub(lt)))?;
        put_uv(&mut w, zz(c.wrapping_sub(lc)))?;
        (lt, lc) = (t, c);
    }
    w.finish()?.flush()?;
    Ok(())
}

pub fn read_compact_px(path: &Path) -> anyhow::Result<Vec<(i64, f64)>> {
    let mut r = compact_open(path, "px", None)?;
    let (mut lt, mut lc) = (0i64, 0i64);
    let mut out = vec![];
    while let Some(d) = get_uv(&mut r)? {
        let t = lt.wrapping_add(unzz(d));
        let c = lc.wrapping_add(unzz(need_uv(&mut r)?));
        out.push((t, c as f64 / 100.0));
        (lt, lc) = (t, c);
    }
    Ok(out)
}

// ---------------------------------------------------------------- driver

pub struct ReplayParams {
    pub start: i64,
    pub end: i64,
    pub latency_ms: i64,
    pub fill_rpi: bool,
    pub depth: usize,
}

/// BTC prints up to `upto` (inclusive): the tape state, and inside the window the pull of every quoter with `btc_k` < z.
fn feed_btc(m: &mut Market, quoters: &mut [Quoter], btc: &[(i64, f64)], bi: &mut usize, upto: i64, p: &ReplayParams, min_k: f64) {
    while *bi < btc.len() && btc[*bi].0 <= upto {
        let (ts, px) = btc[*bi];
        *bi += 1;
        let z = m.on_btc(ts, px);
        if z > min_k && ts >= p.start && ts < p.end {
            for q in quoters.iter_mut() {
                q.advance(ts, m);
                q.on_btc(ts, z);
            }
        }
    }
}

/// Runs every quoter over one symbol-day: book messages from `book`, trades `trades` (time-sorted), the BTC price tape
/// `btc` (time-sorted; empty = no BTC pull).
#[cfg(test)]
pub fn run_replay(m: &mut Market, quoters: &mut [Quoter], book: &mut dyn BufRead, trades: &[Tr], btc: &[(i64, f64)], p: &ReplayParams) -> anyhow::Result<()> {
    let mut src = JsonBook::new(book, None);
    run_replay_src(m, quoters, &mut src, trades, btc, p)
}

/// `run_replay` over any book source (JSON lines or a compact file).
pub fn run_replay_src(m: &mut Market, quoters: &mut [Quoter], src: &mut dyn BookSource, trades: &[Tr], btc: &[(i64, f64)], p: &ReplayParams) -> anyhow::Result<()> {
    let mut ti = 0usize;
    let mut bi = 0usize;
    let min_k = quoters.iter().map(|q| q.spec.btc_k).filter(|k| *k > 0.0).fold(f64::INFINITY, f64::min);
    let step = |m: &mut Market, quoters: &mut [Quoter], tr: &Tr| {
        if tr.ts < p.end {
            for q in quoters.iter_mut() {
                q.advance(tr.ts, m);
            }
        }
        let liq = m.on_trade(tr);
        if tr.ts >= p.start && tr.ts < p.end {
            for q in quoters.iter_mut() {
                q.on_trade(tr, m);
                if liq {
                    q.on_liq(tr.ts);
                }
            }
        }
    };
    while let Some((cts, msg)) = src.next_msg()? {
        while ti < trades.len() && trades[ti].ts <= cts {
            feed_btc(m, quoters, btc, &mut bi, trades[ti].ts, p, min_k);
            step(m, quoters, &trades[ti]);
            ti += 1;
        }
        feed_btc(m, quoters, btc, &mut bi, cts, p, min_k);
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
        feed_btc(m, quoters, btc, &mut bi, trades[ti].ts, p, min_k);
        step(m, quoters, &trades[ti]);
        ti += 1;
    }
    feed_btc(m, quoters, btc, &mut bi, p.end, p, min_k);
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
    /// public.bybit.com trade archive (.csv.gz or .csv); not needed with --compact.
    #[arg(long)]
    trades: Option<PathBuf>,
    /// Replay the compact inputs in this directory (book.bin.gz, trades.bin.gz written by --write-compact) instead of
    /// --ob200 / --trades.
    #[arg(long)]
    compact: Option<PathBuf>,
    /// Also write the book and trades this run reads as compact files into this directory (lossless on the day's grid).
    #[arg(long)]
    write_compact: Option<PathBuf>,
    /// BTCUSDT price tape in the compact format (instead of --btc-trades).
    #[arg(long)]
    btc_compact: Option<PathBuf>,
    /// Write the BTC price tape read from --btc-trades in the compact format.
    #[arg(long)]
    write_btc_compact: Option<PathBuf>,
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
    /// Only write bars_1m.csv (and tape_stats.json) from the trades.
    #[arg(long)]
    bars_only: bool,
    /// Structural zones (levels_file format) for mode "ladder".
    #[arg(long)]
    zones: Option<PathBuf>,
    /// BTCUSDT public trade archive of the same day (H-MMW2 pull).
    #[arg(long)]
    btc_trades: Option<PathBuf>,
    /// BTCUSDT 1m bars of earlier days (BTC sigma warm-up).
    #[arg(long, value_delimiter = ',')]
    btc_warmup_bars: Vec<PathBuf>,
    /// Liquidation-cluster proxy notional floor (USD; 10 x the previous day's median print); 0 = proxy off.
    #[arg(long, default_value_t = 0.0)]
    liq_min_notional: f64,
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
    anyhow::ensure!(!(a.compact.is_some() && a.write_compact.is_some()), "--compact and --write-compact exclude each other");
    let trades = match (&a.compact, &a.trades) {
        (Some(d), _) => read_compact_trades(&d.join("trades.bin.gz"), m.grid())?,
        (None, Some(t)) => read_trades(t, m.grid())?,
        (None, None) => bail!("--trades or --compact"),
    };
    if let Some(d) = &a.write_compact {
        write_compact_trades(&d.join("trades.bin.gz"), &symbol, m.grid(), &trades)?;
    }
    if a.bars_only {
        for t in &trades {
            m.on_trade(t);
        }
        let (lo, hi) = (trades.first().map_or(0, |t| t.ts.div_euclid(MIN_MS) * MIN_MS), trades.last().map_or(0, |t| t.ts + 1));
        write_bars(&a.out.join("bars_1m.csv"), &m.bars, lo, hi)?;
        std::fs::write(a.out.join("tape_stats.json"), serde_json::to_string_pretty(&json!({"symbol": symbol, "prints": trades.len(), "median_print_notional": m.median_print_notional()}))?)?;
        println!("{symbol}: {} trades -> {} 1m bars in {}", trades.len(), m.bars.len(), a.out.display());
        return Ok(());
    }
    for p in &a.warmup_bars {
        read_bars(p, &mut m.bars)?;
    }
    m.liq.min_notional = a.liq_min_notional;
    let zones = std::rc::Rc::new(match &a.zones {
        Some(z) => read_zones(z, &symbol)?,
        None => vec![],
    });
    let btc = match (&a.btc_compact, &a.btc_trades) {
        (Some(_), _) | (None, Some(_)) => {
            let mut wb = BTreeMap::new();
            for p in &a.btc_warmup_bars {
                read_bars(p, &mut wb)?;
            }
            for (k, b) in wb {
                m.btc.seed_close(k, b.c);
            }
            match (&a.btc_compact, &a.btc_trades) {
                (Some(c), _) => read_compact_px(c)?,
                (None, Some(b)) => read_px_tape(b)?,
                _ => vec![],
            }
        }
        (None, None) => vec![],
    };
    if let Some(w) = &a.write_btc_compact {
        anyhow::ensure!(a.btc_trades.is_some(), "--write-btc-compact needs --btc-trades");
        write_compact_px(w, &btc)?;
    }
    if let Some(f) = &a.forecast {
        let v = read_forecast(f)?;
        FORECAST.with(|x| *x.borrow_mut() = std::rc::Rc::new(v));
    }
    let vf: VariantsFile = serde_json::from_str(&std::fs::read_to_string(a.variants.as_ref().context("--variants")?)?)?;
    let mut quoters = vec![];
    let mut gates: BTreeMap<String, std::rc::Rc<Vec<(i64, bool)>>> = BTreeMap::new();
    let mut widens: BTreeMap<String, std::rc::Rc<Vec<(i64, f64)>>> = BTreeMap::new();
    for v in &vf.variants {
        let widen = match &v.widen_file {
            Some(w) => {
                if !widens.contains_key(w) {
                    widens.insert(w.clone(), std::rc::Rc::new(read_widen(&ctx.s.root.join(w))?));
                }
                widens.get(w).cloned()
            }
            None => None,
        };
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
        let lats = if v.latencies_ms.is_empty() { vec![a.latency_ms] } else { v.latencies_ms.clone() };
        for lat in lats {
            for qm in &v.queues {
                let mut q = Quoter::new(cfg.clone(), v.clone(), *qm, &symbol, &rules, fees, lat, start, end, a.fill_rpi);
                if let Some(g) = &gate {
                    q.set_gate(g.clone());
                }
                if let Some(w) = &widen {
                    q.set_widen(w.clone());
                }
                q.set_zones(zones.clone());
                quoters.push(q);
            }
        }
    }
    let p = ReplayParams { start, end, latency_ms: a.latency_ms, fill_rpi: a.fill_rpi, depth: a.depth };
    if let Some(d) = &a.compact {
        let mut src = CompactBook::open(&d.join("book.bin.gz"), m.grid())?;
        run_replay_src(&mut m, &mut quoters, &mut src, &trades, &btc, &p)?;
    } else {
        let mut book = open_lines(a.ob200.as_ref().context("--ob200")?)?;
        let tee = match &a.write_compact {
            Some(d) => Some(CompactBookWriter::create(&d.join("book.bin.gz"), &symbol, m.grid())?),
            None => None,
        };
        let mut src = JsonBook::new(&mut *book, tee);
        run_replay_src(&mut m, &mut quoters, &mut src, &trades, &btc, &p)?;
        if let Some(w) = src.tee.take() {
            w.finish()?;
        }
    }
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
            v["latency_ms"] = json!(q.latency());
            v["mid_1s"] = mk[0].clone();
            v["mid_10s"] = mk[1].clone();
            v["mid_60s"] = mk[2].clone();
            v["mid_300s"] = json!(m.mid_at((f.ts + MARKOUT_LONG_S * 1000).min(p.end)));
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
                   "replayer_check_mismatches": m.check_mismatch, "mid_changes": m.mids.len(), "mid_start": first_mid, "mid_end": m.mid_at(p.end),
                   "median_print_notional": m.median_print_notional(), "liq_min_notional": m.liq.min_notional, "liq_clusters": m.liq_events.len(),
                   "liq_cluster_ts": m.liq_events, "btc_prints": m.btc_prints, "btc_z_gt_2": m.btc_z_counts[0], "btc_z_gt_3": m.btc_z_counts[1],
                   "btc_z_gt_4": m.btc_z_counts[2]},
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
        VariantSpec { name: name.into(), config: "config/mm_illiq20.yaml".into(), sigma_mode: SigmaMode::Fixed, fixed_bps: 10.0, ..VariantSpec::default() }
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
        let (mut m, mut q) = setup(QueueModel::Prorata);
        // a sell sweep through our price 100 ms after sending: the order is not live yet
        let t = trade(2_100, 0.2999, 500, false);
        q.advance(t.ts, &mut m);
        q.on_trade(&t, &m);
        assert!(q.fills.is_empty(), "not live before the latency");
        // 150 ms after sending it is live and joins behind the 100 lots
        let t = trade(2_200, 0.2999, 30, false);
        q.advance(t.ts, &mut m);
        assert_eq!(q.orders[0].q.ahead, 100.0);
        q.on_trade(&t, &m);
        assert_eq!(q.fills.len(), 1, "a print through our price fills us");
        assert_eq!(q.fills[0].qty, 20.0);
        assert_eq!(q.fills[0].kind, "through");
        // a cancel sent at 3000 takes effect at 3150: a fill at 3100 still counts, at 3200 it does not
        let g = *m.grid();
        q.place(Side::Sell, 0.3002, 20.0, true, 2_500, &g);
        q.advance(2_700, &mut m);
        q.cancel_idx(1, 3_000);
        let t = trade(3_100, 0.3003, 10, true);
        q.advance(t.ts, &mut m);
        q.on_trade(&t, &m);
        assert_eq!(q.fills.len(), 2, "fills before the cancel is effective count");
        let t = trade(3_200, 0.3003, 10, true);
        q.advance(t.ts, &mut m);
        q.on_trade(&t, &m);
        assert_eq!(q.fills.len(), 2, "no fill after the cancel took effect");
    }

    #[test]
    fn queue_position_under_the_three_models_end_to_end() {
        let mut got = vec![];
        for model in [QueueModel::Pessimistic, QueueModel::Prorata, QueueModel::Optimistic] {
            let (mut m, mut q) = setup(model);
            q.advance(2_200, &mut m);
            // 30 lots join behind us, then the level drops by 60 with no print (cancels): 130 -> 70
            m.apply_book(&msg("delta", 2_300, vec![lv(0.3000, 130.0)], vec![]), 2_300).unwrap();
            q.advance(2_300, &mut m);
            q.on_book(&m, 2_300);
            m.apply_book(&msg("delta", 2_400, vec![lv(0.3000, 70.0)], vec![]), 2_400).unwrap();
            q.on_book(&m, 2_400);
            let ahead = q.orders[0].q.ahead;
            // then a 60-lot sell print at our price
            let t = trade(2_500, 0.3000, 60, false);
            q.advance(t.ts, &mut m);
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
        q.advance(2_200, &mut m);
        // asks move down to our bid price (a resting seller at 0.3000): we are filled there
        m.apply_book(&msg("delta", 2_300, vec![lv(0.3000, 0.0)], vec![lv(0.3000, 5.0)]), 2_300).unwrap();
        q.on_book(&m, 2_300);
        assert_eq!(q.fills.len(), 1);
        assert_eq!((q.fills[0].kind, q.fills[0].qty), ("cross", 5.0));
        // a bid sent at or above the ask is rejected PostOnly at activation
        let g = *m.grid();
        q.place(Side::Buy, 0.3001, 20.0, false, 3_000, &g);
        m.apply_book(&msg("delta", 3_100, vec![], vec![lv(0.3001, 9.0)]), 3_100).unwrap();
        q.advance(3_200, &mut m);
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
        run_replay(&mut m, &mut quoters, &mut rd, &trades, &[], &p).unwrap();
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
        run_replay(&mut m_new(), &mut quoters, &mut rd, &trades, &[], &p).unwrap();
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

    // ------------------------------------------------------------ H-MMW engine paths

    fn mmw_quoter(model: QueueModel, s: VariantSpec) -> Quoter {
        Quoter::new(cfg_with(&s), s, model, "TRXUSDT", &rules(), FeeRates { maker_bps: 2.0, taker_bps: 5.5, slippage_bps: 0.0 }, 150, 0, 10_000_000, false)
    }

    fn book_0300() -> Market {
        let mut m = Market::new(&rules(), 200).unwrap();
        m.apply_book(&msg("snapshot", 1_000, vec![lv(0.3000, 100.0), lv(0.2999, 50.0)], vec![lv(0.3002, 100.0), lv(0.3003, 50.0)]), 1_000).unwrap();
        m
    }

    #[test]
    fn ladder_quotes_levels_takes_profit_and_stops_at_market() {
        let mut m = book_0300();
        let mut s = spec("ladder");
        s.mode = QuoteMode::Ladder;
        s.sigma_mode = SigmaMode::Config;
        let mut q = mmw_quoter(QueueModel::Prorata, s);
        // sigma = the 20 bp floor (no bars): band 30-70 bp; support top 0.2985 (53 bp), resistance bottom 0.3020 (63 bp);
        // 0.2995 (20 bp) is too close, 0.2900 too far
        q.set_zones(std::rc::Rc::new(vec![(0.2980, 0.2985), (0.2990, 0.2995), (0.2890, 0.2900), (0.3020, 0.3030)]));
        q.next_tick = 2_000;
        q.advance(2_001, &mut m);
        let live: Vec<(Side, f64, i64)> = q.orders.iter().filter(|o| !o.closed).map(|o| (o.side, o.price, o.tag)).collect();
        assert_eq!(live, vec![(Side::Buy, 0.2985, TAG_ADD), (Side::Sell, 0.3020, TAG_ADD)], "{live:?}");
        assert!((q.st.sigma - 0.002).abs() < 1e-12);
        // a sell sweep through the bid after the latency fills one minimum lot (5.25 USD / 0.2985 -> 18 lots)
        let t = trade(2_500, 0.2984, 500, false);
        q.advance(t.ts, &mut m);
        q.on_trade(&t, &m);
        assert_eq!(q.fills.len(), 1);
        assert_eq!((q.fills[0].kind, q.fills[0].qty), ("through", 18.0));
        assert_eq!(q.lots.len(), 1);
        // take-profit at the mid at fill time (0.3001, rounded up) sent at the fill; the adding ask is cancelled
        let tp: Vec<&SimOrder> = q.orders.iter().filter(|o| !o.closed && o.tag > 0).collect();
        assert_eq!(tp.len(), 1);
        assert!(tp[0].reduce_only && tp[0].side == Side::Sell && (tp[0].price - 0.3001).abs() < 1e-9);
        assert!(q.orders.iter().any(|o| o.tag == TAG_ADD && o.side == Side::Sell && o.cancel_ms == Some(2_650)));
        let stop = q.lots[0].stop;
        assert!((stop - 0.2985 * (-1.5f64 * 0.002).exp()).abs() < 1e-12);
        // a print at the stop: a taker close of the lot one latency later, at the bid (0.3000 in this static book)
        let t = trade(3_000, 0.2975, 1, false);
        q.advance(t.ts, &mut m);
        q.on_trade(&t, &m);
        assert_eq!(q.funnel.ladder_stops, 1);
        q.advance(3_200, &mut m);
        let last = q.fills.last().unwrap();
        assert_eq!((last.kind, last.maker, last.qty, last.pos_after), ("stop", false, 18.0, 0.0));
        assert!(q.lots.is_empty() && q.position_qty() == 0.0);
        assert!(q.orders.iter().filter(|o| o.tag > 0).all(|o| o.closed || o.cancel_ms.is_some()), "the TP is cancelled");
    }

    #[test]
    fn ladder_take_profit_fill_closes_its_lot() {
        let mut m = book_0300();
        let mut s = spec("ladder");
        s.mode = QuoteMode::Ladder;
        let mut q = mmw_quoter(QueueModel::Prorata, s);
        q.set_zones(std::rc::Rc::new(vec![(0.2980, 0.2985)]));
        q.next_tick = 2_000;
        q.advance(2_001, &mut m);
        let t = trade(2_500, 0.2984, 500, false);
        q.advance(t.ts, &mut m);
        q.on_trade(&t, &m);
        // the TP (0.3001) is live after 150 ms; a buy print through it fills it
        let t = trade(2_800, 0.3002, 500, true);
        q.advance(t.ts, &mut m);
        q.on_trade(&t, &m);
        assert_eq!(q.fills.len(), 2);
        assert!((q.fills[1].price - 0.3001).abs() < 1e-9 && q.fills[1].reduce_only && q.fills[1].maker);
        assert_eq!(q.funnel.ladder_tp_fills, 1);
        assert!(q.lots.is_empty() && q.position_qty() == 0.0);
    }

    #[test]
    fn btc_pull_cancels_everything_for_the_cooldown() {
        let mut m = book_0300();
        let mut s = spec("pull");
        s.btc_k = 3.0;
        s.btc_cooldown_s = 10.0;
        let mut q = mmw_quoter(QueueModel::Prorata, s);
        q.next_tick = 2_000;
        q.advance(2_001, &mut m);
        let n0 = q.funnel.placed;
        assert!(n0 >= 2);
        q.on_btc(2_500, 2.9);
        assert!(q.orders.iter().all(|o| o.cancel_ms.is_none()), "below k: nothing");
        q.on_btc(2_500, 3.5);
        assert!(q.orders.iter().all(|o| o.cancel_ms == Some(2_650)), "every quote cancelled, effective one latency later");
        assert_eq!(q.funnel.btc_pulls, 1);
        // the 10 s ticks inside the cooldown place nothing; the first tick after it quotes again
        q.advance(12_001, &mut m);
        assert_eq!(q.funnel.placed, n0);
        q.advance(22_001, &mut m);
        assert!(q.funnel.placed > n0);
    }

    #[test]
    fn liquidation_pull_cancels_adding_quotes_only() {
        let mut m = book_0300();
        let mut s = spec("liq");
        s.liq_pull_min = 60.0;
        let mut q = mmw_quoter(QueueModel::Prorata, s);
        let g = *m.grid();
        q.next_tick = i64::MAX;
        q.place(Side::Buy, 0.2990, 20.0, false, 1_500, &g);
        q.place(Side::Sell, 0.3011, 20.0, true, 1_500, &g);
        q.on_liq(2_000);
        assert_eq!(q.orders[0].cancel_ms, Some(2_150));
        assert_eq!(q.orders[1].cancel_ms, None, "a reduce-only quote stays");
        assert_eq!(q.liq_until, 2_000 + 3_600_000);
        // the market's proxy: 5 sells over 3 levels within 30 s, notional >= the floor
        m.liq.min_notional = 10.0;
        let fired: Vec<bool> = (0..5).map(|i| m.on_trade(&trade(3_000 + i * 100, 0.3000 - 0.0001 * (i % 3) as f64, 100, false))).collect();
        assert_eq!(fired, vec![false, false, false, false, true]);
        assert_eq!(m.liq_events, vec![3_400]);
    }

    #[test]
    fn touch_joins_short_queues_and_leaves_when_the_queue_behind_thins() {
        let mut m = book_0300();
        let mut s = spec("touch");
        s.mode = QuoteMode::Touch;
        s.touch_q = 0.5;
        let mut q = mmw_quoter(QueueModel::Pessimistic, s);
        q.next_tick = 2_000;
        // median touch size 100 lots: 100 ahead is not a short queue on either side
        q.advance(2_001, &mut m);
        assert_eq!(q.funnel.touch_skips, 2);
        assert_eq!(q.funnel.placed, 0);
        // the bid level thins to 30 lots: the next tick joins it (at the touch, 0.3000)
        m.apply_book(&msg("delta", 5_000, vec![lv(0.3000, 30.0)], vec![]), 5_000).unwrap();
        q.advance(12_001, &mut m);
        let o: Vec<(Side, f64)> = q.orders.iter().filter(|o| !o.closed).map(|o| (o.side, o.price)).collect();
        assert_eq!(o, vec![(Side::Buy, 0.3000)]);
        q.advance(12_200, &mut m);
        assert_eq!(q.orders[0].q.ahead, 30.0);
        // 50 lots join behind us, then 40 cancel (pessimistic: from behind first): behind 50 -> 10 within 2 s: leave
        m.apply_book(&msg("delta", 12_300, vec![lv(0.3000, 80.0)], vec![]), 12_300).unwrap();
        q.on_book(&m, 12_300);
        assert_eq!(q.funnel.touch_leaves, 0);
        m.apply_book(&msg("delta", 13_000, vec![lv(0.3000, 40.0)], vec![]), 13_000).unwrap();
        q.on_book(&m, 13_000);
        assert_eq!(q.funnel.touch_leaves, 1);
        assert_eq!(q.orders[0].cancel_ms, Some(13_150));
    }

    #[test]
    fn mmw_modes_end_to_end_keep_the_ledger_and_are_deterministic() {
        let run = || {
            let (lines, trades) = synthetic_day();
            let mut m = Market::new(&rules(), 200).unwrap();
            m.liq.min_notional = 30.0;
            let mut specs = vec![];
            let mut base = spec("base");
            base.overrides.insert("refresh_s".into(), json!(5));
            base.fixed_bps = 3.0;
            let mut ladder = base.clone();
            ladder.name = "ladder".into();
            ladder.mode = QuoteMode::Ladder;
            ladder.sigma_mode = SigmaMode::Config;
            ladder.liq_pull_min = 5.0;
            ladder.ladder_band = (0.2, 3.5);
            specs.push(ladder);
            let mut touch = base.clone();
            touch.name = "touch".into();
            touch.mode = QuoteMode::Touch;
            touch.touch_q = 1.0;
            specs.push(touch);
            let mut btc = base.clone();
            btc.name = "btc".into();
            btc.btc_k = 3.0;
            btc.btc_cooldown_s = 30.0;
            specs.push(btc);
            for (n, sig) in [("imb", SideSignal::Imb5), ("flow", SideSignal::Flow30)] {
                let mut x = base.clone();
                x.name = n.into();
                x.side_signal = Some(sig);
                x.side_threshold = 0.3;
                x.side_widen_sigma = if n == "imb" { 1.0 } else { 0.0 };
                specs.push(x);
            }
            let fee = FeeRates { maker_bps: 2.0, taker_bps: 5.5, slippage_bps: 0.0 };
            let zones = std::rc::Rc::new(vec![(0.2985, 0.2990), (0.2960, 0.2965), (0.3011, 0.3016), (0.3040, 0.3045)]);
            let mut quoters = vec![];
            for s in &specs {
                for lat in [150, 300] {
                    for qm in [QueueModel::Pessimistic, QueueModel::Prorata] {
                        let mut q = Quoter::new(cfg_with(s), s.clone(), qm, "TRXUSDT", &rules(), fee, lat, 0, 3 * 3_600_000, false);
                        q.set_zones(zones.clone());
                        quoters.push(q);
                    }
                }
            }
            // BTC: a print every 200 ms, flat with a 40 bp jump every 20 min after the first hour
            let btc: Vec<(i64, f64)> = (0..54_000i64).map(|i| (i * 200, if i * 200 > 3_600_000 && (i * 200) % 1_200_000 < 30_000 { 60_240.0 } else { 60_000.0 + (i % 7) as f64 })).collect();
            let text = lines.join("\n");
            let mut rd = BufReader::new(text.as_bytes());
            let p = ReplayParams { start: 0, end: 3 * 3_600_000, latency_ms: 150, fill_rpi: false, depth: 200 };
            run_replay(&mut m, &mut quoters, &mut rd, &trades, &btc, &p).unwrap();
            let rows: Vec<Value> = quoters.iter().map(|q| q.summary(&m, p.end)).collect();
            (rows, m.liq_events.len(), m.btc_z_counts)
        };
        let (a, liq_a, z_a) = run();
        let (b, liq_b, z_b) = run();
        assert_eq!(serde_json::to_string(&a).unwrap(), serde_json::to_string(&b).unwrap(), "deterministic");
        assert_eq!((liq_a, z_a), (liq_b, z_b));
        assert!(liq_a > 0, "the proxy fires on the synthetic tape");
        assert!(z_a[2] > 0, "BTC jumps are seen");
        for r in &a {
            assert!(r["pnl"]["ledger_check"].as_f64().unwrap().abs() < 1e-9, "{}", r["name"]);
        }
        let get = |name: &str, key: &str| a.iter().filter(|r| r["name"] == name).map(|r| r["funnel"][key].as_u64().unwrap()).sum::<u64>();
        assert!(get("ladder", "ladder_adds_placed") > 0 && get("ladder", "liq_pulls") > 0);
        assert!(get("touch", "placed") > 0 && get("touch", "touch_skips") > 0);
        assert!(get("btc", "btc_pulls") > 0);
        assert!(get("imb", "side_pull_ticks") > 0 && get("flow", "side_pull_ticks") > 0);
        // one quoter per latency x queue, and the latency is reported
        assert_eq!(a.iter().filter(|r| r["name"] == "btc").count(), 4);
        assert!(a.iter().any(|r| r["latency_ms"] == 300));
    }

    // ------------------------------------------------------------ WMQ quote rules, probe, compact inputs

    #[test]
    fn glft_and_as_half_spreads() {
        // flat: symmetric; long inventory moves both quotes down (ask closer, bid further); the skew per order is c2
        let (b0, a0) = glft_half_spreads(0.5, 0.01, 2.0, 0.0).unwrap();
        assert!((b0 - a0).abs() < 1e-12);
        let c1 = (1.0f64 + 0.25).ln() / 0.5;
        let c2 = (0.5f64 / (2.0 * 2.0 * 0.01) / 3600.0 * 1.25f64.powf(5.0)).sqrt();
        assert!((b0 - (c1 + 0.5 * c2)).abs() < 1e-12, "{b0} vs {}", c1 + 0.5 * c2);
        let (b1, a1) = glft_half_spreads(0.5, 0.01, 2.0, 1.0).unwrap();
        assert!((b1 - b0 - c2).abs() < 1e-12 && (a0 - a1 - c2).abs() < 1e-12);
        // more risk aversion: a larger skew per order
        let s = |g: f64| {
            let (b, a) = glft_half_spreads(g, 0.01, 2.0, 1.0).unwrap();
            b - a
        };
        assert!(s(0.1) < s(0.5) && s(0.5) < s(2.0));
        // vanishing risk aversion: the half-spread tends to 1 / k (the single-fill profit maximiser)
        let (bz, _) = glft_half_spreads(1e-6, 0.01, 2.0, 0.0).unwrap();
        assert!((bz - 0.5).abs() < 1e-3, "{bz}");
        // AS: reservation shift q g tau, spread g tau + 2 ln(1 + g / k) / g
        let (b, a) = as_half_spreads(0.1, 2.0, 3600.0, 2.0).unwrap();
        let h = (1.0f64 + 0.05).ln() / 0.1 + 0.05;
        assert!((b - (h + 0.2)).abs() < 1e-12 && (a - (h - 0.2)).abs() < 1e-12);
        assert!(glft_half_spreads(0.0, 0.01, 2.0, 0.0).is_none() && as_half_spreads(0.1, 0.0, 3600.0, 0.0).is_none());
    }

    fn mspec(rule: ModelPrice) -> VariantSpec {
        VariantSpec { name: "m".into(), config: "config/mm_illiq20.yaml".into(), mode: QuoteMode::Model, price_rule: rule, cap_usd: 25.0, ..VariantSpec::default() }
    }

    #[test]
    fn model_quotes_depth_caps_sizes_and_unwind() {
        // TRX-like grid (tick 0.0001, step 1, min notional 5 -> 18 lots at 0.30), 1h sigma 40 bp, cap 83 lots (25 USD)
        let r = rules();
        let base = ModelIn { bid1: 0.3000, ask1: 0.3002, sigma: 0.004, pos: 0, lot: 18, cap: 83, hold_s: 0.0, no_add_bid: false, no_add_ask: false, widen: 0.0, mult: 0.0 };
        let mut s = mspec(ModelPrice::Depth);
        s.depth_sigma = 1.0;
        let (b, a) = model_quotes(&s, &r, &base);
        let (b, a) = (b.unwrap(), a.unwrap());
        // mid 0.3001 -/+ 40 bp: 0.2988996 -> 0.2988 (down), 0.3013004 -> 0.3014 (up); one minimum lot each, adding
        assert!((b.price - 0.2988).abs() < 1e-12 && (a.price - 0.3014).abs() < 1e-12, "{b:?} {a:?}");
        assert_eq!((b.qty, a.qty, b.reduce_only, a.reduce_only), (18.0, 18.0, false, false));
        // long 72 lots: the ask reduces (reduce-only, one order), the bid adds only what fits under the cap
        let long = ModelIn { pos: 72, ..base };
        let (b, a) = model_quotes(&s, &r, &long);
        assert_eq!(a.unwrap().qty, 18.0);
        assert!(a.unwrap().reduce_only && b.is_none(), "72 + 18 > 83: no adding bid");
        // size skew (3 lots): adding shrinks, reducing grows with |q| / cap
        let mut sk = s.clone();
        sk.lots = 3;
        sk.size_rule = SizeRule::Linear;
        let x = ModelIn { pos: 108, cap: 216, ..base };
        let (b, a) = model_quotes(&sk, &r, &x);
        assert_eq!((b.unwrap().qty, a.unwrap().qty), (36.0, 90.0), "frac 0.5: add round(1.5) = 2 lots, reduce round(4.5) = 5 lots");
        // passive unwind: after the holding time the reducing quote joins the touch
        let mut u = s.clone();
        u.unwind_passive_s = 600.0;
        let (_, a) = model_quotes(&u, &r, &ModelIn { pos: 18, hold_s: 601.0, ..base });
        assert!((a.unwrap().price - 0.3002).abs() < 1e-12);
        let (_, a) = model_quotes(&u, &r, &ModelIn { pos: 18, hold_s: 599.0, ..base });
        assert!((a.unwrap().price - 0.3014).abs() < 1e-12);
        // no-add flags and the widen
        let (b, a) = model_quotes(&s, &r, &ModelIn { no_add_bid: true, ..base });
        assert!(b.is_none() && a.is_some());
        let (b, _) = model_quotes(&s, &r, &ModelIn { widen: 1.0, ..base });
        assert!((b.unwrap().price - 0.2976).abs() < 1e-12);
        // a multiplicative widen of 2 doubles the half-spread (the same price as +1 sigma here)
        let (b, _) = model_quotes(&s, &r, &ModelIn { mult: 2.0, ..base });
        assert!((b.unwrap().price - 0.2976).abs() < 1e-12);
        // GLFT: a long position pulls the ask in, never crossing the bid
        let mut g = mspec(ModelPrice::Glft);
        (g.glft_gamma, g.glft_a, g.glft_k) = (0.5, 0.01, 2.0);
        let (_, a_flat) = model_quotes(&g, &r, &base);
        let (_, a_long) = model_quotes(&g, &r, &ModelIn { pos: 72, ..base });
        assert!(a_long.unwrap().price < a_flat.unwrap().price && a_long.unwrap().price >= 0.3001);
        let mut far = g.clone();
        far.glft_gamma = 50.0;
        let (_, a) = model_quotes(&far, &r, &ModelIn { pos: 72, ..base });
        assert!((a.unwrap().price - 0.3001).abs() < 1e-12, "clamped one tick above the bid");
    }

    #[test]
    fn compact_inputs_replay_identically() {
        let (lines, trades) = synthetic_day();
        let dir = std::env::temp_dir().join(format!("mm_replay_compact_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let g = *m_new().grid();
        let fee = FeeRates { maker_bps: 2.0, taker_bps: 5.5, slippage_bps: 0.0 };
        let mk = || {
            let mut s = spec("fixed10");
            s.overrides.insert("refresh_s".into(), json!(5));
            all_queues().into_iter().map(|qm| Quoter::new(cfg_with(&s), s.clone(), qm, "TRXUSDT", &rules(), fee, 150, 0, 3 * 3_600_000, false)).collect::<Vec<_>>()
        };
        let p = ReplayParams { start: 0, end: 3 * 3_600_000, latency_ms: 150, fill_rpi: false, depth: 200 };
        // off-grid and RPI prints survive the round trip
        let mut tr = trades.clone();
        tr[3].px += 0.00001;
        tr[3].rpi = true;
        tr[5].rpi = true;
        write_compact_trades(&dir.join("trades.bin.gz"), "TRXUSDT", &g, &tr).unwrap();
        assert_eq!(read_compact_trades(&dir.join("trades.bin.gz"), &g).unwrap(), tr);
        let px: Vec<(i64, f64)> = (0..1000).map(|i| (i * 37, 60_000.0 + (i % 13) as f64 * 0.1)).collect();
        write_compact_px(&dir.join("px.bin.gz"), &px).unwrap();
        assert_eq!(read_compact_px(&dir.join("px.bin.gz")).unwrap(), px);
        // replay from the JSON lines while writing the compact book, then from the compact book: same stats
        let text = lines.join("\n");
        let mut rd = BufReader::new(text.as_bytes());
        let (mut m1, mut q1) = (m_new(), mk());
        let mut src = JsonBook::new(&mut rd, Some(CompactBookWriter::create(&dir.join("book.bin.gz"), "TRXUSDT", &g).unwrap()));
        run_replay_src(&mut m1, &mut q1, &mut src, &trades, &[], &p).unwrap();
        src.tee.take().unwrap().finish().unwrap();
        let (mut m2, mut q2) = (m_new(), mk());
        let mut src2 = CompactBook::open(&dir.join("book.bin.gz"), &g).unwrap();
        run_replay_src(&mut m2, &mut q2, &mut src2, &trades, &[], &p).unwrap();
        let rows = |m: &Market, q: &[Quoter]| serde_json::to_string(&q.iter().map(|x| x.summary(m, p.end)).collect::<Vec<_>>()).unwrap();
        assert_eq!(rows(&m1, &q1), rows(&m2, &q2));
        assert_eq!((m1.n_msgs, m1.mids.len(), m1.check_mismatch), (m2.n_msgs, m2.mids.len(), 0));
        // a different grid is refused
        let g2 = Grid::new(0.001, 1.0).unwrap();
        assert!(CompactBook::open(&dir.join("book.bin.gz"), &g2).is_err());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn probe_and_model_modes_end_to_end() {
        let (lines, trades) = synthetic_day();
        let fee = FeeRates { maker_bps: 2.0, taker_bps: 5.5, slippage_bps: 0.0 };
        let run = || {
            let mut probe = spec("probe");
            probe.sigma_mode = SigmaMode::Config;
            probe.mode = QuoteMode::Probe;
            probe.probe_x = vec![0.0, 0.25, 1.0, 3.0];
            let mut dep = mspec(ModelPrice::Depth);
            dep.name = "depth".into();
            dep.depth_sigma = 0.25;
            dep.unwind_aggr_s = 300.0;
            let mut gl = mspec(ModelPrice::Glft);
            gl.name = "glft".into();
            (gl.glft_gamma, gl.glft_a, gl.glft_k) = (0.5, 0.01, 2.0);
            gl.lots = 2;
            gl.size_rule = SizeRule::Linear;
            gl.cap_usd = 60.0;
            gl.trend_k = 1.0;
            let mut quoters = vec![];
            for s in [&probe, &dep, &gl] {
                for qm in [QueueModel::Pessimistic, QueueModel::Prorata] {
                    quoters.push(Quoter::new(cfg_with(s), s.clone(), qm, "TRXUSDT", &rules(), fee, 150, 0, 3 * 3_600_000, false));
                }
            }
            let mut m = m_new();
            let text = lines.join("\n");
            let mut rd = BufReader::new(text.as_bytes());
            let p = ReplayParams { start: 0, end: 3 * 3_600_000, latency_ms: 150, fill_rpi: false, depth: 200 };
            run_replay(&mut m, &mut quoters, &mut rd, &trades, &[], &p).unwrap();
            let rows: Vec<Value> = quoters.iter().map(|q| q.summary(&m, p.end)).collect();
            let tagged = quoters[0].fills.iter().all(|f| f.probe.is_some()) && quoters[2].fills.iter().all(|f| f.probe.is_none());
            (rows, tagged)
        };
        let (a, tagged) = run();
        let (b, _) = run();
        assert_eq!(serde_json::to_string(&a).unwrap(), serde_json::to_string(&b).unwrap(), "deterministic");
        assert!(tagged, "probe fills carry their distance, others do not");
        for r in &a {
            assert!(r["pnl"]["ledger_check"].as_f64().unwrap().abs() < 1e-9, "{}", r["name"]);
            assert!(r["pnl"]["adverse_300s"].is_number());
        }
        // the probe: every leg is live; the touch fills most, the far legs least
        let legs = a[1]["probe"].as_array().unwrap();
        assert_eq!(legs.len(), 8);
        assert!(legs.iter().all(|l| l["live_ms"].as_i64().unwrap() > 0 && l["placed"].as_u64().unwrap() > 0));
        let fills = |j: usize| legs.iter().filter(|l| l["j"] == j).map(|l| l["fills"].as_u64().unwrap()).sum::<u64>();
        assert!(fills(0) > fills(2) && fills(2) >= fills(3), "{} {} {}", fills(0), fills(2), fills(3));
        assert!(legs[2]["x_eff_mean"].as_f64().unwrap() > 0.1, "{}", legs[2]);
        // model quoters trade; the time limit closes at market
        let f = |name: &str, key: &str| a.iter().filter(|r| r["name"] == name).map(|r| r["model"][key].as_u64().unwrap()).sum::<u64>();
        assert!(a.iter().filter(|r| r["name"] == "depth").all(|r| r["fills"].as_u64().unwrap() > 0));
        assert!(f("depth", "unwind_time") > 0 && f("depth", "ticks") > 0);
        assert!(f("glft", "ticks") > 0);
    }
}
