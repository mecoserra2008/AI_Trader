//! Per-symbol 3-second aggregation on **exchange time**, closed on a trade + book watermark.
//!
//! Trades accumulate OHLC / buy / sell / counts in the candle of their exchange time `T`. Book features
//! are *time-weighted* inside the candle: each book state is weighted by how long it was live (a book
//! that flickers for 5 ms barely counts). OFI is summed over every book update.
//!
//! The trade stream (`publicTrade`) and the book stream (`orderbook`) are separate topics: a book message
//! stamped after a candle's end can arrive before trades executed inside that candle. Closing a candle
//! on the first later event (the rule before 2026-11-27) booked those late trades into a later candle
//! (WS02, H-WS02-9: per-3 s trade counts matched the public archive only 57% of the time). A candle now
//! stays open until one of:
//!
//! - **both watermarks** (newest exchange time seen on this symbol's trade stream and on its book
//!   stream) have passed the candle end;
//! - the wall clock is past end + `grace_ms` and every stream that has not passed the end has been
//!   silent for `grace_ms` (quiet symbols; a stream that still delivers older events is lagging, so the
//!   candle waits for it);
//! - the wall clock is past end + `max_hold_ms` (hard cap while a stream lags).
//!
//! A trade older than the oldest open candle (it lagged past the cap) is booked into the oldest open
//! candle, so totals stay exact, and counted in `late_trades`. `known_ts = max(local close time, end)`.

use std::collections::VecDeque;

use at_core::time::Ms;
use at_store::Batch;

use crate::book::{Book, BookFeat, L1, ofi};

pub const CANDLE_MS: Ms = 3_000;
/// Gaps longer than this (e.g. after an outage) are skipped instead of emitting empty candles.
const MAX_FILL_CANDLES: i64 = 200;
const NF: usize = 12;

fn bucket(t: Ms) -> Ms {
    t - t.rem_euclid(CANDLE_MS)
}

fn feat_vec(f: &BookFeat) -> [f64; NF] {
    [
        f.spread_bps, f.micro_bps, f.obi_l1,
        f.obi[0], f.obi[1], f.obi[2], f.obi[3], f.obi[4],
        f.depth_bid[2], f.depth_ask[2], f.depth_bid[4], f.depth_ask[4],
    ]
}

/// When an open candle may be closed (see the module docs).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ClosePolicy {
    /// Wall-clock grace past the candle end, applied once the streams that have not passed it are idle.
    pub grace_ms: Ms,
    /// Wall-clock hard cap past the candle end, applied even while a stream lags.
    pub max_hold_ms: Ms,
}

impl Default for ClosePolicy {
    fn default() -> Self {
        Self { grace_ms: 1_500, max_hold_ms: 30_000 }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct FlowRow {
    pub symbol: String,
    pub start: Ms,
    pub open: Option<f64>,
    pub high: Option<f64>,
    pub low: Option<f64>,
    pub close: Option<f64>,
    pub vwap: Option<f64>,
    pub buy_vol: f64,
    pub sell_vol: f64,
    pub buy_notional: f64,
    pub sell_notional: f64,
    pub n_buy: i32,
    pub n_sell: i32,
    pub max_trade: f64,
    pub mid: Option<f64>,
    /// Time-weighted book features (None if no valid book during the candle).
    pub twa: Option<[f64; NF]>,
    pub ofi: Option<f64>,
    pub span_bps: Option<f64>,
    pub book_cov: f64,
    pub book_updates: i32,
    pub known_ts: Ms,
}

/// Trade side of one open candle.
#[derive(Clone, Copy, Debug)]
struct TradeAcc {
    open: Option<f64>,
    high: f64,
    low: f64,
    last: Option<f64>,
    pv: f64,
    buy_vol: f64,
    sell_vol: f64,
    buy_not: f64,
    sell_not: f64,
    n_buy: i32,
    n_sell: i32,
    max_trade: f64,
}

impl Default for TradeAcc {
    fn default() -> Self {
        Self { open: None, high: f64::MIN, low: f64::MAX, last: None, pv: 0.0, buy_vol: 0.0, sell_vol: 0.0, buy_not: 0.0, sell_not: 0.0, n_buy: 0, n_sell: 0, max_trade: 0.0 }
    }
}

impl TradeAcc {
    fn add(&mut self, price: f64, size: f64, is_buy: bool) {
        self.open.get_or_insert(price);
        self.high = self.high.max(price);
        self.low = self.low.min(price);
        self.last = Some(price);
        self.pv += price * size;
        self.max_trade = self.max_trade.max(size);
        if is_buy {
            self.buy_vol += size;
            self.buy_not += price * size;
            self.n_buy += 1;
        } else {
            self.sell_vol += size;
            self.sell_not += price * size;
            self.n_sell += 1;
        }
    }

    fn traded(&self) -> bool {
        self.n_buy + self.n_sell > 0
    }
}

/// Book side of one candle (complete once the book cursor has moved past it).
#[derive(Clone, Copy, Debug)]
struct BookAcc {
    tw: [f64; NF],
    tw_ms: f64,
    ofi: f64,
    ofi_seen: bool,
    span_min: f64,
    updates: i32,
    /// Mid of the book state live at the candle end.
    mid: Option<f64>,
}

impl BookAcc {
    fn fresh(feat: Option<BookFeat>) -> Self {
        Self { tw: [0.0; NF], tw_ms: 0.0, ofi: 0.0, ofi_seen: false, span_min: feat.map(|f| f.span_bps).unwrap_or(f64::MAX), updates: 0, mid: None }
    }
}

#[derive(Clone, Debug)]
struct Slot {
    trade: TradeAcc,
    book: Option<BookAcc>,
}

impl Slot {
    fn empty() -> Self {
        Self { trade: TradeAcc::default(), book: None }
    }

    fn has_data(&self) -> bool {
        self.trade.traded() || self.book.is_some_and(|b| b.updates > 0)
    }
}

#[derive(Debug)]
pub struct SymbolFlow {
    pub symbol: String,
    pub book: Book,
    policy: ClosePolicy,
    initialised: bool,
    /// Start of the oldest open candle (the next one to emit).
    next: Ms,
    /// Open candles `next, next + 3 s, ...` (contiguous). Slots before the book cursor hold a complete book part.
    slots: VecDeque<Slot>,
    /// Book cursor: the candle the live book accumulator `bacc` covers (always >= `next`).
    bstart: Ms,
    bacc: BookAcc,
    feat: Option<BookFeat>,
    feat_ts: Ms,
    /// Last trade price, carried forward through trade-less candles.
    last_px: Option<f64>,
    /// Newest exchange time seen on the trade / book stream, and the local time of the last message of each.
    wm_trade: Ms,
    wm_book: Ms,
    rx_trade: Ms,
    rx_book: Ms,
    /// Counters (monotonic) for the collector's heartbeat.
    pub trades: u64,
    pub late_trades: u64,
    pub late_books: u64,
}

impl SymbolFlow {
    pub fn new(symbol: &str) -> Self {
        Self::with_policy(symbol, ClosePolicy::default())
    }

    pub fn with_policy(symbol: &str, policy: ClosePolicy) -> Self {
        Self {
            symbol: symbol.to_string(),
            book: Book::default(),
            policy,
            initialised: false,
            next: 0,
            slots: VecDeque::new(),
            bstart: 0,
            bacc: BookAcc::fresh(None),
            feat: None,
            feat_ts: 0,
            last_px: None,
            wm_trade: Ms::MIN,
            wm_book: Ms::MIN,
            rx_trade: Ms::MIN / 2,
            rx_book: Ms::MIN / 2,
            trades: 0,
            late_trades: 0,
            late_books: 0,
        }
    }

    /// Number of candles held open (for tests and the heartbeat).
    pub fn open_candles(&self) -> usize {
        (((self.bstart - self.next) / CANDLE_MS) as usize).max(self.slots.len()).max(1)
    }

    fn init(&mut self, t: Ms) {
        if !self.initialised {
            self.initialised = true;
            self.next = bucket(t);
            self.bstart = self.next;
            self.feat_ts = self.next;
        }
    }

    fn slot_mut(&mut self, start: Ms) -> &mut Slot {
        let i = ((start - self.next) / CANDLE_MS) as usize;
        while self.slots.len() <= i {
            self.slots.push_back(Slot::empty());
        }
        &mut self.slots[i]
    }

    /// Accumulate the current book state's time weight up to `t` (clipped to the book cursor's candle).
    fn accrue(&mut self, t: Ms) {
        let t = t.min(self.bstart + CANDLE_MS);
        if let Some(f) = self.feat {
            let dt = (t - self.feat_ts.max(self.bstart)) as f64;
            if dt > 0.0 {
                for (acc, v) in self.bacc.tw.iter_mut().zip(feat_vec(&f)) {
                    *acc += v * dt;
                }
                self.bacc.tw_ms += dt;
            }
        }
        self.feat_ts = self.feat_ts.max(t);
    }

    /// Complete the book part of the cursor's candle and move the cursor one candle on.
    fn finish_book_candle(&mut self) {
        self.accrue(self.bstart + CANDLE_MS);
        let mut acc = std::mem::replace(&mut self.bacc, BookAcc::fresh(self.feat));
        acc.mid = self.feat.map(|f| f.mid);
        let start = self.bstart;
        self.slot_mut(start).book = Some(acc);
        self.bstart += CANDLE_MS;
        self.feat_ts = self.feat_ts.max(self.bstart);
    }

    /// Emit the oldest open candle.
    fn emit_front(&mut self, known_ts: Ms, out: &mut Vec<FlowRow>) {
        let start = self.next;
        if self.bstart == start {
            self.finish_book_candle();
        }
        let slot = self.slots.pop_front().unwrap_or_else(Slot::empty);
        let b = slot.book.unwrap_or_else(|| BookAcc::fresh(None));
        let t = slot.trade;
        if t.last.is_some() {
            self.last_px = t.last;
        }
        let vol = t.buy_vol + t.sell_vol;
        let twa = (b.tw_ms > 0.0).then(|| {
            let mut a = b.tw;
            for x in a.iter_mut() {
                *x /= b.tw_ms;
            }
            a
        });
        out.push(FlowRow {
            symbol: self.symbol.clone(),
            start,
            open: t.open,
            high: t.traded().then_some(t.high),
            low: t.traded().then_some(t.low),
            close: self.last_px,
            vwap: (vol > 0.0).then(|| t.pv / vol),
            buy_vol: t.buy_vol,
            sell_vol: t.sell_vol,
            buy_notional: t.buy_not,
            sell_notional: t.sell_not,
            n_buy: t.n_buy,
            n_sell: t.n_sell,
            max_trade: t.max_trade,
            mid: b.mid,
            twa,
            ofi: b.ofi_seen.then_some(b.ofi),
            span_bps: (b.span_min < f64::MAX).then_some(b.span_min),
            book_cov: (b.tw_ms / CANDLE_MS as f64).clamp(0.0, 1.0),
            book_updates: b.updates,
            // Exchange and local clocks differ (Bybit runs ~0.3-2.9 s ahead): a candle is never known before it ends.
            known_ts: known_ts.max(start + CANDLE_MS),
        });
        self.next += CANDLE_MS;
    }

    /// No open candle holds a trade or a book update.
    fn quiet(&self) -> bool {
        !self.slots.iter().any(Slot::has_data) && self.bacc.updates == 0
    }

    /// Drop every open candle and restart at `start` (an outage longer than MAX_FILL_CANDLES).
    fn jump_to(&mut self, start: Ms) {
        self.slots.clear();
        self.next = start;
        if self.bstart <= start {
            self.bstart = start;
            self.bacc = BookAcc::fresh(self.feat);
            self.feat_ts = start;
        }
    }

    /// Close every open candle (data jumped far ahead), then restart at `start`.
    fn close_all_and_jump(&mut self, start: Ms, known_ts: Ms, out: &mut Vec<FlowRow>) {
        let last = self.bstart.max(self.next + (self.slots.len() as Ms - 1).max(0) * CANDLE_MS);
        while self.next <= last && self.next < start {
            self.emit_front(known_ts, out);
        }
        self.jump_to(start);
    }

    fn too_far(&self, start: Ms) -> bool {
        (start - self.next) / CANDLE_MS > MAX_FILL_CANDLES
    }

    /// Exchange time up to which candles may close at local time `now` (module docs).
    fn ready_upto(&self, now: Ms) -> Ms {
        let p = self.policy;
        let trade_ok = if now - self.rx_trade >= p.grace_ms { Ms::MAX } else { self.wm_trade };
        let book_ok = if now - self.rx_book >= p.grace_ms { Ms::MAX } else { self.wm_book };
        self.wm_trade.min(self.wm_book).max((now - p.grace_ms).min(trade_ok).min(book_ok)).max(now - p.max_hold_ms)
    }

    /// Close every candle that is ready at local time `now` (call after each message and on a timer).
    pub fn advance(&mut self, now: Ms, out: &mut Vec<FlowRow>) {
        self.init(now - self.policy.grace_ms);
        let upto = self.ready_upto(now);
        self.close_until(upto, now, out);
    }

    /// Close every candle that ends at or before `upto` (exchange time), stamped `known_ts = now`.
    /// On shutdown the collector calls this with `upto = now`: ended candles are written, the open one is not.
    pub fn close_until(&mut self, upto: Ms, now: Ms, out: &mut Vec<FlowRow>) {
        if !self.initialised {
            return;
        }
        while self.next + CANDLE_MS <= upto {
            if self.quiet() {
                let target = bucket(upto);
                if self.too_far(target) && self.bstart <= target {
                    self.jump_to(target);
                    break;
                }
            }
            self.emit_front(now, out);
        }
    }

    pub fn on_trade(&mut self, ts: Ms, price: f64, size: f64, is_buy: bool, received_ts: Ms, out: &mut Vec<FlowRow>) {
        self.init(ts);
        let mut b = bucket(ts);
        if b < self.next {
            self.late_trades += 1;
            b = self.next;
        } else if self.too_far(b) {
            self.close_all_and_jump(b, received_ts, out);
        }
        self.slot_mut(b).trade.add(price, size, is_buy);
        self.trades += 1;
        self.wm_trade = self.wm_trade.max(ts);
        self.rx_trade = received_ts;
    }

    pub fn on_book(&mut self, ts: Ms, kind: &str, data: &serde_json::Value, received_ts: Ms, out: &mut Vec<FlowRow>) {
        self.init(ts);
        let b = bucket(ts);
        if b < self.bstart {
            self.late_books += 1;
        } else {
            if self.too_far(b) {
                self.close_all_and_jump(b, received_ts, out);
            }
            while self.bstart < b {
                self.finish_book_candle();
            }
        }
        self.wm_book = self.wm_book.max(ts);
        self.rx_book = received_ts;
        self.accrue(ts.max(self.bstart));
        let prev: Option<L1> = self.feat.map(|f| f.l1);
        if !self.book.apply(kind, data) {
            return;
        }
        self.feat = self.book.features();
        self.bacc.updates += 1;
        if let Some(f) = self.feat {
            self.bacc.span_min = self.bacc.span_min.min(f.span_bps);
            if let (Some(p), true) = (prev, kind == "delta") {
                self.bacc.ofi += ofi(&p, &f.l1);
                self.bacc.ofi_seen = true;
            }
        }
    }

    /// The connection dropped at local time `ts`: the book is invalid until the next snapshot.
    pub fn reset_book(&mut self, ts: Ms) {
        if self.initialised {
            let b = bucket(ts);
            if b > self.bstart && !self.too_far(b) {
                while self.bstart < b {
                    self.finish_book_candle();
                }
            }
            self.accrue(ts);
        }
        self.book.invalidate();
        self.feat = None;
    }
}

pub fn rows_to_batch(venue: &str, category: &str, rows: &[FlowRow]) -> Batch {
    let n = rows.len();
    let f32v = |g: &dyn Fn(&FlowRow) -> f64| rows.iter().map(|r| g(r) as f32).collect::<Vec<_>>();
    let twa = |k: usize| rows.iter().map(|r| r.twa.map(|a| a[k] as f32)).collect::<Vec<_>>();
    Batch::new()
        .utf8_const("venue", venue, n)
        .utf8_const("category", category, n)
        .utf8("symbol", rows.iter().map(|r| r.symbol.clone()).collect())
        .ts("event_ts", rows.iter().map(|r| r.start).collect())
        .f64_opt("open", rows.iter().map(|r| r.open).collect())
        .f64_opt("high", rows.iter().map(|r| r.high).collect())
        .f64_opt("low", rows.iter().map(|r| r.low).collect())
        .f64_opt("close", rows.iter().map(|r| r.close).collect())
        .f64_opt("vwap", rows.iter().map(|r| r.vwap).collect())
        .f32("buy_vol", f32v(&|r| r.buy_vol))
        .f32("sell_vol", f32v(&|r| r.sell_vol))
        .f32("buy_notional", f32v(&|r| r.buy_notional))
        .f32("sell_notional", f32v(&|r| r.sell_notional))
        .i32("n_buy", rows.iter().map(|r| r.n_buy).collect())
        .i32("n_sell", rows.iter().map(|r| r.n_sell).collect())
        .f32("max_trade", f32v(&|r| r.max_trade))
        .f64_opt("mid", rows.iter().map(|r| r.mid).collect())
        .f32_opt("spread_bps", twa(0))
        .f32_opt("micro_bps", twa(1))
        .f32_opt("obi_l1", twa(2))
        .f32_opt("obi_5", twa(3))
        .f32_opt("obi_10", twa(4))
        .f32_opt("obi_25", twa(5))
        .f32_opt("obi_50", twa(6))
        .f32_opt("obi_100", twa(7))
        .f32_opt("depth_bid_25", twa(8))
        .f32_opt("depth_ask_25", twa(9))
        .f32_opt("depth_bid_100", twa(10))
        .f32_opt("depth_ask_100", twa(11))
        .f32_opt("ofi", rows.iter().map(|r| r.ofi.map(|x| x as f32)).collect())
        .f32_opt("book_span_bps", rows.iter().map(|r| r.span_bps.map(|x| x as f32)).collect())
        .f32("book_cov", f32v(&|r| r.book_cov))
        .i32("book_updates", rows.iter().map(|r| r.book_updates).collect())
        .ts("known_ts", rows.iter().map(|r| r.known_ts).collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn snap() -> serde_json::Value {
        json!({"b": [["99", "1"]], "a": [["101", "1"]]})
    }

    #[test]
    fn trades_and_time_weighted_book() {
        let mut s = SymbolFlow::new("X");
        let mut out = vec![];
        let t0 = 1_000_000_002_000 - 1_000_000_002_000 % CANDLE_MS; // aligned
        s.on_book(t0, "snapshot", &snap(), t0, &mut out);
        s.on_trade(t0 + 100, 100.0, 2.0, true, t0 + 100, &mut out);
        s.on_trade(t0 + 200, 101.0, 1.0, false, t0 + 200, &mut out);
        // book becomes bid-heavy for the last third of the candle
        s.on_book(t0 + 2000, "delta", &json!({"b": [["99", "3"]], "a": []}), t0 + 2000, &mut out);
        s.advance(t0 + CANDLE_MS + 1500, &mut out);
        assert_eq!(out.len(), 1);
        let r = &out[0];
        assert_eq!((r.open, r.high, r.low, r.close), (Some(100.0), Some(101.0), Some(100.0), Some(101.0)));
        assert_eq!((r.buy_vol, r.sell_vol, r.n_buy, r.n_sell), (2.0, 1.0, 1, 1));
        assert!((r.vwap.unwrap() - 301.0 / 3.0).abs() < 1e-12);
        let obi_l1 = r.twa.unwrap()[2];
        // 2/3 of the time at 0, 1/3 at (3-1)/(3+1) = 0.5
        assert!((obi_l1 - 0.5 / 3.0).abs() < 1e-9, "time-weighted: {obi_l1}");
        assert_eq!(r.ofi, Some(2.0), "bid size grew by 2");
        assert!((r.book_cov - 1.0).abs() < 1e-9);
        assert_eq!(r.known_ts, t0 + CANDLE_MS + 1500);
    }

    #[test]
    fn quiet_candles_carry_close_and_book() {
        let mut s = SymbolFlow::new("X");
        let mut out = vec![];
        let t0 = 3_000_000;
        s.on_book(t0, "snapshot", &snap(), t0, &mut out);
        s.on_trade(t0 + 10, 100.0, 1.0, true, t0 + 10, &mut out);
        s.advance(t0 + 4 * CANDLE_MS + 1500, &mut out);
        assert_eq!(out.len(), 4);
        assert!(out[1..].iter().all(|r| r.n_buy + r.n_sell == 0 && r.close == Some(100.0) && r.open.is_none()));
        assert!(out[1..].iter().all(|r| (r.book_cov - 1.0).abs() < 1e-9), "book persists through quiet candles");
        assert!(out.iter().all(|r| r.mid == Some(100.0)));
    }

    #[test]
    fn reset_book_stops_accrual_until_snapshot() {
        let mut s = SymbolFlow::new("X");
        let mut out = vec![];
        let t0 = 3_000_000;
        s.on_book(t0, "snapshot", &snap(), t0, &mut out);
        s.reset_book(t0 + 1500);
        s.on_book(t0 + 2000, "delta", &json!({"b": [["99", "5"]], "a": []}), t0 + 2000, &mut out);
        s.advance(t0 + CANDLE_MS + 1500, &mut out);
        assert!((out[0].book_cov - 0.5).abs() < 1e-9, "only the first 1.5 s had a valid book");
        assert_eq!(out[0].ofi, None);
    }

    /// The WS02 defect: a book message stamped in the next candle arrives before trades of this candle.
    /// The old rule closed the candle on that book message and booked the trades one candle late.
    #[test]
    fn watermark_keeps_candle_open_for_late_trades_on_interleaved_streams() {
        let mut s = SymbolFlow::new("X");
        let mut out = vec![];
        let t0 = 3_000_000;
        // (kind, exchange ts, local receive ts): trades lag the book stream by ~400 ms.
        s.on_book(t0, "snapshot", &snap(), t0 + 5, &mut out);
        s.on_trade(t0 + 100, 100.0, 1.0, true, t0 + 500, &mut out);
        s.advance(t0 + 500, &mut out);
        s.on_book(t0 + CANDLE_MS + 100, "delta", &json!({"b": [["99", "2"]], "a": []}), t0 + CANDLE_MS + 105, &mut out);
        s.advance(t0 + CANDLE_MS + 105, &mut out);
        assert!(out.is_empty(), "book passed the end, trade stream has not: candle stays open");
        // late trades of the first candle arrive after the book has moved on
        s.on_trade(t0 + 2_900, 101.0, 2.0, false, t0 + CANDLE_MS + 300, &mut out);
        s.on_trade(t0 + 2_950, 102.0, 3.0, true, t0 + CANDLE_MS + 310, &mut out);
        s.advance(t0 + CANDLE_MS + 310, &mut out);
        assert!(out.is_empty());
        // first trade of the next candle: both watermarks past the end -> close now, with every trade inside
        s.on_trade(t0 + CANDLE_MS + 50, 103.0, 4.0, true, t0 + CANDLE_MS + 450, &mut out);
        s.advance(t0 + CANDLE_MS + 450, &mut out);
        assert_eq!(out.len(), 1);
        let r = &out[0];
        assert_eq!((r.n_buy, r.n_sell), (2, 1), "all three trades of [t0, t0+3s) in the first candle");
        assert_eq!((r.buy_vol, r.sell_vol), (4.0, 2.0));
        assert_eq!((r.open, r.close, r.high), (Some(100.0), Some(102.0), Some(102.0)));
        assert_eq!(r.known_ts, t0 + CANDLE_MS + 450, "known when the last watermark passed");
        assert!((r.book_cov - 1.0).abs() < 1e-9);
        assert_eq!(s.late_trades, 0);
        // the second candle keeps its own trade and the book update
        s.advance(t0 + 2 * CANDLE_MS + 1_500, &mut out);
        assert_eq!(out.len(), 2);
        assert_eq!((out[1].n_buy, out[1].buy_vol, out[1].book_updates), (1, 4.0, 1));
    }

    /// The same interleaved stream replayed with the old rule's assumption shows the difference: counts by candle
    /// must equal the exchange-time truth, whatever the arrival order (within the hold cap).
    #[test]
    fn watermark_matches_exchange_time_truth_on_a_synthetic_stream() {
        let mut s = SymbolFlow::new("X");
        let mut out = vec![];
        let t0 = 30_000_000;
        s.on_book(t0, "snapshot", &snap(), t0, &mut out);
        // trades every 250 ms for 30 s; each arrives 700 ms after it happened; book deltas every 100 ms arrive at once
        let mut events: Vec<(Ms, bool, Ms)> = vec![]; // (receive, is_trade, exchange ts)
        for k in 0..120 {
            let ts = t0 + 10 + k * 250;
            events.push((ts + 700, true, ts));
        }
        for k in 1..300 {
            let ts = t0 + k * 100;
            events.push((ts + 5, false, ts));
        }
        events.sort();
        for (rx, is_trade, ts) in events {
            if is_trade {
                s.on_trade(ts, 100.0, 1.0, true, rx, &mut out);
            } else {
                let q = format!("{}", 1 + (ts / 100) % 7);
                s.on_book(ts, "delta", &json!({"b": [["99", q]], "a": []}), rx, &mut out);
            }
            s.advance(rx, &mut out);
        }
        s.advance(t0 + 40_000, &mut out);
        let first10: Vec<i32> = out.iter().take(10).map(|r| r.n_buy).collect();
        assert_eq!(first10, vec![12; 10], "12 trades in every 3 s candle by exchange time");
        assert_eq!(s.late_trades, 0);
        assert!(out.iter().take(10).all(|r| r.known_ts >= r.start + CANDLE_MS));
    }

    #[test]
    fn quiet_symbol_closes_on_wall_clock_grace_and_lagging_trades_hold_until_cap() {
        let p = ClosePolicy { grace_ms: 1_500, max_hold_ms: 10_000 };
        let mut s = SymbolFlow::with_policy("X", p);
        let mut out = vec![];
        let t0 = 3_000_000;
        s.on_book(t0, "snapshot", &snap(), t0, &mut out);
        s.on_trade(t0 + 10, 100.0, 1.0, true, t0 + 20, &mut out);
        // quiet: both streams idle -> closes at end + grace
        s.advance(t0 + CANDLE_MS + 1_499, &mut out);
        assert!(out.is_empty());
        s.advance(t0 + CANDLE_MS + 1_500, &mut out);
        assert_eq!(out.len(), 1);
        // lagging trade stream: a trade of the 2nd candle keeps arriving late -> hold past grace
        s.on_trade(t0 + CANDLE_MS + 2_000, 101.0, 1.0, true, t0 + 2 * CANDLE_MS + 1_400, &mut out);
        s.advance(t0 + 2 * CANDLE_MS + 1_600, &mut out);
        assert_eq!(out.len(), 1, "trade stream active and behind the end: wait");
        s.advance(t0 + 2 * CANDLE_MS + 1_400 + 1_500, &mut out);
        assert_eq!(out.len(), 2, "trade stream went idle: close on grace");
        // a trade that lags past the hard cap goes to the oldest open candle and is counted
        s.on_trade(t0 + 2 * CANDLE_MS + 100, 102.0, 1.0, true, t0 + 6 * CANDLE_MS, &mut out);
        s.advance(t0 + 6 * CANDLE_MS, &mut out);
        assert_eq!(s.late_trades, 0, "still inside an open candle (held by the active trade stream)");
        s.on_trade(t0 + 100, 99.0, 1.0, false, t0 + 6 * CANDLE_MS + 10, &mut out);
        assert_eq!(s.late_trades, 1);
        let total: i32 = {
            s.advance(t0 + 20 * CANDLE_MS, &mut out);
            out.iter().map(|r| r.n_buy + r.n_sell).sum()
        };
        assert_eq!(total, 4, "totals are exact even for late trades");
        assert!(out.windows(2).all(|w| w[1].start == w[0].start + CANDLE_MS), "contiguous candles");
    }

    #[test]
    fn hard_cap_closes_while_a_stream_lags() {
        let p = ClosePolicy { grace_ms: 1_500, max_hold_ms: 10_000 };
        let mut s = SymbolFlow::with_policy("X", p);
        let mut out = vec![];
        let t0 = 3_000_000;
        s.on_book(t0, "snapshot", &snap(), t0, &mut out);
        // trades keep arriving every 500 ms but are all stamped at t0 + 10 (a stuck, lagging feed)
        for k in 0..30 {
            let rx = t0 + 500 * k;
            s.on_trade(t0 + 10, 100.0, 1.0, true, rx, &mut out);
            s.advance(rx, &mut out);
        }
        assert!(!out.is_empty(), "hard cap closed candles");
        assert!(out[0].known_ts >= out[0].start + CANDLE_MS + 10_000);
        assert!(s.open_candles() <= 5);
    }

    #[test]
    fn outage_skips_instead_of_filling() {
        let mut s = SymbolFlow::new("X");
        let mut out = vec![];
        let t0 = 3_000_000;
        s.on_book(t0, "snapshot", &snap(), t0, &mut out);
        s.on_trade(t0 + 10, 100.0, 1.0, true, t0 + 10, &mut out);
        // machine asleep for 2 h; first tick after resume
        let wake = t0 + 7_200_000;
        s.advance(wake, &mut out);
        assert!(out.len() <= MAX_FILL_CANDLES as usize + 2, "no 2,400 empty candles: {}", out.len());
        assert_eq!(out[0].n_buy, 1);
        // fresh data after the resume lands on the right candle
        s.on_trade(wake + 100, 105.0, 1.0, false, wake + 150, &mut out);
        s.advance(wake + CANDLE_MS + 2_000, &mut out);
        let r = out.iter().find(|r| r.n_sell == 1).unwrap();
        assert_eq!(r.start, bucket(wake + 100));
        assert_eq!(s.late_trades, 0);
    }

    #[test]
    fn far_future_event_closes_everything_then_restarts() {
        let mut s = SymbolFlow::new("X");
        let mut out = vec![];
        let t0 = 3_000_000;
        s.on_trade(t0 + 10, 100.0, 1.0, true, t0 + 10, &mut out);
        let later = t0 + 3_600_000;
        s.on_trade(later, 101.0, 2.0, true, later + 5, &mut out);
        assert_eq!(out.len(), 1, "the open candle was closed by the jump");
        assert_eq!(out[0].n_buy, 1);
        s.advance(later + CANDLE_MS + 1_500, &mut out);
        assert_eq!(out.last().unwrap().buy_vol, 2.0);
        assert_eq!(out.last().unwrap().start, bucket(later));
    }
}
