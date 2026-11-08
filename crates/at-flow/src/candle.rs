//! Per-symbol 3-second aggregation on **exchange time**.
//!
//! Trades accumulate OHLC / buy / sell / counts. Book features are *time-weighted* inside the
//! candle: each book state is weighted by how long it was live (a book that flickers for 5 ms
//! barely counts). OFI is summed over every book update. A candle is finalised once a later
//! event arrives, or once the wall clock passes its end plus a grace period (quiet symbols).

use at_core::time::Ms;
use at_store::Batch;

use crate::book::{Book, BookFeat, L1, ofi};

pub const CANDLE_MS: Ms = 3_000;
/// Gaps longer than this (e.g. after an outage) are skipped instead of emitting empty candles.
const MAX_FILL_CANDLES: i64 = 200;
const NF: usize = 12;

fn feat_vec(f: &BookFeat) -> [f64; NF] {
    [
        f.spread_bps, f.micro_bps, f.obi_l1,
        f.obi[0], f.obi[1], f.obi[2], f.obi[3], f.obi[4],
        f.depth_bid[2], f.depth_ask[2], f.depth_bid[4], f.depth_ask[4],
    ]
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

#[derive(Debug)]
pub struct SymbolFlow {
    pub symbol: String,
    pub book: Book,
    start: Ms,
    initialised: bool,
    // trades
    open: Option<f64>,
    high: f64,
    low: f64,
    last_px: Option<f64>,
    pv: f64,
    buy_vol: f64,
    sell_vol: f64,
    buy_not: f64,
    sell_not: f64,
    n_buy: i32,
    n_sell: i32,
    max_trade: f64,
    // book
    feat: Option<BookFeat>,
    feat_ts: Ms,
    tw: [f64; NF],
    tw_ms: f64,
    ofi: f64,
    ofi_seen: bool,
    span_min: f64,
    updates: i32,
}

impl SymbolFlow {
    pub fn new(symbol: &str) -> Self {
        Self {
            symbol: symbol.to_string(),
            book: Book::default(),
            start: 0,
            initialised: false,
            open: None,
            high: f64::MIN,
            low: f64::MAX,
            last_px: None,
            pv: 0.0,
            buy_vol: 0.0,
            sell_vol: 0.0,
            buy_not: 0.0,
            sell_not: 0.0,
            n_buy: 0,
            n_sell: 0,
            max_trade: 0.0,
            feat: None,
            feat_ts: 0,
            tw: [0.0; NF],
            tw_ms: 0.0,
            ofi: 0.0,
            ofi_seen: false,
            span_min: f64::MAX,
            updates: 0,
        }
    }

    fn reset_candle(&mut self, start: Ms) {
        self.start = start;
        self.open = None;
        self.high = f64::MIN;
        self.low = f64::MAX;
        self.pv = 0.0;
        self.buy_vol = 0.0;
        self.sell_vol = 0.0;
        self.buy_not = 0.0;
        self.sell_not = 0.0;
        self.n_buy = 0;
        self.n_sell = 0;
        self.max_trade = 0.0;
        self.tw = [0.0; NF];
        self.tw_ms = 0.0;
        self.ofi = 0.0;
        self.ofi_seen = false;
        self.span_min = self.feat.map(|f| f.span_bps).unwrap_or(f64::MAX);
        self.updates = 0;
        self.feat_ts = self.feat_ts.max(start);
    }

    /// Accumulate the current book state's time weight up to `t` (clipped to the candle).
    fn accrue(&mut self, t: Ms) {
        let t = t.min(self.start + CANDLE_MS);
        if let Some(f) = self.feat {
            let dt = (t - self.feat_ts.max(self.start)) as f64;
            if dt > 0.0 {
                for (acc, v) in self.tw.iter_mut().zip(feat_vec(&f)) {
                    *acc += v * dt;
                }
                self.tw_ms += dt;
            }
        }
        self.feat_ts = self.feat_ts.max(t);
    }

    fn finalize(&mut self, known_ts: Ms) -> FlowRow {
        self.accrue(self.start + CANDLE_MS);
        let traded = self.n_buy + self.n_sell > 0;
        let vol = self.buy_vol + self.sell_vol;
        let twa = (self.tw_ms > 0.0).then(|| {
            let mut a = self.tw;
            for x in a.iter_mut() {
                *x /= self.tw_ms;
            }
            a
        });
        FlowRow {
            symbol: self.symbol.clone(),
            start: self.start,
            open: self.open,
            high: traded.then_some(self.high),
            low: traded.then_some(self.low),
            close: self.last_px,
            vwap: (vol > 0.0).then(|| self.pv / vol),
            buy_vol: self.buy_vol,
            sell_vol: self.sell_vol,
            buy_notional: self.buy_not,
            sell_notional: self.sell_not,
            n_buy: self.n_buy,
            n_sell: self.n_sell,
            max_trade: self.max_trade,
            mid: self.feat.map(|f| f.mid),
            twa,
            ofi: self.ofi_seen.then_some(self.ofi),
            span_bps: (self.span_min < f64::MAX).then_some(self.span_min),
            book_cov: (self.tw_ms / CANDLE_MS as f64).clamp(0.0, 1.0),
            book_updates: self.updates,
            // Exchange and local clocks differ (Bybit runs ~0.3 s ahead): a candle can never be
            // known before it ends on exchange time.
            known_ts: known_ts.max(self.start + CANDLE_MS),
        }
    }

    /// Close every candle that ends at or before `t`, appending them to `out`.
    pub fn roll_to(&mut self, t: Ms, known_ts: Ms, out: &mut Vec<FlowRow>) {
        let bucket = t - t.rem_euclid(CANDLE_MS);
        if !self.initialised {
            self.initialised = true;
            self.reset_candle(bucket);
            return;
        }
        if bucket <= self.start {
            return;
        }
        out.push(self.finalize(known_ts));
        let gap = (bucket - self.start) / CANDLE_MS;
        if gap > MAX_FILL_CANDLES {
            self.reset_candle(bucket);
            return;
        }
        let mut s = self.start + CANDLE_MS;
        while s < bucket {
            self.reset_candle(s);
            out.push(self.finalize(known_ts));
            s += CANDLE_MS;
        }
        self.reset_candle(bucket);
    }

    pub fn on_trade(&mut self, ts: Ms, price: f64, size: f64, is_buy: bool, known_ts: Ms, out: &mut Vec<FlowRow>) {
        self.roll_to(ts, known_ts, out);
        self.open.get_or_insert(price);
        self.high = self.high.max(price);
        self.low = self.low.min(price);
        self.last_px = Some(price);
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

    pub fn on_book(&mut self, ts: Ms, kind: &str, data: &serde_json::Value, known_ts: Ms, out: &mut Vec<FlowRow>) {
        self.roll_to(ts, known_ts, out);
        self.accrue(ts.max(self.start));
        let prev: Option<L1> = self.feat.map(|f| f.l1);
        if !self.book.apply(kind, data) {
            return;
        }
        self.feat = self.book.features();
        self.updates += 1;
        if let Some(f) = self.feat {
            self.span_min = self.span_min.min(f.span_bps);
            if let (Some(p), true) = (prev, kind == "delta") {
                self.ofi += ofi(&p, &f.l1);
                self.ofi_seen = true;
            }
        }
    }

    pub fn reset_book(&mut self, ts: Ms) {
        self.accrue(ts);
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

    #[test]
    fn trades_and_time_weighted_book() {
        let mut s = SymbolFlow::new("X");
        let mut out = vec![];
        let t0 = 1_000_000_002_000 - 1_000_000_002_000 % CANDLE_MS; // aligned
        s.on_book(t0, "snapshot", &json!({"b": [["99", "1"]], "a": [["101", "1"]]}), t0, &mut out);
        s.on_trade(t0 + 100, 100.0, 2.0, true, t0, &mut out);
        s.on_trade(t0 + 200, 101.0, 1.0, false, t0, &mut out);
        // book becomes bid-heavy for the last third of the candle
        s.on_book(t0 + 2000, "delta", &json!({"b": [["99", "3"]], "a": []}), t0, &mut out);
        s.roll_to(t0 + CANDLE_MS, t0 + CANDLE_MS + 1500, &mut out);
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
    }

    #[test]
    fn quiet_candles_carry_close_and_book() {
        let mut s = SymbolFlow::new("X");
        let mut out = vec![];
        let t0 = 3_000_000;
        s.on_book(t0, "snapshot", &json!({"b": [["99", "1"]], "a": [["101", "1"]]}), t0, &mut out);
        s.on_trade(t0 + 10, 100.0, 1.0, true, t0, &mut out);
        s.roll_to(t0 + 4 * CANDLE_MS, 0, &mut out);
        assert_eq!(out.len(), 4);
        assert!(out[1..].iter().all(|r| r.n_buy + r.n_sell == 0 && r.close == Some(100.0) && r.open.is_none()));
        assert!(out[1..].iter().all(|r| (r.book_cov - 1.0).abs() < 1e-9), "book persists through quiet candles");
    }

    #[test]
    fn reset_book_stops_accrual_until_snapshot() {
        let mut s = SymbolFlow::new("X");
        let mut out = vec![];
        let t0 = 3_000_000;
        s.on_book(t0, "snapshot", &json!({"b": [["99", "1"]], "a": [["101", "1"]]}), t0, &mut out);
        s.reset_book(t0 + 1500);
        s.on_book(t0 + 2000, "delta", &json!({"b": [["99", "5"]], "a": []}), t0, &mut out);
        s.roll_to(t0 + CANDLE_MS, 0, &mut out);
        assert!((out[0].book_cov - 0.5).abs() < 1e-9, "only the first 1.5 s had a valid book");
        assert_eq!(out[0].ofi, None);
    }
}
