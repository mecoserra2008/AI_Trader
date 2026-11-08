//! L2 order book maintained from Bybit `orderbook.{depth}.{symbol}` messages.
//! A snapshot replaces the book; deltas update levels (size "0" deletes). Until a snapshot has
//! been received, or after a reconnect, the book is invalid and produces no features.

use std::cmp::Ordering;
use std::collections::BTreeMap;

use serde_json::Value;

use crate::BANDS_BPS;

#[derive(Clone, Copy, Debug, PartialEq)]
struct Px(f64);

impl Eq for Px {}

impl PartialOrd for Px {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Px {
    fn cmp(&self, other: &Self) -> Ordering {
        self.0.total_cmp(&other.0)
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct L1 {
    pub bid: f64,
    pub bid_qty: f64,
    pub ask: f64,
    pub ask_qty: f64,
}

/// Book state summarised at one instant.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BookFeat {
    pub l1: L1,
    pub mid: f64,
    pub spread_bps: f64,
    /// (microprice − mid) / mid in bps; microprice = (bid·ask_qty + ask·bid_qty) / (bid_qty + ask_qty).
    pub micro_bps: f64,
    pub obi_l1: f64,
    /// Notional imbalance (bid − ask) / (bid + ask) within each band of BANDS_BPS.
    pub obi: [f64; 5],
    pub depth_bid: [f64; 5],
    pub depth_ask: [f64; 5],
    /// Distance (bps) to the farthest level received on the thinner side: bands beyond it are truncated.
    pub span_bps: f64,
}

#[derive(Debug, Default)]
pub struct Book {
    bids: BTreeMap<Px, f64>,
    asks: BTreeMap<Px, f64>,
    pub valid: bool,
    pub update_id: i64,
}

fn levels(v: &Value) -> impl Iterator<Item = (f64, f64)> + '_ {
    v.as_array().into_iter().flatten().filter_map(|lv| {
        let p = lv.get(0)?.as_str()?.parse::<f64>().ok()?;
        let q = lv.get(1)?.as_str()?.parse::<f64>().ok()?;
        Some((p, q))
    })
}

impl Book {
    pub fn invalidate(&mut self) {
        self.bids.clear();
        self.asks.clear();
        self.valid = false;
    }

    /// Apply one message. Returns false if it was ignored (a delta before any snapshot).
    pub fn apply(&mut self, kind: &str, data: &Value) -> bool {
        match kind {
            "snapshot" => {
                self.bids.clear();
                self.asks.clear();
                self.valid = true;
            }
            "delta" if self.valid => {}
            _ => return false,
        }
        for (p, q) in levels(&data["b"]) {
            if q == 0.0 {
                self.bids.remove(&Px(p));
            } else {
                self.bids.insert(Px(p), q);
            }
        }
        for (p, q) in levels(&data["a"]) {
            if q == 0.0 {
                self.asks.remove(&Px(p));
            } else {
                self.asks.insert(Px(p), q);
            }
        }
        self.update_id = data["u"].as_i64().unwrap_or(self.update_id);
        true
    }

    pub fn l1(&self) -> Option<L1> {
        let (b, bq) = self.bids.iter().next_back()?;
        let (a, aq) = self.asks.iter().next()?;
        Some(L1 { bid: b.0, bid_qty: *bq, ask: a.0, ask_qty: *aq })
    }

    pub fn features(&self) -> Option<BookFeat> {
        if !self.valid {
            return None;
        }
        let l1 = self.l1()?;
        if l1.bid >= l1.ask || l1.bid <= 0.0 {
            return None; // crossed / broken book: treat as invalid rather than emit garbage
        }
        let mid = 0.5 * (l1.bid + l1.ask);
        let micro = (l1.bid * l1.ask_qty + l1.ask * l1.bid_qty) / (l1.bid_qty + l1.ask_qty);
        let mut depth_bid = [0.0; 5];
        let mut depth_ask = [0.0; 5];
        for (p, q) in self.bids.iter().rev() {
            let d = (mid - p.0) / mid * 1e4;
            if d > BANDS_BPS[4] {
                break;
            }
            for (k, b) in BANDS_BPS.iter().enumerate() {
                if d <= *b {
                    depth_bid[k] += p.0 * q;
                }
            }
        }
        for (p, q) in self.asks.iter() {
            let d = (p.0 - mid) / mid * 1e4;
            if d > BANDS_BPS[4] {
                break;
            }
            for (k, b) in BANDS_BPS.iter().enumerate() {
                if d <= *b {
                    depth_ask[k] += p.0 * q;
                }
            }
        }
        let mut obi = [0.0; 5];
        for k in 0..5 {
            let t = depth_bid[k] + depth_ask[k];
            obi[k] = if t > 0.0 { (depth_bid[k] - depth_ask[k]) / t } else { 0.0 };
        }
        let far_bid = self.bids.keys().next().map(|p| (mid - p.0) / mid * 1e4).unwrap_or(0.0);
        let far_ask = self.asks.keys().next_back().map(|p| (p.0 - mid) / mid * 1e4).unwrap_or(0.0);
        Some(BookFeat {
            l1,
            mid,
            spread_bps: (l1.ask - l1.bid) / mid * 1e4,
            micro_bps: (micro - mid) / mid * 1e4,
            obi_l1: (l1.bid_qty - l1.ask_qty) / (l1.bid_qty + l1.ask_qty),
            obi,
            depth_bid,
            depth_ask,
            span_bps: far_bid.min(far_ask),
        })
    }
}

/// L1 order-flow imbalance between two book states (Cont, Kukanov & Stoikov 2014), base units.
pub fn ofi(prev: &L1, cur: &L1) -> f64 {
    let mut e = 0.0;
    if cur.bid >= prev.bid {
        e += cur.bid_qty;
    }
    if cur.bid <= prev.bid {
        e -= prev.bid_qty;
    }
    if cur.ask <= prev.ask {
        e -= cur.ask_qty;
    }
    if cur.ask >= prev.ask {
        e += prev.ask_qty;
    }
    e
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn book() -> Book {
        let mut b = Book::default();
        assert!(b.apply("snapshot", &json!({"b": [["99.99", "5"], ["99.95", "10"], ["99.0", "100"]], "a": [["100.01", "1"], ["100.05", "2"]], "u": 1})));
        b
    }

    #[test]
    fn snapshot_delta_and_features() {
        let mut b = book();
        let f = b.features().unwrap();
        assert!((f.mid - 100.0).abs() < 1e-9);
        assert!((f.spread_bps - 2.0).abs() < 1e-6);
        assert!(f.obi_l1 > 0.0, "5 on the bid vs 1 on the ask");
        assert!(f.micro_bps > 0.0, "microprice leans toward the thin side (ask)");
        // within 5 bps: bids 99.99, 99.95; asks 100.01, 100.05
        assert!((f.depth_bid[0] - (99.99 * 5.0 + 99.95 * 10.0)).abs() < 1e-6);
        assert!((f.depth_ask[0] - (100.01 + 100.05 * 2.0)).abs() < 1e-6);
        assert!(f.depth_bid[4] > f.depth_bid[0], "99.0 is inside 100 bps");
        // delete the best ask, add a new one
        assert!(b.apply("delta", &json!({"b": [], "a": [["100.01", "0"], ["100.02", "3"]], "u": 2})));
        assert_eq!(b.l1().unwrap().ask, 100.02);
    }

    #[test]
    fn deltas_before_snapshot_are_ignored_and_crossed_books_invalid() {
        let mut b = Book::default();
        assert!(!b.apply("delta", &json!({"b": [["1", "1"]], "a": []})));
        assert!(b.features().is_none());
        let mut b = book();
        b.apply("delta", &json!({"b": [["100.5", "1"]], "a": []}));
        assert!(b.features().is_none(), "bid above ask is rejected");
    }

    #[test]
    fn ofi_signs() {
        let p = L1 { bid: 100.0, bid_qty: 5.0, ask: 101.0, ask_qty: 5.0 };
        assert_eq!(ofi(&p, &L1 { bid_qty: 8.0, ..p }), 3.0, "bid size added = buy pressure");
        assert_eq!(ofi(&p, &L1 { ask_qty: 9.0, ..p }), -4.0, "ask size added = sell pressure");
        assert_eq!(ofi(&p, &L1 { bid: 100.5, bid_qty: 2.0, ..p }), 2.0, "bid steps up");
    }
}
