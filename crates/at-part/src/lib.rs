//! # at-part: ob200 -> 1 s book state + maker-side flows (PART study reducer)
//!
//! Mirrors `ai_trader.research.participants.FlowBook` / `reduce_book` exactly (the Python version is the reference and
//! is unit-tested; this is the fast path). For each bin of the publication time `ts` the reducer emits one row: the
//! book state at the END of the bin plus the flows accumulated inside it:
//!
//! * near-touch (<= `near_bp` from the pre-update mid) added / removed quoted notional per side and update counts;
//! * refills: size added back at the current best price after a decrease of that best level in the same bin;
//! * L1 order-flow imbalance (Cont, Kukanov, Stoikov 2014) per update, times the post-update mid (notional);
//! * number of updates that moved a best price.
//!
//! Snapshots reset the book and are not counted as flows. Deltas before the first snapshot are ignored. Bins without
//! messages emit nothing (the Python side builds the complete grid and carries the state forward).

use std::collections::{BTreeMap, HashSet};

use serde::Deserialize;

pub const BANDS_BP: [f64; 3] = [5.0, 10.0, 25.0];
/// ts, 16 state metrics, 11 flows.
pub const N_COLS: usize = 28;

#[derive(Deserialize)]
pub struct Msg<'a> {
    pub ts: i64,
    #[serde(rename = "type", borrow)]
    pub kind: &'a str,
    #[serde(borrow)]
    pub data: Data<'a>,
}

#[derive(Deserialize)]
pub struct Data<'a> {
    #[serde(default, borrow)]
    pub b: Vec<(&'a str, &'a str)>,
    #[serde(default, borrow)]
    pub a: Vec<(&'a str, &'a str)>,
    #[serde(default)]
    pub u: Option<i64>,
}

/// Positive finite f64 -> order-preserving key (bit pattern of a positive float is monotonic).
#[inline]
fn key(p: f64) -> u64 {
    p.to_bits()
}

#[inline]
fn unkey(k: u64) -> f64 {
    f64::from_bits(k)
}

#[derive(Default, Clone, Copy, Debug, PartialEq)]
pub struct Flows {
    pub n_updates: f64,
    pub add_bid: f64,
    pub add_ask: f64,
    pub rem_bid: f64,
    pub rem_ask: f64,
    pub n_add: f64,
    pub n_rem: f64,
    pub refill_bid: f64,
    pub refill_ask: f64,
    pub ofi: f64,
    pub n_best_moves: f64,
}

impl Flows {
    pub fn as_array(&self) -> [f64; 11] {
        [self.n_updates, self.add_bid, self.add_ask, self.rem_bid, self.rem_ask, self.n_add, self.n_rem, self.refill_bid,
         self.refill_ask, self.ofi, self.n_best_moves]
    }
}

/// One-update L1 OFI in base units (bid-side contribution minus ask-side contribution).
/// `now` = (bid px, bid size, ask px, ask size) after the update, `prev` = before.
pub fn cks_ofi(now: (f64, f64, f64, f64), prev: (f64, f64, f64, f64)) -> f64 {
    let ((pb, qb, pa, qa), (pb0, qb0, pa0, qa0)) = (now, prev);
    let mut e = 0.0;
    if pb >= pb0 {
        e += qb;
    }
    if pb <= pb0 {
        e -= qb0;
    }
    if pa <= pa0 {
        e -= qa;
    }
    if pa >= pa0 {
        e += qa0;
    }
    e
}

pub struct FlowBook {
    pub bids: BTreeMap<u64, f64>,
    pub asks: BTreeMap<u64, f64>,
    pub ready: bool,
    near: f64,
    pub acc: Flows,
    dec_b: HashSet<u64>,
    dec_a: HashSet<u64>,
}

impl FlowBook {
    pub fn new(near_bp: f64) -> Self {
        Self { bids: BTreeMap::new(), asks: BTreeMap::new(), ready: false, near: near_bp * 1e-4, acc: Flows::default(),
               dec_b: HashSet::new(), dec_a: HashSet::new() }
    }

    pub fn reset_bin(&mut self) {
        self.acc = Flows::default();
        self.dec_b.clear();
        self.dec_a.clear();
    }

    pub fn best_bid(&self) -> f64 {
        self.bids.keys().next_back().map_or(f64::NAN, |k| unkey(*k))
    }

    pub fn best_ask(&self) -> f64 {
        self.asks.keys().next().map_or(f64::NAN, |k| unkey(*k))
    }

    fn l1(&self) -> (f64, f64, f64, f64) {
        let (bb, ba) = (self.best_bid(), self.best_ask());
        let qb = if bb.is_nan() { 0.0 } else { *self.bids.get(&key(bb)).unwrap_or(&0.0) };
        let qa = if ba.is_nan() { 0.0 } else { *self.asks.get(&key(ba)).unwrap_or(&0.0) };
        (bb, qb, ba, qa)
    }

    /// Apply one message (levels as parsed floats). Returns false for a delta before any snapshot.
    pub fn apply(&mut self, snapshot: bool, bids: &[(f64, f64)], asks: &[(f64, f64)]) -> bool {
        if snapshot {
            self.bids.clear();
            self.asks.clear();
            for &(p, s) in bids {
                if s > 0.0 {
                    self.bids.insert(key(p), s);
                }
            }
            for &(p, s) in asks {
                if s > 0.0 {
                    self.asks.insert(key(p), s);
                }
            }
            self.ready = true;
            return true;
        }
        if !self.ready {
            return false;
        }
        let (pb0, qb0, pa0, qa0) = self.l1();
        let mid = 0.5 * (pb0 + pa0);
        let thr = if mid.is_nan() { 0.0 } else { mid * self.near };
        // bids: the running best is the max key after each level update (same as the Python cached best)
        for &(px, sz) in bids {
            let best = self.best_bid();
            let old = *self.bids.get(&key(px)).unwrap_or(&0.0);
            let dv = sz - old;
            if dv != 0.0 && (px - mid).abs() <= thr {
                if dv > 0.0 {
                    self.acc.add_bid += dv * px;
                    self.acc.n_add += 1.0;
                    if px == best && self.dec_b.contains(&key(px)) {
                        self.acc.refill_bid += dv * px;
                    }
                } else {
                    self.acc.rem_bid -= dv * px;
                    self.acc.n_rem += 1.0;
                    if px == best {
                        self.dec_b.insert(key(px));
                    }
                }
            }
            if sz == 0.0 {
                self.bids.remove(&key(px));
            } else {
                self.bids.insert(key(px), sz);
            }
        }
        for &(px, sz) in asks {
            let best = self.best_ask();
            let old = *self.asks.get(&key(px)).unwrap_or(&0.0);
            let dv = sz - old;
            if dv != 0.0 && (px - mid).abs() <= thr {
                if dv > 0.0 {
                    self.acc.add_ask += dv * px;
                    self.acc.n_add += 1.0;
                    if px == best && self.dec_a.contains(&key(px)) {
                        self.acc.refill_ask += dv * px;
                    }
                } else {
                    self.acc.rem_ask -= dv * px;
                    self.acc.n_rem += 1.0;
                    if px == best {
                        self.dec_a.insert(key(px));
                    }
                }
            }
            if sz == 0.0 {
                self.asks.remove(&key(px));
            } else {
                self.asks.insert(key(px), sz);
            }
        }
        let (pb, qb, pa, qa) = self.l1();
        if !pb0.is_nan() && !pa0.is_nan() && !pb.is_nan() && !pa.is_nan() {
            self.acc.ofi += cks_ofi((pb, qb, pa, qa), (pb0, qb0, pa0, qa0)) * 0.5 * (pb + pa);
            if pb != pb0 || pa != pa0 {
                self.acc.n_best_moves += 1.0;
            }
        }
        self.acc.n_updates += 1.0;
        true
    }

    /// (bid1, ask1, bid1_size, ask1_size, mid, rel_spread_bp, depth bid/ask per band, imb_5, imb_10, span_bid, span_ask).
    pub fn metrics(&self) -> [f64; 16] {
        let mut out = [f64::NAN; 16];
        let (Some((kb, sb)), Some((ka, sa))) = (self.bids.iter().next_back(), self.asks.iter().next()) else {
            return out;
        };
        let (b1, a1) = (unkey(*kb), unkey(*ka));
        let mid = 0.5 * (b1 + a1);
        out[0] = b1;
        out[1] = a1;
        out[2] = *sb;
        out[3] = *sa;
        out[4] = mid;
        out[5] = (a1 - b1) / mid * 1e4;
        let mut dep = [(0.0, 0.0); 3];
        for (i, b) in BANDS_BP.iter().enumerate() {
            let (lo, hi) = (mid * (1.0 - b * 1e-4), mid * (1.0 + b * 1e-4));
            let db: f64 = self.bids.iter().rev().map(|(k, s)| (unkey(*k), *s)).take_while(|(p, _)| *p >= lo).map(|(p, s)| p * s).sum();
            let da: f64 = self.asks.iter().map(|(k, s)| (unkey(*k), *s)).take_while(|(p, _)| *p <= hi).map(|(p, s)| p * s).sum();
            dep[i] = (db, da);
            out[6 + 2 * i] = db;
            out[7 + 2 * i] = da;
        }
        for (j, i) in [0usize, 1].iter().enumerate() {
            let (db, da) = dep[*i];
            out[12 + j] = if db + da > 0.0 { (db - da) / (db + da) } else { f64::NAN };
        }
        let bmin = unkey(*self.bids.keys().next().unwrap());
        let amax = unkey(*self.asks.keys().next_back().unwrap());
        out[14] = (mid - bmin) / mid * 1e4;
        out[15] = (amax - mid) / mid * 1e4;
        out
    }

    pub fn row(&self, ts: i64) -> [f64; N_COLS] {
        let mut r = [0.0; N_COLS];
        r[0] = ts as f64;
        r[1..17].copy_from_slice(&self.metrics());
        r[17..28].copy_from_slice(&self.acc.as_array());
        r
    }
}

#[derive(Default, Debug)]
pub struct Qa {
    pub messages: u64,
    pub snapshots: u64,
    pub before_snapshot: u64,
    pub u_gaps: u64,
    pub bad_lines: u64,
}

fn levels(v: &[(&str, &str)]) -> Vec<(f64, f64)> {
    v.iter().filter_map(|(p, s)| Some((p.parse::<f64>().ok()?, s.parse::<f64>().ok()?))).collect()
}

/// Reduce a stream of JSON lines; `emit` receives every finished bin row.
pub fn reduce_lines<R: std::io::BufRead>(mut rd: R, bin_ms: i64, near_bp: f64, mut emit: impl FnMut(&[f64; N_COLS])) -> anyhow::Result<Qa> {
    let mut fb = FlowBook::new(near_bp);
    let mut qa = Qa::default();
    let mut cur: Option<i64> = None;
    let mut last_u: Option<i64> = None;
    let mut line = String::new();
    loop {
        line.clear();
        if rd.read_line(&mut line)? == 0 {
            break;
        }
        let s = line.trim();
        if s.is_empty() {
            continue;
        }
        let msg: Msg = match serde_json::from_str(s) {
            Ok(m) => m,
            Err(_) => {
                qa.bad_lines += 1;
                continue;
            }
        };
        qa.messages += 1;
        let b = msg.ts.div_euclid(bin_ms);
        if let Some(c) = cur
            && b != c
            && fb.ready
        {
            emit(&fb.row((c + 1) * bin_ms));
            fb.reset_bin();
        }
        cur = Some(b);
        let snap = msg.kind == "snapshot";
        if snap {
            qa.snapshots += 1;
        }
        if !fb.apply(snap, &levels(&msg.data.b), &levels(&msg.data.a)) {
            qa.before_snapshot += 1;
            continue;
        }
        let u = msg.data.u;
        if !snap && let (Some(l), Some(u)) = (last_u, u) && u != l + 1 {
            qa.u_gaps += 1;
        }
        last_u = u;
    }
    if let Some(c) = cur
        && fb.ready
    {
        emit(&fb.row((c + 1) * bin_ms));
    }
    Ok(qa)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ofi_signs() {
        assert_eq!(cks_ofi((101.0, 2.0, 102.0, 5.0), (100.0, 3.0, 102.0, 5.0)), 2.0);
        assert_eq!(cks_ofi((100.0, 3.0, 101.0, 4.0), (100.0, 3.0, 102.0, 5.0)), -4.0);
        assert_eq!(cks_ofi((100.0, 1.0, 102.0, 5.0), (100.0, 3.0, 102.0, 5.0)), -2.0);
    }

    #[test]
    fn flows_match_python_reference() {
        // same scenario as tests/test_participants.py::test_flowbook_adds_removals_refill_and_best
        let mut fb = FlowBook::new(50.0);
        fb.apply(true, &[(100.0, 1.0), (99.9, 2.0)], &[(100.1, 1.0), (100.2, 3.0)]);
        fb.reset_bin();
        fb.apply(false, &[(100.0, 0.4)], &[]);
        fb.apply(false, &[(100.0, 1.0)], &[]);
        fb.apply(false, &[(100.05, 2.0)], &[]);
        fb.apply(false, &[], &[(100.1, 0.0)]);
        let a = fb.acc;
        assert!((a.rem_bid - 60.0).abs() < 1e-9);
        assert!((a.refill_bid - 60.0).abs() < 1e-9);
        assert!((a.add_bid - (60.0 + 2.0 * 100.05)).abs() < 1e-9);
        assert!((a.rem_ask - 100.1).abs() < 1e-9);
        assert_eq!(fb.best_bid(), 100.05);
        assert_eq!(fb.best_ask(), 100.2);
        assert_eq!(a.n_updates, 4.0);
        assert_eq!(a.n_best_moves, 2.0);
    }

    #[test]
    fn before_snapshot_and_bins() {
        let lines = concat!(
            "{\"ts\":5,\"type\":\"delta\",\"data\":{\"b\":[[\"1\",\"1\"]],\"a\":[],\"u\":0}}\n",
            "{\"ts\":10,\"type\":\"snapshot\",\"data\":{\"b\":[[\"100\",\"1\"]],\"a\":[[\"100.1\",\"1\"]],\"u\":1}}\n",
            "{\"ts\":1500,\"type\":\"delta\",\"data\":{\"b\":[[\"100\",\"3\"]],\"a\":[],\"u\":2}}\n",
            "{\"ts\":1600,\"type\":\"delta\",\"data\":{\"b\":[[\"100\",\"2\"]],\"a\":[],\"u\":4}}\n",
        );
        let mut rows = Vec::new();
        let qa = reduce_lines(lines.as_bytes(), 1000, 5.0, |r| rows.push(*r)).unwrap();
        assert_eq!(qa.before_snapshot, 1);
        assert_eq!(qa.u_gaps, 1);
        assert_eq!(rows.len(), 2);
        assert_eq!(rows[0][0], 1000.0);
        assert_eq!(rows[1][0], 2000.0);
        assert!((rows[1][18] - 200.0).abs() < 1e-9); // add_bid in the second bin
        assert!((rows[1][20] - 100.0).abs() < 1e-9); // rem_bid
        assert_eq!(rows[1][3], 2.0); // bid1_size at bin end
    }

    #[test]
    fn depth_bands() {
        let mut fb = FlowBook::new(5.0);
        fb.apply(true, &[(100.0, 1.0), (99.0, 1.0)], &[(100.02, 2.0), (101.0, 1.0)]);
        let m = fb.metrics();
        let mid = 100.01;
        assert!((m[4] - mid).abs() < 1e-12);
        assert!((m[6] - 100.0).abs() < 1e-9); // bid depth 5 bp: only 100
        assert!((m[7] - 200.04).abs() < 1e-9);
        assert!((m[11] - (100.02 * 2.0)).abs() < 1e-9); // ask 25 bp: 101 is ~99 bp away
    }
}
