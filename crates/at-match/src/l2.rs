//! Bybit `orderbook.<depth>` messages from the engine's book. The first message is a snapshot (u = 1, the Bybit
//! signal for "reset your local book"); each later message is a delta holding exactly the levels of the top-`depth`
//! window that changed since the previous message: size "0" removes a level, any other size inserts or replaces it.
//! A level that drops out of the window is sent as size "0" and a level that enters it is sent with its size, as
//! Bybit does for a depth-limited book. `u` increases by 1 per message, `seq` is the engine's book version, `ts` is
//! the publication time and `cts` the time of the last event applied.
//!
//! Line shape (prices / sizes are strings with the tick / step decimals):
//! `{"topic":"orderbook.200.BTCUSDT","type":"delta","ts":..,"data":{"s":"BTCUSDT","b":[["p","s"],..],"a":[..],"u":..,"seq":..},"cts":..}`

use std::fmt::Write as _;
use std::io::{self, Write};

use crate::book::{L2, OrderBook};
use crate::grid::Grid;

pub struct Publisher {
    symbol: String,
    topic: String,
    depth: usize,
    last: L2,
    cur: L2,
    u: u64,
    buf: String,
    pub snapshots: u64,
    pub deltas: u64,
}

/// Changed levels between two best-first level lists (`better(a, b)`: a is closer to the market than b).
fn diff(old: &[(i64, i64)], new: &[(i64, i64)], better: impl Fn(i64, i64) -> bool, out: &mut Vec<(i64, i64)>) {
    let (mut i, mut j) = (0, 0);
    while i < old.len() || j < new.len() {
        match (old.get(i), new.get(j)) {
            (Some(&(po, so)), Some(&(pn, sn))) if po == pn => {
                if so != sn {
                    out.push((pn, sn));
                }
                i += 1;
                j += 1;
            }
            (Some(&(po, _)), Some(&(pn, sn))) => {
                if better(pn, po) {
                    out.push((pn, sn));
                    j += 1;
                } else {
                    out.push((po, 0));
                    i += 1;
                }
            }
            (Some(&(po, _)), None) => {
                out.push((po, 0));
                i += 1;
            }
            (None, Some(&(pn, sn))) => {
                out.push((pn, sn));
                j += 1;
            }
            (None, None) => unreachable!(),
        }
    }
}

impl Publisher {
    pub fn new(symbol: &str, depth: usize) -> Self {
        Self { symbol: symbol.to_string(), topic: format!("orderbook.{depth}.{symbol}"), depth, last: L2::default(), cur: L2::default(), u: 0, buf: String::new(), snapshots: 0, deltas: 0 }
    }

    /// Last published book (what a consumer of the stream holds).
    pub fn published(&self) -> &L2 {
        &self.last
    }

    /// Write a snapshot (first call) or a delta (later calls, only if the window changed). Returns true if a line was
    /// written.
    pub fn publish<W: Write + ?Sized>(&mut self, book: &OrderBook, ts: i64, cts: i64, out: &mut W) -> io::Result<bool> {
        book.snapshot_into(self.depth, &mut self.cur);
        let first = self.u == 0;
        let (mut db, mut da) = (Vec::new(), Vec::new());
        if first {
            db.extend_from_slice(&self.cur.bids);
            da.extend_from_slice(&self.cur.asks);
        } else {
            diff(&self.last.bids, &self.cur.bids, |a, b| a > b, &mut db);
            diff(&self.last.asks, &self.cur.asks, |a, b| a < b, &mut da);
            if db.is_empty() && da.is_empty() {
                return Ok(false);
            }
        }
        self.u += 1;
        let g = *book.grid();
        let b = &mut self.buf;
        b.clear();
        let _ = write!(b, "{{\"topic\":\"{}\",\"type\":\"{}\",\"ts\":{ts},\"data\":{{\"s\":\"{}\",\"b\":", self.topic, if first { "snapshot" } else { "delta" }, self.symbol);
        write_levels(b, &g, &db);
        b.push_str(",\"a\":");
        write_levels(b, &g, &da);
        let _ = writeln!(b, ",\"u\":{},\"seq\":{}}},\"cts\":{cts}}}", self.u, book.version());
        out.write_all(b.as_bytes())?;
        std::mem::swap(&mut self.last, &mut self.cur);
        if first {
            self.snapshots += 1;
        } else {
            self.deltas += 1;
        }
        Ok(true)
    }
}

fn write_levels(b: &mut String, g: &Grid, levels: &[(i64, i64)]) {
    b.push('[');
    for (k, &(p, s)) in levels.iter().enumerate() {
        if k > 0 {
            b.push(',');
        }
        b.push_str("[\"");
        g.write_price(p, b);
        b.push_str("\",\"");
        if s == 0 {
            b.push('0');
        } else {
            g.write_qty(s, b);
        }
        b.push_str("\"]");
    }
    b.push(']');
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn diff_emits_inserts_updates_and_removals() {
        let old = [(105, 1), (104, 2), (102, 3)];
        let new = [(106, 5), (104, 1), (103, 7)];
        let mut out = vec![];
        diff(&old, &new, |a, b| a > b, &mut out);
        assert_eq!(out, vec![(106, 5), (105, 0), (104, 1), (103, 7), (102, 0)]);
        out.clear();
        diff(&new, &new, |a, b| a > b, &mut out);
        assert!(out.is_empty());
    }
}
