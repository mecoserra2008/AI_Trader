//! Replay of Bybit `orderbook.<depth>` messages (snapshot + deltas) as order events, so a real book can seed the
//! engine or be checked against it. L2 data carries no order identities, so a queue model is needed:
//! a level that grows gets a new order at the back of its queue; a level that shrinks loses quantity from the back
//! (most recent order first: cancel it, or amend its qty down, which keeps its priority). Within one message all
//! decreases are applied before all increases, so a message that moves the touch never crosses the book.

use std::collections::BTreeMap;

use serde_json::Value;

use crate::book::L2;
use crate::grid::Grid;
use crate::sim::InputEvent;
use crate::types::{OrdType, Side, Tif};

pub struct Ob200Replayer {
    grid: Grid,
    owner: u64,
    next_id: u64,
    bids: BTreeMap<i64, Vec<(u64, i64)>>,
    asks: BTreeMap<i64, Vec<(u64, i64)>>,
    ready: bool,
    pub ignored_before_snapshot: u64,
}

fn parse_levels(grid: &Grid, v: &Value) -> Result<Vec<(i64, i64)>, String> {
    let mut out = vec![];
    for lv in v.as_array().map(|a| a.as_slice()).unwrap_or(&[]) {
        let num = |x: &Value| -> Result<f64, String> {
            match x {
                Value::String(s) => s.parse::<f64>().map_err(|e| format!("{s:?}: {e}")),
                Value::Number(n) => n.as_f64().ok_or_else(|| "bad number".to_string()),
                _ => Err(format!("bad level entry {x}")),
            }
        };
        let (p, s) = (num(&lv[0])?, num(&lv[1])?);
        let t = grid.ticks(p).ok_or_else(|| format!("price {p} off the tick grid"))?;
        let l = if s == 0.0 { 0 } else { grid.lots(s).ok_or_else(|| format!("size {s} off the qty grid"))? };
        out.push((t, l));
    }
    Ok(out)
}

impl Ob200Replayer {
    /// Orders get ids `first_id, first_id + 1, ...` and owner `owner`.
    pub fn new(grid: Grid, owner: u64, first_id: u64) -> Self {
        Self { grid, owner, next_id: first_id, bids: BTreeMap::new(), asks: BTreeMap::new(), ready: false, ignored_before_snapshot: 0 }
    }

    /// The L2 book implied by the messages so far (all levels).
    pub fn book(&self) -> L2 {
        let tot = |q: &Vec<(u64, i64)>| q.iter().map(|x| x.1).sum::<i64>();
        L2 { bids: self.bids.iter().rev().map(|(p, q)| (*p, tot(q))).collect(), asks: self.asks.iter().map(|(p, q)| (*p, tot(q))).collect() }
    }

    /// Translate one message (`{"type": "snapshot"|"delta", "data": {"b": .., "a": ..}}`) into events at `ts`.
    pub fn apply(&mut self, msg: &Value, ts: i64, out: &mut Vec<InputEvent>) -> Result<(), String> {
        let kind = msg["type"].as_str().unwrap_or_default();
        let data = &msg["data"];
        let (b, a) = (parse_levels(&self.grid, &data["b"])?, parse_levels(&self.grid, &data["a"])?);
        let mut changes: Vec<(Side, i64, i64)> = vec![];
        match kind {
            "snapshot" => {
                self.ready = true;
                for (side, new, cur) in [(Side::Buy, &b, &self.bids), (Side::Sell, &a, &self.asks)] {
                    let target: BTreeMap<i64, i64> = new.iter().copied().collect();
                    changes.extend(cur.keys().filter(|p| !target.contains_key(p)).map(|p| (side, *p, 0)));
                    changes.extend(target.iter().map(|(p, l)| (side, *p, *l)));
                }
            }
            "delta" if self.ready => {
                changes.extend(b.iter().map(|(p, l)| (Side::Buy, *p, *l)));
                changes.extend(a.iter().map(|(p, l)| (Side::Sell, *p, *l)));
            }
            "delta" => {
                self.ignored_before_snapshot += 1;
                return Ok(());
            }
            other => return Err(format!("unknown message type {other:?}")),
        }
        // Decreases first, then increases.
        for pass in [false, true] {
            for &(side, px, target) in &changes {
                let book = match side {
                    Side::Buy => &mut self.bids,
                    Side::Sell => &mut self.asks,
                };
                let q = book.entry(px).or_default();
                let cur: i64 = q.iter().map(|x| x.1).sum();
                if !pass && target < cur {
                    let mut d = cur - target;
                    while d > 0 {
                        let (id, lots) = q.pop().expect("level has qty");
                        if lots <= d {
                            out.push(InputEvent::Cancel { ts, id });
                            d -= lots;
                        } else {
                            out.push(InputEvent::Amend { ts, id, price: None, qty: Some(self.grid.qty(lots - d)) });
                            q.push((id, lots - d));
                            d = 0;
                        }
                    }
                } else if pass && target > cur {
                    let id = self.next_id;
                    self.next_id += 1;
                    let lots = target - cur;
                    out.push(InputEvent::New { ts, id, owner: self.owner, side, ord_type: OrdType::Limit, tif: Tif::GTC, price: Some(self.grid.price(px)), qty: self.grid.qty(lots), reduce_only: false });
                    q.push((id, lots));
                }
                if q.is_empty() {
                    book.remove(&px);
                }
            }
        }
        Ok(())
    }
}
