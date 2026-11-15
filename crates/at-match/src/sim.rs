//! JSONL driver behind `atx match-sim`: one input event per line, outputs as JSON lines.
//!
//! Input line (unknown fields ignored; `price`, `qty`, `slippage` may be numbers or strings; `id` and `owner` may
//! be non-negative integers or strings):
//! `{"ts_ms": 1731456000000, "kind": "new|cancel|amend", "id": 17, "side": "Buy|Sell", "type": "Limit|Market",
//!   "tif": "GTC|IOC|FOK|PostOnly", "price": 75812.3, "qty": 0.01, "reduce_only": false, "owner": "mm1",
//!   "slippage_type": "Percent|TickSize", "slippage": 0.05}`
//! - new: `side`, `qty` required; `type` defaults to Limit, `tif` to GTC (market orders always run as IOC);
//!   `price` required for limits; `slippage_type` / `slippage` override the market protection band per order.
//! - cancel: `id`. amend: `id` plus `price` and / or `qty` (qty = total order qty after the amendment).
//! - `ts_ms` must be non-decreasing; a line that goes back in time is rejected as BadInput.
//! - Ids must be unique among live orders; an id may be reused once its order is closed.
//!
//! Outputs (`<out>/`): `trades.jsonl`, `book.jsonl` (orderbook.<depth> shape, see [`crate::l2`]), `rejects.jsonl`,
//! optional `orders.jsonl` (every order update), and `stats.json`.

use std::borrow::Cow;
use std::collections::{BTreeMap, HashMap};
use std::fmt;
use std::fmt::Write as _;
use std::fs::File;
use std::io::{self, BufRead, BufReader, BufWriter, Write};
use std::path::Path;
use std::time::Instant;

use anyhow::Context;
use serde::de::{self, Deserializer, Visitor};
use serde::{Deserialize, Serialize};

use crate::book::{EngineConfig, Limits, OrderBook};
use crate::grid::Grid;
use crate::hash::IntMap;
use crate::l2::Publisher;
use crate::types::*;

/// Internal ids at or above this value belong to string ids; numeric ids must be below it.
pub const STR_ID_BASE: u64 = 1 << 62;

// ------------------------------------------------------------------ input tokens

enum IdTok<'a> {
    Num(u64),
    Str(Cow<'a, str>),
}

impl<'de: 'a, 'a> Deserialize<'de> for IdTok<'a> {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        struct V<'a>(std::marker::PhantomData<&'a ()>);
        impl<'de: 'a, 'a> Visitor<'de> for V<'a> {
            type Value = IdTok<'a>;
            fn expecting(&self, f: &mut fmt::Formatter) -> fmt::Result {
                f.write_str("a non-negative integer or a string")
            }
            fn visit_u64<E: de::Error>(self, v: u64) -> Result<Self::Value, E> {
                Ok(IdTok::Num(v))
            }
            fn visit_i64<E: de::Error>(self, v: i64) -> Result<Self::Value, E> {
                u64::try_from(v).map(IdTok::Num).map_err(|_| E::custom("negative id"))
            }
            fn visit_f64<E: de::Error>(self, v: f64) -> Result<Self::Value, E> {
                if v >= 0.0 && v.fract() == 0.0 && v < 9.0e15 { Ok(IdTok::Num(v as u64)) } else { Err(E::custom("id must be an integer")) }
            }
            fn visit_borrowed_str<E: de::Error>(self, v: &'de str) -> Result<Self::Value, E> {
                Ok(IdTok::Str(Cow::Borrowed(v)))
            }
            fn visit_str<E: de::Error>(self, v: &str) -> Result<Self::Value, E> {
                Ok(IdTok::Str(Cow::Owned(v.to_string())))
            }
            fn visit_string<E: de::Error>(self, v: String) -> Result<Self::Value, E> {
                Ok(IdTok::Str(Cow::Owned(v)))
            }
        }
        d.deserialize_any(V(std::marker::PhantomData))
    }
}

struct NumTok(f64);

impl<'de> Deserialize<'de> for NumTok {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        struct V;
        impl Visitor<'_> for V {
            type Value = NumTok;
            fn expecting(&self, f: &mut fmt::Formatter) -> fmt::Result {
                f.write_str("a number or a numeric string")
            }
            fn visit_f64<E: de::Error>(self, v: f64) -> Result<NumTok, E> {
                Ok(NumTok(v))
            }
            fn visit_u64<E: de::Error>(self, v: u64) -> Result<NumTok, E> {
                Ok(NumTok(v as f64))
            }
            fn visit_i64<E: de::Error>(self, v: i64) -> Result<NumTok, E> {
                Ok(NumTok(v as f64))
            }
            fn visit_str<E: de::Error>(self, v: &str) -> Result<NumTok, E> {
                v.trim().parse::<f64>().map(NumTok).map_err(|_| E::custom(format!("not a number: {v:?}")))
            }
        }
        d.deserialize_any(V)
    }
}

#[derive(Deserialize)]
struct Raw<'a> {
    ts_ms: i64,
    #[serde(borrow)]
    kind: Cow<'a, str>,
    #[serde(borrow, default)]
    id: Option<IdTok<'a>>,
    #[serde(borrow, default)]
    side: Option<Cow<'a, str>>,
    #[serde(rename = "type", borrow, default)]
    typ: Option<Cow<'a, str>>,
    #[serde(borrow, default)]
    tif: Option<Cow<'a, str>>,
    #[serde(default)]
    price: Option<NumTok>,
    #[serde(default)]
    qty: Option<NumTok>,
    #[serde(default)]
    reduce_only: Option<bool>,
    #[serde(borrow, default)]
    owner: Option<IdTok<'a>>,
    #[serde(borrow, default)]
    slippage_type: Option<Cow<'a, str>>,
    #[serde(default)]
    slippage: Option<NumTok>,
}

fn json_str(s: &str) -> String {
    serde_json::to_string(s).unwrap_or_else(|_| "\"?\"".into())
}

// ------------------------------------------------------------------ typed input (for generators and tests)

/// A typed input event with numeric ids and owners; `write_json` gives the JSONL line the driver reads.
#[derive(Clone, Debug, PartialEq)]
pub enum InputEvent {
    New { ts: i64, id: u64, owner: u64, side: Side, ord_type: OrdType, tif: Tif, price: Option<f64>, qty: f64, reduce_only: bool },
    Cancel { ts: i64, id: u64 },
    Amend { ts: i64, id: u64, price: Option<f64>, qty: Option<f64> },
}

impl InputEvent {
    pub fn ts(&self) -> i64 {
        match self {
            InputEvent::New { ts, .. } | InputEvent::Cancel { ts, .. } | InputEvent::Amend { ts, .. } => *ts,
        }
    }

    pub fn write_json(&self, out: &mut String) {
        match self {
            InputEvent::New { ts, id, owner, side, ord_type, tif, price, qty, reduce_only } => {
                let _ = write!(out, "{{\"ts_ms\":{ts},\"kind\":\"new\",\"id\":{id},\"owner\":{owner},\"side\":\"{}\",\"type\":\"{}\",\"tif\":\"{}\"", side.as_str(), if *ord_type == OrdType::Market { "Market" } else { "Limit" }, tif.as_str());
                if let Some(p) = price {
                    let _ = write!(out, ",\"price\":{p}");
                }
                let _ = write!(out, ",\"qty\":{qty},\"reduce_only\":{reduce_only}}}");
            }
            InputEvent::Cancel { ts, id } => {
                let _ = write!(out, "{{\"ts_ms\":{ts},\"kind\":\"cancel\",\"id\":{id}}}");
            }
            InputEvent::Amend { ts, id, price, qty } => {
                let _ = write!(out, "{{\"ts_ms\":{ts},\"kind\":\"amend\",\"id\":{id}");
                if let Some(p) = price {
                    let _ = write!(out, ",\"price\":{p}");
                }
                if let Some(q) = qty {
                    let _ = write!(out, ",\"qty\":{q}");
                }
                out.push('}');
            }
        }
    }

    pub fn to_json(&self) -> String {
        let mut s = String::new();
        self.write_json(&mut s);
        s
    }
}

// ------------------------------------------------------------------ driver

#[derive(Clone, Debug)]
pub struct SimConfig {
    pub symbol: String,
    /// Publish book messages at the end of each bin of this many ms (100 = Bybit's ob200 cadence); 0 = after every
    /// input event.
    pub snapshot_every_ms: i64,
    pub depth: usize,
}

/// Output sinks; `orders` is optional.
pub struct Sinks<'a> {
    pub trades: &'a mut dyn Write,
    pub book: &'a mut dyn Write,
    pub rejects: &'a mut dyn Write,
    pub orders: Option<&'a mut dyn Write>,
}

#[derive(Clone, Debug, Default, Serialize)]
pub struct EventCounts {
    pub new: u64,
    pub cancel: u64,
    pub amend: u64,
}

#[derive(Clone, Debug, Serialize)]
pub struct OwnerOut {
    pub owner: serde_json::Value,
    pub position: f64,
    pub cash: f64,
    pub fees: f64,
    pub maker_qty: f64,
    pub taker_qty: f64,
    pub trades: u64,
}

#[derive(Clone, Debug, Serialize)]
pub struct FinalBook {
    pub best_bid: Option<f64>,
    pub best_ask: Option<f64>,
    pub bid_levels: usize,
    pub ask_levels: usize,
    pub resting_orders: usize,
    pub version: u64,
}

#[derive(Clone, Debug, Serialize)]
pub struct Perf {
    pub elapsed_s: f64,
    pub lines_per_s: f64,
}

#[derive(Clone, Debug, Serialize)]
pub struct SimStats {
    pub symbol: String,
    pub lines: u64,
    pub events: EventCounts,
    pub bad_input: u64,
    pub trades: u64,
    pub traded_qty: f64,
    pub traded_notional: f64,
    pub taker_buy_qty: f64,
    pub taker_sell_qty: f64,
    pub fees_total: f64,
    /// Reject reason -> count (rejects.jsonl).
    pub rejects: BTreeMap<String, u64>,
    /// "status" or "status:reason" of every order update that closed an order -> count.
    pub closes: BTreeMap<String, u64>,
    pub book_snapshots: u64,
    pub book_deltas: u64,
    pub final_book: Option<FinalBook>,
    pub owners: Vec<OwnerOut>,
    pub tick_size: f64,
    pub qty_step: f64,
    pub limits: Limits,
    pub maker_bps: f64,
    pub taker_bps: f64,
    pub engine: EngineConfig,
    pub snapshot_every_ms: i64,
    pub depth: usize,
    /// Wall-clock timing: the only non-deterministic field.
    pub perf: Option<Perf>,
}

pub struct Sim {
    book: OrderBook,
    publ: Publisher,
    cfg: SimConfig,
    str_ids: HashMap<String, u64>,
    str_rev: IntMap<u64, String>,
    next_str: u64,
    owner_ids: HashMap<String, OwnerId>,
    owner_tok: Vec<String>,
    cur_bin: Option<i64>,
    last_ts: i64,
    events: Vec<Event>,
    buf: String,
    stats: SimStats,
    started: Instant,
}

impl Sim {
    pub fn new(book: OrderBook, cfg: SimConfig) -> Self {
        let g = *book.grid();
        let stats = SimStats {
            symbol: cfg.symbol.clone(),
            lines: 0,
            events: EventCounts::default(),
            bad_input: 0,
            trades: 0,
            traded_qty: 0.0,
            traded_notional: 0.0,
            taker_buy_qty: 0.0,
            taker_sell_qty: 0.0,
            fees_total: 0.0,
            rejects: BTreeMap::new(),
            closes: BTreeMap::new(),
            book_snapshots: 0,
            book_deltas: 0,
            final_book: None,
            owners: vec![],
            tick_size: g.tick_size(),
            qty_step: g.qty_step(),
            limits: *book.limits(),
            maker_bps: book.fees().maker_bps,
            taker_bps: book.fees().taker_bps,
            engine: *book.config(),
            snapshot_every_ms: cfg.snapshot_every_ms,
            depth: cfg.depth,
            perf: None,
        };
        Self {
            publ: Publisher::new(&cfg.symbol, cfg.depth),
            book,
            cfg,
            str_ids: HashMap::new(),
            str_rev: IntMap::default(),
            next_str: STR_ID_BASE,
            owner_ids: HashMap::new(),
            owner_tok: vec!["null".into()],
            cur_bin: None,
            last_ts: i64::MIN,
            events: Vec::with_capacity(64),
            buf: String::with_capacity(512),
            stats,
            started: Instant::now(),
        }
    }

    pub fn book(&self) -> &OrderBook {
        &self.book
    }

    pub fn publisher(&self) -> &Publisher {
        &self.publ
    }

    fn owner_id(&mut self, tok: Option<IdTok>) -> OwnerId {
        let key = match tok {
            None => return 0,
            Some(IdTok::Num(n)) => n.to_string(),
            Some(IdTok::Str(s)) => json_str(&s),
        };
        if let Some(&o) = self.owner_ids.get(&key) {
            return o;
        }
        let o = self.owner_tok.len() as OwnerId;
        self.owner_tok.push(key.clone());
        self.owner_ids.insert(key, o);
        o
    }

    fn write_id(&self, id: OrderId, out: &mut String) {
        if id >= STR_ID_BASE {
            match self.str_rev.get(&id) {
                Some(s) => out.push_str(s),
                None => out.push_str("null"),
            }
        } else {
            let _ = write!(out, "{id}");
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn reject_line(&mut self, sinks: &mut Sinks, ts: i64, op: &str, id_tok: &str, owner: &str, reason: Reason, detail: &str) -> io::Result<()> {
        *self.stats.rejects.entry(reason.as_str().to_string()).or_default() += 1;
        let b = &mut self.buf;
        b.clear();
        let _ = write!(b, "{{\"ts\":{ts},\"op\":\"{op}\",\"id\":{id_tok},\"owner\":{owner},\"reason\":\"{}\",\"line\":{}", reason.as_str(), self.stats.lines);
        if !detail.is_empty() {
            let _ = write!(b, ",\"detail\":{}", json_str(detail));
        }
        b.push_str("}\n");
        sinks.rejects.write_all(b.as_bytes())
    }

    fn flush_book(&mut self, sinks: &mut Sinks, ts: i64) -> io::Result<()> {
        let cts = if self.last_ts == i64::MIN { ts } else { self.last_ts };
        self.publ.publish(&self.book, ts, cts, sinks.book)?;
        Ok(())
    }

    /// Process one input line.
    pub fn process_line(&mut self, line: &str, sinks: &mut Sinks) -> io::Result<()> {
        let line = line.trim();
        if line.is_empty() {
            return Ok(());
        }
        self.stats.lines += 1;
        let raw: Raw = match serde_json::from_str(line) {
            Ok(r) => r,
            Err(e) => {
                self.stats.bad_input += 1;
                return self.reject_line(sinks, self.last_ts.max(0), "unknown", "null", "null", Reason::BadInput, &e.to_string());
            }
        };
        let ts = raw.ts_ms;
        let id_tok = match &raw.id {
            Some(IdTok::Num(n)) => n.to_string(),
            Some(IdTok::Str(s)) => json_str(s),
            None => "null".into(),
        };
        let op = match &*raw.kind {
            "new" => Op::New,
            "cancel" => Op::Cancel,
            "amend" => Op::Amend,
            other => {
                self.stats.bad_input += 1;
                return self.reject_line(sinks, ts, "unknown", &id_tok, "null", Reason::BadInput, &format!("unknown kind {other:?}"));
            }
        };
        if ts < self.last_ts {
            self.stats.bad_input += 1;
            return self.reject_line(sinks, ts, op.as_str(), &id_tok, "null", Reason::BadInput, "ts_ms goes backwards");
        }
        // Book messages for completed bins are published before the event that opens a new bin.
        let every = self.cfg.snapshot_every_ms;
        if every > 0 {
            let bin = ts.div_euclid(every);
            match self.cur_bin {
                Some(b) if bin > b => {
                    self.flush_book(sinks, (b + 1) * every)?;
                    self.cur_bin = Some(bin);
                }
                None => self.cur_bin = Some(bin),
                _ => {}
            }
        }
        self.last_ts = ts;

        let Some(id_raw) = raw.id else {
            self.stats.bad_input += 1;
            return self.reject_line(sinks, ts, op.as_str(), "null", "null", Reason::BadInput, "missing id");
        };
        let mut fresh_str = None;
        let id = match id_raw {
            IdTok::Num(n) if n < STR_ID_BASE => n,
            IdTok::Num(_) => {
                self.stats.bad_input += 1;
                return self.reject_line(sinks, ts, op.as_str(), &id_tok, "null", Reason::BadInput, "numeric id >= 2^62");
            }
            IdTok::Str(s) => match self.str_ids.get(&*s) {
                Some(&i) => i,
                None if op == Op::New => {
                    let i = self.next_str;
                    self.next_str += 1;
                    self.str_ids.insert(s.to_string(), i);
                    self.str_rev.insert(i, id_tok.clone());
                    fresh_str = Some(s.to_string());
                    i
                }
                None => return self.reject_line(sinks, ts, op.as_str(), &id_tok, "null", Reason::OrderNotExist, ""),
            },
        };
        self.events.clear();
        match op {
            Op::New => {
                self.stats.events.new += 1;
                let owner = self.owner_id(raw.owner);
                let side = raw.side.as_deref().and_then(Side::parse);
                let ord_type = match raw.typ.as_deref() {
                    None => Some(OrdType::Limit),
                    Some(t) => OrdType::parse(t),
                };
                let tif = match raw.tif.as_deref() {
                    None => Some(if ord_type == Some(OrdType::Market) { Tif::IOC } else { Tif::GTC }),
                    Some(t) => Tif::parse(t),
                };
                let band = match (raw.slippage_type.as_deref(), raw.slippage.as_ref()) {
                    (None, None) => Ok(None),
                    (Some("Percent"), Some(v)) if v.0 > 0.0 && v.0.is_finite() => Ok(Some(Band::Percent(v.0))),
                    (Some("TickSize"), Some(v)) if v.0 >= 1.0 && v.0.fract() == 0.0 => Ok(Some(Band::Ticks(v.0 as i64))),
                    _ => Err("slippage_type must be Percent or TickSize with a valid slippage"),
                };
                let (Some(side), Some(ord_type), Some(tif), Some(qty), Ok(band)) = (side, ord_type, tif, raw.qty.as_ref(), band) else {
                    self.stats.bad_input += 1;
                    if let Some(s) = fresh_str {
                        self.str_ids.remove(&s);
                        self.str_rev.remove(&id);
                    }
                    let detail = "new needs a valid side, type, tif, qty (and slippage pair)";
                    let ow = self.owner_tok[owner as usize].clone();
                    return self.reject_line(sinks, ts, "new", &id_tok, &ow, Reason::BadInput, detail);
                };
                let o = NewOrder { ts, id, owner, side, ord_type, tif, price: raw.price.as_ref().map(|p| p.0), qty: qty.0, reduce_only: raw.reduce_only.unwrap_or(false), band };
                self.book.submit_into(&o, &mut self.events);
            }
            Op::Cancel => {
                self.stats.events.cancel += 1;
                self.book.cancel_into(ts, id, &mut self.events);
            }
            Op::Amend => {
                self.stats.events.amend += 1;
                if raw.price.is_none() && raw.qty.is_none() {
                    self.stats.bad_input += 1;
                    return self.reject_line(sinks, ts, "amend", &id_tok, "null", Reason::BadInput, "amend needs price and/or qty");
                }
                self.book.amend_into(ts, id, raw.price.map(|p| p.0), raw.qty.map(|q| q.0), &mut self.events);
            }
        }
        self.emit(sinks, &id_tok, fresh_str)?;
        if every == 0 {
            self.flush_book(sinks, ts)?;
        }
        Ok(())
    }

    fn emit(&mut self, sinks: &mut Sinks, id_tok: &str, fresh_str: Option<String>) -> io::Result<()> {
        let events = std::mem::take(&mut self.events);
        let g: Grid = *self.book.grid();
        let mut closed: Vec<OrderId> = vec![];
        for ev in &events {
            match ev {
                Event::Trade(t) => {
                    let (q, p) = (g.qty(t.qty), g.price(t.price));
                    self.stats.trades += 1;
                    self.stats.traded_qty += q;
                    self.stats.traded_notional += p * q;
                    self.stats.fees_total += t.taker_fee + t.maker_fee;
                    match t.taker_side {
                        Side::Buy => self.stats.taker_buy_qty += q,
                        Side::Sell => self.stats.taker_sell_qty += q,
                    }
                    let mut b = std::mem::take(&mut self.buf);
                    b.clear();
                    let _ = write!(b, "{{\"ts\":{},\"id\":{},\"price\":", t.ts, t.seq);
                    g.write_price(t.price, &mut b);
                    b.push_str(",\"size\":");
                    g.write_qty(t.qty, &mut b);
                    let _ = write!(b, ",\"side\":\"{}\",\"taker\":", t.taker_side.as_str());
                    self.write_id(t.taker_id, &mut b);
                    b.push_str(",\"maker\":");
                    self.write_id(t.maker_id, &mut b);
                    let _ = writeln!(
                        b,
                        ",\"taker_owner\":{},\"maker_owner\":{},\"taker_fee\":{},\"maker_fee\":{}}}",
                        self.owner_tok[t.taker_owner as usize], self.owner_tok[t.maker_owner as usize], t.taker_fee, t.maker_fee
                    );
                    sinks.trades.write_all(b.as_bytes())?;
                    self.buf = b;
                }
                Event::Order(u) => {
                    if u.status.is_closed() {
                        let key = match u.reason {
                            Some(r) => format!("{}:{}", u.status.as_str(), r.as_str()),
                            None => u.status.as_str().to_string(),
                        };
                        *self.stats.closes.entry(key).or_default() += 1;
                        if u.id >= STR_ID_BASE {
                            closed.push(u.id);
                        }
                    }
                    if let Some(w) = sinks.orders.as_mut() {
                        let mut b = std::mem::take(&mut self.buf);
                        b.clear();
                        let _ = write!(b, "{{\"ts\":{},\"id\":", u.ts);
                        self.write_id(u.id, &mut b);
                        let _ = write!(b, ",\"owner\":{},\"side\":\"{}\",\"price\":", self.owner_tok[u.owner as usize], u.side.as_str());
                        g.write_price(u.price, &mut b);
                        b.push_str(",\"qty\":");
                        g.write_qty(u.qty, &mut b);
                        b.push_str(",\"leaves\":");
                        g.write_qty(u.leaves, &mut b);
                        b.push_str(",\"cum\":");
                        g.write_qty(u.cum, &mut b);
                        let _ = write!(b, ",\"status\":\"{}\",\"reason\":", u.status.as_str());
                        match u.reason {
                            Some(r) => {
                                let _ = write!(b, "\"{}\"", r.as_str());
                            }
                            None => b.push_str("null"),
                        }
                        b.push_str("}\n");
                        w.write_all(b.as_bytes())?;
                        self.buf = b;
                    }
                }
                Event::Reject(r) => {
                    let ow = self.owner_tok[r.owner as usize].clone();
                    self.reject_line(sinks, r.ts, r.op.as_str(), id_tok, &ow, r.reason, "")?;
                    if r.op == Op::New
                        && let Some(s) = &fresh_str
                    {
                        self.str_ids.remove(s);
                        self.str_rev.remove(&r.id);
                    }
                }
            }
        }
        for id in closed {
            if let Some(tok) = self.str_rev.remove(&id) {
                let s: String = serde_json::from_str(&tok).unwrap_or_default();
                self.str_ids.remove(&s);
            }
        }
        self.events = events;
        Ok(())
    }

    /// Publish the last bin and return the run statistics.
    pub fn finish(mut self, sinks: &mut Sinks) -> io::Result<SimStats> {
        if let (Some(b), every) = (self.cur_bin, self.cfg.snapshot_every_ms)
            && every > 0
        {
            self.flush_book(sinks, (b + 1) * every)?;
        }
        let g = *self.book.grid();
        let mut st = self.stats;
        st.book_snapshots = self.publ.snapshots;
        st.book_deltas = self.publ.deltas;
        st.final_book = Some(FinalBook {
            best_bid: self.book.best_bid().map(|b| g.price(b.0)),
            best_ask: self.book.best_ask().map(|a| g.price(a.0)),
            bid_levels: self.book.n_levels().0,
            ask_levels: self.book.n_levels().1,
            resting_orders: self.book.n_orders(),
            version: self.book.version(),
        });
        let mut owners: Vec<OwnerOut> = self
            .book
            .owners()
            .iter()
            .enumerate()
            .filter(|(_, o)| o.trades > 0 || o.position != 0)
            .map(|(i, o)| OwnerOut {
                owner: serde_json::from_str(self.owner_tok.get(i).map_or("null", |s| s.as_str())).unwrap_or(serde_json::Value::Null),
                position: g.qty(o.position),
                cash: o.cash,
                fees: o.fees,
                maker_qty: g.qty(o.maker_lots),
                taker_qty: g.qty(o.taker_lots),
                trades: o.trades,
            })
            .collect();
        owners.sort_by(|a, b| a.owner.to_string().cmp(&b.owner.to_string()));
        st.owners = owners;
        let el = self.started.elapsed().as_secs_f64();
        st.perf = Some(Perf { elapsed_s: el, lines_per_s: if el > 0.0 { st.lines as f64 / el } else { 0.0 } });
        Ok(st)
    }
}

/// Run a whole file: `<out>/trades.jsonl`, `book.jsonl`, `rejects.jsonl`, (`orders.jsonl`), `stats.json`.
pub fn run_files(book: OrderBook, cfg: SimConfig, input: &Path, out_dir: &Path, write_orders: bool) -> anyhow::Result<SimStats> {
    std::fs::create_dir_all(out_dir).with_context(|| out_dir.display().to_string())?;
    let open = |name: &str| -> anyhow::Result<BufWriter<File>> {
        let p = out_dir.join(name);
        Ok(BufWriter::with_capacity(1 << 16, File::create(&p).with_context(|| p.display().to_string())?))
    };
    let (mut tw, mut bw, mut rw) = (open("trades.jsonl")?, open("book.jsonl")?, open("rejects.jsonl")?);
    let mut ow = if write_orders { Some(open("orders.jsonl")?) } else { None };
    let reader = BufReader::with_capacity(1 << 16, File::open(input).with_context(|| input.display().to_string())?);
    let mut sim = Sim::new(book, cfg);
    let mut sinks = Sinks { trades: &mut tw, book: &mut bw, rejects: &mut rw, orders: ow.as_mut().map(|w| w as &mut dyn Write) };
    let mut line = String::new();
    let mut reader = reader;
    loop {
        line.clear();
        if reader.read_line(&mut line).with_context(|| input.display().to_string())? == 0 {
            break;
        }
        sim.process_line(&line, &mut sinks)?;
    }
    let stats = sim.finish(&mut sinks)?;
    tw.flush()?;
    bw.flush()?;
    rw.flush()?;
    if let Some(w) = ow.as_mut() {
        w.flush()?;
    }
    std::fs::write(out_dir.join("stats.json"), serde_json::to_string_pretty(&stats)?)?;
    Ok(stats)
}
