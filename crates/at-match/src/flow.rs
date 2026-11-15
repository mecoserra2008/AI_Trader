//! Deterministic synthetic order flow for tests and the benchmark (not a market model: F5's generator is).

use crate::sim::InputEvent;
use crate::types::{OrdType, Side, Tif};

/// xorshift64* (deterministic, no dependencies).
#[derive(Clone, Debug)]
pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Self {
        Rng(seed.wrapping_mul(0x9e37_79b9_7f4a_7c15) | 1)
    }

    #[inline]
    pub fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_f491_4f6c_dd1d)
    }

    /// Uniform in [0, n).
    #[inline]
    pub fn below(&mut self, n: u64) -> u64 {
        ((self.next_u64() >> 32) * n) >> 32
    }

    #[inline]
    pub fn unit(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }
}

#[derive(Clone, Debug)]
pub struct FlowParams {
    pub tick: f64,
    pub step: f64,
    /// Starting mid in ticks.
    pub mid: i64,
    /// Passive orders rest 1..=depth ticks from the mid.
    pub depth: u64,
    pub max_lots: u64,
    pub owners: u64,
    /// Include reduce-only orders.
    pub reduce_only: bool,
}

impl Default for FlowParams {
    fn default() -> Self {
        Self { tick: 0.1, step: 0.001, mid: 1_000_000, depth: 200, max_lots: 20, owners: 8, reduce_only: true }
    }
}

/// `n` events: ~40% passive GTC, 5% PostOnly near the touch, 35% cancels, 8% amends, 7% aggressive limits
/// (GTC / IOC / FOK), 4% market orders, 1% reduce-only; the mid random-walks by one tick.
pub fn random_flow(seed: u64, n: usize, p: &FlowParams) -> Vec<InputEvent> {
    let mut r = Rng::new(seed);
    let mut out = Vec::with_capacity(n);
    let mut live: Vec<u64> = Vec::new();
    let (mut mid, mut ts, mut next_id) = (p.mid, 1_700_000_000_000i64, 1u64);
    let px = |t: i64| (t as f64 * p.tick * 1e6).round() / 1e6;
    let qty = |l: u64| (l as f64 * p.step * 1e9).round() / 1e9;
    while out.len() < n {
        ts += r.below(3) as i64;
        if r.below(20) == 0 {
            mid += if r.below(2) == 0 { 1 } else { -1 };
        }
        let side = if r.below(2) == 0 { Side::Buy } else { Side::Sell };
        let owner = 1 + r.below(p.owners.max(1));
        let lots = 1 + r.below(p.max_lots.max(1));
        let u = r.below(100);
        let new = |id: u64, ord_type, tif, price: Option<i64>, lots: u64, reduce_only| InputEvent::New { ts, id, owner, side, ord_type, tif, price: price.map(px), qty: qty(lots), reduce_only };
        let away = |off: i64| if side == Side::Buy { mid - off } else { mid + off };
        let ev = if u < 40 {
            let off = 1 + r.below(p.depth) as i64;
            live.push(next_id);
            new(next_id, OrdType::Limit, Tif::GTC, Some(away(off)), lots, false)
        } else if u < 45 {
            live.push(next_id);
            new(next_id, OrdType::Limit, Tif::PostOnly, Some(away(r.below(3) as i64)), lots, false)
        } else if u < 80 {
            if live.is_empty() {
                continue;
            }
            let k = r.below(live.len() as u64) as usize;
            InputEvent::Cancel { ts, id: live.swap_remove(k) }
        } else if u < 88 {
            if live.is_empty() {
                continue;
            }
            let id = live[r.below(live.len() as u64) as usize];
            match r.below(3) {
                0 => InputEvent::Amend { ts, id, price: None, qty: Some(qty(1 + r.below(p.max_lots))) },
                1 => InputEvent::Amend { ts, id, price: Some(px(away(1 + r.below(p.depth) as i64))), qty: None },
                _ => InputEvent::Amend { ts, id, price: Some(px(away(1 + r.below(p.depth) as i64))), qty: Some(qty(1 + r.below(p.max_lots))) },
            }
        } else if u < 95 {
            let tif = [Tif::GTC, Tif::IOC, Tif::FOK][r.below(3) as usize];
            if tif == Tif::GTC {
                live.push(next_id);
            }
            new(next_id, OrdType::Limit, tif, Some(away(-(r.below(6) as i64))), lots, false)
        } else if u < 99 || !p.reduce_only {
            new(next_id, OrdType::Market, Tif::IOC, None, lots, false)
        } else {
            let market = r.below(2) == 0;
            if !market {
                live.push(next_id);
            }
            if market {
                new(next_id, OrdType::Market, Tif::IOC, None, lots, true)
            } else {
                new(next_id, OrdType::Limit, Tif::GTC, Some(away(1 + r.below(20) as i64)), lots, true)
            }
        };
        if matches!(ev, InputEvent::New { .. }) {
            next_id += 1;
        }
        if live.len() > 50_000 {
            let k = r.below(live.len() as u64) as usize;
            live.swap_remove(k);
        }
        out.push(ev);
    }
    out
}
