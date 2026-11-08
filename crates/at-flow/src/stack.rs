//! O(1) moving-average filters for deriving feature stacks from the 3 s base candles
//! (e.g. EMA of delta over 1 min / 5 min / 15 min / 1 h / 4 h, or the EMA of an SMA).
//! These are never stored: they're recomputed from `flow_3s` (live) or `flow_1m` (history).

use std::collections::VecDeque;

/// EMA with a half-life in samples: α = 1 − 2^(−1/h).
#[derive(Clone, Debug)]
pub struct Ema {
    alpha: f64,
    value: Option<f64>,
}

impl Ema {
    pub fn with_half_life(h: f64) -> Self {
        Self { alpha: 1.0 - 0.5f64.powf(1.0 / h.max(1e-9)), value: None }
    }

    pub fn update(&mut self, x: f64) -> f64 {
        let v = match self.value {
            None => x,
            Some(v) => v + self.alpha * (x - v),
        };
        self.value = Some(v);
        v
    }

    pub fn value(&self) -> Option<f64> {
        self.value
    }
}

/// Simple moving average over the last `n` samples (None until the window is full).
#[derive(Clone, Debug)]
pub struct Sma {
    n: usize,
    buf: VecDeque<f64>,
    sum: f64,
}

impl Sma {
    pub fn new(n: usize) -> Self {
        Self { n: n.max(1), buf: VecDeque::with_capacity(n.max(1)), sum: 0.0 }
    }

    pub fn update(&mut self, x: f64) -> Option<f64> {
        self.buf.push_back(x);
        self.sum += x;
        if self.buf.len() > self.n {
            self.sum -= self.buf.pop_front().unwrap();
        }
        (self.buf.len() == self.n).then(|| self.sum / self.n as f64)
    }
}

/// EMA-of-SMA stack: for each window w, SMA_w then EMA with half-life w (smooth, lag-controlled).
#[derive(Clone, Debug)]
pub struct Stack {
    pub windows: Vec<usize>,
    smas: Vec<Sma>,
    emas: Vec<Ema>,
}

impl Stack {
    pub fn new(windows: &[usize]) -> Self {
        Self {
            windows: windows.to_vec(),
            smas: windows.iter().map(|&w| Sma::new(w)).collect(),
            emas: windows.iter().map(|&w| Ema::with_half_life(w as f64)).collect(),
        }
    }

    /// Returns one value per window (None during warm-up).
    pub fn update(&mut self, x: f64) -> Vec<Option<f64>> {
        self.smas.iter_mut().zip(self.emas.iter_mut()).map(|(s, e)| s.update(x).map(|m| e.update(m))).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ema_half_life_and_sma_window() {
        let mut e = Ema::with_half_life(10.0);
        e.update(0.0);
        let mut v = 0.0;
        for _ in 0..10 {
            v = e.update(1.0);
        }
        assert!((v - 0.5).abs() < 1e-9, "after one half-life the gap is halved: {v}");
        let mut s = Sma::new(3);
        assert_eq!(s.update(1.0), None);
        s.update(2.0);
        assert_eq!(s.update(3.0), Some(2.0));
        assert_eq!(s.update(4.0), Some(3.0));
        let mut st = Stack::new(&[2, 4]);
        let out: Vec<_> = (0..5).map(|i| st.update(i as f64)).collect();
        assert!(out[0][0].is_none() && out[1][0].is_some() && out[2][1].is_none() && out[3][1].is_some());
    }
}
