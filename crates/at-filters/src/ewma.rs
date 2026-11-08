//! RiskMetrics-style EWMA variance: σ²_{t+1} = λ σ²_t + (1 − λ) r²_t.

use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Ewma {
    pub lambda: f64,
    pub var: Option<f64>,
    pub n: u64,
}

impl Ewma {
    pub fn new(lambda: f64) -> Self {
        assert!((0.0..1.0).contains(&lambda), "lambda must be in [0, 1)");
        Self { lambda, var: None, n: 0 }
    }

    /// Feed return r_t; returns σ_{t+1|t}, the one-step-ahead volatility.
    pub fn update(&mut self, r: f64) -> f64 {
        let v = match self.var {
            None => r * r,
            Some(v) => self.lambda * v + (1.0 - self.lambda) * r * r,
        };
        self.var = Some(v);
        self.n += 1;
        v.sqrt()
    }

    pub fn sigma(&self) -> Option<f64> {
        self.var.map(f64::sqrt)
    }
}

/// One-step-ahead σ̂ aligned so that `out[t]` uses returns up to `t − 1` only (look-ahead safe).
/// The first `warmup` entries are NaN. The seed is the sample variance of the warm-up window.
pub fn ewma_sigma_ahead(returns: &[f64], lambda: f64, warmup: usize) -> Vec<f64> {
    let mut out = vec![f64::NAN; returns.len()];
    if returns.len() <= warmup || warmup == 0 {
        return out;
    }
    let seed = returns[..warmup].iter().map(|r| r * r).sum::<f64>() / warmup as f64;
    let mut e = Ewma { lambda, var: Some(seed), n: warmup as u64 };
    out[warmup] = seed.sqrt();
    for t in warmup..returns.len() - 1 {
        out[t + 1] = e.update(returns[t]);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn alignment_has_no_lookahead() {
        let mut r = vec![0.01; 50];
        r[40] = 0.5; // shock at t = 40
        let s = ewma_sigma_ahead(&r, 0.94, 20);
        assert!((s[40] - 0.01).abs() < 1e-9, "σ̂ at t must not see r_t");
        assert!(s[41] > 0.1, "σ̂ at t+1 reacts to the shock");
    }
}
