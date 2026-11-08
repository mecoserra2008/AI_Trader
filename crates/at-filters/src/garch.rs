//! GARCH(1,1) with Student-t innovations: filtering and multi-step forecasts.
//! Parameters come from the Python fit (`ai_trader.tools.vol.garch`, via the `arch` package),
//! stored as JSON in `data/models/vol.garch_t/<symbol>.json`.

use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GarchParams {
    pub mu: f64,
    pub omega: f64,
    pub alpha: f64,
    pub beta: f64,
    pub nu: f64,
}

impl GarchParams {
    pub fn persistence(&self) -> f64 {
        self.alpha + self.beta
    }

    pub fn unconditional_var(&self) -> Option<f64> {
        let p = self.persistence();
        (p < 1.0).then(|| self.omega / (1.0 - p))
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GarchFilter {
    pub params: GarchParams,
    /// Conditional variance for the *next* return, h_{t+1|t}.
    pub h_next: f64,
}

impl GarchFilter {
    pub fn new(params: GarchParams, h0: f64) -> Self {
        Self { params, h_next: h0 }
    }

    /// Feed return r_t; returns h_{t+1|t}.
    pub fn update(&mut self, r: f64) -> f64 {
        let p = &self.params;
        let e = r - p.mu;
        self.h_next = p.omega + p.alpha * e * e + p.beta * self.h_next;
        self.h_next
    }

    /// Variance forecasts h_{t+1..t+k}.
    pub fn forecast(&self, k: usize) -> Vec<f64> {
        let p = &self.params;
        let pers = p.persistence();
        match p.unconditional_var() {
            Some(lr) => (0..k).map(|j| lr + pers.powi(j as i32) * (self.h_next - lr)).collect(),
            None => {
                let mut h = self.h_next;
                (0..k)
                    .map(|_| {
                        let out = h;
                        h = p.omega + pers * h;
                        out
                    })
                    .collect()
            }
        }
    }

    /// Volatility of the cumulative return over the next k bars (sqrt of summed variances).
    pub fn horizon_sigma(&self, k: usize) -> f64 {
        self.forecast(k).iter().sum::<f64>().sqrt()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn forecast_mean_reverts_to_unconditional() {
        let p = GarchParams { mu: 0.0, omega: 1e-6, alpha: 0.08, beta: 0.9, nu: 5.0 };
        let lr = p.unconditional_var().unwrap();
        let f = GarchFilter::new(p, 4.0 * lr).forecast(2000);
        assert!((f[0] - 4.0 * lr).abs() < 1e-18);
        assert!((f[1999] / lr - 1.0).abs() < 1e-6);
    }
}
