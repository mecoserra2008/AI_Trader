//! Seasonal, jump-robust, bias-corrected EWMA: the live counterpart of
//! `ai_trader.features.vol_models` (validated: H-V1b supported in-sample and holdout).
//!
//! Per bar: deseasonalise by the weekend multiplier, winsorise at ±clip·σ̂, update the base EWMA,
//! and track slow EWMAs of raw and clipped squares whose ratio restores the level that clipping
//! removes. The weekend flag comes from the caller (calendar logic lives with the session clock).

use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RobustCfg {
    pub lambda: f64,
    pub clip: f64,
    pub lambda_bias: f64,
    pub m_prior: f64,
    pub kappa: f64,
    pub window: usize,
}

impl Default for RobustCfg {
    fn default() -> Self {
        Self { lambda: 0.94, clip: 4.0, lambda_bias: 0.999, m_prior: 0.6, kappa: 8.0, window: 26 }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RobustEwma {
    pub cfg: RobustCfg,
    v: f64,
    raw_e: f64,
    clip_e: f64,
    m: f64,
    ratios: Vec<f64>,
    in_weekend: bool,
    v_fri: f64,
    acc: f64,
    nb: usize,
}

impl RobustEwma {
    pub fn new(cfg: RobustCfg, seed_var: f64) -> Self {
        let m = cfg.m_prior;
        Self { cfg, v: seed_var, raw_e: seed_var, clip_e: seed_var, m, ratios: Vec::new(), in_weekend: false, v_fri: 0.0, acc: 0.0, nb: 0 }
    }

    fn bias(&self) -> f64 {
        if self.clip_e > 0.0 { self.raw_e / self.clip_e } else { 1.0 }
    }

    /// Weekend vol multiplier known now.
    pub fn m(&self) -> f64 {
        self.m
    }

    /// Bias-corrected, weekday-equivalent per-bar variance for the next bar.
    pub fn base_var(&self) -> f64 {
        self.v * self.bias()
    }

    /// σ forecast for the next bar given whether it lies in the TradFi weekend window.
    pub fn sigma_next(&self, next_is_weekend: bool) -> f64 {
        self.base_var().sqrt() * if next_is_weekend { self.m } else { 1.0 }
    }

    /// Feed the return of a completed bar.
    pub fn update(&mut self, r: f64, is_weekend: bool) {
        if is_weekend && !self.in_weekend {
            (self.in_weekend, self.v_fri, self.acc, self.nb) = (true, self.v, 0.0, 0);
        } else if !is_weekend && self.in_weekend {
            self.in_weekend = false;
            if self.nb >= 12 && self.v_fri > 0.0 {
                self.ratios.push(self.acc / (self.v_fri * self.nb as f64));
                if self.ratios.len() > self.cfg.window {
                    self.ratios.remove(0);
                }
                let s: f64 = self.ratios.iter().sum();
                self.m = ((s + self.cfg.kappa * self.cfg.m_prior.powi(2)) / (self.ratios.len() as f64 + self.cfg.kappa)).sqrt();
            }
        }
        let s = if is_weekend { self.m } else { 1.0 };
        let x_raw = r / s;
        let lim = self.cfg.clip * self.v.sqrt();
        let x = x_raw.clamp(-lim, lim);
        let c = &self.cfg;
        self.v = c.lambda * self.v + (1.0 - c.lambda) * x * x;
        self.raw_e = c.lambda_bias * self.raw_e + (1.0 - c.lambda_bias) * x_raw * x_raw;
        self.clip_e = c.lambda_bias * self.clip_e + (1.0 - c.lambda_bias) * x * x;
        if self.in_weekend {
            self.acc += r * r;
            self.nb += 1;
        }
    }

    /// Cumulative variance over the next bars, given their weekend flags (known calendar).
    pub fn horizon_var(&self, future_weekend: &[bool]) -> f64 {
        let b = self.base_var();
        future_weekend.iter().map(|&w| if w { b * self.m * self.m } else { b }).sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn jump_is_winsorised_and_level_restored_slowly() {
        let mut plain = crate::ewma::Ewma::new(0.94);
        let mut rob = RobustEwma::new(RobustCfg::default(), 1e-4);
        plain.update(0.01);
        for _ in 0..500 {
            plain.update(0.01);
            rob.update(0.01, false);
        }
        plain.update(0.5);
        rob.update(0.5, false);
        assert!(rob.sigma_next(false) < 0.5 * plain.sigma().unwrap(), "single jump barely moves the robust sigma");
        assert!(rob.base_var() > rob.v, "bias factor > 1 after a clipped jump");
    }

    #[test]
    fn learns_quiet_weekends() {
        let mut m = RobustEwma::new(RobustCfg { kappa: 2.0, ..Default::default() }, 1e-4);
        for week in 0..40 {
            for h in 0..168 {
                let wk = h >= 120;
                let r = if (week * 168 + h) % 2 == 0 { 0.01 } else { -0.01 } * if wk { 0.4 } else { 1.0 };
                m.update(r, wk);
            }
        }
        assert!(m.m() < 0.5, "m = {}", m.m());
        let h = m.horizon_var(&[false, true, true]);
        assert!((h - m.base_var() * (1.0 + 2.0 * m.m() * m.m())).abs() < 1e-15);
    }
}
