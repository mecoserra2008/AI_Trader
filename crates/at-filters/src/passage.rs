//! Two-barrier first passage for Brownian motion with drift (ARCHITECTURE §3.11, Layer 1).
//! No time variable: absorption probabilities don't depend on the clock.

/// P(hit +a before −b) for X_t = μt + σW_t starting at 0, with a, b > 0 in log-price units,
/// μ and σ² per bar. Uses expm1 forms that stay finite for any drift sign:
/// θ = 2μ/σ²;  θ ≥ 0: P = expm1(−θb) / expm1(−θ(a+b));  θ < 0: P = 1 − expm1(θa) / expm1(θ(a+b)).
pub fn p_up_first(a: f64, b: f64, mu: f64, sigma2: f64) -> f64 {
    assert!(a > 0.0 && b > 0.0 && sigma2 > 0.0, "need a, b, sigma2 > 0");
    let theta = 2.0 * mu / sigma2;
    if (theta * (a + b)).abs() < 1e-10 {
        return b / (a + b);
    }
    let p = if theta >= 0.0 {
        (-theta * b).exp_m1() / (-theta * (a + b)).exp_m1()
    } else {
        1.0 - (theta * a).exp_m1() / (theta * (a + b)).exp_m1()
    };
    p.clamp(0.0, 1.0)
}

/// Expected R of a trade: p·(reward/risk) − (1 − p). Costs are handled by the risk engine.
pub fn expected_r(p_target: f64, reward: f64, risk: f64) -> f64 {
    p_target * reward / risk - (1.0 - p_target)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn driftless_is_gamblers_ruin_and_zero_ev() {
        let p = p_up_first(0.02, 0.01, 0.0, 1e-4);
        assert!((p - 1.0 / 3.0).abs() < 1e-12);
        assert!(expected_r(p, 0.02, 0.01).abs() < 1e-12, "no drift -> zero EV regardless of R:R");
    }

    #[test]
    fn drift_moves_probability_monotonically_and_stays_finite() {
        let base = p_up_first(0.02, 0.02, 0.0, 1e-4);
        assert!(p_up_first(0.02, 0.02, 1e-4, 1e-4) > base);
        assert!(p_up_first(0.02, 0.02, -1e-4, 1e-4) < base);
        for mu in [-1.0, -1e-2, 1e-2, 1.0] {
            let p = p_up_first(0.5, 0.5, mu, 1e-6);
            assert!(p.is_finite() && (0.0..=1.0).contains(&p));
        }
        // symmetry: P_up(a, b, μ) = 1 − P_up(b, a, −μ)
        let (a, b, mu, s2) = (0.03, 0.015, 2e-4, 3e-4);
        assert!((p_up_first(a, b, mu, s2) - (1.0 - p_up_first(b, a, -mu, s2))).abs() < 1e-12);
    }
}
