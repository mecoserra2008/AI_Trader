//! Range-based volatility estimators (per-bar variance in log units). They're far more
//! efficient than close-to-close: Garman–Klass is about 7x, Yang–Zhang is drift- and
//! opening-gap-robust.

/// Parkinson (1980): (ln H/L)² / (4 ln 2).
pub fn parkinson(high: f64, low: f64) -> f64 {
    (high / low).ln().powi(2) / (4.0 * std::f64::consts::LN_2)
}

/// Garman–Klass (1980), per bar.
pub fn garman_klass(open: f64, high: f64, low: f64, close: f64) -> f64 {
    let hl = (high / low).ln();
    let co = (close / open).ln();
    0.5 * hl * hl - (2.0 * std::f64::consts::LN_2 - 1.0) * co * co
}

/// Yang–Zhang (2000) variance over a window of bars. Inputs are aligned slices; `prev_close[i]`
/// is the close of bar i−1. Returns the per-bar variance.
pub fn yang_zhang(prev_close: &[f64], open: &[f64], high: &[f64], low: &[f64], close: &[f64]) -> Option<f64> {
    let n = open.len();
    if n < 2 || [prev_close.len(), high.len(), low.len(), close.len()].iter().any(|&l| l != n) {
        return None;
    }
    let var = |xs: &[f64]| {
        let m = xs.iter().sum::<f64>() / xs.len() as f64;
        xs.iter().map(|x| (x - m).powi(2)).sum::<f64>() / (xs.len() - 1) as f64
    };
    let o: Vec<f64> = (0..n).map(|i| (open[i] / prev_close[i]).ln()).collect();
    let c: Vec<f64> = (0..n).map(|i| (close[i] / open[i]).ln()).collect();
    let rs = (0..n)
        .map(|i| {
            let (h, l, op, cl) = (high[i], low[i], open[i], close[i]);
            (h / cl).ln() * (h / op).ln() + (l / cl).ln() * (l / op).ln()
        })
        .sum::<f64>()
        / n as f64;
    let nf = n as f64;
    let k = 0.34 / (1.34 + (nf + 1.0) / (nf - 1.0));
    Some(var(&o) + k * var(&c) + (1.0 - k) * rs)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn estimators_are_nonnegative_and_scale() {
        assert!(parkinson(101.0, 99.0) > 0.0);
        assert!(parkinson(102.0, 98.0) > parkinson(101.0, 99.0));
        let pc = [100.0, 100.5, 99.8, 100.2];
        let o = [100.1, 100.4, 99.9, 100.3];
        let h = [101.0, 101.2, 100.5, 101.0];
        let l = [99.5, 99.9, 99.2, 99.8];
        let c = [100.5, 99.8, 100.2, 100.7];
        assert!(yang_zhang(&pc, &o, &h, &l, &c).unwrap() > 0.0);
    }
}
