//! Level chain: absorbing Markov chain over market-structure levels (ARCHITECTURE §3.11, Layer 2).
//!
//! Coordinates are **direction-normalised log prices**: "up" means toward the target. For a
//! short trade, pass negated log prices (x → −x) and swap the meaning of the break probabilities.
//!
//! States are (level, direction of arrival). Arriving at level i moving up: it breaks with
//! probability `p_break_up` (continues to the next level up), otherwise it is rejected and
//! travels to the next level down. Arriving moving down works symmetrically with `p_break_down`.
//! The first segment from entry uses the drift/vol first-passage probability.

use anyhow::{bail, ensure};

use crate::passage::p_up_first;

#[derive(Clone, Debug)]
pub struct Level {
    /// Direction-normalised log price.
    pub x: f64,
    /// P(break through | arrived moving toward the target).
    pub p_break_up: f64,
    /// P(break through | arrived moving toward the stop).
    pub p_break_down: f64,
}

#[derive(Clone, Debug)]
pub struct ChainInput {
    pub entry: f64,
    pub stop: f64,
    pub target: f64,
    /// Intermediate levels strictly between stop and target (any order).
    pub levels: Vec<Level>,
    /// Per-bar drift and variance (log units) for the first segment.
    pub mu: f64,
    pub sigma2: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ChainOutput {
    pub p_target: f64,
    /// Volatility-only probability of reaching the first level above before the first below.
    pub p_first_segment_up: f64,
    /// Volatility-only P(target before stop), ignoring levels: the Layer-1 baseline.
    pub p_vol_only: f64,
}

const EPS: f64 = 1e-6;

pub fn solve(input: &ChainInput) -> anyhow::Result<ChainOutput> {
    let ChainInput { entry, stop, target, mu, sigma2, .. } = *input;
    ensure!(stop < entry && entry < target, "need stop < entry < target (direction-normalised)");
    ensure!(sigma2 > 0.0, "sigma2 must be positive");

    let mut levels = input.levels.clone();
    for l in &levels {
        if !(l.x > stop && l.x < target) || l.x == entry {
            bail!("level {} must lie strictly between stop and target and differ from entry", l.x);
        }
    }
    levels.sort_by(|a, b| a.x.total_cmp(&b.x));
    levels.dedup_by(|a, b| a.x == b.x);
    let n = levels.len();

    // Node positions: 0 = stop, 1..=n levels, n+1 = target.
    let pos = |i: usize| match i {
        0 => stop,
        i if i == n + 1 => target,
        i => levels[i - 1].x,
    };
    // Unknown index for (level i in 1..=n, dir: 0 = up, 1 = down).
    let idx = |i: usize, dir: usize| 2 * (i - 1) + dir;
    let m = 2 * n;

    // Build A h = c.
    let mut a = vec![vec![0.0; m]; m];
    let mut c = vec![0.0; m];
    for i in 1..=n {
        let lv = &levels[i - 1];
        let pu = lv.p_break_up.clamp(EPS, 1.0 - EPS);
        let pd = lv.p_break_down.clamp(EPS, 1.0 - EPS);
        // (i, up): break -> (i+1, up) | reject -> (i-1, down)
        let r = idx(i, 0);
        a[r][r] = 1.0;
        add_transition(&mut a, &mut c, r, i + 1, 0, pu, n, &idx);
        add_transition(&mut a, &mut c, r, i - 1, 1, 1.0 - pu, n, &idx);
        // (i, down): break -> (i-1, down) | bounce -> (i+1, up)
        let r = idx(i, 1);
        a[r][r] = 1.0;
        add_transition(&mut a, &mut c, r, i - 1, 1, pd, n, &idx);
        add_transition(&mut a, &mut c, r, i + 1, 0, 1.0 - pd, n, &idx);
    }
    let h = if m > 0 { gauss_solve(a, c)? } else { Vec::new() };
    let value = |node: usize, dir: usize| match node {
        0 => 0.0,
        nd if nd == n + 1 => 1.0,
        nd => h[idx(nd, dir)],
    };

    let upper = (1..=n + 1).find(|&i| pos(i) > entry).expect("target is above entry");
    let lower = upper - 1;
    let p_seg = p_up_first(pos(upper) - entry, entry - pos(lower), mu, sigma2);
    let p_target = p_seg * value(upper, 0) + (1.0 - p_seg) * value(lower, 1);

    Ok(ChainOutput {
        p_target: p_target.clamp(0.0, 1.0),
        p_first_segment_up: p_seg,
        p_vol_only: p_up_first(target - entry, entry - stop, mu, sigma2),
    })
}

#[allow(clippy::too_many_arguments)]
fn add_transition(a: &mut [Vec<f64>], c: &mut [f64], row: usize, node: usize, dir: usize, p: f64, n: usize, idx: &dyn Fn(usize, usize) -> usize) {
    if node == 0 {
        // absorbed at stop: contributes 0
    } else if node == n + 1 {
        c[row] += p;
    } else {
        a[row][idx(node, dir)] -= p;
    }
}

fn gauss_solve(mut a: Vec<Vec<f64>>, mut b: Vec<f64>) -> anyhow::Result<Vec<f64>> {
    let n = b.len();
    for col in 0..n {
        let piv = (col..n).max_by(|&i, &j| a[i][col].abs().total_cmp(&a[j][col].abs())).unwrap();
        ensure!(a[piv][col].abs() > 1e-14, "singular level-chain system");
        a.swap(col, piv);
        b.swap(col, piv);
        for row in col + 1..n {
            let f = a[row][col] / a[col][col];
            if f != 0.0 {
                for k in col..n {
                    a[row][k] -= f * a[col][k];
                }
                b[row] -= f * b[col];
            }
        }
    }
    let mut x = vec![0.0; n];
    for row in (0..n).rev() {
        let s: f64 = (row + 1..n).map(|k| a[row][k] * x[k]).sum();
        x[row] = (b[row] - s) / a[row][row];
    }
    Ok(x)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn base(levels: Vec<Level>) -> ChainInput {
        ChainInput { entry: 0.0, stop: -0.02, target: 0.04, levels, mu: 0.0, sigma2: 1e-4 }
    }

    #[test]
    fn no_levels_equals_first_passage() {
        let out = solve(&base(vec![])).unwrap();
        assert!((out.p_target - out.p_vol_only).abs() < 1e-12);
        assert!((out.p_target - 1.0 / 3.0).abs() < 1e-12);
    }

    #[test]
    fn stronger_resistance_lowers_probability() {
        let mk = |pu| base(vec![Level { x: 0.02, p_break_up: pu, p_break_down: 0.5 }]);
        let weak = solve(&mk(0.8)).unwrap().p_target;
        let strong = solve(&mk(0.2)).unwrap().p_target;
        assert!(weak > strong, "weak {weak} vs strong {strong}");
        // a level that always breaks reduces to first passage from entry to it, then certain target
        let always = solve(&mk(1.0)).unwrap();
        assert!((always.p_target - p_up_first(0.02, 0.02, 0.0, 1e-4)).abs() < 1e-4);
    }

    #[test]
    fn support_below_entry_helps() {
        let none = solve(&base(vec![])).unwrap().p_target;
        let support = solve(&base(vec![Level { x: -0.01, p_break_up: 0.5, p_break_down: 0.2 }])).unwrap().p_target;
        assert!(support > none);
    }

    #[test]
    fn rejects_bad_geometry() {
        let mut b = base(vec![]);
        b.stop = 0.01;
        assert!(solve(&b).is_err());
        assert!(solve(&base(vec![Level { x: 0.05, p_break_up: 0.5, p_break_down: 0.5 }])).is_err());
    }
}
