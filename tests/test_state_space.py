import numpy as np
import pytest

from ai_trader.features.state_space import _conc_loglik, llt_filter, llt_mle, steady_state_gain


def test_filter_recovers_constant_slope():
    rng = np.random.default_rng(0)
    n, slope, sd = 3000, 0.002, 0.01
    y = 5.0 + slope * np.arange(n) + sd * rng.standard_normal(n)
    out = llt_filter(y, r=sd**2, q_zeta=1e-6 * sd**2)
    assert abs(np.mean(out.slope[-500:]) - slope) < 2e-4
    assert abs(out.level[-1] - (5.0 + slope * (n - 1))) < 3 * sd


def test_filter_tracks_slope_change_and_sign():
    rng = np.random.default_rng(1)
    n = 4000
    beta = np.r_[np.full(n // 2, 0.003), np.full(n - n // 2, -0.003)]
    y = np.cumsum(beta) + 0.01 * rng.standard_normal(n)
    out = llt_filter(y, r=1e-4, q_zeta=1e-4 * 1e-3)
    assert np.mean(out.slope[n // 2 - 300:n // 2]) > 0.002
    assert np.mean(out.slope[-300:]) < -0.002


def test_filter_is_causal():
    rng = np.random.default_rng(2)
    y = np.cumsum(0.01 * rng.standard_normal(2000))
    r = 1e-4 * (1 + rng.random(2000))
    full = llt_filter(y, r, 1e-3 * r)
    for k in (300, 1111, 1999):
        part = llt_filter(y[:k], r[:k], 1e-3 * r[:k])
        assert np.allclose(part.slope, full.slope[:k], equal_nan=True)
        assert np.allclose(part.level, full.level[:k], equal_nan=True)


def test_gain_depends_only_on_ratio_and_matches_filter():
    k1, k2 = steady_state_gain(1e-3, r=1.0)
    a1, a2 = steady_state_gain(1e-3 * 4e-4, r=4e-4)
    assert k1 == pytest.approx(a1, rel=1e-6) and k2 == pytest.approx(a2, rel=1e-6)
    assert 0 < k2 < k1 < 1
    # faster filter for larger q
    assert steady_state_gain(1e-2)[1] > k2 > steady_state_gain(1e-4)[1]


def test_mle_recovers_variance_ratios():
    rng = np.random.default_rng(3)
    n = 3000
    s_eta, s_zeta, s_eps = 1.0, 0.05, 0.3   # q_zeta = 0.0025, q_eps = 0.09
    beta = np.cumsum(s_zeta * rng.standard_normal(n))
    mu = np.cumsum(beta + s_eta * rng.standard_normal(n))
    y = mu + s_eps * rng.standard_normal(n)
    fit = llt_mle(y)
    assert 0.0025 / 3 < fit.q_zeta < 0.0025 * 3
    assert fit.sigma2_eta == pytest.approx(1.0, rel=0.3)
    # the optimum beats the truth-shifted alternatives
    yl = (y - y[0]).tolist()
    assert fit.loglik >= _conc_loglik(yl, 0.09, 0.0025 * 30) - 1e-6


def test_mle_random_walk_has_tiny_slope_variance():
    rng = np.random.default_rng(4)
    y = np.cumsum(rng.standard_normal(1500))
    fit = llt_mle(y)
    assert fit.q_zeta < 1e-3
