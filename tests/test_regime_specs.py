"""Tests for research.regime_specs (WS07): change-point pieces, causality of the trailing transforms, statistics."""

from __future__ import annotations

import numpy as np

from ai_trader.research import regime_specs as rs


def test_bocpd_tracks_level_shift_and_resets_run_length():
    rng = np.random.default_rng(0)
    x = np.r_[rng.normal(0, 0.3, 200), rng.normal(2.0, 0.3, 200)]
    pm, pv, rl = rs.bocpd(x, mu0=0.0, kappa0=1.0, alpha0=2.0, beta0=0.1, hazard=1 / 50)
    assert abs(pm[190]) < 0.2
    assert abs(pm[260] - 2.0) < 0.2
    assert rl[215] < 20                      # the run restarted after the shift
    assert np.all(pv > 0)


def test_bocpd_is_causal():
    rng = np.random.default_rng(1)
    x = rng.normal(0, 1, 300)
    a = rs.bocpd(x[:200], 0.0, 1.0, 2.0, 1.0)[0]
    b = rs.bocpd(np.r_[x[:200], x[200:] + 5], 0.0, 1.0, 2.0, 1.0)[0]
    np.testing.assert_allclose(a, b[:200])


def test_pelt_last_segment_start():
    rng = np.random.default_rng(2)
    x = np.r_[rng.normal(0, 1, 150), rng.normal(3, 1, 150)]
    last = rs.pelt_last_start(x, beta=2 * np.log(300), sigma2=1.0)
    assert abs(last[300] - 150) <= 3
    assert last[100] == 0                    # before the shift: one segment from the start
    # online property: last[t] depends on x[:t] only
    last2 = rs.pelt_last_start(np.r_[x[:200], x[200:] - 10], beta=2 * np.log(300), sigma2=1.0)
    np.testing.assert_array_equal(last[:201], last2[:201])


def test_trailing_transforms_are_causal():
    rng = np.random.default_rng(3)
    x = rng.normal(size=400)
    y = x.copy()
    y[300:] += 10
    np.testing.assert_allclose(rs.trailing_pct(x, 180)[:300], rs.trailing_pct(y, 180)[:300])
    np.testing.assert_allclose(rs.trailing_z(x, 90)[:300], rs.trailing_z(y, 90)[:300])
    lv = rng.normal(size=400)
    a = rs.tercile_vs_trailing(lv, x)
    b = rs.tercile_vs_trailing(lv, y)
    np.testing.assert_array_equal(a[:301], b[:301])   # uses strictly past x


def test_agreement_measures():
    a = np.array([0, 0, 1, 1, 2, 2] * 10)
    assert rs.variation_of_information(a, a) < 1e-12
    assert rs.variation_of_information(a, (a + 1) % 3) < 1e-12   # permutation-invariant
    assert abs(rs._ari(a, (a + 1) % 3) - 1.0) < 1e-12
    rng = np.random.default_rng(4)
    assert abs(rs._ari(a, rng.integers(0, 3, len(a)))) < 0.2


def test_mcs_drops_clearly_worse_model_keeps_equals():
    rng = np.random.default_rng(5)
    T = 600
    base = rng.normal(1.0, 0.3, T)
    L = np.column_stack([base + rng.normal(0, 0.05, T), base + rng.normal(0, 0.05, T), base + 0.3 + rng.normal(0, 0.05, T)])
    out = rs.mcs(L, ["a", "b", "bad"], B=300)
    assert "bad" not in out["set"]
    assert set(out["set"]) == {"a", "b"}


def test_dm_and_holm():
    rng = np.random.default_rng(6)
    d = rng.normal(0, 1, 500)
    r = rs.dm_test(d)
    assert abs(r["t"]) < 3 and 0 <= r["p"] <= 1
    r2 = rs.dm_test(d - 0.5)
    assert r2["t"] < -5
    h = rs.holm({"a": 0.01, "b": 0.04, "c": 0.5})
    assert h["a"] == 0.03 and abs(h["b"] - 0.08) < 1e-12 and h["c"] == 0.5


def test_ridge_shrinks_state_coefficients_only():
    rng = np.random.default_rng(7)
    n = 400
    X = np.column_stack([np.ones(n), rng.normal(size=n), rng.integers(0, 2, n)])
    y = 1.0 + 2.0 * X[:, 1] + 0.5 * X[:, 2] + rng.normal(0, 0.1, n)
    b = rs._ridge(X, y, 2)
    assert abs(b[1] - 2.0) < 0.05
    assert 0.3 < b[2] < 0.55                # shrunk toward 0 by lambda 20 out of ~200 ones


def test_t_density_has_requested_variance():
    x = np.linspace(-40, 40, 200001)
    dx = x[1] - x[0]
    f = np.exp(rs.t_logpdf_var(x, 0.0, np.full_like(x, 4.0), 5.0))
    assert abs(f.sum() * dx - 1) < 1e-3
    assert abs((x ** 2 * f).sum() * dx - 4.0) < 0.1


def test_fold_and_segment_helpers():
    from datetime import date
    assert rs.fold_of(date(2021, 12, 31)) == -1
    assert rs.fold_of(date(2022, 1, 1)) == 0
    assert rs.fold_of(date(2026, 8, 1)) == len(rs.ORIGINS) - 1
    assert rs.seg_of(date(2026, 6, 30)) == "llm_in_sample" and rs.seg_of(date(2026, 7, 1)) == "clean_holdout"
