"""Unit tests for the RFSV (rough_vol_study) and KFD (kalman_fv_study) building blocks: causality and simple identities."""

import numpy as np
import polars as pl

from ai_trader.research import kalman_fv_study as kf
from ai_trader.research import rough_vol_study as rv


def test_gjr_weights_normalised_and_decreasing():
    w = rv.gjr_weights(0.1, 1)
    assert abs(w.sum() - 1) < 1e-12
    assert np.all(np.diff(w) < 0)


def test_kernel_pred_is_causal():
    rng = np.random.default_rng(0)
    x = rng.normal(size=600)
    w = rv.gjr_weights(0.1, 1, L=50)
    a = rv.kernel_pred(x, w)
    y = x.copy()
    y[300:] += 100.0          # changing the future must not change predictions up to t = 300
    b = rv.kernel_pred(y, w)
    assert np.allclose(a[:301], b[:301], equal_nan=True)
    assert not np.allclose(a[301:], b[301:])


def test_trailing_mean_excludes_current():
    x = np.arange(10, dtype=float)
    m = rv.trailing_mean(x, 3, 1.0)
    assert np.isnan(m[2]) and m[3] == 1.0 and m[9] == 7.0


def test_structure_function_recovers_brownian_H():
    rng = np.random.default_rng(1)
    x = np.cumsum(rng.normal(size=20000))      # Brownian motion: H = 0.5
    assert abs(rv.hurst(x, np.arange(1, 21)) - 0.5) < 0.05


def test_local_level_pred_uses_past_only():
    x = np.array([0.0, 1.0, 1.0, 5.0])
    p = kf.ll_pred(x, 0.5)
    assert p[0] == 0.0 and p[1] == 0.0 and p[2] == 0.5 and p[3] == 0.75


def test_fit_K_random_walk_goes_to_one():
    rng = np.random.default_rng(2)
    x = np.cumsum(rng.normal(size=5000))
    K, S = kf.fit_K(x, np.ones(5000), np.ones(5000, bool))
    assert K > 0.95 and abs(S - 1) < 0.1


def test_sim_single_requires_trade_through():
    # a short signal at t = 1 (z > k) is not filled when bar 2's high does not exceed close_1
    d = {"ts": np.arange(6) * 3600, "open": np.full(6, 100.0), "high": np.array([100, 100, 100, 100, 100, 100.0]),
         "low": np.full(6, 99.0), "close": np.full(6, 100.0), "sig": np.full(6, 0.01)}
    z = np.array([0, 3.0, 0, 0, 0, 0])
    assert kf.sim_single(d, z, np.full(6, 0.01), 2.5, 12) == []
    d["high"] = np.array([100, 100, 100.5, 100, 100, 100.0])
    tr = kf.sim_single(d, z, np.full(6, 0.01), 2.5, 12)
    assert len(tr) == 1


def test_pbo_identical_variants_is_not_nan():
    rng = np.random.default_rng(3)
    M = rng.normal(size=(200, 4))
    assert 0.0 <= kf.pbo_cscv(M) <= 1.0


def test_liquid_rows_masks_low_turnover():
    from ai_trader.research.xs_v4_study import liquid_rows

    ts = pl.datetime_range(pl.datetime(2026, 1, 1, time_zone="UTC"), pl.datetime(2026, 1, 3, 23, time_zone="UTC"), "1h", eager=True)
    f = pl.DataFrame({"event_ts": ts, "turnover": [5e5] * 36 + [1e7] * (len(ts) - 36)})
    rows = pl.DataFrame({"event_ts": ts, "lsig2": np.zeros(len(ts))})
    out = liquid_rows(f, rows, 2e7 * 1.0)
    # window t-24..t-1: t <= 36 holds only 5e5 bars (1.2e7 < 2e7); t = 37 holds one 1e7 bar (2.15e7)
    assert out["lsig2"][:37].null_count() == 37
    assert out["lsig2"][37:].null_count() == 0
