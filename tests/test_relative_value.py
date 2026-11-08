from datetime import UTC, datetime, timedelta

import numpy as np
import polars as pl

from ai_trader.features.relative_value import (
    ar1_half_life,
    basis,
    basis_excursions,
    basis_frame,
    factor_residuals,
    rolling_ols,
    spread_z,
)


def _factors(n: int, seed: int = 1):
    g = np.random.default_rng(seed)
    btc = g.standard_t(5, n) * 0.006
    eth = 0.8 * btc + g.standard_t(5, n) * 0.004
    return g, np.column_stack([btc, eth])


def test_rolling_ols_matches_numpy_lstsq():
    g, F = _factors(400)
    y = 0.3 * F[:, 0] + 0.9 * F[:, 1] + g.normal(size=400) * 0.003
    fit = rolling_ols(y, F, 200)
    t = 350
    X = np.column_stack([np.ones(200), F[t - 199 : t + 1]])
    b, *_ = np.linalg.lstsq(X, y[t - 199 : t + 1], rcond=None)
    np.testing.assert_allclose(fit["coef"][t], b, rtol=1e-8, atol=1e-12)
    e = y[t - 199 : t + 1] - X @ b
    assert abs(fit["resid_std"][t] - np.sqrt(e @ e / 197)) < 1e-10
    assert np.isnan(fit["coef"][100]).all()  # fewer than 0.8*window observations


def test_betas_recover_known_loading():
    n = 3000
    g, F = _factors(n)
    y = 1.4 * F[:, 0] + 0.5 * F[:, 1] + g.standard_t(5, n) * 0.004
    out = factor_residuals(y, F, window=720)
    b = out["beta"][-500:]
    assert abs(np.nanmean(b[:, 0]) - 1.4) < 0.1
    assert abs(np.nanmean(b[:, 1]) - 0.5) < 0.1
    z = out["z"][800:]
    assert 0.8 < np.nanstd(z) < 1.3  # z is ~unit scale under the null


def test_factor_residual_z_detects_injected_deviation():
    n = 2000
    g, F = _factors(n, 3)
    y = 1.0 * F[:, 0] + 0.5 * F[:, 1] + g.normal(size=n) * 0.003
    y[1500:1510] += 0.012  # +12% idiosyncratic move over 10 h (~4.7 sigma of a 72h residual sum)
    price = 100 * np.exp(np.cumsum(y))
    out = factor_residuals(y, F, window=720, lookback=72, alt_price=price)
    assert np.nanmax(out["z"][1505:1560]) > 3
    assert np.nanmax(np.abs(out["z"][1000:1490])) < 3.5
    assert out["gap_pct"][1515] > 0.08  # actual above factor-implied fair value
    np.testing.assert_allclose(out["fair_price"][1515] * (1 + out["gap_pct"][1515]), price[1515], rtol=1e-10)


def test_factor_residuals_truncation_invariant():
    n = 1500
    g, F = _factors(n, 5)
    y = 0.9 * F[:, 0] + 0.3 * F[:, 1] + g.normal(size=n) * 0.003
    y[700] = np.nan
    full = factor_residuals(y, F, window=500, lookback=72)
    cut = factor_residuals(y[:1100], F[:1100], window=500, lookback=72)
    for k in ("z", "resid", "hedged", "gap_pct", "resid_std"):
        np.testing.assert_allclose(full[k][:1100], cut[k], equal_nan=True)
    np.testing.assert_allclose(full["beta"][:1100], cut["beta"], equal_nan=True)
    # beta applied at t is the fit through t-1
    np.testing.assert_allclose(full["beta"][1000], full["beta_next"][999])


def _cointegrated(n: int, seed: int = 7, phi: float = 0.95):
    g = np.random.default_rng(seed)
    lb = np.log(80) + np.cumsum(g.normal(size=n) * 0.01)
    u = np.zeros(n)
    for t in range(1, n):
        u[t] = phi * u[t - 1] + g.normal() * 0.002
    return lb + 0.05 + u, lb, u


def test_spread_z_detects_deviation_and_is_causal():
    la, lb, _ = _cointegrated(3000)
    la[2500] += 0.03  # one-bar 3% dislocation (spread std ~0.0064)
    out = spread_z(la, lb, window=720)
    assert out["z"][2500] > 4
    assert np.nanmax(np.abs(out["z"][800:2490])) < 4.5
    cut = spread_z(la[:2200], lb[:2200], window=720)
    np.testing.assert_allclose(out["z"][:2200], cut["z"], equal_nan=True)
    assert np.isnan(out["z"][:720]).all()


def test_spread_z_hedged_recovers_ratio():
    g = np.random.default_rng(11)
    n = 3000
    lb = np.log(50) + np.cumsum(g.normal(size=n) * 0.01)
    la = 0.2 + 1.5 * lb + g.normal(size=n) * 0.003
    out = spread_z(la, lb, window=720, hedge=True)
    assert abs(np.nanmedian(out["beta"][1000:]) - 1.5) < 0.05
    assert 0.8 < np.nanstd(out["z"][1000:]) < 1.3


def test_ar1_half_life():
    _, _, u = _cointegrated(20000, phi=0.9)
    hl = ar1_half_life(u)
    assert abs(hl["phi"] - 0.9) < 0.02
    assert abs(hl["half_life"] - (-np.log(2) / np.log(0.9))) < 1.0


def test_basis_and_excursions():
    assert abs(basis(np.array([100.1]), np.array([100.0]))[0] - 1e4 * np.log(1.001)) < 1e-9
    g = np.random.default_rng(2)
    n = 6000
    t0 = datetime(2026, 11, 1, tzinfo=UTC)
    ts = [t0 + timedelta(seconds=3 * i) for i in range(n)]
    spot = 100 * np.exp(np.cumsum(g.normal(size=n) * 1e-4))
    b = g.normal(size=n) * 1.0  # bps noise
    b[5000] = 15.0
    perp = spot * np.exp(b / 1e4)
    P = pl.DataFrame({"event_ts": ts, "mid": perp})
    S = pl.DataFrame({"event_ts": ts, "mid": spot})
    bf = basis_frame(P, S, window=timedelta(hours=1), min_obs=500)
    assert bf.filter(pl.col("event_ts") == ts[5000])["b_z"][0] > 8
    ex = basis_excursions(bf, z_thresh=8, horizon=timedelta(minutes=5))
    assert ex.height == 1 and ex["reversion"][0] > 0.8
