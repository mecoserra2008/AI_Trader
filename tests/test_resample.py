"""Tests for ai_trader.sim.resample (B5 generators)."""

from __future__ import annotations

import numpy as np
import pytest

from ai_trader.sim.resample import (
    acf,
    aggregate,
    excess_kurtosis,
    fit_transition,
    politis_white_block,
    reconstruct,
    regime_indices,
    relative_bars,
    run_lengths,
    simulate_chain,
    stationary_indices,
)


def _garch_like(n: int, rng: np.random.Generator) -> np.ndarray:
    """Returns with vol clustering (GARCH(1,1), t innovations)."""
    r = np.empty(n)
    v = 1e-4
    for t in range(n):
        r[t] = np.sqrt(v) * rng.standard_t(5) / np.sqrt(5 / 3)
        v = 1e-6 + 0.1 * r[t] ** 2 + 0.88 * v
    return r


def _bars(n: int, rng: np.random.Generator):
    r = _garch_like(n, rng)
    c = 100 * np.exp(np.cumsum(r))
    o = np.r_[100.0, c[:-1]] * np.exp(rng.normal(0, 1e-4, n))
    sd = np.abs(r) + 1e-3
    h = np.maximum(o, c) * np.exp(np.abs(rng.normal(0, 1, n)) * sd)
    lo = np.minimum(o, c) * np.exp(-np.abs(rng.normal(0, 1, n)) * sd)
    return o, h, lo, c


def test_bootstrap_preserves_marginal_distribution():
    rng = np.random.default_rng(1)
    x = _garch_like(6000, rng)
    idx = stationary_indices(len(x), len(x), mean_block=25, n_paths=40, rng=rng)
    assert idx.shape == (40, 6000) and idx.min() >= 0 and idx.max() < len(x)
    y = x[idx].ravel()
    qs = [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]
    sd = x.std()
    assert np.allclose(np.quantile(y, qs), np.quantile(x, qs), atol=0.06 * sd)
    assert abs(y.std() / sd - 1) < 0.03
    # two-sample KS distance between the pooled bootstrap draws and the source
    grid = np.sort(x)
    ks = np.max(np.abs(np.searchsorted(np.sort(y), grid, side="right") / len(y) - np.arange(1, len(x) + 1) / len(x)))
    assert ks < 0.02
    # blocks: consecutive source indices inside a block, mean block length close to b
    steps = np.diff(idx, axis=1)
    cont = (steps == 1) | (steps == -(len(x) - 1))
    breaks = (~cont).sum(axis=1) + 1
    assert 20 < (idx.shape[1] / breaks).mean() < 31


def test_bootstrap_keeps_vol_clustering_at_short_lags():
    rng = np.random.default_rng(2)
    x = _garch_like(8000, rng)
    a_src = acf(np.abs(x), [1, 5])
    b = politis_white_block(np.abs(x))["b"]
    idx = stationary_indices(len(x), len(x), mean_block=b, n_paths=20, rng=rng)
    a_gen = acf(np.abs(x[idx]), [1, 5]).mean(axis=0)
    assert a_src[0] > 0.1
    assert a_gen[0] > 0.6 * a_src[0]
    # iid shuffle destroys it
    assert abs(acf(np.abs(rng.permutation(x)), [1])[0]) < 0.05


def test_politis_white_longer_for_persistent_series():
    rng = np.random.default_rng(3)
    iid = rng.normal(size=5000)
    ar = np.empty(5000)
    ar[0] = 0
    for t in range(1, 5000):
        ar[t] = 0.9 * ar[t - 1] + rng.normal()
    assert politis_white_block(ar)["b"] > 3 * politis_white_block(iid)["b"]
    pw = politis_white_block(ar)
    assert pw["b"] <= pw["b_max"]


def test_regime_resampling_preserves_transition_frequencies():
    rng = np.random.default_rng(4)
    P_true = np.array([[0.95, 0.04, 0.01], [0.03, 0.94, 0.03], [0.02, 0.08, 0.90]])
    src = simulate_chain(P_true, 30000, 1, rng, start_probs=np.ones(3))[0]
    names = ["A", "B", "C"]
    cells = [names[s] for s in src]
    fit = fit_transition(cells)
    assert fit["states"] == names
    assert np.allclose(fit["P"], P_true, atol=0.02)
    chain = simulate_chain(fit["P"], 20000, 10, rng, start_probs=fit["freq"])
    cnt = np.zeros((3, 3))
    for row in chain:
        np.add.at(cnt, (row[:-1], row[1:]), 1)
    P_sim = cnt / cnt.sum(axis=1, keepdims=True)
    assert np.allclose(P_sim, fit["P"], atol=0.02)
    # durations: mean run length of state A ~ 1 / (1 - P_AA)
    runs = run_lengths([names[s] if s == 0 else None for s in chain[0]])
    assert abs(runs.mean() - 1 / (1 - fit["P"][0, 0])) < 5
    # every resampled bar comes from the simulated state's pool
    src_state = src.copy()
    src_state[:100] = -1  # unlabelled warm-up is never drawn
    idx = regime_indices(src_state, chain, mean_block=30, rng=rng)
    assert np.array_equal(src_state[idx], chain)


def test_fit_transition_skips_unlabelled_pairs():
    fit = fit_transition(["A", "A", None, "B", "B", "A"])
    assert fit["counts"].tolist() == [[1.0, 0.0], [1.0, 1.0]]


def test_reconstructed_ohlc_is_valid_and_exact_on_identity():
    rng = np.random.default_rng(5)
    o, h, lo, c = _bars(3000, rng)
    rb = relative_bars(o, h, lo, c)
    assert np.all(rb.u >= 0) and np.all(rb.d >= 0)
    # identity resampling reproduces the source bars 1..n-1
    ident = reconstruct(rb, np.arange(len(rb))[None, :], c0=c[0])
    for k, src in (("open", o), ("high", h), ("low", lo), ("close", c)):
        assert np.allclose(ident[k][0], src[1:], rtol=1e-10)
    idx = stationary_indices(len(rb), 2400, mean_block=10, n_paths=8, rng=rng)
    p = reconstruct(rb, idx, c0=c[0])
    assert np.all(p["high"] >= np.maximum(p["open"], p["close"]) * (1 - 1e-12))
    assert np.all(p["low"] <= np.minimum(p["open"], p["close"]) * (1 + 1e-12))
    assert np.all(p["low"] > 0)
    # close-to-close log returns are exactly the resampled r
    lr = np.diff(np.log(p["close"]), axis=1)
    assert np.allclose(lr, rb.r[idx][:, 1:])
    a = aggregate(p, 4)
    assert a["close"].shape == (8, 600)
    assert np.all(a["high"] >= np.maximum(a["open"], a["close"]) * (1 - 1e-12))
    assert np.all(a["low"] <= np.minimum(a["open"], a["close"]) * (1 + 1e-12))
    assert np.allclose(a["close"], p["close"][:, 3::4]) and np.allclose(a["open"], p["open"][:, ::4])


def test_relative_bars_rejects_invalid():
    with pytest.raises(ValueError):
        relative_bars(np.array([1.0, 1.0]), np.array([1.0, 0.9]), np.array([0.9, 0.8]), np.array([1.0, 0.95]))


def test_excess_kurtosis_gaussian_near_zero():
    rng = np.random.default_rng(6)
    assert abs(float(excess_kurtosis(rng.normal(size=200000)))) < 0.05
