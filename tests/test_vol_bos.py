"""Tests for features.vol_bos and research.vol_bos_study (H-VBOS1-4)."""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta

import numpy as np
import polars as pl
import pytest

from ai_trader.features import vol_bos as vb
from ai_trader.research import vol_bos_study as st


def _rv(n: int, seed: int = 0, shift_at: int | None = None, mult: float = 4.0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    rv = np.exp(rng.normal(np.log(1e-4), 0.35, n))
    if shift_at is not None:
        rv[shift_at:] *= mult
    return rv


def _ohlc(n: int, seed: int = 1) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    c = 100 * np.exp(np.cumsum(rng.normal(0, 0.02, n)))
    hi = c * np.exp(np.abs(rng.normal(0, 0.01, n)))
    lo = c * np.exp(-np.abs(rng.normal(0, 0.01, n)))
    return c, hi, lo


# ------------------------------------------------------------------ causality
@pytest.mark.parametrize("bpd", [1, 6])
def test_detectors_are_causal(bpd):
    n = 900 * bpd if bpd == 1 else 300 * bpd
    rv = _rv(n, 3)
    cut = int(n * 0.7)
    full = vb.detector_onsets(rv, bpd)
    part = vb.detector_onsets(rv[:cut], bpd)
    assert full.shape[0] == len(vb.detector_grid(bpd)) == 62
    np.testing.assert_array_equal(full[:, :cut], part)


def test_semivariance_structure_seasonal_causal():
    n = 1200
    rp, rn = _rv(n, 4), _rv(n, 5)
    c, hi, lo = _ohlc(n)
    cut = 800
    for w, nn, q in vb.semi_grid()[:5]:
        np.testing.assert_array_equal(vb.semi_breaks(rp, rn, 1, w, nn, q)[:cut], vb.semi_breaks(rp[:cut], rn[:cut], 1, w, nn, q))
    np.testing.assert_array_equal(vb.semi_dominance(rp, rn, 1)[:cut], vb.semi_dominance(rp[:cut], rn[:cut], 1))
    for k in (2, 5):
        b1, c1 = vb.structure(c, hi, lo, k)
        b2, c2 = vb.structure(c[:cut], hi[:cut], lo[:cut], k)
        np.testing.assert_array_equal(b1[:cut], b2)
        np.testing.assert_array_equal(c1[:cut], c2)
    slot = np.arange(n) % 6
    np.testing.assert_allclose(vb.seasonal_factor(rp, slot, 6)[:cut], vb.seasonal_factor(rp[:cut], slot[:cut], 6))
    np.testing.assert_allclose(vb.har_level_forecast(np.log(rp), 1)[:cut], vb.har_level_forecast(np.log(rp[:cut]), 1), equal_nan=True)


def test_trailing_quantile_excludes_current_bar():
    x = np.arange(10, dtype=float)
    thr = vb.trailing_quantile(x, 3, 1.0)
    assert np.isnan(thr[2]) and thr[3] == 2.0 and thr[9] == 8.0


# ------------------------------------------------------------------ planted regime shift
def test_planted_vol_shift_is_detected():
    n, at = 600, 400
    rv = _rv(n, 7, shift_at=at, mult=4.0)
    assert np.flatnonzero(vb.range_break(rv, 1, 5, 120, 1.0) & (np.arange(n) >= at))[0] <= at + 5
    cus = vb.cusum(np.log(rv), 1, 60, 0.5, 4.0)
    assert np.flatnonzero(cus & (np.arange(n) >= at))[0] <= at + 5
    F = vb.har_level_forecast(np.log(rv), 1)
    har = vb.har_surprise(rv, F, 1.5, 2)
    assert np.flatnonzero(har & (np.arange(n) >= at))[0] <= at + 5
    # a calm series rarely fires the strict range break: far fewer onsets before the shift than bars
    assert vb.range_break(rv, 1, 20, 180, 1.0)[:at].sum() <= 5


def test_planted_upside_semivariance_break_is_signed():
    n, at = 600, 400
    rp, rn = _rv(n, 8), _rv(n, 9)
    rp[at:] *= 5.0
    s = vb.semi_breaks(rp, rn, 1, 5, 120, 0.95)
    first = np.flatnonzero((s != 0) & (np.arange(n) >= at))[0]
    assert first <= at + 5 and s[first] == 1


# ------------------------------------------------------------------ structure: hand-made example
def test_structure_hand_example():
    high = np.array([10, 11, 13, 11, 10, 11, 12, 14, 12, 11, 9, 8], float)
    low = np.array([9, 10, 12, 10, 9, 10, 11, 13, 11, 10, 7, 6], float)
    close = (high + low) / 2
    hc, lc = vb.fractal_pivots(high, low, 2)
    assert {i: hc[i] for i in np.flatnonzero(np.isfinite(hc))} == {4: 13.0, 9: 14.0}
    assert {i: lc[i] for i in np.flatnonzero(np.isfinite(lc))} == {6: 9.0}
    bos, choch = vb.structure(close, high, low, 2)
    assert {int(i): int(bos[i]) for i in np.flatnonzero(bos)} == {7: 1, 10: -1}
    assert np.flatnonzero(choch).tolist() == [10]


def test_decluster_and_bars_since():
    on = np.zeros(30, bool)
    on[[2, 5, 13, 14, 25]] = True
    assert np.flatnonzero(vb.decluster(on, 10)).tolist() == [2, 13, 25]
    bs = vb.bars_since(on)
    assert np.isinf(bs[0]) and bs[2] == 0 and bs[4] == 2 and bs[13] == 0


# ------------------------------------------------------------------ nested walk-forward never sees the outer block
def test_outer_plan_folds_precede_block():
    first = st.dix(date(2020, 10, 1))
    for p in st.outer_plan(first, st.EMBARGO):
        a, _ = p["block"]
        assert p["train_end"] == a - st.EMBARGO
        for lo, hi in p["folds"]:
            assert first <= lo < hi <= p["train_end"] < a


def test_inner_selection_ignores_outer_and_purge_zone_data():
    first = st.dix(date(2020, 7, 1))
    plan = st.outer_plan(first, st.EMBARGO)
    rng = np.random.default_rng(0)
    for p in plan[2:5]:
        a, _ = p["block"]
        limit = p["train_end"]
        d0 = np.arange(first, a + 180)
        n = len(d0)
        # cfg 0: modestly positive everywhere; cfg 1: hugely positive ONLY in the purge zone and the outer block
        sym = np.zeros(2 * n, int)
        cfg = np.r_[np.zeros(n, int), np.ones(n, int)]
        dd = np.r_[d0, d0]
        v0 = 0.3 + rng.normal(0, 1, n)
        v1 = np.where(d0 + st.H_MAX > limit, 50.0, -0.3) + rng.normal(0, 1, n)
        ev = st.Events(sym, cfg, dd, {h: np.r_[v0, v1] for h in st.H})
        c, _ = st.ev_select(ev, None, [0, 1], p["folds"], limit, 10)
        assert c == 0
    # strategies: a config that is great only from the outer block on is never selected for that block
    nd = st.dix(date(2026, 7, 1))
    ew = np.vstack([0.02 + rng.normal(0, 0.1, nd), rng.normal(0, 0.1, nd)])
    for p in plan[2:5]:
        a, _ = p["block"]
        x = ew.copy()
        x[1, a - st.EMBARGO:] += 1.0
        j, _ = st.select_median(st.book_fold_sharpes(x, p["folds"]))
        assert j == 0


# ------------------------------------------------------------------ trade machinery reproduces evaluate.strategy_pnl
def test_trades_reproduce_strategy_pnl(monkeypatch):
    from ai_trader.research import evaluate

    rng = np.random.default_rng(2)
    n = 1500
    s = np.tanh(np.convolve(rng.normal(0, 1, n), np.ones(15) / 4, mode="same"))
    s[np.abs(s) < 0.1] = 0.0
    sig = np.abs(rng.normal(0.02, 0.003, n))
    fwd = rng.normal(0, 0.02, n)
    frame = pl.DataFrame({"event_ts": pl.datetime_range(datetime(2021, 1, 1, tzinfo=UTC), datetime(2021, 1, 1, tzinfo=UTC) + timedelta(hours=4 * (n - 1)), "4h", eager=True),
                          "sigma_next": sig, "fwd_ret": fwd})
    ref = evaluate.strategy_pnl(frame, pl.Series(s), cost_bps=7.5)["pnl"].to_numpy()
    sg = np.sign(s)
    starts = np.flatnonzero((sg != 0) & (sg != np.r_[0, sg[:-1]]))
    d = {"ens": s, "sigma_next": sig, "fwd_ret": fwd, "bos": np.zeros((2, n), np.int8), "choch": np.zeros((2, n), bool),
         "acc_day": np.arange(n), "start_day": 0}
    tr = st.build_trades(d, 0, False, 7.5e-4, entries=starts, dirs=sg[starts].astype(int))
    got = np.bincount(tr.days, weights=tr.vals, minlength=n)
    np.testing.assert_allclose(got, ref, atol=1e-12)
    assert st.take_trades(tr, np.ones(len(starts), bool)) == list(range(len(starts)))


def test_pbo_chunked_matches_reference():
    from ai_trader.research.synthetic_robustness_study import pbo_cscv

    rng = np.random.default_rng(5)
    perf = rng.normal(0, 1, (320, 7)) + np.linspace(0, 0.1, 7)
    a, b = pbo_cscv(perf, 8), st.pbo_cscv_chunked(perf, 8, chunk=13)
    assert a["pbo"] == pytest.approx(b["pbo"]) and a["logit_median"] == pytest.approx(b["logit_median"])
    assert a["is_winner_counts"] == b["is_winner_counts"]


def test_clustered_t_and_holm():
    v = np.array([1.0, 1.0, 3.0, 2.0, 2.0, 0.0, 1.0])
    d = np.array([0, 1, 2, 15, 16, 30, 45])
    r = st.clustered_t(v, d, bucket=10)
    assert r["n_clusters"] == 4 and r["mean"] == pytest.approx(np.mean([5 / 3, 2.0, 0.0, 1.0]))
    h = st.holm({"a": 0.01, "b": 0.04, "c": 0.5})
    assert h == {"a": 0.03, "b": 0.08, "c": 0.5}


def test_bar_realized_semivariances_sum_to_rv():
    ts = pl.datetime_range(datetime(2022, 1, 1, tzinfo=UTC), datetime(2022, 1, 10, 23, tzinfo=UTC), "1h", eager=True)
    rng = np.random.default_rng(1)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, len(ts))))
    h1 = pl.DataFrame({"event_ts": ts, "close": close})
    bars = pl.datetime_range(datetime(2022, 1, 2, tzinfo=UTC), datetime(2022, 1, 10, tzinfo=UTC), "1d", eager=True).to_numpy()
    r = vb.bar_realized(h1, bars, 24)
    np.testing.assert_allclose(r["rs_pos"] + r["rs_neg"], r["rv"])
    lr = np.diff(np.log(close))
    np.testing.assert_allclose(r["rv"][0], np.sum(lr[23:47] ** 2))
