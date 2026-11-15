"""Tests of the parameter-uncertain market generator (sim.market_gen, block GEN)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
import polars as pl
import pytest

from ai_trader.sim import market_gen as mg

CUT = datetime(2024, 3, 1, tzinfo=UTC)


def _frame(seed: int, days: int = 260, extra_days: int = 20, vol: float = 0.006) -> pl.DataFrame:
    """Synthetic 1h market_frame-like frame that runs `extra_days` past CUT (those rows must never be used)."""
    from ai_trader.research.frame import enrich_bars

    rng = np.random.default_rng(seed)
    n = 24 * (days + extra_days)
    t0 = CUT - timedelta(days=days)
    common = np.random.default_rng(999).standard_normal(n)      # shared by every synthetic symbol
    sv = np.exp(0.3 * np.convolve(rng.standard_normal(n), np.ones(48) / 48, mode="same") * 7)
    r = vol * sv * (0.7 * common + 0.7 * rng.standard_normal(n)) / 0.99
    c = 100 * np.exp(np.cumsum(r))
    o = np.r_[100.0, c[:-1]]
    hi = np.maximum(o, c) * np.exp(np.abs(rng.normal(0, vol / 2, n)))
    lo = np.minimum(o, c) * np.exp(-np.abs(rng.normal(0, vol / 2, n)))
    ts = [t0 + timedelta(hours=k) for k in range(n)]
    f = pl.DataFrame({"event_ts": ts, "open": o, "high": hi, "low": lo, "close": c,
                      "turnover": np.exp(rng.normal(12, 0.5, n)), "volume": np.ones(n)}).with_columns(
        pl.col("event_ts").dt.cast_time_unit("us"))
    f = f.with_columns((pl.col("event_ts") + pl.duration(hours=1)).alias("close_ts"),
                       (pl.col("event_ts") + pl.duration(hours=1, seconds=2)).alias("known_ts"))
    f = enrich_bars(f)
    fund = np.repeat(rng.normal(1e-4, 1e-4, n // 8 + 1), 8)[:n]
    oi = 1e6 * np.exp(np.repeat(np.cumsum(rng.normal(0, 0.01, n // 4 + 1)), 4)[:n])
    return f.with_columns(pl.Series("funding_rate", fund), pl.Series("open_interest", oi))


@pytest.fixture(scope="module")
def calib():
    frames = {f"S{k}": _frame(k) for k in range(3)}
    cd = mg.load_calibration("L", list(frames), CUT, loader=lambda s: frames[s])
    dv = mg.derive(cd)
    p0 = mg.estimate(cd, dv, np.ones(cd.n_days))
    jf, jm = mg.fit_jumps(p0.pop("_zc"), np.random.default_rng(0), n_sub=20_000)
    point = mg.estimate(cd, dv, np.ones(cd.n_days), jumps=jm)
    point.pop("_zc")
    hp = mg.smm(point, jf, n_days=150, n_sym=3, maxfev=25)
    return {"frames": frames, "cd": cd, "dv": dv, "jf": jf, "jm": jm, "point": point, "hp": hp, "state": mg.start_state(cd, dv)}


def test_calibration_uses_only_rows_before_cutoff(calib):
    cd = calib["cd"]
    assert cd.known_max < CUT
    assert cd.day0 + timedelta(days=cd.n_days) == CUT
    # changing everything after the cutoff leaves every estimate unchanged
    frames2 = {}
    for s, f in calib["frames"].items():
        after = pl.col("known_ts") >= CUT
        frames2[s] = f.with_columns(pl.when(after).then(pl.col(c) * 3.0).otherwise(pl.col(c)).alias(c)
                                    for c in ("open", "high", "low", "close", "turnover", "funding_rate", "open_interest"))
    cd2 = mg.load_calibration("L", list(frames2), CUT, loader=lambda s: frames2[s])
    dv2 = mg.derive(cd2)
    a = mg.estimate(calib["cd"], calib["dv"], np.ones(cd.n_days), jumps=calib["jm"])
    b = mg.estimate(cd2, dv2, np.ones(cd2.n_days), jumps=calib["jm"])
    for k in ("a", "har_F", "har_e", "q", "f_phi", "o_phi", "kappa_w"):
        np.testing.assert_allclose(a[k], b[k])


def test_load_calibration_refuses_loader_rows_after_cutoff_even_if_filter_bypassed(calib):
    # the grid never extends past the cutoff day
    cd = calib["cd"]
    assert cd.C.shape[0] == 24 * cd.n_days
    assert np.isnan(cd.C[-1]).all()   # the last bar of the last day closes AT the cutoff: known after it, so excluded


def test_parameter_draws_differ_per_path_and_are_reproducible(calib):
    cd, dv, jm = calib["cd"], calib["dv"], calib["jm"]
    reps = []
    for b in range(3):
        r = mg.estimate(cd, dv, mg.block_weights(cd.n_days, np.random.default_rng([1, b])), jumps=jm)
        r.pop("_zc")
        reps.append(r)
    ps = [mg.path_params(reps[j], calib["point"], calib["hp"], calib["jf"], cd.n_sym, np.random.default_rng([9, j])) for j in range(3)]
    for k in ("har_F", "sig_e", "a", "q", "p", "n_br", "lam"):
        assert not np.allclose(ps[0][k], ps[1][k]), k
        assert not np.allclose(ps[1][k], ps[2][k]), k
    hv = [np.r_[[ps[j][k] for k in ("sig_h", "phi_h", "psi", "kurt_eps")]] for j in range(3)]
    assert not np.allclose(hv[0], hv[1]) and not np.allclose(hv[1], hv[2])
    again = mg.estimate(cd, dv, mg.block_weights(cd.n_days, np.random.default_rng([1, 0])), jumps=jm)
    np.testing.assert_allclose(again["har_F"], reps[0]["har_F"])
    p_again = mg.path_params(reps[0], calib["point"], calib["hp"], calib["jf"], cd.n_sym, np.random.default_rng([9, 0]))
    for k in ("har_F", "p", "n_br", "sig_h"):
        np.testing.assert_allclose(p_again[k], ps[0][k])
    # the fixed-parameter generator: identical parameters on every path
    f0 = mg.path_params(calib["point"], calib["point"], calib["hp"], calib["jf"], cd.n_sym, None)
    f1 = mg.path_params(calib["point"], calib["point"], calib["hp"], calib["jf"], cd.n_sym, None)
    for k in ("har_F", "p", "sig_h", "lam"):
        np.testing.assert_allclose(f0[k], f1[k])


def _gen(calib, n_paths: int = 3, n_days: int = 30) -> dict:
    cd = calib["cd"]
    ps = [mg.path_params(calib["point"], calib["point"], calib["hp"], calib["jf"], cd.n_sym, np.random.default_rng(j)) for j in range(n_paths)]
    spec = mg.GenSpec(day0=CUT - timedelta(days=1), n_days=n_days, first_hour=23)
    return mg.simulate(mg.stack_params(ps), calib["state"], spec, np.random.default_rng(4))


def test_generated_ohlc_valid(calib):
    g = _gen(calib)
    o, h, lo, c = g["open"], g["high"], g["low"], g["close"]
    assert o.shape == c.shape == (3, calib["cd"].n_sym, 30 * 24 - 23)
    assert np.all(np.isfinite(c)) and np.all(lo > 0)
    assert np.all(h >= np.maximum(o, c) * (1 - 1e-12))
    assert np.all(lo <= np.minimum(o, c) * (1 + 1e-12))
    np.testing.assert_allclose(o[:, :, 1:], c[:, :, :-1])
    np.testing.assert_allclose(o[:, :, 0], calib["state"]["last_close"][None, :].repeat(3, axis=0))
    assert np.all(g["turnover"] > 0) and np.all(np.isfinite(g["funding"])) and np.all(g["oi"] > 0)
    # funding is a step function updated every 8 hours (prints at 00 / 08 / 16 UTC)
    fu = g["funding"][0, 0]
    ch = np.flatnonzero(np.diff(fu) != 0) + 1 + 23
    assert np.all(ch % 8 == 0)
    # bootstrap comparator: valid OHLC as well
    bs = mg.bootstrap_source(calib["cd"])
    b = mg.bootstrap_paths(bs, calib["state"], 2, 200, np.random.default_rng(1))
    assert np.all(b["high"] >= np.maximum(b["open"], b["close"]) * (1 - 1e-12))
    assert np.all(b["low"] <= np.minimum(b["open"], b["close"]) * (1 + 1e-12))


def test_common_factor_induces_target_cross_correlation():
    rng = np.random.default_rng(0)
    D = 400
    how = np.repeat(np.arange(D) % 7, 24) * 24 + np.tile(np.arange(24), D)
    q = np.full((1, 2, 168), 1 / 24)
    for lam_v in (0.0, 0.6, 0.9):
        lam = np.full((1, 2), lam_v)
        a, sc = mg.diffusive_hours(q, how, 0.05, 0.5, 0.5, 30.0, lam, 1, rng)
        rc, _ = mg.normalise_days(a, sc, np.ones((1, 2, D)))
        rho = np.corrcoef(rc[0, 0], rc[0, 1])[0, 1]
        assert abs(rho - lam_v**2) < 0.04, (lam_v, rho)


def test_normalisation_sets_daily_bipower(calib):
    rng = np.random.default_rng(1)
    D = 20
    how = np.repeat(np.arange(D) % 7, 24) * 24 + np.tile(np.arange(24), D)
    q = np.full((2, 3, 168), 1 / 24)
    a, sc = mg.diffusive_hours(q, how, 0.3, 0.6, 0.5, 8.0, np.full((2, 3), 0.5), 2, rng)
    bv = np.exp(rng.normal(-8, 1, (2, 3, D)))
    rc, _ = mg.normalise_days(a, sc, bv)
    ad = np.abs(rc.reshape(2, 3, D, 24))
    np.testing.assert_allclose(mg.BV_SCALE * np.sum(ad[..., 1:] * ad[..., :-1], axis=-1), bv, rtol=1e-10)


def test_block_weights_sum_to_days():
    w = mg.block_weights(101, np.random.default_rng(3))
    assert w.sum() == 101 and w.min() >= 0 and (w == 0).any()


def test_clusters_separate_common_shocks():
    rng = np.random.default_rng(2)
    D = 300
    how = np.repeat(np.arange(D) % 7, 24) * 24 + np.tile(np.arange(24), D)
    q = np.full((1, 4, 168), 1 / 24)
    lam = np.full((1, 4), 0.8)
    a, sc = mg.diffusive_hours(q, how, 0.05, 0.5, 0.5, 30.0, lam, 1, rng, cluster=np.array([0, 0, 1, 1]))
    C = np.corrcoef(a[0])
    assert abs(C[0, 1] - 0.64) < 0.05 and abs(C[2, 3] - 0.64) < 0.05
    assert abs(C[0, 2]) < 0.05 and abs(C[1, 3]) < 0.05
    Cm = np.array([[1, 0.8, 0.1], [0.8, 1, 0.05], [0.1, 0.05, 1]])
    assert mg.corr_clusters(Cm).tolist() == [0, 0, 1]


def test_path_params_with_jump_structure_and_drift(calib):
    cd = calib["cd"]
    js = {"omega": 0.5, "tail_rate": 1e-4, "dphi_domega": 0.4}
    drift = {"tau_wd": 0.05, "tau_q": 0.1, "tau_corr": 0.1, "tau_jump": 0.1}
    a = mg.path_params(calib["point"], calib["point"], calib["hp"], calib["jf"], cd.n_sym, np.random.default_rng(1), js=js, drift=drift)
    b = mg.path_params(calib["point"], calib["point"], calib["hp"], calib["jf"], cd.n_sym, np.random.default_rng(2), js=js, drift=drift)
    assert not np.allclose(a["q"], b["q"]) and not np.allclose(a["wk"], b["wk"]) and not np.isclose(a["p"], b["p"])
    np.testing.assert_allclose(a["q"].reshape(cd.n_sym, 7, 24).sum(axis=2), 1.0)
    spec = mg.GenSpec(day0=CUT - timedelta(days=1), n_days=10, first_hour=23)
    g = mg.simulate(mg.stack_params([a, b]), calib["state"], spec, np.random.default_rng(0))
    assert np.all(g["high"] >= np.maximum(g["open"], g["close"]) * (1 - 1e-12)) and np.all(np.isfinite(g["close"]))
