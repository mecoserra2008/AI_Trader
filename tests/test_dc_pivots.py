from types import SimpleNamespace

import numpy as np
import pytest

from ai_trader.features.dc_pivots import dc_engine, ensemble, geometry, indicator_sigma
from ai_trader.research.frame import enrich_bars
from ai_trader.strategies import REGISTRY, assert_causal

from .conftest import make_bars

# Hand-built path with delta = 3 on every bar (k = 3, sigma_prev = 1). See the trace in each test.
PATH = [10, 14, 20, 16, 14, 17, 24, 20, 18, 17, 20, 15, 14, 18, 15, 19]


def run(path, k=3.0, sig=1.0, max_retr=0.618):
    p = np.asarray(path, dtype=float)
    return dc_engine(p, np.full(len(p), sig), k, max_retr)


def test_confirms_at_exactly_k_sigma():
    r = run([10, 11, 12, 12.999])
    assert r.events.height == 0 and (r.m == 0).all()
    r = run([10, 11, 12, 13])
    ev = r.events.row(0, named=True)
    assert ev == {"price": 10.0, "type": -1, "pivot_bar": 0, "confirm_bar": 3}
    assert r.m[3] == 1 and r.ext[3] == 13 and r.L0[3] == 10 and np.isnan(r.L0[2])
    # warm-up high side
    r = run([10, 12, 11, 9])
    assert r.events.row(0) == (12.0, 1, 1, 3) and r.m[3] == -1


def test_reversal_tested_before_extreme_update_and_run_restarts_at_p():
    r = run([10, 14, 20, 16])
    hi = r.events.filter(r.events["type"] == 1).row(0, named=True)
    assert hi == {"price": 20.0, "type": 1, "pivot_bar": 2, "confirm_bar": 3}
    assert r.m[3] == -1 and r.ext[3] == 16  # the new run starts at the confirmation close
    # a drop of 2.9 from the extreme does not confirm; a new high just moves the extreme
    r = run([10, 14, 20, 17.1, 21])
    assert r.events.height == 1 and r.ext[4] == 21 and r.m[4] == 1


def test_levels_shift_and_struct():
    r = run(PATH)
    assert r.L0[1] == 10 and np.isnan(r.L1[1])
    assert r.H0[3] == 20 and r.L0[5] == 14 and r.L1[5] == 10
    assert r.H0[7] == 24 and r.H1[7] == 20 and r.last_ty[7] == 1
    assert r.struct[6] == 0 and r.struct[7] == 1  # needs H1 and L1; HH + HL
    assert r.L0[10] == 17 and r.L1[10] == 14 and r.last_ty[10] == -1
    assert r.H0[11] == 20 and r.H1[11] == 24 and r.struct[11] == 0  # LH + HL = mixed
    assert r.L0[13] == 14 and r.L1[13] == 17 and r.struct[13] == -1  # LH + LL


def test_pivot_bo_is_stateful():
    r = run(PATH)
    assert list(r.pivot_bo[:6]) == [0] * 6
    assert r.pivot_bo[6] == 1 and r.pivot_bo[7] == 1 and r.pivot_bo[10] == 1  # held inside the range
    assert r.pivot_bo[11] == -1 and r.pivot_bo[13] == -1  # below L0 = 17, then held


def test_flag_bo_entry_ratchet_stop_and_no_reentry_on_exit_bar():
    r = run(PATH)
    # bar 6: higher low (14 > 10), close 24 > H0 = 20, retr = (20-14)/(20-10) = 0.6 <= 0.618 -> long
    assert r.flag_bo[5] == 0 and r.flag_bo[6] == 1 and r.flag_entry[6] == 1 and r.fstop[6] == 14
    assert r.fstop[7] == 14 and r.fstop[10] == 17  # ratchets to the new higher low
    # bar 11: close 15 < stop 17 -> flat; short conditions hold (LH 20<24, close < L0, retr 3/7) but exit bar blocks entry
    assert r.flag_bo[11] == 0 and np.isnan(r.fstop[11]) and r.flag_entry[11] == 0
    assert r.flag_bo[12] == -1 and r.flag_entry[12] == -1 and r.fstop[12] == 20
    assert r.fstop[14] == 18  # short stop ratchets down to the lower high
    assert r.flag_bo[15] == 0  # close 19 > 18 -> stopped out


def test_flag_bo_retracement_filter():
    deep = list(PATH)
    deep[4] = 13  # retr = (20-13)/(20-10) = 0.7
    r = run(deep)
    assert r.pivot_bo[6] == 1 and r.flag_bo[6] == 0
    assert run(PATH, max_retr=0.5).flag_bo[6] == 0
    assert run(PATH, max_retr=0.6).flag_bo[6] == 1  # inclusive


def test_threshold_scales_with_k_and_invalid_delta_holds():
    assert run([10, 11, 12, 13], k=4).events.height == 0
    r = dc_engine(np.array([10, 20, 10, 14.0]), np.array([np.nan, 0.0, np.nan, 1.0]), 3)
    assert r.events.height == 0 and r.ext[3] == 14  # engine starts at the first valid delta


def test_indicator_sigma_recursion_and_ready():
    rng = np.random.default_rng(1)
    c = 100 + np.cumsum(rng.standard_normal(200))
    sig, ready = indicator_sigma(c, vol_hl=60, warm=60)
    a = 1 - 0.5 ** (1 / 60)
    mean, var = c[1] - c[0], 0.0
    exp = [np.nan, 0.0]
    for t in range(2, 200):
        dev = (c[t] - c[t - 1]) - mean
        mean += a * dev
        var = (1 - a) * (var + a * dev * dev)
        exp.append(np.sqrt(var))
    assert np.allclose(sig, exp, equal_nan=True)
    assert not ready[:61].any() and ready[61:].all()


def test_ensemble_mean_of_k_votes_and_ready_gate():
    votes = [np.array([1, 1, -1, 0.0]), np.array([1, 0, -1, 0.0]), np.array([1, 1, 1, 0.0]),
             np.array([1, -1, -1, 0.0]), np.array([1, 1, -1, 0.0])]
    res = {k: SimpleNamespace(struct=v, pivot_bo=v, flag_bo=-v) for k, v in zip((3, 4, 6, 8, 12), votes)}
    e = ensemble(res, np.array([False, True, True, True]))
    assert np.allclose(e["struct"], [0, 0.4, -0.6, 0]) and np.allclose(e["flag_bo"], [0, -0.4, 0.6, 0])


@pytest.fixture(scope="module")
def frame():
    return enrich_bars(make_bars(3000, freq_h=4, seed=3))


def test_geometry_ensemble_matches_engines(frame):
    g = geometry(frame)
    assert set(np.unique(np.round(g["geo_pivot_bo"].to_numpy() * 5))) <= set(range(-5, 6))
    assert (g["geo_struct"].to_numpy()[~g["geo_ready"].to_numpy()] == 0).all()
    assert np.abs(g["geo_flag_bo"].to_numpy()).sum() > 0 and np.isfinite(g["geo_H0_6"].to_numpy()[-1])


@pytest.mark.parametrize("name", ["geo_struct", "geo_pivot_bo", "geo_flag_bo"])
def test_geo_strategies_causal(frame, name):
    s = REGISTRY[name]()
    assert_causal(s, frame)
    sig = s(frame).to_numpy()
    assert (sig[:61] == 0).all() and (sig != 0).sum() > 100
    assert s.spec.family == "trend" and "H-GT1" in s.spec.hypothesis
