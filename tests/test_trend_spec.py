"""WS16 trend_spec: the generalised walker reproduces stress.walk_signal, signals are causal, statistics behave."""

from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
import polars as pl
import pytest

from ai_trader.research import stress as S
from ai_trader.research import trend_spec as T


@pytest.fixture(scope="module")
def frames():
    rng = np.random.default_rng(1)
    n = 24 * 400
    r = rng.standard_t(4, n) * 0.006 + 0.0002 * np.sin(np.arange(n) / 900)
    c = 100 * np.exp(np.cumsum(r))
    o = np.r_[100.0, c[:-1]]
    h = np.maximum(o, c) * np.exp(np.abs(rng.normal(0, 0.002, n)))
    lo = np.minimum(o, c) * np.exp(-np.abs(rng.normal(0, 0.002, n)))
    f1, f4 = S.synth_frames(o, h, lo, c, datetime(2024, 1, 1, tzinfo=UTC))
    return f1, f4


def test_walk_matches_stress_walk_signal(frames):
    f1, f4 = frames
    path = S.Path.from_frame(f1, 60.0)
    sig = T.ensembles(T.zoo_signals(f4), f4, None)["zoo_ew"]
    b = T.make_bars(f4, f1, path, None)
    ref = S.walk_signal(sig, S._ms(f4["close_ts"]), f4["close"].to_numpy().astype(float), f4["sigma_risk_next"].to_numpy().astype(float), path)
    got = T.walk(sig, b, T.Spec("x"))
    assert len(ref) > 5
    assert len(got) == len(ref)
    for a, g in zip(ref, got, strict=True):
        for k in ("side", "entry_ms", "exit_ms", "why"):
            assert a[k] == g[k]
        for k in ("entry", "stop", "exit"):
            assert a[k] == pytest.approx(g[k])
    ref5 = S.walk_signal(sig, S._ms(f4["close_ts"]), f4["close"].to_numpy().astype(float), f4["sigma_risk_next"].to_numpy().astype(float), path, latency=5.0)
    got5 = T.walk(sig, b, T.Spec("x"), latency=5.0)
    assert [x["exit"] for x in ref5] == pytest.approx([x["exit"] for x in got5])


def test_variants_run_and_respect_rules(frames):
    f1, f4 = frames
    path = S.Path.from_frame(f1, 60.0)
    sig = T.ensembles(T.zoo_signals(f4), f4, None)["zoo_ew"]
    b = T.make_bars(f4, f1, path, None)
    sp = T.specs()
    for name in ("stop_pd", "stop_pw", "exit_chand", "exit_time30", "exit_flip", "time_no12", "time_delay1h", "filt_weekend", "order_taker"):
        tr = T.walk(sig, b, sp[name])
        assert tr, name
        for x in tr:
            k = abs(np.log(x["entry"] / x["stop"])) / 1.0
            assert k > 0
            if name == "time_no12":
                assert x["entry_hour"] != 12
            if name == "order_taker":
                assert x["kind"] == "taker"
    # structural stops stay inside the cap
    tr = T.walk(sig, b, sp["stop_pd"])
    cms = b.close_ms
    for x in tr:
        t = int(np.searchsorted(cms, x["entry_ms"]))
        dist = abs(np.log(x["entry"] / x["stop"])) / b.sigma[t]
        assert 0.99 <= dist <= 4.0 + 1e-9
    # chandelier: the recorded stop is the initial (risk) stop, never the trailed one
    for x in T.walk(sig, b, sp["exit_chand"]):
        t = int(np.searchsorted(cms, x["entry_ms"]))
        assert abs(np.log(x["entry"] / x["stop"])) / b.sigma[t] == pytest.approx(4.0, rel=1e-6)
    # time stop: no trade lasts longer than 180 4h bars + 1
    for x in T.walk(sig, b, sp["exit_time30"]):
        assert (x["exit_ms"] - x["entry_ms"]) <= (181 * 4 + 1) * 3_600_000


def test_signals_are_causal(frames):
    _, f4 = frames
    cut = f4.height - 300
    full = T.sign_signal(f4, 14)
    part = T.sign_signal(f4.head(cut), 14)
    assert np.array_equal(full[:cut], part)
    zf = T.ensembles(T.zoo_signals(f4), f4, None)
    zp = T.ensembles(T.zoo_signals(f4.head(cut)), f4.head(cut), None)
    for k in ("zoo_ew", "zoo_fam", "zoo_vote", "zoo_slow"):
        assert np.allclose(zf[k][:cut], zp[k], equal_nan=True)
    kf, kp = T.kalman_signal(f4), T.kalman_signal(f4.head(cut))
    assert np.allclose(kf[:cut], kp, equal_nan=True)
    for name in ("donch_14", "ema_4_16_64", "slope_30d", "cfg_ema_stack", "mac_7_30"):
        a = T.single_signal(name, f4, 1.2)
        b = T.single_signal(name, f4.head(cut), 1.2)
        assert np.allclose(a[:cut], b, equal_nan=True), name


def test_prior_extremes_point_in_time(frames):
    f1, f4 = frames
    cms = S._ms(f4["close_ts"])
    pdl, pdh, pwl, pwh = T.prior_extremes(f1, cms)
    i = 200
    t = datetime.fromtimestamp(cms[i] / 1000, UTC)
    day = t.date()
    prev = f1.filter(pl.col("event_ts").dt.date() == (day - __import__("datetime").timedelta(days=1)))
    assert pdl[i] == pytest.approx(prev["low"].min())
    assert pdh[i] == pytest.approx(prev["high"].max())
    assert prev["close_ts"].max() <= t


def test_paired_romano_wolf_and_spa():
    rng = np.random.default_rng(3)
    D = 600
    nb = rng.poisson(3, D).astype(float)
    sb = rng.normal(0, 1, D) * np.sqrt(nb) + 0.05 * nb
    good = (nb, sb + 0.3 * nb)
    noise = {f"n{k}": (nb, sb + rng.normal(0, 0.2, D) * np.sqrt(nb)) for k in range(5)}
    idx = T.boot_idx(D, 500)
    res = T.paired((nb, sb), {"good": good, **noise}, idx)
    assert res["good"]["p_rw"] < 0.05
    assert min(res[k]["p_rw"] for k in noise) > 0.01
    assert res["_spa"]["p"] < 0.05
    res0 = T.paired((nb, sb), noise, idx)
    assert res0["_spa"]["p"] > 0.05


def test_pbo_and_dsr():
    rng = np.random.default_rng(5)
    M = rng.normal(0, 1, (800, 10))
    p = T.pbo_cscv(M, 8)
    assert 0.2 < p["pbo"] < 0.8
    d = T.deflated_sharpe(rng.normal(0.05, 1, 2000), 50, 0.001)
    assert 0 <= d["dsr"] <= d["psr0"] <= 1


def test_dk_ols_recovers_slope():
    rng = np.random.default_rng(7)
    n = 5000
    x = rng.normal(size=n)
    y = 0.2 * x + rng.normal(size=n)
    day = np.repeat(np.arange(n // 10), 10)
    out = T.dk_ols(np.c_[np.ones(n), x], y, day)
    assert out["b"][1] == pytest.approx(0.2, abs=0.05)
    assert out["t"][1] > 5


def test_garch_and_sbb_paths(frames):
    f1, _ = frames
    rng = np.random.default_rng(11)
    cache: dict = {}
    p = T.synth_path(f1, "sbb", rng, cache)
    assert cache["b"] <= T.SBB_CAP
    assert np.all(p["high"] >= np.maximum(p["open"], p["close"]) - 1e-12)
    g = T.synth_path(f1, "garch", rng, cache)
    assert len(g["close"]) == len(p["close"])
    assert np.all(g["low"] <= np.minimum(g["open"], g["close"]) + 1e-12)
    st = T.stylised(np.diff(np.log(g["close"])))
    assert st["exkurt_1h"] > 0
