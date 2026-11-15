from datetime import UTC, date, datetime, timedelta

import numpy as np
import polars as pl
import pytest

from ai_trader.research import options_compression as oc
from ai_trader.research.options_download import parse_instrument


def test_black76_put_call_parity_and_bounds():
    F, T = 50_000.0, 30 / 365
    for K in (30_000.0, 50_000.0, 70_000.0):
        for s in (0.2, 0.6, 1.2):
            c, p = oc.b76(F, K, T, s, "C"), oc.b76(F, K, T, s, "P")
            assert c - p == pytest.approx(F - K, abs=1e-6)
            assert max(F - K, 0) <= c <= F
            assert max(K - F, 0) <= p <= K
    # expiry / zero vol -> intrinsic
    assert oc.b76(F, 40_000, 0.0, 0.5, "C") == pytest.approx(10_000)
    assert oc.b76(F, 40_000, 0.1, 0.0, "P") == 0.0


def test_delta_signs_and_parity():
    F, K, T, s = 50_000.0, 52_000.0, 21 / 365, 0.5
    gc, gp = oc.b76_greeks(F, K, T, s, "C"), oc.b76_greeks(F, K, T, s, "P")
    assert 0 < gc["delta"] < 1 and -1 < gp["delta"] < 0
    assert gc["delta"] - gp["delta"] == pytest.approx(1.0, abs=1e-12)
    assert gc["vega"] == pytest.approx(gp["vega"]) and gc["vega"] > 0
    assert gc["theta"] < 0
    # deep ITM put: delta near -1, small vega relative to ATM
    deep = oc.b76_greeks(F, 70_000, T, s, "P")
    atm = oc.b76_greeks(F, F, T, s, "P")
    assert deep["delta"] < -0.9 and deep["vega"] < 0.3 * atm["vega"]
    # finite-difference check of delta and vega
    h = 1.0
    fd = (oc.b76(F + h, K, T, s, "C") - oc.b76(F - h, K, T, s, "C")) / (2 * h)
    assert gc["delta"] == pytest.approx(fd, rel=1e-5)
    fv = (oc.b76(F, K, T, s + 1e-4, "C") - oc.b76(F, K, T, s - 1e-4, "C")) / 2e-4
    assert gc["vega"] == pytest.approx(fv, rel=1e-4)


def test_implied_vol_roundtrip():
    F, T = 60_000.0, 45 / 365
    for K, cp in ((50_000.0, "P"), (60_000.0, "C"), (75_000.0, "C")):
        v = oc.b76(F, K, T, 0.55, cp)
        assert oc.implied_vol(v, F, K, T, cp) == pytest.approx(0.55, abs=1e-5)


def test_smile_through_pillars_and_extrapolation():
    atm, rr, bf, T = 0.5, -0.04, 0.02, 30 / 365
    sc, sp = atm + bf + rr / 2, atm + bf - rr / 2
    st = np.sqrt(T)
    kc = oc.Z25 * sc * st + 0.5 * sc * sc * T
    kp = -oc.Z25 * sp * st + 0.5 * sp * sp * T
    assert oc.smile_iv(0.0, T, atm, rr, bf) == pytest.approx(atm)
    assert oc.smile_iv(kc, T, atm, rr, bf) == pytest.approx(sc)
    assert oc.smile_iv(kp, T, atm, rr, bf) == pytest.approx(sp)
    # 25-delta check: the call pillar strike has Black-76 delta 0.25 at its own vol
    F = 100.0
    assert float(oc.b76_greeks(F, F * np.exp(kc), T, sc, "C")["delta"]) == pytest.approx(0.25, abs=1e-9)
    # continuity at the pillars and bounded far wings
    e = 1e-7
    assert oc.smile_iv(kc + e, T, atm, rr, bf) == pytest.approx(oc.smile_iv(kc - e, T, atm, rr, bf), abs=1e-5)
    far = oc.smile_iv(np.array([-3.0, 3.0]), T, atm, rr, bf)
    assert np.all(far >= 0.5 * atm) and np.all(far <= 3 * atm)


def test_pillar_interp_total_variance():
    atm = np.array([[0.4, 0.5, 0.6]])
    rr = np.array([[0.0, -0.02, -0.04]])
    bf = np.array([[0.01, 0.02, 0.03]])
    a, r, b = oc.pillar_interp(np.array([30.0]), atm, rr, bf)
    assert a[0] == pytest.approx(0.5) and r[0] == pytest.approx(-0.02)
    a, r, b = oc.pillar_interp(np.array([60.0]), atm, rr, bf)
    w = 0.5 ** 2 * 30 + 0.5 * (0.6 ** 2 * 90 - 0.5 ** 2 * 30)
    assert a[0] == pytest.approx(np.sqrt(w / 60))
    a, _, _ = oc.pillar_interp(np.array([2.0, 200.0]), np.repeat(atm, 2, 0), np.repeat(rr, 2, 0), np.repeat(bf, 2, 0))
    assert a[0] == pytest.approx(0.4) and a[1] == pytest.approx(0.6)


def test_cost_model_floor_and_spread():
    assert oc.option_cost_btc(0.0) == pytest.approx(0.000425)
    assert oc.option_cost_btc(0.0001) == pytest.approx(0.000425)
    assert oc.option_cost_btc(0.002) == pytest.approx(0.002)
    assert oc.option_cost_btc(float("nan")) == pytest.approx(0.000425)
    assert np.allclose(oc.option_cost_btc(np.array([0.0, 0.001])), [0.000425, 0.001])


def test_inverse_settlement_pnl():
    # long call bought at 0.05 BTC with S = 50k (USD value 2,500), sold at 0.04 BTC with S = 75k (USD value 3,000)
    S0, S1 = 50_000.0, 75_000.0
    V0, V1 = 0.05 * S0, 0.04 * S1
    r = oc.inverse_pnl(1.0, V0, S0, V1, S1)
    assert r["usd"] == pytest.approx(500.0)        # linear USD gain
    assert r["btc"] == pytest.approx(-0.01)        # but the BTC-settled P&L is a loss of 0.01 BTC
    assert r["btc_usd"] == pytest.approx(-750.0)
    # the gap is the BTC premium's own exposure: usd - btc*S1 = (V0/S0) * (S1 - S0)
    assert r["usd"] - r["btc_usd"] == pytest.approx(0.05 * (S1 - S0))
    # short side and costs
    s = oc.inverse_pnl(-1.0, V0, S0, V1, S1, cost_btc=0.001)
    assert s["btc"] == pytest.approx(0.01 - 0.001)


def test_parse_and_build_instrument_name():
    assert parse_instrument("BTC-29MAR24-15000-P") == (date(2024, 3, 29), 15000.0, "P")
    assert parse_instrument("BTC-5JAN24-42000-C") == (date(2024, 1, 5), 42000.0, "C")
    assert parse_instrument("BTC_USDC-5JAN24-42000-C") is None
    assert oc.inst_name(date(2024, 1, 5), 42000.0, "C") == "BTC-5JAN24-42000-C"


def _synthetic_inputs(n_days=900, seed=0):
    rng = np.random.default_rng(seed)
    d0 = date(2021, 1, 1)
    dates = [d0 + timedelta(days=i) for i in range(n_days)]
    rv = pl.DataFrame({"date": dates, "rv7": np.exp(rng.normal(0, 0.5, n_days)) * 0.01})
    iv = pl.DataFrame({"date": dates, "iv": 0.5 + 0.1 * rng.standard_normal(n_days)})
    return rv, iv


def test_detector_is_causal_and_declustered():
    rv, iv = _synthetic_inputs()
    base = oc.detector(rv, iv)
    cut = 700
    d_cut = rv["date"][cut]
    # perturb everything after the cut: rows up to the cut must not change
    rv2 = rv.with_columns(pl.when(pl.col("date") > d_cut).then(pl.col("rv7") * 50).otherwise(pl.col("rv7")).alias("rv7"))
    iv2 = iv.with_columns(pl.when(pl.col("date") > d_cut).then(pl.lit(0.01)).otherwise(pl.col("iv")).alias("iv"))
    alt = oc.detector(rv2, iv2)
    a = base.filter(pl.col("date") <= d_cut)
    b = alt.filter(pl.col("date") <= d_cut)
    for c in ("pct_rv", "pct_iv", "trigger", "event"):
        assert np.array_equal(np.nan_to_num(a[c].to_numpy().astype(float), nan=-1), np.nan_to_num(b[c].to_numpy().astype(float), nan=-1)), c
    # warm-up: no percentile before MIN_TRAIL values
    assert base["pct_rv"][: oc.MIN_TRAIL - 1].is_nan().all()
    # entries are >= 14 days apart and each event is a trigger day
    ev = base.filter(pl.col("event"))
    assert ev["trigger"].all()
    e = ev["entry_date"].to_list()
    assert all((b_ - a_).days >= oc.DECLUSTER_DAYS for a_, b_ in zip(e, e[1:]))


def test_rv7_window_ends_at_0800():
    t0 = datetime(2024, 1, 1, tzinfo=UTC)
    n = 24 * 12
    close = np.ones(n) * 100.0
    # one jump in the bar opening at 08:00 on Jan 10 (it closes at 09:00, after the Jan-10 08:00 cut)
    k = 24 * 9 + 8
    close[k:] = 110.0
    bars = pl.DataFrame({"event_ts": [t0 + timedelta(hours=i) for i in range(n)], "close": close})
    r = oc.rv7_at_0800(bars)
    v9 = r.filter(pl.col("date") == date(2024, 1, 10))["rv7"][0]
    v10 = r.filter(pl.col("date") == date(2024, 1, 11))["rv7"][0]
    assert v9 == pytest.approx(0.0)
    assert v10 == pytest.approx(np.log(1.1) ** 2)


def test_scale_out_fills_only_through():
    n = 10
    lc = np.log(np.array([100, 101, 103, 106, 108, 109, 110, 111, 112, 112.0]))
    fr = np.r_[np.diff(lc), 0.0]
    lev = np.r_[np.ones(8), 0.0, 0.0]
    lh = lc + 0.001
    ll = lc - 0.001
    thr = np.full(n, 0.02)
    r = oc.scale_out_trade(lev, fr, lc, lh, ll, 0, 7, thr, 0.0)
    assert r["gross0"] == pytest.approx(lc[8] - lc[0])
    assert r["n_trig"] == 3 and r["n_fill"] == 3
    # thirds exit at closes 2, 3, 4 (profit > 2% first at close 2)
    exp = (lc[2] - lc[0]) / 3 + (lc[3] - lc[0]) / 3 + (lc[4] - lc[0]) / 3
    assert r["gross1"] == pytest.approx(exp)
    # if the next bar never trades through (high <= close), nothing fills
    lh2 = np.r_[lc[:1], np.minimum(lc[1:], lc[:-1])]
    r2 = oc.scale_out_trade(lev, fr, lc, lh2, ll, 0, 7, thr, 0.0)
    assert r2["n_fill"] == 0 and r2["gross1"] == pytest.approx(r2["gross0"])


def test_boot_diff_and_cluster_t():
    rng = np.random.default_rng(0)
    vals = rng.normal(0, 1, 500)
    ev = np.zeros(500, bool)
    ev[::25] = True
    vals[ev] += 2.0
    wk = np.arange(500) // 7
    r = oc.boot_diff_cond_uncond(vals, ev, wk, B=500)
    assert r["diff"] > 1.5 and r["ci95"][0] > 0
    x = rng.normal(0.5, 1, 200)
    assert oc.cluster_t(x, np.arange(200) // 5) > 2
