from datetime import UTC, date, datetime

import numpy as np
import polars as pl
import pytest

from ai_trader.research import wms_select as w


def test_calendar():
    assert w.friday_cutoff(date(2026, 11, 28)) == datetime(2026, 11, 27, 22, tzinfo=UTC)
    assert w.window_start(date(2026, 11, 28)) == datetime(2026, 11, 20, 22, tzinfo=UTC)
    with pytest.raises(ValueError):
        w.friday_cutoff(date(2026, 11, 27))
    assert w.thanksgiving(2026) == date(2026, 11, 26)
    assert w.thanksgiving(2021) == date(2021, 11, 25)
    assert w.holiday_flag(date(2026, 11, 28)) == "thanksgiving"
    assert w.holiday_flag(date(2025, 12, 27)) == "christmas"
    assert w.holiday_flag(date(2022, 1, 1)) == "new_year"
    assert w.holiday_flag(date(2026, 11, 21)) == ""


def test_variance_ratio_iid_and_ar():
    rng = np.random.default_rng(1)
    r = rng.standard_normal(20000)
    vr, z = w.variance_ratio(r, 4)
    assert abs(vr - 1) < 0.05 and abs(z) < 3
    e = rng.standard_normal(20001)
    ma = e[1:] - 0.5 * e[:-1]  # MA(1): rho1 = -0.5 / 1.25 = -0.4 -> VR(2) = 0.6
    vr2, z2 = w.variance_ratio(ma, 2)
    assert abs(vr2 - 0.6) < 0.03 and z2 < -10
    assert np.isnan(w.variance_ratio(r[:5], 4)[0])


def test_ar1_kendall():
    rng = np.random.default_rng(2)
    x = np.zeros(50000)
    e = rng.standard_normal(50000)
    for t in range(1, x.size):
        x[t] = -0.2 * x[t - 1] + e[t]
    assert abs(w.ar1(x) + 0.2) < 0.01


def test_ou_half_life():
    rng = np.random.default_rng(3)
    b = -0.05
    p = np.zeros(20000)
    for t in range(1, p.size):
        p[t] = (1 + b) * p[t - 1] + 0.01 * rng.standard_normal()
    hl = w.ou_half_life(p, 0.25)
    assert abs(hl - (-np.log(2) / np.log1p(b) * 0.25)) < 0.6
    assert w.ou_half_life(np.cumsum(np.ones(100)), 0.25) == w.HL_CAP_H  # explosive / trending -> cap


def test_dfa_and_efficiency():
    rng = np.random.default_rng(4)
    a = np.mean([w.dfa_alpha(rng.standard_normal(2016)) for _ in range(20)])
    assert 0.42 < a < 0.58
    assert w.efficiency(np.linspace(0, 1, 50)) == pytest.approx(1.0)
    assert w.efficiency(np.array([0, 1, 0, 1, 0.0])) == pytest.approx(0.0)


def test_bos_events_consumed_and_confirmed():
    # swing high at bar 3 (value 5), confirmed after bar 6; bar 7 closes above it -> one bull break; later closes above don't re-break
    h = np.array([1, 2, 3, 5, 3, 2, 1, 6, 7, 8.0])
    lo = h - 0.5
    c = h - 0.1
    ev = w.bos_events(h, lo, c, k=3)
    bull = [e for e in ev if e[1] > 0]
    assert bull == [(7, 1, 5.0)]
    # a close above the pivot before confirmation is not a break
    h2 = np.array([1, 2, 3, 5, 3, 2, 5.5, 1, 1, 1.0])
    assert [e for e in w.bos_events(h2, h2 - 0.5, h2 - 0.1, k=3) if e[1] > 0] == []


def test_indices_and_rule():
    rng = np.random.default_rng(5)
    n = w.FW_BARS
    c = np.cumsum(0.002 * rng.standard_normal(n))
    out = w.indices(c + 0.0005, c - 0.0005, c)
    for k in ("vr5_2", "vr15_4", "vr15_16", "ar1_15", "dfa", "er_1h", "bos_imb", "ou_hl"):
        assert np.isfinite(out[k])
    df = pl.DataFrame({"weekend_start": [date(2026, 1, 3)] * 6, "symbol": list("ABCDEF"),
                       "vr15_4": [0.7, 0.8, 0.9, 1.0, 1.1, 1.2], "vr15_16": [0.7, 0.8, 0.9, 1.0, 1.1, 1.2],
                       "ar1_15": [-0.2, -0.1, 0, 0.05, 0.1, 0.2], "dfa": [0.4, 0.45, 0.5, 0.55, 0.6, 0.65],
                       "er_1h": [0.01, 0.02, 0.03, 0.04, 0.05, 0.06], "bos_imb": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
                       "calm_p": [0.6] * 6})
    s = w.select(w.composite(df), gate=True, calm_p="calm_p")
    assert s.filter(pl.col("eligible"))["symbol"].to_list() == ["A", "B"]
    s2 = w.select(w.composite(df.with_columns(pl.lit(0.4).alias("calm_p"))), gate=True, calm_p="calm_p")
    assert not s2["eligible"].any() and s2["eligible_name"].sum() == 2


def test_blom_within_groups():
    df = pl.DataFrame({"g": [1, 1, 1, 2, 2, 2], "x": [1.0, 2.0, 3.0, 30.0, 10.0, None]})
    out = df.with_columns(w.blom(pl.col("x"), "g").alias("z"))["z"].to_list()
    assert out[0] < 0 < out[2] and abs(out[1]) < 1e-12
    assert out[3] > 0 > out[4] and out[5] is None or np.isnan(out[5])


def test_blom_nan_is_null():
    df = pl.DataFrame({"g": [1, 1, 1], "x": [1.0, float("nan"), 3.0]})
    z = df.with_columns(w.blom(pl.col("x"), "g").alias("z"))["z"].to_list()
    assert z[1] is None and z[0] < 0 < z[2]


def test_gate_unknown_closes():
    df = pl.DataFrame({"weekend_start": [date(2026, 1, 3)] * 3, "score": [1.0, 0.0, -1.0], "calm_p": [None, None, None]},
                      schema_overrides={"calm_p": pl.Float64})
    s = w.select(df, calm_p="calm_p", gate=True)
    assert s["eligible_name"].to_list() == [True, False, False] and s["eligible"].to_list() == [False, False, False]


def test_frozen_rule_constants_and_calm_probability():
    assert set(w.S1_COMPONENTS) == {"vr5_2", "vr5_12", "vr5_48", "vr15_16", "ar1_5", "ar1_15", "bos_count"}
    rule = {"coef": {"const": 0.0, **{c: 0.0 for c in w.CALM_X}}, "mu": {c: 0.0 for c in w.CALM_X}, "sd": {c: 1.0 for c in w.CALM_X}}
    x = {c: 1.0 for c in w.CALM_X}
    assert w.calm_probability(x, rule) == pytest.approx(0.5)
    rule["coef"]["x1_rvfw_vs_med"] = -1.0
    assert w.calm_probability(x, rule) < 0.5
    assert w.calm_probability({**x, "x3_dvol_vs_med": None}, rule) is None
