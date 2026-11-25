"""Pure functions of ai_trader.research.macro_models (MACRO block, H-MAC3..H-MAC9)."""

from datetime import UTC, date, datetime

import numpy as np
import pytest

from ai_trader.research import macro_models as mm


def _vintages(rows):
    """rows: (obs date, value, known_ts datetime, pseudo)."""
    v = object.__new__(mm.Vintages)
    obs = np.array([r[0] for r in rows], dtype="datetime64[D]")
    kt = np.array([int(r[2].timestamp()) for r in rows])
    val = np.array([r[1] for r in rows], float)
    ps = np.array([r[3] for r in rows], bool)
    o = np.lexsort((kt, obs))
    v._s = {"X": {"obs": obs[o], "kt": kt[o], "val": val[o], "pseudo": ps[o], "first_vintage_ts": np.array([kt[~ps].min()])}}
    return v


def test_asof_point_in_time_and_latest_revision():
    t1, t2, t3 = (datetime(2020, 2, 15, tzinfo=UTC), datetime(2020, 3, 15, tzinfo=UTC), datetime(2020, 4, 15, tzinfo=UTC))
    V = _vintages([(date(2020, 1, 1), 100.0, t1, False), (date(2020, 1, 1), 101.0, t3, False), (date(2020, 2, 1), 102.0, t2, False)])
    o, v, _ = V.asof("X", datetime(2020, 2, 20, tzinfo=UTC))
    assert list(o) == [date(2020, 1, 1)] and list(v) == [100.0]          # February not known yet
    o, v, _ = V.asof("X", datetime(2020, 3, 20, tzinfo=UTC))
    assert list(v) == [100.0, 102.0]                                     # revision of January not known yet
    o, v, _ = V.asof("X", datetime(2020, 5, 1, tzinfo=UTC))
    assert list(v) == [101.0, 102.0]
    fr = V.first_release("X")
    assert fr[date(2020, 1, 1)][0] == 100.0


def test_fill_gaps_and_logdiff():
    m = {0: 100.0, 2: 121.0}
    f = mm.fill_gaps(m)
    assert f[1] == pytest.approx(110.0)
    d = mm.logdiff(f)
    assert d[1] == pytest.approx(d[2]) == pytest.approx(100 * np.log(1.1))
    assert mm.fill_gaps({0: 4.0, 2: 5.0}, log=False)[1] == pytest.approx(4.5)


def test_month_index_roundtrip():
    for d in (date(1999, 1, 1), date(2026, 12, 1)):
        assert mm.index_month(mm.month_index(d)) == d


def test_ar_recovers_coefficients_and_forecasts():
    rng = np.random.default_rng(0)
    y = np.zeros(4000)
    for t in range(1, len(y)):
        y[t] = 0.1 + 0.6 * y[t - 1] + rng.normal(0, 0.1)
    b, s = mm.ar_fit(y, 1)
    assert b[1] == pytest.approx(0.6, abs=0.04) and s == pytest.approx(0.1, abs=0.01)
    f = mm.ar_forecast(np.array([1.0]), np.array([0.1, 0.5]), 2)
    assert f.tolist() == pytest.approx([0.6, 0.4])


def test_var_minnesota_ols_limit_and_shrinkage():
    rng = np.random.default_rng(1)
    Y = rng.normal(size=(300, 2))
    for t in range(1, 300):
        Y[t] += 0.5 * Y[t - 1]
    B_ols = mm.var_minnesota_fit(Y, 1, None)
    B_big = mm.var_minnesota_fit(Y, 1, 1e6)
    B_tight = mm.var_minnesota_fit(Y, 1, 1e-6)
    assert np.allclose(B_ols, B_big, atol=1e-3)
    assert np.all(np.abs(B_tight[1:]) < 1e-3)                          # lag coefficients shrunk to the prior mean 0
    f = mm.var_forecast(Y, B_ols, 1, 3)
    assert f.shape == (3, 2)


def test_dm_test_direction():
    rng = np.random.default_rng(2)
    e_b = rng.normal(0, 1.0, 300)
    e_m = rng.normal(0, 0.7, 300)
    assert mm.dm_test(e_m, e_b, 1)["p_one_sided"] < 0.01
    assert mm.dm_test(e_b, e_m, 1)["p_one_sided"] > 0.99
    assert abs(mm.dm_test(e_b, e_b, 2)["dm"]) < 1e-6


def test_t_fit_and_intervals():
    rng = np.random.default_rng(3)
    e = rng.standard_t(6, 5000) * 0.2
    df, sc = mm.t_fit(e)
    assert sc == pytest.approx(0.2, rel=0.1)
    lo, hi = mm.t_interval(1.0, df, sc, 0.8)
    assert np.mean((e + 1.0 >= lo) & (e + 1.0 <= hi)) == pytest.approx(0.8, abs=0.03)
    c = mm.calib(e, df, sc)
    assert c["cov80"] == pytest.approx(0.8, abs=0.03) and sum(c["pit_hist10"]) == 5000


def test_print_probs_sum_and_center():
    p = mm.print_probs(0.3, 30, 0.05, [0.1, 0.2, 0.3, 0.4, 0.5])
    assert sum(p.values()) == pytest.approx(1.0, abs=0.01)
    assert max(p, key=p.get) == "0.3"


def test_scores():
    P = np.array([[0.0, 1.0, 0.0], [0.5, 0.5, 0.0]])
    assert mm.brier_multi(P, np.array([1, 0])).tolist() == pytest.approx([0.0, 0.5])
    assert mm.auc(np.array([0.9, 0.8, 0.1]), np.array([1, 1, 0])) == 1.0
    assert mm.auc(np.array([0.1, 0.9]), np.array([1, 0])) == 0.0
    m, lo, hi = mm.block_bootstrap_mean(np.ones(100), 10)
    assert m == lo == hi == 1.0


def test_probit_recovers():
    rng = np.random.default_rng(4)
    x = rng.normal(size=5000)
    y = (0.3 - 0.8 * x + rng.normal(size=5000) > 0).astype(float)
    b = mm.probit_fit(np.column_stack([np.ones(5000), x]), y)
    assert b == pytest.approx([0.3, -0.8], abs=0.08)


def test_ordered_logit_recovers_and_classes():
    rng = np.random.default_rng(5)
    x = rng.normal(size=6000)
    z = 1.5 * x + rng.logistic(size=6000)
    y = np.where(z < -2, 0, np.where(z < 2, 1, 2))
    th = mm.ologit_fit(x[:, None], y, ridge=0.0)
    assert th[0] == pytest.approx(1.5, abs=0.12) and th[1] == pytest.approx(-2.0, abs=0.15)
    P = mm.ologit_predict(th, np.array([[0.0], [3.0]]))
    assert np.allclose(P.sum(1), 1.0) and P[1, 2] > P[0, 2]
    assert [mm.move_class(d) for d in (-0.25, 0.0, 0.25, 0.05)] == [0, 1, 2, 1]


def test_month_end_series():
    daily = [(date(2020, 1, 5), 1.0), (date(2020, 1, 31), 1.25), (date(2020, 3, 10), 1.5)]
    s = mm.month_end_series(daily, mm.month_index(date(2020, 1, 1)), mm.month_index(date(2020, 3, 1)))
    assert list(s.values()) == [1.25, 1.25, 1.5]


def test_nw_ols_slope():
    rng = np.random.default_rng(6)
    x = rng.normal(size=2000)
    y = -0.5 * x + rng.normal(size=2000)
    r = mm.nw_ols(y, x, 3)
    assert r["slope"] == pytest.approx(-0.5, abs=0.07) and r["t"] < -10
