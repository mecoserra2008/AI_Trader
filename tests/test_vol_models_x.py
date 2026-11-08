import numpy as np

from ai_trader.features.vol_models import har_forecast
from ai_trader.features.vol_models_x import forward_sum, har_forecast_x, iv_to_daily_var, qlike, trailing_mean


def _rv(n=500, seed=0):
    rng = np.random.default_rng(seed)
    lv = np.zeros(n)
    for t in range(1, n):
        lv[t] = 0.95 * lv[t - 1] + 0.3 * rng.standard_normal()
    return np.exp(lv - 8) * rng.chisquare(24, n) / 24


def test_reduces_to_har_forecast():
    rv = _rv()
    np.testing.assert_allclose(har_forecast_x(rv), har_forecast(rv), rtol=1e-9, equal_nan=True)


def test_causal_no_lookahead():
    rv = _rv()
    ex = np.log(_rv(seed=1))
    for h in (1, 5):
        base = har_forecast_x(rv, ex, horizon=h)
        rv2, ex2 = rv.copy(), ex.copy()
        rv2[400:] *= 50.0
        ex2[400:] += 3.0
        alt = har_forecast_x(rv2, ex2, horizon=h)
        np.testing.assert_array_equal(base[:400], alt[:400])


def test_extra_regressor_is_used_when_informative():
    rng = np.random.default_rng(3)
    n = 900
    signal = rng.standard_normal(n)
    rv = np.exp(-8 + 0.8 * np.r_[0.0, signal[:-1]] + 0.1 * rng.standard_normal(n))
    f0 = har_forecast_x(rv)
    f1 = har_forecast_x(rv, signal)
    ok = np.isfinite(f0) & np.isfinite(f1)
    nxt = np.r_[rv[1:], np.nan]
    ok &= np.isfinite(nxt)
    assert np.nanmean(qlike(nxt[ok], f1[ok])) < np.nanmean(qlike(nxt[ok], f0[ok]))


def test_nan_extra_blanks_rows_only():
    rv = _rv()
    ex = np.log(_rv(seed=2))
    ex[250] = np.nan
    f = har_forecast_x(rv, ex)
    assert np.isnan(f[250]) and np.isfinite(f[251]) and np.isfinite(f[300])


def test_helpers():
    x = np.arange(10, dtype=float)
    assert trailing_mean(x, 3)[2] == 1.0 and np.isnan(trailing_mean(x, 3)[1])
    assert forward_sum(x, 2)[0] == 3.0 and np.isnan(forward_sum(x, 2)[8])
    assert qlike(np.array([2.0]), np.array([2.0]))[0] == 0.0
    np.testing.assert_allclose(iv_to_daily_var(np.array([60.0]), True), iv_to_daily_var(np.array([0.6]), False))
    np.testing.assert_allclose(iv_to_daily_var(np.array([0.6]), False)[0], 0.36 / 365)
