import numpy as np

from ai_trader.features.transforms import ewma_sigma_ahead
from ai_trader.features.vol_models import VolModel, horizon_variance, run


def test_plain_configuration_equals_ewma():
    r = np.random.default_rng(0).standard_t(4, 2000) * 0.01
    out = run(r, np.zeros(2000, bool), VolModel(clip=np.inf, use_weekend=False, warmup=30))
    ref = ewma_sigma_ahead(r, 0.94, 30)
    np.testing.assert_allclose(out["sigma_prev"][31:], ref[31:], rtol=1e-12)


def test_clipping_limits_jump_impact():
    r = np.full(500, 0.01)
    r[300] = 0.5
    plain = run(r, np.zeros(500, bool), VolModel(clip=np.inf, use_weekend=False))
    rob = run(r, np.zeros(500, bool), VolModel(clip=4.0, use_weekend=False))
    assert rob["sigma_prev"][301] < 0.3 * plain["sigma_prev"][301]


def test_weekend_multiplier_learns_quiet_weekends_causally():
    rng = np.random.default_rng(1)
    n = 24 * 7 * 40
    wk = (np.arange(n) % 168) >= 120          # last 48 h of each synthetic week
    r = rng.normal(0, 0.01, n) * np.where(wk, 0.4, 1.0)
    out = run(r, wk, VolModel(kappa=2.0))
    assert out["m"][-1] < 0.5, "learns ~0.4 from the data"
    k = n // 2
    part = run(r[:k], wk[:k], VolModel(kappa=2.0))
    np.testing.assert_allclose(part["sigma_prev"], out["sigma_prev"][:k])


def test_horizon_variance_uses_future_calendar():
    v = np.full(10, 1.0)
    m = np.full(10, 0.5)
    wk = np.array([0, 0, 1, 1, 0, 0, 0, 0, 0, 0], bool)
    f = horizon_variance(v, m, wk, 3)
    assert f[0] == (1 + 0.25 + 0.25)  # bars 1, 2, 3: one weekday, two weekend
    assert np.isnan(f[-1])


def test_har_is_causal_and_tracks_persistent_vol():
    from ai_trader.features.vol_models import har_forecast

    rng = np.random.default_rng(3)
    n = 600
    lv = np.zeros(n)
    for t in range(1, n):
        lv[t] = 0.97 * lv[t - 1] + rng.normal(0, 0.2)
    rv = np.exp(lv - 9) * rng.chisquare(4, n) / 4
    full = har_forecast(rv)
    part = har_forecast(rv[:400])
    np.testing.assert_allclose(part[:400], full[:400], equal_nan=True)
    ok = np.isfinite(full[:-1])
    assert np.corrcoef(np.log(full[:-1][ok]), np.log(rv[1:][ok]))[0, 1] > 0.5


def test_har_survives_missing_days():
    from ai_trader.features.vol_models import har_forecast

    rv = np.exp(np.random.default_rng(5).normal(-9, 0.5, 400))
    rv[0] = np.nan
    rv[250] = np.nan
    f = har_forecast(rv)
    assert np.isfinite(f[-1]) and np.isfinite(f[240])
