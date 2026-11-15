import math
from datetime import date

import numpy as np
import polars as pl
import pytest

from ai_trader.features.jump_diffusion import _hawkes_nll, hawkes_intensity
from ai_trader.research import vol_signal_study as vs


def _squeeze_series(seed: int = 0, up: bool = True):
    """Noisy walk, a 40-bar mean-reverting compression (tight closes, normal ranges), then a directional run."""
    rng = np.random.default_rng(seed)
    c1 = 100 * np.exp(np.cumsum(rng.normal(0, 0.02, 120)))
    c2 = c1[-1] * (1 + rng.normal(0, 0.0005, 40))
    c3 = c2[-1] * np.exp(np.cumsum(np.full(15, 0.01 if up else -0.01)))
    c = np.r_[c1, c2, c3]
    rng_w = np.r_[np.full(120, 0.015), np.full(40, 0.003), np.full(15, 0.006)]
    h, lo = c * (1 + rng_w), c * (1 - rng_w)
    return c, h, lo


def test_squeeze_detector_fires_after_compression_with_direction():
    for up in (True, False):
        c, h, lo = _squeeze_series(up=up)
        b = vs.squeeze_bands(c, h, lo)
        assert b["squeeze"][130:160].sum() >= 6  # compression detected as a squeeze
        idx, sgn = vs.squeeze_release_breakouts(c, b)
        late = idx >= 150
        assert late.any()
        assert sgn[late][0] == (1 if up else -1)


def test_squeeze_detector_no_lookahead():
    c, h, lo = _squeeze_series(seed=3)
    b_full = vs.squeeze_bands(c, h, lo)
    i_full, s_full = vs.squeeze_release_breakouts(c, b_full)
    p_full, _ = vs.plain_bb_breakouts(c, b_full)
    for k in (100, 150, 165, 170):
        b = vs.squeeze_bands(c[:k], h[:k], lo[:k])
        for key in ("bb_up", "bb_lo", "kc_up", "kc_lo"):
            np.testing.assert_allclose(b[key], b_full[key][:k], equal_nan=True)
        i, s = vs.squeeze_release_breakouts(c[:k], b)
        np.testing.assert_array_equal(i, i_full[i_full < k])
        np.testing.assert_array_equal(s, s_full[i_full < k])
        p, _ = vs.plain_bb_breakouts(c[:k], b)
        np.testing.assert_array_equal(p, p_full[p_full < k])


def test_plain_breakouts_exclude_recent_release_and_squeeze_bars():
    c, h, lo = _squeeze_series(seed=1)
    b = vs.squeeze_bands(c, h, lo)
    p, _ = vs.plain_bb_breakouts(c, b)
    assert not b["squeeze"][p].any()
    rel = [t for t in range(1, len(c)) if not b["squeeze"][t] and b["squeeze"][max(0, t - 6):t].all()]
    for t in p:
        assert all(not (0 <= t - r < vs.SQ_WINDOW) for r in rel)


def test_decluster_and_vol_surprise():
    np.testing.assert_array_equal(vs.decluster(np.array([0, 1, 2, 4, 5, 9]), 3), [True, False, False, True, False, True])
    f = np.full(20, 1e-4)
    rv = f.copy()
    ret = np.zeros(20)
    rv[[5, 6, 12]] = 4e-4
    ret[[5, 6, 12]] = [0.02, -0.03, 0.005]  # day 12 fails |ret| > sigma (0.01)
    np.testing.assert_array_equal(vs.vol_surprise_events(rv, ret, f), [5])


def test_hawkes_grid_nll_matches_reference():
    rng = np.random.default_rng(0)
    flags = (rng.random(800) < 0.05).astype(float)
    ev = np.where(flags > 0)[0].astype(float)
    for th in ([math.log(0.03), math.log(0.1), math.log(0.4)], [math.log(0.05), math.log(0.01), math.log(2.0)]):
        assert vs.hawkes_nll_grid(np.array(th), flags, ev) == pytest.approx(_hawkes_nll(th, ev, float(len(flags))), rel=1e-9)


def test_hawkes_fit_recovers_excitation_and_intensity_is_causal():
    rng = np.random.default_rng(2)
    mu, a, b = 0.02, 0.3, 0.5
    flags = np.zeros(20000)
    A = 0.0
    for t in range(len(flags)):
        lam = mu + a * A
        flags[t] = rng.random() < 1 - math.exp(-lam)
        A = math.exp(-b) * (A + flags[t])
    m, al, be, lr = vs.fit_hawkes_grid(flags)
    assert lr > 20 and 0.3 < al / be < 0.9
    lam_full = hawkes_intensity(flags.astype(bool), m, al, be)
    lam_cut = hawkes_intensity(flags[:5000].astype(bool), m, al, be)
    np.testing.assert_allclose(lam_cut, lam_full[:5000])


def test_hawkes_expected_count():
    assert vs.hawkes_expected_count(0.05, 0.05, 0.0, 1.0, 24) == pytest.approx(1.2)
    mu, a, b, lam0, H = 0.02, 0.2, 0.5, 0.3, 24.0
    # Euler integration of dE/ds = b (mu - E) + a E (E lambda of an exponential Hawkes)
    e, tot, ds = lam0, 0.0, 1e-4
    for _ in range(int(H / ds)):
        tot += e * ds
        e += (b * (mu - e) + a * e) * ds
    assert vs.hawkes_expected_count(lam0, mu, a, b, H) == pytest.approx(tot, rel=1e-3)


def test_t_exceed_prob_and_df_fit():
    from scipy import stats

    assert vs.t_exceed_prob(np.array([2.0]), 1e6)[0] == pytest.approx(2 * stats.norm.sf(2.0), rel=1e-4)
    rng = np.random.default_rng(0)
    nu = 4.0
    z = rng.standard_t(nu, 200_000) * math.sqrt((nu - 2) / nu)
    assert vs.fit_t_df(z) == pytest.approx(nu, rel=0.08)
    assert vs.poisson_logpmf(np.array([0.0]), np.array([1.0]))[0] == pytest.approx(-1.0)


def test_pooled_har_no_lookahead():
    rng = np.random.default_rng(0)
    n = 400
    dts = np.arange(np.datetime64("2026-01-01"), np.datetime64("2026-01-01") + np.timedelta64(n, "D"))
    series = {s: (dts, np.exp(rng.normal(-8, 0.5, n))) for s in ("A", "B")}
    f0 = vs.pooled_har_forecast(series, min_pairs=100)
    k = 250
    pert = {s: (d, np.r_[rv[:k], rv[k:] * 50]) for s, (d, rv) in series.items()}
    f1 = vs.pooled_har_forecast(pert, min_pairs=100)
    for s in series:
        assert np.isfinite(f0[s][200:k]).all()
        np.testing.assert_allclose(f0[s][:k], f1[s][:k], equal_nan=True)


def test_calendar_extra_weekdays():
    d = np.array(["2026-11-06", "2026-11-07", "2026-11-08", "2026-11-09"], dtype="datetime64[D]")  # Fri Sat Sun Mon
    e = vs.calendar_extra(d)
    np.testing.assert_array_equal(e[:, 0], [1, 1, 0, 0])  # target Sat, Sun
    np.testing.assert_array_equal(e[:, 1], [0, 0, 1, 0])  # target Mon


def test_held_leverage_matches_strategy_pnl():
    from ai_trader.research.evaluate import strategy_pnl

    rng = np.random.default_rng(0)
    n = 500
    frame = pl.DataFrame({"event_ts": np.arange(n), "sigma_next": np.abs(rng.normal(0.01, 0.002, n)) + 1e-3,
                          "fwd_ret": rng.normal(0, 0.01, n)})
    sig = pl.Series(np.clip(np.cumsum(rng.normal(0, 0.2, n)), -1, 1))
    ref = strategy_pnl(frame, sig, cost_bps=7.5)["pnl"].to_numpy()
    lev = vs.held_leverage(sig.to_numpy(), frame["sigma_next"].to_numpy())
    np.testing.assert_allclose(vs.scaled_pnl(lev, frame["fwd_ret"].to_numpy(), np.ones(n)), ref, atol=1e-12)


def test_realtime_c_uses_past_only_and_holm():
    rng = np.random.default_rng(0)
    base, raw = rng.normal(0, 1, 300), rng.normal(0, 3, 300)
    c = vs._realtime_c(base, raw, 50)
    raw2 = raw.copy()
    raw2[200:] *= 10
    c2 = vs._realtime_c(base, raw2, 50)
    np.testing.assert_allclose(c[:201], c2[:201], equal_nan=True)
    assert np.isnan(c[:50]).all() and c[250] == pytest.approx(1 / 3, rel=0.2)
    h = vs.holm({"a": 0.01, "b": 0.04, "c": 0.03})
    assert h == pytest.approx({"a": 0.03, "c": 0.06, "b": 0.06})


def test_date_mean_t_clusters_by_date():
    vals = np.r_[np.ones(30), -np.ones(30) * 0.5]
    dts = np.array([date(2026, 1, 1 + i % 30) for i in range(60)])
    r = vs.date_mean_t(vals, dts)
    assert r["n_dates"] == 30 and r["n_events"] == 60 and r["date_mean"] == pytest.approx(0.25)
