from datetime import UTC, date, datetime, timedelta

import numpy as np
import polars as pl

from ai_trader.features import seasonality as sz


def _hours(n, start=datetime(2024, 1, 1, tzinfo=UTC)):  # 2024-01-01 is a Monday
    return pl.Series("event_ts", [start + timedelta(hours=i) for i in range(n)], dtype=pl.Datetime("us", "UTC"))


def test_hour_of_week():
    ts = _hours(200)
    k = sz.hour_of_week(ts)
    assert k[0] == 0 and k[23] == 23 and k[24] == 24 and k[167] == 167 and k[168] == 0
    assert list(sz.crypto_session([0, 7, 8, 12, 13, 19, 20, 23])) == ["asia", "asia", "europe", "europe", "us", "us", "late", "late"]


def test_seasonal_factor_is_causal_and_recovers_profile():
    rng = np.random.default_rng(0)
    n = 168 * 40
    ts = _hours(n)
    k = sz.hour_of_week(ts)
    true = np.where((k % 24 >= 13) & (k % 24 < 20), 2.0, 0.75)
    u = true * rng.chisquare(1, n)
    f = sz.seasonal_factor(k, u, min_weeks=8)
    assert np.isnan(f[: 8 * 168]).all() and np.isfinite(f[8 * 168:]).all()
    # causality: changing u at t (and later) must not change the factor at t or before
    u2 = u.copy()
    t0 = 20 * 168 + 5
    u2[t0:] *= 100
    f2 = sz.seasonal_factor(k, u2, min_weeks=8)
    np.testing.assert_allclose(f[: t0 + 1], f2[: t0 + 1], equal_nan=True)
    assert not np.allclose(f[t0 + 1:], f2[t0 + 1:])
    # recovers the shape (normalised to a weekly mean of 1)
    last = f[-168:]
    shape = true / true.mean()
    assert np.corrcoef(last, shape[k[-168:]])[0, 1] > 0.9
    # profile_asof equals the causal factor's information set
    prof = sz.profile_asof(ts, u, ts[-1])
    np.testing.assert_allclose(prof[k[-1]], f[-1], rtol=1e-12)


def test_trailing_window_forgets():
    n = 168 * 30
    ts = _hours(n)
    k = sz.hour_of_week(ts)
    u = np.ones(n)
    u[: 168 * 15][k[: 168 * 15] == 10] = 10.0  # hour 10 loud only in the first half
    f_exp = sz.seasonal_factor(k, u, kappa=0.0, bandwidth=0)
    f_tr = sz.seasonal_factor(k, u, window_weeks=4, kappa=0.0, bandwidth=0)
    i = n - 168 + 10
    assert f_exp[i] > 2 and abs(f_tr[i] - 1.0) < 1e-9


def test_expiry_calendar():
    assert sz.deribit_expiry_kind(date(2026, 3, 27)) == "quarterly"
    assert sz.deribit_expiry_kind(date(2026, 10, 30)) == "monthly"
    assert sz.deribit_expiry_kind(date(2026, 10, 23)) == "weekly"
    assert sz.deribit_expiry_kind(date(2026, 10, 22)) is None
    assert sz.last_friday(2026, 12) == date(2026, 12, 25)
    cal = sz.expiry_calendar(date(2026, 1, 1), date(2027, 1, 1))
    assert cal.height == 52 and cal["is_monthly"].sum() == 12 and cal["is_quarterly"].sum() == 4
    assert (cal["ts"].dt.hour() == 8).all() and (cal["ts"].dt.weekday() == 5).all()


def test_funding_settlement_mapping():
    t0 = datetime(2025, 1, 1, tzinfo=UTC)
    ev = [t0 + timedelta(hours=8 * i) for i in range(6)]                       # 8h regime 00/08/16
    ev += [ev[-1] + timedelta(hours=4 * i) for i in range(1, 5)]               # switch to 4h
    df = pl.DataFrame({"symbol": "X", "event_ts": ev, "rate": 0.0001, "known_ts": [e + timedelta(seconds=2) for e in ev]},
                      schema_overrides={"event_ts": pl.Datetime("us", "UTC"), "known_ts": pl.Datetime("us", "UTC")})
    s = sz.funding_settlements(df)
    assert s.height == 5  # first print has no interval; the 4h prints drop out
    assert set(s["hour"].to_list()) <= {0, 8, 16} and (s["interval_min"] == 480).all()
    assert sz.funding_settlements(df, interval_min=240, hours=None).height == 4


def test_nyse_open_bar_dst():
    b = sz.nyse_open_bar([date(2026, 7, 6), date(2026, 12, 7), date(2026, 7, 3), date(2026, 7, 4)])
    assert b == [datetime(2026, 7, 6, 13, tzinfo=UTC), datetime(2026, 12, 7, 14, tzinfo=UTC)]
