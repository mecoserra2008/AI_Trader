from datetime import UTC, date, datetime, timedelta

import math

import numpy as np
import polars as pl
import pytest

from ai_trader.research import tradfi_session_study as ts


def u(y, m, d, h, mi=0):
    return datetime(y, m, d, h, mi, tzinfo=UTC)


# ---------------------------------------------------------------- calendar


def test_dst_transitions_shift_the_session_in_utc():
    # US DST 2026: starts Sun Mar 8, ends Sun Nov 1.
    assert ts.nyse_session(date(2026, 3, 6)) == (u(2026, 3, 6, 14, 30), u(2026, 3, 6, 21))   # EST
    assert ts.nyse_session(date(2026, 3, 9)) == (u(2026, 3, 9, 13, 30), u(2026, 3, 9, 20))   # EDT
    assert ts.nyse_session(date(2026, 10, 30)) == (u(2026, 10, 30, 13, 30), u(2026, 10, 30, 20))
    assert ts.nyse_session(date(2026, 11, 2)) == (u(2026, 11, 2, 14, 30), u(2026, 11, 2, 21))


def test_holidays_weekends_and_early_closes():
    for d in (date(2026, 4, 3), date(2026, 6, 19), date(2026, 7, 3), date(2026, 9, 7), date(2026, 11, 26), date(2026, 12, 25)):
        assert ts.nyse_session(d) is None, d
    assert ts.nyse_session(date(2026, 11, 7)) is None  # Saturday
    assert ts.nyse_session(date(2026, 11, 27)) == (u(2026, 11, 27, 14, 30), u(2026, 11, 27, 18))  # 13:00 EST
    assert ts.nyse_session(date(2026, 12, 24))[1] == u(2026, 12, 24, 18)
    with pytest.raises(ValueError):
        ts.nyse_session(date(2027, 1, 4))


def test_cme_sunday_reopen_dst():
    assert ts.cme_sunday_reopen(date(2026, 3, 1)) == u(2026, 3, 1, 23)
    assert ts.cme_sunday_reopen(date(2026, 3, 8)) == u(2026, 3, 8, 22)  # DST starts that morning
    assert ts.cme_sunday_reopen(date(2026, 10, 25)) == u(2026, 10, 25, 22)
    assert ts.cme_sunday_reopen(date(2026, 11, 1)) == u(2026, 11, 1, 23)  # DST ended that morning
    with pytest.raises(ValueError):
        ts.cme_sunday_reopen(date(2026, 11, 2))


def test_first_hour_is_the_bar_containing_the_open():
    assert ts.first_cash_bar(u(2026, 6, 1, 13, 30)) == u(2026, 6, 1, 13)  # 09:00 ET bar (EDT)
    assert ts.first_cash_bar(u(2026, 11, 2, 14, 30)) == u(2026, 11, 2, 14)  # 09:00 ET bar (EST)
    assert ts.bar_containing(u(2026, 6, 1, 14)) == u(2026, 6, 1, 14)


def test_sessions_carry_previous_close_across_weekend_holiday_and_early_close():
    s = {x["date"]: x for x in ts.nyse_sessions(date(2026, 11, 23), date(2026, 11, 30))}
    assert date(2026, 11, 26) not in s
    fri = s[date(2026, 11, 27)]
    assert fri["prev_close_utc"] == u(2026, 11, 25, 21) and fri["n_bars"] == 4 and fri["early_close"]
    mon = s[date(2026, 11, 30)]
    assert mon["prev_close_utc"] == u(2026, 11, 27, 18) and mon["n_bars"] == 7 and mon["after_closure"]
    assert mon["fb_start"] == u(2026, 11, 30, 14) and mon["gap_hours"] == 68
    tue = s[date(2026, 11, 24)]
    assert not tue["after_closure"] and tue["gap_hours"] == 17
    # DST weekend: Fri Oct 30 close 20:00 UTC, Mon Nov 2 first bar 14:00 UTC
    m2 = [x for x in ts.nyse_sessions(date(2026, 11, 2), date(2026, 11, 2))][0]
    assert m2["prev_close_utc"] == u(2026, 10, 30, 20) and m2["fb_start"] == u(2026, 11, 2, 14)


# ---------------------------------------------------------------- gaps and session rows


def _bars(start: datetime, closes: list[float], hi_pad=0.0, lo_pad=0.0, overrides=None) -> pl.DataFrame:
    rows, prev = [], closes[0]
    for i, c in enumerate(closes):
        t = start + timedelta(hours=i)
        o = prev
        h, lo = max(o, c) + hi_pad, min(o, c) - lo_pad
        if overrides and t in overrides:
            h, lo = overrides[t]
        rows.append({"event_ts": t, "open": o, "high": h, "low": lo, "close": c})
        prev = c
    return pl.DataFrame(rows)


def test_gap_and_fill_on_synthetic_day():
    # Tue 2026-06-02 (EDT): prev close Mon 20:00 UTC, first cash bar 13:00 UTC, close 20:00 UTC.
    start = u(2026, 6, 1, 0)
    n = 48
    closes = [100.0] * n
    # Mon close (bar 19:00 UTC ending 20:00) = 100. Overnight drift to 102 by Tue 13:00 UTC.
    for i in range(20, 37):
        closes[i] = 100.0 + 2.0 * (i - 19) / 17  # bar starting at hour i; bar 36 (Tue 12:00) ends 13:00 at 102
    for i in range(37, 44):  # Tue cash bars 13:00..19:00 UTC
        closes[i] = 101.0
    sess_low_bar = u(2026, 6, 2, 15)
    bars = _bars(start, closes, overrides={sess_low_bar: (101.5, 99.9)})
    g = ts.Grid.from_bars(bars)
    sess = ts.nyse_sessions(date(2026, 6, 2), date(2026, 6, 2))
    tab = ts.session_table(g, sess)
    r = tab.row(0, named=True)
    assert r["p_prev"] == pytest.approx(100.0)
    assert r["p_open"] == pytest.approx(102.0)
    assert r["on"] == pytest.approx(math.log(1.02))
    assert r["intraday"] == pytest.approx(math.log(101 / 102))
    assert r["n_bars"] == 7 and r["fb_start"] == u(2026, 6, 2, 13)
    assert ts.gap_fill(r)  # the 15:00 UTC bar's low 99.9 touched the previous close
    r2 = dict(r, sess_low=100.5)
    assert not ts.gap_fill(r2)
    # gap down mirror
    r3 = dict(r, p_open=98.0, sess_high=99.0)
    assert not ts.gap_fill(r3)
    assert ts.gap_fill(dict(r3, sess_high=100.0))


def test_missing_bar_drops_the_session_and_trailing_sigma_is_causal():
    start = u(2026, 6, 1, 0)
    closes = list(100 * np.exp(np.cumsum(np.random.default_rng(0).normal(0, 0.002, 24 * 40))))
    bars = _bars(start, closes)
    g = ts.Grid.from_bars(bars)
    sess = ts.nyse_sessions(date(2026, 6, 2), date(2026, 7, 8))
    tab = ts.session_table(g, sess)
    # sigma_on of row i uses rows < i only
    on = tab["on"].to_numpy()
    i = 15
    assert tab["sigma_on"][i] == pytest.approx(np.sqrt(np.mean(on[max(0, i - 20):i] ** 2)))
    assert tab["sigma_on"][: ts.TRAIL_MIN].is_null().all()
    # remove a cash bar of 2026-06-10
    k = u(2026, 6, 10, 15)
    g2 = ts.Grid.from_bars(bars.filter(pl.col("event_ts") != k))
    tab2 = ts.session_table(g2, sess)
    assert date(2026, 6, 10) not in tab2["date"].to_list()
    assert tab2.height == tab.height - 1


def test_orb_trade_first_breakout_held_to_close():
    row = {"px": [100, 101, 103, 102, 104, 104, 104, 105], "n_bars": 7, "or_high": 102.0, "or_low": 99.0}
    tr = ts.orb_trade(row)
    assert tr["side"] == 1 and tr["entry_k"] == 2
    assert tr["gross"] == pytest.approx(math.log(105 / 103))
    assert ts.orb_trade({"px": [100, 101, 101, 101, 101, 101, 101, 110], "n_bars": 7, "or_high": 102.0, "or_low": 99.0}) is None


def test_holm_and_cluster_ols():
    adj = ts.holm({"a": 0.01, "b": 0.04, "c": 0.5})
    assert adj["a"] == pytest.approx(0.03) and adj["b"] == pytest.approx(0.08) and adj["c"] == pytest.approx(0.5)
    rng = np.random.default_rng(1)
    x = rng.normal(size=500)
    y = 0.5 * x + rng.normal(size=500)
    r = ts.cluster_ols(x, y, np.arange(500) // 5)
    assert abs(r["b"] - 0.5) < 0.15 and r["t_b"] > 5
