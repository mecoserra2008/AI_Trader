"""Phase 1: missing 5m intervals and their classes."""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta

import polars as pl

from vecopen import gaps as G
from vecopen.data import TS

FIVE = timedelta(minutes=5)
T0 = datetime(2024, 1, 1, tzinfo=UTC)


def series(start: datetime, end: datetime, drop: list[tuple[datetime, datetime]] = ()) -> pl.Series:
    s = pl.datetime_range(start, end, "5m", eager=True, time_unit="ms", time_zone="UTC")
    for a, b in drop:
        s = s.filter((s < a) | (s > b))
    return s.cast(TS)


def test_missing_runs_and_clip():
    s = series(T0, T0 + timedelta(days=1), [(T0 + timedelta(hours=1), T0 + timedelta(hours=2)),
                                            (T0 + timedelta(hours=5), T0 + timedelta(hours=5))])
    r = G.missing_runs(s)
    assert r["n_missing"].to_list() == [13, 1]
    assert r["gap_first"][0] == T0 + timedelta(hours=1) and r["gap_last"][0] == T0 + timedelta(hours=2)
    c = G.clip_runs(r, T0 + timedelta(hours=1, minutes=30), None)
    assert c["n_missing"].to_list() == [7, 1]
    assert G.clip_runs(r, None, T0 + timedelta(hours=1, minutes=10))["n_missing"].to_list() == [3]
    assert G.missing_runs(s.head(1)).height == 0


def test_crypto_venue_wide_vs_single_symbol():
    out_a, out_b = (T0 + timedelta(hours=3), T0 + timedelta(hours=4)), (T0 + timedelta(hours=10), T0 + timedelta(hours=10, minutes=20))
    end = T0 + timedelta(days=2)
    w = {}
    for k in "ABCDE":
        drops = [out_a] + ([out_b] if k == "A" else [])
        s = series(T0, end, drops)
        w[k] = G.Window(s[0], s[-1], G.missing_runs(s))
    cls = G.classify_crypto(w)
    a = cls["A"]
    assert a["class"].to_list() == ["exchange_outage", "archive_gap"]
    assert a["peers"].to_list() == [4, 4] and a["peers_missing"].to_list() == [4, 0]
    # too few peers listed: never called venue-wide
    few = G.classify_crypto({k: w[k] for k in "AB"})
    assert few["A"]["class"][0] == "archive_gap"
    # a peer listed after the run does not count
    late = series(T0 + timedelta(hours=6), end)
    w2 = dict(w, E=G.Window(late[0], late[-1], G.missing_runs(late)))
    assert G.classify_crypto(w2)["A"]["peers"][0] == 3


def test_fx_weekend_daily_break_holiday_and_feed_gap():
    hours = {"weekly_open": "sun 17:00", "weekly_close": "fri 17:00", "daily_break": None}
    fx = G.WeeklySchedule.from_hours("America/New_York", hours)
    # Fri 2024-01-05 17:00 NY = 22:00 UTC .. Sun 2024-01-07 17:00 NY = 22:00 UTC
    wk = (datetime(2024, 1, 5, 22, 5, tzinfo=UTC), datetime(2024, 1, 7, 22, 0, tzinfo=UTC))
    feed = (datetime(2024, 1, 3, 10, 5, tzinfo=UTC), datetime(2024, 1, 3, 11, 0, tzinfo=UTC))
    s = series(datetime(2024, 1, 2, tzinfo=UTC), datetime(2024, 1, 9, tzinfo=UTC), [wk, feed])
    c = G.classify_cfd(G.missing_runs(s), fx)
    assert c["class"].to_list() == ["feed_gap", "weekend"]
    assert c["in_schedule"].to_list() == [12, 0]
    hol = [G.Holiday(date(2024, 1, 3), False, 10 * 3600, 12 * 3600, "UTC")]
    assert G.classify_cfd(G.missing_runs(s), fx, hol)["class"].to_list() == ["holiday", "weekend"]
    rec = [G.Holiday(date(2019, 1, 3), True, None, None, "UTC")]
    assert G.classify_cfd(G.missing_runs(s), fx, rec)["class"][0] == "holiday"
    # CME-style daily break 17:00-18:00 NY
    cme = G.WeeklySchedule.from_hours("America/New_York", {"weekly_open": "sun 18:00", "weekly_close": "fri 17:00",
                                                           "daily_break": ["17:00", "18:00"]})
    brk = (datetime(2024, 1, 3, 22, 5, tzinfo=UTC), datetime(2024, 1, 3, 23, 0, tzinfo=UTC))
    s2 = series(datetime(2024, 1, 2, tzinfo=UTC), datetime(2024, 1, 4, tzinfo=UTC), [brk])
    assert G.classify_cfd(G.missing_runs(s2), cme)["class"].to_list() == ["daily_break"]


def test_ctrader_intervals_and_summary():
    # Mon 00:00 - Fri 22:00 in UTC, as cTrader seconds from Sunday 00:00
    sched = G.WeeklySchedule.from_intervals("UTC", [(86_400, 5 * 86_400 + 22 * 3600)])
    ts = pl.Series([datetime(2024, 1, 1, 0, 0, tzinfo=UTC), datetime(2024, 1, 5, 21, 55, tzinfo=UTC),
                    datetime(2024, 1, 5, 22, 0, tzinfo=UTC), datetime(2024, 1, 7, 12, 0, tzinfo=UTC)]).cast(TS)
    assert sched.is_open(ts).tolist() == [True, True, False, False]
    always = G.WeeklySchedule.from_hours("UTC", {"open": "always"})
    assert always.open_slots.all()
    df = pl.DataFrame({"class": ["a", "a", "b"], "n_missing": [1, 2, 5]})
    assert G.summarise(df) == {"a": {"runs": 2, "bars": 3}, "b": {"runs": 1, "bars": 5}}


def test_zero_volume_runs_and_flat_classes():
    s = series(T0, T0 + timedelta(days=1))
    windows = {}
    for k in "ABCD":
        zero = s.is_between(T0 + timedelta(hours=2), T0 + timedelta(hours=2, minutes=20))
        if k == "A":
            zero = zero | (s == T0 + timedelta(hours=9))
        df = pl.DataFrame({"ts_close": s, "v": pl.Series([0.0 if z else 1.0 for z in zero])})
        windows[k] = G.Window(s[0], s[-1], G.zero_volume_runs(df))
    assert windows["A"].runs["n_missing"].to_list() == [5, 1]
    c = G.classify_flat(windows)
    assert c["A"]["class"].to_list() == ["venue_flat", "symbol_flat"]
    assert c["B"]["class"].to_list() == ["venue_flat"]
