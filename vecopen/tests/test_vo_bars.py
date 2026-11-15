"""Phase 1: sessions and resample(). The property test checks resample() against an independent oracle that
buckets every base bar with plain datetime + zoneinfo arithmetic and aggregates in Python."""

from __future__ import annotations

from collections import OrderedDict
from datetime import UTC, date, datetime, time, timedelta
from zoneinfo import ZoneInfo

import polars as pl
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from vecopen.bars import TIMEFRAMES, UTC_24H, Session, check_base, load_sessions, resample

from vecopen.config import load_config

FIVE = timedelta(minutes=5)
SESS = load_sessions()
SESSIONS = [SESS.conventions[k] for k in ("utc_24h", "fx_ny17", "cme_ny18", "ice_brent_ldn23", "eurex_cet00")]


# ---------------------------------------------------------------- oracle

def _day_start(d: date, s: Session) -> datetime:
    h, m = s.day_start_hm
    local = datetime.combine(d, time(h, m)) - (timedelta(days=1) if s.shift_minutes else timedelta(0))
    return local.replace(tzinfo=ZoneInfo(s.tz)).astimezone(UTC)


def _trade_date(ts_open: datetime, s: Session) -> date:
    local = ts_open.astimezone(ZoneInfo(s.tz)).replace(tzinfo=None)
    return (local + timedelta(minutes=s.shift_minutes)).date()


def oracle_bucket(ts_close: datetime, tf: str, s: Session) -> tuple[datetime, datetime]:
    t = ts_close - FIVE
    td = _trade_date(t, s)
    if tf == "1d":
        return _day_start(td, s), _day_start(td + timedelta(days=1), s)
    if tf == "1w":
        mon = td - timedelta(days=td.weekday())
        return _day_start(mon, s), _day_start(mon + timedelta(days=7), s)
    if tf == "1M":
        first = td.replace(day=1)
        nxt = (first + timedelta(days=32)).replace(day=1)
        return _day_start(first, s), _day_start(nxt, s)
    minutes = {"5m": 5, "15m": 15, "30m": 30, "1h": 60, "2h": 120, "4h": 240}[tf]
    ds, de = _day_start(td, s), _day_start(td + timedelta(days=1), s)
    k = (t - ds) // timedelta(minutes=minutes)
    start = ds + k * timedelta(minutes=minutes)
    return start, min(start + timedelta(minutes=minutes), de)


def oracle_resample(rows: list[dict], tf: str, s: Session, complete_only: bool = True) -> list[dict]:
    buckets: OrderedDict[tuple[datetime, datetime], list[dict]] = OrderedDict()
    for r in rows:
        buckets.setdefault(oracle_bucket(r["ts_close"], tf, s), []).append(r)
    last = rows[-1]["ts_close"]
    out = []
    for (a, b), rs in buckets.items():
        if complete_only and b > last:
            continue
        out.append({"ts_open": a, "ts_close": b, "o": rs[0]["o"], "h": max(x["h"] for x in rs),
                    "l": min(x["l"] for x in rs), "c": rs[-1]["c"], "v": sum(x["v"] for x in rs),
                    "buy_v": sum(x["buy_v"] for x in rs), "sell_v": sum(x["sell_v"] for x in rs), "n_base": len(rs)})
    return out


# ---------------------------------------------------------------- generators

# starts near DST changes (US: Mar/Nov, EU: Mar/Oct), month and year ends, and anywhere
ANCHORS = [datetime(2024, 3, 9, 12, tzinfo=UTC), datetime(2024, 3, 30, 12, tzinfo=UTC),
           datetime(2024, 10, 26, 12, tzinfo=UTC), datetime(2024, 11, 2, 12, tzinfo=UTC),
           datetime(2023, 12, 30, 12, tzinfo=UTC), datetime(2024, 2, 28, 12, tzinfo=UTC)]


@st.composite
def base_bars(draw) -> list[dict]:
    if draw(st.booleans()):
        start = draw(st.sampled_from(ANCHORS)) + draw(st.integers(-600, 600)) * FIVE
    else:
        start = datetime(2019, 1, 1, tzinfo=UTC) + draw(st.integers(0, 700_000)) * FIVE
    n = draw(st.integers(1, 900))
    # steps of 1 bar mostly, sometimes a gap of up to 3 days (weekends, outages)
    steps = draw(st.lists(st.one_of(st.just(1), st.just(1), st.just(1), st.integers(2, 864)), min_size=n, max_size=n))
    rows, t, px = [], start, 100.0
    for i, k in enumerate(steps):
        t = t + k * FIVE if i else t
        o = px + draw(st.integers(-50, 50)) / 100
        c = o + draw(st.integers(-50, 50)) / 100
        h = max(o, c) + draw(st.integers(0, 30)) / 100
        l_ = min(o, c) - draw(st.integers(0, 30)) / 100
        v = draw(st.integers(0, 10_000)) / 8  # exact binary fractions: sums are order-independent
        b = draw(st.integers(0, int(v * 8))) / 8
        rows.append({"ts_close": t, "o": o, "h": h, "l": l_, "c": c, "v": v, "buy_v": b, "sell_v": v - b})
        px = c
    return rows


def to_frame(rows: list[dict]) -> pl.DataFrame:
    return pl.DataFrame(rows).with_columns(pl.col("ts_close").cast(pl.Datetime("ms", "UTC")))


# ---------------------------------------------------------------- property test (acceptance)

@settings(max_examples=300, deadline=None, suppress_health_check=[HealthCheck.too_slow, HealthCheck.data_too_large])
@given(rows=base_bars(), tf=st.sampled_from(TIMEFRAMES), s=st.sampled_from(SESSIONS), complete=st.booleans())
def test_resampled_ohlcv_equals_aggregation_of_base_bars(rows, tf, s, complete):
    df = to_frame(rows)
    got = resample(df, tf, s, complete_only=complete).to_dicts()
    want = oracle_resample(rows, tf, s, complete)
    assert len(got) == len(want)
    for g, w in zip(got, want):
        assert g == w
    # no duplicate keys, increasing, and every base bar used once (complete_only drops only the final bucket)
    keys = [g["ts_close"] for g in got]
    assert keys == sorted(set(keys))
    if not complete:
        assert sum(g["n_base"] for g in got) == len(rows)
        assert sum(g["v"] for g in got) == sum(r["v"] for r in rows)


# ---------------------------------------------------------------- conventions

def _ts(s: str) -> datetime:
    return datetime.fromisoformat(s).astimezone(UTC)


def _range(a: str, b: str) -> pl.DataFrame:
    ts = pl.datetime_range(_ts(a), _ts(b), "5m", eager=True, time_unit="ms", time_zone="UTC")
    return pl.DataFrame({"ts_close": ts}).with_columns(o=pl.lit(1.0), h=pl.lit(2.0), l=pl.lit(0.5), c=pl.lit(1.5),
                                                       v=pl.lit(1.0))


def test_fx_day_rolls_at_17_new_york_dst_aware():
    fx = SESS.for_asset("EURUSD")
    d = resample(_range("2024-03-07T22:05:00+00:00", "2024-03-13T00:00:00+00:00"), "1d", fx)
    opens = [t.astimezone(ZoneInfo("America/New_York")).strftime("%a %H:%M") for t in d["ts_open"]]
    assert set(t[-5:] for t in opens) == {"17:00"}
    lengths = (d["ts_close"] - d["ts_open"]).dt.total_hours().to_list()
    assert 23 in lengths and set(lengths) <= {23, 24}  # 2024-03-10 US DST start: a 23-hour trade day
    # Sunday 17:00 NY opens the Monday trade date
    w = resample(_range("2024-03-03T21:05:00+00:00", "2024-03-20T00:00:00+00:00"), "1w", fx)
    assert all(t.astimezone(ZoneInfo("America/New_York")).strftime("%a %H:%M") == "Sun 17:00" for t in w["ts_open"])


def test_crypto_utc_day_week_month():
    s = SESS.for_asset("BTCUSDT")
    assert s == UTC_24H.__class__("utc_24h", "UTC", "00:00")
    df = _range("2024-01-01T00:05:00+00:00", "2024-03-01T00:00:00+00:00")
    d = resample(df, "1d", s)
    assert d.height == 60 and all(t.hour == 0 and t.minute == 0 for t in d["ts_open"])
    w = resample(df, "1w", s)
    assert all(t.weekday() == 0 and t.hour == 0 for t in w["ts_open"])  # Monday 00:00 UTC
    m = resample(df, "1M", s)
    assert m["ts_open"].to_list() == [_ts("2024-01-01T00:00:00+00:00"), _ts("2024-02-01T00:00:00+00:00")]
    assert m["n_base"].to_list() == [31 * 288, 29 * 288]
    h4 = resample(df, "4h", s)
    assert all(t.hour % 4 == 0 for t in h4["ts_open"]) and set(h4["n_base"].to_list()) == {48}


def test_bar_is_keyed_by_nominal_close_and_incomplete_last_bucket_dropped():
    s = SESS.for_asset("BTCUSDT")
    df = _range("2024-01-01T00:05:00+00:00", "2024-01-01T10:00:00+00:00")
    h4 = resample(df, "4h", s)
    assert h4["ts_close"].to_list() == [_ts("2024-01-01T04:00:00+00:00"), _ts("2024-01-01T08:00:00+00:00")]
    allb = resample(df, "4h", s, complete_only=False)
    assert allb["ts_close"][-1] == _ts("2024-01-01T12:00:00+00:00")  # nominal close, after the last base bar
    # a bucket with a missing tail is still keyed at its nominal end
    gappy = df.filter(pl.col("ts_close") <= _ts("2024-01-01T02:00:00+00:00")).vstack(df.tail(1))
    assert resample(gappy, "4h", s)["ts_close"][0] == _ts("2024-01-01T04:00:00+00:00")


def test_5m_identity_and_bad_input_rejected():
    s = SESS.for_asset("BTCUSDT")
    df = _range("2024-01-01T00:05:00+00:00", "2024-01-01T01:00:00+00:00")
    r = resample(df, "5m", s)
    assert r["ts_close"].to_list() == df["ts_close"].to_list() and set(r["n_base"]) == {1}
    with pytest.raises(ValueError, match="strictly increasing"):
        resample(pl.concat([df, df.tail(1)]), "1h", s)
    with pytest.raises(ValueError, match="strictly increasing"):
        resample(df.reverse(), "1h", s)
    with pytest.raises(ValueError, match="grid"):
        check_base(df.with_columns(pl.col("ts_close") + pl.duration(minutes=1)))
    with pytest.raises(ValueError, match="UTC"):
        check_base(df.with_columns(pl.col("ts_close").dt.replace_time_zone(None)))
    with pytest.raises(ValueError, match="timeframe"):
        resample(df, "3h", s)
    assert resample(df.head(0), "1d", s).height == 0


def test_session_table_covers_the_universe_and_is_hashed():
    cfg = load_config().data["assets"]
    assets = set(cfg["crypto"]["symbols"]) | set(cfg["cfd"]["candidates"])
    assert assets == set(SESS.assets)
    for a in cfg["crypto"]["symbols"]:
        assert SESS.for_asset(a).name == "utc_24h" and SESS.roll(a) == "none"
    for a in ("EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD"):
        s = SESS.for_asset(a)
        assert (s.tz, s.day_start) == ("America/New_York", "17:00")
    assert SESS.roll("BUND") == "panama"
    assert SESS.roll("USOIL") == SESS.roll("UKOIL") == "panama_if_futures_based"
    assert len(SESS.sha256) == 64 and load_sessions().sha256 == SESS.sha256


def test_session_validation():
    with pytest.raises(ValueError):
        Session("x", "UTC", "17:02")
    with pytest.raises(ValueError):
        Session("x", "UTC", "00:00", "sunday")
    assert Session("x", "UTC", "17:00").shift_minutes == 7 * 60
    assert Session("x", "UTC", "00:00").shift_minutes == 0


def test_oracle_comparison_is_not_vacuous():
    """The oracle tells sessions apart: resampling under the FX convention does not match the UTC oracle."""
    rows = to_frame(oracle_rows := [{"ts_close": datetime(2024, 3, 4, tzinfo=UTC) + i * FIVE, "o": float(i), "h": i + 1.0,
                                     "l": i - 1.0, "c": i + 0.5, "v": 1.0, "buy_v": 0.5, "sell_v": 0.5}
                                    for i in range(1, 2000)])
    fx, utc = SESS.for_asset("EURUSD"), SESS.for_asset("BTCUSDT")
    for tf in ("4h", "1d", "1w"):
        assert resample(rows, tf, fx).to_dicts() == oracle_resample(oracle_rows, tf, fx)
        assert resample(rows, tf, fx).to_dicts() != oracle_resample(oracle_rows, tf, utc)
