"""Non-GDELT news-attention sources (news/alt.py): parsers on canned JSON, resume, known_ts / no look-ahead, abnormal math."""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta

import numpy as np
import polars as pl
import pytest

from ai_trader.news import alt


@pytest.fixture()
def man(tmp_path):
    return alt.alt_manifest(tmp_path / "alt")


def _ts(d: date, h: int = 12) -> int:
    return int(datetime(d.year, d.month, d.day, h, tzinfo=UTC).timestamp())


# ---------------------------------------------------------------- parsers

def test_parse_pageviews():
    d = {"items": [{"project": "en.wikipedia", "article": "Bitcoin", "granularity": "daily", "timestamp": "2015070100", "access": "all-access",
                    "agent": "user", "views": 12957}, {"article": "Bitcoin", "timestamp": "2015070200", "views": 9802}]}
    assert alt.parse_pageviews(d) == {date(2015, 7, 1): 12957.0, date(2015, 7, 2): 9802.0}
    assert alt.parse_pageviews({}) == {}


def test_parse_revisions_v1_and_v2():
    v1 = {"continue": {"rvcontinue": "x"}, "query": {"pages": {"28249265": {"pageid": 28249265, "title": "Bitcoin",
          "revisions": [{"timestamp": "2009-03-08T16:41:07Z"}, {"timestamp": "2009-03-08T23:59:59Z"}]}}}}
    v2 = {"query": {"pages": [{"pageid": 1, "title": "Bitcoin", "revisions": [{"timestamp": "2020-01-01T00:00:00Z"}]}]}}
    assert [t.date() for t in alt.parse_revisions(v1)] == [date(2009, 3, 8)] * 2
    assert alt.parse_revisions(v2)[0] == datetime(2020, 1, 1, tzinfo=UTC)


def test_parse_hn_and_fng_and_guardian():
    nb, ts = alt.parse_hn({"nbHits": 2, "hits": [{"created_at_i": 1512172092, "objectID": "1"}, {"created_at_i": 1512169220}]})
    assert nb == 2 and ts == [1512172092, 1512169220]
    f = alt.parse_fng({"name": "Fear and Greed Index", "data": [{"value": "20", "value_classification": "Extreme Fear", "timestamp": "1794528000",
                                                                  "time_until_update": "4108"}, {"value": "28", "timestamp": "1794441600"}]})
    assert f["value"].to_list() == [28.0, 20.0] and f["ts"][1] == datetime(2026, 11, 13, tzinfo=UTC)
    days, pages = alt.parse_guardian({"response": {"status": "ok", "pages": 3, "results": [{"webPublicationDate": "2017-12-07T18:00:01Z"}]}})
    assert days == [date(2017, 12, 7)] and pages == 3


# ---------------------------------------------------------------- HN recursion and resume

def _hn_fake(per_day: dict[date, int], calls: list):
    def get(params):
        calls.append(params)
        lo, hi = (int(x.split("=" if ">=" in x else "<")[-1]) for x in params["numericFilters"].split(","))
        hits = []
        for d, n in per_day.items():
            t = _ts(d)
            if lo <= t < hi:
                hits += [{"created_at_i": t}] * n
        return {"nbHits": len(hits), "hits": hits[: int(params["hitsPerPage"])]}
    return get


def test_hn_counts_split_and_single_day_cap():
    per_day = {date(2021, 1, 3): 700, date(2021, 1, 20): 600, date(2021, 1, 25): 1500, date(2021, 1, 31): 1}
    calls: list = []
    stats: dict = {}
    c = alt.hn_counts("bitcoin", "comment", date(2021, 1, 1), date(2021, 1, 31), _hn_fake(per_day, calls), stats)
    assert c == {k: float(v) for k, v in per_day.items()}
    assert stats["approx_days"] == 1 and stats["requests"] == len(calls) > 1
    assert all(p["typoTolerance"] == "false" and p["queryType"] == "prefixNone" for p in calls)


def test_hn_resume_never_refetches(man):
    calls: list = []
    get = _hn_fake({date(2020, 6, 1): 3}, calls)
    alt.fetch_hn(man, date(2021, 3, 1), terms=("bitcoin",), tags=("story",), start=date(2020, 1, 1), get=get, log=lambda *_: None)
    n = len(calls)
    assert n == 12 + 2  # monthly seeds: 2020 (12) + Jan-Feb 2021
    alt.fetch_hn(alt.alt_manifest(man.root), date(2021, 3, 1), terms=("bitcoin",), tags=("story",), start=date(2020, 1, 1), get=get, log=lambda *_: None)
    assert len(calls) == n
    alt.fetch_hn(man, date(2021, 3, 5), terms=("bitcoin",), tags=("story",), start=date(2020, 1, 1), get=get, log=lambda *_: None)
    assert len(calls) == n + 1 and str(alt._ts(date(2021, 3, 1))) in calls[-1]["numericFilters"]
    s = alt.load_series(man, "hn", "bitcoin_story")
    assert s.height == (date(2021, 3, 4) - date(2020, 1, 1)).days + 1
    assert s.filter(pl.col("date") == date(2020, 6, 1))["value"].item() == 3.0 and s["value"].sum() == 3.0


def test_pageviews_store_and_null_before_existence(man):
    def get(url):
        assert "/daily/2015070100/" in url
        return {"items": [{"timestamp": "2015080100", "views": 10}, {"timestamp": "2015080300", "views": 30}]}
    alt.fetch_pageviews(man, date(2015, 8, 5), concepts={"bitcoin": ("Bitcoin",)}, get=get, log=lambda *_: None)
    assert alt.fetch_pageviews(man, date(2015, 8, 5), concepts={"bitcoin": ("Bitcoin",)}, get=get, log=lambda *_: None) == []
    w = alt.wiki_days(man)
    v = dict(zip(w["date"].to_list(), w["pv_bitcoin"].to_list(), strict=True))
    assert v[date(2015, 7, 31)] is None and v[date(2015, 8, 1)] == 10 and v[date(2015, 8, 2)] == 0 and v[date(2015, 8, 4)] == 0


def test_guardian_refused_is_recorded(man):
    def get(params):
        raise alt.Refused("HTTP 401 content.guardianapis.com")
    assert alt.fetch_guardian(man, date(2016, 1, 1), get=get, log=lambda *_: None) == 0
    assert man.get("guardian/status")["status"] == "refused"


# ---------------------------------------------------------------- abnormal attention

def test_abnormal_math():
    x = np.full(40, 99.0)
    a = alt.abnormal(x)
    assert np.isnan(a[:20]).all() and np.allclose(a[20:], 0.0)
    y = x.copy()
    y[35] = 999.0  # 10x spike
    b = alt.abnormal(y)
    assert np.isclose(b[35], np.log(1000.0) - np.log(100.0))
    assert np.isclose(b[36], 0.0)  # the median of the prior 28 days ignores one spike
    # strictly trailing: the value at d never enters its own median
    z = np.r_[np.full(28, 0.0), 1e6]
    assert np.isclose(alt.abnormal(z)[28], np.log1p(1e6))
    # missing values: nan at d -> nan; fewer than 20 prior values -> nan
    w = np.r_[np.full(19, 5.0), np.nan, 5.0]
    assert np.isnan(alt.abnormal(w)).all()


# ---------------------------------------------------------------- known_ts / no look-ahead

def _seed(man, start: date, days: int):
    d0 = start
    span = [d0 + timedelta(days=i) for i in range(days)]
    rng = np.random.default_rng(1)
    for t in ("Bitcoin", "Cryptocurrency", "Ethereum"):
        v = rng.poisson(1000, size=days).astype(float)
        alt._put_chunk(man, "wiki", f"pv_{t}", pl.DataFrame({"date": span, "value": v}), span[0], span[-1])
    for t in alt.EDIT_TITLES:
        alt._put_chunk(man, "edits", t, pl.DataFrame({"date": span, "value": rng.poisson(5, size=days).astype(float)}), span[0], span[-1])
    for term in alt.HN_TERMS:
        for tag in alt.HN_TAGS:
            v = rng.poisson(20, size=days).astype(float)
            v[-1] = 2000.0  # last day spikes
            alt._put_chunk(man, "hn", f"{term}_{tag}", pl.DataFrame({"date": span, "value": v}), span[0], span[-1])
    f = pl.DataFrame({"ts": [datetime.combine(d, datetime.min.time(), tzinfo=UTC) for d in span], "value": np.linspace(10, 90, days)},
                     schema={"ts": pl.Datetime("us", "UTC"), "value": pl.Float64})
    (man.root / "fng" / "index").mkdir(parents=True, exist_ok=True)
    f.write_parquet(man.root / "fng" / "index" / "x.parquet")
    man.put("fng/index/x", path="fng/index/x.parquet", rows=days, status="ok")
    return span


def test_daily_alt_features_known_ts(man):
    span = _seed(man, date(2024, 1, 1), 60)
    last = span[-1]  # 2024-02-29
    midnight = datetime.combine(last + timedelta(days=1), datetime.min.time(), tzinfo=UTC)
    r0 = alt.daily_alt_features(midnight, man=man).row(0, named=True)
    r1 = alt.daily_alt_features(midnight + timedelta(minutes=30), man=man).row(0, named=True)
    r2 = alt.daily_alt_features((midnight + timedelta(days=1)).replace(tzinfo=None), man=man).row(0, named=True)  # naive = UTC
    # HN and edits for day d are known at d + 1 00:30, not at d + 1 00:00
    assert r0["day_hn"] == last - timedelta(days=1) and r1["day_hn"] == last
    assert r1["hn_story"] > 3 and abs(r0["hn_story"]) < 1
    # pageviews for day d known at d + 2 00:00
    assert r0["day_wiki"] == last - timedelta(days=1) and r2["day_wiki"] == last
    # Fear and Greed stamped 00:00 of its day: the last value is the day `last`
    assert r0["day_fng"] == last and r0["kt_fng"] == datetime.combine(last, datetime.min.time(), tzinfo=UTC)
    for r in (r0, r1, r2):
        for k, v in r.items():
            if k.startswith("kt_") and v is not None:
                assert v <= r["asof"]
    # stale: far after the data ends everything is null
    r9 = alt.daily_alt_features(midnight + timedelta(days=10), man=man).row(0, named=True)
    assert all(r9[c] is None for c in alt.ALT_COLS)
    assert set(alt.ALT_COLS) <= set(alt.daily_alt_features(midnight, man=man).columns)


def test_alt_table_matches_live_and_is_point_in_time(man):
    span = _seed(man, date(2024, 1, 1), 60)
    asofs = pl.datetime_range(datetime(2024, 1, 2, tzinfo=UTC), datetime(2024, 3, 2, tzinfo=UTC), "1d", eager=True, time_zone="UTC")
    t = alt.alt_table(asofs, man=man)
    alt.assert_point_in_time(t)
    one = alt.daily_alt_features(datetime(2024, 2, 20, tzinfo=UTC), man=man)
    row = t.filter(pl.col("asof") == datetime(2024, 2, 20, tzinfo=UTC))
    assert row.select(alt.ALT_COLS).rows() == one.select(alt.ALT_COLS).rows()
    # fng_chg = (F&G_d - F&G_{d-1}) / 50 for consecutive days
    step = (90 - 10) / (len(span) - 1)
    assert np.isclose(one["fng_chg"].item(), step / 50)
    # features of an earlier as_of do not change when later data arrive
    before = alt.daily_alt_features(datetime(2024, 2, 10, tzinfo=UTC), man=man)
    alt._put_chunk(man, "hn", "bitcoin_story", pl.DataFrame({"date": [date(2024, 3, 1)], "value": [5.0]}), date(2024, 3, 1), date(2024, 3, 1))
    after = alt.daily_alt_features(datetime(2024, 2, 10, tzinfo=UTC), man=man)
    assert before.select(alt.ALT_COLS).rows() == after.select(alt.ALT_COLS).rows()
