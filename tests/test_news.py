"""News / events / on-chain features (agent F3): point-in-time guarantees, manifest, parsers, dashboard on synthetic inputs."""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta

import numpy as np
import polars as pl
import pytest

from ai_trader.news import events, features, gdelt, onchain, rss
from ai_trader.news.store import Manifest

T0 = datetime(2024, 1, 1, tzinfo=UTC)


def _fake_timeline(start: datetime, end: datetime, step: timedelta, value_fn, mode: str) -> dict:
    data, t = [], start
    while t < end:
        x = {"date": t.strftime("%Y%m%dT%H%M%SZ"), "value": value_fn(t)}
        if mode == "timelinevolraw":
            x["norm"] = 100000
        data.append(x)
        t += step
    return {"query_details": {"date_resolution": "day" if step >= timedelta(days=1) else "15min"}, "timeline": [{"data": data}]}


SPIKE = datetime(2024, 3, 1, tzinfo=UTC)


def _req_factory(counter: list, spike_day: datetime = SPIKE):
    def req(params, log=print):
        counter.append(params)
        s = datetime.strptime(params["startdatetime"], "%Y%m%d%H%M%S").replace(tzinfo=UTC)
        e = datetime.strptime(params["enddatetime"], "%Y%m%d%H%M%S").replace(tzinfo=UTC) + timedelta(seconds=1)
        daily = (e - s) > timedelta(days=10)
        step = timedelta(days=1) if daily else timedelta(minutes=15)
        if params["mode"] == "timelinetone":
            return _fake_timeline(s, e, step, lambda t: -1.0, params["mode"])
        return _fake_timeline(s, e, step, lambda t: 1000.0 if spike_day <= t < spike_day + timedelta(days=1) else 100.0, params["mode"])
    return req


@pytest.fixture()
def man(tmp_path):
    return Manifest(tmp_path)


def test_manifest_prevents_redownload(man):
    calls = []
    req = _req_factory(calls)
    a = gdelt.fetch_daily(man, T0 + timedelta(days=120), qids=["bitcoin"], modes=("timelinevolraw",), request=req, log=lambda *_: None)
    assert len(calls) == 1 and a
    gdelt.fetch_daily(man, T0 + timedelta(days=120), qids=["bitcoin"], modes=("timelinevolraw",), request=req, log=lambda *_: None)
    assert len(calls) == 1  # nothing re-downloaded
    gdelt.fetch_daily(man, T0 + timedelta(days=125), qids=["bitcoin"], modes=("timelinevolraw",), request=req, log=lambda *_: None)
    assert len(calls) == 2  # only the increment
    assert calls[1]["startdatetime"] == (T0 + timedelta(days=120)).strftime("%Y%m%d%H%M%S")
    # a reopened manifest remembers
    m2 = Manifest(man.root)
    gdelt.fetch_daily(m2, T0 + timedelta(days=125), qids=["bitcoin"], modes=("timelinevolraw",), request=req, log=lambda *_: None)
    assert len(calls) == 2


def test_failed_chunk_is_retried(man):
    calls = []
    gdelt.fetch_chunk("bitcoin", "timelinevolraw", T0, T0 + timedelta(days=20), man, request=lambda p, log=None: calls.append(1) or None, log=lambda *_: None)
    gdelt.fetch_chunk("bitcoin", "timelinevolraw", T0, T0 + timedelta(days=20), man, request=lambda p, log=None: calls.append(1) or None, log=lambda *_: None)
    assert len(calls) == 2


def _build_store(man, start=T0 - timedelta(days=200), end=T0 + timedelta(days=150)):
    calls = []
    req = _req_factory(calls)
    T0s = start
    for q in gdelt.QUERIES:
        gdelt.fetch_chunk(q, "timelinevolraw", T0s, end, man, request=req, log=lambda *_: None)
        gdelt.fetch_chunk(q, "timelinetone", T0s, end, man, request=req, log=lambda *_: None)
    # intraday window around the spike
    for q in gdelt.CORE_INTRADAY:
        gdelt.fetch_chunk(q, "timelinevolraw", SPIKE - timedelta(days=5), SPIKE + timedelta(days=2), man, kind="intraday", request=req, log=lambda *_: None)

    def get(chart, params):
        days = [(T0s + timedelta(days=i)) for i in range((end - T0s).days)]
        return {"values": [{"x": int(d.timestamp()), "y": (5.0 if d == SPIKE else 1.0) * (1 + 0.001 * i)} for i, d in enumerate(days)]}
    onchain.fetch(man, end, get=get, log=lambda *_: None)
    ev = pl.DataFrame({"kind": ["cpi", "fomc", "halving"], "name": ["CPI", "FOMC", "halving"],
                       "scheduled_ts": [SPIKE + timedelta(hours=13, minutes=30), SPIKE + timedelta(days=10, hours=19), SPIKE + timedelta(days=20)],
                       "schedule_known_ts": [SPIKE - timedelta(days=30), SPIKE + timedelta(days=10, hours=19), SPIKE + timedelta(days=13)],
                       "source": ["t", "t", "t"]}).with_columns(pl.col("scheduled_ts", "schedule_known_ts").cast(pl.Datetime("us", "UTC")))
    return ev


def test_daily_features_point_in_time(man):
    ev = _build_store(man)
    d = features.daily_table(man, ev, T0, T0 + timedelta(days=140))
    features.assert_point_in_time(d)
    # every component known at or before its use time
    for c in ("kt_news", "kt_onchain", "kt_events"):
        sub = d.filter(pl.col(c).is_not_null())
        assert (sub[c] <= sub["asof"]).all()
    row = lambda t: d.filter(pl.col("asof") == t).row(0, named=True)  # noqa: E731
    # the news spike on day SPIKE is invisible at SPIKE 00:00 and visible at SPIKE + 1 d 00:00
    assert abs(row(SPIKE)["att_btc"]) < 1e-9
    assert row(SPIKE + timedelta(days=1))["att_btc"] > 1.5
    assert row(SPIKE + timedelta(days=2))["att_btc_l1"] > 1.5 and abs(row(SPIKE + timedelta(days=1))["att_btc_l1"]) < 1e-9
    # on-chain day d is known only at d + 2 days
    assert abs(row(SPIKE + timedelta(days=1))["oc_ntx"] or 0) < 0.1
    assert row(SPIKE + timedelta(days=2))["oc_ntx"] > 1.0
    # events: CPI on SPIKE is "next" at SPIKE 00:00; the unscheduled-style FOMC (schedule known only at its time) never is
    assert row(SPIKE)["ev_macro_next"] == 1.0 and row(SPIKE + timedelta(days=1))["ev_macro_today"] == 1.0
    assert row(SPIKE + timedelta(days=10))["ev_macro_next"] == 0.0
    assert row(SPIKE + timedelta(days=11))["ev_macro_today"] == 1.0
    # halving schedule known 7 d before: flagged the day before only
    assert row(SPIKE + timedelta(days=20))["ev_crypto_next"] == 1.0


def test_lookahead_guard_raises():
    d = pl.DataFrame({"asof": [T0], "kt_news": [T0 + timedelta(minutes=1)]}).with_columns(pl.col("asof", "kt_news").cast(pl.Datetime("us", "UTC")))
    with pytest.raises(AssertionError):
        features.assert_point_in_time(d)


def test_hourly_features_point_in_time(man):
    ev = _build_store(man)
    d = features.daily_table(man, ev, T0, T0 + timedelta(days=140))
    h = features.hourly_table(man, ev, SPIKE - timedelta(days=4), SPIKE + timedelta(days=1), daily=d)
    features.assert_point_in_time(h)
    r = lambda t: h.filter(pl.col("asof") == t).row(0, named=True)  # noqa: E731
    # the spike's first hour [SPIKE, SPIKE+1h) is known at SPIKE+1h, not at SPIKE
    assert r(SPIKE)["gbitcoin_n"] == 400.0
    assert r(SPIKE + timedelta(hours=1))["gbitcoin_n"] == 4000.0
    # event proximity uses only known schedules
    assert r(SPIKE + timedelta(hours=13))["ev_next_1h"] == 1.0
    assert r(SPIKE + timedelta(hours=14))["h_since_event"] == pytest.approx(0.5)
    assert (h.filter(pl.col("d_asof").is_not_null())["d_asof"] <= h.filter(pl.col("d_asof").is_not_null())["asof"]).all()


def test_et_to_utc_dst():
    assert events.et_to_utc(date(2024, 1, 11), 8, 30) == datetime(2024, 1, 11, 13, 30, tzinfo=UTC)
    assert events.et_to_utc(date(2024, 7, 11), 8, 30) == datetime(2024, 7, 11, 12, 30, tzinfo=UTC)
    assert events.et_to_utc(date(2024, 3, 10), 14, 0) == datetime(2024, 3, 10, 18, 0, tzinfo=UTC)  # DST starts that Sunday


def test_fomc_parsers_and_build():
    cur = ('<h4>2025 FOMC Meetings</h4><div class="fomc-meeting__month col-xs-5"><strong>January</strong></div>'
           '<div class="fomc-meeting__date col-xs-4">28-29</div><div class="fomc-meeting__month"><strong>Apr/May</strong></div>'
           '<div class="fomc-meeting__date">30-1</div><div class="fomc-meeting__month"><strong>August</strong></div>'
           '<div class="fomc-meeting__date">22 (notation vote)</div>')
    p = events.parse_fomc_current(cur)
    assert (date(2025, 1, 29), "") in p and (date(2025, 5, 1), "") in p and (date(2025, 8, 22), "notation") in p
    hist = "<h5>March 15 (unscheduled) Meeting - 2020</h5><h5>June 9-10 Meeting - 2020</h5>"
    q = events.parse_fomc_historical(hist)
    assert (date(2020, 3, 15), "unscheduled") in q and (date(2020, 6, 10), "") in q
    ev = events.build(p + q)
    assert (ev["schedule_known_ts"] <= ev["scheduled_ts"]).all()
    un = ev.filter(pl.col("kind") == "fomc_unscheduled")
    assert (un["schedule_known_ts"] == un["scheduled_ts"]).all()
    assert ev.filter(pl.col("kind") == "notation").height == 0


def test_rss_parse_and_dedupe(man):
    xml = ("<rss><channel><item><title>x</title><guid>a1</guid><pubDate>Fri, 13 Nov 2026 10:00:00 +0000</pubDate></item>"
           "<item><link>https://e/2</link><pubDate>Fri, 13 Nov 2026 11:00:00 GMT</pubDate></item></channel></rss>")
    items = rss.parse_items(xml)
    assert len(items) == 2 and items[0][1] == datetime(2026, 11, 13, 10, tzinfo=UTC)
    t = datetime(2026, 11, 13, 12, 5, tzinfo=UTC)
    assert rss.poll(man, get=lambda u: xml, now=t, log=lambda *_: None) == 2  # deduped across feeds (same items)
    assert rss.poll(man, get=lambda u: xml, now=t + timedelta(minutes=15), log=lambda *_: None) == 0
    hc = rss.hourly_counts(man)
    assert (hc["known_ts"] > datetime(2026, 11, 13, 12, 5, tzinfo=UTC)).all()


def test_study_on_synthetic():
    from ai_trader.news import study

    n = 700
    rng = np.random.default_rng(0)
    dates = pl.date_range(date(2022, 1, 1), date(2022, 1, 1) + timedelta(days=n - 1), "1d", eager=True)
    shock = rng.standard_normal(n)
    lv = np.zeros(n)
    for i in range(1, n):
        lv[i] = 0.9 * lv[i - 1] + 0.3 * rng.standard_normal() + 0.5 * shock[i - 1]  # yesterday's news moves today's vol
    rv = 1e-4 * np.exp(lv)
    ret = np.sqrt(rv) * rng.standard_normal(n)
    df = pl.DataFrame({"date": dates, "rv": rv, "ret": ret})
    feats = pl.DataFrame({"asof": [datetime(d.year, d.month, d.day, tzinfo=UTC) + timedelta(days=1) for d in dates.to_list()], "att_btc": shock})
    fr = study.frame(df, feats)
    r = study.run_h1(fr, date(2023, 6, 1))
    p = r["variants"]["primary"]["llm_in_sample"]
    assert p["diff"] < 0 and p["nw_t"] < -2  # the planted news effect is found
    h2 = study.run_h2(fr, date(2023, 6, 1))
    assert h2["segments"]["llm_in_sample"]["skill"] < 0.02


def test_dashboard_builder_synthetic(tmp_path):
    import importlib.util
    from pathlib import Path

    spec = importlib.util.spec_from_file_location("build_dashboard", Path(__file__).resolve().parents[1] / "scripts" / "build_dashboard.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    out = tmp_path / "dash.html"
    data = mod.synthetic_inputs(seed=1)
    html = mod.render(data)
    out.write_text(html, encoding="utf-8")
    assert "plotly" in html and "btc_field" in html.lower() or "BTC field" in html
    for k in ("state", "latent", "card", "news", "mi"):
        assert k in data["panels"]
