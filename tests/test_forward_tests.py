"""Forward-test runners: the statistic never touches data before the registration date, frozen parameters never touch
data after it, and nothing is recorded below the registered sample size (or twice)."""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta

import numpy as np
import polars as pl
import pytest

from ai_trader.research import forward_tests as ft

START = datetime(2026, 11, 13, tzinfo=UTC)


def _daily(n_days: int = 420, syms=("AAA", "BBB", "CCC", "DDD", "EEE", "FFF"), seed: int = 0) -> pl.DataFrame:
    """Synthetic universe-L daily panel with the columns of leverage_study.build + add_loghar."""
    from ai_trader.research.leverage_study import add_features

    rng = np.random.default_rng(seed)
    d0 = START.date() - timedelta(days=n_days - 60)   # 60 forward days
    parts = []
    for i, s in enumerate(syms):
        lv = np.zeros(n_days)
        for t in range(1, n_days):
            lv[t] = 0.9 * lv[t - 1] + rng.normal(0, 0.4)
        rv = 1e-4 * (1 + i / 10) * np.exp(lv)
        ret = rng.normal(0, np.sqrt(rv))
        neg = rv * rng.uniform(0.3, 0.7, n_days)
        d = pl.DataFrame({"date": [d0 + timedelta(days=k) for k in range(n_days)], "rv": rv, "rs_neg": neg, "rs_pos": rv - neg,
                          "ret": ret, "close": 100 * np.exp(np.cumsum(ret))})
        d = add_features(d).with_columns(pl.lit(s).alias("symbol"), pl.lit("L").alias("universe"),
                                         pl.Series("fz", rng.normal(0, 1, n_days)))
        parts.append(d)
    return ft.add_loghar(pl.concat(parts))


def _corrupt(df: pl.DataFrame, before: datetime | None = None, after: datetime | None = None) -> pl.DataFrame:
    """Overwrite every numeric value of rows before (or at/after) a time with garbage."""
    m = (pl.col("origin") < before) if before is not None else (pl.col("origin") >= after)
    num = [c for c, t in df.schema.items() if t in (pl.Float64, pl.Float32) and c not in ("origin",)]
    return df.with_columns([pl.when(m).then(pl.lit(1e6)).otherwise(pl.col(c)).alias(c) for c in num])


def test_forward_only_gate():
    df = pl.DataFrame({"origin": [START - timedelta(days=1), START, START + timedelta(days=1)], "x": [1, 2, 3]})
    assert ft.forward_only(df, "origin", START)["x"].to_list() == [2, 3]
    dd = pl.DataFrame({"date": [date(2026, 11, 12), date(2026, 11, 13)], "x": [1, 2]})
    assert ft.forward_only(dd, "date", START)["x"].to_list() == [2]


@pytest.mark.parametrize("name", ["lev6", "vm6", "nn7", "lev7", "ds4"])
def test_statistic_ignores_pre_registration_rows(name):
    d = _daily()
    if name == "lev6":
        par = ft.lev6_freeze(d, START)
        f = lambda x: ft.lev6_stat(x, START, par)  # noqa: E731
    elif name == "vm6":
        par = ft.vm6_freeze(d.with_columns(pl.lit("L").alias("universe")), START)
        par["T"] = par["L"]
        f = lambda x: ft.vm6_stat(pl.concat([x, x.with_columns(pl.lit("T").alias("universe"))]), START, par)  # noqa: E731
    elif name == "nn7":
        f = lambda x: ft.nn7_stat(x, START)  # noqa: E731
    elif name == "lev7":
        f = lambda x: ft.lev7_stat(x, START)  # noqa: E731
    else:
        par = ft.ds4_freeze(d, START, START - timedelta(days=1))
        v0 = d.select("symbol", "date").with_columns(pl.lit("up").alias("v0_dir"))
        f = lambda x: ft.ds4_stat(ft.ds4_rows(x, v0, START, par))  # noqa: E731
    a = f(d)
    if name in ("nn7",):   # the high-vol flag uses a trailing 1-year median of lrv_w: keep that column (a feature known at t)
        b = f(d.with_columns(pl.when(pl.col("origin") < START).then(pl.lit(1e6)).otherwise(pl.col("rv_next")).alias("rv_next"),
                             pl.when(pl.col("origin") < START).then(pl.lit(1e6)).otherwise(pl.col("lrv_next")).alias("lrv_next")))
    else:
        b = f(_corrupt(d, before=START))
    assert a["n"] > 0
    assert a["n"] == b["n"]
    for k in ("est", "t"):
        if a.get(k) is not None and np.isfinite(a[k]):
            assert a[k] == pytest.approx(b[k], rel=1e-9, abs=1e-12)


def test_frozen_parameters_ignore_forward_rows():
    d = _daily()
    # every row decided at or after START (its target day ends after START) is garbage: the frozen fits must not move
    bad = _corrupt(d, after=START)
    assert ft.lev6_freeze(d, START) == ft.lev6_freeze(bad, START)
    v1, v2 = ft.vm6_freeze(d, START), ft.vm6_freeze(bad, START)
    assert v1["L"]["wd"]["beta"] == pytest.approx(v2["L"]["wd"]["beta"])
    # and the boundary is tight: corrupting the last admissible origin (target complete exactly at START) does move them
    edge = _corrupt(d, after=START - timedelta(days=1))
    assert ft.vm6_freeze(edge, START)["L"]["wd"]["beta"] != pytest.approx(v1["L"]["wd"]["beta"])
    te = START - timedelta(days=1)   # DS4 trains on targets complete by the registration date
    s1, s2 = ft.ds4_freeze(d, START, te), ft.ds4_freeze(_corrupt(d, after=te), START, te)
    assert s1["ubar"] == pytest.approx(s2["ubar"]) and s1["coefs"] == s2["coefs"]


def test_decide():
    assert ft.decide(5, 120, True, False) == "pending"
    assert ft.decide(120, 120, True, False) == "supports"
    assert ft.decide(130, 120, False, True) == "contradicts"
    assert ft.decide(130, 120, False, False) == "inconclusive"


def _fake_spec(monkeypatch, n: int, need: int = 10, met: bool = True):
    spec = ft.Spec("H-FAKE1", "2026-11-12", START, need, "forward dates", "x > 0")
    monkeypatch.setitem(ft.FORWARD, "H-FAKE1", spec)
    monkeypatch.setitem(ft.EVALUATORS, "H-FAKE1", lambda ctx: {"n": n, "est": 0.5, "t": 3.0, "met": met, "wrong_side": False})


def test_refuses_to_record_below_sample_size(monkeypatch):
    _fake_spec(monkeypatch, n=3)
    calls = []
    monkeypatch.setattr(ft, "already_recorded", lambda hid: False)
    out = ft.evaluate_one("H-FAKE1", ft.Context(START + timedelta(days=5)), record=True, recorder=lambda *a, **k: calls.append(a))
    assert out["verdict"] == "pending" and out["status"] == "collecting"
    assert out["record"].startswith("refused")
    assert calls == []


def test_records_once_when_sample_reached(monkeypatch):
    _fake_spec(monkeypatch, n=12)
    calls = []
    monkeypatch.setattr(ft, "already_recorded", lambda hid: False)
    out = ft.evaluate_one("H-FAKE1", ft.Context(START + timedelta(days=20)), record=True, recorder=lambda *a, **k: calls.append(a))
    assert out["verdict"] == "supports" and out["record"] == "recorded" and len(calls) == 1
    assert calls[0][1] == "clean_holdout"
    monkeypatch.setattr(ft, "already_recorded", lambda hid: True)
    out2 = ft.evaluate_one("H-FAKE1", ft.Context(START + timedelta(days=21)), record=True, recorder=lambda *a, **k: calls.append(a))
    assert out2["record"].startswith("skipped") and len(calls) == 1


def test_tp1b_uses_only_forward_sessions():
    start = datetime(2026, 11, 10, tzinfo=UTC)
    rows = []
    for k in range(10):
        day = date(2026, 11, 3) + timedelta(days=k)
        rows.append({"symbol": "NVDAUSDT", "date": day, "open_utc": datetime(day.year, day.month, day.day, 14, 30, tzinfo=UTC),
                     "on": 0.02, "on_z": 2.0, "first2": 0.01 if day >= start.date() else -0.5})
    out = ft.tp1b_stat(pl.DataFrame(rows), start)
    assert out["n"] == 3          # 11-10, 11-11, 11-12 only
    assert out["est"] > 0         # the losing pre-registration sessions are not counted


def test_pd5_reversal_sign_and_forward_weeks():
    rng = np.random.default_rng(1)
    rows = []
    strategies = [f"s{i}" for i in range(8)]
    prev = rng.normal(size=8)
    for w in range(12):
        wk = date(2026, 11, 9) + timedelta(days=7 * w)   # the first week starts before PD5_FIRST_WEEK
        cur = -prev + rng.normal(0, 0.1, 8)
        for s, x in zip(strategies, cur, strict=True):
            for k in range(7):
                rows.append({"universe": "L", "strategy": s, "symbol": "AAA", "date": wk + timedelta(days=k), "pnl_d": x / 7})
        prev = cur
    weekly = ft.pd5_weekly(pl.DataFrame(rows))
    assert weekly["week"].min() >= ft.PD5_FIRST_WEEK
    out = ft.pd5_stat(weekly)
    assert out["n"] == 10 and out["est"] < -0.5


def test_sp4_check():
    rows = [{"members": ["log_sigma", "lrv_w"], "prob": 0.95}, {"members": ["next_is_weekend", "dow_sin"], "prob": 0.9},
            {"members": ["er_7d"], "prob": 0.4}]
    assert ft.sp4_check(rows)["ok"]
    rows[2]["prob"] = 0.6
    assert not ft.sp4_check(rows)["ok"]
    assert ft.sp4_due(datetime(2026, 11, 20, tzinfo=UTC)) == []
    assert ft.sp4_due(datetime(2027, 1, 5, tzinfo=UTC)) == [date(2026, 12, 1), date(2027, 1, 1)]
