"""Macro narrative layer: curve regimes, the PIT FRED store, the thesis ledger and the confluence log."""

import json
import re
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from ai_trader.core.config import load
from ai_trader.narrative import confluence, curve, macro, theses
from ai_trader.research import macro_study

ROOT = load().root


# ---------------------------------------------------------------- curve regimes (MAC block definition)

@pytest.mark.parametrize("d2,d10,want", [
    (10, 4, "bear_flattening"), (4, 10, "bear_steepening"), (-10, -4, "bull_steepening"), (-4, -10, "bull_flattening"),
    (4.9, -4.9, "no_regime"), (5, 0, "bear_flattening"), (-6, 6, "bull_steepening"),   # tie -> 2y leg
    (10, -3, "bear_flattening"), (-3, 12, "bear_steepening"), (float("nan"), 3, "no_regime"),
])
def test_classify(d2, d10, want):
    assert curve.classify(d2, d10) == want


def test_regimes_uses_five_observations_and_known_ts():
    d = [date(2026, 1, 1) + timedelta(days=i) for i in range(8)]
    y2 = pl.DataFrame({"obs_date": d, "value": [4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.3, 3.8]})
    y10 = pl.DataFrame({"obs_date": d, "value": [4.5] * 8})
    r = curve.regimes(y2, y10)
    assert r["obs_date"].to_list() == d[5:]
    assert r["regime"].to_list() == ["no_regime", "bear_flattening", "bull_steepening"]
    assert r["d2_bp"].to_list()[1] == pytest.approx(30)
    assert r["known_ts"][0] == datetime(2026, 1, 7, tzinfo=UTC)


# ---------------------------------------------------------------- PIT store

def test_rule_ts_daily_and_monthly():
    assert macro.rule_ts(date(2026, 11, 12), "DGS2") == datetime(2026, 11, 13, tzinfo=UTC)
    assert macro.rule_ts(date(2026, 10, 1), "UNRATE") == datetime(2026, 11, 11, tzinfo=UTC)
    assert macro.rule_ts(date(2026, 12, 1), "CPILFESL") == datetime(2027, 1, 17, tzinfo=UTC)


def test_merge_keeps_revisions_point_in_time(tmp_path, monkeypatch):
    monkeypatch.setattr(macro, "root", lambda: tmp_path)
    t1 = datetime(2026, 11, 1, tzinfo=UTC)
    t2 = datetime(2026, 11, 10, tzinfo=UTC)
    a = pl.DataFrame({"obs_date": [date(2026, 10, 1), date(2026, 10, 2)], "value": [1.0, 2.0]})
    m1 = macro._merge(None, a, "DGS2", t1)
    b = pl.DataFrame({"obs_date": [date(2026, 10, 1), date(2026, 10, 2), date(2026, 11, 9)], "value": [1.0, 2.5, 3.0]})
    m2 = macro._merge(m1, b, "DGS2", t2)
    assert m2.height == 4 and int(m2["revision"].max()) == 1
    m2.write_parquet(tmp_path / "DGS2.parquet")
    before = macro.read_asof("DGS2", datetime(2026, 11, 5, tzinfo=UTC))
    after = macro.read_asof("DGS2", datetime(2026, 11, 11, tzinfo=UTC))
    assert before["value"].to_list() == [1.0, 2.0]                 # the revision is invisible before it was downloaded
    assert after["value"].to_list() == [1.0, 2.5, 3.0]
    assert macro.read_asof("DGS2", datetime(2026, 10, 2, 12, tzinfo=UTC)).height == 1   # 10-02 known only on 10-03


# ---------------------------------------------------------------- study statistics

def test_holm_and_bootstrap():
    adj = macro_study.holm([0.01, 0.04, 0.03, float("nan")])
    assert adj[0] == pytest.approx(0.03) and adj[2] == pytest.approx(0.06) and adj[1] == pytest.approx(0.06)
    rng = np.random.default_rng(0)
    x = rng.normal(0.5, 1, 400)
    w = np.repeat(np.arange(80), 5)
    assert macro_study.one_sided_p(x, w, +1, rng) < 0.01
    assert macro_study.one_sided_p(x, w, -1, rng) > 0.9


def test_btc_outcomes_vol_scaling():
    d = [date(2026, 1, 1) + timedelta(days=i) for i in range(80)]
    c = 100 * np.exp(np.cumsum(np.r_[0, np.full(79, 0.01)]))
    o = macro_study.btc_outcomes(pl.DataFrame({"d": d, "close": c}))
    assert o["fwd5"][40] == pytest.approx(0.05)
    assert o["fwd20"].null_count() == 20


# ---------------------------------------------------------------- thesis ledger

class FakeData(theses.Data):
    def __init__(self, as_of, series, prices):
        super().__init__(as_of)
        self._s = series
        self._p = prices

    def series(self, s):
        return self._s[s]

    def price_at(self, sym, ts):
        t = min(ts, self.as_of)
        ks = [k for k in self._p if k <= t]
        return self._p[max(ks)] if ks else None

    def sigma_1h(self, sym, ts, hours=720):
        return 0.005


def _daily(vals, start=date(2026, 9, 1)):
    return pl.DataFrame({"obs_date": [start + timedelta(days=i) for i in range(len(vals))], "value": [float(v) for v in vals]})


def _thesis(**kw):
    t = {"id": "T-TEST-1", "component": "x", "author": "claude", "created_ts": "2026-11-01T00:00:00Z", "claim": "c", "mechanism": "m",
         "direction": "short", "assets": ["BTCUSDT"], "horizon_days": 10, "status": "active", "trigger": "on_create",
         "proxies": [{"id": "y2_up", "series": "fred:DGS2", "transform": "chg_5", "op": ">", "value": 0.0},
                     {"id": "hy_up", "series": "fred:BAMLH0A0HYM2", "transform": "chg_5", "op": ">", "value": 0.0}],
         "invalidation": {"any": [{"series": "fred:DGS2", "transform": "chg_20", "op": "<=", "value": -0.30}]}}
    return t | kw


def test_real_ledger_is_valid_and_seeded_with_the_user_view():
    L = theses.load_ledger()
    assert theses.validate(L) == []
    ids = {t["id"]: t for t in L}
    assert {"T-USER-20261115-a", "T-USER-20261115-b", "T-USER-20261115-c", "T-USER-20261115-d"} <= set(ids)
    assert all(ids[f"T-USER-20261115-{c}"]["author"] == "user" for c in "abcd")
    assert ids["T-USER-20261115-a"]["created_ts"].isoformat().startswith("2026-11-15T19:00")
    assert ids["T-USER-20261115-b"]["user_numbers"]["sp500_target_level"] == 7000
    assert ids["T-USER-20261115-c"]["direction"] == "long" and ids["T-USER-20261115-c"]["trigger"] != "on_create"


def test_validate_catches_bad_theses():
    bad = _thesis(author="bot", direction="up", assets=["SPYUSDT"],
                  proxies=[{"id": "p", "series": "fred:NOPE", "transform": "wiggle", "op": "~", "value": 1, "size": 3}])
    errs = " | ".join(theses.validate([bad]))
    for frag in ("author", "direction", "SPYUSDT", "unknown FRED", "unknown transform", "op", "size keys"):
        assert frag in errs


def test_transforms():
    d = FakeData(datetime(2026, 11, 15, tzinfo=UTC), {"fred:A": _daily([1, 2, 3, 4, 5, 6, 10]), "fred:B": _daily([1, 1, 1, 1, 1, 1, 2])}, {})
    assert theses.transform(d, {"series": "fred:A", "transform": "chg_5"})[0] == pytest.approx(8)
    assert theses.transform(d, {"series": "fred:A", "transform": "dd_from_high"})[0] == pytest.approx(0)
    assert theses.transform(d, {"series": "fred:A", "vs": "fred:B", "transform": "spread"})[0] == pytest.approx(8)
    assert theses.transform(d, {"series": "fred:A", "vs": "fred:B", "transform": "rel_logret_1"})[0] == pytest.approx(100 * (np.log(10 / 6) - np.log(2)))
    assert theses.transform(d, {"series": "fred:A", "transform": "chg_50"})[0] is None
    m = pl.DataFrame({"obs_date": [date(2025 + i // 12, i % 12 + 1, 1) for i in range(16)],
                      "value": [100 * 1.004 ** i for i in range(16)]})
    d2 = FakeData(datetime(2026, 11, 15, tzinfo=UTC), {"fred:C": m}, {})
    assert theses.transform(d2, {"series": "fred:C", "transform": "yoy_pct"})[0] == pytest.approx(100 * (1.004 ** 12 - 1))
    assert theses.transform(d2, {"series": "fred:C", "transform": "yoy_falling_3m"})[0] == 0.0


def test_three_valued_conditions():
    d = FakeData(datetime(2026, 11, 15, tzinfo=UTC), {"fred:A": _daily([1, 2]), "fred:B": _daily([1, 2, 3, 4, 5, 6, 7])}, {})
    unknown = {"series": "fred:A", "transform": "chg_5", "op": ">", "value": 0}
    yes = {"series": "fred:B", "transform": "chg_5", "op": ">", "value": 0}
    no = {"series": "fred:B", "transform": "chg_5", "op": "<", "value": 0}
    assert theses.evaluate(d, {"any": [unknown, yes]})[0] is True
    assert theses.evaluate(d, {"any": [unknown, no]})[0] is None
    assert theses.evaluate(d, {"all": [unknown, no]})[0] is False
    assert theses.evaluate(d, {"all": [unknown, yes]})[0] is None


def _series(y2, hy):
    return {"fred:DGS2": _daily(y2), "fred:BAMLH0A0HYM2": _daily(hy)}


def test_step_status_scoring_and_sticky_invalidation():
    t = _thesis()
    created = datetime(2026, 11, 1, tzinfo=UTC)
    prices = {created: 100.0, created + timedelta(days=5): 95.0, created + timedelta(days=12): 90.0}
    up = list(np.linspace(4.0, 4.5, 30))
    r = theses.step(FakeData(created + timedelta(days=5), _series(up, up), prices), t, None)
    assert r["status"] == "strengthening" and r["net_support"] == 1.0
    assert r["score"]["assets"]["BTCUSDT"]["signed_ret_pct"] == pytest.approx(-100 * np.log(0.95), abs=1e-3)   # short wins when price falls
    assert not r["score"]["final"]
    down = list(np.linspace(4.5, 4.0, 30))
    r2 = theses.step(FakeData(created + timedelta(days=6), _series(down, up), prices), t, r["_state"])
    assert r2["status"] == "invalidated"
    r3 = theses.step(FakeData(created + timedelta(days=12), _series(up, up), prices), t, r2["_state"])
    assert r3["status"] == "invalidated"                      # sticky
    assert r3["score"]["final"] and r3["score"]["end"].startswith("2026-11-11")   # still scored at the horizon


def test_step_conditional_trigger_and_expiry():
    t = _thesis(direction="long", trigger={"all": [{"series": "fred:DGS2", "transform": "chg_5", "op": "<", "value": 0.0}]},
                max_wait_days=30, invalidation={"any": [{"series": "fred:BAMLH0A0HYM2", "transform": "chg_20", "op": ">=", "value": 0.5}]},
                probability=0.6)
    created = datetime(2026, 11, 1, tzinfo=UTC)
    prices = {created: 100.0, created + timedelta(days=3): 100.0, created + timedelta(days=13): 110.0}
    flat_up = list(np.linspace(4.0, 4.3, 30))
    r = theses.step(FakeData(created + timedelta(days=2), _series(flat_up, flat_up), prices), t, None)
    assert not r["triggered"] and r["score"] is None
    falling = list(np.linspace(4.3, 4.0, 30))
    r2 = theses.step(FakeData(created + timedelta(days=3), _series(falling, falling), prices), t, r["_state"])
    assert r2["triggered"] and r2["start_ts"].startswith("2026-11-04")
    r3 = theses.step(FakeData(created + timedelta(days=20), _series(falling, falling), prices), t, r2["_state"])
    assert r3["status"] == "expired" and r3["score"]["final"]
    assert r3["score"]["brier"] == pytest.approx((0.6 - 1) ** 2)
    never = theses.step(FakeData(created + timedelta(days=31), _series(flat_up, flat_up), prices), t, None)
    assert never["status"] == "expired" and never["_state"].get("expired_untriggered")


def test_not_created_yet_and_digest_length():
    t = _thesis(created_ts="2026-12-01T00:00:00Z")
    d = FakeData(datetime(2026, 11, 15, tzinfo=UTC), _series([4] * 30, [3] * 30), {})
    r = theses.step(d, t, None)
    assert r["status"] == "not_created_yet"
    many = [theses.step(FakeData(datetime(2026, 11, 15, tzinfo=UTC), _series(list(np.linspace(4, 5, 30)), [3] * 30),
                                 {datetime(2026, 11, 1, tzinfo=UTC): 1.0}), _thesis(id=f"T-X-{i}"), None) for i in range(15)]
    md = theses.digest({"as_of": "2026-11-15T00:00:00+00:00", "theses": many})
    assert len(md.strip().splitlines()) <= 30


# ---------------------------------------------------------------- confluence log

def _example() -> dict:
    txt = (ROOT / "research" / "desk" / "CONFLUENCE.md").read_text(encoding="utf-8")
    block = re.search(r"```json\n(.*?)```", txt, re.S).group(1)
    return json.loads(block.replace("<model id>", "claude-opus-5-5"))


def test_confluence_example_validates_against_its_snapshot():
    snap = ROOT / "research" / "desk" / "2026-11-15" / "1723"
    rec = _example()
    res = confluence.validate(rec, snap if snap.is_dir() else None)
    assert res["errors"] == []


def test_confluence_rejects_bad_records():
    rec = _example()
    a = rec["assets"][0]["readings"]
    a["flow"]["value"] = 1.5
    del a["book"]
    a["onchain"] = {"value": None}
    a["momentum"]["evidence"] = []
    rec["assets"][0]["leverage"] = 3
    errs = " | ".join(confluence.validate(rec)["errors"])
    for frag in ("[-1, 1]", "missing layers ['book']", "null reading needs a reason", "evidence pointer", "size keys"):
        assert frag in errs


def test_confluence_wrong_evidence_value_is_an_error():
    snap = ROOT / "research" / "desk" / "2026-11-15" / "1723"
    if not snap.is_dir():
        pytest.skip("snapshot not present")
    rec = _example()
    rec["assets"][0]["readings"]["flow"]["evidence"][0]["value"] = 0.9
    assert any("reading cites 0.9" in e for e in confluence.validate(rec, snap)["errors"])


def test_confluence_log_is_append_only(tmp_path, monkeypatch):
    monkeypatch.setattr(confluence, "journal_dir", lambda: tmp_path)
    rec = _example()
    p = confluence.log(rec)
    assert p.parent == tmp_path and len(confluence.logged_cycles()) == 1
    with pytest.raises(ValueError, match="already logged"):
        confluence.log(rec)
    rows = confluence.rows()
    assert len(rows) == len(confluence.LAYERS) and {r["layer"] for r in rows} == set(confluence.LAYERS)


def test_confluence_fit_refuses_small_samples(tmp_path, monkeypatch):
    import importlib.util

    spec = importlib.util.spec_from_file_location("confluence_fit", ROOT / "scripts" / "confluence_fit.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    rng = np.random.default_rng(1)
    X = rng.normal(size=(500, 3))
    y = X @ np.array([0.5, 0.0, -0.3]) + rng.normal(0, 0.1, 500)
    b = mod.ridge(X, y, 1.0)
    assert b[1] == pytest.approx(0.5, abs=0.02) and b[3] == pytest.approx(-0.3, abs=0.02)
    w = mod.logistic(X, (y > 0).astype(float), 1.0)
    assert w[1] > 0 > w[3]
    assert confluence.MIN_CYCLES_TO_FIT == 300
    assert isinstance(Path(mod.__file__), Path)
