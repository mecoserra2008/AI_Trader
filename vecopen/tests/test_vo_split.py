"""Phase 1: research / holdout split, the holdout lock, and acceptance checks on the real split.

The real-data tests never read, list or sample holdout contents: they only check that opening a holdout path
named in ``split.json`` raises ``PermissionError`` (a file that opened would be closed unread and the test fails).
"""

from __future__ import annotations

import json
import os
import subprocess
from datetime import UTC, datetime, timedelta

import polars as pl
import pytest
from vecopen.bars import load_sessions, resample

from vecopen import gaps as G
from vecopen import split as S
from vecopen.data import GAPS_PATH, HOLDOUT_DIR, SPLIT_PATH, TS, HoldoutAccessError, load_bars

SESS = load_sessions()
windows_only = pytest.mark.skipif(os.name != "nt", reason="icacls deny ACL is Windows-specific")
needs_split = pytest.mark.skipif(not SPLIT_PATH.exists(), reason="no split has been run yet")


def _split() -> dict:
    return json.loads(SPLIT_PATH.read_text(encoding="utf-8"))


# ---------------------------------------------------------------- boundary

def test_boundary_is_75_percent_floored_to_trade_day_start():
    utc = SESS.for_asset("BTCUSDT")
    a, b = datetime(2020, 1, 1, tzinfo=UTC), datetime(2024, 1, 1, tzinfo=UTC)
    got = S.split_boundary(a, b, utc, 0.75)
    raw = a + (b - a) * 0.75
    assert got <= raw < got + timedelta(days=1) and (got.hour, got.minute) == (0, 0)
    fx = SESS.for_asset("EURUSD")
    g2 = S.split_boundary(a, b, fx, 0.75)
    from zoneinfo import ZoneInfo
    assert g2.astimezone(ZoneInfo("America/New_York")).strftime("%H:%M") == "17:00" and g2 <= raw


def test_lift_command_is_documented_not_run():
    cmd = S.lift_command()
    assert cmd.startswith("icacls ") and "/remove:d" in cmd and str(HOLDOUT_DIR.resolve()) in cmd


# ---------------------------------------------------------------- lock on a temporary holdout

def _plain_acl(d) -> None:
    """pytest's tmp folders come from mkdir(mode=0o700), whose ACL has an OWNER RIGHTS entry; give the test folder
    the same shape as vecopen/data (explicit grants, no OWNER RIGHTS) so the deny can be lifted afterwards."""
    user = os.environ["USERNAME"]
    subprocess.run(["icacls", str(d), "/inheritance:r", "/grant:r", f"{user}:(OI)(CI)F", "/grant:r",
                    "*S-1-5-18:(OI)(CI)F"], capture_output=True, check=True)


@windows_only
def test_lock_refuses_a_folder_with_owner_rights(tmp_path):
    hold = tmp_path / "holdout"  # created under pytest's 0o700 tmp folder: inherits OWNER RIGHTS
    hold.mkdir()
    if ";OW)" not in S.folder_sddl(hold):
        pytest.skip("this Python does not write OWNER RIGHTS for 0o700 folders")
    with pytest.raises(RuntimeError, match="OWNER RIGHTS"):
        S.lock_holdout(hold)
    os.listdir(hold)  # still readable: nothing was applied


@pytest.fixture
def locked(tmp_path):
    base = tmp_path / "data"
    base.mkdir()
    _plain_acl(base)
    hold = base / "holdout"
    S.lock_holdout(hold)
    yield hold
    # lift the deny on this temporary folder only, so pytest can delete it
    subprocess.run(["icacls", str(hold), "/remove:d", os.environ["USERNAME"], "/T", "/C"], capture_output=True, check=False)


def _frame(n: int, start: datetime) -> pl.DataFrame:
    ts = [start + (i + 1) * timedelta(minutes=5) for i in range(n)]
    return pl.DataFrame({"ts_close": ts, "o": 1.0, "h": 2.0, "l": 0.5, "c": 1.5, "v": 3.0}).with_columns(
        pl.col("ts_close").cast(TS))


@windows_only
def test_holdout_unreadable_after_write_and_new_parts_still_writable(locked):
    df = _frame(20, datetime(2024, 1, 1, tzinfo=UTC))
    info = S.write_holdout_part("XUSDT", df, 0, holdout=locked)
    p = locked.parent / info["path"]
    with pytest.raises(PermissionError):
        open(p, "rb").close()
    with pytest.raises(PermissionError):
        pl.read_parquet(p)
    with pytest.raises(PermissionError):
        os.listdir(locked)
    with pytest.raises(PermissionError):
        os.listdir(locked / "XUSDT")
    with pytest.raises(HoldoutAccessError):
        load_bars("XUSDT", path=p)
    # appending to an existing file or overwriting it is impossible; new data goes into a new part file
    with pytest.raises(PermissionError):
        open(p, "ab").close()
    with pytest.raises((FileExistsError, PermissionError)):
        S.write_holdout_part("XUSDT", df, 0, holdout=locked)
    later = _frame(5, datetime(2024, 2, 1, tzinfo=UTC))
    S.lock_holdout(locked)  # re-locking an already locked folder (the later CFD split does this) works
    info1 = S.write_holdout_part("XUSDT", later, 1, holdout=locked)
    S.write_holdout_part("NEWCFD", later, 0, holdout=locked)  # a new asset folder inside the locked folder
    S.write_holdout_json("XUSDT", "gaps.json", {"runs": []}, holdout=locked)
    # lift this temporary lock and check the bytes written blind are intact
    subprocess.run(["icacls", str(locked), "/remove:d", os.environ["USERNAME"], "/T", "/C"], capture_output=True, check=True)
    assert pl.read_parquet(p).equals(df)
    assert pl.read_parquet(locked.parent / info1["path"]).equals(later)


# ---------------------------------------------------------------- the real split (after `phase1`)

@needs_split
@windows_only
def test_real_holdout_is_unreadable():
    with pytest.raises(PermissionError):
        os.listdir(HOLDOUT_DIR)
    sp = _split()
    assert sp["assets"]
    for a, rec in sp["assets"].items():
        for part in rec["holdout"]["parts"]:
            p = HOLDOUT_DIR.parent / part["path"]
            S.assert_unreadable(p)  # opening for read raises PermissionError
            with pytest.raises(HoldoutAccessError):
                load_bars(a, path=p)
        with pytest.raises(HoldoutAccessError):
            load_bars(a, root=HOLDOUT_DIR / a)


@needs_split
def test_real_research_files_unique_sorted_and_before_boundary():
    for a, rec in _split()["assets"].items():
        df = load_bars(a)  # validates schema, 5m grid, strictly increasing ts_close (no duplicates)
        assert df.height == rec["research"]["rows"]
        assert df["ts_close"].is_unique().all()
        b = datetime.fromisoformat(rec["boundary_utc"])
        assert df["ts_close"].max() <= b
        assert abs(rec["research_fraction_actual"] - 0.75) < 0.01
        if SESS.assets[a]["class"] == "crypto":
            assert {"buy_v", "sell_v"} <= set(df.columns)
            assert ((df["buy_v"] + df["sell_v"] - df["v"]).abs() <= 1e-9 * (1 + df["v"])).all() or \
                (df["sell_v"] >= 0).all()


@needs_split
def test_real_research_gaps_match_gaps_json():
    """Runs and classes recomputed from data/research alone equal those written at split time."""
    from vecopen.phase1 import research_gap_check

    doc = json.loads(GAPS_PATH.read_text(encoding="utf-8"))
    for a in _split()["assets"]:
        runs = G.missing_runs(load_bars(a, columns=["ts_close"])["ts_close"])
        stored = doc["assets"][a]["runs"]
        assert [t.isoformat() for t in runs["gap_first"]] == [r["gap_first"] for r in stored]
    res = research_gap_check()
    assert res["match"], res


@needs_split
def test_real_resample_equals_aggregation_on_a_research_slice():
    from test_vo_bars import oracle_resample

    sp = _split()["assets"]
    a = "BTCUSDT" if "BTCUSDT" in sp else next(iter(sp))
    df = load_bars(a)
    sl = df.slice(df.height // 3, 12 * 24 * 40)  # 40 days from the middle of the research sample
    rows = sl.to_dicts()
    s = SESS.for_asset(a)
    for tf in ("15m", "1h", "4h", "1d", "1w"):
        got = resample(sl, tf, s).to_dicts()
        want = oracle_resample(rows, tf, s)
        assert len(got) == len(want)
        for g, w in zip(got, want):
            for k in ("ts_open", "ts_close", "o", "h", "l", "c", "n_base"):
                assert g[k] == w[k]
            for k in ("v", "buy_v", "sell_v"):
                assert g[k] == pytest.approx(w[k], rel=1e-12, abs=1e-9)


# ---------------------------------------------------------------- end to end on synthetic staged data

@windows_only
def test_build_split_end_to_end_on_synthetic_archive(tmp_path, monkeypatch, request):
    from vecopen import data as D
    from vecopen import ingest as I
    from vecopen import phase1 as P

    base = tmp_path / "data"
    base.mkdir()
    _plain_acl(base)
    request.addfinalizer(lambda: subprocess.run(["icacls", str(base / "holdout"), "/remove:d", os.environ["USERNAME"],
                                                 "/T", "/C"], capture_output=True, check=False))
    for name, sub in [("RESEARCH_DIR", "research"), ("HOLDOUT_DIR", "holdout"), ("STAGING_DIR", "staging")]:
        monkeypatch.setattr(D, name, base / sub)
    for name, f in [("SPLIT_PATH", "split.json"), ("GAPS_PATH", "gaps.json"), ("MANIFEST_PATH", "manifest.json")]:
        monkeypatch.setattr(D, name, base / f)
    monkeypatch.setattr(I, "MANIFEST_PATH", base / "manifest.json")
    monkeypatch.setattr(I, "DATA_DIR", base)

    t0 = datetime(2024, 1, 1, tzinfo=UTC)
    outage = (datetime(2024, 1, 10, 3, tzinfo=UTC), datetime(2024, 1, 10, 4, tzinfo=UTC))   # all four symbols
    single = (datetime(2024, 1, 12, 7, tzinfo=UTC), datetime(2024, 1, 12, 7, 25, tzinfo=UTC))  # BTC only
    man = I.Manifest()
    assets = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT"]
    for k, a in enumerate(assets):
        ts = pl.datetime_range(t0 + timedelta(minutes=5), datetime(2024, 3, 1, tzinfo=UTC), "5m", eager=True,
                               time_unit="ms", time_zone="UTC")
        keep = ~ts.is_between(*outage)
        if a == "BTCUSDT":
            keep &= ~ts.is_between(*single)
        ts = ts.filter(keep)
        df = pl.DataFrame({"ts_close": ts}).with_columns(o=pl.lit(10.0 + k), h=pl.lit(11.0 + k), l=pl.lit(9.0 + k),
                                                         c=pl.lit(10.5 + k), v=pl.lit(2.0), buy_v=pl.lit(1.5),
                                                         sell_v=pl.lit(0.5)).with_columns(pl.col("ts_close").cast(TS))
        for month, part in df.group_by(pl.col("ts_close").dt.strftime("%Y-%m"), maintain_order=True):
            I.write_parquet_atomic(part, base / "staging" / "binance" / a / f"{month[0]}.parquet")
        # a daily file duplicating the last day with a conflicting close: the monthly rows must win
        dup = df.tail(288).with_columns(c=pl.lit(-1.0))
        I.write_parquet_atomic(dup, base / "staging" / "binance" / a / "2024-02-29.parquet")
        man.assets[a] = {"source": "binance_usdm_archive", "status": "staged"}
    man.save()

    res = S.build_split(assets, source="binance", log=lambda *_: None)
    assert res["split"] == assets
    sp = json.loads((base / "split.json").read_text())
    for a in assets:
        rec = sp["assets"][a]
        b = datetime.fromisoformat(rec["boundary_utc"])
        assert (b.hour, b.minute) == (0, 0) and abs(rec["research_fraction_actual"] - 0.75) < 0.02
        assert rec["dedup"]["conflicting"] == 288
        r = load_bars(a)
        assert r["ts_close"].max() <= b and (r["c"] > 0).all()
        assert rec["research"]["rows"] + rec["holdout"]["parts"][0]["rows"] == rec["rows_total"]
        S.assert_unreadable(base / rec["holdout"]["parts"][0]["path"])
        assert not (base / "staging" / "binance" / a).exists()
    with pytest.raises(PermissionError):
        os.listdir(base / "holdout")
    g = json.loads((base / "gaps.json").read_text())
    assert [x["class"] for x in g["assets"]["BTCUSDT"]["runs"]] == ["exchange_outage", "archive_gap"]
    assert [x["class"] for x in g["assets"]["ETHUSDT"]["runs"]] == ["exchange_outage"]
    assert P.research_gap_check()["match"]
    # a second call does nothing: assets already split are never rebuilt
    assert S.build_split(assets, source="binance", log=lambda *_: None) == {"split": []}
    tables, metrics = P.build_tables()
    assert "| BTCUSDT |" in tables and metrics["assets"]["BTCUSDT"]["research_gaps"]["archive_gap"]["runs"] == 1


@windows_only
def test_cfd_path_stage_and_split_fx_with_weekends(tmp_path, monkeypatch, request):
    """cTrader trend-bar files (schema of at-ctrader store::write_trendbars) -> staging -> split, tick volume only,
    gaps classed against the symbol's cTrader schedule."""
    from vecopen import data as D
    from vecopen import ingest as I

    base = tmp_path / "data"
    base.mkdir()
    _plain_acl(base)
    request.addfinalizer(lambda: subprocess.run(["icacls", str(base / "holdout"), "/remove:d", os.environ["USERNAME"],
                                                 "/T", "/C"], capture_output=True, check=False))
    for name, sub in [("RESEARCH_DIR", "research"), ("HOLDOUT_DIR", "holdout"), ("STAGING_DIR", "staging")]:
        monkeypatch.setattr(D, name, base / sub)
    for name, f in [("SPLIT_PATH", "split.json"), ("GAPS_PATH", "gaps.json"), ("MANIFEST_PATH", "manifest.json")]:
        monkeypatch.setattr(D, name, base / f)
    monkeypatch.setattr(I, "MANIFEST_PATH", base / "manifest.json")
    monkeypatch.setattr(I, "DATA_DIR", base)
    monkeypatch.setattr(I, "STAGING_DIR", base / "staging")
    monkeypatch.setattr(I, "CTRADER_RAW", base / "staging" / "ctrader_raw")

    # FX trades Sun 22:00 UTC .. Fri 22:00 UTC (winter); one feed gap on a Wednesday
    opens = pl.datetime_range(datetime(2024, 1, 7, 22, tzinfo=UTC), datetime(2024, 2, 29, 21, 55, tzinfo=UTC), "5m",
                              eager=True, time_unit="ms", time_zone="UTC")
    wd, hr = opens.dt.weekday(), opens.dt.hour()
    live = ((wd <= 4) | ((wd == 5) & (hr < 22)) | ((wd == 7) & (hr >= 22))) & ~opens.is_between(
        datetime(2024, 1, 17, 9, tzinfo=UTC), datetime(2024, 1, 17, 9, 55, tzinfo=UTC))
    opens = opens.filter(live)
    raw = pl.DataFrame({"ts_open": opens}).with_columns(o=pl.lit(1.1), h=pl.lit(1.2), l=pl.lit(1.0), c=pl.lit(1.15),
                                                        tick_volume=pl.lit(42, pl.Int64), known_ts=pl.col("ts_open"))
    for month, part in raw.group_by(pl.col("ts_open").dt.strftime("%Y-%m"), maintain_order=True):
        (base / "staging" / "ctrader_raw" / "EURUSD" / "M5").mkdir(parents=True, exist_ok=True)
        part.write_parquet(base / "staging" / "ctrader_raw" / "EURUSD" / "M5" / f"{month[0]}.parquet")
    man = I.Manifest()
    # schedule in cTrader form: seconds from Sunday 00:00 UTC, Sun 22:00 -> Fri 22:00
    man.d["ctrader"]["map"] = {"map": {"EURUSD": {"ctrader_name": "EURUSD", "ctrader_id": 1, "details": {
        "schedule_time_zone": "UTC", "schedule": [[22 * 3600, 5 * 86_400 + 22 * 3600]], "holidays": []}}}}
    man.assets["EURUSD"] = {"ctrader_name": "EURUSD", "source": "ctrader_trendbars_blackbull"}
    assert I.stage_ctrader("EURUSD", man) == 2
    res = S.build_split(["EURUSD"], source="ctrader", log=lambda *_: None)
    assert res["split"] == ["EURUSD"]
    r = load_bars("EURUSD")
    assert "buy_v" not in r.columns and set(r["v"].unique()) == {42.0}
    assert r["ts_close"][0] == datetime(2024, 1, 7, 22, 5, tzinfo=UTC)
    g = json.loads((base / "gaps.json").read_text())["assets"]["EURUSD"]
    classes = [x["class"] for x in g["runs"]]
    assert classes.count("feed_gap") == 1 and set(classes) == {"weekend", "feed_gap"} and g["schedule"] == "ctrader"
    b = datetime.fromisoformat(json.loads((base / "split.json").read_text())["assets"]["EURUSD"]["boundary_utc"])
    from zoneinfo import ZoneInfo
    assert b.astimezone(ZoneInfo("America/New_York")).strftime("%H:%M") == "17:00"
    assert not (base / "staging" / "ctrader_raw" / "EURUSD").exists()


def test_ctrader_keys_detection(monkeypatch):
    from vecopen import ingest as I

    class R:
        def __init__(self, out, rc):
            self.stdout, self.returncode = out, rc
    missing = "cTrader keys in .env (values never shown):\n  CTRADER_CLIENT_ID        missing\n  CTRADER_ENV  missing\n"
    present = ("  CTRADER_CLIENT_ID        present\n  CTRADER_ENV              present\n"
               "  CTRADER_REFRESH_TOKEN    missing  (optional)\n")
    monkeypatch.setattr(I.subprocess, "run", lambda *a, **k: R(missing, 1))
    assert not I.ctrader_keys_present()
    monkeypatch.setattr(I.subprocess, "run", lambda *a, **k: R(present, 0))
    assert I.ctrader_keys_present()


@pytest.mark.parametrize("name, desc, roll", [
    ("USOIL", "US Crude Oil Spot", "none"),
    ("XTIUSD", "WTI Crude Oil", "none"),
    ("USOIL.F", "WTI Crude Oil Futures", "panama"),
    ("CL_Z6", "Crude oil Dec 2026 contract", "panama"),
    ("UKOIL", "Brent (rolling futures based)", "panama"),
    ("BRENT", None, "none"),
])
def test_roll_decision_from_metadata_only(name, desc, roll):
    from vecopen.ingest import roll_decision
    assert roll_decision(name, desc)["roll"] == roll


def test_build_refuses_undecided_futures_based_roll():
    from vecopen.ingest import Manifest
    m = Manifest.__new__(Manifest)
    m.d = {"assets": {"USOIL": {}}}
    with pytest.raises(RuntimeError, match="metadata decision"):
        S.effective_roll("USOIL", SESS, m)
    m.d["assets"]["USOIL"]["roll_decision"] = {"roll": "panama"}
    assert S.effective_roll("USOIL", SESS, m) == "panama"
    assert S.effective_roll("EURUSD", SESS, m) == "none" and S.effective_roll("BUND", SESS, m) == "panama"
