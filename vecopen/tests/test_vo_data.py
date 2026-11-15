"""Phase 1: load_bars holdout guard, duplicate rule, Panama roll adjustment, archive parsing and ingestion."""

from __future__ import annotations

import hashlib
import io
import json
import zipfile
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import polars as pl
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from vecopen import data as D
from vecopen import ingest as I
from vecopen.data import HOLDOUT_DIR, HoldoutAccessError, dedup_bars, load_bars, panama_adjust

T0 = datetime(2024, 1, 1, tzinfo=UTC)
FIVE = timedelta(minutes=5)


def bars(n: int, start: datetime = T0, price: float = 100.0, step: float = 0.1) -> pl.DataFrame:
    ts = [start + (i + 1) * FIVE for i in range(n)]
    o = [price + i * step for i in range(n)]
    return pl.DataFrame({"ts_close": ts, "o": o, "h": [x + 0.5 for x in o], "l": [x - 0.5 for x in o],
                         "c": [x + step for x in o], "v": [10.0] * n}).with_columns(pl.col("ts_close").cast(D.TS))


# ---------------------------------------------------------------- holdout guard

@pytest.mark.parametrize("make", [
    lambda: dict(path=HOLDOUT_DIR / "BTCUSDT" / "part-0000.parquet"),
    lambda: dict(path=HOLDOUT_DIR / "nonexistent.parquet"),
    lambda: dict(root=HOLDOUT_DIR),
    lambda: dict(root=HOLDOUT_DIR / "BTCUSDT"),
    lambda: dict(path=D.RESEARCH_DIR / ".." / "holdout" / "BTCUSDT.parquet"),
    lambda: dict(path=Path("vecopen") / "data" / "holdout" / "x.parquet"),
    lambda: dict(path=str(HOLDOUT_DIR).upper() + "\\X.parquet"),
])
def test_load_bars_refuses_holdout_paths_before_touching_them(make, monkeypatch):
    def boom(*a, **k):
        raise AssertionError("read_parquet must not be called for a holdout path")
    monkeypatch.setattr(pl, "read_parquet", boom)
    with pytest.raises(HoldoutAccessError):
        load_bars("BTCUSDT", **make())


def test_holdout_guard_also_catches_other_data_holdout_folders(tmp_path):
    assert D.is_holdout_path(tmp_path / "data" / "holdout" / "a.parquet")
    assert not D.is_holdout_path(tmp_path / "data" / "research" / "a.parquet")
    assert not D.is_holdout_path(D.RESEARCH_DIR / "BTCUSDT.parquet")
    assert issubclass(HoldoutAccessError, PermissionError)


def test_load_bars_reads_and_validates_research_files(tmp_path):
    df = bars(50).with_columns(buy_v=pl.lit(4.0), sell_v=pl.lit(6.0))
    df.write_parquet(tmp_path / "X.parquet")
    got = load_bars("X", root=tmp_path)
    assert got.equals(df)
    assert load_bars("X", root=tmp_path, columns=["c"]).columns == ["ts_close", "c"]
    pl.concat([df, df.tail(1)]).write_parquet(tmp_path / "DUP.parquet")
    with pytest.raises(ValueError, match="strictly increasing"):
        load_bars("DUP", root=tmp_path)
    df.drop("v").write_parquet(tmp_path / "NOV.parquet")
    with pytest.raises(ValueError, match="missing column v"):
        load_bars("NOV", root=tmp_path)


# ---------------------------------------------------------------- duplicates

def test_dedup_rule_monthly_beats_daily_and_later_row_wins():
    a = bars(5)
    daily = a.slice(3, 2).with_columns(c=pl.col("c") + 9.0, _src_rank=pl.lit(1, pl.Int32))  # conflicting copy
    revised = a.slice(1, 1).with_columns(c=pl.lit(-1.0), _src_rank=pl.lit(0, pl.Int32))     # later monthly row
    exact = a.slice(0, 1).with_columns(_src_rank=pl.lit(0, pl.Int32))                         # identical copy
    df = pl.concat([a.with_columns(_src_rank=pl.lit(0, pl.Int32)), daily, revised, exact]).with_row_index("_row")
    out, rep = dedup_bars(df)
    assert out.columns == a.columns
    assert out["ts_close"].to_list() == a["ts_close"].to_list()
    assert out["c"][1] == -1.0                      # same source: the row written last wins
    assert out["c"][3] == a["c"][3]                 # monthly beats the daily copy
    assert (rep.rows_in, rep.rows_out, rep.identical, rep.conflicting) == (9, 5, 1, 3)


@settings(max_examples=60, deadline=None)
@given(st.lists(st.integers(0, 40), min_size=1, max_size=200))
def test_no_duplicate_timestamps_after_dedup(idx):
    df = pl.DataFrame({"ts_close": [T0 + (i + 1) * FIVE for i in idx], "o": [float(i) for i in range(len(idx))],
                       "h": 1.0, "l": 0.0, "c": 0.5, "v": 1.0}).with_columns(pl.col("ts_close").cast(D.TS))
    out, rep = dedup_bars(df)
    assert out["ts_close"].is_unique().all() and out["ts_close"].is_sorted()
    assert out.height == len(set(idx)) == rep.rows_out
    assert rep.identical + rep.conflicting == len(idx) - len(set(idx))
    # the kept row is the last one written for that timestamp
    last = {}
    for k, i in enumerate(idx):
        last[i] = float(k)
    assert out["o"].to_list() == [last[i] for i in sorted(last)]


# ---------------------------------------------------------------- roll adjustment

def test_panama_back_adjustment_preserves_distances_and_keeps_unadjusted():
    old = bars(100, T0, 150.0, 0.01)
    new = bars(100, T0 + 100 * FIVE, 148.0, 0.01)  # next contract trades 2.0 lower (roughly)
    df = pl.concat([old, new])
    roll = T0 + 100 * FIVE  # first bar opening at or after this is the new contract
    out, ap = panama_adjust(df, pl.DataFrame({"roll_ts": [roll]}).with_columns(pl.col("roll_ts").cast(D.TS)))
    gap = new["o"][0] - old["c"][-1]
    assert ap["gap"][0] == pytest.approx(gap) and ap["gap_source"][0] == "measured"
    # latest contract unadjusted; history shifted by the gap; no jump at the roll
    assert out.slice(100)["o"].to_list() == new["o"].to_list()
    assert out.slice(0, 100)["o"].to_list() == pytest.approx([x + gap for x in old["o"]])
    assert out["o"][100] - out["c"][99] == pytest.approx(0.0)
    # point distances inside a contract are preserved; unadjusted prices kept
    assert (out["h"] - out["l"]).to_list() == pytest.approx((df["h"] - df["l"]).to_list())
    assert out["c_unadj"].to_list() == df["c"].to_list()
    assert (out["o"] - out["o_unadj"] - out["roll_offset"]).abs().max() < 1e-12


def test_panama_two_rolls_and_given_gap():
    a, b, c = bars(10, T0, 100.0), bars(10, T0 + 10 * FIVE, 90.0), bars(10, T0 + 20 * FIVE, 80.0)
    df = pl.concat([a, b, c])
    rolls = pl.DataFrame({"roll_ts": [T0 + 10 * FIVE, T0 + 20 * FIVE], "gap": [-10.0, None]})
    out, ap = panama_adjust(df, rolls.with_columns(pl.col("roll_ts").cast(D.TS)))
    g2 = c["o"][0] - b["c"][-1]
    assert ap["gap"].to_list() == pytest.approx([-10.0, g2])
    assert out["roll_offset"].to_list() == pytest.approx([-10.0 + g2] * 10 + [g2] * 10 + [0.0] * 10)
    with pytest.raises(ValueError, match="outside"):
        panama_adjust(df, pl.DataFrame({"roll_ts": [T0 + 99 * FIVE]}).with_columns(pl.col("roll_ts").cast(D.TS)))


def test_roll_calendar_required_for_panama_assets(tmp_path):
    with pytest.raises(FileNotFoundError, match="roll calendar"):
        D.read_roll_calendar("BUND", tmp_path)
    (tmp_path / "BUND.csv").write_text("roll_ts_utc,gap,source\n2024-03-07T22:00:00Z,,test\n2024-06-06T22:00:00Z,-0.5,test\n")
    cal = D.read_roll_calendar("BUND", tmp_path)
    assert cal["roll_ts"].dtype == D.TS and cal["gap"].to_list() == [None, -0.5]


# ---------------------------------------------------------------- archive parsing / ingestion

ROW = "1577836800000,7189.43,7190.52,7172.94,7175.25,583.119,1577837099999,4186536.25,921,175.877,1262598.11,0\n"
ROW2 = "1577837100000,7174.71,7186.69,7173.28,7186.69,163.860,1577837399999,1176439.50,345,163.861,914720.55,0\n"
HEADER = ("open_time,open,high,low,close,volume,close_time,quote_volume,count,taker_buy_volume,"
          "taker_buy_quote_volume,ignore\n")


@pytest.mark.parametrize("header", [False, True])
def test_parse_klines_with_and_without_header(header):
    df, q = I.parse_klines(((HEADER if header else "") + ROW + ROW2).encode())
    assert df.columns == ["ts_close", "o", "h", "l", "c", "v", "buy_v", "sell_v"]
    assert df["ts_close"][0] == datetime(2020, 1, 1, 0, 5, tzinfo=UTC)  # open_time + 5m
    assert df["buy_v"][0] == 175.877 and df["sell_v"][0] == pytest.approx(583.119 - 175.877)
    assert df["sell_v"][1] == 0.0 and q["buy_gt_volume"] == 1  # rounding: buy > volume, clipped and counted
    assert q["rows"] == 2 and q["off_grid"] == q["close_time_mismatch"] == q["ohlc_violations"] == 0


def test_parse_klines_microseconds_and_off_grid():
    us = ROW.replace("1577836800000", "1577836800000000").replace("1577837099999", "1577837099999000")
    off = ROW2.replace("1577837100000", "1577837160000")
    df, q = I.parse_klines((us + off).encode())
    assert df.height == 1 and df["ts_close"][0] == datetime(2020, 1, 1, 0, 5, tzinfo=UTC)
    assert q["off_grid"] == 1


def _listing(keys: list[tuple[str, int]]) -> bytes:
    body = "".join(f"<Contents><Key>{k}</Key><Size>{s}</Size></Contents>" for k, s in keys)
    return (f'<?xml version="1.0"?><ListBucketResult xmlns="http://s3.amazonaws.com/doc/2006-03-01/">'
            f"<IsTruncated>false</IsTruncated>{body}</ListBucketResult>").encode()


def test_binance_plan_monthly_then_daily(monkeypatch):
    m = "data/futures/um/monthly/klines/XUSDT/5m/XUSDT-5m-"
    d = "data/futures/um/daily/klines/XUSDT/5m/XUSDT-5m-"

    def get(url: str) -> bytes:
        if "monthly" in url:
            return _listing([(m + "2026-09.zip", 10), (m + "2026-09.zip.CHECKSUM", 1), (m + "2026-10.zip", 11)])
        if "2026-11" in url:
            return _listing([(d + "2026-11-01.zip", 1), (d + "2026-11-02.zip", 1), (d + "2026-11-03.zip", 1)])
        return _listing([])

    plan = I.binance_plan("XUSDT", date(2026, 11, 3), get)
    assert [(p["kind"], p["period"]) for p in plan] == [("monthly", "2026-09"), ("monthly", "2026-10"),
                                                       ("daily", "2026-11-01"), ("daily", "2026-11-02")]


def _zip(csv: str) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("XUSDT-5m-2020-01.csv", csv)
    return buf.getvalue()


@pytest.fixture
def tmp_store(tmp_path, monkeypatch):
    monkeypatch.setattr(I, "RAW_DIR", tmp_path / "raw")
    monkeypatch.setattr(I, "STAGING_DIR", tmp_path / "staging")
    monkeypatch.setattr(I, "DATA_DIR", tmp_path)
    return tmp_path


def test_ingest_one_verifies_checksum_converts_and_deletes_zip(tmp_store):
    z = _zip(ROW + ROW2)
    sha = hashlib.sha256(z).hexdigest()
    files = {"/x.zip": z, "/x.zip.CHECKSUM": f"{sha}  XUSDT-5m-2020-01.zip\n".encode()}
    get = lambda url: files["/" + url.rsplit("/", 1)[1]]
    man = I.Manifest(tmp_store / "manifest.json")
    e = I.ingest_one({"key": "a/x.zip", "kind": "monthly", "period": "2020-01"}, "XUSDT", man, get)
    assert e["checksum_verified"] and e["zip_sha256"] == sha and e["quality"]["rows"] == 2
    staged = tmp_store / e["staged"]
    assert pl.read_parquet(staged).height == 2
    assert not list((tmp_store / "raw").iterdir()), "zip deleted after conversion"
    assert json.loads((tmp_store / "manifest.json").read_text())["files"]["a/x.zip"]["status"] == "staged"


def test_ingest_one_rejects_bad_checksum_and_still_deletes_zip(tmp_store):
    files = {"/x.zip": _zip(ROW), "/x.zip.CHECKSUM": b"0" * 64 + b"  x.zip\n"}
    man = I.Manifest(tmp_store / "manifest.json")
    with pytest.raises(RuntimeError, match="does not match CHECKSUM"):
        I.ingest_one({"key": "a/x.zip", "kind": "monthly", "period": "2020-01"}, "XUSDT", man,
                     lambda url: files["/" + url.rsplit("/", 1)[1]])
    assert not list((tmp_store / "raw").iterdir()) and "a/x.zip" not in man.files


def test_ingest_skips_files_already_in_manifest(tmp_store, monkeypatch):
    man = I.Manifest(tmp_store / "manifest.json")
    man.files["k1"] = {"asset": "XUSDT"}
    monkeypatch.setattr(I, "binance_plan", lambda s, t, g: [{"key": "k1", "size": 5, "kind": "monthly", "period": "2020-01"}])
    called = []
    monkeypatch.setattr(I, "ingest_one", lambda *a, **k: called.append(a))
    res = I.ingest_binance(["XUSDT"], man, today=date(2026, 1, 1), check_storage=False, log=lambda *_: None)
    assert res["files_downloaded"] == 0 and not called


def test_fill_from_daily_fetches_days_of_missing_runs(tmp_store):
    # monthly file truncated: 2020-01-01 complete, 2020-01-02 missing, 2020-01-03 00:00-00:10 present
    t0 = datetime(2020, 1, 1, tzinfo=UTC)
    present = [t0 + (i + 1) * FIVE for i in range(288)] + [t0 + timedelta(days=2) + (i + 1) * FIVE for i in range(2)]
    st = tmp_store / "staging" / "binance" / "XUSDT"
    st.mkdir(parents=True)
    pl.DataFrame({"ts_close": present}).with_columns(pl.col("ts_close").cast(D.TS)).write_parquet(st / "2020-01.parquet")
    day2 = "".join(f"{int((t0 + timedelta(days=1) + i * FIVE).timestamp() * 1000)},1,2,0.5,1.5,3,0,0,1,1,0,0\n"
                   for i in range(288))
    z = _zip(day2)
    served = {"XUSDT-5m-2020-01-02.zip": z,
              "XUSDT-5m-2020-01-02.zip.CHECKSUM": f"{hashlib.sha256(z).hexdigest()}  x\n".encode()}

    def get(url: str) -> bytes:
        name = url.rsplit("/", 1)[1]
        if name not in served:
            raise I.urllib.error.HTTPError(url, 404, "nf", {}, None)
        return served[name]

    man = I.Manifest(tmp_store / "manifest.json")
    man.assets["XUSDT"] = {"status": "staged"}
    res = I.fill_from_daily(["XUSDT"], man, get=get, log=lambda *_: None)
    assert res["XUSDT"]["runs"] == 1 and res["XUSDT"]["daily_fetched"] == 1
    assert (st / "2020-01-02.parquet").exists()
    k = "data/futures/um/daily/klines/XUSDT/5m/XUSDT-5m-2020-01-02.zip"
    assert man.files[k]["purpose"] == "archive-gap repair"
    # the day before/after the run that the archive lacks is recorded as absent, and nothing is fetched twice
    again = I.fill_from_daily(["XUSDT"], man, get=get, log=lambda *_: None)
    assert again["XUSDT"]["daily_fetched"] == 0
