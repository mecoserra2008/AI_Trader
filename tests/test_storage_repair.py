"""Reversible lake repairs (D8): exact-duplicate compaction, dropped rows, replaced rows, undo / purge, hot refusal."""

import json
from datetime import UTC, date, datetime, timedelta

import polars as pl
import pyarrow as pa
import pytest

from ai_trader.storage import repair
from ai_trader.storage.lake import Lake

T0 = datetime(2020, 12, 13, tzinfo=UTC)
NOW = datetime(2026, 11, 27, tzinfo=UTC)


def bars(n=3, symbol="BCHUSDT", start=T0, open0=None) -> pl.DataFrame:
    ts = [start + timedelta(days=i) for i in range(n)]
    o = [100.0 + i for i in range(n)]
    if open0 is not None:
        o[open0] = 0.0
    return pl.DataFrame({
        "venue": "bybit", "price_type": "last", "freq": "1d", "symbol": symbol, "event_ts": ts,
        "close_ts": [t + timedelta(days=1) for t in ts], "open": o, "high": [110.0] * n, "low": [90.0] * n,
        "close": [105.0] * n, "volume": [1.0] * n, "turnover": [100.0] * n,
        "known_ts": [t + timedelta(days=1, seconds=2) for t in ts],
    }).with_columns(pl.col("event_ts", "close_ts", "known_ts").dt.cast_time_unit("ms"))


@pytest.fixture
def lake(tmp_path):
    return Lake(root=tmp_path / "lake")


def part(lake):
    return lake.root / "bars/venue=bybit/price_type=last/freq=1d/year=2020"


def test_dedup_is_exact_reversible_and_keeps_types(lake):
    lake.write("bars", bars())
    lake.write("bars", bars())  # an overlapping backfill: exact duplicates
    lake.write("bars", bars(symbol="BTCUSDT"))
    before = lake.read_asof("bars", NOW)
    run = repair.new_run(lake.root, "t")
    log = repair.dedup(lake.root, run, ["bars"], hot=[], allow_hot=False, min_age_min=0)
    assert any("3 exact duplicates removed" in x for x in log), log
    files = repair.part_files(part(lake))
    assert len(files) == 1 and "-dedup-" in files[0].name and files[0].name.startswith("part-")
    after = lake.read_asof("bars", NOW)
    assert after.equals(before), "as-of reads unchanged"
    raw = pl.read_parquet(files[0])
    assert raw.height == 6 and raw.schema["event_ts"] == pl.Datetime("ms", "UTC") and raw.schema["open"] == pl.Float64
    man = [json.loads(x) for x in (run.dir / "manifest.jsonl").read_text().splitlines()]
    assert man[0]["verify"]["ok"] and man[0]["exact_duplicates"] == 3 and len(man[0]["files_in"]) == 3
    # undo restores the original files exactly
    repair.undo(lake.root, run.dir)
    assert len(repair.part_files(part(lake))) == 3
    assert lake.read_asof("bars", NOW).equals(before)


def test_nothing_to_do_without_duplicates(lake):
    lake.write("bars", bars())
    run = repair.new_run(lake.root, "t")
    assert repair.dedup(lake.root, run, ["bars"], hot=[], allow_hot=False, min_age_min=0) == []
    assert not (run.dir / "manifest.jsonl").exists()


def test_drop_predicate_and_purge(lake):
    lake.write("bars", bars(open0=1))
    run = repair.new_run(lake.root, "t")
    ln = repair.rewrite_partition(lake.root, part(lake), "bars", lake.spec("bars"), run, tag="d8bch", reason="open = 0",
                                  drop_sql="symbol = 'BCHUSDT' AND open <= 0")
    assert ln["rows_dropped"] == 1 and ln["rows_out"] == 2 and len(ln["removed_rows"]) == 1
    left = lake.read_asof("bars", NOW)
    assert left.height == 2 and (left["open"] > 0).all()
    msg = repair.purge(run.dir)
    assert "purged" in msg and not run.dir.exists()
    assert (run.root / "manifests" / f"{run.run_id}.jsonl").exists(), "the manifest outlives the backup"


def test_replace_rows_keeps_key_and_known_ts(lake):
    lake.write("bars", bars())
    run = repair.new_run(lake.root, "t")
    fix = bars().filter(pl.col("event_ts") == T0).with_columns(pl.lit(99.5).alias("close"))
    add = fix.drop("venue", "price_type", "freq").to_arrow()
    ln = repair.rewrite_partition(lake.root, part(lake), "bars", lake.spec("bars"), run, tag="fix", reason="refetch",
                                  drop_sql="event_ts = TIMESTAMPTZ '2020-12-13 00:00:00+00'", add=add)
    assert (ln["rows_dropped"], ln["rows_added"], ln["rows_out"]) == (1, 1, 3)
    r = lake.read_asof("bars", NOW).filter(pl.col("event_ts") == T0)
    assert r["close"].item() == 99.5 and r["known_ts"].item() == T0 + timedelta(days=1, seconds=2)


def test_hot_partitions_are_refused(lake):
    lake.write("bars", bars(start=datetime(2026, 11, 20, tzinfo=UTC)))
    lake.write("bars", bars(start=datetime(2026, 11, 20, tzinfo=UTC)))
    settings = {"live": {"price_types": ["last"], "bar_freqs": ["1d"]}}
    hot = repair.live_partitions(settings, date(2026, 11, 27))
    assert "bars/venue=bybit/price_type=last/freq=1d/year=2026" in hot
    run = repair.new_run(lake.root, "t")
    log = repair.dedup(lake.root, run, ["bars"], hot=hot, allow_hot=False, min_age_min=0)
    assert log and log[0].startswith("REFUSED") and len(repair.part_files(lake.root / "bars/venue=bybit/price_type=last/freq=1d/year=2026")) == 2
    # a fresh file makes any partition hot
    log = repair.dedup(lake.root, run, ["bars"], hot=[], allow_hot=False, min_age_min=60)
    assert log[0].startswith("REFUSED") and "min old" in log[0]
    # the deploy window
    log = repair.dedup(lake.root, run, ["bars"], hot=hot, allow_hot=True, min_age_min=0)
    assert "exact duplicates removed" in log[0]


def test_live_partitions_predeploy_excludes_d8_streams():
    s = {"live": {"instruments": True, "stablecoins": True, "extra": [{"kind": "ls_ratio", "period": "4h", "symbols": ["BTCUSDT"]}]}}
    after = repair.live_partitions(s, date(2026, 11, 27))
    before = repair.live_partitions(s, date(2026, 11, 27), predeploy=True)
    assert "ls_ratio/venue=bybit/period=4h/year=2026" in after and "ls_ratio/venue=bybit/period=4h/year=2026" not in before
    assert "flow_3s/venue=bybit/category=linear/date=2026-11-27" in before


def test_written_ms_from_name(tmp_path):
    p = tmp_path / "part-1794066913028-66346cba.parquet"
    p.write_bytes(b"")
    assert repair.written_ms(p) == 1794066913028
    assert pa is not None


def test_daily_flow_bars_plan_only_missing_recent_days():
    from ai_trader.storage.daily import flow_bars_plan

    today = date(2026, 11, 27)
    newest = {"BTCUSDT": date(2026, 11, 10), "ETHUSDT": date(2026, 11, 26), "SOLUSDT": date(2026, 11, 25)}
    assert flow_bars_plan(newest, today, 2, None) == [("BTCUSDT", date(2026, 11, 25)), ("BTCUSDT", date(2026, 11, 26)), ("SOLUSDT", date(2026, 11, 26))]
    hole = flow_bars_plan(newest, today, 2, date(2026, 11, 11))
    assert hole[0] == ("BTCUSDT", date(2026, 11, 11)) and len([x for x in hole if x[0] == "BTCUSDT"]) == 16, "explicit backfill closes the hole"


def test_svi_rows_follow_the_import_convention():
    from ai_trader.storage import deribit_svi as ds

    T = int(datetime(2026, 11, 26, 8, tzinfo=UTC).timestamp() * 1000)
    summary = {s: 0.1 for s in ds.SERIES} | {"atm_90": float("nan")}
    r = ds.rows_for("BTC", T, summary)
    assert r.height == len(ds.SERIES) - 1, "non-finite values are not written"
    assert set(r["series_id"]) == {f"iv_{s}" for s in ds.SERIES if s != "atm_90"}
    assert (r["known_ts"] - r["event_ts"]).unique().to_list() == [timedelta(seconds=60)]
    assert r["event_ts"][0] == datetime(2026, 11, 26, 8, tzinfo=UTC) and r["vintage"][0] == "btf_svi_v2"
