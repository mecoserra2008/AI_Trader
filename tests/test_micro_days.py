"""Tests for the multi-day microstructure pipeline (ai_trader.research.micro_days)."""

from __future__ import annotations

import json

import numpy as np
import polars as pl
import pytest

from ai_trader.research import micro_days as m

D0 = m.day0_ms("2026-01-01")


def _trades():
    # second 0: buy order of 3 prints at ms 100, then sell print at ms 100 (new order), buy at ms 101 (new order)
    # second 1: two sells at ms 1500 (one order), one sell at ms 1500 after an interleaved buy -> three orders
    ms = np.array([100, 100, 100, 100, 101, 1500, 1500, 1500, 1500], dtype=np.int64) + D0
    side = np.array([1, 1, 1, -1, 1, -1, -1, 1, -1], dtype=np.int8)
    price = np.array([10.0, 10.1, 10.2, 9.9, 10.0, 9.8, 9.7, 9.9, 9.6])
    size = np.array([1.0, 2.0, 1.0, 1.0, 3.0, 1.0, 1.0, 2.0, 1.0])
    return {"ms": ms, "side": side, "price": price, "size": size}


def test_merge_orders_same_ms_same_side_runs():
    od = m.merge_orders(_trades())
    assert od["n_prints"].tolist() == [3, 1, 1, 2, 1, 1]
    assert od["side"].tolist() == [1, -1, 1, -1, 1, -1]
    assert od["size"].tolist() == [4.0, 1.0, 3.0, 2.0, 2.0, 1.0]
    assert od["first_price"][0] == 10.0 and od["last_price"][0] == 10.2
    assert od["notional"][0] == pytest.approx(10.0 + 20.2 + 10.2)
    assert od["n_prints"].sum() == len(_trades()["ms"])


def test_bars_1s_aggregation():
    tr = _trades()
    b = m.bars_1s(tr, m.merge_orders(tr), D0)
    assert b["ts"].to_list() == [D0, D0 + 1000]
    r0, r1 = b.row(0, named=True), b.row(1, named=True)
    assert (r0["open"], r0["high"], r0["low"], r0["close"]) == (10.0, 10.2, 9.9, 10.0)
    assert r0["n_trades"] == 5 and r0["n_buy"] == 4 and r0["n_sell"] == 1
    assert r0["n_orders"] == 3 and r0["n_buy_orders"] == 2 and r0["n_sell_orders"] == 1
    assert r0["buy_vol"] == pytest.approx(7.0) and r0["sell_vol"] == pytest.approx(1.0)
    assert r0["buy_notional"] == pytest.approx(10.0 + 20.2 + 10.2 + 30.0)
    assert r0["vwap"] == pytest.approx((10.0 + 20.2 + 10.2 + 9.9 + 30.0) / 8.0)
    assert r1["n_orders"] == 3 and r1["n_sell_orders"] == 2 and r1["close"] == 9.6
    assert r1["sell_notional"] == pytest.approx(9.8 + 9.7 + 9.6)


def test_load_csv_trades_rounding_and_order(tmp_path):
    import gzip

    rows = ["timestamp,symbol,side,size,price,tickDirection,trdMatchID,grossValue,homeNotional,foreignNotional,RPI",
            f"{(D0 + 1500) / 1000:.4f},X,Sell,1,9,MinusTick,a,0,1,9,0",
            f"{(D0 + 100) / 1000 + 0.0004:.4f},X,Buy,2,10,PlusTick,b,0,2,20,0",
            f"{(D0 + 100) / 1000:.4f},X,Sell,3,11,PlusTick,c,0,3,33,0",
            f"{(D0 - 5) / 1000:.4f},X,Sell,3,11,PlusTick,d,0,3,33,0"]
    p = tmp_path / "x.csv.gz"
    with gzip.open(p, "wt") as f:
        f.write("\n".join(rows) + "\n")
    tr = m.load_csv_trades(p, D0)
    assert tr["ms"].tolist() == [D0 + 100, D0 + 100, D0 + 1500]   # 100.4 ms rounds to 100; previous-day row dropped
    assert tr["side"].tolist() == [1, -1, -1]                      # stable sort keeps file order within the ms
    assert tr["size"].tolist() == [2.0, 3.0, 1.0]


def test_tick_estimate():
    assert m.tick_estimate(np.array([0.04585, 0.04586, 0.04590, 0.04585])) == pytest.approx(1e-5)
    assert m.tick_estimate(np.array([100.0])) is None


def _book100(n_sec=3):
    t = np.arange(1, n_sec * 10 + 1) * 100 + D0
    k = np.arange(len(t), dtype=float)
    return pl.DataFrame({"ts": t, "cts": t - 5, "bid1": 100 + k, "ask1": 101 + k, "mid": 100.5 + k, "n_updates": np.ones(len(t), np.int32)}).with_columns(
        pl.col("ts").cast(pl.Datetime("ms")).dt.replace_time_zone("UTC"))


def test_downsample_book_keeps_end_of_second_state():
    b = m.downsample_book(_book100(3))
    t = b["ts"].dt.epoch("ms").to_list()
    assert t == [D0 + 1000, D0 + 2000, D0 + 3000]
    assert b["bid1"].to_list() == [109.0, 119.0, 129.0]           # the state at the second boundary (10th, 20th, 30th bin)
    assert b["n_updates"].to_list() == [10, 10, 10]


def test_downsample_matches_direct_1s_build():
    from ai_trader.sim.book import build_from_messages

    msgs = [{"ts": D0 + 50, "type": "snapshot", "data": {"b": [["100", "1"]], "a": [["101", "1"]]}}]
    for i in range(1, 40):
        msgs.append({"ts": D0 + 50 + i * 70, "type": "delta", "data": {"b": [[str(100 + i * 0.01), "1"]], "a": []}})
    b100, _ = build_from_messages(iter(msgs), 100)
    b1s, _ = build_from_messages(iter(msgs), 1000)
    ds = m.downsample_book(b100)
    j = ds.select("ts", "bid1").join(b1s.select("ts", pl.col("bid1").alias("b")), on="ts")
    assert j.height >= 2
    assert np.allclose(j["bid1"].to_numpy(), j["b"].to_numpy())


def _write_day(out, sym="AAAUSDT", book=True, bad_close=False):
    tr = _trades()
    b = m.bars_1s(tr, m.merge_orders(tr), D0)
    if bad_close:
        b = b.with_columns(pl.lit(float("nan")).alias("close"))
    (out / "trades_1s").mkdir(parents=True)
    b.write_parquet(out / "trades_1s" / f"{sym}.parquet")
    (out / "orders").mkdir()
    pl.DataFrame(m.merge_orders(tr)).write_parquet(out / "orders" / f"{sym}.parquet")
    (out / "laws").mkdir()
    (out / "laws" / f"{sym}.json").write_text(json.dumps({"halves": {"fit": {"rates": {}}, "test": {}}}))
    if book:
        (out / "book_1s").mkdir()
        pl.DataFrame({"mid": np.full(86400, 100.0)}).write_parquet(out / "book_1s" / f"{sym}.parquet")
    (out / "day_stats.json").write_text("{}")
    (out / "qa.json").write_text("{}")


def test_verify_then_delete_only_when_ok(tmp_path):
    out, raw = tmp_path / "out", tmp_path / "raw"
    raw.mkdir()
    (raw / "x.csv.gz").write_bytes(b"x")
    _write_day(out)
    ver = m.verify_day(out, ["AAAUSDT"], ["AAAUSDT"])
    assert ver["ok"], ver
    assert m.cleanup_raw(raw, ver, keep_raw=True) is False and raw.exists()
    assert m.cleanup_raw(raw, ver) is True and not raw.exists()


def test_no_delete_when_verification_fails(tmp_path):
    out, raw = tmp_path / "out", tmp_path / "raw"
    raw.mkdir()
    (raw / "x.csv.gz").write_bytes(b"x")
    _write_day(out, bad_close=True)
    ver = m.verify_day(out, ["AAAUSDT", "MISSINGUSDT"], ["AAAUSDT"])
    assert not ver["ok"]
    assert any("non-finite" in p for p in ver["problems"]) and any("MISSINGUSDT" in p for p in ver["problems"])
    assert m.cleanup_raw(raw, ver) is False and raw.exists()


def test_no_delete_when_book_short(tmp_path):
    out, raw = tmp_path / "out", tmp_path / "raw"
    raw.mkdir()
    _write_day(out, book=False)
    (out / "book_1s").mkdir()
    pl.DataFrame({"mid": np.full(1000, 100.0)}).write_parquet(out / "book_1s" / "AAAUSDT.parquet")
    ver = m.verify_day(out, ["AAAUSDT"], ["AAAUSDT"])
    assert not ver["ok"] and m.cleanup_raw(raw, ver) is False and raw.exists()


def test_ms_vwap():
    t, lp = m.ms_vwap(_trades())
    assert t.tolist() == [D0 + 100, D0 + 101, D0 + 1500]
    assert np.exp(lp[0]) == pytest.approx((10.0 + 20.2 + 10.2 + 9.9) / 5.0)


def test_resumed_task_only_for_complete_outputs(tmp_path):
    out, tmp = tmp_path / "out", tmp_path / "tmp"
    _write_day(out)
    tmp.mkdir()
    assert m.resumed_task("2026-01-01", "trades", "AAAUSDT", out, tmp) is None      # ms-VWAP temp missing -> recompute
    t, lp = m.ms_vwap(_trades())
    np.savez(tmp / "AAAUSDT_msvwap.npz", t=t, lp=lp)
    r = m.resumed_task("2026-01-01", "trades", "AAAUSDT", out, tmp)
    assert r is not None and r["resumed"] and r["bars_1s"] == 2
    assert m.resumed_task("2026-01-01", "book", "AAAUSDT", out, tmp)["rows"] == 86400
    (out / "book_1s" / "BBBUSDT.parquet").write_bytes(b"truncated")                  # partial write -> recompute
    assert m.resumed_task("2026-01-01", "book", "BBBUSDT", out, tmp) is None


def test_load_csv_trades_streaming_path_matches(tmp_path, monkeypatch):
    import gzip

    rows = ["timestamp,symbol,side,size,price,tickDirection,trdMatchID,grossValue,homeNotional,foreignNotional,RPI"]
    rows += [f"{(D0 + 10 * i) / 1000:.4f},X,{'Buy' if i % 3 else 'Sell'},{i + 1},{100 + i % 7},PlusTick,id{i},0,1,1,0" for i in range(500)]
    p = tmp_path / "y.csv.gz"
    with gzip.open(p, "wt") as f:
        f.write("\n".join(rows) + "\n")
    a = m.load_csv_trades(p, D0)
    real_stat = type(p).stat

    class Big:
        st_size = 1e9

    monkeypatch.setattr(type(p), "stat", lambda self, *k, **kw: Big() if self == p else real_stat(self, *k, **kw))
    b = m.load_csv_trades(p, D0)
    assert all(np.array_equal(a[k], b[k]) for k in a) and len(a["ms"]) == 500
    assert not (tmp_path / "y.csv.gz.tmp.csv").exists()
