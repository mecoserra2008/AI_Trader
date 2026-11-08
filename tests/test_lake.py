from datetime import UTC, datetime, timedelta

import polars as pl
import pytest

from ai_trader.core.time import LookAheadError
from ai_trader.storage.lake import Lake
from ai_trader.tools.vol import ewma

from .conftest import T0, make_bars


def funding(rows):
    return pl.DataFrame(rows, schema={"venue": pl.Utf8, "symbol": pl.Utf8, "event_ts": pl.Datetime("ms", "UTC"), "rate": pl.Float64, "known_ts": pl.Datetime("ms", "UTC")}, orient="row")


def test_asof_returns_revision_known_at_the_time(lake):
    m = timedelta(minutes=1)
    lake.write("funding", funding([("bybit", "BTCUSDT", T0, 0.0001, T0)]))
    lake.write("funding", funding([("bybit", "BTCUSDT", T0, 0.0005, T0 + 10 * m)]))  # revision
    assert lake.read_asof("funding", T0 - m).height == 0
    assert lake.read_asof("funding", T0 + 5 * m)["rate"].to_list() == [0.0001]
    assert lake.read_asof("funding", T0 + 20 * m)["rate"].to_list() == [0.0005]


def test_rejects_rows_known_before_they_happen(lake):
    with pytest.raises(LookAheadError):
        lake.write("funding", funding([("bybit", "BTCUSDT", T0, 0.0, T0 - timedelta(seconds=1))]))


def test_tool_output_is_invariant_to_future_data(tmp_path):
    """The central look-ahead test: deleting everything after as_of must not change any output."""
    bars = make_bars(600)
    full, truncated = Lake(root=tmp_path / "full"), Lake(root=tmp_path / "trunc")
    full.write("bars", bars)
    as_of = T0 + timedelta(hours=400, minutes=30)
    truncated.write("bars", bars.filter(pl.col("known_ts") <= as_of))
    a = ewma("TESTUSDT", "1h", as_of=as_of, lake=full)
    b = ewma("TESTUSDT", "1h", as_of=as_of, lake=truncated)
    assert a.value == b.value and a.data_max_ts == b.data_max_ts
    assert a.data_max_ts <= as_of


def test_reads_partition_filters(lake):
    lake.write("bars", make_bars(30, "AAAUSDT"))
    lake.write("bars", make_bars(30, "BBBUSDT"))
    df = lake.bars("BBBUSDT", "1h", T0 + timedelta(days=5))
    assert set(df["symbol"]) == {"BBBUSDT"} and df.height == 30


@pytest.mark.skipif(not Lake().has_data("bars"), reason="no ingested data (run `atx klines` first)")
def test_reads_rust_written_bars():
    now = datetime.now(UTC)
    df = Lake().bars("BTCUSDT", "1h", now)
    assert df.height > 0
    assert df["known_ts"].max() <= now
    assert df["event_ts"].is_sorted()
    assert ((df["close_ts"] - df["event_ts"]) == timedelta(hours=1)).all()
    assert (df["known_ts"] >= df["close_ts"]).all(), "a bar is never known before it closes"
