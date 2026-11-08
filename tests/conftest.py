from datetime import UTC, datetime, timedelta

import numpy as np
import polars as pl
import pytest

from ai_trader.storage.lake import Lake

T0 = datetime(2026, 1, 1, tzinfo=UTC)


@pytest.fixture
def lake(tmp_path):
    return Lake(root=tmp_path / "lake")


def make_bars(n: int, symbol: str = "TESTUSDT", freq_h: int = 1, seed: int = 0) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(rng.standard_t(5, n) * 0.01))
    open_ = np.r_[100.0, close[:-1]]
    ev = [T0 + timedelta(hours=freq_h * i) for i in range(n)]
    ct = [t + timedelta(hours=freq_h) for t in ev]
    return pl.DataFrame({
        "venue": "bybit", "price_type": "last", "freq": f"{freq_h}h", "symbol": symbol,
        "event_ts": ev, "close_ts": ct,
        "open": open_, "high": np.maximum(open_, close) * 1.002, "low": np.minimum(open_, close) * 0.998, "close": close,
        "volume": 1.0, "turnover": close,
        "known_ts": [t + timedelta(seconds=2) for t in ct],
    }).with_columns(pl.col(["event_ts", "close_ts", "known_ts"]).dt.replace_time_zone("UTC").dt.cast_time_unit("ms"))
