from datetime import UTC, date, datetime, timedelta

import numpy as np
import polars as pl

from ai_trader.storage import flow
from ai_trader.storage.lake import Lake

T0 = datetime(2026, 11, 1, tzinfo=UTC)


def synth_3s(n: int = 20 * 30, symbols=("AAAUSDT", "BBBUSDT"), seed: int = 0, start: datetime = T0) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    frames = []
    for s in symbols:
        ts = [start + timedelta(seconds=3 * i) for i in range(n)]
        px = 100 * np.exp(np.cumsum(rng.normal(0, 1e-4, n)))
        traded = rng.random(n) < 0.7
        b, sv = rng.exponential(1, n) * traded, rng.exponential(1, n) * traded
        cov = np.where(rng.random(n) < 0.1, 0.5, 1.0).astype(np.float32)
        d = {
            "venue": "bybit", "category": "linear", "symbol": s, "event_ts": ts,
            "open": np.where(traded, px, np.nan), "high": np.where(traded, px * 1.0001, np.nan), "low": np.where(traded, px * 0.9999, np.nan),
            "close": px, "vwap": np.where(traded, px, np.nan),
            "buy_vol": b.astype(np.float32), "sell_vol": sv.astype(np.float32),
            "buy_notional": (b * px).astype(np.float32), "sell_notional": (sv * px).astype(np.float32),
            "n_buy": (traded * 3).astype(np.int32), "n_sell": (traded * 2).astype(np.int32), "max_trade": b.astype(np.float32),
            "mid": px, "book_span_bps": np.full(n, 30.0, np.float32), "book_cov": cov,
            "ofi": rng.normal(0, 1, n).astype(np.float32), "book_updates": np.full(n, 30, np.int32),
            "known_ts": [t + timedelta(seconds=4) for t in ts],
        }
        for c in flow.TWA_COLS:
            d[c] = rng.uniform(-1, 1, n).astype(np.float32)
        frames.append(pl.DataFrame(d).with_columns(pl.col("open", "high", "low", "vwap").fill_nan(None)))
    return pl.concat(frames).with_columns(pl.col("event_ts", "known_ts").dt.cast_time_unit("ms"))


def test_rollup_is_exact():
    df3 = synth_3s()
    m = flow.rollup(df3)
    assert m.height == 2 * 30 and (m["_n"] == 20).all(), "20 three-second candles per minute"
    a = df3.filter((pl.col("symbol") == "AAAUSDT") & (pl.col("event_ts") < T0 + timedelta(minutes=1)))
    r = m.filter(pl.col("symbol") == "AAAUSDT").row(0, named=True)
    assert abs(r["buy_vol"] - a["buy_vol"].sum()) < 1e-3
    assert r["n_buy"] == a["n_buy"].sum()
    w = (a["obi_5"] * a["book_cov"]).sum() / a["book_cov"].sum()
    assert abs(r["obi_5"] - w) < 1e-5, "book features weighted by coverage"
    assert r["close"] == a["close"][-1] and r["high"] == a["high"].max()
    assert r["known_ts"] >= r["event_ts"] + timedelta(minutes=1), "a minute is known only after its last candle"


def test_derive_is_causal_and_per_symbol():
    df3 = synth_3s()
    full = flow.derive(df3, [5, 20], ["delta", "obi_5"])
    k = 400
    part = flow.derive(df3.filter(pl.col("event_ts") < T0 + timedelta(seconds=3 * k)), [5, 20], ["delta", "obi_5"])
    cols = ["delta_sma_5", "delta_ema_20", "obi_5_z_20"]
    a = full.filter(pl.col("event_ts") < T0 + timedelta(seconds=3 * k)).select("symbol", "event_ts", *cols)
    assert a.equals(part.select("symbol", "event_ts", *cols))
    b = full.filter(pl.col("symbol") == "BBBUSDT").head(5)["delta_sma_5"]
    assert b.null_count() == 4, "windows restart per symbol (no bleed from AAAUSDT)"


def test_maintain_rolls_up_then_prunes_within_budget(tmp_path, monkeypatch):
    lake = Lake(root=tmp_path / "lake")
    for d in range(3):
        lake.write("flow_3s", synth_3s(start=T0 + timedelta(days=d), seed=d))
    cfg = flow.load().settings["storage"]
    monkeypatch.setitem(cfg, "budget_gb", 0.0)  # force pruning
    monkeypatch.setitem(cfg, "flow_3s_min_days", 1)
    log = flow.maintain(lake, today=date(2026, 11, 4))
    assert sum("rolled" in x for x in log) == 3
    assert lake.has_data("flow_1m")
    remaining = sorted(p.name for p in (lake.root / "flow_3s").rglob("date=*"))
    assert remaining == ["date=2026-11-03"], f"keeps the newest {cfg['flow_3s_min_days']} day(s): {remaining}"
    assert lake.read_asof("flow_1m", datetime(2026, 11, 5, tzinfo=UTC)).height == 3 * 2 * 30
