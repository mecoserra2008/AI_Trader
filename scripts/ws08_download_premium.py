"""WS08: download the public hourly candles needed for the Coinbase premium (raw and USDT-adjusted).

Coinbase Exchange BTC-USD, ETH-USD, USDT-USD (public REST, 300 candles per call) and Binance spot BTCUSDT, ETHUSDT
(public REST, 1000 per call), 1h, 2020-01-01 -> now. Raw JSON pages go to data/raw/WS08/ (transient), the merged
close series to data/research/ws08/premium_1h.parquet. Prices are native (never revised): known_ts = candle open + 1h.

    .venv/Scripts/python.exe scripts/ws08_download_premium.py
"""

from __future__ import annotations

import json
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "WS08"
OUT = ROOT / "data" / "research" / "ws08"
START = int(datetime(2020, 1, 1, tzinfo=timezone.utc).timestamp())
H = 3600


def get(url: str) -> object:
    for k in range(6):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "ai-trader-research/1.0"})
            with urllib.request.urlopen(req, timeout=30) as r:
                return json.loads(r.read().decode())
        except Exception as e:  # noqa: BLE001 - retry any transport error
            print(f"retry {k} {url[:90]}: {e}", flush=True)
            time.sleep(2 + 3 * k)
    raise RuntimeError(url)


def coinbase(product: str, start: int) -> pl.DataFrame:
    now = int(time.time()) // H * H
    rows, t = [], start
    while t < now:
        e = min(t + 299 * H, now)
        s_iso = datetime.fromtimestamp(t, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        e_iso = datetime.fromtimestamp(e, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        page = get(f"https://api.exchange.coinbase.com/products/{product}/candles?granularity=3600&start={s_iso}&end={e_iso}")
        if isinstance(page, list):
            rows.extend(page)
        t = e + H
        time.sleep(0.15)
    (RAW / f"cb_{product}.json").write_text(json.dumps(rows))
    df = pl.DataFrame({"ts": [int(r[0]) for r in rows], "close": [float(r[4]) for r in rows], "vol": [float(r[5]) for r in rows]})
    return df.unique("ts").sort("ts")


def binance(symbol: str, start: int) -> pl.DataFrame:
    now = int(time.time()) // H * H
    rows, t = [], start
    while t < now:
        page = get(f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval=1h&startTime={t * 1000}&limit=1000")
        if not page:
            break
        rows.extend(page)
        t = int(page[-1][0]) // 1000 + H
        time.sleep(0.1)
    (RAW / f"bn_{symbol}.json").write_text(json.dumps(rows))
    df = pl.DataFrame({"ts": [int(r[0]) // 1000 for r in rows], "close": [float(r[4]) for r in rows], "vol": [float(r[7]) for r in rows]})
    return df.unique("ts").sort("ts")


def main() -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    frames = {}
    for prod in ("BTC-USD", "ETH-USD", "USDT-USD"):
        frames[prod] = coinbase(prod, START)
        print(prod, frames[prod].height, flush=True)
    for sym in ("BTCUSDT", "ETHUSDT"):
        frames[sym] = binance(sym, START)
        print(sym, frames[sym].height, flush=True)
    out = None
    for k, df in frames.items():
        d = df.select("ts", pl.col("close").alias(f"c_{k.replace('-', '_')}"), pl.col("vol").alias(f"v_{k.replace('-', '_')}"))
        out = d if out is None else out.join(d, on="ts", how="full", coalesce=True)
    out = out.sort("ts").with_columns(
        pl.from_epoch("ts", time_unit="s").dt.replace_time_zone("UTC").alias("event_ts"),
        pl.from_epoch(pl.col("ts") + H, time_unit="s").dt.replace_time_zone("UTC").alias("known_ts"),
    )
    out.write_parquet(OUT / "premium_1h.parquet")
    print("rows", out.height, "->", OUT / "premium_1h.parquet")


if __name__ == "__main__":
    main()
