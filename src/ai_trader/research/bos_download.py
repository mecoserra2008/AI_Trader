"""Polite downloader of Bybit public linear klines for the BOS study (block BOS, research/hypotheses.yaml).

Writes data/research/bos/bars/<interval>/<SYMBOL>_<YEAR>.parquet (event_ts UTC, open, high, low, close, volume,
turnover) and a manifest (data/research/bos/bars/manifest.json) with row counts, first/last bar, gaps and bytes.
Public REST only (/v5/market/kline, category linear), ~6 requests per second, retries with backoff, resumable:
a (symbol, year) file that already exists and is complete is skipped. Only closed bars are kept.

    uv run python -m ai_trader.research.bos_download --interval 5 --start 2022-01-01
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.parse
import urllib.request
from datetime import UTC, datetime
from pathlib import Path

import polars as pl

from ai_trader.core.config import load

URL = "https://api.bybit.com/v5/market/kline"
UNIVERSE_L = (
    "BTCUSDT ETHUSDT SOLUSDT XRPUSDT DOGEUSDT ADAUSDT LINKUSDT LTCUSDT BCHUSDT AVAXUSDT DOTUSDT BNBUSDT "
    "TRXUSDT NEARUSDT ATOMUSDT ETCUSDT FILUSDT UNIUSDT AAVEUSDT XLMUSDT SUIUSDT ENAUSDT HYPEUSDT"
).split()
PAUSE = 0.16
MAX_BYTES = 1.5e9


def root() -> Path:
    return load().root / "data" / "research" / "bos" / "bars"


def _get(params: dict) -> list:
    q = URL + "?" + urllib.parse.urlencode(params)
    for attempt in range(8):
        try:
            req = urllib.request.Request(q, headers={"User-Agent": "ai-trader-research/0.1"})
            with urllib.request.urlopen(req, timeout=20) as r:
                js = json.loads(r.read().decode())
            if js.get("retCode") == 0:
                return js["result"]["list"]
            if js.get("retCode") in (10006, 10016, 10018):  # rate limit / transient service error
                time.sleep(5 * (attempt + 1))
                continue
            raise RuntimeError(f"bybit error {js.get('retCode')}: {js.get('retMsg')}")
        except (OSError, TimeoutError, json.JSONDecodeError):
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"failed after retries: {q}")


def first_bar_ms(sym: str) -> int:
    """First 1h bar in the lake (listing), so pre-listing windows are not requested."""
    b = "data/lake/bars/venue=bybit/price_type=last/freq=1h"
    d = pl.scan_parquet(str(load().root / b / "**" / "*.parquet"), hive_partitioning=True).filter(pl.col("symbol") == sym)
    t = d.select(pl.col("event_ts").min()).collect().item()
    return int(t.timestamp() * 1000) if t is not None else 0


def fetch_range(sym: str, interval: int, start_ms: int, end_ms: int) -> pl.DataFrame:
    step = interval * 60_000
    rows: list = []
    s = start_ms
    while s < end_ms:
        e = min(s + 1000 * step - 1, end_ms - 1)
        lst = _get({"category": "linear", "symbol": sym, "interval": str(interval), "start": s, "end": e, "limit": 1000})
        rows.extend(lst)
        s = e + 1
        time.sleep(PAUSE)
    if not rows:
        return pl.DataFrame()
    df = pl.DataFrame({
        "ts": [int(r[0]) for r in rows], "open": [float(r[1]) for r in rows], "high": [float(r[2]) for r in rows],
        "low": [float(r[3]) for r in rows], "close": [float(r[4]) for r in rows], "volume": [float(r[5]) for r in rows],
        "turnover": [float(r[6]) for r in rows],
    }).unique("ts").sort("ts")
    df = df.filter((pl.col("ts") >= start_ms) & (pl.col("ts") + step <= end_ms))
    return df.with_columns(pl.from_epoch("ts", time_unit="ms").dt.replace_time_zone("UTC").alias("event_ts")).drop("ts").select(
        "event_ts", "open", "high", "low", "close", "volume", "turnover")


def run(interval: int, start: str, symbols: list[str]) -> dict:
    out = root() / f"{interval}m"
    out.mkdir(parents=True, exist_ok=True)
    mpath = root() / "manifest.json"
    man = json.loads(mpath.read_text()) if mpath.exists() else {"source": URL, "category": "linear", "files": {}}
    now_ms = int(datetime.now(UTC).timestamp() * 1000) // (interval * 60_000) * (interval * 60_000)
    y0 = int(start[:4])
    for sym in symbols:
        lst = max(first_bar_ms(sym), int(datetime.fromisoformat(start).replace(tzinfo=UTC).timestamp() * 1000))
        for y in range(y0, datetime.now(UTC).year + 1):
            ys = int(datetime(y, 1, 1, tzinfo=UTC).timestamp() * 1000)
            ye = min(int(datetime(y + 1, 1, 1, tzinfo=UTC).timestamp() * 1000), now_ms)
            key = f"{interval}m/{sym}_{y}"
            f = out / f"{sym}_{y}.parquet"
            if ye <= lst:
                continue
            done = man["files"].get(key, {})
            if f.exists() and done.get("complete"):
                continue
            t0 = time.time()
            df = fetch_range(sym, interval, max(ys, lst), ye)
            if df.height:
                df.write_parquet(f, compression="zstd")
            step = interval * 60_000
            exp = (ye - max(ys, lst)) // step
            man["files"][key] = {"rows": df.height, "expected": int(exp), "first": str(df["event_ts"].min()) if df.height else None,
                                 "last": str(df["event_ts"].max()) if df.height else None, "bytes": f.stat().st_size if f.exists() else 0,
                                 "complete": ye < now_ms or y < datetime.now(UTC).year, "fetched_at": datetime.now(UTC).isoformat(timespec="seconds")}
            man["total_bytes"] = sum(v["bytes"] for v in man["files"].values())
            mpath.write_text(json.dumps(man, indent=1))
            print(f"{key} rows={df.height}/{exp} {time.time() - t0:.0f}s total={man['total_bytes'] / 1e6:.0f}MB", flush=True)
            if man["total_bytes"] > MAX_BYTES:
                raise SystemExit("byte budget reached")
    return man


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--interval", type=int, default=5)
    ap.add_argument("--start", default="2022-01-01")
    ap.add_argument("--symbols", default=",".join(UNIVERSE_L))
    a = ap.parse_args()
    run(a.interval, a.start, a.symbols.split(","))


if __name__ == "__main__":
    main()
