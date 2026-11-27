"""5m realised-variance store behind the clock log-HAR risk sigma (decision D1, 2026-11-27; ai_trader.risk.sigma).

One hourly row per symbol, built from Bybit linear 5m LAST-price klines (public /v5/market/kline: the source of the lake's
bars; the live updater writes only 1h / 4h / 1d, so 5m bars are not in the lake for the shadow universe):

    event_ts  hour open (UTC)            rv    sum of squared 5m log returns in the hour, rescaled to 12 returns
    open / high / low / close  of the hour (from its 5m bars)   (null when fewer than 10 valid returns: vol_spec rule)
    n5        valid 5m returns           known_ts  event_ts + 1h + 2 s (lake bars convention: a closed bar is final)

The rv / n5 definitions are ai_trader.research.vol_spec.hourly_grid (WS04), so the forecasts reproduce the tested spec.
Rows exist only for hours whose last 5m bar had closed when the chunk was fetched. Readers filter known_ts <= as_of.

    seed(symbols)      once: from the research 5m caches (read only): data/research/bos/bars/5m/<SYM>_<year>.parquet and
                       data/research/intraday_geometry/bars/<SYM>_ext.parquet, then refresh() for the tail
    refresh(symbols)   append the hours since the last stored one from public REST (one call per ~83 h per symbol)
    load(symbol, as_of)  the stored hours known at as_of, on a complete hourly calendar (missing hours -> null rv)

File: data/state/risk_sigma/rv1h/<SYMBOL>.parquet (~0.3 MB per symbol-year). Research only reads it; the shadow loop
(scripts/sigma_shadow.py) is the only writer. Writes are atomic (tmp + replace, retried while a reader holds the file).
"""

from __future__ import annotations

import json
import math
import os
import time
import urllib.parse
import urllib.request
from datetime import UTC, datetime, timedelta
from functools import lru_cache
from pathlib import Path

import polars as pl

from ai_trader.core.config import load as load_config

URL = "https://api.bybit.com/v5/market/kline"
LATENCY = timedelta(seconds=2)
STEP_MS = 300_000
PAUSE = 0.12
COLS = ("event_ts", "open", "high", "low", "close", "rv", "n5", "known_ts")
TS = pl.Datetime("us", "UTC")


def store_dir() -> Path:
    p = load_config().root / "data" / "state" / "risk_sigma" / "rv1h"
    p.mkdir(parents=True, exist_ok=True)
    return p


def path(symbol: str) -> Path:
    return store_dir() / f"{symbol}.parquet"


# ------------------------------------------------------------------ 5m bars -> hourly rows


def hours_from_5m(d5: pl.DataFrame, first_hour: datetime | None = None) -> pl.DataFrame:
    """Hourly rows from a sorted 5m frame (event_ts, open, high, low, close). Only hours that are complete in time (the
    5m bar opening at HH:55 had closed) and, if given, start at or after `first_hour` (the first hour whose opening 5m
    return is computable from the chunk) are returned."""
    from ai_trader.research.vol_spec import hourly_grid

    if d5.height == 0:
        return pl.DataFrame(schema={"event_ts": TS, "open": pl.Float64, "high": pl.Float64, "low": pl.Float64, "close": pl.Float64,
                                    "rv": pl.Float64, "n5": pl.Int64, "known_ts": TS})
    d5 = (d5.select("event_ts", "open", "high", "low", "close").with_columns(pl.col("event_ts").cast(TS))
          .filter((pl.col("close") > 0) & (pl.col("low") > 0)).unique("event_ts", keep="last").sort("event_ts"))
    g = hourly_grid(d5).select("event_ts", "rv", pl.col("n5").cast(pl.Int64))
    ohlc = (d5.group_by_dynamic("event_ts", every="1h", closed="left", label="left")
            .agg(pl.col("open").first(), pl.col("high").max(), pl.col("low").min(), pl.col("close").last()))
    h = g.join(ohlc, on="event_ts", how="left")
    end = d5["event_ts"].max() + timedelta(minutes=5)          # close of the newest 5m bar
    h = h.filter(pl.col("event_ts") + pl.duration(hours=1) <= end)
    if first_hour is not None:
        h = h.filter(pl.col("event_ts") >= pl.lit(first_hour).cast(TS))
    ok = pl.col("rv").is_not_null()
    return h.with_columns(
        *[pl.when(ok).then(pl.col(c)).otherwise(None).alias(c) for c in ("open", "high", "low", "close")],
        (pl.col("event_ts") + pl.duration(hours=1) + pl.duration(seconds=int(LATENCY.total_seconds()))).alias("known_ts"),
    ).select(*COLS).sort("event_ts")


def _research_5m(symbol: str) -> pl.DataFrame:
    base = load_config().root / "data" / "research"
    files = sorted((base / "bos" / "bars" / "5m").glob(f"{symbol}_*.parquet"))
    ext = base / "intraday_geometry" / "bars" / f"{symbol}_ext.parquet"
    if ext.is_file():
        files.append(ext)
    if not files:
        return pl.DataFrame()
    parts = [pl.read_parquet(f, columns=["event_ts", "open", "high", "low", "close"]).with_columns(pl.col("event_ts").cast(TS)) for f in files]
    return pl.concat(parts).unique("event_ts", keep="last").sort("event_ts")


# ------------------------------------------------------------------ REST


def _get(params: dict, attempts: int = 3) -> list:
    q = URL + "?" + urllib.parse.urlencode(params)
    err = None
    for k in range(attempts):
        try:
            req = urllib.request.Request(q, headers={"User-Agent": "ai-trader-risk-sigma/0.1"})
            with urllib.request.urlopen(req, timeout=10) as r:
                js = json.loads(r.read().decode())
            if js.get("retCode") == 0:
                return js["result"]["list"]
            err = f"bybit retCode {js.get('retCode')}: {js.get('retMsg')}"
        except (OSError, TimeoutError, json.JSONDecodeError) as e:
            err = repr(e)
        time.sleep(1.5 * (k + 1))
    raise RuntimeError(f"5m kline fetch failed: {err}")


def fetch_5m(symbol: str, start: datetime, end: datetime) -> pl.DataFrame:
    """Closed 5m klines with open in [start, end) (a bar is kept only if it closed by `end`)."""
    s_ms = int(start.timestamp() * 1000) // STEP_MS * STEP_MS
    e_ms = int(end.timestamp() * 1000)
    rows: list = []
    s = s_ms
    while s < e_ms:
        e = min(s + 1000 * STEP_MS - 1, e_ms - 1)
        rows.extend(_get({"category": "linear", "symbol": symbol, "interval": "5", "start": s, "end": e, "limit": 1000}))
        s = e + 1
        time.sleep(PAUSE)
    if not rows:
        return pl.DataFrame()
    df = pl.DataFrame({"ts": [int(r[0]) for r in rows], "open": [float(r[1]) for r in rows], "high": [float(r[2]) for r in rows],
                       "low": [float(r[3]) for r in rows], "close": [float(r[4]) for r in rows]}).unique("ts").sort("ts")
    df = df.filter((pl.col("ts") >= s_ms) & (pl.col("ts") + STEP_MS <= e_ms))
    return df.with_columns(pl.from_epoch("ts", time_unit="ms").dt.replace_time_zone("UTC").cast(TS).alias("event_ts")).select(
        "event_ts", "open", "high", "low", "close")


# ------------------------------------------------------------------ store I/O


def _write(symbol: str, df: pl.DataFrame) -> None:
    p = path(symbol)
    tmp = p.with_suffix(f".tmp{os.getpid()}")
    df.select(*COLS).sort("event_ts").write_parquet(tmp, compression="zstd")
    for k in range(20):
        try:
            os.replace(tmp, p)
            return
        except PermissionError:          # a reader has the file open (Windows); retry briefly
            time.sleep(0.25 * (k + 1))
    tmp.unlink(missing_ok=True)
    raise PermissionError(f"could not replace {p}")


def _upsert(symbol: str, new: pl.DataFrame) -> pl.DataFrame:
    p = path(symbol)
    if p.is_file():
        old = pl.read_parquet(p)
        if new.height:
            old = old.filter(~pl.col("event_ts").is_in(new["event_ts"].implode()))
        df = pl.concat([old.select(*COLS), new.select(*COLS)], how="vertical_relaxed") if new.height else old
    else:
        df = new
    df = df.unique("event_ts", keep="last").sort("event_ts")
    _write(symbol, df)
    return df


def seed(symbols: list[str]) -> dict:
    """Build the store from the research 5m caches (read only). Existing stores are left alone."""
    out = {}
    for s in symbols:
        if path(s).is_file():
            out[s] = "exists"
            continue
        d5 = _research_5m(s)
        if d5.height == 0:
            out[s] = "no research 5m cache"
            continue
        h = hours_from_5m(d5)
        _write(s, h)
        out[s] = {"hours": h.height, "first": str(h["event_ts"].min()), "last": str(h["event_ts"].max())}
    return out


def refresh(symbols: list[str], now: datetime | None = None, max_days: int = 40) -> dict:
    """Fetch the 5m bars since the last stored hour (re-fetching the last 2 hours, which replaces them) and append.
    A symbol without a store starts `max_days` back (enough for the 720 h HAR window, not for a refit: seed() first)."""
    now = now or datetime.now(UTC)
    out = {}
    for s in symbols:
        try:
            p = path(s)
            last = pl.read_parquet(p, columns=["event_ts"])["event_ts"].max() if p.is_file() else None
            first_hour = (last - timedelta(hours=1)) if last is not None else (now - timedelta(days=max_days)).replace(minute=0, second=0, microsecond=0)
            first_hour = max(first_hour, (now - timedelta(days=max_days)).replace(minute=0, second=0, microsecond=0))
            d5 = fetch_5m(s, first_hour - timedelta(minutes=5), now)
            h = hours_from_5m(d5, first_hour=first_hour) if d5.height else pl.DataFrame()
            if h.height:
                df = _upsert(s, h)
                out[s] = {"new_hours": h.height, "last": str(df["event_ts"].max())}
            else:
                out[s] = {"new_hours": 0, "last": str(last) if last is not None else None}
        except Exception as e:  # noqa: BLE001 - one symbol's network error must not stop the others
            out[s] = {"error": repr(e)[:300]}
    return out


@lru_cache(maxsize=64)
def _read(p: str, mtime_ns: int) -> pl.DataFrame:
    return pl.read_parquet(p)


def load(symbol: str, as_of: datetime | None = None) -> pl.DataFrame:
    """Stored hours with known_ts <= as_of on a complete hourly calendar (gaps -> null rv / ohlc). Empty if no store."""
    p = path(symbol)
    if not p.is_file():
        return pl.DataFrame(schema={c: (TS if c in ("event_ts", "known_ts") else pl.Int64 if c == "n5" else pl.Float64) for c in COLS})
    df = _read(str(p), p.stat().st_mtime_ns)
    if as_of is not None:
        df = df.filter(pl.col("known_ts") <= pl.lit(as_of).cast(TS))
    if df.height == 0:
        return df
    cal = pl.DataFrame({"event_ts": pl.datetime_range(df["event_ts"].min(), df["event_ts"].max(), "1h", time_zone="UTC", eager=True)}).with_columns(
        pl.col("event_ts").cast(TS))
    return cal.join(df, on="event_ts", how="left").sort("event_ts")


def last_hour(symbol: str) -> datetime | None:
    p = path(symbol)
    if not p.is_file():
        return None
    return pl.read_parquet(p, columns=["event_ts"])["event_ts"].max()


def parkinson(high, low):
    """Parkinson range variance ln(H/L)^2 / (4 ln 2) (polars expressions or numpy arrays)."""
    if isinstance(high, pl.Expr):
        return (high / low).log() ** 2 / (4 * math.log(2))
    import numpy as np

    with np.errstate(all="ignore"):
        return np.log(np.asarray(high, float) / np.asarray(low, float)) ** 2 / (4 * math.log(2))
