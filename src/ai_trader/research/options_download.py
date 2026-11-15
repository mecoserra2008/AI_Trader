"""Deribit BTC option trade history -> hourly per-instrument parquet (block OPX data; research only, public API, no keys).

Source: history.deribit.com/api/v2/public/get_last_trades_by_currency_and_time (currency BTC, kind option, count 1000,
sorting asc), paged by time. Raw responses are never written to disk: each UTC day is fetched into memory, aggregated,
and dropped. A month is written once, atomically, to data/research/options/btc_<YYYY-MM>.parquet and recorded in
manifest.json, so nothing is downloaded twice and a machine sleep loses at most the month in progress.

Aggregate per (instrument, hour), block trades excluded (they are negotiated off-book; counted in n_block):
  hour (UTC hour start), instrument, expiry (date, 08:00 UTC settlement), strike, cp, n, volume (BTC amount),
  last_price / vwap / first_price / min_price / max_price (BTC per option), last_iv (%), last_mark (BTC), last_index (USD),
  last_ts (ms), mad_mark (median |price - mark_price|, BTC; the half-spread proxy of options_study), n_block.

Pacing: >= MIN_INTERVAL s between requests (<= ~4 req/s), exponential backoff on 429 / 5xx / network errors.
"""

from __future__ import annotations

import json
import os
import shutil
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "data" / "research" / "options"
URL = "https://history.deribit.com/api/v2/public/get_last_trades_by_currency_and_time"
MIN_INTERVAL = 0.25
BUDGET_DATASET_GB = 3.0
BUDGET_LAKE_GB = 30.0
MONTHS = {m: i + 1 for i, m in enumerate(("JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"))}

_last_req = [0.0]


def _get(params: dict, tries: int = 8) -> dict:
    q = URL + "?" + urllib.parse.urlencode(params)
    wait = 1.0
    for i in range(tries):
        dt = time.time() - _last_req[0]
        if dt < MIN_INTERVAL:
            time.sleep(MIN_INTERVAL - dt)
        _last_req[0] = time.time()
        try:
            with urllib.request.urlopen(urllib.request.Request(q, headers={"User-Agent": "ai-trader-research"}), timeout=40) as r:
                return json.loads(r.read())["result"]
        except urllib.error.HTTPError as e:
            if e.code not in (429, 500, 502, 503, 504) or i == tries - 1:
                raise
        except (urllib.error.URLError, TimeoutError, ConnectionError, OSError):
            if i == tries - 1:
                raise
        time.sleep(wait)
        wait = min(wait * 2, 60)
    raise RuntimeError("unreachable")


def parse_instrument(name: str) -> tuple[date, float, str] | None:
    """'BTC-29MAR24-15000-P' -> (date(2024,3,29), 15000.0, 'P'); None for anything else."""
    p = name.split("-")
    if len(p) != 4 or p[0] != "BTC" or p[3] not in ("C", "P"):
        return None
    e = p[1]
    try:
        d, mon, yy = int(e[:-5]), MONTHS[e[-5:-2]], int(e[-2:])
        return date(2000 + yy, mon, d), float(p[2]), p[3]
    except (KeyError, ValueError):
        return None


def fetch_day(d: date) -> list[dict]:
    """All BTC option trades with timestamp in [d 00:00, d+1 00:00) UTC, deduplicated by trade_id."""
    s = int(datetime(d.year, d.month, d.day, tzinfo=UTC).timestamp() * 1000)
    e = s + 86_400_000 - 1
    seen, out, cur = set(), [], s
    while cur <= e:
        r = _get({"currency": "BTC", "kind": "option", "start_timestamp": cur, "end_timestamp": e,
                  "count": 1000, "sorting": "asc"})
        tr = r.get("trades", [])
        new = [t for t in tr if t["trade_id"] not in seen]
        for t in new:
            seen.add(t["trade_id"])
        out.extend(new)
        if not r.get("has_more") or not tr:
            break
        last = tr[-1]["timestamp"]
        # a full page inside one millisecond would loop forever: step past it (rare; logged by the caller via counts)
        cur = last if new else last + 1
    return out


def aggregate(trades: list[dict]) -> pl.DataFrame:
    rows = []
    for t in trades:
        pi = parse_instrument(t["instrument_name"])
        if pi is None:
            continue
        rows.append((t["timestamp"], t["instrument_name"], pi[0], pi[1], pi[2], float(t["price"]),
                     float(t.get("mark_price") or 0.0), float(t.get("iv") or 0.0), float(t.get("index_price") or 0.0),
                     float(t.get("amount") or 0.0), t.get("block_trade_id") is not None))
    if not rows:
        return pl.DataFrame()
    df = pl.DataFrame(rows, schema=["ts", "instrument", "expiry", "strike", "cp", "price", "mark", "iv", "index", "amount",
                                    "block"], orient="row")
    nb = df.filter(pl.col("block")).group_by("instrument", (pl.col("ts") // 3_600_000).alias("h")).agg(pl.len().alias("n_block"))
    df = df.filter(~pl.col("block")).sort("ts")
    g = (df.group_by("instrument", (pl.col("ts") // 3_600_000).alias("h"))
         .agg(pl.col("expiry").first(), pl.col("strike").first(), pl.col("cp").first(), pl.len().alias("n"),
              pl.col("amount").sum().alias("volume"), pl.col("price").last().alias("last_price"),
              ((pl.col("price") * pl.col("amount")).sum() / pl.col("amount").sum()).alias("vwap"),
              pl.col("price").first().alias("first_price"), pl.col("price").min().alias("min_price"),
              pl.col("price").max().alias("max_price"), pl.col("iv").last().alias("last_iv"),
              pl.col("mark").last().alias("last_mark"), pl.col("index").last().alias("last_index"),
              pl.col("ts").last().alias("last_ts"), (pl.col("price") - pl.col("mark")).abs().median().alias("mad_mark")))
    g = g.join(nb, on=["instrument", "h"], how="full", coalesce=True).with_columns(pl.col("n_block").fill_null(0))
    g = g.filter(pl.col("n").is_not_null())
    return g.with_columns((pl.col("h") * 3_600_000).cast(pl.Datetime("ms")).dt.replace_time_zone("UTC").alias("hour")).drop("h")


def _manifest() -> dict:
    f = OUT / "manifest.json"
    return json.loads(f.read_text()) if f.is_file() else {"months": {}}


def _dir_gb(p: Path) -> float:
    tot = 0
    for dp, _, fs in os.walk(p):
        for f in fs:
            try:
                tot += os.path.getsize(os.path.join(dp, f))
            except OSError:
                pass
    return tot / 1e9


def download_month(y: int, m: int, log=print, lake_gb: float | None = None) -> dict | None:
    key = f"{y:04d}-{m:02d}"
    man = _manifest()
    if key in man["months"]:
        return None
    OUT.mkdir(parents=True, exist_ok=True)
    ds_gb = _dir_gb(OUT)
    free_gb = shutil.disk_usage(OUT).free / 1e9
    if ds_gb > BUDGET_DATASET_GB or (lake_gb is not None and lake_gb + ds_gb > BUDGET_LAKE_GB) or free_gb < 5:
        raise RuntimeError(f"storage budget: dataset {ds_gb:.2f} GB, lake {lake_gb} GB, free {free_gb:.1f} GB")
    d = date(y, m, 1)
    end = date(y + (m == 12), m % 12 + 1, 1)
    today = datetime.now(UTC).date()
    parts, n_tr, t0 = [], 0, time.time()
    while d < end and d < today:
        tr = fetch_day(d)
        n_tr += len(tr)
        a = aggregate(tr)
        if a.height:
            parts.append(a)
        d += timedelta(days=1)
    df = pl.concat(parts, how="vertical_relaxed") if parts else pl.DataFrame()
    f = OUT / f"btc_{key}.parquet"
    tmp = f.with_suffix(".tmp")
    if df.height:
        df.sort("hour", "instrument").write_parquet(tmp, compression="zstd")
        os.replace(tmp, f)
    complete = end <= today
    rec = {"trades": n_tr, "rows": df.height, "bytes": f.stat().st_size if f.is_file() else 0,
           "complete": complete, "last_day": (min(end, today) - timedelta(days=1)).isoformat(),
           "fetched_at": datetime.now(UTC).isoformat(timespec="seconds"), "seconds": round(time.time() - t0)}
    if complete:  # an incomplete (current) month is written but not marked done, so it is re-fetched next time
        man = _manifest()
        man["months"][key] = rec
        (OUT / "manifest.json").write_text(json.dumps(man, indent=1))
    log(f"{key}: {n_tr} trades, {df.height} rows, {rec['bytes'] / 1e6:.1f} MB, {rec['seconds']} s, dataset {ds_gb:.2f} GB")
    return rec


def download(start: str = "2020-01", end: str | None = None, order: list[str] | None = None, log=print) -> None:
    lake_gb = _dir_gb(ROOT / "data")
    log(f"repo data/ = {lake_gb:.2f} GB")
    if order is None:
        y, m = map(int, start.split("-"))
        e = end or datetime.now(UTC).strftime("%Y-%m")
        order = []
        while f"{y:04d}-{m:02d}" <= e:
            order.append(f"{y:04d}-{m:02d}")
            y, m = (y + 1, 1) if m == 12 else (y, m + 1)
    for i, k in enumerate(order):
        if i and i % 6 == 0:
            lake_gb = _dir_gb(ROOT / "data")
        y, m = map(int, k.split("-"))
        download_month(y, m, log=log, lake_gb=lake_gb)


def load(start: datetime | None = None, end: datetime | None = None) -> pl.DataFrame:
    fs = sorted(OUT.glob("btc_*.parquet"))
    if not fs:
        return pl.DataFrame()
    lf = pl.scan_parquet([str(f) for f in fs])
    if start is not None:
        lf = lf.filter(pl.col("hour") >= start)
    if end is not None:
        lf = lf.filter(pl.col("hour") < end)
    return lf.collect()


if __name__ == "__main__":
    import sys

    a = sys.argv[1:]
    download(*(a[:2] if a else []), log=lambda s: print(s, flush=True))
