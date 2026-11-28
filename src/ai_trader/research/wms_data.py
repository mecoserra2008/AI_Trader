"""WMS (program WMM2) data: polite downloader of Bybit public linear klines, funding and open interest.

Public REST only (/v5/market/kline, /funding/history, /open-interest; category linear). Never a private endpoint.
About 6 requests per second per thread, retries with backoff, resumable.

Candidates: Bybit USDT linear perpetuals that are Trading or Closed (delisted perps are served by the kline endpoint, which
reduces survivorship bias), symbolType '' (not innovation zone, not stock / ETF), not BTC / ETH / stablecoins / gold tokens,
never SPYUSDT. The listing snapshot is WS02's public listing (data/research/ws02/instruments_live_2026-11-26.json).

Outputs (data/research/wms/):
  daily.parquet           1d klines of every candidate from 2020-12-01 (symbol, day, OHLC, volume, turnover)
  bars5m/<SYM>.parquet    5m klines for the symbol-weeks in the download band (see `plan_5m`)
  funding.parquet, oi.parquet

    uv run python -m ai_trader.research.wms_data daily
    uv run python -m ai_trader.research.wms_data plan
    uv run python -m ai_trader.research.wms_data fives --threads 2
    uv run python -m ai_trader.research.wms_data deriv --threads 2
"""

from __future__ import annotations

import argparse
import json
import threading
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta
from pathlib import Path

import polars as pl

from ai_trader.core.config import load

API = "https://api.bybit.com/v5/market"
PAUSE = 0.16
EXCLUDE_BASES = {"BTC", "ETH", "ETHBTC", "USDC", "USDE", "USD1", "RLUSD", "FDUSD", "PYUSD", "TUSD", "DAI", "USDT", "PAXG", "XAUT",
                 "SPY", "XAU", "XAG", "XPT", "XPD", "CL", "BZ", "NG", "COPPER", "EURUSD", "GBPUSD", "USDJPY"}
START = datetime(2020, 12, 1, tzinfo=UTC)
DL_BAND = (5e6, 50e6)  # download band = analysis band (USD median daily turnover over the 7 days before the Friday)


def root() -> Path:
    p = load().root / "data" / "research" / "wms"
    p.mkdir(parents=True, exist_ok=True)
    return p


def ms(dt: datetime) -> int:
    return int(dt.timestamp() * 1000)


def _get(path: str, params: dict) -> dict:
    q = f"{API}/{path}?" + urllib.parse.urlencode(params)
    for attempt in range(8):
        try:
            req = urllib.request.Request(q, headers={"User-Agent": "ai-trader-research/0.1"})
            with urllib.request.urlopen(req, timeout=20) as r:
                js = json.loads(r.read().decode())
            if js.get("retCode") == 0:
                return js["result"]
            if js.get("retCode") in (10006, 10016, 10018):
                time.sleep(5 * (attempt + 1))
                continue
            return {"list": [], "error": f"{js.get('retCode')}: {js.get('retMsg')}"}
        except (OSError, TimeoutError, json.JSONDecodeError):
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"failed after retries: {q}")


def candidates() -> list[dict]:
    p = load().root / "data" / "research" / "ws02" / "instruments_live_2026-11-26.json"
    rows = json.loads(p.read_text())["rows"]
    out, seen = [], set()
    for r in rows:
        if r["category"] != "linear" or r.get("contractType") != "LinearPerpetual" or r.get("quoteCoin") != "USDT":
            continue
        if r["status"] not in ("Trading", "Closed") or r.get("symbolType", "") != "":
            continue
        s = r["symbol"]
        if not s.endswith("USDT") or s[:-4] in EXCLUDE_BASES or s == "SPYUSDT" or s in seen:
            continue
        seen.add(s)
        out.append({"symbol": s, "status": r["status"], "launch_ms": int(r.get("launchTime") or 0),
                    "delivery_ms": int(r.get("deliveryTime") or 0)})
    return out


def klines(sym: str, interval: str, start_ms: int, end_ms: int, step_ms: int) -> list[list]:
    rows: list = []
    s = start_ms
    while s < end_ms:
        e = min(s + 1000 * step_ms - 1, end_ms - 1)
        res = _get("kline", {"category": "linear", "symbol": sym, "interval": interval, "start": s, "end": e, "limit": 1000})
        rows.extend(res.get("list", []))
        s = e + 1
        time.sleep(PAUSE)
    return rows


def _frame(sym: str, rows: list[list], step_ms: int, end_ms: int) -> pl.DataFrame:
    if not rows:
        return pl.DataFrame()
    df = pl.DataFrame({"ts": [int(r[0]) for r in rows], "open": [float(r[1]) for r in rows], "high": [float(r[2]) for r in rows],
                       "low": [float(r[3]) for r in rows], "close": [float(r[4]) for r in rows],
                       "volume": [float(r[5]) for r in rows], "turnover": [float(r[6]) for r in rows]})
    df = df.unique("ts").sort("ts").filter(pl.col("ts") + step_ms <= end_ms)  # closed bars only
    return df.with_columns(pl.lit(sym).alias("symbol"))


def daily() -> None:
    now = ms(datetime.now(UTC)) // 86_400_000 * 86_400_000
    out = root() / "daily.parquet"
    cands = candidates()
    lock, frames = threading.Lock(), []

    def one(c: dict) -> None:
        st = max(ms(START), c["launch_ms"] // 86_400_000 * 86_400_000)
        en = min(now, c["delivery_ms"] + 86_400_000) if c["status"] == "Closed" and c["delivery_ms"] else now
        if en <= st:
            return
        df = _frame(c["symbol"], klines(c["symbol"], "D", st, en, 86_400_000), 86_400_000, en)
        if df.height:
            with lock:
                frames.append(df.with_columns(pl.lit(c["status"]).alias("status")))

    with ThreadPoolExecutor(2) as ex:
        list(ex.map(one, cands))
    d = pl.concat(frames).with_columns(pl.from_epoch("ts", time_unit="ms").dt.date().alias("day"))
    d.write_parquet(out, compression="zstd")
    print(f"daily: {d['symbol'].n_unique()} symbols, {d.height} rows -> {out}")


def saturdays(first: datetime = datetime(2021, 1, 2, tzinfo=UTC), last: datetime | None = None) -> list[datetime]:
    last = last or datetime.now(UTC)
    out, d = [], first
    while d <= last:
        out.append(d)
        d += timedelta(days=7)
    return out


def band_table(d: pl.DataFrame, lo: float, hi: float, min_age_days: int = 30) -> pl.DataFrame:
    """Friday-known band membership per (weekend, symbol): median daily turnover over the 7 complete UTC days before the
    Friday (prev Fri .. Thu) in [lo, hi], and first daily bar at least `min_age_days` before the Friday."""
    first = d.group_by("symbol").agg(pl.col("day").min().alias("first_day"))
    rows = []
    for sat in saturdays():
        fri = (sat - timedelta(days=1)).date()
        win = d.filter((pl.col("day") >= fri - timedelta(days=7)) & (pl.col("day") <= fri - timedelta(days=1)))
        g = win.group_by("symbol").agg(pl.col("turnover").median().alias("med_turnover"), pl.len().alias("n_days"))
        g = g.join(first, on="symbol").filter((pl.col("n_days") == 7) & (pl.col("first_day") <= fri - timedelta(days=min_age_days)))
        g = g.filter((pl.col("med_turnover") >= lo) & (pl.col("med_turnover") <= hi))
        rows.append(g.with_columns(pl.lit(sat.date()).alias("weekend_start")))
    return pl.concat(rows).select("weekend_start", "symbol", "med_turnover")


def plan_5m() -> pl.DataFrame:
    """Download plan: per symbol, merged ranges [Fri-7d 21:00, Mon 00:00) of the weekends in the download band."""
    d = pl.read_parquet(root() / "daily.parquet")
    b = band_table(d, *DL_BAND)
    rows = []
    for (sym,), g in b.group_by(["symbol"]):
        rng = []
        for ws in sorted(g["weekend_start"].to_list()):
            sat = datetime(ws.year, ws.month, ws.day, tzinfo=UTC)
            a, z = sat - timedelta(days=7, hours=3), sat + timedelta(days=2)  # 1 h buffer before the 7-day window
            if rng and ms(a) <= rng[-1][1]:
                rng[-1][1] = max(rng[-1][1], ms(z))
            else:
                rng.append([ms(a), ms(z)])
        rows += [{"symbol": sym, "start_ms": a, "end_ms": z} for a, z in rng]
    p = pl.DataFrame(rows).sort("symbol", "start_ms")
    p.write_parquet(root() / "plan_5m.parquet")
    n_calls = int(((p["end_ms"] - p["start_ms"]) / (300_000 * 1000)).ceil().sum())
    print(f"band {DL_BAND}: {b.height} symbol-weekends, {b['symbol'].n_unique()} symbols, {p.height} ranges, ~{n_calls} calls")
    return p


def fives(threads: int = 2, symbols: list[str] | None = None) -> None:
    p = pl.read_parquet(root() / "plan_5m.parquet")
    for extra in symbols or []:  # whole-history references (BTC) are added explicitly
        p = pl.concat([p, pl.DataFrame({"symbol": [extra], "start_ms": [ms(START)], "end_ms": [ms(datetime.now(UTC))]})])
    out = root() / "bars5m"
    out.mkdir(exist_ok=True)
    mpath = root() / "manifest_5m.json"
    man = json.loads(mpath.read_text()) if mpath.exists() else {}
    lock = threading.Lock()
    now = ms(datetime.now(UTC)) // 300_000 * 300_000

    def one(sym: str) -> None:
        f = out / f"{sym}.parquet"
        rngs = p.filter(pl.col("symbol") == sym).sort("start_ms")
        key = ",".join(f"{a}-{z}" for a, z in zip(rngs["start_ms"], rngs["end_ms"]))
        with lock:
            if f.exists() and man.get(sym, {}).get("key") == key:
                return
        t0, frames = time.time(), []
        for a, z in zip(rngs["start_ms"], rngs["end_ms"]):
            z = min(z, now)
            df = _frame(sym, klines(sym, "5", a, z, 300_000), 300_000, z)
            if df.height:
                frames.append(df)
        if frames:
            df = pl.concat(frames).unique("ts").sort("ts").select(
                "ts", *[pl.col(c).cast(pl.Float32) for c in ("high", "low", "close", "turnover")])
            # float32 keeps ~7 significant digits: enough for log returns at tick precision (checked in the report)
            df.write_parquet(f, compression="zstd", compression_level=9)
        with lock:
            man[sym] = {"key": key, "rows": sum(x.height for x in frames), "bytes": f.stat().st_size if f.exists() else 0,
                        "secs": round(time.time() - t0), "fetched_at": datetime.now(UTC).isoformat(timespec="seconds")}
            mpath.write_text(json.dumps(man, indent=0))
            tot = sum(v["bytes"] for v in man.values())
            print(f"{sym} rows={man[sym]['rows']} {man[sym]['secs']}s total={tot / 1e6:.0f}MB done={len(man)}", flush=True)
            if tot > 800e6:
                raise SystemExit("5m byte budget reached")

    syms = sorted(p["symbol"].unique().to_list())
    with ThreadPoolExecutor(threads) as ex:
        list(ex.map(one, syms))


def deriv(threads: int = 2, extra: list[str] | None = None) -> None:
    """Funding history and 1d open interest over each planned 5m range (plus 9 days before it, for 7-day changes); `extra`
    symbols (BTC) over the whole span."""
    p = pl.read_parquet(root() / "plan_5m.parquet").select("symbol", "start_ms", "end_ms")
    for x in extra or []:
        p = pl.concat([p, pl.DataFrame({"symbol": [x], "start_ms": [ms(START) + 9 * 86_400_000], "end_ms": [ms(datetime.now(UTC))]})])
    lock, fund, oi = threading.Lock(), [], []
    now = ms(datetime.now(UTC))

    def one(sym: str) -> None:
        f, o = [], []
        for a, z in zip(*[p.filter(pl.col("symbol") == sym)[c].to_list() for c in ("start_ms", "end_ms")]):
            a, z = a - 9 * 86_400_000, min(z, now)
            e = z
            while e > a:  # funding: page backwards by endTime, 200 settlements per call (any interval)
                res = _get("funding/history", {"category": "linear", "symbol": sym, "startTime": a, "endTime": e, "limit": 200})
                lst = res.get("list", [])
                f += [(sym, int(x["fundingRateTimestamp"]), float(x["fundingRate"])) for x in lst]
                time.sleep(PAUSE)
                if len(lst) < 200:
                    break
                e = min(int(x["fundingRateTimestamp"]) for x in lst) - 1
            s = a
            while s < z:
                e = min(s + 199 * 86_400_000, z)
                res = _get("open-interest", {"category": "linear", "symbol": sym, "intervalTime": "1d", "startTime": s, "endTime": e,
                                             "limit": 200})
                o += [(sym, int(x["timestamp"]), float(x["openInterest"])) for x in res.get("list", [])]
                s = e + 1
                time.sleep(PAUSE)
        with lock:
            fund.extend(f)
            oi.extend(o)
            print(f"{sym} funding={len(f)} oi={len(o)}", flush=True)

    with ThreadPoolExecutor(threads) as ex:
        list(ex.map(one, sorted(p["symbol"].unique().to_list())))
    pl.DataFrame(fund, schema=["symbol", "ts", "rate"], orient="row").unique().sort("symbol", "ts").write_parquet(root() / "funding.parquet")
    pl.DataFrame(oi, schema=["symbol", "ts", "oi"], orient="row").unique().sort("symbol", "ts").write_parquet(root() / "oi.parquet")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["daily", "plan", "fives", "deriv"])
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--extra", default="")
    a = ap.parse_args()
    if a.step == "daily":
        daily()
    elif a.step == "plan":
        plan_5m()
    elif a.step == "fives":
        fives(a.threads, [s for s in a.extra.split(",") if s])
    else:
        deriv(a.threads, [s for s in a.extra.split(",") if s])


if __name__ == "__main__":
    main()
