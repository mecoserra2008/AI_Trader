"""Bybit public REST data for the shadow xs.v4 (block XS4, H-XS4): universe, 1h klines, funding history, 4h open interest.

    PYTHONIOENCODING=utf-8 uv run python -m ai_trader.research.xs_v4_data universe     (writes data/research/xs_v4/candidates.json)
    ... download   (history from config/live_xs_v4.yaml history_start, resumable, incremental)

Cache (not the lake; research only): data/research/xs_v4/raw/{kline,funding,oi}_<SYMBOL>.parquet. Only closed klines are kept.
Point in time by construction: klines are final once closed (known_ts = close + 2 s), funding known at its settlement + 2 s,
4h OI known at its timestamp + 4 h + 2 s (the lake's conventions). Public endpoints only; no keys.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.parse
import urllib.request
from datetime import UTC, datetime
from pathlib import Path

import polars as pl
import yaml

from ai_trader.core.config import load

API = "https://api.bybit.com"
PAUSE = 0.15
HOUR_MS = 3600_000


def cfg() -> dict:
    return yaml.safe_load((load().root / "config" / "live_xs_v4.yaml").read_text(encoding="utf-8"))


def root() -> Path:
    p = load().root / "data" / "research" / "xs_v4"
    (p / "raw").mkdir(parents=True, exist_ok=True)
    return p


def get(path: str, params: dict) -> dict:
    q = API + path + "?" + urllib.parse.urlencode(params)
    for attempt in range(8):
        try:
            req = urllib.request.Request(q, headers={"User-Agent": "ai-trader-research/0.1"})
            with urllib.request.urlopen(req, timeout=20) as r:
                js = json.loads(r.read().decode())
            if js.get("retCode") == 0:
                time.sleep(PAUSE)
                return js["result"]
            if js.get("retCode") in (10006, 10016, 10018):
                time.sleep(5 * (attempt + 1))
                continue
            raise RuntimeError(f"bybit {path} {params}: {js.get('retCode')} {js.get('retMsg')}")
        except (OSError, TimeoutError, json.JSONDecodeError):
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"failed after retries: {q}")


# ------------------------------------------------------------------ universe


def instruments() -> list[dict]:
    out, cur = [], ""
    while True:
        p = {"category": "linear", "limit": 1000}
        if cur:
            p["cursor"] = cur
        r = get("/v5/market/instruments-info", p)
        out += r["list"]
        cur = r.get("nextPageCursor", "")
        if not cur:
            return out


def snapshot(min_turnover: float, max_spread_bp: float) -> list[dict]:
    """Crypto USDT linear perps (symbolType '', Trading, not pre-listing, not SPY) with their turnover / spread now."""
    c = cfg()
    tk = {t["symbol"]: t for t in get("/v5/market/tickers", {"category": "linear"})["list"]}
    rows = []
    for i in instruments():
        s = i["symbol"]
        t = tk.get(s)
        if (t is None or i["status"] != "Trading" or i["quoteCoin"] != "USDT" or i["contractType"] != "LinearPerpetual"
                or i.get("symbolType", "") != "" or i.get("isPreListing") or s in c["exclude"]):
            continue
        b, a = float(t.get("bid1Price") or 0), float(t.get("ask1Price") or 0)
        if b <= 0 or a <= 0:
            continue
        sp = (a - b) / ((a + b) / 2) * 1e4
        to = float(t["turnover24h"])
        if to >= min_turnover and sp <= max_spread_bp:
            rows.append({"symbol": s, "turnover24h": to, "spread_bp": sp, "launch_ms": int(i["launchTime"]), "funding_interval_min": int(i.get("fundingInterval") or 480)})
    return sorted(rows, key=lambda r: -r["turnover24h"])


def candidates() -> list[str]:
    p = root() / "candidates.json"
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))["symbols"]
    from ai_trader.research.trend_zoo_study import UNIVERSE

    c = cfg()
    snap = snapshot(c["candidate_turnover_usd"], c["max_spread_bp"])
    syms = sorted({r["symbol"] for r in snap} | set(UNIVERSE))
    p.write_text(json.dumps({"as_of": datetime.now(UTC).isoformat(), "rule": "turnover24h >= candidate_turnover_usd and spread <= max_spread_bp now, plus universe L",
                             "symbols": syms, "snapshot": snap}, indent=1), encoding="utf-8")
    return syms


# ------------------------------------------------------------------ downloads


def _read(p: Path) -> pl.DataFrame | None:
    return pl.read_parquet(p) if p.exists() else None


def _write(p: Path, d: pl.DataFrame, key: str) -> None:
    d = d.unique(key, keep="last").sort(key)
    tmp = p.with_suffix(".tmp")
    d.write_parquet(tmp)
    tmp.replace(p)


def klines(sym: str, start_ms: int, now_ms: int) -> pl.DataFrame:
    p = root() / "raw" / f"kline_{sym}.parquet"
    old = _read(p)
    lo = start_ms if old is None or old.height == 0 else int(old["ts"].max()) + HOUR_MS
    rows = []
    end = now_ms
    last_closed = (now_ms // HOUR_MS) * HOUR_MS - HOUR_MS       # newest bar whose close <= now
    while end >= lo:
        r = get("/v5/market/kline", {"category": "linear", "symbol": sym, "interval": "60", "start": lo, "end": end, "limit": 1000})["list"]
        if not r:
            break
        rows += r
        oldest = int(r[-1][0])
        if oldest <= lo or len(r) < 1000:
            break
        end = oldest - 1
    new = pl.DataFrame({"ts": [int(x[0]) for x in rows], "open": [float(x[1]) for x in rows], "high": [float(x[2]) for x in rows],
                        "low": [float(x[3]) for x in rows], "close": [float(x[4]) for x in rows], "volume": [float(x[5]) for x in rows],
                        "turnover": [float(x[6]) for x in rows]}, schema={"ts": pl.Int64, "open": pl.Float64, "high": pl.Float64, "low": pl.Float64,
                                                                         "close": pl.Float64, "volume": pl.Float64, "turnover": pl.Float64})
    new = new.filter((pl.col("ts") >= lo) & (pl.col("ts") <= last_closed))
    d = new if old is None else pl.concat([old, new])
    _write(p, d, "ts")
    return d


def funding(sym: str, start_ms: int, now_ms: int) -> pl.DataFrame:
    p = root() / "raw" / f"funding_{sym}.parquet"
    old = _read(p)
    lo = start_ms if old is None or old.height == 0 else int(old["ts"].max()) + 1
    rows, end = [], now_ms
    while end > lo:
        r = get("/v5/market/funding/history", {"category": "linear", "symbol": sym, "startTime": lo, "endTime": end, "limit": 200})["list"]
        if not r:
            break
        rows += r
        oldest = min(int(x["fundingRateTimestamp"]) for x in r)
        if len(r) < 200 or oldest <= lo:
            break
        end = oldest - 1
    new = pl.DataFrame({"ts": [int(x["fundingRateTimestamp"]) for x in rows], "rate": [float(x["fundingRate"]) for x in rows]},
                       schema={"ts": pl.Int64, "rate": pl.Float64}).filter(pl.col("ts") >= lo)
    d = new if old is None else pl.concat([old, new])
    _write(p, d, "ts")
    return d


def open_interest(sym: str, start_ms: int, now_ms: int) -> pl.DataFrame:
    p = root() / "raw" / f"oi_{sym}.parquet"
    old = _read(p)
    lo = start_ms if old is None or old.height == 0 else int(old["ts"].max()) + 1
    rows, end = [], now_ms
    while end > lo:
        r = get("/v5/market/open-interest", {"category": "linear", "symbol": sym, "intervalTime": "4h", "startTime": lo, "endTime": end, "limit": 200})["list"]
        if not r:
            break
        rows += r
        oldest = min(int(x["timestamp"]) for x in r)
        if len(r) < 200 or oldest <= lo:
            break
        end = oldest - 1
    new = pl.DataFrame({"ts": [int(x["timestamp"]) for x in rows], "oi": [float(x["openInterest"]) for x in rows]},
                       schema={"ts": pl.Int64, "oi": pl.Float64}).filter(pl.col("ts") >= lo)
    d = new if old is None else pl.concat([old, new])
    _write(p, d, "ts")
    return d


def download(symbols: list[str] | None = None, verbose: bool = True) -> None:
    c = cfg()
    start = int(datetime.fromisoformat(c["history_start"]).replace(tzinfo=UTC).timestamp() * 1000)
    now = int(time.time() * 1000)
    for s in symbols or candidates():
        t0 = time.time()
        k = klines(s, start, now)
        f = funding(s, start, now)
        o = open_interest(s, start, now)
        if verbose:
            sys.stdout.write(f"{s}: {k.height} h, {f.height} funding, {o.height} oi, {time.time() - t0:.1f}s\n")
            sys.stdout.flush()


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="xs_v4_data")
    ap.add_argument("cmd", choices=["universe", "download"])
    a = ap.parse_args(argv)
    if a.cmd == "universe":
        print(json.dumps(candidates(), indent=0))
    else:
        download()


if __name__ == "__main__":
    main()
