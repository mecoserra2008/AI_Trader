"""GET-only public market-data clients (Bybit v5 /market, Deribit /public). No API keys, no private or trading endpoint.

Every path is checked against a whitelist before a request is made, so a trading call cannot be issued from here by
mistake. Responses are stamped with the local receive time (`known_ts`), which the snapshot uses as its data cut.
"""

from __future__ import annotations

import json
import threading
import time
import urllib.parse
import urllib.request
from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta

import polars as pl

BYBIT = "https://api.bybit.com"
DERIBIT = "https://www.deribit.com"
UA = "AI_Trader desk (research, public market data)"
BYBIT_ALLOWED = frozenset({
    "/v5/market/tickers", "/v5/market/kline", "/v5/market/funding/history", "/v5/market/open-interest",
    "/v5/market/orderbook", "/v5/market/instruments-info",
})
DERIBIT_ALLOWED = frozenset({"/api/v2/public/get_book_summary_by_currency"})
INTERVAL = {"1m": "1", "5m": "5", "15m": "15", "1h": "60", "4h": "240", "1d": "D"}
TF_DELTA = {"1m": timedelta(minutes=1), "5m": timedelta(minutes=5), "15m": timedelta(minutes=15), "1h": timedelta(hours=1),
            "4h": timedelta(hours=4), "1d": timedelta(days=1)}
WORKERS = 2


class PublicError(RuntimeError):
    pass


def pmap(fn: Callable, items: Iterable, workers: int = WORKERS) -> list:
    """Map with at most `workers` threads (resource limit: <= 2 threads). Exceptions are returned, not raised."""
    def safe(x):
        try:
            return fn(x)
        except Exception as e:  # noqa: BLE001 - one bad symbol never stops the snapshot
            return e
    items = list(items)
    if workers <= 1 or len(items) <= 1:
        return [safe(x) for x in items]
    with ThreadPoolExecutor(max_workers=workers) as ex:
        return list(ex.map(safe, items))


class Http:
    """Throttled JSON GET with retries. `get_raw` is injectable for tests."""

    def __init__(self, min_gap: float = 0.03, timeout: float = 15.0, attempts: int = 3, get_raw: Callable[[str], bytes] | None = None):
        self.min_gap, self.timeout, self.attempts = min_gap, timeout, attempts
        self._lock = threading.Lock()
        self._last = 0.0
        self._get_raw = get_raw
        self.calls = 0
        self.last_known: datetime | None = None

    def _raw(self, url: str) -> bytes:
        if self._get_raw is not None:
            return self._get_raw(url)
        with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": UA}), timeout=self.timeout) as r:
            return r.read()

    def get(self, url: str) -> tuple[dict, datetime]:
        err = None
        for i in range(self.attempts):
            with self._lock:
                wait = self._last + self.min_gap - time.monotonic()
                if wait > 0:
                    time.sleep(wait)
                self._last = time.monotonic()
                self.calls += 1
            try:
                d = json.loads(self._raw(url))
                kt = datetime.now(UTC)
                with self._lock:
                    self.last_known = kt if self.last_known is None else max(self.last_known, kt)
                return d, kt
            except Exception as e:  # noqa: BLE001
                err = e
                time.sleep(0.5 * (2 ** i))
        raise PublicError(f"GET {url[:120]} failed: {err}")


class Bybit:
    def __init__(self, http: Http | None = None, base: str = BYBIT):
        self.http = http or Http()
        self.base = base

    def get(self, path: str, **params) -> tuple[dict, datetime]:
        if path not in BYBIT_ALLOWED:
            raise PublicError(f"{path} is not a whitelisted public market endpoint")
        d, kt = self.http.get(f"{self.base}{path}?{urllib.parse.urlencode(params)}")
        if d.get("retCode") != 0:
            raise PublicError(f"{path} {params}: retCode {d.get('retCode')} {d.get('retMsg')}")
        return d["result"], kt

    def tickers(self) -> tuple[list[dict], datetime]:
        r, kt = self.get("/v5/market/tickers", category="linear")
        return r["list"], kt

    def instruments(self) -> list[dict]:
        out, cursor = [], ""
        for _ in range(10):
            p = {"category": "linear", "limit": 1000}
            if cursor:
                p["cursor"] = cursor
            r, _kt = self.get("/v5/market/instruments-info", **p)
            out += r["list"]
            cursor = r.get("nextPageCursor") or ""
            if not cursor:
                break
        return out

    def kline(self, symbol: str, tf: str, limit: int = 200, end: datetime | None = None) -> pl.DataFrame:
        p = {"category": "linear", "symbol": symbol, "interval": INTERVAL[tf], "limit": min(limit, 1000)}
        if end is not None:
            p["end"] = int(end.timestamp() * 1000)
        r, kt = self.get("/v5/market/kline", **p)
        return kline_frame(r["list"], tf, kt)

    def kline_n(self, symbol: str, tf: str, n: int) -> pl.DataFrame:
        """Up to n bars (paged backwards in chunks of 1000)."""
        frames, end = [], None
        while n > 0:
            f = self.kline(symbol, tf, min(n, 1000), end)
            if f.height == 0:
                break
            frames.append(f)
            n -= f.height
            end = f["event_ts"].min() - timedelta(milliseconds=1)
            if f.height < 1000:
                break
        return pl.concat(frames).unique("event_ts").sort("event_ts") if frames else kline_frame([], tf, datetime.now(UTC))

    def funding_history(self, symbol: str, limit: int = 200) -> pl.DataFrame:
        r, kt = self.get("/v5/market/funding/history", category="linear", symbol=symbol, limit=limit)
        rows = [{"event_ts": datetime.fromtimestamp(int(x["fundingRateTimestamp"]) / 1000, UTC), "rate": float(x["fundingRate"])} for x in r["list"]]
        return _frame(rows, {"event_ts": pl.Datetime("us", "UTC"), "rate": pl.Float64}, kt)

    def open_interest(self, symbol: str, interval: str = "1h", limit: int = 200) -> pl.DataFrame:
        r, kt = self.get("/v5/market/open-interest", category="linear", symbol=symbol, intervalTime=interval, limit=limit)
        rows = [{"event_ts": datetime.fromtimestamp(int(x["timestamp"]) / 1000, UTC), "oi": float(x["openInterest"])} for x in r["list"]]
        return _frame(rows, {"event_ts": pl.Datetime("us", "UTC"), "oi": pl.Float64}, kt)

    def orderbook(self, symbol: str, limit: int = 50) -> dict:
        r, kt = self.get("/v5/market/orderbook", category="linear", symbol=symbol, limit=limit)
        return {"bids": [(float(p), float(q)) for p, q in r.get("b", [])], "asks": [(float(p), float(q)) for p, q in r.get("a", [])],
                "ts": datetime.fromtimestamp(int(r.get("ts", 0)) / 1000, UTC), "known_ts": kt}


class Deribit:
    def __init__(self, http: Http | None = None, base: str = DERIBIT):
        self.http = http or Http()
        self.base = base

    def option_book(self, currency: str) -> tuple[list[dict], datetime]:
        path = "/api/v2/public/get_book_summary_by_currency"
        if path not in DERIBIT_ALLOWED:
            raise PublicError(path)
        d, kt = self.http.get(f"{self.base}{path}?{urllib.parse.urlencode({'currency': currency, 'kind': 'option'})}")
        return d.get("result", []), kt


def _frame(rows: list[dict], schema: dict, kt: datetime) -> pl.DataFrame:
    f = pl.DataFrame(rows, schema=schema) if rows else pl.DataFrame(schema=schema)
    return f.with_columns(pl.lit(kt).cast(pl.Datetime("us", "UTC")).alias("known_ts")).sort("event_ts")


def kline_frame(rows: list[list], tf: str, kt: datetime) -> pl.DataFrame:
    """Bybit kline rows [start, o, h, l, c, volume, turnover] -> event_ts / close_ts / ohlcv / known_ts frame (sorted).

    A bar's close_ts = start + interval; for a still-forming bar close_ts > known_ts (callers split it off by as_of)."""
    schema = {"event_ts": pl.Datetime("us", "UTC"), "open": pl.Float64, "high": pl.Float64, "low": pl.Float64, "close": pl.Float64,
              "volume": pl.Float64, "turnover": pl.Float64}
    recs = [{"event_ts": datetime.fromtimestamp(int(x[0]) / 1000, UTC), "open": float(x[1]), "high": float(x[2]), "low": float(x[3]),
             "close": float(x[4]), "volume": float(x[5]), "turnover": float(x[6]) if len(x) > 6 else None} for x in rows]
    f = pl.DataFrame(recs, schema=schema) if recs else pl.DataFrame(schema=schema)
    return (f.with_columns((pl.col("event_ts") + pl.duration(milliseconds=int(TF_DELTA[tf].total_seconds() * 1000))).alias("close_ts"),
                           pl.lit(kt).cast(pl.Datetime("us", "UTC")).alias("known_ts"))
            .unique("event_ts").sort("event_ts"))


def closed(bars: pl.DataFrame, as_of: datetime) -> pl.DataFrame:
    """Bars fully closed at as_of (close_ts <= as_of). The point-in-time filter used by every desk compute function."""
    if bars.height == 0:
        return bars
    return bars.filter(pl.col("close_ts") <= as_of).sort("event_ts")


def forming_open(bars: pl.DataFrame, as_of: datetime) -> float | None:
    """Open of the bar that contains as_of (its first trade is known; its high / low / close are not)."""
    if bars.height == 0:
        return None
    f = bars.filter((pl.col("event_ts") <= as_of) & (pl.col("close_ts") > as_of))
    return float(f["open"][-1]) if f.height else None


__all__ = ["Bybit", "Deribit", "Http", "PublicError", "closed", "forming_open", "kline_frame", "pmap"]
