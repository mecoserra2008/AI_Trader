"""Cross-venue tape tracker: public REST polling (no keys), every 10 s; one summary line per symbol every 60 s.

    .venv/Scripts/python.exe scripts/xvenue_tape.py [--symbols KAVA SOPH] [--minutes N]

Venues (taker side convention per venue, normalised to sign +1 = taker buy):
    by  Bybit linear perp   /v5/market/recent-trade (latest 1000 fills, de-duplicated by execId; no paging -> gap counter)
    bn  Binance USD-M perp  /fapi/v1/aggTrades with fromId paging (no gaps)
    bs  Binance spot        /api/v3/aggTrades with fromId paging (no gaps)
    ok  OKX swap            /api/v5/market/trades + history-trades paging; contracts -> base via the instrument's ctVal
    cb  Coinbase spot (USD) /products/{p}/trades + `after` paging; Coinbase `side` is the MAKER side -> taker = opposite
    hl  Hyperliquid perp    info recentTrades (only the last ~10 trades per call -> SAMPLED, excluded from flow alerts)
    db  Deribit perp        public/get_last_trades_by_instrument with start_seq paging (BTC, ETH only)
    up  Upbit KRW spot      /v1/trades/ticks (KRW -> USDT via Upbit KRW-USDT)

A "print" = the fills of one venue sharing a millisecond timestamp and taker side (approximates one taker order).
small / large / whale = print notional <= p95 / > p95 / > p99 of that venue's own prints over the last 60 min, so the
buckets are venue-relative. Premiums and bases are log ratios in bp: 1e4 * ln(P_venue / P_bybit_perp_mid).

THE LABELS ARE DESCRIPTIVE, NOT TESTED SIGNALS. No hypothesis on these quantities has been registered or tested; an
ALERT says "this is unusual versus the last hour", not "trade this".

Outputs: logs/xvenue_tape.log (header + one line per symbol per minute + ALERT lines), data/journal/xvenue/<date>.jsonl,
logs/xvenue_tape.pid. Two threads at most (main + one worker). Backs off on HTTP 429/418.
"""

from __future__ import annotations

import argparse
import gzip
import http.client
import json
import math
import os
import ssl
import statistics
import sys
import threading
import time
from collections import defaultdict, deque
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urlencode, urlsplit

ROOT = Path(__file__).resolve().parents[1]
LOG = ROOT / "logs" / "xvenue_tape.log"
PIDF = ROOT / "logs" / "xvenue_tape.pid"
JDIR = ROOT / "data" / "journal" / "xvenue"
DEFAULT = ["BTC", "ETH", "SOL", "XRP", "DOGE", "SUI", "ZEC", "HYPE"]
UA = "Mozilla/5.0 (AI_Trader research; public market data)"
CTX = ssl.create_default_context()
POLL_S = 10                       # target; measured 15-17 s per cycle from this host (~0.4 s RTT x ~55 requests / 2 threads)
BUF_MS = 60 * 60 * 1000          # keep 60 min of prints
STALE_RESET_S = 300              # no successful poll for 5 min (suspend, outage): drop the paging cursor
RETRY_DISABLED_S = 600           # a rejected feed is probed again after 10 min (40 min if it never worked)
WINDOWS = {"1m": 60_000, "5m": 300_000, "15m": 900_000}
VENUES = ["by", "bn", "bs", "ok", "cb", "hl", "db", "up"]
SAMPLED = {"hl"}                  # partial coverage: never used to trigger flow alerts
PERP = {"by", "bn", "ok", "hl", "db"}


class RateLimited(Exception):
    pass


class BadSymbol(Exception):
    pass


class Http:
    """Persistent HTTPS connections, one per host, owned by one thread. gzip; per-host 429 back-off."""

    pause_until: dict[str, float] = {}
    lock = threading.Lock()

    def __init__(self) -> None:
        self.conns: dict[str, http.client.HTTPSConnection] = {}

    def req(self, url: str, body: dict | None = None, timeout: float = 10.0):
        u = urlsplit(url)
        host, path = u.netloc, u.path + ("?" + u.query if u.query else "")
        with Http.lock:
            if Http.pause_until.get(host, 0) > time.time():
                raise RateLimited(host)
        last: Exception | None = None
        for attempt in range(3):
            c = self.conns.get(host)
            if c is None:
                c = http.client.HTTPSConnection(host, timeout=timeout, context=CTX)
                self.conns[host] = c
            try:
                hdr = {"User-Agent": UA, "Accept-Encoding": "gzip", "Accept": "application/json"}
                if body is not None:
                    hdr["Content-Type"] = "application/json"
                    c.request("POST", path, body=json.dumps(body), headers=hdr)
                else:
                    c.request("GET", path, headers=hdr)
                r = c.getresponse()
                data = r.read()
                if r.getheader("Content-Encoding") == "gzip":
                    data = gzip.decompress(data)
                if r.status in (429, 418):
                    wait = float(r.getheader("Retry-After") or 30)
                    with Http.lock:
                        Http.pause_until[host] = time.time() + min(max(wait, 10), 300)
                    raise RateLimited(f"{host} {r.status}")
                if r.status in (400, 404):
                    raise BadSymbol(f"{r.status} {data[:160]!r}")
                if r.status >= 500:
                    raise OSError(f"{host} {r.status}")
                return json.loads(data)
            except (http.client.HTTPException, OSError, json.JSONDecodeError) as e:
                last = e
                c.close()
                self.conns.pop(host, None)
                time.sleep(0.4 * (attempt + 1))
        raise OSError(f"{host}: {last!r}")


# ---------------------------------------------------------------- trade buffers

class Buf:
    """Prints (ts_ms, px, usd_signed, usd_abs) for one venue/symbol, oldest first, 60 min."""

    def __init__(self) -> None:
        self.prints: deque[list] = deque()
        self.last_id: int | None = None       # monotonic-id venues
        self.seen: deque = deque(maxlen=6000)  # non-monotonic ids (Bybit execId, HL tid)
        self.seen_set: set = set()
        self.last_new = 1000
        self.gaps = 0                          # polls where a gap could not be closed
        self.polls = 0
        self.ok = True                         # symbol listed on the venue
        self.err = ""
        self.last_ok_ts = 0.0                  # wall clock of the last successful poll
        self.retry_at = 0.0                    # when a disabled feed is probed again

    def reset_cursor(self) -> None:
        """Forget paging state (after a suspend or a long outage the old cursor is rejected or pages forever)."""
        self.last_id = None
        self.seen.clear()
        self.seen_set.clear()
        self.polls = 0

    def seen_add(self, k) -> bool:
        if k in self.seen_set:
            return False
        if len(self.seen) == self.seen.maxlen:
            self.seen_set.discard(self.seen[0])
        self.seen.append(k)
        self.seen_set.add(k)
        return True

    def add(self, trades: list[tuple[int, float, float, int]]) -> None:
        """trades: (ts_ms, px_usd, qty_base, sign). Merges same-ms same-side fills into one print."""
        trades.sort(key=lambda t: t[0])
        for ts, px, q, s in trades:
            usd = px * q
            p = self.prints[-1] if self.prints else None
            if p is not None and p[0] == ts and (p[2] > 0) == (s > 0):
                tot = p[3] + usd
                p[1] = (p[1] * p[3] + px * usd) / tot if tot > 0 else px
                p[2] += s * usd
                p[3] = tot
            elif p is not None and ts < p[0]:
                self.prints.append([p[0], px, s * usd, usd])   # late arrival: keep order by clamping ts
            else:
                self.prints.append([ts, px, s * usd, usd])
        cut = int(time.time() * 1000) - BUF_MS
        while self.prints and self.prints[0][0] < cut:
            self.prints.popleft()


# ---------------------------------------------------------------- venue fetchers

class Venues:
    def __init__(self, syms: list[str]) -> None:
        self.syms = syms
        self.buf: dict[tuple[str, str], Buf] = {(v, s): Buf() for v in VENUES for s in syms}
        for s in syms:
            if s not in ("BTC", "ETH"):
                self.buf[("db", s)].ok = False
                self.buf[("db", s)].err = "not listed"
        self.okx_ctval: dict[str, float] = {}
        self.usdtkrw: float | None = None
        self.perf: dict[str, float] = {}       # seconds spent per venue in the current cycle

    # each fetcher appends trades to its buffer; raise BadSymbol when not listed
    def bybit(self, h: Http, s: str, b: Buf) -> None:
        lim = 1000 if not b.polls else int(min(1000, max(100, 2 * b.last_new + 60)))   # adaptive: smaller payloads
        j = h.req(f"https://api.bybit.com/v5/market/recent-trade?category=linear&symbol={s}USDT&limit={lim}")
        if j.get("retCode") != 0:
            raise BadSymbol(j.get("retMsg"))
        lst = j["result"]["list"]
        new = [t for t in lst if b.seen_add(t["execId"])]
        if b.polls and lst and len(new) == len(lst):
            if lim < 1000:                                # window too small: refetch the full 1000 now
                j = h.req(f"https://api.bybit.com/v5/market/recent-trade?category=linear&symbol={s}USDT&limit=1000")
                lst = j["result"]["list"]
                new += [t for t in lst if b.seen_add(t["execId"])]
            if len(new) >= len(lst):
                b.gaps += 1                               # no overlap with the previous poll: possible gap
        b.last_new = len(new)
        b.add([(int(t["time"]), float(t["price"]), float(t["size"]), 1 if t["side"] == "Buy" else -1) for t in new])

    def _binance(self, h: Http, base: str, s: str, b: Buf) -> None:
        out = []
        if b.last_id is None:
            j = h.req(f"{base}?symbol={s}USDT&limit=1000")
        else:
            j = h.req(f"{base}?symbol={s}USDT&fromId={b.last_id + 1}&limit=1000")
        pages = 1
        while True:
            for t in j:
                out.append((int(t["T"]), float(t["p"]), float(t["q"]), -1 if t["m"] else 1))
            if j:
                b.last_id = max(b.last_id or 0, int(j[-1]["a"]))
            if len(j) < 1000 or pages >= 6:
                if len(j) == 1000:
                    b.gaps += 1
                break
            time.sleep(0.05)
            j = h.req(f"{base}?symbol={s}USDT&fromId={b.last_id + 1}&limit=1000")
            pages += 1
        b.add(out)

    def binance_perp(self, h: Http, s: str, b: Buf) -> None:
        self._binance(h, "https://fapi.binance.com/fapi/v1/aggTrades", s, b)

    def binance_spot(self, h: Http, s: str, b: Buf) -> None:
        self._binance(h, "https://api.binance.com/api/v3/aggTrades", s, b)

    def okx(self, h: Http, s: str, b: Buf) -> None:
        inst = f"{s}-USDT-SWAP"
        if inst not in self.okx_ctval:
            j = h.req(f"https://www.okx.com/api/v5/public/instruments?instType=SWAP&instId={inst}")
            if j.get("code") != "0" or not j.get("data"):
                raise BadSymbol(j.get("msg"))
            d = j["data"][0]
            if d.get("ctValCcy") != s:
                raise BadSymbol(f"ctValCcy {d.get('ctValCcy')} != {s}")
            self.okx_ctval[inst] = float(d["ctVal"]) * float(d.get("ctMult") or 1)
        cv = self.okx_ctval[inst]
        j = h.req(f"https://www.okx.com/api/v5/market/trades?instId={inst}&limit=500")
        if j.get("code") != "0":
            raise BadSymbol(j.get("msg"))
        rows = j["data"]
        if b.last_id is not None and rows:
            lo = min(int(t["tradeId"]) for t in rows)
            pages = 0
            while lo > b.last_id + 1 and pages < 4:
                k = h.req(f"https://www.okx.com/api/v5/market/history-trades?instId={inst}&type=1&after={lo}&limit=100")
                more = k.get("data") or []
                if not more:
                    break
                rows += more
                lo = min(int(t["tradeId"]) for t in more)
                pages += 1
            if lo > b.last_id + 1:
                b.gaps += 1
        new = [t for t in rows if b.last_id is None or int(t["tradeId"]) > b.last_id]
        seen = set()
        out = []
        for t in new:
            if t["tradeId"] in seen:
                continue
            seen.add(t["tradeId"])
            out.append((int(t["ts"]), float(t["px"]), float(t["sz"]) * cv, 1 if t["side"] == "buy" else -1))
        if rows:
            b.last_id = max(b.last_id or 0, max(int(t["tradeId"]) for t in rows))
        b.add(out)

    def coinbase(self, h: Http, s: str, b: Buf) -> None:
        p = f"{s}-USD"
        rows = h.req(f"https://api.exchange.coinbase.com/products/{p}/trades?limit=1000")
        if isinstance(rows, dict):
            raise BadSymbol(rows.get("message"))
        if b.last_id is not None and rows:
            lo = min(t["trade_id"] for t in rows)
            pages = 0
            while lo > b.last_id + 1 and pages < 5:
                time.sleep(0.15)
                more = h.req(f"https://api.exchange.coinbase.com/products/{p}/trades?limit=1000&after={lo}")
                if not more or isinstance(more, dict):
                    break
                rows += more
                lo = min(t["trade_id"] for t in more)
                pages += 1
            if lo > b.last_id + 1:
                b.gaps += 1
        out, seen = [], set()
        for t in rows:
            tid = t["trade_id"]
            if (b.last_id is not None and tid <= b.last_id) or tid in seen:
                continue
            seen.add(tid)
            ts = int(datetime.fromisoformat(t["time"].replace("Z", "+00:00")).timestamp() * 1000)
            # Coinbase `side` = maker side: maker sold -> taker bought
            out.append((ts, float(t["price"]), float(t["size"]), 1 if t["side"] == "sell" else -1))
        if rows:
            b.last_id = max(b.last_id or 0, max(t["trade_id"] for t in rows))
        b.add(out)

    def hyperliquid(self, h: Http, s: str, b: Buf) -> None:
        rows = h.req("https://api.hyperliquid.xyz/info", {"type": "recentTrades", "coin": s})
        if not rows:
            raise BadSymbol("no trades / unknown coin")
        new = [t for t in rows if b.seen_add(t["tid"])]
        b.add([(int(t["time"]), float(t["px"]), float(t["sz"]), 1 if t["side"] == "B" else -1) for t in new])

    def deribit(self, h: Http, s: str, b: Buf) -> None:
        inst = f"{s}-PERPETUAL"
        q = {"instrument_name": inst, "count": 1000}
        if b.last_id is not None:
            q.update(start_seq=b.last_id + 1, sorting="asc")
        out, pages = [], 0
        while True:
            j = h.req("https://www.deribit.com/api/v2/public/get_last_trades_by_instrument?" + urlencode(q))
            r = j.get("result") or {}
            tr = r.get("trades") or []
            for t in tr:
                if b.last_id is None or t["trade_seq"] > b.last_id:
                    out.append((int(t["timestamp"]), float(t["price"]), float(t["amount"]) / float(t["price"]),
                                1 if t["direction"] == "buy" else -1))   # perp amount is USD
            if tr:
                b.last_id = max(b.last_id or 0, max(t["trade_seq"] for t in tr))
            pages += 1
            if not r.get("has_more") or b.last_id is None or pages >= 5 or "start_seq" not in q:
                if r.get("has_more") and pages >= 5:
                    b.gaps += 1
                break
            q["start_seq"] = b.last_id + 1
            time.sleep(0.1)
        b.add(out)

    def upbit(self, h: Http, s: str, b: Buf) -> None:
        if self.usdtkrw is None:
            raise OSError("no USDT-KRW quote yet")
        rows = h.req(f"https://api.upbit.com/v1/trades/ticks?market=KRW-{s}&count=500")
        if isinstance(rows, dict):
            raise BadSymbol(str(rows)[:120])
        new = [t for t in rows if b.last_id is None or t["sequential_id"] > b.last_id]
        if b.last_id is not None and len(new) == len(rows) and rows:
            b.gaps += 1
        if rows:
            b.last_id = max(b.last_id or 0, max(t["sequential_id"] for t in rows))
        k = self.usdtkrw
        b.add([(int(t["timestamp"]), float(t["trade_price"]) / k, float(t["trade_volume"]),
                1 if t["ask_bid"] == "BID" else -1) for t in new])

    def fetch(self, h: Http, v: str, s: str) -> None:
        b = self.buf[(v, s)]
        now = time.time()
        if not b.ok:
            # a feed that worked before (or any feed, slowly) is probed again: a 400 after a PC suspend or an outage
            # usually means a stale paging cursor, not a delisting (2026-11-26: Binance stayed dead until a restart)
            if b.err == "not listed" or now < b.retry_at:
                return
            b.ok = True
            b.reset_cursor()
        elif b.last_ok_ts and now - b.last_ok_ts > STALE_RESET_S:
            b.reset_cursor()                   # resume from the latest trades, not from a cursor hours old
        f = {"by": self.bybit, "bn": self.binance_perp, "bs": self.binance_spot, "ok": self.okx, "cb": self.coinbase,
             "hl": self.hyperliquid, "db": self.deribit, "up": self.upbit}[v]
        t0 = time.time()
        try:
            f(h, s, b)
            b.polls += 1
            b.err = ""
            b.last_ok_ts = time.time()
        except BadSymbol as e:
            b.ok = False
            b.err = f"rejected (retry in {RETRY_DISABLED_S // 60} min): {str(e)[:70]}"
            b.retry_at = time.time() + (RETRY_DISABLED_S if b.last_ok_ts else 4 * RETRY_DISABLED_S)
            b.reset_cursor()
        except RateLimited as e:
            b.err = f"429 {e}"
        except Exception as e:  # noqa: BLE001 - one venue failing must not stop the tracker
            b.err = f"{type(e).__name__}: {str(e)[:80]}"
        self.perf[v] = self.perf.get(v, 0.0) + time.time() - t0
        time.sleep(0.04)


# ---------------------------------------------------------------- quotes (mids, funding, OI) once a minute

def quotes(h: Http, which: str, syms: list[str], V: Venues) -> dict:
    """Returns {(venue, sym): {...}} for the group `which` ('a' or 'b')."""
    out: dict = {}

    def mid(b, a):
        b, a = float(b or 0), float(a or 0)
        return (a + b) / 2 if a > 0 and b > 0 else None

    def safe(fn):
        try:
            fn()
        except Exception as e:  # noqa: BLE001
            out.setdefault(("err", which), []).append(f"{fn.__name__}: {str(e)[:80]}")

    if which == "a":
        def bybit():
            lin = {t["symbol"]: t for t in h.req("https://api.bybit.com/v5/market/tickers?category=linear")["result"]["list"]}
            for s in syms:
                t = lin.get(f"{s}USDT")
                if t:
                    out[("by", s)] = {"mid": mid(t["bid1Price"], t["ask1Price"]), "mark": float(t["markPrice"]),
                                      "index": float(t["indexPrice"]), "funding": float(t["fundingRate"] or 0),
                                      "oi_usd": float(t["openInterestValue"] or 0)}

        def binance():
            bt = {x["symbol"]: x for x in h.req("https://fapi.binance.com/fapi/v1/ticker/bookTicker")}
            pi = {x["symbol"]: x for x in h.req("https://fapi.binance.com/fapi/v1/premiumIndex")}
            for s in syms:
                k = f"{s}USDT"
                if k in bt:
                    out[("bn", s)] = {"mid": mid(bt[k]["bidPrice"], bt[k]["askPrice"]),
                                      "mark": float(pi.get(k, {}).get("markPrice") or 0) or None,
                                      "index": float(pi.get(k, {}).get("indexPrice") or 0) or None,
                                      "funding": float(pi.get(k, {}).get("lastFundingRate") or 0)}
            ok = [f"{s}USDT" for s in syms if V.buf[("bs", s)].ok]
            if ok:
                for x in h.req("https://api.binance.com/api/v3/ticker/bookTicker?symbols=" + json.dumps(ok, separators=(",", ":"))):
                    out[("bs", x["symbol"][:-4])] = {"mid": mid(x["bidPrice"], x["askPrice"])}

        def hyper():
            meta, ctx = h.req("https://api.hyperliquid.xyz/info", {"type": "metaAndAssetCtxs"})
            names = [u["name"] for u in meta["universe"]]
            for s in syms:
                if s in names:
                    c = ctx[names.index(s)]
                    out[("hl", s)] = {"mid": float(c.get("midPx") or 0) or None, "mark": float(c["markPx"]),
                                      "index": float(c["oraclePx"]), "funding": float(c["funding"]),
                                      "oi_usd": float(c["openInterest"]) * float(c["markPx"])}

        for fn in (bybit, binance, hyper):
            safe(fn)
    else:
        def okx():
            sw = {x["instId"]: x for x in h.req("https://www.okx.com/api/v5/market/tickers?instType=SWAP")["data"]}
            sp = {x["instId"]: x for x in h.req("https://www.okx.com/api/v5/market/tickers?instType=SPOT")["data"]}
            oi = {x["instId"]: x for x in h.req("https://www.okx.com/api/v5/public/open-interest?instType=SWAP")["data"]}
            for s in syms:
                k = f"{s}-USDT-SWAP"
                if k in sw:
                    d = {"mid": mid(sw[k]["bidPx"], sw[k]["askPx"])}
                    if k in oi and oi[k].get("oiUsd"):
                        d["oi_usd"] = float(oi[k]["oiUsd"])
                    if f"{s}-USDT" in sp:            # OKX index-tickers lag (ts ~20 s): basis = swap mid vs spot mid
                        d["mark"], d["index"] = d["mid"], mid(sp[f"{s}-USDT"]["bidPx"], sp[f"{s}-USDT"]["askPx"])
                    out[("ok", s)] = d
                if f"{s}-USDT" in sp:
                    x = sp[f"{s}-USDT"]
                    out[("ok_spot", s)] = {"mid": mid(x["bidPx"], x["askPx"])}

        def coinbase():
            for p in [f"{s}-USD" for s in syms if V.buf[("cb", s)].ok] + ["USDT-USD"]:
                try:
                    t = h.req(f"https://api.exchange.coinbase.com/products/{p}/ticker")
                    out[("cb", p[:-4])] = {"mid": mid(t["bid"], t["ask"])}
                except BadSymbol:
                    pass
                time.sleep(0.11)

        def deribit():
            for s in ("BTC", "ETH"):
                if s in syms:
                    r = h.req(f"https://www.deribit.com/api/v2/public/ticker?instrument_name={s}-PERPETUAL")["result"]
                    out[("db", s)] = {"mid": mid(r["best_bid_price"], r["best_ask_price"]), "mark": r["mark_price"],
                                      "index": r["index_price"], "funding": r.get("funding_8h"),
                                      "oi_usd": r.get("open_interest")}

        def upbit():
            mk = ["KRW-USDT"] + [f"KRW-{s}" for s in syms if V.buf[("up", s)].ok]
            for x in h.req("https://api.upbit.com/v1/orderbook?markets=" + ",".join(mk)):
                u = x["orderbook_units"][0]
                out[("up_krw", x["market"][4:])] = {"mid": mid(u["bid_price"], u["ask_price"])}
            k = out.get(("up_krw", "USDT"), {}).get("mid")
            if k:
                V.usdtkrw = k
                for (v, s), d in list(out.items()):
                    if v == "up_krw" and s != "USDT" and d["mid"]:
                        out[("up", s)] = {"mid": d["mid"] / k}

        for fn in (okx, coinbase, deribit, upbit):
            safe(fn)
    return out


# ---------------------------------------------------------------- analytics

def pct(xs: list[float], q: float) -> float | None:
    if len(xs) < 50:
        return None
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(q * len(xs)))]


def corr(a: list[float], b: list[float]) -> float | None:
    n = len(a)
    if n < 20:
        return None
    ma, mb = sum(a) / n, sum(b) / n
    va = sum((x - ma) ** 2 for x in a)
    vb = sum((y - mb) ** 2 for y in b)
    if va <= 0 or vb <= 0:
        return None
    return sum((x - ma) * (y - mb) for x, y in zip(a, b, strict=True)) / math.sqrt(va * vb)


def bucket_returns(prints: deque, t_end: int, n: int = 90, step: int = 10_000) -> list[float] | None:
    """10-s log returns of the last price per bucket, n buckets ending at t_end (forward filled)."""
    t0 = t_end - n * step
    last: list = [None] * (n + 1)          # last[0] = last price before t0; last[i + 1] = last price in bucket i
    for p in prints:
        if p[0] >= t_end:
            break
        i = 0 if p[0] < t0 else (p[0] - t0) // step + 1
        last[min(i, n)] = p[1]
    filled = 0
    for i in range(n + 1):
        if last[i] is None:
            last[i] = last[i - 1] if i else None
        else:
            filled += 1
    if last[0] is None or filled < n * 0.5:
        return None
    return [math.log(last[i] / last[i - 1]) for i in range(1, n + 1)]


def lead_lag(x: list[float], y: list[float]) -> tuple[int, float] | None:
    """Best k in -3..3 for corr(x[t], y[t+k]); k > 0 means x leads y by k*10 s."""
    best = None
    for k in range(-3, 4):
        if k >= 0:
            a, b = x[: len(x) - k], y[k:]
        else:
            a, b = x[-k:], y[: len(y) + k]
        c = corr(a, b)
        if c is not None and (best is None or c > best[1]):
            best = (k, c)
    return best


def fmt_usd(x: float | None) -> str:
    if x is None:
        return "na"
    a = abs(x)
    s = "+" if x >= 0 else "-"
    if a >= 1e6:
        return f"{s}{a / 1e6:.1f}M"
    if a >= 1e3:
        return f"{s}{a / 1e3:.0f}k"
    return f"{s}{a:.0f}"


def lr_bp(a, b) -> float | None:
    return round(1e4 * math.log(a / b), 2) if a and b and a > 0 and b > 0 else None


class Tracker:
    def __init__(self, syms: list[str]) -> None:
        self.syms = syms
        self.V = Venues(syms)
        self.h = [Http(), Http()]
        self.large_hist: dict = defaultdict(lambda: deque(maxlen=60))   # (v,s) -> |large delta 5m| per minute
        self.prem_hist: dict = defaultdict(lambda: deque(maxlen=60))    # series key -> values per minute
        self.lead_run: dict = defaultdict(int)                          # (v,s) -> consecutive minutes leading
        self.cooldown: dict = {}
        self.t_start = time.time()

    def write(self, line: str) -> None:
        with LOG.open("a", encoding="utf-8") as f:
            f.write(line + "\n")

    def alert(self, key: str, msg: str, now_iso: str, cool_s: int = 600) -> dict | None:
        t = time.time()
        if self.cooldown.get(key, 0) > t:
            return None
        self.cooldown[key] = t + cool_s
        self.write(f"{now_iso} ALERT {msg}")
        return {"key": key, "msg": msg}

    def poll(self) -> None:
        """Shared job list consumed by the main thread and one worker (2 threads in total)."""
        self.n_poll = getattr(self, "n_poll", 0) + 1
        # Hyperliquid returns only ~10 trades per call (sampled anyway): poll it every 3rd cycle to keep the others fast
        jobs = [(v, s) for s in self.syms for v in VENUES if self.V.buf[(v, s)].ok and (v != "hl" or self.n_poll % 3 == 1)]
        jobs.sort(key=lambda j: j[0] != "by")          # slowest venue first
        lock = threading.Lock()

        def run(h: Http) -> None:
            while True:
                with lock:
                    if not jobs:
                        return
                    v, s = jobs.pop(0)
                self.V.fetch(h, v, s)

        th = threading.Thread(target=run, args=(self.h[1],))
        th.start()
        run(self.h[0])
        th.join()

    def report(self) -> None:
        res: dict = {}
        th = threading.Thread(target=lambda: res.update(quotes(self.h[1], "b", self.syms, self.V)))
        th.start()
        res.update(quotes(self.h[0], "a", self.syms, self.V))
        th.join()
        now = int(time.time() * 1000)
        now_iso = datetime.fromtimestamp(now / 1000, UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
        warm = (time.time() - self.t_start) / 60
        recs = []
        for s in self.syms:
            rec: dict = {"ts": now_iso, "sym": s, "warmup_min": round(warm, 1), "venues": {}, "alerts": []}
            by_mid = (res.get(("by", s)) or {}).get("mid")
            rec["bybit"] = res.get(("by", s))
            rets: dict = {}
            for v in VENUES:
                b = self.V.buf[(v, s)]
                if not b.ok:
                    continue
                pr = list(b.prints)
                notion = [p[3] for p in pr]
                p95, p99 = pct(notion, 0.95), pct(notion, 0.99)
                d: dict = {"n_prints_60m": len(pr), "p95_usd": round(p95, 0) if p95 else None,
                           "p99_usd": round(p99, 0) if p99 else None, "gaps": b.gaps, "polls": b.polls,
                           "err": b.err or None, "coverage": "sampled" if v in SAMPLED else ("gap" if b.gaps else "full")}
                for w, ms in WINDOWS.items():
                    ws = [p for p in pr if p[0] >= now - ms]
                    tot = sum(p[2] for p in ws)
                    vol = sum(p[3] for p in ws)
                    large = sum(p[2] for p in ws if p95 and p[3] > p95)
                    whale = sum(p[2] for p in ws if p99 and p[3] > p99)
                    vw = sum(p[1] * p[3] for p in ws) / vol if vol > 0 else None
                    d[w] = {"delta": round(tot), "small": round(tot - large) if p95 else None,
                            "large": round(large) if p95 else None, "whale": round(whale) if p99 else None,
                            "vol": round(vol), "n": len(ws), "vwap": vw}
                q = res.get((v, s)) or {}
                d["mid"] = q.get("mid")
                d["prem_mid_bp"] = lr_bp(q.get("mid"), by_mid) if v != "by" else 0.0   # quotes not simultaneous
                if v in PERP:
                    d["basis_bp"] = lr_bp(q.get("mark"), q.get("index"))      # mark vs the venue's spot index / oracle
                for k in ("funding", "oi_usd"):
                    if q.get(k) is not None:
                        d[k] = q[k]
                rr = bucket_returns(b.prints, now - 20_000)
                if rr:
                    rets[v] = rr
                rec["venues"][v] = d
            # premiums from 1-min VWAPs over the same window (polls are not simultaneous; VWAPs are)
            vw = {v: d["1m"]["vwap"] for v, d in rec["venues"].items() if d["1m"]["vwap"]}
            for v, d in rec["venues"].items():
                d["prem_bp"] = lr_bp(vw.get(v), vw.get("by")) if v != "by" else 0.0
            cb, bs = vw.get("cb"), vw.get("bs")
            usdt = (res.get(("cb", "USDT")) or {}).get("mid")
            up = vw.get("up")
            rec["coinbase_prem_bp"] = lr_bp(cb, bs)
            rec["coinbase_prem_usdt_adj_bp"] = lr_bp(cb, bs * usdt) if bs and usdt else None
            rec["korea_prem_bp"] = lr_bp(up, bs) if bs else lr_bp(up, vw.get("by"))
            rec["usdt_usd"] = usdt
            rec["usdt_krw"] = self.V.usdtkrw
            # lead-lag vs Bybit (10-s returns, last 15 min)
            ll = {}
            if "by" in rets:
                for v, r in rets.items():
                    if v == "by":
                        continue
                    x = lead_lag(r, rets["by"])
                    if x:
                        ll[v] = {"lag": x[0], "corr": round(x[1], 3)}
            rec["lead_vs_bybit"] = ll
            score = {}
            for v in rets:
                ks = []
                for u in rets:
                    if u != v:
                        x = lead_lag(rets[v], rets[u])
                        if x and x[1] > 2 / math.sqrt(87):
                            ks.append(x[0])
                if ks:
                    score[v] = sum(ks) / len(ks)
            rec["leader"] = max(score, key=score.get) if score and max(score.values()) > 0 else None
            rec["leader_score"] = {k: round(x, 2) for k, x in score.items()}
            self.alerts(s, rec, now_iso)
            recs.append(rec)
            self.write(self.line(rec))
        JDIR.mkdir(parents=True, exist_ok=True)
        with (JDIR / f"{now_iso[:10]}.jsonl").open("a", encoding="utf-8") as f:
            for r in recs:
                f.write(json.dumps(r, separators=(",", ":")) + "\n")
        errs = [e for k, e in res.items() if isinstance(k, tuple) and k[0] == "err"]
        if errs:
            self.write(f"{now_iso} quote-errors {errs}")

    def alerts(self, s: str, rec: dict, now_iso: str) -> None:
        vs = rec["venues"]
        # 1) large-taker divergence over 5 min
        large = {v: d["5m"]["large"] for v, d in vs.items() if d["5m"]["large"] is not None and v not in SAMPLED}
        for v, x in large.items():
            hist = self.large_hist[(v, s)]
            med = statistics.median(hist) if len(hist) >= 10 else None
            others = {u: y for u, y in large.items() if u != v and y}
            osum = sum(others.values())
            same = [u for u, y in others.items() if osum and (y > 0) == (osum > 0)]
            if med and med > 0 and abs(x) > 3 * med and osum and (x > 0) != (osum > 0) and len(same) >= 2:
                a = self.alert(f"div:{s}:{v}", f"{s} DIVERGENCE {v} large-taker 5m {fmt_usd(x)} ({abs(x) / med:.1f}x its "
                               f"1h median |{fmt_usd(med)}|) vs others {fmt_usd(osum)} [{','.join(same)} agree] "
                               "- descriptive: who is the informed side is untested", now_iso)
                if a:
                    rec["alerts"].append(a)
            hist.append(abs(x))
        # 2) premium / basis z vs its 1h history
        series = {f"prem:{v}": d.get("prem_bp") for v, d in vs.items() if v != "by"}
        series.update({f"basis:{v}": d.get("basis_bp") for v, d in vs.items()})
        series.update({"cb_prem": rec["coinbase_prem_bp"], "kr_prem": rec["korea_prem_bp"]})
        for k, x in series.items():
            if x is None:
                continue
            hist = self.prem_hist[(s, k)]
            if len(hist) >= 20:
                m = statistics.fmean(hist)
                sd = statistics.pstdev(hist)
                if sd > 0 and abs(x - m) > 3 * sd and abs(x - m) > 1.0:   # also >= 1 bp so tick noise does not fire
                    a = self.alert(f"prem:{s}:{k}", f"{s} PREMIUM {k} {x:+.1f} bp vs 1h mean {m:+.1f} sd {sd:.1f} "
                                   f"(z {(x - m) / sd:+.1f})", now_iso)
                    if a:
                        rec["alerts"].append(a)
            hist.append(x)
        # 3) consistent lead vs Bybit
        thr = 2 / math.sqrt(87)
        for v in VENUES:
            if v == "by":
                continue
            x = rec["lead_vs_bybit"].get(v)
            if x and x["lag"] >= 1 and x["corr"] > thr:
                self.lead_run[(v, s)] += 1
            else:
                self.lead_run[(v, s)] = 0
            if self.lead_run[(v, s)] >= 5:
                a = self.alert(f"lead:{s}:{v}", f"{s} LEAD {v} leads Bybit by {x['lag'] * 10}s (corr {x['corr']:.2f}) "
                               f"for {self.lead_run[(v, s)]} consecutive minutes", now_iso, cool_s=900)
                if a:
                    rec["alerts"].append(a)

    @staticmethod
    def line(rec: dict) -> str:
        vs = rec["venues"]
        by = rec.get("bybit") or {}
        px = by.get("mid")
        parts = [f"{rec['ts']} {rec['sym']:<5} by_mid {px:.6g}" if px else f"{rec['ts']} {rec['sym']:<5} by_mid na"]
        parts.append("d5m " + " ".join(f"{v}:{fmt_usd(d['5m']['delta'])}" for v, d in vs.items()))
        parts.append("L5m " + " ".join(f"{v}:{fmt_usd(d['5m']['large'])}" for v, d in vs.items() if d["5m"]["large"] is not None))
        parts.append("W15m " + " ".join(f"{v}:{fmt_usd(d['15m']['whale'])}" for v, d in vs.items() if d["15m"]["whale"]))
        parts.append("prem " + " ".join(f"{v}:{d['prem_bp']:+.1f}" for v, d in vs.items() if v != "by" and d.get("prem_bp") is not None))
        extra = []
        if rec["coinbase_prem_bp"] is not None:
            extra.append(f"cbP {rec['coinbase_prem_bp']:+.1f}")
        if rec["korea_prem_bp"] is not None:
            extra.append(f"krP {rec['korea_prem_bp']:+.1f}")
        if extra:
            parts.append(" ".join(extra))
        parts.append("basis " + " ".join(f"{v}:{d['basis_bp']:+.1f}" for v, d in vs.items() if d.get("basis_bp") is not None))
        if by.get("funding") is not None:
            parts.append(f"fund_by {by['funding'] * 1e4:+.2f}bp oi_by {fmt_usd(by.get('oi_usd'))}")
        ll = rec["lead_vs_bybit"]
        lead = [f"{v}{x['lag']:+d}({x['corr']:.2f})" for v, x in ll.items() if x["lag"] != 0]
        parts.append(f"lead {rec['leader'] or '-'} vsBy[{' '.join(lead) or 'all lag0'}]")
        bad = [f"{v}:{d['coverage']}" for v, d in vs.items() if d["coverage"] != "full"]
        if bad:
            parts.append("cov " + " ".join(bad))
        return " | ".join(parts)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", nargs="*", default=[])
    ap.add_argument("--minutes", type=float, default=0, help="stop after N minutes (0 = run until killed)")
    a = ap.parse_args()
    syms = list(dict.fromkeys(DEFAULT + [s.upper().removesuffix("USDT") for s in a.symbols]))
    if "SPY" in syms:
        syms.remove("SPY")
    LOG.parent.mkdir(parents=True, exist_ok=True)
    PIDF.write_text(str(os.getpid()))
    T = Tracker(syms)
    T.write(f"# {datetime.now(UTC):%Y-%m-%dT%H:%M:%SZ} xvenue_tape start pid {os.getpid()} symbols {','.join(syms)}. "
            "LABELS ARE DESCRIPTIVE, NOT TESTED SIGNALS. venues by=Bybit perp bn=Binance perp bs=Binance spot "
            "ok=OKX swap cb=Coinbase spot hl=Hyperliquid(sampled) db=Deribit perp up=Upbit KRW. d5m = taker delta USD 5m; "
            "L5m = large (>p95 venue print) delta; W15m = whale (>p99) delta 15m; prem = 1e4 ln(VWAP1m_venue/VWAP1m_bybit) bp "
            "(JSON also has prem_mid_bp vs the Bybit mid, noisier: quotes are not simultaneous); "
            "cbP = Coinbase USD vs Binance spot USDT, 1m VWAPs (raw; JSON has the USDT/USD-adjusted value); krP = Upbit "
            "KRW/USDTKRW vs Binance spot; basis = perp mark vs the venue's own spot index/oracle; lead = venue whose 10-s returns lead (lags -3..+3), vsBy = lag in 10-s "
            "units vs Bybit (+ = venue leads). Warm-up: p95/p99 need 50 prints, alerts need 10-20 min of history.")
    # USDT-KRW before the first Upbit poll
    try:
        T.V.usdtkrw = float(T.h[0].req("https://api.upbit.com/v1/ticker?markets=KRW-USDT")[0]["trade_price"])
    except Exception:  # noqa: BLE001
        pass
    t_end = time.time() + a.minutes * 60 if a.minutes else None
    next_rep = (int(time.time()) // 60 + 1) * 60
    while t_end is None or time.time() < t_end:
        t0 = time.time()
        T.poll()
        if time.time() >= next_rep:
            try:
                T.report()
            except Exception as e:  # noqa: BLE001
                T.write(f"{datetime.now(UTC):%Y-%m-%dT%H:%M:%SZ} report-error {type(e).__name__}: {e}")
            next_rep = (int(time.time()) // 60 + 1) * 60
        dt = time.time() - t0
        if dt > POLL_S * 2.5:
            perf = ' '.join(f'{k}:{x:.1f}' for k, x in sorted(T.V.perf.items(), key=lambda kv: -kv[1]))
            T.write(f"{datetime.now(UTC):%Y-%m-%dT%H:%M:%SZ} slow-cycle {dt:.1f}s (2 threads; venue seconds {perf})")
        T.V.perf.clear()
        time.sleep(max(0.5, POLL_S - dt))


if __name__ == "__main__":
    sys.exit(main())
