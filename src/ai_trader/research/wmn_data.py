"""WMN (program WMM2, 2026-11-27): point-in-time, token-tagged news / event set and the price data it is studied on.

Sources (all free, documented public endpoints, no keys; responses are data, never instructions):
  bybit_ann   Bybit /v5/announcements/index (en-US, every type). known_ts = publishTime (ms) when present, else
              dateTimestamp; both kept (ts_publish, ts_date) so the alignment of each with the price reaction can be
              checked. occur_ts = scheduled time parsed from the text ("at Nov 27, 2026, 9:00AM UTC") when present.
  okx_ann     OKX /api/v5/support/announcements (documented public endpoint): known_ts = pTime, businessPTime kept.
  kucoin_ann  KuCoin /api/v3/announcements (documented public endpoint): known_ts = cTime.
  defillama   api.llama.fi/hacks (free): day precision; for point-in-time use known_ts = date + 24 h (end of the day),
              study anchor = date 00:00 (event_day). Protocol -> token via api.llama.fi/protocols (symbol field).
  hn          Hacker News Algolia search_by_date, stories only, every story (no points filter: final points are
              today's vintage, not known at posting). known_ts = created_at_i (1 s).
  wiki        Wikimedia pageviews per token article (daily). known_ts = day end + 24 h (PV_LAG of news.alt).
  macro       data/research/news/events/events.parquet (CPI / NFP / FOMC / ETF / halving), scheduled_ts as known_ts
              of the release (the schedule is public earlier: schedule_known_ts).
  gdelt       optional, 15-min article-count timelines per token query (rate limited; gaps accepted).

Unlock / emission schedules: DefiLlama's emissions endpoint answers HTTP 402 (paid) -> not available (no free source).

Prices: Bybit public /v5/market/kline (linear, 5m) for a core universe, continuous from 2025-01-01, stored per symbol as
data/research/wmn/bars5m/<SYM>.parquet (ts = bar start ms, o h l c float32, turnover float32). Delisted perps have no
klines; their trades come from public.bybit.com/trading/<SYM>/ (kept for days around the event only, raw deleted).

Layout: data/research/wmn/ (quota 500 MB), raw downloads under data/raw/WMN/ (deleted after use).
"""

from __future__ import annotations

import gzip
import hashlib
import io
import json
import re
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl

from ai_trader.core.config import find_root

UA = "AI_Trader research (non-commercial; WMN news study)"
ROOT = find_root()
OUT = ROOT / "data" / "research" / "wmn"
RAW = ROOT / "data" / "raw" / "WMN"
BARS = OUT / "bars5m"
WIN = OUT / "bars5m_win"
CUTOFF = datetime(2026, 7, 1, tzinfo=UTC)
START = datetime(2025, 1, 1, tzinfo=UTC)

# Core universe (continuous 5m): market factor majors + the MM illiquid names (config/mm_illiq20.yaml crypto names,
# research/mm_weekend/universe.json) + crypto alts already in the lake. TradFi perps excluded.
MAJORS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "DOGEUSDT", "BNBUSDT"]
MM_NAMES = ["CARVUSDT", "MAVIAUSDT", "KASUSDT", "ETCUSDT", "ATOMUSDT", "WIFUSDT", "FILUSDT", "RENDERUSDT", "CRVUSDT",
            "MONUSDT", "ASTERUSDT", "TRXUSDT", "WLFIUSDT", "DOTUSDT", "XLMUSDT", "TAOUSDT", "GRAMUSDT", "TRUMPUSDT",
            "ICPUSDT", "WLDUSDT", "PUMPFUNUSDT"]
LAKE_ALTS = ["ADAUSDT", "AVAXUSDT", "LINKUSDT", "LTCUSDT", "BCHUSDT", "NEARUSDT", "SUIUSDT", "UNIUSDT", "AAVEUSDT",
             "ENAUSDT", "HYPEUSDT", "APTUSDT", "OPUSDT", "TIAUSDT", "SEIUSDT", "JUPUSDT", "LDOUSDT", "PENDLEUSDT",
             "PYTHUSDT", "RUNEUSDT", "STRKUSDT", "GALAUSDT", "HBARUSDT", "ALGOUSDT", "ETHFIUSDT", "VIRTUALUSDT",
             "PLUMEUSDT", "MNTUSDT", "1000BONKUSDT", "SHIB1000USDT", "1000PEPEUSDT"]
CORE = MAJORS + MM_NAMES + LAKE_ALTS

_last: dict[str, float] = {}


def get(url: str, params: dict | None = None, *, host_gap: float = 0.25, attempts: int = 6, timeout: int = 60,
        raw: bool = False):
    """Polite GET with per-host spacing and backoff. Returns parsed JSON (or bytes if raw) or None."""
    if params:
        url = f"{url}?{urllib.parse.urlencode(params)}"
    host = urllib.parse.urlparse(url).netloc
    for a in range(attempts):
        w = host_gap - (time.time() - _last.get(host, 0.0))
        if w > 0:
            time.sleep(w)
        _last[host] = time.time()
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept-Encoding": "gzip"})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                b = r.read()
                if r.headers.get("Content-Encoding") == "gzip":
                    b = gzip.decompress(b)
            if raw:
                return b
            return json.loads(b.decode("utf-8", "replace"))
        except Exception as e:  # noqa: BLE001
            msg = str(e)
            if "404" in msg:
                return None
            time.sleep(min(120, 3 * 2 ** a) if ("429" in msg or "403" in msg) else 2 * (a + 1))
    return None


def ms(dt: datetime) -> int:
    return int(dt.timestamp() * 1000)


# --------------------------------------------------------------------------------------------- instruments / tags
def fetch_instruments() -> pl.DataFrame:
    """All Bybit linear USDT perps (Trading + Closed + PreLaunch); base coin, status, launch and delivery times."""
    rows = []
    for status in ("Trading", "Closed", "PreLaunch"):
        cursor = ""
        while True:
            p = {"category": "linear", "limit": 1000, "status": status}
            if cursor:
                p["cursor"] = cursor
            d = get("https://api.bybit.com/v5/market/instruments-info", p)
            if not d or d.get("retCode") != 0:
                break
            for x in d["result"]["list"]:
                if x.get("quoteCoin") != "USDT" or x.get("contractType") != "LinearPerpetual":
                    continue
                rows.append({"symbol": x["symbol"], "base": x["baseCoin"], "status": x["status"],
                             "launch_ms": int(x.get("launchTime") or 0), "delivery_ms": int(x.get("deliveryTime") or 0),
                             "symbol_type": x.get("symbolType") or ""})
            cursor = d["result"].get("nextPageCursor") or ""
            if not cursor:
                break
    return pl.DataFrame(rows).unique("symbol", keep="first").sort("symbol")


def norm_base(base: str) -> str:
    """1000PEPE -> PEPE, SHIB1000 -> SHIB, 10000000AIDOGE -> AIDOGE."""
    b = re.sub(r"^1(0)+", "", base)
    b = re.sub(r"1(0)+$", "", b)
    return b or base


# Tagging uses explicit pairs (XXXUSDT, XXX/USDT) and parenthesised tickers only: bare capitalised words are too
# ambiguous (ONE, NEW, TRUE, MOVE ...).
PAIR = re.compile(r"\b([A-Z0-9]{2,20})(?:/| )?USDT\b")
PAREN = re.compile(r"\(([A-Z0-9]{2,15})\)")


CRYPTO_TYPES = ("", "innovation")


def build_tag_map(inst: pl.DataFrame) -> dict[str, list[str]]:
    """Base coin -> crypto perp symbols (TradFi perps: stock / ETF / commodity / forex are never tagged)."""
    m: dict[str, list[str]] = {}
    for sym, base in inst.filter(pl.col("symbol_type").is_in(CRYPTO_TYPES)).select("symbol", "base").iter_rows():
        for k in {base, norm_base(base)}:
            m.setdefault(k, [])
            if sym not in m[k]:
                m[k].append(sym)
    return m


CAPS = re.compile(r"(?<![A-Za-z0-9])([A-Z][A-Z0-9]{1,11})(?![A-Za-z0-9])")
CAPS_STOP = {"USDT", "USDC", "USD", "OKX", "EEA", "UTC", "API", "VIP", "P2P", "ETF", "NFT", "AI", "KYC", "OTC", "TV", "UK", "US",
             "EU", "DEX", "CEX", "APR", "APY", "NEW", "ALL", "ONE", "THE", "AND", "FOR", "TOP", "BIG", "HOT", "WIN", "FAQ",
             "IEO", "IDO", "ICO", "AMA", "BTC", "TRUE", "MOVE", "UP", "ME", "GO", "ID", "IT", "MY", "PRO", "MAX", "MIN",
             "SPOT", "PERP", "LIVE", "BOT", "BOTS", "WEB3", "USDE", "EUR", "EURC", "GBP", "JPY", "TRY", "BRL", "SOON"}


def tag_text(text: str, tag_map: dict[str, list[str]], symbols: set[str], caps: bool = False) -> list[str]:
    """Bybit perp symbols named in a title / description: explicit pairs (XXXUSDT, XXX/USDT), (XXX) tickers, and with
    caps=True (exchange listing / delisting / maintenance titles) any all-caps ticker that is a perp base coin and not a
    common acronym (also catches 'RSR/WNXM/BAL' lists and 'USDT-Margined XYZ')."""
    out: list[str] = []
    if not text:
        return out
    for m in PAIR.finditer(text):
        tok = m.group(1)
        cand = tok + "USDT"
        if cand in symbols:
            out.append(cand)
        elif tok in tag_map:
            out.extend(tag_map[tok])
    for m in PAREN.finditer(text):
        tok = m.group(1)
        if tok in tag_map and not tok.isdigit():
            out.extend(tag_map[tok])
    if caps:
        for m in CAPS.finditer(text):
            tok = m.group(1)
            if tok in CAPS_STOP or tok.isdigit():
                continue
            if tok.endswith("USDT") and tok[:-4] in tag_map:
                tok = tok[:-4]
            if tok in tag_map:
                out.extend(tag_map[tok])
    seen, res = set(), []
    for s in out:
        if s not in seen:
            seen.add(s)
            res.append(s)
    return res


# ------------------------------------------------------------------------------------------------- announcements
MON = {m: i for i, m in enumerate(["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"], 1)}
SCHED = re.compile(r"(?:on|at|from)\s+([A-Z][a-z]{2,8})\s+(\d{1,2}),\s*(\d{4}),?\s*(?:at\s*)?(\d{1,2}):(\d{2})\s*(AM|PM)?\s*UTC",
                   re.I)


def parse_sched(text: str) -> int | None:
    """First 'Mon DD, YYYY, H:MM AM UTC' time in a text (ms), or None."""
    if not text:
        return None
    m = SCHED.search(text)
    if not m:
        return None
    mon = MON.get(m.group(1)[:3].lower())
    if mon is None:
        return None
    hh, mm = int(m.group(4)), int(m.group(5))
    ap = (m.group(6) or "").upper()
    if ap == "PM" and hh < 12:
        hh += 12
    if ap == "AM" and hh == 12:
        hh = 0
    try:
        return ms(datetime(int(m.group(3)), mon, int(m.group(2)), hh, mm, tzinfo=UTC))
    except ValueError:
        return None


def fetch_bybit_ann(log=print) -> list[dict]:
    out, page = [], 1
    while True:
        d = get("https://api.bybit.com/v5/announcements/index", {"locale": "en-US", "limit": 100, "page": page})
        if not d or d.get("retCode") != 0:
            log(f"  bybit ann page {page}: failed")
            break
        lst = d["result"]["list"]
        if not lst:
            break
        out.extend(lst)
        page += 1
    log(f"  bybit announcements: {len(out)} items, {page - 1} pages")
    return out


OKX_TYPES = ["announcements-new-listings", "announcements-delistings", "announcements-trading-updates",
             "announcements-deposit-withdrawal-suspension-resumption"]


def fetch_okx_ann(log=print) -> list[dict]:
    out = []
    for t in OKX_TYPES:
        page = 1
        while True:
            d = get("https://www.okx.com/api/v5/support/announcements", {"annType": t, "page": page}, host_gap=0.5)
            if not d or d.get("code") != "0" or not d.get("data"):
                break
            det = d["data"][0].get("details") or []
            if not det:
                break
            out.extend(det)
            tp = int(d["data"][0].get("totalPage") or 0)
            if page >= tp:
                break
            page += 1
        log(f"  okx {t}: {page} pages, total {len(out)}")
    return out


KUCOIN_TYPES = ["new-listings", "delistings", "futures-announcements", "maintenance-updates"]


def fetch_kucoin_ann(start: datetime = START, log=print) -> list[dict]:
    out = []
    stop = datetime.now(UTC)
    for t in KUCOIN_TYPES:
        s = start
        while s < stop:
            e = min(s + timedelta(days=31), stop)
            page = 1
            while True:
                d = get("https://api.kucoin.com/api/v3/announcements",
                        {"annType": t, "pageSize": 50, "currentPage": page, "lang": "en_US", "startTime": ms(s), "endTime": ms(e)},
                        host_gap=0.4)
                if not d or d.get("code") != "200000":
                    break
                items = d["data"].get("items") or []
                for x in items:
                    x["_query_type"] = t
                out.extend(items)
                if page >= int(d["data"].get("totalPage") or 0):
                    break
                page += 1
            s = e
        log(f"  kucoin {t}: total {len(out)}")
    return out


# --------------------------------------------------------------------------------------------------- hacks / HN
def fetch_hacks(log=print) -> tuple[list[dict], dict[int, str]]:
    hacks = get("https://api.llama.fi/hacks", timeout=120) or []
    protos = get("https://api.llama.fi/protocols", timeout=180) or []
    pmap = {}
    for p in protos:
        sym = (p.get("symbol") or "").strip()
        if sym and sym != "-":
            try:
                pmap[int(p["id"])] = sym.upper()
            except (KeyError, ValueError, TypeError):
                pass
    log(f"  defillama hacks {len(hacks)}, protocols with a symbol {len(pmap)}")
    return hacks, pmap


# Unambiguous token names for HN / Wikipedia (term -> perp symbol). Market terms map to "market".
NAME_TERMS = {
    "bitcoin": "market", "cryptocurrency": "market", "crypto": "market", "stablecoin": "market",
    "ethereum": "ETHUSDT", "solana": "SOLUSDT", "dogecoin": "DOGEUSDT", "cardano": "ADAUSDT", "polkadot": "DOTUSDT",
    "chainlink": "LINKUSDT", "litecoin": "LTCUSDT", "filecoin": "FILUSDT", "kaspa": "KASUSDT", "bittensor": "TAOUSDT",
    "worldcoin": "WLDUSDT", "hyperliquid": "HYPEUSDT", "uniswap": "UNIUSDT", "xrp": "XRPUSDT",
    "ethereum classic": "ETCUSDT", "dfinity": "ICPUSDT", "celestia": "TIAUSDT", "aptos": "APTUSDT", "binance": "BNBUSDT",
    "starknet": "STRKUSDT", "hedera": "HBARUSDT", "algorand": "ALGOUSDT", "thorchain": "RUNEUSDT",
    "shiba inu": "SHIB1000USDT", "pump.fun": "PUMPFUNUSDT", "trump memecoin": "TRUMPUSDT",
    "world liberty financial": "WLFIUSDT", "lido": "LDOUSDT", "aave": "AAVEUSDT", "ethena": "ENAUSDT",
}
WIKI_ARTICLES = {
    "Ethereum": "ETHUSDT", "Solana_(blockchain_platform)": "SOLUSDT", "Dogecoin": "DOGEUSDT", "Cardano_(blockchain_platform)": "ADAUSDT",
    "Polkadot_(cryptocurrency)": "DOTUSDT", "Chainlink_(blockchain)": "LINKUSDT", "Litecoin": "LTCUSDT", "Filecoin": "FILUSDT",
    "Ripple_Labs": "XRPUSDT", "Ethereum_Classic": "ETCUSDT", "Tron_(cryptocurrency)": "TRXUSDT", "Stellar_(payment_network)": "XLMUSDT",
    "Avalanche_(blockchain_platform)": "AVAXUSDT", "Binance": "BNBUSDT", "Shiba_Inu_(cryptocurrency)": "SHIB1000USDT",
    "Worldcoin": "WLDUSDT", "Internet_Computer": "ICPUSDT", "Hyperliquid": "HYPEUSDT", "Uniswap": "UNIUSDT",
    "Pump.fun": "PUMPFUNUSDT", "World_Liberty_Financial": "WLFIUSDT", "Bitcoin": "market", "Cryptocurrency": "market",
    "Pepe_(cryptocurrency)": "1000PEPEUSDT", "Cosmos_(blockchain)": "ATOMUSDT", "Bitcoin_Cash": "BCHUSDT",
}


def fetch_hn_term(term: str, start: datetime = START, log=print) -> list[dict]:
    out = []
    a = int(start.timestamp())
    now = int(datetime.now(UTC).timestamp())
    step = 30 * 86400
    while a < now:
        b = min(a + step, now)
        d = get("https://hn.algolia.com/api/v1/search_by_date",
                {"query": term, "tags": "story", "hitsPerPage": 1000, "typoTolerance": "false", "queryType": "prefixNone",
                 "numericFilters": f"created_at_i>={a},created_at_i<{b}"}, host_gap=0.4)
        hits = (d or {}).get("hits") or []
        if len(hits) >= 1000 and step > 86400:
            step //= 2
            continue
        pat = re.compile(r"(?<![a-z0-9])" + r"\s+".join(re.escape(w) for w in term.lower().split()) + r"(?![a-z0-9])")
        for h in hits:
            title = h.get("title") or ""
            if not pat.search(title.lower()):  # the word(s) must be in the title (no typo / prefix / body matches)
                continue
            out.append({"term": term, "created_at_i": int(h["created_at_i"]), "title": title,
                        "objectID": h.get("objectID"), "url": h.get("url") or ""})
        a = b
        step = 30 * 86400
    return out


def fetch_wiki(article: str, start: datetime = datetime(2024, 1, 1, tzinfo=UTC)) -> list[tuple[str, int]]:
    url = ("https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/en.wikipedia/all-access/user/"
           f"{urllib.parse.quote(article, safe='')}/daily/{start:%Y%m%d}00/{datetime.now(UTC):%Y%m%d}00")
    d = get(url, host_gap=0.3)
    return [(x["timestamp"][:8], int(x["views"])) for x in (d or {}).get("items", [])]


# ------------------------------------------------------------------------------------------------------- klines
KLINE_FAILS: list[tuple[str, int]] = []


def fetch_klines(symbol: str, start: datetime, end: datetime, interval: str = "5") -> pl.DataFrame:
    """Bybit public linear klines [start, end) paged backwards (1000 per call)."""
    step_ms = int(interval) * 60_000
    rows: dict[int, list] = {}
    e = ms(end) - 1
    s0 = ms(start)
    while e >= s0:
        d = None
        for a in range(6):
            d = get("https://api.bybit.com/v5/market/kline",
                    {"category": "linear", "symbol": symbol, "interval": interval, "limit": 1000, "end": e}, host_gap=0.15)
            if d and d.get("retCode") == 0:
                break
            time.sleep(5 * 2 ** a)
        if not d or d.get("retCode") != 0:
            KLINE_FAILS.append((symbol, e))
            break
        lst = d["result"]["list"]
        if not lst:
            break
        for x in lst:
            t = int(x[0])
            if s0 <= t < ms(end):
                rows[t] = x
        first = min(int(x[0]) for x in lst)
        if first <= s0 or len(lst) < 1000:
            break
        e = first - 1
    if not rows:
        return pl.DataFrame(schema={"ts": pl.Int64, "o": pl.Float32, "h": pl.Float32, "l": pl.Float32, "c": pl.Float32,
                                    "turnover": pl.Float32})
    ts = np.array(sorted(rows), dtype=np.int64)
    arr = np.array([[float(rows[t][i]) for i in (1, 2, 3, 4, 6)] for t in ts], dtype=np.float64)
    assert np.all(np.diff(ts) % step_ms == 0)
    return pl.DataFrame({"ts": ts, "o": arr[:, 0].astype(np.float32), "h": arr[:, 1].astype(np.float32),
                         "l": arr[:, 2].astype(np.float32), "c": arr[:, 3].astype(np.float32),
                         "turnover": arr[:, 4].astype(np.float32)})


def update_core_bars(symbols=CORE, start: datetime = START, end: datetime | None = None, workers: int = 4, log=print) -> dict:
    """Continuous 5m bars per symbol (incremental: only bars after the stored last bar are fetched)."""
    end = end or datetime.now(UTC).replace(second=0, microsecond=0)
    BARS.mkdir(parents=True, exist_ok=True)

    def one(sym: str):
        p = BARS / f"{sym}.parquet"
        old = pl.read_parquet(p) if p.is_file() else None
        s = start if old is None or old.height == 0 else datetime.fromtimestamp((int(old["ts"].max()) + 300_000) / 1000, UTC)
        new = fetch_klines(sym, s, end)
        df = new if old is None else pl.concat([old, new]).unique("ts", keep="first").sort("ts")
        if df.height:
            df.write_parquet(p, compression="zstd", compression_level=9)
        return sym, df.height, new.height

    res = {}
    with ThreadPoolExecutor(workers) as ex:
        for sym, n, k in ex.map(one, symbols):
            res[sym] = {"rows": n, "new": k}
            log(f"  {sym}: {n} rows (+{k})")
    return res


def backfill_core_bars(symbols=CORE, start: datetime = START, workers: int = 3, log=print) -> dict:
    """Fetch the history before each stored first bar back to max(start, launch) when it is missing (earlier runs
    stopped on rate-limit replies)."""
    inst = pl.read_parquet(OUT / "instruments.parquet")
    launch = dict(inst.select("symbol", "launch_ms").iter_rows())

    def one(sym: str):
        p = BARS / f"{sym}.parquet"
        old = pl.read_parquet(p) if p.is_file() else None
        s0 = max(ms(start), int(launch.get(sym, 0) or 0))
        first = int(old["ts"].min()) if old is not None and old.height else ms(datetime.now(UTC))
        if first - s0 <= 2 * 86_400_000:
            return sym, 0 if old is None else old.height, 0
        new = fetch_klines(sym, datetime.fromtimestamp(s0 / 1000, UTC), datetime.fromtimestamp(first / 1000, UTC))
        df = new if old is None else pl.concat([new, old]).unique("ts", keep="first").sort("ts")
        if df.height:
            df.write_parquet(p, compression="zstd", compression_level=9)
        return sym, df.height, new.height

    res = {}
    with ThreadPoolExecutor(workers) as ex:
        for sym, n, k in ex.map(one, symbols):
            res[sym] = {"rows": n, "backfilled": k}
            log(f"  {sym}: {n} rows (+{k} older)", flush=True)
    res["_fails"] = KLINE_FAILS
    return res


def window_bars(symbol: str, t0_ms: int, before_h: int = 50, after_h: int = 50) -> pl.DataFrame:
    """5m bars around an event for a non-core symbol, cached per symbol (bars5m_win/<SYM>.parquet)."""
    WIN.mkdir(parents=True, exist_ok=True)
    p = WIN / f"{symbol}.parquet"
    old = pl.read_parquet(p) if p.is_file() else None
    a, b = t0_ms - before_h * 3_600_000, t0_ms + after_h * 3_600_000
    if old is not None and old.height:
        have = old.filter((pl.col("ts") >= a) & (pl.col("ts") < b)).height
        if have >= 0.95 * (b - a) / 300_000:
            return old.filter((pl.col("ts") >= a) & (pl.col("ts") < b))
    new = fetch_klines(symbol, datetime.fromtimestamp(a / 1000, UTC), datetime.fromtimestamp(b / 1000, UTC))
    df = new if old is None else pl.concat([old, new]).unique("ts", keep="first").sort("ts")
    if df.height:
        df.write_parquet(p, compression="zstd", compression_level=9)
    return df.filter((pl.col("ts") >= a) & (pl.col("ts") < b))


# ------------------------------------------------------------------------------------------------ trade archives
def _download(url: str, dest: Path, attempts: int = 4) -> bool:
    """Stream a file to disk in 1 MB blocks (no whole-file buffer). False on 404 / repeated failure."""
    for a in range(attempts):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=300) as r, open(dest, "wb") as f:
                while True:
                    blk = r.read(1 << 20)
                    if not blk:
                        break
                    f.write(blk)
            return True
        except Exception as e:  # noqa: BLE001
            dest.unlink(missing_ok=True)
            if "404" in str(e):
                return False
            time.sleep(3 * (a + 1))
    return False


ARCHIVE_SKIPPED: list[tuple[str, str, float]] = []


def _content_mb(url: str) -> float | None:
    try:
        req = urllib.request.Request(url, headers={"User-Agent": UA}, method="HEAD")
        with urllib.request.urlopen(req, timeout=60) as r:
            n = r.headers.get("Content-Length")
            return int(n) / 1e6 if n else None
    except Exception:  # noqa: BLE001
        return None


def archive_trades(symbol: str, day: str, max_mb: float = 150.0) -> pl.DataFrame | None:
    """One day of public trades (public.bybit.com/trading). Columns t (s, float64), d (+1 taker buy / -1, int8), size,
    price (float64), notional (float32). The gz file is streamed to data/raw/WMN/, parsed in chunks and deleted."""
    import pandas as pd
    RAW.mkdir(parents=True, exist_ok=True)
    url = f"https://public.bybit.com/trading/{symbol}/{symbol}{day}.csv.gz"
    tmp = RAW / f"{symbol}{day}.csv.gz"
    w = time.time() - _last.get("public.bybit.com", 0.0)
    if w < 0.3:
        time.sleep(0.3 - w)
    _last["public.bybit.com"] = time.time()
    mb = _content_mb(url)
    if mb is not None and mb > max_mb:  # e.g. TRUMPUSDT 2025-01-18 is 406 MB compressed: skipped, logged
        ARCHIVE_SKIPPED.append((symbol, day, mb))
        return None
    if not _download(url, tmp):
        return None
    parts = []
    try:
        for ch in pd.read_csv(tmp, compression="gzip", usecols=["timestamp", "side", "size", "price"], chunksize=1_000_000,
                              dtype={"timestamp": "float64", "side": "category", "size": "float64", "price": "float64"}):
            parts.append(pl.DataFrame({"t": ch["timestamp"].to_numpy(), "d": np.where(ch["side"].astype(str).to_numpy() == "Buy", 1, -1).astype(np.int8),
                                       "size": ch["size"].to_numpy(), "price": ch["price"].to_numpy(),
                                       "notional": (ch["size"].to_numpy() * ch["price"].to_numpy()).astype(np.float32)}))
    finally:
        tmp.unlink(missing_ok=True)
    if not parts:
        return None
    return pl.concat(parts).sort("t")


def trades_to_5m(tr: pl.DataFrame) -> pl.DataFrame:
    return (tr.with_columns(((pl.col("t") * 1000).cast(pl.Int64) // 300_000 * 300_000).alias("ts"))
              .group_by("ts").agg(pl.col("price").first().alias("o"), pl.col("price").max().alias("h"),
                                  pl.col("price").min().alias("l"), pl.col("price").last().alias("c"),
                                  pl.col("notional").sum().alias("turnover"))
              .sort("ts").with_columns(pl.col(c).cast(pl.Float32) for c in ("o", "h", "l", "c", "turnover")))


# --------------------------------------------------------------------------------------------- event-set builder
def _eid(*parts) -> str:
    return hashlib.sha1("|".join(str(p) for p in parts).encode()).hexdigest()[:16]


def classify_bybit(it: dict) -> str:
    t = (it.get("title") or "")
    tl = t.lower()
    key = (it.get("type") or {}).get("key", "")
    tags = [x.lower() for x in it.get("tags") or []]
    deriv = ("perpetual" in tl or "derivatives" in tags or "contract" in tl or "futures" in tl)
    if "tradfi" in tl or "xstock" in tl or "stock" in tl or "cfd" in tl:
        return "tradfi"
    if key == "delistings" or "delist" in tl:
        return "delist_perp" if deriv else "delist_spot"
    if any(w in tl for w in ("launchpool", "launchpad", "token splash", "airdrop", "puzzle hunt", "megadrop", "prize pool")):
        return "launch_promo"
    if key == "new_crypto" or "listing" in tl or "new arrivals" in tl or "will list" in tl:
        if "pre-market" in tl or "premarket" in tl or "pre-listing" in tl:
            return "premarket"
        return "listing_perp" if deriv else "listing_spot"
    if key == "maintenance_updates" or "upgrade" in tl or "maintenance" in tl or "suspension" in tl or "hard fork" in tl:
        return "maintenance"
    if any(w in tl for w in ("launchpool", "launchpad", "token splash", "airdrop", "puzzle hunt", "megadrop")):
        return "launch_promo"
    if key == "product_updates" and any(w in tl for w in ("funding", "leverage", "risk limit", "margin", "tick size",
                                                           "price limit", "adjust", "interval")):
        return "contract_update"
    return "other"


def classify_generic(title: str) -> str:
    tl = title.lower()
    if "delist" in tl or "remove" in tl or "removal" in tl:
        return "x_delist"
    if "list" in tl or "launch" in tl or "will add" in tl or "now available" in tl or "trading pairs" in tl:
        return "x_listing"
    if "suspen" in tl or "maintenance" in tl or "upgrade" in tl or "resum" in tl or "migration" in tl or "fork" in tl:
        return "x_maintenance"
    return "x_other"


def build_events(bybit: list[dict], okx: list[dict], kucoin: list[dict], hacks: list[dict], pmap: dict[int, str],
                 hn: list[dict], wiki: dict[str, list[tuple[str, int]]], inst: pl.DataFrame, macro: pl.DataFrame | None) -> pl.DataFrame:
    tag_map = build_tag_map(inst)
    syms = set(inst.filter(pl.col("symbol_type").is_in(CRYPTO_TYPES))["symbol"].to_list())
    rows = []

    def add(source, category, scope, symbols, known_ts, occur_ts, precision, title, url, extra=None, anchor_ts=None):
        rows.append({"event_id": _eid(source, known_ts, title, ",".join(symbols)), "source": source, "category": category,
                     "scope": scope, "symbols": symbols, "known_ts": int(known_ts),
                     "occur_ts": int(occur_ts) if occur_ts else int(known_ts),
                     "anchor_ts": int(anchor_ts) if anchor_ts else int(known_ts),
                     "precision": precision, "title": (title or "")[:240], "url": (url or "")[:300],
                     "extra": json.dumps(extra or {}, default=str)})

    for it in bybit:
        title = it.get("title") or ""
        desc = it.get("description") or ""
        pub = int(it.get("publishTime") or 0)
        dts = int(it.get("dateTimestamp") or 0)
        kt = pub or dts
        if not kt:
            continue
        cat = classify_bybit(it)
        s = tag_text(title + " " + desc, tag_map, syms)
        if cat in ("listing_spot", "listing_perp", "delist_spot", "delist_perp", "maintenance"):
            s = tag_text(title, tag_map, syms, caps=True) + [x for x in s if x not in tag_text(title, tag_map, syms, caps=True)]
        occ = parse_sched(desc) or parse_sched(title)
        add("bybit_ann", cat, "token" if s else "untagged", s, kt, occ, "1ms" if pub else "1ms_date", title, it.get("url"),
            {"type": (it.get("type") or {}).get("key"), "tags": it.get("tags"), "ts_date": dts, "ts_publish": pub,
             "start": it.get("startDateTimestamp"), "end": it.get("endDateTimestamp")})
    for it in okx:
        title = it.get("title") or ""
        kt = int(it.get("pTime") or 0)
        if not kt:
            continue
        cg = classify_generic(title)
        s = tag_text(title, tag_map, syms, caps=cg != "x_other")
        add("okx_ann", cg, "token" if s else "untagged", s, kt, None, "1ms", title, it.get("url"),
            {"annType": it.get("annType"), "businessPTime": it.get("businessPTime")})
    for it in kucoin:
        title = it.get("annTitle") or ""
        kt = int(it.get("cTime") or 0)
        if not kt:
            continue
        cg = classify_generic(title)
        s = tag_text(title + " " + (it.get("annDesc") or ""), tag_map, syms, caps=cg != "x_other")
        add("kucoin_ann", cg, "token" if s else "untagged", s, kt, None, "1ms", title, it.get("annUrl"),
            {"annType": it.get("annType"), "query_type": it.get("_query_type"), "annId": it.get("annId")})
    for h in hacks:
        try:
            d0 = int(h["date"]) * 1000
        except (KeyError, TypeError, ValueError):
            continue
        day0 = d0 // 86_400_000 * 86_400_000
        sym = pmap.get(int(h["defillamaId"])) if h.get("defillamaId") not in (None, "") and str(h.get("defillamaId")).isdigit() else None
        s = tag_map.get(sym, []) if sym else []
        amt = float(h.get("amount") or 0.0)
        scope = "token" if s else ("market" if amt >= 100e6 else "untagged")
        add("defillama", "hack", scope, s, day0 + 86_400_000, None, "1d", h.get("name") or "", h.get("source") or "",
            {"amount": amt, "chain": h.get("chain"), "technique": h.get("technique"), "classification": h.get("classification"),
             "protocol_symbol": sym}, anchor_ts=day0)
    for x in hn:
        tgt = NAME_TERMS.get(x["term"])
        if not tgt:
            continue
        kt = x["created_at_i"] * 1000
        if tgt == "market":
            add("hn", "hn_story", "market", [], kt, None, "1s", x["title"], x["url"], {"term": x["term"], "id": x["objectID"]})
        else:
            add("hn", "hn_story", "token", [tgt], kt, None, "1s", x["title"], x["url"], {"term": x["term"], "id": x["objectID"]})
    for art, series in wiki.items():
        tgt = WIKI_ARTICLES.get(art)
        if not tgt or len(series) < 90:
            continue
        v = np.log1p(np.array([s[1] for s in series], dtype=float))
        days = [s[0] for s in series]
        for i in range(60, len(v)):
            base = v[i - 60:i]
            mu, sd = np.median(base), 1.4826 * np.median(np.abs(base - np.median(base))) + 1e-6
            z = (v[i] - mu) / sd
            if z >= 4.0:
                d = datetime.strptime(days[i], "%Y%m%d").replace(tzinfo=UTC)
                kt = ms(d + timedelta(days=1) + timedelta(hours=24))
                add("wiki", "wiki_spike", "market" if tgt == "market" else "token", [] if tgt == "market" else [tgt], kt,
                    None, "1d", art, "", {"z": round(float(z), 2), "views": int(np.expm1(v[i]))}, anchor_ts=ms(d))
    if macro is not None:
        for kind, name, sts in macro.select("kind", "name", pl.col("scheduled_ts").dt.epoch("ms")).iter_rows():
            if kind in ("cpi", "nfp", "fomc", "fomc_unscheduled"):
                add("macro", "macro_release", "market", [], sts, sts, "1min", name, "", {"kind": kind})
    df = pl.DataFrame(rows, schema={"event_id": pl.Utf8, "source": pl.Utf8, "category": pl.Utf8, "scope": pl.Utf8,
                                    "symbols": pl.List(pl.Utf8), "known_ts": pl.Int64, "occur_ts": pl.Int64,
                                    "anchor_ts": pl.Int64, "precision": pl.Utf8, "title": pl.Utf8, "url": pl.Utf8,
                                    "extra": pl.Utf8})
    df = df.unique("event_id", keep="first").sort("known_ts")
    return df.with_columns(pl.from_epoch("known_ts", "ms").dt.replace_time_zone("UTC").alias("known_dt"),
                           (pl.from_epoch("known_ts", "ms").dt.weekday() >= 6).alias("weekend"),
                           pl.when(pl.col("known_ts") < ms(CUTOFF)).then(pl.lit("llm_in_sample")).otherwise(pl.lit("clean_holdout")).alias("segment"))


def collect_all(log=print) -> dict:
    """Fetch every event source into data/raw/WMN/*.json (raw; deleted by the builder after use). Resumable: a source
    whose raw file exists is not fetched again."""
    RAW.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    res = {}

    def step(name, fn):
        p = RAW / name
        if p.is_file():
            res[name] = "cached"
            return
        x = fn()
        p.write_text(json.dumps(x), encoding="utf-8")
        res[name] = len(x)
        log(f"{name}: {len(x)} ({time.time() - t0:.0f} s)", flush=True)

    if not (OUT / "instruments.parquet").is_file():
        fetch_instruments().write_parquet(OUT / "instruments.parquet")
    step("bybit_ann.json", lambda: fetch_bybit_ann(log))
    step("okx_ann.json", lambda: fetch_okx_ann(log))
    step("kucoin_ann.json", lambda: fetch_kucoin_ann(log=log))

    def hacks():
        hk, pm = fetch_hacks(log)
        return {"hacks": hk, "pmap": pm}
    step("hacks.json", hacks)

    def hn():
        out = []
        for term in NAME_TERMS:
            x = fetch_hn_term(term, log=log)
            log(f"  hn {term}: {len(x)}", flush=True)
            out.extend(x)
        return out
    step("hn.json", hn)
    step("wiki.json", lambda: {art: fetch_wiki(art) for art in WIKI_ARTICLES})
    res["secs"] = round(time.time() - t0, 1)
    return res


def build_from_raw(log=print, delete_raw: bool = True) -> pl.DataFrame:
    inst = pl.read_parquet(OUT / "instruments.parquet")
    rd = lambda n: json.loads((RAW / n).read_text(encoding="utf-8"))  # noqa: E731
    hk = rd("hacks.json")
    macro_p = ROOT / "data" / "research" / "news" / "events" / "events.parquet"
    macro = pl.read_parquet(macro_p) if macro_p.is_file() else None
    ev = build_events(rd("bybit_ann.json"), rd("okx_ann.json"), rd("kucoin_ann.json"), hk["hacks"],
                      {int(k): v for k, v in hk["pmap"].items()}, rd("hn.json"), rd("wiki.json"), inst, macro)
    ev.write_parquet(OUT / "events.parquet", compression="zstd")
    log(f"events: {ev.height}")
    if delete_raw:
        for n in ("bybit_ann.json", "okx_ann.json", "kucoin_ann.json", "hacks.json", "hn.json", "wiki.json"):
            (RAW / n).unlink(missing_ok=True)
    return ev


def _cli(argv=None) -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["events", "build", "bars", "backfill", "gdelt"])
    ap.add_argument("--keep-raw", action="store_true")
    a = ap.parse_args(argv)
    OUT.mkdir(parents=True, exist_ok=True)
    if a.cmd == "events":
        print(json.dumps(collect_all(log=print), indent=1), flush=True)
    elif a.cmd == "build":
        build_from_raw(delete_raw=not a.keep_raw)
    elif a.cmd == "gdelt":
        fetch_gdelt_recent()
    elif a.cmd == "backfill":
        r = backfill_core_bars()
        (OUT / "bars_backfill.json").write_text(json.dumps({"fetched_at": datetime.now(UTC).isoformat(), "symbols": r}, indent=1, default=str))
    elif a.cmd == "bars":
        r = update_core_bars()
        (OUT / "bars_manifest.json").write_text(json.dumps({"fetched_at": datetime.now(UTC).isoformat(), "symbols": r}, indent=1))



# ------------------------------------------------------------------------------------------ GDELT (live read only)
GDELT_TERMS = {"market": "(crypto OR cryptocurrency OR bitcoin)", "hack": "(hack OR exploit OR hacked) (crypto OR defi)",
               "WLDUSDT": "worldcoin", "WLFIUSDT": "\"world liberty financial\"", "HYPEUSDT": "hyperliquid"}


def fetch_gdelt_recent(days: int = 7, log=print) -> pl.DataFrame:
    """15-min article counts for the last `days` per term (GDELT DOC timelinevolraw). Throttled: gaps accepted."""
    from ai_trader.news import gdelt as gd
    end = datetime.now(UTC).replace(minute=0, second=0, microsecond=0)
    start = end - timedelta(days=days)
    rows = []
    for sym, q in GDELT_TERMS.items():
        d = gd._request({"query": f"{q} sourcelang:english", "mode": "timelinevolraw", "format": "json",
                         "startdatetime": start.strftime("%Y%m%d%H%M%S"), "enddatetime": end.strftime("%Y%m%d%H%M%S")},
                        attempts=3, log=log)
        if d is None:
            log(f"  gdelt {sym}: failed", flush=True)
            rows.append({"symbol": sym, "ts": None, "value": None, "norm": None, "status": "failed"})
            continue
        df = gd.parse_timeline(d, "timelinevolraw")
        log(f"  gdelt {sym}: {df.height} buckets", flush=True)
        for ts, v, n in df.iter_rows():
            rows.append({"symbol": sym, "ts": ts, "value": v, "norm": n, "status": "ok"})
    out = pl.DataFrame(rows, schema={"symbol": pl.Utf8, "ts": pl.Datetime("us", "UTC"), "value": pl.Float64, "norm": pl.Float64, "status": pl.Utf8})
    out.write_parquet(OUT / "gdelt_recent.parquet")
    return out


if __name__ == "__main__":
    _cli()
