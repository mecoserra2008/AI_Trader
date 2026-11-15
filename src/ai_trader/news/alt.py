"""Non-GDELT news-attention sources as point-in-time daily features (agent W; H-NEWS3 / H-NEWS4).

Live API (stable; called hourly by the live runner)
---------------------------------------------------
    daily_alt_features(as_of: datetime, *, man: Manifest | None = None, refresh: bool = False) -> pl.DataFrame

  Returns ONE row (polars DataFrame, height 1) for the decision time `as_of` (tz-aware UTC; a naive datetime is read
  as UTC). Columns:
    asof                       the as_of passed in (Datetime us UTC)
    ALT_COLS                   wp_btc, wp_ind, wp_max, wp_edits, hn_story, hn_comment, fng, fng_chg (Float64, null =
                               no value known by as_of, or the latest known value is older than STALE)
    raw_*                      raw levels behind them: raw_pv_bitcoin (views), raw_edits (Bitcoin + Cryptocurrency
                               edits), raw_hn_story, raw_hn_comment (summed over terms), raw_fng (0-100)
    day_<src>, kt_<src>        source day used and its known_ts, for src in wiki, edits, hn, fng (kt_<src> <= asof)
  It only reads data/research/news/alt/ (fast, no network) unless refresh=True, which first runs `update(now)` (a few
  incremental requests; do that at most once per day). Every value in the row has known_ts <= as_of.
    alt_table(asof: pl.Series, man=None) -> pl.DataFrame     the same row for many decision times (research path)
    update(until=None, sources=SOURCES, log=print) -> dict   incremental fetch of every source (resume-safe)

Sources (all free; counts and dates only, never article text; User-Agent UA, nothing secret in any request)
  wiki     Wikimedia pageviews REST API, en.wikipedia, all-access, agent=user, daily, from 2015-07-01; one request per
           title. Concepts sum their titles (redirects are counted under their own title): FTX = FTX + FTX (company);
           MicroStrategy = MicroStrategy + Strategy (company). "Bitcoin ETF" does not exist on en.wikipedia (checked
           2026-11-13), so it is not included.
  edits    MediaWiki API prop=revisions (timestamps only, rvlimit=max, paged) for Bitcoin and Cryptocurrency from
           2015-01-01 -> daily edit counts (zero-filled).
  hn       Hacker News via Algolia search_by_date, exact word (typoTolerance=false, queryType=prefixNone; otherwise
           'crypto' prefix-matches 'cryptography'), terms bitcoin / crypto / ethereum x tags story / comment, from
           2017-01-01. Day-aligned windows: a window with nbHits <= 1000 returns every created_at_i (binned by UTC day);
           larger windows are halved; a single day with > 1000 hits uses nbHits. Pace HN_GAP s (< 10k requests / h).
  fng      alternative.me Crypto Fear and Greed index (daily, from 2018-02-01), value 0-100, stamped 00:00 UTC.
  guardian The Guardian Open Platform with api-key=test: refused (HTTP 401) on 2026-11-13, so it is recorded as
           'refused' in the manifest and not in the block. The fetcher stays for a future key-less retry.

Point-in-time (known_ts per source day d)
  wiki     d + 2 days 00:00 UTC (day end + PV_LAG = 24 h; on 2026-11-13 22:53 UTC the latest day served was 11-12).
  edits/hn d + 1 day + COUNT_LAG (00:30 UTC next day; revisions / items are timestamped in real time).
  fng      its own timestamp (00:00 UTC of d; the API's time_until_update points at 00:00 UTC).
  A source row is used at as_of only if known_ts <= as_of, and only if as_of - known_ts <= STALE (3 days).

Features (pre-declared in H-NEWS3: abnormal = ln(1 + x_d) minus the median of ln(1 + x) over days d-28..d-1, needing
>= 20 of them; computed on the full source history, so a source-day row depends only on days <= d)
  wp_btc     abnormal pageviews, Bitcoin
  wp_ind     mean abnormal pageviews over the other concepts (cryptocurrency, ethereum, blockchain, binance, coinbase,
             tether, stablecoin, ftx, satoshi, microstrategy, exchange, bubble, dogecoin), nulls ignored
  wp_max     max abnormal pageviews over all concepts
  wp_edits   abnormal edits, Bitcoin + Cryptocurrency combined
  hn_story   mean over terms of abnormal HN story counts;  hn_comment  the same for comments
  fng        (F&G - 50) / 50;   fng_chg  (F&G_d - F&G_{d-1}) / 50 (null unless the previous value is exactly 1 day older)
"""

from __future__ import annotations

import json
import random
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl

from ai_trader.news.store import Manifest, news_root

UA = "AI_Trader research (news attention counts)"
SOURCES = ("wiki", "edits", "hn", "fng", "guardian")
PV_START, EDITS_START, HN_START = date(2015, 7, 1), date(2015, 1, 1), date(2017, 1, 1)
PV_LAG = timedelta(hours=24)
COUNT_LAG = timedelta(minutes=30)
STALE = timedelta(days=3)
TRAIL, TRAIL_MIN = 28, 20
HN_GAP, WIKI_GAP, DEFAULT_GAP = 0.75, 0.5, 1.0

# concept -> en.wikipedia titles whose daily views are summed
WIKI: dict[str, tuple[str, ...]] = {
    "bitcoin": ("Bitcoin",), "cryptocurrency": ("Cryptocurrency",), "ethereum": ("Ethereum",), "blockchain": ("Blockchain",),
    "binance": ("Binance",), "coinbase": ("Coinbase",), "tether": ("Tether_(cryptocurrency)",), "stablecoin": ("Stablecoin",),
    "ftx": ("FTX", "FTX_(company)"), "satoshi": ("Satoshi_Nakamoto",), "microstrategy": ("MicroStrategy", "Strategy_(company)"),
    "exchange": ("Cryptocurrency_exchange",), "bubble": ("Cryptocurrency_bubble",), "dogecoin": ("Dogecoin",),
}
EDIT_TITLES = ("Bitcoin", "Cryptocurrency")
HN_TERMS = ("bitcoin", "crypto", "ethereum")
HN_TAGS = ("story", "comment")
ALT_COLS = ("wp_btc", "wp_ind", "wp_max", "wp_edits", "hn_story", "hn_comment", "fng", "fng_chg")
RAW_COLS = ("raw_pv_bitcoin", "raw_edits", "raw_hn_story", "raw_hn_comment", "raw_fng")

PV_API = "https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/en.wikipedia/all-access/user"
WIKI_API = "https://en.wikipedia.org/w/api.php"
HN_API = "https://hn.algolia.com/api/v1/search_by_date"
FNG_API = "https://api.alternative.me/fng/"
GUARDIAN_API = "https://content.guardianapis.com/search"


def alt_manifest(root: Path | None = None) -> Manifest:
    """Separate manifest under data/research/news/alt/ (so concurrent writers of the main news manifest never clash)."""
    return Manifest(Path(root) if root is not None else news_root() / "alt")


# ---------------------------------------------------------------- HTTP

class Refused(RuntimeError):
    """HTTP 401 / 403 / 404: the server refuses this request; retrying will not help."""


_last: dict[str, float] = {}


def get_json(url: str, params: dict | None = None, *, min_gap: float = DEFAULT_GAP, attempts: int = 7, base: float = 2.0,
             timeout: float = 60.0, log=print) -> dict:
    """GET JSON with per-host pacing, exponential backoff with jitter (honours Retry-After), and Refused on 401/403/404."""
    full = url + ("?" + urllib.parse.urlencode(params) if params else "")
    host = urllib.parse.urlparse(url).netloc
    err: Exception | None = None
    for a in range(attempts):
        wait = min_gap - (time.time() - _last.get(host, 0.0))
        if wait > 0:
            time.sleep(wait)
        _last[host] = time.time()
        delay = min(600.0, base * 2 ** a) * (1 + 0.25 * random.random())
        try:
            req = urllib.request.Request(full, headers={"User-Agent": UA, "Accept": "application/json"})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.loads(r.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            if e.code in (401, 403, 404):
                raise Refused(f"HTTP {e.code} {host}") from e
            ra = e.headers.get("Retry-After") if e.headers else None
            if ra and ra.isdigit():
                delay = max(delay, float(ra))
            if e.code == 429:
                delay = max(delay, 60.0)
            err = e
        except (urllib.error.URLError, TimeoutError, ConnectionError, json.JSONDecodeError, OSError) as e:
            err = e
        log(f"    {host} attempt {a + 1}/{attempts}: {err}; sleeping {delay:.0f} s")
        time.sleep(delay)
    raise RuntimeError(f"{host}: all {attempts} attempts failed ({err})")


# ---------------------------------------------------------------- chunk store

def _put_chunk(man: Manifest, source: str, series: str, df: pl.DataFrame, first: date, last: date, **meta) -> None:
    """Store daily counts for [first, last] (inclusive) as one parquet chunk."""
    key = f"{source}/{series}/{first:%Y%m%d}_{last:%Y%m%d}"
    rel = Path(source) / series / f"{first:%Y%m%d}_{last:%Y%m%d}.parquet"
    (man.root / rel).parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(man.root / rel)
    man.put(key, path=rel.as_posix(), rows=df.height, status="ok", first=first.isoformat(), last=last.isoformat(), **meta)


def covered_until(man: Manifest, source: str, series: str) -> date | None:
    """Last day (inclusive) covered by an ok chunk of this series."""
    last = None
    for k in man.keys(f"{source}/{series}/"):
        e = man.get(k)
        if e.get("status") == "ok" and e.get("last"):
            d = date.fromisoformat(e["last"])
            last = d if last is None or d > last else last
    return last


def load_series(man: Manifest, source: str, series: str) -> pl.DataFrame:
    """date, value (first-fetched vintage of each day) plus the covered span in .attrs-like columns first/last."""
    frames = [pl.read_parquet(man.root / man.get(k)["path"]) for k in man.keys(f"{source}/{series}/") if man.get(k).get("status") == "ok"]
    if not frames:
        return pl.DataFrame(schema={"date": pl.Date, "value": pl.Float64})
    return pl.concat([f.select("date", pl.col("value").cast(pl.Float64)) for f in frames]).unique("date", keep="first").sort("date")


def _span(start: date, end_incl: date, step_days: int | None = None, by_year: bool = False):
    """Yield (a, b) inclusive spans covering [start, end_incl]: calendar years or fixed steps."""
    a = start
    while a <= end_incl:
        b = date(a.year, 12, 31) if by_year else a + timedelta(days=(step_days or 1) - 1)
        b = min(b, end_incl)
        yield a, b
        a = b + timedelta(days=1)


def _daily_frame(counts: dict[date, float], a: date, b: date, fill: float | None = 0.0) -> pl.DataFrame:
    days = [a + timedelta(days=i) for i in range((b - a).days + 1)]
    return pl.DataFrame({"date": days, "value": [counts.get(d, fill) for d in days]}, schema={"date": pl.Date, "value": pl.Float64})


# ---------------------------------------------------------------- wiki pageviews

def parse_pageviews(d: dict) -> dict[date, float]:
    return {datetime.strptime(x["timestamp"][:8], "%Y%m%d").date(): float(x["views"]) for x in d.get("items", [])}


def fetch_pageviews(man: Manifest, until: date, *, concepts=None, get=None, log=print) -> list[str]:
    """One request per title for [last covered + 1 or PV_START, until - 1]. Days the API omits are stored as null."""
    get = get or (lambda url: get_json(url, min_gap=WIKI_GAP, log=log))
    done = []
    end = until - timedelta(days=1)
    for concept, titles in (concepts or WIKI).items():
        for title in titles:
            series = f"pv_{title}"
            last = covered_until(man, "wiki", series)
            a = (last + timedelta(days=1)) if last else PV_START
            if a > end:
                continue
            url = f"{PV_API}/{urllib.parse.quote(title, safe='')}/daily/{a:%Y%m%d}00/{end:%Y%m%d}00"
            try:
                c = parse_pageviews(get(url))
            except Refused:
                c = {}  # 404: no views recorded for this title in the window
            except Exception as e:  # noqa: BLE001
                log(f"  wiki {title}: {e}")
                continue
            _put_chunk(man, "wiki", series, _daily_frame(c, a, end, fill=None), a, end, concept=concept, n_days=len(c))
            log(f"  wiki {title}: {len(c)} days with views {a}..{end}")
            done.append(title)
    return done


# ---------------------------------------------------------------- wiki edits

def parse_revisions(d: dict) -> list[datetime]:
    """Revision timestamps from a prop=revisions response (formatversion 1 or 2)."""
    pages = d.get("query", {}).get("pages", [])
    pages = pages.values() if isinstance(pages, dict) else pages
    out = []
    for p in pages:
        for r in p.get("revisions", []) or []:
            out.append(datetime.strptime(r["timestamp"], "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=UTC))
    return out


def fetch_edits(man: Manifest, until: date, *, titles=EDIT_TITLES, get=None, log=print) -> list[str]:
    """Daily edit counts for [last covered + 1 or EDITS_START, until - 1] (complete days only), paged 500 at a time."""
    get = get or (lambda params: get_json(WIKI_API, params, min_gap=WIKI_GAP, log=log))
    done = []
    end = until - timedelta(days=1)
    for title in titles:
        last = covered_until(man, "edits", title)
        a = (last + timedelta(days=1)) if last else EDITS_START
        if a > end:
            continue
        params = {"action": "query", "format": "json", "formatversion": "2", "prop": "revisions", "titles": title.replace("_", " "),
                  "rvprop": "timestamp", "rvlimit": "max", "rvdir": "newer", "rvstart": f"{a:%Y-%m-%d}T00:00:00Z",
                  "rvend": f"{end:%Y-%m-%d}T23:59:59Z"}
        counts: dict[date, float] = {}
        pages = 0
        try:
            cont: dict = {}
            while True:
                d = get({**params, **cont})
                pages += 1
                for ts in parse_revisions(d):
                    counts[ts.date()] = counts.get(ts.date(), 0.0) + 1.0
                if "continue" not in d:
                    break
                cont = d["continue"]
        except Exception as e:  # noqa: BLE001
            log(f"  edits {title}: {e} (nothing stored)")
            continue
        _put_chunk(man, "edits", title, _daily_frame(counts, a, end), a, end, requests=pages, edits=int(sum(counts.values())))
        log(f"  edits {title}: {int(sum(counts.values()))} edits {a}..{end} ({pages} requests)")
        done.append(title)
    return done


# ---------------------------------------------------------------- Hacker News (Algolia)

def _ts(d: date) -> int:
    return int(datetime.combine(d, datetime.min.time(), tzinfo=UTC).timestamp())


def hn_params(term: str, tag: str, a: date, b_excl: date, hits: int) -> dict:
    return {"query": term, "tags": tag, "hitsPerPage": str(hits), "typoTolerance": "false", "queryType": "prefixNone",
            "attributesToRetrieve": "created_at_i", "attributesToHighlight": "[]",
            "numericFilters": f"created_at_i>={_ts(a)},created_at_i<{_ts(b_excl)}"}


def parse_hn(d: dict) -> tuple[int, list[int]]:
    return int(d.get("nbHits", 0)), [int(h["created_at_i"]) for h in d.get("hits", []) if "created_at_i" in h]


def hn_counts(term: str, tag: str, a: date, b_incl: date, get, stats: dict | None = None) -> dict[date, float]:
    """Exact daily counts on [a, b_incl] by recursive day-aligned halving (see module docstring)."""
    stats = stats if stats is not None else {}
    d = get(hn_params(term, tag, a, b_incl + timedelta(days=1), 1000))
    stats["requests"] = stats.get("requests", 0) + 1
    nb, ts = parse_hn(d)
    if nb <= 1000 and len(ts) >= nb:
        out: dict[date, float] = {}
        for t in ts:
            k = datetime.fromtimestamp(t, tz=UTC).date()
            out[k] = out.get(k, 0.0) + 1.0
        return out
    if a == b_incl:
        stats["approx_days"] = stats.get("approx_days", 0) + 1
        return {a: float(max(nb, len(ts)))}
    mid = a + timedelta(days=(b_incl - a).days // 2)
    out = hn_counts(term, tag, a, mid, get, stats)
    out.update(hn_counts(term, tag, mid + timedelta(days=1), b_incl, get, stats))
    return out


def fetch_hn(man: Manifest, until: date, *, terms=HN_TERMS, tags=HN_TAGS, start: date = HN_START, stop: date | None = None,
             get=None, log=print) -> list[str]:
    """Per (term, tag), one chunk per calendar year (the last one partial up to until - 1); resumes per year."""
    get = get or (lambda params: get_json(HN_API, params, min_gap=HN_GAP, log=log))
    end = min(until - timedelta(days=1), stop) if stop else until - timedelta(days=1)
    done = []
    for term in terms:
        for tag in tags:
            series = f"{term}_{tag}"
            last = covered_until(man, "hn", series)
            a0 = (last + timedelta(days=1)) if last else start
            for a, b in _span(a0, end, by_year=True):
                stats: dict = {}
                counts: dict[date, float] = {}
                try:
                    m = a
                    while m <= b:  # monthly seeds, halved as needed
                        mb = min(b, (date(m.year + (m.month == 12), m.month % 12 + 1, 1) - timedelta(days=1)))
                        counts.update(hn_counts(term, tag, m, mb, get, stats))
                        m = mb + timedelta(days=1)
                except Exception as e:  # noqa: BLE001
                    log(f"  hn {series} {a.year}: {e} (year not stored; rerun resumes here)")
                    break
                _put_chunk(man, "hn", series, _daily_frame(counts, a, b), a, b, requests=stats.get("requests", 0),
                           approx_days=stats.get("approx_days", 0), items=int(sum(counts.values())))
                log(f"  hn {series} {a}..{b}: {int(sum(counts.values()))} items, {stats.get('requests', 0)} requests")
                done.append(f"{series}/{a.year}")
    return done


# ---------------------------------------------------------------- Fear and Greed

def parse_fng(d: dict) -> pl.DataFrame:
    rows = d.get("data") or []
    return pl.DataFrame({"ts": [datetime.fromtimestamp(int(x["timestamp"]), tz=UTC) for x in rows], "value": [float(x["value"]) for x in rows]},
                        schema={"ts": pl.Datetime("us", "UTC"), "value": pl.Float64}).sort("ts")


def fetch_fng(man: Manifest, now: datetime, *, get=None, log=print) -> int:
    """Values with timestamp <= now; incremental (limit = days since the last stored value + 2)."""
    get = get or (lambda params: get_json(FNG_API, params, log=log))
    old = load_fng(man)
    last = old["ts"].max() if old.height else None
    limit = 0 if last is None else max(2, (now - last).days + 2)
    try:
        df = parse_fng(get({"limit": str(limit), "format": "json"}))
    except Exception as e:  # noqa: BLE001
        log(f"  fng: {e}")
        return 0
    df = df.filter(pl.col("ts") <= now)
    if last is not None:
        df = df.filter(pl.col("ts") > last)
    if df.height == 0:
        return 0
    a, b = df["ts"].min().date(), df["ts"].max().date()
    key = f"fng/index/{a:%Y%m%d}_{b:%Y%m%d}"
    rel = Path("fng") / "index" / f"{a:%Y%m%d}_{b:%Y%m%d}.parquet"
    (man.root / rel).parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(man.root / rel)
    man.put(key, path=rel.as_posix(), rows=df.height, status="ok", first=a.isoformat(), last=b.isoformat())
    log(f"  fng: {df.height} values {a}..{b}")
    return df.height


def load_fng(man: Manifest) -> pl.DataFrame:
    frames = [pl.read_parquet(man.root / man.get(k)["path"]) for k in man.keys("fng/index/") if man.get(k).get("status") == "ok"]
    if not frames:
        return pl.DataFrame(schema={"ts": pl.Datetime("us", "UTC"), "value": pl.Float64})
    return pl.concat(frames).unique("ts", keep="first").sort("ts")


# ---------------------------------------------------------------- Guardian

def parse_guardian(d: dict) -> tuple[list[date], int]:
    r = d.get("response", {})
    days = [datetime.strptime(x["webPublicationDate"][:10], "%Y-%m-%d").date() for x in r.get("results", [])]
    return days, int(r.get("pages", 0))


def fetch_guardian(man: Manifest, until: date, *, key: str = "test", start: date = date(2015, 1, 1), get=None, log=print) -> int:
    """Daily counts of 'bitcoin OR cryptocurrency' articles (webPublicationDate only), one chunk per year."""
    get = get or (lambda params: get_json(GUARDIAN_API, params, log=log))
    end = until - timedelta(days=1)
    last = covered_until(man, "guardian", "bitcoin_crypto")
    n = 0
    for a, b in _span((last + timedelta(days=1)) if last else start, end, by_year=True):
        counts: dict[date, float] = {}
        page, pages = 1, 1
        try:
            while page <= pages:
                days, pages = parse_guardian(get({"q": "bitcoin OR cryptocurrency", "from-date": a.isoformat(), "to-date": b.isoformat(),
                                                  "page-size": "200", "page": str(page), "order-by": "oldest", "api-key": key}))
                for d in days:
                    counts[d] = counts.get(d, 0.0) + 1.0
                page += 1
        except Refused as e:
            man.put("guardian/status", path=None, rows=0, status="refused", error=str(e), key_kind="test")
            log(f"  guardian: refused ({e}); skipped")
            return n
        except Exception as e:  # noqa: BLE001
            log(f"  guardian {a.year}: {e}")
            return n
        _put_chunk(man, "guardian", "bitcoin_crypto", _daily_frame(counts, a, b), a, b, items=int(sum(counts.values())))
        n += 1
    return n


# ---------------------------------------------------------------- update

def update(until: datetime | None = None, *, sources=SOURCES, man: Manifest | None = None, log=print, hn_start: date = HN_START,
           hn_stop: date | None = None) -> dict:
    """Incremental fetch. Daily-count sources store complete UTC days < until's midnight; Fear and Greed stores <= until."""
    man = man or alt_manifest()
    now = until or datetime.now(UTC)
    if now.tzinfo is None:
        now = now.replace(tzinfo=UTC)
    day = now.astimezone(UTC).date()
    out = {}
    if "wiki" in sources:
        out["wiki"] = fetch_pageviews(man, day, log=log)
    if "edits" in sources:
        out["edits"] = fetch_edits(man, day, log=log)
    if "hn" in sources:
        out["hn"] = fetch_hn(man, day, start=hn_start, stop=hn_stop, log=log)
    if "fng" in sources:
        out["fng"] = fetch_fng(man, now, log=log)
    if "guardian" in sources and not (man.get("guardian/status") or {}).get("status") == "refused":
        out["guardian"] = fetch_guardian(man, day, log=log)
    return out


# ---------------------------------------------------------------- features

def abnormal(x: np.ndarray, w: int = TRAIL, min_n: int = TRAIL_MIN) -> np.ndarray:
    """ln(1 + x_d) - median(ln(1 + x_{d-w..d-1})), null (nan) if fewer than min_n prior values or x_d missing."""
    lx = np.log1p(np.asarray(x, dtype=float))
    med = pl.Series(lx).fill_nan(None).shift(1).rolling_median(w, min_samples=min_n).to_numpy().astype(float)
    return lx - med


def _grid(frames: list[pl.DataFrame]) -> pl.DataFrame:
    lo = min(f["date"].min() for f in frames)
    hi = max(f["date"].max() for f in frames)
    return pl.DataFrame({"date": pl.date_range(lo, hi, "1d", eager=True)})


def _kt(dates: pl.Expr, lag: timedelta) -> pl.Expr:
    return dates.cast(pl.Datetime("us", "UTC")) + lag


def wiki_days(man: Manifest) -> pl.DataFrame:
    """date, known_ts, pv_<concept> (raw sum of titles), wp_btc, wp_ind, wp_max."""
    cols: dict[str, pl.DataFrame] = {}
    for concept, titles in WIKI.items():
        parts = []
        for t in titles:
            s = load_series(man, "wiki", f"pv_{t}")
            if s.height == 0:
                continue
            # title days before its first recorded view are 'did not exist' (null); later omitted days are zero views
            first = s.filter(pl.col("value").is_not_null())["date"].min()
            if first is None:
                continue
            parts.append(s.with_columns(pl.when(pl.col("date") >= first).then(pl.col("value").fill_null(0.0)).otherwise(None).alias("value")))
        if not parts:
            continue
        g = pl.concat(parts).group_by("date").agg(pl.when(pl.col("value").is_not_null().any()).then(pl.col("value").sum()).otherwise(None).alias(f"pv_{concept}"))
        cols[concept] = g.sort("date")
    if not cols:
        return pl.DataFrame()
    out = _grid(list(cols.values()))
    for g in cols.values():
        out = out.join(g, on="date", how="left")
    abn = {c: abnormal(out[f"pv_{c}"].cast(pl.Float64).fill_null(np.nan).to_numpy()) for c in cols}
    ind = [c for c in cols if c != "bitcoin"]
    A = pl.DataFrame({f"abn_{c}": v for c, v in abn.items()}).with_columns(pl.all().fill_nan(None))
    out = pl.concat([out, A], how="horizontal")
    return out.with_columns(
        (pl.col("abn_bitcoin") if "bitcoin" in cols else pl.lit(None, pl.Float64)).alias("wp_btc"),
        pl.mean_horizontal([f"abn_{c}" for c in ind]).alias("wp_ind") if ind else pl.lit(None, pl.Float64).alias("wp_ind"),
        pl.max_horizontal([f"abn_{c}" for c in cols]).alias("wp_max"),
        _kt(pl.col("date"), timedelta(days=1) + PV_LAG).alias("known_ts"),
    ).sort("date")


def edits_days(man: Manifest) -> pl.DataFrame:
    frames = [load_series(man, "edits", t).rename({"value": f"ed_{t}"}) for t in EDIT_TITLES]
    frames = [f for f in frames if f.height]
    if not frames:
        return pl.DataFrame()
    out = _grid(frames)
    for f in frames:
        out = out.join(f, on="date", how="left")
    tot = out.select(pl.sum_horizontal([f.columns[1] for f in frames]).cast(pl.Float64).alias("raw_edits"))["raw_edits"]
    return out.with_columns(tot.alias("raw_edits"), pl.Series("wp_edits", abnormal(tot.to_numpy())).fill_nan(None),
                            _kt(pl.col("date"), timedelta(days=1) + COUNT_LAG).alias("known_ts")).sort("date")


def hn_days(man: Manifest) -> pl.DataFrame:
    frames = {}
    for term in HN_TERMS:
        for tag in HN_TAGS:
            s = load_series(man, "hn", f"{term}_{tag}")
            if s.height:
                frames[(term, tag)] = s.rename({"value": f"hn_{term}_{tag}"})
    if not frames:
        return pl.DataFrame()
    out = _grid(list(frames.values()))
    for f in frames.values():
        out = out.join(f, on="date", how="left")
    ex = []
    for tag in HN_TAGS:
        ks = [k for k in frames if k[1] == tag]
        if not ks:
            ex += [pl.lit(None, pl.Float64).alias(f"hn_{tag}"), pl.lit(None, pl.Float64).alias(f"raw_hn_{tag}")]
            continue
        abn = pl.DataFrame({f"a_{t}": abnormal(out[f"hn_{t}_{tag}"].cast(pl.Float64).fill_null(np.nan).to_numpy()) for t, _ in ks}).with_columns(pl.all().fill_nan(None))
        ex += [pl.Series(f"hn_{tag}", abn.select(pl.mean_horizontal(pl.all())).to_series()),
               pl.sum_horizontal([pl.col(f"hn_{t}_{tag}") for t, _ in ks]).alias(f"raw_hn_{tag}")]
    return out.with_columns(*ex, _kt(pl.col("date"), timedelta(days=1) + COUNT_LAG).alias("known_ts")).sort("date")


def fng_days(man: Manifest) -> pl.DataFrame:
    f = load_fng(man)
    if f.height == 0:
        return pl.DataFrame()
    f = f.with_columns(pl.col("ts").dt.date().alias("date")).unique("date", keep="first").sort("date")
    prev_ok = (pl.col("date") - pl.col("date").shift(1)) == timedelta(days=1)
    return f.select("date", pl.col("ts").alias("known_ts"), pl.col("value").alias("raw_fng"),
                    ((pl.col("value") - 50.0) / 50.0).alias("fng"),
                    pl.when(prev_ok).then((pl.col("value") - pl.col("value").shift(1)) / 50.0).otherwise(None).alias("fng_chg"))


def source_tables(man: Manifest | None = None) -> dict[str, pl.DataFrame]:
    """Per-source daily tables (one row per source day, with known_ts)."""
    man = man or alt_manifest()
    w = wiki_days(man)
    if w.height:
        w = w.select("date", "known_ts", "wp_btc", "wp_ind", "wp_max", pl.col("pv_bitcoin").alias("raw_pv_bitcoin"))
    e = edits_days(man)
    if e.height:
        e = e.select("date", "known_ts", "wp_edits", "raw_edits")
    h = hn_days(man)
    if h.height:
        h = h.select("date", "known_ts", "hn_story", "hn_comment", "raw_hn_story", "raw_hn_comment")
    return {"wiki": w, "edits": e, "hn": h, "fng": fng_days(man)}


_SRC_COLS = {"wiki": ("wp_btc", "wp_ind", "wp_max", "raw_pv_bitcoin"), "edits": ("wp_edits", "raw_edits"),
             "hn": ("hn_story", "hn_comment", "raw_hn_story", "raw_hn_comment"), "fng": ("fng", "fng_chg", "raw_fng")}


def alt_table(asof: pl.Series | list, man: Manifest | None = None, tables: dict[str, pl.DataFrame] | None = None) -> pl.DataFrame:
    """One row per decision time: the latest source row with known_ts <= asof (null if older than STALE)."""
    a = pl.Series("asof", list(asof) if not isinstance(asof, pl.Series) else asof).cast(pl.Datetime("us", "UTC"))
    base = pl.DataFrame({"asof": a}).with_row_index("_i").sort("asof")
    tables = tables if tables is not None else source_tables(man)
    for src, cols in _SRC_COLS.items():
        t = tables.get(src)
        if t is None or t.height == 0:
            base = base.with_columns([pl.lit(None, pl.Float64).alias(c) for c in cols] + [pl.lit(None, pl.Date).alias(f"day_{src}"),
                                     pl.lit(None, pl.Datetime("us", "UTC")).alias(f"kt_{src}")])
            continue
        t = t.select(pl.col("date").alias(f"day_{src}"), pl.col("known_ts").cast(pl.Datetime("us", "UTC")).alias(f"kt_{src}"), *cols).sort(f"kt_{src}")
        base = base.join_asof(t, left_on="asof", right_on=f"kt_{src}", strategy="backward")
        fresh = (pl.col("asof") - pl.col(f"kt_{src}")) <= STALE
        base = base.with_columns([pl.when(fresh).then(pl.col(c)).otherwise(None).alias(c) for c in (*cols, f"day_{src}", f"kt_{src}")])
    order = ["asof", *ALT_COLS, *RAW_COLS] + [f"{p}_{s}" for s in _SRC_COLS for p in ("day", "kt")]
    return base.sort("_i").select(order)


def daily_alt_features(as_of: datetime, *, man: Manifest | None = None, refresh: bool = False) -> pl.DataFrame:
    """The stable live entry point (see module docstring): one row of point-in-time alt-news features for `as_of`."""
    if as_of.tzinfo is None:
        as_of = as_of.replace(tzinfo=UTC)
    man = man or alt_manifest()
    if refresh:
        update(as_of, man=man, log=lambda *_: None)
    return alt_table([as_of.astimezone(UTC)], man=man)


def assert_point_in_time(df: pl.DataFrame) -> None:
    for c in df.columns:
        if c.startswith("kt_"):
            bad = df.filter(pl.col(c).is_not_null() & (pl.col(c) > pl.col("asof")))
            if bad.height:
                raise AssertionError(f"{c} after asof in {bad.height} rows")


__all__ = ["ALT_COLS", "RAW_COLS", "Refused", "abnormal", "alt_manifest", "alt_table", "assert_point_in_time", "daily_alt_features",
           "fetch_edits", "fetch_fng", "fetch_guardian", "fetch_hn", "fetch_pageviews", "get_json", "hn_counts", "parse_fng",
           "parse_guardian", "parse_hn", "parse_pageviews", "parse_revisions", "source_tables", "update"]
