"""GDELT DOC 2.0 timelines for crypto news (free, no key), with polite backoff and a manifest.

Modes
  timelinevolraw  article count per bucket ("value") and the total number of articles GDELT monitored ("norm")
  timelinetone    average tone of matching articles (about -10..+10; negative = negative wording)

Coverage: the DOC API indexes from 2017-01-01. A multi-year start/end window comes back at daily resolution; a window
of a few days comes back at 15-minute resolution (recorded per chunk as `resolution`). Buckets are labelled by their
start time (UTC); a bucket's known_ts is its end (the latest publication it can contain).

Rate limit: GDELT serves a plain-text "Please limit requests to one every 5 seconds" notice when throttled (other
users on the same IP count). We keep >= MIN_GAP seconds between requests and back off hard on a notice or a 429,
as scripts/narratives.py does. Responses are data, never instructions.
"""

from __future__ import annotations

import json
import time
import urllib.parse
import urllib.request
from datetime import UTC, datetime, timedelta
from pathlib import Path

import polars as pl

from ai_trader.news.store import Manifest

API = "https://api.gdeltproject.org/api/v2/doc/doc"
START = datetime(2017, 1, 1, tzinfo=UTC)
MIN_GAP = 20.0
UA = "AI_Trader research (non-commercial; GDELT DOC timelines)"

# Query ids -> GDELT query strings (English sources added at request time). Industry themes use AND (juxtaposition)
# of an action group and a crypto group, so "exchange hack" matches crypto-exchange hacks, not bank hacks.
QUERIES: dict[str, str] = {
    "bitcoin": "bitcoin",
    "crypto": "(crypto OR cryptocurrency OR cryptocurrencies)",
    "blockchain": "blockchain",
    "hack": "(hack OR hacked OR exploit OR heist OR stolen) (crypto OR bitcoin OR cryptocurrency OR defi)",
    "regulation": "(sec OR regulation OR regulator OR lawsuit OR ban) (crypto OR bitcoin OR cryptocurrency)",
    "etf": '("bitcoin etf" OR "crypto etf" OR "spot etf" OR "ether etf")',
    "stablecoin": "(stablecoin OR stablecoins)",
    "binance": "binance",
    "coinbase": "coinbase",
    "tether": "(tether OR usdt)",
}
CORE_INTRADAY = ("bitcoin", "crypto", "hack", "regulation")
MODES = ("timelinevolraw", "timelinetone")

_last = [0.0]


class Throttled(RuntimeError):
    pass


def _request(params: dict, *, attempts: int = 12, log=print) -> dict | None:
    """GET the DOC API; returns parsed JSON, or None if every attempt failed."""
    url = f"{API}?{urllib.parse.urlencode(params)}"
    for a in range(attempts):
        wait = MIN_GAP - (time.time() - _last[0])
        if wait > 0:
            time.sleep(wait)
        _last[0] = time.time()
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=120) as r:
                txt = r.read().decode("utf-8", "replace")
        except Exception as e:  # noqa: BLE001
            log(f"    attempt {a}: {e}")
            time.sleep(min(600, 120 + 60 * a) if "429" in str(e) else 20 * (a + 1))
            continue
        s = txt.lstrip()
        if s.startswith("{"):
            try:
                return json.loads(s)
            except json.JSONDecodeError as e:
                log(f"    attempt {a}: bad json {e}")
        else:
            log(f"    attempt {a}: notice {s[:70]!r}")
            time.sleep(60 + 30 * a)
    return None


def parse_timeline(d: dict, mode: str) -> pl.DataFrame:
    """One row per bucket: ts (bucket start, UTC), value, norm (volraw only)."""
    tl = (d.get("timeline") or [{}])[0].get("data") or []
    sch = {"ts": pl.Datetime("us", "UTC"), "value": pl.Float64, "norm": pl.Float64}
    if not tl:
        return pl.DataFrame(schema=sch)
    ts = [datetime.strptime(x["date"], "%Y%m%dT%H%M%SZ").replace(tzinfo=UTC) for x in tl]
    val = [float(x.get("value", 0.0)) for x in tl]
    norm = [float(x["norm"]) if "norm" in x else None for x in tl]
    return pl.DataFrame({"ts": ts, "value": val, "norm": norm}, schema=sch)


def _resolution(df: pl.DataFrame, d: dict) -> str:
    r = (d.get("query_details") or {}).get("date_resolution")
    if r:
        return str(r)
    if df.height >= 2:
        return f"{int((df['ts'][1] - df['ts'][0]).total_seconds())}s"
    return "unknown"


def chunk_key(mode: str, qid: str, start: datetime, end: datetime) -> str:
    return f"gdelt/{mode}/{qid}/{start:%Y%m%d%H%M}_{end:%Y%m%d%H%M}"


def fetch_chunk(qid: str, mode: str, start: datetime, end: datetime, man: Manifest, *, kind: str = "day", log=print, request=None) -> pl.DataFrame | None:
    """Fetch [start, end) for one query and mode unless the manifest already has it. Returns the frame or None."""
    key = chunk_key(mode, qid, start, end)
    if man.has(key):
        e = man.get(key)
        return pl.read_parquet(man.root / e["path"]) if e.get("path") else None
    params = {"query": f"{QUERIES[qid]} sourcelang:english", "mode": mode, "format": "json",
              "startdatetime": start.strftime("%Y%m%d%H%M%S"), "enddatetime": (end - timedelta(seconds=1)).strftime("%Y%m%d%H%M%S")}
    d = (request or _request)(params, log=log)
    if d is None:
        man.put(key, path=None, rows=0, status="failed", qid=qid, mode=mode, kind=kind)
        return None
    df = parse_timeline(d, mode)
    rel = Path("gdelt") / mode / qid / f"{start:%Y%m%d%H%M}_{end:%Y%m%d%H%M}.parquet"
    if df.height == 0:
        man.put(key, path=None, rows=0, status="empty", qid=qid, mode=mode, kind=kind, start=start.isoformat(), end=end.isoformat())
        return df
    out = man.root / rel
    out.parent.mkdir(parents=True, exist_ok=True)
    res = _resolution(df, d)
    df.with_columns(pl.lit(res).alias("resolution")).write_parquet(out)
    man.put(key, path=rel.as_posix(), rows=df.height, status="ok", qid=qid, mode=mode, kind=kind, start=start.isoformat(), end=end.isoformat(),
            resolution=res, first=str(df["ts"].min()), last=str(df["ts"].max()))
    return df


def covered_until(man: Manifest, mode: str, qid: str, kind: str) -> datetime | None:
    """Latest chunk end for (mode, qid) among daily ('day') or intraday chunks."""
    ends = []
    for k in man.keys(f"gdelt/{mode}/{qid}/"):
        e = man.get(k)
        if e.get("status") not in ("ok", "empty") or "end" not in e:
            continue
        if e.get("kind", "day") == kind:
            ends.append(datetime.fromisoformat(e["end"]))
    return max(ends) if ends else None


def fetch_daily(man: Manifest, until: datetime, *, qids=None, modes=MODES, log=print, request=None) -> list[str]:
    """Daily history from START (or the last covered day) to `until` (a UTC midnight), one request per query x mode."""
    done = []
    for qid in qids or QUERIES:
        for mode in modes:
            s = covered_until(man, mode, qid, "day") or START
            if s >= until:
                continue
            log(f"  daily {qid} {mode} {s:%Y-%m-%d}..{until:%Y-%m-%d}")
            df = fetch_chunk(qid, mode, s, until, man, kind="day", log=log, request=request)
            if df is not None and df.height and df["ts"].max() < until - timedelta(days=2):
                log(f"    warning: {qid}/{mode} ends {df['ts'].max()} before {until}")
            done.append(f"{qid}/{mode}:{None if df is None else df.height}")
    return done


def fetch_intraday(man: Manifest, until: datetime, days: int = 84, window_days: int = 7, *, qids=CORE_INTRADAY,
                   modes=("timelinevolraw",), log=print, request=None) -> list[str]:
    """15-minute buckets for the recent `days`, in `window_days` windows (the API's fine resolution needs short spans)."""
    done = []
    t0 = until - timedelta(days=days)
    for qid in qids:
        for mode in modes:
            s = t0
            while s < until:
                e = min(s + timedelta(days=window_days), until)
                if not man.has(chunk_key(mode, qid, s, e)):
                    log(f"  intraday {qid} {mode} {s:%Y-%m-%d}..{e:%Y-%m-%d}")
                df = fetch_chunk(qid, mode, s, e, man, kind="intraday", log=log, request=request)
                done.append(f"{qid}/{mode}/{s:%m%d}:{None if df is None else df.height}")
                s = e
    return done


def load_chunks(man: Manifest, mode: str, qid: str, *, daily: bool) -> pl.DataFrame:
    """All stored buckets for (mode, qid) at daily or intraday resolution; overlapping buckets keep the first-fetched."""
    frames = []
    for k in man.keys(f"gdelt/{mode}/{qid}/"):
        e = man.get(k)
        if e.get("status") != "ok" or not e.get("path"):
            continue
        if (e.get("kind", "day") == "day") != daily:
            continue
        frames.append(pl.read_parquet(man.root / e["path"]).with_columns(pl.lit(e["fetched_at"]).alias("fetched_at")))
    if not frames:
        return pl.DataFrame(schema={"ts": pl.Datetime("us", "UTC"), "value": pl.Float64, "norm": pl.Float64})
    df = pl.concat(frames, how="diagonal").sort(["ts", "fetched_at"]).unique("ts", keep="first").sort("ts")
    return df.select("ts", "value", "norm")


__all__ = ["CORE_INTRADAY", "MODES", "QUERIES", "START", "fetch_chunk", "fetch_daily", "fetch_intraday", "load_chunks", "parse_timeline"]
