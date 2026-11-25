"""Point-in-time store for FRED macro series (public CSV endpoint, no key).

    PYTHONIOENCODING=utf-8 uv run python -m ai_trader.narrative.macro refresh [--max-age-h 6] [--series DGS2,DGS10]
    PYTHONIOENCODING=utf-8 uv run python -m ai_trader.narrative.macro show DGS2 [--as-of 2026-11-15T19:00:00Z]

Layout (`data/research/macro/`):
  raw/<SERIES>/<stamp>.csv          every download, unchanged, append-only (the audit trail)
  <SERIES>.parquet                  obs_date, value, rule_ts, first_seen_ts, effective_ts, revision
  manifest.json                     per series: url, frequency, known_ts rule, rows, first/last obs, last download, sha256

Point-in-time rule. `rule_ts` is the conservative publication time:
  daily   obs_date + 1 day 00:00 UTC (registered in the MAC block; the market value itself is observable at the US close,
          FRED posts it about a day later, so a stricter +1 day robustness is run in the study);
  monthly a release-lag rule per series (UNRATE / SAHMREALTIME: month end + 10 days; CPILFESL: + 16 days;
          PCEPILFE: + 32 days). Known caveat: a delayed release (e.g. the 2025 shutdown) breaks the rule historically.
`effective_ts` = min(rule_ts, first_seen_ts) for the first value seen of an observation and first_seen_ts for any
later revised value, so a replay never sees a revision before it was downloaded. `read_asof(series, as_of)` returns,
per obs_date, the latest value with effective_ts <= as_of. Monthly values are the latest vintage at download time
(ALFRED `vintage_date` gives exact past vintages, one request per date; `vintage()` fetches one when needed).
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import time
import urllib.request
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

import polars as pl

from ai_trader.core.config import load
from ai_trader.core.time import now

FRED_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={sid}"
ALFRED_URL = "https://alfred.stlouisfed.org/graph/alfredgraph.csv?id={sid}&vintage_date={vd}"

# id -> (frequency, description, monthly lag in days after the observation month's end)
SERIES: dict[str, tuple[str, str, int]] = {
    "DGS2": ("daily", "2-year Treasury constant maturity, %", 0),
    "DGS10": ("daily", "10-year Treasury constant maturity, %", 0),
    "VIXCLS": ("daily", "CBOE VIX close", 0),
    "BAMLH0A0HYM2": ("daily", "ICE BofA US high yield OAS, %", 0),
    "BAMLC0A0CM": ("daily", "ICE BofA US corporate (IG) OAS, %", 0),
    "BAMLC0A4CBBB": ("daily", "ICE BofA BBB US corporate OAS, %", 0),
    "DTWEXBGS": ("daily", "Nominal broad US dollar index", 0),
    "SP500": ("daily", "S&P 500 close (FRED keeps 10 years)", 0),
    "NASDAQCOM": ("daily", "NASDAQ Composite close", 0),
    "NASDAQ100": ("daily", "NASDAQ-100 close", 0),
    "THREEFYTP10": ("daily", "Kim-Wright 10-year term premium, % (ACM is not on FRED)", 0),
    "DFII10": ("daily", "10-year TIPS real yield, %", 0),
    "DFEDTARU": ("daily", "Fed funds target range upper limit, % (2008-12-16 on)", 0),
    "DFEDTARL": ("daily", "Fed funds target range lower limit, % (2008-12-16 on)", 0),
    "DFF": ("daily", "Effective federal funds rate, %", 0),
    "DTB3": ("daily", "3-month Treasury bill secondary market rate (discount basis), %", 0),
    "DGS3MO": ("daily", "3-month Treasury constant maturity, %", 0),
    "T10Y3M": ("daily", "10-year minus 3-month Treasury constant maturity, pp", 0),
    "T10YIE": ("daily", "10-year breakeven inflation, %", 0),
    "BAA10Y": ("daily", "Moody's Baa corporate yield minus 10-year Treasury, pp", 0),
    "DCOILWTICO": ("daily", "WTI crude oil spot, USD/bbl", 0),
    "ECBDFR": ("daily", "ECB deposit facility rate, %", 0),
    "UNRATE": ("monthly", "Unemployment rate, %", 10),
    "SAHMREALTIME": ("monthly", "Real-time Sahm rule recession indicator, pp", 10),
    "CPILFESL": ("monthly", "Core CPI index (SA)", 16),
    "PCEPILFE": ("monthly", "Core PCE price index (SA)", 32),
}
CORE = ["DGS2", "DGS10", "VIXCLS", "BAMLH0A0HYM2", "DTWEXBGS", "SP500", "NASDAQCOM"]


def root() -> Path:
    p = load().root / "data" / "research" / "macro"
    p.mkdir(parents=True, exist_ok=True)
    return p


def _utc(ts: datetime) -> datetime:
    return ts.replace(tzinfo=UTC) if ts.tzinfo is None else ts.astimezone(UTC)


def _write_atomic(path: Path, text: str) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def _fetch(url: str, tries: int = 3) -> str:
    err: Exception | None = None
    for k in range(tries):
        try:
            # A custom User-Agent makes FRED stall (observed 2026-11-15); urllib's default works.
            with urllib.request.urlopen(url, timeout=60) as r:
                txt = r.read().decode()
            if not txt.startswith("observation_date") and not txt.startswith("DATE"):
                raise ValueError(f"unexpected response from {url}: {txt[:80]!r}")
            return txt
        except Exception as e:  # network or format: retry, then raise
            err = e
            time.sleep(2 * (k + 1))
    raise RuntimeError(f"FRED download failed: {url}: {err}")


def parse_csv(txt: str) -> pl.DataFrame:
    df = pl.read_csv(io.StringIO(txt), null_values=["."], infer_schema_length=0)
    df = df.rename({df.columns[0]: "obs_date", df.columns[1]: "value"})
    return (df.with_columns(pl.col("obs_date").str.to_date(), pl.col("value").cast(pl.Float64, strict=False))
            .drop_nulls().sort("obs_date"))


def rule_ts(obs: date, sid: str) -> datetime:
    freq, _, lag = SERIES[sid]
    if freq == "daily":
        return datetime(obs.year, obs.month, obs.day, tzinfo=UTC) + timedelta(days=1)
    nm = date(obs.year + (obs.month == 12), obs.month % 12 + 1, 1)   # first day of the next month = end of obs month
    return datetime(nm.year, nm.month, nm.day, tzinfo=UTC) + timedelta(days=lag)


def _merge(old: pl.DataFrame | None, new: pl.DataFrame, sid: str, seen: datetime) -> pl.DataFrame:
    """Appends new observations and revised values; never rewrites what was already stored."""
    rt = [rule_ts(d, sid) for d in new["obs_date"].to_list()]
    fresh = new.with_columns(pl.Series("rule_ts", rt, dtype=pl.Datetime("us", "UTC")),
                             pl.lit(seen).cast(pl.Datetime("us", "UTC")).alias("first_seen_ts"))
    if old is None or old.height == 0:
        fresh = fresh.with_columns(pl.min_horizontal("rule_ts", "first_seen_ts").alias("effective_ts"), pl.lit(0).alias("revision"))
        return fresh.select("obs_date", "value", "rule_ts", "first_seen_ts", "effective_ts", "revision")
    last = old.sort("revision").group_by("obs_date").agg(pl.col("value").last().alias("v_old"), pl.col("revision").max().alias("r_old"))
    j = fresh.join(last, on="obs_date", how="left")
    brand_new = j.filter(pl.col("v_old").is_null()).with_columns(
        pl.min_horizontal("rule_ts", "first_seen_ts").alias("effective_ts"), pl.lit(0).alias("revision"))
    revised = j.filter(pl.col("v_old").is_not_null() & ((pl.col("value") - pl.col("v_old")).abs() > 1e-9)).with_columns(
        pl.col("first_seen_ts").alias("effective_ts"), (pl.col("r_old") + 1).alias("revision"))
    cols = ["obs_date", "value", "rule_ts", "first_seen_ts", "effective_ts", "revision"]
    return pl.concat([old.select(cols), brand_new.select(cols), revised.select(cols)]).sort("obs_date", "revision")


def manifest() -> dict[str, Any]:
    p = root() / "manifest.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.is_file() else {"series": {}}


def refresh(series: list[str] | None = None, max_age_h: float = 6.0, log=print) -> dict[str, Any]:
    """Downloads each series whose last download is older than max_age_h; returns the manifest."""
    man = manifest()
    t = now()
    for sid in series or list(SERIES):
        ent = man["series"].get(sid, {})
        last = ent.get("last_download")
        if last and (t - datetime.fromisoformat(last)).total_seconds() < max_age_h * 3600:
            continue
        url = FRED_URL.format(sid=sid)
        txt = _fetch(url)
        stamp = f"{t:%Y%m%dT%H%M%S}"
        rd = root() / "raw" / sid
        rd.mkdir(parents=True, exist_ok=True)
        (rd / f"{stamp}.csv").write_text(txt, encoding="utf-8")
        new = parse_csv(txt)
        pq = root() / f"{sid}.parquet"
        old = pl.read_parquet(pq) if pq.is_file() else None
        merged = _merge(old, new, sid, t)
        merged.write_parquet(pq)
        n_rev = int((merged["revision"] > 0).sum())
        man["series"][sid] = {
            "url": url, "frequency": SERIES[sid][0], "description": SERIES[sid][1],
            "known_ts_rule": "obs_date + 1 day 00:00 UTC" if SERIES[sid][0] == "daily" else f"end of obs month + {SERIES[sid][2]} days 00:00 UTC",
            "rows": merged.height, "obs": new.height, "first_obs": str(new["obs_date"].min()), "last_obs": str(new["obs_date"].max()),
            "last_value": float(new["value"][-1]), "last_download": t.isoformat(), "raw": f"raw/{sid}/{stamp}.csv",
            "sha256": hashlib.sha256(txt.encode()).hexdigest(), "revised_rows": n_rev,
            "vintage": "latest at download (FRED graph CSV)",
        }
        log(f"{sid}: {new.height} obs {new['obs_date'].min()}..{new['obs_date'].max()} last {new['value'][-1]}")
        time.sleep(0.5)
    man["updated"] = t.isoformat()
    man["note"] = ("Point-in-time: read with ai_trader.narrative.macro.read_asof(series, as_of); effective_ts = min(rule_ts, "
                   "first_seen_ts) for first values, first_seen_ts for revisions. Never SPYUSDT.")
    _write_atomic(root() / "manifest.json", json.dumps(man, indent=1))
    return man


def vintage(sid: str, vintage_date: date) -> pl.DataFrame:
    """One ALFRED vintage (the series as published on vintage_date); raw file archived."""
    txt = _fetch(ALFRED_URL.format(sid=sid, vd=vintage_date.isoformat()))
    rd = root() / "raw" / sid
    rd.mkdir(parents=True, exist_ok=True)
    (rd / f"vintage_{vintage_date:%Y%m%d}.csv").write_text(txt, encoding="utf-8")
    return parse_csv(txt)


def read_asof(sid: str, as_of: datetime, start: date | None = None) -> pl.DataFrame:
    """obs_date, value, known_ts: the latest value of each observation that was known at as_of."""
    pq = root() / f"{sid}.parquet"
    if not pq.is_file():
        raise FileNotFoundError(f"{pq} missing: run `python -m ai_trader.narrative.macro refresh`")
    df = pl.read_parquet(pq).filter(pl.col("effective_ts") <= _utc(as_of))
    if start is not None:
        df = df.filter(pl.col("obs_date") >= start)
    out = (df.sort("obs_date", "revision").group_by("obs_date", maintain_order=True)
           .agg(pl.col("value").last(), pl.col("effective_ts").first().alias("known_ts")))
    return out.sort("obs_date")


def panel(sids: list[str], as_of: datetime) -> pl.DataFrame:
    """Wide daily frame on the union of observation dates (no forward fill), plus each row's known_ts (max over series)."""
    out: pl.DataFrame | None = None
    for s in sids:
        d = read_asof(s, as_of).select("obs_date", pl.col("value").alias(s))
        out = d if out is None else out.join(d, on="obs_date", how="full", coalesce=True)
    assert out is not None
    return out.sort("obs_date")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("refresh")
    r.add_argument("--max-age-h", type=float, default=6.0)
    r.add_argument("--series", default="")
    s = sub.add_parser("show")
    s.add_argument("series")
    s.add_argument("--as-of", default="")
    a = ap.parse_args()
    if a.cmd == "refresh":
        refresh([x for x in a.series.split(",") if x] or None, a.max_age_h)
    else:
        t = datetime.fromisoformat(a.as_of.replace("Z", "+00:00")) if a.as_of else now()
        print(read_asof(a.series, t).tail(15))


if __name__ == "__main__":
    main()
