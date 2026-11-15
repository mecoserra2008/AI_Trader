"""Bitcoin on-chain activity from the blockchain.com charts API (public, no key).

    https://api.blockchain.info/charts/<chart>?timespan=all&format=json&sampled=false

Charts (period as served):
  hash-rate             TH/s, daily              network hash-rate estimate
  n-transactions        count, daily             confirmed transactions per day
  transaction-fees-usd  USD, daily               total fees paid to miners per day
  mempool-size          bytes, 15-minute samples unconfirmed-transaction backlog (aggregated to daily mean / hourly last)

Publication lag (observed 2026-11-13 ~20:00 UTC: the latest daily point was 2026-11-12, i.e. day d appears <= ~20 h after
it ends). Conservative rule used everywhere: a daily value for day d has known_ts = d + 1 day + ONCHAIN_LAG (24 h), i.e.
d + 2 days 00:00 UTC. An intraday mempool sample at time s has known_ts = s + MEMPOOL_LAG (1 h). Values may be revised
by the provider; the manifest keeps the first-fetched vintage of each day.
"""

from __future__ import annotations

import json
import urllib.parse
import urllib.request
from datetime import UTC, datetime, timedelta
from pathlib import Path

import polars as pl

from ai_trader.news.store import Manifest

API = "https://api.blockchain.info/charts"
CHARTS = ("hash-rate", "n-transactions", "transaction-fees-usd", "mempool-size")
ONCHAIN_LAG = timedelta(hours=24)
MEMPOOL_LAG = timedelta(hours=1)
UA = "AI_Trader research (blockchain.com charts)"


def _get(chart: str, params: dict) -> dict:
    url = f"{API}/{chart}?{urllib.parse.urlencode(params)}"
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(r.read().decode("utf-8"))


def parse(d: dict) -> pl.DataFrame:
    v = d.get("values") or []
    return pl.DataFrame({"ts": [datetime.fromtimestamp(int(x["x"]), tz=UTC) for x in v], "value": [float(x["y"]) for x in v]},
                        schema={"ts": pl.Datetime("us", "UTC"), "value": pl.Float64})


def fetch(man: Manifest, until: datetime, charts=CHARTS, *, log=print, get=None) -> list[str]:
    """Fetch each chart from its last covered timestamp (or all history) to `until`. Samples at or after `until` are dropped."""
    out = []
    get = get or _get
    for c in charts:
        last = None
        for k in man.keys(f"onchain/{c}/"):
            e = man.get(k)
            if e.get("status") == "ok":
                t = datetime.fromisoformat(e["last"])
                last = t if last is None or t > last else last
        if last is not None and last + timedelta(days=1) >= until:
            continue
        if last is None:
            params = {"timespan": "all", "format": "json", "sampled": "false"}
        else:
            ndays = max(2, (until - last).days + 1)
            params = {"start": (last + timedelta(seconds=1)).strftime("%Y-%m-%dT%H:%M:%S"), "timespan": f"{ndays}days", "format": "json", "sampled": "false"}
        try:
            df = parse(get(c, params))
        except Exception as e:  # noqa: BLE001
            log(f"  onchain {c}: {e}")
            continue
        df = df.filter(pl.col("ts") < until)
        if last is not None:
            df = df.filter(pl.col("ts") > last)
        if df.height == 0:
            continue
        a, b = df["ts"].min(), df["ts"].max()
        rel = Path("onchain") / c / f"{a:%Y%m%d%H%M}_{b:%Y%m%d%H%M}.parquet"
        (man.root / rel).parent.mkdir(parents=True, exist_ok=True)
        df.write_parquet(man.root / rel)
        man.put(f"onchain/{c}/{a:%Y%m%d%H%M}_{b:%Y%m%d%H%M}", path=rel.as_posix(), rows=df.height, first=a.isoformat(), last=b.isoformat())
        log(f"  onchain {c}: {df.height} rows {a:%Y-%m-%d}..{b:%Y-%m-%d}")
        out.append(c)
    return out


def load(man: Manifest, chart: str) -> pl.DataFrame:
    frames = [pl.read_parquet(man.root / man.get(k)["path"]) for k in man.keys(f"onchain/{chart}/") if man.get(k).get("status") == "ok"]
    if not frames:
        return pl.DataFrame(schema={"ts": pl.Datetime("us", "UTC"), "value": pl.Float64})
    return pl.concat(frames).unique("ts", keep="first").sort("ts")


def daily(man: Manifest) -> pl.DataFrame:
    """One row per UTC day: date, known_ts (= date + 1 d + ONCHAIN_LAG) and one column per chart (mempool: daily mean)."""
    out = None
    for c in CHARTS:
        df = load(man, c)
        if df.height == 0:
            continue
        g = df.with_columns(pl.col("ts").dt.date().alias("date")).group_by("date").agg(pl.col("value").mean().alias(c.replace("-", "_")))
        out = g if out is None else out.join(g, on="date", how="full", coalesce=True)
    if out is None:
        return pl.DataFrame()
    return out.sort("date").with_columns((pl.col("date").cast(pl.Datetime("us", "UTC")) + timedelta(days=1) + ONCHAIN_LAG).alias("known_ts"))


__all__ = ["CHARTS", "MEMPOOL_LAG", "ONCHAIN_LAG", "daily", "fetch", "load", "parse"]
