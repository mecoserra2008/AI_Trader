"""Staleness of the live tables: per symbol and stream, the age of the newest row knowable now.

    PYTHONIOENCODING=utf-8 uv run python scripts/freshness.py [--days 7] [--all] [SYMBOL ...]

Age = now - the time the newest row's period ended (bars/OI/DVOL: event_ts + period; funding and
Fear & Greed: event_ts). A trailing "!" marks a stream older than its expected cadence
(two periods + slack; funding 8 h + slack). "-" = no row in the lookback window.
Symbols default to settings `live.symbols` (written by `atx update`).
"""

from __future__ import annotations

import argparse
from datetime import timedelta
from pathlib import Path

import polars as pl

from ai_trader.core.config import load
from ai_trader.core.time import now
from ai_trader.storage.lake import Lake

PERIOD = {"1h": timedelta(hours=1), "4h": timedelta(hours=4), "1d": timedelta(days=1)}
SLACK = timedelta(minutes=10)


def live_symbols(cfg) -> list[str]:
    s = cfg.settings.get("live", {}).get("symbols")
    if isinstance(s, str) and "," not in s:
        text = (Path(cfg.root) / s).read_text(encoding="utf-8")
        return [t.upper() for line in text.splitlines() for t in line.split("#")[0].replace(",", " ").split()]
    if isinstance(s, str):
        return [t.strip().upper() for t in s.split(",") if t.strip()]
    return [str(t).upper() for t in (s or [])]


def fmt(age: timedelta | None, limit: timedelta | None) -> str:
    if age is None:
        return "-"
    m = age.total_seconds() / 60
    txt = f"{m:.0f}m" if m < 120 else f"{m / 60:.1f}h" if m < 72 * 60 else f"{m / 1440:.1f}d"
    return txt + ("!" if limit is not None and age > limit else "")


def newest(df: pl.DataFrame, keys: list[str]) -> dict[tuple, object]:
    if df.height == 0:
        return {}
    g = df.group_by(keys).agg(pl.col("event_ts").max())
    return {tuple(r[k] for k in keys): r["event_ts"] for r in g.iter_rows(named=True)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("symbols", nargs="*")
    ap.add_argument("--days", type=float, default=7.0, help="lookback window for the read (default 7)")
    ap.add_argument("--all", action="store_true", help="every symbol found in the window, not just live.symbols")
    a = ap.parse_args()

    cfg = load()
    lake = Lake(cfg)
    t = now()
    start = t - timedelta(days=a.days)
    live = cfg.settings.get("live", {})
    freqs = [str(f) for f in live.get("bar_freqs", ["1h", "4h", "1d"])]
    pts = [str(p) for p in live.get("price_types", ["last", "mark"])]
    oi = (live.get("open_interest") or {}).get("interval", "4h") if live.get("open_interest") is not False else None
    want = [s.upper() for s in a.symbols] or (None if a.all else live_symbols(cfg))

    bars = newest(lake.read_asof("bars", t, symbols=want, start=start, filters={"venue": "bybit"},
                                 columns=["price_type", "freq", "symbol", "event_ts"]), ["price_type", "freq", "symbol"])
    fund = newest(lake.read_asof("funding", t, symbols=want, start=start, filters={"venue": "bybit"},
                                 columns=["symbol", "event_ts"]), ["symbol"])
    ois = newest(lake.read_asof("open_interest", t, symbols=want, start=start, filters={"venue": "bybit", "interval": oi},
                                columns=["symbol", "event_ts"]), ["symbol"]) if oi else {}

    syms = want or sorted({k[2] for k in bars} | {k[0] for k in fund} | {k[0] for k in ois})
    cols = [(pt, f) for pt in pts for f in freqs]
    head = ["symbol"] + [f"{pt} {f}" for pt, f in cols] + ["funding"] + ([f"oi {oi}"] if oi else [])
    rows, n_stale, n_missing = [], 0, 0

    def cell(ts, period: timedelta, limit: timedelta) -> str:
        nonlocal n_stale, n_missing
        if ts is None:
            n_missing += 1
            return fmt(None, None)
        age = t - (ts + period)
        n_stale += age > limit
        return fmt(age, limit)

    for s in syms:
        r = [s]
        for pt, f in cols:
            p = PERIOD.get(f, timedelta(0))
            r.append(cell(bars.get((pt, f, s)), p, p + p + SLACK))
        r.append(cell(fund.get((s,)), timedelta(0), timedelta(hours=8) + SLACK))
        if oi:
            p = PERIOD.get(oi, timedelta(hours=4))
            r.append(cell(ois.get((s,)), p, p + p + SLACK))
        rows.append(r)

    w = [max(len(h), *(len(r[i]) for r in rows)) if rows else len(h) for i, h in enumerate(head)]
    print(f"as of {t:%Y-%m-%d %H:%M:%S} UTC  (age of newest knowable row; '!' = stale, '-' = none in {a.days:g} d)\n")
    print("  ".join(h.ljust(w[i]) if i == 0 else h.rjust(w[i]) for i, h in enumerate(head)))
    for r in rows:
        print("  ".join(c.ljust(w[i]) if i == 0 else c.rjust(w[i]) for i, c in enumerate(r)))

    alt = lake.read_asof("alt_series", t, start=t - timedelta(days=max(a.days, 3)), columns=["source", "series_id", "entity", "event_ts"])
    alt = alt.filter(pl.col("source").is_in(["deribit", "alternative_me"])) if alt.height else alt
    print("\nalt_series")
    res_s = int(live.get("dvol_resolution", 3600))
    for (src, sid, ent), ts in sorted(newest(alt, ["source", "series_id", "entity"]).items()):
        period = timedelta(seconds=res_s) if src == "deribit" else timedelta(0)
        limit = period * 2 + SLACK if src == "deribit" else timedelta(days=1, hours=2)
        age = t - (ts + period)
        n_stale += age > limit
        print(f"  {src:<15} {sid:<12} {ent:<7} newest {ts:%Y-%m-%d %H:%M}  age {fmt(age, limit)}")
    print(f"\n{len(syms)} symbols, {n_stale} stale streams, {n_missing} streams without rows in the window")


if __name__ == "__main__":
    main()
