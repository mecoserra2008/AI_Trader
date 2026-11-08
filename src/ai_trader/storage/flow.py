"""Order-flow storage lifecycle and derivations (docs/ORDER_FLOW.md).

Base layer: flow_3s (written live by `atx collect`). From it:
  rollup()     exact 3 s -> 1 m candles (flow_1m, the long-term tier)
  compact()    merge a day's 5-minute flush files into one file
  derive()     SMA/EMA feature stacks over many windows, computed on read and never stored
  enforce_budget()   prune the oldest rolled-up flow_3s days while the lake exceeds storage.budget_gb

    uv run python -m ai_trader.storage.flow status | maintain [--dry-run]
"""

from __future__ import annotations

import argparse
import shutil
import sys
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import polars as pl

from ai_trader.core.config import load
from ai_trader.storage.lake import Lake

SUM_COLS = ["buy_vol", "sell_vol", "buy_notional", "sell_notional", "n_buy", "n_sell", "book_updates"]
TWA_COLS = ["spread_bps", "micro_bps", "obi_l1", "obi_5", "obi_10", "obi_25", "obi_50", "obi_100",
            "depth_bid_25", "depth_ask_25", "depth_bid_100", "depth_ask_100"]


# ---------------------------------------------------------------- rollup & compaction

def rollup(df3: pl.DataFrame, every: str = "1m") -> pl.DataFrame:
    """Exact aggregation of 3 s candles. Flows are summed; time-weighted book features are averaged
    with book_cov as weight (so a candle with half a valid book counts half); prices combine as OHLC."""
    cov = pl.col("book_cov")
    w = lambda c: ((pl.col(c) * cov).sum() / cov.filter(pl.col(c).is_not_null()).sum()).cast(pl.Float32).alias(c)  # noqa: E731
    vol = pl.col("buy_vol") + pl.col("sell_vol")
    out = (
        df3.sort("event_ts")
        .group_by_dynamic("event_ts", every=every, group_by=["venue", "category", "symbol"], closed="left", label="left")
        .agg(
            pl.col("open").drop_nulls().first(),
            pl.col("high").max(),
            pl.col("low").min(),
            pl.col("close").drop_nulls().last(),
            ((pl.col("vwap") * vol).sum() / vol.filter(pl.col("vwap").is_not_null()).sum()).alias("vwap"),
            *[pl.col(c).sum() for c in SUM_COLS],
            pl.col("max_trade").max(),
            pl.col("mid").drop_nulls().last(),
            *[w(c) for c in TWA_COLS],
            pl.when(pl.col("ofi").is_not_null().any()).then(pl.col("ofi").sum()).otherwise(None).cast(pl.Float32).alias("ofi"),
            pl.col("book_span_bps").min(),
            cov.mean().cast(pl.Float32).alias("book_cov"),
            pl.col("known_ts").max(),
            pl.len().alias("_n"),
        )
    )
    return out.with_columns(pl.col("vwap").fill_nan(None), *[pl.col(c).fill_nan(None) for c in TWA_COLS])


def _day_dirs(lake: Lake, table: str) -> dict[tuple[str, str, str], Path]:
    out = {}
    for d in (lake.root / table).glob("venue=*/category=*/date=*"):
        if d.is_dir():
            out[(d.parent.parent.name.split("=")[1], d.parent.name.split("=")[1], d.name.split("=")[1])] = d
    return out


def compact(day_dir: Path) -> int:
    """Merge all part files of one partition into a single file. Returns files removed."""
    files = sorted(day_dir.glob("*.parquet"))
    if len(files) <= 1:
        return 0
    df = pl.read_parquet(files)
    tmp = day_dir / "compacted.parquet.tmp"
    df.sort("symbol", "event_ts").write_parquet(tmp, compression="zstd", compression_level=6, statistics=True)
    final = day_dir / f"part-compacted-{files[-1].stem.split('-')[1]}.parquet"
    tmp.replace(final)
    for f in files:
        if f != final:
            f.unlink()
    return len(files) - 1


def rollup_day(lake: Lake, venue: str, category: str, day: str) -> int:
    d0 = datetime.fromisoformat(day).replace(tzinfo=UTC)
    df3 = lake.read_asof("flow_3s", datetime.now(UTC), start=d0, end=d0 + timedelta(days=1), filters={"venue": venue, "category": category})
    if df3.height == 0:
        return 0
    m = rollup(df3).drop("_n")
    lake.write("flow_1m", m)
    return m.height


# ---------------------------------------------------------------- derived feature stacks

BASE_FEATURES = {
    "delta": pl.col("buy_vol") - pl.col("sell_vol"),
    "delta_notional": pl.col("buy_notional") - pl.col("sell_notional"),
    "volume_notional": pl.col("buy_notional") + pl.col("sell_notional"),
    "ofi": pl.col("ofi").fill_null(0.0),
    "obi_l1": pl.col("obi_l1"),
    "obi_5": pl.col("obi_5"),
    "obi_10": pl.col("obi_10"),
    "obi_25": pl.col("obi_25"),
    "micro_bps": pl.col("micro_bps"),
    "spread_bps": pl.col("spread_bps"),
    "depth_imb_25": (pl.col("depth_bid_25") - pl.col("depth_ask_25")) / (pl.col("depth_bid_25") + pl.col("depth_ask_25")),
}


def derive(df: pl.DataFrame, windows: list[int], features: list[str] | None = None, *, ema_of_sma: bool = True) -> pl.DataFrame:
    """Per symbol, for each feature f and window w (in candles of `df`):
         f_sma_w  rolling mean over w candles
         f_ema_w  EMA with half-life w applied to f_sma_w (EMA-of-SMA; ema_of_sma=False -> EMA of f)
         f_z_w    (f − rolling mean) / rolling std over w: standardised flow, comparable across symbols.
    Strictly trailing (causal). Computed on read, never stored."""
    feats = features or list(BASE_FEATURES)
    keys = [k for k in ("category", "symbol") if k in df.columns]
    base = df.sort(*keys, "event_ts").with_columns([BASE_FEATURES[f].cast(pl.Float64).alias(f"f_{f}") for f in feats])
    exprs = []
    for f in feats:
        x = pl.col(f"f_{f}")
        for w in windows:
            sma = x.rolling_mean(w, min_samples=w)
            src = sma if ema_of_sma else x
            exprs += [
                sma.over(keys).alias(f"{f}_sma_{w}"),
                src.ewm_mean(half_life=w, adjust=False, ignore_nulls=True).over(keys).alias(f"{f}_ema_{w}"),
                ((x - x.rolling_mean(w, min_samples=w)) / x.rolling_std(w, min_samples=w)).over(keys).alias(f"{f}_z_{w}"),
            ]
    return base.with_columns(exprs)


# ---------------------------------------------------------------- storage budget

def dir_size(p: Path) -> int:
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file()) if p.exists() else 0


def status(lake: Lake | None = None) -> pl.DataFrame:
    lake = lake or Lake()
    rows = [{"table": t.name, "gb": dir_size(t) / 1e9, "files": sum(1 for _ in t.rglob("*.parquet"))} for t in sorted(lake.root.iterdir()) if t.is_dir()]
    return pl.DataFrame(rows).sort("gb", descending=True) if rows else pl.DataFrame()


def maintain(lake: Lake | None = None, dry_run: bool = False, today: date | None = None) -> list[str]:
    """Daily job: compact + roll up every complete flow_3s day, then prune oldest rolled-up 3 s days
    (never younger than flow_3s_min_days) until the lake fits storage.budget_gb."""
    lake = lake or Lake()
    cfg = load().settings["storage"]
    today = today or datetime.now(UTC).date()
    log: list[str] = []
    rolled = set(_day_dirs(lake, "flow_1m"))
    for key, d in sorted(_day_dirs(lake, "flow_3s").items()):
        if date.fromisoformat(key[2]) >= today:
            continue  # the current day is still being written
        if key not in rolled:
            if not dry_run:
                n = compact(d)
                rows = rollup_day(lake, *key)
                log.append(f"rolled {key}: {rows} 1m rows (compacted {n} files)")
            else:
                log.append(f"would roll {key}")
            rolled.add(key)
    budget = cfg["budget_gb"] * 1e9
    total = dir_size(lake.root)
    min_keep = today - timedelta(days=cfg["flow_3s_min_days"])
    for key, d in sorted(_day_dirs(lake, "flow_3s").items(), key=lambda kv: kv[0][2]):
        if total <= budget:
            break
        if key not in rolled or date.fromisoformat(key[2]) >= min_keep:
            continue
        sz = dir_size(d)
        log.append(f"{'would prune' if dry_run else 'pruned'} flow_3s {key} ({sz / 1e6:.0f} MB)")
        if not dry_run:
            shutil.rmtree(d)
        total -= sz
    if total > budget:
        log.append(f"WARNING: lake {total / 1e9:.1f} GB still above budget {cfg['budget_gb']} GB (nothing prunable)")
    return log


def _cli(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="flow")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("status")
    m = sub.add_parser("maintain")
    m.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    if a.cmd == "status":
        s = status()
        sys.stdout.write(f"{s}\ntotal {s['gb'].sum():.3f} GB of {load().settings['storage']['budget_gb']} GB\n")
    else:
        for line in maintain(dry_run=a.dry_run):
            sys.stdout.write(line + "\n")


if __name__ == "__main__":
    _cli()
