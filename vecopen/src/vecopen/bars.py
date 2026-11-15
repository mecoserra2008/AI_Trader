"""Phase 1: session conventions and resampling of 5m base bars (SPEC sections 2, 6 and 8).

``resample(df, tf, session)`` aggregates base bars into one of the SPEC timeframes
(5m, 15m, 30m, 1h, 2h, 4h, 1d, 1w, 1M). The session convention of each asset is fixed in
``vecopen/config/sessions.yaml`` (construction rule 4) and read with ``load_sessions()``.

Bar keys (construction rule 1). A base bar is keyed by its close ``ts_close``; its open is
``ts_close - 5m``. A higher-timeframe bar is keyed by the *nominal* close of its bucket, never
by the close of its last base bar: a bucket whose base bars stop early (weekend, gap) is still
only known at the bucket's end. Joining it to base bars must use a backward as-of join on
``ts_close <= t``. By default the last bucket is dropped when its nominal close lies after the
last base bar (it is not complete yet), so truncating the input at ``t`` never creates a bar
keyed at or before ``t`` that the full data would not also have.

Buckets (see the header of ``sessions.yaml``): the trade date of a base bar is the calendar date
of its local open time shifted by ``24h - day_start``; a day bucket runs from ``day_start`` on the
eve of the trade date to the next one, in the session time zone (DST-aware); intraday buckets
count elapsed real time from the start of the trade day and are cut at the next day start;
weeks are Monday..Sunday trade dates; months are calendar months of trade dates.

Aggregation: ``o`` first, ``h`` max, ``l`` min, ``c`` last, ``v``/``buy_v``/``sell_v`` sums,
``n_base`` = number of base bars in the bucket. For roll-adjusted series the unadjusted columns
aggregate the same way and ``roll_offset`` takes the last value.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Any

import polars as pl
import yaml

from vecopen.config import PROJECT_ROOT, config_sha256

SESSIONS_PATH = PROJECT_ROOT / "config" / "sessions.yaml"

BASE_MINUTES = 5
BASE = timedelta(minutes=BASE_MINUTES)
INTRADAY_MINUTES = {"5m": 5, "15m": 15, "30m": 30, "1h": 60, "2h": 120, "4h": 240}
CALENDAR_TFS = ("1d", "1w", "1M")
TIMEFRAMES = (*INTRADAY_MINUTES, *CALENDAR_TFS)

PRICE_COLS = ("o", "h", "l", "c")
SUM_COLS = ("v", "buy_v", "sell_v")
UNADJ_COLS = ("o_unadj", "h_unadj", "l_unadj", "c_unadj")


@dataclass(frozen=True)
class Session:
    """One bar-boundary convention: time zone, local start of the trading day, week start."""

    name: str
    tz: str
    day_start: str = "00:00"
    week_start: str = "monday"

    def __post_init__(self) -> None:
        if self.week_start != "monday":
            raise ValueError(f"session {self.name}: only week_start=monday is supported")
        h, m = self.day_start_hm
        if not (0 <= h < 24 and 0 <= m < 60) or m % BASE_MINUTES:
            raise ValueError(f"session {self.name}: day_start {self.day_start!r} must be HH:MM on the 5-minute grid")

    @property
    def day_start_hm(self) -> tuple[int, int]:
        h, m = self.day_start.split(":")
        return int(h), int(m)

    @property
    def shift_minutes(self) -> int:
        """Minutes added to local time to get the trade date (0 for a midnight day start)."""
        h, m = self.day_start_hm
        return (24 * 60 - (60 * h + m)) % (24 * 60)


UTC_24H = Session("utc_24h", "UTC", "00:00")


@dataclass(frozen=True)
class SessionTable:
    conventions: dict[str, Session]
    assets: dict[str, dict[str, Any]]
    raw: dict[str, Any]
    sha256: str
    path: Path

    def for_asset(self, asset: str) -> Session:
        try:
            return self.conventions[self.assets[asset]["convention"]]
        except KeyError as exc:
            raise KeyError(f"asset {asset!r} has no session convention in {self.path.name}") from exc

    def roll(self, asset: str) -> str:
        return str(self.assets[asset].get("roll", "none"))


def load_sessions(path: str | Path | None = None) -> SessionTable:
    """Parse ``sessions.yaml`` and hash it (same LF-normalised sha256 as the baseline config)."""
    p = Path(path) if path is not None else SESSIONS_PATH
    raw_bytes = p.read_bytes()
    raw = yaml.safe_load(raw_bytes.decode("utf-8"))
    convs = {
        name: Session(name, c["tz"], str(c.get("day_start", "00:00")), str(c.get("week_start", "monday")))
        for name, c in raw["conventions"].items()
    }
    for asset, a in raw["assets"].items():
        if a.get("convention") not in convs:
            raise ValueError(f"{p.name}: asset {asset} uses unknown convention {a.get('convention')!r}")
        if a.get("roll", "none") not in ("none", "panama", "panama_if_futures_based"):
            raise ValueError(f"{p.name}: asset {asset} has unknown roll {a.get('roll')!r}")
    return SessionTable(convs, dict(raw["assets"]), raw, config_sha256(raw_bytes), p)


# ---------------------------------------------------------------- bucket arithmetic

def _day_start_utc(dates: pl.Expr, s: Session) -> pl.Expr:
    """UTC instant at which trade date ``dates`` starts (local ``day_start`` on its eve, or at midnight)."""
    naive = dates.cast(pl.Datetime("ms")) - pl.duration(minutes=s.shift_minutes)
    return naive.dt.replace_time_zone(s.tz, ambiguous="raise", non_existent="raise").dt.convert_time_zone("UTC")


def trade_date(ts_open: pl.Expr, s: Session) -> pl.Expr:
    local = ts_open.dt.convert_time_zone(s.tz).dt.replace_time_zone(None)
    return (local + pl.duration(minutes=s.shift_minutes)).dt.date()


def bucket_bounds(df: pl.DataFrame, tf: str, s: Session) -> pl.DataFrame:
    """``df`` with columns ``_start``/``_end`` (UTC, ms): the bucket of each base bar (by its open)."""
    if tf not in TIMEFRAMES:
        raise ValueError(f"unknown timeframe {tf!r}; expected one of {TIMEFRAMES}")
    ts_open = pl.col("ts_close") - pl.duration(minutes=BASE_MINUTES)
    out = df.with_columns(_ts_open=ts_open, _td=trade_date(ts_open, s))
    if tf in INTRADAY_MINUTES:
        step = INTRADAY_MINUTES[tf] * 60_000
        out = out.with_columns(_ds=_day_start_utc(pl.col("_td"), s),
                               _de=_day_start_utc(pl.col("_td").dt.offset_by("1d"), s))
        k = (pl.col("_ts_open") - pl.col("_ds")).dt.total_milliseconds() // step
        out = out.with_columns(_start=pl.col("_ds") + pl.duration(milliseconds=k * step))
        out = out.with_columns(_end=pl.min_horizontal(pl.col("_start") + pl.duration(milliseconds=step), pl.col("_de")))
    else:
        if tf == "1d":
            first = pl.col("_td")
            nxt = pl.col("_td").dt.offset_by("1d")
        elif tf == "1w":
            first = pl.col("_td") - pl.duration(days=pl.col("_td").dt.weekday() - 1)
            nxt = first.dt.offset_by("7d")
        else:  # 1M
            first = pl.col("_td").dt.month_start()
            nxt = first.dt.offset_by("1mo")
        out = out.with_columns(_start=_day_start_utc(first, s), _end=_day_start_utc(nxt, s))
    return out.drop([c for c in ("_ts_open", "_td", "_ds", "_de") if c in out.columns])


# ---------------------------------------------------------------- validation and resampling

def check_base(df: pl.DataFrame) -> None:
    """Raise unless ``df`` is a valid base frame: UTC ``ts_close`` on the 5m grid, strictly increasing."""
    if "ts_close" not in df.columns:
        raise ValueError("base bars need a ts_close column")
    dt = df.schema["ts_close"]
    if not isinstance(dt, pl.Datetime) or dt.time_zone != "UTC":
        raise ValueError(f"ts_close must be a UTC datetime, got {dt}")
    if df.height == 0:
        return
    ms = df["ts_close"].dt.epoch("ms")
    if (ms % (BASE_MINUTES * 60_000) != 0).any():
        raise ValueError("ts_close must lie on the 5-minute grid")
    if df.height > 1 and not (ms.diff().drop_nulls() > 0).all():
        raise ValueError("ts_close must be strictly increasing (sorted, no duplicate timestamps)")


def resample(df: pl.DataFrame, tf: str, session: Session, *, complete_only: bool = True) -> pl.DataFrame:
    """Aggregate base bars ``df`` to timeframe ``tf`` under ``session``.

    Returns ``ts_open, ts_close`` (bucket bounds, UTC) and the aggregated columns present in ``df``
    plus ``n_base``. With ``complete_only`` the final bucket is dropped if its nominal close is after
    the last base bar's close.
    """
    check_base(df)
    if tf not in TIMEFRAMES:
        raise ValueError(f"unknown timeframe {tf!r}; expected one of {TIMEFRAMES}")
    cols = set(df.columns)
    aggs: list[pl.Expr] = []
    for c in PRICE_COLS:
        if c in cols:
            aggs.append({"o": pl.col(c).first(), "h": pl.col(c).max(), "l": pl.col(c).min(), "c": pl.col(c).last()}[c])
    for c in UNADJ_COLS:
        if c in cols:
            fn = {"o_unadj": "first", "h_unadj": "max", "l_unadj": "min", "c_unadj": "last"}[c]
            aggs.append(getattr(pl.col(c), fn)())
    for c in SUM_COLS:
        if c in cols:
            aggs.append(pl.col(c).sum())
    if "roll_offset" in cols:
        aggs.append(pl.col("roll_offset").last())
    aggs.append(pl.len().cast(pl.Int64).alias("n_base"))

    if df.height == 0:
        empty = df.head(0).with_columns(ts_open=pl.col("ts_close"), n_base=pl.lit(0, pl.Int64))
        keep = ["ts_open", "ts_close", *[c for c in (*PRICE_COLS, *UNADJ_COLS, *SUM_COLS, "roll_offset") if c in cols], "n_base"]
        return empty.select(keep)

    b = bucket_bounds(df, tf, session)
    out = (b.group_by("_start", "_end", maintain_order=True).agg(aggs)
           .rename({"_start": "ts_open", "_end": "ts_close"}))
    if complete_only:
        out = out.filter(pl.col("ts_close") <= df["ts_close"].max())
    order = ["ts_open", "ts_close", *[c for c in (*PRICE_COLS, *UNADJ_COLS, *SUM_COLS, "roll_offset") if c in cols], "n_base"]
    return out.select(order)
