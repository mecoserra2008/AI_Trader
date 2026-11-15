"""Phase 1: missing 5m intervals per asset and their classification (SPEC section 8, phase 1 acceptance).

A gap is a maximal run of consecutive 5m bars that are absent between an asset's first and last bar. Each run is
given one class:

Crypto (Binance USDT-M trades 24/7, so every missing bar is a gap):
* ``exchange_outage``: venue-wide. At least ``min_peers`` other symbols were listed across the run and at least
  half of them miss at least half of the run's bars too (Binance futures maintenance or outage).
* ``archive_gap``: only this symbol (or fewer than half of its peers) is missing. The archive lacks bars the venue
  printed for the others; a symbol-level halt cannot be told apart from the archive alone, so it is reported here
  and listed individually in the report.
* ``listing_start`` is not a run inside the series: it is the asset's first bar, reported in the history table.
* Present but flat bars (``v == 0``) are not gaps; their runs are classed ``venue_flat`` (at least half the listed
  peers flat too: Binance futures maintenance prints flat bars instead of none) or ``symbol_flat``, and reported
  next to the gaps. They are kept in the data as archived.

CFDs (cTrader trend bars, exchange sessions):
* ``weekend``: every missing bar lies outside the weekly trading schedule and the run covers a Saturday (local).
* ``daily_break``: every missing bar lies outside the schedule, within one weekday (the session's daily pause).
* ``holiday``: the missing bars that fall inside the schedule all fall on the symbol's holiday list.
* ``feed_gap``: some missing bars fall inside the schedule on a non-holiday (missing server history).
The schedule is the symbol's cTrader schedule when resolved, else the nominal ``hours`` of ``sessions.yaml``.

Classification of research-sample gaps uses research-sample data only (each peer is cut at its own split
boundary), so ``python -m vecopen phase1`` recomputes the research gap table from ``data/research`` exactly and no
holdout bar of any asset informs it.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

import numpy as np
import polars as pl

from vecopen.data import TS

STEP_MS = 300_000
SLOTS_PER_WEEK = 7 * 24 * 12
DAYS = {"sun": 0, "mon": 1, "tue": 2, "wed": 3, "thu": 4, "fri": 5, "sat": 6}

RUN_SCHEMA = {"gap_first": TS, "gap_last": TS, "n_missing": pl.Int64}


def missing_runs(ts_close: pl.Series) -> pl.DataFrame:
    """Runs of absent 5m bars between consecutive present bars: first and last missing ``ts_close``, count."""
    ms = ts_close.dt.epoch("ms").to_numpy()
    if ms.size < 2:
        return pl.DataFrame(schema=RUN_SCHEMA)
    d = np.diff(ms)
    idx = np.nonzero(d > STEP_MS)[0]
    first = ms[idx] + STEP_MS
    last = ms[idx + 1] - STEP_MS
    return pl.DataFrame({
        "gap_first": pl.Series(first, dtype=pl.Int64).cast(pl.Datetime("ms")).dt.replace_time_zone("UTC"),
        "gap_last": pl.Series(last, dtype=pl.Int64).cast(pl.Datetime("ms")).dt.replace_time_zone("UTC"),
        "n_missing": pl.Series((last - first) // STEP_MS + 1, dtype=pl.Int64),
    })


def clip_runs(runs: pl.DataFrame, lo: datetime | None, hi: datetime | None) -> pl.DataFrame:
    """Runs cut to missing bars with ``lo <= ts_close <= hi`` (either bound optional)."""
    r = runs
    if lo is not None:
        r = r.filter(pl.col("gap_last") >= lo).with_columns(gap_first=pl.max_horizontal("gap_first", pl.lit(lo).cast(TS)))
    if hi is not None:
        r = r.filter(pl.col("gap_first") <= hi).with_columns(gap_last=pl.min_horizontal("gap_last", pl.lit(hi).cast(TS)))
    return r.with_columns(n_missing=((pl.col("gap_last") - pl.col("gap_first")).dt.total_milliseconds() // STEP_MS + 1)
                          .cast(pl.Int64))


# ---------------------------------------------------------------- crypto

@dataclass(frozen=True)
class Window:
    """Bars of one asset that may be compared: its present span and its runs inside that span."""

    first: datetime
    last: datetime
    runs: pl.DataFrame


def _overlap_bars(runs: pl.DataFrame, a: int, b: int) -> int:
    """Missing bars of ``runs`` inside [a, b] (epoch ms, inclusive)."""
    if runs.height == 0:
        return 0
    f = runs["gap_first"].dt.epoch("ms").to_numpy()
    l = runs["gap_last"].dt.epoch("ms").to_numpy()
    lo = np.maximum(f, a)
    hi = np.minimum(l, b)
    ok = hi >= lo
    return int(((hi[ok] - lo[ok]) // STEP_MS + 1).sum())


def classify_crypto(windows: Mapping[str, Window], *, min_peers: int = 3, share: float = 0.5) -> dict[str, pl.DataFrame]:
    """Class per run for every asset; peers are compared only inside their own windows."""
    out: dict[str, pl.DataFrame] = {}
    epoch = {k: (int(w.first.timestamp() * 1000), int(w.last.timestamp() * 1000)) for k, w in windows.items()}
    for asset, w in windows.items():
        rows = []
        for g in w.runs.iter_rows(named=True):
            a, b = int(g["gap_first"].timestamp() * 1000), int(g["gap_last"].timestamp() * 1000)
            peers = [k for k in windows if k != asset and epoch[k][0] < a and epoch[k][1] > b]
            co = sum(1 for k in peers if _overlap_bars(windows[k].runs, a, b) * 2 >= g["n_missing"])
            cls = "exchange_outage" if len(peers) >= min_peers and co >= share * len(peers) else "archive_gap"
            rows.append({**g, "class": cls, "peers": len(peers), "peers_missing": co})
        out[asset] = pl.DataFrame(rows, schema={**RUN_SCHEMA, "class": pl.Utf8, "peers": pl.Int64,
                                                "peers_missing": pl.Int64}) if rows else \
            pl.DataFrame(schema={**RUN_SCHEMA, "class": pl.Utf8, "peers": pl.Int64, "peers_missing": pl.Int64})
    return out


# ---------------------------------------------------------------- CFD schedules

@dataclass(frozen=True)
class WeeklySchedule:
    """Open 5-minute slots of the week (slot 0 = Sunday 00:00 local) in time zone ``tz``."""

    tz: str
    open_slots: np.ndarray  # bool[SLOTS_PER_WEEK]
    source: str

    @staticmethod
    def from_intervals(tz: str, intervals: Iterable[tuple[int, int]], source: str = "ctrader") -> WeeklySchedule:
        """cTrader ``ProtoOAInterval`` pairs: seconds from Sunday 00:00 in ``tz``, ``[start, end)``."""
        m = np.zeros(SLOTS_PER_WEEK, dtype=bool)
        for s, e in intervals:
            m[int(s) // 300: -(-int(e) // 300)] = True
        return WeeklySchedule(tz, m, source)

    @staticmethod
    def from_hours(tz: str, hours: Mapping[str, object], source: str = "sessions.yaml") -> WeeklySchedule:
        """Nominal hours: ``weekly_open``/``weekly_close`` ("sun 17:00"), optional ``daily_break`` [from, to]."""
        if hours.get("open") == "always":
            return WeeklySchedule(tz, np.ones(SLOTS_PER_WEEK, dtype=bool), source)

        def slot(spec: str) -> int:
            d, hm = str(spec).split()
            h, mi = (int(x) for x in hm.split(":"))
            return DAYS[d.lower()] * 288 + h * 12 + mi // 5

        o, c = slot(str(hours["weekly_open"])), slot(str(hours["weekly_close"]))
        m = np.zeros(SLOTS_PER_WEEK, dtype=bool)
        if o < c:
            m[o:c] = True
        else:
            m[o:] = True
            m[:c] = True
        brk = hours.get("daily_break")
        if brk:
            a = [int(x) for x in str(brk[0]).split(":")]
            b = [int(x) for x in str(brk[1]).split(":")]
            sa, sb = a[0] * 12 + a[1] // 5, b[0] * 12 + b[1] // 5
            tod = np.arange(SLOTS_PER_WEEK) % 288
            in_break = (tod >= sa) & (tod < sb) if sa < sb else (tod >= sa) | (tod < sb)
            m &= ~in_break
        return WeeklySchedule(tz, m, source)

    def local_slots(self, ts_open: pl.Series) -> np.ndarray:
        loc = ts_open.dt.convert_time_zone(self.tz)
        wd = loc.dt.weekday().to_numpy().astype(np.int64) % 7  # polars: Monday=1 .. Sunday=7 -> Sunday=0
        return wd * 288 + loc.dt.hour().to_numpy().astype(np.int64) * 12 + loc.dt.minute().to_numpy().astype(np.int64) // 5

    def is_open(self, ts_open: pl.Series) -> np.ndarray:
        return self.open_slots[self.local_slots(ts_open)]


@dataclass(frozen=True)
class Holiday:
    """cTrader ``ProtoOAHoliday``: a date (local, in ``tz``), optionally recurring yearly, closed [start_s, end_s)
    seconds of that day (whole day if both are None)."""

    day: date
    recurring: bool = False
    start_s: int | None = None
    end_s: int | None = None
    tz: str = "UTC"

    def covers(self, ts_open: datetime) -> bool:
        loc = ts_open.astimezone(ZoneInfo(self.tz))
        same = (loc.month, loc.day) == (self.day.month, self.day.day) if self.recurring else loc.date() == self.day
        if not same:
            return False
        if self.start_s is None or self.end_s is None:
            return True
        sec = loc.hour * 3600 + loc.minute * 60
        return self.start_s <= sec < self.end_s


def classify_cfd(runs: pl.DataFrame, schedule: WeeklySchedule, holidays: Iterable[Holiday] = ()) -> pl.DataFrame:
    """Class per run of a CFD (module docstring). Adds ``in_schedule`` = missing bars inside the schedule."""
    hol = list(holidays)
    rows = []
    for g in runs.iter_rows(named=True):
        closes = pl.datetime_range(g["gap_first"], g["gap_last"], "5m", eager=True, time_unit="ms", time_zone="UTC")
        opens = closes - timedelta(minutes=5)
        inside = schedule.is_open(opens)
        n_in = int(inside.sum())
        if n_in == 0:
            loc = opens.dt.convert_time_zone(schedule.tz)
            cls = "weekend" if (loc.dt.weekday() == 6).any() or loc.dt.date().n_unique() > 2 else "daily_break"
        else:
            inside_ts = opens.filter(pl.Series(inside)).to_list()
            cls = "holiday" if hol and all(any(h.covers(t) for h in hol) for t in inside_ts) else "feed_gap"
        rows.append({**g, "class": cls, "in_schedule": n_in})
    return pl.DataFrame(rows, schema={**RUN_SCHEMA, "class": pl.Utf8, "in_schedule": pl.Int64}) if rows else \
        pl.DataFrame(schema={**RUN_SCHEMA, "class": pl.Utf8, "in_schedule": pl.Int64})


def summarise(classified: pl.DataFrame) -> dict[str, dict[str, int]]:
    """{class: {runs, bars}}."""
    if classified.height == 0:
        return {}
    s = classified.group_by("class").agg(pl.len().alias("runs"), pl.col("n_missing").sum().alias("bars")).sort("class")
    return {r["class"]: {"runs": int(r["runs"]), "bars": int(r["bars"])} for r in s.iter_rows(named=True)}


# ---------------------------------------------------------------- present but flat bars (venue maintenance)

FLAT_CLASSES = {"exchange_outage": "venue_flat", "archive_gap": "symbol_flat"}


def zero_volume_runs(df: pl.DataFrame) -> pl.DataFrame:
    """Runs of consecutive PRESENT bars with ``v == 0`` (same columns as ``missing_runs``). Binance prints such
    flat bars, rather than no bars, during futures maintenance."""
    z = df.filter(pl.col("v") == 0)["ts_close"]
    if z.len() == 0:
        return pl.DataFrame(schema=RUN_SCHEMA)
    ms = z.dt.epoch("ms").to_numpy()
    brk = np.concatenate([[True], np.diff(ms) != STEP_MS])
    starts = np.nonzero(brk)[0]
    ends = np.concatenate([starts[1:] - 1, [ms.size - 1]])
    return pl.DataFrame({
        "gap_first": pl.Series(ms[starts], dtype=pl.Int64).cast(pl.Datetime("ms")).dt.replace_time_zone("UTC"),
        "gap_last": pl.Series(ms[ends], dtype=pl.Int64).cast(pl.Datetime("ms")).dt.replace_time_zone("UTC"),
        "n_missing": pl.Series(ends - starts + 1, dtype=pl.Int64),
    })


def classify_flat(windows: Mapping[str, Window]) -> dict[str, pl.DataFrame]:
    """Zero-volume runs classed like crypto gaps: ``venue_flat`` when at least half the listed peers are flat over
    the run too (maintenance), ``symbol_flat`` otherwise."""
    out = classify_crypto(windows)
    return {a: d.with_columns(pl.col("class").replace(FLAT_CLASSES)) for a, d in out.items()}
