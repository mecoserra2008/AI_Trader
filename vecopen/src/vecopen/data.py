"""Phase 1: base-bar store, ``load_bars``, duplicate rule and roll adjustment (SPEC sections 6, 8 and 9).

Layout (all under ``vecopen/data/``, git-ignored)
-------------------------------------------------
* ``research/<ASSET>.parquet``: the first 75% of the asset's history by calendar (``vecopen.split``).
* ``holdout/<ASSET>/part-NNNN.parquet``: the last 25%. The folder carries a deny-read ACL for the current
  user and is never read, listed or sampled before phase 9 (standing rule 4).
* ``staging/<source>/<ASSET>/``: converted archive months waiting for the split; deleted by the split.
* ``raw/``: one archive zip at a time, deleted right after conversion.
* ``manifest.json``, ``split.json``, ``gaps.json``: bookkeeping (``vecopen.ingest``, ``vecopen.split``).

Base-bar schema
---------------
``ts_close`` (Datetime ms, UTC; a 5m bar's open is ``ts_close - 5m``), ``o, h, l, c, v`` (Float64), and for
crypto ``buy_v`` (taker-buy base volume, the true aggressor volume) and ``sell_v = v - buy_v``. CFD ``v`` is
tick volume and ``buy_v``/``sell_v`` are absent (``delta_div`` is dropped for CFDs, SPEC section 4). Rolling
futures-based CFDs (``roll: panama`` in ``config/sessions.yaml``) carry difference-adjusted ``o, h, l, c`` plus
the unadjusted ``o_unadj .. c_unadj`` and ``roll_offset`` (adjusted = unadjusted + roll_offset).

Duplicate rule
--------------
Rows are unique by ``ts_close``. Among rows sharing a ``ts_close``: the row from the higher-priority source wins
(``_src_rank`` 0 = monthly archive, 1 = daily archive); within one source, the row written last wins (a later
revision of the same bar). ``dedup_bars`` reports how many duplicates were identical and how many conflicted.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import polars as pl

from vecopen.config import PROJECT_ROOT

DATA_DIR = PROJECT_ROOT / "data"
RESEARCH_DIR = DATA_DIR / "research"
HOLDOUT_DIR = DATA_DIR / "holdout"
STAGING_DIR = DATA_DIR / "staging"
RAW_DIR = DATA_DIR / "raw"
MANIFEST_PATH = DATA_DIR / "manifest.json"
SPLIT_PATH = DATA_DIR / "split.json"
GAPS_PATH = DATA_DIR / "gaps.json"
ROLLS_DIR = PROJECT_ROOT / "config" / "rolls"

TS = pl.Datetime("ms", "UTC")
BASE_SCHEMA: dict[str, pl.DataType] = {"ts_close": TS, "o": pl.Float64, "h": pl.Float64, "l": pl.Float64,
                                       "c": pl.Float64, "v": pl.Float64}
AGGRESSOR_SCHEMA: dict[str, pl.DataType] = {"buy_v": pl.Float64, "sell_v": pl.Float64}
ROLL_SCHEMA: dict[str, pl.DataType] = {"o_unadj": pl.Float64, "h_unadj": pl.Float64, "l_unadj": pl.Float64,
                                       "c_unadj": pl.Float64, "roll_offset": pl.Float64}


class HoldoutAccessError(PermissionError):
    """Raised for any attempt to load bars from ``data/holdout`` before phase 9 (standing rule 4)."""


def _norm(p: Path) -> str:
    return os.path.normcase(os.path.abspath(os.path.realpath(p)))


def is_holdout_path(path: str | Path) -> bool:
    """True if ``path`` is inside ``vecopen/data/holdout`` or inside any ``data/holdout`` folder."""
    p = Path(path)
    target, hold = _norm(p), _norm(HOLDOUT_DIR)
    if target == hold or target.startswith(hold + os.sep):
        return True
    parts = [x.lower() for x in Path(target).parts]
    return any(a == "data" and b == "holdout" for a, b in zip(parts, parts[1:]))


def refuse_holdout(path: str | Path) -> None:
    if is_holdout_path(path):
        raise HoldoutAccessError(f"refusing to read {path}: data/holdout is locked until phase 9 (standing rule 4)")


def research_path(asset: str, root: str | Path | None = None) -> Path:
    return Path(root if root is not None else RESEARCH_DIR) / f"{asset}.parquet"


def validate_bars(df: pl.DataFrame, *, require_aggressor: bool = False) -> None:
    """Raise ``ValueError`` unless ``df`` follows the base-bar schema with unique, sorted, gridded ``ts_close``."""
    from vecopen.bars import check_base

    for col, dt in BASE_SCHEMA.items():
        if col not in df.columns:
            raise ValueError(f"missing column {col}")
        if df.schema[col] != dt:
            raise ValueError(f"column {col} has dtype {df.schema[col]}, expected {dt}")
    if require_aggressor and not all(c in df.columns for c in AGGRESSOR_SCHEMA):
        raise ValueError("missing buy_v / sell_v")
    check_base(df)


def load_bars(asset: str, *, root: str | Path | None = None, path: str | Path | None = None,
              columns: list[str] | None = None) -> pl.DataFrame:
    """Research-sample 5m bars of ``asset``: ``ts_close`` (UTC), ``o, h, l, c, v`` and optional
    ``buy_v``/``sell_v`` (and the roll columns for Panama-adjusted CFDs).

    Reads ``data/research/<asset>.parquet`` (or ``path``). Any path under ``data/holdout`` is refused with
    ``HoldoutAccessError`` before the file is touched, whatever ``root`` or ``path`` say.
    """
    p = Path(path) if path is not None else research_path(asset, root)
    refuse_holdout(p)
    if root is not None:
        refuse_holdout(root)
    if columns is not None and "ts_close" not in columns:
        columns = ["ts_close", *columns]
    df = pl.read_parquet(p, columns=columns)
    if columns is None:
        validate_bars(df)
    return df


# ---------------------------------------------------------------- duplicates

@dataclass(frozen=True)
class DedupReport:
    rows_in: int
    rows_out: int
    identical: int
    conflicting: int


def dedup_bars(df: pl.DataFrame) -> tuple[pl.DataFrame, DedupReport]:
    """Sort by ``ts_close`` and keep one row per timestamp under the documented rule (module docstring).

    Optional helper columns: ``_src_rank`` (lower wins) and ``_row`` (file order; higher wins). Both are dropped.
    """
    n = df.height
    work = df
    if "_src_rank" not in work.columns:
        work = work.with_columns(_src_rank=pl.lit(0, pl.Int32))
    if "_row" not in work.columns:
        work = work.with_row_index("_row")
    value_cols = [c for c in df.columns if not c.startswith("_")]
    dup_mask = work.select(pl.col("ts_close").is_duplicated()).to_series()
    identical = conflicting = 0
    if dup_mask.any():
        d = work.filter(dup_mask)
        others = [c for c in value_cols if c != "ts_close"]
        per_ts = d.group_by("ts_close").agg(pl.len().alias("n"), pl.struct(others).n_unique().alias("u"))
        identical = int(per_ts.filter(pl.col("u") == 1).select((pl.col("n") - 1).sum()).item() or 0)
        conflicting = int(per_ts.filter(pl.col("u") > 1).select((pl.col("n") - 1).sum()).item() or 0)
    out = (work.sort(["ts_close", "_src_rank", "_row"], descending=[False, False, True])
           .unique(subset="ts_close", keep="first", maintain_order=True)
           .select(value_cols))
    return out, DedupReport(n, out.height, identical, conflicting)


# ---------------------------------------------------------------- roll adjustment

def panama_adjust(df: pl.DataFrame, rolls: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Difference (Panama) back-adjustment of a continuous, unadjusted rolling-CFD series.

    ``rolls`` has ``roll_ts`` (UTC; the first bar *opening* at or after it belongs to the new contract) and an
    optional ``gap`` (new minus old contract price; null = measured as the open of the first bar at or after the
    roll minus the close of the last bar before it). Every bar before a roll is shifted by the sum of the gaps of
    all later rolls, so the latest contract is unadjusted and point distances are preserved (construction rule 5).
    Returns the adjusted frame (``o..c`` adjusted, ``o_unadj..c_unadj``, ``roll_offset``) and the applied rolls.
    """
    from vecopen.bars import BASE

    if df.height == 0:
        raise ValueError("cannot roll-adjust an empty series")
    r = rolls.sort("roll_ts")
    if "gap" not in r.columns:
        r = r.with_columns(gap=pl.lit(None, pl.Float64))
    ts_open = df["ts_close"] - BASE
    applied = []
    for roll_ts, gap in zip(r["roll_ts"].to_list(), r["gap"].to_list()):
        after = (ts_open >= roll_ts)
        if not after.any() or after.all():
            raise ValueError(f"roll at {roll_ts} lies outside the series")
        i = int(after.arg_max())  # first bar opening at or after the roll
        measured = float(df["o"][i] - df["c"][i - 1])
        g = measured if gap is None else float(gap)
        applied.append({"roll_ts": roll_ts, "gap": g, "measured_gap": measured, "first_new_bar": df["ts_close"][i],
                        "gap_source": "measured" if gap is None else "given"})
    ap = pl.DataFrame(applied, schema={"roll_ts": TS, "gap": pl.Float64, "measured_gap": pl.Float64,
                                       "first_new_bar": TS, "gap_source": pl.Utf8})
    # offset of a bar = sum of gaps of all rolls strictly after its open
    offset = pl.Series("roll_offset", [0.0] * df.height)
    for roll_ts, g in zip(ap["roll_ts"].to_list(), ap["gap"].to_list()):
        offset = offset + (ts_open < roll_ts).cast(pl.Float64) * g
    out = df.with_columns(
        o_unadj=pl.col("o"), h_unadj=pl.col("h"), l_unadj=pl.col("l"), c_unadj=pl.col("c"),
        roll_offset=offset,
    ).with_columns([(pl.col(f"{c}_unadj") + pl.col("roll_offset")).alias(c) for c in ("o", "h", "l", "c")])
    return out, ap


def read_roll_calendar(asset: str, rolls_dir: str | Path | None = None) -> pl.DataFrame:
    """``config/rolls/<ASSET>.csv`` (roll_ts_utc, gap, source); raises ``FileNotFoundError`` if missing."""
    p = Path(rolls_dir if rolls_dir is not None else ROLLS_DIR) / f"{asset}.csv"
    if not p.exists():
        raise FileNotFoundError(f"{asset} is roll: panama but its roll calendar {p} does not exist; "
                                "add the broker's roll dates before building it")
    raw = pl.read_csv(p, schema_overrides={"roll_ts_utc": pl.Utf8, "gap": pl.Float64, "source": pl.Utf8})
    return raw.select(roll_ts=pl.col("roll_ts_utc").str.to_datetime(time_unit="ms", time_zone="UTC"),
                      gap=pl.col("gap"), source=pl.col("source"))
