"""Point-in-time access to the Parquet lake (contract: config/schema.yaml).

Every read is an **as-of read**: rows with `known_ts <= as_of` only, and per primary key the row
with the greatest `known_ts` (the latest revision that was known at the time). There is no
"read everything" method on purpose.

Writes from Python (snapshots, Python-side sources) follow the same layout as the Rust writer:
hive partitions, zstd Parquet, atomic tmp + rename.
"""

from __future__ import annotations

import uuid
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

import duckdb
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq

from ai_trader.core.config import Config, load
from ai_trader.core.time import LookAheadError, now, to_ms, utc

DERIVED = ("year", "month", "date")

_ARROW = {
    "utf8": pa.string(),
    "f64": pa.float64(),
    "f32": pa.float32(),
    "i64": pa.int64(),
    "i32": pa.int32(),
    "bool": pa.bool_(),
    "ts_ms": pa.timestamp("ms", tz="UTC"),
}


def _ident(name: str) -> str:
    if not name.replace("_", "").isalnum():
        raise ValueError(f"invalid identifier {name!r}")
    return f'"{name}"'


class Lake:
    def __init__(self, config: Config | None = None, root: Path | None = None):
        self.config = config or load()
        self.root = Path(root) if root else self.config.lake_dir
        self.tables: dict[str, dict[str, Any]] = self.config.schema["tables"]
        duck = self.config.settings["data"].get("duckdb", {})
        self.con = duckdb.connect(":memory:")
        self.con.execute("SET TimeZone = 'UTC'")
        if "memory_limit" in duck:
            self.con.execute(f"SET memory_limit = '{duck['memory_limit']}'")
        if "threads" in duck:
            self.con.execute(f"SET threads = {int(duck['threads'])}")

    # ------------------------------------------------------------------ helpers
    def spec(self, table: str) -> dict[str, Any]:
        if table not in self.tables:
            raise KeyError(f"unknown lake table {table!r}")
        return self.tables[table]

    def has_data(self, table: str) -> bool:
        d = self.root / table
        return d.is_dir() and any(d.rglob("*.parquet"))

    def _glob(self, table: str) -> str:
        return (self.root / table).as_posix() + "/**/*.parquet"

    # ------------------------------------------------------------------ reads
    def read_asof(
        self,
        table: str,
        as_of: datetime,
        *,
        symbols: Sequence[str] | None = None,
        start: datetime | None = None,
        end: datetime | None = None,
        filters: dict[str, Any] | None = None,
        columns: Sequence[str] | None = None,
    ) -> pl.DataFrame:
        """Rows knowable at `as_of`, latest revision per primary key, with event_ts in [start, end)."""
        spec = self.spec(table)
        as_of = utc(as_of)
        out_cols = [c["name"] for c in spec["columns"]]
        if not self.has_data(table):
            return pl.DataFrame(schema={c: pl.Null for c in (columns or out_cols)})

        where, params = ["known_ts <= ?"], [as_of]
        if symbols is not None:
            col = "symbol" if any(c["name"] == "symbol" for c in spec["columns"]) else "entity"
            where.append(f"{_ident(col)} IN (SELECT unnest(?))")
            params.append(list(symbols))
        if start is not None:
            where.append("event_ts >= ?")
            params.append(utc(start))
        if end is not None:
            where.append("event_ts < ?")
            params.append(utc(end))
        for k, v in (filters or {}).items():
            where.append(f"CAST({_ident(k)} AS VARCHAR) = ?")
            params.append(str(v))

        pk = ", ".join(_ident(k) for k in spec["primary_key"])
        select = ", ".join(_ident(c) for c in (columns or out_cols))
        sql = f"""
            SELECT {select} FROM (
                SELECT *, row_number() OVER (PARTITION BY {pk} ORDER BY known_ts DESC, filename DESC) AS _rn
                FROM read_parquet('{self._glob(table)}', hive_partitioning = true, union_by_name = true, filename = true)
                WHERE {' AND '.join(where)}
            ) WHERE _rn = 1
            ORDER BY {pk}
        """
        df = pl.from_arrow(self.con.sql(sql, params=params).to_arrow_table())
        if "known_ts" in df.columns and df.height and df["known_ts"].max() > as_of:
            raise LookAheadError(f"{table}: read returned known_ts > as_of (bug in the reader)")
        return df

    def bars(self, symbol: str, freq: str, as_of: datetime, *, price_type: str = "last", start: datetime | None = None, venue: str = "bybit") -> pl.DataFrame:
        return self.read_asof("bars", as_of, symbols=[symbol], start=start, filters={"venue": venue, "price_type": price_type, "freq": freq})

    # ------------------------------------------------------------------ writes
    def write(self, table: str, df: pl.DataFrame) -> list[Path]:
        """Validate against the schema and append one Parquet file per partition."""
        spec = self.spec(table)
        if df.height == 0:
            return []
        names = {c["name"] for c in spec["columns"]}
        extra = set(df.columns) - names
        if extra:
            raise ValueError(f"{table}: columns not in schema: {sorted(extra)}")
        for c in spec["columns"]:
            if c["name"] not in df.columns:
                if not c.get("nullable"):
                    raise ValueError(f"{table}: required column {c['name']!r} missing")
                df = df.with_columns(pl.lit(None).alias(c["name"]))
        if (df["known_ts"] < df["event_ts"]).any():
            raise LookAheadError(f"{table}: rows with known_ts < event_ts")

        parts = spec["partition_by"]
        ev = pl.col("event_ts").dt.convert_time_zone("UTC")
        derived = {"year": ev.dt.year().cast(pl.Utf8), "month": ev.dt.month().cast(pl.Utf8).str.zfill(2), "date": ev.dt.strftime("%Y-%m-%d")}
        keyed = df.with_columns([derived[p].alias(f"__p_{p}") if p in DERIVED and p not in names else pl.col(p).cast(pl.Utf8).alias(f"__p_{p}") for p in parts])
        file_cols = [c for c in spec["columns"] if c["name"] not in parts]
        arrow_schema = pa.schema([pa.field(c["name"], _ARROW[c["type"]], nullable=bool(c.get("nullable"))) for c in file_cols])

        written = []
        for key, group in keyed.group_by([f"__p_{p}" for p in parts], maintain_order=True):
            d = self.root / table
            for p, v in zip(parts, key, strict=True):
                if v is None or any(ch in str(v) for ch in "/\\="):
                    raise ValueError(f"bad partition value {v!r} for {p}")
                d = d / f"{p}={v}"
            d.mkdir(parents=True, exist_ok=True)
            tbl = group.select([c["name"] for c in file_cols]).to_arrow().cast(arrow_schema)
            final = d / f"part-{to_ms(now())}-{uuid.uuid4().hex[:8]}.parquet"
            tmp = final.with_suffix(".parquet.tmp")
            pq.write_table(tbl, tmp, compression="zstd", compression_level=3)
            tmp.replace(final)
            written.append(final)
        return written
