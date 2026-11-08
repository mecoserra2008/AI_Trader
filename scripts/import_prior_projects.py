"""Import reusable datasets from the user's earlier projects into the lake (point-in-time, with provenance).

Sources reviewed in research/notes/{ETH_BMNR_PROJECT,BACKTEST_TREND_FOLLOWING}.md. Every dataset is
validated before writing: Bybit bars must match our own 1h bars, and timestamps must be monotonic and
gap-checked. Never imports secrets (live_snapshot.json, .env, keys).

    PYTHONIOENCODING=utf-8 uv run python scripts/import_prior_projects.py [--only NAME] [--dry-run]
"""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
from pathlib import Path

import polars as pl

from ai_trader.storage.lake import Lake

BTF = Path("C:/Users/Asus/Documents/BACKTEST_TREND_FOLLOWING/data")
EBM = Path("C:/Users/Asus/Documents/ETH_BMNR_PROJECT/data/processed")
LAT = 2_000  # ms after period close
FREQ_MS = {"1m": 60_000, "5m": 300_000, "1h": 3_600_000}


def ms(col: str) -> pl.Expr:
    return pl.col(col).cast(pl.Int64).cast(pl.Datetime("ms", "UTC"))


def bars_frame(df: pl.DataFrame, ts_ms: str, venue: str, symbol: str, freq: str, turnover: str | None = None) -> pl.DataFrame:
    d = FREQ_MS[freq]
    return df.select(
        pl.lit(venue).alias("venue"), pl.lit("last").alias("price_type"), pl.lit(freq).alias("freq"), pl.lit(symbol).alias("symbol"),
        ms(ts_ms).alias("event_ts"), (pl.col(ts_ms).cast(pl.Int64) + d).cast(pl.Datetime("ms", "UTC")).alias("close_ts"),
        pl.col("open").cast(pl.Float64), pl.col("high").cast(pl.Float64), pl.col("low").cast(pl.Float64), pl.col("close").cast(pl.Float64),
        pl.col("volume").cast(pl.Float64),
        (pl.col(turnover).cast(pl.Float64) if turnover else pl.lit(None, dtype=pl.Float64)).alias("turnover"),
        (pl.col(ts_ms).cast(pl.Int64) + d + LAT).cast(pl.Datetime("ms", "UTC")).alias("known_ts"),
    ).filter((pl.col("low") > 0) & (pl.col("high") >= pl.max_horizontal("open", "close")) & (pl.col("low") <= pl.min_horizontal("open", "close"))).unique("event_ts", keep="first").sort("event_ts")


def check_spacing(df: pl.DataFrame, freq: str, name: str) -> None:
    gaps = df["event_ts"].diff().drop_nulls()
    bad = int((gaps.dt.total_milliseconds() != FREQ_MS[freq]).sum())
    print(f"  {name}: {df.height} rows {df['event_ts'].min()} -> {df['event_ts'].max()}, irregular steps {bad}")


def btc_5m(lake: Lake, dry: bool) -> None:
    raw = pl.read_csv(BTF / "btc_jump_regime" / "bybit_btcusdt_5m.csv")
    b = bars_frame(raw, "start_ms", "bybit", "BTCUSDT", "5m", "turnover")
    check_spacing(b, "5m", "bybit BTCUSDT 5m")
    # Validation gate: 5m aggregated to 1h must equal our own Bybit 1h bars.
    agg = b.group_by_dynamic("event_ts", every="1h").agg(pl.col("open").first(), pl.col("high").max(), pl.col("low").min(), pl.col("close").last(), pl.len().alias("n")).filter(pl.col("n") == 12)
    ours = lake.bars("BTCUSDT", "1h", datetime.now(UTC)).select(pl.col("event_ts").dt.cast_time_unit("ms"), "open", "high", "low", "close")
    j = agg.join(ours, on="event_ts", suffix="_ours")
    rel = j.select(*[((pl.col(c) / pl.col(f"{c}_ours")) - 1).abs().max().alias(c) for c in ("open", "high", "low", "close")])
    share = j.select((((pl.col("close") / pl.col("close_ours")) - 1).abs() < 1e-9).mean()).item()
    print(f"  validation vs lake 1h: {j.height} hours compared, exact-close share {share:.5f}, max rel diffs {rel.row(0)}")
    # Bybit's own 2020-21 history is inconsistent across frequencies (data audit 2026-11-07: 1h vs 4h/1d
    # mismatches in the same months). So: exact match required from 2022 on, >= 99.5% overall.
    late = j.filter(pl.col("event_ts").dt.year() >= 2022).select((((pl.col("close") / pl.col("close_ours")) - 1).abs() < 1e-9).mean()).item()
    print(f"  exact-close share 2022+: {late:.6f}")
    if late < 1.0 or share < 0.995:
        raise SystemExit("ABORT: 5m BTC does not reconcile with our Bybit 1h bars")
    if not dry:
        lake.write("bars", b)


def binance_eth(lake: Lake, dry: bool) -> None:
    for freq, f in (("1h", "binance_klines_ETHUSDT_1h.parquet"), ("1m", "binance_klines_ETHUSDT_1m.parquet")):
        raw = pl.read_parquet(EBM / f).with_columns((pl.col("ts_ns") // 1_000_000).alias("ts_ms"))
        b = bars_frame(raw, "ts_ms", "binance", "ETHUSDT", freq)
        check_spacing(b, freq, f"binance ETHUSDT {freq}")
        if not dry:
            lake.write("bars", b)


def deribit(lake: Lake, dry: bool) -> None:
    for cur in ("btc", "eth"):
        d = BTF / f"{cur}_options"
        if not d.is_dir():
            print(f"  skip {cur}: no folder")
            continue
        sym = f"{cur.upper()}-PERPETUAL"
        if (d / "deribit_perp_1h.csv").is_file():
            p = bars_frame(pl.read_csv(d / "deribit_perp_1h.csv"), "ts", "deribit", sym, "1h")
            check_spacing(p, "1h", f"deribit {sym} 1h")
            if not dry:
                lake.write("bars", p)
        if (d / "deribit_funding_1h.csv").is_file():
            f = pl.read_csv(d / "deribit_funding_1h.csv")
            rows = []
            for col, sid in (("interest_8h", "perp_funding_8h"), ("interest_1h", "perp_funding_1h")):
                rows.append(f.select(
                    pl.lit("deribit").alias("source"), pl.lit(sid).alias("series_id"), pl.lit(cur.upper()).alias("entity"),
                    (pl.col("timestamp").cast(pl.Int64) - 3_600_000).cast(pl.Datetime("ms", "UTC")).alias("event_ts"),
                    pl.col(col).cast(pl.Float64).alias("value"), pl.lit(None, dtype=pl.Utf8).alias("vintage"), pl.lit("native").alias("pit_quality"),
                    (pl.col("timestamp").cast(pl.Int64) + LAT).cast(pl.Datetime("ms", "UTC")).alias("known_ts"),
                ))
            a = pl.concat(rows)
            print(f"  deribit {cur} funding: {a.height} rows")
            if not dry:
                lake.write("alt_series", a)
        if (d / "surface_daily_svi_v2.csv").is_file():
            s = pl.read_csv(d / "surface_daily_svi_v2.csv", infer_schema_length=5000)
            series = [f"{m}_{t}" for t in (7, 30, 90) for m in ("atm", "rr25", "bf25")] + ["ts_slope_30_7", "ts_slope_90_30", "basis_ann_30"]
            s = s.filter(pl.col("has_surface") == 1)
            long = s.select("T_ms", *[pl.col(c).cast(pl.Float64) for c in series if c in s.columns]).unpivot(index="T_ms", variable_name="series_id", value_name="value")
            a = long.select(
                pl.lit("deribit_svi").alias("source"), ("iv_" + pl.col("series_id")).alias("series_id"), pl.lit(cur.upper()).alias("entity"),
                # Surface fitted on trades in the window ending at T (08:00 UTC): event = T, known = T + latency.
                ms("T_ms").alias("event_ts"), pl.col("value"), pl.lit("btf_svi_v2").alias("vintage"), pl.lit("native").alias("pit_quality"),
                (pl.col("T_ms").cast(pl.Int64) + 60_000).cast(pl.Datetime("ms", "UTC")).alias("known_ts"),
            ).filter(pl.col("value").is_not_null() & pl.col("value").is_finite())
            print(f"  deribit {cur} SVI surface: {s.height} days -> {a.height} rows ({len(series)} series)")
            if not dry:
                lake.write("alt_series", a)


JOBS = {"btc_5m": btc_5m, "binance_eth": binance_eth, "deribit": deribit}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=list(JOBS))
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    lake = Lake()
    for name, fn in JOBS.items():
        if a.only and name != a.only:
            continue
        print(f"## {name}{' (dry run)' if a.dry_run else ''}")
        fn(lake, a.dry_run)


if __name__ == "__main__":
    main()
