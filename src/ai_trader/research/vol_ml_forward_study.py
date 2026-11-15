"""H-NN10 forward evaluation: the frozen hourly ML volatility model (data/models/ml/hourly_vol_v1) vs robust_bc x the
hour-of-week factor, on hours after 2026-11-12, universe L.

    PYTHONIOENCODING=utf-8 uv run --extra ml python -m ai_trader.research.vol_ml_forward_study [--record]

Source: the forward log data/research/vol_ml_forward/*.jsonl written by scripts/vol_ml_log.py. For each (symbol,
forecast_for) the first logged line is kept. A line is admissible if:
  * forecast_for >= 2026-11-13 00:00 UTC;
  * it carries both forecasts and has no flags;
  * data_max_ts <= forecast_for + 60 s (the forecast saw nothing of its own bar).
Realised: r^2 of the forecast bar from the lake (ml.hourly.symbol_hourly on market_frame as of now). A bar whose previous
bar is missing has no return and is skipped.
Loss: per-bar QLIKE difference ML - baseline = r^2 (1/F_ml - 1/F_base) + log(F_ml/F_base), averaged over symbols per UTC
date, NW t over dates.
Cross-check (not part of the verdict): the same frozen model reconstructed from the lake for every hour after
2026-11-12 (point-in-time features), its agreement with the logged forecasts, and its own paired difference.
Verdict (registry): supports if diff < 0 and t < -2 over >= 60 forward days; contradicts if diff >= 0 or t > 2;
inconclusive otherwise. Before 60 forward days the study only reports progress and REFUSES to record. There is one row,
segment clean_holdout (all forward dates are after the knowledge cutoff), and it is recorded once.
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import polars as pl

from ai_trader.ml.hourly import qlike_diff

HID = "H-NN10"
START = datetime(2026, 11, 13, tzinfo=UTC)
MIN_DAYS = 60
LOG_COLS = ("logged_at", "symbol", "forecast_for", "data_max_ts", "inputs_hash", "model_id")


def log_dir() -> Path:
    from ai_trader.core.config import load

    return load().root / "data" / "research" / "vol_ml_forward"


def read_log(d: Path | None = None) -> pl.DataFrame:
    d = d or log_dir()
    rows = []
    for p in sorted(d.glob("*.jsonl")):
        rows += [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]
    if not rows:
        return pl.DataFrame()
    recs = [{k: (None if r.get(k) is None else str(r.get(k))) for k in LOG_COLS}
            | {"sigma_next_hour_ml": r.get("sigma_next_hour_ml"), "sigma_next_hour_baseline": r.get("sigma_next_hour_baseline"),
               "flags": ";".join(r.get("flags") or [])} for r in rows]
    schema = {**dict.fromkeys(LOG_COLS, pl.Utf8), "sigma_next_hour_ml": pl.Float64, "sigma_next_hour_baseline": pl.Float64, "flags": pl.Utf8}
    df = pl.DataFrame(recs, schema=schema)
    return df.with_columns(*[pl.col(c).str.to_datetime(time_zone="UTC", strict=False) for c in ("logged_at", "forecast_for", "data_max_ts")])


def admissible(df: pl.DataFrame) -> pl.DataFrame:
    if df.height == 0:
        return df
    return (df.sort("logged_at").unique(["symbol", "forecast_for"], keep="first", maintain_order=True)
            .filter((pl.col("forecast_for") >= START) & (pl.col("flags") == "") & pl.col("sigma_next_hour_ml").is_not_null()
                    & pl.col("sigma_next_hour_baseline").is_not_null() & (pl.col("data_max_ts") <= pl.col("forecast_for") + pl.duration(seconds=60))))


def realised(symbols, as_of: datetime, lake=None) -> pl.DataFrame:
    from ai_trader.ml.hourly import symbol_hourly
    from ai_trader.research.frame import market_frame

    parts = []
    for s in symbols:
        f = market_frame(s, "1h", as_of, funding=False, oi=False, lake=lake)
        if f.height == 0:
            continue
        h = symbol_hourly(f.select("event_ts", "ret", "sigma_risk_next")).with_columns(pl.col("event_ts").cast(pl.Datetime("us", "UTC")))
        parts.append(h.filter((pl.col("event_ts") >= START) & (pl.col("event_ts") + pl.duration(hours=1) <= as_of)).with_columns(pl.lit(s).alias("symbol")))
    return pl.concat(parts) if parts else pl.DataFrame()


def paired_by_date(df: pl.DataFrame, a: str, b: str) -> dict:
    from ai_trader.research.evaluate import newey_west_t

    if df.height == 0:
        return {"diff": float("nan"), "t": float("nan"), "n_dates": 0, "n_obs": 0}
    d = qlike_diff(df["r2"].to_numpy(), df[a].to_numpy(), df[b].to_numpy())
    ok = np.isfinite(d)
    if ok.sum() == 0:
        return {"diff": float("nan"), "t": float("nan"), "n_dates": 0, "n_obs": 0}
    g = pl.DataFrame({"d": df["forecast_for"].dt.date().to_numpy()[ok], "x": d[ok]}).group_by("d").agg(pl.col("x").mean()).sort("d")
    m, t, nd = newey_west_t(g["x"].to_numpy())
    return {"diff": float(m), "t": float(t), "n_dates": int(nd), "n_obs": int(ok.sum()), "first": str(g["d"][0]), "last": str(g["d"][-1])}


def verdict(r: dict) -> str:
    if r["n_dates"] < MIN_DAYS:
        return "pending"
    if r["diff"] < 0 and r["t"] < -2:
        return "supports"
    if r["diff"] >= 0 or r["t"] > 2:
        return "contradicts"
    return "inconclusive"


def reconstruct(R: pl.DataFrame, model_path=None) -> pl.DataFrame:
    """Frozen-model forecasts for every realised hour (point-in-time features of each bar)."""
    from ai_trader.ml.frozen import load_frozen
    from ai_trader.ml.hourly import HOURLY_FEATURES

    m = load_frozen(model_path)
    q = R.filter(pl.all_horizontal([pl.col(c).is_not_null() & pl.col(c).is_finite() for c in (*HOURLY_FEATURES, "s2", "fac")]))
    if q.height == 0:
        return q
    X = q.select(HOURLY_FEATURES).to_numpy().astype(float)
    s2 = q["s2"].to_numpy().astype(float)
    return q.with_columns(pl.Series("F_ml", m.variance(X, s2)), (pl.col("s2") * pl.col("fac")).alias("F_base"), pl.col("event_ts").alias("forecast_for"))


def run(as_of: datetime | None = None, record: bool = False, lake=None, log: Path | None = None, model_path=None) -> dict:
    from ai_trader.core.time import now
    from ai_trader.research.trend_zoo_study import UNIVERSE

    as_of = as_of or now()
    lg = admissible(read_log(log))
    out: dict = {"hypothesis": HID, "as_of": as_of.isoformat(), "start": START.isoformat(), "min_days": MIN_DAYS, "log_lines_admissible": lg.height}
    R = realised(sorted(set(lg["symbol"].to_list())) if lg.height else UNIVERSE, as_of, lake) if as_of > START else pl.DataFrame()
    j = pl.DataFrame()
    if lg.height and R.height:
        j = lg.join(R.select("symbol", pl.col("event_ts").alias("forecast_for"), "r2"), on=["symbol", "forecast_for"], how="inner").filter(
            pl.col("r2").is_not_null()).with_columns((pl.col("sigma_next_hour_ml") ** 2).alias("F_ml"), (pl.col("sigma_next_hour_baseline") ** 2).alias("F_base"))
    out["logged"] = paired_by_date(j, "F_ml", "F_base")
    try:
        rc = reconstruct(R, model_path) if R.height else pl.DataFrame()
        out["reconstructed"] = paired_by_date(rc.filter(pl.col("r2").is_not_null()), "F_ml", "F_base") if rc.height else {"n_dates": 0}
        if lg.height and rc.height:
            k = lg.join(rc.select("symbol", "forecast_for", pl.col("F_ml").sqrt().alias("sig_rc")), on=["symbol", "forecast_for"], how="inner")
            out["log_vs_reconstructed_max_rel_diff"] = float((k["sigma_next_hour_ml"] / k["sig_rc"] - 1).abs().max()) if k.height else None
    except FileNotFoundError:
        out["reconstructed"] = {"n_dates": 0, "note": "frozen model missing"}
    out["forward_days"] = out["logged"]["n_dates"]
    out["verdict"] = verdict(out["logged"])
    out["status"] = "progress" if out["verdict"] == "pending" else "decidable"
    if record:
        if out["verdict"] == "pending":
            out["record"] = f"refused: {out['forward_days']} forward days < {MIN_DAYS}"
        else:
            from ai_trader.research.hypotheses import record as rec
            from ai_trader.research.hypotheses import results

            if any(r["segment"] == "clean_holdout" for r in results(HID)):
                out["record"] = "skipped: already recorded"
            else:
                rec(HID, "clean_holdout", out["verdict"], {"logged": out["logged"], "reconstructed": out.get("reconstructed")},
                    note="frozen hourly_vol_v1 vs robust_bc x hour-of-week, forward log; per-bar QLIKE diff, date means, NW t",
                    data="data/research/vol_ml_forward/*.jsonl; src/ai_trader/research/vol_ml_forward_study.py")
                out["record"] = "recorded"
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--record", action="store_true")
    ap.add_argument("--as-of", default="")
    a = ap.parse_args()
    as_of = datetime.fromisoformat(a.as_of.replace("Z", "+00:00")).astimezone(UTC) if a.as_of else None
    out = run(as_of, a.record)
    log_dir().mkdir(parents=True, exist_ok=True)
    (log_dir() / "progress.json").write_text(json.dumps(out, indent=1, default=str), encoding="utf-8")
    print(json.dumps(out, indent=1, default=str))


if __name__ == "__main__":
    main()


__all__ = ["MIN_DAYS", "START", "admissible", "paired_by_date", "read_log", "reconstruct", "run", "verdict"]
