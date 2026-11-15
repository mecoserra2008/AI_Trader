"""Point-in-time daily feature panel for the universe-L ML signal study (H-NN1..H-NN4, program 2026-11-12).

One row per symbol and UTC day D; decision time T = 00:00 UTC of D+1 (the close of the last 1h bar of D). Every
feature of row D is computed from bars with event_ts < T, funding / OI prints with known_ts <= T and 4h-regime rows
known by T + 60 s (decision_budget_study.symbol_daily, which this module extends). Targets are day D+1 quantities.

Feature list (fixed before any model was fitted; ML_FEATURES, 52 columns, in this order):
  the H-IC1 list (decision_budget_study.FEATURES32, 32 columns: vol-scaled returns 1/3/7/30d, log sigma, RV surprise
  and term, vol percentile, 7d semivariance ratio, 7d jump share, vol-of-vol, the v0-style regime statistics
  er / tstat / dvr / vr / ac1, funding level and 90d z, OI changes, next-day weekend / Monday, 30d range position,
  90d drawdown z, day range z, BTC's 1d z, two interactions), plus
  lrv_d, lrv_w, lrv_m   log-HAR regressors: log RV_D, mean log RV over 7 / 30 calendar days (>= 5 / >= 20 present),
                        exactly vol_panel.design('loghar') (the R-VOL-20 champion's inputs)
  lrv_q                 mean log RV over 90 calendar days (>= 60 present): long-run level
  neg_share_d           RS-_D / RV_D (daily downside semivariance share)
  rq_ratio_d            log(sqrt(RQ_D) / RV_D) (HAR-Q measurement-error proxy, scale free)
  jump_share_d          JV_D / RV_D (BV-based jump share of the day)
  pk_ratio_d, gk_ratio_d  log(Parkinson_D / RV_D), log(Garman-Klass_D / RV_D) (range vs return variance)
  abs_ret_1d_z          |ret_1d_z|
  dow_sin, dow_cos      day of week of D+1 (the forecast day) on the unit circle
  next_wk_hours         hours of the TradFi weekend (research.calendar) inside D+1 (0..24)
  next_is_mexpiry       D+1 is a Deribit monthly / quarterly expiry Friday
  v0_dir, v0_vol, v0_dir_score   regime labeller v0 (research.regimes.label) on 4h bars at the 20:00 bar of D:
                        direction state up/range/down = +1/0/-1, vol state low/normal/high = 0/1/2, the raw score
  mkt_lrv_d, mkt_lrv_w  crypto vol factor: cross-sectional mean of lrv_d / lrv_w over universe-L symbols present on D
                        (>= 5 symbols), the same day-D information set
  mkt_ret_1d_z          cross-sectional mean ret_1d_z on D
Targets: lrv_next = log RV_{D+1}, rv_next, y_dir = 1{ret_{D+1} > 0}, ret_next_z.
Segments (decision_panel convention): llm_in_sample if D + 2 days <= cutoff (label complete before it), clean_holdout
if D + 1 day >= cutoff.

Standardisation (Standardizer) uses training rows only; a column with zero training variance is left centred.
"""

from __future__ import annotations

import json
from concurrent.futures import ProcessPoolExecutor
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl

from ai_trader.research.decision_budget_study import FEATURES32

EXTRA = (
    "lrv_d", "lrv_w", "lrv_m", "lrv_q", "neg_share_d", "rq_ratio_d", "jump_share_d", "pk_ratio_d", "gk_ratio_d",
    "abs_ret_1d_z", "dow_sin", "dow_cos", "next_wk_hours", "next_is_mexpiry", "v0_dir", "v0_vol", "v0_dir_score",
    "mkt_lrv_d", "mkt_lrv_w", "mkt_ret_1d_z",
)
ML_FEATURES = (*FEATURES32, *EXTRA)
assert len(ML_FEATURES) == 52 and len(set(ML_FEATURES)) == 52

# concept groups (for ablation, grouped attributions and the 'what to call when' table)
GROUPS: dict[str, tuple[str, ...]] = {
    "vol_level": ("log_sigma", "lrv_d", "lrv_w", "lrv_m", "lrv_q", "vol_pct_1y"),
    "vol_shape": ("rv_surprise", "rv_term", "vol_of_vol_30d", "rq_ratio_d"),
    "jumps_semivar": ("semivar_7d", "jump_share_7d", "neg_share_d", "jump_share_d"),
    "range_measures": ("pk_ratio_d", "gk_ratio_d", "day_range_z"),
    "returns": ("ret_1d_z", "ret_3d_z", "ret_7d_z", "ret_30d_z", "abs_ret_1d_z", "x_ret7_volpct"),
    "trend_regime": ("er_7d", "er_30d", "tstat_7d", "tstat_30d", "dvr_7d", "dvr_30d", "vr_30d", "ac1_4h_30d",
                     "ac1_1d_30d", "v0_dir", "v0_vol", "v0_dir_score", "range_pos_30d", "dd_90d_z"),
    "funding_oi": ("funding_1d", "funding_z_90d", "x_ret1_fundz", "oi_chg_1d", "oi_chg_7d"),
    "calendar": ("next_is_weekend", "next_is_monday", "dow_sin", "dow_cos", "next_wk_hours", "next_is_mexpiry"),
    "market": ("btc_ret_1d_z", "mkt_lrv_d", "mkt_lrv_w", "mkt_ret_1d_z"),
}
assert sorted(sum(GROUPS.values(), ())) == sorted(ML_FEATURES)
FEATURE_GROUP = {f: g for g, fs in GROUPS.items() for f in fs}

# named state variables for the SAE / probe study (H-NN4)
STATE_VARS = {"vol_level": "lrv_m", "vol_of_vol": "vol_of_vol_30d", "recent_jump": "jump_share_7d", "weekend": "next_is_weekend",
              "trend_strength": "er_30d", "funding_z": "funding_z_90d", "drawdown": "dd_90d_z"}

TARGETS = ("lrv_next", "rv_next", "y_dir", "ret_next_z")
EPS = 1e-12


def _logpos(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    with np.errstate(all="ignore"):
        v = np.log(x)
    return np.where(np.isfinite(x) & (x > 0), v, np.nan)


def _trail_mean(x: np.ndarray, w: int, min_n: int) -> np.ndarray:
    s = pl.Series(np.asarray(x, dtype=float)).fill_nan(None)
    return s.rolling_mean(w, min_samples=min_n).to_numpy().astype(float)


def _monthly_expiry(d: date) -> bool:
    from ai_trader.features.seasonality import deribit_expiry_kind

    return deribit_expiry_kind(d) in ("monthly", "quarterly")


def symbol_features(sym: str, f1: pl.DataFrame, f4: pl.DataFrame, lake, as_of: datetime | None = None) -> pl.DataFrame:
    """Per-symbol rows (no cross-sectional columns yet). f1 / f4: market_frame 1h / 4h frames (as of `as_of`)."""
    from ai_trader.features.realized import daily_measures
    from ai_trader.research.calendar import add_calendar
    from ai_trader.research.decision_budget_study import symbol_daily
    from ai_trader.research.regimes import label

    d = symbol_daily(sym, f1, f4, lake, as_of)
    dm = daily_measures(f1.select("event_ts", "open", "high", "low", "close"), "1h").select(
        "date", "rs_neg", "rq", "bv", "pk", "gk", "jv", pl.col("rv").alias("_rv"))
    d = d.join(dm, on="date", how="left").sort("date")

    # v0 regime at the 20:00 4h bar of D (its close = T)
    lab = label(f4.sort("event_ts"), bar_hours=4.0)
    lab = lab.filter(pl.col("event_ts").dt.hour() == 20).select(
        pl.col("event_ts").dt.date().alias("date"),
        pl.when(pl.col("dir_state") == "up").then(1.0).when(pl.col("dir_state") == "down").then(-1.0)
        .when(pl.col("dir_state") == "range").then(0.0).otherwise(None).alias("v0_dir"),
        pl.when(pl.col("vol_state") == "low").then(0.0).when(pl.col("vol_state") == "normal").then(1.0)
        .when(pl.col("vol_state") == "high").then(2.0).otherwise(None).alias("v0_vol"),
        pl.col("dir_score").fill_nan(None).alias("v0_dir_score"))
    d = d.join(lab, on="date", how="left")

    rv = d["rv"].cast(pl.Float64).fill_null(np.nan).to_numpy()
    lrv = _logpos(rv)
    npf = lambda c: d[c].cast(pl.Float64).fill_null(np.nan).to_numpy()  # noqa: E731
    with np.errstate(all="ignore"):
        extra = {
            "lrv_d": lrv,
            "lrv_w": _trail_mean(lrv, 7, 5),
            "lrv_m": _trail_mean(lrv, 30, 20),
            "lrv_q": _trail_mean(lrv, 90, 60),
            "neg_share_d": npf("rs_neg") / rv,
            "rq_ratio_d": _logpos(np.sqrt(npf("rq")) / rv),
            "jump_share_d": npf("jv") / rv,
            "pk_ratio_d": _logpos(npf("pk") / rv),
            "gk_ratio_d": _logpos(npf("gk") / rv),
            "abs_ret_1d_z": np.abs(npf("ret_1d_z")),
        }
    nxt = d.select((pl.col("date") + pl.duration(days=1)).alias("n"))["n"]
    wd = nxt.dt.weekday().to_numpy().astype(float)  # 1 = Monday .. 7 = Sunday
    extra["dow_sin"] = np.sin(2 * np.pi * (wd - 1) / 7)
    extra["dow_cos"] = np.cos(2 * np.pi * (wd - 1) / 7)
    extra["next_is_mexpiry"] = np.array([float(_monthly_expiry(x)) for x in nxt.to_list()])
    # TradFi-weekend hours inside D+1 = [T, T + 24h) (pure calendar function of timestamps)
    lo, hi = d["T"].min(), d["T"].max()
    hh = pl.DataFrame({"event_ts": pl.datetime_range(lo, hi + timedelta(hours=23), "1h", eager=True)})
    wk = (add_calendar(hh).with_columns(pl.col("event_ts").dt.date().alias("date")).group_by("date")
          .agg(pl.col("tradfi_weekend").cast(pl.Float64).sum().alias("next_wk_hours")))
    d = d.with_columns(nxt.alias("_next")).join(wk.rename({"date": "_next"}), on="_next", how="left")

    rv_next = np.r_[rv[1:], np.nan]
    out = d.with_columns(
        *[pl.Series(k, np.where(np.isfinite(v), v, np.nan)) for k, v in extra.items()],
        pl.Series("rv_next", rv_next),
        pl.Series("lrv_next", _logpos(rv_next)),
    ).drop("_next", "rs_neg", "rq", "bv", "pk", "gk", "jv", "_rv")
    return out.with_columns(pl.col(pl.Float64).fill_nan(None))


def add_cross_section(daily: pl.DataFrame, cut: date) -> pl.DataFrame:
    """Market-wide columns (same-day information), interactions, targets, segment (decision_budget conventions)."""
    from ai_trader.research.decision_budget_study import decision_panel

    p = decision_panel(daily, cut)  # btc_ret_1d_z, x_ret7_volpct, x_ret1_fundz, y, label_end, segment
    mk = (p.group_by("date").agg(pl.col("lrv_d").mean().alias("mkt_lrv_d"), pl.col("lrv_w").mean().alias("mkt_lrv_w"),
                                 pl.col("ret_1d_z").mean().alias("mkt_ret_1d_z"), pl.col("lrv_d").count().alias("_n")))
    mk = mk.with_columns(*[pl.when(pl.col("_n") >= 5).then(pl.col(c)).otherwise(None).alias(c) for c in ("mkt_lrv_d", "mkt_lrv_w", "mkt_ret_1d_z")]).drop("_n")
    p = p.join(mk, on="date", how="left").with_columns(
        pl.when(pl.col("ret_next").is_null()).then(None).otherwise(pl.col("y")).alias("y_dir"))
    return p.sort("T", "symbol")


def _build_symbol(sym: str, as_of_iso: str) -> pl.DataFrame | None:
    from ai_trader.research.frame import market_frame
    from ai_trader.storage.lake import Lake

    as_of = datetime.fromisoformat(as_of_iso)
    lake = Lake()
    f1 = market_frame(sym, "1h", as_of, funding=False, oi=False, lake=lake)
    f4 = market_frame(sym, "4h", as_of, funding=False, oi=False, lake=lake)
    if f1.height < 24 * 120 or f4.height < 6 * 120:
        return None
    return symbol_features(sym, f1, f4, lake, as_of)


def build_panel(as_of: datetime, symbols, cache: Path | None = None, workers: int = 3) -> pl.DataFrame:
    """Full point-in-time panel (all rows, incl. incomplete ones). Cached to `cache` (parquet) when given."""
    from ai_trader.research.evaluate import knowledge_cutoff

    if cache is not None and Path(cache).is_file():
        return pl.read_parquet(cache)
    syms = list(symbols)
    with ProcessPoolExecutor(max_workers=max(1, min(workers, 4))) as ex:
        res = list(ex.map(_build_symbol, syms, [as_of.isoformat()] * len(syms)))
    daily = pl.concat([r for r in res if r is not None], how="vertical_relaxed")
    p = add_cross_section(daily, knowledge_cutoff().date())
    if cache is not None:
        Path(cache).parent.mkdir(parents=True, exist_ok=True)
        p.write_parquet(cache)
        Path(str(cache) + ".json").write_text(json.dumps({"as_of": as_of.isoformat(), "symbols": syms, "rows": p.height}), encoding="utf-8")
    return p


def complete_rows(panel: pl.DataFrame, features=ML_FEATURES, targets=("lrv_next", "y_dir")) -> pl.DataFrame:
    """Rows with every feature and target finite and a segment (identical sample for every model)."""
    cols = [*features, *targets]
    return panel.filter(pl.all_horizontal([pl.col(c).is_not_null() & pl.col(c).is_finite() for c in cols])
                        & pl.col("segment").is_not_null() & (pl.col("rv_next") > 0)).sort("T", "symbol")


class Standardizer:
    """Column standardisation with training statistics only."""

    def __init__(self) -> None:
        self.mu: np.ndarray | None = None
        self.sd: np.ndarray | None = None

    def fit(self, X: np.ndarray) -> Standardizer:
        X = np.asarray(X, dtype=float)
        self.mu = X.mean(axis=0)
        sd = X.std(axis=0)
        self.sd = np.where(sd > 0, sd, 1.0)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        assert self.mu is not None and self.sd is not None, "fit first"
        return (np.asarray(X, dtype=float) - self.mu) / self.sd

    def to_dict(self) -> dict:
        return {"mu": self.mu.tolist(), "sd": self.sd.tolist()}


__all__ = ["EXTRA", "FEATURE_GROUP", "GROUPS", "ML_FEATURES", "STATE_VARS", "Standardizer", "TARGETS", "add_cross_section",
           "build_panel", "complete_rows", "symbol_features"]
