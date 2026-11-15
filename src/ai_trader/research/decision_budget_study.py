"""Decision budget (program 2026-11-10, agent A7): how fast information saturates with the number of inputs
(H-IC1), with the length of the training window (H-IC2), and how fast signal edges decay with decision latency
(H-IC3). The LLM layer itself is never backtested (CLAUDE.md principle 1); H-IC4 is its forward-only A/B and is
NOT run here. This module answers the deterministic part only.

Universe L = trend_zoo_study.UNIVERSE (23 crypto perps). Everything is computed on the FULL history and split by
segment afterwards (RESEARCH_PROTOCOL 3b): llm_in_sample if the label is complete before the knowledge cutoff,
clean_holdout if the decision is on/after it.

Decision rows (H-IC1, H-IC2)
  One row per symbol and UTC day D. Decision time T = 00:00 UTC of D+1 = the close of the last 1h bar of D (repo
  convention: executed at that close; bars are published with known_ts ~ close + 2 s, a feature bar is accepted if
  known_ts <= T + 60 s). Every input bar has event_ts < T; funding / OI prints count only if known_ts <= T.
  sigma_d = sigma_risk_next of the 1h frame (robust_bc, H-V1b) at the 23:00 bar of D x sqrt(24).
  Daily measures: features.realized.daily_measures on 1h bars (complete days only).
  Target y = 1{ret(D+1) > 0}, ret(D+1) = log close(00:00 D+2) / close(00:00 D+1): the sign of the vol-scaled
  next-24h return. Label interval [T, T + 24h].

THE FIXED 32-FEATURE LIST (ordered simple -> complex; model k uses features 1..k; fixed in the diary entry
"H-IC1 feature list fixed before testing" before any code was run):
   1 ret_1d_z        ret_D / sigma_d
   2 ret_3d_z        sum ret D-2..D / (sigma_d sqrt 3)
   3 ret_7d_z        sum ret D-6..D / (sigma_d sqrt 7)
   4 ret_30d_z       sum ret D-29..D / (sigma_d sqrt 30)
   5 log_sigma       log sigma_d
   6 rv_surprise     log(RV_D / mean RV D-29..D)
   7 rv_term         log(mean RV 7d / mean RV 30d)
   8 vol_pct_1y      features.regime_features (4h)
   9 semivar_7d      log(sum RS- / sum RS+) over 7d (1h)
  10 jump_share_7d   sum JV / sum RV over 7d
  11 vol_of_vol_30d  features.regime_features
  12 er_7d   13 er_30d   14 tstat_7d   15 tstat_30d   16 dvr_7d   17 dvr_30d   18 vr_30d   19 ac1_4h_30d
  20 ac1_1d_30d      (features.regime_features on 4h bars, known at 00:00 D+1)
  21 funding_1d      mean funding rate (bps) of prints known in (D 00:00, D+1 00:00], else the last known
  22 funding_z_90d   (funding_1d - mean 90d) / sd 90d (>= 60 valid days)
  23 oi_chg_1d       log OI(last known <= T) / OI(last known <= T - 24h), Bybit 4h OI (stale > 1 day = missing)
  24 oi_chg_7d       same over 7 days
  25 next_is_weekend 1{D+1 is Sat/Sun}
  26 next_is_monday  1{D+1 is Monday}
  27 range_pos_30d   (close_D - min low 30d) / (max high 30d - min low 30d) - 0.5
  28 dd_90d_z        log(close_D / max close 90d) / (sigma_d sqrt 90)
  29 day_range_z     log(high_D / low_D) / sigma_d
  30 btc_ret_1d_z    BTCUSDT's ret_1d_z on D (market factor)
  31 ret_7d_z x (vol_pct_1y - 0.5)
  32 ret_1d_z x funding_z_90d
  Sample = rows with all 32 features and the target finite (identical rows for every k and every window).

Walk-forward (H-IC1, H-IC2)
  Monthly refits. Test rows of month m: T in [m, m+1). Training rows: label_end = T + 24h <= m - 1 day (purge 24h
  + embargo 1 day: purged_train_mask); rolling windows keep label_end > m - 1 day - window. First test month = the
  first with >= 365 distinct training dates (expanding). Ridge logistic by Newton (regime_switch_study.fit_logit,
  lambda = 1 on standardised coefficients, intercept free), standardised with the training mean / sd.
  Brier skill = 1 - sum (p - y)^2 / sum (b - y)^2, b = the base rate of the training sample of that refit, pooled per
  segment. 95% CI: moving-block bootstrap over decision dates, blocks of 10 days, 1000 draws, seed 0 (all k jointly).
  H-IC1 verdict per segment: inconclusive if every k's CI contains 0; else supports iff argmax k <= 8 and
  skill(32) - max_{k<=8} skill <= 0; contradicts otherwise. pooled: supports iff both segments support, contradicts
  if any contradicts, else inconclusive. p = P*(Delta >= 0).

H-IC2
  k = the llm_in_sample argmax of H-IC1. Windows 90 / 180 / 365 / 730 days (of label ends) and expanding, for the
  logistic model (Brier loss) and a pooled panel log-HAR (1, log RV_D, log mean RV 5d, log mean RV 22d, common
  intercept; forecast exp(xb + s2/2); QLIKE on RV_{D+1}), same purged monthly refits; rows common to all windows.
  Paired loss difference per date (mean over symbols, window - expanding), NW t over dates. A leg supports in a
  segment iff some shorter window has mean diff < 0 with t < -2 AND the mean loss is not strictly monotone in window
  length; contradicts if strictly monotone; else inconclusive. Segment: supports iff both legs support, contradicts if
  either contradicts, else inconclusive. pooled = the registry criterion: for each model a t < -2 win in at least one
  segment and no strictly monotone segment (contradicts if a model is monotone in llm_in_sample). p = max over models
  of min(1, 4 min_w Phi(t_w)).

H-IC3 (latency)
  shock_regime (REGISTRY, as configured) on 1h bars; the 4h trend-zoo equal-weight ensemble (trend_zoo.ensemble)
  computed on 4h bars and APPLIED TO 1h BARS: on the 1h frame, the position held over the bar after close c (row t,
  P&L = L_t x fwd_ret_t) is the latest source signal whose availability time (the source bar's close: 4h event_ts
  + 4h, or 1h event_ts + 1h) is <= c - d hours (lag_onto, backward as-of). d = 0 on the 4h ensemble therefore
  reproduces the 4h strategy on the 1h return path (sized on the 1h sigma at entry); d = 1..8 h delays it in 1h
  steps. Shock exits are delayed along with entries (the whole signal path shifts). P&L: evaluate.strategy_pnl
  (risk units, 7.5 bps per side), daily net P&L by UTC date of event_ts; the ensemble's dates start after its
  warm-up (trend_zoo.lookback_bars). edge(d) = mean over dates of the cross-symbol mean daily P&L.
  R(d) = edge(d) / edge(0); test statistic g(d) = edge(d) - 0.5 edge(0), 10-day block bootstrap CI (equivalent to
  the ratio CI excluding 0.5 when edge(0) > 0). Half-life = linear interpolation of the first crossing of 0.5.
  Trend leg: undefined -> inconclusive if edge(0) <= 0; supports iff R(4) > 0.5 and g(4) CI lower > 0; contradicts
  if R(4) < 0.5; else inconclusive. Shock leg: supports iff R(2) < 0.5 and g(2) CI upper < 0; contradicts if
  R(2) > 0.5. Segment: contradicts if a leg contradicts, supports if both support, else inconclusive. pooled = the
  llm_in_sample verdict (the registry criterion). p = max(P*(g4 <= 0), P*(g2 >= 0)).

Holm over H-IC1..3 within each segment; a 'supports' with Holm-adjusted p > 0.05 is downgraded to inconclusive.

    uv run python -m ai_trader.research.decision_budget_study --cache <prefix> [--workers 4] [--out f.json] [--md f.md] [--record]
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl
from numpy.lib.stride_tricks import sliding_window_view
from scipy.stats import norm

from ai_trader.core.time import now
from ai_trader.features.realized import daily_measures
from ai_trader.features.regime_features import regime_features
from ai_trader.features.vol_models_x import qlike
from ai_trader.research.evaluate import knowledge_cutoff, newey_west_t, strategy_pnl
from ai_trader.research.frame import market_frame
from ai_trader.research.regime_switch_study import fit_logit
from ai_trader.research.trend_zoo_study import UNIVERSE

FEATURES32 = (
    "ret_1d_z", "ret_3d_z", "ret_7d_z", "ret_30d_z",
    "log_sigma", "rv_surprise", "rv_term", "vol_pct_1y", "semivar_7d", "jump_share_7d", "vol_of_vol_30d",
    "er_7d", "er_30d", "tstat_7d", "tstat_30d", "dvr_7d", "dvr_30d", "vr_30d", "ac1_4h_30d", "ac1_1d_30d",
    "funding_1d", "funding_z_90d", "oi_chg_1d", "oi_chg_7d",
    "next_is_weekend", "next_is_monday",
    "range_pos_30d", "dd_90d_z", "day_range_z", "btc_ret_1d_z",
    "x_ret7_volpct", "x_ret1_fundz",
)
assert len(FEATURES32) == 32
REGIME_COLS = ("vol_pct_1y", "vol_of_vol_30d", "er_7d", "er_30d", "tstat_7d", "tstat_30d", "dvr_7d", "dvr_30d",
               "vr_30d", "ac1_4h_30d", "ac1_1d_30d")
HAR_COLS = ("lrv_d", "lrv_w", "lrv_m")
KS = (1, 2, 4, 8, 16, 32)
WINDOWS = (90, 180, 365, 730, None)  # None = expanding
LAGS = (0, 1, 2, 4, 8)
SEGS = ("llm_in_sample", "clean_holdout")
HORIZON = timedelta(hours=24)
EMBARGO = timedelta(days=1)
MIN_TRAIN_DAYS = 365
MIN_TRAIN_ROWS = 200
RIDGE = 1.0
BLOCK = 10
N_BOOT = 1000
FEATURE_LATENCY = timedelta(seconds=60)
US = 1_000_000  # microseconds per second


# ------------------------------------------------------------------ reusable helpers (unit-tested)

def purged_train_mask(label_end: np.ndarray, test_start: np.datetime64, embargo: timedelta = EMBARGO,
                      window_days: int | None = None) -> np.ndarray:
    """Training rows for a test block starting at test_start: label_end <= test_start - embargo (purge of every label
    that overlaps the test block, plus an embargo); a rolling window keeps label_end > test_start - embargo - window."""
    cut = test_start - np.timedelta64(int(embargo.total_seconds() * US), "us")
    m = label_end <= cut
    if window_days is not None:
        m &= label_end > cut - np.timedelta64(int(window_days) * 86400 * US, "us")
    return m


def test_mask(decision_ts: np.ndarray, test_start: np.datetime64, test_end: np.datetime64) -> np.ndarray:
    return (decision_ts >= test_start) & (decision_ts < test_end)


def lag_onto(target_avail_us: np.ndarray, source_avail_us: np.ndarray, source_signal: np.ndarray, lag_hours: float) -> np.ndarray:
    """For each target row (a decision at time target_avail), the latest source signal whose availability time is
    <= target_avail - lag (backward as-of); 0 before the first one. Both time arrays in epoch microseconds, sorted."""
    src = np.asarray(source_avail_us, dtype=np.int64)
    tgt = np.asarray(target_avail_us, dtype=np.int64) - int(round(lag_hours * 3600 * US))
    idx = np.searchsorted(src, tgt, side="right") - 1
    sig = np.asarray(source_signal, dtype=float)
    return np.where(idx >= 0, sig[np.clip(idx, 0, None)], 0.0)


def block_bootstrap_idx(n: int, block: int = BLOCK, n_boot: int = N_BOOT, seed: int = 0) -> np.ndarray:
    """(n_boot, n) indices of a moving-block bootstrap (contiguous blocks of `block`, truncated to n)."""
    rng = np.random.default_rng(seed)
    b = max(1, min(block, n))
    nb = int(np.ceil(n / b))
    starts = rng.integers(0, n - b + 1, size=(n_boot, nb))
    return (starts[:, :, None] + np.arange(b)[None, None, :]).reshape(n_boot, -1)[:, :n]


def brier_skill(p: np.ndarray, y: np.ndarray, base: np.ndarray) -> float:
    ref = np.sum((base - y) ** 2)
    return float(1.0 - np.sum((p - y) ** 2) / ref) if ref > 0 else float("nan")


def holm(ps: dict[str, float]) -> dict[str, float]:
    """Holm step-down adjusted p-values."""
    items = sorted(ps.items(), key=lambda kv: kv[1])
    m = len(items)
    out, run = {}, 0.0
    for i, (k, p) in enumerate(items):
        run = max(run, min(1.0, (m - i) * p))
        out[k] = run
    return out


def _trail(x: np.ndarray, n: int, fn) -> np.ndarray:
    """fn over x[d-n+1..d]; NaN if the window is short or holds a NaN."""
    x = np.asarray(x, dtype=float)
    out = np.full(len(x), np.nan)
    if len(x) >= n:
        w = sliding_window_view(x, n)
        with np.errstate(all="ignore"):
            v = fn(w, axis=1)
        out[n - 1:] = np.where(np.isfinite(w).all(axis=1), v, np.nan)
    return out


def _trail_z(x: np.ndarray, n: int, min_valid: int) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    out = np.full(len(x), np.nan)
    if len(x) >= n:
        w = sliding_window_view(x, n)
        cnt = np.isfinite(w).sum(axis=1)
        with np.errstate(all="ignore"):
            mu, sd = np.nanmean(w, axis=1), np.nanstd(w, axis=1, ddof=1)
            z = (x[n - 1:] - mu) / sd
        out[n - 1:] = np.where((cnt >= min_valid) & (sd > 0), z, np.nan)
    return out


def _log(x: np.ndarray) -> np.ndarray:
    with np.errstate(all="ignore"):
        v = np.log(np.asarray(x, dtype=float))
    return np.where(np.isfinite(v), v, np.nan)


def _epoch_us(s: pl.Series) -> np.ndarray:
    return s.dt.epoch("us").to_numpy().astype(np.int64)


# ------------------------------------------------------------------ per-symbol build

def symbol_daily(sym: str, f1: pl.DataFrame, f4: pl.DataFrame, lake, as_of: datetime | None = None) -> pl.DataFrame:
    """Daily decision rows (features, target, HAR inputs) for one symbol."""
    bars = f1.select("event_ts", "open", "high", "low", "close")
    dm = daily_measures(bars, "1h")
    hl = (f1.with_columns(pl.col("event_ts").dt.date().alias("date")).group_by("date")
          .agg(pl.col("high").max().alias("hi"), pl.col("low").min().alias("lo"), pl.len().alias("nb")).filter(pl.col("nb") == 24))
    sg = f1.filter(pl.col("event_ts").dt.hour() == 23).select(pl.col("event_ts").dt.date().alias("date"),
                                                                 (pl.col("sigma_risk_next") * np.sqrt(24.0)).alias("sigma_d"))
    cal = pl.DataFrame({"date": pl.date_range(dm["date"].min(), dm["date"].max(), "1d", eager=True)})
    d = cal.join(dm.select("date", "ret", "rv", "rs_neg", "rs_pos", "jv", "close"), on="date", how="left")
    d = d.join(hl.select("date", "hi", "lo"), on="date", how="left").join(sg, on="date", how="left").sort("date")
    d = d.with_columns((pl.col("date").cast(pl.Datetime("us", "UTC")) + pl.duration(days=1)).alias("T"))

    # regime features (4h), point-in-time checked
    rf = regime_features(f4).select("date", "known_ts", *REGIME_COLS)
    rf = rf.with_columns((pl.col("date").cast(pl.Datetime("us", "UTC")) + pl.duration(days=1)).alias("_T"))
    bad = rf.filter(pl.col("known_ts").cast(pl.Datetime("us", "UTC")) > pl.col("_T") + pl.duration(seconds=FEATURE_LATENCY.seconds))
    rf = rf.join(bad.select("date"), on="date", how="anti").drop("known_ts", "_T")
    d = d.join(rf, on="date", how="left")

    # funding: prints known in (D 00:00, D+1 00:00] -> day D; carried forward (last known)
    fu = lake.read_asof("funding", as_of or now(), symbols=[sym], filters={"venue": "bybit"}) if lake is not None else pl.DataFrame()
    if fu.height:
        fd = (fu.with_columns((pl.col("known_ts") - pl.duration(microseconds=1)).dt.date().alias("date"))
              .group_by("date").agg((pl.col("rate").mean() * 1e4).alias("funding_1d")))
        d = d.join(fd, on="date", how="left").with_columns(pl.col("funding_1d").forward_fill())
    else:
        d = d.with_columns(pl.lit(None, dtype=pl.Float64).alias("funding_1d"))

    # open interest (4h snapshots), last known strictly before T, stale > 1 day -> missing
    oi = lake.read_asof("open_interest", as_of or now(), symbols=[sym], filters={"venue": "bybit", "interval": "4h"}) if lake is not None else pl.DataFrame()
    oi_cols = {}
    if oi.height:
        oi = oi.select(pl.col("known_ts").cast(pl.Datetime("us", "UTC")).alias("k"), pl.col("open_interest").cast(pl.Float64)).sort("k")
        for name, back in (("oi_T", 0), ("oi_T1", 1), ("oi_T7", 7)):
            q = d.select((pl.col("T") - pl.duration(days=back, microseconds=1)).alias("q")).with_row_index("i").sort("q")
            j = q.join_asof(oi, left_on="q", right_on="k", strategy="backward", tolerance="1d").sort("i")
            oi_cols[name] = j["open_interest"].to_numpy().astype(float)
    n = d.height
    for name in ("oi_T", "oi_T1", "oi_T7"):
        oi_cols.setdefault(name, np.full(n, np.nan))

    ret = d["ret"].to_numpy().astype(float)
    rv = d["rv"].to_numpy().astype(float)
    sig = d["sigma_d"].to_numpy().astype(float)
    close = d["close"].to_numpy().astype(float)
    hi, lo = d["hi"].to_numpy().astype(float), d["lo"].to_numpy().astype(float)
    with np.errstate(all="ignore"):
        sig = np.where(sig > 0, sig, np.nan)
        f: dict[str, np.ndarray] = {
            "ret_1d_z": ret / sig,
            "ret_3d_z": _trail(ret, 3, np.sum) / (sig * np.sqrt(3)),
            "ret_7d_z": _trail(ret, 7, np.sum) / (sig * np.sqrt(7)),
            "ret_30d_z": _trail(ret, 30, np.sum) / (sig * np.sqrt(30)),
            "log_sigma": _log(sig),
            "rv_surprise": _log(rv / _trail(rv, 30, np.mean)),
            "rv_term": _log(_trail(rv, 7, np.mean) / _trail(rv, 30, np.mean)),
            "semivar_7d": _log(_trail(d["rs_neg"].to_numpy().astype(float), 7, np.sum) / _trail(d["rs_pos"].to_numpy().astype(float), 7, np.sum)),
            "jump_share_7d": _trail(d["jv"].to_numpy().astype(float), 7, np.sum) / _trail(rv, 7, np.sum),
            "funding_z_90d": _trail_z(d["funding_1d"].to_numpy().astype(float), 90, 60),
            "oi_chg_1d": _log(oi_cols["oi_T"] / oi_cols["oi_T1"]),
            "oi_chg_7d": _log(oi_cols["oi_T"] / oi_cols["oi_T7"]),
            "range_pos_30d": (close - _trail(lo, 30, np.min)) / (_trail(hi, 30, np.max) - _trail(lo, 30, np.min)) - 0.5,
            "dd_90d_z": _log(close / _trail(close, 90, np.max)) / (sig * np.sqrt(90)),
            "day_range_z": _log(hi / lo) / sig,
        }
    wd_next = d.select((pl.col("date") + pl.duration(days=1)).dt.weekday())["date"].to_numpy()
    f["next_is_weekend"] = (wd_next >= 6).astype(float)
    f["next_is_monday"] = (wd_next == 1).astype(float)
    ret_next = np.r_[ret[1:], np.nan]
    rv_next = np.r_[rv[1:], np.nan]
    har = {"lrv_d": _log(rv), "lrv_w": _log(_trail(rv, 5, np.mean)), "lrv_m": _log(_trail(rv, 22, np.mean))}
    out = d.select("date", "T", "funding_1d", *REGIME_COLS).with_columns(
        pl.lit(sym).alias("symbol"),
        *[pl.Series(k, np.where(np.isfinite(v), v, np.nan)) for k, v in f.items()],
        *[pl.Series(k, v) for k, v in har.items()],
        pl.Series("rv", rv), pl.Series("rv_next", rv_next), pl.Series("ret_next", ret_next),
        pl.Series("ret_next_z", ret_next / sig),
    )
    return out.with_columns(pl.col(pl.Float64).fill_nan(None))


def symbol_latency(sym: str, f1: pl.DataFrame, f4: pl.DataFrame) -> tuple[pl.DataFrame, dict]:
    """Daily net P&L of shock_regime (1h) and the 4h trend ensemble (applied to 1h bars) for each lag d."""
    from ai_trader.strategies import REGISTRY
    from ai_trader.strategies.trend_zoo import ensemble, lookback_bars

    t1 = _epoch_us(f1["event_ts"])
    c1 = t1 + 3600 * US                      # close of each 1h bar = when its position starts
    shock = REGISTRY["shock_regime"]()(f1).to_numpy()
    ens = ensemble(f4).to_numpy()
    a4 = _epoch_us(f4["event_ts"]) + 4 * 3600 * US
    lb = lookback_bars(6)
    ens_start = f4["event_ts"][lb].date() + timedelta(days=1) if f4.height > lb else None
    first = f1["event_ts"][0].date() + timedelta(days=1)
    last = f1["event_ts"][-1].date()  # incomplete last date dropped
    dates = f1["event_ts"].dt.date()
    rows = []
    for name, src_t, src_s, start in (("shock_regime", c1, shock, first), ("trend_ens_4h", a4, ens, ens_start)):
        if start is None:
            continue
        for d in LAGS:
            s = lag_onto(c1, src_t, src_s, d)
            p = strategy_pnl(f1, pl.Series(s))
            daily = (p.with_columns(dates.alias("date")).group_by("date")
                     .agg(pl.col("pnl").sum(), pl.col("active").mean().alias("exposure"))
                     .filter((pl.col("date") >= start) & (pl.col("date") < last)))
            rows.append(daily.with_columns(pl.lit(sym).alias("symbol"), pl.lit(name).alias("strategy"), pl.lit(d).alias("lag")))
    # native 4h ensemble (reference for d = 0)
    info = {"symbol": sym, "n_shocks": int(np.sum((np.abs(shock) > 0) & (np.r_[0.0, shock[:-1]] != shock) & (shock != 0)))}
    if ens_start is not None:
        p4 = strategy_pnl(f4, pl.Series(ens))
        d4 = (p4.with_columns(f4["event_ts"].dt.date().alias("date")).group_by("date").agg(pl.col("pnl").sum(), pl.col("active").mean().alias("exposure"))
              .filter((pl.col("date") >= ens_start) & (pl.col("date") < last)))
        rows.append(d4.with_columns(pl.lit(sym).alias("symbol"), pl.lit("trend_ens_4h_native").alias("strategy"), pl.lit(0).alias("lag")))
    out = pl.concat(rows, how="vertical_relaxed").select("symbol", "strategy", "lag", "date", "pnl", "exposure") if rows else pl.DataFrame()
    return out, info


def build_symbol(sym: str, as_of_iso: str) -> dict:
    from ai_trader.storage.lake import Lake

    as_of = datetime.fromisoformat(as_of_iso)
    lake = Lake()
    f1 = market_frame(sym, "1h", as_of, funding=False, oi=False, lake=lake)
    f4 = market_frame(sym, "4h", as_of, funding=False, oi=False, lake=lake)
    if f1.height < 24 * 120 or f4.height < 6 * 120:
        return {"symbol": sym, "daily": None, "pnl": None, "info": {"symbol": sym, "skipped": True}}
    daily = symbol_daily(sym, f1, f4, lake, as_of)
    pnl, info = symbol_latency(sym, f1, f4)
    info.update({"bars_1h": f1.height, "first_1h": str(f1["event_ts"].min()), "oi_rows": int(daily["oi_chg_1d"].is_not_null().sum())})
    return {"symbol": sym, "daily": daily, "pnl": pnl, "info": info}


def build(as_of: datetime, symbols=UNIVERSE, workers: int = 4, cache: str = "") -> tuple[pl.DataFrame, pl.DataFrame, list[dict]]:
    if cache and Path(cache + ".daily.parquet").is_file():
        info = json.loads(Path(cache + ".info.json").read_text(encoding="utf-8"))
        return pl.read_parquet(cache + ".daily.parquet"), pl.read_parquet(cache + ".pnl.parquet"), info
    with ProcessPoolExecutor(max_workers=min(4, workers)) as ex:
        res = list(ex.map(build_symbol, list(symbols), [as_of.isoformat()] * len(symbols)))
    daily = pl.concat([r["daily"] for r in res if r["daily"] is not None], how="vertical_relaxed")
    pnl = pl.concat([r["pnl"] for r in res if r["pnl"] is not None and r["pnl"].height], how="vertical_relaxed")
    info = [r["info"] for r in res]
    if cache:
        Path(cache).parent.mkdir(parents=True, exist_ok=True)
        daily.write_parquet(cache + ".daily.parquet")
        pnl.write_parquet(cache + ".pnl.parquet")
        Path(cache + ".info.json").write_text(json.dumps({"as_of": as_of.isoformat(), "symbols": info}, default=str), encoding="utf-8")
        info = {"as_of": as_of.isoformat(), "symbols": info}
    else:
        info = {"as_of": as_of.isoformat(), "symbols": info}
    return daily, pnl, info


# ------------------------------------------------------------------ panel

def decision_panel(daily: pl.DataFrame, cut: date) -> pl.DataFrame:
    btc = daily.filter(pl.col("symbol") == "BTCUSDT").select("date", pl.col("ret_1d_z").alias("btc_ret_1d_z"))
    p = daily.join(btc, on="date", how="left").with_columns(
        (pl.col("ret_7d_z") * (pl.col("vol_pct_1y") - 0.5)).alias("x_ret7_volpct"),
        (pl.col("ret_1d_z") * pl.col("funding_z_90d")).alias("x_ret1_fundz"),
        (pl.col("ret_next") > 0).cast(pl.Float64).alias("y"),
        (pl.col("T") + pl.duration(hours=24)).alias("label_end"),
    )
    seg = (pl.when(pl.col("date") + pl.duration(days=2) <= cut).then(pl.lit("llm_in_sample"))
           .when(pl.col("date") + pl.duration(days=1) >= cut).then(pl.lit("clean_holdout")).otherwise(None))
    return p.with_columns(seg.alias("segment")).sort("T", "symbol")


def _months(t: np.ndarray) -> list[np.datetime64]:
    lo = t.min().astype("datetime64[M]")
    hi = t.max().astype("datetime64[M]")
    return [np.datetime64(m, "us") for m in np.arange(lo, hi + np.timedelta64(1, "M"))]


def _first_test_month(T: np.ndarray, label_end: np.ndarray, min_days: int = MIN_TRAIN_DAYS) -> np.datetime64:
    for m in _months(T):
        tr = purged_train_mask(label_end, m)
        if len(np.unique(T[tr].astype("datetime64[D]"))) >= min_days:
            return m
    return _months(T)[-1] + np.timedelta64(1, "D")


def _sig(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -35, 35)))


def wf_logit(X: np.ndarray, y: np.ndarray, T: np.ndarray, label_end: np.ndarray, ks, window_days: int | None, m0: np.datetime64) -> tuple[dict, np.ndarray]:
    """Monthly purged walk-forward ridge logistic for each k (first k columns). Returns ({k: p}, base rate)."""
    preds = {k: np.full(len(y), np.nan) for k in ks}
    base = np.full(len(y), np.nan)
    for m in _months(T):
        if m < m0:
            continue
        nxt = np.datetime64((m.astype("datetime64[M]") + np.timedelta64(1, "M")), "us")
        te = test_mask(T, m, nxt)
        if not te.any():
            continue
        tr = purged_train_mask(label_end, m, window_days=window_days)
        if tr.sum() < MIN_TRAIN_ROWS:
            continue
        mu, sd = X[tr].mean(0), X[tr].std(0)
        sd = np.where(sd > 0, sd, 1.0)
        Ztr, Zte = (X[tr] - mu) / sd, (X[te] - mu) / sd
        base[te] = y[tr].mean()
        for k in ks:
            b, _ = fit_logit(Ztr[:, :k], y[tr], ridge=RIDGE)
            preds[k][te] = _sig(b[0] + Zte[:, :k] @ b[1:])
    return preds, base


def wf_har(X: np.ndarray, ly: np.ndarray, T: np.ndarray, label_end: np.ndarray, window_days: int | None, m0: np.datetime64) -> np.ndarray:
    """Monthly purged walk-forward pooled log-HAR; forecast of RV_{D+1} = exp(xb + s2/2)."""
    Z = np.c_[np.ones(len(ly)), X]
    out = np.full(len(ly), np.nan)
    for m in _months(T):
        if m < m0:
            continue
        nxt = np.datetime64((m.astype("datetime64[M]") + np.timedelta64(1, "M")), "us")
        te = test_mask(T, m, nxt)
        if not te.any():
            continue
        tr = purged_train_mask(label_end, m, window_days=window_days)
        if tr.sum() < MIN_TRAIN_ROWS:
            continue
        b, *_ = np.linalg.lstsq(Z[tr], ly[tr], rcond=None)
        s2 = float(np.var(ly[tr] - Z[tr] @ b))
        out[te] = np.exp(Z[te] @ b + 0.5 * s2)
    return out


def _date_codes(T: np.ndarray) -> tuple[np.ndarray, int]:
    _, inv = np.unique(T.astype("datetime64[D]"), return_inverse=True)
    return inv, int(inv.max()) + 1 if len(inv) else 0


def _sum_by(codes: np.ndarray, n: int, x: np.ndarray) -> np.ndarray:
    out = np.zeros(n)
    np.add.at(out, codes, x)
    return out


# ------------------------------------------------------------------ H-IC1

def ic1_segment(panel: pl.DataFrame, preds: dict, base: np.ndarray, seg: str) -> dict:
    sel = (panel["segment"] == seg).to_numpy() & np.isfinite(base)
    y = panel["y"].to_numpy()[sel]
    T = panel["T"].to_numpy()[sel]
    codes, nd = _date_codes(T)
    ref_d = _sum_by(codes, nd, (base[sel] - y) ** 2)
    sq_d = {k: _sum_by(codes, nd, (preds[k][sel] - y) ** 2) for k in preds}
    idx = block_bootstrap_idx(nd)
    ref_b = ref_d[idx].sum(1)
    skill = {k: float(1 - sq_d[k].sum() / ref_d.sum()) for k in preds}
    boot = {k: 1 - sq_d[k][idx].sum(1) / ref_b for k in preds}
    small = [k for k in preds if k <= 8]
    best_small_b = np.max(np.vstack([boot[k] for k in small]), axis=0)
    delta = skill[32] - max(skill[k] for k in small)
    delta_b = boot[32] - best_small_b
    ci = {k: [float(np.quantile(boot[k], 0.025)), float(np.quantile(boot[k], 0.975))] for k in preds}
    kstar = max(preds, key=lambda k: skill[k])
    all_zero = all(ci[k][0] <= 0 <= ci[k][1] for k in preds)
    if all_zero:
        verdict = "inconclusive"
    else:
        verdict = "supports" if kstar <= 8 and delta <= 0 else "contradicts"
    p = float((1 + np.sum(delta_b >= 0)) / (len(delta_b) + 1))
    # diagnostics: log loss, accuracy, AUC-free hit rate
    yb = float(y.mean())
    diag = {}
    for k in preds:
        pk = np.clip(preds[k][sel], 1e-6, 1 - 1e-6)
        diag[k] = {"log_loss": float(-np.mean(y * np.log(pk) + (1 - y) * np.log(1 - pk))), "accuracy": float(np.mean((pk > 0.5) == (y == 1))),
                   "sd_p": float(pk.std())}
    return {"n_obs": int(sel.sum()), "n_dates": nd, "n_symbols": int(panel.filter(pl.Series(sel))["symbol"].n_unique()),
            "base_rate_realised": yb, "skill": {str(k): skill[k] for k in preds}, "ci95": {str(k): ci[k] for k in preds},
            "argmax_k": kstar, "delta_32_vs_best_le8": float(delta),
            "delta_ci95": [float(np.quantile(delta_b, 0.025)), float(np.quantile(delta_b, 0.975))],
            "all_ci_contain_0": all_zero, "p_delta_ge_0": p, "verdict": verdict, "diag": {str(k): v for k, v in diag.items()}}


# ------------------------------------------------------------------ H-IC2

def _paired(loss: dict, T: np.ndarray, sel: np.ndarray) -> dict:
    codes, nd = _date_codes(T[sel])
    cnt = _sum_by(codes, nd, np.ones(sel.sum()))
    exp_l = loss[None][sel]
    res = {}
    for w in WINDOWS:
        lw = loss[w][sel]
        r = {"mean_loss": float(lw.mean())}
        if w is not None:
            dd = _sum_by(codes, nd, lw - exp_l) / cnt
            m, t, n = newey_west_t(dd)
            r.update({"diff_vs_expanding": m, "t_nw": t, "n_dates": n})
        res["expanding" if w is None else str(w)] = r
    return res


def _leg(res: dict) -> dict:
    losses = [res[str(w) if w is not None else "expanding"]["mean_loss"] for w in WINDOWS]
    dif = np.diff(losses)
    monotone = bool(np.all(dif > 0) or np.all(dif < 0))
    wins = [str(w) for w in WINDOWS if w is not None and res[str(w)]["diff_vs_expanding"] < 0 and res[str(w)]["t_nw"] < -2]
    best = [str(w) if w is not None else "expanding" for w in WINDOWS][int(np.argmin(losses))]
    ts = [res[str(w)]["t_nw"] for w in WINDOWS if w is not None]
    p = float(min(1.0, 4 * norm.cdf(np.nanmin(ts)))) if np.isfinite(ts).any() else 1.0
    v = "supports" if wins and not monotone else ("contradicts" if monotone else "inconclusive")
    return {"monotone": monotone, "direction": ("decreasing" if np.all(dif < 0) else "increasing") if monotone else "none",
            "significant_wins": wins, "best_window": best, "p_bonf": p, "verdict": v}


def ic2(panel: pl.DataFrame, k: int, m0: np.datetime64, har_panel: pl.DataFrame, m0_har: np.datetime64) -> dict:
    X = panel.select(FEATURES32).to_numpy()[:, :k]
    y = panel["y"].to_numpy()
    T = panel["T"].to_numpy()
    le = panel["label_end"].to_numpy()
    brier = {}
    for w in WINDOWS:
        p, _ = wf_logit(X, y, T, le, (k,), w, m0)
        brier[w] = (p[k] - y) ** 2
    ok = np.all(np.vstack([np.isfinite(brier[w]) for w in WINDOWS]), axis=0)
    Xh = har_panel.select(HAR_COLS).to_numpy()
    rvn = har_panel["rv_next"].to_numpy()
    ly = np.log(np.maximum(rvn, 1e-12))
    Th, leh = har_panel["T"].to_numpy(), har_panel["label_end"].to_numpy()
    ql = {w: qlike(rvn, wf_har(Xh, ly, Th, leh, w, m0_har)) for w in WINDOWS}
    okh = np.all(np.vstack([np.isfinite(ql[w]) for w in WINDOWS]), axis=0)
    out: dict = {"k": k}
    for seg in SEGS:
        s1 = ok & (panel["segment"] == seg).to_numpy()
        s2 = okh & (har_panel["segment"] == seg).to_numpy()
        lg, lh = _paired(brier, T, s1), _paired(ql, Th, s2)
        vl, vh = _leg(lg), _leg(lh)
        legs = (vl["verdict"], vh["verdict"])
        v = "supports" if legs == ("supports", "supports") else ("contradicts" if "contradicts" in legs else "inconclusive")
        out[seg] = {"logit": {"n_obs": int(s1.sum()), "windows": lg, **vl}, "har": {"n_obs": int(s2.sum()), "windows": lh, **vh},
                    "verdict": v, "p": max(vl["p_bonf"], vh["p_bonf"])}
    # pooled registry criterion
    pooled_ok, contra = True, False
    for leg in ("logit", "har"):
        win_any = any(out[s][leg]["significant_wins"] for s in SEGS)
        mono_any = any(out[s][leg]["monotone"] for s in SEGS)
        pooled_ok &= win_any and not mono_any
        contra |= out["llm_in_sample"][leg]["monotone"]
    out["pooled_verdict"] = "supports" if pooled_ok else ("contradicts" if contra else "inconclusive")
    return out


# ------------------------------------------------------------------ H-IC3

def ic3_segment(pnl: pl.DataFrame, strategy: str, seg: str, cut: date) -> dict:
    s = pnl.filter((pl.col("strategy") == strategy) & (pl.col("lag").is_in(LAGS)))
    s = s.filter(pl.col("date") < cut) if seg == "llm_in_sample" else s.filter(pl.col("date") >= cut)
    xs = s.group_by("date", "lag").agg(pl.col("pnl").mean(), pl.col("exposure").mean()).sort("date")
    wide = xs.pivot(on="lag", index="date", values="pnl").sort("date").drop_nulls()
    if wide.height < 20:
        return {"n_dates": wide.height, "verdict_leg": "inconclusive", "note": "too few dates"}
    M = np.column_stack([wide[str(d)].to_numpy() for d in LAGS])  # dates x lags
    edge = M.mean(0)
    idx = block_bootstrap_idx(M.shape[0], seed=7)
    Eb = M[idx].mean(1)  # n_boot x lags
    out: dict = {"n_dates": int(M.shape[0]), "n_symbols": int(s["symbol"].n_unique()),
                 "edge": {str(d): float(e) for d, e in zip(LAGS, edge, strict=True)},
                 "edge_t_nw": {str(d): newey_west_t(M[:, i])[1] for i, d in enumerate(LAGS)},
                 "sharpe_ann": {str(d): float(M[:, i].mean() / M[:, i].std(ddof=1) * np.sqrt(365)) if M[:, i].std() > 0 else float("nan")
                                for i, d in enumerate(LAGS)},
                 "exposure": float(xs.filter(pl.col("lag") == 0)["exposure"].mean())}
    e0 = edge[0]
    out["edge0_positive"] = bool(e0 > 0)
    with np.errstate(all="ignore"):
        ratio = edge / e0 if e0 != 0 else np.full(len(LAGS), np.nan)
        rb = Eb / Eb[:, :1]
    rb = np.where(Eb[:, :1] > 0, rb, np.nan)
    g = edge - 0.5 * e0
    gb = Eb - 0.5 * Eb[:, :1]
    out["ratio"] = {str(d): float(r) for d, r in zip(LAGS, ratio, strict=True)} if e0 > 0 else None
    out["ratio_ci95"] = {str(d): [float(np.nanquantile(rb[:, i], 0.025)), float(np.nanquantile(rb[:, i], 0.975))] for i, d in enumerate(LAGS)} if e0 > 0 else None
    out["share_boot_edge0_le0"] = float(np.mean(Eb[:, 0] <= 0))
    out["g"] = {str(d): float(v) for d, v in zip(LAGS, g, strict=True)}
    out["g_ci95"] = {str(d): [float(np.quantile(gb[:, i], 0.025)), float(np.quantile(gb[:, i], 0.975))] for i, d in enumerate(LAGS)}
    out["p_g_le0"] = {str(d): float((1 + np.sum(gb[:, i] <= 0)) / (len(gb) + 1)) for i, d in enumerate(LAGS)}
    out["p_g_ge0"] = {str(d): float((1 + np.sum(gb[:, i] >= 0)) / (len(gb) + 1)) for i, d in enumerate(LAGS)}
    out["half_life_h"] = half_life(LAGS, ratio) if e0 > 0 else None
    return out


def half_life(lags, ratio) -> float | str:
    """Linear interpolation of the first crossing of 0.5; '>max' if none."""
    r = np.asarray(ratio, dtype=float)
    for i in range(1, len(lags)):
        if r[i] <= 0.5:
            r0, r1 = r[i - 1], r[i]
            return float(lags[i - 1] + (r0 - 0.5) / (r0 - r1) * (lags[i] - lags[i - 1])) if r0 != r1 else float(lags[i])
    return f">{lags[-1]}"


def ic3(pnl: pl.DataFrame, cut: date) -> dict:
    out: dict = {}
    for seg in SEGS:
        tr = ic3_segment(pnl, "trend_ens_4h", seg, cut)
        sh = ic3_segment(pnl, "shock_regime", seg, cut)
        if not tr.get("edge0_positive"):
            tv, tp = "inconclusive", 1.0
            tr["leg_note"] = "edge(0) <= 0: ratio undefined"
        else:
            r4 = tr["ratio"]["4"]
            tp = tr["p_g_le0"]["4"]
            tv = "supports" if r4 > 0.5 and tr["g_ci95"]["4"][0] > 0 else ("contradicts" if r4 < 0.5 else "inconclusive")
        if not sh.get("edge0_positive"):
            sv, sp = "inconclusive", 1.0
            sh["leg_note"] = "edge(0) <= 0: ratio undefined"
        else:
            r2 = sh["ratio"]["2"]
            sp = sh["p_g_ge0"]["2"]
            sv = "supports" if r2 < 0.5 and sh["g_ci95"]["2"][1] < 0 else ("contradicts" if r2 > 0.5 else "inconclusive")
        legs = (tv, sv)
        v = "contradicts" if "contradicts" in legs else ("supports" if legs == ("supports", "supports") else "inconclusive")
        nat = pnl.filter(pl.col("strategy") == "trend_ens_4h_native")
        nat = nat.filter(pl.col("date") < cut) if seg == "llm_in_sample" else nat.filter(pl.col("date") >= cut)
        nat_x = nat.group_by("date").agg(pl.col("pnl").mean())["pnl"].to_numpy()
        tr["native_4h_edge"] = float(nat_x.mean()) if len(nat_x) else float("nan")
        out[seg] = {"trend": {**tr, "leg_verdict": tv}, "shock": {**sh, "leg_verdict": sv}, "verdict": v, "p": max(tp, sp)}
    out["pooled_verdict"] = out["llm_in_sample"]["verdict"]
    return out


# ------------------------------------------------------------------ run

def run(as_of: datetime | None = None, cache: str = "", workers: int = 4) -> dict:
    as_of = as_of or now()
    cut = knowledge_cutoff().date()
    daily, pnl, info = build(as_of, workers=workers, cache=cache)
    panel_all = decision_panel(daily, cut)
    panel = panel_all.filter(pl.all_horizontal([pl.col(c).is_not_null() for c in (*FEATURES32, "y")]) & pl.col("segment").is_not_null())
    X = panel.select(FEATURES32).to_numpy()
    y = panel["y"].to_numpy()
    T = panel["T"].to_numpy()
    le = panel["label_end"].to_numpy()
    m0 = _first_test_month(T, le)
    preds, base = wf_logit(X, y, T, le, KS, None, m0)
    out: dict = {"as_of": as_of.isoformat(), "cutoff": str(cut), "universe": list(UNIVERSE),
                 "panel": {"rows_all": panel_all.height, "rows_complete": panel.height, "symbols": panel["symbol"].n_unique(),
                           "first_T": str(panel["T"].min()), "last_T": str(panel["T"].max()), "first_test_month": str(m0)[:7],
                           "rows_by_symbol": dict(panel.group_by("symbol").len().sort("symbol").iter_rows()),
                           "missing_share_by_feature": {c: float(panel_all[c].is_null().mean()) for c in FEATURES32}},
                 "build": info}
    out["H-IC1"] = {seg: ic1_segment(panel, preds, base, seg) for seg in SEGS}
    v = [out["H-IC1"][s]["verdict"] for s in SEGS]
    out["H-IC1"]["pooled_verdict"] = "supports" if v == ["supports", "supports"] else ("contradicts" if "contradicts" in v else "inconclusive")
    kstar = int(out["H-IC1"]["llm_in_sample"]["argmax_k"])
    har_panel = panel_all.filter(pl.all_horizontal([pl.col(c).is_not_null() for c in (*HAR_COLS, "rv_next")]) & pl.col("segment").is_not_null()
                                 & (pl.col("rv_next") > 0) & (pl.col("rv") > 0))
    m0h = _first_test_month(har_panel["T"].to_numpy(), har_panel["label_end"].to_numpy())
    out["H-IC2"] = ic2(panel, kstar, m0, har_panel, m0h)
    out["H-IC2"]["har_first_test_month"] = str(m0h)[:7]
    out["H-IC3"] = ic3(pnl, cut)
    # Holm within the block, per segment
    out["holm"] = {}
    for seg in SEGS:
        ps = {"H-IC1": out["H-IC1"][seg]["p_delta_ge_0"], "H-IC2": out["H-IC2"][seg]["p"], "H-IC3": out["H-IC3"][seg]["p"]}
        adj = holm(ps)
        out["holm"][seg] = {"p": ps, "p_holm": adj}
    out["verdicts"] = {}
    for hid in ("H-IC1", "H-IC2", "H-IC3"):
        vs = {}
        for seg in SEGS:
            vv = out[hid][seg]["verdict"]
            if vv == "supports" and out["holm"][seg]["p_holm"][hid] > 0.05:
                vv = "inconclusive"
            vs[seg] = vv
        pv = out[hid]["pooled_verdict"]
        if pv == "supports" and any(out["holm"][s]["p_holm"][hid] > 0.05 for s in (SEGS if hid != "H-IC3" else ("llm_in_sample",))):
            pv = "inconclusive"
        vs["pooled"] = pv
        out["verdicts"][hid] = vs
    return out


DATA = ("bybit 1h + 4h bars, universe L (23 crypto perps), market_frame(funding=False, oi=False); funding and 4h OI via "
        "Lake.read_asof (known_ts <= T); features.realized.daily_measures; features.regime_features; strategies.shock_regime; "
        "trend_zoo.ensemble; src/ai_trader/research/decision_budget_study.py")


def record_all(out: dict) -> None:
    from ai_trader.research.hypotheses import record

    v = out["verdicts"]
    for seg in SEGS:
        r = out["H-IC1"][seg]
        m = {k: r[k] for k in ("n_obs", "n_dates", "n_symbols", "skill", "ci95", "argmax_k", "delta_32_vs_best_le8", "delta_ci95", "all_ci_contain_0", "p_delta_ge_0")}
        m["p_holm"] = out["holm"][seg]["p_holm"]["H-IC1"]
        record("H-IC1", seg, v["H-IC1"][seg], m, data=DATA,
               note=f"daily 00:00 UTC decisions, sign of next-24h return; monthly purged (24h) + embargoed (1d) ridge-logistic, first k of the fixed 32; "
                    f"Brier skill vs training base rate, 10-day block bootstrap; raw verdict {r['verdict']}")
    record("H-IC1", "pooled", v["H-IC1"]["pooled"], {s: {"argmax_k": out["H-IC1"][s]["argmax_k"], "skill": out["H-IC1"][s]["skill"]} for s in SEGS},
           data=DATA, note="registry criterion over both segments")
    for seg in SEGS:
        r = out["H-IC2"][seg]
        m = {"k": out["H-IC2"]["k"], "logit": r["logit"], "har": r["har"], "p": r["p"], "p_holm": out["holm"][seg]["p_holm"]["H-IC2"]}
        record("H-IC2", seg, v["H-IC2"][seg], m, data=DATA,
               note=f"windows 90/180/365/730/expanding; logit (Brier, k={out['H-IC2']['k']}) and pooled log-HAR (QLIKE); paired diff vs expanding, NW t over dates; raw verdict {r['verdict']}")
    record("H-IC2", "pooled", v["H-IC2"]["pooled"], {s: {"logit_best": out["H-IC2"][s]["logit"]["best_window"], "har_best": out["H-IC2"][s]["har"]["best_window"],
                                                       "logit_wins": out["H-IC2"][s]["logit"]["significant_wins"], "har_wins": out["H-IC2"][s]["har"]["significant_wins"]} for s in SEGS},
           data=DATA, note="registry criterion: a t<-2 shorter-window win in >= 1 segment and no monotone pattern, both models")
    for seg in SEGS:
        r = out["H-IC3"][seg]
        keep = ("n_dates", "edge", "edge_t_nw", "ratio", "ratio_ci95", "g_ci95", "half_life_h", "exposure", "leg_verdict", "native_4h_edge", "leg_note")
        m = {"trend": {k: r["trend"].get(k) for k in keep}, "shock": {k: r["shock"].get(k) for k in keep}, "p": r["p"], "p_holm": out["holm"][seg]["p_holm"]["H-IC3"]}
        record("H-IC3", seg, v["H-IC3"][seg], m, data=DATA,
               note=f"lag d in 0,1,2,4,8 h on 1h bars (4h ensemble applied to 1h returns by as-of availability); strategy_pnl net 7.5 bps; edge ratio with 10-day block bootstrap; raw verdict {r['verdict']}")
    record("H-IC3", "pooled", v["H-IC3"]["pooled"], {s: {"trend_ratio": out["H-IC3"][s]["trend"].get("ratio"), "shock_ratio": out["H-IC3"][s]["shock"].get("ratio")} for s in SEGS},
           data=DATA, note="registry criterion is llm_in_sample")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", default="")
    ap.add_argument("--record", action="store_true")
    ap.add_argument("--build-only", action="store_true")
    a = ap.parse_args()
    if a.build_only:
        d, p, _ = build(now(), workers=a.workers, cache=a.cache)
        print(d.height, p.height)
        return
    out = run(cache=a.cache, workers=a.workers)
    txt = json.dumps(out, indent=1, default=str)
    if a.out:
        Path(a.out).write_text(txt, encoding="utf-8")
    print(json.dumps(out["verdicts"], indent=1))
    if a.record:
        record_all(out)


if __name__ == "__main__":
    main()
