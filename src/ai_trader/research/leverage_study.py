"""Leverage effect / volatility asymmetry block (H-LEV1..H-LEV5, program 2026-11-10).

Design (fixed before any result was seen; diary 2026-11-10 "A1 leverage block ... design fixed before results"):

Data
  * Daily measures from 1h bars with the shared definition `ai_trader.features.realized.daily_measures` (complete UTC
    days only; rv, rs_neg, rs_pos, ret, close). Bars come from `research.frame.market_frame` (price sanity, listing
    burn-in). Each symbol's daily series is reindexed to the full calendar, so every rolling window (RV_w = mean of
    days t-4..t, RV_m = t-21..t, 30-day drawdown window) needs consecutive complete days; gaps give nulls, never a
    row-based window that silently spans a hole. Target = RV of day t+1 (must exist).
  * Origin (forecast / decision time) of row t = end of day t = its known_ts. Segments by origin vs
    research.knowledge_cutoff: llm_in_sample (origin < cutoff), clean_holdout (origin >= cutoff), pooled (all).
    One-day targets never straddle the cutoff.
  * Universe L = the 23 trend-zoo crypto perps. Universe T = Bybit TradFi perps with >= 120 complete days.
    H-LEV1 equity/index subset = T ∩ {TSLA INTC GOOGL AAPL MSTR NVDA MU QQQ SOXL SPCX}. External index series are
    used only if already in data/lake/alt_series (they are not: `index_series_available()`).

Models
  * Primary SHAR (Patton and Sheppard 2015), levels:  RV_{i,t+1} = a_i + b- RS-_it + b+ RS+_it + b_w RV_w + b_m RV_m.
    Every RV-type variable of a symbol is divided by that symbol's mean RV in the estimation sample (unit-free,
    equal weight per symbol). Symbol fixed effects by within-demeaning; CR1 standard errors clustered by origin
    date. Test: b- - b+ (t from the clustered covariance; p from Student t with G-1 df).
  * Log SHAR (robustness): log RV_{t+1} on log(RS-), log(RS+) (each floored at 1% of RV_t, as H-VCL2), log RV_w,
    log RV_m. Other robustness: Driscoll-Kraay SE (date-summed scores, NW lag), target-weekday dummies (TradFi
    weekends are quiet), and for H-LEV2 a 4h SHAR (4h semivariances from 1h bars, HAR lags 6/30/132 blocks,
    target-hour dummies).
  * GJR-GARCH(1,1)-t (`arch` 8.0, constant mean, daily close-to-close log returns in percent, robust SE): gamma;
    EGARCH(1,1)-t asymmetry term as robustness (leverage <=> EGARCH gamma < 0). Per asset per segment with >= 250
    daily returns, otherwise n/a; full-sample fits of shorter TradFi series are exploratory and never used in a
    verdict.
  * H-LEV3: D_t = 1{ln(close_t / max close_{t-29..t}) < -1.5 * std(ret_{t-29..t}) * sqrt(30)}. Primary = SHAR +
    RS- x D + RS+ x D (as registered); robustness adds D.
  * H-LEV4: pooled panel log-HAR (log RV_d, log RV_w, log RV_m) vs log-SHAR with symbol FE, expanding window, refit
    every 30 days at 00:00 of the refit day on rows whose target day is already complete; lognormal mean
    correction exp(s2/2). First fit when the pool has >= 250 rows over >= 60 origin dates; a symbol is forecast
    once it has >= 30 training rows (its FE). Levels (scaled by the training mean, floored at 1% of it) as
    robustness. Paired QLIKE(SHAR) - QLIKE(HAR) averaged by origin date, Newey-West t.
  * H-LEV5 (universe L): levels HAR (RV_d, RV_w, RV_m) + RS- + fz + RS- x fz; fz = (latest funding print known at
    the origin - mean) / std over prints with known_ts in the trailing 90 days (>= 90 prints), clipped to +-5.
  * Holm within the LEV block, per segment, over each hypothesis' primary test(s); raw and adjusted p reported.
    Verdicts apply the registered success criteria literally (Holm survival is reported next to it).

Run (in-sample recorded before the holdout is computed):
    uv run python -m ai_trader.research.leverage_study dry       # in-sample only, printed, not recorded
    uv run python -m ai_trader.research.leverage_study insample  # records llm_in_sample
    uv run python -m ai_trader.research.leverage_study rest      # records clean_holdout + pooled, writes reports
"""

from __future__ import annotations

import json
import os
import sys
import warnings
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl
from scipy import stats

from ai_trader.core.config import load
from ai_trader.core.time import now
from ai_trader.features.realized import daily_measures
from ai_trader.features.vol_models_x import qlike
from ai_trader.research.evaluate import knowledge_cutoff, newey_west_t

L_UNIVERSE = (
    "BTCUSDT ETHUSDT SOLUSDT XRPUSDT DOGEUSDT ADAUSDT LINKUSDT LTCUSDT BCHUSDT AVAXUSDT DOTUSDT BNBUSDT "
    "TRXUSDT NEARUSDT ATOMUSDT ETCUSDT FILUSDT UNIUSDT AAVEUSDT XLMUSDT SUIUSDT ENAUSDT HYPEUSDT"
).split()
T_CANDIDATES = (
    "XAUUSDT XAGUSDT CLUSDT BZUSDT TSLAUSDT INTCUSDT GOOGLUSDT AAPLUSDT MSTRUSDT NVDAUSDT MUUSDT QQQUSDT "
    "SOXLUSDT SPCXUSDT PAXGUSDT XAUTUSDT"
).split()
EQUITY = "TSLAUSDT INTCUSDT GOOGLUSDT AAPLUSDT MSTRUSDT NVDAUSDT MUUSDT QQQUSDT SOXLUSDT SPCXUSDT".split()
MIN_T_DAYS = 120
MIN_GARCH = 250
MIN_CLUSTERS = 20
MIN_HOLDOUT = 8
SEGS = ("llm_in_sample", "clean_holdout", "pooled")
DD_WIN, DD_K = 30, 1.5
FUND_WIN, FUND_MIN, FUND_CLIP = timedelta(days=90), 90, 5.0
REFIT, MIN_TRAIN_OBS, MIN_TRAIN_DATES, MIN_SYM_OBS = 30, 250, 60, 30
INDEX_HINTS = ("sp500", "nasdaq", "spx", "ndx", "djia", "fred", "vix")

DATA = {
    "H-LEV1": "universe T equity/index perps (Bybit 1h bars -> daily RV/RS from features.realized), GJR/EGARCH-t on daily returns; TradFi perps are mostly clean_holdout",
    "H-LEV2": "universe L 23 crypto perps (Bybit 1h bars -> daily RV/RS), GJR/EGARCH-t; 4h SHAR robustness",
    "H-LEV3": "universes L and T, daily SHAR with 30d drawdown interactions",
    "H-LEV4": "universes L and T, expanding-window panel log-HAR vs log-SHAR, refit 30d, next-day QLIKE",
    "H-LEV5": "universe L, daily HAR + RS- x funding z (Bybit funding as of origin, trailing 90d)",
}


# ------------------------------------------------------------------ data

def _ts(d: date) -> datetime:
    return datetime(d.year, d.month, d.day, tzinfo=UTC)


def index_series_available() -> list[str]:
    """Sources/series in data/lake/alt_series that look like equity indices (read-only directory listing)."""
    root = load().root / "data" / "lake" / "alt_series"
    if not root.is_dir():
        return []
    return [p.name for p in root.iterdir() if any(h in p.name.lower() for h in INDEX_HINTS)]


def calendar_reindex(d: pl.DataFrame) -> pl.DataFrame:
    """One symbol's daily rows reindexed to every calendar day between its first and last date (missing -> null)."""
    if d.height == 0:
        return d
    cal = pl.DataFrame({"date": pl.date_range(d["date"].min(), d["date"].max(), "1d", eager=True)})
    return cal.join(d, on="date", how="left").sort("date")


def add_features(d: pl.DataFrame) -> pl.DataFrame:
    """Per-symbol SHAR/HAR regressors on a calendar-reindexed daily frame. All regressors known at the end of day t."""
    rv = pl.col("rv")
    floor = 0.01 * rv
    hi = pl.col("close").rolling_max(DD_WIN)
    sd = pl.col("ret").rolling_std(DD_WIN)
    return d.sort("date").with_columns(
        rv.rolling_mean(5).alias("rv_w"),
        rv.rolling_mean(22).alias("rv_m"),
        rv.shift(-1).alias("rv_next"),
        pl.col("date").shift(-1).alias("_nd"),
        pl.max_horizontal(pl.col("rs_neg"), floor).alias("rs_neg_f"),
        pl.max_horizontal(pl.col("rs_pos"), floor).alias("rs_pos_f"),
        ((pl.col("close") / hi).log() < -DD_K * sd * np.sqrt(DD_WIN)).cast(pl.Float64).alias("dd"),
        (pl.col("date").cast(pl.Datetime("us", "UTC")) + pl.duration(days=1)).alias("origin"),
        (pl.col("date") + pl.duration(days=1)).dt.weekday().alias("tgt_wd"),
    ).drop("_nd")


def funding_z(fund_ts: np.ndarray, fund_rate: np.ndarray, origins: np.ndarray, window: timedelta = FUND_WIN,
              min_prints: int = FUND_MIN, clip: float = FUND_CLIP) -> np.ndarray:
    """z of the latest funding print known at each origin vs prints with known_ts in (origin - window, origin].

    fund_ts / origins: datetime64 arrays (same unit); fund_ts sorted ascending. Fewer than min_prints -> NaN."""
    ft = np.asarray(fund_ts).astype("datetime64[us]").astype(np.int64)
    fr = np.asarray(fund_rate, dtype=float)
    o = np.asarray(origins).astype("datetime64[us]").astype(np.int64)
    w = int(window / timedelta(microseconds=1))
    c1 = np.r_[0.0, np.cumsum(fr)]
    c2 = np.r_[0.0, np.cumsum(fr * fr)]
    hi = np.searchsorted(ft, o, side="right")          # prints with known_ts <= origin
    lo = np.searchsorted(ft, o - w, side="right")      # first print with known_ts > origin - window
    n = hi - lo
    out = np.full(len(o), np.nan)
    ok = n >= max(min_prints, 2)
    s1, s2 = c1[hi] - c1[lo], c2[hi] - c2[lo]
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = s1 / n
        var = (s2 - n * mean**2) / (n - 1)
        last = fr[np.clip(hi - 1, 0, len(fr) - 1)] if len(fr) else np.zeros(len(o))
        z = (last - mean) / np.sqrt(var)
    ok &= np.isfinite(z) & (var > 0)
    out[ok] = np.clip(z[ok], -clip, clip)
    return out


def blocks_4h(bars: pl.DataFrame) -> pl.DataFrame:
    """4h realised measures from 1h bars (blocks aligned to 00/04/../20 UTC, complete blocks only), calendar
    reindexed, with 4h HAR lags (6 / 30 / 132 blocks) and the next-block target."""
    b = bars.sort("event_ts").with_columns((pl.col("close").log() - pl.col("close").shift(1).log()).alias("r")).drop_nulls("r")
    b = b.with_columns(pl.col("event_ts").dt.truncate("4h").alias("blk"))
    g = b.group_by("blk").agg(
        pl.len().alias("n"), (pl.col("r") ** 2).sum().alias("rv"),
        pl.when(pl.col("r") < 0).then(pl.col("r") ** 2).otherwise(0.0).sum().alias("rs_neg"),
        pl.when(pl.col("r") > 0).then(pl.col("r") ** 2).otherwise(0.0).sum().alias("rs_pos"),
    ).filter(pl.col("n") == 4).drop("n")
    if g.height == 0:
        return g
    cal = pl.DataFrame({"blk": pl.datetime_range(g["blk"].min(), g["blk"].max(), "4h", eager=True)})
    g = cal.join(g, on="blk", how="left").sort("blk")
    rv = pl.col("rv")
    return g.with_columns(
        rv.rolling_mean(6).alias("rv_d"), rv.rolling_mean(30).alias("rv_w"), rv.rolling_mean(132).alias("rv_m"),
        rv.shift(-1).alias("rv_next"),
        (pl.col("blk") + pl.duration(hours=4)).alias("origin"),
        ((pl.col("blk") + pl.duration(hours=4)).dt.hour() // 4).alias("tgt_h"),
    )


def build(as_of: datetime | None = None, cache: str | None = None) -> dict:
    """Load both universes. Returns {'daily': panel, 'h4': 4h panel (L only), 'fund': funding z added, 'meta'}."""
    cache = cache or os.environ.get("AT_LEV_CACHE")
    if cache and (Path(cache) / "daily.parquet").is_file():
        meta = json.loads((Path(cache) / "meta.json").read_text())
        return {"daily": pl.read_parquet(Path(cache) / "daily.parquet"), "h4": pl.read_parquet(Path(cache) / "h4.parquet"), "meta": meta}
    from ai_trader.research.frame import market_frame
    from ai_trader.storage.lake import Lake

    as_of = as_of or now()
    lake = Lake()
    daily, h4, counts = [], [], {}
    for sym in L_UNIVERSE + T_CANDIDATES:
        uni = "L" if sym in L_UNIVERSE else "T"
        f = market_frame(sym, "1h", as_of, lake=lake, funding=False, oi=False)
        if f.height == 0:
            counts[sym] = 0
            continue
        bars = f.select("event_ts", "open", "high", "low", "close").filter(pl.col("event_ts") + timedelta(hours=1) <= as_of)
        d = daily_measures(bars, "1h").filter(pl.col("known_ts") <= as_of)
        counts[sym] = d.height
        if uni == "T" and d.height < MIN_T_DAYS:
            continue
        d = add_features(calendar_reindex(d.drop("known_ts"))).with_columns(pl.lit(sym).alias("symbol"), pl.lit(uni).alias("universe"))
        if uni == "L":
            fu = lake.read_asof("funding", as_of, symbols=[sym], filters={"venue": "bybit"})
            if fu.height:
                fu = fu.sort("known_ts")
                z = funding_z(fu["known_ts"].to_numpy(), fu["rate"].to_numpy(), d["origin"].to_numpy())
            else:
                z = np.full(d.height, np.nan)
            d = d.with_columns(pl.Series("fz", z))
            h = blocks_4h(bars)
            if h.height:
                h4.append(h.with_columns(pl.lit(sym).alias("symbol")))
        else:
            d = d.with_columns(pl.lit(None, dtype=pl.Float64).alias("fz"))
        daily.append(d)
    out = {
        "daily": pl.concat(daily, how="vertical_relaxed"),
        "h4": pl.concat(h4, how="vertical_relaxed"),
        "meta": {"as_of": as_of.isoformat(), "complete_days": counts, "index_series_in_lake": index_series_available()},
    }
    if cache:
        Path(cache).mkdir(parents=True, exist_ok=True)
        out["daily"].write_parquet(Path(cache) / "daily.parquet")
        out["h4"].write_parquet(Path(cache) / "h4.parquet")
        (Path(cache) / "meta.json").write_text(json.dumps(out["meta"]))
    return out


def segment(df: pl.DataFrame, seg: str, col: str = "origin") -> pl.DataFrame:
    k = knowledge_cutoff()
    if seg == "llm_in_sample":
        return df.filter(pl.col(col) < k)
    if seg == "clean_holdout":
        return df.filter(pl.col(col) >= k)
    return df


# ------------------------------------------------------------------ panel regression

def demean(a: np.ndarray, g: np.ndarray) -> np.ndarray:
    """Within transformation: subtract group means (g: integer codes 0..G-1). a: (n,) or (n, k)."""
    a2 = a.reshape(len(a), -1).astype(float)
    cnt = np.bincount(g)
    out = np.empty_like(a2)
    for j in range(a2.shape[1]):
        m = np.bincount(g, weights=a2[:, j]) / np.maximum(cnt, 1)
        out[:, j] = a2[:, j] - m[g]
    return out.reshape(a.shape) if a.ndim == 1 else out


def cluster_cov(X: np.ndarray, u: np.ndarray, cl: np.ndarray, k_absorbed: int = 0) -> np.ndarray:
    """CR1 cluster-robust covariance of OLS coefficients (X already demeaned). k_absorbed = absorbed FE count."""
    n, k = X.shape
    bread = np.linalg.pinv(X.T @ X)
    s = X * u[:, None]
    _, inv = np.unique(cl, return_inverse=True)
    G = inv.max() + 1
    sg = np.zeros((G, k))
    np.add.at(sg, inv, s)
    meat = sg.T @ sg
    adj = G / max(G - 1, 1) * (n - 1) / max(n - k - k_absorbed, 1)
    return adj * bread @ meat @ bread


def driscoll_kraay_cov(X: np.ndarray, u: np.ndarray, time_id: np.ndarray, lag: int | None = None) -> np.ndarray:
    """Driscoll-Kraay covariance: Newey-West (Bartlett) on the per-period sums of scores (cross-sectional and serial
    dependence). time_id must sort chronologically."""
    k = X.shape[1]
    bread = np.linalg.pinv(X.T @ X)
    _, inv = np.unique(time_id, return_inverse=True)
    T = inv.max() + 1
    h = np.zeros((T, k))
    np.add.at(h, inv, X * u[:, None])
    lag = int(np.floor(4 * (T / 100) ** (2 / 9))) if lag is None else lag
    S = h.T @ h
    for j in range(1, lag + 1):
        g = h[j:].T @ h[:-j]
        S += (1 - j / (lag + 1)) * (g + g.T)
    return bread @ S @ bread


def fe_ols(df: pl.DataFrame, y: str, xs: list[str], fe: str = "symbol", cluster: str = "origin", dummies: str | None = None) -> dict:
    """Pooled OLS with `fe` fixed effects (demeaning), optional categorical dummies (first level dropped), CR1 SE
    clustered by `cluster`, Driscoll-Kraay SE as robustness. Rows with any null/non-finite value are dropped."""
    cols = [y, *xs, fe, cluster] + ([dummies] if dummies else [])
    d = df.select(cols).drop_nulls()
    d = d.filter(pl.all_horizontal([pl.col(c).is_finite() for c in [y, *xs]]))
    names = list(xs)
    X = d.select(xs).to_numpy().astype(float)
    if dummies and d.height:
        lv = sorted(d[dummies].unique().to_list())[1:]
        D = np.column_stack([(d[dummies] == v).to_numpy().astype(float) for v in lv]) if lv else np.empty((d.height, 0))
        X = np.column_stack([X, D])
        names += [f"{dummies}={v}" for v in lv]
    n, k = X.shape
    G = d[cluster].n_unique() if n else 0
    if n < k + 10 or G < 3:
        return {"n": n, "n_clusters": G, "ok": False}
    _, g = np.unique(d[fe].to_numpy(), return_inverse=True)
    Xd, yd = demean(X, g), demean(d[y].to_numpy().astype(float), g)
    beta, *_ = np.linalg.lstsq(Xd, yd, rcond=None)
    u = yd - Xd @ beta
    ts = d[cluster].to_numpy()
    V = cluster_cov(Xd, u, ts, k_absorbed=int(g.max() + 1))
    Vdk = driscoll_kraay_cov(Xd, u, ts)
    r2 = 1 - (u @ u) / (yd @ yd) if yd @ yd > 0 else float("nan")
    return {"ok": True, "n": n, "n_clusters": G, "n_symbols": int(g.max() + 1), "names": names, "beta": beta, "V": V, "Vdk": Vdk, "r2_within": float(r2)}


def lincomb(res: dict, w: dict[str, float], dk: bool = False) -> dict:
    """Estimate, SE, t and one-sided p (H1: > 0, Student t with G-1 df; DK uses the normal) of sum w_j beta_j."""
    if not res.get("ok"):
        return {"est": float("nan"), "se": float("nan"), "t": float("nan"), "p_pos": float("nan")}
    a = np.array([w.get(nm, 0.0) for nm in res["names"]])
    est = float(a @ res["beta"])
    se = float(np.sqrt(max(a @ (res["Vdk"] if dk else res["V"]) @ a, 0.0)))
    t = est / se if se > 0 else float("nan")
    p = float(stats.norm.sf(t)) if dk else float(stats.t.sf(t, df=max(res["n_clusters"] - 1, 1)))
    return {"est": est, "se": se, "t": t, "p_pos": p}


def coef_table(res: dict) -> dict:
    if not res.get("ok"):
        return {}
    se = np.sqrt(np.maximum(np.diag(res["V"]), 0))
    return {nm: {"b": float(b), "se": float(s), "t": float(b / s) if s > 0 else float("nan")} for nm, b, s in zip(res["names"], res["beta"], se, strict=True) if "=" not in nm}


def scale_by_symbol(df: pl.DataFrame, cols: list[str], ref: str = "rv") -> pl.DataFrame:
    """Divide `cols` by each symbol's mean of `ref` in this (estimation) sample: unit-free, equal weight per symbol."""
    c = pl.col(ref).mean().over("symbol")
    return df.with_columns([(pl.col(x) / c).alias(f"{x}_s") for x in cols])


LEVEL_COLS = ["rv_next", "rs_neg", "rs_pos", "rv", "rv_w", "rv_m"]


def shar(df: pl.DataFrame, form: str = "level", extra: list[str] | None = None, dummies: str | None = None) -> dict:
    """SHAR fit on a segment of the daily panel. form 'level' (scaled) or 'log'. extra: additional regressor names
    in the scaled/log space, built by the caller. Returns the fit plus the b(RS-) - b(RS+) test."""
    extra = extra or []
    if form == "level":
        d = scale_by_symbol(df, LEVEL_COLS)
        y, xs = "rv_next_s", ["rs_neg_s", "rs_pos_s", "rv_w_s", "rv_m_s"]
    else:
        d = df.filter(pl.col("rv_next") > 0).with_columns(
            pl.col("rv_next").log().alias("l_next"), pl.col("rs_neg_f").log().alias("l_neg"), pl.col("rs_pos_f").log().alias("l_pos"),
            pl.col("rv_w").log().alias("l_w"), pl.col("rv_m").log().alias("l_m"))
        y, xs = "l_next", ["l_neg", "l_pos", "l_w", "l_m"]
    res = fe_ols(d, y, xs + extra, dummies=dummies)
    neg, pos = xs[0], xs[1]
    return {"fit": res, "diff": lincomb(res, {neg: 1, pos: -1}), "diff_dk": lincomb(res, {neg: 1, pos: -1}, dk=True), "coefs": coef_table(res),
            "n": res["n"], "n_dates": res["n_clusters"]}


def with_scaled(df: pl.DataFrame) -> pl.DataFrame:
    return scale_by_symbol(df, LEVEL_COLS)


# ------------------------------------------------------------------ GARCH

def garch_asym(r: np.ndarray) -> dict:
    """GJR-GARCH(1,1)-t and EGARCH(1,1)-t on daily log returns (fraction; fitted in percent). Robust SE."""
    from arch import arch_model

    r = np.asarray(r, dtype=float)
    r = r[np.isfinite(r)] * 100.0
    out: dict = {"n": len(r)}
    for name, vol in (("gjr", "GARCH"), ("egarch", "EGARCH")):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                m = arch_model(r, mean="Constant", vol=vol, p=1, o=1, q=1, dist="t")
                fit = m.fit(disp="off", options={"maxiter": 1000})
            g, se = float(fit.params["gamma[1]"]), float(fit.std_err["gamma[1]"])
            out[name] = {"gamma": g, "se": se, "t": g / se if se > 0 else float("nan"), "converged": int(fit.convergence_flag) == 0,
                         "nu": float(fit.params.get("nu", np.nan))}
            if name == "gjr":
                out[name].update(alpha=float(fit.params["alpha[1]"]), beta=float(fit.params["beta[1]"]))
        except Exception as e:  # noqa: BLE001 - a failed fit is reported, not fatal
            out[name] = {"error": str(e)[:120]}
    return out


def garch_table(panel: pl.DataFrame, symbols: list[str], seg: str, exploratory: bool = False) -> dict:
    """Per asset GJR/EGARCH on the segment's daily returns (dates < cutoff, >= cutoff, all). n < MIN_GARCH -> n/a
    unless exploratory (full sample only, flagged)."""
    k = knowledge_cutoff().date()
    rows = {}
    for s in symbols:
        d = panel.filter((pl.col("symbol") == s) & pl.col("ret").is_not_null()).sort("date")
        if seg == "llm_in_sample":
            d = d.filter(pl.col("date") < k)
        elif seg == "clean_holdout":
            d = d.filter(pl.col("date") >= k)
        n = d.height
        if n < MIN_GARCH and not (exploratory and n >= MIN_T_DAYS):
            rows[s] = {"n": n, "status": "n/a (< 250 days)"}
            continue
        g = garch_asym(d["ret"].to_numpy())
        g["status"] = "ok" if n >= MIN_GARCH else "exploratory (< 250 days)"
        rows[s] = g
    gam = [v["gjr"]["gamma"] for v in rows.values() if v.get("status") == "ok" and "gamma" in v.get("gjr", {})]
    eg = [v["egarch"]["gamma"] for v in rows.values() if v.get("status") == "ok" and "gamma" in v.get("egarch", {})]
    tt = [v["gjr"]["t"] for v in rows.values() if v.get("status") == "ok" and "gamma" in v.get("gjr", {})]
    summ = {"n_assets_fit": len(gam), "median_gjr_gamma": float(np.median(gam)) if gam else None,
            "n_gamma_pos": int(np.sum(np.array(gam) > 0)) if gam else 0, "n_gamma_t_gt2": int(np.sum(np.array(tt) > 2)) if tt else 0,
            "n_gamma_t_lt_m2": int(np.sum(np.array(tt) < -2)) if tt else 0,
            "median_egarch_gamma": float(np.median(eg)) if eg else None, "n_egarch_neg": int(np.sum(np.array(eg) < 0)) if eg else 0}
    return {"summary": summ, "assets": rows}


# ------------------------------------------------------------------ H-LEV4 out of sample

def oos_panel(df: pl.DataFrame, spec: str, form: str = "log", refit: int = REFIT) -> pl.DataFrame:
    """Expanding-window pooled FE forecasts of RV_{t+1}. spec 'har' or 'shar'. Training at refit day tau uses rows
    with origin <= tau - 1 day (target day complete by tau); forecasts rows with origin in [tau, tau + refit)."""
    if form == "log":
        d = df.with_columns(pl.col("rv").log().alias("x_d"), pl.col("rs_neg_f").log().alias("x_neg"), pl.col("rs_pos_f").log().alias("x_pos"),
                            pl.col("rv_w").log().alias("x_w"), pl.col("rv_m").log().alias("x_m"),
                            pl.when(pl.col("rv_next") > 0).then(pl.col("rv_next").log()).alias("y"))
    else:
        d = df.with_columns(pl.col("rv").alias("x_d"), pl.col("rs_neg").alias("x_neg"), pl.col("rs_pos").alias("x_pos"),
                            pl.col("rv_w").alias("x_w"), pl.col("rv_m").alias("x_m"), pl.col("rv_next").alias("y"))
    xs = ["x_d", "x_w", "x_m"] if spec == "har" else ["x_neg", "x_pos", "x_w", "x_m"]
    d = d.select("symbol", "origin", "rv", "rv_next", "y", *xs).drop_nulls([*xs]).filter(pl.all_horizontal([pl.col(c).is_finite() for c in xs])).sort("origin")
    origins = d["origin"].to_numpy()
    sym = d["symbol"].to_numpy()
    X = d.select(xs).to_numpy().astype(float)
    y = d["y"].to_numpy().astype(float)
    rv = d["rv"].to_numpy().astype(float)
    trainable = np.isfinite(y)
    days = np.unique(origins)
    one = np.timedelta64(1, "D")
    tau0 = None
    for t in days:
        m = trainable & (origins <= t - one)
        if m.sum() >= MIN_TRAIN_OBS and len(np.unique(origins[m])) >= MIN_TRAIN_DATES:
            tau0 = t
            break
    fc = np.full(len(d), np.nan)
    if tau0 is None:
        return d.select("symbol", "origin", "rv_next").with_columns(pl.Series("f", fc))
    tau = tau0
    last = days.max()
    while tau <= last:
        tr = trainable & (origins <= tau - one)
        te = (origins >= tau) & (origins < tau + np.timedelta64(refit, "D"))
        if te.any():
            usyms, g = np.unique(sym[tr], return_inverse=True)
            cnt = np.bincount(g)
            if form == "level":   # scale by the training mean RV per symbol (causal)
                c = np.bincount(g, weights=rv[tr]) / cnt
                Xs, ys = X[tr] / c[g][:, None], y[tr] / c[g]
            else:
                Xs, ys = X[tr], y[tr]
            beta, *_ = np.linalg.lstsq(demean(Xs, g), demean(ys, g), rcond=None)
            alpha = np.bincount(g, weights=ys) / cnt - np.column_stack([np.bincount(g, weights=Xs[:, j]) / cnt for j in range(Xs.shape[1])]) @ beta
            resid = ys - (alpha[g] + Xs @ beta)
            s2 = float(resid @ resid / max(len(ys) - len(beta) - len(usyms), 1))
            idx = {s: i for i, s in enumerate(usyms)}
            for i in np.where(te)[0]:
                j = idx.get(sym[i])
                if j is None or cnt[j] < MIN_SYM_OBS:
                    continue
                if form == "log":
                    fc[i] = np.exp(alpha[j] + X[i] @ beta + 0.5 * s2)
                else:
                    cj = c[j]
                    fc[i] = cj * max(alpha[j] + (X[i] / cj) @ beta, 0.01)
        tau = tau + np.timedelta64(refit, "D")
    return d.select("symbol", "origin", "rv_next").with_columns(pl.Series("f", fc))


def paired_qlike(a: pl.DataFrame, b: pl.DataFrame) -> pl.DataFrame:
    """Join two forecast frames (symbol, origin, rv_next, f) and return per-row QLIKE of each and the difference a - b."""
    j = a.rename({"f": "fa"}).join(b.select("symbol", "origin", pl.col("f").alias("fb")), on=["symbol", "origin"], how="inner")
    j = j.filter(pl.col("fa").is_finite() & pl.col("fb").is_finite() & (pl.col("rv_next") > 0))
    qa = qlike(j["rv_next"].to_numpy(), j["fa"].to_numpy())
    qb = qlike(j["rv_next"].to_numpy(), j["fb"].to_numpy())
    return j.with_columns(pl.Series("qa", qa), pl.Series("qb", qb), pl.Series("dq", qa - qb))


def date_avg_test(q: pl.DataFrame, col: str = "dq") -> dict:
    """Average the paired difference across symbols per origin date, then Newey-West t (clustering by time)."""
    if q.height == 0:
        return {"mean": float("nan"), "t": float("nan"), "n_dates": 0, "n_obs": 0}
    x = q.group_by("origin").agg(pl.col(col).mean()).sort("origin")[col].to_numpy()
    m, t, n = newey_west_t(x)
    return {"mean": m, "t": t, "n_dates": n, "n_obs": q.height, "mean_qa": float(q["qa"].mean()), "mean_qb": float(q["qb"].mean()),
            "p_neg": float(stats.norm.cdf(t)) if np.isfinite(t) else float("nan"),
            "p_two": float(2 * stats.norm.sf(abs(t))) if np.isfinite(t) else float("nan")}


# ------------------------------------------------------------------ 4h robustness

def shar_4h(h4: pl.DataFrame, seg: str) -> dict:
    d = segment(h4, seg)
    c = pl.col("rv").mean().over("symbol")
    d = d.with_columns(*[(pl.col(x) / c).alias(f"{x}_s") for x in ("rv_next", "rs_neg", "rs_pos", "rv_d", "rv_w", "rv_m")],
                       pl.col("origin").dt.date().alias("odate"))
    res = fe_ols(d, "rv_next_s", ["rs_neg_s", "rs_pos_s", "rv_d_s", "rv_w_s", "rv_m_s"], cluster="odate", dummies="tgt_h")
    return {"diff": lincomb(res, {"rs_neg_s": 1, "rs_pos_s": -1}), "diff_dk": lincomb(res, {"rs_neg_s": 1, "rs_pos_s": -1}, dk=True),
            "coefs": coef_table(res), "n": res["n"], "n_dates": res["n_clusters"]}


# ------------------------------------------------------------------ hypotheses

def _r(x, nd=4):
    if isinstance(x, dict):
        return {k: _r(v, nd) for k, v in x.items()}
    if isinstance(x, list | tuple):
        return [_r(v, nd) for v in x]
    if isinstance(x, float | np.floating):
        return float(round(float(x), nd)) if np.isfinite(x) else None
    if isinstance(x, np.integer):
        return int(x)
    return x


def _shar_summary(s: dict) -> dict:
    return {"n": s["n"], "n_dates": s["n_dates"], "diff": s["diff"]["est"], "diff_t": s["diff"]["t"], "diff_p_pos": s["diff"]["p_pos"],
            "diff_t_dk": s["diff_dk"]["t"], "coefs": s["coefs"]}


def lev_shar_block(panel: pl.DataFrame, seg: str) -> dict:
    """Primary + robustness SHAR fits for one universe subset and segment."""
    d = segment(panel, seg)
    prim = shar(d, "level")
    return {"primary_level": _shar_summary(prim), "log": _shar_summary(shar(d, "log")),
            "level_cal": _shar_summary(shar(d, "level", dummies="tgt_wd")), "n_symbols": d.filter(pl.col("rv_next").is_not_null())["symbol"].n_unique(),
            "_p": prim["diff"]["p_pos"], "_t": prim["diff"]["t"], "_est": prim["diff"]["est"], "_G": prim["n_dates"]}


def lev3_fit(panel: pl.DataFrame, seg: str, main_effect: bool = False) -> dict:
    d = with_scaled(segment(panel, seg)).with_columns((pl.col("rs_neg_s") * pl.col("dd")).alias("neg_dd"), (pl.col("rs_pos_s") * pl.col("dd")).alias("pos_dd"))
    xs = ["rs_neg_s", "rs_pos_s", "neg_dd", "pos_dd", "rv_w_s", "rv_m_s"] + (["dd"] if main_effect else [])
    res = fe_ols(d, "rv_next_s", xs)
    ok_rows = d.drop_nulls(["rv_next_s", *xs])
    n_dd_dates = ok_rows.filter(pl.col("dd") > 0)["origin"].n_unique() if ok_rows.height else 0
    return {"inter_neg": lincomb(res, {"neg_dd": 1}), "inter_pos": lincomb(res, {"pos_dd": 1}), "inter_neg_dk": lincomb(res, {"neg_dd": 1}, dk=True),
            "neg_in_dd_minus_pos_in_dd": lincomb(res, {"rs_neg_s": 1, "neg_dd": 1, "rs_pos_s": -1, "pos_dd": -1}),
            "coefs": coef_table(res), "n": res["n"], "n_dates": res["n_clusters"], "n_dd_dates": n_dd_dates,
            "dd_share": float(ok_rows["dd"].mean()) if ok_rows.height else None}


def lev5_fit(panel: pl.DataFrame, seg: str) -> dict:
    d = with_scaled(segment(panel, seg)).with_columns((pl.col("rs_neg_s") * pl.col("fz")).alias("neg_fz"))
    xs = ["rv_s", "rv_w_s", "rv_m_s", "rs_neg_s", "fz", "neg_fz"]
    res = fe_ols(d, "rv_next_s", xs)
    return {"inter": lincomb(res, {"neg_fz": 1}), "inter_dk": lincomb(res, {"neg_fz": 1}, dk=True), "coefs": coef_table(res),
            "n": res["n"], "n_dates": res["n_clusters"]}


def holm(pvals: dict[str, float]) -> dict[str, float]:
    """Holm step-down adjusted p-values (NaN entries are left out and returned as NaN)."""
    items = [(k, p) for k, p in pvals.items() if p is not None and np.isfinite(p)]
    items.sort(key=lambda kv: kv[1])
    m = len(items)
    out = {k: float("nan") for k in pvals}
    run = 0.0
    for i, (k, p) in enumerate(items):
        run = max(run, min(1.0, (m - i) * p))
        out[k] = run
    return out


def _judge(seg: str, ok: bool, sign_ok: bool, enough: bool) -> str:
    """Registered criterion met -> supports; sign opposite to the claim -> contradicts; else inconclusive.
    Not enough data -> inconclusive."""
    if not enough:
        return "inconclusive"
    if ok:
        return "supports"
    return "contradicts" if not sign_ok else "inconclusive"


def evaluate_segment(B: dict, seg: str) -> dict:
    """All five hypotheses for one segment: {hid: (verdict, metrics, note)} plus the Holm table."""
    P = B["daily"]
    L = P.filter(pl.col("universe") == "L")
    T = P.filter(pl.col("universe") == "T")
    t_syms = sorted(T["symbol"].unique().to_list())
    eq_syms = [s for s in EQUITY if s in t_syms]
    E = T.filter(pl.col("symbol").is_in(eq_syms))
    hold_note = "TradFi perps start Mar-Jun 2026: their sample is mostly clean_holdout. " if seg != "clean_holdout" else ""
    out, tests = {}, {}

    # H-LEV1 ---------------------------------------------------------------
    s1 = lev_shar_block(E, seg)
    g1 = garch_table(P, eq_syms, seg)
    g1x = garch_table(P, eq_syms, "pooled", exploratory=True) if seg == "pooled" else None
    med = g1["summary"]["median_gjr_gamma"]
    enough = s1["_G"] >= (MIN_HOLDOUT if seg == "clean_holdout" else MIN_CLUSTERS)
    ok = (s1["_est"] > 0 and s1["_t"] > 2) and (med is None or med > 0)
    sign_ok = s1["_est"] > 0 and (med is None or med > 0)
    v = _judge(seg, ok, sign_ok, enough and np.isfinite(s1["_t"]))
    tests["H-LEV1:diff>0"] = s1["_p"]
    m1 = {"universe": "T equity/index", "symbols": eq_syms, "shar": {k: v_ for k, v_ in s1.items() if not k.startswith("_")},
          "gjr_summary": g1["summary"], "gjr_assets": g1["assets"], "index_series_in_lake": B["meta"].get("index_series_in_lake", [])}
    if g1x:
        m1["gjr_exploratory_full_sample"] = g1x
    out["H-LEV1"] = [v, m1, hold_note + ("GJR n/a in this segment (< 250 days per asset); verdict from SHAR only. " if med is None else "")
                     + "No equity index series in data/lake/alt_series, long-history robustness unavailable. Leverage is a textbook result: in-sample support is weak evidence."]

    # H-LEV2 ---------------------------------------------------------------
    s2 = lev_shar_block(L, seg)
    g2 = garch_table(P, L_UNIVERSE, seg)
    h4 = shar_4h(B["h4"], seg)
    med2 = g2["summary"]["median_gjr_gamma"]
    enough = s2["_G"] >= (MIN_HOLDOUT if seg == "clean_holdout" else MIN_CLUSTERS) and np.isfinite(s2["_t"])
    if not enough:
        v = "inconclusive"
    elif s2["_t"] >= 2:
        v = "contradicts"
    else:                                         # registered success: "difference t < 2 ... supports no leverage"
        v = "supports"
    tests["H-LEV2:diff!=0"] = float(2 * min(s2["_p"], 1 - s2["_p"])) if np.isfinite(s2["_p"]) else float("nan")
    out["H-LEV2"] = [v, {"universe": "L", "shar": {k: v_ for k, v_ in s2.items() if not k.startswith("_")}, "gjr_summary": g2["summary"],
                         "gjr_assets": g2["assets"], "shar_4h": {"n": h4["n"], "n_dates": h4["n_dates"], "diff": h4["diff"]["est"], "diff_t": h4["diff"]["t"],
                                                                  "diff_t_dk": h4["diff_dk"]["t"], "coefs": h4["coefs"]},
                         "sign": "inverse" if s2["_t"] < -2 else ("leverage" if s2["_t"] > 2 else "none")},
                     ("GJR n/a in this segment (< 250 days); verdict from SHAR only. " if med2 is None else f"Median GJR gamma {med2:.4f} reported, not part of the registered success test. ") + "Supports = no significant positive asymmetry (t < 2, registered criterion); t < -2 would be inverse asymmetry."]

    # H-LEV3 ---------------------------------------------------------------
    per = {}
    for name, U in (("L", L), ("T", T)):
        f = lev3_fit(U, seg)
        fm = lev3_fit(U, seg, main_effect=True)
        need = MIN_HOLDOUT if seg == "clean_holdout" else MIN_CLUSTERS
        enough = f["n_dates"] >= MIN_CLUSTERS and f["n_dd_dates"] >= need and np.isfinite(f["inter_neg"]["t"])
        est, t = f["inter_neg"]["est"], f["inter_neg"]["t"]
        if seg == "clean_holdout":
            vv = _judge(seg, est > 0, est > 0, enough)
        else:
            vv = _judge(seg, est > 0 and t > 2, est > 0, enough)
        per[name] = {"verdict": vv, "fit": f, "with_dd_main_effect": {"inter_neg": fm["inter_neg"], "inter_pos": fm["inter_pos"], "dd": fm["coefs"].get("dd")}}
        tests[f"H-LEV3:{name}:inter>0"] = f["inter_neg"]["p_pos"]
    vs = [per["L"]["verdict"], per["T"]["verdict"]]
    v = "supports" if all(x == "supports" for x in vs) else ("contradicts" if "contradicts" in vs else "inconclusive")
    out["H-LEV3"] = [v, {"L": per["L"], "T": per["T"]}, hold_note + "Requires both universes; T in-sample is tiny. Drawdown = close below 30d max close by > 1.5 sigma_30d sqrt(30)."]

    # H-LEV4 ---------------------------------------------------------------
    per = {}
    for name, U in (("L", L), ("T", T)):
        res = {}
        for form in ("log", "level"):
            q = paired_qlike(B["oos"][(name, form, "shar")], B["oos"][(name, form, "har")])
            res[form] = date_avg_test(segment(q, seg))
        per[name] = res
    tl, tt_ = per["L"]["log"], per["T"]["log"]
    enough_t = tt_["n_dates"] >= (MIN_HOLDOUT if seg == "clean_holdout" else MIN_CLUSTERS) and np.isfinite(tt_["t"])
    enough_l = tl["n_dates"] >= MIN_CLUSTERS and np.isfinite(tl["t"])
    l_ok = enough_l and abs(tl["t"]) < 2
    l_bad = enough_l and tl["t"] <= -2           # SHAR significantly better in crypto: contradicts "not for crypto"
    t_ok = enough_t and tt_["mean"] < 0 and tt_["t"] < -2
    t_bad = enough_t and tt_["mean"] >= 0
    if seg == "llm_in_sample":                   # registered T criterion is holdout-only
        v = "contradicts" if l_bad else ("supports" if l_ok and (t_ok or not enough_t) else "inconclusive")
    else:
        v = "contradicts" if (l_bad or t_bad) else ("supports" if l_ok and t_ok else "inconclusive")
    tests["H-LEV4:T:dq<0"] = tt_.get("p_neg", float("nan"))
    tests["H-LEV4:L:dq!=0"] = tl.get("p_two", float("nan"))
    out["H-LEV4"] = [v, {"L": per["L"], "T": per["T"], "primary": "log form", "design": f"expanding window, refit {REFIT}d, FE, first fit >= {MIN_TRAIN_OBS} rows / {MIN_TRAIN_DATES} dates"},
                     hold_note + "dq = QLIKE(SHAR) - QLIKE(HAR), date-averaged, NW t; negative = SHAR better. Registered T criterion applies to the holdout."]

    # H-LEV5 ---------------------------------------------------------------
    f5 = lev5_fit(L, seg)
    est, t = f5["inter"]["est"], f5["inter"]["t"]
    enough = f5["n_dates"] >= (MIN_HOLDOUT if seg == "clean_holdout" else MIN_CLUSTERS) and np.isfinite(t)
    if seg == "clean_holdout":
        v = _judge(seg, est > 0, est > 0, enough)
    else:
        v = _judge(seg, est > 0 and t > 2, est > 0, enough)
    tests["H-LEV5:inter>0"] = f5["inter"]["p_pos"]
    out["H-LEV5"] = [v, {"universe": "L", **f5}, "fz = funding z vs trailing 90d prints known at origin, clipped +-5; levels scaled by symbol mean RV."]

    adj = holm(tests)
    for hid in out:
        out[hid][1]["tests"] = {k: {"p_raw": tests[k], "p_holm": adj[k]} for k in tests if k.startswith(hid + ":")}
    return {"results": out, "holm": {k: {"p_raw": tests[k], "p_holm": adj[k]} for k in tests}}


def add_oos(B: dict) -> dict:
    P = B["daily"]
    B["oos"] = {}
    for name in ("L", "T"):
        U = P.filter(pl.col("universe") == name)
        for form in ("log", "level"):
            for spec in ("har", "shar"):
                B["oos"][(name, form, spec)] = oos_panel(U, spec, form)
    return B


# ------------------------------------------------------------------ reporting

def _fmt(x, nd=3):
    if x is None:
        return "n/a"
    try:
        return f"{float(x):.{nd}f}" if np.isfinite(float(x)) else "n/a"
    except (TypeError, ValueError):
        return str(x)


def write_report(res: dict, meta: dict, stem: str = "leverage_2026-11-10") -> tuple[Path, Path]:
    root = load().root / "research" / "reports"
    jp, mp = root / f"{stem}.json", root / f"{stem}.md"
    jp.write_text(json.dumps(_r({"meta": meta, "segments": res}), indent=1, default=str), encoding="utf-8")
    L = ["# Leverage effect and volatility asymmetry (H-LEV1..H-LEV5)", "",
         f"Generated by `ai_trader.research.leverage_study` as of {meta['as_of']}. Data: Bybit 1h bars -> daily realised measures "
         "(`features.realized`); Bybit funding (as of). Program: research/plans/2026-11-10_volatility_program.md.", "",
         "## Methods", "", (__doc__ or "").split("Run (")[0].strip().replace("\n  ", "\n"), "",
         "## Verdicts", "", "| hypothesis | llm_in_sample | clean_holdout | pooled |", "|---|---|---|---|"]
    for hid in ("H-LEV1", "H-LEV2", "H-LEV3", "H-LEV4", "H-LEV5"):
        L.append(f"| {hid} | " + " | ".join(res[s]["results"][hid][0] for s in SEGS) + " |")
    L += ["", "## SHAR asymmetry b(RS-) - b(RS+) (levels scaled by symbol mean RV, symbol FE, date-clustered t)", "",
          "| universe | segment | n | dates | b(RS-) | b(RS+) | diff | t | t (DK) | log diff | log t | cal-dummies t |", "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for hid, lab in (("H-LEV1", "T equity"), ("H-LEV2", "L crypto")):
        for s in SEGS:
            sh = res[s]["results"][hid][1]["shar"]
            p, lg, cal = sh["primary_level"], sh["log"], sh["level_cal"]
            c = p["coefs"]
            L.append(f"| {lab} | {s} | {p['n']} | {p['n_dates']} | {_fmt(c.get('rs_neg_s', {}).get('b'))} | {_fmt(c.get('rs_pos_s', {}).get('b'))} | "
                     f"{_fmt(p['diff'])} | {_fmt(p['diff_t'], 2)} | {_fmt(p['diff_t_dk'], 2)} | {_fmt(lg['diff'])} | {_fmt(lg['diff_t'], 2)} | {_fmt(cal['diff_t'], 2)} |")
    L += ["", "4h SHAR robustness (universe L, target-hour dummies):", "", "| segment | n | diff | t | t (DK) |", "|---|---|---|---|---|"]
    for s in SEGS:
        h = res[s]["results"]["H-LEV2"][1]["shar_4h"]
        L.append(f"| {s} | {h['n']} | {_fmt(h['diff'])} | {_fmt(h['diff_t'], 2)} | {_fmt(h['diff_t_dk'], 2)} |")
    L += ["", "## GJR-GARCH(1,1)-t gamma (assets with >= 250 days in the segment)", "",
          "| universe | segment | assets fit | median gamma | gamma > 0 | t > 2 | t < -2 | median EGARCH gamma | EGARCH < 0 |", "|---|---|---|---|---|---|---|---|---|"]
    for hid, lab in (("H-LEV1", "T equity"), ("H-LEV2", "L crypto")):
        for s in SEGS:
            g = res[s]["results"][hid][1]["gjr_summary"]
            L.append(f"| {lab} | {s} | {g['n_assets_fit']} | {_fmt(g['median_gjr_gamma'])} | {g['n_gamma_pos']} | {g['n_gamma_t_gt2']} | {g['n_gamma_t_lt_m2']} | "
                     f"{_fmt(g['median_egarch_gamma'])} | {g['n_egarch_neg']} |")
    ex = res["pooled"]["results"]["H-LEV1"][1].get("gjr_exploratory_full_sample", {}).get("assets", {})
    if ex:
        L += ["", "Exploratory full-sample GJR/EGARCH for equity perps (< 250 days, not used in verdicts):", "", "| symbol | n | GJR gamma | t | EGARCH gamma | t |", "|---|---|---|---|---|---|"]
        for s, a in ex.items():
            gj, eg = a.get("gjr", {}), a.get("egarch", {})
            L.append(f"| {s} | {a['n']} | {_fmt(gj.get('gamma'))} | {_fmt(gj.get('t'), 2)} | {_fmt(eg.get('gamma'))} | {_fmt(eg.get('t'), 2)} |")
    L += ["", "## H-LEV3 drawdown interactions (RS- x D, RS+ x D)", "",
          "| universe | segment | n | dates | dd share | dd dates | RS-xD | t | RS+xD | t | RS-xD t (with D) |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for s in SEGS:
        m = res[s]["results"]["H-LEV3"][1]
        for u in ("L", "T"):
            f = m[u]["fit"]
            L.append(f"| {u} | {s} | {f['n']} | {f['n_dates']} | {_fmt(f['dd_share'])} | {f['n_dd_dates']} | {_fmt(f['inter_neg']['est'])} | {_fmt(f['inter_neg']['t'], 2)} | "
                     f"{_fmt(f['inter_pos']['est'])} | {_fmt(f['inter_pos']['t'], 2)} | {_fmt(m[u]['with_dd_main_effect']['inter_neg']['t'], 2)} |")
    L += ["", "## H-LEV4 out-of-sample QLIKE, SHAR minus HAR (negative = SHAR better)", "",
          "| universe | segment | form | dates | obs | QLIKE HAR | QLIKE SHAR | diff | NW t |", "|---|---|---|---|---|---|---|---|---|"]
    for s in SEGS:
        m = res[s]["results"]["H-LEV4"][1]
        for u in ("L", "T"):
            for form in ("log", "level"):
                q = m[u][form]
                L.append(f"| {u} | {s} | {form} | {q['n_dates']} | {q['n_obs']} | {_fmt(q.get('mean_qb'), 4)} | {_fmt(q.get('mean_qa'), 4)} | {_fmt(q['mean'], 4)} | {_fmt(q['t'], 2)} |")
    L += ["", "## H-LEV5 RS- x funding z (universe L)", "", "| segment | n | dates | RS-xfz | t | t (DK) | fz | t |", "|---|---|---|---|---|---|---|---|"]
    for s in SEGS:
        m = res[s]["results"]["H-LEV5"][1]
        L.append(f"| {s} | {m['n']} | {m['n_dates']} | {_fmt(m['inter']['est'])} | {_fmt(m['inter']['t'], 2)} | {_fmt(m['inter_dk']['t'], 2)} | "
                 f"{_fmt(m['coefs'].get('fz', {}).get('b'))} | {_fmt(m['coefs'].get('fz', {}).get('t'), 2)} |")
    L += ["", "## Holm adjustment (within the LEV block, per segment)", "", "| segment | test | p raw | p Holm |", "|---|---|---|---|"]
    for s in SEGS:
        for k, v in res[s]["holm"].items():
            L.append(f"| {s} | {k} | {_fmt(v['p_raw'], 4)} | {_fmt(v['p_holm'], 4)} |")
    L.append("")
    mp.write_text("\n".join(L), encoding="utf-8")
    return jp, mp


def main(arg: str) -> None:
    from ai_trader.research.hypotheses import record

    B = add_oos(build())
    segs = {"dry": ["llm_in_sample"], "insample": ["llm_in_sample"], "rest": list(SEGS)}[arg]
    res = {}
    for seg in segs:
        res[seg] = evaluate_segment(B, seg)
        for hid, (verdict, metrics, note) in res[seg]["results"].items():
            print(hid, seg, verdict, json.dumps(_r({k: v for k, v in metrics.items() if k not in ("gjr_assets",)}), default=str)[:1500])
            if (arg == "insample" and seg == "llm_in_sample") or (arg == "rest" and seg != "llm_in_sample"):
                record(hid, seg, verdict, _r(metrics), note=note, data=DATA[hid])
    if arg == "rest":
        jp, mp = write_report(res, B["meta"])
        print("wrote", jp, mp)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "dry")
