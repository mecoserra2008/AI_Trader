"""Less obvious strategies, block NS (H-NS1..H-NS5, program 2026-11-11, agent B3).

    PYTHONIOENCODING=utf-8 uv run python -m ai_trader.research.new_strategies_study build [--workers 4]
    PYTHONIOENCODING=utf-8 uv run python -m ai_trader.research.new_strategies_study evaluate [--record]

build:    per symbol (<= 4 worker processes) research.frame.market_frame(sym, "1h", AS_OF, funding=False, oi=False) for
          universe L (trend_zoo_study.UNIVERSE), the TradFi candidates of the vol tournament and QQQ; cached to
          data/research/new_strategies/f1h_<SYMBOL>.parquet. Funding is read point-in-time from the lake at evaluate.
evaluate: the five tests below, Holm within the block, the report research/reports/new_strategies_2026-11-11.{json,md},
          and with --record one registry row per hypothesis and segment.

Common definitions (all fixed before any result was seen):
  * vol-scaled return of bar t: z_t = r_t / s_t, s_t = sigma_risk_next of bar t - 1h (the robust, bias-corrected
    per-bar sigma without weekend shrink, known at the open of bar t; bar t - 1h must exist, else z_t is missing);
  * segment: llm_in_sample if the event time < research.knowledge_cutoff (2026-07-01) else clean_holdout;
  * date-clustered SE = OLS with standard errors clustered by UTC date (statsmodels 'cluster'); NW t = evaluate.newey_west_t;
  * directional verdict per segment, strict form (|t| > 2 required): supports if every part has the predicted sign with
    |t| > 2; contradicts if any part has the wrong sign with |t| > 2, or every part has the wrong sign; else inconclusive.
    Sign form (holdout rows where the criterion only asks for the same sign): supports if every part has the predicted
    sign; contradicts / inconclusive as in the strict form;
  * Holm within the block: one primary p per hypothesis (one-sided normal p in the predicted direction; for a two-part
    criterion the larger of the two p's, an intersection-union test), Holm-adjusted over the 5; a 'supports' in the
    primary segment is downgraded to 'inconclusive' if the Holm-adjusted p >= 0.05.

H-NS1 funding-settlement pressure (universe L). Settlements are mapped per symbol from the lake's funding table
  (features.seasonality.funding_settlements): only prints reached after exactly 480 minutes at 00/08/16 UTC (ENA's
  4h regime from 2025-01 and SOL's 2h episode in 2022 drop out; every L symbol is otherwise 8h). For settlement T:
  origin = T - 1h; fz = z of the latest print known at the origin (the T - 8h settlement) vs prints known in the trailing
  90 days (>= 90 prints, clipped +-5; leverage_study.funding_z, as H-LEV5); x = sign(latest rate) * |fz|.
  pre = z of the bar [T-1h, T); post = z of the bar [T, T+1h). Pooled OLS y = a + b x, date-clustered SE.
  Criterion: b_pre < 0 and b_post > 0, |t| > 2 in-sample, same signs in the holdout. Diagnostics (not in the verdict):
  x_alt = latest rate / trailing RMS of rates (scale-only z), and a placebo at the mid-interval hours 04/12/20 UTC.

H-NS2 options expiry (BTC, ETH). 24h windows anchored at 08:00 UTC (window d = [d 08:00, d+1 08:00)), RV = sum of the
  24 squared 1h log returns (complete windows only), on a gap-aware calendar. Forecast: the per-asset log-HAR of
  features.vol_panel (lags 1 / 7 / 30 windows, expanding, refit every 30 windows, >= 180 windows) fitted on this
  08:00-anchored series; residual e_d = log RV_d - log F_d. Every Friday is a Deribit expiry, so:
  C1 (primary, the registered test): per expiry week, pre = window Thu 08 -> Fri 08, post = Fri 08 -> Sat 08,
     control = mean of the Tue 08 -> Wed 08 and Wed 08 -> Thu 08 windows of the same week (Mon 08 -> Tue 08 is excluded
     because its HAR lags hold the quiet weekend, which biases its residual upward). d_pre = e_pre - control,
     d_post = e_post - control, averaged over BTC and ETH per week, NW t over weeks. Known bias: the post window holds
     Saturday 00-08 UTC, which the HAR over-forecasts, so d_post is biased DOWN (against the hypothesis).
  C2 (modifier, 'strongest at monthly'): e_pre and e_post of monthly (incl. quarterly) minus non-monthly expiry weeks,
     Welch t over weeks. Same weekday, so weekday-clean. Reported, not part of the verdict. Quarterly reported too.
  Criterion (C1): d_pre < 0 and d_post > 0, |t| > 2 in-sample, same signs in the holdout.

H-NS3 session seasonality (universe L). Sessions by bar open hour: Asia 00-08, Europe 08-13, US 13-20, late 20-24 UTC.
  z averaged over symbols per timestamp, then per (date, session); dates with all four sessions. Wald test of equal
  session means with a HAC (Bartlett, NW lag) covariance of the daily 4-vector, F = W / 3 on (3, n - 3) df.
  Spearman of the four in-sample vs holdout session means. Tilt strategy: long the in-sample best session, short the
  in-sample worst, flat otherwise (the signal of bar t is the session of bar t + 1, held over it), evaluate.strategy_pnl
  at 7.5 bps per side, summed per symbol-date then averaged over symbols per date.
  In-sample verdict: supports if F p < 0.01, inconclusive if 0.01 <= p < 0.10, contradicts if p >= 0.10.
  Holdout verdict: supports if Spearman > 0.5 and the holdout tilt net mean > 0; contradicts if neither; else inconclusive.

H-NS4 hour-of-week vol profile (universes L and T; T = TradFi candidates with >= 120 complete UTC days). Base F0_t =
  s_t^2; seasonal F1_t = F0_t * f_t, f from features.seasonality.seasonal_factor on u = r^2 / F0 (expanding, kappa 20,
  bandwidth 1 h, cap 50, floor 0.1, >= 8 weeks of history, else the bar is excluded). Paired QLIKE difference per bar
  d = QLIKE(F1) - QLIKE(F0) = r^2 (1/F1 - 1/F0) + log(F1/F0) (the log r^2 term cancels, so zero returns are fine),
  averaged over all symbol-bars per UTC date, NW t over dates. Criterion: d < 0 with t < -2 in both segments, for each
  universe. Secondary (not in the verdict): a trailing 26-week profile, and the weekend-aware sigma_rb_next as base.

H-NS5 QQQ first NYSE hour (QQQ perp, BTC, ETH). Signal bar = the 1h bar containing 09:30 America/New_York on NYSE
  trading days (features.seasonality.nyse_open_bar: 13:00 UTC under EDT, 14:00 UTC under EST; NYSE holidays dropped);
  x = QQQ z of that bar. y = sum of the next three 1h log returns of BTC (ETH) / (sigma_risk_next at the signal bar
  x sqrt 3). Pooled OLS over BTC and ETH, date-clustered SE. Criterion: slope > 0 with t > 2 pooled over BTC and ETH;
  the primary segment is 'pooled' (all dates: QQQ 1h bars start 2026-06-06 after the listing burn-in, so the in-sample
  segment is ~3 weeks); per-segment rows use the strict form.
"""

from __future__ import annotations

import argparse
import json
import math
import time as _time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl

from ai_trader.core.config import load
from ai_trader.features import seasonality as sz
from ai_trader.features import vol_panel as vp
from ai_trader.research.evaluate import knowledge_cutoff, newey_west_t, strategy_pnl
from ai_trader.research.trend_zoo_study import UNIVERSE as UNIVERSE_L

AS_OF = datetime(2026, 11, 10, tzinfo=UTC)
T_CANDIDATES = ["XAUUSDT", "XAGUSDT", "CLUSDT", "BZUSDT", "TSLAUSDT", "INTCUSDT", "GOOGLUSDT", "AAPLUSDT", "MSTRUSDT", "NVDAUSDT", "MUUSDT", "QQQUSDT", "SOXLUSDT", "SPCXUSDT", "PAXGUSDT", "XAUTUSDT"]
MIN_T_DAYS = 120
COST_BPS = 7.5
SEGS = ("llm_in_sample", "clean_holdout")
REPORT_TAG = "2026-11-11"
HIDS = ("H-NS1", "H-NS2", "H-NS3", "H-NS4", "H-NS5")
FRAME_COLS = ("event_ts", "open", "high", "low", "close", "ret", "fwd_ret", "sigma_next", "sigma_risk_next", "sigma_rb_next")


def out_dir() -> Path:
    d = load().root / "data" / "research" / "new_strategies"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _cut() -> datetime:
    return knowledge_cutoff()


# ------------------------------------------------------------------ build


def symbol_frame(symbol: str, refresh: bool = False) -> str:
    from ai_trader.research.frame import market_frame

    p = out_dir() / f"f1h_{symbol}.parquet"
    if p.exists() and not refresh:
        return str(p)
    t0 = _time.time()
    f = market_frame(symbol, "1h", AS_OF, funding=False, oi=False)
    if f.height == 0:
        return ""
    f.select(*FRAME_COLS).write_parquet(p)
    print(f"{symbol}: {f.height} bars, {_time.time() - t0:.0f}s", flush=True)
    return str(p)


def build(workers: int = 4, refresh: bool = False) -> None:
    syms = list(dict.fromkeys([*UNIVERSE_L, *T_CANDIDATES]))
    with ProcessPoolExecutor(max_workers=min(workers, 4)) as ex:
        futs = [ex.submit(symbol_frame, s, refresh) for s in syms]
        for fu in as_completed(futs):
            fu.result()


def load_frame(symbol: str) -> pl.DataFrame:
    p = out_dir() / f"f1h_{symbol}.parquet"
    if not p.exists():
        return pl.DataFrame()
    f = pl.read_parquet(p).sort("event_ts")
    prev = f.select((pl.col("event_ts") + pl.duration(hours=1)).alias("event_ts"), pl.col("sigma_risk_next").alias("s_pre"),
                    pl.col("sigma_rb_next").alias("s_pre_rb"))
    return f.join(prev, on="event_ts", how="left").with_columns((pl.col("ret") / pl.col("s_pre")).alias("z"))


def complete_days(f: pl.DataFrame) -> int:
    return int(f.group_by(pl.col("event_ts").dt.date()).len().filter(pl.col("len") == 24).height)


# ------------------------------------------------------------------ statistics helpers


def ols_cluster(y: np.ndarray, X: np.ndarray, groups: np.ndarray) -> dict:
    """OLS with an intercept and date-clustered SE. Returns {b, t, se, n, n_clusters} for each slope (list order)."""
    import statsmodels.api as sm

    ok = np.isfinite(y) & np.isfinite(X).all(axis=1)
    y, X, g = y[ok], X[ok], groups[ok]
    if len(y) < 30:
        return {"n": len(y), "b": [float("nan")] * X.shape[1], "t": [float("nan")] * X.shape[1]}
    codes = np.unique(g, return_inverse=True)[1]
    res = sm.OLS(y, sm.add_constant(X, has_constant="add")).fit(cov_type="cluster", cov_kwds={"groups": codes})
    return {"n": len(y), "n_clusters": int(codes.max() + 1), "a": float(res.params[0]), "t_a": float(res.tvalues[0]),
            "b": [float(v) for v in res.params[1:]], "se": [float(v) for v in res.bse[1:]], "t": [float(v) for v in res.tvalues[1:]]}


def p_one(t: float, sign: int) -> float:
    """One-sided normal p for the predicted sign (+1: slope > 0, -1: slope < 0)."""
    if not np.isfinite(t):
        return 1.0
    return 0.5 * math.erfc(sign * t / math.sqrt(2))


def verdict(parts: list[tuple[float, float, int]], strict: bool) -> str:
    """parts: (estimate, t, predicted sign). See the module docstring."""
    ok_sign = [np.isfinite(b) and np.sign(b) == s for b, _, s in parts]
    sig = [np.isfinite(t) and abs(t) > 2 for _, t, _ in parts]
    if any((not o) and g and np.isfinite(b) for (b, _, _), o, g in zip(parts, ok_sign, sig, strict=True)) or not any(ok_sign):
        return "contradicts"
    if all(ok_sign) and (all(sig) or not strict):
        return "supports"
    return "inconclusive"


def holm(p: dict[str, float]) -> dict[str, float]:
    return vp.holm(p)


def seg_of(ts: pl.Expr) -> pl.Expr:
    return pl.when(ts < _cut()).then(pl.lit("llm_in_sample")).otherwise(pl.lit("clean_holdout")).alias("segment")


def _nw(x: np.ndarray) -> dict:
    m, t, n = newey_west_t(np.asarray(x, dtype=float))
    return {"mean": m, "t": t, "n": n}


# ------------------------------------------------------------------ H-NS1 funding settlement


def ns1_events(frames: dict[str, pl.DataFrame], funding: pl.DataFrame, hours: tuple[int, ...] = (0, 8, 16),
               interval_min: int | None = 480) -> pl.DataFrame:
    """One row per (symbol, settlement T): x (sign(rate) |fz|), x_alt, pre and post z. For the placebo pass hours
    (4, 12, 20) and interval_min None: T is then a mid-interval hour (no settlement)."""
    from ai_trader.research.leverage_study import funding_z

    rows = []
    for sym, f in frames.items():
        fu = funding.filter(pl.col("symbol") == sym).sort("known_ts")
        if fu.height < 100 or f.height == 0:
            continue
        if interval_min is not None:
            st = sz.funding_settlements(fu, interval_min=interval_min, hours=hours).select("event_ts")
        else:
            st = f.filter(pl.col("event_ts").dt.hour().is_in(list(hours))).select("event_ts")
        if st.height == 0:
            continue
        T = st["event_ts"]
        origin = (T - timedelta(hours=1)).to_numpy()
        kt = fu["known_ts"].to_numpy()
        fr = fu["rate"].to_numpy().astype(float)
        fz = funding_z(kt, fr, origin)
        idx = np.searchsorted(kt.astype("datetime64[us]").astype(np.int64), origin.astype("datetime64[us]").astype(np.int64), side="right") - 1
        last = np.where(idx >= 0, fr[np.clip(idx, 0, None)], np.nan)
        # scale-only z: latest rate / RMS of prints known in the trailing 90 days
        ki = kt.astype("datetime64[us]").astype(np.int64)
        oi = origin.astype("datetime64[us]").astype(np.int64)
        lo = np.searchsorted(ki, oi - int(timedelta(days=90) / timedelta(microseconds=1)), side="right")
        c2 = np.r_[0.0, np.cumsum(fr * fr)]
        n = idx + 1 - lo
        with np.errstate(invalid="ignore", divide="ignore"):
            rms = np.sqrt((c2[idx + 1] - c2[lo]) / n)
            x_alt = np.where(n >= 90, np.clip(last / rms, -5, 5), np.nan)
        ev = pl.DataFrame({"T": T, "x": np.sign(last) * np.abs(fz), "fz": fz, "rate": last, "x_alt": x_alt})
        zz = f.select("event_ts", "z")
        ev = ev.join(zz.rename({"event_ts": "T", "z": "post"}), on="T", how="left").join(
            zz.with_columns((pl.col("event_ts") + pl.duration(hours=1)).alias("T")).select("T", pl.col("z").alias("pre")), on="T", how="left")
        rows.append(ev.with_columns(pl.lit(sym).alias("symbol")))
    out = pl.concat(rows) if rows else pl.DataFrame()
    return out.with_columns(seg_of(pl.col("T")), pl.col("T").dt.date().alias("date"))


def _ns1_fit(ev: pl.DataFrame, xcol: str = "x") -> dict:
    out = {}
    for seg in (*SEGS, "pooled"):
        e = ev if seg == "pooled" else ev.filter(pl.col("segment") == seg)
        g = e["date"].cast(pl.Utf8).to_numpy()
        X = e[xcol].to_numpy().reshape(-1, 1).astype(float)
        pre = ols_cluster(e["pre"].to_numpy().astype(float), X, g)
        post = ols_cluster(e["post"].to_numpy().astype(float), X, g)
        out[seg] = {"pre": {k: (v[0] if isinstance(v, list) else v) for k, v in pre.items()},
                    "post": {k: (v[0] if isinstance(v, list) else v) for k, v in post.items()},
                    "n_symbols": int(e["symbol"].n_unique()), "n_events": int(e.height)}
    return out


def ns1(frames: dict[str, pl.DataFrame], funding: pl.DataFrame) -> dict:
    ev = ns1_events(frames, funding)
    main = _ns1_fit(ev, "x")
    alt = _ns1_fit(ev, "x_alt")
    placebo = _ns1_fit(ns1_events(frames, funding, hours=(4, 12, 20), interval_min=None), "x")
    # decile-style description: mean pre/post z by x bucket (pooled)
    buckets = (ev.drop_nulls(["x", "pre", "post"]).filter(pl.col("x").is_finite() & pl.col("pre").is_finite() & pl.col("post").is_finite()).with_columns(
        pl.when(pl.col("x") <= -1).then(pl.lit("x<=-1")).when(pl.col("x") < 1).then(pl.lit("-1<x<1")).otherwise(pl.lit("x>=1")).alias("bucket"))
        .group_by(["segment", "bucket"]).agg(pl.len().alias("n"), pl.col("pre").mean(), pl.col("post").mean()).sort(["segment", "bucket"]))
    res = {"main": main, "x_alt": alt, "placebo_mid_interval": placebo, "buckets": buckets.to_dicts(),
           "symbols": sorted(ev["symbol"].unique().to_list())}
    parts = {seg: [(main[seg]["pre"]["b"], main[seg]["pre"]["t"], -1), (main[seg]["post"]["b"], main[seg]["post"]["t"], 1)] for seg in (*SEGS, "pooled")}
    res["verdicts"] = {"llm_in_sample": verdict(parts["llm_in_sample"], True), "clean_holdout": verdict(parts["clean_holdout"], False)}
    res["p_primary"] = max(p_one(main["llm_in_sample"]["pre"]["t"], -1), p_one(main["llm_in_sample"]["post"]["t"], 1))
    res["primary_segment"] = "llm_in_sample"
    return res


# ------------------------------------------------------------------ H-NS2 options expiry


def window_series(f: pl.DataFrame, anchor_hour: int = 8) -> pl.DataFrame:
    """24h windows [d anchor, d+1 anchor): rv over complete windows on a gap-aware calendar, with the per-asset log-HAR
    forecast F (made at the window's start) and the residual e = log rv - log F."""
    from ai_trader.features.realized import daily_measures

    b = f.select((pl.col("event_ts") - pl.duration(hours=anchor_hour)).alias("event_ts"), "open", "high", "low", "close")
    d = daily_measures(b, "1h").select("date", "rv")
    cal = pl.DataFrame({"date": pl.date_range(d["date"].min(), d["date"].max(), "1d", eager=True)})
    d = cal.join(d, on="date", how="left").sort("date")
    rv = d["rv"].cast(pl.Float64).fill_null(np.nan).to_numpy()
    X, _ = vp.design("loghar", {"rv": rv})
    fc = vp.per_asset_forecast(X, vp.target(rv, 1, True), 1, log=True, refit_every=30, min_days=180, first=0)
    F = np.r_[np.nan, fc[:-1]]  # forecast made at the end of window d-1 for window d
    with np.errstate(invalid="ignore", divide="ignore"):
        e = np.log(rv) - np.log(F)
    return d.with_columns(pl.Series("F", F), pl.Series("e", e), pl.col("date").dt.weekday().alias("wd"))


def ns2(frames: dict[str, pl.DataFrame]) -> dict:
    weeks = []
    for sym in ("BTCUSDT", "ETHUSDT"):
        w = window_series(frames[sym])
        # Friday expiry date of the week each window belongs to (Tue..Fri windows map to that week's Friday)
        w = w.filter(pl.col("wd").is_in([2, 3, 4, 5])).with_columns((pl.col("date") + pl.duration(days=5 - pl.col("wd"))).alias("fri"))
        p = w.pivot(on="wd", index="fri", values="e", aggregate_function="first")
        p = p.rename({c: f"e{c}" for c in p.columns if c != "fri"})
        for c in ("e2", "e3", "e4", "e5"):
            if c not in p.columns:
                p = p.with_columns(pl.lit(None, dtype=pl.Float64).alias(c))
        p = p.with_columns(((pl.col("e2") + pl.col("e3")) / 2).alias("ctrl")).with_columns(
            (pl.col("e4") - pl.col("ctrl")).alias("d_pre"), (pl.col("e5") - pl.col("ctrl")).alias("d_post"),
            pl.col("e4").alias("e_pre"), pl.col("e5").alias("e_post"), pl.lit(sym).alias("symbol"))
        weeks.append(p.select("fri", "symbol", "e_pre", "e_post", "ctrl", "d_pre", "d_post"))
    wk = pl.concat(weeks).drop_nulls(["d_pre", "d_post"]).filter(pl.col("d_pre").is_finite() & pl.col("d_post").is_finite())
    kinds = {r["date"]: r["kind"] for r in sz.expiry_calendar(wk["fri"].min(), wk["fri"].max() + timedelta(days=1)).to_dicts()}
    wk = wk.with_columns(pl.col("fri").replace_strict(kinds, default=None).alias("kind")).with_columns(
        pl.col("kind").is_in(["monthly", "quarterly"]).alias("monthly"),
        pl.when(pl.col("fri").cast(pl.Datetime("us", "UTC")) + pl.duration(hours=8) < _cut()).then(pl.lit("llm_in_sample")).otherwise(pl.lit("clean_holdout")).alias("segment"))
    per_week = wk.group_by(["fri", "segment", "kind", "monthly"]).agg(pl.col("d_pre", "d_post", "e_pre", "e_post", "ctrl").mean(), pl.len().alias("n_sym")).sort("fri")
    out: dict = {"n_weeks": {}, "c1": {}, "c2": {}, "quarterly": {}, "per_symbol": {}}
    for seg in (*SEGS, "pooled"):
        s = per_week if seg == "pooled" else per_week.filter(pl.col("segment") == seg)
        out["n_weeks"][seg] = int(s.height)
        out["c1"][seg] = {"pre": _nw(s["d_pre"].to_numpy()), "post": _nw(s["d_post"].to_numpy()),
                          "mean_e_pre": float(s["e_pre"].mean()) if s.height else None, "mean_e_post": float(s["e_post"].mean()) if s.height else None,
                          "mean_ctrl": float(s["ctrl"].mean()) if s.height else None}
        out["c2"][seg] = {k: _welch(s.filter(pl.col("monthly"))[k].to_numpy(), s.filter(~pl.col("monthly"))[k].to_numpy()) for k in ("e_pre", "e_post", "d_pre", "d_post")}
        out["quarterly"][seg] = {k: _welch(s.filter(pl.col("kind") == "quarterly")[k].to_numpy(), s.filter(pl.col("kind") != "quarterly")[k].to_numpy()) for k in ("e_pre", "e_post")}
        for sym in ("BTCUSDT", "ETHUSDT"):
            ss = wk.filter(pl.col("symbol") == sym)
            ss = ss if seg == "pooled" else ss.filter(pl.col("segment") == seg)
            out["per_symbol"].setdefault(sym, {})[seg] = {"pre": _nw(ss["d_pre"].to_numpy()), "post": _nw(ss["d_post"].to_numpy())}
    c = out["c1"]
    out["verdicts"] = {"llm_in_sample": verdict([(c["llm_in_sample"]["pre"]["mean"], c["llm_in_sample"]["pre"]["t"], -1),
                                                 (c["llm_in_sample"]["post"]["mean"], c["llm_in_sample"]["post"]["t"], 1)], True),
                       "clean_holdout": verdict([(c["clean_holdout"]["pre"]["mean"], c["clean_holdout"]["pre"]["t"], -1),
                                                 (c["clean_holdout"]["post"]["mean"], c["clean_holdout"]["post"]["t"], 1)], False)}
    out["p_primary"] = max(p_one(c["llm_in_sample"]["pre"]["t"], -1), p_one(c["llm_in_sample"]["post"]["t"], 1))
    out["primary_segment"] = "llm_in_sample"
    return out


def _welch(a: np.ndarray, b: np.ndarray) -> dict:
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    if len(a) < 3 or len(b) < 3:
        return {"diff": float("nan"), "t": float("nan"), "n_a": len(a), "n_b": len(b)}
    se = math.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
    d = float(a.mean() - b.mean())
    return {"diff": d, "t": d / se if se > 0 else float("nan"), "n_a": len(a), "n_b": len(b)}


# ------------------------------------------------------------------ H-NS3 session seasonality


def _hac_cov(M: np.ndarray) -> np.ndarray:
    n = M.shape[0]
    E = M - M.mean(axis=0)
    lag = int(np.floor(4 * (n / 100) ** (2 / 9)))
    S = E.T @ E / n
    for k in range(1, lag + 1):
        G = E[k:].T @ E[:-k] / n
        S += (1 - k / (lag + 1)) * (G + G.T)
    return S


def wald_equal_means(M: np.ndarray) -> dict:
    from scipy import stats

    n, k = M.shape
    C = np.column_stack([np.eye(k - 1), -np.ones(k - 1)])  # mean_i - mean_last
    m = M.mean(axis=0)
    V = C @ (_hac_cov(M) / n) @ C.T
    cm = C @ m
    W = float(cm @ np.linalg.solve(V, cm))
    F = W / (k - 1)
    return {"F": F, "df1": k - 1, "df2": n - (k - 1), "p": float(stats.f.sf(F, k - 1, n - (k - 1))), "n_dates": int(n)}


def ns3(frames: dict[str, pl.DataFrame]) -> dict:
    from scipy import stats

    zs = []
    for sym in UNIVERSE_L:
        f = frames.get(sym)
        if f is None or f.height == 0:
            continue
        zs.append(f.select("event_ts", "z").drop_nulls().filter(pl.col("z").is_finite()))
    ts = pl.concat(zs).group_by("event_ts").agg(pl.col("z").mean()).sort("event_ts")
    h = ts["event_ts"].dt.hour().to_numpy()
    ts = ts.with_columns(pl.Series("session", sz.crypto_session(h).astype(str)), pl.col("event_ts").dt.date().alias("date"), seg_of(pl.col("event_ts")))
    ds = ts.group_by(["date", "segment", "session"]).agg(pl.col("z").mean()).pivot(on="session", index=["date", "segment"], values="z").drop_nulls().sort("date")
    out: dict = {"sessions": list(sz.SESSIONS), "means": {}, "wald": {}, "hour_of_day": {}}
    for seg in (*SEGS, "pooled"):
        s = ds if seg == "pooled" else ds.filter(pl.col("segment") == seg)
        M = s.select(list(sz.SESSIONS)).to_numpy()
        out["means"][seg] = {k: _nw(M[:, i]) for i, k in enumerate(sz.SESSIONS)}
        out["wald"][seg] = wald_equal_means(M)
        hh = ts if seg == "pooled" else ts.filter(pl.col("segment") == seg)
        out["hour_of_day"][seg] = (hh.group_by(pl.col("event_ts").dt.hour().alias("hour")).agg(pl.col("z").mean()).sort("hour")["z"].to_list())
    mi = np.array([out["means"]["llm_in_sample"][k]["mean"] for k in sz.SESSIONS])
    mh = np.array([out["means"]["clean_holdout"][k]["mean"] for k in sz.SESSIONS])
    out["spearman_in_vs_holdout"] = float(stats.spearmanr(mi, mh).statistic)
    best, worst = sz.SESSIONS[int(np.argmax(mi))], sz.SESSIONS[int(np.argmin(mi))]
    out["tilt"] = {"long": best, "short": worst, **tilt_strategy(frames, best, worst)}
    w = out["wald"]["llm_in_sample"]["p"]
    v_in = "supports" if w < 0.01 else ("inconclusive" if w < 0.10 else "contradicts")
    th = out["tilt"]["clean_holdout"]["mean"]
    c1, c2 = out["spearman_in_vs_holdout"] > 0.5, (np.isfinite(th) and th > 0)
    out["verdicts"] = {"llm_in_sample": v_in, "clean_holdout": "supports" if (c1 and c2) else ("contradicts" if not (c1 or c2) else "inconclusive")}
    out["p_primary"] = w
    out["primary_segment"] = "llm_in_sample"
    return out


def tilt_strategy(frames: dict[str, pl.DataFrame], best: str, worst: str) -> dict:
    daily = []
    for sym in UNIVERSE_L:
        f = frames.get(sym)
        if f is None or f.height == 0:
            continue
        f = f.filter(pl.col("sigma_next").is_finite() & (pl.col("sigma_next") > 0))
        nh = ((f["event_ts"].dt.hour().to_numpy() + 1) % 24)
        nxt = sz.crypto_session(nh).astype(str)
        sig = pl.Series(np.where(nxt == best, 1.0, np.where(nxt == worst, -1.0, 0.0)))
        p = strategy_pnl(f, sig, cost_bps=COST_BPS)
        pc = "pnl" if "pnl" in p.columns else p.columns[-1]
        daily.append(p.group_by(pl.col("event_ts").dt.date().alias("date")).agg(pl.col(pc).sum().alias("pnl"),
                                                                                 *([pl.col("gross").sum()] if "gross" in p.columns else [])).with_columns(pl.lit(sym).alias("symbol")))
    d = pl.concat(daily, how="diagonal_relaxed").group_by("date").agg(pl.col("pnl").mean(), *([pl.col("gross").mean()] if "gross" in daily[0].columns else [])).sort("date")
    d = d.with_columns(pl.when(pl.col("date") < _cut().date()).then(pl.lit("llm_in_sample")).otherwise(pl.lit("clean_holdout")).alias("segment"))
    out = {}
    for seg in (*SEGS, "pooled"):
        s = d if seg == "pooled" else d.filter(pl.col("segment") == seg)
        x = s["pnl"].to_numpy()
        r = _nw(x)
        r["sharpe"] = float(x.mean() / x.std(ddof=1) * math.sqrt(365)) if len(x) > 30 and x.std() > 0 else float("nan")
        if "gross" in s.columns:
            r["gross_mean"] = float(s["gross"].mean())
        out[seg] = r
    return out


# ------------------------------------------------------------------ H-NS4 hour-of-week vol profile


def ns4_symbol(f: pl.DataFrame, base: str = "s_pre", window_weeks: int | None = None) -> pl.DataFrame:
    r = f["ret"].to_numpy().astype(float)
    s = f[base].cast(pl.Float64).fill_null(np.nan).to_numpy()
    F0 = s * s
    with np.errstate(invalid="ignore", divide="ignore"):
        u = r * r / F0
    k = sz.hour_of_week(f["event_ts"])
    fac = sz.seasonal_factor(k, u, window_weeks=window_weeks)
    F1 = F0 * fac
    with np.errstate(invalid="ignore", divide="ignore"):
        d = r * r * (1 / F1 - 1 / F0) + np.log(F1 / F0)
    ok = np.isfinite(d) & (F0 > 0)
    return pl.DataFrame({"event_ts": f["event_ts"].filter(pl.Series(ok)), "d": d[ok], "fac": fac[ok]})


def ns4(frames: dict[str, pl.DataFrame], universes: dict[str, list[str]]) -> dict:
    out: dict = {"universes": {u: s for u, s in universes.items()}, "main": {}, "trailing26": {}, "base_rb": {}, "profile_hod": {}}
    for uni, syms in universes.items():
        for key, kw in (("main", {}), ("trailing26", {"window_weeks": 26}), ("base_rb", {"base": "s_pre_rb"})):
            parts = [ns4_symbol(frames[s], **kw).with_columns(pl.lit(s).alias("symbol")) for s in syms if frames.get(s) is not None and frames[s].height]
            d = pl.concat(parts)
            dd = d.group_by(pl.col("event_ts").dt.date().alias("date")).agg(pl.col("d").mean(), pl.len().alias("n")).sort("date").with_columns(
                pl.when(pl.col("date") < _cut().date()).then(pl.lit("llm_in_sample")).otherwise(pl.lit("clean_holdout")).alias("segment"))
            res = {}
            for seg in (*SEGS, "pooled"):
                s = dd if seg == "pooled" else dd.filter(pl.col("segment") == seg)
                res[seg] = {**_nw(s["d"].to_numpy()), "n_obs": int(s["n"].sum())}
            out[key][uni] = res
        # the profile known at AS_OF, averaged over the universe (hour of day and hour of week)
        profs = []
        for s in syms:
            f = frames.get(s)
            if f is None or f.height == 0:
                continue
            r = f["ret"].to_numpy().astype(float)
            sp = f["s_pre"].cast(pl.Float64).fill_null(np.nan).to_numpy()
            with np.errstate(invalid="ignore", divide="ignore"):
                u = r * r / (sp * sp)
            profs.append(sz.profile_asof(f["event_ts"], u, AS_OF))
        P = np.mean(profs, axis=0)
        out["profile_hod"][uni] = {"hour_of_day": [float(x) for x in P.reshape(7, 24).mean(axis=0)],
                                   "day_of_week": [float(x) for x in P.reshape(7, 24).mean(axis=1)],
                                   "top_hours_of_week": [(int(i), float(P[i])) for i in np.argsort(P)[::-1][:8]],
                                   "bottom_hours_of_week": [(int(i), float(P[i])) for i in np.argsort(P)[:8]]}
    parts = {seg: [(out["main"][u][seg]["mean"], out["main"][u][seg]["t"], -1) for u in universes] for seg in SEGS}
    out["verdicts"] = {seg: verdict(parts[seg], True) for seg in SEGS}
    out["p_primary"] = max(p_one(out["main"][u]["llm_in_sample"]["t"], -1) for u in universes)
    out["primary_segment"] = "llm_in_sample"
    return out


# ------------------------------------------------------------------ H-NS5 QQQ open -> crypto


def ns5(frames: dict[str, pl.DataFrame]) -> dict:
    q = frames["QQQUSDT"]
    days = sorted(set(q["event_ts"].dt.date().to_list()))
    b0 = pl.DataFrame({"b0": sz.nyse_open_bar(days)}, schema={"b0": pl.Datetime("us", "UTC")})
    x = b0.join(q.select(pl.col("event_ts").alias("b0"), pl.col("z").alias("x"), pl.col("ret").alias("q_ret")), on="b0", how="inner")
    rows = []
    for sym in ("BTCUSDT", "ETHUSDT"):
        f = frames[sym].select("event_ts", "ret", "sigma_risk_next", "z")
        fwd = None
        for k in (1, 2, 3):
            part = f.select((pl.col("event_ts") - pl.duration(hours=k)).alias("b0"), pl.col("ret").alias(f"r{k}"))
            fwd = part if fwd is None else fwd.join(part, on="b0", how="inner")
        e = x.join(f.select(pl.col("event_ts").alias("b0"), pl.col("sigma_risk_next").alias("s0"), pl.col("z").alias("z_same")), on="b0", how="inner").join(fwd, on="b0", how="inner")
        e = e.with_columns(((pl.col("r1") + pl.col("r2") + pl.col("r3")) / (pl.col("s0") * math.sqrt(3))).alias("y"), pl.lit(sym).alias("symbol"))
        rows.append(e)
    ev = pl.concat(rows).drop_nulls(["x", "y"]).with_columns(seg_of(pl.col("b0")), pl.col("b0").dt.date().alias("date"))
    out: dict = {"n_days": {}, "pooled_fit": {}, "per_symbol": {}, "same_hour_beta": {}}
    for seg in (*SEGS, "pooled"):
        e = ev if seg == "pooled" else ev.filter(pl.col("segment") == seg)
        out["n_days"][seg] = int(e["date"].n_unique())
        g = e["date"].cast(pl.Utf8).to_numpy()
        r = ols_cluster(e["y"].to_numpy(), e["x"].to_numpy().reshape(-1, 1), g)
        out["pooled_fit"][seg] = {k: (v[0] if isinstance(v, list) else v) for k, v in r.items()}
        rs = ols_cluster(e["z_same"].to_numpy().astype(float), e["x"].to_numpy().reshape(-1, 1), g)
        out["same_hour_beta"][seg] = {k: (v[0] if isinstance(v, list) else v) for k, v in rs.items()}
        for sym in ("BTCUSDT", "ETHUSDT"):
            es = e.filter(pl.col("symbol") == sym)
            rr = ols_cluster(es["y"].to_numpy(), es["x"].to_numpy().reshape(-1, 1), es["date"].cast(pl.Utf8).to_numpy())
            out["per_symbol"].setdefault(sym, {})[seg] = {k: (v[0] if isinstance(v, list) else v) for k, v in rr.items()}
    pf = out["pooled_fit"]
    out["verdicts"] = {seg: verdict([(pf[seg]["b"], pf[seg]["t"], 1)], True) for seg in (*SEGS, "pooled")}
    out["p_primary"] = p_one(pf["pooled"]["t"], 1)
    out["primary_segment"] = "pooled"
    out["first_date"], out["last_date"] = str(ev["date"].min()), str(ev["date"].max())
    return out


# ------------------------------------------------------------------ orchestration


def evaluate(record: bool = False) -> dict:
    from ai_trader.storage.lake import Lake

    syms = list(dict.fromkeys([*UNIVERSE_L, *T_CANDIDATES]))
    frames = {s: load_frame(s) for s in syms}
    missing = [s for s, f in frames.items() if f.height == 0]
    uni_t = [s for s in T_CANDIDATES if frames[s].height and complete_days(frames[s]) >= MIN_T_DAYS]
    excl_t = {s: (complete_days(frames[s]) if frames[s].height else 0) for s in T_CANDIDATES if s not in uni_t}
    funding = Lake().read_asof("funding", AS_OF, symbols=UNIVERSE_L, filters={"venue": "bybit"})
    t0 = _time.time()
    rep: dict = {"as_of": AS_OF.isoformat(), "cutoff": _cut().isoformat(), "missing_frames": missing,
                 "universe_L": UNIVERSE_L, "universe_T": uni_t, "excluded_T": excl_t}
    rep["H-NS1"] = ns1(frames, funding)
    print(f"NS1 done {_time.time() - t0:.0f}s", flush=True)
    rep["H-NS2"] = ns2(frames)
    print(f"NS2 done {_time.time() - t0:.0f}s", flush=True)
    rep["H-NS3"] = ns3(frames)
    print(f"NS3 done {_time.time() - t0:.0f}s", flush=True)
    rep["H-NS4"] = ns4(frames, {"L": [s for s in UNIVERSE_L if frames[s].height], "T": uni_t})
    print(f"NS4 done {_time.time() - t0:.0f}s", flush=True)
    rep["H-NS5"] = ns5(frames)
    print(f"NS5 done {_time.time() - t0:.0f}s", flush=True)
    pr = {h: rep[h]["p_primary"] for h in HIDS}
    adj = holm(pr)
    rep["holm"] = {h: {"p": pr[h], "p_holm": adj[h], "primary_segment": rep[h]["primary_segment"]} for h in HIDS}
    for h in HIDS:
        seg = rep[h]["primary_segment"]
        raw = rep[h]["verdicts"][seg]
        rep[h]["verdicts_raw"] = dict(rep[h]["verdicts"])
        if raw == "supports" and adj[h] >= 0.05:
            rep[h]["verdicts"][seg] = "inconclusive"
            rep[h]["holm_downgrade"] = True
    write_report(rep)
    if record:
        record_results(rep)
    return rep


def _jsonable(o):
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return float(o) if np.isfinite(o) else None
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (date, datetime)):
        return o.isoformat()
    return o


def report_paths() -> tuple[Path, Path]:
    d = load().root / "research" / "reports"
    return d / f"new_strategies_{REPORT_TAG}.json", d / f"new_strategies_{REPORT_TAG}.md"


def write_report(rep: dict) -> None:
    pj, _ = report_paths()
    pj.write_text(json.dumps(_jsonable(rep), indent=1), encoding="utf-8")


def _compact(h: str, rep: dict, seg: str) -> dict:
    r = rep[h]
    if h == "H-NS1":
        m = r["main"][seg]
        return {"b_pre": m["pre"]["b"], "t_pre": m["pre"]["t"], "b_post": m["post"]["b"], "t_post": m["post"]["t"], "n_events": m["n_events"],
                "n_dates": m["pre"].get("n_clusters"), "x_alt": {k: r["x_alt"][seg][k]["b"] for k in ("pre", "post")},
                "x_alt_t": {k: r["x_alt"][seg][k]["t"] for k in ("pre", "post")},
                "placebo_b": {k: r["placebo_mid_interval"][seg][k]["b"] for k in ("pre", "post")},
                "placebo_t": {k: r["placebo_mid_interval"][seg][k]["t"] for k in ("pre", "post")}}
    if h == "H-NS2":
        c = r["c1"][seg]
        return {"d_pre": c["pre"]["mean"], "t_pre": c["pre"]["t"], "d_post": c["post"]["mean"], "t_post": c["post"]["t"], "n_weeks": r["n_weeks"][seg],
                "monthly_minus_other_e_pre": r["c2"][seg]["e_pre"], "monthly_minus_other_e_post": r["c2"][seg]["e_post"]}
    if h == "H-NS3":
        return {"session_means": {k: r["means"][seg][k]["mean"] for k in sz.SESSIONS}, "session_t": {k: r["means"][seg][k]["t"] for k in sz.SESSIONS},
                "F": r["wald"][seg]["F"], "p": r["wald"][seg]["p"], "n_dates": r["wald"][seg]["n_dates"], "spearman_in_vs_holdout": r["spearman_in_vs_holdout"],
                "tilt": {"long": r["tilt"]["long"], "short": r["tilt"]["short"], **r["tilt"][seg]}}
    if h == "H-NS4":
        return {u: {"diff": r["main"][u][seg]["mean"], "t": r["main"][u][seg]["t"], "n_dates": r["main"][u][seg]["n"], "n_obs": r["main"][u][seg]["n_obs"],
                    "trailing26_diff": r["trailing26"][u][seg]["mean"], "trailing26_t": r["trailing26"][u][seg]["t"],
                    "base_rb_diff": r["base_rb"][u][seg]["mean"], "base_rb_t": r["base_rb"][u][seg]["t"]} for u in r["main"]} | {"n_T": len(r["universes"]["T"])}
    if h == "H-NS5":
        p = r["pooled_fit"][seg]
        return {"slope": p["b"], "t": p["t"], "n_obs": p["n"], "n_days": r["n_days"][seg],
                "btc": {k: r["per_symbol"]["BTCUSDT"][seg].get(k) for k in ("b", "t")}, "eth": {k: r["per_symbol"]["ETHUSDT"][seg].get(k) for k in ("b", "t")},
                "same_hour_beta": r["same_hour_beta"][seg]["b"]}
    return {}


NOTES = {
    "H-NS1": "universe L, 8h settlements (00/08/16 UTC) mapped per symbol from the lake funding table; x = sign(rate) |fz|, fz vs trailing 90d prints known 1h before T; pre/post = robust-sigma-scaled 1h returns; OLS, date-clustered SE.",
    "H-NS2": "BTC+ETH; 08:00-anchored 24h windows, per-asset log-HAR residual log(RV/F); C1: Thu08-Fri08 (pre) and Fri08-Sat08 (post) minus mean of Tue08-Wed08 and Wed08-Thu08 of the same week, NW t over weeks; post is biased down by Saturday morning.",
    "H-NS3": "universe L; z = r / robust sigma; per-timestamp then per date-session means; HAC Wald F; tilt long best / short worst in-sample session, 7.5 bps per side.",
    "H-NS4": "universes L and T (TradFi >= 120 complete days); expanding hour-of-week factor (kappa 20, 3h triangular kernel, cap 50, floor 0.1, >= 8 weeks); paired per-bar QLIKE vs sigma_risk_next, date means, NW t.",
    "H-NS5": "QQQ perp bar containing 09:30 ET (DST-aware, NYSE holidays dropped) -> BTC/ETH next 3h vol-scaled return; pooled OLS, date-clustered SE; QQQ 1h bars from 2026-06-06 (burn-in), holdout-dominated.",
}


def record_results(rep: dict) -> None:
    from ai_trader.research.hypotheses import record, results

    for h in HIDS:
        done = {r["segment"] for r in results(h)}
        segs = (*SEGS, "pooled") if h == "H-NS5" else SEGS
        for seg in segs:
            if seg in done:
                print(f"{h} {seg}: already recorded, skipped", flush=True)
                continue
            m = _compact(h, rep, seg) | {"holm_p": rep["holm"][h]["p_holm"], "primary_segment": rep[h]["primary_segment"]}
            extra = " Holm-downgraded from supports." if rep[h].get("holm_downgrade") and seg == rep[h]["primary_segment"] else ""
            record(h, seg, rep[h]["verdicts"][seg], _jsonable(m), note=NOTES[h] + extra, data=f"research/reports/new_strategies_{REPORT_TAG}.json")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="new_strategies_study")
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--workers", type=int, default=4)
    b.add_argument("--refresh", action="store_true")
    e = sub.add_parser("evaluate")
    e.add_argument("--record", action="store_true")
    a = ap.parse_args(argv)
    if a.cmd == "build":
        build(a.workers, a.refresh)
    else:
        rep = evaluate(a.record)
        for h in HIDS:
            print(h, rep[h]["verdicts"], "holm", round(rep["holm"][h]["p_holm"], 4), flush=True)


if __name__ == "__main__":
    main()
