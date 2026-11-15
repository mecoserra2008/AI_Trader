"""Volatility-model tournament (H-VM1..H-VM5, program 2026-11-10, agent A2).

    PYTHONIOENCODING=utf-8 OMP_NUM_THREADS=1 uv run python -m ai_trader.research.vol_tournament_study build [--workers 4]
    PYTHONIOENCODING=utf-8 uv run python -m ai_trader.research.vol_tournament_study evaluate [--record]

build:    per symbol (process pool, <= 4 workers) daily realised measures (features.realized, 1h bars, complete UTC days)
          on a gap-aware calendar plus every per-asset forecast; then the panel models on the common day grid.
          Cache: data/research/vol_tournament/sym_<SYMBOL>.parquet and forecasts.parquet.
evaluate: paired loss differences (date-averaged, NW t), MCS per universe / horizon / segment, Holm within the VM block,
          the report (research/reports/vol_tournament_2026-11-10.{json,md}) and, with --record, the registry rows.

Every forecast is made at the end of UTC day t (known_ts = t + 1 00:00) for RV_{t+1} (h = 1) or mean RV_{t+1..t+5}
(h = 5). Segment = llm_in_sample if known_ts < research.knowledge_cutoff else clean_holdout (a 5d window starting in the
last 4 in-sample days overlaps the cutoff by <= 4 days; immaterial).

robust_bc aggregation (documented choice): at the last 1h bar of day t (hour 23, so the day is complete),
  robust_bc   = 24 * sigma_risk_next^2               (risk version, no weekend shrink: what sizing uses; = har_vs_robust)
  robust_bc_c = sigma_risk_next^2 * (n_wd + m^2 n_wk) over the 24 (resp. 120, then / 5) hours of the target window,
                n_wk the TradFi-weekend hours from the known calendar, m = wk_m at that bar (= tools/vol.robust_bc central)
For h = 5 both are per-day means, so they are flat (risk) or calendar-shaped (central) term structures.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl

from ai_trader.core.config import load
from ai_trader.features import vol_panel as vp
from ai_trader.research.trend_zoo_study import UNIVERSE as UNIVERSE_L

AS_OF = datetime(2026, 11, 10, tzinfo=UTC)
T_CANDIDATES = ("XAUUSDT XAGUSDT CLUSDT BZUSDT TSLAUSDT INTCUSDT GOOGLUSDT AAPLUSDT MSTRUSDT NVDAUSDT MUUSDT QQQUSDT "
                "SOXLUSDT SPCXUSDT PAXGUSDT XAUTUSDT").split()
MIN_T_DAYS = 120
HORIZONS = (1, 5)
REFIT = 30
MIN_DAYS = 180
PANEL_START = 60
PANEL_MIN_PAIRS = 30
SEGS = ("llm_in_sample", "clean_holdout")
REPORT_TAG = "2026-11-10"

PER_ASSET = ["ewma", "robust_bc", "robust_bc_c", "garch", "gjr", "egarch", "sv_r2", "sv_r2_cal", "sv_rv",
             "har", "harq", "shar", "harj", "har_gk", "har_pk", "har_rvgk", "loghar"]
PANEL = [f"p_{m}" for m in (*vp.PANEL_LEVEL, *vp.PANEL_LOG)]
MODELS = PER_ASSET + PANEL
SHORT_SET = ["ewma", "robust_bc", "robust_bc_c", *PANEL]  # need no 180-day own history
EXTENSIONS = {"sv_r2_cal", "sv_rv", "robust_bc_c", "har_pk", "har_rvgk"}  # beyond the minimum registered list


def out_dir() -> Path:
    d = load().root / "data" / "research" / "vol_tournament"
    d.mkdir(parents=True, exist_ok=True)
    return d


# ------------------------------------------------------------------ inputs


def calendar_daily(symbol: str, as_of: datetime = AS_OF) -> pl.DataFrame:
    """Gap-aware daily calendar: realised measures (complete days only, else null) + robust_bc state at hour 23."""
    from ai_trader.features.realized import daily_measures
    from ai_trader.research.calendar import add_calendar
    from ai_trader.research.frame import market_frame

    f = market_frame(symbol, "1h", as_of, funding=False, oi=False)
    if f.height == 0:
        return pl.DataFrame()
    d = daily_measures(f.select("event_ts", "open", "high", "low", "close"), "1h").filter(pl.col("known_ts") <= as_of)
    if d.height == 0:
        return pl.DataFrame()
    last = (f.sort("event_ts").with_columns(pl.col("event_ts").dt.date().alias("date")).group_by("date", maintain_order=True)
            .agg(pl.col("sigma_risk_next").last().alias("sig_risk"), pl.col("wk_m").last().alias("wk_m"),
                 pl.col("event_ts").last().dt.hour().alias("last_h")))
    d = d.drop("known_ts").join(last, on="date", how="left").with_columns(
        pl.when(pl.col("last_h") == 23).then(pl.col("sig_risk")).otherwise(None).alias("sig_risk"))
    cal = pl.DataFrame({"date": pl.date_range(d["date"].min(), d["date"].max(), "1d", eager=True)})
    d = cal.join(d, on="date", how="left").sort("date")
    t0 = datetime.combine(d["date"].min(), datetime.min.time(), tzinfo=UTC)
    t1 = datetime.combine(d["date"].max(), datetime.min.time(), tzinfo=UTC) + timedelta(days=7)
    hrs = pl.DataFrame({"event_ts": pl.datetime_range(t0, t1, "1h", time_zone="UTC", eager=True)})
    wk = (add_calendar(hrs).with_columns(pl.col("event_ts").dt.date().alias("date")).group_by("date")
          .agg(pl.col("tradfi_weekend").cast(pl.Int32).sum().alias("n_wk")).sort("date"))
    d = d.join(wk, on="date", how="left")
    nwk = wk["n_wk"].to_numpy().astype(float)
    wd = {dt: i for i, dt in enumerate(wk["date"].to_list())}
    idx = np.array([wd[x] for x in d["date"].to_list()])
    c = np.concatenate([[0.0], np.cumsum(nwk)])
    d = d.with_columns(pl.Series("n_wk1", c[idx + 2] - c[idx + 1]), pl.Series("n_wk5", c[idx + 6] - c[idx + 1]),
                       (pl.col("date").cast(pl.Datetime("us", "UTC")) + pl.duration(days=1)).alias("known_ts"))
    return d.with_columns(pl.lit(symbol).alias("symbol"))


def _np(d: pl.DataFrame, c: str) -> np.ndarray:
    return d[c].cast(pl.Float64).fill_null(np.nan).to_numpy().astype(float)


def symbol_forecasts(symbol: str, universe: str, refresh: bool = False) -> str:
    """Worker: per-asset forecasts for one symbol, cached to sym_<symbol>.parquet. Returns the path ('' if no data)."""
    path = out_dir() / f"sym_{symbol}.parquet"
    if path.exists() and not refresh:
        return str(path)
    t_start = time.time()
    d = calendar_daily(symbol)
    if d.height == 0:
        return ""
    n = d.height
    a = {k: _np(d, k) for k in ("rv", "rs_pos", "rs_neg", "rq", "jv", "gk", "pk", "ret", "sig_risk", "wk_m", "n_wk1", "n_wk5")}
    rv = a["rv"]
    cols: dict[str, np.ndarray] = {"y1": vp.forward_mean(rv, 1), "y5": vp.forward_mean(rv, 5)}
    e = vp.ewma_daily(a["ret"])
    s2 = a["sig_risk"] ** 2
    m2 = a["wk_m"] ** 2
    cols |= {"F1_ewma": e, "F5_ewma": e, "F1_robust_bc": 24 * s2, "F5_robust_bc": 24 * s2,
             "F1_robust_bc_c": s2 * (24 - a["n_wk1"] + m2 * a["n_wk1"]), "F5_robust_bc_c": s2 * (120 - a["n_wk5"] + m2 * a["n_wk5"]) / 5}
    # return-based models on observed days only (gap-free series), mapped back to the calendar
    obs = np.flatnonzero(np.isfinite(a["ret"]) & np.isfinite(rv))
    rc = a["ret"][obs]
    for kind in vp.ARCH_SPECS:
        fc = vp.arch_forecast(rc, kind, HORIZONS, refit_every=REFIT, min_days=MIN_DAYS)
        for h in HORIZONS:
            z = np.full(n, np.nan)
            z[obs] = fc[h]
            cols[f"F{h}_{kind}"] = z
    sv = vp.sv_forecast(vp.fuller_log_r2(rc), "r2", HORIZONS, REFIT, MIN_DAYS, calibrate_target=rc * rc)
    svr = vp.sv_forecast(vp.safe_log(rv[obs]), "rv", HORIZONS, REFIT, MIN_DAYS)
    for h in HORIZONS:
        for key, src in ((f"F{h}_sv_r2", sv[h]), (f"F{h}_sv_r2_cal", sv[-h]), (f"F{h}_sv_rv", svr[h])):
            z = np.full(n, np.nan)
            z[obs] = src
            cols[key] = z
    for name in (*vp.LEVEL_MODELS, *vp.LOG_MODELS):
        X, _ = vp.design(name, a)
        lg = vp.is_log(name)
        for h in HORIZONS:
            cols[f"F{h}_{name}"] = vp.per_asset_forecast(X, vp.target(rv, h, lg), h, log=lg, refit_every=REFIT, min_days=MIN_DAYS, first=0)
    out = d.with_columns(pl.lit(universe).alias("universe"), *[pl.Series(k, v) for k, v in cols.items()])
    out.write_parquet(path)
    print(f"{symbol}: {n} days, {time.time() - t_start:.0f}s", flush=True)
    return str(path)


# ------------------------------------------------------------------ panel step


def add_panel(df: pl.DataFrame) -> pl.DataFrame:
    """Panel forecasts on the common day grid (pooled over L u T, symbol FE); crypto factor = mean log RV over L."""
    g0 = df["date"].min()
    G = (df["date"].max() - g0).days + 1
    syms = df["symbol"].unique(maintain_order=True).to_list()
    per = {s: df.filter(pl.col("symbol") == s).sort("date") for s in syms}
    off = {s: (per[s]["date"][0] - g0).days for s in syms}
    cols = ("rv", "rs_pos", "rs_neg", "rq", "jv", "gk", "pk")
    A: dict[str, dict[str, np.ndarray]] = {}
    for s in syms:
        A[s] = {}
        for c in cols:
            z = np.full(G, np.nan)
            v = _np(per[s], c)
            z[off[s]: off[s] + len(v)] = v
            A[s][c] = z
        A[s]["is_crypto"] = per[s]["universe"][0] == "L"
    lmat = np.vstack([vp.safe_log(A[s]["rv"]) for s in syms if A[s]["is_crypto"]])
    cnt = np.isfinite(lmat).sum(axis=0)
    with np.errstate(invalid="ignore"):
        f = np.where(cnt >= 5, np.nanmean(np.where(np.isfinite(lmat), lmat, np.nan), axis=0), np.nan)
    for s in syms:
        A[s]["f"] = f
    new: dict[str, dict[str, np.ndarray]] = {s: {} for s in syms}
    for name in (*vp.PANEL_LEVEL, *vp.PANEL_LOG):
        lg = vp.is_log(name)
        des = [vp.design(name, A[s]) for s in syms]
        for h in HORIZONS:
            ys = [vp.target(A[s]["rv"], h, lg) for s in syms]
            F = vp.panel_forecast([x for x, _ in des], ys, h, log=lg, powers=des[0][1], refit_every=REFIT, start=PANEL_START,
                                  min_pairs=PANEL_MIN_PAIRS)
            for i, s in enumerate(syms):
                new[s][f"F{h}_p_{name}"] = F[i, off[s]: off[s] + per[s].height]
    fac = pl.DataFrame({"date": pl.date_range(g0, df["date"].max(), "1d", eager=True), "factor": f})
    out = [per[s].with_columns(*[pl.Series(k, v) for k, v in new[s].items()],
                               pl.int_range(0, per[s].height).alias("hist_days")) for s in syms]
    return pl.concat(out, how="vertical_relaxed").join(fac, on="date", how="left")


def build(workers: int = 4, refresh: bool = False) -> pl.DataFrame:
    jobs = [(s, "L") for s in UNIVERSE_L] + [(s, "T") for s in T_CANDIDATES]
    paths = []
    with ProcessPoolExecutor(max_workers=min(workers, 4)) as ex:
        futs = {ex.submit(symbol_forecasts, s, u, refresh): s for s, u in jobs}
        for fu in as_completed(futs):
            p = fu.result()
            if p:
                paths.append(p)
    frames = [pl.read_parquet(p) for p in paths]
    keep = []
    for fr in frames:
        n_days = int(fr["rv"].is_not_null().sum())
        if fr["universe"][0] == "T" and n_days < MIN_T_DAYS:
            print(f"exclude {fr['symbol'][0]}: {n_days} complete days < {MIN_T_DAYS}", flush=True)
            continue
        keep.append(fr)
    df = add_panel(pl.concat(keep, how="vertical_relaxed"))
    cut = _cutoff()
    df = df.with_columns(pl.when(pl.col("known_ts") < cut).then(pl.lit("llm_in_sample")).otherwise(pl.lit("clean_holdout")).alias("segment"))
    df.write_parquet(out_dir() / "forecasts.parquet")
    return df


def _cutoff() -> datetime:
    from ai_trader.research.evaluate import knowledge_cutoff

    return knowledge_cutoff()


# ------------------------------------------------------------------ evaluation


def _loss(df: pl.DataFrame, model: str, h: int, loss: str) -> np.ndarray:
    y = _np(df, f"y{h}")
    F = _np(df, f"F{h}_{model}")
    F = np.where(F > 0, F, np.nan)
    return vp.qlike(y, F) if loss == "qlike" else vp.mse_log(y, F)


def paired(df: pl.DataFrame, a: str, b: str, h: int, loss: str = "qlike") -> dict:
    """Date-averaged loss difference a - b on the common (symbol, date) sample; NW t on the date series."""
    from ai_trader.research.evaluate import newey_west_t

    la, lb = _loss(df, a, h, loss), _loss(df, b, h, loss)
    ok = np.isfinite(la) & np.isfinite(lb)
    if ok.sum() < 20:
        return {"diff": float("nan"), "t": float("nan"), "n_dates": int(0), "n_obs": int(ok.sum())}
    t = pl.DataFrame({"date": df["date"].filter(pl.Series(ok)), "la": la[ok], "lb": lb[ok]}).group_by("date").agg(
        pl.col("la").mean(), pl.col("lb").mean()).sort("date")
    dd = (t["la"] - t["lb"]).to_numpy()
    m, tt, nd = newey_west_t(dd)
    mb = float(t["lb"].mean())
    return {"diff": m, "t": tt, "n_dates": nd, "n_obs": int(ok.sum()), "loss_a": float(t["la"].mean()), "loss_b": mb,
            "rel": m / mb if mb else float("nan")}


def date_matrix(df: pl.DataFrame, models: list[str], h: int, loss: str) -> tuple[np.ndarray, int, int, list]:
    """(dates x models) date-averaged losses on the common sample (all models finite)."""
    Ls = np.column_stack([_loss(df, m, h, loss) for m in models])
    ok = np.isfinite(Ls).all(axis=1)
    if ok.sum() == 0:
        return np.zeros((0, len(models))), 0, 0, []
    t = pl.DataFrame({"date": df["date"].filter(pl.Series(ok)), **{m: Ls[ok, i] for i, m in enumerate(models)}})
    g = t.group_by("date").agg(*[pl.col(m).mean() for m in models]).sort("date")
    return g.select(models).to_numpy(), g.height, int(ok.sum()), g["symbol"].to_list() if "symbol" in g.columns else []


def leaderboard(df: pl.DataFrame, models: list[str], h: int, loss: str, stat: str = "Tmax", B: int = 1000) -> dict:
    L, nd, no, _ = date_matrix(df, models, h, loss)
    if nd < 30:
        return {"n_dates": nd, "n_obs": no, "rows": []}
    res = vp.mcs(L, alpha=0.10, block=10, B=B, stat=stat, seed=h)
    mean = L.mean(axis=0)
    ref = mean[models.index("har")] if "har" in models else mean[models.index("robust_bc")]
    order = np.argsort(mean)
    rows = [{"model": models[i], "loss": float(mean[i]), "ratio_vs_ref": float(mean[i] / ref), "rank": int(r + 1),
             "in_mcs90": bool(res["included"][i]), "mcs_p": float(res["pvalues"][i])} for r, i in enumerate(order)]
    n_sym = int(df.filter(pl.Series(np.isfinite(np.column_stack([_loss(df, m, h, loss) for m in models])).all(axis=1)))["symbol"].n_unique())
    return {"n_dates": nd, "n_obs": no, "n_symbols": n_sym, "ref": "har" if "har" in models else "robust_bc", "stat": stat, "rows": rows,
            "mcs": [r["model"] for r in rows if r["in_mcs90"]]}


def _seg(df: pl.DataFrame, seg: str) -> pl.DataFrame:
    return df if seg == "pooled" else df.filter(pl.col("segment") == seg)


def _verdict(parts: list[dict]) -> str:
    """supports: every part diff < 0 with t < -2; contradicts: any part diff > 0 with t > 2, or no part with diff < 0;
    inconclusive otherwise."""
    ds = [p.get("diff", float("nan")) for p in parts]
    ts = [p.get("t", float("nan")) for p in parts]
    if all(np.isfinite(d) and d < 0 and np.isfinite(t) and t < -2 for d, t in zip(ds, ts, strict=True)):
        return "supports"
    if any(np.isfinite(d) and d > 0 and np.isfinite(t) and t > 2 for d, t in zip(ds, ts, strict=True)) or all(not (np.isfinite(d) and d < 0) for d in ds):
        return "contradicts"
    return "inconclusive"


def _p_one_sided(t: float) -> float:
    return 0.5 * math.erfc(-t / math.sqrt(2)) if np.isfinite(t) else 1.0  # P(Z <= t): small when t << 0


def _hist_bucket(df: pl.DataFrame) -> pl.DataFrame:
    return df.with_columns(pl.when(pl.col("hist_days") < 180).then(pl.lit("<180d")).when(pl.col("hist_days") < 365).then(pl.lit("180-364d"))
                           .when(pl.col("hist_days") < 730).then(pl.lit("365-729d")).otherwise(pl.lit(">=730d")).alias("hist"))


def factor_r2(df: pl.DataFrame) -> dict:
    """Per alt (L minus BTC), per segment: R^2 of log RV_t on the factor (inclusive, and leave-one-out)."""
    L = df.filter(pl.col("universe") == "L").with_columns(pl.col("rv").log().alias("lrv"))
    wide = L.pivot(on="symbol", index="date", values="lrv").sort("date")
    syms = [c for c in wide.columns if c != "date"]
    M = wide.select(syms).to_numpy().astype(float)
    cnt = np.isfinite(M).sum(axis=1)
    tot = np.nansum(M, axis=1)
    cut = _cutoff().date()
    seg = np.where(np.array([d < cut for d in wide["date"].to_list()]), "llm_in_sample", "clean_holdout")
    out: dict = {}
    for sg in SEGS:
        r2i, r2l = {}, {}
        for j, s in enumerate(syms):
            if s == "BTCUSDT":
                continue
            x = M[:, j]
            f_inc = np.where(cnt >= 5, tot / np.maximum(cnt, 1), np.nan)
            f_loo = np.where(cnt - 1 >= 5, (tot - np.nan_to_num(x)) / np.maximum(cnt - 1, 1), np.nan)
            for f, store in ((f_inc, r2i), (f_loo, r2l)):
                ok = (seg == sg) & np.isfinite(x) & np.isfinite(f)
                if ok.sum() < 60:
                    continue
                store[s] = float(np.corrcoef(x[ok], f[ok])[0, 1] ** 2)
        out[sg] = {"median_r2": float(np.median(list(r2i.values()))) if r2i else float("nan"),
                   "median_r2_loo": float(np.median(list(r2l.values()))) if r2l else float("nan"),
                   "min_r2": float(min(r2i.values())) if r2i else float("nan"), "n_alts": len(r2i), "per_alt": r2i, "per_alt_loo": r2l}
    return out


def evaluate(df: pl.DataFrame, record: bool = False, B: int = 1000) -> dict:
    df = _hist_bucket(df)
    rep: dict = {"as_of": AS_OF.isoformat(), "cutoff": _cutoff().date().isoformat(), "models": MODELS, "extensions": sorted(EXTENSIONS),
                 "universe_L": sorted(df.filter(pl.col("universe") == "L")["symbol"].unique().to_list()),
                 "universe_T": sorted(df.filter(pl.col("universe") == "T")["symbol"].unique().to_list())}
    U = {"L": df.filter(pl.col("universe") == "L"), "T": df.filter(pl.col("universe") == "T"), "LT": df}
    alts = U["L"].filter(pl.col("symbol") != "BTCUSDT")

    # coverage: forecast-days per model per universe / segment
    cov = {}
    for u in ("L", "T"):
        for sg in SEGS:
            sub = _seg(U[u], sg)
            cov[f"{u}|{sg}"] = {m: int(np.isfinite(_np(sub, f"F1_{m}")).sum()) for m in MODELS} | {"rows_with_target": int(np.isfinite(_np(sub, "y1")).sum())}
    rep["coverage_1d"] = cov

    # ---- paired comparisons
    def comp(u: str, a: str, b: str, h: int, sg: str, loss: str = "qlike", frame: pl.DataFrame | None = None) -> dict:
        return paired(_seg(frame if frame is not None else U[u], sg), a, b, h, loss)

    hyp: dict = {}
    # H-VM1
    v1: dict = {}
    for sg in SEGS:
        prim = {f"h{h}": comp("LT", "p_har", "har", h, sg) for h in HORIZONS}
        sec = {f"{u}|{a}-{b}|h{h}": comp(u, a, b, h, sg) for u in ("L", "T", "LT") for a, b in (("p_har", "har"), ("p_loghar", "loghar")) for h in HORIZONS}
        msel = {f"{u}|{a}-{b}|h{h}": comp(u, a, b, h, sg, "mse_log") for u in ("LT",) for a, b in (("p_har", "har"), ("p_loghar", "loghar")) for h in HORIZONS}
        hist = {}
        for bk in ("<180d", "180-364d", "365-729d", ">=730d"):
            fr = _seg(df, sg).filter(pl.col("hist") == bk)
            for h in HORIZONS:
                hist[f"{bk}|p_har-har|h{h}"] = paired(fr, "p_har", "har", h)
                hist[f"{bk}|p_loghar-loghar|h{h}"] = paired(fr, "p_loghar", "loghar", h)
                hist[f"{bk}|p_loghar-robust_bc|h{h}"] = paired(fr, "p_loghar", "robust_bc", h)
                hist[f"{bk}|p_har-ewma|h{h}"] = paired(fr, "p_har", "ewma", h)
        v1[sg] = {"primary": prim, "verdict": _verdict(list(prim.values())), "secondary": sec, "mse_log": msel, "by_history": hist}
    hyp["H-VM1"] = v1
    # H-VM2
    v2: dict = {}
    for sg in SEGS:
        prim = {f"{a}-{b}": comp("LT", a, b, 1, sg) for a, b in (("harq", "har"), ("loghar", "har"), ("p_harq", "p_har"), ("p_loghar", "p_har"))}
        sec = {f"{u}|{a}-{b}|h{h}": comp(u, a, b, h, sg) for u in ("L", "T") for a, b in (("harq", "har"), ("loghar", "har"), ("p_harq", "p_har"), ("p_loghar", "p_har")) for h in HORIZONS}
        msel = {f"{a}-{b}": comp("LT", a, b, 1, sg, "mse_log") for a, b in (("harq", "har"), ("loghar", "har"), ("p_harq", "p_har"), ("p_loghar", "p_har"))}
        v2[sg] = {"primary": prim, "verdict": _verdict(list(prim.values())), "secondary": sec, "mse_log": msel}
    hyp["H-VM2"] = v2
    # H-VM3
    v3: dict = {}
    for sg in SEGS:
        prim = {f"{a}-{b}|h{h}": comp("LT", a, b, h, sg) for a, b in (("har_gk", "har"), ("har_rvgk", "har"), ("har_rvgk", "har_gk")) for h in HORIZONS}
        sec = {f"{u}|{a}-{b}|h{h}": comp(u, a, b, h, sg) for u in ("L", "T", "LT") for a, b in (("har_pk", "har"), ("p_har_gk", "p_har"), ("p_har_rvgk", "p_har"), ("har_gk", "har")) for h in HORIZONS}
        v3[sg] = {"primary": prim, "verdict": _verdict(list(prim.values())), "secondary": sec,
                  "verdict_gk_only": _verdict([prim["har_gk-har|h1"], prim["har_gk-har|h5"]])}
    hyp["H-VM3"] = v3
    # H-VM4
    r2 = factor_r2(df)
    v4: dict = {}
    for sg in SEGS:
        q = paired(_seg(alts, sg), "p_loghar_f", "p_loghar", 1)
        sec = {f"{u}|h{h}": comp(u, "p_loghar_f", "p_loghar", h, sg) for u in ("L", "T") for h in HORIZONS} | {"alts|h5": paired(_seg(alts, sg), "p_loghar_f", "p_loghar", 5)}
        r2_ok = np.isfinite(r2[sg]["median_r2"]) and r2[sg]["median_r2"] >= 0.5
        vq = _verdict([q])
        verdict = "supports" if (r2_ok and vq == "supports") else ("contradicts" if (vq == "contradicts" or (np.isfinite(r2[sg]["median_r2"]) and r2[sg]["median_r2"] < 0.5 and vq != "supports")) else "inconclusive")
        v4[sg] = {"r2": {k: v for k, v in r2[sg].items() if not k.startswith("per_alt")}, "qlike_alts_h1": q, "secondary": sec, "verdict": verdict}
    rep["factor_r2_per_alt"] = {sg: {"inclusive": r2[sg]["per_alt"], "leave_one_out": r2[sg]["per_alt_loo"]} for sg in SEGS}
    hyp["H-VM4"] = v4

    # ---- leaderboards and MCS
    boards: dict = {}
    for u in ("L", "T"):
        for h in HORIZONS:
            for sg in SEGS:
                sub = _seg(U[u], sg)
                for loss in ("qlike", "mse_log"):
                    boards[f"{u}|h{h}|{sg}|{loss}|all"] = leaderboard(sub, MODELS, h, loss, B=B)
                boards[f"{u}|h{h}|{sg}|qlike|short_set"] = leaderboard(sub, SHORT_SET, h, "qlike", B=B)
                if u == "L":
                    boards[f"{u}|h{h}|{sg}|qlike|all|TR"] = leaderboard(sub, MODELS, h, "qlike", stat="TR", B=B)
    rep["leaderboards"] = boards
    # H-VM5
    v5: dict = {}
    for u in ("L", "T"):
        for h in HORIZONS:
            ins = boards[f"{u}|h{h}|llm_in_sample|qlike|all"]
            hol = boards[f"{u}|h{h}|clean_holdout|qlike|all"]
            win = ins["rows"][0]["model"] if ins["rows"] else None
            hol_tr = boards.get(f"{u}|h{h}|clean_holdout|qlike|all|TR", {})
            v5[f"{u}|h{h}"] = {"in_sample_winner": win, "winner_in_sample_loss": ins["rows"][0]["loss"] if ins["rows"] else None,
                               "in_sample_mcs": ins.get("mcs", []), "holdout_mcs": hol.get("mcs", []),
                               "winner_in_holdout_mcs": bool(win in hol.get("mcs", [])) if win and hol.get("rows") else None,
                               "winner_holdout_mcs_p": next((r["mcs_p"] for r in hol.get("rows", []) if r["model"] == win), None),
                               "winner_holdout_rank": next((r["rank"] for r in hol.get("rows", []) if r["model"] == win), None),
                               "holdout_best": hol["rows"][0]["model"] if hol.get("rows") else None,
                               "winner_in_holdout_mcs_TR": bool(win in hol_tr.get("mcs", [])) if hol_tr.get("rows") else None,
                               "n_dates_in": ins.get("n_dates"), "n_dates_hold": hol.get("n_dates"),
                               "holdout_mcs_size": len(hol.get("mcs", [])), "in_sample_mcs_size": len(ins.get("mcs", [])), "n_models": len(MODELS)}
    hyp["H-VM5"] = v5

    # ---- Holm within the VM block (intersection-union p = max over required parts; VM4 qlike part)
    holm_out = {}
    for sg in SEGS:
        ps = {"H-VM1": max(_p_one_sided(p["t"]) for p in hyp["H-VM1"][sg]["primary"].values()),
              "H-VM2": max(_p_one_sided(p["t"]) for p in hyp["H-VM2"][sg]["primary"].values()),
              "H-VM3": max(_p_one_sided(p["t"]) for p in hyp["H-VM3"][sg]["primary"].values()),
              "H-VM4": _p_one_sided(hyp["H-VM4"][sg]["qlike_alts_h1"]["t"])}
        holm_out[sg] = {"raw": ps, "holm": vp.holm(ps)}
    rep["holm"] = holm_out
    rep["hypotheses"] = hyp
    rep["recommendation"] = recommend(boards, hyp)
    rep["weekday_breakdown_1d"] = weekday_breakdown(df)
    if record:
        record_results(rep)
    return rep


def weekday_breakdown(df: pl.DataFrame, models: tuple[str, ...] = ("loghar", "p_loghar", "p_harq", "p_har", "ewma", "robust_bc", "robust_bc_c")) -> dict:
    """Descriptive (no hypothesis): next-day QLIKE and mean log(F / RV) by target day type (Sat / Sun / Mon / Tue-Fri),
    per universe and segment. Shows how much of the TradFi loss is the weekend RV drop that no contestant models."""
    d = df.with_columns((pl.col("date") + pl.duration(days=1)).dt.weekday().alias("twd"))
    d = d.with_columns(pl.when(pl.col("twd") == 6).then(pl.lit("Sat")).when(pl.col("twd") == 7).then(pl.lit("Sun"))
                       .when(pl.col("twd") == 1).then(pl.lit("Mon")).otherwise(pl.lit("Tue-Fri")).alias("tday"))
    out: dict = {}
    for u in ("L", "T"):
        for sg in SEGS:
            sub = _seg(d.filter(pl.col("universe") == u), sg)
            rows = {}
            for td in ("Sat", "Sun", "Mon", "Tue-Fri"):
                x = sub.filter(pl.col("tday") == td)
                y = _np(x, "y1")
                r = {"n": int(np.isfinite(y).sum()), "median_rv_ratio_to_tue_fri": None}
                for m in models:
                    F = _np(x, f"F1_{m}")
                    ok = np.isfinite(F) & np.isfinite(y) & (F > 0) & (y > 0)
                    r[m] = {"qlike": float(np.mean(vp.qlike(y[ok], F[ok]))) if ok.sum() else None,
                            "mean_log_F_over_RV": float(np.mean(np.log(F[ok] / y[ok]))) if ok.sum() else None}
                rows[td] = r
            base = np.nanmedian(_np(sub.filter(pl.col("tday") == "Tue-Fri"), "y1"))
            for td in rows:
                rows[td]["median_rv_ratio_to_tue_fri"] = float(np.nanmedian(_np(sub.filter(pl.col("tday") == td), "y1")) / base) if np.isfinite(base) else None
            out[f"{u}|{sg}"] = rows
    return out


def recommend(boards: dict, hyp: dict) -> dict:
    """Pre-declared rule: crypto (L) = the in-sample QLIKE winner if it is in the holdout 90% MCS, else the best in-sample
    model that is in both MCSs. TradFi (T): the crypto pick if it is in T's holdout MCS on the short-history set (panel
    and recursive models, which cover new listings), else the lowest-loss member of that MCS (selection on holdout ->
    flagged in-sample by protocol). Reported with numbers."""
    out = {}
    for h in HORIZONS:
        ins, hol = boards[f"L|h{h}|llm_in_sample|qlike|all"], boards[f"L|h{h}|clean_holdout|qlike|all"]
        if not ins.get("rows"):
            continue
        hmcs = set(hol.get("mcs", []))
        imcs = set(ins.get("mcs", []))
        pick = next((r["model"] for r in ins["rows"] if r["model"] in hmcs), ins["rows"][0]["model"])
        if pick not in imcs:
            pick = next((r["model"] for r in ins["rows"] if r["model"] in hmcs and r["model"] in imcs), pick)
        row_i = next(r for r in ins["rows"] if r["model"] == pick)
        row_h = next((r for r in hol.get("rows", []) if r["model"] == pick), {})
        out[f"crypto|h{h}"] = {"model": pick, "qlike_in": row_i["loss"], "ratio_vs_har_in": row_i["ratio_vs_ref"], "qlike_hold": row_h.get("loss"),
                               "ratio_vs_har_hold": row_h.get("ratio_vs_ref"), "in_holdout_mcs": pick in hmcs, "in_insample_mcs": pick in imcs,
                               "robust_bc_ratio_in": next((r["ratio_vs_ref"] for r in ins["rows"] if r["model"] == "robust_bc"), None),
                               "robust_bc_ratio_hold": next((r["ratio_vs_ref"] for r in hol.get("rows", []) if r["model"] == "robust_bc"), None)}
        ts = boards[f"T|h{h}|clean_holdout|qlike|short_set"]
        if ts.get("rows"):
            tm = set(ts.get("mcs", []))
            cpick = pick if pick in tm else None
            if cpick is None:
                cands = [r for r in ts["rows"] if r["model"].startswith("p_") and r["model"] in tm] or [r for r in ts["rows"] if r["model"] in tm]
                cpick = cands[0]["model"]
            rr = next(r for r in ts["rows"] if r["model"] == cpick)
            out[f"tradfi|h{h}"] = {"model": cpick, "qlike_hold": rr["loss"], "ratio_vs_robust_bc_hold": rr["ratio_vs_ref"], "in_short_set_mcs": cpick in tm,
                                   "short_set_mcs": sorted(tm), "selected_on_holdout": cpick != pick, "n_dates": ts["n_dates"], "n_symbols": ts.get("n_symbols")}
    return out


def _compact(p: dict) -> dict:
    return {k: (round(v, 5) if isinstance(v, float) else v) for k, v in p.items()}


def record_results(rep: dict) -> None:
    from ai_trader.research.hypotheses import record

    data = ("bybit 1h -> daily RV (features.realized), L = 23 crypto perps, T = " + ",".join(rep["universe_T"]) +
            f"; expanding window, refit {REFIT}d, per-asset min {MIN_DAYS}d, panel L u T with symbol FE; "
            "date-averaged paired QLIKE, NW t; cache data/research/vol_tournament/forecasts.parquet; report research/reports/vol_tournament_2026-11-10.json")
    H = rep["hypotheses"]
    holm_ = rep["holm"]
    tradfi_note = "TradFi perps are holdout-only (except PAXG/XAUT tokens from 2025)."
    for sg in SEGS:
        x = H["H-VM1"][sg]
        record("H-VM1", sg, x["verdict"], {"p_har_minus_har": {k: _compact(v) for k, v in x["primary"].items()},
                                            "p_loghar_minus_loghar_LT": {f"h{h}": _compact(x["secondary"][f"LT|p_loghar-loghar|h{h}"]) for h in HORIZONS},
                                            "holm_p": holm_[sg]["holm"]["H-VM1"]},
               note=f"primary = level panel HAR (FE, scale-normalised) vs per-asset level HAR, pooled L u T; both horizons required. {tradfi_note}", data=data)
        x = H["H-VM2"][sg]
        record("H-VM2", sg, x["verdict"], {"h1": {k: _compact(v) for k, v in x["primary"].items()}, "holm_p": holm_[sg]["holm"]["H-VM2"]},
               note=f"all four comparisons (harq-har, loghar-har, panel both) must have t < -2. {tradfi_note}", data=data)
        x = H["H-VM3"][sg]
        record("H-VM3", sg, x["verdict"], {"primary": {k: _compact(v) for k, v in x["primary"].items()}, "verdict_gk_only": x["verdict_gk_only"],
                                            "holm_p": holm_[sg]["holm"]["H-VM3"]},
               note=f"GK-HAR vs HAR and RV+GK vs each, h1 and h5, pooled L u T, per-asset level models. {tradfi_note}", data=data)
        x = H["H-VM4"][sg]
        record("H-VM4", sg, x["verdict"], {"median_r2": x["r2"]["median_r2"], "median_r2_loo": x["r2"]["median_r2_loo"], "n_alts": x["r2"]["n_alts"],
                                            "factor_panel_minus_panel_alts_h1": _compact(x["qlike_alts_h1"]), "holm_p": holm_[sg]["holm"]["H-VM4"]},
               note="22 alts (L minus BTC); factor = cross-sectional mean log RV of L (inclusive; LOO reported); factor terms d/w/m added to the panel log-HAR.", data=data)
    v5 = H["H-VM5"]
    hold = {k: {kk: v[kk] for kk in ("in_sample_winner", "winner_in_holdout_mcs", "winner_holdout_mcs_p", "winner_holdout_rank", "holdout_best", "holdout_mcs", "winner_in_holdout_mcs_TR", "holdout_mcs_size", "n_models")} for k, v in v5.items()}
    ok = [v5[f"L|h{h}"]["winner_in_holdout_mcs"] for h in HORIZONS]
    verdict = "supports" if all(ok) else "contradicts"
    record("H-VM5", "clean_holdout", verdict, hold, note="MCS Tmax, 90%, circular block bootstrap (10 days, 1000 reps) on date-averaged QLIKE, common sample of all 25 models; "
           "verdict on universe L both horizons; T reported (holdout only, common sample mostly PAXG/XAUT).", data=data)
    ins = {k: {kk: v[kk] for kk in ("in_sample_winner", "winner_in_sample_loss", "in_sample_mcs", "in_sample_mcs_size", "n_dates_in")} for k, v in v5.items()}
    record("H-VM5", "llm_in_sample", "inconclusive", ins, note="selection segment: defines the winner; the registered test is its holdout MCS membership (clean_holdout row).", data=data)


# ------------------------------------------------------------------ Markdown tables


def _f(x, nd: int = 4) -> str:
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "n/a"
    return f"{x:.{nd}f}" if isinstance(x, float) else str(x)


def md_paired(rows: dict[str, dict], title: str = "comparison") -> str:
    out = [f"| {title} | mean dQLIKE (a - b) | rel. to b | NW t | dates | obs | loss a | loss b |", "|---|---|---|---|---|---|---|---|"]
    for k, p in rows.items():
        out.append(f"| {k.replace('|', ' / ')} | {_f(p.get('diff'), 4)} | {_f(100 * p['rel'], 1) + '%' if np.isfinite(p.get('rel', float('nan'))) else 'n/a'} | "
                   f"{_f(p.get('t'), 2)} | {p.get('n_dates', 0)} | {p.get('n_obs', 0)} | {_f(p.get('loss_a'))} | {_f(p.get('loss_b'))} |")
    return "\n".join(out)


def md_board(b: dict, top: int | None = None) -> str:
    if not b.get("rows"):
        return f"_too few common dates ({b.get('n_dates', 0)})_"
    rows = b["rows"][:top] if top else b["rows"]
    out = [f"common sample: {b['n_dates']} dates, {b['n_obs']} symbol-days, {b.get('n_symbols')} symbols; ratio vs `{b['ref']}`; MCS {b['stat']} 90%",
           "", "| rank | model | mean loss | ratio | in MCS | MCS p |", "|---|---|---|---|---|---|"]
    for r in rows:
        out.append(f"| {r['rank']} | {r['model']}{' *' if r['model'] in EXTENSIONS else ''} | {_f(r['loss'])} | {_f(r['ratio_vs_ref'], 3)} | "
                   f"{'**yes**' if r['in_mcs90'] else 'no'} | {_f(r['mcs_p'], 3)} |")
    return "\n".join(out)


def md_tables(rep: dict) -> str:
    H, B = rep["hypotheses"], rep["leaderboards"]
    s = []
    for hid, key in (("H-VM1", "primary"), ("H-VM2", "primary"), ("H-VM3", "primary")):
        s.append(f"### {hid}\n")
        for sg in SEGS:
            s.append(f"**{sg}: {H[hid][sg]['verdict']}** (Holm p {_f(rep['holm'][sg]['holm'][hid], 3)})\n")
            s.append(md_paired(H[hid][sg][key]) + "\n")
        s.append("<details><summary>secondary comparisons</summary>\n")
        for sg in SEGS:
            s.append(f"{sg}\n\n" + md_paired(H[hid][sg]["secondary"]) + "\n")
            if "by_history" in H[hid][sg]:
                s.append(f"{sg}, by history length at the forecast origin\n\n" + md_paired(H[hid][sg]["by_history"]) + "\n")
            if "mse_log" in H[hid][sg]:
                s.append(f"{sg}, loss = MSE of log variance\n\n" + md_paired(H[hid][sg]["mse_log"]) + "\n")
        s.append("</details>\n")
    s.append("### H-VM4\n")
    s.append("| segment | verdict | median R^2 (inclusive) | median R^2 (leave-one-out) | min R^2 | alts |\n|---|---|---|---|---|---|")
    for sg in SEGS:
        r = H["H-VM4"][sg]["r2"]
        s.append(f"| {sg} | {H['H-VM4'][sg]['verdict']} | {_f(r['median_r2'], 3)} | {_f(r['median_r2_loo'], 3)} | {_f(r['min_r2'], 3)} | {r['n_alts']} |")
    s.append("")
    for sg in SEGS:
        s.append(f"{sg} (Holm p {_f(rep['holm'][sg]['holm']['H-VM4'], 3)})\n\n" + md_paired({"alts h1: p_loghar_f - p_loghar": H["H-VM4"][sg]["qlike_alts_h1"], **H["H-VM4"][sg]["secondary"]}) + "\n")
    s.append("### H-VM5\n")
    s.append("| universe, horizon | in-sample winner | in holdout MCS (Tmax) | holdout MCS p | holdout rank | in holdout MCS (TR) | holdout best | holdout MCS |\n|---|---|---|---|---|---|---|---|")
    for k, v in H["H-VM5"].items():
        s.append(f"| {k.replace('|', ' ')} | {v['in_sample_winner']} | {v['winner_in_holdout_mcs']} | {_f(v['winner_holdout_mcs_p'], 3)} | {v['winner_holdout_rank']} | "
                 f"{v['winner_in_holdout_mcs_TR']} | {v['holdout_best']} | {', '.join(v['holdout_mcs'])} |")
    s.append("\n## Leaderboards (QLIKE, date-averaged, common sample of all models)\n\n`*` = extension beyond the registered list.\n")
    for u in ("L", "T"):
        for h in HORIZONS:
            for sg in SEGS:
                s.append(f"### {u}, h = {h}d, {sg}\n\n" + md_board(B[f"{u}|h{h}|{sg}|qlike|all"]) + "\n")
    s.append("## TradFi short-history set (models that need no 180-day own history), QLIKE\n")
    for h in HORIZONS:
        for sg in SEGS:
            s.append(f"### T, h = {h}d, {sg}\n\n" + md_board(B[f"T|h{h}|{sg}|qlike|short_set"]) + "\n")
    s.append("## MSE of log variance (top 8 and MCS)\n")
    for u in ("L", "T"):
        for h in HORIZONS:
            for sg in SEGS:
                b = B[f"{u}|h{h}|{sg}|mse_log|all"]
                s.append(f"### {u}, h = {h}d, {sg}\n\n" + md_board(b, top=8) + f"\n\nMCS: {', '.join(b.get('mcs', []))}\n")
    return "\n".join(s)


# ------------------------------------------------------------------ CLI


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="vol_tournament_study")
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--workers", type=int, default=4)
    b.add_argument("--refresh", action="store_true")
    e = sub.add_parser("evaluate")
    e.add_argument("--record", action="store_true")
    e.add_argument("--B", type=int, default=1000)
    args = ap.parse_args(argv)
    if args.cmd == "build":
        df = build(args.workers, args.refresh)
        print(f"forecasts: {df.height} rows, {df['symbol'].n_unique()} symbols", flush=True)
    else:
        df = pl.read_parquet(out_dir() / "forecasts.parquet")
        rep = evaluate(df, record=args.record, B=args.B)
        p = load().root / "research" / "reports" / f"vol_tournament_{REPORT_TAG}.json"
        p.write_text(json.dumps(rep, indent=1, default=lambda o: o.isoformat() if hasattr(o, "isoformat") else str(o)), encoding="utf-8")
        (out_dir() / "tables.md").write_text(md_tables(rep), encoding="utf-8")
        sys.stdout.write(f"wrote {p} and {out_dir() / 'tables.md'}\n")


if __name__ == "__main__":
    main()
