"""Rough volatility and Volterra kernels (block RFSV: H-RFSV1, H-RFSV2, H-VOLT1; research agent, 2026-11-15/16).

    PYTHONIOENCODING=utf-8 OMP_NUM_THREADS=2 uv run python -m ai_trader.research.rough_vol_study build
    ... rough        H-RFSV1  structure-function roughness per token / segment / v0 vol state
    ... forecast     H-RFSV2  RFSV-GJR / RFSV-reg vs log-HAR, HAR, GARCH(1,1)-t at 1h, 4h (hourly grid) and 24h (daily grid)
    ... volterra     H-VOLT1  power-law kernel of past z vs EWMA momentum, 1h and 4h lake bars, net of costs
    ... report       research/reports/rough_vol_2026-11-16.{md,json}  [--record]

Design (registered in research/hypotheses.yaml before any data was read):
  RV      5m Bybit public klines (data/research/bos/bars/5m, universe L, 2022-01 -> 2026-11-15). Hourly RV = sum of the 5m
          squared log returns of the hour (>= 10 of 12 returns, scaled by 12 / n), daily RV the same over the UTC day (>= 260
          of 288). Returns across a gap are dropped. RV == 0 -> missing.
  Season  hourly log RV is deseasonalised by an hour-of-week profile updated weekly from all hours strictly before the week
          (expanding, >= 8 weeks, circular [1/4, 1/2, 1/4] smoothing, mean-zero shape). Inputs use the profile of their own
          week, targets the profile of the origin's week: point in time.
  Grid    hourly: origin = open of hour t, information = hours <= t-1, target hours t .. t+h-1 (h = 1, 4; 4h origins every 4h).
          daily: origin 00 UTC of day d, information = days <= d-1, target = RV of day d (24h).
  Walk    quarterly expanding refits 2023Q1 -> 2026Q4; training origins have t + h <= quarter start - 1 day; first test
          quarter with >= 365 training days. Segment by origin time vs research.knowledge_cutoff.
  Models  all forecast the deseasonalised mean level Y and get a QLIKE-optimal multiplicative constant fitted on training
          rows (F = P * mean(Y / P)):
            har      OLS of Y on [1, u_1, u_d, u_w] (hourly: last hour / 24h / 168h means; daily: 1d / 7d / 30d)
            loghar   OLS of log Y on the logs of the same three means
            rfsv     GJR rough predictor: mean over D = 1..h of sum_k w_k(H, D) lu_{t-1-k}, w_k = D^(H+1/2) /
                     ((k + 1/2 + D)(k + 1/2)^(H+1/2)), k < 500, renormalised; H = structure-function H on the training fold
            rfsv_reg OLS of log Y on [1, rfsv predictor], H chosen on the training fold by MSE from HGRID
            garch    GARCH(1,1)-t (features.vol_panel) on deseasonalised hourly returns (last 2 years of training) / daily
                     returns; h-step mean variance
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl

from ai_trader.core.config import load
from ai_trader.research.trend_zoo_study import UNIVERSE

TAG = "2026-11-16"
AS_OF = datetime(2026, 11, 15, 23, tzinfo=UTC)
SEGS = ("llm_in_sample", "clean_holdout")
QS = (0.5, 1.0, 1.5, 2.0, 3.0)
D_DAY = np.arange(1, 21)
D_HOUR = np.arange(1, 49)
L_KERNEL = 500
HGRID = (0.03, 0.06, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5)
MIN_TRAIN_DAYS = 365
N_BOOT = 300
MODELS = ("har", "loghar", "rfsv", "rfsv_reg", "garch")
HOUR = 3600


def root() -> Path:
    p = load().root / "data" / "research" / "rough_vol"
    p.mkdir(parents=True, exist_ok=True)
    return p


def cutoff() -> datetime:
    from ai_trader.research.evaluate import knowledge_cutoff

    return knowledge_cutoff()


def nw_t(x: np.ndarray, min_lag: int = 0) -> tuple[float, float, int]:
    """Mean, Newey-West t (Bartlett, lag = max(auto, min_lag)), n."""
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 10:
        return (float(np.mean(x)) if n else float("nan"), float("nan"), n)
    m = x.mean()
    e = x - m
    lag = max(int(np.floor(4 * (n / 100) ** (2 / 9))), min_lag)
    var = e @ e / n
    for k in range(1, min(lag, n - 1) + 1):
        var += 2 * (1 - k / (lag + 1)) * (e[k:] @ e[:-k]) / n
    se = math.sqrt(max(var, 1e-300) / n)
    return float(m), float(m / se), n


# ------------------------------------------------------------------ data


def load_5m(sym: str) -> pl.DataFrame:
    d = load().root / "data" / "research" / "bos" / "bars" / "5m"
    files = sorted(d.glob(f"{sym}_*.parquet"))
    df = pl.concat([pl.read_parquet(f, columns=["event_ts", "close"]) for f in files]).unique("event_ts").sort("event_ts")
    return df.filter(pl.col("event_ts") < AS_OF)


def realised(sym: str) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Hourly and daily RV on complete calendar grids (missing = null), plus hourly / daily close-to-close returns."""
    d = load_5m(sym).with_columns(pl.col("close").log().alias("lc"))
    d = d.with_columns((pl.col("lc") - pl.col("lc").shift(1)).alias("r"),
                       (pl.col("event_ts") - pl.col("event_ts").shift(1)).dt.total_seconds().alias("dt"))
    d = d.with_columns(pl.when(pl.col("dt") == 300).then(pl.col("r")).otherwise(None).alias("r"))
    out = []
    for every, full, need in (("1h", 12, 10), ("1d", 288, 260)):
        g = (d.group_by_dynamic("event_ts", every=every, closed="left", label="left")
             .agg((pl.col("r") ** 2).sum().alias("rv"), pl.col("r").is_not_null().sum().alias("n"), pl.col("lc").last().alias("lc_last")))
        g = g.with_columns(pl.when(pl.col("n") >= need).then(pl.col("rv") * full / pl.col("n")).otherwise(None).alias("rv"))
        g = g.with_columns(pl.when(pl.col("rv") > 0).then(pl.col("rv")).otherwise(None).alias("rv"))
        t0, t1 = g["event_ts"].min(), g["event_ts"].max()
        cal = pl.DataFrame({"event_ts": pl.datetime_range(t0, t1, every, time_zone="UTC", eager=True)}).with_columns(pl.col("event_ts").cast(g.schema["event_ts"]))
        g = cal.join(g, on="event_ts", how="left").sort("event_ts")
        g = g.with_columns((pl.col("lc_last") - pl.col("lc_last").shift(1)).alias("ret")).drop("lc_last")
        out.append(g)
    return out[0], out[1]


def seasonal_profile(ts: pl.Series, lrv: np.ndarray, min_weeks: int = 8) -> tuple[np.ndarray, np.ndarray]:
    """Weekly-updated expanding hour-of-week profile of log RV (mean-zero shape). Returns (P[n_weeks, 168], week index per
    hour); P[w] uses hours strictly before week w's Monday 00:00 UTC."""
    how = (ts.dt.weekday().to_numpy().astype(int) - 1) * 24 + ts.dt.hour().to_numpy().astype(int)
    monday = (ts.dt.truncate("1w")).to_numpy()
    _, wk = np.unique(monday, return_inverse=True)
    nw = wk.max() + 1
    P = np.full((nw, 168), np.nan)
    s = np.zeros(168)
    c = np.zeros(168)
    fin = np.isfinite(lrv)
    for w in range(nw):
        if c.min() >= min_weeks:
            m = s / c
            m = 0.25 * np.roll(m, 1) + 0.5 * m + 0.25 * np.roll(m, -1)
            P[w] = m - m.mean()
        sel = (wk == w) & fin
        np.add.at(s, how[sel], lrv[sel])
        np.add.at(c, how[sel], 1)
    return P, wk


def daily_vol_state(sym: str) -> pl.DataFrame:
    """Own-token daily v0 cell (regimes.label on lake 1d bars, point in time): date -> cell, vol_state of that day's close."""
    from ai_trader.research.frame import market_frame
    from ai_trader.research.regimes import label

    f = market_frame(sym, "1d", AS_OF, funding=False, oi=False)
    if f.height < 60:
        return pl.DataFrame(schema={"date": pl.Date, "cell": pl.Utf8, "vol_state": pl.Utf8})
    f = label(f)
    return f.select(pl.col("event_ts").dt.date().alias("date"), "cell", "vol_state")


def build(symbols: list[str] | None = None) -> None:
    """Per token: hourly grid (rv, lu = deseasonalised log RV, ret, profile index), daily grid, daily v0 cells. Cached."""
    for s in symbols or UNIVERSE:
        p = root() / f"rv_{s}.npz"
        if p.exists():
            continue
        t0 = time.time()
        h, d = realised(s)
        lrv = np.log(h["rv"].cast(pl.Float64).fill_null(np.nan).to_numpy())
        P, wk = seasonal_profile(h["event_ts"], lrv)
        how = (h["event_ts"].dt.weekday().to_numpy().astype(int) - 1) * 24 + h["event_ts"].dt.hour().to_numpy().astype(int)
        lu = lrv - P[wk, how]
        cells = daily_vol_state(s)
        dd = d.with_columns(pl.col("event_ts").dt.date().alias("date")).join(cells, on="date", how="left")
        np.savez_compressed(
            p, h_ts=h["event_ts"].dt.epoch("s").to_numpy(), h_rv=h["rv"].cast(pl.Float64).fill_null(np.nan).to_numpy(),
            h_ret=h["ret"].cast(pl.Float64).fill_null(np.nan).to_numpy(), h_lu=lu, h_how=how, h_wk=wk, P=P,
            d_ts=d["event_ts"].dt.epoch("s").to_numpy(), d_rv=d["rv"].cast(pl.Float64).fill_null(np.nan).to_numpy(),
            d_ret=d["ret"].cast(pl.Float64).fill_null(np.nan).to_numpy(),
            d_cell=np.array([c or "" for c in dd["cell"].to_list()]), d_vs=np.array([c or "" for c in dd["vol_state"].to_list()]))
        sys.stdout.write(f"{s}: {h.height} h, {d.height} d, {time.time() - t0:.1f}s\n")
        sys.stdout.flush()


def load_sym(s: str) -> dict:
    z = np.load(root() / f"rv_{s}.npz", allow_pickle=False)
    return {k: z[k] for k in z.files}


# ------------------------------------------------------------------ roughness (H-RFSV1)


def struct_contrib(x: np.ndarray, Ds: np.ndarray, q: float) -> np.ndarray:
    """C[t, i] = |x[t + D_i] - x[t]|^q (NaN where either is missing)."""
    T = len(x)
    C = np.full((T, len(Ds)), np.nan)
    for i, D in enumerate(Ds):
        C[: T - D, i] = np.abs(x[D:] - x[:-D]) ** q
    return C


def slope_H(m: np.ndarray, Ds: np.ndarray, q: float) -> float:
    ok = np.isfinite(m) & (m > 0)
    if ok.sum() < 3:
        return float("nan")
    b = np.polyfit(np.log(Ds[ok]), np.log(m[ok]), 1)[0]
    return float(b / q)


def hurst(x: np.ndarray, Ds: np.ndarray) -> float:
    C = struct_contrib(x, Ds, 2.0)
    return slope_H(np.nanmean(C, 0), Ds, 2.0)


def block_boot_H(C: np.ndarray, Ds: np.ndarray, block: int, B: int = N_BOOT, seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    T = C.shape[0]
    nb = max(1, T // block)
    S = np.where(np.isfinite(C), C, 0.0)
    N = np.isfinite(C).astype(float)
    cs, cn = np.vstack([np.zeros(C.shape[1]), np.cumsum(S, 0)]), np.vstack([np.zeros(C.shape[1]), np.cumsum(N, 0)])
    hs = []
    for _ in range(B):
        st = rng.integers(0, max(1, T - block), nb)
        ss = (cs[st + block] - cs[st]).sum(0)
        nn_ = (cn[st + block] - cn[st]).sum(0)
        hs.append(slope_H(ss / np.maximum(nn_, 1), Ds, 2.0))
    return float(np.nanpercentile(hs, 2.5)), float(np.nanpercentile(hs, 97.5))


def seg_mask(ts_s: np.ndarray, seg: str) -> np.ndarray:
    c = cutoff().timestamp()
    return ts_s < c if seg == "llm_in_sample" else ts_s >= c


def rough() -> dict:
    out: dict = {"tokens": {}, "pooled": {}, "by_vol_state": {}, "by_cell": {}}
    pool_state: dict = {}
    for s in UNIVERSE:
        z = load_sym(s)
        tok = {}
        for grid, Ds, block in (("daily", D_DAY, 30), ("hourly", D_HOUR, 168)):
            if grid == "daily":
                x, ts = 0.5 * np.log(z["d_rv"]), z["d_ts"]
            else:
                x, ts = 0.5 * z["h_lu"], z["h_ts"]
            g = {}
            for seg in SEGS:
                xm = np.where(seg_mask(ts, seg), x, np.nan)
                Cs = {q: struct_contrib(xm, Ds, q) for q in QS}
                ms = {q: np.nanmean(Cs[q], 0) for q in QS}
                zeta = np.array([slope_H(ms[q], Ds, q) * q for q in QS])
                H = zeta[QS.index(2.0)] / 2
                r2 = float(np.corrcoef(np.array(QS), zeta)[0, 1] ** 2) if np.all(np.isfinite(zeta)) else float("nan")
                lo, hi = block_boot_H(Cs[2.0], Ds, block, seed=hash(s) % 1000) if np.isfinite(H) else (float("nan"), float("nan"))
                # short-lag noise check: H from lags >= 5 only
                H_long = slope_H(ms[2.0][4:], Ds[4:], 2.0)
                g[seg] = {"H": float(H), "lo": lo, "hi": hi, "zeta": zeta.tolist(), "zeta_r2": r2, "H_lags5plus": H_long,
                          "n": int(np.isfinite(xm).sum()), "m2": ms[2.0].tolist()}
                if grid == "daily":
                    vs = z["d_vs"]
                    cell = z["d_cell"]
                    C2 = Cs[2.0]
                    for key, lab in (("vs", vs), ("cell", cell)):
                        for v in np.unique(lab):
                            if not v:
                                continue
                            sel = lab == v
                            a = pool_state.setdefault((seg, key, v), {"sum": np.zeros(len(Ds)), "n": np.zeros(len(Ds)), "tok": []})
                            S = np.nansum(C2[sel], 0)
                            N = np.isfinite(C2[sel]).sum(0)
                            a["sum"] += S
                            a["n"] += N
                            a["tok"].append((S, N))
            tok[grid] = g
        out["tokens"][s] = tok
        sys.stdout.write(f"{s}: H daily {tok['daily']['llm_in_sample']['H']:.3f} / {tok['daily']['clean_holdout']['H']:.3f}, "
                         f"hourly {tok['hourly']['llm_in_sample']['H']:.3f} / {tok['hourly']['clean_holdout']['H']:.3f}\n")
        sys.stdout.flush()
    for grid in ("daily", "hourly"):
        for seg in SEGS:
            hs = np.array([out["tokens"][s][grid][seg]["H"] for s in UNIVERSE])
            his = np.array([out["tokens"][s][grid][seg]["hi"] for s in UNIVERSE])
            ok = np.isfinite(hs)
            out["pooled"][f"{grid}_{seg}"] = {"median_H": float(np.nanmedian(hs)), "mean_H": float(np.nanmean(hs)),
                                              "share_hi_below_0.35": float(np.mean(his[ok] < 0.35)), "n_tokens": int(ok.sum()),
                                              "min": float(np.nanmin(hs)), "max": float(np.nanmax(hs))}
    rng = np.random.default_rng(0)
    for (seg, key, v), a in pool_state.items():
        if a["n"].min() < 200:
            continue
        H = slope_H(a["sum"] / a["n"], D_DAY, 2.0)
        toks = a["tok"]
        bs = []
        for _ in range(N_BOOT):
            pick = rng.integers(0, len(toks), len(toks))
            S = sum(toks[i][0] for i in pick)
            N = sum(toks[i][1] for i in pick)
            bs.append(slope_H(S / np.maximum(N, 1), D_DAY, 2.0))
        out["by_vol_state" if key == "vs" else "by_cell"][f"{seg}|{v}"] = {"H": H, "lo": float(np.nanpercentile(bs, 2.5)),
                                                                           "hi": float(np.nanpercentile(bs, 97.5)), "n_min": int(a["n"].min())}
    (root() / "rough.json").write_text(json.dumps(out, indent=1, default=float), encoding="utf-8")
    return out


# ------------------------------------------------------------------ forecasting (H-RFSV2)


def gjr_weights(H: float, D: int, L: int = L_KERNEL) -> np.ndarray:
    k = np.arange(L) + 0.5
    w = D ** (H + 0.5) / ((k + D) * k ** (H + 0.5))
    return w / w.sum()


def kernel_pred(x: np.ndarray, w: np.ndarray, min_mass: float = 0.8) -> np.ndarray:
    """out[t] = sum_k w_k x[t-1-k] / sum_k w_k 1{x finite}: information up to t-1; NaN if < min_mass of the weight."""
    from scipy.signal import fftconvolve

    T = len(x)
    fin = np.isfinite(x)
    num = fftconvolve(np.where(fin, x, 0.0), w)[:T]
    den = fftconvolve(fin.astype(float), w)[:T]
    out = np.full(T, np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        v = num / den
    ok = den > min_mass
    out[1:] = np.where(ok[:-1], v[:-1], np.nan)
    return out


def trailing_mean(x: np.ndarray, w: int, min_frac: float = 0.7) -> np.ndarray:
    """mean of x[t-w .. t-1] (information up to t-1)."""
    fin = np.isfinite(x)
    cs = np.r_[0.0, np.cumsum(np.where(fin, x, 0.0))]
    cn = np.r_[0, np.cumsum(fin)]
    T = len(x)
    out = np.full(T, np.nan)
    idx = np.arange(T)
    lo = np.maximum(idx - w, 0)
    n = cn[idx] - cn[lo]
    s = cs[idx] - cs[lo]
    ok = (n >= min_frac * w) & (idx >= w)
    out[ok] = s[ok] / n[ok]
    return out


def ols(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.linalg.lstsq(X, y, rcond=None)[0]


def quarters(start: datetime, end: datetime) -> list[datetime]:
    q = []
    t = datetime(start.year, 3 * ((start.month - 1) // 3) + 1, 1, tzinfo=UTC)
    while t < end:
        q.append(t)
        t = datetime(t.year + (t.month + 2) // 12, (t.month + 2) % 12 + 1, 1, tzinfo=UTC)
    q.append(t)
    return q


def grid_inputs(z: dict, grid: str, h: int) -> dict:
    """Targets and baseline features on one grid (hourly: h in hours; daily: h = 1 day)."""
    if grid == "hourly":
        rv, lu, ts = z["h_rv"], z["h_lu"], z["h_ts"]
        P, wk, how = z["P"], z["h_wk"], z["h_how"]
        T = len(rv)
        u = np.exp(lu)
        # target: sum_{j<h} rv_{t+j} / sum_j S^{(t)}_{t+j}
        num = np.zeros(T)
        den = np.zeros(T)
        okt = np.ones(T, bool)
        for j in range(h):
            r = np.r_[rv[j:], np.full(j, np.nan)]
            okt &= np.isfinite(r)
            num += np.where(np.isfinite(r), r, 0.0)
            den += np.exp(P[wk, (how + j) % 168])
        Y = np.where(okt & np.isfinite(den), num / den, np.nan)
        feats = {"u1": trailing_mean(u, 1, 1.0), "ud": trailing_mean(u, 24), "uw": trailing_mean(u, 168)}
        x_log = lu
        steps = list(range(1, h + 1))
        e = z["h_ret"] / np.exp(0.5 * P[wk, how])
    else:
        rv, ts = z["d_rv"], z["d_ts"]
        Y = rv.copy()
        feats = {"u1": trailing_mean(rv, 1, 1.0), "ud": trailing_mean(rv, 7), "uw": trailing_mean(rv, 30)}
        x_log = np.log(rv)
        steps = [1]
        e = z["d_ret"]
    return {"Y": Y, "ts": ts, "feats": feats, "x_log": x_log, "steps": steps, "e": e}


def rfsv_pred(x_log: np.ndarray, H: float, steps: list[int]) -> np.ndarray:
    return np.mean([kernel_pred(x_log, gjr_weights(H, D)) for D in steps], axis=0)


def garch_path(e: np.ndarray, train: np.ndarray, steps: list[int], max_train: int) -> np.ndarray | None:
    from ai_trader.features import vol_panel as vp

    idx = np.flatnonzero(train & np.isfinite(e))
    if len(idx) < 300:
        return None
    idx = idx[-max_train:]
    pr = vp.fit_arch(e[idx], "garch")
    if not vp.arch_sane(pr, "garch"):
        return None
    ef = np.where(np.isfinite(e), e, pr["mu"])
    s1 = vp.arch_filter(ef, pr, "garch")      # s1[t] = var of e_{t+1} given e_..t
    s1 = np.r_[np.nan, s1[:-1]]               # aligned: var of e_t given info up to t-1
    return vp.arch_multi(np.where(np.isfinite(s1), s1, np.nanmean(s1)), pr, "garch", len(steps))


def forecast_one(s: str, grid: str, h: int) -> pl.DataFrame:
    z = load_sym(s)
    g = grid_inputs(z, grid, h)
    Y, ts, F = g["Y"], g["ts"], g["feats"]
    T = len(Y)
    step_s = HOUR * (h if grid == "hourly" else 24)
    origin_ok = np.ones(T, bool)
    if grid == "hourly" and h > 1:
        origin_ok = (ts // HOUR) % h == 0
    Xl = np.column_stack([np.ones(T), F["u1"], F["ud"], F["uw"]])
    with np.errstate(divide="ignore", invalid="ignore"):
        Xg = np.column_stack([np.ones(T), np.log(F["u1"]), np.log(F["ud"]), np.log(F["uw"])])
        lY = np.log(Y)
    base_ok = np.all(np.isfinite(Xl), 1) & np.all(np.isfinite(Xg), 1)
    t_first = datetime.fromtimestamp(int(ts[np.flatnonzero(np.isfinite(Y))[0]]), UTC)
    qs = quarters(datetime(2023, 1, 1, tzinfo=UTC), AS_OF)
    xs = g["x_log"]
    Ds = D_HOUR if grid == "hourly" else D_DAY
    rows = []
    preds_cache: dict[float, np.ndarray] = {}

    def pred_H(H: float) -> np.ndarray:
        key = round(H, 4)
        if key not in preds_cache:
            preds_cache[key] = rfsv_pred(xs, H, g["steps"])
        return preds_cache[key]

    for q0, q1 in zip(qs[:-1], qs[1:], strict=True):
        if (q0 - t_first).days < MIN_TRAIN_DAYS:
            continue
        tr = (ts + step_s <= (q0 - timedelta(days=1)).timestamp()) & np.isfinite(Y) & (Y > 0) & base_ok
        te = (ts >= q0.timestamp()) & (ts < q1.timestamp()) & np.isfinite(Y) & (Y > 0) & base_ok & origin_ok
        tr_o = tr & origin_ok
        if tr_o.sum() < 200 or te.sum() == 0:
            continue
        out = {}
        # HAR level
        b = ols(Xl[tr_o], Y[tr_o])
        p = np.maximum(Xl @ b, 1e-3 * np.mean(Y[tr_o]))
        out["har"] = (p * np.mean(Y[tr_o] / p[tr_o]), np.log(p))
        # log-HAR
        b = ols(Xg[tr_o], lY[tr_o])
        p = Xg @ b
        out["loghar"] = (np.exp(p) * np.mean(Y[tr_o] / np.exp(p[tr_o])), p)
        # RFSV-GJR: H from the training fold's structure function
        xtr = np.where(ts < (q0 - timedelta(days=1)).timestamp(), 0.5 * xs, np.nan)
        H = float(np.clip(hurst(xtr, Ds), 0.01, 0.6))
        p = pred_H(H)
        ok = tr_o & np.isfinite(p)
        out["rfsv"] = (np.exp(p) * np.mean(Y[ok] / np.exp(p[ok])), p)
        # RFSV-reg
        best = None
        for Hg in HGRID:
            pg = pred_H(Hg)
            okg = tr_o & np.isfinite(pg)
            A = np.column_stack([np.ones(okg.sum()), pg[okg]])
            bb = ols(A, lY[okg])
            mse = float(np.mean((lY[okg] - A @ bb) ** 2))
            if best is None or mse < best[0]:
                best = (mse, Hg, bb)
        _, Hr, bb = best
        p = bb[0] + bb[1] * pred_H(Hr)
        ok = tr_o & np.isfinite(p)
        out["rfsv_reg"] = (np.exp(p) * np.mean(Y[ok] / np.exp(p[ok])), p)
        # GARCH
        gp = garch_path(g["e"], ts < (q0 - timedelta(days=1)).timestamp(), g["steps"], 17520 if grid == "hourly" else 5000)
        if gp is not None:
            ok = tr_o & np.isfinite(gp) & (gp > 0)
            out["garch"] = (gp * np.mean(Y[ok] / gp[ok]), np.log(np.maximum(gp, 1e-300)))
        idx = np.flatnonzero(te)
        cols = {"symbol": [s] * len(idx), "ts": ts[idx].astype(np.int64), "Y": Y[idx], "quarter": [q0.strftime("%YQ") + str((q0.month - 1) // 3 + 1)] * len(idx),
                "H_train": np.full(len(idx), H), "H_reg": np.full(len(idx), Hr)}
        for m in MODELS:
            if m in out:
                cols[f"F_{m}"] = out[m][0][idx]
                cols[f"L_{m}"] = out[m][1][idx]
            else:
                cols[f"F_{m}"] = np.full(len(idx), np.nan)
                cols[f"L_{m}"] = np.full(len(idx), np.nan)
        rows.append(pl.DataFrame(cols))
    return pl.concat(rows) if rows else pl.DataFrame()


def forecast(symbols: list[str] | None = None) -> None:
    for grid, h in (("hourly", 1), ("hourly", 4), ("daily", 1)):
        tag = f"{grid}_{h}"
        p = root() / f"fc_{tag}.parquet"
        if p.exists():
            continue
        parts = []
        for s in symbols or UNIVERSE:
            t0 = time.time()
            d = forecast_one(s, grid, h)
            if d.height:
                parts.append(d)
            sys.stdout.write(f"{tag} {s}: {d.height} origins, {time.time() - t0:.1f}s\n")
            sys.stdout.flush()
        pl.concat(parts, how="diagonal_relaxed").write_parquet(p)


def qlike(y: np.ndarray, f: np.ndarray) -> np.ndarray:
    r = y / f
    return r - np.log(r) - 1


def evaluate_forecasts() -> dict:
    from scipy.stats import norm

    cut = cutoff().timestamp()
    res: dict = {}
    for grid, h, lab in (("hourly", 1, "1h"), ("hourly", 4, "4h"), ("daily", 1, "24h")):
        d = pl.read_parquet(root() / f"fc_{grid}_{h}.parquet")
        d = d.filter(pl.all_horizontal([pl.col(f"F_{m}").is_finite() for m in MODELS if m != "garch"]) & (pl.col("Y") > 0))
        lY = np.log(d["Y"].to_numpy())
        Y = d["Y"].to_numpy()
        ql = {m: qlike(Y, d[f"F_{m}"].to_numpy()) for m in MODELS}
        ms = {m: (lY - d[f"L_{m}"].to_numpy()) ** 2 for m in MODELS}
        base = d.select("symbol", "ts").with_columns(*[pl.Series(f"q_{m}", ql[m]) for m in MODELS], *[pl.Series(f"m_{m}", ms[m]) for m in MODELS])
        seg = np.where(d["ts"].to_numpy() < cut, "llm_in_sample", "clean_holdout")
        base = base.with_columns(pl.Series("seg", seg))
        r = {}
        for sg in SEGS:
            b = base.filter(pl.col("seg") == sg)
            rr = {"n_cells": b.height, "qlike": {m: float(b[f"q_{m}"].drop_nans().mean()) for m in MODELS},
                  "mse_log": {m: float(b[f"m_{m}"].drop_nans().mean()) for m in MODELS}, "garch_cells": int(b["q_garch"].is_finite().sum()), "dm": {}}
            for a, c in (("rfsv", "loghar"), ("rfsv", "har"), ("rfsv", "garch"), ("rfsv_reg", "loghar"), ("loghar", "har"), ("loghar", "garch")):
                bb = b.filter(pl.col(f"q_{a}").is_finite() & pl.col(f"q_{c}").is_finite())
                per_ts = bb.group_by("ts").agg(*[pl.col(f"{x}_{m}").mean() for m in (a, c) for x in ("q", "m")]).sort("ts")
                rr.setdefault("n_origins", per_ts.height)
                for loss in ("q", "m"):
                    diff = (per_ts[f"{loss}_{a}"] - per_ts[f"{loss}_{c}"]).to_numpy()
                    mu, t, n = nw_t(diff, min_lag=h)
                    rel = mu / float(per_ts[f"{loss}_{c}"].mean())
                    rr["dm"][f"{loss}:{a}-{c}"] = {"mean": mu, "t": t, "n": n, "rel": rel, "p_one_sided": float(norm.cdf(t)) if np.isfinite(t) else None}
            # per-token DM count, rfsv vs loghar QLIKE
            wins = losses = 0
            for s in b["symbol"].unique().to_list():
                bs = b.filter(pl.col("symbol") == s).sort("ts")
                mu, t, n = nw_t((bs["q_rfsv"] - bs["q_loghar"]).to_numpy(), min_lag=h)
                wins += int(np.isfinite(t) and t < -2)
                losses += int(np.isfinite(t) and t > 2)
            rr["per_token_rfsv_vs_loghar"] = {"t<-2": wins, "t>2": losses, "n_tokens": b["symbol"].n_unique()}
            rr["H_train_median"] = float(d.filter(pl.Series(seg) == sg)["H_train"].median()) if b.height else None
            rr["H_reg_median"] = float(d.filter(pl.Series(seg) == sg)["H_reg"].median()) if b.height else None
            r[sg] = rr
        res[lab] = r
    # Holm over the 3 horizons per segment (QLIKE rfsv - loghar), verdicts
    ver = {}
    for sg in SEGS:
        ps = {lab: res[lab][sg]["dm"]["q:rfsv-loghar"]["p_one_sided"] for lab in res}
        order = sorted(ps, key=lambda k: ps[k])
        adj, run = {}, 0.0
        for i, k in enumerate(order):
            run = max(run, min(1.0, (len(order) - i) * ps[k]))
            adj[k] = run
        n_win = sum(1 for k in res if res[k][sg]["dm"]["q:rfsv-loghar"]["mean"] < 0 and adj[k] < 0.05)
        any_bad = any(res[k][sg]["dm"]["q:rfsv-loghar"]["t"] > 2 for k in res)
        all_ge = all(res[k][sg]["dm"]["q:rfsv-loghar"]["mean"] >= 0 for k in res)
        v = "supports" if (n_win >= 2 and not any_bad) else ("contradicts" if all_ge else "inconclusive")
        ver[sg] = {"verdict": v, "holm": adj, "n_win": n_win}
    res["verdict"] = ver
    (root() / "forecast_eval.json").write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")
    return res


# ------------------------------------------------------------------ Volterra direction (H-VOLT1)

BETAS = (0.1, 0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5)
HALFLIVES = (1, 2, 4, 8, 16, 32, 64, 128)
K_VOLT = 168
MAKER_BP, TAKER_BP = 2.0, 7.5   # taker 5.5 + 2 bp slippage (config/fees.yaml)


def kernel_signal(z: np.ndarray, w: np.ndarray) -> np.ndarray:
    """s[t] = sum_{k=1..K} w_k z[t-k+1] (information up to the close of t)."""
    from scipy.signal import fftconvolve

    return fftconvolve(np.where(np.isfinite(z), z, 0.0), w)[: len(z)]


def volterra_frames(freq: str) -> dict[str, dict]:
    from ai_trader.research.frame import market_frame

    p = root() / f"volt_{freq}.npz"
    if p.exists():
        z = np.load(p, allow_pickle=True)
        return z["d"].item()
    out = {}
    for s in UNIVERSE:
        f = market_frame(s, freq, AS_OF, funding=False, oi=False).sort("event_ts")
        if f.height < 2000:
            continue
        sig = f["sigma_risk_next"].to_numpy()          # known at the close of t, for bar t+1
        ret = f["ret"].to_numpy()
        sig_prev = np.r_[np.nan, sig[:-1]]
        out[s] = {"ts": f["event_ts"].dt.epoch("s").to_numpy(), "z": ret / sig_prev, "sig": sig, "ret": ret,
                  "close": f["close"].to_numpy(), "high": f["high"].to_numpy(), "low": f["low"].to_numpy()}
    np.savez_compressed(p, d=np.array(out, dtype=object))
    return out


def pnl_arm(d: dict, s: np.ndarray) -> np.ndarray:
    """Per-bar net P&L (risk units) of target position sign(s_t) for bar t+1, limit at close_t filled iff bar t+1 trades
    through it, else the old position is carried through bar t+1 and the change is crossed at close_{t+1} (taker)."""
    n = len(s)
    tgt = np.sign(np.nan_to_num(s))
    pos = 0.0
    out = np.full(n, np.nan)
    close, high, low, sig, ret = d["close"], d["high"], d["low"], d["sig"], d["ret"]
    for t in range(n - 1):
        if not (np.isfinite(sig[t]) and sig[t] > 0 and np.isfinite(ret[t + 1])):
            continue
        r1 = ret[t + 1] / sig[t]
        want = tgt[t]
        if want == pos:
            out[t + 1] = pos * r1
            continue
        dq = want - pos
        through = (low[t + 1] < close[t]) if dq > 0 else (high[t + 1] > close[t])
        if through:
            out[t + 1] = want * r1 - abs(dq) * MAKER_BP / 1e4 / sig[t]
        else:
            out[t + 1] = pos * r1 - abs(dq) * TAKER_BP / 1e4 / sig[t]
        pos = want
    return out


def volterra() -> dict:
    cut = cutoff().timestamp()
    res: dict = {}
    for freq, bars_day in (("1h", 24), ("4h", 6)):
        D = volterra_frames(freq)
        for d in D.values():   # winsorise z at +/- 10 (a near-zero sigma after listing gives |z| ~ 1e12)
            with np.errstate(invalid="ignore"):
                d["z"] = np.where(np.isfinite(d["z"]), np.clip(d["z"], -10, 10), np.nan)
        wv = {b: (lambda w: w / w.sum())(np.arange(1, K_VOLT + 1) ** (-b)) for b in BETAS}
        we = {hl: (lambda w: w / w.sum())(0.5 ** (np.arange(K_VOLT) / hl)) for hl in HALFLIVES}
        sigs = {s: {"v": {b: kernel_signal(d["z"], wv[b]) for b in BETAS}, "e": {hl: kernel_signal(d["z"], we[hl]) for hl in HALFLIVES}} for s, d in D.items()}
        qs = quarters(datetime(2021, 1, 1, tzinfo=UTC), AS_OF)
        first = min(d["ts"][0] for d in D.values())
        recs = []
        for q0, q1 in zip(qs[:-1], qs[1:], strict=True):
            if (q0.timestamp() - first) / 86400 < MIN_TRAIN_DAYS:
                continue
            lim = (q0 - timedelta(days=1)).timestamp()

            def train_ic(kind: str, key) -> float:
                num = den = 0.0
                for s, d in D.items():
                    x = sigs[s][kind][key]
                    y = np.r_[d["z"][1:], np.nan]
                    m = (d["ts"] + 3600 * 24 / bars_day <= lim) & np.isfinite(x) & np.isfinite(y) & (np.abs(y) < 50)
                    if m.sum() < 500:
                        continue
                    xs_ = x[m] / x[m].std()
                    num += float(np.sum(xs_ * y[m]))
                    den += m.sum()
                return num / den if den else -np.inf

            bv = max(BETAS, key=lambda b: train_ic("v", b))
            he = max(HALFLIVES, key=lambda hl: train_ic("e", hl))
            for s, d in D.items():
                te = (d["ts"] >= q0.timestamp()) & (d["ts"] < q1.timestamp())
                if not te.any():
                    continue
                trm = (d["ts"] <= lim) & np.isfinite(d["z"])
                if trm.sum() < 1000:   # token too new for a training-fold scale
                    continue
                xv, xe = sigs[s]["v"][bv], sigs[s]["e"][he]
                sdv, sde = np.nanstd(xv[trm]), np.nanstd(xe[trm])
                y = np.r_[d["z"][1:], np.nan]
                idx = np.flatnonzero(te)
                recs.append(pl.DataFrame({"symbol": [s] * len(idx), "ts": d["ts"][idx], "sv": xv[idx] / sdv, "se": xe[idx] / sde,
                                          "y": y[idx], "beta": np.full(len(idx), bv), "hl": np.full(len(idx), he)}))
        R = pl.concat(recs).filter(pl.col("y").is_finite() & (pl.col("y").abs() < 50))
        # trading arms on the full series using each quarter's chosen kernel (signals are causal; the choice is per quarter)
        pn = []
        for s, d in D.items():
            r = R.filter(pl.col("symbol") == s).sort("ts")
            if r.height == 0:
                continue
            pos_ts = {int(t): i for i, t in enumerate(d["ts"])}
            ii = np.array([pos_ts[int(t)] for t in r["ts"].to_list()])
            sv = np.full(len(d["ts"]), np.nan)
            se = np.full(len(d["ts"]), np.nan)
            sv[ii] = r["sv"].to_numpy()
            se[ii] = r["se"].to_numpy()
            pv, pe = pnl_arm(d, sv), pnl_arm(d, se)
            # pnl at bar t+1 belongs to the decision at t: shift back to the decision timestamp
            pn.append(pl.DataFrame({"symbol": [s] * len(ii), "ts": d["ts"][ii], "pv": np.r_[pv[1:], np.nan][ii], "pe": np.r_[pe[1:], np.nan][ii]}))
        PN = pl.concat(pn)
        out = {}
        for sg in SEGS:
            m = (pl.col("ts") < cut) if sg == "llm_in_sample" else (pl.col("ts") >= cut)
            r = R.filter(m)
            per = r.group_by("ts").agg((pl.col("sv") * pl.col("y")).mean().alias("icv"), (pl.col("se") * pl.col("y")).mean().alias("ice")).sort("ts")
            icv, ice = per["icv"].to_numpy(), per["ice"].to_numpy()
            dm, dt, dn = nw_t(icv - ice)
            p = PN.filter(m).drop_nulls()
            ew = p.group_by("ts").agg(pl.col("pv").mean(), pl.col("pe").mean()).sort("ts")
            ann = math.sqrt(365 * bars_day)
            shv = float(ew["pv"].mean() / ew["pv"].std() * ann) if ew.height > 10 else float("nan")
            she = float(ew["pe"].mean() / ew["pe"].std() * ann) if ew.height > 10 else float("nan")
            dd_m, dd_t, _ = nw_t((ew["pv"] - ew["pe"]).to_numpy())
            rf = r.filter(pl.col("sv").is_finite() & pl.col("se").is_finite())
            corr_v = float(np.corrcoef(rf["sv"].to_numpy(), rf["y"].to_numpy())[0, 1])
            corr_e = float(np.corrcoef(rf["se"].to_numpy(), rf["y"].to_numpy())[0, 1])
            out[sg] = {"ic_volterra": float(np.nanmean(icv)), "ic_ewma": float(np.nanmean(ice)), "ic_diff": dm, "ic_diff_t": dt, "n_ts": dn,
                       "t_ic_volterra": nw_t(icv)[1], "t_ic_ewma": nw_t(ice)[1], "corr_volterra": corr_v, "corr_ewma": corr_e,
                       "sharpe_net_volterra": shv, "sharpe_net_ewma": she, "pnl_diff_mean": dd_m, "pnl_diff_t": dd_t,
                       "beta_chosen": sorted(set(r["beta"].to_list())), "hl_chosen": sorted(set(r["hl"].to_list()))}
            sup = out[sg]["ic_diff"] > 0 and out[sg]["ic_diff_t"] > 2 and shv > she and shv > 0
            out[sg]["supports_freq"] = bool(sup)
        res[freq] = out
    ver = {}
    for sg in SEGS:
        sup = any(res[f][sg]["supports_freq"] for f in res)
        contra = all(res[f][sg]["ic_diff"] <= 0 for f in res) or all(res[f][sg]["sharpe_net_volterra"] <= 0 and res[f][sg]["sharpe_net_ewma"] <= 0 for f in res)
        ver[sg] = "supports" if sup else ("contradicts" if contra else "inconclusive")
    res["verdict"] = ver
    (root() / "volterra.json").write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")
    return res


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="rough_vol_study")
    ap.add_argument("cmd", choices=["build", "rough", "forecast", "evaluate", "volterra"])
    a = ap.parse_args(argv)
    os.environ.setdefault("OMP_NUM_THREADS", "2")
    if a.cmd == "build":
        build()
    elif a.cmd == "rough":
        rough()
    elif a.cmd == "forecast":
        forecast()
    elif a.cmd == "evaluate":
        print(json.dumps(evaluate_forecasts(), indent=1, default=float)[:6000])
    elif a.cmd == "volterra":
        print(json.dumps(volterra(), indent=1, default=float))


if __name__ == "__main__":
    main()
