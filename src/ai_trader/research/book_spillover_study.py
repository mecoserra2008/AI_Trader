"""H-MS5 (spread -> own future volatility) and H-MS6 (ETH / SOL spillovers) on the 2026-11-10 microstructure pilot
(research/hypotheses.yaml, block "Microstructure pilot"; agent C2).

All data is one UTC day after the knowledge cutoff, so every result is segment clean_holdout. FIT = 00:00-12:00 UTC,
TEST = 12:00-24:00 UTC (contains the 13:30 UTC CPI release; robustness rows drop 13:00-16:00).

    PYTHONIOENCODING=utf-8 uv run python -m ai_trader.research.book_spillover_study --cache <dir> [--record]

Inputs: the 100 ms book series of sim/book.py (data/raw/pilot_2026-11-10/book_100ms/) and lake table trades_raw
(read point in time with Lake.read_asof, as of 2026-11-11 00:01 UTC, i.e. after the day closed).

H-MS5  per L2 symbol, 1 m grid; decision time T = end of minute m. y = log RV over (T, T+h], h = 5, 15 min, from 1 s
       mid log returns; x = log of the time-averaged relative spread over minute m; controls log RV over the last 5 and
       60 min, log depth within 10 bp (bid + ask, time-averaged over minute m), log hour-of-week factor
       (features.seasonality.profile_asof as of 2026-11-10 00:00 from 1h last bars; a pooled crypto-major profile
       where a symbol has < 8 weeks of bars). Pooled OLS with symbol fixed effects; SE clustered by 5-minute calendar
       block (CR1). Fit on FIT; on TEST the coefficient is re-estimated (t) and the FIT model is scored out of sample
       (R^2 gain = R^2 of the full model - R^2 of the model without spread). FIT rows have T >= 01:00 and T + h <= 12:00;
       TEST rows T >= 12:00 and T + h <= 24:00 (no outcome crosses the split).
H-MS6  (a) Hayashi-Yoshida cross-correlation rho(l) = corr(leader at t, target at t + l) on trade-time log prices
       (per-ms VWAP: the within-ms order is not stored in trades_raw), lags 0, +-0.1 ... 60 s; peak = argmax rho;
       5-minute block bootstrap of the HY contributions (B = 1000) for the peak-lag distribution and the lead-lag ratio
       LLR = sum_{l>0} rho^2 / sum_{l<0} rho^2. (b) Granger: bivariate VAR on 1 s last-trade log returns, lag by BIC
       (<= 10), heteroskedasticity-robust (HC0) Wald F of the leader's lags in the target equation (classical F also
       reported), FIT and TEST, both directions. (c) Diebold-Yilmaz generalized-FEVD connectedness (Pesaran-Shin,
       H = 10) of a VAR (BIC lag <= 4) on log 1 m realised volatility (12 x 5 s last-trade returns per minute) on TEST;
       net directional connectedness with a 30-minute moving-block bootstrap CI (B = 500).
Holm over H-MS5 and H-MS6 (the MS block's other hypotheses are tested by another agent).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
from scipy import stats

from ai_trader.features.seasonality import hour_of_week, profile_asof
from ai_trader.sim.flow_data import asset_class

DAY0 = datetime(2026, 11, 10, tzinfo=UTC)
D0 = int(DAY0.timestamp() * 1000)
HALF = 12 * 3_600_000
DAY = 24 * 3_600_000
AS_OF_TRADES = datetime(2026, 11, 11, 0, 1, tzinfo=UTC)
CPI_EX = (D0 + 13 * 3_600_000, D0 + 16 * 3_600_000)
WINDOWS = {"fit": (D0, D0 + HALF), "test": (D0 + HALF, D0 + DAY)}
PILOT = Path("data/raw/pilot_2026-11-10")
BOOK_DIR = PILOT / "book_100ms"
L2 = ("BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "DOGEUSDT", "HYPEUSDT", "CARVUSDT", "KASUSDT", "ATOMUSDT", "NVDAUSDT",
      "QQQUSDT", "XAUUSDT")
LEADERS = ("ETHUSDT", "SOLUSDT", "BTCUSDT")
LAGS_S = (0.1, 0.2, 0.3, 0.5, 0.75, 1, 1.5, 2, 3, 5, 7.5, 10, 15, 20, 30, 45, 60)
LAGS_MS = np.array(sorted([-int(round(x * 1000)) for x in LAGS_S] + [0] + [int(round(x * 1000)) for x in LAGS_S]), dtype=np.int64)
POS_LO, POS_HI = 100, 30_000
FINE_MS = np.array([-500, -200, -100, -50, -20, -10, 0, 10, 20, 50, 100, 200, 500], dtype=np.int64)  # diagnostic only
BLOCK_MS = 300_000
B_HY = 1000
B_DY = 500
SEED = 20261111
HORIZONS = (5, 15)
MIN_DY_COVER = 0.98
MAX_DY = 15
METHOD = {
    "split": "FIT 2026-11-10 00:00-12:00 UTC, TEST 12:00-24:00 UTC; robustness: TEST without 13:00-16:00 (CPI 13:30)",
    "book": "sim/book.py: ob200 snapshot + deltas rebuilt per message; state at the end of each 100 ms bin of the push ts; depth = notional within x bp of mid",
    "ms5_y": "log RV over (T, T+h], h = 5, 15 min, sum of squared 1 s mid log returns; zero RV floored at the symbol's 1st percentile of positive values",
    "ms5_x": "log time-averaged relative spread (bp) over the minute ending at T; robustness: spread level in bp",
    "ms5_controls": "log RV last 5 min, log RV last 60 min, log(depth_bid_10 + depth_ask_10) averaged over the last minute, log hour-of-week factor (profile_asof 2026-11-10 00:00, u = r^2 / trailing-168h mean r^2 on 1h last bars; pooled crypto-major profile for symbols with < 8 weeks of bars)",
    "ms5_model": "pooled OLS, symbol fixed effects, CR1 SE clustered by 5-minute calendar block (robustness: 30-minute blocks); OOS R^2 gain of the FIT model on TEST; Diebold-Mariano t on 5-minute block sums of the squared-error differential",
    "ms6_prices": "trade-time log price = per-ms VWAP (trades_raw stores no within-ms order); 1 s and 5 s grids = last trade price at or before the grid time",
    "ms6_hy": "Hayashi-Yoshida rho(l) = corr(leader at t, target at t+l), lags 0, +-0.1..60 s (37); peak = argmax rho; positive peak = peak in [0.1, 30] s; 5-minute block bootstrap B = 1000 of HY contributions (peak-lag distribution, LLR = sum_{l>0} rho^2 / sum_{l<0} rho^2, p = P*(LLR <= 1))",
    "ms6_granger": "bivariate VAR on 1 s last-trade log returns, BIC lag <= 10; HC0-robust Wald F of the other series' lags (classical F reported too), both directions",
    "ms6_dy": "VAR (BIC lag <= 4) on log 1 m RV (12 x 5 s returns), generalized FEVD (Pesaran-Shin) H = 10, row-normalised; net = to - from (percent); 30-minute moving-block bootstrap B = 500 with the lag fixed",
    "ms6_universe": "crypto = the 51-symbol trades set minus TradFi (sim.flow_data.asset_class); DY set = BTC, ETH, SOL + crypto with nonzero 1 m RV in >= 98% of the minutes of both halves, top by trade count, at most 15",
    "verdict_ms5": "supports = TEST coefficient > 0 with t > 2 and OOS R^2 gain > 0, both horizons; contradicts = coefficient <= 0 or gain <= 0 at a horizon with the one-sided p of the opposite > 0.95 (clearly wrong sign); else inconclusive",
    "verdict_ms6": "supports = share of crypto targets with a positive peak > 0.70 for ETH and SOL and both DY net > 0 on TEST; contradicts = a share <= 0.50 or a net CI entirely below 0; else inconclusive",
    "holm": "Holm over H-MS5, H-MS6; p(H-MS5) = max over horizons of the one-sided coefficient p and the DM p; p(H-MS6) = max over ETH, SOL of the binomial p (positive peaks vs 0.5) and the bootstrap P*(net <= 0)",
}


# ------------------------------------------------------------------ data


def load_trades(symbols: list[str] | None = None, cache: Path | None = None) -> pl.DataFrame:
    """Every trade of 2026-11-10 (symbol, t ms, side, price, size), point in time as of 2026-11-11 00:01 UTC."""
    if cache is not None and (cache / "trades.parquet").exists():
        lf = pl.scan_parquet(cache / "trades.parquet")
        return (lf if symbols is None else lf.filter(pl.col("symbol").is_in(list(symbols)))).collect()
    from ai_trader.storage.lake import Lake

    lake = Lake()
    df = lake.read_asof("trades_raw", AS_OF_TRADES, symbols=symbols, start=DAY0, end=DAY0 + timedelta(days=1),
                        columns=["symbol", "event_ts", "side", "price", "size"])
    df = (df.with_columns(pl.col("event_ts").dt.epoch("ms").cast(pl.Int64).alias("t")).drop("event_ts")
          .sort(["symbol", "t"]))
    if cache is not None and symbols is None:
        cache.mkdir(parents=True, exist_ok=True)
        df.write_parquet(cache / "trades.parquet")
    return df


def instruments() -> dict[str, dict]:
    from ai_trader.storage.lake import Lake

    df = Lake().read_asof("instruments", DAY0, columns=["symbol", "symbol_type", "tick_size", "event_ts"])
    df = df.sort("event_ts").unique("symbol", keep="last")
    return {r["symbol"]: r for r in df.iter_rows(named=True)}


def symbol_class(sym: str, inst: dict[str, dict]) -> str:
    return asset_class(sym, (inst.get(sym) or {}).get("symbol_type"))


def ms_prices(tr: pl.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Unique-ms trade times and log VWAP per ms."""
    g = (tr.group_by("t").agg(((pl.col("price") * pl.col("size")).sum() / pl.col("size").sum()).alias("vwap")).sort("t"))
    return g["t"].to_numpy().astype(np.int64), np.log(g["vwap"].to_numpy())


def grid_last(t: np.ndarray, lp: np.ndarray, start: int, end: int, step: int) -> np.ndarray:
    """Last log price at or before each grid time start, start+step, ..., end (NaN before the first trade)."""
    g = np.arange(start, end + 1, step, dtype=np.int64)
    i = np.searchsorted(t, g, side="right") - 1
    out = np.where(i >= 0, lp[np.clip(i, 0, None)], np.nan)
    return out


# ------------------------------------------------------------------ Hayashi-Yoshida


def _hy_base(tb: np.ndarray, pb: np.ndarray, to: np.ndarray, po: np.ndarray, shifts: np.ndarray, lo: int,
             block_ms: int, nblocks: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """HY contributions per block of corr(base at t, other at t + d) for each shift d: other's times move to `to - d`.
    Returns (C[n_shift, nblocks], Vbase[nblocks], Vother[nblocks]); blocks by the end time of each interval."""
    rb = np.diff(pb)
    a, b = tb[:-1], tb[1:]
    blk_b = np.clip((b - lo) // block_ms, 0, nblocks - 1)
    m = len(to) - 1
    C = np.zeros((len(shifts), nblocks))
    for k, d in enumerate(shifts):
        s = to - d
        jlo = np.maximum(np.searchsorted(s, a, side="right"), 1)
        jhi = np.minimum(np.searchsorted(s, b, side="left"), m)
        ok = jhi >= jlo
        osum = np.where(ok, po[np.clip(jhi, 0, m)] - po[np.clip(jlo - 1, 0, m)], 0.0)
        C[k] = np.bincount(blk_b, weights=rb * osum, minlength=nblocks)
    vb = np.bincount(blk_b, weights=rb * rb, minlength=nblocks)
    ro = np.diff(po)
    blk_o = np.clip((to[1:] - lo) // block_ms, 0, nblocks - 1)
    vo = np.bincount(blk_o, weights=ro * ro, minlength=nblocks)
    return C, vb, vo


def hy_blocks(t1: np.ndarray, p1: np.ndarray, t2: np.ndarray, p2: np.ndarray, lags_ms: np.ndarray, window: tuple[int, int],
              block_ms: int = BLOCK_MS) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Block contributions of rho_12(l) = corr(series 1 at t, series 2 at t + l) within `window` = [lo, hi) ms.
    The series with fewer observations is the base (speed); corr(1(t), 2(t+l)) = corr(2(u), 1(u-l))."""
    lo, hi = window
    k1 = (t1 >= lo) & (t1 < hi)
    k2 = (t2 >= lo) & (t2 < hi)
    t1, p1, t2, p2 = t1[k1], p1[k1], t2[k2], p2[k2]
    nb = int(math.ceil((hi - lo) / block_ms))
    if len(t1) < 3 or len(t2) < 3:
        return np.zeros((len(lags_ms), nb)), np.zeros(nb), np.zeros(nb)
    if len(t1) <= len(t2):
        return _hy_base(t1, p1, t2, p2, lags_ms, lo, block_ms, nb)
    C, v2, v1 = _hy_base(t2, p2, t1, p1, -lags_ms, lo, block_ms, nb)
    return C, v1, v2


def hy_rho(C: np.ndarray, v1: np.ndarray, v2: np.ndarray, keep: np.ndarray | None = None) -> np.ndarray:
    w = np.ones(C.shape[1]) if keep is None else keep.astype(float)
    den = math.sqrt(float(v1 @ w) * float(v2 @ w))
    return (C @ w) / den if den > 0 else np.full(C.shape[0], np.nan)


def hy_ccf(t1, p1, t2, p2, lags_ms, window=None) -> np.ndarray:
    """rho_12(l) for each lag (whole sample when window is None)."""
    if window is None:
        window = (int(min(t1[0], t2[0])), int(max(t1[-1], t2[-1])) + 1)
    C, v1, v2 = hy_blocks(np.asarray(t1, np.int64), np.asarray(p1, float), np.asarray(t2, np.int64), np.asarray(p2, float),
                          np.asarray(lags_ms, np.int64), window, block_ms=max(1, window[1] - window[0]))
    return hy_rho(C, v1, v2)


def lead_stats(rho: np.ndarray, lags_ms: np.ndarray = LAGS_MS) -> dict:
    ip = int(np.nanargmax(rho)) if np.isfinite(rho).any() else 0
    pos, neg = lags_ms > 0, lags_ms < 0
    sneg = float(np.nansum(rho[neg] ** 2))
    return {"peak_lag_s": float(lags_ms[ip]) / 1000, "peak_rho": float(rho[ip]), "rho0": float(rho[lags_ms == 0][0]),
            "llr": float(np.nansum(rho[pos] ** 2) / sneg) if sneg > 0 else float("nan"),
            "positive_peak": bool(POS_LO <= lags_ms[ip] <= POS_HI)}


def hy_bootstrap(C: np.ndarray, v1: np.ndarray, v2: np.ndarray, keep: np.ndarray, rng: np.random.Generator,
                 B: int = B_HY, lags_ms: np.ndarray = LAGS_MS) -> dict:
    idx = np.flatnonzero(keep)
    draws = rng.integers(0, len(idx), size=(B, len(idx)))
    W = np.zeros((B, C.shape[1]))
    np.add.at(W, (np.repeat(np.arange(B), len(idx)), idx[draws].ravel()), 1.0)
    num = W @ C.T
    den = np.sqrt((W @ v1) * (W @ v2))[:, None]
    with np.errstate(invalid="ignore", divide="ignore"):
        R = num / den
    R = np.where(np.isfinite(R), R, -np.inf)
    pk = lags_ms[np.argmax(R, axis=1)]
    pos, neg = lags_ms > 0, lags_ms < 0
    Rf = np.where(np.isfinite(R), R, 0.0)
    llr = (Rf[:, pos] ** 2).sum(1) / np.maximum((Rf[:, neg] ** 2).sum(1), 1e-300)
    return {"share_positive_peak": float(np.mean((pk >= POS_LO) & (pk <= POS_HI))), "share_peak_at_0": float(np.mean(pk == 0)),
            "share_negative_peak": float(np.mean(pk < 0)), "p_llr_le_1": float(np.mean(llr <= 1.0)),
            "llr_ci": [float(np.quantile(llr, 0.025)), float(np.quantile(llr, 0.975))],
            "peak_lag_ci_s": [float(np.quantile(pk, 0.025)) / 1000, float(np.quantile(pk, 0.975)) / 1000]}


# ------------------------------------------------------------------ Granger on 1 s returns


def _lags(x: np.ndarray, p: int) -> np.ndarray:
    n = len(x)
    return np.column_stack([x[p - k:n - k] for k in range(1, p + 1)])


def _ols(y: np.ndarray, X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    xtx_inv = np.linalg.pinv(X.T @ X)
    b = xtx_inv @ (X.T @ y)
    return b, y - X @ b, xtx_inv


def granger(y: np.ndarray, x: np.ndarray, pmax: int = 10) -> dict:
    """Does x Granger-cause y (and y -> x)? Bivariate VAR on aligned return series, lag by BIC on a common sample."""
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = np.where(ok, x, 0.0), np.where(ok, y, 0.0)
    sx, sy = x.std(), y.std()
    if sx == 0 or sy == 0 or len(x) < 10 * pmax:
        return {"p": None}
    x, y = x / sx, y / sy
    Ly, Lx = _lags(y, pmax), _lags(x, pmax)
    Y = np.column_stack([y[pmax:], x[pmax:]])
    T = len(Y)
    best, bic_best = 1, np.inf
    for p in range(1, pmax + 1):
        Z = np.column_stack([np.ones(T), Ly[:, :p], Lx[:, :p]])
        _, U, _ = _ols(Y, Z)
        S = U.T @ U / T
        bic = math.log(max(np.linalg.det(S), 1e-300)) + 4 * p * math.log(T) / T
        if bic < bic_best:
            best, bic_best = p, bic
    p = best
    out = {"p": p, "T": T}
    for name, dep, own, other in (("x_to_y", Y[:, 0], Ly, Lx), ("y_to_x", Y[:, 1], Lx, Ly)):
        Zr = np.column_stack([np.ones(T), own[:, :p]])
        Zu = np.column_stack([Zr, other[:, :p]])
        _, ur, _ = _ols(dep, Zr)
        bu, uu, xinv = _ols(dep, Zu)
        ssr_r, ssr_u = float(ur @ ur), float(uu @ uu)
        df2 = T - Zu.shape[1]
        F = ((ssr_r - ssr_u) / p) / (ssr_u / df2)
        meat = (Zu * (uu ** 2)[:, None]).T @ Zu
        V = xinv @ meat @ xinv
        sel = slice(1 + p, 1 + 2 * p)
        bR = bu[sel]
        Wd = float(bR @ np.linalg.pinv(V[sel, sel]) @ bR)
        out[name] = {"F": float(F), "p_F": float(stats.f.sf(F, p, df2)), "F_robust": Wd / p, "p_robust": float(stats.chi2.sf(Wd, p)),
                     "sum_coef": float(bR.sum())}
    return out


# ------------------------------------------------------------------ VAR and Diebold-Yilmaz


def var_design(Y: np.ndarray, p: int, seg: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """VAR(p) regression design. seg: segment id per row (-1 = excluded); a row is used only if its lag window lies in
    one segment (seg[t - p] == seg[t] >= 0, segments contiguous), so lags never cross a gap or a bootstrap block edge."""
    T = len(Y)
    Z = np.column_stack([np.ones(T - p)] + [Y[p - k:T - k] for k in range(1, p + 1)])
    Yd = Y[p:]
    if seg is not None:
        ok = (seg[p:] >= 0) & (seg[:T - p] == seg[p:])
        Z, Yd = Z[ok], Yd[ok]
    return Yd, Z


def var_fit(Y: np.ndarray, p: int, seg: np.ndarray | None = None) -> tuple[list[np.ndarray], np.ndarray]:
    Yd, Z = var_design(Y, p, seg)
    B = np.linalg.lstsq(Z, Yd, rcond=None)[0]
    U = Yd - Z @ B
    S = U.T @ U / (len(Yd) - Z.shape[1])
    N = Y.shape[1]
    A = [B[1 + k * N:1 + (k + 1) * N].T for k in range(p)]
    return A, S


def var_bic(Y: np.ndarray, pmax: int, seg: np.ndarray | None = None) -> int:
    N = Y.shape[1]
    best, bb = 1, np.inf
    for p in range(1, pmax + 1):
        Yd, Z = var_design(Y[pmax - p:], p, None if seg is None else seg[pmax - p:])
        B = np.linalg.lstsq(Z, Yd, rcond=None)[0]
        U = Yd - Z @ B
        T = len(Yd)
        _, logdet = np.linalg.slogdet(U.T @ U / T)
        bic = logdet + N * N * p * math.log(T) / T
        if bic < bb:
            best, bb = p, bic
    return best


def gfevd(A: list[np.ndarray], S: np.ndarray, H: int = 10) -> np.ndarray:
    """Generalized FEVD (Pesaran-Shin), row-normalised: theta[i, j] = share of i's H-step error variance due to j."""
    N = S.shape[0]
    p = len(A)
    Psi = [np.eye(N)]
    for h in range(1, H):
        Psi.append(sum(A[k] @ Psi[h - 1 - k] for k in range(min(p, h))))
    num = np.zeros((N, N))
    den = np.zeros(N)
    for P in Psi:
        PS = P @ S
        num += PS ** 2
        den += np.einsum("ij,jk,ik->i", P, S, P)
    th = num / np.diag(S)[None, :] / den[:, None]
    return th / th.sum(1, keepdims=True)


def connectedness(th: np.ndarray) -> dict:
    N = th.shape[0]
    off = th - np.diag(np.diag(th))
    frm = off.sum(1) * 100
    to = off.sum(0) * 100
    return {"from": frm, "to": to, "net": to - frm, "total": float(off.sum() / N * 100)}


def dy_table(Y: np.ndarray, names: list[str], *, pmax: int = 4, H: int = 10, seg: np.ndarray | None = None, boot: int = 0,
             block: int = 30, rng: np.random.Generator | None = None) -> dict:
    p = var_bic(Y, pmax, seg)
    A, S = var_fit(Y, p, seg)
    th = gfevd(A, S, H)
    c = connectedness(th)
    out = {"symbols": names, "lag": p, "H": H, "total": c["total"], "theta": th.round(5).tolist(),
           "to": dict(zip(names, c["to"].round(3).tolist())), "from": dict(zip(names, c["from"].round(3).tolist())),
           "net": dict(zip(names, c["net"].round(3).tolist()))}
    out["pairwise_pct"] = {f"{names[j]}->{names[i]}": float(th[i, j] * 100) for i in range(len(names)) for j in range(len(names)) if i != j}
    if boot and rng is not None:
        T = len(Y)
        nb = int(math.ceil(T / block))
        nets, tots = [], []
        for _ in range(boot):
            starts = rng.integers(0, T - block + 1, size=nb)
            ix = np.concatenate([np.arange(st, st + block) for st in starts])[:T]
            sb = np.repeat(np.arange(nb), block)[:T]
            if seg is not None:
                sb = np.where(seg[ix] >= 0, sb * 10 + seg[ix] + 1, -1)
            Ab, Sb = var_fit(Y[ix], p, sb)
            cb = connectedness(gfevd(Ab, Sb, H))
            nets.append(cb["net"])
            tots.append(cb["total"])
        nets = np.array(nets)
        out["net_ci"] = {n: [float(np.quantile(nets[:, i], 0.025)), float(np.quantile(nets[:, i], 0.975))] for i, n in enumerate(names)}
        out["p_net_le_0"] = {n: float(np.mean(nets[:, i] <= 0)) for i, n in enumerate(names)}
        out["total_ci"] = [float(np.quantile(tots, 0.025)), float(np.quantile(tots, 0.975))]
    return out


def rv_minutes(t: np.ndarray, lp: np.ndarray, step_ms: int = 5000) -> np.ndarray:
    """1 m realised variance over the day from `step_ms` last-trade log returns (1440 values; NaN before first trade)."""
    g = grid_last(t, lp, D0, D0 + DAY, step_ms)
    r = np.diff(g)
    per = 60_000 // step_ms
    return np.nansum((r ** 2).reshape(-1, per), axis=1) * np.where(np.isnan(r).reshape(-1, per).all(1), np.nan, 1.0)


# ------------------------------------------------------------------ H-MS5 panel


def how_factors(symbols: list[str]) -> dict[str, np.ndarray]:
    """168-vector hour-of-week factor per symbol known at 2026-11-10 00:00 (pooled major profile if < 8 weeks)."""
    from ai_trader.storage.lake import Lake

    lake = Lake()
    prof, nbars = {}, {}
    for s in sorted(set(symbols) | {"BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "DOGEUSDT"}):
        b = lake.bars(s, "1h", DAY0).sort("event_ts")
        if b.height < 200:
            nbars[s] = b.height
            continue
        r = np.diff(np.log(b["close"].to_numpy()), prepend=np.nan)
        r2 = pl.Series(r ** 2)
        f0 = r2.shift(1).rolling_mean(168, min_samples=48).to_numpy()
        u = r ** 2 / f0
        prof[s] = profile_asof(b["event_ts"], u, DAY0)
        nbars[s] = b.height
    pooled = np.mean([prof[s] for s in ("BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "DOGEUSDT") if s in prof], axis=0)
    out = {}
    for s in symbols:
        out[s] = prof[s] if (s in prof and nbars[s] >= 8 * 168) else pooled
    return out, {s: {"bars_1h": nbars.get(s, 0), "profile": "own" if (s in prof and nbars.get(s, 0) >= 8 * 168) else "pooled_majors"}
                 for s in symbols}


def book_minutes(sym: str) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Per-minute book aggregates (minute m = 0..1439: states with ts in (m, m+1] min) and the 1 s mid series."""
    b = pl.read_parquet(BOOK_DIR / f"{sym}.parquet").with_columns(pl.col("ts").dt.epoch("ms").cast(pl.Int64).alias("t"))
    b = b.filter((pl.col("t") > D0) & (pl.col("t") <= D0 + DAY))
    b = b.with_columns(((pl.col("t") - D0 - 1) // 60_000).alias("m"),
                       (pl.col("depth_bid_10") + pl.col("depth_ask_10")).alias("depth10"))
    sec = b.filter(pl.col("t") % 1000 == 0).select("t", "m", "mid", "rel_spread_bp").sort("t")
    sec = sec.with_columns(pl.col("mid").log().diff().alias("r"))
    rv = sec.group_by("m").agg((pl.col("r") ** 2).sum().alias("rv1m"), pl.col("r").is_not_null().sum().alias("nsec"))
    mins = (b.group_by("m").agg(pl.col("rel_spread_bp").mean().alias("spread"), pl.col("depth10").mean().alias("depth10"),
                                pl.col("n_updates").sum().alias("updates"))
            .join(rv, on="m", how="left").sort("m"))
    grid = pl.DataFrame({"m": np.arange(1440, dtype=np.int64)})
    return grid.join(mins, on="m", how="left"), sec


def _floor_log(x: np.ndarray) -> np.ndarray:
    pos = x[np.isfinite(x) & (x > 0)]
    fl = np.quantile(pos, 0.01) if len(pos) else 1e-12
    return np.log(np.maximum(x, fl))


def ms5_rows(sym: str, factor: np.ndarray) -> pl.DataFrame:
    mins, _ = book_minutes(sym)
    rv = mins["rv1m"].to_numpy().astype(float)
    cs = np.concatenate([[0.0], np.nancumsum(rv)])
    n = 1440
    m = np.arange(n)

    def window(lo, hi):  # sum of rv over minutes lo..hi inclusive, NaN if out of range
        ok = (lo >= 0) & (hi < n)
        return np.where(ok, cs[np.clip(hi + 1, 0, n)] - cs[np.clip(lo, 0, n)], np.nan)

    T_ms = D0 + (m + 1) * 60_000
    how = hour_of_week(pl.Series((T_ms).astype("datetime64[ms]")))
    out = {"symbol": [sym] * n, "m": m, "T": T_ms, "spread": mins["spread"].to_numpy(), "depth10": mins["depth10"].to_numpy(),
           "rv5p": window(m - 4, m), "rv60p": window(m - 59, m), "how_f": factor[how]}
    for h in HORIZONS:
        out[f"y{h}"] = window(m + 1, m + h)
    df = pl.DataFrame(out)
    cols = {}
    for c in ["rv5p", "rv60p"] + [f"y{h}" for h in HORIZONS]:
        cols["l" + c] = _floor_log(df[c].to_numpy())
    df = df.with_columns([pl.Series(k, v) for k, v in cols.items()]).with_columns(
        pl.col("spread").log().alias("lspread"), pl.col("depth10").log().alias("ldepth10"), pl.col("how_f").log().alias("lhow"))
    return df


def cluster_ols(y: np.ndarray, X: np.ndarray, groups: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    b, u, xinv = _ols(y, X)
    G = np.unique(groups)
    gi = np.searchsorted(G, groups)
    S = np.zeros((len(G), X.shape[1]))
    np.add.at(S, gi, X * u[:, None])
    n, k, g = X.shape[0], X.shape[1], len(G)
    V = xinv @ (S.T @ S) @ xinv * (g / (g - 1)) * ((n - 1) / (n - k))
    return b, np.sqrt(np.maximum(np.diag(V), 0))


def _design(df: pl.DataFrame, syms: list[str], xname: str, with_x: bool = True) -> np.ndarray:
    cols = ([xname] if with_x else []) + ["lrv5p", "lrv60p", "ldepth10", "lhow"]
    X = df.select(cols).to_numpy()
    fe = (df["symbol"].to_numpy()[:, None] == np.array(syms)[None, :]).astype(float)
    return np.column_stack([X, fe])


def ms5_split(panel: pl.DataFrame, h: int, half: str, exclude_cpi: bool = False) -> pl.DataFrame:
    lo, hi = WINDOWS[half]
    T = pl.col("T")
    d = panel.filter((T >= (D0 + 3_600_000 if half == "fit" else lo)) & (T + h * 60_000 <= hi))
    if exclude_cpi:
        d = d.filter(~((T + h * 60_000 > CPI_EX[0]) & (T < CPI_EX[1])))
    need = ["lspread", "lrv5p", "lrv60p", "ldepth10", "lhow", f"ly{h}", "spread"]
    return d.drop_nulls(need).filter(pl.all_horizontal([pl.col(c).is_finite() for c in need]))


def ms5_fit(panel: pl.DataFrame, h: int, xname: str = "lspread", cluster_min: int = 5, exclude_cpi: bool = False) -> dict:
    fit, test = ms5_split(panel, h, "fit"), ms5_split(panel, h, "test", exclude_cpi)
    syms = sorted(set(fit["symbol"].to_list()) & set(test["symbol"].to_list()))
    fit, test = fit.filter(pl.col("symbol").is_in(syms)), test.filter(pl.col("symbol").is_in(syms))
    yname = f"ly{h}"
    res: dict[str, Any] = {"h": h, "x": xname, "n_fit": fit.height, "n_test": test.height, "symbols": syms, "cluster_min": cluster_min}
    for half, d in (("fit", fit), ("test", test)):
        X, y = _design(d, syms, xname), d[yname].to_numpy()
        g = (d["T"].to_numpy() - 1) // (cluster_min * 60_000)
        b, se = cluster_ols(y, X, g)
        res[half] = {"coef": float(b[0]), "se": float(se[0]), "t": float(b[0] / se[0]), "n_clusters": int(len(np.unique(g))),
                     "controls": {n: [float(b[i]), float(b[i] / se[i])] for i, n in enumerate(["lrv5p", "lrv60p", "ldepth10", "lhow"], 1)}}
    Xf, yf = _design(fit, syms, xname), fit[yname].to_numpy()
    Xr = _design(fit, syms, xname, with_x=False)
    bf = _ols(yf, Xf)[0]
    br = _ols(yf, Xr)[0]
    Xt, yt = _design(test, syms, xname), test[yname].to_numpy()
    ef, er = yt - Xt @ bf, yt - _design(test, syms, xname, with_x=False) @ br
    sst = float(((yt - yt.mean()) ** 2).sum())
    res["r2_oos_full"] = 1 - float(ef @ ef) / sst
    res["r2_oos_restricted"] = 1 - float(er @ er) / sst
    res["r2_oos_gain"] = res["r2_oos_full"] - res["r2_oos_restricted"]
    d = er ** 2 - ef ** 2
    g = (test["T"].to_numpy() - 1) // (cluster_min * 60_000)
    gs = np.bincount(np.searchsorted(np.unique(g), g), weights=d)
    dm_t = gs.mean() / (gs.std(ddof=1) / math.sqrt(len(gs)))
    res["dm_t"] = float(dm_t)
    res["p_gain"] = float(stats.norm.sf(dm_t))
    res["p_coef"] = float(stats.norm.sf(res["test"]["t"]))
    # per-symbol OOS gain
    per = {}
    for s in syms:
        k = test["symbol"].to_numpy() == s
        sst_s = float(((yt[k] - yt[k].mean()) ** 2).sum())
        per[s] = float((er[k] @ er[k] - ef[k] @ ef[k]) / sst_s) if sst_s > 0 else float("nan")
    res["r2_gain_by_symbol"] = per
    return res


def ms5_per_symbol(panel: pl.DataFrame, h: int) -> dict:
    out = {}
    for s in sorted(set(panel["symbol"].to_list())):
        row = {}
        for half in ("fit", "test"):
            d = ms5_split(panel.filter(pl.col("symbol") == s), h, half)
            if d.height < 60:
                continue
            X = np.column_stack([d.select(["lspread", "lrv5p", "lrv60p", "ldepth10"]).to_numpy(), np.ones(d.height)])
            g = (d["T"].to_numpy() - 1) // 300_000
            b, se = cluster_ols(d[f"ly{h}"].to_numpy(), X, g)
            row[half] = {"coef": float(b[0]), "t": float(b[0] / se[0]), "n": d.height}
        out[s] = row
    return out


# ------------------------------------------------------------------ book descriptives for the simulator


def acf(x: np.ndarray, lags: list[int]) -> dict[int, float]:
    x = x[np.isfinite(x)]
    x = x - x.mean()
    v = float(x @ x)
    return {k: float(x[:-k] @ x[k:] / v) if v > 0 and k < len(x) else float("nan") for k in lags}


def book_descriptives(sym: str, tick: float | None) -> dict:
    b = pl.read_parquet(BOOK_DIR / f"{sym}.parquet").with_columns(pl.col("ts").dt.epoch("ms").cast(pl.Int64).alias("t"))
    b = b.filter((pl.col("t") > D0) & (pl.col("t") <= D0 + DAY))
    sp = b["rel_spread_bp"].to_numpy()
    q = [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]
    out: dict[str, Any] = {"rows_100ms": b.height, "updates_per_s": float(b["n_updates"].mean() * 10),
                           "spread_bp_q": dict(zip([str(x) for x in q], np.nanquantile(sp, q).round(4).tolist())),
                           "spread_bp_mean": float(np.nanmean(sp)), "mid_mean": float(b["mid"].mean())}
    if tick:
        ticks = np.round((b["ask1"] - b["bid1"]).to_numpy() / tick)
        out["tick"] = tick
        out["tick_bp"] = float(tick / b["mid"].mean() * 1e4)
        out["spread_ticks_share"] = {"1": float(np.mean(ticks <= 1)), "2": float(np.mean(ticks == 2)), "3-5": float(np.mean((ticks >= 3) & (ticks <= 5))),
                                     ">5": float(np.mean(ticks > 5))}
        out["spread_ticks_median"] = float(np.median(ticks))
    for bnd in (5, 10, 25):
        d = (b[f"depth_bid_{bnd}"] + b[f"depth_ask_{bnd}"]).to_numpy()
        out[f"depth{bnd}_usd_q"] = dict(zip(["0.05", "0.25", "0.5", "0.75", "0.95"], np.nanquantile(d, [0.05, 0.25, 0.5, 0.75, 0.95]).round(0).tolist()))
    if "span_bid_bp" in b.columns:
        span = np.minimum(b["span_bid_bp"].to_numpy(), b["span_ask_bp"].to_numpy())
        out["book_span_bp_median"] = float(np.nanmedian(span))
        out["band_truncated_share"] = {str(bnd): float(np.nanmean(span < bnd)) for bnd in (5, 10, 25)}
    top = (b["bid1"] * b["bid1_size"] + b["ask1"] * b["ask1_size"]).to_numpy() / 2
    out["top_level_usd_median"] = float(np.nanmedian(top))
    out["abs_imb5_mean"] = float(np.nanmean(np.abs(b["imb_5"].to_numpy())))
    out["abs_imb10_mean"] = float(np.nanmean(np.abs(b["imb_10"].to_numpy())))
    s1 = b.filter(pl.col("t") % 1000 == 0)["rel_spread_bp"].to_numpy()
    out["spread_acf_1s"] = acf(s1, [1, 5, 10, 30, 60, 300])
    mins, sec = book_minutes(sym)
    sm = mins["spread"].to_numpy()
    out["spread_acf_1m"] = acf(sm, [1, 5, 15, 60])
    rv = mins["rv1m"].to_numpy().astype(float)
    ok = np.isfinite(sm) & np.isfinite(rv) & (rv > 0) & (sm > 0)
    if ok.sum() > 100:
        X = np.column_stack([np.log(rv[ok]), np.ones(ok.sum())])
        g = np.arange(1440)[ok] // 5
        bb, se = cluster_ols(np.log(sm[ok]), X, g)
        out["elasticity_spread_on_rv"] = {"coef": float(bb[0]), "t": float(bb[0] / se[0]), "n": int(ok.sum()),
                                          "note": "log 1m TWA spread on log same-minute RV of 1 s mid returns; spread ~ RV^coef (vol^(2 coef))"}
        out["spread_vs_vol_spearman"] = float(stats.spearmanr(sm[ok], rv[ok]).statistic)
    for half, (lo, hi) in WINDOWS.items():
        k = (b["t"].to_numpy() > lo) & (b["t"].to_numpy() <= hi)
        out[f"spread_bp_median_{half}"] = float(np.nanmedian(sp[k]))
        out[f"depth10_usd_median_{half}"] = float(np.nanmedian((b["depth_bid_10"] + b["depth_ask_10"]).to_numpy()[k]))
    return out


# ------------------------------------------------------------------ H-MS6 workers


def _pair_worker(args: tuple) -> dict:
    target, leaders, cache = args
    tr = load_trades(list({target, *leaders}), Path(cache))
    series = {s: ms_prices(tr.filter(pl.col("symbol") == s)) for s in {target, *leaders}}
    rng = np.random.default_rng(SEED + sum(map(ord, target)))
    out: dict[str, Any] = {"target": target, "n_ms": int(len(series[target][0]))}
    tt, tp = series[target]
    sec_t = np.diff(grid_last(tt, tp, D0, D0 + DAY, 1000))
    for L in leaders:
        if L == target:
            continue
        lt, lp = series[L]
        sec_l = np.diff(grid_last(lt, lp, D0, D0 + DAY, 1000))
        res = {}
        for half, (lo, hi) in WINDOWS.items():
            C, v1, v2 = hy_blocks(lt, lp, tt, tp, LAGS_MS, (lo, hi))
            keep = np.ones(C.shape[1], bool)
            rho = hy_rho(C, v1, v2, keep)
            st = lead_stats(rho)
            st["rho"] = rho.round(5).tolist()
            st["boot"] = hy_bootstrap(C, v1, v2, keep, rng)
            Cf, f1, f2 = hy_blocks(lt, lp, tt, tp, FINE_MS, (lo, hi))
            st["rho_fine"] = hy_rho(Cf, f1, f2).round(5).tolist()
            if half == "test":
                bl = lo + np.arange(C.shape[1]) * BLOCK_MS
                keep_x = ~((bl + BLOCK_MS > CPI_EX[0]) & (bl < CPI_EX[1]))
                rx = hy_rho(C, v1, v2, keep_x)
                sx = lead_stats(rx)
                sx["boot"] = hy_bootstrap(C, v1, v2, keep_x, rng, B=300)
                st["ex_cpi"] = sx
            i0, i1 = (lo - D0) // 1000, (hi - D0) // 1000
            st["granger"] = granger(sec_t[i0:i1], sec_l[i0:i1])
            if half == "test":
                ks = np.arange(i0, i1)
                kk = ~((D0 + ks * 1000 >= CPI_EX[0]) & (D0 + ks * 1000 < CPI_EX[1]))
                # lags must not cross the dropped window: use the longer clean segment, 16:00-24:00
                st["granger_ex_cpi"] = granger(sec_t[i0:i1][kk][-8 * 3600:], sec_l[i0:i1][kk][-8 * 3600:])
            res[half] = st
        out[L] = res
    return out


def mid_hy(leaders: list[str], targets: list[str]) -> dict:
    """Robustness: HY on book mid changes (100 ms resolution, no bid-ask bounce) for the L2 symbols, TEST half."""
    mids = {}
    for s in set(leaders) | set(targets):
        b = pl.read_parquet(BOOK_DIR / f"{s}.parquet", columns=["ts", "mid"]).with_columns(pl.col("ts").dt.epoch("ms").cast(pl.Int64).alias("t"))
        b = b.filter(pl.col("mid").diff() != 0)
        mids[s] = (b["t"].to_numpy(), np.log(b["mid"].to_numpy()))
    out = {}
    for L in leaders:
        for tgt in targets:
            if tgt == L:
                continue
            rows = {}
            for half, w in WINDOWS.items():
                rho = hy_ccf(*mids[L], *mids[tgt], LAGS_MS, window=w)
                st = lead_stats(rho)
                rows[half] = st
            out[f"{L}->{tgt}"] = rows
    return out


# ------------------------------------------------------------------ run


def holm(p: dict[str, float]) -> dict[str, float]:
    ks = sorted(p, key=lambda k: p[k])
    m, out, run = len(ks), {}, 0.0
    for i, k in enumerate(ks):
        run = max(run, min(1.0, (m - i) * p[k]))
        out[k] = run
    return out


def run(cache: Path, workers: int = 4) -> dict:
    t0 = time.time()
    rng = np.random.default_rng(SEED)
    inst = instruments()
    trades = load_trades(None, cache)
    syms = sorted(trades["symbol"].unique().to_list())
    counts = dict(trades.group_by("symbol").len().iter_rows())
    cls = {s: symbol_class(s, inst) for s in syms}
    crypto = [s for s in syms if cls[s] != "tradfi"]
    tradfi = [s for s in syms if cls[s] == "tradfi"]
    out: dict[str, Any] = {"generated": datetime.now(UTC).isoformat(timespec="seconds"), "as_of_trades": AS_OF_TRADES.isoformat(),
                           "method": METHOD, "universe": {"crypto": crypto, "tradfi": tradfi, "trades": counts}}
    print(f"[{time.time() - t0:.0f}s] trades loaded: {trades.height} rows, {len(syms)} symbols", flush=True)

    # ---- book QA and validation against trades
    qa = json.loads((BOOK_DIR / "_qa.json").read_text()) if (BOOK_DIR / "_qa.json").exists() else {}
    from ai_trader.sim.book import validate_trades

    val, desc = {}, {}
    for s in L2:
        bk = pl.read_parquet(BOOK_DIR / f"{s}.parquet", columns=["cts", "bid1", "ask1"])
        tr = trades.filter(pl.col("symbol") == s).with_columns(pl.col("t").cast(pl.Datetime("ms")).dt.replace_time_zone("UTC").alias("event_ts"))
        val[s] = validate_trades(bk, tr)
        desc[s] = book_descriptives(s, (inst.get(s) or {}).get("tick_size"))
        desc[s]["qa"] = qa.get(s, {})
    out["validation"] = val
    out["book_descriptives"] = desc
    print(f"[{time.time() - t0:.0f}s] book validation + descriptives done", flush=True)

    # ---- H-MS5
    wide = [s for s in L2 if desc[s].get("spread_ticks_share", {}).get("1", 1.0) < 0.9]
    factors, fmeta = how_factors(list(L2))
    panel = pl.concat([ms5_rows(s, factors[s]) for s in L2])
    ms5: dict[str, Any] = {"how_factor": fmeta, "primary": {}, "robust": {}, "not_tick_bound": wide}
    for h in HORIZONS:
        ms5["primary"][h] = ms5_fit(panel, h)
        ms5["robust"][f"h{h}_ex_cpi"] = ms5_fit(panel, h, exclude_cpi=True)
        ms5["robust"][f"h{h}_cluster30"] = ms5_fit(panel, h, cluster_min=30)
        ms5["robust"][f"h{h}_level_spread"] = ms5_fit(panel, h, xname="spread")
        ms5["robust"][f"h{h}_crypto_only"] = ms5_fit(panel.filter(~pl.col("symbol").is_in(list(tradfi))), h)
        # exploratory (not registered): symbols whose spread is not pinned at one tick, where relative spread is not just tick / mid
        ms5["robust"][f"h{h}_not_tick_bound"] = ms5_fit(panel.filter(pl.col("symbol").is_in(wide)), h)
        ms5[f"per_symbol_h{h}"] = ms5_per_symbol(panel, h)
    out["ms5"] = ms5
    print(f"[{time.time() - t0:.0f}s] H-MS5 done", flush=True)

    # ---- H-MS6 pairwise (HY + Granger)
    targets = [s for s in syms if s not in ("ETHUSDT", "SOLUSDT", "BTCUSDT")] + ["ETHUSDT", "SOLUSDT", "BTCUSDT"]
    jobs = [(s, list(LEADERS), str(cache)) for s in targets]
    pair = {}
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for r in ex.map(_pair_worker, jobs):
            pair[r["target"]] = r
            print(f"  [{time.time() - t0:.0f}s] pair {r['target']}", flush=True)
    out["pairs"] = pair
    out["mid_hy"] = mid_hy(["ETHUSDT", "SOLUSDT", "BTCUSDT"], [s for s in L2 if s not in ("NVDAUSDT", "QQQUSDT", "XAUUSDT")] + ["NVDAUSDT", "QQQUSDT", "XAUUSDT"])
    print(f"[{time.time() - t0:.0f}s] pairs done", flush=True)

    # ---- H-MS6 Diebold-Yilmaz
    rvm = {}
    for s in syms:
        t, lp = ms_prices(trades.filter(pl.col("symbol") == s))
        rvm[s] = rv_minutes(t, lp)
    test_m = slice(720, 1440)
    fit_m = slice(0, 720)
    cover = {s: float(np.mean(np.isfinite(rvm[s][test_m]) & (rvm[s][test_m] > 0))) for s in syms}
    cover_fit = {s: float(np.mean(np.isfinite(rvm[s][fit_m]) & (rvm[s][fit_m] > 0))) for s in syms}
    elig = [s for s in crypto if s not in LEADERS and cover[s] >= MIN_DY_COVER and cover_fit[s] >= MIN_DY_COVER]
    elig = sorted(elig, key=lambda s: -counts[s])[:MAX_DY - 3]
    dy_set = ["BTCUSDT", "ETHUSDT", "SOLUSDT"] + elig

    def mat(names, sl):
        M = np.column_stack([_floor_log(np.where(np.isfinite(rvm[s][sl]) & (rvm[s][sl] > 0), rvm[s][sl], 0.0)) for s in names])
        return 0.5 * M  # log volatility = 0.5 log variance

    dy = {"coverage_test": cover, "coverage_fit": cover_fit, "set": dy_set}
    dy["test"] = dy_table(mat(dy_set, test_m), dy_set, boot=B_DY, rng=rng)
    dy["fit"] = dy_table(mat(dy_set, fit_m), dy_set, boot=B_DY, rng=rng)
    mins = np.arange(720, 1440)
    tm = D0 + mins * 60_000
    seg = np.where(tm < CPI_EX[0], 0, np.where(tm >= CPI_EX[1], 1, -1))
    dy["test_ex_cpi"] = dy_table(mat(dy_set, test_m), dy_set, seg=seg, boot=B_DY, rng=rng)
    tf = [s for s in tradfi if cover[s] >= 0.9]
    tf = sorted(tf, key=lambda s: -counts[s])[:8]
    dy["tradfi_set"] = ["BTCUSDT", "ETHUSDT", "SOLUSDT"] + tf
    if tf:
        dy["tradfi_test"] = dy_table(mat(dy["tradfi_set"], test_m), dy["tradfi_set"], boot=200, rng=rng)
    out["dy"] = dy
    print(f"[{time.time() - t0:.0f}s] DY done", flush=True)

    out["verdicts"] = verdicts(out)
    out["runtime_s"] = time.time() - t0
    return out


def _share(pairs: dict, leader: str, targets: list[str], half: str, key: str = "positive_peak", sub: str | None = None) -> tuple[int, int]:
    k = n = 0
    for tg in targets:
        if tg == leader or leader not in pairs.get(tg, {}):
            continue
        st = pairs[tg][leader][half]
        if sub:
            st = st[sub]
        n += 1
        k += int(bool(st[key]))
    return k, n


def verdicts(out: dict) -> dict:
    v: dict[str, Any] = {}
    # H-MS5
    prim = out["ms5"]["primary"]
    ok = all(prim[h]["test"]["coef"] > 0 and prim[h]["test"]["t"] > 2 and prim[h]["r2_oos_gain"] > 0 for h in HORIZONS)
    wrong = any((prim[h]["test"]["t"] < -1.645) or (prim[h]["p_gain"] > 0.95) for h in HORIZONS)
    p5 = max(max(prim[h]["p_coef"], prim[h]["p_gain"]) for h in HORIZONS)
    v["H-MS5"] = {"verdict": "supports" if ok else ("contradicts" if wrong else "inconclusive"), "p": p5}
    # H-MS6
    crypto = out["universe"]["crypto"]
    pairs = out["pairs"]
    comp = {}
    ps = []
    for L in ("ETHUSDT", "SOLUSDT", "BTCUSDT"):
        k, n = _share(pairs, L, crypto, "test")
        kx, nx = _share(pairs, L, crypto, "test", sub="ex_cpi")
        kf, nf = _share(pairs, L, crypto, "fit")
        net = out["dy"]["test"]["net"][L]
        ci = out["dy"]["test"]["net_ci"][L]
        pb = float(stats.binomtest(k, n, 0.5, alternative="greater").pvalue) if n else 1.0
        comp[L] = {"share_positive_peak_test": k / n if n else float("nan"), "k": k, "n": n, "share_fit": kf / nf if nf else float("nan"),
                   "share_test_ex_cpi": kx / nx if nx else float("nan"), "net_test": net, "net_ci": ci,
                   "p_net_le_0": out["dy"]["test"]["p_net_le_0"][L], "p_binom": pb}
        if L != "BTCUSDT":
            ps += [pb, comp[L]["p_net_le_0"]]
    ok6 = all(comp[L]["share_positive_peak_test"] > 0.7 and comp[L]["net_test"] > 0 for L in ("ETHUSDT", "SOLUSDT"))
    bad6 = any(comp[L]["share_positive_peak_test"] <= 0.5 or comp[L]["net_ci"][1] < 0 for L in ("ETHUSDT", "SOLUSDT"))
    v["H-MS6"] = {"verdict": "supports" if ok6 else ("contradicts" if bad6 else "inconclusive"), "p": max(ps), "components": comp}
    hp = holm({k: v[k]["p"] for k in ("H-MS5", "H-MS6")})
    for k in hp:
        v[k]["p_holm"] = hp[k]
        if v[k]["verdict"] == "supports" and hp[k] >= 0.05:
            v[k]["verdict"] = "inconclusive"
            v[k]["holm_downgrade"] = True
    return v


# ------------------------------------------------------------------ report tables


def _f(x, d=2):
    if x is None:
        return "n/a"
    try:
        if not np.isfinite(x):
            return "n/a"
    except TypeError:
        return str(x)
    return f"{x:.{d}f}"


def _cell(d: dict, k: str) -> str:
    return f"{_f(d[k]['coef'], 3)} ({_f(d[k]['t'])})" if k in d else "n/a"


def _ll_table(pairs: dict, targets: list[str]) -> str:
    L = ["| target | n ms (day) | ETH peak s / rho / rho0 / LLR | ETH boot P(+peak) | ETH->T F (p) | T->ETH F (p) | SOL peak s / rho / rho0 / LLR | SOL boot P(+peak) | SOL->T F (p) | T->SOL F (p) | BTC peak s / LLR | BTC->T F (p) |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for t in targets:
        r = pairs.get(t)
        if r is None:
            continue
        cells = [t, str(r["n_ms"])]
        for ld in ("ETHUSDT", "SOLUSDT", "BTCUSDT"):
            if ld not in r:
                cells += ["(self)", "", "", ""] if ld != "BTCUSDT" else ["(self)", ""]
                continue
            st = r[ld]["test"]
            g = st["granger"]
            gx = g.get("x_to_y", {}) if g.get("p") else {}
            gy = g.get("y_to_x", {}) if g.get("p") else {}
            if ld == "BTCUSDT":
                cells += [f"{_f(st['peak_lag_s'], 1)} / {_f(st['llr'])}", f"{_f(gx.get('F_robust'))} ({_f(gx.get('p_robust'), 3)})"]
            else:
                cells += [f"{_f(st['peak_lag_s'], 1)} / {_f(st['peak_rho'], 3)} / {_f(st['rho0'], 3)} / {_f(st['llr'])}", _f(st["boot"]["share_positive_peak"]),
                          f"{_f(gx.get('F_robust'))} ({_f(gx.get('p_robust'), 3)})", f"{_f(gy.get('F_robust'))} ({_f(gy.get('p_robust'), 3)})"]
        L.append("| " + " | ".join(cells) + " |")
    return "\n".join(L)


def _dy_tab(d: dict) -> str:
    L = ["| symbol | to others | from others | net | net 95% CI | P*(net <= 0) |", "|---|---|---|---|---|---|"]
    for s in d["symbols"]:
        c = d.get("net_ci", {}).get(s)
        ci = f"[{_f(c[0], 1)}, {_f(c[1], 1)}]" if c else "n/a"
        L.append(f"| {s} | {_f(d['to'][s], 1)} | {_f(d['from'][s], 1)} | {_f(d['net'][s], 1)} | {ci} | {_f(d.get('p_net_le_0', {}).get(s), 3)} |")
    tci = f"[{_f(d['total_ci'][0], 1)}, {_f(d['total_ci'][1], 1)}]" if "total_ci" in d else ""
    L.append(f"| total connectedness | {_f(d['total'], 1)}% | VAR lag {d['lag']} | H {d['H']} | {tci} | |")
    return "\n".join(L)


def render_tables(res: dict) -> dict[str, str]:
    """Markdown tables from the raw result (the prose around them is written in the report)."""
    T: dict[str, str] = {}
    L = ["| symbol | messages | snapshots | u gaps | crossed bins | trades | strict outside [bid1, ask1] | loose (before or after) | sweep share | loose ex sweeps | violations in sweeps | wrong side |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for s, v in res["validation"].items():
        q = res["book_descriptives"][s].get("qa", {})
        L.append(f"| {s} | {q.get('messages', 'n/a')} | {q.get('snapshots', 'n/a')} | {q.get('u_gaps', 'n/a')} | {q.get('crossed_bins', 'n/a')} | {v['n_trades']} | "
                 f"{100 * v['strict_violation_rate']:.2f}% | {100 * v['loose_violation_rate']:.2f}% | {100 * v['sweep_share']:.1f}% | "
                 f"{100 * v['loose_violation_rate_ex_sweeps']:.2f}% | {_f(100 * v['violations_in_sweeps_share'], 0)}% | {100 * v['wrong_side_rate']:.2f}% |")
    T["validation"] = "\n".join(L)

    m5 = res["ms5"]
    prim = {int(k): v for k, v in m5["primary"].items()}
    L = ["| spec | h | n FIT / TEST | FIT coef (t) | TEST coef (t) | OOS R^2 full | OOS R^2 without spread | gain | DM t |", "|---|---|---|---|---|---|---|---|---|"]
    rows = [("primary (log spread)", prim[h]) for h in HORIZONS]
    for k in sorted(m5["robust"]):
        rows.append((k.split("_", 1)[1], m5["robust"][k]))
    for name, r in rows:
        L.append(f"| {name} | {r['h']} | {r['n_fit']} / {r['n_test']} | {_f(r['fit']['coef'], 3)} ({_f(r['fit']['t'])}) | {_f(r['test']['coef'], 3)} ({_f(r['test']['t'])}) | "
                 f"{_f(r['r2_oos_full'], 3)} | {_f(r['r2_oos_restricted'], 3)} | {_f(r['r2_oos_gain'], 4)} | {_f(r['dm_t'])} |")
    T["ms5"] = "\n".join(L)
    L = ["| control | FIT h5 | TEST h5 | FIT h15 | TEST h15 |", "|---|---|---|---|---|"]
    for c in ["lrv5p", "lrv60p", "ldepth10", "lhow"]:
        cs = [prim[h][half]["controls"][c] for h in HORIZONS for half in ("fit", "test")]
        L.append(f"| {c} | " + " | ".join(f"{_f(a[0], 3)} ({_f(a[1])})" for a in cs) + " |")
    T["ms5_controls"] = "\n".join(L)
    L = ["| symbol | h5 FIT coef (t) | h5 TEST coef (t) | h15 FIT coef (t) | h15 TEST coef (t) | OOS gain h5 (pooled model) | OOS gain h15 |", "|---|---|---|---|---|---|---|"]
    for s in L2:
        a, b = m5["per_symbol_h5"].get(s, {}), m5["per_symbol_h15"].get(s, {})
        L.append(f"| {s} | {_cell(a, 'fit')} | {_cell(a, 'test')} | {_cell(b, 'fit')} | {_cell(b, 'test')} | "
                 f"{_f(prim[5]['r2_gain_by_symbol'].get(s), 4)} | {_f(prim[15]['r2_gain_by_symbol'].get(s), 4)} |")
    T["ms5_symbols"] = "\n".join(L)

    pairs, uni = res["pairs"], res["universe"]
    T["leadlag_crypto"] = _ll_table(pairs, sorted(uni["crypto"], key=lambda s: -uni["trades"][s]))
    T["leadlag_tradfi"] = _ll_table(pairs, sorted(uni["tradfi"], key=lambda s: -uni["trades"][s]))

    L = ["| leader | half | targets | peak at 0 | peak in [0.1, 30] s | peak < 0 | median LLR | LLR > 1, P*(LLR<=1) < 0.05 | LLR < 1, P*(LLR<=1) > 0.95 | Granger L->T p < 0.01 | T->L p < 0.01 |",
         "|---|---|---|---|---|---|---|---|---|---|---|"]
    for group, tg in (("crypto", uni["crypto"]), ("tradfi", uni["tradfi"])):
        for ld in ("ETHUSDT", "SOLUSDT", "BTCUSDT"):
            for half in (("fit", "test", "test_ex_cpi") if group == "crypto" else ("fit", "test")):
                sts = []
                for t in tg:
                    if t == ld or ld not in pairs.get(t, {}):
                        continue
                    st = pairs[t][ld]["fit" if half == "fit" else "test"]
                    sts.append((st["ex_cpi"] if half == "test_ex_cpi" else st, st))
                gk = "granger_ex_cpi" if half == "test_ex_cpi" else "granger"
                row = [len(sts), sum(a["peak_lag_s"] == 0 for a, _ in sts), sum(a["positive_peak"] for a, _ in sts), sum(a["peak_lag_s"] < 0 for a, _ in sts)]
                med = _f(float(np.nanmedian([a["llr"] for a, _ in sts])))
                sp = sum(a["llr"] > 1 and a["boot"]["p_llr_le_1"] < 0.05 for a, _ in sts)
                sn = sum(a["llr"] < 1 and a["boot"]["p_llr_le_1"] > 0.95 for a, _ in sts)
                g1 = sum(bool(b[gk].get("p")) and b[gk]["x_to_y"]["p_robust"] < 0.01 for _, b in sts)
                g2 = sum(bool(b[gk].get("p")) and b[gk]["y_to_x"]["p_robust"] < 0.01 for _, b in sts)
                L.append(f"| {ld} -> {group} | {half} | " + " | ".join(str(x) for x in row) + f" | {med} | {sp} | {sn} | {g1} | {g2} |")
    T["leadlag_summary"] = "\n".join(L)

    L = ["| leader | half | median rho over crypto targets at lag " + " / ".join(str(int(x)) for x in FINE_MS) + " ms |", "|---|---|---|"]
    for ld in ("ETHUSDT", "SOLUSDT", "BTCUSDT"):
        for half in ("fit", "test"):
            R = np.array([pairs[t][ld][half]["rho_fine"] for t in uni["crypto"] if t != ld and ld in pairs.get(t, {})])
            L.append(f"| {ld} | {half} | " + " / ".join(_f(x, 3) for x in np.nanmedian(R, axis=0)) + " |")
    T["fine"] = "\n".join(L)

    L = ["| pair (book mid changes, 100 ms) | FIT peak s / rho / LLR | TEST peak s / rho / LLR |", "|---|---|---|"]
    for k, v in res["mid_hy"].items():
        L.append(f"| {k} | {_f(v['fit']['peak_lag_s'], 1)} / {_f(v['fit']['peak_rho'], 3)} / {_f(v['fit']['llr'])} | "
                 f"{_f(v['test']['peak_lag_s'], 1)} / {_f(v['test']['peak_rho'], 3)} / {_f(v['test']['llr'])} |")
    T["mid_hy"] = "\n".join(L)

    dy = res["dy"]
    T["dy_test"] = _dy_tab(dy["test"])
    T["dy_fit"] = _dy_tab(dy["fit"])
    T["dy_ex_cpi"] = _dy_tab(dy["test_ex_cpi"])
    if "tradfi_test" in dy:
        T["dy_tradfi"] = _dy_tab(dy["tradfi_test"])
    syms = dy["test"]["symbols"]
    L = ["| from \\ to (TEST, % of target's 10-step FEV) | " + " | ".join(s.replace("USDT", "") for s in syms) + " |", "|---" * (len(syms) + 1) + "|"]
    for src in ("BTCUSDT", "ETHUSDT", "SOLUSDT"):
        L.append(f"| {src} | " + " | ".join("-" if src == t else _f(dy["test"]["pairwise_pct"][f"{src}->{t}"], 1) for t in syms) + " |")
    T["dy_pairwise"] = "\n".join(L)

    bd = res["book_descriptives"]
    L = ["| symbol | tick bp | spread bp p5 / p50 / p95 / p99 | mean | 1-tick share | depth5 USD p50 | depth10 USD p5 / p50 / p95 | depth25 USD p50 | book span bp p50 | 10 / 25 bp band truncated | top level USD p50 | updates/s |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for s in L2:
        d = bd[s]
        q = d["spread_bp_q"]
        tr = d.get("band_truncated_share", {})
        L.append(f"| {s} | {_f(d.get('tick_bp'), 3)} | {_f(q['0.05'], 3)} / {_f(q['0.5'], 3)} / {_f(q['0.95'], 3)} / {_f(q['0.99'], 3)} | {_f(d['spread_bp_mean'], 3)} | "
                 f"{_f(100 * d.get('spread_ticks_share', {}).get('1', float('nan')), 1)}% | {d['depth5_usd_q']['0.5']:,.0f} | "
                 f"{d['depth10_usd_q']['0.05']:,.0f} / {d['depth10_usd_q']['0.5']:,.0f} / {d['depth10_usd_q']['0.95']:,.0f} | {d['depth25_usd_q']['0.5']:,.0f} | "
                 f"{_f(d.get('book_span_bp_median'), 1)} | {_f(100 * tr.get('10', float('nan')), 0)}% / {_f(100 * tr.get('25', float('nan')), 0)}% | "
                 f"{d['top_level_usd_median']:,.0f} | {_f(d['updates_per_s'], 1)} |")
    T["desc_spread_depth"] = "\n".join(L)
    L = ["| symbol | spread ACF 1 s, lags 1 / 10 / 60 / 300 s | spread ACF 1 m, lags 1 / 5 / 15 / 60 min | elasticity d log spread / d log RV (t) | Spearman(spread, RV), 1 m | spread p50 FIT / TEST | depth10 p50 FIT / TEST |",
         "|---|---|---|---|---|---|---|"]
    for s in L2:
        d = bd[s]
        a1, am = d["spread_acf_1s"], d["spread_acf_1m"]
        e = d.get("elasticity_spread_on_rv", {})
        L.append(f"| {s} | {_f(a1['1'])} / {_f(a1['10'])} / {_f(a1['60'])} / {_f(a1['300'])} | {_f(am['1'])} / {_f(am['5'])} / {_f(am['15'])} / {_f(am['60'])} | "
                 f"{_f(e.get('coef'), 3)} ({_f(e.get('t'))}) | {_f(d.get('spread_vs_vol_spearman'))} | {_f(d['spread_bp_median_fit'], 3)} / {_f(d['spread_bp_median_test'], 3)} | "
                 f"{d['depth10_usd_median_fit']:,.0f} / {d['depth10_usd_median_test']:,.0f} |")
    T["desc_dyn"] = "\n".join(L)
    return T


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args(argv)
    res = run(a.cache, min(4, a.workers))
    p = a.out or (a.cache / "book_spillover_raw.json")
    p.write_text(json.dumps(res, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o), indent=1))
    print("wrote", p, json.dumps(res["verdicts"], default=str)[:2000])


if __name__ == "__main__":
    sys.exit(main())
