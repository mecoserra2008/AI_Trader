"""Reaction functions of aggressive order flow on 3 s and 1 m bars (H-SIM3).

Per symbol and UTC day, on contiguous ok segments (lags never cross a gap):

  direction equation (signed taker notional, scaled by depth):
      y_t = F_t / D  with F = buy - sell notional, D = the day's median bid + ask notional within 25 bps
      y_t = c + b1 r_{t-1} + b2 sum r_{t-2..t-5} + b3 sum r_{t-6..t-20} + k obi5_{t-1} + f1 y_{t-1} + f2 sum y_{t-2..t-5}
            + v vol_{t-1} + e
      r in units of the day's 3 s sigma; vol_{t-1} = mean |r_{t-1..t-20}| (sigma units)
  intensity equation:
      log(1 + n_t) = c + s log spread_{t-1} + v vol_{t-1} + h1 log(1 + n_{t-1}) + h2 mean log(1 + n_{t-2..t-20}) + e

Primary responses (registered as "response to past returns, book imbalance and spread"):
  momentum  = b1 + 4 b2 + 15 b3   cumulative flow response (depth units) over 1 min to a 1-sigma return per candle
              (> 0 momentum takers, < 0 contrarian takers)
  obi       = k                   book-imbalance takers (buy when the bid side is deeper)
  spread    = s                   liquidity-sensitive takers (trade less when the spread is wide)
The signed direction equation has no sign relation to the symmetric spread, so the spread response is measured in
the intensity equation (stated in the METHOD of the study).
Trade-size scaling is the alternative scale: y'_t = F_t / (m * nbar), m = median notional per trade, nbar = mean
trades per candle (reported alongside).
Standard errors: cluster-robust by 5-minute clock blocks (clusters are shared across symbols, so stacked
influence functions give the cross-symbol covariance for the within-class Wald tests).
VAR(5) on (y, r): cumulative 20-step responses of flow to a unit return shock and of returns to a unit flow shock.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy import stats

from ai_trader.sim.flow_data import STEP_MS, Grid

CLUSTER_MS = 5 * 60_000
DIR_NAMES = ("const", "r1", "r2_5", "r6_20", "obi5", "y1", "y2_5", "vol")
INT_NAMES = ("const", "log_spread", "vol", "a1", "a2_20")
PRIMARY = ("momentum", "obi", "spread")


def _shift(x, k):
    out = np.full(len(x), np.nan)
    if k < len(x):
        out[k:] = x[:-k] if k else x
    return out


def _lagsum(x, a, b):
    c = np.r_[0.0, np.cumsum(np.nan_to_num(x))]
    out = np.full(len(x), np.nan)
    t = np.arange(b, len(x))
    out[b:] = c[t - a + 1] - c[t - b]
    return out


@dataclass
class Design:
    y_dir: np.ndarray
    X_dir: np.ndarray
    y_int: np.ndarray
    X_int: np.ndarray
    y_trade: np.ndarray     # trade-size scaled flow
    clus: np.ndarray        # 5-minute clock block id
    r: np.ndarray           # sigma units (for VAR)


def design(g: Grid, mask: np.ndarray, *, agg: int = 1, lags: tuple[int, int, int] = (1, 5, 20), min_len: int = 60) -> Design | None:
    """Regression design for one symbol-day. agg = candles per bar (1 = 3 s, 20 = 1 m)."""
    gg = g.slice(mask)
    if agg > 1:
        gg = _aggregate(gg, agg)
    D = float(np.nanmedian(gg.depth[gg.ok])) if gg.ok.any() else float("nan")
    if not (D > 0):
        return None
    segs = gg.segments(min_len)
    if not segs:
        return None
    parts = []
    l1, l2, l3 = lags
    for a, b in segs:
        sl = slice(a, b)
        mid = gg.mid[sl]
        r = np.r_[np.nan, np.diff(np.log(mid))]
        F = gg.buy[sl] - gg.sell[sl]
        n = gg.n_buy[sl] + gg.n_sell[sl]
        V = gg.buy[sl] + gg.sell[sl]
        parts.append(dict(r=r, F=F, n=n, V=V, obi=gg.obi_5[sl], s=gg.spread_bps[sl], t=gg.t_ms[sl]))
    cat = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    sig = float(np.nanstd(cat["r"]))
    if not (sig > 0):
        return None
    tr = cat["n"] > 0
    m_trade = float(np.median(cat["V"][tr] / cat["n"][tr])) if tr.any() else float("nan")
    nbar = float(cat["n"].mean())
    rows = []
    for p in parts:
        r = p["r"] / sig
        y = p["F"] / D
        a = np.log1p(p["n"])
        absr = np.abs(np.nan_to_num(r))
        vol = _lagsum(absr, 1, l3) / l3
        X_dir = np.column_stack([np.ones(len(r)), _shift(r, 1), _lagsum(r, 2, l2), _lagsum(r, l2 + 1, l3), _shift(p["obi"], 1),
                                 _shift(y, 1), _lagsum(y, 2, l2), vol])
        X_int = np.column_stack([np.ones(len(r)), _shift(np.log(p["s"]), 1), vol, _shift(a, 1), _lagsum(a, 2, l3) / (l3 - 1)])
        X_dir[:l3 + 1] = np.nan
        X_int[:l3 + 1] = np.nan
        rows.append((y, X_dir, a, X_int, p["F"] / (m_trade * max(nbar, 1e-9)), p["t"] // CLUSTER_MS, r))
    return Design(*(np.concatenate([x[i] for x in rows]) for i in range(7)))


def _aggregate(g: Grid, k: int) -> Grid:
    """k consecutive 3 s candles -> one bar (only bars whose k candles are all ok are ok)."""
    n = len(g) // k
    if n == 0:
        return g
    t = g.t_ms[: n * k].reshape(n, k)

    def rs(x):
        return x[: n * k].reshape(n, k)

    ok = rs(g.ok).all(axis=1)
    hi = np.nanmax(np.where(np.isfinite(rs(g.high)), rs(g.high), -np.inf), axis=1)
    lo = np.nanmin(np.where(np.isfinite(rs(g.low)), rs(g.low), np.inf), axis=1)
    return Grid(g.symbol, t[:, 0], ok, rs(g.present).all(axis=1), rs(g.mid)[:, -1], np.nanmean(rs(g.spread_bps), axis=1),
                np.where(np.isfinite(hi), hi, np.nan), np.where(np.isfinite(lo), lo, np.nan), rs(g.buy).sum(axis=1),
                rs(g.sell).sum(axis=1), rs(g.n_buy).sum(axis=1), rs(g.n_sell).sum(axis=1), np.nanmean(rs(g.obi_5), axis=1),
                np.nanmean(rs(g.obi_25), axis=1), np.nanmean(rs(g.depth), axis=1))


@dataclass
class OLSFit:
    beta: np.ndarray
    cov: np.ndarray
    n: int
    clusters: np.ndarray     # cluster ids present
    infl: np.ndarray         # (n_clusters, k) cluster sums of influence functions
    r2: float


def ols_cluster(y: np.ndarray, X: np.ndarray, clus: np.ndarray) -> OLSFit | None:
    m = np.isfinite(y) & np.isfinite(X).all(axis=1)
    if m.sum() < 5 * X.shape[1] + 50:
        return None
    y, X, cl = y[m], X[m], clus[m]
    XtX = X.T @ X
    try:
        A = np.linalg.inv(XtX)
    except np.linalg.LinAlgError:
        return None
    beta = A @ (X.T @ y)
    e = y - X @ beta
    u, inv = np.unique(cl, return_inverse=True)
    sc = np.zeros((len(u), X.shape[1]))
    np.add.at(sc, inv, X * e[:, None])
    infl = sc @ A.T          # cluster sums of influence: A x_t e_t
    G = len(u)
    adj = G / max(G - 1, 1)
    cov = adj * infl.T @ infl
    r2 = 1 - (e @ e) / max(((y - y.mean()) ** 2).sum(), 1e-300)
    return OLSFit(beta, cov, int(m.sum()), u, infl * math.sqrt(adj), float(r2))


MOM_W = np.array([0, 1, 4, 15, 0, 0, 0, 0], dtype=float)  # momentum = b1 + 4 b2 + 15 b3 (3 s lags 1, 2-5, 6-20)


def mom_weights(lags: tuple[int, int, int]) -> np.ndarray:
    l1, l2, l3 = lags
    return np.array([0, 1, l2 - 1, l3 - l2, 0, 0, 0, 0], dtype=float)


def primary(fd: OLSFit, fi: OLSFit, lags=(1, 5, 20)) -> dict:
    """Primary responses with clustered SEs and their influence rows (n_clusters x 3) for stacking."""
    w = mom_weights(lags)
    e_obi = np.zeros(len(DIR_NAMES))
    e_obi[DIR_NAMES.index("obi5")] = 1
    e_spr = np.zeros(len(INT_NAMES))
    e_spr[INT_NAMES.index("log_spread")] = 1
    est = {"momentum": float(w @ fd.beta), "obi": float(e_obi @ fd.beta), "spread": float(e_spr @ fi.beta)}
    se = {"momentum": float(math.sqrt(w @ fd.cov @ w)), "obi": float(math.sqrt(e_obi @ fd.cov @ e_obi)), "spread": float(math.sqrt(e_spr @ fi.cov @ e_spr))}
    return {"est": est, "se": se, "infl_dir": fd.infl @ np.column_stack([w, e_obi]), "clus_dir": fd.clusters,
            "infl_int": fi.infl @ e_spr[:, None], "clus_int": fi.clusters}


def var_irf(y: np.ndarray, r: np.ndarray, p: int = 5, h: int = 20, seg_break: np.ndarray | None = None) -> dict | None:
    """Bivariate VAR(p) on (y, r) by OLS; cumulative h-step responses to unit (non-orthogonalised) shocks."""
    Z = np.column_stack([y, r])
    n = len(Z)
    if n < 20 * p + 100:
        return None
    rows = [np.ones(n - p)] + [Z[p - k - 1:n - k - 1, j] for k in range(p) for j in range(2)]
    X = np.column_stack(rows)
    Y = Z[p:]
    ok = np.isfinite(X).all(axis=1) & np.isfinite(Y).all(axis=1)
    if seg_break is not None:
        ok &= seg_break[p:]
    if ok.sum() < 20 * p + 50:
        return None
    B, *_ = np.linalg.lstsq(X[ok], Y[ok], rcond=None)
    A = [B[1 + 2 * k:3 + 2 * k].T for k in range(p)]  # A_k (2x2): Z_t = sum A_k Z_{t-k}
    resp = {}
    for j, name in ((1, "ret_shock"), (0, "flow_shock")):
        hist = [np.zeros(2) for _ in range(p)]
        x0 = np.zeros(2)
        x0[j] = 1.0
        path = [x0]
        hist = [x0] + hist[:-1]
        for _ in range(h):
            nxt = sum(A[k] @ hist[k] for k in range(p))
            path.append(nxt)
            hist = [nxt] + hist[:-1]
        cum = np.sum(path[1:], axis=0)
        resp[name] = cum
    return {"flow_to_ret": float(resp["ret_shock"][0]), "ret_to_flow": float(resp["flow_shock"][1]),
            "flow_persist": float(resp["flow_shock"][0])}


def fit_symbol_day(g: Grid, mask: np.ndarray, agg: int = 1) -> dict | None:
    """All fits for one symbol-day (3 s if agg = 1, 1 m if agg = 20)."""
    lags = (1, 5, 20) if agg == 1 else (1, 5, 15)
    d = design(g, mask, agg=agg, lags=lags, min_len=60 if agg == 1 else 30)
    if d is None:
        return None
    fd = ols_cluster(d.y_dir, d.X_dir, d.clus)
    fi = ols_cluster(d.y_int, d.X_int, d.clus)
    if fd is None or fi is None:
        return None
    ft = ols_cluster(d.y_trade, d.X_dir, d.clus)
    out = {"n": fd.n, "r2_dir": fd.r2, "r2_int": fi.r2,
           "dir": {k: float(v) for k, v in zip(DIR_NAMES, fd.beta, strict=True)},
           "dir_t": {k: float(v / math.sqrt(c)) if c > 0 else float("nan") for k, v, c in zip(DIR_NAMES, fd.beta, np.diag(fd.cov), strict=True)},
           "int": {k: float(v) for k, v in zip(INT_NAMES, fi.beta, strict=True)},
           "int_t": {k: float(v / math.sqrt(c)) if c > 0 else float("nan") for k, v, c in zip(INT_NAMES, fi.beta, np.diag(fi.cov), strict=True)}}
    pr = primary(fd, fi, lags)
    out["primary"] = pr["est"]
    out["primary_se"] = pr["se"]
    if ft is not None:
        w = mom_weights(lags)
        out["momentum_trade_scaled"] = float(w @ ft.beta)
        out["obi_trade_scaled"] = float(ft.beta[DIR_NAMES.index("obi5")])
    v = var_irf(d.y_dir, d.r, p=5, h=20, seg_break=np.isfinite(d.X_dir).all(axis=1))
    if v:
        out["var"] = v
    return out


def fit_pooled(g: Grid, masks: list[np.ndarray], agg: int = 1) -> dict | None:
    """One symbol pooled over days (each day scaled by its own depth and sigma): primary responses + influence rows."""
    lags = (1, 5, 20) if agg == 1 else (1, 5, 15)
    ds = [d for d in (design(g, m, agg=agg, lags=lags, min_len=60 if agg == 1 else 30) for m in masks) if d is not None]
    if not ds:
        return None
    cat = Design(*(np.concatenate([getattr(d, f) for d in ds]) for f in ("y_dir", "X_dir", "y_int", "X_int", "y_trade", "clus", "r")))
    fd = ols_cluster(cat.y_dir, cat.X_dir, cat.clus)
    fi = ols_cluster(cat.y_int, cat.X_int, cat.clus)
    if fd is None or fi is None:
        return None
    pr = primary(fd, fi, lags)
    return {"est": pr["est"], "se": pr["se"], "infl_dir": pr["infl_dir"], "clus_dir": pr["clus_dir"], "infl_int": pr["infl_int"],
            "clus_int": pr["clus_int"], "n": fd.n}


def stacked_wald(blocks: list[tuple[np.ndarray, np.ndarray, np.ndarray]]) -> dict:
    """Wald test that a coefficient vector is equal across symbols.

    blocks: per symbol (est (k,), cluster ids (G_i,), influence (G_i, k)); clusters shared across symbols, so
    cov(b_i, b_j) = sum_c psi_ic psi_jc'. H0: b_i = b_1 for all i."""
    if len(blocks) < 2:
        return {"W": float("nan"), "df": 0, "p": float("nan"), "symbols": len(blocks)}
    k = len(blocks[0][0])
    allc = np.unique(np.concatenate([b[1] for b in blocks]))
    S = len(blocks)
    Psi = np.zeros((len(allc), S * k))
    theta = np.zeros(S * k)
    for i, (est, cl, inf) in enumerate(blocks):
        pos = np.searchsorted(allc, cl)
        Psi[pos, i * k:(i + 1) * k] = inf
        theta[i * k:(i + 1) * k] = est
    V = Psi.T @ Psi
    R = np.zeros(((S - 1) * k, S * k))
    for i in range(1, S):
        for j in range(k):
            R[(i - 1) * k + j, i * k + j] = 1
            R[(i - 1) * k + j, j] = -1
    d = R @ theta
    M = R @ V @ R.T
    rank = int(np.linalg.matrix_rank(M))
    W = float(d @ np.linalg.pinv(M) @ d)
    return {"W": W, "df": rank, "p": float(stats.chi2.sf(W, rank)) if rank > 0 else float("nan"), "symbols": S, "clusters": int(len(allc))}


def sign_stability(values: list[float]) -> dict:
    v = np.array([x for x in values if np.isfinite(x)])
    if len(v) == 0:
        return {"n": 0, "modal_sign": 0, "share": float("nan")}
    pos = float((v > 0).mean())
    return {"n": int(len(v)), "modal_sign": 1 if pos >= 0.5 else -1, "share": max(pos, 1 - pos), "median": float(np.median(v))}


__all__ = ["CLUSTER_MS", "DIR_NAMES", "INT_NAMES", "PRIMARY", "STEP_MS", "design", "fit_symbol_day", "ols_cluster", "primary",
           "sign_stability", "stacked_wald", "var_irf"]
