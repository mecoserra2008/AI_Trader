"""PnL decomposition of the strategy book (Program 2, block PD: H-PD1..H-PD4; agent B1).

Input: data/priors/strategy_panel_daily.parquet, ..._trades.parquet and ..._context.parquet (research.strategy_panel).
All comparisons use pnl_d (daily P&L / sqrt(bars per day): daily-sigma units, scale-free across 1h and 4h).
Segments: llm_in_sample before research.knowledge_cutoff (2026-07-01), clean_holdout on/after (by P&L date, entry date
for trades, target-week start for IPCA). Everything below was fixed before any PD result was computed.

Sleeves
-------
A sleeve is strategy x universe; its daily P&L is the mean over the universe's symbols present that date of pnl_d
(inactive days count as 0). T sleeves need >= 3 symbols that date. Sleeves with < 60 days or zero variance in a
segment are dropped from that segment.

H-PD1 (PCA / effective number of bets)
--------------------------------------
Per segment: correlation matrix of the sleeves on the dates where all included sleeves exist (in-sample: L only, since
T sleeves start in 2026-06..08; holdout: L + T, and L-only reported alongside). Eigenvalues lam; top-3 share
sum(lam_1..3)/sum(lam); effective number of bets ENB = exp(-sum p log p), p = lam / sum(lam) (Meucci 2009 entropy
form, as registered). CI: 1000 circular moving-block bootstrap draws of dates (10-day blocks). Verdict per segment on
the point estimates (registered: share > 0.6 AND ENB < 6); bootstrap p = max(P*(share <= 0.6), P*(ENB >= 6)) goes into
Holm.

H-PD2 (returns-based factor attribution)
----------------------------------------
Daily factors, point in time (context table; z = ret_d / sigma_d with sigma_d = sqrt(mean rv over d-30..d-1), >= 20
days, i.e. vol-standardised by a trailing estimate known at the start of d):
  MKT    mean_i z_i over universe L
  TSMOM  mean_i sign(sum_{k=1..30} ret_{i,d-k}) z_i            (Moskowitz, Ooi & Pedersen 2012, 30-day lookback)
  VOL    mean_i log rv_{i,d} - mean_i log rv_{i,d-1}            (assets with both; features.realized rv from 1h bars)
  CARRY  sum_i w_i z_i, w = -(rank funding_sod - mean rank) / sum|.| * 2 (long low funding, short high; >= 5 assets;
         funding payments themselves are not added)
  LIQ    sum_i w_i z_i, w = +(rank Amihud - mean rank) / sum|.| * 2, Amihud = mean_{d-30..d-1} |ret| / USD turnover
  QQQ    z of QQQUSDT;  GOLD  z of XAUUSDT where it exists, else PAXGUSDT;  OIL  z of CLUSDT
Factor sets are segment specific because the TradFi perps start in 2025-2026: llm_in_sample core = MKT TSMOM VOL CARRY
LIQ (2020+), plus an "extended" robustness fit adding GOLD (PAXG, 2025+); clean_holdout = all eight. OLS with an
intercept on every date where the sleeve and all factors exist, NW HAC SEs (Bartlett, lag floor(4 (n/100)^(2/9))).
Alpha = intercept (daily-sigma units per day; annualised x365), its t; residual information ratio. Verdict per
segment (registered): median R^2 > 0.4 AND no sleeve has a residual alpha with t > 2 in BOTH segments (only L sleeves
have both); an alpha counts as t > 2 only if it is also Holm-significant (one-sided p) within the PD block.
Families: average-linkage hierarchical clustering (scipy) of the L sleeves on their in-sample standardised loadings
(beta_k sd(f_k) / sd(sleeve)) on the core factors; the number of clusters is the largest gap in merge heights
(2..6). The context-ranking table is repeated per cluster family.

H-PD3 (IPCA vs trailing Sharpe)
-------------------------------
Units: strategy x symbol in universe L (T is too short for weekly expanding fits; not used). Decision at each Monday
00:00 UTC; target r = sum of pnl_d over that Monday..Sunday (7 dates required). Characteristics known at the
decision: context c (per symbol) = shares of up / down direction and low / high vol states among the 7 start-of-day
cells Tue..Mon, vol_pct (percentile of the Monday start-of-day robust sigma among its trailing 365 daily values,
>= 120, minus 0.5), funding z (vs trailing 30 daily start-of-day values, clipped +-3, /3), trend strength
(|dir_score_sod| clipped at 3, /3), drawdown flag (ln(max close of the last 30 days / last close) > 1.5 sigma_30
sqrt(30), as tools.market_state); strategy-specific = own trailing-90-day Sharpe (>= 60 days, clipped +-5, /5) and
weekend share (share of the trailing-90-day exposure on Sat/Sun UTC dates, minus 2/7). Instruments z = family dummy x
[1, c] for the 6 registered families (6 x 9) plus the two strategy-specific terms: L = 56.
IPCA (Kelly, Pruitt & Su 2019): r_{i,t+1} = z_{i,t}' Gamma f_{t+1} + e, alternating least squares on managed
portfolios (W_t = Z'Z/N, x_t = Z'r/N): f_t = (G'W_tG)^-1 G'x_t; vec(G) from sum_t (W_t (x) f f') vec(G) = sum_t x_t (x) f_t;
normalisation G'G = I, F F' diagonal descending, mean f > 0; SVD start; tol 1e-6. Expanding window refit every week
(warm start) once >= 52 training weeks exist; forecast E[r_{i,t+1}] = z_{i,t}' G lambda, lambda = mean of fitted f.
Strategy score = mean over symbols of the forecast. Benchmark: trailing-90-day daily Sharpe of the strategy's L sleeve.
IC_t = Spearman across strategies of score vs the realised next-week sleeve P&L (mean over symbols of r). K in
{1, 2, 3} is chosen as the highest mean IC over llm_in_sample weeks only (pseudo out of sample). Verdict (registered):
IC difference (IPCA - Sharpe) > 0 with NW t > 2 (and Holm p < 0.05) in llm_in_sample, > 0 in clean_holdout.

H-PD4 (trade-level decomposition)
---------------------------------
Trades from the panel (maximal same-sign runs; strategy_panel docstring), all directional strategies at 1h and 4h, in
daily-sigma units (/ sqrt(bars per day)). Per trade gross G, cost C, favourable excursion MFE >= max(G, 0). Pool:
D = sum MFE (the gross positive P&L the signals produced); kept = sum max(G, 0); exit leakage E = D - kept; direction
loss = sum max(-G, 0); costs = sum C. Identity: net = D - E - direction loss - C. Registered share = (C + E) / D.
Sensitivity: (C + E_win) / sum max(G, 0), E_win = sum over G > 0 of (MFE - G). CI: 2000 bootstrap draws resampling
entry weeks. Verdict: share > 0.5 with CI lower bound > 0.5 and Holm p < 0.05 -> supports; share <= 0.5 ->
contradicts; else inconclusive.

Holm within the PD block, per segment, over {PD1 p, every L-sleeve PD2 alpha p (one-sided), PD3 p, PD4 p}.

Labelled extensions (requested by the orchestrator from the literature notes after registration and before any PD
result was read; descriptive, never part of a verdict):
  * PD1: Meucci (2009) ENB of the actual equal-weight book (entropy of the principal portfolios' risk contributions on
    the covariance matrix), and the registered ENB per v0 cell (sleeve P&L restricted per date to the symbols in that
    cell, >= 3 symbols, >= 30 days).
  * PD2: a convexity factor (Fung & Hsieh 2001: trend following ~ a lookback straddle), CONVEX = |MKT_d| minus its
    trailing 90-day mean (known at the start of d); attribution reported with and without it.
  * PD3: IPCA vs trailing-30d and trailing-365d sleeve Sharpe (factor momentum, Ehsani & Linnainmaa 2022).

    uv run python -m ai_trader.research.pnl_decomposition_study [--record]
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import NormalDist

import numpy as np
import polars as pl

from ai_trader.core.config import load
from ai_trader.research.evaluate import knowledge_cutoff, newey_west_t
from ai_trader.research.strategy_panel import (SEGS, block_indices, context_path, context_table, holm, panel_path, priors_dir,
                                               spearman, trades_path)

FAMILIES = ("trend", "mean_reversion", "volatility", "positioning", "calendar", "shock")
CORE = ("MKT", "TSMOM", "VOL", "CARRY", "LIQ")
ALL_F = ("MKT", "TSMOM", "VOL", "CARRY", "LIQ", "QQQ", "GOLD", "OIL")
MIN_DAYS = 60
N_BOOT_PCA = 1000
N_BOOT = 2000
BLOCK = 10
IPCA_MIN_WEEKS = 52
REPORT = "pnl_decomposition_2026-11-11"
ND = NormalDist()


def seg_of(d) -> str:
    return "llm_in_sample" if d < knowledge_cutoff().date() else "clean_holdout"


# ====================================================================== sleeves
def sleeves(panel: pl.DataFrame) -> pl.DataFrame:
    """Wide date x sleeve matrix ('strategy|U') of the mean pnl_d over symbols (T needs >= 3 symbols)."""
    g = (panel.group_by("universe", "strategy", "date").agg(pl.col("pnl_d").mean(), pl.len().alias("n_sym"))
         .filter((pl.col("universe") == "L") | (pl.col("n_sym") >= 3))
         .with_columns((pl.col("strategy") + "|" + pl.col("universe")).alias("sleeve")))
    return g.pivot(on="sleeve", index="date", values="pnl_d").sort("date")


def effective_bets(eig: np.ndarray) -> float:
    lam = np.clip(np.asarray(eig, float), 0, None)
    p = lam / lam.sum()
    p = p[p > 0]
    return float(np.exp(-np.sum(p * np.log(p))))


def meucci_enb(X: np.ndarray, w: np.ndarray) -> float:
    """Meucci (2009) effective number of bets of a book with weights w: entropy of the risk contributions of the
    uncorrelated principal portfolios of the covariance matrix (extension; the registered ENB is the eigenvalue form)."""
    lam, E = np.linalg.eigh(np.cov(X, rowvar=False))
    v = (E.T @ w) ** 2 * np.clip(lam, 0, None)
    p = v / v.sum()
    p = p[p > 0]
    return float(np.exp(-np.sum(p * np.log(p))))


def pd1_by_regime(panel: pl.DataFrame) -> dict:
    """Extension: ENB of the L sleeves restricted, per date, to the symbols in v0 cell c (>= 3 symbols that date)."""
    out = {}
    L = panel.filter((pl.col("universe") == "L") & pl.col("cell").is_not_null())
    for (seg, c), g in L.group_by("seg", "cell"):
        W = (g.group_by("strategy", "date").agg(pl.col("pnl_d").mean(), pl.len().alias("n")).filter(pl.col("n") >= 3)
             .pivot(on="strategy", index="date", values="pnl_d").drop_nulls())
        cols = [k for k in W.columns if k != "date"]
        X = W.select(cols).to_numpy().astype(float)
        keep = X.std(axis=0) > 0 if len(X) else np.zeros(len(cols), bool)
        X = X[:, keep]
        if X.shape[0] < 30 or X.shape[1] < 3:
            continue
        st = pca_stats(X)
        out[f"{seg}|{c}"] = {"n_days": X.shape[0], "n_sleeves": X.shape[1], "top3_share": st["top3_share"], "enb": st["enb"],
                             "enb_meucci_ew": meucci_enb(X, np.full(X.shape[1], 1 / X.shape[1]))}
    return dict(sorted(out.items()))


def pca_stats(X: np.ndarray) -> dict:
    c = np.corrcoef(X, rowvar=False)
    lam = np.sort(np.clip(np.linalg.eigvalsh(c), 0, None))[::-1]
    return {"top3_share": float(lam[:3].sum() / lam.sum()), "top1_share": float(lam[0] / lam.sum()), "enb": effective_bets(lam),
            "eig": lam.tolist()}


def _segment_matrix(W: pl.DataFrame, seg: str, universes: tuple[str, ...]) -> tuple[list[str], np.ndarray, list]:
    d = W.with_columns(pl.col("date").map_elements(seg_of, return_dtype=pl.Utf8).alias("seg")).filter(pl.col("seg") == seg)
    cols = [c for c in W.columns if c != "date" and c.split("|")[1] in universes]
    keep = []
    for c in cols:
        x = d[c].to_numpy().astype(float)
        ok = np.isfinite(x)
        if ok.sum() >= MIN_DAYS and np.nanstd(x) > 0:
            keep.append(c)
    sub = d.select("date", *keep).drop_nulls()
    X = sub.select(keep).to_numpy().astype(float)
    sd = X.std(axis=0)
    keep2 = [k for k, s in zip(keep, sd) if s > 0]
    X = sub.select(keep2).to_numpy().astype(float)
    return keep2, X, sub["date"].to_list()


def pd1(W: pl.DataFrame, seed: int = 21) -> dict:
    out = {}
    rng = np.random.default_rng(seed)
    for seg in SEGS:
        for label, univ in (("LT", ("L", "T")), ("L", ("L",))):
            if seg == "llm_in_sample" and label == "LT":
                continue
            names, X, dates = _segment_matrix(W, seg, univ)
            if X.shape[0] < MIN_DAYS or X.shape[1] < 3:
                out[f"{seg}|{label}"] = {"n_sleeves": X.shape[1], "n_days": X.shape[0], "skip": "too few"}
                continue
            st = pca_stats(X)
            bs = np.full((N_BOOT_PCA, 2), np.nan)
            for b in range(N_BOOT_PCA):
                Y = X[block_indices(len(X), BLOCK, rng)]
                if np.any(Y.std(axis=0) == 0):
                    continue
                s = pca_stats(Y)
                bs[b] = (s["top3_share"], s["enb"])
            ok = np.isfinite(bs).all(axis=1)
            p_share = float((np.sum(bs[ok, 0] <= 0.6) + 1) / (ok.sum() + 1))
            p_enb = float((np.sum(bs[ok, 1] >= 6) + 1) / (ok.sum() + 1))
            c = np.corrcoef(X, rowvar=False)
            w, v = np.linalg.eigh(c)
            order = np.argsort(w)[::-1]
            load_ = {f"PC{k + 1}": {n: round(float(v[i, order[k]] * np.sqrt(max(w[order[k]], 0))), 3) for i, n in enumerate(names)}
                     for k in range(3)}
            off = c[~np.eye(len(names), dtype=bool)]
            out[f"{seg}|{label}"] = {
                "n_sleeves": len(names), "n_days": X.shape[0], "first_date": str(dates[0]), "last_date": str(dates[-1]),
                **{k: st[k] for k in ("top3_share", "top1_share", "enb")}, "eig_top6": st["eig"][:6],
                "ci95_top3_share": [float(np.quantile(bs[ok, 0], q)) for q in (0.025, 0.975)],
                "ci95_enb": [float(np.quantile(bs[ok, 1], q)) for q in (0.025, 0.975)],
                "p_share_le_0.6": p_share, "p_enb_ge_6": p_enb, "p_test": max(p_share, p_enb),
                "mean_abs_corr": float(np.mean(np.abs(off))), "mean_corr": float(np.mean(off)),
                "enb_meucci_ew_book": meucci_enb(X, np.full(X.shape[1], 1 / X.shape[1])),  # extension: actual EW book weights
                "loadings": load_, "sleeves": names,
            }
    return out


# ====================================================================== factors
def factor_panel(ctx: pl.DataFrame, univ_l: list[str]) -> pl.DataFrame:
    """Daily point-in-time factor returns (see module docstring)."""
    c = (ctx.sort("symbol", "date")
         .with_columns(pl.col("rv").shift(1).rolling_mean(30, min_samples=20).over("symbol").alias("rv30"),
                       pl.col("ret").shift(1).rolling_sum(30, min_samples=30).over("symbol").alias("mom30"),
                       (pl.col("ret").abs() / pl.col("turnover_day")).shift(1).rolling_mean(30, min_samples=20).over("symbol").alias("amihud"),
                       pl.col("rv").shift(1).over("symbol").alias("rv_prev"))
         .with_columns((pl.col("ret") / pl.col("rv30").sqrt()).alias("z")))
    L = c.filter(pl.col("symbol").is_in(univ_l) & pl.col("z").is_finite())
    mkt = L.group_by("date").agg(pl.col("z").mean().alias("MKT"), pl.len().alias("n_mkt"))
    ts = L.filter(pl.col("mom30").is_not_null()).group_by("date").agg((pl.col("mom30").sign() * pl.col("z")).mean().alias("TSMOM"))
    vol = (L.filter((pl.col("rv") > 0) & (pl.col("rv_prev") > 0)).group_by("date")
           .agg((pl.col("rv").log().mean() - pl.col("rv_prev").log().mean()).alias("VOL")))

    def ls(col: str, sign: float, name: str) -> pl.DataFrame:
        q = L.filter(pl.col(col).is_finite()).with_columns(pl.col(col).rank("average").over("date").alias("_r"),
                                                             pl.len().over("date").alias("_n"))
        q = q.filter(pl.col("_n") >= 5).with_columns((pl.col("_r") - pl.col("_r").mean().over("date")).alias("_d"))
        q = q.with_columns((sign * 2 * pl.col("_d") / pl.col("_d").abs().sum().over("date")).alias("_w"))
        return q.group_by("date").agg((pl.col("_w") * pl.col("z")).sum().alias(name))

    carry = ls("funding_sod", -1.0, "CARRY")
    liq = ls("amihud", +1.0, "LIQ")
    tf = c.filter(pl.col("z").is_finite())

    def single(sym: str, name: str) -> pl.DataFrame:
        return tf.filter(pl.col("symbol") == sym).select("date", pl.col("z").alias(name))

    gold = single("XAUUSDT", "GOLD").join(single("PAXGUSDT", "GOLD_P"), on="date", how="full", coalesce=True)
    gold = gold.select("date", pl.coalesce("GOLD", "GOLD_P").alias("GOLD"), pl.col("GOLD").is_not_null().alias("gold_is_xau"))
    out = mkt
    for t in (ts, vol, carry, liq, single("QQQUSDT", "QQQ"), gold, single("CLUSDT", "OIL")):
        out = out.join(t, on="date", how="left")
    # a long-short leg with all-equal ranks (e.g. identical funding prints) is 0/0: no factor that day, not NaN
    out = out.with_columns(pl.col(pl.Float64).fill_nan(None)).sort("date")
    # convexity proxy (extension): |MKT_d| minus its trailing 90-day mean over d-90..d-1 (known at the start of d)
    return out.with_columns((pl.col("MKT").abs() - pl.col("MKT").abs().shift(1).rolling_mean(90, min_samples=60)).alias("CONVEX"))


def ols_nw(y: np.ndarray, X: np.ndarray) -> dict:
    n, k = X.shape
    XtX_inv = np.linalg.pinv(X.T @ X)
    b = XtX_inv @ X.T @ y
    e = y - X @ b
    lag = int(np.floor(4 * (n / 100) ** (2 / 9)))
    u = X * e[:, None]
    S = u.T @ u
    for j in range(1, lag + 1):
        G = u[j:].T @ u[:-j]
        S += (1 - j / (lag + 1)) * (G + G.T)
    V = XtX_inv @ S @ XtX_inv
    se = np.sqrt(np.clip(np.diag(V), 1e-300, None))
    r2 = 1 - (e @ e) / np.sum((y - y.mean()) ** 2) if np.sum((y - y.mean()) ** 2) > 0 else float("nan")
    return {"b": b, "t": b / se, "r2": float(r2), "resid": e, "n": n}


def pd2(W: pl.DataFrame, F: pl.DataFrame) -> dict:
    out: dict = {}
    D = W.join(F, on="date", how="inner").with_columns(pl.col("date").map_elements(seg_of, return_dtype=pl.Utf8).alias("seg"))
    sleeves_ = [c for c in W.columns if c != "date"]
    specs = {"llm_in_sample": CORE, "llm_in_sample_ext": (*CORE, "GOLD"), "clean_holdout": ALL_F,
             # labelled extension (orchestrator / literature note, not the registered factor list): Fung & Hsieh (2001)
             # convexity of trend following, proxied by |MKT| demeaned with its trailing 90-day mean (point in time)
             "llm_in_sample_convex": (*CORE, "CONVEX"), "clean_holdout_convex": (*ALL_F, "CONVEX")}
    for key, fs in specs.items():
        seg = "llm_in_sample" if key.startswith("llm") else "clean_holdout"
        d = D.filter(pl.col("seg") == seg)
        res = {}
        for s in sleeves_:
            sub = d.select(s, *fs).drop_nulls()
            if sub.height < MIN_DAYS:
                continue
            y = sub[s].to_numpy().astype(float)
            if y.std() == 0:
                continue
            Xf = sub.select(fs).to_numpy().astype(float)
            r = ols_nw(y, np.column_stack([np.ones(len(y)), Xf]))
            sdf = Xf.std(axis=0)
            res[s] = {"n": r["n"], "r2": r["r2"], "alpha_d": float(r["b"][0]), "alpha_ann": float(r["b"][0] * 365), "alpha_t": float(r["t"][0]),
                      "alpha_p_one_sided": float(1 - ND.cdf(r["t"][0])),
                      "beta": {f: float(b) for f, b in zip(fs, r["b"][1:])}, "beta_t": {f: float(t) for f, t in zip(fs, r["t"][1:])},
                      "std_loading": {f: float(b * sd / y.std()) for f, b, sd in zip(fs, r["b"][1:], sdf)},
                      "resid_ir_ann": float(r["resid"].mean() / r["resid"].std() * np.sqrt(365)) if r["resid"].std() > 0 else float("nan"),
                      "raw_sharpe_ann": float(y.mean() / y.std() * np.sqrt(365))}
        r2s = [v["r2"] for v in res.values()]
        out[key] = {"factors": list(fs), "sleeves": res, "median_r2": float(np.median(r2s)) if r2s else float("nan"),
                    "n_sleeves": len(res), "first_date": str(d.select(*fs).drop_nulls().height and d.drop_nulls(list(fs))["date"].min()),
                    "factor_corr": np.round(np.corrcoef(d.select(fs).drop_nulls().to_numpy(), rowvar=False), 3).tolist() if d.select(fs).drop_nulls().height > 10 else None}
    # factor summaries
    fsumm = {}
    for seg in SEGS:
        d = F.with_columns(pl.col("date").map_elements(seg_of, return_dtype=pl.Utf8).alias("seg")).filter(pl.col("seg") == seg)
        for f in ALL_F:
            x = d[f].drop_nulls().to_numpy()
            if len(x) > 10:
                m, t, n = newey_west_t(x)
                fsumm[f"{f}|{seg}"] = {"n": n, "mean": m, "t": t, "sd": float(x.std()), "sharpe_ann": float(x.mean() / x.std() * np.sqrt(365)) if x.std() > 0 else None}
    out["factor_summary"] = fsumm
    return out


def clusters(pd2_out: dict, max_k: int = 6) -> dict:
    from scipy.cluster.hierarchy import fcluster, linkage

    res = pd2_out["llm_in_sample"]["sleeves"]
    names = sorted(n for n in res if n.endswith("|L"))
    M = np.array([[res[n]["std_loading"][f] for f in CORE] for n in names])
    Z = linkage(M, method="average", metric="euclidean")
    h = Z[:, 2]
    best_k, best_gap = 2, -1.0
    for k in range(2, min(max_k, len(names) - 1) + 1):
        # cutting into k clusters sits between merge heights h[-k] and h[-k+1]
        gap = h[-k + 1] - h[-k]
        if gap > best_gap:
            best_k, best_gap = k, gap
    lab = fcluster(Z, t=best_k, criterion="maxclust")
    fam = {}
    for n, c in zip(names, lab):
        fam.setdefault(int(c), []).append(n.split("|")[0])
    # name each cluster by its dominant loading
    named = {}
    for c, members in fam.items():
        idx = [names.index(m + "|L") for m in members]
        centre = M[idx].mean(axis=0)
        j = int(np.argmax(np.abs(centre)))
        named[f"F{c}"] = {"members": members, "centre": {f: round(float(x), 3) for f, x in zip(CORE, centre)},
                          "dominant": f"{'+' if centre[j] >= 0 else '-'}{CORE[j]}"}
    return {"k": best_k, "gap": float(best_gap), "families": named, "linkage_heights": [float(x) for x in h],
            "map": {m: c for c, v in named.items() for m in v["members"]}}


# ====================================================================== IPCA
def ipca_fit(Ws: np.ndarray, xs: np.ndarray, K: int, G0: np.ndarray | None = None, tol: float = 1e-6, max_iter: int = 500) -> tuple[np.ndarray, np.ndarray, int]:
    """Kelly-Pruitt-Su (2019) ALS on managed portfolios. Ws: (T, L, L) = Z'Z/N per period; xs: (T, L) = Z'r/N.
    Returns Gamma (L, K), factors F (T, K), iterations."""
    T, L = xs.shape
    if G0 is None:
        U, _, _ = np.linalg.svd(xs.T, full_matrices=False)
        G = U[:, :K].copy()
    else:
        G = G0.copy()
    it = 0
    F = np.zeros((T, K))

    W2 = Ws.reshape(T, L * L).T  # (L*L, T) for the BLAS form of the Gamma normal equations

    def f_step(G: np.ndarray) -> np.ndarray:
        A = np.matmul(G.T[None], np.matmul(Ws, G))
        b = xs @ G
        try:
            return np.linalg.solve(A, b[..., None])[..., 0]
        except np.linalg.LinAlgError:
            return np.stack([np.linalg.lstsq(A[t], b[t], rcond=None)[0] for t in range(T)])

    for it in range(1, max_iter + 1):
        F = f_step(G)
        # vec(G) row-major: index l*K + k
        FF = (F[:, :, None] * F[:, None, :]).reshape(T, K * K)
        A = (W2 @ FF).reshape(L, L, K, K).transpose(0, 2, 1, 3).reshape(L * K, L * K)  # sum_t W_t (x) f_t f_t
        bvec = np.einsum("ti,tk->ik", xs, F).reshape(L * K)
        Gn = np.linalg.lstsq(A, bvec, rcond=None)[0].reshape(L, K)
        # normalisation: G'G = I, F'F diagonal descending, mean f > 0
        Q, R = np.linalg.qr(Gn)
        Fn = F @ R.T
        w, V = np.linalg.eigh(Fn.T @ Fn)
        V = V[:, np.argsort(w)[::-1]]
        Gn, Fn = Q @ V, Fn @ V
        sgn = np.where(Fn.mean(axis=0) >= 0, 1.0, -1.0)
        Gn, Fn = Gn * sgn, Fn * sgn
        d = float(np.max(np.abs(Gn - G)))
        G, F = Gn, Fn
        if d < tol:
            break
    F = f_step(G)  # factors consistent with the final Gamma
    return G, F, it


def managed(Z: list[np.ndarray], R: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    Ws = np.stack([z.T @ z / len(z) for z in Z])
    xs = np.stack([z.T @ r / len(z) for z, r in zip(Z, R)])
    return Ws, xs


def ipca_characteristics(panel: pl.DataFrame, ctx: pl.DataFrame) -> pl.DataFrame:
    """One row per (strategy, symbol, Monday) in universe L: z characteristics known at Monday 00:00 and the target."""
    P = panel.filter(pl.col("universe") == "L").select("symbol", "date", "strategy", "family", "pnl_d", "exposure")
    c = (ctx.filter(pl.col("universe") == "L").sort("symbol", "date")
         .with_columns(
             (pl.col("dir_state_sod") == "up").cast(pl.Float64).alias("_up"), (pl.col("dir_state_sod") == "down").cast(pl.Float64).alias("_dn"),
             (pl.col("vol_state_sod") == "low").cast(pl.Float64).alias("_lo"), (pl.col("vol_state_sod") == "high").cast(pl.Float64).alias("_hi"))
         .with_columns(
             *[pl.col(k).rolling_mean(7, min_samples=7).over("symbol").alias(k[1:] + "_sh") for k in ("_up", "_dn", "_lo", "_hi")],
             (pl.col("sigma_risk_sod").rolling_map(lambda s: float((s[:-1] < s[-1]).mean() + 0.5 * (s[:-1] == s[-1]).mean()) if len(s) > 120 else float("nan"),
                                                   window_size=366, min_samples=121).over("symbol") - 0.5).alias("vol_pct"),
             pl.col("funding_sod").shift(1).rolling_mean(30, min_samples=20).over("symbol").alias("_fm"),
             pl.col("funding_sod").shift(1).rolling_std(30, min_samples=20).over("symbol").alias("_fs"),
             (pl.col("dir_score_sod").abs().clip(0, 3) / 3).alias("trend"),
             pl.col("close").shift(1).rolling_max(30, min_samples=20).over("symbol").alias("_hi30"),
             pl.col("close").shift(1).over("symbol").alias("_c1"),
             pl.col("rv").shift(1).rolling_mean(30, min_samples=20).over("symbol").sqrt().alias("_s30"))
         .with_columns(((pl.col("_hi30") / pl.col("_c1")).log() > 1.5 * pl.col("_s30") * np.sqrt(30)).cast(pl.Float64).alias("dd"),
                       pl.when(pl.col("_fs") > 1e-12).then(((pl.col("funding_sod") - pl.col("_fm")) / pl.col("_fs")).clip(-3, 3) / 3)
                       .otherwise(0.0).alias("fz"))
         .select("symbol", "date", "up_sh", "dn_sh", "lo_sh", "hi_sh", "vol_pct", "fz", "trend", "dd"))
    # strategy-specific trailing stats (days d-90..d-1)
    P = P.sort("strategy", "symbol", "date").with_columns(
        (pl.col("date").dt.weekday() >= 6).cast(pl.Float64).alias("_wk"))
    P = P.with_columns(
        pl.col("pnl_d").shift(1).rolling_mean(90, min_samples=60).over("strategy", "symbol").alias("_m90"),
        pl.col("pnl_d").shift(1).rolling_std(90, min_samples=60).over("strategy", "symbol").alias("_s90"),
        (pl.col("exposure") * pl.col("_wk")).shift(1).rolling_sum(90, min_samples=60).over("strategy", "symbol").alias("_wnum"),
        pl.col("exposure").shift(1).rolling_sum(90, min_samples=60).over("strategy", "symbol").alias("_wden"),
        pl.col("date").dt.truncate("1w").alias("week"),
    ).with_columns(  # rolling sums leave ~1e-17 residue where the true sum is 0: treat < 1e-9 as no exposure
        pl.when(pl.col("_wden") > 1e-9).then((pl.col("_wnum") / pl.col("_wden")).clip(0, 1)).otherwise(None).alias("_wks"))
    P = P.with_columns((pl.col("_wks").fill_nan(None).fill_null(2 / 7) - 2 / 7).alias("wk_sh"),
                       pl.when(pl.col("_s90") > 1e-9).then((pl.col("_m90") / pl.col("_s90") * np.sqrt(365)).clip(-5, 5) / 5)
                       .when(pl.col("_s90").is_not_null()).then(0.0).otherwise(None).alias("sh90"))
    tgt = P.group_by("strategy", "symbol", "week").agg(pl.col("pnl_d").sum().alias("r"), pl.len().alias("_n")).filter(pl.col("_n") == 7)
    mon = P.filter(pl.col("date").dt.weekday() == 1).select("strategy", "family", "symbol", "date", "week", "sh90", "wk_sh")
    X = (mon.join(c, on=["symbol", "date"], how="left").join(tgt.select("strategy", "symbol", "week", "r"), on=["strategy", "symbol", "week"], how="inner")
         .filter(pl.col("sh90").is_not_null())
         # a unit flat for the whole trailing window has a 0/0 Sharpe: no information, set to 0
         .with_columns(pl.col("sh90").fill_nan(0.0)))
    ctx_cols = ("up_sh", "dn_sh", "lo_sh", "hi_sh", "vol_pct", "fz", "trend", "dd")
    X = X.with_columns([pl.col(k).fill_nan(None).fill_null(0.0) for k in ctx_cols])
    return X.sort("week", "strategy", "symbol")


CTX = ("up_sh", "dn_sh", "lo_sh", "hi_sh", "vol_pct", "fz", "trend", "dd")


def zmatrix(X: pl.DataFrame) -> np.ndarray:
    fam = X["family"].to_numpy()
    C = np.column_stack([np.ones(X.height)] + [X[k].to_numpy().astype(float) for k in CTX])
    blocks = [C * (fam == f)[:, None] for f in FAMILIES]
    Z = np.column_stack(blocks + [X["sh90"].to_numpy().astype(float), X["wk_sh"].to_numpy().astype(float)])
    if not np.isfinite(Z).all():
        raise ValueError(f"non-finite IPCA instruments: {np.argwhere(~np.isfinite(Z))[:5].tolist()}")
    return Z


def pd3(panel: pl.DataFrame, ctx: pl.DataFrame, Ks=(1, 2, 3)) -> dict:
    X = ipca_characteristics(panel, ctx)
    Zall = zmatrix(X)
    weeks = sorted(X["week"].unique().to_list())
    wk = X["week"].to_numpy()
    rr = X["r"].to_numpy().astype(float)
    strat = X["strategy"].to_numpy()
    by_w = {w: np.where(wk == np.datetime64(w))[0] for w in weeks}
    Ws, xs = managed([Zall[by_w[w]] for w in weeks], [rr[by_w[w]] for w in weeks])
    # benchmark: trailing 90d Sharpe of the L sleeve (mean over symbols), as of the decision date (registered); 30d and
    # 365d windows are descriptive extras (factor momentum, Ehsani & Linnainmaa 2022), not part of the verdict
    sl = panel.filter(pl.col("universe") == "L").group_by("strategy", "date").agg(pl.col("pnl_d").mean()).sort("strategy", "date")
    benches = {}
    for win, ms in ((90, 60), (30, 20), (365, 240)):
        b_ = (sl.with_columns(pl.col("pnl_d").shift(1).rolling_mean(win, min_samples=ms).over("strategy").alias("_m"),
                              pl.col("pnl_d").shift(1).rolling_std(win, min_samples=ms).over("strategy").alias("_s"))
              .with_columns(pl.when(pl.col("_s") > 1e-9).then(pl.col("_m") / pl.col("_s")).when(pl.col("_s").is_not_null()).then(0.0)
                            .otherwise(None).alias("shs")))
        benches[win] = {(r[0], r[1]): r[2] for r in b_.select("strategy", "date", "shs").iter_rows() if r[2] is not None}
    bench = benches[90]
    out = {"n_obs": X.height, "n_weeks": len(weeks), "n_units": X.select("strategy", "symbol").n_unique(), "L_chars": Zall.shape[1],
           "first_week": str(weeks[0]), "last_week": str(weeks[-1])}
    perK = {}
    for K in Ks:
        G = None
        rows = []
        for ti in range(IPCA_MIN_WEEKS, len(weeks)):
            # train on (z_s, r_{s+1}) pairs fully realised before week ti starts: weeks[0..ti-1]
            G, Fh, it = ipca_fit(Ws[:ti], xs[:ti], K, G0=G, max_iter=500 if G is None else 200)
            lam = Fh.mean(axis=0)
            w = weeks[ti]
            idx = by_w[w]
            pred = Zall[idx] @ G @ lam
            s_ = strat[idx]
            us = sorted(set(s_.tolist()))
            score = np.array([pred[s_ == u].mean() for u in us])
            real = np.array([rr[idx][s_ == u].mean() for u in us])
            dmon = X["date"].to_numpy()[idx][0]
            bdate = dmon.astype("datetime64[D]").astype(object)
            bs = np.array([bench.get((u, bdate), np.nan) for u in us])
            ic_ip = spearman(score, real)
            ic_sh = spearman(bs, real)
            ic_unit = spearman(pred, rr[idx])
            extra = {f"ic_sharpe{win}": spearman(np.array([benches[win].get((u, bdate), np.nan) for u in us]), real) for win in (30, 365)}
            rows.append({"week": w, "seg": seg_of(w), "ic_ipca": ic_ip, "ic_sharpe": ic_sh, "ic_unit": ic_unit, "n_strat": len(us), "iters": it, **extra})
        df = pl.DataFrame(rows)
        res = {}
        for seg in SEGS:
            d = df.filter(pl.col("seg") == seg).drop_nulls(["ic_ipca", "ic_sharpe"]).filter(pl.col("ic_ipca").is_not_nan() & pl.col("ic_sharpe").is_not_nan())
            a, b = d["ic_ipca"].to_numpy(), d["ic_sharpe"].to_numpy()
            m_ip, t_ip, n = newey_west_t(a)
            m_sh, t_sh, _ = newey_west_t(b)
            m_d, t_d, _ = newey_west_t(a - b)
            res[seg] = {"n_weeks": n, "ic_ipca": m_ip, "t_ipca": t_ip, "ic_sharpe": m_sh, "t_sharpe": t_sh, "ic_diff": m_d, "t_diff": t_d,
                        "p_diff_one_sided": float(1 - ND.cdf(t_d)) if np.isfinite(t_d) else float("nan"),
                        "ic_unit": float(d["ic_unit"].drop_nans().mean()) if d.height else float("nan")}
            for win in (30, 365):  # descriptive (not registered)
                q = df.filter(pl.col("seg") == seg).select("ic_ipca", f"ic_sharpe{win}").fill_nan(None).drop_nulls()
                if q.height >= 10:
                    mm, tt, nn = newey_west_t(q[f"ic_sharpe{win}"].to_numpy())
                    md, td, _ = newey_west_t(q["ic_ipca"].to_numpy() - q[f"ic_sharpe{win}"].to_numpy())
                    res[seg][f"sharpe{win}"] = {"n_weeks": nn, "ic": mm, "t": tt, "ipca_minus": md, "t_diff": td}
        perK[K] = {"segments": res, "last_gamma_norm_by_block": _gamma_blocks(G)}
    best = max(Ks, key=lambda k: np.nan_to_num(perK[k]["segments"]["llm_in_sample"]["ic_ipca"], nan=-9))
    out["perK"] = perK
    out["K_chosen"] = best
    out["chosen"] = perK[best]["segments"]
    # interpretable full-in-sample fit with the chosen K: expected next-week P&L by family x context (scenario table)
    n_is = sum(1 for w in weeks if seg_of(w) == "llm_in_sample")
    G, Fh, _ = ipca_fit(Ws[:n_is], xs[:n_is], best)
    lam = Fh.mean(axis=0)
    out["scenarios_in_sample_fit"] = scenario_table(G, lam)
    return out


def _gamma_blocks(G: np.ndarray) -> dict:
    out = {}
    for i, f in enumerate(FAMILIES):
        out[f] = float(np.linalg.norm(G[i * 9:(i + 1) * 9]))
    out["sh90"] = float(np.linalg.norm(G[-2]))
    out["wk_sh"] = float(np.linalg.norm(G[-1]))
    return out


def scenario_table(G: np.ndarray, lam: np.ndarray) -> dict:
    """Model-implied expected weekly pnl_d per family under stylised contexts (all other characteristics at 0)."""
    sc = {"neutral": {}, "trend_up_lowvol": {"up_sh": 1.0, "lo_sh": 1.0, "trend": 0.5}, "trend_up_highvol": {"up_sh": 1.0, "hi_sh": 1.0, "trend": 0.5},
          "range_lowvol": {"lo_sh": 1.0}, "range_highvol": {"hi_sh": 1.0}, "trend_down_highvol": {"dn_sh": 1.0, "hi_sh": 1.0, "trend": 0.5, "dd": 1.0},
          "trend_down_lowvol": {"dn_sh": 1.0, "lo_sh": 1.0, "trend": 0.5}, "funding_hot": {"fz": 0.67}, "vol_pct_high": {"vol_pct": 0.4}}
    out = {}
    for name, vals in sc.items():
        out[name] = {}
        for i, f in enumerate(FAMILIES):
            z = np.zeros(G.shape[0])
            z[i * 9] = 1.0
            for j, k in enumerate(CTX):
                z[i * 9 + 1 + j] = vals.get(k, 0.0)
            out[name][f] = float(z @ G @ lam)
    return out


# ====================================================================== H-PD4
def decompose(t: pl.DataFrame) -> dict:
    g, c, m = (t[k].to_numpy().astype(float) for k in ("gross_d", "cost_d", "mfe_d"))
    D = m.sum()
    kept = np.maximum(g, 0).sum()
    E = D - kept
    dirl = np.maximum(-g, 0).sum()
    C = c.sum()
    win = g > 0
    Ew = (m[win] - g[win]).sum()
    return {"n_trades": len(g), "D_mfe": D, "kept": kept, "exit_leak": E, "dir_loss": dirl, "costs": C, "net": float((g - c).sum()),
            "share": float((C + E) / D) if D > 0 else float("nan"),
            "share_alt": float((C + Ew) / kept) if kept > 0 else float("nan"),
            "capture": float(kept / D) if D > 0 else float("nan"), "hit_gross": float(win.mean()) if len(g) else float("nan"),
            "hit_net": float(((g - c) > 0).mean()) if len(g) else float("nan"),
            "cost_over_gross_pos": float(C / kept) if kept > 0 else float("nan")}


def pd4(trades: pl.DataFrame, seed: int = 41) -> dict:
    s = pl.col("bpd").cast(pl.Float64).sqrt()
    t = trades.with_columns((pl.col("gross") / s).alias("gross_d"), (pl.col("cost") / s).alias("cost_d"), (pl.col("mfe") / s).alias("mfe_d"),
                            pl.col("entry_date").dt.truncate("1w").alias("wk"))
    rng = np.random.default_rng(seed)
    out = {}
    for seg in SEGS:
        d = t.filter(pl.col("seg") == seg)
        res = {"all": decompose(d)}
        for fq in ("1h", "4h"):
            res[fq] = decompose(d.filter(pl.col("freq") == fq))
        res["by_strategy"] = {k[0]: decompose(g) for k, g in d.group_by("strategy")}
        res["by_universe"] = {k[0]: decompose(g) for k, g in d.group_by("universe")}
        # week-cluster bootstrap of the pooled share
        agg = (d.with_columns(pl.max_horizontal(pl.col("gross_d"), pl.lit(0.0)).alias("_k"))
               .group_by("wk").agg(pl.col("mfe_d").sum().alias("D"), pl.col("_k").sum().alias("K"), pl.col("cost_d").sum().alias("C")))
        Dw, Kw, Cw = (agg[k].to_numpy() for k in ("D", "K", "C"))
        nw = len(Dw)
        bs = np.empty(N_BOOT)
        for b in range(N_BOOT):
            cnt = np.bincount(rng.integers(0, nw, nw), minlength=nw)
            Dd, Kk, Cc = Dw @ cnt, Kw @ cnt, Cw @ cnt
            bs[b] = (Cc + Dd - Kk) / Dd
        res["ci95_share"] = [float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975))]
        res["p_share_le_0.5"] = float((np.sum(bs <= 0.5) + 1) / (N_BOOT + 1))
        res["n_weeks"] = nw
        out[seg] = res
    return out


# ====================================================================== orchestration
def run() -> dict:
    panel = pl.read_parquet(panel_path())
    ctx = pl.read_parquet(context_path())
    trades = pl.read_parquet(trades_path())
    meta = json.loads((priors_dir() / "strategy_panel_meta.json").read_text(encoding="utf-8"))
    W = sleeves(panel)
    out: dict = {"meta": {k: meta[k] for k in ("as_of", "symbols", "skipped_strategies", "skipped_strategies_T", "f4_aggregated_from_1h")}}
    print("PD1", flush=True)
    out["H-PD1"] = pd1(W)
    out["H-PD1_by_regime"] = pd1_by_regime(panel)
    print("PD2", flush=True)
    F = factor_panel(ctx, meta["symbols"]["L"])
    out["H-PD2"] = pd2(W, F)
    out["clusters"] = clusters(out["H-PD2"])
    cmap = out["clusters"]["map"]
    cp = panel.filter(pl.col("universe") == "L").with_columns(pl.col("strategy").replace_strict(cmap, default=None).alias("family")).filter(pl.col("family").is_not_null())
    out["context_cluster"] = context_table(cp, "family")
    print("PD4", flush=True)
    out["H-PD4"] = pd4(trades)
    print("PD3", flush=True)
    out["H-PD3"] = pd3(panel, ctx)
    out["holm"] = holm_block(out)
    out["verdicts"] = verdicts(out)
    return out


def holm_block(out: dict) -> dict:
    res = {}
    for seg in SEGS:
        p = {}
        k1 = f"{seg}|L" if seg == "llm_in_sample" else f"{seg}|LT"
        p["H-PD1"] = out["H-PD1"].get(k1, {}).get("p_test", float("nan"))
        for s, v in out["H-PD2"][seg]["sleeves"].items():
            if s.endswith("|L"):
                p[f"H-PD2:{s}"] = v["alpha_p_one_sided"]
        p["H-PD3"] = out["H-PD3"]["chosen"][seg]["p_diff_one_sided"]
        p["H-PD4"] = out["H-PD4"][seg]["p_share_le_0.5"]
        res[seg] = {"raw": p, "adjusted": holm(p), "m": int(sum(np.isfinite(list(p.values()))))}
    return res


def verdicts(out: dict) -> dict:
    v: dict = {}
    # PD1
    v["H-PD1"] = {}
    for seg in SEGS:
        r = out["H-PD1"].get(f"{seg}|L" if seg == "llm_in_sample" else f"{seg}|LT", {})
        if "top3_share" not in r:
            v["H-PD1"][seg] = "inconclusive"
        else:
            v["H-PD1"][seg] = "supports" if (r["top3_share"] > 0.6 and r["enb"] < 6) else "contradicts"
    # PD2
    sig = {}
    for seg in SEGS:
        adj = out["holm"][seg]["adjusted"]
        sig[seg] = {s for s, x in out["H-PD2"][seg]["sleeves"].items()
                    if s.endswith("|L") and x["alpha_t"] > 2 and adj.get(f"H-PD2:{s}", 1) < 0.05}
    both = sig["llm_in_sample"] & sig["clean_holdout"]
    v["H-PD2"] = {}
    v["H-PD2_alpha_both"] = sorted(both)
    for seg in SEGS:
        mr = out["H-PD2"][seg]["median_r2"]
        v["H-PD2"][seg] = "supports" if (mr > 0.4 and not both) else "contradicts"
    # PD3
    v["H-PD3"] = {}
    for seg in SEGS:
        r = out["H-PD3"]["chosen"][seg]
        adj = out["holm"][seg]["adjusted"]["H-PD3"]
        if seg == "llm_in_sample":
            ok = r["ic_diff"] > 0 and r["t_diff"] > 2 and adj < 0.05
            v["H-PD3"][seg] = "supports" if ok else "contradicts"
        else:
            v["H-PD3"][seg] = "inconclusive" if r["n_weeks"] < 8 else ("supports" if r["ic_diff"] > 0 else "contradicts")
    # PD4
    v["H-PD4"] = {}
    for seg in SEGS:
        r = out["H-PD4"][seg]
        adj = out["holm"][seg]["adjusted"]["H-PD4"]
        sh = r["all"]["share"]
        if sh > 0.5 and r["ci95_share"][0] > 0.5 and adj < 0.05:
            v["H-PD4"][seg] = "supports"
        elif sh <= 0.5:
            v["H-PD4"][seg] = "contradicts"
        else:
            v["H-PD4"][seg] = "inconclusive"
    return v


DATA = ("data/priors/strategy_panel_{{daily,trades,context}}.parquet (research.strategy_panel, as of {as_of}); universes L (23 crypto) "
        "and T (TradFi perps >= 120 d); pnl_d daily-sigma units; 7.5 bps per side")


def _already(hid: str, seg: str) -> bool:
    from ai_trader.research.hypotheses import results

    return any(r["segment"] == seg for r in results(hid))


def record_all(out: dict) -> list[str]:
    from ai_trader.research.hypotheses import record

    data = DATA.format(as_of=out["meta"]["as_of"])
    done = []
    V = out["verdicts"]
    for seg in SEGS:
        hk = out["holm"][seg]["adjusted"]
        if not _already("H-PD1", seg):
            k = f"{seg}|L" if seg == "llm_in_sample" else f"{seg}|LT"
            r = {x: y for x, y in out["H-PD1"][k].items() if x not in ("loadings", "sleeves")}
            r["p_holm"] = hk.get("H-PD1")
            if seg == "clean_holdout":
                r["L_only"] = {x: out["H-PD1"]["clean_holdout|L"].get(x) for x in ("n_sleeves", "top3_share", "enb", "ci95_enb")}
            record("H-PD1", seg, V["H-PD1"][seg], r, note=f"correlation PCA of daily sleeve P&L ({'L' if seg == 'llm_in_sample' else 'L+T'} sleeves); "
                   "ENB = exp(entropy of normalised eigenvalues); 10-day block bootstrap (1000)", data=data)
            done.append(f"H-PD1/{seg}")
        if not _already("H-PD2", seg):
            p2 = out["H-PD2"][seg]
            m = {"median_r2": p2["median_r2"], "n_sleeves": p2["n_sleeves"], "factors": p2["factors"],
                 "r2": {s: round(x["r2"], 3) for s, x in p2["sleeves"].items()},
                 "alpha_t": {s: round(x["alpha_t"], 2) for s, x in p2["sleeves"].items()},
                 "alpha_sig_both_segments_holm": V["H-PD2_alpha_both"]}
            if seg == "llm_in_sample":
                m["median_r2_with_gold_2025plus"] = out["H-PD2"]["llm_in_sample_ext"]["median_r2"]
            record("H-PD2", seg, V["H-PD2"][seg], m, note="time-series OLS of daily sleeve P&L on point-in-time factors, NW HAC SEs; "
                   "in-sample core factors MKT TSMOM VOL CARRY LIQ (TradFi factors exist only from 2025-2026), holdout adds QQQ GOLD OIL; "
                   "alpha counted only if t > 2 and Holm-significant in the PD block", data=data)
            done.append(f"H-PD2/{seg}")
        if not _already("H-PD3", seg):
            r = dict(out["H-PD3"]["chosen"][seg])
            r["K_chosen_in_sample"] = out["H-PD3"]["K_chosen"]
            r["p_holm"] = hk.get("H-PD3")
            r["allK"] = {str(k): {x: v["segments"][seg][x] for x in ("ic_ipca", "ic_sharpe", "ic_diff", "t_diff")} for k, v in out["H-PD3"]["perK"].items()}
            record("H-PD3", seg, V["H-PD3"][seg], r, note="IPCA (KPS 2019 ALS, numpy) on strategy x symbol weeks, L = 56 instruments (family x context + "
                   "own trailing Sharpe + weekend share), weekly expanding refit; Spearman IC across strategies vs trailing-90d sleeve Sharpe; "
                   "NW t over weeks; K chosen on llm_in_sample IC only", data=data)
            done.append(f"H-PD3/{seg}")
        if not _already("H-PD4", seg):
            r4 = out["H-PD4"][seg]
            m = {**r4["all"], "ci95_share": r4["ci95_share"], "p_share_le_0.5": r4["p_share_le_0.5"], "p_holm": hk.get("H-PD4"),
                 "share_1h": r4["1h"]["share"], "share_4h": r4["4h"]["share"], "n_weeks": r4["n_weeks"]}
            record("H-PD4", seg, V["H-PD4"][seg], m, note="trade-level: share = (costs + exit leakage) / sum MFE; exit leakage = sum MFE - "
                   "sum max(gross, 0); week-cluster bootstrap (2000)", data=data)
            done.append(f"H-PD4/{seg}")
    return done


# ====================================================================== report
def _f(x, nd=3) -> str:
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return "n/a"
    return f"{x:.{nd}f}" if isinstance(x, float) else str(x)


def write_report(out: dict, rule_yaml: str) -> tuple[Path, Path]:
    rep = load().root / "research" / "reports"
    pj = rep / f"{REPORT}.json"
    pj.write_text(json.dumps(out, indent=1, default=str), encoding="utf-8")
    V = out["verdicts"]
    L = [f"# PnL decomposition of the strategy book (H-PD1..H-PD4) — {REPORT[-10:]}", "",
         "Agent B1, Program 2 ([plan](../plans/2026-11-11_strategy_program.md)); methods fixed in the docstring of "
         "`ai_trader.research.pnl_decomposition_study` before running. Input: the strategy panel "
         "(see [strategy_map_2026-11-11.md](strategy_map_2026-11-11.md)). Units: daily-sigma units (pnl_d). "
         "Segments split at 2026-07-01.", "",
         "## Headline", "",
         f"- H-PD1 (few bets): llm_in_sample **{V['H-PD1']['llm_in_sample']}**, clean_holdout **{V['H-PD1']['clean_holdout']}**: the "
         "correlation matrix has ~8 eigen-bets (L), but the equal-weight book is ~1-2.5 Meucci bets (extension): one trend exposure dominates.",
         f"- H-PD2 (factors explain P&L, no alpha): **{V['H-PD2']['llm_in_sample']}** / **{V['H-PD2']['clean_holdout']}**: no sleeve has "
         "alpha in both segments, but the median R^2 is only ~0.25 (< 0.4). Trend sleeves are TSMOM + long convexity; 1h mean reversion is "
         "short convexity / short vol-change; OI divergence is market beta.",
         f"- H-PD3 (IPCA beats trailing Sharpe): **{V['H-PD3']['llm_in_sample']}** / **{V['H-PD3']['clean_holdout']}**: largely because "
         "trailing-Sharpe ranks carry NEGATIVE information week to week; the IPCA IC itself is small (0.10 in-sample t 3.2, 0.13 holdout t 1.2).",
         f"- H-PD4 (costs + exit leakage > 50%): **{V['H-PD4']['llm_in_sample']}** / **{V['H-PD4']['clean_holdout']}**: ~70-75% of the "
         "favourable excursion is lost, almost all of it to exits (costs ~5-9% of MFE; MFE is a bar high/low upper bound).", ""]
    # PD1
    L += ["## H-PD1: how many independent bets?", "", "| segment | sleeves | days | top-3 share [95% CI] | ENB [95% CI] | mean abs corr | verdict |", "|---|---|---|---|---|---|---|"]
    for k, r in out["H-PD1"].items():
        if "top3_share" not in r:
            continue
        seg = k.split("|")[0]
        vv = V["H-PD1"][seg] if (k.endswith("|L") and seg == "llm_in_sample") or k.endswith("|LT") else "(robustness)"
        L.append(f"| {k} | {r['n_sleeves']} | {r['n_days']} | {_f(r['top3_share'])} [{_f(r['ci95_top3_share'][0])}, {_f(r['ci95_top3_share'][1])}] | "
                 f"{_f(r['enb'], 2)} [{_f(r['ci95_enb'][0], 2)}, {_f(r['ci95_enb'][1], 2)}] | {_f(r['mean_abs_corr'])} | {vv} |")
    L += ["", "Extension (not registered): Meucci ENB of the actual equal-weight book: " + "; ".join(
        f"{k} {_f(r_.get('enb_meucci_ew_book'), 2)}" for k, r_ in out["H-PD1"].items() if "enb_meucci_ew_book" in r_) + ".", "",
        "Extension: registered ENB per v0 cell (universe L; sleeve P&L per date over the symbols in that cell):", "",
        "| segment | cell | days | sleeves | top-3 share | ENB | Meucci ENB (EW book) |", "|---|---|---|---|---|---|---|"]
    for k, x in out.get("H-PD1_by_regime", {}).items():
        seg, c = k.split("|")
        L.append(f"| {seg} | {c} | {x['n_days']} | {x['n_sleeves']} | {_f(x['top3_share'])} | {_f(x['enb'], 2)} | {_f(x['enb_meucci_ew'], 2)} |")
    r = out["H-PD1"].get("llm_in_sample|L", {})
    if "loadings" in r:
        L += ["", "In-sample loadings (correlation-scaled) on the first three components, universe L:", "", "| sleeve | PC1 | PC2 | PC3 |", "|---|---|---|---|"]
        for s in r["sleeves"]:
            L.append(f"| {s} | {r['loadings']['PC1'][s]} | {r['loadings']['PC2'][s]} | {r['loadings']['PC3'][s]} |")
    # PD2
    L += ["", "## H-PD2: what each strategy really is (factor attribution)", ""]
    for key in ("llm_in_sample", "clean_holdout", "llm_in_sample_ext", "llm_in_sample_convex", "clean_holdout_convex"):
        p2 = out["H-PD2"][key]
        tag = " (EXTENSION, not registered)" if key.endswith(("_ext", "_convex")) else ""
        L += [f"### {key}{tag} (factors {', '.join(p2['factors'])}; median R^2 {_f(p2['median_r2'])}, {p2['n_sleeves']} sleeves)", "",
              "| sleeve | R^2 | alpha/yr | alpha t | " + " | ".join(f"b {f} (t)" for f in p2["factors"]) + " | raw Sharpe |",
              "|---|---|---|---|" + "---|" * len(p2["factors"]) + "---|"]
        for s, x in sorted(p2["sleeves"].items()):
            L.append(f"| {s} | {_f(x['r2'], 2)} | {_f(x['alpha_ann'], 2)} | {_f(x['alpha_t'], 1)} | "
                     + " | ".join(f"{_f(x['beta'][f], 2)} ({_f(x['beta_t'][f], 1)})" for f in p2["factors"]) + f" | {_f(x['raw_sharpe_ann'], 2)} |")
        L.append("")
    L += [f"- Sleeves with a Holm-significant alpha t > 2 in both segments: {', '.join(V['H-PD2_alpha_both']) or 'none'}.",
          f"- **Verdicts: llm_in_sample {V['H-PD2']['llm_in_sample']}, clean_holdout {V['H-PD2']['clean_holdout']}.**", "",
          "Factor returns (daily, NW t of the mean):", "", "| factor | segment | n | mean | t | ann. Sharpe |", "|---|---|---|---|---|---|"]
    for k, x in out["H-PD2"]["factor_summary"].items():
        f, seg = k.split("|")
        L.append(f"| {f} | {seg} | {x['n']} | {_f(x['mean'], 4)} | {_f(x['t'], 1)} | {_f(x['sharpe_ann'], 2)} |")
    cl = out["clusters"]
    L += ["", f"### Families from hierarchical clustering of in-sample standardised loadings (k = {cl['k']})", "",
          "| family | dominant loading | members | centre (std loadings " + ", ".join(CORE) + ") |", "|---|---|---|---|"]
    for c, x in cl["families"].items():
        L.append(f"| {c} | {x['dominant']} | {', '.join(x['members'])} | {', '.join(f'{v:+.2f}' for v in x['centre'].values())} |")
    L += ["", "Context table by cluster family (universe L; shrunk mean pnl_d x100 per active symbol-day; grade as in the strategy map):", "",
          "| family | cell | IS x100 | IS t | HO x100 | HO t | grade |", "|---|---|---|---|---|---|---|"]
    for x in out["context_cluster"]:
        L.append(f"| {x['name']} | {x['cell']} | {_f(100 * x['is_shrunk'], 2)} | {_f(x['is_t'], 1)} | {_f(100 * x['ho_shrunk'], 2)} | {_f(x['ho_t'], 1)} | {x['grade']} |")
    # PD3
    p3 = out["H-PD3"]
    L += ["", "## H-PD3: does a conditional factor model (IPCA) rank strategies better than trailing Sharpe?", "",
          f"- {p3['n_obs']} strategy x symbol weeks, {p3['n_units']} units, {p3['n_weeks']} weeks ({p3['first_week']} to {p3['last_week']}), "
          f"{p3['L_chars']} instruments; first forecast after {IPCA_MIN_WEEKS} training weeks; K chosen in-sample = **{p3['K_chosen']}**.", "",
          "| K | segment | weeks | IC IPCA (t) | IC trailing Sharpe (t) | diff (NW t) | unit-level IC |", "|---|---|---|---|---|---|---|"]
    for K, v in p3["perK"].items():
        for seg in SEGS:
            x = v["segments"][seg]
            L.append(f"| {K} | {seg} | {x['n_weeks']} | {_f(x['ic_ipca'])} ({_f(x['t_ipca'], 1)}) | {_f(x['ic_sharpe'])} ({_f(x['t_sharpe'], 1)}) | "
                     f"{_f(x['ic_diff'])} ({_f(x['t_diff'], 1)}) | {_f(x['ic_unit'])} |")
    L += ["", "Extension (descriptive): chosen-K IPCA vs trailing-30d / trailing-365d sleeve Sharpe:", "",
          "| segment | window | weeks | IC trailing Sharpe (t) | IPCA minus (NW t) |", "|---|---|---|---|---|"]
    for seg in SEGS:
        for win in (30, 365):
            x = p3["chosen"][seg].get(f"sharpe{win}")
            if x:
                L.append(f"| {seg} | {win}d | {x['n_weeks']} | {_f(x['ic'])} ({_f(x['t'], 1)}) | {_f(x['ipca_minus'])} ({_f(x['t_diff'], 1)}) |")
    L += ["", f"- **Verdicts: llm_in_sample {V['H-PD3']['llm_in_sample']}, clean_holdout {V['H-PD3']['clean_holdout']}.**", "",
          "Model-implied expected weekly pnl_d by family under stylised contexts (full in-sample fit, chosen K; x100):", "",
          "| context | " + " | ".join(FAMILIES) + " |", "|---|" + "---|" * len(FAMILIES)]
    for name, x in p3["scenarios_in_sample_fit"].items():
        L.append(f"| {name} | " + " | ".join(_f(100 * x[f], 2) for f in FAMILIES) + " |")
    # PD4
    L += ["", "## H-PD4: where does the gross edge go? (trade level)", "",
          "| segment | subset | trades | sum MFE | kept | exit leak | dir loss | costs | net | share (C+E)/MFE | alt (C+E_win)/kept | capture | hit gross |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for seg in SEGS:
        for sub in ("all", "1h", "4h"):
            x = out["H-PD4"][seg][sub]
            L.append(f"| {seg} | {sub} | {x['n_trades']} | {_f(x['D_mfe'], 1)} | {_f(x['kept'], 1)} | {_f(x['exit_leak'], 1)} | {_f(x['dir_loss'], 1)} | "
                     f"{_f(x['costs'], 1)} | {_f(x['net'], 1)} | {_f(x['share'])} | {_f(x['share_alt'])} | {_f(x['capture'])} | {_f(x['hit_gross'])} |")
    for seg in SEGS:
        x = out["H-PD4"][seg]
        L.append(f"\n- {seg}: share {_f(x['all']['share'])}, 95% CI [{_f(x['ci95_share'][0])}, {_f(x['ci95_share'][1])}] ({x['n_weeks']} entry weeks); "
                 f"verdict **{V['H-PD4'][seg]}**.")
    L += ["", "Per strategy (llm_in_sample | clean_holdout): share, capture, costs / kept gross:", "",
          "| strategy | IS share | IS capture | IS cost/kept | HO share | HO capture | HO cost/kept |", "|---|---|---|---|---|---|---|"]
    bs_is, bs_ho = out["H-PD4"]["llm_in_sample"]["by_strategy"], out["H-PD4"]["clean_holdout"]["by_strategy"]
    for s in sorted(set(bs_is) | set(bs_ho)):
        a, b = bs_is.get(s, {}), bs_ho.get(s, {})
        L.append(f"| {s} | {_f(a.get('share'))} | {_f(a.get('capture'))} | {_f(a.get('cost_over_gross_pos'))} | {_f(b.get('share'))} | {_f(b.get('capture'))} | {_f(b.get('cost_over_gross_pos'))} |")
    L += ["", "## Multiple testing", ""]
    for seg in SEGS:
        h = out["holm"][seg]
        L.append(f"- {seg}: Holm over m = {h['m']} PD tests; adjusted: " + ", ".join(f"{k} {_f(v, 4)}" for k, v in h["adjusted"].items() if not k.startswith("H-PD2:")))
    L += ["", "## Caveats", "",
          "- Factor returns are built from the same 23 coins the strategies trade: a high R^2 on TSMOM says 'this is trend', not that "
          "an external risk premium was found. Survivorship: L is today's survivors.",
          "- TradFi factors (QQQ, gold, oil) exist only from 2025-2026, so the in-sample attribution is crypto-only (PAXG gold from 2025 as robustness).",
          "- IPCA characteristics are context of the symbol plus the strategy's own trailing Sharpe; the panel has 15-16 strategies, so a "
          "weekly cross-sectional IC is noisy (each IC uses ~16 points).",
          "- MFE uses bar highs and lows (1h or 4h), so it is an upper bound on what any exit could capture; exit leakage is therefore an upper bound too.",
          "- shock_regime is data-generated; calendar strategies trade rarely. T sleeves are holdout-only.",
          "", "## Rule candidates", "", "```yaml", rule_yaml.rstrip(), "```", ""]
    import re

    pm = rep / f"{REPORT}.md"
    # sleeve keys use 'name|U'; a bare '|' would split Markdown table cells
    txt = "\n".join(re.sub(r"(\w)\|(LT|L|T)\b", r"\1/\2", x) if x.startswith("| ") else x for x in L)
    pm.write_text(txt, encoding="utf-8")
    return pj, pm


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--record", action="store_true")
    ap.add_argument("--out", default=None, help="raw JSON output path (scratch)")
    a = ap.parse_args()
    out = run()
    s = json.dumps(out, indent=1, default=str)
    if a.out:
        Path(a.out).write_text(s, encoding="utf-8")
    print(json.dumps({"verdicts": out["verdicts"], "K": out["H-PD3"]["K_chosen"]}, indent=1, default=str))
    if a.record:
        print(record_all(out))


if __name__ == "__main__":
    main()


__all__ = ["effective_bets", "meucci_enb", "ipca_fit", "managed", "pca_stats", "decompose", "factor_panel"]
