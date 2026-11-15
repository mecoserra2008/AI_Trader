"""Trend / breakout / state-space block of the 2026-11-10 program: H-BO1, H-MC1, H-KF1, H-KF2.

Registered in research/hypotheses.yaml (block "Program 2026-11-10", registered BEFORE testing). Everything below was
fixed before any result was computed (diary 2026-11-10, "A6 trend-state-space: start").

Common conventions
------------------
* Data: market_frame(sym, freq, now(), funding=False, oi=False) (point in time; sigma_next = EWMA sigma for bar t+1
  known at the close of t; sigma_risk_next = robust bias-corrected sigma, no weekend shrink). Signals and events are
  computed on the FULL history; results are split by DECISION (entry / event / forecast) date at
  research.knowledge_cutoff (2026-07-01): llm_in_sample before, clean_holdout on/after (RESEARCH_PROTOCOL §3b).
* Costs: config/risk.yaml taker + slippage = 7.5 bps per side (15 bps round trip).
* Universes: L = trend_zoo_study.UNIVERSE (23 crypto perps); T = 16 Bybit TradFi perps (>= 120 complete days
  required; they start Mar-Jun 2026, so they are mostly holdout).
* Multiple testing: Holm across this block's primary tests, per segment (H-BO1: 4 = {L, T} x {4h, 12h}; H-MC1: 4
  variants' trade-mean t; H-KF1: 3 values of q; H-KF2: 1 IC) = 12 tests. Two-sided normal p from each t. A
  registered "t > 2" requirement counts as met only if t > 2 AND Holm-adjusted p < 0.05; a registered "|t| < 2"
  (null) requirement fails only for a test that is Holm-significant.

H-BO1 (floor pivots; the registered statement is the NULL "pivots are not special")
----------------------------------------------------------------------------------
* Prior day d-1 (UTC) must have 24 complete 1h bars and a non-zero range (TradFi perps freeze at weekends). Its
  H, L, C (C = close of the 23:00 bar = C_prev) give P = (H+L+C)/3, R1 = 2P - L, S1 = 2P - H (R1 >= C >= S1 always).
* Real events on day d: the first 1h bar whose close is > R1 (side +1) and the first whose close is < S1 (side -1);
  both can occur on one day. Signed net return over h in {4, 12} bars from the event close:
      y = (side * (log c_{i+h} - log c_i) - 2 * cost) / (sigma_next_i * sqrt(h))     (sigma units, net)
  Events whose forward window has a time gap are dropped.
* Placebo (primary; exactly one per real event): the event's distance k = |log(level / C_prev)| / sigma_0, where
  sigma_0 = sigma_next of the last bar of d-1 (the 1h sigma known at the prior close). A day d' != d is drawn at
  random (seeded, uniform) from the same symbol's valid days in the same segment; the placebo level is
  C_prev(d') * exp(side * k * sigma_0(d')), i.e. the same side and the same sigma distance from that day's prior
  close, but not a pivot. The placebo event is the first 1h close beyond it on d' (same side), with the same return
  definition. If d' never crosses, d' is redrawn (max 200 draws; otherwise the real event has no placebo and is
  kept with a flag, n reported).
  Secondary diagnostic (not used for the verdict): the mirrored placebo on the same day d at the same distance k on
  the opposite side (level C_prev * exp(-side * k * sigma_0), side -side). It exists only on days that ALSO had a
  real event, i.e. it is conditioned on a wide two-sided range, so it is biased toward whipsaw days.
* Statistic: OLS of y on [1, is_pivot] over real + placebo events pooled across the universe's symbols, standard
  errors clustered by event UTC date (CR1). The pivot-minus-placebo coefficient's t is the test.
* Verdict (literal, Holm as above): supports if every (universe, horizon) test with >= 30 real events has
  |t| < 2; contradicts if any has t > 2; a Holm-significant t < -2 alone -> inconclusive.

H-MC1 (Market Cipher B dots; registered as the NULL "no edge")
---------------------------------------------------------------
* features.cipher (LazyBear WaveTrend 9/12/4, dots at -/+53, VuManChu money-flow proxy). Variants (fixed): {1h, 4h}
  x {MF filter on, off}. Warm-up: 200 bars. Trade: enter at the dot close, exit at the next opposite cross or after
  12 bars; net return in risk units (side * log return - 15 bps) / sigma_next at entry; segment by entry date.
* NW t over the pooled trade series ordered by entry time. Random-entry baseline (1000 draws): for each real trade a
  random entry bar from the same symbol and segment (after warm-up, with 12 bars of future data), keeping the real
  trade's side and holding period; p = share of draws whose pooled mean >= the real mean.
* Deflated Sharpe (Bailey & Lopez de Prado; trend_zoo_study.deflated_sharpe) over the 4 variants on each variant's
  daily portfolio series (per symbol: sum of trade P&L by exit date, 0 on days without exits; mean over the symbols
  live that day), N = 4, V = variance of the 4 daily Sharpes, per segment.
* Verdict (literal): llm_in_sample supports iff no variant has DSR > 0.95; clean_holdout supports iff no variant has
  holdout DSR > 0.95 and the variant with the best in-sample mean trade return has holdout mean <= 0; otherwise
  contradicts. Universe T is reported for the holdout only, as robustness (not in the verdict).

H-KF1 (vol-adaptive vs fixed-gain local-linear-trend slope)
-----------------------------------------------------------
* Model: smooth local linear trend on 4h log close (level noise 0): y = mu + eps, mu' = mu + beta, beta' = beta +
  zeta. q = Var(zeta) / Var(eps) in {1e-4, 1e-3, 1e-2} (median q = 1e-3).
* Constant-R filter: R = 1, Q_zeta = q (the gain depends only on the ratio, so this is the fixed-gain filter).
* Adaptive filter: R_t = sigma_risk_next_{t-1}^2 (robust sigma for bar t known at the close of t-1) and
  Q_zeta,t = q * Rref_t, Rref_t = trailing 180-day (1080-bar) median of R (causal, min 30 bars). Matching: whenever
  R_t = Rref_t the two filters have the same Riccati ratio and hence the same steady-state gain; the adaptive filter
  only lowers (raises) its gain when the current vol is above (below) its trailing median.
* Signal tanh(slope_t / sd_t), sd_t = trailing 1080-bar std of the filtered slope (min 180 bars), same rule for both.
* evaluate.strategy_pnl, daily sums; common start = first full UTC date after bar 1080 + 1. Paired daily difference
  (adaptive - constant) averaged over symbols per date; NW t over dates per segment.
* Verdict: llm_in_sample supports iff q = 1e-3 difference > 0 with t > 2 (Holm-significant); clean_holdout supports
  iff the q = 1e-3 difference > 0 (same sign); else contradicts.

H-KF2 (rolling MLE signal-to-noise ratio vs next-month trend profitability)
---------------------------------------------------------------------------
* Decision dates D = the first day of each month (00:00 UTC). Per symbol, fit the full LLT (eps, eta, zeta) by
  concentrated diffuse MLE (features.state_space.llt_mle) on 4h bars with close_ts in (D - 180d, D] (>= 900 bars),
  y = 100 * log close. q_hat = Var(zeta) / Var(eta) (the registry's definition).
* Target: sum of the 'ensemble' daily P&L of data/priors/trend_zoo_daily_4h.parquet over dates [D, D + 30d)
  (>= 25 days present). Per D: Spearman IC across symbols (>= 8). Mean IC, NW t over months, per segment by D.
* Verdict: llm_in_sample supports iff mean IC > 0 with t > 2 (Holm-significant); clean_holdout supports iff mean
  IC > 0 (>= 3 months, else inconclusive); else contradicts.

    uv run python -m ai_trader.research.breakout_cipher_kalman_study [--record] [--workers 4]
"""

from __future__ import annotations

import argparse
import json
import math
import time
import zlib
from concurrent.futures import ProcessPoolExecutor
from datetime import UTC, datetime, timedelta
from pathlib import Path
from statistics import NormalDist

import numpy as np
import polars as pl

from ai_trader.core.config import load
from ai_trader.core.time import now
from ai_trader.features.cipher import cipher_trades
from ai_trader.features.state_space import llt_filter, llt_mle, steady_state_gain
from ai_trader.research.evaluate import knowledge_cutoff, newey_west_t, strategy_pnl
from ai_trader.research.trend_zoo_study import UNIVERSE, deflated_sharpe

UNIV_L = tuple(UNIVERSE)
UNIV_T = tuple("XAUUSDT XAGUSDT CLUSDT BZUSDT TSLAUSDT INTCUSDT GOOGLUSDT AAPLUSDT MSTRUSDT NVDAUSDT MUUSDT QQQUSDT "
               "SOXLUSDT SPCXUSDT PAXGUSDT XAUTUSDT".split())
SEGS = ("llm_in_sample", "clean_holdout")
BO_H = (4, 12)
MC_VARIANTS = (("1h", True), ("1h", False), ("4h", True), ("4h", False))
MC_HOLD = 12
MC_WARM = 200
N_DRAWS = 1000
KF_Q = (1e-4, 1e-3, 1e-2)
KF_WIN = 1080          # 180 days of 4h bars
KF_SD_MIN = 180
KF2_DAYS = 180
KF2_MIN_BARS = 900
KF2_HORIZON = 30
MIN_T_DAYS = 120
REPORT = "trend_state_space_2026-11-10"


def cost_frac() -> float:
    c = load().risk["costs"]
    return (c["taker_bps"] + c["slippage_bps"]) / 1e4


def seg_of(d) -> str:
    return "llm_in_sample" if d < knowledge_cutoff().date() else "clean_holdout"


def _seed(*parts) -> int:
    return zlib.crc32("|".join(map(str, parts)).encode())


# ====================================================================== statistics
def cluster_ols(y: np.ndarray, X: np.ndarray, groups: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """OLS coefficients and CR1 cluster-robust standard errors."""
    y = np.asarray(y, float)
    X = np.asarray(X, float)
    n, k = X.shape
    xtx_inv = np.linalg.inv(X.T @ X)
    b = xtx_inv @ X.T @ y
    u = y - X @ b
    _, g = np.unique(groups, return_inverse=True)
    G = int(g.max()) + 1
    s = np.zeros((G, k))
    np.add.at(s, g, X * u[:, None])
    meat = s.T @ s
    c = (G / (G - 1)) * ((n - 1) / (n - k)) if G > 1 and n > k else 1.0
    V = c * xtx_inv @ meat @ xtx_inv
    return b, np.sqrt(np.clip(np.diag(V), 0, None))


def p_two(t: float) -> float:
    return 2 * (1 - NormalDist().cdf(abs(t))) if np.isfinite(t) else float("nan")


def holm(pvals: dict[str, float]) -> dict[str, float]:
    items = [(k, p) for k, p in pvals.items() if np.isfinite(p)]
    items.sort(key=lambda kv: kv[1])
    m = len(items)
    out, run = {}, 0.0
    for i, (k, p) in enumerate(items):
        run = max(run, min(1.0, (m - i) * p))
        out[k] = run
    for k, p in pvals.items():
        out.setdefault(k, float("nan"))
    return out


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    """Spearman correlation with tie-averaged ranks (q_hat piles up at its lower bound: ties are common)."""
    from scipy.stats import rankdata

    ra, rb = rankdata(a), rankdata(b)
    return float(np.corrcoef(ra, rb)[0, 1]) if ra.std() > 0 and rb.std() > 0 else float("nan")


def _sharpe_daily(x: np.ndarray) -> float:
    x = x[np.isfinite(x)]
    if len(x) <= 30 or x.std(ddof=1) == 0:
        return float("nan")
    return float(x.mean() / x.std(ddof=1) * np.sqrt(365))


# ====================================================================== H-BO1
def _day_table(f: pl.DataFrame) -> dict:
    ts = f["event_ts"]
    dates = ts.dt.date().to_numpy()
    hrs = ts.dt.hour().to_numpy()
    uniq, first = np.unique(dates, return_index=True)
    bounds = np.r_[first, len(dates)]
    return {"dates": uniq, "a": bounds[:-1], "b": bounds[1:], "hrs": hrs}


def pivot_events(f: pl.DataFrame, sym: str) -> list[dict]:
    """Real pivot events + one time-shifted placebo + one mirrored placebo per real event (see module doc)."""
    c = cost_frac()
    lc = np.log(f["close"].to_numpy())
    hi = f["high"].to_numpy()
    lo = f["low"].to_numpy()
    sig = f["sigma_next"].to_numpy()
    t_ns = f["event_ts"].dt.epoch("s").to_numpy()
    tab = _day_table(f)
    D = len(tab["dates"])
    # per-day prior stats (index = day d, stats from day d-1)
    valid = np.zeros(D, bool)
    cprev = np.full(D, np.nan)
    s0 = np.full(D, np.nan)
    r1 = np.full(D, np.nan)
    s1 = np.full(D, np.nan)
    for d in range(1, D):
        a, b = tab["a"][d - 1], tab["b"][d - 1]
        if (tab["dates"][d] - tab["dates"][d - 1]).astype("timedelta64[D]").astype(int) != 1 or b - a != 24:
            continue
        H, L = hi[a:b].max(), lo[a:b].min()
        C = math.exp(lc[b - 1])
        sg = sig[b - 1]
        if not (H > L) or not np.isfinite(sg) or sg <= 0 or tab["b"][d] - tab["a"][d] < 1:
            continue
        P = (H + L + C) / 3
        valid[d], cprev[d], s0[d], r1[d], s1[d] = True, C, sg, 2 * P - L, 2 * P - H
    seg_day = np.array([seg_of(x.astype(object)) for x in tab["dates"]])

    def fwd(i: int, side: int) -> dict | None:
        out = {}
        for h in BO_H:
            if i + h >= len(lc) or t_ns[i + h] - t_ns[i] != 3600 * h or not np.isfinite(sig[i]) or sig[i] <= 0:
                return None
            g = side * (lc[i + h] - lc[i])
            out[f"y{h}"] = (g - 2 * c) / (sig[i] * math.sqrt(h))
            out[f"g{h}"] = g / (sig[i] * math.sqrt(h))
        return out

    def first_cross(d: int, log_level: float, side: int) -> int | None:
        a, b = tab["a"][d], tab["b"][d]
        x = lc[a:b]
        hit = np.nonzero(x > log_level)[0] if side > 0 else np.nonzero(x < log_level)[0]
        return int(a + hit[0]) if len(hit) else None

    rng = np.random.default_rng(_seed("BO1", sym))
    pool = {s: np.nonzero(valid & (seg_day == s))[0] for s in SEGS}
    rows = []
    for d in np.nonzero(valid)[0]:
        seg = seg_day[d]
        date = tab["dates"][d].astype(object)
        for side, lvl in ((1, r1[d]), (-1, s1[d])):
            i = first_cross(d, math.log(lvl), side)
            if i is None:
                continue
            fr = fwd(i, side)
            if fr is None:
                continue
            k = abs(math.log(lvl / cprev[d])) / s0[d]
            base = {"symbol": sym, "seg": seg, "side": side, "k": k}
            rows.append({**base, "kind": "pivot", "date": date, "hour": int(tab["hrs"][i]), **fr})
            # time-shifted placebo
            cand = pool[seg][pool[seg] != d]
            got = None
            if len(cand):
                for _ in range(200):
                    dd = int(cand[rng.integers(len(cand))])
                    j = first_cross(dd, math.log(cprev[dd]) + side * k * s0[dd], side)
                    if j is None:
                        continue
                    fp = fwd(j, side)
                    if fp is None:
                        continue
                    got = {**base, "kind": "placebo", "date": tab["dates"][dd].astype(object), "hour": int(tab["hrs"][j]), **fp}
                    break
            if got is not None:
                rows.append(got)
            else:
                rows[-1]["no_placebo"] = True
            # mirrored placebo (diagnostic)
            j = first_cross(d, math.log(cprev[d]) - side * k * s0[d], -side)
            if j is not None:
                fm = fwd(j, -side)
                if fm is not None:
                    rows.append({**base, "side": -side, "kind": "mirror", "date": date, "hour": int(tab["hrs"][j]), **fm})
    return rows


def bo1_stats(ev: pl.DataFrame) -> dict:
    out = {}
    for seg in SEGS:
        e = ev.filter(pl.col("seg") == seg)
        res = {}
        for h in BO_H:
            r = {}
            piv = e.filter(pl.col("kind") == "pivot")
            r["n_pivot"] = piv.height
            r["n_dates"] = int(piv["date"].n_unique()) if piv.height else 0
            r["n_symbols"] = int(piv["symbol"].n_unique()) if piv.height else 0
            if piv.height < 30:
                res[f"{h}h"] = r
                continue
            dnum = lambda s: np.array([x.toordinal() for x in s.to_list()])  # noqa: E731
            yp = piv[f"y{h}"].to_numpy()
            b, se = cluster_ols(yp, np.ones((len(yp), 1)), dnum(piv["date"]))
            r.update({"pivot_mean_net": float(b[0]), "pivot_t_net": float(b[0] / se[0]),
                      "pivot_mean_gross": float(piv[f"g{h}"].mean()), "frac_pos": float((yp > 0).mean()),
                      "k_median": float(piv["k"].median()), "no_placebo": int(piv["no_placebo"].sum()) if "no_placebo" in piv.columns else 0})
            for side in (1, -1):
                ps = piv.filter(pl.col("side") == side)
                if ps.height >= 30:
                    bs, ss = cluster_ols(ps[f"y{h}"].to_numpy(), np.ones((ps.height, 1)), dnum(ps["date"]))
                    r[f"pivot_mean_side{side:+d}"] = float(bs[0])
                    r[f"pivot_t_side{side:+d}"] = float(bs[0] / ss[0])
            for kind in ("placebo", "mirror"):
                pl_ = e.filter(pl.col("kind") == kind)
                if pl_.height < 30:
                    continue
                y = np.r_[yp, pl_[f"y{h}"].to_numpy()]
                X = np.c_[np.ones(len(y)), np.r_[np.ones(len(yp)), np.zeros(pl_.height)]]
                g = np.r_[dnum(piv["date"]), dnum(pl_["date"])]
                b, se = cluster_ols(y, X, g)
                r[f"{kind}_n"] = pl_.height
                r[f"{kind}_mean_net"] = float(b[0])
                r[f"diff_vs_{kind}"] = float(b[1])
                r[f"t_vs_{kind}"] = float(b[1] / se[1])
            res[f"{h}h"] = r
        out[seg] = res
    return out


# ====================================================================== H-MC1
def mc_symbol(f: pl.DataFrame, sym: str, freq: str, mf: bool, n_draws: int = N_DRAWS) -> dict:
    c = cost_frac()
    lc = np.log(f["close"].to_numpy())
    sig = f["sigma_next"].to_numpy()
    dates = f["event_ts"].dt.date().to_numpy()
    seg = np.array(["llm_in_sample" if x < knowledge_cutoff().date() else "clean_holdout" for x in dates.tolist()])
    tr = cipher_trades(f["open"].to_numpy(), f["high"].to_numpy(), f["low"].to_numpy(), f["close"].to_numpy(), mf, MC_HOLD, MC_WARM)
    trades = []
    for i, j, s in tr:
        if not np.isfinite(sig[i]) or sig[i] <= 0:
            continue
        g = s * (lc[j] - lc[i])
        trades.append({"symbol": sym, "variant": f"{freq}_mf{'on' if mf else 'off'}", "entry_ts": f["event_ts"][i],
                       "exit_date": dates[j].astype(object), "seg": seg[i], "side": s, "hold": j - i,
                       "net": (g - 2 * c) / sig[i], "gross": g / sig[i]})
    # random-entry baseline sums per segment
    base = {}
    n = len(lc)
    elig_all = np.arange(MC_WARM, n - MC_HOLD)
    elig_all = elig_all[np.isfinite(sig[elig_all]) & (sig[elig_all] > 0)]
    for sg in SEGS:
        tt = [t for t in trades if t["seg"] == sg]
        elig = elig_all[seg[elig_all] == sg]
        if not tt or len(elig) == 0:
            base[sg] = {"n": 0, "sums": np.zeros(n_draws)}
            continue
        rng = np.random.default_rng(_seed("MC1", sym, freq, mf, sg))
        sides = np.array([t["side"] for t in tt])
        holds = np.array([t["hold"] for t in tt])
        e = elig[rng.integers(len(elig), size=(n_draws, len(tt)))]
        ex = np.minimum(e + holds[None, :], n - 1)
        r = (sides[None, :] * (lc[ex] - lc[e]) - 2 * c) / sig[e]
        base[sg] = {"n": len(tt), "sums": r.sum(axis=1)}
    # daily series: P&L by exit date on all live dates after warm-up
    live = np.unique(dates[MC_WARM:]) if n > MC_WARM else np.array([], dtype=dates.dtype)
    daily = pl.DataFrame({"date": live}).with_columns(pl.lit(0.0).alias("pnl"))
    if trades:
        tp = pl.DataFrame({"date": [t["exit_date"] for t in trades], "x": [t["net"] for t in trades]}).group_by("date").agg(pl.col("x").sum())
        daily = daily.join(tp, on="date", how="left").with_columns((pl.col("pnl") + pl.col("x").fill_null(0.0)).alias("pnl")).drop("x")
    daily = daily.with_columns(pl.lit(sym).alias("symbol"), pl.lit(f"{freq}_mf{'on' if mf else 'off'}").alias("variant"))
    return {"trades": trades, "base": base, "daily": daily}


def mc1_stats(trades: pl.DataFrame, base: dict, daily: pl.DataFrame, universe: str) -> dict:
    out: dict = {}
    variants = [f"{fq}_mf{'on' if m else 'off'}" for fq, m in MC_VARIANTS]
    for seg in SEGS:
        res = {}
        port = {}
        for v in variants:
            t = trades.filter((pl.col("variant") == v) & (pl.col("seg") == seg)).sort("entry_ts")
            r = {"n_trades": t.height}
            if t.height:
                m, tt, _ = newey_west_t(t["net"].to_numpy())
                sums = sum(b[seg]["sums"] for (u, vv, s_), b in base.items() if vv == v and u == universe)
                cnt = sum(b[seg]["n"] for (u, vv, s_), b in base.items() if vv == v and u == universe)
                rm = sums / cnt if cnt else np.full(N_DRAWS, np.nan)
                r.update({"mean_net": m, "t_nw": tt, "mean_gross": float(t["gross"].mean()), "hit": float((t["net"] > 0).mean()),
                          "hold_mean": float(t["hold"].mean()), "frac_long": float((t["side"] > 0).mean()),
                          "n_symbols": int(t["symbol"].n_unique()),
                          "random_mean": float(np.nanmean(rm)), "random_q05_q95": [float(q) for q in np.nanquantile(rm, [0.05, 0.95])],
                          "p_random_ge_real": float(np.mean(rm >= m)), "excess_vs_random": float(m - np.nanmean(rm))})
                for side in (1, -1):
                    ts_ = t.filter(pl.col("side") == side)
                    if ts_.height >= 10:
                        r[f"mean_side{side:+d}"] = float(ts_["net"].mean())
            d = (daily.filter(pl.col("variant") == v).with_columns(
                pl.when(pl.col("date") < knowledge_cutoff().date()).then(pl.lit("llm_in_sample")).otherwise(pl.lit("clean_holdout")).alias("seg"))
                .filter(pl.col("seg") == seg).group_by("date").agg(pl.col("pnl").mean()).sort("date"))
            port[v] = d["pnl"].to_numpy()
            r["port_sharpe"] = _sharpe_daily(port[v])
            res[v] = r
        ok = [v for v in variants if len(port[v]) > 30 and np.std(port[v]) > 0]
        if len(ok) >= 2:
            srs = np.array([port[v].mean() / port[v].std(ddof=1) for v in ok])
            for v in ok:
                res[v]["dsr"] = deflated_sharpe(port[v], srs, len(variants))
        out[seg] = res
    return out


# ====================================================================== H-KF1
def kf_signals(f: pl.DataFrame) -> dict[str, np.ndarray]:
    y = np.log(f["close"].to_numpy())
    rs = f["sigma_risk_next"].to_numpy() if "sigma_risk_next" in f.columns else f["sigma_next"].to_numpy()
    rs = np.where(np.isfinite(rs) & (rs > 0), rs, f["sigma_next"].to_numpy())
    R = np.r_[np.nan, rs[:-1] ** 2]  # R_t known at the close of t-1
    R[0] = R[1] if len(R) > 1 else np.nan
    rref = pl.Series(R).rolling_median(window_size=KF_WIN, min_samples=30).to_numpy()
    out = {}
    for q in KF_Q:
        for kind in ("adaptive", "constant"):
            if kind == "adaptive":
                fo = llt_filter(y, R, q * rref, q_eta=0.0, p0_slope=1.0)
            else:
                fo = llt_filter(y, 1.0, q, q_eta=0.0, p0_slope=1.0)
            slope = fo.slope
            if kind == "adaptive":
                slope = np.where(np.isfinite(rref), slope, np.nan)
            sd = pl.Series(slope).rolling_std(window_size=KF_WIN, min_samples=KF_SD_MIN).to_numpy()
            with np.errstate(invalid="ignore", divide="ignore"):
                s = np.tanh(slope / sd)
            out[f"{kind}_q{q:g}"] = np.where(np.isfinite(s), s, 0.0)
    return out


def kf1_symbol(f: pl.DataFrame, sym: str) -> pl.DataFrame | None:
    if f.height < KF_WIN + 400:
        return None
    sigs = kf_signals(f)
    start = f["event_ts"][KF_WIN].date() + timedelta(days=1)
    frames = []
    corr = {}
    keep = (f["event_ts"].dt.date() >= start).to_numpy()
    for q in KF_Q:
        a, b = sigs[f"adaptive_q{q:g}"][keep], sigs[f"constant_q{q:g}"][keep]
        corr[q] = float(np.corrcoef(a, b)[0, 1]) if a.std() > 0 and b.std() > 0 else float("nan")
    for name, s in sigs.items():
        p = strategy_pnl(f, pl.Series(name, s))
        d = (p.with_columns(pl.col("event_ts").dt.date().alias("date")).group_by("date")
             .agg(pl.col("pnl").sum(), pl.col("cost").sum()).filter(pl.col("date") >= start)
             .with_columns(pl.lit(sym).alias("symbol"), pl.lit(name).alias("variant")))
        frames.append(d)
    out = pl.concat(frames)
    return out.with_columns(pl.lit(json.dumps({str(k): v for k, v in corr.items()})).alias("sigcorr"))


def kf1_stats(daily: pl.DataFrame) -> dict:
    d = daily.with_columns(pl.when(pl.col("date") < knowledge_cutoff().date()).then(pl.lit("llm_in_sample")).otherwise(pl.lit("clean_holdout")).alias("seg"))
    sc = {}
    for s in d.select("symbol", "sigcorr").unique()["sigcorr"].to_list():
        for k, v in json.loads(s).items():
            sc.setdefault(k, []).append(v)
    out = {"signal_corr_adaptive_vs_constant": {k: float(np.nanmean(v)) for k, v in sc.items()},
           "steady_state_gain": {f"{q:g}": steady_state_gain(q) for q in KF_Q}}
    for seg in SEGS:
        e = d.filter(pl.col("seg") == seg)
        res = {}
        for q in KF_Q:
            a = e.filter(pl.col("variant") == f"adaptive_q{q:g}").select("symbol", "date", pl.col("pnl").alias("a"), pl.col("cost").alias("ca"))
            b = e.filter(pl.col("variant") == f"constant_q{q:g}").select("symbol", "date", pl.col("pnl").alias("b"), pl.col("cost").alias("cb"))
            j = a.join(b, on=["symbol", "date"])
            if j.height == 0:
                res[f"q{q:g}"] = {"n_days": 0}
                continue
            per = j.group_by("date").agg((pl.col("a") - pl.col("b")).mean().alias("d"), pl.col("a").mean(), pl.col("b").mean()).sort("date")
            m, t, n = newey_west_t(per["d"].to_numpy())
            sym = j.group_by("symbol").agg(pl.col("a"), pl.col("b"))
            sa = np.array([_sharpe_daily(np.asarray(x)) for x in sym["a"].to_list()])
            sb = np.array([_sharpe_daily(np.asarray(x)) for x in sym["b"].to_list()])
            ok = np.isfinite(sa) & np.isfinite(sb)
            res[f"q{q:g}"] = {
                "n_days": n, "n_symbols": int(sym.height), "diff_mean_daily": m, "diff_t_nw": t,
                "sharpe_adaptive_mean_sym": float(np.mean(sa[ok])) if ok.any() else float("nan"),
                "sharpe_constant_mean_sym": float(np.mean(sb[ok])) if ok.any() else float("nan"),
                "frac_sym_adaptive_better": float(np.mean(sa[ok] > sb[ok])) if ok.any() else float("nan"),
                "port_sharpe_adaptive": _sharpe_daily(per["a"].to_numpy()), "port_sharpe_constant": _sharpe_daily(per["b"].to_numpy()),
                "port_sharpe_diff_series": _sharpe_daily(per["d"].to_numpy()),
                "cost_share_adaptive": float(j["ca"].sum() / max(1e-12, abs(j["a"].sum()) + j["ca"].sum())),
                "cost_share_constant": float(j["cb"].sum() / max(1e-12, abs(j["b"].sum()) + j["cb"].sum())),
            }
        out[seg] = res
    return out


# ====================================================================== H-KF2
def kf2_symbol(f: pl.DataFrame, sym: str) -> list[dict]:
    ct = f["close_ts"].to_numpy().astype("datetime64[s]")
    y = 100 * np.log(f["close"].to_numpy())
    if f.height < KF2_MIN_BARS:
        return []
    first = ct[0] + np.timedelta64(KF2_DAYS, "D")
    m0 = first.astype("datetime64[M]") + np.timedelta64(1, "M")
    last = ct[-1].astype("datetime64[M]")
    rows = []
    mon = m0
    while mon <= last:
        D = mon.astype("datetime64[s]")
        sel = (ct <= D) & (ct > D - np.timedelta64(KF2_DAYS, "D"))
        if sel.sum() >= KF2_MIN_BARS:
            fit = llt_mle(y[sel])
            rows.append({"symbol": sym, "D": mon.astype("datetime64[D]").astype(object), "q_hat": fit.q_zeta, "q_eps": fit.q_eps,
                         "sigma2_eta": fit.sigma2_eta, "loglik": fit.loglik, "n_bars": int(sel.sum())})
        mon = mon + np.timedelta64(1, "M")
    return rows


def kf2_stats(fits: pl.DataFrame, panel: pl.DataFrame) -> dict:
    ens = panel.filter(pl.col("variant") == "ensemble").select("symbol", "date", "pnl")
    rows = []
    for r in fits.iter_rows(named=True):
        D = r["D"]
        w = ens.filter((pl.col("symbol") == r["symbol"]) & (pl.col("date") >= D) & (pl.col("date") < D + timedelta(days=KF2_HORIZON)))
        if w.height >= 25:
            rows.append({**r, "target": float(w["pnl"].sum()), "n_days": w.height})
    t = pl.DataFrame(rows)
    out = {"n_fits": fits.height, "n_with_target": t.height,
           "q_hat_median": float(fits["q_hat"].median()), "q_hat_q10_q90": [float(fits["q_hat"].quantile(0.1)), float(fits["q_hat"].quantile(0.9))],
           "q_eps_median": float(fits["q_eps"].median()), "frac_q_hat_at_lower_bound": float((fits["q_hat"] < 2e-7).mean())}
    ics = []
    for D, g in t.group_by("D"):
        if g.height >= 8:
            ics.append({"D": D[0], "ic": spearman(np.log(g["q_hat"].to_numpy()), g["target"].to_numpy()), "n": g.height})
    ic = pl.DataFrame(ics).sort("D").with_columns(pl.col("D").map_elements(lambda d: seg_of(d), return_dtype=pl.Utf8).alias("seg"))
    out["ic_by_month"] = [{"D": str(r["D"]), "ic": r["ic"], "n": r["n"], "seg": r["seg"]} for r in ic.iter_rows(named=True)]
    # persistence of q_hat (is it a stable symbol trait?) and pooled level relation
    for seg in SEGS:
        x = ic.filter(pl.col("seg") == seg)["ic"].to_numpy()
        m, tt, n = newey_west_t(x)
        tg = t.with_columns(pl.col("D").map_elements(lambda d: seg_of(d), return_dtype=pl.Utf8).alias("seg")).filter(pl.col("seg") == seg)
        out[seg] = {"n_months": int(len(x)), "mean_ic": float(np.nanmean(x)) if len(x) else float("nan"), "ic_t_nw": tt,
                    "ic_t_plain": float(np.nanmean(x) / (np.nanstd(x, ddof=1) / np.sqrt(np.isfinite(x).sum()))) if np.isfinite(x).sum() > 2 else float("nan"),
                    "frac_pos": float(np.mean(x > 0)) if len(x) else float("nan"),
                    "pooled_spearman": spearman(np.log(tg["q_hat"].to_numpy()), tg["target"].to_numpy()) if tg.height > 10 else float("nan"),
                    "n_obs": tg.height}
    return out


# ====================================================================== per-symbol job
def frame_4h_from_1h(sym: str, as_of: datetime) -> pl.DataFrame:
    """4h frame aggregated from lake 1h bars (UTC-aligned 00/04/.../20h groups with all 4 bars present), then the same
    sanity filter, listing burn-in, enrich_bars and add_robust_vol as market_frame. Used only when the lake's native 4h
    history is much shorter than its 1h history (the Bybit equity perps: native 4h only from 2026-11-04). A 4h bar is
    known when its 4th 1h bar is known (known_ts = max), so it is point in time."""
    from ai_trader.research.frame import add_robust_vol, enrich_bars
    from ai_trader.storage.lake import Lake

    b = Lake().bars(sym, "1h", as_of).select("event_ts", "known_ts", "open", "high", "low", "close", "volume", "turnover").sort("event_ts")
    b = b.filter((pl.col("open") > 0) & (pl.col("low") > 0) & (pl.col("high") >= pl.max_horizontal("open", "close"))
                 & (pl.col("low") <= pl.min_horizontal("open", "close")))
    g = (b.group_by_dynamic("event_ts", every="4h").agg(
        pl.col("open").first(), pl.col("high").max(), pl.col("low").min(), pl.col("close").last(), pl.col("volume").sum(),
        pl.col("turnover").sum(), pl.col("known_ts").max(), pl.len().alias("_n"))
        .filter(pl.col("_n") == 4).drop("_n").with_columns((pl.col("event_ts") + pl.duration(hours=4)).alias("close_ts")))
    if g.height == 0:
        return g
    rs = load().settings["research"]
    first = g["event_ts"].min()
    days = rs["burn_in_days_new"] if first.year >= rs["burn_in_new_listing_year"] else rs["burn_in_days"]
    g = g.filter(pl.col("event_ts") >= first + timedelta(days=days))
    if g.height == 0:
        return g
    return add_robust_vol(enrich_bars(g.select("event_ts", "close_ts", "known_ts", "open", "high", "low", "close", "volume", "turnover")))


def symbol_job(sym: str, universe: str, as_of_iso: str) -> dict:
    from ai_trader.research.frame import market_frame

    as_of = datetime.fromisoformat(as_of_iso)
    t0 = time.time()
    out: dict = {"symbol": sym, "universe": universe, "bo": [], "mc_trades": [], "mc_base": {}, "mc_daily": [], "kf1": None, "kf2": []}
    f1 = market_frame(sym, "1h", as_of, funding=False, oi=False)
    f4 = market_frame(sym, "4h", as_of, funding=False, oi=False)
    if f1.height and f4.height < 0.8 * f1.height / 4:
        f4 = frame_4h_from_1h(sym, as_of)
        out["f4_source"] = "aggregated_from_1h"
    if f1.height == 0 or f4.height == 0:
        out["skip"] = "no data"
        return out
    days = (f1["event_ts"].max() - f1["event_ts"].min()).days
    if universe == "T" and days < MIN_T_DAYS:
        out["skip"] = f"{days} days < {MIN_T_DAYS}"
        return out
    out["days"] = days
    out["bo"] = pivot_events(f1, sym)
    for fq, mf in MC_VARIANTS:
        r = mc_symbol(f1 if fq == "1h" else f4, sym, fq, mf)
        v = f"{fq}_mf{'on' if mf else 'off'}"
        out["mc_trades"] += r["trades"]
        out["mc_base"][v] = r["base"]
        out["mc_daily"].append(r["daily"])
    if universe == "L":
        out["kf1"] = kf1_symbol(f4, sym)
        out["kf2"] = kf2_symbol(f4, sym)
    out["secs"] = round(time.time() - t0, 1)
    return out


# ====================================================================== orchestration
def run(workers: int = 4, as_of: datetime | None = None) -> dict:
    as_of = as_of or now()
    jobs = [(s, "L") for s in UNIV_L] + [(s, "T") for s in UNIV_T]
    res = []
    with ProcessPoolExecutor(max_workers=min(4, workers)) as ex:
        futs = [ex.submit(symbol_job, s, u, as_of.isoformat()) for s, u in jobs]
        for fu in futs:
            r = fu.result()
            print(f"  {r['symbol']} ({r['universe']}) {r.get('skip', '')} {r.get('secs', '')}s", flush=True)
            res.append(r)
    used = {u: [r["symbol"] for r in res if r["universe"] == u and "skip" not in r] for u in ("L", "T")}
    skipped = {r["symbol"]: r["skip"] for r in res if "skip" in r}
    f4_agg = [r["symbol"] for r in res if r.get("f4_source")]
    out: dict = {"as_of": as_of.isoformat(), "cost_per_side_bps": cost_frac() * 1e4, "knowledge_cutoff": knowledge_cutoff().date().isoformat(),
                 "symbols": used, "skipped": skipped, "f4_aggregated_from_1h": f4_agg}
    # H-BO1
    out["H-BO1"] = {}
    for u in ("L", "T"):
        ev = [e for r in res if r["universe"] == u for e in r["bo"]]
        evdf = pl.DataFrame(ev, infer_schema_length=None) if ev else pl.DataFrame()
        out["H-BO1"][u] = bo1_stats(evdf) if ev else {}
    # H-MC1
    out["H-MC1"] = {}
    for u in ("L", "T"):
        rr = [r for r in res if r["universe"] == u and "skip" not in r]
        tr = pl.DataFrame([t for r in rr for t in r["mc_trades"]], infer_schema_length=None)
        base = {(u, v, r["symbol"]): b for r in rr for v, b in r["mc_base"].items()}
        daily = pl.concat([d for r in rr for d in r["mc_daily"]], how="vertical_relaxed")
        out["H-MC1"][u] = mc1_stats(tr, base, daily, u)
    # H-KF1
    kd = [r["kf1"] for r in res if r["universe"] == "L" and r["kf1"] is not None]
    out["H-KF1"] = kf1_stats(pl.concat(kd))
    # H-KF2
    fits = pl.DataFrame([x for r in res if r["universe"] == "L" for x in r["kf2"]])
    panel_p = load().root / "data" / "priors" / "trend_zoo_daily_4h.parquet"
    if not panel_p.exists():
        from ai_trader.research.trend_zoo_study import build
        build("4h", as_of)
    panel = pl.read_parquet(panel_p)
    out["H-KF2"] = kf2_stats(fits, panel)
    out["H-KF2"]["panel_last_date"] = str(panel["date"].max())
    out["holm"], out["verdicts"] = verdicts(out)
    return out


# ====================================================================== verdicts
def primary_tests(out: dict, seg: str) -> dict[str, float]:
    t = {}
    for u in ("L", "T"):
        for h in BO_H:
            r = out["H-BO1"].get(u, {}).get(seg, {}).get(f"{h}h", {})
            if r.get("n_pivot", 0) >= 30 and "t_vs_placebo" in r:
                t[f"H-BO1:{u}:{h}h"] = r["t_vs_placebo"]
    for v, r in out["H-MC1"]["L"][seg].items():
        if r.get("n_trades", 0) >= 10 and np.isfinite(r.get("t_nw", np.nan)):
            t[f"H-MC1:{v}"] = r["t_nw"]
    for q in KF_Q:
        r = out["H-KF1"][seg].get(f"q{q:g}", {})
        if np.isfinite(r.get("diff_t_nw", np.nan)):
            t[f"H-KF1:q{q:g}"] = r["diff_t_nw"]
    r = out["H-KF2"][seg]
    if np.isfinite(r.get("ic_t_nw", np.nan)):
        t["H-KF2:ic"] = r["ic_t_nw"]
    return t


def verdicts(out: dict) -> tuple[dict, dict]:
    hb, vd = {}, {}
    for seg in SEGS:
        ts = primary_tests(out, seg)
        hp = holm({k: p_two(v) for k, v in ts.items()})
        hb[seg] = {k: {"t": ts[k], "p": p_two(ts[k]), "p_holm": hp[k]} for k in ts}
        sig = lambda k: k in ts and abs(ts[k]) > 2 and hp[k] < 0.05  # noqa: E731
        v = {}
        # BO1
        bo = [k for k in ts if k.startswith("H-BO1")]
        if not bo:
            v["H-BO1"] = "inconclusive"
        elif any(sig(k) and ts[k] > 2 for k in bo):
            v["H-BO1"] = "contradicts"
        elif any(sig(k) and ts[k] < -2 for k in bo):
            v["H-BO1"] = "inconclusive"
        else:
            v["H-BO1"] = "supports"
        # MC1
        mc = out["H-MC1"]["L"]
        dsr_any = any(r.get("dsr", {}).get("dsr", 0) > 0.95 for r in mc[seg].values())
        if seg == "llm_in_sample":
            v["H-MC1"] = "contradicts" if dsr_any else "supports"
        else:
            best = max(mc["llm_in_sample"], key=lambda k: np.nan_to_num(mc["llm_in_sample"][k].get("mean_net", np.nan), nan=-9))
            hm = mc[seg][best].get("mean_net", np.nan)
            if not np.isfinite(hm):
                v["H-MC1"] = "inconclusive"
            else:
                v["H-MC1"] = "supports" if (hm <= 0 and not dsr_any) else "contradicts"
            v["H-MC1_best_in_sample_variant"] = best
        # KF1
        r = out["H-KF1"][seg].get("q0.001", {})
        if not r.get("n_days"):
            v["H-KF1"] = "inconclusive"
        elif seg == "llm_in_sample":
            v["H-KF1"] = "supports" if (r["diff_mean_daily"] > 0 and r["diff_t_nw"] > 2 and sig("H-KF1:q0.001")) else "contradicts"
        else:
            v["H-KF1"] = "supports" if r["diff_mean_daily"] > 0 else "contradicts"
        # KF2
        r = out["H-KF2"][seg]
        if r["n_months"] < 3:
            v["H-KF2"] = "inconclusive"
        elif seg == "llm_in_sample":
            v["H-KF2"] = "supports" if (r["mean_ic"] > 0 and np.nan_to_num(r["ic_t_nw"]) > 2 and sig("H-KF2:ic")) else "contradicts"
        else:
            v["H-KF2"] = "supports" if r["mean_ic"] > 0 else "contradicts"
        vd[seg] = v
    return hb, vd


# ====================================================================== recording
DATA_NOTE = ("bybit 1h/4h bars via market_frame(funding=False, oi=False), as of {as_of}; universe L = trend_zoo_study.UNIVERSE "
             "(23 crypto perps), universe T = 16 TradFi perps (holdout-dominated, Mar-Jun 2026 starts); segments by decision date "
             "at 2026-07-01; 7.5 bps per side; Holm across the block's 12 primary tests per segment; "
             "src/ai_trader/research/breakout_cipher_kalman_study.py, research/reports/" + REPORT + ".json")


def record_all(out: dict) -> None:
    from ai_trader.research.hypotheses import record

    data = DATA_NOTE.format(as_of=out["as_of"])
    for seg in SEGS:
        v = out["verdicts"][seg]
        hb = out["holm"][seg]
        bo = {u: out["H-BO1"].get(u, {}).get(seg, {}) for u in ("L", "T")}
        m = {f"{u}_{h}": {k: bo[u][h].get(k) for k in ("n_pivot", "pivot_mean_net", "pivot_t_net", "placebo_mean_net", "diff_vs_placebo",
                                                        "t_vs_placebo", "diff_vs_mirror", "t_vs_mirror", "k_median")}
             for u in ("L", "T") for h in ("4h", "12h") if h in bo[u]}
        m["holm"] = {k: x for k, x in hb.items() if k.startswith("H-BO1")}
        record("H-BO1", seg, v["H-BO1"], m, data=data,
               note="first 1h close beyond R1/S1 of the prior UTC day; net signed return in sigma*sqrt(h) units; pivot minus time-shifted "
                    "placebo (same side, same sigma distance from prior close, random other day), SE clustered by date; null-type "
                    "hypothesis: supports = pivots not special" + ("; universe T is TradFi, holdout-dominated" if seg == "clean_holdout" else ""))
        mc = out["H-MC1"]["L"][seg]
        mm = {k: {x: r.get(x) for x in ("n_trades", "mean_net", "t_nw", "random_mean", "p_random_ge_real", "port_sharpe")} | {"dsr": r.get("dsr", {}).get("dsr")}
              for k, r in mc.items()}
        if seg == "clean_holdout":
            mm["best_in_sample_variant"] = v.get("H-MC1_best_in_sample_variant")
            mm["universe_T_robustness"] = {k: {x: r.get(x) for x in ("n_trades", "mean_net", "t_nw", "p_random_ge_real")} for k, r in out["H-MC1"]["T"][seg].items()}
        record("H-MC1", seg, v["H-MC1"], mm, data=data,
               note="LazyBear WaveTrend 9/12/4 dots at +/-53, VuManChu MF proxy; exit opposite cross or 12 bars; trade net risk units; "
                    "random entries matched in side and holding (1000 draws); DSR over 4 variants on daily portfolio series; null-type hypothesis")
        kf = out["H-KF1"][seg]
        record("H-KF1", seg, v["H-KF1"], {**kf, "signal_corr": out["H-KF1"]["signal_corr_adaptive_vs_constant"],
                                          "holm": {k: x for k, x in hb.items() if k.startswith("H-KF1")}}, data=data,
               note="smooth LLT on 4h log close; adaptive R_t = robust sigma^2 with Q = q * trailing-180d median R vs constant ratio q; "
                    "tanh(slope / trailing sd); strategy_pnl; paired daily diff (adaptive - constant), NW t; primary q = 1e-3")
        k2 = out["H-KF2"][seg]
        record("H-KF2", seg, v["H-KF2"], {**k2, "q_hat_median": out["H-KF2"]["q_hat_median"],
                                          "holm": {k: x for k, x in hb.items() if k.startswith("H-KF2")}}, data=data,
               note="monthly 180d concentrated-diffuse MLE of the full LLT, q_hat = var(zeta)/var(eta); Spearman IC of log q_hat vs next-30d "
                    "trend-zoo ensemble P&L (data/priors/trend_zoo_daily_4h.parquet); NW t over months")


def _json_default(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--record", action="store_true")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    out = run(a.workers)
    path = Path(a.out) if a.out else load().root / "research" / "reports" / f"{REPORT}.json"
    path.write_text(json.dumps(out, indent=1, default=_json_default), encoding="utf-8")
    print(json.dumps(out["verdicts"], indent=1))
    print(json.dumps(out["holm"], indent=1, default=_json_default))
    if a.record:
        record_all(out)


if __name__ == "__main__":
    main()
