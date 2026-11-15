"""Stage 3 of the BOS study: inference across timeframes (H-BOS1..5) -> data/research/bos/results.json.

Reads data/research/bos/agg/* (bos_regimes_analysis) and, for the stability tests, the trade caches.
Design frozen in the diary 2026-11-15 18:05 UTC and data/research/bos/design.json.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from itertools import combinations
from statistics import NormalDist

import numpy as np
import polars as pl

from ai_trader.research import bos_regimes_analysis as A
from ai_trader.research import bos_regimes_study as S
from ai_trader.research.trend_zoo_study import deflated_sharpe

ND = NormalDist()
ARMS = S.ARMS
MAIN = [ARMS.index(a) for a in S.MAIN_ARMS]
W = A.W
# common window for selection statistics (all timeframes live): 2022-01-03 .. last in-sample week
WK_LO = (int(datetime(2022, 1, 3, tzinfo=UTC).timestamp() * 1e6) - A.T0_US) // A.WEEK_US
WK_HI = A.W_CUT                                  # exclusive: weeks fully before the cutoff week
WK_HO = A.W_CUT                                   # holdout weeks start (signals split by ts; week 2026-06-29 mixes)
EVENTS = ["2021-05-19", "2021-11-10", "2022-05-09", "2022-06-13", "2022-11-08", "2023-03-10", "2024-01-10", "2024-04-20",
          "2024-08-05", "2024-11-05", "2025-02-21", "2025-04-07", "2025-10-10"]
EVENT_NAMES = ["China mining ban", "BTC top", "LUNA", "Celsius/3AC", "FTX", "SVB/USDC", "spot ETF", "halving", "carry unwind",
               "US election", "Bybit hack", "tariff crash", "liquidation cascade"]


def wk_date(w: int) -> str:
    return (datetime(2020, 1, 6, tzinfo=UTC) + timedelta(weeks=int(w))).date().isoformat()


def day_date(d: int) -> str:
    return (datetime(2020, 1, 1, tzinfo=UTC) + timedelta(days=int(d))).date().isoformat()


# ====================================================================== statistics helpers
def boot_means(ws: np.ndarray, wn: np.ndarray) -> np.ndarray:
    """(k, W) weekly sums / counts -> (B, k) week-cluster multiplier-bootstrap means."""
    num = A.BOOTW @ ws.T.astype(np.float64)
    den = A.BOOTW @ wn.T.astype(np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        return num / den


def bh(p: np.ndarray, q: float = 0.10) -> np.ndarray:
    p = np.asarray(p, float)
    n = len(p)
    if n == 0:
        return np.zeros(0, bool)
    o = np.argsort(p)
    thr = q * np.arange(1, n + 1) / n
    ok = p[o] <= thr
    k = np.flatnonzero(ok).max() + 1 if ok.any() else 0
    rej = np.zeros(n, bool)
    rej[o[:k]] = True
    return rej


def nw_t(x: np.ndarray) -> tuple[float, float, int]:
    from ai_trader.research.evaluate import newey_west_t

    return newey_west_t(np.asarray(x, float))


def sharpe(x: np.ndarray, ppy: float) -> float:
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) < 3 or x.std(ddof=1) == 0:
        return float("nan")
    return float(x.mean() / x.std(ddof=1) * np.sqrt(ppy))


def pbo(perf: np.ndarray, n_blocks: int = 16, chunk: int = 200) -> dict:
    """CSCV PBO (Bailey et al. 2017), identical to synthetic_robustness_study.pbo_cscv, chunked over splits."""
    x = np.where(np.isfinite(perf), perf, 0.0).astype(np.float64)
    t, n = x.shape
    L = t // n_blocks
    xb = x[: L * n_blocks].reshape(n_blocks, L, n)
    s1, s2 = xb.sum(axis=1), (xb**2).sum(axis=1)
    combos = np.array(list(combinations(range(n_blocks), n_blocks // 2)))
    lams, isb, oob, rel = [], [], [], []
    for a in range(0, len(combos), chunk):
        cb = combos[a:a + chunk]
        ind = np.zeros((len(cb), n_blocks))
        ind[np.arange(len(cb))[:, None], cb] = 1.0

        def sh(w):
            cnt = w.sum(axis=1, keepdims=True) * L
            m = (w @ s1) / cnt
            v = ((w @ s2) / cnt - m**2) * cnt / (cnt - 1)
            with np.errstate(divide="ignore", invalid="ignore"):
                return np.where(v > 0, m / np.sqrt(np.where(v > 0, v, 1.0)), -np.inf)

        si, so = sh(ind), sh(1.0 - ind)
        idx = np.arange(len(cb))
        best = np.argmax(si, axis=1)
        ob = so[idx, best]
        rank = (so < ob[:, None]).sum(axis=1) + 0.5 * ((so == ob[:, None]).sum(axis=1) - 1) + 1
        om = rank / (n + 1)
        lams.append(np.log(om / (1 - om)))
        isb.append(si[idx, best])
        oob.append(ob)
        rel.append(om)
    lam = np.concatenate(lams)
    isb_, oob_ = np.concatenate(isb), np.concatenate(oob)
    ok = np.isfinite(isb_) & np.isfinite(oob_)
    slope = float(np.polyfit(isb_[ok], oob_[ok], 1)[0]) if ok.sum() > 2 else float("nan")
    return {"pbo": float((lam <= 0).mean()), "median_logit": float(np.median(lam)), "is_winner_mean_oos_rank": float(np.mean(np.concatenate(rel))),
            "slope": slope, "n_splits": int(len(lam)), "n_configs": int(n), "n_blocks": n_blocks, "rows": int(t)}


def dsr(x: np.ndarray, sr_trials: np.ndarray, n_trials: int, ppy: float) -> dict:
    d = deflated_sharpe(np.asarray(x, float), np.asarray(sr_trials, float), n_trials)
    k = np.sqrt(ppy / 365)
    d["sr_ann"] = d["sr_ann"] * k
    d["sr0_ann"] = d["sr0_ann"] * k
    return d


# ====================================================================== cells
def load_cells() -> tuple[pl.DataFrame, dict]:
    rows, mats = [], {}
    for tf in A.TFS:
        p = A.agg_dir() / f"{tf}.npz"
        if not p.exists():
            continue
        z = np.load(p)
        st, WS, WN = z["stats"], z["WS"], z["WN"]
        mats[tf] = (WS, WN)
        ci = np.flatnonzero(np.isfinite(st[:, 7]))
        sg = ci % 2
        rid = (ci // 2) % A.NREG
        ai = (ci // (2 * A.NREG)) % len(ARMS)
        vl = ci // (2 * A.NREG * len(ARMS))
        n, sm, nw, sw, sl, nl, top, nsig = st[ci].T
        weeks = (WN[ci] > 0).sum(axis=1)
        rows.append(pl.DataFrame({
            "tf": tf, "ci": ci, "vid": (A.TFS.index(tf) * 60 + vl).astype(np.int32), "struct": (vl // 12).astype(np.int8),
            "sig": ((vl % 12) // 4).astype(np.int8), "exit": (vl % 4).astype(np.int8), "arm": ai.astype(np.int8), "rid": rid.astype(np.int8),
            "seg": sg.astype(np.int8), "n": n, "sum": sm, "n_sig": nsig, "weeks": weeks, "n_win": nw, "sum_win": sw, "sum_loss": sl,
            "n_loss": nl, "top5": top}))
    df = pl.concat(rows).with_columns(
        (pl.col("sum") / pl.col("n")).alias("mean"),
        (pl.col("sum") / pl.col("n_sig")).alias("mean_sig"),
        (pl.col("n_win") / pl.col("n")).alias("hit"),
        ((pl.col("sum_win") / pl.col("n_win")) / (pl.col("sum_loss") / pl.col("n_loss")).abs()).alias("payoff"),
        pl.when(pl.col("sum") > 0).then(pl.col("top5") / pl.col("sum")).otherwise(None).alias("top5_share"),
        ((pl.col("n") >= A.MIN_N) & (pl.col("weeks") >= A.MIN_WEEKS)).alias("tested"),
    )
    # bootstrap SE per cell
    se = np.full(df.height, np.nan)
    for tf, (WS, WN) in mats.items():
        m = (df["tf"] == tf).to_numpy() & df["tested"].to_numpy()
        ix = np.flatnonzero(m)
        cis = df["ci"].to_numpy()[ix]
        for a in range(0, len(ix), 2000):
            bm = boot_means(WS[cis[a:a + 2000]], WN[cis[a:a + 2000]])
            se[ix[a:a + 2000]] = np.nanstd(bm, axis=0, ddof=1)
    df = df.with_columns(pl.Series("se", se)).with_columns((pl.col("mean") / pl.col("se")).alias("z"))
    df = df.with_columns(
        pl.col("z").map_elements(lambda z: 1 - ND.cdf(z), return_dtype=pl.Float64).alias("p_pos"),
        pl.col("z").map_elements(lambda z: ND.cdf(z), return_dtype=pl.Float64).alias("p_neg"),
        pl.col("vid").map_elements(A.variant_name, return_dtype=pl.Utf8).alias("variant"),
        pl.col("arm").map_elements(lambda a: S.ARM_NAME[ARMS[a]], return_dtype=pl.Utf8).alias("arm_name"),
        pl.col("rid").map_elements(lambda r: A.REG_NAMES[r], return_dtype=pl.Utf8).alias("regime"),
    )
    return df, mats


def label_cells(df: pl.DataFrame) -> pl.DataFrame:
    return df.with_columns(
        pl.when(~pl.col("tested")).then(pl.lit("untested"))
        .when((pl.col("mean") < 0) & (pl.col("z") < -2)).then(pl.lit("detrimental"))
        .when((pl.col("mean") > 0) & (pl.col("top5_share") > 1)).then(pl.lit("explosive"))
        .when((pl.col("mean") > 0) & (pl.col("z") > 2)).then(pl.lit("power"))
        .otherwise(pl.lit("neutral")).alias("label"))


def weekly(mats: dict, tf: str, ci: int, lo: int, hi: int) -> np.ndarray:
    return mats[tf][0][ci, lo:hi].astype(np.float64)


# ====================================================================== H-BOS1 + selection steps
def h1(df: pl.DataFrame, mats: dict) -> dict:
    out = {}
    fam = df.filter(pl.col("arm").is_in(MAIN))
    for sg, seg in enumerate(A.SEGS):
        t = fam.filter((pl.col("seg") == sg) & pl.col("tested"))
        rp = bh(t["p_pos"].to_numpy())
        rn = bh(t["p_neg"].to_numpy())
        out[seg] = {"tested": t.height, "bh_pos": int(rp.sum()), "bh_neg": int(rn.sum()),
                    "share_mean_pos": float((t["mean"] > 0).mean()) if t.height else None}
        if sg == 0:
            surv = t.filter(pl.Series(rp)).select("tf", "ci", "vid", "arm", "rid", "variant", "arm_name", "regime", "n", "weeks", "mean", "z", "p_pos", "hit", "payoff", "top5_share")
            neg = t.filter(pl.Series(rn)).select("tf", "ci", "vid", "arm", "rid", "variant", "arm_name", "regime", "n", "weeks", "mean", "z", "p_neg")
    ho = fam.filter(pl.col("seg") == 1).select("vid", "arm", "rid", pl.col("n").alias("n_ho"), pl.col("mean").alias("mean_ho"), pl.col("z").alias("z_ho"),
                                                pl.col("weeks").alias("weeks_ho"))
    surv = surv.join(ho, on=["vid", "arm", "rid"], how="left").sort("z", descending=True)
    neg = neg.join(ho, on=["vid", "arm", "rid"], how="left").sort("z")
    # base rate: holdout sign of all tested in-sample cells with positive in-sample mean
    base = fam.filter((pl.col("seg") == 0) & pl.col("tested") & (pl.col("mean") > 0)).select("vid", "arm", "rid").join(ho, on=["vid", "arm", "rid"], how="inner")
    out["survivors"] = surv.to_dicts()
    out["negative_survivors_n"] = neg.height
    out["negative_survivors_holdout_neg_share"] = float((neg["mean_ho"].drop_nulls() < 0).mean()) if neg.height else None
    out["negative_top"] = neg.head(30).to_dicts()
    nho = surv.filter(pl.col("n_ho").is_not_null())
    out["survivors_with_holdout"] = nho.height
    out["survivors_holdout_pos"] = int((nho["mean_ho"] > 0).sum()) if nho.height else 0
    out["survivors_holdout_pos_n10"] = int(((nho["mean_ho"] > 0) & (nho["n_ho"] >= 10)).sum()) if nho.height else 0
    out["base_rate_holdout_pos"] = float((base["mean_ho"] > 0).mean()) if base.height else None
    # selection step 1: best in-sample cell, weekly series over the common window
    t = fam.filter((pl.col("seg") == 0) & pl.col("tested"))
    best = t.sort("z", descending=True).row(0, named=True)
    X = np.stack([weekly(mats, r["tf"], r["ci"], WK_LO, WK_HI) for r in t.select("tf", "ci").iter_rows(named=True)], axis=1)
    srs = X.mean(axis=0) / np.where(X.std(axis=0, ddof=1) > 0, X.std(axis=0, ddof=1), np.nan)
    xb = weekly(mats, best["tf"], best["ci"], WK_LO, WK_HI)
    out["step1_best_cell"] = {"variant": best["variant"], "arm": best["arm_name"], "regime": best["regime"], "mean": best["mean"], "z": best["z"],
                              "n": best["n"], "dsr": dsr(xb, srs[np.isfinite(srs)], int(np.isfinite(srs).sum()), 52),
                              "pbo": pbo(X), "holdout": ho.filter((pl.col("vid") == best["vid"]) & (pl.col("arm") == best["arm"]) & (pl.col("rid") == best["rid"])).to_dicts()}
    del X
    # selection step 2: static in-sample-best variant x arm (ALL regime), weekly Sharpe on the common window
    t2 = fam.filter((pl.col("seg") == 0) & (pl.col("rid") == 0) & pl.col("tested"))
    X2 = np.stack([weekly(mats, r["tf"], r["ci"], WK_LO, WK_HI) for r in t2.select("tf", "ci").iter_rows(named=True)], axis=1)
    sd2 = X2.std(axis=0, ddof=1)
    sr2 = X2.mean(axis=0) / np.where(sd2 > 0, sd2, np.nan)
    j = int(np.nanargmax(sr2))
    r2 = t2.row(j, named=True)
    out["step2_static_best"] = {"variant": r2["variant"], "arm": r2["arm_name"], "vid": r2["vid"], "arm_id": r2["arm"], "sr_week_ann": float(sr2[j] * np.sqrt(52)),
                                "mean": r2["mean"], "n": r2["n"], "dsr": dsr(X2[:, j], sr2[np.isfinite(sr2)], int(np.isfinite(sr2).sum()), 52),
                                "pbo": pbo(X2), "holdout": ho.filter((pl.col("vid") == r2["vid"]) & (pl.col("arm") == r2["arm"]) & (pl.col("rid") == 0)).to_dicts(),
                                "grid_sr_quantiles": [float(np.nanquantile(sr2, q) * np.sqrt(52)) for q in (0, 0.1, 0.5, 0.9, 1.0)],
                                "share_variants_mean_pos": float((t2["mean"] > 0).mean())}
    return out


# ====================================================================== H-BOS2 family pools + per-regime summary
def family(mats_unused=None) -> dict:
    res = {}
    for cls in ("intraday", "swing"):
        tfs = [tf for tf in A.TFS if A.CLASS[tf] == cls and (A.agg_dir() / f"{tf}.npz").exists()]
        if not tfs:
            continue
        hc = hs = fws = fwn = None
        for tf in tfs:
            z = np.load(A.agg_dir() / f"{tf}.npz")
            hc = z["hist_c"] if hc is None else hc + z["hist_c"]
            hs = z["hist_s"] if hs is None else hs + z["hist_s"]
            fws = z["fam_ws"] if fws is None else fws + z["fam_ws"]
            fwn = z["fam_wn"] if fwn is None else fwn + z["fam_wn"]
        rc = {}
        for ai, a in enumerate(ARMS):
            for rid in range(A.NREG):
                for sg in (0, 1):
                    c, s_ = hc[ai, rid, sg], hs[ai, rid, sg]
                    n = c.sum()
                    if n < A.MIN_N:
                        continue
                    tot = s_.sum()
                    k = 0.05 * n
                    cc = np.cumsum(c[::-1])
                    b = np.searchsorted(cc, k)            # bins from the top until 5% of trades
                    full = s_[::-1][:b].sum()
                    prev = cc[b - 1] if b > 0 else 0
                    part = s_[::-1][b] * (k - prev) / c[::-1][b] if c[::-1][b] > 0 else 0
                    top = full + part
                    bm = boot_means(fws[ai, rid, sg][None, :], fwn[ai, rid, sg][None, :])[:, 0]
                    se = np.nanstd(bm, ddof=1)
                    mean = tot / n
                    rc[f"{a}|{A.REG_NAMES[rid]}|{sg}"] = {"n": float(n), "mean": float(mean), "z": float(mean / se) if se > 0 else None,
                                                          "top5_share": float(top / tot) if tot > 0 else None, "top5_sum_per_trade": float(top / n),
                                                          "rest_mean": float((tot - top) / (n - k)), "weeks": int((fwn[ai, rid, sg] > 0).sum())}
        res[cls] = {"cells": rc, "fws": fws, "fwn": fwn}
    return res


def h2(fam: dict) -> dict:
    out = {}
    for cls, d in fam.items():
        rc = d["cells"]
        o = {}
        for a in ARMS:
            regs = [r for r in A.REG_NAMES[1:] if f"{a}|{r}|0" in rc]
            A_cells = {sg: [r for r in regs if f"{a}|{r}|{sg}" in rc and rc[f"{a}|{r}|{sg}"]["mean"] > 0 and (rc[f"{a}|{r}|{sg}"]["top5_share"] or 0) > 1] for sg in (0, 1)}
            B_cells = {sg: [r for r in regs if f"{a}|{r}|{sg}" in rc and rc[f"{a}|{r}|{sg}"]["mean"] < 0 and (rc[f"{a}|{r}|{sg}"]["z"] or 0) < -2] for sg in (0, 1)}
            strictA = sorted(set(A_cells[0]) & set(A_cells[1]))
            strictB = sorted(set(B_cells[0]) & set(B_cells[1]))
            # best minus worst in-sample regime (excluding ALL), paired week bootstrap
            ins = [(r, rc[f"{a}|{r}|0"]["mean"]) for r in regs]
            diff = {}
            if ins:
                rb_, rw_ = max(ins, key=lambda x: x[1])[0], min(ins, key=lambda x: x[1])[0]
                for sg in (0, 1):
                    ib, iw = A.REG_NAMES.index(rb_), A.REG_NAMES.index(rw_)
                    ai = ARMS.index(a)
                    mb = boot_means(d["fws"][ai, ib, sg][None], d["fwn"][ai, ib, sg][None])[:, 0]
                    mw = boot_means(d["fws"][ai, iw, sg][None], d["fwn"][ai, iw, sg][None])[:, 0]
                    kb, kw = f"{a}|{rb_}|{sg}", f"{a}|{rw_}|{sg}"
                    if kb in rc and kw in rc:
                        dd = rc[kb]["mean"] - rc[kw]["mean"]
                        sd = np.nanstd(mb - mw, ddof=1)
                        diff[A.SEGS[sg]] = {"best": rb_, "worst": rw_, "diff": dd, "z": dd / sd if sd > 0 else None}
            o[a] = {"explosive": {A.SEGS[s]: A_cells[s] for s in (0, 1)}, "detrimental_t2": {A.SEGS[s]: B_cells[s] for s in (0, 1)},
                    "strict_explosive_both": strictA, "strict_detrimental_both": strictB,
                    "strict": bool(strictA and strictB), "loose_in_sample": bool(A_cells[0] and B_cells[0]),
                    "loose_holdout": bool(A_cells[1] and B_cells[1]), "best_minus_worst": diff}
        out[cls] = o
    return out


# ====================================================================== H-BOS3 stability
def trade_series(tf: str, vid: int, arm: int, rid: int) -> pl.DataFrame:
    r = vid % 60
    s, g, e = r // 12, (r % 12) // 4, r % 4
    a = ARMS[arm]
    ex = S.EXITS[e]
    xc = f"{A.X_COL[a]}_{ex}_x"
    cols = ["struct", "sig_ts", "side", "choch", "mom", "cell", "vt", "st", f"{a}_{ex}", xc] + ([A.FILL[a]] if a in A.FILL else [])
    df = A._keys(A.read_cols(tf, cols), xc).rename({f"{a}_{ex}": "v"}).filter(pl.col("struct") == s)
    if g == 1:
        df = df.filter(pl.col("choch"))
    if g == 2:
        df = df.filter(pl.col("mom"))
    if rid > 0:
        col = "rcs" if rid <= 18 else ("rvt" if rid <= 21 else "rst")
        df = df.filter(pl.col(col) == rid)
    return df.filter((pl.col("seg") >= 0) & pl.col("v").is_not_nan()).sort("sig_ts").select("sig_ts", "v", "seg", "wk")


def bai_perron(x: np.ndarray, max_breaks: int = 3, trim: float = 0.15, nblk: int = 300) -> dict:
    """Break-in-mean by dynamic programming on consecutive blocks (block means weighted by size); sup-F for one break
    (Andrews 5% critical 8.85) and BIC for the number of breaks. Returns break positions as trade indices."""
    n = len(x)
    bs = max(1, n // nblk)
    m = n // bs
    xb = x[: m * bs].reshape(m, bs)
    s1, s2 = xb.sum(axis=1), (xb**2).sum(axis=1)
    c1, c2 = np.r_[0, np.cumsum(s1)], np.r_[0, np.cumsum(s2)]

    def ssr(i, j):
        k = (j - i) * bs
        return c2[j] - c2[i] - (c1[j] - c1[i]) ** 2 / k

    h = max(1, int(np.ceil(trim * m)))
    INF = np.inf
    cost = {0: np.array([ssr(0, j) if j >= h else INF for j in range(m + 1)])}
    arg = {}
    for b in range(1, max_breaks + 1):
        cb = np.full(m + 1, INF)
        ab = np.zeros(m + 1, int)
        for j in range((b + 1) * h, m + 1):
            best, bi = INF, 0
            for i in range(b * h, j - h + 1):
                v = cost[b - 1][i] + ssr(i, j)
                if v < best:
                    best, bi = v, i
            cb[j], ab[j] = best, bi
        cost[b], arg[b] = cb, ab
    ssr0 = cost[0][m]
    nn = m * bs
    bic = {b: nn * np.log(cost[b][m] / nn) + (2 * b + 1) * np.log(nn) for b in range(max_breaks + 1) if np.isfinite(cost[b][m])}
    kbest = min(bic, key=bic.get)
    supf = (ssr0 - cost[1][m]) / (cost[1][m] / (nn - 2)) if np.isfinite(cost[1][m]) else float("nan")
    brk = []
    j = m
    for b in range(kbest, 0, -1):
        i = arg[b][j]
        brk.append(i * bs)
        j = i
    return {"supF": float(supf), "break_5pct": bool(supf > 8.85), "n_breaks_bic": int(kbest), "breaks": sorted(brk)}


def btc_regime_shifts() -> list[str]:
    reg = S.regime_table("BTCUSDT")
    c = reg["cell"].to_numpy()
    up = np.isin(c, [0, 1, 2]).astype(float)
    dn = np.isin(c, [6, 7, 8]).astype(float)
    w = 90 * 6
    net = pl.Series(up - dn).rolling_mean(w).to_numpy()
    sgn = np.sign(net)
    ts = reg["close_ts"].to_list()
    out, last = [], None
    for i in range(1, len(sgn)):
        if np.isfinite(net[i]) and np.isfinite(net[i - 1]) and sgn[i] != sgn[i - 1] and sgn[i] != 0:
            if last is None or (ts[i] - last).days >= 60:
                out.append(ts[i].date().isoformat())
                last = ts[i]
    return out


def h3(df: pl.DataFrame, h1o: dict) -> dict:
    surv = h1o["survivors"]
    if len(surv) >= 10:
        top = [(r["tf"], r["vid"], r["arm"], r["rid"], r["variant"], r["arm_name"], r["regime"]) for r in surv[:25]]
        basis = "BH survivors (top 25 by z)"
    else:
        t = df.filter((pl.col("seg") == 0) & pl.col("tested") & pl.col("arm").is_in(MAIN)).sort("z", descending=True).head(10)
        top = [(r["tf"], r["vid"], r["arm"], r["rid"], r["variant"], r["arm_name"], r["regime"]) for r in t.iter_rows(named=True)]
        basis = "top 10 tested in-sample cells by z (fewer than 10 BH survivors)"
    shifts = btc_regime_shifts()
    ev = [datetime.fromisoformat(e).date() for e in EVENTS]
    sh = [datetime.fromisoformat(e).date() for e in shifts]
    cells = []
    for tf, vid, arm, rid, vn, an, rn in top:
        ts = trade_series(tf, vid, arm, rid)
        x = ts["v"].to_numpy().astype(float)
        dts = ts["sig_ts"].to_numpy()
        ins = ts["seg"].to_numpy() == 0
        mu_in = float(x[ins].mean())
        dt = pl.Series(dts).cast(pl.Datetime("us", "UTC"))
        hy = (dt.dt.year() * 10 + (dt.dt.month() > 6).cast(pl.Int32) + 1).to_numpy()
        halves = []
        for h_ in np.unique(hy):
            m = (hy == h_) & ins
            if m.sum() >= 10:
                halves.append({"half": f"{h_ // 10}H{h_ % 10}", "n": int(m.sum()), "mean": float(x[m].mean())})
        flips = sum(1 for h_ in halves if np.sign(h_["mean"]) != np.sign(mu_in))
        flip_share = flips / len(halves) if halves else float("nan")
        from statsmodels.stats.diagnostic import breaks_cusumolsresid

        xi = x[ins]
        cus = breaks_cusumolsresid(xi - xi.mean(), ddof=1)
        bp = bai_perron(xi)
        lo, hi = np.quantile(xi, [0.01, 0.99])
        bpw = bai_perron(np.clip(xi, lo, hi))
        dti = dt.filter(pl.Series(ins))
        bdates = [dti[int(b)].date() for b in bp["breaks"]]
        bpf = bai_perron(x)
        bdates_full = [dt[int(b)].date() for b in bpf["breaks"]]
        align = [{"break": b.isoformat(), "event": next((f"{EVENT_NAMES[i]} {EVENTS[i]}" for i, e in enumerate(ev) if abs((b - e).days) <= 30), None),
                  "btc_regime_shift": next((s.isoformat() for s in sh if abs((b - s).days) <= 30), None)} for b in bdates_full]
        ho = x[~ins]
        cells.append({"tf": tf, "vid": vid, "arm_id": arm, "rid": rid, "variant": vn, "arm": an, "regime": rn, "trades_ts": None, "n_in": int(ins.sum()), "n_ho": int((~ins).sum()), "mean_in": mu_in,
                      "mean_ho": float(ho.mean()) if len(ho) else None, "halves": halves, "flip_share": flip_share, "flips_ge_third": bool(flip_share >= 1 / 3),
                      "cusum_stat": float(cus[0]), "cusum_p": float(cus[1]), "bai_perron_in": {**bp, "break_dates": [b.isoformat() for b in bdates]},
                      "bai_perron_in_winsor": {k: bpw[k] for k in ("supF", "break_5pct", "n_breaks_bic")},
                      "bai_perron_full": {**bpf, "break_dates": [b.isoformat() for b in bdates_full]}, "alignment": align,
                      "break_5pct": bool(cus[1] < 0.05 or bp["break_5pct"]),
                      "both": bool(flip_share >= 1 / 3 and (cus[1] < 0.05 or bp["break_5pct"])),
                      "holdout_sign_flip": bool(len(ho) >= 10 and np.sign(ho.mean()) != np.sign(mu_in))})
    # chance rate of alignment: share of days 2021-2026 within +-30 days of an event / regime shift
    days = [datetime(2021, 1, 1).date() + timedelta(days=i) for i in range(0, (datetime(2026, 11, 15) - datetime(2021, 1, 1)).days)]
    ch_ev = float(np.mean([any(abs((d - e).days) <= 30 for e in ev) for d in days]))
    ch_sh = float(np.mean([any(abs((d - e).days) <= 30 for e in sh) for d in days]))
    allb = [a for c in cells for a in c["alignment"]]
    return {"basis": basis, "cells": cells, "majority_both": bool(sum(c["both"] for c in cells) > len(cells) / 2),
            "n_both": sum(c["both"] for c in cells), "n_flip": sum(c["flips_ge_third"] for c in cells), "n_break": sum(c["break_5pct"] for c in cells),
            "n_cells": len(cells), "holdout_sign_flips": sum(c["holdout_sign_flip"] for c in cells),
            "holdout_n10": sum(1 for c in cells if c["n_ho"] >= 10), "btc_regime_shifts": shifts,
            "breaks_total": len(allb), "breaks_near_event": sum(1 for a in allb if a["event"]), "breaks_near_shift": sum(1 for a in allb if a["btc_regime_shift"]),
            "chance_event": ch_ev, "chance_shift": ch_sh}


# ====================================================================== H-BOS4 allocator
PHI = (0.90, 0.95, 0.98, 0.99, 1.00)
KAPPA0, Z_ON, BURN_WK, PURGE_D = 20.0, 0.84, 26, 14
NV = 240


def _dix(y, m, d):
    return (datetime(y, m, d, tzinfo=UTC) - datetime(2020, 1, 1, tzinfo=UTC)).days


def allocator(arm: str = "p") -> dict:
    fs = [A.agg_dir() / f"{tf}_alloc_{arm}.parquet" for tf in A.TFS]
    d = pl.concat([pl.read_parquet(f) for f in fs if f.exists()])
    vid, rcs, wk, xday, xwk, v = (d[c].to_numpy() for c in ("vid", "rcs", "wk", "xday", "xwk", "v"))
    v = v.astype(np.float64)
    cs = rcs.astype(int) - 1
    has = cs >= 0
    ND_ = A.ND_DAYS
    C = np.zeros((NV, 18, W))
    Ssum = np.zeros((NV, 18, W))
    np.add.at(C, (vid[has], cs[has], xwk[has]), 1.0)
    np.add.at(Ssum, (vid[has], cs[has], xwk[has]), v[has])
    first_day = int(xday.min())
    VB = np.zeros((NV, ND_))
    np.add.at(VB, (vid, xday), v)
    VC = np.zeros((NV, ND_))
    np.add.at(VC, (vid, xday), 1.0)
    vcnt = np.cumsum(VC, axis=1)
    del VC

    def online(phi: float, sig: np.ndarray) -> np.ndarray:
        n = np.zeros((NV, 18))
        s = np.zeros((NV, 18))
        on = np.zeros((W, NV, 18), bool)
        for w in range(W):
            post = s / (sig[:, None] * np.sqrt(n + KAPPA0))
            on[w] = post >= Z_ON
            n = phi * n + C[:, :, w]
            s = phi * s + Ssum[:, :, w]
        return on

    def book(on: np.ndarray | None, sel: np.ndarray | None = None) -> np.ndarray:
        if on is None:
            take = np.ones(len(v), bool) if sel is None else sel
        else:
            take = np.zeros(len(v), bool)
            take[has] = on[wk[has], vid[has], cs[has]]
        return np.bincount(xday[take], weights=v[take], minlength=ND_)[:ND_] / NV

    def var_sd(end_day: int) -> np.ndarray:
        m = xday < end_day
        cnt = np.bincount(vid[m], minlength=NV)
        s1 = np.bincount(vid[m], weights=v[m], minlength=NV)
        s2 = np.bincount(vid[m], weights=v[m] ** 2, minlength=NV)
        with np.errstate(invalid="ignore", divide="ignore"):
            sd = np.sqrt((s2 - s1**2 / cnt) / (cnt - 1))
        return np.where((cnt > 30) & (sd > 0), sd, np.inf)

    blocks = [(_dix(y, m, 1), _dix(y + (m == 7), 1 if m == 7 else 7, 1)) for y in (2023, 2024, 2025, 2026) for m in (1, 7) if not (y == 2026 and m == 7)]
    blocks.append((_dix(2026, 7, 1), ND_))
    ew = book(None)
    res_blocks = []
    series = {k: np.zeros(ND_) for k in ("alloc", "static", "ew", *[f"phi{p}" for p in PHI])}
    raw_phi = {p: np.zeros(ND_) for p in PHI}
    on_share = []
    for b0, b1 in blocks:
        tr_end = b0 - PURGE_D
        sig = var_sd(tr_end)
        score_lo = first_day + BURN_WK * 7
        books = {}
        for p in PHI:
            on = online(p, sig)
            books[p] = (book(on), on)
        sc = {p: sharpe(books[p][0][score_lo:tr_end], 365) for p in PHI}
        pstar = max(sc, key=lambda p: -np.inf if not np.isfinite(sc[p]) else sc[p])
        # static best variant: best training daily Sharpe (all regimes)
        srv = np.full(NV, -np.inf)
        for vv in np.flatnonzero(vcnt[:, max(tr_end - 1, 0)] >= 100):
            srv[vv] = sharpe(VB[vv, score_lo:tr_end], 365)
        vbest = int(np.argmax(srv))
        st_book = VB[vbest]
        tr = slice(score_lo, tr_end)

        def scl(x):
            s_ = np.std(x[tr], ddof=1)
            return x[b0:b1] / s_ if s_ > 0 else x[b0:b1] * np.nan

        series["alloc"][b0:b1] = scl(books[pstar][0])
        series["static"][b0:b1] = scl(st_book)
        series["ew"][b0:b1] = scl(ew)
        for p in PHI:
            series[f"phi{p}"][b0:b1] = scl(books[p][0])
            raw_phi[p][b0:b1] = books[p][0][b0:b1]
        onw = books[pstar][1][(b0 // 7):(b1 // 7)]
        on_share.append(float(onw.mean()))
        res_blocks.append({"block": f"{day_date(b0)}..{day_date(b1 - 1)}", "phi": pstar, "inner_sharpe": {str(k): v_ for k, v_ in sc.items()},
                           "static_variant": A.variant_name(vbest), "static_train_sr": float(srv[vbest]), "on_share": on_share[-1],
                           "sr_alloc": sharpe(books[pstar][0][b0:b1], 365), "sr_static": sharpe(st_book[b0:b1], 365), "sr_ew": sharpe(ew[b0:b1], 365)})
    ins = slice(blocks[0][0], blocks[-1][0])
    ho = slice(blocks[-1][0], ND_)
    out = {"arm": arm, "blocks": res_blocks}
    for nm, sl in (("llm_in_sample", ins), ("clean_holdout", ho)):
        a_, s_, e_ = series["alloc"][sl], series["static"][sl], series["ew"][sl]
        o = {"sr_alloc": sharpe(a_, 365), "sr_static": sharpe(s_, 365), "sr_ew": sharpe(e_, 365), "days": int(len(a_))}
        for k, base in (("vs_static", s_), ("vs_ew", e_)):
            m_, t_, n_ = nw_t(a_ - base)
            o[k] = {"mean_diff": m_, "t": t_, "n": n_}
        for p in PHI:
            o[f"sr_phi{p}"] = sharpe(series[f"phi{p}"][sl], 365)
        out[nm] = o
    # DSR of the WF allocator over the phi grid and PBO over the 5 fixed-phi books (in-sample WF days)
    srt = np.array([sharpe(series[f"phi{p}"][ins], 365) / np.sqrt(365) for p in PHI])
    out["dsr_in_sample"] = dsr(series["alloc"][ins], srt, len(PHI), 365)
    out["pbo_phi"] = pbo(np.stack([raw_phi[p][ins] for p in PHI], axis=1))
    out["equity"] = {k: np.nancumsum(np.nan_to_num(series[k][blocks[0][0]:])).tolist() for k in ("alloc", "static", "ew")}
    out["equity_start"] = day_date(blocks[0][0])
    out["holdout_start_index"] = blocks[-1][0] - blocks[0][0]
    return out


# ====================================================================== H-BOS5 execution
def h5() -> dict:
    out = {}
    for cls in ("intraday", "swing"):
        tfs = [tf for tf in A.TFS if A.CLASS[tf] == cls and (A.agg_dir() / f"{tf}_exec.npz").exists()]
        if not tfs:
            continue
        acc = {}
        for tf in tfs:
            z = np.load(A.agg_dir() / f"{tf}_exec.npz")
            for k in z.files:
                g, sg, nm = k.split("|")
                kk = (int(sg), nm)
                acc[kk] = z[k] if kk not in acc else acc[kk] + z[k]
                acc[(tf, int(sg), nm)] = z[k] if (tf, int(sg), nm) not in acc else acc[(tf, int(sg), nm)] + z[k]
        o = {}
        for scope in [None, *tfs]:
            for sg in (0, 1):
                seg = A.SEGS[sg]
                key = seg if scope is None else f"{scope}|{seg}"
                r = {}
                for nm in ("m", "p", "r", "rb", "r24", "r-m", "r-p", "p-m", "rb-m", "rb-p", "r24-m", "r24-p"):
                    kk = (sg, nm) if scope is None else (scope, sg, nm)
                    if kk not in acc:
                        continue
                    ws, wn = acc[kk]
                    mean = ws.sum() / wn.sum()
                    bm = boot_means(ws[None], wn[None])[:, 0]
                    lo, hi = np.nanquantile(bm, [0.025, 0.975])
                    r[nm] = {"mean": float(mean), "lo": float(lo), "hi": float(hi), "n": float(wn.sum())}
                kk = (sg, "mw") if scope is None else (scope, sg, "mw")
                if kk in acc:
                    c = acc[kk]
                    r["missed"] = {"signals": float(c[0]), "missed_share": float(c[5] / c[0]), "winners": float(c[1]),
                                   "winners_missed_share": float(c[2] / c[1]), "top5": float(c[3]), "top5_missed_share": float(c[4] / c[3]),
                                   "top5_R_missed_share": float(c[7] / c[6]) if c[6] else None}
                o[key] = r
        out[cls] = o
    return out


def h5_verdicts(e: dict) -> dict:
    v = {}
    for sg, seg in enumerate(A.SEGS):
        sup, contra = [], []
        for cls, o in e.items():
            r = o.get(seg, {})
            if "r-m" not in r:
                continue
            up = r["r-m"]["lo"] > 0 and r["r-p"]["lo"] > 0
            dn = r["r-m"]["hi"] < 0 or r["r-p"]["hi"] < 0
            sup.append(up)
            contra.append(dn)
        v[seg] = "supports" if any(sup) else ("contradicts" if contra and all(contra) else "inconclusive")
    return v


# ====================================================================== exec-arm books + per-regime summary
def arm_books(mats: dict, df: pl.DataFrame) -> dict:
    """Selection step 4: equal-weight book of all variants per arm and class (ALL regime), weekly; DSR n=3, PBO."""
    out = {}
    for cls in ("intraday", "swing"):
        cols, names = [], []
        for ai in MAIN:
            x = np.zeros(WK_HI - WK_LO)
            hx = np.zeros(W - WK_HO)
            got = False
            for tf in A.TFS:
                if A.CLASS[tf] != cls or tf not in mats:
                    continue
                t = df.filter((pl.col("tf") == tf) & (pl.col("arm") == ai) & (pl.col("rid") == 0))
                for r in t.select("ci", "seg").iter_rows(named=True):
                    if r["seg"] == 0:
                        x += weekly(mats, tf, r["ci"], WK_LO, WK_HI)
                    else:
                        hx += weekly(mats, tf, r["ci"], WK_HO, W)
                    got = True
            if got:
                cols.append((x, hx))
                names.append(S.ARM_NAME[ARMS[ai]])
        if not cols:
            continue
        X = np.stack([c[0] for c in cols], axis=1)
        sr = X.mean(axis=0) / X.std(axis=0, ddof=1)
        j = int(np.argmax(sr))
        out[cls] = {"arms": names, "sr_week_ann_in": (sr * np.sqrt(52)).tolist(),
                    "sr_week_ann_ho": [sharpe(c[1], 52) for c in cols], "best": names[j],
                    "dsr": dsr(X[:, j], sr, len(names), 52), "pbo": pbo(X)}
    return out


def regime_summary(df: pl.DataFrame) -> dict:
    """Per class x regime: counts of variant cells by label (market/pegged/retest), in-sample, with holdout sign agreement."""
    out = {}
    lab = label_cells(df)
    ins = lab.filter(pl.col("seg") == 0)
    ho = lab.filter(pl.col("seg") == 1).select("vid", "arm", "rid", pl.col("mean").alias("mean_ho"), pl.col("n").alias("n_ho"))
    j = ins.join(ho, on=["vid", "arm", "rid"], how="left").with_columns(pl.col("tf").replace_strict(A.CLASS).alias("cls"))
    g = (j.filter(pl.col("arm").is_in(MAIN)).group_by("cls", "regime", "label")
         .agg(pl.len().alias("cells"), (pl.col("mean_ho") > 0).sum().alias("ho_pos"), pl.col("mean_ho").is_not_null().sum().alias("ho_n"),
              pl.col("mean").median().alias("median_mean")))
    for r in g.iter_rows(named=True):
        out.setdefault(r["cls"], {}).setdefault(r["regime"], {})[r["label"]] = {"cells": r["cells"], "ho_pos": r["ho_pos"], "ho_n": r["ho_n"],
                                                                              "median_mean_in": r["median_mean"]}
    return out


def tradfi_check(surv: list[dict]) -> list[dict]:
    """Robustness: the in-sample survivors (swing only) on the TradFi perps (1h/4h caches with suffix _T)."""
    out = []
    for r in surv[:25]:
        if r["tf"] not in ("1h", "4h") or not list((S.root() / "cache" / f"{r['tf']}_T").glob("*.parquet")):
            continue
        tfT = f"{r['tf']}_T"
        old = A.files
        A.files = lambda tf, _o=old: sorted((S.root() / "cache" / tfT).glob("*.parquet")) if tf == r["tf"] else _o(tf)
        try:
            ts = trade_series(r["tf"], r["vid"], r["arm"], r["rid"])
        finally:
            A.files = old
        x = ts["v"].to_numpy()
        sg = ts["seg"].to_numpy()
        out.append({"variant": r["variant"], "arm": r["arm_name"], "regime": r["regime"],
                    "T_in_n": int((sg == 0).sum()), "T_in_mean": float(x[sg == 0].mean()) if (sg == 0).any() else None,
                    "T_ho_n": int((sg == 1).sum()), "T_ho_mean": float(x[sg == 1].mean()) if (sg == 1).any() else None})
    return out


def run() -> dict:
    df, mats = load_cells()
    df = label_cells(df)
    df.drop("ci").write_parquet(A.agg_dir() / "cells.parquet")
    o1 = h1(df, mats)
    fam = family()
    o2 = h2(fam)
    fam_cells = {cls: d["cells"] for cls, d in fam.items()}
    o3 = h3(df, o1)
    o4 = {"pegged": allocator("p"), "market": allocator("m")}
    o5 = h5()
    res = {"generated": datetime.now(UTC).isoformat(timespec="seconds"), "timeframes": [tf for tf in A.TFS if tf in mats],
           "trials": {"cells_total": int(df.filter(pl.col("arm").is_in(MAIN)).height),
                      "tested_in_sample": int(df.filter(pl.col("arm").is_in(MAIN) & (pl.col("seg") == 0) & pl.col("tested")).height),
                      "tested_holdout": int(df.filter(pl.col("arm").is_in(MAIN) & (pl.col("seg") == 1) & pl.col("tested")).height)},
           "h1": o1, "h2": o2, "family_cells": fam_cells, "h3": o3, "h4": o4, "h5": o5, "h5_verdicts": h5_verdicts(o5),
           "arm_books": arm_books(mats, df), "regime_summary": regime_summary(df),
           "tradfi": tradfi_check(o1["survivors"] or [{"tf": c["tf"], "vid": c["vid"], "arm": c["arm_id"], "rid": c["rid"], "variant": c["variant"],
                                                      "arm_name": c["arm"], "regime": c["regime"]} for c in o3["cells"]])}
    p = S.root() / "results.json"
    p.write_text(json.dumps(res, indent=1, default=lambda x: x.item() if hasattr(x, "item") else str(x)))
    return res


if __name__ == "__main__":
    run()
