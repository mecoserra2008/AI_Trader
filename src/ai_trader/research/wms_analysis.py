"""WMS analysis (program WMM2): the registered tests H-WMS-1..13 on data/research/wms/features.parquet and friends.

    uv run python -m ai_trader.research.wms_analysis run       # -> results.json, selection.parquet, tables
Segments by weekend_start: llm_in_sample < 2026-07-01 <= clean_holdout < 2026-11-28 <= forward.
"""

from __future__ import annotations

import argparse
import json
import math
from datetime import UTC, date, datetime, timedelta

import numpy as np
import polars as pl
from scipy import stats

from ai_trader.core.config import load
from ai_trader.research import wms_select as ws
from ai_trader.research.wms_data import root

CUT = date(2026, 7, 1)
FWD = date(2026, 11, 28)
IDX = ["vr5_2", "vr5_12", "vr5_48", "vr15_4", "vr15_16", "ar1_5", "ar1_15", "ou_hl", "dfa", "er_1h", "bos_count", "bos_imb"]
LOGT = {"vr5_2", "vr5_12", "vr5_48", "vr15_4", "vr15_16", "ou_hl"}  # analysed in logs (ranks are invariant anyway)
EXPECT_O4 = {k: +1 for k in IDX} | {"bos_count": 0}
RNG = np.random.default_rng(20261127)


def seg(d: date) -> str:
    return "llm_in_sample" if d < CUT else ("clean_holdout" if d < FWD else "forward")


# ---------------------------------------------------------------- inference helpers

def nw(x, lag: int = 4) -> dict:
    x = np.asarray([v for v in x if v is not None and np.isfinite(v)], dtype=np.float64)
    T = x.size
    if T < 3:
        return {"mean": float(x.mean()) if T else None, "t": None, "p": None, "n": T, "se": None}
    m = x.mean()
    e = x - m
    s = e @ e / T
    for j in range(1, min(lag, T - 1) + 1):
        s += 2 * (1 - j / (lag + 1)) * (e[j:] @ e[:-j]) / T
    se = math.sqrt(max(s, 1e-30) / T)
    t = m / se
    return {"mean": float(m), "se": se, "t": float(t), "p": float(2 * stats.t.sf(abs(t), T - 1)), "n": T, "mde": 2.8 * se}


def bh(ps: list[float | None]) -> list[float | None]:
    idx = [i for i, p in enumerate(ps) if p is not None]
    m = len(idx)
    q = [None] * len(ps)
    if not m:
        return q
    order = sorted(idx, key=lambda i: ps[i])
    prev = 1.0
    for rank, i in reversed(list(enumerate(order, 1))):
        prev = min(prev, ps[i] * m / rank)
        q[i] = prev
    return q


def by(ps: list[float | None]) -> list[float | None]:
    m = sum(p is not None for p in ps)
    c = sum(1 / k for k in range(1, m + 1)) if m else 1
    return [None if q is None else min(1.0, q * c) for q in bh(ps)]


def weekly(df: pl.DataFrame, f) -> list[tuple[date, float]]:
    out = []
    for (wk,), g in df.group_by(["weekend_start"], maintain_order=True):
        v = f(g)
        if v is not None and np.isfinite(v):
            out.append((wk, float(v)))
    return sorted(out)


def spearman(g: pl.DataFrame, x: str, y: str, min_n: int = 10) -> float | None:
    h = g.select(x, y).drop_nulls().filter(pl.col(x).is_finite() & pl.col(y).is_finite())
    if h.height < min_n:
        return None
    a, b = h[x].rank().to_numpy(), h[y].rank().to_numpy()
    if a.std() == 0 or b.std() == 0:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def ic_test(df: pl.DataFrame, x: str, y: str) -> dict:
    out = {}
    for s in ("llm_in_sample", "clean_holdout"):
        ser = weekly(df.filter(pl.col("segment") == s), lambda g: spearman(g, x, y))
        out[s] = nw([v for _, v in ser], 4) | {"weeks": len(ser)}
    return out


def fe_cluster(df: pl.DataFrame, x: str, y: str, controls: list[str] = ()) -> dict:
    """y on x (+ controls) with weekend fixed effects; weekend-clustered SE (CR1)."""
    cols = [x, y, *controls]
    h = df.select("weekend_start", *cols).drop_nulls()
    h = h.filter(pl.all_horizontal([pl.col(c).is_finite() for c in cols]))
    if h.height < 30:
        return {}
    h = h.with_columns([(pl.col(c) - pl.col(c).mean().over("weekend_start")).alias(c) for c in cols])
    X = h.select(x, *controls).to_numpy()
    Y = h[y].to_numpy()
    g = h["weekend_start"].to_numpy()
    XtX = np.linalg.pinv(X.T @ X)
    b = XtX @ X.T @ Y
    e = Y - X @ b
    meat = np.zeros((X.shape[1], X.shape[1]))
    ug = np.unique(g)
    for k in ug:
        s = (X[g == k] * e[g == k, None]).sum(axis=0)
        meat += np.outer(s, s)
    G = ug.size
    V = XtX @ meat @ XtX * G / max(G - 1, 1)
    se = math.sqrt(V[0, 0])
    return {"b": float(b[0]), "se": se, "t": float(b[0] / se), "n": int(h.height), "clusters": int(G)}


def auc_by_week(df: pl.DataFrame, score: str, label: str) -> tuple[float, float]:
    """Within-weekend pooled AUC: concordant pairs / pairs (ties 1/2). Returns (sum U, sum pairs)."""
    U = P = 0.0
    for (_,), g in df.group_by(["weekend_start"]):
        s = g[score].to_numpy()
        y = g[label].to_numpy().astype(bool)
        npos, nneg = int(y.sum()), int((~y).sum())
        if npos == 0 or nneg == 0:
            continue
        r = stats.rankdata(s)
        U += r[y].sum() - npos * (npos + 1) / 2
        P += npos * nneg
    return U, P


def block_boot_auc(df: pl.DataFrame, score: str, label: str, B: int = 2000, block: int = 4) -> tuple[float, float, float]:
    wks = sorted(df["weekend_start"].unique().to_list())
    parts = {}
    for (wk,), g in df.group_by(["weekend_start"]):
        parts[wk] = auc_by_week(g, score, label)
    u = np.array([parts[w][0] for w in wks])
    p = np.array([parts[w][1] for w in wks])
    point = u.sum() / p.sum() if p.sum() else math.nan
    n = len(wks)
    nb = math.ceil(n / block)
    res = []
    for _ in range(B):
        st = RNG.integers(0, max(1, n - block + 1), nb)
        ix = np.concatenate([np.arange(s, min(s + block, n)) for s in st])[:n]
        if p[ix].sum() > 0:
            res.append(u[ix].sum() / p[ix].sum())
    return float(point), float(np.percentile(res, 2.5)), float(np.percentile(res, 97.5))


def verdict_ic(fit: dict, hold: dict, sign: int, q_fit: float | None = None, alpha: float = 0.05) -> str:
    """supports: FIT right sign and significant (q or p < alpha), holdout right sign; contradicts: FIT wrong sign (or 0), or holdout
    significantly wrong; else inconclusive. sign = +1 / -1 expected direction."""
    mf, mh = fit.get("mean"), hold.get("mean")
    if mf is None:
        return "inconclusive"
    sig = (q_fit if q_fit is not None else fit.get("p", 1.0)) < alpha
    if sign * mf <= 0:
        return "contradicts"
    if mh is not None and hold.get("t") is not None and sign * hold["t"] <= -2:
        return "contradicts"
    if sig and mh is not None and sign * mh > 0:
        return "supports"
    return "inconclusive"


# ---------------------------------------------------------------- data prep

def load_universe() -> pl.DataFrame:
    F = pl.read_parquet(root() / "features.parquet")
    U = F.filter((pl.col("cov_fw") >= 0.98) & (pl.col("cov_we") >= 0.98) & (pl.col("zero_to_fw") <= 0.10))
    need = list(ws.S0_COMPONENTS) + ["rv_we", "rv_fw", "rv_d1", "rv_pw"]
    U = U.drop_nulls(need).filter(pl.all_horizontal([pl.col(c).is_finite() for c in need]) & (pl.col("rv_we") > 0) & (pl.col("rv_pw") > 0)
                                   & (pl.col("rv_d1") > 0))
    U = U.with_columns(pl.col("weekend_start").map_elements(seg, return_dtype=pl.Utf8).alias("segment"))
    for c in LOGT:
        U = U.with_columns(pl.col(c).log().alias(c), pl.col(f"w_{c}").log().alias(f"w_{c}"))
    return U


def loghar(U: pl.DataFrame, prefix: str = "") -> tuple[pl.DataFrame, dict]:
    """Panel log-HAR forecast of the weekend RV from Friday-known RVs, OLS on FIT rows; lognormal back-transform."""
    X = lambda d: np.column_stack([np.ones(d.height), np.log(d["rv_d1"]), np.log(d["rv_fw"]), np.log(d["rv_pw"])])
    fit = U.filter(pl.col("weekend_start") < CUT)
    Xf, yf = X(fit), np.log(fit["rv_we"].to_numpy())
    b, *_ = np.linalg.lstsq(Xf, yf, rcond=None)
    e = yf - Xf @ b
    s2 = float(e.var())
    lf = X(U) @ b + s2 / 2
    out = U.with_columns(pl.Series(f"{prefix}logF", lf)).with_columns(
        (pl.col("rv_we").log() - pl.col(f"{prefix}logF")).alias("O1_vsurp"))
    naive = np.log(fit["rv_fw"].to_numpy() * 576 / 2016)
    ql = lambda lfc, y: float(np.mean(np.exp(y - lfc) - (y - lfc) - 1))
    info = {"coef": dict(zip(["const", "log_rv_d1", "log_rv_fw", "log_rv_pw"], map(float, b))), "resid_var": s2,
            "r2": float(1 - e.var() / yf.var()), "n_fit": int(fit.height),
            "qlike_fit_loghar": ql(Xf @ b + s2 / 2, yf), "qlike_fit_naive_scaled": ql(naive, yf)}
    return out, info


def outcomes(U: pl.DataFrame) -> pl.DataFrame:
    return U.with_columns((pl.col("mx_we") / pl.col("logF").mul(0.5).exp()).alias("O3_mxs"),
                          pl.col("rtr_we").alias("O2_rtr"), pl.col("trend_we").alias("O4_trend"))


def add_scores(U: pl.DataFrame) -> pl.DataFrame:
    comps = {k: ("neg" if k not in LOGT else "neg") for k in ws.S0_COMPONENTS}  # vr15 are already logs here -> negate
    U = ws.composite(U, components=comps, name="S0")
    wcomps = {f"w_{k}": "neg" for k in ws.S0_COMPONENTS}
    return ws.composite(U, components=wcomps, name="S0_w")


def deriv_inputs(U: pl.DataFrame) -> pl.DataFrame:
    """Friday-known funding (last settlement <= T_F, per 8 h) and 7-day log change of 1d OI (ts + 1 d <= T_F)."""
    fp, op = root() / "funding.parquet", root() / "oi.parquet"
    if not fp.exists():
        return U.with_columns(pl.lit(None, pl.Float64).alias("fund_last"), pl.lit(None, pl.Float64).alias("oi_chg7"))
    fu = pl.read_parquet(fp).sort("symbol", "ts")
    fu = fu.with_columns((pl.col("ts") - pl.col("ts").shift(1).over("symbol")).alias("dt"))
    fu = fu.with_columns((pl.col("rate") * 8 * 3_600_000 / pl.col("dt").clip(3_600_000, 8 * 3_600_000)).alias("rate8"))
    oi = pl.read_parquet(op).sort("symbol", "ts")
    keys = U.select("weekend_start", "symbol").unique().with_columns(
        pl.col("weekend_start").map_elements(lambda d: int(ws.friday_cutoff(d).timestamp() * 1000), return_dtype=pl.Int64).alias("tf"))
    k = keys.sort("tf")
    f1 = k.join_asof(fu.select("symbol", "ts", "rate8").sort("ts"), left_on="tf", right_on="ts", by="symbol", strategy="backward",
                     tolerance=86_400_000)
    o_now = k.with_columns((pl.col("tf") - 86_400_000).alias("t1")).sort("t1").join_asof(
        oi.select("symbol", "ts", "oi").sort("ts"), left_on="t1", right_on="ts", by="symbol", strategy="backward", tolerance=2 * 86_400_000)
    o_then = k.with_columns((pl.col("tf") - 8 * 86_400_000).alias("t1")).sort("t1").join_asof(
        oi.select("symbol", "ts", "oi").sort("ts"), left_on="t1", right_on="ts", by="symbol", strategy="backward", tolerance=2 * 86_400_000)
    d = f1.select("weekend_start", "symbol", pl.col("rate8").alias("fund_last")).join(
        o_now.select("weekend_start", "symbol", pl.col("oi").alias("oi_now")), on=["weekend_start", "symbol"]).join(
        o_then.select("weekend_start", "symbol", pl.col("oi").alias("oi_then")), on=["weekend_start", "symbol"])
    d = d.with_columns((pl.col("oi_now") / pl.col("oi_then")).log().alias("oi_chg7")).drop("oi_now", "oi_then")
    return U.join(d, on=["weekend_start", "symbol"], how="left")


# ---------------------------------------------------------------- tests

def h_persistence(U: pl.DataFrame) -> dict:
    r1 = ic_test(U, "S0", "S0_w") | {"panel_fit": fe_cluster(U.filter(pl.col("segment") == "llm_in_sample"), "S0", "S0_w"),
                                    "panel_hold": fe_cluster(U.filter(pl.col("segment") == "clean_holdout"), "S0", "S0_w")}
    r1["verdict"] = verdict_ic(r1["llm_in_sample"], r1["clean_holdout"], +1)
    per = {k: ic_test(U, k, f"w_{k}") for k in IDX}
    q = bh([per[k]["llm_in_sample"]["p"] for k in IDX])
    for k, qq in zip(IDX, q):
        per[k]["q_fit"] = qq
        per[k]["verdict"] = verdict_ic(per[k]["llm_in_sample"], per[k]["clean_holdout"], +1, qq)
    n_sup = sum(per[k]["verdict"] == "supports" for k in IDX)
    r2 = {"indices": per, "n_supports": n_sup, "verdict": "supports" if n_sup >= 6 else ("contradicts" if n_sup == 0 and all(
        (per[k]["llm_in_sample"]["mean"] or 0) <= 0 for k in IDX) else "inconclusive")}
    return {"H-WMS-1": r1, "H-WMS-2": r2}


def h_outcomes(U: pl.DataFrame) -> dict:
    outs = {"O1_vsurp": 0, "O2_rtr": +1, "O3_mxs": -1, "O4_trend": -1}
    res = {o: ic_test(U, "S0", o) for o in outs}
    q = bh([res[o]["llm_in_sample"]["p"] for o in outs])
    for o, qq in zip(outs, q):
        res[o]["q_fit"] = qq
    ctr = ["log_turn", "log_rvfw"]
    Uc = U.with_columns(pl.col("med_turnover").log().alias("log_turn"), pl.col("rv_fw").log().alias("log_rvfw"))
    for o in outs:
        res[o]["panel_fit"] = fe_cluster(Uc.filter(pl.col("segment") == "llm_in_sample"), "S0", o, ctr)
        res[o]["panel_hold"] = fe_cluster(Uc.filter(pl.col("segment") == "clean_holdout"), "S0", o, ctr)
    v = verdict_ic(res["O4_trend"]["llm_in_sample"], res["O4_trend"]["clean_holdout"], -1, res["O4_trend"]["q_fit"])
    wrong = any((res[o]["q_fit"] or 1) < 0.05 and outs[o] * (res[o]["llm_in_sample"]["mean"] or 0) < 0 for o in ("O2_rtr", "O3_mxs"))
    if v == "supports" and wrong:
        v = "inconclusive"
    r3 = {"outcomes": res, "verdict": v}
    per = {k: ic_test(U, k, "O4_trend") for k in IDX}
    q = bh([per[k]["llm_in_sample"]["p"] for k in IDX])
    passing = []
    for k, qq in zip(IDX, q):
        per[k]["q_fit"] = qq
        sgn = EXPECT_O4[k]
        if sgn == 0:
            mf = per[k]["llm_in_sample"]["mean"] or 0
            sgn = 1 if mf > 0 else -1
            per[k]["verdict"] = "supports" if (qq or 1) < 0.05 and (per[k]["clean_holdout"]["mean"] or 0) * sgn > 0 else "inconclusive"
        else:
            mf = per[k]["llm_in_sample"]["mean"] or 0
            if sgn * mf < 0 and (qq or 1) < 0.05:
                per[k]["verdict"] = "contradicts"
            else:
                per[k]["verdict"] = "supports" if (qq or 1) < 0.05 and sgn * mf > 0 and (per[k]["clean_holdout"]["mean"] or 0) * sgn > 0 \
                    else "inconclusive"
        if (qq or 1) < 0.05 and sgn * mf > 0:
            passing.append((k, sgn))
    n_sup = sum(per[k]["verdict"] == "supports" for k in IDX)
    r4 = {"indices": per, "fit_passing": [k for k, _ in passing], "n_supports": n_sup,
          "verdict": "supports" if n_sup >= 3 else "inconclusive"}
    if passing:
        comps = {k: ("neg" if s > 0 else "pos") for k, s in passing}
        U1 = ws.composite(U, components=comps, name="S1")
        r4["S1"] = ic_test(U1, "S1", "O4_trend")
        r4["S1_trials"] = len(IDX)
    return {"H-WMS-3": r3, "H-WMS-4": r4}


def h_rule(U: pl.DataFrame, score: str = "S0") -> dict:
    S = ws.select(U, score=score)
    res = {}
    for o in ("O1_vsurp", "O2_rtr", "O3_mxs", "O4_trend"):
        res[o] = {}
        for s in ("llm_in_sample", "clean_holdout"):
            ser = weekly(S.filter(pl.col("segment") == s), lambda g: (g.filter(pl.col("eligible_name"))[o].mean() or math.nan)
                         - (g.filter(~pl.col("eligible_name"))[o].mean() or math.nan) if g.height >= 10 else None)
            res[o][s] = nw([v for _, v in ser], 4)
    f, h = res["O4_trend"]["llm_in_sample"], res["O4_trend"]["clean_holdout"]
    if f["mean"] is None or f["mean"] >= 0:
        v = "contradicts"
    elif f["t"] <= -2 and h["mean"] is not None and h["mean"] < 0:
        v = "supports"
    else:
        v = "inconclusive"
    return {"diffs": res, "verdict": v, "score": score}


def h_bos(U: pl.DataFrame) -> dict:
    E = pl.read_parquet(root() / "bos_events.parquet").join(U.select("weekend_start", "symbol", "segment"), on=["weekend_start", "symbol"])
    out, ps, keys = {}, [], []
    for tf in ("15m", "1h"):
        for kind in ("c", "ch"):
            for hz in ("1h", "4h", "close"):
                col = f"{kind}_{hz}"
                r = {}
                for s in ("llm_in_sample", "clean_holdout"):
                    e = E.filter((pl.col("tf") == tf) & (pl.col("segment") == s)).drop_nulls(col)
                    ser = e.group_by("weekend_start").agg(pl.col(col).mean()).sort("weekend_start")[col].to_list()
                    r[s] = nw(ser, 2) | {"events": e.height}
                    if kind == "c":
                        bp = e.group_by("weekend_start").agg(pl.col(f"bp_{hz}").mean()).sort("weekend_start")[f"bp_{hz}"].to_list()
                        r[s]["bp"] = nw(bp, 2)
                key = f"{tf}_{'raw' if kind == 'c' else 'hedged'}_{hz}"
                out[key] = r
                keys.append(key)
                ps.append(r["llm_in_sample"]["p"])
    q = bh(ps)
    for k, qq in zip(keys, q):
        out[k]["q_fit"] = qq
    p = out["15m_raw_4h"]
    f, h = p["llm_in_sample"], p["clean_holdout"]
    if f["mean"] > 0 and (p["q_fit"] or 1) < 0.05 and (h["mean"] or 0) > 0:
        v = "supports"
    elif f["mean"] < 0 and (p["q_fit"] or 1) < 0.05 and (h["mean"] or 0) <= 0:
        v = "contradicts"
    else:
        v = "inconclusive"
    return {"tests": out, "verdict": v, "events_per_symbol_weekend": {
        tf: E.filter(pl.col("tf") == tf).height / max(1, U.height) for tf in ("15m", "1h")}}


def h_drift(U: pl.DataFrame, B: pl.DataFrame) -> dict:
    res = {}
    # H-WMS-7: BTC
    Bw = B.filter(pl.col("have_we")).with_columns(pl.col("weekend_start").map_elements(seg, return_dtype=pl.Utf8).alias("segment"))
    r7 = {}
    for s in ("llm_in_sample", "clean_holdout"):
        b = Bw.filter(pl.col("segment") == s)
        x = b["ret_we"].to_numpy()
        r7[s] = nw(x, 2) | {"share_pos": float((x > 0).mean()), "binom_p": float(stats.binomtest(int((x > 0).sum()), x.size).pvalue),
                            "median": float(np.median(x)),
                            "paired_vs_weekday_rate": nw(x - 0.4 * b["ret_wd"].to_numpy(), 2), "weekday_rate_2d": nw(0.4 * b["ret_wd"].to_numpy(), 2)}
    f, h = r7["llm_in_sample"], r7["clean_holdout"]
    r7["verdict"] = "contradicts" if f["mean"] <= 0 else ("supports" if f["t"] >= 2 and h["mean"] > 0 else "inconclusive")
    res["H-WMS-7"] = r7
    # H-WMS-8: alts minus BTC
    r8 = {}
    for s in ("llm_in_sample", "clean_holdout"):
        u = U.filter(pl.col("segment") == s)
        g = u.group_by("weekend_start").agg(pl.col("s_we").mean().alias("m"), pl.col("s_we").median().alias("md"),
                                            pl.col("s_tt").mean().alias("mt"), pl.len().alias("n")).filter(pl.col("n") >= 10).sort("weekend_start")
        r8[s] = {"mean": nw(g["m"], 2), "median": nw(g["md"], 2), "weekday_tt": nw(g["mt"], 2),
                 "paired_we_minus_tt": nw((g["m"] - g["mt"]).to_list(), 2), "weeks": g.height,
                 "share_weeks_neg": float((g["m"] < 0).mean()) if g.height else None}
    f, h = r8["llm_in_sample"]["mean"], r8["clean_holdout"]["mean"]
    r8["verdict"] = "contradicts" if f["mean"] >= 0 else ("supports" if f["t"] <= -2 and h["mean"] < 0 else "inconclusive")
    res["H-WMS-8"] = r8

    # H-WMS-9 / 10: concentration and skew
    def share(x: np.ndarray) -> float:
        x = x - x.mean()
        k = max(1, math.ceil(0.05 * x.size))
        sq = np.sort(x * x)[::-1]
        return float(sq[:k].sum() / sq.sum()) if sq.sum() > 0 else math.nan

    r9, r10 = {}, {}
    for s in ("llm_in_sample", "clean_holdout"):
        sh, sk, skt, top_share = [], [], [], []
        for (_,), g in U.filter(pl.col("segment") == s).group_by(["weekend_start"]):
            if g.height < 20:
                continue
            a, b = g["s_we"].to_numpy(), g["s_tt"].to_numpy()
            sh.append((share(a), share(b)))
            sk.append(float(stats.skew(a, bias=False)))
            skt.append(float(stats.skew(b, bias=False)))
        sh = np.array(sh)
        r9[s] = {"we": nw(sh[:, 0], 2), "tt": nw(sh[:, 1], 2), "paired": nw(sh[:, 0] - sh[:, 1], 2), "gaussian": 0.279, "weeks": len(sh)}
        r10[s] = {"we": nw(sk, 2), "tt": nw(skt, 2), "paired": nw(np.array(sk) - np.array(skt), 2)}
    f, h = r9["llm_in_sample"]["paired"], r9["clean_holdout"]["paired"]
    r9["verdict"] = "contradicts" if f["mean"] <= 0 else ("supports" if f["t"] >= 2 and h["mean"] > 0 else "inconclusive")
    f, h = r10["llm_in_sample"]["we"], r10["clean_holdout"]["we"]
    r10["verdict"] = "contradicts" if f["mean"] <= 0 else ("supports" if f["t"] >= 2 and h["mean"] > 0 else "inconclusive")
    res["H-WMS-9"], res["H-WMS-10"] = r9, r10
    return res


class _SkLogit:
    """Fallback when the MLE Hessian is singular: L2 logistic (C = 10), same predict / params interface."""

    def __init__(self, y, X):
        from sklearn.linear_model import LogisticRegression
        self.m = LogisticRegression(C=10.0, fit_intercept=False, max_iter=2000).fit(X, y)
        self.params = self.m.coef_[0]
        self.tvalues = np.full(self.params.size, np.nan)
        self.fallback = True

    def predict(self, X):
        return self.m.predict_proba(X)[:, 1]


def _logit(y, X):
    import statsmodels.api as sm
    try:
        r = sm.Logit(y, X).fit(disp=0, maxiter=200)
        if not np.all(np.isfinite(r.params)):
            raise ValueError("non-finite")
        return r
    except Exception:
        return _SkLogit(y, X)


TRENDER_X = ["bos_count", "bos_imb", "er_1h", "fund_last", "abs_fund", "oi_chg7", "abs_fri24_t", "abs_sfw_t", "S0", "log_rvfw"]


def h_trenders(U: pl.DataFrame) -> tuple[dict, pl.DataFrame]:
    import statsmodels.api as sm
    U = U.with_columns(pl.col("fund_last").abs().alias("abs_fund"), (pl.col("ret_fri24").abs() / pl.col("rv_d1").sqrt()).alias("abs_fri24_t"),
                       (pl.col("s_fw").abs() / pl.col("rv_fw").sqrt()).alias("abs_sfw_t"), pl.col("rv_fw").log().alias("log_rvfw"))
    dev = (pl.col("s_we") - pl.col("s_we").mean().over("weekend_start")).abs()
    U = U.with_columns(dev.alias("dev"))
    U = U.with_columns((pl.col("dev").rank("ordinal", descending=True).over("weekend_start")
                        <= (pl.col("dev").count().over("weekend_start") * 0.05).ceil()).alias("trender"))
    for c in TRENDER_X:
        U = U.with_columns(ws.blom(pl.col(c), "weekend_start").fill_null(0.0).fill_nan(0.0).alias(f"n_{c}"))
    Xc = [f"n_{c}" for c in TRENDER_X]
    fit = U.filter(pl.col("segment") == "llm_in_sample")
    hold = U.filter(pl.col("segment") == "clean_holdout")
    years = sorted(fit["weekend_start"].dt.year().unique().to_list())
    cv = []
    for y in years:
        tr, te = fit.filter(pl.col("weekend_start").dt.year() != y), fit.filter(pl.col("weekend_start").dt.year() == y)
        m = _logit(tr["trender"].cast(pl.Int8).to_numpy(), sm.add_constant(tr.select(Xc).to_numpy(), has_constant="add"))
        cv.append(te.with_columns(pl.Series("p_cv", m.predict(sm.add_constant(te.select(Xc).to_numpy(), has_constant="add")))))
    cvd = pl.concat(cv)
    m = _logit(fit["trender"].cast(pl.Int8).to_numpy(), sm.add_constant(fit.select(Xc).to_numpy(), has_constant="add"))
    hold = hold.with_columns(pl.Series("p_h", m.predict(sm.add_constant(hold.select(Xc).to_numpy(), has_constant="add"))))
    a_cv = block_boot_auc(cvd, "p_cv", "trender")
    a_h = block_boot_auc(hold, "p_h", "trender") if hold.height else (math.nan,) * 3
    single = {}
    for c in TRENDER_X:
        single[c] = {"fit": block_boot_auc(fit, f"n_{c}", "trender", B=500), "hold": block_boot_auc(hold, f"n_{c}", "trender", B=500)}
    # two-sided p for single AUCs from the bootstrap normal approximation
    ps = []
    for c in TRENDER_X:
        pt, lo, hi = single[c]["fit"]
        se = (hi - lo) / 3.92
        ps.append(float(2 * stats.norm.sf(abs(pt - 0.5) / se)) if se > 0 else None)
    for c, q in zip(TRENDER_X, bh(ps)):
        single[c]["q_fit"] = q
    if a_cv[1] > 0.5 and a_h[0] > 0.5:
        v = "supports"
    elif a_cv[1] <= 0.5 and a_cv[0] <= 0.52:
        v = "contradicts"
    else:
        v = "inconclusive"
    coefs = dict(zip(["const", *TRENDER_X], map(float, m.params)))
    # direction: do thrivers (top-5% positive) differ from bleeders?
    dirn = {}
    for s in ("llm_in_sample", "clean_holdout"):
        t = U.filter((pl.col("segment") == s) & pl.col("trender"))
        dirn[s] = {"n": t.height, "share_up": float((t["s_we"] > 0).mean()) if t.height else None,
                   "mean_s_up": t.filter(pl.col("s_we") > 0)["s_we"].mean(), "mean_s_down": t.filter(pl.col("s_we") <= 0)["s_we"].mean()}
    return {"cv_auc": a_cv, "hold_auc": a_h, "single": single, "coef": coefs, "verdict": v, "direction": dirn,
            "n_fit": fit.height, "n_hold": hold.height}, U


# ---------------------------------------------------------------- calm classifier (BTC)

def dvol_series() -> pl.DataFrame:
    d = pl.scan_parquet(str(load().root / "data/lake/alt_series/source=deribit/**/*.parquet"), hive_partitioning=True)
    return d.filter((pl.col("series_id") == "dvol_close") & (pl.col("entity") == "BTC")).select("event_ts", "known_ts", "value").collect().sort("known_ts")


def release_times() -> list[datetime]:
    ev = pl.read_parquet(load().root / "data/research/news/events/events.parquet").filter(pl.col("kind").is_in(["cpi", "nfp", "fomc"]))
    return [t.replace(tzinfo=UTC) if t.tzinfo is None else t for t in ev["scheduled_ts"].to_list()]


def calm_frame(B: pl.DataFrame, extra: dict | None = None) -> pl.DataFrame:
    """BTC per-weekend inputs x1..x8 and the calm label (Friday-known thresholds)."""
    B = B.sort("weekend_start")
    if extra:
        B = pl.concat([B, pl.DataFrame([extra])], how="diagonal_relaxed").sort("weekend_start")
    dv = dvol_series()
    rel = release_times()
    fu = pl.read_parquet(root() / "funding.parquet").filter(pl.col("symbol") == "BTCUSDT").sort("ts") if (root() / "funding.parquet").exists() else None
    oi = pl.read_parquet(root() / "oi.parquet").filter(pl.col("symbol") == "BTCUSDT").sort("ts") if (root() / "oi.parquet").exists() else None
    rows = []
    rv_hist: list[float] = []
    dv_hist: list[float] = []
    for r in B.iter_rows(named=True):
        wk = r["weekend_start"]
        tf = ws.friday_cutoff(wk)
        x = {"weekend_start": wk}
        dsel = dv.filter(pl.col("known_ts") <= tf)
        dvol = float(dsel["value"][-1]) if dsel.height and (tf - dsel["known_ts"][-1]).total_seconds() < 6 * 3600 else None
        x["dvol"] = dvol
        med_rv = float(np.median(rv_hist[-52:])) if len(rv_hist) >= 52 else None
        med_dv = float(np.median(dv_hist[-52:])) if len(dv_hist) >= 52 else None
        x["x1_rvfw_vs_med"] = math.log(r["rv_fw"] * 576 / 2016 / med_rv) if med_rv else None
        x["x2_day_vs_week"] = math.log(7 * r["rv_d1"] / r["rv_fw"])
        x["x3_dvol_vs_med"] = math.log(dvol / med_dv) if (dvol and med_dv) else None
        tms = int(tf.timestamp() * 1000)
        if fu is not None:
            f = fu.filter(pl.col("ts") <= tms)
            x["x4_funding"] = float(f["rate"][-1]) * 1e4 if f.height else None
        if oi is not None:
            o1 = oi.filter(pl.col("ts") <= tms - 86_400_000)
            o0 = oi.filter(pl.col("ts") <= tms - 8 * 86_400_000)
            x["x5_oi_chg7"] = math.log(o1["oi"][-1] / o0["oi"][-1]) if (o1.height and o0.height) else None
        x["x6_release_fri"] = float(any(tf - timedelta(hours=24) <= t <= tf for t in rel))
        mon = tf + timedelta(hours=2, days=2)
        x["x7_release_montue"] = float(any(mon <= t < mon + timedelta(days=2) for t in rel))
        x["holiday"] = ws.holiday_flag(wk)
        x["x8_holiday"] = float(x["holiday"] != "")
        x["med_rv52"] = med_rv
        if r.get("have_we") and r.get("rv_we") is not None and med_rv:
            x["calm"] = float(r["rv_we"] < med_rv and r["n_jump5"] == 0)
            x["rv_we"], x["n_jump5"] = r["rv_we"], r["n_jump5"]
        rows.append(x)
        if r.get("have_we") and r.get("rv_we") is not None:
            rv_hist.append(r["rv_we"])
        if dvol:
            dv_hist.append(dvol)
    return pl.DataFrame(rows, infer_schema_length=None).with_columns(
        pl.col("weekend_start").map_elements(seg, return_dtype=pl.Utf8).alias("segment"))


CALM_X = ["x1_rvfw_vs_med", "x2_day_vs_week", "x3_dvol_vs_med", "x4_funding", "x5_oi_chg7", "x6_release_fri", "x7_release_montue", "x8_holiday"]


def h_calm(C: pl.DataFrame, B: pl.DataFrame) -> tuple[dict, object, dict, pl.DataFrame]:
    import statsmodels.api as sm
    D = C.drop_nulls([*CALM_X, "calm"])
    fit = D.filter(pl.col("segment") == "llm_in_sample")
    hold = D.filter(pl.col("segment") == "clean_holdout")
    mu = {c: float(fit[c].mean()) for c in CALM_X}
    sd = {c: float(fit[c].std()) or 1.0 for c in CALM_X}
    Z = lambda d: sm.add_constant(np.column_stack([(d[c].to_numpy() - mu[c]) / sd[c] for c in CALM_X]), has_constant="add")

    def fitm(d):
        return _logit(d["calm"].to_numpy(), Z(d))

    cv = []
    for y in sorted(fit["weekend_start"].dt.year().unique().to_list()):
        tr, te = fit.filter(pl.col("weekend_start").dt.year() != y), fit.filter(pl.col("weekend_start").dt.year() == y)
        m = fitm(tr)
        base = float(tr["calm"].mean())
        cv.append(te.with_columns(pl.Series("p", m.predict(Z(te))), pl.lit(base).alias("p_clim"),
                                  pl.Series("p_pers", _logit(tr["calm"].to_numpy(), sm.add_constant(tr["x1_rvfw_vs_med"].to_numpy(), has_constant="add"))
                                            .predict(sm.add_constant(te["x1_rvfw_vs_med"].to_numpy(), has_constant="add")))))
    cvd = pl.concat(cv)
    m = fitm(fit)
    base = float(fit["calm"].mean())

    def score(d, pcol):
        y, p, pc = d["calm"].to_numpy(), d[pcol].to_numpy(), d["p_clim"].to_numpy()
        bs, bc = float(np.mean((p - y) ** 2)), float(np.mean((pc - y) ** 2))
        dd = (pc - y) ** 2 - (p - y) ** 2
        t = nw(dd, 4)
        T = dd.size
        hln = math.sqrt((T + 1 - 2 * 4 + 4 * 3 / T) / T) if T > 8 else 1.0
        ll = float(np.mean(y * np.log(np.clip(p, 1e-6, 1)) + (1 - y) * np.log(np.clip(1 - p, 1e-6, 1))))
        llc = float(np.mean(y * np.log(np.clip(pc, 1e-6, 1)) + (1 - y) * np.log(np.clip(1 - pc, 1e-6, 1))))
        return {"brier": bs, "brier_clim": bc, "skill": 1 - bs / bc if bc > 0 else None, "dm_t": t["t"], "dm_t_hln": (t["t"] or 0) * hln,
                "logscore": ll, "logscore_clim": llc, "n": T, "base_rate": float(y.mean())}

    res = {"cv": score(cvd, "p"), "cv_persistence_only": score(cvd, "p_pers")}
    hold = hold.with_columns(pl.Series("p", m.predict(Z(hold))) if hold.height else pl.lit(None), pl.lit(base).alias("p_clim"))
    res["holdout"] = score(hold, "p") if hold.height else {}
    # calibration on CV predictions
    p, y = cvd["p"].to_numpy(), cvd["calm"].to_numpy()
    bins = np.quantile(p, [0, 0.2, 0.4, 0.6, 0.8, 1.0])
    rel = []
    for i in range(5):
        k = (p >= bins[i]) & ((p <= bins[i + 1]) if i == 4 else (p < bins[i + 1]))
        rel.append({"p_mean": float(p[k].mean()), "y_mean": float(y[k].mean()), "n": int(k.sum())})
    lg = np.log(np.clip(p, 1e-6, 1 - 1e-6) / np.clip(1 - p, 1e-6, 1))
    cal = _logit(y, sm.add_constant(lg, has_constant="add"))
    res["calibration"] = {"bins": rel, "intercept": float(cal.params[0]), "slope": float(cal.params[1])}
    res["coef"] = dict(zip(["const", *CALM_X], map(float, m.params)))
    res["coef_t"] = dict(zip(["const", *CALM_X], map(float, m.tvalues)))
    res["fallback_fit"] = bool(getattr(m, "fallback", False))
    cvr, hr = res["cv"], res.get("holdout", {})
    if cvr["skill"] is not None and cvr["skill"] <= 0:
        v = "contradicts"
    elif (cvr["skill"] or 0) > 0 and (cvr["dm_t"] or 0) >= 2 and (hr.get("skill") or -1) > 0:
        v = "supports"
    else:
        v = "inconclusive"
    res["verdict"] = v
    res["fit_pass"] = bool((cvr["skill"] or 0) > 0 and (cvr["dm_t"] or 0) >= 2)
    # holiday subset (descriptive)
    allp = pl.concat([cvd.select("weekend_start", "p"), hold.select("weekend_start", "p")]) if hold.height else cvd.select("weekend_start", "p")
    hol = C.filter(pl.col("x8_holiday") > 0).join(allp, on="weekend_start", how="left").join(
        B.select("weekend_start", "rv_fw"), on="weekend_start", how="left")
    res["holidays"] = hol.select("weekend_start", "holiday", "calm", "p", "rv_we", "med_rv52", "n_jump5").to_dicts()
    nh = C.drop_nulls("calm")
    res["holiday_summary"] = {"n": int(nh.filter(pl.col("x8_holiday") > 0).height),
                              "calm_rate_holiday": nh.filter(pl.col("x8_holiday") > 0)["calm"].mean(),
                              "calm_rate_other": nh.filter(pl.col("x8_holiday") == 0)["calm"].mean(),
                              "log_rv_ratio_holiday": nh.filter(pl.col("x8_holiday") > 0).select((pl.col("rv_we") / pl.col("med_rv52")).log().mean()).item(),
                              "log_rv_ratio_other": nh.filter(pl.col("x8_holiday") == 0).select((pl.col("rv_we") / pl.col("med_rv52")).log().mean()).item()}
    nrm = {"mu": mu, "sd": sd}
    rest = C.filter(pl.col("segment") != "llm_in_sample").drop_nulls(CALM_X)
    pt = pl.concat([cvd.select("weekend_start", pl.col("p").alias("calm_p")),
                    rest.select("weekend_start").with_columns(pl.Series("calm_p", m.predict(Z(rest)) if rest.height else []))])
    return res, m, nrm, pt


def predict_calm(m, norm_: dict, row: dict) -> float | None:
    import statsmodels.api as sm
    if any(row.get(c) is None for c in CALM_X):
        return None
    z = np.array([[1.0] + [(row[c] - norm_["mu"][c]) / norm_["sd"][c] for c in CALM_X]])
    return float(m.predict(z)[0])


def properties(U: pl.DataFrame, B: pl.DataFrame) -> dict:
    """ADF + KPSS, moments and Ljung-Box of the weekly series the tests rely on (FIT weekends)."""
    from statsmodels.stats.diagnostic import acorr_ljungbox
    from statsmodels.tsa.stattools import adfuller, kpss
    fit = U.filter(pl.col("segment") == "llm_in_sample")
    ser = {
        "btc_weekend_logret": B.filter(pl.col("have_we") & (pl.col("weekend_start") < CUT)).sort("weekend_start")["ret_we"].to_numpy(),
        "btc_weekend_logrv": np.log(B.filter(pl.col("have_we") & (pl.col("weekend_start") < CUT)).sort("weekend_start")["rv_we"].to_numpy()),
        "xs_mean_alt_spread": fit.group_by("weekend_start").agg(pl.col("s_we").mean()).sort("weekend_start")["s_we"].to_numpy(),
        "ic_S0_O4": np.array([v for _, v in weekly(fit, lambda g: spearman(g, "S0", "O4_trend"))]),
        "ic_S0_S0w": np.array([v for _, v in weekly(fit, lambda g: spearman(g, "S0", "S0_w"))]),
    }
    out = {}
    for k, x in ser.items():
        x = x[np.isfinite(x)]
        with np.errstate(all="ignore"):
            import warnings
            warnings.simplefilter("ignore")
            adf = adfuller(x, autolag="AIC")
            kp = kpss(x, regression="c", nlags="auto")
        lb = acorr_ljungbox(x, lags=[4], return_df=True)
        lb2 = acorr_ljungbox((x - x.mean()) ** 2, lags=[4], return_df=True)
        out[k] = {"n": int(x.size), "mean": float(x.mean()), "sd": float(x.std()), "skew": float(stats.skew(x)), "kurt_excess": float(stats.kurtosis(x)),
                  "adf_p": float(adf[1]), "kpss_p": float(kp[1]), "lb4_p": float(lb["lb_pvalue"].iloc[0]), "lb4_sq_p": float(lb2["lb_pvalue"].iloc[0])}
    pan = {}
    for c in ["S0", "O1_vsurp", "O2_rtr", "O3_mxs", "O4_trend", *IDX]:
        x = fit[c].drop_nulls().to_numpy()
        x = x[np.isfinite(x)]
        pan[c] = {"mean": float(x.mean()), "sd": float(x.std()), "p05": float(np.percentile(x, 5)), "p50": float(np.median(x)),
                  "p95": float(np.percentile(x, 95)), "skew": float(stats.skew(x)), "kurt_excess": float(stats.kurtosis(x))}
    return {"series": out, "panel_fit": pan}


__all__ = ["block_boot_auc", "bh", "by", "calm_frame", "fe_cluster", "h_bos", "h_calm", "h_drift", "h_outcomes", "h_persistence", "h_rule",
           "h_trenders", "ic_test", "load_universe", "loghar", "nw", "predict_calm", "spearman"]
