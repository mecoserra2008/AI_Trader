"""WMQ analysis: probe curves and intensity fits (H-WMQ-1/2), selection on SEL with overfitting diagnostics, held tests
(H-WMQ-3..10), funding, decomposition, capacity (H-WMQ-11), and the report research/reports/wmq_2026-11-27.md.

    PYTHONIOENCODING=utf-8 uv run python -m ai_trader.research.wmq_report select     # SEL only: choices frozen to selection.json
    PYTHONIOENCODING=utf-8 uv run python -m ai_trader.research.wmq_report held       # HOLD tests (needs selection.json)
    PYTHONIOENCODING=utf-8 uv run --extra viz python -m ai_trader.research.wmq_report report
"""

from __future__ import annotations

import json
import math
import sys
from datetime import UTC, datetime
from itertools import combinations
from pathlib import Path
from statistics import NormalDist
from typing import Any

import numpy as np
import polars as pl

from ai_trader.research import wmq_study as S
from ai_trader.sim.ob_gen import write_json

OUT = S.OUT
SEL_FILE = OUT / "selection.json"
HELD_FILE = OUT / "held.json"
B = 10_000
SEED = 20261127
EULER = 0.5772156649015329
PNL = ["net", "spread", "adverse_1s", "adverse_10s", "adverse_60s", "adverse_300s", "inventory", "fees", "funding", "net_f", "closed"]
FAMILIES = {"D": "H-WMQ-6", "A": "H-WMQ-3"}


def cell_name(c: tuple[str, int]) -> str:
    return f"{c[0]} {c[1]} ms"


def lots_of(name: str) -> int:
    return 3 if name.startswith(("B_", "C_")) else 1


def family_of(name: str) -> str:
    return "B0" if name == "B0" else "P" if name == "PROBE" else name.split("_")[0]


# ----------------------------------------------------------------------------------------------------------------------
# loading
# ----------------------------------------------------------------------------------------------------------------------


def funding_by_quoter(sym: str, d: str) -> pl.DataFrame:
    """Funding paid per quoter of one symbol-day: -position x mark x rate at every funding time inside the day."""
    f = S.day_dir(sym, d) / "fills.parquet"
    rows = [(t, r) for t, r in S.funding(sym) if S.day_ms(d) < t < S.day_ms(d) + S.DAY]
    if not f.is_file() or not rows:
        return pl.DataFrame(schema={"variant": pl.Utf8, "queue": pl.Utf8, "latency": pl.Int64, "funding": pl.Float64})
    bars = pl.read_csv(S.day_dir(sym, d) / "bars_1m.csv")
    bm, bc = bars["minute_ms"].to_numpy(), bars["close"].to_numpy()
    fl = pl.read_parquet(f, columns=["variant", "queue", "latency", "ts", "pos_after"]).filter(pl.col("variant") != "PROBE").sort("ts")
    out = []
    for (v, q, lat), g in fl.group_by(["variant", "queue", "latency"]):
        ts, pos = g["ts"].to_numpy(), g["pos_after"].to_numpy()
        tot = 0.0
        for t, rate in rows:
            i = np.searchsorted(ts, t, side="left") - 1  # last fill strictly before the funding time
            if i < 0:
                continue
            k = max(np.searchsorted(bm, t, side="left") - 1, 0)  # the last 1m bar closed by the funding time
            tot += -pos[i] * bc[k] * rate
        out.append({"variant": v, "queue": q, "latency": int(lat), "funding": tot})
    return pl.DataFrame(out, schema={"variant": pl.Utf8, "queue": pl.Utf8, "latency": pl.Int64, "funding": pl.Float64})


def load_quoters(phases: tuple[str, ...] = ("sel", "hold")) -> pl.DataFrame:
    spread_bp = {r["symbol"]: r["spread_bp"] for r in S._UNI["all"] if "error" not in r}
    dfs = []
    for ph in phases:
        for d in S.phase_days(ph):
            for s in S.SYMBOLS:
                p = S.day_dir(s, d) / "quoters.parquet"
                if not p.is_file():
                    continue
                q = pl.read_parquet(p).filter(pl.col("mode") != "probe")
                fu = funding_by_quoter(s, d)
                q = q.join(fu, on=["variant", "queue", "latency"], how="left").with_columns(pl.col("funding").fill_null(0.0))
                dfs.append(q.with_columns(pl.lit(ph).alias("phase"), pl.lit(S.WEEKEND_OF.get(d, d)).alias("weekend"),
                                          pl.lit(spread_bp.get(s, 5.0)).alias("spread_snap_bp")))
    df = pl.concat(dfs, how="diagonal_relaxed")
    lots = pl.col("variant").map_elements(lots_of, return_dtype=pl.Int64)
    df = df.with_columns(lots.alias("lots"), pl.col("variant").map_elements(family_of, return_dtype=pl.Utf8).alias("family"))
    df = df.with_columns((pl.col("net") + pl.col("funding")).alias("net_f"),
                         (pl.col("net") - pl.col("end_inventory_usd").abs() * (5.5 + 0.5 * pl.col("spread_snap_bp")) / 1e4).alias("closed"))
    for k in PNL:
        if k in df.columns:
            df = df.with_columns((pl.col(k) / pl.col("lots")).alias(f"{k}_lot"))
    return df


# ----------------------------------------------------------------------------------------------------------------------
# statistics
# ----------------------------------------------------------------------------------------------------------------------


def boot(vals: np.ndarray, days: np.ndarray, seed: int = SEED) -> dict[str, Any]:
    v = np.asarray(vals, float)
    ok = np.isfinite(v)
    v, days = v[ok], np.asarray(days)[ok]
    if len(v) == 0:
        return {"mean": float("nan"), "ci": [float("nan")] * 2, "n": 0, "n_days": 0}
    u, inv = np.unique(days, return_inverse=True)
    s = np.bincount(inv, weights=v, minlength=len(u))
    c = np.bincount(inv, minlength=len(u)).astype(float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(u), size=(B, len(u)))
    m = s[idx].sum(1) / c[idx].sum(1)
    lo, hi = np.percentile(m, [2.5, 97.5])
    return {"mean": float(v.mean()), "ci": [float(lo), float(hi)], "n": int(len(v)), "n_days": int(len(u)),
            "sd": float(v.std(ddof=1)) if len(v) > 1 else float("nan"), "p_le0": float(np.mean(m <= 0))}


def cell_df(df: pl.DataFrame, c: tuple[str, int]) -> pl.DataFrame:
    return df.filter((pl.col("queue") == c[0]) & (pl.col("latency") == c[1]))


def sd_matrix(df: pl.DataFrame, names: list[str], c: tuple[str, int], col: str = "net_lot") -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    d = cell_df(df, c).filter(pl.col("variant").is_in(names))
    w = d.pivot(on="variant", index=["symbol", "date"], values=col).sort("date", "symbol")
    for n in names:
        if n not in w.columns:
            w = w.with_columns(pl.lit(None, dtype=pl.Float64).alias(n))
    w = w.drop_nulls(subset=names)
    return w.select(names).to_numpy(), w["date"].to_numpy(), w["symbol"].to_numpy()


def cv_select(df: pl.DataFrame, names: list[str], weekends: list[str], k_test: int = 3) -> dict[str, Any]:
    """Combinatorial CV over weekends: per split, a variant's statistic = min over the four cells of its test mean
    (per base lot); score = median over splits; procedure OOS = the train-best's test statistic."""
    mats = {c: sd_matrix(df, names, c) for c in S.CELLS}
    stats, proc, picks = [], [], []
    for test in combinations(weekends, k_test):
        tr = [w for w in weekends if w not in test]
        te_s, tr_s = [], []
        for c in S.CELLS:
            m, dts, _ = mats[c]
            wk = np.array([S.WEEKEND_OF[x] for x in dts])
            te, trm = np.isin(wk, test), np.isin(wk, tr)
            te_s.append(m[te].mean(axis=0) if te.any() else np.full(len(names), np.nan))
            tr_s.append(m[trm].mean(axis=0) if trm.any() else np.full(len(names), np.nan))
        te_min, tr_min = np.min(np.vstack(te_s), axis=0), np.min(np.vstack(tr_s), axis=0)
        b = int(np.nanargmax(tr_min)) if np.isfinite(tr_min).any() else 0
        stats.append(te_min)
        proc.append(float(te_min[b]))
        picks.append(b)
    sm = np.vstack(stats)
    score = np.median(sm, axis=0)
    ch = int(np.nanargmax(score)) if np.isfinite(score).any() else 0
    return {"names": names, "score": [float(x) for x in score], "choice": names[ch], "choice_score": float(score[ch]), "n_splits": len(stats),
            "procedure_oos_mean": float(np.mean(proc)), "procedure_oos_median": float(np.median(proc)),
            "train_pick_counts": {names[i]: picks.count(i) for i in sorted(set(picks))}}


def day_means(df: pl.DataFrame, days: list[str], names: list[str], c: tuple[str, int], col: str = "net_lot") -> np.ndarray:
    d = cell_df(df, c).filter(pl.col("date").is_in(days) & pl.col("variant").is_in(names)).group_by("date", "variant").agg(pl.col(col).mean())
    m = np.full((len(days), len(names)), np.nan)
    di, vi = {x: i for i, x in enumerate(days)}, {x: i for i, x in enumerate(names)}
    for r in d.iter_rows(named=True):
        m[di[r["date"]], vi[r["variant"]]] = r[col]
    return m


def pbo_mc(m: np.ndarray, n_splits: int = 2000, seed: int = SEED) -> dict[str, Any]:
    """CSCV with one-day blocks, Monte-Carlo over half splits: PBO = share of splits where the IS winner ranks at or below
    the OOS median (logit(omega) <= 0)."""
    x = m[np.all(np.isfinite(m), axis=1)]
    s = x.shape[0] - x.shape[0] % 2
    x = x[:s]
    n = x.shape[1]
    if n < 2 or s < 4:
        return {"pbo": float("nan"), "note": "too few candidates or days"}
    rng = np.random.default_rng(seed)
    lam, deg = [], []
    for _ in range(n_splits):
        ins = np.zeros(s, bool)
        ins[rng.choice(s, s // 2, replace=False)] = True
        mi, mo = x[ins].mean(axis=0), x[~ins].mean(axis=0)
        b = int(np.argmax(mi))
        rank = (mo < mo[b]).sum() + 0.5 * ((mo == mo[b]).sum() - 1) + 1
        om = rank / (n + 1)
        lam.append(math.log(om / (1 - om)))
        deg.append((mi[b], mo[b]))
    lam = np.array(lam)
    d = np.array(deg)
    return {"pbo": float(np.mean(lam <= 0)), "logit_median": float(np.median(lam)), "n_splits": n_splits, "blocks": s,
            "is_winner_mean_is": float(d[:, 0].mean()), "is_winner_mean_oos": float(d[:, 1].mean())}


def deflated_sharpe(x: np.ndarray, n_trials: int) -> dict[str, Any]:
    """Bailey & Lopez de Prado (2014) per-day Sharpe, null trial variance V = 1 / (T - 1)."""
    nd = NormalDist()
    x = x[np.isfinite(x)]
    t = len(x)
    if t < 3 or np.std(x, ddof=1) == 0:
        return {"sr_day": float("nan"), "dsr": float("nan"), "T": t}
    sr = float(x.mean() / x.std(ddof=1))
    z = (x - x.mean()) / x.std(ddof=0)
    skew, kurt = float(np.mean(z**3)), float(np.mean(z**4))
    nn = max(float(n_trials), 2.0)
    zmax = (1 - EULER) * nd.inv_cdf(1 - 1 / nn) + EULER * nd.inv_cdf(1 - 1 / (nn * math.e))
    den = math.sqrt(max(1 - skew * sr + (kurt - 1) / 4 * sr**2, 1e-12))
    sr0 = math.sqrt(1.0 / (t - 1)) * zmax
    return {"sr_day": sr, "sr0_day": sr0, "n_trials": n_trials, "T": t, "skew": skew, "kurt": kurt,
            "psr0": nd.cdf(sr * math.sqrt(t - 1) / den), "dsr": nd.cdf((sr - sr0) * math.sqrt(t - 1) / den)}


# ----------------------------------------------------------------------------------------------------------------------
# selection (SEL only)
# ----------------------------------------------------------------------------------------------------------------------


def grid_names() -> list[str]:
    cal = json.loads(S.CALIB.read_text(encoding="utf-8"))
    return [v["name"] for v in S.grid(cal, S.SYMBOLS[0], "sel")]


def select(df: pl.DataFrame) -> dict[str, Any]:
    sel = df.filter(pl.col("phase") == "sel")
    names = grid_names()
    quoting = [n for n in names if n != "PROBE"]
    wmq = [n for n in quoting if n != "B0"]
    sel_days = S.phase_days("sel")
    allm = day_means(sel, sel_days, quoting, S.PRIMARY)

    def diag(cands: list[str], choice: str) -> dict[str, Any]:
        m = day_means(sel, sel_days, cands, S.PRIMARY)
        x = allm[:, quoting.index(choice)]
        prim = cell_df(sel, S.PRIMARY).filter(pl.col("variant") == choice)
        return {"pbo": pbo_mc(m) if len(cands) >= 2 else {"pbo": float("nan"), "note": "one candidate"}, "dsr": deflated_sharpe(x, len(quoting)),
                "sel_mean_primary": float(prim["net_lot"].mean()) if prim.height else float("nan"),
                "sel_cells": {cell_name(c): float(cell_df(sel, c).filter(pl.col("variant") == choice)["net_lot"].mean()) for c in S.CELLS}}

    res: dict[str, Any] = {"written": datetime.now(UTC).isoformat(timespec="seconds"), "n_quoting": len(quoting), "families": {}}
    fams = {"A": [n for n in wmq if n.startswith("A_")], "D": [n for n in wmq if n.startswith("D_")],
            "B_lin": [n for n in wmq if n.startswith("B_") and n.endswith("_lin")], "C_lin": [n for n in wmq if n.startswith("C_") and n.endswith("_lin")],
            "E": [n for n in wmq if n.startswith("E_")], "A2": [n for n in wmq if n.startswith("A2_")],
            "ALL": [n for n in wmq if not n.startswith("A2_")]}  # H-WMQ-10: the 36 registered candidates (A2 is an extra trial)
    for f, cands in fams.items():
        cv = cv_select(sel, cands, S.SEL_W)
        res["families"][f] = {**cv, **diag(cands, cv["choice"])}
    # controls of the chosen B / C members and E bases
    res["families"]["B_lin"]["control"] = res["families"]["B_lin"]["choice"].replace("_lin", "_fla")
    res["families"]["C_lin"]["control"] = res["families"]["C_lin"]["choice"].replace("_lin", "_fla")
    e = res["families"]["E"]["choice"]
    res["families"]["E"]["control"] = "D_m1" if e.startswith("E_D_m1") else "A_glft_s0.5_p"
    return res


# ----------------------------------------------------------------------------------------------------------------------
# held tests
# ----------------------------------------------------------------------------------------------------------------------


def held_eval(df: pl.DataFrame, name: str, col: str = "net_lot") -> dict[str, Any]:
    out: dict[str, Any] = {"variant": name, "cells": {}}
    for c in S.CELLS:
        h = cell_df(df, c).filter((pl.col("variant") == name) & (pl.col("phase") == "hold"))
        b = boot(h[col].to_numpy(), h["date"].to_numpy())
        b["decomposition"] = {k: float(h[f"{k}_lot"].mean()) for k in PNL if f"{k}_lot" in h.columns} if h.height else {}
        b["fills_per_sd"] = float(h["fills"].mean()) if h.height else float("nan")
        b["through_share"] = float(h["fills_through"].sum() / max(h["fills"].sum(), 1)) if h.height else float("nan")
        b["stopped_share"] = float(h["stopped"].mean()) if h.height else float("nan")
        b["pos_share"] = float((h[col] > 0).mean()) if h.height else float("nan")
        b["worst_sd"] = float(h[col].min()) if h.height else float("nan")
        out["cells"][cell_name(c)] = b
    cs = list(out["cells"].values())
    out["passes"] = all(x["mean"] > 0 and x["ci"][0] > 0 for x in cs)
    p = out["cells"][cell_name(S.PRIMARY)]
    out["contradicts"] = p["mean"] <= 0 and p["ci"][1] < 0
    return out


def paired(df: pl.DataFrame, a: str, b: str, col: str = "net_lot") -> dict[str, Any]:
    out: dict[str, Any] = {"a": a, "b": b, "cells": {}}
    for c in S.CELLS:
        h = cell_df(df, c).filter(pl.col("phase") == "hold")
        x = h.filter(pl.col("variant") == a).select("symbol", "date", pl.col(col).alias("na"))
        y = h.filter(pl.col("variant") == b).select("symbol", "date", pl.col(col).alias("nb"))
        j = x.join(y, on=["symbol", "date"])
        r = boot((j["na"] - j["nb"]).to_numpy(), j["date"].to_numpy())
        r.update(mean_a=float(j["na"].mean()) if j.height else float("nan"), mean_b=float(j["nb"].mean()) if j.height else float("nan"))
        out["cells"][cell_name(c)] = r
    cs = list(out["cells"].values())
    out["all_pos"] = all(x["mean"] > 0 and x["ci"][0] > 0 for x in cs)
    p = out["cells"][cell_name(S.PRIMARY)]
    out["neg_primary"] = p["mean"] < 0 and p["ci"][1] < 0
    out["pos_primary"] = p["mean"] > 0
    return out


def verdict_abs(h: dict[str, Any], sel_mean: float) -> str:
    return "supports" if h["passes"] and sel_mean > 0 else "contradicts" if h["contradicts"] else "inconclusive"


def verdict_paired(p: dict[str, Any]) -> str:
    return "supports" if p["all_pos"] else "contradicts" if p["neg_primary"] else "inconclusive"


def verdict_two(p1: dict[str, Any], p2: dict[str, Any]) -> str:
    if (p1["all_pos"] and p2["pos_primary"]) or (p2["all_pos"] and p1["pos_primary"]):
        return "supports"
    if p1["neg_primary"] and p2["neg_primary"]:
        return "contradicts"
    return "inconclusive"


def held_tests(df: pl.DataFrame, sel: dict[str, Any]) -> dict[str, Any]:
    fam = sel["families"]
    r: dict[str, Any] = {"written": datetime.now(UTC).isoformat(timespec="seconds")}
    a = fam["A"]["choice"]
    r["H-WMQ-3"] = {"variant": a, "held": held_eval(df, a), "vs_B0": paired(df, a, "B0"), "sel_mean": fam["A"]["sel_mean_primary"]}
    r["H-WMQ-3"]["verdict"] = verdict_abs(r["H-WMQ-3"]["held"], fam["A"]["sel_mean_primary"])
    a2 = fam["A2"]["choice"]
    r["H-WMQ-13"] = {"variant": a2, "held": held_eval(df, a2), "vs_B0": paired(df, a2, "B0"), "sel_mean": fam["A2"]["sel_mean_primary"]}
    r["H-WMQ-13"]["verdict"] = verdict_abs(r["H-WMQ-13"]["held"], fam["A2"]["sel_mean_primary"])
    d = fam["D"]["choice"]
    r["H-WMQ-6"] = {"variant": d, "held": held_eval(df, d), "vs_B0": paired(df, d, "B0"), "sel_mean": fam["D"]["sel_mean_primary"]}
    r["H-WMQ-6"]["verdict"] = verdict_abs(r["H-WMQ-6"]["held"], fam["D"]["sel_mean_primary"])
    for h, f in (("H-WMQ-4", "B_lin"), ("H-WMQ-5", "C_lin"), ("H-WMQ-7", "E")):
        ch, ctl = fam[f]["choice"], fam[f]["control"]
        p = paired(df, ch, ctl)
        r[h] = {"variant": ch, "control": ctl, "paired": p, "held": held_eval(df, ch), "held_control": held_eval(df, ctl), "verdict": verdict_paired(p)}
    for h, suf in (("H-WMQ-8", "btc"), ("H-WMQ-9", "trend")):
        f = "F" if suf == "btc" else "T"
        p1 = paired(df, f"{f}_D_m1_{suf}", "D_m1")
        p2 = paired(df, f"{f}_A_glft_s0.5_p_{suf}", "A_glft_s0.5_p")
        r[h] = {"paired_D_m1": p1, "paired_A_glft_s0.5_p": p2, "verdict": verdict_two(p1, p2)}
    best = fam["ALL"]["choice"]
    hb, pb = held_eval(df, best), paired(df, best, "B0")
    v = "supports" if hb["passes"] and pb["all_pos"] and fam["ALL"]["sel_mean_primary"] > 0 else "contradicts" if hb["contradicts"] or pb["neg_primary"] else "inconclusive"
    r["H-WMQ-10"] = {"variant": best, "held": hb, "vs_B0": pb, "sel_mean": fam["ALL"]["sel_mean_primary"], "verdict": v,
                     "core": held_eval(df.filter(pl.col("symbol").is_in(S.CORE)), best), "extra": held_eval(df.filter(pl.col("symbol").is_in(S.EXTRA)), best)}
    r["B0"] = held_eval(df, "B0")
    r["all_variants_primary"] = {n: boot(cell_df(df, S.PRIMARY).filter((pl.col("variant") == n) & (pl.col("phase") == "hold"))["net_lot"].to_numpy(),
                                         cell_df(df, S.PRIMARY).filter((pl.col("variant") == n) & (pl.col("phase") == "hold"))["date"].to_numpy())
                                 for n in grid_names() if n != "PROBE"}
    return r


# ----------------------------------------------------------------------------------------------------------------------
# probe: curves, H-WMQ-1, H-WMQ-2
# ----------------------------------------------------------------------------------------------------------------------


def probe_curves(phase: str, cell: tuple[str, int] = S.PRIMARY) -> pl.DataFrame:
    days = S.phase_days(phase)
    legs = S.load_probe_legs(days, cell)
    fills = S.load_probe_fills(days, cell)
    if not legs.height:
        return pl.DataFrame()
    lt = S.leg_table(legs)
    t = lt.group_by("symbol", "j").agg(pl.col("x").first(), pl.col("live_s").sum(), pl.col("fills").sum(),
                                       ((pl.col("x_eff") * pl.col("live_s")).sum() / pl.col("live_s").sum()).alias("x_eff"),
                                       ((pl.col("d_bps") * pl.col("live_s")).sum() / pl.col("live_s").sum()).alias("d_bps"))
    f = fills.group_by("symbol", "probe_j").agg(pl.len().alias("n"), *[pl.col(f"e_{h}_bp").mean().alias(f"e{h[:-1]}") for h in ("1s", "10s", "60s", "300s")],
                                                pl.col("spread_bp").mean().alias("spread_bp"), *[pl.col(f"adv_{h}_bp").mean().alias(f"adv{h[:-1]}") for h in ("1s", "10s", "60s", "300s")],
                                                (pl.col("kind") == "through").mean().alias("through_share")).rename({"probe_j": "j"})
    return t.join(f, on=["symbol", "j"], how="left").with_columns((pl.col("fills") / pl.col("live_s") * 3600).alias("fills_per_h")).sort("symbol", "j")


def h1(cal: dict[str, Any]) -> dict[str, Any]:
    xs = {s: cal["symbols"][s]["params"]["xstar"] for s in S.SYMBOLS}
    out: dict[str, Any] = {"xstar": xs, "cells": {}, "sel": {}}
    for ph in ("sel", "hold"):
        for c in S.CELLS:
            f = S.load_probe_fills(S.phase_days(ph), c)
            if not f.height:
                continue
            f = f.with_columns(pl.col("symbol").replace_strict(xs, return_dtype=pl.Float64).alias("xstar"),
                               pl.col("probe_j").map_elements(lambda j: S.PROBE_X[j] if 0 <= j < len(S.PROBE_X) else float("nan"), return_dtype=pl.Float64).alias("xj"))
            g = f.filter((pl.col("xj") - pl.col("xstar")).abs() < 1e-9)
            r = {"e60": boot(g["e_60s_bp"].to_numpy(), g["date"].to_numpy()), "e300": boot(g["e_300s_bp"].to_numpy(), g["date"].to_numpy()),
                 "e1": float(g["e_1s_bp"].mean()) if g.height else float("nan"), "spread_bp": float(g["spread_bp"].mean()) if g.height else float("nan"),
                 "n_fills": g.height}
            (out["cells"] if ph == "hold" else out["sel"])[cell_name(c)] = r
    cs = list(out["cells"].values())
    passes = len(cs) == 4 and all(x["e60"]["mean"] > 0 and x["e60"]["ci"][0] > 0 and x["e300"]["mean"] > 0 for x in cs)
    selm = out["sel"].get(cell_name(S.PRIMARY), {}).get("e60", {}).get("mean", float("nan"))
    p = out["cells"].get(cell_name(S.PRIMARY), {}).get("e60", {"mean": float("nan"), "ci": [float("nan")] * 2})
    out["verdict"] = "supports" if passes and selm > 0 else "contradicts" if p["mean"] <= 0 and p["ci"][1] < 0 else "inconclusive"
    return out


def _pois_ll(n: np.ndarray, mu: np.ndarray) -> np.ndarray:
    from scipy.special import gammaln

    return n * np.log(np.maximum(mu, 1e-300)) - mu - gammaln(n + 1)


def h2(cal: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {"phases": {}, "stability": {}}
    for ph in ("sel", "hold"):
        lt = S.leg_table(S.load_probe_legs(S.phase_days(ph))).filter(pl.col("x_eff").is_not_null() & (pl.col("live_s") > 0))
        rows = []
        for s in S.SYMBOLS:
            fit = cal["symbols"][s]["fit"] or cal["pooled"]["fit"]
            t = lt.filter(pl.col("symbol") == s)
            if not t.height:
                continue
            n, e, x, d = t["fills"].to_numpy().astype(float), t["live_s"].to_numpy(), t["x_eff"].to_numpy(), t["d_bps"].to_numpy()
            mu_x = e * fit["exp_sigma"]["A"] * np.exp(-fit["exp_sigma"]["kappa"] * x)
            mu_b = e * fit["exp_bps"]["A"] * np.exp(-fit["exp_bps"]["k_bps"] * d)
            mu_p = e * fit["power"]["A"] * (x + 0.05) ** (-fit["power"]["alpha"])
            rows.append(pl.DataFrame({"date": t["date"], "ll_x": _pois_ll(n, mu_x), "ll_b": _pois_ll(n, mu_b), "ll_p": _pois_ll(n, mu_p)}))
        if not rows:
            continue
        a = pl.concat(rows).group_by("date").agg(pl.col("ll_x").sum(), pl.col("ll_b").sum(), pl.col("ll_p").sum())
        out["phases"][ph] = {"vs_bps": boot((a["ll_x"] - a["ll_b"]).to_numpy(), a["date"].to_numpy()), "vs_power": boot((a["ll_x"] - a["ll_p"]).to_numpy(), a["date"].to_numpy()),
                             "ll_x": float(a["ll_x"].sum()), "ll_b": float(a["ll_b"].sum()), "ll_p": float(a["ll_p"].sum()), "n_days": a.height}
        for s in S.SYMBOLS:
            t = lt.filter(pl.col("symbol") == s)
            if t.height:
                f = S.poisson_fit(t["fills"].to_numpy(), t["live_s"].to_numpy(), t["x_eff"].to_numpy())
                out["stability"].setdefault(s, {})[ph] = {"kappa": -f["b1"], "kappa_se": f["se1"], "A": math.exp(f["b0"])}
    for s in S.SYMBOLS:
        f = cal["symbols"][s]["fit"]
        out["stability"].setdefault(s, {})["cal"] = {"kappa": f["exp_sigma"]["kappa"], "kappa_se": f["exp_sigma"]["se1"], "A": f["exp_sigma"]["A"]} if f else None
    stable = [s for s, v in out["stability"].items() if v.get("cal") and v.get("hold") and abs(v["hold"]["kappa"] / v["cal"]["kappa"] - 1) <= 0.3]
    out["stable_symbols"] = stable
    ho = out["phases"].get("hold")
    if ho:
        ok = ho["vs_bps"]["ci"][1] >= 0 and ho["vs_power"]["ci"][1] >= 0
        bad = ho["vs_bps"]["ci"][1] < 0 or ho["vs_power"]["ci"][1] < 0
        out["verdict"] = "supports" if ok and len(stable) >= 8 else "contradicts" if bad or len(stable) < 4 else "inconclusive"
    else:
        out["verdict"] = "inconclusive"
    return out


# ----------------------------------------------------------------------------------------------------------------------
# capacity: the book on one clock (H-WMQ-11)
# ----------------------------------------------------------------------------------------------------------------------


def end_mid(sym: str, d: str) -> float:
    b = pl.read_csv(S.day_dir(sym, d) / "bars_1m.csv")
    return float(b["close"][-1])


def portfolio(name: str, cell: tuple[str, int], net_cap: float = 40.0, gross_cap: float = 80.0, days: list[str] | None = None,
              symbols: list[str] | None = None) -> dict[str, Any]:
    """The variant's fills of all symbols of each day merged on one clock under net / gross caps (USD). An adding fill
    that would breach a cap, counting one resting adding order of the symbol's last fill size per other quoting symbol
    and side as filled (the net cap) is dropped together with that symbol's later fills until its replayed position is
    flat again. Day P&L = cash of the kept fills + kept inventory at the day's last 1m close."""
    days = days or S.phase_days("hold")
    symbols = symbols or S.SYMBOLS
    rows = []
    for d in days:
        fl = []
        for s in symbols:
            p = S.day_dir(s, d) / "fills.parquet"
            if not p.is_file():
                continue
            f = pl.read_parquet(p).filter((pl.col("variant") == name) & (pl.col("queue") == cell[0]) & (pl.col("latency") == cell[1]))
            fl.append(f.with_columns(pl.lit(s).alias("symbol")))
        if not fl:
            continue
        f = pl.concat(fl).sort("ts")
        pos = {s: 0.0 for s in symbols}
        rep = {s: 0.0 for s in symbols}
        blocked = {s: False for s in symbols}
        last_px = {s: 0.0 for s in symbols}
        cash, n_drop, n_keep, max_names, max_gross = 0.0, 0, 0, 0, 0.0
        lot_usd = {s: 0.0 for s in symbols}
        for r in f.iter_rows(named=True):
            s, q = r["symbol"], r["s"] * r["qty"]
            rep_before = rep[s]
            rep[s] = r["pos_after"]
            last_px[s] = r["price"]
            lot_usd[s] = max(lot_usd[s], r["qty"] * r["price"])
            if blocked[s]:
                if abs(rep[s]) < 1e-12:
                    blocked[s] = False
                n_drop += 1
                continue
            adding = abs(pos[s] + q) > abs(pos[s]) + 1e-12
            if adding:
                usd = {k: pos[k] * (last_px[k] or r["price"]) for k in symbols}
                after = dict(usd)
                after[s] = (pos[s] + q) * r["price"]
                pend = sum(lot_usd[k] for k in symbols if k != s and lot_usd[k] > 0) * (1 if q > 0 else -1)
                net_after = sum(after.values()) + pend
                gross_after = sum(abs(v) for v in after.values())
                net_ok = abs(net_after) <= net_cap or abs(sum(after.values())) < abs(sum(usd.values()))
                if not (net_ok and gross_after <= gross_cap):
                    blocked[s] = abs(rep[s]) > 1e-12
                    n_drop += 1
                    continue
            pos[s] += q
            cash -= q * r["price"] + r["fee"]
            n_keep += 1
            max_names = max(max_names, sum(1 for k in symbols if abs(pos[k]) > 1e-12))
            max_gross = max(max_gross, sum(abs(pos[k] * (last_px[k] or r["price"])) for k in symbols))
            _ = rep_before
        pnl = cash + sum(pos[s] * end_mid(s, d) for s in symbols if abs(pos[s]) > 1e-12)
        rows.append({"date": d, "pnl": pnl, "kept": n_keep, "dropped": n_drop, "max_names": max_names, "max_gross": max_gross})
    t = pl.DataFrame(rows)
    b = boot(t["pnl"].to_numpy(), t["date"].to_numpy()) if t.height else {"mean": float("nan"), "ci": [float("nan")] * 2}
    return {"variant": name, "cell": cell_name(cell), "net_cap": net_cap, "gross_cap": gross_cap, "per_day": b,
            "worst_day": float(t["pnl"].min()) if t.height else float("nan"), "best_day": float(t["pnl"].max()) if t.height else float("nan"),
            "drop_share": float(t["dropped"].sum() / max(t["kept"].sum() + t["dropped"].sum(), 1)) if t.height else float("nan"),
            "max_names": int(t["max_names"].max()) if t.height else 0, "max_gross": float(t["max_gross"].max()) if t.height else 0.0, "days": t.height}


# ----------------------------------------------------------------------------------------------------------------------
# report tables
# ----------------------------------------------------------------------------------------------------------------------


def fmt(x: Any, n: int = 4) -> str:
    try:
        x = float(x)
    except (TypeError, ValueError):
        return "n/a"
    return "n/a" if not math.isfinite(x) else f"{x:+.{n}f}"


def ci(b: dict[str, Any], n: int = 4) -> str:
    return f"{fmt(b['mean'], n)} [{fmt(b['ci'][0], n)}, {fmt(b['ci'][1], n)}]"


def t_cells(rows: list[tuple[str, str, dict[str, Any]]], n: int = 4) -> list[str]:
    out = ["| test | variant | " + " | ".join(cell_name(c) for c in S.CELLS) + " |", "|---|---|" + "---|" * len(S.CELLS)]
    for h, v, cells in rows:
        out.append(f"| {h} | {v} | " + " | ".join(ci(cells[cell_name(c)], n) for c in S.CELLS) + " |")
    return out


def t_decomp(items: list[tuple[str, dict[str, Any]]]) -> list[str]:
    ks = ["spread", "adverse_1s", "adverse_10s", "adverse_60s", "adverse_300s", "inventory", "fees", "funding", "net_f", "closed"]
    out = ["| variant | cell | net [95% CI] | " + " | ".join(ks) + " | fills / sd | through share | worst sd |", "|---|---|---|" + "---|" * (len(ks) + 3)]
    for name, h in items:
        for c in S.CELLS:
            b = h["cells"][cell_name(c)]
            dd = b.get("decomposition", {})
            out.append(f"| {name} | {cell_name(c)} | {ci(b)} | " + " | ".join(fmt(dd.get(k), 4) for k in ks)
                       + f" | {b['fills_per_sd']:.1f} | {b['through_share']:.2f} | {fmt(b['worst_sd'], 3)} |")
    return out


def t_probe(cur: pl.DataFrame, title: str) -> list[str]:
    if not cur.height:
        return [f"({title}: no data)"]
    g = cur.group_by("j").agg(pl.col("x").first(), pl.col("live_s").sum(), pl.col("fills").sum(), pl.col("n").sum(),
                              ((pl.col("x_eff") * pl.col("live_s")).sum() / pl.col("live_s").sum()).alias("x_eff"),
                              ((pl.col("d_bps") * pl.col("live_s")).sum() / pl.col("live_s").sum()).alias("d_bps"),
                              *[((pl.col(k) * pl.col("n")).sum() / pl.col("n").sum()).alias(k) for k in ("spread_bp", "adv1", "adv10", "adv60", "adv300", "e1", "e60", "e300", "through_share")]).sort("j")
    out = [f"**{title}** (pooled over symbols, primary cell; per-fill means in bp; lambda = fills per hour of one resting order-side)", "",
           "| x (sigma) | x_eff | distance bp | fills / h | fills | spread | adv 1 s | adv 10 s | adv 60 s | adv 300 s | e1 | e60 | e300 | through share |",
           "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in g.iter_rows(named=True):
        out.append(f"| {r['x']:g} | {r['x_eff']:.2f} | {r['d_bps']:.1f} | {3600 * r['fills'] / r['live_s']:.3f} | {int(r['fills'])} | {fmt(r['spread_bp'], 2)} | {fmt(r['adv1'], 2)} | "
                   f"{fmt(r['adv10'], 2)} | {fmt(r['adv60'], 2)} | {fmt(r['adv300'], 2)} | {fmt(r['e1'], 2)} | {fmt(r['e60'], 2)} | {fmt(r['e300'], 2)} | {r['through_share']:.2f} |")
    return out


# ----------------------------------------------------------------------------------------------------------------------
# conditional tests: R-WMS (H-WMQ-14), WMN gate (H-WMQ-12 / 15), capacity re-runs (H-WMQ-11)
# ----------------------------------------------------------------------------------------------------------------------


def boot_diff(v: np.ndarray, grp: np.ndarray, days: np.ndarray, seed: int = SEED) -> dict[str, Any]:
    """Mean(v | grp) - mean(v | ~grp) with a day-clustered bootstrap CI."""
    v, grp, days = np.asarray(v, float), np.asarray(grp, bool), np.asarray(days)
    u, inv = np.unique(days, return_inverse=True)
    s1 = np.bincount(inv, weights=v * grp, minlength=len(u))
    c1 = np.bincount(inv, weights=grp.astype(float), minlength=len(u))
    s0 = np.bincount(inv, weights=v * ~grp, minlength=len(u))
    c0 = np.bincount(inv, weights=(~grp).astype(float), minlength=len(u))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(u), size=(B, len(u)))
    with np.errstate(invalid="ignore", divide="ignore"):
        d = s1[idx].sum(1) / c1[idx].sum(1) - s0[idx].sum(1) / c0[idx].sum(1)
    d = d[np.isfinite(d)]
    est = v[grp].mean() - v[~grp].mean() if grp.any() and (~grp).any() else float("nan")
    return {"diff": float(est), "ci": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))] if len(d) else [float("nan")] * 2,
            "n_in": int(grp.sum()), "n_out": int((~grp).sum())}


def boot_slope(y: np.ndarray, x: np.ndarray, days: np.ndarray, seed: int = SEED) -> dict[str, Any]:
    y, x, days = np.asarray(y, float), np.asarray(x, float), np.asarray(days)
    ok = np.isfinite(y) & np.isfinite(x)
    y, x, days = y[ok], x[ok], days[ok]
    u, inv = np.unique(days, return_inverse=True)
    groups = [np.where(inv == i)[0] for i in range(len(u))]
    rng = np.random.default_rng(seed)

    def slope(ix: np.ndarray) -> float:
        xx, yy = x[ix], y[ix]
        v = np.var(xx)
        return float(np.cov(xx, yy, bias=True)[0, 1] / v) if v > 0 else float("nan")

    est = slope(np.arange(len(y)))
    bs = [slope(np.concatenate([groups[i] for i in rng.integers(0, len(u), len(u))])) for _ in range(2000)]
    bs = np.array([b for b in bs if np.isfinite(b)])
    return {"slope": est, "ci": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))], "n": int(len(y))}


def h14(df: pl.DataFrame, best: str) -> dict[str, Any]:
    w = pl.read_parquet(OUT.parents[0] / "wms" / "selection.parquet").select(
        pl.col("weekend_start").cast(pl.Utf8).alias("weekend"), "symbol", "eligible_name", "eligible", "score", "gate_on")
    out: dict[str, Any] = {"variant": best, "cells": {}}
    for c in S.CELLS:
        h = cell_df(df, c).filter(pl.col("phase") == "hold")
        x = h.filter(pl.col("variant") == best).select("symbol", "date", "weekend", pl.col("net_lot").alias("nv"))
        y = h.filter(pl.col("variant") == "B0").select("symbol", "date", pl.col("net").alias("nb"))
        j = x.join(y, on=["symbol", "date"]).join(w, on=["weekend", "symbol"], how="left")
        missing = int(j["eligible_name"].is_null().sum())
        j = j.drop_nulls("eligible_name").with_columns((pl.col("nv") - pl.col("nb")).alias("d"))
        el = j.filter(pl.col("eligible_name"))
        r = {"paired_eligible": boot(el["d"].to_numpy(), el["date"].to_numpy()), "net_eligible": boot(el["nv"].to_numpy(), el["date"].to_numpy()),
             "b0_eligible": boot(el["nb"].to_numpy(), el["date"].to_numpy()), "missing_rows": missing, "n_eligible": el.height, "n_all": j.height,
             "diff_eligible_minus_rest": boot_diff(j["d"].to_numpy(), j["eligible_name"].to_numpy(), j["date"].to_numpy()),
             "net_diff_eligible_minus_rest": boot_diff(j["nv"].to_numpy(), j["eligible_name"].to_numpy(), j["date"].to_numpy())}
        g = j.filter(pl.col("eligible").fill_null(False))
        r["paired_gate_on"] = boot(g["d"].to_numpy(), g["date"].to_numpy())
        r["net_gate_on"] = boot(g["nv"].to_numpy(), g["date"].to_numpy())
        if c == S.PRIMARY:
            r["slope_paired_on_score"] = boot_slope(j["d"].to_numpy(), j["score"].to_numpy(), j["date"].to_numpy())
            r["slope_net_on_score"] = boot_slope(j["nv"].to_numpy(), j["score"].to_numpy(), j["date"].to_numpy())
        out["cells"][cell_name(c)] = r
    cs = list(out["cells"].values())
    ok = all(x["paired_eligible"]["mean"] > 0 and x["paired_eligible"]["ci"][0] > 0 and x["net_eligible"]["mean"] > 0 and x["net_eligible"]["ci"][0] > 0 for x in cs)
    p = out["cells"][cell_name(S.PRIMARY)]["paired_eligible"]
    out["verdict"] = "supports" if ok else "contradicts" if p["mean"] < 0 and p["ci"][1] < 0 else "inconclusive"
    return out


def write_wmn_gates() -> dict[str, Any]:
    """Per compact-subset symbol-day: gates/<sym>/<d>.csv (on = not pull) and widen/<sym>/<d>.csv (factor) from the WMN gate;
    each row applies from its ts; the latest row at or before 00:00 UTC opens the day."""
    g = pl.read_parquet(OUT.parents[0] / "wmn" / "gate.parquet").with_columns(pl.col("ts").dt.epoch("ms").cast(pl.Int64).alias("t"))
    info: dict[str, Any] = {"written": 0, "unknown": []}
    days = [x for w in S.KEEP_COMPACT["weekends"] for x in S.wdays(w)]
    for sym in S.KEEP_COMPACT["symbols"]:
        gs = g.filter(pl.col("symbol") == sym).sort("t")
        for d in days:
            a, b = S.day_ms(d), S.day_ms(d) + S.DAY
            pre = gs.filter(pl.col("t") <= a).tail(1)
            rows = pl.concat([pre, gs.filter((pl.col("t") > a) & (pl.col("t") < b))])
            if not rows.height or rows["t"][0] > a:
                info["unknown"].append(f"{sym}:{d}")
                continue
            t = np.maximum(rows["t"].to_numpy(), a)
            for sub, col in (("gates", (~rows["pull"]).cast(pl.Int8)), ("widen", rows["widen"])):
                p = OUT / "wmn_gate" / sub / sym / f"{d}.csv"
                p.parent.mkdir(parents=True, exist_ok=True)
                hdr = "valid_from_ms,on\n" if sub == "gates" else "valid_from_ms,factor\n"
                p.write_text(hdr + "".join(f"{x},{v}\n" for x, v in zip(t, col.to_list(), strict=True)), encoding="utf-8")
            info["written"] += 1
    return info


def rerun_variants(best: str) -> list[dict[str, Any]]:
    cal = json.loads(S.CALIB.read_text(encoding="utf-8"))
    g = {v["name"]: v for v in S.grid(cal, "XLMUSDT", "hold")}  # the per-symbol fields are re-filled per symbol below
    gate = "data/research/wmq/wmn_gate/gates/{sym}/{d}.csv"
    widen = "data/research/wmq/wmn_gate/widen/{sym}/{d}.csv"
    out = []
    for base in (best, "B0"):
        v = dict(g[base])
        out.append({**v, "name": f"{base}__ctl"})
        out.append({**v, "name": f"{base}__pull", "gate": gate, "gate_flatten": False})
        out.append({**v, "name": f"{base}__pullwiden", "gate": gate, "gate_flatten": False, "widen_file": widen})
    v = dict(g[best])
    if v.get("mode") == "model":
        for L in (4, 10):
            out.append({**v, "name": f"{best}__L{L}", "lots": L, "cap_usd": 25.0 * L,
                        "overrides": {**v.get("overrides", {}), "symbol_loss_limit_usd": 1.5 * L, "loss_limit_usd": 10.0 * L}})
    return out


def run_reruns(best: str) -> None:
    """Gate and capacity arms on the stored compact subset, with per-symbol parameters from the calibration."""
    cal = json.loads(S.CALIB.read_text(encoding="utf-8"))
    tmpl = rerun_variants(best)
    for sym in S.KEEP_COMPACT["symbols"]:
        g = {v["name"]: v for v in S.grid(cal, sym, "hold")}
        vs = []
        for v in tmpl:
            base = v["name"].split("__")[0]
            vs.append({**g[base], **{k: x for k, x in v.items() if k not in g[base] or k in ("name", "gate", "gate_flatten", "widen_file", "lots", "cap_usd", "overrides")}})
        S.rerun_compact(f"cond_{best}", vs, symbols=[sym])


def load_reruns(tag: str) -> pl.DataFrame:
    dfs = []
    root = OUT / "reruns" / tag
    for p in sorted(root.glob("*/*/quoters.parquet")):
        sym, d = p.parent.parent.name, p.parent.name
        q = pl.read_parquet(p)
        fu = funding_by_quoter_dir(p.parent, sym, d)
        dfs.append(q.join(fu, on=["variant", "queue", "latency"], how="left").with_columns(pl.col("funding").fill_null(0.0)))
    if not dfs:
        return pl.DataFrame()
    df = pl.concat(dfs, how="diagonal_relaxed").with_columns(pl.lit("hold").alias("phase"), (pl.col("net") + pl.col("funding")).alias("net_f"))
    return df.with_columns(pl.col("net").alias("net_lot"))


def funding_by_quoter_dir(dd: Path, sym: str, d: str) -> pl.DataFrame:
    f = dd / "fills.parquet"
    rows = [(t, r) for t, r in S.funding(sym) if S.day_ms(d) < t < S.day_ms(d) + S.DAY]
    if not f.is_file() or not rows:
        return pl.DataFrame(schema={"variant": pl.Utf8, "queue": pl.Utf8, "latency": pl.Int64, "funding": pl.Float64})
    bars = pl.read_csv(S.day_dir(sym, d) / "bars_1m.csv")
    bm, bc = bars["minute_ms"].to_numpy(), bars["close"].to_numpy()
    fl = pl.read_parquet(f, columns=["variant", "queue", "latency", "ts", "pos_after"]).sort("ts")
    out = []
    for (v, q, lat), g in fl.group_by(["variant", "queue", "latency"]):
        ts, pos = g["ts"].to_numpy(), g["pos_after"].to_numpy()
        tot = 0.0
        for t, rate in rows:
            i = np.searchsorted(ts, t, side="left") - 1
            if i >= 0:
                tot += -pos[i] * bc[max(np.searchsorted(bm, t, side="left") - 1, 0)] * rate
        out.append({"variant": v, "queue": q, "latency": int(lat), "funding": tot})
    return pl.DataFrame(out, schema={"variant": pl.Utf8, "queue": pl.Utf8, "latency": pl.Int64, "funding": pl.Float64})


def cond_eval(best: str, main_df: pl.DataFrame) -> dict[str, Any]:
    tag = f"cond_{best}"
    rr = load_reruns(tag)
    res: dict[str, Any] = {"variant": best, "n_rows": rr.height}
    if not rr.height:
        return res
    # consistency: the re-run control equals the main HOLD run on the same symbol-days
    ctl = cell_df(rr, S.PRIMARY).filter(pl.col("variant") == f"{best}__ctl").select("symbol", "date", pl.col("net").alias("r"))
    mainp = cell_df(main_df, S.PRIMARY).filter((pl.col("variant") == best) & (pl.col("phase") == "hold")).select("symbol", "date", pl.col("net").alias("m"))
    j = ctl.join(mainp, on=["symbol", "date"])
    res["control_matches_main"] = {"n": j.height, "max_abs_diff": float((j["r"] - j["m"]).abs().max()) if j.height else float("nan")}
    for h, arm in (("H-WMQ-12", "pullwiden"), ("H-WMQ-15", "pull")):
        p = paired(rr, f"{best}__{arm}", f"{best}__ctl", col="net")
        pb = paired(rr, f"B0__{arm}", "B0__ctl", col="net")
        res[h] = {"paired": p, "b0_paired": pb, "verdict": verdict_paired(p), "arm": arm}
    for L in (4, 10):
        n = f"{best}__L{L}"
        if rr.filter(pl.col("variant") == n).height:
            res[f"L{L}"] = held_eval(rr.with_columns(*[pl.col(k).alias(f"{k}_lot") for k in PNL if k in rr.columns]), n, col="net")
            res[f"L{L}_per_lot"] = {c: res[f"L{L}"]["cells"][c]["mean"] / L for c in res[f"L{L}"]["cells"]}
    res["ctl_subset"] = held_eval(rr.with_columns(*[pl.col(k).alias(f"{k}_lot") for k in PNL if k in rr.columns]), f"{best}__ctl", col="net")
    return res


def render_cond(held: dict[str, Any], cond: dict[str, Any]) -> str:
    L: list[str] = []
    h = held.get("H-WMQ-14")
    if h:
        L += ["## Conditional test on R-WMS (H-WMQ-14)", "", f"Variant {h['variant']} vs B0 on HOLD symbol-days with eligible_name (top third of S1); "
              f"{h['cells'][cell_name(S.PRIMARY)]['n_eligible']} of {h['cells'][cell_name(S.PRIMARY)]['n_all']} symbol-days eligible, "
              f"{h['cells'][cell_name(S.PRIMARY)]['missing_rows']} without a selection row. Verdict: **{h['verdict']}**.", "",
              "| cell | paired (variant - B0) on eligible | variant net on eligible | B0 on eligible | eligible - rest (paired) | gate-on paired (3 weekends) |", "|---|---|---|---|---|---|"]
        for c in S.CELLS:
            r = h["cells"][cell_name(c)]
            d = r["diff_eligible_minus_rest"]
            L.append(f"| {cell_name(c)} | {ci(r['paired_eligible'])} | {ci(r['net_eligible'])} | {ci(r['b0_eligible'])} | {fmt(d['diff'])} [{fmt(d['ci'][0])}, {fmt(d['ci'][1])}] | {ci(r['paired_gate_on'])} (n {r['paired_gate_on']['n']}) |")
        p = h["cells"][cell_name(S.PRIMARY)]
        L += ["", f"Score as a continuous moderator (primary cell): slope of the paired difference on S1 {fmt(p['slope_paired_on_score']['slope'])} "
              f"[{fmt(p['slope_paired_on_score']['ci'][0])}, {fmt(p['slope_paired_on_score']['ci'][1])}], of the variant's net {fmt(p['slope_net_on_score']['slope'])} "
              f"[{fmt(p['slope_net_on_score']['ci'][0])}, {fmt(p['slope_net_on_score']['ci'][1])}] USD per symbol-day per unit of S1.", ""]
    if cond and cond.get("n_rows"):
        L += ["## WMN news gate (H-WMQ-12 pull + widen, H-WMQ-15 pull only) and size (H-WMQ-11) on the compact subset", "",
              f"7 core names x the last 8 HOLD weekends; the re-run control equals the main HOLD run (max |diff| {fmt(cond['control_matches_main']['max_abs_diff'], 9)} over {cond['control_matches_main']['n']} symbol-days).", "",
              "| test | arm | " + " | ".join(cell_name(c) for c in S.CELLS) + " | verdict |", "|---|---|" + "---|" * len(S.CELLS) + "---|"]
        for hh in ("H-WMQ-12", "H-WMQ-15"):
            r = cond[hh]
            L.append(f"| {hh} | {cond['variant']} {r['arm']} - ungated | " + " | ".join(ci(r["paired"]["cells"][cell_name(c)]) for c in S.CELLS) + f" | {r['verdict']} |")
            L.append(f"| {hh} (B0, not scored) | B0 {r['arm']} - ungated | " + " | ".join(ci(r["b0_paired"]["cells"][cell_name(c)]) for c in S.CELLS) + " | |")
        L += ["", "| size | " + " | ".join(cell_name(c) for c in S.CELLS) + " | per base lot (primary) |", "|---|" + "---|" * len(S.CELLS) + "---|"]
        L.append("| 1 lot (subset) | " + " | ".join(ci(cond["ctl_subset"]["cells"][cell_name(c)]) for c in S.CELLS) + " | |")
        for Lx in (4, 10):
            if f"L{Lx}" in cond:
                L.append(f"| {Lx} lots | " + " | ".join(ci(cond[f'L{Lx}']["cells"][cell_name(c)]) for c in S.CELLS) + f" | {fmt(cond[f'L{Lx}_per_lot'][cell_name(S.PRIMARY)])} |")
        L.append("")
    if held.get("portfolio"):
        L += ["## The book on one clock (H-WMQ-11 portfolio, HOLD, 12 names)", "", "| variant | cell | caps net / gross USD | mean per day [95% CI] | worst day | best day | dropped fills | max names with inventory | max gross USD |", "|---|---|---|---|---|---|---|---|---|"]
        for nm, key in ((held["H-WMQ-10"]["variant"], "portfolio"), ("B0", "portfolio_B0")):
            for c in S.CELLS:
                r = held[key][cell_name(c)]
                L.append(f"| {nm} | {cell_name(c)} | {r['net_cap']:g} / {r['gross_cap']:g} | {ci(r['per_day'])} | {fmt(r['worst_day'], 3)} | {fmt(r['best_day'], 3)} | {r['drop_share']:.2f} | {r['max_names']} | {r['max_gross']:.0f} |")
        L.append("")
    return "\n".join(L) + "\n"


def render_tables(held: dict[str, Any], sel: dict[str, Any], cal: dict[str, Any], extra: dict[str, Any]) -> str:
    """The generated part of the report (dossiers): verdict table, probe dossier, selection, decomposition, paired tests."""
    L: list[str] = []
    L += ["## Verdict table (held = HOLD, 21 weekends 2026-07-04 .. 11-22, 12 names, clean_holdout)", "",
          "| hypothesis | verdict | variant | primary-cell held mean [95% CI] | SEL mean (primary) |", "|---|---|---|---|---|"]
    for h in ("H-WMQ-1", "H-WMQ-2", "H-WMQ-3", "H-WMQ-4", "H-WMQ-5", "H-WMQ-6", "H-WMQ-7", "H-WMQ-8", "H-WMQ-9", "H-WMQ-10"):
        r = held.get(h, {})
        if h == "H-WMQ-1":
            p = r["cells"][cell_name(S.PRIMARY)]["e60"]
            L.append(f"| {h} | {r['verdict']} | probe at x* | e60 {ci(p, 2)} bp per fill | {fmt(r['sel'][cell_name(S.PRIMARY)]['e60']['mean'], 2)} bp |")
        elif h == "H-WMQ-2":
            ho = r["phases"].get("hold", {})
            L.append(f"| {h} | {r['verdict']} | exp in sigma | log-lik vs bp {ci(ho.get('vs_bps', {'mean': float('nan'), 'ci': [float('nan')] * 2}), 1)}, vs power {ci(ho.get('vs_power', {'mean': float('nan'), 'ci': [float('nan')] * 2}), 1)} per day | stable k: {len(r['stable_symbols'])} / 12 |")
        elif "paired" in r:
            L.append(f"| {h} | {r['verdict']} | {r['variant']} - {r['control']} | {ci(r['paired']['cells'][cell_name(S.PRIMARY)])} | |")
        elif "paired_D_m1" in r:
            L.append(f"| {h} | {r['verdict']} | on D_m1 / A_glft_s0.5_p | {ci(r['paired_D_m1']['cells'][cell_name(S.PRIMARY)])} / {ci(r['paired_A_glft_s0.5_p']['cells'][cell_name(S.PRIMARY)])} | |")
        else:
            L.append(f"| {h} | {r['verdict']} | {r['variant']} | {ci(r['held']['cells'][cell_name(S.PRIMARY)])} | {fmt(r['sel_mean'])} |")
    L += ["", "USD per symbol-day per base lot (minimum lot ~5 USD; the 3-lot families divided by 3) unless stated.", ""]
    L += ["### Held mean net in every cell", ""]
    rows = [("baseline", "B0", held["B0"]["cells"])]
    for h in ("H-WMQ-3", "H-WMQ-6", "H-WMQ-10"):
        rows.append((h, held[h]["variant"], held[h]["held"]["cells"]))
    for h in ("H-WMQ-4", "H-WMQ-5", "H-WMQ-7"):
        rows.append((h, held[h]["variant"], held[h]["held"]["cells"]))
        rows.append((h + " control", held[h]["control"], held[h]["held_control"]["cells"]))
    L += t_cells(rows)
    L += ["", "### Paired tests (held, per symbol-day)", ""]
    prow = [("H-WMQ-10 vs B0", held["H-WMQ-10"]["variant"] + " - B0", held["H-WMQ-10"]["vs_B0"]["cells"]),
            ("H-WMQ-3 vs B0", held["H-WMQ-3"]["variant"] + " - B0", held["H-WMQ-3"]["vs_B0"]["cells"]),
            ("H-WMQ-6 vs B0", held["H-WMQ-6"]["variant"] + " - B0", held["H-WMQ-6"]["vs_B0"]["cells"])]
    for h in ("H-WMQ-4", "H-WMQ-5", "H-WMQ-7"):
        prow.append((h, f"{held[h]['variant']} - {held[h]['control']}", held[h]["paired"]["cells"]))
    for h in ("H-WMQ-8", "H-WMQ-9"):
        prow.append((h, "on D_m1", held[h]["paired_D_m1"]["cells"]))
        prow.append((h, "on A_glft_s0.5_p", held[h]["paired_A_glft_s0.5_p"]["cells"]))
    L += t_cells(prow)
    L += ["", "### Decomposition (held, USD per symbol-day per base lot)", ""]
    items = [("B0", held["B0"]), (held["H-WMQ-10"]["variant"], held["H-WMQ-10"]["held"])]
    for h in ("H-WMQ-3", "H-WMQ-6"):
        if held[h]["variant"] != held["H-WMQ-10"]["variant"]:
            items.append((held[h]["variant"], held[h]["held"]))
    L += t_decomp(items)
    L += ["", "net = spread + adverse 60 s + inventory - fees (each day starts flat, marked at 24:00 UTC); adverse 1 / 10 / 300 s are "
          "markouts from the fill mid; funding = -position x mark x rate at funding times inside the day; net_f = net + funding; "
          "closed = net - |end inventory| x (5.5 bp + half the snapshot spread).", ""]
    L += ["## Selection on SEL (11 weekends 2026-04-18 .. 06-28, llm_in_sample)", "",
          "| family | candidates | choice | CV score (median of 165 splits, worst cell) | procedure OOS (worst cell) | SEL mean primary | PBO | deflated Sharpe | train-pick counts |",
          "|---|---|---|---|---|---|---|---|---|"]
    for f, v in sel["families"].items():
        L.append(f"| {f} | {len(v['names'])} | {v['choice']} | {fmt(v['choice_score'])} | {fmt(v['procedure_oos_mean'])} | {fmt(v['sel_mean_primary'])} | "
                 f"{fmt(v['pbo'].get('pbo'), 2)} | {fmt(v['dsr'].get('dsr'), 3)} | {', '.join(f'{k} {n}' for k, n in sorted(v['train_pick_counts'].items(), key=lambda z: -z[1])[:4])} |")
    L += ["", "### Every variant, held primary cell (descriptive; only the registered choices are tested)", "", "| variant | held mean [95% CI] | n |", "|---|---|---|"]
    for n, b in sorted(held["all_variants_primary"].items(), key=lambda z: -(z[1]["mean"] if math.isfinite(z[1]["mean"]) else -9)):
        L.append(f"| {n} | {ci(b)} | {b['n']} |")
    L += ["", "## Dossier: fill intensity and adverse selection by depth (probe)", ""]
    for ph in ("cal", "sel", "hold"):
        L += t_probe(extra["probe_curves"][ph], ph.upper()) + [""]
    L += ["### Intensity fits per symbol (CAL, primary cell): lambda(x) = A exp(-k x), x in sigma; A per second per resting order-side", "",
          "| symbol | used | A (per s) | k (per sigma) [se] | k_bps (per bp) | power alpha | CAL far fills | x* | k SEL | k HOLD |", "|---|---|---|---|---|---|---|---|---|---|"]
    st = held["H-WMQ-2"]["stability"]
    for s in S.SYMBOLS:
        c = cal["symbols"][s]
        f = c["fit"]
        ks = st.get(s, {})
        L.append(f"| {s} | {c['used']} | {c['params']['A']:.5f} | {c['params']['kappa']:.2f} [{(f['exp_sigma']['se1'] if f else float('nan')):.2f}] | "
                 f"{(f['exp_bps']['k_bps'] if f else float('nan')):.3f} | {(f['power']['alpha'] if f else float('nan')):.2f} | {c['n_far_fills']} | {c['params']['xstar']:g} | "
                 f"{ks.get('sel', {}).get('kappa', float('nan')):.2f} | {ks.get('hold', {}).get('kappa', float('nan')):.2f} |")
    return "\n".join(L) + "\n"


# ----------------------------------------------------------------------------------------------------------------------
# entry points
# ----------------------------------------------------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    a = (argv if argv is not None else sys.argv[1:]) or ["select"]
    if a[0] == "select":
        if SEL_FILE.is_file() and "--force" not in a:
            raise SystemExit("selection.json exists: the selection is frozen")
        df = load_quoters(("sel",))
        res = select(df)
        write_json(SEL_FILE, res)
        for f, v in res["families"].items():
            print(f, v["choice"], round(v["choice_score"], 4), "pbo", v["pbo"].get("pbo"), "dsr", round(v["dsr"].get("dsr", float("nan")), 3),
                  "proc_oos", round(v["procedure_oos_mean"], 4))
    elif a[0] == "held":
        sel = json.loads(SEL_FILE.read_text(encoding="utf-8"))
        df = load_quoters(("sel", "hold"))
        df.write_parquet(OUT / "quoters_all.parquet")
        cal = json.loads(S.CALIB.read_text(encoding="utf-8"))
        res = held_tests(df, sel)
        res["H-WMQ-1"] = h1(cal)
        res["H-WMQ-2"] = h2(cal)
        res["H-WMQ-14"] = h14(df, sel["families"]["ALL"]["choice"])
        bst = sel["families"]["ALL"]["choice"]
        res["portfolio"] = {cell_name(c): portfolio(bst, c, 40.0 * lots_of(bst), 80.0 * lots_of(bst)) for c in S.CELLS}
        res["portfolio_B0"] = {cell_name(c): portfolio("B0", c) for c in S.CELLS}
        write_json(HELD_FILE, res)
        for h in sorted(k for k in res if k.startswith("H-")):
            print(h, res[h].get("verdict"))
    elif a[0] == "tables":
        held = json.loads(HELD_FILE.read_text(encoding="utf-8"))
        sel = json.loads(SEL_FILE.read_text(encoding="utf-8"))
        cal = json.loads(S.CALIB.read_text(encoding="utf-8"))
        cond = json.loads((OUT / "cond.json").read_text(encoding="utf-8")) if (OUT / "cond.json").is_file() else {}
        extra = {"probe_curves": {ph: probe_curves(ph) for ph in ("cal", "sel", "hold")}}
        txt = render_tables(held, sel, cal, extra) + "\n" + render_cond(held, cond)
        (OUT / "tables.md").write_text(txt, encoding="utf-8")
        print(f"wrote {OUT / 'tables.md'} ({len(txt)} chars)")
    elif a[0] == "cond":
        sel = json.loads(SEL_FILE.read_text(encoding="utf-8"))
        best = sel["families"]["ALL"]["choice"]
        print(write_wmn_gates())
        run_reruns(best)
    elif a[0] == "cond_eval":
        sel = json.loads(SEL_FILE.read_text(encoding="utf-8"))
        best = sel["families"]["ALL"]["choice"]
        df = pl.read_parquet(OUT / "quoters_all.parquet")
        res = cond_eval(best, df)
        write_json(OUT / "cond.json", res)
        print({k: (v.get("verdict") if isinstance(v, dict) else v) for k, v in res.items()})
    else:
        raise SystemExit(f"unknown command {a[0]}")


if __name__ == "__main__":
    main()
