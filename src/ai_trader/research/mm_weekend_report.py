"""Analysis of the weekend alt market-making study (agent MMW; H-MMW1 .. H-MMW6): per-family selection on FIT days by
combinatorial purged CV over weekends (worst of four cells), PBO (CSCV over days) and deflated Sharpe, one held-out
evaluation per family with day-clustered bootstraps in every cell (pro-rata / FIFO-pessimistic x 150 / 300 ms), paired
tests, the P&L decomposition, figures, the report and the recorded results. Design: data/research/mm_weekend/split.json
(frozen before any P&L) and src/ai_trader/research/mm_weekend_study.py.

    PYTHONIOENCODING=utf-8 uv run --extra viz python -m ai_trader.research.mm_weekend_report [--no-record]
"""

from __future__ import annotations

import gzip
import json
import math
import sys
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from ai_trader.research.mm_replay_report import deflated_sharpe, pbo_cscv_days, sharpe_cols
from ai_trader.research.mm_weekend_study import (CELLS, DAYS, FIT_DAYS, FIT_WEEKENDS, HELD_DAYS, LIQ_PROXY, MANIFEST, OUT, PRIMARY, ROOT, SPLIT, UNIVERSE,
                                                 WEEKEND_OF, base_grid, grid)

REPORT_MD = ROOT / "research" / "reports" / "mm_weekend_2026-11-21.md"
REPORT_JSON = ROOT / "research" / "reports" / "mm_weekend_2026-11-21.json"
FIG = ROOT / "research" / "figures" / "mm_weekend_2026-11-21"
PNL = ("net", "spread", "adverse_1s", "adverse_10s", "adverse_60s", "inventory", "fees")
B = 10_000
SEED = 20261121
FAMILIES = {"H-MMW1": "W1", "H-MMW2": "W2", "H-MMW3": "W3", "H-MMW4": "W4"}


def cell_name(c: tuple[str, int]) -> str:
    return f"{c[0]}_{c[1]}ms"


# ----------------------------------------------------------------------------------------------------------------------
# load
# ----------------------------------------------------------------------------------------------------------------------


def load_days() -> pl.DataFrame:
    fam = {v["name"]: v["family"] for v in grid()}
    rows = []
    for sym in UNIVERSE:
        for d in DAYS:
            p = OUT / "days" / sym / d / "stats.json"
            if not p.is_file():
                continue
            st = json.loads(p.read_text(encoding="utf-8"))
            for q in st["quoters"]:
                f = q["funnel"]
                rows.append({"symbol": sym, "date": d, "weekend": WEEKEND_OF[d], "split": "fit" if d in FIT_DAYS else "held", "variant": q["name"],
                             "base": q["name"].removesuffix("__G"), "gated": q["name"].endswith("__G"), "family": fam[q["name"]],
                             "queue": q["queue"], "latency": int(q["latency_ms"]), **{k: float(q["pnl"][k]) for k in PNL},
                             "fills": int(q["fills"]), "volume_usd": float(q["volume_usd"]), "stopped": q["stopped"] is not None,
                             "fills_queue": f["fills_queue"], "fills_through": f["fills_through"], "fills_cross": f["fills_cross"], "fills_stop": f["fills_stop"],
                             "placed": f["placed"], "dist_bps": float(f["dist_bps_mean"]) if f["dist_bps_mean"] is not None else float("nan"),
                             "live_share": 0.5 * (f["live_share_bid"] + f["live_share_ask"]), "gate_off_share": f["gate_off_ticks"] / max(f["ticks"], 1),
                             "btc_pulls": f.get("btc_pulls", 0), "liq_pulls": f.get("liq_pulls", 0), "side_pull_ticks": f.get("side_pull_ticks", 0),
                             "touch_skips": f.get("touch_skips", 0), "touch_leaves": f.get("touch_leaves", 0), "ladder_lots": f.get("ladder_lots", 0),
                             "ladder_tp_fills": f.get("ladder_tp_fills", 0), "ladder_stops": f.get("ladder_stops", 0),
                             "ladder_adds_placed": f.get("ladder_adds_placed", 0), "max_inv_usd": float(q["max_inventory_usd"])})
    return pl.DataFrame(rows)


def market_rows() -> list[dict]:
    man = json.loads(MANIFEST.read_text(encoding="utf-8")) if MANIFEST.is_file() else {}
    out = []
    for sym in UNIVERSE:
        for d in DAYS:
            p = OUT / "days" / sym / d / "stats.json"
            if p.is_file():
                st = json.loads(p.read_text(encoding="utf-8"))
                m = st["market"]
                rec = man.get("days", {}).get(f"{sym}:{d}", {})
                out.append({"symbol": sym, "date": d, **{k: m.get(k) for k in ("book_messages", "trades", "rpi_trades", "trades_outside_touch", "engine_rejects",
                                                                              "replayer_check_mismatches", "mid_start", "mid_end", "median_print_notional",
                                                                              "liq_min_notional", "liq_clusters", "btc_prints", "btc_z_gt_2", "btc_z_gt_3", "btc_z_gt_4")},
                            "liq_cluster_ts": m.get("liq_cluster_ts", []), "seconds": st["perf"]["seconds"], "gate_on_share": rec.get("gate", {}).get("on_share"),
                            "raw_bytes": rec.get("raw_bytes"), "tick_override": rec.get("grid_override")})
    return out


# ----------------------------------------------------------------------------------------------------------------------
# statistics
# ----------------------------------------------------------------------------------------------------------------------


def boot(vals: np.ndarray, days: np.ndarray, rng_seed: int = SEED) -> dict[str, Any]:
    """Mean per symbol-day and a day-clustered percentile bootstrap 95% CI (days resampled with all their symbol-days)."""
    v = np.asarray(vals, float)
    if len(v) == 0:
        return {"mean": float("nan"), "ci": [float("nan")] * 2, "n": 0, "n_days": 0}
    u, inv = np.unique(np.asarray(days), return_inverse=True)
    s = np.bincount(inv, weights=v, minlength=len(u))
    c = np.bincount(inv, minlength=len(u)).astype(float)
    rng = np.random.default_rng(rng_seed)
    idx = rng.integers(0, len(u), size=(B, len(u)))
    m = s[idx].sum(1) / c[idx].sum(1)
    lo, hi = np.percentile(m, [2.5, 97.5])
    return {"mean": float(v.mean()), "ci": [float(lo), float(hi)], "n": int(len(v)), "n_days": int(len(u)), "sd": float(v.std(ddof=1)) if len(v) > 1 else float("nan")}


def cell_df(df: pl.DataFrame, cell: tuple[str, int]) -> pl.DataFrame:
    return df.filter((pl.col("queue") == cell[0]) & (pl.col("latency") == cell[1]))


def sd_matrix(df: pl.DataFrame, days: list[str], names: list[str], cell: tuple[str, int]) -> tuple[np.ndarray, np.ndarray]:
    """(symbol-days x variants) net matrix for one cell over `days`, plus each row's date; rows with any variant missing dropped."""
    d = cell_df(df, cell).filter(pl.col("date").is_in(days) & pl.col("variant").is_in(names))
    w = d.pivot(on="variant", index=["symbol", "date"], values="net").sort("date", "symbol")
    for n in names:
        if n not in w.columns:
            w = w.with_columns(pl.lit(None, dtype=pl.Float64).alias(n))
    w = w.drop_nulls(subset=names)
    return w.select(names).to_numpy(), w["date"].to_numpy()


def cv_weekends(df: pl.DataFrame, names: list[str]) -> dict[str, Any]:
    """Purged CV over the FIT weekends: every choice of 2 test weekends; per split, each variant's statistic = min over the
    four cells of its test mean net per symbol-day; score = median over splits; procedure OOS = the train-best's statistic."""
    mats = {c: sd_matrix(df, FIT_DAYS, names, c) for c in CELLS}
    stats, proc, picks = [], [], []
    for test in combinations(FIT_WEEKENDS, 2):
        tr = [w for w in FIT_WEEKENDS if w not in test]
        te_stat, tr_stat = [], []
        for c in CELLS:
            m, dts = mats[c]
            wk = np.array([WEEKEND_OF[x] for x in dts])
            te, trm = np.isin(wk, test), np.isin(wk, tr)
            te_stat.append(m[te].mean(axis=0) if te.any() else np.full(len(names), np.nan))
            tr_stat.append(m[trm].mean(axis=0) if trm.any() else np.full(len(names), np.nan))
        te_min, tr_min = np.min(np.vstack(te_stat), axis=0), np.min(np.vstack(tr_stat), axis=0)
        b = int(np.nanargmax(tr_min)) if np.isfinite(tr_min).any() else 0
        stats.append(te_min)
        proc.append(float(te_min[b]))
        picks.append(b)
    sm = np.vstack(stats)
    score = np.median(sm, axis=0)
    choice = int(np.nanargmax(score)) if np.isfinite(score).any() else 0
    return {"names": names, "score": score.tolist(), "choice": names[choice], "choice_score": float(score[choice]), "n_splits": len(stats),
            "procedure_oos_mean": float(np.mean(proc)), "train_pick_counts": {names[i]: picks.count(i) for i in set(picks)}}


def day_means(df: pl.DataFrame, days: list[str], names: list[str], cell: tuple[str, int]) -> np.ndarray:
    d = cell_df(df, cell).filter(pl.col("date").is_in(days) & pl.col("variant").is_in(names)).group_by("date", "variant").agg(pl.col("net").mean())
    m = np.full((len(days), len(names)), np.nan)
    di, vi = {x: i for i, x in enumerate(days)}, {x: i for i, x in enumerate(names)}
    for r in d.iter_rows(named=True):
        m[di[r["date"]], vi[r["variant"]]] = r["net"]
    return m


def held_eval(df: pl.DataFrame, name: str) -> dict[str, Any]:
    out: dict[str, Any] = {"variant": name, "cells": {}}
    for c in CELLS:
        h = cell_df(df, c).filter((pl.col("variant") == name) & (pl.col("split") == "held"))
        b = boot(h["net"].to_numpy(), h["date"].to_numpy())
        b["decomposition"] = {k: float(h[k].mean()) for k in PNL} if h.height else {}
        b["fills_per_sd"] = float(h["fills"].mean()) if h.height else float("nan")
        b["fills_through_share"] = float(h["fills_through"].sum() / max(h["fills"].sum(), 1)) if h.height else float("nan")
        b["stopped_share"] = float(h["stopped"].mean()) if h.height else float("nan")
        b["pos_share"] = float((h["net"] > 0).mean()) if h.height else float("nan")
        out["cells"][cell_name(c)] = b
    cs = out["cells"].values()
    out["passes"] = all(x["mean"] > 0 and x["ci"][0] > 0 for x in cs)
    p = out["cells"][cell_name(PRIMARY)]
    out["contradicts"] = p["mean"] <= 0 and p["ci"][1] < 0
    return out


def paired(df: pl.DataFrame, a: str, b: str) -> dict[str, Any]:
    out = {"a": a, "b": b, "cells": {}}
    for c in CELLS:
        h = cell_df(df, c).filter(pl.col("split") == "held")
        x = h.filter(pl.col("variant") == a).select("symbol", "date", pl.col("net").alias("na"))
        y = h.filter(pl.col("variant") == b).select("symbol", "date", pl.col("net").alias("nb"))
        j = x.join(y, on=["symbol", "date"])
        r = boot((j["na"] - j["nb"]).to_numpy(), j["date"].to_numpy())
        r.update(mean_a=float(j["na"].mean()) if j.height else float("nan"), mean_b=float(j["nb"].mean()) if j.height else float("nan"))
        out["cells"][cell_name(c)] = r
    return out


def verdict(h: dict[str, Any]) -> str:
    return "supports" if h["passes"] else "contradicts" if h["contradicts"] else "inconclusive"


def family_selection(df: pl.DataFrame, fam: str) -> dict[str, Any]:
    names = [v["name"] for v in base_grid() if v["family"] == fam]
    cv = cv_weekends(df, names)
    m = day_means(df, FIT_DAYS, names, PRIMARY)
    pbo = pbo_cscv_days(m) if len(names) >= 2 else {"pbo": float("nan"), "note": "one candidate"}
    allnames = [v["name"] for v in grid()]
    mall = day_means(df, FIT_DAYS, allnames, PRIMARY)
    x = m[:, names.index(cv["choice"])]
    dsr = deflated_sharpe(x, sharpe_cols(mall), len(allnames))
    return {"cv": cv, "pbo": pbo, "dsr": dsr, "fit_mean_primary": float(np.nanmean(x))}


def h5(df: pl.DataFrame, best: str) -> dict[str, Any]:
    out: dict[str, Any] = {"pairs": {}}
    for base in ("B0", best):
        g = base + "__G"
        p = paired(df, g, base)
        he = held_eval(df, g)
        ok = all(p["cells"][cell_name(c)]["mean"] > 0 and p["cells"][cell_name(c)]["ci"][0] > 0 and he["cells"][cell_name(c)]["mean"] > 0 for c in CELLS)
        bad = he["contradicts"]
        out["pairs"][base] = {"paired": p, "gated": he, "supports": ok, "gated_contradicts": bad}
    pr = list(out["pairs"].values())
    out["verdict"] = "supports" if any(x["supports"] for x in pr) else "contradicts" if all(x["gated_contradicts"] for x in pr) else "inconclusive"
    return out


def gate_desc(df: pl.DataFrame) -> dict[str, Any]:
    g = df.filter(pl.col("gated") & (pl.col("variant") == "B0__G") & (pl.col("queue") == PRIMARY[0]) & (pl.col("latency") == PRIMARY[1]))
    return {"gate_on_share_fit": float(1 - g.filter(pl.col("split") == "fit")["gate_off_share"].mean()) if g.height else float("nan"),
            "gate_on_share_held": float(1 - g.filter(pl.col("split") == "held")["gate_off_share"].mean()) if g.height else float("nan")}


def liq_validation(mk: list[dict]) -> dict[str, Any]:
    """Proxy triggers vs the lake's liquidation prints (WebSocket allLiquidation, from 2026-11-08) on the covered days."""
    from ai_trader.storage.lake import Lake

    try:
        lake = Lake()
        lq = lake.read_asof("liquidations", __import__("datetime").datetime(2026, 11, 17, tzinfo=__import__("datetime").UTC), symbols=UNIVERSE,
                            start=__import__("datetime").datetime(2026, 11, 7, tzinfo=__import__("datetime").UTC))
    except Exception as e:  # noqa: BLE001
        return {"error": repr(e)}
    if lq.height == 0:
        return {"note": "no lake liquidations for the universe"}
    tcol = "event_ts" if "event_ts" in lq.columns else lq.columns[1]
    lq = lq.with_columns(pl.col(tcol).dt.epoch("ms").alias("t"))
    first = int(lq["t"].min())
    rows = []
    for r in mk:
        d0 = int(__import__("datetime").datetime.fromisoformat(r["date"] + "T00:00:00+00:00").timestamp() * 1000)
        if d0 < first:
            continue
        lt = np.sort(lq.filter(pl.col("symbol") == r["symbol"])["t"].to_numpy())
        lt = lt[(lt >= d0) & (lt < d0 + 86_400_000)]
        trig = np.array(r["liq_cluster_ts"], dtype=np.int64)
        hit = [bool(len(lt) and np.min(np.abs(lt - t)) <= 30_000) for t in trig]
        cov = [bool(len(trig) and np.min(np.abs(trig - t)) <= 30_000) for t in lt]
        rows.append({"symbol": r["symbol"], "date": r["date"], "proxy_triggers": len(trig), "lake_liq_prints": len(lt),
                     "precision": float(np.mean(hit)) if hit else float("nan"), "recall_prints": float(np.mean(cov)) if cov else float("nan")})
    return {"rows": rows, "lake_first": first, "time_col": tcol}


# ----------------------------------------------------------------------------------------------------------------------
# H-MMW6 (only if a variant passes)
# ----------------------------------------------------------------------------------------------------------------------


def portfolio(name: str) -> dict[str, Any]:
    """Merged held fills of `name` on one clock per cell with net 40 / gross 80 caps (one resting minimum-lot adding order per
    quoting symbol and side counted as filled). An adding fill that would breach a cap is dropped together with the symbol's
    later fills until it is flat again in the original sequence."""
    out: dict[str, Any] = {"variant": name, "cells": {}}
    for c in CELLS:
        per_day: dict[str, float] = {}
        for d in HELD_DAYS:
            fills = []
            for sym in UNIVERSE:
                p = OUT / "days" / sym / d / "fills.jsonl.gz"
                if not p.is_file():
                    continue
                with gzip.open(p, "rt", encoding="utf-8") as f:
                    for line in f:
                        x = json.loads(line)
                        if x["variant"] == name and x["queue"] == c[0] and x["latency_ms"] == c[1]:
                            fills.append({**x, "symbol": sym})
            fills.sort(key=lambda x: x["ts"])
            pos = {s: 0.0 for s in UNIVERSE}
            orig = {s: 0.0 for s in UNIVERSE}
            skip = {s: False for s in UNIVERSE}
            cash, last_mid = 0.0, {}
            for x in fills:
                s, sg = x["symbol"], (1.0 if x["side"] == "Buy" else -1.0)
                orig[s] += sg * x["qty"]
                last_mid[s] = x["mid"]
                if skip[s]:
                    if abs(orig[s]) < 1e-12:
                        skip[s] = False
                    continue
                adds = abs(pos[s] + sg * x["qty"]) > abs(pos[s]) + 1e-12
                if adds:
                    lot = x["qty"] * x["price"]
                    net = sum(pos[k] * last_mid.get(k, x["mid"]) for k in pos) + sg * lot
                    pend = lot * sum(1 for k in pos if k != s and abs(pos[k]) < 1e-12)
                    gross = sum(abs(pos[k]) * last_mid.get(k, x["mid"]) for k in pos) + lot
                    if abs(net) + pend > 40.0 or gross + pend > 80.0:
                        skip[s] = abs(orig[s]) > 1e-12
                        continue
                pos[s] += sg * x["qty"]
                cash -= sg * x["qty"] * x["price"] + x["fee"]
            mids = {}
            for sym in UNIVERSE:
                p = OUT / "days" / sym / d / "stats.json"
                if p.is_file():
                    mids[sym] = json.loads(p.read_text(encoding="utf-8"))["market"]["mid_end"]
            per_day[d] = cash + sum(pos[s] * mids.get(s, 0.0) for s in pos)
        v = np.array(list(per_day.values()))
        out["cells"][cell_name(c)] = boot(v, np.array(list(per_day.keys())))
    out["passes"] = all(x["mean"] > 0 and x["ci"][0] > 0 for x in out["cells"].values())
    return out


# ----------------------------------------------------------------------------------------------------------------------
# analysis, figures, report
# ----------------------------------------------------------------------------------------------------------------------


def analyse() -> tuple[dict[str, Any], pl.DataFrame]:
    df = load_days()
    mk = market_rows()
    split = json.loads(SPLIT.read_text(encoding="utf-8"))
    res: dict[str, Any] = {"split_written": split["written"], "amendments": split.get("amendments", []), "n_rows": df.height,
                           "coverage": {s: sorted(df.filter(pl.col("symbol") == s)["date"].unique().to_list()) for s in UNIVERSE}, "families": {}}
    for hid, fam in FAMILIES.items():
        sel = family_selection(df, fam)
        he = held_eval(df, sel["cv"]["choice"])
        r = {"selection": sel, "held": he, "verdict": verdict(he)}
        if fam == "W1":
            r["a2_minus_a1"] = paired(df, "W1_A2", "W1_A1")
        if fam == "W2":
            ch = sel["cv"]["choice"]
            r["vs_no_pull"] = paired(df, ch, "B0h" if "_B0h_" in ch else "B0")
        if fam == "W3":
            r["vs_b0"] = paired(df, sel["cv"]["choice"], "B0")
        res["families"][hid] = r
    best_hid = max(FAMILIES, key=lambda h: res["families"][h]["selection"]["cv"]["choice_score"])
    best = res["families"][best_hid]["selection"]["cv"]["choice"]
    res["best_w14"] = {"hypothesis": best_hid, "variant": best}
    res["h5"] = h5(df, best)
    res["baselines"] = {b: held_eval(df, b) for b in ("B0", "B0h")}
    passing = [res["families"][h]["selection"]["cv"]["choice"] for h in FAMILIES if res["families"][h]["held"]["passes"]]
    if res["h5"]["verdict"] == "supports":
        passing += [b + "__G" for b, x in res["h5"]["pairs"].items() if x["supports"]]
    res["passing"] = passing
    res["h6"] = portfolio(passing[0]) if passing else {"note": "not evaluated: no variant passed H-MMW1-5 (pre-registered condition)"}
    res["h6_verdict"] = ("supports" if res["h6"].get("passes") else "inconclusive") if passing else "inconclusive"
    res["gate"] = gate_desc(df)
    res["market"] = mk
    res["liq_validation"] = liq_validation(mk)
    res["family_best_by_cell_fit"] = {}
    return res, df


def _style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.3)
    ax.axhline(0, color="#555", lw=0.8)


def figures(res: dict[str, Any]) -> list[str]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:  # noqa: BLE001
        return []
    FIG.mkdir(parents=True, exist_ok=True)
    out = []
    items = [("B0", res["baselines"]["B0"]), ("B0h", res["baselines"]["B0h"])] + [(f"{h}\n{r['held']['variant']}", r["held"]) for h, r in res["families"].items()]
    items += [(f"H-MMW5\n{k}__G", v["gated"]) for k, v in res["h5"]["pairs"].items()]
    fig, ax = plt.subplots(figsize=(11, 4.2))
    cols = ["#3b6ea8", "#9cbbe0", "#b5523b", "#e3a796"]
    w = 0.2
    for j, c in enumerate(CELLS):
        cn = cell_name(c)
        mu = [x[1]["cells"][cn]["mean"] for x in items]
        lo = [x[1]["cells"][cn]["mean"] - x[1]["cells"][cn]["ci"][0] for x in items]
        hi = [x[1]["cells"][cn]["ci"][1] - x[1]["cells"][cn]["mean"] for x in items]
        ax.bar(np.arange(len(items)) + (j - 1.5) * w, mu, w, yerr=[lo, hi], color=cols[j], label=cn.replace("_", " "), capsize=2)
    ax.set_xticks(np.arange(len(items)), [x[0] for x in items], fontsize=7)
    ax.set_ylabel("held net USD per symbol-day")
    ax.legend(fontsize=7, ncol=4, frameon=False)
    _style(ax)
    fig.tight_layout()
    p = FIG / "held_net.png"
    fig.savefig(p, dpi=130)
    plt.close(fig)
    out.append(p.name)
    fig, ax = plt.subplots(figsize=(11, 4.2))
    comps = ["spread", "adverse_60s", "inventory", "fees"]
    cc = ["#3b6ea8", "#b5523b", "#7a9a3b", "#777"]
    cn = cell_name(PRIMARY)
    bottom_p, bottom_n = np.zeros(len(items)), np.zeros(len(items))
    for k, col in zip(comps, cc, strict=True):
        v = np.array([x[1]["cells"][cn]["decomposition"].get(k, 0.0) * (-1 if k == "fees" else 1) for x in items])
        b = np.where(v >= 0, bottom_p, bottom_n)
        ax.bar(np.arange(len(items)), v, 0.6, bottom=b, color=col, label=k if k != "fees" else "-fees")
        bottom_p += np.where(v >= 0, v, 0)
        bottom_n += np.where(v < 0, v, 0)
    ax.plot(np.arange(len(items)), [x[1]["cells"][cn]["mean"] for x in items], "k_", ms=18, mew=2, label="net")
    ax.set_xticks(np.arange(len(items)), [x[0] for x in items], fontsize=7)
    ax.set_ylabel("USD per symbol-day (held, pro-rata 150 ms)")
    ax.legend(fontsize=7, ncol=5, frameon=False)
    _style(ax)
    fig.tight_layout()
    p = FIG / "decomposition.png"
    fig.savefig(p, dpi=130)
    plt.close(fig)
    out.append(p.name)
    return out


def fmt(x: float, n: int = 4) -> str:
    return "n/a" if x is None or (isinstance(x, float) and not math.isfinite(x)) else f"{x:+.{n}f}"


def ci(b: dict) -> str:
    return f"{fmt(b['mean'])} [{fmt(b['ci'][0])}, {fmt(b['ci'][1])}]"


def record_all(res: dict[str, Any]) -> list[dict]:
    from ai_trader.research.hypotheses import record

    data = "data/research/mm_weekend (split.json, days/, manifest.json); report research/reports/mm_weekend_2026-11-21.md"
    seg = "All 16 days (8 weekends 2026-09-26 .. 11-15) are after research.knowledge_cutoff 2026-07-01: clean_holdout only."
    out = []

    def cells(h):
        return {k: {kk: v[kk] for kk in ("mean", "ci", "n", "n_days", "fills_per_sd", "decomposition")} for k, v in h["cells"].items()}

    for hid, r in res["families"].items():
        m = {"choice": r["held"]["variant"], "held": cells(r["held"]), "passes": r["held"]["passes"], "cv_score": r["selection"]["cv"]["choice_score"],
             "procedure_oos_mean": r["selection"]["cv"]["procedure_oos_mean"], "pbo": r["selection"]["pbo"].get("pbo"), "dsr": r["selection"]["dsr"].get("dsr")}
        for k in ("a2_minus_a1", "vs_no_pull", "vs_b0"):
            if k in r:
                m[k] = {c: {kk: v[kk] for kk in ("mean", "ci", "n")} for c, v in r[k]["cells"].items()}
        out.append(record(hid, "clean_holdout", r["verdict"], m, note=seg, data=data))
    m5 = {b: {"gated": cells(x["gated"]), "gated_minus_ungated": {c: {kk: v[kk] for kk in ("mean", "ci", "n")} for c, v in x["paired"]["cells"].items()},
              "supports": x["supports"]} for b, x in res["h5"]["pairs"].items()}
    m5["gate"] = res["gate"]
    out.append(record("H-MMW5", "clean_holdout", res["h5"]["verdict"], m5, note=seg + " Gate J = BTC field P(jump) (BTC-only) as the market jump state.", data=data))
    m6 = res["h6"] if res["passing"] else {"evaluated": False}
    out.append(record("H-MMW6", "clean_holdout", res["h6_verdict"], m6, note=seg + (" Not evaluated: no variant passed H-MMW1-5." if not res["passing"] else ""), data=data))
    return out


def decomp_rows(items: list[tuple[str, dict]]) -> list[str]:
    L = ["| hypothesis | variant | cell | net [95% CI] | spread | adverse 1 s | adverse 10 s | adverse 60 s | inventory | fees | fills / symbol-day | through share |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for lab, h in items:
        for c in CELLS:
            b = h["cells"][cell_name(c)]
            dd = b["decomposition"]
            if not dd:
                continue
            L.append(f"| {lab} | {h['variant']} | {c[0]} {c[1]} ms | {ci(b)} | {fmt(dd['spread'])} | {fmt(dd['adverse_1s'])} | {fmt(dd['adverse_10s'])} | "
                     f"{fmt(dd['adverse_60s'])} | {fmt(dd['inventory'])} | {fmt(-dd['fees'])} | {b['fills_per_sd']:.1f} | {b['fills_through_share']:.2f} |")
    return L


def render_md(res: dict[str, Any], figs: list[str], extra: str = "") -> str:
    fams = res["families"]
    L = ["# Weekend alt market making on replayed Bybit books (H-MMW1 .. H-MMW6), 2026-11-21", "",
         "Agent MMW. Research only: no live orders, no Bybit trading or private calls. The quoting code is the live `atx mm` code plus the weekend "
         "rules added to crates/atx/src/mm.rs; the market is the real ob200 book and public trade tape of each symbol-day (plus BTCUSDT's tape for "
         "the BTC pull), replayed by `atx mm-replay`. Every day is after the knowledge cutoff (2026-07-01): every result is `clean_holdout`.", "",
         "## Verdicts", ""]
    for hid, r in fams.items():
        L.append(f"- **{hid}: {r['verdict']}** (chosen on FIT: `{r['held']['variant']}`).")
    L.append(f"- **H-MMW5: {res['h5']['verdict']}** (pairs B0 and `{res['best_w14']['variant']}`).")
    L.append(f"- **H-MMW6: {res['h6_verdict']}** ({'evaluated for ' + res['passing'][0] if res['passing'] else 'not evaluated: no variant passed H-MMW1-5'}).")
    L += ["", "Held mean net USD per symbol-day [day-clustered 95% CI], in each cell:", "",
          "| hypothesis | variant | pro-rata 150 ms | pro-rata 300 ms | FIFO-pess. 150 ms | FIFO-pess. 300 ms | passes |", "|---|---|---|---|---|---|---|"]
    rows = [(h, r["held"]) for h, r in fams.items()] + [("H-MMW5 gated", x["gated"]) for x in res["h5"]["pairs"].values()] + [("baseline", res["baselines"][b]) for b in ("B0", "B0h")]
    for lab, h in rows:
        cs = [ci(h["cells"][cell_name(c)]) for c in (("prorata", 150), ("prorata", 300), ("pessimistic", 150), ("pessimistic", 300))]
        L.append(f"| {lab} | {h['variant']} | " + " | ".join(cs) + f" | {'yes' if h['passes'] else 'no'} |")
    L += ["", extra, "", "## Selection on FIT days and overfitting", "",
          "| hypothesis | choice | CV score (median split, worst cell) | procedure OOS (worst cell) | FIT mean (pro-rata 150) | PBO | deflated Sharpe | candidates |", "|---|---|---|---|---|---|---|---|"]
    for hid, r in fams.items():
        s = r["selection"]
        L.append(f"| {hid} | {s['cv']['choice']} | {fmt(s['cv']['choice_score'])} | {fmt(s['cv']['procedure_oos_mean'])} | {fmt(s['fit_mean_primary'])} | "
                 f"{fmt(s['pbo'].get('pbo', float('nan')), 2)} | {fmt(s['dsr'].get('dsr', float('nan')), 3)} | {len(s['cv']['names'])} |")
    L += ["", "## Paired tests (held, per symbol-day)", "", "| test | a - b | pro-rata 150 ms | pro-rata 300 ms | FIFO-pess. 150 ms | FIFO-pess. 300 ms |", "|---|---|---|---|---|---|"]
    pts = []
    for hid, key in (("H-MMW1", "a2_minus_a1"), ("H-MMW2", "vs_no_pull"), ("H-MMW3", "vs_b0")):
        if key in fams[hid]:
            pts.append((hid, fams[hid][key]))
    for b, x in res["h5"]["pairs"].items():
        pts.append(("H-MMW5", x["paired"]))
    for lab, p in pts:
        cs = [ci(p["cells"][cell_name(c)]) for c in (("prorata", 150), ("prorata", 300), ("pessimistic", 150), ("pessimistic", 300))]
        L.append(f"| {lab} | {p['a']} - {p['b']} | " + " | ".join(cs) + " |")
    items = [(h, r["held"]) for h, r in fams.items()] + [("H-MMW5", x["gated"]) for x in res["h5"]["pairs"].values()] + [("baseline", res["baselines"]["B0"])]
    L += ["", "## P&L decomposition (held, USD per symbol-day)", ""] + decomp_rows(items)
    L += ["", "net = spread + adverse 60 s + inventory - fees; adverse 1 s and 10 s are the earlier parts of adverse 60 s; through share = fills "
          "from trades through our price.", ""]
    if figs:
        L += ["## Figures", ""] + [f"![{f[:-4]}](../../research/figures/mm_weekend_2026-11-21/{f})" for f in figs] + [""]
    return "\n".join(L)


# Written after the held evaluation, from the numbers in research/reports/mm_weekend_2026-11-21.json.
EXTRA = r"""## In plain words

- **Nothing here earns real money.** At the venue minimum lot every variant makes or loses cents per symbol-day; the best held result is about +0.8 cent. The question was the sign and the mechanism.
- **H-MMW2 (pull alt quotes when BTC jumps) passes the frozen rule, but not for the registered reason.** The FIT-chosen `W2_B0_k2_c10` (B0 width; pull when |BTC 1 s or 5 s return| > 2 x the BTC 1h sigma scaled to the horizon; 10 s cooldown) has held means of +0.0066 to +0.0085 USD per symbol-day with day-clustered 95% CIs above 0 in all four cells. But (a) it is **worse than B0 without the pull** (paired -0.002 to -0.011): the pull mostly removes fills; (b) spread minus adverse selection minus fees is about zero (+0.0001 USD per symbol-day excluding inventory, FIFO-pessimistic 300 ms), so the net is the **inventory mark** of small positions held to 24:00; (c) only 19 of 56 held symbol-days have a fill, 3 are negative, and one weekend (11-07/08) has 43 of its 73 fills. The CI is above zero because most days are exactly zero and a few are positive. Read it as "the pull makes a nearly idle quoter more idle without losing", not as an edge from avoiding toxic fills.
- **The BTC trigger is much busier than intended:** z > 2 on 45% of BTC prints (z > 4 on 26%). 1 s returns are far from Gaussian and prints cluster in moving seconds.
- **H-MMW4 (join only short queues at the touch) loses clearly: contradicts.** About 520-600 fills per symbol-day, -1.11 to -1.25 USD per symbol-day. Adverse selection inside **1 s** (-0.50 to -0.72) is ten times the spread captured (+0.06 pro-rata, negative under FIFO-pessimistic); fees (-0.60 to -0.66) come on top. This repeats R-MMR-1 on these alts.
- **H-MMW1 (level ladders) never really traded: inconclusive.** The zones of scripts/levels.py (top 8 by score, recency weighted) almost never sit 1.5-3.5 x the 1h sigma from the mid: A1 placed 9233 resting level orders over 112 symbol-days and filled **one** lot (zero on held days). A2's liquidation pull is on nearly all day (caveats), so A2 - A1 is exactly zero and says nothing about R-FLOW-2.
- **H-MMW3 (pull or widen the threatened side) is level with B0: inconclusive.** `W3_flow30_t0.3_widen` +0.007 to +0.009 USD per symbol-day, CIs straddle zero; paired vs B0 -0.002 to -0.011 (CIs straddle zero).
- **H-MMW5 (jump gate) makes the quoter smaller, not better: inconclusive.** The gate is on about 30% of the time. Gated B0 has a held mean above 0 with the CI above 0 in every cell (+0.003), but gated minus ungated is negative (-0.006 to -0.017; one CI below 0), so the registered condition fails; the same for gated W3. As on weekdays (R-MMR-5), the gate mostly stops trading.
- **B0 (the live mm_illiq20 width) is slightly positive but not significant** in three of four cells (+0.009 to +0.020 USD per symbol-day, about 4 fills), consistent with R-MMR-2.
- **H-MMW6 (portfolio caps) was evaluated because H-MMW2 passed: inconclusive.** With net 40 / gross 80 USD caps on one clock, `W2_B0_k2_c10` makes +0.034 to +0.047 USD per portfolio day; the CI is above 0 under pro-rata but not under FIFO-pessimistic (lower bounds -0.003 and -0.002).

## Caveats and what went wrong

- **The liquidation-cluster proxy does not discriminate.** It was declared before results, because the lake's liquidation stream starts 2026-11-08, after the FIT days: >= 5 same-side prints in 30 s over >= 3 levels, with summed notional >= 10 x the previous day's median print. Median prints are 10-30 USD with heavy tails, so it fired a median of about 1700 times per symbol-day. Against the lake's liquidation prints on 2026-11-14/15 (7 symbols) it catches 94-99% of them, but only 9-27% of its triggers fall within 30 s of a real one. The smoke run showed this before any P&L was computed (split.json `amendments`, diary). The frozen proxy was kept, so **A2 and the R-FLOW-2 check are uninformative**, not negative.
- **The ladder is limited by the levels logic.** Only the top 8 zones by score are kept, and they sit far from the price: on 09-26 all 8 XLM zones were at least 5% below it, while 3.5 sigma is about 2%. A ladder built on all clusters or on 1h pivots would quote, but it was not registered and was not run.
- **The H-MMW2 pass rests on 8 day-clusters and on inventory P&L.** The W2 selection has PBO 0.41, the purged-CV estimate of the selection procedure is -0.033 USD per symbol-day (worst cell), and the deflated Sharpe is 0.05. The training-best pick changes across splits (k2_c10, k2_c30, k3_c10, B0h_k3_c60).
- **Reading of "field P(jump)":** the field is BTC-only, so for H-MMW5 its P(jump) serves as the market-wide jump state for every alt. Its FIT tercile and median are almost equal (0.01583 vs 0.01597), so the hysteresis barely matters.
- **Clarifications made before results** (split.json): ladder quotes sit at the zone edge nearest the mid; the take-profit is never less than one tick beyond the entry; sigma_risk is the quoter's unshrunk 1h sigma; selection uses the worst of the four cells.
- **Not modelled:** Bybit internals (hidden priority, internal latency), our own impact, funding, the live link's stale tops, and structural zones in the B0 quoter. Each day starts flat and is marked to the mid at 24:00 UTC. About 21% of prints (median symbol-day) fall outside the pre-trade touch, so queue positions are approximate.
- **One re-download:** BTCUSDT 2026-09-26 trades were downloaded twice (about 25 MB; the smoke run had deleted them; logged in the manifest `redownloads`).
- **Small money, small sample:** 4 held weekends, minimum order about 5 USD.
"""

TAIL = r"""## Data

- 7 symbols (research/mm_weekend/universe.json) x the Sat + Sun of 8 weekends 2026-09-26 .. 11-15 = 112 symbol-days, all verified: 0 engine rejects, replayer book equal to the engine at every check, ledger identity < 1e-6 for all 296 quoters, BTC tape present. FIT 09-26 .. 10-18, HELD 10-24 .. 11-15. split.json was written 2026-11-21 01:16 UTC, before any P&L.
- Sources: quote-saver.bycsi.com ob200 and public.bybit.com trades (each alt plus BTCUSDT), processed one day at a time and deleted (manifest data/research/mm_weekend/manifest.json; 0 raw files left). Zones come from Bybit public 4h klines (cached). The first ob200 file (XLMUSDT 2026-11-15) was 20 MB; the largest was about 40 MB.
- Storage: 9.70 GB before (lake 2.82, research 6.48, raw 0.40), 9.90 GB after (lake 2.84, research 6.66, raw 0.40). The study's outputs are about 190 MB; peak transient raw stayed below 0.2 GB.
- Runtime: 61 min for 112 symbol-days (release build, single-threaded, 296 quoters per symbol-day), plus 12 min for the release build.

## Method

- **Engine:** `atx mm-replay` (private release copy data/research/mm_weekend/bin/atx-mmw.exe, built from commit 7254e26). It runs the live `atx mm` decision code with config/mm_illiq20.yaml for every variant, plus the weekend rules as pure, unit-tested functions in crates/atx/src/mm.rs: `ladder_levels`, `level_stop`, `ladder_tp`, `BtcTape` / `btc_move_z`, `top_imbalance`, `threatened_side`, `side_pull`, `short_queue`, `behind_thinned`, `LiqProxy`. Orders are virtual minimum lots.
- **Cells:** pro-rata and FIFO-pessimistic queue models x 150 and 300 ms latency. Orders, cancels, stops and pulls take effect one latency after they are sent. Fees: maker 2.0 bp, taker 5.5 bp.
- **Grid:** 37 variants, each also gated, so 74. B0 (ask 0.45 / bid 0.90 x the 1h sigma) and B0h (0.5 x B0); W1 ladder A1 / A2; W2 width {B0, B0h} x k {2, 3, 4} x cooldown {10, 30, 60} s; W3 {imb5, flow30} x {0.3, 0.5, 0.7} x {pull, widen 1 sigma}; W4 q {0.25, 0.5, 1}. Full definitions are in split.json `design`.
- **Selection:** FIT days only. Purged CV over the 4 FIT weekends (2 test weekends per split, 6 splits). A variant's split statistic is the worst of the four cells' test mean; its score is the median over splits. Best H-MMW1-4 = the family choice with the highest score (W3). PBO by CSCV over the 8 FIT days; deflated Sharpe with N = 74.
- **Tests:** held mean net USD per symbol-day with a day-clustered bootstrap 95% CI (B = 10000, seed 20261121). Pass = mean > 0 and CI above 0 in all four cells. Contradicts = mean <= 0 with the CI below 0 in the pro-rata 150 ms cell. H-MMW6: the passing variant's held fills on all 7 symbols merged on one clock with net 40 / gross 80 USD caps (one resting minimum-lot adding order per quoting symbol and side counted as filled; a blocked adding fill is dropped together with that symbol's later fills until it is flat again).
- **Segment:** every day is after research.knowledge_cutoff (2026-07-01): clean_holdout.

## Rule candidates (suggestions only; research/rules.yaml not edited)

```yaml
- id: R-MMW-1 (candidate)
  kind: avoid
  rule: "WHEN quoting minimum lots at the touch on thin Bybit alt perps (spread 4-12 bp) at crypto maker fees THEN avoid it, even when joining only short queues and leaving when the queue behind thins"
  effect: "held weekends 10-24..11-15, 7 alts: W4_q0.25 -1.11 to -1.25 USD per symbol-day in every cell, CI far below 0; adverse 1 s -0.50 to -0.72 vs spread +0.06 / -0.12, fees -0.60 to -0.66"
  confidence: B
  evidence: [H-MMW4, H-MMR1]
- id: R-MMW-2 (candidate)
  kind: expect
  rule: "WHEN the live-width alt quoter adds a BTC-move pull or a jump gate THEN expect it to trade less, not better: net within a cent of zero and below the unpulled quoter"
  effect: "held: W2_B0_k2_c10 +0.007 to +0.009 USD per symbol-day (CI above 0, mostly inventory marks, 19 of 56 symbol-days with a fill), paired vs B0 -0.002 to -0.011; gated B0 vs B0 -0.006 to -0.017"
  confidence: C
  evidence: [H-MMW2, H-MMW5, R-MMR-5]
  review: "more weekends; measure the markouts of the fills the pull removes"
- id: R-MMW-3 (candidate)
  kind: expect
  rule: "WHEN detecting liquidation clusters from Bybit's public tape with a count / levels / 10 x median-print rule THEN expect near-continuous triggers on thin alts; use the allLiquidation stream instead"
  effect: "median ~1700 triggers per symbol-day; vs lake liquidations 11-14/15: recall 94-99%, precision 9-27%"
  confidence: B
  evidence: [H-MMW1]
```
"""


def main(argv: list[str] | None = None) -> None:
    a = argv if argv is not None else sys.argv[1:]
    res, df = analyse()
    figs = figures(res)
    slim = {**res, "market": [{k: v for k, v in r.items() if k != "liq_cluster_ts"} for r in res["market"]]}  # trigger times stay in stats.json
    REPORT_JSON.write_text(json.dumps(slim, indent=1, default=float), encoding="utf-8")
    md = render_md(res, figs, EXTRA) + chr(10) + TAIL
    REPORT_MD.write_text(md, encoding="utf-8")
    print(f"wrote {REPORT_MD} and {REPORT_JSON}")
    if "--no-record" not in a:
        for r in record_all(res):
            print("recorded", r["id"], r["verdict"])


if __name__ == "__main__":
    main()
