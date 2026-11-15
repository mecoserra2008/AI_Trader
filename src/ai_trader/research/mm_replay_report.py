"""Analysis of the market-making replay study (agent M; H-MMR0, H-MMR1, H-MMR2): selection on fit days by combinatorial
purged CV over days, PBO (CSCV) and deflated Sharpe over the grid, held-out verdicts with day-clustered bootstraps, the
live-fill comparison, the P&L decomposition, figures, the report and the recorded results. Design: split.json (frozen
before any P&L) and src/ai_trader/research/mm_replay_study.py.

    PYTHONIOENCODING=utf-8 uv run --extra viz python -m ai_trader.research.mm_replay_report [--no-record]
"""

from __future__ import annotations

import gzip
import json
import math
import sys
from itertools import combinations
from pathlib import Path
from statistics import NormalDist
from typing import Any

import numpy as np
import polars as pl

from ai_trader.research.mm_replay_study import DAYS, FIT_DAYS, GROUP_OF, GROUPS, HELD_DAYS, LIVE_DAYS, LIVE_SESSIONS, MANIFEST, OUT, ROOT, SPLIT, grid, ms

REPORT_MD = ROOT / "research" / "reports" / "mm_replay_2026-11-14.md"
REPORT_JSON = ROOT / "research" / "reports" / "mm_replay_2026-11-14.json"
FIG = ROOT / "research" / "figures" / "mm_replay_2026-11-14"
QUEUES = ("pessimistic", "prorata", "optimistic")
PNL = ("net", "spread", "adverse_1s", "adverse_10s", "adverse_60s", "inventory", "fees")
B = 10_000
SEED = 20261114
EULER = 0.5772156649015329


# ----------------------------------------------------------------------------------------------------------------------
# load
# ----------------------------------------------------------------------------------------------------------------------


def family_of(name: str) -> str:
    return name.split("_", 1)[0]


def load_days() -> pl.DataFrame:
    rows = []
    for sym, g in GROUP_OF.items():
        for d in DAYS:
            p = OUT / "days" / sym / d / "stats.json"
            if not p.is_file():
                continue
            st = json.loads(p.read_text(encoding="utf-8"))
            for q in st["quoters"]:
                f = q["funnel"]
                rows.append({"symbol": sym, "group": g, "date": d, "split": "fit" if d in FIT_DAYS else "held", "variant": q["name"],
                             "family": family_of(q["name"]), "queue": q["queue"], **{k: float(q["pnl"][k]) for k in PNL},
                             "fills": int(q["fills"]), "volume_usd": float(q["volume_usd"]), "stopped": q["stopped"] is not None,
                             "fills_queue": f["fills_queue"], "fills_through": f["fills_through"], "fills_cross": f["fills_cross"],
                             "fills_stop": f["fills_stop"], "placed": f["placed"], "postonly_rejects": f["postonly_rejects"],
                             "dist_bps": f["dist_bps_mean"] if f["dist_bps_mean"] is not None else float("nan"),
                             "live_share": 0.5 * (f["live_share_bid"] + f["live_share_ask"]), "max_inv_usd": float(q["max_inventory_usd"]),
                             "end_inv_usd": float(q["end_inventory_usd"])})
    return pl.DataFrame(rows)


def market_rows() -> list[dict]:
    out = []
    for sym in GROUP_OF:
        for d in DAYS:
            p = OUT / "days" / sym / d / "stats.json"
            if p.is_file():
                st = json.loads(p.read_text(encoding="utf-8"))
                m = st["market"]
                out.append({"symbol": sym, "date": d, **{k: m[k] for k in ("book_messages", "trades", "rpi_trades", "trades_outside_touch", "engine_rejects",
                                                                        "replayer_check_mismatches", "mid_start", "mid_end")}, "seconds": st["perf"]["seconds"]})
    return out


# ----------------------------------------------------------------------------------------------------------------------
# statistics
# ----------------------------------------------------------------------------------------------------------------------


def day_matrix(df: pl.DataFrame, group: str, days: list[str], queue: str, names: list[str]) -> np.ndarray:
    """(days, variants) mean net P&L per symbol-day (mean over the group's symbols on that day); NaN when missing."""
    d = df.filter((pl.col("group") == group) & (pl.col("queue") == queue) & pl.col("date").is_in(days))
    w = d.group_by("date", "variant").agg(pl.col("net").mean())
    m = np.full((len(days), len(names)), np.nan)
    di, vi = {x: i for i, x in enumerate(days)}, {x: i for i, x in enumerate(names)}
    for r in w.iter_rows(named=True):
        if r["variant"] in vi:
            m[di[r["date"]], vi[r["variant"]]] = r["net"]
    return m


def cpcv(m: np.ndarray, days: list[str], k_test: int = 2, purge_days: int = 1) -> dict[str, Any]:
    """Combinatorial purged CV over days: every choice of k_test test days; training days within purge_days calendar days
    of a test day are dropped. Per split: the training-best variant and its test mean (the procedure's OOS estimate);
    per variant: its test mean in every split. Score = median over splits of the variant's test mean."""
    dts = np.array([np.datetime64(x) for x in days])
    ok_days = np.where(np.all(np.isfinite(m), axis=1))[0]
    splits, sel_oos, sel_idx, test_means = [], [], [], []
    for test in combinations(ok_days.tolist(), k_test):
        te = np.array(test)
        gap = np.min(np.abs((dts[ok_days][:, None] - dts[te][None, :]).astype("timedelta64[D]").astype(int)), axis=1)
        tr = ok_days[gap > purge_days]
        if len(tr) == 0:
            continue
        tm = m[te].mean(axis=0)
        best = int(np.argmax(m[tr].mean(axis=0)))
        splits.append({"test": [days[i] for i in te], "train": [days[i] for i in tr]})
        sel_idx.append(best)
        sel_oos.append(float(tm[best]))
        test_means.append(tm)
    tmm = np.vstack(test_means)
    score = np.median(tmm, axis=0)
    return {"n_splits": len(splits), "score": score, "choice": int(np.argmax(score)), "procedure_oos_mean": float(np.mean(sel_oos)),
            "procedure_oos_median": float(np.median(sel_oos)), "train_choice_counts": np.bincount(sel_idx, minlength=m.shape[1]).tolist(),
            "n_days": int(len(ok_days))}


def pbo_cscv_days(m: np.ndarray) -> dict[str, Any]:
    """CSCV with one-day blocks (S = number of complete days, even): C(S, S/2) IS / OOS halves; IS winner by mean net;
    omega = its OOS rank / (N + 1) (ties averaged); PBO = share of splits with logit(omega) <= 0."""
    x = m[np.all(np.isfinite(m), axis=1)]
    s = x.shape[0] - x.shape[0] % 2
    x = x[:s]
    n = x.shape[1]
    lam, deg = [], []
    for isx in combinations(range(s), s // 2):
        ins = np.zeros(s, bool)
        ins[list(isx)] = True
        mi, mo = x[ins].mean(axis=0), x[~ins].mean(axis=0)
        b = int(np.argmax(mi))
        rank = (mo < mo[b]).sum() + 0.5 * ((mo == mo[b]).sum() - 1) + 1
        om = rank / (n + 1)
        lam.append(math.log(om / (1 - om)))
        deg.append((mi[b], mo[b]))
    lam = np.array(lam)
    d = np.array(deg)
    return {"pbo": float(np.mean(lam <= 0)), "logit_median": float(np.median(lam)), "n_splits": len(lam), "blocks": s,
            "is_winner_mean_is": float(d[:, 0].mean()), "is_winner_mean_oos": float(d[:, 1].mean())}


def deflated_sharpe(x: np.ndarray, sr_trials: np.ndarray, n_trials: int) -> dict[str, Any]:
    """Bailey & Lopez de Prado (2014), per-day Sharpe; V = variance of the trials' Sharpes (finite ones)."""
    nd = NormalDist()
    x = x[np.isfinite(x)]
    t = len(x)
    if t < 3 or np.std(x, ddof=1) == 0:
        return {"sr_day": float("nan"), "dsr": float("nan"), "psr0": float("nan"), "T": t, "note": "zero variance (no fills) or too few days"}
    sr = float(x.mean() / x.std(ddof=1))
    z = (x - x.mean()) / x.std(ddof=0)
    skew, kurt = float(np.mean(z**3)), float(np.mean(z**4))
    tr = sr_trials[np.isfinite(sr_trials)]
    v_trials = float(np.var(tr, ddof=1)) if len(tr) > 1 else 0.0
    v_null = 1.0 / (t - 1)
    nn = max(float(n_trials), 2.0)
    zmax = (1 - EULER) * nd.inv_cdf(1 - 1 / nn) + EULER * nd.inv_cdf(1 - 1 / (nn * math.e))
    den = math.sqrt(max(1 - skew * sr + (kurt - 1) / 4 * sr**2, 1e-12))
    sr0_null, sr0_trials = math.sqrt(v_null) * zmax, math.sqrt(v_trials) * zmax
    # Primary: V = 1 / (T - 1), the sampling variance of a zero-Sharpe trial (the cross-trial variance is dominated by
    # variants that hit the loss stop every day, near-constant losses with |SR| ~ 50, which makes it meaningless).
    return {"sr_day": sr, "sr0_day": sr0_null, "sr0_day_trials": sr0_trials, "n_trials": n_trials, "T": t, "skew": skew, "kurt": kurt,
            "psr0": nd.cdf(sr * math.sqrt(t - 1) / den), "dsr": nd.cdf((sr - sr0_null) * math.sqrt(t - 1) / den),
            "dsr_trials_var": nd.cdf((sr - sr0_trials) * math.sqrt(t - 1) / den)}


def sharpe_cols(m: np.ndarray) -> np.ndarray:
    with np.errstate(invalid="ignore", divide="ignore"):
        sd = np.nanstd(m, axis=0, ddof=1)
        return np.where(sd > 0, np.nanmean(m, axis=0) / sd, np.nan)


def boot_mean(by_day: dict[str, list[float]], rng: np.random.Generator) -> dict[str, Any]:
    """Mean per symbol-day and a day-clustered bootstrap 95% CI (days resampled with all their symbol-days)."""
    days = sorted(by_day)
    vals = [np.asarray(by_day[d], float) for d in days]
    allv = np.concatenate(vals) if vals else np.array([])
    if len(days) < 2:
        return {"mean": float(allv.mean()) if len(allv) else float("nan"), "ci": [float("nan")] * 2, "n_days": len(days), "n": int(len(allv))}
    sums = np.array([v.sum() for v in vals])
    cnts = np.array([len(v) for v in vals])
    idx = rng.integers(0, len(days), size=(B, len(days)))
    bm = sums[idx].sum(axis=1) / cnts[idx].sum(axis=1)
    return {"mean": float(allv.mean()), "ci": [float(np.quantile(bm, 0.025)), float(np.quantile(bm, 0.975))], "n_days": len(days), "n": int(len(allv)),
            "sd": float(allv.std(ddof=1)) if len(allv) > 1 else float("nan")}


def by_day(df: pl.DataFrame, col: str = "net") -> dict[str, list[float]]:
    out: dict[str, list[float]] = {}
    for r in df.select("date", col).iter_rows():
        out.setdefault(r[0], []).append(float(r[1]))
    return out


# ----------------------------------------------------------------------------------------------------------------------
# H-MMR0: live fills
# ----------------------------------------------------------------------------------------------------------------------


def live_fills(day: str, sym: str) -> list[dict]:
    s = LIVE_SESSIONS[day]
    a, b = ms(s["start"]), ms(s["end"])
    p = ROOT / s["fills"]
    out = []
    if p.suffix == ".json":
        for x in json.loads(p.read_text(encoding="utf-8")):
            t = int(x["execTime"])
            if x["symbol"] == sym and a <= t <= b + 60_000 and str(x.get("orderLinkId", "")).startswith("mm") and x.get("execType", "Trade") == "Trade":
                out.append({"ts": t, "side": x["side"], "price": float(x["execPrice"]), "qty": float(x["execQty"]), "fee": float(x["execFee"]), "maker": bool(x["isMaker"])})
    else:
        # journal fills carry no qty: take it from the session's placed order at that price and side
        places = []
        for line in (ROOT / "data" / "journal_mainnet" / day / "order.jsonl").read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            if r.get("action") == "place" and r.get("session_id") == s["session"] and r.get("symbol") == sym:
                places.append(r)
        for line in p.read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            if r.get("session_id") != s["session"] or r.get("symbol") != sym:
                continue
            q = [o["qty"] for o in places if o["side"] == r["side"] and abs(o["price"] - r["price"]) < 1e-12]
            qty = float(q[-1]) if q else float("nan")
            out.append({"ts": int(r["exec_ms"]), "side": r["side"], "price": float(r["price"]), "qty": qty, "fee": 2e-4 * qty * float(r["price"]), "maker": True})
    return out


def h0(rng: np.random.Generator) -> dict[str, Any]:
    rows = []
    for day in LIVE_DAYS:
        for sym in GROUPS["illiquid"]:
            p = OUT / "live" / sym / day / "stats.json"
            if not p.is_file():
                continue
            st = json.loads(p.read_text(encoding="utf-8"))
            mid_end = st["market"]["mid_end"]
            lf = live_fills(day, sym)
            live_pnl = sum((1 if f["side"] == "Buy" else -1) * f["qty"] * (mid_end - f["price"]) - f["fee"] for f in lf if np.isfinite(f["qty"]))
            row = {"date": day, "symbol": sym, "window": [LIVE_SESSIONS[day]["start"], LIVE_SESSIONS[day]["end"]], "live_fills": len(lf), "live_pnl": live_pnl,
                   "live_fills_detail": lf}
            for q in st["quoters"]:
                row[f"replay_fills_{q['queue']}"] = q["fills"]
                row[f"replay_pnl_{q['queue']}"] = q["pnl"]["net"]
                row[f"replay_placed_{q['queue']}"] = q["funnel"]["placed"]
            rows.append(row)
    res: dict[str, Any] = {"rows": rows, "by_queue": {}}
    sgn = lambda x: 0 if abs(x) < 1e-9 else (1 if x > 0 else -1)  # noqa: E731
    for q in QUEUES:
        if not rows:
            break
        err = [abs(r[f"replay_fills_{q}"] - r["live_fills"]) / max(r["live_fills"], 1) for r in rows]
        agree = [sgn(r[f"replay_pnl_{q}"]) == sgn(r["live_pnl"]) for r in rows]
        res["by_queue"][q] = {"median_abs_fill_error": float(np.median(err)), "sign_agreement": float(np.mean(agree)), "n": len(rows),
                              "replay_fills_total": int(sum(r[f"replay_fills_{q}"] for r in rows)), "live_fills_total": int(sum(r["live_fills"] for r in rows))}
    pr = res["by_queue"].get("prorata")
    if not pr or pr["n"] < 4:
        res["verdict"] = "inconclusive"
    else:
        res["verdict"] = "supports" if pr["median_abs_fill_error"] <= 0.30 and pr["sign_agreement"] >= 0.70 else "contradicts"
    return res


# ----------------------------------------------------------------------------------------------------------------------
# selection and held-out tests
# ----------------------------------------------------------------------------------------------------------------------


def select(df: pl.DataFrame, group: str, names: list[str]) -> dict[str, Any]:
    m = day_matrix(df, group, FIT_DAYS, "prorata", names)
    cv = cpcv(m, FIT_DAYS)
    pbo = pbo_cscv_days(m)
    ch = cv["choice"]
    ds = deflated_sharpe(m[:, ch], sharpe_cols(m), len(names))
    return {"names": names, "choice": names[ch], "cv_score": float(cv["score"][ch]), "cv": {k: v for k, v in cv.items() if k != "score"},
            "scores": {n: float(s) for n, s in zip(names, cv["score"], strict=True)}, "fit_mean": {n: float(v) for n, v in zip(names, np.nanmean(m, axis=0), strict=True)},
            "pbo": pbo, "dsr": ds}


def held(df: pl.DataFrame, group: str, variant: str, queue: str, rng: np.random.Generator) -> dict[str, Any]:
    d = df.filter((pl.col("group") == group) & (pl.col("variant") == variant) & (pl.col("queue") == queue) & (pl.col("split") == "held"))
    r = boot_mean(by_day(d), rng)
    r["decomposition_mean"] = {k: float(d[k].mean()) if d.height else float("nan") for k in PNL}
    r["fills_mean"] = float(d["fills"].mean()) if d.height else float("nan")
    r["volume_usd_mean"] = float(d["volume_usd"].mean()) if d.height else float("nan")
    r["stopped_share"] = float(d["stopped"].mean()) if d.height else float("nan")
    return r


def paired(df: pl.DataFrame, group: str, a: str, b: str, queue: str, rng: np.random.Generator) -> dict[str, Any]:
    """Paired per symbol-day difference a - b on held days, day-clustered bootstrap."""
    f = df.filter((pl.col("group") == group) & (pl.col("queue") == queue) & (pl.col("split") == "held"))
    x = f.filter(pl.col("variant") == a).select("symbol", "date", pl.col("net").alias("a"))
    y = f.filter(pl.col("variant") == b).select("symbol", "date", pl.col("net").alias("b"))
    j = x.join(y, on=["symbol", "date"]).with_columns((pl.col("a") - pl.col("b")).alias("d"))
    r = boot_mean(by_day(j, "d"), rng)
    r.update({"a": a, "b": b, "mean_a": float(j["a"].mean()) if j.height else float("nan"), "mean_b": float(j["b"].mean()) if j.height else float("nan")})
    return r


def verdict_h1(groups: dict[str, dict]) -> str:
    pr = [g["held"]["prorata"] for g in groups.values()]
    if any(x["ci"][0] > 0 for x in pr):
        return "supports"
    if all(x["mean"] <= 0 for x in pr):
        return "contradicts"
    return "inconclusive"


def verdict_h2(tests: list[dict]) -> str:
    if any(t["ci"][0] > 0 for t in tests):
        return "supports"
    if all(t["mean"] <= 0 for t in tests):
        return "contradicts"
    return "inconclusive"


# ----------------------------------------------------------------------------------------------------------------------
# figures
# ----------------------------------------------------------------------------------------------------------------------

COL = {"L": "#2a78d6", "M": "#eb6834", "F": "#1baf7a", "N": "#eda100"}
INK, INK2, GRIDC, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


def _style(ax):
    ax.set_facecolor(SURF)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK2)
    ax.tick_params(colors=INK2, labelsize=8)
    ax.grid(axis="y", color=GRIDC, linewidth=0.6)
    ax.set_axisbelow(True)


def figures(res: dict[str, Any], df: pl.DataFrame) -> list[str]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return []
    FIG.mkdir(parents=True, exist_ok=True)
    paths = []
    # 1. held-out net per symbol-day of each family's fit-chosen variant, with the bootstrap CI (pro-rata)
    fig, axes = plt.subplots(1, len(GROUPS), figsize=(10, 3.6), facecolor=SURF)
    for ax, g in zip(np.atleast_1d(axes), GROUPS, strict=True):
        _style(ax)
        fams = res["groups"][g]["family_best"]
        names = list(fams)
        for i, fam in enumerate(names):
            h = fams[fam]["held"]["prorata"]
            m, (lo, hi) = h["mean"], h["ci"]
            ax.bar(i, m, width=0.55, color=COL.get(fam, INK2))
            ax.errorbar(i, m, yerr=[[m - lo], [hi - m]], color=INK, capsize=3, linewidth=1)
        ax.axhline(0, color=INK2, linewidth=0.8)
        ax.set_xticks(range(len(names)), [f"{f}\n{fams[f]['variant']}" for f in names], fontsize=7)
        ax.set_title(f"{g}: held-out net P&L per symbol-day (USD)", fontsize=9, color=INK, loc="left")
    fig.tight_layout()
    p = FIG / "held_net_by_family.png"
    fig.savefig(p, dpi=140)
    plt.close(fig)
    paths.append(str(p.relative_to(ROOT)).replace("\\", "/"))
    # 2. decomposition of the chosen variant (held, mean per symbol-day) under the three queue models
    fig, axes = plt.subplots(1, len(GROUPS), figsize=(10, 4.0), facecolor=SURF)
    parts = ["spread", "adverse_60s", "inventory", "fees"]
    pc = {"spread": "#2a78d6", "adverse_60s": "#eb6834", "inventory": "#1baf7a", "fees": "#4a3aa7"}
    for ax, g in zip(np.atleast_1d(axes), GROUPS, strict=True):
        _style(ax)
        for i, q in enumerate(QUEUES):
            dm = res["groups"][g]["held"][q]["decomposition_mean"]
            for j, k in enumerate(parts):
                v = -dm[k] if k == "fees" else dm[k]
                ax.bar(i + (j - 1.5) * 0.18, v, width=0.16, color=pc[k], label=("-fees" if k == "fees" else k) if i == 0 else None)
            ax.plot([i - 0.4, i + 0.4], [dm["net"]] * 2, color=INK, linewidth=1.5, label="net" if i == 0 else None)
        ax.axhline(0, color=INK2, linewidth=0.8)
        ax.set_xticks(range(3), QUEUES, fontsize=8)
        ax.set_title(f"{g}: {res['groups'][g]['choice']} held-out decomposition (USD / symbol-day)", fontsize=9, color=INK, loc="left")
        ax.legend(fontsize=7, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=5)
    fig.tight_layout()
    p = FIG / "decomposition.png"
    fig.savefig(p, dpi=140)
    plt.close(fig)
    paths.append(str(p.relative_to(ROOT)).replace("\\", "/"))
    # 3. fit vs held mean net per variant (pro-rata): does the fit ranking carry over?
    fig, axes = plt.subplots(1, len(GROUPS), figsize=(10, 3.8), facecolor=SURF)
    for ax, g in zip(np.atleast_1d(axes), GROUPS, strict=True):
        _style(ax)
        ax.grid(axis="x", color=GRIDC, linewidth=0.6)
        d = df.filter((pl.col("group") == g) & (pl.col("queue") == "prorata")).group_by("variant", "family", "split").agg(pl.col("net").mean())
        w = d.pivot(on="split", index=["variant", "family"], values="net")
        for fam in COL:
            s = w.filter(pl.col("family") == fam)
            if s.height and "fit" in s.columns and "held" in s.columns:
                ax.scatter(s["fit"], s["held"], s=22, color=COL[fam], label=fam, edgecolors=SURF, linewidths=1.5)
        ax.axhline(0, color=INK2, linewidth=0.8)
        ax.axvline(0, color=INK2, linewidth=0.8)
        ax.set_xlabel("fit days: mean net / symbol-day", fontsize=8, color=INK2)
        ax.set_ylabel("held days", fontsize=8, color=INK2)
        ax.set_title(f"{g}: grid variants, pro-rata", fontsize=9, color=INK, loc="left")
        ax.legend(fontsize=7, frameon=False, title="family", title_fontsize=7)
    fig.tight_layout()
    p = FIG / "fit_vs_held.png"
    fig.savefig(p, dpi=140)
    plt.close(fig)
    paths.append(str(p.relative_to(ROOT)).replace("\\", "/"))
    return paths


# ----------------------------------------------------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------------------------------------------------


def analyse() -> tuple[dict[str, Any], pl.DataFrame]:
    rng = np.random.default_rng(SEED)
    df = load_days()
    res: dict[str, Any] = {"split": json.loads(SPLIT.read_text(encoding="utf-8"))["design"], "groups": {}, "coverage": {}}
    for g, syms in GROUPS.items():
        names = [v["name"] for v in grid(g)]
        cov = df.filter((pl.col("group") == g) & (pl.col("queue") == "prorata") & (pl.col("variant") == names[0]))
        res["coverage"][g] = {s: sorted(cov.filter(pl.col("symbol") == s)["date"].to_list()) for s in syms}
        sel = select(df, g, names)
        gr: dict[str, Any] = {"selection": sel, "choice": sel["choice"]}
        gr["held"] = {q: held(df, g, sel["choice"], q, rng) for q in QUEUES}
        fam_best = {}
        for fam in sorted({family_of(n) for n in names}, key=lambda f: "LMFN".index(f)):
            fn = [n for n in names if family_of(n) == fam]
            s = select(df, g, fn)
            fam_best[fam] = {"variant": s["choice"], "cv_score": s["cv_score"], "pbo": s["pbo"]["pbo"], "held": {q: held(df, g, s["choice"], q, rng) for q in QUEUES}}
        gr["family_best"] = fam_best
        # H-MMR2: forecast-informed widths vs the best fixed width (both chosen on fit days)
        tests = {}
        for fam in ("M", "N"):
            if fam in fam_best and "F" in fam_best:
                tests[fam] = {q: paired(df, g, fam_best[fam]["variant"], fam_best["F"]["variant"], q, rng) for q in QUEUES}
        gr["h2"] = tests
        # all-variant held table (pro-rata) for the report
        t = (df.filter((pl.col("group") == g) & (pl.col("queue") == "prorata")).group_by("variant", "split")
             .agg(pl.col("net").mean().alias("net"), pl.col("fills").mean().alias("fills"), pl.col("stopped").mean().alias("stopped")))
        gr["table"] = t.sort("variant", "split").to_dicts()
        res["groups"][g] = gr
    res["h0"] = h0(rng)
    res["h1_verdict"] = verdict_h1(res["groups"])
    h2_tests = [v["prorata"] for g in res["groups"].values() for v in g["h2"].values()]
    res["h2_verdict"] = verdict_h2(h2_tests) if h2_tests else "inconclusive"
    res["market"] = market_rows()
    man = json.loads(MANIFEST.read_text(encoding="utf-8")) if MANIFEST.is_file() else {}
    res["storage"] = man.get("storage", {})
    res["failed_days"] = {k: v for k, v in man.get("days", {}).items() if v.get("status") != "done"}
    res["downloads"] = {"n": len(man.get("downloads", {})), "bytes": sum(int(v.get("bytes", 0)) for v in man.get("downloads", {}).values()),
                        "all_deleted": all(v.get("deleted") for v in man.get("downloads", {}).values() if v.get("status") == "ok")}
    return res, df


def fmt(x: float, n: int = 4) -> str:
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return "n/a"
    return f"{x:+.{n}f}"


def ci(h: dict) -> str:
    return f"{fmt(h['mean'])} [{fmt(h['ci'][0])}, {fmt(h['ci'][1])}]"


def record_all(res: dict[str, Any]) -> list[dict]:
    from ai_trader.research.hypotheses import record, results

    done = {(r["id"], r["segment"]) for r in results()}
    data = "data/research/mm_replay (split.json, days/, live/); report research/reports/mm_replay_2026-11-14.md"
    note_seg = "All replay days (2026-10-25..11-13) are after research.knowledge_cutoff 2026-07-01: clean_holdout only."
    out = []
    h0r = res["h0"]
    if ("H-MMR0", "clean_holdout") not in done:
        out.append(record("H-MMR0", "clean_holdout", h0r["verdict"], {"by_queue": h0r["by_queue"], "rows": [{k: v for k, v in r.items() if k != "live_fills_detail"} for r in h0r["rows"]]},
                          note=f"{note_seg} Live symbol-days with 0-2 fills: little power.", data=data))
    if ("H-MMR1", "clean_holdout") not in done:
        m = {g: {"choice": x["choice"], "held": {q: {k: x["held"][q][k] for k in ("mean", "ci", "n_days", "n", "decomposition_mean")} for q in QUEUES},
                 "pbo": x["selection"]["pbo"]["pbo"], "dsr": x["selection"]["dsr"].get("dsr"), "cv_score": x["selection"]["cv_score"]} for g, x in res["groups"].items()}
        out.append(record("H-MMR1", "clean_holdout", res["h1_verdict"], m, note=note_seg, data=data))
    if ("H-MMR2", "clean_holdout") not in done:
        m = {g: {fam: {q: {k: t[q][k] for k in ("a", "b", "mean", "ci", "n_days", "n", "mean_a", "mean_b")} for q in QUEUES} for fam, t in x["h2"].items()} for g, x in res["groups"].items()}
        out.append(record("H-MMR2", "clean_holdout", res["h2_verdict"], m,
                          note=note_seg + " N = field hour-of-week x log-vol sigma except 2026-11-13 20-24 UTC (live network band).", data=data))
    return out


def _decomp_rows(res: dict[str, Any]) -> list[str]:
    L = ["| group | variant | queue | net | spread | adverse 1 s | adverse 10 s | adverse 60 s | inventory | fees | fills / symbol-day | 95% CI of net |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for g, x in res["groups"].items():
        for q in QUEUES:
            h = x["held"][q]
            d = h["decomposition_mean"]
            L.append(f"| {g} | {x['choice']} | {q} | {fmt(d['net'])} | {fmt(d['spread'])} | {fmt(d['adverse_1s'])} | {fmt(d['adverse_10s'])} | "
                     f"{fmt(d['adverse_60s'])} | {fmt(d['inventory'])} | {fmt(-d['fees'])} | {h['fills_mean']:.1f} | [{fmt(h['ci'][0])}, {fmt(h['ci'][1])}] |")
    return L


def render_md(res: dict[str, Any], rules_yaml: str, extra: str = "") -> str:
    L: list[str] = ["# Market making on replayed real Bybit order books (H-MMR0, H-MMR1, H-MMR2), 2026-11-14", "",
                    "Agent M. Research only: no live orders, no Bybit trading calls. The quoting code is the live `atx mm` code; the market is the "
                    "real ob200 book and the real trade tape of each symbol-day, replayed by `atx mm-replay` (crates/atx/src/mm_replay.rs) through "
                    "at-match. All days are after the knowledge cutoff (2026-07-01), so every result is `clean_holdout`.", ""]
    L += ["## Verdicts", "",
          f"- **H-MMR0 (replay reproduces live fills): {res['h0']['verdict']}.**",
          f"- **H-MMR1 (held-out net P&L > 0): {res['h1_verdict']}.**",
          f"- **H-MMR2 (forecast widths beat the best fixed width): {res['h2_verdict']}.**", ""]
    L += [extra, ""] if extra else []
    L += ["## What was run", ""]
    for k, v in res["split"].items():
        L.append(f"- **{k}**: {v}")
    L += ["", "Coverage (symbol-days with a verified replay):", ""]
    for g, c in res["coverage"].items():
        for s, ds in c.items():
            L.append(f"- {g} {s}: {len(ds)} days ({ds[0] if ds else '-'} .. {ds[-1] if ds else '-'})")
    if res["failed_days"]:
        L.append(f"- not completed: {sorted(res['failed_days'])}")
    L += ["", "## H-MMR0: replay vs the live bot's own fills", "",
          "| day | symbol | live window (UTC) | live fills | replay fills (pess / pro-rata / opt) | live P&L | replay P&L pro-rata |", "|---|---|---|---|---|---|---|"]
    for r in res["h0"]["rows"]:
        L.append(f"| {r['date']} | {r['symbol']} | {r['window'][0][11:16]}-{r['window'][1][11:16]} | {r['live_fills']} | "
                 f"{r.get('replay_fills_pessimistic')} / {r.get('replay_fills_prorata')} / {r.get('replay_fills_optimistic')} | {fmt(r['live_pnl'])} | {fmt(r.get('replay_pnl_prorata', float('nan')))} |")
    L += ["", "| queue model | median abs fill-count error | P&L sign agreement | replay fills | live fills |", "|---|---|---|---|---|"]
    for q, v in res["h0"]["by_queue"].items():
        L.append(f"| {q} | {v['median_abs_fill_error']:.2f} | {v['sign_agreement']:.2f} | {v['replay_fills_total']} | {v['live_fills_total']} |")
    L += ["", "Error = |replay - live| / max(live, 1) per symbol-day. Success needed <= 0.30 and >= 0.70 under pro-rata.", ""]
    L += ["## Selection on fit days (pro-rata) and overfitting", "",
          "| group | chosen | CV score (median test-day net) | procedure OOS mean | PBO | deflated Sharpe | fit-days daily Sharpe |", "|---|---|---|---|---|---|---|"]
    for g, x in res["groups"].items():
        s = x["selection"]
        L.append(f"| {g} | {x['choice']} | {fmt(s['cv_score'])} | {fmt(s['cv']['procedure_oos_mean'])} | {s['pbo']['pbo']:.2f} | "
                 f"{fmt(s['dsr'].get('dsr', float('nan')), 3)} | {fmt(s['dsr'].get('sr_day', float('nan')), 3)} |")
    L += ["", "Best variant per family (chosen on fit days the same way):", "", "| group | family | variant | held net pro-rata [95% CI] | pessimistic | optimistic | family PBO |", "|---|---|---|---|---|---|---|"]
    for g, x in res["groups"].items():
        for fam, fb in x["family_best"].items():
            h = fb["held"]
            L.append(f"| {g} | {fam} | {fb['variant']} | {ci(h['prorata'])} | {fmt(h['pessimistic']['mean'])} | {fmt(h['optimistic']['mean'])} | {fb['pbo']:.2f} |")
    L += ["", "## H-MMR1: held-out P&L and its decomposition (USD per symbol-day)", ""] + _decomp_rows(res)
    L += ["", "net = spread + adverse 60 s + inventory - fees; adverse 1 s and 10 s are the earlier parts of adverse 60 s.", ""]
    L += ["## H-MMR2: forecast-informed widths minus the best fixed width (held days, paired per symbol-day)", "",
          "| group | forecast | variant | fixed | difference pro-rata [95% CI] | pessimistic | optimistic |", "|---|---|---|---|---|---|---|"]
    for g, x in res["groups"].items():
        for fam, t in x["h2"].items():
            L.append(f"| {g} | {'micro sigma' if fam == 'M' else '1h network / field band'} | {t['prorata']['a']} | {t['prorata']['b']} | {ci(t['prorata'])} | "
                     f"{fmt(t['pessimistic']['mean'])} | {fmt(t['optimistic']['mean'])} |")
    if res.get("figures"):
        L += ["", "## Figures", ""] + [f"![{Path(f).stem}](../../{f})" for f in res["figures"]]
    L += ["", "## Rule candidates", "", "```yaml", rules_yaml.rstrip(), "```", ""]
    return "\n".join(L)


def main(argv: list[str] | None = None) -> None:
    argv = argv if argv is not None else sys.argv[1:]
    res, df = analyse()
    res["figures"] = figures(res, df)
    if "--no-record" not in argv:
        res["recorded"] = record_all(res)
    REPORT_JSON.write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")
    print(json.dumps({"h0": res["h0"]["by_queue"], "h0_verdict": res["h0"]["verdict"], "h1": res["h1_verdict"], "h2": res["h2_verdict"],
                      "choices": {g: x["choice"] for g, x in res["groups"].items()}}, indent=1, default=float))


if __name__ == "__main__":
    main()
