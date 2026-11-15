"""Figures, tables and the verdict records of the BOS study (H-BOS1..5).

    uv run python -m ai_trader.research.bos_regimes_report figures
    uv run python -m ai_trader.research.bos_regimes_report tables      # markdown tables -> data/research/bos/tables.md

Figures: research/figures/bos_regimes_2026-11-15/. Palette: diverging blue <-> red with a gray midpoint (heatmap),
categorical blue / orange / aqua (dataviz reference palette).
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import polars as pl  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

from ai_trader.core.config import load  # noqa: E402
from ai_trader.research import bos_regimes_analysis as A  # noqa: E402
from ai_trader.research import bos_regimes_study as S  # noqa: E402

BLUE, ORANGE, AQUA, YELLOW, VIOLET = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#4a3aa7"
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
DIV = LinearSegmentedColormap.from_list("div", ["#e34948", "#f0efec", "#2a78d6"])
TAG = "2026-11-15"


def fig_dir():
    d = load().root / "research" / "figures" / f"bos_regimes_{TAG}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _style(ax):
    ax.set_facecolor(SURF)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK2)
    ax.tick_params(colors=INK2, labelsize=8)
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.set_axisbelow(True)


def results() -> dict:
    return json.loads((S.root() / "results.json").read_text())


def cells() -> pl.DataFrame:
    return pl.read_parquet(A.agg_dir() / "cells.parquet")


# ---------------------------------------------------------------- 1 heatmap
def heatmap(df: pl.DataFrame, arm: int = 1) -> str:
    d = df.filter((pl.col("arm") == arm) & (pl.col("n") >= A.MIN_N))
    d = d.with_columns((pl.col("vid") // 4).alias("grp"))
    g = d.group_by("grp", "rid", "seg").agg(pl.col("mean").mean())
    grps = sorted(g["grp"].unique().to_list())
    names = []
    for gg in grps:
        v = gg * 4
        nm = A.variant_name(v).rsplit("|", 1)[0]
        names.append(nm)
    fig, axes = plt.subplots(1, 2, figsize=(15, max(6, 0.17 * len(grps) + 2)), sharey=True)
    for sg, ax in enumerate(axes):
        M = np.full((len(grps), A.NREG), np.nan)
        for r in g.filter(pl.col("seg") == sg).iter_rows(named=True):
            M[grps.index(r["grp"]), r["rid"]] = r["mean"]
        im = ax.imshow(M, cmap=DIV, vmin=-0.5, vmax=0.5, aspect="auto", interpolation="nearest")
        ax.set_xticks(range(A.NREG))
        ax.set_xticklabels(A.REG_NAMES, rotation=90, fontsize=7, color=INK2)
        ax.set_yticks(range(len(grps)))
        ax.set_yticklabels(names, fontsize=6, color=INK2)
        ax.set_title(f"{A.SEGS[sg]}: mean net R per trade ({S.ARM_NAME[S.ARMS[arm]]} entry, mean over the 4 exits)", fontsize=9, color=INK)
        for x in (0.5, 18.5, 21.5):
            ax.axvline(x, color="white", lw=2)
    cb = fig.colorbar(im, ax=axes, shrink=0.5, pad=0.01)
    cb.set_label("net R per trade (clipped at +-0.5)", fontsize=8, color=INK2)
    fig.suptitle("BoS variants x regime: blue = positive, red = negative, gray = about zero, blank = fewer than 30 trades", fontsize=10, color=INK)
    p = fig_dir() / "heatmap_variant_regime.png"
    fig.savefig(p, dpi=130, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return p.name


# ---------------------------------------------------------------- 2 tail decomposition
def tails(res: dict, arm: str = "p") -> str:
    fc = res["family_cells"]
    clss = [c for c in ("intraday", "swing") if c in fc]
    regs = A.REG_NAMES[1:]
    fig, axes = plt.subplots(len(clss), 2, figsize=(15, 4.2 * len(clss)), squeeze=False)
    for i, cls in enumerate(clss):
        for sg in (0, 1):
            ax = axes[i, sg]
            _style(ax)
            top, rest = [], []
            for r in regs:
                c = fc[cls].get(f"{arm}|{r}|{sg}")
                if c is None:
                    top.append(np.nan)
                    rest.append(np.nan)
                    continue
                top.append(c["top5_sum_per_trade"])
                rest.append(c["mean"] - c["top5_sum_per_trade"])
            x = np.arange(len(regs))
            ax.bar(x - 0.2, top, 0.38, color=BLUE, label="top 5% of trades (per-trade contribution)")
            ax.bar(x + 0.2, rest, 0.38, color=ORANGE, label="other 95% of trades")
            mean = [fc[cls].get(f"{arm}|{r}|{sg}", {}).get("mean", np.nan) for r in regs]
            ax.plot(x, mean, "o", color=INK, ms=4, label="cell mean net R")
            ax.axhline(0, color=INK2, lw=0.8)
            ax.set_xticks(x)
            ax.set_xticklabels(regs, rotation=90, fontsize=7)
            ax.set_title(f"{cls}, {A.SEGS[sg]} ({S.ARM_NAME[arm]} entry, all variants pooled)", fontsize=9, color=INK)
            ax.set_ylabel("net R per trade", fontsize=8, color=INK2)
            if i == 0 and sg == 0:
                ax.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    p = fig_dir() / "tail_decomposition.png"
    fig.savefig(p, dpi=130, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return p.name


# ---------------------------------------------------------------- 3 stability
def stability(res: dict) -> str:
    cs = res["h3"]["cells"]
    seen, pick = set(), []
    for c in cs:
        k = (c["variant"], c["regime"])
        if k not in seen:
            seen.add(k)
            pick.append(c)
        if len(pick) == 4:
            break
    fig, axes = plt.subplots(len(pick), 1, figsize=(12, 2.6 * len(pick)), squeeze=False)
    ev = [(datetime.fromisoformat(e), n) for e, n in zip(res_events(), res_event_names(), strict=True)]
    for ax, c in zip(axes[:, 0], pick, strict=True):
        _style(ax)
        hs = c["halves"]
        xs = [datetime(int(h["half"][:4]), 1 if h["half"].endswith("1") else 7, 1) + timedelta(days=91) for h in hs]
        ax.bar(xs, [h["mean"] for h in hs], width=150, color=[BLUE if h["mean"] > 0 else ORANGE for h in hs])
        if c["mean_ho"] is not None:
            ax.bar([datetime(2026, 9, 1)], [c["mean_ho"]], width=100, color=AQUA, label="clean_holdout")
        for b in c["bai_perron_full"]["break_dates"]:
            ax.axvline(datetime.fromisoformat(b), color=INK, lw=1.4, ls="--")
        for d, _n in ev:
            ax.axvline(d, color=GRID, lw=0.8)
        ax.axvline(datetime(2026, 7, 1), color=AQUA, lw=1)
        ax.axhline(0, color=INK2, lw=0.8)
        ax.set_title(f"{c['variant']} | {c['arm']} | {c['regime']}: half-year mean net R (blue +, orange -), dashed = Bai-Perron break, "
                     f"gray = frozen events; CUSUM p {c['cusum_p']:.3f}", fontsize=8, color=INK)
        ax.set_ylabel("net R", fontsize=8, color=INK2)
    fig.tight_layout()
    p = fig_dir() / "rolling_stability.png"
    fig.savefig(p, dpi=130, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return p.name


def res_events():
    from ai_trader.research.bos_regimes_infer import EVENTS

    return EVENTS


def res_event_names():
    from ai_trader.research.bos_regimes_infer import EVENT_NAMES

    return EVENT_NAMES


# ---------------------------------------------------------------- 4 allocator
def allocator(res: dict) -> str:
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.2))
    for ax, arm in zip(axes, ("pegged", "market"), strict=True):
        _style(ax)
        a = res["h4"][arm]
        t0 = datetime.fromisoformat(a["equity_start"])
        for k, col, lab in (("alloc", BLUE, "Bayesian on/off allocator"), ("static", ORANGE, "static in-sample-best variant"), ("ew", AQUA, "equal weight, all variants")):
            y = np.array(a["equity"][k])
            ax.plot([t0 + timedelta(days=i) for i in range(len(y))], y, color=col, lw=2, label=lab)
        ax.axvline(t0 + timedelta(days=a["holdout_start_index"]), color=INK2, lw=1, ls="--")
        ax.set_title(f"{arm} entries: cumulative P&L, each book scaled by its training-window daily sd (WF from 2023)", fontsize=8, color=INK)
        ax.set_ylabel("cumulative scaled P&L (training sd units)", fontsize=8, color=INK2)
        ax.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    p = fig_dir() / "allocator_equity.png"
    fig.savefig(p, dpi=130, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return p.name


# ---------------------------------------------------------------- 5 execution arms
def execution(res: dict) -> str:
    e = res["h5"]
    clss = [c for c in ("intraday", "swing") if c in e]
    fig, axes = plt.subplots(1, len(clss), figsize=(6.5 * len(clss), 4), squeeze=False)
    arms = [("m", "market"), ("p", "pegged"), ("r", "retest N6"), ("rb", "retest N6 best case"), ("r24", "retest N24")]
    for ax, cls in zip(axes[0], clss, strict=True):
        _style(ax)
        for sg, col in ((0, BLUE), (1, ORANGE)):
            r = e[cls].get(A.SEGS[sg], {})
            xs = np.arange(len(arms)) + (sg - 0.5) * 0.3
            m = [r.get(k, {}).get("mean", np.nan) for k, _ in arms]
            lo = [r.get(k, {}).get("lo", np.nan) for k, _ in arms]
            hi = [r.get(k, {}).get("hi", np.nan) for k, _ in arms]
            ax.errorbar(xs, m, yerr=[np.subtract(m, lo), np.subtract(hi, m)], fmt="o", color=col, ms=6, capsize=3, lw=1.5, label=A.SEGS[sg])
        ax.axhline(0, color=INK2, lw=0.8)
        ax.set_xticks(range(len(arms)))
        ax.set_xticklabels([n for _, n in arms], fontsize=8)
        mw = e[cls].get("llm_in_sample", {}).get("missed", {})
        ax.set_title(f"{cls}: mean net R per signal (missed retest = 0), 95% week-bootstrap CI\n"
                     f"retest misses {mw.get('missed_share', np.nan):.0%} of signals, {mw.get('top5_missed_share', np.nan):.0%} of the market arm's top-5% trades (in-sample)",
                     fontsize=8, color=INK)
        ax.set_ylabel("net R per signal", fontsize=8, color=INK2)
        ax.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    p = fig_dir() / "execution_arms.png"
    fig.savefig(p, dpi=130, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return p.name


def figures() -> list[str]:
    res = results()
    df = cells()
    return [heatmap(df, 1), tails(res, "p"), stability(res), allocator(res), execution(res)]


if __name__ == "__main__":
    if sys.argv[1:] == ["figures"]:
        print(figures())


# ---------------------------------------------------------------- markdown tables
def _f(x, d=3, sign=True):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "n/a"
    return f"{x:+.{d}f}" if sign else f"{x:.{d}f}"


def _lab(c: dict | None) -> str:
    if not c:
        return "-"
    z, m, t = c.get("z"), c["mean"], c.get("top5_share")
    if m < 0 and z is not None and z < -2:
        return "detrimental"
    if m > 0 and t is not None and t > 1:
        return "explosive" + (" + power" if z is not None and z > 2 else "")
    if m > 0 and z is not None and z > 2:
        return "power"
    return "neutral"


def tables() -> str:
    res = results()
    df = cells()
    L = []
    # per-regime family pools
    for cls in ("intraday", "swing"):
        fc = res["family_cells"].get(cls)
        if not fc:
            continue
        for arm in ("m", "p", "r"):
            L += [f"\n#### {cls}, {S.ARM_NAME[arm]} entry: every variant of the class pooled, per regime\n",
                  "| regime | in-sample n | mean R | z | top-5% share | label | holdout n | mean R | z | top-5% share | label |",
                  "|---|---|---|---|---|---|---|---|---|---|---|"]
            for r in A.REG_NAMES:
                a, b = fc.get(f"{arm}|{r}|0"), fc.get(f"{arm}|{r}|1")
                row = [r]
                for c in (a, b):
                    if c:
                        row += [f"{c['n']:.0f}", _f(c["mean"]), _f(c["z"], 2), _f(c["top5_share"], 2, False), _lab(c)]
                    else:
                        row += ["-"] * 5
                L.append("| " + " | ".join(row) + " |")
    # top cells
    ins = df.filter((pl.col("seg") == 0) & pl.col("tested") & pl.col("arm").is_in([0, 1, 2]))
    ho = df.filter(pl.col("seg") == 1).select("vid", "arm", "rid", pl.col("n").alias("n_ho"), pl.col("mean").alias("mean_ho"), pl.col("z").alias("z_ho"))
    j = ins.join(ho, on=["vid", "arm", "rid"], how="left")
    for title, d in (("Top 20 tested in-sample cells by z (none survives BH unless marked)", j.sort("z", descending=True).head(20)),
                     ("Bottom 20 tested in-sample cells by z", j.sort("z").head(20))):
        L += [f"\n#### {title}\n", "| variant | arm | regime | n | weeks | mean R | z | p | hit | payoff | top-5% share | holdout n | holdout mean | holdout z |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for r in d.iter_rows(named=True):
            p = r["p_pos"] if r["z"] > 0 else r["p_neg"]
            nho = "-" if r["n_ho"] is None else f"{r['n_ho']:.0f}"
            L.append(f"| {r['variant']} | {r['arm_name']} | {r['regime']} | {r['n']:.0f} | {r['weeks']} | {_f(r['mean'])} | {_f(r['z'], 2)} | {p:.1e} | "
                     f"{_f(r['hit'], 2, False)} | {_f(r['payoff'], 2, False)} | {_f(r['top5_share'], 2, False)} | "
                     f"{nho} | {_f(r['mean_ho'])} | {_f(r['z_ho'], 2)} |")
    (S.root() / "tables.md").write_text("\n".join(L), encoding="utf-8")
    return "\n".join(L)


if __name__ == "__main__" and sys.argv[1:] == ["tables"]:
    print(len(tables()))


# ---------------------------------------------------------------- verdicts and records
DATA = ("bybit universe L (23 crypto perps): lake 1h/4h 2020-2026, public-REST 5m (15m aggregated) 2022-2026; "
        "src/ai_trader/research/bos_regimes_{study,analysis,infer,report}.py; data/research/bos/results.json")


def verdicts(res: dict) -> dict:
    v = {}
    h1 = res["h1"]
    ins1 = "supports" if h1["llm_in_sample"]["bh_pos"] >= 1 else "contradicts"
    if not h1["survivors"]:
        ho1 = "inconclusive"
    else:
        ho1 = "supports" if h1["survivors_holdout_pos"] >= 1 else "contradicts"
    v["H-BOS1"] = (ins1, ho1, {"tested_in_sample": h1["llm_in_sample"]["tested"], "bh_pos_in_sample": h1["llm_in_sample"]["bh_pos"],
                               "bh_neg_in_sample": h1["llm_in_sample"]["bh_neg"], "tested_holdout": h1["clean_holdout"]["tested"],
                               "bh_pos_holdout": h1["clean_holdout"]["bh_pos"], "bh_neg_holdout": h1["clean_holdout"]["bh_neg"],
                               "survivors_holdout_pos": h1["survivors_holdout_pos"], "best_cell": {k: h1["step1_best_cell"][k] for k in ("variant", "arm", "regime", "mean", "z", "n")},
                               "best_cell_dsr": h1["step1_best_cell"]["dsr"]["dsr"], "cells_pbo": h1["step1_best_cell"]["pbo"]["pbo"],
                               "negative_survivors_holdout_neg_share": h1["negative_survivors_holdout_neg_share"]})
    h2 = res["h2"]
    loose_in = any(h2[c]["m"]["loose_in_sample"] for c in h2)
    strict = any(h2[c]["m"]["strict"] for c in h2)
    any_ho = any(h2[c]["m"]["explosive"]["clean_holdout"] or h2[c]["m"]["detrimental_t2"]["clean_holdout"] for c in h2)
    v["H-BOS2"] = ("supports" if loose_in else "contradicts", "supports" if strict else ("inconclusive" if any_ho else "contradicts"),
                   {c: {k: h2[c]["m"][k] for k in ("strict_explosive_both", "strict_detrimental_both", "loose_in_sample", "loose_holdout")}
                    | {"detrimental_in_sample": h2[c]["m"]["detrimental_t2"]["llm_in_sample"], "detrimental_holdout": h2[c]["m"]["detrimental_t2"]["clean_holdout"],
                       "best_minus_worst": h2[c]["m"]["best_minus_worst"]} for c in h2})
    h3 = res["h3"]
    v["H-BOS3"] = ("supports" if h3["majority_both"] else "contradicts", "inconclusive",
                   {k: h3[k] for k in ("basis", "n_cells", "n_both", "n_flip", "n_break", "holdout_sign_flips", "holdout_n10", "breaks_total",
                                       "breaks_near_event", "breaks_near_shift", "chance_event", "chance_shift")})
    a = res["h4"]["pegged"]
    i4, o4 = a["llm_in_sample"], a["clean_holdout"]
    ins4 = "supports" if a["dsr_in_sample"]["dsr"] > 0.95 and i4["vs_static"]["mean_diff"] > 0 and i4["vs_ew"]["mean_diff"] > 0 else "contradicts"
    if o4["vs_static"]["t"] > 2 and o4["vs_ew"]["t"] > 2:
        ho4 = "supports"
    elif o4["vs_static"]["mean_diff"] < 0 and o4["vs_ew"]["mean_diff"] < 0:
        ho4 = "contradicts"
    else:
        ho4 = "inconclusive"
    v["H-BOS4"] = (ins4, ho4, {"arm": "pegged", "in_sample": {k: i4[k] for k in ("sr_alloc", "sr_static", "sr_ew", "vs_static", "vs_ew")},
                               "holdout": {k: o4[k] for k in ("sr_alloc", "sr_static", "sr_ew", "vs_static", "vs_ew")},
                               "dsr_in_sample": a["dsr_in_sample"]["dsr"], "pbo_phi": a["pbo_phi"]["pbo"], "phi_picks": [b["phi"] for b in a["blocks"]],
                               "market_holdout": {k: res["h4"]["market"]["clean_holdout"][k] for k in ("sr_alloc", "sr_static", "sr_ew", "vs_static", "vs_ew")}})
    hv = res["h5_verdicts"]
    v["H-BOS5"] = (hv["llm_in_sample"], hv["clean_holdout"],
                   {c: {s: {k: res["h5"][c][s].get(k) for k in ("r-m", "r-p", "p-m", "missed")} for s in A.SEGS} for c in res["h5"]})
    return v


def record_all(res: dict) -> dict:
    from ai_trader.research.hypotheses import record

    v = verdicts(res)
    for hid, (ins, ho, m) in v.items():
        record(hid, "llm_in_sample", ins, m, note="block BOS; design frozen in the diary 2026-11-15 18:05 UTC", data=DATA)
        record(hid, "clean_holdout", ho, m, note="holdout 2026-07-01..2026-11-15, read once", data=DATA)
        pooled = "supports" if ins == ho == "supports" else ("contradicts" if "contradicts" in (ins, ho) else "inconclusive")
        record(hid, "pooled", pooled, {"llm_in_sample": ins, "clean_holdout": ho}, note="pooled = supports iff both support; contradicts iff either contradicts", data=DATA)
    return {k: (a, b) for k, (a, b, _) in v.items()}


if __name__ == "__main__" and sys.argv[1:] == ["verdicts"]:
    print({k: (a, b) for k, (a, b, _) in verdicts(results()).items()})
