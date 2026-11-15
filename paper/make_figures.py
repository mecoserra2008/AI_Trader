"""Figures for paper/ai_trader_paper.tex. Read-only on the repo; writes only into paper/figures/.

    PYTHONIOENCODING=utf-8 uv run --extra viz python paper/make_figures.py

New figures are computed from research/hypotheses.yaml + research/hypotheses_results.jsonl (the same status and
calibration rules as scripts/knowledge_book.py), from the report tables cited in each caption, and from the live
score journal data/journal/live_nn/scores.jsonl. Reused figures are copied unchanged from research/figures/.
"""

from __future__ import annotations

import json
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "paper" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ROOT / "scripts"))

INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e2dc"
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED = (
    "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948")
GRAY, LIGHT = "#9a9890", "#d4d2cc"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK2,
    "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "savefig.dpi": 200, "savefig.bbox": "tight", "legend.frameon": False,
})

# ------------------------------------------------------------------ registry statistics

GROUPS = {
    "Volatility and calendar": ["vol", "vol_model", "leverage", "volatility", "calendar", "macro_event"],
    "ML and probabilistic models": ["ml_models", "ml_sparsity", "ml_interpretability", "state_space_discrete",
                                    "btc_field", "btc_pnn", "synthetic", "simulation", "hypothesis_mining"],
    "Direction and price signals": ["levels", "auction", "regime", "shock", "vol_signal", "positioning",
                                    "fundamentals", "relative_value", "carry", "decision_budget", "state_space",
                                    "panel"],
    "Strategies and trend": ["trend", "trend_breakout", "vol_bos_trend", "new_strategies", "pnl_attribution",
                             "regime_strategy", "tradfi_session"],
    "Execution and market making": ["execution", "mm_replay", "adverse_selection", "xvenue_arb"],
    "Microstructure and order flow": ["microstructure", "microstructure_dist", "order_flow", "flow", "btc_orderbook"],
    "Options": ["options", "options_compression"],
    "News, sentiment, geopolitics": ["btc_news", "geopolitics"],
    "Post-crash (liquidation flush)": ["post_cascade"],
    "Cross-section (XS v3)": ["xs_network"],
    "Live, AI decisions, brain, vecopen": ["ai_decision", "live_desk", "btc_live", "brain", "vecopen"],
}
STATUS_ORDER = ["supported", "flip", "refuted", "open", "forward", "untested"]
STATUS_COLOR = {"supported": AQUA, "flip": YELLOW, "refuted": RED, "open": GRAY, "forward": BLUE, "untested": LIGHT}
STATUS_LABEL = {"supported": "supported (both segments)", "flip": "segment flip", "refuted": "refuted",
                "open": "open / inconclusive", "forward": "forward test running", "untested": "registered, untested"}


def registry_stats() -> dict:
    import knowledge_book as kb  # read-only helpers: load_sources, latest_verdicts, status_of, brier_block

    S = kb.load_sources()
    reg = S["reg"]
    lat, first, _ = kb.latest_verdicts(S["rows"])
    fwd = {t["hypothesis"] for t in S["fwd"]["tests"]}
    stat = {h: kb.status_of(lat.get(h, {}), reg[h], fwd, h) for h in reg}
    fam_of = {h: reg[h].get("family", "?") for h in reg}
    group_of_fam = {f: g for g, fs in GROUPS.items() for f in fs}
    missing = sorted({f for f in fam_of.values() if f not in group_of_fam})
    if missing:
        raise SystemExit(f"families without a group: {missing}")
    by_group: dict[str, Counter] = defaultdict(Counter)
    for h in reg:
        by_group[group_of_fam[fam_of[h]]][stat[h]] += 1
    strict = [(reg[h]["prior"], kb.outcome_strict(stat[h])) for h in reg if kb.outcome_strict(stat[h]) is not None]
    bins = []
    for lo, hi in [(0.0, 0.3), (0.3, 0.45), (0.45, 0.6), (0.6, 1.01)]:
        sel = [(a, b) for a, b in strict if lo <= a < hi]
        bins.append(kb.brier_block(sel))
    seg = defaultdict(Counter)
    for h, v in lat.items():
        for s, x in v.items():
            seg[s][x] += 1
    wave = defaultdict(list)
    for h in reg:
        o = kb.outcome_strict(stat[h])
        if o is None:
            continue
        d = first.get(h, "")
        wave[d].append((reg[h]["prior"], o))
    return {"n": len(reg), "rows": len(S["rows"]), "rules": len(S["rules"]), "status": Counter(stat.values()),
            "by_group": by_group, "strict": kb.brier_block(strict), "bins": bins, "seg": seg, "wave": wave}


def fig_verdicts(st: dict) -> None:
    groups = list(GROUPS)[::-1]
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    left = np.zeros(len(groups))
    for s in STATUS_ORDER:
        vals = np.array([st["by_group"][g][s] for g in groups], float)
        ax.barh(groups, vals, left=left, color=STATUS_COLOR[s], edgecolor="white", linewidth=1.2, height=0.7,
                label=STATUS_LABEL[s])
        for i, v in enumerate(vals):
            if v >= 3:
                ax.text(left[i] + v / 2, i, f"{int(v)}", ha="center", va="center", fontsize=7,
                        color="white" if s in ("refuted", "forward", "supported") else INK)
        left += vals
    for i, g in enumerate(groups):
        ax.text(left[i] + 0.6, i, f"{int(left[i])}", va="center", fontsize=7.5, color=INK2)
    ax.set_xlabel("registered hypotheses")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower right", fontsize=7.5, ncol=1)
    c = st["status"]
    ax.set_title(f"{st['n']} registered hypotheses by programme and protocol status "
                 f"(supported {c['supported']}, refuted {c['refuted']}, flip {c['flip']})", fontsize=9, color=INK,
                 loc="left")
    fig.savefig(OUT / "verdicts_by_group.png")
    plt.close(fig)


def fig_calibration(st: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), gridspec_kw={"width_ratios": [1.1, 1]})
    ax = axes[0]
    b = [x for x in st["bins"] if x.get("n")]
    mp = [x["mean_prior"] for x in b]
    hr = [x["hit_rate"] for x in b]
    ns = [x["n"] for x in b]
    ax.plot([0, 1], [0, 1], color=GRAY, lw=1, ls="--", label="perfect calibration")
    ax.plot(mp, hr, color=BLUE, lw=2, marker="o", ms=7, label="our priors (strict outcome)")
    for x, y, n in zip(mp, hr, ns):
        ax.annotate(f"n={n}", (x, y), textcoords="offset points", xytext=(6, -10), fontsize=7, color=INK2)
    s = st["strict"]
    ax.scatter([s["mean_prior"]], [s["hit_rate"]], s=70, color=ORANGE, zorder=5, label="all decided")
    ax.annotate(f"mean prior {s['mean_prior']:.2f}\nhit rate {s['hit_rate']:.2f}", (s["mean_prior"], s["hit_rate"]),
                textcoords="offset points", xytext=(-8, 18), fontsize=7.5, color=INK, ha="right")
    ax.set_xlim(0, 0.85)
    ax.set_ylim(0, 0.85)
    ax.set_xlabel("prior P(hypothesis holds), bin mean")
    ax.set_ylabel("share supported")
    ax.set_title(f"Prior reliability (n={s['n']}, AUC {s['auc']:.2f})", fontsize=9, loc="left", color=INK)
    ax.legend(fontsize=7, loc="upper left")
    ax = axes[1]
    segs = [("llm_in_sample", "in-sample"), ("clean_holdout", "holdout"), ("pooled", "pooled")]
    verd = [("supports", AQUA), ("inconclusive", GRAY), ("contradicts", RED)]
    x = np.arange(len(segs))
    w = 0.26
    for j, (v, col) in enumerate(verd):
        vals = [st["seg"][s][v] for s, _ in segs]
        ax.bar(x + (j - 1) * w, vals, w * 0.92, color=col, label=v)
        for xi, val in zip(x + (j - 1) * w, vals):
            ax.text(xi, val + 1, str(val), ha="center", fontsize=7, color=INK2)
    ax.set_xticks(x, [lab for _, lab in segs])
    ax.set_ylabel("hypotheses (latest verdict)")
    ax.set_title("Latest verdict per segment", fontsize=9, loc="left", color=INK)
    ax.legend(fontsize=7)
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    fig.savefig(OUT / "calibration_now.png")
    plt.close(fig)


# ------------------------------------------------------------------ model figures (numbers from the cited reports)


def fig_field_pnn() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9))
    # btc_field_2026-11-13.md, H-FLD1 table: log-score differences (nats per origin), field minus B0 and minus B1
    ax = axes[0]
    hz = ["1h", "4h", "24h"]
    b0 = {"in": [0.0330, 0.0270, 0.0093], "ho": [0.0511, 0.0623, 0.0226]}
    b1 = {"in": [0.0038, -0.0110, -0.0357], "ho": [0.0024, -0.0069, -0.0371]}
    x = np.arange(3)
    w = 0.2
    ax.bar(x - 1.5 * w, b0["in"], w * 0.92, color=BLUE, label="vs log-HAR-t + jumps, in-sample")
    ax.bar(x - 0.5 * w, b0["ho"], w * 0.92, color=BLUE, alpha=0.55, label="vs log-HAR-t + jumps, holdout")
    ax.bar(x + 0.5 * w, b1["in"], w * 0.92, color=ORANGE, label="vs log-HAR-t x hour-of-week, in-sample")
    ax.bar(x + 1.5 * w, b1["ho"], w * 0.92, color=ORANGE, alpha=0.55, label="vs log-HAR-t x hour-of-week, holdout")
    ax.axhline(0, color=INK2, lw=0.8)
    ax.set_xticks(x, hz)
    ax.set_ylabel("log-score gain of the field (nats)")
    ax.set_title("Probability field (H-FLD1)", fontsize=9, loc="left", color=INK)
    ax.legend(fontsize=6.2, loc="lower left")
    ax.grid(axis="x", visible=False)
    # btc_pnn_2026-11-13.md, H-PNN1 table: pinball differences (x1e3), paired per date, NW t in labels
    ax = axes[1]
    lab = ["1h in", "1h hold", "1d in", "1d hold"]
    pnn_har = [-0.82, -1.24, 2.44, -8.19]
    mlp_har = [-0.83, -0.97, 0.20, -7.22]
    pnn_mlp = [0.02, -0.27, 2.24, -0.98]
    t_ph = [-7.26, -2.95, 1.75, -1.84]
    x = np.arange(4)
    ax.bar(x - w, pnn_har, w * 0.92, color=VIOLET, label="PNN - log-HAR-t")
    ax.bar(x, mlp_har, w * 0.92, color=AQUA, label="MLP - log-HAR-t")
    ax.bar(x + w, pnn_mlp, w * 0.92, color=GRAY, label="PNN - MLP")
    for xi, v, t in zip(x - w, pnn_har, t_ph):
        ax.text(xi + w, 3.0, f"PNN-HAR t {t:+.1f}", ha="center", fontsize=6.3, color=INK2)
    ax.axhline(0, color=INK2, lw=0.8)
    ax.set_ylim(-9, 3.6)
    ax.set_xticks(x, lab)
    ax.set_ylabel("pinball difference (x 1e-3)")
    ax.set_title("Polynomial-neuron network (H-PNN1)", fontsize=9, loc="left", color=INK)
    ax.legend(fontsize=6.5, loc="lower left")
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    fig.savefig(OUT / "field_pnn_scores.png")
    plt.close(fig)


def fig_live() -> None:
    rows = [json.loads(x) for x in (ROOT / "data/journal/live_nn/scores.jsonl").read_text(encoding="utf-8").splitlines()
            if x.strip()]
    from datetime import datetime

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8), gridspec_kw={"width_ratios": [2.2, 1]})
    ax = axes[0]
    h1 = [r for r in rows if r["horizon"] == "1h" and r.get("z") is not None]
    t = [datetime.fromisoformat(r["window_start"]) for r in h1]
    z = [r["z"] for r in h1]
    hit = [r.get("hit_aci80") for r in h1]
    rep = ["replay" in r.get("flags", []) for r in h1]
    for ti, zi, hi, re in zip(t, z, hit, rep):
        ax.scatter([ti], [zi], s=16, color=(BLUE if hi else RED), marker=("o" if not re else "s"),
                   edgecolor="white", linewidth=0.4, zorder=3)
    ax.axhline(1.2816, color=GRAY, ls="--", lw=0.8)
    ax.axhline(-1.2816, color=GRAY, ls="--", lw=0.8)
    ax.axhline(0, color=INK2, lw=0.6)
    ax.set_ylabel("realised 1h return / network sigma")
    n1, k1 = len(h1), sum(bool(x) for x in hit)
    ax.set_title(f"Live v1, 1h: {k1}/{n1} inside ACI 80% (red = outside)",
                 fontsize=8.5, loc="left", color=INK)
    ax.tick_params(axis="x", labelsize=7, rotation=20)
    ax = axes[1]
    labels, cov, nom = [], [], []
    for hz in ("1h", "24h"):
        rr = [r for r in rows if r["horizon"] == hz]
        for lv in (80, 90, 95):
            k = f"hit_aci{lv}"
            vals = [r[k] for r in rr if r.get(k) is not None]
            if vals:
                labels.append(f"{hz}\n{lv}%")
                cov.append(sum(vals) / len(vals))
                nom.append(lv / 100)
    x = np.arange(len(labels))
    ax.bar(x, cov, 0.6, color=[BLUE] * 3 + [ORANGE] * (len(labels) - 3))
    ax.scatter(x, nom, marker="_", s=200, color=INK, zorder=4, label="nominal")
    ax.set_xticks(x, labels, fontsize=7)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("coverage")
    ax.set_title("ACI coverage", fontsize=8.5, loc="left", color=INK)
    ax.legend(fontsize=7, loc="lower right")
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    fig.savefig(OUT / "live_v1_scores.png")
    plt.close(fig)
    print("live scored lines:", Counter(r["horizon"] for r in rows), "coverage:", dict(zip(labels, cov)))


TIMELINE = {  # research/diary/2026-11-07..15 (entry times in UTC); "|" = line break
    "11-07": {"infra": "lake; 3 s order-flow|collector (70 streams);|updater daemon",
              "research": "cutoff split; weekend|study; robust_bc vol", "live": "", "mm": ""},
    "11-08": {"infra": "risk-engine gate on|demo; liquidations", "research": "trend zoo (DSR 0.72)",
              "live": "demo forward test;|AI/RULE/RANDOM card", "mm": "first mainnet MM|(H-MM1); multi-asset|MM (H-MM2)"},
    "11-09": {"infra": "", "research": "", "live": "", "mm": "H-MM2 stopped by|the operator"},
    "11-10": {"infra": "", "research": "volatility programme|(27 hyps); strategy|programme (22 hyps)",
              "live": "", "mm": "illiquid-20 MM run,|shut down"},
    "11-11": {"infra": "mm.rs sigma_last fix", "research": "programme 2 synthesis;|microstructure pilot",
              "live": "", "mm": "NVDA quoters|on mainnet"},
    "11-12": {"infra": "forward-test|runners (12)", "research": "overnight ML; multi-|day micro; FDR mining",
              "live": "H-NN10 vol probation", "mm": "research-only|decision: all live|bots off"},
    "11-13": {"infra": "Rust intent risk|engine, PaperBook,|at-match", "research": "knowledge book; BTC|field, PNN, news,|OB generator",
              "live": "live network v1 +|Claude forecast log", "mm": ""},
    "11-14": {"infra": "vecopen phases 0-4|(G1 leak audit pass)", "research": "MM replay; options|after compression",
              "live": "4-hourly paper|decision harness", "mm": ""},
    "11-15": {"infra": "dashboard, brain map;|live limits in at-risk", "research": "H-EXE1; post-cascade;|XS v3; brain atlas",
              "live": "first paper trade|-1.28 R; v2.1 shadow;|live desk authorised", "mm": ""},
}


def fig_timeline() -> None:
    lanes = [("infra", "Infrastructure", BLUE), ("research", "Research", VIOLET),
             ("live", "Forward and paper", AQUA), ("mm", "Mainnet MM", ORANGE)]
    days = list(TIMELINE)
    fig, ax = plt.subplots(figsize=(7.2, 6.4))
    ax.set_xlim(-0.05, len(lanes))
    ax.set_ylim(len(days) - 0.5, -0.9)
    ax.axis("off")
    for j, (_, title, col) in enumerate(lanes):
        ax.text(j + 0.05, -0.75, title, fontsize=8.5, color=col, fontweight="bold", va="center")
    for i, d in enumerate(days):
        ax.axhline(i - 0.5, color=GRID, lw=0.6)
        ax.text(-0.02, i, "2026-" + d, fontsize=7.5, color=INK2, ha="right", va="center")
        for j, (key, _, col) in enumerate(lanes):
            txt = TIMELINE[d][key]
            if txt:
                ax.scatter([j + 0.06], [i], s=22, color=col, zorder=3)
                ax.text(j + 0.13, i, txt.replace("|", chr(10)), fontsize=6.6, color=INK, va="center", linespacing=1.1)
    fig.savefig(OUT / "timeline.png")
    plt.close(fig)


REUSE = {
    "vol_tournament_crypto_daily.png": "vol/tournament_crypto_daily.png",
    "vol_hour_of_week_crypto.png": "vol/hour_of_week_crypto.png",
    "pnl_effective_bets.png": "pnl/effective_bets.png",
    "selection_overfit_pbo.png": "strategies/selection_overfit_pbo.png",
    "mm_replay_decomposition.png": "mm_replay_2026-11-14/decomposition.png",
    "opx1_straddle.png": "options_compression_2026-11-14/h_opx1_straddle.png",
    "post_flush_path.png": "post_cascade_2026-11-15/post_flush_path.png",
    "xs_ic_vs_baselines.png": "xs_network_2026-11-15/ic_vs_baselines.png",
    "brain_receptive_fields.png": "brain_atlas_2026-11-15/receptive_fields.png",
    "brain_hour_of_week.png": "brain_atlas_2026-11-15/hour_of_week.png",
}


def main() -> None:
    st = registry_stats()
    print("registry:", st["n"], "hypotheses,", st["rows"], "result rows,", st["rules"], "rules; status", dict(st["status"]))
    print("strict calibration:", st["strict"])
    print("by group:", {g: dict(c) for g, c in st["by_group"].items()})
    fig_verdicts(st)
    fig_calibration(st)
    fig_field_pnn()
    fig_live()
    fig_timeline()
    for dst, src in REUSE.items():
        shutil.copyfile(ROOT / "research" / "figures" / src, OUT / dst)
    json.dump({"status": st["status"], "strict": st["strict"], "bins": st["bins"],
               "by_group": {g: dict(c) for g, c in st["by_group"].items()},
               "seg": {k: dict(v) for k, v in st["seg"].items()}, "n": st["n"], "rows": st["rows"], "rules": st["rules"]},
              open(OUT / "registry_stats.json", "w"), indent=1, default=dict)


if __name__ == "__main__":
    main()
