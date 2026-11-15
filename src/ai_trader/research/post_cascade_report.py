"""Report for the post-cascade study (agent RB, 2026-11-15): research/reports/post_cascade_2026-11-15.{md,json} + figures.

    PYTHONIOENCODING=utf-8 uv run --extra viz python -m ai_trader.research.post_cascade_study report
"""

from __future__ import annotations

import json
import math
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from ai_trader.research.post_cascade_study import CELLS, D, FIG, OUT, REPORT, ROOT, verdict_rb3, verdict_rb12, write_json
from ai_trader.research.regimes import CELL_LABELS

# reference palette (dataviz skill, light mode)
BLUE, ORANGE, AQUA, GREY, INK, INK2 = "#2a78d6", "#eb6834", "#1baf7a", "#a3a29c", "#0b0b0b", "#52514e"
SURF = "#fcfcfb"
SEGN = {"llm_in_sample": "llm_in_sample (pre 2026-07-01)", "clean_holdout": "clean_holdout (2026-07-01 on)"}


def _ci(o: dict, k: int = 2, sign: bool = True) -> str:
    if not o or "lo" not in o:
        return "n/a"
    f = f"{{:{'+' if sign else ''}.{k}f}}"
    return f"{f.format(o['mean'] if 'mean' in o else o['diff'])} [{f.format(o['lo'])}, {f.format(o['hi'])}]"


def _pct(x: float | None, k: int = 1) -> str:
    return "n/a" if x is None else f"{100 * x:+.{k}f}%"


def _share(m: dict, c: str) -> str:
    return f"{100 * m[c] / m['n']:.0f}%" if m.get("n") else "n/a"


# ------------------------------------------------------------------ figures
def _style(plt) -> None:
    plt.rcParams.update({"figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF, "axes.edgecolor": GREY,
                         "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "text.color": INK, "font.size": 9,
                         "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": "#e6e5e0",
                         "grid.linewidth": 0.6, "axes.titlesize": 10, "axes.titleweight": "bold", "legend.frameon": False})


def figures(res: dict, events: list[dict]) -> list[tuple[str, str]]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _style(plt)
    FIG.mkdir(parents=True, exist_ok=True)
    out = []

    # 1. events per year
    fig, ax = plt.subplots(figsize=(6.4, 2.8))
    yrs = sorted(res["counts_year"])
    tot = [res["counts_year"][y] for y in yrs]
    btc = [sum(1 for e in events if e["symbol"] == "BTCUSDT" and str(e["year"]) == y) for y in yrs]
    x = np.arange(len(yrs))
    ax.bar(x - 0.2, tot, 0.38, color=BLUE, label="all 23 assets")
    ax.bar(x + 0.2, btc, 0.38, color=ORANGE, label="BTC")
    for i, (a, b) in enumerate(zip(tot, btc, strict=True)):
        ax.text(i - 0.2, a + 2, str(a), ha="center", fontsize=8, color=INK2)
        ax.text(i + 0.2, b + 2, str(b), ha="center", fontsize=8, color=INK2)
    ax.set_xticks(x, yrs)
    ax.set_ylabel("flush events")
    ax.set_title("Flush events per year (declustered, one per asset per 7 days)")
    ax.legend(loc="upper left")
    p = FIG / "events_per_year.png"
    fig.tight_layout()
    fig.savefig(p, dpi=130)
    plt.close(fig)
    out.append((p.name, "Flush events per year, pooled and BTC"))

    # 2. net R with CIs
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.2), sharey=True)
    for ax, hid, title in ((axes[0], "H-RB1", "H-RB1 rebound long"), (axes[1], "H-RB2", "H-RB2 short the failed rebound")):
        rows = []
        for seg in ("llm_in_sample", "clean_holdout"):
            o = res[hid][f"pooled|{seg}"]
            rows += [(f"{seg}\nevent", o["R"], BLUE), (f"{seg}\nmatched random", o["baseline_R"], GREY), (f"{seg}\nevent - matched", o["diff"], ORANGE)]
        for i, (lab, o, col) in enumerate(rows):
            if "lo" in o:
                ax.errorbar(o["mean"], i, xerr=[[o["mean"] - o["lo"]], [o["hi"] - o["mean"]]], fmt="o", color=col, ms=6, capsize=3, lw=1.6)
        ax.axvline(0, color=INK2, lw=0.8)
        ax.set_yticks(range(len(rows)), [r[0] for r in rows], fontsize=7.5)
        ax.invert_yaxis()
        ax.set_xlabel("mean net R per trade (95% week-cluster CI)")
        ax.set_title(title)
    p = FIG / "net_r.png"
    fig.tight_layout()
    fig.savefig(p, dpi=130)
    plt.close(fig)
    out.append((p.name, "Mean net R per trade with 95% CIs, events vs regime-matched random entries, by segment"))

    # 3. transition heatmap
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 2.9))
    for ax, scope in zip(axes, ("pooled", "BTC"), strict=True):
        m = res["transitions"][scope]
        labs = ["+1d", "+3d", "+7d", "next_after_R9"]
        M = np.array([[m[lab][c] / max(m[lab]["n"], 1) for c in CELLS] for lab in labs])
        ax.imshow(M, cmap="Blues", vmin=0, vmax=1, aspect="auto")
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                if M[i, j] >= 0.005:
                    ax.text(j, i, f"{100 * M[i, j]:.0f}", ha="center", va="center", fontsize=7.5, color="white" if M[i, j] > 0.5 else INK)
        ax.set_xticks(range(9), CELLS)
        ax.set_yticks(range(4), [f"{lab} (n {m[lab]['n']})" for lab in labs])
        ax.set_title(f"v0 cell after a flush, {scope} (% of events)")
        ax.grid(False)
    p = FIG / "transitions.png"
    fig.tight_layout()
    fig.savefig(p, dpi=130)
    plt.close(fig)
    out.append((p.name, "v0 cell at +1/+3/+7 days after flush events and the first cell after the R9 episode ends"))

    # 4. path distributions
    P = res["paths"]["pooled"]["raw"]
    B = res["paths"]["BTC"]["raw"]
    fig, axes = plt.subplots(1, 3, figsize=(9.2, 2.8))
    for ax, k, lab, sc in ((axes[0], "mdd", "further drop to the 7-day low (%)", 100), (axes[1], "rebound", "rebound from the low to the later high (%)", 100),
                           (axes[2], "hours_to_low", "hours from the event close to the 7-day low", 1)):
        x = np.array(P[k]) * sc
        ax.hist(x, bins=30, color=BLUE, alpha=0.85, label="all assets")
        ax.axvline(np.median(x), color=INK, lw=1.2, ls="--")
        ax.axvline(np.median(np.array(B[k]) * sc), color=ORANGE, lw=1.6)
        ax.set_xlabel(lab)
        ax.text(0.02 if k == "mdd" else 0.98, 0.95, f"median all {np.median(x):.1f}\nmedian BTC {np.median(np.array(B[k]) * sc):.1f}", transform=ax.transAxes,
                ha="left" if k == "mdd" else "right", va="top", fontsize=7.5, color=INK2)
    axes[0].set_ylabel("events")
    fig.suptitle("Post-flush 7-day path (dashed = pooled median, orange = BTC median)", fontsize=10, fontweight="bold")
    p = FIG / "post_flush_path.png"
    fig.tight_layout()
    fig.savefig(p, dpi=130)
    plt.close(fig)
    out.append((p.name, "Distribution of the post-flush path: further drawdown, rebound size, time to the low"))

    # 5. current episode BTC: price, OI, funding, liquidations (small multiples, shared time axis)
    a = res["current"]["assets"]["BTCUSDT"]
    ts = [datetime.fromtimestamp(t / 1000, tz=UTC) for t in a["h1"]["ts"]]
    bars = a["bars"]
    bts = [datetime.fromisoformat(r["event_ts"]) for r in bars]
    fig, axes = plt.subplots(4, 1, figsize=(7.6, 7.2), sharex=True, gridspec_kw={"height_ratios": [2.2, 1, 1, 1.4]})
    axes[0].plot(ts, a["h1"]["c"], color=BLUE, lw=1.6)
    for e in a["flush_events"]:
        et = datetime.fromisoformat(e["ts"])
        axes[0].axvline(et, color=ORANGE, lw=1.2)
        axes[0].text(et, max(a["h1"]["c"]), " flush event (4h bar)", color=INK2, fontsize=7.5, va="top")
    for lr in a.get("live_rb1", []):
        axes[0].axhline(lr["stop"], color=GREY, lw=0.9, ls="--")
        axes[0].text(ts[0], lr["stop"], " H-RB1 stop", fontsize=7, color=INK2, va="bottom")
    axes[0].set_ylabel("BTCUSDT 1h close")
    axes[0].set_title("BTC, 2026-11-12 to now: price, open interest, funding, liquidation prints")
    axes[1].plot(bts, [r["open_interest"] / 1e3 for r in bars], color=AQUA, lw=1.6, marker="o", ms=3)
    axes[1].set_ylabel("OI (k BTC, 4h)")
    axes[2].plot(bts, [1e4 * r["funding_rate"] for r in bars], color=BLUE, lw=1.4, drawstyle="steps-post", label="funding")
    axes[2].plot(bts, [1e4 * r["f_thr"] for r in bars], color=GREY, lw=1.0, ls="--", drawstyle="steps-post", label="90d 10th pct")
    axes[2].axhline(0, color=INK2, lw=0.6)
    axes[2].set_ylabel("funding (bp/print)")
    axes[2].legend(loc="upper right", fontsize=7)
    lq = a.get("liq_1h", [])
    lt = sorted({r["h1"] for r in lq})
    lmap = {(r["h1"], r["side"]): r["usd"] for r in lq}
    xt = [datetime.fromisoformat(t) for t in lt]
    w = 1 / 24 * 0.8
    axes[3].bar(xt, [lmap.get((t, "long"), 0) / 1e6 for t in lt], width=w, color=ORANGE, label="longs liquidated")
    axes[3].bar(xt, [-lmap.get((t, "short"), 0) / 1e6 for t in lt], width=w, color=BLUE, label="shorts liquidated")
    axes[3].axhline(0, color=INK2, lw=0.6)
    axes[3].set_ylabel("liq. USD m / h")
    axes[3].legend(loc="lower left", fontsize=7)
    fig.autofmt_xdate()
    p = FIG / "current_btc.png"
    fig.tight_layout()
    fig.savefig(p, dpi=130)
    plt.close(fig)
    out.append((p.name, "BTC current episode: 1h price with the flush event, OI, funding vs its 90d 10th percentile, hourly liquidation prints"))
    return out


# ------------------------------------------------------------------ rules
def rule_candidates(res: dict) -> list[dict[str, Any]]:
    r1i, r1h = res["H-RB1"]["pooled|llm_in_sample"], res["H-RB1"]["pooled|clean_holdout"]
    r2i, r2h = res["H-RB2"]["pooled|llm_in_sample"], res["H-RB2"]["pooled|clean_holdout"]
    r3i, r3h = res["H-RB3"]["pooled|llm_in_sample"], res["H-RB3"]["pooled|clean_holdout"]
    tp, th = res["transitions"]["pooled"], res["transitions_by_seg"]["clean_holdout"]
    pp, pb, ph = res["paths"]["pooled"], res["paths"]["BTC"], res["paths_by_seg"]["clean_holdout"]
    rep = "research/reports/post_cascade_2026-11-15.md"
    flush = ("flush event: 4h close with 48h log return below its trailing-1y 5th pct, 4h OI >= 5% below its 72h high, v0 cell R9 "
             "(post_cascade_study.flush_events)")
    return [
        {"id": "R-RB-1", "kind": "avoid",
         "rule": "WHEN a crypto perp has just printed a flush (48h return below its 1y 5th pct, OI down >= 5% from its 72h high, v0 R9) THEN do not buy the next 4h close expecting a rebound: the low usually comes later",
         "when": flush + "; candidate is a long entered within the first 4h after the event",
         "then": "expect a negative net R for a rebound long at the next 4h close with a stop under the flush low (2R target, 5-day time stop), and worse than a random long in the same R9 cell",
         "scope": {"assets": "crypto", "horizon": "5 days", "sides": "long"},
         "effect": f"net R per trade, 7.5 bp/side: llm_in_sample {_ci(r1i['R'])} (n {r1i['R']['n']}, {r1i['R']['clusters']} weeks), clean_holdout {_ci(r1h['R'])} (n {r1h['R']['n']}, {r1h['R']['clusters']} weeks); minus matched R9 random longs {_ci(r1i['diff'])} / {_ci(r1h['diff'])}; pegged-limit entry changes it by < 0.01 R",
         "confidence": "B", "evidence": ["H-RB1", rep],
         "review": "a registered later-entry variant (e.g. after the R9 episode ends or after a confirmed higher low) tested the same way; 20+ more holdout events",
         "status": "active"},
        {"id": "R-RB-2", "kind": "expect",
         "rule": "WHEN a flush has just printed THEN expect the 7-day low still ahead (median a further 5-9% down, usually one to three days later and later still in the 2026 holdout) followed by a rebound of about 8-15% off that low, with no net drift over the week",
         "when": flush,
         "then": "describes the path from the event close over 7 days (1h bars); use it to set expectations for timing, not as a direction signal",
         "scope": {"assets": "crypto", "horizon": "7 days", "sides": "both"},
         "effect": f"pooled n {pp['n']}: further drop to the 7-day low median {_pct(pp['mdd']['median'])} (IQR {_pct(pp['mdd']['q25'])} to {_pct(pp['mdd']['q75'])}), "
                   f"low after median {pp['hours_to_low']['median']:.0f} h (only {100 * pp['share_low_in_first_24h']:.0f}% within 24 h), rebound from the low median {_pct(pp['rebound']['median'])}, 7-day return median {_pct(pp['ret7d']['median'])}; "
                   f"BTC n {pb['n']}: {_pct(pb['mdd']['median'])}, {pb['hours_to_low']['median']:.0f} h, {_pct(pb['rebound']['median'])}; clean_holdout n {ph['n']}: {_pct(ph['mdd']['median'])}, {ph['hours_to_low']['median']:.0f} h, {_pct(ph['rebound']['median'])}",
         "confidence": "C", "evidence": ["H-RB1", rep],
         "review": "descriptive, mostly in-sample; re-check the holdout path quantiles at 50+ holdout events",
         "status": "provisional"},
        {"id": "R-RB-3", "kind": "expect",
         "rule": "WHEN the R9 episode that contains a flush ends THEN expect R6 whipsaw next (about 80%), otherwise R8 trend down; R9 itself usually lasts another 1-2.5 days after the flush",
         "when": flush + "; v0 cell on 4h bars (regimes.label, bar_hours 4)",
         "then": "describes the state path; refines R-PX-8 (62% whipsaw after R9 in general) for flush-led R9 episodes",
         "scope": {"assets": "crypto", "horizon": "1-7 days", "sides": "both"},
         "effect": f"first cell after R9: R6 {_share(tp['next_after_R9'], 'R6')}, R8 {_share(tp['next_after_R9'], 'R8')} (n {tp['next_after_R9']['n']}; BTC R6 {_share(res['transitions']['BTC']['next_after_R9'], 'R6')}, clean_holdout R6 {_share(th['next_cell'], 'R6')} of {th['next_cell']['n']}); "
                   f"R9 hours left after the event median {tp['r9_hours_left_median']:.0f} (IQR {tp['r9_hours_left_q'][0]:.0f}-{tp['r9_hours_left_q'][1]:.0f}); cell at +7d: R6 {_share(tp['+7d'], 'R6')}, R9 {_share(tp['+7d'], 'R9')}, R5 {_share(tp['+7d'], 'R5')}, R8 {_share(tp['+7d'], 'R8')}",
         "confidence": "B", "evidence": ["H-RB1", "H-RB2", "H-RL3", rep],
         "review": "the same matrix on daily cells, or a holdout share of R6 below 60% at 40+ holdout events",
         "status": "active"},
        {"id": "R-RB-4", "kind": "avoid",
         "rule": "WHEN funding drops below its 90-day 10th percentile while open interest rebuilds >= 3% within 72h of a flush THEN don't expect a short squeeze from it: the next 24h is if anything weaker than after flushes without that pattern",
         "when": flush + "; then at a 4h close within 72h: latest funding print < trailing-90d 10th pct of prints and OI >= 3% above its post-flush low",
         "then": "treat new shorts piling in after a flush as no evidence for a rebound (in line with R-AVOID-1 / R-MAP-5 on funding-contrarian signals)",
         "scope": {"assets": "crypto", "horizon": "24-72h", "sides": "both"},
         "effect": f"pattern minus no-pattern vol-scaled return, llm_in_sample 24h {_ci(r3i['z24h'], 2)} sigma, 72h {_ci(r3i['z72h'], 2)} (pattern {r3i['n_pattern']} of {r3i['n_events']} events); clean_holdout 24h {_ci(r3h['z24h'], 2)}, 72h {_ci(r3h['z72h'], 2)} (pattern {r3h['n_pattern']} of {r3h['n_events']})",
         "confidence": "C", "evidence": ["H-RB3", rep],
         "review": "15+ holdout pattern events",
         "status": "provisional"},
        {"id": "R-RB-5", "kind": "expect",
         "rule": "WHEN shorting the first 4h close back below the EMA20 after a >= 1.5 sigma post-flush rebound THEN expect about breakeven net R: better than a random short in the same cell, but no edge yet",
         "when": flush + "; then within 7 days a close >= 1.5 sigma above the post-flush low, then a 4h close crossing below the 4h EMA20",
         "then": "a short here is not worse than other shorts in the whipsaw that follows, but there is no tested positive expectation",
         "scope": {"assets": "crypto", "horizon": "5 days", "sides": "short"},
         "effect": f"net R per trade: llm_in_sample {_ci(r2i['R'])} (n {r2i['R']['n']}), clean_holdout {_ci(r2h['R'])} (n {r2h['R']['n']}, {r2h['R']['clusters']} weeks); minus matched random shorts {_ci(r2i['diff'])} / {_ci(r2h['diff'])}; entries mostly in R6 ({r2i['entry_cells'].get('R6', 0)} of {r2i['R']['n']})",
         "confidence": "C", "evidence": ["H-RB2", rep],
         "review": "holdout CI clearing zero at 40+ trades, or a registered variant",
         "status": "provisional"},
    ]


# ------------------------------------------------------------------ markdown
def build() -> None:
    from ai_trader.research.rules import load_rules, validate

    res = json.loads((OUT / "results.json").read_text(encoding="utf-8"))
    events = json.loads((OUT / "events.json").read_text(encoding="utf-8"))
    figs = figures(res, events)
    rules = rule_candidates(res)
    errs = validate(rules)
    existing = [r for r in load_rules() if not str(r.get("id", "")).startswith("R-RB-")]
    errs_all = validate(existing + rules)
    if errs or errs_all:
        raise SystemExit(f"rule candidates invalid: {errs or errs_all}")
    md = markdown(res, events, figs, rules)
    REPORT.with_suffix(".md").write_text(md, encoding="utf-8")
    slim = {k: v for k, v in res.items() if k not in ("current", "paths")}
    slim["paths"] = {s: {k: v for k, v in p.items() if k != "raw"} for s, p in res["paths"].items()}
    cur = {}
    for s, a in res["current"]["assets"].items():
        cur[s] = {k: v for k, v in a.items() if k not in ("h1", "liq_1h")}
    slim["current"] = {"assets": cur, **{k: v for k, v in res["current"].items() if k != "assets"}}
    slim["verdicts"] = {h: {s: (verdict_rb3 if h == "H-RB3" else verdict_rb12)(res[h][f"pooled|{s}"]) for s in ("llm_in_sample", "clean_holdout")}
                        for h in ("H-RB1", "H-RB2", "H-RB3")}
    slim["figures"] = [f"research/figures/post_cascade_2026-11-15/{n}" for n, _ in figs]
    slim["rule_candidates"] = rules
    write_json(REPORT.with_suffix(".json"), slim)
    print(REPORT.with_suffix(".md"))


def _cell_row(m: dict) -> str:
    return " | ".join(f"{100 * m[c] / m['n']:.0f}" if m.get("n") and m[c] else "." for c in CELLS)


def markdown(res: dict, events: list[dict], figs: list[tuple[str, str]], rules: list[dict]) -> str:
    v = {h: {s: (verdict_rb3 if h == "H-RB3" else verdict_rb12)(res[h][f"pooled|{s}"]) for s in ("llm_in_sample", "clean_holdout")}
         for h in ("H-RB1", "H-RB2", "H-RB3")}
    r1i, r1h = res["H-RB1"]["pooled|llm_in_sample"], res["H-RB1"]["pooled|clean_holdout"]
    r2i, r2h = res["H-RB2"]["pooled|llm_in_sample"], res["H-RB2"]["pooled|clean_holdout"]
    r3i, r3h = res["H-RB3"]["pooled|llm_in_sample"], res["H-RB3"]["pooled|clean_holdout"]
    b1, b2, b3 = res["H-RB1"]["BTC|llm_in_sample"], res["H-RB2"]["BTC|llm_in_sample"], res["H-RB3"]["BTC|llm_in_sample"]
    tp, tb = res["transitions"]["pooled"], res["transitions"]["BTC"]
    th = res["transitions_by_seg"]["clean_holdout"]
    pp, pb, ph = res["paths"]["pooled"], res["paths"]["BTC"], res["paths_by_seg"]["clean_holdout"]
    cur = res["current"]["assets"]
    L: list[str] = []
    w = L.append
    w("# Post-cascade plays: rebound longs vs continuation shorts after a liquidation flush (H-RB1, H-RB2, H-RB3), 2026-11-15\n")
    w("Agent RB. Research only. The three hypotheses were registered before any test (research/hypotheses.yaml, block RB) and the design "
      "was frozen in the diary (2026-11-15, kind decision) and in `data/research/post_cascade/design.json` before any outcome was computed. "
      "Code: `src/ai_trader/research/post_cascade_study.py`, `post_cascade_report.py`. Universe L (23 Bybit USDT perps), 4h bars 2020-03 to "
      f"2026-11-15 04:00 UTC, funding and 4h open interest read point in time on known_ts, v0 regime cell from `regimes.label` on 4h bars. "
      "Liquidation prints (only from 2026-11-08) are used only to describe the current episode. Percent moves are log returns unless marked simple.\n")
    w("## Verdicts\n")
    w(f"- **H-RB1 (rebound long after a flush pays): contradicts**, in both segments. llm_in_sample: {v['H-RB1']['llm_in_sample']}; clean_holdout: {v['H-RB1']['clean_holdout']}. "
      f"Mean net R {_ci(r1i['R'])} before the cutoff and {_ci(r1h['R'])} after it. Both CIs are below zero, and the trade is also worse than a random long in the same R9 cell.")
    w(f"- **H-RB2 (short the failed rebound pays): inconclusive.** llm_in_sample: {v['H-RB2']['llm_in_sample']}; clean_holdout: {v['H-RB2']['clean_holdout']}. "
      f"About breakeven in-sample ({_ci(r2i['R'])}), positive but very noisy in the holdout ({_ci(r2h['R'])}, {r2h['R']['n']} trades in {r2h['R']['clusters']} weeks). It does beat random shorts in the same cell (which lose).")
    w(f"- **H-RB3 (negative funding + rebuilding OI after a flush precedes a squeeze): inconclusive**, leaning the other way. llm_in_sample: {v['H-RB3']['llm_in_sample']}; clean_holdout: {v['H-RB3']['clean_holdout']}. "
      f"In-sample the next 24h is weaker after the pattern ({_ci(r3i['z24h'])} sigma), not stronger; at 72h there is no difference ({_ci(r3i['z72h'])}).")
    w(f"- Trials: 3 hypotheses x 2 segments x 2 cost variants, plus BTC-only and descriptive extras. Treat |t| ~ 2 as interesting, not proven.\n")

    w("## In plain words\n")
    w(f"**What a flush is.** A 4h close where the 48h return is below the asset's own 1-year 5th percentile, open interest is at least 5% below its 72h high "
      f"(positions were closed, not just repriced) and the v0 cell is R9 (liquidation). One event per asset per 7 days. That gives **{res['n_events']} events** "
      f"({res['counts_seg'].get('llm_in_sample', 0)} before the cutoff, {res['counts_seg'].get('clean_holdout', 0)} after), {sum(1 for e in events if e['symbol'] == 'BTCUSDT')} of them on BTC.\n")
    w(f"**Buying the flush loses (H-RB1).** Going long at the next 4h close with the stop under the flush low (median stop {r1i['stop_sigma_median']:.1f} sigma, "
      f"{100 * r1i['capped_share']:.0f}% capped at 4 sigma), a 2R target and a 5-day time stop:")
    w(f"- llm_in_sample: {_ci(r1i['R'])} R per trade, {r1i['R']['n']} trades in {r1i['R']['clusters']} weeks; {100 * r1i['R']['hit']:.0f}% winners; exits: {r1i['exits']}.")
    w(f"- clean_holdout: {_ci(r1h['R'])} R, {r1h['R']['n']} trades in {r1h['R']['clusters']} weeks.")
    w(f"- A random long on a non-event R9 day of the same asset earns {r1i['diff']['baseline_mean']:+.2f} R in-sample, so the flush long is {_ci(r1i['diff'])} R worse "
      f"(holdout {_ci(r1h['diff'])}). The flush is a worse moment to buy than an ordinary R9 bar.")
    w(f"- Pegged-limit entries (maker 2.0 + 1.3 bp instead of 7.5 bp) change almost nothing: {_ci(r1i['R_peg'])} in-sample, {_ci(r1h['R_peg'])} holdout. Costs are not the problem; timing is.")
    w(f"- BTC alone: {_ci(b1['R'])} R over {b1['R']['n']} in-sample events (CI includes 0; no resolved holdout event yet).\n")
    w("**Why: the low is usually still ahead.** Over the 7 days after the event close (1h bars):")
    w(f"- price makes a further low a median **{_pct(pp['mdd']['median'])}** below the event close (IQR {_pct(pp['mdd']['q25'])} to {_pct(pp['mdd']['q75'])}); "
      f"{100 * pp['share_new_low_below_minus5pct']:.0f}% of events go at least 5% lower. BTC: median {_pct(pb['mdd']['median'])}, IQR {_pct(pb['mdd']['q25'])} to {_pct(pb['mdd']['q75'])}.")
    w(f"- the low comes a median **{pp['hours_to_low']['median']:.0f} h** after the event (BTC {pb['hours_to_low']['median']:.0f} h); only {100 * pp['share_low_in_first_24h']:.0f}% bottom within 24 h.")
    w(f"- from that low the rebound is large: median **{_pct(pp['rebound']['median'])}** (BTC {_pct(pb['rebound']['median'])}); {100 * pp['share_rebound_ge_10pct']:.0f}% rebound at least 10%.")
    w(f"- the time to the low is bimodal: either the event close is already near the bottom (a quarter of events bottom within {pp['hours_to_low']['q25']:.0f} h) or the decline grinds on and the low comes late in the week (a quarter after {pp['hours_to_low']['q75']:.0f} h).")
    w(f"- the 7-day return from the event close is a coin flip: median {_pct(pp['ret7d']['median'])}, {100 * pp['share_up_7d']:.0f}% up (BTC {_pct(pb['ret7d']['median'])}, {100 * pb['share_up_7d']:.0f}% up).")
    w(f"- clean_holdout ({ph['n']} events): further low {_pct(ph['mdd']['median'])}, low after {ph['hours_to_low']['median']:.0f} h, rebound {_pct(ph['rebound']['median'])}: the low came even later in 2026-H2.")
    w("So a rebound does usually come, but a stop under the flush low is usually hit first. That is the R-FLOW-2 pattern (don't fade liquidation clusters) on a multi-day scale.\n")
    w(f"**Shorting the failed rebound is roughly breakeven (H-RB2).** After a rebound of at least 1.5 sigma off the post-flush low, short the first 4h close back below the EMA20, stop above the rebound high:")
    w(f"- {r2i['R']['n'] + r2h['R']['n']} setups out of {res['n_events']} events (no setup within 7 days: {res['rb2_status']['no_setup']}). Most entries are in R6 (whipsaw): {r2i['entry_cells']}.")
    w(f"- llm_in_sample {_ci(r2i['R'])} R; clean_holdout {_ci(r2h['R'])} R.")
    w(f"- versus random shorts in the same asset and cell: {_ci(r2i['diff'])} in-sample, {_ci(r2h['diff'])} holdout. Random shorts lose ({r2i['diff']['baseline_mean']:+.2f} R), so 'better than random' mostly means 'less bad'.")
    w(f"- pegged entries: {_ci(r2i['R_peg'])} / {_ci(r2h['R_peg'])}. BTC alone: {_ci(b2['R'])} ({b2['R']['n']} trades).\n")
    w(f"**New shorts piling in after a flush is not a squeeze signal (H-RB3).** The pattern (funding below its 90-day 10th percentile while OI rebuilds >= 3% within 72h) "
      f"is common: {r3i['n_pattern']} of {r3i['n_events']} in-sample events, usually within the first 20 hours.")
    w(f"- 24h vol-scaled return after the pattern minus no-pattern events at the same lag: llm_in_sample {_ci(r3i['z24h'])} sigma, clean_holdout {_ci(r3h['z24h'])} ({r3h['z24h'].get('n_pattern', 0)} resolved pattern events).")
    w(f"- 72h: {_ci(r3i['z72h'])} / {_ci(r3h['z72h'])}.")
    w(f"- BTC alone (in-sample, {b3['n_pattern']} pattern events): 24h {_ci(b3['z24h'])}, 72h {_ci(b3['z72h'])}: positive point estimates, CIs wide.")
    w("- This agrees with R-AVOID-1 and R-MAP-5: funding positioning is not a usable contrarian signal, including after a flush.\n")

    w("## Regime after a flush (check of R-PX-8)\n")
    w("Share of events (%) by v0 cell (4h labels) at +1, +3 and +7 days after the event bar, and the first cell after the R9 episode ends. "
      "'Unconditional R9' is every R9 4h close, for comparison.\n")
    w("| scope | horizon | n | " + " | ".join(f"{c} {CELL_LABELS[c]}" for c in CELLS) + " |")
    w("|---|---|---|" + "---|" * 9)
    for scope, m in (("pooled", tp), ("BTC", tb)):
        for lab in ("+1d", "+3d", "+7d", "next_after_R9"):
            w(f"| {scope} | {lab} | {m[lab]['n']} | {_cell_row(m[lab])} |")
        for lab in ("+1d", "+3d", "+7d"):
            u = m["unconditional_R9"][lab]
            w(f"| {scope}, unconditional R9 | {lab} | {u['n']} | {_cell_row(u)} |")
    for lab in ("+1d", "+3d", "+7d", "next_cell"):
        w(f"| clean_holdout | {lab.replace('next_cell', 'next_after_R9')} | {th[lab]['n']} | {_cell_row(th[lab])} |")
    w("")
    w(f"- When the flush's R9 episode ends, the next cell is **R6 whipsaw {_share(tp['next_after_R9'], 'R6')}** of the time and R8 trend down {_share(tp['next_after_R9'], 'R8')} "
      f"(BTC: R6 {_share(tb['next_after_R9'], 'R6')}; clean_holdout R6 {_share(th['next_cell'], 'R6')} of {th['next_cell']['n']}). "
      "That confirms R-PX-8's direction (whipsaw, not a clean trend) with a higher share than its 62% for R9 in general. A switch straight to an uptrend (R1-R3) is rare: under 2%.")
    w(f"- R9 itself lasts a median **{tp['r9_hours_left_median']:.0f} h** more after the flush event (IQR {tp['r9_hours_left_q'][0]:.0f}-{tp['r9_hours_left_q'][1]:.0f} h). "
      f"A week later the asset is in R6 {_share(tp['+7d'], 'R6')}, still R9 {_share(tp['+7d'], 'R9')}, R5 {_share(tp['+7d'], 'R5')}, R8 {_share(tp['+7d'], 'R8')}.")
    w("- Flush-led R9 is a bit more persistent than R9 in general (62% vs 46% still R9 a day later).\n")

    w("## Events per asset and year\n")
    yrs = sorted(res["counts_year"])
    w("| asset | " + " | ".join(yrs) + " | total |")
    w("|---|" + "---|" * (len(yrs) + 1))
    for r in res["counts_asset_year"]:
        w(f"| {r['symbol']} | " + " | ".join(str(r.get(y, 0)) for y in yrs) + f" | {sum(r.get(y, 0) for y in yrs)} |")
    w("| all | " + " | ".join(str(res["counts_year"][y]) for y in yrs) + f" | {res['n_events']} |")
    w(f"\nTrade status: H-RB1 {res['rb1_status']}; H-RB2 {res['rb2_status']} (unresolved = the 5-day path runs past the data, i.e. the current episode).\n")

    w("## The current episode (2026-11-13 to 15), descriptive only\n")
    b, e, s = cur["BTCUSDT"], cur["ETHUSDT"], cur["SOLUSDT"]
    w("**Did it meet the flush definition?** Times are the 4h bar's open (the bar closes 4h later).")
    for sym, a in (("BTC", b), ("ETH", e), ("SOL", s)):
        ev = a["flush_events"]
        allc = ", ".join(t[5:16] for t in a["bars_meeting_all_conditions"])
        if ev:
            x = ev[0]
            w(f"- **{sym}: yes, at the {x['ts'][5:16]} UTC bar** (48h return {_pct(x['r48'])} vs a 5th-pct threshold of {_pct(x['r48_thr'])}; OI {_pct(x['oi_dd'])} from its 72h high; R9). "
              f"All three conditions also held at {allc}. Previous event: {a['previous_event'][:10] if a['previous_event'] else 'none'}.")
        else:
            w(f"- **{sym}: the conditions held (bars {allc}), but no new event was counted** because the 7-day decluster still covered ETH's previous event on {a['previous_event'][5:16]} "
              "(the 11-10 sell-off). By the definition this is the same episode, continued.")
    w("")
    w("**What players did (BTC).**")
    bars = b["bars"]
    bb = {r["event_ts"][5:16]: r for r in bars}
    pk = max((r for r in bars if "2026-11-12" <= r["event_ts"] <= "2026-11-13 16"), key=lambda r: r["open_interest"])
    lo_oi = min((r for r in bars if r["event_ts"] >= pk["event_ts"]), key=lambda r: r["open_interest"])
    hi_oi_after = max((r for r in bars if r["event_ts"] > lo_oi["event_ts"]), key=lambda r: r["open_interest"])
    w(f"- **Longs were forced out in waves.** Bybit BTC long liquidations since 11-13: {b['liq_total'].get('long', 0) / 1e6:.0f} M USD vs {b['liq_total'].get('short', 0) / 1e6:.0f} M of shorts. "
      "The biggest 4h buckets: " + ", ".join(f"{r['h4'][5:16]} {r['usd'] / 1e6:.0f} M" for r in sorted([r for r in b["liq_4h"] if r["side"] == "long"], key=lambda r: -r["usd"])[:5]) + ".")
    w(f"- **Open interest fell, then rebuilt while price kept falling.** OI peaked at {pk['open_interest'] / 1e3:.1f}k BTC ({pk['event_ts'][5:16]} bar), fell to {lo_oi['open_interest'] / 1e3:.1f}k at {lo_oi['event_ts'][5:16]} ({_pct(lo_oi['open_interest'] / pk['open_interest'] - 1)}), "
      f"then {hi_oi_after['open_interest'] / 1e3:.1f}k at {hi_oi_after['event_ts'][5:16]} ({_pct(hi_oi_after['open_interest'] / lo_oi['open_interest'] - 1)}) while the close went from "
      f"{lo_oi['close']:,.0f} to {hi_oi_after['close']:,.0f}. New positions were opened into the fall; with funding turning negative, mostly shorts.")
    w(f"- **Funding** was positive through the flush (rank {bb.get('11-13 16:00', {}).get('f_rank', float('nan')):.2f} in its 90-day range at the event) and turned negative in the latest print "
      f"({1e4 * bars[-1]['funding_rate']:+.3f} bp, rank {bars[-1]['f_rank']:.2f}, below the 10th-pct threshold {1e4 * bars[-1]['f_thr']:+.3f} bp).")
    w(f"- **Short liquidations** were small: the largest 4h buckets were " + ", ".join(f"{r['h4'][5:16]} {r['usd'] / 1e6:.1f} M" for r in sorted([r for r in b["liq_4h"] if r["side"] == "short"], key=lambda r: -r["usd"])[:3]) +
      ", the squeeze hours of the bounce.")
    for lr in b.get("live_rb1", []):
        w(f"- **The H-RB3 pattern fired for BTC** (lag {lr['rb3']['lag']} bars after the event, i.e. the {datetime.fromisoformat(lr['event_ts']).replace(tzinfo=UTC).strftime('%m-%d %H:%M')} event + {4 * lr['rb3']['lag']} h). Its outcome is not known yet; the test says not to read it as a squeeze signal.")
        w(f"- **The H-RB1 long would already be stopped:** entry {lr['entry']:,.0f} at the close of the bar opened {lr['entry_bar'][5:16]}, stop {lr['stop']:,.0f} ({lr['stop_sigma']:.1f} sigma), "
          f"low since {lr['low_since']:,.0f} ({_pct(math.log(lr['low_since'] / lr['entry']))} from entry), last 1h close {lr['last']:,.0f}.")
    w(f"- ETH and SOL show the same picture: long liquidations {e['liq_total'].get('long', 0) / 1e6:.0f} M / {s['liq_total'].get('long', 0) / 1e6:.0f} M vs shorts {e['liq_total'].get('short', 0) / 1e6:.0f} M / {s['liq_total'].get('short', 0) / 1e6:.1f} M; "
      f"latest funding ranks {e['bars'][-1]['f_rank']:.2f} / {s['bars'][-1]['f_rank']:.2f}; all three are still in R9.")
    for lr in s.get("live_rb1", []):
        w(f"- SOL's H-RB1 long (entry {lr['entry']:.2f}) is stopped too: low since {lr['low_since']:.2f} ({_pct(math.log(lr['low_since'] / lr['entry']))}).")
    # where are we on the typical path?
    h1 = b["h1"]
    ev0 = b["flush_events"][0] if b["flush_events"] else None
    if ev0:
        et = datetime.fromisoformat(ev0["ts"]).replace(tzinfo=UTC).timestamp() * 1000 + 4 * 3_600_000
        idx = [i for i, t in enumerate(h1["ts"]) if t >= et]
        jl = min(idx, key=lambda i: h1["l"][i])
        mdd = math.log(h1["l"][jl] / ev0["close"])
        hrs = (h1["ts"][jl] - et) / 3_600_000 + 1
        reb = math.log(h1["c"][-1] / h1["l"][jl])
        w(f"- **Where BTC is on the typical path:** from the event close ({ev0['close']:,.0f}) the low so far is {h1['l'][jl]:,.0f} ({_pct(mdd)}, {hrs:.0f} h later) and the last close is "
          f"{_pct(reb)} above that low. Historically BTC's further drop had a median of {_pct(pb['mdd']['median'])} (q25 {_pct(pb['mdd']['q25'])}), so this one is already a deep case; the median rebound off the 7-day low was {_pct(pb['rebound']['median'])}. "
          f"Whether {h1['l'][jl]:,.0f} is the 7-day low is unknown; in {100 * float(np.mean(np.array(res['paths']['BTC']['raw']['hours_to_low']) > 120)):.0f}% of past BTC events (pooled {100 * float(np.mean(np.array(res['paths']['pooled']['raw']['hours_to_low']) > 120)):.0f}%) the low came more than 5 days after the event.")
    w("")
    w("**Nearest historical analogues** (chosen after the fact, so this is a description, not evidence). Features at the event bar, z-scored over the "
      "historical events before 2026-11-08: vol-scaled 48h return, OI drop from its 72h high, funding rank in its 90 days, vol-scaled 30-day return. Euclidean distance, 5 nearest across all assets.\n")
    for sym, a in (("BTC", b), ("ETH", e), ("SOL", s)):
        f = a["nn_features"]
        w(f"*{sym}* (reference: {a['nn_reference']['ts'][5:16]}, {a['nn_reference']['kind']}; z48 {f['z48']:.2f}, OI {_pct(f['oi_dd'])}, funding rank {f['f_rank']:.2f}, z30d {f['z30d']:.2f})\n")
        w("| analogue | dist | further low | hours to low | rebound off low | 1d | 3d | 7d | H-RB1 R | H-RB2 R | cell +7d | next after R9 |")
        w("|---|---|---|---|---|---|---|---|---|---|---|---|")
        for n in a["neighbours"]:
            p = n["path"]
            w(f"| {n['symbol']} {n['ts'][:13]} | {n['dist']:.2f} | {_pct(p['mdd'])} | {p['hours_to_low']} | {_pct(p['rebound'])} | {_pct(p['ret1d'])} | {_pct(p['ret3d'])} | {_pct(p['ret7d'])} | "
              f"{'n/a' if n['rb1_R'] is None else f'{n['rb1_R']:+.2f}'} | {'n/a' if n['rb2_R'] is None else f'{n['rb2_R']:+.2f}'} | {n['cell+7d']} | {n['next_cell']} |")
        w("")
    nb = b["neighbours"]
    w(f"- BTC's closest analogues split two ways: two V-shaped recoveries (BTC 2025-11-22, ETC 2025-04-07: the low came within hours and the long made +1.4 / +1.7 R) "
      f"and two deep continuations (BCH 2022-06-13 and BTC 2021-05-13: a further 25-42% down over the week in simple terms, the long stopped); AAVE 2026-06-05 ended the week up but stopped the long first. Median 7-day return of the five: "
      f"{_pct(float(np.median([n['path']['ret7d'] for n in nb])))}. That spread is the point: on these features the flush does not tell you which branch you are on.")
    w(f"- After R9, all five of BTC's analogues went to {', '.join(sorted({n['next_cell'] for n in nb}))}; a week later they were in {', '.join(n['cell+7d'] for n in nb)}, in line with the transition table.\n")

    w("## What this means for long and short plays now\n")
    w("- **Long:** the tested rebound entry (buy the next close after a flush, stop under the flush low) loses in both segments. It was stopped in this episode too. "
      "The rebound usually comes after the low, which is a median 1.5-2.5 days after the flush (later in the 2026 holdout) and 5-9% lower, and these tests do not show how to time it. A later entry is untested (it would need a new registered hypothesis).")
    w("- **Short:** shorting the first failed rebound (back below the 4h EMA20) has been roughly breakeven. That is better than random shorts, which lose in the whipsaw that follows, but it is not an edge.")
    w("- **Regime:** expect R9 to persist for about another day or two, then R6 whipsaw (about 80%), not a clean uptrend. R-PX-8 holds and is sharper for flush-led episodes.")
    w(f"- **Players:** longs have been forced out (about {b['liq_total'].get('long', 0) / max(b['liq_total'].get('short', 1), 1):.0f}:1 long vs short liquidations on BTC), and new shorts have opened into the fall (OI rebuilt, funding below its 10th percentile). Historically that pattern has not predicted a squeeze. R-OPX-5 (mild continuation after tail moves) and R-FLOW-2 point the same way.\n")

    w("## Method details\n")
    w(f"- Entries and exits: H-RB1 long at the close of the 4h bar after the event; stops at the structural extreme with a 0.5 sigma buffer (4h sigma_risk_next), pulled in to 4 sigma; "
      f"2R target; 120 h time stop; 1h bars walked from the entry close, stop checked before the target in each hour, stops fill at the worse of the stop and the hour's open. "
      f"Costs: 7.5 bp per side; variant: 3.3 bp pegged-limit entry (R-EXE-1, risk.yaml paper_entry).")
    w(f"- Baselines: one random 4h close per asset per UTC date on non-event days (no flush within +-7 days), same geometry (stop beyond the 13-bar extreme). "
      f"Matched on asset + R9 (RB1) or asset + entry cell (RB2) + segment; {r1i['diff']['baseline_n'] // 2} in-sample long baselines.")
    w("- Inference: cluster bootstrap by ISO week (all assets in a week share a cluster: crypto moves together), 5000 draws, events and baselines resampled independently.")
    w("- Verdict rule (frozen): supports = mean > 0 with the CI above 0 and above the matched baseline in that segment; contradicts = mean or paired difference <= 0 with its CI below 0; else inconclusive.")
    w("- Survivorship: the universe is today's survivors. Limits: the 4h close-to-close flush timing is coarse; 1h paths cannot resolve stop and target inside one hour (worst case assumed).\n")

    w("## Figures\n")
    for n, cap in figs:
        w(f"![{cap}](../figures/post_cascade_2026-11-15/{n})\n")
        w(f"_{cap}._\n")

    w("## Rule candidates\n")
    w("Validated with `ai_trader.research.rules.validate` (no problems, alone and together with research/rules.yaml). Not yet merged into rules.yaml.\n")
    w("```yaml")
    w(yaml.safe_dump(rules, sort_keys=False, allow_unicode=True, width=10_000).rstrip())
    w("```\n")
    return "\n".join(L)


if __name__ == "__main__":
    build()
