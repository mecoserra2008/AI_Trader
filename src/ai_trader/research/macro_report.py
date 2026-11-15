"""Report for the MAC block: research/reports/macro_narrative_<date>.{md,json} + figures.

    PYTHONIOENCODING=utf-8 POLARS_MAX_THREADS=2 uv run python -m ai_trader.research.macro_report

Reads data/research/macro/macro_study.json (written by ai_trader.research.macro_study) and the thesis check of the
same as_of; descriptive sections are labelled as such. Ends with R-MAC-n rule candidates validated by
ai_trader.research.rules.validate.
"""

from __future__ import annotations

import json
import re
from datetime import UTC, datetime

import numpy as np
import yaml

from ai_trader.core.config import load

ROOT = load().root
DAY = "2026-11-15"
FIG = ROOT / "research" / "figures" / f"macro_narrative_{DAY}"
REP = ROOT / "research" / "reports" / f"macro_narrative_{DAY}"
BLUE, ORANGE, AQUA, YELLOW, GREY, INK, INK2, SURF = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#a3a29c", "#0b0b0b", "#52514e", "#fcfcfb"
REG_COL = {"bear_flattening": ORANGE, "bear_steepening": YELLOW, "bull_flattening": AQUA, "bull_steepening": BLUE, "no_regime": GREY}
REG_ORDER = ["bear_flattening", "bear_steepening", "bull_flattening", "bull_steepening", "no_regime"]


def _style(plt) -> None:
    plt.rcParams.update({"figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF, "axes.edgecolor": GREY,
                         "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "text.color": INK, "font.size": 8.5,
                         "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": "#e6e5e0",
                         "grid.linewidth": 0.5, "legend.frameon": False, "axes.titlesize": 9, "axes.titleweight": "bold"})


def _d(x):
    return np.array([np.datetime64(s[:10]) for s in x])


def figures(R: dict) -> list[tuple[str, str]]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _style(plt)
    FIG.mkdir(parents=True, exist_ok=True)
    out = []
    H = R["history"]
    d = _d([h["d"] for h in H])
    btc = np.array([h["btc"] for h in H], float)
    reg = [h["regime"] for h in H]
    # 1. regime history with BTC
    fig, axs = plt.subplots(3, 1, figsize=(11, 6.4), sharex=True, gridspec_kw={"height_ratios": [2.2, 1.3, 1.4]})
    axs[0].plot(d, btc, color=INK, lw=1.2)
    axs[0].set_yscale("log")
    axs[0].set_ylabel("BTC (log scale, USD)")
    axs[0].set_title("BTC and the registered yield-curve regime per FRED day, 2020-01 .. 2026-11 (2017-2019: no BTC in the lake)")
    for k, r in enumerate(REG_ORDER):
        m = np.array([x == r for x in reg])
        axs[1].scatter(d[m], np.full(m.sum(), k), marker="|", s=60, color=REG_COL[r], lw=1.0)
    axs[1].set_yticks(range(len(REG_ORDER)), [r.replace("_", " ") for r in REG_ORDER])
    axs[1].grid(False)
    axs[1].set_ylim(-0.6, len(REG_ORDER) - 0.4)
    y2 = np.array([h["y2"] for h in H], float)
    y10 = np.array([h["y10"] for h in H], float)
    axs[2].plot(d, y2, color=ORANGE, lw=1.4, label="2y (DGS2)")
    axs[2].plot(d, y10, color=BLUE, lw=1.4, label="10y (DGS10)")
    axs[2].set_ylabel("yield, %")
    axs[2].legend(loc="upper left", ncol=2)
    axs[2].axvline(np.datetime64(R["cutoff"]), color=GREY, lw=0.8, ls="--")
    axs[0].axvline(np.datetime64(R["cutoff"]), color=GREY, lw=0.8, ls="--")
    axs[0].text(np.datetime64(R["cutoff"]), axs[0].get_ylim()[1], " cutoff", color=INK2, va="top", fontsize=7.5)
    fig.tight_layout()
    p = FIG / "regime_history.png"
    fig.savefig(p, dpi=140)
    plt.close(fig)
    out.append((p.name, "Top: BTC (Deribit to 2020-03-25, then Bybit), log scale. Middle: the registered curve regime on each FRED day (one row per regime). Bottom: 2y and 10y yields. Dashed: knowledge cutoff."))

    # 2. H-MAC1 cells
    fig, ax = plt.subplots(figsize=(8.5, 3.4))
    labels, xs = [], []
    k = 0
    for c_in, c_ho in zip(R["mac1"]["llm_in_sample"]["cells"], R["mac1"]["clean_holdout"]["cells"], strict=True):
        for off, c, col, nm in ((-0.12, c_in, BLUE, "llm_in_sample"), (0.12, c_ho, ORANGE, "clean_holdout")):
            ci = c.get("ci95", [np.nan, np.nan])
            ax.errorbar(k + off, c["mean"], yerr=[[c["mean"] - ci[0]], [ci[1] - c["mean"]]], fmt="o", color=col, ms=6, lw=1.6, capsize=0,
                        label=nm if k == 0 else None)
            ax.annotate(f"n={c['n_days']}", (k + off, ci[1]), xytext=(0, 3), textcoords="offset points", ha="center", fontsize=7, color=INK2)
        labels.append(f"{c_in['regime'].replace('_', ' ')}\n+{c_in['h']}d (hyp. {'<' if c_in['hyp_sign'] < 0 else '>'} 0)")
        xs.append(k)
        k += 1
    ax.axhline(0, color=GREY, lw=0.8)
    ax.set_xticks(xs, labels)
    ax.set_ylabel("mean vol-scaled fwd return (z)")
    ax.set_title("H-MAC1: BTC forward return after bear-flattening / bull-steepening days, 95% week-clustered CI")
    ax.legend(loc="upper left")
    fig.tight_layout()
    p = FIG / "mac1_cells.png"
    fig.savefig(p, dpi=140)
    plt.close(fig)
    out.append((p.name, "H-MAC1 cells: mean of r / (sigma30 sqrt h) with 95% week-clustered bootstrap intervals, by segment. The hypothesis needs bear flattening below 0 and bull steepening above 0 in all four cells."))

    # 3. today: last 60 FRED days of 5-obs changes
    last = H[-60:]
    dl = _d([h["d"] for h in last])
    y2l = np.array([h["y2"] for h in last])
    y10l = np.array([h["y10"] for h in last])
    t20 = R["today"]["last20"]
    fig, axs = plt.subplots(2, 1, figsize=(10, 5.0), sharex=False, gridspec_kw={"height_ratios": [1.3, 1]})
    axs[0].plot(dl, y2l, color=ORANGE, lw=1.6, marker="o", ms=3, label="2y")
    axs[0].plot(dl, y10l, color=BLUE, lw=1.6, marker="o", ms=3, label="10y")
    axs[0].set_ylabel("yield, %")
    axs[0].legend(loc="upper left", ncol=2)
    axs[0].set_title(f"The last 60 FRED days to {R['today']['last_obs']}: yields (top), 5-observation changes and regime (bottom)")
    dd = _d([x["obs_date"] for x in t20])
    w = 0.38
    xi = np.arange(len(t20))
    axs[1].bar(xi - w / 2, [x["d2_bp"] for x in t20], width=w, color=ORANGE, label="2y 5-obs change, bp")
    axs[1].bar(xi + w / 2, [x["d10_bp"] for x in t20], width=w, color=BLUE, label="10y 5-obs change, bp")
    for i, x in enumerate(t20):
        lab = x["regime"].replace("_", " ").replace("flattening", "flat").replace("steepening", "steep").replace("no regime", "none")
        axs[1].text(i, -22, lab, ha="center", va="top", fontsize=6.5, color=INK2, rotation=90)
    axs[1].set_xticks(xi, [str(v)[5:] for v in dd], fontsize=7)
    axs[1].axhline(0, color=GREY, lw=0.8)
    axs[1].axhspan(-5, 5, color=GREY, alpha=0.15, lw=0)
    axs[1].set_ylim(-48, 34)
    axs[1].legend(loc="upper left", ncol=2)
    fig.tight_layout()
    p = FIG / "today_curve.png"
    fig.savefig(p, dpi=140)
    plt.close(fig)
    out.append((p.name, "Today's regime and its 20-day history. Grey band: the +-5 bp no-regime band. Bear flattening appears only on the last two FRED days (Nov 10 and 12, after the hot October CPI); the 20 days before were mostly bull flattening."))

    # 4. S&P drawdown vs BTC drawdown
    sp = R["_sp_series"]
    fig, ax = plt.subplots(figsize=(10, 3.6))
    ax.plot(_d(sp["d"]), sp["spx_dd"], color=BLUE, lw=1.4, label="S&P 500 below its high, %")
    ax.plot(_d(sp["d"]), sp["btc_dd"], color=ORANGE, lw=1.2, label="BTC below its high, %")
    lvl = R["sp_drawdowns_10"]["current"]["user_level_7000_dd_from_ath_pct"]
    ax.axhline(lvl, color=INK2, lw=0.8, ls="--")
    ax.text(_d(sp["d"])[len(sp["d"]) // 2], lvl - 1, f"user's 7000 = {lvl:.1f}% below today's S&P high", va="top", fontsize=7.5, color=INK2)
    ax.set_ylabel("% below running high")
    ax.set_title("S&P 500 and BTC drawdowns from their running highs, 2020-2026")
    ax.legend(loc="lower left")
    fig.tight_layout()
    p = FIG / "drawdowns.png"
    fig.savefig(p, dpi=140)
    plt.close(fig)
    out.append((p.name, "Drawdowns from the running high (simple %), S&P 500 (FRED) and BTC. Dashed: the user's 7000 level expressed as a drawdown from the current S&P high."))

    # 5. semis / AI divergence
    dv = R["divergence"]
    fig, axs = plt.subplots(1, 2, figsize=(11, 3.4), gridspec_kw={"width_ratios": [1.6, 1]})
    s = [x for x in dv["series"] if x["rel20"] is not None]
    axs[0].plot(_d([x["d"] for x in s]), [x["rel20"] for x in s], color=BLUE, lw=1.0)
    axs[0].axhline(0, color=GREY, lw=0.8)
    axs[0].set_ylabel("NDX minus SPX, 20-day log return, %")
    axs[0].set_title("Tech leadership: NASDAQ-100 vs S&P 500, 20-day relative return")
    for sym, col in (("SOXLUSDT", ORANGE), ("NVDAUSDT", BLUE), ("MUUSDT", AQUA)):
        ss = dv["semis"].get(sym, {}).get("series", [])
        if ss:
            nm = "SOX proxy (SOXL / 3)" if sym == "SOXLUSDT" else sym.replace("USDT", "")
            axs[1].plot(_d([x["d"] for x in ss]), [x["rel20"] for x in ss], color=col, lw=1.4, label=nm)
    axs[1].axhline(0, color=GREY, lw=0.8)
    axs[1].set_title("Semis vs S&P 500, 20d relative (Bybit perps)")
    axs[1].legend(loc="lower left", fontsize=7)
    axs[1].tick_params(axis="x", labelrotation=30)
    fig.tight_layout()
    p = FIG / "divergence.png"
    fig.savefig(p, dpi=140)
    plt.close(fig)
    out.append((p.name, "Left: NDX minus SPX 20-day log return since 2020 (FRED). Right: semis proxies vs SPX since May 2026 (Bybit TradFi perps; SOXL divided by 3 is a rough de-levered SOX)."))

    # 6. thesis (d) proxies: today's 1y percentile
    P = R["proxies_today"]
    fig, ax = plt.subplots(figsize=(8, 3.6))
    names = [x["series"] for x in P]
    ax.barh(range(len(P)), [100 * x["pct_1y"] for x in P], color=BLUE, height=0.6)
    for i, x in enumerate(P):
        ax.text(100 * x["pct_1y"] + 1, i, f"{x['value']:.2f} ({x['obs_date']})", va="center", fontsize=7, color=INK2)
    ax.set_yticks(range(len(P)), names)
    ax.invert_yaxis()
    ax.set_xlim(0, 125)
    ax.set_xlabel("1-year percentile of today's value (monthly: last 12 obs)")
    ax.set_title("Thesis (d) proxies today: level and 1-year percentile")
    fig.tight_layout()
    p = FIG / "proxies_today.png"
    fig.savefig(p, dpi=140)
    plt.close(fig)
    out.append((p.name, "Today's values of the thesis (d) proxies and their 1-year percentiles (daily: last 252 obs; monthly: last 12)."))
    return out


def sp_series(as_of: datetime) -> dict:
    from ai_trader.narrative import macro
    from ai_trader.research.macro_study import btc_daily

    sp = macro.read_asof("SP500", as_of)
    btc, _ = btc_daily(as_of)
    v = sp["value"].to_numpy()
    dd = 100 * (v / np.maximum.accumulate(v) - 1)
    ds = [str(x) for x in sp["obs_date"].to_list()]
    bc = dict(zip([str(x) for x in btc["d"].to_list()], btc["close"].to_list(), strict=True))
    bv = np.array([bc.get(x, np.nan) for x in ds], float)
    keep = np.array([x >= "2020-01-01" for x in ds]) & np.isfinite(bv)
    bvk = bv[keep]
    bdd = 100 * (bvk / np.maximum.accumulate(bvk) - 1)
    return {"d": list(np.array(ds)[keep]), "spx_dd": list(dd[keep]), "btc_dd": list(bdd)}


def _ord(x: float) -> str:
    n = int(round(x))
    suf = "th" if 10 <= n % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suf}"


def _f(x, nd=2, sign=False):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "n/a"
    return f"{x:+.{nd}f}" if sign else f"{x:.{nd}f}"


RULES_YAML = """rules:
  - id: R-MAC-1
    kind: avoid
    rule: "WHEN the yield-curve regime is bear flattening or bull steepening THEN do not read it as a BTC direction signal over 5 or 20 days"
    when: "curve regime (narrative.curve, 5-obs DGS2 / DGS10 changes) = bear_flattening or bull_steepening"
    then: "treat the regime as macro context only; any directional idea needs its own evidence"
    scope: {assets: crypto, horizon: "5d-20d", sides: both}
    effect: "{effect1}"
    confidence: C
    evidence: [H-MAC1, research/reports/macro_narrative_2026-11-15.md]
    review: "re-run H-MAC1 when the clean holdout has >= 40 bear-flattening and >= 40 bull-steepening days, or with pre-2020 BTC if the user adds it"
    status: provisional
  - id: R-MAC-2
    kind: expect
    rule: "WHEN a bull steepening comes with the HY spread easing THEN expect a better 20-day BTC outcome than when it comes with HY widening, as an unconfirmed tendency only"
    when: "curve regime = bull_steepening and the 5-obs change in BAMLH0A0HYM2 < 0 (vs > 0)"
    then: "weight the user's thesis (c) trigger (bull steepening with credit easing) as the better branch; do not trade the split by itself"
    scope: {assets: crypto, horizon: "20d", sides: both}
    effect: "{effect2}"
    confidence: C
    evidence: [H-MAC2, research/reports/macro_narrative_2026-11-15.md]
    review: "HY OAS on FRED starts 2023-11-14; re-test with a longer credit series or after more holdout bull-steepening days"
    status: provisional
  - id: R-MAC-3
    kind: avoid
    rule: "WHEN a ledger thesis is active or strengthening THEN cite it as narrative context and never as the reason a trade beats the driftless base rate"
    when: "research/desk/<snapshot>/theses.json status in (active, strengthening) for an asset's thesis"
    then: "put the thesis in the narrative layer of the confluence log and in honesty.conflicts when it cuts against an idea; direction claims still need R-BUDGET-1's justification"
    scope: {assets: crypto, horizon: "1h-30d", sides: both}
    effect: "no tested edge: H-MAC1 and H-MAC2 inconclusive; the ledger is scored forward only"
    confidence: C
    evidence: [H-MAC1, H-MAC2, research/desk/theses.yaml]
    review: "after 20 scored theses, compare their signed horizon returns with zero and with the paired desk baselines"
    status: provisional
"""


def main() -> None:
    R = json.loads((ROOT / "data" / "research" / "macro" / "macro_study.json").read_text(encoding="utf-8"))
    as_of = datetime.fromisoformat(R["as_of"]).astimezone(UTC)
    R["_sp_series"] = sp_series(as_of)
    figs = figures(R)
    from ai_trader.narrative import theses

    th = theses.public(theses.check(max(as_of, datetime(2026, 11, 15, 19, 30, tzinfo=UTC))))
    m1, m2, td = R["mac1"], R["mac2"], R["today"]
    ci = {(c["regime"], c["h"], s): c for s in ("llm_in_sample", "clean_holdout", "full") for c in m1[s]["cells"]}
    bf5, bs5 = ci[("bear_flattening", 5, "llm_in_sample")], ci[("bull_steepening", 5, "llm_in_sample")]
    eff1 = (f"in-sample bear flattening +5d z {bf5['mean']:+.3f} (Holm p {bf5['p_holm']:.2f}), +20d {ci[('bear_flattening', 20, 'llm_in_sample')]['mean']:+.3f}; "
            f"bull steepening +5d {bs5['mean']:+.3f} (Holm p {bs5['p_holm']:.2f}), +20d {ci[('bull_steepening', 20, 'llm_in_sample')]['mean']:+.3f}; "
            "holdout signs mixed; H-MAC1 inconclusive in both segments")
    f2 = m2["full"]
    eff2 = (f"eased minus widened 20d z {f2['diff']:+.2f}, 95% CI [{f2['diff_ci95'][0]:+.2f}, {f2['diff_ci95'][1]:+.2f}] (n {f2['n_eased']} / {f2['n_widened']} days, "
            f"2023-11..2026-11); same sign in both segments; H-MAC2 inconclusive")
    rules_yaml = RULES_YAML.replace("{effect1}", eff1).replace("{effect2}", eff2)
    from ai_trader.research.rules import validate

    rv = validate(yaml.safe_load(rules_yaml)["rules"])
    if rv:
        raise SystemExit("rule candidates invalid: " + "; ".join(rv))

    L: list[str] = []
    a = L.append
    a(f"# Macro narrative layer: curve regimes, the user's thesis and the confluence log ({DAY})")
    a("")
    a("Agent MACRO. Research only. The MAC block (curve-regime definition, H-MAC1, H-MAC2) was registered in `research/hypotheses.yaml` before any data; the "
      "operational reading of it (dominant-leg rule, entry, vol scaling, verdict rules) was frozen in the diary at 18:53 UTC before any result. "
      "Code: `src/ai_trader/narrative/{macro,curve,theses,confluence}.py`, `src/ai_trader/research/macro_study.py`, `macro_report.py`; "
      "scripts `scripts/thesis_check.py`, `scripts/confluence_fit.py`; tests `tests/test_narrative.py`. Log returns unless marked.")
    a("")
    a("> **Read this first.** Neither registered test found the effect. H-MAC1 (BTC weak after bear flattening, strong after bull steepening) is "
      "**inconclusive** in both segments: the signs are mixed and no cell is significant after Holm. H-MAC2 (bull steepening is better when credit eases) "
      "points the hypothesised way in both segments but its interval includes zero: **inconclusive**. The user's curve thesis therefore enters the ledger "
      "as a narrative to be scored forward, not as a tested edge.")
    a("")
    a("**Data and gaps.** FRED public CSV, no key (a custom User-Agent stalls FRED; the default works). The series are stored point in time under "
      "`data/research/macro/` (raw downloads, parquet with `rule_ts` / `effective_ts`, `manifest.json`); daily `known_ts` = obs_date + 1 day 00:00 UTC as registered. "
      "BTC is Deribit BTC-PERPETUAL 1h until 2020-03-25, then Bybit BTCUSDT, as UTC-day closes 2020-01-01 .. "
      f"{R['btc']['last_day']}. Gaps:")
    for g in R["data_gaps"]:
        a(f"- {g};")
    a("- Bybit SPYUSDT is never used.")
    a("")
    a("## 1. Today's curve regime: is it bear flattening now?")
    a("")
    a(f"**Yes, but only on the last two FRED days.** The last observation known at {as_of:%Y-%m-%d %H:%M} UTC is **{td['last_obs']}** "
      f"(Nov 13 is not on FRED yet; Nov 11 was Veterans Day): 2y **{td['y2']:.2f}%**, 10y **{td['y10']:.2f}%**, 10y-2y **{td['spread_bp']:.0f} bp**. "
      f"Over 5 observations the 2y rose **{td['d2_bp']:+.0f} bp** and the 10y **{td['d10_bp']:+.0f} bp**: the 2y leads upward, so the regime is "
      f"**{td['regime'].replace('_', ' ')}**. Over 20 observations the picture is different: 2y {td['d2_20obs_bp']:+.0f} bp, 10y {td['d10_20obs_bp']:+.0f} bp. "
      "The 20 FRED days before were mostly bull flattening (long end falling), and the bear flattening began with the hot October CPI (Nov 10).")
    a("")
    a("| regime in the last 20 FRED days | days |")
    a("|---|---|")
    for c in td["regime_20d_counts"]:
        a(f"| {c['regime'].replace('_', ' ')} | {c['n']} |")
    a("")
    a("| FRED day | 2y | 10y | 2y 5-obs, bp | 10y 5-obs, bp | regime |")
    a("|---|---|---|---|---|---|")
    for x in td["last20"][-10:]:
        a(f"| {x['obs_date']} | {x['y2']:.2f} | {x['y10']:.2f} | {x['d2_bp']:+.0f} | {x['d10_bp']:+.0f} | {x['regime'].replace('_', ' ')} |")
    a("")
    a(f"![today](../figures/macro_narrative_{DAY}/today_curve.png)")
    a("")
    a("_" + figs[2][1] + "_")
    a("")
    a("So the user's read of the front end is right: the 2y is moving a lot (+40 bp from Nov 5 to Nov 12). The bear flattening is about two days old, "
      "and a 5-observation regime this young flips often (see the 20-day history).")
    a("")
    a("## 2. H-MAC1: BTC forward returns by curve regime")
    a("")
    a(f"Regime-days: {', '.join(f'{c['regime']} {c['n']}' for c in m1['full']['regime_counts'])} (full sample {m1['full']['first']} .. {m1['full']['last']}). "
      "Mean of the vol-scaled forward return z = r / (sigma30 sqrt h); one-sided p from a week-clustered bootstrap with the null imposed; Holm over the 4 cells "
      "within a segment. NW t uses lags = h, because 20-day windows overlap across weeks (the week clusters do not fully cover that overlap).")
    a("")
    a("| segment | regime | h | days | weeks | mean z | 95% CI | vs unconditional | one-sided p | Holm p | NW t | mean raw % | hyp. sign? |")
    a("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for s in ("llm_in_sample", "clean_holdout"):
        for c in m1[s]["cells"]:
            a(f"| {s} | {c['regime'].replace('_', ' ')} | {c['h']}d | {c['n_days']} | {c['n_weeks']} | {_f(c['mean'], 3, True)} | "
              f"[{_f(c['ci95'][0], 2, True)}, {_f(c['ci95'][1], 2, True)}] | {_f(c['diff_vs_uncond'], 3, True)} | {_f(c['p_one_sided'], 3)} | {_f(c['p_holm'], 3)} | "
              f"{_f(c['nw_t'], 2, True)} | {_f(c['raw_mean_pct'], 2, True)} | {'yes' if c['right_sign'] else 'no'} |")
    a("")
    v = m1["verdicts"]
    a(f"**Verdict: llm_in_sample {v['llm_in_sample']}, clean_holdout {v['clean_holdout']}, pooled {v['pooled']}** (robustness with known_ts + 1 day: "
      f"{', '.join(f'{k} {x}' for k, x in R['mac1_robust_lag1']['verdicts'].items())}). In-sample, two cells have the hypothesised sign (bear flattening +5d, "
      "bull steepening +5d) and two do not (both +20d); none survives Holm. In the holdout, bear flattening was followed by *higher* BTC (n = 21 days in "
      "9 weeks), and the one large cell (bull steepening +20d, z +1.17, n = 11) rests on 7 weeks. Nothing here supports trading the regime.")
    a("")
    a(f"![cells](../figures/macro_narrative_{DAY}/mac1_cells.png)")
    a("")
    a("_" + figs[1][1] + "_")
    a("")
    a("## 3. H-MAC2: bull steepening with credit easing vs widening")
    a("")
    a(f"HY OAS (BAMLH0A0HYM2) is on FRED only from {m2['hy_first_obs']}, so H-MAC2 covers 2023-11 .. 2026-11 (registered as 'full sample'; that sample is shorter than "
      "intended and is a data gap).")
    a("")
    a("| segment | eased: days (weeks) | widened: days (weeks) | mean z eased | mean z widened | diff | 95% CI | raw mean % eased / widened |")
    a("|---|---|---|---|---|---|---|---|")
    for s in ("full", "llm_in_sample", "clean_holdout"):
        x = m2[s]
        a(f"| {s} | {x['n_eased']} ({x['weeks_eased']}) | {x['n_widened']} ({x['weeks_widened']}) | {_f(x.get('mean_eased'), 3, True)} | {_f(x.get('mean_widened'), 3, True)} | "
          f"{_f(x.get('diff'), 3, True)} | [{_f(x.get('diff_ci95', [None, None])[0], 2, True)}, {_f(x.get('diff_ci95', [None, None])[1], 2, True)}] | "
          f"{_f(x.get('raw_mean_eased_pct'), 2, True)} / {_f(x.get('raw_mean_widened_pct'), 2, True)} |")
    a("")
    a(f"**Verdict: {m2['verdict']}** (full sample; segments: {', '.join(f'{k} {x}' for k, x in m2['segment_verdicts'].items())}; known_ts + 1 day: "
      f"{R['mac2_robust_lag1']['verdict']}). The difference has the hypothesised sign in both segments (eased better by about 0.3 sigma over 20 days), "
      "but the full-sample CI includes zero (bootstrap P(diff <= 0) = "
      f"{_f(f2.get('p_diff_le_0'), 3)}), and the holdout has 9 days in 6 weeks. This is the most interesting lead for the user's component (c): "
      "the steepening's cause (credit easing vs widening) is the right thing to condition on, but it isn't established.")
    a("")
    a("## 4. Descriptive: BTC realised vol by regime")
    a("")
    a("Not a test. Median annualised realised vol of daily BTC returns over the same 5 days, the next 5 days, and the 30-day sigma at entry (full sample).")
    a("")
    a("| regime | days | RV same 5d, % | RV next 5d, % | sigma30 at entry, % ann. |")
    a("|---|---|---|---|---|")
    for x in R["vol_by_regime"]["full"]:
        a(f"| {x['regime'].replace('_', ' ')} | {x['n']} | {_f(x['rv_same5_med_pct'], 1)} | {_f(x['rv_next5_med_pct'], 1)} | {_f(x['sigma30_ann_med_pct'], 1)} |")
    a("")
    a("The differences are a few vol points: bull flattening (long end falling, usually risk-off) carries the highest 30-day sigma, bull steepening the lowest "
      "next-5-day vol. No regime is a vol regime for BTC.")
    a("")
    a(f"![history](../figures/macro_narrative_{DAY}/regime_history.png)")
    a("")
    a("_" + figs[0][1] + "_")
    a("")
    a("## 5. Descriptive: S&P 500 drawdowns, AI leadership and BTC")
    a("")
    cur = R["sp_drawdowns_10"]["current"]
    a(f"S&P 500 (FRED) **{cur['level']:.2f}** on {cur['date']}, **{abs(cur['dd_pct']):.1f}%** below its high of {cur['ath']:.2f}. The user's level, **7000**, is "
      f"{abs(cur['user_level_7000_logdist_pct']):.1f}% (log) below the last close and {abs(cur['user_level_7000_dd_from_ath_pct']):.1f}% below the high, so it would be a "
      "standard correction (it is the user's number, not ours).")
    a("")
    a("Episodes since 2020 with a drawdown of 5% or more from the running high (the 2023 correction sits inside the 2022 episode, because the S&P had not regained "
      "its January 2022 high). BTC columns: log % from the S&P peak to the S&P trough, and the 20 days after the trough.")
    a("")
    a("| S&P peak | trough | S&P depth % | BTC peak to trough, log % | BTC +20d after trough, log % |")
    a("|---|---|---|---|---|")
    for e in R["sp_drawdowns_5"]["episodes"]:
        a(f"| {e['peak']} | {e['trough']} | {e['depth_pct']:.1f} | {_f(e.get('btc_peak_to_trough_pct'), 1, True)} | {_f(e.get('btc_trough_plus20d_pct'), 1, True)} |")
    a("")
    eps = R["sp_drawdowns_5"]["episodes"]
    n_fell = sum(1 for e in eps if (e.get("btc_peak_to_trough_pct") or 0) < 0)
    n_more = sum(1 for e in eps if e.get("btc_peak_to_trough_pct") is not None and e["btc_peak_to_trough_pct"] < 100 * np.log(1 + e["depth_pct"] / 100))
    after = [e["btc_trough_plus20d_pct"] for e in eps if e.get("btc_trough_plus20d_pct") is not None]
    a(f"BTC fell from the S&P peak to the S&P trough in {n_fell} of {len(eps)} episodes, by more than the S&P in {n_more}, and it was higher 20 days after "
      f"the trough in {sum(x > 0 for x in after)} of the {len(after)} with data. That fits the user's 'no reason to buy crypto' while the S&P is still falling. "
      "It is also hindsight: the trough is known only afterwards.")
    a("")
    a(f"![drawdowns](../figures/macro_narrative_{DAY}/drawdowns.png)")
    a("")
    a("_" + figs[3][1] + "_")
    a("")
    nd = R["divergence"]["ndx_spx"]
    sm = R["divergence"]["semis"]
    a(f"**AI concentration.** NDX/SPX ratio {nd['ratio']:.3f}: {_ord(100 * nd['ratio_pct_since_2016'])} percentile since 2016 (concentration is high), but "
      f"{_ord(100 * nd['ratio_pct_1y'])} within the last year (it is coming off). NDX lagged SPX by {abs(nd['rel20_pct']):.1f}% over 20 days (z {nd['rel20_z']:+.2f} vs the past year). "
      f"Semis vs SPX over 20 days to {sm['NVDAUSDT'].get('date_now')}: NVDA {_f(sm['NVDAUSDT'].get('rel20_now_pct'), 1, True)}%, MU {_f(sm['MUUSDT'].get('rel20_now_pct'), 1, True)}%, "
      f"SOX proxy {_f(sm['SOXLUSDT'].get('rel20_now_pct'), 1, True)}%. The AI leaders are already underperforming the index sharply, while the S&P is only "
      f"{abs(cur['dd_pct']):.1f}% off its high: the 'index still very high' the user describes is the broad index holding up while semis fall.")
    a("")
    a("| NDX vs SPX 20d (since 2020) | days | BTC same 20d, mean log % | BTC next 20d, mean log % |")
    a("|---|---|---|---|")
    for k, g in nd["groups"].items():
        a(f"| {k} | {g['n_days']} | {_f(g['btc_same20_mean_pct'], 2, True)} | {_f(g['btc_fwd20_mean_pct'], 2, True)} |")
    a("")
    gl, gn = nd["groups"]["ndx_lags_spx_z<-1.5"], nd["groups"]["normal"]
    a(f"When NDX lags SPX by more than 1.5 sigma, BTC has been about flat over the same 20 days ({gl['btc_same20_mean_pct']:+.1f}% vs {gn['btc_same20_mean_pct']:+.1f}% "
      f"in normal times) and lower over the next 20 ({gl['btc_fwd20_mean_pct']:+.1f}% vs {gn['btc_fwd20_mean_pct']:+.1f}%). Overlapping windows, few episodes: "
      "context, not a test.")
    a("")
    a(f"![divergence](../figures/macro_narrative_{DAY}/divergence.png)")
    a("")
    a("_" + figs[4][1] + "_")
    a("")
    a("## 6. The user's addendum (component d): proxies today")
    a("")
    a("| series | obs | value | 1y percentile | change (20 obs / 3 m) | 1y median |")
    a("|---|---|---|---|---|---|")
    for x in R["proxies_today"]:
        ch = x.get("chg_20obs", x.get("chg_3m"))
        extra = f" (last 4 y/y: {x['last4_yoy']}; 3m ann. {x['ann3m_pct']:.2f}%)" if "last4_yoy" in x else ""
        a(f"| {x['series']} | {x['obs_date']} | {x['value']:.3f}{extra} | {100 * x['pct_1y']:.0f} | {_f(ch, 3, True)} | {_f(x.get('median_1y'), 3)} |")
    a("")
    a("Reading (descriptive): the front end is high (2y at its 98th 1-year percentile), real yields are high (DFII10 96th) and IG / BBB spreads are wide for the year "
      "(92nd) and widened over 20 days, but the HY spread narrowed over 20 days and the Kim-Wright term premium fell 20 bp over 20 observations (it is still at its "
      "81st percentile). So the 'spreads widen, term premium anchors yields' part is half there: IG yes, HY and term premium not this month. The no-intervention "
      "conditions hold: Sahm 0.00 (< 0.5), unemployment 4.1%, core PCE 3.12% y/y (> 3%). Core CPI is 2.57% y/y (Sep; the October y/y can't be computed because FRED "
      "has no October 2025 value), which does not clear 3% on its own, though its 3-month annualised rate is 3.7%.")
    a("")
    a(f"![proxies](../figures/macro_narrative_{DAY}/proxies_today.png)")
    a("")
    a("_" + figs[5][1] + "_")
    a("")
    sv = R["spreads_vs_2y"]
    a(f"**Descriptive check (not registered): spreads widening while the 2y stays high vs while it falls.** Days with HY and IG OAS both wider over 20 obs "
      f"({sv['first']} .. ; {sv['n_widening']} of {sv['n_days']} days):")
    a("")
    a("| case | days | weeks | BTC same 20d, mean log % | BTC next 20d, mean log % | next 20d up share |")
    a("|---|---|---|---|---|---|")
    for k in ("widening_2y_up_or_flat", "widening_2y_down", "widening_2y_down_25bp"):
        g = sv[k]
        a(f"| {k.replace('_', ' ')} | {g['n_days']} | {g['n_weeks']} | {_f(g['btc_same20_mean_pct'], 2, True)} | {_f(g['btc_fwd20_mean_pct'], 2, True)} | {_f(g['btc_fwd20_hit_up'], 2)} |")
    g = sv["tightening_any_2y"]
    a(f"| spreads tightening (any 2y) | {g['n_days']} | | {_f(g['btc_same20_mean_pct'], 2, True)} | {_f(g['btc_fwd20_mean_pct'], 2, True)} | |")
    a("")
    a("BTC did worst *during* spread widening when the 2y was falling (-7% over the same 20 days: a growth scare, risk-off) and only mildly negative when the 2y "
      "stayed high (-1%). The next 20 days were near flat in both cases. Since 2023-11 only (HY licence window), overlapping windows: suggestive at most.")
    a("")
    a("## 7. The thesis ledger (Part 2)")
    a("")
    a("`research/desk/theses.yaml` holds the definitions, `src/ai_trader/narrative/theses.py` loads, validates, evaluates and scores them, and `scripts/thesis_check.py` "
      "runs every desk cycle after `desk_snapshot.py`, writing `<snapshot>/theses.json` and a digest of 30 lines at most. Running state: `data/journal/theses/`. "
      "Status rules: invalidated and expired are sticky; otherwise net support of 0.5 or more = strengthening, below 0 = weakening, else active. Scoring is forward "
      "only: signed realised log return over the horizon, scored at the horizon even after an invalidation; Brier only with a probability (the user gave none).")
    a("")
    a("Seeded with the user's view (author user, created 2026-11-15T19:00Z; component d at 19:15Z), split into components:")
    a("")
    a("| id | component | direction | horizon | trigger | proxies (supporting sign) | invalidation | status at 19:30 UTC |")
    a("|---|---|---|---|---|---|---|---|")
    ledger = {t["id"]: t for t in theses.load_ledger()}
    for r in th["theses"]:
        t = ledger[r["id"]]
        prox = "; ".join(f"{p['id']}" for p in t["proxies"])
        trig = "on creation" if t["trigger"] == "on_create" else "bull steepening AND HY 5-obs change < 0"
        inv = {"a": "2y down >= 30 bp in 20 obs, or 20-obs bull steepening (spread +25 bp with 2y down)",
               "b": "S&P back at its high with NDX leading again, or VIX < 14",
               "c": "HY +50 bp in 20 obs, or Sahm >= 0.5 (recession steepening)",
               "d": "Sahm >= 0.5; or core y/y falling 3 months with 2y -50 bp in a month; or IG OAS below its 1y median while the 10y falls"}[t["component"]]
        sup = sum(1 for p in r["proxies"] if p["supports"] is True)
        tot = sum(1 for p in r["proxies"] if p["supports"] is not None)
        a(f"| {r['id']} | ({t['component']}) {t['title']} | {t['direction']} {'/'.join(x.replace('USDT', '') for x in t['assets'])} | {t['horizon_days']}d | {trig} | {prox} | {inv} | "
          f"**{r['status']}** ({sup}/{tot} support) |")
    a("")
    a("User's numbers kept as theirs: S&P 7000 (b); project IRR 25-30% falling to about 15%, financing 6% rising to 9-10% (d). Watch items that can't be backtested "
      "(hyperscaler capex guidance, the AI bond issuance calendar) go to `research/narratives/events.yaml` through the news agent; the next test is Nvidia's results "
      "on Tue 2026-11-17 22:00 UTC.")
    a("")
    a("## 8. The confluence log (Part 3)")
    a("")
    a("`research/desk/CONFLUENCE.md` + `src/ai_trader/narrative/confluence.py`. Each cycle the trader writes `<snapshot>/reports/confluence.json`: for each candidate "
      "asset, one signed reading in [-1, 1] per layer, each with an evidence pointer that the validator checks against the snapshot. Layers: **levels / liquidity, "
      "flow, book, structure, momentum, vol / liquidity state, positioning, narrative, on-chain** (null, 'no source', until a source exists). Logged append-only "
      "to `data/journal/confluence/`. `scripts/confluence_fit.py --hyp <registered id>` refuses to fit before 300 cycles and before the hypothesis is registered; "
      "then it fits ridge (vol-scaled return), L2 logistic (P(up)) and ridge (move size on |readings|) walk-forward (5 expanding folds, purged by the horizon) and "
      "reports the weights with their fold spread, OOS IC, hit rate and Brier vs the base rate. The CONTRACT paragraph for the orchestrator is "
      "`research/desk/CONTRACT_addendum_confluence.md`. Suggested registration before the first fit (not registered here): 'H-CONF1: a walk-forward linear "
      "combination of the nine layer readings has OOS IC > 0 for the 24h vol-scaled return, and beats the base-rate Brier for P(up), in both segments'.")
    a("")
    a("This is the practical route from the user's discretionary process to model execution: log the read of each layer as a number, with its evidence, "
      "*before* the outcome. After a few hundred cycles the fit shows which layers carry information once the others are known, and which only feel informative "
      "because they agree with the rest.")
    a("")
    a("## 9. On-chain sources memo (Part 4)")
    a("")
    a("What can legally give 'suspicious transaction / whale movement' signals. No collector needing a key was built; the user would have to supply the keys "
      "or subscriptions marked below. Latency is from the on-chain event to availability.")
    a("")
    a("| source | what it gives | access | cost | latency | what the user must supply |")
    a("|---|---|---|---|---|---|")
    for row in ONCHAIN:
        a("| " + " | ".join(row) + " |")
    a("")
    a("Caveats: exchange and entity labels (Whale Alert, Arkham, CryptoQuant, Glassnode) are heuristics and get revised, and a vendor's history may have been "
      "relabelled after the fact, so a backtest on today's labels has look-ahead. Treat it like the desk's `liq_map`: an estimate. Free mempool / Etherscan data "
      "is raw, and labelling it is our own deterministic code. R-NEWS-7 already says on-chain activity counts were not a BTC direction signal in our tests, so any "
      "new on-chain feature needs its own registered hypothesis before it enters the confluence layer.")
    a("")
    a("## 10. What this changes")
    a("")
    a("- The curve regime is context, not direction (R-MAC-1). The user's bear-flattening read of today is correct on the 5-day definition, but the history "
      "(2020-2026) doesn't say BTC does badly after it.")
    a("- The credit leg is where the signal lead is (H-MAC2 points the right way in both segments, not significant). Component (c)'s trigger already requires credit easing.")
    a("- The ledger makes the user's narrative falsifiable and scored. The confluence log turns the rest of their process into data.")
    a("")
    a("## Rule candidates")
    a("")
    a("All grade C (no hypothesis supported), so status provisional, not active. Validated with `ai_trader.research.rules.validate` (no errors).")
    a("")
    a("```yaml")
    a(rules_yaml.rstrip())
    a("```")
    md = "\n".join(L) + "\n"
    REP.with_suffix(".md").write_text(md, encoding="utf-8")
    js = {k: v for k, v in R.items() if not k.startswith("_") and k != "history"}
    js |= {"theses_check": th, "figures": [f"research/figures/macro_narrative_{DAY}/{n}" for n, _ in figs], "onchain_memo": ONCHAIN_JSON,
           "rule_candidates": yaml.safe_load(rules_yaml)["rules"], "rules_validation_errors": rv}
    REP.with_suffix(".json").write_text(json.dumps(js, indent=1, default=str), encoding="utf-8")
    print(f"wrote {REP.with_suffix('.md')} ({len(md.splitlines())} lines), {len(figs)} figures; rules valid")


ONCHAIN = [
    ("blockchain.com charts / API", "BTC network aggregates (tx count, volume, mempool size, hash rate, exchange-trade volume)", "public, no key (already used: src/ai_trader/news/onchain.py)", "free", "daily aggregates, about 1 day", "nothing"),
    ("mempool.space REST + websocket", "BTC mempool, fee rates, every unconfirmed / confirmed tx and block; large-tx detection is our own filter (value threshold, address reuse)", "public, no key (fair-use rate limits; self-hostable)", "free", "seconds (mempool), about 10 min (block)", "nothing for the public API; optional self-hosted node"),
    ("Etherscan API", "ETH / ERC-20 transfers, token holders, contract events, gas; large-transfer scans by address or token", "free API key", "free tier (about 5 calls/s, 100k/day); paid tiers above", "about 12 s (one block)", "an Etherscan API key in .env"),
    ("Whale Alert", "labelled large transfers (exchange in / out, mint / burn) across BTC, ETH, stablecoins", "free tier = public X/Telegram feed; REST / websocket API needs a key", "free (social feed); API subscription paid (from about $30/month unverified, from memory; check before buying)", "seconds to about 1 min", "an API key for programmatic use (scraping the social feed is not an option)"),
    ("Arkham Intelligence", "entity-labelled flows (funds, exchanges, governments), alerts on addresses / entities", "account; API by application / key", "platform free with limits; API access paid / by agreement", "seconds to minutes", "an account and API approval"),
    ("CryptoQuant", "exchange in / outflows, reserves, miner flows, fund flows, stablecoin flows (entity-labelled)", "account; API key on paid plans", "paid (roughly $30-$800/month by tier unverified, from memory; check before buying)", "10 min to 1 h on lower tiers, faster on higher", "a subscription and API key"),
    ("Glassnode", "on-chain metrics (exchange balances, SOPR, MVRV, whale cohorts, supply by age)", "account; API on Professional tier", "paid (API only on the top tiers, hundreds of $/month and up; unverified, check)", "1 h to 1 day by metric and tier", "a Professional subscription and API key"),
]
ONCHAIN_JSON = [dict(zip(("source", "gives", "access", "cost", "latency", "user_supplies"), r, strict=True)) for r in ONCHAIN]

if __name__ == "__main__":
    main()
