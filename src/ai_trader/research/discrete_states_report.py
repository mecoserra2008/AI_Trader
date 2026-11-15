"""Report for the discretised-state study (research/reports/discrete_states_2026-11-12.{json,md}), generated from the
evaluated numbers (discrete_states_study.run_evaluate). Narrative sentences are filled from the results."""

from __future__ import annotations

import json
import math

from ai_trader.core.config import load

TAG = "2026-11-12"
SEGS = ("llm_in_sample", "clean_holdout")
NAMES = {
    "WD7": "weekday (7 cells)", "V0": "v0 regime (9)", "LAT1": "weekday group x vol tercile x v0 direction (36)",
    "LAT2": "weekday group x 7d level tercile x market-day tercile (36)", "LAT3": "weekday x vol tercile x v0 direction (63)",
    "KM8": "k-means 8", "KM16": "k-means 16", "KM32": "k-means 32", "KM64": "k-means 64", "HT3": "honest tree depth 3",
    "HT6": "honest tree depth 6", "HT3r": "honest return tree depth 3", "HT6r": "honest return tree depth 6",
}


def pct(logv: float | None) -> str:
    return "n/a" if logv is None or logv != logv else f"{100 * (math.exp(logv) - 1):+.1f}%"


def rel(x: float | None) -> str:
    return "n/a" if x is None or x != x else f"{100 * x:+.1f}%"


def f2(x: float | None, nd: int = 2) -> str:
    return "n/a" if x is None or x != x else f"{x:.{nd}f}"


def q(s: str) -> str:
    return json.dumps(s, ensure_ascii=False)


GROUP_WORDS = {
    "Saturday, v0 range or trend down": ("the forecast day is Saturday and the v0 trend state is range or down", "Saturday (range / down trend)"),
    "Saturday, v0 trend up": ("the forecast day is Saturday and the v0 trend state is up", "Saturday (up trend)"),
    "Sunday": ("the forecast day is Sunday", "Sunday"),
    "Monday": ("the forecast day is Monday", "Monday"),
    "Monday, vol in the top third and v0 trend up": ("the forecast day is Monday, vol is in the top third of its own 1-year range and the v0 trend state is up", "Monday, high vol, up trend"),
    "Tue-Fri, v0 trend up": ("the forecast day is Tuesday to Friday and the v0 trend state is up", "Tue-Fri (up trend)"),
    "Tue-Fri, v0 range": ("the forecast day is Tuesday to Friday and the v0 trend state is range", "Tue-Fri (range)"),
    "Tue-Fri, v0 trend down": ("the forecast day is Tuesday to Friday and the v0 trend state is down", "Tue-Fri (down trend)"),
}


def rule_groups(res: dict, min_eff: float = 0.08) -> list[tuple[str, dict]]:
    g = res.get("groups_post_hoc", {})
    out = []
    for name, v in g.items():
        a, b = v[SEGS[0]], v[SEGS[1]]
        if a["n"] >= 30 and b["n"] >= 30 and a["effect"] * b["effect"] > 0 and min(abs(a["effect"]), abs(b["effect"])) >= min_eff:
            out.append((name, v))
    return out


def rules_yaml(res: dict) -> str:
    ds1, ds2, ds3, ver = res["H-DS1"], res["H-DS2"], res["H-DS3"], res["verdicts"]
    best = ds1["best"]
    ins, ho = ds1[SEGS[0]], ds1[SEGS[1]]
    out = []
    n = 0
    for name, v in rule_groups(res):
        n += 1
        a, b = v[SEGS[0]], v[SEGS[1]]
        z_ok = all(x is not None and x == x and abs(x) > 2 for x in (a["z"], b["z"])) and b["n_dates"] >= 15
        m = 0.5 * (a["effect"] + b["effect"])
        direction = "above" if m > 0 else "below"
        words = GROUP_WORDS.get(name, (name, name))[0]
        out.append({
            "id": f"R-DS-{n}", "kind": "expect",
            "rule": f"WHEN {words} THEN expect next-day realised variance about {abs(100 * (math.exp(m) - 1)):.0f}% {direction} the per-asset log-HAR forecast",
            "when": f"daily (1d) variance expectation for a universe-L crypto perp; merged cells of the frozen {best} partition ({NAMES.get(best, best)}): {name}",
            "then": f"central next-day variance expectation = log-HAR x {math.exp(m):.2f}; stops and risk keep the unshrunk risk sigma (R-VOL-2)",
            "effect": f"next-day RV vs log-HAR (relative to the segment average) {pct(a['effect'])} in-sample ({a['n']} rows, {a['n_dates']} dates, z {f2(a['z'], 1)}), "
                      f"{pct(b['effect'])} holdout ({b['n']} rows, {b['n_dates']} dates, z {f2(b['z'], 1)}); cell group defined after the cell table was seen; grade B needs |z| > 2 in both segments and >= 15 holdout dates",
            "confidence": "B" if z_ok else "C",
            "review": "register the merged cell as a forward hypothesis; demote if the effect on dates after 2026-11-12 changes sign or falls below half",
        })
    n += 1
    b_in, b_ho = ins[best], ho[best]
    out.append({
        "id": f"R-DS-{n}", "kind": "prefer",
        "rule": f"WHEN a central next-day variance expectation is needed and no validated nonlinear tool is available THEN apply the {best} cell adjustment to the per-asset log-HAR forecast",
        "when": "daily (1d) variance expectation for a universe-L crypto perp",
        "then": "forecast = log-HAR x exp(EB-shrunk cell value of the cell table); fall back to plain log-HAR for cells with fewer than 30 training rows",
        "effect": f"QLIKE vs log-HAR {rel(b_in['rel'])} (t {f2(b_in['t'])}) in-sample, {rel(b_ho['rel'])} (t {f2(b_ho['t'])}) holdout; share of the XGB gain {f2(ins['best_share']['share'])} / {f2(ho['best_share']['share'])}; H-DS1 {ver['H-DS1']['pooled']}, H-DS2 rho {f2(ds2['rho'])}",
        "confidence": "B" if ver["H-DS1"]["pooled"] == "supports" else "C",
        "review": "forward test against log-HAR + weekday and the frozen XGB tool for 3 months; demote if the gain is < 2% or t > -2",
    })
    n += 1
    bd = ds3["best"]
    out.append({
        "id": f"R-DS-{n}", "kind": "avoid",
        "rule": "WHEN a discrete market state (weekday, vol tercile, trend state, k-means or tree cell) is known THEN don't infer next-day direction from it",
        "when": "next-24h direction for crypto perps, from any cell table of standard features",
        "then": "treat direction as a coin flip; cell tables are for variance expectations only",
        "effect": f"best in-sample direction cells ({bd}): holdout Brier skill {f2(ds3[SEGS[1]]['skill'], 4)} [{f2(ds3[SEGS[1]]['ci95'][0], 4)}, {f2(ds3[SEGS[1]]['ci95'][1], 4)}]; cell ordering by next-day return Spearman {f2(ds3['ordering']['rho'])} between segments; H-DS3 {ver['H-DS3']['pooled']}",
        "confidence": "B" if ver["H-DS3"]["pooled"] == "supports" else "C",
        "review": "a pre-registered direction cell with holdout Brier skill CI lower bound > 0",
    })
    lines = []
    for r in out:
        lines.append(f"- id: {q(r['id'])}")
        lines.append(f"  kind: {q(r['kind'])}")
        lines.append(f"  rule: {q(r['rule'])}")
        lines.append(f"  when: {q(r['when'])}")
        lines.append(f"  then: {q(r['then'])}")
        lines.append("  scope:")
        lines.append(f"    assets: {q('crypto perps (universe L)')}")
        lines.append(f"    horizon: {q('1d')}")
        lines.append(f"    sides: {q('both')}")
        lines.append(f"  effect: {q(r['effect'])}")
        lines.append(f"  confidence: {q(r['confidence'])}")
        ev = ["H-DS1", "H-DS2"] if r["kind"] != "avoid" else ["H-DS3", "H-NN2"]
        lines.append("  evidence: [" + ", ".join(q(e) for e in ev) + "]")
        lines.append(f"  review: {q(r['review'])}")
        lines.append(f"  status: {q('provisional')}")
    return "\n".join(lines)


def narrative(res: dict) -> list[str]:
    ds1, ds2, ds3, ver = res["H-DS1"], res["H-DS2"], res["H-DS3"], res["verdicts"]
    best = ds1["best"]
    ins, ho = ds1[SEGS[0]], ds1[SEGS[1]]
    g = res.get("groups_post_hoc", {})
    n_in = len(ins["passing"])
    n_ho = len(ho["passing"])
    n_all = len([k for k in ins if k in NAMES])
    wk_b = ds1[SEGS[1]]["by_weekend"]["1.0"]
    L = []
    L.append(f"**How much do a few cells carry?** The best discretisation chosen in-sample is {best} ({NAMES.get(best, best)}): log-HAR times one "
             f"shrunk factor per cell. It cuts next-day QLIKE by {rel(ins[best]['rel'])} (t {f2(ins[best]['t'])}) in-sample and {rel(ho[best]['rel'])} "
             f"(t {f2(ho[best]['t'])}) in the holdout. That beats XGB in-sample, where the XGB gain itself is noise ({rel(ins['xgb_vs_har']['rel'])}, t {f2(ins['xgb_vs_har']['t'])}), "
             f"but recovers only {f2(ho['best_share']['share'])} of XGB's holdout gain ({rel(ho['xgb_vs_har']['rel'])}); the share CI [{f2(ho['best_share']['ci95'][0])}, "
             f"{f2(ho['best_share']['ci95'][1])}] neither clears nor rules out the 0.8 bar, so H-DS1 is {ver['H-DS1']['pooled']}. {n_in} of {n_all} discretisations clear "
             f"the bar in-sample, {n_ho} in the holdout. The holdout shortfall is concentrated on weekends: on Saturday / Sunday forecast days the cells give "
             f"{rel(wk_b['best']['rel'])} against XGB's {rel(wk_b['xgb']['rel'])}.")
    if g:
        a = g["Saturday, v0 range or trend down"]
        b = g["Monday"]
        c = g["Tue-Fri, v0 trend up"]
        d = g["Saturday, v0 trend up"]
        L.append("")
        L.append(f"**What the cells say (cell table below).** Almost all of the cell signal is the forecast weekday, which log-HAR lacks, modulated by the v0 trend state. "
                 f"Relative to log-HAR: Saturday with a range or down trend {pct(a[SEGS[0]]['effect'])} in-sample / {pct(a[SEGS[1]]['effect'])} holdout; "
                 f"Monday {pct(b[SEGS[0]]['effect'])} / {pct(b[SEGS[1]]['effect'])}; Tue-Fri in a v0 up-trend {pct(c[SEGS[0]]['effect'])} / {pct(c[SEGS[1]]['effect'])}. "
                 f"An up-trend shrinks the Saturday discount ({pct(d[SEGS[0]]['effect'])} in-sample, {pct(d[SEGS[1]]['effect'])} on {d[SEGS[1]]['n']} holdout rows). "
                 "Within a weekday the vol tercile matters much less than in the H-NN3 attributions, because log-HAR already carries the level.")
    L.append("")
    L.append(f"**Do the cells keep their meaning?** Over {ds2['n_cells_both']} cells with at least 30 rows in both segments, the frozen {best} cell ordering has "
             f"Spearman {f2(ds2['rho'])} (bootstrap CI [{f2(ds2['ci95'][0])}, {f2(ds2['ci95'][1])}], Fisher [{f2(ds2['ci95_fisher'][0])}, {f2(ds2['ci95_fisher'][1])}]), "
             f"so H-DS2 is {ver['H-DS2']['pooled']} against the bar of 0.8. The weekday-level structure is stable; the order within a weekday group is noisier "
             "(holdout cells have 20-120 rows).")
    L.append("")
    L.append(f"**Direction.** No cell table forecasts the sign of the next-day return. The best in-sample direction cells ({ds3['best']}) have in-sample Brier skill "
             f"{f2(ds3['brier'][SEGS[0]][ds3['best']]['skill'], 4)} and holdout skill {f2(ds3[SEGS[1]]['skill'], 4)} [{f2(ds3[SEGS[1]]['ci95'][0], 4)}, {f2(ds3[SEGS[1]]['ci95'][1], 4)}]. "
             "For most partitions the empirical-Bayes estimate puts the between-cell variance of returns at zero, which shrinks every cell to the base rate "
             f"(the ordering is then defined as rho = 0). The return ordering of {ds3['best']} has Spearman {f2(ds3['ordering']['rho'])} across segments. "
             f"H-DS3 (the null) is {ver['H-DS3']['pooled']}.")
    return L


def caveats(res: dict) -> list[str]:
    return [
        "## Caveats",
        "",
        "- The holdout is 133 dates (2026-07 to 2026-11). The XGB holdout gain that H-DS1 is measured against rests on those dates and is concentrated on weekends.",
        "- 'Best' was chosen on in-sample QLIKE. Other discretisations (LAT3, KM64) come close in the holdout; see the H-DS1 table.",
        "- The k-means inputs are the top in-sample H-NN3 TreeSHAP features, computed on all in-sample walk-forward rows. Early in-sample months therefore carry mild "
        "feature-selection look-ahead; the holdout is clean. The lattices use only calendar, own-vol percentile, v0 trend and market-day inputs, fixed in advance.",
        "- Percentile bootstrap intervals of a Spearman between two noisy cell tables are biased toward 0: resampling adds a second layer of noise, so the bootstrap median sits "
        "below the point estimate. Verdicts use the registered bootstrap; Fisher-z intervals are shown for reference.",
        "- Cell effects are relative to each segment's average log-HAR residual, and are geometric (mean log ratio). Under QLIKE the arithmetic scale differs by roughly "
        "s^2/2 per cell, which matters little for the ordering but does matter for an exact multiplier.",
        "- The merged groups behind the expect rules were defined after the cell table was seen. They are rule candidates, not tested hypotheses, and should be "
        "registered as forward hypotheses before being promoted.",
        "- Mistake (fixed before recording): with a between-cell variance of 0, EB weights of about 1e-12 left a float-residue ordering that the Spearman read as signal. "
        "Cell values are now exactly 0 and the ordering is defined as rho = 0 in that case (diary 'mistake' entry). The walk-forward forecasts were produced before "
        "the fix; the difference is below 1e-8 in relative terms.",
    ]


def build(res: dict) -> str:
    ds1, ds2, ds3, ver, hp = res["H-DS1"], res["H-DS2"], res["H-DS3"], res["verdicts"], res["holm"]
    best = ds1["best"]
    ins, ho = ds1[SEGS[0]], ds1[SEGS[1]]
    ct = res["cell_table"]
    L: list[str] = []
    L.append("# Discretised observation space: how many cells carry the volatility signal (H-DS1 to H-DS3)")
    L.append("")
    L.append(f"_Program {TAG} (acceleration, agent C). Data as of {TAG}; knowledge cutoff {load().settings['research']['knowledge_cutoff']}. "
             "Code: `src/ai_trader/ml/discretize.py` (lattices, k-means codebooks, honest trees, empirical-Bayes cells) and "
             "`src/ai_trader/research/discrete_states_study.py`; tests `tests/test_discretize.py`; artefacts `data/models/ml/discrete/`. "
             "Deterministic research tools only; the LLM decision layer is not trained or backtested here (principle 1)._")
    L.append("")
    L.append("## Summary")
    L.append("")
    L.append("| hypothesis | in-sample | holdout | pooled | headline |")
    L.append("|---|---|---|---|---|")
    L.append(f"| H-DS1: <= 64 cells recover >= 80% of the XGB vol gain | {ver['H-DS1'][SEGS[0]]} | {ver['H-DS1'][SEGS[1]]} | **{ver['H-DS1']['pooled']}** | "
             f"best ({best}, chosen in-sample) {rel(ins[best]['rel'])} QLIKE (t {f2(ins[best]['t'])}) in-sample, {rel(ho[best]['rel'])} (t {f2(ho[best]['t'])}) holdout; "
             f"share of the XGB gain {f2(ins['best_share']['share'])} [{f2(ins['best_share']['ci95'][0])}, {f2(ins['best_share']['ci95'][1])}] / "
             f"{f2(ho['best_share']['share'])} [{f2(ho['best_share']['ci95'][0])}, {f2(ho['best_share']['ci95'][1])}] |")
    L.append(f"| H-DS2: the cell ordering is stable across the cutoff | - | - | **{ver['H-DS2']['pooled']}** | Spearman {f2(ds2['rho'])} "
             f"[{f2(ds2['ci95'][0])}, {f2(ds2['ci95'][1])}] over {ds2['n_cells_both']} of {ds2['n_cells']} cells |")
    L.append(f"| H-DS3: no discrete state predicts direction stably (null) | {ver['H-DS3'][SEGS[0]]} | {ver['H-DS3'][SEGS[1]]} | **{ver['H-DS3']['pooled']}** | "
             f"best in-sample direction cells {ds3['best']}: holdout Brier skill {f2(ds3[SEGS[1]]['skill'], 4)} [{f2(ds3[SEGS[1]]['ci95'][0], 4)}, {f2(ds3[SEGS[1]]['ci95'][1], 4)}]; "
             f"return ordering Spearman {f2(ds3['ordering']['rho'])} [{f2(ds3['ordering']['ci95'][0])}, {f2(ds3['ordering']['ci95'][1])}] |")
    L.append("")
    L.append(f"Holm within DS (p): H-DS1 {f2(hp['H-DS1'], 3)}, H-DS2 {f2(hp['H-DS2'], 3)}, H-DS3 {f2(hp['H-DS3'], 3)}. "
             f"Common sample {res['n_common']} rows; the re-fitted log-HAR matches H-NN1's to a max relative difference of {res['har_replication_max_rel_diff']:.1e}.")
    L.append("")
    L.extend(narrative(res))
    L.append("")
    L.append("## Methods")
    L.append("")
    L.append("- **Panel and protocol** = H-NN1: universe L (23 crypto perps), 52 point-in-time daily features, complete rows, decision at 00:00 UTC of D+1, "
             f"expanding monthly refits {res['first_test_month']} to 2026-11 with training labels ending at least one day before the test month. "
             "The XGB forecasts are H-NN1's own walk-forward forecasts on the same rows.")
    L.append("- **Cell value**: in each refit, u = log(RV_{D+1} / log-HAR fitted value) on the training rows; the cell value is the empirical-Bayes "
             "shrunk mean of u - mean(u): date-clustered sampling variances, one pooled kappa = sigma^2 / tau^2 per refit (moment estimator), "
             "weight n / (n + kappa), shrink toward 0 (= plain log-HAR). Forecast = log-HAR x exp(cell value).")
    L.append("- **Discretisations** (fitted on training rows only, <= 64 cells): WD7 weekday; V0 the 9 v0 regime cells; LAT1 weekday group (Mon / Tue-Fri / Sat / Sun) x "
             "vol_pct_1y tercile x v0 direction (36); LAT2 weekday group x lrv_w tercile x market-day tercile (36); LAT3 weekday x vol tercile x v0 direction (63); "
             "KM8-KM64 k-means on the top-10 H-NN3 TreeSHAP inputs + dow_cos; HT3 / HT6 honest trees (split on alternating training weeks, leaf values from the other weeks). "
             "Direction uses the same partitions with return trees HT3r / HT6r.")
    L.append("- **Selection**: the best vol discretisation has the lowest walk-forward llm_in_sample QLIKE; the best direction one the highest in-sample Brier skill. "
             f"H-DS2 / H-DS3 orderings use the partition frozen at the {res['frozen_month']} refit (labels before the cutoff), with out-of-sample HAR residuals, "
             "cells with >= 30 rows in both segments, and a 10-day date-block bootstrap within each segment.")
    L.append("")
    L.append("## Results")
    L.append("")
    L.append("### H-DS1: QLIKE vs per-asset log-HAR, and share of the XGB gain")
    L.append("")
    L.append("| discretisation | cells | in-sample rel | t | share of XGB | holdout rel | t | share of XGB |")
    L.append("|---|---|---|---|---|---|---|---|")
    for ref in ("xgb_vs_har", "mlp_vs_har", "har_dow_vs_har"):
        lab = {"xgb_vs_har": "XGB (H-NN1 reference)", "mlp_vs_har": "MLP (H-NN1)", "har_dow_vs_har": "log-HAR + weekday dummies (H-NN1 exploratory)"}[ref]
        L.append(f"| {lab} | - | {rel(ins[ref]['rel'])} | {f2(ins[ref]['t'])} | - | {rel(ho[ref]['rel'])} | {f2(ho[ref]['t'])} | - |")
    cells_n = res.get("n_cells", {})
    for d in [k for k in NAMES if k in ins]:
        star = " **(best)**" if d == best else ""
        L.append(f"| {d}: {NAMES[d]}{star} | {cells_n.get(d) or '-'} | {rel(ins[d]['rel'])} | {f2(ins[d]['t'])} | {f2(ins[d]['share_of_xgb'])} | "
                 f"{rel(ho[d]['rel'])} | {f2(ho[d]['t'])} | {f2(ho[d]['share_of_xgb'])} |")
    L.append("")
    L.append(f"XGB gain (diff) {ins['xgb_vs_har']['diff']:+.4f} in-sample (t {f2(ins['xgb_vs_har']['t'])}), {ho['xgb_vs_har']['diff']:+.4f} holdout (t {f2(ho['xgb_vs_har']['t'])}), "
             f"on {ins['xgb_vs_har']['n_dates']} / {ho['xgb_vs_har']['n_dates']} dates.")
    L.append("")
    L.append(f"Best ({best}) against the references (diff, NW t): vs XGB {ins['best_vs_xgb']['diff']:+.4f} ({f2(ins['best_vs_xgb']['t'])}) in-sample, "
             f"{ho['best_vs_xgb']['diff']:+.4f} ({f2(ho['best_vs_xgb']['t'])}) holdout; vs log-HAR + weekday {ins['best_vs_har_dow']['diff']:+.4f} ({f2(ins['best_vs_har_dow']['t'])}) / "
             f"{ho['best_vs_har_dow']['diff']:+.4f} ({f2(ho['best_vs_har_dow']['t'])}). "
             f"Passing the H-DS1 bar (share >= 0.8 and t < -2): in-sample {', '.join(ins['passing']) or 'none'}; holdout {', '.join(ho['passing']) or 'none'}.")
    L.append("")
    L.append("| segment | forecast day | best rel vs HAR | t | XGB rel vs HAR | t |")
    L.append("|---|---|---|---|---|---|")
    for sg in SEGS:
        for k, lab in (("0.0", "Mon-Fri"), ("1.0", "Sat-Sun")):
            bw = ds1[sg]["by_weekend"][k]
            L.append(f"| {sg} | {lab} | {rel(bw['best']['rel'])} | {f2(bw['best']['t'])} | {rel(bw['xgb']['rel'])} | {f2(bw['xgb']['t'])} |")
    L.append("")
    L.append("### H-DS2: is the cell ordering stable across the cutoff? (frozen partitions, Spearman over cells with >= 30 rows in both segments)")
    L.append("")
    L.append("| discretisation | rho | bootstrap 95% CI | Fisher 95% CI | cells used |")
    L.append("|---|---|---|---|---|")
    for d, v in ds2["all"].items():
        star = " **(best, the test)**" if d == best else ""
        L.append(f"| {d}{star} | {f2(v['rho'])} | [{f2(v['ci95'][0])}, {f2(v['ci95'][1])}] | [{f2(v['ci95_fisher'][0])}, {f2(v['ci95_fisher'][1])}] | {v['n_cells_both']} |")
    L.append("")
    L.append(f"Spread of the shrunk cell effects (sd over the cells used): {f2(ds2['sd_cells_in'], 3)} in-sample, {f2(ds2['sd_cells_ho'], 3)} holdout log points; "
             f"segment kappas {f2(ds2['kappa_in'], 0)} / {f2(ds2['kappa_ho'], 0)}.")
    L.append("")
    L.append(f"### Cell table: {best} ({NAMES.get(best, best)}), partition frozen at the {res['frozen_month']} refit")
    L.append("")
    L.append("Effect = EB-shrunk next-day RV relative to the per-asset log-HAR forecast (exp(cell value) - 1), relative to the segment's average residual. "
             "'train' is the value the frozen refit actually used (estimated on its own training rows). z = raw cell mean / date-clustered SE.")
    L.append("")
    L.append("| cell | state in plain words | rows in | effect in | z in | rows ho | effect ho | z ho | train |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    for r in ct["rows"]:
        L.append(f"| {r['cell']} | {r['words']} | {r['n_in']} | {pct(r['effect_in'])} | {f2(r['z_in'], 1)} | {r['n_ho']} | {pct(r['effect_ho'])} | {f2(r['z_ho'], 1)} | {pct(r['train_effect'])} |")
    L.append("")
    L.append(f"Kappa (pseudo-rows): in-sample {f2(ct['kappa_in'], 0)}, holdout {f2(ct['kappa_ho'], 0)}, frozen training {f2(ct['kappa_train'], 0)}.")
    L.append("")
    g = res.get("groups_post_hoc", {})
    if g:
        L.append("**Merged cells (post hoc, defined after reading the table; raw centred mean log(RV / log-HAR), date-clustered z):**")
        L.append("")
        L.append("| merged cells | rows in | effect in | z in | rows ho | effect ho | z ho |")
        L.append("|---|---|---|---|---|---|---|")
        for name, v in g.items():
            a, b = v[SEGS[0]], v[SEGS[1]]
            L.append(f"| {name} | {a['n']} | {pct(a['effect'])} | {f2(a['z'], 1)} | {b['n']} | {pct(b['effect'])} | {f2(b['z'], 1)} |")
        L.append("")
    L.append("### H-DS3: direction from cells (Brier skill vs the training base rate; cell ordering by next-day vol-scaled return)")
    L.append("")
    L.append("| discretisation | in-sample skill | 95% CI | holdout skill | 95% CI | return ordering rho | bootstrap 95% CI | Fisher 95% CI |")
    L.append("|---|---|---|---|---|---|---|---|")
    for d in ds3["brier"][SEGS[0]]:
        bi, bh, o = ds3["brier"][SEGS[0]][d], ds3["brier"][SEGS[1]][d], ds3["ordering_all"][d]
        star = " **(best in-sample)**" if d == ds3["best"] else ""
        L.append(f"| {d}{star} | {f2(bi['skill'], 4)} | [{f2(bi['ci95'][0], 4)}, {f2(bi['ci95'][1], 4)}] | {f2(bh['skill'], 4)} | [{f2(bh['ci95'][0], 4)}, {f2(bh['ci95'][1], 4)}] | "
                 f"{f2(o['rho'])} | [{f2(o['ci95'][0])}, {f2(o['ci95'][1])}] | [{f2(o['ci95_fisher'][0])}, {f2(o['ci95_fisher'][1])}] |")
    L.append("")
    L.append(f"P*(return ordering rho > 0.3) {f2(ds3['ordering']['p_gt_03'], 3)}; P*(holdout skill > 0) {f2(ds3['p_skill_gt0'], 3)}.")
    L.append("")
    L.append("### Verdicts (after Holm within DS)")
    L.append("")
    L.append("| hypothesis | llm_in_sample | clean_holdout | pooled | Holm p |")
    L.append("|---|---|---|---|---|")
    for h in ("H-DS1", "H-DS2", "H-DS3"):
        v = ver[h]
        L.append(f"| {h} | {v.get(SEGS[0], '-')} | {v.get(SEGS[1], '-')} | {v['pooled']} | {f2(hp[h], 3)} |")
    L.append("")
    L.extend(caveats(res))
    L.append("")
    L.append("## Rule candidates")
    L.append("")
    L.append("```yaml")
    L.append(rules_yaml(res))
    L.append("```")
    L.append("")
    return "\n".join(L)


def write(res: dict) -> None:
    from ai_trader.research.discrete_states_study import _clean

    root = load().root / "research" / "reports"
    res = _clean(res)
    (root / f"discrete_states_{TAG}.json").write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")
    (root / f"discrete_states_{TAG}.md").write_text(build(res), encoding="utf-8")
    print("wrote", root / f"discrete_states_{TAG}.md")
