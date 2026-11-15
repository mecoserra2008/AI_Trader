"""Narrative and rule candidates for the sparsity report (written after the results were seen; descriptive text only,
verdicts come from sparsity_report.verdicts)."""

from __future__ import annotations

import numpy as np


def narrative(sp1: dict, sp2: dict, sp3: dict, V: dict) -> str:
    si, sh = sp1["segments"]["llm_in_sample"], sp1["segments"]["clean_holdout"]
    r95, r98, r99 = sp1["rounds_at"]["0.95"], sp1["rounds_at"]["0.98"], sp1["rounds_at"]["0.99"]
    c_in, c_ho = si["curve"], sh["curve"]
    surv99 = sorted(sp1["input_survival"]["99"].items(), key=lambda kv: -kv[1])
    surv98 = sorted(sp1["input_survival"]["98"].items(), key=lambda kv: -kv[1])
    cl = sp2.get("clusters", {}).get("segments", {})
    cl_in = cl.get("llm_in_sample", {}).get("top", [])
    cl_ho = cl.get("clean_holdout", {}).get("top", [])
    ab_in = sp3["ablation"]["llm_in_sample"]["mean_ablation"]
    ab_ho = sp3["ablation"]["clean_holdout"]["mean_ablation"]
    each = sp3["ablation"]["clean_holdout"]["each_unit_ratio"]
    n_l1 = sum(1 for u in sp3["units"] if u["unit"][0] == 1)
    n_l2 = sum(1 for u in sp3["units"] if u["unit"][0] == 2)
    d = sp1["direction"]
    dsi, dsh = d["segments"]["llm_in_sample"], d["segments"]["clean_holdout"]
    rd = d["rounds_at"]["0.95"]
    top_units = sorted(sp3["units"], key=lambda q: -q["attribution"])[:5]
    L = []
    L.append(
        "**The vol network is mostly redundant weight.** Pruning the H-NN1 MLP to "
        f"{100 * sp1['sparsity'][r95]:.1f}% sparsity leaves {sp1['n_alive'][r95]} of {sp1['n_alive'][0]} weights, about "
        f"{sp1['units']['95']['mean']:.0f} of 48 hidden units and {sp1['inputs_alive']['95']['mean']:.0f} of 52 inputs. "
        f"It forecasts as well as the dense ensemble or slightly better. The QLIKE ratio is {c_in[r95]['ratio']:.3f} "
        f"[{c_in[r95]['ci95'][0]:.3f}, {c_in[r95]['ci95'][1]:.3f}] in-sample and {c_ho[r95]['ratio']:.3f} "
        f"[{c_ho[r95]['ci95'][0]:.3f}, {c_ho[r95]['ci95'][1]:.3f}] in the holdout. Against log-HAR the 95% network gets "
        f"{100 * c_in[r95]['vs_har']['rel']:+.1f}% (t {c_in[r95]['vs_har']['t']:.1f}) in-sample and {100 * c_ho[r95]['vs_har']['rel']:+.1f}% "
        f"(t {c_ho[r95]['vs_har']['t']:.1f}) in the holdout; the dense network got -8.0% / -6.2%. "
        "H-SP1 passes in-sample but is inconclusive in the holdout: 133 dates leave the CI upper bound at "
        f"{c_ho[r95]['ci95'][1]:.3f}, above the 1.02 bar. "
        f"Beyond about 97% the circuit starts to break in-sample: {c_in[r98]['ratio']:.3f} at {100 * sp1['sparsity'][r98]:.1f}% "
        f"({sp1['n_alive'][r98]} weights) and {c_in[r99]['ratio']:.3f} at {100 * sp1['sparsity'][r99]:.1f}% ({sp1['n_alive'][r99]} weights).")
    L.append("")
    L.append(
        "**What survives is the vol level and the calendar.** At 99% sparsity, across 56 refits x 3 seeds, the inputs that keep a path to the output are "
        + ", ".join(f"{k} {100 * v:.0f}%" for k, v in surv99[:8])
        + ". At 98%: " + ", ".join(f"{k} {100 * v:.0f}%" for k, v in surv98[:8]) + ". "
        "Jump, semivariance, range-estimator, funding, OI and raw-return inputs are pruned first. "
        f"Stability selection on a {sp2['p']}-column library agrees, at the level of concepts rather than columns. "
        f"No single column reaches probability 0.8 in either segment ({sp2['segments']['llm_in_sample']['n_selected']} / {sp2['segments']['clean_holdout']['n_selected']} selected), "
        "because the vol-level measures substitute for each other. So H-SP2 contradicts as registered. "
        "The post-hoc cluster view (|rho| >= 0.8) shows what is stable:"
        + (f" the vol-level cluster ({cl_in[0]['prob']:.2f} in-sample / {cl_ho[0]['prob']:.2f} holdout);" if cl_in and cl_ho else "")
        + (f" the weekend-calendar cluster ({cl_in[1]['prob']:.2f} / {cl_ho[1]['prob']:.2f});" if len(cl_in) > 1 and len(cl_ho) > 1 else "")
        + " and nothing else in both segments. "
        "The minimal input set for next-day vol is one vol-level measure (robust sigma or the 7-day mean log RV, with the 30/90-day level as a "
        "slower anchor) plus the forecast day's weekend / TradFi-weekend hours. Every other tool output is optional for this target.")
    L.append("")
    plain = "; ".join(f"L{u['unit'][0]}u{u['unit'][1]}: {u['describe']['plain']}" for u in top_units)
    L.append(
        f"**The surviving neurons are readable, but redundant rather than monosemantic.** The analysed network (seed 0, 2026-07 refit, "
        f"{sp3['level']}% level) has {sp3['n_units']} surviving units ({n_l1} in layer 1, {n_l2} in layer 2). "
        f"Layer-1 units are mostly 1-4-input ramps. The five layer-2 units carry most attribution, and each is a mix of vol level and weekday. "
        f"{sp3['clean_holdout']['n_meet_r2']} / {sp3['clean_holdout']['n_evaluable']} units are fitted with holdout R^2 > 0.8 by an additive spline of 2 inputs "
        f"({100 * sp3['clean_holdout']['share_meet_1input']:.0f}% by 1 input), short of the 80% bar. "
        f"Mean-ablating the top-3 units raises QLIKE x{ab_in['ratio']:.2f} [{ab_in['ci95'][0]:.2f}, {ab_in['ci95'][1]:.2f}] in-sample and "
        f"x{ab_ho['ratio']:.2f} [{ab_ho['ci95'][0]:.2f}, {ab_ho['ci95'][1]:.2f}] in the holdout. "
        f"Ablating any single unit changes holdout QLIKE by {100 * (min(each.values()) - 1):+.0f}% to {100 * (max(each.values()) - 1):+.0f}%. "
        "The circuit is a few redundant copies of one function, 'vol level, damped on calm weekends', so H-SP3 contradicts. "
        f"Top units in plain words: {plain}.")
    L.append("")
    L.append(
        "**Direction does not collapse to the base rate, but it carries nothing.** The pruned direction network keeps "
        f"{d['units'][str(rd)]['mean']:.0f} units at 95%; its forecasts keep sd {dsi[rd]['sd_p']:.3f}. Its in-sample Brier skill stays at "
        f"{dsi[rd]['skill']:+.4f} [{dsi[rd]['ci95'][0]:+.4f}, {dsi[rd]['ci95'][1]:+.4f}] (dense {dsi[0]['skill']:+.4f}). "
        f"The holdout is {dsh[rd]['skill']:+.4f} [{dsh[rd]['ci95'][0]:+.4f}, {dsh[rd]['ci95'][1]:+.4f}] on 133 dates, one of 23 rounds looked at. "
        "The inputs that survive at 99% are market-wide day-level variables (market log RV, downside share, market and BTC returns), "
        "the same date-noise signature as H-NN2. Read this as no signal, not as a lead.")
    L.append("")
    L.append(
        "**What this means for the decision layer.** For next-day vol, call the vol-level tool (robust sigma / log-HAR level) and the calendar "
        "(weekend and TradFi-weekend hours). The ML vol forecaster adds a modest nonlinear refinement of those two, which a ~100-weight network "
        "reproduces. Lagged, transformed, interacted or quantile versions of the other inputs add no stable information. Don't ask a tool for "
        "next-day direction.")
    _ = np
    return "\n".join(L)


RULES_YAML = """rules:
- id: "R-SP-1"
  kind: "prefer"
  rule: "WHEN preparing next-day volatility context for a crypto perp THEN call the vol-level tool (robust sigma or 7-day mean log RV, with the 30/90-day level as anchor) and the calendar tool (weekend / TradFi-weekend hours of the forecast day) first; other inputs are optional for this target"
  when: "a next-day (1d) volatility expectation is needed for a universe-L crypto perp"
  then: "tools: vol level (robust sigma or lrv_w) + calendar (next_is_weekend, next_wk_hours); jumps, semivariances, range estimators, funding, OI, lags and products may be skipped"
  scope:
    assets: "crypto universe L"
    horizon: "1d"
    sides: "both"
  effect: "a 95%-pruned MLP (100 of 2192 weights) matches the dense one: QLIKE ratio 0.986 [0.964, 1.007] in-sample, 0.974 [0.943, 1.027] holdout; at 99% only log_sigma and lrv_w survive in > 90% of 168 networks; the vol-level cluster is the only stability-selected concept in both segments (1.00 / 0.99)"
  confidence: "B"
  evidence: ["H-SP1", "H-SP2", "H-NN3"]
  review: "a holdout quarter in which a 95%-pruned network is worse than the dense one by > 2% QLIKE, or a non-vol-level input becomes stability-selected in both segments"
  status: "active"
- id: "R-SP-2"
  kind: "avoid"
  rule: "WHEN reading a vol model's attributions THEN do not treat individual vol-level columns (robust sigma, 7d / 30d / 90d log RV, lagged daily RV) or individual hidden units as distinct signals; they are interchangeable copies of one concept"
  when: "explaining or citing why a vol forecast is high or low"
  then: "cite 'vol level' and 'calendar' as concepts; do not argue from which vol column or neuron carries the weight"
  scope:
    assets: "crypto universe L"
    horizon: "1d"
    sides: "both"
  effect: "no single library column reaches stability probability 0.8 in either segment (substitution), while the vol-level cluster does (1.00 / 0.99); single-unit ablations change holdout QLIKE by -5% to +6% while top-3 ablation gives x1.37 (redundant circuit)"
  confidence: "B"
  evidence: ["H-SP2", "H-SP3"]
  review: "a study in which a single vol-level column dominates the others out of sample"
  status: "active"
- id: "R-SP-3"
  kind: "expect"
  rule: "WHEN the forecast day is Fri-Sun and the 7-day vol level is low THEN expect the ML vol forecast to sit below the vol-level baseline (the network's strongest units are calm-weekend dampers); treat this as an expectation, never as a reason to loosen risk"
  when: "next_is_weekend == 1 or next_wk_hours > 0, and lrv_w in its lower half"
  then: "central sigma expectation below the unscaled level; risk sigma unchanged (R-VOL-2)"
  scope:
    assets: "crypto universe L"
    horizon: "1d"
    sides: "both"
  effect: "the two highest-attribution surviving units are fitted by lrv_w / lrv_m + weekday with holdout R^2 0.94 / 0.92, fire more on calm Fri-Sun days and lower the forecast"
  confidence: "C"
  evidence: ["H-SP3", "H-NN1", "H-W1"]
  review: "forward weekends where calm-week realised vol is not below the weekday-scaled level"
  status: "provisional"
- id: "R-SP-4"
  kind: "avoid"
  rule: "WHEN a tool reports a next-day direction probability from the daily ML features THEN do not use it; pruned or dense, the network has no in-sample skill and its surviving inputs are market-wide date noise"
  when: "any next-day direction output of the ML direction networks"
  then: "ignore for direction; no tool call for daily direction"
  scope:
    assets: "crypto universe L"
    horizon: "1d"
    sides: "both"
  effect: "Brier skill in-sample -0.0024 dense to -0.0004 at 99% sparsity, all CIs include 0; holdout +0.003 to +0.012 on 133 dates over 23 rounds looked at (not a lead)"
  confidence: "B"
  evidence: ["H-NN2", "H-SP1"]
  review: "forward direction skill > 0.005 with CI lower > 0 over >= 120 days"
  status: "active"
"""
