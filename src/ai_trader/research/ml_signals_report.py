"""Markdown report for the ML signal study (research/reports/ml_signals_2026-11-12.md): narrative filled from the
evaluated numbers, the generated result tables, the 'what to call when' synthesis and the rule candidates."""

from __future__ import annotations

import math

SEGS = ("llm_in_sample", "clean_holdout")


def _p(x: float) -> str:
    return f"{100 * x:+.1f}%"


def _t(x: float) -> str:
    return f"{x:.2f}" if x == x else "nan"


def _mult(logv: float) -> str:
    return f"{math.exp(logv):.2f}x"


def build(out: dict, tables: str) -> str:
    n1, n2, n3, n4, it = out["H-NN1"], out["H-NN2"], out["H-NN3"], out["H-NN4"], out["interpret"]
    v = out["verdicts"]
    ex = out["exploratory_har_dow"]
    ce = out["calendar_effect"]
    at = out["attributions"]
    ga = out["group_ablation"]
    ins, ho = "llm_in_sample", "clean_holdout"
    r = lambda s, m: n1[s][f"{m}_vs_har"]  # noqa: E731
    wins_sym = sum(1 for x in n1[ho]["by_symbol_rel"].values() if x["xgb"] < 0)
    n_sym = len(n1[ho]["by_symbol_rel"])
    cal_in, cal_ho = ce[ins]["by_weekday"], ce[ho]["by_weekday"]
    mean_in = sum(cal_in.values()) / 7
    mean_ho = sum(cal_ho.values()) / 7
    gin, gho = ce[ins]["weekend_by_vol_state"], ce[ho]["weekend_by_vol_state"]
    vl_share = (n3[ins]["group_main_share"]["vol_level"], n3[ho]["group_main_share"]["vol_level"])
    cal_share = (n3[ins]["group_main_share"]["calendar"], n3[ho]["group_main_share"]["calendar"])

    def gab(task, s, fam, g, key="rel"):
        return ga[task][s][f"{fam}:{g}"][key]

    L: list[str] = []
    L.append("# Neural-network and tree signal generators, with interpretability (H-NN1 to H-NN4)\n")
    L.append(f"_Program 2026-11-12 (overnight loop, agent B). Data as of {out['as_of'][:10]}; knowledge cutoff {out['cutoff']}. "
             "Code: `src/ai_trader/ml/` (features, models, interpret) and `src/ai_trader/research/ml_signals_study.py`; tests: "
             "`tests/test_ml.py`; models and artifacts: `data/models/ml/`. These models are deterministic research tools. The LLM "
             "decision layer is not trained or backtested here (principle 1)._\n")
    L.append("## Summary\n")
    L.append("| hypothesis | in-sample | holdout | pooled | headline |")
    L.append("|---|---|---|---|---|")
    L.append(f"| H-NN1: nonlinear models beat log-HAR for next-day vol | {v['H-NN1'][ins]} | {v['H-NN1'][ho]} | **{v['H-NN1']['pooled']}** | "
             f"in-sample MLP {_p(r(ins, 'mlp')['rel'])} QLIKE (t {_t(r(ins, 'mlp')['t'])}), XGB {_p(r(ins, 'xgb')['rel'])} (t {_t(r(ins, 'xgb')['t'])}); "
             f"holdout XGB {_p(r(ho, 'xgb')['rel'])} (t {_t(r(ho, 'xgb')['t'])}), MLP {_p(r(ho, 'mlp')['rel'])} (t {_t(r(ho, 'mlp')['t'])}); "
             "the winning model is different in each segment |")
    L.append(f"| H-NN2: nonlinear models do not beat the base rate for direction (null) | {v['H-NN2'][ins]} | {v['H-NN2'][ho]} | {v['H-NN2']['pooled']} | "
             f"Brier skill in-sample MLP {n2[ins]['mlp']['skill']:+.4f}, XGB {n2[ins]['xgb']['skill']:+.4f}; holdout {n2[ho]['mlp']['skill']:+.4f} / "
             f"{n2[ho]['xgb']['skill']:+.4f}, with CI upper bounds {n2[ho]['mlp']['ci95'][1]:+.4f} / {n2[ho]['xgb']['ci95'][1]:+.4f} (above +0.005) |")
    L.append(f"| H-NN3: interactions matter for vol (TreeSHAP) | {v['H-NN3'][ins]} | {v['H-NN3'][ho]} | **{v['H-NN3']['pooled']}** | "
             f"interaction share {100 * n3[ins]['interaction_share']:.1f}% / {100 * n3[ho]['interaction_share']:.1f}% of \\|phi\\|; "
             f"top-5 pair overlap across segments {n3['top5_overlap']}/5 |")
    L.append(f"| H-NN4: the SAE on the MLP's hidden layer finds named market states | {v['H-NN4'][ins]} | {v['H-NN4'][ho]} | **{v['H-NN4']['pooled']}** | "
             f"{it['n_ties_holdout']} of 10 features tie to a state; top-3 ablation {it['ablation'][ho]['top3']['mean']:+.3f} QLIKE "
             f"(t {_t(it['ablation'][ho]['top3']['t'])}) |\n")
    L.append(f"**What the models use.** Next-day volatility runs on two things. The first is the **vol level**: the 7-day mean log RV "
             f"and the robust sigma, {100 * vl_share[0]:.0f}% / {100 * vl_share[1]:.0f}% of all TreeSHAP attribution as main effects. "
             f"The second is the **forecast day's weekday**: the calendar group carries a {100 * cal_share[0]:.0f}% main effect plus "
             f"the largest cross-group interaction. Relative to the model's average day, Saturday is "
             f"{_mult(cal_in['Sat'] - mean_in)} / {_mult(cal_ho['Sat'] - mean_ho)} (in-sample / holdout), Sunday "
             f"{_mult(cal_in['Sun'] - mean_in)} / {_mult(cal_ho['Sun'] - mean_ho)} and Monday "
             f"{_mult(cal_in['Mon'] - mean_in)} / {_mult(cal_ho['Mon'] - mean_ho)}. "
             "Trend regime, funding and the market factor add about 1-3% each. Jump, semivariance, range-estimator and raw-return "
             f"inputs add nothing. Trees and networks agree on this ranking: Spearman {at['vol'][ins]['spearman_feature_ranking']:.2f} / "
             f"{at['vol'][ho]['spearman_feature_ranking']:.2f} over the 52 features, and {at['vol'][ins]['top10_overlap']} / "
             f"{at['vol'][ho]['top10_overlap']} of the top 10 in common. **Direction has no learnable signal.** The models stop after "
             f"{out['walk']['mean_epochs_dir']:.1f} epochs / {out['walk']['mean_xgb_iter_dir']:.0f} trees on average. The families "
             f"disagree on what matters (Spearman {at['dir'][ho]['spearman_feature_ranking']:.2f} in the holdout), and the tree's "
             f"direction attributions sit mostly on market-wide day-level inputs ({100 * at['dir'][ho]['groups']['market']['xgb_shap']:.0f}% "
             "of its attribution), which is the signature of fitting date noise.\n")
    L.append(f"**How much nonlinearity adds over log-HAR.** It adds 6-16% QLIKE, but most of that is calendar knowledge, which "
             f"log-HAR lacks. A ridge-linear model on the same 52 inputs gets {_p(r(ins, 'lin')['rel'])} (t {_t(r(ins, 'lin')['t'])}) "
             f"in-sample and {_p(r(ho, 'lin')['rel'])} (t {_t(r(ho, 'lin')['t'])}) in the holdout. Per-asset log-HAR plus six "
             f"day-of-week dummies (exploratory, post hoc) gets {_p(ex[ins]['har_dow_vs_har']['rel'])} "
             f"(t {_t(ex[ins]['har_dow_vs_har']['t'])}) in-sample and {_p(ex[ho]['har_dow_vs_har']['rel'])} "
             f"(t {_t(ex[ho]['har_dow_vs_har']['t'])}) in the holdout. Beyond that, the MLP and XGB add "
             f"{_p(ex[ins]['mlp_vs_har_dow']['rel'])} / {_p(ex[ins]['xgb_vs_har_dow']['rel'])} in-sample and "
             f"{_p(ex[ho]['mlp_vs_har_dow']['rel'])} / {_p(ex[ho]['xgb_vs_har_dow']['rel'])} "
             f"(t {_t(ex[ho]['mlp_vs_har_dow']['t'])} / {_t(ex[ho]['xgb_vs_har_dow']['t'])}) in the holdout. None of these is "
             f"significant. The holdout XGB gain is broad, {wins_sym} of {n_sym} symbols, but it rests on {r(ho, 'xgb')['n_dates']} "
             f"dates, and holdout weekend days alone show {_p(n1[ho]['by_weekend']['1.0']['xgb']['rel'])} "
             f"(t {_t(n1[ho]['by_weekend']['1.0']['xgb']['t'])}).\n")

    L.append("## Methods\n")
    p = out["panel"]
    L.append(f"**Panel.** `ai_trader.ml.features`, universe L ({p['symbols']} crypto perps), one row per symbol and UTC day D. "
             "Decision time T = 00:00 UTC of D+1. There are 52 point-in-time features:\n")
    L.append("- the fixed H-IC1 list of 32;")
    L.append("- the log-HAR inputs (log RV_D, 7d / 30d mean log RV, as `vol_panel.design('loghar')`) and the 90d mean log RV;")
    L.append("- daily downside share, log(sqrt RQ / RV), daily jump share, log(Parkinson / RV), log(GK / RV), |ret_1d_z|;")
    L.append("- the forecast day's weekday (sin/cos), TradFi-weekend hours in D+1, a Deribit monthly-expiry flag;")
    L.append("- the v0 regime (direction, vol state, score) at the 20:00 4h bar;")
    L.append("- the crypto vol factor and the mean cross-sectional return.\n")
    L.append(f"Only complete rows are used, so every model sees an identical sample: {p['rows']:,} rows from {p['first_T'][:10]} to "
             f"{p['last_T'][:10]} ({p['rows_by_segment'][ins]:,} llm_in_sample, {p['rows_by_segment'][ho]:,} clean_holdout). "
             "A unit test checks causality: appending future bars, funding or OI leaves past rows bit-identical. Standardisation "
             "uses fit-block statistics only. Concept groups for attribution: "
             + "; ".join(f"{g}: {', '.join(fs)}" for g, fs in out["groups"].items()) + ".\n")
    w = out["walk"]
    L.append(f"**Walk-forward.** Monthly and expanding, {w['first_test_month']} to 2026-11 ({w['n_months']} refits, "
             f"{w['secs_total'] / 60:.0f} CPU-minutes). Training rows have label_end <= month - 1 day (24h purge plus 1-day "
             "embargo). Inside each window:\n")
    L.append("- the last 15% of dates form a purged validation block;")
    L.append("- both families stop early on that block (MLP: patience 10, max 200 epochs; XGB: 50 rounds, max 2000);")
    L.append("- the variance bias correction exp(f + c), with c = log mean(RV / exp(f)), is estimated on that block.\n")
    L.append("**Models.**\n")
    L.append("- MLP: 2 ReLU layers, AdamW lr 1e-3, batch 512, MSE on standardised log RV or BCE, 3-seed ensemble.")
    L.append("- XGB: hist, eta 0.03, subsample and colsample 0.8.")
    L.append(f"- Equal tuning effort: 4 configs each, chosen only on rows whose labels end before {w['first_test_month']} "
             "(diary entries 'ML design fixed before evaluation' and 'walk-forward started'). Chosen: "
             f"vol MLP {tuple(w['cfg']['mlp_vol']['hidden'])}, wd {w['cfg']['mlp_vol']['weight_decay']}; "
             f"vol XGB depth {w['cfg']['xgb_vol']['max_depth']}, mcw {w['cfg']['xgb_vol']['min_child_weight']}; "
             f"direction MLP {tuple(w['cfg']['mlp_dir']['hidden'])}; direction XGB depth {w['cfg']['xgb_dir']['max_depth']}. "
             f"Mean early-stopped size: vol {w['mean_epochs_vol']:.0f} epochs / {w['mean_xgb_iter_vol']:.0f} trees.")
    L.append("- Baseline: per-asset log-HAR (OLS on the symbol's own rows, at least 180 of them, exp(xb + s^2/2), R-VOL-20), "
             "with the same months and purge.")
    L.append("- References: the same HAR with an in-sample QLIKE-optimal scale, and ridge-linear on the same 52 features.\n")
    L.append("**Tests.**\n")
    L.append("- H-NN1: QLIKE on RV_{D+1}, paired difference averaged over symbols per date, NW t.")
    L.append("- H-NN2: Brier skill vs the training base rate, with a 10-day block bootstrap.")
    L.append("- H-NN3: xgboost exact TreeSHAP interaction values on each month's own out-of-sample rows (bias excluded), "
             "10-day block bootstrap over dates.")
    L.append(f"- H-NN4: frozen MLP = seed-0 member of the {it['frozen']['month']} vol ensemble (labels to {it['frozen']['train_last_label_end'][:10]}).")
    L.append(f"  - SAE: untied, dictionary {it['sae']['dict_size']} = 4 x {it['sae']['width']}, unit-norm decoder, pre-encoder bias, "
             "dead-feature resampling.")
    L.append(f"  - L1 = the largest of 0.03 / 0.1 / 0.3 / 1.0 with FVU <= 0.10 on the last 20% of in-sample dates, which selected {it['sae']['l1']}.")
    L.append("  - The 10 most-used features are ranked by in-sample firing frequency, then correlated (Spearman) with the "
             "named states on holdout activations.")
    L.append("  - Ablation is error-preserving: H - sum z_k a_k is passed through the frozen head.")
    L.append("- Holm within NN, per segment. Full specification in the study docstring.\n")
    L.append("## Results\n")
    L.append(tables)

    L.append("\n## Interpretability findings\n")
    L.append(f"1. **Trees: interactions are real, stable and mostly 'level x context'.** A third of the absolute TreeSHAP attribution is "
             f"pairwise interaction, and the top-5 pairs are identical in both segments. About {100 * n3[ins]['group_within_interaction_share']:.0f}% "
             "of |phi| is interaction *within* the vol-level group: log_sigma x lrv_w and log_sigma x lrv_q are two estimates of the same "
             "level splitting credit, which is a collinearity artefact rather than knowledge. The economically meaningful interactions are:")
    L.append(f"   - **vol level x calendar** ({100 * dict(((a, b), c) for a, b, c in n3[ins]['top_group_pairs']).get(('vol_level', 'calendar'), float('nan')):.1f}% "
             "of |phi| in-sample). The weekend discount is smaller when vol is already high: the TreeSHAP weekend gap is "
             f"{gin['low']['gap']:+.2f} / {gin['normal']['gap']:+.2f} / {gin['high']['gap']:+.2f} log points in low / normal / high v0 vol "
             f"in-sample, and {gho['low']['gap']:+.2f} / {gho['normal']['gap']:+.2f} / {gho['high']['gap']:+.2f} in the holdout.")
    dp = out["dependence"]
    L.append(f"   - **BTC x market return**. Their joint contribution rises with the size of the market move (rho "
             f"{dp['vol']['llm_in_sample_market_abs']:+.2f} / {dp['vol']['clean_holdout_market_abs']:+.2f} with |mkt_ret_1d_z|). "
             "The trees build an absolute value out of two signed inputs.")
    L.append("   - **vol level x trend regime**. Up-trends raise the forecast (v0_dir_score sign +0.50 / +0.61).")
    L.append("   Descriptive and not yet registered: market-wide *down* days raise the forecast (mkt_ret_1d_z sign -0.44 / -0.75), and a "
             "day dominated by one big hour lowers it (rq_ratio_d -0.93 / -0.96, the HAR-Q attenuation, R-VOL-23).")
    L.append(f"2. **Networks: no monosemantic state features.** The 16-unit last layer of the vol MLP is a distributed code for vol level "
             f"and weekend. The SAE (FVU {it['sae']['metrics_ho']['fvu']:.3f}, L0 {it['sae']['metrics_ho']['l0']:.1f} of 64) splits it into "
             "graded mixtures, not named concepts: only feature 4 (a 'low vol' detector, rho -0.75 with lrv_w) clears |rho| > 0.6. It is also "
             f"the only feature whose ablation moves QLIKE materially (holdout {it['ablation'][ho]['each']['4']['mean']:+.3f}, "
             f"t {_t(it['ablation'][ho]['each']['4']['t'])}; the top-3 ablation is significant in-sample, t {_t(it['ablation'][ins]['top3']['t'])}); "
             "ablating the others individually changes QLIKE by less than 0.01. The probes explain why. The first hidden layer still "
             f"encodes vol level (R^2 {it['probes']['hidden1']['vol_level']:.2f}) and weekend ({it['probes']['hidden1']['weekend']:.2f}). "
             f"The last layer keeps vol level ({it['probes']['hidden2']['vol_level']:.2f}) but has largely discarded vol-of-vol "
             f"({it['probes']['hidden2']['vol_of_vol']:.2f}), trend ({it['probes']['hidden2']['trend_strength']:.2f}) and funding "
             f"({it['probes']['hidden2']['funding_z']:.2f}): it throws away what does not help the vol forecast. With so few concepts in "
             "play there is little superposition for dictionary learning to undo, and a 4x dictionary over-splits the vol-level direction.")
    L.append(f"3. **Integrated gradients and TreeSHAP agree for vol** (feature-ranking Spearman {at['vol'][ho]['spearman_feature_ranking']:.2f}, "
             f"group Spearman {at['vol'][ho]['group_spearman']:.2f} in the holdout). The MLP spreads more weight onto trend-regime inputs "
             f"({100 * at['vol'][ho]['groups']['trend_regime']['mlp_ig']:.0f}% vs {100 * at['vol'][ho]['groups']['trend_regime']['xgb_shap']:.0f}%), "
             "but ablating that group costs little in either model.")
    L.append("4. **Group ablation (causal)**, relative QLIKE increase when a group is set to its training mean:")
    L.append("   (XGB in-sample / holdout; MLP in-sample / holdout; NW t in the table above)")
    for g, lab in (("vol_level", "vol level"), ("calendar", "calendar"), ("trend_regime", "trend regime"), ("funding_oi", "funding / OI"),
                   ("market", "market factor"), ("jumps_semivar", "jumps / semivariance"), ("range_measures", "range measures"), ("returns", "returns")):
        L.append(f"   - {lab}: {_p(gab('vol', ins, 'xgb', g))} / {_p(gab('vol', ho, 'xgb', g))}; {_p(gab('vol', ins, 'mlp', g))} / {_p(gab('vol', ho, 'mlp', g))}.")
    L.append("   For direction, no group moves Brier skill by more than 0.005.\n")

    L.append("## What to call when (for the decision layer)\n")
    L.append("| target | call first | then | marginal (tie-breakers) | noise (do not call for this target) | nonlinear model adds over log-HAR |")
    L.append("|---|---|---|---|---|---|")
    L.append(f"| next-day variance (crypto perp, 1d) | `vol.har` per-asset log-HAR: 7d mean log RV and robust sigma, about half of all attribution | "
             f"forecast-day weekday: Saturday about {_mult(cal_in['Sat'] - mean_in)}, Monday about {_mult(cal_in['Mon'] - mean_in)} of the "
             "average day; the discount is smaller in high vol | trend regime (v0 dir score, up-trend raises vol), funding z / OI, market "
             "factor and the size of the market move; 1-3% QLIKE each, not robustly significant | jump variation, semivariances, "
             "Parkinson / GK range, raw vol-scaled returns, expiry flag, vol-of-vol on its own | "
             f"in-sample {_p(r(ins, 'mlp')['rel'])} (MLP), holdout {_p(r(ho, 'xgb')['rel'])} (XGB); after adding weekday dummies to HAR, "
             "the remainder is not significant |")
    L.append(f"| next-day direction (sign of next-24h return) | nothing: no tool forecasts it | none | none | every standard feature, linear or "
             f"nonlinear (MLP / XGB / linear Brier skill <= 0 in-sample, holdout CIs include 0, families disagree) | none "
             f"(skill {n2[ins]['xgb']['skill']:+.4f} in-sample) |")
    L.append("\nOrder for the decision layer: (1) the vol level from `vol.har`; (2) the calendar adjustment for the forecast day "
             "(the hour-of-week profile R-VOL-26/27 at intraday horizons, the weekday effect here at 1d); (3) at most one context "
             "line (trend regime or funding crowding) as a tie-breaker. A nonlinear forecaster is worth deploying only as a "
             "*central* next-day expectation tool, after a forward test, never for risk or stops (weekends are jump-prone, R-VOL-2). "
             "For direction the budget rule R-BUDGET-1 stands: spending inputs or nonlinear capacity on it adds noise.\n")

    L.append("## Caveats\n")
    L.append(f"- The holdout is {r(ho, 'xgb')['n_dates']} dates (2026-07 to 2026-11). The holdout XGB win (t {_t(r(ho, 'xgb')['t'])}) "
             f"passes Holm only at {out['holm'][ho]['p_holm']['H-NN1']:.3f}, and the in-sample winner (MLP) does not win in the holdout. "
             "The family-level conclusion (nonlinear beats plain log-HAR) is stable; the choice of family is not.")
    L.append("- The log-HAR baseline is evaluated on the complete-case sample (rows where all 52 features exist), which is about 10% "
             "smaller than the vol tournament's; the HAR fit itself uses all of the symbol's rows.")
    L.append(f"- The frozen single MLP used for H-NN4 has holdout QLIKE {it['frozen']['holdout_qlike']:.3f}, better than the walk-forward "
             f"MLP ensemble ({n1[ho]['mean_qlike']['mlp']:.3f}): month-to-month refit noise in the MLP is large.")
    L.append("- TreeSHAP interaction shares among collinear vol-level proxies are partly an artefact of how trees share credit "
             "between correlated inputs. The cross-group interactions, the signs and the ablations are the robust part.")
    L.append("- The har_dow comparison, the calendar-by-vol table and the effect signs are exploratory, added after the H-NN1 "
             "result was seen. They inform rule candidates at grade C and should be registered before being relied on.")
    L.append("- H-NN5 / H-NN6 (1 s micro data, CNN) are not run in this round.\n")

    L.append("## Rule candidates\n")
    L.append("```yaml")
    L.append(RULES.strip())
    L.append("```\n")
    return "\n".join(L)


RULES = """
- id: R-ML-1
  kind: prefer
  rule: WHEN forecasting next-day variance for a crypto perp THEN start from the per-asset log-HAR level and adjust it for the forecast day's weekday before reading any other input
  when: daily (1d) variance expectation for a universe-L perp
  then: central forecast = log-HAR x weekday factor (Saturday about 0.6x, Sunday about 0.85x, Monday about 1.3x of the model's average day); risk and stops keep the unshrunk risk sigma (R-VOL-2)
  scope:
    assets: crypto perps (universe L)
    horizon: 1d
    sides: both
  effect: vol level carries 40-43% of TreeSHAP attribution and ablating it raises QLIKE 75%; calendar ablation +5-8% QLIKE; per-asset log-HAR + weekday dummies -5.5% QLIKE (t -2.9) in-sample, -2.3% (t -0.6) holdout vs plain log-HAR (exploratory)
  confidence: B
  evidence: [H-NN1, H-NN3, H-VM5, H-NS4]
  review: register and test log-HAR + weekday as a tool (vol.har_dow); demote if the holdout weekday gain stays insignificant over 2 more months
  status: provisional
- id: R-ML-2
  kind: expect
  rule: WHEN vol is already high (v0 high-vol state) THEN expect a smaller weekend discount in next-day variance than in calm markets
  when: the forecast day is Saturday or Sunday
  then: weekend gap about -0.39 log points in high vol vs -0.48 in low vol (in-sample), -0.48 vs -0.57 (holdout); do not apply a calm-market weekend discount in a high-vol regime
  scope:
    assets: crypto perps (universe L)
    horizon: 1d
    sides: both
  effect: TreeSHAP vol_level x calendar is the largest cross-group interaction (5.2% / 4.4% of |phi|), the same direction in both segments
  confidence: C
  evidence: [H-NN3]
  review: register a direct test (weekend RV ratio by vol tercile, both segments) before promoting
  status: provisional
- id: R-ML-3
  kind: avoid
  rule: WHEN forecasting next-day crypto variance THEN don't call jump-variation, semivariance, range-estimator (Parkinson / GK) or raw-return inputs on top of the HAR level; they carry no information a nonlinear model can use either
  when: daily variance forecast with the HAR level already known
  then: skip those tools for this target; they remain useful descriptively (for example jumps for stop placement)
  scope:
    assets: crypto perps (universe L)
    horizon: 1d
    sides: both
  effect: group ablation in MLP and XGB changes QLIKE by -0.7% to +1.5%, none with |t| > 2 in the holdout; TreeSHAP shares 0.5-2%
  confidence: B
  evidence: [H-NN1, H-NN3, H-VCL2, H-VM3]
  review: a holdout quarter where any of these groups raises QLIKE by more than 3% (t > 2) when ablated
  status: active
- id: R-ML-4
  kind: prefer
  rule: WHEN a central next-day variance expectation is needed for a crypto perp and a validated nonlinear tool exists THEN prefer gradient-boosted trees on the 52-feature panel over plain log-HAR, but only for central expectations, never for risk or stops
  when: next-day expectation (quote widths, target distances); not risk
  then: use the tree forecast with its in-sample bias correction; fall back to R-ML-1 if the tool is stale (no refit for more than 35 days)
  scope:
    assets: crypto perps (universe L)
    horizon: 1d
    sides: both
  effect: XGB holdout QLIKE -15.7% vs log-HAR (t -2.4, 22/23 symbols), but in-sample -4.5% (t -1.0) and only -13.7% (t -1.6) vs log-HAR + weekday; the MLP wins in-sample (-8.0%, t -2.9) and not in the holdout
  confidence: C
  evidence: [H-NN1]
  review: forward test against log-HAR + weekday for 3 months after the tool exists; demote if QLIKE gain < 3% or t > -2
  status: provisional
- id: R-ML-5
  kind: avoid
  rule: WHEN looking for next-day direction THEN don't call a neural network or boosted-tree model on standard price, vol, regime, funding or calendar features; nonlinear capacity doesn't create a signal that linear models miss
  when: next-24h direction for crypto perps
  then: treat the direction as a coin flip; any direction view must come from a separately validated, registered signal
  scope:
    assets: crypto perps (universe L)
    horizon: 1d
    sides: both
  effect: Brier skill in-sample MLP -0.0024 [-0.0072, 0.0021], XGB -0.0016 [-0.0055, 0.0022]; holdout +0.0054 / +0.0018 with CIs including 0; models stop after about 1.5 epochs / 24 trees; tree and network attributions disagree (Spearman 0.22 holdout)
  confidence: B
  evidence: [H-NN2, H-IC1]
  review: a pre-registered nonlinear direction model with holdout Brier skill CI lower bound > 0
  status: active
- id: R-ML-6
  kind: prefer
  rule: WHEN explaining a model-based volatility forecast to the decision layer THEN cite grouped TreeSHAP attributions (vol level, calendar, context) rather than hidden-unit or dictionary features of a neural network
  when: a nonlinear vol forecast is shown with an explanation
  then: report the vol-level, weekday and context contributions in log-RV units; don't name network features
  scope:
    assets: crypto perps (universe L)
    horizon: 1d
    sides: both
  effect: SAE on the MLP's last layer, only 1/10 most-used features tie to a named state (|rho| > 0.6), top-3 ablation +0.09 QLIKE (t 1.3, holdout); TreeSHAP groups are stable across segments (top-5 pairs 5/5) and agree with integrated gradients (Spearman 0.75)
  confidence: B
  evidence: [H-NN4, H-NN3]
  review: an SAE on a wider or multi-day micro model (H-NN5) that does yield >= 6/10 named features
  status: active
"""
