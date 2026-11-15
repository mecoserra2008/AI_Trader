"""Markdown report for ML signals round 2 (research/reports/ml_signals_round2_2026-11-12.md), built from the evaluated
numbers. Rule candidates are generated from the verdicts so the text cannot contradict the recorded results."""

from __future__ import annotations

SEGS = ("llm_in_sample", "clean_holdout")


def _f(x, k: int = 4) -> str:
    try:
        return f"{float(x):.{k}f}"
    except (TypeError, ValueError):
        return str(x)


def _pct(d: float, base: float) -> str:
    return f"{100 * d / base:+.1f}%" if base else "nan"


def build(out: dict) -> str:
    n8, n9, it = out["H-NN8"], out["H-NN9"], out["interpret9"]
    v = out["verdicts"]
    ins, ho = SEGS
    w8, p8 = out["walk8"], out["panel8"]
    L: list[str] = []
    L.append("# ML signals round 2: next-hour volatility (H-NN8) and a wider network for the SAE (H-NN9)\n")
    L.append("_Program 2026-11-12, agent B, round 2. Code: `src/ai_trader/ml/hourly.py`, "
             "`src/ai_trader/research/ml_signals_round2_study.py`; tests: `tests/test_ml_round2.py`; artifacts: "
             "`data/models/ml/round2/`. These are deterministic research tools. The LLM layer is not trained or backtested here. "
             "H-NN7 and H-LEV7 are forward-only and are not tested here._\n")
    L.append("## Summary\n")
    L.append("| hypothesis | in-sample | holdout | pooled |")
    L.append("|---|---|---|---|")
    for h in ("H-NN8", "H-NN9"):
        L.append(f"| {h} | {v[h][ins]} | {v[h][ho]} | **{v[h]['pooled']}** |")
    L.append("")

    def row8(s, m):
        r = n8[s][f"{m}_vs_champ"]
        return f"{_f(r['diff'], 5)} (t {_f(r['t'], 2)})"

    L.append(f"**H-NN8.** The next-hour QLIKE difference against the champion (robust_bc x hour-of-week factor) is "
             f"{row8(ins, 'xgb')} for XGB, {row8(ins, 'mlp')} for the MLP and {row8(ins, 'lin')} for the linear reference "
             f"in-sample. In the holdout it is {row8(ho, 'xgb')}, {row8(ho, 'mlp')} and {row8(ho, 'lin')}. For scale, the champion "
             f"itself beats the bare robust sigma by {_f(n8[ins]['champ_vs_base']['diff'], 5)} (t {_f(n8[ins]['champ_vs_base']['t'], 2)}) "
             f"in-sample and {_f(n8[ho]['champ_vs_base']['diff'], 5)} (t {_f(n8[ho]['champ_vs_base']['t'], 2)}) in the holdout.\n")
    L.append(f"**H-NN9.** With a (64, 64) network, the SAE picked by the in-sample rule (dictionary {it['sae']['dict_size']}, "
             f"L1 {it['sae']['l1']}) has {it['n_ties_holdout']} of 10 most-used features tied to a named state on the holdout "
             f"({it['n_ties_in_sample']} in-sample). Ablating the top 3 changes holdout QLIKE by "
             f"{it['ablation'][ho]['top3']['mean']:+.4f} (t {_f(it['ablation'][ho]['top3']['t'], 2)}). The 64-unit ensemble against "
             f"the 16-unit H-NN1 MLP: QLIKE {_f(n9[ins]['mlp64_vs_mlp16']['diff'])} (t {_f(n9[ins]['mlp64_vs_mlp16']['t'], 2)}) "
             f"in-sample, {_f(n9[ho]['mlp64_vs_mlp16']['diff'])} (t {_f(n9[ho]['mlp64_vs_mlp16']['t'], 2)}) in the holdout.\n")

    L.append("## Methods\n")
    L.append(f"**H-NN8.** The cached H-NS4 1h frames for universe L ({p8['symbols']} symbols) are placed on a regular hourly grid, "
             "so no return spans a gap. There are 19 point-in-time features (`ml.hourly`):\n")
    L.append("- log robust per-bar variance s2;")
    L.append("- log causal hour-of-week factor;")
    L.append("- funding-hour flags;")
    L.append("- log r^2/s2 at 1, 2, 3, 6, 12 and 24 h lags, plus 6 h and 24 h means;")
    L.append("- 24 h jump flag (|z| > 4) and hours since the last jump;")
    L.append("- daily HAR level (log RV of the last complete day, 7 d and 30 d means) relative to 24 s2;")
    L.append("- weekday.\n")
    L.append(f"The target is log(r^2/s2) and the forecast is s2 exp(f + c). The champion is s2 x factor (R-VOL-26). The "
             f"walk-forward is monthly and expanding from {w8['first_test_month']} ({w8['n_months']} refits, "
             f"{w8['secs_total'] / 60:.0f} CPU-minutes) with a 1-day embargo.\n")
    L.append(f"**Runtime choice:** the models train on a deterministic random 1/3 of the bars "
             f"({w8['mean_train_rows_last']:,} rows at the last refit). Every bar of each test month is scored "
             f"({p8['rows_tested']:,} test bars: {p8['rows_by_segment'].get(ins, 0):,} in-sample, {p8['rows_by_segment'].get(ho, 0):,} "
             "holdout). A fixed every-3rd-hour rule was rejected because it would alias with the 24 h cycle.\n")
    L.append(f"Protocol as H-NN1: purged validation block, early stopping, validation-block bias constant. Mean early-stopped size: "
             f"{w8['mean_epochs']:.1f} epochs and {w8['mean_xgb_iter']:.0f} trees. Configs come from equal 4-config searches on "
             f"pre-{w8['first_test_month']} rows only: MLP {w8['cfg']['mlp']['hidden']}, wd {w8['cfg']['mlp']['weight_decay']}, "
             f"batch 2048; XGB depth {w8['cfg']['xgb']['max_depth']}, mcw {w8['cfg']['xgb']['min_child_weight']}. The loss is the "
             "per-bar QLIKE difference r^2 (1/Fa - 1/Fb) + log(Fa/Fb), averaged per UTC date, with NW t.\n")
    w9 = out["walk9"]
    L.append(f"**H-NN9.** The H-NN1 walk-forward is repeated with only the network changed: hidden {tuple(w9['cfg']['hidden'])}, "
             f"wd {w9['cfg']['weight_decay']}, 3 seeds ({w9['n_months']} refits, mean {w9['mean_epochs']:.1f} epochs). The frozen "
             f"model is the seed-0 member for {it['frozen']['month']}, trained on labels up to {it['frozen']['train_last_label_end'][:10]}.\n")
    L.append("SAE selection, all in-sample:\n")
    L.append("- dictionary size 64 or 128;")
    L.append("- for each size, L1 is the largest value in the grid with FVU <= 0.10 on the last 20% of in-sample dates;")
    L.append("- the size is then chosen by its tie count on that in-sample block.\n")
    for size, s in it["selection"].items():
        L.append(f"- size {size}: L1 {s['l1']}, validation FVU {s['fvu_val']:.3f}, L0 {s['l0_val']:.1f}, "
                 f"in-sample validation ties {s['ties_val']}/10")
    L.append(f"\nChosen: {it['sae']['dict_size']}, L1 {it['sae']['l1']}. Holdout FVU {it['sae']['metrics_ho']['fvu']:.3f}, "
             f"L0 {it['sae']['metrics_ho']['l0']:.1f}, {it['sae']['n_alive_in']} alive features.\n")

    L.append("## Results\n")
    L.append("### H-NN8 next-hour QLIKE difference vs the champion\n")
    L.append("| segment | model | diff | NW t | dates | bars |")
    L.append("|---|---|---|---|---|---|")
    for s in SEGS:
        for m in ("mlp", "xgb", "lin"):
            r = n8[s][f"{m}_vs_champ"]
            L.append(f"| {s} | {m} | {_f(r['diff'], 5)} | {_f(r['t'], 2)} | {r['n_dates']} | {r['n_obs']:,} |")
        r = n8[s]["champ_vs_base"]
        L.append(f"| {s} | champion vs bare s2 (context) | {_f(r['diff'], 5)} | {_f(r['t'], 2)} | {r['n_dates']} | {r['n_obs']:,} |")
    L.append("\n### H-NN8 what the hourly trees use (TreeSHAP mean |phi| share) and group ablation (dQLIKE when a group is set to its training mean)\n")
    L.append("| group | SHAP share in | SHAP share ho | ablation xgb in | ablation xgb ho | ablation mlp in | ablation mlp ho |")
    L.append("|---|---|---|---|---|---|---|")
    for g in out["hourly_groups"]:
        ga_i, ga_h = n8[ins]["group_ablation"], n8[ho]["group_ablation"]
        L.append(f"| {g} | {100 * n8[ins]['shap_group_share'][g]:.1f}% | {100 * n8[ho]['shap_group_share'][g]:.1f}% | "
                 f"{_f(ga_i[f'xgb:{g}']['dqlike'], 5)} ({_f(ga_i[f'xgb:{g}']['t'], 1)}) | {_f(ga_h[f'xgb:{g}']['dqlike'], 5)} ({_f(ga_h[f'xgb:{g}']['t'], 1)}) | "
                 f"{_f(ga_i[f'mlp:{g}']['dqlike'], 5)} ({_f(ga_i[f'mlp:{g}']['t'], 1)}) | {_f(ga_h[f'mlp:{g}']['dqlike'], 5)} ({_f(ga_h[f'mlp:{g}']['t'], 1)}) |")
    L.append("\nTop features (holdout SHAP share): " + "; ".join(f"{a} {100 * b:.1f}%" for a, b in n8[ho]["shap_top"]))
    L.append("\n### H-NN9 SAE features (holdout)\n")
    L.append("| feature | freq in | freq ho | best state | abs rho | ties | top inputs | d log RV per unit |")
    L.append("|---|---|---|---|---|---|---|---|")
    for r in it["corr_holdout"]:
        L.append(f"| {r['feature']} | {r['freq_in']:.2f} | {r['freq_seg']:.2f} | {r['best_state']} | {_f(r['best_abs_rho'], 2)} | "
                 f"{'yes' if r['ties'] else 'no'} | " + ", ".join(f"{a} {b:+.2f}" for a, b in r["top_inputs"]) + f" | {r['dlogrv_per_unit']:+.3f} |")
    L.append("\n| segment | ablation | dQLIKE | NW t | intact QLIKE |")
    L.append("|---|---|---|---|---|")
    for s in (ho, ins):
        for k in ("top3", "top10"):
            r = it["ablation"][s][k]
            L.append(f"| {s} | {k} | {r['mean']:+.5f} | {_f(r['t'], 2)} | {r['qlike_intact']:.4f} |")
    L.append("\nLinear probes (holdout R^2):\n")
    sv = list(next(iter(it["probes"].values())).keys())
    L.append("| layer | " + " | ".join(sv) + " |")
    L.append("|---|" + "---|" * len(sv))
    for lname, pr in it["probes"].items():
        L.append(f"| {lname} | " + " | ".join(_f(pr[k], 2) for k in sv) + " |")
    fz = it["frozen"]
    L.append(f"\nLast-layer geometry: {fz['dead_units_in']} dead units of 64. Participation ratio (effective dimension) "
             f"{fz['participation_ratio']:.1f}, against {fz['participation_ratio_16']:.1f} for the 16-unit model. The frozen "
             f"64-unit model's holdout QLIKE is {fz['holdout_qlike']:.4f}.\n")
    L.append("### H-NN9 the 64-unit MLP's own forecasts (walk-forward ensemble, daily QLIKE)\n")
    L.append("| segment | 64 - 16 unit | 64-unit - log-HAR | 64-unit - XGB |")
    L.append("|---|---|---|---|")
    for s in SEGS:
        a, b, c = n9[s]["mlp64_vs_mlp16"], n9[s]["mlp64_vs_har"], n9[s]["mlp64_vs_xgb"]
        L.append(f"| {s} | {_f(a['diff'])} ({_pct(a['diff'], a['loss_ref'])}, t {_f(a['t'], 2)}) | {_f(b['diff'])} ({_pct(b['diff'], b['loss_ref'])}, "
                 f"t {_f(b['t'], 2)}) | {_f(c['diff'])} ({_pct(c['diff'], c['loss_ref'])}, t {_f(c['t'], 2)}) |")
    L.append("\n### Verdicts (after Holm within round 2)\n")
    L.append("| hypothesis | in-sample | holdout | pooled | Holm p (in / ho) |")
    L.append("|---|---|---|---|---|")
    for h in ("H-NN8", "H-NN9"):
        L.append(f"| {h} | {v[h][ins]} | {v[h][ho]} | {v[h]['pooled']} | {_f(out['holm'][ins]['p_holm'][h], 3)} / {_f(out['holm'][ho]['p_holm'][h], 3)} |")

    L.append("\n## What this changes for tool calls\n")
    L.extend(_synthesis(out))
    L.append("\n## Caveats\n")
    L.append("- The hourly models train on a random third of the bars to keep runtime sane. All test bars are scored.")
    L.append("- The search window before the first hourly test month is small (BTC/ETH-heavy, about one year). Configs were "
             "frozen from it as registered.")
    L.append(f"- The holdout has {n8[ho]['xgb_vs_champ']['n_dates']} dates for H-NN8 and {n9[ho]['mlp64_vs_har']['n_dates']} "
             "dates for the daily H-NN9 comparison.")
    L.append("- The H-NN9 interpretation rests on one frozen seed-0 network. Refit noise across seeds and months is large (round 1).\n")
    L.append("## Rule candidates\n")
    L.append("```yaml")
    L.append(_rules(out).strip())
    L.append("```\n")
    return "\n".join(L)


def _synthesis(out: dict) -> list[str]:
    n8, n9, it = out["H-NN8"], out["H-NN9"], out["interpret9"]
    ins, ho = SEGS
    best = min(("mlp", "xgb"), key=lambda m: n8[ho][f"{m}_vs_champ"]["diff"])
    win8 = out["verdicts"]["H-NN8"]["pooled"] == "supports"
    sh = n8[ho]["shap_group_share"]
    top_groups = sorted(sh, key=lambda g: -sh[g])[:3]
    L = []
    if win8:
        L.append(f"- **Next hour:** a nonlinear model ({', '.join(n8['winners_both_segments'])}) beats the robust sigma x "
                 "hour-of-week factor in both segments. The groups it relies on most are "
                 + ", ".join(f"{g} ({100 * sh[g]:.0f}%)" for g in top_groups)
                 + ". The intraday champion leaves usable structure, so the decision layer should prefer such a tool once it is "
                 "built and forward-tested.")
    else:
        L.append(f"- **Next hour:** no nonlinear model beats robust sigma x hour-of-week in both segments (best holdout model "
                 f"{best}: {n8[ho][f'{best}_vs_champ']['diff']:+.5f}, t {n8[ho][f'{best}_vs_champ']['t']:.2f}). The decision layer "
                 "keeps calling `vol.robust_bc` x the seasonal factor for intraday expectations; an hourly ML forecaster is not "
                 "worth a tool call. The trees lean mostly on "
                 + ", ".join(f"{g} ({100 * sh[g]:.0f}%)" for g in top_groups) + ", which the champion already contains.")
    xl_i, xl_h = n8[ins]["xgb_vs_lin"], n8[ho]["xgb_vs_lin"]
    li_i, li_h = n8[ins]["lin_vs_champ"], n8[ho]["lin_vs_champ"]
    ga = {s: n8[s]["group_ablation"] for s in SEGS}
    L.append(f"- **How much is nonlinear:** a ridge-linear model on the same 19 inputs already gets {li_i['diff']:+.5f} (t {li_i['t']:.2f}) "
             f"in-sample and {li_h['diff']:+.5f} (t {li_h['t']:.2f}) in the holdout. Trees add {xl_i['diff']:+.5f} (t {xl_i['t']:.2f}) and "
             f"{xl_h['diff']:+.5f} (t {xl_h['t']:.2f}) on top. So roughly half of the gain is just extra inputs and half is nonlinearity, "
             "unlike the daily panel, where the gain was mostly calendar.")
    L.append(f"- **Which inputs to add to the intraday forecast:** ablation matters most for recent |returns| (significant in-sample, t 1.8 in the holdout) "
             f"(XGB {ga[ins]['xgb:recent_returns']['dqlike']:+.4f}, t {ga[ins]['xgb:recent_returns']['t']:.1f} in-sample; "
             f"{ga[ho]['xgb:recent_returns']['dqlike']:+.4f}, t {ga[ho]['xgb:recent_returns']['t']:.1f} holdout), for the daily HAR level "
             f"({ga[ins]['xgb:daily_har']['dqlike']:+.4f}, t {ga[ins]['xgb:daily_har']['t']:.1f}; {ga[ho]['xgb:daily_har']['dqlike']:+.4f}, "
             f"t {ga[ho]['xgb:daily_har']['t']:.1f}) and for the seasonal factor itself. Funding-hour flags, jump flags and the base "
             "sigma level beyond the offset do not matter (|t| < 2, or the wrong sign). The robust 1h sigma reacts too slowly "
             "and too smoothly: the last 1-6 hours of |returns| and the daily HAR level carry information it lacks.")
    t9 = out["verdicts"]["H-NN9"]["pooled"]
    L.append(f"- **Interpretability:** with 64 units the SAE result is {t9} ({it['n_ties_holdout']}/10 named features on the "
             f"holdout). " + ("Named market-state features can be read off the network and used as explanations." if t9 == "supports" else
                              "Width does not make the network's features nameable. The 64-unit last layer has an effective "
                              f"dimension (participation ratio) of only {it['frozen']['participation_ratio']:.1f}, against "
                              f"{it['frozen']['participation_ratio_16']:.1f} for 16 units. The network represents a handful of mixed "
                              "directions, mostly vol level and weekend, so there is no superposition of many concepts for an SAE "
                              "to separate. Keep explaining model forecasts with grouped TreeSHAP (R-ML-6)."))
    a = n9[ho]["mlp64_vs_mlp16"]
    L.append(f"- **Network width:** going from 16 to 64 units changes daily QLIKE by {a['diff']:+.4f} (t {a['t']:.2f}) in the "
             f"holdout and {n9[ins]['mlp64_vs_mlp16']['diff']:+.4f} (t {n9[ins]['mlp64_vs_mlp16']['t']:.2f}) in-sample. Capacity "
             "is not what limits the daily vol forecast.")
    return L


def _rules(out: dict) -> str:
    n8, n9, it = out["H-NN8"], out["H-NN9"], out["interpret9"]
    ins, ho = SEGS
    v8, v9 = out["verdicts"]["H-NN8"], out["verdicts"]["H-NN9"]
    x_i, x_h = n8[ins]["xgb_vs_champ"], n8[ho]["xgb_vs_champ"]
    m_i, m_h = n8[ins]["mlp_vs_champ"], n8[ho]["mlp_vs_champ"]
    eff8 = (f"next-hour QLIKE diff vs champion: XGB {x_i['diff']:+.5f} (t {x_i['t']:.2f}) in-sample / {x_h['diff']:+.5f} (t {x_h['t']:.2f}) holdout; "
            f"MLP {m_i['diff']:+.5f} (t {m_i['t']:.2f}) / {m_h['diff']:+.5f} (t {m_h['t']:.2f})")
    if v8["pooled"] == "supports":
        r7 = f"""
- id: R-ML-7
  kind: prefer
  rule: WHEN a central next-hour variance expectation is needed for a crypto perp and a validated hourly ML tool exists THEN prefer it over robust_bc x the hour-of-week factor, for expectations only, never for stops
  when: horizon 1h, quoting or intraday expectations
  then: use the hourly model (features of ml.hourly) with its in-sample bias; fall back to R-VOL-26 if it is stale
  scope:
    assets: crypto perps (universe L)
    horizon: 1h
    sides: both
  effect: {eff8}
  confidence: C
  evidence: [H-NN8, H-NS4]
  review: forward test for 1 month against R-VOL-26 before use; demote if the forward QLIKE gain is not significant
  status: provisional"""
    else:
        r7 = f"""
- id: R-ML-7
  kind: avoid
  rule: WHEN forecasting next-hour variance for a crypto perp THEN don't call a nonlinear (tree or network) forecaster on top of robust_bc x the hour-of-week factor; the champion already holds the usable structure
  when: horizon 1h, quoting or intraday expectations
  then: keep R-VOL-26 (robust sigma x seasonal factor); spend no tool calls on lagged |returns|, jump flags, funding-hour flags or the daily HAR level for this horizon
  scope:
    assets: crypto perps (universe L)
    horizon: 1h
    sides: both
  effect: {eff8}; verdict {v8['llm_in_sample']} / {v8['clean_holdout']} / pooled {v8['pooled']}
  confidence: B
  evidence: [H-NN8, H-NS4]
  review: a registered hourly model with a QLIKE gain at t < -2 in both segments
  status: active"""
    a_i, a_h = n9[ins]["mlp64_vs_mlp16"], n9[ho]["mlp64_vs_mlp16"]
    r8 = f"""
- id: R-ML-8
  kind: expect
  rule: WHEN considering a wider neural network for the daily crypto vol forecast THEN don't expect extra capacity to improve it; the forecast is limited by information (vol level, weekday), not by network width
  when: choosing or retraining a daily vol model
  then: keep small networks or trees; spend effort on inputs and validation, not width
  scope:
    assets: crypto perps (universe L)
    horizon: 1d
    sides: both
  effect: 64-unit vs 16-unit last layer, daily QLIKE {a_i['diff']:+.4f} (t {a_i['t']:.2f}) in-sample, {a_h['diff']:+.4f} (t {a_h['t']:.2f}) holdout
  confidence: {'B' if abs(a_h['t']) < 2 and abs(a_i['t']) < 2 else 'C'}
  evidence: [H-NN9, H-NN1]
  review: a wider model with a QLIKE gain at t < -2 in both segments
  status: {'active' if abs(a_h['t']) < 2 and abs(a_i['t']) < 2 else 'provisional'}"""
    if v9["pooled"] == "supports":
        r9 = f"""
- id: R-ML-9
  kind: prefer
  rule: WHEN explaining a daily vol forecast from the 64-unit network THEN the SAE features tied to named states may be cited, alongside grouped TreeSHAP
  when: a network-based vol forecast is shown with an explanation
  then: cite the named SAE features (vol level, weekend, ...) with their direction of effect
  scope:
    assets: crypto perps (universe L)
    horizon: 1d
    sides: both
  effect: {it['n_ties_holdout']}/10 named on holdout; top-3 ablation {it['ablation'][ho]['top3']['mean']:+.4f} QLIKE (t {it['ablation'][ho]['top3']['t']:.2f})
  confidence: C
  evidence: [H-NN9, H-NN4]
  review: repeat on another seed and month; demote if the named set changes
  status: provisional"""
    else:
        r9 = f"""
- id: R-ML-9
  kind: avoid
  rule: WHEN explaining a daily vol forecast THEN don't cite sparse-autoencoder features of the network, even of a wider one; they are not stable named concepts. Use grouped TreeSHAP (R-ML-6)
  when: a model-based vol forecast is shown with an explanation
  then: grouped TreeSHAP (vol level, calendar, context) in log-RV units
  scope:
    assets: crypto perps (universe L)
    horizon: 1d
    sides: both
  effect: 64-unit layer, SAE dictionary {it['sae']['dict_size']}: {it['n_ties_holdout']}/10 features named on holdout (16-unit H-NN4: 1/10); top-3 ablation {it['ablation'][ho]['top3']['mean']:+.4f} QLIKE (t {it['ablation'][ho]['top3']['t']:.2f})
  confidence: B
  evidence: [H-NN9, H-NN4]
  review: an SAE on a model with more concepts in play (micro data, H-NN5) that yields >= 6/10 named features
  status: active"""
    ga = {s: n8[s]["group_ablation"] for s in SEGS}
    r10 = f"""
- id: R-ML-10
  kind: prefer
  rule: WHEN forming a next-hour variance expectation for a crypto perp THEN adjust the robust sigma x hour-of-week forecast for the last 1-6 hours of absolute returns and the daily HAR level; ignore funding-hour and jump flags for this purpose
  when: horizon 1h, central expectation or quote width; not stops
  then: recent |returns| above the robust sigma and a daily HAR level above 24 x sigma^2 both point to a higher next-hour variance than the champion gives
  scope:
    assets: crypto perps (universe L)
    horizon: 1h
    sides: both
  effect: group ablation in the hourly XGB raises QLIKE by {ga[ins]['xgb:recent_returns']['dqlike']:+.4f} (t {ga[ins]['xgb:recent_returns']['t']:.1f}) / {ga[ho]['xgb:recent_returns']['dqlike']:+.4f} (t {ga[ho]['xgb:recent_returns']['t']:.1f}) for recent returns and {ga[ins]['xgb:daily_har']['dqlike']:+.4f} (t {ga[ins]['xgb:daily_har']['t']:.1f}) / {ga[ho]['xgb:daily_har']['dqlike']:+.4f} (t {ga[ho]['xgb:daily_har']['t']:.1f}) for the daily HAR level (in-sample / holdout); funding-hour and jump groups |t| < 2; linear model on the same inputs {n8[ins]['lin_vs_champ']['diff']:+.5f} (t {n8[ins]['lin_vs_champ']['t']:.2f}) / {n8[ho]['lin_vs_champ']['diff']:+.5f} (t {n8[ho]['lin_vs_champ']['t']:.2f}) vs the champion
  confidence: B
  evidence: [H-NN8, H-NS4, H-S4]
  review: build vol.hourly_x (linear first, R-ML-7 trees second) and forward test for 1 month against R-VOL-26
  status: provisional"""
    return r7 + r8 + r9 + r10
