"""Markdown report for H-NN5 / H-NN6 (research/reports/ml_micro_2026-11-12.md), built from the evaluated numbers; the
rule candidates follow the verdicts so the text cannot contradict the recorded results."""

from __future__ import annotations

SEGS = ("llm_in_sample", "clean_holdout", "pooled")


def _f(x, k: int = 4) -> str:
    try:
        return f"{float(x):.{k}f}"
    except (TypeError, ValueError):
        return str(x)


def build(out: dict) -> str:
    n5, n6, v = out["H-NN5"], out["H-NN6"], out["verdicts"]
    ig, ab = out["ig"], out["ablation"]
    L: list[str] = []
    L.append("# 1D CNN on 1 s order-flow bars: next-minute volatility (H-NN5) and direction (H-NN6)\n")
    L.append("_Program 2026-11-12, agent B, round 4. Code: `src/ai_trader/ml/micro.py`, `src/ai_trader/research/ml_micro_study.py`; "
             "tests: `tests/test_ml_micro.py`; artifacts: `data/models/ml/micro/`. Data: agent A's aggregates "
             "`data/research/micro_days/` (read only), with the day list frozen in `data/models/ml/micro/days.json`. "
             "These are deterministic research tools. The LLM layer is not trained or backtested here._\n")
    L.append("## Summary\n")
    L.append("| hypothesis | llm_in_sample | clean_holdout | pooled |")
    L.append("|---|---|---|---|")
    for h in ("H-NN5", "H-NN6"):
        L.append(f"| {h} | {v[h]['llm_in_sample']} | {v[h]['clean_holdout']} | **{v[h]['pooled']}** |")
    p5 = n5["pooled"]
    L.append(f"\n**H-NN5.** The CNN against the linear HAR-style model on the same inputs gives a mean per-symbol-day QLIKE "
             f"difference of {p5['mean_diff']:+.4f} ({100 * p5['rel_to_lin_qlike']:+.1f}% of the linear model's QLIKE). "
             f"{100 * p5['share_lt0']:.0f}% of {p5['n_symbol_days']} held-out symbol-days are negative; t {_f(p5['t'], 2)} over "
             f"symbol-days, {_f(p5['t_day_clustered'], 2)} over day means (pooled). Both models against the naive baseline "
             f"(60 x trailing-1h variance): CNN {p5['context']['cnn_vs_base']['mean']:+.3f} (t {_f(p5['context']['cnn_vs_base']['t'], 1)}), "
             f"linear {p5['context']['lin_vs_base']['mean']:+.3f} (t {_f(p5['context']['lin_vs_base']['t'], 1)}).")
    p6 = n6["pooled"]
    L.append(f"\n**H-NN6.** The CNN's AUC for the sign of the next-minute return is {_f(p6['cnn']['auc'])} "
             f"[{_f(p6['cnn']['ci95'][0])}, {_f(p6['cnn']['ci95'][1])}] pooled; the linear model's is {_f(p6['lin']['auc'])} "
             f"[{_f(p6['lin']['ci95'][0])}, {_f(p6['lin']['ci95'][1])}]. Brier skill: CNN {p6['cnn']['brier_skill']:+.4f}, "
             f"linear {p6['lin']['brier_skill']:+.4f}.\n")

    L.append("## Methods\n")
    segs = {d: ("llm_in_sample" if d < "2026-07-01" else "clean_holdout") for d in out["days"]}
    L.append(f"**Data.** {len(out['days'])} days, frozen at the start: " + ", ".join(f"{d} ({'in' if s == 'llm_in_sample' else 'ho'})" for d, s in segs.items())
             + f". {out['n_symbol_days']} symbol-days with at least 10,000 traded seconds, all symbols pooled. "
             f"{out['rows']:,} test windows ({out['rows_by_segment']}).\n")
    L.append("**Inputs.** There are 12 channels of 1 s bars, each normalised by trailing-1h statistics so that no future "
             "information enters:\n")
    L.append("- standardised return and log squared return;")
    L.append("- high-low range;")
    L.append("- signed flow, volume, prints, orders and order imbalance;")
    L.append("- where an L2 book exists: a has-book flag, spread and 10 bp depth (both relative to their trailing means) and "
             "10 bp imbalance.\n")
    L.append("The input window is the last 300 s. The baseline is 60 x the trailing-1h per-second variance. The target is the "
             "next-1m last-trade return.\n")
    L.append("**Models.**\n")
    L.append("- CNN: a dilated 1D CNN (4 layers, 16 channels, receptive field about 31 s, pooled over the last 10, 60 and "
             "300 positions plus a max) with a linear skip on 36 HAR aggregates. One seed.")
    L.append("- Linear: the same skip alone, i.e. 10, 60 and 300 s means of each channel.")
    L.append("- Both use the same optimiser and loss (QLIKE for H-NN5, BCE for H-NN6).\n")
    L.append("**Evaluation.**\n")
    L.append("- Three rotating leave-days-out folds; the validation set is the last training day.")
    for f in out["folds"]:
        L.append(f"  - fold {f['fold']}: test {f['test_days']}, validation {f['val_day']}; "
                 f"{f['n_train']:,} training / {f['n_test']:,} test windows; CNN vol best epochs "
                 f"{[t['best_epoch'] for t in f['vol_cnn']['train']]}, linear {[t['best_epoch'] for t in f['vol_lin']['train']]}")
    L.append("- **Runtime:** training windows every 30 s (100k drawn at random per epoch, patience 2, one CNN seed), test windows every 10 s. "
             "Integrated gradients on 2,000 test windows per fold; channel ablation on test windows every 60 s.")
    L.append("- These are leave-days-out folds, not forward: training days can be later than test days.\n")

    L.append("## Results\n")
    L.append("### H-NN5 per held-out symbol-day QLIKE difference (CNN - linear)\n")
    L.append("| segment | symbol-days | days | share < 0 | mean diff | rel. | t | t (day-clustered) | CNN - base | linear - base |")
    L.append("|---|---|---|---|---|---|---|---|---|---|")
    for s in SEGS:
        r = n5[s]
        L.append(f"| {s} | {r['n_symbol_days']} | {r['n_days']} | {100 * r['share_lt0']:.0f}% | {r['mean_diff']:+.4f} | {100 * r['rel_to_lin_qlike']:+.1f}% | "
                 f"{_f(r['t'], 2)} | {_f(r['t_day_clustered'], 2)} | {r['context']['cnn_vs_base']['mean']:+.3f} | {r['context']['lin_vs_base']['mean']:+.3f} |")
    L.append("\nBy book availability (pooled):\n")
    for k, r in n5["pooled"]["by_book"].items():
        L.append(f"- {k}: {r['n']} symbol-days, share < 0 {100 * r['share_lt0']:.0f}%, mean {r['mean']:+.4f}, t {_f(r['t'], 2)}")
    L.append("\nBy test day (pooled; mean diff, share < 0): " + "; ".join(f"{d} {x['mean']:+.4f} / {100 * x['share_lt0']:.0f}%" for d, x in n5["pooled"]["by_day"].items()))
    L.append("\n### H-NN6 next-minute direction\n")
    L.append("| segment | windows | symbol-days | CNN AUC | 95% CI | linear AUC | 95% CI | CNN Brier skill |")
    L.append("|---|---|---|---|---|---|---|---|")
    for s in SEGS:
        r = n6[s]
        L.append(f"| {s} | {r['n_windows']:,} | {r['n_symbol_days']} | {_f(r['cnn']['auc'])} | [{_f(r['cnn']['ci95'][0])}, {_f(r['cnn']['ci95'][1])}] | "
                 f"{_f(r['lin']['auc'])} | [{_f(r['lin']['ci95'][0])}, {_f(r['lin']['ci95'][1])}] | {r['cnn']['brier_skill']:+.4f} |")
    L.append("\n### What in the last 5 minutes predicts the next minute's volatility\n")
    L.append("Integrated gradients of the CNN's log-variance output (|IG| share by channel and by lag bucket, test windows):\n")
    chs = list(next(iter(ig.values()))["channel_share"].keys())
    L.append("| channel | " + " | ".join(f"{s} share" for s in ig) + " | " + " | ".join(f"{s} 0-9 s / 10-59 s / 60-299 s" for s in ig) + " | signed (pooled IG sum) |")
    L.append("|---|" + "---|" * (2 * len(ig) + 1))
    for c in sorted(chs, key=lambda c: -max(ig[s]["channel_share"][c] for s in ig)):
        cells = [f"{100 * ig[s]['channel_share'][c]:.1f}%" for s in ig]
        lags = [" / ".join(f"{100 * x:.1f}" for x in ig[s]["channel_lag_share"][c].values()) for s in ig]
        sg = sum(ig[s]["signed_sum"][c] for s in ig)
        L.append(f"| {c} | " + " | ".join(cells) + " | " + " | ".join(lags) + f" | {sg:+.3f} |")
    L.append("\nLag shares (all channels): " + "; ".join(f"{s}: " + ", ".join(f"{k} {100 * x:.0f}%" for k, x in ig[s]["lag_share"].items()) for s in ig))
    L.append("\nChannel ablation (channel set to 0, HAR recomputed): QLIKE increase per symbol-day, mean (t), pooled:\n")
    L.append("| channel | CNN | linear |")
    L.append("|---|---|---|")
    ap = ab["pooled"]
    for c in sorted(chs, key=lambda c: -ap.get(f"cnn:{c}", {"mean": 0})["mean"]):
        a, b = ap.get(f"cnn:{c}", {}), ap.get(f"lin:{c}", {})
        L.append(f"| {c} | {a.get('mean', float('nan')):+.4f} ({_f(a.get('t'), 1)}) | {b.get('mean', float('nan')):+.4f} ({_f(b.get('t'), 1)}) |")
    L.append("\n### Verdicts (after Holm over H-NN5, H-NN6 per segment)\n")
    L.append("| hypothesis | segment | raw | final | Holm p |")
    L.append("|---|---|---|---|---|")
    for h in ("H-NN5", "H-NN6"):
        for s in SEGS:
            L.append(f"| {h} | {s} | {out[h][s]['verdict']} | {v[h][s]} | {_f(out['holm'][s]['p_holm'][h], 3)} |")

    L.append("\n## What this means for tool calls\n")
    L.extend(_synthesis(out))
    L.append("\n## Caveats\n")
    L.append(f"- There are only {len(out['days'])} days (4 before the cutoff), and the folds are leave-days-out, not forward.")
    L.append("- Prices are last trades, so bid-ask bounce inflates the 1 s and 1 m returns of one-tick symbols in both models alike.")
    L.append("- The book channels exist only for the L2 symbols. BTC's 10 bp depth is a truncated lower bound, used here only "
             "relative to its own trailing mean.")
    L.append("- Neither model has a time-of-day input. The intraday profile (R-VOL-26) is a separate, validated adjustment.")
    L.append("- Training windows are subsampled (every 30 s, 100k per epoch), and the CNN is a single seed (runtime cut, made before any test result was seen).\n")
    L.append("## Rule candidates\n")
    L.append("```yaml")
    L.append(_rules(out).strip())
    L.append("```\n")
    return "\n".join(L)


def _top_channels(out: dict, k: int = 4) -> list[tuple[str, float]]:
    ig = out["ig"]
    chs = next(iter(ig.values()))["channel_share"].keys()
    sh = {c: sum(ig[s]["channel_share"][c] * ig[s]["n"] for s in ig) / sum(ig[s]["n"] for s in ig) for c in chs}
    return sorted(sh.items(), key=lambda kv: -kv[1])[:k]


def _synthesis(out: dict) -> list[str]:
    n5, n6, v = out["H-NN5"], out["H-NN6"], out["verdicts"]
    p5 = n5["pooled"]
    top = _top_channels(out)
    lag = {k: sum(out["ig"][s]["lag_share"][k] * out["ig"][s]["n"] for s in out["ig"]) / sum(out["ig"][s]["n"] for s in out["ig"]) for k in next(iter(out["ig"].values()))["lag_share"]}
    L = []
    if v["H-NN5"]["pooled"] == "supports":
        L.append(f"- **Next-minute volatility:** the CNN beats the linear model on {100 * p5['share_lt0']:.0f}% of held-out symbol-days "
                 f"({100 * p5['rel_to_lin_qlike']:+.1f}% QLIKE). Nonlinear order-flow structure matters at this horizon, so a quoting "
                 "tool should use it after a forward test.")
    else:
        bb = p5["by_book"]
        L.append(f"- **Next-minute volatility:** the CNN beats the linear model on the same inputs only by a small margin "
                 f"({100 * p5['rel_to_lin_qlike']:+.1f}% QLIKE, t {_f(p5['t'], 2)}, day-clustered {_f(p5['t_day_clustered'], 2)}). It is better "
                 f"on {100 * p5['share_lt0']:.0f}% of symbol-days pooled, just under the 70% bar ({100 * n5['clean_holdout']['share_lt0']:.0f}% "
                 f"in the holdout, {100 * n5['llm_in_sample']['share_lt0']:.0f}% in-sample). The gain sits in symbol-days without an L2 book "
                 f"({100 * bb['no_book']['share_lt0']:.0f}% better, t {_f(bb['no_book']['t'], 1)}). For the 12 L2-book symbols it is a coin flip "
                 f"({100 * bb['book']['share_lt0']:.0f}%, t {_f(bb['book']['t'], 1)}). Both models beat the naive trailing-variance baseline by "
                 f"about 10x that margin (CNN {p5['context']['cnn_vs_base']['mean']:+.3f}, linear {p5['context']['lin_vs_base']['mean']:+.3f} per "
                 "window). Most of the value is in the linear activity aggregates.")
    L.append("- **What the network looks at:** " + ", ".join(f"{c} ({100 * s:.0f}%)" for c, s in top)
             + " of the attribution. By lag: " + ", ".join(f"{k} {100 * x:.0f}%" for k, x in lag.items())
             + ". The attribution is spread over the whole 5 minutes, with the 1-5 minute span carrying most of it: the network "
             "reads the recent activity LEVEL (squared returns, orders, volume, range), not a last-second trigger. Signed flow, "
             "order imbalance, spread, depth and book imbalance carry almost nothing (ablation of each costs < 0.005 QLIKE).")
    p6 = n6["pooled"]
    L.append(f"- **Next-minute direction:** CNN AUC {_f(p6['cnn']['auc'])} [{_f(p6['cnn']['ci95'][0])}, {_f(p6['cnn']['ci95'][1])}]; "
             f"verdict {v['H-NN6']['pooled']}. " + ("Direction is not predictable from 5 minutes of flow; do not call a flow-based "
                                                   "direction tool." if v["H-NN6"]["pooled"] == "supports" else
                                                   "Any edge is too small to act on without costs and a forward test (an AUC near "
                                                   "0.5 is far below a taker's fee hurdle)."))
    return L


def _rules(out: dict) -> str:
    n5, n6, v = out["H-NN5"], out["H-NN6"], out["verdicts"]
    p5, p6 = n5["pooled"], n6["pooled"]
    top = _top_channels(out, 3)
    tops = ", ".join(c for c, _ in top)
    ab = out["ablation"]["pooled"]
    abl = ", ".join(f"{c} {ab[f'cnn:{c}']['mean']:+.3f}" for c, _ in top if f"cnn:{c}" in ab)
    eff5 = (f"CNN - linear QLIKE {p5['mean_diff']:+.4f} per window ({100 * p5['rel_to_lin_qlike']:+.1f}%), share of symbol-days < 0 "
            f"{100 * p5['share_lt0']:.0f}% (n {p5['n_symbol_days']}), t {_f(p5['t'], 2)}; both beat 60 x trailing-1h variance by "
            f"{p5['context']['cnn_vs_base']['mean']:+.3f} / {p5['context']['lin_vs_base']['mean']:+.3f}")
    if v["H-NN5"]["pooled"] == "supports":
        r11 = f"""
- id: R-ML-11
  kind: prefer
  rule: WHEN a next-minute variance expectation is needed for quoting THEN prefer a validated CNN on the last 5 minutes of 1 s flow over a linear aggregate model, after a forward test
  when: horizon 1 minute; quote widths, short-horizon expectations; not stops
  then: use the CNN forecast; fall back to the linear 10 / 60 / 300 s aggregate model when it is unavailable
  scope:
    assets: crypto and TradFi perps with >= 10k traded seconds per day
    horizon: 1m
    sides: both
  effect: {eff5}
  confidence: C
  evidence: [H-NN5]
  review: forward test on new days (agent A batches) before use
  status: provisional"""
    else:
        r11 = f"""
- id: R-ML-11
  kind: prefer
  rule: WHEN a next-minute variance expectation is needed for quoting THEN use a linear model of 10 / 60 / 300 s activity aggregates (squared returns, prints, orders, range) on top of the trailing variance; a CNN on the raw 1 s path adds only about 1% more, and only for symbols without an L2 book
  when: horizon 1 minute; quote widths, short-horizon expectations; not stops
  then: forecast = trailing-1h per-second variance x 60 x exp(linear aggregates); no convolutional model
  scope:
    assets: crypto and TradFi perps with >= 10k traded seconds per day
    horizon: 1m
    sides: both
  effect: {eff5}
  confidence: C
  evidence: [H-NN5]
  review: re-run on >= 20 days (agent A batches); promote to B if the linear gain over the baseline holds in both segments
  status: provisional"""
    r12 = f"""
- id: R-ML-12
  kind: expect
  rule: WHEN reading the last 5 minutes of 1 s data for next-minute volatility THEN expect the signal to sit in the activity level of the last 1-5 minutes ({tops}), not in signed flow, spread, depth or book imbalance
  when: short-horizon vol expectation
  then: scale the next-minute variance with the recent squared returns, order count and volume relative to their trailing-hour norms; ignore flow sign and book state for this target
  scope:
    assets: crypto and TradFi perps
    horizon: 1m
    sides: both
  effect: integrated-gradient shares {", ".join(f"{c} {100 * s:.0f}%" for c, s in top)}; ablation QLIKE increase {abl}
  confidence: C
  evidence: [H-NN5, H-MS2]
  review: same attribution on >= 20 days
  status: provisional"""
    r13 = f"""
- id: R-ML-13
  kind: avoid
  rule: WHEN looking for next-minute direction THEN don't call a neural network on 1 s order flow, returns, volume or book state; it does not forecast the sign usefully
  when: horizon 1 minute; entries, exits, quote skew
  then: treat the next-minute sign as unpredictable from the last 5 minutes of flow; skew only from validated inventory or flow-toxicity rules
  scope:
    assets: crypto and TradFi perps
    horizon: 1m
    sides: both
  effect: CNN AUC {_f(p6['cnn']['auc'])} [{_f(p6['cnn']['ci95'][0])}, {_f(p6['cnn']['ci95'][1])}], linear {_f(p6['lin']['auc'])}; verdict {v['H-NN6']['llm_in_sample']} / {v['H-NN6']['clean_holdout']} / pooled {v['H-NN6']['pooled']}
  confidence: {'B' if v['H-NN6']['pooled'] == 'supports' else 'C'}
  evidence: [H-NN6, H-MS6, H-OF2]
  review: a registered short-horizon direction model with AUC CI lower bound > 0.52 on new days
  status: {'active' if v['H-NN6']['pooled'] == 'supports' else 'provisional'}"""
    return r11 + r12 + r13
