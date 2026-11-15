"""Verdicts, Holm within SP, recording and the report for the sparsity study (H-SP1..H-SP3).

    PYTHONIOENCODING=utf-8 uv run --extra ml python -m ai_trader.research.sparsity_study evaluate [--record]

Writes research/reports/sparsity_2026-11-12.{json,md}. Design: sparsity_study docstring.
"""

from __future__ import annotations

import json

import numpy as np

from ai_trader.core.config import load
from ai_trader.ml.features import FEATURE_GROUP, ML_FEATURES
from ai_trader.research import sparsity_study as ss
from ai_trader.research.decision_budget_study import holm

HYP = ("H-SP1", "H-SP2", "H-SP3")
REPORT = "sparsity_2026-11-12"


def _f(x, d=4):
    return "nan" if x is None or not np.isfinite(x) else f"{x:.{d}f}"


def _pct(x, d=1):
    return "nan" if x is None or not np.isfinite(x) else f"{100 * x:+.{d}f}%"


def verdicts(sp1: dict, sp2: dict, sp3: dict) -> dict:
    raw = {"H-SP1": {g: (sp1["segments"][g]["verdict"], sp1["segments"][g]["p"]) for g in ss.SEGS},
           "H-SP2": {g: (sp2["segments"][g]["verdict"], sp2["p_holm"]) for g in ss.SEGS},
           "H-SP3": {g: (sp3[g]["verdict"], sp3[g]["p"]) for g in ss.SEGS}}
    pooled = {"H-SP1": sp1["pooled_verdict"], "H-SP2": sp2["pooled_verdict"], "H-SP3": sp3["pooled_verdict"]}
    hp = {g: holm({h: raw[h][g][1] for h in HYP}) for g in ss.SEGS}
    out = {}
    for h in HYP:
        row = {}
        for g in ss.SEGS:
            v, p = raw[h][g]
            v2 = "inconclusive" if (v == "supports" and hp[g][h] > 0.05) else v
            row[g] = {"verdict_raw": v, "p": p, "holm_p": hp[g][h], "verdict": v2}
        if h == "H-SP3":
            ph = hp["clean_holdout"][h]
        else:
            ph = max(hp[g][h] for g in ss.SEGS)
        pv = pooled[h]
        row["pooled"] = {"verdict_raw": pv, "holm_p": ph, "verdict": "inconclusive" if (pv == "supports" and ph > 0.05) else pv}
        out[h] = row
    return out


def _sp1_metrics(sp1: dict, g: str) -> dict:
    lv = sp1["segments"][g]["levels"]
    return {k: {"round": v["round"], "sparsity": round(v["sparsity"], 4), "ratio": round(v["ratio"], 4), "ci95": [round(x, 4) for x in v["ci95"]],
                "verdict": v["verdict"]} for k, v in lv.items()} | {"n_dates": lv["95"]["n_dates"], "months": sp1["months"]}


def record_all(sp1: dict, sp2: dict, sp3: dict, V: dict) -> list:
    from ai_trader.research.hypotheses import record, results

    data = f"data/models/ml/sparse/ ; research/reports/{REPORT}.json"
    done = []
    for h in HYP:
        have = {r["segment"] for r in results(h)}
        for g in (*ss.SEGS, "pooled"):
            if g in have:
                continue
            v = V[h][g]
            if h == "H-SP1":
                m = _sp1_metrics(sp1, g) if g != "pooled" else {s: _sp1_metrics(sp1, s)["95"] for s in ss.SEGS} | {"passing_levels": sp1["passing_levels"]}
                note = ("IMP with rewinding of the H-NN1 MLP, 56 monthly refits, 3-seed ensemble; QLIKE ratio sparse/dense at 95.4% sparsity "
                        "(round 14); criterion ratio <= 1.01 and CI upper <= 1.02")
            elif h == "H-SP2":
                if g != "pooled":
                    s = sp2["segments"][g]
                    m = {"n_selected": s["n_selected"], "selected": s["selected"], "q_hat": round(s["q_hat"], 2), "pfer_bound": round(s["pfer_bound"], 3),
                         "p_holm_input": sp2["p_holm"], "library_size": sp2["p"]}
                else:
                    m = {"jaccard": sp2["overlap"]["jaccard"], "intersection": sp2["overlap"]["intersection"], "p_hypergeom": sp2["overlap"]["p_hypergeom"]}
                note = "randomised-lasso stability selection, 236-column library, 200 date-half subsamples, pi 0.8, E[V] <= 1"
            else:
                gg = "clean_holdout" if g == "pooled" else g
                s = sp3[gg]
                m = {"level": sp3["level"], "n_units": sp3["n_units"], "n_evaluable": s["n_evaluable"], "share_r2_gt_0.8": s["share_meet"],
                     "ablation_ratio": sp3["ablation"][gg]["mean_ablation"]["ratio"], "ablation_ci95": sp3["ablation"][gg]["mean_ablation"]["ci95"]}
                note = "seed-0 pruned member of the 2026-07 refit; additive spline on <= 2 inputs; top-3 units mean-ablated"
            m |= {"holm_p": v["holm_p"], "verdict_before_holm": v["verdict_raw"]}
            record(h, g, v["verdict"], m, note=note, data=data)
            done.append((h, g, v["verdict"]))
    return done


# ------------------------------------------------------------------ report


def _curve_table(sp1: dict) -> list[str]:
    L = ["| round | sparsity | weights | ratio in-sample [95% CI] | ratio holdout [95% CI] | vs log-HAR in / ho | mean epochs |", "|---|---|---|---|---|---|---|"]
    for r in range(ss.ROUNDS + 1):
        a = sp1["segments"]["llm_in_sample"]["curve"][r]
        b = sp1["segments"]["clean_holdout"]["curve"][r]
        L.append(f"| {r} | {100 * sp1['sparsity'][r]:.1f}% | {sp1['n_alive'][r]} | {a['ratio']:.4f} [{a['ci95'][0]:.4f}, {a['ci95'][1]:.4f}] | "
                 f"{b['ratio']:.4f} [{b['ci95'][0]:.4f}, {b['ci95'][1]:.4f}] | {_pct(a['vs_har']['rel'])} ({_f(a['vs_har']['t'], 1)}) / "
                 f"{_pct(b['vs_har']['rel'])} ({_f(b['vs_har']['t'], 1)}) | {sp1['epochs_mean'][r]:.0f} |")
    return L


def _survival(sp1: dict, lab: str, top: int = 14) -> str:
    s = sp1["input_survival"][lab]
    items = sorted(s.items(), key=lambda kv: -kv[1])[:top]
    return "; ".join(f"{k} {100 * v:.0f}%" for k, v in items)


def _group_survival(sp1: dict, lab: str) -> str:
    s = sp1["input_survival"][lab]
    g: dict = {}
    for f, v in s.items():
        g.setdefault(FEATURE_GROUP[f], []).append(v)
    return "; ".join(f"{k} {np.mean(v) * 100:.0f}%" for k, v in sorted(g.items(), key=lambda kv: -np.mean(kv[1])))


def build_md(sp1: dict, sp2: dict, sp3: dict, V: dict, rules_yaml: str, narrative: str) -> str:
    cut = load().settings["research"]["knowledge_cutoff"]
    L = ["# Sparsity: the inputs, nodes and neurons that forecast daily volatility (H-SP1 to H-SP3)", "",
         f"_Program 2026-11-12, agent D. Data as of 2026-11-12; knowledge cutoff {cut}. Code: `src/ai_trader/ml/sparse.py`, "
         "`src/ai_trader/research/sparsity_study.py` (design in its docstring), `src/ai_trader/research/sparsity_report.py`; tests: `tests/test_sparse.py`; "
         "artifacts: `data/models/ml/sparse/`. These models are deterministic research tools. The LLM decision layer is not trained or backtested here._", "",
         "## Summary", "", "| hypothesis | in-sample | holdout | pooled | headline |", "|---|---|---|---|---|"]
    r95 = sp1["rounds_at"]["0.95"]
    a = sp1["segments"]["llm_in_sample"]["levels"]["95"]
    b = sp1["segments"]["clean_holdout"]["levels"]["95"]
    L.append(f"| H-SP1: a 95%-pruned net forecasts as well as the dense one | {V['H-SP1']['llm_in_sample']['verdict']} | {V['H-SP1']['clean_holdout']['verdict']} | "
             f"**{V['H-SP1']['pooled']['verdict']}** | QLIKE ratio sparse/dense at {100 * a['sparsity']:.1f}%: {a['ratio']:.4f} [{a['ci95'][0]:.4f}, {a['ci95'][1]:.4f}] in-sample, "
             f"{b['ratio']:.4f} [{b['ci95'][0]:.4f}, {b['ci95'][1]:.4f}] holdout; passing levels: {', '.join(sp1['passing_levels']) or 'none'} |")
    s_in, s_ho = sp2["segments"]["llm_in_sample"], sp2["segments"]["clean_holdout"]
    L.append(f"| H-SP2: stability selection finds <= 10 stable inputs, overlap >= 70% | {V['H-SP2']['llm_in_sample']['verdict']} | {V['H-SP2']['clean_holdout']['verdict']} | "
             f"**{V['H-SP2']['pooled']['verdict']}** | {s_in['n_selected']} / {s_ho['n_selected']} selected (pi >= 0.8, E[V] <= {s_in['pfer_bound']:.2f} / {s_ho['pfer_bound']:.2f}); "
             f"Jaccard {sp2['overlap']['jaccard']:.2f} |")
    h3 = sp3["clean_holdout"]
    ab = sp3["ablation"]["clean_holdout"]["mean_ablation"]
    L.append(f"| H-SP3: surviving units are simple functions of <= 2 inputs, top-3 ablation > +5% | {V['H-SP3']['llm_in_sample']['verdict']} | {V['H-SP3']['clean_holdout']['verdict']} | "
             f"**{V['H-SP3']['pooled']['verdict']}** | {sp3['n_units']} surviving units at {sp3['level']}%; {h3['n_meet_r2']} / {h3['n_evaluable']} with holdout R^2 > 0.8; "
             f"top-3 ablation QLIKE x{ab['ratio']:.3f} [{ab['ci95'][0]:.3f}, {ab['ci95'][1]:.3f}] |")
    L += ["", narrative, "", "## Methods", "",
          "**Panel and protocol.** Identical to H-NN1 (`ai_trader.ml.features`, universe L, 52 point-in-time inputs, complete rows, decision at 00:00 UTC of D+1; "
          "monthly expanding walk-forward 2022-04 to 2026-11 with 24 h purge + 1 day embargo; last 15% of training dates as the purged validation block; "
          "fit-block standardisation; early stopping; validation-block bias constant). Network: vol MLP (32, 16), wd 0.01, 3 seeds; direction MLP (64, 32), wd 1e-4. "
          "The dense network is re-trained with `ml.sparse.train_masked` (no mask), which is bit-identical to `models.fit_mlp`.",
          "",
          f"**H-SP1 pruning.** Iterative magnitude pruning, layer-wise (20% of surviving hidden weights, 10% of the head per round; biases kept), rewinding the "
          f"survivors to the end of epoch 1 (weights and batch order) and retraining with the mask re-applied after every step; {ss.ROUNDS} rounds per seed and refit, "
          f"full walk-forward ({sp1['months']} refits, no subset). Sparsity is deterministic: {100 * sp1['sparsity'][r95]:.1f}% at round {r95} (test level), "
          f"{100 * sp1['sparsity'][sp1['rounds_at']['0.98']]:.1f}% at {sp1['rounds_at']['0.98']}, {100 * sp1['sparsity'][sp1['rounds_at']['0.99']]:.2f}% at {sp1['rounds_at']['0.99']}. "
          "Ratio = date-averaged QLIKE of the sparse ensemble / dense ensemble; 10-day moving-block bootstrap over dates (1000).",
          "",
          f"**H-SP2 stability selection.** Library of {sp2['p']} columns: " + ", ".join(f"{k} {v}" for k, v in sorted(sp2["family_counts"].items())) +
          ". Lags are exact calendar lags within the symbol; products are of the standardised H-NN3 top-10 inputs; quantile dummies are 1{x > q} with in-sample edges. "
          "Target next-day log RV. Per segment: 200 subsamples of half of the dates, randomised lasso (penalty 1/W, W ~ U[0.5, 1]), 40 lambdas, region where the mean "
          "union size <= q_max = sqrt(0.6 p) (E[V] <= 1), threshold 0.8.",
          "",
          "**H-SP3 units.** Seed-0 member of the 2026-07 refit (trained on labels to 2026-06-30) at the H-SP1-selected level. Additive cubic splines (6 quantile knots; "
          "one-hot for inputs with <= 8 values) on the best input and the best additive pair (chosen by fitting R^2); holdout R^2. Attribution |a x dy/da| on in-sample rows; "
          "top-3 mean ablation.", "", "## H-SP1: lottery-ticket pruning of the H-NN1 vol network", ""]
    L += _curve_table(sp1)
    L += ["", "Test levels:", "", "| segment | level | round | ratio | 95% CI | P*(ratio >= 1.02) | verdict |", "|---|---|---|---|---|---|---|"]
    for g in ss.SEGS:
        for lab, v in sp1["segments"][g]["levels"].items():
            L.append(f"| {g} | {lab}% | {v['round']} | {v['ratio']:.4f} | [{v['ci95'][0]:.4f}, {v['ci95'][1]:.4f}] | {v['p_ge_1.02']:.3f} | {v['verdict']} |")
    L += ["", "Single seed-0 member (sparse member / dense member), descriptive:", "", "| segment | 95% | 98% | 99% |", "|---|---|---|---|"]
    for g in ss.SEGS:
        c = sp1["segments"][g]["curve"]
        L.append(f"| {g} | " + " | ".join(f"{c[r]['member0']['ratio']:.4f} [{c[r]['member0']['ci95'][0]:.3f}, {c[r]['member0']['ci95'][1]:.3f}]"
                                          for r in (sp1["rounds_at"]["0.95"], sp1["rounds_at"]["0.98"], sp1["rounds_at"]["0.99"])) + " |")
    L += ["", "Sub-segments (ratio sparse/dense):", "", "| segment | subset | n obs | 95% | 99% |", "|---|---|---|---|---|"]
    for g in ss.SEGS:
        for k, v in sp1["segments"][g]["subsegments"].items():
            L.append(f"| {g} | {k} | {v['n_obs']} | {v['r95']['ratio']:.4f} [{v['r95']['ci95'][0]:.3f}, {v['r95']['ci95'][1]:.3f}] | "
                     f"{v['r99']['ratio']:.4f} [{v['r99']['ci95'][0]:.3f}, {v['r99']['ci95'][1]:.3f}] |")
    dv = sp1["segments"]
    L += ["", f"Sanity: re-trained dense ensemble vs the stored H-NN1 forecasts, QLIKE ratio {dv['llm_in_sample']['dense_vs_hnn1_stored']['ratio']:.4f} in-sample, "
          f"{dv['clean_holdout']['dense_vs_hnn1_stored']['ratio']:.4f} holdout (thread count differs: 1 vs 4). Masked weights exactly zero in every fit: {sp1['mask_ok']}.", "",
          "**Which inputs survive** (share of the 56 x 3 pruned networks in which the input still has a path to the output):", ""]
    for lab in ("95", "98", "99"):
        ia = sp1["inputs_alive"][lab]
        un = sp1["units"][lab]
        L.append(f"- {lab}%: inputs alive mean {ia['mean']:.1f} (min {ia['min']}, max {ia['max']}), hidden units mean {un['mean']:.1f}; top: {_survival(sp1, lab)}")
        L.append(f"  - by concept group: {_group_survival(sp1, lab)}")
    d = sp1["direction"]
    L += ["", "### Direction (descriptive): the same pruning of the H-NN2 direction network", "",
          "| round | sparsity | skill in-sample [95% CI] | sd(p) in | skill holdout [95% CI] | sd(p) ho |", "|---|---|---|---|---|---|"]
    for r in sorted({0, 5, 10, d["rounds_at"]["0.95"], d["rounds_at"]["0.98"], d["rounds_at"]["0.99"], ss.ROUNDS}):
        a_, b_ = d["segments"]["llm_in_sample"][r], d["segments"]["clean_holdout"][r]
        L.append(f"| {r} | {100 * d['sparsity'][r]:.1f}% | {a_['skill']:+.5f} [{a_['ci95'][0]:+.5f}, {a_['ci95'][1]:+.5f}] | {a_['sd_p']:.4f} | "
                 f"{b_['skill']:+.5f} [{b_['ci95'][0]:+.5f}, {b_['ci95'][1]:+.5f}] | {b_['sd_p']:.4f} |")
    L += ["", "Surviving units (mean over refits x seeds; share of networks with no surviving unit): " +
          "; ".join(f"round {k}: {v['mean']:.1f} ({100 * v['share_zero']:.0f}% empty)" for k, v in d["units"].items()),
          "", "Direction inputs surviving at 99%: " + "; ".join(f"{k} {100 * v:.0f}%" for k, v in sorted(d["input_survival_99"].items(), key=lambda kv: -kv[1])[:10]), ""]
    # SP2
    L += ["## H-SP2: stability selection over a 236-column library", ""]
    for g in ss.SEGS:
        s = sp2["segments"][g]
        L += [f"**{g}** ({s['n_rows']} rows, {s['n_dates']} dates; q_hat {s['q_hat']:.1f}, region down to {s['lambda_min_rel']:.3f} lambda_max, PFER bound {s['pfer_bound']:.2f}): "
              f"{s['n_selected']} selected: " + (", ".join(f"{k} ({v:.2f})" for k, v in sorted(s["selected_probs"].items(), key=lambda kv: -kv[1])) or "none") +
              f". At pi 0.6: {', '.join(s['selected_at_0.6'])}.", ""]
    L += ["| rank | llm_in_sample: feature (max prob, family) | clean_holdout: feature (max prob, family) |", "|---|---|---|"]
    for i in range(20):
        x, y = s_in["top25"][i], s_ho["top25"][i]
        L.append(f"| {i + 1} | {x[0]} ({x[1]:.2f}, {x[2]}) | {y[0]} ({y[1]:.2f}, {y[2]}) |")
    o = sp2["overlap"]
    L += ["", f"Overlap: intersection {o['intersection']}; union {o['union']}; Jaccard {o['jaccard']:.2f}; share of the smaller set {o['share_of_smaller']:.2f}; "
          f"hypergeometric p {o['p_hypergeom']:.2g}.", ""]
    cl = sp2.get("clusters")
    if cl:
        L += [f"**Exploratory (post hoc, after nothing was selected): cluster stability.** Single-linkage clusters of library columns with |rho| >= {cl['rho']} "
              f"on in-sample rows ({cl['n_clusters']} clusters); the same subsamples (probabilities identical to the registered run: "
              f"{all(v['identical_to_registered_run'] for v in cl['segments'].values())}); probability that at least one member is selected, max over Lambda.", "",
              "| segment | cluster probability | size | members |", "|---|---|---|---|"]
        for g in ss.SEGS:
            for r in cl["segments"][g]["top"][:8]:
                L.append(f"| {g} | {r['prob']:.2f} | {r['size']} | {', '.join(r['members'][:9])}{' ...' if r['size'] > 9 else ''} |")
        L.append("")
    # SP3
    L += [f"## H-SP3: the surviving neurons ({sp3['level']}% level, round {sp3['round']}, {sp3['n_alive_weights']} weights)", "",
          f"Weights alive per layer: {sp3['alive_per_layer']}; inputs with a path to the output: {', '.join(sp3['inputs_with_path'])}. "
          f"Other seeds at the same round: " + "; ".join(f"seed {k}: {v['n_units']} units, inputs {', '.join(v['inputs_with_path'])}" for k, v in sp3["other_seeds"].items()) + ".", "",
          "| unit | attribution | surviving inputs (weight) | best 1 input (holdout R^2) | best additive pair (holdout R^2) | pair + interaction R^2 | in-sample block R^2 (pair) | plain words |",
          "|---|---|---|---|---|---|---|---|"]
    for r in sorted(sp3["units"], key=lambda q: -q["attribution"]):
        h = r["clean_holdout"]
        nm = f"L{r['unit'][0]}u{r['unit'][1]}"
        if not h.get("evaluable"):
            L.append(f"| {nm} | {r['attribution']:.4f} | - | not evaluable (sd {h['sd_eval']:.2g}) | | | | |")
            continue
        de = r["describe"]
        ins = ", ".join(f"{k} {v:+.2f}" for k, v in de["inputs"].items())
        i2 = r["llm_in_sample"]
        L.append(f"| {nm} | {r['attribution']:.4f} | {ins} | {h['best1']} ({h['r2_1']:.2f}) | {' + '.join(h['best2'])} ({h['r2_2']:.2f}) | {h['r2_bivariate']:.2f} | "
                 f"{_f(i2.get('r2_2', float('nan')), 2)} | {de['plain']} |")
    L += ["", "| segment | evaluable units | R^2 > 0.8 (pair) | R^2 > 0.8 (1 input) | top-3 mean ablation | zero ablation | intact QLIKE |", "|---|---|---|---|---|---|---|"]
    for g in ss.SEGS:
        s = sp3[g]
        ab = sp3["ablation"][g]
        L.append(f"| {g} | {s['n_evaluable']} | {s['n_meet_r2']} ({_f(s['share_meet'], 2)}) | {_f(s['share_meet_1input'], 2)} | x{ab['mean_ablation']['ratio']:.3f} "
                 f"[{ab['mean_ablation']['ci95'][0]:.3f}, {ab['mean_ablation']['ci95'][1]:.3f}] | x{ab['zero_ablation']['ratio']:.3f} | {ab['intact_qlike']:.4f} |")
    L += ["", f"Top-3 by attribution: {', '.join(sp3['top3'])}. Each unit alone (holdout QLIKE ratio, mean ablation): " +
          "; ".join(f"{k} x{v:.3f}" for k, v in sp3["ablation"]["clean_holdout"]["each_unit_ratio"].items()), ""]
    # Holm
    L += ["## Verdicts after Holm within SP", "", "| hypothesis | segment | verdict before Holm | p | Holm p | verdict |", "|---|---|---|---|---|---|"]
    for h in HYP:
        for g in (*ss.SEGS, "pooled"):
            v = V[h][g]
            L.append(f"| {h} | {g} | {v['verdict_raw']} | {_f(v.get('p', float('nan')), 3)} | {v['holm_p']:.3f} | **{v['verdict']}** |")
    L += ["", "## Rule candidates", "", "```yaml", rules_yaml.rstrip(), "```", ""]
    return "\n".join(L)


def evaluate(record: bool = False, narrative: str = "", rules_yaml: str = "") -> dict:
    sp1 = json.loads((ss.root() / "sp1_eval.json").read_text(encoding="utf-8"))
    sp2 = ss.eval_sp2()
    cp = ss.root() / "sp2_clusters.json"
    if cp.is_file():
        sp2["clusters"] = json.loads(cp.read_text(encoding="utf-8"))
    sp3 = json.loads((ss.root() / "sp3_units.json").read_text(encoding="utf-8"))
    V = verdicts(sp1, sp2, sp3)
    if not narrative or not rules_yaml:
        from ai_trader.research import sparsity_text as tx

        narrative = narrative or tx.narrative(sp1, sp2, sp3, V)
        rules_yaml = rules_yaml or tx.RULES_YAML
    rep = load().root / "research" / "reports"
    out = {"verdicts": V, "sp1": {k: v for k, v in sp1.items() if k != "direction"}, "direction": sp1["direction"], "sp2": sp2, "sp3": sp3,
           "features": list(ML_FEATURES)}
    (rep / f"{REPORT}.json").write_text(json.dumps(out, indent=1, default=float), encoding="utf-8")
    (rep / f"{REPORT}.md").write_text(build_md(sp1, sp2, sp3, V, rules_yaml, narrative), encoding="utf-8")
    if record:
        out["recorded"] = record_all(sp1, sp2, sp3, V)
        print("recorded", out["recorded"])
    print(json.dumps({h: {g: V[h][g]["verdict"] for g in V[h]} for h in V}, indent=1))
    return out
