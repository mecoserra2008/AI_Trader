"""Markdown report for the BTC polynomial-neuron network study (pnn_study): research/reports/btc_pnn_<TAG>.md.
Generated from the JSON twin; do not edit the .md by hand."""

from __future__ import annotations

from pathlib import Path

SEGS = ("llm_in_sample", "clean_holdout")
HS = ("1h", "1d")
LEVELS = ("0.8", "0.9", "0.95")
VC = {"supports": "supports", "contradicts": "contradicts", "inconclusive": "inconclusive"}


def f(x, k=4) -> str:
    try:
        x = float(x)
    except (TypeError, ValueError):
        return str(x)
    if x != x:
        return "n/a"
    return f"{x:.{k}f}"


def pt(d: dict, k: int = 5) -> str:
    return f"{f(d['diff'], k)} ({f(d['t'], 2)})"


def ci(c) -> str:
    return f"[{f(c[0], 3)}, {f(c[1], 3)}]"


def _rules(o: dict) -> list[str]:
    E, V = o["eval"], o["verdicts"]
    ho = {h: E[h]["clean_holdout"] for h in HS}
    ins = {h: E[h]["llm_in_sample"] for h in HS}
    L = []

    def rule(i, kind, text, when, then, horizon, effect, conf, evidence, review, status):
        L.extend([f'- id: "R-PNN-{i}"', f'  kind: "{kind}"', f'  rule: "{text}"', f'  when: "{when}"', f'  then: "{then}"',
                  "  scope:", '    assets: "BTCUSDT"', f'    horizon: "{horizon}"', '    sides: "both"', f'  effect: "{effect}"',
                  f'  confidence: "{conf}"', "  evidence: [" + ", ".join(f'"{e}"' for e in evidence) + "]", f'  review: "{review}"',
                  f'  status: "{status}"'])

    c = {h: ho[h]["conformal"] for h in HS}
    cov_txt = "; ".join(f"{h} " + ", ".join(
        f"{int(float(lv) * 100)}%: ACI {f(c[h][lv]['aci']['coverage'], 3)} / HAR-t {f(c[h][lv]['har']['coverage'], 3)}" for lv in LEVELS) for h in HS)
    wr = "; ".join(f"{h} " + ", ".join(f"{f(c[h][lv]['aci_over_har_width'], 3)}" for lv in LEVELS) for h in HS)
    p = {h: {"ho_pnn": ho[h]["pnn_vs_har"], "ho_mlp": ho[h]["mlp_vs_har"], "in_pnn": ins[h]["pnn_vs_har"], "in_mlp": ins[h]["mlp_vs_har"],
             "ho_pm": ho[h]["pnn_vs_mlp"], "in_pm": ins[h]["pnn_vs_mlp"]} for h in HS}
    rule(1, "prefer",
         "WHEN a BTC next-hour return distribution is needed THEN prefer a small quantile network on the hourly vol features (robust sigma, "
         "hour-of-week factor, recent |returns|, daily HAR level) over log-HAR-t quantiles; the polynomial-edge and ReLU versions are interchangeable",
         "next-hour BTC quantiles or bands (targets, stop distances, quote widths)",
         "quantiles from the 1h PNN or MLP (data/models/pnn/latent_1h.parquet q05..q95); log-HAR-t as the fallback", "1h",
         f"pinball in / holdout: pnn - log-HAR-t {pt(p['1h']['in_pnn'])} / {pt(p['1h']['ho_pnn'])}, mlp - log-HAR-t "
         f"{pt(p['1h']['in_mlp'])} / {pt(p['1h']['ho_mlp'])}, pnn - mlp {pt(p['1h']['in_pm'])} / {pt(p['1h']['ho_pm'])}",
         "B", ["H-PNN1", "H-NN8"], "a forward quarter in which the 1h network's pinball is not below log-HAR-t", "provisional")
    rule(2, "expect",
         "WHEN a BTC next-day return distribution is needed THEN expect log-HAR-t (Student-t on the HAR variance) to be as good as a "
         "~450-parameter network; do not cite a network quantile as an improvement at 1d",
         "next-day BTC quantiles or bands", "log-HAR-t quantiles; networks only as a cross-check", "1d",
         f"pinball in / holdout: pnn - log-HAR-t {pt(p['1d']['in_pnn'])} / {pt(p['1d']['ho_pnn'])}, mlp - log-HAR-t "
         f"{pt(p['1d']['in_mlp'])} / {pt(p['1d']['ho_mlp'])}",
         "C", ["H-PNN1"], "a holdout year in which a network beats log-HAR-t at 1d with t < -2", "provisional")
    rule(3, "expect",
         "WHEN quoting a BTC 80/90/95% band THEN expect unconformalised log-HAR-t bands to be close to nominal already; an adaptive-conformal "
         "wrapper keeps long-run coverage on target but does not make the band narrower",
         "a band whose coverage matters (probability-of-touch, range expectations)",
         "log-HAR-t band or the ACI band (aci_lo / aci_hi); a running ACI alpha far from nominal is a calibration alarm", "1h, 1d",
         f"clean_holdout coverage {cov_txt}; ACI / HAR-t width {wr}", "C", ["H-PNN2"],
         "a forward quarter with log-HAR-t band coverage off nominal by > 3 pp", "provisional")
    lat = "; ".join(f"{h} R^2 gain over persistence {f(ho[h]['latent']['gain'], 3)} (t {f(ho[h]['latent']['gain_t'], 1)}), "
                    f"fb - ctrl pinball {pt(ho[h]['fb_vs_ctrl'])}" for h in HS)
    rule(4, "avoid",
         "WHEN a model's latent state is predictable THEN do not assume that feeding the predicted next latent back improves the forecast; "
         "the latent trajectory is a display of the vol / calendar state, not an extra signal",
         "designing feedback loops between the real and latent spaces (dashboard, world-model ideas)",
         "show z and zhat on the dashboard; keep the forecaster without the feedback input", "1h, 1d", f"clean_holdout {lat}", "C",
         ["H-PNN3"], "a feedback variant with pinball gain t < -2 in both segments", "provisional")
    return L


def write_md(o: dict, path: Path) -> None:
    E, V, P, W, S = o["eval"], o["verdicts"], o["params"], o["walk"], o["symbolic"]
    L: list[str] = []
    a = L.append
    a("# BTC polynomial-neuron network: quantiles, conformal bands and latent dynamics (H-PNN1 to H-PNN3)")
    a("")
    a(f"_BTC field program 2026-11-13, agent F2. Data as of {o['as_of'][:10]}; knowledge cutoff {o['cutoff']}. Code: "
      "`src/ai_trader/ml/pnn.py` (network, conformal, latent dynamics, symbolic extraction), `src/ai_trader/research/pnn_study.py` "
      "(design in its docstring), `src/ai_trader/research/pnn_report.py`; tests: `tests/test_pnn.py`; artifacts: `data/models/pnn/`. "
      "Deterministic research tools; the LLM decision layer is not trained or backtested here._")
    a("")
    a("## Summary")
    a("")
    a("| hypothesis | in-sample | holdout | pooled |")
    a("|---|---|---|---|")
    for hid in ("H-PNN1", "H-PNN2", "H-PNN3"):
        a(f"| {hid} | {V['llm_in_sample'][hid]['verdict']} | {V['clean_holdout'][hid]['verdict']} | **{V['pooled'][hid]['verdict']}** |")
    a("")
    for s in SEGS:
        a(f"Holm p ({s}): H-PNN1 {f(V[s]['H-PNN1']['holm_p'], 3)} (raw {f(V[s]['H-PNN1']['p'], 3)}), "
          f"H-PNN3 {f(V[s]['H-PNN3']['holm_p'], 3)} (raw {f(V[s]['H-PNN3']['p'], 3)}).")
    a("")
    a("**Parameter counts.** " + "; ".join(
        f"{h}: PNN {P[h]['pnn']}, MLP {P[h]['mlp']} (hidden {P[h]['mlp_hidden']}), PNN with feedback head {P[h]['pnn_fb']}" for h in HS)
      + ". Cap 2,000.")
    a("")
    a("**Walk-forward.** " + "; ".join(
        f"{h}: {W[h]['refits']} monthly refits {W[h]['first']} to {W[h]['last']}, mean early-stopped epochs PNN {f(W[h]['mean_epochs_pnn'], 1)} / "
        f"MLP {f(W[h]['mean_epochs_mlp'], 1)}, {f(W[h]['cpu_minutes'], 1)} CPU-minutes" for h in HS) + ".")
    a("")
    ho1, hd1, in1, id1 = E["1h"]["clean_holdout"], E["1d"]["clean_holdout"], E["1h"]["llm_in_sample"], E["1d"]["llm_in_sample"]
    a("## Findings")
    a("")
    a(f"- **1h: a ~500-parameter quantile network beats log-HAR-t, but the polynomial edges add nothing over ReLU.** PNN - log-HAR-t "
      f"pinball {pt(in1['pnn_vs_har'])} in-sample and {pt(ho1['pnn_vs_har'])} in the holdout; MLP - log-HAR-t {pt(in1['mlp_vs_har'])} / "
      f"{pt(ho1['mlp_vs_har'])}; PNN - MLP {pt(in1['pnn_vs_mlp'])} / {pt(ho1['pnn_vs_mlp'])}. The gain is small (0.4-0.6% of the loss) and "
      "comes from the hour-of-week factor and recent |returns|, which log-HAR-t enters only linearly in log variance.")
    a(f"- **1d: nothing beats log-HAR-t.** PNN - log-HAR-t {pt(id1['pnn_vs_har'])} in-sample, {pt(hd1['pnn_vs_har'])} holdout "
      f"({hd1['dates']} dates); the MLP is no better ({pt(id1['mlp_vs_har'])} / {pt(hd1['mlp_vs_har'])}). With about 2,000 daily rows the "
      "networks stay close to the unconditional quantiles. H-PNN1 contradicts in-sample and is inconclusive in the holdout, so pooled contradicts.")
    a("- **Calibration is not the bottleneck.** log-HAR-t bands are already within about 2 pp of nominal on BTC. ACI holds 1h holdout coverage "
      "within 0.2 pp at all three levels, but its 1h bands are 2-3% wider than log-HAR-t; at 1d (132 dates) the 90% ACI band covers 0.880. "
      "H-PNN2 is inconclusive: the coverage half nearly passes, the width half fails at 1h.")
    a(f"- **The latent space is predictable but useless as feedback.** Linear dynamics with the next-step calendar beat persistence by R^2 "
      f"{f(ho1['latent']['gain'], 3)} (1h) and {f(hd1['latent']['gain'], 3)} (1d) in the holdout (t {f(ho1['latent']['gain_t'], 1)} / "
      f"{f(hd1['latent']['gain_t'], 1)}). Feeding zhat back changes pinball by {pt(ho1['fb_vs_ctrl'])} / {pt(hd1['fb_vs_ctrl'])}. The "
      "predictable part is the calendar and vol persistence, which the head already sees, so part (b) fails and H-PNN3 is inconclusive.")
    a("- **What the neurons compute.** At 1h the most important inputs are the hour-of-week factor (lfac), the robust base sigma and the last "
      "hour's |return|. The tail increments (o05, o95) are driven by one latent that mixes vol level and hour of week through roughly cubic "
      "edges. At 1d one latent is a weekend clock (next_wk_hours -> z3, quadratic, R^2 0.995) that narrows the band on weekends; the others are "
      "vol-level mixes. About a third (1h) / a fifth (1d) of encoder edges are cubics with R^2 >= 0.99. The rest saturate in the tails "
      "(tanh basis), so the closed forms are approximations over the bulk of the data.")
    a("- **Inputs.** Signed flow (delta) has no multi-year BTC history in the lake (flow_bars start 2026-05), so it is absent. F1's "
      "`field.state` appeared after this design was fixed. It carries the same information (vol speeds, hour of week, jumps, momentum, "
      "funding, OI, regime); a rerun on it would be a new track record, not a change to this one.")
    a("")
    a("## H-PNN1: pinball loss (vol-standardised units), paired differences per date, NW t")
    a("")
    a("| segment | horizon | dates | PNN | MLP | log-HAR-t | PNN - MLP (t) | PNN - HAR (t) | MLP - HAR (t) |")
    a("|---|---|---|---|---|---|---|---|---|")
    for s in SEGS:
        for h in HS:
            r = E[h][s]
            a(f"| {s} | {h} | {r['dates']} | {f(r['pinball']['pnn'], 5)} | {f(r['pinball']['mlp'], 5)} | {f(r['pinball']['har'], 5)} | "
              f"{pt(r['pnn_vs_mlp'])} | {pt(r['pnn_vs_har'])} | {pt(r['mlp_vs_har'])} |")
    a("")
    a("Quantile calibration (share of outcomes below each forecast quantile, clean_holdout; nominal 0.05 ... 0.95):")
    a("")
    a("| horizon | model | " + " | ".join(f"q{t}" for t in ("05", "10", "25", "50", "75", "90", "95")) + " |")
    a("|---|---|" + "---|" * 7)
    for h in HS:
        for m in ("pnn", "mlp", "har"):
            a(f"| {h} | {m} | " + " | ".join(f(x, 3) for x in E[h]["clean_holdout"]["hit_rate"][m]) + " |")
    a("")
    a("## H-PNN2: conformal coverage and width")
    a("")
    a("Width = mean central-interval width in bps of return. ACI = adaptive conformal (Gibbs-Candes, gamma 0.005); SC = split conformal "
      "(each refit's validation block); HAR = unconformalised log-HAR-t at the nominal level; raw = PNN quantile band without conformal "
      "(80%: q10-q90, 90%: q5-q95). 10-day block-bootstrap 95% CIs over dates.")
    a("")
    a("| segment | horizon | level | ACI cov [CI] | SC cov | HAR cov | raw cov | ACI width | HAR width | ACI / HAR width [CI] |")
    a("|---|---|---|---|---|---|---|---|---|---|")
    for s in SEGS:
        for h in HS:
            for lv in LEVELS:
                c = E[h][s]["conformal"][lv]
                raw = f(c["raw"]["coverage"], 3) if "raw" in c else "n/a"
                a(f"| {s} | {h} | {int(float(lv) * 100)}% | {f(c['aci']['coverage'], 3)} {ci(c['aci']['cov_ci'])} | {f(c['sc']['coverage'], 3)} | "
                  f"{f(c['har']['coverage'], 3)} | {raw} | {f(c['aci']['width_bps'], 1)} | {f(c['har']['width_bps'], 1)} | "
                  f"{f(c['aci_over_har_width'], 3)} {ci(c['aci_over_har_ci'])} |")
    a("")
    a("## H-PNN3: latent predictability and feedback")
    a("")
    a("Latent = the 4-unit bottleneck of each refit's seed-0 PNN, standardised with its fit-block mean / sd. Dynamics "
      "z_{t+1} = A z_t + B u_t + c (u = calendar of the next step) fitted on fit-block pairs; R^2 on test pairs (t, t+1) under the same "
      "refit network. fb - ctrl: the same frozen encoder, head retrained with vs without zhat_{t+1}.")
    a("")
    a("| segment | horizon | pairs | R^2 model (per dim) | R^2 persistence (per dim) | lag-1 autocorr | gain (t) | fb - ctrl pinball (t) |")
    a("|---|---|---|---|---|---|---|---|")
    for s in SEGS:
        for h in HS:
            r = E[h][s]
            la = r["latent"]
            a(f"| {s} | {h} | {la['pairs']} | {', '.join(f(x, 2) for x in la['r2_model'])} | {', '.join(f(x, 2) for x in la['r2_persist'])} | "
              f"{', '.join(f(x, 2) for x in la['lag1_autocorr'])} | {f(la['gain'], 3)} ({f(la['gain_t'], 1)}) | {pt(r['fb_vs_ctrl'])} |")
    a("")
    a("## The polynomials inside the neurons (symbolic extraction)")
    a("")
    a(f"Network: seed-0 PNN of the {S['1h']['month']} refit (labels to the end of the previous month). Each edge function phi(x) "
      "(x = standardised, clipped input; see `std_mu` / `std_sd` in the JSON to convert) is fitted on the training rows by least-squares "
      "polynomials of degree 1-3; the closed form is the lowest degree with R^2 >= 0.99 (else degree 3). Importance = sd of phi over the "
      "training rows. A latent unit is z_j = b_j + sum_i phi_ji(x_i); a quantile is built from the head edges of the 4 latents.")
    a("")
    for h in HS:
        sy = S[h]
        a(f"### {h}")
        a("")
        sh = sy["share_edges_poly_r2_099"]
        a(f"Share of encoder edges that are polynomials with R^2 >= 0.99: degree 1 {f(sh['1'], 2)}, <= 2 {f(sh['2'], 2)}, <= 3 {f(sh['3'], 2)}.")
        a("")
        a("Input importance (summed edge sd into the latents): " + ", ".join(f"{k} {f(v, 3)}" for k, v in list(sy["input_importance"].items())[:8]) + ".")
        a("")
        a("What each latent tracks (Spearman with inputs, training rows): " + "; ".join(
            f"{z}: " + ", ".join(f"{n} {f(c, 2)}" for n, c in tr[:3]) for z, tr in sy["latent_tracks"].items()) + ".")
        a("")
        a("| edge | importance | degree | R^2 (deg 1 / 2 / 3) | closed form |")
        a("|---|---|---|---|---|")
        for r in sy["encoder_edges"][:12]:
            a(f"| {r['in']} -> {r['out']} | {f(r['importance'], 3)} | {r['deg']} | {' / '.join(f(x, 3) for x in r['r2_by_deg'])} | {r['formula']} |")
        a("")
        a("Head edges (latent -> raw head output; o50 = median, o05..o25 / o75..o95 = log-softplus increments):")
        a("")
        a("| edge | importance | degree | R^2 (deg 1 / 2 / 3) | closed form |")
        a("|---|---|---|---|---|")
        for r in sy["head_edges"][:10]:
            a(f"| {r['in']} -> {r['out']} | {f(r['importance'], 3)} | {r['deg']} | {' / '.join(f(x, 3) for x in r['r2_by_deg'])} | {r['formula']} |")
        a("")
    a("## Artifacts (dashboard contract)")
    a("")
    sc = o["schema"]
    a(f"{sc['file']}.")
    a("")
    a("| column | meaning |")
    a("|---|---|")
    for k, v in sc["columns"].items():
        a(f"| {k} | {v} |")
    a("")
    a(f"Frozen models: {sc['frozen']}. Rows / bytes: " + "; ".join(f"{h} {o['artifacts'][h]['rows']} rows, {o['artifacts'][h]['bytes'] // 1024} KiB "
                                                                 f"(latest refit {o['artifacts'][h]['latest_refit']})" for h in HS) + ".")
    a("Per-refit checkpoints: `data/models/pnn/wf_<h>/<YYYY-MM>.{npz,pt}`.")
    a("")
    a("## Rule candidates")
    a("")
    a("```yaml")
    a("rules:")
    L.extend(_rules(o))
    a("```")
    a("")
    path.write_text("\n".join(L) + "\n", encoding="utf-8")


__all__ = ["write_md"]
