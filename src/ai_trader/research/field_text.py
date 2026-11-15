"""Text blocks for the BTC field report: plain-words summary, the coefficient-file schema and rule candidates.
The wording of the rule candidates was chosen after reading the 2026-11-13 run; numbers are filled from the results."""

from __future__ import annotations

import math

import numpy as np

from ai_trader.research import field_study as fs


def _g(o, seg, key, h, f="diff"):
    return o["segments"][seg][key][h].get(f, float("nan"))


def _t(x, d=2):
    return "-" if x is None or not np.isfinite(x) else f"{x:+.{d}f}"


def _raw_coef(fc: dict, comp: str, name: str) -> float | None:
    """Coefficient of a linear term converted to raw units (per unit of the coordinate)."""
    sd = dict(zip(fc["standardizer"]["names"], fc["standardizer"]["sd"], strict=True))
    for t in fc["terms"][comp]:
        if t["term"] == name:
            return t["coef"] / sd[name]
    return None


TERM_WORDS = {
    "s": "hour of week: the variance follows the hour-of-week factor, amplified (raw slope above 2 on s = 0.5 ln f, so the smoothed, shrunk profile understates the swing)",
    "x1": "a short-term vol shock (fast EWMA above the ~6-day one) lowers the next-hour variance relative to the fast EWMA: the fast EWMA overshoots and the field pulls it back",
    "x2": "the same pull-back one scale up (6-day vol above the ~41-day level)",
    "L": "the long-run vol level",
    "L*s": "the hour-of-week swing is smaller when the long-run vol level is high",
    "m24*dir": "when 24h momentum and the multi-day trend score agree (a running trend, up or down), variance and jump risk rise; sign-symmetric, so no direction content",
    "jx": "recent jump excitation",
    "dir^2": "trend strength (either sign)",
}


def term_words(fc: dict) -> list[str]:
    out = []
    for comp, lab in (("logD", "diffusion"), ("loglam", "jump intensity")):
        for t in fc["terms"][comp]:
            if t["term"] == "1":
                continue
            w = TERM_WORDS.get(t["term"], "a state interaction")
            out.append(f"  - {lab}, {t['term']} ({t['coef']:+.3f} per sd, exp = x{math.exp(t['coef']):.2f} per +1 sd): {w}.")
    return out


def narrative(o: dict) -> list[str]:
    fc = o["field_cutoff"]
    v = o["verdicts"]
    ins, ho = "llm_in_sample", "clean_holdout"
    s_raw = _raw_coef(fc, "logD", "s")
    terms_D = sorted([t for t in fc["terms"]["logD"] if t["term"] != "1"], key=lambda t: -abs(t["t"] or 0))
    terms_L = [t for t in fc["terms"]["loglam"] if t["term"] != "1"]
    base_p = math.exp(next(t["coef"] for t in fc["terms"]["loglam"] if t["term"] == "1"))
    L = ["## Summary in plain words", ""]
    L.append(f"- **Verdicts.** H-FLD1 {v['H-FLD1'][ins]} / {v['H-FLD1'][ho]}; H-FLD2 (drift null) {v['H-FLD2'][ins]} / {v['H-FLD2'][ho]}; "
             f"H-FLD3 (online vs frozen) {v['H-FLD3'][ho]} (in-sample / holdout).")
    L.append("- **H-FLD1 numbers** (field minus log-HAR-t + constant Kou, nats per origin, NW t): " + "; ".join(
        f"{seg}: " + ", ".join(f"{h}h {_t(_g(o, seg, 'fld1', h), 4)} (t {_t(_g(o, seg, 'fld1', h, 't'))})" for h in fs.HORIZONS) for seg in (ins, ho)) + ".")
    L.append("- **Against the calendar baseline** B1 (log-HAR-t + hour-of-week, not the registered comparator): " + "; ".join(
        f"{seg}: " + ", ".join(f"{h}h {_t(_g(o, seg, 'vs_B1', h), 4)} (t {_t(_g(o, seg, 'vs_B1', h, 't'))})" for h in fs.HORIZONS) for seg in (ins, ho))
        + ". Whatever the field gains over B0 beyond these numbers is the known hour-of-week shape (H-NS4).")
    L.append(f"- **The diffusion equation.** ln D = ln vf + a polynomial with {len(terms_D)} surviving terms (stability selection, 220 candidates). "
             + (f"The hour-of-week coordinate enters with a raw coefficient {s_raw:.2f} on s = 0.5 ln f (2 would mean the variance scales one-for-one "
                f"with the hour-of-week factor). " if s_raw is not None else "")
             + "Largest terms by |t|: " + ", ".join(f"{t['term']} ({t['coef']:+.3f}, t {t['t']:+.1f})" for t in terms_D[:6]) + ".")
    L.append(f"- **The jump equation.** Baseline jump probability {100 * base_p:.2f}% per hour at the average state; "
             + ("terms: " + ", ".join(f"{t['term']} ({t['coef']:+.3f}, t {_t(t['t'], 1)})" for t in terms_L) if terms_L else "no state term survived")
             + f". Jump sizes (Kou, diffusion-sd units): up share {fc['kou']['p_up']:.2f}, mean up {fc['kou']['mean_up_sd']:.2f} sd, mean down "
             f"{fc['kou']['mean_down_sd']:.2f} sd (the floor is 3 sd). Off-jump innovations are Student-t with df {fc['t_dof']:.1f}.")
    L.append("- **The terms in words** (cutoff fit; each coefficient multiplies the variance or the jump probability by exp(coef) per +1 sd of the term):")
    L += term_words(fc)
    mu_terms = [t for t in fc["terms"]["mu"] if t["term"] != "1"]
    L.append("- **The drift equation.** " + ("Only a constant survived: " if not mu_terms else f"{len(mu_terms)} state terms survived: ")
             + fc["equations"]["mu"] + ". H-FLD2 tests whether it carries direction skill.")
    vf = o["vol_field"]
    L.append(f"- **The vol field.** The fast log-vol coordinate mean-reverts with half-life {vf['ou_v']['half_life_h']:.0f} h (OU, finite-dt corrected); "
             f"its increments are uncorrelated with the sign of the return (corr {vf['corr_z_dv_all']:+.3f}) and correlated with its size "
             f"(corr with |z| {vf['corr_absz_dv']:+.3f}): no leverage effect, consistent with H-LEV2.")
    f3 = o["fld3"]
    L.append("- **Online updating** (H-FLD3, holdout): " + ", ".join(f"{h}h {_t(f3[h]['diff'], 4)} (t {_t(f3[h]['t'])})" for h in fs.HORIZONS) + ".")
    return L


def schema_doc() -> list[str]:
    return ["## Coefficient files (data/models/field/) for F2 and the dashboard", "",
            "- `index.json`: names of the frozen cutoff field, the latest field and the monthly history; loader `ai_trader.field.model.Field.load(path)`.",
            "- `btc_field_cutoff_2026-07-01.json` (frozen, H-FLD3 prior) and `btc_field_<as_of>.json` (latest; cutoff support, coefficients through as_of). Schema `field.v1`:",
            "  - `meta`: fit_as_of, train_start, n_train, jump_share, em_iters, loglik_per_bar (and support_from for the latest file).",
            "  - `state.config`: EWMA lambdas (lam_fast / lam_med / lam_slow), clip (sd), jx_half_life (h), m24_bars, fund_window, oi_lag.",
            "  - `state.coords`: [x1, x2, L, s, jx, m24, fz, oi, dir] (definitions in ai_trader/field/state.py); `state.standardizer`: names, mean, sd "
            "(x~ = (x - mean) / sd); `state.hour_of_week_profile`: 168 variance factors (Monday 00 UTC = 0).",
            "  - `components.{logD, mu, loglam}`: link, family, offset, phi (Pearson dispersion), `terms` = list of {exponents: 9 ints (power of each "
            "standardised coord), coef, se (robust)}, `cov` (model-based covariance, the online prior), `equation` (readable).",
            "  - `law`: one line; `core` ('t'); `t_dof`; `kou`: p_up, eta_up, eta_down (rates in diffusion-sd units), mean_up_sd, mean_down_sd.",
            "  - `vol_field`: descriptive KM of v = 0.5 ln vf (OU fits, equations of E[dv], D_v, rho(r, dv)).",
            "  - Evaluate: D = vf * exp(sum_k coef_k prod_i x~_i^e_ik); mu = sqrt(D) * P_mu(x~); p_jump = min(exp(P_lam(x~)), 0.5); "
            "propagate with `ai_trader.field.propagate.simulate` (it moves vf/vm/vs/jx/m24 inside each path with `state.step`).",
            "- `btc_field_monthly.json` (`field.monthly.v1`): per monthly refit the terms (exponents, coef, se) of the three components, Kou, standardizer.", ""]


def rule_candidates(o: dict) -> str:
    ins, ho = "llm_in_sample", "clean_holdout"
    fc = o["field_cutoff"]
    s_raw = _raw_coef(fc, "logD", "s")
    r = []

    def rule(i, kind, text, when, then, effect, conf, ev, horizon="1h-1d"):
        r.append("\n".join([
            f'- id: "R-FLD-{i}"', f'  kind: "{kind}"', f'  rule: "{text}"', f'  when: "{when}"', f'  then: "{then}"', "  scope:",
            '    assets: "BTCUSDT perp (Bybit)"', f'    horizon: "{horizon}"', '    sides: "both"', f'  effect: "{effect}"', f'  confidence: "{conf}"',
            f'  evidence: [{", ".join(chr(34) + e + chr(34) for e in ev)}]',
            '  review: "re-run research.field_study when the holdout passes 180 days or the state gains delta / news coordinates"', '  status: "provisional"']))

    def lst(key, seg):
        return ", ".join(f"{h}h {_t(_g(o, seg, key, h), 3)} (t {_t(_g(o, seg, key, h, 't'), 1)})" for h in fs.HORIZONS)

    v1 = o["verdicts"]["H-FLD1"]
    conf1 = "B" if v1[ins] == "supports" and v1[ho] == "supports" else "C"
    rule(1, "prefer", "WHEN a BTC return distribution for the next 1h-4h is needed for targets, stops or quote widths THEN prefer a calendar-aware density "
         "(the probability field, or log-HAR-t scaled by the hour-of-week factor) over plain log-HAR-t with a constant jump rate",
         "building p_tp / range / quantile inputs for BTC from the state at a bar close",
         "read 1h quantiles from ai_trader.field (closed form) or from the calendar-scaled log-HAR-t; they are level at 1h (holdout)",
         f"field minus log-HAR-t + Kou, nats: in-sample {lst('fld1', ins)}; holdout {lst('fld1', ho)}; field minus calendar baseline: in-sample {lst('vs_B1', ins)}; holdout {lst('vs_B1', ho)}",
         conf1, ["H-FLD1", "H-NS4", f"research/reports/btc_field_{fs.REPORT_DATE}.md"])
    rule(5, "prefer", "WHEN a BTC range or touch probability over 4h-1d is needed THEN prefer the calendar-scaled log-HAR-t direct-horizon density over the field's Monte-Carlo, and expect the field's multi-hour ranges to run wide",
         "multi-hour targets, range estimates or quote widths for BTC",
         "use a direct-horizon log-HAR-t with the hour-of-week factor for 4h-1d widths; read field cards at 4h-1d as an upper bound on the range",
         f"field minus calendar baseline: in-sample {lst('vs_B1', ins)}; holdout {lst('vs_B1', ho)}; field central-50% coverage at 4h "
         f"{o['segments'][ins]['pit'][4]['field']['coverage']['50']:.3f} / {o['segments'][ho]['pit'][4]['field']['coverage']['50']:.3f} and at 1d "
         f"{o['segments'][ins]['pit'][24]['field']['coverage']['50']:.3f} / {o['segments'][ho]['pit'][24]['field']['coverage']['50']:.3f} (in-sample / holdout; nominal 0.50)",
         "C", ["H-FLD1", "H-VM5", "R-VOL-21", f"research/reports/btc_field_{fs.REPORT_DATE}.md"], horizon="4h-1d")
    rule(2, "expect", "WHEN reading the BTC field's variance THEN expect it to scale with the hour-of-week factor and to fall back after a vol shock, not to depend on the sign of recent moves",
         "interpreting the diffusion polynomial or a distribution card",
         "explain a card's width by vol level, hour of week and recent shocks; do not read the drift or the width as a direction call",
         (f"raw coefficient of s = 0.5 ln f in ln D: {s_raw:.2f} (2 = one-for-one); " if s_raw is not None else "")
         + f"corr(sign-free vol increment, return sign) {o['vol_field']['corr_z_dv_all']:+.3f}; fast log-vol half-life {o['vol_field']['ou_v']['half_life_h']:.0f} h",
         "B", ["H-FLD1", "H-LEV2", "H-NS4", f"research/reports/btc_field_{fs.REPORT_DATE}.md"])
    v2 = o["verdicts"]["H-FLD2"]
    b = {seg: {h: o["segments"][seg]["brier"][h] for h in (1, 24)} for seg in (ins, ho)}
    rule(3, "avoid", "WHEN the BTC field's drift or P(up) differs from one half THEN avoid reading it as a direction signal",
         "a distribution card or field output shows a non-zero drift or P(close up) away from 0.5",
         "use the field for widths, tails and touch probabilities only; direction stays with the paired baselines",
         "Brier skill of the field's P(R > 0) vs the base rate: " + "; ".join(
             f"{seg} 1h {_t(b[seg][1].get('bss'), 4)} [{_t(b[seg][1].get('ci90', [np.nan] * 2)[0], 4)}, {_t(b[seg][1].get('ci90', [np.nan] * 2)[1], 4)}], "
             f"1d {_t(b[seg][24].get('bss'), 4)} [{_t(b[seg][24].get('ci90', [np.nan] * 2)[0], 4)}, {_t(b[seg][24].get('ci90', [np.nan] * 2)[1], 4)}]" for seg in (ins, ho))
         + f"; H-FLD2 {v2[ins]} / {v2[ho]}",
         "A" if v2[ins] == "supports" and v2[ho] == "supports" else "B", ["H-FLD2", "R-ML-5", f"research/reports/btc_field_{fs.REPORT_DATE}.md"])
    f3 = o["fld3"]
    v3 = o["verdicts"]["H-FLD3"][ho]
    if v3 == "supports":
        rule(4, "prefer", "WHEN using the BTC field after its fit date THEN prefer the online-updated coefficients over the frozen fit",
             "the field is used live or forward", "run ai_trader.field.online.run_online from the frozen fit each hour",
             "online minus frozen, holdout: " + ", ".join(f"{h}h {_t(f3[h]['diff'], 4)} (t {_t(f3[h]['t'], 1)})" for h in fs.HORIZONS), "C",
             ["H-FLD3", f"research/reports/btc_field_{fs.REPORT_DATE}.md"])
    else:
        rule(4, "expect", "WHEN tempted to update the BTC field's coefficients hourly THEN expect no gain over the frozen fit; its coefficients barely move through the holdout",
             "deciding how often to refresh the field", "refit monthly with the expanding window; keep the online filter as a monitor of coefficient drift only",
             "online (RLS, 60-day memory) minus frozen, holdout: " + ", ".join(f"{h}h {_t(f3[h]['diff'], 4)} (t {_t(f3[h]['t'], 1)})" for h in fs.HORIZONS)
             + f"; H-FLD3 {v3}", "C", ["H-FLD3", f"research/reports/btc_field_{fs.REPORT_DATE}.md"])
    r.sort(key=lambda b: int(b.split('"')[1].split("-")[-1]))
    return "\n".join(r)


__all__ = ["narrative", "rule_candidates", "schema_doc"]
