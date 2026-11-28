"""WMS tables (markdown) from data/research/wms/results.json, nulls.json and bounce.json -> data/research/wms/tables.md.

    uv run python -m ai_trader.research.wms_report
"""

from __future__ import annotations

import json

from ai_trader.research.wms_data import root

SEG = ("llm_in_sample", "clean_holdout")


def f(x, d=3):
    if x is None:
        return "n/a"
    if isinstance(x, (int,)) and not isinstance(x, bool):
        return str(x)
    try:
        return f"{x:+.{d}f}"
    except (TypeError, ValueError):
        return str(x)


def ic_cells(r: dict) -> list[str]:
    out = []
    for s in SEG:
        d = r.get(s, {})
        out.append(f"{f(d.get('mean'))} (t {f(d.get('t'), 2)}, {d.get('weeks', d.get('n', ''))} wk)")
    return out


def main() -> None:
    R = json.loads((root() / "results.json").read_text())
    N = json.loads((root() / "nulls.json").read_text()) if (root() / "nulls.json").exists() else {}
    L = []
    u = R["universe"]
    L.append("## Universe\n")
    L.append("| segment | symbol-weekends | weekends | symbols |\n|---|---|---|---|")
    for s in sorted(u["segments"], key=lambda x: x["segment"]):
        L.append(f"| {s['segment']} | {s['rows']} | {s['weekends']} | {s['symbols']} |")
    L.append(f"\nBand rows {u['band_rows']}; dropped for coverage {u['dropped_cov']}; for zero turnover {u['dropped_zero_turnover']}; "
             f"median names per weekend {u['per_weekend_median']}; symbols {u['symbols']} (delisted today: {u['delisted_symbols']}).\n")
    h = R["loghar"]
    L.append(f"Weekend panel log-HAR (FIT, n {h['n_fit']}): {', '.join(f'{k} {v:+.3f}' for k, v in h['coef'].items())}; R2 {h['r2']:.3f}; "
             f"QLIKE {h['qlike_fit_loghar']:.3f} vs scaled 7-day RV {h['qlike_fit_naive_scaled']:.3f}.\n")
    L.append("## H-WMS-1 / 2: persistence (Spearman IC Friday index vs weekend index)\n")
    L.append("| index | FIT IC | holdout IC | FIT BH q | verdict |\n|---|---|---|---|---|")
    r = R["H-WMS-1"]
    L.append(f"| **S0** | {' | '.join(ic_cells(r))} | | {r['verdict']} |")
    for k, v in R["H-WMS-2"]["indices"].items():
        L.append(f"| {k} | {' | '.join(ic_cells(v))} | {f(v['q_fit'], 4)} | {v['verdict']} |")
    L.append(f"\nPanel slope S0_w on S0 (weekend FE, clustered): FIT {r['panel_fit']}, holdout {r['panel_hold']}\n")
    L.append("## H-WMS-3: S0 vs weekend outcomes (IC)\n")
    L.append("| outcome | FIT IC | holdout IC | FIT BH q | panel FIT b (t) | panel holdout b (t) |\n|---|---|---|---|---|---|")
    for o, v in R["H-WMS-3"]["outcomes"].items():
        pf, ph = v.get("panel_fit", {}), v.get("panel_hold", {})
        L.append(f"| {o} | {' | '.join(ic_cells(v))} | {f(v['q_fit'], 4)} | {f(pf.get('b'))} ({f(pf.get('t'), 2)}) | {f(ph.get('b'))} ({f(ph.get('t'), 2)}) |")
    L.append(f"\nVerdict H-WMS-3: {R['H-WMS-3']['verdict']}\n")
    L.append("## H-WMS-4: single indices vs O4 (trend ratio)\n")
    L.append("| index | FIT IC | holdout IC | FIT BH q | verdict |\n|---|---|---|---|---|")
    for k, v in R["H-WMS-4"]["indices"].items():
        L.append(f"| {k} | {' | '.join(ic_cells(v))} | {f(v['q_fit'], 4)} | {v['verdict']} |")
    L.append(f"\nFIT-passing: {R['H-WMS-4']['fit_passing']}; S1: {R['H-WMS-4'].get('S1')}; verdict {R['H-WMS-4']['verdict']}\n")
    L.append("## H-WMS-5: rule (top third of the score) minus the rest, per weekend\n")
    L.append("| outcome | FIT diff (t) | holdout diff (t) |\n|---|---|---|")
    for o, v in R["H-WMS-5"]["diffs"].items():
        L.append(f"| {o} | {f(v['llm_in_sample']['mean'])} ({f(v['llm_in_sample']['t'], 2)}) | {f(v['clean_holdout']['mean'])} ({f(v['clean_holdout']['t'], 2)}) |")
    L.append(f"\nVerdict {R['H-WMS-5']['verdict']}; rule {R['rule']}\n")
    L.append("## H-WMS-6: weekend BoS continuation (signed return / sigma; bp)\n")
    L.append("| test | FIT mean (t) | FIT bp | FIT events | BH q | holdout mean (t) | holdout bp | holdout events |\n|---|---|---|---|---|---|---|---|")
    for k, v in R["H-WMS-6"]["tests"].items():
        a, b = v["llm_in_sample"], v["clean_holdout"]
        L.append(f"| {k} | {f(a['mean'])} ({f(a['t'], 2)}) | {f(a.get('bp', {}).get('mean'), 1)} | {a.get('events')} | {f(v['q_fit'], 4)} | "
                 f"{f(b['mean'])} ({f(b['t'], 2)}) | {f(b.get('bp', {}).get('mean'), 1)} | {b.get('events')} |")
    L.append(f"\nVerdict {R['H-WMS-6']['verdict']}; events per symbol-weekend {R['H-WMS-6']['events_per_symbol_weekend']}\n")
    L.append("## H-WMS-7: BTC weekend log return\n")
    L.append("| segment | mean (t) | median | share > 0 (binom p) | weekday 2-day rate | weekend - weekday (t) | MDE |\n|---|---|---|---|---|---|---|")
    for s in SEG:
        v = R["H-WMS-7"][s]
        L.append(f"| {s} | {f(v['mean'], 4)} ({f(v['t'], 2)}) | {f(v['median'], 4)} | {v['share_pos']:.3f} ({v['binom_p']:.3f}) | "
                 f"{f(v['weekday_rate_2d']['mean'], 4)} | {f(v['paired_vs_weekday_rate']['mean'], 4)} ({f(v['paired_vs_weekday_rate']['t'], 2)}) | {f(v.get('mde'), 4)} |")
    L.append(f"\nVerdict {R['H-WMS-7']['verdict']}\n")
    L.append("## H-WMS-8: alt minus BTC weekend log spread (equal weight over U_W)\n")
    L.append("| segment | mean (t) | median of weekly medians (t) | Tue-Thu mean (t) | weekend - Tue-Thu (t) | weeks | share of weeks < 0 |\n|---|---|---|---|---|---|---|")
    for s in SEG:
        v = R["H-WMS-8"][s]
        L.append(f"| {s} | {f(v['mean']['mean'], 4)} ({f(v['mean']['t'], 2)}) | {f(v['median']['mean'], 4)} ({f(v['median']['t'], 2)}) | "
                 f"{f(v['weekday_tt']['mean'], 4)} ({f(v['weekday_tt']['t'], 2)}) | {f(v['paired_we_minus_tt']['mean'], 4)} ({f(v['paired_we_minus_tt']['t'], 2)}) | "
                 f"{v['weeks']} | {f(v['share_weeks_neg'], 3)} |")
    L.append(f"\nVerdict {R['H-WMS-8']['verdict']}\n")
    L.append("## H-WMS-9 / 10: concentration (top-5% share of demeaned s^2) and skew\n")
    L.append("| segment | share weekend | share Tue-Thu | paired (t) | skew weekend (t) | skew Tue-Thu | paired skew (t) |\n|---|---|---|---|---|---|---|")
    for s in SEG:
        a, b = R["H-WMS-9"][s], R["H-WMS-10"][s]
        L.append(f"| {s} | {f(a['we']['mean'])} | {f(a['tt']['mean'])} | {f(a['paired']['mean'])} ({f(a['paired']['t'], 2)}) | "
                 f"{f(b['we']['mean'])} ({f(b['we']['t'], 2)}) | {f(b['tt']['mean'])} | {f(b['paired']['mean'])} ({f(b['paired']['t'], 2)}) |")
    L.append(f"\nGaussian share 0.279. Verdicts H-WMS-9 {R['H-WMS-9']['verdict']}, H-WMS-10 {R['H-WMS-10']['verdict']}\n")
    L.append("## H-WMS-11: picking the top-5% weekend movers (AUC within weekends)\n")
    t = R["H-WMS-11"]
    L.append(f"Logistic: FIT leave-one-year-out AUC {t['cv_auc'][0]:.3f} [{t['cv_auc'][1]:.3f}, {t['cv_auc'][2]:.3f}]; holdout AUC "
             f"{t['hold_auc'][0]:.3f} [{t['hold_auc'][1]:.3f}, {t['hold_auc'][2]:.3f}]; n FIT {t['n_fit']}, holdout {t['n_hold']}. Verdict {t['verdict']}.\n")
    L.append("| input | FIT AUC [95%] | BH q | holdout AUC [95%] | coef |\n|---|---|---|---|---|")
    for k, v in t["single"].items():
        a, b = v["fit"], v["hold"]
        L.append(f"| {k} | {a[0]:.3f} [{a[1]:.3f}, {a[2]:.3f}] | {f(v['q_fit'], 4)} | {b[0]:.3f} [{b[1]:.3f}, {b[2]:.3f}] | {f(t['coef'].get(k))} |")
    L.append(f"\nDirection of the top-5% movers: {t['direction']}\n")
    L.append("## H-WMS-12: BTC calm-weekend classifier\n")
    c = R["H-WMS-12"]
    for k in ("cv", "cv_persistence_only", "holdout"):
        v = c.get(k, {})
        if v:
            L.append(f"- {k}: Brier {v['brier']:.4f} vs climatology {v['brier_clim']:.4f} (skill {f(v['skill'])}), DM t {f(v['dm_t'], 2)} "
                     f"(HLN {f(v['dm_t_hln'], 2)}), log score {v['logscore']:.4f} vs {v['logscore_clim']:.4f}, n {v['n']}, base rate {v['base_rate']:.3f}")
    L.append(f"- calibration (CV): intercept {c['calibration']['intercept']:+.3f}, slope {c['calibration']['slope']:.3f}; bins "
             + "; ".join(f"p {b['p_mean']:.2f} -> {b['y_mean']:.2f} (n {b['n']})" for b in c["calibration"]["bins"]))
    L.append("- coefficients (standardised; t): " + ", ".join(f"{k} {v:+.2f} ({c['coef_t'].get(k, 0):+.1f})" for k, v in c["coef"].items()))
    L.append(f"- verdict {c['verdict']}; FIT pass {c['fit_pass']}")
    L.append(f"- holidays: {c['holiday_summary']}\n")
    L.append("| holiday weekend | kind | calm | P(calm) | weekend RV / median52 | 5-sigma jumps |\n|---|---|---|---|---|---|")
    for x in c["holidays"]:
        ratio = (x["rv_we"] / x["med_rv52"]) if (x.get("rv_we") and x.get("med_rv52")) else None
        L.append(f"| {x['weekend_start']} | {x['holiday']} | {x.get('calm')} | {f(x.get('p'), 2)} | {f(ratio, 2)} | {x.get('n_jump5')} |")
    if R.get("forward"):
        fw = R["forward"]
        L.append("\n## Forward weekend (Friday-known at Fri 22:00 UTC)\n")
        L.append(f"P(calm) {f(fw.get('calm_p'))}; inputs {fw.get('inputs')}; percentiles "
                 + ", ".join(f"{k} {v:.2f}" for k, v in fw.items() if k.startswith("pct_") or k.endswith("_pct") and v is not None))
        L.append(f"\nUniverse {fw['n_universe']} of {fw['n_band']} in band; eligible_name (top third of the score):\n")
        L.append("| symbol | score | median turnover (M USD) | vr15_4 | vr15_16 | ar1_15 | dfa | er_1h | bos_imb |\n|---|---|---|---|---|---|---|---|---|")
        for x in fw["eligible_name"]:
            L.append(f"| {x['symbol']} | {x['score']:+.2f} | {x['med_turnover'] / 1e6:.1f} | {x['vr15_4']:.2f} | {x['vr15_16']:.2f} | {x['ar1_15']:+.3f} | "
                     f"{x['dfa']:.2f} | {x['er_1h']:.3f} | {x['bos_imb']:.2f} |")
        L.append(f"\neligible (with the gate if enabled): {fw['eligible']}\n")
    if N:
        L.append("## Index null distributions (2016-bar window; 1000 simulations)\n")
        L.append("| index | RW mean (sd) | RW + 8 bp bounce mean (sd) | OU half-life 2 h mean (sd) |\n|---|---|---|---|")
        for k in N["rw"]:
            if k.startswith("z") or k == "bos_net":
                continue
            L.append(f"| {k} | {N['rw'][k]['mean']:.3f} ({N['rw'][k]['sd']:.3f}) | {N['rw_bounce'][k]['mean']:.3f} ({N['rw_bounce'][k]['sd']:.3f}) | "
                     f"{N['ou_2h'][k]['mean']:.3f} ({N['ou_2h'][k]['sd']:.3f}) |")
    if "properties" in R and "series" in R["properties"]:
        L.append("\n## Properties of the weekly series (FIT)\n")
        L.append("| series | n | mean | sd | skew | exc. kurt | ADF p | KPSS p | LB(4) p | LB(4) sq p |\n|---|---|---|---|---|---|---|---|---|---|")
        for k, v in R["properties"]["series"].items():
            L.append(f"| {k} | {v['n']} | {v['mean']:+.4f} | {v['sd']:.4f} | {v['skew']:+.2f} | {v['kurt_excess']:.2f} | {v['adf_p']:.3f} | {v['kpss_p']:.3f} | "
                     f"{v['lb4_p']:.3f} | {v['lb4_sq_p']:.3f} |")
        L.append("\n| panel variable (FIT) | mean | sd | p05 | p50 | p95 | skew | exc. kurt |\n|---|---|---|---|---|---|---|---|")
        for k, v in R["properties"]["panel_fit"].items():
            L.append(f"| {k} | {v['mean']:+.3f} | {v['sd']:.3f} | {v['p05']:+.3f} | {v['p50']:+.3f} | {v['p95']:+.3f} | {v['skew']:+.2f} | {v['kurt_excess']:.2f} |")
    (root() / "tables.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
