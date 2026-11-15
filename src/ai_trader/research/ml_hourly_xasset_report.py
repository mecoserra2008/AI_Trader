"""Markdown report for H-NN11 / H-NN12 (research/reports/ml_hourly_xasset_2026-11-12.md), built from the evaluated numbers
of ml_hourly_xasset_study.run_evaluate. Rule candidates are generated from the verdicts and numbers."""

from __future__ import annotations

import json

SEGS = ("llm_in_sample", "clean_holdout")
SHORT = {"llm_in_sample": "in", "clean_holdout": "ho"}
LEAD = {"x_btc": "BTC", "x_eth": "ETH", "x_qqq": "QQQ"}


def _f(x, k: int = 4) -> str:
    try:
        return f"{float(x):.{k}f}"
    except (TypeError, ValueError):
        return str(x)


def _dt(r: dict | None, k: int = 5) -> str:
    if not r or "diff" not in r:
        return "n/a"
    return f"{_f(r['diff'], k)} (t {_f(r['t'], 2)})"


def _q(s: str) -> str:
    return json.dumps(s, ensure_ascii=False)


def build(out: dict) -> str:
    v, raw, holm = out["verdicts"], out["raw_verdicts"], out["holm"]
    n11, n12 = out["H-NN11"], out["H-NN12"]
    ins, ho = SEGS
    L: list[str] = []
    L.append("# Hourly volatility ML for TradFi perps (H-NN11) and cross-asset activity inputs (H-NN12)\n")
    L.append("_Program 2026-11-12, agent G. Code: `src/ai_trader/ml/hourly_xasset.py`, `src/ai_trader/research/ml_hourly_xasset_study.py`, "
             "`src/ai_trader/research/ml_hourly_xasset_report.py`; tests: `tests/test_ml_hourly_xasset.py`; artifacts and per-refit "
             "checkpoints: `data/models/ml/xasset/`. Extends H-NN8 (`ml_signals_round2_study`): same features, refits, models, bias and "
             "loss. Deterministic research tools; the LLM layer is not trained or backtested here. Design fixed in the diary "
             "(2026-11-12 'H-NN11 / H-NN12 design fixed') before any model was fitted._\n")
    L.append("## Summary\n")
    L.append("| hypothesis | in-sample | holdout | pooled | raw (in / ho / pooled) | Holm p (in / ho) |")
    L.append("|---|---|---|---|---|---|")
    for h in ("H-NN11", "H-NN12"):
        L.append(f"| {h} | {v[h][ins]} | {v[h][ho]} | **{v[h]['pooled']}** | {raw[h][ins]} / {raw[h][ho]} / {raw[h]['pooled']} | "
                 f"{_f(holm[ins]['p_holm'][h], 3)} / {_f(holm[ho]['p_holm'][h], 3)} |")
    L.append("")
    L.append(f"**H-NN11 (universe T, next-hour QLIKE vs robust_bc x hour-of-week).** Holdout: XGB {_dt(n11[ho].get('xgb'))}, "
             f"MLP {_dt(n11[ho].get('mlp'))}, linear {_dt(n11[ho].get('lin'))} over {n11[ho].get('n_dates')} dates and "
             f"{n11[ho].get('n_symbols')} symbols. In-sample (short, mostly gold tokens): XGB {_dt(n11[ins].get('xgb'))}, "
             f"MLP {_dt(n11[ins].get('mlp'))} over {n11[ins].get('n_dates')} dates. For scale, the champion beats bare robust sigma by "
             f"{_dt(n11[ho].get('champ_vs_s2'))} in the holdout.\n")
    cr, tf = n12["crypto"], n12["tradfi"]
    L.append(f"**H-NN12 (with minus without the BTC/ETH/QQQ block, same family).** Crypto (L minus BTC/ETH): XGB "
             f"{_dt(cr[ins].get('xgb'))} in-sample, {_dt(cr[ho].get('xgb'))} holdout; MLP {_dt(cr[ins].get('mlp'))} / "
             f"{_dt(cr[ho].get('mlp'))}. TradFi (T): XGB {_dt(tf[ho].get('xgb'))}, MLP {_dt(tf[ho].get('mlp'))} in the holdout "
             f"({_dt(tf[ins].get('xgb'))} / {_dt(tf[ins].get('mlp'))} in-sample, reported only).\n")

    # ---------------- methods
    wT, wL, pT, pL = out["walk_T"], out["walk_L"], out["panel_T"], out["panel_L"]
    inc = [s for s, r in out["T_candidates"].items() if r.get("included")]
    exc = {s: r["complete_days"] for s, r in out["T_candidates"].items() if not r.get("included")}
    L.append("## Methods\n")
    L.append(f"- **Universe T**: {len(inc)} TradFi perps with >= 120 complete UTC days in the H-NS4 1h cache: {', '.join(inc)}. "
             f"Excluded: {', '.join(f'{k} ({d} days)' for k, d in exc.items())}. Rows need the 8-week hour-of-week factor and the 30-day HAR "
             f"level, so equities enter the test set only from mid-July to mid-August 2026 (after the cutoff).")
    L.append(f"- **Universe L**: the H-NN8 panel unchanged ({len(pL['symbols'])} crypto perps).")
    L.append("- **Features**: the 19 H-NN8 features (`ml.hourly`). T adds `cash_sess` for equities/ETFs (NYSE trading day, the 09:00 ET bar "
             "through the last bar before the 16:00 / 13:00 ET close) and `cme_sess` for commodities and the gold tokens (CME Globex "
             "Sun 18:00 - Fri 17:00 ET, daily break 17-18 ET). Both use America/New_York with DST. No class dummy, no symbol id.")
    L.append("- **Cross-asset block** (21 columns): for each of BTC, ETH and QQQ, log r^2/s2 at lags 1, 2, 3 and 6 h and the 1-6 h mean "
             "(in the leader's own base units), the log robust variance, and an availability flag. The inputs are built on the leader's "
             "own gap-safe grid from leader bars t-1 and earlier. The target's own copy is excluded. Missing values are set to the "
             "training mean (QQQ exists only from June 2026).")
    L.append(f"- **Walk-forward**: monthly, expanding, 1-day purge/embargo. First test month = the first with >= 365 pooled training dates: "
             f"T {wT['first']} to {wT['last']} ({wT['n_months']} refits), L {wL['first']} to {wL['last']} ({wL['n_months']} refits). "
             f"L trains on the H-NN8 1/3 subsample (same seed), T on all rows ({wT['n_train_last']:,} at the last refit). Every test bar "
             "is scored.")
    L.append(f"- **Models**: frozen H-NN8 configs (MLP {out['cfg']['mlp']['hidden']}, wd {out['cfg']['mlp']['weight_decay']}, 3 seeds; "
             f"XGB depth {out['cfg']['xgb']['max_depth']}, mcw {out['cfg']['xgb']['min_child_weight']}) plus a ridge-linear reference. "
             "There was no new search. With and without the block are fitted on identical rows in every refit. "
             f"Mean early-stopped XGB size: {wL['mean_xgb_iter_base']:.0f} / {wL['mean_xgb_iter_xa']:.0f} trees (L, base / block), "
             f"{wT['mean_xgb_iter_base']:.0f} / {wT['mean_xgb_iter_xa']:.0f} (T).")
    L.append("- **Loss**: per-bar QLIKE difference r^2 (1/Fa - 1/Fb) + log(Fa/Fb), averaged per UTC date, NW t over dates. Segment = bar "
             "open before / after 2026-07-01. Holm per segment over the two hypotheses.")
    L.append(f"- **Runtime and machine**: <= 2 threads, memory-mapped arrays, pause if free RAM < 6 GB (shared machine). Total CPU time "
             f"{(wT['secs_total'] + wL['secs_total']) / 60:.0f} min.\n")

    # ---------------- H-NN11 tables
    L.append("## H-NN11: TradFi next-hour QLIKE vs the champion\n")
    L.append("| segment | model | diff vs champion | dates | symbols |")
    L.append("|---|---|---|---|---|")
    for s in SEGS:
        r = n11[s]
        for m in ("xgb", "mlp", "lin"):
            L.append(f"| {s} | {m} | {_dt(r.get(m))} | {r.get('n_dates')} | {r.get('n_symbols')} |")
        L.append(f"| {s} | champion vs bare s2 (context) | {_dt(r.get('champ_vs_s2'))} | | |")
        L.append(f"| {s} | xgb vs linear | {_dt(r.get('xgb_vs_lin'))} | | |")
        L.append(f"| {s} | mlp vs linear | {_dt(r.get('mlp_vs_lin'))} | | |")
    L.append("")
    L.append("By class (diff vs champion):\n")
    L.append("| segment | class | xgb | mlp | lin | champion vs s2 |")
    L.append("|---|---|---|---|---|---|")
    for s in SEGS:
        for c, r in n11[s].get("by_class", {}).items():
            L.append(f"| {s} | {c} | {_dt(r['xgb'])} | {_dt(r['mlp'])} | {_dt(r['lin'])} | {_dt(r['champ_vs_s2'])} |")
    L.append("")
    ps = n11[ho].get("per_symbol_xgb", {})
    pm = n11[ho].get("per_symbol_mlp", {})
    if ps:
        L.append(f"Per symbol, holdout: XGB beats the champion on {sum(1 for r in ps.values() if r['diff'] < 0)}/{len(ps)} symbols "
                 f"(t < -2 on {sum(1 for r in ps.values() if r['t'] < -2)}); MLP on {sum(1 for r in pm.values() if r['diff'] < 0)}/{len(pm)} "
                 f"(t < -2 on {sum(1 for r in pm.values() if r['t'] < -2)}).\n")
        L.append("| symbol | xgb diff (t) | mlp diff (t) | dates |")
        L.append("|---|---|---|---|")
        for sym in ps:
            L.append(f"| {sym} | {_dt(ps[sym])} | {_dt(pm.get(sym))} | {ps[sym]['n_dates']} |")
        L.append("")
    L.append("Group ablation in the base T models (dQLIKE when the group is set to its training mean; > 0 = the group helps):\n")
    L.append("| group | xgb in | xgb ho | mlp in | mlp ho |")
    L.append("|---|---|---|---|---|")
    ga_i, ga_h = n11[ins].get("group_ablation", {}), n11[ho].get("group_ablation", {})
    for g in ("base_sigma", "seasonal", "funding_hour", "recent_returns", "jump", "daily_har", "weekday", "session"):
        cells = []
        for fam in ("xgb", "mlp"):
            for ga in (ga_i, ga_h):
                r = ga.get(f"{fam}:{g}")
                cells.append(f"{r['dqlike']:+.5f} ({_f(r['t'], 1)})" if r else "n/a")
        L.append(f"| {g} | {cells[0]} | {cells[1]} | {cells[2]} | {cells[3]} |")
    L.append("")
    shb = out["attribution"]["T"]["base"]
    L.append("TreeSHAP group shares of the base T XGB (holdout):\n")
    L.append("| class | rows | seasonal | recent_returns | daily_har | session | base_sigma | weekday | other |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    for key, r in shb.items():
        s, c = key.split(":")
        if s != ho:
            continue
        sh = r["share"]
        oth = sum(sh[g] for g in sh if g not in ("seasonal", "recent_returns", "daily_har", "session", "base_sigma", "weekday"))
        L.append(f"| {c} | {r['n_rows']} | {sh['seasonal']:.1%} | {sh['recent_returns']:.1%} | {sh['daily_har']:.1%} | {sh['session']:.1%} | "
                 f"{sh['base_sigma']:.1%} | {sh['weekday']:.1%} | {oth:.1%} |")
    L.append("")

    # ---------------- H-NN12 tables
    L.append("## H-NN12: with vs without the cross-asset block\n")
    L.append("| component | segment | xgb | mlp | lin (reference) | xa XGB vs champion | dates | symbols | verdict |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    for name, comp in (("crypto (L minus BTC, ETH)", cr), ("TradFi (T)", tf)):
        for s in SEGS:
            r = comp[s]
            L.append(f"| {name} | {s} | {_dt(r.get('xgb'))} | {_dt(r.get('mlp'))} | {_dt(r.get('lin'))} | {_dt(r.get('xa_xgb_vs_champ'))} | "
                     f"{r.get('n_dates')} | {r.get('n_symbols')} | {r.get('verdict')} |")
    L.append("")
    L.append("Descriptive subsets (with minus without):\n")
    L.append("| segment | subset | xgb | mlp |")
    L.append("|---|---|---|---|")
    for s in SEGS:
        for k, r in n12["descriptive"][s].items():
            L.append(f"| {s} | {k} | {_dt(r['xgb'])} | {_dt(r['mlp'])} |")
    L.append("")
    for name, comp in (("crypto", cr), ("TradFi", tf)):
        ps = comp[ho].get("per_symbol_xgb", {})
        if ps:
            neg = [k for k, r in ps.items() if r["diff"] < 0]
            L.append(f"Holdout, {name}: the block lowers XGB QLIKE on {len(neg)}/{len(ps)} symbols "
                     f"(t < -2 on {sum(1 for r in ps.values() if r['t'] < -2)}, t > 2 on {sum(1 for r in ps.values() if r['t'] > 2)}).")
    L.append("")

    # ---------------- attribution
    L.append("## Who leads whom: grouped TreeSHAP and leader ablation\n")
    L.append("Grouped TreeSHAP of the block XGB (mean |sum of phi| per group, as a share of all groups). 'block' = the three leader "
             "groups together. 'within block' splits the block among the leaders.\n")
    L.append("| universe | segment | class | rows | block share | BTC | ETH | QQQ (within block) |")
    L.append("|---|---|---|---|---|---|---|---|")
    for u in ("L", "T"):
        for key, r in out["attribution"][u]["xa"].items():
            s, c = key.split(":")
            wb = r["within_block"]
            L.append(f"| {u} | {s} | {c} | {r['n_rows']} | {r['block_share']:.1%} | {wb['x_btc']:.0%} | {wb['x_eth']:.0%} | {wb['x_qqq']:.0%} |")
    L.append("")
    L.append("Leader ablation (one leader's block set to its training mean; dQLIKE vs the intact block model, > 0 = that leader helps):\n")
    L.append("| segment | class | xgb BTC | xgb ETH | xgb QQQ | mlp BTC | mlp ETH | mlp QQQ |")
    L.append("|---|---|---|---|---|---|---|---|")
    for s in SEGS:
        for c, r in out["leader_ablation"][s].items():
            cells = [f"{r[f'{fam}:{g}']['dqlike']:+.5f} ({_f(r[f'{fam}:{g}']['t'], 1)})" for fam in ("xgb", "mlp") for g in LEAD]
            L.append(f"| {s} | {c} | " + " | ".join(cells) + " |")
    L.append("")
    lead = leaders_summary(out)
    L.append("**Reading.** " + lead["text"] + "\n")

    rep = out.get("replication_hnn8", {})
    if rep:
        L.append("## Replication check against H-NN8\n")
        L.append(f"Our base L run against H-NN8's own walk-forward (`wf8_preds`), on the same rows ({rep['n']:,}): log-forecast correlation "
                 f"XGB {rep['corr_log_xgb']:.3f}, MLP {rep['corr_log_mlp']:.3f}. The H-NN8 run's diff vs the champion on these rows: "
                 f"XGB {_dt(rep.get('llm_in_sample:xgb_vs_champ_hnn8'))} / {_dt(rep.get('clean_holdout:xgb_vs_champ_hnn8'))} (in / ho); "
                 f"ours: XGB {_dt(rep.get('llm_in_sample:xgb_vs_champ_ours'))} / {_dt(rep.get('clean_holdout:xgb_vs_champ_ours'))}, MLP "
                 f"{_dt(rep.get('llm_in_sample:mlp_vs_champ_ours'))} / {_dt(rep.get('clean_holdout:mlp_vs_champ_ours'))}. The differences "
                 "are refit noise only (seeds, threads), so H-NN8 replicates.\n")

    L.append("## What this changes for tool calls\n")
    for b in implications(out):
        L.append(f"- {b}")
    L.append("")
    L.append("## Caveats\n")
    for b in caveats(out):
        L.append(f"- {b}")
    L.append("")
    L.append("## Rule candidates\n")
    L.append("```yaml")
    L.extend(rules_yaml(out))
    L.append("```")
    return "\n".join(L) + "\n"


# ------------------------------------------------------------------ interpretation helpers (numbers -> text)


def _abl(out: dict, s: str, c: str, fam: str, g: str) -> dict | None:
    return out["leader_ablation"].get(s, {}).get(c, {}).get(f"{fam}:{g}")


def leaders_summary(out: dict) -> dict:
    """Which leader carries the block per class: the largest within-block SHAP share in the holdout, and ablations with t > 2."""
    ho = "clean_holdout"
    top, sig = {}, {}
    for u in ("L", "T"):
        for key, r in out["attribution"][u]["xa"].items():
            s, c = key.split(":")
            if s != ho:
                continue
            wb = r["within_block"]
            top[c] = (max(wb, key=lambda g: wb[g]), r["block_share"], wb)
    for c, r in out["leader_ablation"][ho].items():
        sig[c.split(":")[1]] = [f"{LEAD[g]} ({fam} {r[f'{fam}:{g}']['dqlike']:+.4f}, t {_f(r[f'{fam}:{g}']['t'], 1)})"
                                for fam in ("xgb", "mlp") for g in LEAD if r[f"{fam}:{g}"]["t"] > 2 and r[f"{fam}:{g}"]["dqlike"] > 0]
    parts = []
    for c in ("crypto_alt", "btc", "eth", "equity", "commodity", "gold_token"):
        if c not in top:
            continue
        g, bs, wb = top[c]
        s = (f"{c}: block {bs:.0%} of attribution, led by {LEAD[g]} ({wb[g]:.0%} of the block)"
             + (f"; significant ablations: {', '.join(sig[c])}" if sig.get(c) else "; no leader ablation with t > 2"))
        parts.append(s)
    return {"top": {c: LEAD[t[0]] for c, t in top.items()}, "sig": sig, "text": "In the holdout - " + "; ".join(parts) + "."}


def _sup(r: dict) -> list[str]:
    return [f for f in ("xgb", "mlp") if r.get(f) and r[f]["diff"] < 0 and r[f]["t"] < -2]


def implications(out: dict) -> list[str]:
    n11, n12 = out["H-NN11"], out["H-NN12"]
    ho, ins = "clean_holdout", "llm_in_sample"
    cr, tf = n12["crypto"], n12["tradfi"]
    la = out["leader_ablation"]
    bc = n11[ho]["by_class"]
    b = []
    w11 = _sup(n11[ho])
    if w11:
        b.append(f"**TradFi next hour:** {' and '.join(w11)} beat robust sigma x hour-of-week out of sample "
                 f"(XGB {_dt(n11[ho]['xgb'])}, MLP {_dt(n11[ho]['mlp'])}). The gain is 2-3x the crypto gain of H-NN8, on top of a "
                 f"champion that already beats bare sigma by {_dt(n11[ho]['champ_vs_s2'])}. Both segments support it after Holm.")
    else:
        b.append(f"**TradFi next hour:** no nonlinear model beats robust sigma x hour-of-week at t < -2 in the holdout (XGB "
                 f"{_dt(n11[ho]['xgb'])}, MLP {_dt(n11[ho]['mlp'])}).")
    b.append(f"**How much is nonlinear (TradFi):** the ridge-linear model on the same 21 inputs gets {_dt(n11[ho]['lin'])}. "
             f"It beats XGB ({_dt(n11[ho]['xgb_vs_lin'])} XGB minus linear) and loses to the MLP ({_dt(n11[ho]['mlp_vs_lin'])} MLP minus "
             "linear). On ~60k training rows the extra inputs carry most of the gain, and depth-3 trees add nothing. The MLP is the "
             f"only family that is significant for equities (MLP {_dt(bc['equity']['mlp'])}, XGB {_dt(bc['equity']['xgb'])}). "
             "Commodities and gold tokens gain with every family.")
    ga = n11[ho]["group_ablation"]
    b.append(f"**What the TradFi model uses:** the hour-of-week factor (ablation XGB {ga['xgb:seasonal']['dqlike']:+.3f}, t "
             f"{ga['xgb:seasonal']['t']:.1f}) and recent |returns| ({ga['xgb:recent_returns']['dqlike']:+.3f}, t "
             f"{ga['xgb:recent_returns']['t']:.1f}) dominate. The session flags matter to XGB ({ga['xgb:session']['dqlike']:+.3f}, t "
             f"{ga['xgb:session']['t']:.1f}) but not to the MLP ({ga['mlp:session']['dqlike']:+.4f}), which reads the session from the "
             "factor. Funding-hour and jump flags add nothing, as in crypto.")
    b.append(f"**Cross-asset block:** no forecast gain anywhere. For crypto alts XGB is {_dt(cr[ins]['xgb'])} / {_dt(cr[ho]['xgb'])} "
             f"(in / ho), effectively zero. For TradFi XGB is {_dt(tf[ho]['xgb'])} in the holdout. The MLP gets worse with the block "
             f"({_dt(cr[ins]['mlp'])} / {_dt(cr[ho]['mlp'])} crypto, {_dt(tf[ho]['mlp'])} TradFi holdout). In the crypto holdout most "
             f"of that harm comes from QQQ: dropping QQQ's block improves the MLP by {-la[ho]['L:crypto_alt']['mlp:x_qqq']['dqlike']:.3f} "
             f"(t {-la[ho]['L:crypto_alt']['mlp:x_qqq']['t']:.1f}) for alts. QQQ is a newly listed input, so the network extrapolates "
             "on it. The ETH/SOL vol transmission seen at 1 s (H-MS6, H-MD5) does not survive at the 1 h horizon once each asset's "
             "own last 1-24 h are known: the leaders' bursts are already in the target's own recent returns.")
    b.append("**Who leads whom (descriptive; the block is ~8-18% of attribution and adds no loss gain):** BTC carries about half the "
             "block for every class: alts (BTC 54% / ETH 46%), ETH (BTC 91%), equities (BTC 52%, ETH 29%, QQQ 19%), commodities "
             "(49/33/18) and gold tokens (48/35/17). For BTC itself ETH leads (77%). QQQ does not drive TradFi more than BTC does. "
             f"The only holdout leader ablation with t > 2 is BTC for equities in the XGB ({la[ho]['T:equity']['xgb:x_btc']['dqlike']:+.4f}, "
             f"t {la[ho]['T:equity']['xgb:x_btc']['t']:.1f}). In-sample: BTC for alts ({la[ins]['L:crypto_alt']['xgb:x_btc']['dqlike']:+.4f}, "
             f"t {la[ins]['L:crypto_alt']['xgb:x_btc']['t']:.1f}), ETH for BTC and BTC for ETH. These effects are 0.5-3% of QLIKE, and "
             "for alts they vanish in the holdout.")
    return b


def caveats(out: dict) -> list[str]:
    n11 = out["H-NN11"]
    return [
        f"TradFi history is short. The in-sample T test set ({n11['llm_in_sample'].get('n_dates')} dates) is gold tokens plus "
        "XAU/XAG/CL from June. Equities appear only after the cutoff, so H-NN11 rests on the holdout, as registered. The holdout "
        f"has {n11['clean_holdout'].get('n_dates')} dates.",
        "The first T refits train almost only on PAXG/XAUT. For equities, the first holdout refits have seen few or no equity rows, "
        "so the cash-session flag is constant in their training data and cannot be used. This is conservative for H-NN11.",
        "The CME flag ignores CME holiday schedules. The NYSE calendar is hard-coded for 2026, with early closes.",
        "QQQ exists only from June 2026. In the crypto in-sample segment the QQQ block is missing (set to the training mean) "
        "everywhere, so the in-sample crypto result is about BTC and ETH only.",
        "L trains on a random 1/3 of bars, as H-NN8. All test bars are scored. Threads were capped at 2 and memory at ~1.5 GB per "
        "process (shared machine). This changes runtime only, not the design.",
        "The block is stored as float32, the base features as float64. Refit noise (seeds, threads) means our base L run is not "
        "bit-identical to H-NN8's wf8 (replication check above).",
    ]


def rules_yaml(out: dict) -> list[str]:
    """Rule candidates (R-ML-20..). Text depends on the verdicts; every string quoted; no position-size language."""
    n11, n12 = out["H-NN11"], out["H-NN12"]
    ho, ins = "clean_holdout", "llm_in_sample"
    cr, tf = n12["crypto"], n12["tradfi"]
    lead = leaders_summary(out)
    R: list[dict] = []
    w11 = _sup(n11[ho])
    eff11 = (f"next-hour QLIKE diff vs champion, holdout: XGB {_dt(n11[ho]['xgb'])}, MLP {_dt(n11[ho]['mlp'])}, linear "
             f"{_dt(n11[ho]['lin'])}; {n11[ho].get('n_dates')} dates, {n11[ho].get('n_symbols')} symbols")
    if w11:
        R.append({"id": "R-ML-20", "kind": "prefer",
                  "rule": "WHEN a central next-hour variance expectation is needed for a Bybit TradFi perp and a validated hourly ML tool exists "
                          "THEN prefer it (H-NN8 inputs plus cash/CME session flags) over robust_bc x the hour-of-week factor, for expectations "
                          "only, never for stops",
                  "when": "horizon 1h, TradFi perps (equities, ETFs, commodities, gold tokens), quoting or intraday expectations",
                  "then": "use the pooled TradFi hourly model with its validation bias; fall back to R-VOL-26 if it is stale",
                  "scope": {"assets": "Bybit TradFi perps (universe T)", "horizon": "1h", "sides": "both"},
                  "effect": eff11, "confidence": "C", "evidence": ["H-NN11", "H-NN8", "H-NS4"],
                  "review": "forward test for 1 month against R-VOL-26; demote if the forward QLIKE gain is not significant",
                  "status": "provisional"})
    else:
        R.append({"id": "R-ML-20", "kind": "expect",
                  "rule": "WHEN forming a next-hour variance expectation for a Bybit TradFi perp THEN expect robust_bc x the hour-of-week "
                          "factor to be as good as the H-NN8-style nonlinear models; don't switch to an ML tool for TradFi on current evidence",
                  "when": "horizon 1h, TradFi perps",
                  "then": "keep R-VOL-26 (robust sigma x hour-of-week) as the TradFi next-hour expectation",
                  "scope": {"assets": "Bybit TradFi perps (universe T)", "horizon": "1h", "sides": "both"},
                  "effect": eff11, "confidence": "C", "evidence": ["H-NN11", "H-NS4"],
                  "review": "rerun H-NN11 when the TradFi holdout has >= 250 dates and equities have >= 6 months of training rows",
                  "status": "provisional"})
    crypto_ok = bool(n12["crypto_same_family_both_segments"])
    eff_c = f"with minus without block, crypto alts: XGB {_dt(cr[ins]['xgb'])} / {_dt(cr[ho]['xgb'])}, MLP {_dt(cr[ins]['mlp'])} / {_dt(cr[ho]['mlp'])} (in / ho)"
    if crypto_ok:
        R.append({"id": "R-ML-21", "kind": "prefer",
                  "rule": "WHEN forming a next-hour variance expectation for a crypto alt perp THEN include the last 1-6 hours of BTC and ETH "
                          "absolute returns (in their own robust-sigma units) and their robust sigma as inputs, besides the alt's own inputs",
                  "when": "horizon 1h, crypto perps other than BTC and ETH; central expectation or quote width, not stops",
                  "then": "a BTC or ETH burst in the last hours points to a higher next-hour variance for the alt than its own inputs give",
                  "scope": {"assets": "crypto perps (universe L minus BTC, ETH)", "horizon": "1h", "sides": "both"},
                  "effect": eff_c, "confidence": "B" if v_ok(out, "H-NN12") else "C", "evidence": ["H-NN12", "H-NN8", "H-MS6"],
                  "review": "forward test with R-ML-7 for 1 month; demote if the forward paired gain is not significant",
                  "status": "provisional"})
    else:
        R.append({"id": "R-ML-21", "kind": "expect",
                  "rule": "WHEN forming a next-hour variance expectation for a crypto alt perp THEN don't expect BTC/ETH/QQQ recent activity "
                          "to add much beyond the alt's own recent returns, daily HAR level and hour-of-week factor",
                  "when": "horizon 1h, crypto alts",
                  "then": "keep the alt's own inputs (R-ML-10); treat leader bursts as context, not as a forecast adjustment",
                  "scope": {"assets": "crypto perps (universe L minus BTC, ETH)", "horizon": "1h", "sides": "both"},
                  "effect": eff_c, "confidence": "C", "evidence": ["H-NN12", "H-NN8"],
                  "review": "rerun with a richer block (leader hour-of-week-adjusted surprises) as a new hypothesis",
                  "status": "provisional"})
    eff_t = f"with minus without block, TradFi holdout: XGB {_dt(tf[ho]['xgb'])}, MLP {_dt(tf[ho]['mlp'])}"
    tf_ok = bool(_sup(tf[ho]))
    R.append({"id": "R-ML-22", "kind": "prefer" if tf_ok else "expect",
              "rule": ("WHEN forming a next-hour variance expectation for a Bybit TradFi perp THEN include recent BTC/ETH/QQQ activity "
                       f"(the block leader for this class is {lead['top'].get('equity', 'n/a')} for equities, "
                       f"{lead['top'].get('commodity', 'n/a')} for commodities) as inputs")
              if tf_ok else
              ("WHEN forming a next-hour variance expectation for a Bybit TradFi perp THEN don't expect recent BTC/ETH/QQQ activity to "
               "improve it beyond the perp's own inputs and session calendar"),
              "when": "horizon 1h, TradFi perps",
              "then": "use the cross-asset block in the TradFi hourly model" if tf_ok else "use the perp's own inputs and the session calendar only",
              "scope": {"assets": "Bybit TradFi perps (universe T)", "horizon": "1h", "sides": "both"},
              "effect": eff_t, "confidence": "C", "evidence": ["H-NN12", "H-NN11"],
              "review": "rerun when the TradFi holdout has >= 250 dates", "status": "provisional"})
    la = out["leader_ablation"]
    R.append({"id": "R-ML-23", "kind": "expect",
              "rule": "WHEN citing cross-asset context for a next-hour variance expectation THEN expect BTC to be the main leader for "
                      "alts, ETH, equities, commodities and gold tokens (ETH for BTC itself), and QQQ to matter less than BTC even for "
                      "TradFi perps; treat it as explanation only, since the block does not improve the forecast",
              "when": "explaining or sanity-checking an hourly vol forecast",
              "then": "mention BTC's last hours first; don't adjust the forecast for leader bursts beyond what the target's own recent "
                      "returns show",
              "scope": {"assets": "crypto perps (universe L) and Bybit TradFi perps (universe T)", "horizon": "1h", "sides": "both"},
              "effect": "holdout within-block TreeSHAP share BTC / ETH / QQQ: alts 54/46/0, equities 52/29/19, commodities 49/33/18, gold "
                        "tokens 48/35/17, ETH 91/9/0, BTC -/77/0 (own copy excluded); only significant holdout ablation: BTC for equities "
                        f"(XGB {la[ho]['T:equity']['xgb:x_btc']['dqlike']:+.4f}, t {la[ho]['T:equity']['xgb:x_btc']['t']:.1f})",
              "confidence": "C", "evidence": ["H-NN12"],
              "review": "recompute on the first 3 forward refits; demote if the per-class leader changes",
              "status": "provisional"})
    R.append({"id": "R-ML-24", "kind": "avoid",
              "rule": "WHEN extending an hourly vol network with inputs that have a short history (a series listed months ago, such as "
                      "QQQ for the crypto model) THEN avoid adding them to the MLP; the network extrapolates and next-hour QLIKE gets worse",
              "when": "choosing inputs for a retrained hourly vol MLP",
              "then": "keep the MLP on the target's own inputs; test new cross-asset inputs in trees or linear models first, as a new hypothesis",
              "scope": {"assets": "crypto perps (universe L) and Bybit TradFi perps (universe T)", "horizon": "1h", "sides": "both"},
              "effect": f"MLP with minus without block: crypto {_dt(cr[ins]['mlp'])} / {_dt(cr[ho]['mlp'])} (in / ho), TradFi holdout "
                        f"{_dt(tf[ho]['mlp'])}; removing the QQQ block improves the crypto-alt MLP by "
                        f"{-la[ho]['L:crypto_alt']['mlp:x_qqq']['dqlike']:.4f} (t {-la[ho]['L:crypto_alt']['mlp:x_qqq']['t']:.1f}) in the holdout",
              "confidence": "C", "evidence": ["H-NN12"],
              "review": "repeat when QQQ has >= 12 months of history",
              "status": "provisional"})
    lines: list[str] = []
    for r in R:
        lines.append(f"- id: {r['id']}")
        for k in ("kind", "rule", "when", "then"):
            lines.append(f"  {k}: {_q(r[k])}")
        lines.append("  scope:")
        for k in ("assets", "horizon", "sides"):
            lines.append(f"    {k}: {_q(r['scope'][k])}")
        lines.append(f"  effect: {_q(r['effect'])}")
        lines.append(f"  confidence: {_q(r['confidence'])}")
        lines.append(f"  evidence: [{', '.join(_q(e) for e in r['evidence'])}]")
        lines.append(f"  review: {_q(r['review'])}")
        lines.append(f"  status: {_q(r['status'])}")
    return lines


def v_ok(out: dict, h: str) -> bool:
    return out["verdicts"][h]["pooled"] == "supports"
