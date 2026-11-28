"""Record H-WMN-1..14 from data/research/wmn/results/*.json with the registered success rules (run under
`governor.py lock results`). H-WMN-15 is forward (evaluated after 2026-12-21)."""

from __future__ import annotations

import json
import math

from ai_trader.research import hypotheses
from ai_trader.research.wmn_run import load

DATA = "data/research/wmn (events.parquet, bars5m, es_*.parquet, results/*.json)"


def fin(x) -> bool:
    return x is not None and isinstance(x, (int, float)) and math.isfinite(x)


def rec(hid, seg, verdict, metrics, note=""):
    hypotheses.record(hid, seg, verdict, metrics, note=note, data=DATA)
    print(hid, seg, verdict)


def main() -> None:
    det = load("detector_tests")
    et = load("event_tests")
    dc = load("decay")
    hk = load("hawkes")
    mk = load("markouts")
    ge = load("gate_eval")
    mkt = load("market_tests")

    # H-WMN-1
    h = det["H-WMN-1_D"]
    a, b = h["llm_in_sample"], h["clean_holdout"]
    v = "supports" if (a["lift"] >= 3 and fin(a["t"]) and a["t"] >= 2.33) else ("contradicts" if a["lift"] < 1.5 else "inconclusive")
    rec("H-WMN-1", "llm_in_sample", v, {"D": a, "Dp": det["H-WMN-1_Dp"]["llm_in_sample"]})
    vb = "supports" if (v == "supports" and b["lift"] > 1) else ("contradicts" if (fin(b["lift"]) and b["lift"] <= 1 and fin(b["t"]) and b["t"] < -2) else "inconclusive")
    rec("H-WMN-1", "clean_holdout", vb, {"D": b, "Dp": det["H-WMN-1_Dp"]["clean_holdout"]},
        note="holdout contradicts only if lift <= 1 with t < -2 (no holdout contradiction rule was registered)")
    # H-WMN-2
    h = det["H-WMN-2_D"]
    for seg in ("llm_in_sample", "clean_holdout"):
        s = h[seg]["explained_share"]
        v = "supports" if s < 0.25 else ("contradicts" if (s >= 0.5 and seg == "llm_in_sample") else "inconclusive")
        rec("H-WMN-2", seg, v, {"D": h[seg], "Dp": det["H-WMN-2_Dp"][seg]})
    # H-WMN-3
    h = et["H-WMN-3"]
    a, b = h["llm_in_sample"], h["clean_holdout"]
    v = "supports" if (a["diff"] >= 0.15 and a["t"] >= 3) else ("contradicts" if (a["diff"] <= 0 or a["t"] < 1) else "inconclusive")
    rec("H-WMN-3", "llm_in_sample", v, {"ARV_h0_1": a, "extra": {k: x["llm_in_sample"] for k, x in et["H-WMN-3_extra"].items()}})
    vb = "supports" if (v == "supports" and b["diff"] > 0) else ("contradicts" if b["diff"] <= 0 else "inconclusive")
    rec("H-WMN-3", "clean_holdout", vb, {"ARV_h0_1": b, "extra": {k: x["clean_holdout"] for k, x in et["H-WMN-3_extra"].items()}})
    # H-WMN-4: per category (one record per segment with the table)
    fam = et["H-WMN-4"]["bh_family"]
    tab = et["H-WMN-4"]["table"]
    cats = {}
    for g in sorted({r["group"] for r in tab}):
        rows = [r for r in tab if r["group"] == g]
        arv = [r for r in rows if r["metric"].startswith("ARV_h0_1") or r["metric"].startswith("ARV_h0_24")][0]
        famrow = [r for r in fam if r["group"] == g and r["metric"] == arv["metric"]]
        bh_ok = bool(famrow and famrow[0]["bh"] and famrow[0]["llm_in_sample_diff"] > 0)
        all_neg = all((r["llm_in_sample_diff"] is not None and fin(r["llm_in_sample_diff"]) and r["llm_in_sample_diff"] <= 0) for r in rows)
        n_in = arv["llm_in_sample_n"]
        if n_in < 30:
            cats[g] = {"in": "inconclusive (n < 30)", "hold": "inconclusive", "n_in": n_in}
            continue
        vin = "supports" if bh_ok else ("contradicts" if all_neg else "inconclusive")
        hd = arv["clean_holdout_diff"]
        vh = "supports" if (vin == "supports" and fin(hd) and hd > 0) else ("contradicts" if (fin(hd) and hd <= 0) else "inconclusive")
        cats[g] = {"in": vin, "hold": vh, "n_in": n_in, "n_hold": arv["clean_holdout_n"], "diff_in": arv["llm_in_sample_diff"],
                   "t_in": arv["llm_in_sample_t"], "diff_hold": hd, "t_hold": arv["clean_holdout_t"]}
    vin = "supports" if any(c["in"] == "supports" for c in cats.values()) else ("contradicts" if all(c["in"] == "contradicts" for c in cats.values()) else "inconclusive")
    vh = "supports" if any(c["hold"] == "supports" for c in cats.values()) else "inconclusive"
    rec("H-WMN-4", "llm_in_sample", vin, {"per_category": cats, "bh_family": fam}, note="per-category verdicts in metrics.per_category; overall = any category passes")
    rec("H-WMN-4", "clean_holdout", vh, {"per_category": cats}, note="holdout sign check per category")
    # H-WMN-5
    a, b = et["H-WMN-5"]["llm_in_sample"], et["H-WMN-5"]["clean_holdout"]
    v = "supports" if (a["diff"] > 0 and a["t_diff"] > 2) else ("contradicts" if (a["diff"] < 0 and a["t_diff"] < -2) else "inconclusive")
    rec("H-WMN-5", "llm_in_sample", v, a)
    vb = "supports" if (v == "supports" and b["diff"] > 0) else ("contradicts" if (fin(b["diff"]) and b["diff"] < 0 and b.get("t_diff", 0) < -2) else "inconclusive")
    rec("H-WMN-5", "clean_holdout", vb, b)
    # H-WMN-6
    a, b = et["H-WMN-6"]["llm_in_sample"], et["H-WMN-6"]["clean_holdout"]
    v = "supports" if (a["diff"] >= 0.10 and a["t"] >= 2) else ("contradicts" if a["diff"] <= 0 else "inconclusive")
    rec("H-WMN-6", "llm_in_sample", v, a)
    vb = "supports" if (v == "supports" and b["diff"] > 0) else ("contradicts" if b["diff"] <= 0 else "inconclusive")
    rec("H-WMN-6", "clean_holdout", vb, b)
    # H-WMN-7
    ai, bi = dc["a_news"].get("llm_in_sample", {}), dc["b_unlabelled"].get("llm_in_sample", {})
    ah, bh_ = dc["a_news"].get("clean_holdout", {}), dc["b_unlabelled"].get("clean_holdout", {})
    ok_in = all(fin(x.get("diff")) and x["diff"] > 0 and x["t"] > 2 for x in (ai, bi))
    bad_in = any(fin(x.get("diff")) and x["diff"] < 0 and x["t"] < -2 for x in (ai, bi))
    v = "supports" if ok_in else ("contradicts" if bad_in else "inconclusive")
    rec("H-WMN-7", "llm_in_sample", v, {"a_news": ai, "b_unlabelled": bi, "fits": {"a": dc["a_news"].get("fit_all"), "b": dc["b_unlabelled"].get("fit_all")}})
    okh = all(fin(x.get("diff")) and x["diff"] > 0 for x in (ah, bh_))
    badh = any(fin(x.get("diff")) and x["diff"] < 0 for x in (ah, bh_))
    vb = "supports" if (v == "supports" and okh) else ("contradicts" if (badh and v == "contradicts") else "inconclusive")
    rec("H-WMN-7", "clean_holdout", vb, {"a_news": ah, "b_unlabelled": bh_})
    # H-WMN-8
    a, b = dc["H-WMN-8"]["llm_in_sample"], dc["H-WMN-8"]["clean_holdout"]
    if "ratio" in a:
        v = "supports" if (a["ratio"] < 1 and a["ci90"][1] < 1) else ("contradicts" if a["ci90"][0] > 1 else "inconclusive")
    else:
        v = "inconclusive"
    rec("H-WMN-8", "llm_in_sample", v, {k: a[k] for k in a if not k.startswith("profile")})
    vb = "supports" if (v == "supports" and b.get("ratio", 9) < 1) else ("contradicts" if (b.get("ratio", 0) > 1 and v == "contradicts") else "inconclusive")
    rec("H-WMN-8", "clean_holdout", vb, {k: b[k] for k in b if not k.startswith("profile")})
    # H-WMN-9
    a, b = hk["llm_in_sample"], hk["clean_holdout"]
    v = "supports" if (a["diff"] > 0 and a["t"] > 2) else ("contradicts" if a["diff"] <= 0 else "inconclusive")
    rec("H-WMN-9", "llm_in_sample", v, {**a, "fit_M1": hk["fit_M1"], "branching_self": hk["branching_self"], "shares": hk["intensity_share_at_jumps"]["llm_in_sample"]})
    vb = "supports" if (v == "supports" and b["diff"] > 0) else ("contradicts" if b["diff"] <= 0 else "inconclusive")
    rec("H-WMN-9", "clean_holdout", vb, {**b, "shares": hk["intensity_share_at_jumps"]["clean_holdout"]})
    # H-WMN-10 / 11
    for hid, key in (("H-WMN-10", "news60"), ("H-WMN-11", "jump60")):
        vs = {}
        for seg in ("llm_in_sample", "clean_holdout"):
            c60, c300 = mk[seg]["h60"].get(key, {}), mk[seg]["h300"].get(key, {})
            vs[seg] = (c60, c300)
        (a60, a300), (b60, b300) = vs["llm_in_sample"], vs["clean_holdout"]
        ok = all(fin(x.get("coef")) and x["coef"] <= -1.0 and x["t"] <= -2 for x in (a60, a300))
        bad = all(fin(x.get("coef")) and x["coef"] >= 0 for x in (a60, a300))
        v = "supports" if ok else ("contradicts" if bad else "inconclusive")
        rec(hid, "llm_in_sample", v, {"h60": a60, "h300": a300, "n": mk["llm_in_sample"]["n_trades"], "symbol_days": mk["llm_in_sample"]["symbol_days"]})
        okh = all(fin(x.get("coef")) and x["coef"] < 0 for x in (b60, b300))
        badh = all(fin(x.get("coef")) and x["coef"] >= 0 for x in (b60, b300))
        vb = "supports" if (v == "supports" and okh) else ("contradicts" if (badh and v == "contradicts") else "inconclusive")
        rec(hid, "clean_holdout", vb, {"h60": b60, "h300": b300, "n": mk["clean_holdout"]["n_trades"], "symbol_days": mk["clean_holdout"]["symbol_days"]})
    # H-WMN-12
    a, b = mkt["macro_BTC"]["llm_in_sample"], mkt["macro_BTC"]["clean_holdout"]
    v = "supports" if (a["diff"] >= 0.2 and a["t"] >= 3) else ("contradicts" if a["diff"] <= 0 else "inconclusive")
    rec("H-WMN-12", "llm_in_sample", v, {"macro_BTC": a, "secondary": mkt["secondary"]})
    vb = "supports" if (v == "supports" and b["diff"] > 0) else ("contradicts" if b["diff"] <= 0 else "inconclusive")
    rec("H-WMN-12", "clean_holdout", vb, {"macro_BTC": b})
    # H-WMN-13
    a, b = ge["llm_in_sample"], ge["clean_holdout"]
    vin = "supports" if (a["qlike_diff"] < 0 and a["t"] < -2 and a["ratio_norm"] >= 2) else ("contradicts" if a["qlike_diff"] > 0 else "inconclusive")
    rec("H-WMN-13", "llm_in_sample", vin, a, note="parameters fit on FIT shocks: in-sample only")
    vb = "supports" if (b["qlike_diff"] < 0 and b["t"] < -2 and fin(b["ratio_norm"]) and b["ratio_norm"] >= 2) else ("contradicts" if b["qlike_diff"] > 0 else "inconclusive")
    rec("H-WMN-13", "clean_holdout", vb, b)
    # H-WMN-14
    b = mk["H-WMN-14"]["clean_holdout"]["h60"].get("pull", {})
    vb = "supports" if (fin(b.get("coef")) and b["coef"] <= -2 and b["t"] <= -2) else ("contradicts" if (fin(b.get("coef")) and b["coef"] >= 0) else "inconclusive")
    rec("H-WMN-14", "clean_holdout", vb, {"h60": b, "h300": mk["H-WMN-14"]["clean_holdout"]["h300"].get("pull"),
                                          "pull_share": mk["H-WMN-14"]["clean_holdout"]["pull_share"]})
    a = mk["H-WMN-14"]["llm_in_sample"]["h60"].get("pull", {})
    va = "supports" if (fin(a.get("coef")) and a["coef"] <= -2 and a["t"] <= -2) else ("contradicts" if (fin(a.get("coef")) and a["coef"] >= 0) else "inconclusive")
    rec("H-WMN-14", "llm_in_sample", va, {"h60": a, "pull_share": mk["H-WMN-14"]["llm_in_sample"]["pull_share"]}, note="gate fit on FIT: in-sample")


if __name__ == "__main__":
    main()
