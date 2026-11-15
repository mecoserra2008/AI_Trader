"""Stages, verdicts, figures and the Markdown report of the VBOS study (see research.vol_bos_study).

insample  -> <cache>/insample.json, <cache>/selection.json (frozen), <cache>/insample_series.npz; records llm_in_sample
holdout   -> <cache>/holdout.json (ONE look with the frozen selection); records clean_holdout + pooled
report    -> research.vol_bos_writeup: research/reports/vol_bos_2026-11-13.{json,md} and research/figures/vol_bos/*.png
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ai_trader.core.config import load
from ai_trader.core.time import now
from ai_trader.features import vol_bos as vb
from ai_trader.research import vol_bos_study as st

REPORT = "vol_bos_2026-11-13"
DATA_NOTE = "bybit L 23 crypto (pooled), T 16 TradFi (holdout robustness), BTC look; 1d + 4h; research.vol_bos_study"


def _save(cache: Path, name: str, obj) -> Path:
    p = cache / name
    p.write_text(json.dumps(st._jsonable(obj), indent=1), encoding="utf-8")
    return p


def _load(cache: Path, name: str) -> dict:
    return json.loads((cache / name).read_text(encoding="utf-8"))


def _already(hid: str, seg: str) -> bool:
    from ai_trader.research.hypotheses import results

    return any(r["segment"] == seg for r in results(hid))


def _record(hid: str, seg: str, verdict: str, metrics: dict, note: str) -> None:
    from ai_trader.research.hypotheses import record

    if _already(hid, seg):
        print(f"skip record {hid} {seg}: already recorded")
        return
    record(hid, seg, verdict, st._jsonable(metrics), note=note, data=DATA_NOTE)
    print(f"recorded {hid} {seg} {verdict}")


# ====================================================================== pilot
def pilot(cache: Path, as_of) -> dict:
    r = st.stage_insample(cache, as_of, syms=["BTCUSDT"], tag="BTC-pilot", min_ev=3, freqs=("1d",))
    _save(cache, "pilot_btc_1d.json", r)
    print(json.dumps(st._jsonable({"vbos1": {k: v["h10"] for k, v in r["vbos1"].items()}, "vbos2": {k: v["h10"] for k, v in r["vbos2"].items()},
                                   "vbos3": r["vbos3"]["final_in_sample"]}), indent=1))
    return r


# ====================================================================== in-sample stage
def _freeze(L: dict, B: dict) -> dict:
    def ev_sel(r):
        return {"vbos1": {k: {"cfg": v["final"]["cfg"], "name": v["final"].get("name")} for k, v in r["vbos1"].items()},
                "vbos2": {k: {"cfg": v["final"]["cfg"], "params": v["final"].get("params")} for k, v in r["vbos2"].items()},
                "vbos3": r["vbos3"]["final_cfg"], "vbos3_label": r["vbos3"]["walk_forward"]["final"]["label"]}

    v1, best1 = st.verdict_vbos1_is(L)
    v2, best2 = st.verdict_vbos2_is(L)
    return {"frozen_at": now().isoformat(), "L": ev_sel(L), "BTC": ev_sel(B), "vbos1_primary": best1, "vbos2_primary": best2,
            "vbos4_in_sample": {k: v["in_sample"] for k, v in L["vbos4_split"].items()}}


def insample_verdicts(L: dict) -> dict:
    v1, best1 = st.verdict_vbos1_is(L)
    v2, best2 = st.verdict_vbos2_is(L)
    v3 = st.verdict_vbos3_is(L)
    hits = [L["vbos4_split"][k]["check"]["hit"] for k in ("range|1d", "range|4h", "semi|1d", "semi|4h")]
    v4 = st.verdict_vbos4(hits)
    p = {"H-VBOS1": L["vbos1_holm"]["p_holm"][best1], "H-VBOS2": L["vbos2_holm"]["p_holm"][best2],
         "H-VBOS3": st.p_one(L["vbos3"]["walk_forward"]["wf_diff_t"])}
    return {"H-VBOS1": v1, "H-VBOS2": v2, "H-VBOS3": v3, "H-VBOS4": v4, "vbos1_primary": best1, "vbos2_primary": best2,
            "vbos4_split_hits": hits, "holm_vbos": {"p": p, "p_holm": st.holm(p)}}


def run_insample(cache: Path, as_of, record: bool = False) -> dict:
    L = st.stage_insample(cache, as_of)
    B = st.stage_insample(cache, as_of, syms=["BTCUSDT"], tag="BTC", min_ev=3)
    sel = _freeze(L, B)
    _save(cache, "selection.json", sel)
    ver = insample_verdicts(L)
    wf = L["_wf"]
    np.savez_compressed(cache / "insample_series.npz", wf=wf["wf_series"], wf_ens=wf["wf_ens_series"], wf_days=wf["wf_days_idx"],
                        logits=np.array(L["vbos3"]["pbo"]["logits"]))
    L["vbos3"]["pbo"].pop("logits", None)
    B["vbos3"]["pbo"].pop("logits", None)
    out = {"L": L, "BTC": B, "verdicts": ver}
    _save(cache, "insample.json", out)
    print(json.dumps(st._jsonable(ver), indent=1))
    if record:
        record_insample(out)
    return out


def _m1(L: dict, key: str) -> dict:
    r = L["vbos1"][key]
    return {"primary": key, "wf_h10": r["h10"], "wf_h5": r["h5"], "wf_h20": r["h20"], "holm_p": L["vbos1_holm"]["p_holm"],
            "final_detector": r["final"].get("name"), "picks": [p["cfg"] for p in r["picks"]]}


def record_insample(out: dict) -> None:
    L, v = out["L"], out["verdicts"]
    _record("H-VBOS1", "llm_in_sample", v["H-VBOS1"], _m1(L, v["vbos1_primary"]),
            "nested purged WF (half-year outer blocks 2022H1-2026H1, inner median-of-folds), excess = ensemble daily P&L over 10 days after a declustered vol-BoS onset minus the symbol's block mean; 10-day-bucket clustered NW t; Holm over 4 families x 2 freqs")
    r2 = L["vbos2"][v["vbos2_primary"]]
    _record("H-VBOS2", "llm_in_sample", v["H-VBOS2"], {"primary": v["vbos2_primary"], "wf_h10": r2["h10"], "wf_h5": r2["h5"], "wf_h20": r2["h20"],
                                                       "holm_p": L["vbos2_holm"]["p_holm"], "final_params_w_N_q": r2["final"].get("params"),
                                                       "momentum_agreement": r2["final"].get("momentum_agreement")},
            "signed 10-day net return in daily sigma after an RS+ (RS-) break above its trailing quantile while the other side is not; nested purged WF; Holm over 2 freqs")
    s3 = L["vbos3"]
    _record("H-VBOS3", "llm_in_sample", v["H-VBOS3"], {"final": s3["walk_forward"]["final"]["label"], "dsr": s3["dsr_final"].get("dsr"),
                                                       "sr_ann": s3["dsr_final"].get("sr_ann"), "sr0_ann": s3["dsr_final"].get("sr0_ann"),
                                                       "pbo": s3["pbo"]["pbo"], "n_configs": s3["n_configs"], "final_in_sample": s3["final_in_sample"],
                                                       "wf_sharpe": s3["walk_forward"]["wf_sharpe"], "wf_ens_sharpe": s3["walk_forward"]["wf_ens_sharpe"],
                                                       "wf_diff_t": s3["walk_forward"]["wf_diff_t"]},
            "992-config grid (62 detectors x k x semi x exit x 1d/4h), EW L book; DSR of the final in-sample selection vs the grid; PBO CSCV 16 blocks")
    _record("H-VBOS4", "llm_in_sample", v["H-VBOS4"], {"split_half_hits": v["vbos4_split_hits"],
                                                       "checks": {k: x["check"] for k, x in L["vbos4_split"].items()}},
            "in-sample analogue: second-half (2024-01..2026-06) argmax inside the first-half (2021-2023) top-quintile region, 4 surfaces (range / semi x 1d / 4h); >=3 of 4 supports")


# ====================================================================== holdout stage (one look)
def _holdout_universe(cache: Path, syms, sel: dict, as_of, placebo: bool) -> dict:
    end, cut = st.end_day_of(as_of), st.cutoff_day()
    out = {"vbos1": {}, "vbos2": {}, "vbos3": {}}
    dbf = {fr: st.load_universe(cache, syms, fr) for fr in st.FREQS}
    out["n_symbols"] = {fr: len(ds) for fr, ds in dbf.items()}
    evs = {}
    for fr, ds in dbf.items():
        if not ds:
            continue
        ev1, base = st.vbos1_events(ds, end), st.Baseline(ds, end)
        ev2 = st.vbos2_events(ds, end)
        evs[fr] = (ev1, base, ev2)
        for key, s in sel["vbos1"].items():
            if key.endswith("|" + fr) and s["cfg"] >= 0:
                out["vbos1"][key] = {"detector": s["name"], **st.ev_segment(ev1, base, s["cfg"], cut, end + 1, end + 1)}
        s = sel["vbos2"].get(fr)
        if s and s["cfg"] >= 0:
            r = st.ev_segment(ev2, None, s["cfg"], cut, end + 1, end + 1)
            m = (ev2.cfg == s["cfg"]) & (ev2.d0 >= cut) & (ev2.d0 + st.H_PRIMARY <= end + 1)
            r["by_side"] = {str(sd): st.clustered_t(ev2.out[st.H_PRIMARY][m & (ev2.side == sd)], ev2.d0[m & (ev2.side == sd)]) for sd in (1, -1)}
            r["momentum_agreement"] = float(np.mean(ev2.extra["mom"][m] == ev2.side[m])) if m.any() else None
            out["vbos2"][fr] = {"params": s["params"], **r}
    fr, cfg = sel["vbos3"][0], tuple(sel["vbos3"][1])
    ds = dbf[fr]
    if ds:
        bk = st.build_books(ds, fr, end, grid=[cfg], verbose=False)
        ew, ens = bk.ew()[0], bk.ens_ew()
        W = np.arange(cut, end + 1)
        m = np.isfinite(ens[W])
        x, y = np.nan_to_num(ew[W][m]), ens[W][m]
        r = {"label": st.cfg_label(fr, cfg), "days": int(m.sum()), "sharpe": st.sharpe_ann(x), "ens_sharpe": st.sharpe_ann(y),
             "diff": dict(zip(("mean", "t", "n"), st.nw_t(x - y))), "mean_t": st.nw_t(x)[1],
             "exposure_days": float(bk.act[0, W].sum() / max(bk.live[W].sum(), 1)), "series": x.tolist(), "ens_series": y.tolist(),
             "days_idx": W[m].tolist()}
        r["sharpe_diff"] = (r["sharpe"] - r["ens_sharpe"]) if np.isfinite(r["sharpe"]) and np.isfinite(r["ens_sharpe"]) else None
        # the other freq's ensemble, for reference
        of = [f for f in st.FREQS if f != fr][0]
        if dbf[of]:
            bo = st.build_books(dbf[of], of, end, grid=[(0, 0, 0, 0)], verbose=False)
            eo = bo.ens_ew()[W]
            r["ens_sharpe_other_freq"] = {of: st.sharpe_ann(eo[np.isfinite(eo)])}
        if placebo:
            pb = st.placebo_books(ds, fr, cfg, end)
            ph = np.array([st.sharpe_ann(np.nan_to_num(p[W][m])) for p in pb])
            r["placebo"] = {"n": len(ph), "sharpe_q05_q50_q95": np.nanquantile(ph, [0.05, 0.5, 0.95]).tolist(),
                            "pct_below_actual": float(np.mean(ph < r["sharpe"])), "draws": ph.tolist()}
            Wi = np.arange(int(np.flatnonzero(bk.live > 0)[0]), cut)
            mi = np.isfinite(bk.ens_ew()[Wi])
            pi = np.array([st.sharpe_ann(np.nan_to_num(p[Wi][mi])) for p in pb])
            xi = st.sharpe_ann(np.nan_to_num(ew[Wi][mi]))
            r["placebo_in_sample"] = {"actual": xi, "sharpe_q05_q50_q95": np.nanquantile(pi, [0.05, 0.5, 0.95]).tolist(),
                                      "pct_below_actual": float(np.mean(pi < xi))}
        out["vbos3"] = r
    out["_evs"] = evs
    return out


def run_holdout(cache: Path, as_of, record: bool = False) -> dict:
    sel = _load(cache, "selection.json")
    ins = _load(cache, "insample.json")
    end, cut = st.end_day_of(as_of), st.cutoff_day()
    res = {"window": [st.ddate(cut), st.ddate(end)], "selection": sel}
    L = _holdout_universe(cache, st.UNIV_L, sel["L"], as_of, placebo=True)
    res["T"] = {k: v for k, v in _holdout_universe(cache, st.UNIV_T, sel["L"], as_of, placebo=False).items() if not k.startswith("_")}
    res["BTC"] = {k: v for k, v in _holdout_universe(cache, ["BTCUSDT"], sel["BTC"], as_of, placebo=False).items() if not k.startswith("_")}
    # VBOS4 holdout surfaces (L)
    v4 = {}
    for fr, (ev1, base, ev2) in L["_evs"].items():
        bpd = st.FREQS[fr]
        rc = [i for i, d in enumerate(vb.detector_grid(bpd)) if d.family == "range"]
        s1 = st.surfaces(ev1, base, rc, {"ho": (cut, end + 1)})["ho"]
        s2 = st.surfaces(ev2, None, list(range(len(vb.semi_grid()))), {"ho": (cut, end + 1)})["ho"]
        v4[f"range|{fr}"] = {"holdout": s1, "check": st.top_quintile_hit(sel["vbos4_in_sample"][f"range|{fr}"], s1)}
        v4[f"semi|{fr}"] = {"holdout": s2, "check": st.top_quintile_hit(sel["vbos4_in_sample"][f"semi|{fr}"], s2)}
    res["vbos4"] = v4
    res["L"] = {k: v for k, v in L.items() if not k.startswith("_")}
    res["verdicts"] = holdout_verdicts(ins, res)
    _save(cache, "holdout.json", res)
    print(json.dumps(st._jsonable(res["verdicts"]), indent=1))
    if record:
        record_holdout(ins, res)
    return res


def holdout_verdicts(ins: dict, res: dict) -> dict:
    vi = ins["verdicts"]
    L = res["L"]
    p1 = vi["vbos1_primary"]
    h1 = L["vbos1"].get(p1, {}).get("h10", {})
    v1 = "inconclusive" if not h1.get("n_events") else ("supports" if (h1.get("mean") or 0) > 0 else "contradicts")
    p2 = vi["vbos2_primary"]
    h2 = L["vbos2"].get(p2, {}).get("h10", {})
    v2 = "inconclusive" if not h2.get("n_events") else ("supports" if (h2.get("mean") or 0) > 0 else "contradicts")
    sd = L["vbos3"].get("sharpe_diff")
    v3h = "inconclusive" if sd is None else ("supports" if sd > 0 else "contradicts")
    hits = [res["vbos4"][k]["check"].get("hit") for k in ("range|1d", "range|4h", "semi|1d", "semi|4h")]
    v4 = st.verdict_vbos4(hits)
    s3 = ins["L"]["vbos3"]
    crit = [bool((s3["dsr_final"].get("dsr") or 0) > 0.95), bool(s3["pbo"]["pbo"] < 0.5), v3h == "supports"]
    v3p = "supports" if all(crit) else ("contradicts" if sum(crit) <= 1 else "inconclusive")

    def pooled(a: str, b: str) -> str:
        if a == b == "supports":
            return "supports"
        return "contradicts" if "contradicts" in (a, b) else "inconclusive"

    return {"clean_holdout": {"H-VBOS1": v1, "H-VBOS2": v2, "H-VBOS3": v3h, "H-VBOS4": v4},
            "pooled": {"H-VBOS1": pooled(vi["H-VBOS1"], v1), "H-VBOS2": pooled(vi["H-VBOS2"], v2), "H-VBOS3": v3p,
                       "H-VBOS4": pooled(vi["H-VBOS4"], v4)},
            "vbos3_criteria_dsr_pbo_holdout": crit, "vbos4_holdout_hits": hits}


def record_holdout(ins: dict, res: dict) -> None:
    v = res["verdicts"]
    L = res["L"]
    vi = ins["verdicts"]
    _record("H-VBOS1", "clean_holdout", v["clean_holdout"]["H-VBOS1"], {"primary": vi["vbos1_primary"], **L["vbos1"].get(vi["vbos1_primary"], {}),
                                                                         "all_families": L["vbos1"]},
            "frozen in-sample selection, one look; excess vs the symbol's holdout mean; 10-day-bucket clustered t")
    _record("H-VBOS2", "clean_holdout", v["clean_holdout"]["H-VBOS2"], {"primary": vi["vbos2_primary"], **L["vbos2"].get(vi["vbos2_primary"], {})},
            "frozen in-sample semivariance threshold, one look")
    m3 = {k: x for k, x in L["vbos3"].items() if k not in ("series", "ens_series", "days_idx")}
    if "placebo" in m3:
        m3["placebo"] = {k: x for k, x in m3["placebo"].items() if k != "draws"}
    _record("H-VBOS3", "clean_holdout", v["clean_holdout"]["H-VBOS3"], m3, "frozen final config vs the same-freq unfiltered ensemble (EW L book); paired daily diff NW t")
    _record("H-VBOS4", "clean_holdout", v["clean_holdout"]["H-VBOS4"], {"hits": v["vbos4_holdout_hits"], "checks": {k: x["check"] for k, x in res["vbos4"].items()}},
            "holdout argmax inside the in-sample top-quintile region; >=3 of 4 surfaces supports")
    for h in ("H-VBOS1", "H-VBOS2", "H-VBOS3", "H-VBOS4"):
        _record(h, "pooled", v["pooled"][h], {"llm_in_sample": vi[h], "clean_holdout": v["clean_holdout"][h],
                                              **({"criteria_dsr_pbo_holdout": v["vbos3_criteria_dsr_pbo_holdout"]} if h == "H-VBOS3" else {})},
                "pooled = supports only if both segments support (VBOS3: all three criteria; contradicts if <= 1 holds)")
