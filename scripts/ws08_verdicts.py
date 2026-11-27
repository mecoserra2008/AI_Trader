"""WS08 verdicts from data/research/ws08/*.json, applying the registered success rules literally (H-WS08-1..22).

    .venv/Scripts/python.exe scripts/ws08_verdicts.py            # print summary + verdicts
    .venv/Scripts/python.exe scripts/ops/governor.py lock results -- .venv/Scripts/python.exe scripts/ws08_verdicts.py --record
"""

from __future__ import annotations

import json
import sys

import numpy as np

from ai_trader.research.ws08_predict import OUT, holm

DATA = "data/research/ws08/*.json; src/ai_trader/research/ws08_predict.py"


def j(name):
    p = OUT / f"{name}.json"
    return json.loads(p.read_text()) if p.exists() else None


def comb_ivx(c):
    """h > 1 cells must pass both the one-period KMS IVX and the long-horizon variant (diary, amended before results)."""
    ps = [c.get("ivx_p"), c.get("ivx_p1")]
    ps = [p for p in ps if p is not None]
    return max(ps) if ps else 1.0


def is_pass(c):
    if c.get("p_rw") is None:
        return False
    hod_ok = c["h"] == 1 or (c.get("t_hodrick") is not None and abs(c["t_hodrick"]) > 1.96)
    return c["p_rw"] < 0.05 and comb_ivx(c) < 0.05 and hod_ok


def grid_verdicts(G):
    cells = [c for c in G["cells"] if "b" in c]
    out = {}
    surv = [c for c in cells if is_pass(c)]
    if surv:
        v_is = "supports"
    elif not any(c["p_rw"] < 0.10 for c in cells):
        v_is = "contradicts"
    else:
        v_is = "inconclusive"
    ho = [c for c in surv if c.get("b_ho") is not None and c.get("n_eff_ho", 0) >= 30]
    if not surv or not ho:
        v_ho = "inconclusive"
    else:
        v_ho = "supports" if any(np.sign(c["b_ho"]) == np.sign(c["b"]) for c in ho) else "contradicts"
    out["H-WS08-1"] = (v_is, v_ho, {"n_cells": len(cells), "c_rw95": G["c_rw95"], "survivors": [c["cell"] for c in surv],
                                    "min_p_rw": min(c["p_rw"] for c in cells),
                                    "top5": [{k: c.get(k) for k in ("cell", "b", "t_nw", "t_boot", "p_rw", "ivx_p", "ivx_p1", "t_hodrick", "b_ho", "t_ho")}
                                             for c in sorted(cells, key=lambda c: c["p_boot"])[:5]]})
    fam = {}
    for c in cells:
        fam.setdefault(c["hyp"], []).append(c)
    for hyp, cs in sorted(fam.items()):
        def right_sign(c):
            s = c["sign_reg"]
            return s == 0 or np.sign(c["b"]) == s
        sig = [c for c in cs if is_pass(c)]
        good = [c for c in sig if right_sign(c)]
        if good:
            v_is = "supports"
        elif all(comb_ivx(c) > 0.10 for c in cs) or (sig and not good):
            v_is = "contradicts"
        else:
            v_is = "inconclusive"
        hog = [c for c in good if c.get("b_ho") is not None]
        if hog:
            v_ho = "supports" if any(np.sign(c["b_ho"]) == np.sign(c["b"]) for c in hog) else "contradicts"
        else:
            v_ho = "inconclusive"
        best = sorted(cs, key=lambda c: c["p_boot"])[:4]
        out[hyp] = (v_is, v_ho, {"n_cells": len(cs), "n_ivx_lt_10": sum(comb_ivx(c) < 0.10 for c in cs),
                                 "n_rw_lt_10": sum(c["p_rw"] < 0.10 for c in cs), "passing": [c["cell"] for c in sig],
                                 "best": [{k: c.get(k) for k in ("cell", "b", "t_nw", "t_hodrick", "t_boot", "p_rw", "ivx_p", "ivx_p1", "rho",
                                                                  "b_stambaugh", "b_ho", "t_ho", "n_eff_ho", "oos_r2_is", "cw_p_is", "oos_r2_ho")}
                                          for c in best]})
    # OOS (H-WS08-13)
    cw = [c for c in cells if c.get("cw_p_is") is not None]
    adj = holm([c["cw_p_is"] for c in cw])
    # The registered statement asks for CW (Holm) AND the conditional GW; the success line names CW (Holm) + holdout OOS R^2 > 0
    # only. Both readings are reported; the stricter one (both) is recorded (decided before recording, after seeing the grid).
    cw_only = [c for c, a in zip(cw, adj, strict=True) if a < 0.05]
    passing = [c for c in cw_only if (c.get("gw_p_cond_is") or 1) < 0.05 and (c.get("gw_t_is") or 0) > 0]
    ok_ho = [c for c in passing if (c.get("oos_r2_ho") or -1) > 0]
    v_is = "supports" if passing else ("contradicts" if not any(c["cw_p_is"] < 0.05 for c in cw) else "inconclusive")
    v_ho = "supports" if ok_ho else ("contradicts" if passing else "inconclusive")
    lenient = {"cw_holm_pass": [c["cell"] for c in cw_only],
               "cw_holm_pass_and_ho_oos_r2_pos": [c["cell"] for c in cw_only if (c.get("oos_r2_ho") or -1) > 0]}
    pos_r2 = sorted([c for c in cw if (c.get("oos_r2_is") or -1) > 0], key=lambda c: -c["oos_r2_is"])[:6]
    out["H-WS08-13"] = (v_is, v_ho, {"n_cells": len(cw), "n_cw_raw_lt_05": sum(c["cw_p_is"] < 0.05 for c in cw),
                                     "n_holm_lt_05": sum(a < 0.05 for a in adj), "passing": [c["cell"] for c in passing],
                                     "success_line_only_reading": lenient,
                                     "n_oos_r2_pos_is": sum((c.get("oos_r2_is") or -1) > 0 for c in cw),
                                     "n_oos_r2_pos_ho": sum((c.get("oos_r2_ho") or -1) > 0 for c in cw),
                                     "best_oos": [{k: c.get(k) for k in ("cell", "oos_r2_is", "cw_t_is", "cw_p_is", "gw_p_cond_is", "oos_r2_ho", "cw_t_ho", "oos_r2_zero_is")}
                                                  for c in pos_r2]})
    return out, surv


def main(record: bool, only: set[str] | None = None) -> None:
    G, Q, E, X, M, R = (j(n) for n in ("grid", "qr", "econ", "xsect", "macro", "rv"))
    V = {}
    surv = []
    if G:
        gv, surv = grid_verdicts(G)
        V.update(gv)
    if E and G:
        cells = {c["cell"]: c for c in G["cells"] if "b" in c}
        pick = [c["cell"] for c in surv] or [c["cell"] for c in sorted(cells.values(), key=lambda c: c["p_boot"])[:3]]
        rows = {k: E["cells"].get(k) for k in pick}
        good = [k for k in (c["cell"] for c in surv) if rows.get(k) and rows[k].get("is", {}).get("net_sigma_taker_ci", [0])[0] > 0
                and rows[k].get("ho", {}).get("net_sigma_taker", -1) > 0]
        v = "supports" if good else ("contradicts" if not surv or all(rows[k]["is"]["net_sigma_taker"] <= 0 for k in pick if rows.get(k)) else "inconclusive")
        V["H-WS08-14"] = (v, v if v != "supports" else "supports", {"selected_flag": not surv, "n_trials": E["n_trials"], "cells": rows})
    if Q:
        sig = [r for r in Q if r["spread_p_holm"] < 0.05 and r["p50"] > 0.10]
        okh = [r for r in sig if r.get("spread_ho") is not None and np.sign(r["spread_ho"]) == np.sign(r["spread"])]
        v_is = "supports" if sig else ("contradicts" if not any(r["spread_p"] < 0.05 for r in Q) else "inconclusive")
        v_ho = "supports" if okh else ("contradicts" if sig else "inconclusive")
        top = sorted(Q, key=lambda r: r["spread_p"])[:8]
        V["H-WS08-12"] = (v_is, v_ho, {"n_cells": len(Q), "n_raw_lt_05": sum(r["spread_p"] < 0.05 for r in Q), "passing": [r["cell"] for r in sig],
                                      "top": [{k: r.get(k) for k in ("cell", "b10", "b50", "b90", "p50", "spread", "spread_p", "spread_p_holm", "spread_ho", "n_ho")} for r in top]})
    if X:
        xs, ts = X["xs"], X["trend_sleeve"]
        isx, hox = xs.get("is", {}), xs.get("ho", {})
        def xsv(c, sgn, ls_key, ls_kind):
            fm = isx[f"fm_{c}"]
            rw = isx["rw_p"][c]
            ls = isx[f"ls_{c}_{ls_kind}_net"]["mean_wk"]
            ok = np.sign(fm["mean"]) == sgn and abs(fm["t"]) > 2 and rw < 0.05 and ls > 0
            v_is = "supports" if ok else ("contradicts" if np.sign(fm["mean"]) != sgn else "inconclusive")
            hs = hox.get(f"fm_{c}", {}).get("mean")
            v_ho = ("supports" if np.sign(hs) == sgn else "contradicts") if (ok and hs is not None) else "inconclusive"
            return v_is, v_ho, {"fm_is": fm, "rw_p": rw, "ls_net_is": isx[f"ls_{c}_{ls_kind}_net"], "fm_ho": hox.get(f"fm_{c}"),
                                "ls_net_ho": hox.get(f"ls_{c}_{ls_kind}_net"), "fm1_is": isx.get(f"fm1_{c}"), "weeks": [isx.get("weeks"), hox.get("weeks")]}
        V["H-WS08-15"] = xsv("mom", 1, "mom", "price")
        V["H-WS08-16"] = xsv("rev", -1, "rev", "price")
        V["H-WS08-17"] = xsv("carry", -1, "carry", "total")
        b = isx["fm_beta"]
        okb = b["mean"] < 0 and b["t"] < -2 and isx["rw_p"]["beta"] < 0.05
        v_is = "supports" if okb else ("contradicts" if (b["mean"] > 0 and b["t"] > 2) else "inconclusive")
        hb = hox.get("fm_beta", {}).get("mean")
        V["H-WS08-18"] = (v_is, ("supports" if hb is not None and hb < 0 else "contradicts") if okb else "inconclusive",
                          {"fm_is": b, "rw_p": isx["rw_p"]["beta"], "raw_beta_is": isx.get("fm_beta_raw"), "fm_ho": hox.get("fm_beta"),
                           "raw_beta_ho": hox.get("fm_beta_raw"), "ls_net_is": isx.get("ls_beta_price_net"), "ls_net_ho": hox.get("ls_beta_price_net")})
        ti, th = ts.get("llm_in_sample", {}), ts.get("clean_holdout", {})
        if "t" in ti:
            ok = ti["coef"]["s_mom"] > 0 and ti["t"]["s_mom"] > 2 and ti["filter_top_half_mom"]["diff"] > 0
            v_is = "supports" if ok else ("contradicts" if ti["coef"]["s_mom"] <= 0 else "inconclusive")
            hok = "t" in th and th["coef"]["s_mom"] > 0 and th["filter_top_half_mom"]["diff"] > 0
            V["H-WS08-19"] = (v_is, ("supports" if hok else "contradicts") if ok else "inconclusive", {"is": ti, "ho": th})
    if M:
        win_is = sum(M[a]["is"]["dm_mse_kalman_vs_const"]["t"] < -2 and M[a]["is"]["dm_mse_kalman_vs_roll"]["t"] < -2 for a in M)
        lose_is = sum(M[a]["is"]["dm_mse_kalman_vs_const"]["t"] > 2 for a in M)
        v_is = "supports" if win_is >= 2 else ("contradicts" if lose_is >= 2 else "inconclusive")
        win_ho = sum(M[a]["ho"]["dm_mse_kalman_vs_const"]["mean"] < 0 and M[a]["ho"]["dm_mse_kalman_vs_roll"]["mean"] < 0 for a in M)
        V["H-WS08-20"] = (v_is, ("supports" if win_ho >= 2 else "contradicts") if v_is == "supports" else "inconclusive",
                          {a: {s: {k: M[a][s][k] for k in ("n", "mse", "logscore", "dm_mse_kalman_vs_const", "dm_mse_kalman_vs_roll", "dm_ls_kalman_vs_const",
                                                           "dm_ls_kalman_vs_roll", "beta_ndx_kalman_mean", "beta_ndx_kalman_sd")} for s in ("is", "ho")} for a in M})
        ok = [a for a in M if M[a]["is"]["asym"]["diff_dn_minus_up"] > 0 and M[a]["is"]["asym"]["p_holm"] < 0.05]
        v_is = "supports" if ok else ("contradicts" if all(M[a]["is"]["asym"]["diff_dn_minus_up"] < 0 for a in M) else "inconclusive")
        v_ho = ("supports" if any(M[a]["ho"]["asym"]["diff_dn_minus_up"] > 0 for a in ok) else "contradicts") if ok else "inconclusive"
        V["H-WS08-21"] = (v_is, v_ho, {a: {s: M[a][s]["asym"] for s in ("is", "ho")} for a in M} | {"const_beta": {a: {s: M[a][s]["const_beta_raw"] for s in ("is", "ho")} for a in M},
                                                                                                  "with_hy": {a: {s: M[a][s].get("with_hy") for s in ("is", "ho")} for a in M}})
    if R:
        cells = list(R)
        win = sum(R[c]["is"]["dm_dvol_vs_har"]["t"] < -2 for c in cells)
        lose = sum(R[c]["is"]["dm_dvol_vs_har"]["t"] > 0 for c in cells)
        skew_ok = all(abs(R[c]["is"].get("dm_skew_increment", {}).get("t", 0) or 0) < 2 for c in cells)
        ho_ok = sum(R[c]["ho"]["dm_dvol_vs_har"]["mean"] < 0 for c in cells) >= 3
        v_is = "supports" if (win >= 3 and skew_ok) else ("contradicts" if lose >= 3 else "inconclusive")
        V["H-WS08-22"] = (v_is, ("supports" if ho_ok else "contradicts") if v_is == "supports" else ("supports" if ho_ok else "inconclusive"),
                          {c: {s: {k: R[c][s].get(k) for k in ("n", "qlike", "dm_dvol_vs_har", "n4", "qlike4", "dm_skew_increment", "dm_skew_vs_har", "mcs90")}
                               for s in ("is", "ho")} for c in cells})
    for k in sorted(V, key=lambda s: int(s.split("-")[-1])):
        v_is, v_ho, met = V[k]
        print(f"{k}: IS {v_is} | HO {v_ho}")
        print("   ", json.dumps(met, default=float)[:1500])
    if record:
        from ai_trader.research import hypotheses

        for k, (v_is, v_ho, met) in V.items():
            if only and k not in only:
                continue
            note = "WS08 (registered success rule applied literally; h > 1 needs both IVX forms; see research/reports/ws08_2026-11-27.md)"
            hypotheses.record(k, "llm_in_sample", v_is, met, note=note, data=DATA)
            hypotheses.record(k, "clean_holdout", v_ho, met, note=note, data=DATA)
        print("recorded", len(only) if only else len(V))


if __name__ == "__main__":
    only = {a.split("=", 1)[1] for a in sys.argv if a.startswith("--only=")} or None
    main("--record" in sys.argv, only)
