"""D7F equivalence / non-superiority re-tests H-D7F-1..6 (registered 2026-11-28 in research/hypotheses_pending/D7F.yaml
before this code computed anything). Inputs are the latest recorded row per segment of each source hypothesis
(research/hypotheses_results.jsonl, read only); the estimate's SE is recovered from its recorded t or 95% CI.

    PYTHONIOENCODING=utf-8 uv run python -m ai_trader.research.d7f tost [--record]   (record inside `governor.py lock results`)
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable
from typing import Any

from ai_trader.research.hypotheses import results

SEGS = ("llm_in_sample", "clean_holdout")
ALPHA = 0.05
NOTE = "D7F equivalence re-test (extra, labelled trial on recorded estimates; margins anchored before computing, see the registration)"


def phi(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def holm_min(ps: list[float]) -> float:
    """Smallest Holm-adjusted p of a family."""
    ps = sorted(p for p in ps if p is not None and math.isfinite(p))
    if not ps:
        return float("nan")
    m = len(ps)
    adj, run = [], 0.0
    for i, p in enumerate(ps):
        run = max(run, min(1.0, (m - i) * p))
        adj.append(run)
    return adj[0]


def cell(est: float, se: float, margin: float, upper: bool = True) -> dict:
    """Non-superiority cell. upper=True: the claim is est < margin (H0: est >= margin), p_claim = Phi((est - margin) / se);
    upper=False: the claim is est > margin, p_claim = 1 - Phi((est - margin) / se). p_opp is the opposite one-sided p."""
    if not (se and math.isfinite(se) and se > 0 and math.isfinite(est)):
        return {"est": est, "se": se, "z": float("nan"), "p_claim": float("nan"), "p_opp": float("nan")}
    z = (est - margin) / se
    pc = phi(z) if upper else 1.0 - phi(z)
    return {"est": est, "se": se, "z": z, "p_claim": pc, "p_opp": 1.0 - pc}


def se_t(est: float, t: float) -> float:
    return abs(est / t) if t not in (0, None) and math.isfinite(t) and t != 0 else float("nan")


def judge(cells: dict[str, dict], alpha: float = ALPHA) -> dict:
    pcs = [c["p_claim"] for c in cells.values()]
    if not pcs or any(not math.isfinite(p) for p in pcs):
        return {"verdict": "inconclusive", "p_iut": float("nan"), "holm_opp": float("nan"), "why": "a cell has no usable SE"}
    p_iut = max(pcs)
    h_opp = holm_min([c["p_opp"] for c in cells.values()])
    v = "supports" if p_iut < alpha else "contradicts" if h_opp < alpha else "inconclusive"
    return {"verdict": v, "p_iut": p_iut, "holm_opp": h_opp, "verdict_alpha_0025": "supports" if p_iut < 0.025 else v if v != "supports" else "inconclusive",
            "worst_cell": max(cells, key=lambda k: cells[k]["p_claim"])}


def latest(hid: str) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for r in results(hid):
        if r.get("segment") in SEGS:
            out[r["segment"]] = r
    return out


# ---------------------------------------------------------------------- one builder per registered re-test
def t_mc1(m: dict) -> dict[str, dict]:
    return {k: cell(m[k]["mean_net"], se_t(m[k]["mean_net"], m[k]["t_nw"]), 0.0) for k in ("1h_mfon", "1h_mfoff", "4h_mfon", "4h_mfoff") if k in m}


def t_ws04_2(m: dict) -> dict[str, dict]:
    out = {}
    for h in ("1h", "4h", "1d"):
        for spec, t in ((m.get(h) or {}).get("garch_t_vs_best") or {}).items():
            # the recorded statistic is the DM t of (spec - best clock HAR); the claim is diff > 0 -> unit SE in t units
            out[f"{h}:{spec}"] = cell(float(t), 1.0, 0.0, upper=False)
    return out


def t_bo1(m: dict) -> dict[str, dict]:
    return {k: cell(v["diff_vs_placebo"], se_t(v["diff_vs_placebo"], v["t_vs_placebo"]), 0.10)
            for k, v in m.items() if isinstance(v, dict) and "diff_vs_placebo" in v}


def t_lev2(m: dict) -> dict[str, dict]:
    p = m["shar"]["primary_level"]
    return {"primary_level": cell(p["diff"], se_t(p["diff"], p["diff_t"]), 0.10)}


def t_lev2_robust(m: dict) -> dict[str, dict]:
    p = (m.get("shar") or {}).get("log")
    return {"log": cell(p["diff"], se_t(p["diff"], p["diff_t"]), 0.10)} if p else {}


def t_ws07_5(m: dict) -> dict[str, dict]:
    return {k: cell(v["mean"], se_t(v["mean"], v["t"]), 0.005) for k, v in m.items() if k.startswith("harmu|") and isinstance(v, dict)}


def _ci_se(v: dict) -> float:
    return (v["hi"] - v["lo"]) / (2 * 1.96)


def t_ws16_2(m: dict) -> dict[str, dict]:
    return {k: cell(m[k]["diff"], _ci_se(m[k]), 0.05) for k in ("zoo_iv", "zoo_shr", "zoo_fam", "zoo_vote") if k in m}


def t_ws16_2_extra(m: dict) -> dict[str, dict]:
    return {"zoo_slow": cell(m["zoo_slow"]["diff"], _ci_se(m["zoo_slow"]), 0.05)} if "zoo_slow" in m else {}


TESTS: dict[str, tuple[str, Callable[[dict], dict], Callable[[dict], dict] | None, str]] = {
    "H-D7F-1": ("H-MC1", t_mc1, None, "margin 0 (net mean < 0)"),
    "H-D7F-2": ("H-WS04-2", t_ws04_2, None, "margin 0 (GARCH-family QLIKE - best clock HAR > 0)"),
    "H-D7F-3": ("H-BO1", t_bo1, None, "margin +0.10 sigma (pivot - placebo < 0.10)"),
    "H-D7F-4": ("H-LEV2", t_lev2, t_lev2_robust, "margin +0.10 (b(RS-) - b(RS+) < 0.10)"),
    "H-D7F-5": ("H-WS07-5", t_ws07_5, None, "margin 0.005 nats (direction-term log-score gain < 0.005)"),
    "H-D7F-6": ("H-WS16-2", t_ws16_2, t_ws16_2_extra, "margin +0.05 R/trade (weighting - zoo_ew < 0.05)"),
}


def run(record: bool = False) -> dict:
    from ai_trader.research.d7f import OUT

    out: dict[str, Any] = {}
    for hid, (src, fn, extra, margin) in TESTS.items():
        rows = latest(src)
        out[hid] = {"source": src, "margin": margin, "segments": {}}
        for seg in SEGS:
            r = rows.get(seg)
            if r is None:
                out[hid]["segments"][seg] = {"verdict": "inconclusive", "why": "no recorded row"}
                continue
            m = r.get("metrics") or {}
            cells = fn(m)
            j = judge(cells)
            res = {**j, "cells": cells, "source_ts": r.get("ts"), "source_verdict": r.get("verdict")}
            if extra:
                res["reported_not_judged"] = extra(m)
            out[hid]["segments"][seg] = res
            print(f"{hid} ({src}) {seg:14s} -> {j['verdict']:12s} p_iut={j['p_iut']:.4g} holm_opp={j['holm_opp']:.4g} worst={j.get('worst_cell')}", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "tost.json").write_text(json.dumps(out, indent=1, default=float), encoding="utf-8")
    if record:
        from ai_trader.research.hypotheses import record as rec

        for hid, o in out.items():
            for seg, res in o["segments"].items():
                met = {"source": o["source"], "margin": o["margin"], "p_iut": res.get("p_iut"), "holm_opp": res.get("holm_opp"),
                       "worst_cell": res.get("worst_cell"), "verdict_alpha_0025": res.get("verdict_alpha_0025"),
                       "cells": {k: {kk: round(vv, 6) if isinstance(vv, float) and math.isfinite(vv) else vv for kk, vv in c.items()}
                                 for k, c in (res.get("cells") or {}).items()},
                       "source_row_ts": res.get("source_ts"), "source_verdict": res.get("source_verdict")}
                if res.get("reported_not_judged"):
                    met["reported_not_judged"] = res["reported_not_judged"]
                rec(hid, seg, res["verdict"], met, note=NOTE, data="data/research/d7f/tost.json")
        print("recorded", sum(len(o["segments"]) for o in out.values()), "rows")
    return out
