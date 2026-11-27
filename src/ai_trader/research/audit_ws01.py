"""WS01 AUDIT (specification program 2026-11-27): registry-wide multiple-testing control, holdout power, effect decline
and the research-debt ledger. Research only; reads research/hypotheses.yaml, research/hypotheses_results.jsonl and
research/rules.yaml, never edits them.

    python -m ai_trader.research.audit_ws01 run      # writes data/research/ws01/{fdr,rows,research_debt}.json

Frozen design (diary 2026-11-26 "WS01 design frozen before results", hypotheses H-WS01-1..7):
- Snapshot = the first SNAPSHOT_ROWS rows of the results file (later rows belong to other workstreams' running work).
- One primary two-sided p per row (`extract`), rows with no statistic keep p = 1 in every family.
- Families F_all (all rows), F_disc (llm_in_sample + pooled rows of ids with no llm_in_sample row), F_hold (clean_holdout).
- BH and BY step-up at q = 0.05 / 0.10.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from scipy import stats

ROOT = Path(__file__).resolve().parents[3]
REG = ROOT / "research" / "hypotheses.yaml"
RES = ROOT / "research" / "hypotheses_results.jsonl"
RULES = ROOT / "research" / "rules.yaml"
OUT = ROOT / "data" / "research" / "ws01"
SNAPSHOT_ROWS = 596
SNAPSHOT_SHA = "1be16ba2a6bdee17d547b2dafbe0215e9c055339dee4313179b10459239b2e0b"

P_RAW = ("p_raw", "p", "p_value", "pval", "p_two_sided")
T_KEYS = ("t_nw", "t", "t_stat", "z", "binom_z", "dm_t", "z_stat", "t_diff", "diff_t")
P_ADJ = ("p_holm", "holm_p")
N_KEYS = ("n_eff", "n_clusters", "n_weeks", "n_weekends", "n_days", "n_dates", "n_events", "n_trades", "n_obs", "n")
CI_KEYS = {"ci95": 1.959964, "ci": 1.959964, "ci90": 1.644854}
CI_PAIRS = {"diff_ci95": ("diff", 1.959964), "bss_ci90": ("bss", 1.644854)}
EST_KEYS = ("diff", "mean_diff", "mean", "estimate", "excess")
SEGMENTS = ("llm_in_sample", "clean_holdout", "pooled")

NULL_CLAIM = re.compile(
    r"\(null\)|\bdo not\b|\bdoes not\b|\bno edge\b|\bno leverage\b|\bnot special\b|\bnothing\b|\bless than\b|\bat most\b|"
    r"\brarely\b|\bno parameter cliffs\b|\bdo not reproduce\b|\bdoes not flip\b|\bno discrete state\b", re.I)

# Manual map for supports rows the generic rules miss (frozen rule 7): (id, segment) -> function(metrics) -> (p, t, how).
# Each is the statistic the registered success rule names (union over cells -> Simes; 'all of' -> max p, i.e. the
# intersection-union test; share-of-units claims -> two-sided binomial vs 1/2), chosen from the text, never the value.
def _ci_p(est: float, lo: float, hi: float, z: float = 1.959964, log: bool = False) -> float:
    if log:
        est, lo, hi = math.log(est), math.log(lo), math.log(hi)
    se = (hi - lo) / (2 * z)
    return float(2 * stats.norm.sf(abs(est / se))) if se > 0 else float("nan")


def _binom_p(frac: float, n: float) -> float:
    n = int(round(n))
    return float(stats.binomtest(int(round(frac * n)), n, 0.5).pvalue)


def _tp(t: float, n: float | None = None) -> float:
    return p_from_t(t, n)


def _simes_paths(m: dict, fn) -> float:
    return simes([x for x in fn(m) if x is not None and math.isfinite(x)])


MANUAL: dict[tuple[str, str], tuple[Any, str]] = {
    ("H-W1", "llm_in_sample"): (lambda m: _binom_p(m["frac_below_1"], m["n_weekends"]), "sign test vs 1 (registered test)"),
    ("H-W1", "clean_holdout"): (lambda m: _binom_p(m["frac_below_1"], m["n_weekends"]), "sign test vs 1 (registered test)"),
    ("H-VC1", "clean_holdout"): (lambda m: _tp(m["t24"], m["n_clusters"]), "t at h = 24 (success rule)"),
    ("H-C1", "llm_in_sample"): (lambda m: _tp(m["net"]["nw_t"], m["n_weeks"]), "net NW t (net Sharpe rule)"),
    ("H-C1", "clean_holdout"): (lambda m: _tp(m["net"]["nw_t"], m["n_weeks"]), "net NW t (net return rule)"),
    ("H-TR1", "llm_in_sample"): (lambda m: _tp(m["port_t_nw"]), "portfolio NW t (ensemble Sharpe > 0)"),
    ("H-DOW3", "llm_in_sample"): (lambda m: simes([_tp(m["pooled_crypto"]["t_down"], m["pooled_crypto"]["n"]),
                                                    _tp(m["pooled_tradfi"]["t_down"], m["pooled_tradfi"]["n"])]),
                                  "pooled down-week coefficient, Simes over crypto / TradFi"),
    ("H-JD2", "llm_in_sample"): (lambda m: _ci_p(m["bss"], *m["ci90"], z=1.644854), "Brier skill, block-bootstrap CI90"),
    ("H-JD2", "clean_holdout"): (lambda m: _ci_p(m["bss"], *m["ci90"], z=1.644854), "Brier skill, block-bootstrap CI90"),
    ("H-JD3", "llm_in_sample"): (lambda m: simes([_tp(m["low"]["t_f24"], m["low"]["n"]), _tp(m["normal_high"]["t_f24"], m["normal_high"]["n"])]),
                                 "24h legs, Simes (at least one leg |t| > 2)"),
    ("H-RC2", "clean_holdout"): (lambda m: max(m["p_vs_eq"], m["p_vs_uncond"]), "IUT over both alternatives"),
    ("H-NN8", "pooled"): (lambda m: max(_simes_paths(m[s], lambda d: [_tp(d[k]["t"], d[k].get("n_dates")) for k in ("mlp", "xgb", "lin") if k in d])
                                        for s in ("llm_in_sample", "clean_holdout")), "IUT over segments of Simes over models"),
    ("H-NN11", "pooled"): (lambda m: _simes_paths(m["clean_holdout"], lambda d: [_tp(d[k]["t"], d[k].get("n_dates")) for k in ("mlp", "xgb", "lin") if k in d]),
                           "Simes over models in clean_holdout (success rule)"),
    ("H-MINE2", "pooled"): (lambda m: _binom_p(m["replication_rate"], m["n_discoveries"]), "replication rate vs 50% bar"),
    ("H-MINE1", "pooled"): (lambda m: _binom_p(m["rate"], m["discoveries"]), "replication rate vs 50% bar"),
    ("H-TVTP1s", "clean_holdout"): (lambda m: _tp(m["clean_holdout"]["tvtp_minus_svar"]["t_hac"]), "DM HAC TVTP - SVAR (success rule)"),
}
for _seg in ("llm_in_sample", "clean_holdout", "pooled"):
    MANUAL[("H-EXE1", _seg)] = (lambda m: simes([_ci_p(m[f"diff_N{n}_prorata"]["mean"], m[f"diff_N{n}_prorata"]["lo"], m[f"diff_N{n}_prorata"]["hi"])
                                                for n in (1, 5, 15) if f"diff_N{n}_prorata" in m]), "Simes over N (at least one N)")
for _seg in ("llm_in_sample", "clean_holdout"):
    MANUAL[("H-BOS5", _seg)] = ((lambda s: lambda m: simes([max(_ci_p(m[tf][s][a]["mean"], m[tf][s][a]["lo"], m[tf][s][a]["hi"]) for a in ("r-m", "r-p"))
                                                            for tf in ("intraday", "swing") if tf in m and s in m[tf]]))(_seg),
                                "IUT over both alternatives, Simes over timeframes")
    MANUAL[("H-GEO3", _seg)] = (lambda m: _ci_p(m["0-5"]["ratio"], m["0-5"]["lo"], m["0-5"]["hi"], log=True), "[0,5) ratio, bootstrap CI, null 1")
    MANUAL[("H-TVTP2", _seg)] = (lambda m: max(_tp(m[f"qlike_tvtp_minus_{b}"]["t_hac"]) for b in ("rolling120", "ewma097", "dcc")),
                                 "IUT: DM HAC vs all three baselines")
    MANUAL[("H-RFSV1", _seg)] = (lambda m: _binom_p(m["daily"]["share_hi_below_0.35"], m["daily"]["n_tokens"]), "share of tokens vs 1/2")
MANUAL[("H-GEO6", "clean_holdout")] = (lambda m: _ci_p(m["us_all"]["geo_mean_ratio"], *m["us_all"]["geo_ci"], log=True), "US ratio, day-bootstrap CI, null 1")
MANUAL[("H-GEO9", "llm_in_sample")] = (lambda m: min(1.0, min(s["p_llm"] for s in m["survivors"]) * m["grid_cells"]), "Bonferroni min-p over the 156-cell grid")
MANUAL[("H-GEO9", "clean_holdout")] = (lambda m: simes([min(1.0, 2 * s["p_ho_1s"]) for s in m["survivors"]]), "Simes over survivors (two-sided)")
MANUAL[("H-PART2", "llm_in_sample")] = (lambda m: simes([_tp(v["FIT_llm_in_sample"]["t"], v["FIT_llm_in_sample"]["n_days"]) for v in m["d60_by_type"].values()]),
                                        "Simes over types (some type passes)")
MANUAL[("H-PART2", "clean_holdout")] = (lambda m: simes([_tp(v["HELD"]["t"], v["HELD"]["n_days"]) for v in m["d60_by_type"].values()]),
                                        "Simes over types on HELD")
MANUAL_ALL_ROWS_OF = {"H-STRESS0": (lambda m: _ci_p(m["mean_R"], *m["ci"]), "mean_R with its CI (per unit)")}


# ----------------------------------------------------------------------------------------------------------- loading
def load_registry() -> dict[str, dict[str, Any]]:
    return yaml.safe_load(REG.read_text(encoding="utf-8"))["hypotheses"]


def load_rows(n: int = SNAPSHOT_ROWS) -> list[dict[str, Any]]:
    lines = [x for x in RES.read_text(encoding="utf-8").splitlines() if x.strip()][:n]
    return [json.loads(x) for x in lines]


def load_rules() -> list[dict[str, Any]]:
    return yaml.safe_load(RULES.read_text(encoding="utf-8"))["rules"]


def claim_type(h: dict[str, Any]) -> str:
    text = f"{h.get('title', '')} {h.get('statement', '')}"
    return "null" if NULL_CLAIM.search(text) else "effect"


def forward_only(h: dict[str, Any]) -> bool:
    text = f"{h.get('title', '')} {h.get('statement', '')} {h.get('test', '')}"
    return bool(re.search(r"\bforward\b", text, re.I))


def post_hoc(h: dict[str, Any]) -> bool:
    text = f"{h.get('title', '')} {h.get('statement', '')} {h.get('rationale', '')}"
    return bool(re.search(r"post hoc|post-hoc|registered after|after seeing", text, re.I))


# -------------------------------------------------------------------------------------------------------- extraction
def _num(x: Any) -> float | None:
    if isinstance(x, bool) or not isinstance(x, (int, float)):
        return None
    x = float(x)
    return x if math.isfinite(x) else None


def _count(d: dict[str, Any]) -> tuple[str | None, float | None]:
    for k in N_KEYS:
        v = _num(d.get(k))
        if v is not None and v >= 2:
            return k, v
    return None, None


def p_from_t(t: float, n: float | None) -> float:
    if n is not None and n >= 3:
        return float(2 * stats.t.sf(abs(t), df=n - 1))
    return float(2 * stats.norm.sf(abs(t)))


def _ci(d: dict[str, Any]) -> tuple[float, float, str] | None:
    """(z, estimate, how) from a CI and its point estimate, or None."""
    for key, (est_key, zl) in CI_PAIRS.items():
        ci, est = d.get(key), _num(d.get(est_key))
        if isinstance(ci, list) and len(ci) == 2 and est is not None and all(_num(c) is not None for c in ci):
            se = (ci[1] - ci[0]) / (2 * zl)
            if se > 0:
                return est / se, est, f"{key}+{est_key}"
    for key, zl in CI_KEYS.items():
        ci = d.get(key)
        if not (isinstance(ci, list) and len(ci) == 2 and all(_num(c) is not None for c in ci)):
            continue
        lo, hi = float(ci[0]), float(ci[1])
        ratio = _num(d.get("ratio"))
        if ratio is not None and ratio > 0 and lo > 0 and hi > lo:
            se = (math.log(hi) - math.log(lo)) / (2 * zl)
            return math.log(ratio) / se, ratio, f"{key}+ratio(log)"
        for ek in EST_KEYS:
            est = _num(d.get(ek))
            if est is not None and hi > lo:
                se = (hi - lo) / (2 * zl)
                return est / se, est, f"{key}+{ek}"
    return None


def _scope_stat(d: dict[str, Any]) -> dict[str, Any] | None:
    """Raw p, then t/z, then CI, then adjusted-only p, within one dict."""
    nk, n = _count(d)
    t = None
    for k in T_KEYS:
        t = _num(d.get(k))
        if t is not None:
            tk = k
            break
    for k in P_RAW:
        p = _num(d.get(k))
        if p is not None and 0 <= p <= 1:
            return {"p": p, "source": f"p:{k}", "t": t, "n": n, "n_key": nk}
    if t is not None:
        return {"p": p_from_t(t, n), "source": f"t:{tk}", "t": t, "n": n, "n_key": nk}
    ci = _ci(d)
    if ci is not None:
        z, _, how = ci
        return {"p": float(2 * stats.norm.sf(abs(z))), "source": f"ci:{how}", "t": z, "n": n, "n_key": nk}
    for k in P_ADJ:
        p = _num(d.get(k))
        if p is not None and 0 <= p <= 1:
            return {"p": p, "source": f"adj:{k}", "t": None, "n": n, "n_key": nk, "flag": "adjusted_only"}
    return None


def simes(ps: list[float]) -> float:
    p = np.sort(np.asarray(ps, float))
    m = len(p)
    return float(min(1.0, np.min(m * p / np.arange(1, m + 1))))


def _get(d: Any, path: str) -> Any:
    for part in path.split("."):
        if not isinstance(d, dict) or part not in d:
            return None
        d = d[part]
    return d


def extract(row: dict[str, Any]) -> dict[str, Any]:
    m = row.get("metrics") or {}
    seg = row.get("segment")
    key = (row["id"], seg)
    fn = MANUAL.get(key) or (MANUAL_ALL_ROWS_OF.get(row["id"]) if row.get("verdict") == "supports" else None)
    if fn is not None and row.get("verdict") == "supports":
        try:
            p = float(fn[0](m))
        except (KeyError, TypeError, ValueError, ZeroDivisionError):
            p = float("nan")
        if math.isfinite(p):
            return {"p": min(max(p, 0.0), 1.0), "source": f"manual:{row['id']}", "t": None, "n": None, "n_key": None, "why": fn[1]}
    scopes = []
    if isinstance(m.get(seg), dict):
        scopes.append((seg, m[seg]))
    if isinstance(m.get("primary"), dict):
        scopes.append(("primary", m["primary"]))
    scopes.append(("top", m))
    for name, d in scopes:
        s = _scope_stat(d)
        if s is not None:
            s["scope"] = name
            return s
    # multi-cell: depth-1 sub-dicts carrying a statistic, else depth-2 (Simes 'some cell' p)
    for depth in (1, 2):
        cells = []
        subs = [v for k, v in m.items() if isinstance(v, dict) and k not in SEGMENTS]
        if depth == 2:
            subs = [vv for v in subs for vv in v.values() if isinstance(vv, dict)]
        for d in subs:
            s = _scope_stat(d)
            if s is not None and s.get("flag") != "adjusted_only":
                cells.append(s["p"])
        if cells:
            return {"p": simes(cells), "source": f"simes:depth{depth}", "t": None, "n": None, "n_key": None,
                    "flag": "cell_combined", "k_cells": len(cells)}
    return {"p": 1.0, "source": "none", "t": None, "n": None, "n_key": None, "flag": "no_statistic"}


# -------------------------------------------------------------------------------------------------------------- FDR
def bh(p: np.ndarray, q: float) -> np.ndarray:
    p = np.asarray(p, float)
    m = len(p)
    if m == 0:
        return np.zeros(0, bool)
    order = np.argsort(p, kind="mergesort")
    below = p[order] <= q * np.arange(1, m + 1) / m
    rej = np.zeros(m, bool)
    if below.any():
        k = int(np.max(np.nonzero(below)[0]))
        rej[order[: k + 1]] = True
    return rej


def by(p: np.ndarray, q: float) -> np.ndarray:
    m = len(p)
    c = float(np.sum(1.0 / np.arange(1, m + 1))) if m else 1.0
    return bh(p, q / c)


def bh_adjusted(p: np.ndarray, dependence: bool = False) -> np.ndarray:
    p = np.asarray(p, float)
    m = len(p)
    if m == 0:
        return p
    c = float(np.sum(1.0 / np.arange(1, m + 1))) if dependence else 1.0
    order = np.argsort(p, kind="mergesort")
    adj = p[order] * m * c / np.arange(1, m + 1)
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    out = np.empty(m)
    out[order] = np.minimum(adj, 1.0)
    return out


# ------------------------------------------------------------------------------------------------------------ audit
@dataclass
class Row:
    idx: int
    id: str
    segment: str
    verdict: str
    family: str
    claim: str
    p: float
    source: str
    t: float | None
    n: float | None
    n_key: str | None
    flag: str | None = None
    n_any: float | None = None
    n_any_key: str | None = None
    surv: dict[str, bool] = field(default_factory=dict)
    q_bh: dict[str, float] = field(default_factory=dict)
    q_by: dict[str, float] = field(default_factory=dict)


def build_rows(reg: dict, raw: list[dict]) -> list[Row]:
    out = []
    for i, r in enumerate(raw):
        h = reg.get(r["id"], {})
        s = extract(r)
        n, nk = s.get("n"), s.get("n_key")
        if n is None:                      # H-WS01-8: count key of the statistic's scope, else the top level
            m = r.get("metrics") or {}
            nk, n = _count(m[r["segment"]]) if isinstance(m.get(r["segment"]), dict) else (None, None)
            if n is None:
                nk, n = _count(m)
        out.append(Row(i, r["id"], r["segment"], r["verdict"], h.get("family", "?"), claim_type(h), s["p"], s["source"],
                       s.get("t"), s.get("n"), s.get("n_key"), s.get("flag"), n_any=n, n_any_key=nk))
    return out


def power_z(rows: list[Row]) -> dict[str, Any]:
    """H-WS01-8: holdout power from |z| = Phi^-1(1 - p/2) and the rows' own count keys."""
    ok = ("p:", "t:", "ci:")
    items = []
    for r_in, r_h in paired_in_hold(rows):
        if not (r_in.source.startswith(ok) and r_h.source.startswith(ok)):
            continue
        if r_in.n_any_key is None or r_in.n_any_key != r_h.n_any_key or not r_in.n_any or not r_h.n_any:
            continue
        z_in = float(stats.norm.isf(max(r_in.p, 1e-300) / 2))
        exp_z = z_in * math.sqrt(r_h.n_any / r_in.n_any)
        n_req = r_in.n_any * (2.486 / z_in) ** 2 if z_in > 0 else float("inf")
        items.append({"id": r_in.id, "family": r_in.family, "z_in": z_in, "n_in": r_in.n_any, "n_hold": r_h.n_any,
                      "n_key": r_in.n_any_key, "p_hold": r_h.p, "hold_verdict": r_h.verdict, "expected_z_hold": exp_z,
                      "power_hold": float(stats.norm.sf(1.644854 - exp_z)), "n_hold_required_80": n_req})
    pw = np.array([x["power_hold"] for x in items])
    return {"n": len(items), "median_power": float(np.median(pw)) if len(pw) else float("nan"),
            "share_power_below_0.5": float(np.mean(pw < 0.5)) if len(pw) else float("nan"), "items": items}


def families(rows: list[Row]) -> dict[str, list[int]]:
    has_in = {r.id for r in rows if r.segment == "llm_in_sample"}
    return {
        "F_all": [r.idx for r in rows],
        "F_disc": [r.idx for r in rows if r.segment == "llm_in_sample" or (r.segment == "pooled" and r.id not in has_in)],
        "F_hold": [r.idx for r in rows if r.segment == "clean_holdout"],
    }


def apply_fdr(rows: list[Row], fams: dict[str, list[int]], qs=(0.05, 0.10), p_override: dict[int, float] | None = None,
              tag: str = "") -> None:
    for fam, idx in fams.items():
        p = np.array([(p_override or {}).get(i, rows[i].p) for i in idx])
        qbh, qby = bh_adjusted(p), bh_adjusted(p, dependence=True)
        for q in qs:
            rb, ry = bh(p, q), by(p, q)
            for j, i in enumerate(idx):
                rows[i].surv[f"{fam}{tag}:BH{q:.2f}"] = bool(rb[j])
                rows[i].surv[f"{fam}{tag}:BY{q:.2f}"] = bool(ry[j])
        for j, i in enumerate(idx):
            rows[i].q_bh[f"{fam}{tag}"] = float(qbh[j])
            rows[i].q_by[f"{fam}{tag}"] = float(qby[j])


def survival_table(rows: list[Row], fams: dict[str, list[int]], tag: str = "") -> dict[str, Any]:
    out = {}
    for fam, idx in fams.items():
        sup = [rows[i] for i in idx if rows[i].verdict == "supports" and rows[i].claim == "effect"]
        sup_stat = [r for r in sup if r.source != "none"]
        d = {"m": len(idx), "m_with_stat": sum(rows[i].source != "none" for i in idx), "effect_supports": len(sup),
             "effect_supports_with_stat": len(sup_stat)}
        for q in (0.05, 0.10):
            for meth in ("BH", "BY"):
                k = f"{fam}{tag}:{meth}{q:.2f}"
                n_s = sum(r.surv.get(k, False) for r in sup)
                d[f"{meth}{q:.2f}"] = {"rejections_all": sum(rows[i].surv.get(k, False) for i in idx), "supports_surviving": n_s,
                                       "share_of_supports": n_s / len(sup) if sup else float("nan"),
                                       "share_of_supports_with_stat": n_s / len(sup_stat) if sup_stat else float("nan")}
        out[fam] = d
    return out


def latest(rows: list[Row]) -> dict[tuple[str, str], Row]:
    d = {}
    for r in rows:
        d[(r.id, r.segment)] = r          # rows are in file order, so the last one wins
    return d


def paired_in_hold(rows: list[Row], effect_only: bool = True) -> list[tuple[Row, Row]]:
    lt = latest(rows)
    pairs = []
    for (hid, seg), r_in in lt.items():
        if seg != "llm_in_sample" or r_in.verdict != "supports" or (effect_only and r_in.claim != "effect"):
            continue
        r_h = lt.get((hid, "clean_holdout"))
        if r_h is not None:
            pairs.append((r_in, r_h))
    return pairs


def power_decline(rows: list[Row], effect_only: bool = True) -> dict[str, Any]:
    items = []
    for r_in, r_h in paired_in_hold(rows, effect_only):
        if r_in.t is None or r_h.t is None or r_in.n_key is None or r_in.n_key != r_h.n_key or not r_in.n or not r_h.n:
            continue
        if not (r_in.source.startswith("t:") or r_in.source.startswith("p:")) or not (r_h.source.startswith("t:") or r_h.source.startswith("p:")):
            continue
        exp_t = abs(r_in.t) * math.sqrt(r_h.n / r_in.n)
        power = float(stats.norm.sf(1.644854 - exp_t))
        d_in, d_h = r_in.t / math.sqrt(r_in.n), r_h.t / math.sqrt(r_h.n)
        ratio = d_h / d_in if d_in != 0 else float("nan")
        items.append({"id": r_in.id, "family": r_in.family, "t_in": r_in.t, "n_in": r_in.n, "t_hold": r_h.t, "n_hold": r_h.n,
                      "n_key": r_in.n_key, "expected_t_hold": exp_t, "power_hold": power, "effect_ratio": ratio,
                      "hold_verdict": r_h.verdict})
    ratios = np.array([x["effect_ratio"] for x in items if math.isfinite(x["effect_ratio"])])
    powers = np.array([x["power_hold"] for x in items])
    k_below = int(np.sum(ratios < 1))
    sign_p = float(stats.binomtest(k_below, len(ratios), 0.5, alternative="greater").pvalue) if len(ratios) else float("nan")
    return {"n": len(items), "median_power": float(np.median(powers)) if len(powers) else float("nan"),
            "share_power_below_0.5": float(np.mean(powers < 0.5)) if len(powers) else float("nan"),
            "median_effect_ratio": float(np.median(ratios)) if len(ratios) else float("nan"),
            "n_ratio_below_1": k_below, "sign_test_p": sign_p, "items": items}


def replication(rows: list[Row]) -> dict[str, Any]:
    pairs = paired_in_hold(rows)
    a = b = c = d = 0
    det = []
    for r_in, r_h in pairs:
        s = r_in.surv.get("F_disc:BY0.10", False)
        rep = r_h.verdict == "supports"
        a += s and rep
        b += s and not rep
        c += (not s) and rep
        d += (not s) and not rep
        det.append({"id": r_in.id, "by_survives": s, "holdout": r_h.verdict, "p_in": r_in.p, "p_hold": r_h.p})
    p = float(stats.fisher_exact([[a, b], [c, d]], alternative="greater")[1]) if pairs else float("nan")
    return {"table": {"surv_rep": a, "surv_norep": b, "nosurv_rep": c, "nosurv_norep": d},
            "share_rep_surv": a / (a + b) if a + b else float("nan"), "share_rep_nosurv": c / (c + d) if c + d else float("nan"),
            "fisher_one_sided_p": p, "items": det}


def caliper(rows: list[Row], centre: float = 1.959964, width: float = 0.1, kinds: tuple[str, ...] | None = None) -> dict[str, Any]:
    zs = []
    for r in rows:
        if r.source == "none" or (kinds and not r.source.startswith(kinds)):
            continue
        p = min(max(r.p, 1e-300), 1.0)
        zs.append(float(stats.norm.isf(p / 2)))
    zs = np.array(zs)
    h = centre * width
    over = int(np.sum((zs >= centre) & (zs < centre + h)))
    under = int(np.sum((zs >= centre - h) & (zs < centre)))
    p = float(stats.binomtest(over, over + under, 0.5, alternative="greater").pvalue) if over + under else float("nan")
    return {"centre": centre, "half_width": h, "over": over, "under": under, "binom_one_sided_p": p, "n_z": len(zs)}


def verdicts_h(fdr: dict, pw: dict, rep: dict, cal: dict, hold_sign: dict) -> dict[str, tuple[str, dict]]:
    a = fdr["F_all"]
    s_by, s_bh = a["BY0.10"]["share_of_supports"], a["BH0.10"]["share_of_supports"]
    out = {
        "H-WS01-1": ("supports" if s_by < 0.5 else "contradicts", {"share": s_by, "n_supports": a["effect_supports"]}),
        "H-WS01-2": ("supports" if s_bh < 0.7 else "contradicts", {"share": s_bh, "n_supports": a["effect_supports"]}),
        "H-WS01-3": ("supports" if hold_sign["share_p_gt_0.10"] > 0.4 else "contradicts", hold_sign),
    }
    if pw["n"] < 8:
        out["H-WS01-4"] = ("inconclusive", {"n": pw["n"], "median_power": pw["median_power"]})
        out["H-WS01-7"] = ("inconclusive", {"n": pw["n"], "median_ratio": pw["median_effect_ratio"]})
    else:
        out["H-WS01-4"] = ("supports" if pw["median_power"] < 0.5 else "contradicts", {"n": pw["n"], "median_power": pw["median_power"]})
        mr = pw["median_effect_ratio"]
        v7 = "supports" if (mr < 1 and pw["sign_test_p"] < 0.10) else ("contradicts" if mr >= 1 else "inconclusive")
        out["H-WS01-7"] = (v7, {"n": pw["n"], "median_ratio": mr, "sign_test_p": pw["sign_test_p"]})
    t = rep["table"]
    small = (t["surv_rep"] + t["surv_norep"] < 5) or (t["nosurv_rep"] + t["nosurv_norep"] < 5)
    if small:
        v5 = "inconclusive"
    elif rep["fisher_one_sided_p"] < 0.10:
        v5 = "supports"
    elif rep["share_rep_surv"] <= rep["share_rep_nosurv"]:
        v5 = "contradicts"
    else:
        v5 = "inconclusive"
    out["H-WS01-5"] = (v5, {"table": t, "fisher_p": rep["fisher_one_sided_p"], "share_rep_surv": rep["share_rep_surv"],
                            "share_rep_nosurv": rep["share_rep_nosurv"]})
    v6 = "supports" if cal["binom_one_sided_p"] < 0.10 else ("contradicts" if cal["over"] <= cal["under"] else "inconclusive")
    out["H-WS01-6"] = (v6, {k: cal[k] for k in ("over", "under", "binom_one_sided_p", "n_z")})
    return out


def holdout_sign(rows: list[Row]) -> dict[str, Any]:
    hs = [r for r in rows if r.segment == "clean_holdout" and r.verdict == "supports" and r.claim == "effect"]
    st = [r for r in hs if r.source != "none"]
    return {"n_holdout_supports": len(hs), "n_with_stat": len(st), "n_p_gt_0.10": sum(r.p > 0.10 for r in st),
            "share_p_gt_0.10": (sum(r.p > 0.10 for r in st) / len(st)) if st else float("nan"),
            "n_no_stat": len(hs) - len(st)}


def hypothesis_table(rows: list[Row], reg: dict) -> list[dict[str, Any]]:
    """Per hypothesis with any supports row: discovery (F_disc, BY 10%) then two-stage replication: the holdout p of the
    selected discoveries tested with BH at 10% within the selected set (Bogomolov-Heller style; descriptive)."""
    lt = latest(rows)
    ids = sorted({r.id for r in rows if r.verdict == "supports"})
    sel = []
    out = []
    for hid in ids:
        r_in = lt.get((hid, "llm_in_sample")) or lt.get((hid, "pooled"))
        r_h = lt.get((hid, "clean_holdout"))
        d = {"id": hid, "family": reg.get(hid, {}).get("family"), "claim": claim_type(reg.get(hid, {})),
             "disc_segment": r_in.segment if r_in else None, "disc_verdict": r_in.verdict if r_in else None,
             "disc_p": r_in.p if r_in else None, "disc_source": r_in.source if r_in else None,
             "disc_q_by": (r_in.q_by.get("F_disc") if r_in else None), "disc_q_by_all": (r_in.q_by.get("F_all") if r_in else None),
             "disc_by10": bool(r_in and r_in.verdict == "supports" and r_in.surv.get("F_disc:BY0.10")),
             "hold_verdict": r_h.verdict if r_h else None, "hold_p": r_h.p if r_h else None,
             "hold_source": r_h.source if r_h else None, "hold_q_by": r_h.q_by.get("F_hold") if r_h else None}
        out.append(d)
        if d["disc_by10"] and r_h is not None and r_h.verdict == "supports" and r_h.source != "none":
            sel.append(d)
    if sel:
        rej = bh(np.array([d["hold_p"] for d in sel]), 0.10)
        for d, k in zip(sel, rej):
            d["replicated_two_stage"] = bool(k)
    for d in out:
        d.setdefault("replicated_two_stage", False)
        if d["claim"] == "null":
            d["class"] = "null_claim (needs equivalence / power, not FDR)"
        elif d["replicated_two_stage"]:
            d["class"] = "robust: discovery survives BY10 and holdout replicates (BH10 within selected)"
        elif d["disc_by10"] and d["hold_verdict"] == "supports":
            d["class"] = "discovery survives, holdout sign-only"
        elif d["disc_by10"] and d["hold_verdict"] is None:
            d["class"] = "discovery survives, no holdout row"
        elif d["disc_by10"]:
            d["class"] = f"discovery survives, holdout {d['hold_verdict']}"
        elif d["hold_verdict"] == "supports" and d["hold_q_by"] is not None and d["hold_q_by"] <= 0.10:
            d["class"] = "holdout-only significant (discovery fails BY10)"
        else:
            d["class"] = "fragile: supports not surviving FDR"
    return out


# --------------------------------------------------------------------------------------------------- rules x evidence
def rules_evidence(rules: list[dict], rows: list[Row], reg: dict) -> list[dict[str, Any]]:
    by_id = defaultdict(list)
    for r in rows:
        by_id[r.id].append(r)
    out = []
    for ru in rules:
        if ru.get("status") == "retired":
            continue
        ev = [e for e in (ru.get("evidence") or []) if isinstance(e, str)]
        ev_rows = [r for e in ev for r in by_id.get(e, [])]
        sup = [r for r in ev_rows if r.verdict == "supports" and r.claim == "effect"]
        null_sup = [r for r in ev_rows if r.verdict == "supports" and r.claim == "null"]
        surv = [r for r in sup if r.surv.get("F_all:BY0.10")]
        hold = [r for r in ev_rows if r.segment == "clean_holdout"]
        by_id_lt = {}
        for r in ev_rows:
            by_id_lt[(r.id, r.segment)] = r
        flipped = sorted({hid for (hid, seg), r in by_id_lt.items() if seg == "clean_holdout" and r.verdict == "contradicts"
                          and (hid, "llm_in_sample") in by_id_lt and by_id_lt[(hid, "llm_in_sample")].verdict == "supports"})
        out.append({
            "rule": ru["id"], "kind": ru.get("kind"), "confidence": ru.get("confidence"), "status": ru.get("status"),
            "text": ru.get("rule", "")[:200], "evidence": ev, "n_rows": len(ev_rows), "n_supports": len(sup), "n_supports_BY10": len(surv),
            "n_any_BY10": sum(bool(r.surv.get("F_all:BY0.10")) for r in ev_rows),
            "holdout_verdicts": sorted({f"{r.id}:{r.verdict}" for r in hold}),
            "untested_evidence": [e for e in ev if e in reg and not by_id.get(e)],
            "non_hypothesis_evidence": [e for e in ev if e not in reg],
            "flag_fdr_fragile": bool(sup) and not surv,
            "n_null_claim_supports": len(null_sup),
            "flag_null_based": bool(ev_rows) and not sup and not any(r.surv.get("F_all:BY0.10") for r in ev_rows),
            "flag_in_sample_support_flipped_in_holdout": flipped,
        })
    return out


# ------------------------------------------------------------------------------------------------------ debt ledger
LIVE_IDS = {"H-NN10", "H-LIVE1", "H-LIVE2", "H-XS4", "H-SENT1", "H-PNN1", "H-STRESS0", "H-TR1", "H-EXE1", "H-STOP1", "H-NS4", "H-NN8"}
DV_W = {"high": 3.0, "medium": 2.0, "low": 1.0}
TYPE_W = {"contradicted-but-used": 3.0, "look-ahead": 3.0, "verdict-error": 2.5, "mis-specified": 2.5, "holdout-contradicts": 2.5,
          "fdr-fragile": 2.0, "sign-only-holdout": 2.0, "underpowered": 2.0, "selection": 1.5, "no-statistic": 1.5,
          "null-claim-untested-equivalence": 1.5, "untested": 1.0, "powered-null": 0.3}
COST_W = {"cheap": 1.25, "medium": 1.0, "expensive": 0.8}


def decision_value(hid: str, rules_by_ev: dict[str, list[dict]]) -> tuple[str, list[str]]:
    rs = rules_by_ev.get(hid, [])
    active = [r["id"] for r in rs if r.get("status") == "active"]
    prov = [r["id"] for r in rs if r.get("status") == "provisional"]
    if hid in LIVE_IDS or any(r.get("confidence") == "A" for r in rs if r.get("status") == "active"):
        return "high", active + prov
    if active:
        return "medium", active + prov
    return ("low" if not prov else "medium"), prov


def ledger_auto(res: dict, reg: dict, rows: list[Row]) -> list[dict[str, Any]]:
    rules = load_rules()
    rules_by_ev: dict[str, list[dict]] = defaultdict(list)
    for ru in rules:
        if ru.get("status") != "retired":
            for e in ru.get("evidence") or []:
                rules_by_ev[e].append(ru)
    have = {r.id for r in rows}
    items: list[dict[str, Any]] = []

    def add(hid, typ, problem, suggested, cost="cheap", evidence=""):
        dv, used = decision_value(hid, rules_by_ev)
        h = reg.get(hid, {})
        items.append({"hypothesis": hid, "family": h.get("family"), "title": h.get("title", "")[:120], "type": typ,
                      "problem": problem, "evidence": evidence, "decision_value": dv, "used_by_rules": used,
                      "suggested_test": suggested, "cost": cost, "source": "auto"})

    for d in res["hypotheses"]:
        hid, c = d["id"], d["class"]
        if c.startswith("fragile"):
            add(hid, "fdr-fragile", "recorded supports without an auditable statistic (no p, t, z or CI in the row)"
                if d["disc_source"] in (None, "none") else f"recorded supports but discovery p={d['disc_p']:.3g} (q_BY={d['disc_q_by'] if d['disc_q_by'] is None else round(d['disc_q_by'], 3)}) "
                f"does not survive registry-wide BY10; holdout {d['hold_verdict']} p={d['hold_p'] if d['hold_p'] is None else round(d['hold_p'], 3)}"
                if d["disc_p"] is not None else "recorded supports without an auditable statistic",
                "re-test on the growing clean_holdout with the registered statistic recorded (estimate, SE, n, HAC); count it in the registry family",
                evidence=f"fdr.json hypotheses[{hid}]")
        elif c == "discovery survives, holdout sign-only":
            add(hid, "sign-only-holdout", f"in-sample survives BY10 but the holdout 'supports' is a sign agreement (p={d['hold_p']:.3g})",
                "fix the holdout read date for when n_hold reaches 80% power (see power_z) and re-read once", evidence=f"fdr.json hypotheses[{hid}]")
        elif c == "discovery survives, holdout contradicts":
            add(hid, "holdout-contradicts", f"in-sample survives BY10 (p={d['disc_p']:.3g}) but the holdout contradicts",
                "treat as not replicated; retire dependent rules or downgrade to C", evidence=f"fdr.json hypotheses[{hid}]")
        elif c.startswith("null_claim"):
            add(hid, "null-claim-untested-equivalence", "a null / bound claim recorded as supported without an equivalence test (TOST) or a power statement",
                "state the smallest effect of interest and run TOST / report the MDE", evidence=f"fdr.json hypotheses[{hid}]")
    for hid, h in reg.items():
        if hid.startswith("H-WS") or hid in have:
            continue
        fwd = forward_only(h)
        add(hid, "untested", "registered, no result row" + (" (forward-only by design)" if fwd else ""),
            "schedule the forward read (date, n needed) in forward_tests" if fwd else "run the registered test or withdraw the id",
            cost="medium" if fwd else "cheap")
    for hid, h in reg.items():
        if post_hoc(h) and not hid.startswith("H-WS"):
            add(hid, "selection", "registered after seeing results (post hoc)", "count as an extra trial; confirm only on data after its registration date")
    for x in res["power_z"]["items"]:
        if x["power_hold"] < 0.5:
            add(x["id"], "underpowered", f"holdout power {x['power_hold']:.2f} for the in-sample effect (n_hold {x['n_hold']:.0f}, needs {x['n_hold_required_80']:.0f} {x['n_key']})",
                "do not read the holdout before n_hold reaches the required n", evidence="fdr.json power_z")
    for ru in res["rules"]:
        if ru["status"] != "active":
            continue
        dv = "high" if ru["confidence"] == "A" else "medium"
        base = {"family": "rules", "title": f"{ru['rule']} ({ru['confidence']}, {ru['kind']}): {ru['text'][:140]}", "evidence": "research/rules.yaml",
                "decision_value": dv, "used_by_rules": [ru["rule"]], "cost": "cheap", "source": "auto"}
        if ru["flag_in_sample_support_flipped_in_holdout"]:
            items.append({**base, "hypothesis": ",".join(ru["flag_in_sample_support_flipped_in_holdout"]), "type": "contradicted-but-used",
                          "problem": "cites a hypothesis that was supported in-sample and contradicted in the clean holdout",
                          "suggested_test": "check the rule's direction against the holdout; downgrade to C unless a later read replicates"})
        if ru["flag_fdr_fragile"]:
            items.append({**base, "hypothesis": ",".join(e for e in ru["evidence"] if e.startswith("H-")), "type": "fdr-fragile",
                          "problem": f"rests on {ru['n_supports']} supports rows, none surviving registry-wide BY10",
                          "suggested_test": "grade by FDR-surviving evidence; re-test the key hypothesis on the grown holdout with the statistic recorded"})
        if ru["flag_null_based"] and ru["confidence"] in ("A", "B"):
            items.append({**base, "hypothesis": ",".join(e for e in ru["evidence"] if e.startswith("H-")), "type": "underpowered",
                          "problem": f"{ru['kind']} rule built only on contradicts / inconclusive rows (absence of evidence); power never stated",
                          "suggested_test": "report the MDE of each cited test; keep the rule only where the MDE is below the smallest effect that would matter"})
    merged: dict[tuple[str, str], dict[str, Any]] = {}
    for it in items:
        k = (it["type"], it["hypothesis"])
        if k in merged and it["family"] == "rules" and merged[k]["family"] == "rules":
            m0 = merged[k]
            m0["used_by_rules"] = sorted(set(m0["used_by_rules"]) | set(it["used_by_rules"]))
            m0["title"] = "rules " + ", ".join(m0["used_by_rules"])
            if it["decision_value"] == "high":
                m0["decision_value"] = "high"
        else:
            merged[k] = it
    items = list(merged.values())
    for it in items:
        it["score"] = round(DV_W[it["decision_value"]] * TYPE_W.get(it["type"], 1.0) * COST_W.get(it["cost"], 1.0), 3)
    items.sort(key=lambda x: -x["score"])
    for k, it in enumerate(items, 1):
        it["rank"] = k
    return items


LEDGER_MANUAL = OUT / "ledger_manual.json"     # qualitative items (family audit, red team, refine), merged when present
LEDGER_COPY = ROOT / "research" / "reports" / "ws01_research_debt_2026-11-27.json"


def write_ledger(res: dict[str, Any]) -> dict[str, Any]:
    items = list(res["ledger_auto"])
    if LEDGER_MANUAL.is_file():
        manual = json.loads(LEDGER_MANUAL.read_text(encoding="utf-8"))
        drop = {(r["type"], r["hypothesis"]) for it in manual for r in it.get("replaces", [])}
        items = [it for it in items if (it["type"], it["hypothesis"]) not in drop]
        for it in manual:
            it.setdefault("source", "manual")
            it["score"] = round(DV_W.get(it.get("decision_value", "low"), 1.0) * TYPE_W.get(it.get("type"), 1.0)
                                * COST_W.get(it.get("cost", "medium"), 1.0) * (1.2 if it.get("severity") == "high" else 1.0), 3)
            items.append(it)
    items.sort(key=lambda x: (-x["score"], x["type"], x["hypothesis"]))
    for k, it in enumerate(items, 1):
        it["rank"] = k
    f = res["fdr"]["F_all"]
    out = {
        "version": "v1" if not LEDGER_MANUAL.is_file() else "v2",
        "workstream": "WS01 AUDIT", "snapshot": res["snapshot"],
        "scoring": {"score": "decision_value weight x type weight x cost weight (x1.2 for high-severity manual items)",
                    "decision_value": DV_W, "type": TYPE_W, "cost": COST_W,
                    "decision_value_rule": "high = live-desk id or cited by an active A-grade rule; medium = cited by an active or provisional rule; low = otherwise"},
        "fdr_headline": {"m": f["m"], "m_with_stat": f["m_with_stat"], "effect_supports": f["effect_supports"],
                         "BH10_supports_surviving": f["BH0.10"]["supports_surviving"], "BY10_supports_surviving": f["BY0.10"]["supports_surviving"]},
        "n_items": len(items), "items": items,
    }
    txt = json.dumps(out, indent=1, default=float)
    (OUT / "research_debt.json").write_text(txt, encoding="utf-8")
    LEDGER_COPY.write_text(txt, encoding="utf-8")
    return out


# -------------------------------------------------------------------------------------------------------- run / CLI
def run(write: bool = True) -> dict[str, Any]:
    reg, raw = load_registry(), load_rows()
    rows = build_rows(reg, raw)
    fams = families(rows)
    apply_fdr(rows, fams)
    # sensitivities
    no_adj = {r.idx: 1.0 for r in rows if r.flag == "adjusted_only"}
    apply_fdr(rows, fams, p_override=no_adj, tag="|no_adj")
    one_sided = {r.idx: r.p / 2 for r in rows if r.verdict == "supports" and r.source != "none"}
    apply_fdr(rows, fams, p_override=one_sided, tag="|one_sided")
    lt_idx = {r.idx for r in latest(rows).values()}
    fams_latest = {k: [i for i in v if i in lt_idx] for k, v in fams.items()}
    apply_fdr(rows, fams_latest, tag="|latest")
    fdr = survival_table(rows, fams)
    sens = {"no_adjusted_only": survival_table(rows, fams, "|no_adj"), "one_sided_supports": survival_table(rows, fams, "|one_sided"),
            "latest_per_id_segment": survival_table(rows, fams_latest, "|latest")}
    pw, pw_all = power_decline(rows), power_decline(rows, effect_only=False)
    rep = replication(rows)
    cal = caliper(rows)
    cal2 = caliper(rows, centre=2.0, width=0.1)
    cal_rt = caliper(rows, kinds=("p:", "t:"))
    hs = holdout_sign(rows)
    vh = verdicts_h(fdr, pw, rep, cal, hs)
    pz = power_z(rows)
    vh["H-WS01-8"] = (("inconclusive" if pz["n"] < 8 else ("supports" if pz["median_power"] < 0.5 else "contradicts")),
                      {"n": pz["n"], "median_power": pz["median_power"], "share_power_below_0.5": pz["share_power_below_0.5"]})
    src = defaultdict(int)
    for r in rows:
        src[r.source.split(":")[0]] += 1
    claims = {h: claim_type(v) for h, v in reg.items()}
    res = {
        "snapshot": {"rows": len(rows), "sha256_first_rows": SNAPSHOT_SHA, "registry_ids": len(reg)},
        "extraction_sources": dict(src),
        "null_claim_ids": sorted(h for h, c in claims.items() if c == "null"),
        "fdr": fdr, "sensitivity": sens, "holdout_sign": hs, "power_decline": pw, "power_decline_all_claims": pw_all, "power_z": pz,
        "replication": rep, "caliper_1.96": cal, "caliper_2.0": cal2, "caliper_raw_t_only": cal_rt,
        "verdicts": {k: {"verdict": v, "metrics": m} for k, (v, m) in vh.items()},
        "rules": rules_evidence(load_rules(), rows, reg),
        "hypotheses": hypothesis_table(rows, reg),
    }
    res["ledger_auto"] = ledger_auto(res, reg, rows)
    if write:
        OUT.mkdir(parents=True, exist_ok=True)
        write_ledger(res)
        (OUT / "fdr.json").write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")
        (OUT / "rows.json").write_text(json.dumps([asdict(r) for r in rows], default=float), encoding="utf-8")
    res["_rows"] = rows
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "coverage"])
    a = ap.parse_args()
    if a.cmd == "coverage":
        reg, raw = load_registry(), load_rows()
        rows = build_rows(reg, raw)
        for r in rows:
            if r.verdict == "supports" and r.source == "none":
                print(r.idx, r.id, r.segment, json.dumps(raw[r.idx].get("metrics"), default=str)[:400])
        return
    res = run()
    print(json.dumps({k: res[k] for k in ("extraction_sources", "verdicts")}, indent=1, default=float))


if __name__ == "__main__":
    main()
