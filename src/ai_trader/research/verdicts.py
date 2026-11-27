"""Mechanical verdicts for pre-registered hypotheses (research/hypotheses.yaml).

Each function receives the study outputs and returns {segment: (verdict, metrics)}. The success
criteria are transcribed from the registry *before* the results are inspected.

Holdout rule (D7a, 2026-11-27, user-approved; WS01 RT-SYS-2): a clean_holdout "supports" must pass a
declared ONE-SIDED test of the registered direction at `HOLDOUT_ALPHA` (5%, the registry's level; a
hypothesis that registers its own level passes it as `alpha`), not just keep the sign. Fewer than
`MIN_HOLDOUT` observations is "inconclusive"; the wrong sign is "contradicts"; the right sign that fails
the test is "inconclusive". The one-sided p is stored in the metrics as `p_one_sided` (with `alpha`).
Before 2026-11-27 the holdout was judged on sign only: research/reports/verdict_fix_diff_2026-11-27.md
lists the recorded verdicts that would change (dry run; recorded rows are never rewritten).

Weekend statistics cluster by weekend: each weekend's value is first averaged across symbols,
because all coins share the same weekend news.
"""

from __future__ import annotations

import math

import numpy as np
import polars as pl
from scipy import stats

from ai_trader.research import evaluate as ev

MIN_HOLDOUT = 8
HOLDOUT_ALPHA = 0.05   # one-sided level a clean_holdout "supports" must pass (D7a)
Verdicts = dict[str, tuple[str, dict]]


# ---------------------------------------------------------------------- one-sided holdout tests (D7a)

def p_one_sided_t(t: float | None, n: float | None, direction: int = 1) -> float:
    """One-sided p of a t statistic in the registered direction (+1: effect > 0, -1: effect < 0); Student-t with
    n - 1 df when n >= 3 is known, else the normal. NaN in, NaN out."""
    if t is None or not np.isfinite(t):
        return float("nan")
    x = float(t) * (1 if direction >= 0 else -1)
    if n is not None and np.isfinite(n) and n >= 3:
        return float(stats.t.sf(x, df=float(n) - 1))
    return float(stats.norm.sf(x))


def p_one_sided_binom(k: int, n: int, p0: float = 0.5) -> float:
    """One-sided sign test: P(X >= k) for X ~ Binomial(n, p0)."""
    if n <= 0:
        return float("nan")
    return float(stats.binomtest(int(k), int(n), p0, alternative="greater").pvalue)


def p_one_sided_ci(est: float, lo: float, hi: float, null: float = 0.0, direction: int = 1, z: float = 1.959964, log: bool = False) -> float:
    """Approximate one-sided p from a point estimate and its (normal-theory) CI: se = (hi - lo) / (2 z); on the log
    scale for ratios (log=True, null = 1). Used where only a CI was recorded."""
    vals = (est, lo, hi)
    if any(v is None or not np.isfinite(v) for v in vals) or hi <= lo:
        return float("nan")
    if log:
        if min(vals) <= 0 or null <= 0:
            return float("nan")
        est, lo, hi, null = math.log(est), math.log(lo), math.log(hi), math.log(null)
    se = (hi - lo) / (2 * z)
    return float(stats.norm.sf((est - null) / se * (1 if direction >= 0 else -1))) if se > 0 else float("nan")


def p_one_sided_boot(draws: np.ndarray, null: float = 0.0, direction: int = 1) -> float:
    """One-sided bootstrap p: share of draws at or beyond the null against the registered direction, (k + 1) / (B + 1)."""
    d = np.asarray(draws, float)
    d = d[np.isfinite(d)]
    if not len(d):
        return float("nan")
    k = int((d <= null).sum()) if direction >= 0 else int((d >= null).sum())
    return (k + 1) / (len(d) + 1)


def holdout_passes(p: float | None, alpha: float = HOLDOUT_ALPHA) -> bool:
    return p is not None and bool(np.isfinite(p)) and p <= alpha


def _t(x: np.ndarray) -> tuple[float, float, int]:
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 3:
        return (float(np.mean(x)) if n else float("nan"), float("nan"), n)
    return float(x.mean()), float(x.mean() / (x.std(ddof=1) / np.sqrt(n))), n


def _by_weekend(w: pl.DataFrame, col: str) -> np.ndarray:
    # fill_nan(None): polars aggregates propagate NaN, while nulls are skipped.
    x = w.group_by("weekend_id").agg(pl.col(col).cast(pl.Float64).fill_nan(None).mean()).sort("weekend_id")[col].to_numpy()
    return x[np.isfinite(x)]


def _judge(seg: str, ok_in: bool, sign_ok: bool, n: int, p_hold: float | None = None, alpha: float = HOLDOUT_ALPHA) -> str:
    """In-sample: the registered success rule (`ok_in`), "contradicts" on the wrong sign, else "inconclusive".
    clean_holdout: < MIN_HOLDOUT -> inconclusive; wrong sign -> contradicts; right sign -> "supports" only if the
    declared one-sided test passes (`p_hold` <= alpha), else "inconclusive". A caller that passes no p cannot
    get a holdout "supports" (sign alone is not evidence, D7a)."""
    if seg == "clean_holdout":
        if n < MIN_HOLDOUT:
            return "inconclusive"
        if not sign_ok:
            return "contradicts"
        return "supports" if holdout_passes(p_hold, alpha) else "inconclusive"
    return "supports" if ok_in else ("contradicts" if not sign_ok else "inconclusive")


def h_w1(w: pl.DataFrame) -> Verdicts:
    out = {}
    for seg, d in ev.split_by_cutoff(w).items():
        x = _by_weekend(d, "rv_ratio")
        med, frac = float(np.median(x)) if len(x) else float("nan"), float((x < 1).mean()) if len(x) else float("nan")
        ok = med < 0.9 and frac > 0.6
        p = p_one_sided_binom(int((x < 1).sum()), len(x)) if len(x) else float("nan")   # median < 1: sign test
        out[seg] = (_judge(seg, ok, med < 1, len(x), p), {"median_rv_ratio": med, "frac_below_1": frac, "n_weekends": len(x),
                                                          "p_one_sided": p, "test": "sign test, rv_ratio < 1", "alpha": HOLDOUT_ALPHA})
    return out


def _signed_bucket(w: pl.DataFrame, mask: pl.Expr, want_sign: int, t_need: float) -> Verdicts:
    out = {}
    for seg, d in ev.split_by_cutoff(w).items():
        x = _by_weekend(d.filter(mask), "signed_post_z")
        m, t, n = _t(x)
        ok = np.sign(m) == want_sign and np.isfinite(t) and t * want_sign > t_need
        p = p_one_sided_t(t, n, want_sign)
        out[seg] = (_judge(seg, ok, np.sign(m) == want_sign, n, p), {"mean_signed_post_z": m, "t": t, "n_weekends": n, "p_one_sided": p,
                                                                     "test": "one-sided t over weekends", "alpha": HOLDOUT_ALPHA})
    return out


def h_w2(w: pl.DataFrame) -> Verdicts:
    return _signed_bucket(w, (pl.col("wk_z").abs() >= 1) & (pl.col("vol_ratio") < 0.9), -1, 2.0)


def h_w3(w: pl.DataFrame) -> Verdicts:
    return _signed_bucket(w, (pl.col("wk_z").abs() >= 2) & (pl.col("vol_ratio") >= 1.2), +1, 2.0)


def h_w4(w: pl.DataFrame) -> Verdicts:
    out = {}
    for seg, d in ev.split_by_cutoff(w).items():
        d = d.with_columns((pl.col("gap_filled").cast(pl.Float64) - pl.col("fill_baseline")).alias("excess"))
        m, t, n = _t(_by_weekend(d, "excess"))
        ok = m > 0.05 and t > 2
        p = p_one_sided_t(t, n, +1)
        out[seg] = (_judge(seg, ok, m > 0, n, p), {"fill_rate": float(d["gap_filled"].mean()), "baseline": float(d["fill_baseline"].fill_nan(None).mean()),
                                                   "excess": m, "t": t, "n_weekends": n, "p_one_sided": p, "test": "one-sided t over weekends",
                                                   "alpha": HOLDOUT_ALPHA})
    return out


def h_w5(w: pl.DataFrame, n_boot: int = 2000, seed: int = 0, col: str = "mon_open_abs_z", base_col: str = "other_days_open_abs_z") -> Verdicts:
    out = {}
    rng = np.random.default_rng(seed)
    for seg, d in ev.split_by_cutoff(w).items():
        e = d.filter(pl.col("wk_z").abs() >= 2)
        base = float(d[base_col].fill_nan(None).mean()) if d.height else float("nan")
        x = _by_weekend(e, col) if e.height else np.array([])
        x = x[np.isfinite(x)]
        if len(x) < 3 or not np.isfinite(base):
            out[seg] = ("inconclusive", {"n_weekends": len(x)})
            continue
        ratio = float(x.mean() / base)
        boots = np.array([rng.choice(x, len(x)).mean() / base for _ in range(n_boot)])
        lo, hi = np.quantile(boots, [0.025, 0.975])
        ok = ratio > 1.3 and lo > 1
        p = p_one_sided_boot(boots, 1.0, +1)
        out[seg] = (_judge(seg, ok, ratio > 1, len(x), p), {"ratio": ratio, "ci95": [float(lo), float(hi)], "n_weekends": len(x), "p_one_sided": p,
                                                            "test": "one-sided bootstrap, ratio > 1", "alpha": HOLDOUT_ALPHA})
    return out


def h_w5b(w: pl.DataFrame) -> Verdicts:
    return h_w5(w, col="mon_open_abs_rs", base_col="other_days_open_abs_rs")


def _cell_signs(sb: pl.DataFrame, strategies: list[str], pos_cells: list[str], neg_cells: list[str], need: int) -> Verdicts:
    out = {}
    for seg in ("llm_in_sample", "clean_holdout"):
        d = sb.filter(pl.col("segment") == seg)
        passed, passed_test, detail, p_iut = 0, 0, {}, {}
        for s in strategies:
            rows = {r["cell"]: r for r in d.filter(pl.col("strategy") == s).iter_rows(named=True)}
            x = {c: r["mean_shrunk"] for c, r in rows.items()}
            ok = all(x.get(c, np.nan) > 0 for c in pos_cells) and all(x.get(c, np.nan) <= 0 for c in neg_cells)
            passed += ok
            # intersection-union test: every sign condition one-sided on its cell's NW t (normal); the largest p decides
            ps = [p_one_sided_t((rows.get(c) or {}).get("t_nw", np.nan), None, +1 if c in pos_cells else -1) for c in pos_cells + neg_cells]
            p_iut[s] = float(max(ps)) if ps and all(np.isfinite(ps)) else float("nan")
            passed_test += bool(ok and holdout_passes(p_iut[s]))
            detail[s] = {c: x.get(c) for c in pos_cells + neg_cells}
        n = int(d.filter(pl.col("cell") == "ALL")["n_ts"].min() or 0) if d.height else 0
        if seg == "llm_in_sample":
            verdict = "supports" if passed >= need else "contradicts"
        elif n < 500:
            verdict = "inconclusive"
        else:   # holdout (D7a): the signs of `need` strategies AND their one-sided tests
            verdict = "supports" if passed_test >= need else ("contradicts" if passed < need else "inconclusive")
        out[seg] = (verdict, {"strategies_passing": passed, "strategies_passing_test": passed_test, "needed": need, "cells": detail, "n_ts_min": n,
                              "p_one_sided": p_iut, "test": "intersection-union of one-sided NW t per cell", "alpha": HOLDOUT_ALPHA})
    return out


def h_r1(sb: pl.DataFrame) -> Verdicts:
    return _cell_signs(sb, ["tsmom", "ema_stack", "donchian_breakout"], ["R1", "R2", "R7", "R8"], ["R5", "R6"], 2)


def h_r2(sb: pl.DataFrame) -> Verdicts:
    return _cell_signs(sb, ["zscore_reversion", "rsi_reversion"], ["R4", "R5", "R6"], ["R2", "R3", "R8", "R9"], 2)


def h_s1(shocks: pl.DataFrame, h_bars: int) -> Verdicts:
    out = {}
    for seg in ("llm_in_sample", "clean_holdout"):
        d = shocks.filter((pl.col("segment") == seg) & (pl.col("h_bars") == h_bars))
        g = {r["cell"]: r for r in d.iter_rows(named=True)}
        cont = [g[c] for c in ("R3", "R9") if c in g]
        rev = [g[c] for c in ("R4", "R5") if c in g]
        sign_ok = all(r["mean_signed_z"] > 0 for r in cont) and all(r["mean_signed_z"] < 0 for r in rev) and cont and rev
        strong = sign_ok and all(abs(r["t"]) > 2 for r in cont + rev if np.isfinite(r["t"]))
        n = sum(r["n_events"] for r in cont + rev)
        ps = [p_one_sided_t(r["t"], r["n_events"], +1) for r in cont] + [p_one_sided_t(r["t"], r["n_events"], -1) for r in rev]
        p = float(max(ps)) if ps and all(np.isfinite(ps)) else float("nan")   # intersection-union over the four cells
        m = {c: {"mean": g[c]["mean_signed_z"], "t": g[c]["t"], "n": g[c]["n_events"]} for c in ("R3", "R9", "R4", "R5") if c in g}
        out[seg] = (_judge(seg, bool(strong), bool(sign_ok), n, p), {**m, "p_one_sided": p, "test": "intersection-union of one-sided t per cell",
                                                                     "alpha": HOLDOUT_ALPHA})
    return out
