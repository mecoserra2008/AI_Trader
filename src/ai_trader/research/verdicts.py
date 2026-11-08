"""Mechanical verdicts for pre-registered hypotheses (research/hypotheses.yaml).

Each function receives the study outputs and returns {segment: (verdict, metrics)}. The success
criteria are transcribed from the registry *before* the results are inspected. In the holdout,
only the sign is required (samples are small); fewer than `MIN_HOLDOUT` observations is
"inconclusive".

Weekend statistics cluster by weekend: each weekend's value is first averaged across symbols,
because all coins share the same weekend news.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from ai_trader.research import evaluate as ev

MIN_HOLDOUT = 8
Verdicts = dict[str, tuple[str, dict]]


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


def _judge(seg: str, ok_in: bool, sign_ok: bool, n: int) -> str:
    if seg == "clean_holdout":
        if n < MIN_HOLDOUT:
            return "inconclusive"
        return "supports" if sign_ok else "contradicts"
    return "supports" if ok_in else ("contradicts" if not sign_ok else "inconclusive")


def h_w1(w: pl.DataFrame) -> Verdicts:
    out = {}
    for seg, d in ev.split_by_cutoff(w).items():
        x = _by_weekend(d, "rv_ratio")
        med, frac = float(np.median(x)) if len(x) else float("nan"), float((x < 1).mean()) if len(x) else float("nan")
        ok = med < 0.9 and frac > 0.6
        out[seg] = (_judge(seg, ok, med < 1, len(x)), {"median_rv_ratio": med, "frac_below_1": frac, "n_weekends": len(x)})
    return out


def _signed_bucket(w: pl.DataFrame, mask: pl.Expr, want_sign: int, t_need: float) -> Verdicts:
    out = {}
    for seg, d in ev.split_by_cutoff(w).items():
        x = _by_weekend(d.filter(mask), "signed_post_z")
        m, t, n = _t(x)
        ok = np.sign(m) == want_sign and np.isfinite(t) and t * want_sign > t_need
        out[seg] = (_judge(seg, ok, np.sign(m) == want_sign, n), {"mean_signed_post_z": m, "t": t, "n_weekends": n})
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
        out[seg] = (_judge(seg, ok, m > 0, n), {"fill_rate": float(d["gap_filled"].mean()), "baseline": float(d["fill_baseline"].fill_nan(None).mean()), "excess": m, "t": t, "n_weekends": n})
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
        out[seg] = (_judge(seg, ok, ratio > 1, len(x)), {"ratio": ratio, "ci95": [float(lo), float(hi)], "n_weekends": len(x)})
    return out


def h_w5b(w: pl.DataFrame) -> Verdicts:
    return h_w5(w, col="mon_open_abs_rs", base_col="other_days_open_abs_rs")


def _cell_signs(sb: pl.DataFrame, strategies: list[str], pos_cells: list[str], neg_cells: list[str], need: int) -> Verdicts:
    out = {}
    for seg in ("llm_in_sample", "clean_holdout"):
        d = sb.filter(pl.col("segment") == seg)
        passed, detail = 0, {}
        for s in strategies:
            x = {r["cell"]: r["mean_shrunk"] for r in d.filter(pl.col("strategy") == s).iter_rows(named=True)}
            ok = all(x.get(c, np.nan) > 0 for c in pos_cells) and all(x.get(c, np.nan) <= 0 for c in neg_cells)
            passed += ok
            detail[s] = {c: x.get(c) for c in pos_cells + neg_cells}
        n = int(d.filter(pl.col("cell") == "ALL")["n_ts"].min() or 0) if d.height else 0
        verdict = ("supports" if passed >= need else "contradicts") if (seg == "llm_in_sample" or n >= 500) else "inconclusive"
        out[seg] = (verdict, {"strategies_passing": passed, "needed": need, "cells": detail, "n_ts_min": n})
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
        out[seg] = (_judge(seg, bool(strong), bool(sign_ok), n), {c: {"mean": g[c]["mean_signed_z"], "t": g[c]["t"], "n": g[c]["n_events"]} for c in ("R3", "R9", "R4", "R5") if c in g})
    return out
