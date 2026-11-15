"""H-OBG1 / H-OBG2: generated BTC order books (sim.ob_gen, executed by at-match) vs held-out real days, and half-real /
half-generated training data for next-minute volatility (agent F5, plan research/plans/2026-11-13_btc_field_program.md).
Deterministic research tools only (principle 1); generated data never reach the LLM layer.

    PYTHONIOENCODING=utf-8 uv run --extra ml python -m ai_trader.research.obgen_study split
    ... daycal [dates]      per-day calibration on the FIT half (00-12 UTC) + indirect inference of theta / phi
    ... obg1 [dates]        generate the TEST half (12-24 UTC) of every held-out day driven by its real 1-min path
    ... twins [dates]       generate a 24 h "twin" of every day driven by a block bootstrap of its own 1-min path
    ... obg2                leave-days-out next-minute vol models (real-only / 50-50 / 25 % real), QLIKE
    ... evaluate [--record] verdicts, research/reports/btc_obgen_2026-11-13.{json,md}

Design (fixed before evaluation; diary 'OBG design fixed'): see METHOD.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from ai_trader.sim import ob_gen as og

TAG = "2026-11-13"
CUT = "2026-07-01"
ROOT = og.ROOT
OUT = og.OUT
GEN = OUT / "gen"
REPORT = ROOT / "research" / "reports" / f"btc_obgen_{TAG}"
FACTS = ("spread", "depth", "depletion", "size", "interarrival", "impact", "absr_acf", "flow_acf")

METHOD = {
    "split": ("day-level and fixed before any evaluation (data/research/obgen/split.json): calibration days = the ob200 event-level "
              "days og.CAL_DAYS (2026-04-15, 2026-09-13, 2026-10-28; none is a micro_days day) for the queue-reactive tables; "
              "held-out days = every verified micro_days day with BTC trades_1s / book_1s / orders / laws (31 days); per held-out "
              "day the generator is calibrated on its FIT half (00-12 UTC: queue and depth scales, arrival law, empirical order-size "
              "quantiles, sign model; theta (reprice threshold), lam (flow-bridge impact) and deep_corr (deep-level size) by 5 "
              "rounds of indirect inference on 06-08 UTC against the share of seconds with a mid change, the impact slope and "
              "the median depth) and evaluated on its TEST half (12-24 UTC); segment = "
              "llm_in_sample if the day < 2026-07-01 else clean_holdout; pooled = all"),
    "conditioning": ("H-OBG1 paths = the held-out day's real TEST-half 1-minute bars (book mid at each minute end, realized variance "
                     "of 1 s mid returns inside the minute, taker buy / sell notional and order counts): everything below one "
                     "minute (book, queues, order timing and signs, 1 s returns) is generated"),
    "engine": ("H-OBG1 days: generator events -> atx match-sim (crates/at-match, F4) -> trades.jsonl / book.jsonl -> sim.book "
               "(bin 1 s) and micro_days.bars_1s / merge_orders; twins (H-OBG2): the generator's own price-time book, whose prints "
               "and 100 ms book states equal the engine's exactly (tests/test_ob_gen.py, pilots), to fit the runtime budget"),
    "facts": {
        "spread": "share of 1 s book states with spread > 1.5 ticks",
        "depth": "two components, both must be inside: median log10 depth within 25 bp (USD, both sides, the 200-level view, "
                 "as the public feed) and median touch share = (bid1_size + ask1_size) mid / depth_25",
        "depletion": "queue depletion rate at the touch: mean over consecutive 1 s states with an unchanged best price of "
                     "max(q_t - q_t+1, 0) / q_t, both sides pooled",
        "size": "two components: median log10 taker-order notional and log10(p99 / p50) (orders = same-ms same-side print runs)",
        "interarrival": "two components: share of taker-order inter-arrival times <= 10 ms and log10 median inter-arrival (ms)",
        "impact": "OLS slope of the 1 s mid log return (bp) on 1 s signed taker notional (USD million)",
        "absr_acf": "mean autocorrelation of |1 s mid log return| over lags 1-60 s",
        "flow_acf": "mean autocorrelation of 1 s signed taker notional over lags 1-60 s",
    },
    "envelope": ("per segment and fact: [5%, 95%] empirical quantiles of the real held-out days' TEST-half values (the "
                 "day-to-day envelope); the generated fact = median over the same days of the generated TEST-half values; "
                 "a fact is inside when every component is inside; H-OBG1 supports iff >= 6 of 8 facts are inside, else "
                 "contradicts (segments with < 5 days: inconclusive). Reported, not in the criterion: per-day share of "
                 "generated values inside the envelope, Spearman of real vs generated per-day values"),
    "obg2": ("BTC next-minute vol (ml.micro channels, decision every 30 s train / 10 s test, target r1m^2, QLIKE loss, forecast "
             "F = 60 b_s exp(f + c) with c fitted on the validation day), models ml.micro make_linear and make_cnn trained with "
             "research.ml_micro_study.train (same optimiser and early stopping). Leave-days-out: the 31 held-out days sorted, "
             "day i -> fold i mod 3; train on the other folds, validation = last training day in date order (real), test = "
             "the fold's days (real only). Variants: R100 = all real training days; M50 = R100 + the generated twins of the "
             "same days (50% real / 50% generated windows); R25 = a fixed 25% of the training days (seed 2026, at least 2 "
             "days); M25 = R25 + the twins of those days. Twin of day d = a 24 h generated day driven by a 30-minute block "
             "bootstrap of day d's own 1-minute bars (calibrated on d's FIT half), so no test-day information reaches "
             "training. Per test day QLIKE difference B - A = r^2 (1/F_B - 1/F_A) + log(F_B / F_A) over windows; ratio = 1 + "
             "mean difference / mean Patton QLIKE of A (windows with r^2 > 0); 95% CI by a bootstrap over test days (B = 2000). "
             "Part 1: M50 vs R100 ratio CI upper < 1.01; part 2: M25 vs R25 ratio CI upper < 1. Primary model = CNN (lin "
             "reported). supports iff both parts; contradicts iff part 1 CI lower > 1.01 or part 2 point ratio >= 1; else "
             "inconclusive"),
    "recording": "one row per segment (llm_in_sample, clean_holdout, pooled) per hypothesis; skip segments already recorded",
}


def seg_of(day: str) -> str:
    return "llm_in_sample" if day < CUT else "clean_holdout"


def write_json(p: Path, obj: Any) -> None:
    og.write_json(p, obj)


def read_json(p: Path, default=None):
    return json.loads(p.read_text(encoding="utf-8")) if p.is_file() else default


# ----------------------------------------------------------------------------------------------------------------------
# split
# ----------------------------------------------------------------------------------------------------------------------


def heldout_days() -> list[str]:
    from ai_trader.research import micro_days as md

    out = []
    for d in md.processed_days():
        r = og.MICRO / d
        if all((r / s / f"{og.SYMBOL}.parquet").is_file() for s in ("trades_1s", "book_1s", "orders")) and (r / "laws" / f"{og.SYMBOL}.json").is_file():
            out.append(d)
    return sorted(out)


def split() -> dict:
    p = OUT / "split.json"
    s = read_json(p)
    if s is not None:
        return s
    days = heldout_days()
    s = {"frozen_at": datetime.now(UTC).isoformat(timespec="seconds"), "calibration_days": list(og.CAL_DAYS), "heldout_days": days,
         "segments": {d: seg_of(d) for d in days}, "folds": {d: i % 3 for i, d in enumerate(days)}, "method": METHOD}
    write_json(p, s)
    return s


# ----------------------------------------------------------------------------------------------------------------------
# facts
# ----------------------------------------------------------------------------------------------------------------------


def _acf_mean(x: np.ndarray, lags: int = 60) -> float:
    x = np.asarray(x, float)
    x = x - x.mean()
    v = float((x * x).mean())
    if v <= 0:
        return float("nan")
    n = len(x)
    f = np.fft.rfft(x, 2 * n)
    ac = np.fft.irfft(f * np.conj(f))[: lags + 1] / n
    return float(np.mean(ac[1:] / v))


def window_frames(day_root: Path, lo: int, hi: int) -> dict[str, Any]:
    tr = pl.read_parquet(day_root / "trades_1s" / f"{og.SYMBOL}.parquet").filter((pl.col("ts") >= lo) & (pl.col("ts") < hi))
    bk = pl.read_parquet(day_root / "book_1s" / f"{og.SYMBOL}.parquet")
    ms = (bk["ts"].dt.epoch("ms") if bk["ts"].dtype != pl.Int64 else bk["ts"]).cast(pl.Int64)
    bk = bk.with_columns(ms.alias("_ms")).filter((pl.col("_ms") > lo) & (pl.col("_ms") <= hi)).sort("_ms")
    od = pl.read_parquet(day_root / "orders" / f"{og.SYMBOL}.parquet").filter((pl.col("ms") >= lo) & (pl.col("ms") < hi))
    return {"trades": tr, "book": bk, "orders": od, "lo": lo, "hi": hi}


def facts(fr: dict[str, Any]) -> dict[str, Any]:
    """The eight H-OBG1 facts (METHOD['facts']) of one window."""
    tr, bk, od, lo, hi = fr["trades"], fr["book"], fr["orders"], fr["lo"], fr["hi"]
    n_sec = (hi - lo) // 1000
    out: dict[str, Any] = {}
    b1, a1 = bk["bid1"].to_numpy(), bk["ask1"].to_numpy()
    out["spread"] = [float(np.mean((a1 - b1) / og.TICK > 1.5))]
    dep = (bk["depth_bid_25"] + bk["depth_ask_25"]).to_numpy()
    touch = ((bk["bid1_size"] + bk["ask1_size"]) * bk["mid"]).to_numpy()
    out["depth"] = [float(np.nanmedian(np.log10(dep))), float(np.nanmedian(touch / dep))]
    vals = []
    for p, q in ((b1, bk["bid1_size"].to_numpy()), (a1, bk["ask1_size"].to_numpy())):
        same = (p[1:] == p[:-1]) & (q[:-1] > 0)
        vals.append(np.maximum(q[:-1] - q[1:], 0)[same] / q[:-1][same])
    out["depletion"] = [float(np.mean(np.concatenate(vals)))]
    v = od["notional"].to_numpy()
    out["size"] = [float(np.log10(np.median(v))), float(np.log10(np.quantile(v, 0.99) / np.median(v)))]
    dt = np.diff(np.sort(od["ms"].to_numpy()))
    out["interarrival"] = [float(np.mean(dt <= 10)), float(np.log10(max(np.median(dt), 0.5)))]
    # 1 s grid: mid at the end of second s (book ts = lo + (s + 1) 1000), signed notional of second s
    mid = np.full(n_sec, np.nan)
    k = ((bk["_ms"].to_numpy() - lo) // 1000 - 1).astype(int)
    okk = (k >= 0) & (k < n_sec)
    mid[k[okk]] = bk["mid"].to_numpy()[okk]
    mid = pl.Series(mid).fill_nan(None).forward_fill().backward_fill().to_numpy()
    r = np.diff(np.log(mid), prepend=np.log(mid[0])) * 1e4
    fl = np.zeros(n_sec)
    s = ((tr["ts"].to_numpy() - lo) // 1000).astype(int)
    oks = (s >= 0) & (s < n_sec)
    fl[s[oks]] = (tr["buy_notional"] - tr["sell_notional"]).to_numpy()[oks] / 1e6
    x, y = fl[1:], r[1:]
    xc = x - x.mean()
    out["impact"] = [float((xc * (y - y.mean())).sum() / max((xc * xc).sum(), 1e-12))]
    out["absr_acf"] = [_acf_mean(np.abs(r[1:]))]
    out["flow_acf"] = [_acf_mean(fl)]
    out["_aux"] = {"p_move_1s": float(np.mean(np.diff(mid) != 0)), "n_orders": int(od.height), "rv_bp2": float((r[1:] ** 2).sum())}
    return out


def calib_targets(day_root: Path, lo: int, hi: int) -> dict[str, float]:
    f = facts(window_frames(day_root, lo, hi))
    return {"p_move_1s": f["_aux"]["p_move_1s"], "impact": f["impact"][0]}


def real_facts(days: list[str] | None = None, refresh: bool = False) -> dict[str, dict]:
    """TEST-half (12-24 UTC) facts of the real held-out days (cached in data/research/obgen/facts_real.json)."""
    p = OUT / "facts_real.json"
    cache = {} if refresh else read_json(p, {})
    days = days or split()["heldout_days"]
    for d in days:
        if d in cache:
            continue
        d0 = og.day0_ms(d)
        cache[d] = facts(window_frames(og.MICRO / d, d0 + og.HALF_MS, d0 + og.DAY_MS))
        write_json(p, cache)
    return cache


# ----------------------------------------------------------------------------------------------------------------------
# per-day calibration: FIT-half laws + indirect inference of theta (move frequency), phi (impact), deep_corr (depth)
# ----------------------------------------------------------------------------------------------------------------------

II_WINDOW_H = (6, 8)        # FIT-half window (UTC hours) used for indirect inference
II_ITERS = 5


def _sim_facts(tables, cal, date: str, root: Path, lo: int, hi: int, seed: int) -> dict:
    p = og.real_path(date, lo, hi, root=root)
    out = og.Generator(tables, cal, seed=seed).run(p, warmup_s=120)
    tmp = og.SCRATCH / f"ii_{date}"
    og.aggregate(tmp, og.prints_dict(out.trades), out.book_rows, og.day0_ms(date))
    f = facts(window_frames(tmp, lo, hi))
    import shutil

    shutil.rmtree(tmp, ignore_errors=True)
    return f


def fit_day(date: str, tables, root: Path = og.MICRO, log=print) -> og.DayCal:
    """DayCal of a day from its FIT half, then II_ITERS rounds of multiplicative indirect-inference updates on the FIT
    window II_WINDOW_H: theta_scale <- theta (p_gen / p_real)^0.5 for the share of seconds with a mid change,
    lam <- lam (impact_real / impact_gen)^1.2 (flow-bridge impact), deep_corr <- deep_corr x 10^(depth_real - depth_gen).
    Targets come from the FIT half only; the TEST half is never touched here."""
    cal = og.day_calibration(date, tables, root)
    d0 = og.day0_ms(date)
    lo, hi = d0 + II_WINDOW_H[0] * 3_600_000, d0 + II_WINDOW_H[1] * 3_600_000
    real = facts(window_frames(root / date, lo, hi))
    pr, ir, dr = real["_aux"]["p_move_1s"], real["impact"][0], real["depth"][0]
    hist = []
    for it in range(II_ITERS):
        g = _sim_facts(tables, cal, date, root, lo, hi, seed=100 + it)
        pg, ig, dg = g["_aux"]["p_move_1s"], g["impact"][0], g["depth"][0]
        hist.append({"theta": cal.theta_scale, "lam": cal.lam, "deep_corr": cal.deep_corr, "p_move": pg, "impact": ig, "depth": dg})
        log(f"{date} ii {it}: theta {cal.theta_scale:.3f} lam {cal.lam:.3g} deep {cal.deep_corr:.3f} | p_move {pg:.3f}/{pr:.3f} "
            f"impact {ig:.3f}/{ir:.3f} depth {dg:.3f}/{dr:.3f}")
        if it == II_ITERS - 1:
            break
        cal.theta_scale = float(np.clip(cal.theta_scale * (max(pg, 1e-3) / max(pr, 1e-3)) ** 0.5, 0.1, 10.0))
        ratio = ir / ig if ig > 1e-6 and ir > 0 else 2.0
        cal.lam = float(np.clip(cal.lam * ratio ** 1.2, 1e-10, 1e-5))
        cal.deep_corr = float(np.clip(cal.deep_corr * 10 ** (dr - dg), 0.1, 10.0))
    cal.notes.update({"ii_targets": {"p_move_1s": pr, "impact": ir, "depth": dr}, "ii_history": hist, "ii_window_h": list(II_WINDOW_H)})
    return cal


# ----------------------------------------------------------------------------------------------------------------------
# pipeline steps (checkpointed per day through the manifest)
# ----------------------------------------------------------------------------------------------------------------------


def tables():
    return og.load_tables(og.CAL_DAYS)


def daycal(days: list[str], T=None, log=print) -> dict[str, og.DayCal]:
    man = og.Manifest()
    T = T or tables()
    out = {}
    for d in days:
        p = OUT / "daycal" / f"{d}.json"
        if man.done("daycal", d) and p.is_file():
            out[d] = og.DayCal(**read_json(p))
            continue
        t0 = time.time()
        cal = fit_day(d, T, log=log)
        write_json(p, cal.__dict__)
        man.put("daycal", d, {"status": "done", "secs": time.time() - t0, "theta": cal.theta_scale,
                              "lam": cal.lam, "deep_corr": cal.deep_corr})
        out[d] = cal
    return out


def _seed(d: str, salt: int) -> int:
    return int(d.replace("-", "")) * 10 + salt


def obg1(days: list[str], T=None, log=print) -> dict[str, dict]:
    """Generate the TEST half of each held-out day driven by its real 1-minute path; facts cached in facts_gen.json."""
    T = T or tables()
    p_f = OUT / "facts_gen.json"
    for d in days:
        cals = daycal([d], T, log)
        cache = read_json(p_f, {})
        d0 = og.day0_ms(d)
        lo, hi = d0 + og.HALF_MS, d0 + og.DAY_MS
        od = GEN / "obg1" / d
        og.generate_day(f"obg1:{d}", T, cals[d], og.real_path(d, lo, hi), od, d0, seed=_seed(d, 1), log=log)
        if d not in cache:
            cache[d] = facts(window_frames(od, lo, hi))
            write_json(p_f, cache)
    return cache


def twins(days: list[str], T=None, log=print) -> None:
    """24 h generated twin of each day: block bootstrap (30 min) of the day's own 1-minute bars, calibrated on its FIT half."""
    T = T or tables()
    for d in days:
        cals = daycal([d], T, log)
        d0 = og.day0_ms(d)
        rp = og.real_path(d, d0, d0 + og.DAY_MS)
        bp = og.bootstrap_path(rp, np.random.default_rng(_seed(d, 2)), block_bars=30)
        og.generate_day(f"twin:{d}", T, cals[d], bp, GEN / "twin" / d, d0, seed=_seed(d, 3), log=log)


# ----------------------------------------------------------------------------------------------------------------------
# H-OBG2: next-minute vol models on real / real + generated windows
# ----------------------------------------------------------------------------------------------------------------------

VARIANTS = ("R100", "M50", "R25", "M25")
OBG2_DIR = OUT / "obg2"


def _store(keys: list[tuple[str, str]]):
    """ml.micro Store over (source, day) keys: source 'real' = data/research/micro_days, 'twin' = generated twins."""
    from ai_trader.ml import micro as mc

    parts, offsets, meta, aux = [], {}, {}, {}
    pos = 0
    for src, day in keys:
        root = og.MICRO if src == "real" else GEN / "twin"
        r = mc.load_symbol_day(root, day, og.SYMBOL)
        if r is None or r["traded"] < mc.MIN_TRADED_SECONDS:
            continue
        k = (src, day)
        parts.append(r["X"].astype(np.float16))
        offsets[k] = pos
        pos += r["X"].shape[0]
        meta[k] = {"day": day, "symbol": og.SYMBOL, "book": True, "traded": r["traded"]}
        aux[k] = {"logp": r["logp"].astype(np.float64), "b": r["b"].astype(np.float32)}
    return mc.Store(np.concatenate(parts), offsets, meta), aux


def r25_days(fit_days: list[str], fold: int) -> list[str]:
    rng = np.random.default_rng(2026 + fold)
    k = max(2, int(math.floor(len(fit_days) / 4)))
    return sorted(rng.choice(fit_days, size=k, replace=False).tolist())


def obg2_fit(log=print) -> None:
    """Leave-days-out folds (METHOD['obg2']); per fold and variant: lin + cnn, predictions on the real test days.
    Checkpoint: data/research/obgen/obg2/fold_<k>.parquet."""
    import torch

    from ai_trader.research import ml_micro_study as mm

    torch.set_num_threads(2)
    s = split()
    days = s["heldout_days"]
    folds = {k: [d for d in days if s["folds"][d] == k] for k in range(3)}
    OBG2_DIR.mkdir(parents=True, exist_ok=True)
    for fk, test_days in folds.items():
        ck = OBG2_DIR / f"fold_{fk}.parquet"
        if ck.is_file():
            log(f"fold {fk}: checkpoint present")
            continue
        train_days = [d for d in days if d not in test_days]
        val_day = train_days[-1]
        fit_days = [d for d in train_days if d != val_day]
        sub = r25_days(fit_days, fk)
        sets = {"R100": [("real", d) for d in fit_days], "M50": [("real", d) for d in fit_days] + [("twin", d) for d in fit_days],
                "R25": [("real", d) for d in sub], "M25": [("real", d) for d in sub] + [("twin", d) for d in sub]}
        keys = sorted({k for v in sets.values() for k in v} | {("real", val_day)} | {("real", d) for d in test_days})
        store, aux = _store(keys)
        G = mm.Gather(store)
        va = mm.window_set(store, aux, [("real", val_day)], mm.TRAIN_STEP, offset=15)
        te_keys = [("real", d) for d in test_days if ("real", d) in store.offsets]
        te = mm.window_set(store, aux, te_keys, mm.TEST_STEP)
        cols = {"day": [te_keys[i][1] for i in te["kid"]], "sec": te["sec"], "r1": te["r1"], "base": te["base"]}
        info = {"fold": fk, "test_days": test_days, "val_day": val_day, "fit_days": fit_days, "r25_days": sub}
        for var in VARIANTS:
            ks = [k for k in sets[var] if k in store.offsets]
            tr = mm.window_set(store, aux, ks, mm.TRAIN_STEP)
            hmu, hsd = tr["H"].mean(0), tr["H"].std(0)
            hsd = np.where(hsd > 0, hsd, 1.0)
            info[f"n_train_{var}"] = int(len(tr["r1"]))
            info[f"n_gen_{var}"] = int(sum(1 for k in ks if k[0] == "twin"))
            for kind in ("lin", "cnn"):
                t1 = time.time()
                m, ti = mm.train(kind, "vol", 0, G, tr, va, hmu, hsd)
                fv = mm.predict([m], G, va, hmu, hsd)
                c = float(np.log(np.mean(va["r1"] ** 2 / (va["base"] * np.exp(fv)))))
                f = mm.predict([m], G, te, hmu, hsd)
                cols[f"F_{var}_{kind}"] = te["base"] * np.exp(f + c)
                info[f"{var}_{kind}"] = {**ti, "bias": c, "secs": round(time.time() - t1, 1)}
                log(json.dumps({"fold": fk, "variant": var, "kind": kind, "n_train": len(tr["r1"]), **info[f"{var}_{kind}"]}))
            del tr
        pl.DataFrame(cols).write_parquet(ck)
        write_json(OBG2_DIR / f"fold_{fk}.json", info)
        del store, aux, G, va, te


def _qlike_patton(r2: np.ndarray, F: np.ndarray) -> np.ndarray:
    with np.errstate(all="ignore"):
        q = np.where(r2 > 0, r2 / F - np.log(r2 / F) - 1, np.nan)
    return q


def ratio_by_days(pr: pl.DataFrame, a: str, b: str, days: list[str], B: int = 2000, seed: int = 0) -> dict:
    """QLIKE ratio of forecast b vs a over the given test days: 1 + mean diff / mean Patton QLIKE of a; day bootstrap."""
    sub = pr.filter(pl.col("day").is_in(days))
    if sub.height == 0:
        return {"n_days": 0}
    r2 = sub["r1"].to_numpy() ** 2
    Fa, Fb = sub[a].to_numpy(), sub[b].to_numpy()
    d = r2 * (1 / Fb - 1 / Fa) + np.log(Fb / Fa)
    qa = _qlike_patton(r2, Fa)
    dd = sub["day"].to_numpy()
    ud = np.unique(dd)
    idx = {u: np.flatnonzero(dd == u) for u in ud}
    sd = np.array([d[idx[u]].sum() for u in ud])
    nd = np.array([len(idx[u]) for u in ud])
    sq = np.array([np.nansum(qa[idx[u]]) for u in ud])
    nq = np.array([np.isfinite(qa[idx[u]]).sum() for u in ud])
    point = 1 + (sd.sum() / nd.sum()) / (sq.sum() / nq.sum())
    rng = np.random.default_rng(seed)
    bs = []
    for _ in range(B):
        j = rng.integers(0, len(ud), len(ud))
        bs.append(1 + (sd[j].sum() / nd[j].sum()) / (sq[j].sum() / nq[j].sum()))
    lo, hi = np.quantile(bs, [0.025, 0.975])
    per_day = {str(u): float(1 + (sd[i] / nd[i]) / (sq[i] / nq[i])) for i, u in enumerate(ud)}
    return {"n_days": int(len(ud)), "n_windows": int(sub.height), "ratio": float(point), "ci": [float(lo), float(hi)],
            "share_days_better": float(np.mean(np.array(list(per_day.values())) < 1)), "per_day": per_day}


# ----------------------------------------------------------------------------------------------------------------------
# evaluation, report, recording
# ----------------------------------------------------------------------------------------------------------------------

DATA = ("ob200 event-level BTC days 2026-04-15, 2026-09-13, 2026-10-28 (quote-saver.bycsi.com + public.bybit.com, raw deleted "
        "after aggregation; data/research/obgen/cal, real/); held-out days data/research/micro_days (31 days, BTC); generated "
        "aggregates data/research/obgen/gen/{obg1,twin}; src/ai_trader/sim/ob_gen.py, src/ai_trader/research/obgen_study.py")


def _spearman(a, b) -> float:
    from scipy.stats import spearmanr

    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    return float(spearmanr(a[ok], b[ok]).statistic) if ok.sum() >= 4 else float("nan")


def obg1_eval(seg_days: dict[str, list[str]]) -> dict:
    fr, fg = read_json(OUT / "facts_real.json", {}), read_json(OUT / "facts_gen.json", {})
    out = {}
    for sg, days in seg_days.items():
        days = [d for d in days if d in fr and d in fg]
        res = {"n_days": len(days), "facts": {}}
        if len(days) < 5:
            res["verdict"] = "inconclusive"
            out[sg] = res
            continue
        n_in = 0
        for f in FACTS:
            R = np.array([fr[d][f] for d in days], float)
            Gm = np.array([fg[d][f] for d in days], float)
            lo, hi = np.nanquantile(R, 0.05, axis=0), np.nanquantile(R, 0.95, axis=0)
            gmed = np.nanmedian(Gm, axis=0)
            inside = bool(np.all((gmed >= lo) & (gmed <= hi)))
            per_day = np.all((Gm >= lo) & (Gm <= hi), axis=1)
            n_in += inside
            res["facts"][f] = {"envelope_lo": lo.tolist(), "envelope_hi": hi.tolist(), "real_median": np.nanmedian(R, axis=0).tolist(),
                               "gen_median": gmed.tolist(), "inside": inside, "share_days_inside": float(per_day.mean()),
                               "spearman_real_gen": [_spearman(R[:, c], Gm[:, c]) for c in range(R.shape[1])]}
        res["n_inside"] = int(n_in)
        res["verdict"] = "supports" if n_in >= 6 else "contradicts"
        out[sg] = res
    return out


def obg2_eval(seg_days: dict[str, list[str]]) -> dict:
    parts = [pl.read_parquet(p) for p in sorted(OBG2_DIR.glob("fold_*.parquet"))]
    if len(parts) < 3:
        return {"status": "incomplete", "folds": len(parts)}
    pr = pl.concat(parts, how="vertical")
    out = {}
    for sg, days in seg_days.items():
        r = {}
        for kind in ("cnn", "lin"):
            p1 = ratio_by_days(pr, f"F_R100_{kind}", f"F_M50_{kind}", days)
            p2 = ratio_by_days(pr, f"F_R25_{kind}", f"F_M25_{kind}", days)
            ctx = {"R25_vs_R100": ratio_by_days(pr, f"F_R100_{kind}", f"F_R25_{kind}", days),
                   "M25_vs_R100": ratio_by_days(pr, f"F_R100_{kind}", f"F_M25_{kind}", days)}
            ok1 = p1.get("n_days", 0) > 0 and p1["ci"][1] < 1.01
            ok2 = p2.get("n_days", 0) > 0 and p2["ci"][1] < 1.0
            if p1.get("n_days", 0) < 3:
                v = "inconclusive"
            elif ok1 and ok2:
                v = "supports"
            elif p1["ci"][0] > 1.01 or p2["ratio"] >= 1.0:
                v = "contradicts"
            else:
                v = "inconclusive"
            r[kind] = {"part1_M50_vs_R100": p1, "part2_M25_vs_R25": p2, "context": ctx, "part1_ok": bool(ok1), "part2_ok": bool(ok2), "verdict": v}
        r["verdict"] = r["cnn"]["verdict"]
        out[sg] = r
    return out


def _fmt(x, nd=4):
    if isinstance(x, (list, tuple)):
        return "[" + ", ".join(_fmt(v, nd) for v in x) + "]"
    if x is None:
        return "n/a"
    return f"{x:.{nd}g}" if isinstance(x, float) else str(x)


def rule_candidates(o1: dict, o2: dict) -> list[dict]:
    pooled1, pooled2 = o1.get("pooled", {}), o2.get("pooled", {})
    rules = []
    n_in = pooled1.get("n_inside")
    miss = [f for f, v in pooled1.get("facts", {}).items() if not v["inside"]]
    rules.append({"id": "R-OBG-1", "rule": (f"Generated BTC books (ob_gen v1 through at-match) reproduce {n_in} of 8 held-out book and trade facts "
                                             f"(pooled); facts outside the day-to-day envelope: {', '.join(miss) if miss else 'none'}. "
                                             "Use generated books only for tests that depend on the reproduced facts."),
                  "evidence": "H-OBG1", "status": pooled1.get("verdict", "n/a")})
    if pooled2 and "cnn" in pooled2:
        c = pooled2["cnn"]
        rules.append({"id": "R-OBG-2", "rule": (f"For BTC next-minute volatility, adding generated twin days to the real training days gives a held-out QLIKE "
                                                 f"ratio of {c['part1_M50_vs_R100']['ratio']:.4f} (CNN, pooled; 95% CI {_fmt(c['part1_M50_vs_R100']['ci'])}); "
                                                 f"with a quarter of the real days the ratio is {c['part2_M25_vs_R25']['ratio']:.4f} "
                                                 f"(CI {_fmt(c['part2_M25_vs_R25']['ci'])}). Treat generated days as "
                                                 + ("a usable augmentation." if c["verdict"] == "supports" else "no substitute for real days.")),
                      "evidence": "H-OBG2", "status": c["verdict"]})
    rules.append({"id": "R-OBG-3", "rule": ("Model BTC book depth beyond the touch as a slow stationary profile (occupancy and size per tick distance) "
                                             "that follows the price, and the touch as a queue-reactive process with flicker: a cancel right after "
                                             "an add is likelier and usually removes that order."),
                  "evidence": "ob200 calibration days (descriptive)", "status": "descriptive"})
    return rules


def evaluate(record_rows: bool = False) -> dict:
    s = split()
    days = s["heldout_days"]
    seg_days = {"llm_in_sample": [d for d in days if seg_of(d) == "llm_in_sample"],
                "clean_holdout": [d for d in days if seg_of(d) == "clean_holdout"], "pooled": days}
    o1 = obg1_eval(seg_days)
    o2 = obg2_eval(seg_days)
    man = og.Manifest()
    gen = man.data.get("generated", {})
    T = tables()
    eng = [v for v in gen.values() if "engine_vs_internal" in v]
    out = {"tag": TAG, "method": METHOD, "split": {k: s[k] for k in ("calibration_days", "heldout_days", "folds")},
           "H-OBG1": o1, "H-OBG2": o2, "rules": rule_candidates(o1, o2),
           "calibration": {"days": list(og.CAL_DAYS), "aes_pooled_btc": T.aes, "touch_med_btc": T.touch_med_btc, "depth_med_btc": T.depth_med_btc,
                           "touch_mult": T.touch_mult.tolist(), "counts": T.counts,
                           "p_cancel_touch_by_q": T.p_can[0, :, 1, 0].tolist(), "p_add_touch_by_q": T.p_add[0, :, 1, 0].tolist(),
                           "p_market_touch_by_q": T.mk_rate[:, 1].tolist(), "qbins_aes": [q if math.isfinite(q) else None for q in og.QBINS]},
           "daycal": {d: {"targets": (read_json(OUT / "daycal" / f"{d}.json", {}) or {}).get("notes", {}).get("ii_targets"),
                          **{k: (read_json(OUT / "daycal" / f"{d}.json", {}) or {}).get(k) for k in ("theta_scale", "lam", "deep_corr", "s_touch", "s_deep")}}
                      for d in days},
           "generation": {"n_obg1": sum(1 for k in gen if k.startswith("obg1:")), "n_twin": sum(1 for k in gen if k.startswith("twin:")),
                          "engines": sorted({str(v.get("engine")) for v in gen.values()}),
                          "engine_rejects": int(sum(v.get("engine_rejects", 0) or 0 for v in gen.values())),
                          "engine_days": len(eng),
                          "engine_prints_equal_internal": all(v["engine_vs_internal"]["n_prints_engine"] == v["engine_vs_internal"]["n_prints_internal"]
                                                              and abs(v["engine_vs_internal"]["qty_engine"] - v["engine_vs_internal"]["qty_internal"]) < 1e-6
                                                              for v in eng),
                          "events_total": int(sum(v.get("n_events", 0) for v in gen.values())),
                          "gen_seconds_median": float(np.median([v.get("total_s", np.nan) for v in gen.values()])) if gen else None},
           "storage_gb": og.storage_gb()}
    write_json(Path(str(REPORT) + ".json"), out)
    Path(str(REPORT) + ".md").write_text(render_md(out), encoding="utf-8")
    if record_rows:
        record_all(out)
    return out


def render_md(o: dict) -> str:
    L = [f"# BTC order-book generator (H-OBG1, H-OBG2), {TAG}", "",
         "Agent F5. Queue-reactive order-flow generator `src/ai_trader/sim/ob_gen.py`, executed by the Rust matching engine (`atx match-sim`, "
         "crates/at-match), evaluated by `src/ai_trader/research/obgen_study.py`. Research only; generated data never reach the LLM layer.", "",
         "## Design (fixed before evaluation)", ""]
    for k, v in o["method"].items():
        if isinstance(v, dict):
            L.append(f"- **{k}**:")
            L += [f"  - {a}: {b}" for a, b in v.items()]
        else:
            L.append(f"- **{k}**: {v}")
    c = o["calibration"]
    L += ["", "## Calibration (ob200 event-level days)", "",
          f"- days {', '.join(c['days'])}; pooled AES {c['aes_pooled_btc']:.3f} BTC; median touch {c['touch_med_btc']:.2f} BTC; "
          f"median 200-level depth {c['depth_med_btc']:.0f} BTC (both sides)",
          f"- queue bins (q / AES): {c['qbins_aes']}",
          f"- touch cancel probability per 100 ms by queue bin (mid vol state, 1-tick spread): {_fmt(c['p_cancel_touch_by_q'], 3)}",
          f"- touch add probability: {_fmt(c['p_add_touch_by_q'], 3)}",
          f"- P(prints at the touch in a 100 ms interval) by queue bin (lambda_M(q), descriptive): {_fmt(c['p_market_touch_by_q'], 3)}",
          f"- touch event memory multipliers, rows previous none / add / cancel, columns (add, cancel): {_fmt(c['touch_mult'], 3)}",
          f"- generation: {o['generation']}", ""]
    L += ["## H-OBG1: facts vs held-out real days", ""]
    for sg, r in o["H-OBG1"].items():
        L.append(f"### {sg}: {r.get('n_inside', 'n/a')} of 8 inside -> **{r['verdict']}** ({r['n_days']} days)")
        L.append("")
        L.append("| fact | real median | envelope 5-95% | generated median | inside | share of days inside | Spearman real vs gen |")
        L.append("|---|---|---|---|---|---|---|")
        for f, v in r.get("facts", {}).items():
            L.append(f"| {f} | {_fmt(v['real_median'])} | {_fmt(v['envelope_lo'])} - {_fmt(v['envelope_hi'])} | {_fmt(v['gen_median'])} | "
                     f"{'yes' if v['inside'] else 'no'} | {v['share_days_inside']:.2f} | {_fmt(v['spearman_real_gen'], 2)} |")
        L.append("")
    L += ["## H-OBG2: half real / half generated training (next-minute vol, QLIKE ratio, day bootstrap)", ""]
    o2 = o["H-OBG2"]
    if "status" in o2:
        L.append(f"- {o2}")
    else:
        L.append("| segment | model | M50 / R100 (95% CI) | M25 / R25 (95% CI) | R25 / R100 | verdict |")
        L.append("|---|---|---|---|---|---|")
        for sg, r in o2.items():
            for kind in ("cnn", "lin"):
                k = r[kind]
                L.append(f"| {sg} | {kind} | {k['part1_M50_vs_R100']['ratio']:.4f} {_fmt(k['part1_M50_vs_R100']['ci'])} | "
                         f"{k['part2_M25_vs_R25']['ratio']:.4f} {_fmt(k['part2_M25_vs_R25']['ci'])} | "
                         f"{k['context']['R25_vs_R100'].get('ratio', float('nan')):.4f} | {k['verdict']} |")
        L.append("")
        L.append("Primary model: CNN (the segment verdict); lin reported. Ratio < 1 = the generated-augmented model is better.")
    L += ["", "## What the generator does not reproduce, and caveats", "",
          "- Deep levels (>= 30 ticks from the touch) are re-drawn from a stationary profile; their queue dynamics (estimated, reported) are not simulated.",
          "- On H-OBG1 days the 1-minute path (mid, realized variance, taker counts and notional per side) is real: only sub-minute structure is generated.",
          "- Per-day indirect inference on the FIT half targets the share of seconds with a mid change, the impact slope and the median depth; "
          "the TEST half is never used for calibration, but these three facts are calibrated quantities, not emergent ones.",
          "- Public ob200 shows ~200 levels (~4 bp of BTC): depth facts are on that truncated view, for real and generated books alike.",
          "- Bybit internals (risk engine, liquidations, hidden priority rules, latency) are outside at-match; market orders use a 1% protection band (assumption).",
          "", "## Rule candidates", "", "```yaml"]
    for r in o["rules"]:
        L.append(f"- id: {r['id']}")
        L.append(f"  rule: \"{r['rule']}\"")
        L.append(f"  evidence: \"{r['evidence']}\"")
        L.append(f"  status: \"{r['status']}\"")
    L += ["```", "", f"Storage at evaluation (GB): {o['storage_gb']}", ""]
    return "\n".join(L)


def record_all(out: dict) -> None:
    from ai_trader.research.hypotheses import record, results

    for h in ("H-OBG1", "H-OBG2"):
        done = {r["segment"] for r in results(h)}
        res = out[h]
        if "status" in res:
            print(f"{h}: {res['status']}, not recorded")
            continue
        for sg in ("llm_in_sample", "clean_holdout", "pooled"):
            if sg in done:
                print(f"skip {h} {sg}")
                continue
            r = res[sg]
            if h == "H-OBG1":
                m = {"n_days": r["n_days"], "n_inside": r.get("n_inside"),
                     "facts": {f: {k: v[k] for k in ("inside", "real_median", "gen_median", "envelope_lo", "envelope_hi", "share_days_inside")}
                               for f, v in r.get("facts", {}).items()}}
                note = (f"{r.get('n_inside')} of 8 facts inside the real day-to-day [5,95]% envelope (generated median over the same days); "
                        "TEST half 12-24 UTC, per-day calibration on the FIT half, real 1-min path conditioning, at-match engine")
            else:
                c, li = r["cnn"], r["lin"]
                m = {"cnn": {"part1_ok": c["part1_ok"], "part2_ok": c["part2_ok"],
                             "part1": {k: c["part1_M50_vs_R100"][k] for k in ("ratio", "ci", "n_days")},
                             "part2": {k: c["part2_M25_vs_R25"][k] for k in ("ratio", "ci", "n_days")}},
                     "lin": {"part1": {k: li["part1_M50_vs_R100"][k] for k in ("ratio", "ci")},
                             "part2": {k: li["part2_M25_vs_R25"][k] for k in ("ratio", "ci")}, "verdict": li["verdict"]}}
                note = "leave-days-out (3 folds), CNN primary, QLIKE ratio with a day bootstrap; generated = 24 h twins of the training days"
            record(h, sg, r["verdict"], m, note=note, data=DATA)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("split", "daycal", "obg1", "twins", "obg2", "evaluate", "real-facts"))
    ap.add_argument("days", nargs="*")
    ap.add_argument("--record", action="store_true")
    ap.add_argument("--no-engine", action="store_true")
    ap.add_argument("--reverse", action="store_true")
    a = ap.parse_args()
    if a.no_engine:
        og._ATX["bin"] = None
    days = a.days or split()["heldout_days"]
    if a.cmd == "split":
        print(json.dumps(split(), indent=1)[:2000])
    elif a.cmd == "real-facts":
        real_facts(days)
    elif a.cmd == "daycal":
        daycal(days)
    elif a.cmd == "obg1":
        obg1(days)
    elif a.cmd == "twins":
        twins(days[::-1] if a.reverse else days)
    elif a.cmd == "obg2":
        obg2_fit()
    else:
        o = evaluate(a.record)
        print(json.dumps({"H-OBG1": {k: (v.get("n_inside"), v["verdict"]) for k, v in o["H-OBG1"].items()},
                          "H-OBG2": ({k: v.get("verdict") for k, v in o["H-OBG2"].items()} if "status" not in o["H-OBG2"] else o["H-OBG2"])}, indent=1))


if __name__ == "__main__":
    main()
