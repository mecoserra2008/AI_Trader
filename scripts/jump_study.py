"""H-JD1-4: walk-forward tests of the jump-diffusion model (registered 2026-11-08 22:00 UTC).

    PYTHONIOENCODING=utf-8 uv run python scripts/jump_study.py [--record] [--symbols BTCUSDT ETHUSDT SOLUSDT]

Monthly refits from 2023-01 on a trailing 2-year window of 1h bars (lake). Out of sample each month:
  JD1 per-bar log score: jump-diffusion mixture vs Student-t vs Gaussian (same bipower sigma as scale)
  JD2 daily (00:00 UTC) P(>= 1 detected jump in the next 24 h): Hawkes vs constant rate, Brier skill
  JD3 detected jumps (posterior > 0.9 and Lee-Mykland 1%, declustered 24 h): signed next 6 h / 24 h return by vol state
  JD4 brackets every 72 bars, long and short, stop 1.5 x daily bipower sigma, target 3R, 168-bar cap: Monte-Carlo
      P(target first) with jumps vs without, Brier vs outcomes from the realised highs / lows (stop first on ties)
"""

from __future__ import annotations

import argparse
import json
import math
import warnings
from datetime import UTC, date, datetime, timedelta

import numpy as np
import polars as pl
import statsmodels.api as sm
from scipy import stats

from ai_trader.core.config import load
from ai_trader.core.time import now
from ai_trader.features.jump_diffusion import (
    bipower_sigma,
    fit_hawkes,
    fit_mixture,
    hawkes_intensity,
    lee_mykland,
    mixture_logpdf,
    p_jump_within,
    posterior,
    simulate_bracket,
)
from ai_trader.research.frame import market_frame
from ai_trader.research.regimes import label
from ai_trader.storage.lake import Lake

warnings.filterwarnings("ignore")
CUTOFF = date(2026, 7, 1)
TRAIN_BARS = 2 * 365 * 24


def block_bss(y, p, base, block=14 * 24, reps=600, seed=0):
    y, p, base = map(np.asarray, (y, p, base))
    bs, br = np.mean((p - y) ** 2), np.mean((base - y) ** 2)
    rng = np.random.default_rng(seed)
    n = len(y)
    starts = np.arange(0, n, block)
    boot = []
    for _ in range(reps):
        idx = np.concatenate([np.arange(s, min(n, s + block)) for s in rng.choice(starts, size=len(starts))])
        boot.append(1 - np.mean((p[idx] - y[idx]) ** 2) / np.mean((base[idx] - y[idx]) ** 2))
    return float(1 - bs / br), float(np.percentile(boot, 5)), float(np.percentile(boot, 95))


def run_symbol(sym: str, t) -> dict:
    lake = Lake()
    f = label(market_frame(sym, "1h", t, lake=lake, funding=False, oi=False), 1.0)
    f = f.select("event_ts", "open", "high", "low", "close", "vol_state").sort("event_ts")
    c = f["close"].to_numpy()
    hi, lo = f["high"].to_numpy(), f["low"].to_numpy()
    ts = f["event_ts"].to_list()
    r = np.r_[np.nan, np.diff(np.log(c))]
    sig = bipower_sigma(r, 72)
    vs = f["vol_state"].to_list()
    n = len(r)
    out = {"jd1": [], "jd2": [], "jd3": [], "jd4": [], "fits": []}
    months = sorted({(x.year, x.month) for x in ts if x.date() >= date(2023, 1, 1)})
    for (yy, mm) in months:
        m0 = datetime(yy, mm, 1, tzinfo=UTC)
        m1 = datetime(yy + (mm == 12), mm % 12 + 1, 1, tzinfo=UTC)
        i0 = next((i for i, x in enumerate(ts) if x >= m0), None)
        i1 = next((i for i, x in enumerate(ts) if x >= m1), n)
        if i0 is None or i0 < 2000 or i1 - i0 < 48:
            continue
        tr = slice(max(80, i0 - TRAIN_BARS), i0)
        rt, st = r[tr], sig[tr]
        ok = np.isfinite(rt) & np.isfinite(st) & (st > 0)
        prm = fit_mixture(rt[ok], st[ok])
        L, thr = lee_mykland(r, sig)
        post_all = posterior(np.nan_to_num(r), np.where(np.isfinite(sig) & (sig > 0), sig, 1.0), prm)
        det = (post_all > 0.9) & (L > thr) & np.isfinite(sig)
        tr_idx = np.arange(tr.start, tr.stop)
        times = tr_idx[det[tr_idx]] - tr.start
        mu, a, b, lr = fit_hawkes(times.astype(float), float(len(tr_idx)))
        prm.hawkes_mu, prm.hawkes_alpha, prm.hawkes_beta, prm.hawkes_lr = mu, a, b, lr
        out["fits"].append({"month": f"{yy}-{mm:02d}", "p": prm.p, "p_up": prm.p_up, "mean_up": prm.mean_up, "mean_down": prm.mean_down,
                            "hawkes_alpha": a, "hawkes_beta": b, "lr": lr, "n_detected": int(len(times))})
        # JD1: log scores on the month
        z_tr = (rt[ok] / st[ok])
        df_t, _, sc_t = stats.t.fit(z_tr, floc=0)
        sc_n = float(np.std(z_tr))
        te = np.arange(i0, i1)
        okt = np.isfinite(r[te]) & np.isfinite(sig[te]) & (sig[te] > 0)
        te = te[okt]
        ljd = mixture_logpdf(r[te], sig[te], prm.p, prm.p_up, prm.eta_up, prm.eta_down)
        lt = stats.t.logpdf(r[te] / (sig[te] * sc_t), df_t) - np.log(sig[te] * sc_t)
        lg = stats.norm.logpdf(r[te] / (sig[te] * sc_n)) - np.log(sig[te] * sc_n)
        for k, i in enumerate(te):
            out["jd1"].append((ts[i], float(ljd[k] - lt[k]), float(ljd[k] - lg[k])))
        # Hawkes intensity over the whole history (causal), detected flags from this fit
        lam = hawkes_intensity(det, mu, a, b)
        const_rate = len(times) / len(tr_idx)
        # JD2: daily forecasts at 00:00 UTC within the month
        for i in te:
            if ts[i].hour == 0 and i + 24 < n:
                excess = lam[i] - mu
                p_h = p_jump_within(excess, mu, a, b, 24)
                p_c = 1 - (1 - const_rate) ** 24
                y = float(det[i:i + 24].any())
                out["jd2"].append((ts[i], y, p_h, p_c))
        # JD3: declustered detected jumps in the month
        last = -10**9
        for i in te:
            if det[i] and i - last > 24 and i + 24 < n:
                last = i
                sgn = np.sign(r[i])
                f6 = sgn * (math.log(c[i + 6] / c[i])) / (sig[i] * math.sqrt(6))
                f24 = sgn * (math.log(c[i + 24] / c[i])) / (sig[i] * math.sqrt(24))
                out["jd3"].append({"ts": ts[i], "vol_state": vs[i], "z": float(r[i] / sig[i]), "f6": float(f6), "f24": float(f24)})
        # JD4: brackets every 72 bars
        for i in te[::72]:
            if i + 168 >= n or not np.isfinite(sig[i]):
                continue
            sd = 1.5 * sig[i] * math.sqrt(24)
            excess = lam[i] - mu
            for side in (1, -1):
                entry = c[i]
                stop = entry * math.exp(-side * sd)
                tgt = entry * math.exp(side * 3 * sd)
                pj = simulate_bracket(entry, stop, tgt, sig[i], prm, 168, excess0=excess, n_paths=1200, seed=i % 9973 + (side > 0), use_jumps=True)["p_tp"]
                pg = simulate_bracket(entry, stop, tgt, sig[i], prm, 168, n_paths=1200, seed=i % 9973 + (side > 0), use_jumps=False)["p_tp"]
                y = 0.0
                for k in range(i + 1, i + 169):
                    hit_sl = lo[k] <= stop if side > 0 else hi[k] >= stop
                    hit_tp = hi[k] >= tgt if side > 0 else lo[k] <= tgt
                    if hit_sl:
                        break
                    if hit_tp:
                        y = 1.0
                        break
                out["jd4"].append((ts[i], side, y, pj, pg))
    return out


def seg_of(x) -> str:
    d = x.date() if hasattr(x, "date") else x
    return "llm_in_sample" if d < CUTOFF else "clean_holdout"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--record", action="store_true")
    ap.add_argument("--symbols", nargs="*", default=["BTCUSDT", "ETHUSDT", "SOLUSDT"])
    a = ap.parse_args()
    t = now()
    allr = {s: run_symbol(s, t) for s in a.symbols}
    res = {"fits_last": {s: v["fits"][-1] for s, v in allr.items() if v["fits"]}, "JD1": {}, "JD2": {}, "JD3": {}, "JD4": {}}
    for seg in ("llm_in_sample", "clean_holdout"):
        # JD1 pooled
        d_t = np.array([x[1] for s in allr.values() for x in s["jd1"] if seg_of(x[0]) == seg])
        d_g = np.array([x[2] for s in allr.values() for x in s["jd1"] if seg_of(x[0]) == seg])
        if len(d_t) > 100:
            mt = sm.OLS(d_t, np.ones(len(d_t))).fit(cov_type="HAC", cov_kwds={"maxlags": 24})
            mg = sm.OLS(d_g, np.ones(len(d_g))).fit(cov_type="HAC", cov_kwds={"maxlags": 24})
            res["JD1"][seg] = {"n": len(d_t), "mean_vs_t": float(d_t.mean()), "t_vs_t": float(mt.tvalues[0]), "mean_vs_gauss": float(d_g.mean()), "t_vs_gauss": float(mg.tvalues[0]),
                               "per_symbol_vs_t": {s: float(np.mean([x[1] for x in v["jd1"] if seg_of(x[0]) == seg])) for s, v in allr.items()}}
        rows = [x for s in allr.values() for x in s["jd2"] if seg_of(x[0]) == seg]
        if len(rows) > 30:
            y, ph, pc = (np.array([x[k] for x in rows]) for k in (1, 2, 3))
            bss, lo_, hi_ = block_bss(y, ph, pc, block=14)
            res["JD2"][seg] = {"n_days": len(rows), "base_rate": float(y.mean()), "mean_p_hawkes": float(ph.mean()), "mean_p_const": float(pc.mean()), "bss": bss, "ci90": [lo_, hi_],
                               "hawkes_alpha_last": {s: v["fits"][-1]["hawkes_alpha"] for s, v in allr.items() if v["fits"]},
                               "lr_median": float(np.median([fz["lr"] for v in allr.values() for fz in v["fits"]]))}
        ev = pl.DataFrame([e for s in allr.values() for e in s["jd3"] if seg_of(e["ts"]) == seg]) if any(allr[s]["jd3"] for s in allr) else pl.DataFrame()
        if ev.height:
            ev = ev.with_columns(pl.col("ts").dt.date().alias("day"))
            g = {}
            for leg, q in (("low", ev.filter(pl.col("vol_state") == "low")), ("normal_high", ev.filter(pl.col("vol_state") != "low"))):
                if q.height >= 5:
                    groups = pl.Series(q["day"]).rank("dense").to_numpy().astype(int)
                    m24 = sm.OLS(q["f24"].to_numpy(), np.ones(q.height)).fit(cov_type="cluster", cov_kwds={"groups": groups})
                    g[leg] = {"n": q.height, "mean_f6": float(q["f6"].mean()), "mean_f24": float(q["f24"].mean()), "t_f24": float(m24.tvalues[0])}
                else:
                    g[leg] = {"n": q.height}
            res["JD3"][seg] = g
        rows4 = [x for s in allr.values() for x in s["jd4"] if seg_of(x[0]) == seg]
        if len(rows4) > 30:
            y, side, pj, pg = (np.array([x[k] for x in rows4]) for k in (2, 1, 3, 4))
            dif = (pj - y) ** 2 - (pg - y) ** 2
            rng = np.random.default_rng(1)
            n = len(dif)
            starts = np.arange(0, n, 10)
            boot = [dif[np.concatenate([np.arange(s0, min(n, s0 + 10)) for s0 in rng.choice(starts, size=len(starts))])].mean() for _ in range(800)]
            res["JD4"][seg] = {"n": n, "hit_rate": float(y.mean()), "mean_p_jd": float(pj.mean()), "mean_p_gauss": float(pg.mean()),
                               "brier_jd": float(np.mean((pj - y) ** 2)), "brier_gauss": float(np.mean((pg - y) ** 2)), "diff": float(dif.mean()),
                               "ci90": [float(np.percentile(boot, 5)), float(np.percentile(boot, 95))],
                               "long": {"hit": float(y[side > 0].mean()), "p_jd": float(pj[side > 0].mean()), "p_gauss": float(pg[side > 0].mean())},
                               "short": {"hit": float(y[side < 0].mean()), "p_jd": float(pj[side < 0].mean()), "p_gauss": float(pg[side < 0].mean())}}
    out = load().root / "research" / "reports" / "jump_study_2026-11-08.json"
    out.write_text(json.dumps(res, indent=1, default=str))
    print(json.dumps(res, indent=1, default=lambda x: round(x, 4) if isinstance(x, float) else str(x)))
    if a.record:
        from ai_trader.research.hypotheses import record
        R = res
        rules = {
            "H-JD1": lambda s, seg: s.get("mean_vs_t", -1) > 0 and (s.get("t_vs_t", 0) > 2 if seg == "llm_in_sample" else True),
            "H-JD2": lambda s, seg: s.get("bss", -1) > 0 and (s["ci90"][0] > 0 if seg == "llm_in_sample" else True),
            "H-JD3": lambda s, seg: (s.get("low", {}).get("mean_f24", 0) > 0 and s.get("normal_high", {}).get("mean_f24", 0) < 0
                                     and (max(abs(s.get("low", {}).get("t_f24", 0)), abs(s.get("normal_high", {}).get("t_f24", 0))) > 2 if seg == "llm_in_sample" else True)),
            "H-JD4": lambda s, seg: s.get("diff", 1) < 0 and (s["ci90"][1] < 0 if seg == "llm_in_sample" else True),
        }
        for hid, rule in rules.items():
            key = hid.replace("H-", "")
            for seg in ("llm_in_sample", "clean_holdout"):
                s = R[key].get(seg)
                verdict = "inconclusive" if not s else ("supports" if rule(s, seg) else "contradicts")
                record(hid, seg, verdict, s or {}, data=f"lake 1h {','.join(a.symbols)}; monthly walk-forward from 2023-01; scripts/jump_study.py")
        print("recorded")


if __name__ == "__main__":
    main()
