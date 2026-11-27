"""WS10 fits, cost table and re-pricing on the reductions of ai_trader.research.micro_ws10 (H-WS10-1..15).

    uv run python -m ai_trader.research.micro_ws10_fit costs      # cost table -> data/research/ws10/cost_inputs.parquet
    uv run python -m ai_trader.research.micro_ws10_fit reprice    # zoo_ensemble / trend_4h / hrb2 through stress with the model
    uv run python -m ai_trader.research.micro_ws10_fit impact     # Kyle specs K0..K5, DM-HAC
    uv run python -m ai_trader.research.micro_ws10_fit meta       # square-root law on metaorders
    uv run python -m ai_trader.research.micro_ws10_fit prop       # propagator
    uv run python -m ai_trader.research.micro_ws10_fit spread     # spread decomposition, Roll, Huang-Stoll
    uv run python -m ai_trader.research.micro_ws10_fit infoshare  # Hasbrouck / Gonzalo-Granger
Results JSON: data/research/ws10/results/<cmd>.json
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import polars as pl

from ai_trader.research.micro_ws10 import MAKER, OUT, QS, ROOT, STRESS_SYMS, SYMS, TAKER, split_of

RES = OUT / "results"
SEED = 20261127
N_BOOT = 2000


def _w(name: str, obj: dict) -> None:
    RES.mkdir(parents=True, exist_ok=True)
    (RES / f"{name}.json").write_text(json.dumps(obj, indent=1, default=lambda x: None if (isinstance(x, float) and not math.isfinite(x)) else float(x)))


def days_done() -> list[str]:
    return sorted(p.name for p in (OUT / "day").iterdir() if p.is_dir())


def read_kind(kind: str, syms=SYMS) -> pl.DataFrame:
    parts = []
    for d in days_done():
        for s in syms:
            p = OUT / "day" / d / f"{s}_{kind}.parquet"
            if p.exists():
                parts.append(pl.read_parquet(p).with_columns(pl.lit(d).alias("date"), pl.lit(s).alias("symbol")))
    return pl.concat(parts, how="diagonal_relaxed") if parts else pl.DataFrame()


def seg_col(e: pl.Expr) -> pl.Expr:
    return pl.when(e < "2026-07-01").then(pl.lit("PRE")).when(e <= "2026-10-29").then(pl.lit("FIT")).otherwise(pl.lit("TEST"))


# ====================================================================== statistics helpers
def nw_var(x: np.ndarray, lag: int) -> float:
    x = x - x.mean()
    n = len(x)
    v = float(x @ x) / n
    for k in range(1, min(lag, n - 1) + 1):
        v += 2 * (1 - k / (lag + 1)) * float(x[k:] @ x[:-k]) / n
    return v


def dm_test(d: np.ndarray, lag: int = 10, h: int = 1) -> dict:
    """Diebold-Mariano on a loss-differential series d (positive = the second model is better if d = L1 - L2),
    Newey-West HAC, Harvey-Leybourne-Newbold small-sample correction."""
    d = d[np.isfinite(d)]
    n = len(d)
    if n < 5:
        return {"n": n, "mean": float("nan"), "t": float("nan")}
    v = nw_var(d, lag)
    t = d.mean() / math.sqrt(v / n) if v > 0 else float("nan")
    hln = math.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    return {"n": n, "mean": float(d.mean()), "t": float(t * hln)}


def day_boot_ci(vals_by_day: np.ndarray, weights: np.ndarray | None = None, n: int = N_BOOT, seed: int = SEED) -> tuple[float, float, float]:
    """Mean of day-level values (weighted) with a day-resampling 95% CI."""
    v = np.asarray(vals_by_day, float)
    w = np.ones_like(v) if weights is None else np.asarray(weights, float)
    ok = np.isfinite(v) & (w > 0)
    v, w = v[ok], w[ok]
    if len(v) < 3:
        return float(np.sum(v * w) / max(w.sum(), 1e-12)) if len(v) else float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(v), (n, len(v)))
    bs = (v[idx] * w[idx]).sum(1) / w[idx].sum(1)
    return float(np.sum(v * w) / w.sum()), float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975))


def qs(x: np.ndarray, probs=(0.5, 0.9, 0.95, 0.99)) -> dict:
    x = x[np.isfinite(x)]
    if not len(x):
        return {f"p{int(p * 100)}": float("nan") for p in probs} | {"mean": float("nan"), "n": 0}
    return {"mean": float(x.mean()), "n": int(len(x))} | {f"p{int(p * 100)}": float(np.quantile(x, p)) for p in probs}


# ====================================================================== cost table (H-WS10-1, -2, -3)
def peg_frame() -> pl.DataFrame:
    pg = read_kind("peg")
    tk = read_kind("taker").select("date", "symbol", "i", "side", "hour", "sig60", "mv60", "taker_100", "hs", "drift")
    return pg.join(tk, on=["date", "symbol", "i", "side"], how="left").with_columns(seg_col(pl.col("date")).alias("seg"))


def impact_at_q(q: float, sym_coef: dict) -> float:
    """Own impact (bp) of a $q order from the 60 s square-root / K2 fit: kappa * sigma_60 * (q / V_60)^beta."""
    if not sym_coef:
        return float("nan")
    x = q / max(sym_coef["v60"], 1.0)
    return float(sym_coef["sig60"] * sym_coef["kappa"] * x ** sym_coef.get("beta", 1.0))


def cost_table() -> dict:
    tk = read_kind("taker").with_columns(seg_col(pl.col("date")).alias("seg"))
    pg = peg_frame()
    st = read_kind("stop").with_columns(seg_col(pl.col("date")).alias("seg"))
    imp = json.loads((RES / "impact.json").read_text()).get("coef_k3_60", {}) if (RES / "impact.json").exists() else {}
    post = ["FIT", "TEST"]
    rows: list[dict] = []
    out: dict = {"notes": "costs in bp of the decision-time mid, fees included; clean_holdout days (FIT post-cutoff + TEST)"}
    # ---- taker by symbol x hour x size
    t_post = tk.filter(pl.col("seg").is_in(post))
    for (sym, hour), g in t_post.group_by(["symbol", "hour"], maintain_order=True):
        for q in QS:
            c = g[f"taker_{int(q)}"].to_numpy().astype(float)
            ib = impact_at_q(q, imp.get(sym, {}))
            rows.append({"symbol": sym, "hour": int(hour), "order": "taker", "q_usd": q, **qs(c), "fee_bp": TAKER,
                         "hs_bp": float(g["hs"].mean()), "walk_bp": float(g[f"walk_{int(q)}"].mean()), "drift_sd_bp": float(g["drift"].std()),
                         "impact_bp": ib, "source": "ms_tape"})
    # ---- pegged by symbol x hour ($100, pessimistic primary; pro-rata kept as a column)
    p_post = pg.filter(pl.col("seg").is_in(post))
    for (sym, hour), g in p_post.filter(~pl.col("prorata")).group_by(["symbol", "hour"], maintain_order=True):
        c = g["cost"].to_numpy().astype(float)
        f = g["filled"].to_numpy()
        run = g.filter(pl.col("mv60") >= pl.col("sig60"))
        gp = p_post.filter(pl.col("prorata") & (pl.col("symbol") == sym) & (pl.col("hour") == hour))
        rows.append({"symbol": sym, "hour": int(hour), "order": "peg", "q_usd": 100.0, **qs(c), "fee_bp": MAKER,
                     "fill_60s": float(f.mean()), "fill_5s": float((f & (g["t_fill"].to_numpy() <= 5)).mean()),
                     "t_fill_med_s": float(np.nanmedian(np.where(f, g["t_fill"].to_numpy(), np.nan))) if f.any() else float("nan"),
                     "mk60_bp": float(np.nanmean(g["mk60"].to_numpy())), "mk5_bp": float(np.nanmean(g["mk5"].to_numpy())),
                     "fallback_share": float(1 - f.mean()), "runaway_mean_bp": float(run["cost"].mean()) if run.height else float("nan"),
                     "runaway_share": float(run.height / g.height), "prorata_mean_bp": float(gp["cost"].mean()) if gp.height else float("nan"),
                     "taker_paired_bp": float(g["taker_100"].mean()), "source": "ms_tape"})
    # ---- stops by symbol (pooled hours) and by k
    s_post = st.filter(pl.col("seg").is_in(post)).with_columns((pl.col("s300") + pl.col("walk100") + TAKER).alias("stop_cost"))
    for (sym, k), g in s_post.group_by(["symbol", "k"], maintain_order=True):
        rows.append({"symbol": sym, "hour": -1, "order": f"stop_k{k}", "q_usd": 100.0, **qs(g["stop_cost"].to_numpy()), "fee_bp": TAKER,
                     "slip_mean_bp": float(g["s300"].mean()), "slip1s_mean_bp": float(g["s1000"].mean()), "source": "ms_tape"})
    for sym, g in s_post.group_by("symbol", maintain_order=True):
        rows.append({"symbol": sym[0], "hour": -1, "order": "stop", "q_usd": 100.0, **qs(g["stop_cost"].to_numpy()), "fee_bp": TAKER,
                     "slip_mean_bp": float(g["s300"].mean()), "slip1s_mean_bp": float(g["s1000"].mean()), "source": "ms_tape"})
    # stop by trigger hour (pooled k) for the hour profile
    for (sym, hour), g in s_post.group_by(["symbol", "trig_hour"], maintain_order=True):
        rows.append({"symbol": sym, "hour": int(hour), "order": "stop", "q_usd": 100.0, **qs(g["stop_cost"].to_numpy()), "fee_bp": TAKER,
                     "slip_mean_bp": float(g["s300"].mean()), "source": "ms_tape"})

    # ---- alts from flow_3s with proxies calibrated on the 5 (overlap days)
    fl = read_flow()
    calib = calibrate_proxies(fl, pg, st) if fl.height else {}
    out["proxy_calibration"] = calib
    if fl.height:
        hr = fl.filter(pl.col("hour") >= 0)
        for (sym, hour), g in hr.group_by(["symbol", "hour"], maintain_order=True):
            if sym in SYMS:
                continue
            hs = float(g["hs_mean"].mean())
            dep = float(g["depth25_med"].median())
            sig3 = float(np.sqrt(np.nanmean(g["sig3"].to_numpy() ** 2)))
            for q in QS:
                dens = max(dep, 1.0) / max(25.0 - hs, 1.0)  # USD per bp beyond the touch (linear book)
                walk = 0.5 * q / dens if q > 0 else 0.0
                drift_sd = sig3 / math.sqrt(3.0) * calib.get("drift_ratio", 1.0)
                mean = TAKER + hs + walk
                rows.append({"symbol": sym, "hour": int(hour), "order": "taker", "q_usd": q, "mean": mean, "n": int(g["n3"].sum()),
                             "p50": mean, "p90": mean + 1.2816 * drift_sd, "p95": mean + 1.6449 * drift_sd, "p99": mean + 2.3263 * drift_sd,
                             "fee_bp": TAKER, "hs_bp": hs, "walk_bp": walk, "drift_sd_bp": drift_sd, "impact_bp": float("nan"), "source": "flow3s_proxy"})
            p3 = float(np.nanmean(g["peg3_cost"].to_numpy()))
            pc = calib.get("peg", {})
            # level calibration: raw 3 s proxy minus its mean bias vs the ms tape on the 5 perps (the OLS slope is
            # attenuated by proxy noise, so it is not used to extrapolate to wide-spread alts)
            bias = float(np.mean([v["proxy_raw_mean"] - v["ms_mean"] for v in pc.get("loso", {}).values()])) if pc.get("loso") else 0.0
            pm = p3 - bias
            rows.append({"symbol": sym, "hour": int(hour), "order": "peg", "q_usd": 100.0, "mean": pm, "n": int(g["peg3_n"].sum()),
                         "fee_bp": MAKER, "fill_60s": float(np.nanmean(g["peg3_fill"].to_numpy())), "peg3_raw_bp": p3, "source": "flow3s_proxy"})
        # alt stops: structural model (labelled deviation: the registered 3 s stop proxy failed calibration, r ~ 0.3):
        # slip = alt half-spread + the 5 perps' mean excess over their half-spread; tail excess scaled by 3 s vol
        s5 = s_post.join(fl.filter(pl.col("hour") >= 0).group_by("symbol").agg(pl.col("hs_mean").mean().alias("hs5"),
                                                                                   (pl.col("sig3") ** 2).mean().sqrt().alias("sg5")), on="symbol")
        ex = (s5["s300"] - s5["hs5"]).to_numpy()
        ex_mean = float(np.mean(ex))
        ex_q = {q_: float(np.quantile(ex, q_ / 100)) for q_ in (50, 90, 95, 99)}
        sg5 = float(np.median(s5.group_by("symbol").agg(pl.col("sg5").first())["sg5"].to_numpy()))
        out["alt_stop_model"] = {"excess_mean": ex_mean, "excess_q": ex_q, "sig3_ref": sg5}
        for sym, g in hr.group_by("symbol", maintain_order=True):
            if sym[0] in SYMS:
                continue
            hs = float(g["hs_mean"].mean())
            sc_ = float(np.sqrt(np.nanmean(g["sig3"].to_numpy() ** 2))) / sg5
            mm = hs + ex_mean * max(sc_, 1.0)
            rows.append({"symbol": sym[0], "hour": -1, "order": "stop", "q_usd": 100.0, "mean": mm + TAKER, "n": int(g["n3"].sum()),
                         **{f"p{q_}": hs + v * max(sc_, 1.0) + TAKER for q_, v in ex_q.items()}, "fee_bp": TAKER, "slip_mean_bp": mm,
                         "hs_bp": hs, "vol_scale": sc_, "source": "flow3s_structural"})
    tab = pl.DataFrame(rows, infer_schema_length=None).with_columns(pl.lit("clean_holdout").alias("segment"))
    tab.write_parquet(OUT / "cost_inputs.parquet", compression="zstd")
    out["n_rows"] = tab.height

    # ---- H-WS10-2: paired taker - peg, day-cluster CI per symbol (TEST)
    h2 = {}
    pt = pg.filter((pl.col("seg") == "TEST") & ~pl.col("prorata")).with_columns((pl.col("taker_100") - pl.col("cost")).alias("gain"))
    for sym in SYMS:
        g = pt.filter(pl.col("symbol") == sym)
        dd = g.group_by("date").agg(pl.col("gain").mean(), pl.len()).sort("date")
        m, lo, hi = day_boot_ci(dd["gain"].to_numpy(), dd["len"].to_numpy())
        byh = g.group_by("hour").agg(pl.col("gain").mean())
        h2[sym] = {"gain": m, "lo": lo, "hi": hi, "hours_pos": float((byh["gain"] > 0).mean()), "n": g.height,
                   "peg_mean": float(g["cost"].mean()), "taker_mean": float(g["taker_100"].mean()), "fill_60s": float(g["filled"].mean()),
                   "runaway_gain": float(g.filter(pl.col("mv60") >= pl.col("sig60"))["gain"].mean())}
    # llm_in_sample (PRE days) same statistic
    pp = pg.filter((pl.col("seg") == "PRE") & ~pl.col("prorata")).with_columns((pl.col("taker_100") - pl.col("cost")).alias("gain"))
    h2_pre = {s: float(pp.filter(pl.col("symbol") == s)["gain"].mean()) for s in SYMS}
    out["H-WS10-2"] = {"test": h2, "pre_gain": h2_pre}

    # ---- H-WS10-3: taker slippage beyond touch-mid and stop slippage (TEST)
    h3 = {}
    tt = tk.filter(pl.col("seg") == "TEST").with_columns((pl.col("taker_100") - TAKER).alias("slip"))
    stt = st.filter(pl.col("seg") == "TEST")
    for sym in SYMS:
        g = tt.filter(pl.col("symbol") == sym)
        dd = g.group_by("date").agg(pl.col("slip").mean(), pl.len())
        a = day_boot_ci(dd["slip"].to_numpy(), dd["len"].to_numpy())
        gs = stt.filter(pl.col("symbol") == sym)
        ds = gs.group_by("date").agg(pl.col("s300").mean(), pl.len())
        b = day_boot_ci(ds["s300"].to_numpy(), ds["len"].to_numpy())
        h3[sym] = {"taker_slip": a, "stop_slip": b, "stop_p99": float(np.quantile(gs["s300"].to_numpy(), 0.99)) if gs.height else float("nan"),
                   "stop_n": gs.height, "stop_by_k": {str(k): float(gs.filter(pl.col("k") == k)["s300"].mean()) for k in (0.5, 1.0, 2.0)}}
    out["H-WS10-3"] = h3

    # ---- validation: FIT-based quantiles vs TEST outcomes (coverage) and live / paper fills
    val = {}
    for sym in SYMS:
        gf = pg.filter((pl.col("symbol") == sym) & (pl.col("seg") == "FIT") & ~pl.col("prorata"))["cost"].to_numpy()
        gt = pg.filter((pl.col("symbol") == sym) & (pl.col("seg") == "TEST") & ~pl.col("prorata"))["cost"].to_numpy()
        sf = st.filter((pl.col("symbol") == sym) & (pl.col("seg") == "FIT"))["s300"].to_numpy()
        stt_ = st.filter((pl.col("symbol") == sym) & (pl.col("seg") == "TEST"))["s300"].to_numpy()
        val[sym] = {"peg_cov90": float((gt <= np.quantile(gf, 0.9)).mean()), "peg_cov99": float((gt <= np.quantile(gf, 0.99)).mean()),
                    "peg_mean_fit": float(gf.mean()), "peg_mean_test": float(gt.mean()),
                    "stop_cov90": float((stt_ <= np.quantile(sf, 0.9)).mean()) if len(sf) and len(stt_) else float("nan"),
                    "stop_cov99": float((stt_ <= np.quantile(sf, 0.99)).mean()) if len(sf) and len(stt_) else float("nan")}
    out["validation_fit_to_test"] = val
    out["live_fills"] = live_fill_check(pg, st)
    out["paper_fills"] = paper_fill_check()
    out["funding"] = funding_summary()
    _w("costs", out)
    return out


def read_flow() -> pl.DataFrame:
    fs = sorted((OUT / "flow").glob("20*.parquet"))
    return pl.concat([pl.read_parquet(f) for f in fs], how="diagonal_relaxed") if fs else pl.DataFrame()


def calibrate_proxies(fl: pl.DataFrame, pg: pl.DataFrame, st: pl.DataFrame) -> dict:
    """On the 5 PART perps x overlap days: ms-tape pegged cost and stop slip vs the 3 s proxies (OLS a + b x across
    symbol-day-hours / symbol-days), leave-one-symbol-out error; the 1 s / 3 s drift ratio."""
    out: dict = {}
    ph = (pg.filter(~pl.col("prorata")).group_by(["date", "symbol", "hour"]).agg(pl.col("cost").mean().alias("ms_cost"), pl.col("filled").mean().alias("ms_fill")))
    fh = fl.filter(pl.col("hour") >= 0).select("date", "symbol", "hour", "peg3_cost", "peg3_fill", "sig3", "hs_mean")
    j = ph.join(fh, on=["date", "symbol", "hour"]).drop_nulls(["peg3_cost", "ms_cost"])
    if j.height > 20:
        x, y = j["peg3_cost"].to_numpy(), j["ms_cost"].to_numpy()
        b, a = np.polyfit(x, y, 1)
        loso = {}
        for s in SYMS:
            m = j["symbol"].to_numpy() == s
            if m.sum() < 5 or (~m).sum() < 5:
                continue
            bb, aa = np.polyfit(x[~m], y[~m], 1)
            loso[s] = {"bias": float(np.mean(aa + bb * x[m] - y[m])), "mae": float(np.mean(np.abs(aa + bb * x[m] - y[m]))), "ms_mean": float(y[m].mean()),
                       "proxy_raw_mean": float(x[m].mean())}
        out["peg"] = {"a": float(a), "b": float(b), "n": j.height, "r": float(np.corrcoef(x, y)[0, 1]), "loso": loso}
    sd = st.group_by(["date", "symbol"]).agg(pl.col("s300").mean().alias("ms_stop"), pl.len().alias("n"))
    f3 = fl.filter(pl.col("hour") == -1).select("date", "symbol", "stop3_mean")
    js = sd.join(f3, on=["date", "symbol"]).drop_nulls()
    if js.height > 10:
        x, y = js["stop3_mean"].to_numpy(), js["ms_stop"].to_numpy()
        b, a = np.polyfit(x, y, 1)
        out["stop"] = {"a": float(a), "b": float(b), "n": js.height, "r": float(np.corrcoef(x, y)[0, 1]),
                       "ms_mean": float(y.mean()), "proxy_mean": float(x.mean())}
    tk = read_kind("taker").group_by(["date", "symbol"]).agg(pl.col("drift").std().alias("sd1"))
    f3d = fl.filter(pl.col("hour") >= 0).group_by(["date", "symbol"]).agg((pl.col("sig3") ** 2).mean().sqrt().alias("sig3"))
    jd = tk.join(f3d, on=["date", "symbol"]).drop_nulls()
    if jd.height:
        out["drift_ratio"] = float(np.median(jd["sd1"].to_numpy() / (jd["sig3"].to_numpy() / math.sqrt(3.0))))
    return out


def live_fill_check(pg: pl.DataFrame, st: pl.DataFrame) -> dict:
    """Real fills (data/journal/live/*/fill.jsonl): maker entries' slip vs mid and time to fill, stop slip vs stop; compared
    with the model distribution of the same symbol (all post-cutoff days). No balances are read or written."""
    rows = []
    for f in sorted((ROOT / "data" / "journal" / "live").glob("*/fill.jsonl")):
        for line in f.read_text(encoding="utf-8").splitlines():
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            rows.append({k: r.get(k) for k in ("ts", "exec_id", "symbol", "leg", "is_maker", "slip_bps_vs_mid", "time_to_fill_ms", "slip_bps_vs_stop",
                                                "exit_kind", "fee_rate", "side")})
    seen, uniq = set(), []
    for r in rows:  # the journal repeats some fills (same exec_id)
        if r.get("exec_id") in seen:
            continue
        seen.add(r.get("exec_id"))
        uniq.append(r)
    rows = uniq
    out = {"fills": rows}
    for r in rows:
        sym = r["symbol"]
        if r.get("leg") == "entry" and r.get("is_maker") and r.get("slip_bps_vs_mid") is not None and sym in SYMS:
            g = pg.filter((pl.col("symbol") == sym) & ~pl.col("prorata") & pl.col("filled"))
            slip = (g["cost"] - MAKER).to_numpy()
            r["model_slip_pctile"] = float((slip <= r["slip_bps_vs_mid"]).mean())
            r["model_tfill_pctile"] = float((g["t_fill"].to_numpy() <= r["time_to_fill_ms"] / 1000).mean()) if r.get("time_to_fill_ms") else None
        if r.get("slip_bps_vs_stop") is not None and r.get("exit_kind") == "StopLoss" and sym in SYMS:
            g = st.filter(pl.col("symbol") == sym)["s300"].to_numpy()
            r["model_stop_pctile"] = float((g <= r["slip_bps_vs_stop"]).mean()) if len(g) else None
    return out


def paper_fill_check() -> dict:
    """Paper book fills vs the real 1 s book mid at the fill time (PART days, 5 perps): the paper book's assumed costs
    against the market."""
    p = ROOT / "data" / "exec" / "paper_book" / "paper" / "ai" / "fills.jsonl"
    if not p.exists():
        return {}
    rows = [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]
    part = ROOT / "data" / "research" / "participants" / "days"
    res = []
    for r in rows:
        sym = r.get("symbol")
        if sym not in SYMS:
            continue
        d = __import__("datetime").datetime.fromtimestamp(r["ts"] / 1000, __import__("datetime").UTC).strftime("%Y-%m-%d")
        bp = part / d / "book_1s" / f"{sym}.parquet"
        if not bp.exists():
            continue
        b = pl.read_parquet(bp, columns=["ts", "mid"])
        i = int(np.searchsorted(b["ts"].to_numpy(), r["ts"], "left"))
        if i >= b.height:
            continue
        mid = float(b["mid"][i])
        s = 1 if r["side"] == "buy" else -1
        res.append({"date": d, "symbol": sym, "kind": r.get("kind"), "reason": r.get("reason"), "liquidity": r.get("liquidity"),
                    "slip_vs_mid_bp": s * (r["price"] / mid - 1) * 1e4})
    df = pl.DataFrame(res) if res else pl.DataFrame()
    if not df.height:
        return {"n": 0}
    return {"n": df.height, "by": df.group_by(["liquidity", "reason"]).agg(pl.len(), pl.col("slip_vs_mid_bp").mean().alias("mean"),
                                                                          pl.col("slip_vs_mid_bp").median().alias("median")).to_dicts()}


def load_funding(syms) -> pl.DataFrame:
    lf = pl.scan_parquet(str(ROOT / "data" / "lake" / "funding" / "venue=bybit" / "*" / "*.parquet"))
    return (lf.filter(pl.col("symbol").is_in(list(syms))).select("symbol", "event_ts", "rate").unique(["symbol", "event_ts"])
            .sort(["symbol", "event_ts"]).collect())


def funding_summary() -> dict:
    f = load_funding(STRESS_SYMS)
    last = f.filter(pl.col("event_ts") >= pl.col("event_ts").max() - pl.duration(days=90))
    out = {}
    for sym, g in last.group_by("symbol", maintain_order=True):
        r = g["rate"].to_numpy() * 1e4
        out[sym[0]] = {"mean_bp_per_settle": float(r.mean()), "abs_p90": float(np.quantile(np.abs(r), 0.9)), "n": int(len(r)),
                       "settles_per_day": float(len(r) / 90.0)}
    return out


# ====================================================================== re-pricing (H-WS10-4, -5)
UNITS = ("zoo_ensemble", "trend_4h", "hrb2")


def reprice() -> dict:
    from ai_trader.research import stress

    tab = pl.read_parquet(OUT / "cost_inputs.parquet")
    res_stress = json.loads((ROOT / "data" / "research" / "stress" / "results.json").read_text())["units"]
    trades = []
    for s in STRESS_SYMS:
        p = ROOT / "data" / "research" / "stress" / "trades" / f"{s}.parquet"
        if p.exists():
            t = pl.read_parquet(p).filter(pl.col("unit").is_in(UNITS) & pl.col("variant").is_in(["base", "lat5"]))
            trades.append(t)
    tr = pl.concat(trades, how="vertical_relaxed")
    fund = load_funding(STRESS_SYMS)
    # cost lookups: symbol x hour (fallback: symbol pooled; then the median of the symbols)
    def lut(order: str, col: str = "mean", q: float = 100.0) -> dict:
        g = tab.filter((pl.col("order") == order) & (pl.col("q_usd") == q) & (pl.col("hour") >= 0))
        d = {(r["symbol"], r["hour"]): r[col] for r in g.select("symbol", "hour", col).to_dicts() if r[col] is not None}
        for sym, gg in g.group_by("symbol"):
            v = gg[col].drop_nulls().to_numpy()
            if len(v):
                d[(sym[0], -1)] = float(np.mean(v))
        allv = [v for v in d.values() if v is not None and np.isfinite(v)]
        d[("*", -1)] = float(np.median(allv)) if allv else float("nan")
        return d
    peg_m, peg_q = lut("peg", "mean"), lut("peg", "p90")
    tak_m, tak_q = lut("taker", "mean"), lut("taker", "p90")
    stops = tab.filter((pl.col("order") == "stop") & (pl.col("hour") == -1))
    stop_m = {r["symbol"]: r["mean"] for r in stops.to_dicts()}
    stop_q = {r["symbol"]: (r.get("p90") if r.get("p90") is not None else r.get("p99")) for r in stops.to_dicts()}
    stop_m["*"] = float(np.median([v for v in stop_m.values() if v is not None]))
    stop_q["*"] = float(np.median([v for v in stop_q.values() if v is not None]))

    def get(d, sym, hour):
        for k in ((sym, hour), (sym, -1), ("*", -1)):
            v = d.get(k)
            if v is not None and np.isfinite(v):
                return float(v)
        return float("nan")

    e_h = tr["entry_ts"].dt.hour().to_numpy()
    x_h = tr["exit_ts"].dt.hour().to_numpy()
    syms = tr["symbol"].to_numpy()
    why = tr["why"].to_numpy()
    kind = tr["kind"].to_numpy()
    out: dict = {}
    # funding per trade: sum of settlements in (entry, exit], long pays positive rates
    fpay = np.zeros(tr.height)
    ets = tr["entry_ts"].dt.epoch("ms").to_numpy()
    xts = tr["exit_ts"].dt.epoch("ms").to_numpy()
    for sym in np.unique(syms):
        g = fund.filter(pl.col("symbol") == sym)
        if not g.height:
            continue
        fts = g["event_ts"].dt.epoch("ms").to_numpy()
        cr = np.concatenate([[0.0], np.cumsum(g["rate"].to_numpy())])
        m = syms == sym
        a = np.searchsorted(fts, ets[m], "right")
        b = np.searchsorted(fts, xts[m], "right")
        fpay[m] = cr[b] - cr[a]
    side = tr["side"].to_numpy().astype(float)
    e, s, x = (tr[k].to_numpy().astype(float) for k in ("entry", "stop", "exit"))
    den = np.abs(e - s) + e * stress.base_budget(kind)

    def r_model(pess: bool, fee_mult: float = 1.0, extra: float = 0.0, with_funding: bool = True) -> np.ndarray:
        c_in = np.array([get(peg_q if pess else peg_m, sy, h) if k == "peg" else get(tak_q if pess else tak_m, sy, h)
                         for sy, h, k in zip(syms, e_h, kind, strict=True)])
        fee_in = np.where(kind == "peg", MAKER, TAKER)
        c_in = c_in + (fee_mult - 1) * fee_in + extra
        sym_key = [sy if sy in stop_m else "*" for sy in syms]
        c_out = np.array([(stop_q if pess else stop_m)[k_] if w == "stop" else get(tak_q if pess else tak_m, sy, h)
                          for sy, k_, w, h in zip(syms, sym_key, why, x_h, strict=True)])
        c_out = c_out + (fee_mult - 1) * TAKER + extra
        fcost = side * fpay * e if with_funding else 0.0
        return (side * (x - e) - c_in / 1e4 * e - c_out / 1e4 * x - fcost) / den

    def ci_block(t: pl.DataFrame, r: np.ndarray, block: float) -> tuple[float, float, float, float]:
        """Mean R per trade, stationary bootstrap of entry days with mean block `block` days (RT-STRESS-1), P(mean <= 0)."""
        dd = stress.daily(t, r)
        n_, s_ = dd["n"].to_numpy().astype(float), dd["R"].to_numpy().astype(float)
        rng = np.random.default_rng(stress.SEED)
        idx = stress.stationary_boot_idx(len(n_), stress.N_BOOT, block, rng)
        ns, ss_ = n_[idx].sum(axis=1), s_[idx].sum(axis=1)
        m = ss_[ns > 0] / ns[ns > 0]
        return float(np.mean(r)), float(np.quantile(m, 0.025)), float(np.quantile(m, 0.975)), float((m <= 0).mean())

    for u in UNITS:
        mb = (tr["unit"].to_numpy() == u) & (tr["variant"].to_numpy() == "base") & (why != "end")  # closed trades only (RT-STRESS-2)
        ml = (tr["unit"].to_numpy() == u) & (tr["variant"].to_numpy() == "lat5") & (why != "end")
        tb, tl = tr.filter(pl.Series(mb)), tr.filter(pl.Series(ml))
        hold = (xts[mb] - ets[mb]) / 86400000
        block = max(5.0, float(np.quantile(hold, 0.9)))
        rs = {"harness": stress.r_net(tb), "model": r_model(False)[mb], "model_nofund": r_model(False, with_funding=False)[mb],
              "model_q90": r_model(True)[mb]}
        u_out = {"n_closed": int(mb.sum()), "n_open_dropped": int(((tr["unit"].to_numpy() == u) & (tr["variant"].to_numpy() == "base") & (why == "end")).sum()),
                 "block_days": block, "hold_days_median": float(np.median(hold)), "hold_days_p90": float(np.quantile(hold, 0.9)),
                 "funding_R_mean": float(np.mean((side * fpay * e / den)[mb])), "funding_bp_mean": float(np.mean((side * fpay)[mb]) * 1e4),
                 "cost_in_bp_mean_model": float(np.mean(np.array([get(peg_m, sy, h) if k == "peg" else get(tak_m, sy, h)
                                                                  for sy, h, k in zip(syms[mb], e_h[mb], kind[mb], strict=True)]))),
                 "stop_share": float((why[mb] == "stop").mean())}
        seg = tb["seg"].to_numpy()
        ho_m = seg == "clean_holdout"
        for k, r in rs.items():
            if k == "harness":
                f15r, s5r, l5r = stress.r_net(tb, stress.Costs(fee_mult=1.5)), stress.r_net(tb, stress.Costs(extra_bps=5.0)), stress.r_net(tl)
            else:
                pess, wf = k == "model_q90", k != "model_nofund"
                f15r = r_model(pess, fee_mult=1.5, with_funding=wf)[mb]
                s5r = r_model(pess, extra=5.0, with_funding=wf)[mb]
                l5r = r_model(pess, with_funding=wf)[ml]
            m, lo, hi, p0 = ci_block(tb, r, block)
            base_m = stress.metrics(tb, r, ci=False) | {"mean": m, "lo": lo, "hi": hi}
            hm, hlo, hhi, hp0 = ci_block(tb.filter(pl.Series(ho_m)), r[ho_m], block) if ho_m.sum() > 2 else (float("nan"),) * 4
            ho = {"n": int(ho_m.sum()), "mean": hm}
            f15 = {"n": len(f15r), "mean": float(np.mean(f15r))}
            s5 = {"n": len(s5r), "mean": float(np.mean(s5r))}
            l5 = {"n": len(l5r), "mean": float(np.mean(l5r))}
            v, fired = stress.verdict({"base": base_m, "fees15": f15, "slip5": s5, "lat5": l5, "holdout": ho,
                                       "cliff": res_stress[u].get("cliff"), "crisis_engine_min": res_stress[u].get("crisis_engine_min")})
            srt = np.sort(r[ho_m])
            u_out[k] = {"verdict": v, "fired": fired, "mean": m, "lo": lo, "hi": hi, "p_le0": p0, "n": int(len(r)),
                        "holdout_mean": hm, "holdout_lo": hlo, "holdout_hi": hhi, "holdout_n": int(ho_m.sum()),
                        "holdout_ex_top5": float(srt[:-5].sum()) if len(srt) > 5 else float("nan"),
                        "fees15": f15["mean"], "slip5": s5["mean"], "lat5": l5["mean"], "total_ex_top5": base_m.get("total_ex_top5")}
        u_out["stress_reported_verdict"] = res_stress[u].get("verdict")
        out[u] = u_out
    _w("reprice", out)
    return out


# ====================================================================== impact (H-WS10-6..9)
def _fit_k(g: pl.DataFrame) -> dict:
    dm, fl, sig, vh, ofi, dh = (g[c].to_numpy().astype(float) for c in ("dm", "flow", "sig", "vhat", "ofi", "dhat"))
    ok = np.isfinite(dm) & (sig > 0) & (vh > 0) & (dh > 0)
    dm, fl, sig, vh, ofi, dh = dm[ok], fl[ok], sig[ok], vh[ok], ofi[ok], dh[ok]
    lam = float(fl @ dm / (fl @ fl))
    x, z = fl / vh, dm / sig
    kap = float(x @ z / (x @ x))
    best = (np.inf, 1.0, kap)
    ax = np.abs(x)
    sx = np.sign(x)
    for b in np.arange(0.1, 1.51, 0.02):
        u = sx * ax ** b
        k = float(u @ z / (u @ u))
        sse = float(np.sum((z - k * u) ** 2))
        if sse < best[0]:
            best = (sse, float(b), k)
    u4 = ofi / dh
    b4 = float(u4 @ dm / (u4 @ u4))
    X5 = np.column_stack([u4, sig * x])
    c5, *_ = np.linalg.lstsq(X5, dm, rcond=None)
    pred = {"K1": lam * fl, "K2": sig * kap * x, "K3": sig * best[2] * sx * ax ** best[1], "K4": b4 * u4, "K5": X5 @ c5}
    zq = {k: np.quantile((dm - p) / sig, [0.1, 0.5, 0.9]) for k, p in pred.items()}
    return {"lam": lam, "kappa": kap, "beta": best[1], "kappa3": best[2], "b4": b4, "c5": [float(c) for c in c5], "zq": {k: v.tolist() for k, v in zq.items()}}


def _pred(g: pl.DataFrame, c: dict) -> dict:
    dm, fl, sig, vh, ofi, dh = (g[k].to_numpy().astype(float) for k in ("dm", "flow", "sig", "vhat", "ofi", "dhat"))
    x = fl / np.where(vh > 0, vh, np.nan)
    u4 = ofi / np.where(dh > 0, dh, np.nan)
    p = {"K0": np.zeros_like(dm), "K1": c["lam"] * fl, "K2": sig * c["kappa"] * x, "K3": sig * c["kappa3"] * np.sign(x) * np.abs(x) ** c["beta"],
         "K4": c["b4"] * u4, "K5": c["c5"][0] * u4 + c["c5"][1] * sig * x}
    return p


def pinball(y, q, tau):
    e = y - q
    return np.maximum(tau * e, (tau - 1) * e)


def impact() -> dict:
    iv = read_kind("iv").with_columns(seg_col(pl.col("date")).alias("seg")).drop("hour")
    out: dict = {"coef_k3_60": {}}
    for h in (10, 60, 300):
        hv = iv.filter(pl.col("h") == h)
        res_h: dict = {}
        losses: dict = {}
        for sym in SYMS:
            g = hv.filter(pl.col("symbol") == sym)
            fit = g.filter(pl.col("seg").is_in(["PRE", "FIT"]))
            test = g.filter(pl.col("seg") == "TEST")
            c = _fit_k(fit)
            p = _pred(test, c)
            dm = test["dm"].to_numpy().astype(float)
            sig = test["sig"].to_numpy().astype(float)
            r: dict = {"coef": c, "n_fit": fit.height, "n_test": test.height}
            lw = {"date": test["date"], "i": test["i"]}
            for k, pr in p.items():
                e2 = (dm - pr) ** 2
                r[k] = {"r2": float(1 - np.nansum(e2) / np.nansum(dm ** 2)), "mse": float(np.nanmean(e2))}
                if k != "K0":
                    zq = c["zq"][k]
                    r[k]["pinball"] = float(np.nanmean([pinball(dm, pr + sig * zq[j], t).mean() for j, t in enumerate((0.1, 0.5, 0.9))]))
                lw[k] = e2.astype(np.float32)
            losses[sym] = pl.DataFrame(lw)
            # llm_in_sample check: fit post-cutoff FIT only, score PRE
            fpost = g.filter(pl.col("seg") == "FIT")
            pre = g.filter(pl.col("seg") == "PRE")
            if pre.height and fpost.height:
                c2 = _fit_k(fpost)
                p2 = _pred(pre, c2)
                d2 = pre["dm"].to_numpy().astype(float)
                r["pre"] = {k: float(1 - np.nansum((d2 - v) ** 2) / np.nansum(d2 ** 2)) for k, v in p2.items()}
                r["pre_lam"] = c2["lam"]
                # PRE-day fit (sign check) and beta from PRE-only
                r["pre_fit"] = _fit_k(pre)
            # stability: day-level lambda and kappa (all days), CV and a Chow-type F on FIT days
            dl = []
            for (dd,), gd in g.group_by(["date"]):
                f_, d_ = gd["flow"].to_numpy().astype(float), gd["dm"].to_numpy().astype(float)
                x_ = f_ / gd["vhat"].to_numpy().astype(float)
                z_ = d_ / gd["sig"].to_numpy().astype(float)
                ok = np.isfinite(x_) & np.isfinite(z_)
                dl.append({"date": dd, "lam": float(f_ @ d_ / (f_ @ f_)), "kappa": float(x_[ok] @ z_[ok] / (x_[ok] @ x_[ok])), "n": len(d_)})
            dl = pl.DataFrame(dl)
            r["day_cv"] = {"lam": float(dl["lam"].std() / abs(dl["lam"].mean())), "kappa": float(dl["kappa"].std() / abs(dl["kappa"].mean()))}
            fit_days = fit.partition_by("date")
            ssr_u = ssr_r = 0.0
            nn = 0
            f_all, d_all = fit["flow"].to_numpy().astype(float), fit["dm"].to_numpy().astype(float)
            lam_all = f_all @ d_all / (f_all @ f_all)
            ssr_r = float(np.sum((d_all - lam_all * f_all) ** 2))
            for gd in fit_days:
                f_, d_ = gd["flow"].to_numpy().astype(float), gd["dm"].to_numpy().astype(float)
                l_ = f_ @ d_ / (f_ @ f_)
                ssr_u += float(np.sum((d_ - l_ * f_) ** 2))
                nn += len(d_)
            D = len(fit_days)
            r["chow_F_lambda"] = ((ssr_r - ssr_u) / (D - 1)) / (ssr_u / (nn - D)) if D > 1 else float("nan")
            r["chow_df"] = [D - 1, nn - D]
            res_h[sym] = r
            if h == 60:
                out["coef_k3_60"][sym] = {"kappa": c["kappa3"], "beta": c["beta"], "sig60": float(np.nanmedian(g["sig"].to_numpy())),
                                          "v60": float(np.nanmedian(g["vhat"].to_numpy())), "kappa_lin": c["kappa"]}
        dmr = {}
        for a, b in (("K1", "K2"), ("K2", "K3"), ("K2", "K4"), ("K0", "K1"), ("K4", "K5"), ("K1", "K4")):
            per = {s_: lf.select("date", "i", (pl.col(a) - pl.col(b)).alias("d")) for s_, lf in losses.items()}
            pooled = (pl.concat(list(per.values())).group_by(["date", "i"]).agg(pl.col("d").mean()).sort(["date", "i"])["d"].to_numpy())
            dmr[f"{a}_vs_{b}"] = {"pooled": dm_test(pooled.astype(float), lag=10),
                                  **{s_: dm_test(v.sort(["date", "i"])["d"].to_numpy().astype(float), lag=10) for s_, v in per.items()}}
        res_h["dm"] = dmr
        out[f"h{h}"] = res_h
    # beta CI (H-WS10-8): day bootstrap of the K3 fit on FIT days at 60 s
    rng = np.random.default_rng(SEED)
    bci = {}
    hv = iv.filter((pl.col("h") == 60) & pl.col("seg").is_in(["PRE", "FIT"]))
    for sym in SYMS:
        parts = hv.filter(pl.col("symbol") == sym).partition_by("date")
        bs = []
        for _ in range(200):
            pick = rng.integers(0, len(parts), len(parts))
            bs.append(_fit_k(pl.concat([parts[i] for i in pick]))["beta"])
        bci[sym] = [float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975))]
    out["beta_ci_60"] = bci
    _w("impact", out)
    return out


# ====================================================================== metaorders (H-WS10-10)
def meta() -> dict:
    mt = read_kind("meta").with_columns(seg_col(pl.col("date")).alias("seg"))
    mt = mt.filter((pl.col("sig_d") > 0) & (pl.col("v_d") > 0) & pl.col("imp_end").is_finite())
    mt = mt.with_columns((pl.col("imp_end") / pl.col("sig_d")).alias("I"), (pl.col("q") / pl.col("v_d")).alias("x"))
    mt = mt.filter(pl.col("I").is_finite() & pl.col("x").is_finite() & pl.col("sig_d").is_finite() & pl.col("v_d").is_finite())  # NaN > 0 is true in polars
    out: dict = {"n": mt.height, "by_symbol_n": mt.group_by("symbol").len().to_dicts()}

    def huber_fit(x, y, d, c=1.345, it=30):
        u = x ** d
        Y = float(u @ y / (u @ u))
        for _ in range(it):
            r = y - Y * u
            s = 1.4826 * np.median(np.abs(r - np.median(r))) + 1e-12
            w = np.minimum(1.0, c * s / np.maximum(np.abs(r), 1e-12))
            Y = float((w * u) @ y / ((w * u) @ u))
        return Y

    def free_fit(x, y, cap=60000):
        if len(x) > cap:  # fixed-seed subsample for speed (grid x Huber on 300k+ metaorders)
            k = np.random.default_rng(SEED).choice(len(x), cap, replace=False)
            x, y = x[k], y[k]
        best = (np.inf, 0.5, 0.0)
        for d in np.arange(0.05, 1.51, 0.05):
            Y = huber_fit(x, y, d, it=8)
            r = y - Y * x ** d
            s = np.median(np.abs(r))
            loss = np.sum(np.where(np.abs(r) < 1.345 * s, 0.5 * r ** 2, 1.345 * s * (np.abs(r) - 0.5 * 1.345 * s)))
            if loss < best[0]:
                best = (loss, float(d), Y)
        return best[1], best[2]

    rng = np.random.default_rng(SEED)
    for sym in (*SYMS, "pooled"):
        g = mt if sym == "pooled" else mt.filter(pl.col("symbol") == sym)
        fit = g.filter(pl.col("seg").is_in(["PRE", "FIT"]))
        test = g.filter(pl.col("seg") == "TEST")
        if fit.height < 200 or test.height < 50:
            out[sym] = {"n_fit": fit.height, "n_test": test.height}
            continue
        xf, yf = fit["x"].to_numpy(), fit["I"].to_numpy()
        xt, yt = test["x"].to_numpy(), test["I"].to_numpy()
        d_free, Y_free = free_fit(xf, yf)
        specs = {"const": (0.0, float(np.median(yf))), "sqrt": (0.5, huber_fit(xf, yf, 0.5)), "linear": (1.0, huber_fit(xf, yf, 1.0)),
                 "power": (d_free, Y_free)}
        r: dict = {"n_fit": fit.height, "n_test": test.height, "delta_free": d_free, "Y": {k: v[1] for k, v in specs.items()}}
        preds = {k: (v[1] * np.ones_like(xt) if k == "const" else v[1] * xt ** v[0]) for k, v in specs.items()}
        for k, p in preds.items():
            r[k] = {"mse": float(np.mean((yt - p) ** 2)), "pin50": float(pinball(yt, p, 0.5).mean())}
        # DM clustered by day: daily mean loss differences sqrt - linear
        dfr = pl.DataFrame({"date": test["date"], "d": (yt - preds["linear"]) ** 2 - (yt - preds["sqrt"]) ** 2,
                            "dp": pinball(yt, preds["linear"], 0.5) - pinball(yt, preds["sqrt"], 0.5)})
        daily = dfr.group_by("date").agg(pl.col("d").mean(), pl.col("dp").mean()).sort("date")
        r["dm_lin_minus_sqrt_mse"] = dm_test(daily["d"].to_numpy(), lag=2)
        r["dm_lin_minus_sqrt_pin"] = dm_test(daily["dp"].to_numpy(), lag=2)
        # delta CI by day bootstrap on FIT
        parts = fit.partition_by("date")
        bs = []
        for _ in range(60):
            pick = rng.integers(0, len(parts), len(parts))
            gg = pl.concat([parts[i] for i in pick])
            bs.append(free_fit(gg["x"].to_numpy(), gg["I"].to_numpy(), cap=20000)[0])
        r["delta_ci"] = [float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975))]
        # binned means (log x deciles) on all post-cutoff data
        allg = g.filter(pl.col("seg") != "PRE")
        qb = np.quantile(allg["x"].to_numpy(), np.linspace(0, 1, 11))
        bins = []
        for a, b in zip(qb[:-1], qb[1:], strict=True):
            m = allg.filter((pl.col("x") >= a) & (pl.col("x") <= b))
            bins.append({"x_med": float(m["x"].median()), "I_mean": float(m["I"].mean()), "I_med": float(m["I"].median()), "n": m.height,
                         "imp_bp": float(m["imp_end"].mean()), "q_med": float(m["q"].median())})
        r["bins"] = bins
        xb = np.array([b["x_med"] for b in bins])
        ib = np.array([b["I_mean"] for b in bins])
        okb = ib > 0
        r["binned_slope"] = float(np.polyfit(np.log(xb[okb]), np.log(ib[okb]), 1)[0]) if okb.sum() > 3 else float("nan")
        # decay after completion
        r["decay"] = {"imp_end": float(allg["imp_end"].mean()), "imp60": float(allg["imp60"].mean()), "imp300": float(allg["imp300"].mean())}
        dd = allg.group_by("date").agg(pl.col("imp_end").mean(), pl.col("imp300").mean())
        rr = dd["imp300"].to_numpy() / dd["imp_end"].to_numpy()
        r["decay"]["ratio300_day_median"] = float(np.median(rr))
        out[sym] = r
    _w("meta", out)
    return out


# ====================================================================== propagator (H-WS10-11)
def prop() -> dict:
    from scipy.linalg import solve_toeplitz, toeplitz

    out: dict = {}
    pooled_d = {}
    for sym in SYMS:
        days = []
        for d in days_done():
            p = OUT / "day" / d / f"{sym}_prop.npz"
            if p.exists():
                z = np.load(p)
                days.append({"date": d, "seg": split_of(d), "n": int(z["n"]), "yy": float(z["yy"]), "r": z["r"], "c": z["c"], "cn": int(z["c_n"]),
                             "resp": z["resp"]})
        fit = [x for x in days if x["seg"] in ("PRE", "FIT")]
        test = [x for x in days if x["seg"] == "TEST"]
        L = len(fit[0]["r"]) - 1
        cF = sum(x["c"] for x in fit)
        rF = sum(x["r"] for x in fit)
        g_free = solve_toeplitz(cF[: L + 1], rF[: L + 1])
        theta = rF[0] / cF[0]
        g_perm = np.zeros(L + 1)
        g_perm[0] = theta
        TF = toeplitz(cF[: L + 1])

        def kern_pow(beta):
            ell = np.arange(1, L + 3, dtype=float)
            G = ell ** (-beta)
            u = np.r_[G[0], np.diff(G)[: L]]
            return u

        def kern_exp(tau):
            ell = np.arange(1, L + 3, dtype=float)
            e_ = np.exp(-(ell - 1) / tau)
            u1 = np.r_[e_[0], np.diff(e_)[:L]]  # transient part
            u0 = np.r_[1.0, np.zeros(L)]  # permanent part
            return u0, u1

        def fit_lin(us):
            U = np.column_stack(us)
            A = U.T @ TF @ U
            b = U.T @ rF[: L + 1]
            w = np.linalg.solve(A, b)
            return U @ w

        best_pow = min(((float(rF[: L + 1] @ (g := fit_lin([kern_pow(b)])) * -2 + g @ TF @ g), b, g) for b in np.arange(0.02, 1.2, 0.02)), key=lambda t: t[0])
        best_exp = min(((float(rF[: L + 1] @ (g := fit_lin(list(kern_exp(t)))) * -2 + g @ TF @ g), t, g) for t in (1, 2, 3, 5, 7, 10, 15, 20, 30, 50, 70, 100)),
                       key=lambda t: t[0])
        kernels = {"perm": g_perm, "free": g_free, "pow": best_pow[2], "exp": best_exp[2]}

        def mse(x, g):
            T = toeplitz(x["c"][: L + 1])
            return (x["yy"] - 2 * g @ x["r"][: L + 1] + g @ T @ g) / x["n"]

        r: dict = {"theta": float(theta), "beta_pow": float(best_pow[1]), "tau_exp": float(best_exp[1]),
                   "G_free": np.cumsum(g_free)[[0, 1, 4, 9, 19, 49, 99, 199]].tolist(), "G_pow": np.cumsum(best_pow[2])[[0, 1, 4, 9, 19, 49, 99, 199]].tolist()}
        for k, g in kernels.items():
            r[f"mse_{k}"] = float(np.mean([mse(x, g) for x in test]))
            r[f"r2_{k}"] = float(1 - sum(mse(x, g) * x["n"] for x in test) / sum(x["yy"] for x in test))
        dd = np.array([mse(x, g_perm) - mse(x, g_free) for x in test])
        r["dm_perm_minus_free_daily"] = dm_test(dd, lag=2)
        r["dm_perm_minus_pow_daily"] = dm_test(np.array([mse(x, g_perm) - mse(x, best_pow[2]) for x in test]), lag=2)
        pooled_d[sym] = {x["date"]: mse(x, g_perm) - mse(x, g_free) for x in test}
        pre = [x for x in days if x["seg"] == "PRE"]
        if pre:
            fpost = [x for x in days if x["seg"] == "FIT"]
            c2, r2_ = sum(x["c"] for x in fpost), sum(x["r"] for x in fpost)
            g2 = solve_toeplitz(c2[: L + 1], r2_[: L + 1])
            p2 = np.r_[r2_[0] / c2[0], np.zeros(L)]
            r["pre_free_minus_perm_gain"] = float(np.mean([mse(x, p2) - mse(x, g2) for x in pre]))
        # long memory: gamma from C(l)/C(0), lags 10..1000 (all days pooled, and FIT / TEST), day bootstrap CI
        def gamma_of(ds):
            c = sum(x["c"] for x in ds)
            n = sum(x["cn"] for x in ds)
            lag = np.arange(len(c))
            cc = c / (n - lag)
            sel = (lag >= 10) & (lag <= 1000) & (cc > 0)
            return float(-np.polyfit(np.log(lag[sel]), np.log(cc[sel] / cc[0]), 1)[0]), cc[[1, 10, 100, 1000]].tolist()
        gm, ccs = gamma_of(days)
        rng = np.random.default_rng(SEED)
        bs = [gamma_of([days[i] for i in rng.integers(0, len(days), len(days))])[0] for _ in range(300)]
        r["gamma"] = gm
        r["gamma_ci"] = [float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975))]
        r["gamma_test"] = gamma_of(test)[0]
        r["gamma_pre"] = gamma_of(pre)[0] if pre else None
        r["C_at"] = dict(zip(("1", "10", "100", "1000"), ccs, strict=True))
        r["efficiency_gap"] = float(best_pow[1] - (1 - gm) / 2)
        r["resp"] = dict(zip([str(x) for x in (1, 2, 3, 5, 7, 10, 15, 20, 30, 50, 70, 100, 150, 200, 300, 500, 700, 1000)],
                             np.mean([x["resp"] for x in days if x["seg"] != "PRE"], axis=0).tolist(), strict=True))
        out[sym] = r
    dates = sorted(set.intersection(*[set(v) for v in pooled_d.values()]))
    out["pooled_dm_perm_minus_free"] = dm_test(np.array([np.mean([pooled_d[s][d] for s in SYMS]) for d in dates]), lag=2)
    _w("prop", out)
    return out


# ====================================================================== spread decomposition (H-WS10-14, -15)
def spread() -> dict:
    out: dict = {}
    for sym in SYMS:
        rows = []
        for d in days_done():
            p = OUT / "day" / d / f"{sym}_stats.json"
            if not p.exists():
                continue
            js = json.loads(p.read_text())
            sh = pl.DataFrame(js["spread_hour"]).filter(pl.col("notional") > 0)
            w = sh["notional"].to_numpy()
            rows.append({"date": d, "seg": js["split"], "notional": float(w.sum()),
                         "eff_w": float(np.sum(w * sh["eff_w"].to_numpy()) / w.sum()), "rs5_w": float(np.sum(w * sh["rs5_w"].to_numpy()) / w.sum()),
                         "rs60_w": float(np.sum(w * sh["rs60_w"].to_numpy()) / w.sum()),
                         "rs60_e": float(np.nansum(sh["n"].to_numpy() * sh["rs60_e"].to_numpy()) / sh["n"].sum()),
                         "eff_e": float(np.nansum(sh["n"].to_numpy() * sh["eff_e"].to_numpy()) / sh["n"].sum()),
                         "rs60_small": float(np.nansum(sh["n_small"].to_numpy() * np.nan_to_num(sh["rs60_small"].to_numpy())) / max(sh["n_small"].sum(), 1)),
                         "cov1": js["roll"]["cov1"], "roll_bp": js["roll"]["roll_bp"], "eff_full_e": js["roll"]["eff_full_e"], "quoted": js["roll"]["quoted_bp"],
                         "hs_alpha": js["huang_stoll"]["alpha"], "hs_beta": js["huang_stoll"]["beta"], "hs_S": js["huang_stoll"]["S_bp"],
                         "hs_pi": js["huang_stoll"]["pi"]})
            if sym == SYMS[0] and False:
                pass
        df = pl.DataFrame(rows)
        t = df.filter(pl.col("seg") == "TEST")
        pre = df.filter(pl.col("seg") == "PRE")
        r = {"rs60_w": day_boot_ci(t["rs60_w"].to_numpy()), "rs5_w": day_boot_ci(t["rs5_w"].to_numpy()), "eff_w": day_boot_ci(t["eff_w"].to_numpy()),
             "rs60_e": day_boot_ci(t["rs60_e"].to_numpy()), "eff_e": day_boot_ci(t["eff_e"].to_numpy()), "rs60_small": day_boot_ci(t["rs60_small"].to_numpy()),
             "pre_rs60_w": float(pre["rs60_w"].mean()) if pre.height else None,
             "hs_alpha_median": float(t["hs_alpha"].median()), "hs_beta_median": float(t["hs_beta"].median()), "hs_S_median": float(t["hs_S"].median()),
             "hs_alpha_in_01_share": float(((t["hs_alpha"] >= 0) & (t["hs_alpha"] <= 1)).mean()),
             "roll_undefined_share": float((df["cov1"] >= 0).mean()),
             "roll_vs_eff_median": float((df.filter(pl.col("roll_bp").is_not_null())["roll_bp"] / df.filter(pl.col("roll_bp").is_not_null())["eff_full_e"]).median()),
             "quoted_median": float(df["quoted"].median()), "eff_full_e_median": float(df["eff_full_e"].median())}
        # hour profile of the notional-weighted realized spread (post-cutoff)
        out[sym] = r
    _w("spread", out)
    return out


# ====================================================================== information share (H-WS10-12, -13)
def vecm_is(y: np.ndarray, pmax: int = 10) -> dict:
    """Two log prices (rows = time). ECT = y1 - y2 - mean. Lag by BIC. Returns IS bounds for price 1 and the GG share."""
    dy = np.diff(y, axis=0)
    z = (y[:, 0] - y[:, 1])
    z = z - z.mean()
    best = None
    T = len(dy)
    for p in range(1, pmax + 1):
        Y = dy[p:]
        X = [z[p:-1, None] if len(z[p:-1]) == len(Y) else z[p: p + len(Y), None]]
        for i in range(1, p + 1):
            X.append(dy[p - i: T - i])
        X = np.column_stack(X + [np.ones((len(Y), 1))])
        B, *_ = np.linalg.lstsq(X, Y, rcond=None)
        E = Y - X @ B
        Om = E.T @ E / len(Y)
        bic = math.log(np.linalg.det(Om)) + (B.size) * math.log(len(Y)) / len(Y)
        if best is None or bic < best[0]:
            best = (bic, p, B, Om)
    _, p, B, Om = best
    a1, a2 = B[0, 0], B[0, 1]
    psi = np.array([a2, -a1])  # alpha_perp' (up to scale)
    if psi.sum() == 0:
        return {"p": p}
    gg1 = float(psi[0] / psi.sum())
    tot = float(psi @ Om @ psi)
    F = np.linalg.cholesky(Om)
    up = float((psi @ F)[0] ** 2 / tot)
    P = np.array([[0, 1], [1, 0]])
    F2 = np.linalg.cholesky(P @ Om @ P)
    lo = float(1 - (psi[::-1] @ F2)[0] ** 2 / tot)
    return {"p": p, "alpha": [float(a1), float(a2)], "is_upper": max(up, lo), "is_lower": min(up, lo), "is_mid": 0.5 * (up + lo), "gg": gg1,
            "corr_e": float(Om[0, 1] / math.sqrt(Om[0, 0] * Om[1, 1])), "n": int(len(y))}


def infoshare() -> dict:
    out: dict = {"perp_spot": {}, "xvenue": {}}
    flow = ROOT / "data" / "lake" / "flow_3s" / "venue=bybit"
    from ai_trader.research.micro_ws10 import flow_days
    for sym in ("BTCUSDT", "ETHUSDT"):
        per = []
        for d in flow_days():
            parts = {}
            for cat in ("linear", "spot"):
                fs = sorted((flow / f"category={cat}" / f"date={d}").glob("*.parquet"))
                xs = []
                for fpath in fs:
                    x = pl.read_parquet(fpath, columns=["symbol", "event_ts", "mid", "book_cov"])
                    x = x.filter((pl.col("symbol") == sym) & (pl.col("book_cov") >= 0.9) & pl.col("mid").is_not_null())
                    if x.height:
                        xs.append(x)
                if xs:
                    parts[cat] = pl.concat(xs).unique("event_ts").select("event_ts", pl.col("mid").alias(cat))
            if len(parts) < 2:
                continue
            j = parts["linear"].join(parts["spot"], on="event_ts").sort("event_ts")
            if j.height < 0.8 * 28800:
                per.append({"date": d, "rows": j.height, "dropped": True})
                continue
            y = np.log(j.select("linear", "spot").to_numpy())
            r = vecm_is(y)
            r["date"] = d
            per.append(r)
        ok = [x for x in per if "is_mid" in x]
        out["perp_spot"][sym] = {"days": per, "share_mid_gt_half": float(np.mean([x["is_mid"] > 0.5 for x in ok])) if ok else None,
                                 "share_gg_gt_half": float(np.mean([x["gg"] > 0.5 for x in ok])) if ok else None,
                                 "median_is_mid": float(np.median([x["is_mid"] for x in ok])) if ok else None,
                                 "median_gg": float(np.median([x["gg"] for x in ok])) if ok else None, "n_days": len(ok)}
    # xvenue journal: Bybit vs Binance mids on the snapshot grid
    rows = []
    for f in sorted((ROOT / "data" / "journal" / "xvenue").glob("*.jsonl")):
        for line in f.read_text(encoding="utf-8").splitlines():
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            v = r.get("venues", {})
            by, bn = (v.get("by") or {}).get("mid"), (v.get("bn") or {}).get("mid")
            if by and bn:
                rows.append({"ts": r["ts"], "sym": r["sym"], "bn": bn, "by": by})
    xv = pl.DataFrame(rows).with_columns(pl.col("ts").str.to_datetime(time_zone="UTC")).sort(["sym", "ts"])
    for sym in ("BTC", "ETH", "SOL", "XRP", "DOGE"):
        g = xv.filter(pl.col("sym") == sym)
        if g.height < 500:
            continue
        gap = g["ts"].diff().dt.total_seconds().fill_null(9999).to_numpy()
        seg_id = np.cumsum(gap > 180)
        ys = [np.log(g.filter(pl.Series(seg_id == k)).select("bn", "by").to_numpy()) for k in np.unique(seg_id)]
        ys = [y for y in ys if len(y) > 60]
        y = np.concatenate(ys)  # segments joined; the few joins add a small error
        r = vecm_is(y, pmax=5)
        r["median_gap_s"] = float(np.median(gap[gap < 180]))
        out["xvenue"][sym] = r
    _w("infoshare", out)
    return out


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["costs", "reprice", "impact", "meta", "prop", "spread", "infoshare"])
    a = ap.parse_args(argv)
    fn = {"costs": cost_table, "reprice": reprice, "impact": impact, "meta": meta, "prop": prop, "spread": spread, "infoshare": infoshare}[a.cmd]
    r = fn()
    print(json.dumps(r, default=str)[:3000])


if __name__ == "__main__":
    main()
