"""H-PD6: trailing exits at k x sigma from the running best close for the trend family (Program 2 follow-up; agent I).

Design fixed before any result was computed (diary 2026-11-12, "H-PD6 exit-policy test: design fixed before evaluation").

Trend family
------------
The PD2 cluster F1 of research/reports/pnl_decomposition_2026-11-11.md: tsmom, donchian_breakout, ema_stack,
compression_breakout (1h) and geo_struct, geo_pivot_bo, geo_flag_bo, zoo_ensemble (4h). compression_breakout is registry
family 'volatility' but loads on +TSMOM; a robustness book without it is reported.

Baseline
--------
Identical to the strategy panel (research.strategy_panel): same frames (frames_for at the panel's as_of), same signals
(strategy_fn on the FULL history), same execution (evaluate.strategy_pnl: risk units, rebalance band 0.2, 7.5 bps per
side), same dating, warm-up and complete-date filter. The recomputed baseline is checked against the stored panel.

Trailing exit (k in {1.5, 2.5, 3.5}, fixed in the registry)
------------------------------------------------------------
* sigma_t = sigma_risk_next at close t (robust bias-corrected EWMA one-bar sigma at the strategy's own bar size, without
  the weekend shrink: the frame's stop/risk sigma), falling back to sigma_next where it is null. Known at close t.
  The distance is k * sigma_t in log price (one-bar horizon), re-read at every bar.
* A trade starts at the close of the first bar where the wrapped signal is non-zero with a new sign; best_t is the max
  close (long) / min close (short) over entry..t. At close t, if side * (log best_t - log close_t) > k * sigma_t
  (strict), the wrapped signal is 0 at t: the position is closed at close_t (the same timing as a strategy exit).
* Re-entry: after a trailing exit of side d the wrapped signal stays 0 while sign(raw) == d; the first bar with
  sign(raw) != d (flat or opposite) is the fresh signal change, from which the wrapper follows the raw signal again (an
  opposite raw signal opens the opposite trade at that close). Otherwise wrapped == raw (the strategy's own exits apply).

Evaluation
----------
Daily pnl_d per symbol-date as in the panel; sleeve = strategy x universe (mean over symbols, T needs >= 3 symbols);
family book = equal-weight mean of the available trend sleeves per date (L and T; L-only, T-only and ex-compression books
reported). Paired daily difference book_k - book_orig, Newey-West t, one-sided p (normal); Holm over the 3 k per segment
(llm_in_sample before research.knowledge_cutoff, clean_holdout on/after, pooled all dates). Verdicts (median k = 2.5):
llm_in_sample supports if diff > 0, t > 2 and Holm p < 0.05, contradicts if diff <= 0, else inconclusive;
clean_holdout supports if diff > 0, else contradicts; pooled supports if both segments support, contradicts if either
contradicts, else inconclusive. Also: the H-PD4 trade decomposition (pnl_decomposition_study.decompose) per variant on
trend-family trades, trade counts, holding times, share of trades closed by the trailing rule, annualised Sharpe.

    uv run python -m ai_trader.research.exit_policy_study build [--workers 4]
    uv run python -m ai_trader.research.exit_policy_study analyse [--record]
"""

from __future__ import annotations

import argparse
import json
import math
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timedelta
from pathlib import Path
from statistics import NormalDist

import numpy as np
import polars as pl

from ai_trader.core.config import load
from ai_trader.research.evaluate import knowledge_cutoff, newey_west_t, strategy_pnl

TREND = ("tsmom", "donchian_breakout", "ema_stack", "compression_breakout", "geo_struct", "geo_pivot_bo", "geo_flag_bo", "zoo_ensemble")
KS = (1.5, 2.5, 3.5)
K_MEDIAN = 2.5
VARIANTS = ("orig",) + tuple(f"k{k}" for k in KS)
BAR_HOURS = {"1h": 1, "4h": 4}
SEGS3 = ("llm_in_sample", "clean_holdout", "pooled")
REPORT = "exit_policy_2026-11-12"
ND = NormalDist()


# ====================================================================== the rule
def trail_sigma(frame: pl.DataFrame) -> np.ndarray:
    """sigma_risk_next (robust one-bar stop sigma known at close t), sigma_next where it is null / non-positive."""
    sn = frame["sigma_next"].cast(pl.Float64).fill_null(float("nan")).to_numpy()
    if "sigma_risk_next" not in frame.columns:
        return sn.copy()
    sr = frame["sigma_risk_next"].cast(pl.Float64).fill_null(float("nan")).to_numpy()
    return np.where(np.isfinite(sr) & (sr > 0), sr, sn)


def trailing_exit(signal: np.ndarray, close: np.ndarray, sigma: np.ndarray, k: float) -> tuple[np.ndarray, np.ndarray]:
    """Wrap a raw signal with the trailing exit. Returns (wrapped signal, stop flag per bar). Causal: the value at t
    depends only on signal, close and sigma at bars <= t."""
    s = np.nan_to_num(np.asarray(signal, dtype=float)).tolist()
    lc = np.log(np.asarray(close, dtype=float)).tolist()
    sg = np.asarray(sigma, dtype=float).tolist()
    n = len(s)
    out = [0.0] * n
    stop = [False] * n
    side, best, blocked = 0, 0.0, 0
    for t in range(n):
        st = s[t]
        d = (st > 0) - (st < 0)
        if blocked:
            if d == blocked:
                continue  # still the same-direction signal that was stopped out: no re-entry
            blocked = 0   # fresh signal change (flat or opposite)
        if d == 0:
            side = 0
            continue
        c = lc[t]
        if d != side:
            side, best = d, c  # new trade, entered at close t
        elif (side > 0 and c > best) or (side < 0 and c < best):
            best = c
        sig = sg[t]
        if sig == sig and sig > 0 and side * (best - c) > k * sig:
            stop[t] = True
            blocked, side = side, 0
            continue
        out[t] = st
    return np.asarray(out), np.asarray(stop)


# ====================================================================== per symbol (checkpointed)
def ckpt_dir() -> Path:
    return load().root / "data" / "priors" / "exit_policy_ckpt"


def daily_path() -> Path:
    return load().root / "data" / "priors" / "exit_policy_daily.parquet"


def trades_out_path() -> Path:
    return load().root / "data" / "priors" / "exit_policy_trades.parquet"


def symbol_job(sym: str, universe: str, as_of_iso: str) -> dict:
    from ai_trader.research.strategy_panel import (BPD, MIN_T_DAYS, WARM_DAYS, cost_bps, daily_from_bars, frames_for, held_leverage,
                                                   strategy_fn, trade_table, write_atomic)

    t0 = time.time()
    d_ck, t_ck = ckpt_dir() / f"{sym}_{universe}_daily.parquet", ckpt_dir() / f"{sym}_{universe}_trades.parquet"
    if d_ck.exists() and t_ck.exists():
        return {"symbol": sym, "universe": universe, "cached": True}
    as_of = datetime.fromisoformat(as_of_iso)
    f1, f4, src = frames_for(sym, universe, as_of)
    out = {"symbol": sym, "universe": universe, "f4_source": src}
    if f1.height == 0 or f4.height == 0:
        out["skip"] = "no data"
        return out
    days = (f1["event_ts"].max() - f1["event_ts"].min()).days
    if universe == "T" and days < MIN_T_DAYS:
        out["skip"] = f"{days} days < {MIN_T_DAYS}"
        return out
    start = max(f1["event_ts"].min(), f4["event_ts"].min()).date() + timedelta(days=WARM_DAYS)
    last = as_of.date()
    c_frac = cost_bps() / 1e4
    daily, trades = [], []
    for name in TREND:
        freq = "4h" if name.startswith("geo_") or name == "zoo_ensemble" else "1h"
        f = f1 if freq == "1h" else f4
        raw = strategy_fn(name)(f).to_numpy().astype(float)
        close = f["close"].to_numpy()
        tsig = trail_sigma(f)
        sn = f["sigma_next"].to_numpy()
        fr = np.nan_to_num(f["fwd_ret"].to_numpy())
        lc, lh, ll = (np.log(f[c].to_numpy()) for c in ("close", "high", "low"))
        ets = f["close_ts"]
        for var in VARIANTS:
            if var == "orig":
                sig, stop = np.nan_to_num(raw), np.zeros(len(raw), dtype=bool)
            else:
                sig, stop = trailing_exit(raw, close, tsig, float(var[1:]))
            p = strategy_pnl(f, pl.Series(name, sig))
            hs, lev = held_leverage(sig, sn)
            if not np.allclose(lev * fr, p["gross"].to_numpy(), atol=1e-12):
                raise AssertionError(f"{sym} {name} {var}: replicated leverage does not match strategy_pnl")
            dd = daily_from_bars(f, p, hs, lev)
            st_day = pl.DataFrame({"date": f["close_ts"].dt.date(), "n_stops": stop.astype(np.int32)}).group_by("date").agg(pl.col("n_stops").sum())
            dd = dd.join(st_day, on="date", how="left").filter((pl.col("date") >= start) & (pl.col("date") < last))
            daily.append(dd.with_columns(pl.lit(sym).alias("symbol"), pl.lit(universe).alias("universe"), pl.lit(name).alias("strategy"),
                                         pl.lit(var).alias("variant"), pl.lit(freq).alias("freq"), pl.lit(BPD[freq]).alias("bpd")))
            for tr in trade_table(lev, p["gross"].to_numpy(), lc, lh, ll, c_frac):
                ed = ets[tr["i"]].date()
                if ed < start or ets[tr["j"] + 1].date() >= last:
                    continue
                trades.append({"symbol": sym, "universe": universe, "strategy": name, "variant": var, "freq": freq, "bpd": BPD[freq],
                               "entry_ts": ets[tr["i"]], "entry_date": ed, "stopped": bool(stop[tr["j"] + 1]),
                               **{c: tr[c] for c in ("side", "bars", "gross", "cost", "mfe", "mae")}})
    write_atomic(pl.concat(daily, how="vertical_relaxed"), d_ck)
    write_atomic(pl.DataFrame(trades) if trades else pl.DataFrame({"symbol": []}), t_ck)
    out["secs"] = round(time.time() - t0, 1)
    return out


def build(workers: int = 4) -> dict:
    from ai_trader.research.strategy_panel import UNIV_L, UNIV_T, priors_dir, write_atomic

    meta = json.loads((priors_dir() / "strategy_panel_meta.json").read_text(encoding="utf-8"))
    as_of = meta["as_of"]  # the panel's as_of: identical point-in-time data and baseline
    ckpt_dir().mkdir(parents=True, exist_ok=True)
    jobs = [(s, "L") for s in UNIV_L] + [(s, "T") for s in UNIV_T]
    res = []
    with ProcessPoolExecutor(max_workers=min(4, workers)) as ex:
        futs = [ex.submit(symbol_job, s, u, as_of) for s, u in jobs]
        for fu in futs:
            r = fu.result()
            print(f"  {r['symbol']} ({r['universe']}) {r.get('skip', '')}{'cached' if r.get('cached') else ''} {r.get('secs', '')}", flush=True)
            res.append(r)
    ok = [r for r in res if "skip" not in r]
    daily = pl.concat([pl.read_parquet(ckpt_dir() / f"{r['symbol']}_{r['universe']}_daily.parquet") for r in ok], how="vertical_relaxed")
    s = pl.col("bpd").cast(pl.Float64).sqrt()
    cut = knowledge_cutoff().date()
    daily = daily.with_columns((pl.col("pnl") / s).alias("pnl_d"), (pl.col("gross") / s).alias("gross_d"), (pl.col("cost") / s).alias("cost_d"),
                               pl.when(pl.col("date") < cut).then(pl.lit("llm_in_sample")).otherwise(pl.lit("clean_holdout")).alias("seg"))
    write_atomic(daily, daily_path())
    tl = [pl.read_parquet(ckpt_dir() / f"{r['symbol']}_{r['universe']}_trades.parquet") for r in ok]
    trades = pl.concat([t for t in tl if t.height], how="vertical_relaxed").with_columns(
        pl.when(pl.col("entry_date") < cut).then(pl.lit("llm_in_sample")).otherwise(pl.lit("clean_holdout")).alias("seg"))
    write_atomic(trades, trades_out_path())
    return {"as_of": as_of, "symbols": {u: [r["symbol"] for r in ok if r["universe"] == u] for u in ("L", "T")},
            "skipped": {r["symbol"]: r["skip"] for r in res if "skip" in r}, "rows": daily.height, "trades": trades.height}


# ====================================================================== statistics
def seg_filter(df: pl.DataFrame, seg: str, col: str = "seg") -> pl.DataFrame:
    return df if seg == "pooled" else df.filter(pl.col(col) == seg)


def sleeve_matrix(daily: pl.DataFrame, variant: str, strategies=TREND, universes=("L", "T")) -> pl.DataFrame:
    """date x sleeve ('strategy|U') of the mean pnl_d over symbols (T sleeves need >= 3 symbols), as pnl_decomposition."""
    d = daily.filter((pl.col("variant") == variant) & pl.col("strategy").is_in(list(strategies)) & pl.col("universe").is_in(list(universes)))
    g = (d.group_by("universe", "strategy", "date").agg(pl.col("pnl_d").mean(), pl.len().alias("n_sym"))
         .filter((pl.col("universe") == "L") | (pl.col("n_sym") >= 3))
         .with_columns((pl.col("strategy") + "|" + pl.col("universe")).alias("sleeve")))
    return g.pivot(on="sleeve", index="date", values="pnl_d").sort("date")


def book(daily: pl.DataFrame, variant: str, **kw) -> pl.DataFrame:
    """Equal-weight family book: per date the mean over the available sleeves."""
    W = sleeve_matrix(daily, variant, **kw)
    cols = [c for c in W.columns if c != "date"]
    if not cols:
        return pl.DataFrame({"date": [], "book": []})
    return W.select("date", pl.mean_horizontal(cols).alias("book"))


def sharpe_ann(x: np.ndarray) -> float:
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    sd = x.std(ddof=1) if len(x) > 1 else float("nan")
    return float(x.mean() / sd * math.sqrt(365)) if len(x) >= 20 and sd > 0 else float("nan")


def holm(p: dict[str, float]) -> dict[str, float]:
    from ai_trader.research.strategy_panel import holm as _h

    return _h(p)


def paired(daily: pl.DataFrame, cut, **kw) -> dict:
    """Paired daily book differences (variant - orig) per segment."""
    base = book(daily, "orig", **kw).rename({"book": "orig"})
    for v in VARIANTS[1:]:
        base = base.join(book(daily, v, **kw).rename({"book": v}), on="date", how="inner")
    base = base.with_columns(pl.when(pl.col("date") < cut).then(pl.lit("llm_in_sample")).otherwise(pl.lit("clean_holdout")).alias("seg"))
    out = {}
    for seg in SEGS3:
        b = seg_filter(base, seg)
        o = b["orig"].to_numpy()
        r = {"days": b.height, "orig_mean": float(np.mean(o)) if len(o) else float("nan"), "orig_sharpe": sharpe_ann(o)}
        ks = {}
        for v in VARIANTS[1:]:
            x = b[v].to_numpy()
            m, t, n = newey_west_t(x - o)
            ks[v] = {"diff_mean": m, "t_nw": t, "n": n, "p_one_sided": float(1 - ND.cdf(t)) if np.isfinite(t) else float("nan"),
                     "mean": float(np.mean(x)) if len(x) else float("nan"), "sharpe": sharpe_ann(x),
                     "share_days_better": float(np.mean((x - o) > 1e-15)) if len(x) else float("nan"),
                     "share_days_worse": float(np.mean((x - o) < -1e-15)) if len(x) else float("nan"),
                     # descriptive (labelled extension): is the trail a short-convexity overlay on the trend book?
                     "corr_diff_orig": float(np.corrcoef(x - o, o)[0, 1]) if len(x) > 2 and np.std(o) > 0 and np.std(x - o) > 0 else float("nan"),
                     "skew_diff": float(((x - o - (x - o).mean()) ** 3).mean() / (x - o).std() ** 3) if len(x) > 2 and np.std(x - o) > 0 else float("nan")}
        adj = holm({v: ks[v]["p_one_sided"] for v in ks})
        for v in ks:
            ks[v]["p_holm"] = adj[v]
        r["k"] = ks
        out[seg] = r
    return out


def per_sleeve(daily: pl.DataFrame, cut) -> dict:
    out = {}
    for u in ("L", "T"):
        W0 = sleeve_matrix(daily, "orig", universes=(u,))
        for v in VARIANTS[1:]:
            W1 = sleeve_matrix(daily, v, universes=(u,))
            j = W0.join(W1, on="date", how="inner", suffix="__v")
            for c in [c for c in W0.columns if c != "date"]:
                for seg in SEGS3:
                    jj = j if seg == "pooled" else j.filter((pl.col("date") < cut) if seg == "llm_in_sample" else (pl.col("date") >= cut))
                    jj = jj.filter(pl.col(c).is_not_null() & pl.col(c + "__v").is_not_null())
                    if jj.height < 30:
                        continue
                    o, x = jj[c].to_numpy(), jj[c + "__v"].to_numpy()
                    m, t, n = newey_west_t(x - o)
                    out.setdefault(c, {}).setdefault(seg, {})[v] = {"diff_mean": m, "t_nw": t, "n": n, "sharpe_orig": sharpe_ann(o), "sharpe": sharpe_ann(x)}
    return out


def decomposition(trades: pl.DataFrame) -> dict:
    from ai_trader.research.pnl_decomposition_study import decompose

    s = pl.col("bpd").cast(pl.Float64).sqrt()
    t = trades.with_columns((pl.col("gross") / s).alias("gross_d"), (pl.col("cost") / s).alias("cost_d"), (pl.col("mfe") / s).alias("mfe_d"),
                            (pl.col("mae") / s).alias("mae_d"),
                            (pl.col("bars") * pl.when(pl.col("freq") == "4h").then(4).otherwise(1)).alias("hours"))
    out = {}
    for seg in SEGS3:
        d = seg_filter(t, seg)
        r = {}
        for v in VARIANTS:
            dv = d.filter(pl.col("variant") == v)
            if dv.height == 0:
                continue
            x = decompose(dv)
            x.update({"mean_hours": float(dv["hours"].mean()), "median_hours": float(dv["hours"].median()),
                      "share_stopped": float(dv["stopped"].mean()), "mean_net_per_trade": float((dv["gross_d"] - dv["cost_d"]).mean())})
            x["by_freq"] = {}
            for fq in ("1h", "4h"):
                df = dv.filter(pl.col("freq") == fq)
                if df.height:
                    y = decompose(df)
                    y.update({"mean_hours": float(df["hours"].mean()), "share_stopped": float(df["stopped"].mean())})
                    x["by_freq"][fq] = y
            x["by_strategy"] = {}
            for (name,), g in dv.group_by("strategy"):
                y = decompose(g)
                y.update({"mean_hours": float(g["hours"].mean()), "share_stopped": float(g["stopped"].mean())})
                x["by_strategy"][name] = y
            r[v] = x
        out[seg] = r
    return out


def baseline_check(daily: pl.DataFrame) -> dict:
    """Recomputed orig rows vs the stored strategy panel (must be identical)."""
    from ai_trader.research.strategy_panel import panel_path

    panel = pl.read_parquet(panel_path()).filter(pl.col("strategy").is_in(list(TREND))).select("symbol", "universe", "strategy", "date", pl.col("pnl_d").alias("p"))
    mine = daily.filter(pl.col("variant") == "orig").select("symbol", "universe", "strategy", "date", "pnl_d")
    j = panel.join(mine, on=["symbol", "universe", "strategy", "date"], how="full", coalesce=True)
    both = j.filter(pl.col("p").is_not_null() & pl.col("pnl_d").is_not_null())
    return {"panel_rows": panel.height, "rows": mine.height, "matched": both.height,
            "max_abs_diff": float((both["p"] - both["pnl_d"]).abs().max()) if both.height else float("nan"),
            "only_panel": int(j.filter(pl.col("pnl_d").is_null()).height), "only_mine": int(j.filter(pl.col("p").is_null()).height)}


def verdicts(main: dict) -> dict:
    km = f"k{K_MEDIAN}"
    v = {}
    a = main["llm_in_sample"]["k"][km]
    if a["diff_mean"] > 0 and a["t_nw"] > 2 and a["p_holm"] < 0.05:
        v["llm_in_sample"] = "supports"
    elif a["diff_mean"] <= 0:
        v["llm_in_sample"] = "contradicts"
    else:
        v["llm_in_sample"] = "inconclusive"
    b = main["clean_holdout"]["k"][km]
    v["clean_holdout"] = "supports" if b["diff_mean"] > 0 else "contradicts"
    if v["llm_in_sample"] == "supports" and v["clean_holdout"] == "supports":
        v["pooled"] = "supports"
    elif "contradicts" in (v["llm_in_sample"], v["clean_holdout"]):
        v["pooled"] = "contradicts"
    else:
        v["pooled"] = "inconclusive"
    return v


def analyse() -> dict:
    daily = pl.read_parquet(daily_path())
    trades = pl.read_parquet(trades_out_path())
    cut = knowledge_cutoff().date()
    out = {"baseline_check": baseline_check(daily)}
    out["main"] = paired(daily, cut)
    out["robust"] = {"L": paired(daily, cut, universes=("L",)), "T": paired(daily, cut, universes=("T",)),
                     "ex_compression": paired(daily, cut, strategies=tuple(s for s in TREND if s != "compression_breakout"))}
    out["sleeves"] = per_sleeve(daily, cut)
    out["decomposition"] = decomposition(trades)
    out["verdicts"] = verdicts(out["main"])
    out["exposure"] = {seg: {v: float(seg_filter(daily, seg).filter(pl.col("variant") == v)["exposure"].mean()) for v in VARIANTS} for seg in SEGS3}
    out["dates"] = {"first": str(daily["date"].min()), "last": str(daily["date"].max()), "cutoff": str(cut)}
    out["n_symbols"] = {u: daily.filter(pl.col("universe") == u)["symbol"].n_unique() for u in ("L", "T")}
    return out


# ====================================================================== recording and report
DATA = ("strategy-panel frames at the panel as_of (2026-11-10T23:37:59Z); universes L (23 crypto) and T (TradFi perps >= 120 d); "
        "trend family F1 (8 strategies) at 1h / 4h; pnl_d daily-sigma units; 7.5 bps per side")


def record_all(out: dict) -> list[str]:
    from ai_trader.research.hypotheses import record, results

    done = []
    have = {r["segment"] for r in results("H-PD6")}
    for seg in SEGS3:
        if seg in have:
            continue
        m = out["main"][seg]
        dec = out["decomposition"][seg]
        metrics = {"days": m["days"], "orig_sharpe": m["orig_sharpe"],
                   "k": {v: {x: m["k"][v][x] for x in ("diff_mean", "t_nw", "p_one_sided", "p_holm", "sharpe")} for v in m["k"]},
                   "median_k": K_MEDIAN,
                   "capture": {v: dec[v]["capture"] for v in dec}, "share_cost_plus_leak": {v: dec[v]["share"] for v in dec},
                   "n_trades": {v: dec[v]["n_trades"] for v in dec}, "mean_hours": {v: dec[v]["mean_hours"] for v in dec},
                   "L_only_k2.5": {x: out["robust"]["L"][seg]["k"]["k2.5"][x] for x in ("diff_mean", "t_nw")},
                   "T_only_k2.5": {x: out["robust"]["T"][seg]["k"]["k2.5"][x] for x in ("diff_mean", "t_nw")} if seg in out["robust"]["T"] and out["robust"]["T"][seg]["days"] >= 10 else None}
        record("H-PD6", seg, out["verdicts"][seg], metrics,
               note="trailing exit at k x sigma_risk_next (one-bar, strategy bar size) from the running best close since entry, close-based; re-entry "
                    "only after sign(raw signal) changes; paired daily book (equal-weight trend sleeves, L+T) diff vs own exits, NW t, Holm over 3 k; "
                    "verdict on k = 2.5", data=DATA)
        done.append(seg)
    return done


def _f(x, nd=3) -> str:
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "-"
    return f"{x:.{nd}f}" if isinstance(x, float) else str(x)


def rule_candidates(out: dict) -> str:
    """Rule candidates. The wording was chosen after reading the 2026-11-12 run (in-sample: the trail lowers net P&L at
    every k; holdout: it raises it, not significantly, while the trend book itself loses; capture of MFE unchanged
    in-sample); the numbers are filled in from `out`. Never prescribes a quantity."""
    m, dec = out["main"], out["decomposition"]
    is_, ho = m["llm_in_sample"]["k"], m["clean_holdout"]["k"]
    di, dh = dec["llm_in_sample"], dec["clean_holdout"]
    ks = VARIANTS[1:]

    def rng(seg_k, key, nd=2):
        v = [seg_k[k][key] for k in ks]
        return f"{min(v):+.{nd}f} to {max(v):+.{nd}f}"

    sh_is = ", ".join(f"{is_[k]['sharpe']:.2f}" for k in ks)
    sh_ho = ", ".join(f"{ho[k]['sharpe']:.2f}" for k in ks)
    r1_effect = (f"paired book diff per day over k = 1.5/2.5/3.5: in-sample {rng(is_, 'diff_mean', 4)} (NW t {rng(is_, 't_nw')}; Sharpe "
                 f"{m['llm_in_sample']['orig_sharpe']:.2f} -> {sh_is}); holdout {rng(ho, 'diff_mean', 4)} (t {rng(ho, 't_nw')}, Holm p "
                 f"{ho['k2.5']['p_holm']:.2f}; Sharpe {m['clean_holdout']['orig_sharpe']:.2f} -> {sh_ho}); {m['llm_in_sample']['days']} / "
                 f"{m['clean_holdout']['days']} days")
    ex = out["exposure"]["llm_in_sample"]
    r2_effect = (f"exposure in-sample {ex['orig']:.2f} -> " + "/".join(f"{ex[k]:.2f}" for k in ks) + f"; in-sample corr(diff, own-exit book) {rng(is_, 'corr_diff_orig')}, skew of diff {rng(is_, 'skew_diff')}, diff > 0 on "
                 f"{min(is_[k]['share_days_better'] for k in ks):.2f}-{max(is_[k]['share_days_better'] for k in ks):.2f} of days yet negative on "
                 f"average; holdout corr {rng(ho, 'corr_diff_orig')} while the own-exit book had Sharpe {m['clean_holdout']['orig_sharpe']:.2f}")
    r3_effect = (f"capture of MFE in-sample {di['orig']['capture']:.3f} -> " + "/".join(f"{di[k]['capture']:.3f}" for k in ks)
                 + f", holdout {dh['orig']['capture']:.3f} -> " + "/".join(f"{dh[k]['capture']:.3f}" for k in ks)
                 + f"; (C+E)/MFE in-sample {di['orig']['share']:.3f} -> " + "/".join(f"{di[k]['share']:.3f}" for k in ks)
                 + f"; sum MFE in-sample {di['orig']['D_mfe']:.0f} -> " + "/".join(f"{di[k]['D_mfe']:.0f}" for k in ks)
                 + f"; mean hold {di['orig']['mean_hours']:.0f} h -> " + "/".join(f"{di[k]['mean_hours']:.0f}" for k in ks)
                 + f" h; costs / kept gross {di['orig']['cost_over_gross_pos']:.2f} -> " + "/".join(f"{di[k]['cost_over_gross_pos']:.2f}" for k in ks))
    fam = "a trend-family position (tsmom, donchian_breakout, ema_stack, compression_breakout, geo_*, zoo_ensemble) is open on 1h or 4h bars"
    rules = [
        {"id": "R-EXIT-1", "kind": "avoid",
         "rule": "WHEN a trend-family trade is open THEN do not replace the strategy's own exit with a close-based trail of 1.5-3.5 one-bar sigma from the best close; over six years it lowered the trend book's net P&L at every k",
         "when": fam,
         "then": "keep the strategy's own exit; treat a k-sigma trail on the best close as a cost to the trend book, not an edge",
         "effect": r1_effect, "confidence": "B"},
        {"id": "R-EXIT-2", "kind": "expect",
         "rule": "WHEN a tight trailing exit is laid over the trend book THEN expect it to act mainly as an exposure cut with short convexity: small gains on most days, losses on the large trend days, so it helps only while the trend book itself is losing",
         "when": "comparing a trailed trend book with the untrailed one, day by day",
         "then": "read a trailing exit's holdout gain as a hedge of a losing trend stretch, not as better exits; do not switch it on from a few months of trend losses",
         "effect": r2_effect, "confidence": "C"},
        {"id": "R-EXIT-3", "kind": "expect",
         "rule": "WHEN trying to recover the H-PD4 exit leakage THEN expect a k-sigma trail on closes not to raise the share of favourable excursion kept in-sample; it shortens trades and shrinks the excursion pool instead",
         "when": "exit designs evaluated with the H-PD4 MFE decomposition on trend-family trades",
         "then": "rank exit rules on paired net P&L; read capture and leakage only as diagnostics, and look at time-in-trade or profit-only rules (each needs its own registration)",
         "effect": r3_effect, "confidence": "B"},
    ]
    common = {"scope": {"assets": "23 crypto perps (L) and 15 Bybit TradFi perps (T)", "horizon": "1h-4h bars, trades of hours to weeks", "sides": "both"},
              "evidence": ["H-PD6", "H-PD4", "research/reports/exit_policy_2026-11-12.md"],
              "review": "re-run when the holdout passes 180 days or the trend book turns profitable in the holdout; a sqrt(time-in-trade) distance or a profit-only trail needs its own registration",
              "status": "provisional"}
    lines = []
    for r in rules:
        r = {**r, **common}
        lines.append(f"- id: \"{r['id']}\"")
        lines.append(f"  kind: \"{r['kind']}\"")
        for key in ("rule", "when", "then"):
            lines.append(f"  {key}: \"{r[key]}\"")
        lines.append("  scope:")
        for key in ("assets", "horizon", "sides"):
            lines.append(f"    {key}: \"{r['scope'][key]}\"")
        lines.append(f"  effect: \"{r['effect']}\"")
        lines.append(f"  confidence: \"{r['confidence']}\"")
        lines.append("  evidence: [" + ", ".join(f"\"{e}\"" for e in r["evidence"]) + "]")
        lines.append(f"  review: \"{r['review']}\"")
        lines.append(f"  status: \"{r['status']}\"")
    return "\n".join(lines)


def headline(out: dict) -> list[str]:
    m, dec = out["main"], out["decomposition"]
    is_, ho = m["llm_in_sample"]["k"], m["clean_holdout"]["k"]
    di, dh = dec["llm_in_sample"], dec["clean_holdout"]
    ks = VARIANTS[1:]
    return [
        "- In-sample the trail lowers the trend book's net P&L at every k: " + ", ".join(f"k {k[1:]} {is_[k]['diff_mean']:+.4f}/day (t {is_[k]['t_nw']:.2f})" for k in ks)
        + f"; Sharpe {m['llm_in_sample']['orig_sharpe']:.2f} -> " + " / ".join(f"{is_[k]['sharpe']:.2f}" for k in ks) + ".",
        "- In the holdout it raises it, not significantly: " + ", ".join(f"k {k[1:]} {ho[k]['diff_mean']:+.4f} (t {ho[k]['t_nw']:.2f})" for k in ks)
        + f", while the own-exit trend book itself had Sharpe {m['clean_holdout']['orig_sharpe']:.2f}; the trail mostly cuts exposure to a losing book.",
        "- Decomposition: the share of the favourable excursion kept barely moves in-sample (capture " + f"{di['orig']['capture']:.3f} -> "
        + "/".join(f"{di[k]['capture']:.3f}" for k in ks) + f"; holdout {dh['orig']['capture']:.3f} -> " + "/".join(f"{dh[k]['capture']:.3f}" for k in ks)
        + "). The trail shortens trades (mean hold " + f"{di['orig']['mean_hours']:.0f} h -> " + "/".join(f"{di[k]['mean_hours']:.0f}" for k in ks)
        + " h) and shrinks the excursion pool (sum MFE " + f"{di['orig']['D_mfe']:.0f} -> " + "/".join(f"{di[k]['D_mfe']:.0f}" for k in ks)
        + "), so exit leakage falls in absolute terms but not as a share; costs per unit of kept gross rise ("
        + f"{di['orig']['cost_over_gross_pos']:.2f} -> " + "/".join(f"{di[k]['cost_over_gross_pos']:.2f}" for k in ks) + ").",
        "- Mean exposure (share of bars in a position, trend symbol-days): in-sample " + f"{out['exposure']['llm_in_sample']['orig']:.2f} -> "
        + "/".join(f"{out['exposure']['llm_in_sample'][k]:.2f}" for k in ks) + ", holdout " + f"{out['exposure']['clean_holdout']['orig']:.2f} -> "
        + "/".join(f"{out['exposure']['clean_holdout'][k]:.2f}" for k in ks) + "; the daily difference is almost the negative of the own-exit book "
        + f"(corr {is_['k2.5']['corr_diff_orig']:.2f} in-sample at k 2.5), so the paired test mostly asks whether the trend book made money in that segment.",
        "- Trade counts are almost unchanged (" + f"{di['orig']['n_trades']} -> " + "/".join(str(di[k]['n_trades']) for k in ks)
        + " in-sample) because re-entry needs a fresh signal change; " + "/".join(f"{di[k]['share_stopped']:.0%}" for k in ks)
        + " of trades end on the trail.",
    ]


def write_report(out: dict) -> tuple[Path, Path]:
    rep = load().root / "research" / "reports"
    rep.mkdir(parents=True, exist_ok=True)
    jp, mp = rep / f"{REPORT}.json", rep / f"{REPORT}.md"
    rules = rule_candidates(out)
    out["rule_candidates_yaml"] = rules
    jp.write_text(json.dumps(out, indent=1, default=str), encoding="utf-8")
    V, m, dec, bc = out["verdicts"], out["main"], out["decomposition"], out["baseline_check"]
    L = [f"# Exit policies for the trend family (H-PD6) — {REPORT[-10:]}", "",
         "Agent I, Program 2 follow-up. Design fixed in the docstring of `ai_trader.research.exit_policy_study` and in the diary "
         "(2026-11-12, 'H-PD6 exit-policy test: design fixed before evaluation') before any result was computed. Units: pnl_d (daily-sigma units). "
         f"Segments split at {out['dates']['cutoff']}; dates {out['dates']['first']} .. {out['dates']['last']}; symbols L {out['n_symbols']['L']}, T {out['n_symbols']['T']}.", "",
         "Rule: at each close, exit if the close has retraced more than k x sigma_t from the best close since entry (sigma_t = sigma_risk_next, "
         "the robust one-bar stop sigma at the strategy's bar size, known at that close); after such an exit, no re-entry until the raw signal "
         "turns flat or flips. k in {1.5, 2.5, 3.5} (registered; no other value was run). Costs 7.5 bps per side (evaluate.strategy_pnl).", "",
         f"Baseline check vs the stored strategy panel: {bc['matched']} of {bc['panel_rows']} trend rows matched, max |diff| {bc['max_abs_diff']:.2e}, "
         f"only-panel {bc['only_panel']}, only-recomputed {bc['only_mine']}.", "",
         "## Verdicts", "",
         f"- llm_in_sample: **{V['llm_in_sample']}**; clean_holdout: **{V['clean_holdout']}**; pooled: **{V['pooled']}** (registered test on k = 2.5).",
         *headline(out), "",
         "## Paired daily book difference (trailing - own exits), equal-weight trend sleeves L+T", "",
         "corr and skew (labelled extension, descriptive): correlation of the daily difference with the own-exit book, and the skewness of the difference.", "",
         "| segment | days | k | diff/day | NW t | Holm p (one-sided) | Sharpe own -> trail | days better / worse | corr(diff, own) | skew(diff) |", "|---|---|---|---|---|---|---|---|---|---|"]
    for seg in SEGS3:
        r = m[seg]
        for v in VARIANTS[1:]:
            x = r["k"][v]
            L.append(f"| {seg} | {r['days']} | {v[1:]} | {x['diff_mean']:+.4f} | {_f(x['t_nw'], 2)} | {_f(x['p_holm'])} | {_f(r['orig_sharpe'], 2)} -> {_f(x['sharpe'], 2)} | "
                     f"{x['share_days_better']:.2f} / {x['share_days_worse']:.2f} | {_f(x['corr_diff_orig'], 2)} | {_f(x['skew_diff'], 2)} |")
    L += ["", "Robustness books (diff/day, NW t):", "", "| book | segment | days | k1.5 | k2.5 | k3.5 | Sharpe own |", "|---|---|---|---|---|---|---|"]
    for name, rb in out["robust"].items():
        for seg in SEGS3:
            r = rb[seg]
            if r["days"] < 10:
                continue
            cells = " | ".join(f"{r['k'][v]['diff_mean']:+.4f} ({_f(r['k'][v]['t_nw'], 1)})" for v in VARIANTS[1:])
            L.append(f"| {name} | {seg} | {r['days']} | {cells} | {_f(r['orig_sharpe'], 2)} |")
    L += ["", "## H-PD4 decomposition on trend-family trades, by variant", "",
          "| segment | variant | trades | sum MFE | kept | exit leak | dir loss | costs | net | (C+E)/MFE | capture | hit gross | mean / median hold (h) | closed by trail |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for seg in SEGS3:
        for v in VARIANTS:
            x = dec[seg].get(v)
            if not x:
                continue
            L.append(f"| {seg} | {v} | {x['n_trades']} | {x['D_mfe']:.1f} | {x['kept']:.1f} | {x['exit_leak']:.1f} | {x['dir_loss']:.1f} | {x['costs']:.1f} | {x['net']:.1f} | "
                     f"{x['share']:.3f} | {x['capture']:.3f} | {x['hit_gross']:.3f} | {x['mean_hours']:.1f} / {x['median_hours']:.0f} | {x['share_stopped']:.2f} |")
    L += ["", "By strategy (llm_in_sample / clean_holdout): net (sum pnl_d over trades), capture, mean hold (h), trades:", "",
          "| strategy | variant | IS net | IS capture | IS hold h | IS trades | HO net | HO capture | HO hold h | HO trades |", "|---|---|---|---|---|---|---|---|---|---|"]
    for s in TREND:
        for v in VARIANTS:
            a = dec["llm_in_sample"].get(v, {}).get("by_strategy", {}).get(s)
            b = dec["clean_holdout"].get(v, {}).get("by_strategy", {}).get(s)
            if not a and not b:
                continue
            fa = (f"{a['net']:.1f} | {a['capture']:.3f} | {a['mean_hours']:.0f} | {a['n_trades']}" if a else "- | - | - | -")
            fb = (f"{b['net']:.1f} | {b['capture']:.3f} | {b['mean_hours']:.0f} | {b['n_trades']}" if b else "- | - | - | -")
            L.append(f"| {s} | {v} | {fa} | {fb} |")
    L += ["", "## Per sleeve (diff/day x100, NW t)", "", "| sleeve | segment | k1.5 | k2.5 | k3.5 | Sharpe own -> k2.5 |", "|---|---|---|---|---|---|"]
    for sl in sorted(out["sleeves"]):
        for seg in ("llm_in_sample", "clean_holdout"):
            r = out["sleeves"][sl].get(seg)
            if not r:
                continue
            cells = " | ".join(f"{100 * r[v]['diff_mean']:+.2f} ({_f(r[v]['t_nw'], 1)})" if v in r else "-" for v in VARIANTS[1:])
            k25 = r.get("k2.5", {})
            L.append(f"| {sl} | {seg} | {cells} | {_f(k25.get('sharpe_orig'), 2)} -> {_f(k25.get('sharpe'), 2)} |")
    L += ["", "## Caveats", "",
          "- Close-based trail: no intrabar stop orders, so fills are at the bar close (1h or 4h) after the retracement; a resting stop would exit earlier and at a different price.",
          "- The distance is one-bar sigma, not scaled with time in trade; on 1h bars a 1.5-3.5 sigma retracement is reached within hours, so the rule mostly truncates trades early.",
          "- Re-entry needs a sign change of the raw signal; continuous signals (tsmom, ema_stack conviction) rarely pass through zero, so after a trailing exit those strategies can stay flat for a long time (lower exposure).",
          "- MFE is a bar high/low upper bound (as in H-PD4); capture and leakage are diagnostics, the test is paired net P&L.",
          "- The holdout is short (about 4.5 months) and T sleeves exist almost only in the holdout.", "",
          "## Rule candidates", "", "```yaml", rules, "```", ""]
    mp.write_text("\n".join(L), encoding="utf-8")
    return jp, mp


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--workers", type=int, default=4)
    a = sub.add_parser("analyse")
    a.add_argument("--record", action="store_true")
    args = ap.parse_args()
    if args.cmd == "build":
        print(json.dumps(build(args.workers), indent=1, default=str))
    else:
        out = analyse()
        jp, mp = write_report(out)
        print(json.dumps({"verdicts": out["verdicts"], "baseline_check": out["baseline_check"]}, indent=1, default=str))
        print(jp, mp)
        if args.record:
            print("recorded:", record_all(out))


if __name__ == "__main__":
    main()
