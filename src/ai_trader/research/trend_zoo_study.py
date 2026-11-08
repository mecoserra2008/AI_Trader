"""Trend model zoo study: leaderboard, ensemble (H-TR1) and walk-forward selection (H-TR2).

Design (RESEARCH_PROTOCOL §3b, H-X2b method): for each of the 23 symbols, market_frame(sym, freq, now(),
funding=False, oi=False); every zoo variant's signal (strategies.trend_zoo) is computed on the FULL history;
evaluate.strategy_pnl (sized at entry, risk units, 7.5 bps per side) gives per-bar net P&L, summed per UTC
date. All variants of a symbol share one evaluation window that starts on the first full UTC date after
every variant is warm (trend_zoo.lookback_bars), so variants are compared on identical days. Annualised
Sharpe = mean/std*sqrt(365) per segment (> 30 days), then averaged over symbols. Segments split at the
knowledge cutoff by P&L date.

Outputs: data/priors/trend_zoo_daily_{4h,1d}.parquet (symbol, date, variant, pnl; variant 'ensemble' =
P&L of the ensemble signal), written atomically as soon as the panel exists (H-TR3/H-TR4 consume it).

H-TR2: per symbol, each calendar month M with >= 365 days of evaluation history before M, rank the 21
variants by trailing-365-day net daily Sharpe, hold the top 3 equally (mean of their daily P&L) during M.
Compared with the ensemble on the same symbol-days: cross-symbol mean Sharpe per segment, and a paired
monthly test: d_m = mean over symbols of [mean_{days in m}(sel/sd_sel) - mean(ens/sd_ens)] * sqrt(365)
(sd per symbol and segment), whose average over months is ~ the Sharpe difference; NW t over months.

    uv run python -m ai_trader.research.trend_zoo_study --segment llm_in_sample [--record]
    uv run python -m ai_trader.research.trend_zoo_study --segment clean_holdout [--record]
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timedelta
from pathlib import Path
from statistics import NormalDist

import numpy as np
import polars as pl

from ai_trader.core.config import load
from ai_trader.core.time import now
from ai_trader.research.evaluate import knowledge_cutoff, newey_west_t, strategy_pnl
from ai_trader.research.frame import market_frame
from ai_trader.strategies.trend_zoo import lookback_bars, zoo

UNIVERSE = (
    "BTCUSDT ETHUSDT SOLUSDT XRPUSDT DOGEUSDT ADAUSDT LINKUSDT LTCUSDT BCHUSDT AVAXUSDT DOTUSDT BNBUSDT "
    "TRXUSDT NEARUSDT ATOMUSDT ETCUSDT FILUSDT UNIUSDT AAVEUSDT XLMUSDT SUIUSDT ENAUSDT HYPEUSDT"
).split()
FREQS = {"4h": 6, "1d": 1}
SEGS = ("llm_in_sample", "clean_holdout")
MIN_DAYS = 30
TOP_K = 3
TRAIL_DAYS = 365
STOP_SIGMA = 3.0  # rough R mapping: a stop ~3 bar-sigmas away (the session's 3.2% stop is ~3 sigma_4h on majors)
EULER = 0.5772156649


def panel_path(freq: str) -> Path:
    return load().root / "data" / "priors" / f"trend_zoo_daily_{freq}.parquet"


def family(v: str) -> str:
    return "ensemble" if v == "ensemble" else v.split("_")[0]


def _seg_expr(col: str = "date") -> pl.Expr:
    return pl.when(pl.col(col) < knowledge_cutoff().date()).then(pl.lit("llm_in_sample")).otherwise(pl.lit("clean_holdout")).alias("seg")


# ------------------------------------------------------------------ panel
def _trades(sig: np.ndarray, pnl: np.ndarray, ts: pl.Series) -> list[tuple]:
    """Maximal runs of a constant-sign position -> (entry date, net pnl in risk units, bars)."""
    sgn = np.sign(sig)
    out = []
    i, n = 0, len(sgn)
    while i < n:
        if sgn[i] == 0:
            i += 1
            continue
        j = i
        while j + 1 < n and sgn[j + 1] == sgn[i]:
            j += 1
        # the exit cost is charged on the bar after the run (turnover when flat/flipped)
        tot = float(pnl[i:j + 1].sum() + (pnl[j + 1] if j + 1 < n and sgn[j + 1] == 0 else 0.0))
        out.append((ts[i].date(), tot, j - i + 1))
        i = j + 1
    return out


def symbol_panel(sym: str, freq: str, as_of: datetime) -> dict | None:
    bpd = FREQS[freq]
    f = market_frame(sym, freq, as_of, funding=False, oi=False)
    lb = lookback_bars(bpd)
    if f.height < lb + 90 * bpd:
        return None
    variants = zoo(bpd)
    sigs = {name: fn(f) for name, fn in variants.items()}
    sigs["ensemble"] = pl.Series("ensemble", np.mean(np.vstack([s.to_numpy() for s in sigs.values()]), axis=0))
    start = f["event_ts"][lb].date() + timedelta(days=1)
    cost_bps = load().risk["costs"]["taker_bps"] + load().risk["costs"]["slippage_bps"]
    sn = f["sigma_next"].to_numpy()
    frames, trades = [], []
    for name, s in sigs.items():
        p = strategy_pnl(f, s)
        # turnover in signal units: |Δ leverage| x σ̂ at the bar ≈ |Δ held signal| (a full flip = 2)
        turn = p["cost"].to_numpy() * 1e4 / cost_bps * np.nan_to_num(sn)
        d = (p.with_columns(pl.Series("turn", turn), pl.col("event_ts").dt.date().alias("date"))
             .group_by("date").agg(pl.col("pnl").sum(), pl.col("gross").sum(), pl.col("cost").sum(), pl.col("turn").sum(),
                                   pl.col("active").any())
             .filter(pl.col("date") >= start).with_columns(pl.lit(sym).alias("symbol"), pl.lit(name).alias("variant")))
        frames.append(d)
        keep = (f["event_ts"].dt.date() >= start).to_numpy()
        sig_held = np.where(p["active"].to_numpy(), p["signal"].to_numpy(), 0.0)
        for dt, tot, bars in _trades(sig_held[keep], p["pnl"].to_numpy()[keep], f["event_ts"].filter(pl.Series(keep))):
            trades.append({"symbol": sym, "variant": name, "date": dt, "pnl": tot, "bars": bars})
    keep = (f["event_ts"].dt.date() >= start).to_numpy()
    names = [n for n in sigs if n != "ensemble"]
    m = np.vstack([sigs[n].to_numpy()[keep] for n in names])
    with np.errstate(invalid="ignore", divide="ignore"):
        sc = np.corrcoef(m)
    return {"symbol": sym, "daily": pl.concat(frames, how="vertical_relaxed"), "trades": pl.DataFrame(trades), "sig_corr": sc, "names": names}


def write_panel(daily: pl.DataFrame, freq: str) -> Path:
    path = panel_path(freq)
    path.parent.mkdir(parents=True, exist_ok=True)
    out = daily.select(pl.col("symbol").cast(pl.Utf8), pl.col("date").cast(pl.Date), pl.col("variant").cast(pl.Utf8),
                       pl.col("pnl").cast(pl.Float64)).sort("symbol", "variant", "date")
    tmp = path.with_suffix(".parquet.tmp")
    out.write_parquet(tmp)
    os.replace(tmp, path)
    return path


def build(freq: str, as_of: datetime) -> dict:
    per = [r for r in (symbol_panel(s, freq, as_of) for s in UNIVERSE) if r is not None]
    daily = pl.concat([r["daily"] for r in per], how="vertical_relaxed")
    write_panel(daily, freq)  # FIRST: another study waits for this file
    names = per[0]["names"]
    sc = np.nanmean(np.stack([r["sig_corr"] for r in per]), axis=0)
    return {"freq": freq, "daily": daily, "trades": pl.concat([r["trades"] for r in per], how="vertical_relaxed"),
            "names": names, "sig_corr": sc, "symbols": [r["symbol"] for r in per]}


# ------------------------------------------------------------------ statistics
def _sharpe(x: np.ndarray) -> float:
    x = x[np.isfinite(x)]
    if len(x) <= MIN_DAYS:
        return float("nan")
    sd = x.std(ddof=1)
    return float(x.mean() / sd * np.sqrt(365)) if sd > 0 else float("nan")


def _mdd(x: np.ndarray) -> float:
    """Max drawdown of the cumulative P&L in units of the series' annualised vol (scale-free)."""
    if len(x) < 2 or x.std(ddof=1) == 0:
        return float("nan")
    eq = np.cumsum(x)
    dd = np.max(np.maximum.accumulate(np.r_[0.0, eq])[1:] - eq)
    return float(dd / (x.std(ddof=1) * np.sqrt(365)))


def leaderboard(b: dict) -> dict:
    d = b["daily"].with_columns(_seg_expr())
    rows = {}
    per_sym = (d.group_by("variant", "symbol", "seg").agg(
        pl.col("pnl"), pl.col("gross"),
        ((pl.col("pnl") > 0).sum() / (pl.col("pnl") != 0).sum()).alias("hit"),
        (pl.col("turn").sum() / pl.len() * 365).alias("turn_yr"),
        pl.col("cost").sum().alias("cost_sum"), pl.col("gross").sum().alias("gross_sum"),
        pl.col("active").mean().alias("exposure")))
    port = d.group_by("variant", "seg", "date").agg(pl.col("pnl").mean()).sort("date")
    tr = b["trades"].with_columns(_seg_expr())
    for v in [*b["names"], "ensemble"]:
        rows[v] = {"family": family(v)}
        for seg in SEGS:
            ps = per_sym.filter((pl.col("variant") == v) & (pl.col("seg") == seg))
            net = np.array([_sharpe(np.asarray(x)) for x in ps["pnl"].to_list()])
            gro = np.array([_sharpe(np.asarray(x)) for x in ps["gross"].to_list()])
            px = port.filter((pl.col("variant") == v) & (pl.col("seg") == seg))["pnl"].to_numpy()
            pm, pt, pn = newey_west_t(px)
            t = tr.filter((pl.col("variant") == v) & (pl.col("seg") == seg))
            ok = np.isfinite(net)
            rows[v][seg] = {
                "sharpe": float(np.nanmean(net)) if ok.any() else float("nan"),
                "sharpe_gross": float(np.nanmean(gro)) if np.isfinite(gro).any() else float("nan"),
                "n_sym": int(ok.sum()), "frac_pos": float(np.mean(net[ok] > 0)) if ok.any() else float("nan"),
                "hit": float(ps["hit"].mean()), "turn_yr": float(ps["turn_yr"].mean()), "exposure": float(ps["exposure"].mean()),
                "cost_over_gross": float(ps["cost_sum"].sum() / ps["gross_sum"].sum()) if ps["gross_sum"].sum() > 0 else float("nan"),
                "port_sharpe": _sharpe(px), "port_t_nw": pt, "port_days": pn, "port_mdd_annvol": _mdd(px),
                "trades_per_sym_yr": float(t.height / max(1, ps.height) / (len(px) / 365)) if len(px) else float("nan"),
                "trade_mean_risk": float(t["pnl"].mean()) if t.height else float("nan"),
                "trade_bars_med": float(t["bars"].median()) if t.height else float("nan"),
            }
            rows[v][seg]["trade_mean_R_3sig"] = rows[v][seg]["trade_mean_risk"] / STOP_SIGMA
            rows[v][seg]["sharpe_cost_drag"] = rows[v][seg]["sharpe_gross"] - rows[v][seg]["sharpe"]
    return rows


def port_matrix(b: dict, seg: str) -> tuple[list[str], np.ndarray]:
    d = b["daily"].with_columns(_seg_expr()).filter(pl.col("seg") == seg)
    w = (d.group_by("variant", "date").agg(pl.col("pnl").mean()).pivot(on="variant", index="date", values="pnl").sort("date"))
    names = [*b["names"], "ensemble"]
    return names, w.select(names).to_numpy()


def deflated_sharpe(x: np.ndarray, sr_trials: np.ndarray, n_trials: float) -> dict:
    """Bailey & Lopez de Prado (2014). Per-period (daily) Sharpe; V = variance of the trials' Sharpes."""
    nd = NormalDist()
    x = x[np.isfinite(x)]
    T = len(x)
    sr = x.mean() / x.std(ddof=1)
    z = (x - x.mean()) / x.std(ddof=0)
    skew, kurt = float(np.mean(z**3)), float(np.mean(z**4))
    v = float(np.var(sr_trials, ddof=1))
    n = max(float(n_trials), 2.0)
    sr0 = np.sqrt(v) * ((1 - EULER) * nd.inv_cdf(1 - 1 / n) + EULER * nd.inv_cdf(1 - 1 / (n * np.e)))
    den = np.sqrt(max(1 - skew * sr + (kurt - 1) / 4 * sr**2, 1e-12))
    return {"sr_ann": float(sr * np.sqrt(365)), "sr0_ann": float(sr0 * np.sqrt(365)), "n_trials": float(n_trials), "T": T,
            "skew": skew, "kurt": kurt, "psr0": nd.cdf(sr * np.sqrt(T - 1) / den), "dsr": nd.cdf((sr - sr0) * np.sqrt(T - 1) / den)}


def corr_summary(b: dict) -> dict:
    names, m = port_matrix(b, "llm_in_sample")
    zoo_n = b["names"]
    k = len(zoo_n)
    c = np.corrcoef(m[:, :k], rowvar=False)
    off = c[~np.eye(k, dtype=bool)]
    lam = np.clip(np.linalg.eigvalsh(c), 0, None)[::-1]
    fam = [family(v) for v in zoo_n]
    blocks = {}
    for a in dict.fromkeys(fam):
        for bb in dict.fromkeys(fam):
            ia = [i for i, f in enumerate(fam) if f == a]
            ib = [i for i, f in enumerate(fam) if f == bb]
            vals = [c[i, j] for i in ia for j in ib if i != j]
            if vals:
                blocks[f"{a}|{bb}"] = float(np.mean(vals))
    pairs = sorted(((c[i, j], zoo_n[i], zoo_n[j]) for i in range(k) for j in range(i + 1, k)), reverse=True)
    ens = np.corrcoef(m, rowvar=False)[-1, :k]
    s = b["sig_corr"]
    soff = s[~np.eye(k, dtype=bool)]
    return {
        "pnl_corr_mean": float(off.mean()), "pnl_corr_q10_q50_q90": [float(q) for q in np.quantile(off, [0.1, 0.5, 0.9])],
        "pnl_corr_min": float(off.min()), "eig_top_share": float(lam[0] / lam.sum()),
        "n_eff_participation": float(lam.sum() ** 2 / np.sum(lam**2)),
        "n_eff_eig_gt_1": int(np.sum(lam > 1)),
        "family_blocks": blocks,
        "most_correlated": [(p[1], p[2], round(float(p[0]), 3)) for p in pairs[:5]],
        "least_correlated": [(p[1], p[2], round(float(p[0]), 3)) for p in pairs[-5:]],
        "corr_with_ensemble": {v: float(x) for v, x in zip(zoo_n, ens)},
        "signal_corr_mean": float(np.nanmean(soff)), "signal_corr_q10_q50_q90": [float(q) for q in np.nanquantile(soff, [0.1, 0.5, 0.9])],
        "pnl_corr_matrix": {"names": zoo_n, "values": np.round(c, 3).tolist()},
    }


def dsr_block(b: dict, lb: dict, extra_trials: int = 0) -> dict:
    names, m = port_matrix(b, "llm_in_sample")
    sr = np.array([np.nanmean(m[:, i]) / np.nanstd(m[:, i], ddof=1) for i in range(len(names))])
    best = max(names, key=lambda v: lb[v]["llm_in_sample"]["sharpe"])  # leaderboard rank (cross-symbol mean Sharpe)
    best_port = names[int(np.nanargmax(sr))]
    cs = corr_summary(b)
    out = {"best_by_mean_sharpe": best, "best_by_port_sharpe": best_port}
    for label, n in (("N_all_4h", len(names)), ("N_eff", cs["n_eff_participation"]), ("N_incl_other_freq", len(names) + extra_trials)):
        out[label] = deflated_sharpe(m[:, names.index(best)], sr, n)
        if best_port != best:
            out[label + "_port_best"] = deflated_sharpe(m[:, names.index(best_port)], sr, n)
    return out


# ------------------------------------------------------------------ H-TR2 walk-forward selection
def walk_forward(b: dict) -> pl.DataFrame:
    names = b["names"]
    rows = []
    for sym in b["symbols"]:
        w = (b["daily"].filter(pl.col("symbol") == sym).pivot(on="variant", index="date", values="pnl").sort("date"))
        dates = w["date"].to_numpy().astype("datetime64[D]")
        z = w.select(names).to_numpy()
        ens = w["ensemble"].to_numpy()
        first = dates[0]
        m0 = (first + np.timedelta64(TRAIL_DAYS, "D")).astype("datetime64[M]")
        if dates[0] + np.timedelta64(TRAIL_DAYS, "D") > m0.astype("datetime64[D]"):
            m0 = m0 + np.timedelta64(1, "M")
        last_m = dates[-1].astype("datetime64[M]")
        mon = m0
        while mon <= last_m:
            ms = mon.astype("datetime64[D]")
            me = (mon + np.timedelta64(1, "M")).astype("datetime64[D]")
            tr = (dates >= ms - np.timedelta64(TRAIL_DAYS, "D")) & (dates < ms)
            te = (dates >= ms) & (dates < me)
            if te.sum() and tr.sum() > MIN_DAYS:
                x = z[tr]
                sd = x.std(axis=0, ddof=1)
                trail = np.where(sd > 0, x.mean(axis=0) / np.where(sd > 0, sd, 1.0), -np.inf)
                top = np.argsort(-trail, kind="stable")[:TOP_K]
                sel = z[te][:, top].mean(axis=1)
                # rank IC of the trailing ranking vs the month's realised Sharpe (does the ranking persist?)
                y = z[te]
                ysd = y.std(axis=0, ddof=1) if te.sum() > 2 else np.zeros(len(names))
                realised = np.where(ysd > 0, y.mean(axis=0) / np.where(ysd > 0, ysd, 1.0), 0.0)
                ok = np.isfinite(trail)
                ic = _spearman(trail[ok], realised[ok]) if ok.sum() > 3 else float("nan")
                for di, s_, e_ in zip(dates[te], sel, ens[te]):
                    rows.append({"symbol": sym, "date": di.astype(object), "month": str(mon), "sel": float(s_), "ens": float(e_),
                                 "top": ",".join(names[i] for i in top), "ic": ic})
            mon = mon + np.timedelta64(1, "M")
    return pl.DataFrame(rows).with_columns(_seg_expr())


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1]) if np.std(ra) > 0 and np.std(rb) > 0 else float("nan")


def tr2_stats(wf: pl.DataFrame) -> dict:
    out = {}
    for seg in SEGS:
        s = wf.filter(pl.col("seg") == seg)
        if s.height == 0:
            out[seg] = {"n_months": 0}
            continue
        per = s.group_by("symbol").agg(pl.col("sel"), pl.col("ens"))
        sh_sel = np.array([_sharpe(np.asarray(x)) for x in per["sel"].to_list()])
        sh_ens = np.array([_sharpe(np.asarray(x)) for x in per["ens"].to_list()])
        ok = np.isfinite(sh_sel) & np.isfinite(sh_ens)
        sd = s.group_by("symbol").agg(pl.col("sel").std().alias("sd_sel"), pl.col("ens").std().alias("sd_ens"))
        z = (s.join(sd, on="symbol").filter((pl.col("sd_sel") > 0) & (pl.col("sd_ens") > 0))
             .with_columns((pl.col("sel") / pl.col("sd_sel") - pl.col("ens") / pl.col("sd_ens")).alias("dz")))
        dm = (z.group_by("month", "symbol").agg(pl.col("dz").mean()).group_by("month").agg(pl.col("dz").mean() * np.sqrt(365))
              .sort("month")["dz"].to_numpy())
        m, t_nw, n = newey_west_t(dm)
        t_plain = float(dm.mean() / (dm.std(ddof=1) / np.sqrt(len(dm)))) if len(dm) > 2 else float("nan")
        ic = s.group_by("symbol", "month").agg(pl.col("ic").first()).group_by("month").agg(pl.col("ic").mean()).sort("month")["ic"].to_numpy()
        icm, ict, _ = newey_west_t(ic)
        picks = (s.group_by("symbol", "month").agg(pl.col("top").first())["top"].str.split(",").explode().value_counts(sort=True))
        out[seg] = {
            "sel_sharpe": float(np.mean(sh_sel[ok])), "ens_sharpe": float(np.mean(sh_ens[ok])), "n_sym": int(ok.sum()),
            "frac_sym_sel_beats_ens": float(np.mean(sh_sel[ok] > sh_ens[ok])),
            "paired_monthly_diff": float(dm.mean()), "paired_t_nw": t_nw, "paired_t_plain": t_plain, "n_months": int(len(dm)),
            "rank_ic_mean": float(np.nanmean(ic)), "rank_ic_t_nw": ict,
            "top_picks": {r[0]: int(r[1]) for r in picks.head(8).iter_rows()},
        }
    return out


# ------------------------------------------------------------------ verdicts (transcribed from the registry)
def verdict_tr1(lb: dict, seg: str) -> str:
    r = lb["ensemble"][seg]
    if not r["n_sym"] or not np.isfinite(r["sharpe"]):
        return "inconclusive"
    return "supports" if r["sharpe"] > (0.3 if seg == "llm_in_sample" else 0.0) else "contradicts"


def verdict_tr2(t: dict, seg: str) -> str:
    r = t[seg]
    if r.get("n_months", 0) < 3:
        return "inconclusive"
    d = r["sel_sharpe"] - r["ens_sharpe"]
    if seg == "llm_in_sample":
        return "supports" if d > 0 and r["paired_t_nw"] > 2 else "contradicts"
    return "supports" if d > 0 and r["paired_monthly_diff"] > 0 else "contradicts"


def run(as_of: datetime | None = None, freqs: tuple[str, ...] = ("4h", "1d")) -> dict:
    as_of = as_of or now()
    out = {}
    for fq in freqs:
        b = build(fq, as_of)
        lb = leaderboard(b)
        out[fq] = {"n_symbols": len(b["symbols"]), "leaderboard": lb, "corr": corr_summary(b),
                   "dsr": dsr_block(b, lb, extra_trials=22 * (len(freqs) - 1)), "tr2": tr2_stats(walk_forward(b))}
    return out


def view(out: dict, seg: str) -> dict:
    v = {}
    for fq, o in out.items():
        lb = o["leaderboard"]
        ranked = sorted(lb, key=lambda k: -np.nan_to_num(lb[k]["llm_in_sample"]["sharpe"], nan=-9))
        v[fq] = {"n_symbols": o["n_symbols"], "rank_by_in_sample": ranked,
                 "leaderboard": {k: {"family": lb[k]["family"], seg: lb[k][seg]} for k in ranked},
                 "tr2": o["tr2"][seg], "verdicts": {"H-TR1": verdict_tr1(lb, seg), "H-TR2": verdict_tr2(o["tr2"], seg)}}
        if seg == "llm_in_sample":
            v[fq]["dsr"] = o["dsr"]
            v[fq]["corr"] = {k: x for k, x in o["corr"].items() if k != "pnl_corr_matrix"}
    return v


def record_verdicts(out: dict, seg: str) -> None:
    from ai_trader.research.hypotheses import record

    o = out["4h"]
    lb, t2 = o["leaderboard"], o["tr2"]
    data = ("bybit 4h, 23-symbol universe, market_frame(funding=False, oi=False), signals on full history, common per-symbol "
            "evaluation window after zoo warm-up; panel data/priors/trend_zoo_daily_4h.parquet")
    e = lb["ensemble"][seg]
    m1 = {"ensemble_sharpe": e["sharpe"], "n_sym": e["n_sym"], "frac_pos": e["frac_pos"], "sharpe_gross": e["sharpe_gross"],
          "port_sharpe": e["port_sharpe"], "port_t_nw": e["port_t_nw"], "turn_yr": e["turn_yr"], "n_variants": len(lb) - 1}
    if "1d" in out:
        m1["ensemble_sharpe_1d"] = out["1d"]["leaderboard"]["ensemble"][seg]["sharpe"]
    if seg == "llm_in_sample":
        m1["dsr_best_variant"] = {k: o["dsr"][k] for k in ("best_by_mean_sharpe", "N_all_4h")}
    record("H-TR1", seg, verdict_tr1(lb, seg), m1, note="mean over symbols of annualised daily net Sharpe of the equal-weight zoo-signal ensemble", data=data)
    m2 = {k: x for k, x in t2[seg].items() if k != "top_picks"}
    record("H-TR2", seg, verdict_tr2(t2, seg), m2,
           note="monthly walk-forward, per symbol top-3 of 21 variants by trailing 365d net Sharpe, equal weight; paired monthly diff of vol-normalised daily P&L", data=data)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--segment", choices=SEGS, required=True)
    ap.add_argument("--record", action="store_true")
    ap.add_argument("--json", default=None, help="also write the segment view to this file")
    a = ap.parse_args()
    out = run()
    v = view(out, a.segment)
    s = json.dumps(v, indent=1, default=str)
    print(s)
    if a.json:
        Path(a.json).write_text(s, encoding="utf-8")
    if a.record:
        record_verdicts(out, a.segment)


if __name__ == "__main__":
    main()
