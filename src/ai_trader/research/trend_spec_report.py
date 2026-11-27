"""WS16 analysis: verdicts, paired tests, multiplicity, dossier tables, synthetic null, selection, report numbers.

    PYTHONIOENCODING=utf-8 uv run python -m ai_trader.research.trend_spec analyse [--record]

Reads data/research/ws16/{trades,trades_extra,synth,dossier}/ and writes results.json (+ the H-WS16 records with --record).
"""

from __future__ import annotations

import json
import math
from datetime import UTC, date, datetime
from typing import Any

import numpy as np
import polars as pl

from ai_trader.research import stress as S
from ai_trader.research import trend_spec as T

CUT = date(2026, 7, 1)
SEGS = ("llm_in_sample", "clean_holdout")
MOD_FAMILIES = ("threshold", "stop", "exit", "timing", "order", "filter", "sizing")


def _load(sub: str) -> pl.DataFrame:
    d = T.OUT / sub
    fs = sorted(d.glob("*.parquet")) if d.exists() else []
    return pl.concat([pl.read_parquet(f) for f in fs], how="vertical_relaxed") if fs else pl.DataFrame()


def r_of(t: pl.DataFrame, sp: T.Spec | None, costs: S.Costs | None = None) -> np.ndarray:
    if t.height == 0:
        return np.zeros(0)
    r = S.r_net(t, costs)
    if sp is not None and sp.conviction:
        s = t["s_entry"].to_numpy().astype(float)
        seg = t["seg"].to_numpy()
        w = np.ones(len(s))
        for g in SEGS:
            m = seg == g
            if m.any() and s[m].mean() > 0:
                w[m] = s[m] / s[m].mean()
        r = r * w
    return r


def hold_block(t: pl.DataFrame) -> float:
    """Red-team RT-STRESS-1: bootstrap block = max(5, p90 holding period in days of the closed trades)."""
    if t.height < 5:
        return S.BLOCK_DAYS
    h = (t["exit_ts"] - t["entry_ts"]).dt.total_seconds().to_numpy() / 86400
    return float(max(S.BLOCK_DAYS, np.quantile(h, 0.9)))


def mean_ci_b(t: pl.DataFrame, r: np.ndarray, block: float, n_boot: int = S.N_BOOT) -> tuple[float, float, float, float]:
    """stress.mean_ci with a chosen block; also the one-sided bootstrap P(mean <= 0)."""
    dd = S.daily(t, r)
    n, s = dd["n"].to_numpy().astype(float), dd["R"].to_numpy().astype(float)
    if len(n) < 3:
        return float(np.mean(r)), float("nan"), float("nan"), float("nan")
    idx = T.boot_idx(len(n), n_boot, block)
    ns, ss = n[idx].sum(axis=1), s[idx].sum(axis=1)
    m = ss[ns > 0] / ns[ns > 0]
    return float(np.mean(r)), float(np.quantile(m, 0.025)), float(np.quantile(m, 0.975)), float((m <= 0).mean())


def m_(t: pl.DataFrame, sp: T.Spec | None, ci: bool = True, costs: S.Costs | None = None, block: float | None = None,
       closed_only: bool = True) -> dict:
    if t.height == 0:
        return {"n": 0}
    r = r_of(t, sp, costs)
    out = S.metrics(t, r, ci=False, closed_only=closed_only)   # stress.metrics drops open-at-end trades by default since D7b
    if ci:
        b = block if block is not None else hold_block(t)
        out["mean"], out["lo"], out["hi"], out["p_le0"] = mean_ci_b(t, r, b)
        out["block_days"] = b
    return out


def engine(t: pl.DataFrame, sp: T.Spec | None) -> tuple[pl.DataFrame, dict]:
    if t.height == 0:
        return t, {"n": 0}
    acc, info = S.live_engine(t, r_of(t, sp), S.LiveLimits.from_config())
    if acc.height == 0:
        return acc, {"n": 0}
    es = S.engine_stats(acc, info)
    rr = acc["usd"].to_numpy() / 1.30
    es["R"] = S.metrics(acc, rr, ci=True)
    days = (acc["exit_ts"].max() - acc["entry_ts"].min()).total_seconds() / 86400
    es["trades_per_month"] = acc.height / max(days / 30.44, 1e-9)
    es["R_per_month"] = float(rr.sum()) / max(days / 30.44, 1e-9)
    # one-year drawdown distribution (stationary bootstrap of engine days)
    dly = S.daily(acc, acc["usd"].to_numpy(), by="exit_ts")["R"].to_numpy()
    if len(dly) > 400:
        idx = T.boot_idx(len(dly), 2000, hold_block(acc))[:, :365]
        paths = np.cumsum(dly[idx], axis=1)
        dd = (paths - np.maximum.accumulate(np.concatenate([np.zeros((paths.shape[0], 1)), paths], axis=1), axis=1)[:, 1:]).min(axis=1)
        es["dd1y_usd_median"], es["dd1y_usd_q05"] = float(np.median(dd)), float(np.quantile(dd, 0.05))
        es["p_1y_negative"] = float((paths[:, -1] < 0).mean())
        es["usd_1y_median"] = float(np.median(paths[:, -1]))
    return acc, es


def crisis_min(acc: pl.DataFrame, wins: dict, before: date | None = None) -> float | None:
    if acc.height == 0:
        return None
    rr = acc["usd"].to_numpy() / 1.30
    sums = []
    for w, (a, b) in wins.items():
        if w.startswith("2020-03") or (before and b >= before):
            continue
        st = S.window_stats(acc, rr, a, b)
        if st.get("n"):
            sums.append(st["sum"])
    return float(min(sums)) if sums else None


def cliff(P: pl.DataFrame, name: str, base_d8: pl.DataFrame, sp: T.Spec, seg: str | None = None) -> dict:
    """G1: base D8 mean > 0 and some +-20% one-at-a-time perturbation <= 0."""
    if seg:
        base_d8 = base_d8.filter(pl.col("seg") == seg)
    b = float(r_of(base_d8, sp).mean()) if base_d8.height else float("nan")
    pv = P.filter(pl.col("variant").str.starts_with(name + "|"))
    if seg:
        pv = pv.filter(pl.col("seg") == seg)
    surf = {}
    for v in pv["variant"].unique().to_list():
        sub = pv.filter(pl.col("variant") == v)
        surf[v.split("|", 1)[1]] = float(r_of(sub, sp).mean()) if sub.height else float("nan")
    worst = min(surf.values()) if surf else float("nan")
    return {"base_d8": b, "surface": surf, "worst": worst, "cliff": (bool(b > 0 and worst <= 0) if np.isfinite(b) and surf else None)}


def two_sample_diff(t: pl.DataFrame, r: np.ndarray, mask: np.ndarray, days: np.ndarray, idx: np.ndarray) -> dict:
    """mean(r | mask) - mean(r | ~mask) with the day bootstrap."""
    n1, s1 = T.day_grid(t.filter(pl.Series(mask)), r[mask], days)
    n0, s0 = T.day_grid(t.filter(pl.Series(~mask)), r[~mask], days)
    d = s1.sum() / max(n1.sum(), 1) - s0.sum() / max(n0.sum(), 1)
    b = s1[idx].sum(1) / np.maximum(n1[idx].sum(1), 1) - s0[idx].sum(1) / np.maximum(n0[idx].sum(1), 1)
    return {"diff": float(d), "lo": float(np.quantile(b, 0.025)), "hi": float(np.quantile(b, 0.975)), "n1": int(n1.sum()), "n0": int(n0.sum())}


def capacity() -> dict:
    from ai_trader.storage.lake import Lake

    lake = Lake()
    out = {}
    for sym in T._universe():
        d = lake.bars(sym, "1d", T.AS_OF).select("event_ts", "close", "turnover").sort("event_ts")
        if d.height < 60:
            continue
        r = np.diff(np.log(d["close"].to_numpy().astype(float)))
        sig = np.sqrt(pl.Series(np.r_[np.nan, r] ** 2).fill_nan(None).ewm_mean(alpha=0.06, ignore_nulls=True).to_numpy())
        tv = d["turnover"].to_numpy().astype(float)
        med = pl.Series(tv).rolling_median(30, min_samples=20).shift(1).to_numpy()
        ins = (d["event_ts"].dt.date() < CUT).to_numpy() & np.isfinite(med) & np.isfinite(sig)
        if not ins.any():
            continue
        v05 = float(np.quantile(med[ins], 0.05))
        s_med = float(np.median(sig[ins]))
        out[sym] = {"turnover_q05_usd": v05, "sigma_d": s_med, "impact_bp_200": 1e4 * s_med * math.sqrt(200 / v05),
                    "impact_bp_50": 1e4 * s_med * math.sqrt(50 / v05)}
    return out


def dossier_tables() -> dict:
    fs = sorted((T.OUT / "dossier").glob("panel_*.parquet"))
    if not fs:
        return {}
    P = pl.concat([pl.read_parquet(f) for f in fs], how="vertical_relaxed").with_columns(
        pl.when(pl.col("close_ts").dt.date() < CUT).then(pl.lit("llm_in_sample")).otherwise(pl.lit("clean_holdout")).alias("seg"),
        pl.col("close_ts").dt.date().alias("d"))
    out: dict[str, Any] = {"sign_vs_tanh": {}, "ic": {}, "ic_by_year": {}}
    for seg in SEGS:
        p = P.filter(pl.col("seg") == seg)
        y = p["y7"].to_numpy().astype(float)
        day = p["d"].to_numpy().astype("datetime64[D]").astype(np.int64)
        for k in (7, 14, 30):
            z = p[f"z{k}"].to_numpy().astype(float)
            X = np.c_[np.ones(len(z)), np.sign(z), np.tanh(z)]
            out["sign_vs_tanh"][f"{seg}|{k}"] = T.dk_ols(X, y, day, lag=7)
            X1 = np.c_[np.ones(len(z)), np.tanh(z)]
            out["sign_vs_tanh"][f"{seg}|{k}|tanh_only"] = T.dk_ols(X1, y, day, lag=7)
            X2 = np.c_[np.ones(len(z)), np.sign(z)]
            out["sign_vs_tanh"][f"{seg}|{k}|sign_only"] = T.dk_ols(X2, y, day, lag=7)
        for sname in ("zoo_ew", "tsmom_30d", "ema_4_16_64", "donch_30", "slope_30d", "cfg_ema_stack", "kalman_llt"):
            s = p[sname].to_numpy().astype(float)
            ok = np.isfinite(s) & np.isfinite(y) & (s != 0)
            if ok.sum() < 100:
                continue
            sd_s, sd_y = s[ok].std(), y[ok].std()
            r = T.dk_ols(np.c_[np.ones(ok.sum()), (s[ok] - s[ok].mean()) / sd_s], (y[ok] - y[ok].mean()) / sd_y, day[ok], lag=7)
            out["ic"][f"{seg}|{sname}"] = {"ic": r["b"][1], "t": r["t"][1], "n": r["n"], "days": r["days"]}
    for yr in sorted(P["close_ts"].dt.year().unique().to_list()):
        p = P.filter(pl.col("close_ts").dt.year() == yr)
        s, y = p["zoo_ew"].to_numpy().astype(float), p["y7"].to_numpy().astype(float)
        ok = np.isfinite(s) & np.isfinite(y) & (s != 0)
        if ok.sum() > 200:
            day = p["d"].to_numpy().astype("datetime64[D]").astype(np.int64)[ok]
            r = T.dk_ols(np.c_[np.ones(ok.sum()), (s[ok] - s[ok].mean()) / s[ok].std()], (y[ok] - y[ok].mean()) / y[ok].std(), day, lag=7)
            out["ic_by_year"][str(yr)] = {"ic": r["b"][1], "t": r["t"][1], "n": r["n"]}
    # binned response of y7 to zoo_ew (linearity / threshold check), in-sample
    p = P.filter(pl.col("seg") == "llm_in_sample")
    s, y = p["zoo_ew"].to_numpy().astype(float), p["y7"].to_numpy().astype(float)
    ok = np.isfinite(s) & np.isfinite(y)
    bins = [-1.0, -0.6, -0.3, -0.1, 0.1, 0.3, 0.6, 1.0]
    resp = []
    for a, b in zip(bins[:-1], bins[1:], strict=True):
        m = ok & (s >= a) & (s < b if b < 1 else s <= b)
        resp.append({"bin": f"[{a:+.1f},{b:+.1f})", "n": int(m.sum()), "mean_y7": float(y[m].mean()) if m.any() else float("nan")})
    out["binned_zoo_ew"] = resp
    props = json.loads((T.OUT / "dossier_props.json").read_text(encoding="utf-8")) if (T.OUT / "dossier_props.json").exists() else []
    pr = [x["props"] for x in props if isinstance(x, dict) and "props" in x]
    if pr:
        keys = [k for k in pr[0] if isinstance(pr[0][k], int | float)]
        out["target_props_median"] = {k: float(np.nanmedian([x.get(k, np.nan) for x in pr])) for k in keys}
        out["target_props_share"] = {"adf_reject_5pct": float(np.mean([x.get("adf_p", 1) < 0.05 for x in pr])),
                                     "kpss_not_reject_5pct": float(np.mean([x.get("kpss_p", 0) > 0.05 for x in pr])),
                                     "lb_levels_reject_42": float(np.mean([x.get("lb_p_42", 1) < 0.05 for x in pr])),
                                     "lb_squares_reject_42": float(np.mean([x.get("lb2_p_42", 1) < 0.05 for x in pr])),
                                     "arch_reject": float(np.mean([x.get("arch_lm_p", 1) < 0.05 for x in pr]))}
        out["z2_by_vol_tercile_median"] = [float(np.nanmedian([x["z2_by_vol_tercile"][i] for x in pr])) for i in range(3)]
    return out


def synth_tables(real_pooled: dict[str, float]) -> dict:
    out: dict[str, Any] = {}
    fs = sorted((T.OUT / "synth").glob("null_*.parquet"))
    if fs:
        N = pl.concat([pl.read_parquet(f) for f in fs], how="diagonal_relaxed")
        for (gen, spec), g in N.group_by(["gen", "spec"]):
            rep = g.group_by("rep").agg(pl.col("n").sum(), pl.col("sum").sum()).filter(pl.col("n") > 0)
            m = (rep["sum"] / rep["n"]).to_numpy()
            real = real_pooled.get(spec, float("nan"))
            out[f"{gen}|{spec}"] = {"reps": len(m), "null_mean": float(np.mean(m)), "null_q50": float(np.median(m)), "null_q95": float(np.quantile(m, 0.95)),
                                    "real": real, "p_mc": float((1 + (m >= real).sum()) / (1 + len(m))), "block": float(g["block"].drop_nulls().mean()) if "block" in g.columns and g["block"].drop_nulls().len() else None}
    fp = T.OUT / "facts.json"
    if fp.exists():
        F = json.loads(fp.read_text(encoding="utf-8"))
        facts = {}
        fr, fsb, fg, fg9 = [], [], [], []
        for x in F:
            if "real" not in x:
                continue
            facts[x["src"]] = {"real": x["real"], "sbb": x["sbb"], "garch": x["garch"], "garch99": x.get("garch99"), "block": x.get("block"), "pw": x.get("pw"),
                               "garch_params": x.get("garch_params"), "garch99_params": x.get("garch99_params")}
            fr += x["feat_real"]
            fsb += x["feat_sbb"]
            fg += x["feat_garch"]
            fg9 += x.get("feat_garch99", [])
        out["facts"] = facts
        if fr:
            out["auc_sbb"] = T.discriminator_auc(np.array(fr), np.array(fsb))
            ok = [row for row in fg if np.all(np.isfinite(row))]
            out["auc_garch"] = T.discriminator_auc(np.array(fr), np.array(ok)) if len(ok) > 20 else None
            if fg9:
                out["auc_garch99"] = T.discriminator_auc(np.array(fr), np.array([row for row in fg9 if np.all(np.isfinite(row))]))
    return out


def analyse(record: bool = False) -> dict:
    sp_all = T.specs()
    A = _load("trades")
    X = _load("trades_extra")
    if X.height:
        A = pl.concat([A, X], how="vertical_relaxed")
    OPEN = A.filter((pl.col("why") == "end") & ~pl.col("variant").str.contains(r"\|") & (pl.col("exec") == "base"))
    ALL_B = A.filter(~pl.col("variant").str.contains(r"\|") & (pl.col("exec") == "base"))
    A = A.filter(pl.col("why") != "end")        # RT-STRESS-2: verdicts on CLOSED trades only
    base_rows = A.filter(~pl.col("variant").str.contains(r"\|"))
    P = A.filter(pl.col("variant").str.contains(r"\|"))
    wins = S.crisis_windows()
    B = base_rows.filter(pl.col("exec") == "base")
    L5 = base_rows.filter(pl.col("exec") == "lat5")
    alldays = np.arange(np.datetime64(B["entry_ts"].min().date()), np.datetime64(B["entry_ts"].max().date()) + 1)
    seg_days = {"llm_in_sample": alldays[alldays < np.datetime64(CUT)], "clean_holdout": alldays[alldays >= np.datetime64(CUT)]}
    blocks = {n: hold_block(B.filter((pl.col("variant") == n) & (pl.col("seg") == "llm_in_sample"))) for n in B["variant"].unique().to_list()}
    common_block = float(max(blocks.values()))
    idx = {g: T.boot_idx(len(seg_days[g]), S.N_BOOT, common_block) for g in SEGS}
    res: dict[str, Any] = {}
    grids: dict[str, dict[str, tuple]] = {g: {} for g in SEGS}
    grids5: dict[str, dict[str, tuple]] = {g: {} for g in SEGS}       # order family window (entries >= 2022)
    d5 = np.datetime64(T.FIVE_MIN_START.date())
    for name, sp in sp_all.items():
        tb = B.filter(pl.col("variant") == name)
        if tb.height == 0:
            continue
        R: dict[str, Any] = {"family": sp.family, "data_informed": sp.data_informed, "spec": {k: v for k, v in sp.__dict__.items()}}
        bl = blocks.get(name, S.BLOCK_DAYS)
        R["block_days"] = bl
        R["base"] = m_(tb, sp, block=bl)
        R["segments"] = {g: m_(tb.filter(pl.col("seg") == g), sp, block=bl) for g in SEGS}
        R["closed"] = R["segments"]
        ob = OPEN.filter(pl.col("variant") == name)
        ab = ALL_B.filter(pl.col("variant") == name)
        R["open_at_end"] = {"n": ob.height, "sum_R": float(r_of(ob, sp).sum()) if ob.height else 0.0,
                            "incl_open": {g: m_(ab.filter(pl.col("seg") == g), sp, ci=False, closed_only=False).get("mean") for g in SEGS}}
        R["costs"] = {k: m_(tb, sp, ci=False, costs=S.Costs(fee_mult=fm, extra_bps=x, drought=dr)) for k, (fm, x, dr) in S.COST_VARIANTS.items()}
        R["lat5"] = m_(L5.filter(pl.col("variant") == name), sp, ci=False)
        acc, es = engine(tb, sp)
        R["engine"] = es
        R["crisis_engine_min"] = crisis_min(acc, wins)
        d8 = tb.filter(pl.col("symbol").is_in(list(S.D8)))
        R["cliff"] = cliff(P, name, d8, sp)
        R["cliff_is"] = cliff(P, name, d8, sp, "llm_in_sample")
        v, fired = S.verdict({"base": R["base"], "fees15": R["costs"]["fees_x1.5"], "slip5": R["costs"]["slip+5"], "lat5": R["lat5"],
                              "holdout": R["segments"]["clean_holdout"], "cliff": R["cliff"]["cliff"], "crisis_engine_min": R["crisis_engine_min"]})
        R["verdict"], R["fired"] = v, fired
        tis = tb.filter(pl.col("seg") == "llm_in_sample")
        acc_is, _ = engine(tis, sp)
        vi, fi = S.verdict({"base": R["segments"]["llm_in_sample"], "fees15": m_(tis, sp, ci=False, costs=S.Costs(fee_mult=1.5)),
                            "slip5": m_(tis, sp, ci=False, costs=S.Costs(extra_bps=5.0)),
                            "lat5": m_(L5.filter((pl.col("variant") == name) & (pl.col("seg") == "llm_in_sample")), sp, ci=False),
                            "holdout": None, "cliff": R["cliff_is"]["cliff"], "crisis_engine_min": crisis_min(acc_is, wins, CUT)})
        R["verdict_is"], R["fired_is"] = vi, fi
        R["cells"] = {c: m_(tb.filter(pl.col("cell") == c), sp, ci=False).get("mean") for c in [f"R{i}" for i in range(1, 10)]}
        R["weekend"] = {"weekend": m_(tb.filter(pl.col("weekend")), sp, ci=False).get("mean"), "weekday": m_(tb.filter(~pl.col("weekend")), sp, ci=False).get("mean")}
        res[name] = R
        for g in SEGS:
            sub = tb.filter(pl.col("seg") == g)
            grids[g][name] = T.day_grid(sub, r_of(sub, sp), seg_days[g])
            sub5 = sub.filter(pl.col("entry_ts").dt.date() >= T.FIVE_MIN_START.date())
            grids5[g][name] = T.day_grid(sub5, r_of(sub5, sp), seg_days[g])
        print(f"{name:18s} n={R['base']['n']:5d} mean={R['base'].get('mean', float('nan')):+.3f} [{R['base'].get('lo', float('nan')):+.3f},{R['base'].get('hi', float('nan')):+.3f}] "
              f"IS {R['segments']['llm_in_sample'].get('mean', float('nan')):+.3f} HO {R['segments']['clean_holdout'].get('mean', float('nan')):+.3f} -> {v} {fired}", flush=True)
    # ---- paired comparisons vs zoo_ew (and H-WS16-3 vs slope_30d)
    cmp: dict[str, Any] = {}
    for g in SEGS:
        base = grids[g]["zoo_ew"]
        cg: dict[str, Any] = {}
        for fam in ("weighting", "single") + MOD_FAMILIES:
            names = [n for n in res if res[n]["family"] == fam and n in grids[g]]
            if not names:
                continue
            if fam == "order":
                cg[fam] = T.paired(grids5[g]["zoo_ew"], {n: grids5[g][n] for n in names}, idx[g])
            else:
                cg[fam] = T.paired(base, {n: grids[g][n] for n in names}, idx[g])
        mods = [n for n in res if res[n]["family"] in MOD_FAMILIES and res[n]["family"] != "order"]
        cg["all_mods"] = T.paired(base, {n: grids[g][n] for n in mods}, idx[g])
        others = [n for n in res if n != "zoo_ew" and res[n]["family"] not in ("order", "combo")]
        cg["all"] = T.paired(base, {n: grids[g][n] for n in others}, idx[g])
        if "kalman_llt" in grids[g] and "slope_30d" in grids[g]:
            cg["kalman_vs_slope30"] = T.paired(grids[g]["slope_30d"], {"kalman_llt": grids[g]["kalman_llt"]}, idx[g])
        cmp[g] = cg
    # ---- PBO and DSR inputs (in-sample daily R by entry day per spec)
    names_pbo = [n for n in res if res[n]["family"] not in ("order", "combo")]
    M = np.vstack([grids["llm_in_sample"][n][1] for n in names_pbo]).T
    pbo = T.pbo_cscv(M, 16)
    daily_sr = {n: T.sharpe(grids["llm_in_sample"][n][1]) for n in names_pbo}
    var_sr = float(np.nanvar(list(daily_sr.values()), ddof=1))
    n_trials = len(res) + 23 + 44 + 1          # + H-TR1, which read the same holdout
    dsr = {n: T.deflated_sharpe(grids["llm_in_sample"][n][1], n_trials, var_sr) for n in ("zoo_ew",) + tuple(k for k in res if res[k]["family"] == "combo")}
    # ---- H-WS16-7 (peg fill)
    zb = B.filter((pl.col("variant") == "zoo_ew") & (pl.col("peg_filled") >= 0))
    peg: dict[str, Any] = {}
    for g in SEGS:
        sub = zb.filter(pl.col("seg") == g)
        if sub.height < 20:
            continue
        r = S.r_net(sub)
        f = sub["peg_filled"].to_numpy() == 1
        peg[g] = {"n": sub.height, "fill_rate": float(f.mean()), "filled_mean": float(r[f].mean()) if f.any() else float("nan"),
                  "unfilled_mean": float(r[~f].mean()) if (~f).any() else float("nan"),
                  "unfilled_minus_filled": two_sample_diff(sub, r, ~f, seg_days[g], idx[g])}
    # ---- H-WS16-10 conviction
    conv = {}
    for g in SEGS:
        sub = B.filter((pl.col("variant") == "zoo_ew") & (pl.col("seg") == g))
        r = S.r_net(sub)
        s = sub["s_entry"].to_numpy().astype(float)
        day = sub["entry_ts"].dt.date().to_numpy().astype("datetime64[D]").astype(np.int64)
        conv[g] = T.dk_ols(np.c_[np.ones(len(s)), s], r, day, lag=0)
    # ---- entry hour (descriptive, H-WS16-6)
    zz = B.filter(pl.col("variant") == "zoo_ew")
    by_hour = {}
    for hr in sorted(zz["entry_hour"].unique().to_list()):
        sub = zz.filter(pl.col("entry_hour") == hr)
        by_hour[str(hr)] = {g: m_(sub.filter(pl.col("seg") == g), None, ci=False).get("mean") for g in SEGS} | {"n": sub.height,
                            "stop_share": float((sub["why"] == "stop").mean())}
    # ---- real pooled means on the synthetic sources (for the MC test)
    real_pooled = {}
    for n in res:
        sub = B.filter((pl.col("variant") == n) & pl.col("symbol").is_in(list(T.SYN_SOURCES)))
        if sub.height:
            real_pooled[n] = float(S.r_net(sub).mean())
    syn = synth_tables(real_pooled)
    for n, R in res.items():
        k = f"sbb|{n}"
        if k in syn:
            R["g8"] = {"p_mc": syn[k]["p_mc"], "null_q95": syn[k]["null_q95"], "real": syn[k]["real"]}
            if not (syn[k]["real"] > syn[k]["null_q95"]):
                if R["verdict"] == "SURVIVES":
                    R["verdict"] = "FRAGILE"
                R["fired"] = R["fired"] + ["G8 real mean <= 95th pct of the SBB(1d) null"]
    # ---- zoo_ew trade R distribution (dossier, the trend trade)
    from scipy.stats import kurtosis, skew
    from statsmodels.stats.diagnostic import acorr_ljungbox

    rz = S.r_net(zz.filter(pl.col("seg") == "llm_in_sample"))
    dz = grids["llm_in_sample"]["zoo_ew"][1]
    hold_h = ((zz["exit_ts"] - zz["entry_ts"]).dt.total_hours()).to_numpy()
    trade_dist = {"n": len(rz), "mean": float(rz.mean()), "sd": float(rz.std()), "skew": float(skew(rz)), "exkurt": float(kurtosis(rz)),
                  "hit": float((rz > 0).mean()), "q": {str(q): float(np.quantile(rz, q)) for q in (0.05, 0.25, 0.5, 0.75, 0.95, 0.99)},
                  "share_stop": float((zz.filter(pl.col("seg") == "llm_in_sample")["why"] == "stop").mean()),
                  "hold_days_median": float(np.median(hold_h) / 24), "hold_days_q90": float(np.quantile(hold_h, 0.9) / 24),
                  "lb_daily_p_10": float(acorr_ljungbox(dz, lags=[10])["lb_pvalue"].iloc[0]),
                  "top5_share": res["zoo_ew"]["base"].get("top5_share")}
    same_day = zz.group_by(zz["entry_ts"].dt.date().alias("d"), "side").agg(pl.len().alias("k"))
    trade_dist["mean_same_day_same_side"] = float(same_day["k"].mean())
    out = {"blocks": blocks, "common_block": common_block, "specs": res, "cmp": cmp, "pbo": pbo, "dsr": dsr, "var_sr_day": var_sr, "n_trials": n_trials, "peg": peg, "conviction": conv,
           "by_hour": by_hour, "synthetic": syn, "dossier": dossier_tables(), "capacity": capacity(), "trade_dist": trade_dist}
    out["selection"] = select(out)
    (T.OUT / "results.json").write_text(json.dumps(out, indent=1, default=_json), encoding="utf-8")
    if record:
        record_all(out)
    return out


def _json(x: Any) -> Any:
    if isinstance(x, np.integer | np.floating):
        return x.item()
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, date | datetime):
        return x.isoformat()
    if isinstance(x, tuple):
        return list(x)
    return str(x)


def select(out: dict) -> dict:
    """H-WS16-13 frozen rule (llm_in_sample only)."""
    res, cmp = out["specs"], out["cmp"]["llm_in_sample"]
    allm = cmp.get("all_mods", {})
    mods = {n: v for n, v in allm.items() if not n.startswith("_") and v.get("p_rw", 1) < 0.10 and v["diff"] > 0}
    om = cmp.get("order", {})
    mods.update({n: v for n, v in om.items() if not n.startswith("_") and v.get("p_rw", 1) < 0.10 and v["diff"] > 0})
    best_by_family: dict[str, tuple[str, float]] = {}
    for n, v in mods.items():
        fam = res[n]["family"]
        if fam not in best_by_family or v["diff"] > best_by_family[fam][1]:
            best_by_family[fam] = (n, v["diff"])
    chosen = [n for n, _ in best_by_family.values()]
    cand2 = [(n, R["segments"]["llm_in_sample"].get("lo", -9)) for n, R in res.items()
             if R["family"] in ("weighting", "single") and R.get("verdict_is") == "SURVIVES"]
    cand2.sort(key=lambda z: -z[1])
    return {"spec1_modifications": chosen, "mods_passing": mods, "spec2": cand2[0][0] if cand2 else None, "spec2_ranking": cand2[:6]}


def combo_spec(chosen: list[str]) -> dict:
    """Spec 1 = zoo_ew + the chosen modifications (fields merged; masks joined with '+')."""
    sp_all = T.specs()
    base = T.Spec("zoo_ew")
    d = dict(base.__dict__)
    masks = []
    for n in chosen:
        s = sp_all[n]
        for k, v in s.__dict__.items():
            if k in ("name", "family", "data_informed", "keys"):
                continue
            if k == "mask" and v:
                masks.append(v)
            elif v != getattr(base, k):
                d[k] = v
    d["mask"] = "+".join(masks) if masks else None
    d["name"] = "combo1"
    d["family"] = "combo"
    keys = ["theta", "bpd"] + (["stop_k"] if d["stop"] == "sigma" else ["margin"])
    if d.get("trail_k"):
        keys.append("trail_k")
    if d.get("time_bars"):
        keys.append("time_bars")
    d["keys"] = keys
    d["data_informed"] = any(sp_all[n].data_informed for n in chosen)
    return d


def record_all(out: dict) -> None:
    from ai_trader.research.hypotheses import record

    note = "WS16 trend_spec; data/research/ws16/results.json"
    dz = out["dossier"].get("sign_vs_tanh", {})
    for seg in SEGS:
        ts = [dz.get(f"{seg}|{k}", {}).get("t", [0, 0, 0])[2] for k in (7, 14, 30)]
        bs = [dz.get(f"{seg}|{k}", {}).get("b", [0, 0, 0])[2] for k in (7, 14, 30)]
        if seg == "llm_in_sample":
            v = "supports" if sum(t >= 2 for t in ts) >= 2 else ("contradicts" if all(t < 1 for t in ts) else "inconclusive")
            is_ok = [t >= 2 for t in ts]
        else:
            v = "supports" if any(is_ok) and all(b > 0 for b, ok in zip(bs, is_ok, strict=True) if ok) else "inconclusive"
            if not any(is_ok):
                v = "inconclusive"
        record("H-WS16-1", seg, v, {"t_tanh": ts, "b_tanh": bs}, note=note)
    for seg in SEGS:
        c = out["cmp"][seg]
        w = c.get("weighting", {})
        beats = [n for n, v in w.items() if not n.startswith("_") and n != "zoo_slow" and v.get("p_rw", 1) < 0.05 and v["diff"] >= 0.05]
        record("H-WS16-2", seg, "contradicts" if beats else "supports",
               {n: {"diff": v["diff"], "lo": v["lo"], "hi": v["hi"], "p_rw": v.get("p_rw")} for n, v in w.items() if not n.startswith("_")}, note=note)
        k = c.get("kalman_vs_slope30", {}).get("kalman_llt")
        if k:
            v = "contradicts" if (k["diff"] >= 0.05 and k["lo"] > 0) else "supports"
            record("H-WS16-3", seg, v, k, note=note)
        e = c.get("exit", {})
        beats = [n for n, v in e.items() if not n.startswith("_") and v.get("p_rw", 1) < 0.05 and v["diff"] >= 0.05]
        record("H-WS16-4", seg, "contradicts" if beats else "supports", {n: v for n, v in e.items() if not n.startswith("_")}, note=note)
        st = c.get("stop", {}).get("stop_pd")
        if st:
            if seg == "llm_in_sample":
                v = "supports" if st["lo"] > 0 else ("contradicts" if st["diff"] <= 0 else "inconclusive")
            else:
                v = "supports" if st["diff"] > 0 else "contradicts"
            record("H-WS16-5", seg, v, {**st, "stop_pw": c["stop"].get("stop_pw"), "stop_2s": c["stop"].get("stop_2s"), "stop_3s": c["stop"].get("stop_3s")}, note=note)
        tm = c.get("timing", {}).get("time_no12")
        if tm:
            v = "supports" if (tm["diff"] >= 0.02 and tm["lo"] > 0) else ("contradicts" if tm["diff"] <= 0 else "inconclusive")
            record("H-WS16-6", seg, v, {**tm, "by_hour": out["by_hour"]}, note=note)
        pg = out["peg"].get(seg)
        if pg:
            a_ok = pg["fill_rate"] >= 0.5
            d = pg["unfilled_minus_filled"]
            b_ok = d["lo"] > 0
            ch = out["cmp"][seg].get("order", {})
            v = "supports" if (a_ok and b_ok) else ("contradicts" if (not a_ok or d["diff"] <= 0) else "inconclusive")
            record("H-WS16-7", seg, v, {"fill_rate": pg["fill_rate"], "adverse": d, "order_cmp": {n: x for n, x in ch.items() if not n.startswith("_")}}, note=note)
        fe = c.get("filter", {}).get("filt_expected")
        if fe:
            if seg == "llm_in_sample":
                v = "supports" if (fe["diff"] >= 0.05 and fe["lo"] > 0) else ("contradicts" if fe["diff"] <= 0 else "inconclusive")
            else:
                v = "supports" if fe["diff"] > 0 else "contradicts"
            record("H-WS16-8", seg, v, {**fe, "diragree": c["filter"].get("filt_diragree"), "no_r6": c["filter"].get("filt_no_r6")}, note=note)
        fw = c.get("filter", {}).get("filt_weekend")
        if fw:
            v = "supports" if (fw["diff"] >= 0.05 and fw["lo"] > 0) else ("contradicts" if fw["diff"] <= 0 else "inconclusive")
            record("H-WS16-9", seg, v, fw, note=note + "; confirmatory on data already seen by STRESS")
        cv = out["conviction"].get(seg)
        if cv:
            t_, b_ = cv["t"][1], cv["b"][1]
            v = ("supports" if t_ >= 2 else ("contradicts" if b_ <= 0 else "inconclusive")) if seg == "llm_in_sample" else ("supports" if b_ > 0 else "contradicts")
            record("H-WS16-10", seg, v, {"slope": b_, "t": t_, "conv_mean": out["specs"].get("size_conviction", {}).get("segments", {}).get(seg, {}).get("mean")}, note=note)
        spa = c.get("all", {}).get("_spa", {})
        if spa and seg == "llm_in_sample":
            v = "supports" if spa["p"] < 0.05 else ("contradicts" if spa["p"] >= 0.20 else "inconclusive")
            record("H-WS16-11", seg, v, {"spa": spa, "pbo": out["pbo"], "dsr": out["dsr"], "n_trials": out["n_trials"]}, note=note)
    syn = out["synthetic"]
    a, b = syn.get("sbb|zoo_ew"), syn.get("garch99|zoo_ew") or syn.get("garch|zoo_ew")
    if a and b:
        v = "supports" if (a["p_mc"] <= 0.05 and b["p_mc"] <= 0.05) else ("contradicts" if (a["p_mc"] > 0.2 and b["p_mc"] > 0.2) else "inconclusive")
        record("H-WS16-12", "pooled", v, {"sbb": a, "garch99": b, "garch_uncapped": syn.get("garch|zoo_ew"), "auc_sbb": syn.get("auc_sbb"), "auc_garch": syn.get("auc_garch"), "auc_garch99": syn.get("auc_garch99")}, note=note)
    cap = out["capacity"]
    if cap:
        mx = max(x["impact_bp_200"] for x in cap.values())
        record("H-WS16-14", "llm_in_sample", "supports" if mx < 1 else ("contradicts" if mx > 3 else "inconclusive"), {"max_impact_bp_200": mx, "per_symbol": cap}, note=note)


# ====================================================================== markdown tables for the report
def _f(x: Any, nd: int = 3) -> str:
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return "n/a"
    return f"{x:+.{nd}f}" if isinstance(x, float) else str(x)


def tables(out: dict | None = None) -> str:
    out = out or json.loads((T.OUT / "results.json").read_text(encoding="utf-8"))
    res, L = out["specs"], []
    L.append("| spec | family | verdict (rules fired) | n | mean R [95% CI] | IS mean (n) | HO mean (n) | HO closed | +5 bp | lat5 | worst +-20% (D8) | engine trades/mo | engine R/mo | IS verdict |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for n, R in res.items():
        b, si, ho = R["base"], R["segments"]["llm_in_sample"], R["segments"]["clean_holdout"]
        fired = ", ".join(x.split(" ")[0] for x in R["fired"]) or "-"
        L.append(f"| {n}{'*' if R['data_informed'] else ''} | {R['family']} | {R['verdict']} ({fired}) | {b.get('n', 0)} | {_f(b.get('mean'))} [{_f(b.get('lo'))}, {_f(b.get('hi'))}] | "
                 f"{_f(si.get('mean'))} ({si.get('n', 0)}) | {_f(ho.get('mean'))} ({ho.get('n', 0)}) | {_f(R['closed']['clean_holdout'].get('mean'))} | {_f(R['costs']['slip+5'].get('mean'))} | "
                 f"{_f(R['lat5'].get('mean'))} | {_f(R['cliff'].get('worst'))} | {R['engine'].get('trades_per_month', float('nan')):.1f} | {_f(R['engine'].get('R_per_month'), 2)} | {R['verdict_is']} |")
    L.append("")
    for g in SEGS:
        L.append(f"Paired differences vs zoo_ew, {g} (mean R/trade; 95% day-bootstrap CI; Romano-Wolf p within the family):")
        L.append("")
        L.append("| family | spec | diff | 95% CI | p_RW | n |")
        L.append("|---|---|---|---|---|---|")
        for fam, d in out["cmp"][g].items():
            if fam in ("all", "all_mods"):
                continue
            for n, v in d.items():
                if n.startswith("_"):
                    continue
                L.append(f"| {fam} | {n} | {_f(v['diff'])} | [{_f(v['lo'])}, {_f(v['hi'])}] | {v.get('p_rw', float('nan')):.3f} | {v['n']} |")
        spa = out["cmp"][g].get("all", {}).get("_spa", {})
        L.append(f"\nSPA over all {spa.get('K')} variants: p = {spa.get('p', float('nan')):.3f}; Romano-Wolf over all trade modifications: min p = "
                 f"{min((v.get('p_rw', 1) for k, v in out['cmp'][g].get('all_mods', {}).items() if not k.startswith('_')), default=float('nan')):.3f}\n")
    return "\n".join(L)
