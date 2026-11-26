"""STRESS analysis: every unit x every scenario from the stored trade tables (stress.py), verdicts (H-STRESS0), the
registered hypotheses H-STRESS1..10, results.json and research/reports/stress_test_2026-11-26.md.

    PYTHONIOENCODING=utf-8 uv run python -m ai_trader.research.stress analyse [--record]
"""

from __future__ import annotations

import json
import math
from datetime import date
from typing import Any

import numpy as np
import polars as pl

from ai_trader.research import stress as S

REPORT = S.ROOT / "research" / "reports" / "stress_test_2026-11-26.md"
TREND_H1 = ("tsmom", "ema_stack", "donchian_breakout", "compression_breakout", "geo_struct", "geo_pivot_bo", "geo_flag_bo", "zoo_ensemble", "trend_4h")
MRCAL_H2 = ("zscore_reversion", "rsi_reversion", "weekend_shock_follow", "weekend_drift_fade", "cme_gap_fill")
CELLS = [f"R{i}" for i in range(1, 10)]


def _load(kind: str) -> pl.DataFrame:
    d = S.OUT / kind
    fs = sorted(d.glob("*.parquet")) if d.exists() else []
    if not fs:
        return pl.DataFrame()
    return pl.concat([pl.read_parquet(f) for f in fs], how="vertical_relaxed")


def _f(x: Any, nd: int = 3) -> str:
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return "n/a"
    return f"{x:+.{nd}f}" if isinstance(x, float) else str(x)


def _m(t: pl.DataFrame, ci: bool = False, costs: S.Costs | None = None) -> dict:
    if t.height == 0:
        return {"n": 0}
    return S.metrics(t, S.r_net(t, costs), ci=ci)


def _sel(t: pl.DataFrame, unit: str, variant: str) -> pl.DataFrame:
    if t.height == 0:
        return t
    return t.filter((pl.col("unit") == unit) & (pl.col("variant") == variant))


def _engine(t: pl.DataFrame, gate: bool = True) -> tuple[pl.DataFrame, dict, dict]:
    if t.height == 0:
        return t, {}, {"n": 0}
    acc, info = S.live_engine(t, S.r_net(t), S.LiveLimits.from_config(ev_gate=gate))
    if acc.height == 0:
        return acc, info, {"n": 0, "refused": info.get("refused", {})}
    es = S.engine_stats(acc, info)
    r_eq = acc["usd"].to_numpy() / 1.30
    es["R"] = S.metrics(acc, r_eq, ci=True)
    return acc, info, es


def _engine_r(acc: pl.DataFrame) -> np.ndarray:
    return acc["usd"].to_numpy() / 1.30 if acc.height else np.zeros(0)


def next_open_slip(t: pl.DataFrame, a: date, b: date) -> dict:
    """Stop exits filled at the NEXT bar open (S3 stop_next_open) for trades entered in [a, b]: slippage beyond the stop price in R."""
    if t.height == 0:
        return {"stops": 0}
    w = t.filter((pl.col("entry_ts").dt.date() >= a) & (pl.col("entry_ts").dt.date() <= b))
    stp = w.filter(pl.col("why") == "stop")
    if stp.height == 0:
        return {"stops": 0, "sum": float(S.r_net(w).sum()) if w.height else 0.0}
    e, s_, x, d = (stp[k].to_numpy().astype(float) for k in ("entry", "stop", "exit", "side"))
    g = d * (x - s_) / (np.abs(e - s_) + e * S.base_budget(stp["kind"].to_numpy()))
    r = S.r_net(w)
    return {"stops": stp.height, "mean_slip_R": float(g.mean()), "worst_slip_R": float(g.min()), "sum": float(r.sum()), "worst_trade": float(r.min())}


def perturb_summary(P: pl.DataFrame, unit: str, seg: str | None = None) -> dict:
    if P.height == 0:
        return {}
    t = P.filter(pl.col("unit") == unit)
    if seg:
        t = t.filter(pl.col("seg") == seg)
    if t.height == 0:
        return {}
    t = t.with_columns(pl.Series("R", S.r_net(t)))
    g = t.group_by("variant").agg(pl.len().alias("n"), pl.col("R").mean().alias("mean"), pl.col("R").sum().alias("total"))
    mp = {r["variant"]: r for r in g.iter_rows(named=True)}
    base = mp.get("base", {"n": 0, "mean": float("nan")})
    surf: dict[str, dict[str, float]] = {}
    for v, r in mp.items():
        if "@" in v:
            k, f = v.split("@")
            surf.setdefault(k, {})[f] = r["mean"]
    cliff_keys = []
    if base.get("n") and base["mean"] > 0:
        for k, d in surf.items():
            if any(d.get(f, 1.0) <= 0 for f in ("0.8", "1.2")):
                cliff_keys.append(k)
    joint = np.array([r["mean"] for v, r in mp.items() if v.startswith("joint") and r["n"] > 0])
    worst20 = min((d.get(f, np.inf) for d in surf.values() for f in ("0.8", "1.2")), default=float("nan"))
    worst50 = min((d.get(f, np.inf) for d in surf.values() for f in ("0.5", "1.5")), default=float("nan"))
    return {"base_n": base.get("n", 0), "base_mean": base.get("mean"), "surface": surf, "cliff_keys": cliff_keys,
            "cliff": bool(cliff_keys) if base.get("n") and base["mean"] > 0 else None, "worst_pm20": worst20, "worst_pm50": worst50,
            "joint_share_pos": float((joint > 0).mean()) if joint.size else float("nan"),
            "joint_median": float(np.median(joint)) if joint.size else float("nan"),
            "joint_q10": float(np.quantile(joint, 0.1)) if joint.size else float("nan")}


def synth_summary(Y: pl.DataFrame, unit: str) -> dict:
    if Y.height == 0:
        return {}
    out = {}
    for gen in ("sbb", "rs"):
        t = Y.filter((pl.col("unit") == unit) & (pl.col("gen") == gen) & (pl.col("n") > 0))
        if t.height == 0:
            continue
        m = t["mean"].to_numpy()
        out[gen] = {"paths": t.height, "median_mean": float(np.median(m)), "p_mean_le0": float((m <= 0).mean()),
                    "q05_total": float(np.quantile(t["total"].to_numpy(), 0.05)), "median_total": float(np.median(t["total"].to_numpy())),
                    "q05_max_dd": float(np.quantile(t["max_dd"].to_numpy(), 0.05)), "median_n": float(np.median(t["n"].to_numpy()))}
    return out


def hist_per_symbol_year(tb: pl.DataFrame) -> dict:
    """Historical comparators for S6 (one symbol, one year): median over symbols of total R per year and max DD."""
    if tb.height == 0:
        return {}
    rows = []
    for sym in tb["symbol"].unique().to_list():
        t = tb.filter(pl.col("symbol") == sym)
        r = S.r_net(t)
        yrs = max((t["exit_ts"].max() - t["entry_ts"].min()).days / 365.25, 0.25)
        dd = S.drawdown_stats(t, r)["max_dd"]
        rows.append((r.sum() / yrs, dd / yrs, len(r) / yrs))
    a = np.array(rows)
    return {"total_per_year_median": float(np.median(a[:, 0])), "max_dd_per_year_median": float(np.median(a[:, 1])), "n_per_year_median": float(np.median(a[:, 2]))}


def mm_summary() -> dict:
    p = S.ROOT / "research" / "reports" / "mm_weekend_2026-11-21.json"
    if not p.exists():
        return {}
    d = json.loads(p.read_text(encoding="utf-8"))
    rows = {}
    for name, b in (d.get("baselines") or {}).items():
        rows[name] = b["cells"]
    for h, fam in (d.get("families") or {}).items():
        held = fam.get("held") or {}
        if held.get("variant"):
            rows[f"{h}:{held['variant']}"] = held["cells"]
    out = {}
    for k, cells in rows.items():
        ms = {c: v["mean"] for c, v in cells.items()}
        out[k] = {"cells": {c: {"mean": v["mean"], "ci": v["ci"], "n_days": v.get("n_days")} for c, v in cells.items()},
                  "worst_cell_mean": min(ms.values()), "all_ci_above_0": all(v["ci"][0] > 0 for v in cells.values()),
                  "pessimistic_worst": min(v["mean"] for c, v in cells.items() if c.startswith("pessimistic"))}
    return out


def rule_candidates(res: dict) -> list[dict]:
    """Cells where a unit's base trades lose on average in BOTH segments (n >= 10 each, pooled n >= 50, pooled mean <= -0.05 R)."""
    out = []
    for u, r in res.items():
        for c, v in (r.get("cells") or {}).items():
            a, b, p = v.get("llm_in_sample", {}), v.get("clean_holdout", {}), v.get("pooled", {})
            if p.get("n", 0) >= 50 and a.get("n", 0) >= 10 and b.get("n", 0) >= 10 and p["mean"] <= -0.05 and a["mean"] < 0 and b["mean"] < 0:
                out.append({"unit": u, "cell": c, "pooled": p["mean"], "n": p["n"], "is": a["mean"], "ho": b["mean"], "n_ho": b["n"]})
    return sorted(out, key=lambda x: x["pooled"])


def analyse(record: bool = False) -> dict:
    design = json.loads((S.OUT / "design.json").read_text(encoding="utf-8"))
    U = S.units()
    T = _load("trades")
    P = _load("perturb")
    Y = _load("synthetic")
    DER = pl.read_parquet(S.OUT / "deribit" / "BTC-PERPETUAL_2020H1.parquet") if (S.OUT / "deribit" / "BTC-PERPETUAL_2020H1.parquet").exists() else pl.DataFrame()
    st = pl.read_parquet(S.OUT / "btc_state.parquet").sort("ms")
    down_ms, down_flag = st["ms"].to_numpy(), st["cell"].is_in(["R7", "R8", "R9"]).to_numpy()
    wins = S.crisis_windows()
    latcheck = json.loads((S.OUT / "latcheck.json").read_text(encoding="utf-8")) if (S.OUT / "latcheck.json").exists() else {}
    res: dict[str, dict] = {}
    for name, u in U.items():
        tb = _sel(T, name, "base")
        if tb.height == 0:
            res[name] = {"unit": asdict_unit(u), "base": {"n": 0}, "verdict": "n/a", "fired": ["no trades"]}
            continue
        r0 = S.r_net(tb)
        R: dict[str, Any] = {"unit": asdict_unit(u)}
        R["base"] = S.metrics(tb, r0)
        R["segments"] = {s: _m(tb.filter(pl.col("seg") == s), ci=True) for s in ("llm_in_sample", "clean_holdout")}
        cells = {}
        for c in CELLS:
            sub = tb.filter(pl.col("cell") == c)
            cells[c] = {"pooled": _m(sub), **{s: _m(sub.filter(pl.col("seg") == s)) for s in ("llm_in_sample", "clean_holdout")}}
        R["cells"] = cells
        R["slices"] = {k: _m(tb.filter(expr)) for k, expr in (
            ("weekend", pl.col("weekend")), ("weekday", ~pl.col("weekend")), ("nyse_holiday", pl.col("holiday")),
            ("funding_window", pl.col("funding_win")), ("off_funding", ~pl.col("funding_win")))}
        R["costs"] = {k: _m(tb, costs=S.Costs(fee_mult=fm, extra_bps=x, drought=dr)) for k, (fm, x, dr) in S.COST_VARIANTS.items()}
        R["costs_by_seg"] = {k: {s: _m(tb.filter(pl.col("seg") == s), costs=S.Costs(fee_mult=fm, extra_bps=x, drought=dr)) for s in ("llm_in_sample", "clean_holdout")}
                             for k, (fm, x, dr) in S.COST_VARIANTS.items() if k in ("fees_x1.5", "slip+5")}
        R["exec"] = {}
        for v in ("gap", "stop_next_open", "no_stop", "lat1", "lat5", "lat15"):
            tv = _sel(T, name, v)
            R["exec"][v] = _m(tv)
            if v == "lat5":
                R["exec_lat5_by_seg"] = {s: _m(tv.filter(pl.col("seg") == s)) for s in ("llm_in_sample", "clean_holdout")}
        tg = _sel(T, name, "gap")
        if tg.height:
            stp = tg.filter(pl.col("why") == "stop")
            e, s_, x, d = (stp[k].to_numpy().astype(float) for k in ("entry", "stop", "exit", "side"))
            den = np.abs(e - s_) + e * S.base_budget(stp["kind"].to_numpy())
            g = d * (x - s_) / den
            R["stop_gaps"] = {"stops": stp.height, "gapped": int(stp["gap"].sum()), "mean_gap_R": float(g[stp["gap"].to_numpy()].mean()) if stp["gap"].any() else 0.0,
                              "worst_gap_R": float(g.min()) if g.size else 0.0}
        # S9 live engine (registered: EV gate where defined) and without the gate
        acc, info, es = _engine(tb, gate=True)
        R["engine"] = es
        acc_ng, _, es_ng = _engine(tb, gate=False)
        R["engine_nogate"] = es_ng
        # S2 crisis windows: raw universe stream, engine stream (registered), engine stream without the gate
        cr = {}
        tno = _sel(T, name, "stop_next_open")
        for wname, (a, b) in wins.items():
            if wname.startswith("2020-03"):
                dt = _sel(DER, name, "base")
                if dt.height:
                    ra = S.r_net(dt)
                    accd, _, _ = _engine(dt, gate=True)
                    accd_ng, _, _ = _engine(dt, gate=False)
                    cr[wname] = {"raw": S.window_stats(dt, ra, a, b), "engine": S.window_stats(accd, _engine_r(accd), a, b),
                                 "engine_nogate": S.window_stats(accd_ng, _engine_r(accd_ng), a, b), "source": "Deribit BTC-PERPETUAL 1h"}
                else:
                    cr[wname] = {"raw": {"n": 0}, "engine": {"n": 0}, "engine_nogate": {"n": 0}, "source": "n/a"}
                continue
            cr[wname] = {"raw": S.window_stats(tb, r0, a, b), "engine": S.window_stats(acc, _engine_r(acc), a, b),
                         "engine_nogate": S.window_stats(acc_ng, _engine_r(acc_ng), a, b), "next_open": next_open_slip(tno, a, b)}
        R["crises"] = cr
        eng_sums = [v["engine"].get("sum") for v in cr.values() if v["engine"].get("n")]
        R["crisis_engine_min"] = float(min(eng_sums)) if eng_sums else None
        ng_sums = [v["engine_nogate"].get("sum") for v in cr.values() if v["engine_nogate"].get("n")]
        R["crisis_engine_nogate_min"] = float(min(ng_sums)) if ng_sums else None
        # S7 correlation stress
        cs = S.co_stop(tb, down_ms, down_flag)
        R["corr"] = _m(cs)
        R["corr"]["co_stopped"] = int((cs["why"] == "co_stop").sum())
        acc7, _, es7 = _engine(cs, gate=True)
        acc7n, _, es7n = _engine(cs, gate=False)
        R["corr_engine"] = {"cvar5_day": (es7.get("R") or {}).get("cvar5_day"), "base_cvar5_day": (es.get("R") or {}).get("cvar5_day"),
                            "n": es7.get("n", 0), "mean": (es7.get("R") or {}).get("mean")}
        R["corr_engine_nogate"] = {"cvar5_day": (es7n.get("R") or {}).get("cvar5_day"), "base_cvar5_day": (es_ng.get("R") or {}).get("cvar5_day"),
                                   "n": es7n.get("n", 0), "mean": (es7n.get("R") or {}).get("mean")}
        # S5, S6
        R["perturb"] = perturb_summary(P, name)
        R["perturb_is"] = perturb_summary(P, name, "llm_in_sample")
        R["synthetic"] = synth_summary(Y, name)
        R["hist_symbol_year"] = hist_per_symbol_year(tb)
        R["latcheck"] = latcheck.get(name)
        v, fired = S.verdict({"base": R["base"], "fees15": R["costs"]["fees_x1.5"], "slip5": R["costs"]["slip+5"], "lat5": R["exec"]["lat5"],
                              "holdout": R["segments"]["clean_holdout"], "cliff": R["perturb"].get("cliff"), "crisis_engine_min": R["crisis_engine_min"]})
        R["verdict"], R["fired"] = v, fired
        # sensitivity: closed trades only (positions still open at the data end, marked to the last close, removed)
        cl = pl.col("why") != "end"
        tbc = tb.filter(cl)
        R["open_at_end"] = {"n": int((tb["why"] == "end").sum()), "sum_R": float(r0[(tb["why"] == "end").to_numpy()].sum())}
        bc = _m(tbc, ci=True)
        vc, fc = S.verdict({"base": bc, "fees15": _m(tbc, costs=S.Costs(fee_mult=1.5)), "slip5": _m(tbc, costs=S.Costs(extra_bps=5.0)),
                            "lat5": _m(_sel(T, name, "lat5").filter(cl)), "holdout": _m(tbc.filter(pl.col("seg") == "clean_holdout")),
                            "cliff": R["perturb"].get("cliff"), "crisis_engine_min": R["crisis_engine_min"]})
        R["closed_only"] = {"base": bc, "holdout": _m(tbc.filter(pl.col("seg") == "clean_holdout")), "verdict": vc, "fired": fc}
        # in-sample-only verdict (H-STRESS10)
        tis = tb.filter(pl.col("seg") == "llm_in_sample")
        acc_is, _, _ = _engine(tis, gate=True)
        is_cr = [S.window_stats(acc_is, _engine_r(acc_is), a, b).get("sum") for w, (a, b) in wins.items() if b < date(2026, 7, 1) and not w.startswith("2020-03")]
        is_cr = [x for x in is_cr if x is not None]
        vi, fi = S.verdict({"base": _m(tis, ci=True), "fees15": R["costs_by_seg"]["fees_x1.5"]["llm_in_sample"],
                            "slip5": R["costs_by_seg"]["slip+5"]["llm_in_sample"], "lat5": R.get("exec_lat5_by_seg", {}).get("llm_in_sample"),
                            "holdout": None, "cliff": R["perturb_is"].get("cliff"), "crisis_engine_min": min(is_cr) if is_cr else None})
        R["verdict_is"], R["fired_is"] = vi, fi
        res[name] = R
        print(f"{name:22s} n={R['base']['n']:6d} mean={_f(R['base'].get('mean'))} [{_f(R['base'].get('lo'))},{_f(R['base'].get('hi'))}] -> {v} {fired}", flush=True)
    out = {"design_as_of": design["as_of"], "windows": {k: [str(a), str(b)] for k, (a, b) in wins.items()}, "units": res,
           "mm": mm_summary(), "rule_candidates": rule_candidates(res)}
    out["hypotheses"] = evaluate_hypotheses(res)
    (S.OUT / "results.json").write_text(json.dumps(out, indent=1, default=_json), encoding="utf-8")
    if record:
        record_all(out)
    from ai_trader.research.stress_writeup import write_report

    write_report(out, REPORT)
    print(REPORT)
    return out


def _json(x: Any) -> Any:
    if isinstance(x, float | np.floating):
        return float(x) if math.isfinite(float(x)) else None
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, np.bool_):
        return bool(x)
    return str(x)


def asdict_unit(u: S.Unit) -> dict:
    from dataclasses import asdict

    return asdict(u)


def evaluate_hypotheses(res: dict) -> dict:
    H: dict[str, dict] = {}

    def ok(x: dict | None) -> bool:
        return bool(x and x.get("n") and x["mean"] > 0)

    # H1
    det = {}
    for u in TREND_H1:
        r = res.get(u) or {}
        cb = r.get("costs_by_seg") or {}
        det[u] = all(ok((cb.get(k) or {}).get(s)) for k in ("fees_x1.5", "slip+5") for s in ("llm_in_sample", "clean_holdout"))
    k = sum(det.values())
    H["H-STRESS1"] = {"verdict": "supports" if k >= 6 else "contradicts" if k <= 2 else "inconclusive", "met": k, "of": len(TREND_H1), "detail": det}
    # H2
    det = {u: (res.get(u) or {}).get("verdict") for u in MRCAL_H2}
    k = sum(v == "FAILS" for v in det.values())
    H["H-STRESS2"] = {"verdict": "supports" if k >= 4 else "contradicts" if k <= 1 else "inconclusive", "fails": k, "detail": det}
    # H3
    det = {}
    for u, r in res.items():
        b, l5 = r.get("base") or {}, (r.get("exec") or {}).get("lat5") or {}
        if b.get("n", 0) >= 30 and l5.get("n"):
            det[u] = l5["mean"] - b["mean"]
    bad = [u for u, d in det.items() if abs(d) >= 0.05]
    H["H-STRESS3"] = {"verdict": "supports" if not bad else "contradicts" if len(bad) >= 3 else "inconclusive", "exceed": bad, "delta": det}
    # H4
    hits = [(u, w, v["engine"]["sum"]) for u, r in res.items() for w, v in (r.get("crises") or {}).items() if v["engine"].get("n") and v["engine"]["sum"] < -20]
    H["H-STRESS4"] = {"verdict": "supports" if not hits else "contradicts" if len(hits) >= 3 else "inconclusive", "hits": hits}
    # H5
    det = {}
    for u, r in res.items():
        ce = r.get("corr_engine") or {}
        if ce.get("n", 0) >= 30 and ce.get("cvar5_day") is not None and ce.get("base_cvar5_day") is not None:
            det[u] = {"s9": ce["base_cvar5_day"], "s7": ce["cvar5_day"], "ok": ce["cvar5_day"] >= 1.5 * ce["base_cvar5_day"]}
    nb = sum(not v["ok"] for v in det.values())
    H["H-STRESS5"] = {"verdict": "supports" if det and nb == 0 else "contradicts" if nb >= 3 else "inconclusive", "detail": det, "n_units": len(det)}
    # H6
    det = {}
    for u, r in res.items():
        e = r.get("engine") or {}
        if ok(r.get("base")) and e.get("n", 0) >= 30:
            det[u] = {"p_daily_stop": e.get("p_daily_stop_hit"), "p_loss50": e.get("p_loss_ge_50"),
                      "ok": (e.get("p_daily_stop_hit") or 0) < 0.02 and (e.get("p_loss_ge_50") if e.get("p_loss_ge_50") is not None else 1) < 0.05}
    nb = sum(not v["ok"] for v in det.values())
    H["H-STRESS6"] = {"verdict": "inconclusive" if not det else "supports" if nb == 0 else "contradicts" if nb >= len(det) / 2 else "inconclusive", "detail": det}
    # H7
    det = {u: (r.get("perturb") or {}).get("cliff") for u, r in res.items() if (r.get("perturb") or {}).get("cliff") is not None}
    share = (sum(not v for v in det.values()) / len(det)) if det else float("nan")
    H["H-STRESS7"] = {"verdict": "inconclusive" if not det else "supports" if share >= 2 / 3 else "contradicts" if share <= 1 / 3 else "inconclusive",
                      "share_no_cliff": share, "detail": det}
    # H8
    det = {}
    for u in TREND_H1:
        r = res.get(u) or {}
        sb = (r.get("synthetic") or {}).get("sbb")
        if ok(r.get("base")) and sb:
            det[u] = {"median_path_mean": sb["median_mean"], "ok": sb["median_mean"] <= 0}
    share = (sum(v["ok"] for v in det.values()) / len(det)) if det else float("nan")
    H["H-STRESS8"] = {"verdict": "inconclusive" if not det else "supports" if share >= 2 / 3 else "contradicts" if share <= 1 / 3 else "inconclusive",
                      "share": share, "detail": det}
    # H9
    det = {u: ok((r.get("costs") or {}).get("drought")) for u, r in res.items() if ok(r.get("base"))}
    nb = sum(not v for v in det.values())
    H["H-STRESS9"] = {"verdict": "inconclusive" if not det else "supports" if nb == 0 else "contradicts" if nb >= 3 else "inconclusive", "detail": det}
    # H10
    surv = [u for u, r in res.items() if r.get("verdict_is") == "SURVIVES"]
    det = {u: (res[u]["segments"]["clean_holdout"].get("mean"), res[u]["segments"]["clean_holdout"].get("n")) for u in surv}
    neg = sum(1 for m, n in det.values() if not (n and m is not None and m > 0))
    H["H-STRESS10"] = {"verdict": "inconclusive" if not surv else "supports" if neg == 0 else "contradicts" if neg >= len(surv) / 2 else "inconclusive",
                       "is_survivors": surv, "holdout": det}
    return H


def record_all(out: dict) -> None:
    from ai_trader.research.hypotheses import record

    note = "STRESS run; frozen design in research/hypotheses.yaml block STRESS; data/research/stress/results.json"
    for u, r in out["units"].items():
        vmap = {"SURVIVES": "supports", "FRAGILE": "inconclusive", "FAILS": "contradicts"}
        b = r.get("base") or {}
        record("H-STRESS0", "pooled", vmap.get(r.get("verdict"), "inconclusive"),
               {"unit": u, "stress_verdict": r.get("verdict"), "fired": r.get("fired"), "n": b.get("n"), "mean_R": b.get("mean"),
                "ci": [b.get("lo"), b.get("hi")], "holdout_mean_R": (r.get("segments") or {}).get("clean_holdout", {}).get("mean"),
                "holdout_n": (r.get("segments") or {}).get("clean_holdout", {}).get("n"),
                "data_generated": (r.get("unit") or {}).get("data_generated")},
               note=note + (" | data-generated unit: historical = not evidence" if (r.get("unit") or {}).get("data_generated") else ""),
               data="data/research/stress/results.json")
    for h, v in out["hypotheses"].items():
        seg = "clean_holdout" if h == "H-STRESS10" else "pooled"
        record(h, seg, v["verdict"], {k: x for k, x in v.items() if k != "verdict"}, note=note, data="data/research/stress/results.json")
