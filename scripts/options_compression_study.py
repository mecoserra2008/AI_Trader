"""Block OPX study runner (H-OPX1..H-OPX4 + the descriptive downside plan). Research only: no orders anywhere.

    uv run python scripts/options_compression_study.py build      # trades, predictive models, trend book -> data/research/options/study/
    uv run python scripts/options_compression_study.py analyse [--record]   # statistics, report, figures, (records verdicts)

Design: src/ai_trader/research/options_compression.py docstring and the diary entry of 2026-11-14 16:57 UTC.
"""

from __future__ import annotations

import argparse
import json
import os
import math
import pickle
import time
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl

from ai_trader.core.config import load
from ai_trader.research import options_compression as oc
from ai_trader.research.evaluate import knowledge_cutoff

ROOT = load().root
STUDY = Path(os.environ.get("OPX_STUDY", str(ROOT / "data" / "research" / "options" / "study")))
REPORT = "options_compression_2026-11-14"
CUT = knowledge_cutoff().date()
DATA_DESC = ("Deribit BTC option trades (history.deribit.com, hourly per-instrument aggregates 2020-01..2026-11), Bybit BTCUSDT/ETHUSDT 1h + "
             "funding, deribit_svi smile summary and DVOL (lake)")


def log(s: str) -> None:
    print(f"[{datetime.now(UTC):%H:%M:%S}] {s}", flush=True)


def seg_of(entry: date, exit_: date) -> str:
    if entry >= CUT:
        return "clean_holdout"
    if exit_ < CUT:
        return "llm_in_sample"
    return "straddles_cutoff"


# ====================================================================== trades per entry day

def leg_opt(expiry: date, K: float, cp: str, qty: float) -> oc.Leg:
    return oc.Leg("opt", qty, oc.inst_name(expiry, K, cp), K, expiry, cp)


def trades_for_day(m: oc.Market, E: date, last_day: date) -> dict | None:
    """Straddle (unhedged, hedged), user structure, comparator for entry day E (14-day hold) + to-expiry variant."""
    if E not in m.S or E not in m.pillars:
        return None
    lst = oc.listed(m, E)
    X = oc.choose_expiry(lst, E)
    if X is None:
        return None
    ks = oc.strikes_for(lst, X)
    T = oc.tau(E, X)
    F = m.F(E, T)
    out = {"entry": E, "expiry": X, "dte": (X - E).days, "S0": m.S[E], "F0": F}
    exit_d = E + timedelta(days=oc.HOLD_DAYS)
    if exit_d > last_day:
        return None
    # ---- straddle
    Ka = float(ks[np.argmin(np.abs(ks - F))])
    sl = [leg_opt(X, Ka, "C", 1.0), leg_opt(X, Ka, "P", 1.0)]
    r_u = oc.run_position(m, E, sl, oc.HOLD_DAYS)
    r_h = oc.run_position(m, E, sl, oc.HOLD_DAYS, hedge=True)
    if r_u is not None and r_h is not None:
        prem = r_u["prem_usd"]
        out["straddle"] = {"K": Ka, "prem_usd": prem, "prem_pct_S": prem / m.S[E] * 100, "src": r_u["src"], "src_exit": r_u["src_exit"],
                           "iv_c": r_u["q0"][0].iv, "iv_p": r_u["q0"][1].iv,
                           "net_prem_unhedged": r_u["net_usd"] / prem, "net_prem_hedged": r_h["net_usd"] / prem,
                           "net_btc_prem_unhedged": r_u["net_btc"] / (prem / m.S[E]), "net_btc_prem_hedged": r_h["net_btc"] / (prem / m.S[E]),
                           "net_usd_unhedged": r_u["net_usd"], "net_usd_hedged": r_h["net_usd"], "net_btc_usd_unhedged": r_u["net_btc_usd"],
                           "cost_usd": r_u["cost_usd"], "hedge_cost_usd": r_h["hedge_cost_usd"], "hedge_usd": r_h["hedge_usd"],
                           "hedge_fund_usd": r_h["hedge_fund_usd"], "ST": r_u["ST"],
                           "rv_14d": float(np.sum(np.diff(np.log(r_u["S_path"])) ** 2) * 365 / oc.HOLD_DAYS) ** 0.5,
                           "marks": r_u["marks"].tolist(), "S_path": r_u["S_path"].tolist()}
    # ---- user structure vs comparator
    Kc = oc.pick_by_delta(m, E, X, ks, oc.CALL_DELTA, "C")
    Kp = oc.pick_by_delta(m, E, X, ks, oc.PUT_DELTA, "P")
    if Kc is not None and Kp is not None:
        call, put = leg_opt(X, Kc, "C", 1.0), leg_opt(X, Kp, "P", -1.0)
        qc = oc.quote(m, E, call.inst, Kc, X, "C")
        qp = oc.quote(m, E, put.inst, Kp, X, "P")
        if qc is not None and qp is not None:
            n = -qp.delta
            res = {"Kc": Kc, "Kp": Kp, "n_perp": n, "src_c": qc.src, "src_p": qp.src, "iv_c": qc.iv, "iv_p": qp.iv,
                   "delta_c": qc.delta, "delta_p": qp.delta, "vega_c": qc.vega / 100, "vega_p": qp.vega / 100,
                   "theta_c": qc.theta, "theta_p": qp.theta, "V_c": qc.v, "V_p": qp.v,
                   "intrinsic_p": max(Kp - qp.F, 0.0), "tv_share_p": (qp.v - max(Kp - qp.F, 0.0)) / qp.v if qp.v > 0 else float("nan")}
            for tag, days in (("14d", oc.HOLD_DAYS), ("exp", (X - E).days)):
                if E + timedelta(days=days) > last_day:
                    continue
                rs = oc.run_position(m, E, [call, put], days)
                rc = oc.run_position(m, E, [call, oc.Leg("perp", n)], days)
                ro = oc.run_position(m, E, [call], days)
                rp = oc.run_position(m, E, [put], days)
                rf = oc.run_position(m, E, [oc.Leg("perp", n)], days)
                if rs is None or rc is None or ro is None or rp is None or rf is None:
                    continue
                S0 = m.S[E]
                res[tag] = {"struct_pct": rs["net_usd"] / S0 * 100, "comp_pct": rc["net_usd"] / S0 * 100,
                            "diff_pct": (rp["net_usd"] - rf["net_usd"]) / S0 * 100, "call_pct": ro["net_usd"] / S0 * 100,
                            "shortput_pct": rp["net_usd"] / S0 * 100, "perp_pct": rf["net_usd"] / S0 * 100,
                            "struct_btc": rs["net_btc"], "comp_btc": rc["net_btc"], "struct_btc_usd_pct": rs["net_btc_usd"] / S0 * 100,
                            "perp_fund_pct": rf["fund_usd"] / S0 * 100, "put_cost_pct": rp["cost_usd"] / S0 * 100,
                            "perp_cost_pct": rf["cost_usd"] / S0 * 100, "src_exit_p": rp["src_exit"][0], "ST": rs["ST"],
                            "marks_struct": rs["marks"].tolist() if tag == "14d" else None,
                            "S_path": rs["S_path"].tolist() if tag == "14d" else None}
            out["user"] = res
    return out


def build_trades(m: oc.Market, det: pl.DataFrame, last_day: date) -> list[dict]:
    d = det.filter(pl.col("pct_rv").is_finite() & pl.col("pct_iv").is_finite())
    days = d["entry_date"].to_list()
    ev = set(d.filter(pl.col("event"))["entry_date"].to_list())
    out = []
    t0 = time.time()
    for i, E in enumerate(days):
        r = trades_for_day(m, E, last_day)
        if r is None:
            continue
        r["event"] = E in ev
        r["seg"] = seg_of(E, E + timedelta(days=oc.HOLD_DAYS))
        out.append(r)
        if i % 200 == 0:
            log(f"  trades {i}/{len(days)} ({time.time() - t0:.0f}s)")
    return out


# ====================================================================== trend book (H-OPX4 primary arm)

def trend_book(as_of_iso: str, models: dict, only: tuple[str, ...] | None = None) -> pl.DataFrame:
    from ai_trader.research.exit_policy_study import TREND
    from ai_trader.research.strategy_panel import (
        UNIV_L,
        WARM_DAYS,
        cost_bps,
        frames_for,
        held_leverage,
        strategy_fn,
    )

    as_of = datetime.fromisoformat(as_of_iso)
    c_frac = cost_bps() / 1e4
    rows = []
    for sym in (only or UNIV_L):
        ck = STUDY / f"trend_{sym}.parquet"
        if ck.exists():
            rows.append(pl.read_parquet(ck))
            continue
        t0 = time.time()
        if models.get(sym) is None:
            continue
        pr, _ = models[sym]
        po = pr.filter(pl.col("s").is_finite()).select(pl.col("origin").dt.cast_time_unit("us"), "s", "qz_95").sort("origin")
        f1, f4, _src = frames_for(sym, "L", as_of)
        if f1.height == 0:
            pl.DataFrame({"symbol": []}).write_parquet(ck)
            continue
        start = max(f1["event_ts"].min(), f4["event_ts"].min()).date() + timedelta(days=WARM_DAYS)
        rr = []
        for name in TREND:
            freq = "4h" if name.startswith("geo_") or name == "zoo_ensemble" else "1h"
            f = f1 if freq == "1h" else f4
            bh = 4 if freq == "4h" else 1
            raw = np.nan_to_num(strategy_fn(name)(f).to_numpy().astype(float))
            _hs, lev = held_leverage(raw, f["sigma_next"].to_numpy())
            fr = np.nan_to_num(f["fwd_ret"].to_numpy())
            lc, lh, ll = (np.log(f[c].to_numpy()) for c in ("close", "high", "low"))
            ct = f.select(pl.col("close_ts").dt.replace_time_zone("UTC") if f["close_ts"].dtype.time_zone is None else pl.col("close_ts")
                          ).to_series().dt.cast_time_unit("us")
            # ex-ante scale at each close: latest origin <= close
            j = pl.DataFrame({"t": ct}).join_asof(po.rename({"origin": "t"}), on="t", strategy="backward")
            s = j["s"].to_numpy()
            qz = j["qz_95"].to_numpy()
            sg = np.sign(lev)
            n = len(lev)
            i = 0
            ctl = ct.to_list()
            while i < n:
                if sg[i] == 0:
                    i += 1
                    continue
                k = i
                while k + 1 < n and sg[k + 1] == sg[i]:
                    k += 1
                if k + 1 >= n:
                    break
                ed = ctl[i].date()
                if ed >= start and ctl[k + 1].date() < as_of.date() and np.isfinite(s[i]) and np.isfinite(qz[i]):
                    el = (np.arange(n) - i) * bh
                    thr = np.where(el > 0, qz[i] * s[i] * np.sqrt(np.maximum(el, 1) / 24.0), np.inf)
                    r = oc.scale_out_trade(lev, fr, lc, lh, ll, i, k, thr, c_frac)
                    sq = math.sqrt(24 / bh)
                    rr.append({"symbol": sym, "strategy": name, "freq": freq, "entry_ts": ctl[i], "entry_date": ed, "side": int(sg[i]),
                               "bars": k - i + 1, "R0": (r["gross0"] - r["cost0"]) / sq, "R1": (r["gross1"] - r["cost1"]) / sq,
                               "n_trig": r["n_trig"], "n_fill": r["n_fill"]})
                i = k + 1
        df = pl.DataFrame(rr) if rr else pl.DataFrame({"symbol": []})
        df.write_parquet(ck)
        rows.append(df)
        log(f"  trend {sym}: {df.height} trades ({time.time() - t0:.0f}s)")
    rows = [r for r in rows if r.height]
    return pl.concat(rows, how="vertical_relaxed") if rows else pl.DataFrame()


# ====================================================================== option scale-out (H-OPX4 option arm) and downside plan

def hourly_extremes(instruments: set[str]) -> dict:
    from ai_trader.research.options_download import OUT

    fs = sorted(str(f) for f in OUT.glob("btc_*.parquet"))
    d = (pl.scan_parquet(fs).filter(pl.col("instrument").is_in(list(instruments)))
         .select("instrument", "hour", "min_price", "max_price").collect())
    out = {}
    for (inst,), g in d.group_by("instrument"):
        g = g.sort("hour")
        out[inst] = (g["hour"].dt.epoch("ms").to_numpy(), g["min_price"].to_numpy(), g["max_price"].to_numpy())
    return out


def opt_filled(ext: dict, inst: str, day: date, limit_btc: float, sell: bool) -> bool:
    """Limit at the mark, resting from day 08:00 for 24 h: filled only if a trade prints through it."""
    if inst not in ext:
        return False
    ts, lo, hi = ext[inst]
    t0 = int(oc.fixing(day).timestamp() * 1000)
    a, b = np.searchsorted(ts, t0), np.searchsorted(ts, t0 + 86_400_000)
    if b <= a:
        return False
    return bool(np.any(hi[a:b] > limit_btc)) if sell else bool(np.any(lo[a:b] < limit_btc))


def perp_filled(m: oc.Market, day: date, px: float, sell: bool) -> bool:
    b = m.hourly_px
    t0 = oc.fixing(day)
    w = b.filter((pl.col("event_ts") >= t0) & (pl.col("event_ts") < t0 + timedelta(days=1)))
    if w.height == 0:
        return False
    return bool((w["high"] > px).any()) if sell else bool((w["low"] < px).any())


def scaled_position(m: oc.Market, E: date, legs: list[oc.Leg], marks: np.ndarray, S_path: np.ndarray, start_k: int,
                    trig_days: list[int], ext: dict) -> tuple[float, int, int]:
    """Net USD of `legs` (opened at mark 0 = day E) with up to three 1/3 limit exits at the trigger marks, the rest
    closed at the last mark. Returns (net_usd, n_triggers, n_leg_fills)."""
    days = marks.shape[0] - 1
    net, fills = 0.0, 0
    for j, lg in enumerate(legs):
        frac_exits = []
        for k in trig_days[:3]:
            d = E + timedelta(days=k)
            if lg.kind == "opt":
                ok = opt_filled(ext, lg.inst, d, marks[k, j] / S_path[k], sell=lg.qty > 0)
            else:
                ok = perp_filled(m, d, marks[k, j], sell=lg.qty > 0)
            if ok:
                frac_exits.append(k)
        fills += len(frac_exits)
        rem = 1.0 - len(frac_exits) / 3.0
        pnl = sum(lg.qty / 3.0 * (marks[k, j] - marks[0, j]) for k in frac_exits) + rem * lg.qty * (marks[-1, j] - marks[0, j])
        if lg.kind == "opt":
            q0 = oc.quote(m, E, lg.inst, lg.K, lg.expiry, lg.cp)
            hs0 = q0.hs_btc if q0 else 0.0
            cost = oc.option_cost_btc(hs0) * abs(lg.qty) * S_path[0]
            for k in frac_exits:
                qk = oc.quote(m, E + timedelta(days=k), lg.inst, lg.K, lg.expiry, lg.cp)
                cost += oc.option_cost_btc(qk.hs_btc if qk else 0.0) * abs(lg.qty) / 3.0 * S_path[k]
            if rem > 0:
                qe = oc.quote(m, E + timedelta(days=days), lg.inst, lg.K, lg.expiry, lg.cp)
                if qe is not None and qe.src != "settle":
                    cost += oc.option_cost_btc(qe.hs_btc) * abs(lg.qty) * rem * S_path[-1]
            fund = 0.0
        else:
            cost = oc.PERP_COST * abs(lg.qty) * (marks[0, j] + sum(marks[k, j] / 3.0 for k in frac_exits) + rem * marks[-1, j])
            fund = 0.0
            if lg.kind == "perp":
                for k in frac_exits:
                    fund += oc.funding_cost(m, lg.qty / 3.0, oc.fixing(E), oc.fixing(E + timedelta(days=k)))
                fund += oc.funding_cost(m, lg.qty * rem, oc.fixing(E), oc.fixing(E + timedelta(days=days)))
        net += pnl - cost - fund
    return net, len(trig_days[:3]), fills


def pnl_fn_factory(m: oc.Market, E: date, legs: list[oc.Leg], q0: list, k: int):
    S0 = m.S[E]
    vals = []
    for lg, q in zip(legs, q0):
        vals.append((lg, q))

    def f(R):
        S = S0 * np.exp(np.asarray(R, float))
        tot = np.zeros_like(S)
        for lg, q in vals:
            if lg.kind == "opt":
                T = max(q.T - k / 365.0, 0.0)
                F = S * math.exp(m.basis.get(E, 0.0) * T)
                tot = tot + lg.qty * (oc.b76(F, lg.K, T, q.iv, lg.cp) - q.v)
            else:
                tot = tot + lg.qty * (S - S0)
        return tot

    return f


def option_arm(m: oc.Market, trades: list[dict], pred1: tuple[pl.DataFrame, dict], ext: dict) -> pl.DataFrame:
    pr, grids = pred1
    pmap = {d: (s, int(r)) for d, s, r in zip(pr["date"].to_list(), pr["s"].to_list(), pr["refit"].to_list())}
    rows = []
    for t in trades:
        E = t["entry"]
        if E not in pmap or not np.isfinite(pmap[E][0]) or pmap[E][1] < 0:
            continue
        s, rid = pmap[E]
        ugrid = grids[rid]
        X = t["expiry"]
        books = []
        if "straddle" in t:
            st = t["straddle"]
            books.append(("straddle", [leg_opt(X, st["K"], "C", 1.0), leg_opt(X, st["K"], "P", 1.0)], np.asarray(st["marks"]), np.asarray(st["S_path"])))
        if "user" in t and "14d" in t["user"] and t["user"]["14d"]["marks_struct"] is not None:
            u = t["user"]
            books.append(("user", [leg_opt(X, u["Kc"], "C", 1.0), leg_opt(X, u["Kp"], "P", -1.0)], np.asarray(u["14d"]["marks_struct"]),
                          np.asarray(u["14d"]["S_path"])))
        for name, legs, marks, S_path in books:
            q0 = [oc.quote(m, E, lg.inst, lg.K, lg.expiry, lg.cp) for lg in legs]
            if any(q is None for q in q0):
                continue
            qty = np.array([lg.qty for lg in legs])
            trig = []
            for k in range(1, oc.HOLD_DAYS):
                fn = pnl_fn_factory(m, E, legs, q0, k)
                thr = oc.pnl_quantile(fn, s * math.sqrt(k) * ugrid, 0.95)
                if float(qty @ (marks[k] - marks[0])) > thr:
                    trig.append(k)
                if len(trig) >= 3:
                    break
            net1, ntr, nf = scaled_position(m, E, legs, marks, S_path, 0, trig, ext)
            net0, _, _ = scaled_position(m, E, legs, marks, S_path, 0, [], ext)
            rows.append({"entry": E, "book": name, "event": t["event"], "seg": t["seg"], "R0": net0 / S_path[0] * 100,
                         "R1": net1 / S_path[0] * 100, "n_trig": ntr, "n_fill": nf})
    return pl.DataFrame(rows)


def downside_plan(m: oc.Market, trades: list[dict], pred1: tuple[pl.DataFrame, dict], ext_fn) -> list[dict]:
    """Descriptive only. Bearish-compression events: compression event with S(E+1) < S(E)."""
    pr, grids = pred1
    pmap = {d: (s, int(r)) for d, s, r in zip(pr["date"].to_list(), pr["s"].to_list(), pr["refit"].to_list())}
    out = []
    for t in trades:
        if not t["event"] or "user" not in t or "14d" not in t["user"]:
            continue
        E, X = t["entry"], t["expiry"]
        E1 = E + timedelta(days=1)
        if E1 not in m.S or m.S[E1] >= m.S[E]:
            continue
        u = t["user"]
        base = u["14d"]["struct_pct"]
        lst = oc.listed(m, E1)
        Kp25 = oc.pick_by_delta(m, E1, X, oc.strikes_for(lst, X), -0.25, "P")
        if Kp25 is None or E1 not in pmap or pmap[E1][1] < 0:
            continue
        legs = [leg_opt(X, Kp25, "P", oc.PROT_PUTS), oc.Leg("perp", -oc.PROT_PERP)]
        days = oc.HOLD_DAYS - 1
        r = oc.run_position(m, E1, legs, days)
        if r is None:
            continue
        marks, S_path, q0 = r["marks"], r["S_path"], r["q0"]
        s, rid = pmap[E1]
        ugrid = grids[rid]
        qty = np.array([lg.qty for lg in legs])
        hedge_k = None
        for k in range(1, days):
            fn = pnl_fn_factory(m, E1, legs, q0, k)
            thr = oc.pnl_quantile(fn, s * math.sqrt(k) * ugrid, 0.95)
            if float(qty @ (marks[k] - marks[0])) > thr:
                hedge_k = k
                break
        prot_hold = r["net_usd"]
        if hedge_k is None:
            prot_plan, spot_qty, nf = prot_hold, 0.0, 0
        else:
            dk = E1 + timedelta(days=hedge_k)
            qk = oc.quote(m, dk, legs[0].inst, legs[0].K, X, "P")
            delta_prot = oc.PROT_PUTS * (qk.delta if qk else -0.5) - oc.PROT_PERP
            spot_qty = -delta_prot
            # spot leg opened at mark hedge_k; the plan then scales out protection + spot over the next three marks
            spot_marks = np.array([m.perp.get(E1 + timedelta(days=k), np.nan) for k in range(days + 1)])
            trig = [k for k in range(hedge_k + 1, min(hedge_k + 4, days))]
            legs2 = legs + [oc.Leg("spot", spot_qty)]
            marks2 = np.column_stack([marks, spot_marks])
            # the spot leg's P&L starts at hedge_k: shift its entry mark
            marks2[: hedge_k + 1, 2] = spot_marks[hedge_k]
            ext = ext_fn({legs[0].inst})
            prot_plan, _, nf = scaled_position(m, E1, legs2, marks2, S_path, 0, trig, ext)
        out.append({"entry": E.isoformat(), "seg": t["seg"], "first_day_ret_pct": (m.S[E1] / m.S[E] - 1) * 100,
                    "ret_14d_pct": (u["14d"]["ST"] / m.S[E] - 1) * 100, "struct_pct": base,
                    "prot_hold_pct": prot_hold / m.S[E] * 100, "prot_plan_pct": prot_plan / m.S[E] * 100,
                    "struct_plus_plan_pct": base + prot_plan / m.S[E] * 100, "spot_hedge_day": hedge_k, "spot_qty": spot_qty,
                    "n_fills": nf, "Kp25": Kp25})
    return out


# ====================================================================== build

def build_preds(as_of: datetime) -> dict:
    STUDY.mkdir(parents=True, exist_ok=True)
    preds = {}
    for sym in ("BTCUSDT", "ETHUSDT"):
        bars = None
        for H in (1, 3, 7):
            f = STUDY / f"pred_{sym}_{H}.pkl"
            if f.exists():
                preds[(sym, H)] = pickle.loads(f.read_bytes())
                continue
            if bars is None:
                bars = oc.lake_inputs(as_of, sym)["bars"]
            t0 = time.time()
            preds[(sym, H)] = oc.predictive(bars, H)
            f.write_bytes(pickle.dumps(preds[(sym, H)]))
            log(f"  predictive {sym} H={H} ({time.time() - t0:.0f}s)")
    return preds


def build_trend() -> None:
    from ai_trader.research.strategy_panel import UNIV_L, priors_dir

    STUDY.mkdir(parents=True, exist_ok=True)
    meta = json.loads((priors_dir() / "strategy_panel_meta.json").read_text(encoding="utf-8"))
    tb_as_of = datetime.fromisoformat(meta["as_of"])
    models = {}
    for sym in UNIV_L:
        f = STUDY / f"pred1_{sym}.pkl"
        if (STUDY / f"trend_{sym}.parquet").exists():
            models[sym] = None
            continue
        if f.exists():
            models[sym] = pickle.loads(f.read_bytes())
        else:
            t0 = time.time()
            bars = oc.lake_inputs(tb_as_of, sym)["bars"]
            models[sym] = oc.predictive(bars, 1)
            f.write_bytes(pickle.dumps(models[sym]))
            log(f"  predictive {sym} 1d ({time.time() - t0:.0f}s)")
        trend_book(meta["as_of"], {sym: models[sym]}, only=(sym,))
    tb = trend_book(meta["as_of"], {})
    tb.write_parquet(STUDY / "trend_book.parquet")
    log(f"trend book: {tb.height} trades")


def build_options(as_of: datetime) -> None:
    STUDY.mkdir(parents=True, exist_ok=True)
    log("market")
    m, raw = oc.build_market(as_of, log=log)
    rv = oc.rv7_at_0800(raw["bars"])
    iv = oc.iv_series(raw["svi_w"], raw["dvol"])
    det = oc.detector(rv, iv.select("date", "iv"))
    det = det.join(iv.select("date", "iv_src"), on="date", how="left")
    det.write_parquet(STUDY / "detector.parquet")
    last_day = max(d for d in m.S if d <= as_of.date())
    log(f"detector: {det.filter(pl.col('event')).height} events, last day {last_day}")
    rq = oc.real_quote_table(m, raw["quotes"])
    rq.select("date", "instrument", "dte", "cp", "strike", "S", "F", "T", "iv", "delta", "model_iv", "v_real", "v_model", "n", "mad_mark"
              ).write_parquet(STUDY / "real_quotes.parquet")
    pk = STUDY / "trades.pkl"
    if pk.exists():
        trades = pickle.loads(pk.read_bytes())
    else:
        trades = build_trades(m, det, last_day)
        pk.write_bytes(pickle.dumps(trades))
    log(f"trades: {len(trades)} entry days, {sum(t['event'] for t in trades)} events")
    preds = build_preds(as_of)
    insts = set()
    for t in trades:
        X = t["expiry"]
        if "straddle" in t:
            insts |= {oc.inst_name(X, t["straddle"]["K"], c) for c in "CP"}
        if "user" in t:
            insts |= {oc.inst_name(X, t["user"]["Kc"], "C"), oc.inst_name(X, t["user"]["Kp"], "P")}
    ext = hourly_extremes(insts)
    oa = option_arm(m, trades, preds[("BTCUSDT", 1)], ext)
    oa.write_parquet(STUDY / "option_arm.parquet")
    log(f"option arm: {oa.height} positions")
    dp = downside_plan(m, trades, preds[("BTCUSDT", 1)], hourly_extremes)
    (STUDY / "downside_plan.json").write_text(json.dumps(dp, indent=1, default=str))
    log(f"downside plan: {len(dp)} bearish-compression events")


# ====================================================================== analyse

PAL = {"s1": "#2a78d6", "s2": "#eb6834", "s3": "#1baf7a", "ink": "#0b0b0b", "ink2": "#52514e", "grid": "#e4e3df", "surface": "#fcfcfb"}
FIG = ROOT / "research" / "figures" / "options_compression_2026-11-14"


def data_summary() -> dict:
    from ai_trader.research.options_download import OUT

    man = json.loads((OUT / "manifest.json").read_text())["months"]
    files = sorted(OUT.glob("btc_*.parquet"))
    return {"months_complete": len(man), "first": min(man), "last": max(man), "trades": int(sum(v["trades"] for v in man.values())),
            "rows": int(sum(v["rows"] for v in man.values())), "mb_on_disk": round(sum(f.stat().st_size for f in files) / 1e6, 1),
            "files": len(files), "partial_months": [f.stem[4:] for f in files if f.stem[4:] not in man]}


def model_error(rq: pl.DataFrame) -> dict:
    d = rq.filter((pl.col("dte") >= 7) & (pl.col("dte") <= 120) & (pl.col("delta").abs() >= 0.05) & (pl.col("delta").abs() <= 0.95)
                  & pl.col("model_iv").is_finite() & pl.col("iv").is_finite() & (pl.col("v_real") > 0))
    d = d.with_columns(((pl.col("model_iv") - pl.col("iv")) * 100).alias("e_iv"), (pl.col("v_model") / pl.col("v_real") - 1).alias("e_px"),
                       ((pl.col("v_model") - pl.col("v_real")) / pl.col("S")).alias("e_btc"),
                       pl.when(pl.col("delta").abs() < 0.35).then(pl.lit("|d|<0.35")).when(pl.col("delta").abs() <= 0.65).then(pl.lit("0.35-0.65"))
                       .otherwise(pl.lit("|d|>0.65")).alias("bucket"))

    def s(x):
        if x.height == 0:
            return {"n": 0}
        e = x["e_iv"].to_numpy()
        p = x["e_px"].to_numpy()
        b = x["e_btc"].to_numpy()
        return {"n": int(x.height), "bias_vol_pts": float(np.mean(e)), "mae_vol_pts": float(np.mean(np.abs(e))),
                "median_abs_vol_pts": float(np.median(np.abs(e))), "rmse_vol_pts": float(np.sqrt(np.mean(e * e))),
                "median_abs_price_err_pct": float(np.median(np.abs(p)) * 100), "mean_price_err_pct": float(np.mean(p) * 100),
                "median_abs_err_btc": float(np.median(np.abs(b)))}

    tenor = pl.when(pl.col("dte") < 21).then(pl.lit("7-20d")).when(pl.col("dte") <= 45).then(pl.lit("21-45d")).otherwise(pl.lit("46-120d"))
    deep = ((pl.col("cp") == "P") & (pl.col("delta") <= -0.75)) | ((pl.col("cp") == "C") & (pl.col("delta") >= 0.1) & (pl.col("delta") <= 0.25))
    out = {"all": s(d), "by_bucket": {k: s(g) for (k,), g in d.group_by("bucket")},
           "by_year": {str(k): s(g) for (k,), g in d.with_columns(pl.col("date").dt.year().alias("y")).group_by("y")},
           "by_tenor": {k: s(g) for (k,), g in d.with_columns(tenor.alias("t")).group_by("t")},
           "strikes_like_the_short_put": s(d.filter(deep))}
    out["_hist"] = d["e_iv"].to_numpy()
    return out


def trade_frame(trades: list[dict]) -> pl.DataFrame:
    rows = []
    for t in trades:
        r = {"entry": t["entry"], "event": t["event"], "seg": t["seg"], "week": oc.week_of(t["entry"]), "dte": t["dte"], "S0": t["S0"]}
        if "straddle" in t:
            st = t["straddle"]
            r.update({f"st_{k}": st[k] for k in ("net_prem_unhedged", "net_prem_hedged", "net_btc_prem_unhedged", "net_btc_prem_hedged",
                                                 "prem_pct_S", "rv_14d", "iv_c", "iv_p", "net_usd_unhedged", "net_btc_usd_unhedged")})
            r["st_src_real"] = sum(x == "real" for x in st["src"]) / 2
            r["st_src_exit_real"] = sum(x == "real" for x in st["src_exit"]) / 2
        if "user" in t:
            u = t["user"]
            r.update({f"u_{k}": u[k] for k in ("n_perp", "delta_c", "delta_p", "vega_c", "vega_p", "theta_c", "theta_p", "V_c", "V_p",
                                               "tv_share_p", "iv_c", "iv_p")})
            r["u_src_c"], r["u_src_p"] = u["src_c"], u["src_p"]
            for tag in ("14d", "exp"):
                if tag in u:
                    for k in ("struct_pct", "comp_pct", "diff_pct", "call_pct", "shortput_pct", "perp_pct", "struct_btc", "comp_btc",
                              "struct_btc_usd_pct", "perp_fund_pct", "put_cost_pct", "perp_cost_pct"):
                        r[f"u{tag}_{k}"] = u[tag][k]
                    r[f"u{tag}_src_exit_p"] = u[tag]["src_exit_p"]
        rows.append(r)
    return pl.DataFrame(rows, infer_schema_length=None)


def h1_verdict(diff: float, lo: float, n_ev: int, seg: str) -> str:
    if not np.isfinite(diff) or n_ev == 0:
        return "inconclusive"
    if seg == "clean_holdout":
        if n_ev < 8:
            return "inconclusive"
        return "supports" if diff > 0 else "contradicts"
    if diff > 0 and lo > 0:
        return "supports"
    return "contradicts" if diff <= 0 else "inconclusive"


def opx1(tf: pl.DataFrame) -> dict:
    out = {}
    for seg in oc.SEGS:
        d = tf.filter((pl.col("seg") == seg) & pl.col("st_net_prem_hedged").is_not_null())
        r = {}
        for v in ("hedged", "unhedged"):
            for unit in ("", "btc_"):
                r[f"{unit}{v}"] = oc.boot_diff_cond_uncond(d[f"st_net_{unit}prem_{v}"].to_numpy(), d["event"].to_numpy(), d["week"].to_numpy())
        ev = d.filter(pl.col("event"))
        iv = lambda x: (x["st_iv_c"] + x["st_iv_p"]) / 2
        r["rv_iv_event_median"] = float((ev["st_rv_14d"] / iv(ev)).median()) if ev.height else float("nan")
        r["rv_iv_all_median"] = float((d["st_rv_14d"] / iv(d)).median()) if d.height else float("nan")
        r["iv_event_mean"] = float(iv(ev).mean()) if ev.height else float("nan")
        r["iv_all_mean"] = float(iv(d).mean()) if d.height else float("nan")
        r["share_entry_legs_real"] = float(d["st_src_real"].mean()) if d.height else float("nan")
        r["share_entry_legs_real_events"] = float(ev["st_src_real"].mean()) if ev.height else float("nan")
        r["share_exit_legs_real"] = float(d["st_src_exit_real"].mean()) if d.height else float("nan")
        r["events"] = [{"entry": e.isoformat(), "hedged": h, "unhedged": u} for e, h, u in
                       zip(ev["entry"].to_list(), ev["st_net_prem_hedged"].to_list(), ev["st_net_prem_unhedged"].to_list())]
        h = r["hedged"]
        r["verdict"] = h1_verdict(h["diff"], h["ci95"][0], h["n_event"], seg)
        out[seg] = r
    return out


def opx2(tf: pl.DataFrame) -> dict:
    out = {}
    for seg in oc.SEGS:
        d = tf.filter((pl.col("seg") == seg) & pl.col("u14d_diff_pct").is_not_null())
        ev = d.filter(pl.col("event"))
        r = {"events_14d": oc.boot_mean(ev["u14d_diff_pct"].to_numpy(), ev["week"].to_numpy()),
             "all_days_14d": oc.boot_mean(d["u14d_diff_pct"].to_numpy(), d["week"].to_numpy())}
        de = d.filter(pl.col("uexp_diff_pct").is_not_null())
        dee = de.filter(pl.col("event"))
        r["events_to_expiry"] = oc.boot_mean(dee["uexp_diff_pct"].to_numpy(), dee["week"].to_numpy())
        r["all_days_to_expiry"] = oc.boot_mean(de["uexp_diff_pct"].to_numpy(), de["week"].to_numpy())
        for name, x in (("events", ev), ("all", d)):
            if x.height == 0:
                continue
            lg = {k: float(x[f"u14d_{k}"].mean()) for k in ("struct_pct", "comp_pct", "call_pct", "shortput_pct", "perp_pct",
                                                           "perp_fund_pct", "put_cost_pct", "perp_cost_pct", "struct_btc_usd_pct")}
            lg["share_put_real_entry"] = float((x["u_src_p"] == "real").mean())
            lg["share_call_real_entry"] = float((x["u_src_c"] == "real").mean())
            lg["share_put_real_exit"] = float((x["u14d_src_exit_p"] == "real").mean())
            r[f"legs_{name}"] = lg
        m = r["events_14d"]
        if (seg == "clean_holdout" and m["n"] < 8) or not np.isfinite(m["mean"]):
            v = "inconclusive"
        elif m["mean"] > 0 and m["ci95"][0] > 0:
            v = "supports"
        elif m["mean"] <= 0:
            v = "contradicts"
        else:
            v = "inconclusive"
        r["verdict"] = v
        out[seg] = r
    return out


def greeks(tf: pl.DataFrame) -> dict:
    out = {}
    base = tf.filter(pl.col("u_delta_c").is_not_null() & (pl.col("seg") != "straddles_cutoff"))
    for name, d in (("events", base.filter(pl.col("event"))), ("all_days", base)):
        if d.height == 0:
            continue
        S = d["S0"].to_numpy()
        dc, dp = d["u_delta_c"].to_numpy(), d["u_delta_p"].to_numpy()
        vc, vp = d["u_vega_c"].to_numpy(), d["u_vega_p"].to_numpy()
        tc, tp = d["u_theta_c"].to_numpy(), d["u_theta_p"].to_numpy()
        n = d["u_n_perp"].to_numpy()
        Vc, Vp = d["u_V_c"].to_numpy(), d["u_V_p"].to_numpy()
        tv = d["u_tv_share_p"].to_numpy()

        def row(delta, vega, theta, prem):
            return {"net_delta": float(np.mean(delta)), "vega_usd_per_volpt": float(np.mean(vega)), "vega_pct_S_per_volpt": float(np.mean(vega / S) * 100),
                    "theta_usd_per_day": float(np.mean(theta)), "theta_pct_S_per_day": float(np.mean(theta / S) * 100),
                    "net_premium_pct_S": float(np.mean(prem / S) * 100), "vega_per_unit_delta_pct_S": float(np.mean(vega / S / delta) * 100)}

        out[name] = {"n": d.height, "call_alone": row(dc, vc, tc, Vc), "call_plus_perp": row(dc + n, vc, tc, Vc),
                     "user_structure": row(dc - dp, vc - vp, tc - tp, Vc - Vp), "short_put_alone": row(-dp, -vp, -tp, -Vp),
                     "put_vega_over_call_vega_median": float(np.median(vp / vc)),
                     "short_put_time_value_share": {"mean": float(np.mean(tv)), "median": float(np.median(tv))},
                     "short_put_premium_pct_S": float(np.mean(Vp / S) * 100), "short_put_time_value_pct_S": float(np.mean(Vp * tv / S) * 100),
                     "iv_call_mean": float(np.mean(d["u_iv_c"].to_numpy())), "iv_put_mean": float(np.mean(d["u_iv_p"].to_numpy())),
                     "mean_put_delta": float(np.mean(dp)), "mean_call_delta": float(np.mean(dc))}
    return out


def btc_vs_usd(tf: pl.DataFrame) -> dict:
    out = {}
    base = tf.filter(pl.col("seg") != "straddles_cutoff")
    for name, d in (("events", base.filter(pl.col("event"))), ("all_days", base)):
        x = d.filter(pl.col("st_net_prem_unhedged").is_not_null())
        y = d.filter(pl.col("u14d_struct_pct").is_not_null())
        out[name] = {"straddle_unhedged_usd_pct_prem": float(x["st_net_prem_unhedged"].mean() * 100),
                     "straddle_unhedged_btc_pct_prem": float(x["st_net_btc_prem_unhedged"].mean() * 100),
                     "straddle_hedged_usd_pct_prem": float(x["st_net_prem_hedged"].mean() * 100),
                     "straddle_hedged_btc_pct_prem": float(x["st_net_btc_prem_hedged"].mean() * 100),
                     "user_struct_usd_pct_S": float(y["u14d_struct_pct"].mean()),
                     "user_struct_btc_settled_in_usd_pct_S": float(y["u14d_struct_btc_usd_pct"].mean()),
                     "corr_btc_vs_usd_user": float(np.corrcoef(y["u14d_struct_pct"].to_numpy(), y["u14d_struct_btc_usd_pct"].to_numpy())[0, 1]) if y.height > 2 else float("nan")}
    return out


def opx3() -> dict:
    out = {}
    for sym in ("BTCUSDT", "ETHUSDT"):
        out[sym] = {}
        for H in (1, 3, 7):
            p, _ = pickle.loads((STUDY / f"pred_{sym}_{H}.pkl").read_bytes())
            ev = oc.tail_events(p, H)
            r = {"llm_in_sample": oc.tail_table(ev.filter(pl.col("date") + timedelta(days=2 * H) < CUT)),
                 "clean_holdout": oc.tail_table(ev.filter(pl.col("date") >= CUT))}
            for seg in oc.SEGS:
                r[seg]["call"] = oc.horizon_call(r[seg])
            pf = p.filter(pl.col("s").is_finite())
            r["model"] = {"nu_last": float(pf["nu"][-1]), "p_jump_last": float(pf["p_jump"][-1]), "first_forecast": pf["date"][0].isoformat()}
            ok = pf.filter(pl.col("R").is_finite())
            r["calibration_two_sided_exceedance"] = {str(q): float(((ok["R"] > ok["s"] * ok[f"qz_hi_{q}"]) | (ok["R"] < ok["s"] * ok[f"qz_lo_{q}"])).mean())
                                                     for q in oc.Q_GRID}
            out[sym][H] = r
        for seg in oc.SEGS:
            out[sym][f"verdict_{seg}"] = oc.opx3_verdict([out[sym][H][seg]["call"] for H in (1, 3, 7)])
    return out


def opx4(tb: pl.DataFrame, oa: pl.DataFrame) -> dict:
    tb = tb.with_columns((pl.col("R1") - pl.col("R0")).alias("diff"),
                         pl.when(pl.col("entry_date") < CUT).then(pl.lit("llm_in_sample")).otherwise(pl.lit("clean_holdout")).alias("seg"),
                         pl.col("entry_date").map_elements(oc.week_of, return_dtype=pl.Date).alias("week"))
    out = {"trend": {}, "options": {}}
    for seg in oc.SEGS:
        d = tb.filter(pl.col("seg") == seg)
        b = oc.boot_mean(d["diff"].to_numpy(), d["week"].to_numpy())
        trig = d.filter(pl.col("n_trig") > 0)
        r = {"diff": b, "n_trades": d.height, "mean_R0": float(d["R0"].mean()), "mean_R1": float(d["R1"].mean()),
             "share_triggered": float((d["n_trig"] > 0).mean()), "mean_fills_when_triggered": float(trig["n_fill"].mean()) if trig.height else float("nan"),
             "diff_when_triggered": oc.boot_mean(trig["diff"].to_numpy(), trig["week"].to_numpy()) if trig.height else None,
             "by_strategy": {k: {"n": g.height, "diff_mean": float(g["diff"].mean()), "R0": float(g["R0"].mean()), "R1": float(g["R1"].mean()),
                                 "share_trig": float((g["n_trig"] > 0).mean())} for (k,), g in d.group_by("strategy")}}
        r["verdict"] = oc.verdict_diff(b["mean"], b["ci95"][0], b["n"], seg)
        out["trend"][seg] = r
    if oa.height:
        oa = oa.with_columns((pl.col("R1") - pl.col("R0")).alias("diff"), pl.col("entry").map_elements(oc.week_of, return_dtype=pl.Date).alias("week"))
        for seg in oc.SEGS:
            for book in ("straddle", "user"):
                for sub in ("events", "all"):
                    d = oa.filter((pl.col("seg") == seg) & (pl.col("book") == book))
                    if sub == "events":
                        d = d.filter(pl.col("event"))
                    if d.height == 0:
                        continue
                    tr = d.filter(pl.col("n_trig") > 0)
                    out["options"].setdefault(seg, {})[f"{book}_{sub}"] = {
                        **oc.boot_mean(d["diff"].to_numpy(), d["week"].to_numpy()), "mean_R0": float(d["R0"].mean()), "mean_R1": float(d["R1"].mean()),
                        "share_triggered": float((d["n_trig"] > 0).mean()), "mean_leg_fills_when_triggered": float(tr["n_fill"].mean()) if tr.height else float("nan")}
    return out


def figures(det: pl.DataFrame, tf: pl.DataFrame, o3: dict, me: dict, gk: dict) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIG.mkdir(parents=True, exist_ok=True)
    c = PAL
    plt.rcParams.update({"figure.facecolor": c["surface"], "axes.facecolor": c["surface"], "axes.edgecolor": c["ink2"], "axes.labelcolor": c["ink"],
                         "xtick.color": c["ink2"], "ytick.color": c["ink2"], "text.color": c["ink"], "axes.grid": True, "grid.color": c["grid"],
                         "grid.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False, "font.size": 10})
    paths = []
    d = det.filter(pl.col("pct_rv").is_finite() & pl.col("pct_iv").is_finite())
    fig, ax = plt.subplots(figsize=(10, 4.0))
    x = d["date"].to_list()
    ax.plot(x, d["pct_iv"].to_numpy(), color=c["s1"], lw=1.2, label="30-day ATM IV percentile (trailing 365 d)")
    ax.plot(x, d["pct_rv"].to_numpy(), color=c["s2"], lw=1.0, alpha=0.8, label="7-day RV percentile (trailing 365 d)")
    for e in d.filter(pl.col("event"))["date"].to_list():
        ax.axvline(e, color=c["s3"], lw=1.0, alpha=0.55, zorder=0)
    ax.axvline(CUT, color=c["ink2"], lw=1, ls="--")
    ax.text(CUT, 1.02, " cutoff", color=c["ink2"], fontsize=9, va="bottom")
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("percentile")
    ax.set_title("Compression events (green): 7-day RV in its bottom quintile and 30-day IV in its bottom tercile", loc="left", fontsize=11)
    ax.legend(loc="upper left", bbox_to_anchor=(0, -0.09), frameon=False, fontsize=9, ncol=2)
    fig.tight_layout()
    paths.append(FIG / "detector.png")
    fig.savefig(paths[-1], dpi=130)
    plt.close(fig)

    fig, axs = plt.subplots(1, 2, figsize=(10, 3.6))
    for ax, v, t in zip(axs, ("st_net_prem_hedged", "st_net_prem_unhedged"), ("daily delta-hedged", "unhedged")):
        a = tf.filter(pl.col(v).is_not_null() & (pl.col("seg") != "straddles_cutoff"))
        bins = np.linspace(-1.2, 2.5, 38)
        ax.hist(np.clip(a[v].to_numpy(), -1.2, 2.5), bins=bins, color=c["s1"], alpha=0.55, density=True, label=f"all days (n={a.height})",
                edgecolor=c["surface"], linewidth=0.5)
        e = a.filter(pl.col("event"))[v].to_numpy()
        ax.hist(np.clip(e, -1.2, 2.5), bins=bins, color=c["s2"], alpha=0.85, density=True, label=f"compression events (n={len(e)})",
                edgecolor=c["surface"], linewidth=0.5)
        ax.axvline(0, color=c["ink2"], lw=0.8)
        ax.set_title(f"14-day long ATM straddle, {t}", loc="left", fontsize=11)
        ax.set_xlabel("net P&L / premium paid (clipped to [-1.2, 2.5])")
        ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    paths.append(FIG / "h_opx1_straddle.png")
    fig.savefig(paths[-1], dpi=130)
    plt.close(fig)

    fig, axs = plt.subplots(1, 3, figsize=(11, 3.6), sharey=True)
    for ax, H in zip(axs, (1, 3, 7)):
        for seg, col, mk, off in (("llm_in_sample", c["s1"], "o", -0.08), ("clean_holdout", c["s2"], "s", 0.08)):
            t = o3["BTCUSDT"][H][seg]["by_q"]
            m = np.array([t[q]["mean_z"] for q in oc.Q_GRID])
            tt = np.array([t[q]["t_cluster"] for q in oc.Q_GRID])
            with np.errstate(invalid="ignore", divide="ignore"):
                se = np.where(np.isfinite(tt) & (tt != 0), np.abs(m / tt), np.nan)
            ax.errorbar(np.arange(len(m)) + off, m, yerr=1.96 * se, color=col, marker=mk, ms=6, lw=1.6, capsize=3, label=seg.replace("_", " "))
        ax.axhline(0, color=c["ink2"], lw=0.8)
        ax.set_xticks(range(len(oc.Q_GRID)))
        ax.set_xticklabels([f"q{q}" for q in oc.Q_GRID])
        ax.set_title(f"BTC, H = {H} day{'s' if H > 1 else ''}", loc="left", fontsize=11)
        ax.set_xlabel("move beyond the ex-ante percentile")
    axs[0].set_ylabel("next-H move, same direction (sigma)")
    axs[0].legend(frameon=False, fontsize=9)
    fig.tight_layout()
    paths.append(FIG / "h_opx3_tail_continuation.png")
    fig.savefig(paths[-1], dpi=130)
    plt.close(fig)

    if "events" in gk:
        g = gk["events"]
        names = [("call alone", "call_alone"), ("call + perp", "call_plus_perp"), ("call + short put", "user_structure")]
        fig, axs = plt.subplots(1, 2, figsize=(10, 3.2))
        for ax, key, lab in zip(axs, ("net_delta", "vega_pct_S_per_volpt"), ("net delta (BTC per structure)", "vega (% of spot per vol point)")):
            vals = [g[k][key] for _, k in names]
            ax.barh([n for n, _ in names], vals, color=[c["s1"], c["s2"], c["s3"]], height=0.55)
            for i, v in enumerate(vals):
                ax.text(v, i, f" {v:.3f}", va="center", fontsize=9, color=c["ink"])
            ax.set_title(lab, loc="left", fontsize=11)
            ax.grid(axis="y", visible=False)
            ax.set_xlim(0, max(vals) * 1.25)
        fig.suptitle(f"Entry Greeks at compression events (means over {g['n']} events)", x=0.01, ha="left", fontsize=11)
        fig.tight_layout()
        paths.append(FIG / "greeks.png")
        fig.savefig(paths[-1], dpi=130)
        plt.close(fig)

    h = me["_hist"]
    fig, ax = plt.subplots(figsize=(7, 3.2))
    ax.hist(np.clip(h, -20, 20), bins=np.linspace(-20, 20, 81), color=c["s1"], edgecolor=c["surface"], linewidth=0.4)
    ax.axvline(0, color=c["ink2"], lw=0.8)
    ax.set_xlabel("model IV - trade IV (vol points, clipped to +-20)")
    ax.set_title(f"Smile-model error vs real 08:00 trades (n = {len(h):,})", loc="left", fontsize=11)
    fig.tight_layout()
    paths.append(FIG / "model_error.png")
    fig.savefig(paths[-1], dpi=130)
    plt.close(fig)
    return [str(x.relative_to(ROOT)).replace("\\", "/") for x in paths]


def _strip(o):
    if isinstance(o, dict):
        return {str(k): _strip(v) for k, v in o.items() if not str(k).startswith("_")}
    if isinstance(o, (list, tuple)):
        return [_strip(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return float(o) if np.isfinite(o) else None
    if isinstance(o, (np.integer, np.bool_)):
        return o.item()
    if isinstance(o, (date, datetime)):
        return o.isoformat()
    return o


def analyse(record: bool = False) -> dict:
    det = pl.read_parquet(STUDY / "detector.parquet")
    trades = pickle.loads((STUDY / "trades.pkl").read_bytes())
    tf = trade_frame(trades)
    rq = pl.read_parquet(STUDY / "real_quotes.parquet")
    tb = pl.read_parquet(STUDY / "trend_book.parquet")
    oa = pl.read_parquet(STUDY / "option_arm.parquet")
    dp = json.loads((STUDY / "downside_plan.json").read_text())
    res = {"data": data_summary(), "model_error": model_error(rq)}
    evs = det.filter(pl.col("event"))
    res["events"] = [{"signal_date": d, "entry": e, "pct_rv": a, "pct_iv": b, "iv_src": s, "seg": seg_of(e, e + timedelta(days=oc.HOLD_DAYS))}
                     for d, e, a, b, s in zip(evs["date"].to_list(), evs["entry_date"].to_list(), evs["pct_rv"].to_list(),
                                              evs["pct_iv"].to_list(), evs["iv_src"].to_list())]
    res["trigger_days"] = int(det["trigger"].sum())
    res["n_entry_days"] = {s: int(tf.filter(pl.col("seg") == s).height) for s in ("llm_in_sample", "clean_holdout", "straddles_cutoff")}
    res["H-OPX1"] = opx1(tf)
    res["H-OPX2"] = opx2(tf)
    res["greeks"] = greeks(tf)
    res["btc_vs_usd"] = btc_vs_usd(tf)
    res["H-OPX3"] = opx3()
    res["H-OPX4"] = opx4(tb, oa)
    summ = {"n": len(dp)}
    if dp:
        a = lambda k: np.array([x[k] for x in dp], float)
        summ.update({"mean_struct_pct": float(a("struct_pct").mean()), "mean_struct_plus_plan_pct": float(a("struct_plus_plan_pct").mean()),
                     "mean_plan_pct": float(a("prot_plan_pct").mean()), "mean_prot_hold_pct": float(a("prot_hold_pct").mean()),
                     "median_struct_pct": float(np.median(a("struct_pct"))), "median_struct_plus_plan_pct": float(np.median(a("struct_plus_plan_pct"))),
                     "worst_struct_pct": float(a("struct_pct").min()), "worst_struct_plus_plan_pct": float(a("struct_plus_plan_pct").min()),
                     "best_struct_pct": float(a("struct_pct").max()), "best_struct_plus_plan_pct": float(a("struct_plus_plan_pct").max()),
                     "n_spot_hedged": int(sum(x["spot_hedge_day"] is not None for x in dp))})
    res["downside_plan"] = {"note": "DESCRIPTIVE ONLY - not a registered hypothesis", "events": dp, "summary": summ}
    res["figures"] = figures(det, tf, res["H-OPX3"], res["model_error"], res["greeks"])
    out = _strip(res)
    (ROOT / "research" / "reports" / f"{REPORT}.json").write_text(json.dumps(out, indent=1, default=str), encoding="utf-8")
    if record:
        record_all(out)
    log("analyse done")
    return out


def record_all(out: dict) -> None:
    from ai_trader.research import hypotheses

    o1, o2, o3, o4 = out["H-OPX1"], out["H-OPX2"], out["H-OPX3"], out["H-OPX4"]
    for seg in oc.SEGS:
        m = {k: o1[seg][k] for k in ("hedged", "unhedged", "btc_hedged", "btc_unhedged", "rv_iv_event_median", "rv_iv_all_median",
                                     "iv_event_mean", "iv_all_mean", "share_entry_legs_real", "share_entry_legs_real_events")}
        hypotheses.record("H-OPX1", seg, o1[seg]["verdict"], m, data=DATA_DESC,
                          note="conditional - unconditional mean net P&L / premium, 14-day long ATM straddle (first expiry >= 21 d), ISO-week block bootstrap; verdict on the daily delta-hedged variant")
        m = {k: o2[seg].get(k) for k in ("events_14d", "all_days_14d", "events_to_expiry", "all_days_to_expiry", "legs_events", "legs_all")}
        m["greeks_events"] = out["greeks"].get("events")
        hypotheses.record("H-OPX2", seg, o2[seg]["verdict"], m, data=DATA_DESC,
                          note="paired diff = P&L(short -0.85d put) - P&L(n perp, n = put delta, funding incl.), % of S0, 14-day hold, compression events; ISO-week block bootstrap")
        b = o3["BTCUSDT"]
        e = o3["ETHUSDT"]
        m = {"btc": {str(H): {"call": b[str(H)][seg]["call"], "by_q": b[str(H)][seg]["by_q"], "slope": b[str(H)][seg]["slope"],
                              "slope_ci90": b[str(H)][seg]["slope_ci90"]} for H in (1, 3, 7)},
             "eth_robustness": {str(H): {"call": e[str(H)][seg]["call"], "slope": e[str(H)][seg]["slope"],
                                         "mean_z_q90": e[str(H)][seg]["by_q"]["90"]["mean_z"], "t_q90": e[str(H)][seg]["by_q"]["90"]["t_cluster"]}
                                for H in (1, 3, 7)},
             "eth_verdict": e[f"verdict_{seg}"]}
        hypotheses.record("H-OPX3", seg, b[f"verdict_{seg}"], m, data=DATA_DESC,
                          note="log-HAR-t x weekday (field_study.LogHarT + origin-weekday dummies, monthly expanding refits), daily 08:00 origins, signed next-H return / predictive sd; cluster t (blocks of max(7, 2H) days)")
        m = {"trend": {k: v for k, v in o4["trend"][seg].items() if k != "by_strategy"}, "trend_by_strategy": o4["trend"][seg]["by_strategy"],
             "options": o4["options"].get(seg)}
        hypotheses.record("H-OPX4", seg, o4["trend"][seg]["verdict"], m,
                          data=DATA_DESC + "; trend book = UNIV_L x 8 H-PD6 trend strategies at the strategy-panel as_of",
                          note="per-trade net R: thirds at the first three closes beyond the ex-ante q95 (limit filled only if the next bar trades through) minus own exit; ISO-week cluster bootstrap")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["preds", "trend", "options", "analyse"])
    ap.add_argument("--record", action="store_true")
    a = ap.parse_args()
    from ai_trader.core.time import now

    if a.mode == "preds":
        build_preds(now())
    elif a.mode == "trend":
        build_trend()
    elif a.mode == "options":
        build_options(now())
    else:
        analyse(record=a.record)
