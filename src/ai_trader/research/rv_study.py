"""Relative value studies: H-RV1 (alt residuals vs a BTC/ETH factor model), H-RV2 (Brent-WTI perps),
H-RV3 (XAUT vs XAU perp, weekend split). H-RV4 is forward-only (features.relative_value.basis_excursions).

Design (RESEARCH_PROTOCOL §3b), fixed before any result was seen:
  * Bybit 1h `last` closes. Legs are synchronous: inner join on the bar's event_ts (every leg has its own close
    for that hour; all bars have volume > 0). A 1h return exists only between rows exactly 1 h apart.
  * Everything is computed causally on the full history, then split by segment: in-sample = outcome window ends
    <= 2026-07-01 (knowledge cutoff), holdout = event decided >= cutoff, pooled = all events. Events whose window
    straddles the cutoff are in neither segment (only in pooled).
  * Decision at the close of bar t (`origin` = close_ts of bar t); outcome over the next 24 h (bars t+1..t+24,
    which must be 24 consecutive hours).
  * Costs: 7.5 bps per leg per side, 4 legs -> 30 bps per event (as registered; H-RV1's two hedge instruments are
    still charged as "4 legs"; a beta-weighted cost is shown as a sensitivity).
  * One event per instrument per 24 h: greedy, the next event must start >= 24 h after the previous one.
H-RV1: factor_residuals(window=720, lookback=72): OLS with intercept of the alt's 1h log return on BTC and ETH
    returns over the 720 previous hours (>= 576 finite); z_t = Σ last-72h residuals / (resid std √72). Event
    |z| >= 2. Trade: short the alt if z > 0 (long if z < 0), static dollar hedge b_BTC, b_ETH (fit through t).
    P&L % = -sign(z)·(R_alt - b_BTC R_BTC - b_ETH R_ETH) with simple 24 h returns, minus 30 bps.
    Residual units = -sign(z)·Σ_{t+1..t+24} (r - a - b·f) / (resid std √24). Statistics: per-origin mean across
    alts (cluster by timestamp), then Newey-West t on the time-ordered cluster means (primary), plain t also shown.
    21 alts = universe minus BTC, ETH. Simulated null (selection-bias check): r* = ā + b̄·f + s·e with full-sample
    OLS residuals e and iid random signs s (keeps the factor structure and vol clustering, kills residual
    autocorrelation), run through the same event code, 5 seeds.
H-RV2 / H-RV3: s = ln A - ln B; spread_z(window=720, min_obs=720) (mean/std of the 720 previous synchronous
    hours). Event |z| >= 2. Outcome: toward = -sign(z)·(s_{t+24h} - s_t). Primary = (toward - 30 bps) / std_t
    ("units of the trailing std, net of 4 legs"); t over events (non-overlapping; plain t, NW also shown).
    H-RV3 weekend split: the decision time (origin) falls in the TradFi weekend (research.calendar: Fri 17:00 ET
    -> Sun 18:00 ET). "Revert faster": share of the deviation (s_t - mean_t) reverted after 4 h, weekend vs
    weekday (Welch t), reported; the verdict uses the registered success criterion only.
Verdicts (verdicts._judge): in-sample supports if mean > 0 and t > 2, contradicts if mean <= 0, otherwise
    inconclusive; holdout supports/contradicts on sign only (>= 8 observations). H-RV2's in-sample segment is
    recorded as inconclusive (the registry says it is too short for its own verdict); its verdict is pooled.

Run:  uv run python -m ai_trader.research.rv_study insample|holdout|dry
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta

import numpy as np
import polars as pl

from ai_trader.core.time import now
from ai_trader.features.relative_value import factor_residuals, spread_z
from ai_trader.research.calendar import add_calendar
from ai_trader.research.evaluate import knowledge_cutoff, newey_west_t
from ai_trader.research.verdicts import MIN_HOLDOUT, _judge
from ai_trader.storage.lake import Lake

UNIVERSE = (
    "BTCUSDT ETHUSDT SOLUSDT XRPUSDT DOGEUSDT ADAUSDT LINKUSDT LTCUSDT BCHUSDT AVAXUSDT DOTUSDT BNBUSDT "
    "TRXUSDT NEARUSDT ATOMUSDT ETCUSDT FILUSDT UNIUSDT AAVEUSDT XLMUSDT SUIUSDT ENAUSDT HYPEUSDT"
).split()
ALTS = [s for s in UNIVERSE if s not in ("BTCUSDT", "ETHUSDT")]
COST_LEG = 7.5e-4
COST = 4 * COST_LEG
WINDOW, LOOKBACK, HORIZON, Z_TH = 720, 72, 24, 2.0
H = timedelta(hours=1)
DATA = {"H-RV1": "bybit 1h last, 21 alts vs BTCUSDT+ETHUSDT", "H-RV2": "bybit 1h last BZUSDT, CLUSDT",
        "H-RV3": "bybit 1h last XAUTUSDT, XAUUSDT"}


# ------------------------------------------------------------------ data

def closes(sym: str, as_of: datetime | None = None, lake: Lake | None = None) -> pl.DataFrame:
    b = (lake or Lake()).bars(sym, "1h", as_of or now())
    return b.filter(pl.col("volume") > 0).select("event_ts", "close_ts", pl.col("close").alias(sym)).unique("event_ts").sort("event_ts")


def synchronous(frames: list[pl.DataFrame]) -> pl.DataFrame:
    out = frames[0]
    for f in frames[1:]:
        out = out.join(f.drop("close_ts"), on="event_ts", how="inner")
    return out.sort("event_ts")


def log_ret(px: np.ndarray, ts: np.ndarray) -> np.ndarray:
    r = np.full(len(px), np.nan)
    ok = np.diff(ts) == np.timedelta64(1, "h")
    r[1:] = np.where(ok, np.diff(np.log(px)), np.nan)
    return r


def _decluster(ts: np.ndarray, cand: np.ndarray, gap=np.timedelta64(24, "h")) -> list[int]:
    keep, last = [], None
    for i in np.flatnonzero(cand):
        if last is None or ts[i] - last >= gap:
            keep.append(int(i))
            last = ts[i]
    return keep


def _contig(ts: np.ndarray, i: int, h: int) -> bool:
    return i + h < len(ts) and ts[i + h] - ts[i] == np.timedelta64(h, "h")


# ------------------------------------------------------------------ H-RV1

def rv1_events(sym: str, d: pl.DataFrame, r: np.ndarray | None = None) -> list[dict]:
    """Events for one alt from a synchronous frame with columns event_ts, close_ts, sym, BTCUSDT, ETHUSDT.
    `r` overrides the alt's log returns (simulated null); prices are then rebuilt from it."""
    ts = d["event_ts"].to_numpy()
    pa = d[sym].to_numpy()
    rb, re = log_ret(d["BTCUSDT"].to_numpy(), ts), log_ret(d["ETHUSDT"].to_numpy(), ts)
    if r is None:
        r = log_ret(pa, ts)
    else:
        pa = pa[0] * np.exp(np.nancumsum(np.r_[0.0, r[1:]]))
    F = np.column_stack([rb, re])
    fr = factor_residuals(r, F, WINDOW, LOOKBACK)
    z = fr["z"]
    pb, pe = d["BTCUSDT"].to_numpy(), d["ETHUSDT"].to_numpy()
    close_ts = d["close_ts"].to_list()
    rows = []
    for t in _decluster(ts, np.isfinite(z) & (np.abs(z) >= Z_TH)):
        if not _contig(ts, t, HORIZON):
            continue
        seg = slice(t + 1, t + HORIZON + 1)
        a_n, b_n = fr["alpha_next"][t], fr["beta_next"][t]
        sd_n = fr["resid_std"][t + 1]  # fit through t
        if not (np.isfinite(b_n).all() and np.isfinite(sd_n) and np.isfinite(r[seg]).all()):
            continue
        sgn = -np.sign(z[t])
        R = pa[t + HORIZON] / pa[t] - 1
        RB, RE = pb[t + HORIZON] / pb[t] - 1, pe[t + HORIZON] / pe[t] - 1
        gross = sgn * (R - b_n[0] * RB - b_n[1] * RE)
        res_units = sgn * np.sum(r[seg] - a_n - F[seg] @ b_n) / (sd_n * np.sqrt(HORIZON))
        cost_beta = 2 * COST_LEG * (1 + abs(b_n[0]) + abs(b_n[1]))
        o = close_ts[t]
        rows.append({"symbol": sym, "origin": o, "end": o + timedelta(hours=HORIZON), "z": float(z[t]),
                     "b_btc": float(b_n[0]), "b_eth": float(b_n[1]), "gross": float(gross), "net": float(gross - COST),
                     "net_beta_cost": float(gross - cost_beta), "res_units": float(res_units)})
    return rows


def rv1_panel(as_of: datetime | None = None, null_seed: int | None = None) -> pl.DataFrame:
    lake = Lake()
    base = synchronous([closes("BTCUSDT", as_of, lake), closes("ETHUSDT", as_of, lake)])
    rows = []
    for sym in ALTS:
        d = synchronous([closes(sym, as_of, lake), base])
        if d.height < WINDOW + 100:
            continue
        r = None
        if null_seed is not None:
            ts = d["event_ts"].to_numpy()
            r0 = log_ret(d[sym].to_numpy(), ts)
            F = np.column_stack([log_ret(d["BTCUSDT"].to_numpy(), ts), log_ret(d["ETHUSDT"].to_numpy(), ts)])
            ok = np.isfinite(r0) & np.isfinite(F).all(axis=1)
            X = np.column_stack([np.ones(ok.sum()), F[ok]])
            coef, *_ = np.linalg.lstsq(X, r0[ok], rcond=None)
            e = np.full(len(r0), np.nan)
            e[ok] = r0[ok] - X @ coef
            s = np.random.default_rng(1000 * null_seed + ALTS.index(sym)).choice([-1.0, 1.0], len(r0))
            r = coef[0] + F @ coef[1:] + s * e
        rows += rv1_events(sym, d, r)
    return pl.DataFrame(rows)


def _cluster_stats(df: pl.DataFrame, col: str) -> dict:
    if df.height == 0:
        return {"mean": float("nan"), "t_nw": float("nan"), "t": float("nan"), "n_clusters": 0, "n_events": 0}
    g = df.group_by("origin").agg(pl.col(col).mean()).sort("origin")[col].to_numpy()
    m, t_nw, n = newey_west_t(g)
    t = float(g.mean() / (g.std(ddof=1) / np.sqrt(len(g)))) if len(g) > 2 else float("nan")
    return {"mean": float(m), "t_nw": float(t_nw), "t": t, "n_clusters": int(n), "n_events": df.height}


def rv1_summary(ev: pl.DataFrame) -> dict:
    out = {k: _cluster_stats(ev, k) for k in ("net", "gross", "res_units", "net_beta_cost")}
    out["hit_rate_net"] = float((ev["net"] > 0).mean()) if ev.height else float("nan")
    out["mean_abs_beta_sum"] = float((ev["b_btc"].abs() + ev["b_eth"].abs()).mean()) if ev.height else float("nan")
    out["by_symbol"] = {s: {"n": g.height, "net_pct": 100 * float(g["net"].mean()), "res_units": float(g["res_units"].mean())}
                        for (s,), g in ev.group_by("symbol")}
    return out


# ------------------------------------------------------------------ H-RV2 / H-RV3

def pair_events(a: str, b: str, as_of: datetime | None = None) -> pl.DataFrame:
    lake = Lake()
    d = synchronous([closes(a, as_of, lake), closes(b, as_of, lake)])
    ts = d["event_ts"].to_numpy()
    s = np.log(d[a].to_numpy()) - np.log(d[b].to_numpy())
    sz = spread_z(np.log(d[a].to_numpy()), np.log(d[b].to_numpy()), WINDOW)
    z, mu, sd = sz["z"], sz["mean"], sz["std"]
    pos = {v: i for i, v in enumerate(ts.tolist())}
    step = np.timedelta64(1, "h")
    close_ts = d["close_ts"].to_list()
    rows = []
    for t in _decluster(ts, np.isfinite(z) & (np.abs(z) >= Z_TH)):
        j = pos.get((ts[t] + HORIZON * step).tolist())
        j4 = pos.get((ts[t] + 4 * step).tolist())
        if j is None:
            continue
        sgn = -np.sign(z[t])
        toward = sgn * (s[j] - s[t])
        dev = s[t] - mu[t]
        o = close_ts[t]
        rows.append({"origin": o, "end": o + timedelta(hours=HORIZON), "z": float(z[t]), "std_bps": 1e4 * float(sd[t]),
                     "dev_bps": 1e4 * float(dev), "toward_bps": 1e4 * float(toward), "net_bps": 1e4 * float(toward - COST),
                     "net_sd": float((toward - COST) / sd[t]), "gross_sd": float(toward / sd[t]),
                     "frac_rev_24h": float(1 - (s[j] - mu[t]) / dev),
                     "frac_rev_4h": float(1 - (s[j4] - mu[t]) / dev) if j4 is not None else float("nan")})
    ev = pl.DataFrame(rows)
    if ev.height:
        cal = add_calendar(ev.select(pl.col("origin").alias("event_ts")))
        ev = ev.with_columns(cal["tradfi_weekend"].alias("weekend"))
    return ev


def _t(x: np.ndarray) -> dict:
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 3:
        return {"mean": float(np.mean(x)) if n else float("nan"), "t": float("nan"), "t_nw": float("nan"), "n": n}
    return {"mean": float(x.mean()), "t": float(x.mean() / (x.std(ddof=1) / np.sqrt(n))), "t_nw": newey_west_t(x)[1], "n": n,
            "median": float(np.median(x))}


def pair_summary(ev: pl.DataFrame) -> dict:
    if ev.height == 0:
        return {"net_sd": _t(np.array([])), "n": 0}
    out = {k: _t(ev[k].to_numpy()) for k in ("net_sd", "gross_sd", "net_bps", "toward_bps", "frac_rev_24h", "frac_rev_4h")}
    out["median_std_bps"] = float(ev["std_bps"].median())
    out["mean_abs_dev_bps"] = float(ev["dev_bps"].abs().mean())
    out["first_event"], out["last_event"] = str(ev["origin"].min()), str(ev["origin"].max())
    return out


def weekend_split(ev: pl.DataFrame) -> dict:
    out = {}
    for name, m in (("weekend", True), ("weekday", False)):
        out[name] = pair_summary(ev.filter(pl.col("weekend") == m)) if ev.height else {}
    if ev.height:
        a = ev.filter(pl.col("weekend"))["frac_rev_4h"].drop_nans().drop_nulls().to_numpy()
        b = ev.filter(~pl.col("weekend"))["frac_rev_4h"].drop_nans().drop_nulls().to_numpy()
        if len(a) > 2 and len(b) > 2:
            se = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
            out["frac_rev_4h_diff"] = {"weekend_minus_weekday": float(a.mean() - b.mean()), "welch_t": float((a.mean() - b.mean()) / se)}
    return out


# ------------------------------------------------------------------ segments / verdicts

def segments(df: pl.DataFrame) -> dict[str, pl.DataFrame]:
    if df.height == 0:
        return {"llm_in_sample": df, "clean_holdout": df, "pooled": df}
    k = knowledge_cutoff()
    return {"llm_in_sample": df.filter(pl.col("end") <= k), "clean_holdout": df.filter(pl.col("origin") >= k), "pooled": df}


def verdict_rv1(seg: str, s: dict) -> str:
    m, t, n = s["net"]["mean"], s["net"]["t_nw"], s["net"]["n_clusters"]
    return _judge(seg, m > 0 and np.isfinite(t) and t > 2, m > 0, n)


def verdict_pair(hid: str, seg: str, s: dict) -> str:
    x = s["net_sd"]
    m, t, n = x["mean"], x["t"], x["n"]
    if hid == "H-RV2" and seg == "llm_in_sample":  # registry: in-sample too short for its own verdict
        return "inconclusive"
    if seg == "pooled":  # H-RV2's registered criterion: mean > 0 with t > 2 pooled
        if n < 3:
            return "inconclusive"
        return "supports" if (m > 0 and t > 2) else ("contradicts" if m <= 0 else "inconclusive")
    return _judge(seg, m > 0 and np.isfinite(t) and t > 2, m > 0, n)


def run(segs: tuple[str, ...], as_of: datetime | None = None, with_null: bool = True) -> dict:
    out: dict = {}
    rv1 = rv1_panel(as_of)
    S = segments(rv1)
    for seg in segs:
        s = rv1_summary(S[seg])
        out.setdefault("H-RV1", {})[seg] = (verdict_rv1(seg, s), s)
    if with_null:
        nulls = []  # pair spreads need no null: E[s_{t+h} - s_t | past] = 0 for a martingale, whatever the selection
        for seed in range(5):
            nS = segments(rv1_panel(as_of, null_seed=seed))
            nulls.append({seg: {k: _cluster_stats(nS[seg], k)["mean"] for k in ("gross", "net", "res_units")} | {"n": nS[seg].height} for seg in segs})
        out["H-RV1_null"] = nulls
    for hid, (a, b) in (("H-RV2", ("BZUSDT", "CLUSDT")), ("H-RV3", ("XAUTUSDT", "XAUUSDT"))):
        ev = pair_events(a, b, as_of)
        S = segments(ev)
        for seg in segs:
            s = pair_summary(S[seg])
            if hid == "H-RV3":
                s["weekend_split"] = weekend_split(S[seg])
            out.setdefault(hid, {})[seg] = (verdict_pair(hid, seg, s), s)
    return out


def main(arg: str) -> None:
    from ai_trader.research.hypotheses import record

    segs = {"insample": ("llm_in_sample",), "dry": ("llm_in_sample",), "holdout": ("clean_holdout", "pooled")}[arg]
    res = run(segs)
    for hid in ("H-RV1", "H-RV2", "H-RV3"):
        for seg, (verdict, metrics) in res[hid].items():
            if seg == "pooled" and hid == "H-RV1":
                print(hid, seg, "(context, not recorded)", verdict)
                print(json.dumps({k: v for k, v in metrics.items() if k != "by_symbol"}, default=float, indent=1))
                continue
            if seg == "pooled" and hid == "H-RV3":
                print(hid, seg, "(context, not recorded)", verdict)
                print(json.dumps(metrics, default=float, indent=1))
                continue
            note = ""
            if hid == "H-RV2" and seg == "llm_in_sample":
                note = "registry: in-sample too short for its own verdict; verdict is on pooled"
            if arg != "dry":
                record(hid, seg, verdict, metrics, note=note, data=DATA[hid])
            print(hid, seg, verdict)
            print(json.dumps(metrics, default=float, indent=1))
    if "H-RV1_null" in res:
        print("H-RV1 simulated null (sign-flipped residuals, 5 seeds):")
        print(json.dumps(res["H-RV1_null"], default=float, indent=1))
    print("MIN_HOLDOUT", MIN_HOLDOUT)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "dry")
