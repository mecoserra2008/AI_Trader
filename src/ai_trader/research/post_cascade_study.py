"""Post-liquidation-cascade plays (agent RB, 2026-11-15): H-RB1 rebound long, H-RB2 short the failed rebound,
H-RB3 negative funding + rebuilding OI after a flush. Research only: no orders, no live code.

Design frozen in the diary (2026-11-15, kind decision, "RB design frozen before results") and copied to
data/research/post_cascade/design.json before any outcome is computed. Summary:

  data      universe L (trend_zoo_study.UNIVERSE), market_frame(sym, '4h', as_of) with funding + 4h OI joined on
            known_ts; v0 cell = regimes.label(frame, 4.0); 1h bars only to walk trade paths.
  flush     at 4h close t: r48 = ln(c_t / c_{t-12}) < trailing-365d 5th pct of r48 (window ends at t-1, >= 180 d),
            OI_t / max(OI_{t-17..t}) - 1 <= -5%, cell_t == R9; one per asset per 7 days (first qualifying bar).
  geometry  sigma = 4h sigma_risk_next at the entry bar; structural extreme +/- 0.5 sigma buffer (log units), pulled in
            to 4 sigma if farther, blocked if < 0.5 sigma; target 2R; time stop 120 h; 1h path, stop before target
            within an hour; costs 7.5 bp per side, variant pegged-limit entry 2.0 + 1.3 bp.
  H-RB1     long at the close of t+1; flush low = min low t-11..t+1.
  H-RB2     within 42 bars after t: armed at the first close >= post-flush low * exp(1.5 sigma_t); short at the first
            later cross of the close below the 4h EMA20; stop above the rebound high.
  H-RB3     pattern = first close in t+1..t+18 with funding < trailing-90d 10th pct of prints and OI >= 3% above its
            post-flush low; vol-scaled 24h / 72h return vs no-pattern events anchored at the same lag.
  inference week-cluster bootstrap (5000, seed 20261115), segments split at the knowledge cutoff by entry date.

    PYTHONIOENCODING=utf-8 uv run python -m ai_trader.research.post_cascade_study design
    PYTHONIOENCODING=utf-8 uv run python -m ai_trader.research.post_cascade_study run [--record]
    PYTHONIOENCODING=utf-8 uv run --extra viz python -m ai_trader.research.post_cascade_study report
"""

from __future__ import annotations

import argparse
import json
import math
import os
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from ai_trader.core.config import load
from ai_trader.core.time import now
from ai_trader.research.evaluate import knowledge_cutoff
from ai_trader.research.frame import market_frame
from ai_trader.research.regimes import CELL_LABELS, label
from ai_trader.research.trend_zoo_study import UNIVERSE
from ai_trader.storage.lake import Lake

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "data" / "research" / "post_cascade"
FIG = ROOT / "research" / "figures" / "post_cascade_2026-11-15"
REPORT = ROOT / "research" / "reports" / "post_cascade_2026-11-15"
SEED = 20261115
NBOOT = 5000
CELLS = [f"R{i}" for i in range(1, 10)]

D = {  # frozen design (also written to design.json)
    "universe": UNIVERSE,
    "freq": "4h",
    "r48_bars": 12,
    "r48_pct": 0.05,
    "r48_window_bars": 2190,
    "r48_min_bars": 1080,
    "oi_window_bars": 18,
    "oi_drop": -0.05,
    "cell": "R9",
    "decluster_bars": 42,
    "buffer_sigma": 0.5,
    "cap_sigma": 4.0,
    "min_sigma": 0.5,
    "target_r": 2.0,
    "time_stop_hours": 120,
    "cost_side": 7.5e-4,
    "pegged_entry_cost": 3.3e-4,
    "rb2_rebound_sigma": 1.5,
    "rb2_window_bars": 42,
    "ema_span": 20,
    "rb3_window_bars": 18,
    "rb3_funding_pct": 0.10,
    "rb3_funding_days": 90,
    "rb3_oi_rise": 0.03,
    "rb3_horizons_bars": [6, 18],
    "baseline_exclusion_days": 7,
    "baseline_lookback_bars": 13,
    "baseline_min_match": 5,
    "transition_bars": {"+1d": 6, "+3d": 18, "+7d": 42},
    "path_hours": 168,
    "nboot": NBOOT,
    "seed": SEED,
    "cutoff": "2026-07-01",
}


def seg_of(ts: datetime) -> str:
    return "llm_in_sample" if ts < knowledge_cutoff() else "clean_holdout"


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, default=str), encoding="utf-8")
    os.replace(tmp, path)


# ------------------------------------------------------------------ data
def funding_features(lake: Lake, sym: str, as_of: datetime) -> pl.DataFrame:
    """Funding prints with the trailing-90d 10th percentile and percentile rank (both include the print itself)."""
    f = lake.read_asof("funding", as_of, symbols=[sym], filters={"venue": "bybit"}).sort("event_ts")
    if f.height == 0:
        return pl.DataFrame({"known_ts": [], "f_thr": [], "f_rank": []})
    f = f.with_columns(pl.col("rate").rolling_quantile_by("event_ts", window_size=f"{D['rb3_funding_days']}d",
                                                          quantile=D["rb3_funding_pct"]).alias("f_thr"))
    ts = f["event_ts"].dt.epoch("ms").to_numpy()
    r = f["rate"].to_numpy()
    w = D["rb3_funding_days"] * 86_400_000
    lo = np.searchsorted(ts, ts - w, side="right")
    rank = np.array([np.mean(r[lo[i]:i + 1] <= r[i]) for i in range(len(r))])
    return f.with_columns(pl.Series("f_rank", rank)).select("known_ts", "f_thr", "f_rank")


def load_symbol(sym: str, as_of: datetime, lake: Lake) -> dict | None:
    f = market_frame(sym, "4h", as_of, lake=lake, funding=True, oi=True)
    if f.height < D["r48_min_bars"] + 100:
        return None
    f = label(f, 4.0)
    ff = funding_features(lake, sym, as_of)
    if ff.height:
        r = ff.rename({"known_ts": "_k"}).sort("_k")
        f = f.sort("known_ts").join_asof(r, left_on="known_ts", right_on="_k", strategy="backward").drop("_k").sort("event_ts")
    else:
        f = f.with_columns(pl.lit(None, dtype=pl.Float64).alias("f_thr"), pl.lit(None, dtype=pl.Float64).alias("f_rank"))
    k = D["r48_bars"]
    f = f.with_columns((pl.col("close").log() - pl.col("close").log().shift(k)).alias("r48"))
    f = f.with_columns(
        pl.col("r48").rolling_quantile(D["r48_pct"], window_size=D["r48_window_bars"], min_samples=D["r48_min_bars"]).shift(1).alias("r48_thr"),
        (pl.col("open_interest") / pl.col("open_interest").rolling_max(D["oi_window_bars"], min_samples=D["oi_window_bars"]) - 1).alias("oi_dd"),
        pl.col("close").ewm_mean(span=D["ema_span"], adjust=False).alias("ema20"),
        (pl.col("close").log() - pl.col("close").log().shift(180)).alias("r30d"),
    )
    h = lake.bars(sym, "1h", as_of).select("event_ts", "open", "high", "low", "close")
    h = h.filter((pl.col("open") > 0) & (pl.col("low") > 0) & (pl.col("high") >= pl.max_horizontal("open", "close"))
                 & (pl.col("low") <= pl.min_horizontal("open", "close"))).sort("event_ts")
    return {"sym": sym, "f": f, "h1": {"ts": h["event_ts"].dt.epoch("ms").to_numpy(), "o": h["open"].to_numpy(),
                                       "h": h["high"].to_numpy(), "l": h["low"].to_numpy(), "c": h["close"].to_numpy()}}


def flush_events(f: pl.DataFrame) -> list[int]:
    cond = ((pl.col("r48") < pl.col("r48_thr")) & (pl.col("oi_dd") <= D["oi_drop"]) & (pl.col("cell") == D["cell"])).fill_null(False)
    idx = np.flatnonzero(f.select(cond).to_series().to_numpy())
    ts = f["event_ts"].to_list()
    out, last = [], None
    for i in idx:
        if last is None or ts[i] - ts[last] >= timedelta(hours=4 * D["decluster_bars"]):
            out.append(int(i))
            last = i
    return out


# ------------------------------------------------------------------ trades
def place(d: int, entry: float, extreme: float, sigma: float) -> tuple[float, float, float, bool] | None:
    """Stop beyond the extreme by the buffer, capped / blocked; returns (stop, target, stop_sigma, capped)."""
    if not (np.isfinite(sigma) and sigma > 0 and np.isfinite(extreme)):
        return None
    stop = extreme * math.exp(-d * D["buffer_sigma"] * sigma)
    k = abs(math.log(entry / stop)) / sigma
    if (stop - entry) * d >= 0:  # extreme on the wrong side (cannot happen for valid extremes) -> treat as too close
        return None
    capped = k > D["cap_sigma"]
    if capped:
        k = D["cap_sigma"]
        stop = entry * math.exp(-d * k * sigma)
    if k < D["min_sigma"]:
        return None
    target = entry * math.exp(d * D["target_r"] * k * sigma)
    return stop, target, k, capped


def walk(h1: dict, start_ms: int, d: int, entry: float, stop: float, target: float) -> dict | None:
    """1h path from the entry close: stop before target within an hour; time exit at the 120th hour's close."""
    end = start_ms + D["time_stop_hours"] * 3_600_000
    ts = h1["ts"]
    if len(ts) == 0 or ts[-1] < end - 3_600_000:
        return None  # unresolved: path runs past the data
    i0, i1 = np.searchsorted(ts, start_ms), np.searchsorted(ts, end)
    if i1 <= i0:
        return None
    o, hi, lo, c = h1["o"][i0:i1], h1["h"][i0:i1], h1["l"][i0:i1], h1["c"][i0:i1]
    hs = np.flatnonzero(lo <= stop) if d > 0 else np.flatnonzero(hi >= stop)
    ht = np.flatnonzero(hi >= target) if d > 0 else np.flatnonzero(lo <= target)
    js = hs[0] if hs.size else None
    jt = ht[0] if ht.size else None
    if js is not None and (jt is None or js <= jt):
        fill = min(o[js], stop) if d > 0 else max(o[js], stop)
        j, why = js, "stop"
    elif jt is not None:
        fill, j, why = target, jt, "target"
    else:
        fill, j, why = c[-1], len(c) - 1, "time"
    return {"exit": float(fill), "why": why, "hours": int(j + 1)}


def r_mult(d: int, entry: float, stop: float, fill: float, cin: float, cout: float) -> float:
    return (d * (fill - entry) - cin * entry - cout * fill) / (abs(entry - stop) + entry * (cin + cout))


def trade(s: dict, k: int, d: int, extreme: float) -> dict | None:
    f = s["f"]
    entry = float(f["close"][k])
    sigma = float(f["sigma_risk_next"][k])
    p = place(d, entry, extreme, sigma)
    if p is None:
        return {"blocked": True}
    stop, target, ks, capped = p
    start = int(f["close_ts"][k].timestamp() * 1000)
    w = walk(s["h1"], start, d, entry, stop, target)
    if w is None:
        return {"unresolved": True}
    ts = f["event_ts"][k]
    return {"symbol": s["sym"], "entry_ts": ts, "year": ts.year, "seg": seg_of(ts), "week": ts.strftime("%G-%V"), "side": d,
            "entry": entry, "stop": stop, "target": target, "stop_sigma": ks, "capped": capped, "cell": f["cell"][k],
            "exit_why": w["why"], "hours": w["hours"],
            "R": r_mult(d, entry, stop, w["exit"], D["cost_side"], D["cost_side"]),
            "R_peg": r_mult(d, entry, stop, w["exit"], D["pegged_entry_cost"], D["cost_side"])}


def rb1(s: dict, t: int) -> dict | None:
    f = s["f"]
    k = t + 1
    if k >= f.height:
        return {"unresolved": True}
    low = float(f["low"][max(0, t - 11):k + 1].min())
    return trade(s, k, +1, low)


def rb2(s: dict, t: int) -> dict | None:
    f = s["f"]
    c, lo, hi, ema = (f[x].to_numpy() for x in ("close", "low", "high", "ema20"))
    sig_t = float(f["sigma_risk_next"][t])
    n = f.height
    a0 = max(0, t - 11)
    arm, low_bar = None, None
    for k in range(t + 1, min(n, t + 1 + D["rb2_window_bars"])):
        if arm is None:
            seg = lo[a0:k + 1]
            jl = int(np.argmin(seg)) + a0
            if c[k] >= lo[jl] * math.exp(D["rb2_rebound_sigma"] * sig_t):
                arm, low_bar = k, jl
            continue
        if c[k] < ema[k] and c[k - 1] >= ema[k - 1] and k - 1 >= arm:
            reb_high = float(hi[low_bar:k + 1].max())
            r = trade(s, k, -1, reb_high)
            if r is not None:
                r["arm_lag"] = arm - t
                r["entry_lag"] = k - t
            return r
    if t + D["rb2_window_bars"] >= n:
        return {"unresolved": True}
    return {"no_setup": True}


def rb3(s: dict, t: int) -> dict:
    f = s["f"]
    c = f["close"].to_numpy()
    oi = f["open_interest"].to_numpy()
    fr, thr = f["funding_rate"].to_numpy(), f["f_thr"].to_numpy()
    sig = f["sigma_risk_next"].to_numpy()
    n = f.height
    hs = D["rb3_horizons_bars"]

    def fwd(k: int) -> dict:
        return {f"z{h * 4}h": (math.log(c[k + h] / c[k]) / (sig[k] * math.sqrt(h)) if k + h < n and np.isfinite(sig[k]) else None)
                for h in hs}

    lag = None
    for k in range(t + 1, min(n, t + 1 + D["rb3_window_bars"])):
        if not (np.isfinite(fr[k]) and np.isfinite(thr[k]) and np.isfinite(oi[k])):
            continue
        m = np.nanmin(oi[t:k + 1])
        if fr[k] < thr[k] and oi[k] / m - 1 >= D["rb3_oi_rise"]:
            lag = k - t
            break
    out = {"pattern": lag is not None, "lag": lag}
    if lag is not None:
        out.update(fwd(t + lag))
    else:
        out["by_lag"] = {L: fwd(t + L) for L in range(1, D["rb3_window_bars"] + 1) if t + L < n}
    return out


# ------------------------------------------------------------------ baselines
def baseline(s: dict, events: list[int], rng: np.random.Generator) -> list[dict]:
    f = s["f"]
    ts = f["event_ts"].to_list()
    ev = [ts[i] for i in events]
    excl = timedelta(days=D["baseline_exclusion_days"])
    lb = D["baseline_lookback_bars"]
    dates = f["event_ts"].dt.date().to_numpy()
    ok = np.array(f.select(pl.col("cell").is_not_null() & pl.col("sigma_risk_next").is_finite()).to_series().to_numpy(), dtype=bool)
    ok[:lb] = False
    ev_ms = np.array([int(e.timestamp() * 1000) for e in ev], dtype=np.int64)
    t_ms = f["event_ts"].dt.epoch("ms").to_numpy()
    if ev_ms.size:
        j = np.searchsorted(ev_ms, t_ms)
        near = np.zeros(len(t_ms), bool)
        for jj in (j - 1, j):
            v = (jj >= 0) & (jj < ev_ms.size)
            near[v] |= np.abs(t_ms[v] - ev_ms[jj[v]]) < excl.total_seconds() * 1000
        ok &= ~near
    idx = np.flatnonzero(ok)
    if idx.size == 0:
        return []
    out = []
    _, first = np.unique(dates[idx], return_index=True)
    bounds = list(first) + [idx.size]
    lo, hi = f["low"].to_numpy(), f["high"].to_numpy()
    for a, b in zip(bounds[:-1], bounds[1:], strict=True):
        k = int(idx[a + rng.integers(0, b - a)])
        for d in (+1, -1):
            ext = float(lo[k - lb + 1:k + 1].min()) if d > 0 else float(hi[k - lb + 1:k + 1].max())
            r = trade(s, k, d, ext)
            if r and "R" in r:
                out.append(r)
    return out


# ------------------------------------------------------------------ inference
def _weeks(w: list[str]) -> tuple[np.ndarray, np.ndarray]:
    u, inv = np.unique(np.asarray(w), return_inverse=True)
    return u, inv


def boot_mean(x: np.ndarray, weeks: list[str], rng: np.random.Generator) -> dict:
    if len(x) == 0:
        return {"n": 0}
    u, inv = _weeks(weeks)
    S = np.bincount(inv, weights=x, minlength=len(u))
    C = np.bincount(inv, minlength=len(u)).astype(float)
    m = rng.multinomial(len(u), np.full(len(u), 1 / len(u)), size=NBOOT).astype(float)
    with np.errstate(invalid="ignore", divide="ignore"):
        bs = (m @ S) / (m @ C)
    bs = bs[np.isfinite(bs)]
    return {"n": int(len(x)), "clusters": int(len(u)), "mean": float(np.mean(x)), "lo": float(np.percentile(bs, 2.5)),
            "hi": float(np.percentile(bs, 97.5)), "median": float(np.median(x)), "hit": float(np.mean(x > 0))}


def boot_paired(ev: pl.DataFrame, bl: pl.DataFrame, key: str, col: str, rng: np.random.Generator) -> dict:
    """mean_i (R_i - baseline mean of i's match key); events and baseline resampled by week independently."""
    if ev.height == 0 or bl.height == 0:
        return {"n": int(ev.height)}
    keys = sorted(set(bl[key].to_list()) | set(ev[key].to_list()))
    kix = {k: i for i, k in enumerate(keys)}
    bk = np.array([kix[k] for k in bl[key].to_list()])
    bx = bl[col].to_numpy()
    bu, binv = _weeks(bl["week"].to_list())
    Sb = np.zeros((len(bu), len(keys)))
    Cb = np.zeros((len(bu), len(keys)))
    np.add.at(Sb, (binv, bk), bx)
    np.add.at(Cb, (binv, bk), 1)
    full = Sb.sum(0) / np.maximum(Cb.sum(0), 1)
    full_n = Cb.sum(0)
    ek = np.array([kix[k] for k in ev[key].to_list()])
    ex = ev[col].to_numpy()
    if np.any(full_n[ek] == 0):
        keep = full_n[ek] > 0
        ek, ex = ek[keep], ex[keep]
        weeks = [w for w, k in zip(ev["week"].to_list(), keep, strict=True) if k]
    else:
        weeks = ev["week"].to_list()
    if ex.size == 0:
        return {"n": 0}
    eu, einv = _weeks(weeks)
    point = float(np.mean(ex - full[ek]))
    out = np.empty(NBOOT)
    for r in range(NBOOT):
        mb = rng.multinomial(len(bu), np.full(len(bu), 1 / len(bu))).astype(float)
        s_, c_ = mb @ Sb, mb @ Cb
        b = np.where(c_ > 0, s_ / np.maximum(c_, 1e-12), full)
        me = rng.multinomial(len(eu), np.full(len(eu), 1 / len(eu))).astype(float)
        w = me[einv]
        out[r] = np.sum(w * (ex - b[ek])) / np.sum(w)
    return {"n": int(ex.size), "clusters": int(len(eu)), "mean": point, "lo": float(np.percentile(out, 2.5)),
            "hi": float(np.percentile(out, 97.5)), "baseline_mean": float(np.mean(full[ek])),
            "baseline_n": int(bl.height)}


def match_keys(ev: pl.DataFrame, bl: pl.DataFrame, cell_col: str | None) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Key = asset|cell|seg; fall back to asset|seg where the baseline has < baseline_min_match trades."""
    def k(df: pl.DataFrame, cc: str | None) -> pl.Expr:
        parts = [pl.col("symbol"), pl.col(cc) if cc else pl.lit("R9"), pl.col("seg")]
        return pl.concat_str(parts, separator="|")
    bl = bl.with_columns(k(bl, "cell").alias("k_cell"), pl.concat_str([pl.col("symbol"), pl.col("seg")], separator="|").alias("k_as"))
    ev = ev.with_columns(k(ev, cell_col).alias("k_cell"), pl.concat_str([pl.col("symbol"), pl.col("seg")], separator="|").alias("k_as"))
    cnt = dict(bl.group_by("k_cell").len().iter_rows())
    ev = ev.with_columns(pl.Series("key", [kc if cnt.get(kc, 0) >= D["baseline_min_match"] else ka
                                           for kc, ka in zip(ev["k_cell"].to_list(), ev["k_as"].to_list(), strict=True)]))
    # baseline rows are usable under both key types: duplicate them with the fallback key
    bl = pl.concat([bl.with_columns(pl.col("k_cell").alias("key")), bl.with_columns(pl.col("k_as").alias("key"))])
    return ev, bl


# ------------------------------------------------------------------ run
def design() -> None:
    p = OUT / "design.json"
    if p.exists():
        print("design.json exists (frozen); not overwritten")
        return
    write_json(p, {"frozen_at": now().isoformat(), "design": D})
    print(p)


def run(record: bool) -> dict:
    as_of = now()
    lake = Lake()
    rng = np.random.default_rng(SEED)
    events, t1, t2, t3, base, trans, paths, uncond = [], [], [], [], [], [], [], []
    syms = {}
    for sym in UNIVERSE:
        s = load_symbol(sym, as_of, lake)
        if s is None:
            print(sym, "skipped (short history)")
            continue
        f = s["f"]
        evs = flush_events(f)
        syms[sym] = s
        cells = f["cell"].to_list()
        ts = f["event_ts"].to_list()
        n = f.height
        for t in evs:
            row = f.row(t, named=True)
            e = {"symbol": sym, "t": t, "ts": row["event_ts"], "year": row["event_ts"].year, "seg": seg_of(row["event_ts"]),
                 "week": row["event_ts"].strftime("%G-%V"), "close": row["close"], "r48": row["r48"], "r48_thr": row["r48_thr"],
                 "oi_dd": row["oi_dd"], "funding": row["funding_rate"], "f_rank": row["f_rank"], "sigma": row["sigma_risk_next"],
                 "z48": row["r48"] / (row["sigma_risk_next"] * math.sqrt(12)),
                 "z30d": (row["r30d"] / (row["sigma_risk_next"] * math.sqrt(180))) if row["r30d"] is not None else None}
            for lab, h in D["transition_bars"].items():
                e[f"cell{lab}"] = cells[t + h] if t + h < n and ts[t + h] - ts[t] == timedelta(hours=4 * h) else None
            # next distinct cell after this R9 episode ends
            j = t
            while j < n and cells[j] == "R9":
                j += 1
            e["next_cell"] = cells[j] if j < n else None
            e["r9_bars_left"] = j - t if j < n else None
            # 7-day 1h path from the event close
            h1 = s["h1"]
            st = int(row["close_ts"].timestamp() * 1000)
            i0, i1 = np.searchsorted(h1["ts"], st), np.searchsorted(h1["ts"], st + D["path_hours"] * 3_600_000)
            if h1["ts"].size and h1["ts"][-1] >= st + (D["path_hours"] - 1) * 3_600_000 and i1 > i0:
                lo, hi, cc = h1["l"][i0:i1], h1["h"][i0:i1], h1["c"][i0:i1]
                jl = int(np.argmin(lo))
                e["path"] = {"mdd": math.log(lo[jl] / row["close"]), "hours_to_low": jl + 1,
                             "rebound": math.log(hi[jl:].max() / lo[jl]), "ret7d": math.log(cc[-1] / row["close"]),
                             "ret1d": math.log(cc[23] / row["close"]), "ret3d": math.log(cc[71] / row["close"])}
            else:
                e["path"] = None
            r1, r2, r3 = rb1(s, t), rb2(s, t), rb3(s, t)
            e["rb1"], e["rb2"], e["rb3"] = r1, r2, r3
            events.append(e)
            if r1 and "R" in r1:
                t1.append(r1)
            if r2 and "R" in r2:
                t2.append(r2)
            r3.update({"symbol": sym, "seg": e["seg"], "week": e["week"], "ts": e["ts"]})
            t3.append(r3)
        base += baseline(s, evs, rng)
        # unconditional R9 transition (all R9 4h closes, any time) for comparison with the post-flush matrix
        for t in range(n):
            if cells[t] == "R9":
                rr = {"symbol": sym}
                for lab, h in D["transition_bars"].items():
                    rr[lab] = cells[t + h] if t + h < n else None
                uncond.append(rr)
        print(sym, "events", len(evs), flush=True)

    ev_df = pl.DataFrame([{k: v for k, v in e.items() if k not in ("rb1", "rb2", "rb3", "path")} for e in events])
    T1, T2, BL = pl.DataFrame(t1), pl.DataFrame(t2), pl.DataFrame(base)
    res: dict[str, Any] = {"as_of": as_of.isoformat(), "design": D, "n_events": len(events)}

    # ---- counts
    res["counts_asset_year"] = (ev_df.group_by("symbol", "year").len().sort("symbol", "year")
                                .pivot(on="year", index="symbol", values="len", sort_columns=True).fill_null(0).to_dicts())
    res["counts_seg"] = dict(ev_df.group_by("seg").len().iter_rows())
    res["counts_year"] = dict(ev_df.group_by("year").len().sort("year").iter_rows())
    res["rb1_status"] = _status([e["rb1"] for e in events])
    res["rb2_status"] = _status([e["rb2"] for e in events])

    # ---- H-RB1 / H-RB2
    for hid, T, cell_col, side in (("H-RB1", T1, None, +1), ("H-RB2", T2, "cell", -1)):
        out = {}
        blside = BL.filter(pl.col("side") == side)
        for scope in ("pooled", "BTC"):
            Ts = T if scope == "pooled" else T.filter(pl.col("symbol") == "BTCUSDT")
            Bs = blside if scope == "pooled" else blside.filter(pl.col("symbol") == "BTCUSDT")
            for seg in ("llm_in_sample", "clean_holdout", "all"):
                Tg = Ts if seg == "all" else Ts.filter(pl.col("seg") == seg)
                Bg = Bs if seg == "all" else Bs.filter(pl.col("seg") == seg)
                if Tg.height == 0:
                    out[f"{scope}|{seg}"] = {"n": 0}
                    continue
                ev_k, bl_k = match_keys(Tg, Bg, cell_col)
                o = {"R": boot_mean(Tg["R"].to_numpy(), Tg["week"].to_list(), rng),
                     "R_peg": boot_mean(Tg["R_peg"].to_numpy(), Tg["week"].to_list(), rng),
                     "diff": boot_paired(ev_k, bl_k, "key", "R", rng),
                     "diff_peg": boot_paired(ev_k, bl_k, "key", "R_peg", rng),
                     "baseline_R": boot_mean(Bg["R"].to_numpy(), Bg["week"].to_list(), rng) if Bg.height else {"n": 0},
                     "exits": dict(Tg.group_by("exit_why").len().iter_rows()),
                     "capped_share": float(Tg["capped"].mean()), "stop_sigma_median": float(Tg["stop_sigma"].median()),
                     "hit_target": float((Tg["exit_why"] == "target").mean())}
                if cell_col:
                    o["entry_cells"] = dict(Tg.group_by("cell").len().sort("cell").iter_rows())
                out[f"{scope}|{seg}"] = o
        res[hid] = out

    # ---- H-RB3
    out3 = {}
    for scope in ("pooled", "BTC"):
        rows = [r for r in t3 if scope == "pooled" or r["symbol"] == "BTCUSDT"]
        for seg in ("llm_in_sample", "clean_holdout", "all"):
            rs = [r for r in rows if seg == "all" or r["seg"] == seg]
            out3[f"{scope}|{seg}"] = rb3_stats(rs, rng)
    res["H-RB3"] = out3

    # ---- transitions
    res["transitions"] = {}
    for scope in ("pooled", "BTC"):
        E = ev_df if scope == "pooled" else ev_df.filter(pl.col("symbol") == "BTCUSDT")
        m = {}
        for lab in D["transition_bars"]:
            col = E[f"cell{lab}"].drop_nulls()
            m[lab] = {c: int((col == c).sum()) for c in CELLS} | {"n": int(col.len())}
        col = E["next_cell"].drop_nulls()
        m["next_after_R9"] = {c: int((col == c).sum()) for c in CELLS} | {"n": int(col.len())}
        rl = E["r9_bars_left"].drop_nulls()
        m["r9_hours_left_median"] = float(rl.median() * 4) if rl.len() else None
        m["r9_hours_left_q"] = [float(rl.quantile(q) * 4) for q in (0.25, 0.75)] if rl.len() else None
        U = [u for u in uncond if scope == "pooled" or u["symbol"] == "BTCUSDT"]
        m["unconditional_R9"] = {lab: {c: int(sum(1 for u in U if u[lab] == c)) for c in CELLS} | {"n": sum(1 for u in U if u[lab])}
                                 for lab in D["transition_bars"]}
        res["transitions"][scope] = m

    # ---- path distribution
    res["paths"] = {}
    for scope in ("pooled", "BTC"):
        P = [e["path"] | {"seg": e["seg"]} for e in events if e["path"] and (scope == "pooled" or e["symbol"] == "BTCUSDT")]
        res["paths"][scope] = path_stats(P)

    # ---- the same, split by segment (pooled assets)
    res["transitions_by_seg"], res["paths_by_seg"] = {}, {}
    for seg in ("llm_in_sample", "clean_holdout"):
        E = ev_df.filter(pl.col("seg") == seg)
        m = {}
        for lab in [*D["transition_bars"], "next_cell"]:
            col = E[lab if lab == "next_cell" else f"cell{lab}"].drop_nulls()
            m[lab] = {c: int((col == c).sum()) for c in CELLS} | {"n": int(col.len())}
        res["transitions_by_seg"][seg] = m
        res["paths_by_seg"][seg] = path_stats([e["path"] | {"seg": e["seg"]} for e in events if e["path"] and e["seg"] == seg])

    # ---- current episode + nearest neighbours (descriptive)
    res["current"] = current_episode(syms, events, lake, as_of)

    write_json(OUT / "events.json", [_ser(e) for e in events])
    write_json(OUT / "results.json", res)
    if record:
        record_all(res)
    return res


def _ser(e: dict) -> dict:
    return json.loads(json.dumps(e, default=str))


def _status(rs: list[dict | None]) -> dict:
    out = {"traded": 0, "blocked": 0, "unresolved": 0, "no_setup": 0}
    for r in rs:
        if r is None:
            out["blocked"] += 1
        elif "R" in r:
            out["traded"] += 1
        else:
            for k in ("blocked", "unresolved", "no_setup"):
                if r.get(k):
                    out[k] += 1
    return out


def rb3_stats(rs: list[dict], rng: np.random.Generator) -> dict:
    pat = [r for r in rs if r["pattern"]]
    nop = [r for r in rs if not r["pattern"]]
    out = {"n_events": len(rs), "n_pattern": len(pat), "n_no_pattern": len(nop),
           "lags": sorted(r["lag"] for r in pat)}
    for h in D["rb3_horizons_bars"]:
        z = f"z{h * 4}h"
        P = [r for r in pat if r.get(z) is not None]
        if not P or not nop:
            out[z] = {"n_pattern": len(P)}
            continue
        # control matrix: no-pattern events x lag
        L = D["rb3_window_bars"]
        M = np.full((len(nop), L + 1), np.nan)
        for i, r in enumerate(nop):
            for lag, v in r["by_lag"].items():
                if v.get(z) is not None:
                    M[i, int(lag)] = v[z]
        px = np.array([r[z] for r in P])
        pl_ = np.array([r["lag"] for r in P])
        allw = sorted({r["week"] for r in rs})
        wix = {w: i for i, w in enumerate(allw)}
        pw = np.array([wix[r["week"]] for r in P])
        nw = np.array([wix[r["week"]] for r in nop])

        def stat(wts: np.ndarray) -> float:
            wp, wn = wts[pw], wts[nw]
            if wp.sum() == 0:
                return np.nan
            ok = ~np.isnan(M)
            ctrl = np.array([(np.nansum(M[:, l] * wn) / np.sum(wn * ok[:, l])) if np.sum(wn * ok[:, l]) > 0 else np.nan for l in range(L + 1)])
            d = px - ctrl[pl_]
            m = ~np.isnan(d)
            return float(np.sum(wp[m] * d[m]) / np.sum(wp[m]))

        point = stat(np.ones(len(allw)))
        bs = np.array([stat(rng.multinomial(len(allw), np.full(len(allw), 1 / len(allw))).astype(float)) for _ in range(NBOOT)])
        bs = bs[np.isfinite(bs)]
        ctrl_mean = float(np.nanmean([np.nanmean(M[:, l]) for l in pl_]))
        out[z] = {"n_pattern": len(P), "pattern_mean": float(px.mean()), "control_mean_at_lags": ctrl_mean, "diff": point,
                  "lo": float(np.percentile(bs, 2.5)), "hi": float(np.percentile(bs, 97.5)), "clusters": len(allw)}
    return out


def path_stats(P: list[dict]) -> dict:
    if not P:
        return {"n": 0}
    out = {"n": len(P)}
    for k in ("mdd", "rebound", "hours_to_low", "ret1d", "ret3d", "ret7d"):
        x = np.array([p[k] for p in P], float)
        out[k] = {"q10": float(np.quantile(x, 0.1)), "q25": float(np.quantile(x, 0.25)), "median": float(np.median(x)),
                  "q75": float(np.quantile(x, 0.75)), "q90": float(np.quantile(x, 0.9)), "mean": float(x.mean())}
    out["share_low_in_first_24h"] = float(np.mean([p["hours_to_low"] <= 24 for p in P]))
    out["share_new_low_below_minus5pct"] = float(np.mean([p["mdd"] <= -0.05 for p in P]))
    out["share_rebound_ge_10pct"] = float(np.mean([p["rebound"] >= 0.10 for p in P]))
    out["share_up_7d"] = float(np.mean([p["ret7d"] > 0 for p in P]))
    out["by_seg"] = {s: int(sum(1 for p in P if p["seg"] == s)) for s in ("llm_in_sample", "clean_holdout")}
    out["raw"] = {k: [p[k] for p in P] for k in ("mdd", "rebound", "hours_to_low", "ret7d")}
    return out


def current_episode(syms: dict, events: list[dict], lake: Lake, as_of: datetime) -> dict:
    t0 = datetime(2026, 11, 12, tzinfo=UTC)
    out: dict[str, Any] = {"assets": {}}
    feats = ("z48", "oi_dd", "f_rank", "z30d")
    hist = [e for e in events if e["ts"] < datetime(2026, 11, 8, tzinfo=UTC) and all(e.get(k) is not None and np.isfinite(e[k]) for k in feats)]
    X = np.array([[e[k] for k in feats] for e in hist])
    mu, sd = X.mean(0), X.std(0)
    for sym in ("BTCUSDT", "ETHUSDT", "SOLUSDT"):
        s = syms[sym]
        f = s["f"].filter(pl.col("event_ts") >= t0)
        rows = f.select("event_ts", "close", "r48", "r48_thr", "open_interest", "oi_dd", "cell", "funding_rate", "f_thr", "f_rank",
                        "sigma_risk_next", "r30d").to_dicts()
        ev = [e for e in events if e["symbol"] == sym and e["ts"] >= t0]
        a: dict[str, Any] = {"bars": rows, "flush_events": [{"ts": e["ts"], "close": e["close"], "r48": e["r48"], "r48_thr": e["r48_thr"],
                                                             "oi_dd": e["oi_dd"], "funding": e["funding"], "f_rank": e["f_rank"]} for e in ev]}
        # which conditions held, bar by bar
        a["conditions"] = [{"ts": r["event_ts"], "r48_ok": (r["r48"] is not None and r["r48_thr"] is not None and r["r48"] < r["r48_thr"]),
                            "oi_ok": (r["oi_dd"] is not None and r["oi_dd"] <= D["oi_drop"]), "r9": r["cell"] == "R9"} for r in rows]
        # reference state for neighbours: the flush event if any, else the first bar meeting all three conditions
        # (suppressed by the 7-day decluster), else the R9 bar with the lowest vol-scaled 48h return
        allc = [r for r, cd in zip(rows, a["conditions"], strict=True) if cd["r48_ok"] and cd["oi_ok"] and cd["r9"] and r["event_ts"] >= datetime(2026, 11, 13, tzinfo=UTC)]
        a["bars_meeting_all_conditions"] = [r["event_ts"] for r in allc]
        prev = [e["ts"] for e in events if e["symbol"] == sym and e["ts"] < datetime(2026, 11, 13, tzinfo=UTC)]
        a["previous_event"] = prev[-1] if prev else None
        if ev:
            ref = ev[0]
            q = np.array([ref[k] for k in feats])
            a["nn_reference"] = {"ts": ref["ts"], "kind": "flush event"}
        elif allc:
            r = allc[0]
            q = np.array([r["r48"] / (r["sigma_risk_next"] * math.sqrt(12)), r["oi_dd"], r["f_rank"],
                          r["r30d"] / (r["sigma_risk_next"] * math.sqrt(180))])
            a["nn_reference"] = {"ts": r["event_ts"], "kind": "first bar meeting all flush conditions (suppressed by the 7-day decluster)"}
        else:
            cand = [r for r in rows if r["r48"] is not None and r["sigma_risk_next"]]
            r = min(cand, key=lambda r: r["r48"] / r["sigma_risk_next"])
            q = np.array([r["r48"] / (r["sigma_risk_next"] * math.sqrt(12)), r["oi_dd"], r["f_rank"],
                          r["r30d"] / (r["sigma_risk_next"] * math.sqrt(180))])
            a["nn_reference"] = {"ts": r["event_ts"], "kind": "most extreme 48h bar (no flush event)"}
        a["nn_features"] = dict(zip(feats, map(float, q), strict=True))
        # hypothetical H-RB1 / H-RB3 state of the current event(s), descriptive (unresolved, not in any test)
        a["live_rb1"] = []
        for e in ev:
            t = e["t"]
            ff = s["f"]
            if t + 1 >= ff.height:
                continue
            entry = float(ff["close"][t + 1])
            low = float(ff["low"][max(0, t - 11):t + 2].min())
            p = place(+1, entry, low, float(ff["sigma_risk_next"][t + 1]))
            if p is None:
                continue
            st_ms = int(ff["close_ts"][t + 1].timestamp() * 1000)
            h = s["h1"]
            mm = h["ts"] >= st_ms
            path_low = float(h["l"][mm].min()) if mm.any() else None
            path_high = float(h["h"][mm].max()) if mm.any() else None
            a["live_rb1"].append({"event_ts": e["ts"], "entry_bar": ff["event_ts"][t + 1], "entry": entry, "flush_low": low, "stop": p[0],
                                  "target": p[1], "stop_sigma": p[2], "capped": p[3], "low_since": path_low, "high_since": path_high,
                                  "stopped": path_low is not None and path_low <= p[0], "last": float(h["c"][-1]),
                                  "rb3": {k: v for k, v in e["rb3"].items() if k != "by_lag"}})
        dist = np.sqrt((((X - q) / sd) ** 2).sum(1))
        order = np.argsort(dist)[:5]
        a["neighbours"] = [{"symbol": hist[i]["symbol"], "ts": hist[i]["ts"], "dist": float(dist[i]),
                            **{k: hist[i][k] for k in feats}, "path": hist[i]["path"],
                            "rb1_R": hist[i]["rb1"].get("R") if hist[i]["rb1"] else None,
                            "rb1_exit": hist[i]["rb1"].get("exit_why") if hist[i]["rb1"] else None,
                            "rb2_R": hist[i]["rb2"].get("R") if hist[i]["rb2"] else None,
                            "rb3_pattern": hist[i]["rb3"]["pattern"], "cell+7d": hist[i]["cell+7d"], "next_cell": hist[i]["next_cell"]}
                           for i in order]
        btc_hist = [i for i in range(len(hist)) if hist[i]["symbol"] == sym]
        if btc_hist:
            bo = sorted(btc_hist, key=lambda i: dist[i])[:3]
            a["neighbours_same_asset"] = [{"ts": hist[i]["ts"], "dist": float(dist[i]), **{k: hist[i][k] for k in feats},
                                           "path": hist[i]["path"], "rb1_R": hist[i]["rb1"].get("R") if hist[i]["rb1"] else None,
                                           "cell+7d": hist[i]["cell+7d"]} for i in bo]
        # liquidation prints (descriptive only), hourly by liquidated side, USD notional
        lq = lake.read_asof("liquidations", as_of, symbols=[sym], start=datetime(2026, 11, 13, tzinfo=UTC), filters={"venue": "bybit"})
        if lq.height:
            hq = (lq.with_columns((pl.col("price") * pl.col("size")).alias("usd"), pl.col("event_ts").dt.truncate("4h").alias("h4"))
                  .group_by("h4", "side").agg(pl.col("usd").sum(), pl.len().alias("prints")).sort("h4"))
            a["liq_4h"] = hq.to_dicts()
            a["liq_total"] = dict(lq.with_columns((pl.col("price") * pl.col("size")).alias("usd")).group_by("side").agg(pl.col("usd").sum()).iter_rows())
            hh = (lq.with_columns((pl.col("price") * pl.col("size")).alias("usd"), pl.col("event_ts").dt.truncate("1h").alias("h1"))
                  .group_by("h1", "side").agg(pl.col("usd").sum()).sort("h1"))
            a["liq_1h"] = hh.to_dicts()
        h = s["h1"]
        m = h["ts"] >= int(datetime(2026, 11, 12, tzinfo=UTC).timestamp() * 1000)
        a["h1"] = {"ts": h["ts"][m].tolist(), "c": h["c"][m].tolist(), "l": h["l"][m].tolist(), "h": h["h"][m].tolist()}
        out["assets"][sym] = a
    out["nn_note"] = "descriptive; reference episode chosen after the fact; features z-scored over historical events before 2026-11-08"
    out["nn_feature_mean_sd"] = {"mean": dict(zip(feats, map(float, mu), strict=True)), "sd": dict(zip(feats, map(float, sd), strict=True))}
    return out


# ------------------------------------------------------------------ verdicts
def verdict_rb12(o: dict) -> str:
    R, d = o.get("R", {}), o.get("diff", {})
    if not R.get("n") or "lo" not in R or "lo" not in d:
        return "inconclusive"
    if R["mean"] > 0 and R["lo"] > 0 and d["mean"] > 0 and d["lo"] > 0:
        return "supports"
    if (R["mean"] <= 0 and R["hi"] < 0) or (d["mean"] <= 0 and d["hi"] < 0):
        return "contradicts"
    return "inconclusive"


def verdict_rb3(o: dict) -> str:
    hs = [o.get(f"z{h * 4}h", {}) for h in D["rb3_horizons_bars"]]
    if any(x.get("lo", -1) > 0 and x.get("diff", 0) > 0 for x in hs):
        return "supports"
    if all("hi" in x and x["diff"] <= 0 and x["hi"] < 0 for x in hs):
        return "contradicts"
    return "inconclusive"


def _slim(o: dict) -> dict:
    return json.loads(json.dumps(o, default=float))


def record_all(res: dict) -> None:
    from ai_trader.research.hypotheses import record

    data = "research/reports/post_cascade_2026-11-15.json"
    for hid in ("H-RB1", "H-RB2"):
        for seg in ("llm_in_sample", "clean_holdout"):
            o = res[hid][f"pooled|{seg}"]
            v = verdict_rb12(o)
            m = {"pooled": _slim(o), "btc": _slim(res[hid].get(f"BTC|{seg}", {})), "n_events_total": res["n_events"],
                 "trials": "3 hypotheses x 2 segments x 2 cost variants; verdict on 7.5 bp/side"}
            record(hid, seg, v, m, note="agent RB post-cascade study; universe L 4h, week-cluster bootstrap; pegged-limit variant in R_peg/diff_peg", data=data)
        o = res[hid]["pooled|all"]
        record(hid, "pooled", verdict_rb12(o), {"pooled": _slim(o)}, note="both segments together (descriptive)", data=data)
    for seg in ("llm_in_sample", "clean_holdout"):
        o = res["H-RB3"][f"pooled|{seg}"]
        record("H-RB3", seg, verdict_rb3(o), {"pooled": _slim(o), "btc": _slim(res["H-RB3"].get(f"BTC|{seg}", {}))},
               note="agent RB; pattern vs no-pattern flush events, controls anchored at the same lag", data=data)
    o = res["H-RB3"]["pooled|all"]
    record("H-RB3", "pooled", verdict_rb3(o), {"pooled": _slim(o)}, note="both segments together (descriptive)", data=data)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["design", "run", "report"])
    ap.add_argument("--record", action="store_true")
    a = ap.parse_args()
    if a.cmd == "design":
        design()
    elif a.cmd == "run":
        if not (OUT / "design.json").exists():
            raise SystemExit("freeze the design first (design)")
        res = run(a.record)
        print(json.dumps({k: res[k] for k in ("n_events", "counts_seg", "rb1_status", "rb2_status")}, default=str))
    else:
        from ai_trader.research.post_cascade_report import build

        build()


if __name__ == "__main__":
    main()
