"""Event studies for the vector-candle hypotheses H-VC1, H-VC2, H-VC3, H-VC5 (research/hypotheses.yaml).

Design (RESEARCH_PROTOCOL §3b): features, events and forward outcomes are computed on each symbol's FULL
1h history; only then is the event panel split at the knowledge cutoff by the event bar's event_ts.

Statistics: events sharing an event_ts (several symbols firing in the same hour) are first averaged into one
observation per timestamp ("cluster by timestamp"); the t-stat is Newey-West over that time-ordered series.
Rates vs a fixed null use a binomial z on the raw events (cluster t reported alongside).
"""

from __future__ import annotations

from datetime import datetime

import numpy as np
import polars as pl
from scipy import stats

from ai_trader.features.candles import vector_candles, vector_events
from ai_trader.research.calendar import gap_fill_baseline
from ai_trader.research.evaluate import knowledge_cutoff, newey_west_t
from ai_trader.research.frame import market_frame
from ai_trader.research.regimes import label

UNIVERSE = (
    "BTCUSDT ETHUSDT SOLUSDT XRPUSDT DOGEUSDT ADAUSDT LINKUSDT LTCUSDT BCHUSDT AVAXUSDT DOTUSDT BNBUSDT "
    "TRXUSDT NEARUSDT ATOMUSDT ETCUSDT FILUSDT UNIUSDT AAVEUSDT XLMUSDT SUIUSDT ENAUSDT HYPEUSDT"
).split()
HORIZONS = (4, 24)
REVISIT_BARS = 72
TOUCH_BARS = 168
RACE_BARS = 168
FLOW_MIN_IMB = 0.2


# ------------------------------------------------------------------ per-event outcomes

def _vc3_outcome(i: int, d: int, edge: float, hi: np.ndarray, lo: np.ndarray, close: np.ndarray, sig: np.ndarray) -> tuple[str, int | None]:
    """First touch of the near edge within TOUCH_BARS after the event; then race on 1h closes from the touch bar on.

    Up vector: edge = zone top (its close); touch when a later bar's low <= edge; reject = close >= edge*e^{+s},
    break = close <= edge*e^{-s}. Down vector mirrors. s = sigma_prev of the touch bar.
    Returns (outcome, touch index); outcome in {reject, break, unresolved, no_touch, no_data}."""
    n = len(close)
    end = min(n, i + 1 + TOUCH_BARS)
    if i + 1 >= n:
        return "no_data", None
    seg = lo[i + 1:end] <= edge if d > 0 else hi[i + 1:end] >= edge
    hits = np.flatnonzero(seg)
    if len(hits) == 0:
        return ("no_touch" if end - (i + 1) == TOUCH_BARS else "no_data"), None
    j = i + 1 + int(hits[0])
    s = sig[j]
    if not np.isfinite(s) or s <= 0:
        return "no_data", j
    up, dn = np.log(edge) + s, np.log(edge) - s
    lc = np.log(close[j:min(n, j + RACE_BARS)])
    rej = np.flatnonzero(lc >= up) if d > 0 else np.flatnonzero(lc <= dn)
    brk = np.flatnonzero(lc <= dn) if d > 0 else np.flatnonzero(lc >= up)
    r = rej[0] if len(rej) else None
    b = brk[0] if len(brk) else None
    if r is None and b is None:
        return ("unresolved" if len(lc) == RACE_BARS else "no_data"), j
    if b is None or (r is not None and r < b):
        return "reject", j
    return "break", j


def symbol_events(symbol: str, as_of: datetime) -> pl.DataFrame:
    """All declustered vector candles for one symbol with every forward outcome (labels only)."""
    f = market_frame(symbol, "1h", as_of, flow=True, funding=False, oi=False)
    if f.height < 400:
        return pl.DataFrame()
    f = vector_candles(label(f, bar_hours=1.0))
    ev = vector_events(f)
    if ev.height == 0:
        return pl.DataFrame()
    close, hi, lo, sig = (f[c].to_numpy().astype(float) for c in ("close", "high", "low", "sigma_prev"))
    lc = np.log(close)
    n = len(close)
    rows = []
    for e in ev.iter_rows(named=True):
        i, d, s = e["i"], int(e["direction"]), e["sigma_prev"]
        r = {"symbol": symbol, "event_ts": e["event_ts"], "i": i, "d": d, "sigma_prev": s, "range_sig": e["range_sig"],
             "vol_rel": e["vol_rel"], "body": e["body"], "thin": e["thin"], "vol_state": e["vol_state"], "cell": e["cell"],
             "flow_imbalance": e.get("flow_imbalance")}
        for h in HORIZONS:
            r[f"sz_{h}"] = d * (lc[i + h] - lc[i]) / (s * np.sqrt(h)) if i + h < n and s > 0 else np.nan
        # H-VC2: zone midpoint revisit within 72 bars, bars strictly after the event bar
        mid = e["zone_mid"]
        if i + REVISIT_BARS < n and s > 0:
            w_lo, w_hi = lo[i + 1:i + 1 + REVISIT_BARS], hi[i + 1:i + 1 + REVISIT_BARS]
            r["revisit"] = float(w_lo.min() <= mid <= w_hi.max())  # traded through mid (gaps count as crossing)
            dist = abs(np.log(close[i] / mid)) / s
            r["dist_sig"] = dist
            r["revisit_base"] = float(gap_fill_baseline(np.array([dist]), np.array([REVISIT_BARS]))[0])
        else:
            r["revisit"] = r["dist_sig"] = r["revisit_base"] = np.nan
        edge = e["zone_hi"] if d > 0 else e["zone_lo"]
        out, j = _vc3_outcome(i, d, edge, hi, lo, close, sig) if d != 0 else ("no_data", None)
        r["vc3"] = out
        r["touch_lag"] = (j - i) if j is not None else None
        rows.append(r)
    return pl.DataFrame(rows, infer_schema_length=None)


def event_panel(symbols: list[str] | tuple[str, ...] = UNIVERSE, as_of: datetime | None = None) -> pl.DataFrame:
    from ai_trader.core.time import now

    as_of = as_of or now()
    parts = [p for s in symbols if (p := symbol_events(s, as_of)).height]
    k = knowledge_cutoff()
    panel = pl.concat(parts, how="diagonal_relaxed")
    return panel.with_columns(
        pl.when(pl.col("event_ts") < k).then(pl.lit("llm_in_sample")).otherwise(pl.lit("clean_holdout")).alias("segment")
    ).sort("event_ts", "symbol")


# ------------------------------------------------------------------ statistics

def cluster_t(panel: pl.DataFrame, col: str) -> dict:
    """Average `col` across events at the same event_ts, then NW mean / t over the time-ordered clusters."""
    g = panel.filter(pl.col(col).is_not_null() & pl.col(col).is_not_nan()).group_by("event_ts").agg(pl.col(col).mean()).sort("event_ts")
    x = g[col].to_numpy()
    m, t, n = newey_west_t(x)
    if n < 10 and n >= 2:  # NW needs >= 10: fall back to the iid t
        t = float(x.mean() / (x.std(ddof=1) / np.sqrt(n))) if x.std(ddof=1) > 0 else float("nan")
    n_ev = panel.filter(pl.col(col).is_not_null() & pl.col(col).is_not_nan()).height
    return {"mean": m, "t": t, "n_clusters": n, "n_events": n_ev}


def binom_z(k: int, n: int, p0: float = 0.5) -> float:
    return float((k - n * p0) / np.sqrt(n * p0 * (1 - p0))) if n else float("nan")


def vc1(panel: pl.DataFrame) -> dict:
    out = {f"h{h}": cluster_t(panel, f"sz_{h}") for h in HORIZONS}
    out["p_continue_24"] = float((panel["sz_24"].drop_nans().drop_nulls() > 0).mean()) if panel.height else float("nan")
    return out


def vc1_split(panel: pl.DataFrame, by: str) -> pl.DataFrame:
    rows = []
    for key in sorted({v for v in panel[by].to_list() if v is not None}, key=str):
        sub = panel.filter(pl.col(by) == key)
        for h in HORIZONS:
            c = cluster_t(sub, f"sz_{h}")
            rows.append({by: str(key), "h": h, **c})
    return pl.DataFrame(rows)


def vc2(panel: pl.DataFrame) -> dict:
    p = panel.filter(pl.col("revisit").is_not_nan()).with_columns((pl.col("revisit") - pl.col("revisit_base")).alias("excess"))
    c = cluster_t(p, "excess")
    return {"rate": float(p["revisit"].mean()) if p.height else float("nan"),
            "baseline": float(p["revisit_base"].mean()) if p.height else float("nan"),
            "excess": c["mean"], "t": c["t"], "n_events": p.height, "n_clusters": c["n_clusters"],
            "median_dist_sig": float(p["dist_sig"].median()) if p.height else float("nan")}


def vc3(panel: pl.DataFrame) -> dict:
    counts = {k: int((panel["vc3"] == k).sum()) for k in ("reject", "break", "unresolved", "no_touch", "no_data")}
    res = panel.filter(pl.col("vc3").is_in(["reject", "break"])).with_columns((pl.col("vc3") == "reject").cast(pl.Float64).alias("rej"))
    k, n = int(res["rej"].sum()), res.height
    c = cluster_t(res.with_columns(pl.col("rej") - 0.5), "rej")
    lag = res["touch_lag"].drop_nulls()
    return {"reject_rate": k / n if n else float("nan"), "n_resolved": n, "binom_z": binom_z(k, n),
            "cluster_t": c["t"], **counts, "touch_lag_median": float(lag.median()) if len(lag) else float("nan"),
            "touch_next_bar_share": float((lag == 1).mean()) if len(lag) else float("nan")}


def vc5(panel: pl.DataFrame) -> dict:
    p = panel.filter(pl.col("symbol").is_in(["BTCUSDT", "ETHUSDT"]) & pl.col("flow_imbalance").is_not_null() & pl.col("sz_24").is_not_nan())
    p = p.with_columns(((pl.col("flow_imbalance").sign() == pl.col("d")) & (pl.col("flow_imbalance").abs() >= FLOW_MIN_IMB)).alias("aligned"))

    def clustered(sub: pl.DataFrame) -> np.ndarray:
        return sub.group_by("event_ts").agg(pl.col("sz_24").mean())["sz_24"].to_numpy()

    a, u = clustered(p.filter(pl.col("aligned"))), clustered(p.filter(~pl.col("aligned")))
    out = {"n_aligned": int(p["aligned"].sum()), "n_unaligned": int((~p["aligned"]).sum()),
           "mean_aligned": float(a.mean()) if len(a) else float("nan"), "mean_unaligned": float(u.mean()) if len(u) else float("nan")}
    if len(a) >= 2 and len(u) >= 2:
        t = stats.ttest_ind(a, u, equal_var=False)
        out.update(diff=float(a.mean() - u.mean()), t=float(t.statistic))
    else:
        out.update(diff=float("nan"), t=float("nan"))
    out["n_events"] = p.height
    return out


# ------------------------------------------------------------------ verdicts (mechanical, from the registry)

def verdict_vc1(r: dict, segment: str) -> str:
    h = r["h24"]
    if segment == "clean_holdout":
        return "inconclusive" if h["n_events"] < 8 else ("supports" if h["mean"] > 0 else "contradicts")
    return "supports" if h["mean"] > 0 and h["t"] > 2 else "contradicts"


def verdict_vc2(r: dict, segment: str) -> str:
    if segment == "clean_holdout":
        return "inconclusive" if r["n_events"] < 8 else ("supports" if r["excess"] > 0 else "contradicts")
    return "supports" if r["excess"] > 0.05 and r["t"] > 2 else "contradicts"


def verdict_vc3(r: dict, segment: str) -> str:
    if segment == "clean_holdout":
        return "inconclusive" if r["n_resolved"] < 8 else ("supports" if r["reject_rate"] > 0.5 else "contradicts")
    return "supports" if r["reject_rate"] > 0.55 and r["binom_z"] > 2 else "contradicts"


def verdict_vc5(r: dict, segment: str) -> str:
    if r["n_aligned"] + r["n_unaligned"] < 8 or r["n_aligned"] < 2 or r["n_unaligned"] < 2:
        return "inconclusive"
    return "supports" if r["diff"] > 0 and r["t"] > 2 else "contradicts"
