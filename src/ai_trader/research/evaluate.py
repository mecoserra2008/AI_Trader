"""Deterministic evaluation of strategy signals by regime cell and around shocks.

Units: **risk units**. A signal s_t held over bar t+1 is sized as s_t / σ̂_{t+1|t}, so
    pnl_t = L_t · fwd_ret_t − cost · |L_t − L_{t−1}|,   L = s / σ̂ fixed at entry (see strategy_pnl)
which makes results comparable across assets, frequencies and volatility regimes. A strategy
earning 0.02 risk units per active bar earns 2% of one bar's σ per bar.

Statistics follow the house rules: Newey–West HAC t-stats (overlapping, autocorrelated P&L),
effective sample = number of regime *episodes*, empirical-Bayes shrinkage of cell means toward
the strategy's pooled mean, and an explicit split at the LLM knowledge cutoff.
"""

from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
import polars as pl

from ai_trader.core.config import load


def knowledge_cutoff() -> datetime:
    return datetime.fromisoformat(load().settings["research"]["knowledge_cutoff"]).replace(tzinfo=UTC)


def newey_west_t(x: np.ndarray) -> tuple[float, float, int]:
    """Mean, HAC t-stat (Bartlett kernel, Newey–West automatic lag), n."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 10:
        return (float(np.mean(x)) if n else float("nan"), float("nan"), n)
    m = x.mean()
    e = x - m
    lag = int(np.floor(4 * (n / 100) ** (2 / 9)))
    var = e @ e / n
    for k in range(1, lag + 1):
        var += 2 * (1 - k / (lag + 1)) * (e[k:] @ e[:-k]) / n
    se = np.sqrt(max(var, 1e-300) / n)
    return float(m), float(m / se), n


def episodes(cells: pl.Series) -> dict[str, int]:
    """Number of contiguous runs per cell (the effective sample size of regime statistics)."""
    v = cells.to_list()
    out: dict[str, int] = {}
    prev = None
    for c in v:
        if c is not None and c != prev:
            out[c] = out.get(c, 0) + 1
        prev = c
    return out


def strategy_pnl(frame: pl.DataFrame, signal: pl.Series, cost_bps: float | None = None, rebalance_band: float = 0.2) -> pl.DataFrame:
    """Per-bar P&L in risk units. Requires fwd_ret and sigma_next (market_frame provides both).

    Execution policy: a position is sized (s / σ̂) when it is opened and held at constant notional;
    it is re-sized only when the signal flips sign, goes flat, or moves by more than
    `rebalance_band`. That's how a real trade behaves: no costs are charged just because σ̂ drifts."""
    if cost_bps is None:
        c = load().risk["costs"]
        cost_bps = c["taker_bps"] + c["slippage_bps"]  # per unit of turnover, one side
    s = np.nan_to_num(signal.to_numpy().astype(float))
    sig = frame["sigma_next"].to_numpy()
    lev = np.zeros(len(s))
    held_s, held_lev = 0.0, 0.0
    for t in range(len(s)):
        st = s[t]
        if st == 0.0:
            held_s, held_lev = 0.0, 0.0
        elif held_s == 0.0 or np.sign(st) != np.sign(held_s) or abs(st - held_s) > rebalance_band:
            held_s = st
            held_lev = st / sig[t] if np.isfinite(sig[t]) and sig[t] > 0 else 0.0
        lev[t] = held_lev
    turnover = np.abs(np.diff(np.r_[0.0, lev]))
    gross = lev * np.nan_to_num(frame["fwd_ret"].to_numpy())
    cost = turnover * cost_bps / 1e4
    return frame.select("event_ts").with_columns(
        pl.Series("signal", s),
        pl.Series("gross", gross),
        pl.Series("cost", cost),
        pl.Series("pnl", gross - cost),
        pl.Series("active", lev != 0),
    )


def _summ(p: np.ndarray, active: np.ndarray, bars_per_year: float) -> dict:
    mean, t, n = newey_west_t(p[active]) if active.any() else (float("nan"), float("nan"), 0)
    allp = p[np.isfinite(p)]
    sd = allp.std() if len(allp) > 1 else float("nan")
    return {
        "n_active": int(active.sum()),
        "exposure": float(active.mean()) if len(active) else float("nan"),
        "mean_active": mean,
        "t_nw": t,
        "hit_rate": float((p[active] > 0).mean()) if active.any() else float("nan"),
        "sharpe_ann": float(allp.mean() / sd * np.sqrt(bars_per_year)) if sd and sd > 0 else float("nan"),
        "total": float(allp.sum()),
    }


def by_regime(pnl: pl.DataFrame, cells: pl.Series, bar_hours: float, kappa: float | None = None) -> pl.DataFrame:
    """Per-cell statistics + empirical-Bayes shrunk mean (toward the pooled active mean)."""
    rs = load().settings["research"]
    kappa = rs["shrinkage_kappa"] if kappa is None else kappa
    bpy = 24 * 365 / bar_hours
    df = pnl.with_columns(cells.alias("cell"))
    p_all, a_all = df["pnl"].to_numpy(), df["active"].to_numpy()
    pooled = _summ(p_all, a_all, bpy)
    eps = episodes(df["cell"])
    rows = []
    for c in sorted(x for x in df["cell"].unique().to_list() if x is not None):
        sub = df.filter(pl.col("cell") == c)
        s = _summ(sub["pnl"].to_numpy(), sub["active"].to_numpy(), bpy)
        n_eff = eps.get(c, 0)
        w = n_eff / (n_eff + kappa)
        shrunk = w * s["mean_active"] + (1 - w) * pooled["mean_active"] if np.isfinite(s["mean_active"]) else pooled["mean_active"]
        rows.append({"cell": c, "episodes": n_eff, "bars": sub.height, **s, "shrink_w": w, "mean_shrunk": shrunk})
    rows.append({"cell": "ALL", "episodes": sum(eps.values()), "bars": df.height, **pooled, "shrink_w": 1.0, "mean_shrunk": pooled["mean_active"]})
    return pl.DataFrame(rows)


def split_by_cutoff(df: pl.DataFrame) -> dict[str, pl.DataFrame]:
    """`llm_in_sample` (before the knowledge cutoff: my training data may encode this period) vs
    `clean_holdout` (after it: genuinely out of sample for any hypothesis the LLM proposes)."""
    k = knowledge_cutoff()
    return {"llm_in_sample": df.filter(pl.col("event_ts") < k), "clean_holdout": df.filter(pl.col("event_ts") >= k)}


# ---------------------------------------------------------------- forward paths & shocks

def forward_paths(frame: pl.DataFrame, horizons: list[int]) -> pl.DataFrame:
    """For each bar t: cumulative forward return at each horizon h in units of σ̂_t·√h, plus
    max favourable / adverse excursion (MFE/MAE) over h in units of σ̂_t. Labels only."""
    lc = np.log(frame["close"].to_numpy())
    hi = np.log(frame["high"].to_numpy())
    lo = np.log(frame["low"].to_numpy())
    sig = frame["sigma_next"].to_numpy()
    n = len(lc)
    out = {}
    for h in horizons:
        fwd = np.full(n, np.nan)
        mfe = np.full(n, np.nan)
        mae = np.full(n, np.nan)
        if n > h:
            fwd[: n - h] = (lc[h:] - lc[: n - h]) / (sig[: n - h] * np.sqrt(h))
            win_hi = np.lib.stride_tricks.sliding_window_view(hi[1:], h).max(axis=1)
            win_lo = np.lib.stride_tricks.sliding_window_view(lo[1:], h).min(axis=1)
            m = len(win_hi)
            mfe[:m] = (win_hi - lc[:m]) / sig[:m]
            mae[:m] = (win_lo - lc[:m]) / sig[:m]
        out[f"fwd_z_{h}"] = fwd
        out[f"mfe_{h}"] = mfe
        out[f"mae_{h}"] = mae
    return frame.select("event_ts").with_columns([pl.Series(k, v) for k, v in out.items()])


def path_priors(frame: pl.DataFrame, cells: pl.Series, horizons: list[int]) -> pl.DataFrame:
    """Per regime cell and horizon: distribution of forward paths (the 'expected future path' prior)."""
    fp = forward_paths(frame, horizons).with_columns(cells.alias("cell"))
    eps = episodes(cells)
    rows = []
    for c in sorted(x for x in fp["cell"].unique().to_list() if x is not None):
        sub = fp.filter(pl.col("cell") == c)
        for h in horizons:
            z = sub[f"fwd_z_{h}"].drop_nulls().drop_nans().to_numpy()
            if len(z) < 30:
                continue
            q = np.quantile(z, [0.1, 0.25, 0.5, 0.75, 0.9])
            rows.append({
                "cell": c, "h": h, "episodes": eps.get(c, 0), "n": len(z),
                "mean_z": float(z.mean()), "p_up": float((z > 0).mean()),
                "q10": q[0], "q25": q[1], "q50": q[2], "q75": q[3], "q90": q[4],
                "mfe_med": float(np.nanmedian(sub[f"mfe_{h}"].to_numpy())),
                "mae_med": float(np.nanmedian(sub[f"mae_{h}"].to_numpy())),
                "sd_z": float(z.std()),
            })
    return pl.DataFrame(rows)


def shock_events(frame: pl.DataFrame, z_thresh: float, decluster: int) -> pl.DataFrame:
    """Bars with |z| >= z_thresh, declustered: an event within `decluster` bars of the previous one is dropped."""
    z = frame["z"].to_numpy()
    idx = []
    last = -10**9
    for i in np.where(np.abs(np.nan_to_num(z)) >= z_thresh)[0]:
        if i - last > decluster:
            idx.append(i)
            last = i
    return frame[idx].with_columns(pl.Series("i", idx), pl.col("z").sign().alias("shock_sign"))


def shock_study(frame: pl.DataFrame, cells: pl.Series, horizons: list[int], z_thresh: float, decluster: int) -> pl.DataFrame:
    """Post-shock paths: continuation (>0) or reversal (<0) in the shock's direction, by cell at shock time."""
    fp = forward_paths(frame, horizons).with_columns(cells.alias("cell"))
    ev = shock_events(frame, z_thresh, decluster)
    if ev.height == 0:
        return pl.DataFrame()
    rows = []
    sub = fp[ev["i"].to_list()].with_columns(ev["shock_sign"].alias("sign"))
    for c in [*sorted(x for x in sub["cell"].unique().to_list() if x is not None), "ALL"]:
        g = sub if c == "ALL" else sub.filter(pl.col("cell") == c)
        for h in horizons:
            x = (g[f"fwd_z_{h}"] * g["sign"]).drop_nulls().drop_nans().to_numpy()
            if len(x) < 5:
                continue
            m, t, n = newey_west_t(x) if len(x) >= 10 else (float(x.mean()), float("nan"), len(x))
            rows.append({"cell": c, "h": h, "n_events": n, "mean_signed_z": m, "t": t, "p_continue": float((x > 0).mean())})
    return pl.DataFrame(rows)
