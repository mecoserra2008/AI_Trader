"""Daily net P&L panel of every bar-runnable registered strategy plus the trend-zoo ensemble, and the regime-conditional
strategy map (Program 2, block RC: H-RC1, H-RC2). Plan: research/plans/2026-11-11_strategy_program.md (agent B1).

Panel (data/priors/strategy_panel_daily.parquet, written atomically as soon as it exists)
-----------------------------------------------------------------------------------------
* Universes: L = trend_zoo_study.UNIVERSE (23 crypto perps); T = breakout_cipher_kalman_study.UNIV_T (16 Bybit TradFi
  perps) kept only with >= 120 days of 1h history after the listing burn-in (as in the 2026-11-10 program).
* Strategies: every class in strategies.REGISTRY that runs on bars, at its configured frequency: the 1h-parameterised
  ones (config/strategies.yaml says "defaults are priors for 1h bars") on 1h bars; geo_struct / geo_pivot_bo /
  geo_flag_bo on 4h (H-GT1/H-GT2 were 4h); plus `zoo_ensemble` = trend_zoo.ensemble with zoo(6) on 4h (H-TR1).
  Skipped: flow_momentum and cvd_divergence (flow_bars exist for BTCUSDT only, from 2026-05-11, so they cannot run on
  the universes); oi_price_divergence on T (Bybit TradFi open interest only from 2026-11-05).
  shock_regime is DATA-GENERATED (designed after seeing H-S2/H-S3 in both segments): it is in the panel and the map,
  but its in-sample and holdout rows are not evidence for it.
* Frames: research.frame.market_frame(sym, freq, as_of) (funding on 1h; OI on 1h for L only). Equity perps have no
  native 4h before 2026-11-04: the 4h frame is built from complete groups of four 1h bars
  (breakout_cipher_kalman_study.frame_4h_from_1h) whenever native 4h has < 80% of the 1h-implied bar count.
* P&L: evaluate.strategy_pnl (risk units: position s / sigma_next fixed at entry, re-sized only on flip / flat /
  |delta s| > 0.2; costs config/risk.yaml taker + slippage = 7.5 bps per side). Signals on the FULL history.
* Dating: the position of row t is held from close_ts(t) to close_ts(t+1), so its P&L belongs to the UTC date of
  close_ts(t). A date's P&L is therefore exactly the 00:00 -> 24:00 UTC holding interval for 1h and 4h alike, which
  aligns it with close-to-close daily factor returns. Only complete dates before the as_of date are kept. Per
  symbol the panel starts WARM_DAYS after the later of the first 1h / 4h bar (every strategy warm).
* Columns: symbol, universe, date, strategy, family, freq, bpd (bars per day), pnl / gross / cost (bar-risk units,
  summed over the day), pnl_d / gross_d / cost_d (= x / sqrt(bpd): daily-sigma units, scale-free across frequencies;
  used for every cross-strategy comparison), turnover (sum |delta held signal|, a full flip = 2), n_trades (entries
  that day), exposure (share of the day's bars with a position), active (any position or any cost that day), cell
  (v0 regime cell of the symbol at the START of the day: research.regimes.label on the 1h frame, the row whose
  close_ts is 00:00 UTC, i.e. known when the day's first position is held), seg (llm_in_sample before
  research.knowledge_cutoff, clean_holdout on/after).
* Side tables: data/priors/strategy_panel_trades.parquet (one row per trade = maximal run of a same-sign held
  position: entry, side, bars, gross, cost, MFE, MAE in bar-risk units) and data/priors/strategy_panel_context.parquet
  (per symbol-date: start-of-day cell / dir_score / vol_state / sigma / funding, and the day's realised measures from
  features.realized.daily_measures: ret, rv, close, plus the day's turnover). Context columns ending in _sod are known
  at 00:00 UTC of the date; ret / rv / turnover are the date's own outcomes (factor returns, never predictors).

Trade MFE / MAE (H-PD4 input): for a trade over rows i..j with leverage lev_k (sign = side), the gross P&L path after
row k is G_k = sum_{m<k} lev_m * fwd_ret_m; the intrabar favourable excursion during bar k+1 is
lev_k * (log high_{k+1} - log close_k) for longs (log low for shorts). MFE = max(0, max_k G_k + that), MAE likewise
with the adverse extreme (<= 0). Trade cost = c * (|lev_i| + sum of in-trade |delta lev| + |lev_j|), which sums
exactly to strategy_pnl's cost. Trades still open at the last bar are dropped.

H-RC1 (registered: Spearman > 0 with CI excluding 0)
----------------------------------------------------
* Universe L, unit = symbol-day. Per strategy and segment, evaluate.by_regime(bar_hours=24) on pnl_d with the
  start-of-day cell (rows with a null cell are dropped; symbols are concatenated with a null-cell separator row so
  evaluate.episodes never merges runs across symbols). Shrinkage kappa = settings research.shrinkage_kappa (10).
* Pairs (strategy, cell) with >= 5 episodes in BOTH segments; Spearman of mean_shrunk in-sample vs holdout.
* CI: 2000 bootstrap draws resampling regime episodes (symbol-level runs of one cell) with replacement, independently
  per segment and jointly across strategies; shrunk means recomputed exactly as by_regime does; percentile 95% CI;
  one-sided p = share of draws with Spearman <= 0. Recorded once, segment "pooled" (the statistic spans both).

H-RC2 (registered: conditional minus both alternatives > 0 with bootstrap CI excluding 0 in the holdout)
---------------------------------------------------------------------------------------------------------
* Books on universe L, per symbol-day weights over the strategies: equal-weight (1/N all), unconditional (1/N_sel over
  strategies with llm_in_sample pooled mean_active > 0), conditional (1/N_sel(cell) over strategies whose
  llm_in_sample shrunk mean in the day's start-of-day cell is > 0; 0 if none). Book daily P&L = mean over symbols of
  sum_s w_s pnl_d(s); annualised Sharpe sqrt(365). Paired Sharpe differences with a circular moving-block bootstrap
  of dates (10-day blocks, 2000 draws, the three books resampled jointly). Holdout is the test; the in-sample row is
  in-sample selection (reported, verdict inconclusive by construction).

Context-ranking table: registered family x v0 cell -> shrunk mean pnl_d per segment (family P&L per symbol-day = mean
of its members' pnl_d; by_regime as above), NW t of the date-clustered cell mean (per date, mean over symbols in the
cell), sign agreement. Grade A: same sign and t >= 2 in both segments; B: same sign, t >= 2 in-sample and >= 1 in
the holdout; C: same sign otherwise; '-': signs disagree (t counted in the direction of the shrunk mean). T sleeves are reported for the holdout only.

Holm within the RC block: {H-RC1 p, H-RC2 cond-EW p, H-RC2 cond-uncond p}; a registered "CI excluding 0" counts as
met only if the Holm-adjusted p < 0.05 as well.

    uv run python -m ai_trader.research.strategy_panel build [--workers 4]
    uv run python -m ai_trader.research.strategy_panel map [--record]
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl

from ai_trader.core.config import load
from ai_trader.core.time import now
from ai_trader.research.evaluate import by_regime, knowledge_cutoff, newey_west_t, strategy_pnl
from ai_trader.research.trend_zoo_study import UNIVERSE

UNIV_L = tuple(UNIVERSE)
UNIV_T = tuple("XAUUSDT XAGUSDT CLUSDT BZUSDT TSLAUSDT INTCUSDT GOOGLUSDT AAPLUSDT MSTRUSDT NVDAUSDT MUUSDT QQQUSDT "
               "SOXLUSDT SPCXUSDT PAXGUSDT XAUTUSDT".split())
MIN_T_DAYS = 120
WARM_DAYS = 71           # >= zoo lookback (414 4h bars = 69 d) and the 1h strategies' 744-bar windows (31 d)
BAND = 0.2               # evaluate.strategy_pnl rebalance band (default)
BPD = {"1h": 24, "4h": 6}
SKIPPED = {
    "flow_momentum": "needs flow_bars: only BTCUSDT from 2026-05-11 in the lake",
    "cvd_divergence": "needs flow_bars: only BTCUSDT from 2026-05-11 in the lake",
}
SKIPPED_T = {"oi_price_divergence": "Bybit TradFi open interest only from 2026-11-05"}
SEGS = ("llm_in_sample", "clean_holdout")
MIN_EPISODES = 5
N_BOOT = 2000
BLOCK = 10
REPORT = "strategy_map_2026-11-11"


def priors_dir() -> Path:
    return load().root / "data" / "priors"


def panel_path() -> Path:
    return priors_dir() / "strategy_panel_daily.parquet"


def trades_path() -> Path:
    return priors_dir() / "strategy_panel_trades.parquet"


def context_path() -> Path:
    return priors_dir() / "strategy_panel_context.parquet"


def cost_bps() -> float:
    c = load().risk["costs"]
    return float(c["taker_bps"] + c["slippage_bps"])


def seg_expr(col: str = "date") -> pl.Expr:
    return (pl.when(pl.col(col) < knowledge_cutoff().date()).then(pl.lit("llm_in_sample"))
            .otherwise(pl.lit("clean_holdout")).alias("seg"))


def write_atomic(df: pl.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.write_parquet(tmp)
    os.replace(tmp, path)
    return path


# ====================================================================== strategy list
def strategy_specs(universe: str = "L") -> list[tuple[str, str, str]]:
    """(name, family, freq) of every strategy in the panel, in a fixed order."""
    from ai_trader.strategies import REGISTRY

    out = []
    for name, cls in REGISTRY.items():
        if name in SKIPPED or (universe == "T" and name in SKIPPED_T):
            continue
        out.append((name, cls.spec.family, "4h" if name.startswith("geo_") else "1h"))
    out.append(("zoo_ensemble", "trend", "4h"))
    return out


def strategy_fn(name: str):
    from ai_trader.strategies import REGISTRY
    from ai_trader.strategies.trend_zoo import ensemble, zoo

    if name == "zoo_ensemble":
        z = zoo(6)
        return lambda f: ensemble(f, 6, z)
    return REGISTRY[name]()


# ====================================================================== per-bar mechanics
def held_leverage(signal: np.ndarray, sigma: np.ndarray, band: float = BAND) -> tuple[np.ndarray, np.ndarray]:
    """The held signal and leverage of evaluate.strategy_pnl's execution policy (replicated: strategy_pnl does not
    return them). Returns (held_signal, lev) with held_signal = 0 wherever lev = 0."""
    s = np.nan_to_num(np.asarray(signal, dtype=float))
    n = len(s)
    lev = np.zeros(n)
    hs = np.zeros(n)
    held_s, held_lev = 0.0, 0.0
    for t in range(n):
        st = s[t]
        if st == 0.0:
            held_s, held_lev = 0.0, 0.0
        elif held_s == 0.0 or np.sign(st) != np.sign(held_s) or abs(st - held_s) > band:
            held_s = st
            held_lev = st / sigma[t] if np.isfinite(sigma[t]) and sigma[t] > 0 else 0.0
        lev[t] = held_lev
        hs[t] = held_s if held_lev != 0.0 else 0.0
    return hs, lev


def trade_table(lev: np.ndarray, gross: np.ndarray, lc: np.ndarray, lh: np.ndarray, ll: np.ndarray, c_frac: float) -> list[dict]:
    """Maximal runs of same-sign leverage -> per-trade gross, cost, MFE, MAE (bar-risk units). Open trades dropped."""
    n = len(lev)
    sg = np.sign(lev)
    out = []
    i = 0
    while i < n:
        if sg[i] == 0:
            i += 1
            continue
        j = i
        while j + 1 < n and sg[j + 1] == sg[i]:
            j += 1
        if j + 1 >= n:  # still open at the last bar (its last return is unknown)
            break
        ks = np.arange(i, j + 1)
        g = gross[i:j + 1]
        G = np.r_[0.0, np.cumsum(g)[:-1]]
        lk = lev[i:j + 1]
        if sg[i] > 0:
            fav, adv = lh[ks + 1], ll[ks + 1]
        else:
            fav, adv = ll[ks + 1], lh[ks + 1]
        ef = G + lk * (fav - lc[ks])
        ea = G + lk * (adv - lc[ks])
        cost = c_frac * (abs(lev[i]) + float(np.abs(np.diff(lk)).sum()) + abs(lev[j]))
        out.append({"i": i, "j": j, "side": int(sg[i]), "bars": j - i + 1, "gross": float(g.sum()), "cost": float(cost),
                    "mfe": float(max(0.0, np.nanmax(ef))), "mae": float(min(0.0, np.nanmin(ea)))})
        i = j + 1
    return out


def daily_from_bars(f: pl.DataFrame, p: pl.DataFrame, hs: np.ndarray, lev: np.ndarray) -> pl.DataFrame:
    """Aggregate strategy_pnl rows to UTC dates of close_ts (the start of each row's holding interval)."""
    turn = np.abs(np.diff(np.r_[0.0, hs]))
    entry = (lev != 0) & ((np.r_[0.0, lev[:-1]] == 0) | (np.sign(np.r_[0.0, lev[:-1]]) != np.sign(lev)))
    d = pl.DataFrame({
        "date": f["close_ts"].dt.date(), "pnl": p["pnl"], "gross": p["gross"], "cost": p["cost"],
        "turnover": turn, "entry": entry.astype(np.int32), "pos": (lev != 0),
    })
    return (d.group_by("date").agg(pl.col("pnl").sum(), pl.col("gross").sum(), pl.col("cost").sum(), pl.col("turnover").sum(),
                                   pl.col("entry").sum().alias("n_trades"), pl.col("pos").mean().alias("exposure"),
                                   (pl.col("pos").any() | (pl.col("cost") > 0).any()).alias("active"))
            .sort("date"))


def start_of_day_context(f1: pl.DataFrame) -> pl.DataFrame:
    """Per date: the v0 cell etc. of the 1h row whose close_ts is 00:00 UTC (known when the date's first position is
    held), plus the date's realised measures and turnover (outcomes of the date)."""
    from ai_trader.features.realized import daily_measures
    from ai_trader.research.regimes import label

    lab = label(f1, bar_hours=1.0)
    cols = [pl.col("close_ts").dt.date().alias("date"), pl.col("cell").alias("cell_sod"), pl.col("dir_score").alias("dir_score_sod"),
            pl.col("dir_state").alias("dir_state_sod"), pl.col("vol_state").alias("vol_state_sod"),
            pl.col("sigma_next").alias("sigma_1h_sod"), pl.col("sigma_risk_next").alias("sigma_risk_sod"),
            pl.col("close").alias("close_sod")]
    if "funding_rate" in lab.columns:
        cols.append(pl.col("funding_rate").alias("funding_sod"))
    sod = lab.filter((pl.col("close_ts").dt.hour() == 0) & (pl.col("close_ts").dt.minute() == 0)).select(cols)
    if "funding_sod" not in sod.columns:
        sod = sod.with_columns(pl.lit(None, dtype=pl.Float64).alias("funding_sod"))
    dm = daily_measures(f1.select("event_ts", "open", "high", "low", "close"), "1h").select("date", "ret", "rv", "close", "n_bars")
    tv = f1.group_by(pl.col("event_ts").dt.date().alias("date")).agg(pl.col("turnover").sum().alias("turnover_day"))
    return sod.join(dm, on="date", how="full", coalesce=True).join(tv, on="date", how="left").sort("date")


def frames_for(sym: str, universe: str, as_of: datetime) -> tuple[pl.DataFrame, pl.DataFrame, str]:
    from ai_trader.research.breakout_cipher_kalman_study import frame_4h_from_1h
    from ai_trader.research.frame import market_frame

    f1 = market_frame(sym, "1h", as_of, funding=True, oi=(universe == "L"))
    f4 = market_frame(sym, "4h", as_of, funding=False, oi=False)
    src = "native"
    if f1.height and f4.height < 0.8 * f1.height / 4:
        f4 = frame_4h_from_1h(sym, as_of)
        src = "aggregated_from_1h"
    return f1, f4, src


def symbol_job(sym: str, universe: str, as_of_iso: str) -> dict:
    as_of = datetime.fromisoformat(as_of_iso)
    t0 = time.time()
    out: dict = {"symbol": sym, "universe": universe}
    f1, f4, src = frames_for(sym, universe, as_of)
    out["f4_source"] = src
    if f1.height == 0 or f4.height == 0:
        out["skip"] = "no data"
        return out
    days = (f1["event_ts"].max() - f1["event_ts"].min()).days
    if universe == "T" and days < MIN_T_DAYS:
        out["skip"] = f"{days} days < {MIN_T_DAYS}"
        return out
    start = max(f1["event_ts"].min(), f4["event_ts"].min()).date() + timedelta(days=WARM_DAYS)
    last = as_of.date()  # dates < as_of date only (complete days)
    c_frac = cost_bps() / 1e4
    daily, trades = [], []
    for name, fam, freq in strategy_specs(universe):
        f = f1 if freq == "1h" else f4
        sig = strategy_fn(name)(f)
        p = strategy_pnl(f, sig)
        sn = f["sigma_next"].to_numpy()
        hs, lev = held_leverage(sig.to_numpy(), sn)
        fr = np.nan_to_num(f["fwd_ret"].to_numpy())
        if not np.allclose(lev * fr, p["gross"].to_numpy(), atol=1e-12):
            raise AssertionError(f"{sym} {name}: replicated leverage does not match strategy_pnl")
        d = daily_from_bars(f, p, hs, lev).filter((pl.col("date") >= start) & (pl.col("date") < last))
        bpd = BPD[freq]
        daily.append(d.with_columns(pl.lit(sym).alias("symbol"), pl.lit(universe).alias("universe"), pl.lit(name).alias("strategy"),
                                    pl.lit(fam).alias("family"), pl.lit(freq).alias("freq"), pl.lit(bpd).alias("bpd")))
        lc, lh, ll = (np.log(f[k].to_numpy()) for k in ("close", "high", "low"))
        ets = f["close_ts"]
        for tr in trade_table(lev, p["gross"].to_numpy(), lc, lh, ll, c_frac):
            ed = ets[tr["i"]].date()
            if ed < start or ets[tr["j"] + 1].date() >= last:
                continue
            trades.append({"symbol": sym, "universe": universe, "strategy": name, "family": fam, "freq": freq, "bpd": bpd,
                           "entry_ts": ets[tr["i"]], "entry_date": ed, **{k: tr[k] for k in ("side", "bars", "gross", "cost", "mfe", "mae")}})
    ctx = start_of_day_context(f1).with_columns(pl.lit(sym).alias("symbol"), pl.lit(universe).alias("universe"))
    dd = pl.concat(daily, how="vertical_relaxed").join(ctx.select("date", pl.col("cell_sod").alias("cell")), on="date", how="left")
    out.update({"daily": dd, "trades": pl.DataFrame(trades) if trades else None, "context": ctx, "days": days,
                "start": start.isoformat(), "secs": round(time.time() - t0, 1)})
    return out


PANEL_COLS = ["symbol", "universe", "date", "strategy", "family", "freq", "bpd", "pnl", "gross", "cost", "pnl_d", "gross_d",
              "cost_d", "turnover", "n_trades", "exposure", "active", "cell", "seg"]


def finalize_panel(daily: pl.DataFrame) -> pl.DataFrame:
    s = pl.col("bpd").cast(pl.Float64).sqrt()
    return (daily.with_columns((pl.col("pnl") / s).alias("pnl_d"), (pl.col("gross") / s).alias("gross_d"), (pl.col("cost") / s).alias("cost_d"),
                               pl.col("n_trades").cast(pl.Int32), pl.col("bpd").cast(pl.Int32), seg_expr())
            .select(PANEL_COLS).sort("universe", "strategy", "symbol", "date"))


def build(workers: int = 4, as_of: datetime | None = None) -> dict:
    as_of = as_of or now()
    jobs = [(s, "L") for s in UNIV_L] + [(s, "T") for s in UNIV_T]
    res = []
    with ProcessPoolExecutor(max_workers=min(4, workers)) as ex:
        futs = [ex.submit(symbol_job, s, u, as_of.isoformat()) for s, u in jobs]
        for fu in futs:
            r = fu.result()
            print(f"  {r['symbol']} ({r['universe']}) {r.get('skip', '')} {r.get('secs', '')}s f4={r.get('f4_source')}", flush=True)
            res.append(r)
    ok = [r for r in res if "skip" not in r]
    panel = finalize_panel(pl.concat([r["daily"] for r in ok], how="vertical_relaxed"))
    write_atomic(panel, panel_path())  # FIRST: other agents may wait for this file
    trades = pl.concat([r["trades"] for r in ok if r["trades"] is not None], how="vertical_relaxed").with_columns(seg_expr("entry_date"))
    write_atomic(trades, trades_path())
    ctx = pl.concat([r["context"] for r in ok], how="vertical_relaxed")
    write_atomic(ctx, context_path())
    meta = {"as_of": as_of.isoformat(), "cost_per_side_bps": cost_bps(), "knowledge_cutoff": knowledge_cutoff().date().isoformat(),
            "symbols": {u: [r["symbol"] for r in ok if r["universe"] == u] for u in ("L", "T")},
            "skipped_symbols": {r["symbol"]: r["skip"] for r in res if "skip" in r},
            "f4_aggregated_from_1h": [r["symbol"] for r in ok if r.get("f4_source") == "aggregated_from_1h"],
            "starts": {r["symbol"]: r["start"] for r in ok},
            "strategies_L": strategy_specs("L"), "strategies_T": strategy_specs("T"),
            "skipped_strategies": SKIPPED, "skipped_strategies_T": SKIPPED_T,
            "rows": panel.height, "trades": trades.height}
    (priors_dir() / "strategy_panel_meta.json").write_text(json.dumps(meta, indent=1, default=str), encoding="utf-8")
    return meta


# ====================================================================== statistics helpers
def spearman(a: np.ndarray, b: np.ndarray) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    a, b = a[ok], b[ok]
    if len(a) < 3:
        return float("nan")
    ra = _rank(a)
    rb = _rank(b)
    if ra.std() == 0 or rb.std() == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def _rank(x: np.ndarray) -> np.ndarray:
    """Average ranks (ties share the mean rank)."""
    order = np.argsort(x, kind="mergesort")
    r = np.empty(len(x))
    r[order] = np.arange(len(x), dtype=float)
    xs = x[order]
    i = 0
    while i < len(xs):
        j = i
        while j + 1 < len(xs) and xs[j + 1] == xs[i]:
            j += 1
        if j > i:
            r[order[i:j + 1]] = (i + j) / 2
        i = j + 1
    return r


def holm(p: dict[str, float]) -> dict[str, float]:
    keys = [k for k in p if np.isfinite(p[k])]
    keys.sort(key=lambda k: p[k])
    m = len(keys)
    out, run = {}, 0.0
    for i, k in enumerate(keys):
        run = max(run, min(1.0, (m - i) * p[k]))
        out[k] = run
    for k in p:
        out.setdefault(k, float("nan"))
    return out


def sharpe_ann(x: np.ndarray) -> float:
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) < 20:
        return float("nan")
    sd = x.std(ddof=1)
    return float(x.mean() / sd * np.sqrt(365)) if sd > 0 else float("nan")


def block_indices(n: int, block: int, rng: np.random.Generator) -> np.ndarray:
    """Circular moving-block bootstrap index of length n."""
    nb = int(np.ceil(n / block))
    starts = rng.integers(0, n, size=nb)
    idx = (starts[:, None] + np.arange(block)[None, :]).ravel()[:n] % n
    return idx


# ====================================================================== regime tables
def _with_separators(df: pl.DataFrame) -> tuple[pl.DataFrame, pl.Series]:
    """Concatenate per-symbol rows (sorted by date) with a null-cell separator row between symbols."""
    parts = []
    for _, g in df.sort("symbol", "date").group_by("symbol", maintain_order=True):
        parts.append(g.select(pl.col("pnl_d").alias("pnl"), "active", "cell"))
        parts.append(pl.DataFrame({"pnl": [0.0], "active": [False], "cell": [None]}, schema={"pnl": pl.Float64, "active": pl.Boolean, "cell": pl.Utf8}))
    x = pl.concat(parts, how="vertical_relaxed")
    return x.select("pnl", "active"), x["cell"]


def regime_table(df: pl.DataFrame) -> pl.DataFrame:
    """evaluate.by_regime on daily pnl_d of one sleeve (rows: symbol, date, pnl_d, active, cell) in one segment."""
    df = df.filter(pl.col("cell").is_not_null())
    if df.height == 0:
        return pl.DataFrame()
    p, cells = _with_separators(df)
    return by_regime(p, cells, bar_hours=24.0)


def cell_t_clustered(df: pl.DataFrame) -> dict[str, tuple[float, float, int]]:
    """Per cell: NW t of the per-date mean (over symbols) of active pnl_d (clustering by time)."""
    out = {}
    a = df.filter(pl.col("active") & pl.col("cell").is_not_null())
    for c, g in a.group_by("cell"):
        x = g.group_by("date").agg(pl.col("pnl_d").mean()).sort("date")["pnl_d"].to_numpy()
        out[c[0]] = newey_west_t(x)
    return out


def episode_ids(df: pl.DataFrame) -> pl.DataFrame:
    """Adds `ep` (global id of the symbol-level contiguous run of one cell) to rows sorted by symbol, date."""
    d = df.sort("symbol", "date")
    new = (pl.col("cell") != pl.col("cell").shift(1)).fill_null(True) | (pl.col("symbol") != pl.col("symbol").shift(1)).fill_null(True)
    return d.with_columns(new.cum_sum().alias("ep"))


# ====================================================================== H-RC1
def rc1(panel: pl.DataFrame, n_boot: int = N_BOOT, seed: int = 11) -> dict:
    L = panel.filter((pl.col("universe") == "L") & pl.col("cell").is_not_null())
    kappa = float(load().settings["research"]["shrinkage_kappa"])
    strats = sorted(L["strategy"].unique().to_list())
    tables = {}
    for seg in SEGS:
        for s in strats:
            tables[(s, seg)] = regime_table(L.filter((pl.col("strategy") == s) & (pl.col("seg") == seg)))
    pairs = []
    for s in strats:
        a, b = tables[(s, "llm_in_sample")], tables[(s, "clean_holdout")]
        if a.height == 0 or b.height == 0:
            continue
        ja = a.filter(pl.col("cell") != "ALL").join(b.filter(pl.col("cell") != "ALL"), on="cell", suffix="_ho")
        for r in ja.iter_rows(named=True):
            if r["episodes"] >= MIN_EPISODES and r["episodes_ho"] >= MIN_EPISODES:
                pairs.append({"strategy": s, "cell": r["cell"], "is_shrunk": r["mean_shrunk"], "ho_shrunk": r["mean_shrunk_ho"],
                              "is_eps": r["episodes"], "ho_eps": r["episodes_ho"], "is_raw": r["mean_active"], "ho_raw": r["mean_active_ho"]})
    pr = pl.DataFrame(pairs)
    rho = spearman(pr["is_shrunk"].to_numpy(), pr["ho_shrunk"].to_numpy())
    # ---- bootstrap over episodes, per segment, joint across strategies
    rng = np.random.default_rng(seed)
    pair_idx = [(strats.index(r["strategy"]), r["cell"]) for r in pairs]
    boot_seg = {}
    for seg in SEGS:
        d = episode_ids(L.filter(pl.col("seg") == seg).select("symbol", "date", "cell").unique().sort("symbol", "date"))
        key = d.select("symbol", "date", "ep", "cell")
        ep_cell = key.group_by("ep").agg(pl.col("cell").first()).sort("ep")
        n_ep = ep_cell.height
        cells_of_ep = ep_cell["cell"].to_numpy()
        x = (L.filter(pl.col("seg") == seg).join(key, on=["symbol", "date", "cell"])
             .filter(pl.col("active")).group_by("strategy", "ep").agg(pl.col("pnl_d").sum().alias("s"), pl.len().alias("n")))
        S = np.zeros((len(strats), n_ep))
        N = np.zeros((len(strats), n_ep))
        si = {s: i for i, s in enumerate(strats)}
        epi = x["ep"].to_numpy() - 1
        ii = np.array([si[s] for s in x["strategy"].to_list()])
        S[ii, epi] = x["s"].to_numpy()
        N[ii, epi] = x["n"].to_numpy()
        boot_seg[seg] = (S, N, cells_of_ep, n_ep)

    def shrunk(S, N, cells_of_ep, counts):
        tot_s, tot_n = S @ counts, N @ counts
        pooled = np.where(tot_n > 0, tot_s / np.maximum(tot_n, 1), np.nan)
        res = {}
        for c in set(cells_of_ep.tolist()):
            m = cells_of_ep == c
            cs, cn = S[:, m] @ counts[m], N[:, m] @ counts[m]
            ne = counts[m].sum()
            w = ne / (ne + kappa)
            raw = np.where(cn > 0, cs / np.maximum(cn, 1), np.nan)
            res[c] = np.where(np.isfinite(raw), w * raw + (1 - w) * pooled, pooled)
        return res

    # sanity: the bootstrap machinery at counts = 1 reproduces by_regime's shrunk means
    chk = {}
    for seg in SEGS:
        S, N, ce, ne = boot_seg[seg]
        r0 = shrunk(S, N, ce, np.ones(ne))
        chk[seg] = float(np.nanmax(np.abs([r0[c][i] - (pr.filter((pl.col("strategy") == strats[i]) & (pl.col("cell") == c))[
            "is_shrunk" if seg == "llm_in_sample" else "ho_shrunk"][0]) for i, c in pair_idx])))
    rhos = np.full(n_boot, np.nan)
    for b in range(n_boot):
        vals = {}
        for seg in SEGS:
            S, N, ce, ne = boot_seg[seg]
            counts = np.bincount(rng.integers(0, ne, ne), minlength=ne).astype(float)
            vals[seg] = shrunk(S, N, ce, counts)
        a = [vals["llm_in_sample"].get(c, np.full(len(strats), np.nan))[i] for i, c in pair_idx]
        h = [vals["clean_holdout"].get(c, np.full(len(strats), np.nan))[i] for i, c in pair_idx]
        rhos[b] = spearman(np.array(a), np.array(h))
    ok = np.isfinite(rhos)
    ci = [float(np.quantile(rhos[ok], 0.025)), float(np.quantile(rhos[ok], 0.975))]
    p = float((np.sum(rhos[ok] <= 0) + 1) / (ok.sum() + 1))
    sign_agree = float(np.mean(np.sign(pr["is_shrunk"].to_numpy()) == np.sign(pr["ho_shrunk"].to_numpy())))
    # by-strategy within-strategy rank agreement (descriptive)
    per_strat = {}
    for s in strats:
        q = pr.filter(pl.col("strategy") == s)
        if q.height >= 4:
            per_strat[s] = {"n_cells": q.height, "spearman": spearman(q["is_shrunk"].to_numpy(), q["ho_shrunk"].to_numpy())}
    # demeaned version: does the CELL ordering within a strategy persist (removes each strategy's level)?
    dm = pr.with_columns((pl.col("is_shrunk") - pl.col("is_shrunk").mean().over("strategy")).alias("is_dm"),
                         (pl.col("ho_shrunk") - pl.col("ho_shrunk").mean().over("strategy")).alias("ho_dm"))
    return {"n_pairs": pr.height, "n_strategies": pr["strategy"].n_unique(), "spearman": rho, "ci95": ci, "p_one_sided": p,
            "n_boot": int(ok.sum()), "sign_agreement": sign_agree,
            "spearman_within_strategy_demeaned": spearman(dm["is_dm"].to_numpy(), dm["ho_dm"].to_numpy()),
            "per_strategy": per_strat, "bootstrap_check_max_abs_diff": chk, "pairs": pr.to_dicts(),
            "tables": {f"{s}|{seg}": t.to_dicts() for (s, seg), t in tables.items() if t.height}}


# ====================================================================== H-RC2
def rc2(panel: pl.DataFrame, rc1_out: dict, n_boot: int = N_BOOT, seed: int = 12, exclude: tuple[str, ...] = ()) -> dict:
    L = panel.filter((pl.col("universe") == "L") & pl.col("cell").is_not_null() & ~pl.col("strategy").is_in(list(exclude)))
    strats = sorted(L["strategy"].unique().to_list())
    tabs = rc1_out["tables"]
    pooled_is = {}
    cell_is = {}
    for s in strats:
        t = tabs.get(f"{s}|llm_in_sample", [])
        for r in t:
            if r["cell"] == "ALL":
                pooled_is[s] = r["mean_active"]
            else:
                cell_is[(s, r["cell"])] = r["mean_shrunk"]
    uncond = [s for s in strats if np.isfinite(pooled_is.get(s, np.nan)) and pooled_is[s] > 0]
    cells = sorted({c for (_, c) in cell_is})
    cond = {c: [s for s in strats if cell_is.get((s, c), -np.inf) > 0] for c in cells}
    w = pl.DataFrame([{"strategy": s, "cell": c, "w_eq": 1.0 / len(strats),
                       "w_un": (1.0 / len(uncond)) if s in uncond else 0.0,
                       "w_co": (1.0 / len(cond[c])) if (c in cond and s in cond[c]) else 0.0}
                      for s in strats for c in cells])
    book = (L.join(w, on=["strategy", "cell"], how="left").with_columns(pl.col("w_eq", "w_un", "w_co").fill_null(0.0))
            .group_by("symbol", "date", "seg").agg((pl.col("w_eq") * pl.col("pnl_d")).sum().alias("eq"),
                                                   (pl.col("w_un") * pl.col("pnl_d")).sum().alias("un"),
                                                   (pl.col("w_co") * pl.col("pnl_d")).sum().alias("co"))
            .group_by("date", "seg").agg(pl.col("eq", "un", "co").mean()).sort("date"))
    out = {"unconditional_set": uncond, "conditional_sets": cond, "n_strategies": len(strats)}
    rng = np.random.default_rng(seed)
    for seg in SEGS:
        b = book.filter(pl.col("seg") == seg)
        X = b.select("co", "eq", "un").to_numpy()
        n = len(X)
        sh = {k: sharpe_ann(X[:, i]) for i, k in enumerate(("co", "eq", "un"))}
        d_eq, d_un = sh["co"] - sh["eq"], sh["co"] - sh["un"]
        bd = np.full((n_boot, 2), np.nan)
        for k in range(n_boot):
            idx = block_indices(n, BLOCK, rng)
            Y = X[idx]
            s_ = [sharpe_ann(Y[:, i]) for i in range(3)]
            bd[k] = (s_[0] - s_[1], s_[0] - s_[2])
        ok = np.isfinite(bd).all(axis=1)
        res = {"n_days": n, "sharpe": sh, "diff_vs_eq": d_eq, "diff_vs_uncond": d_un}
        for j, nm in enumerate(("vs_eq", "vs_uncond")):
            res[f"ci95_{nm}"] = [float(np.quantile(bd[ok, j], 0.025)), float(np.quantile(bd[ok, j], 0.975))]
            res[f"p_{nm}"] = float((np.sum(bd[ok, j] <= 0) + 1) / (ok.sum() + 1))
        res["mean_daily"] = {k: float(np.mean(X[:, i])) for i, k in enumerate(("co", "eq", "un"))}
        res["corr_co_eq"] = float(np.corrcoef(X[:, 0], X[:, 1])[0, 1]) if n > 2 else float("nan")
        out[seg] = res
    return out


# ====================================================================== context-ranking table
GRADE_NOTE = ("A: same sign, t >= 2 in both segments; B: same sign, t >= 2 in-sample and >= 1 holdout; C: same sign "
              "otherwise; '-': signs disagree. t = NW t of the per-date mean over symbols (clustered by time), counted in the "
              "direction of the shrunk mean.")


def grade(m_is: float, t_is: float, m_ho: float, t_ho: float) -> str:
    if not (np.isfinite(m_is) and np.isfinite(m_ho)) or np.sign(m_is) != np.sign(m_ho):
        return "-"
    # t counted in the direction of the shrunk mean (shrunk symbol-day means and date-clustered t can disagree in sign)
    ti, th = np.nan_to_num(t_is) * np.sign(m_is), np.nan_to_num(t_ho) * np.sign(m_ho)
    if ti >= 2 and th >= 2:
        return "A"
    if ti >= 2 and th >= 1:
        return "B"
    return "C"


def family_panel(panel: pl.DataFrame) -> pl.DataFrame:
    return (panel.group_by("universe", "family", "symbol", "date", "seg", "cell")
            .agg(pl.col("pnl_d").mean(), pl.col("active").any()).rename({"family": "strategy"}))


def context_table(panel: pl.DataFrame, level: str = "family") -> list[dict]:
    src = family_panel(panel) if level == "family" else panel
    rows = []
    for (u, s), g in src.group_by("universe", "strategy"):
        res = {}
        for seg in SEGS:
            d = g.filter(pl.col("seg") == seg)
            tb = regime_table(d)
            ts = cell_t_clustered(d)
            res[seg] = ({r["cell"]: r for r in tb.to_dicts()} if tb.height else {}, ts)
        cells = sorted(set(res["llm_in_sample"][0]) | set(res["clean_holdout"][0]))
        for c in cells:
            if c == "ALL":
                continue
            ri, rh = res["llm_in_sample"][0].get(c), res["clean_holdout"][0].get(c)
            ti = res["llm_in_sample"][1].get(c, (np.nan, np.nan, 0))
            th = res["clean_holdout"][1].get(c, (np.nan, np.nan, 0))
            mi = ri["mean_shrunk"] if ri else float("nan")
            mh = rh["mean_shrunk"] if rh else float("nan")
            rows.append({"universe": u, "level": level, "name": s, "cell": c,
                         "is_shrunk": mi, "is_raw": ri["mean_active"] if ri else float("nan"), "is_eps": ri["episodes"] if ri else 0,
                         "is_t": ti[1], "is_days": ti[2],
                         "ho_shrunk": mh, "ho_raw": rh["mean_active"] if rh else float("nan"), "ho_eps": rh["episodes"] if rh else 0,
                         "ho_t": th[1], "ho_days": th[2],
                         "sign_agree": bool(np.isfinite(mi) and np.isfinite(mh) and np.sign(mi) == np.sign(mh)),
                         "grade": grade(mi, ti[1], mh, th[1]) if u == "L" else "holdout-only"})
    return sorted(rows, key=lambda r: (r["universe"], r["name"], r["cell"]))


# ====================================================================== orchestration / verdicts
def run_map(panel: pl.DataFrame | None = None) -> dict:
    panel = panel if panel is not None else pl.read_parquet(panel_path())
    out = {"panel_rows": panel.height, "panel_last_date": str(panel["date"].max()),
           "strategies": {u: sorted(panel.filter(pl.col("universe") == u)["strategy"].unique().to_list()) for u in ("L", "T")}}
    out["H-RC1"] = rc1(panel)
    out["H-RC2"] = rc2(panel, out["H-RC1"])
    # sensitivity (not the registered test): without the data-generated shock_regime, whose design saw holdout evidence
    out["H-RC2_ex_shock"] = rc2(panel, out["H-RC1"], exclude=("shock_regime",))
    pr = pl.DataFrame(out["H-RC1"]["pairs"]).filter(pl.col("strategy") != "shock_regime")
    out["H-RC1"]["spearman_ex_shock"] = spearman(pr["is_shrunk"].to_numpy(), pr["ho_shrunk"].to_numpy())
    out["context_family"] = context_table(panel, "family")
    out["context_strategy"] = context_table(panel, "strategy")
    r1, r2 = out["H-RC1"], out["H-RC2"]["clean_holdout"]
    p = {"H-RC1": r1["p_one_sided"], "H-RC2_vs_eq": r2["p_vs_eq"], "H-RC2_vs_uncond": r2["p_vs_uncond"]}
    out["holm"] = {"raw": p, "adjusted": holm(p)}
    out["verdicts"] = verdicts(out)
    return out


def verdicts(out: dict) -> dict:
    adj = out["holm"]["adjusted"]
    r1 = out["H-RC1"]
    if not np.isfinite(r1["spearman"]) or r1["n_pairs"] < 10:
        v1 = "inconclusive"
    elif r1["spearman"] > 0 and r1["ci95"][0] > 0 and adj["H-RC1"] < 0.05:
        v1 = "supports"
    elif r1["spearman"] <= 0:
        v1 = "contradicts"
    else:
        v1 = "inconclusive"
    ho = out["H-RC2"]["clean_holdout"]
    if ho["n_days"] < 30:
        v2 = "inconclusive"
    elif (ho["diff_vs_eq"] > 0 and ho["ci95_vs_eq"][0] > 0 and adj["H-RC2_vs_eq"] < 0.05
          and ho["diff_vs_uncond"] > 0 and ho["ci95_vs_uncond"][0] > 0 and adj["H-RC2_vs_uncond"] < 0.05):
        v2 = "supports"
    elif ho["diff_vs_eq"] <= 0 or ho["diff_vs_uncond"] <= 0:
        v2 = "contradicts"
    else:
        v2 = "inconclusive"
    return {"H-RC1": {"pooled": v1}, "H-RC2": {"clean_holdout": v2, "llm_in_sample": "inconclusive"}}


DATA_NOTE = ("data/priors/strategy_panel_daily.parquet (research.strategy_panel): universe L = 23 crypto perps, daily net P&L "
             "in daily-sigma units (pnl_d) of {n} strategies at configured frequency (1h; geo_* and zoo_ensemble 4h), "
             "evaluate.strategy_pnl, 7.5 bps per side; start-of-day v0 cells from 1h; as of {as_of}")


def _already(hid: str, seg: str) -> bool:
    from ai_trader.research.hypotheses import results

    return any(r["segment"] == seg for r in results(hid))


def record_all(out: dict, meta: dict) -> list[str]:
    from ai_trader.research.hypotheses import record

    done = []
    data = DATA_NOTE.format(n=len(out["strategies"]["L"]), as_of=meta.get("as_of"))
    r1 = out["H-RC1"]
    m1 = {k: r1[k] for k in ("n_pairs", "n_strategies", "spearman", "ci95", "p_one_sided", "n_boot", "sign_agreement",
                             "spearman_within_strategy_demeaned")}
    m1["p_holm"] = out["holm"]["adjusted"]["H-RC1"]
    if not _already("H-RC1", "pooled"):
        record("H-RC1", "pooled", out["verdicts"]["H-RC1"]["pooled"], m1,
               note="Spearman over (strategy, v0 cell) pairs with >= 5 episodes in both segments of evaluate.by_regime shrunk means "
                    "(daily pnl_d, start-of-day cell), llm_in_sample vs clean_holdout; CI from 2000 episode-resampling draws; "
                    "Holm over the RC block (3 tests). shock_regime is data-generated (in the map, not evidence).", data=data)
        done.append("H-RC1/pooled")
    r2 = out["H-RC2"]
    for seg in SEGS:
        if _already("H-RC2", seg):
            continue
        m = dict(r2[seg])
        m["n_uncond"] = len(r2["unconditional_set"])
        m["mean_n_cond"] = float(np.mean([len(v) for v in r2["conditional_sets"].values()]))
        if seg == "clean_holdout":
            m["p_holm"] = {k: out["holm"]["adjusted"][k] for k in ("H-RC2_vs_eq", "H-RC2_vs_uncond")}
        note = ("books on universe L: equal-weight vs unconditional (in-sample pooled mean > 0) vs conditional (in-sample shrunk cell "
                "mean > 0, start-of-day cell); paired annualised Sharpe differences, 10-day circular block bootstrap (2000)")
        if seg == "llm_in_sample":
            note += "; IN-SAMPLE SELECTION: the sets were chosen on this segment, so this row is descriptive, not evidence"
        record("H-RC2", seg, out["verdicts"]["H-RC2"][seg], m, note=note, data=data)
        done.append(f"H-RC2/{seg}")
    return done


# ====================================================================== report
def _f(x, nd=3) -> str:
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return "n/a"
    return f"{x:.{nd}f}" if isinstance(x, float) else str(x)


def write_report(out: dict, meta: dict, rule_yaml: str) -> tuple[Path, Path]:
    rep = load().root / "research" / "reports"
    js = {k: v for k, v in out.items() if k != "H-RC1"}
    js["H-RC1"] = {k: v for k, v in out["H-RC1"].items() if k != "tables"}
    js["meta"] = meta
    pj = rep / f"{REPORT}.json"
    pj.write_text(json.dumps(js, indent=1, default=str), encoding="utf-8")
    r1, r2 = out["H-RC1"], out["H-RC2"]
    L: list[str] = [f"# Strategy map by regime (H-RC1, H-RC2) and the context-ranking table — {REPORT[-10:]}", ""]
    L += ["Agent B1, Program 2 ([plan](../plans/2026-11-11_strategy_program.md)). Hypotheses registered 2026-11-11 before testing.", "",
          "## Data and methods", "",
          f"- Panel `data/priors/strategy_panel_daily.parquet` ({meta.get('rows')} rows; built by `ai_trader.research.strategy_panel`), as of {meta.get('as_of')}.",
          f"- Universe L: {len(meta['symbols']['L'])} crypto perps; universe T: {len(meta['symbols']['T'])} TradFi perps with >= 120 days "
          f"({', '.join(meta['symbols']['T'])}); skipped symbols: {meta.get('skipped_symbols') or 'none'}. 4h built from 1h for: "
          f"{', '.join(meta.get('f4_aggregated_from_1h', [])) or 'none'}.",
          f"- Strategies (L): {', '.join(out['strategies']['L'])}. Skipped: {', '.join(f'{k} ({v})' for k, v in meta['skipped_strategies'].items())}; "
          f"on T also {', '.join(f'{k} ({v})' for k, v in meta['skipped_strategies_T'].items())}.",
          "- 1h-parameterised strategies on 1h bars; geo_* and zoo_ensemble on 4h. evaluate.strategy_pnl (risk units, sized at entry, "
          "7.5 bps per side). Daily P&L dated by the start of each holding interval (UTC 00-24h); `pnl_d` = daily sum / sqrt(bars per day) "
          "(daily-sigma units, scale-free across frequencies). Regime = v0 cell of the symbol at 00:00 UTC (1h labeller).",
          "- by_regime: empirical-Bayes shrinkage kappa = 10 pseudo-episodes toward the strategy's pooled active mean; effective n = episodes.",
          "- Segments: llm_in_sample < 2026-07-01 <= clean_holdout. `shock_regime` is data-generated: shown, never evidence.", ""]
    L += ["## H-RC1: are per-regime rankings stable across the cutoff?", "",
          f"- Pairs (strategy x cell, >= 5 episodes in both segments): **{r1['n_pairs']}** over {r1['n_strategies']} strategies.",
          f"- Spearman(in-sample shrunk mean, holdout shrunk mean) = **{_f(r1['spearman'])}**, 95% CI [{_f(r1['ci95'][0])}, {_f(r1['ci95'][1])}] "
          f"(episode bootstrap, {r1['n_boot']} draws), one-sided p {_f(r1['p_one_sided'], 4)}, Holm p {_f(out['holm']['adjusted']['H-RC1'], 4)}.",
          f"- Sign agreement over pairs: {_f(r1['sign_agreement'])}. Within-strategy (strategy level removed) Spearman: "
          f"{_f(r1['spearman_within_strategy_demeaned'])}: does the *cell ordering inside a strategy* persist?",
          f"- **Verdict: {out['verdicts']['H-RC1']['pooled']}** (recorded once, segment pooled).", "",
          "| strategy | cells | within-strategy Spearman |", "|---|---|---|"]
    for s, v in sorted(r1["per_strategy"].items()):
        L.append(f"| {s} | {v['n_cells']} | {_f(v['spearman'])} |")
    L += ["", "## H-RC2: does a regime-conditional allocator beat equal weight out of sample?", "",
          f"- Unconditional set (in-sample pooled mean > 0): {', '.join(r2['unconditional_set']) or 'none'}.",
          "- Conditional sets per cell (in-sample shrunk mean > 0): " + "; ".join(f"{c}: {', '.join(v) or 'none'}" for c, v in r2["conditional_sets"].items()), "",
          "| segment | days | Sharpe cond | Sharpe EW | Sharpe uncond | cond-EW [95% CI] | cond-uncond [95% CI] |", "|---|---|---|---|---|---|---|"]
    for seg in SEGS:
        x = r2[seg]
        L.append(f"| {seg} | {x['n_days']} | {_f(x['sharpe']['co'], 2)} | {_f(x['sharpe']['eq'], 2)} | {_f(x['sharpe']['un'], 2)} | "
                 f"{_f(x['diff_vs_eq'], 2)} [{_f(x['ci95_vs_eq'][0], 2)}, {_f(x['ci95_vs_eq'][1], 2)}] | "
                 f"{_f(x['diff_vs_uncond'], 2)} [{_f(x['ci95_vs_uncond'][0], 2)}, {_f(x['ci95_vs_uncond'][1], 2)}] |")
    ex = out.get("H-RC2_ex_shock")
    if ex:
        L += ["", "Sensitivity (not the registered test): without the data-generated shock_regime (its design saw holdout evidence):", ""]
        for seg in SEGS:
            x = ex[seg]
            L.append(f"- {seg}: Sharpe cond {_f(x['sharpe']['co'], 2)}, EW {_f(x['sharpe']['eq'], 2)}, uncond {_f(x['sharpe']['un'], 2)}; "
                     f"cond-EW {_f(x['diff_vs_eq'], 2)} [{_f(x['ci95_vs_eq'][0], 2)}, {_f(x['ci95_vs_eq'][1], 2)}], "
                     f"cond-uncond {_f(x['diff_vs_uncond'], 2)} [{_f(x['ci95_vs_uncond'][0], 2)}, {_f(x['ci95_vs_uncond'][1], 2)}]")
        L.append(f"- H-RC1 Spearman without shock_regime pairs: {_f(out['H-RC1'].get('spearman_ex_shock'))}.")
    L += ["", f"- Holm (RC block): {json.dumps({k: round(v, 4) for k, v in out['holm']['adjusted'].items()})}.",
          f"- **Verdict: clean_holdout {out['verdicts']['H-RC2']['clean_holdout']}**; llm_in_sample inconclusive by construction (selection in-sample).", ""]
    L += ["## Context-ranking table (family x v0 cell)", "",
          "Shrunk mean net P&L per active symbol-day in daily-sigma units (x100 = % of one daily sigma), in-sample | holdout, "
          "with date-clustered NW t and grade. " + GRADE_NOTE, "",
          "| family | cell | IS shrunk x100 | IS t | IS eps | HO shrunk x100 | HO t | HO eps | sign agree | grade |", "|---|---|---|---|---|---|---|---|---|---|"]
    from ai_trader.research.regimes import CELL_LABELS

    for r in out["context_family"]:
        if r["universe"] != "L":
            continue
        L.append(f"| {r['name']} | {r['cell']} {CELL_LABELS.get(r['cell'], '')} | {_f(100 * r['is_shrunk'], 2)} | {_f(r['is_t'], 1)} | {r['is_eps']} | "
                 f"{_f(100 * r['ho_shrunk'], 2)} | {_f(r['ho_t'], 1)} | {r['ho_eps']} | {'yes' if r['sign_agree'] else 'no'} | **{r['grade']}** |")
    L += ["", "### Universe T (TradFi perps, holdout only, robustness)", "",
          "| family | cell | HO shrunk x100 | HO t | HO eps |", "|---|---|---|---|---|"]
    for r in out["context_family"]:
        if r["universe"] == "T" and r["ho_eps"] > 0:
            L.append(f"| {r['name']} | {r['cell']} | {_f(100 * r['ho_shrunk'], 2)} | {_f(r['ho_t'], 1)} | {r['ho_eps']} |")
    L += ["", "### Per-strategy cells graded A or B (universe L)", "",
          "| strategy | cell | IS shrunk x100 | IS t | HO shrunk x100 | HO t | grade |", "|---|---|---|---|---|---|---|"]
    for r in out["context_strategy"]:
        if r["universe"] == "L" and r["grade"] in ("A", "B"):
            L.append(f"| {r['name']} | {r['cell']} | {_f(100 * r['is_shrunk'], 2)} | {_f(r['is_t'], 1)} | {_f(100 * r['ho_shrunk'], 2)} | {_f(r['ho_t'], 1)} | {r['grade']} |")
    L += ["", "## Caveats", "",
          "- Universe L is today's survivors (survivorship bias in any cross-sectional statement).",
          "- The holdout is ~4 months: holdout cell means rest on a handful of episodes per symbol, pooled across 23 co-moving coins.",
          "- Cells and strategies are both functions of past prices, so a strategy's P&L by cell partly restates its construction "
          "(trend strategies are long in up cells by design); the question is whether the *net* P&L ordering persists.",
          "- 1h strategy parameters are the pre-registered config values, not tuned; families with one member (volatility, shock) are single strategies.",
          "- shock_regime is data-generated (H-S2/H-S3 seen in both segments) and is not evidence; the calendar strategies trade only around weekends.",
          "", "## Rule candidates", "", "```yaml", rule_yaml.rstrip(), "```", ""]
    pm = rep / f"{REPORT}.md"
    pm.write_text("\n".join(L), encoding="utf-8")
    return pj, pm


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--workers", type=int, default=4)
    m = sub.add_parser("map")
    m.add_argument("--record", action="store_true")
    m.add_argument("--out", default=None, help="write the raw map JSON here (scratch)")
    a = ap.parse_args()
    if a.cmd == "build":
        meta = build(a.workers)
        print(json.dumps({k: v for k, v in meta.items() if k not in ("starts",)}, indent=1, default=str))
    else:
        out = run_map()
        if a.out:
            Path(a.out).write_text(json.dumps({k: v for k, v in out.items() if k != "H-RC1"} | {"H-RC1": {k: v for k, v in out["H-RC1"].items() if k != "tables"}},
                                              indent=1, default=str), encoding="utf-8")
        print(json.dumps({"verdicts": out["verdicts"], "holm": out["holm"],
                          "H-RC1": {k: out["H-RC1"][k] for k in ("n_pairs", "spearman", "ci95", "p_one_sided", "sign_agreement")},
                          "H-RC2": {s: out["H-RC2"][s] for s in SEGS}}, indent=1, default=str))
        if a.record:
            meta = json.loads((priors_dir() / "strategy_panel_meta.json").read_text(encoding="utf-8"))
            print(record_all(out, meta))


if __name__ == "__main__":
    main()
