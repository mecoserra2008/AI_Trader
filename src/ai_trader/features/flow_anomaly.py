"""Causal order-flow anomaly detectors on flow_3s (docs/ORDER_FLOW.md, plan 2026-11-10 §3, H-OF1..H-OF5).

Pipeline (all strictly causal: the score at minute t uses only minutes <= t):

  rollup_1m(df3)            flow_3s -> complete 1-minute grid per symbol (gaps stay null, never zero-filled)
  add_features(m1)          imbalance, trades, volume, spread, depth, abs_ret (+ ret, sigma_1m)
  robust_z(x)               (x_t - median(x_{t-W..t-1})) / (1.4826 * MAD(x_{t-W..t-1})), W = 1440 (24h)
  liquidation_minutes(liq)  1m liquidation notional / count / dominant side per symbol
  liquidation_clusters(...) notional > trailing-24h 99th percentile (zeros included) and >= 3 prints
  RobustMahalanobis         C-step (MCD-style) location/scatter + Ledoit-Wolf shrinkage, numpy only
  latest_anomalies(...)     per-symbol counts and the latest anomaly, for the market-state report

Conventions
- Minute t covers [t, t+1m); its values are known at t+1m (the 3 s candles' known_ts are at most a few seconds
  later; `rollup_1m` keeps max(known_ts)). The trailing window excludes the current minute.
- MAD = 0 (e.g. an illiquid symbol with >50% trade-less minutes) falls back to 1.2533 * mean absolute deviation
  about the median; the scale is floored at REL_FLOOR * |median| (otherwise a one-tick spread, whose TWA in bps
  moves only with the price level, gives |z| > 4 in half of all minutes); if the scale is still 0, z = 0 where x
  equals the median and +-inf otherwise.
- 3 s candles whose known_ts is more than MAX_LAG_S after the candle end were finalised late (after a reconnect)
  and are dropped before the rollup, so a minute's values are knowable at minute end + ~1 min.
- The first `window` minutes of each symbol have no full trailing window: z is null there (warm-up).
- Liquidation `side` is the side of the liquidated position (schema): "long" liquidations are forced sells.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

WINDOW = 1440          # 24h of 1-minute bars
MIN_VALID = 360        # at least 6h of valid minutes inside the trailing 24h window (collector gaps)
REL_FLOOR = 0.01       # scale >= 1% of |median|: a 1-tick spread's TWA moves only with the price level
MAX_LAG_S = 60.0       # 3 s candles finalised > 60 s after their end (reconnect artefacts) are not usable live
Z_THRESH = 4.0
MAD_K = 1.4826
MEANAD_K = 1.2533      # sqrt(pi/2): mean absolute deviation -> sigma under normality
Z_FEATURES = ("imbalance", "trades", "volume", "spread", "depth", "abs_ret")
MAHA_FEATURES = ("imbalance", "trades", "volume", "spread", "depth", "liq", "abs_ret")
SIDE_LABELS = {
    "imbalance": ("buy", "sell"), "trades": ("high", "low"), "volume": ("high", "low"), "spread": ("widen", "tighten"),
    "depth": ("thick", "thin"), "abs_ret": ("large", "small"), "liq": ("high", "low"),
}
FLOW_COLUMNS = ["symbol", "event_ts", "buy_notional", "sell_notional", "n_buy", "n_sell", "mid", "spread_bps",
                "depth_bid_25", "depth_ask_25", "book_cov", "known_ts"]
MODEL_PATH = Path("data/models/flow_anomaly/mahalanobis_v1.json")


# ------------------------------------------------------------------------------------------- rollup

def rollup_1m(df3: pl.DataFrame, max_lag_s: float | None = MAX_LAG_S) -> pl.DataFrame:
    """flow_3s -> 1-minute bars on a complete per-symbol grid (first to last minute seen).

    Sums for taker flow and counts; book features averaged with book_cov weights (as storage.flow.rollup);
    mid is the last non-null mid in the minute. Minutes without any 3 s candle (collector gaps) stay null."""
    if df3.height == 0:
        return pl.DataFrame()
    if max_lag_s is not None:
        df3 = df3.filter(pl.col("known_ts") <= pl.col("event_ts") + timedelta(seconds=3 + max_lag_s))
    cov = pl.col("book_cov").cast(pl.Float64)

    def w(c: str) -> pl.Expr:
        x = pl.col(c).cast(pl.Float64)
        return ((x * cov).sum() / cov.filter(x.is_not_null()).sum()).alias(c)

    m = (
        df3.sort("symbol", "event_ts")
        .group_by_dynamic("event_ts", every="1m", group_by="symbol", closed="left", label="left")
        .agg(
            pl.col("buy_notional").cast(pl.Float64).sum(),
            pl.col("sell_notional").cast(pl.Float64).sum(),
            (pl.col("n_buy") + pl.col("n_sell")).cast(pl.Float64).sum().alias("trades"),
            pl.col("mid").drop_nulls().last(),
            w("spread_bps"), w("depth_bid_25"), w("depth_ask_25"),
            pl.col("known_ts").max(),
            pl.len().alias("n3s"),
        )
        .with_columns(pl.col(["spread_bps", "depth_bid_25", "depth_ask_25"]).fill_nan(None))
    )
    grids = []
    for (sym,), g in m.group_by("symbol", maintain_order=True):
        rng = pl.datetime_range(g["event_ts"].min(), g["event_ts"].max(), "1m", eager=True, time_zone="UTC")
        grid = pl.DataFrame({"event_ts": rng.cast(g["event_ts"].dtype)}).with_columns(pl.lit(sym).alias("symbol"))
        grids.append(grid.join(g, on=["symbol", "event_ts"], how="left"))
    return pl.concat(grids).select("symbol", "event_ts", *[c for c in m.columns if c not in ("symbol", "event_ts")])


def add_features(m1: pl.DataFrame) -> pl.DataFrame:
    """Raw per-minute features (before z-scoring). ret = 1m log mid return (null across gaps)."""
    return m1.sort("symbol", "event_ts").with_columns(
        (pl.col("buy_notional") - pl.col("sell_notional")).alias("imbalance"),
        (pl.col("buy_notional") + pl.col("sell_notional")).alias("volume"),
        pl.col("spread_bps").alias("spread"),
        (pl.col("depth_bid_25") + pl.col("depth_ask_25")).alias("depth"),
        (pl.col("mid").log() - pl.col("mid").log().shift(1)).over("symbol").alias("ret"),
    ).with_columns(pl.col("ret").abs().alias("abs_ret"))


# ------------------------------------------------------------------------------------- rolling stats

def _trailing_view(x: np.ndarray, window: int):
    """view[i] = x[i-window : i] (current excluded; NaN before the start) and the valid count per row."""
    x = np.asarray(x, dtype=float)
    pad = np.concatenate([np.full(window, np.nan), x])
    view = np.lib.stride_tricks.sliding_window_view(pad, window)[: len(x)]
    fin = np.concatenate([[0], np.cumsum(np.isfinite(pad))])
    k = fin[window:window + len(x)] - fin[: len(x)]
    return view, k


def _sorted_quantile(s: np.ndarray, k: np.ndarray, q: float) -> np.ndarray:
    """Linear-interpolated quantile of each row's first k (valid, sorted) values."""
    pos = q * np.maximum(k - 1, 0)
    lo, hi = np.floor(pos).astype(int), np.ceil(pos).astype(int)
    r = np.arange(len(s))
    out = s[r, lo] + (pos - lo) * (s[r, hi] - s[r, lo])
    return np.where(k > 0, out, np.nan)


def _median_full(w: np.ndarray) -> np.ndarray:
    """Row medians of a NaN-free 2-D block via partition (O(W) per row instead of a full sort)."""
    n = w.shape[1]
    lo, hi = (n - 1) // 2, n // 2
    part = np.partition(w, [lo, hi], axis=1)
    return 0.5 * (part[:, lo] + part[:, hi])


def _chunks(rows: np.ndarray, chunk: int):
    for a in range(0, len(rows), chunk):
        yield rows[a:a + chunk]


def rolling_median_mad(x: np.ndarray, window: int = WINDOW, min_valid: int = MIN_VALID, chunk: int = 256, from_index: int = 0):
    """Trailing (exclusive) median, MAD and mean absolute deviation about the median. NaNs ignored.

    Rows whose window is complete use a partition (fast path); rows with missing minutes sort their window.
    Only rows >= from_index are computed (the rest stay NaN), so a live caller pays for its lookback only."""
    n = len(x)
    med, mad, mean_ad = (np.full(n, np.nan) for _ in range(3))
    view, k = _trailing_view(x, window)
    ok = (k >= min_valid) & (np.arange(n) >= from_index)
    for r in _chunks(np.flatnonzero(ok & (k == window)), chunk):
        w = view[r]
        m = _median_full(w)
        dev = np.abs(w - m[:, None])
        med[r], mad[r], mean_ad[r] = m, _median_full(dev), dev.mean(axis=1)
    for r in _chunks(np.flatnonzero(ok & (k < window)), chunk):
        srt = np.sort(view[r], axis=1)  # NaN sort to the end
        kk = k[r]
        m = _sorted_quantile(srt, kk, 0.5)
        dev = np.abs(srt - m[:, None])
        med[r], mad[r] = m, _sorted_quantile(np.sort(dev, axis=1), kk, 0.5)
        mean_ad[r] = np.nansum(dev, axis=1) / np.maximum(kk, 1)
    return med, mad, mean_ad


def robust_z(x: np.ndarray, window: int = WINDOW, min_valid: int = MIN_VALID, rel_floor: float = REL_FLOOR,
             from_index: int = 0) -> np.ndarray:
    """Online robust z: (x_t - med_{t-W..t-1}) / (1.4826 MAD_{t-W..t-1}); causal by construction."""
    x = np.asarray(x, dtype=float)
    med, mad, mean_ad = rolling_median_mad(x, window, min_valid, from_index=from_index)
    scale = np.where(mad > 0, MAD_K * mad, MEANAD_K * mean_ad)
    scale = np.maximum(scale, rel_floor * np.abs(med))
    with np.errstate(divide="ignore", invalid="ignore"):
        z = (x - med) / scale
        z = np.where((scale == 0) & (x == med), 0.0, z)
        z = np.where((scale == 0) & (x != med) & np.isfinite(x), np.sign(x - med) * np.inf, z)
    return np.where(np.isfinite(x) & np.isfinite(med), z, np.nan)


def rolling_quantile(x: np.ndarray, q: float, window: int = WINDOW, min_valid: int = MIN_VALID, chunk: int = 256,
                     from_index: int = 0) -> np.ndarray:
    """Trailing (exclusive) q-quantile over the previous `window` values; NaNs ignored."""
    n = len(x)
    out = np.full(n, np.nan)
    view, k = _trailing_view(x, window)
    for r in _chunks(np.flatnonzero((k >= min_valid) & (np.arange(n) >= from_index)), chunk):
        out[r] = _sorted_quantile(np.sort(view[r], axis=1), k[r], q)
    return out


def rolling_mean_sq(x: np.ndarray, window: int = WINDOW, min_valid: int = MIN_VALID) -> np.ndarray:
    """Trailing (exclusive) mean of x^2 (for sigma_h = sqrt(h * mean r^2))."""
    x = np.asarray(x, dtype=float)
    v = np.where(np.isfinite(x), x * x, 0.0)
    c = np.isfinite(x).astype(float)
    cs, cc = np.concatenate([[0.0], np.cumsum(v)]), np.concatenate([[0.0], np.cumsum(c)])
    i = np.arange(len(x))
    lo = np.maximum(i - window, 0)
    s, k = cs[i] - cs[lo], cc[i] - cc[lo]
    full = i >= window
    return np.where(full & (k >= min_valid), s / np.maximum(k, 1), np.nan)


def add_robust_z(f: pl.DataFrame, features: Sequence[str] = Z_FEATURES, window: int = WINDOW, min_valid: int = MIN_VALID,
                 rel_floor: float = REL_FLOOR, score_from: datetime | None = None) -> pl.DataFrame:
    """Per symbol: z_<feature> columns plus sigma_1m (trailing RMS of 1m returns). Rows of the first
    `window` minutes of each symbol get null z (warm-up). `score_from`: compute z only for minutes >= it
    (earlier rows stay null; their values still feed later windows)."""
    parts = []
    for (_sym,), g in f.sort("symbol", "event_ts").group_by("symbol", maintain_order=True):
        cols = {}
        fi = 0 if score_from is None else int((g["event_ts"] < score_from).sum())
        for c in features:
            z = robust_z(g[c].cast(pl.Float64).fill_null(np.nan).to_numpy(), window, min_valid, rel_floor, fi)
            z[:window] = np.nan
            cols[f"z_{c}"] = z
        ms = rolling_mean_sq(g["ret"].cast(pl.Float64).fill_null(np.nan).to_numpy(), window, min_valid)
        cols["sigma_1m"] = np.sqrt(ms)
        parts.append(g.with_columns([pl.Series(k, v).fill_nan(None) for k, v in cols.items()]))
    return pl.concat(parts) if parts else f


# ------------------------------------------------------------------------------------- liquidations

def liquidation_minutes(liq: pl.DataFrame) -> pl.DataFrame:
    """1m liquidation notional, count and per-side notional per symbol (only minutes with prints)."""
    if liq.height == 0:
        return pl.DataFrame(schema={"symbol": pl.Utf8, "event_ts": pl.Datetime("ms", "UTC"), "liq_notional": pl.Float64,
                                    "liq_n": pl.Int64, "liq_long": pl.Float64, "liq_short": pl.Float64})
    n = (pl.col("price") * pl.col("size")).cast(pl.Float64)
    return (
        liq.with_columns(pl.col("event_ts").dt.truncate("1m").alias("event_ts"))
        .group_by("symbol", "event_ts")
        .agg(n.sum().alias("liq_notional"), pl.len().cast(pl.Int64).alias("liq_n"),
             n.filter(pl.col("side") == "long").sum().alias("liq_long"),
             n.filter(pl.col("side") == "short").sum().alias("liq_short"))
        .sort("symbol", "event_ts")
    )


def liquidation_clusters(grid: pl.DataFrame, q: float = 0.99, min_n: int = 3, window: int = WINDOW,
                         min_valid: int = MIN_VALID, liq_start: datetime | None = None,
                         score_from: datetime | None = None) -> pl.DataFrame:
    """Flag minutes with liquidation notional > trailing-window q-quantile (zeros included) and >= min_n prints.

    `grid` must be a complete minute grid per symbol with liq_notional / liq_n / liq_long / liq_short
    (nulls = no prints). `liq_start`: the first minute liquidations were collected; earlier minutes are
    unknown (null), not zero; so are minutes where the grid has no flow candle (column n3s null). Adds liq_q99, liq_cluster (bool) and liq_side ("long" | "short", by notional)."""
    parts = []
    for (_sym,), g in grid.sort("symbol", "event_ts").group_by("symbol", maintain_order=True):
        g = g.with_columns(pl.col(["liq_notional", "liq_long", "liq_short"]).fill_null(0.0), pl.col("liq_n").fill_null(0))
        if liq_start is not None:
            g = g.with_columns(pl.when(pl.col("event_ts") < liq_start).then(None).otherwise(pl.col("liq_notional")).alias("liq_notional"))
        if "n3s" in g.columns:  # collector down (no flow candle in the minute): liquidations unknown, not zero
            g = g.with_columns(pl.when(pl.col("n3s").is_null()).then(None).otherwise(pl.col("liq_notional")).alias("liq_notional"))
        x = g["liq_notional"].cast(pl.Float64).fill_null(np.nan).to_numpy()
        fi = 0 if score_from is None else int((g["event_ts"] < score_from).sum())
        thr = rolling_quantile(x, q, window, min_valid, from_index=fi)
        thr[:window] = np.nan
        flag = np.isfinite(thr) & np.isfinite(x) & (x > thr) & (g["liq_n"].to_numpy() >= min_n)
        parts.append(g.with_columns(
            pl.Series("liq_q99", thr).fill_nan(None),
            pl.Series("liq_cluster", flag),
            pl.when(pl.col("liq_long") >= pl.col("liq_short")).then(pl.lit("long")).otherwise(pl.lit("short")).alias("liq_side"),
        ))
    return pl.concat(parts) if parts else grid


# ---------------------------------------------------------------------------------- multivariate

def ledoit_wolf(x: np.ndarray) -> tuple[np.ndarray, float]:
    """Ledoit-Wolf (2004) shrinkage of the sample covariance toward mu*I. Returns (cov, delta)."""
    n, p = x.shape
    xc = x - x.mean(axis=0)
    s = xc.T @ xc / n
    mu = np.trace(s) / p
    d2 = np.sum((s - mu * np.eye(p)) ** 2)
    # b2 = (1/n^2) sum_k ||x_k x_k' - S||_F^2 = (sum_k ||x_k||^4 / n - ||S||_F^2) / n
    b2 = (np.sum(np.sum(xc ** 2, axis=1) ** 2) / n - np.sum(s ** 2)) / n
    delta = float(np.clip(min(b2, d2) / d2, 0.0, 1.0)) if d2 > 0 else 1.0
    return (1 - delta) * s + delta * mu * np.eye(p), delta


@dataclass
class RobustMahalanobis:
    """Robust Mahalanobis distance: MCD-style concentration steps from a coordinate-median start, a
    Ledoit-Wolf-shrunk covariance of the retained h-subset, and a chi2 consistency rescaling."""

    support_fraction: float = 0.75
    clip: float = 20.0
    max_iter: int = 50
    location: np.ndarray | None = None
    covariance: np.ndarray | None = None
    precision: np.ndarray | None = None
    shrinkage: float = float("nan")
    train_quantiles: dict[str, float] = field(default_factory=dict)
    features: tuple[str, ...] = MAHA_FEATURES
    n_train: int = 0

    def _prep(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        return np.clip(np.nan_to_num(x, nan=np.nan, posinf=self.clip, neginf=-self.clip), -self.clip, self.clip)

    @staticmethod
    def _d2(x: np.ndarray, loc: np.ndarray, prec: np.ndarray) -> np.ndarray:
        e = x - loc
        return np.einsum("ij,jk,ik->i", e, prec, e)

    def fit(self, x: np.ndarray) -> RobustMahalanobis:
        from scipy.stats import chi2  # scipy ships with statsmodels (already a dependency)

        x = self._prep(x)
        x = x[np.all(np.isfinite(x), axis=1)]
        n, p = x.shape
        if n < 5 * p:
            raise ValueError(f"need at least {5 * p} complete rows, got {n}")
        h = max(int(self.support_fraction * n), p + 1)
        loc = np.median(x, axis=0)
        mad = MAD_K * np.median(np.abs(x - loc), axis=0)
        cov = np.diag(np.where(mad > 0, mad, x.std(axis=0) + 1e-12) ** 2)
        keep = None
        for _ in range(self.max_iter):
            d2 = self._d2(x, loc, np.linalg.pinv(cov))
            new = np.argpartition(d2, h - 1)[:h]
            if keep is not None and np.array_equal(np.sort(new), np.sort(keep)):
                break
            keep = new
            loc = x[keep].mean(axis=0)
            cov, _ = ledoit_wolf(x[keep])
        cov, self.shrinkage = ledoit_wolf(x[keep])
        d2 = self._d2(x, loc, np.linalg.pinv(cov))
        cov = cov * np.median(d2) / chi2.ppf(0.5, p)  # consistency at the normal model
        self.location, self.covariance, self.precision = loc, cov, np.linalg.pinv(cov)
        d = np.sqrt(self._d2(x, loc, self.precision))
        self.train_quantiles = {k: float(np.quantile(d, float(k))) for k in ("0.5", "0.9", "0.99", "0.999")}
        self.n_train = int(n)
        return self

    def score(self, x: np.ndarray) -> np.ndarray:
        """Distance per row; NaN when any feature is missing."""
        if self.precision is None:
            raise RuntimeError("fit first")
        x = self._prep(x)
        ok = np.all(np.isfinite(x), axis=1)
        out = np.full(len(x), np.nan)
        out[ok] = np.sqrt(self._d2(x[ok], self.location, self.precision))
        return out

    def to_json(self) -> dict[str, Any]:
        return {"features": list(self.features), "clip": self.clip, "support_fraction": self.support_fraction,
                "location": self.location.tolist(), "covariance": self.covariance.tolist(),
                "precision": self.precision.tolist(), "shrinkage": self.shrinkage,
                "train_quantiles": self.train_quantiles, "n_train": self.n_train}

    @classmethod
    def from_json(cls, d: dict[str, Any]) -> RobustMahalanobis:
        m = cls(support_fraction=d["support_fraction"], clip=d["clip"], features=tuple(d["features"]))
        m.location, m.covariance, m.precision = (np.asarray(d[k]) for k in ("location", "covariance", "precision"))
        m.shrinkage, m.train_quantiles, m.n_train = d["shrinkage"], d["train_quantiles"], d["n_train"]
        return m


# ---------------------------------------------------------------------------- end-to-end builders

def build_panel(df3: pl.DataFrame, liq: pl.DataFrame | None = None, liq_start: datetime | None = None,
                window: int = WINDOW, min_valid: int = MIN_VALID, rel_floor: float = REL_FLOOR,
                max_lag_s: float | None = MAX_LAG_S, score_from: datetime | None = None) -> pl.DataFrame:
    """flow_3s (+ liquidations) -> per-minute panel with features, z-scores, sigma_1m and liquidation flags.
    `score_from` limits the (expensive) rolling statistics to minutes >= it."""
    return build_panel_1m(rollup_1m(df3, max_lag_s), liq, liq_start, window, min_valid, rel_floor, score_from)


def build_panel_1m(m1: pl.DataFrame, liq: pl.DataFrame | None = None, liq_start: datetime | None = None,
                   window: int = WINDOW, min_valid: int = MIN_VALID, rel_floor: float = REL_FLOOR,
                   score_from: datetime | None = None) -> pl.DataFrame:
    """As build_panel, from the output of rollup_1m (lets callers roll 3 s data up batch by batch)."""
    f = add_robust_z(add_features(m1), Z_FEATURES, window, min_valid, rel_floor, score_from)
    lm = liquidation_minutes(liq if liq is not None else pl.DataFrame())
    if lm.height:
        lm = lm.with_columns(pl.col("event_ts").cast(f["event_ts"].dtype))
        f = f.join(lm, on=["symbol", "event_ts"], how="left")
    else:
        f = f.with_columns(pl.lit(None, pl.Float64).alias("liq_notional"), pl.lit(None, pl.Int64).alias("liq_n"),
                           pl.lit(None, pl.Float64).alias("liq_long"), pl.lit(None, pl.Float64).alias("liq_short"))
    f = liquidation_clusters(f, window=window, min_valid=min_valid, liq_start=liq_start, score_from=score_from)
    parts = []
    for (_sym,), g in f.group_by("symbol", maintain_order=True):
        fi = 0 if score_from is None else int((g["event_ts"] < score_from).sum())
        z = robust_z(g["liq_notional"].cast(pl.Float64).fill_null(np.nan).to_numpy(), window, min_valid, rel_floor, fi)
        z[:window] = np.nan
        parts.append(g.with_columns(pl.Series("z_liq", z).fill_nan(None)))
    return pl.concat(parts)


def flow_symbols(lake, as_of: datetime, category: str = "linear") -> list[str]:
    """Symbols with flow_3s rows known at as_of."""
    q = f"""SELECT DISTINCT symbol FROM read_parquet('{lake._glob('flow_3s')}', hive_partitioning = true, union_by_name = true)
            WHERE category = ? AND known_ts <= ? ORDER BY symbol"""
    return [r[0] for r in lake.con.execute(q, [category, as_of]).fetchall()]


def liquidation_start(lake, as_of: datetime) -> datetime | None:
    """First liquidation print known at as_of (= start of liquidation collection)."""
    liq = lake.read_asof("liquidations", as_of, columns=["event_ts", "known_ts"])
    return liq["event_ts"].min() if liq.height else None


def read_1m(lake, as_of: datetime, symbols: Sequence[str], start: datetime, category: str = "linear",
            batch: int = 12, max_lag_s: float | None = MAX_LAG_S) -> pl.DataFrame:
    """Point-in-time flow_3s read in symbol batches, each rolled up to 1 minute at once (bounded memory)."""
    parts = []
    for i in range(0, len(symbols), batch):
        df3 = lake.read_asof("flow_3s", as_of, symbols=list(symbols[i:i + batch]), start=start,
                             filters={"category": category, "venue": "bybit"}, columns=FLOW_COLUMNS)
        if df3.height:
            parts.append(rollup_1m(df3, max_lag_s))
    return pl.concat(parts, how="diagonal_relaxed") if parts else pl.DataFrame()


def load_model(path: Path | None = None) -> RobustMahalanobis | None:
    from ai_trader.core.config import load

    p = path or (load().root / MODEL_PATH)
    return RobustMahalanobis.from_json(json.loads(Path(p).read_text())) if Path(p).is_file() else None


def latest_anomalies(symbols: Sequence[str] | None, as_of: datetime, lookback_h: int = 24, *, z_thresh: float = Z_THRESH,
                     lake=None, category: str = "linear", model: RobustMahalanobis | None = None,
                     use_model: bool = True, maha_quantile: str = "0.999") -> dict[str, dict[str, Any]]:
    """Per-symbol flow-anomaly summary over the `lookback_h` hours before `as_of` (point in time).

    Reads flow_3s and liquidations as of `as_of` (lookback + 24h warm-up), keeps only complete minutes
    (minute end <= as_of) and returns, per symbol:
      {"counts": {type: n}, "n_total": n, "latest": {"type", "z", "time", "side"} | None,
       "latest_by_type": {type: {...}}, "minutes": n_scored}
    ("latest" = the most recent anomalous minute; within a minute the largest |z|; z is the Mahalanobis
    distance for type multivariate and the liquidation-notional robust z for liq_cluster.)
    types: imbalance, trades, volume, spread, depth, abs_ret (|robust z| > z_thresh), liq_cluster, and
    multivariate (Mahalanobis > the fitted model's training `maha_quantile`; `model` or, if None and use_model,
    the JSON fitted by research/flow_anomaly_study.py in data/models/flow_anomaly/).
    Live callers: rows are flushed every 5 minutes, so as_of <= now - 5 min gives a complete read."""
    from ai_trader.core.time import utc
    from ai_trader.storage.lake import Lake

    lake = lake or Lake()
    as_of = utc(as_of)
    t0 = as_of - timedelta(hours=lookback_h)
    start = t0 - timedelta(minutes=WINDOW + 5)
    syms = list(symbols) if symbols is not None else flow_symbols(lake, as_of, category)
    m1 = read_1m(lake, as_of, syms, start, category)
    if m1.height == 0:
        return {}
    liq = lake.read_asof("liquidations", as_of, symbols=syms, start=start, filters={"venue": "bybit"})
    ls = liquidation_start(lake, as_of)
    p = build_panel_1m(m1, liq, liq_start=None if ls is None or ls <= start else ls, score_from=t0)
    p = p.filter((pl.col("event_ts") >= t0) & (pl.col("event_ts") + timedelta(minutes=1) <= as_of))
    model = model if model is not None else (load_model() if use_model else None)
    out: dict[str, dict[str, Any]] = {}
    for (sym,), g in p.sort("event_ts").group_by("symbol", maintain_order=True):
        events = []
        for c in Z_FEATURES:
            z = g[f"z_{c}"].to_numpy()
            with np.errstate(invalid="ignore"):
                hit = np.abs(z) > z_thresh  # NaN -> False, +-inf -> True
            for i in map(int, np.flatnonzero(hit)):
                pos, neg = SIDE_LABELS[c]
                side = ("up" if (g["ret"][i] or 0) > 0 else "down") if c == "abs_ret" else (pos if z[i] > 0 else neg)
                events.append((g["event_ts"][i], c, float(z[i]), side))
        lc = g["liq_cluster"].to_numpy()
        for i in map(int, np.flatnonzero(lc)):
            events.append((g["event_ts"][i], "liq_cluster", float(g["z_liq"][i]) if g["z_liq"][i] is not None else float("nan"), g["liq_side"][i]))
        if model is not None:
            x = np.column_stack([g[f"z_{c}"].cast(pl.Float64).fill_null(np.nan).to_numpy() for c in model.features])
            d = model.score(x)
            thr = model.train_quantiles.get(maha_quantile, np.inf)
            for i in map(int, np.flatnonzero(np.isfinite(d) & (d > thr))):
                zi = g["z_imbalance"][i]
                events.append((g["event_ts"][i], "multivariate", float(d[i]), "buy" if (zi or 0) > 0 else "sell"))
        counts: dict[str, int] = {}
        for e in events:
            counts[e[1]] = counts.get(e[1], 0) + 1
        def key(e):
            return (e[0], abs(e[2]) if np.isfinite(e[2]) else np.inf)

        def as_dict(e):
            return {"type": e[1], "z": e[2], "time": e[0].isoformat(), "side": e[3]}

        by_type = {}
        for e in sorted(events, key=key):
            by_type[e[1]] = as_dict(e)
        out[sym] = {
            "counts": counts, "n_total": len(events),
            "minutes": int(g["z_imbalance"].is_not_null().sum()),
            "latest": as_dict(max(events, key=key)) if events else None,  # latest minute; ties -> largest |z|
            "latest_by_type": by_type,
        }
    return out
