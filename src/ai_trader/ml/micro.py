"""1 s micro-structure inputs and models for H-NN5 (next-minute volatility) and H-NN6 (next-minute direction).

Source: agent A's aggregates data/research/micro_days/<date>/{trades_1s, book_1s}/<SYMBOL>.parquet (read only; schema in
research/reports/multiday_micro_batch2.md, 'Aggregate schema'). One symbol-day becomes a dense (86400, C) channel matrix,
one row per UTC second s (bin [s, s+1)). Row s uses information known at the end of second s only. Trade aggregates are
for second s. The book state is the one at the end of second s (book ts = trade ts + 1000 ms), forward-filled. Every
normaliser is a TRAILING statistic over seconds s-3599..s (point in time, per symbol-day).

  P_s     last trade price up to the end of s (forward-filled; seconds before the first print are NaN)
  r_s     log P_s - log P_{s-1} (0 when no trade)
  b_s     trailing-1h mean of r^2 (per-second variance); n_s, k_s, o_s trailing-1h means of notional / prints / orders

Channels (CHANNELS, 12; clipped to [-10, 10]):
  0 z       r_s / sqrt(b_s)
  1 la      log(r_s^2 / b_s + 0.01)
  2 hl      log(high_s / low_s) / sqrt(b_s)                   (0 without a print)
  3 flow    (buy_notional - sell_notional) / n_s              (signed flow)
  4 vol     log1p((buy + sell notional) / n_s)
  5 prints  log1p(n_trades / k_s)
  6 orders  log1p(n_orders / o_s)
  7 oimb    (n_buy_orders - n_sell_orders) / max(n_orders, 1)
  8 book    1 if an L2 book exists for the symbol-day, else 0
  9 spread  log rel_spread_bp - its trailing-1h mean           (0 without a book)
  10 depth  log(depth_bid_10 + depth_ask_10) - trailing-1h mean (BTC 10 bp depth is a truncated lower bound; relative)
  11 imb    imb_10                                             (0 without a book)

Decision at the end of second s: input window = rows s-299..s (the last 5 minutes); baseline variance 60 b_s;
targets r1m = log P_{s+60} - log P_s (squared for H-NN5, sign for H-NN6). Decision seconds run from 01:00 + 5 min, so
that the trailing hour is full, to 23:59:00, on a fixed grid (every `step` seconds).
HAR aggregates (same inputs, for the linear model and the CNN's linear skip): for every channel the mean over the last
10, 60 and 300 seconds (36 values).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl

CHANNELS = ("z", "la", "hl", "flow", "vol", "prints", "orders", "oimb", "book", "spread", "depth", "imb")
C = len(CHANNELS)
WIN = 300
HAR_W = (10, 60, 300)
DAY_S = 86400
TRAIL = 3600
HORIZON = 60
MIN_TRADED_SECONDS = 10_000


def _trail_mean(x: np.ndarray, w: int = TRAIL) -> np.ndarray:
    c = np.concatenate([[0.0], np.cumsum(x)])
    n = np.arange(1, len(x) + 1)
    lo = np.maximum(n - w, 0)
    return (c[n] - c[lo]) / (n - lo)


def symbol_day_channels(trades: pl.DataFrame, book: pl.DataFrame | None, day_start_ms: int) -> dict:
    """Dense per-second channels for one symbol-day. Returns {'X': (86400, C) float32, 'logp': (86400,), 'b': (86400,),
    'traded': int}."""
    t = trades.sort("ts")
    idx = ((t["ts"].to_numpy() - day_start_ms) // 1000).astype(np.int64)
    ok = (idx >= 0) & (idx < DAY_S)
    t, idx = t.filter(pl.Series(ok)), idx[ok]

    def dense(col: str, fill: float = 0.0) -> np.ndarray:
        z = np.full(DAY_S, fill)
        z[idx] = t[col].cast(pl.Float64).fill_null(fill).to_numpy()
        return z

    close = np.full(DAY_S, np.nan)
    close[idx] = t["close"].to_numpy().astype(float)
    logp = pl.Series(np.log(close)).fill_nan(None).forward_fill().to_numpy().astype(float)
    r = np.nan_to_num(np.diff(logp, prepend=np.nan), nan=0.0)
    hi, lo = dense("high", np.nan), dense("low", np.nan)
    bn, sn = dense("buy_notional"), dense("sell_notional")
    nt, no = dense("n_trades"), dense("n_orders")
    nbo, nso = dense("n_buy_orders"), dense("n_sell_orders")
    b = _trail_mean(r * r)
    nmean, kmean, omean = _trail_mean(bn + sn), _trail_mean(nt), _trail_mean(no)
    with np.errstate(all="ignore"):
        sb = np.sqrt(b)
        X = np.zeros((DAY_S, C))
        X[:, 0] = np.where(b > 0, r / sb, 0.0)
        X[:, 1] = np.where(b > 0, np.log(r * r / b + 0.01), np.log(0.01))
        X[:, 2] = np.where(np.isfinite(hi) & np.isfinite(lo) & (lo > 0) & (b > 0), np.log(hi / lo) / sb, 0.0)
        X[:, 3] = np.where(nmean > 0, (bn - sn) / nmean, 0.0)
        X[:, 4] = np.where(nmean > 0, np.log1p((bn + sn) / nmean), 0.0)
        X[:, 5] = np.where(kmean > 0, np.log1p(nt / kmean), 0.0)
        X[:, 6] = np.where(omean > 0, np.log1p(no / omean), 0.0)
        X[:, 7] = (nbo - nso) / np.maximum(no, 1.0)
    if book is not None and book.height:
        bk = book.with_columns(pl.col("ts").cast(pl.Datetime("ms", "UTC")).dt.epoch("ms").alias("ms")).sort("ms")
        # book state at the end of second s has ts = day_start + (s + 1) * 1000
        bi = ((bk["ms"].to_numpy() - day_start_ms) // 1000 - 1).astype(np.int64)
        okb = (bi >= 0) & (bi < DAY_S)
        bk, bi = bk.filter(pl.Series(okb)), bi[okb]

        def dense_b(col: str) -> np.ndarray:
            z = np.full(DAY_S, np.nan)
            z[bi] = bk[col].cast(pl.Float64).to_numpy()
            return pl.Series(z).fill_nan(None).forward_fill().to_numpy().astype(float)

        with np.errstate(all="ignore"):
            ls = np.log(dense_b("rel_spread_bp"))
            ld = np.log(dense_b("depth_bid_10") + dense_b("depth_ask_10"))
            im = dense_b("imb_10")
            for j, v in ((9, ls), (10, ld)):
                fin = np.isfinite(v)
                vv = np.where(fin, v, 0.0)
                cnt = _trail_mean(fin.astype(float)) * np.minimum(np.arange(1, DAY_S + 1), TRAIL)
                sm = _trail_mean(vv) * np.minimum(np.arange(1, DAY_S + 1), TRAIL)
                mu = np.where(cnt > 0, sm / np.maximum(cnt, 1), np.nan)
                X[:, j] = np.where(fin & np.isfinite(mu), v - mu, 0.0)
            X[:, 11] = np.where(np.isfinite(im), im, 0.0)
            X[:, 8] = 1.0
    X = np.clip(np.nan_to_num(X, nan=0.0, posinf=10.0, neginf=-10.0), -10, 10).astype(np.float32)
    return {"X": X, "logp": logp, "b": b, "traded": int(ok.sum())}


def decision_seconds(step: int, offset: int = 0) -> np.ndarray:
    start = TRAIL + WIN
    return np.arange(start + offset, DAY_S - HORIZON, step, dtype=np.int64)


def targets(logp: np.ndarray, b: np.ndarray, s: np.ndarray) -> dict:
    r1 = logp[s + HORIZON] - logp[s]
    base = HORIZON * b[s]
    ok = np.isfinite(r1) & (base > 0) & np.isfinite(logp[s - WIN + 1])
    return {"r1": np.where(ok, r1, np.nan), "base": base, "ok": ok}


def har_features(X: np.ndarray, s: np.ndarray) -> np.ndarray:
    """(n, C * 3) means of every channel over the last 10 / 60 / 300 seconds ending at s (inclusive)."""
    cs = np.vstack([np.zeros((1, X.shape[1])), np.cumsum(X.astype(np.float64), axis=0)])
    out = [(cs[s + 1] - cs[s + 1 - w]) / w for w in HAR_W]
    return np.concatenate(out, axis=1).astype(np.float32)


# ------------------------------------------------------------------ store of symbol-days


@dataclass
class Store:
    """Concatenated channel matrices of many symbol-days (float16) and per-window indices."""
    X: np.ndarray            # (N, C) float16, all symbol-days back to back
    offsets: dict            # key -> start row in X
    meta: dict               # key -> {'day', 'symbol', 'book', 'traded'}

    def windows(self, key, s: np.ndarray) -> np.ndarray:
        return self.offsets[key] + s


def load_symbol_day(root: Path, day: str, symbol: str) -> dict | None:
    from datetime import UTC, datetime

    tp = root / day / "trades_1s" / f"{symbol}.parquet"
    if not tp.is_file():
        return None
    tr = pl.read_parquet(tp)
    bp = root / day / "book_1s" / f"{symbol}.parquet"
    bk = pl.read_parquet(bp) if bp.is_file() else None
    d0 = int(datetime.fromisoformat(day).replace(tzinfo=UTC).timestamp() * 1000)
    return symbol_day_channels(tr, bk, d0)


# ------------------------------------------------------------------ models (torch)


def make_cnn(n_har: int = C * len(HAR_W), width: int = 16, task: str = "vol"):
    from ai_trader.ml.models import _torch

    torch = _torch()
    nn = torch.nn

    class CNN(nn.Module):
        """Dilated 1D CNN over (C, 300) with multi-scale pooling + a linear skip on the HAR aggregates."""

        def __init__(self) -> None:
            super().__init__()
            self.convs = nn.ModuleList([nn.Conv1d(C, width, 3), nn.Conv1d(width, width, 3, dilation=2),
                                        nn.Conv1d(width, width, 3, dilation=4), nn.Conv1d(width, width, 3, dilation=8)])
            self.head = nn.Sequential(nn.Linear(4 * width, 32), nn.ReLU(), nn.Linear(32, 1))
            self.skip = nn.Linear(n_har, 1)

        def forward(self, x, h):  # x (B, 300, C), h (B, n_har)
            z = x.transpose(1, 2)
            for cv in self.convs:
                z = torch.relu(cv(z))
            pooled = torch.cat([z[:, :, -10:].mean(2), z[:, :, -60:].mean(2), z.mean(2), z.amax(2)], dim=1)
            return (self.head(pooled) + self.skip(h)).squeeze(-1)

    return CNN()


def make_linear(n_har: int = C * len(HAR_W)):
    from ai_trader.ml.models import _torch

    torch = _torch()

    class Lin(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.skip = torch.nn.Linear(n_har, 1)

        def forward(self, x, h):
            return self.skip(h).squeeze(-1)

    return Lin()


def qlike_loss(f, r2, base):
    """Mean QLIKE (up to constants) of the variance forecast base * exp(f) for r^2: r^2 / (base e^f) + f."""
    from ai_trader.ml.models import _torch

    torch = _torch()
    return (r2 / base * torch.exp(-f) + f).mean()


__all__ = ["C", "CHANNELS", "HAR_W", "HORIZON", "MIN_TRADED_SECONDS", "Store", "WIN", "decision_seconds", "har_features", "load_symbol_day",
           "make_cnn", "make_linear", "qlike_loss", "symbol_day_channels", "targets"]
