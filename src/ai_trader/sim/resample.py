"""Resampling generators of synthetic price paths (H-SYN1, H-SYN2; research/plans/2026-11-11_strategy_program.md, B5).

Use: generated paths regularise parameter choices for DETERMINISTIC strategies only. They never train,
prompt or backtest the LLM decision layer (CLAUDE.md principle 1). A generator must be fed one data segment
(e.g. llm_in_sample bars only) and never sees anything after it.

Three pieces:

(a) Stationary block bootstrap (Politis & Romano 1994). Source = the sequence of real 1h bars. A path is a
    concatenation of blocks; each block starts at a uniformly drawn source bar and its length is geometric
    with mean b (implemented as "start a new block at each step with probability 1/b"); blocks wrap
    circularly at the end of the source. b is the Politis-White (2004) automatic block length with the
    Patton-Politis-White (2009) correction (arch.bootstrap.optimal_block_length, column "stationary"), computed on
    |r| so that the blocks are long enough to carry volatility clustering, capped at min(3 sqrt(n), n/3)
    (Politis-White's b_max).

(b) Regime-switching resampling. The source bars carry v0 regime cells (research.regimes.label, causal).
    fit_transition() estimates the cell transition matrix from consecutive labelled bars; simulate_chain()
    draws a cell path from it; regime_indices() fills each simulated cell run with stationary blocks drawn
    from the pool of real bars in that same cell (pool = that cell's bars in time order; a block continues
    along the pool and therefore may join two episodes of the same cell, like the circular wrap in (a)).
    A new block starts whenever the simulated cell changes, and otherwise with probability 1/b.

(c) Path reconstruction to OHLC. We resample WHOLE BARS as relative OHLC tuples, never closes alone:
        r_t = log(c_t / c_{t-1})                   close-to-close log return
        g_t = log(o_t / c_{t-1})                   open gap (about 0 on 24/7 perps)
        u_t = log(h_t / max(o_t, c_t)) >= 0        upper wick
        d_t = log(min(o_t, c_t) / l_t) >= 0        lower wick
    A generated path starts at a close c_0 and rebuilds c_t = c_{t-1} e^{r_t}, o_t = c_{t-1} e^{g_t},
    h_t = max(o_t, c_t) e^{u_t}, l_t = min(o_t, c_t) e^{-d_t}. So high >= max(open, close) and
    low <= min(open, close) hold by construction, and the joint law of (return, gap, wicks) of a real bar is
    kept (wicks scale with that bar's volatility). aggregate() rolls k consecutive bars into one (open of
    the first, max high, min low, close of the last), e.g. 1h -> 4h for the trend zoo. Volume is not
    generated (no zoo or breakout variant uses it).

All functions are vectorised over paths: arrays are (n_paths, n_bars).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np


# ---------------------------------------------------------------- (c) relative bars and reconstruction
@dataclass(frozen=True)
class RelBars:
    """Relative OHLC of consecutive source bars (bar i of the table = source bar i + 1)."""

    r: np.ndarray
    g: np.ndarray
    u: np.ndarray
    d: np.ndarray

    def __len__(self) -> int:
        return len(self.r)


def relative_bars(open_: np.ndarray, high: np.ndarray, low: np.ndarray, close: np.ndarray) -> RelBars:
    """Relative OHLC tuples of bars 1..n-1 (bar 0 only supplies the previous close)."""
    o, h, lo, c = (np.asarray(x, dtype=float) for x in (open_, high, low, close))
    if not (np.all(o > 0) and np.all(lo > 0) and np.all(h >= np.maximum(o, c)) and np.all(lo <= np.minimum(o, c))):
        raise ValueError("invalid source bars: need positive prices with low <= min(open, close) <= max(open, close) <= high")
    lc, lo_, lh, ll = np.log(c), np.log(o), np.log(h), np.log(lo)
    r = np.diff(lc)
    g = lo_[1:] - lc[:-1]
    u = lh[1:] - np.maximum(lo_[1:], lc[1:])
    d = np.minimum(lo_[1:], lc[1:]) - ll[1:]
    return RelBars(r=r, g=g, u=np.maximum(u, 0.0), d=np.maximum(d, 0.0))


def reconstruct(rb: RelBars, idx: np.ndarray, c0: float) -> dict[str, np.ndarray]:
    """OHLC paths from resampled bar indices idx (n_paths, n) into rb, starting from close c0."""
    idx = np.atleast_2d(idx)
    r, g, u, d = rb.r[idx], rb.g[idx], rb.u[idx], rb.d[idx]
    lc = np.log(c0) + np.cumsum(r, axis=1)
    lc_prev = np.concatenate([np.full((idx.shape[0], 1), np.log(c0)), lc[:, :-1]], axis=1)
    lo_ = lc_prev + g
    hi = np.maximum(lo_, lc) + u
    lw = np.minimum(lo_, lc) - d
    return {"open": np.exp(lo_), "high": np.exp(hi), "low": np.exp(lw), "close": np.exp(lc)}


def aggregate(ohlc: dict[str, np.ndarray], k: int) -> dict[str, np.ndarray]:
    """Roll k consecutive bars into one; a trailing partial group is dropped."""
    n = ohlc["close"].shape[1] // k * k
    p = ohlc["close"].shape[0]

    def blk(a: np.ndarray) -> np.ndarray:
        return a[:, :n].reshape(p, n // k, k)

    return {"open": blk(ohlc["open"])[:, :, 0], "high": blk(ohlc["high"]).max(axis=2),
            "low": blk(ohlc["low"]).min(axis=2), "close": blk(ohlc["close"])[:, :, -1]}


# ---------------------------------------------------------------- (a) stationary block bootstrap
def politis_white_block(x: np.ndarray, cap: bool = True) -> dict[str, float]:
    """Politis-White / PPW expected block length for the stationary bootstrap (arch's implementation)."""
    from arch.bootstrap import optimal_block_length

    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)
    raw = float(optimal_block_length(x)["stationary"].iloc[0])
    b_max = float(np.ceil(min(3 * np.sqrt(n), n / 3)))
    b = min(max(raw, 1.0), b_max) if cap else max(raw, 1.0)
    return {"b": b, "b_raw": raw, "b_max": b_max, "n": n}


def _fill_blocks(new: np.ndarray, start_pos: np.ndarray, pool_len: np.ndarray) -> np.ndarray:
    """Positions along a (circular) pool: a block begins where new is True at start_pos and advances by 1.

    new, start_pos, pool_len: (n_paths, n). start_pos/pool_len are read at block starts only."""
    n = new.shape[1]
    t = np.arange(n)
    new = new.copy()
    new[:, 0] = True
    # step index of each block's first bar, broadcast to every step of the block
    first_t = np.maximum.accumulate(np.where(new, t[None, :], 0), axis=1)
    off = t[None, :] - first_t
    # values drawn at the block start, propagated forward
    sp = np.take_along_axis(start_pos, first_t, axis=1)
    pl_ = np.take_along_axis(pool_len, first_t, axis=1)
    return (sp + off) % pl_


def stationary_indices(n_src: int, n_out: int, mean_block: float, n_paths: int, rng: np.random.Generator) -> np.ndarray:
    """(n_paths, n_out) source indices of a stationary (circular) block bootstrap with mean block length b."""
    q = 1.0 / max(float(mean_block), 1.0)
    new = rng.random((n_paths, n_out)) < q
    start = rng.integers(0, n_src, size=(n_paths, n_out))
    return _fill_blocks(new, start, np.full((n_paths, n_out), n_src))


# ---------------------------------------------------------------- (b) regime-switching resampling
def fit_transition(cells: Sequence[str | None]) -> dict:
    """Transition matrix of a cell sequence from consecutive labelled pairs (None breaks a pair).

    Returns states (sorted), counts C[i, j], row-stochastic P, and the empirical state frequencies."""
    states = sorted({c for c in cells if c is not None})
    ix = {s: i for i, s in enumerate(states)}
    k = len(states)
    cnt = np.zeros((k, k))
    for a, b in zip(cells[:-1], cells[1:], strict=True):
        if a is not None and b is not None:
            cnt[ix[a], ix[b]] += 1
    rs = cnt.sum(axis=1, keepdims=True)
    # A state seen only as the last labelled bar has no outgoing pair: let it stay put.
    P = np.where(rs > 0, cnt / np.where(rs > 0, rs, 1.0), np.eye(k))
    freq = np.array([sum(1 for c in cells if c == s) for s in states], dtype=float)
    return {"states": states, "counts": cnt, "P": P, "freq": freq / freq.sum()}


def simulate_chain(P: np.ndarray, n: int, n_paths: int, rng: np.random.Generator, start_probs: np.ndarray | None = None) -> np.ndarray:
    """(n_paths, n) integer state paths of a Markov chain with transition matrix P."""
    k = P.shape[0]
    cum = np.cumsum(P, axis=1)
    cum[:, -1] = 1.0
    sp = np.full(k, 1.0 / k) if start_probs is None else np.asarray(start_probs, dtype=float) / np.sum(start_probs)
    out = np.empty((n_paths, n), dtype=np.int64)
    out[:, 0] = np.minimum(np.searchsorted(np.cumsum(sp), rng.random(n_paths), side="right"), k - 1)
    u = rng.random((n_paths, n))
    for t in range(1, n):
        s = out[:, t - 1]
        out[:, t] = (cum[s] <= u[:, t, None]).sum(axis=1)
    return np.minimum(out, k - 1)


def regime_indices(src_state: np.ndarray, chain: np.ndarray, mean_block: float, rng: np.random.Generator) -> np.ndarray:
    """Source indices that follow a simulated state path, with stationary blocks drawn within each state.

    src_state: (n_src,) integer state of each source bar, -1 = unlabelled (never drawn).
    chain: (n_paths, n) simulated states (from simulate_chain)."""
    src_state = np.asarray(src_state)
    k = int(chain.max()) + 1
    pools = [np.flatnonzero(src_state == s) for s in range(k)]
    lens = np.array([len(p) for p in pools])
    if np.any(lens[np.unique(chain)] == 0):
        raise ValueError("the chain visits a state with no source bars")
    offs = np.r_[0, np.cumsum(lens)[:-1]]
    flat = np.concatenate([p for p in pools if len(p)]) if lens.sum() else np.array([], dtype=int)
    q = 1.0 / max(float(mean_block), 1.0)
    p_, n = chain.shape
    switch = np.zeros((p_, n), dtype=bool)
    switch[:, 1:] = chain[:, 1:] != chain[:, :-1]
    new = switch | (rng.random((p_, n)) < q)
    pool_len = lens[chain]
    start = (rng.random((p_, n)) * pool_len).astype(np.int64)
    pos = _fill_blocks(new, start, np.maximum(pool_len, 1))
    return flat[offs[chain] + pos]


# ---------------------------------------------------------------- diagnostics helpers
def excess_kurtosis(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    m = x.mean(axis=axis, keepdims=True)
    v = ((x - m) ** 2).mean(axis=axis)
    return ((x - m) ** 4).mean(axis=axis) / v**2 - 3.0


def acf(x: np.ndarray, lags: Sequence[int], axis: int = -1) -> np.ndarray:
    """Sample autocorrelation at the given lags along the last axis; output (..., len(lags))."""
    x = np.moveaxis(np.asarray(x, dtype=float), axis, -1)
    x = x - x.mean(axis=-1, keepdims=True)
    den = (x * x).sum(axis=-1)
    return np.stack([(x[..., k:] * x[..., :-k]).sum(axis=-1) / den for k in lags], axis=-1)


def run_lengths(states: Sequence) -> np.ndarray:
    """Lengths of maximal runs of equal, non-None values (episode durations in bars)."""
    out, prev, n = [], object(), 0
    for s in states:
        if s is None:
            if n:
                out.append(n)
            prev, n = object(), 0
            continue
        if s == prev:
            n += 1
        else:
            if n:
                out.append(n)
            prev, n = s, 1
    if n:
        out.append(n)
    return np.asarray(out, dtype=float)
