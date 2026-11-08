"""Stationarity and standardisation (ARCHITECTURE §1.1–1.3).

- log returns and one-step-ahead volatility standardisation (no look-ahead: z_t = r_t / σ̂_{t|t-1})
- ADF + KPSS joint verdict
- fixed-width-window fractional differencing (López de Prado, AFML ch. 5) and minimal-d search
- Population Stability Index for drift monitoring
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
from statsmodels.tsa.stattools import adfuller, kpss


def log_returns(close: np.ndarray) -> np.ndarray:
    close = np.asarray(close, dtype=float)
    out = np.full_like(close, np.nan)
    out[1:] = np.diff(np.log(close))
    return out


def ewma_sigma_ahead(returns: np.ndarray, lam: float = 0.94, warmup: int = 20) -> np.ndarray:
    """σ̂_{t|t-1}: out[t] uses returns up to t-1 only. Same alignment as at_filters::ewma."""
    r = np.asarray(returns, dtype=float)
    out = np.full_like(r, np.nan)
    if len(r) <= warmup:
        return out
    var = float(np.nanmean(r[:warmup] ** 2))
    out[warmup] = np.sqrt(var)
    for t in range(warmup, len(r) - 1):
        if np.isfinite(r[t]):
            var = lam * var + (1 - lam) * r[t] ** 2
        out[t + 1] = np.sqrt(var)
    return out


def vol_standardise(returns: np.ndarray, sigma_ahead: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.asarray(returns) / np.asarray(sigma_ahead)


@dataclass(frozen=True)
class StationarityVerdict:
    adf_p: float
    kpss_p: float
    stationary: bool  # ADF rejects unit root AND KPSS fails to reject stationarity


def adf_kpss(x: np.ndarray, alpha: float = 0.05) -> StationarityVerdict:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    adf_p = float(adfuller(x, autolag="AIC", result_object=False)[1])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # KPSS warns when p is outside its lookup table
        kpss_p = float(kpss(x, regression="c", nlags="auto")[1])
    return StationarityVerdict(adf_p, kpss_p, adf_p < alpha and kpss_p > alpha)


def fracdiff_weights(d: float, threshold: float = 1e-4, max_len: int = 10_000) -> np.ndarray:
    """Weights w_k of (1-B)^d, truncated when |w_k| < threshold (fixed-width window)."""
    w = [1.0]
    k = 1
    while k < max_len:
        wk = -w[-1] * (d - k + 1) / k
        if abs(wk) < threshold:
            break
        w.append(wk)
        k += 1
    return np.array(w)


def fracdiff(x: np.ndarray, d: float, threshold: float = 1e-4) -> np.ndarray:
    """Fixed-width fractional difference. out[t] uses x[t-width+1..t] only (causal)."""
    x = np.asarray(x, dtype=float)
    w = fracdiff_weights(d, threshold)
    width = len(w)
    out = np.full_like(x, np.nan)
    for t in range(width - 1, len(x)):
        out[t] = np.dot(w, x[t - width + 1 : t + 1][::-1])
    return out


def min_fracdiff_d(x: np.ndarray, grid: np.ndarray | None = None, alpha: float = 0.05, threshold: float = 1e-4) -> float | None:
    """Smallest d on the grid whose fractional difference passes ADF+KPSS (keeps maximum memory)."""
    for d in grid if grid is not None else np.round(np.arange(0.0, 1.01, 0.05), 2):
        y = fracdiff(x, float(d), threshold)
        y = y[np.isfinite(y)]
        if len(y) > 50 and adf_kpss(y, alpha).stationary:
            return float(d)
    return None


def psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    """Population Stability Index of `actual` vs `expected` using expected-quantile bins. >0.25 = drifted."""
    e = np.asarray(expected, dtype=float)
    a = np.asarray(actual, dtype=float)
    e, a = e[np.isfinite(e)], a[np.isfinite(a)]
    edges = np.unique(np.quantile(e, np.linspace(0, 1, bins + 1)))
    edges[0], edges[-1] = -np.inf, np.inf
    pe = np.histogram(e, edges)[0] / len(e)
    pa = np.histogram(a, edges)[0] / len(a)
    pe, pa = np.clip(pe, 1e-6, None), np.clip(pa, 1e-6, None)
    return float(np.sum((pa - pe) * np.log(pa / pe)))
