"""Percentile of a realised move inside the network's 7-quantile predictive distribution.

The network gives quantiles q_k at taus (0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95). The CDF between knots is
interpolated linearly in probit space: z(x) is piecewise linear through (q_k, Phi^-1(tau_k)) and F(x) = Phi(z(x)).
Beyond q05 / q95 the outermost segment's slope is extended (a Gaussian-shaped tail matched to the 90 -> 95 or 5 -> 10
spacing), so F is continuous, strictly increasing and reproduces every tau exactly at its knot. Coincident knots
(zero width) are given a tiny width. The same algorithm runs in the browser (research/live/dashboard/pct.js); the
tests check both agree.
"""

from __future__ import annotations

import math
from statistics import NormalDist

TAUS = (0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95)
QNAMES = ("q05", "q10", "q25", "q50", "q75", "q90", "q95")
_N = NormalDist()
ZK = tuple(_N.inv_cdf(t) for t in TAUS)
EPS = 1e-9


def _knots(q) -> list[float]:
    q = [q.get(k) for k in QNAMES] if isinstance(q, dict) else list(q)
    if len(q) != len(TAUS) or any(v is None for v in q):
        raise ValueError("need 7 finite quantiles")
    q = [float(v) for v in q]
    if not all(math.isfinite(v) for v in q):
        raise ValueError("need 7 finite quantiles")
    out = [q[0]]
    for v in q[1:]:
        out.append(max(v, out[-1] + EPS))
    return out


def zscore(x: float, q) -> float:
    """Probit-space position z(x) of x (so that percentile = Phi(z))."""
    k = _knots(q)
    x = float(x)
    if x <= k[0]:
        i = 0
    elif x >= k[-1]:
        i = len(k) - 2
    else:
        i = max(j for j in range(len(k) - 1) if k[j] <= x)
    return ZK[i] + (x - k[i]) * (ZK[i + 1] - ZK[i]) / (k[i + 1] - k[i])


def percentile(x: float, q) -> float:
    """F(x) in (0, 1): the predictive probability of a move at or below x."""
    return _N.cdf(zscore(x, q))


def scaled_quantiles(q, frac: float) -> list[float]:
    """Quantiles scaled by sqrt(frac) (the square-root-of-time cone at a fraction `frac` of the window)."""
    s = math.sqrt(max(0.0, min(1.0, float(frac))))
    return [v * s for v in _knots(q)]


def describe(p: float) -> str:
    """Plain words for a percentile (used by the page and the tests)."""
    if p < 0.05:
        return "an extreme move down, below the network's 90% band"
    if p < 0.15:
        return "an unusually strong move down"
    if p < 0.30:
        return "a softer hour than the network's median, still ordinary"
    if p <= 0.70:
        return "an ordinary hour, close to the middle of what it expected"
    if p <= 0.85:
        return "a firmer hour than the network's median, still ordinary"
    if p <= 0.95:
        return "an unusually strong move up"
    return "an extreme move up, above the network's 90% band"


def ordinal(n: int) -> str:
    n = int(n)
    suf = "th" if 10 <= n % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suf}"


def sentence(p: float, horizon: str = "hour") -> str:
    k = min(99, max(1, math.floor(100 * p + 0.5)))  # half up, as Math.round in pct.js
    return f"BTC is at the {ordinal(k)} percentile of what the network expected for this {horizon}: {describe(p)}."


__all__ = ["QNAMES", "TAUS", "ZK", "describe", "ordinal", "percentile", "scaled_quantiles", "sentence", "zscore"]
