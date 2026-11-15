"""Passive-fill model for post-only quotes on 3 s candles (H-SIM1).

A quote at price P (side +1 buy / -1 sell) rests over 3 s candles j. A candle "trades through" the quote when
    touch variant :  buy  low_j <= P        sell  high_j >= P
    strict variant:  buy  low_j <  P - tick/2   sell  high_j > P + tick/2
Per-candle fill probability (a discrete-time hazard), with w_j the fraction of candle j the quote was resting:
    p_j = w_j * theta(pen_j)   if candle j trades through
    p_j = w_j * eps            otherwise (fills the candles cannot explain: missed prints, own print only, ...)
    theta(pen) = 1 - (1 - theta0) * exp(-b * pen)      pen = penetration beyond P in bps (0 at a touch)
theta0 is the queue-survival factor at the touch; b >= 0 makes deeper trade-throughs likelier to reach us.
Fitted by maximum likelihood on the at-risk candles of each quote: a filled quote contributes
prod_{j<k}(1 - p_j) * p_k (k = the candle of its first fill), an unfilled one prod_j (1 - p_j) (censored at cancel).

Prediction for held-out quotes uses the compensator: E[#fills] = sum over at-risk candles of p_j (unbiased under
the model whatever the cancel policy), and the predicted 60 s markout is the hazard-weighted mean of the markout a
fill at each candle would have had: m_j = side * (mid_{j+20} - P) / P in bps (mid at the close of the candle
60 s after the fill candle). Observed markouts use the actual fill candle the same way.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np
from scipy.optimize import minimize

from ai_trader.sim.flow_data import STEP_MS, Grid

MARKOUT_CANDLES = 20  # 60 s


@dataclass
class FillModel:
    variant: str = "touch"      # touch | strict
    theta0: float = 0.3
    b: float = 0.0              # per bp of penetration
    eps: float = 0.0

    def theta(self, pen_bps):
        return 1.0 - (1.0 - self.theta0) * np.exp(-self.b * np.maximum(pen_bps, 0.0))

    def through(self, side: int, price: float, high: float, low: float, tick: float) -> tuple[bool, float]:
        """(trades through?, penetration in bps) for one candle (high / low NaN = no trade)."""
        if side > 0:
            if not (low == low):  # NaN
                return False, 0.0
            hit = low <= price + 1e-12 * price if self.variant == "touch" else low < price - 0.5 * tick
            return bool(hit), max(0.0, (price - low) / price * 1e4)
        if not (high == high):
            return False, 0.0
        hit = high >= price - 1e-12 * price if self.variant == "touch" else high > price + 0.5 * tick
        return bool(hit), max(0.0, (high - price) / price * 1e4)

    def p_candle(self, side: int, price: float, high: float, low: float, tick: float, w: float = 1.0) -> float:
        hit, pen = self.through(side, price, high, low, tick)
        if hit:
            return w * (1.0 - (1.0 - self.theta0) * math.exp(-self.b * pen))
        return w * self.eps

    def to_json(self) -> dict:
        return asdict(self)


# ------------------------------------------------------------------------------------- quote exposure

@dataclass
class Exposure:
    """Flattened at-risk candles of a set of quotes (one row per quote-candle pair)."""
    q: np.ndarray       # quote index
    w: np.ndarray       # overlap weight
    hit_touch: np.ndarray
    hit_strict: np.ndarray
    pen: np.ndarray     # penetration bps (0 if no trade)
    fill: np.ndarray    # this candle is the quote's fill candle
    mark: np.ndarray    # markout bps of a fill in this candle (NaN if the later mid is unknown)


def quote_exposure(quotes: list[dict], grids: dict[str, Grid], ticks: dict[str, float], min_cov: float = 0.95):
    """At-risk candles for each quote. Quote dicts: symbol, side, price, created_ms, end_ms, fill_ms (or None).

    Returns (exposure, per-quote info list). A quote is usable when its symbol has a grid, >= min_cov of its at-risk
    candle weight is present in the lake, its fill candle (if any) is present, and the 60 s markout mid is known."""
    rows = {k: [] for k in ("q", "w", "hit_touch", "hit_strict", "pen", "fill", "mark")}
    info = []
    for i, x in enumerate(quotes):
        g = grids.get(x["symbol"])
        rec = {"i": i, "usable": False, "reason": "", "obs_mark": np.nan}
        info.append(rec)
        if g is None or len(g) == 0:
            rec["reason"] = "no_flow"
            continue
        c = int(x["created_ms"])
        e = int(x["fill_ms"]) if x.get("fill_ms") is not None else int(x["end_ms"])
        t0 = int(g.t_ms[0])
        a = (c - t0) // STEP_MS
        b = (e - t0) // STEP_MS  # candle containing the end (fill candle for filled quotes)
        if a < 0 or b >= len(g) or e <= c:
            rec["reason"] = "outside_flow" if e > c else "zero_life"
            continue
        js = np.arange(a, b + 1)
        lo = np.maximum(g.t_ms[js], c)
        hi = g.t_ms[js] + STEP_MS
        if x.get("fill_ms") is None:
            hi = np.minimum(hi, e)  # cancelled: at risk until the cancel
        w = np.clip((hi - lo) / STEP_MS, 0.0, 1.0)
        keep = w > 0
        js, w = js[keep], w[keep]
        if len(js) == 0:
            rec["reason"] = "zero_life"
            continue
        pres = g.present[js]
        cov = float(w[pres].sum() / w.sum())
        rec["coverage"] = cov
        filled = x.get("fill_ms") is not None
        if cov < min_cov or (filled and not g.present[js[-1]]):
            rec["reason"] = "gap"
            continue
        side, P, tick = int(x["side"]), float(x["price"]), float(ticks.get(x["symbol"], 0.0))
        hi_, lo_ = g.high[js], g.low[js]
        if side > 0:
            pen = np.where(np.isfinite(lo_), (P - lo_) / P * 1e4, -np.inf)
            touch = np.isfinite(lo_) & (lo_ <= P * (1 + 1e-12))
            strict = np.isfinite(lo_) & (lo_ < P - 0.5 * tick)
        else:
            pen = np.where(np.isfinite(hi_), (hi_ - P) / P * 1e4, -np.inf)
            touch = np.isfinite(hi_) & (hi_ >= P * (1 - 1e-12))
            strict = np.isfinite(hi_) & (hi_ > P + 0.5 * tick)
        pen = np.maximum(pen, 0.0)
        mk = js + MARKOUT_CANDLES
        midf = np.full(len(js), np.nan)
        inside = mk < len(g)
        midf[inside] = np.where(g.ok[mk[inside]], g.mid[mk[inside]], np.nan)
        mark = side * (midf - P) / P * 1e4
        if filled and not np.isfinite(mark[-1]):
            rec["reason"] = "no_markout_mid"
            continue
        fl = np.zeros(len(js), dtype=bool)
        if filled:
            fl[-1] = True
            rec["obs_mark"] = float(mark[-1])
        pm = pres  # missing candles: no information, not at risk in the likelihood
        for k, v in (("q", np.full(pm.sum(), i)), ("w", w[pm]), ("hit_touch", touch[pm]), ("hit_strict", strict[pm]),
                     ("pen", pen[pm]), ("fill", fl[pm]), ("mark", mark[pm])):
            rows[k].append(v)
        rec.update(usable=True, filled=filled, n_candles=int(pm.sum()), self_touch=None)
    ex = Exposure(*(np.concatenate(rows[k]) if rows[k] else np.zeros(0) for k in ("q", "w", "hit_touch", "hit_strict", "pen", "fill", "mark")))
    ex.q = ex.q.astype(np.int64)
    ex.hit_touch, ex.hit_strict, ex.fill = ex.hit_touch.astype(bool), ex.hit_strict.astype(bool), ex.fill.astype(bool)
    return ex, info


# ------------------------------------------------------------------------------------------- fitting

def _hazard(ex: Exposure, m: FillModel) -> np.ndarray:
    hit = ex.hit_touch if m.variant == "touch" else ex.hit_strict
    th = 1.0 - (1.0 - m.theta0) * np.exp(-m.b * ex.pen)
    return np.clip(ex.w * np.where(hit, th, m.eps), 1e-12, 1 - 1e-12)


def loglik(ex: Exposure, m: FillModel, sel: np.ndarray | None = None) -> float:
    p = _hazard(ex, m)
    ll = np.where(ex.fill, np.log(p), np.log1p(-p))
    return float(ll[sel].sum() if sel is not None else ll.sum())


def _sig(x):
    return 1.0 / (1.0 + math.exp(-min(max(x, -40.0), 40.0)))


def fit(ex: Exposure, variant: str, sel: np.ndarray | None = None) -> tuple[FillModel, dict]:
    """MLE of (theta0, b, eps) on the rows selected by `sel` (bool over exposure rows)."""
    def unpack(z):
        return FillModel(variant, _sig(z[0]), math.exp(min(max(z[1], -20.0), 3.0)), _sig(z[2]))

    def nll(z):
        return -loglik(ex, unpack(z), sel)

    best = None
    for z0 in ([-1.0, -2.0, -7.0], [-3.0, 0.0, -9.0], [0.5, -4.0, -6.0]):
        r = minimize(nll, np.array(z0), method="Nelder-Mead", options={"xatol": 1e-6, "fatol": 1e-8, "maxiter": 4000})
        if best is None or r.fun < best.fun:
            best = r
    m = unpack(best.x)
    n_rows = int(sel.sum()) if sel is not None else len(ex.q)
    return m, {"nll": float(best.fun), "aic": float(2 * best.fun + 6), "rows": n_rows, "converged": bool(best.success)}


def predict(ex: Exposure, m: FillModel, sel: np.ndarray) -> dict:
    """Compensator prediction over the selected rows: expected fills and hazard-weighted markout."""
    p = _hazard(ex, m)[sel]
    mk = ex.mark[sel]
    good = np.isfinite(mk)
    return {"exp_fills": float(p.sum()), "pred_markout_bps": float((p[good] * mk[good]).sum() / max(p[good].sum(), 1e-300))}


def predict_by_quote(ex: Exposure, m: FillModel, n_quotes: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per quote: expected fills over its at-risk candles and sum of hazard * markout (for block bootstraps)."""
    p = _hazard(ex, m)
    good = np.isfinite(ex.mark)
    ef = np.bincount(ex.q, weights=p, minlength=n_quotes)
    pm = np.bincount(ex.q[good], weights=(p * ex.mark)[good], minlength=n_quotes)
    pw = np.bincount(ex.q[good], weights=p[good], minlength=n_quotes)
    return ef, np.where(pw > 0, pm / np.maximum(pw, 1e-300), np.nan), pw


def fill_prob_static(m: FillModel, side: int, price: float, highs: np.ndarray, lows: np.ndarray, tick: float) -> float:
    """P(fill) of a quote resting over the given candles (used by tests: monotone in distance)."""
    s = 1.0
    for h, lo in zip(highs, lows, strict=True):
        s *= 1.0 - m.p_candle(side, price, h, lo, tick)
    return 1.0 - s
