"""WMS (program WMM2): Friday-known mean-reversion / trend indices and the frozen weekend quoting selection rule.

Definitions are those registered in research/hypotheses.yaml (H-WMS-1..13) and the diary entry "WMS design frozen before results"
(2026-11-27). Everything here is a pure function of arrays that end at the Friday cutoff (Fri 22:00 UTC); nothing reads data.

Indices (on log prices of last-trade closes on a complete 5m grid; 15m / 1h series are every 3rd / 12th 5m close):
  variance_ratio  Lo-MacKinlay (1988) overlapping VR with the unbiased estimators and the heteroskedasticity-robust z*.
  ar1             OLS AR(1) of returns with Kendall's bias correction rho + (1 + 3 rho) / n.
  ou_half_life    half-life (hours) of the log price around its window mean: dp_t = a + b p_{t-1}, -ln 2 / ln(1 + b).
  dfa_alpha       DFA-1 exponent of returns (profile, non-overlapping boxes, linear detrend).
  efficiency      Kaufman efficiency ratio |p_end - p_start| / sum |dp|.
  bos_events      breaks of structure on fractal pivots (strict, k bars each side, confirmed k bars later, consumed when broken).

Rule (R-WMS, frozen): S0 = mean of the within-weekend Blom normal scores of -log vr15_4, -log vr15_16, -ar1_15, -dfa, -er_1h,
-bos_imb; eligible_name = top third of S0 in the weekend universe (rank <= ceil(n / 3)); eligible = eligible_name AND the weekend
gate (calm_p >= 0.5) when the gate is enabled.
"""

from __future__ import annotations

import math
from datetime import UTC, date, datetime, timedelta

import numpy as np
import polars as pl
from scipy.stats import norm

BAR_MS = 300_000
FW_BARS = 2016  # 7 days of 5m bars
WE_BARS = 576  # Sat 00:00 -> Mon 00:00
GAP_BARS = 24  # Fri 22:00 -> Sat 00:00 (neither window)
DFA_SCALES_FW = (12, 18, 24, 36, 48, 72, 96, 144, 192, 288)
DFA_SCALES_WE = (12, 18, 24, 36, 48, 72, 96)
HL_CAP_H = 1680.0
S0_COMPONENTS = {"vr15_4": "neglog", "vr15_16": "neglog", "ar1_15": "neg", "dfa": "neg", "er_1h": "neg", "bos_imb": "neg"}
TOP_FRAC = 1 / 3
CALM_P = 0.5
# Frozen 2026-11-28 by the pre-registered fallback: S0 failed H-WMS-3 in FIT, so the score is S1 = the FIT-passing indices of
# H-WMS-4 (BH q < 0.05, mean-reversion sign), equal weight (an extra, labelled trial). Higher S1 = more mean-reverting, fewer breaks.
S1_COMPONENTS = {"vr5_2": "neglog", "vr5_12": "neglog", "vr5_48": "neglog", "vr15_16": "neglog", "ar1_5": "neg", "ar1_15": "neg",
                 "bos_count": "neg"}
CALM_X = ("x1_rvfw_vs_med", "x2_day_vs_week", "x3_dvol_vs_med", "x4_funding", "x5_oi_chg7", "x6_release_fri", "x7_release_montue",
          "x8_holiday")


# ---------------------------------------------------------------- calendar

def friday_cutoff(weekend_start: date) -> datetime:
    """Fri 22:00 UTC before the weekend that starts on Saturday `weekend_start`."""
    sat = datetime(weekend_start.year, weekend_start.month, weekend_start.day, tzinfo=UTC)
    if sat.weekday() != 5:
        raise ValueError(f"{weekend_start} is not a Saturday")
    return sat - timedelta(hours=2)


def window_start(weekend_start: date) -> datetime:
    return friday_cutoff(weekend_start) - timedelta(days=7)


def thanksgiving(year: int) -> date:
    d = date(year, 11, 1)
    first_thu = d + timedelta(days=(3 - d.weekday()) % 7)
    return first_thu + timedelta(days=21)


def holiday_flag(weekend_start: date) -> str:
    """'thanksgiving' (the weekend after US Thanksgiving), 'christmas' (Sat in Dec 22-28), 'new_year' (Sat in Dec 29 - Jan 4), or ''."""
    if weekend_start == thanksgiving(weekend_start.year) + timedelta(days=2):
        return "thanksgiving"
    m, d = weekend_start.month, weekend_start.day
    if m == 12 and 22 <= d <= 28:
        return "christmas"
    if (m == 12 and d >= 29) or (m == 1 and d <= 4):
        return "new_year"
    return ""


# ---------------------------------------------------------------- indices

def variance_ratio(r: np.ndarray, q: int) -> tuple[float, float]:
    """Lo-MacKinlay VR(q) on base returns r (overlapping q-sums, unbiased variances) and the robust z*."""
    r = np.asarray(r, dtype=np.float64)
    n = r.size
    if n < 4 * q or q < 2:
        return math.nan, math.nan
    mu = r.mean()
    e = r - mu
    ss = float(e @ e)
    if ss <= 0:
        return math.nan, math.nan
    var_a = ss / (n - 1)
    p = np.concatenate([[0.0], np.cumsum(r)])
    rq = p[q:] - p[:-q] - q * mu
    m = q * (n - q + 1) * (1 - q / n)
    vr = float(rq @ rq) / m / var_a
    e2 = e * e
    theta = 0.0
    for j in range(1, q):
        delta = n * float(e2[j:] @ e2[:-j]) / ss**2
        theta += (2 * (q - j) / q) ** 2 * delta
    z = (vr - 1) * math.sqrt(n) / math.sqrt(theta) if theta > 0 else math.nan
    return vr, z


def ar1(r: np.ndarray) -> float:
    r = np.asarray(r, dtype=np.float64)
    if r.size < 10:
        return math.nan
    x, y = r[:-1] - r[:-1].mean(), r[1:] - r[1:].mean()
    den = float(x @ x)
    if den <= 0:
        return math.nan
    rho = float(x @ y) / den
    return rho + (1 + 3 * rho) / r.size


def ou_half_life(p: np.ndarray, bar_hours: float) -> float:
    """Half-life in hours of the log price p around its window mean (capped at HL_CAP_H; random walks hit the cap often)."""
    p = np.asarray(p, dtype=np.float64)
    if p.size < 20:
        return math.nan
    x, dy = p[:-1] - p[:-1].mean(), np.diff(p)
    den = float(x @ x)
    if den <= 0:
        return math.nan
    b = float(x @ (dy - dy.mean())) / den
    if b >= 0 or b <= -1:
        return HL_CAP_H if b >= 0 else bar_hours * 0.5
    return min(HL_CAP_H, -math.log(2) / math.log1p(b) * bar_hours)


def dfa_alpha(r: np.ndarray, scales=DFA_SCALES_FW) -> float:
    r = np.asarray(r, dtype=np.float64)
    y = np.cumsum(r - r.mean())
    ls, lf = [], []
    for s in scales:
        nb = y.size // s
        if nb < 2:
            continue
        seg = y[: nb * s].reshape(nb, s)
        t = np.arange(s, dtype=np.float64)
        tc = t - t.mean()
        slope = (seg - seg.mean(axis=1, keepdims=True)) @ tc / float(tc @ tc)
        resid = seg - seg.mean(axis=1, keepdims=True) - np.outer(slope, tc)
        f2 = float((resid**2).mean())
        if f2 > 0:
            ls.append(math.log(s))
            lf.append(0.5 * math.log(f2))
    if len(ls) < 3:
        return math.nan
    return float(np.polyfit(ls, lf, 1)[0])


def efficiency(p: np.ndarray) -> float:
    p = np.asarray(p, dtype=np.float64)
    path = float(np.abs(np.diff(p)).sum())
    return abs(float(p[-1] - p[0])) / path if path > 0 else math.nan


def pivots(h: np.ndarray, lo: np.ndarray, k: int = 3) -> tuple[np.ndarray, np.ndarray]:
    """Boolean arrays: swing high / swing low at bar i (strictly beyond the k bars on each side)."""
    n = h.size
    sh, sl = np.zeros(n, bool), np.zeros(n, bool)
    if n < 2 * k + 1:
        return sh, sl
    from numpy.lib.stride_tricks import sliding_window_view as sw
    wh, wl = sw(h, 2 * k + 1), sw(lo, 2 * k + 1)
    ch, cl = wh[:, k], wl[:, k]
    oh = np.delete(wh, k, axis=1).max(axis=1)
    ol = np.delete(wl, k, axis=1).min(axis=1)
    sh[k: n - k] = ch > oh
    sl[k: n - k] = cl < ol
    return sh, sl


def bos_events(h: np.ndarray, lo: np.ndarray, c: np.ndarray, k: int = 3) -> list[tuple[int, int, float]]:
    """Breaks of structure: (bar t, +1 bull / -1 bear, broken level). A pivot at i is usable from bar i + k + 1 (confirmed at the
    close of bar i + k); the most recent confirmed pivot replaces the active one; a break consumes it."""
    h, lo, c = (np.asarray(a, dtype=np.float64) for a in (h, lo, c))
    sh, sl = pivots(h, lo, k)
    act_h = act_l = None
    out: list[tuple[int, int, float]] = []
    for t in range(c.size):
        i = t - k - 1
        if i >= 0:
            if sh[i]:
                act_h = h[i]
            if sl[i]:
                act_l = lo[i]
        if act_h is not None and c[t] > act_h:
            out.append((t, 1, act_h))
            act_h = None
        if act_l is not None and c[t] < act_l:
            out.append((t, -1, act_l))
            act_l = None
    return out


def agg(h5: np.ndarray, l5: np.ndarray, c5: np.ndarray, f: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Aggregate a complete 5m grid (length multiple of f) to f x 5m bars."""
    n = c5.size // f * f
    return h5[:n].reshape(-1, f).max(axis=1), l5[:n].reshape(-1, f).min(axis=1), c5[:n].reshape(-1, f)[:, -1]


def indices(h5: np.ndarray, l5: np.ndarray, c5: np.ndarray, p0: float | None = None, dfa_scales=DFA_SCALES_FW,
            bos_lead: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None) -> dict[str, float]:
    """All registered indices of one window. c5 are log closes; p0 the log close just before the window (for the first return).
    bos_lead: 15m (h, l, c) log bars before the window, so pivots formed earlier can be broken inside it (weekend use)."""
    c5 = np.asarray(c5, dtype=np.float64)
    first = c5[0] if p0 is None else p0
    r5 = np.diff(np.concatenate([[first], c5]))
    h15, l15, c15 = agg(h5, l5, c5, 3)
    r15 = np.diff(np.concatenate([[first], c15]))
    c1h = c5[11::12]
    out: dict[str, float] = {}
    for q in (2, 12, 48):
        out[f"vr5_{q}"], out[f"z5_{q}"] = variance_ratio(r5, q)
    for q in (4, 16):
        out[f"vr15_{q}"], out[f"z15_{q}"] = variance_ratio(r15, q)
    out["ar1_5"], out["ar1_15"] = ar1(r5), ar1(r15)
    out["ou_hl"] = ou_half_life(c15, 0.25)
    out["dfa"] = dfa_alpha(r5, dfa_scales)
    out["er_1h"] = efficiency(np.concatenate([[first], c1h]))
    if bos_lead is not None:
        hh, ll, cc = (np.concatenate([a, b]) for a, b in zip(bos_lead, (h15, l15, c15)))
        lead = bos_lead[2].size
    else:
        hh, ll, cc, lead = h15, l15, c15, 0
    ev = [e for e in bos_events(hh, ll, cc, 3) if e[0] >= lead]
    bull = sum(1 for e in ev if e[1] > 0)
    bear = len(ev) - bull
    out["bos_count"] = float(len(ev))
    out["bos_imb"] = abs(bull - bear) / max(1, len(ev))
    out["bos_net"] = float(bull - bear)
    return out


# ---------------------------------------------------------------- rule

def blom(x: pl.Expr, by: str | None = None) -> pl.Expr:
    """Blom normal score of x within `by` groups (nulls and NaN -> null); the rank and count are per group, the probit elementwise."""
    x = x.fill_nan(None)
    rk, n = (x.rank("average").over(by), x.count().over(by)) if by else (x.rank("average"), x.count())
    return ((rk - 0.375) / (n + 0.25)).map_batches(lambda s: pl.Series(norm.ppf(s.to_numpy().astype(float))), return_dtype=pl.Float64).fill_nan(None)


def composite(df: pl.DataFrame, by: str = "weekend_start", components: dict[str, str] | None = None, name: str = "score") -> pl.DataFrame:
    """Add the equal-weight normal-score composite (higher = more mean-reverting) within each `by` group."""
    comps = components or S0_COMPONENTS
    exprs = []
    for c, how in comps.items():
        v = -pl.col(c).log() if how == "neglog" else (-pl.col(c) if how == "neg" else pl.col(c))
        exprs.append(blom(v, by).alias(f"_ns_{c}"))
    out = df.with_columns(exprs)
    cols = [f"_ns_{c}" for c in comps]
    return out.with_columns(pl.mean_horizontal(cols).alias(name)).drop(cols)


def select(df: pl.DataFrame, score: str = "score", by: str = "weekend_start", calm_p: str | None = None,
           gate: bool = False) -> pl.DataFrame:
    """Frozen rule R-WMS: eligible_name = rank of `score` (descending) <= ceil(n / 3) within the weekend universe; eligible =
    eligible_name and (calm_p >= 0.5 if gate; an unknown calm_p closes the gate)."""
    n = pl.col(score).count().over(by)
    rk = pl.col(score).rank("ordinal", descending=True).over(by)
    out = df.with_columns((rk <= (n * TOP_FRAC).ceil()).fill_null(False).alias("eligible_name"))
    if gate and calm_p:
        return out.with_columns((pl.col("eligible_name") & (pl.col(calm_p) >= CALM_P).fill_null(False)).alias("eligible"))
    return out.with_columns(pl.col("eligible_name").alias("eligible"))


def calm_probability(x: dict, rule: dict) -> float | None:
    """P(calm weekend) from the frozen H-WMS-12 logistic (data/research/wms/rule.json: coef, mu, sd); None if an input is missing."""
    if any(x.get(c) is None for c in CALM_X):
        return None
    z = rule["coef"]["const"] + sum(rule["coef"][c] * (x[c] - rule["mu"][c]) / rule["sd"][c] for c in CALM_X)
    return 1 / (1 + math.exp(-z))


__all__ = ["calm_probability", "agg", "ar1", "blom", "bos_events", "composite", "dfa_alpha", "efficiency", "friday_cutoff", "holiday_flag", "indices",
           "ou_half_life", "pivots", "select", "thanksgiving", "variance_ratio", "window_start"]
