"""Generator of 3 s order flow, trade intensity, spread and mid returns (H-SIM2), with `rescale` for other markets.

State per 3 s candle t (fitted per symbol on contiguous ok segments; innovations Student-t with df by MLE, truncated at the
largest standardised residual seen in the fit, so exp() of a heavy tail can never produce notionals or spreads the
market never showed):
  activity   (default, activity = "copula") Gaussian copula with a latent two-component AR(1) plus noise:
             g_t = sqrt(w1) z1_t + sqrt(w2) z2_t + sqrt(1 - w1 - w2) nu_t, zk AR(1) with unit variance and coefficient phik,
             n_t = empirical quantile of the fitted trade counts at Phi(g_t); (w1, phi1, w2, phi2) by least squares on the
             autocorrelation of the normal scores of log(1 + n) at lags 1..1200 (3 s to 1 h). The slow component is the exogenous
             activity regime (sessions, news); the marginal of n is the fitted market's exactly.
             (alternative, activity = "hawkes") discrete-time Hawkes / INGARCH with two exponential kernels and negative-binomial counts:
             n_t ~ NegBin(mean lam_t, NB2 dispersion alpha),  lam_t = w X_t + L1_t + L2_t,  Lk_t = bk Lk_{t-1} + ak n_{t-1}
             (fast and slow self-excitation; branching ratio a1/(1-b1) + a2/(1-b2) < 1; NB maximum likelihood); a_t = log(1 + n_t)
             X_t = exogenous slow activity regime (sessions, news): log X AR(1) calibrated by moments on the centred 1 h mean of n
             (its variance and its autocorrelation 1 h apart); in the fit the baseline uses the observed 1 h mean
  book       o_t (obi_25) = c + rho o_{t-1} + e, clipped to [-1, 1]
  direction  q_t = (buy - sell) / (buy + sell) | n_t > 0
             = c + g1 q_{t-1} + g2 mean(q_{t-2..t-20}) + b1 r_{t-1}/sigma + b2 sum(r_{t-2..t-20})/sigma + k o_{t-1}
               + s(n_t) e, clipped to [-1, 1] (residual scale by trade-count bucket; clipping gives the mass at +-1 of
               single-print candles)
  notional   log V_t = alpha + eta log n_t + e_t, e_t AR(1) across trade candles;  F_t = q_t V_t (signed taker notional)
  spread     log s_t = c + rho log s_{t-1} + e, floored at the 1st percentile of the fitted spread
  return     r_t (bps) = 0 with probability pi0(n_t = 0 or > 0); otherwise
             r_t = l1 F_t/D + l2 sign(F_t) sqrt(|F_t|/D) + psi r_{t-1} + exp(0.5 h_t) z_t   (linear + sqrt impact, l1, l2 >= 0)
             h_t = h0 + h1 a_t + h2 log(c + mean|r|_{t-1..t-20}) + h3 log(c + mean|r|_{t-21..t-200})   (HAR variance)
  high/low   max(mid_{t-1}, mid_t) * exp((s_t/2 + e_hi sigma)/1e4), min(...) * exp(-(s_t/2 + e_lo sigma)/1e4), with
             (e_hi, e_lo) resampled jointly from the fitted candles in the same direction bucket of q_t
D = median depth within 25 bps (bid + ask), sigma = std of 3 s returns (bps), m = median notional per trade.

`rescale(target_sigma, target_depth, target_trade_size)` moves the calibration to another market: trade notional
scales with the trade size, impact uses F/D with the new depth, return scales (impact and noise) are then adjusted
on a pilot path so that the generated 3 s sigma equals target_sigma; the spread keeps its ratio to sigma unless a
target spread is given. Generated paths come back as a `Grid`, so the replay engine runs on real and generated
paths alike.
"""

from __future__ import annotations

import copy
import math
from dataclasses import asdict, dataclass, field

import numpy as np
from scipy import stats
from scipy.optimize import nnls

from ai_trader.sim.flow_data import STEP_MS, Grid

L_SHORT, L_LONG, L_HOUR = 20, 200, 1200
JUMP_Z = 6.0  # jump threshold in robust units of the standardised return residual (jump variant)
N_BUCKETS = (0, 1, 2, 5, 20, 10**9)  # trade-count buckets for the direction residual scale: 1, 2, 3-5, 6-20, 21+


def _lagmean(x: np.ndarray, a: int, b: int) -> np.ndarray:
    """mean(x_{t-b..t-a}) for each t (NaN for t < b)."""
    c = np.r_[0.0, np.cumsum(x)]
    out = np.full(len(x), np.nan)
    t = np.arange(b, len(x))
    out[b:] = (c[t - a + 1] - c[t - b]) / (b - a + 1)
    return out


def _lagsum(x: np.ndarray, a: int, b: int) -> np.ndarray:
    return _lagmean(x, a, b) * (b - a + 1)


def _shift(x: np.ndarray, k: int) -> np.ndarray:
    out = np.full(len(x), np.nan)
    out[k:] = x[:-k]
    return out


def _tfit(z: np.ndarray) -> float:
    """Student-t df of standardised residuals (MLE, loc 0), bounded to [2.2, 200]."""
    z = z[np.isfinite(z)]
    if len(z) > 50_000:
        z = np.random.default_rng(0).choice(z, 50_000, replace=False)
    sd = z.std()
    if len(z) < 100 or sd <= 0:
        return 30.0
    df, _, _ = stats.t.fit(z / sd, floc=0)
    return float(min(max(df, 2.2), 200.0))


def _tdraw(rng: np.random.Generator, df: float, n: int, clip: float | None = None) -> np.ndarray:
    """Student-t draws scaled to unit variance, truncated at +-clip (in those units)."""
    z = rng.standard_t(df, n)
    z = z * math.sqrt((df - 2.0) / df) if df > 2 else z
    return np.clip(z, -clip, clip) if clip else z


def _sigm(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-min(max(float(x), -50.0), 50.0)))


def _zmax(e: np.ndarray, sd: float) -> float:
    e = e[np.isfinite(e)]
    return float(np.max(np.abs(e)) / sd) if len(e) and sd > 0 else 10.0


def _lagmean_partial(x: np.ndarray, a: int, b: int, min_b: int) -> np.ndarray:
    """mean(x_{max(0, t-b)..t-a}) for t >= min_b (uses the history available in the segment, up to b candles back)."""
    c = np.r_[0.0, np.cumsum(x)]
    out = np.full(len(x), np.nan)
    t = np.arange(min_b, len(x))
    lo = np.maximum(t - b, 0)
    out[min_b:] = (c[t - a + 1] - c[lo]) / (t - a + 1 - lo)
    return out


def _hawkes_lam(n: np.ndarray, seg: np.ndarray, w: float, a1: float, b1: float, a2: float, b2: float,
                base: np.ndarray | None = None) -> np.ndarray:
    """Conditional mean of the two-kernel discrete Hawkes (baseline w * base_t), restarted per segment at its stationary level."""
    from scipy.signal import lfilter
    lam = np.empty(len(n))
    br = a1 / (1 - b1) + a2 / (1 - b2)
    nbar = w / max(1 - br, 1e-6)
    starts = np.r_[0, np.flatnonzero(np.diff(seg)) + 1, len(n)]
    for s0, s1 in zip(starts[:-1], starts[1:], strict=True):
        x = np.r_[nbar, n[s0:s1 - 1]]  # n_{t-1}, pre-sample at the stationary mean
        out = np.full(s1 - s0, w) if base is None else w * base[s0:s1]
        for a, b in ((a1, b1), (a2, b2)):
            l0 = a * nbar / (1 - b)
            y, _ = lfilter([a], [1, -b], x, zi=[b * l0])
            out += y
        lam[s0:s1] = out
    return lam


def _nb_ll(n: np.ndarray, lam: np.ndarray, alpha: float) -> float:
    from scipy.special import gammaln
    r = 1.0 / alpha
    lam = np.maximum(lam, 1e-9)
    return float(np.sum(gammaln(n + r) - gammaln(r) - gammaln(n + 1) + r * np.log(r / (r + lam)) + n * np.log(lam / (r + lam))))


def slow_regime(n: np.ndarray, seg: np.ndarray, window: int = L_HOUR) -> tuple[np.ndarray, float, float]:
    """Centred 1 h mean of n / overall mean (per segment, window clipped), and the log-AR(1) (rho per candle, sigma)."""
    nbar = max(float(n.mean()), 1e-9)
    M = np.ones(len(n))
    starts = np.r_[0, np.flatnonzero(np.diff(seg)) + 1, len(n)]
    pairs_x, pairs_y = [], []
    h = window // 2
    for s0, s1 in zip(starts[:-1], starts[1:], strict=True):
        x = n[s0:s1]
        c = np.r_[0.0, np.cumsum(x)]
        i = np.arange(len(x))
        lo, hi = np.maximum(i - h, 0), np.minimum(i + h + 1, len(x))
        M[s0:s1] = np.maximum((c[hi] - c[lo]) / (hi - lo) / nbar, 0.05)
        lm = np.log(M[s0:s1])
        if len(lm) > window:
            pairs_x.append(lm[:-window])
            pairs_y.append(lm[window:])
    lm = np.log(M)
    v = float(lm.var())
    if pairs_x:
        px, py = np.concatenate(pairs_x), np.concatenate(pairs_y)
        ac = float(np.corrcoef(px, py)[0, 1]) if px.std() > 0 and py.std() > 0 else 0.5
    else:
        ac = 0.5
    rho = float(np.clip(ac, 0.01, 0.995) ** (1.0 / window))
    return M, rho, float(math.sqrt(max(v * (1 - rho ** 2), 0.0)))


PHI_MAX = 0.5 ** (1 / L_HOUR)
ACF_LAGS = np.array([1, 2, 3, 5, 10, 20, 40, 60, 100, 200, 400, 800, 1200])


def seg_acf(x: np.ndarray, seg: np.ndarray, lags=ACF_LAGS) -> np.ndarray:
    """Autocorrelation at the given lags using only pairs inside the same segment."""
    mu, var = x.mean(), x.var()
    out = np.zeros(len(lags))
    for i, k in enumerate(lags):
        same = seg[k:] == seg[:-k]
        out[i] = float(np.mean((x[k:][same] - mu) * (x[:-k][same] - mu)) / var) if same.sum() > 50 and var > 0 else np.nan
    return out


def fit_copula_activity(n: np.ndarray, seg: np.ndarray) -> tuple[list[float], np.ndarray]:
    """(w1, phi1, w2, phi2) of the latent two-component AR(1) and the empirical quantiles of n."""
    from scipy.optimize import minimize as _min
    a = np.log1p(n)
    rk = stats.rankdata(a) / (len(a) + 1)
    gs = stats.norm.ppf(rk)
    ac = seg_acf(gs, seg)
    ok = np.isfinite(ac)
    lags = ACF_LAGS[ok]
    ac = ac[ok]

    def unpack(z):
        zz = np.clip(z[:3], -50, 50)
        w = np.exp(zz) / np.exp(zz).sum()  # w1, w2, noise share
        # persistence capped at a 1 h half-life: slower (day-level) differences are not simulated within a path
        return w[0], PHI_MAX * _sigm(z[3]), w[1], PHI_MAX * _sigm(z[4])

    def loss(z):
        w1, p1, w2, p2 = unpack(z)
        return float(np.sum((w1 * p1 ** lags + w2 * p2 ** lags - ac) ** 2))

    best = None
    for z0 in ([0.0, 0.0, 0.0, 2.0, 7.0], [1.0, -1.0, 0.0, 0.0, 8.0], [-1.0, 1.0, 0.0, 3.0, 9.0]):
        r = _min(loss, np.array(z0), method="Nelder-Mead", options={"maxiter": 4000, "xatol": 1e-7, "fatol": 1e-10})
        if best is None or r.fun < best.fun:
            best = r
    w1, p1, w2, p2 = unpack(best.x)
    qs = np.quantile(n, np.linspace(0, 1, 2001))
    return [float(w1), float(p1), float(w2), float(p2)], qs


def fit_hawkes(n: np.ndarray, seg: np.ndarray, base: np.ndarray | None = None) -> tuple[list[float], float, float]:
    """NB2 maximum likelihood of (w, a1, b1, a2, b2, alpha); returns params, alpha, branching ratio."""
    from scipy.optimize import minimize as _min

    def unpack(z):
        w = math.exp(min(z[0], 50.0))
        b1 = _sigm(z[1])
        b2 = _sigm(z[2])
        # branching shares: total ratio rho in (0, 1) split between the kernels
        rho = _sigm(z[3])
        sh = _sigm(z[4])
        a1 = rho * sh * (1 - b1)
        a2 = rho * (1 - sh) * (1 - b2)
        return w, a1, b1, a2, b2, math.exp(min(z[5], 20.0))

    def nll(z):
        w, a1, b1, a2, b2, al = unpack(z)
        return -_nb_ll(n, _hawkes_lam(n, seg, w, a1, b1, a2, b2, base), al)

    nbar = max(float(n.mean()), 1e-3)
    best = None
    for z0 in ([math.log(0.2 * nbar), 0.0, 6.0, 1.5, 0.0, 0.0], [math.log(0.05 * nbar), -1.0, 7.0, 2.5, 0.5, -0.5]):
        r = _min(nll, np.array(z0), method="Nelder-Mead", options={"maxiter": 3000, "xatol": 1e-5, "fatol": 1e-4})
        if best is None or r.fun < best.fun:
            best = r
    w, a1, b1, a2, b2, al = unpack(best.x)
    return [w, a1, b1, a2, b2], al, a1 / (1 - b1) + a2 / (1 - b2)


def _ols(y: np.ndarray, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    m = np.isfinite(y) & np.isfinite(X).all(axis=1)
    beta, *_ = np.linalg.lstsq(X[m], y[m], rcond=None)
    res = np.full(len(y), np.nan)
    res[m] = y[m] - X[m] @ beta
    return beta, res


@dataclass
class Series:
    """Per-candle variables on contiguous segments (rows outside segments are dropped; seg = segment id)."""
    seg: np.ndarray
    n: np.ndarray
    a: np.ndarray
    V: np.ndarray
    F: np.ndarray
    q: np.ndarray
    r: np.ndarray
    s: np.ndarray
    o: np.ndarray
    hi: np.ndarray
    lo: np.ndarray
    mid: np.ndarray
    mid_prev: np.ndarray
    depth: np.ndarray


def series_from_grid(g: Grid, mask: np.ndarray | None = None, min_len: int = 400) -> Series:
    """Contiguous ok runs of g (within mask) with returns defined from the second candle of each run."""
    gg = g if mask is None else g.slice(mask)
    parts = []
    for k, (a, b) in enumerate(gg.segments(min_len)):
        sl = slice(a, b)
        mid = gg.mid[sl]
        r = np.r_[np.nan, 1e4 * np.diff(np.log(mid))]
        mid_prev = np.r_[np.nan, mid[:-1]]
        n = gg.n_buy[sl] + gg.n_sell[sl]
        V = gg.buy[sl] + gg.sell[sl]
        F = gg.buy[sl] - gg.sell[sl]
        q = np.where(V > 0, F / np.where(V > 0, V, 1.0), 0.0)
        parts.append(dict(seg=np.full(b - a, k), n=n, a=np.log1p(n), V=V, F=F, q=q, r=r, s=gg.spread_bps[sl],
                          o=np.nan_to_num(gg.obi_25[sl], nan=0.0), hi=gg.high[sl], lo=gg.low[sl], mid=mid, mid_prev=mid_prev,
                          depth=gg.depth[sl]))
    if not parts:
        z = np.zeros(0)
        return Series(*(z for _ in range(14)))
    return Series(**{k: np.concatenate([p[k] for p in parts]) for k in parts[0]})


def _design_lags(S: Series) -> dict[str, np.ndarray]:
    """Lagged regressors computed within segments (NaN in the first L_LONG candles of each segment)."""
    out = {k: np.full(len(S.n), np.nan) for k in ("a1", "a2", "a3", "a4", "absr1", "o1", "q1", "q2", "r1", "rs", "ls1", "rlag", "m20", "m200")}
    for k in np.unique(S.seg):
        idx = np.flatnonzero(S.seg == k)
        if len(idx) <= L_LONG + 1:
            continue
        a, q, r, o, s = S.a[idx], S.q[idx], np.nan_to_num(S.r[idx]), S.o[idx], S.s[idx]
        sub = {"a1": _shift(a, 1), "a2": _lagmean(a, 2, L_SHORT), "a3": _lagmean(a, L_SHORT + 1, L_LONG),
               "a4": _lagmean_partial(a, L_LONG + 1, L_HOUR, L_LONG + 1), "absr1": _shift(np.abs(r), 1),
               "o1": _shift(o, 1), "q1": _shift(q, 1), "q2": _lagmean(q, 2, L_SHORT), "r1": _shift(r, 1), "rs": _lagsum(r, 2, L_SHORT),
               "ls1": _shift(np.log(s), 1), "rlag": _shift(r, 1), "m20": _lagmean(np.abs(r), 1, L_SHORT),
               "m200": _lagmean(np.abs(r), L_SHORT + 1, L_LONG)}
        for key, v in sub.items():
            v = v.copy()
            v[:L_LONG] = np.nan
            out[key][idx] = v
    return out


@dataclass
class FlowSim:
    symbol: str = ""
    sigma: float = 1.0          # bps per 3 s
    depth: float = 1e6
    trade_size: float = 100.0   # median notional per trade
    tick_bps: float = 1.0
    act: list = field(default_factory=list)      # Hawkes w, a1, b1, a2, b2 (trade-count conditional mean)
    branching: float = 0.0
    activity: str = "copula"                     # copula | hawkes
    cop: list = field(default_factory=list)      # w1, phi1, w2, phi2
    n_quant: np.ndarray | None = None            # empirical quantiles of n (2001 points)
    regime_rho: float = 0.0                      # log X_t AR(1) per candle (0 = no slow regime)
    regime_sd: float = 0.0
    nb_alpha: float = 0.5                        # NB2 dispersion: Var = mu + alpha mu^2
    clips: dict = field(default_factory=dict)    # truncation of each innovation (standardised units)
    obi: list = field(default_factory=list)      # c, rho
    obi_sd: float = 0.1
    obi_df: float = 10.0
    dirn: list = field(default_factory=list)     # c, g1, g2, b1, b2, k
    dir_sd: list = field(default_factory=list)   # per trade-count bucket
    dir_df: float = 10.0
    vol: list = field(default_factory=list)      # alpha, eta
    vol_sd: float = 1.0
    vol_df: float = 10.0
    spr: list = field(default_factory=list)      # c, rho
    spr_sd: float = 0.01
    spr_df: float = 10.0
    spr_floor: float = 0.5
    spr_cap: float = 1e4
    ret: list = field(default_factory=list)      # l1, l2, psi
    ret_h: list = field(default_factory=list)    # h0, h1 (activity), h2, h3 (HAR of |r|)
    har_c: float = 0.1                           # offset inside the HAR logs (bps)
    vol_rho: float = 0.0                         # AR(1) of the notional residual across trade candles
    a_max: float = 10.0                          # activity cap (1.25 x the fitted maximum of log(1 + n))
    ret_zsd: float = 1.0
    ret_df: float = 5.0
    pi0: list = field(default_factory=list)      # P(r = 0 | n = 0), P(r = 0 | n > 0)
    # jump variant (exogenous news channel, compound Poisson on nonzero-return candles; Huang-Lehalle-Rosenbaum 2015)
    jump_lam: float = 0.0                        # P(jump | nonzero return)
    jump_pool: np.ndarray | None = None          # jump sizes in sigma units (empirical)
    ret_zsd_nj: float = 1.0                      # diffusive residual scale and df without the jump candles
    ret_df_nj: float = 5.0
    jumps: bool = False                          # generate with the jump channel
    ret_scale: float = 1.0                       # multiplies impact + noise (set by rescale)
    exc: dict = field(default_factory=dict)      # q bucket -> (k, 2) array of (e_hi, e_lo) in sigma units
    fit_stats: dict = field(default_factory=dict)

    # ------------------------------------------------------------------------------------------ fit
    @classmethod
    def fit(cls, g: Grid, mask: np.ndarray | None = None, tick: float | None = None, hawkes: bool = False) -> FlowSim:
        S = series_from_grid(g, mask)
        if len(S.n) < 2 * L_LONG:
            raise ValueError(f"{g.symbol}: not enough contiguous data to fit ({len(S.n)} candles)")
        L = _design_lags(S)
        m = cls(symbol=g.symbol)
        r = S.r
        m.sigma = float(np.nanstd(r))
        m.depth = float(np.nanmedian(S.depth[S.depth > 0])) if np.any(S.depth > 0) else 1e6
        tr = S.n > 0
        m.trade_size = float(np.median(S.V[tr] / S.n[tr])) if tr.any() else 100.0
        mid0 = float(np.nanmedian(S.mid))
        m.tick_bps = float(tick / mid0 * 1e4) if tick else float(np.nanpercentile(S.s, 1))
        sg = max(m.sigma, 1e-9)
        one = np.ones(len(S.n))
        # activity: two-kernel discrete Hawkes with negative-binomial counts (maximum likelihood)
        m.cop, m.n_quant = fit_copula_activity(S.n.astype(float), S.seg)
        if hawkes:
            Mreg, m.regime_rho, m.regime_sd = slow_regime(S.n.astype(float), S.seg)
            m.act, m.nb_alpha, m.branching = fit_hawkes(S.n.astype(float), S.seg, Mreg)
        else:
            m.act = [float(S.n.mean()), 0.0, 0.5, 0.0, 0.5]
        m.a_max = float(1.25 * np.nanmax(S.n) + 1.0)   # cap on the conditional mean
        # book imbalance
        b, e = _ols(S.o, np.column_stack([one, L["o1"]]))
        m.obi, m.obi_sd = b.tolist(), float(np.nanstd(e))
        m.obi_df = _tfit(e / max(m.obi_sd, 1e-12))
        m.clips["obi"] = _zmax(e, m.obi_sd)
        # direction
        X = np.column_stack([one, L["q1"], L["q2"], L["r1"] / sg, L["rs"] / sg, L["o1"]])
        y = np.where(tr, S.q, np.nan)
        b, e = _ols(y, X)
        m.dirn = b.tolist()
        sds = []
        for lo_, hi_ in zip(N_BUCKETS[:-1], N_BUCKETS[1:], strict=True):
            sel = (S.n > lo_) & (S.n <= hi_) & np.isfinite(e)
            sds.append(float(np.std(e[sel])) if sel.sum() > 30 else float(np.nanstd(e)))
        m.dir_sd = sds
        bk = np.digitize(S.n, N_BUCKETS[1:-1], right=True)
        scale = np.array(sds)[np.clip(bk, 0, len(sds) - 1)]
        m.dir_df = _tfit(e / scale)
        m.clips["dir"] = _zmax(e / scale, 1.0)
        # notional
        sel = tr & (S.V > 0)
        b, e = _ols(np.where(sel, np.log(np.where(S.V > 0, S.V, 1.0)), np.nan), np.column_stack([one, np.log(np.maximum(S.n, 1))]))
        m.vol = b.tolist()
        ev = e[np.isfinite(e)]
        m.vol_rho = float(np.clip(np.corrcoef(ev[1:], ev[:-1])[0, 1], 0.0, 0.95)) if len(ev) > 100 else 0.0
        innov = ev[1:] - m.vol_rho * ev[:-1] if len(ev) > 1 else ev
        m.vol_sd = float(np.std(innov)) if len(innov) else 1.0
        m.vol_df = _tfit(innov / max(m.vol_sd, 1e-12))
        m.clips["vol"] = _zmax(innov, m.vol_sd)
        # spread
        b, e = _ols(np.log(S.s), np.column_stack([one, L["ls1"]]))
        m.spr, m.spr_sd = b.tolist(), float(max(np.nanstd(e), 1e-6))
        m.spr_df = _tfit(e / m.spr_sd)
        m.clips["spr"] = _zmax(e, m.spr_sd)
        m.spr_floor = float(np.nanpercentile(S.s, 1))
        m.spr_cap = float(1.5 * np.nanmax(S.s))
        # returns: impact on nonzero-return candles, zero mass by trade presence
        okr = np.isfinite(r) & np.isfinite(L["rlag"])
        nz = okr & (r != 0)
        f = S.F / m.depth
        X = np.column_stack([f, np.sign(f) * np.sqrt(np.abs(f)), L["rlag"]])
        b, _ = _ols(np.where(nz, r, np.nan), X)
        psi = float(b[2])
        sel = nz & np.isfinite(X).all(axis=1)
        lam, _ = nnls(X[sel, :2], r[sel] - psi * L["rlag"][sel])
        m.ret = [float(lam[0]), float(lam[1]), psi]
        e = np.full(len(r), np.nan)
        e[sel] = r[sel] - X[sel, :2] @ lam - psi * L["rlag"][sel]
        m.har_c = float(max(0.05 * m.sigma, 1e-3))
        le = np.log(np.where(np.isfinite(e) & (e != 0), e * e, np.nan))
        H = np.column_stack([one, S.a, np.log(m.har_c + L["m20"]), np.log(m.har_c + L["m200"])])
        hb, _ = _ols(le, H)
        m.ret_h = hb.tolist()
        z = e / np.exp(0.5 * (H @ hb))
        m.ret_zsd = float(np.nanstd(z))
        m.ret_df = _tfit(z / m.ret_zsd)
        m.clips["ret"] = _zmax(z, m.ret_zsd)
        # jump variant: |standardised residual| > JUMP_Z robust units = jump; the rest is the diffusive part
        zz = z[np.isfinite(z)]
        rob = 1.4826 * float(np.median(np.abs(zz - np.median(zz)))) if len(zz) else 1.0
        jm = np.isfinite(z) & (np.abs(z) > JUMP_Z * max(rob, 1e-12))
        m.jump_lam = float(jm.sum() / max(np.isfinite(z).sum(), 1))
        m.jump_pool = (e[jm] / sg) if jm.any() else np.zeros(0)
        znj = z[np.isfinite(z) & ~jm]
        m.ret_zsd_nj = float(np.std(znj)) if len(znj) else m.ret_zsd
        m.ret_df_nj = _tfit(znj / max(m.ret_zsd_nj, 1e-12))
        m.clips["ret_nj"] = _zmax(znj, m.ret_zsd_nj)
        m.pi0 = [float(np.mean(r[okr & ~tr] == 0)) if (okr & ~tr).any() else 1.0, float(np.mean(r[okr & tr] == 0)) if (okr & tr).any() else 0.0]
        # high / low excursions beyond the half-spread, in sigma units, by direction bucket
        hh = okr & tr & np.isfinite(S.hi) & np.isfinite(S.lo) & np.isfinite(S.mid_prev)
        top = np.maximum(S.mid_prev, S.mid)
        bot = np.minimum(S.mid_prev, S.mid)
        e_hi = (1e4 * np.log(S.hi / top) - S.s / 2) / sg
        e_lo = (1e4 * np.log(bot / S.lo) - S.s / 2) / sg
        qb = np.digitize(S.q, [-0.5, 0.5])
        rng = np.random.default_rng(0)
        for k in range(3):
            sel = np.flatnonzero(hh & (qb == k))
            if len(sel) > 20000:
                sel = rng.choice(sel, 20000, replace=False)
            m.exc[k] = np.column_stack([e_hi[sel], e_lo[sel]]) if len(sel) else np.zeros((1, 2))
        m.fit_stats = {"candles": int(len(S.n)), "segments": int(len(np.unique(S.seg))), "frac_trade": float(tr.mean()),
                       "mean_trades": float(S.n.mean())}
        return m

    # ------------------------------------------------------------------------------------- generate
    def generate(self, n: int, rng: np.random.Generator, *, burn: int = 400, mid0: float = 100.0, t0_ms: int = 0) -> Grid:
        """n candles (after `burn` discarded ones) as a Grid (all candles ok)."""
        N = n + burn
        hw, ha1, hb1, ha2, hb2 = self.act
        c_o, rho_o = self.obi
        c_q, g1, g2, b1, b2, kq = self.dirn
        al, eta = self.vol
        c_s, rho_s = self.spr
        l1, l2, psi = self.ret
        h0, h1, h2, h3 = self.ret_h
        hc = self.har_c
        a_max = self.a_max
        vrho, vinn = self.vol_rho, math.sqrt(max(1.0 - self.vol_rho ** 2, 0.0))
        sg = max(self.sigma, 1e-9)
        k_ret = self.ret_scale
        cl = self.clips
        alpha = max(self.nb_alpha, 1e-4)
        g_nb = rng.standard_gamma(1.0 / alpha, N) * alpha   # mean-one gamma mixing (NB2)
        # slow exogenous regime X_t (mean one)
        rr, rs_ = self.regime_rho, self.regime_sd
        if rr > 0 and rs_ > 0:
            from scipy.signal import lfilter
            vx = rs_ ** 2 / max(1 - rr ** 2, 1e-12)
            e_x = rng.standard_normal(N) * rs_
            lx, _ = lfilter([1.0], [1.0, -rr], e_x, zi=[rr * rng.standard_normal() * math.sqrt(vx)])
            X = np.exp(lx - 0.5 * vx)
        else:
            X = np.ones(N)
        u_o = _tdraw(rng, self.obi_df, N, cl.get("obi")) * self.obi_sd
        u_q = _tdraw(rng, self.dir_df, N, cl.get("dir"))
        u_v = _tdraw(rng, self.vol_df, N, cl.get("vol")) * self.vol_sd
        u_s = _tdraw(rng, self.spr_df, N, cl.get("spr")) * self.spr_sd
        jumps = self.jumps and self.jump_lam > 0 and self.jump_pool is not None and len(self.jump_pool) > 0
        if jumps:
            u_r = _tdraw(rng, self.ret_df_nj, N, cl.get("ret_nj")) * self.ret_zsd_nj
            pool = np.asarray(self.jump_pool)
            jmp = np.where(rng.random(N) < self.jump_lam, pool[rng.integers(0, len(pool), N)] * sg, 0.0)
        else:
            u_r = _tdraw(rng, self.ret_df, N, cl.get("ret")) * self.ret_zsd
            jmp = np.zeros(N)
        u0 = rng.random(N)
        dir_sd = self.dir_sd
        pi_n0, pi_n1 = self.pi0
        a_mean = float(np.log1p(self.fit_stats.get("mean_trades", 1.0)))
        # histories with L_HOUR pre-sample values: x_k is stored at index L_HOUR + k
        H0 = L_HOUR
        ah = [a_mean] * H0 + [0.0] * N
        qh = [0.0] * H0 + [0.0] * N
        rh = [0.0] * H0 + [0.0] * N
        s2a, s3a = a_mean * (L_SHORT - 1), a_mean * (L_LONG - L_SHORT)   # sums of a_{t-2..t-20}, a_{t-21..t-200}
        s4a = a_mean * (L_HOUR - L_LONG)                                   # sum of a_{t-201..t-1200}
        n4 = L_HOUR - L_LONG
        log_cap = math.log(max(self.spr_cap, 1e-9))
        s2q = s2r = 0.0                                                    # sums of q, r over t-2..t-20
        absr0 = sg * math.sqrt(2 / math.pi) * 0.5
        bh = [absr0] * H0 + [0.0] * N                                  # |r| history
        m20, m200 = absr0 * L_SHORT, absr0 * (L_LONG - L_SHORT)            # sums of |r| over t-1..t-20, t-21..t-200
        ev = 0.0
        out_n, out_q, out_V, out_r, out_s, out_o = (np.zeros(N) for _ in range(6))
        o1 = c_o / (1 - rho_o) if abs(1 - rho_o) > 1e-3 else 0.0
        ls = c_s / (1 - rho_s) if abs(1 - rho_s) > 1e-3 else math.log(max(self.spr_floor, 1e-6))
        log_floor = math.log(max(self.spr_floor, 1e-9))
        a1, q1, r1 = a_mean, 0.0, 0.0
        nbar = hw / max(1 - self.branching, 1e-6)
        lk1, lk2 = ha1 * nbar / (1 - hb1), ha2 * nbar / (1 - hb2)
        n_prev = nbar
        if self.activity == "copula" and self.cop and self.n_quant is not None:
            from scipy.signal import lfilter
            w1, f1, w2, f2 = self.cop
            gl = np.sqrt(max(1 - w1 - w2, 0.0)) * rng.standard_normal(N)
            for w, f in ((w1, f1), (w2, f2)):
                e = rng.standard_normal(N) * math.sqrt(1 - f * f)
                z, _ = lfilter([1.0], [1.0, -f], e, zi=[f * rng.standard_normal()])
                gl += math.sqrt(w) * z
            qn = np.asarray(self.n_quant)
            u = stats.norm.cdf(gl)
            n_arr = np.round(np.interp(u, np.linspace(0, 1, len(qn)), qn)).astype(int).tolist()
        else:
            n_arr = None
        n2, n3 = L_SHORT - 1, L_LONG - L_SHORT
        depth = self.depth
        for t in range(N):
            if n_arr is not None:
                nt = n_arr[t]
            else:
                lk1 = hb1 * lk1 + ha1 * n_prev
                lk2 = hb2 * lk2 + ha2 * n_prev
                nt = int(rng.poisson(min(a_max, hw * X[t] + lk1 + lk2) * g_nb[t]))
                n_prev = nt
            a_obs = math.log1p(nt)
            o = min(1.0, max(-1.0, c_o + rho_o * o1 + u_o[t]))
            if nt > 0:
                bk = 0 if nt <= 1 else 1 if nt <= 2 else 2 if nt <= 5 else 3 if nt <= 20 else 4
                q = c_q + g1 * q1 + g2 * s2q / n2 + b1 * r1 / sg + b2 * s2r / sg + kq * o1 + dir_sd[bk] * u_q[t]
                q = 1.0 if q > 1.0 else -1.0 if q < -1.0 else q
                ev = vrho * ev + (vinn if vrho else 1.0) * u_v[t]
                V = math.exp(al + eta * math.log(nt) + ev)
            else:
                q = V = 0.0
            F = q * V
            ls = min(log_cap, max(log_floor, c_s + rho_s * ls + u_s[t]))
            if u0[t] < (pi_n1 if nt > 0 else pi_n0):
                r = 0.0
            else:
                f = F / depth
                imp = l1 * f + (l2 * math.copysign(math.sqrt(abs(f)), f) if f else 0.0)
                hh = h0 + h1 * a_obs + h2 * math.log(hc + m20 / L_SHORT / k_ret) + h3 * math.log(hc + m200 / (L_LONG - L_SHORT) / k_ret)
                r = k_ret * (imp + math.exp(0.5 * min(hh, 60.0)) * u_r[t] + jmp[t]) + psi * r1
            out_n[t], out_q[t], out_V[t], out_r[t], out_s[t], out_o[t] = nt, q, V, r, math.exp(ls), o
            j = H0 + t
            ah[j], qh[j], rh[j] = a_obs, q, r
            # windows for step t+1: add x_{t-1}, drop x_{t-20} (short); add a_{t-20}, drop a_{t-200} (long)
            s2a += ah[j - 1] - ah[j - L_SHORT]
            s3a += ah[j - L_SHORT] - ah[j - L_LONG]
            s4a += ah[j - L_LONG] - ah[j - L_HOUR]
            s2q += qh[j - 1] - qh[j - L_SHORT]
            s2r += rh[j - 1] - rh[j - L_SHORT]
            bh[j] = abs(r)
            m20 += bh[j] - bh[j - L_SHORT]           # |r_t| enters 1..20 for step t+1, |r_{t-20}| moves to 21..200
            m200 += bh[j - L_SHORT] - bh[j - L_LONG]
            a1, q1, r1, o1 = a_obs, q, r, o
        sl = slice(burn, N)
        r = out_r[sl]
        s = out_s[sl]
        mid = mid0 * np.exp(np.cumsum(r) / 1e4)
        mid_prev = np.r_[mid0, mid[:-1]]
        nn, qq, VV = out_n[sl], out_q[sl], out_V[sl]
        # high / low from excursions beyond the half-spread
        qb = np.digitize(qq, [-0.5, 0.5])
        e_hi = np.zeros(n)
        e_lo = np.zeros(n)
        for k in range(3):
            idx = np.flatnonzero(qb == k)
            pool = self.exc.get(k, self.exc.get(str(k)))
            pool = np.asarray(pool) if pool is not None else np.zeros((1, 2))
            if len(idx) and len(pool):
                pick = rng.integers(0, len(pool), len(idx))
                e_hi[idx], e_lo[idx] = pool[pick, 0], pool[pick, 1]
        trade = nn > 0
        top, bot = np.maximum(mid_prev, mid), np.minimum(mid_prev, mid)
        hi = np.where(trade, top * np.exp((s / 2 + e_hi * self.sigma) / 1e4), np.nan)
        lo = np.where(trade, bot * np.exp(-(s / 2 + e_lo * self.sigma) / 1e4), np.nan)
        hi, lo = np.maximum(hi, lo), np.minimum(hi, lo)
        buy, sell = VV * (1 + qq) / 2, VV * (1 - qq) / 2
        nb = np.round(nn * (1 + qq) / 2)
        o = out_o[sl]
        return Grid(self.symbol, t0_ms + np.arange(n, dtype=np.int64) * STEP_MS, np.ones(n, bool), np.ones(n, bool), mid, s, hi, lo,
                    buy, sell, nb, nn - nb, o, o, np.full(n, self.depth))

    # -------------------------------------------------------------------------------------- rescale
    def rescale(self, target_sigma: float, target_depth: float, target_trade_size: float, *, target_spread_bps: float | None = None,
                pilot: int = 20_000, seed: int = 7, iters: int = 3) -> FlowSim:
        """Copy calibrated to another market's 3 s sigma (bps), depth within 25 bps and median trade notional."""
        m = copy.deepcopy(self)
        m.vol = [self.vol[0] + math.log(target_trade_size / self.trade_size), self.vol[1]]
        m.trade_size = target_trade_size
        m.depth = target_depth
        k_s = (target_spread_bps / self.median_spread()) if target_spread_bps else target_sigma / self.sigma
        m.spr = [self.spr[0] + (1 - self.spr[1]) * math.log(k_s), self.spr[1]]
        m.spr_floor = self.spr_floor * k_s
        m.spr_cap = self.spr_cap * k_s
        m.tick_bps = self.tick_bps * k_s
        m.sigma = target_sigma  # excursions are in sigma units
        m.ret_scale = self.ret_scale * target_sigma / self.sigma
        for it in range(iters):
            g = m.generate(pilot, np.random.default_rng(seed + it), burn=400)
            got = float(np.std(np.diff(np.log(g.mid))) * 1e4)
            if got > 0:
                m.ret_scale *= target_sigma / got
        return m

    def median_spread(self) -> float:
        c, rho = self.spr
        return float(math.exp(c / (1 - rho))) if abs(1 - rho) > 1e-6 else float(self.spr_floor)

    def to_json(self) -> dict:
        d = asdict(self)
        d["n_quant"] = None if self.n_quant is None else np.asarray(self.n_quant)[::100].tolist()
        d["jump_pool"] = {"n": int(len(self.jump_pool)) if self.jump_pool is not None else 0}
        d["exc"] = {str(k): {"n": int(len(v)), "mean": np.asarray(v).mean(axis=0).tolist()} for k, v in self.exc.items()}
        return d


# ------------------------------------------------------------------------------------ stylised facts

FACTS = ("kurtosis", "absr_acf", "flow_acf", "impact", "spread")
ABS_LAGS, FLOW_LAGS = 60, 20
SPREAD_EDGES = np.exp(np.linspace(np.log(0.01), np.log(2000.0), 801))


@dataclass
class BlockStats:
    """Additive sufficient statistics of one block (1 h of candles) for the five facts."""
    mom: np.ndarray          # n, sum r, r^2, r^3, r^4
    acf_abs: np.ndarray      # _acf_stats of |r|
    acf_flow: np.ndarray     # _acf_stats of F
    xy: np.ndarray           # n, sum x, sum y, sum x^2, sum y^2, sum xy   (x = F, y = r)
    hist: np.ndarray         # spread histogram
    cont: np.ndarray         # same-sign pairs of consecutive nonzero mid changes, all pairs (continuation ratio)


def _acf_stats(x: np.ndarray, seg: np.ndarray, lags: int) -> np.ndarray:
    """[n, sum x, sum x^2, S_1..S_L, N_1..N_L] with S_k = sum x_t x_{t+k} over pairs in the same segment."""
    out = np.zeros(3 + 2 * lags)
    ok = np.isfinite(x)
    xx = np.where(ok, x, 0.0)
    out[0], out[1], out[2] = ok.sum(), xx.sum(), (xx * xx).sum()
    for k in range(1, lags + 1):
        same = (seg[k:] == seg[:-k]) & ok[k:] & ok[:-k]
        out[2 + k] = (xx[k:] * xx[:-k] * same).sum()
        out[2 + lags + k] = same.sum()
    return out


def block_stats(r: np.ndarray, F: np.ndarray, s: np.ndarray, seg: np.ndarray, block: int = 1200) -> list[BlockStats]:
    out = []
    for a in range(0, len(r), block):
        sl = slice(a, a + block)
        rr, ff, ss, sg = r[sl], F[sl], s[sl], seg[sl]
        ok = np.isfinite(rr)
        x = rr[ok]
        mom = np.array([ok.sum(), x.sum(), (x ** 2).sum(), (x ** 3).sum(), (x ** 4).sum()])
        okf = ok & np.isfinite(ff)
        xf, yr = ff[okf], rr[okf]
        xy = np.array([okf.sum(), xf.sum(), yr.sum(), (xf * xf).sum(), (yr * yr).sum(), (xf * yr).sum()])
        h, _ = np.histogram(ss[np.isfinite(ss)], SPREAD_EDGES)
        nzi = np.flatnonzero(ok & (np.nan_to_num(rr) != 0))
        same_seg = sg[nzi[1:]] == sg[nzi[:-1]]
        sgn = np.sign(rr[nzi])
        cont = np.array([float(((sgn[1:] == sgn[:-1]) & same_seg).sum()), float(same_seg.sum())])
        out.append(BlockStats(mom, _acf_stats(np.abs(rr), sg, ABS_LAGS), _acf_stats(ff, sg, FLOW_LAGS), xy, h.astype(float), cont))
    return out


def _acf_mean(a: np.ndarray, lags: int) -> float:
    n, s1, s2 = a[0], a[1], a[2]
    if n < 10:
        return float("nan")
    mu = s1 / n
    var = s2 / n - mu * mu
    if var <= 0:
        return float("nan")
    S, N = a[3:3 + lags], a[3 + lags:3 + 2 * lags]
    acf = (S / np.maximum(N, 1) - mu * mu) / var
    return float(np.mean(acf[N > 0])) if (N > 0).any() else float("nan")


def _quantile_hist(h: np.ndarray, q: float) -> float:
    c = np.cumsum(h)
    if c[-1] <= 0:
        return float("nan")
    k = int(np.searchsorted(c, q * c[-1]))
    return float(math.sqrt(SPREAD_EDGES[k] * SPREAD_EDGES[min(k + 1, len(SPREAD_EDGES) - 1)]))


def facts_from_blocks(bs: list[BlockStats], idx: np.ndarray | None = None) -> dict[str, float]:
    sel = bs if idx is None else [bs[i] for i in idx]
    if not sel:
        return {k: float("nan") for k in ("kurtosis", "absr_acf", "flow_acf", "impact", "spread_med", "spread_p90", "continuation")}
    mom = sum(b.mom for b in sel)
    n = mom[0]
    if n > 10:
        m1 = mom[1] / n
        m2, m3, m4 = mom[2] / n, mom[3] / n, mom[4] / n
        var = m2 - m1 ** 2
        c4 = m4 - 4 * m1 * m3 + 6 * m1 ** 2 * m2 - 3 * m1 ** 4
        kurt = c4 / var ** 2 - 3 if var > 0 else float("nan")
    else:
        kurt = float("nan")
    xy = sum(b.xy for b in sel)
    nn = xy[0]
    if nn > 10:
        vx = xy[3] / nn - (xy[1] / nn) ** 2
        vy = xy[4] / nn - (xy[2] / nn) ** 2
        cxy = xy[5] / nn - xy[1] * xy[2] / nn ** 2
        imp = cxy / math.sqrt(vx * vy) if vx > 0 and vy > 0 else float("nan")
    else:
        imp = float("nan")
    h = sum(b.hist for b in sel)
    return {"kurtosis": float(kurt), "absr_acf": _acf_mean(sum(b.acf_abs for b in sel), ABS_LAGS),
            "flow_acf": _acf_mean(sum(b.acf_flow for b in sel), FLOW_LAGS), "impact": float(imp),
            "spread_med": _quantile_hist(h, 0.5), "spread_p90": _quantile_hist(h, 0.9),
            "continuation": float(sum(b.cont[0] for b in sel) / max(sum(b.cont[1] for b in sel), 1.0))}


def facts_of_grid(g: Grid, mask: np.ndarray | None = None, min_len: int = 100) -> tuple[dict[str, float], list[BlockStats]]:
    S = series_from_grid(g, mask, min_len)
    bs = block_stats(S.r, S.F / max(np.nanmedian(S.depth) if len(S.depth) else 1.0, 1e-9), S.s, S.seg)
    return facts_from_blocks(bs), bs
