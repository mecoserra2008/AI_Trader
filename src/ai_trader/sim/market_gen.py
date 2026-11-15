"""Parametric, parameter-uncertain multi-asset market generator at 1h resolution (block GEN: H-GEN1, H-GEN2, H-GEN3).

Use: generated paths are for DETERMINISTIC strategies only (CLAUDE.md principle 1). They never train, prompt or
backtest the LLM decision layer. The generator is calibrated on llm_in_sample rows only (known_ts < the knowledge
cutoff; `load_calibration` enforces it and `CalibData.known_max` records it) and must cover the clean_holdout.

Model (one universe U with symbols i = 1..N; hours t, UTC days d; how = hour of week, Monday 00 UTC = 0)
------------------------------------------------------------------------------------------------------
Daily continuous variance. y_{i,d} = log BV_{i,d}, the jump-robust bipower variation of the 24 hourly returns of day d
(pi/2 * sum |r_k||r_{k-1}| * 24/23). Decomposition
    y_{i,d} = a_i + w_{weekday(d)} + b_i F_d + e_{i,d}
  (a_i, w are fitted on the whole calibration window; the simulated long-run level is the trailing-365-day mean, a_i^365)
* F_d (common crypto vol factor) = cross-sectional mean of x = y - a - w; log-HAR with Student-t shocks:
    F_d = c + b1 F_{d-1} + b5 mean(F_{d-5..d-1}) + b22 mean(F_{d-22..d-1}) + s_F t_{nu_F}
* e_{i,d} (idiosyncratic) pooled log-HAR plus a v0-regime shift gamma_S of the start-of-day cell S_{i,d}:
    e_{i,d} = h1 e_{d-1} + h5 mean5(e) + h22 mean22(e) + gamma_{S_{i,d}} + s_i t_{nu_e}
* S_{i,d}: v0 cells (research.regimes.label at 00:00 UTC, exactly the strategy panel's `cell`) follow the pooled daily
  9x9 transition matrix P (latent chains per symbol in simulation; the v0 labels of a generated path are recomputed
  from its prices).
Hourly returns. Within day d the diffusive part is
    a_t = sqrt(q_how) * exp(h_{i,t}) * eps_{i,t},   eps_{i,t} = lam_i m_t + sqrt(1 - lam_i^2) u_{i,t}
  q = hour-of-week variance shares (sum 1 within each weekday), h = sqrt(psi) h^c_t + sqrt(1-psi) h^i_{i,t} hourly
  AR(1) log-vol deviations (common + idiosyncratic, stationary variance sig_h^2, mean -sig_h^2 so E e^{2h} = 1),
  m and u standardised Student-t(nu_eps): lam_i^2 is the hourly return correlation with the common factor. The common
  shock m (and common jumps) is per correlation cluster (average linkage on the in-sample hourly correlation, cut at
  0.3): one cluster for crypto; T splits metals / energy / equities as the data say.
  The day's diffusive returns are scaled so that their bipower variation equals BV_{i,d} exactly (so the simulated
  measured log BV follows the fitted HAR, measurement noise included). Then, hour by hour,
    r_t = rc_t + phi^h_S rc_{t-1} s_t/s_{t-1} + phi^d_S u_{d-1} sqrt(BV_d)/24 + J_t sqrt(q_t BV_d)
  phi^h_S / phi^d_S = regime-dependent hourly / daily autocorrelation (empirical-Bayes shrunk to the pooled value),
  u_{d-1} = previous day's return / sqrt(BV_{d-1}); drift is zero apart from that (BELIEFS P8). J = Kou jump
  (z units: up w.p. p_up, sizes Exp(eta_up) / Exp(eta_down)) with a Hawkes intensity lambda_t = mu + alpha A_t
  (A decays by exp(-beta) per hour), mu = p (1 - alpha/beta) so the stationary rate is the Kou p.
OHLC. open = previous close (24/7 perps), close from the cumulated returns, high / low = Brownian-bridge extremes of
  the hour with variance kappa_w s_t^2 (kappa_w = in-sample Parkinson / RV ratio of non-jump hours):
  m_up = (r + sqrt(r^2 - 2 kappa_w s^2 log U1)) / 2 >= max(0, r) and m_dn = (r - sqrt(r^2 - 2 kappa_w s^2 log U2)) / 2,
  so high >= max(open, close) and low <= min(open, close) by construction.
Auxiliary series (the strategies use them): log turnover = a^tau_i + tau_how + b_tau x_{i,d} + c_tau (|z_t| - 0.8) + AR(1)
  noise; funding prints every 8h: standardised f_k = phi_f f_{k-1} + beta_f R24_k + s_f t (R24 = trailing 24h return in
  its own sd units); open interest (universe L) prints every 4h: standardised dlogOI_k = phi_o dlogOI_{k-1} + beta_o R4_k
  + s_o t. Values are step functions between prints, as in research.frame (as-of joins).

Parameter uncertainty ("more stochastic")
-----------------------------------------
`estimate(cd, w)` is a weighted estimator: with w = day weights from a circular moving-block bootstrap of the
calibration dates (block 20 days, jointly across symbols, `block_weights`) it returns one bootstrap replicate of every
parameter above (HAR coefficients, factor loadings, weekday and hour-of-week profiles, regime effects, transition matrix,
Student-t dfs, hourly log-vol / correlation statistics, turnover / funding / OI dynamics). The Kou parameters use their
asymptotic covariance (numerical Hessian of the pooled likelihood, x2 design effect for within-day dependence) and the
Hawkes branching ratio / decay per symbol their asymptotic covariance; both are drawn per path. Every path draws a
fresh replicate. The fixed-parameter generator uses the point estimate (w = 1) on every path.

Hourly statistics -> parameters (simulated moments). The hourly log-vol parameters (sig_h, phi_h, psi), the Student-t
kurtosis and lam are not observed directly: the data are standardised by the same day's BV, which damps them. `smm`
finds the parameters whose simulated statistics (same standardisation, same jump classification) match the
in-sample statistics at the point estimate (fixed-point iterations, common random numbers); bootstrap replicates are
mapped by shifting / scaling the point parameters by the replicate's statistic deviation (`hourly_params`).

All simulation arrays are (paths, symbols, time).
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta

import numpy as np

CELLS = tuple(f"R{i}" for i in range(1, 10))
N_CELLS = 9
HOW = 168
BV_SCALE = math.pi / 2 * 24 / 23
G_OFF = 0.01          # offset in g = log(z^2 + G_OFF), the hourly log-vol statistic
BLOCK_DAYS = 20       # moving-block length of the calibration bootstrap
KAPPA_Q = 20.0        # shrinkage (symbol-days) of the pooled hour-of-week shares toward the hour-of-day shares
LEVEL_DAYS = 365      # window of the long-run vol level (see estimate)
TAIL_U = 6.0          # threshold (standardised units) of the GPD tail-jump component
DRIFT_WINDOW = 132    # days per in-sample window for the seasonal-profile drift dispersion
COMMON_SIZE_DISP = 0.3  # lognormal dispersion across symbols of a common jump's size
CLUSTER_CORR = 0.3    # symbols join one common-shock cluster when their hourly correlation (average linkage) exceeds this
KAPPA_SYM = 4.0       # shrinkage (weeks) of each symbol's weekday / hour-of-week profiles toward the universe profile
LAGS_G = (1, 2, 3, 4, 5, 6)


# ====================================================================== calibration data
@dataclass
class CalibData:
    """In-sample data of one universe on a complete hourly grid (NaN = missing)."""

    universe: str
    symbols: list[str]
    day0: datetime                 # UTC midnight of grid day 0
    O: np.ndarray                  # (H, N)
    Hi: np.ndarray
    Lo: np.ndarray
    C: np.ndarray
    TV: np.ndarray                 # turnover
    FU: np.ndarray                 # funding_rate (as-of step function)
    OI: np.ndarray | None          # open_interest (as-of step function), None if not used
    cell: np.ndarray               # (D, N) int index 0..8 of the start-of-day v0 cell, -1 unknown
    known_max: datetime            # max known_ts of any row used
    cutoff: datetime

    @property
    def n_days(self) -> int:
        return self.cell.shape[0]

    @property
    def n_sym(self) -> int:
        return len(self.symbols)


def _grid_from_frames(frames: dict[str, object], cells: dict[str, object], universe: str, cutoff: datetime, use_oi: bool) -> CalibData:
    import polars as pl

    syms = [s for s in frames if frames[s] is not None and frames[s].height > 0]
    t_min = min(frames[s]["event_ts"].min() for s in syms)
    day0 = datetime(t_min.year, t_min.month, t_min.day, tzinfo=UTC)
    D = (cutoff.date() - day0.date()).days
    H = 24 * D
    N = len(syms)
    arr = {k: np.full((H, N), np.nan) for k in ("O", "Hi", "Lo", "C", "TV", "FU", "OI")}
    cell = np.full((D, N), -1, dtype=np.int64)
    kmax = None
    for j, s in enumerate(syms):
        f = frames[s]
        idx = ((f["event_ts"].dt.epoch("s").to_numpy() - int(day0.timestamp())) // 3600).astype(np.int64)
        ok = (idx >= 0) & (idx < H)
        for k, col in (("O", "open"), ("Hi", "high"), ("Lo", "low"), ("C", "close"), ("TV", "turnover"), ("FU", "funding_rate"),
                       ("OI", "open_interest")):
            if col in f.columns:
                arr[k][idx[ok], j] = f[col].cast(pl.Float64).to_numpy()[ok]
        km = f["known_ts"].max()
        kmax = km if kmax is None or km > kmax else kmax
        cs = cells.get(s)
        if cs is not None and cs.height:
            di = ((cs["date"].cast(pl.Datetime("us", "UTC")).dt.epoch("s").to_numpy() - int(day0.timestamp())) // 86400).astype(np.int64)
            ci = np.array([CELLS.index(c) if c in CELLS else -1 for c in cs["cell"].to_list()])
            okc = (di >= 0) & (di < D)
            cell[di[okc], j] = ci[okc]
    return CalibData(universe=universe, symbols=syms, day0=day0, O=arr["O"], Hi=arr["Hi"], Lo=arr["Lo"], C=arr["C"], TV=arr["TV"],
                     FU=arr["FU"], OI=arr["OI"] if use_oi else None, cell=cell, known_max=kmax, cutoff=cutoff)


def load_calibration(universe: str, symbols: Sequence[str], cutoff: datetime,
                     loader: Callable[[str], object] | None = None) -> CalibData:
    """1h frames of `symbols` known strictly before `cutoff` (llm_in_sample), on one hourly grid, plus start-of-day v0 cells.

    loader(sym) -> polars frame with research.frame.market_frame columns; default = market_frame(sym, '1h', as_of=cutoff,
    funding=True, oi=universe == 'L'). Rows with known_ts >= cutoff are dropped here whatever the loader returns."""
    import polars as pl

    from ai_trader.research.regimes import label

    use_oi = universe == "L"
    if loader is None:
        from ai_trader.research.frame import market_frame

        def loader(sym: str):
            return market_frame(sym, "1h", cutoff, funding=True, oi=use_oi)

    frames, cells = {}, {}
    for s in symbols:
        f = loader(s)
        if f is None or f.height == 0:
            continue
        f = f.filter(pl.col("known_ts") < cutoff).sort("event_ts")
        if f.height < 48:
            continue
        frames[s] = f
        lab = label(f, bar_hours=1.0)
        cells[s] = lab.filter((pl.col("close_ts").dt.hour() == 0) & (pl.col("close_ts").dt.minute() == 0)).select(
            pl.col("close_ts").dt.date().alias("date"), "cell").drop_nulls("cell")
    cd = _grid_from_frames(frames, cells, universe, cutoff, use_oi)
    if cd.known_max >= cutoff:
        raise AssertionError("calibration data contains rows known at or after the cutoff")
    return cd


# ====================================================================== derived arrays (parameter-free)
@dataclass
class Derived:
    r: np.ndarray          # (H, N) hourly log returns (NaN if missing or previous bar missing)
    rv: np.ndarray         # (D, N)
    bv: np.ndarray         # (D, N)
    ret: np.ndarray        # (D, N)
    y: np.ndarray          # (D, N) log BV (NaN on incomplete days)
    ok: np.ndarray         # (D, N) complete day
    wd: np.ndarray         # (D,) weekday 0 = Monday
    how: np.ndarray        # (H,)
    hod: np.ndarray        # (H,)
    lt: np.ndarray         # (H, N) log turnover
    hl2: np.ndarray        # (H, N) (log high/low)^2
    fu8: np.ndarray        # (D*3, N) funding at hours 0, 8, 16
    r24: np.ndarray        # (D*3, N) trailing 24h return before each print
    doi: np.ndarray | None  # (D*6, N) dlog OI between 4h prints
    r4: np.ndarray | None   # (D*6, N) 4h return before each print
    day_of_hour: np.ndarray


def derive(cd: CalibData) -> Derived:
    H, N = cd.C.shape
    D = cd.n_days
    lc = np.log(cd.C)
    r = np.full((H, N), np.nan)
    r[1:] = lc[1:] - lc[:-1]
    rr = r.reshape(D, 24, N)
    ok = np.isfinite(rr).all(axis=1)
    rv = np.where(ok, np.nansum(rr**2, axis=1), np.nan)
    a = np.abs(rr)
    bv = np.where(ok, BV_SCALE * np.nansum(a[:, 1:] * a[:, :-1], axis=1), np.nan)
    bv = np.where(ok, np.maximum(bv, np.maximum(0.05 * rv, 1e-10)), np.nan)
    ret = np.where(ok, np.nansum(rr, axis=1), np.nan)
    y = np.log(bv)
    wd0 = cd.day0.weekday()
    wd = (wd0 + np.arange(D)) % 7
    hod = np.tile(np.arange(24), D)
    how = np.repeat(wd, 24) * 24 + hod
    with np.errstate(divide="ignore", invalid="ignore"):
        lt = np.where(cd.TV > 0, np.log(cd.TV), np.nan)
        hl2 = np.where((cd.Hi > 0) & (cd.Lo > 0), np.log(cd.Hi / cd.Lo) ** 2, np.nan)
    rc = np.nan_to_num(r)
    cs = np.vstack([np.zeros((1, N)), np.cumsum(rc, axis=0)])
    cnt = np.vstack([np.zeros((1, N)), np.cumsum(np.isfinite(r), axis=0)])
    p8 = np.arange(0, H, 8)
    fu8 = cd.FU[p8]
    lo8 = np.maximum(p8 - 24, 0)
    r24 = np.where(cnt[p8] - cnt[lo8] >= 20, cs[p8] - cs[lo8], np.nan)
    r24[p8 < 24] = np.nan
    doi = r4 = None
    if cd.OI is not None:
        p4 = np.arange(0, H, 4)
        with np.errstate(divide="ignore", invalid="ignore"):
            loi = np.where(cd.OI[p4] > 0, np.log(cd.OI[p4]), np.nan)
        doi = np.full_like(loi, np.nan)
        doi[1:] = loi[1:] - loi[:-1]
        lo4 = np.maximum(p4 - 4, 0)
        r4 = np.where(cnt[p4] - cnt[lo4] >= 4, cs[p4] - cs[lo4], np.nan)
        r4[p4 < 4] = np.nan
    return Derived(r=r, rv=rv, bv=bv, ret=ret, y=y, ok=ok, wd=wd, how=how, hod=hod, lt=lt, hl2=hl2, fu8=fu8, r24=r24, doi=doi, r4=r4,
                   day_of_hour=np.repeat(np.arange(D), 24))


# ====================================================================== weighted helpers
def block_weights(n_days: int, rng: np.random.Generator, block: int = BLOCK_DAYS) -> np.ndarray:
    """Day counts of a circular moving-block bootstrap of n_days (one replicate)."""
    nb = int(math.ceil(n_days / block))
    starts = rng.integers(0, n_days, size=nb)
    idx = ((starts[:, None] + np.arange(block)[None, :]).ravel()[:n_days]) % n_days
    return np.bincount(idx, minlength=n_days).astype(float)


def _wmean(x: np.ndarray, w: np.ndarray, axis=None):
    m = np.isfinite(x) & (w > 0)
    ws = np.where(m, w, 0.0)
    return np.sum(np.where(m, x, 0.0) * ws, axis=axis) / np.maximum(np.sum(ws, axis=axis), 1e-300)


def _wols(X: np.ndarray, y: np.ndarray, w: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Weighted OLS on finite rows: (beta, se (iid, weights as frequency), mask)."""
    m = np.isfinite(y) & np.all(np.isfinite(X), axis=1) & (w > 0)
    Xm, ym, wm = X[m], y[m], w[m]
    XtW = Xm.T * wm
    A = XtW @ Xm
    beta = np.linalg.lstsq(A, XtW @ ym, rcond=None)[0]
    e = ym - Xm @ beta
    dof = max(wm.sum() - X.shape[1], 1.0)
    s2 = float((wm * e * e).sum() / dof)
    try:
        cov = s2 * np.linalg.pinv(A)
        se = np.sqrt(np.maximum(np.diag(cov), 0.0))
    except np.linalg.LinAlgError:
        se = np.full(X.shape[1], np.nan)
    return beta, se, m


def _wkurt(x: np.ndarray, w: np.ndarray) -> float:
    m = np.isfinite(x) & (w > 0)
    x, w = x[m], w[m]
    mu = np.sum(w * x) / np.sum(w)
    v = np.sum(w * (x - mu) ** 2) / np.sum(w)
    return float(np.sum(w * (x - mu) ** 4) / np.sum(w) / v**2 - 3.0)


def nu_from_kurt(k: float, lo: float = 4.2, hi: float = 60.0) -> float:
    """Student-t df with excess kurtosis k (6 / (nu - 4)), clipped."""
    if not np.isfinite(k) or k <= 6.0 / (hi - 4.0):
        return hi
    return float(np.clip(4.0 + 6.0 / k, lo, hi))


def _by_how(v: np.ndarray, how: np.ndarray) -> np.ndarray:
    """Sum of v (H, N) by hour of week (NaN-free input)."""
    return np.bincount(np.broadcast_to(how[:, None], v.shape).ravel(), weights=v.ravel(), minlength=HOW)


def _by_how_sym(v: np.ndarray, how: np.ndarray) -> np.ndarray:
    """(N, 168) sums of v (H, N) by symbol and hour of week (NaN-free input)."""
    N = v.shape[1]
    g = (np.arange(N)[None, :] * HOW + how[:, None]).ravel()
    return np.bincount(g, weights=v.ravel(), minlength=N * HOW).reshape(N, HOW)


def _shrunk_profile(sums: np.ndarray, wsum: np.ndarray, pooled: np.ndarray, kappa: float) -> np.ndarray:
    """Per-symbol group means (N, G) from sums / weights, shrunk toward the pooled profile (G,) with kappa pseudo-obs."""
    raw = sums / np.maximum(wsum, 1e-12)
    return (wsum * raw + kappa * pooled[None, :]) / (wsum + kappa)


def _lag_means(x: np.ndarray, k: int, min_n: int) -> np.ndarray:
    """Mean of x[d-k..d-1] along axis 0 (NaN-aware, at least min_n values)."""
    D = x.shape[0]
    xs = np.nan_to_num(x)
    c = np.concatenate([np.zeros((1,) + x.shape[1:]), np.cumsum(xs, axis=0)])
    n = np.concatenate([np.zeros((1,) + x.shape[1:]), np.cumsum(np.isfinite(x), axis=0)])
    out = np.full(x.shape, np.nan)
    for d in range(1, D):
        lo = max(0, d - k)
        cnt = n[d] - n[lo]
        out[d] = np.where(cnt >= min_n, (c[d] - c[lo]) / np.maximum(cnt, 1), np.nan)
    return out


def _eb_shrink(est: np.ndarray, se: np.ndarray, pooled: float) -> np.ndarray:
    """Normal-normal empirical Bayes: est_S -> pooled + w_S (est_S - pooled), w = tau^2 / (tau^2 + se^2)."""
    ok = np.isfinite(est) & np.isfinite(se)
    if ok.sum() < 3:
        return np.where(ok, est, pooled)
    tau2 = max(float(np.var(est[ok]) - np.mean(se[ok] ** 2)), 0.0)
    w = tau2 / (tau2 + se**2)
    return np.where(ok, pooled + w * (est - pooled), pooled)


# ====================================================================== hourly statistics
def g_stats(zn: np.ndarray, w_h: np.ndarray, lags: Sequence[int] = LAGS_G, max_pairs: int = 60) -> dict[str, object]:
    """Statistics of non-jump standardised hourly returns zn (H, N), g = log(zn^2 + G_OFF):
    ac = within-symbol autocorrelation of g at `lags` (intraday log-vol persistence), cc1 = mean cross-symbol lag-1
    correlation of g (common persistent log-vol; the common return shock is serially independent so it does not enter),
    cz = mean pairwise correlation of zn (return co-movement), kurt = pooled excess kurtosis of zn."""
    g = np.log(zn**2 + G_OFF)
    Wb = np.broadcast_to(w_h[:, None], g.shape)
    gm = _wmean(g, Wb, axis=0)
    gd = g - gm
    var = _wmean(gd**2, Wb)
    ac = []
    for k in lags:
        prod = gd[k:] * gd[:-k]
        ac.append(_wmean(prod, np.broadcast_to(w_h[k:, None], prod.shape)) / var)
    N = zn.shape[1]
    pairs = [(i, j) for i in range(N) for j in range(N) if i != j]
    if len(pairs) > max_pairs:
        sel = np.random.default_rng(17).choice(len(pairs), max_pairs, replace=False)
        pairs = [pairs[k] for k in sorted(sel)]
    cc, cz = [], []
    for i, j in pairs:
        a, b = gd[1:, i], gd[:-1, j]
        m = np.isfinite(a) & np.isfinite(b) & (w_h[1:] > 0)
        if m.sum() < 500:
            continue
        ww = w_h[1:][m]
        cc.append(np.sum(ww * a[m] * b[m]) / math.sqrt(np.sum(ww * a[m] ** 2) * np.sum(ww * b[m] ** 2)))
        za, zb = zn[:, i], zn[:, j]
        m2 = np.isfinite(za) & np.isfinite(zb) & (w_h > 0)
        w2 = w_h[m2]
        cz.append(np.sum(w2 * za[m2] * zb[m2]) / math.sqrt(np.sum(w2 * za[m2] ** 2) * np.sum(w2 * zb[m2] ** 2)))
    kurt = _wkurt(zn.ravel(), Wb.ravel())
    return {"ac": np.array(ac), "cc1": float(np.mean(cc)) if cc else 0.0, "cz": float(np.mean(cz)) if cz else 0.0, "kurt": kurt}


MOM_SCALE = np.array([0.004] * len(LAGS_G) + [0.004, 0.02, 0.05])


def moment_vector(st: dict) -> np.ndarray:
    return np.r_[np.asarray(st["ac"], float), st["cc1"], st["cz"], st["kurt"]]


def cojump_phi(jumps: np.ndarray, valid: np.ndarray, w_h: np.ndarray, max_pairs: int = 60) -> float:
    """Mean pairwise phi coefficient (correlation of 0/1 indicators) of detected jump hours across symbols."""
    N = jumps.shape[1]
    pairs = [(i, j) for i in range(N) for j in range(i + 1, N)]
    if len(pairs) > max_pairs:
        sel = np.random.default_rng(23).choice(len(pairs), max_pairs, replace=False)
        pairs = [pairs[k] for k in sorted(sel)]
    out = []
    for i, j in pairs:
        m = valid[:, i] & valid[:, j] & (w_h > 0)
        if m.sum() < 500:
            continue
        ww = w_h[m]
        a, b = jumps[m, i].astype(float), jumps[m, j].astype(float)
        ma, mb = np.sum(ww * a) / ww.sum(), np.sum(ww * b) / ww.sum()
        cov = np.sum(ww * (a - ma) * (b - mb)) / ww.sum()
        den = math.sqrt(ma * (1 - ma) * mb * (1 - mb))
        if den > 0:
            out.append(cov / den)
    return float(np.mean(out)) if out else 0.0


def tail_stats(zc: np.ndarray, jumps: np.ndarray, w_h: np.ndarray) -> dict:
    """Co-jump phi, exceedance rate of |zc| > TAIL_U and a GPD fit of the exceedances (weights as repetitions)."""
    from scipy import stats

    valid = np.isfinite(zc)
    Wv = np.where(valid, w_h[:, None], 0.0)
    az = np.where(valid, np.abs(zc), 0.0)
    rate = float(np.sum(Wv * (az > TAIL_U)) / max(np.sum(Wv), 1e-12))
    ex = az[(az > TAIL_U) & (Wv > 0)] - TAIL_U
    reps = np.round(Wv[(az > TAIL_U) & (Wv > 0)]).astype(int)
    exr = np.repeat(ex, reps)
    if len(exr) >= 30:
        xi, _, sc = stats.genpareto.fit(exr, floc=0)
        xi = float(np.clip(xi, -0.2, 0.5))
    else:
        xi, sc = 0.0, 1.0
    return {"cojump_phi": cojump_phi(jumps & valid, valid, w_h), "tail_rate": rate, "gpd_xi": xi, "gpd_scale": float(sc)}


def one_factor(C: np.ndarray, iters: int = 30) -> np.ndarray:
    """Loadings of a one-factor model of a correlation matrix (principal-axis factoring)."""
    C = np.array(C, float)
    n = C.shape[0]
    if n == 1:
        return np.array([0.0])
    C = np.where(np.isfinite(C), C, 0.0)
    h2 = np.clip(np.nanmax(np.abs(C - np.eye(n)), axis=1), 0.01, 0.95)
    for _ in range(iters):
        M = C.copy()
        np.fill_diagonal(M, h2)
        vals, vecs = np.linalg.eigh(M)
        lam = vecs[:, -1] * math.sqrt(max(vals[-1], 1e-9))
        lam *= np.sign(lam.sum()) or 1.0
        h2 = np.clip(lam**2, 0.0, 0.95)
    return np.clip(lam, 0.0, 0.975)


def corr_clusters(C: np.ndarray, thresh: float = CLUSTER_CORR) -> np.ndarray:
    """Cluster ids (0..K-1, ordered by first member) from average-linkage clustering on 1 - correlation, cut at
    1 - thresh. Missing correlations count as 0."""
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import squareform

    n = C.shape[0]
    if n == 1:
        return np.zeros(1, dtype=np.int64)
    Dm = 1.0 - np.where(np.isfinite(C), C, 0.0)
    np.fill_diagonal(Dm, 0.0)
    Dm = np.clip((Dm + Dm.T) / 2, 0.0, 2.0)
    lab = fcluster(linkage(squareform(Dm, checks=False), method="average"), t=1.0 - thresh, criterion="distance")
    order = {c: k for k, c in enumerate(dict.fromkeys(lab.tolist()))}
    return np.array([order[c] for c in lab], dtype=np.int64)


def one_factor_clusters(C: np.ndarray, cl: np.ndarray) -> np.ndarray:
    """Loadings on each symbol's own cluster factor (one-factor model per cluster; singletons load 0)."""
    lam = np.zeros(C.shape[0])
    for k in np.unique(cl):
        ix = np.flatnonzero(cl == k)
        if len(ix) >= 2:
            lam[ix] = one_factor(C[np.ix_(ix, ix)])
    return lam


def zn_corr(zn: np.ndarray, w_h: np.ndarray) -> np.ndarray:
    m = np.isfinite(zn)
    X = np.where(m, zn, 0.0) * np.sqrt(w_h)[:, None]
    M = m * np.sqrt(w_h)[:, None]
    S = X.T @ X
    Q = (X**2).T @ (M > 0)
    with np.errstate(invalid="ignore", divide="ignore"):
        C = S / np.sqrt(Q * Q.T)
    cnt = (M > 0).astype(float).T @ (M > 0).astype(float)
    C[cnt < 500] = np.nan
    np.fill_diagonal(C, 1.0)
    return C


# ====================================================================== jumps (point fit, asymptotic covariance)
@dataclass
class JumpFit:
    theta: np.ndarray          # unconstrained (logit(p/0.2), logit(p_up), log eta_up, log eta_down)
    cov: np.ndarray            # asymptotic covariance of theta (design effect applied)
    sigma0: float              # diffusive scale used in the mixture
    hawkes: list[dict] = field(default_factory=list)  # per symbol: {"th": [log n/(1-n), log beta], "cov": 2x2, "n_jumps"}

    @staticmethod
    def unpack(th: np.ndarray) -> tuple[float, float, float, float]:
        from scipy import special

        return float(special.expit(th[0]) * 0.2), float(special.expit(th[1])), float(math.exp(th[2])), float(math.exp(th[3]))


def _num_hessian(f: Callable[[np.ndarray], float], x: np.ndarray, h: float = 1e-3) -> np.ndarray:
    n = len(x)
    Hm = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            ei, ej = np.zeros(n), np.zeros(n)
            ei[i], ej[j] = h, h
            v = (f(x + ei + ej) - f(x + ei - ej) - f(x - ei + ej) + f(x - ei - ej)) / (4 * h * h)
            Hm[i, j] = Hm[j, i] = v
    return Hm


def _psd_inv(Hm: np.ndarray, floor: float = 1e-8) -> np.ndarray:
    vals, vecs = np.linalg.eigh((Hm + Hm.T) / 2)
    vals = np.maximum(vals, floor)
    return (vecs / vals) @ vecs.T


def fit_jumps(zc: np.ndarray, rng: np.random.Generator, n_sub: int = 300_000, design_effect: float = 2.0) -> tuple[JumpFit, np.ndarray]:
    """Kou mixture on pooled standardised hourly returns zc (H, N) + per-symbol Hawkes on the detected jump hours.
    Returns the fit and the (H, N) boolean jump mask (posterior > 0.5)."""
    from scipy import special

    from ai_trader.features.jump_diffusion import _hawkes_nll, fit_hawkes, fit_mixture, mixture_logpdf

    flat = zc[np.isfinite(zc)]
    sub = flat if len(flat) <= n_sub else rng.choice(flat, n_sub, replace=False)
    s0 = float(1.4826 * np.median(np.abs(sub - np.median(sub))))
    sig = np.full(len(sub), s0)
    jp = fit_mixture(sub, sig)
    th = np.array([special.logit(min(jp.p, 0.1999) / 0.2), special.logit(jp.p_up), math.log(jp.eta_up), math.log(jp.eta_down)])

    def nll(t):
        p, pu, eu, ed = JumpFit.unpack(t)
        return -float(np.sum(mixture_logpdf(sub, sig, p, pu, eu, ed)))

    Hm = _num_hessian(nll, th, h=1e-3)
    cov = _psd_inv(Hm) * (len(sub) / len(flat)) * design_effect
    m = np.isfinite(zc)
    jf0 = JumpFit(theta=th, cov=cov, sigma0=s0)
    jumps = (np.nan_to_num(jump_posterior(zc, jf0)) > 0.5) & m
    hk = []
    for j in range(zc.shape[1]):
        col = m[:, j]
        pos = np.cumsum(col) - 1          # position in the symbol's own series (missing hours skipped)
        times = pos[jumps[:, j]].astype(float)
        T = float(col.sum())
        if len(times) < 20 or T < 1000:
            hk.append({"th": np.array([math.log(0.3 / 0.7), math.log(0.3)]), "cov": np.diag([1.0, 1.0]), "n_jumps": int(len(times)), "pooled": True})
            continue
        mu, a, b, _ = fit_hawkes(times, T)
        nb = min(max(a / b, 1e-3), 0.95)
        t0 = np.array([math.log(nb / (1 - nb)), math.log(b)])

        def hnll(t, times=times, T=T):
            nn = 1 / (1 + math.exp(-t[0]))
            bb = math.exp(t[1])
            rate = len(times) / T
            return _hawkes_nll([math.log(max(rate * (1 - nn), 1e-12)), math.log(nn * bb), math.log(bb)], times, T)

        Hh = _num_hessian(hnll, t0, h=1e-2)
        hk.append({"th": t0, "cov": _psd_inv(Hh, 1e-4), "n_jumps": int(len(times)), "pooled": False})
    # symbols with too few jumps take the median of the fitted ones (with a wide covariance)
    fitted = [h["th"] for h in hk if not h["pooled"]]
    if fitted:
        med = np.median(np.array(fitted), axis=0)
        for h in hk:
            if h["pooled"]:
                h["th"] = med.copy()
                h["cov"] = np.diag([0.5, 0.5])
    return JumpFit(theta=th, cov=cov, sigma0=s0, hawkes=hk), jumps


# ====================================================================== the weighted estimator
def standardise(dv: Derived, q: np.ndarray) -> np.ndarray:
    """z_c = r / sqrt(q_{i,how} * BV_day) on complete days; q: (N, 168)."""
    bvh = np.repeat(dv.bv, 24, axis=0)
    okh = np.repeat(dv.ok, 24, axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        z = dv.r / np.sqrt(q[:, dv.how].T * bvh)
    return np.where(okh & np.isfinite(z), z, np.nan)


def estimate(cd: CalibData, dv: Derived, w: np.ndarray, jumps: np.ndarray | None = None) -> dict:
    """One (weighted) estimate of every bootstrap-able parameter and hourly statistic. w: (D,) day weights."""
    D, N = cd.n_days, cd.n_sym
    W = np.where(dv.ok, w[:, None], 0.0)
    w_h = w[dv.day_of_hour]
    y = dv.y
    p: dict = {}
    # ---- daily level, weekday, factor
    a = _wmean(y, W, axis=0)
    dev = y - a[None, :]
    wk_pool = np.array([_wmean(dev[dv.wd == k], W[dv.wd == k]) for k in range(7)])
    wk_pool -= wk_pool.mean()
    dz = np.where(W > 0, dev, 0.0) * W
    s_k = np.stack([dz[dv.wd == k].sum(axis=0) for k in range(7)], axis=1)          # (N, 7)
    n_k = np.stack([W[dv.wd == k].sum(axis=0) for k in range(7)], axis=1)
    wk = _shrunk_profile(s_k, n_k, wk_pool, KAPPA_SYM)
    wk -= wk.mean(axis=1, keepdims=True)
    x = dev - wk[:, dv.wd].T
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)       # days with no complete symbol -> NaN
        F = np.nanmean(np.where(dv.ok, x, np.nan), axis=1)
    b = np.array([np.sum(np.where(np.isfinite(x[:, i]) & np.isfinite(F), W[:, i] * x[:, i] * F, 0.0))
                  / max(np.sum(np.where(np.isfinite(x[:, i]) & np.isfinite(F), W[:, i] * F * F, 0.0)), 1e-12) for i in range(N)])
    F1 = np.r_[np.nan, F[:-1]]
    F5 = _lag_means(F, 5, 3)
    F22 = _lag_means(F, 22, 10)
    XF = np.column_stack([np.ones(D), F1, F5, F22])
    bF, seF, mF = _wols(XF, F, w)
    resF = F - XF @ bF
    sF = math.sqrt(_wmean(resF**2, np.where(mF, w, 0.0)))
    # long-run level used in simulation: the trailing LEVEL_DAYS in-sample mean of y (in-sample check 2026-11-13: it
    # forecasts the next 132 days' mean log BV better than the full-history mean, MSE 0.31 vs 0.38, 45 origins)
    Wl_ = W.copy()
    Wl_[: max(0, D - LEVEL_DAYS)] = 0.0
    a_lvl = _wmean(dev - wk[:, dv.wd].T, Wl_, axis=0) + a
    a_lvl = np.where(np.isfinite(a_lvl) & (Wl_.sum(axis=0) >= 60), a_lvl, a)
    p.update(a=a_lvl, a_full=a, wk=wk, wk_pool=wk_pool, b=b, har_F=bF, se_har_F=seF, sig_F=sF, nu_F=nu_from_kurt(_wkurt(resF[mF], w[mF])))
    # ---- idiosyncratic HAR + regime shifts
    e = x - b[None, :] * F[:, None]
    e1 = np.vstack([np.full((1, N), np.nan), e[:-1]])
    e5 = _lag_means(e, 5, 3)
    e22 = _lag_means(e, 22, 10)
    cl = cd.cell
    rows = np.isfinite(e) & np.isfinite(e1) & np.isfinite(e5) & np.isfinite(e22) & (cl >= 0) & (W > 0)
    dummies = np.zeros((rows.sum(), N_CELLS))
    dummies[np.arange(rows.sum()), cl[rows]] = 1.0
    Xe = np.column_stack([e1[rows], e5[rows], e22[rows], dummies])
    be, see, _ = _wols(Xe, e[rows], W[rows])
    res = np.full((D, N), np.nan)
    res[rows] = e[rows] - Xe @ be
    sig_e = np.sqrt(_wmean(res**2, W, axis=0))
    sig_e = np.where(np.isfinite(sig_e) & (sig_e > 0), sig_e, np.nanmedian(sig_e))
    gam = be[3:]
    present = np.bincount(cl[rows], minlength=N_CELLS) > 0
    gam = np.where(present, gam, 0.0)
    p.update(har_e=be[:3], se_har_e=see[:3], gamma=gam, sig_e=sig_e,
             nu_e=nu_from_kurt(_wkurt((res / sig_e[None, :])[rows], W[rows])))
    # ---- transition matrix of start-of-day cells
    cnt = np.full((N_CELLS, N_CELLS), 0.1)
    a0, a1 = cl[:-1], cl[1:]
    okp = (a0 >= 0) & (a1 >= 0)
    np.add.at(cnt, (a0[okp], a1[okp]), np.broadcast_to(w[:-1, None], a0.shape)[okp])
    p["P"] = cnt / cnt.sum(axis=1, keepdims=True)
    # ---- hour-of-week variance shares
    rvh = np.repeat(dv.rv, 24, axis=0)
    okh = np.repeat(dv.ok, 24, axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        share = np.where(okh, dv.r**2 / rvh, np.nan)
    cap = np.nanquantile(share, 0.995)
    share = np.minimum(share, cap)
    Wh = np.where(okh & np.isfinite(share), w_h[:, None], 0.0)
    sw = _by_how(np.where(Wh > 0, share, 0.0) * Wh, dv.how)
    nw = _by_how(Wh, dv.how)
    q_raw = sw / np.maximum(nw, 1e-12)
    q_hod = np.array([np.sum(sw[np.arange(HOW) % 24 == h]) / max(np.sum(nw[np.arange(HOW) % 24 == h]), 1e-12) for h in range(24)])
    q = (nw * q_raw + KAPPA_Q * q_hod[np.arange(HOW) % 24]) / (nw + KAPPA_Q)
    q = q.reshape(7, 24)
    q_pool = (q / q.sum(axis=1, keepdims=True)).ravel()
    sw_i = _by_how_sym(np.where(Wh > 0, share, 0.0) * Wh, dv.how)
    nw_i = _by_how_sym(Wh, dv.how)
    qi = _shrunk_profile(sw_i, nw_i, q_pool, KAPPA_SYM).reshape(N, 7, 24)
    q = (qi / qi.sum(axis=2, keepdims=True)).reshape(N, HOW)
    p["q"] = q
    p["q_pool"] = q_pool
    # ---- standardised hourly returns and their statistics
    zc = standardise(dv, q)
    p["_zc"] = zc  # transient (dropped by the caller)
    if jumps is not None:
        zn = np.where(jumps, np.nan, zc)
        p["stats"] = g_stats(zn, w_h)
        p["stats"].update(tail_stats(zc, jumps, w_h))
        Cz = zn_corr(zn, w_h)
        p["cluster"] = corr_clusters(Cz)
        p["lam_raw"] = one_factor_clusters(Cz, p["cluster"])
        # regime-dependent hourly AR(1) of z (winsorised), by start-of-day cell
        zw = np.clip(zc, -5, 5)
        z1 = np.vstack([np.full((1, N), np.nan), zw[:-1]])
        cell_h = np.repeat(cl, 24, axis=0)
        mm = np.isfinite(zw) & np.isfinite(z1) & (Wh > 0)
        est, se = np.full(N_CELLS, np.nan), np.full(N_CELLS, np.nan)
        num = np.sum(np.where(mm, Wh * zw * z1, 0.0))
        den = np.sum(np.where(mm, Wh * z1 * z1, 0.0))
        pooled = num / den
        for s in range(N_CELLS):
            ms = mm & (cell_h == s)
            if ms.sum() < 200:
                continue
            dd = np.sum(Wh[ms] * z1[ms] ** 2)
            est[s] = np.sum(Wh[ms] * zw[ms] * z1[ms]) / dd
            resid = zw[ms] - est[s] * z1[ms]
            se[s] = math.sqrt(np.sum(Wh[ms] * resid**2) / np.sum(Wh[ms]) / dd)
        p["phi_hS"] = _eb_shrink(est, se, pooled)
        p["phi_h_pooled"] = float(pooled)
        # daily AR(1) of u = ret / sqrt(BV) by start-of-day cell of the later day
        u = np.clip(dv.ret / np.sqrt(dv.bv), -6, 6)
        u1 = np.vstack([np.full((1, N), np.nan), u[:-1]])
        md = np.isfinite(u) & np.isfinite(u1) & (W > 0)
        estd, sed = np.full(N_CELLS, np.nan), np.full(N_CELLS, np.nan)
        pool_d = np.sum(np.where(md, W * u * u1, 0.0)) / np.sum(np.where(md, W * u1 * u1, 0.0))
        for s in range(N_CELLS):
            ms = md & (cl == s)
            if ms.sum() < 50:
                continue
            dd = np.sum(W[ms] * u1[ms] ** 2)
            estd[s] = np.sum(W[ms] * u[ms] * u1[ms]) / dd
            resid = u[ms] - estd[s] * u1[ms]
            sed[s] = math.sqrt(np.sum(W[ms] * resid**2) / np.sum(W[ms]) / dd)
        p["phi_dS"] = _eb_shrink(estd, sed, pool_d)
        p["phi_d_pooled"] = float(pool_d)
        # wick factor: Parkinson / RV on non-jump hours
        mk = np.isfinite(dv.hl2) & np.isfinite(dv.r) & ~jumps & (Wh > 0)
        p["kappa_w"] = float(np.sum(Wh[mk] * dv.hl2[mk]) / (4 * math.log(2) * np.sum(Wh[mk] * dv.r[mk] ** 2)))
        # ---- turnover
        Wl = np.where(np.isfinite(dv.lt) & okh, w_h[:, None], 0.0)
        at = _wmean(dv.lt, Wl, axis=0)
        d1 = dv.lt - at[None, :]
        tau_pool = _by_how(np.where(Wl > 0, d1, 0.0) * Wl, dv.how) / np.maximum(_by_how(Wl, dv.how), 1e-12)
        tau_pool -= tau_pool.mean()
        tau = _shrunk_profile(_by_how_sym(np.where(Wl > 0, d1, 0.0) * Wl, dv.how), _by_how_sym(Wl, dv.how), tau_pool, KAPPA_SYM)
        tau -= tau.mean(axis=1, keepdims=True)
        r1 = d1 - tau[:, dv.how].T
        xh = np.repeat(x, 24, axis=0)
        az = np.minimum(np.abs(zc), 6.0) - 0.8
        mt = np.isfinite(r1) & np.isfinite(xh) & np.isfinite(az) & (Wl > 0)
        Xt = np.column_stack([xh[mt], az[mt]])
        bt, _, _ = _wols(Xt, r1[mt], Wl[mt])
        r2 = np.full_like(r1, np.nan)
        r2[mt] = r1[mt] - Xt @ bt
        r2p = np.vstack([np.full((1, N), np.nan), r2[:-1]])
        mp = np.isfinite(r2) & np.isfinite(r2p) & (Wl > 0)
        phi_v = float(np.sum(Wl[mp] * r2[mp] * r2p[mp]) / np.sum(Wl[mp] * r2p[mp] ** 2))
        innov = np.where(mp, r2 - phi_v * r2p, np.nan)
        sig_v = np.sqrt(_wmean(innov**2, np.where(mp, Wl, 0.0), axis=0))
        sig_v = np.where(np.isfinite(sig_v) & (sig_v > 0), sig_v, np.nanmedian(sig_v))
        # recent level (last 30 in-sample days) so the generated turnover continues the cutoff level
        last = slice(max(0, len(dv.lt) - 720), len(dv.lt))
        a_rec = np.array([np.nanmean(r1[last, i]) + at[i] if np.isfinite(r1[last, i]).sum() > 24 else at[i] for i in range(N)])
        p.update(tau=tau, b_tau=float(bt[0]), c_tau=float(bt[1]), phi_v=float(np.clip(phi_v, 0, 0.995)), sig_v=sig_v, a_tau=a_rec)
        # ---- funding (8h prints)
        w8 = w[np.arange(0, len(dv.day_of_hour), 8) // 24]
        p.update(_ar_aux(dv.fu8, dv.r24, w8, prefix="f", level=True))
        if dv.doi is not None:
            w4 = w[np.arange(0, len(dv.day_of_hour), 4) // 24]
            p.update(_ar_aux(dv.doi, dv.r4, w4, prefix="o", level=False))
    return p


def _ar_aux(v: np.ndarray, ret: np.ndarray, w: np.ndarray, prefix: str, level: bool) -> dict:
    """Standardised AR(1) with a lagged-return regressor, pooled across symbols (funding level or OI change)."""
    W = np.where(np.isfinite(v) & np.isfinite(ret), w[:, None], 0.0)
    m = _wmean(v, W, axis=0) if level else np.zeros(v.shape[1])
    sd = np.sqrt(_wmean((v - m[None, :]) ** 2, W, axis=0))
    sd = np.where(np.isfinite(sd) & (sd > 0), sd, np.nan)
    vs = np.clip((v - m[None, :]) / sd[None, :], -8, 8)
    rsd = np.sqrt(_wmean(ret**2, W, axis=0))
    rs = np.clip(ret / rsd[None, :], -8, 8)
    v1 = np.vstack([np.full((1, v.shape[1]), np.nan), vs[:-1]])
    mm = np.isfinite(vs) & np.isfinite(v1) & np.isfinite(rs) & (W > 0)
    X = np.column_stack([v1[mm], rs[mm]])
    beta, _, _ = _wols(X, vs[mm], W[mm])
    res = vs[mm] - X @ beta
    s = math.sqrt(float(np.sum(W[mm] * res**2) / np.sum(W[mm])))
    return {f"{prefix}_m": m, f"{prefix}_sd": np.where(np.isfinite(sd), sd, np.nanmedian(sd)), f"{prefix}_rsd": np.where(np.isfinite(rsd) & (rsd > 0), rsd, np.nanmedian(rsd)),
            f"{prefix}_phi": float(np.clip(beta[0], -0.99, 0.999)), f"{prefix}_beta": float(beta[1]), f"{prefix}_sig": s,
            f"{prefix}_nu": nu_from_kurt(_wkurt(res, W[mm]))}


# ====================================================================== simulated moments for the hourly layer
HOURLY_KEYS = ("sig_h", "phi_h", "psi", "kurt_eps", "lam_scale")


def _t_std(rng: np.random.Generator, nu, size) -> np.ndarray:
    nu = np.asarray(nu, float)
    return rng.standard_t(np.broadcast_to(nu, size)) * np.sqrt((nu - 2.0) / nu)


def _ar1(e: np.ndarray, phi, axis: int = -1) -> np.ndarray:
    """x_t = phi x_{t-1} + e_t along `axis`, x_{-1} = 0; phi is per leading index (per path) for ndim > 1."""
    from scipy.signal import lfilter

    e = np.moveaxis(e, axis, -1)
    if e.ndim == 1:
        return lfilter([1.0], [1.0, -float(np.ravel(phi)[0])], e)
    out = np.empty_like(e)
    phi = np.broadcast_to(np.asarray(phi, float), e.shape[:1])
    for k in range(e.shape[0]):
        out[k] = lfilter([1.0], [1.0, -float(phi[k])], e[k], axis=-1)
    return np.moveaxis(out, -1, axis)


def diffusive_hours(q: np.ndarray, how: np.ndarray, sig_h, phi_h, psi, nu, lam: np.ndarray, n_paths: int,
                    rng: np.random.Generator, cluster: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Unnormalised diffusive draws a (P, N, H) and their scale sqrt(q) exp(h) (P, N, H).
    q: (P, N, 168) shares; how: (H,); sig_h, phi_h, psi, nu: (P,); lam: (P, N). h starts from its stationary law."""
    P, N = lam.shape
    H = len(how)
    sig_h, phi_h, psi, nu = (np.broadcast_to(np.asarray(v, float), (P,)) for v in (sig_h, phi_h, psi, nu))
    sd_in = sig_h * np.sqrt(np.maximum(1 - phi_h**2, 1e-6))
    ec = rng.standard_normal((P, H)) * sd_in[:, None]
    ei = rng.standard_normal((P, N, H)) * sd_in[:, None, None]
    ec[:, 0] = rng.standard_normal(P) * sig_h
    ei[:, :, 0] = rng.standard_normal((P, N)) * sig_h[:, None]
    hc = _ar1(ec, phi_h)
    hi = _ar1(ei, phi_h)
    h = np.sqrt(psi)[:, None, None] * hc[:, None, :] + np.sqrt(1 - psi)[:, None, None] * hi - (sig_h**2)[:, None, None]
    cl = np.zeros((P, N), dtype=np.int64) if cluster is None else np.broadcast_to(np.asarray(cluster, np.int64), (P, N))
    K = int(cl.max()) + 1
    m = _t_std(rng, nu[:, None, None], (P, K, H))
    u = _t_std(rng, nu[:, None, None], (P, N, H))
    mc = np.take_along_axis(m, cl[:, :, None].repeat(H, axis=2), axis=1)       # each symbol's cluster shock
    eps = lam[:, :, None] * mc + np.sqrt(1 - lam**2)[:, :, None] * u
    sq = np.sqrt(np.asarray(q)[:, :, how])
    scale = sq * np.exp(h)
    return scale * eps, scale


def normalise_days(a: np.ndarray, scale: np.ndarray, bv: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Scale each day's 24 hourly draws so that their bipower variation equals bv (P, N, D)."""
    P, N, H = a.shape
    D = H // 24
    ad = np.abs(a.reshape(P, N, D, 24))
    bvd = BV_SCALE * np.sum(ad[..., 1:] * ad[..., :-1], axis=-1)
    c = np.sqrt(bv / np.maximum(bvd, 1e-300))
    ch = np.repeat(c, 24, axis=-1)
    return a * ch, scale * ch


def kou_draw(rng: np.random.Generator, n: int, p_up, eta_u, eta_d) -> np.ndarray:
    up = rng.random(n) < p_up
    return np.where(up, rng.exponential(1.0, n) / eta_u, -rng.exponential(1.0, n) / eta_d)


def jump_posterior(z: np.ndarray, jf: JumpFit) -> np.ndarray:
    """P(jump | z) under the pooled Kou fit; extreme |z| (numerical overflow) count as jumps."""
    from ai_trader.features.jump_diffusion import JumpParams, posterior

    p, pu, eu, ed = JumpFit.unpack(jf.theta)
    prm = JumpParams(p=p, p_up=pu, eta_up=eu, eta_down=ed)
    out = np.full(z.shape, np.nan)
    m = np.isfinite(z)
    with np.errstate(all="ignore"):
        pz = posterior(z[m], np.full(m.sum(), jf.sigma0), prm)
    pz = np.where(np.isfinite(pz), pz, np.where(np.abs(z[m]) > 3 * jf.sigma0, 1.0, 0.0))
    out[m] = pz
    return out


def _theta_to_hp(th: np.ndarray) -> dict:
    from scipy import special

    return {"sig_h": float(math.exp(th[0])), "phi_h": float(special.expit(th[1]) * 0.99), "psi": float(special.expit(th[2])),
            "kurt_eps": float(math.exp(th[3])), "lam_scale": float(math.exp(th[4]))}


def _hp_to_theta(hp: dict) -> np.ndarray:
    from scipy import special

    return np.array([math.log(hp["sig_h"]), special.logit(min(max(hp["phi_h"] / 0.99, 1e-4), 1 - 1e-4)),
                     special.logit(min(max(hp["psi"], 1e-4), 1 - 1e-4)), math.log(max(hp["kurt_eps"], 1e-3)), math.log(hp["lam_scale"])])


def jump_arrivals(rng: np.random.Generator, shape: tuple, p: float, omega: float, tail_rate: float, p_up: float, eta_u: float,
                  eta_d: float, xi: float, gscale: float) -> np.ndarray:
    """Constant-rate jump sizes (z units) of shape (1, N, H): common events (share omega) hit every symbol in the same
    hour with one size (lognormal dispersion across symbols), the rest are idiosyncratic; Kou sizes, plus GPD tail jumps
    (|J| = TAIL_U + GPD(xi, scale)) at `tail_rate`."""
    from scipy import stats

    _, N, H = shape
    J = np.zeros(shape)
    for rate, kind in ((p, "kou"), (tail_rate, "tail")):
        if rate <= 0:
            continue
        c = rng.random(H) < rate * omega
        idio = rng.random(shape) < rate * (1 - omega)

        def size(n):
            if kind == "kou":
                return kou_draw(rng, n, p_up, eta_u, eta_d)
            sg = np.where(rng.random(n) < p_up, 1.0, -1.0)
            return sg * (TAIL_U + stats.genpareto.rvs(xi, scale=gscale, size=n, random_state=rng))

        if c.any():
            sc = size(int(c.sum()))
            disp = np.exp(COMMON_SIZE_DISP * rng.standard_normal((N, int(c.sum()))) - COMMON_SIZE_DISP**2 / 2)
            J[0][:, c] += sc[None, :] * disp
        if idio.any():
            J[idio] += size(int(idio.sum()))
    return J


def _sim_hourly_z(hp: dict, base: dict, jf: JumpFit, seed: int, n_days: int, n_sym: int, omega: float = 0.0,
                  tail_rate: float = 0.0) -> np.ndarray:
    """Standardised hourly returns (H, n_sym) of the hourly layer (constant daily BV; the normalisation removes the
    level), with jumps per `jump_arrivals`; standardised exactly like the data."""
    rng = np.random.default_rng(seed)
    wd = np.arange(n_days) % 7
    how = np.repeat(wd, 24) * 24 + np.tile(np.arange(24), n_days)
    lam = np.clip(np.full((1, n_sym), float(np.mean(base["lam_raw"]))) * hp["lam_scale"], 0, 0.975)
    qp = base["q_pool"]
    a, scale = diffusive_hours(np.broadcast_to(qp, (1, n_sym, HOW)), how, hp["sig_h"], hp["phi_h"], hp["psi"], nu_from_kurt(hp["kurt_eps"]),
                               lam, 1, rng)
    rc, scale = normalise_days(a, scale, np.ones((1, n_sym, n_days)))
    p, pu, eu, ed = JumpFit.unpack(jf.theta)
    st = base["stats"]
    J = jump_arrivals(rng, rc.shape, p, omega, tail_rate, pu, eu, ed, st.get("gpd_xi", 0.0), st.get("gpd_scale", 1.0))
    r = rc + J * np.sqrt(qp[how])[None, None, :]
    rr = r[0].T.reshape(n_days, 24, n_sym)
    ab = np.abs(rr)
    bv = BV_SCALE * np.sum(ab[:, 1:] * ab[:, :-1], axis=1)
    return r[0].T / np.sqrt(qp[how][:, None] * np.repeat(bv, 24, axis=0))


def jump_structure(base: dict, hp: dict, jf: JumpFit, seed: int = 11, n_days: int = 1500, n_sym: int = 6) -> dict:
    """Common-jump share omega (matches the in-sample co-jump phi of detected jumps) and the GPD tail-jump rate
    (matches the in-sample exceedance rate of |zc| > TAIL_U), by simulation with common random numbers.
    Also returns d phi / d omega for mapping bootstrap replicates."""
    st = base["stats"]

    def sim(omega, tail):
        z = _sim_hourly_z(hp, base, jf, seed, n_days, n_sym, omega, tail)
        post = jump_posterior(z, jf)
        jm = np.nan_to_num(post) > 0.5
        valid = np.isfinite(z)
        return cojump_phi(jm, valid, np.ones(z.shape[0])), float(np.mean(np.abs(z[valid]) > TAIL_U))

    tail = 0.0
    lo, hi = 0.0, 1.0
    for _ in range(2):
        lo, hi = 0.0, 1.0
        for _ in range(12):
            mid = (lo + hi) / 2
            if sim(mid, tail)[0] < st["cojump_phi"]:
                lo = mid
            else:
                hi = mid
        omega = (lo + hi) / 2
        rate0 = sim(omega, 0.0)[1]
        tail = max(st["tail_rate"] - rate0, 0.0)
    ph_a, rt = sim(min(omega + 0.05, 1.0), tail)
    ph_b, _ = sim(max(omega - 0.05, 0.0), tail)
    dphi = (ph_a - ph_b) / (min(omega + 0.05, 1.0) - max(omega - 0.05, 0.0))
    return {"omega": float(omega), "tail_rate": float(tail), "dphi_domega": float(dphi), "model_tail_rate_no_gpd": float(rate0),
            "fit_phi": float(sim(omega, tail)[0]), "fit_rate": float(sim(omega, tail)[1])}


def profile_drift(cd: "CalibData", dv: "Derived", point: dict, jumps: np.ndarray | None = None, window: int = DRIFT_WINDOW) -> dict:
    """Between-window dispersion beyond sampling noise (in-sample, non-overlapping windows ending at the cutoff) of
    slowly drifting parameters: tau_wd (weekday log-variance effect), tau_q (log hour-of-week shares), tau_corr
    (Fisher-z of the mean cross-asset hourly return correlation lam^2), tau_jump (log jump intensity)."""
    D, N = cd.n_days, cd.n_sym
    x = dv.y - point["a_full"][None, :] - point["wk"][:, dv.wd].T
    rvh = np.repeat(dv.rv, 24, axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        rel = np.where(np.repeat(dv.ok, 24, axis=0), dv.r**2 / rvh, np.nan) / point["q"][:, dv.how].T
    rel = np.minimum(rel, np.nanquantile(rel, 0.995))
    m_wd, se_wd, m_q, se_q = [], [], [], []
    for e in range(D, window, -window):
        sl = slice(e - window, e)
        xw = x[sl] - np.nanmean(x[sl], axis=0, keepdims=True)
        if np.sum(np.isfinite(xw).any(axis=0)) < min(3, N):
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            v = np.nanmean(xw, axis=1)
            wdw = dv.wd[sl]
            for k in range(7):
                vk = v[(wdw == k) & np.isfinite(v)]
                if len(vk) >= 5:
                    m_wd.append(vk.mean())
                    se_wd.append(vk.var(ddof=1) / len(vk))
            hs = slice((e - window) * 24, e * 24)
            relw = np.nanmean(rel[hs], axis=1)          # symbol-averaged relative share per hour
            howw = dv.how[hs]
            for h in range(HOW):
                vh = relw[(howw == h) & np.isfinite(relw)]
                if len(vh) >= 5 and vh.mean() > 0:
                    m_q.append(math.log(vh.mean()))
                    se_q.append(vh.var(ddof=1) / len(vh) / vh.mean() ** 2)
    tw = max(float(np.mean(np.square(m_wd)) - np.mean(se_wd)), 0.0) if m_wd else 0.0
    tq = max(float(np.mean(np.square(np.array(m_q) - np.mean(m_q))) - np.mean(se_q)), 0.0) if m_q else 0.0
    out = {"tau_wd": math.sqrt(tw), "tau_q": math.sqrt(tq), "n_wd_cells": len(m_wd), "n_q_cells": len(m_q), "tau_corr": 0.0, "tau_jump": 0.0}
    if jumps is not None:
        zc = standardise(dv, point["q"])
        zn = np.where(jumps, np.nan, zc)
        fz, lj, sej = [], [], []
        for e in range(D, window, -window):
            hs = slice((e - window) * 24, e * 24)
            zw = zn[hs]
            valid = np.isfinite(zc[hs])
            if np.sum(valid.any(axis=0)) < 3:
                continue
            C = zn_corr(zw, np.ones(zw.shape[0]))
            iu = np.triu_indices(C.shape[0], 1)
            cv = C[iu]
            cv = cv[np.isfinite(cv)]
            if len(cv):
                fz.append(math.atanh(float(np.clip(np.mean(cv), -0.99, 0.99))))
            nj = int(np.sum(jumps[hs] & valid))
            if nj >= 20:
                lj.append(math.log(nj / valid.sum()))
                sej.append(1.0 / nj)
        if len(fz) >= 3:
            out["tau_corr"] = float(np.std(fz, ddof=1))
        if len(lj) >= 3:
            out["tau_jump"] = math.sqrt(max(float(np.var(lj, ddof=1)) - float(np.mean(sej)), 0.0))
        out["n_windows"] = len(fz)
    return out


def sim_moments(th: np.ndarray, base: dict, jf: JumpFit, seed: int, n_days: int, n_sym: int) -> np.ndarray:
    """Moment vector of the hourly layer simulated at unconstrained parameters th (common random numbers via seed),
    standardised and jump-classified exactly like the data (constant daily BV: the normalisation removes the level)."""
    hp = _theta_to_hp(th)
    rng = np.random.default_rng(seed)
    wd = np.arange(n_days) % 7
    how = np.repeat(wd, 24) * 24 + np.tile(np.arange(24), n_days)
    lam = np.clip(np.full((1, n_sym), float(np.mean(base["lam_raw"]))) * hp["lam_scale"], 0, 0.975)
    qp = base["q_pool"]
    a, scale = diffusive_hours(np.broadcast_to(qp, (1, n_sym, HOW)), how, hp["sig_h"], hp["phi_h"], hp["psi"], nu_from_kurt(hp["kurt_eps"]),
                               lam, 1, rng)
    rc, scale = normalise_days(a, scale, np.ones((1, n_sym, n_days)))
    p, pu, eu, ed = JumpFit.unpack(jf.theta)
    H = n_days * 24
    jm = rng.random((1, n_sym, H)) < p
    J = np.zeros((1, n_sym, H))
    J[jm] = kou_draw(rng, int(jm.sum()), pu, eu, ed)
    r = rc + J * np.sqrt(qp[how])[None, None, :]
    rr = r[0].T.reshape(n_days, 24, n_sym)
    ab = np.abs(rr)
    bv = BV_SCALE * np.sum(ab[:, 1:] * ab[:, :-1], axis=1)
    zc = r[0].T / np.sqrt(qp[how][:, None] * np.repeat(bv, 24, axis=0))
    post = jump_posterior(zc, jf)
    zn = np.where(post > 0.5, np.nan, zc)
    return moment_vector(g_stats(zn, np.ones(H)))


def smm(base: dict, jf: JumpFit, seed: int = 5, n_days: int = 1000, n_sym: int = 4, maxfev: int = 300) -> dict:
    """Hourly parameters whose simulated moments match the in-sample moments (Nelder-Mead on the scaled distance),
    plus the Jacobian used to map bootstrap replicates (`hourly_params`)."""
    from scipy import optimize

    target = moment_vector(base["stats"])
    st = base["stats"]
    hp0 = {"sig_h": 0.3, "phi_h": 0.5, "psi": 0.5, "kurt_eps": max(float(st["kurt"]) + 0.3, 0.1), "lam_scale": 1.0}
    th0 = _hp_to_theta(hp0)

    def obj(th):
        m = sim_moments(th, base, jf, seed, n_days, n_sym)
        return float(np.sum(((m - target) / MOM_SCALE) ** 2))

    simplex = np.vstack([th0] + [th0 + 0.6 * np.eye(len(th0))[k] for k in range(len(th0))])
    res = optimize.minimize(obj, th0, method="Nelder-Mead",
                            options={"maxfev": maxfev, "xatol": 1e-3, "fatol": 1e-3, "initial_simplex": simplex})
    th = res.x
    m0 = sim_moments(th, base, jf, seed, n_days, n_sym)
    J = np.zeros((len(target), len(th)))
    for k in range(len(th)):
        e = np.zeros(len(th))
        e[k] = 0.1
        J[:, k] = (sim_moments(th + e, base, jf, seed, n_days, n_sym) - sim_moments(th - e, base, jf, seed, n_days, n_sym)) / 0.2
    hp = _theta_to_hp(th)
    hp.update({"theta": th, "J": J, "fit_moments": m0, "target": target, "objective": float(res.fun), "nfev": int(res.nfev)})
    return hp


def hourly_params(rep: dict, point: dict, hp: dict) -> dict:
    """Replicate hourly parameters: theta_b = theta_hat + pinv(J) (m_b - m_hat) in scaled moment units (one Gauss-Newton
    step from the SMM solution), bounded to +/- 1.5 in each unconstrained coordinate."""
    Js = hp["J"] / MOM_SCALE[:, None]
    dm = (moment_vector(rep["stats"]) - moment_vector(point["stats"])) / MOM_SCALE
    step = np.linalg.lstsq(Js, dm, rcond=None)[0]
    th = hp["theta"] + np.clip(step, -1.5, 1.5)
    out = _theta_to_hp(th)
    out["nu_eps"] = nu_from_kurt(out["kurt_eps"])
    out["lam"] = np.clip(rep["lam_raw"] * out.pop("lam_scale"), 0.0, 0.975)
    return out


# ====================================================================== start state at the cutoff
def start_state(cd: CalibData, dv: Derived) -> dict:
    """Data needed to continue each symbol from the cutoff (raw values; transformed with each path's parameters)."""
    N = cd.n_sym
    H = cd.C.shape[0]
    last_close = np.array([cd.C[np.flatnonzero(np.isfinite(cd.C[:, i]))[-1], i] for i in range(N)])
    last_t = np.array([np.flatnonzero(np.isfinite(cd.C[:, i]))[-1] for i in range(N)])
    # The grid's last day (cutoff - 1 day) is incomplete (its last bar closes at the cutoff) and is generation day 0:
    # HAR lags and the previous-day return end the day before it.
    ylag = dv.y[-31:-1].copy()
    wd_lag = dv.wd[-31:-1].copy()
    cell_last = np.array([cd.cell[-1, i] if cd.cell[-1, i] >= 0 else -1 for i in range(N)])

    def last_val(a):
        return np.array([a[np.flatnonzero(np.isfinite(a[:, i]))[-1], i] if np.isfinite(a[:, i]).any() else np.nan for i in range(N)])

    fu_last = last_val(cd.FU)
    oi_last = last_val(cd.OI) if cd.OI is not None else None
    ret_last = dv.ret[-2].copy()
    bv_last = dv.bv[-2].copy()
    r24_last = np.array([np.nansum(dv.r[H - 24:, i]) for i in range(N)])
    return {"last_close": last_close, "last_t": last_t, "ylag": ylag, "wd_lag": wd_lag, "cell_last": cell_last, "fu_last": fu_last,
            "oi_last": oi_last, "ret_last": ret_last, "bv_last": bv_last, "r24_last": r24_last}


# ====================================================================== per-path parameter sets
def draw_jump_params(jf: JumpFit, n_sym: int, rng: np.random.Generator | None) -> dict:
    """Kou (pooled) and Hawkes (per symbol) parameters; rng=None returns the point estimate."""
    th = jf.theta if rng is None else rng.multivariate_normal(jf.theta, jf.cov)
    p, pu, eu, ed = JumpFit.unpack(th)
    nb, beta = np.zeros(n_sym), np.zeros(n_sym)
    for i, h in enumerate(jf.hawkes[:n_sym]):
        t = h["th"] if rng is None else rng.multivariate_normal(h["th"], h["cov"])
        nb[i] = float(np.clip(1 / (1 + math.exp(-t[0])), 0.0, 0.95))
        beta[i] = float(np.clip(math.exp(t[1]), 0.02, 5.0))
    return {"p": p, "p_up": pu, "eta_u": eu, "eta_d": ed, "n_br": nb, "beta_h": beta}


def path_params(rep: dict, point: dict, hp: dict, jf: JumpFit, n_sym: int, rng: np.random.Generator | None,
                js: dict | None = None, drift: dict | None = None) -> dict:
    """Full parameter set of one path from one bootstrap replicate. rng=None: point jumps and no profile drift (the
    fixed-parameter generator). js = jump_structure output (common share, tail rate), drift = profile_drift output."""
    out = {k: v for k, v in rep.items() if not k.startswith("_") and k not in ("stats", "lam_raw")}
    out.update(hourly_params(rep, point, hp))
    out.update(draw_jump_params(jf, n_sym, rng))
    st, st0 = rep["stats"], point["stats"]
    if js is not None:
        om = js["omega"] + (st["cojump_phi"] - st0["cojump_phi"]) / js["dphi_domega"] if js["dphi_domega"] > 1e-6 else js["omega"]
        out["omega"] = float(np.clip(om, 0.0, 0.98))
        out["tail_rate"] = float(max(js["tail_rate"] + (st["tail_rate"] - st0["tail_rate"]), 0.0))
    else:
        out["omega"], out["tail_rate"] = 0.0, 0.0
    out["gpd_xi"], out["gpd_scale"] = float(st["gpd_xi"]), float(st["gpd_scale"])
    if drift is not None and rng is not None:
        d = rng.standard_normal(7) * drift["tau_wd"]
        out["wk"] = out["wk"] + (d - d.mean())[None, :]
        lq = np.log(out["q"]) + rng.standard_normal(HOW)[None, :] * drift["tau_q"]
        qq = np.exp(lq).reshape(n_sym, 7, 24)
        out["q"] = (qq / qq.sum(axis=2, keepdims=True)).reshape(n_sym, HOW)
        l2 = np.clip(out["lam"] ** 2, 1e-4, 0.95)
        l2 = np.tanh(np.arctanh(l2) + rng.standard_normal() * drift.get("tau_corr", 0.0))
        out["lam"] = np.sqrt(np.clip(l2, 0.0, 0.95))
        mj = math.exp(rng.standard_normal() * drift.get("tau_jump", 0.0))
        out["p"] = float(min(out["p"] * mj, 0.5))
        out["tail_rate"] = float(out["tail_rate"] * mj)
    return out


def stack_params(ps: Sequence[dict]) -> dict:
    keys = ps[0].keys()
    return {k: np.stack([np.asarray(p[k], float) for p in ps]) for k in keys if not isinstance(ps[0][k], dict | list)}


# ====================================================================== simulation
@dataclass
class GenSpec:
    """Calendar of a generated window: n_days full UTC days from day0 (the first kept hour is `first_hour`)."""

    day0: datetime
    n_days: int
    first_hour: int = 0
    has_oi: bool = True

    @property
    def wd(self) -> np.ndarray:
        return (self.day0.weekday() + np.arange(self.n_days)) % 7

    @property
    def how(self) -> np.ndarray:
        return np.repeat(self.wd, 24) * 24 + np.tile(np.arange(24), self.n_days)


def simulate(pp: dict, state: dict, spec: GenSpec, rng: np.random.Generator) -> dict[str, np.ndarray]:
    """Generate P paths (P = leading dim of the stacked parameters pp) for all N symbols over spec.n_days.

    Returns open, high, low, close, turnover, funding, oi (P, N, H_kept) and diagnostics bv (P, N, D), cells (P, N, D)."""
    P, N = pp["a"].shape
    D = spec.n_days
    H = 24 * D
    wd = spec.wd
    how = spec.how
    L = 22
    # ---- daily layer: lags from the real data transformed with each path's parameters
    ylag = state["ylag"][-L:]                                     # (L, N)
    wdl = state["wd_lag"][-L:]
    xlag = ylag[None, :, :] - pp["a"][:, None, :] - np.transpose(pp["wk"][:, :, wdl], (0, 2, 1))   # (P, L, N)
    with np.errstate(invalid="ignore"):
        Flag = np.nanmean(xlag, axis=2)                                # (P, L)
    Flag = np.where(np.isfinite(Flag), Flag, 0.0)
    elag = xlag - pp["b"][:, None, :] * Flag[:, :, None]
    elag = np.where(np.isfinite(elag), elag, 0.0)
    Fh = np.concatenate([Flag, np.zeros((P, D))], axis=1)
    eh = np.concatenate([np.transpose(elag, (0, 2, 1)), np.zeros((P, N, D))], axis=2)   # (P, N, L+D)
    cells = np.zeros((P, N, D), dtype=np.int64)
    c_prev = np.where(state["cell_last"] >= 0, state["cell_last"], 4)
    c_prev = np.broadcast_to(c_prev, (P, N)).copy()
    cumP = np.cumsum(pp["P"], axis=2)
    cumP[:, :, -1] = 1.0
    nuF, nue = pp["nu_F"], pp["nu_e"]
    for d in range(D):
        k = L + d
        if d > 0:
            u = rng.random((P, N))
            rows = np.take_along_axis(cumP, c_prev[:, :, None].repeat(N_CELLS, axis=2), axis=1)   # (P, N, 9)
            c_prev = np.minimum((rows <= u[:, :, None]).sum(axis=2), N_CELLS - 1)
        cells[:, :, d] = c_prev
        hf = pp["har_F"]
        Fh[:, k] = (hf[:, 0] + hf[:, 1] * Fh[:, k - 1] + hf[:, 2] * Fh[:, k - 5:k].mean(axis=1) + hf[:, 3] * Fh[:, k - 22:k].mean(axis=1)
                    + pp["sig_F"] * _t_std(rng, nuF, (P,)))
        he = pp["har_e"]
        gam = np.take_along_axis(pp["gamma"], c_prev, axis=1)
        eh[:, :, k] = (he[:, 0:1] * eh[:, :, k - 1] + he[:, 1:2] * eh[:, :, k - 5:k].mean(axis=2) + he[:, 2:3] * eh[:, :, k - 22:k].mean(axis=2)
                       + gam + pp["sig_e"] * _t_std(rng, nue[:, None], (P, N)))
    x = pp["b"][:, :, None] * Fh[:, None, L:] + eh[:, :, L:]
    y = pp["a"][:, :, None] + pp["wk"][:, :, wd] + x
    bv = np.exp(y)
    # ---- hourly diffusive layer
    clus = pp["cluster"].astype(np.int64) if "cluster" in pp else np.zeros((P, N), dtype=np.int64)
    a, scale = diffusive_hours(pp["q"], how, pp["sig_h"], pp["phi_h"], pp["psi"], pp["nu_eps"], pp["lam"], P, rng, cluster=clus)
    rc, scale = normalise_days(a, scale, bv)
    qh = np.sqrt(pp["q"][:, :, how])                                     # (P, N, H)
    jscale = qh * np.repeat(np.sqrt(bv), 24, axis=2)
    # ---- sequential: regime AR, daily AR drift, Hawkes-Kou jumps (idiosyncratic per symbol + common events)
    from scipy import stats as _st

    r = np.empty((P, N, H))
    n_br, beta_h = pp["n_br"], pp["beta_h"]
    om = pp["omega"][:, None]
    alpha = n_br * beta_h
    mu = pp["p"][:, None] * (1 - om) * (1 - n_br)
    decay = np.exp(-beta_h)
    K = int(clus.max()) + 1
    onehot = (clus[:, :, None] == np.arange(K)[None, None, :]).astype(float)            # (P, N, K)
    single = onehot.sum(axis=1) <= 1                                                     # singleton clusters: no common events
    nb_c = np.median(n_br, axis=1)[:, None]
    be_c = np.median(beta_h, axis=1)[:, None]
    mu_c = np.where(single, 0.0, (pp["p"] * pp["omega"])[:, None] * (1 - nb_c))          # (P, K)
    al_c = nb_c * be_c
    dec_c = np.exp(-be_c)
    A = np.zeros((P, N))
    Ac = np.zeros((P, K))
    # symbols in singleton clusters keep their whole jump rate idiosyncratic
    om_eff = np.where(np.take_along_axis(single, clus, axis=1), 0.0, om)
    mu = pp["p"][:, None] * (1 - om_eff) * (1 - n_br)
    tail_i = pp["tail_rate"][:, None] * (1 - om_eff)
    tail_c = np.where(single, 0.0, (pp["tail_rate"] * pp["omega"])[:, None])

    def tail_size(n, pu, xi, sc):
        sg = np.where(rng.random(n) < pu, 1.0, -1.0)
        return sg * (TAIL_U + _st.genpareto.rvs(xi, scale=sc, size=n, random_state=rng))

    prev_rt = np.zeros((P, N))
    prev_sc = np.ones((P, N))
    u_prev = np.where(np.isfinite(state["ret_last"] / np.sqrt(state["bv_last"])), state["ret_last"] / np.sqrt(state["bv_last"]), 0.0)
    u_prev = np.broadcast_to(np.clip(u_prev, -6, 6), (P, N)).copy()
    day_ret = np.zeros((P, N))
    drift = np.zeros((P, N))
    for t in range(H):
        d = t // 24
        if t % 24 == 0:
            if d > 0:
                u_prev = np.clip(day_ret / np.sqrt(bv[:, :, d - 1]), -6, 6)
            phd = np.take_along_axis(pp["phi_dS"], cells[:, :, d], axis=1)
            drift = phd * u_prev * np.sqrt(bv[:, :, d]) / 24.0
            day_ret = np.zeros((P, N))
            phh = np.take_along_axis(pp["phi_hS"], cells[:, :, d], axis=1)
        sc = scale[:, :, t]
        rt = rc[:, :, t] + phh * prev_rt * sc / np.maximum(prev_sc, 1e-300)
        lam_t = mu + alpha * A
        jump = rng.random((P, N)) < (1 - np.exp(-lam_t))
        J = np.zeros((P, N))
        nj = int(jump.sum())
        if nj:
            pu = np.broadcast_to(pp["p_up"][:, None], (P, N))[jump]
            eu = np.broadcast_to(pp["eta_u"][:, None], (P, N))[jump]
            ed = np.broadcast_to(pp["eta_d"][:, None], (P, N))[jump]
            J[jump] = kou_draw(rng, nj, pu, eu, ed)
        A = decay * (A + jump)
        cj = rng.random((P, K)) < (1 - np.exp(-(mu_c + al_c * Ac)))
        if cj.any():
            k = int(cj.sum())
            pk = np.broadcast_to(np.arange(P)[:, None], (P, K))[cj]
            szk = np.zeros((P, K))
            szk[cj] = kou_draw(rng, k, pp["p_up"][pk], pp["eta_u"][pk], pp["eta_d"][pk])
            szi = np.einsum("pnk,pk->pn", onehot, szk)
            J += szi * np.exp(COMMON_SIZE_DISP * rng.standard_normal((P, N)) - COMMON_SIZE_DISP**2 / 2)
        Ac = dec_c * (Ac + cj)
        ti = rng.random((P, N)) < tail_i
        if ti.any():
            k = int(ti.sum())
            J[ti] += tail_size(k, np.broadcast_to(pp["p_up"][:, None], (P, N))[ti], np.broadcast_to(pp["gpd_xi"][:, None], (P, N))[ti],
                               np.broadcast_to(pp["gpd_scale"][:, None], (P, N))[ti])
        tc = rng.random((P, K)) < tail_c
        if tc.any():
            k = int(tc.sum())
            pk = np.broadcast_to(np.arange(P)[:, None], (P, K))[tc]
            szk = np.zeros((P, K))
            szk[tc] = tail_size(k, pp["p_up"][pk], pp["gpd_xi"][pk], pp["gpd_scale"][pk])
            szi = np.einsum("pnk,pk->pn", onehot, szk)
            J += szi * np.exp(COMMON_SIZE_DISP * rng.standard_normal((P, N)) - COMMON_SIZE_DISP**2 / 2)
        r[:, :, t] = rt + drift + J * jscale[:, :, t]
        day_ret += r[:, :, t]
        prev_rt, prev_sc = rt, sc
    # ---- OHLC (hours before spec.first_hour exist only to complete day 0's normalisation: no price move there)
    r[:, :, :spec.first_hour] = 0.0
    lc = np.log(state["last_close"])[None, :, None] + np.cumsum(r, axis=2)
    lo_ = np.concatenate([np.broadcast_to(np.log(state["last_close"])[None, :, None], (P, N, 1)), lc[:, :, :-1]], axis=2)
    s2 = pp["kappa_w"][:, None, None] * scale**2
    u1 = rng.random((P, N, H))
    u2 = rng.random((P, N, H))
    m_up = (r + np.sqrt(r**2 - 2 * s2 * np.log(u1))) / 2
    m_dn = (r - np.sqrt(r**2 - 2 * s2 * np.log(u2))) / 2
    hi = lo_ + np.maximum(m_up, np.maximum(r, 0.0))
    lw = lo_ + np.minimum(m_dn, np.minimum(r, 0.0))
    # ---- turnover
    xh = np.repeat(x, 24, axis=2)
    with np.errstate(invalid="ignore", divide="ignore"):
        z = r / jscale
    az = np.minimum(np.abs(np.nan_to_num(z)), 6.0) - 0.8
    v = _ar1(rng.standard_normal((P, N, H)) * pp["sig_v"][:, :, None], pp["phi_v"])
    lt = (pp["a_tau"][:, :, None] + pp["tau"][:, :, how]
          + pp["b_tau"][:, None, None] * (xh - x[:, :, :1]) + pp["c_tau"][:, None, None] * az + v)
    # ---- funding prints every 8h (step function)
    cs = np.concatenate([np.zeros((P, N, 1)), np.cumsum(r, axis=2)], axis=2)
    fu = np.empty((P, N, H))
    fz = np.broadcast_to(np.clip(np.nan_to_num((state["fu_last"] - pp["f_m"]) / pp["f_sd"]), -8, 8), (P, N)).copy()
    cur = np.broadcast_to(np.nan_to_num(state["fu_last"]), (P, N)).copy()
    for t in range(H):
        if t % 8 == 0 and t > 0:
            r24 = cs[:, :, t] - cs[:, :, max(t - 24, 0)]
            rz = np.clip(r24 / pp["f_rsd"], -8, 8)
            fz = pp["f_phi"][:, None] * fz + pp["f_beta"][:, None] * rz + pp["f_sig"][:, None] * _t_std(rng, pp["f_nu"][:, None], (P, N))
            cur = pp["f_m"] + pp["f_sd"] * fz
        fu[:, :, t] = cur
    out = {"open": np.exp(lo_), "high": np.exp(hi), "low": np.exp(lw), "close": np.exp(lc), "turnover": np.exp(lt), "funding": fu}
    if spec.has_oi and "o_phi" in pp and state.get("oi_last") is not None:
        oi = np.empty((P, N, H))
        loi = np.broadcast_to(np.log(np.where(state["oi_last"] > 0, state["oi_last"], 1.0)), (P, N)).copy()
        dz = np.zeros((P, N))
        for t in range(H):
            if t % 4 == 0 and t > 0:
                r4 = cs[:, :, t] - cs[:, :, t - 4]
                rz = np.clip(r4 / pp["o_rsd"], -8, 8)
                dz = pp["o_phi"][:, None] * dz + pp["o_beta"][:, None] * rz + pp["o_sig"][:, None] * _t_std(rng, pp["o_nu"][:, None], (P, N))
                loi = loi + pp["o_sd"] * dz
            oi[:, :, t] = np.exp(loi)
        out["oi"] = oi
    k0 = spec.first_hour
    out = {k: v[:, :, k0:] for k, v in out.items()}
    out["bv"] = bv
    out["cells"] = cells
    return out


# ====================================================================== B5-style stationary block bootstrap comparator (joint)
def bootstrap_source(cd: CalibData) -> dict:
    """Per symbol compact in-sample rows (time order) of the relative bar (r, g, u, d), turnover, funding, dlogOI."""
    from ai_trader.sim.resample import politis_white_block, relative_bars

    src = []
    blocks = []
    for i in range(cd.n_sym):
        m = np.isfinite(cd.C[:, i]) & np.isfinite(cd.O[:, i]) & np.isfinite(cd.Hi[:, i]) & np.isfinite(cd.Lo[:, i])
        ix = np.flatnonzero(m)
        rb = relative_bars(cd.O[ix, i], cd.Hi[ix, i], cd.Lo[ix, i], cd.C[ix, i])
        tv = cd.TV[ix[1:], i]
        fu = cd.FU[ix[1:], i]
        if cd.OI is not None:
            with np.errstate(divide="ignore", invalid="ignore"):
                lo = np.log(np.where(cd.OI[ix, i] > 0, cd.OI[ix, i], np.nan))
            doi = np.nan_to_num(np.diff(lo))
        else:
            doi = np.zeros(len(ix) - 1)
        src.append({"r": rb.r, "g": rb.g, "u": rb.u, "d": rb.d, "tv": np.nan_to_num(tv, nan=np.nanmedian(tv)), "fu": np.nan_to_num(fu), "doi": doi})
        blocks.append(politis_white_block(np.abs(rb.r))["b"])
    return {"src": src, "mean_block": float(np.median(blocks)), "blocks": blocks, "n_max": max(len(s["r"]) for s in src)}


def bootstrap_paths(bs: dict, state: dict, n_paths: int, n_hours: int, rng: np.random.Generator) -> dict[str, np.ndarray]:
    """Joint stationary block bootstrap: one index path on the time axis of the longest history, mapped to every symbol
    end-aligned (circular for shorter histories), so contemporaneous cross-asset dependence is kept."""
    from ai_trader.sim.resample import stationary_indices

    n_max = bs["n_max"]
    idx = stationary_indices(n_max, n_hours, bs["mean_block"], n_paths, rng)          # (P, H)
    N = len(bs["src"])
    out = {k: np.empty((n_paths, N, n_hours)) for k in ("open", "high", "low", "close", "turnover", "funding", "oi")}
    for i, s in enumerate(bs["src"]):
        n_i = len(s["r"])
        ii = (idx - (n_max - n_i)) % n_i
        r, g, u, d = s["r"][ii], s["g"][ii], s["u"][ii], s["d"][ii]
        c0 = math.log(state["last_close"][i])
        lc = c0 + np.cumsum(r, axis=1)
        lcp = np.concatenate([np.full((n_paths, 1), c0), lc[:, :-1]], axis=1)
        lo_ = lcp + g
        out["open"][:, i] = np.exp(lo_)
        out["close"][:, i] = np.exp(lc)
        out["high"][:, i] = np.exp(np.maximum(lo_, lc) + u)
        out["low"][:, i] = np.exp(np.minimum(lo_, lc) - d)
        out["turnover"][:, i] = s["tv"][ii]
        out["funding"][:, i] = s["fu"][ii]
        oi0 = state["oi_last"][i] if state.get("oi_last") is not None and np.isfinite(state["oi_last"][i]) else 1.0
        out["oi"][:, i] = oi0 * np.exp(np.cumsum(s["doi"][ii], axis=1))
    return out
