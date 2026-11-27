"""WS08 PREDICT (+ the cores of WS09 XSECT and WS18 MACRO), specification program 2026-11-27.

Directional return predictability with persistent-regressor and overlapping-horizon inference, cross-sectional pricing of Bybit
perps, and macro betas / implied-vol forecasts of realised variance. Hypotheses H-WS08-1..22 (registered before any fit; design in
the diary entry "WS08 design frozen before results"). Report: research/reports/ws08_2026-11-27.md.

Point in time: every predictor is read as of known_ts <= origin; targets are forward log returns over (origin, origin + h], divided
by the ex-ante EWMA sigma x sqrt(h). Segments split at research.knowledge_cutoff (2026-07-01); in-sample fits use only pairs whose
target window closed before it.

Inference primitives (numpy only, tested in tests/test_ws08_predict.py):
  ols_nw          OLS slope with Newey-West (Bartlett) standard error
  hodrick_t       Hodrick (1992) 1B via the reverse regression of the one-period return on the h-sum of the predictor
  stambaugh       Stambaugh (1999) small-sample bias of the predictive slope with an AR(1) regressor
  ivx_wald        Kostakis-Magdalinos-Stamatogiannis (2015) IVX Wald test, one-period and long-horizon (k > 1) forms
  block_counts    moving-block bootstrap (blocks of days) as multiplicity counts; romano_wolf step-down on studentised slopes
  clark_west      Clark-West (2007) MSPE-adjusted test; giacomini_white conditional predictive ability test (rolling window)
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import polars as pl
from scipy import stats as sst

CUTOFF = datetime(2026, 7, 1, tzinfo=timezone.utc)
ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "data" / "research" / "ws08"
H_MS = 3_600_000
D_MS = 86_400_000
UNIVERSE_L = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "DOGEUSDT", "ADAUSDT", "LINKUSDT", "LTCUSDT", "BCHUSDT", "AVAXUSDT",
              "DOTUSDT", "BNBUSDT", "TRXUSDT", "NEARUSDT", "ATOMUSDT", "ETCUSDT", "FILUSDT", "UNIUSDT", "AAVEUSDT", "XLMUSDT",
              "SUIUSDT", "ENAUSDT", "HYPEUSDT"]
ALTS = [s for s in UNIVERSE_L if s not in ("BTCUSDT", "ETHUSDT")]
TRADFI = {"SOXLUSDT", "QQQUSDT", "NVDAUSDT", "TSLAUSDT", "MSTRUSDT", "MUUSDT", "GOOGLUSDT", "AAPLUSDT", "INTCUSDT", "SPCXUSDT",
          "XAUUSDT", "XAGUSDT", "CLUSDT", "BZUSDT", "XAUTUSDT", "PAXGUSDT", "USDJPYUSDT", "EURUSDUSDT", "SPYUSDT", "IWMUSDT",
          "TLTUSDT", "SMHUSDT", "XLEUSDT", "GDXUSDT", "UVXYUSDT", "IBITUSDT", "CRWVUSDT", "IRENUSDT", "CORZUSDT", "COINUSDT",
          "HOODUSDT", "MARAUSDT", "CRCLUSDT", "AMDSTOCKUSDT", "AVGOUSDT", "TSMUSDT", "METAUSDT", "MSFTUSDT", "ORCLUSDT"}
COST_TAKER_RT_BP = 11.0          # 5.5 bp per side
COST_MAKER_ENTRY_RT_BP = 8.8     # pegged limit entry 2.0 + 1.3 bp, taker exit 5.5 bp


def cutoff_ms() -> int:
    return int(CUTOFF.timestamp() * 1000)


# ====================================================================== inference primitives

def _design(x: np.ndarray) -> np.ndarray:
    return np.column_stack([np.ones(len(x)), x])


def nw_meat(u: np.ndarray, lag: int) -> np.ndarray:
    """Bartlett-weighted long-run covariance (sum, not mean) of score rows u (n x k)."""
    S = u.T @ u
    for lg in range(1, min(lag, len(u) - 1) + 1):
        w = 1.0 - lg / (lag + 1.0)
        G = u[lg:].T @ u[:-lg]
        S = S + w * (G + G.T)
    return S


def nw_lag(n: int, h: int = 1) -> int:
    return int(max(math.ceil(1.5 * h), math.floor(4 * (n / 100.0) ** (2.0 / 9.0))))


def ols_nw(y: np.ndarray, x: np.ndarray, lag: int) -> dict:
    """OLS y = a + b x + e with a Newey-West SE for b. x 1-d."""
    X = _design(x)
    XtXi = np.linalg.inv(X.T @ X)
    beta = XtXi @ (X.T @ y)
    e = y - X @ beta
    V = XtXi @ nw_meat(X * e[:, None], lag) @ XtXi
    se = math.sqrt(max(V[1, 1], 1e-300))
    r2 = 1.0 - float(e @ e) / float(((y - y.mean()) ** 2).sum())
    return {"a": float(beta[0]), "b": float(beta[1]), "se": se, "t": float(beta[1] / se), "r2": r2, "n": int(len(y)), "resid": e}


def ols_nw_multi(y: np.ndarray, X: np.ndarray, lag: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """OLS with intercept already in X; returns beta, NW covariance, residuals."""
    XtXi = np.linalg.inv(X.T @ X)
    beta = XtXi @ (X.T @ y)
    e = y - X @ beta
    return beta, XtXi @ nw_meat(X * e[:, None], lag) @ XtXi, e


def rolling_sum_valid(x: np.ndarray, h: int) -> np.ndarray:
    """out[t] = sum x[t-h+1..t] if all finite, else nan."""
    v = np.isfinite(x)
    c = np.concatenate([[0.0], np.cumsum(np.where(v, x, 0.0))])
    k = np.concatenate([[0], np.cumsum(v.astype(int))])
    out = np.full(len(x), np.nan)
    if len(x) >= h:
        s = c[h:] - c[:-h]
        n = k[h:] - k[:-h]
        out[h - 1:] = np.where(n == h, s, np.nan)
    return out


def forward_sum_valid(r: np.ndarray, h: int) -> np.ndarray:
    """out[t] = sum r[t+1..t+h] if all finite, else nan."""
    back = rolling_sum_valid(r, h)
    out = np.full(len(r), np.nan)
    if len(r) > h:
        out[:-h] = back[h:]
    return out


def hodrick_t(z1: np.ndarray, x: np.ndarray, h: int, mask: np.ndarray) -> float:
    """Hodrick (1992) 1B test of b = 0 in the h-horizon regression, via the reverse regression (Hodrick 1992; Wei-Wright 2013):
    one-period return z1[t] (return over (t, t+1], aligned at origin t) on the rolled-back regressor sum_{i<h} x[t-i], White SE.
    mask selects the origins of the fit segment."""
    xs = rolling_sum_valid(x, h)
    m = mask & np.isfinite(xs) & np.isfinite(z1)
    if m.sum() < 30:
        return float("nan")
    return ols_nw(z1[m], xs[m], 0)["t"]


def stambaugh(z1: np.ndarray, x: np.ndarray, mask: np.ndarray) -> dict:
    """Stambaugh (1999) bias of the one-period slope: E[b_hat - b] = -(s_uv / s_vv)(1 + 3 rho)/T, with AR(1) x."""
    m = mask & np.isfinite(z1) & np.isfinite(x)
    m1 = m[:-1] & np.isfinite(x[1:])
    if m1.sum() < 30:
        return {"rho": float("nan"), "bias1": float("nan"), "b1": float("nan")}
    xl, xn = x[:-1][m1], x[1:][m1]
    ar = ols_nw(xn, xl, 0)
    v = ar["resid"]
    reg = ols_nw(z1[:-1][m1], xl, 0)
    u = reg["resid"]
    rho = ar["b"]
    bias = -(float(np.mean(u * v)) / float(np.mean(v * v))) * (1 + 3 * rho) / len(u)
    return {"rho": float(rho), "bias1": float(bias), "b1": reg["b"], "b1_adj": reg["b"] - bias}


def _fsum(a: np.ndarray, k: int) -> np.ndarray:
    """out[t] = a[t] + ... + a[t+k-1], length len(a) - k + 1."""
    c = np.concatenate([[0.0], np.cumsum(a)])
    return c[k:] - c[:-k]


def ivx_wald(z1: np.ndarray, x: np.ndarray, k: int = 1, beta: float = 0.95, cz: float = 1.0) -> dict:
    """IVX Wald test (Kostakis, Magdalinos and Stamatogiannis 2015, RFS) of b = 0 in y_t = mu + b x_{t-1} + e_t, with the
    mildly integrated instrument z~_t = R z~_{t-1} + dx_t, R = 1 - cz / T^beta, and the FM-corrected variance
    M = Z'Z S_ee - T zbar^2 Omega_FM. Inputs: z1[t] = return over (t, t+1] (aligned at origin t), x[t] known at t, finite and
    consecutive. k = 1: the KMS one-period statistic. k > 1: our long-horizon variant (see below), size-checked by simulation
    (scripts/ws08_run.py size); the published KMS long-horizon variance was not reproduced, so h > 1 cells must pass BOTH."""
    from scipy.signal import lfilter

    y = np.asarray(z1, float)
    x = np.asarray(x, float)
    n = len(x)
    if n < 60:
        return {"W": float("nan"), "p": float("nan"), "A": float("nan")}
    T = n - 1
    R = 1.0 - cz / T ** beta
    zt = np.r_[0.0, lfilter([1.0], [1.0, -R], np.diff(x))]      # zt[t] built from dx up to x[t]
    # one-period pieces for t = 0..n-2: eps[t] (return over (t, t+1]) pairs with u[t] = innovation of x[t+1]
    reg = ols_nw(y[:-1], x[:-1], 0)
    eps = reg["resid"]
    ar = ols_nw(x[1:], x[:-1], 0)
    u = ar["resid"]
    yy, xx, zz = y[:-1], x[:-1], zt[:-1]
    if k == 1:
        ee, eu, uu = float(np.mean(eps * eps)), float(np.mean(eps * u)), float(np.mean(u * u))
        M = int(math.floor(T ** (1 / 3)))
        lam_uu, lam_ue = 0.0, 0.0
        for lg in range(1, M + 1):
            w = 1 - lg / (M + 1)
            lam_uu += w * float(np.mean(u[lg:] * u[:-lg]))
            lam_ue += w * float(np.mean(u[lg:] * eps[:-lg]))
        om_uu = uu + 2 * lam_uu
        om_eu = eu + lam_ue
        om_fm = ee - om_eu ** 2 / om_uu if om_uu > 0 else ee
        yd, xd = yy - yy.mean(), xx - xx.mean()
        sxz = float(xd @ zz)
        A = float(yd @ zz) / sxz
        zbar = float(zz.mean())
        Mk = float(zz @ zz) * ee - len(zz) * zbar ** 2 * om_fm
        if Mk <= 0:
            Mk = float(zz @ zz) * ee
        W = A ** 2 * sxz ** 2 / Mk
        return {"W": float(W), "p": float(sst.chi2.sf(W, 1)), "A": A}
    # long horizon: the k-sum of returns from t on the (unaggregated) instrument z~_t and regressor x_t; null-imposed NW
    # variance (lag 2k). Aggregating z~ forward (z~_{t..t+k-1}) makes it correlated with the overlapping errors (size 73%
    # in our simulation), so it is not done. Simulated size (T = 2000, rho 0.99, corr -0.9): 4.7% / 8.7% / 10.7% at k = 1/7/30.
    yk = _fsum(y, k)
    m_ = len(yk)
    xk, zk = x[:m_], zt[:m_]
    yd, xd = yk - yk.mean(), xk - xk.mean()
    sxz = float(xd @ zk)
    A = float(yd @ zk) / sxz
    sc = ((zk - zk.mean()) * yd)[:, None]
    V = nw_meat(sc, 2 * k)[0, 0] / sxz ** 2
    W = A ** 2 / V
    return {"W": float(W), "p": float(sst.chi2.sf(W, 1)), "A": A}


def block_counts(n_days: int, block: int, B: int, rng: np.random.Generator) -> np.ndarray:
    """Moving-block bootstrap over day indices, returned as a (B, n_days) matrix of draw counts."""
    nb = int(math.ceil(n_days / block))
    K = np.zeros((B, n_days), dtype=np.float64)
    starts = rng.integers(0, max(1, n_days - block + 1), size=(B, nb))
    offs = np.arange(block)
    for b in range(B):
        idx = (starts[b][:, None] + offs[None, :]).ravel()[:n_days]
        np.add.at(K[b], idx, 1.0)
    return K


def slopes_from_stats(S: np.ndarray) -> np.ndarray:
    """S[..., 5] = (n, sx, sy, sxy, sxx) -> OLS slope."""
    n, sx, sy, sxy, sxx = (S[..., i] for i in range(5))
    den = n * sxx - sx * sx
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(den > 0, (n * sxy - sx * sy) / den, np.nan)


def romano_wolf(t_obs: np.ndarray, t_boot_centered: np.ndarray) -> np.ndarray:
    """Romano-Wolf (2005) step-down adjusted p-values for two-sided tests. t_obs (C,) = |b_hat| / sd*, t_boot_centered (B, C) =
    |b* - b_hat| / sd*. Returns adjusted p (C,)."""
    C = len(t_obs)
    order = np.argsort(-t_obs)
    p_adj = np.ones(C)
    prev = 0.0
    for j, c in enumerate(order):
        rem = order[j:]
        mx = np.nanmax(t_boot_centered[:, rem], axis=1)
        p = float(np.mean(mx >= t_obs[c]))
        prev = max(prev, p)
        p_adj[c] = prev
    return p_adj


def clark_west(y: np.ndarray, f_bench: np.ndarray, f_model: np.ndarray, h: int) -> dict:
    e1, e2 = y - f_bench, y - f_model
    fadj = e1 ** 2 - (e2 ** 2 - (f_bench - f_model) ** 2)
    n = len(y)
    if n < 30:
        return {"t": float("nan"), "p": float("nan"), "oos_r2": float("nan"), "n": n}
    m = float(fadj.mean())
    u = (fadj - m)[:, None]
    se = math.sqrt(nw_meat(u, max(h, 1))[0, 0]) / n
    t = m / se if se > 0 else float("nan")
    return {"t": t, "p": float(sst.norm.sf(t)), "oos_r2": 1 - float(e2 @ e2) / float(e1 @ e1), "n": n}


def giacomini_white(d: np.ndarray, h: int) -> dict:
    """Conditional GW test on loss differences d_t = L(bench) - L(model) (positive = model better), instruments (1, d_{t-h})."""
    if len(d) < 50:
        return {"p_uncond": float("nan"), "t_uncond": float("nan"), "p_cond": float("nan"), "mean_d": float("nan")}
    n = len(d)
    m = float(d.mean())
    se = math.sqrt(nw_meat((d - m)[:, None], max(h, 1))[0, 0]) / n
    Z = np.column_stack([d[h:], d[h:] * d[:-h]])
    zb = Z.mean(axis=0)
    Om = nw_meat(Z - zb, max(h - 1, 0)) / len(Z)
    try:
        stat = float(len(Z) * zb @ np.linalg.solve(Om, zb))
    except np.linalg.LinAlgError:
        stat = float("nan")
    t = m / se if se > 0 else float("nan")
    return {"t_uncond": t, "p_uncond": float(2 * sst.norm.sf(abs(t))), "p_cond": float(sst.chi2.sf(stat, 2)), "mean_d": m}


def dm_test(d: np.ndarray, h: int = 1) -> dict:
    """Diebold-Mariano on loss differences d (model - benchmark; negative = model better), NW lag h, HLN small-sample factor."""
    n = len(d)
    if n < 20:
        return {"t": float("nan"), "p": float("nan"), "n": n, "mean": float("nan")}
    m = float(d.mean())
    se = math.sqrt(nw_meat((d - m)[:, None], max(h, 1))[0, 0]) / n
    hln = math.sqrt(max((n + 1 - 2 * h + h * (h - 1) / n) / n, 1e-9))
    t = m / se * hln if se > 0 else float("nan")
    return {"t": t, "p": float(2 * sst.t.sf(abs(t), n - 1)), "n": n, "mean": m}


def mcs_tmax(L: np.ndarray, names: list[str], alpha: float = 0.10, block: int = 30, B: int = 999, seed: int = 8) -> list[str]:
    """Hansen-Lunde-Nason Model Confidence Set with the T_max statistic and a moving-block bootstrap. L (n, m) losses."""
    rng = np.random.default_rng(seed)
    n = L.shape[0]
    alive = list(range(L.shape[1]))
    nb = int(math.ceil(n / block))
    idx_all = [(rng.integers(0, n - block + 1, size=nb)[:, None] + np.arange(block)).ravel()[:n] for _ in range(B)]
    while len(alive) > 1:
        Ls = L[:, alive]
        dbar_i = Ls.mean(axis=0) - Ls.mean()          # relative performance vs the average
        boots = np.array([Ls[ix].mean(axis=0) - Ls[ix].mean() for ix in idx_all])
        var = ((boots - dbar_i) ** 2).mean(axis=0)
        tstat = dbar_i / np.sqrt(var)
        tmax = tstat.max()
        tb = ((boots - dbar_i) / np.sqrt(var)).max(axis=1)
        p = float(np.mean(tb >= tmax))
        if p >= alpha:
            break
        alive.pop(int(np.argmax(tstat)))
    return [names[i] for i in alive]


# ====================================================================== data assembly

def _ms(col: pl.Series) -> np.ndarray:
    s = col
    if s.dtype == pl.Datetime or isinstance(s.dtype, pl.Datetime):
        return s.dt.epoch("ms").to_numpy().astype(np.int64)
    return s.to_numpy().astype(np.int64)


def asof(k_ts: np.ndarray, vals: np.ndarray, origins: np.ndarray, max_age_ms: int | None = None) -> np.ndarray:
    """Latest value with known_ts <= origin (k_ts sorted)."""
    i = np.searchsorted(k_ts, origins, side="right") - 1
    out = np.where(i >= 0, vals[np.clip(i, 0, None)], np.nan).astype(float)
    if max_age_ms is not None:
        age = origins - np.where(i >= 0, k_ts[np.clip(i, 0, None)], 0)
        out = np.where((i >= 0) & (age <= max_age_ms), out, np.nan)
    return out


def window_sum(k_ts: np.ndarray, vals: np.ndarray, origins: np.ndarray, win_ms: int) -> tuple[np.ndarray, np.ndarray]:
    """Sum and count of values with known_ts in (origin - win, origin]."""
    cs = np.concatenate([[0.0], np.cumsum(vals)])
    hi = np.searchsorted(k_ts, origins, side="right")
    lo = np.searchsorted(k_ts, origins - win_ms, side="right")
    return cs[hi] - cs[lo], (hi - lo).astype(float)



def ewma_ahead(r: np.ndarray, lam: float = 0.94, warmup: int = 30) -> np.ndarray:
    """sigma_{t|t-1} like features.transforms.ewma_sigma_ahead, but started after the first `warmup` FINITE returns (series
    that begin with missing values would otherwise stay nan for ever)."""
    r = np.asarray(r, float)
    out = np.full(len(r), np.nan)
    idx = np.flatnonzero(np.isfinite(r))
    if len(idx) <= warmup:
        return out
    var = float(np.mean(r[idx[:warmup]] ** 2))
    start = int(idx[warmup])
    out[start] = math.sqrt(var)
    for t in range(start, len(r) - 1):
        if np.isfinite(r[t]):
            var = lam * var + (1 - lam) * r[t] ** 2
        out[t + 1] = math.sqrt(var)
    return out


def load_hourly(lake, as_of: datetime, symbols: list[str], log=print) -> dict:
    """Hourly grid of origins (bar close times) with 1h log returns and EWMA sigma forecasts per symbol."""
    from ai_trader.core.config import load

    rs = load().settings["research"]
    per = {}
    for s in symbols:
        b = lake.bars(s, "1h", as_of).select("event_ts", "close_ts", "open", "high", "low", "close")
        b = b.filter((pl.col("open") > 0) & (pl.col("low") > 0) & (pl.col("high") >= pl.max_horizontal("open", "close"))
                     & (pl.col("low") <= pl.min_horizontal("open", "close"))).sort("event_ts")
        if b.height < 500:
            continue
        first = b["event_ts"].min()
        days = rs["burn_in_days_new"] if first.year >= rs["burn_in_new_listing_year"] else rs["burn_in_days"]
        b = b.filter(pl.col("event_ts") >= first + timedelta(days=days))
        per[s] = (_ms(b["close_ts"]), b["close"].to_numpy())
    t0 = min(v[0][0] for v in per.values())
    t1 = max(v[0][-1] for v in per.values())
    grid = np.arange(t0, t1 + 1, H_MS, dtype=np.int64)
    R = np.full((len(grid), len(per)), np.nan)
    names = list(per)
    for j, s in enumerate(names):
        ts, c = per[s]
        idx = ((ts - t0) // H_MS).astype(int)
        lc = np.full(len(grid), np.nan)
        lc[idx] = np.log(c)
        R[1:, j] = np.diff(lc)
    R[np.abs(R) > 0.5] = np.nan       # impossible 1h moves (data errors)
    sig = np.full_like(R, np.nan)
    for j in range(R.shape[1]):
        sp = ewma_ahead(R[:, j], lam=0.94, warmup=30)
        # sigma for the NEXT bar known at the close of t: one more EWMA step with r_t
        rt = np.nan_to_num(R[:, j])
        sig[:, j] = np.sqrt(0.94 * sp ** 2 + 0.06 * rt ** 2)
    log(f"hourly grid {len(grid)} rows x {len(names)} symbols")
    return {"grid": grid, "R": R, "sig_next": sig, "names": names}


def basket(R: np.ndarray, cols: list[int], min_n: int = 10) -> np.ndarray:
    sub = R[:, cols]
    n = np.isfinite(sub).sum(axis=1)
    with np.errstate(invalid="ignore"):
        m = np.nanmean(sub, axis=1)
    return np.where(n >= min_n, m, np.nan)


def sigma_next(r: np.ndarray) -> np.ndarray:
    sp = ewma_ahead(r, lam=0.94, warmup=30)
    return np.sqrt(0.94 * sp ** 2 + 0.06 * np.nan_to_num(r) ** 2)


def daily_from_hourly(grid: np.ndarray, r: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Daily origins (00:00 UTC) and the return of the UTC day starting at each origin (24 complete hours)."""
    d0 = (grid[0] // D_MS + 1) * D_MS
    days = np.arange(d0, grid[-1] - D_MS + 1, D_MS, dtype=np.int64)
    # hourly r[i] is the return of the bar ending at grid[i]; day starting at D uses bars ending D+1h .. D+24h
    pos = ((days - grid[0]) // H_MS).astype(int)
    back = rolling_sum_valid(r, 24)
    end = pos + 24
    ok = end < len(r)
    out = np.full(len(days), np.nan)
    out[ok] = back[end[ok]]
    return days, out


def daily_rv(grid: np.ndarray, r: np.ndarray, days: np.ndarray) -> np.ndarray:
    pos = ((days - grid[0]) // H_MS).astype(int)
    back = rolling_sum_valid(r ** 2, 24)
    end = pos + 24
    out = np.full(len(days), np.nan)
    ok = end < len(r)
    out[ok] = back[end[ok]]
    return out


def read_series(lake, as_of, table: str, **kw) -> pl.DataFrame:
    df = lake.read_asof(table, as_of, **kw)
    return df


def alt_series(lake, as_of, source: str, series_id: str, entity: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    d = lake.read_asof("alt_series", as_of, filters={"source": source, "series_id": series_id, "entity": entity})
    d = d.filter(pl.col("value").is_not_null()).sort("known_ts")
    return _ms(d["known_ts"]), d["value"].to_numpy().astype(float), _ms(d["event_ts"])


def macro_series(name: str) -> tuple[np.ndarray, np.ndarray]:
    """data/research/macro/<name>.parquet (FRED graph CSV, obs_date + 1 day 00:00 UTC known)."""
    d = pl.read_parquet(ROOT / "data" / "research" / "macro" / f"{name}.parquet").filter(pl.col("value").is_not_null())
    d = d.sort("effective_ts").unique("obs_date", keep="last").sort("effective_ts")
    return _ms(d["effective_ts"]), d["value"].to_numpy().astype(float)


def pct_trailing(x: np.ndarray, win: int, min_n: int) -> np.ndarray:
    out = np.full(len(x), np.nan)
    for i in range(len(x)):
        if not np.isfinite(x[i]):
            continue
        w = x[max(0, i - win + 1): i + 1]
        w = w[np.isfinite(w)]
        if len(w) >= min_n:
            out[i] = float(np.mean(w <= x[i]))
    return out


def z_trailing(x: np.ndarray, win: int, min_n: int) -> np.ndarray:
    out = np.full(len(x), np.nan)
    for i in range(len(x)):
        w = x[max(0, i - win): i]
        w = w[np.isfinite(w)]
        if len(w) >= min_n and np.isfinite(x[i]):
            sd = w.std()
            out[i] = (x[i] - w.mean()) / sd if sd > 0 else np.nan
    return out


def build(as_of: datetime | None = None, log=print) -> dict:
    """Assemble the hourly and daily panels (targets + predictors). Saves data/research/ws08/panel_{hourly,daily}.parquet."""
    from ai_trader.core.time import now
    from ai_trader.storage.lake import Lake

    as_of = as_of or now()
    lake = Lake()
    H = load_hourly(lake, as_of, UNIVERSE_L, log)
    grid, R, SN, names = H["grid"], H["R"], H["sig_next"], H["names"]
    col = {s: i for i, s in enumerate(names)}
    alt_cols = [col[s] for s in ALTS if s in col]
    r_t = {"BTC": R[:, col["BTCUSDT"]], "ETH": R[:, col["ETHUSDT"]], "ALT": basket(R, alt_cols)}
    sig_t = {"BTC": SN[:, col["BTCUSDT"]], "ETH": SN[:, col["ETHUSDT"]], "ALT": sigma_next(r_t["ALT"])}

    # ---------------- raw predictor inputs
    fund = {}
    for s in UNIVERSE_L:
        f = lake.read_asof("funding", as_of, symbols=[s], filters={"venue": "bybit"}).sort("known_ts")
        if f.height:
            fund[s] = (_ms(f["known_ts"]), f["rate"].to_numpy().astype(float))
    oi = {}
    for s in UNIVERSE_L:
        o = lake.read_asof("open_interest", as_of, symbols=[s], filters={"venue": "bybit", "interval": "4h"}).sort("known_ts")
        o = o.filter(pl.col("open_interest") > 0)
        if o.height:
            oi[s] = (_ms(o["known_ts"]), np.log(o["open_interest"].to_numpy().astype(float)))
    lsr = {}
    for s in UNIVERSE_L:
        q = lake.read_asof("ls_ratio", as_of, symbols=[s], filters={"venue": "bybit", "period": "4h"}).sort("known_ts")
        q = q.filter((pl.col("buy_ratio") > 0) & (pl.col("sell_ratio") > 0))
        if q.height:
            lsr[s] = (_ms(q["known_ts"]), np.log(q["buy_ratio"].to_numpy() / q["sell_ratio"].to_numpy()))
    basis = {}
    for s, a in (("BTCUSDT", "BTC"), ("ETHUSDT", "ETH")):
        ix = lake.bars(s, "1h", as_of, price_type="index").select("close_ts", pl.col("close").alias("ic"))
        la = lake.bars(s, "1h", as_of).select("close_ts", "close")
        j = la.join(ix, on="close_ts", how="inner").filter((pl.col("ic") > 0) & (pl.col("close") > 0)).sort("close_ts")
        basis[a] = (_ms(j["close_ts"]), np.log(j["close"].to_numpy() / j["ic"].to_numpy()))
    dv = {a: alt_series(lake, as_of, "deribit", "dvol_close", a) for a in ("BTC", "ETH")}
    sk = {a: alt_series(lake, as_of, "deribit_svi", "iv_rr25_30", a) for a in ("BTC", "ETH")}
    tb = {a: alt_series(lake, as_of, "deribit_svi", "iv_basis_ann_30", a) for a in ("BTC", "ETH")}
    st = alt_series(lake, as_of, "defillama", "stablecoins_total_usd", "all")
    fg = alt_series(lake, as_of, "alternative_me", "fear_greed", "crypto")
    prem_path = OUT / "premium_1h.parquet"
    prem = {}
    if prem_path.exists():
        p = pl.read_parquet(prem_path).sort("ts")
        k = _ms(p["known_ts"])
        with np.errstate(divide="ignore", invalid="ignore"):
            usdt = np.log(p["c_USDT_USD"].to_numpy().astype(float))
            for a in ("BTC", "ETH"):
                raw = np.log(p[f"c_{a}_USD"].to_numpy().astype(float) / p[f"c_{a}USDT"].to_numpy().astype(float))
                raw = np.where(np.abs(raw) < 0.05, raw, np.nan)       # drop broken prints (> 5%)
                prem[a] = (k, raw, raw - usdt)

    def asset_inputs(a: str):
        return ("BTC" if a == "ALT" else a)

    # ---------------- hourly predictors (origins = grid)
    o = grid
    hp: dict[str, dict[str, np.ndarray]] = {}
    for a in ("BTC", "ETH", "ALT"):
        syms = ALTS if a == "ALT" else [a + "USDT"]
        d: dict[str, np.ndarray] = {}
        fs = []
        for s in syms:
            if s in fund:
                sm, cnt = window_sum(*fund[s], o, D_MS)
                fs.append(np.where(cnt > 0, sm, np.nan))
        d["fund"] = np.nanmean(np.vstack(fs), axis=0) if fs else np.full(len(o), np.nan)
        ds, ls = [], []
        for s in syms:
            if s in oi:
                k, v = oi[s]
                ds.append(asof(k, v, o, 8 * H_MS) - asof(k, v, o - 4 * H_MS, 8 * H_MS))
            if s in lsr:
                ls.append(asof(*lsr[s], o, 8 * H_MS))
        with np.errstate(invalid="ignore"), np.testing.suppress_warnings() as sup:
            sup.filter(RuntimeWarning)
            d["doi"] = np.nanmean(np.vstack(ds), axis=0) if ds else np.full(len(o), np.nan)
            d["ls"] = np.nanmean(np.vstack(ls), axis=0) if ls else np.full(len(o), np.nan)
        b = asset_inputs(a)
        if a != "ALT":
            d["basis"] = asof(*basis[a], o, 2 * H_MS)
        kd, vd, _ = dv[b]
        d["dvol"] = asof(kd, vd, o, 3 * H_MS)
        d["ddvol"] = np.log(d["dvol"] / asof(kd, vd, o - 4 * H_MS, 3 * H_MS))
        if b in prem:
            k, raw, adj = prem[b]
            d["cbp_raw"] = asof(k, raw, o, 2 * H_MS)
            d["cbp_adj"] = asof(k, adj, o, 2 * H_MS)
        hp[a] = d

    hourly = {"grid": grid, "r": r_t, "sig": sig_t, "x": hp}

    # ---------------- daily panel
    days = None
    dr, dsig = {}, {}
    for a in ("BTC", "ETH", "ALT"):
        days, dr[a] = daily_from_hourly(grid, r_t[a])
        dsig[a] = sigma_next_daily(dr[a])
    od = days
    dp: dict[str, dict[str, np.ndarray]] = {}
    nd = len(od)
    for a in ("BTC", "ETH", "ALT"):
        syms = ALTS if a == "ALT" else [a + "USDT"]
        d = {}
        fs, ds, ls = [], [], []
        for s in syms:
            if s in fund:
                sm, cnt = window_sum(*fund[s], od, D_MS)
                fs.append(np.where(cnt > 0, sm, np.nan))
            if s in oi:
                k, v = oi[s]
                ds.append(asof(k, v, od, 8 * H_MS) - asof(k, v, od - D_MS, 8 * H_MS))
            if s in lsr:
                ls.append(asof(*lsr[s], od, 8 * H_MS))
        with np.testing.suppress_warnings() as sup:
            sup.filter(RuntimeWarning)
            d["fund"] = np.nanmean(np.vstack(fs), axis=0) if fs else np.full(nd, np.nan)
            d["doi"] = np.nanmean(np.vstack(ds), axis=0) if ds else np.full(nd, np.nan)
            d["ls"] = np.nanmean(np.vstack(ls), axis=0) if ls else np.full(nd, np.nan)
        d["fund_pct"] = pct_trailing(d["fund"], 365, 180)
        b = asset_inputs(a)
        if a != "ALT":
            kb, vb = basis[a]
            sm, cnt = window_sum(kb, vb, od, D_MS)
            d["basis"] = np.where(cnt >= 18, sm / np.maximum(cnt, 1), np.nan)
        kt, vt, _ = tb[b]
        d["tbasis"] = asof(kt, vt, od, 3 * D_MS)
        kd, vd, _ = dv[b]
        d["dvol"] = asof(kd, vd, od, 3 * H_MS)
        d["ddvol"] = np.log(d["dvol"] / asof(kd, vd, od - D_MS, 3 * H_MS))
        # VRP: implied (DVOL/100)^2 minus trailing 30d realised variance (annualised) of the asset's own hourly returns
        rv30 = rolling_sum_valid(r_t[b] ** 2, 720)
        posd = ((od - grid[0]) // H_MS).astype(int)
        okp = (posd >= 0) & (posd < len(grid))
        rv_at = np.full(nd, np.nan)
        rv_at[okp] = rv30[posd[okp]]
        d["vrp"] = (d["dvol"] / 100.0) ** 2 - rv_at * 365.0 / 30.0
        ks, vs, _ = sk[b]
        d["skew"] = asof(ks, vs, od, 3 * D_MS)
        if b in prem:
            k, raw, adj = prem[b]
            for nm, v in (("cbp_raw", raw), ("cbp_adj", adj)):
                fin = np.isfinite(v)
                sm, cnt = window_sum(k[fin], v[fin], od, D_MS)
                d[nm] = np.where(cnt >= 18, sm / np.maximum(cnt, 1), np.nan)
        kst, vst, _ = st
        s_now = asof(kst, vst, od, 4 * D_MS)
        s_30 = asof(kst, vst, od - 30 * D_MS, 4 * D_MS)
        d["stable"] = z_trailing(np.log(s_now / s_30), 365, 300)
        for nm, ser, kind in (("ry", "DFII10", "diff5"), ("usd", "DTWEXBGS", "ldiff5"), ("ndx", "NASDAQ100", "ldiff1")):
            km, vm = macro_series(ser)
            i = np.searchsorted(km, od, side="right") - 1
            lagn = 5 if kind.endswith("5") else 1
            ok = (i - lagn >= 0) & (od - km[np.clip(i, 0, None)] <= 5 * D_MS)
            cur = np.where(ok, vm[np.clip(i, 0, None)], np.nan)
            prv = np.where(ok, vm[np.clip(i - lagn, 0, None)], np.nan)
            d[nm] = cur - prv if kind.startswith("diff") else np.log(cur / prv)
        kf, vf, _ = fg
        d["fng"] = asof(kf, vf, od, 2 * D_MS)
        dp[a] = d
    daily = {"grid": od, "r": dr, "sig": dsig, "x": dp, "rv": {a: daily_rv(grid, r_t[a], od) for a in ("BTC", "ETH")}}
    log(f"daily grid {nd} days")
    return {"hourly": hourly, "daily": daily}


def sigma_next_daily(R: np.ndarray) -> np.ndarray:
    """sigma forecast for day d made at its 00:00 origin (EWMA 0.94 on previous daily returns)."""
    return ewma_ahead(R, lam=0.94, warmup=30)


# ====================================================================== grid analysis

HOURLY_PRED = ["fund", "basis", "doi", "ls", "cbp_raw", "cbp_adj", "dvol", "ddvol"]
DAILY_PRED = ["fund", "fund_pct", "basis", "tbasis", "doi", "ls", "cbp_raw", "cbp_adj", "stable", "dvol", "ddvol", "vrp", "skew",
              "ry", "usd", "ndx", "fng"]
SIGN = {"fund": -1, "fund_pct": -1, "basis": -1, "tbasis": -1, "doi": 0, "ls": -1, "cbp_raw": 1, "cbp_adj": 1, "stable": 1,
        "dvol": 1, "ddvol": 0, "vrp": 1, "skew": 0, "ry": -1, "usd": -1, "ndx": 1, "fng": -1}
HYP = {"fund": "H-WS08-2", "fund_pct": "H-WS08-2", "basis": "H-WS08-3", "tbasis": "H-WS08-3", "doi": "H-WS08-4", "ls": "H-WS08-5",
       "cbp_raw": "H-WS08-6", "cbp_adj": "H-WS08-6", "stable": "H-WS08-7", "dvol": "H-WS08-8", "ddvol": "H-WS08-8", "vrp": "H-WS08-8",
       "skew": "H-WS08-9", "ry": "H-WS08-10", "usd": "H-WS08-10", "ndx": "H-WS08-10", "fng": "H-WS08-11"}


def targets(panel: dict, h: int) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """y[t] = sum r over (t, t+h] / (sigma_t sqrt h) and the raw forward return, per asset. For the hourly panel sigma_t =
    sig[t] (next-bar sigma known at t); for the daily panel r[d] is the day starting at origin d, so the forward sum starts at d."""
    out = {}
    for a, r in panel["r"].items():
        sig = panel["sig"][a]
        if "rv" in panel:     # daily: r[d] belongs to origin d
            raw = rolling_sum_valid(r[::-1], h)[::-1]
            raw = np.concatenate([raw[: len(r) - h + 1], np.full(h - 1, np.nan)]) if h > 1 else raw
        else:
            raw = forward_sum_valid(r, h)
        out[a] = (raw / (sig * math.sqrt(h)), raw)
    return out


def cell_list() -> list[tuple[str, str, str, int]]:
    cells = []
    for freq, preds, hs in (("H", HOURLY_PRED, (1, 4)), ("D", DAILY_PRED, (1, 7, 30))):
        for p in preds:
            for a in ("BTC", "ETH", "ALT"):
                if p == "basis" and a == "ALT":
                    continue
                for h in hs:
                    cells.append((freq, p, a, h))
    return cells


def analyse_grid(P: dict, B: int = 999, block_days: int = 60, seed: int = 8, log=print) -> dict:
    """Per-cell inference + Romano-Wolf over all cells (in-sample), holdout replication, OOS CW / GW."""
    rng = np.random.default_rng(seed)
    co = cutoff_ms()
    cells = cell_list()
    tg = {("H", h): targets(P["hourly"], h) for h in (1, 4)}
    tg.update({("D", h): targets(P["daily"], h) for h in (1, 7, 30)})
    day0 = int(P["hourly"]["grid"][0] // D_MS)
    n_days = int(co // D_MS) - day0 + 1
    S = np.zeros((n_days, len(cells), 5))
    res = []
    for ci, (freq, p, a, h) in enumerate(cells):
        pan = P["hourly"] if freq == "H" else P["daily"]
        g = pan["grid"]
        hstep = h * (H_MS if freq == "H" else D_MS)
        x = pan["x"][a].get(p)
        if x is None:
            res.append({"cell": f"{freq}:{p}:{a}:{h}", "skip": "no data"})
            continue
        y, raw = tg[(freq, h)][a]
        z1 = tg[(freq, 1)][a][0]
        fin = np.isfinite(x) & np.isfinite(y)
        is_m = fin & (g + hstep <= co)
        ho_m = fin & (g >= co)
        rec = {"cell": f"{freq}:{p}:{a}:{h}", "freq": freq, "pred": p, "asset": a, "h": h, "hyp": HYP[p], "sign_reg": SIGN[p]}
        if is_m.sum() < 200:
            rec["skip"] = f"n_is {int(is_m.sum())}"
            res.append(rec)
            continue
        mu, sd = float(x[is_m].mean()), float(x[is_m].std())
        xs = (x - mu) / sd
        base_per = 24 if freq == "H" else 1     # obs per day
        lag = nw_lag(int(is_m.sum()), h)
        f = ols_nw(y[is_m], xs[is_m], lag)
        rec.update({"n_is": f["n"], "n_eff_is": f["n"] / h, "b": f["b"], "t_nw": f["t"], "r2": f["r2"],
                    "first": datetime.fromtimestamp(g[is_m][0] / 1000, timezone.utc).date().isoformat()})
        unit = H_MS if freq == "H" else D_MS
        rec["t_hodrick"] = hodrick_t(z1, xs, h, g + unit <= co)
        stb = stambaugh(z1, xs, np.isfinite(xs) & (g + (H_MS if freq == "H" else D_MS) <= co))
        rho = stb["rho"]
        geo = sum(rho ** j for j in range(h)) / math.sqrt(h) if np.isfinite(rho) else float("nan")
        rec.update({"rho": rho, "stambaugh_bias_h": stb["bias1"] * geo if np.isfinite(geo) else float("nan")})
        rec["b_stambaugh"] = f["b"] - rec["stambaugh_bias_h"]
        # IVX on the one-period base series, consecutive in-sample rows only
        mm = np.isfinite(xs) & np.isfinite(z1) & (g + (H_MS if freq == "H" else D_MS) <= co)
        iv = ivx_wald(z1[mm], xs[mm], k=h)
        iv1 = iv if h == 1 else ivx_wald(z1[mm], xs[mm], k=1)
        rec.update({"ivx_W": iv["W"], "ivx_p": iv["p"], "ivx_A": iv["A"], "ivx_p1": iv1["p"], "ivx_A1": iv1["A"]})
        # bootstrap sufficient statistics by day
        di = (g[is_m] // D_MS - day0).astype(int)
        xx, yy = xs[is_m], y[is_m]
        for k_, v_ in enumerate((np.ones_like(xx), xx, yy, xx * yy, xx * xx)):
            np.add.at(S[:, ci, k_], di, v_)
        # holdout
        if ho_m.sum() >= 10:
            fh = ols_nw(y[ho_m], xs[ho_m], nw_lag(int(ho_m.sum()), h))
            pred = f["a"] + f["b"] * xs[ho_m]
            yh = y[ho_m]
            rec.update({"n_ho": fh["n"], "n_eff_ho": fh["n"] / h, "b_ho": fh["b"], "t_ho": fh["t"],
                        "oos_r2_fixed_vs_ismean": 1 - float(((yh - pred) ** 2).sum()) / float(((yh - y[is_m].mean()) ** 2).sum()),
                        "oos_r2_fixed_vs_zero": 1 - float(((yh - pred) ** 2).sum()) / float((yh ** 2).sum())})
        # out of sample (expanding purged, CW; rolling GW)
        rec.update(oos_cell(g, xs, y, h, hstep, co, base_per))
        # median of x (in-sample) for the sign rule
        rec["x_median_is"] = float(np.median(xs[is_m]))
        res.append(rec)
    ok = [i for i, r in enumerate(res) if "b" in r]
    log(f"grid: {len(ok)} cells fitted of {len(cells)}")
    K = block_counts(n_days, block_days, B, rng)
    Sb = (K @ S[:, ok, :].reshape(n_days, -1)).reshape(B, len(ok), 5)
    bstar = slopes_from_stats(Sb)
    bhat = slopes_from_stats(S[:, ok, :].sum(axis=0))
    sdb = np.nanstd(bstar, axis=0)
    t_obs = np.abs(bhat) / sdb
    t_bc = np.abs(bstar - bhat) / sdb
    p_rw = romano_wolf(t_obs, t_bc)
    p_ind = np.mean(t_bc >= t_obs, axis=0)
    c_rw = float(np.quantile(np.nanmax(t_bc, axis=1), 0.95))
    for j, i in enumerate(ok):
        r = res[i]
        r.update({"t_boot": float(np.sign(bhat[j]) * t_obs[j]), "p_boot": float(p_ind[j]), "p_rw": float(p_rw[j])})
        te = r["n_is"] / r["h"]
        r["mde_corr_single"] = (1.96 + 0.84) / math.sqrt(te)
        r["mde_corr_rw"] = (c_rw + 0.84) / math.sqrt(te)
    return {"cells": res, "c_rw95": c_rw, "B": B, "block_days": block_days, "n_cells": len(ok)}


def oos_cell(g, xs, y, h, hstep, co, base_per) -> dict:
    """Expanding purged OLS vs prevailing mean (Clark-West), rolling-window conditional GW; split by origin segment."""
    n = len(g)
    v = np.isfinite(xs) & np.isfinite(y)
    stats_ = np.vstack([v, np.where(v, xs, 0), np.where(v, y, 0), np.where(v, xs * y, 0), np.where(v, xs * xs, 0)]).T.astype(float)
    cs = np.vstack([np.zeros(5), np.cumsum(stats_, axis=0)])
    idx = np.arange(n)
    # usable pairs at origin i: s <= i - h  (target closed by i)
    hi = np.clip(idx - h + 1, 0, n)          # cs index: rows < hi -> rows <= i - h
    min_train = 365 * base_per
    W = (365 if base_per == 1 else 90) * base_per
    out = {}
    Se = cs[hi]
    lo = np.clip(hi - W, 0, n)
    Sr = cs[hi] - cs[lo]
    for nm, SS in (("exp", Se), ("roll", Sr)):
        cnt = SS[:, 0]
        b = slopes_from_stats(SS)
        a = (SS[:, 2] - b * SS[:, 1]) / np.maximum(cnt, 1)
        fm = a + b * xs
        fb = SS[:, 2] / np.maximum(cnt, 1)
        ok = v & (cnt >= (min_train if nm == "exp" else W * 0.8)) & np.isfinite(fm)
        for seg, segm in (("is", g + hstep <= co), ("ho", g >= co)):
            m = ok & segm
            if nm == "exp":
                cw = clark_west(y[m], fb[m], fm[m], h)
                out[f"cw_t_{seg}"], out[f"cw_p_{seg}"], out[f"oos_r2_{seg}"], out[f"n_oos_{seg}"] = cw["t"], cw["p"], cw["oos_r2"], cw["n"]
                if m.sum() > 0:
                    out[f"oos_r2_zero_{seg}"] = 1 - float(((y[m] - fm[m]) ** 2).sum()) / float((y[m] ** 2).sum())
            else:
                d = (y[m] - fb[m]) ** 2 - (y[m] - fm[m]) ** 2
                gw = giacomini_white(d, h)
                out[f"gw_p_cond_{seg}"], out[f"gw_t_{seg}"] = gw["p_cond"], gw["t_uncond"]
    return out


def holm(p: list[float]) -> list[float]:
    p = np.asarray(p, float)
    o = np.argsort(p)
    m = len(p)
    adj = np.empty(m)
    run = 0.0
    for k, i in enumerate(o):
        run = max(run, min(1.0, (m - k) * p[i]))
        adj[i] = run
    return adj.tolist()


# ====================================================================== economics of a sign rule

def sign_rule(P: dict, rec: dict, block_days: int = 60, B: int = 999, seed: int = 8) -> dict:
    """Non-overlapping sign rule pos = sign(b_IS) * sign(x - median_IS), held h; net of costs in sigma_h units and bp."""
    freq, p, a, h = rec["freq"], rec["pred"], rec["asset"], rec["h"]
    pan = P["hourly"] if freq == "H" else P["daily"]
    g = pan["grid"]
    unit = H_MS if freq == "H" else D_MS
    y, raw = targets(pan, h)[a]
    x = pan["x"][a][p]
    sig = pan["sig"][a]
    co = cutoff_ms()
    fin = np.isfinite(x) & np.isfinite(y) & np.isfinite(sig)
    is_m = fin & (g + h * unit <= co)
    mu, sd = float(x[is_m].mean()), float(x[is_m].std())
    xs = (x - mu) / sd
    pos = np.sign(rec["b"]) * np.sign(xs - rec["x_median_is"])
    out = {}
    for seg, segm in (("is", is_m), ("ho", fin & (g >= co))):
        idx = np.flatnonzero(segm)
        if len(idx) == 0:
            continue
        keep, last = [], -10 ** 9
        for i in idx:                   # non-overlapping trades
            if g[i] >= last + h * unit:
                keep.append(i)
                last = g[i]
        keep = np.asarray(keep)
        gross = pos[keep] * y[keep]
        sig_h = sig[keep] * math.sqrt(h)
        res = {"n_trades": int(len(keep)), "gross_sigma": float(gross.mean()), "gross_bp": float((pos[keep] * raw[keep]).mean() * 1e4),
               "median_sigma_h_bp": float(np.median(sig_h) * 1e4)}
        for cname, c in (("taker", COST_TAKER_RT_BP), ("maker_entry", COST_MAKER_ENTRY_RT_BP)):
            net = gross - (c / 1e4) / sig_h
            res[f"net_sigma_{cname}"] = float(net.mean())
            res[f"net_bp_{cname}"] = res["gross_bp"] - c
            # day-block bootstrap CI of the net mean (blocks of trades by day)
            if cname == "taker" and len(net) >= 20:
                rng = np.random.default_rng(seed)
                dd = (g[keep] // D_MS)
                dd = dd - dd.min()
                nd = int(dd.max()) + 1
                sums = np.bincount(dd, weights=net, minlength=nd)
                cnts = np.bincount(dd, minlength=nd).astype(float)
                K = block_counts(nd, min(block_days, nd), B, rng)
                mb = (K @ sums) / np.maximum(K @ cnts, 1)
                res["net_sigma_taker_ci"] = [float(np.quantile(mb, 0.025)), float(np.quantile(mb, 0.975))]
                res["sharpe_per_trade"] = float(net.mean() / net.std()) if net.std() > 0 else float("nan")
                res["skew"], res["kurt"] = float(sst.skew(net)), float(sst.kurtosis(net, fisher=False))
        out[seg] = res
    return out


def deflated_sharpe(sr: float, n: int, skew: float, kurt: float, sr_trials: np.ndarray) -> float:
    """Bailey-Lopez de Prado deflated Sharpe (per-trade SR) given the cross-trial SR variance."""
    N = len(sr_trials)
    v = float(np.nanvar(sr_trials))
    g = 0.5772156649
    sr0 = math.sqrt(v) * ((1 - g) * sst.norm.ppf(1 - 1 / N) + g * sst.norm.ppf(1 - 1 / (N * math.e)))
    den = math.sqrt(max(1 - skew * sr + (kurt - 1) / 4 * sr ** 2, 1e-9))
    return float(sst.norm.cdf((sr - sr0) * math.sqrt(n - 1) / den))


# ====================================================================== quantile regressions (H-WS08-12)

QR_PRED = ["fund", "fund_pct", "basis", "doi", "dvol", "vrp", "skew"]


def analyse_qr(P: dict, B: int = 299, seed: int = 8, log=print) -> list[dict]:
    from statsmodels.regression.quantile_regression import QuantReg

    rng = np.random.default_rng(seed)
    pan = P["daily"]
    g = pan["grid"]
    co = cutoff_ms()
    monday = ((g // D_MS) + 3) % 7 == 0          # 1970-01-01 was a Thursday -> day index % 7 == 4 is Monday
    out = []
    for h in (1, 7):
        T = targets(pan, h)
        for p in QR_PRED:
            for a in ("BTC", "ETH", "ALT"):
                x = pan["x"][a].get(p)
                if x is None:
                    continue
                y = T[a][0]
                fin = np.isfinite(x) & np.isfinite(y) & (monday if h == 7 else True)
                is_m, ho_m = fin & (g + h * D_MS <= co), fin & (g >= co)
                if is_m.sum() < 150:
                    continue
                mu, sd = x[is_m].mean(), x[is_m].std()
                X = _design((x - mu) / sd)
                rec = {"cell": f"QR:{p}:{a}:{h}", "pred": p, "asset": a, "h": h, "n_is": int(is_m.sum())}
                fits = {}
                for q in (0.1, 0.5, 0.9):
                    r = QuantReg(y[is_m], X[is_m]).fit(q=q, max_iter=2000)
                    fits[q] = r
                    rec[f"b{int(q * 100)}"] = float(r.params[1])
                    rec[f"p{int(q * 100)}"] = float(r.pvalues[1])
                rec["spread"] = rec["b90"] - rec["b10"]
                ii = np.flatnonzero(is_m)
                blk = 60 if h == 1 else 9
                nb = int(math.ceil(len(ii) / blk))
                spreads = []
                for _ in range(B):
                    st_ = rng.integers(0, len(ii) - blk + 1, size=nb)
                    sel = ii[(st_[:, None] + np.arange(blk)).ravel()[: len(ii)]]
                    try:
                        b90 = QuantReg(y[sel], X[sel]).fit(q=0.9, max_iter=500).params[1]
                        b10 = QuantReg(y[sel], X[sel]).fit(q=0.1, max_iter=500).params[1]
                        spreads.append(b90 - b10)
                    except Exception:  # noqa: BLE001
                        continue
                se = float(np.std(spreads))
                rec["spread_se"] = se
                rec["spread_p"] = float(2 * sst.norm.sf(abs(rec["spread"] / se))) if se > 0 else float("nan")
                if ho_m.sum() >= 30:
                    Xh = _design((x[ho_m] - mu) / sd)
                    try:
                        rec["spread_ho"] = float(QuantReg(y[ho_m], Xh).fit(q=0.9).params[1] - QuantReg(y[ho_m], Xh).fit(q=0.1).params[1])
                        rec["b50_ho"] = float(QuantReg(y[ho_m], Xh).fit(q=0.5).params[1])
                        rec["n_ho"] = int(ho_m.sum())
                    except Exception:  # noqa: BLE001
                        pass
                out.append(rec)
        log(f"QR h={h} done")
    adj = holm([r["spread_p"] for r in out])
    for r, pa in zip(out, adj, strict=True):
        r["spread_p_holm"] = pa
    return out


# ====================================================================== XSECT (WS09 core)

def xsect_panel(as_of: datetime, log=print) -> dict:
    """Daily close / turnover / funding panel of Bybit USDT crypto perps in the lake (TradFi excluded)."""
    import duckdb

    from ai_trader.core.config import load
    from ai_trader.storage.lake import Lake

    lake = Lake()
    con = duckdb.connect()
    syms = [r[0] for r in con.sql(
        f"select distinct symbol from read_parquet('{(lake.root / 'bars').as_posix()}/venue=bybit/price_type=last/freq=1d/**/*.parquet', "
        "hive_partitioning=true, union_by_name=true)").fetchall()]
    syms = sorted(s for s in syms if s.endswith("USDT") and s not in TRADFI)
    rs = load().settings["research"]
    per = {}
    for s in syms:
        b = lake.bars(s, "1d", as_of).select("event_ts", "close_ts", "open", "high", "low", "close", "turnover").sort("event_ts")
        b = b.filter((pl.col("close") > 0) & (pl.col("low") > 0))
        if b.height < 90:
            continue
        first = b["event_ts"].min()
        days = rs["burn_in_days_new"] if first.year >= rs["burn_in_new_listing_year"] else rs["burn_in_days"]
        b = b.filter(pl.col("event_ts") >= first + timedelta(days=days))
        per[s] = b
    t0 = min(int(_ms(b["event_ts"])[0]) for b in per.values())
    t1 = max(int(_ms(b["event_ts"])[-1]) for b in per.values())
    days = np.arange(t0, t1 + 1, D_MS, dtype=np.int64)
    names = list(per)
    C = np.full((len(days), len(names)), np.nan)
    TO = np.full_like(C, np.nan)
    for j, s in enumerate(names):
        b = per[s]
        idx = ((_ms(b["event_ts"]) - t0) // D_MS).astype(int)
        C[idx, j] = b["close"].to_numpy()
        TO[idx, j] = b["turnover"].to_numpy()
    F = np.full_like(C, np.nan)      # funding sum settled during day d (event day start)
    for j, s in enumerate(names):
        f = lake.read_asof("funding", as_of, symbols=[s], filters={"venue": "bybit"})
        if f.height:
            k = _ms(f["event_ts"])
            dd = ((k - 1 - t0) // D_MS).astype(int)    # F[d] = settlements in (d 00:00, d+1 00:00]
            ok = (dd >= 0) & (dd < len(days))
            F[:, j] = 0.0
            np.add.at(F[:, j], dd[ok], f["rate"].to_numpy()[ok])
            first_f = int(dd[ok].min()) if ok.any() else len(days)
            F[:first_f, j] = np.nan
    # delisting / last bar per symbol (survivorship context)
    last = {s: datetime.fromtimestamp(int(_ms(per[s]["event_ts"])[-1]) / 1000, timezone.utc).date().isoformat() for s in names}
    log(f"xsect panel {len(days)} days x {len(names)} symbols")
    return {"days": days, "names": names, "C": C, "TO": TO, "F": F, "last_bar": last}


def xsect_chars(X: dict) -> dict:
    """Daily characteristics known at the origin 00:00 of day index i (day i not yet started): uses days < i."""
    C, TO, F, names = X["C"], X["TO"], X["F"], X["names"]
    lc = np.log(C)
    r = np.vstack([np.full((1, C.shape[1]), np.nan), np.diff(lc, axis=0)])   # r[d] = return of day d (close d-1 -> close d)
    nd, ns = C.shape
    j_btc = names.index("BTCUSDT")
    mom = np.full((nd, ns), np.nan)
    rev = np.full_like(mom, np.nan)
    carry = np.full_like(mom, np.nan)
    beta = np.full_like(mom, np.nan)
    sig30 = np.full_like(mom, np.nan)
    liq = np.full_like(mom, np.nan)
    for i in range(61, nd):
        # origin = start of day i; last closed day = i-1
        mom[i] = lc[i - 2] - lc[i - 31]
        rev[i] = lc[i - 1] - lc[i - 8]
        carry[i] = np.where(np.isfinite(F[i - 7:i]).all(axis=0), np.nansum(F[i - 7:i], axis=0), np.nan)
        w = r[i - 60:i]
        okb = np.isfinite(w) & np.isfinite(w[:, [j_btc]])
        xb = w[:, j_btc]
        for j in range(ns):
            m = okb[:, j]
            if m.sum() >= 45:
                xv = xb[m]
                yv = w[m, j]
                vx = xv.var()
                beta[i, j] = float(((xv - xv.mean()) * (yv - yv.mean())).mean() / vx) if vx > 0 else np.nan
        s30 = r[i - 30:i]
        cnt = np.isfinite(s30).sum(axis=0)
        sig30[i] = np.where(cnt >= 25, np.nanstd(s30, axis=0), np.nan)
        liq[i] = np.nanmedian(TO[i - 30:i], axis=0)
    return {"r": r, "mom": mom, "rev": rev, "carry": carry, "beta": beta, "sig30": sig30, "liq": liq}


def _rank_c(v: np.ndarray) -> np.ndarray:
    out = np.full(len(v), np.nan)
    m = np.isfinite(v)
    if m.sum() >= 2:
        rk = sst.rankdata(v[m])
        out[m] = (rk - 1) / (m.sum() - 1) - 0.5
    return out


def analyse_xsect(X: dict, CH: dict, B: int = 999, seed: int = 8, min_liq: float = 2e7, log=print) -> dict:
    days, C, F, names = X["days"], X["C"], X["F"], X["names"]
    lc = np.log(C)
    co = cutoff_ms()
    nd = len(days)
    mondays = [i for i in range(61, nd - 7) if ((days[i] // D_MS) + 3) % 7 == 0]
    chars = ["mom", "rev", "carry", "beta"]
    rows = []
    w_prev = {c: {} for c in chars}
    ls_rows = []
    for i in mondays:
        ok = (np.isfinite(CH["liq"][i]) & (CH["liq"][i] >= min_liq) & np.isfinite(lc[i - 1]) & np.isfinite(lc[i + 6])
              & np.isfinite(CH["sig30"][i]))
        for c in chars:
            ok &= np.isfinite(CH[c][i])
        if ok.sum() < 9:
            continue
        j = np.flatnonzero(ok)
        ret = lc[i + 6, j] - lc[i - 1, j]
        fnext = np.where(np.isfinite(F[i:i + 7, j]).all(axis=0), F[i:i + 7, j].sum(axis=0), 0.0)
        tot = ret - fnext
        y = ret / (CH["sig30"][i, j] * math.sqrt(7))
        Rk = np.column_stack([_rank_c(CH[c][i, j]) for c in chars])
        Xd = np.column_stack([np.ones(len(j)), Rk])
        bj = np.linalg.lstsq(Xd, y, rcond=None)[0]
        single = [np.linalg.lstsq(np.column_stack([np.ones(len(j)), Rk[:, k]]), y, rcond=None)[0][1] for k in range(4)]
        braw = np.linalg.lstsq(np.column_stack([np.ones(len(j)), CH["beta"][i, j]]), ret, rcond=None)[0][1]
        row = {"ts": int(days[i]), "n": int(len(j)), "seg": "is" if days[i] + 7 * D_MS <= co else ("ho" if days[i] >= co else "x"),
               "btc_ret": float(lc[i + 6, names.index("BTCUSDT")] - lc[i - 1, names.index("BTCUSDT")])}
        for k, c in enumerate(chars):
            row[f"fm_{c}"] = float(bj[k + 1])
            row[f"fm1_{c}"] = float(single[k])
        row["fm_beta_raw"] = float(braw)
        # tercile long-short portfolios
        for c in chars:
            v = CH[c][i, j]
            q1, q2 = np.quantile(v, [1 / 3, 2 / 3])
            top, bot = v >= q2, v <= q1
            sgn = 1 if c == "mom" else -1            # registered direction: MOM top-bottom; REV / CARRY / BETA bottom-top
            w = {}
            for jj, t_, b_ in zip(j, top, bot, strict=True):
                if t_:
                    w[jj] = sgn / top.sum()
                elif b_:
                    w[jj] = -sgn / bot.sum()
            keys = set(w) | set(w_prev[c])
            turn = sum(abs(w.get(k_, 0) - w_prev[c].get(k_, 0)) for k_ in keys)
            w_prev[c] = w
            pr = sum(w[k_] * ret[list(j).index(k_)] for k_ in w)
            tt = sum(w[k_] * tot[list(j).index(k_)] for k_ in w)
            row[f"ls_{c}_price"] = float(pr)
            row[f"ls_{c}_total"] = float(tt)
            row[f"ls_{c}_cost"] = float(turn * 5.5e-4)
        rows.append(row)
    df = pl.DataFrame(rows)
    out = {"n_weeks": df.height, "median_n": float(df["n"].median())}
    rng = np.random.default_rng(seed)
    for seg in ("is", "ho"):
        d = df.filter(pl.col("seg") == seg)
        if d.height < 8:
            continue
        o = {"weeks": d.height, "median_n": float(d["n"].median())}
        for c in chars:
            for kind in ("fm", "fm1"):
                v = d[f"{kind}_{c}"].to_numpy()
                m = v.mean()
                se = math.sqrt(nw_meat((v - m)[:, None], 4)[0, 0]) / len(v)
                o[f"{kind}_{c}"] = {"mean": float(m), "t": float(m / se)}
            for kind in ("price", "total"):
                v = d[f"ls_{c}_{kind}"].to_numpy() - d[f"ls_{c}_cost"].to_numpy()
                m = v.mean()
                se = math.sqrt(nw_meat((v - m)[:, None], 4)[0, 0]) / len(v)
                o[f"ls_{c}_{kind}_net"] = {"mean_wk": float(m), "t": float(m / se), "sharpe_ann": float(m / v.std() * math.sqrt(52)),
                                           "gross_mean_wk": float(d[f"ls_{c}_{kind}"].mean()), "cost_wk": float(d[f"ls_{c}_cost"].mean())}
        # Shanken correction for the raw-beta FM premium (traded factor = BTC weekly return)
        lam = d["fm_beta_raw"].to_numpy()
        lm = lam.mean()
        se_fm2 = (math.sqrt(nw_meat((lam - lm)[:, None], 4)[0, 0]) / len(lam)) ** 2
        sf2 = float(d["btc_ret"].var())
        Tn = len(lam)
        se_sh = math.sqrt(max((1 + lm ** 2 / sf2) * (se_fm2 - sf2 / Tn) + sf2 / Tn, se_fm2 * (1 + lm ** 2 / sf2)))
        o["fm_beta_raw"] = {"mean": float(lm), "t_fm": float(lm / math.sqrt(se_fm2)), "t_shanken": float(lm / se_sh), "btc_mean_wk": float(d["btc_ret"].mean())}
        # Romano-Wolf over the 4 joint FM slopes (week-block bootstrap)
        M = np.column_stack([d[f"fm_{c}"].to_numpy() for c in chars])
        mh = M.mean(axis=0)
        n = len(M)
        K = block_counts(n, 8, B, rng)
        mb = (K @ M) / K.sum(axis=1, keepdims=True)
        sdb = mb.std(axis=0)
        o["rw_p"] = dict(zip(chars, romano_wolf(np.abs(mh) / sdb, np.abs(mb - mh) / sdb).tolist(), strict=True))
        out[seg] = o
    out["weekly"] = rows
    return out


def analyse_trend_sleeve(X: dict, CH: dict, B: int = 999, seed: int = 8, log=print) -> dict:
    """zoo_ensemble / trend_4h base alt trades: does the xs rank at entry (signed by trade side) predict trade R?"""
    tr = pl.read_parquet(ROOT / "data" / "research" / "stress" / "trades" / "*.parquet").filter(
        pl.col("unit").is_in(["zoo_ensemble", "trend_4h"]) & (pl.col("variant") == "base") & ~pl.col("symbol").is_in(["BTCUSDT", "ETHUSDT"])
        & pl.col("exit").is_not_null() & pl.col("stop").is_not_null() & (pl.col("why") != "end"))
    if "kind" in tr.columns:
        kinds = tr["kind"].unique().to_list()
    else:
        kinds = []
    tr = tr.with_columns((pl.col("side") * (pl.col("exit") - pl.col("entry")) / (pl.col("entry") - pl.col("stop")).abs()).alias("R"))
    tr = tr.filter(pl.col("R").is_finite())
    days, names = X["days"], X["names"]
    col = {s: i for i, s in enumerate(names)}
    et = _ms(tr["entry_ts"])
    di = ((et - days[0]) // D_MS).astype(int)          # origin day = day containing entry; chars at its 00:00 use days < di
    rk = {c: np.full(tr.height, np.nan) for c in ("mom", "carry", "beta")}
    liq_ok = CH["liq"] >= 2e7
    cache = {}
    syms = tr["symbol"].to_list()
    for n_, (d_, s_) in enumerate(zip(di, syms, strict=True)):
        if s_ not in col or d_ < 0 or d_ >= len(days):
            continue
        if d_ not in cache:
            ok = liq_ok[d_] & np.isfinite(CH["mom"][d_]) & np.isfinite(CH["carry"][d_]) & np.isfinite(CH["beta"][d_])
            cache[d_] = (ok, {c: _rank_c(np.where(ok, CH[c][d_], np.nan)) for c in rk})
        ok, rr = cache[d_]
        j = col[s_]
        if ok[j]:
            for c in rk:
                rk[c][n_] = rr[c][j]
    side = tr["side"].to_numpy().astype(float)
    R = tr["R"].to_numpy()
    seg = tr["seg"].to_list() if "seg" in tr.columns else ["?"] * tr.height
    wk = (et // (7 * D_MS)).astype(int)
    out = {"n_trades": tr.height, "kinds": kinds, "units": tr["unit"].value_counts().rows()}
    rng = np.random.default_rng(seed)
    for sname in sorted(set(seg)):
        m = np.array([s == sname for s in seg]) & np.isfinite(rk["mom"]) & np.isfinite(rk["carry"]) & np.isfinite(rk["beta"])
        if m.sum() < 30:
            out[sname] = {"n": int(m.sum())}
            continue
        Xs = np.column_stack([np.ones(m.sum())] + [side[m] * rk[c][m] for c in ("mom", "carry", "beta")])
        beta_, *_ = np.linalg.lstsq(Xs, R[m], rcond=None)
        e = R[m] - Xs @ beta_
        XtXi = np.linalg.inv(Xs.T @ Xs)
        meat = np.zeros((4, 4))
        for w_ in np.unique(wk[m]):
            sel = wk[m] == w_
            sc = (Xs[sel] * e[sel, None]).sum(axis=0)
            meat += np.outer(sc, sc)
        V = XtXi @ meat @ XtXi
        se = np.sqrt(np.diag(V))
        smom = side[m] * rk["mom"][m]
        filt = smom > 0
        diff = R[m][filt].mean() - R[m].mean()
        wu = np.unique(wk[m])
        wid = np.searchsorted(wu, wk[m])
        sums_all = np.bincount(wid, weights=R[m], minlength=len(wu))
        cnt_all = np.bincount(wid, minlength=len(wu)).astype(float)
        sums_f = np.bincount(wid, weights=R[m] * filt, minlength=len(wu))
        cnt_f = np.bincount(wid, weights=filt.astype(float), minlength=len(wu))
        K = block_counts(len(wu), 4, B, rng)
        dboot = (K @ sums_f) / np.maximum(K @ cnt_f, 1) - (K @ sums_all) / np.maximum(K @ cnt_all, 1)
        out[sname] = {"n": int(m.sum()), "mean_R": float(R[m].mean()),
                      "coef": dict(zip(["const", "s_mom", "s_carry", "s_beta"], beta_.tolist(), strict=True)),
                      "t": dict(zip(["const", "s_mom", "s_carry", "s_beta"], (beta_ / se).tolist(), strict=True)),
                      "filter_top_half_mom": {"n": int(filt.sum()), "mean_R": float(R[m][filt].mean()), "diff": float(diff),
                                              "ci": [float(np.quantile(dboot, 0.025)), float(np.quantile(dboot, 0.975))]}}
    return out


# ====================================================================== MACRO (WS18 core)

def kalman_rw(y: np.ndarray, X: np.ndarray, q: np.ndarray, r: float, b0: np.ndarray, P0: np.ndarray):
    """Random-walk-coefficient regression filter. Returns one-step predictions, predictive variances, log-likelihood terms."""
    n, k = X.shape
    b, P = b0.copy(), P0.copy()
    Q = np.diag(q)
    yhat, F, ll, betas = np.empty(n), np.empty(n), np.empty(n), np.empty((n, k))
    for t in range(n):
        P = P + Q
        x = X[t]
        yhat[t] = x @ b
        F[t] = x @ P @ x + r
        betas[t] = b
        v = y[t] - yhat[t]
        ll[t] = -0.5 * (math.log(2 * math.pi * F[t]) + v * v / F[t])
        Kg = P @ x / F[t]
        b = b + Kg * v
        P = P - np.outer(Kg, x @ P)
    return yhat, F, ll, betas


def analyse_macro(P: dict, log=print) -> dict:
    from scipy.optimize import minimize

    pan = P["daily"]
    od = pan["grid"]
    co = cutoff_ms()
    # same-day macro moves: obs_date d (US close inside the crypto UTC day d)
    def obs_map(name: str, kind: str) -> np.ndarray:
        d = pl.read_parquet(ROOT / "data" / "research" / "macro" / f"{name}.parquet").filter(pl.col("value").is_not_null())
        d = d.sort("obs_date").unique("obs_date", keep="last").sort("obs_date")
        dd = (d["obs_date"].cast(pl.Datetime("ms")).dt.epoch("ms").to_numpy()).astype(np.int64)
        v = d["value"].to_numpy().astype(float)
        ch = np.r_[np.nan, np.diff(v) if kind == "diff" else np.diff(np.log(v))]
        gap = np.r_[np.nan, np.diff(dd)] <= 4 * D_MS
        ch = np.where(gap, ch, np.nan)
        out = np.full(len(od), np.nan)
        pos = np.searchsorted(od, dd)
        ok = (pos < len(od)) & (od[np.clip(pos, 0, len(od) - 1)] == dd)
        out[pos[ok]] = ch[ok]
        return out

    ndx, ry, usd, hy = obs_map("NASDAQ100", "ldiff"), obs_map("DFII10", "diff"), obs_map("DTWEXBGS", "ldiff"), obs_map("BAMLH0A0HYM2", "diff")
    res = {}
    for a in ("BTC", "ETH", "ALT"):
        r = pan["r"][a]
        sig = pan["sig"][a]
        y = r / sig
        m = np.isfinite(y) & np.isfinite(ndx) & np.isfinite(ry) & np.isfinite(usd)
        idx = np.flatnonzero(m)
        Xf = np.column_stack([np.ones(len(idx)), ndx[idx] * 100, ry[idx], usd[idx] * 100])   # % moves, real-yield pp
        yf = y[idx]
        g = od[idx]
        isr = g < co
        n = len(idx)
        # constant (expanding) and rolling-90 betas, PIT (fit on rows < t)
        pc, vc, pr, vr = (np.full(n, np.nan) for _ in range(4))
        for t in range(120, n):
            b = np.linalg.lstsq(Xf[:t], yf[:t], rcond=None)[0]
            e = yf[:t] - Xf[:t] @ b
            pc[t], vc[t] = Xf[t] @ b, e.var()
            if t >= 90:
                br = np.linalg.lstsq(Xf[t - 90:t], yf[t - 90:t], rcond=None)[0]
                er = yf[t - 90:t] - Xf[t - 90:t] @ br
                pr[t], vr[t] = Xf[t] @ br, er.var()
        b0 = np.linalg.lstsq(Xf[:120], yf[:120], rcond=None)[0]
        P0 = np.eye(4) * 0.5
        fit_rows = np.arange(n) < int(isr.sum())

        def nll(th):
            q, rr = np.exp(th[:4]), math.exp(th[4])
            _, _, ll, _ = kalman_rw(yf[:fit_rows.sum()], Xf[:fit_rows.sum()], q, rr, b0, P0)
            return -ll[120:].sum()

        opt = minimize(nll, np.r_[np.log([1e-4, 1e-4, 1e-4, 1e-4]), 0.0], method="L-BFGS-B", bounds=[(-16, 0)] * 4 + [(-3, 2)])
        q, rr = np.exp(opt.x[:4]), math.exp(opt.x[4])
        pk, vk, llk, betas = kalman_rw(yf, Xf, q, rr, b0, P0)
        rec = {"n": n, "kalman_q": q.tolist(), "kalman_r": rr}
        gauss = lambda yv, mu, v: -0.5 * (np.log(2 * np.pi * v) + (yv - mu) ** 2 / v)  # noqa: E731
        start_oos = int(datetime(2022, 1, 1, tzinfo=timezone.utc).timestamp() * 1000)
        for seg, sm in (("is", (g >= start_oos) & (g < co)), ("ho", g >= co)):
            mm = sm & np.isfinite(pc) & np.isfinite(pr)
            se_k, se_c, se_r = (yf - pk) ** 2, (yf - pc) ** 2, (yf - pr) ** 2
            ls_k, ls_c, ls_r = gauss(yf, pk, vk), gauss(yf, pc, vc), gauss(yf, pr, vr)
            o = {"n": int(mm.sum()), "mse": {"kalman": float(se_k[mm].mean()), "const": float(se_c[mm].mean()), "roll90": float(se_r[mm].mean())},
                 "logscore": {"kalman": float(ls_k[mm].mean()), "const": float(ls_c[mm].mean()), "roll90": float(ls_r[mm].mean())},
                 "dm_mse_kalman_vs_const": dm_test(se_k[mm] - se_c[mm]), "dm_mse_kalman_vs_roll": dm_test(se_k[mm] - se_r[mm]),
                 "dm_ls_kalman_vs_const": dm_test(-(ls_k[mm] - ls_c[mm])), "dm_ls_kalman_vs_roll": dm_test(-(ls_k[mm] - ls_r[mm])),
                 "beta_ndx_kalman_mean": float(betas[mm, 1].mean()), "beta_ndx_kalman_sd": float(betas[mm, 1].std()),
                 "beta_ry_kalman_mean": float(betas[mm, 2].mean()), "beta_usd_kalman_mean": float(betas[mm, 3].mean())}
            for k_ in ("dm_mse_kalman_vs_const", "dm_mse_kalman_vs_roll", "dm_ls_kalman_vs_const", "dm_ls_kalman_vs_roll"):
                o[k_] = {kk: v for kk, v in o[k_].items() if kk in ("t", "p", "mean")}
            # constant betas with HAC on raw % returns; down-day asymmetry
            raw = r[idx] * 100
            Xa = np.column_stack([np.ones(n), np.maximum(ndx[idx], 0) * 100, np.minimum(ndx[idx], 0) * 100, ry[idx], usd[idx] * 100])
            sel = sm if seg == "ho" else (g < co)
            bA, VA, _ = ols_nw_multi(raw[sel], Xa[sel], 5)
            cvec = np.array([0, -1, 1, 0, 0.0])
            dlt = float(cvec @ bA)
            sed = math.sqrt(float(cvec @ VA @ cvec))
            o["asym"] = {"b_up": float(bA[1]), "b_dn": float(bA[2]), "diff_dn_minus_up": dlt, "t": dlt / sed,
                         "p": float(2 * sst.norm.sf(abs(dlt / sed))), "n": int(sel.sum())}
            Xc = np.column_stack([np.ones(n), ndx[idx] * 100, ry[idx], usd[idx] * 100])
            bC, VC, _ = ols_nw_multi(raw[sel], Xc[sel], 5)
            o["const_beta_raw"] = {nm: {"b": float(bC[i]), "t": float(bC[i] / math.sqrt(VC[i, i]))} for i, nm in enumerate(["a", "ndx", "ry", "usd"])}
            mh = sel & np.isfinite(hy[idx])
            if mh.sum() > 100:
                Xh = np.column_stack([np.ones(mh.sum()), ndx[idx][mh] * 100, ry[idx][mh], usd[idx][mh] * 100, hy[idx][mh]])
                bH, VH, _ = ols_nw_multi(raw[mh], Xh, 5)
                o["with_hy"] = {nm: {"b": float(bH[i]), "t": float(bH[i] / math.sqrt(VH[i, i]))} for i, nm in enumerate(["a", "ndx", "ry", "usd", "hy"])}
                o["with_hy"]["n"] = int(mh.sum())
            rec[seg] = o
        res[a] = rec
        log(f"macro {a} done")
    for seg in ("is", "ho"):
        ps = [res[a][seg]["asym"]["p"] for a in ("BTC", "ETH", "ALT")]
        for a, pa in zip(("BTC", "ETH", "ALT"), holm(ps), strict=True):
            res[a][seg]["asym"]["p_holm"] = pa
    return res


def analyse_rv(P: dict, log=print) -> dict:
    """log-HAR vs + log DVOL^2 vs + rr25 vs both, direct h-day, expanding refit every 30 days, QLIKE, DM (HLN), MCS."""
    pan = P["daily"]
    od = pan["grid"]
    co = cutoff_ms()
    out = {}
    for a in ("BTC", "ETH"):
        rv = pan["rv"][a]
        lrv = np.log(np.maximum(rv, 1e-12))
        rvw = rolling_sum_valid(rv, 7) / 7
        rvm = rolling_sum_valid(rv, 30) / 30
        # origin od[i] = start of day i: regressors from days <= i-1
        sh = lambda v: np.r_[np.nan, v[:-1]]  # noqa: E731
        Xd, Xw, Xm = np.log(sh(rv)), np.log(sh(rvw)), np.log(sh(rvm))
        dvol = pan["x"][a]["dvol"]
        ldv = np.log((dvol / 100.0) ** 2 / 365.0)
        skew = pan["x"][a]["skew"]
        for h in (1, 7):
            fwd = rolling_sum_valid(rv[::-1], h)[::-1] / h
            fwd = np.concatenate([fwd[: len(rv) - h + 1], np.full(h - 1, np.nan)]) if h > 1 else fwd
            ly = np.log(np.maximum(fwd, 1e-12))
            specs = {"HAR": [Xd, Xw, Xm], "HAR+DVOL": [Xd, Xw, Xm, ldv], "HAR+SKEW": [Xd, Xw, Xm, skew], "HAR+DVOL+SKEW": [Xd, Xw, Xm, ldv, skew]}
            fc = {k: np.full(len(od), np.nan) for k in specs}
            base_ok = np.isfinite(ly) & np.isfinite(Xd) & np.isfinite(Xw) & np.isfinite(Xm) & np.isfinite(ldv)
            for k, cols in specs.items():
                Xall = np.column_stack([np.ones(len(od))] + cols)
                okr = np.isfinite(Xall).all(axis=1)
                last_fit, coef, s2 = -10 ** 9, None, None
                for i in range(len(od)):
                    if not okr[i]:
                        continue
                    if i - last_fit >= 30 or coef is None:
                        tr = okr & np.isfinite(ly) & (np.arange(len(od)) <= i - h)
                        if tr.sum() >= 365:
                            coef = np.linalg.lstsq(Xall[tr], ly[tr], rcond=None)[0]
                            s2 = float(((ly[tr] - Xall[tr] @ coef) ** 2).mean())
                            last_fit = i
                    if coef is not None:
                        fc[k][i] = math.exp(float(Xall[i] @ coef) + s2 / 2)
            rec = {}
            for seg, sm in (("is", od + h * D_MS <= co), ("ho", od >= co)):
                m2 = sm & base_ok & np.isfinite(fwd) & np.isfinite(fc["HAR"]) & np.isfinite(fc["HAR+DVOL"])
                ql = {k: fwd[m2] / fc[k][m2] - np.log(fwd[m2] / fc[k][m2]) - 1 for k in ("HAR", "HAR+DVOL")}
                o = {"n": int(m2.sum()), "qlike": {k: float(v.mean()) for k, v in ql.items()},
                     "dm_dvol_vs_har": dm_test(ql["HAR+DVOL"] - ql["HAR"], h)}
                m4 = m2 & np.isfinite(skew) & np.isfinite(fc["HAR+SKEW"]) & np.isfinite(fc["HAR+DVOL+SKEW"])
                if m4.sum() >= 60:
                    L = np.column_stack([fwd[m4] / fc[k][m4] - np.log(fwd[m4] / fc[k][m4]) - 1 for k in specs])
                    o["n4"] = int(m4.sum())
                    o["qlike4"] = dict(zip(specs, L.mean(axis=0).tolist(), strict=True))
                    o["dm_skew_increment"] = dm_test(L[:, 3] - L[:, 1], h)
                    o["dm_skew_vs_har"] = dm_test(L[:, 2] - L[:, 0], h)
                    o["mcs90"] = mcs_tmax(L, list(specs), alpha=0.10, block=max(30, 3 * h), B=499)
                rec[seg] = o
            out[f"{a}:{h}"] = rec
        log(f"rv {a} done")
    return out
