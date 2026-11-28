"""WMN Hawkes with exogenous news marks (H-WMN-9): discrete 5m bins, cloglog link with a linear intensity

    P(D episode starts in bin k) = 1 - exp(-lambda_k)
    lambda_k = mu * rate_i * h(hour) + alpha * S_k + a_m * B_k + g_x * U^x_k + g_hn * U^hn_k + g_mac * U^mac_k

S / B = exponentially decayed counts of the symbol's own / BTC's past episode starts (strictly past bins, common
half-life); U = decayed counts of precise exchange announcements, HN stories (tagged to the symbol) and macro releases
(market), entering at the first bar starting at or after known_ts (common news half-life). The two half-lives are
profiled over a small grid; the linear parameters (>= 0) are fit by MLE with an analytic gradient. M0 = without the
news terms. Baseline rate_i and hour factor come from the training window only.
"""

from __future__ import annotations

import math

import numpy as np
import polars as pl
from scipy import optimize
from scipy.signal import lfilter

from ai_trader.research import wmn_data as wd
from ai_trader.research.wmn_study import BAR_MS, CUTOFF_MS, DAY_MS, precise_token_events

HL_SELF = (2.0, 4.0, 8.0)  # bins: 10, 20, 40 min
HL_NEWS = (3.0, 12.0, 48.0, 192.0)  # bins: 15 min, 1 h, 4 h, 16 h
M0 = [0, 1, 2]
M1 = [0, 1, 2, 3, 4, 5]
NAMES = ["base", "self", "btc", "exchange", "hn", "macro"]


def ewfilt(x: np.ndarray, hl_bins: float, lag1: bool) -> np.ndarray:
    """u_k = sum_j x_j a^(k - j) over j <= k (lag1: j <= k - 1, i.e. strictly past), a = 2^(-1 / hl)."""
    a = 2.0 ** (-1.0 / hl_bins)
    u = lfilter([1.0], [1.0, -a], x.astype(float))
    if lag1:
        u = np.r_[0.0, u[:-1]]
    return u


def inputs(log=print) -> dict:
    ev = pl.read_parquet(wd.OUT / "events.parquet")
    pt = precise_token_events(ev)
    E = pl.read_parquet(wd.OUT / "episodes.parquet").filter(pl.col("flag") == "D")
    btc_eps = np.sort(E.filter(pl.col("symbol") == "BTCUSDT")["ts"].to_numpy())
    mac = np.sort(ev.filter(pl.col("category") == "macro_release")["known_ts"].to_numpy())
    data = {}
    for sym in wd.CORE:
        if sym == "BTCUSDT":
            continue
        dp = wd.OUT / "det" / f"{sym}.parquet"
        if not dp.is_file():
            continue
        det = pl.read_parquet(dp, columns=["ts", "z"])
        ts = det["ts"].to_numpy()
        t0, t1 = int(ts[0]), int(ts[-1])
        grid = np.arange(t0, t1 + BAR_MS, BAR_MS)
        valid = np.zeros(grid.size, bool)
        valid[(ts - t0) // BAR_MS] = np.isfinite(det["z"].to_numpy())

        def counts(times, t0=t0, n=grid.size):
            k = np.ceil((np.asarray(times, dtype=np.int64) - t0) / BAR_MS).astype(np.int64)
            k = k[(k >= 0) & (k < n)]
            return np.bincount(k, minlength=n).astype(np.float32)

        def ind(times, t0=t0, n=grid.size):
            v = np.zeros(n, dtype=np.float32)
            k = (np.asarray(times, dtype=np.int64) - t0) // BAR_MS
            v[k[(k >= 0) & (k < n)]] = 1.0
            return v

        s_ev = pt.filter(pl.col("symbol") == sym)
        data[sym] = {"grid": grid, "valid": valid, "y": ind(E.filter(pl.col("symbol") == sym)["ts"].to_numpy()),
                     "yb": ind(btc_eps), "c_exch": counts(s_ev.filter(pl.col("source") != "hn")["known_ts"].to_numpy()),
                     "c_hn": counts(s_ev.filter(pl.col("source") == "hn")["known_ts"].to_numpy()), "c_mac": counts(mac),
                     "hour": ((grid // 3_600_000) % 24).astype(np.int8)}
    log(f"hawkes inputs: {len(data)} symbols, {sum(int(d['valid'].sum()) for d in data.values())} valid bins", flush=True)
    return data


def train_stats(data: dict, lo: int, hi: int) -> tuple[dict, np.ndarray]:
    rates, hy, hn = {}, np.zeros(24), np.zeros(24)
    for sym, d in data.items():
        m = d["valid"] & (d["grid"] >= lo) & (d["grid"] < hi)
        if m.sum() > 2000:
            rates[sym] = float(d["y"][m].mean()) + 1e-6
        np.add.at(hy, d["hour"][m], d["y"][m])
        np.add.at(hn, d["hour"][m], 1)
    rates["_pooled"] = float(np.median([v for k, v in rates.items()])) if rates else 1e-3
    p = hy.sum() / max(hn.sum(), 1)
    hr = (hy + 36 * p) / (hn + 36)  # shrink each hour toward the pooled rate (36 pseudo-bins = 3 days)
    return rates, hr / p


def design(data: dict, hs: float, hnw: float, lo: int, hi: int, rates: dict, hfac: np.ndarray):
    cols, ys, days = [], [], []
    for sym, d in data.items():
        g = d["grid"]
        m = d["valid"] & (g >= lo) & (g < hi)
        if m.sum() == 0:
            continue
        base = rates.get(sym, rates["_pooled"]) * hfac[d["hour"]]
        X = np.column_stack([base[m], ewfilt(d["y"], hs, True)[m], ewfilt(d["yb"], hs, True)[m],
                             ewfilt(d["c_exch"], hnw, False)[m], ewfilt(d["c_hn"], hnw, False)[m],
                             ewfilt(d["c_mac"], hnw, False)[m]]).astype(np.float32)
        cols.append(X)
        ys.append(d["y"][m].astype(np.int8))
        days.append((g[m] // DAY_MS).astype(np.int32))
    return np.vstack(cols), np.concatenate(ys), np.concatenate(days)


def fit_linear(X: np.ndarray, y: np.ndarray, use: list[int]) -> tuple[np.ndarray, float]:
    Xu = np.ascontiguousarray(X[:, use], dtype=np.float32)  # float32 design (memory); intensities in float64
    pos = y == 1

    def f(u):
        th = np.exp(u)
        lam = (Xu @ th.astype(np.float32)).astype(np.float64) + 1e-12
        ll = np.sum(np.log(-np.expm1(-lam[pos]))) - np.sum(lam[~pos])
        w = np.where(pos, np.exp(-lam) / (-np.expm1(-lam)), -1.0)
        g = (Xu.T @ w.astype(np.float32)).astype(np.float64) * th
        return -ll / y.size, -g / y.size

    best = None
    for start in (0.01, 0.1):
        u0 = np.log(np.r_[1.0, np.full(len(use) - 1, start)])
        r = optimize.minimize(f, u0, jac=True, method="L-BFGS-B", options={"maxiter": 800})
        if best is None or r.fun < best.fun:
            best = r
    return np.exp(best.x), float(best.fun)


def score(X: np.ndarray, y: np.ndarray, use: list[int], th: np.ndarray) -> np.ndarray:
    lam = (X[:, use] @ np.asarray(th, dtype=np.float32)).astype(np.float64) + 1e-12
    return np.where(y == 1, np.log(-np.expm1(-lam)), -lam)


def fit(data: dict, lo: int, hi: int, model: list[int], hl_self: float | None = None) -> dict:
    """Profile the half-lives on the training window; M1 takes the self half-life chosen by M0 (hl_self)."""
    rates, hfac = train_stats(data, lo, hi)
    best = None
    for hs in ((hl_self,) if hl_self else HL_SELF):
        for hnw in (HL_NEWS if model == M1 else (HL_NEWS[0],)):
            X, y, _ = design(data, hs, hnw, lo, hi, rates, hfac)
            th, nll = fit_linear(X, y, model)
            if best is None or nll < best["nll"]:
                best = {"hl_self": hs, "hl_news": hnw, "theta": th.tolist(), "nll": nll, "n": int(y.size), "pos": int(y.sum())}
            del X, y
    best.update(rates=rates, hfac=hfac.tolist(), model=model)
    return best


def evaluate(data: dict, f: dict, lo: int, hi: int):
    X, y, days = design(data, f["hl_self"], f["hl_news"], lo, hi, f["rates"], np.array(f["hfac"]))
    return score(X, y, f["model"], np.array(f["theta"])), days


def nw_t(d: np.ndarray, lag: int) -> dict:
    d = d[np.isfinite(d)]
    n = d.size
    mu = d.mean()
    u = d - mu
    v = u @ u / n
    for L in range(1, lag + 1):
        v += 2 * (1 - L / (lag + 1)) * (u[L:] @ u[:-L]) / n
    se = math.sqrt(max(v, 0) / n)
    return {"diff": float(mu), "se": se, "t": float(mu / se) if se > 0 else float("nan"), "n_days": int(n)}


def _daily(sc: np.ndarray, days: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    u, inv = np.unique(days, return_inverse=True)
    return u, np.bincount(inv, weights=sc)


def _shares(data, f, lo, hi):
    X, y, _ = design(data, f["hl_self"], f["hl_news"], lo, hi, f["rates"], np.array(f["hfac"]))
    th = np.zeros(6)
    th[f["model"]] = f["theta"]
    comp = X[y == 1].astype(float) * th
    return dict(zip(NAMES, (comp / comp.sum(1, keepdims=True)).mean(0).round(4).tolist(), strict=True))


def compare(data: dict, log=print) -> dict:
    lo = min(int(d["grid"][0]) for d in data.values())
    hi = max(int(d["grid"][-1]) for d in data.values()) + BAR_MS
    mid = (lo + CUTOFF_MS) // 2 // DAY_MS * DAY_MS
    parts = []
    folds = []
    for tr, te in (((lo, mid), (mid, CUTOFF_MS)), ((mid, CUTOFF_MS), (lo, mid))):
        f0 = fit(data, *tr, M0)
        f1 = fit(data, *tr, M1, hl_self=f0["hl_self"])
        s0, d0 = evaluate(data, f0, *te)
        s1, _ = evaluate(data, f1, *te)
        parts.append(_daily(s1 - s0, d0)[1])
        folds.append({"train": tr, "M0": f0["theta"], "M1": f1["theta"], "hl": [f1["hl_self"], f1["hl_news"]]})
        log(f"  fold done: {folds[-1]}", flush=True)
    out = {"llm_in_sample": nw_t(np.concatenate(parts), 7), "folds": folds}
    f0 = fit(data, lo, CUTOFF_MS, M0)
    f1 = fit(data, lo, CUTOFF_MS, M1, hl_self=f0["hl_self"])
    s0, d0 = evaluate(data, f0, CUTOFF_MS, hi)
    s1, _ = evaluate(data, f1, CUTOFF_MS, hi)
    out["clean_holdout"] = nw_t(_daily(s1 - s0, d0)[1], 7)
    out["holdout_mean_logscore_per_bin"] = {"M0": float(s0.mean()), "M1": float(s1.mean()), "bins": int(s0.size)}
    out["fit_M1"] = {k: f1[k] for k in ("hl_self", "hl_news", "theta", "nll", "n", "pos")}
    out["fit_M0"] = {k: f0[k] for k in ("hl_self", "hl_news", "theta", "nll", "n", "pos")}
    a = 2.0 ** (-1.0 / f1["hl_self"])
    an = 2.0 ** (-1.0 / f1["hl_news"])
    th = f1["theta"]
    out["branching_self"] = th[1] * a / (1 - a)
    out["branching_btc"] = th[2] * a / (1 - a)
    out["jumps_per_news_event"] = {"exchange": th[3] / (1 - an), "hn": th[4] / (1 - an), "macro": th[5] / (1 - an)}
    out["intensity_share_at_jumps"] = {"llm_in_sample": _shares(data, f1, lo, CUTOFF_MS), "clean_holdout": _shares(data, f1, CUTOFF_MS, hi)}
    out["hl_minutes"] = {"self": 5 * f1["hl_self"], "news": 5 * f1["hl_news"]}
    return out
