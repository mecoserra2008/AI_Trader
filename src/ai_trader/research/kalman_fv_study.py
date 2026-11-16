"""Kalman fair value and dynamic-beta pairs (block KFD: H-KFD1, H-KFD2, H-KFD3; research agent, 2026-11-16).

    PYTHONIOENCODING=utf-8 OMP_NUM_THREADS=1 uv run python -m ai_trader.research.kalman_fv_study single  [--freq 1h|5m]
    ... pairs   [--freq 1h|5m]
    ... flow    (H-KFD3: BTC/ETH 15m with flow_bars; 1m from flow_3s with delta + OFI; also H-KFD1 at 1m)
    ... evaluate                 -> data/research/kalman_fv/eval.json

Design as registered in research/hypotheses.yaml (block KFD). Summary:
  single  local level on log close with Var(eta) = Q s_t^2, Var(eps) = R s_t^2 (s_t = robust_bc one-bar-ahead vol known at t-1).
          Steady-state (fixed-gain) Kalman filter: mu_{t+1|t} = mu_{t|t-1} + K v_t; K and S = P + R by Gaussian QML on the training
          fold (yearly expanding refits, 1-day purge); Q = K^2 S, R = (1 - K) S. z_t = v_t / (s_t sqrt(S)).
  pairs   KF regression log P_A = alpha_t + beta_t (log P_B - xbar) + e_t, random-walk states, MLE of (q_a, q_b, R) on data before
          2025-01-01 (1h) / the last 120 days before it (5m); xbar = training mean (conditioning only).
  rule    fade |z| >= k, target z crosses 0, stop 1.5 sqrt(S) beyond entry, time stop; limit entries filled only on trade-through.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import sys
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl

from ai_trader.core.config import load
from ai_trader.research.trend_zoo_study import UNIVERSE

AS_OF = datetime(2026, 11, 16, 0, tzinfo=UTC)
SEGS = ("llm_in_sample", "clean_holdout")
KS = (1.5, 2.0, 2.5, 3.0, 3.5)
TSTOPS = (12, 48)
VARIANTS = [(k, t) for k in KS for t in TSTOPS]
PRIMARY = (2.5, 12)
STOP_MULT = 1.5
MAKER, TAKER = 2e-4, 7.5e-4       # taker 5.5 bp + 2 bp slippage (config/fees.yaml)
HOUR = 3600


def root() -> Path:
    p = load().root / "data" / "research" / "kalman_fv"
    p.mkdir(parents=True, exist_ok=True)
    return p


def cut_s() -> float:
    from ai_trader.research.evaluate import knowledge_cutoff

    return knowledge_cutoff().timestamp()


def nw_t(x) -> tuple[float, float, int]:
    from ai_trader.research.rough_vol_study import nw_t as f

    return f(np.asarray(x, float))


# ------------------------------------------------------------------ data


def _robust_sigma(r: np.ndarray) -> np.ndarray:
    """robust_bc (no weekend shrink) one-bar-ahead sigma for bar t, known at t-1."""
    from ai_trader.features.vol_models import VolModel, run

    out = run(np.nan_to_num(r), np.zeros(len(r), bool), VolModel(bias_correct=True, use_weekend=False))
    s = np.sqrt(out["v_next"])
    return np.r_[np.nan, s[:-1]]


def bars(sym: str, freq: str) -> dict | None:
    p = root() / f"bars_{freq}_{sym}.npz"
    if p.exists():
        z = np.load(p)
        return {k: z[k] for k in z.files}
    if freq in ("1h", "15m"):
        from ai_trader.research.frame import market_frame

        f = market_frame(sym, freq, AS_OF, funding=False, oi=False).sort("event_ts")
        if f.height < 1000:
            return None
        sig = np.r_[np.nan, f["sigma_risk_next"].to_numpy()[:-1]]
        d = {"ts": f["event_ts"].dt.epoch("s").to_numpy(), "open": f["open"].to_numpy(), "high": f["high"].to_numpy(),
             "low": f["low"].to_numpy(), "close": f["close"].to_numpy(), "sig": sig}
    elif freq == "5m":
        dd = load().root / "data" / "research" / "bos" / "bars" / "5m"
        f = pl.concat([pl.read_parquet(x) for x in sorted(dd.glob(f"{sym}_*.parquet"))]).unique("event_ts").sort("event_ts")
        f = f.filter((pl.col("event_ts") < AS_OF) & (pl.col("low") > 0) & (pl.col("high") >= pl.col("low")))
        c = f["close"].to_numpy()
        r = np.r_[np.nan, np.diff(np.log(c))]
        d = {"ts": f["event_ts"].dt.epoch("s").to_numpy(), "open": f["open"].to_numpy(), "high": f["high"].to_numpy(),
             "low": f["low"].to_numpy(), "close": c, "sig": _robust_sigma(r)}
    else:
        raise ValueError(freq)
    np.savez_compressed(p, **d)
    return d


def own_cells_4h(sym: str) -> tuple[np.ndarray, np.ndarray]:
    """(close_ts seconds, cell) of the own-token 4h v0 labeller."""
    from ai_trader.research.frame import market_frame
    from ai_trader.research.regimes import label

    f = market_frame(sym, "4h", AS_OF, funding=False, oi=False).sort("event_ts")
    lab = label(f, bar_hours=4.0).select("close_ts", "cell").drop_nulls()
    return lab["close_ts"].dt.epoch("s").to_numpy(), lab["cell"].to_numpy().astype(str)


# ------------------------------------------------------------------ single-asset KF


def ll_pred(x: np.ndarray, K: float) -> np.ndarray:
    """mu_{t|t-1} of the fixed-gain local-level filter (pred[0] = x[0])."""
    from scipy.signal import lfilter

    xf = np.where(np.isfinite(x), x, np.nan)
    # forward-fill gaps so the filter keeps its last state
    idx = np.where(np.isfinite(xf), np.arange(len(xf)), 0)
    np.maximum.accumulate(idx, out=idx)
    xf = xf[idx]
    zi = np.array([(1 - K) * xf[0]])
    filt, _ = lfilter([K], [1, -(1 - K)], xf, zi=zi)   # mu_{t|t}
    return np.r_[xf[0], filt[:-1]]


def fit_K(x: np.ndarray, s: np.ndarray, mask: np.ndarray) -> tuple[float, float]:
    """Gaussian QML of the fixed gain K (and scale S, vol units) on the masked rows."""
    from scipy.optimize import minimize_scalar

    def obj(lk: float) -> float:
        K = 1 / (1 + math.exp(-lk))
        u = (x - ll_pred(x, K)) / s
        m = mask & np.isfinite(u)
        return float(np.log(np.mean(u[m] ** 2)))

    grid = np.linspace(-6, 8, 29)
    vals = [obj(g) for g in grid]
    g0 = grid[int(np.argmin(vals))]
    res = minimize_scalar(obj, bounds=(g0 - 0.5, g0 + 0.5), method="bounded", options={"xatol": 1e-3})
    K = 1 / (1 + math.exp(-res.x))
    u = (x - ll_pred(x, K)) / s
    m = mask & np.isfinite(u)
    return K, float(np.mean(u[m] ** 2))


def single_z(d: dict, first_test_year: int) -> dict:
    """Yearly expanding refits; z, sd (log units of sqrt(S) s_t), K per row; rows before the first test year are NaN."""
    x = np.log(d["close"])
    s = d["sig"]
    ts = d["ts"]
    T = len(x)
    z = np.full(T, np.nan)
    sd = np.full(T, np.nan)
    Ks = {}
    years = sorted({datetime.fromtimestamp(int(t), UTC).year for t in ts})
    for y in years:
        if y < first_test_year:
            continue
        y0 = datetime(y, 1, 1, tzinfo=UTC).timestamp()
        y1 = datetime(y + 1, 1, 1, tzinfo=UTC).timestamp()
        tr = (ts < y0 - 86400) & np.isfinite(s) & (s > 0)
        if tr.sum() < 2000:
            continue
        K, S = fit_K(x, s, tr)
        Ks[y] = {"K": K, "S": S, "Q": K * K * S, "R": (1 - K) * S}
        pred = ll_pred(x, K)
        te = (ts >= y0) & (ts < y1)
        with np.errstate(invalid="ignore", divide="ignore"):
            z[te] = ((x - pred) / (s * math.sqrt(S)))[te]
            sd[te] = (s * math.sqrt(S))[te]
    return {"z": z, "sd": sd, "K": Ks}


def sim_single(d: dict, z: np.ndarray, sd: np.ndarray, k: float, T: int, allow: np.ndarray | None = None) -> list[tuple]:
    """Trades (entry_ts, R, kind, entry index). allow[t] False -> the signal at t is skipped (guardrail)."""
    o, h, lo, c = (d[x].tolist() for x in ("open", "high", "low", "close"))
    zl = z.tolist()
    ts = d["ts"]
    n = len(c)
    sig_idx = np.flatnonzero(np.isfinite(z) & (np.abs(z) >= k) & np.isfinite(sd) & (sd > 0))
    out = []
    busy_until = -1
    for t in sig_idx.tolist():
        if t <= busy_until or t + 2 >= n:
            continue
        if allow is not None and not allow[t]:
            continue
        dr = -1.0 if zl[t] > 0 else 1.0
        e = c[t]
        if not ((lo[t + 1] < e) if dr > 0 else (h[t + 1] > e)):
            continue
        delta = STOP_MULT * sd[t]
        stop = e * math.exp(-delta) if dr > 0 else e * math.exp(delta)
        unit = delta + MAKER + TAKER
        ex, fee, kind, jx = None, 0.0, "", None
        pend = None
        for j in range(t + 1, min(t + T + 1, n)):
            hit = (lo[j] <= stop) if dr > 0 else (h[j] >= stop)
            if hit:
                px = stop if j == t + 1 else (min(o[j], stop) if dr > 0 else max(o[j], stop))
                ex, fee, kind, jx = px, TAKER, "stop", j
                break
            if pend is not None:
                thr = (h[j] > pend) if dr > 0 else (lo[j] < pend)
                ex, fee, kind, jx = (pend, MAKER, "target", j) if thr else (c[j], TAKER, "target_taker", j)
                break
            zj = zl[j]
            if zj == zj and ((dr > 0 and zj >= 0) or (dr < 0 and zj <= 0)):
                pend = c[j]
                if j + 1 >= n:
                    break
                continue
            if j == t + T:
                ex, fee, kind, jx = c[j], TAKER, "time", j
                break
        if ex is None:
            continue
        gross = dr * (math.log(ex) - math.log(e))
        pnl = gross - MAKER - fee
        out.append((int(ts[t]), pnl / unit, kind, t, gross / unit))
        busy_until = jx
    return out


def run_single(freq: str) -> None:
    first = {"1h": 2022, "5m": 2023}[freq]
    rows, kinfo = [], {}
    for s in UNIVERSE:
        t0 = time.time()
        d = bars(s, freq)
        if d is None:
            continue
        zz = single_z(d, first)
        kinfo[s] = zz["K"]
        cells = own_cells_4h(s) if freq == "1h" else None
        for k, T in VARIANTS:
            tr = sim_single(d, zz["z"], zz["sd"], k, T)
            if not tr:
                continue
            ets = np.array([x[0] for x in tr])
            cell = None
            if cells is not None:
                j = np.searchsorted(cells[0], ets + (3600 if freq == "1h" else 300), side="right") - 1
                cell = np.where(j >= 0, cells[1][np.maximum(j, 0)], "")
            rows.append(pl.DataFrame({"symbol": [s] * len(tr), "k": [k] * len(tr), "T": [T] * len(tr), "ts": ets,
                                      "R": [x[1] for x in tr], "kind": [x[2] for x in tr], "Rg": [x[4] for x in tr],
                                      "cell": cell if cell is not None else [""] * len(tr)}))
        sys.stdout.write(f"single {freq} {s}: K {[round(v['K'], 3) for v in zz['K'].values()]} {time.time() - t0:.1f}s\n")
        sys.stdout.flush()
    pl.concat(rows).write_parquet(root() / f"trades_single_{freq}.parquet")
    (root() / f"kf_params_{freq}.json").write_text(json.dumps(kinfo, indent=1), encoding="utf-8")


# ------------------------------------------------------------------ pairs


def kf_reg(y: np.ndarray, x: np.ndarray, qa: float, qb: float, R: float, a0: float, b0: float, p0: tuple[float, float],
           burn: int = 200) -> tuple[float, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """2-state random-walk regression KF. Returns (loglik after burn, innovation, S, beta_{t|t}, alpha_{t|t})."""
    n = len(y)
    a, b = a0, b0
    P00, P01, P11 = p0[0], 0.0, p0[1]
    yl, xl = y.tolist(), x.tolist()
    V = [math.nan] * n
    Sv = [math.nan] * n
    Bt = [math.nan] * n
    At = [math.nan] * n
    ll = 0.0
    for t in range(n):
        P00 += qa
        P11 += qb
        xt, yt = xl[t], yl[t]
        if xt != xt or yt != yt:
            Bt[t], At[t] = b, a
            continue
        v = yt - a - b * xt
        h0 = P00 + P01 * xt
        h1 = P01 + P11 * xt
        S = h0 + h1 * xt + R
        if S <= 0:
            S = 1e-18
        k0, k1 = h0 / S, h1 / S
        a += k0 * v
        b += k1 * v
        P00 -= k0 * h0
        P01 -= k0 * h1
        P11 -= k1 * h1
        V[t], Sv[t], Bt[t], At[t] = v, S, b, a
        if t >= burn:
            ll += -0.5 * (math.log(S) + v * v / S)
    return ll, np.array(V), np.array(Sv), np.array(Bt), np.array(At)


def fit_pair(y: np.ndarray, x: np.ndarray) -> dict:
    from scipy.optimize import minimize

    ok = np.isfinite(y) & np.isfinite(x)
    A = np.column_stack([np.ones(ok.sum()), x[ok]])
    b0 = np.linalg.lstsq(A[:500], y[ok][:500], rcond=None)[0]
    res0 = y[ok] - A @ np.linalg.lstsq(A, y[ok], rcond=None)[0]
    dv = float(np.var(np.diff(res0)))

    def nll(th):
        qa, qb, R = np.exp(th)
        return -kf_reg(y, x, qa, qb, R, b0[0], b0[1], (1e-2, 1e-2))[0]

    th0 = np.log([dv * 0.5, dv * 0.01, dv * 0.5])
    r = minimize(nll, th0, method="Nelder-Mead", options={"maxiter": 150, "xatol": 1e-2, "fatol": 1e-2})
    qa, qb, R = np.exp(r.x)
    return {"qa": float(qa), "qb": float(qb), "R": float(R), "a0": float(b0[0]), "b0": float(b0[1]), "nll": float(r.fun), "nit": int(r.nit)}


def align(da: dict, db: dict) -> dict:
    ia = {int(t): i for i, t in enumerate(da["ts"])}
    common = [t for t in db["ts"].tolist() if int(t) in ia]
    ja = np.array([ia[int(t)] for t in common])
    ib = {int(t): i for i, t in enumerate(db["ts"])}
    jb = np.array([ib[int(t)] for t in common])
    out = {"ts": np.array(common)}
    for k in ("open", "high", "low", "close"):
        out[f"A_{k}"] = da[k][ja]
        out[f"B_{k}"] = db[k][jb]
    return out


def choose_pairs() -> list[tuple[str, str]]:
    from ai_trader.research.xs_network_study import sectors

    sec = dict(zip(UNIVERSE, sectors(list(UNIVERSE)), strict=True))
    lo, hi = datetime(2023, 1, 1, tzinfo=UTC).timestamp(), datetime(2025, 1, 1, tzinfo=UTC).timestamp()
    R = {}
    for s in UNIVERSE:
        d = bars(s, "1h")
        m = (d["ts"] >= lo) & (d["ts"] < hi)
        R[s] = dict(zip(d["ts"][m].tolist(), np.r_[np.nan, np.diff(np.log(d["close"]))][m].tolist(), strict=True))
    cands = []
    for a, b in itertools.combinations(UNIVERSE, 2):
        if sec[a] != sec[b] or {a, b} in ({"ETHUSDT", "BTCUSDT"}, {"SOLUSDT", "ETHUSDT"}):
            continue
        ks = sorted(set(R[a]) & set(R[b]))
        if len(ks) < 5000:
            continue
        ra, rb = np.array([R[a][k] for k in ks]), np.array([R[b][k] for k in ks])
        ok = np.isfinite(ra) & np.isfinite(rb)
        cands.append((float(np.corrcoef(ra[ok], rb[ok])[0, 1]), b, a))   # A = later (smaller) name, B = earlier (larger)
    cands.sort(reverse=True)
    picked = [("ETHUSDT", "BTCUSDT"), ("SOLUSDT", "ETHUSDT")] + [(a, b) for _, b, a in cands[:6]]
    (root() / "pairs.json").write_text(json.dumps({"pairs": picked, "candidates": cands[:15], "sectors": sec}, indent=1), encoding="utf-8")
    return picked


def sim_pair(P: dict, z: np.ndarray, sdv: np.ndarray, beta: np.ndarray, k: float, T: int, test: np.ndarray) -> list[tuple]:
    lA, lB = np.log(P["A_close"]), np.log(P["B_close"])
    cA, cB = P["A_close"].tolist(), P["B_close"].tolist()
    hA, lAo, hB, lBo = P["A_high"].tolist(), P["A_low"].tolist(), P["B_high"].tolist(), P["B_low"].tolist()
    zl = z.tolist()
    n = len(cA)
    ts = P["ts"]
    out = []
    busy = -1
    for t in np.flatnonzero(test & np.isfinite(z) & (np.abs(z) >= k) & np.isfinite(sdv)).tolist():
        if t <= busy or t + 2 >= n:
            continue
        dr = -1.0 if zl[t] > 0 else 1.0    # +1 = long A, short beta B
        be = float(beta[t])
        # both legs must trade through at the next bar
        a_ok = (lAo[t + 1] < cA[t]) if dr > 0 else (hA[t + 1] > cA[t])
        b_ok = (hB[t + 1] > cB[t]) if dr * be > 0 else (lBo[t + 1] < cB[t])
        if not (a_ok and b_ok):
            continue
        lev = 1 + abs(be)
        delta = STOP_MULT * sdv[t]
        unit = delta + (MAKER + TAKER) * lev
        s_e = lA[t] - be * lB[t]
        ex_pnl, fee, kind, jx, pend = None, 0.0, "", None, None
        for j in range(t + 1, min(t + T + 1, n)):
            pj = dr * ((lA[j] - be * lB[j]) - s_e)
            if pend is not None:
                pa = (hA[j] > cA[pend]) if dr > 0 else (lAo[j] < cA[pend])
                pb = (lBo[j] < cB[pend]) if dr * be > 0 else (hB[j] > cB[pend])
                if pa and pb:
                    ex_pnl, fee, kind = dr * ((lA[pend] - be * lB[pend]) - s_e), MAKER, "target"
                else:
                    ex_pnl, fee, kind = pj, TAKER, "target_taker"
                jx = j
                break
            if pj <= -delta:
                ex_pnl, fee, kind, jx = pj, TAKER, "stop", j
                break
            zj = zl[j]
            if zj == zj and ((dr > 0 and zj >= 0) or (dr < 0 and zj <= 0)):
                pend = j
                continue
            if j == t + T:
                ex_pnl, fee, kind, jx = pj, TAKER, "time", j
                break
        if ex_pnl is None:
            continue
        pnl = ex_pnl - (MAKER + fee) * lev
        out.append((int(ts[t]), pnl / unit, kind, be, ex_pnl / unit))
        busy = jx
    return out


def run_pairs(freq: str) -> None:
    pairs = choose_pairs() if not (root() / "pairs.json").exists() else [tuple(p) for p in json.loads((root() / "pairs.json").read_text())["pairs"]]
    split = datetime(2025, 1, 1, tzinfo=UTC).timestamp()
    rows, params = [], {}
    for a, b in pairs:
        t0 = time.time()
        P = align(bars(a, freq), bars(b, freq))
        y, xr = np.log(P["A_close"]), np.log(P["B_close"])
        tr = P["ts"] < split - 86400
        if freq == "5m":
            tr &= P["ts"] >= split - 86400 - 120 * 86400
        xbar = float(np.mean(xr[tr]))
        x = xr - xbar
        idx = np.flatnonzero(tr)
        pr = fit_pair(y[idx], x[idx])
        start = idx[0]
        _, v, S, beta, _ = kf_reg(y[start:], x[start:], pr["qa"], pr["qb"], pr["R"], pr["a0"], pr["b0"], (1e-2, 1e-2))
        V = np.full(len(y), np.nan)
        SS = np.full(len(y), np.nan)
        BB = np.full(len(y), np.nan)
        V[start:], SS[start:], BB[start:] = v, S, beta
        with np.errstate(invalid="ignore"):
            z = V / np.sqrt(SS)
        test = P["ts"] >= split
        pr.update({"xbar": xbar, "beta_last": float(BB[-1]), "z_sd_test": float(np.nanstd(z[test]))})
        params[f"{a}/{b}"] = pr
        for k, T in VARIANTS:
            trd = sim_pair(P, z, np.sqrt(SS), BB, k, T, test)
            if trd:
                rows.append(pl.DataFrame({"pair": [f"{a}/{b}"] * len(trd), "k": [k] * len(trd), "T": [T] * len(trd), "ts": [x[0] for x in trd],
                                          "R": [x[1] for x in trd], "kind": [x[2] for x in trd], "beta": [x[3] for x in trd], "Rg": [x[4] for x in trd]}))
        sys.stdout.write(f"pairs {freq} {a}/{b}: q {pr['qa']:.2e} {pr['qb']:.2e} R {pr['R']:.2e} beta_last {pr['beta_last']:.2f} {time.time() - t0:.0f}s\n")
        sys.stdout.flush()
    pl.concat(rows).write_parquet(root() / f"trades_pairs_{freq}.parquet")
    (root() / f"pair_params_{freq}.json").write_text(json.dumps(params, indent=1), encoding="utf-8")


# ------------------------------------------------------------------ order flow (H-KFD3) and 1m


def flow15(sym: str) -> dict:
    from ai_trader.storage.lake import Lake

    f = Lake().read_asof("flow_bars", AS_OF, symbols=[sym], filters={"venue": "bybit", "freq": "15m"})
    return dict(zip(f["event_ts"].dt.epoch("s").to_list(), (f["buy_volume"] - f["sell_volume"]).to_list(), strict=True))


def minute_bars() -> dict[str, dict]:
    p = root() / "bars_1m_flow.npz"
    if p.exists():
        return np.load(p, allow_pickle=True)["d"].item()
    from ai_trader.storage.lake import Lake

    lake = Lake()
    inst = lake.read_asof("instruments", AS_OF, filters={"venue": "bybit"})
    crypto = set(inst.filter((pl.col("category") == "linear") & (pl.col("symbol_type").fill_null("") == ""))["symbol"].to_list())
    syms = sorted(set(lake.read_asof("flow_3s", AS_OF, start=AS_OF - timedelta(hours=1), filters={"category": "linear"}, columns=["symbol", "event_ts", "known_ts"])["symbol"].to_list()))
    syms = [s for s in syms if s in crypto and s != "SPYUSDT"]
    out = {}
    for i in range(0, len(syms), 8):
        b = syms[i:i + 8]
        f = lake.read_asof("flow_3s", AS_OF, symbols=b, filters={"category": "linear", "venue": "bybit"},
                           columns=["symbol", "event_ts", "open", "high", "low", "close", "buy_vol", "sell_vol", "ofi", "known_ts"])
        m = (f.sort("symbol", "event_ts").group_by_dynamic("event_ts", every="1m", group_by="symbol", closed="left", label="left")
             .agg(pl.col("open").drop_nulls().first(), pl.col("high").max(), pl.col("low").min(), pl.col("close").drop_nulls().last(),
                  (pl.col("buy_vol") - pl.col("sell_vol")).sum().alias("delta"), pl.col("ofi").fill_null(0).sum().alias("ofi")))
        for s in b:
            g = m.filter((pl.col("symbol") == s) & pl.col("close").is_not_null() & (pl.col("low") > 0)).sort("event_ts")
            if g.height < 3000:
                continue
            c = g["close"].to_numpy()
            out[s] = {"ts": g["event_ts"].dt.epoch("s").to_numpy(), "open": g["open"].to_numpy(), "high": g["high"].to_numpy(),
                      "low": g["low"].to_numpy(), "close": c, "delta": g["delta"].cast(pl.Float64).to_numpy(), "ofi": g["ofi"].cast(pl.Float64).to_numpy(),
                      "sig": _robust_sigma(np.r_[np.nan, np.diff(np.log(c))])}
        sys.stdout.write(f"1m batch {i}: {len(out)} symbols\n")
        sys.stdout.flush()
        del f, m
    np.savez_compressed(p, d=np.array(out, dtype=object))
    return out


def guards(dsig: np.ndarray, z: np.ndarray, n: int, ofi: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """allow (guardrail) and allow_validation arrays. dsig = per-bar taker delta (same bar index as z)."""
    cs = np.r_[0.0, np.cumsum(np.nan_to_num(dsig))]
    T = len(z)
    i = np.arange(T)
    win = cs[i + 1] - cs[np.maximum(i + 1 - n, 0)]
    q = max(1, n // 4)
    last = cs[i + 1] - cs[np.maximum(i + 1 - q, 0)]
    sgn = np.sign(np.nan_to_num(z))
    allow = ~(np.sign(win) == sgn)
    if ofi is not None:
        co = np.r_[0.0, np.cumsum(np.nan_to_num(ofi))]
        ow = co[i + 1] - co[np.maximum(i + 1 - n, 0)]
        allow &= ~(np.sign(ow) == sgn)
    val = allow & (np.sign(last) == -sgn)
    return allow, val


def run_flow() -> None:
    rows = []
    # BTC / ETH 15m with lake flow_bars (2026-05-11 .. 11-10); KF fitted on data before 2026-01-01
    for s in ("BTCUSDT", "ETHUSDT"):
        d = bars(s, "15m")
        zz = single_z(d, 2026)
        fl = flow15(s)
        delta = np.array([fl.get(int(t), np.nan) for t in d["ts"]])
        has = np.isfinite(delta)
        lo = d["ts"][has].min()
        z = np.where(d["ts"] >= lo, zz["z"], np.nan)
        allow, val = guards(delta, z, 4)
        for arm, al in (("all", None), ("guard", allow), ("validation", val)):
            for k, T in VARIANTS:
                tr = sim_single(d, z, zz["sd"], k, T, allow=al)
                rows += [{"src": "15m", "symbol": s, "arm": arm, "k": k, "T": T, "ts": x[0], "R": x[1], "kind": x[2]} for x in tr]
        sys.stdout.write(f"flow 15m {s}: K {zz['K']}\n")
    # 1m from flow_3s: fit on the first 3 days, test on the rest (all clean_holdout)
    M = minute_bars()
    kin = {}
    for s, d in M.items():
        ts = d["ts"]
        x = np.log(d["close"])
        fit_end = ts[0] + 3 * 86400
        tr = (ts < fit_end) & np.isfinite(d["sig"]) & (d["sig"] > 0)
        if tr.sum() < 2000:
            continue
        K, S = fit_K(x, d["sig"], tr)
        kin[s] = {"K": K, "S": S}
        with np.errstate(invalid="ignore", divide="ignore"):
            sd = d["sig"] * math.sqrt(S)
            z = np.where(ts >= fit_end, (x - ll_pred(x, K)) / sd, np.nan)
        allow, val = guards(d["delta"], z, 15, d["ofi"])
        allow_d, _ = guards(d["delta"], z, 15)
        for arm, al in (("all", None), ("guard", allow), ("guard_delta_only", allow_d), ("validation", val)):
            for k, T in VARIANTS:
                tr_ = sim_single(d, z, sd, k, T, allow=al)
                rows += [{"src": "1m", "symbol": s, "arm": arm, "k": k, "T": T, "ts": x_[0], "R": x_[1], "kind": x_[2]} for x_ in tr_]
    pl.DataFrame(rows).write_parquet(root() / "trades_flow.parquet")
    (root() / "kf_params_1m.json").write_text(json.dumps(kin, indent=1), encoding="utf-8")
    sys.stdout.write(f"1m symbols {len(kin)}, median K {np.median([v['K'] for v in kin.values()]):.3f}\n")


# ------------------------------------------------------------------ evaluation


def seg_of(ts: np.ndarray) -> np.ndarray:
    return np.where(ts < cut_s(), "llm_in_sample", "clean_holdout")


def daily_matrix(tr: pl.DataFrame, days: list) -> np.ndarray:
    """(days, variants) sums of R per entry day."""
    t = tr.with_columns((pl.col("ts") // 86400).alias("day"))
    M = np.zeros((len(days), len(VARIANTS)))
    di = {d: i for i, d in enumerate(days)}
    g = t.group_by("day", "k", "T").agg(pl.col("R").sum())
    vi = {v: i for i, v in enumerate(VARIANTS)}
    for dd, k, T, r in g.iter_rows():
        if dd in di and (k, T) in vi:
            M[di[dd], vi[(k, T)]] = r
    return M


def dsr(sr_all: np.ndarray, sr: float, ret: np.ndarray) -> float:
    """Deflated Sharpe (Bailey & Lopez de Prado 2014), per-period Sharpe; N = number of variants."""
    from scipy.stats import kurtosis, norm, skew

    N = len(sr_all)
    v = float(np.var(sr_all, ddof=1)) if N > 1 else 0.0
    g = 0.5772156649
    sr0 = math.sqrt(max(v, 1e-12)) * ((1 - g) * norm.ppf(1 - 1 / N) + g * norm.ppf(1 - 1 / (N * math.e)))
    Tn = len(ret)
    sk, ku = float(skew(ret)), float(kurtosis(ret, fisher=False))
    den = math.sqrt(max(1 - sk * sr + (ku - 1) / 4 * sr * sr, 1e-12))
    return float(norm.cdf((sr - sr0) * math.sqrt(Tn - 1) / den))


def pbo_cscv(M: np.ndarray, S: int = 10) -> float:
    T = M.shape[0] - M.shape[0] % S
    if T < S * 5:
        return float("nan")
    B = np.array_split(np.arange(T), S)
    lam = []
    for comb in itertools.combinations(range(S), S // 2):
        isx = np.concatenate([B[i] for i in comb])
        osx = np.concatenate([B[i] for i in range(S) if i not in comb])

        def shp(x):
            sd = x.std(0)
            return np.where(sd > 0, x.mean(0) / np.where(sd > 0, sd, 1), -np.inf)

        n_star = int(np.argmax(shp(M[isx])))
        oos = shp(M[osx])
        rank = (oos < oos[n_star]).sum() + 0.5 * ((oos == oos[n_star]).sum() - 1)
        w = (rank + 0.5) / (M.shape[1] + 0.0)
        w = min(max(w, 1e-6), 1 - 1e-6)
        lam.append(math.log(w / (1 - w)))
    return float(np.mean(np.array(lam) <= 0))


def summarise(tr: pl.DataFrame, key: str = "symbol") -> dict:
    out = {}
    tr = tr.with_columns(pl.Series("seg", seg_of(tr["ts"].to_numpy())))
    for sg in SEGS:
        t = tr.filter(pl.col("seg") == sg)
        if t.height == 0:
            out[sg] = None
            continue
        days = sorted(set((t["ts"] // 86400).to_list()))
        days = list(range(min(days), max(days) + 1))
        M = daily_matrix(t, days)
        srs = M.mean(0) / np.where(M.std(0) > 0, M.std(0), 1)
        var = {}
        for i, (k, T) in enumerate(VARIANTS):
            v = t.filter((pl.col("k") == k) & (pl.col("T") == T))
            per_day = v.group_by(pl.col("ts") // 86400).agg(pl.col("R").mean()).sort("ts")["R"].to_numpy()
            mu, tt, nd = nw_t(per_day)
            var[f"k{k}_T{T}"] = {"n": v.height, "mean_R": float(v["R"].mean()) if v.height else None,
                                 "mean_R_gross": float(v["Rg"].mean()) if v.height and "Rg" in v.columns else None, "day_mean_R": mu, "t": tt, "days": nd,
                                 "hit": float((v["R"] > 0).mean()) if v.height else None, "sharpe_daily_sum": float(srs[i]),
                                 "exits": dict(v.group_by("kind").len().iter_rows()) if v.height else {}}
        ip = VARIANTS.index(PRIMARY)
        out[sg] = {"variants": var, "primary": var[f"k{PRIMARY[0]}_T{PRIMARY[1]}"], "dsr_primary": dsr(srs, float(srs[ip]), M[:, ip]),
                   "dsr_best": dsr(srs, float(srs.max()), M[:, int(np.argmax(srs))]), "best_variant": f"k{VARIANTS[int(np.argmax(srs))][0]}_T{VARIANTS[int(np.argmax(srs))][1]}",
                   "pbo": pbo_cscv(M) if sg == "llm_in_sample" else None, "n_units": t[key].n_unique()}
    return out


def verdict_kfd(res_by_freq: dict) -> dict:
    ver = {}
    for sg in SEGS:
        sup = False
        allneg = True
        for f, r in res_by_freq.items():
            x = r.get(sg)
            if not x:
                continue
            p = x["primary"]
            ins = r.get("llm_in_sample") or {}
            pbo = ins.get("pbo")
            if p["mean_R"] is not None and p["mean_R"] > 0 and (p["t"] or 0) > 2 and x["dsr_primary"] > 0.95 and pbo is not None and pbo < 0.5:
                sup = True
            if p["mean_R"] is not None and p["mean_R"] > 0:
                allneg = False
        ver[sg] = "supports" if sup else ("contradicts" if allneg else "inconclusive")
    return ver


def boot_diff(a: pl.DataFrame, b: pl.DataFrame, B: int = 2000, seed: int = 0) -> dict:
    """mean R(a) - mean R(b), day-block bootstrap (days resampled jointly)."""
    rng = np.random.default_rng(seed)
    da = a.group_by(pl.col("ts") // 86400).agg(pl.col("R").sum().alias("s"), pl.len().alias("n"))
    db = b.group_by(pl.col("ts") // 86400).agg(pl.col("R").sum().alias("s"), pl.len().alias("n"))
    days = sorted(set(da["ts"].to_list()) | set(db["ts"].to_list()))
    A = dict(zip(da["ts"].to_list(), zip(da["s"].to_list(), da["n"].to_list(), strict=True), strict=True))
    Bd = dict(zip(db["ts"].to_list(), zip(db["s"].to_list(), db["n"].to_list(), strict=True), strict=True))
    sa = np.array([A.get(d, (0, 0))[0] for d in days])
    na = np.array([A.get(d, (0, 0))[1] for d in days])
    sb = np.array([Bd.get(d, (0, 0))[0] for d in days])
    nb = np.array([Bd.get(d, (0, 0))[1] for d in days])
    est = sa.sum() / max(na.sum(), 1) - sb.sum() / max(nb.sum(), 1)
    bs = []
    for _ in range(B):
        i = rng.integers(0, len(days), len(days))
        if na[i].sum() == 0 or nb[i].sum() == 0:
            continue
        bs.append(sa[i].sum() / na[i].sum() - sb[i].sum() / nb[i].sum())
    return {"diff": float(est), "lo": float(np.percentile(bs, 2.5)), "hi": float(np.percentile(bs, 97.5)), "n_a": int(na.sum()), "n_b": int(nb.sum()),
            "mean_a": float(sa.sum() / max(na.sum(), 1)), "mean_b": float(sb.sum() / max(nb.sum(), 1)), "days": len(days)}


def evaluate() -> dict:
    res: dict = {"H-KFD1": {}, "H-KFD2": {}, "H-KFD3": {}}
    for f in ("1h", "5m"):
        p = root() / f"trades_single_{f}.parquet"
        if p.exists():
            tr = pl.read_parquet(p)
            res["H-KFD1"][f] = summarise(tr)
            if f == "1h":
                pr = tr.filter((pl.col("k") == PRIMARY[0]) & (pl.col("T") == PRIMARY[1])).with_columns(pl.Series("seg", seg_of(tr.filter((pl.col("k") == PRIMARY[0]) & (pl.col("T") == PRIMARY[1]))["ts"].to_numpy())))
                res["H-KFD1"]["1h_by_cell"] = {f"{sg}|{c}": {"n": n, "mean_R": m} for sg, c, n, m in pr.group_by("seg", "cell").agg(pl.len(), pl.col("R").mean()).sort("seg", "cell").iter_rows()}
        p = root() / f"trades_pairs_{f}.parquet"
        if p.exists():
            tp = pl.read_parquet(p)
            res["H-KFD2"][f] = summarise(tp, key="pair")
            pr = tp.filter((pl.col("k") == PRIMARY[0]) & (pl.col("T") == PRIMARY[1]))
            res["H-KFD2"][f"{f}_by_pair"] = {a: {"n": n, "mean_R": m} for a, n, m in pr.group_by("pair").agg(pl.len(), pl.col("R").mean()).sort("pair").iter_rows()}
    p = root() / "trades_flow.parquet"
    if p.exists():
        fl = pl.read_parquet(p).with_columns(pl.Series("seg", seg_of(pl.read_parquet(p)["ts"].to_numpy())))
        one = fl.filter((pl.col("src") == "1m") & (pl.col("arm") == "all")).drop("src", "arm", "seg")
        res["H-KFD1"]["1m"] = summarise(one)
        for src in ("15m", "1m"):
            for sg in SEGS:
                x = fl.filter((pl.col("src") == src) & (pl.col("seg") == sg) & (pl.col("k") == PRIMARY[0]) & (pl.col("T") == PRIMARY[1]))
                if x.height == 0:
                    continue
                base = x.filter(pl.col("arm") == "all")
                r = {}
                for arm in sorted(set(x["arm"].to_list()) - {"all"}):
                    r[arm] = boot_diff(x.filter(pl.col("arm") == arm), base)
                # grid robustness: guard - all over every variant pooled
                g = fl.filter((pl.col("src") == src) & (pl.col("seg") == sg))
                r["grid_guard_minus_all"] = {f"k{k}_T{T}": (float(g.filter((pl.col("arm") == "guard") & (pl.col("k") == k) & (pl.col("T") == T))["R"].mean() or 0)
                                                           - float(g.filter((pl.col("arm") == "all") & (pl.col("k") == k) & (pl.col("T") == T))["R"].mean() or 0)) for k, T in VARIANTS}
                res["H-KFD3"][f"{src}|{sg}"] = r
    v1 = verdict_kfd({f: res["H-KFD1"][f] for f in ("1h", "5m", "1m") if f in res["H-KFD1"]})
    v2 = verdict_kfd({f: res["H-KFD2"][f] for f in ("1h", "5m") if f in res["H-KFD2"]})
    v3 = {}
    for sg in SEGS:
        rs = [res["H-KFD3"][k]["guard"] for k in res["H-KFD3"] if k.endswith(sg) and "guard" in res["H-KFD3"][k]]
        rs = [r for r in rs if r["n_a"] >= 30]
        if not rs:
            v3[sg] = "inconclusive"
        elif any(r["lo"] > 0 for r in rs):
            v3[sg] = "supports"
        elif all(r["diff"] <= 0 for r in rs):
            v3[sg] = "contradicts"
        else:
            v3[sg] = "inconclusive"
    res["verdicts"] = {"H-KFD1": v1, "H-KFD2": v2, "H-KFD3": v3}
    (root() / "eval.json").write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")
    return res


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="kalman_fv_study")
    ap.add_argument("cmd", choices=["single", "pairs", "flow", "evaluate"])
    ap.add_argument("--freq", default="1h")
    a = ap.parse_args(argv)
    if a.cmd == "single":
        run_single(a.freq)
    elif a.cmd == "pairs":
        run_pairs(a.freq)
    elif a.cmd == "flow":
        run_flow()
    else:
        r = evaluate()
        print(json.dumps(r["verdicts"], indent=1))


if __name__ == "__main__":
    main()
