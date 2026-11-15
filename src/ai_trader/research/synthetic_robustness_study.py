"""Synthetic robustness of parameter selection (H-SYN1, H-SYN2; research/plans/2026-11-11_strategy_program.md, B5).

Principle 1 (CLAUDE.md): generated data is used only to choose parameters of DETERMINISTIC strategies (the trend
zoo and a Donchian breakout grid). It never touches the LLM layer.

Design (fixed before running; diary 2026-11-10 "B5 start"):

Universe and data. trend_zoo_study.UNIVERSE (23 perps), Bybit, market_frame(funding=False, oi=False).
  * Historical P&L: native 4h frame, signals on the FULL history, evaluate.strategy_pnl conventions (sized at
    entry, risk units, taker + slippage per side, 0.2 rebalance band), common evaluation window from the first
    full UTC date after trend_zoo.lookback_bars(6) (as trend_zoo_study). Daily P&L is assigned to the date of
    the bar that REALISES it (bar t's P&L uses close_t -> close_{t+1}, dated by event_ts_{t+1}), so no holdout
    return enters an in-sample Sharpe. Segments: date < knowledge_cutoff -> llm_in_sample, else clean_holdout.
    Annualised Sharpe = mean / sd * sqrt(365) of daily P&L (> 30 days).
  * Generated P&L: from llm_in_sample 1h bars ONLY (event_ts < cutoff), with sim.resample: 200 paths per
    symbol and seed = 100 stationary-block-bootstrap paths (mean block = Politis-White on |r_1h|, capped at
    min(3 sqrt n, n/3)) + 100 regime-switching paths (hourly v0 cell chain fitted on the in-sample labels,
    blocks drawn within the simulated cell; a source bar carries its contemporaneous cell). Each path has the
    in-sample length (whole days), is rebuilt as 1h OHLC from resampled relative bars, aggregated to 4h, and
    evaluated with the same engine, warm-up and Sharpe definition (realisation-dated days on a synthetic
    calendar starting 00:00 UTC).

Grids.
  * zoo: the 21 trend_zoo variants (4h build).
  * bo:  Donchian breakout (strategies.trend.DonchianBreakout) entry N in {5, 10, 20, 40} days x exit
         M in {0.25 N, 0.5 N} (rounded, >= 1 bar), scale 3.0: 8 configurations, named bo_{N}d_x{frac}.

Selection (per symbol and grid, in-sample information only).
  hist = argmax historical in-sample Sharpe;  med = argmax median generated Sharpe;  q10 = argmax 10th
  percentile of generated Sharpe. Ties break to the first variant in grid order.

Tests.
  H-SYN1: deg(v) = Sharpe_IS(v) - Sharpe_HO(v). Per symbol d = deg(med) - deg(hist), averaged over the two
          grids (each grid also reported). Mean over symbols, 95% percentile CI from 10,000 symbol bootstrap
          resamples. Success (registry): mean < 0 and CI excludes 0.
  H-SYN2: zoo grid, per symbol e = Sharpe_HO(q10) - Sharpe_HO(hist). Mean over symbols, same bootstrap.
          Success: mean > 0 and CI excludes 0. (bo grid reported as secondary.)
  Verdict map (clean_holdout row): success met -> supports; point estimate on the wrong side of 0 ->
  contradicts; right sign but CI includes 0 -> inconclusive. The llm_in_sample rows are recorded as
  inconclusive and carry the in-sample diagnostics (the hypotheses are about the holdout outcome of an
  in-sample selection, so there is no in-sample verdict).
  Robustness (not verdict): a day-block bootstrap of the holdout (mean block 5 days, days resampled jointly
  across symbols, so the common crypto factor is kept), per-generator selections (100 SBB or 100 RS paths
  only), a uniformly-random-variant null (degradation falls mechanically for any non-champion choice), and the
  selection's stability across seeds.

Diagnostics. Generated vs real in-sample: excess kurtosis of 1h and 4h returns, ACF of |r| (1h lags 1/24/168,
4h lags 1/6), lag-1 ACF of r, v0 regime episode durations / stay probability / cell frequencies (re-labelled
on 10 generated paths per generator with research.regimes.label at 1h).

    uv run python -m ai_trader.research.synthetic_robustness_study run [--seeds 0,1,2] [--paths 200] [--workers 4]
    uv run python -m ai_trader.research.synthetic_robustness_study report [--record]
"""

from __future__ import annotations

import argparse
import json
import pickle
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl
from scipy.signal import lfilter

from ai_trader.core.config import load
from ai_trader.core.time import now
from ai_trader.research.evaluate import knowledge_cutoff, strategy_pnl
from ai_trader.research.frame import enrich_bars, market_frame
from ai_trader.research.regimes import label
from ai_trader.research.trend_zoo_study import UNIVERSE
from ai_trader.sim import resample as rs
from ai_trader.strategies.base import REGISTRY
from ai_trader.strategies.trend_zoo import (
    DONCH_DAYS,
    EMA_DAYS,
    MAC_DAYS,
    SLOPE_DAYS,
    TSMOM_DAYS,
    lookback_bars,
    mac_weights,
    slope_weights,
    tsmom_weights,
    zoo,
)

BPD = 6  # 4h bars per day
LAMBDA = 0.94
BAND = 0.2
MIN_DAYS = 30
BO_DAYS = (5, 10, 20, 40)
BO_EXIT = (0.25, 0.5)
N_DIAG_LABEL = 10
CHUNK = 100
ROOT_SCRATCH = Path(r"C:\Users\Asus\AppData\Local\Temp\claude\c--Users-Asus-Desktop-AI-Trader\31faef80-e320-4ca6-ab63-04fa013eeb03\scratchpad\b5")


# ====================================================================== vectorised engine (P paths x T bars)
def variant_specs() -> dict[str, tuple[str, str, dict]]:
    """name -> (grid, kind, params). The zoo part mirrors strategies.trend_zoo.zoo(6) exactly."""
    b = BPD
    v: dict[str, tuple[str, str, dict]] = {}
    for d in TSMOM_DAYS:
        v[f"tsmom_{d}d"] = ("zoo", "lin", {"w": tsmom_weights(d * b)})
    for f, s in MAC_DAYS:
        v[f"mac_{f}_{s}"] = ("zoo", "lin", {"w": mac_weights(f * b, s * b)})
    for d in DONCH_DAYS:
        n = d * b
        v[f"donch_{d}"] = ("zoo", "donch", {"n": n, "m": max(1, n // 2), "scale": 3.0})
    for f, m, s in EMA_DAYS:
        v[f"ema_{f}_{m}_{s}"] = ("zoo", "ema", {"fast": f * b, "mid": m * b, "slow": s * b, "k": float(np.sqrt(s * b) / 1.4)})
    for d in SLOPE_DAYS:
        v[f"slope_{d}d"] = ("zoo", "lin", {"w": slope_weights(d * b)})
    pt, pd_, pe = REGISTRY["tsmom"]().params, REGISTRY["donchian_breakout"]().params, REGISTRY["ema_stack"]().params
    v["cfg_tsmom"] = ("zoo", "tsmom_cfg", {"horizons": [int(h) for h in pt["horizons"]], "scale": float(pt["scale"])})
    v["cfg_ema_stack"] = ("zoo", "ema", {"fast": int(pe["fast"]), "mid": int(pe["mid"]), "slow": int(pe["slow"]), "k": float(pe["k"])})
    v["cfg_donchian_breakout"] = ("zoo", "donch", {"n": int(pd_["entry_window"]), "m": int(pd_["exit_window"]), "scale": float(pd_["scale"])})
    for d in BO_DAYS:
        for x in BO_EXIT:
            n = d * b
            v[f"bo_{d}d_x{x}"] = ("bo", "donch", {"n": n, "m": max(1, int(round(n * x))), "scale": 3.0})
    return v


def enrich_vec(close: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """sigma_next and fwd_ret exactly as frame.enrich_bars (EWMA lambda 0.94, warm-up 30), vectorised."""
    c = np.atleast_2d(close)
    p, t = c.shape
    ret = np.full((p, t), np.nan)
    ret[:, 1:] = np.diff(np.log(c), axis=1)
    r0 = np.nan_to_num(ret)
    warm = max(1, min(30, t - 1))
    sp = np.full((p, t), np.nan)
    if t > warm:
        var0 = np.mean(r0[:, :warm] ** 2, axis=1)
        sp[:, warm] = np.sqrt(var0)
        if t - 1 > warm:
            y, _ = lfilter([1 - LAMBDA], [1, -LAMBDA], r0[:, warm:t - 1] ** 2, axis=1, zi=(LAMBDA * var0)[:, None])
            sp[:, warm + 1:] = np.sqrt(y)
    sn = np.sqrt(LAMBDA * sp**2 + (1 - LAMBDA) * r0**2)
    fwd = np.full((p, t), np.nan)
    fwd[:, :-1] = ret[:, 1:]
    return sn, fwd


def _clean(x: np.ndarray) -> np.ndarray:
    return np.clip(np.where(np.isfinite(x), x, 0.0), -1.0, 1.0)


def _lag(x: np.ndarray, k: int) -> np.ndarray:
    out = np.full_like(x, np.nan)
    if k < x.shape[1]:
        out[:, k:] = x[:, : x.shape[1] - k]
    return out


def _rolling(x: np.ndarray, n: int, fn) -> np.ndarray:
    out = np.full_like(x, np.nan)
    if x.shape[1] >= n:
        out[:, n - 1:] = fn(np.lib.stride_tricks.sliding_window_view(x, n, axis=1), axis=-1)
    return out


def _ema(x: np.ndarray, span: int) -> np.ndarray:
    a = 2.0 / (span + 1.0)
    y, _ = lfilter([a], [1, -(1 - a)], x, axis=1, zi=((1 - a) * x[:, :1]))
    y[:, : span - 1] = np.nan
    return y


def sig_lin(c: np.ndarray, s: np.ndarray, w: np.ndarray) -> np.ndarray:
    n = len(w)
    y = lfilter(w[::-1], [1.0], np.log(c), axis=1)
    y[:, : n - 1] = np.nan
    norm = float(np.sqrt(np.sum(np.cumsum(w)[:-1] ** 2)))
    with np.errstate(divide="ignore", invalid="ignore"):
        z = y / (s * norm)
    return _clean(np.tanh(z))


def sig_tsmom_cfg(c: np.ndarray, s: np.ndarray, horizons: list[int], scale: float) -> np.ndarray:
    lc = np.log(c)
    with np.errstate(divide="ignore", invalid="ignore"):
        zs = np.stack([(lc - _lag(lc, k)) / (s * np.sqrt(k)) for k in horizons])
        m = zs.mean(axis=0)
    return _clean(np.tanh(m / scale))


def sig_ema(c: np.ndarray, s: np.ndarray, fast: int, mid: int, slow: int, k: float) -> np.ndarray:
    f, md, sl = _ema(c, fast), _ema(c, mid), _ema(c, slow)
    with np.errstate(invalid="ignore", divide="ignore"):
        align = np.where((f > md) & (md > sl), 1.0, np.where((f < md) & (md < sl), -1.0, 0.0))
        gap = np.abs(f - sl) / (c * s * k)
    gap = np.where(np.isfinite(gap), gap, np.nan)
    return _clean(align * np.tanh(gap))


def sig_donch(c: np.ndarray, h: np.ndarray, lo: np.ndarray, s: np.ndarray, n: int, m: int, scale: float) -> np.ndarray:
    """strategies.trend.DonchianBreakout, vectorised over paths. The event conditions are precomputed as
    (T, P) matrices; only the position / entry-level state machine runs bar by bar."""
    hi_n, lo_n = _lag(_rolling(h, n, np.max), 1), _lag(_rolling(lo, n, np.min), 1)
    hi_m, lo_m = _lag(_rolling(h, m, np.max), 1), _lag(_rolling(lo, m, np.min), 1)
    with np.errstate(invalid="ignore", divide="ignore"):
        valid = np.isfinite(hi_n) & np.isfinite(s) & (s > 0)
        up = valid & (c > hi_n)
        dn = valid & ~up & (c < lo_n)
        rest = valid & ~up & ~dn
        xl = (rest & (c < lo_m)).T.copy()
        xs = (rest & (c > hi_m)).T.copy()
    UP, DN, HN, LN = up.T.copy(), dn.T.copy(), hi_n.T.copy(), lo_n.T.copy()
    t_, p = UP.shape
    POS = np.empty((t_, p))
    LEV = np.empty((t_, p))
    pos = np.zeros(p)
    level = np.full(p, np.nan)
    for t in range(t_):
        u, d = UP[t], DN[t]
        np.copyto(level, HN[t], where=u & (pos != 1))
        np.copyto(level, LN[t], where=d & (pos != -1))
        ex = (xl[t] & (pos == 1)) | (xs[t] & (pos == -1))
        pos[u] = 1.0
        pos[d] = -1.0
        pos[ex] = 0.0
        POS[t] = pos
        LEV[t] = level
    POS, LEV = POS.T, LEV.T
    with np.errstate(invalid="ignore", divide="ignore"):
        dist = POS * np.log(c / LEV) / s
        out = np.where(valid & (POS != 0), POS * (0.5 + 0.5 * np.tanh(dist / scale)), 0.0)
    return _clean(out)


def signal(spec: tuple[str, str, dict], c: np.ndarray, h: np.ndarray, lo: np.ndarray, s: np.ndarray) -> np.ndarray:
    _, kind, p = spec
    if kind == "lin":
        return sig_lin(c, s, p["w"])
    if kind == "tsmom_cfg":
        return sig_tsmom_cfg(c, s, p["horizons"], p["scale"])
    if kind == "ema":
        return sig_ema(c, s, p["fast"], p["mid"], p["slow"], p["k"])
    if kind == "donch":
        return sig_donch(c, h, lo, s, p["n"], p["m"], p["scale"])
    raise ValueError(kind)


def pnl_vec(sig: np.ndarray, sigma: np.ndarray, fwd: np.ndarray, cost_bps: float, band: float = BAND) -> np.ndarray:
    """evaluate.strategy_pnl, vectorised over paths: per-bar net P&L in risk units.

    Re-sizing rule of strategy_pnl in one test: the held (signal, leverage) is replaced by the bar's when the
    signal differs from the held one AND (it is flat, the held one is flat, the sign flips, or the move exceeds
    the band); flat always maps to leverage 0 because its new leverage is 0 / sigma = 0."""
    s = np.nan_to_num(sig)
    ok = np.isfinite(sigma) & (sigma > 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        nl = np.where(ok, s / np.where(ok, sigma, 1.0), 0.0)
    S, NL = s.T.copy(), nl.T.copy()
    t_, p = S.shape
    LEVT = np.empty((t_, p))
    hs, hl = np.zeros(p), np.zeros(p)
    for t in range(t_):
        st = S[t]
        ch = (st != hs) & ((st * hs <= 0.0) | (np.abs(st - hs) > band))
        np.copyto(hs, st, where=ch)
        np.copyto(hl, NL[t], where=ch)
        LEVT[t] = hl
    lev = LEVT.T
    turn = np.abs(np.diff(np.concatenate([np.zeros((p, 1)), lev], axis=1), axis=1))
    return lev * np.nan_to_num(fwd) - turn * cost_bps / 1e4


def _cost_bps() -> float:
    c = load().risk["costs"]
    return float(c["taker_bps"] + c["slippage_bps"])


def _sharpe_cols(x: np.ndarray) -> np.ndarray:
    """Annualised Sharpe along axis -2 (days) of (..., days, variants)."""
    n = x.shape[-2]
    if n <= MIN_DAYS:
        return np.full(x.shape[:-2] + x.shape[-1:], np.nan)
    sd = x.std(axis=-2, ddof=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(sd > 0, x.mean(axis=-2) / sd * np.sqrt(365), np.nan)


# ====================================================================== per symbol
def historical(sym: str, as_of: datetime, specs: dict, check: bool = True) -> dict:
    f = market_frame(sym, "4h", as_of, funding=False, oi=False)
    c, h, lo = (f[k].to_numpy()[None, :] for k in ("close", "high", "low"))
    s, fwd = f["sigma_next"].to_numpy()[None, :], f["fwd_ret"].to_numpy()[None, :]
    names = list(specs)
    cost = _cost_bps()
    sigs = {v: signal(specs[v], c, h, lo, s) for v in names}
    pnl = np.vstack([pnl_vec(sigs[v], s, fwd, cost)[0] for v in names]).T  # (T, V)
    # equivalence check against the production zoo + strategy_pnl (zoo variants)
    ref = zoo(BPD) if check else {}
    max_sig, max_pnl = 0.0, 0.0
    for v, fn in ref.items():
        rsg = fn(f)
        max_sig = max(max_sig, float(np.max(np.abs(rsg.to_numpy() - sigs[v][0]))))
        rp = strategy_pnl(f, rsg)["pnl"].to_numpy()
        max_pnl = max(max_pnl, float(np.max(np.abs(rp - pnl[:, names.index(v)]))))
    lb = lookback_bars(BPD)
    start = f["event_ts"][lb].date() + timedelta(days=1)
    real_date = (f["event_ts"] + timedelta(hours=24 / BPD)).dt.date().to_numpy().astype("datetime64[D]")
    keep = real_date >= np.datetime64(start)
    dates, inv = np.unique(real_date[keep], return_inverse=True)
    daily = np.zeros((len(dates), len(names)))
    np.add.at(daily, inv, pnl[keep])
    cut = np.datetime64(knowledge_cutoff().date())
    is_m, ho_m = dates < cut, dates >= cut
    return {"names": names, "is_sharpe": _sharpe_cols(daily[is_m]), "ho_sharpe": _sharpe_cols(daily[ho_m]),
            "is_days": int(is_m.sum()), "ho_days": int(ho_m.sum()), "ho_daily": daily[ho_m], "ho_dates": dates[ho_m],
            "is_daily": daily[is_m], "is_dates": dates[is_m], "check_max_abs_signal": max_sig, "check_max_abs_pnl": max_pnl}


def _label_cells(close_1h: np.ndarray, opn: np.ndarray, high: np.ndarray, low: np.ndarray) -> list:
    n = len(close_1h)
    t0 = datetime(2000, 1, 1, tzinfo=UTC)
    ev = pl.datetime_range(t0, t0 + timedelta(hours=n - 1), interval="1h", eager=True, time_zone="UTC")
    bars = pl.DataFrame({"event_ts": ev, "close_ts": ev + timedelta(hours=1), "known_ts": ev + timedelta(hours=1),
                         "open": opn, "high": high, "low": low, "close": close_1h})
    return label(enrich_bars(bars), bar_hours=1.0)["cell"].to_list()


def _regime_stats(cells: list) -> dict:
    lab = [c for c in cells if c is not None]
    runs = rs.run_lengths(cells)
    pairs = [(a, b) for a, b in zip(cells[:-1], cells[1:], strict=True) if a is not None and b is not None]
    stay = float(np.mean([a == b for a, b in pairs])) if pairs else float("nan")
    states = [f"R{i}" for i in range(1, 10)]
    freq = np.array([lab.count(s) for s in states], dtype=float)
    dirs = [None if c is None else ("up" if c in ("R1", "R2", "R3") else "down" if c in ("R7", "R8", "R9") else "range") for c in cells]
    druns = rs.run_lengths(dirs)
    return {"cell_dur_median_h": float(np.median(runs)) if len(runs) else float("nan"),
            "cell_dur_mean_h": float(np.mean(runs)) if len(runs) else float("nan"),
            "dir_dur_mean_h": float(np.mean(druns)) if len(druns) else float("nan"),
            "stay_prob": stay, "cell_freq": (freq / max(freq.sum(), 1)).tolist()}


def _ret_stats(r1: np.ndarray, c4: np.ndarray) -> dict:
    """r1: (P, n) 1h log returns; c4: (P, m) 4h closes."""
    r4 = np.diff(np.log(c4), axis=1)
    a1 = rs.acf(np.abs(r1), [1, 24, 168])
    a4 = rs.acf(np.abs(r4), [1, 6])
    return {"kurt_1h": rs.excess_kurtosis(r1), "kurt_4h": rs.excess_kurtosis(r4), "acf_abs_1h_1": a1[:, 0],
            "acf_abs_1h_24": a1[:, 1], "acf_abs_1h_168": a1[:, 2], "acf_abs_4h_1": a4[:, 0], "acf_abs_4h_6": a4[:, 1],
            "acf_r_1h_1": rs.acf(r1, [1])[:, 0], "sd_1h": r1.std(axis=1)}


def run_symbol(sym: str, seed: int, n_paths: int, diag: bool = True, as_of: datetime | None = None) -> dict:
    cache = ROOT_SCRATCH / "cache" / f"{sym}_s{seed}_p{n_paths}.pkl"
    if cache.exists():
        return pickle.loads(cache.read_bytes())
    t_start = time.time()
    as_of = as_of or now()
    specs = variant_specs()
    names = list(specs)
    hist = historical(sym, as_of, specs) if seed == 0 else None
    cut = knowledge_cutoff()
    f1 = market_frame(sym, "1h", as_of, funding=False, oi=False)
    lab = label(f1, bar_hours=1.0)
    ins = (f1["event_ts"] < cut).to_numpy()
    f_is = f1.filter(pl.Series(ins))
    cells_is = lab["cell"].filter(pl.Series(ins)).to_list()
    rb = rs.relative_bars(*(f_is[k].to_numpy() for k in ("open", "high", "low", "close")))
    n_src = len(rb)
    n_out = n_src // 24 * 24
    pw = rs.politis_white_block(np.abs(rb.r))
    b = pw["b"]
    fit = rs.fit_transition(cells_is)
    st_ix = {s_: i for i, s_ in enumerate(fit["states"])}
    src_state = np.array([st_ix.get(c, -1) if c is not None else -1 for c in cells_is[1:]])
    rng = np.random.default_rng([seed, UNIVERSE.index(sym)])
    c0 = float(f_is["close"][0])
    cost = _cost_bps()
    lb = lookback_bars(BPD)
    d0 = lb // BPD + 1
    half = n_paths // 2
    sh, gen, rstats, regs = [], [], [], {"sbb": [], "rs": []}
    for g_name, n_g in (("sbb", half), ("rs", n_paths - half)):
        done = 0
        while done < n_g:
            k = min(CHUNK, n_g - done)
            if g_name == "sbb":
                idx = rs.stationary_indices(n_src, n_out, b, k, rng)
            else:
                chain = rs.simulate_chain(fit["P"], n_out, k, rng, start_probs=fit["freq"])
                idx = rs.regime_indices(src_state, chain, b, rng)
            p1 = rs.reconstruct(rb, idx, c0)
            p4 = rs.aggregate(p1, 4)
            c, h, lo = p4["close"], p4["high"], p4["low"]
            sn, fwd = enrich_vec(c)
            t4 = c.shape[1]
            nd = t4 // BPD
            out = np.empty((k, len(names)))
            for j, v in enumerate(names):
                pnl = pnl_vec(signal(specs[v], c, h, lo, sn), sn, fwd, cost)
                real = np.concatenate([np.zeros((k, 1)), pnl[:, :-1]], axis=1)  # realisation-dated
                daily = real[:, : nd * BPD].reshape(k, nd, BPD).sum(axis=2)[:, d0:]
                out[:, j] = _sharpe_cols(daily[:, :, None])[:, 0]
            sh.append(out)
            gen += [g_name] * k
            if diag:
                rstats.append(_ret_stats(rb.r[idx], c))
                need = N_DIAG_LABEL - len(regs[g_name])
                for i in range(min(need, k)):
                    regs[g_name].append(_regime_stats(_label_cells(p1["close"][i], p1["open"][i], p1["high"][i], p1["low"][i])))
            done += k
    res = {"symbol": sym, "seed": seed, "n_paths": n_paths, "names": names, "grid": [specs[v][0] for v in names],
           "syn_sharpe": np.vstack(sh), "gen": np.array(gen), "block": pw, "n_src_1h": n_src, "n_out_1h": n_out,
           "trans_P": fit["P"].tolist(), "states": fit["states"], "hist": hist, "secs": 0.0}
    if diag:
        r1 = rb.r[None, :]
        c4_real = rs.aggregate(rs.reconstruct(rb, np.arange(n_src)[None, :n_out], c0), 4)["close"]
        res["diag_real"] = {k_: float(v_[0]) for k_, v_ in _ret_stats(r1[:, :n_out], c4_real).items()}
        res["diag_real"].update(_regime_stats(cells_is))
        allst = {k_: np.concatenate([d[k_] for d in rstats]) for k_ in rstats[0]}
        res["diag_gen"] = {g: {k_: v_[np.array(gen) == g] for k_, v_ in allst.items()} for g in ("sbb", "rs")}
        res["diag_gen_regime"] = regs
    res["secs"] = time.time() - t_start
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_bytes(pickle.dumps(res))
    return res


def _job(a: tuple) -> str:
    sym, seed, n_paths = a
    r = run_symbol(sym, seed, n_paths, diag=(seed == 0))
    msg = f"{sym} seed {seed}: {r['secs']:.0f}s b={r['block']['b']:.0f} (raw {r['block']['b_raw']:.0f})"
    if r["hist"] is not None:
        msg += f" check sig {r['hist']['check_max_abs_signal']:.1e} pnl {r['hist']['check_max_abs_pnl']:.1e}"
    print(msg, flush=True)
    return msg


# ====================================================================== statistics
def _boot_mean_ci(x: np.ndarray, n: int = 10_000, seed: int = 7) -> tuple[float, float, float]:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    rng = np.random.default_rng(seed)
    m = rng.choice(x, size=(n, len(x)), replace=True).mean(axis=1)
    return float(x.mean()), float(np.quantile(m, 0.025)), float(np.quantile(m, 0.975))


def _verdict(mean: float, lo: float, hi: float, want: str) -> str:
    good = mean < 0 if want == "neg" else mean > 0
    excl = (hi < 0) if want == "neg" else (lo > 0)
    if good and excl:
        return "supports"
    return "inconclusive" if good else "contradicts"


def _argmax(x: np.ndarray) -> int:
    x = np.where(np.isfinite(x), x, -np.inf)
    return int(np.argmax(x))


def select(r: dict, grid: str, gen: str | None = None, paths: np.ndarray | None = None) -> dict:
    """Indices (into r['names']) chosen by hist / med / q10 within a grid."""
    gi = np.array([i for i, g in enumerate(r["grid"]) if g == grid])
    S = r["syn_sharpe"][:, gi]
    rows = np.ones(len(S), dtype=bool) if gen is None else (r["gen"] == gen)
    if paths is not None:
        S = S[paths]
        rows = rows[paths]
    S = S[rows]
    with np.errstate(all="ignore"):
        med = np.nanmedian(S, axis=0)
        q10 = np.nanquantile(S, 0.10, axis=0)
    out = {"med": int(gi[_argmax(med)]), "q10": int(gi[_argmax(q10)])}
    if r.get("hist") is not None:
        out["hist"] = int(gi[_argmax(r["hist"]["is_sharpe"][gi])])
    out["gi"] = gi
    out["syn_med"] = med
    out["syn_q10"] = q10
    return out


def _per_symbol(main: dict) -> list[dict]:
    rows = []
    for sym, r in main.items():
        h = r["hist"]
        row = {"symbol": sym, "is_days": h["is_days"], "ho_days": h["ho_days"]}
        for grid in ("zoo", "bo"):
            s = select(r, grid)
            isv, hov = h["is_sharpe"], h["ho_sharpe"]
            deg = isv - hov
            gi = s["gi"]
            for k in ("hist", "med", "q10"):
                row[f"{grid}_{k}"] = r["names"][s[k]]
                row[f"{grid}_{k}_is"] = float(isv[s[k]])
                row[f"{grid}_{k}_ho"] = float(hov[s[k]])
            row[f"{grid}_d1"] = float(deg[s["med"]] - deg[s["hist"]])
            row[f"{grid}_d1_q10"] = float(deg[s["q10"]] - deg[s["hist"]])
            row[f"{grid}_e2"] = float(hov[s["q10"]] - hov[s["hist"]])
            row[f"{grid}_e2_med"] = float(hov[s["med"]] - hov[s["hist"]])
            row[f"{grid}_null_d1"] = float(np.nanmean(deg[gi]) - deg[s["hist"]])
            row[f"{grid}_null_e2"] = float(np.nanmean(hov[gi]) - hov[s["hist"]])
            for g in ("sbb", "rs"):
                sg = select(r, grid, gen=g)
                row[f"{grid}_d1_{g}"] = float(deg[sg["med"]] - deg[s["hist"]])
                row[f"{grid}_e2_{g}"] = float(hov[sg["q10"]] - hov[s["hist"]])
                row[f"{grid}_med_{g}"] = r["names"][sg["med"]]
                row[f"{grid}_q10_{g}"] = r["names"][sg["q10"]]
            # rank agreement between historical in-sample Sharpe and the generated median
            a, bb = isv[gi], s["syn_med"]
            ok = np.isfinite(a) & np.isfinite(bb)
            row[f"{grid}_rank_corr_is_vs_synmed"] = float(np.corrcoef(np.argsort(np.argsort(a[ok])), np.argsort(np.argsort(bb[ok])))[0, 1]) if ok.sum() > 3 else float("nan")
        row["d1"] = 0.5 * (row["zoo_d1"] + row["bo_d1"])
        row["d1_sbb"] = 0.5 * (row["zoo_d1_sbb"] + row["bo_d1_sbb"])
        row["d1_rs"] = 0.5 * (row["zoo_d1_rs"] + row["bo_d1_rs"])
        row["null_d1"] = 0.5 * (row["zoo_null_d1"] + row["bo_null_d1"])
        rows.append(row)
    return rows


def _day_boot(main: dict, n: int = 2000, block: float = 5.0, seed: int = 11) -> dict:
    """Holdout days resampled jointly across symbols (stationary blocks); IS Sharpes and selections fixed."""
    syms = list(main)
    nd = min(main[s]["hist"]["ho_daily"].shape[0] for s in syms)
    rng = np.random.default_rng(seed)
    idx = rs.stationary_indices(nd, nd, block, n, rng)
    sel = {s: {g: select(main[s], g) for g in ("zoo", "bo")} for s in syms}
    d1 = np.zeros((n, len(syms)))
    e2 = np.zeros((n, len(syms)))
    for j, s in enumerate(syms):
        h = main[s]["hist"]
        x = h["ho_daily"][-nd:]
        xb = x[idx]  # (n, nd, V)
        ho = _sharpe_cols(xb)
        dd = 0.0
        for g in ("zoo", "bo"):
            ss = sel[s][g]
            degm = h["is_sharpe"][ss["med"]] - ho[:, ss["med"]]
            degh = h["is_sharpe"][ss["hist"]] - ho[:, ss["hist"]]
            dd = dd + 0.5 * (degm - degh)
            if g == "zoo":
                e2[:, j] = ho[:, ss["q10"]] - ho[:, ss["hist"]]
        d1[:, j] = dd
    m1, m2 = np.nanmean(d1, axis=1), np.nanmean(e2, axis=1)
    return {"n_days": nd, "block_days": block, "n_boot": n,
            "H-SYN1_ci": [float(np.quantile(m1, 0.025)), float(np.quantile(m1, 0.975))],
            "H-SYN2_ci": [float(np.quantile(m2, 0.025)), float(np.quantile(m2, 0.975))]}


def _seed_stability(all_runs: dict) -> dict:
    out = {}
    for grid in ("zoo", "bo"):
        agree_med, agree_q10, n = [], [], 0
        per = {}
        for sym, by_seed in all_runs.items():
            seeds = sorted(by_seed)
            if len(seeds) < 2:
                continue
            ms = [by_seed[sd]["names"][select(by_seed[sd], grid)["med"]] for sd in seeds]
            qs = [by_seed[sd]["names"][select(by_seed[sd], grid)["q10"]] for sd in seeds]
            agree_med.append(max(ms.count(x) for x in ms) / len(ms))
            agree_q10.append(max(qs.count(x) for x in qs) / len(qs))
            per[sym] = {"med": ms, "q10": qs}
            n += 1
        # path-bootstrap stability on seed 0 (resample the 200 paths)
        boot_med, boot_q10, by_sym_med = [], [], {}
        for sym, by_seed in all_runs.items():
            r = by_seed[min(by_seed)]
            base = select(r, grid)
            rng = np.random.default_rng(3)
            bm, bq = 0, 0
            for _ in range(200):
                pth = rng.integers(0, r["syn_sharpe"].shape[0], r["syn_sharpe"].shape[0])
                sb = select(r, grid, paths=pth)
                bm += sb["med"] == base["med"]
                bq += sb["q10"] == base["q10"]
            boot_med.append(bm / 200)
            by_sym_med[sym] = bm / 200
            boot_q10.append(bq / 200)
        out[grid] = {"n_symbols_multi_seed": n, "modal_share_med_mean": float(np.mean(agree_med)) if agree_med else float("nan"),
                     "modal_share_q10_mean": float(np.mean(agree_q10)) if agree_q10 else float("nan"),
                     "frac_symbols_all_seeds_agree_med": float(np.mean([a == 1 for a in agree_med])) if agree_med else float("nan"),
                     "frac_symbols_all_seeds_agree_q10": float(np.mean([a == 1 for a in agree_q10])) if agree_q10 else float("nan"),
                     "path_boot_same_med_by_symbol": by_sym_med, "path_boot_same_med_mean": float(np.mean(boot_med)), "path_boot_same_q10_mean": float(np.mean(boot_q10)),
                     "per_symbol": per}
    return out


def _moments(main: dict) -> dict:
    keys = ["kurt_1h", "kurt_4h", "acf_abs_1h_1", "acf_abs_1h_24", "acf_abs_1h_168", "acf_abs_4h_1", "acf_abs_4h_6", "acf_r_1h_1", "sd_1h"]
    rkeys = ["cell_dur_median_h", "cell_dur_mean_h", "dir_dur_mean_h", "stay_prob"]
    per, summ = {}, {}
    for sym, r in main.items():
        real = r["diag_real"]
        per[sym] = {"real": {k: real[k] for k in keys + rkeys}}
        for g in ("sbb", "rs"):
            dg = r["diag_gen"][g]
            q = {k: [float(np.nanquantile(dg[k], x)) for x in (0.05, 0.5, 0.95)] for k in keys}
            regs = r["diag_gen_regime"][g]
            for k in rkeys:
                v = np.array([x[k] for x in regs])
                q[k] = [float(np.nanquantile(v, x)) for x in (0.05, 0.5, 0.95)]
            cf = np.mean([x["cell_freq"] for x in regs], axis=0)
            q["cell_freq_tv"] = float(0.5 * np.abs(cf - np.array(real["cell_freq"])).sum())
            per[sym][g] = q
    for g in ("sbb", "rs"):
        summ[g] = {}
        for k in keys + rkeys:
            real = np.array([per[s]["real"][k] for s in per])
            med = np.array([per[s][g][k][1] for s in per])
            lo = np.array([per[s][g][k][0] for s in per])
            hi = np.array([per[s][g][k][2] for s in per])
            summ[g][k] = {"real_median": float(np.median(real)), "gen_median": float(np.median(med)),
                          "median_ratio_gen_over_real": float(np.median(med / real)),
                          "frac_real_inside_gen_5_95": float(np.mean((real >= lo) & (real <= hi)))}
        summ[g]["cell_freq_tv_median"] = float(np.median([per[s][g]["cell_freq_tv"] for s in per]))
    return {"per_symbol": per, "summary": summ}



# ====================================================================== overfitting diagnostics (descriptive, orchestrator request)
def pbo_cscv(perf: np.ndarray, n_blocks: int = 16) -> dict:
    """Probability of Backtest Overfitting by combinatorially symmetric cross-validation (Bailey, Borwein,
    Lopez de Prado & Zhu 2017).

    perf: (T, N) per-period P&L of N candidate configurations on the same T periods. The rows are cut into
    S = n_blocks contiguous blocks (trailing T mod S rows dropped). For each of the C(S, S/2) ways to pick
    half the blocks as IS (rest OOS), the configuration with the best IS Sharpe is found; its OOS Sharpe
    rank among the N (1 = worst, ties averaged) gives omega = rank / (N + 1) and lambda = log(omega / (1 - omega)).
    PBO = share of splits with lambda <= 0 (the IS winner lands at or below the OOS median). Also returned: the
    median logit, the mean relative OOS rank of the IS winner, and the OLS slope of the winner's OOS on IS Sharpe."""
    from itertools import combinations

    x = np.asarray(perf, dtype=float)
    x = np.where(np.isfinite(x), x, 0.0)
    t, n = x.shape
    if n_blocks % 2 or n_blocks < 2:
        raise ValueError("n_blocks must be even")
    L = t // n_blocks
    if L < 2:
        raise ValueError("too few rows for the number of blocks")
    xb = x[: L * n_blocks].reshape(n_blocks, L, n)
    s1, s2 = xb.sum(axis=1), (xb**2).sum(axis=1)  # (S, N)
    combos = np.array(list(combinations(range(n_blocks), n_blocks // 2)))
    ind = np.zeros((len(combos), n_blocks))
    ind[np.arange(len(combos))[:, None], combos] = 1.0

    def sharpe(w: np.ndarray) -> np.ndarray:
        cnt = w.sum(axis=1, keepdims=True) * L
        m = (w @ s1) / cnt
        v = ((w @ s2) / cnt - m**2) * cnt / (cnt - 1)
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(v > 0, m / np.sqrt(np.where(v > 0, v, 1.0)), -np.inf)

    sr_is, sr_oos = sharpe(ind), sharpe(1.0 - ind)
    idx = np.arange(len(combos))
    best = np.argmax(sr_is, axis=1)
    oos_best = sr_oos[idx, best]
    rank = (sr_oos < oos_best[:, None]).sum(axis=1) + 0.5 * ((sr_oos == oos_best[:, None]).sum(axis=1) - 1) + 1
    omega = rank / (n + 1)
    lam = np.log(omega / (1 - omega))
    isb = sr_is[idx, best]
    ok = np.isfinite(isb) & np.isfinite(oos_best)
    slope = float(np.polyfit(isb[ok], oos_best[ok], 1)[0]) if ok.sum() > 2 and np.std(isb[ok]) > 0 else float("nan")
    return {"pbo": float(np.mean(lam <= 0)), "logit_median": float(np.median(lam)), "n_splits": len(combos), "n_blocks": n_blocks,
            "block_len": L, "oos_rel_rank_mean_of_is_winner": float(np.mean(rank / n)), "degradation_slope": slope,
            "is_winner_counts": np.bincount(best, minlength=n).tolist()}


def cpcv_select(perf: np.ndarray, n_groups: int = 6, k_test: int = 2, embargo: int = 5) -> dict:
    """Combinatorial purged CV comparator (Lopez de Prado 2018 ch. 12; Arian et al. 2024) for rules with NO
    fitted parameters. Rows are cut into n_groups contiguous groups; each of the C(n_groups, k_test) splits
    takes k_test groups as test, dropping `embargo` rows at each test-group edge. Since the candidates are not
    trained, the train part carries no information and a candidate's score is the MEDIAN of its test-split
    Sharpes: a sub-period-robust selection on real data, the real-data analogue of the median over generated
    paths (it degenerates towards full-sample selection; reported as a comparator only)."""
    from itertools import combinations

    x = np.asarray(perf, dtype=float)
    x = np.where(np.isfinite(x), x, 0.0)
    t = x.shape[0]
    edges = np.linspace(0, t, n_groups + 1).astype(int)
    scores = []
    for test in combinations(range(n_groups), k_test):
        m = np.zeros(t, dtype=bool)
        for g in test:
            a, b = edges[g] + embargo, edges[g + 1] - embargo
            if b > a:
                m[a:b] = True
        scores.append(_sharpe_cols(x[m]))
    sc = np.nanmedian(np.vstack(scores), axis=0)
    return {"choice": _argmax(sc), "scores": sc, "n_splits": len(scores)}


def _is_daily(sym: str, r: dict) -> np.ndarray:
    if "is_daily" in r["hist"]:
        return r["hist"]["is_daily"]
    p = ROOT_SCRATCH / "cache" / f"{sym}_isdaily.pkl"
    if p.exists():
        return pickle.loads(p.read_bytes())
    d = historical(sym, now(), variant_specs(), check=False)
    if not np.allclose(d["is_sharpe"], r["hist"]["is_sharpe"], equal_nan=True):
        raise RuntimeError(f"{sym}: recomputed in-sample Sharpes differ from the cached run")
    p.write_bytes(pickle.dumps(d["is_daily"]))
    return d["is_daily"]


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 4:
        return float("nan")
    ra, rb = np.argsort(np.argsort(a[ok])), np.argsort(np.argsort(b[ok]))
    return float(np.corrcoef(ra, rb)[0, 1]) if np.std(ra) > 0 and np.std(rb) > 0 else float("nan")


def overfit_block(main: dict, rows: list[dict], stab: dict) -> dict:
    """Per symbol and grid on llm_in_sample daily P&L: PBO (CSCV, S = 16), a CPCV-selected comparator, and the
    relation of PBO / selection instability to the degradation reduction. All descriptive (not registered)."""
    out: dict = {"per_symbol": {}, "grid": {}}
    rix = {r["symbol"]: r for r in rows}
    for grid in ("zoo", "bo"):
        pbo, d1, e2, inst, cp_e2, cp_d1, cp_same = [], [], [], [], [], [], []
        for sym, r in main.items():
            gi = np.array([i for i, g in enumerate(r["grid"]) if g == grid])
            x = _is_daily(sym, r)[:, gi]
            pb = pbo_cscv(x, 16)
            cp = cpcv_select(x)
            h = r["hist"]
            ci = int(gi[cp["choice"]])
            s_ = select(r, grid)
            deg = h["is_sharpe"] - h["ho_sharpe"]
            out["per_symbol"].setdefault(sym, {})[grid] = {
                "pbo": pb["pbo"], "logit_median": pb["logit_median"], "degradation_slope": pb["degradation_slope"],
                "cpcv_choice": r["names"][ci], "cpcv_is": float(h["is_sharpe"][ci]), "cpcv_ho": float(h["ho_sharpe"][ci])}
            pbo.append(pb["pbo"])
            d1.append(rix[sym][f"{grid}_d1"])
            e2.append(rix[sym][f"{grid}_e2"])
            inst.append(1.0 - stab[grid]["path_boot_same_med_by_symbol"].get(sym, np.nan))
            cp_e2.append(float(h["ho_sharpe"][ci] - h["ho_sharpe"][s_["hist"]]))
            cp_d1.append(float(deg[ci] - deg[s_["hist"]]))
            cp_same.append(ci == s_["hist"])
        pbo, d1, e2, inst = map(np.asarray, (pbo, d1, e2, inst))
        hi = pbo >= np.median(pbo)
        m_cp, lo_cp, hi_cp = _boot_mean_ci(np.asarray(cp_e2))
        m_cd, lo_cd, hi_cd = _boot_mean_ci(np.asarray(cp_d1))
        out["grid"][grid] = {
            "pbo_median": float(np.median(pbo)), "pbo_mean": float(np.mean(pbo)), "frac_symbols_pbo_gt_0_5": float(np.mean(pbo > 0.5)),
            "spearman_pbo_vs_d1": _spearman(pbo, d1), "spearman_pbo_vs_e2": _spearman(pbo, e2),
            "spearman_instability_vs_d1": _spearman(inst, d1),
            "d1_mean_high_pbo_half": float(np.mean(d1[hi])), "d1_mean_low_pbo_half": float(np.mean(d1[~hi])),
            "e2_mean_high_pbo_half": float(np.mean(e2[hi])), "e2_mean_low_pbo_half": float(np.mean(e2[~hi])),
            "cpcv_ho_minus_hist": {"mean": m_cp, "ci95": [lo_cp, hi_cp]}, "cpcv_deg_minus_hist": {"mean": m_cd, "ci95": [lo_cd, hi_cd]},
            "cpcv_same_as_hist": float(np.mean(cp_same)),
        }
    return out


def report(seeds: tuple[int, ...] = (0, 1, 2), n_paths: int = 200) -> dict:
    all_runs: dict = {}
    for sym in UNIVERSE:
        for sd in seeds:
            p = ROOT_SCRATCH / "cache" / f"{sym}_s{sd}_p{n_paths}.pkl"
            if p.exists():
                all_runs.setdefault(sym, {})[sd] = pickle.loads(p.read_bytes())
    main = {s: v[0] for s, v in all_runs.items() if 0 in v}
    rows = _per_symbol(main)
    col = lambda k: np.array([r[k] for r in rows])  # noqa: E731
    stats = {}
    for k in ("d1", "zoo_d1", "bo_d1", "d1_sbb", "d1_rs", "null_d1", "zoo_d1_q10", "bo_d1_q10",
              "zoo_e2", "bo_e2", "zoo_e2_med", "bo_e2_med", "zoo_e2_sbb", "zoo_e2_rs", "zoo_null_e2", "bo_null_e2"):
        m, lo, hi = _boot_mean_ci(col(k))
        x = col(k)
        stats[k] = {"mean": m, "ci95": [lo, hi], "n": int(np.isfinite(x).sum()), "frac_neg": float(np.mean(x[np.isfinite(x)] < 0)),
                    "frac_zero": float(np.mean(x[np.isfinite(x)] == 0))}
    for grid in ("zoo", "bo"):
        for k in ("hist", "med", "q10"):
            for seg in ("is", "ho"):
                stats[f"{grid}_{k}_{seg}_mean"] = float(np.nanmean(col(f"{grid}_{k}_{seg}")))
        stats[f"{grid}_rank_corr_is_vs_synmed_mean"] = float(np.nanmean(col(f"{grid}_rank_corr_is_vs_synmed")))
        stats[f"{grid}_same_choice_med_hist"] = float(np.mean([r[f"{grid}_med"] == r[f"{grid}_hist"] for r in rows]))
        stats[f"{grid}_same_choice_q10_hist"] = float(np.mean([r[f"{grid}_q10"] == r[f"{grid}_hist"] for r in rows]))
    v1 = _verdict(stats["d1"]["mean"], *stats["d1"]["ci95"], want="neg")
    v2 = _verdict(stats["zoo_e2"]["mean"], *stats["zoo_e2"]["ci95"], want="pos")
    checks = {s: {"sig": r["hist"]["check_max_abs_signal"], "pnl": r["hist"]["check_max_abs_pnl"]} for s, r in main.items()}
    blocks = {s: r["block"] for s, r in main.items()}
    stab = _seed_stability(all_runs)
    return {"overfit": overfit_block(main, rows, stab), "generated": now().isoformat(), "seeds": list(seeds), "n_paths": n_paths, "n_symbols": len(main),
            "knowledge_cutoff": knowledge_cutoff().date().isoformat(), "cost_bps_per_side": _cost_bps(),
            "verdicts": {"H-SYN1": v1, "H-SYN2": v2}, "stats": stats, "day_bootstrap": _day_boot(main),
            "seed_stability": stab, "moments": _moments(main), "per_symbol": rows,
            "engine_check_max_abs": {"signal": max(c["sig"] for c in checks.values()), "pnl": max(c["pnl"] for c in checks.values())},
            "blocks": blocks, "runtime_secs": {s: r["secs"] for s, r in main.items()},
            "variant_names": main[next(iter(main))]["names"]}


def record_results(out: dict) -> list[str]:
    from ai_trader.research.hypotheses import record, results

    st = out["stats"]
    data = (f"bybit {out['n_symbols']}-symbol universe L; historical 4h frames (signals on full history); generator fed llm_in_sample "
            f"1h bars only (< {out['knowledge_cutoff']}); {out['n_paths']} paths/symbol (half SBB, half regime-switching); seed 0 for the "
            "verdict; report research/reports/synthetic_robustness_2026-11-11.md")
    note_map = ("Verdict mapping: the hypothesis concerns the clean_holdout outcome of a selection made on llm_in_sample information, so the "
                "verdict is recorded on clean_holdout; the llm_in_sample row is inconclusive by construction and carries in-sample diagnostics. ")
    mom = out["moments"]["summary"]
    diag = {"block_len_median_h": float(np.median([b["b"] for b in out["blocks"].values()])),
            "kurt_1h_ratio_sbb": mom["sbb"]["kurt_1h"]["median_ratio_gen_over_real"],
            "kurt_4h_ratio_sbb": mom["sbb"]["kurt_4h"]["median_ratio_gen_over_real"],
            "acf_abs_1h_24_ratio_sbb": mom["sbb"]["acf_abs_1h_24"]["median_ratio_gen_over_real"],
            "acf_abs_1h_168_ratio_sbb": mom["sbb"]["acf_abs_1h_168"]["median_ratio_gen_over_real"],
            "cell_dur_mean_ratio_rs": mom["rs"]["cell_dur_mean_h"]["median_ratio_gen_over_real"],
            "cell_dur_mean_ratio_sbb": mom["sbb"]["cell_dur_mean_h"]["median_ratio_gen_over_real"]}
    done = []
    for hid, key, want in (("H-SYN1", "d1", "neg"), ("H-SYN2", "zoo_e2", "pos")):
        prev = results(hid)
        segs_done = {r["segment"] for r in prev}
        if "clean_holdout" not in segs_done:
            m = {"mean_diff": st[key]["mean"], "ci95_symbol_boot": st[key]["ci95"], "n_symbols": st[key]["n"],
                 "ci95_day_block_boot": out["day_bootstrap"][f"{hid}_ci"], "frac_symbols_neg": st[key]["frac_neg"],
                 "frac_symbols_same_choice": st[key]["frac_zero"]}
            if hid == "H-SYN1":
                m.update({"zoo_mean": st["zoo_d1"]["mean"], "zoo_ci": st["zoo_d1"]["ci95"], "bo_mean": st["bo_d1"]["mean"],
                          "bo_ci": st["bo_d1"]["ci95"], "null_random_variant_mean": st["null_d1"]["mean"], "null_ci": st["null_d1"]["ci95"],
                          "sbb_only_mean": st["d1_sbb"]["mean"], "rs_only_mean": st["d1_rs"]["mean"],
                          "holdout_sharpe_med_minus_hist_zoo": st["zoo_e2_med"]["mean"]})
                note = note_map + ("Per symbol: degradation (IS minus holdout Sharpe) of the variant maximising the median Sharpe over generated "
                                   "paths minus that of the in-sample champion, averaged over the zoo (21) and breakout (8) grids; mean over symbols, "
                                   "10k symbol-bootstrap CI. Success: mean < 0 and CI excludes 0.")
            else:
                m.update({"bo_mean": st["bo_e2"]["mean"], "bo_ci": st["bo_e2"]["ci95"], "null_random_variant_mean": st["zoo_null_e2"]["mean"],
                          "sbb_only_mean": st["zoo_e2_sbb"]["mean"], "rs_only_mean": st["zoo_e2_rs"]["mean"],
                          "ho_sharpe_q10_mean": st["zoo_q10_ho_mean"], "ho_sharpe_hist_mean": st["zoo_hist_ho_mean"]})
                note = note_map + ("Per symbol: clean_holdout Sharpe of the zoo variant with the best 10th-percentile Sharpe across generated paths "
                                   "minus that of the best in-sample variant; mean over symbols, 10k symbol-bootstrap CI. Success: mean > 0, CI excludes 0.")
            record(hid, "clean_holdout", out["verdicts"][hid], m, note=note, data=data)
            done.append(f"{hid} clean_holdout {out['verdicts'][hid]}")
        if "llm_in_sample" not in segs_done:
            g = "zoo"
            pick = "med" if hid == "H-SYN1" else "q10"
            m = {"is_sharpe_hist_mean": st[f"{g}_hist_is_mean"], f"is_sharpe_{pick}_mean": st[f"{g}_{pick}_is_mean"],
                 "frac_same_choice": st[f"{g}_same_choice_{pick}_hist"], "rank_corr_is_vs_synmed": st[f"{g}_rank_corr_is_vs_synmed_mean"],
                 "seed_modal_share": out["seed_stability"][g][f"modal_share_{pick}_mean"],
                 "path_boot_same_choice": out["seed_stability"][g][f"path_boot_same_{pick}_mean"], **diag}
            record(hid, "llm_in_sample", "inconclusive", m, note=note_map + "In-sample diagnostics only (generator fit and selection behaviour).", data=data)
            done.append(f"{hid} llm_in_sample inconclusive")
    return done


# ====================================================================== CLI
def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("run")
    a.add_argument("--seeds", default="0")
    a.add_argument("--paths", type=int, default=200)
    a.add_argument("--workers", type=int, default=4)
    a.add_argument("--symbols", default="")
    b = sub.add_parser("report")
    b.add_argument("--seeds", default="0,1,2")
    b.add_argument("--paths", type=int, default=200)
    b.add_argument("--json", default="")
    b.add_argument("--record", action="store_true")
    x = ap.parse_args()
    if x.cmd == "run":
        syms = x.symbols.split(",") if x.symbols else UNIVERSE
        jobs = [(s, int(sd), x.paths) for sd in x.seeds.split(",") for s in syms]
        with ProcessPoolExecutor(max_workers=min(4, x.workers)) as ex:
            for _ in ex.map(_job, jobs):
                pass
    else:
        out = report(tuple(int(s) for s in x.seeds.split(",")), x.paths)
        if x.json:
            Path(x.json).write_text(json.dumps(out, indent=1, default=lambda o: o.tolist() if isinstance(o, np.ndarray) else str(o)), encoding="utf-8")
        print(json.dumps({"verdicts": out["verdicts"], "stats": {k: v for k, v in out["stats"].items() if isinstance(v, dict)},
                          "day_bootstrap": out["day_bootstrap"], "engine_check": out["engine_check_max_abs"]}, indent=1, default=str))
        if x.record:
            print(record_results(out))


if __name__ == "__main__":
    main()
