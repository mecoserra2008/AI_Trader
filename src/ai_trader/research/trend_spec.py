"""WS16 TREND: specification dossiers for "the trend signal" and "the trend trade" (+ the WS15 generative core).

Registered before any data run: research/hypotheses.yaml block WS16 (H-WS16-1..14); diary "WS16 design frozen before
results" (2026-11-26). Research only: no orders, no venue calls; the LLM layer is never tested here.

    PYTHONIOENCODING=utf-8 uv run python -m ai_trader.research.trend_spec weights      # pass 1: zoo signals + causal weights
    PYTHONIOENCODING=utf-8 uv run python -m ai_trader.research.trend_spec run [--workers 3]
    PYTHONIOENCODING=utf-8 uv run python -m ai_trader.research.trend_spec dossier      # signal / target-variable statistics
    PYTHONIOENCODING=utf-8 uv run python -m ai_trader.research.trend_spec synth [--workers 4] [--reps 100]
    PYTHONIOENCODING=utf-8 uv run python -m ai_trader.research.trend_spec analyse [--record]

Pieces
------
* Signals (4h bars, causal): the 21-variant zoo (strategies.trend_zoo), TSMOM sign variants, the H-KF1 adaptive LLT
  Kalman slope, and five ensembles of the zoo: equal weight (zoo_ew = the STRESS SURVIVES unit), inverse variance,
  Ledoit-Wolf minimum variance shrunk 50% to 1/N, family-balanced, sign vote; zoo_slow (drops the 1-3 day variants,
  data-informed). Ensemble weights are monthly, from the trailing 365 days of each variant's pooled per-bar P&L
  (s_{t-1} r_t / sigma_{t-1}) known before the month starts.
* Trade layer (`walk`): a generalisation of stress.walk_signal (identical in its base configuration; tested): entry
  threshold, exit threshold (flat / hysteresis / flip-only), sigma stop or structural stop (beyond the prior UTC day's /
  ISO week's extreme with a margin, inside [1, 4] sigma_4h), chandelier trail ratcheted at 4h closes, time stop, entry
  masks (regime cells, weekend, close hour), entry delay on the 1h path, order type (peg / taker / 5m peg fill model
  with skip or chase), latency.
* Statistics: paired stationary day bootstrap of mean-R differences, Romano-Wolf stepdown, Hansen SPA, deflated Sharpe,
  CSCV PBO, Driscoll-Kraay predictive regressions, verdicts (stress.verdict + G8 synthetic null).
* Synthetic null (WS15 core): stationary block bootstrap of whole 1h bars with mean block min(Politis-White on |r|, 24)
  and a zero-drift GARCH(1,1)-t; stylised facts and a real-vs-synthetic discriminator AUC.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field, replace
from datetime import UTC, date, datetime, timedelta
from pathlib import Path as FsPath
from typing import Any

import numpy as np
import polars as pl

from ai_trader.research import stress as S

ROOT = FsPath(__file__).resolve().parents[3]
OUT = ROOT / "data" / "research" / "ws16"
TMP = OUT / "tmp"
AS_OF = datetime(2026, 11, 26, 23, 0, tzinfo=UTC)
SEED = 20261127
BPD = 6
EXPECTED_REGIMES = ("R1", "R2", "R7", "R8")     # TSMom / EMAStack StrategySpec.expected_regimes (declared before tests)
FAST_ZOO = ("tsmom_1d", "tsmom_3d", "mac_1_7", "mac_3_14", "ema_1_4_16")
FAMILY = {"tsmom": "tsmom", "mac": "mac", "donch": "donch", "ema": "ema", "slope": "slope", "cfg": "cfg"}
SINGLE = ("tsmom_7d", "tsmom_14d", "tsmom_30d", "tsign_7d", "tsign_14d", "tsign_30d", "ema_1_4_16", "ema_4_16_64", "cfg_ema_stack",
          "donch_14", "donch_30", "slope_14d", "slope_30d", "mac_7_30", "kalman_llt")
ENSEMBLES = ("zoo_ew", "zoo_iv", "zoo_shr", "zoo_fam", "zoo_vote", "zoo_slow")
FIVE_MIN_START = datetime(2022, 1, 1, tzinfo=UTC)


# ====================================================================== signals
def zoo_signals(f4: pl.DataFrame, bpd: int = BPD) -> dict[str, np.ndarray]:
    from ai_trader.strategies.trend_zoo import zoo

    return {n: fn(f4).to_numpy().astype(float) for n, fn in zoo(bpd).items()}


def family_of(name: str) -> str:
    return name.split("_")[0]


def sign_signal(f4: pl.DataFrame, days: float, bpd: int = BPD) -> np.ndarray:
    from ai_trader.strategies.trend import arr
    from ai_trader.strategies.trend_zoo import _filter_z, tsmom_weights

    z = _filter_z(arr(f4, "close"), arr(f4, "sigma_next"), tsmom_weights(max(1, int(round(days * bpd)))))
    return np.where(np.isfinite(z), np.sign(z), 0.0)


def kalman_signal(f4: pl.DataFrame, q: float = 1e-3) -> np.ndarray:
    """H-KF1 adaptive LLT slope signal (breakout_cipher_kalman_study.kf_signals, adaptive branch) for any q."""
    from ai_trader.features.state_space import llt_filter
    from ai_trader.research.breakout_cipher_kalman_study import KF_SD_MIN, KF_WIN

    y = np.log(f4["close"].to_numpy())
    rs = f4["sigma_risk_next"].to_numpy() if "sigma_risk_next" in f4.columns else f4["sigma_next"].to_numpy()
    rs = np.where(np.isfinite(rs) & (rs > 0), rs, f4["sigma_next"].to_numpy())
    R = np.r_[np.nan, rs[:-1] ** 2]
    R[0] = R[1] if len(R) > 1 else np.nan
    rref = pl.Series(R).rolling_median(window_size=KF_WIN, min_samples=30).to_numpy()
    slope = np.where(np.isfinite(rref), llt_filter(y, R, q * rref, q_eta=0.0, p0_slope=1.0).slope, np.nan)
    sd = pl.Series(slope).rolling_std(window_size=KF_WIN, min_samples=KF_SD_MIN).to_numpy()
    with np.errstate(invalid="ignore", divide="ignore"):
        s = np.tanh(slope / sd)
    return np.where(np.isfinite(s), s, 0.0)


def single_signal(name: str, f4: pl.DataFrame, scale: float = 1.0, zs: dict[str, np.ndarray] | None = None) -> np.ndarray:
    """One single-rule signal; `scale` multiplies its lookbacks (the G1 perturbation)."""
    from ai_trader.strategies.trend import DonchianBreakout, EMAStack
    from ai_trader.strategies.trend_zoo import _clean, _linear, mac_weights, slope_weights, tsmom_weights

    if scale == 1.0 and zs is not None and name in zs:
        return zs[name]
    b = BPD
    fam, *rest = name.split("_")
    if name == "kalman_llt":
        return kalman_signal(f4, 1e-3 * scale)   # q is the key: x0.8 / x1.2
    if fam == "tsign":
        return sign_signal(f4, float(rest[0][:-1]) * scale)
    if fam == "tsmom":
        return _linear(name, tsmom_weights(max(1, round(float(rest[0][:-1]) * b * scale))))(f4).to_numpy()
    if fam == "slope":
        return _linear(name, slope_weights(max(3, round(float(rest[0][:-1]) * b * scale))))(f4).to_numpy()
    if fam == "mac":
        f, s_ = (max(1, round(int(x) * b * scale)) for x in rest)
        return _linear(name, mac_weights(f, s_))(f4).to_numpy()
    if fam == "donch":
        n = max(2, round(int(rest[0]) * b * scale))
        return _clean(DonchianBreakout(entry_window=n, exit_window=max(1, n // 2), scale=3.0)(f4), name).to_numpy()
    if fam == "ema":
        f, m, s_ = (max(1, round(int(x) * b * scale)) for x in rest)
        return _clean(EMAStack(fast=f, mid=m, slow=s_, k=float(np.sqrt(s_) / 1.4))(f4), name).to_numpy()
    if name == "cfg_ema_stack":
        from ai_trader.strategies import REGISTRY

        p = REGISTRY["ema_stack"]().params
        return _clean(EMAStack(fast=max(1, round(p["fast"] * scale)), mid=max(1, round(p["mid"] * scale)), slow=max(1, round(p["slow"] * scale)),
                               k=float(p["k"]))(f4), name).to_numpy()
    raise KeyError(name)


def bar_pnl(sig: np.ndarray, f4: pl.DataFrame) -> np.ndarray:
    """Per-bar vol-standardised P&L of holding s_{t-1} over bar t: s_{t-1} r_t / sigma_{t-1} (sigma_next known at t-1)."""
    c = f4["close"].to_numpy().astype(float)
    sg = f4["sigma_next"].to_numpy().astype(float)
    r = np.r_[np.nan, np.diff(np.log(c))]
    out = np.full(len(c), np.nan)
    out[1:] = sig[:-1] * r[1:] / sg[:-1]
    return out


def weight_table(daily: pl.DataFrame, names: list[str]) -> dict[str, dict[str, np.ndarray]]:
    """Monthly causal ensemble weights from pooled daily variant P&L (columns d, variant, pnl = mean over symbols).
    Weights for month m use days in [m - 365 d, m); < 180 days of history -> 1/N."""
    from sklearn.covariance import LedoitWolf

    piv = daily.pivot(on="variant", index="d", values="pnl").sort("d")
    piv = piv.select("d", *names)
    D = piv["d"].to_list()
    X = piv.select(names).to_numpy().astype(float)
    months = sorted({date(d.year, d.month, 1) for d in D})
    n = len(names)
    eq = np.full(n, 1.0 / n)
    out: dict[str, dict[str, np.ndarray]] = {"iv": {}, "shr": {}}
    darr = np.array(D, dtype="datetime64[D]")
    for m in months:
        lo = np.datetime64(m - timedelta(days=365))
        hi = np.datetime64(m)
        sel = (darr >= lo) & (darr < hi)
        Xi = X[sel]
        Xi = Xi[np.all(np.isfinite(Xi), axis=1)]
        key = m.isoformat()
        if len(Xi) < 180:
            out["iv"][key], out["shr"][key] = eq, eq
            continue
        v = Xi.var(axis=0, ddof=1)
        iv = np.where(v > 0, 1.0 / v, 0.0)
        out["iv"][key] = iv / iv.sum() if iv.sum() > 0 else eq
        cov = LedoitWolf().fit(Xi).covariance_
        try:
            w = np.linalg.solve(cov, np.ones(n))
        except np.linalg.LinAlgError:
            w = eq.copy()
        w = np.clip(w, 0.0, None)
        w = w / w.sum() if w.sum() > 0 else eq
        out["shr"][key] = 0.5 * w + 0.5 * eq
    return out


def ensembles(zs: dict[str, np.ndarray], f4: pl.DataFrame, wt: dict | None) -> dict[str, np.ndarray]:
    names = list(zs)
    M = np.vstack([zs[k] for k in names])
    out = {"zoo_ew": np.clip(M.mean(axis=0), -1, 1)}
    fams = [family_of(k) for k in names]
    uf = sorted(set(fams))
    fw = np.array([1.0 / (len(uf) * fams.count(f)) for f in fams])
    out["zoo_fam"] = np.clip(fw @ M, -1, 1)
    out["zoo_vote"] = np.sign(M).mean(axis=0)
    slow = [i for i, k in enumerate(names) if k not in FAST_ZOO]
    out["zoo_slow"] = np.clip(M[slow].mean(axis=0), -1, 1)
    if wt is not None:
        ms = f4["close_ts"].dt.offset_by("-1ms").dt.date().to_list()  # month of the decision time
        keys = [date(d.year, d.month, 1).isoformat() for d in ms]
        for kind in ("iv", "shr"):
            tab = wt[kind]
            W = np.vstack([np.asarray(tab.get(k, np.full(len(names), 1.0 / len(names)))) for k in keys])  # (T, n)
            out[f"zoo_{kind}"] = np.clip(np.einsum("tn,nt->t", W, M), -1, 1)
    return out


# ====================================================================== the generalised trade layer
@dataclass(frozen=True)
class Spec:
    name: str
    signal: str = "zoo_ew"
    theta: float = 0.3
    exit_thr: float = 0.0          # exit when s * side <= exit_thr (0 flat; -0.1 hysteresis; -0.3 flip-only)
    stop_k: float = 4.0
    stop: str = "sigma"            # sigma | pd | pw
    margin: float = 0.5            # structural margin in sigma_4h
    trail_k: float | None = None   # chandelier: best 4h close since entry x exp(-k sigma_entry)
    time_bars: int | None = None
    mask: str | None = None        # expected | diragree | no_r6 | weekend | no12
    delay_h: int = 0
    order: str = "peg"             # peg | taker | peg_skip | peg_chase
    grid: str = "base"             # base | offset
    conviction: bool = False
    family: str = "base"
    data_informed: bool = False
    keys: tuple[str, ...] = ("theta", "stop_k", "bpd")


def specs() -> dict[str, Spec]:
    out: dict[str, Spec] = {}

    def add(s: Spec) -> None:
        out[s.name] = s

    add(Spec("zoo_ew"))
    for e in ENSEMBLES[1:]:
        add(Spec(e, signal=e, family="weighting", data_informed=e == "zoo_slow"))
    for s_ in SINGLE:
        add(Spec(s_, signal=s_, family="single", keys=("theta", "stop_k", "lookback")))
    add(Spec("theta_0.2", theta=0.2, family="threshold"))
    add(Spec("theta_0.5", theta=0.5, family="threshold"))
    add(Spec("stop_2s", stop_k=2.0, family="stop", data_informed=True))
    add(Spec("stop_3s", stop_k=3.0, family="stop", data_informed=True))
    add(Spec("stop_pd", stop="pd", family="stop", keys=("theta", "margin", "bpd")))
    add(Spec("stop_pw", stop="pw", family="stop", keys=("theta", "margin", "bpd")))
    add(Spec("exit_hyst", exit_thr=-0.1, family="exit"))
    add(Spec("exit_flip", exit_thr=-0.3, family="exit"))
    add(Spec("exit_chand", trail_k=4.0, family="exit", keys=("theta", "stop_k", "bpd", "trail_k")))
    add(Spec("exit_time30", time_bars=180, family="exit", keys=("theta", "stop_k", "bpd", "time_bars")))
    add(Spec("time_no12", mask="no12", family="timing"))
    add(Spec("time_delay1h", delay_h=1, family="timing"))
    add(Spec("time_offset", grid="offset", family="timing"))
    add(Spec("order_taker", order="taker", family="order"))
    add(Spec("order_peg_skip", order="peg_skip", family="order"))
    add(Spec("order_peg_chase", order="peg_chase", family="order"))
    add(Spec("filt_expected", mask="expected", family="filter"))
    add(Spec("filt_diragree", mask="diragree", family="filter"))
    add(Spec("filt_no_r6", mask="no_r6", family="filter", data_informed=True))
    add(Spec("filt_weekend", mask="weekend", family="filter", data_informed=True))
    add(Spec("size_conviction", conviction=True, family="sizing"))
    extra = OUT / "extra_specs.json"   # frozen-rule combinations (H-WS16-13), written by the analysis, run once
    if extra.exists():
        for d in json.loads(extra.read_text(encoding="utf-8")):
            d = {**d, "keys": tuple(d.get("keys", ("theta", "stop_k", "bpd")))}
            add(Spec(**d))
    return out


def perturbed(s: Spec, key: str, f: float) -> tuple[Spec, float, float]:
    """(spec, bpd factor, lookback factor) for a one-at-a-time perturbation."""
    if key == "theta":
        return replace(s, theta=s.theta * f), 1.0, 1.0
    if key == "stop_k":
        return replace(s, stop_k=s.stop_k * f), 1.0, 1.0
    if key == "margin":
        return replace(s, margin=s.margin * f), 1.0, 1.0
    if key == "trail_k":
        return replace(s, trail_k=(s.trail_k or 4.0) * f), 1.0, 1.0
    if key == "time_bars":
        return replace(s, time_bars=max(1, round((s.time_bars or 180) * f))), 1.0, 1.0
    if key == "bpd":
        return s, f, 1.0
    if key == "lookback":
        return s, 1.0, f
    raise KeyError(key)


@dataclass
class Bars:
    """Everything the walker needs for one symbol and one 4h grid."""
    close_ms: np.ndarray
    close: np.ndarray
    sigma: np.ndarray           # sigma_risk_next per 4h bar
    path: S.Path                # 1h path
    pd_lo: np.ndarray           # prior complete UTC day low / high at each 4h close
    pd_hi: np.ndarray
    pw_lo: np.ndarray           # prior complete ISO week low / high
    pw_hi: np.ndarray
    hour: np.ndarray            # UTC hour of the 4h close
    cell: np.ndarray            # 1h v0 cell at the 4h close (object)
    weekend: np.ndarray
    m5: dict | None = None      # 5m arrays: ts (open ms), lo, hi, c
    extra: dict = field(default_factory=dict)


def prior_extremes(f1: pl.DataFrame, close_ms: np.ndarray) -> tuple[np.ndarray, ...]:
    """Prior complete UTC day / ISO week low and high at each decision time (point in time)."""
    d = (f1.select(pl.col("event_ts").dt.date().alias("d"), "high", "low").group_by("d").agg(pl.col("high").max(), pl.col("low").min()).sort("d"))
    w = (f1.select(pl.col("event_ts").dt.truncate("1w").dt.date().alias("w"), "high", "low").group_by("w")
         .agg(pl.col("high").max(), pl.col("low").min()).sort("w"))
    dmap_lo, dmap_hi = dict(zip(d["d"].to_list(), d["low"].to_list(), strict=True)), dict(zip(d["d"].to_list(), d["high"].to_list(), strict=True))
    wmap_lo, wmap_hi = dict(zip(w["w"].to_list(), w["low"].to_list(), strict=True)), dict(zip(w["w"].to_list(), w["high"].to_list(), strict=True))
    out = [np.full(len(close_ms), np.nan) for _ in range(4)]
    for i, ms in enumerate(close_ms):
        t = datetime.fromtimestamp(ms / 1000, UTC)
        today = t.date()
        pd_ = today - timedelta(days=1)
        wk = today - timedelta(days=today.weekday()) - timedelta(days=7)
        out[0][i] = dmap_lo.get(pd_, np.nan)
        out[1][i] = dmap_hi.get(pd_, np.nan)
        out[2][i] = wmap_lo.get(wk, np.nan)
        out[3][i] = wmap_hi.get(wk, np.nan)
    return tuple(out)


def make_bars(f4: pl.DataFrame, f1: pl.DataFrame, path: S.Path, m5: dict | None = None) -> Bars:
    from ai_trader.research.calendar import add_calendar

    cms = S._ms(f4["close_ts"])
    cells = f1.select(pl.col("close_ts").dt.convert_time_zone("UTC").dt.cast_time_unit("us").alias("_k"), pl.col("cell")).sort("_k")
    k4 = f4.select(pl.col("close_ts").dt.convert_time_zone("UTC").dt.cast_time_unit("us").alias("_t")).with_row_index("_i")
    cj = k4.sort("_t").join_asof(cells, left_on="_t", right_on="_k", strategy="backward").sort("_i")
    wk = add_calendar(pl.DataFrame({"event_ts": f4["close_ts"] - timedelta(hours=1)}))["tradfi_weekend"].to_numpy()
    pdl, pdh, pwl, pwh = prior_extremes(f1, cms)
    return Bars(cms, f4["close"].to_numpy().astype(float), f4["sigma_risk_next"].to_numpy().astype(float), path, pdl, pdh, pwl, pwh,
                f4["close_ts"].dt.hour().to_numpy(), np.array(cj["cell"].to_list(), dtype=object), np.asarray(wk, dtype=bool), m5)


def entry_ok(b: Bars, t: int, side: int, mask: str | None) -> bool:
    if mask is None:
        return True
    if "+" in mask:
        return all(entry_ok(b, t, side, m) for m in mask.split("+"))
    c = b.cell[t]
    if mask == "expected":
        return c in EXPECTED_REGIMES
    if mask == "diragree":
        if c is None:
            return True
        return not ((side > 0 and c in ("R7", "R8", "R9")) or (side < 0 and c in ("R1", "R2", "R3")))
    if mask == "no_r6":
        return c != "R6"
    if mask == "weekend":
        return not bool(b.weekend[t])
    if mask == "no12":
        return int(b.hour[t]) != 12
    raise KeyError(mask)


def struct_stop(b: Bars, t: int, side: int, e: float, sp: Spec) -> float:
    """Beyond the prior day's (then week's) adverse extreme by margin x sigma_4h, used if 1 <= distance <= stop_k sigma."""
    sg = float(b.sigma[t])
    levels = [(b.pd_lo[t], b.pd_hi[t]), (b.pw_lo[t], b.pw_hi[t])] if sp.stop == "pd" else [(b.pw_lo[t], b.pw_hi[t])]
    for lo, hi in levels:
        lvl = lo if side > 0 else hi
        if not np.isfinite(lvl):
            continue
        st = lvl * math.exp(-side * sp.margin * sg)
        k = side * math.log(e / st) / sg
        if 1.0 <= k <= sp.stop_k * (1 - S.TICK_MARGIN):
            return st
    return e * math.exp(-side * sp.stop_k * sg)


def peg_fill(b: Bars, ms: int, side: int, px: float, window_min: float = 15.0) -> tuple[bool, float, int] | None:
    """5m fill model: a limit at px fills if a 5m bar inside [ms, ms + window) trades 1 bp through it.
    Returns (filled, chase_price at window end, chase_ms) or None if no 5m data covers the window."""
    m = b.m5
    if m is None:
        return None
    i0 = int(np.searchsorted(m["ts"], ms, side="left"))
    nb = int(round(window_min / 5))
    if i0 + nb > len(m["ts"]) or m["ts"][i0] != ms or m["ts"][i0 + nb - 1] != ms + (nb - 1) * 300_000:
        return None
    lo, hi = m["lo"][i0:i0 + nb], m["hi"][i0:i0 + nb]
    filled = bool(np.any(lo <= px * (1 - 1e-4))) if side > 0 else bool(np.any(hi >= px * (1 + 1e-4)))
    return filled, float(m["c"][i0 + nb - 1]), int(ms + nb * 300_000)


def walk(sig: np.ndarray, b: Bars, sp: Spec, *, latency: float = 0.0) -> list[dict]:
    """Generalised stress.walk_signal (identical for Spec(name) defaults with order='peg')."""
    path = b.path
    n = len(sig)
    a = np.searchsorted(path.ts, b.close_ms, side="left")
    a = np.r_[a, len(path)]
    out: list[dict] = []
    pos, entry, stop, ets, ebar, et, sg_e, best, s_e, kind, pf0 = 0, 0.0, 0.0, 0, 0, 0, 0.0, 0.0, 0.0, "peg", -1
    stop0 = 0.0
    armed, stopped_side = True, 0
    sig = np.nan_to_num(np.asarray(sig, dtype=float))

    def px_at(t: int) -> float:
        if latency <= 0:
            return float(b.close[t])
        return path.fill(int(a[t]), float(b.close[t]), latency) if a[t] < len(path) else float("nan")

    def close_trade(x: float, xms: int, why: str, gap: bool = False) -> None:
        # R is denominated by the INITIAL stop (the risk taken at entry); a trailed stop only moves the exit
        out.append({"side": pos, "entry_ms": ets, "exit_ms": xms, "entry": entry, "stop": stop0, "exit": x, "why": why, "gap": gap,
                    "kind": kind, "ev": float("nan"), "s_entry": s_e, "entry_hour": int(b.hour[et]) if et < len(b.hour) else -1, "peg_filled": pf0})

    for t in range(n - 1):
        st = sig[t]
        if pos != 0 and (st * pos <= sp.exit_thr or (sp.time_bars is not None and t - et >= sp.time_bars)):
            x = px_at(t)
            if np.isfinite(x):
                close_trade(x, int(b.close_ms[t] + latency * 60_000), "signal" if st * pos <= sp.exit_thr else "time")
            pos = 0
        if not armed and st * stopped_side <= 0:
            armed = True
        if pos == 0 and armed and abs(st) >= sp.theta and np.isfinite(b.sigma[t]) and b.sigma[t] > 0:
            side = 1 if st > 0 else -1
            if entry_ok(b, t, side, sp.mask):
                e, e_ms, e_bar, k_ = px_at(t), int(b.close_ms[t]), int(a[t]), "peg"
                if sp.order == "taker":
                    k_ = "taker"
                if sp.delay_h > 0:
                    j = int(a[t]) + sp.delay_h - 1
                    e = float(path.c[j]) if j < len(path) else float("nan")
                    e_ms, e_bar = int(b.close_ms[t] + sp.delay_h * 3_600_000), int(a[t]) + sp.delay_h
                if sp.order in ("peg_skip", "peg_chase"):
                    pf = peg_fill(b, e_ms, side, e)
                    if pf is None:
                        e = float("nan")            # outside the 5m window: not evaluated
                    elif not pf[0]:
                        if sp.order == "peg_skip":
                            e = float("nan")        # unfilled: stay flat, retry at the next close
                        else:
                            e, e_ms, k_ = pf[1], pf[2], "taker"
                            e_bar = int(np.searchsorted(path.ts, e_ms - 3_600_000 + 1, side="left"))  # the 1h bar containing the chase
                if np.isfinite(e) and e > 0:
                    pos, entry, ets, ebar, et, kind = side, e, e_ms, e_bar, t, k_
                    sg_e, best, s_e = float(b.sigma[t]), e, float(abs(st))
                    pf = peg_fill(b, int(b.close_ms[t]), side, float(b.close[t])) if b.m5 is not None else None
                    pf0 = -1 if pf is None else int(pf[0])
                    stop = struct_stop(b, t, pos, e, sp) if sp.stop in ("pd", "pw") else e * math.exp(-pos * sp.stop_k * sg_e)
                    stop0 = stop
        if pos != 0 and sp.trail_k is not None and t > et:
            best = max(best, b.close[t]) if pos > 0 else min(best, b.close[t])
            ns = best * math.exp(-pos * sp.trail_k * sg_e)
            stop = max(stop, ns) if pos > 0 else min(stop, ns)
        if pos != 0:
            for i in range(max(int(a[t]), ebar), int(a[t + 1])):
                if (pos > 0 and path.lo[i] <= stop) or (pos < 0 and path.h[i] >= stop):
                    x, xms, gap = S._stop_fill(path, i, pos, stop, "price")
                    close_trade(x, xms, "stop", gap)
                    armed, stopped_side, pos = False, pos, 0
                    break
    if pos != 0 and len(path):
        close_trade(float(path.c[-1]), int(path.ts[-1] + path.bar_min * 60_000), "end")
    return out


# ====================================================================== per-symbol data
def offset_frame(f1: pl.DataFrame, hours: int = 2) -> pl.DataFrame:
    """4h bars on the (02, 06, ..) UTC grid from complete groups of four 1h bars, enriched like market_frame."""
    from ai_trader.research.frame import add_robust_vol, enrich_bars

    off = timedelta(hours=hours)
    g = (f1.with_columns(((pl.col("event_ts") - off).dt.truncate("4h") + off).alias("_g"))
         .group_by("_g").agg(pl.col("open").first(), pl.col("high").max(), pl.col("low").min(), pl.col("close").last(),
                             pl.col("volume").sum(), pl.col("turnover").sum(), pl.len().alias("_n"))
         .filter(pl.col("_n") == 4).sort("_g")
         .with_columns(pl.col("_g").alias("event_ts"), (pl.col("_g") + pl.duration(hours=4)).alias("close_ts"))
         .with_columns(pl.col("close_ts").alias("known_ts"))
         .select("event_ts", "close_ts", "known_ts", "open", "high", "low", "close", "volume", "turnover"))
    return add_robust_vol(enrich_bars(g))


def load_5m(sym: str) -> dict | None:
    fs = sorted((ROOT / "data" / "research" / "bos" / "bars" / "5m").glob(f"{sym}_20*.parquet"))
    ext = ROOT / "data" / "research" / "intraday_geometry" / "bars" / f"{sym}_ext.parquet"
    if ext.exists():
        fs.append(ext)
    if not fs:
        return None
    d = (pl.concat([pl.read_parquet(f).select("event_ts", "high", "low", "close") for f in fs], how="vertical_relaxed")
         .unique("event_ts", keep="last").sort("event_ts").filter(pl.col("event_ts") < AS_OF))
    return {"ts": S._ms(d["event_ts"]), "lo": d["low"].to_numpy().astype(float), "hi": d["high"].to_numpy().astype(float),
            "c": d["close"].to_numpy().astype(float)}


def trades_frame(tr: list[dict], ctx: S.Ctx, variant: str, exec_: str) -> pl.DataFrame:
    if not tr:
        return pl.DataFrame()
    df = S.annotate(tr, ctx, "ws16", variant)
    cols = [c for c in S.TRADE_COLS if c != "unit"] + ["s_entry", "entry_hour", "peg_filled"]
    return df.select([pl.col(c) for c in cols]).with_columns(
        pl.lit(exec_).alias("exec"), pl.col("side").cast(pl.Int8), pl.col("ev").cast(pl.Float64), pl.col("cell").cast(pl.Utf8),
        pl.col("gap").cast(pl.Boolean), pl.col("kind").cast(pl.Utf8))


def pass1_symbol(sym: str) -> dict:
    """Zoo signals + per-bar P&L -> tmp/sig_<sym>.parquet and the daily P&L per variant (for the causal weights)."""
    ctx = S.load_ctx(sym, AS_OF, need_rb=False)
    if ctx is None:
        return {"symbol": sym, "skipped": True}
    f4 = ctx.f4
    zs = zoo_signals(f4)
    cols = {"close_ts": f4["close_ts"]}
    for k, v in zs.items():
        cols[k] = pl.Series(k, v.astype(np.float32))
    TMP.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(cols).write_parquet(TMP / f"sig_{sym}.parquet")
    d = f4["close_ts"].dt.offset_by("-1ms").dt.date()
    frames = []
    for k, v in zs.items():
        p = bar_pnl(v, f4)
        frames.append(pl.DataFrame({"d": d, "variant": k, "pnl": p}).filter(pl.col("pnl").is_finite()).group_by("d", "variant").agg(pl.col("pnl").sum()))
    out = pl.concat(frames).with_columns(pl.lit(sym).alias("symbol"))
    out.write_parquet(TMP / f"daily_{sym}.parquet")
    return {"symbol": sym, "rows": out.height}


def build_weights() -> dict:
    fs = sorted(TMP.glob("daily_*.parquet"))
    daily = pl.concat([pl.read_parquet(f) for f in fs]).group_by("d", "variant").agg(pl.col("pnl").mean())
    from ai_trader.strategies.trend_zoo import ZOO

    names = list(ZOO)
    wt = weight_table(daily, names)
    js = {k: {m: [float(x) for x in w] for m, w in v.items()} for k, v in wt.items()}
    (OUT / "weights.json").write_text(json.dumps({"names": names, **js}, indent=0), encoding="utf-8")
    return {"months": len(js["iv"])}


def load_weights() -> dict | None:
    p = OUT / "weights.json"
    if not p.exists():
        return None
    w = json.loads(p.read_text(encoding="utf-8"))
    return {k: {m: np.asarray(v) for m, v in w[k].items()} for k in ("iv", "shr")}


def all_signals(f4: pl.DataFrame, wt: dict | None, bpd: int = BPD, zs: dict | None = None, which: set[str] | None = None) -> dict[str, np.ndarray]:
    zs = zs if zs is not None else zoo_signals(f4, bpd)
    out = dict(ensembles(zs, f4, wt))
    for s_ in SINGLE:
        if which is None or s_ in which:
            out[s_] = single_signal(s_, f4, 1.0, zs if bpd == BPD else None)
    return out


def run_symbol(sym: str, perturb: bool, only: list[str] | None = None, out_name: str = "trades") -> dict:
    """All specs x exec variants (base, lat5) on one symbol; D8 symbols also get the one-at-a-time +-20% perturbations."""
    t0 = time.time()
    ctx = S.load_ctx(sym, AS_OF, need_rb=False)
    if ctx is None:
        return {"symbol": sym, "skipped": True}
    wt = load_weights()
    sp_all = specs()
    if only:
        sp_all = {k: v for k, v in sp_all.items() if k in only}
    m5 = load_5m(sym)
    sigf = TMP / f"sig_{sym}.parquet"
    zs = None
    if sigf.exists():
        z = pl.read_parquet(sigf)
        if z.height == ctx.f4.height:
            zs = {k: z[k].to_numpy().astype(float) for k in z.columns if k != "close_ts"}
    sigs = all_signals(ctx.f4, wt, zs=zs)
    bars = make_bars(ctx.f4, ctx.f1, ctx.path, m5)
    f4o = offset_frame(ctx.f1)
    bars_o = make_bars(f4o, ctx.f1, ctx.path, m5)
    sig_o = ensembles(zoo_signals(f4o), f4o, None)["zoo_ew"]
    frames = []
    for name, sp in sp_all.items():
        b, s_ = (bars_o, sig_o) if sp.grid == "offset" else (bars, sigs[sp.signal])
        for ex, lat in (("base", 0.0), ("lat5", 5.0)):
            frames.append(trades_frame(walk(s_, b, sp, latency=lat), ctx, name, ex))
    if perturb:
        cache: dict[tuple[str, float], dict[str, np.ndarray]] = {}
        for name, sp in sp_all.items():
            for key in sp.keys:
                if key == "stop_k" and sp.stop != "sigma":
                    continue
                for f in (0.8, 1.2):
                    q, fb, fl = perturbed(sp, key, f)
                    if sp.grid == "offset":
                        b = bars_o
                        if fb != 1.0:
                            ck = ("offset", fb)
                            if ck not in cache:
                                cache[ck] = {"zoo_ew": ensembles(zoo_signals(f4o, round(BPD * fb)), f4o, None)["zoo_ew"]}
                            s_ = cache[ck]["zoo_ew"]
                        else:
                            s_ = sig_o
                    else:
                        b = bars
                        if fb != 1.0:
                            ck = ("bpd", fb)
                            if ck not in cache:
                                cache[ck] = all_signals(ctx.f4, wt, round(BPD * fb), which=set())
                            s_ = cache[ck][sp.signal]
                        elif fl != 1.0:
                            s_ = single_signal(sp.signal, ctx.f4, fl)
                        else:
                            s_ = sigs[sp.signal]
                    frames.append(trades_frame(walk(s_, b, q), ctx, f"{name}|{key}@{f}", "base"))
    frames = [f for f in frames if f.height]
    df = pl.concat(frames, how="vertical_relaxed") if frames else pl.DataFrame()
    od = OUT / out_name
    od.mkdir(parents=True, exist_ok=True)
    if df.height:
        df.write_parquet(od / f"{sym}.parquet")
    return {"symbol": sym, "trades": df.height, "seconds": round(time.time() - t0, 1)}


# ====================================================================== statistics
def day_grid(t: pl.DataFrame, r: np.ndarray, days: np.ndarray, by: str = "entry_ts") -> tuple[np.ndarray, np.ndarray]:
    """(n, sumR) per day on a fixed grid of days (datetime64[D])."""
    if t.height == 0:
        return np.zeros(len(days)), np.zeros(len(days))
    d = t[by].dt.date().to_numpy().astype("datetime64[D]")
    ix = np.searchsorted(days, d)
    ix = np.clip(ix, 0, len(days) - 1)
    n = np.bincount(ix, minlength=len(days)).astype(float)
    s = np.bincount(ix, weights=r, minlength=len(days))
    return n, s


def boot_idx(n_days: int, n_boot: int = S.N_BOOT, block: float = S.BLOCK_DAYS, seed: int = SEED) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return S.stationary_boot_idx(n_days, n_boot, block, rng)


def paired(base: tuple[np.ndarray, np.ndarray], vars_: dict[str, tuple[np.ndarray, np.ndarray]], idx: np.ndarray) -> dict[str, dict]:
    """Paired differences of mean R/trade (variant - base) with a stationary day bootstrap; Romano-Wolf and SPA."""
    nb, sb = base
    mb = sb.sum() / max(nb.sum(), 1)
    Bb = sb[idx].sum(1) / np.maximum(nb[idx].sum(1), 1)
    res, D, Dstar = {}, [], []
    names = list(vars_)
    for k in names:
        nv, sv = vars_[k]
        mv = sv.sum() / max(nv.sum(), 1)
        Bv = sv[idx].sum(1) / np.maximum(nv[idx].sum(1), 1)
        d, ds = mv - mb, Bv - Bb
        se = float(ds.std(ddof=1))
        res[k] = {"diff": float(d), "lo": float(np.quantile(ds, 0.025)), "hi": float(np.quantile(ds, 0.975)), "se": se,
                  "t": float(d / se) if se > 0 else float("nan"), "n": int(nv.sum()), "mean": float(mv)}
        D.append(d)
        Dstar.append(ds)
    if not names:
        return res
    D = np.array(D)
    Ds = np.vstack(Dstar)                       # (K, B)
    se = np.array([res[k]["se"] for k in names])
    se = np.where(se > 0, se, np.inf)
    tk = D / se
    cen = (Ds - D[:, None]) / se[:, None]       # centred, studentised
    # Romano-Wolf stepdown (one-sided, H0: diff <= 0)
    order = np.argsort(-tk)
    p_adj = np.zeros(len(names))
    prev = 0.0
    for j, k in enumerate(order):
        rem = order[j:]
        mx = cen[rem].max(axis=0)
        p = float((mx >= tk[k]).mean())
        prev = max(prev, p)
        p_adj[k] = prev
    for k, p in zip(names, p_adj, strict=True):
        res[k]["p_rw"] = float(p)
    # Hansen SPA (consistent), H0: no variant beats the base
    nday = idx.shape[1]
    thr = -math.sqrt(2 * math.log(max(math.log(nday), 1.0001)))
    mu = np.where(tk >= thr, 0.0, D)          # Hansen g_c: variants far below the base keep their negative mean
    T = max(0.0, float(tk.max()))
    Tstar = np.maximum(0.0, ((Ds - D[:, None] + mu[:, None]) / se[:, None]).max(axis=0))
    res["_spa"] = {"T": T, "p": float((Tstar >= T).mean()), "K": len(names)}
    return res


def sharpe(x: np.ndarray) -> float:
    x = x[np.isfinite(x)]
    return float(x.mean() / x.std(ddof=1)) if len(x) > 2 and x.std(ddof=1) > 0 else float("nan")


def deflated_sharpe(x: np.ndarray, n_trials: int, var_sr: float) -> dict:
    """Bailey & Lopez de Prado (2014) on a daily series; SR and V[SR] per day (not annualised)."""
    from scipy.stats import kurtosis, norm, skew

    x = x[np.isfinite(x)]
    T = len(x)
    sr = sharpe(x)
    g3, g4 = float(skew(x)), float(kurtosis(x, fisher=False))
    em = 0.5772156649
    sr0 = math.sqrt(max(var_sr, 0.0)) * ((1 - em) * norm.ppf(1 - 1 / n_trials) + em * norm.ppf(1 - 1 / (n_trials * math.e)))
    den = math.sqrt(max(1 - g3 * sr + (g4 - 1) / 4 * sr**2, 1e-12))
    return {"sr_day": sr, "sr_ann": sr * math.sqrt(365), "sr0_ann": sr0 * math.sqrt(365), "T": T, "skew": g3, "kurt": g4, "N": n_trials,
            "psr0": float(norm.cdf(sr * math.sqrt(T - 1) / den)), "dsr": float(norm.cdf((sr - sr0) * math.sqrt(T - 1) / den))}


def pbo_cscv(M: np.ndarray, blocks: int = 16) -> dict:
    """CSCV probability of backtest overfitting (Bailey et al. 2017) on a (T, K) matrix of daily P&L."""
    from itertools import combinations

    T, K = M.shape
    if K < 2 or T < blocks * 5:
        return {"pbo": float("nan")}
    edges = np.linspace(0, T, blocks + 1).astype(int)
    parts = [M[edges[i]:edges[i + 1]] for i in range(blocks)]
    sums = np.array([p.sum(0) for p in parts])
    sq = np.array([(p**2).sum(0) for p in parts])
    cnt = np.array([len(p) for p in parts])
    logits = []
    for c in combinations(range(blocks), blocks // 2):
        ins = np.zeros(blocks, dtype=bool)
        ins[list(c)] = True

        def sr(mask: np.ndarray) -> np.ndarray:
            n = cnt[mask].sum()
            m = sums[mask].sum(0) / n
            v = sq[mask].sum(0) / n - m**2
            return m / np.sqrt(np.maximum(v, 1e-18))

        a, b = sr(ins), sr(~ins)
        best = int(np.argmax(a))
        rank = (b < b[best]).sum() + 0.5 * ((b == b[best]).sum() - 1) + 1  # 1..K
        w = rank / (K + 1)
        logits.append(math.log(w / (1 - w)))
    lg = np.array(logits)
    return {"pbo": float((lg <= 0).mean()), "n_splits": len(lg), "median_logit": float(np.median(lg))}


def dk_ols(X: np.ndarray, y: np.ndarray, day: np.ndarray, lag: int = 7) -> dict:
    """Pooled OLS with Driscoll-Kraay SE (Newey-West over the daily cross-sectional sums of scores)."""
    ok = np.all(np.isfinite(X), axis=1) & np.isfinite(y)
    X, y, day = X[ok], y[ok], day[ok]
    XtX = X.T @ X
    b = np.linalg.solve(XtX, X.T @ y)
    e = y - X @ b
    ud, inv = np.unique(day, return_inverse=True)
    h = np.zeros((len(ud), X.shape[1]))
    np.add.at(h, inv, X * e[:, None])
    S_ = h.T @ h
    for L in range(1, lag + 1):
        w = 1 - L / (lag + 1)
        G = h[L:].T @ h[:-L]
        S_ += w * (G + G.T)
    Ai = np.linalg.inv(XtX)
    V = Ai @ S_ @ Ai
    se = np.sqrt(np.diag(V))
    return {"b": b.tolist(), "se": se.tolist(), "t": (b / se).tolist(), "n": int(len(y)), "days": int(len(ud))}


# ====================================================================== synthetic null (WS15 core)
SYN_SOURCES = S.SYNTH_SOURCES
SBB_CAP = 24


def garch_t_fit(r: np.ndarray, cap: float | None = None) -> dict:
    """GARCH(1,1)-t MLE. cap (amendment after the stylised-fact check rejected the uncapped fit): if alpha + beta > cap,
    scale both to sum to cap and variance-target omega = var(r) (1 - cap)."""
    from arch import arch_model

    am = arch_model(r * 100, mean="Zero", vol="GARCH", p=1, q=1, dist="t", rescale=False)
    res = am.fit(disp="off")
    p = res.params.to_dict()
    if cap is not None and p["alpha[1]"] + p["beta[1]"] > cap:
        k = cap / (p["alpha[1]"] + p["beta[1]"])
        p["alpha[1]"], p["beta[1]"] = p["alpha[1]"] * k, p["beta[1]"] * k
        x = r * 100
        p["omega"] = float(np.var(x)) * (1 - cap)
        cv = np.empty(len(r))
        h = float(np.var(x))
        for i in range(len(r)):
            cv[i] = math.sqrt(h)
            h = p["omega"] + p["alpha[1]"] * x[i] ** 2 + p["beta[1]"] * h
        return {"params": p, "cond_vol": cv / 100}
    return {"params": p, "cond_vol": res.conditional_volatility / 100}


def garch_t_path(fit: dict, n: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    p = fit["params"]
    om, al, be, nu = p["omega"], p["alpha[1]"], p["beta[1]"], p["nu"]
    from scipy.stats import t as tdist

    z = tdist.rvs(nu, size=n, random_state=rng) * math.sqrt((nu - 2) / nu)
    h = np.empty(n)
    r = np.empty(n)
    h0 = om / max(1 - al - be, 1e-6)
    hp, rp = h0, 0.0
    for i in range(n):
        hi = om + al * rp * rp + be * hp
        h[i] = hi
        rp = math.sqrt(hi) * z[i]
        r[i] = rp
        hp = hi
    return r / 100, np.sqrt(h) / 100


def synth_path(f1: pl.DataFrame, gen: str, rng: np.random.Generator, cache: dict) -> dict[str, np.ndarray]:
    from ai_trader.sim import resample as rs

    o, h, lo, c = (f1[k].to_numpy().astype(float) for k in ("open", "high", "low", "close"))
    if "rb" not in cache:
        cache["rb"] = rs.relative_bars(o, h, lo, c)
        pw = rs.politis_white_block(np.abs(cache["rb"].r))
        cache["pw"] = pw
        cache["b"] = min(pw["b"], SBB_CAP)
    rb = cache["rb"]
    n = len(rb)
    if gen == "sbb":
        idx = rs.stationary_indices(n, n, cache["b"], 1, rng)
        return {k: v[0] for k, v in rs.reconstruct(rb, idx, float(c[0])).items()}
    if cache.get("garch_gen") != gen:
        cache["garch"] = garch_t_fit(rb.r, cap=0.99 if gen == "garch99" else None)
        cache["garch_gen"] = gen
        sv = cache["garch"]["cond_vol"]
        sv = np.where(sv > 0, sv, np.nan)
        cache["wu"], cache["wd"], cache["sgn"] = rb.u / sv, rb.d / sv, np.sign(rb.r)
    r, sv = garch_t_path(cache["garch"], n, rng)
    sgn = cache["sgn"]
    pos, neg = np.flatnonzero(sgn >= 0), np.flatnonzero(sgn < 0)
    j = np.where(r >= 0, rng.choice(pos, n), rng.choice(neg, n))
    u = np.nan_to_num(cache["wu"][j]) * sv
    d = np.nan_to_num(cache["wd"][j]) * sv
    syn = rs.RelBars(r=r, g=np.zeros(n), u=np.maximum(u, 0), d=np.maximum(d, 0))
    return {k: v[0] for k, v in rs.reconstruct(syn, np.arange(n)[None, :], float(c[0])).items()}


def synth_rep(src: str, gen: str, reps: list[int], spec_names: list[str]) -> list[dict]:
    """Null replications for one source: per rep the specs' trades on a full-length synthetic path (signal specs only)."""
    from ai_trader.research.frame import market_frame

    f1 = market_frame(src, "1h", AS_OF, funding=False, oi=False)
    cache: dict = {}
    sp_all = specs()
    rows = []
    for rep in reps:
        rng = np.random.default_rng(SEED + 1009 * rep + {"sbb": 0, "garch": 500_000, "garch99": 900_000}[gen] + sum(map(ord, src)))
        p = synth_path(f1, gen, rng, cache)
        start = datetime(2031, 1, 6, tzinfo=UTC)
        g1, g4 = S.synth_frames(p["open"], p["high"], p["low"], p["close"], start)
        path = S.Path.from_frame(g1, 60.0)
        sigs = all_signals(g4, None, which={sp_all[n].signal for n in spec_names})
        b = make_bars(g4, g1, path, None)
        for nm in spec_names:
            sp = sp_all[nm]
            tr = [x for x in walk(sigs[sp.signal], b, sp) if x["why"] != "end"]   # closed trades (RT-STRESS-2)
            if not tr:
                rows.append({"src": src, "gen": gen, "rep": rep, "spec": nm, "n": 0, "sum": 0.0})
                continue
            df = pl.DataFrame(tr).with_columns(pl.lit("liquid").alias("tier"))
            r = S.r_net(df)
            rows.append({"src": src, "gen": gen, "rep": rep, "spec": nm, "n": len(r), "sum": float(r.sum()), "block": cache.get("b")})
    return rows


def stylised(r: np.ndarray) -> dict:
    """Stylised facts of a 1h log-return series."""
    from scipy.stats import kurtosis

    r = r[np.isfinite(r)]
    a = np.abs(r)

    def acf(x: np.ndarray, k: int) -> float:
        x = x - x.mean()
        return float((x[k:] * x[:-k]).sum() / (x * x).sum())

    srt = np.sort(a)[::-1]
    k = max(10, int(0.025 * len(a)))
    hill = 1.0 / np.mean(np.log(srt[:k] / srt[k]))
    d = r[: len(r) // 24 * 24].reshape(-1, 24).sum(1)
    w = d[: len(d) // 7 * 7].reshape(-1, 7).sum(1)
    lev = float(np.corrcoef(r[:-24], np.array([a[i + 1:i + 25].sum() for i in range(len(r) - 24)]))[0, 1]) if len(r) > 2000 else float("nan")
    return {"exkurt_1h": float(kurtosis(r)), "hill_alpha": float(hill), "acf_abs_1": acf(a, 1), "acf_abs_24": acf(a, 24), "acf_abs_168": acf(a, 168),
            "acf_r_1": acf(r, 1), "leverage_24h": lev, "exkurt_1d": float(kurtosis(d)), "vr_7d": float(w.var() / (7 * d.var()))}


def window_features(r: np.ndarray, L: int = 720) -> np.ndarray:
    from scipy.stats import kurtosis, skew

    rows = []
    for i in range(0, len(r) - L + 1, L):
        x = r[i:i + L]
        if not np.all(np.isfinite(x)):
            continue
        a = np.abs(x)
        am = a - a.mean()
        xm = x - x.mean()
        d = x.reshape(-1, 24).sum(1)
        cum = np.cumsum(x)
        rv = math.sqrt((x**2).sum())
        rows.append([math.log(rv), kurtosis(x), skew(x), (am[1:] * am[:-1]).sum() / (am * am).sum(), (am[24:] * am[:-24]).sum() / (am * am).sum(),
                     float(np.corrcoef(x[:-1], a[1:])[0, 1]),                                    # vol-only block ends (6)
                     (xm[1:] * xm[:-1]).sum() / (xm * xm).sum(), abs(cum[-1]) / rv,
                     float(d[: len(d) // 7 * 7].reshape(-1, 7).sum(1).var() / (7 * d.var())) if d.var() > 0 else 1.0,
                     float((cum - np.maximum.accumulate(cum)).min()) / rv])                      # trend block (4)
    return np.array(rows)


def discriminator_auc(real: np.ndarray, syn: np.ndarray) -> dict:
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import roc_auc_score

    X = np.vstack([real, syn])
    y = np.r_[np.ones(len(real)), np.zeros(len(syn))]
    out = {}
    for nm, cols in (("all", slice(None)), ("vol", slice(0, 6)), ("trend", slice(6, 10))):
        m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
        p = cross_val_predict(m, X[:, cols], y, cv=StratifiedKFold(5, shuffle=True, random_state=SEED), method="predict_proba")[:, 1]
        out[nm] = float(roc_auc_score(y, p))
    out["n_real"], out["n_syn"] = len(real), len(syn)
    return out


def facts_job(src: str) -> dict:
    """Stylised facts and discriminator inputs: real vs 3 paths of each generator for one source."""
    from ai_trader.research.frame import market_frame

    f1 = market_frame(src, "1h", AS_OF, funding=False, oi=False)
    c = f1["close"].to_numpy().astype(float)
    r_real = np.diff(np.log(c))
    out = {"src": src, "real": stylised(r_real), "feat_real": window_features(r_real).tolist()}
    cache: dict = {}
    for gen in ("sbb", "garch", "garch99"):
        rs_, feats = [], []
        for k in range(3):
            rng = np.random.default_rng(SEED + 77 * k + {"sbb": 0, "garch": 9999, "garch99": 19999}[gen])
            p = synth_path(f1, gen, rng, cache)
            r = np.diff(np.log(p["close"]))
            rs_.append(stylised(r))
            feats += window_features(r).tolist()
        out[gen] = {k: float(np.mean([x[k] for x in rs_])) for k in rs_[0]}
        out[f"feat_{gen}"] = feats
    out["block"] = cache.get("b")
    out["pw"] = cache.get("pw")
    out["garch99_params"] = {k: float(v) for k, v in cache["garch"]["params"].items()} if "garch" in cache else None
    out["garch_params"] = {k: float(v) for k, v in garch_t_fit(np.diff(np.log(c)))["params"].items()}
    return out


# ====================================================================== dossier statistics (signal and target variable)
def dossier_symbol(sym: str) -> dict:
    """Daily panel rows (00:00 UTC 4h closes): z_k, sign, tanh, signals, forward 7d standardised return; 4h z for the
    distribution of the target variable."""
    from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
    from statsmodels.tsa.stattools import adfuller, kpss
    from scipy.stats import kurtosis, skew, t as tdist

    from ai_trader.strategies.trend import arr
    from ai_trader.strategies.trend_zoo import _filter_z, tsmom_weights

    ctx = S.load_ctx(sym, AS_OF, need_rb=False)
    if ctx is None:
        return {"symbol": sym, "skipped": True}
    f4 = ctx.f4
    c, sg = arr(f4, "close"), arr(f4, "sigma_next")
    lc = np.log(c)
    H = 42
    y = np.full(len(c), np.nan)
    y[:-H] = (lc[H:] - lc[:-H]) / (sg[:-H] * math.sqrt(H))
    sigf = TMP / f"sig_{sym}.parquet"
    z = pl.read_parquet(sigf)
    zs = {k: z[k].to_numpy().astype(float) for k in z.columns if k != "close_ts"}
    ens = ensembles(zs, f4, load_weights())
    cols = {"symbol": sym, "close_ts": f4["close_ts"], "y7": y, "zoo_ew": ens["zoo_ew"], "kalman_llt": kalman_signal(f4)}
    for k in (7, 14, 30):
        zk = _filter_z(c, sg, tsmom_weights(k * BPD))
        cols[f"z{k}"] = zk
    for k in ("tsmom_30d", "ema_4_16_64", "donch_30", "slope_30d", "cfg_ema_stack"):
        cols[k] = zs[k]
    panel = pl.DataFrame(cols).filter(pl.col("close_ts").dt.hour() == 0)
    # target variable: one-bar standardised 4h return z1 = r_{t+1} / sigma_t (non-overlapping)
    r = np.r_[np.diff(lc), np.nan]
    z1 = r / sg
    z1 = z1[np.isfinite(z1)]
    seg_cut = int(np.searchsorted(S._ms(f4["close_ts"]), int(datetime(2026, 7, 1, tzinfo=UTC).timestamp() * 1000)))
    zi = z1[:max(seg_cut - 1, 0)]
    props = {}
    try:
        props["adf_p"] = float(adfuller(zi, maxlag=12, autolag=None)[1])
        props["kpss_p"] = float(kpss(zi, regression="c", nlags="auto")[1])
    except Exception as e:  # noqa: BLE001
        props["err"] = repr(e)
    props["mean"], props["sd"], props["skew"], props["exkurt"] = float(zi.mean()), float(zi.std()), float(skew(zi)), float(kurtosis(zi))
    nu, loc, sc = tdist.fit(zi, floc=0.0)
    props["t_nu"], props["t_scale"] = float(nu), float(sc)
    a = np.sort(np.abs(zi))[::-1]
    k_ = max(10, int(0.025 * len(a)))
    props["hill"] = float(1.0 / np.mean(np.log(a[:k_] / a[k_])))
    lb = acorr_ljungbox(zi, lags=[6, 42])
    lb2 = acorr_ljungbox(zi**2, lags=[6, 42])
    props["lb_p_6"], props["lb_p_42"] = float(lb["lb_pvalue"].iloc[0]), float(lb["lb_pvalue"].iloc[1])
    props["lb2_p_6"], props["lb2_p_42"] = float(lb2["lb_pvalue"].iloc[0]), float(lb2["lb_pvalue"].iloc[1])
    props["arch_lm_p"] = float(het_arch(zi, nlags=6)[1])
    props["n"] = int(len(zi))
    props["z2_mean"] = float(np.mean(zi**2))
    vs = arr(f4, "sigma_next")[:len(zi)]
    q = np.quantile(vs[np.isfinite(vs)], [1 / 3, 2 / 3])
    props["z2_by_vol_tercile"] = [float(np.nanmean(zi[(vs <= q[0])] ** 2)), float(np.nanmean(zi[(vs > q[0]) & (vs <= q[1])] ** 2)), float(np.nanmean(zi[vs > q[1]] ** 2))]
    od = OUT / "dossier"
    od.mkdir(parents=True, exist_ok=True)
    panel.write_parquet(od / f"panel_{sym}.parquet")
    return {"symbol": sym, "props": props}


# ====================================================================== CLI
def _universe() -> list[str]:
    from ai_trader.research.trend_zoo_study import UNIVERSE

    return list(UNIVERSE)


def _map(fn: Callable, args: list, workers: int) -> list:
    out = []
    if workers <= 1:
        for a in args:
            out.append(fn(*a))
            print(json.dumps(out[-1], default=str)[:300], flush=True)
        return out
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(fn, *a) for a in args]
        for f in futs:
            try:
                out.append(f.result())
            except Exception as e:  # noqa: BLE001
                out.append({"error": repr(e)})
            print(json.dumps(out[-1], default=str)[:300], flush=True)
    return out


def design() -> dict:
    return {"as_of": AS_OF.isoformat(), "seed": SEED, "universe": _universe(), "D8": S.D8, "specs": {k: v.__dict__ for k, v in specs().items()},
            "sbb_cap": SBB_CAP, "syn_sources": SYN_SOURCES, "five_min_start": FIVE_MIN_START.isoformat(), "expected_regimes": EXPECTED_REGIMES,
            "costs": S.Costs().__dict__, "verdict": "STRESS F1/F2/G1..G7 + G8 real mean <= 95th pct of the SBB(min(PW,24)) null"}


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    w = sub.add_parser("weights")
    w.add_argument("--workers", type=int, default=2)
    r = sub.add_parser("run")
    r.add_argument("--workers", type=int, default=2)
    r.add_argument("--symbols", nargs="*")
    r.add_argument("--only", nargs="*")
    r.add_argument("--out", default="trades")
    d = sub.add_parser("dossier")
    d.add_argument("--workers", type=int, default=2)
    s = sub.add_parser("synth")
    s.add_argument("--workers", type=int, default=3)
    s.add_argument("--reps", type=int, default=100)
    s.add_argument("--specs", nargs="*", default=["zoo_ew"])
    s.add_argument("--gens", nargs="*", default=["sbb", "garch"])
    sub.add_parser("facts")
    a = sub.add_parser("analyse")
    a.add_argument("--record", action="store_true")
    args = ap.parse_args(argv)
    OUT.mkdir(parents=True, exist_ok=True)
    TMP.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    if args.cmd == "weights":
        dp = OUT / "design.json"
        if not dp.exists():
            dp.write_text(json.dumps(design(), indent=1, default=str), encoding="utf-8")
        _map(pass1_symbol, [(s_,) for s_ in _universe()], args.workers)
        print(build_weights())
    elif args.cmd == "run":
        syms = args.symbols or _universe()
        _map(run_symbol, [(s_, s_ in S.D8, args.only, args.out) for s_ in syms], args.workers)
    elif args.cmd == "dossier":
        res = _map(dossier_symbol, [(s_,) for s_ in _universe()], args.workers)
        (OUT / "dossier_props.json").write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")
    elif args.cmd == "synth":
        od = OUT / "synth"
        od.mkdir(parents=True, exist_ok=True)
        jobs = []
        for gen in args.gens:
            for src in SYN_SOURCES:
                for j in range(0, args.reps, 10):
                    jobs.append((src, gen, list(range(j, min(j + 10, args.reps))), list(args.specs)))
        res = _map(synth_rep, jobs, args.workers)
        rows = [x for r_ in res if isinstance(r_, list) for x in r_]
        pl.DataFrame(rows).write_parquet(od / f"null_{'_'.join(args.gens)}.parquet")
    elif args.cmd == "facts":
        res = _map(facts_job, [(s_,) for s_ in SYN_SOURCES], 1)
        (OUT / "facts.json").write_text(json.dumps(res, default=float), encoding="utf-8")
    elif args.cmd == "analyse":
        from ai_trader.research.trend_spec_report import analyse

        analyse(record=args.record)
    print(f"done in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
