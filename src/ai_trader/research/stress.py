"""STRESS: stress test of every deterministic strategy in every scenario (block STRESS, H-STRESS0..10).

Research only: no orders, no venue calls. The LLM decision layer is never tested here (CLAUDE.md principle 1): only the
deterministic generators, the desk setups as coded or specified and the live risk engine limits (config/risk.yaml
live_small) applied as the sizing layer. The design was frozen in research/hypotheses.yaml (block STRESS) and the diary
("STRESS design frozen before results") before any stress run; data/research/stress/design.json repeats it.

One call stresses a new strategy:

    from ai_trader.research.stress import stress_signal
    report = stress_signal(my_signal_fn, symbols=["BTCUSDT"], freq="1h")     # base + costs + latency + gaps + engine

Pieces
------
* Trade layer (`walk_signal`): signal s_t in [-1, 1] at the strategy bar close -> enter when flat and |s| >= theta_in in
  sign(s); exit at the close of the first bar with s * side <= 0 (re-entry on the same bar if |s| >= theta_in);
  exchange stop at stop_k x sigma_risk_next (per bar of the strategy frequency) from the fill, checked on the 1h path
  (stop modes: price | gap | next_open | none); after a stop-out no re-entry until the signal is flat or flips.
* Bracket layer (`walk_bracket`): desk setups with structural stop / target / optional time stop, one position per
  symbol, stop before target inside a bar.
* Latency: a fill `m` minutes after a decision is the Brownian-bridge conditional mean between the decision price and
  the close of the path bar containing decision + m (log-linear in elapsed time); `Path.lat` can carry exact prices.
* R: (d (x - e) - c_in e - c_out x) / (|e - stop| + e (c_in_base + c_out_base)): the at-risk sizing budget, so a cost
  shock changes only the numerator (`r_net`).
* Metrics (`metrics`): n, mean R with a stationary day bootstrap CI, hit, PF, max DD, CVaR5% per trade and per day,
  time under water, top-5 share.
* Live engine (`live_engine`): live_small limits on a pooled chronological trade stream, USD per trade.
* Scenarios S0..S9, verdict rules F1/F2, G1..G7 (`verdict`).

CLI:
    PYTHONIOENCODING=utf-8 uv run python -m ai_trader.research.stress design
    PYTHONIOENCODING=utf-8 uv run python -m ai_trader.research.stress run [--workers 2] [--symbols ...]
    PYTHONIOENCODING=utf-8 uv run python -m ai_trader.research.stress perturb [--workers 2]
    PYTHONIOENCODING=utf-8 uv run python -m ai_trader.research.stress synth [--workers 2] [--paths 1000]
    PYTHONIOENCODING=utf-8 uv run python -m ai_trader.research.stress analyse [--record]
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, field
from datetime import UTC, date, datetime, timedelta
from pathlib import Path as FsPath
from typing import Any

import numpy as np
import polars as pl

ROOT = FsPath(__file__).resolve().parents[3]
OUT = ROOT / "data" / "research" / "stress"
SEED = 20261126
THETA_IN = 0.3
STOP_K = 4.0
MIN_SIG = 0.5
TICK_MARGIN = 0.01
H_MS = 3_600_000
D8 = ("BTCUSDT", "ETHUSDT", "SOLUSDT", "ADAUSDT", "AVAXUSDT", "LINKUSDT", "XRPUSDT", "DOGEUSDT")
SYNTH_SOURCES = ("BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "DOGEUSDT", "LINKUSDT")
PERTURB_FACTORS = (0.5, 0.8, 1.2, 1.5)
N_JOINT = 30
N_BOOT = 2000
BLOCK_DAYS = 5.0
PRIOR_EDGE = {"shock_desk": 0.05, "hs2_fade_blackout": 0.05, "trend_4h": 0.03}

# NYSE full-day closures 2020-2026 (calendar.py does not model holidays).
NYSE_HOLIDAYS = {date.fromisoformat(d) for d in (
    "2020-01-01 2020-01-20 2020-02-17 2020-04-10 2020-05-25 2020-07-03 2020-09-07 2020-11-26 2020-12-25 "
    "2021-01-01 2021-01-18 2021-02-15 2021-04-02 2021-05-31 2021-07-05 2021-09-06 2021-11-25 2021-12-24 "
    "2022-01-17 2022-02-21 2022-04-15 2022-05-30 2022-06-20 2022-07-04 2022-09-05 2022-11-24 2022-12-26 "
    "2023-01-02 2023-01-16 2023-02-20 2023-04-07 2023-05-29 2023-06-19 2023-07-04 2023-09-04 2023-11-23 2023-12-25 "
    "2024-01-01 2024-01-15 2024-02-19 2024-03-29 2024-05-27 2024-06-19 2024-07-04 2024-09-02 2024-11-28 2024-12-25 "
    "2025-01-01 2025-01-09 2025-01-20 2025-02-17 2025-04-18 2025-05-26 2025-06-19 2025-07-04 2025-09-01 2025-11-27 "
    "2025-12-25 2026-01-01 2026-01-19 2026-02-16 2026-04-03 2026-05-25 2026-06-19 2026-07-03 2026-09-07 2026-11-26 "
    "2026-12-25").split()}

FIXED_CRISES = {
    "2020-03 covid crash": ("2020-03-09", "2020-03-20"),
    "2021-05-19 crash": ("2021-05-12", "2021-05-26"),
    "2021-12 deleveraging": ("2021-12-01", "2021-12-31"),
    "2022-05 LUNA": ("2022-05-05", "2022-05-20"),
    "2022-06 3AC/Celsius": ("2022-06-10", "2022-06-25"),
    "2022-11 FTX": ("2022-11-06", "2022-11-20"),
    "2023-03 SVB/USDC": ("2023-03-08", "2023-03-20"),
    "2024-08-05 yen carry": ("2024-08-01", "2024-08-10"),
    "2026-11-21 flush": ("2026-11-19", "2026-11-25"),
}

EXEC_VARIANTS = {  # name -> (stop_mode, latency minutes)
    "base": ("price", 0.0), "gap": ("gap", 0.0), "stop_next_open": ("next_open", 0.0), "no_stop": ("none", 0.0),
    "lat1": ("price", 1.0), "lat5": ("price", 5.0), "lat15": ("price", 15.0),
}
COST_VARIANTS = {  # name -> (fee_mult, extra_slip_bps per side, drought)
    "fees_x1.5": (1.5, 0.0, False), "fees_x2": (2.0, 0.0, False), "slip+5": (1.0, 5.0, False),
    "slip+10": (1.0, 10.0, False), "slip+25": (1.0, 25.0, False), "drought": (1.0, 0.0, True),
}


# ====================================================================== costs and R
@dataclass(frozen=True)
class Costs:
    maker_bps: float = 2.0
    taker_bps: float = 5.5
    slip_bps: float = 2.0
    peg_slip_bps: float = 1.3
    fee_mult: float = 1.0
    extra_bps: float = 0.0
    drought: bool = False

    @classmethod
    def live(cls, **kw: Any) -> Costs:
        from ai_trader.core.config import load

        r = load().risk
        k = r["costs"]
        return cls(maker_bps=float(k["maker_bps"]), taker_bps=float(k["taker_bps"]), slip_bps=float(k["slippage_bps"]),
                   peg_slip_bps=float(r["paper_entry"]["slip_bps"]), **kw)


TIERS = {  # live_small tiers: half-spread = max_spread / 2, stop_slip_bps, notional cap
    "liquid": {"min_turnover": 100e6, "half_spread": 1.5, "stop_slip": 10.0, "cap": 200.0},
    "mid": {"min_turnover": 20e6, "half_spread": 5.0, "stop_slip": 25.0, "cap": 100.0},
    "illiquid": {"min_turnover": 3e6, "half_spread": 15.0, "stop_slip": 60.0, "cap": 50.0},
}


def tier_of(turnover: float) -> str:
    if not np.isfinite(turnover):
        return "unknown"
    for name, t in TIERS.items():
        if turnover >= t["min_turnover"]:
            return name
    return "below"


def leg_costs(kind: np.ndarray, why: np.ndarray, tier: np.ndarray, c: Costs) -> tuple[np.ndarray, np.ndarray]:
    """Per-trade entry / exit cost fractions under cost scenario c."""
    fm = c.fee_mult
    peg = kind == "peg"
    c_in = np.where(peg, c.maker_bps * fm + c.peg_slip_bps, c.taker_bps * fm + c.slip_bps) + c.extra_bps
    slip_out = np.full(len(kind), c.slip_bps)
    if c.drought:
        thin = np.isin(tier, ["mid", "illiquid", "below", "unknown"])
        hs = np.array([TIERS.get(t, TIERS["illiquid"])["half_spread"] for t in tier])
        ss = np.array([TIERS.get(t, TIERS["illiquid"])["stop_slip"] for t in tier])
        dx = np.where(thin, 2.0 * hs, 0.0)
        c_in = c_in + dx
        slip_out = np.where(thin & (why == "stop"), math.sqrt(2.0) * ss, slip_out) + dx
    c_out = c.taker_bps * fm + slip_out + c.extra_bps
    return c_in / 1e4, c_out / 1e4


def base_budget(kind: np.ndarray, c: Costs | None = None) -> np.ndarray:
    c = c or Costs()
    return (np.where(kind == "peg", c.maker_bps + c.peg_slip_bps, c.taker_bps + c.slip_bps) + c.taker_bps + c.slip_bps) / 1e4


def r_net(t: pl.DataFrame, c: Costs | None = None, base: Costs | None = None) -> np.ndarray:
    """Net R per trade under cost scenario c; the denominator is the base sizing budget (at-risk)."""
    c = c or Costs()
    if t.height == 0:
        return np.zeros(0)
    kind, why, tier = (t[k].to_numpy() for k in ("kind", "why", "tier"))
    e, s, x, d = (t[k].to_numpy().astype(float) for k in ("entry", "stop", "exit", "side"))
    c_in, c_out = leg_costs(kind, why, tier, c)
    den = np.abs(e - s) + e * base_budget(kind, base or Costs(c.maker_bps, c.taker_bps, c.slip_bps, c.peg_slip_bps))
    return (d * (x - e) - c_in * e - c_out * x) / den


# ====================================================================== price paths and fills
@dataclass
class Path:
    """Bars used to walk trades (1h for every bar unit, 15m for setup_c_proxy). ts = bar open, epoch ms."""

    ts: np.ndarray
    o: np.ndarray
    h: np.ndarray
    lo: np.ndarray
    c: np.ndarray
    bar_min: float = 60.0
    lat: dict[float, np.ndarray] | None = None  # exact prices `m` minutes after each bar's open (optional)

    @classmethod
    def from_frame(cls, f: pl.DataFrame, bar_min: float = 60.0) -> Path:
        return cls(f["event_ts"].dt.epoch("ms").to_numpy().astype(np.int64), *(f[k].to_numpy().astype(float) for k in ("open", "high", "low", "close")),
                   bar_min=bar_min)

    def __len__(self) -> int:
        return len(self.ts)

    def fill(self, i: int, ref: float, minutes: float) -> float:
        """Price `minutes` after the open of bar i (decision price `ref` at that open)."""
        if minutes <= 0:
            return ref
        j = i + int(minutes // self.bar_min)
        rem = minutes - (j - i) * self.bar_min
        if rem <= 1e-9 and j > i:
            return float(self.c[j - 1]) if j - 1 < len(self.c) else float("nan")
        if j >= len(self.ts):
            return float("nan")
        if self.lat is not None and minutes in self.lat and i < len(self.lat[minutes]) and np.isfinite(self.lat[minutes][i]):
            return float(self.lat[minutes][i])
        start = ref if j == i else self.c[j - 1]
        f = rem / self.bar_min
        return float(math.exp(math.log(start) + f * (math.log(self.c[j]) - math.log(start))))


def _stop_fill(p: Path, i: int, d: int, stop: float, mode: str) -> tuple[float, int, bool]:
    gap = (d > 0 and p.o[i] < stop) or (d < 0 and p.o[i] > stop)
    if mode == "gap" and gap:
        return float(p.o[i]), int(p.ts[i] + p.bar_min * 60_000), gap
    if mode == "next_open":
        if i + 1 < len(p):
            return float(p.o[i + 1]), int(p.ts[i + 1]), gap
        return float(p.c[i]), int(p.ts[i] + p.bar_min * 60_000), gap
    return float(stop), int(p.ts[i] + p.bar_min * 60_000), gap


def walk_signal(sig: np.ndarray, close_ms: np.ndarray, close_px: np.ndarray, sigma: np.ndarray, path: Path, *,
                theta: float = THETA_IN, stop_k: float = STOP_K, stop_mode: str = "price", latency: float = 0.0) -> list[dict]:
    """Generic trade layer for a signal strategy (see module docstring). Bars are the strategy's; stops walk `path`."""
    n = len(sig)
    a = np.searchsorted(path.ts, close_ms, side="left")
    a = np.r_[a, len(path)]
    out: list[dict] = []
    pos, entry, stop, ets, ebar = 0, 0.0, 0.0, 0, 0
    armed, stopped_side = True, 0
    sig = np.nan_to_num(np.asarray(sig, dtype=float))

    def px_at(t: int) -> float:
        if latency <= 0:
            return float(close_px[t])
        return path.fill(int(a[t]), float(close_px[t]), latency) if a[t] < len(path) else float("nan")

    for t in range(n - 1):
        st = sig[t]
        if pos != 0 and st * pos <= 0:
            x = px_at(t)
            if np.isfinite(x):
                out.append({"side": pos, "entry_ms": ets, "exit_ms": int(close_ms[t] + latency * 60_000), "entry": entry, "stop": stop,
                            "exit": x, "why": "signal", "gap": False})
            pos = 0
        if not armed and st * stopped_side <= 0:
            armed = True
        if pos == 0 and armed and abs(st) >= theta and np.isfinite(sigma[t]) and sigma[t] > 0:
            e = px_at(t)
            if np.isfinite(e) and e > 0:
                pos = 1 if st > 0 else -1
                entry, ets, ebar = e, int(close_ms[t]), int(a[t])
                stop = e * math.exp(-pos * stop_k * float(sigma[t]))
        if pos != 0 and stop_mode != "none":
            for i in range(max(int(a[t]), ebar), int(a[t + 1])):
                if (pos > 0 and path.lo[i] <= stop) or (pos < 0 and path.h[i] >= stop):
                    x, xms, gap = _stop_fill(path, i, pos, stop, stop_mode)
                    out.append({"side": pos, "entry_ms": ets, "exit_ms": xms, "entry": entry, "stop": stop, "exit": x, "why": "stop", "gap": gap})
                    armed, stopped_side, pos = False, pos, 0
                    break
    if pos != 0 and len(path):  # still open at the end of the data: marked to the last close (why = end)
        out.append({"side": pos, "entry_ms": ets, "exit_ms": int(path.ts[-1] + path.bar_min * 60_000), "entry": entry, "stop": stop,
                    "exit": float(path.c[-1]), "why": "end", "gap": False})
    return out


@dataclass
class Intent:
    ms: int            # decision time (bar close), epoch ms
    side: int
    px: float          # decision price (base fill)
    stop: float
    target: float      # nan = none
    time_stop_h: float = float("nan")
    kind: str = "peg"
    ev: float = float("nan")
    extra: dict = field(default_factory=dict)


def walk_bracket(intents: list[Intent], path: Path, *, stop_mode: str = "price", latency: float = 0.0) -> list[dict]:
    """Bracket trades, one open position at a time (per symbol and unit); stop before target inside a bar."""
    out: list[dict] = []
    busy = -(10**18)
    n = len(path)
    for it in sorted(intents, key=lambda z: z.ms):
        if it.ms < busy:
            continue
        i0 = int(np.searchsorted(path.ts, it.ms, side="left"))
        if i0 >= n:
            continue
        e = it.px if latency <= 0 else path.fill(i0, it.px, latency)
        d, stop, tgt = it.side, it.stop, it.target
        if not np.isfinite(e) or d * (e - stop) <= 0 or (np.isfinite(tgt) and d * (tgt - e) <= 0):
            continue
        end = n if not np.isfinite(it.time_stop_h) else i0 + int(math.ceil(it.time_stop_h * 60 / path.bar_min))
        if end > n:
            continue  # unresolved: the time stop lies beyond the data
        res = None
        lo, hi = path.lo[i0:end], path.h[i0:end]
        hs = np.flatnonzero(lo <= stop) if d > 0 else np.flatnonzero(hi >= stop)
        ht = (np.flatnonzero(hi >= tgt) if d > 0 else np.flatnonzero(lo <= tgt)) if np.isfinite(tgt) else np.array([], dtype=int)
        js = int(hs[0]) if hs.size and stop_mode != "none" else None
        jt = int(ht[0]) if ht.size else None
        if js is not None and (jt is None or js <= jt):
            x, xms, gap = _stop_fill(path, i0 + js, d, stop, stop_mode)
            res = (x, xms, "stop", gap)
        elif jt is not None:
            res = (float(tgt), int(path.ts[i0 + jt] + path.bar_min * 60_000), "target", False)
        elif np.isfinite(it.time_stop_h):
            res = (float(path.c[end - 1]), int(path.ts[end - 1] + path.bar_min * 60_000), "time", False)
        else:  # still open at the end of the data: marked to the last close (why = end)
            res = (float(path.c[-1]), int(path.ts[-1] + path.bar_min * 60_000), "end", False)
        out.append({"side": d, "entry_ms": int(it.ms), "exit_ms": res[1], "entry": float(e), "stop": float(stop), "exit": res[0],
                    "why": res[2], "gap": res[3], "ev": it.ev, "kind": it.kind})
        busy = res[1]
    return out


# ====================================================================== placement helpers (agent.session / post_cascade)
def p_first(entry: float, stop: float, tp: float) -> float:
    a, b = abs(math.log(tp / entry)), abs(math.log(entry / stop))
    return b / (a + b) if a + b > 0 else float("nan")


def net_er(entry: float, stop: float, tp: float, p: float, kind: str, c: Costs | None = None) -> float:
    """at-risk intent.rs expected R after round-trip costs (as agent.session.net_er)."""
    c = c or Costs()
    ein = c.maker_bps + c.peg_slip_bps if kind == "peg" else c.taker_bps + c.slip_bps
    cost = entry * (ein + c.taker_bps + c.slip_bps) / 1e4
    sd, td = abs(entry - stop), abs(tp - entry)
    return p * (td - cost) / (sd + cost) - (1 - p)


def cap_stop(d: int, entry: float, stop: float, tp: float, sig: float, tp_r: float | None, cap: float = STOP_K,
             min_sig: float = MIN_SIG) -> tuple[float, float] | None:
    """agent.session.cap_stop: pull a stop farther than the cap in to cap*(1-1%); tp_r keeps an R-multiple target."""
    if not (np.isfinite(sig) and sig > 0):
        return None
    k = abs(math.log(entry / stop)) / sig
    if k > cap * (1 - TICK_MARGIN):
        k = cap * (1 - TICK_MARGIN)
        stop = entry * math.exp(-d * k * sig)
        if tp_r is not None:
            tp = entry * math.exp(d * tp_r * k * sig)
    if k < min_sig * (1 + TICK_MARGIN):
        return None
    return stop, tp


# ====================================================================== units
@dataclass(frozen=True)
class Unit:
    name: str
    kind: str                 # signal | bracket
    freq: str                 # 1h | 4h | 15m
    family: str
    keys: tuple[str, ...]     # perturbation keys
    data_generated: bool = False
    synthetic: bool = True
    entry_kind: str = "peg"
    symbols: tuple[str, ...] | None = None   # None = universe L
    note: str = ""


SIGNAL_1H = {
    "tsmom": ("trend", ("horizons", "scale")),
    "donchian_breakout": ("trend", ("entry_window", "exit_window", "scale")),
    "ema_stack": ("trend", ("fast", "mid", "slow", "k")),
    "zscore_reversion": ("mean_reversion", ("span", "std_window", "entry", "exit", "scale")),
    "rsi_reversion": ("mean_reversion", ("period", "band", "full")),
    "compression_breakout": ("volatility", ("window", "history", "ratio", "scale")),
    "funding_contrarian": ("positioning", ("window", "threshold", "scale", "extension")),
    "oi_price_divergence": ("positioning", ("horizon", "window", "threshold", "scale", "squeeze_weight")),
    "shock_regime": ("shock", ("z_min", "z_full", "decluster_bars", "trail_sigma", "invalid_sigma")),
    "weekend_shock_follow": ("calendar", ("z_min", "z_full", "vol_ratio_min")),
    "weekend_drift_fade": ("calendar", ("z_min", "z_full", "vol_ratio_max")),
    "cme_gap_fill": ("calendar", ("z_min", "z_full")),
    "flow_momentum": ("flow", ("window", "scale")),
    "cvd_divergence": ("flow", ("window", "scale")),
}
SIGNAL_4H = {
    "geo_struct": ("trend", ("ks", "vol_hl", "max_retr")),
    "geo_pivot_bo": ("trend", ("ks", "vol_hl", "max_retr")),
    "geo_flag_bo": ("trend", ("ks", "vol_hl", "max_retr")),
    "zoo_ensemble": ("trend", ("bpd",)),
}
HARNESS_KEYS = ("_theta", "_stopk")


def units() -> dict[str, Unit]:
    u: dict[str, Unit] = {}
    for n, (fam, keys) in SIGNAL_1H.items():
        flow = fam == "flow"
        u[n] = Unit(n, "signal", "1h", fam, keys + HARNESS_KEYS, data_generated=n == "shock_regime",
                    synthetic=fam not in ("positioning", "flow", "calendar"), symbols=("BTCUSDT",) if flow else None,
                    note="BTCUSDT flow_bars from 2026-05-11 only" if flow else "")
    for n, (fam, keys) in SIGNAL_4H.items():
        u[n] = Unit(n, "signal", "4h", fam, keys + HARNESS_KEYS)
    u["trend_4h"] = Unit("trend_4h", "bracket", "4h", "trend", ("thr", "stop_mult", "tp_r", "fast", "mid", "slow", "k"))
    u["shock_desk"] = Unit("shock_desk", "bracket", "1h", "shock", ("z_min", "decluster_bars", "rev_buffer", "cont_tp_r"),
                           data_generated=True, entry_kind="market")
    u["hs2_fade_blackout"] = Unit("hs2_fade_blackout", "bracket", "1h", "shock", ("z_min", "decluster_bars", "rev_buffer", "blackout_bars"),
                                  data_generated=True, entry_kind="market")
    u["hrb2"] = Unit("hrb2", "bracket", "4h", "positioning", ("r48_pct", "oi_drop", "rebound_sigma", "ema_span", "buffer_sigma", "target_r",
                                                             "time_stop_hours", "rb2_window_bars"), data_generated=True, synthetic=False)
    u["setup_c_proxy"] = Unit("setup_c_proxy", "bracket", "15m", "mean_reversion", ("level_hours", "wait_min", "reclaim_hours", "buffer_sigma",
                                                                                  "time_stop_hours"), synthetic=False, symbols=("BTCUSDT", "ETHUSDT"),
                              note="proxy: spot-absorption and Coinbase-premium confirmations dropped (no history)")
    return u


BRACKET_BASE = {
    "trend_4h": {"thr": 0.3, "stop_mult": 1.5, "tp_r": 3.0, "fast": 12, "mid": 48, "slow": 192, "k": 10.0},
    "shock_desk": {"z_min": 4.0, "decluster_bars": 24, "rev_buffer": 0.5, "cont_tp_r": 2.0},
    "hs2_fade_blackout": {"z_min": 4.0, "decluster_bars": 24, "rev_buffer": 0.5, "blackout_bars": 1},
    "hrb2": {"r48_pct": 0.05, "oi_drop": -0.05, "rebound_sigma": 1.5, "ema_span": 20, "buffer_sigma": 0.5, "target_r": 2.0,
             "time_stop_hours": 120, "rb2_window_bars": 42},
    "setup_c_proxy": {"level_hours": 72, "wait_min": 60, "reclaim_hours": 24, "buffer_sigma": 0.5, "time_stop_hours": 8},
}


def base_params(name: str) -> dict[str, Any]:
    if name in BRACKET_BASE:
        return dict(BRACKET_BASE[name])
    p: dict[str, Any] = {"_theta": THETA_IN, "_stopk": STOP_K}
    if name == "zoo_ensemble":
        p["bpd"] = 6
        return p
    from ai_trader.strategies import REGISTRY

    p.update(REGISTRY[name]().params)
    return p


def perturb(params: dict[str, Any], key: str, f: float) -> dict[str, Any]:
    q = dict(params)
    if key == "band":
        q["low"] = 50 - (50 - float(params["low"])) * f
        q["high"] = 50 + (float(params["high"]) - 50) * f
        return q
    v = params[key]
    if isinstance(v, list | tuple):
        q[key] = [max(1, round(x * f)) if isinstance(x, int) else x * f for x in v]
    elif isinstance(v, bool):
        pass
    elif isinstance(v, int):
        q[key] = max(1, round(v * f))
    else:
        q[key] = v * f
    return q


def joint_draws(name: str, n: int = N_JOINT, seed: int = SEED) -> list[tuple[str, dict[str, Any]]]:
    u = units()[name]
    rng = np.random.default_rng(seed + sum(ord(ch) for ch in name))
    out = []
    for j in range(n):
        p = base_params(name)
        fs = {}
        for k in u.keys:
            f = float(rng.uniform(0.5, 1.5))
            fs[k] = round(f, 3)
            p = perturb(p, k, f)
        out.append((f"joint{j:02d}", p))
    return out


def variants_for(name: str) -> list[tuple[str, dict[str, Any]]]:
    """One-at-a-time (x0.5/0.8/1.2/1.5 per key) + joint draws; names key@f / jointNN."""
    u = units()[name]
    b = base_params(name)
    out = [(f"{k}@{f}", perturb(b, k, f)) for k in u.keys for f in PERTURB_FACTORS]
    return out + joint_draws(name)


# ---------------------------------------------------------------------- signals
def signal_of(name: str, frame: pl.DataFrame, params: dict[str, Any]) -> np.ndarray:
    p = {k: v for k, v in params.items() if not k.startswith("_")}
    if name == "zoo_ensemble":
        from ai_trader.strategies.trend_zoo import ensemble, zoo

        b = int(p.get("bpd", 6))
        return ensemble(frame, b, zoo(b)).to_numpy()
    from ai_trader.strategies import REGISTRY

    return REGISTRY[name](**p)(frame).to_numpy()


# ---------------------------------------------------------------------- per-symbol context
@dataclass
class Ctx:
    sym: str
    f1: pl.DataFrame          # 1h market frame + v0 labels
    f4: pl.DataFrame          # 4h market frame
    path: Path                # 1h path
    f1_flow: pl.DataFrame | None = None
    rb: dict | None = None    # post_cascade_study.load_symbol output
    f15: pl.DataFrame | None = None
    path15: Path | None = None
    har: dict | None = None   # date -> daily sigma forecast for the next day


def load_ctx(sym: str, as_of: datetime, *, venue: str = "bybit", need_rb: bool = True, need_15m: bool = False, need_flow: bool = False) -> Ctx | None:
    from ai_trader.research.frame import market_frame
    from ai_trader.research.regimes import label
    from ai_trader.storage.lake import Lake

    lake = Lake()
    byb = venue == "bybit"
    f1 = market_frame(sym, "1h", as_of, lake=lake, venue=venue, funding=byb, oi=byb, burn_in=byb)
    if f1.height < 1500:
        return None
    f1 = label(f1, 1.0)
    f4 = market_frame(sym, "4h", as_of, lake=lake, venue=venue, funding=False, oi=False, burn_in=byb) if byb else _agg_frame(f1, 4)
    ctx = Ctx(sym, f1, f4, Path.from_frame(f1, 60.0))
    if need_flow:
        ctx.f1_flow = market_frame(sym, "1h", as_of, lake=lake, flow=True)
    if need_rb and byb:
        from ai_trader.research.post_cascade_study import load_symbol

        ctx.rb = load_symbol(sym, as_of, lake)
    if need_15m:
        f15 = market_frame(sym, "15m", as_of, lake=lake, funding=False, oi=False)
        if f15.height > 2000:
            ctx.f15, ctx.path15 = f15, Path.from_frame(f15, 15.0)
    ctx.har = har_daily(f1)
    return ctx


def _agg_frame(f1: pl.DataFrame, k: int) -> pl.DataFrame:
    """k-hour bars from complete groups of 1h bars (aligned to UTC multiples of k hours), enriched like market_frame."""
    from ai_trader.research.frame import add_robust_vol, enrich_bars

    g = (f1.with_columns(pl.col("event_ts").dt.truncate(f"{k}h").alias("_g"))
         .group_by("_g").agg(pl.col("open").first(), pl.col("high").max(), pl.col("low").min(), pl.col("close").last(),
                             pl.col("volume").sum(), pl.col("turnover").sum(), pl.len().alias("_n"))
         .filter(pl.col("_n") == k).sort("_g")
         .with_columns(pl.col("_g").alias("event_ts"), (pl.col("_g") + pl.duration(hours=k)).alias("close_ts"))
         .with_columns(pl.col("close_ts").alias("known_ts"))
         .select("event_ts", "close_ts", "known_ts", "open", "high", "low", "close", "volume", "turnover"))
    return add_robust_vol(enrich_bars(g))


def har_daily(f1: pl.DataFrame) -> dict:
    from ai_trader.features.vol_models import har_forecast

    d = (f1.with_columns(pl.col("event_ts").dt.date().alias("d")).group_by("d")
         .agg((pl.col("ret").fill_nan(None).fill_null(0) ** 2).sum().alias("rv"), pl.len().alias("n")).filter(pl.col("n") == 24).sort("d"))
    if d.height < 40:
        return {}
    fc = har_forecast(d["rv"].to_numpy())
    return {dd: float(np.sqrt(v)) for dd, v in zip(d["d"].to_list(), fc, strict=True) if np.isfinite(v)}


def _ts(e: pl.Expr) -> pl.Expr:
    return pl.from_epoch(e, time_unit="ms").dt.replace_time_zone("UTC").dt.cast_time_unit("us")


def _ms(s: pl.Series) -> np.ndarray:
    return s.dt.epoch("ms").to_numpy().astype(np.int64)


def frame_for(ctx: Ctx, u: Unit) -> pl.DataFrame:
    if u.family == "flow":
        return ctx.f1_flow if ctx.f1_flow is not None else ctx.f1
    return ctx.f4 if u.freq == "4h" else ctx.f1


# ---------------------------------------------------------------------- bracket intents
def shock_events(f1: pl.DataFrame, z_min: float, decluster: int) -> list[int]:
    z = np.nan_to_num(f1["z"].to_numpy().astype(float))
    vs = f1["vol_state"].to_list()
    out, last = [], -(10**9)
    for t in np.flatnonzero(np.abs(z) >= z_min):
        if t - last > decluster:
            last = int(t)
            if vs[t] is not None:
                out.append(int(t))
    return out


def intents_shock(ctx: Ctx, p: dict, fade_only: bool = False, blackout: int = 0) -> list[Intent]:
    f = ctx.f1
    o, h, lo, c = (f[k].to_numpy().astype(float) for k in ("open", "high", "low", "close"))
    sn = f["sigma_next"].to_numpy().astype(float)
    sr = f["sigma_risk_next"].to_numpy().astype(float)
    z = f["z"].to_numpy().astype(float)
    vs = f["vol_state"].to_list()
    cms = _ms(f["close_ts"])
    edge = PRIOR_EDGE["shock_desk"]
    out = []
    for t in shock_events(f, float(p["z_min"]), int(p["decluster_bars"])):
        mode = "continuation" if vs[t] == "low" else "reversal"
        if fade_only and mode != "reversal":
            continue
        k = t + blackout
        if k >= f.height - 1:
            continue
        sgn = float(np.sign(z[t]))
        d = int(sgn if mode == "continuation" else -sgn)
        px = c[k]
        buf = float(p["rev_buffer"])
        if mode == "reversal":
            tp = o[t]
            stop = (lo[t] - buf * sn[t] * c[t]) if d > 0 else (h[t] + buf * sn[t] * c[t])
            if blackout and (np.any(lo[t + 1:k + 1] <= stop) if d > 0 else np.any(h[t + 1:k + 1] >= stop)):
                continue  # invalidated during the blackout
            if blackout and (np.any(h[t + 1:k + 1] >= tp) if d > 0 else np.any(lo[t + 1:k + 1] <= tp)):
                continue  # the fade target was already reached during the blackout
            tp_r = None
        else:
            stop = o[t]
            tp = px * math.exp(d * float(p["cont_tp_r"]) * abs(math.log(px / stop))) if stop > 0 and px != stop else float("nan")
            tp_r = float(p["cont_tp_r"])
        if not (np.isfinite(tp) and (tp - px) * d > 0 and (px - stop) * d > 0):
            continue
        cs = cap_stop(d, px, stop, tp, float(sr[k]), tp_r)
        if cs is None:
            continue
        stop, tp = cs
        pq = min(0.95, max(0.0, p_first(px, stop, tp) + edge))
        out.append(Intent(int(cms[k]), d, float(px), float(stop), float(tp), kind="market", ev=net_er(px, stop, tp, pq, "market"),
                          extra={"mode": mode}))
    return out


def intents_trend4h(ctx: Ctx, p: dict) -> list[Intent]:
    from ai_trader.research.calendar import add_calendar
    from ai_trader.strategies import REGISTRY

    f4 = ctx.f4
    if f4.height < 300 or not ctx.har:
        return []
    s = REGISTRY["ema_stack"](fast=int(p["fast"]), mid=int(p["mid"]), slow=int(p["slow"]), k=float(p["k"]))(f4).to_numpy()
    c = f4["close"].to_numpy().astype(float)
    sr = f4["sigma_risk_next"].to_numpy().astype(float)
    cts = f4["close_ts"]
    cms = _ms(cts)
    wk = add_calendar(pl.DataFrame({"event_ts": cts - timedelta(hours=1)}))["tradfi_weekend"].to_numpy()
    days = cts.dt.offset_by("-1d").dt.date().to_list()
    out = []
    for t in range(299, f4.height - 1):
        if abs(s[t]) < float(p["thr"]) or wk[t]:
            continue
        har = ctx.har.get(days[t])
        if har is None:
            continue
        d = 1 if s[t] > 0 else -1
        px = c[t]
        stop = px * math.exp(-d * float(p["stop_mult"]) * har)
        tp = px * math.exp(d * float(p["tp_r"]) * float(p["stop_mult"]) * har)
        cs = cap_stop(d, px, stop, tp, float(sr[t]), float(p["tp_r"]))
        if cs is None:
            continue
        stop, tp = cs
        pq = min(0.95, max(0.0, p_first(px, stop, tp) + PRIOR_EDGE["trend_4h"]))
        out.append(Intent(int(cms[t]), d, float(px), float(stop), float(tp), kind="peg", ev=net_er(px, stop, tp, pq, "peg")))
    return out


def intents_hrb2(ctx: Ctx, p: dict) -> list[Intent]:
    """post_cascade_study H-RB2 (flush -> rebound >= x sigma -> first close back below the EMA), parameterised."""
    if ctx.rb is None:
        return []
    f = ctx.rb["f"]
    k12 = 12
    f = f.with_columns(
        (pl.col("close").log() - pl.col("close").log().shift(k12)).alias("_r48"),
    ).with_columns(
        pl.col("_r48").rolling_quantile(float(p["r48_pct"]), window_size=2190, min_samples=1080).shift(1).alias("_thr"),
        (pl.col("open_interest") / pl.col("open_interest").rolling_max(18, min_samples=18) - 1).alias("_oidd"),
        pl.col("close").ewm_mean(span=int(p["ema_span"]), adjust=False).alias("_ema"),
    )
    cond = ((pl.col("_r48") < pl.col("_thr")) & (pl.col("_oidd") <= float(p["oi_drop"])) & (pl.col("cell") == "R9")).fill_null(False)
    idx = np.flatnonzero(f.select(cond).to_series().to_numpy())
    ts = f["event_ts"].to_list()
    flush, last = [], None
    for i in idx:
        if last is None or ts[i] - ts[last] >= timedelta(hours=4 * 42):
            flush.append(int(i))
            last = i
    c, lo, hi, ema = (f[x].to_numpy().astype(float) for x in ("close", "low", "high", "_ema"))
    sr = f["sigma_risk_next"].to_numpy().astype(float)
    cms = _ms(f["close_ts"])
    n = f.height
    W = int(p["rb2_window_bars"])
    out = []
    for t in flush:
        sig_t = float(sr[t])
        a0 = max(0, t - 11)
        arm, low_bar = None, None
        for k in range(t + 1, min(n, t + 1 + W)):
            if arm is None:
                jl = int(np.argmin(lo[a0:k + 1])) + a0
                if c[k] >= lo[jl] * math.exp(float(p["rebound_sigma"]) * sig_t):
                    arm, low_bar = k, jl
                continue
            if c[k] < ema[k] and c[k - 1] >= ema[k - 1] and k - 1 >= arm:
                reb_high = float(hi[low_bar:k + 1].max())
                entry, sg = float(c[k]), float(sr[k])
                if not (np.isfinite(sg) and sg > 0):
                    break
                stop = reb_high * math.exp(float(p["buffer_sigma"]) * sg)
                ks = abs(math.log(entry / stop)) / sg
                if stop <= entry:
                    break
                if ks > STOP_K:
                    ks = STOP_K
                    stop = entry * math.exp(ks * sg)
                if ks < MIN_SIG:
                    break
                tgt = entry * math.exp(-float(p["target_r"]) * ks * sg)
                pq = p_first(entry, stop, tgt)
                out.append(Intent(int(cms[k]), -1, entry, float(stop), float(tgt), time_stop_h=float(p["time_stop_hours"]), kind="peg",
                                  ev=net_er(entry, stop, tgt, pq, "peg")))
                break
    return out


def intents_setup_c(ctx: Ctx, p: dict) -> list[Intent]:
    """PLAN_2026-11-15 setup C, bar-only PROXY on 15m bars (see the registered design)."""
    f = ctx.f15
    if f is None:
        return []
    o, h, lo, c = (f[k].to_numpy().astype(float) for k in ("open", "high", "low", "close"))
    sr = f["sigma_risk_next"].to_numpy().astype(float)
    cms = _ms(f["close_ts"])
    n = f.height
    L = max(4, int(round(float(p["level_hours"]) * 4)))
    wait = max(1, int(round(float(p["wait_min"]) / 15)))
    rec = max(wait + 1, int(round(float(p["reclaim_hours"]) * 4)))
    lvl = pl.Series(lo).rolling_min(L, min_samples=L).shift(1).to_numpy()
    hmax = pl.Series(h).rolling_max(L, min_samples=L).shift(1).to_numpy()
    piv = np.zeros(n, dtype=bool)  # pivot high at m (4 bars each side), confirmed at m + 4
    if n > 9:
        w = np.lib.stride_tricks.sliding_window_view(h, 9).max(axis=1)
        piv[4:n - 4] = h[4:n - 4] >= w
    out = []
    i = L + 1
    while i < n - 1:
        if not (np.isfinite(lvl[i]) and lo[i] < lvl[i]):
            i += 1
            continue
        level = lvl[i]
        sweep_low = lo[i]
        done = i + rec
        for j in range(i + 1, min(n - 1, i + rec + 1)):
            sweep_low = min(sweep_low, lo[j])
            if j < i + wait or c[j] <= level:
                continue
            e, sg = float(c[j]), float(sr[j])
            stop = sweep_low * math.exp(-float(p["buffer_sigma"]) * sg)
            ms_ = np.flatnonzero(piv[max(0, i - L):max(0, i - 4)]) + max(0, i - L)
            cand = h[ms_][h[ms_] > e]
            tgt = float(cand.min()) if cand.size else (float(hmax[i]) if np.isfinite(hmax[i]) and hmax[i] > e else float("nan"))
            if np.isfinite(tgt) and stop < e:
                cs = cap_stop(1, e, stop, tgt, sg, None)
                if cs is not None:
                    stop, tgt = cs
                    out.append(Intent(int(cms[j]), 1, e, float(stop), float(tgt), time_stop_h=float(p["time_stop_hours"]), kind="peg",
                                      ev=net_er(e, stop, tgt, p_first(e, stop, tgt), "peg")))
            done = j
            break
        i = max(i + 1, done + 1)
    return out


def bracket_intents(ctx: Ctx, name: str, p: dict) -> tuple[list[Intent], Path | None]:
    if name == "trend_4h":
        return intents_trend4h(ctx, p), ctx.path
    if name == "shock_desk":
        return intents_shock(ctx, p), ctx.path
    if name == "hs2_fade_blackout":
        return intents_shock(ctx, p, fade_only=True, blackout=int(p["blackout_bars"])), ctx.path
    if name == "hrb2":
        return intents_hrb2(ctx, p), ctx.path
    if name == "setup_c_proxy":
        return intents_setup_c(ctx, p), ctx.path15
    raise KeyError(name)


# ---------------------------------------------------------------------- run one unit on one symbol
def run_unit(ctx: Ctx, u: Unit, params: dict[str, Any], stop_mode: str = "price", latency: float = 0.0,
             signal_cache: dict | None = None) -> list[dict]:
    if u.kind == "signal":
        fr = frame_for(ctx, u)
        key = (u.name, json.dumps({k: v for k, v in params.items() if not k.startswith("_")}, sort_keys=True, default=str))
        if signal_cache is not None and key in signal_cache:
            s = signal_cache[key]
        else:
            s = signal_of(u.name, fr, params)
            if signal_cache is not None:
                signal_cache[key] = s
        tr = walk_signal(s, _ms(fr["close_ts"]), fr["close"].to_numpy().astype(float), fr["sigma_risk_next"].to_numpy().astype(float),
                         ctx.path, theta=float(params.get("_theta", THETA_IN)), stop_k=float(params.get("_stopk", STOP_K)),
                         stop_mode=stop_mode, latency=latency)
        for t in tr:
            t["kind"], t["ev"] = u.entry_kind, float("nan")
        return tr
    if stop_mode == "none" and u.name in ("trend_4h", "shock_desk", "hs2_fade_blackout"):
        return []  # no time stop: without an exchange stop the bracket never resolves (n/a)
    it, path = bracket_intents(ctx, u.name, params)
    if path is None:
        return []
    return walk_bracket(it, path, stop_mode=stop_mode, latency=latency)


def annotate(tr: list[dict], ctx: Ctx, unit: str, variant: str) -> pl.DataFrame:
    """Trade table with entry context: v0 cell (1h), segment, weekend, NYSE holiday, funding window, PIT tier."""
    from ai_trader.research.calendar import add_calendar
    from ai_trader.research.evaluate import knowledge_cutoff

    if not tr:
        return pl.DataFrame()
    df = pl.DataFrame(tr).with_columns(
        pl.lit(unit).alias("unit"), pl.lit(variant).alias("variant"), pl.lit(ctx.sym).alias("symbol"),
        _ts(pl.col("entry_ms")).alias("entry_ts"), _ts(pl.col("exit_ms")).alias("exit_ts"),
    )
    cells = ctx.f1.select(pl.col("close_ts").dt.convert_time_zone("UTC").dt.cast_time_unit("us").alias("_k"), pl.col("cell")).sort("_k")
    df = df.sort("entry_ts").join_asof(cells, left_on="entry_ts", right_on="_k", strategy="backward").drop("_k")
    cal = add_calendar(df.select((pl.col("entry_ts") - timedelta(minutes=1)).alias("event_ts")))
    tov = (ctx.f1.with_columns(pl.col("event_ts").dt.date().alias("d")).group_by("d").agg(pl.col("turnover").sum().alias("tv")).sort("d")
           .with_columns(pl.col("tv").rolling_median(30, min_samples=10).shift(1).alias("tv30")))
    tv_map = dict(zip(tov["d"].to_list(), tov["tv30"].to_list(), strict=True))
    ed = df["entry_ts"].dt.date().to_list()
    ny = df["entry_ts"].dt.convert_time_zone("America/New_York").dt.date().to_list()
    r8 = (df["entry_ms"].to_numpy() % (8 * H_MS))
    k = knowledge_cutoff()
    return df.with_columns(
        cal["tradfi_weekend"].alias("weekend"),
        pl.Series("holiday", [d in NYSE_HOLIDAYS for d in ny]),
        pl.Series("funding_win", np.minimum(r8, 8 * H_MS - r8) <= H_MS),
        pl.Series("tier", [tier_of(float(tv_map.get(d) or float("nan"))) for d in ed]),
        pl.when(pl.col("entry_ts") < k).then(pl.lit("llm_in_sample")).otherwise(pl.lit("clean_holdout")).alias("seg"),
    )


TRADE_COLS = ("unit", "variant", "symbol", "side", "entry_ts", "exit_ts", "entry", "stop", "exit", "why", "gap", "kind", "ev",
              "cell", "seg", "weekend", "holiday", "funding_win", "tier")


def _norm(df: pl.DataFrame) -> pl.DataFrame:
    if df.height == 0:
        return df
    return df.select([pl.col(c) for c in TRADE_COLS]).with_columns(pl.col("side").cast(pl.Int8), pl.col("ev").cast(pl.Float64),
                                                                    pl.col("cell").cast(pl.Utf8), pl.col("gap").cast(pl.Boolean))


def run_symbol(sym: str, as_of_iso: str, unit_names: list[str] | None = None, perturbations: bool = False, out_dir: str | None = None) -> dict:
    """Base + execution variants (all units applying to sym); with perturbations=True the S5 variants (base exec only)."""
    t0 = time.time()
    as_of = datetime.fromisoformat(as_of_iso)
    U = units()
    names = [n for n in (unit_names or list(U)) if U[n].symbols is None or sym in U[n].symbols]
    ctx = load_ctx(sym, as_of, need_rb="hrb2" in names, need_15m="setup_c_proxy" in names, need_flow=any(U[n].family == "flow" for n in names))
    if ctx is None:
        return {"symbol": sym, "skipped": "too little history"}
    frames, cache = [], {}
    timing: dict[str, float] = {}
    for n in names:
        tu = time.time()
        u = U[n]
        bp = base_params(n)
        if perturbations:
            for vname, p in [("base", bp), *variants_for(n)]:
                tr = run_unit(ctx, u, p, signal_cache=cache)
                frames.append(_norm(annotate(tr, ctx, n, vname)))
        else:
            for vname, (mode, lat) in EXEC_VARIANTS.items():
                tr = run_unit(ctx, u, bp, stop_mode=mode, latency=lat, signal_cache=cache)
                frames.append(_norm(annotate(tr, ctx, n, vname)))
        cache.clear()
        timing[n] = round(time.time() - tu, 1)
    frames = [f for f in frames if f.height]
    df = pl.concat(frames, how="vertical_relaxed") if frames else pl.DataFrame()
    od = FsPath(out_dir) if out_dir else OUT / ("perturb" if perturbations else "trades")
    od.mkdir(parents=True, exist_ok=True)
    if sym == "BTCUSDT" and not perturbations:  # S7 down-state series
        ctx.f1.select(pl.col("close_ts").dt.epoch("ms").alias("ms"), pl.col("cell")).write_parquet(od.parent / "btc_state.parquet")
    if df.height:
        tmp = od / f"{sym}.parquet.tmp"
        df.write_parquet(tmp)
        os.replace(tmp, od / f"{sym}.parquet")
    return {"symbol": sym, "trades": df.height, "seconds": round(time.time() - t0, 1), "unit_seconds": timing}


# ====================================================================== metrics
def stationary_boot_idx(n_days: int, n_boot: int, block: float, rng: np.random.Generator) -> np.ndarray:
    from ai_trader.sim.resample import stationary_indices

    return stationary_indices(n_days, n_days, block, n_boot, rng)


def daily(t: pl.DataFrame, r: np.ndarray, by: str = "entry_ts") -> pl.DataFrame:
    """Per calendar day (UTC) of `by`: n and sum R, every day of the span included (zeros)."""
    if t.height == 0:
        return pl.DataFrame({"d": [], "n": [], "R": []})
    d = t.select(pl.col(by).dt.date().alias("d")).with_columns(pl.Series("R", r)).group_by("d").agg(pl.len().alias("n"), pl.col("R").sum())
    span = pl.DataFrame({"d": pl.date_range(d["d"].min(), d["d"].max(), "1d", eager=True)})
    return span.join(d, on="d", how="left").fill_null(0).sort("d")


def mean_ci(t: pl.DataFrame, r: np.ndarray, n_boot: int = N_BOOT, seed: int = SEED) -> tuple[float, float, float]:
    """Mean R per trade with a 95% CI from a stationary bootstrap of entry days (mean block 5 days)."""
    if len(r) == 0:
        return float("nan"), float("nan"), float("nan")
    dd = daily(t, r)
    n, s = dd["n"].to_numpy().astype(float), dd["R"].to_numpy().astype(float)
    if len(n) < 3:
        return float(np.mean(r)), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    idx = stationary_boot_idx(len(n), n_boot, BLOCK_DAYS, rng)
    ns, ss = n[idx].sum(axis=1), s[idx].sum(axis=1)
    m = ss[ns > 0] / ns[ns > 0]
    return float(np.mean(r)), float(np.quantile(m, 0.025)), float(np.quantile(m, 0.975))


def drawdown_stats(t: pl.DataFrame, r: np.ndarray) -> dict:
    """Max DD (R) on the realised curve (exit order), daily CVaR5%, share of days under water, longest spell (days)."""
    if len(r) == 0:
        return {"max_dd": float("nan"), "cvar5_day": float("nan"), "underwater_share": float("nan"), "longest_uw_days": 0}
    o = np.argsort(t["exit_ts"].to_numpy(), kind="stable")
    cum = np.cumsum(r[o])
    dd = cum - np.maximum.accumulate(np.r_[0.0, cum])[1:]
    dly = daily(t, r, by="exit_ts")
    dv = dly["R"].to_numpy()
    act = dv[dly["n"].to_numpy() > 0]
    q = max(1, int(math.ceil(0.05 * len(act)))) if len(act) else 0
    cvar_d = float(np.sort(act)[:q].mean()) if q else float("nan")
    dc = np.cumsum(dv)
    uw = dc < np.maximum.accumulate(np.r_[0.0, dc])[1:] - 1e-12
    runs, cur = [], 0
    for x in uw:
        cur = cur + 1 if x else 0
        runs.append(cur)
    return {"max_dd": float(dd.min()) if len(dd) else 0.0, "cvar5_day": cvar_d, "underwater_share": float(uw.mean()) if len(uw) else float("nan"),
            "longest_uw_days": int(max(runs) if runs else 0)}


def metrics(t: pl.DataFrame, r: np.ndarray, ci: bool = True) -> dict:
    n = len(r)
    if n == 0:
        return {"n": 0}
    m, lo, hi = mean_ci(t, r) if ci else (float(np.mean(r)), float("nan"), float("nan"))
    pos, neg = r[r > 0].sum(), -r[r < 0].sum()
    q = max(1, int(math.ceil(0.05 * n)))
    srt = np.sort(r)
    total = float(r.sum())
    top5 = float(srt[-5:].sum()) if n >= 5 else float(srt.sum())
    out = {"n": n, "mean": m, "lo": lo, "hi": hi, "hit": float((r > 0).mean()), "pf": float(pos / neg) if neg > 0 else float("inf"),
           "total": total, "cvar5_trade": float(srt[:q].mean()), "worst": float(srt[0]), "best": float(srt[-1]),
           "top5_share": float(top5 / total) if total > 0 else float("nan"), "total_ex_top5": total - top5}
    out.update(drawdown_stats(t, r))
    return out


# ====================================================================== scenarios on trade tables
def crisis_windows(btc_1d: pl.DataFrame | None = None) -> dict[str, tuple[date, date]]:
    """FIXED_CRISES + the 3 largest non-overlapping BTC 14-day close-to-close declines with the trough in 2025-01-01..2026-11-18."""
    w = {k: (date.fromisoformat(a), date.fromisoformat(b)) for k, (a, b) in FIXED_CRISES.items()}
    if btc_1d is None:
        from ai_trader.core.time import now
        from ai_trader.storage.lake import Lake

        btc_1d = Lake().bars("BTCUSDT", "1d", now()).select("event_ts", "close").sort("event_ts")
    c = btc_1d["close"].to_numpy().astype(float)
    ds = btc_1d["event_ts"].dt.date().to_list()
    r14 = np.full(len(c), np.nan)
    r14[14:] = np.log(c[14:] / c[:-14])
    cand = [(r14[i], ds[i]) for i in range(len(c)) if np.isfinite(r14[i]) and date(2025, 1, 1) <= ds[i] <= date(2026, 11, 18)]
    cand.sort()
    picked: list[date] = []
    for _, d in cand:
        if all(abs((d - p).days) > 30 for p in picked):
            picked.append(d)
        if len(picked) == 3:
            break
    for p in picked:
        k = float(r14[ds.index(p)])
        w[f"BTC 14d drop to {p} ({k:+.1%})"] = (p - timedelta(days=14), p + timedelta(days=3))
    return w


def window_stats(t: pl.DataFrame, r: np.ndarray, a: date, b: date) -> dict:
    if t.height == 0:
        return {"n": 0}
    m = ((t["entry_ts"].dt.date() >= a) & (t["entry_ts"].dt.date() <= b)).to_numpy()
    if not m.any():
        return {"n": 0}
    sub, rr = t.filter(pl.Series(m)), r[m]
    o = np.argsort(sub["exit_ts"].to_numpy(), kind="stable")
    cum = np.cumsum(rr[o])
    dd = float((cum - np.maximum.accumulate(np.r_[0.0, cum])[1:]).min())
    stop = (sub["why"] == "stop").to_numpy()
    e, s, x, d = (sub[k].to_numpy().astype(float) for k in ("entry", "stop", "exit", "side"))
    den = np.abs(e - s) + e * base_budget(sub["kind"].to_numpy())
    gap_r = (d * (x - s) / den)[stop]
    return {"n": int(m.sum()), "sum": float(rr.sum()), "max_dd": dd, "worst": float(rr.min()), "stop_gap_worst_R": float(gap_r.min()) if gap_r.size else 0.0,
            "stops": int(stop.sum())}


def co_stop(t: pl.DataFrame, down_ms: np.ndarray, down_flag: np.ndarray) -> pl.DataFrame:
    """S7: in down states (BTC v0 cell R7/R8/R9 at that time) a stop-out closes every concurrent same-side open trade of
    the unit at its own stop price (rho -> 1, one-sided). Returns a modified copy (why='co_stop')."""
    if t.height == 0:
        return t
    t = t.sort("entry_ts")
    ems = _ms(t["entry_ts"])
    xms = _ms(t["exit_ts"]).copy()
    side = t["side"].to_numpy()
    why = t["why"].to_numpy().astype(object).copy()
    ex = t["exit"].to_numpy().astype(float).copy()
    st = t["stop"].to_numpy().astype(float)
    order = np.argsort(xms, kind="stable")
    for i in order:
        if why[i] != "stop":
            continue
        tau = xms[i]
        j = np.searchsorted(down_ms, tau, side="right") - 1
        if j < 0 or not down_flag[j]:
            continue
        hit = (side == side[i]) & (ems < tau) & (xms > tau)
        hit[i] = False
        if hit.any():
            xms[hit] = tau
            ex[hit] = st[hit]
            why[hit] = "co_stop"
    return t.with_columns(pl.Series("_x", xms), pl.Series("exit", ex), pl.Series("why", why.astype(str))).with_columns(
        _ts(pl.col("_x")).alias("exit_ts")).drop("_x")


# ====================================================================== live risk engine (live_small)
@dataclass(frozen=True)
class LiveLimits:
    risk_usd: float = 1.30
    max_loss_usd: float = 2.00
    max_entries_day: int = 20
    daily_stop_usd: float = 10.0
    max_open: int = 3
    max_same_side: int = 2
    max_gross_usd: float = 400.0
    max_stop_pct: float = 10.0
    min_ev: float = 0.10
    ev_gate: bool = True

    @classmethod
    def from_config(cls, **kw: Any) -> LiveLimits:
        from ai_trader.core.config import load

        ls = load().risk["live_small"]
        return cls(risk_usd=float(ls["risk_to_stop_usd"]), max_loss_usd=float(ls["max_loss_per_trade_usd"]), max_entries_day=int(ls["max_entries_per_day"]),
                   daily_stop_usd=float(ls["daily_loss_stop_usd"]), max_open=int(ls["max_open_positions"]), max_same_side=int(ls["max_same_direction"]),
                   max_gross_usd=float(ls["max_gross_notional_usd"]), max_stop_pct=float(ls["max_stop_distance_pct"]),
                   min_ev=float(ls["min_net_expected_r"]), **kw)


def live_engine(t: pl.DataFrame, r: np.ndarray, lim: LiveLimits | None = None) -> tuple[pl.DataFrame, dict]:
    """Apply live_small to a pooled trade stream (first come, first served by entry time). Returns the accepted trades
    with usd P&L and risk_usd, and refusal counts. Positions free their slot at exit_ts (exits before entries at equal ts)."""
    lim = lim or LiveLimits()
    if t.height == 0:
        return t.with_columns(pl.lit(0.0).alias("usd")), {"refused": {}}
    rank = {s_: i for i, s_ in enumerate(_universe())}  # ties at one timestamp: universe order (liquidity) first
    t = (t.with_columns(pl.Series("R", r), pl.col("symbol").replace_strict(rank, default=99, return_dtype=pl.Int32).alias("_rk"))
         .sort(["entry_ts", "_rk"]).drop("_rk"))
    e, s = t["entry"].to_numpy().astype(float), t["stop"].to_numpy().astype(float)
    kind, tier = t["kind"].to_numpy(), t["tier"].to_numpy()
    stop_pct = np.abs(e - s) / e
    budget = stop_pct + base_budget(kind)
    ems, xms = _ms(t["entry_ts"]), _ms(t["exit_ts"])
    side, R, ev = t["side"].to_numpy(), t["R"].to_numpy(), t["ev"].to_numpy().astype(float)
    open_: list[tuple[int, int, float, float, float, int]] = []  # (exit_ms, side, notional, worst_usd, usd, idx)
    keep = np.zeros(t.height, dtype=bool)
    risk = np.zeros(t.height)
    refused: dict[str, int] = {}
    day, entries, realised = None, 0, 0.0
    day_hit: dict[date, bool] = {}
    day_refused_budget: dict[date, bool] = {}
    day_real: dict[date, float] = {}

    def ref(k: str) -> None:
        refused[k] = refused.get(k, 0) + 1

    def close_until(ts: int) -> None:
        nonlocal realised
        for p in sorted([p for p in open_ if p[0] <= ts]):
            open_.remove(p)
            dd = datetime.fromtimestamp(p[0] / 1000, UTC).date()
            day_real[dd] = day_real.get(dd, 0.0) + p[4]
            if dd == day:
                realised += p[4]
                if realised <= -lim.daily_stop_usd:
                    day_hit[dd] = True

    for i in range(t.height):
        close_until(int(ems[i]))
        dd = datetime.fromtimestamp(ems[i] / 1000, UTC).date()
        if dd != day:
            day, entries, realised = dd, 0, day_real.get(dd, 0.0)
        tr = TIERS.get(str(tier[i]))
        if tr is None:
            ref("tier")
            continue
        if stop_pct[i] * 100 > lim.max_stop_pct:
            ref("stop_distance")
            continue
        if lim.ev_gate and np.isfinite(ev[i]) and ev[i] < lim.min_ev:
            ref("ev_gate")
            continue
        if entries >= lim.max_entries_day:
            ref("entries_day")
            continue
        if len(open_) >= lim.max_open:
            ref("max_open")
            continue
        if sum(1 for p in open_ if p[1] == side[i]) >= lim.max_same_side:
            ref("same_side")
            continue
        rk = lim.risk_usd
        worst = rk * (budget[i] + (tr["stop_slip"] - 2.0) / 1e4) / budget[i]
        if worst > lim.max_loss_usd:
            rk *= lim.max_loss_usd / worst
            worst = lim.max_loss_usd
        notional = rk / budget[i]
        if notional > tr["cap"]:
            rk *= tr["cap"] / notional
            worst *= tr["cap"] / notional
            notional = tr["cap"]
        if sum(p[2] for p in open_) + notional > lim.max_gross_usd:
            ref("gross")
            continue
        if realised <= -lim.daily_stop_usd:
            ref("daily_stop")
            day_refused_budget[dd] = True
            continue
        if realised - sum(p[3] for p in open_) - worst < -lim.daily_stop_usd:
            ref("daily_budget")
            day_refused_budget[dd] = True
            continue
        entries += 1
        keep[i] = True
        risk[i] = rk
        open_.append((int(xms[i]), int(side[i]), notional, worst, float(R[i] * rk), i))
    close_until(2**62)
    for dd, v in day_real.items():
        if v <= -lim.daily_stop_usd:
            day_hit[dd] = True
    acc = t.with_columns(pl.Series("risk_usd", risk), pl.Series("usd", R * risk)).filter(pl.Series(keep))
    return acc, {"refused": refused, "days_stop_hit": sorted(str(d) for d, v in day_hit.items() if v),
                 "days_budget_refusal": sorted(str(d) for d, v in day_refused_budget.items() if v)}


def run_prob_loss_streak(n_trades: int, p_loss: float, k: int = 5) -> float:
    """P(at least one run of >= k consecutive losers in n iid trades) by dynamic programming."""
    if n_trades < k or not (0 < p_loss < 1):
        return 0.0 if p_loss < 1 else float(n_trades >= k)
    st = np.zeros(k)
    st[0] = 1.0
    hit = 0.0
    for _ in range(n_trades):
        new = np.zeros(k)
        new[0] = st.sum() * (1 - p_loss)
        new[1:] = st[:-1] * p_loss
        hit += st[-1] * p_loss
        st = new
    return float(hit)


def engine_stats(acc: pl.DataFrame, info: dict, horizon_days: int = 91, n_boot: int = 5000, seed: int = SEED) -> dict:
    if acc.height == 0:
        return {"n": 0, "refused": info.get("refused", {})}
    usd = acc["usd"].to_numpy()
    dly = daily(acc, usd, by="exit_ts")  # column R holds USD here
    dv = dly["R"].to_numpy()
    active = dly["n"].to_numpy() > 0
    n_days = len(dv)
    rng = np.random.default_rng(seed)
    idx = stationary_boot_idx(n_days, n_boot, BLOCK_DAYS, rng)[:, :min(horizon_days, n_days)] if n_days >= 10 else None
    ruin = {}
    if idx is not None:
        paths = np.cumsum(dv[idx], axis=1)
        mins = paths.min(axis=1)
        for x in (25, 50, 100):
            ruin[f"p_loss_ge_{x}"] = float((mins <= -x).mean())
        ruin["p_3m_negative"] = float((paths[:, -1] < 0).mean())
        ruin["q05_3m_usd"] = float(np.quantile(paths[:, -1], 0.05))
    o = np.argsort(acc["exit_ts"].to_numpy(), kind="stable")
    loss = usd[o] < 0
    run, mx, runs5 = 0, 0, 0
    for x in loss:
        run = run + 1 if x else 0
        mx = max(mx, run)
        runs5 += run == 5
    ex = acc["exit_ts"].to_numpy()[o]
    win_hits, wins = 0, 0
    if len(ex):
        t0, t1 = ex[0], ex[-1]
        step = np.timedelta64(7, "D")
        cur = t0
        while cur + np.timedelta64(horizon_days, "D") <= t1:
            m = (ex >= cur) & (ex < cur + np.timedelta64(horizon_days, "D"))
            r_, b_ = 0, 0
            for x in loss[m]:
                r_ = r_ + 1 if x else 0
                b_ = max(b_, r_)
            win_hits += b_ >= 5
            wins += 1
            cur = cur + step
    trades_per_3m = acc.height / max(1.0, n_days / horizon_days)
    dc = np.cumsum(dv)
    uw = dc < np.maximum.accumulate(np.r_[0.0, dc])[1:] - 1e-9
    spells, cur_ = [], 0
    for x in uw:
        if x:
            cur_ += 1
        elif cur_:
            spells.append(cur_)
            cur_ = 0
    if cur_:
        spells.append(cur_)
    act_days = int(active.sum())
    return {"n": acc.height, "usd_total": float(usd.sum()), "usd_per_trade": float(usd.mean()), "days": n_days, "active_days": act_days,
            "p_daily_stop_hit": len(info.get("days_stop_hit", [])) / act_days if act_days else float("nan"),
            "p_daily_budget_refusal": len(info.get("days_budget_refusal", [])) / act_days if act_days else float("nan"),
            "worst_day_usd": float(dv.min()), "max_loss_run": mx, "p5_losers_3m_empirical": win_hits / wins if wins else float("nan"),
            "p5_losers_3m_iid": run_prob_loss_streak(int(round(trades_per_3m)), float(loss.mean())),
            "mean_uw_spell_days": float(np.mean(spells)) if spells else 0.0, "max_uw_spell_days": int(max(spells) if spells else 0),
            "underwater_share": float(uw.mean()), "refused": info.get("refused", {}), **ruin}


# ====================================================================== verdict rules (frozen, H-STRESS0)
def verdict(m: dict) -> tuple[str, list[str]]:
    """m: base (dict from metrics), fees15, slip5, lat5, holdout (dicts), cliff (bool | None), crisis_engine_min (float)."""
    fired = []
    b = m.get("base") or {}
    if not b.get("n"):
        return "n/a", ["no trades"]
    if not (b["mean"] > 0):
        fired.append("F1 base mean <= 0")
    f15 = m.get("fees15") or {}
    if f15.get("n") and not (f15["mean"] > 0):
        fired.append("F2 fees x1.5 mean <= 0")
    if fired:
        return "FAILS", fired
    if m.get("cliff"):
        fired.append("G1 +-20% parameter cliff")
    lat = m.get("lat5") or {}
    if lat.get("n") and not (lat["mean"] > 0):
        fired.append("G2 5-min latency mean <= 0")
    if m.get("crisis_engine_min") is not None and m["crisis_engine_min"] < -20:
        fired.append("G3 crisis window < -20 R on the engine stream")
    s5 = m.get("slip5") or {}
    if s5.get("n") and not (s5["mean"] > 0):
        fired.append("G4 +5 bp per side mean <= 0")
    ho = m.get("holdout") or {}
    if ho.get("n", 0) >= 20 and not (ho["mean"] > 0):
        fired.append("G5 clean_holdout mean <= 0")
    if not (np.isfinite(b.get("lo", np.nan)) and b["lo"] > 0):
        fired.append("G6 base 95% CI includes 0")
    if not (b.get("total_ex_top5", 0) > 0):
        fired.append("G7 total without the 5 best trades <= 0")
    return ("FRAGILE", fired) if fired else ("SURVIVES", [])


# ====================================================================== synthetic paths (S6)
def synth_frames(o: np.ndarray, h: np.ndarray, lo: np.ndarray, c: np.ndarray, start: datetime) -> tuple[pl.DataFrame, pl.DataFrame]:
    from ai_trader.research.frame import add_robust_vol, enrich_bars
    from ai_trader.research.regimes import label

    n = len(c)
    ts = pl.datetime_range(start, start + timedelta(hours=n - 1), "1h", eager=True, time_zone="UTC")
    b = pl.DataFrame({"event_ts": ts, "open": o, "high": h, "low": lo, "close": c}).with_columns(
        (pl.col("event_ts") + timedelta(hours=1)).alias("close_ts"), pl.lit(1.0).alias("volume"), pl.lit(1.0).alias("turnover"))
    b = b.with_columns(pl.col("close_ts").alias("known_ts")).select("event_ts", "close_ts", "known_ts", "open", "high", "low", "close", "volume", "turnover")
    f1 = label(add_robust_vol(enrich_bars(b)), 1.0)
    return f1, _agg_frame(f1, 4)


def synth_unit_stats(ctx: Ctx, names: list[str], warm_ms: int) -> list[dict]:
    U = units()
    out = []
    for n in names:
        u = U[n]
        tr = run_unit(ctx, u, base_params(n))
        tr = [x for x in tr if x["entry_ms"] >= warm_ms]
        if tr:
            df = pl.DataFrame(tr).with_columns(pl.lit("liquid").alias("tier"))
            r = r_net(df)
            o = np.argsort(df["exit_ms"].to_numpy(), kind="stable")
            cum = np.cumsum(r[o])
            mdd = float((cum - np.maximum.accumulate(np.r_[0.0, cum])[1:]).min())
            out.append({"unit": n, "n": len(r), "mean": float(r.mean()), "total": float(r.sum()), "max_dd": mdd, "hit": float((r > 0).mean())})
        else:
            out.append({"unit": n, "n": 0, "mean": float("nan"), "total": 0.0, "max_dd": 0.0, "hit": float("nan")})
    return out


def run_synth_chunk(src: str, gen: str, path_ids: list[int], as_of_iso: str, n_year: int = 8760, warm: int = 2000) -> list[dict]:
    from ai_trader.research.frame import market_frame
    from ai_trader.research.regimes import label
    from ai_trader.sim import resample as rs

    as_of = datetime.fromisoformat(as_of_iso)
    f = market_frame(src, "1h", as_of, funding=False, oi=False)
    f = label(f, 1.0)
    rb = rs.relative_bars(*(f[k].to_numpy().astype(float) for k in ("open", "high", "low", "close")))
    pw = rs.politis_white_block(np.abs(rb.r))
    n_out = n_year + warm
    names = [n for n, u in units().items() if u.synthetic]
    cells = f["cell"].to_list()[1:]
    fit = rs.fit_transition(cells)
    ix = {s_: i for i, s_ in enumerate(fit["states"])}
    src_state = np.array([ix.get(x, -1) if x is not None else -1 for x in cells])
    out = []
    for pid in path_ids:
        rng = np.random.default_rng(SEED + 7919 * pid)
        if gen == "sbb":
            idx = rs.stationary_indices(len(rb), n_out, pw["b"], 1, rng)
        else:
            chain = rs.simulate_chain(fit["P"], n_out, 1, rng, start_probs=fit["freq"])
            idx = rs.regime_indices(src_state, chain, pw["b"], rng)
        ohlc = rs.reconstruct(rb, idx, float(f["close"][0]))
        start = datetime(2031, 1, 6, tzinfo=UTC)
        f1, f4 = synth_frames(*(ohlc[k][0] for k in ("open", "high", "low", "close")), start)
        ctx = Ctx(f"SYN-{src}", f1, f4, Path.from_frame(f1, 60.0))
        ctx.har = har_daily(f1)
        warm_ms = int((start + timedelta(hours=warm)).timestamp() * 1000)
        for row in synth_unit_stats(ctx, names, warm_ms):
            out.append({"path": pid, "gen": gen, "src": src, "block": pw["b"], **row})
    return out


# ====================================================================== orchestration
def design_dict(as_of: str) -> dict:
    U = units()
    return {"as_of": as_of, "seed": SEED, "theta_in": THETA_IN, "stop_k": STOP_K, "min_sig": MIN_SIG, "D8": D8,
            "synth_sources": SYNTH_SOURCES, "perturb_factors": PERTURB_FACTORS, "n_joint": N_JOINT, "n_boot": N_BOOT, "block_days": BLOCK_DAYS,
            "units": {k: asdict(v) for k, v in U.items()}, "base_params": {k: base_params(k) for k in U},
            "exec_variants": EXEC_VARIANTS, "cost_variants": COST_VARIANTS, "fixed_crises": FIXED_CRISES, "tiers": TIERS,
            "prior_edge": PRIOR_EDGE, "live_limits": asdict(LiveLimits.from_config()), "costs": asdict(Costs.live()),
            "verdict": "FAILS: F1 base <= 0 | F2 fees x1.5 <= 0; FRAGILE: G1 cliff | G2 lat5 <= 0 | G3 crisis < -20 R (engine) | G4 slip+5 <= 0 | "
                       "G5 holdout <= 0 (n >= 20) | G6 CI includes 0 | G7 total ex top-5 <= 0; else SURVIVES"}


def _universe() -> list[str]:
    from ai_trader.research.trend_zoo_study import UNIVERSE

    return list(UNIVERSE)


def _pool(workers: int):
    return ProcessPoolExecutor(max_workers=workers)


def cmd_run(symbols: list[str] | None, workers: int, perturbations: bool) -> None:
    meta = json.loads((OUT / "design.json").read_text(encoding="utf-8"))
    syms = symbols or (list(D8) if perturbations else _universe())
    t0 = time.time()
    with _pool(workers) as ex:
        futs = {ex.submit(run_symbol, s, meta["as_of"], None, perturbations): s for s in syms}
        for f in futs:
            try:
                print(json.dumps(f.result()), flush=True)
            except Exception as e:  # one symbol must not stop the run
                print(json.dumps({"symbol": futs[f], "error": repr(e)}), flush=True)
    print(f"done in {time.time() - t0:.0f}s", flush=True)


def cmd_deribit() -> None:
    """2020-03 window on Deribit BTC-PERPETUAL 1h (bar units only; exec variants)."""
    meta = json.loads((OUT / "design.json").read_text(encoding="utf-8"))
    as_of = datetime.fromisoformat(meta["as_of"])
    ctx = load_ctx("BTC-PERPETUAL", as_of, venue="deribit", need_rb=False)
    if ctx is None:
        print("no deribit data")
        return
    ctx.f1 = ctx.f1.filter(pl.col("event_ts") < datetime(2020, 7, 1, tzinfo=UTC))
    ctx.f4 = ctx.f4.filter(pl.col("event_ts") < datetime(2020, 7, 1, tzinfo=UTC))
    ctx.path = Path.from_frame(ctx.f1, 60.0)
    ctx.har = har_daily(ctx.f1)
    frames = []
    for n, u in units().items():
        if u.family in ("positioning", "flow") or n in ("hrb2", "setup_c_proxy"):
            continue
        for vname in ("base", "gap", "stop_next_open"):
            mode, lat = EXEC_VARIANTS[vname]
            frames.append(_norm(annotate(run_unit(ctx, u, base_params(n), stop_mode=mode, latency=lat), ctx, n, vname)))
    df = pl.concat([f for f in frames if f.height], how="vertical_relaxed").with_columns(pl.lit("liquid").alias("tier"))
    (OUT / "deribit").mkdir(parents=True, exist_ok=True)
    df.write_parquet(OUT / "deribit" / "BTC-PERPETUAL_2020H1.parquet")
    print("deribit trades", df.height)


def cmd_synth(workers: int, paths: int) -> None:
    meta = json.loads((OUT / "design.json").read_text(encoding="utf-8"))
    jobs = []
    half = paths // 2
    for gen, base in (("sbb", 0), ("rs", half)):
        ids = list(range(base, base + half))
        for k, src in enumerate(SYNTH_SOURCES):
            mine = ids[k::len(SYNTH_SOURCES)]
            for j in range(0, len(mine), 25):
                jobs.append((src, gen, mine[j:j + 25]))
    od = OUT / "synthetic"
    od.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    with _pool(workers) as ex:
        futs = {ex.submit(run_synth_chunk, s, g, ids, meta["as_of"]): (s, g, ids[0]) for s, g, ids in jobs
                if not (od / f"{g}_{s}_{ids[0]:04d}.parquet").exists()}
        for f, (s, g, i0) in futs.items():
            try:
                rows = f.result()
                pl.DataFrame(rows).write_parquet(od / f"{g}_{s}_{i0:04d}.parquet")
                print(f"{g} {s} {i0}: {len(rows)} rows, {time.time() - t0:.0f}s", flush=True)
            except Exception as e:
                print(f"{g} {s} {i0}: ERROR {e!r}", flush=True)


def cmd_latcheck() -> dict:
    """Bridge vs exact (BTCUSDT 5m) latency fills for the 1h bar units on BTC, pooled; writes latcheck.json."""
    from ai_trader.storage.lake import Lake

    meta = json.loads((OUT / "design.json").read_text(encoding="utf-8"))
    as_of = datetime.fromisoformat(meta["as_of"])
    ctx = load_ctx("BTCUSDT", as_of, need_rb=False)
    b5 = Lake().bars("BTCUSDT", "5m", as_of).select("event_ts", "open", "close").sort("event_ts")
    m5 = _ms(b5["event_ts"])
    o5 = b5["open"].to_numpy().astype(float)
    lat = {}
    for m in (5.0, 15.0):
        j = np.searchsorted(m5, ctx.path.ts + int(m * 60_000))
        ok = (j < len(m5)) & (m5[np.minimum(j, len(m5) - 1)] == ctx.path.ts + int(m * 60_000))
        lat[m] = np.where(ok, o5[np.minimum(j, len(m5) - 1)], np.nan)
    end5 = int(m5.max())
    res = {}
    for n, u in units().items():
        if u.freq != "1h" or u.family == "flow" or n in ("setup_c_proxy",):
            continue
        row = {}
        for m in (5.0, 15.0):
            ctx.path.lat = None
            br = [x for x in run_unit(ctx, u, base_params(n), latency=m) if x["exit_ms"] <= end5]
            ctx.path.lat = lat
            exa = [x for x in run_unit(ctx, u, base_params(n), latency=m) if x["exit_ms"] <= end5]
            ctx.path.lat = None
            base = [x for x in run_unit(ctx, u, base_params(n)) if x["exit_ms"] <= end5]

            def mr(tr: list[dict]) -> float:
                if not tr:
                    return float("nan")
                df = pl.DataFrame(tr).with_columns(pl.lit("liquid").alias("tier"))
                return float(r_net(df).mean())

            row[f"lat{int(m)}"] = {"base": mr(base), "bridge": mr(br), "exact": mr(exa), "n": len(exa)}
        res[n] = row
    (OUT / "latcheck.json").write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")
    return res


def main(argv: list[str] | None = None) -> None:
    from ai_trader.core.time import now

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("design")
    r = sub.add_parser("run")
    r.add_argument("--workers", type=int, default=2)
    r.add_argument("--symbols", nargs="*")
    p = sub.add_parser("perturb")
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--symbols", nargs="*")
    s = sub.add_parser("synth")
    s.add_argument("--workers", type=int, default=2)
    s.add_argument("--paths", type=int, default=1000)
    sub.add_parser("deribit")
    sub.add_parser("latcheck")
    a = sub.add_parser("analyse")
    a.add_argument("--record", action="store_true")
    args = ap.parse_args(argv)
    OUT.mkdir(parents=True, exist_ok=True)
    if args.cmd == "design":
        dp = OUT / "design.json"
        if dp.exists():
            print("design.json exists (frozen); not overwritten")
            return
        dp.write_text(json.dumps(design_dict(now().isoformat()), indent=1, default=str), encoding="utf-8")
        print(dp)
    elif args.cmd == "run":
        cmd_run(args.symbols, args.workers, False)
    elif args.cmd == "perturb":
        cmd_run(args.symbols, args.workers, True)
    elif args.cmd == "synth":
        cmd_synth(args.workers, args.paths)
    elif args.cmd == "deribit":
        cmd_deribit()
    elif args.cmd == "latcheck":
        print(json.dumps(cmd_latcheck(), indent=1, default=float))
    elif args.cmd == "analyse":
        from ai_trader.research.stress_report import analyse

        analyse(record=args.record)


# ====================================================================== one-call API for new strategies
def stress_signal(fn: Callable[[pl.DataFrame], pl.Series], symbols: list[str], freq: str = "1h", name: str = "custom",
                  as_of: datetime | None = None, params: dict | None = None) -> dict:
    """Stress any causal signal function (frame -> Series in [-1, 1]) through the trade layer: S0, S3 cost / gap / no-stop
    variants, S4 latency, S1 segments and cells, S9 live engine. Returns {scenario: metrics} plus the engine stats."""
    from ai_trader.core.time import now

    as_of = as_of or now()
    hp = {"_theta": THETA_IN, "_stopk": STOP_K, **(params or {})}
    frames = {}
    for sym in symbols:
        ctx = load_ctx(sym, as_of, need_rb=False)
        if ctx is None:
            continue
        fr = ctx.f4 if freq == "4h" else ctx.f1
        s = fn(fr).to_numpy()
        for vname, (mode, lat) in EXEC_VARIANTS.items():
            tr = walk_signal(s, _ms(fr["close_ts"]), fr["close"].to_numpy().astype(float), fr["sigma_risk_next"].to_numpy().astype(float), ctx.path,
                             theta=float(hp["_theta"]), stop_k=float(hp["_stopk"]), stop_mode=mode, latency=lat)
            for x in tr:
                x["kind"], x["ev"] = "peg", float("nan")
            frames.setdefault(vname, []).append(_norm(annotate(tr, ctx, name, vname)))
    out: dict[str, Any] = {}
    tabs = {k: pl.concat([f for f in v if f.height], how="vertical_relaxed") if any(f.height for f in v) else pl.DataFrame() for k, v in frames.items()}
    base = tabs.get("base", pl.DataFrame())
    for k, t in tabs.items():
        out[k] = metrics(t, r_net(t)) if t.height else {"n": 0}
    if base.height:
        for k, (fm, x, dr) in COST_VARIANTS.items():
            out[k] = metrics(base, r_net(base, Costs(fee_mult=fm, extra_bps=x, drought=dr)), ci=False)
        for seg in ("llm_in_sample", "clean_holdout"):
            sub = base.filter(pl.col("seg") == seg)
            out[seg] = metrics(sub, r_net(sub)) if sub.height else {"n": 0}
        acc, info = live_engine(base, r_net(base), LiveLimits())
        out["engine"] = engine_stats(acc, info)
        out["verdict"] = verdict({"base": out["base"], "fees15": out["fees_x1.5"], "slip5": out["slip+5"], "lat5": out.get("lat5"),
                                  "holdout": out["clean_holdout"], "cliff": None, "crisis_engine_min": None})
    return out


if __name__ == "__main__":
    main()
