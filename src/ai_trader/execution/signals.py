"""Point-in-time strategy signals and their entry / exit events (agent K, 2026-11-13).

Every registered strategy (strategies.REGISTRY, 17) plus the trend-zoo ensemble runs here exactly as in the strategy
panel (research.strategy_panel): the 1h-parameterised strategies on 1h bars, geo_* and zoo_ensemble on 4h bars, on
research.frame.market_frame frames (funding on 1h, OI on 1h for crypto, flow_bars only for the flow strategies).

Entry / exit semantics are the strategy's own (R-EXIT-1): the signal s_t in [-1, 1] is turned into a held position
with evaluate.strategy_pnl's execution policy (strategy_panel.held_leverage: the held signal changes on a flip, on
flat, or when |delta s| > 0.2). Only the SIDE of the held position matters here, because size belongs to the risk
engine (CLAUDE.md principle 4). Events at bar t (decided at the close of t, traded from the next bar):

    flat -> long / short    entry
    long / short -> flat    exit (side "flat")
    long <-> short          exit, then entry on the other side (two events, same bar)

Point in time: a signal at row t uses rows <= t only (strategies.assert_causal) and the held-state recursion runs
forward, so events at or before t never change when later bars are appended (tests/test_execution_gate.py checks it).
The decision time of an event is the bar's known_ts (the moment its close is in the lake), never earlier.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
import polars as pl

from ai_trader.research.playbook import TRADFI_SYMS
from ai_trader.research.strategy_panel import BAND, UNIV_T, held_leverage

FREQ_4H = ("geo_struct", "geo_pivot_bo", "geo_flag_bo", "zoo_ensemble")
# registry family -> contract family (docs/EXECUTION_PIPELINE.md enum; 'flow' added by a dated Changes note)
CONTRACT_FAMILY = {"trend": "trend", "mean_reversion": "mean_reversion", "volatility": "breakout", "positioning": "positioning",
                   "flow": "flow", "calendar": "calendar", "shock": "shock"}
# R-PD-1 / R-PD-2: the trend family (incl. compression_breakout and the zoo) is ONE crypto TSMOM bet -> one cluster
TREND_CLUSTER = ("tsmom", "ema_stack", "donchian_breakout", "geo_struct", "geo_pivot_bo", "geo_flag_bo", "compression_breakout", "zoo_ensemble")


@dataclass(frozen=True)
class StrategyRef:
    name: str
    family: str            # registry family (strategies.base.FAMILIES); 'trend' for zoo_ensemble
    freq: str              # '1h' | '4h'
    requires: tuple[str, ...] = ("bars",)

    @property
    def contract_family(self) -> str:
        """The intent's `family`, which the risk engine uses as the cluster (R-PD-1): every trend-cluster strategy
        (R-PD-2, incl. compression_breakout) is 'trend', so the cluster cap treats them as one bet."""
        return "trend" if self.name in TREND_CLUSTER else CONTRACT_FAMILY[self.family]

    @property
    def cluster(self) -> str:
        return self.contract_family

    @property
    def bars_per_day(self) -> int:
        return 6 if self.freq == "4h" else 24


def strategy_refs(names: Iterable[str] | None = None) -> list[StrategyRef]:
    """All runnable strategies (registry order + zoo_ensemble), or the named ones (KeyError on an unknown name)."""
    from ai_trader.strategies import REGISTRY

    allr = [StrategyRef(n, c.spec.family, "4h" if n in FREQ_4H else "1h", tuple(c.spec.requires)) for n, c in REGISTRY.items()]
    allr.append(StrategyRef("zoo_ensemble", "trend", "4h", ("bars",)))
    if names is None:
        return allr
    by = {r.name: r for r in allr}
    out = []
    for n in names:
        if n not in by:
            raise KeyError(f"unknown strategy {n!r}; one of {sorted(by)}")
        out.append(by[n])
    return out


def strategy_callable(name: str) -> Callable[[pl.DataFrame], pl.Series]:
    from ai_trader.research.strategy_panel import strategy_fn

    return strategy_fn(name)


def asset_class_of(symbol: str) -> str:
    return "tradfi" if symbol in TRADFI_SYMS or symbol in UNIV_T else "crypto"


# ====================================================================== frames

@dataclass
class SymbolFrames:
    symbol: str
    asset_class: str
    as_of: datetime
    f1: pl.DataFrame                    # 1h market frame (what the 1h strategies see)
    f4: pl.DataFrame | None = None      # 4h frame (native or aggregated from 1h, as the panel)
    f1_flow: pl.DataFrame | None = None  # 1h frame with flow_bars columns (flow strategies only)
    f4_source: str = "native"
    notes: list[str] = field(default_factory=list)

    def frame_for(self, ref: StrategyRef) -> pl.DataFrame | None:
        if ref.freq == "4h":
            return self.f4
        if "flow" in ref.requires:
            return self.f1_flow
        return self.f1

    def truncated(self, ts: datetime) -> SymbolFrames:
        """The frames as they were known at `ts` (rows with known_ts <= ts): for point-in-time checks."""
        cut = lambda f: None if f is None else f.filter(pl.col("known_ts") <= ts)
        return SymbolFrames(self.symbol, self.asset_class, ts, cut(self.f1), cut(self.f4), cut(self.f1_flow), self.f4_source, list(self.notes))


def load_frames(symbol: str, as_of: datetime, refs: Iterable[StrategyRef]) -> SymbolFrames:
    """Point-in-time frames for one symbol: every lake read is as of `as_of` (Lake.read_asof)."""
    from ai_trader.research.frame import market_frame
    from ai_trader.research.strategy_panel import frames_for

    refs = list(refs)
    ac = asset_class_of(symbol)
    f1, f4, src = frames_for(symbol, "T" if ac == "tradfi" else "L", as_of)
    out = SymbolFrames(symbol, ac, as_of, f1, f4 if any(r.freq == "4h" for r in refs) else None, f4_source=src)
    if any("flow" in r.requires for r in refs):
        ff = market_frame(symbol, "1h", as_of, funding=True, oi=(ac == "crypto"), flow=True)
        if "buy_volume" in ff.columns and ff["buy_volume"].drop_nulls().len() > 0:
            out.f1_flow = ff
        else:
            out.notes.append("no flow_bars for this symbol: flow strategies skipped")
    return out


# ====================================================================== events

@dataclass(frozen=True)
class SignalEvent:
    strategy: str
    family: str             # contract family
    cluster: str
    freq: str
    symbol: str
    asset_class: str
    kind: str               # 'entry' | 'exit'
    side: str               # 'long' | 'short' | 'flat'
    prev_side: str
    signal: float           # raw strategy signal at the bar
    held_signal: float      # held signal after the execution band
    bar_event_ts: datetime  # open time of the decision bar
    bar_close_ts: datetime
    decision_ts: datetime   # known_ts of the decision bar (= as_of of the decision)
    price: float            # close of the decision bar (reference entry price)
    row: int                # row index in the strategy's frame


def _side(x: float) -> str:
    return "long" if x > 0 else ("short" if x < 0 else "flat")


def held_sides(signal: np.ndarray, sigma: np.ndarray, band: float = BAND) -> tuple[np.ndarray, np.ndarray]:
    """(held_signal, side in {-1, 0, +1}) per bar under the panel's execution policy."""
    hs, lev = held_leverage(signal, sigma, band)
    return hs, np.sign(lev).astype(int)


def strategy_events(ref: StrategyRef, sf: SymbolFrames, start: datetime | None = None, end: datetime | None = None) -> tuple[list[SignalEvent], dict]:
    """Entry / exit events of one strategy on one symbol with decision_ts in [start, end]. Also returns the strategy's
    state at the last bar (for paper mode: the position the strategy holds now)."""
    f = sf.frame_for(ref)
    info = {"strategy": ref.name, "symbol": sf.symbol, "freq": ref.freq}
    if f is None or f.height < 50:
        info["skip"] = "no frame" if f is None else f"only {f.height} bars"
        return [], info
    sig = strategy_callable(ref.name)(f).to_numpy().astype(float)
    hs, side = held_sides(sig, f["sigma_next"].to_numpy().astype(float))
    prev = np.r_[0, side[:-1]]
    dts = f["known_ts"].to_list()
    ets = f["event_ts"].to_list()
    cts = f["close_ts"].to_list()
    close = f["close"].to_numpy()
    idx = np.flatnonzero(side != prev)
    out: list[SignalEvent] = []
    for i in idx:
        d = dts[i]
        if (start is not None and d < start) or (end is not None and d > end):
            continue
        base = {"strategy": ref.name, "family": ref.contract_family, "cluster": ref.cluster, "freq": ref.freq, "symbol": sf.symbol, "asset_class": sf.asset_class,
                    "signal": float(sig[i]), "held_signal": float(hs[i]), "bar_event_ts": ets[i], "bar_close_ts": cts[i], "decision_ts": d, "price": float(close[i]), "row": int(i)}
        p, s = int(prev[i]), int(side[i])
        if p != 0:
            out.append(SignalEvent(kind="exit", side="flat", prev_side=_side(p), **base))
        if s != 0:
            out.append(SignalEvent(kind="entry", side=_side(s), prev_side="flat" if p != 0 else _side(p), **base))
    info.update({"last_bar_close": cts[-1], "last_decision_ts": dts[-1], "held_side": _side(int(side[-1])), "held_signal": float(hs[-1]),
                 "signal": float(sig[-1]), "n_events": len(out)})
    return out, info
