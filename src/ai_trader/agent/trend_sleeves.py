"""Paper trend sleeves (WS16 spec A and spec B) and the H-WS16-15 kill monitor. This module is the SLEEVE CONFIG of the
paper decision harness (agent.session builds the candidates from it, agent.decision executes its rules).

User decisions 2026-11-27 ~21:35 UTC (research/reports/synthesis_2026-11-27.md D4 / D5; specs in
research/reports/ws16_2026-11-27.md "Recommended specs"):

  D4  ACTIVE (paper book only)
      trend_zoo_ew   spec A, primary    s = trend_zoo.ensemble(4h frame): equal-weight mean of the 21 trend_zoo variants
      trend_tsmom30  spec B, benchmark  s = tanh(z), z = ln(c_t / c_{t-180}) / (sigma_next_t sqrt 180) (trend_zoo "tsmom_30d")
      trade layer, both (bars: Bybit linear 4h closes 00/04/08/12/16/20 UTC):
        entry   flat in the sleeve and |s| >= 0.30 at the 4h close, side = sign(s); limit pegged to the touch for 15 min;
                unfilled -> stay flat and decide again at the next close (paper book: at-exec fills a pegged entry at the
                next 1h open + 1.3 bp, H-EXE1; the 15-minute fill is measured on the stream below)
        stop    exchange stop 4.0 x sigma_risk_next (per 4h bar) from the entry, i.e. the applied H-STOP1 cap, kept 1% inside
                it for tick rounding (3.96 at a 4.0 cap); no trailing
        exit    a taker at the first 4h close with s x side <= 0 (a deterministic rule exit sent by the harness at prepare,
                never Claude's choice); no target, no time stop, no pyramiding
        re-entry after a stop: none until s x side <= 0 at some 4h close after the stop (the signal is flat or flips)
        size    the risk engine's (fixed risk per trade); the engine checks net_er_r = PRIOR_NET_R (WS16 forward
                expectation, 0 to +0.2 R, central +0.1, net of costs); p_tp = REF_P_TP is the driftless P(+3R before
                the stop), a reference only (no resting target), as execution.gate does for own-exit strategies
        gates   the bias gate as for every sleeve (R-PX-9: no new entries in the TradFi weekend window, as the old trend_4h
                sleeve; exits run on weekends) and the engine limits (risk.yaml; max_positions_per_cluster 1 means ONE
                trend position in the paper book across both sleeves and the legacy trend_4h)
  LEGACY
      trend_4h       4h ema_stack bracket (1.5 x HAR stop, 3R target): no new candidates since D4; open positions (the
                     BTC short from S20261125-0009) run to completion under their bracket
  RETIRED (D5)
      shock_regime / shock_desk  the shock sleeve (H-S2 fade leg and its continuation leg): no paper candidates
      setup_c                    the spot-absorption reversal long of research/desk/PLAN_2026-11-15.md: no desk proposals

H-WS16-15 monitor (registered: "on the unconstrained paper stream, spec A's mean net R per closed trade (entries after
2026-11-27 08:00 UTC) is > 0"). The paper book holds one trend position at a time, so the statistical sample is the
UNCONSTRAINED stream: the same trade layer walked deterministically over universe L (23 perps), every signal, no slots,
no weekend block (the tested base spec), entries counted after FORWARD_START, closed trades only, net R with the live
costs (pegged entry 2.0 + 1.3 bp, taker exit 5.5 + 2.0 bp; stress.r_net), gap-aware stop fills (the paper book's rule).
The walk starts flat at WALK_START (FORWARD_START - 365 days; signals use the full 4h history), so its state at
FORWARD_START is the backtest's in practice (p90 holding 32 days); positions open at FORWARD_START do not count.
  peg      a forward entry fills if a trade prints 1 bp through the decision close within 15 minutes (WS16's 5m model) on
           flow_3s; where the collector did not cover the window the entry is taken as filled (WS16 base, 92% history)
           and counted "unchecked". Results are cached (point in time: an entry is used only when known at as_of).
  kill     mean < -0.30 R over the first 100 closed trades; or the one-sided 95% upper bound of the mean < 0 at >= 300
           closed trades (stationary bootstrap of entry days, mean block max(5, p90 holding days), 2000 draws), evaluated
           at the registered checkpoints 300 / 1000 and at the current count (latched in monitor.json by real sessions).
           A killed sleeve gets no new paper candidates; its open positions run to their rule exits.
  alert    peg fill rate < 70% once >= 20 forward entries were checked.
The same rules are applied to spec B (the benchmark) on its own stream.
"""

from __future__ import annotations

import functools
import json
import math
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from ai_trader.core.time import utc

VERSION = "trend_sleeves.v1"
FORWARD_START = datetime(2026, 11, 27, 8, 0, tzinfo=UTC)   # H-WS16-15: entries strictly after this time count
WALK_START = FORWARD_START - timedelta(days=365)           # the stream walks from here (state burn-in; fixed, so stable)
BARS_PER_DAY = 6
THETA = 0.30
EXIT_THR = 0.0
STOP_K = 4.0
TICK_MARGIN = 0.01                 # as agent.session.cap_stop: 1% inside the cap (the engine rounds the stop away from entry)
PEG_WINDOW_MIN = 15.0
PEG_THROUGH_BPS = 1.0
PEG_MIN_COVERAGE = 0.8             # share of the window's 3 s candles flow_3s must hold for a checked fill
PEG_SETTLE = timedelta(minutes=2)    # a window is checked only once its last candles are known
PEG_FINAL_AFTER = timedelta(minutes=30)
PRIOR_NET_R = 0.10                 # WS16 forward expectation per closed trade (net), sent as net_er_r
REF_P_TP = 0.25                    # driftless P(+3R before -1R) in log space: b / (a + b) with a = 3b
UNIVERSE_L = ("BTCUSDT ETHUSDT SOLUSDT XRPUSDT DOGEUSDT ADAUSDT LINKUSDT LTCUSDT BCHUSDT AVAXUSDT DOTUSDT BNBUSDT "
              "TRXUSDT NEARUSDT ATOMUSDT ETCUSDT FILUSDT UNIUSDT AAVEUSDT XLMUSDT SUIUSDT ENAUSDT HYPEUSDT").split()
KILL_MEAN_N, KILL_MEAN_R = 100, -0.30
KILL_UB_MIN_N = 300
CHECKPOINTS = (100, 300, 1000)
PEG_ALERT_RATE, PEG_ALERT_MIN = 0.70, 20
N_BOOT, SEED, MIN_BLOCK_DAYS = 2000, 20261127, 5.0
DAY_MS = 86_400_000


@dataclass(frozen=True)
class SleeveSpec:
    name: str          # paper sleeve = book strategy name
    spec: str          # WS16 spec letter
    signal: str        # key of compute_signals()
    prefix: str        # candidate id prefix
    role: str
    hypothesis: str


SLEEVES: dict[str, SleeveSpec] = {
    "trend_zoo_ew": SleeveSpec("trend_zoo_ew", "A", "zoo_ew", "zoo", "primary", "H-WS16-15"),
    "trend_tsmom30": SleeveSpec("trend_tsmom30", "B", "tsmom_30d", "tsm", "benchmark", "H-WS16-17 (benchmark; H-WS16-15 rules by analogy)"),
}
LEGACY = {"trend_4h": "D4 2026-11-27: replaced by trend_zoo_ew; open positions run to completion under their bracket (no new candidates)"}
RETIRED = {
    "shock_regime": "D5 2026-11-27: shock fade (H-S2) retired with its continuation leg (one sleeve); no paper candidates",
    "shock_desk": "D5 2026-11-27: the desk shock setup (stress unit shock_desk) retired with shock_regime",
    "setup_c": "D5 2026-11-27: setup C (spot-absorption reversal long, research/desk/PLAN_2026-11-15.md) retired; no desk proposals",
}


# ====================================================================== signals (4h, causal)

def lookback_bars() -> int:
    from ai_trader.strategies.trend_zoo import lookback_bars as lb

    return lb(BARS_PER_DAY)


def compute_signals(f4: pl.DataFrame) -> dict[str, np.ndarray]:
    """Both sleeve signals on a 4h market frame (row t uses rows <= t only)."""
    from ai_trader.strategies.trend_zoo import ZOO, ensemble

    return {"zoo_ew": ensemble(f4, BARS_PER_DAY).to_numpy().astype(float), "tsmom_30d": ZOO["tsmom_30d"](f4).to_numpy().astype(float)}


def stop_sigma_4h(f4: pl.DataFrame) -> np.ndarray:
    """Per-4h-bar risk sigma for the stop (the engine's placement unit): the frame's sigma_risk_next, which
    research.frame.add_robust_vol takes from ai_trader.risk.sigma.stop_sigma_column, i.e. risk.sigma's sigma_risk under
    the spec in force at as_of (D1: robust_bc until the shadowed switch to the clock log-HAR, never weekend-shrunk).
    The signals keep sigma_next (the EWMA scale the WS16 specs were tested with)."""
    return f4["sigma_risk_next"].cast(pl.Float64).fill_null(float("nan")).to_numpy().astype(float)


@dataclass
class SymbolSignals:
    symbol: str
    as_of: datetime
    close_ms: np.ndarray                 # 4h bar close times (epoch ms)
    close: np.ndarray
    sigma4: np.ndarray                   # stop sigma per 4h bar
    s: dict[str, np.ndarray]             # signal name -> value at each close
    warm: bool                           # enough history for every zoo variant
    data_max_ts: str | None = None       # newest known_ts of the 4h frame (look-ahead check)

    def __len__(self) -> int:
        return len(self.close_ms)

    def last(self, name: str) -> float:
        return float(self.s[name][-1]) if len(self) and self.warm else 0.0

    def last_close(self) -> datetime | None:
        return datetime.fromtimestamp(int(self.close_ms[-1]) / 1000, UTC) if len(self) else None


def signals_from_frame(sym: str, f4: pl.DataFrame, as_of: datetime) -> SymbolSignals:
    if f4.height == 0:
        return SymbolSignals(sym, as_of, np.zeros(0, np.int64), np.zeros(0), np.zeros(0), {"zoo_ew": np.zeros(0), "tsmom_30d": np.zeros(0)}, False)
    s = compute_signals(f4)
    return SymbolSignals(sym, as_of, f4["close_ts"].dt.epoch("ms").to_numpy().astype(np.int64), f4["close"].to_numpy().astype(float), stop_sigma_4h(f4), s,
                         f4.height >= lookback_bars(), str(f4["known_ts"].max()))


@functools.lru_cache(maxsize=1)
def _lake():
    from ai_trader.storage.lake import Lake

    return Lake()


class _Prefetched:
    """Read-only stand-in for Lake.bars over one multi-symbol read (one DuckDB scan of the bars table instead of one per
    symbol); market_frame(..., funding=False, oi=False, lake=_Prefetched(...)) makes no other lake call."""

    def __init__(self, df: pl.DataFrame):
        self.by = {k[0]: g for k, g in df.group_by("symbol")} if df.height else {}

    def bars(self, symbol: str, freq: str, as_of: datetime, *, price_type: str = "last", start: datetime | None = None, venue: str = "bybit") -> pl.DataFrame:
        g = self.by.get(symbol)
        if g is None:
            return pl.DataFrame(schema={**{c: pl.Datetime("ms", "UTC") for c in ("event_ts", "close_ts", "known_ts")},
                                        **{c: pl.Float64 for c in ("open", "high", "low", "close", "volume", "turnover")}})
        return g.filter(pl.col("event_ts") >= start) if start is not None else g


_SIG: dict[tuple[str, str], SymbolSignals] = {}
_PATH: dict[tuple[str, str], dict[str, np.ndarray]] = {}


def _bounded(d: dict, cap: int = 256) -> None:
    if len(d) > cap:
        d.clear()


def symbol_signals(sym: str, as_of: datetime) -> SymbolSignals:
    """Point-in-time 4h signals of both sleeves for one symbol (market_frame as of `as_of`; cached per process)."""
    from ai_trader.research.frame import market_frame

    key = (sym, utc(as_of).isoformat())
    if key not in _SIG:
        _bounded(_SIG)
        t = utc(as_of)
        _SIG[key] = signals_from_frame(sym, market_frame(sym, "4h", t, funding=False, oi=False, lake=_lake()), t)
    return _SIG[key]


def _path_from_bars(b: pl.DataFrame) -> dict[str, np.ndarray]:
    b = b.select("event_ts", "open", "high", "low", "close").sort("event_ts")
    # the market_frame price sanity filter (impossible bars exist in venue history)
    b = b.filter((pl.col("open") > 0) & (pl.col("low") > 0) & (pl.col("high") >= pl.max_horizontal("open", "close")) & (pl.col("low") <= pl.min_horizontal("open", "close")))
    return {"ts": b["event_ts"].dt.epoch("ms").to_numpy().astype(np.int64), "o": b["open"].to_numpy().astype(float), "h": b["high"].to_numpy().astype(float),
            "l": b["low"].to_numpy().astype(float), "c": b["close"].to_numpy().astype(float)}


def prefetch(symbols: list[str] | tuple[str, ...], as_of: datetime) -> None:
    """Fill the signal and 1h-path caches for many symbols with two lake reads (point in time, as of `as_of`)."""
    from ai_trader.research.frame import market_frame

    t = utc(as_of)
    iso = t.isoformat()
    need4 = [x for x in symbols if (x, iso) not in _SIG]
    need1 = [x for x in symbols if (x, iso) not in _PATH]
    lk = _lake()
    if need4:
        pf = _Prefetched(lk.read_asof("bars", t, symbols=need4, filters={"venue": "bybit", "price_type": "last", "freq": "4h"}))
        _bounded(_SIG)
        for x in need4:
            _SIG[(x, iso)] = signals_from_frame(x, market_frame(x, "4h", t, funding=False, oi=False, lake=pf), t)
    if need1:
        b = lk.read_asof("bars", t, symbols=need1, start=WALK_START, filters={"venue": "bybit", "price_type": "last", "freq": "1h"})
        by = {k[0]: g for k, g in b.group_by("symbol")} if b.height else {}
        _bounded(_PATH)
        for x in need1:
            _PATH[(x, iso)] = _path_from_bars(by[x]) if x in by else {k: np.zeros(0) for k in ("ts", "o", "h", "l", "c")}


def stop_k(cap: float) -> float:
    """Stop distance in per-4h-bar sigmas: STOP_K, never beyond the applied cap (kept TICK_MARGIN inside it)."""
    return min(STOP_K, float(cap) * (1 - TICK_MARGIN))


# ====================================================================== paper-book rules (entries, re-entry, exits)

def _side(x: Any) -> int:
    return 1 if str(x).lower() == "long" else -1


def reentry_block(sleeve: str, symbol: str, side: str, trades: list[dict], sg: SymbolSignals) -> str | None:
    """No re-entry after a stop until s x side <= 0 at a 4h close after it. `trades`: the paper book's closed trades."""
    spec = SLEEVES[sleeve]
    mine = [t for t in trades if t.get("strategy") == sleeve and t.get("symbol") == symbol]
    if not mine:
        return None
    last = max(mine, key=lambda t: t.get("closed_ts", 0))
    if last.get("exit_reason") != "stop":
        return None
    d = _side(last["side"])
    if _side(side) != d:
        return None                       # a flip re-arms at once
    after = sg.close_ms > int(last["closed_ts"])
    if np.any(sg.s[spec.signal][after] * d <= EXIT_THR):
        return None
    return (f"no re-entry after the {str(last['side']).lower()} stop at {datetime.fromtimestamp(last['closed_ts'] / 1000, UTC):%m-%d %H:%M}Z "
            f"until s x side <= 0 at a 4h close (WS16 trade layer)")


def exits_due(positions: list[dict], as_of: datetime, signals_fn, errors: list[str] | None = None) -> list[dict]:
    """Open sleeve positions whose signal reached s x side <= 0 at a 4h close after the position opened (the first such
    close; `late` when it is not the latest close, e.g. a missed session). Already-requested exits are skipped. A symbol
    whose signals cannot be read is reported in `errors` and checked again next session (the late rule catches it up)."""
    out = []
    for p in positions:
        spec = SLEEVES.get(p.get("strategy"))
        if spec is None or p.get("exit_requested"):
            continue
        try:
            sg = signals_fn(p["symbol"], as_of)
        except Exception as e:
            if errors is None:
                raise
            errors.append(f"{p['symbol']}: {type(e).__name__}: {str(e)[:160]}")
            continue
        d = _side(p["side"])
        idx = np.nonzero((sg.close_ms > int(p["opened_ts"])) & (sg.s[spec.signal] * d <= EXIT_THR))[0]
        if len(idx) == 0:
            continue
        i = int(idx[0])
        out.append({"position_id": p["id"], "symbol": p["symbol"], "strategy": p["strategy"], "family": p.get("family", "trend"), "side": str(p["side"]).lower(),
                    "close_ts": datetime.fromtimestamp(int(sg.close_ms[i]) / 1000, UTC).isoformat(), "s": float(sg.s[spec.signal][i]),
                    "s_now": sg.last(spec.signal), "late": i != len(sg) - 1, "mark": p.get("mark") or p.get("entry_price")})
    return out


# ====================================================================== the unconstrained stream (H-WS16-15 sample)

def _costs() -> dict[str, float]:
    from ai_trader.core.config import load

    r = load().risk
    k = r["costs"]
    return {"maker": float(k["maker_bps"]), "taker": float(k["taker_bps"]), "slip": float(k["slippage_bps"]), "peg_slip": float(r["paper_entry"]["slip_bps"])}


def net_r(side: int, entry: float, stop: float, exit_: float, kind: str = "peg", costs: dict | None = None) -> float:
    """Net R exactly as stress.r_net (base cost scenario): entry leg maker + peg slip (or taker + slip), exit taker + slip,
    denominator = the at-risk sizing budget |entry - stop| + entry x round-trip cost."""
    c = costs or _costs()
    c_in = (c["maker"] + c["peg_slip"] if kind == "peg" else c["taker"] + c["slip"]) / 1e4
    c_out = (c["taker"] + c["slip"]) / 1e4
    return (side * (exit_ - entry) - c_in * entry - c_out * exit_) / (abs(entry - stop) + entry * (c_in + c_out))


def walk(close_ms: np.ndarray, close: np.ndarray, s: np.ndarray, sigma: np.ndarray, path: dict[str, np.ndarray], *, theta: float = THETA,
         stop_k_: float = STOP_K, exit_thr: float = EXIT_THR, start_ms: int | None = None, peg=None, gap_fills: bool = True) -> dict[str, Any]:
    """The WS16 trade layer (stress.walk_signal semantics) on one symbol, every 4h close processed (also the latest).

    At each close t: exit when s_t x side <= exit_thr (at the close); re-arm after a stop once s_t x stopped_side <= 0;
    enter when flat, armed and |s_t| >= theta at the close price, stop = close x exp(-side stop_k sigma_t); then the stop
    is checked on the 1h bars opening in [close_t, close_{t+1}) (gap through the stop -> the open, else the stop price).
    Entries at closes after start_ms ask `peg(close_ms, side, px)`: False = unfilled (stay flat, decide again next close),
    True = filled, None = not checkable (taken as filled). Returns closed trades, forward peg attempts and the open trade."""
    ts, o, h, lo = (np.asarray(path[k]) for k in ("ts", "o", "h", "l"))
    n = len(close_ms)
    a = np.r_[np.searchsorted(ts, close_ms, side="left"), len(ts)]
    s = np.nan_to_num(np.asarray(s, dtype=float))
    trades: list[dict] = []
    attempts: list[dict] = []
    pos, entry, stop, ets, ebar, s_e, pegs = 0, 0.0, 0.0, 0, 0, 0.0, "history"
    armed, stopped_side = True, 0

    def rec(x: float, xms: int, why: str, gap: bool = False) -> None:
        trades.append({"side": pos, "entry_ms": ets, "exit_ms": xms, "entry": entry, "stop": stop, "exit": x, "why": why, "gap": gap,
                       "s_entry": s_e, "peg": pegs, "forward": start_ms is not None and ets > start_ms})

    for t in range(n):
        st = s[t]
        if pos != 0 and st * pos <= exit_thr:
            rec(float(close[t]), int(close_ms[t]), "signal")
            pos = 0
        if not armed and st * stopped_side <= 0:
            armed = True
        if pos == 0 and armed and abs(st) >= theta and np.isfinite(sigma[t]) and sigma[t] > 0:
            side, px = (1 if st > 0 else -1), float(close[t])
            fwd = start_ms is not None and int(close_ms[t]) > start_ms
            filled = peg(int(close_ms[t]), side, px) if (fwd and peg is not None) else None
            if fwd:
                attempts.append({"close_ms": int(close_ms[t]), "side": side, "px": px,
                                 "result": {True: "filled", False: "unfilled", None: "unchecked"}[filled]})
            if filled is not False:
                pos, entry, ets, ebar, s_e = side, px, int(close_ms[t]), int(a[t]), float(abs(st))
                stop = px * math.exp(-pos * stop_k_ * float(sigma[t]))
                pegs = ("filled" if filled else "unchecked") if fwd else "history"
        if pos != 0:
            for i in range(max(int(a[t]), ebar), int(a[t + 1])):
                if (pos > 0 and lo[i] <= stop) or (pos < 0 and h[i] >= stop):
                    gap = (pos > 0 and o[i] < stop) or (pos < 0 and o[i] > stop)
                    rec(float(o[i]) if (gap and gap_fills) else float(stop), int(ts[i] + 3_600_000), "stop", gap)
                    armed, stopped_side, pos = False, pos, 0
                    break
    open_ = None
    if pos != 0:
        open_ = {"side": pos, "entry_ms": ets, "entry": entry, "stop": stop, "s_entry": s_e, "peg": pegs,
                 "forward": start_ms is not None and ets > start_ms, "mark": float(close[-1]) if n else entry}
    return {"trades": trades, "attempts": attempts, "open": open_}


class PegChecker:
    """15-minute pegged-entry fill check on flow_3s (WS16 model: a trade 1 bp through the decision close). Definitive
    results are cached in `path` with the time they became known; a cached result is used only when known at as_of."""

    def __init__(self, as_of: datetime, path: Path | None = None, *, lake=None, write: bool = False):
        self.as_of, self.path, self.write, self._lake = utc(as_of), path, write, lake
        self.cache: dict[str, dict] = {}
        if path is not None and path.is_file():
            try:
                self.cache = json.loads(path.read_text(encoding="utf-8")).get("checks", {})
            except (OSError, ValueError):
                self.cache = {}
        self.dirty = False

    @property
    def lake(self):
        if self._lake is None:
            from ai_trader.storage.lake import Lake

            self._lake = Lake()
        return self._lake

    def check(self, sym: str, close_ms: int, side: int, px: float) -> bool | None:
        key = f"{sym}|{close_ms}|{side}|{px:.10g}"
        c = self.cache.get(key)
        if c is not None and utc(c["known_at"]) <= self.as_of:
            return {"filled": True, "unfilled": False}.get(c["result"])
        t0 = datetime.fromtimestamp(close_ms / 1000, UTC)
        t1 = t0 + timedelta(minutes=PEG_WINDOW_MIN)
        if self.as_of < t1 + PEG_SETTLE:
            return None                                       # window not complete (or its last candles not yet known) at as_of
        d = self._window(sym, t0, t1)
        cov = d["event_ts"].n_unique() / (PEG_WINDOW_MIN * 20) if d.height else 0.0
        result = None
        if cov >= PEG_MIN_COVERAGE:
            if side > 0:
                lo = d["low"].drop_nulls()
                result = "filled" if lo.len() and float(lo.min()) <= px * (1 - PEG_THROUGH_BPS / 1e4) else "unfilled"
            else:
                hi = d["high"].drop_nulls()
                result = "filled" if hi.len() and float(hi.max()) >= px * (1 + PEG_THROUGH_BPS / 1e4) else "unfilled"
            known = t1 + PEG_SETTLE
        elif self.as_of >= t1 + PEG_FINAL_AFTER:
            result, known = "no_data", t1 + PEG_FINAL_AFTER
        if result is not None:
            self.cache[key] = {"result": result, "known_at": known.isoformat(), "coverage": round(cov, 3)}
            self.dirty = True
        return {"filled": True, "unfilled": False}.get(result or "")

    def _window(self, sym: str, t0: datetime, t1: datetime) -> pl.DataFrame:
        """flow_3s candles of sym with event_ts in [t0, t1) known at as_of. A real lake is read from the one date
        partition (the 4h closes never cross midnight), filtered on known_ts <= as_of like Lake.read_asof (duplicate
        revisions are harmless here: extremes, and coverage counts distinct candles); any other lake via read_asof."""
        if self._lake is None:
            self._lake = _lake()
        root = getattr(self.lake, "root", None)
        part = Path(root) / "flow_3s" / "venue=bybit" / "category=linear" / f"date={t0:%Y-%m-%d}" if root is not None else None
        if part is None or not hasattr(self.lake, "tables"):
            return self.lake.read_asof("flow_3s", self.as_of, symbols=[sym], start=t0, end=t1,
                                       filters={"venue": "bybit", "category": "linear", "date": t0.strftime("%Y-%m-%d")}, columns=["event_ts", "high", "low"])
        if not part.is_dir() or not any(part.glob("*.parquet")):
            return pl.DataFrame({"event_ts": [], "high": [], "low": []})
        q = (pl.scan_parquet(str(part / "*.parquet"), hive_partitioning=False)
             .filter((pl.col("symbol") == sym) & (pl.col("event_ts") >= t0) & (pl.col("event_ts") < t1) & (pl.col("known_ts") <= self.as_of))
             .select("event_ts", "high", "low"))
        return q.collect()

    def save(self) -> None:
        if not (self.write and self.dirty and self.path is not None):
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps({"version": VERSION, "rule": f"trade {PEG_THROUGH_BPS:g} bp through the decision close within {PEG_WINDOW_MIN:g} min (flow_3s)",
                                   "checks": self.cache}, indent=0), encoding="utf-8")
        tmp.replace(self.path)


def path_1h(sym: str, as_of: datetime) -> dict[str, np.ndarray]:
    """1h bars of `sym` from WALK_START known at as_of (the stream's stop path; cached per process)."""
    key = (sym, utc(as_of).isoformat())
    if key not in _PATH:
        _bounded(_PATH)
        _PATH[key] = _path_from_bars(_lake().bars(sym, "1h", utc(as_of), start=WALK_START))
    return _PATH[key]


def stream(as_of: datetime, *, symbols: list[str] | tuple[str, ...] = tuple(UNIVERSE_L), signals_fn=symbol_signals, path_fn=path_1h,
           peg: PegChecker | None = None, start: datetime = FORWARD_START, walk_start: datetime | None = WALK_START) -> dict[str, dict]:
    """The unconstrained stream of both sleeves as of `as_of`: per sleeve the closed forward trades (net R), the open
    forward positions, the forward peg attempts and per-symbol errors. The walk starts flat at `walk_start` (None: the
    first close)."""
    costs = _costs()
    start_ms = int(start.timestamp() * 1000)
    walk_ms = int(walk_start.timestamp() * 1000) if walk_start is not None else -(2**62)
    out: dict[str, dict] = {k: {"closed": [], "open": [], "carried": 0, "attempts": [], "errors": {}} for k in SLEEVES}
    if signals_fn is symbol_signals and path_fn is path_1h:
        try:
            prefetch(symbols, as_of)
        except Exception:  # fall back to per-symbol reads (errors are then reported per symbol)
            pass
    for sym in symbols:
        try:
            sg = signals_fn(sym, as_of)
            if not sg.warm:
                raise ValueError(f"only {len(sg)} 4h bars (< {lookback_bars()} for the zoo)")
            p = path_fn(sym, as_of)
        except Exception as e:  # one symbol must not stop the monitor
            for k in SLEEVES:
                out[k]["errors"][sym] = str(e)[:200]
            continue
        for name, spec in SLEEVES.items():
            pf = (lambda ms, side, px, _s=sym: peg.check(_s, ms, side, px)) if peg is not None else None
            k0 = int(np.searchsorted(sg.close_ms, walk_ms, side="left"))
            w = walk(sg.close_ms[k0:], sg.close[k0:], sg.s[spec.signal][k0:], sg.sigma4[k0:], p, start_ms=start_ms, peg=pf)
            for tr in w["trades"]:
                if tr["forward"]:
                    out[name]["closed"].append({**tr, "symbol": sym, "R": net_r(tr["side"], tr["entry"], tr["stop"], tr["exit"], "peg", costs),
                                                "hold_days": (tr["exit_ms"] - tr["entry_ms"]) / DAY_MS})
            if w["open"] and w["open"]["forward"]:
                op = w["open"]
                out[name]["open"].append({**op, "symbol": sym, "R_mark": net_r(op["side"], op["entry"], op["stop"], op["mark"], "peg", costs)})
            elif w["open"]:
                out[name]["carried"] += 1           # opened before FORWARD_START: runs on, never counted
            out[name]["attempts"] += [{**x, "symbol": sym} for x in w["attempts"]]
    for v in out.values():
        v["closed"].sort(key=lambda x: (x["exit_ms"], x["entry_ms"], x["symbol"]))
    return out


# ====================================================================== kill rules (pure)

def upper_bound(trades: list[dict], q: float = 0.95, n_boot: int = N_BOOT, seed: int = SEED) -> dict:
    """One-sided q upper bound of the mean net R per trade: stationary bootstrap of entry days (every day of the span,
    empty days included), mean block max(MIN_BLOCK_DAYS, p90 holding days); statistic sum R / n per draw."""
    from ai_trader.sim.resample import stationary_indices

    r = np.array([t["R"] for t in trades], dtype=float)
    if len(r) < 3:
        return {"ub": float("nan"), "block_days": float("nan"), "n_days": 0}
    day = np.array([t["entry_ms"] // DAY_MS for t in trades], dtype=np.int64)
    d0 = int(day.min())
    nd = int(day.max()) - d0 + 1
    n = np.bincount(day - d0, minlength=nd).astype(float)
    sr = np.bincount(day - d0, weights=r, minlength=nd)
    hold = np.array([t.get("hold_days", (t["exit_ms"] - t["entry_ms"]) / DAY_MS) for t in trades], dtype=float)
    block = max(MIN_BLOCK_DAYS, float(np.quantile(hold, 0.9)))
    idx = stationary_indices(nd, nd, block, n_boot, np.random.default_rng(seed))
    ns, ss = n[idx].sum(axis=1), sr[idx].sum(axis=1)
    m = ss[ns > 0] / ns[ns > 0]
    return {"ub": float(np.quantile(m, q)) if len(m) else float("nan"), "block_days": block, "n_days": nd}


def kill_rules(closed: list[dict]) -> dict:
    """H-WS16-15 rules on closed trades ordered by exit time. Stateless: the first-100 mean, the upper bound at the
    registered checkpoints 300 / 1000 and at the current count."""
    n = len(closed)
    r = np.array([t["R"] for t in closed], dtype=float)
    out: dict[str, Any] = {"n": n, "mean": float(r.mean()) if n else float("nan"), "killed": False, "rule": None, "at_n": None, "checks": []}
    if n >= KILL_MEAN_N:
        m100 = float(r[:KILL_MEAN_N].mean())
        out["checks"].append({"rule": "mean_100", "n": KILL_MEAN_N, "mean": m100, "fires": m100 < KILL_MEAN_R})
        if m100 < KILL_MEAN_R:
            out.update(killed=True, rule=f"mean {m100:+.3f} R < {KILL_MEAN_R} over the first {KILL_MEAN_N} closed trades", at_n=KILL_MEAN_N)
    for k in sorted({c for c in CHECKPOINTS if c >= KILL_UB_MIN_N and c <= n} | ({n} if n >= KILL_UB_MIN_N else set())):
        ub = upper_bound(closed[:k])
        fires = bool(np.isfinite(ub["ub"]) and ub["ub"] < 0)
        out["checks"].append({"rule": "ub95", "n": k, "mean": float(r[:k].mean()), "ub95": ub["ub"], "block_days": ub["block_days"], "fires": fires})
        if fires and not out["killed"]:
            out.update(killed=True, rule=f"one-sided 95% upper bound {ub['ub']:+.3f} R < 0 at {k} closed trades", at_n=k)
    cur = [c for c in out["checks"] if c["rule"] == "ub95" and c["n"] == n]
    out["ub95"] = cur[0]["ub95"] if cur else None
    return out


def peg_stats(attempts: list[dict]) -> dict:
    res = [a["result"] for a in attempts]
    f, u, x = res.count("filled"), res.count("unfilled"), res.count("unchecked")
    rate = f / (f + u) if f + u else float("nan")
    return {"filled": f, "unfilled": u, "unchecked": x, "rate": rate, "alert": bool(f + u >= PEG_ALERT_MIN and rate < PEG_ALERT_RATE)}


def side_split(closed: list[dict]) -> dict:
    out = {}
    for lab, d in (("long", 1), ("short", -1)):
        r = [t["R"] for t in closed if t["side"] == d]
        out[lab] = {"n": len(r), "mean": float(np.mean(r)) if r else float("nan")}
    return out


# ====================================================================== the monitor (stream + rules + latch)

def monitor(as_of: datetime, *, root: Path | None = None, write: bool = False, stream_fn=None, book_trades: list[dict] | None = None) -> dict:
    """Evaluate the H-WS16-15 rules for both sleeves as of `as_of`. `root`: folder of peg_checks.json and monitor.json
    (real sessions write them; pipeline tests and status only read). `stream_fn(as_of, peg)` defaults to stream()."""
    as_of = utc(as_of)
    latch_p = root / "monitor.json" if root is not None else None
    latch = {}
    if latch_p is not None and latch_p.is_file():
        try:
            latch = json.loads(latch_p.read_text(encoding="utf-8")).get("kills", {})
        except (OSError, ValueError):
            latch = {}
    doc: dict[str, Any] = {"version": VERSION, "as_of": as_of.isoformat(), "forward_start": FORWARD_START.isoformat(), "sleeves": {}, "error": None}
    try:
        peg = PegChecker(as_of, root / "peg_checks.json" if root is not None else None, write=write)
        st = (stream_fn or (lambda t, pg: stream(t, peg=pg)))(as_of, peg)
        peg.save()
    except Exception as e:  # the latch still blocks a killed sleeve
        st, doc["error"] = None, f"{type(e).__name__}: {str(e)[:300]}"
    for name, spec in SLEEVES.items():
        lk = latch.get(name)
        if lk is not None and utc(lk["at"]) > as_of:
            lk = None                                   # point in time: a kill latched after as_of is not known yet
        row: dict[str, Any] = {"spec": spec.spec, "role": spec.role, "hypothesis": spec.hypothesis, "latched": lk}
        if st is not None:
            v = st[name]
            kr = kill_rules(v["closed"])
            row.update({"closed": kr["n"], "mean": kr["mean"], "ub95": kr["ub95"], "checks": kr["checks"], "rule_kill": kr["rule"] if kr["killed"] else None,
                        "open": len(v["open"]), "open_mark_R": float(sum(x["R_mark"] for x in v["open"])), "carried": v.get("carried", 0),
                        "sides": side_split(v["closed"]),
                        "peg": peg_stats(v["attempts"]), "errors": v["errors"]})
            if kr["killed"] and lk is None:
                lk = {"at": as_of.isoformat(), "rule": kr["rule"], "n": kr["at_n"]}
                row["latched"] = lk
                if write and latch_p is not None:
                    latch[name] = lk
        if book_trades is not None:
            mine = [t for t in book_trades if t.get("strategy") == name]
            row["book"] = {"closed": len(mine), "sum_R": float(sum(t["r_multiple"] for t in mine))}
        row["killed"] = lk is not None
        row["kill_reason"] = (lk or {}).get("rule")
        doc["sleeves"][name] = row
    if write and latch_p is not None:
        latch_p.parent.mkdir(parents=True, exist_ok=True)
        tmp = latch_p.with_suffix(".json.tmp")
        tmp.write_text(json.dumps({"version": VERSION, "kills": latch, "last": {k: {kk: vv for kk, vv in v.items() if kk not in ("checks", "errors")}
                                                                                for k, v in doc["sleeves"].items()}, "as_of": doc["as_of"]},
                                  indent=1, default=float), encoding="utf-8")
        tmp.replace(latch_p)
    return doc


def killed(doc: dict | None, sleeve: str) -> str | None:
    row = ((doc or {}).get("sleeves") or {}).get(sleeve) or {}
    return row.get("kill_reason") if row.get("killed") else None


def _f(x: Any, fmt: str = "+.3f") -> str:
    return format(x, fmt) if isinstance(x, (int, float)) and math.isfinite(x) else "-"


def describe(doc: dict | None) -> list[str]:
    """Packet / status lines (<= 3 per sleeve)."""
    if not doc:
        return ["H-WS16-15 monitor: not run"]
    L = []
    if doc.get("error"):
        L.append(f"H-WS16-15 monitor: stream unavailable ({doc['error']}); latched kills still apply")
    for name, r in doc["sleeves"].items():
        head = f"- spec {r['spec']} {name} ({r['role']}): " + ("KILLED (" + str(r["kill_reason"]) + "): no new entries" if r["killed"] else "active")
        if "closed" not in r:
            L.append(head)
            continue
        sd, pg = r["sides"], r["peg"]
        L.append(head + f" | unconstrained stream (23 perps, entries after {doc['forward_start'][:16]}Z): closed {r['closed']} "
                 f"(long {sd['long']['n']} {_f(sd['long']['mean'])} / short {sd['short']['n']} {_f(sd['short']['mean'])}), mean {_f(r['mean'])} R"
                 + (f", UB95 {_f(r['ub95'])}" if r.get("ub95") is not None else "") + f", open {r['open']} (marked {_f(r['open_mark_R'], '+.2f')} R)"
                 + (f"; {r['carried']} positions from before the start run on uncounted" if r.get("carried") else ""))
        nxt = next((c for c in CHECKPOINTS if c > r["closed"]), None)
        L.append(f"  kill rules: mean < {KILL_MEAN_R} at {KILL_MEAN_N} / UB95 < 0 at >= {KILL_UB_MIN_N}"
                 + (f" (next checkpoint {nxt})" if nxt else "") + f" | peg 15 min: {pg['filled']} filled / {pg['unfilled']} unfilled / {pg['unchecked']} unchecked"
                 + (f" ({pg['rate']:.0%})" if math.isfinite(pg["rate"]) else "") + (" ALERT < 70%" if pg["alert"] else "")
                 + (f" | book: {r['book']['closed']} closed, sum {r['book']['sum_R']:+.2f} R" if r.get("book") else "")
                 + (f" | data errors: {', '.join(sorted(r['errors']))}" if r.get("errors") else ""))
    return L


__all__ = ["FORWARD_START", "LEGACY", "PRIOR_NET_R", "REF_P_TP", "RETIRED", "SLEEVES", "STOP_K", "THETA", "UNIVERSE_L", "VERSION", "PegChecker",
           "SleeveSpec", "SymbolSignals", "compute_signals", "describe", "exits_due", "kill_rules", "killed", "monitor", "net_r", "peg_stats",
           "reentry_block", "signals_from_frame", "stop_k", "stream", "symbol_signals", "upper_bound", "walk"]
