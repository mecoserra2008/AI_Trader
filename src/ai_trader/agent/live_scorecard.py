"""Live desk scorecard: H-LT1 (net expectancy of live trades) and H-LT2 (live fills vs the paper fill model).

Reads the journal `atx live` writes (data/journal/live for mainnet, data/journal/live_<env> otherwise):
  order.jsonl            action "entry_done": one line per entry attempt (filled qty, reason, time to first fill, mid at submit)
  fill.jsonl             leg "entry" / "exit": every execution (price, qty, fee, slip vs mid / vs stop)
  position_update.jsonl  action "closed": realised net P&L (Bybit closedPnl, fees included) and R

Paper fill model (R-EXE-2, config/risk.yaml paper_entry): a pegged entry always fills, 1.3 bp beyond the decision-time mid
(`paper_slip_bps` is logged with every live fill). Slippage sign: positive = adverse (paid more on a buy, got less on a sell).

  H-LT1  mean net R per closed trade > 0 with the 95% day-block bootstrap CI above 0, >= 100 closed trades.
  H-LT2  (a) >= 90% of entries fill within the 15-minute window; (b) mean entry slip vs mid within 1 bp of the paper 1.3 bp
         (one-sample t on slip - 1.3); (c) stop exits on the liquid tier within 3 bp of the stop on average. >= 50 entries.

Track records are split by (model_id, prompt_version, exec_version, live_profile_hash) (principle 8).

    PYTHONIOENCODING=utf-8 uv run python -m ai_trader.agent.live_scorecard [--env mainnet|demo] [--json]
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

H_LT1_MIN, H_LT2_MIN = 100, 50
WINDOW_MS = 15 * 60_000
STOP_SLIP_MAX_BPS = 3.0
FILL_RATE_MIN = 0.90
SLIP_TOL_BPS = 1.0


def journal_root(env: str = "mainnet", base: Path | None = None) -> Path:
    if base is None:
        from ai_trader.core.config import load

        base = load().root / "data" / "journal"
    return base / ("live" if env == "mainnet" else f"live_{env}")


def _jsonl(p: Path) -> list[dict]:
    if not p.is_file():
        return []
    out = []
    for line in p.read_text(encoding="utf-8").splitlines():
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


def read(root: Path, kind: str) -> list[dict]:
    if not root.is_dir():
        return []
    return [r for d in sorted(x for x in root.iterdir() if x.is_dir()) for r in _jsonl(d / f"{kind}.jsonl")]


def track_key(v: dict | None) -> str:
    v = v or {}
    return "|".join(str(v.get(k)) for k in ("model_id", "prompt_version", "exec_version", "live_profile_hash"))


@dataclass
class Trade:
    trade_id: str
    symbol: str
    side: str
    tier: str
    track: str
    submitted_ms: int
    filled_qty: float = 0.0
    planned_qty: float = 0.0
    reason: str = ""
    time_to_fill_ms: int | None = None
    window_ms: int = WINDOW_MS
    entry_slip_bps: float | None = None
    paper_slip_bps: float = 1.3
    exit_kind: str | None = None
    stop_slip_bps: float | None = None
    realised_usd: float | None = None
    r_net: float | None = None
    closed_ms: int | None = None
    day: str = ""


def _vwap(rows: list[dict], key: str) -> float | None:
    q = sum(float(r.get("exec_qty") or 0) for r in rows)
    if q <= 0:
        return None
    return sum(float(r.get(key) or 0) * float(r.get("exec_qty") or 0) for r in rows) / q


def trades(root: Path) -> list[Trade]:
    # entries run with demo test hooks (--test-behind-bps / --test-window-secs) are mechanics tests, not trades
    orders = [o for o in read(root, "order") if o.get("action") == "entry_done" and not o.get("test_hooks")]
    fills = read(root, "fill")
    closes = {c["trade_id"]: c for c in read(root, "position_update") if c.get("action") == "closed"}
    by_trade: dict[str, dict[str, list[dict]]] = defaultdict(lambda: {"entry": [], "exit": []})
    for f in fills:
        by_trade[f.get("trade_id", "")][f.get("leg", "entry")].append(f)
    out = []
    for o in orders:
        tid = o["trade_id"]
        t = Trade(trade_id=tid, symbol=o.get("symbol", ""), side=str(o.get("side", "")), tier=o.get("tier", ""), track=track_key(o.get("versions")),
                  submitted_ms=int(o.get("submitted_ms") or 0), filled_qty=float(o.get("filled_qty") or 0), planned_qty=float(o.get("qty") or 0),
                  reason=o.get("reason", ""), time_to_fill_ms=o.get("time_to_first_fill_ms"), window_ms=int(o.get("window_ms") or WINDOW_MS),
                  paper_slip_bps=float(o.get("paper_slip_bps") or 1.3))
        t.day = datetime.fromtimestamp(t.submitted_ms / 1000, tz=UTC).strftime("%Y-%m-%d") if t.submitted_ms else ""
        ent = by_trade[tid]["entry"]
        t.entry_slip_bps = _vwap(ent, "slip_bps_vs_mid") if ent else (o.get("slip_bps_vs_mid") if t.filled_qty > 0 else None)
        if ent:
            first = min(int(f.get("exec_ms") or 0) for f in ent)
            t.time_to_fill_ms = first - t.submitted_ms if first and t.submitted_ms else t.time_to_fill_ms
        ex = by_trade[tid]["exit"]
        if ex:
            t.exit_kind = ex[0].get("exit_kind")
            stops = [f for f in ex if f.get("exit_kind") == "StopLoss"]
            t.stop_slip_bps = _vwap(stops, "slip_bps_vs_stop") if stops else None
        c = closes.get(tid)
        if c:
            t.realised_usd, t.r_net, t.closed_ms = float(c.get("realised_usd") or 0), float(c.get("r_net") or 0), c.get("closed_ms")
            t.exit_kind = t.exit_kind or c.get("exit_kind")
        out.append(t)
    return out


def day_bootstrap_ci(values: list[float], days: list[str], n_boot: int = 2000, seed: int = 0) -> tuple[float, float] | None:
    """95% CI of the mean, resampling whole UTC days (trades within a day are not independent)."""
    if len(values) < 2:
        return None
    groups: dict[str, list[float]] = defaultdict(list)
    for v, d in zip(values, days, strict=True):
        groups[d].append(v)
    g = list(groups.values())
    rng = np.random.default_rng(seed)
    means = []
    for _ in range(n_boot):
        pick = [g[i] for i in rng.integers(0, len(g), len(g))]
        flat = [x for grp in pick for x in grp]
        means.append(float(np.mean(flat)))
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(lo), float(hi)


def t_stat(x: list[float], mu: float = 0.0) -> tuple[float, float] | None:
    """(mean - mu, t) of a one-sample t-test."""
    if len(x) < 2:
        return None
    a = np.asarray(x, dtype=float)
    sd = a.std(ddof=1)
    d = float(a.mean() - mu)
    return d, (d / (sd / math.sqrt(len(a))) if sd > 0 else math.inf if d else 0.0)


@dataclass
class Card:
    track: str
    entries: int = 0
    filled_in_window: int = 0
    fill_rate: float | None = None
    slip_n: int = 0
    slip_mean_bps: float | None = None
    slip_minus_paper_bps: float | None = None
    slip_t: float | None = None
    stop_exits_liquid: int = 0
    stop_slip_mean_bps: float | None = None
    closed: int = 0
    mean_r: float | None = None
    mean_usd: float | None = None
    total_usd: float = 0.0
    r_ci: tuple[float, float] | None = None
    h_lt1: str = "insufficient"
    h_lt2: str = "insufficient"
    notes: list[str] = field(default_factory=list)


def card(ts: list[Trade], track: str) -> Card:
    c = Card(track=track, entries=len(ts))
    filled = [t for t in ts if t.filled_qty > 0 and t.time_to_fill_ms is not None and t.time_to_fill_ms <= t.window_ms]
    c.filled_in_window = len(filled)
    c.fill_rate = len(filled) / len(ts) if ts else None
    slips = [t.entry_slip_bps for t in ts if t.filled_qty > 0 and t.entry_slip_bps is not None]
    c.slip_n = len(slips)
    if slips:
        c.slip_mean_bps = float(np.mean(slips))
        paper = float(np.mean([t.paper_slip_bps for t in ts if t.filled_qty > 0])) if slips else 1.3
        tt = t_stat(slips, paper)
        if tt:
            c.slip_minus_paper_bps, c.slip_t = tt
    stops = [t.stop_slip_bps for t in ts if t.tier == "liquid" and t.stop_slip_bps is not None]
    c.stop_exits_liquid = len(stops)
    c.stop_slip_mean_bps = float(np.mean(stops)) if stops else None
    closed = [t for t in ts if t.r_net is not None]
    c.closed = len(closed)
    if closed:
        c.mean_r = float(np.mean([t.r_net for t in closed]))
        c.mean_usd = float(np.mean([t.realised_usd for t in closed]))
        c.total_usd = float(np.sum([t.realised_usd for t in closed]))
        c.r_ci = day_bootstrap_ci([t.r_net for t in closed], [t.day for t in closed])
    if c.closed >= H_LT1_MIN and c.r_ci:
        c.h_lt1 = "supports" if c.r_ci[0] > 0 else ("refutes" if c.r_ci[1] < 0 else "inconclusive")
    else:
        c.notes.append(f"H-LT1 needs >= {H_LT1_MIN} closed trades ({c.closed})")
    if c.entries >= H_LT2_MIN and c.slip_minus_paper_bps is not None:
        ok_a = (c.fill_rate or 0) >= FILL_RATE_MIN
        ok_b = abs(c.slip_minus_paper_bps) <= SLIP_TOL_BPS
        ok_c = c.stop_slip_mean_bps is None or c.stop_slip_mean_bps <= STOP_SLIP_MAX_BPS
        c.h_lt2 = "supports" if ok_a and ok_b and ok_c else "refutes"
        if c.stop_slip_mean_bps is None:
            c.notes.append("H-LT2 (c): no liquid-tier stop exit yet")
    else:
        c.notes.append(f"H-LT2 needs >= {H_LT2_MIN} entries ({c.entries})")
    return c


def scorecard(root: Path) -> list[Card]:
    ts = trades(root)
    by: dict[str, list[Trade]] = defaultdict(list)
    for t in ts:
        by[t.track].append(t)
    return [card(v, k) for k, v in sorted(by.items())]


def _f(x: float | None, nd: int = 3) -> str:
    return "-" if x is None else f"{x:+.{nd}f}"


def render(cards: list[Card], root: Path) -> str:
    L = [f"live scorecard ({root})"]
    if not cards:
        L.append("  no live entries journaled yet")
    for c in cards:
        L += [f"\ntrack {c.track}",
              f"  entries {c.entries}, filled within window {c.filled_in_window} (rate {_f(c.fill_rate, 2)}; paper model 1.00, H-LT2 >= {FILL_RATE_MIN})",
              f"  entry slip vs mid: n {c.slip_n} mean {_f(c.slip_mean_bps, 2)} bp, minus paper {_f(c.slip_minus_paper_bps, 2)} bp (t {_f(c.slip_t, 2)}; H-LT2 |diff| <= {SLIP_TOL_BPS})",
              f"  liquid stop exits: n {c.stop_exits_liquid} mean slip {_f(c.stop_slip_mean_bps, 2)} bp (H-LT2 <= {STOP_SLIP_MAX_BPS})",
              f"  closed {c.closed}: mean net R {_f(c.mean_r)} (day-bootstrap 95% CI {c.r_ci and tuple(round(x, 3) for x in c.r_ci)}), mean {_f(c.mean_usd, 4)} USD, total {c.total_usd:+.4f} USD",
              f"  H-LT1 {c.h_lt1} | H-LT2 {c.h_lt2}"]
        L += [f"  note: {n}" for n in c.notes]
    return "\n".join(L)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--env", default="mainnet", help="mainnet (data/journal/live) | demo (data/journal/live_demo) | testnet")
    ap.add_argument("--root", type=Path, default=None, help="journal directory (overrides --env)")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)
    root = a.root or journal_root(a.env)
    cards = scorecard(root)
    print(json.dumps([asdict(c) for c in cards], indent=1) if a.json else render(cards, root))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
