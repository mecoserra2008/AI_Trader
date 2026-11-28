"""Decision session: the tool layer Claude calls each session (research/PLAYBOOK.md).

    PYTHONIOENCODING=utf-8 uv run python -m ai_trader.agent.session [SYMBOLS ...] [--out DIR]

1. State cards for every symbol (regime, robust / HAR / weekend sigma, the trend sleeves' 4h signals).
2. Candidates from the sleeves of the sleeve config (agent.trend_sleeves; user decisions D4 / D5, 2026-11-27):
     trend_zoo_ew   WS16 spec A (zoo_ew, primary): |s| >= 0.30 at the 4h close, stop 4 x sigma_risk 4h, signal exit
     trend_tsmom30  WS16 spec B (tsmom_30d, benchmark): the same trade layer
   No target: the sleeves exit on their own signal (a rule exit the decision harness sends), so p_quant / R:R do not
   apply and net E[R] is the documented WS16 forward expectation (trend_sleeves.PRIOR_NET_R).
   Retired: shock_regime (H-S2 shock fade, D5) and the 4h ema_stack bracket trend_4h (D4; its open positions run to
   completion under their bracket). The bracket machinery below (p_first, cap_stop, PRIOR_EDGE) stays for them.
   Stop cap (H-STOP1, user decision 2026-11-14): no stop farther than the applied cap (ai_trader.risk.stop_posterior:
   4 sigma prior, weekly posterior, the same value crates/at-risk enforces) in per-bar risk sigmas of the sleeve's
   frequency (4h for the trend sleeves), kept 1% inside it for tick rounding.
3. Playbook gates: TradFi weekend => no new directional entries (S4); stance per scenario.
4. Writes data/sessions/<id>/{cards.txt, candidates.json, proposals.json} and journals session_start / snapshot /
   decision. proposals.json is what `atx propose --file ... [--submit]` sends through the Rust risk engine.
Claude's views are applied on top as bounded log-odds tilts (ARCHITECTURE §2.4) before submission.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass, field
from datetime import timedelta
from pathlib import Path

import numpy as np
import polars as pl

from ai_trader.core.config import load
from ai_trader.core.time import now
from ai_trader.features.vol_models import har_forecast
from ai_trader.journal import Journal
from ai_trader.research.calendar import add_calendar
from ai_trader.research.frame import market_frame
from ai_trader.research.regimes import CELL_LABELS, label

DEFAULT = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "ADAUSDT", "AVAXUSDT", "LINKUSDT", "XRPUSDT", "DOGEUSDT"]
CLUSTER = {"BTCUSDT": "majors", "ETHUSDT": "majors", "SOLUSDT": "l1", "ADAUSDT": "l1", "AVAXUSDT": "l1",
           "LINKUSDT": "defi", "XRPUSDT": "payments", "DOGEUSDT": "meme"}
# Documented prior edges over the driftless null (probability points), from the scoreboard; deliberately small.
# Since 2026-11-14 these are the PRIORS of ai_trader.risk.edge_posterior (30 pseudo-trades, weekly update from closed
# paper trades). Bracket sleeves only: since D4 / D5 (2026-11-27) only the legacy trend_4h remains (open positions;
# execution.gate maps zoo_ensemble to it); shock_regime was removed with its sleeve (D5). The trend sleeves have no
# target, so no p_tp edge (agent.trend_sleeves).
PRIOR_EDGE = {"trend_4h": 0.03}


@dataclass
class Candidate:
    id: str
    symbol: str
    sleeve: str
    side: str
    entry_type: str
    entry: float
    stop: float
    take_profit: float | None            # None: an own-exit sleeve (no resting target)
    p_null: float | None
    p_quant: float | None
    rr: float | None
    blocked: str | None
    rationale: str
    cluster: str
    shock_ts: str | None = None
    sigma_per_bar: float | None = None   # risk sigma of one bar of `freq` (the engine's placement unit)
    freq: str = "1h"
    stop_sigma: float | None = None      # |ln(entry/stop)| / sigma_per_bar after the cap
    flags: list = field(default_factory=list)
    exit_rule: str | None = None         # "signal_flat": exit at the first 4h close with s x side <= 0 (trend sleeves)
    sleeve_signal: float | None = None   # the sleeve signal s at the decision close
    expected_r: float | None = None      # documented net E[R] per closed trade of an own-exit sleeve (no target)
    signal_close: str | None = None      # the 4h close the signal belongs to


def _daily_har(f: pl.DataFrame) -> float:
    d = (f.with_columns(pl.col("event_ts").dt.date().alias("d")).group_by("d")
         .agg((pl.col("ret").fill_nan(None).fill_null(0) ** 2).sum().alias("rv"), pl.len().alias("n")).filter(pl.col("n") == 24).sort("d"))
    fc = har_forecast(d["rv"].to_numpy())
    return float(np.sqrt(fc[-1])) if len(fc) and np.isfinite(fc[-1]) else float("nan")


def entry_cost_bps(entry_type: str) -> float:
    """Entry leg cost (bp) as at-risk charges it: market = taker + slippage, limit = maker, pegged_limit (limit pegged
    to the touch, H-EXE1 / R-EXE-2) = maker + paper_entry.slip_bps from config/risk.yaml."""
    r = load().risk
    k = r["costs"]
    if entry_type == "market":
        return k["taker_bps"] + k["slippage_bps"]
    if entry_type == "pegged_limit":
        return k["maker_bps"] + float((r.get("paper_entry") or {})["slip_bps"])
    return k["maker_bps"]


def net_er(c: "Candidate") -> float:
    """Net expected R exactly as the Rust risk engine computes it (at-risk intent.rs): round-trip costs (entry leg by
    entry type, see entry_cost_bps; taker + slippage exit) added to the stop distance. An own-exit candidate (no target)
    carries its documented net expectation instead (the engine then checks the intent's net_er_r); missing -> -1."""
    if c.take_profit is None:
        return float(c.expected_r) if c.expected_r is not None and math.isfinite(c.expected_r) else -1.0
    k = load().risk["costs"]
    entry_bps = entry_cost_bps(c.entry_type)
    cost = c.entry * (entry_bps + k["taker_bps"] + k["slippage_bps"]) / 1e4
    sd, td = abs(c.entry - c.stop), abs(c.take_profit - c.entry)
    return c.p_quant * (td - cost) / (sd + cost) - (1 - c.p_quant)


def p_first(entry: float, stop: float, tp: float) -> float:
    """Driftless first-passage null in log space: P(TP before SL) = b / (a + b)."""
    a, b = abs(math.log(tp / entry)), abs(math.log(entry / stop))
    return b / (a + b)


def flow_rows(sym: str, t0, t1, as_of, root: Path | None = None) -> pl.DataFrame:
    """flow_3s rows of `sym` (every category) with event_ts in [t0, t1) known at as_of, the latest revision per candle,
    read from the date partitions of the window only. Same rows as Lake.read_asof("flow_3s", ...), which scans every
    file of the table (at ~20 days of 3 s data: > 1 GB and minutes per call)."""
    from ai_trader.core.time import utc

    t0, t1, as_of = utc(t0), utc(t1), utc(as_of)
    base = Path(root or load().lake_dir) / "flow_3s" / "venue=bybit"
    days, d = [], t0.date()
    while d <= (t1 - timedelta(milliseconds=1)).date():
        days.append(d.isoformat())
        d += timedelta(days=1)
    parts = []
    for cdir in sorted(base.glob("category=*")) if base.is_dir() else []:
        for day in days:
            pdir = cdir / f"date={day}"
            if pdir.is_dir() and any(pdir.glob("*.parquet")):
                parts.append(pl.scan_parquet(str(pdir / "*.parquet"), hive_partitioning=False)
                             .filter((pl.col("symbol") == sym) & (pl.col("event_ts") >= t0) & (pl.col("event_ts") < t1) & (pl.col("known_ts") <= as_of))
                             .with_columns(pl.lit(cdir.name.split("=", 1)[1]).alias("category")))
    if not parts:
        return pl.DataFrame()
    out = pl.concat(parts, how="diagonal_relaxed").collect()
    return out.sort("known_ts").unique(["category", "event_ts"], keep="last").sort("event_ts") if out.height else out


def flow_context(sym: str, t, minutes: int = 60) -> tuple[str, dict]:
    """Descriptive order-flow context from flow_3s (H-F*/H-AS* untested: context only, never a signal)."""
    d = flow_rows(sym, t - timedelta(minutes=minutes), t + timedelta(milliseconds=1), t)
    if d.height == 0:
        return "flow: no flow_3s data", {}
    out, parts = {}, []
    for cat, g in d.group_by("category"):
        g = g.sort("event_ts")
        tot = (g["buy_notional"] + g["sell_notional"]).sum()
        delta = float((g["buy_notional"] - g["sell_notional"]).sum() / tot) if tot else 0.0
        mids = g["mid"].drop_nulls()
        ret = float((mids[-1] / mids[0] - 1) * 1e4) if mids.len() > 1 else 0.0
        o = {"delta": delta, "vol_musd": float(tot / 1e6), "ret_bps": ret, "obi_l1": float(g["obi_l1"].mean() or 0),
             "obi_10": float(g["obi_10"].mean() or 0), "obi_25": float(g["obi_25"].mean() or 0), "span_bps": float(g["book_span_bps"].median() or 0),
             "age_s": (t - g["event_ts"].max()).total_seconds()}
        out[cat[0]] = o
        parts.append(f"{cat[0]} delta {delta:+.2f} vol ${o['vol_musd']:.1f}M ret {ret:+.0f}bps obi l1/10/25 {o['obi_l1']:+.2f}/{o['obi_10']:+.2f}/{o['obi_25']:+.2f}")
    txt = f"flow {minutes}m: " + " | ".join(parts)
    if "linear" in out and "spot" in out and abs(out["linear"]["delta"] - out["spot"]["delta"]) > 0.3:
        txt += f"  [spot/perp taker divergence: perp {out['linear']['delta']:+.2f} vs spot {out['spot']['delta']:+.2f}; H-F3 untested]"
    age = min(v["age_s"] for v in out.values())
    if age > 420:
        txt += f"  [STALE: newest candle {age:.0f}s old]"
    return txt, out


def liq_context(sym: str, t) -> tuple[str, dict]:
    """Forced flow from the allLiquidation stream (collected since 2026-11-08 00:43 UTC). Context only."""
    from ai_trader.storage.lake import Lake

    d = Lake().read_asof("liquidations", t, symbols=[sym], start=t - timedelta(hours=24))
    if d.height == 0:
        return "liquidations 24h: none recorded", {}
    d = d.with_columns((pl.col("price") * pl.col("size")).alias("usd"))
    out = {}
    for lab, win in (("1h", d.filter(pl.col("event_ts") >= t - timedelta(hours=1))), ("24h", d)):
        lo = float(win.filter(pl.col("side") == "long")["usd"].sum())
        sh = float(win.filter(pl.col("side") == "short")["usd"].sum())
        out[lab] = {"long_usd": lo, "short_usd": sh, "n": win.height}
    big = d.sort("usd").tail(1).row(0, named=True)
    out["largest"] = {"usd": big["usd"], "side": big["side"], "ts": str(big["event_ts"])}
    f = lambda x: f"${x / 1e3:,.0f}k" if x < 1e6 else f"${x / 1e6:,.1f}M"  # noqa: E731
    return (f"liquidations: 1h longs {f(out['1h']['long_usd'])} / shorts {f(out['1h']['short_usd'])} | 24h longs {f(out['24h']['long_usd'])} / "
            f"shorts {f(out['24h']['short_usd'])} ({out['24h']['n']} events), largest {f(big['usd'])} {big['side']} at {big['event_ts']:%H:%M}"), out


TICK_MARGIN = 0.01   # keep stops 1% inside the sigma bounds (the engine rounds the stop away from entry to the tick)


def p_with_edge(p_null: float, edge: float) -> float:
    return min(0.95, max(0.0, p_null + edge))


def cap_stop(c: Candidate, cap: float, min_sig: float, tp_r: float | None, edge: float | None = None) -> Candidate:
    """Apply the stop cap (H-STOP1) in per-bar sigmas: a farther stop is pulled in to the cap; `tp_r` (a sleeve that sets
    its target as an R multiple) keeps that multiple, None keeps the structural target. Too-close stops are blocked."""
    sig = c.sigma_per_bar
    if not (sig and np.isfinite(sig) and sig > 0):
        c.blocked = c.blocked or "no valid per-bar risk sigma for placement"
        return c
    d = 1 if c.side == "long" else -1
    k = abs(math.log(c.entry / c.stop)) / sig
    if k > cap * (1 - TICK_MARGIN):
        k_new = cap * (1 - TICK_MARGIN)
        c.stop = c.entry * math.exp(-d * k_new * sig)
        if tp_r is not None:
            c.take_profit = c.entry * math.exp(d * tp_r * k_new * sig)
        c.flags.append(f"stop_capped {k:.2f}->{k_new:.2f} sigma (cap {cap:.2f})")
        k = k_new
        c.p_null = p_first(c.entry, c.stop, c.take_profit)
        c.p_quant = p_with_edge(c.p_null, PRIOR_EDGE.get(c.sleeve, 0.0) if edge is None else edge)
        c.rr = abs(math.log(c.take_profit / c.entry)) / abs(math.log(c.entry / c.stop))
    if k < min_sig * (1 + TICK_MARGIN):
        c.blocked = c.blocked or f"stop {k:.2f} sigma inside min_stop_distance_sigma {min_sig}"
    c.stop_sigma = k
    return c


def trend_candidates(sym: str, px: float, sg, *, weekend: bool, cap: float, min_sig: float, cell: str | None) -> list[Candidate]:
    """Entry candidates of the trend sleeves (agent.trend_sleeves) at the latest 4h close: |s| >= THETA, side = sign(s),
    stop STOP_K x sigma_risk 4h from the decision price (never beyond the cap), no target, documented net E[R]. Book
    state (flat in the sleeve, re-entry after a stop, kill, engine caps) is applied by agent.decision.prepare."""
    from ai_trader.agent import trend_sleeves as ts

    out: list[Candidate] = []
    if not sg.warm or len(sg) == 0:
        return out
    sig4 = float(sg.sigma4[-1])
    k = ts.stop_k(cap)
    close_iso = sg.last_close().isoformat() if sg.last_close() else None
    for name, spec in ts.SLEEVES.items():
        s = sg.last(spec.signal)
        if not (abs(s) >= ts.THETA and math.isfinite(sig4) and sig4 > 0):
            continue
        d = 1 if s > 0 else -1
        stop = px * math.exp(-d * k * sig4)
        blocked = "TradFi weekend: no new directional entries (S4)" if weekend else None
        if k < min_sig * (1 + TICK_MARGIN):
            blocked = blocked or f"stop {k:.2f} sigma inside min_stop_distance_sigma {min_sig}"
        flags = [f"stop_capped {ts.STOP_K:.2f}->{k:.2f} sigma (cap {cap:.2f})"] if k < ts.STOP_K * (1 - TICK_MARGIN) - 1e-9 else []
        c = Candidate(f"{spec.prefix}-{sym[:4]}", sym, name, "long" if d > 0 else "short", "limit", px, stop, None, None, None, None, blocked,
                      f"spec {spec.spec} {spec.signal} s {s:+.2f} at the {close_iso[11:16] if close_iso else '?'}Z close in {cell}; exit at the first 4h close "
                      f"with s x side <= 0, no target", CLUSTER.get(sym, "other"), sigma_per_bar=sig4, freq="4h", stop_sigma=k, flags=flags,
                      exit_rule="signal_flat", sleeve_signal=s, expected_r=ts.PRIOR_NET_R, signal_close=close_iso)
        out.append(c)
    return out


def analyse(sym: str, t, lookback_bars: int = 1, cap: float | None = None, edges: dict | None = None) -> tuple[str, list[Candidate], dict]:
    """State card, candidates and state for one symbol as of t: the trend sleeves' entry candidates (agent.trend_sleeves)
    at the latest 4h close, entered at the current price. `cap`: the stop cap in per-bar sigmas (default the applied
    H-STOP1 cap at t). `lookback_bars` (the retired shock sleeve's new-signal window) and `edges` (bracket-sleeve p_tp
    edges; none active since D4 / D5) are kept for callers and recorded in the state."""
    from ai_trader.agent import trend_sleeves as ts
    from ai_trader.risk.edge_posterior import applied_edges
    from ai_trader.risk.stop_posterior import applied_cap

    if cap is None:
        cap = applied_cap(t)[0]
    edges = {**PRIOR_EDGE, **(applied_edges(t)[0] if edges is None else edges)}
    min_sig = float(load().risk["placement"]["min_stop_distance_sigma"])
    f = label(market_frame(sym, "1h", t), 1.0)
    sg = ts.symbol_signals(sym, t)
    last = f.tail(1).row(0, named=True)
    px = float(last["close"])
    cal = add_calendar(f.tail(1)).row(0, named=True)
    weekend = bool(cal["tradfi_weekend"])
    har = _daily_har(f)
    sig_risk_h = float(last["sigma_risk_next"])
    cands = trend_candidates(sym, px, sg, weekend=weekend, cap=cap, min_sig=min_sig, cell=last["cell"])
    zoo, tsm = sg.last("zoo_ew"), sg.last("tsmom_30d")
    sig4 = float(sg.sigma4[-1]) if len(sg) else float("nan")
    lc = sg.last_close()

    flow_txt, flow = flow_context(sym, t)
    liq_txt, liq = liq_context(sym, t)
    card = (f"{sym} {px:,.4f}  {last['event_ts']:%a %H:%M} UTC  {'[TRADFI WEEKEND] ' if weekend else ''}{last['cell']} {CELL_LABELS.get(last['cell'], '')}"
            f"\n  vol: risk σ/h {sig_risk_h:.2%}, risk σ/4h {sig4:.2%}, central σ/h {last['sigma_rb_next']:.2%} (m {last['wk_m']:.2f}), HAR day {har:.2%}"
            + (f"\n  trend sleeves at the {lc:%H:%M}Z close: zoo_ew (spec A) {zoo:+.2f} | tsmom_30d (spec B) {tsm:+.2f}" if lc else "\n  trend sleeves: no 4h bars")
            + ("" if sg.warm else " [history too short]")
            + f"\n  {flow_txt}"
            f"\n  {liq_txt}")
    state = {"symbol": sym, "price": px, "cell": last["cell"], "weekend": weekend, "har_day": har, "sigma_risk_h": sig_risk_h, "sigma_risk_4h": sig4,
             "zoo_ew": zoo, "tsmom_30d": tsm, "signal_close": lc.isoformat() if lc else None, "signals_warm": sg.warm, "flow": flow, "liquidations": liq,
             "bar_ts": str(last["event_ts"]), "data_max_ts": str(f["known_ts"].max()), "data_max_ts_4h": sg.data_max_ts,
             "label": CELL_LABELS.get(last["cell"], ""), "sigma_rb_h": float(last["sigma_rb_next"]), "wk_m": float(last["wk_m"]), "stop_cap": cap, "edges": edges,
             "lookback_bars": lookback_bars}
    return card, cands, state


def run(symbols: list[str], out_root: Path) -> Path:
    t = now()
    sid = f"S{t:%Y%m%d-%H%M}"
    out = out_root / sid
    out.mkdir(parents=True, exist_ok=True)
    cfg = load()
    j = Journal(sid)
    j.write("session_start", {"model_id": cfg.settings["llm"]["model"], "prompt_version": cfg.settings["llm"]["prompt_version"], "symbols": symbols})
    cards, cands, states = [], [], []
    for sym in symbols:
        try:
            c, k, st = analyse(sym, t)
        except Exception as e:  # one bad symbol must not stop the session
            c, k, st = f"{sym}: unavailable ({e})", [], {"symbol": sym, "error": str(e)}
        cards.append(c)
        cands += k
        states.append(st)
    j.write("snapshot", {"states": states})
    (out / "cards.txt").write_text("\n\n".join(cards), encoding="utf-8")
    (out / "candidates.json").write_text(json.dumps([asdict(c) for c in cands], indent=1), encoding="utf-8")
    gate = load().risk["min_net_expected_r"]
    for c in cands:
        if not c.blocked and net_er(c) < gate:
            c.blocked = f"net E[R] {net_er(c):+.3f} after costs < gate {gate} (risk engine would reject)"
    # own-exit candidates (no target) go through the decision harness only (scripts/decision_session.py): atx propose
    # needs a resting target
    live = [c for c in cands if not c.blocked and c.take_profit is not None]
    live.sort(key=lambda c: -net_er(c))
    proposals = {"session_id": sid, "clusters": {c.symbol: c.cluster for c in live},
                 "proposals": [{"id": c.id, "symbol": c.symbol, "side": c.side, "entry_type": c.entry_type, "entry": round(c.entry, 8),
                                "stop": round(c.stop, 8), "take_profit": round(c.take_profit, 8), "p_tp": round(c.p_quant, 4),
                                "cluster": c.cluster, "channel": c.sleeve} for c in live]}
    (out / "proposals.json").write_text(json.dumps(proposals, indent=1), encoding="utf-8")
    j.write("decision", {"candidates": [asdict(c) for c in cands], "proposed": [p["id"] for p in proposals["proposals"]]})
    print("\n\n".join(cards))
    print(f"\n{len(cands)} candidates, {len(live)} proposed, {len(cands) - len(live)} blocked -> {out}")
    for c in cands:
        geo = (f"TP {c.take_profit:,.4f}  RR {c.rr:.1f} p_null {c.p_null:.2f} p {c.p_quant:.2f}" if c.take_profit is not None
               else f"no target ({c.exit_rule}), s {c.sleeve_signal:+.2f}")
        state = "BLOCKED: " + c.blocked if c.blocked else ("proposed" if c.take_profit is not None else "decision harness only")
        print(f"  {c.id:10s} {c.sleeve:13s} {c.side:5s} {c.symbol:9s} entry {c.entry:,.4f} SL {c.stop:,.4f} {geo} net E[R] {net_er(c):+.3f}  {state}  | {c.rationale}")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("symbols", nargs="*", default=DEFAULT)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    run(a.symbols, Path(a.out) if a.out else load().root / "data" / "sessions")
