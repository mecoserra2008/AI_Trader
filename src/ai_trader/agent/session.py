"""Decision session: the tool layer Claude calls each session (research/PLAYBOOK.md).

    PYTHONIOENCODING=utf-8 uv run python -m ai_trader.agent.session [SYMBOLS ...] [--out DIR]

1. State cards for every symbol (regime, robust / HAR / weekend sigma, shock context, sleeve signals).
2. Candidates from validated sleeves only:
     shock_regime  (P1: continuation in calm cells, fade in volatile cells) - forward test from 2026-11-08
     trend_4h      (P3: small ema_stack tilt) - low weight
   Structural placement: stops/targets from the sleeve's own rules and the validated sigma (risk sigma, never
   weekend-shrunk); p_quant = driftless first-passage null + a small documented prior edge per sleeve.
   Stop cap (H-STOP1, user decision 2026-11-14): a stop farther than the applied cap (ai_trader.risk.stop_posterior:
   4 sigma prior, weekly posterior, the same value crates/at-risk enforces) in per-bar risk sigmas of the sleeve's
   frequency (1h for shock_regime, 4h for trend_4h) is pulled in to the cap (1% inside, for tick rounding) and
   flagged `stop_capped`; continuation / trend targets keep their R multiple, the reversal target stays structural.
3. Playbook gates: TradFi weekend => no new directional entries except shock sleeves (S4); stance per scenario.
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
from ai_trader.strategies import REGISTRY

DEFAULT = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "ADAUSDT", "AVAXUSDT", "LINKUSDT", "XRPUSDT", "DOGEUSDT"]
CLUSTER = {"BTCUSDT": "majors", "ETHUSDT": "majors", "SOLUSDT": "l1", "ADAUSDT": "l1", "AVAXUSDT": "l1",
           "LINKUSDT": "defi", "XRPUSDT": "payments", "DOGEUSDT": "meme"}
# Documented prior edges over the driftless null (probability points), from the scoreboard; deliberately small.
# Since 2026-11-14 these are the PRIORS of ai_trader.risk.edge_posterior (30 pseudo-trades, weekly update from closed
# paper trades); analyse() uses the posterior mean in force at t.
PRIOR_EDGE = {"shock_regime": 0.05, "trend_4h": 0.03}


@dataclass
class Candidate:
    id: str
    symbol: str
    sleeve: str
    side: str
    entry_type: str
    entry: float
    stop: float
    take_profit: float
    p_null: float
    p_quant: float
    rr: float
    blocked: str | None
    rationale: str
    cluster: str
    shock_ts: str | None = None
    sigma_per_bar: float | None = None   # risk sigma of one bar of `freq` (the engine's placement unit)
    freq: str = "1h"
    stop_sigma: float | None = None      # |ln(entry/stop)| / sigma_per_bar after the cap
    flags: list = field(default_factory=list)


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
    entry type, see entry_cost_bps; taker + slippage exit) added to the stop distance."""
    k = load().risk["costs"]
    entry_bps = entry_cost_bps(c.entry_type)
    cost = c.entry * (entry_bps + k["taker_bps"] + k["slippage_bps"]) / 1e4
    sd, td = abs(c.entry - c.stop), abs(c.take_profit - c.entry)
    return c.p_quant * (td - cost) / (sd + cost) - (1 - c.p_quant)


def p_first(entry: float, stop: float, tp: float) -> float:
    """Driftless first-passage null in log space: P(TP before SL) = b / (a + b)."""
    a, b = abs(math.log(tp / entry)), abs(math.log(entry / stop))
    return b / (a + b)


def flow_context(sym: str, t, minutes: int = 60) -> tuple[str, dict]:
    """Descriptive order-flow context from flow_3s (H-F*/H-AS* untested: context only, never a signal)."""
    from ai_trader.storage.lake import Lake

    d = Lake().read_asof("flow_3s", t, symbols=[sym], start=t - timedelta(minutes=minutes))
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


def _new_signal_age(s: np.ndarray, lookback: int) -> int | None:
    """Bars since the held shock signal started, if it started within the last `lookback` bars and is still on."""
    n = len(s)
    if n == 0 or s[-1] == 0:
        return None
    sg = np.sign(s[-1])
    for j in range(n - 1, max(-1, n - 1 - lookback), -1):
        if np.sign(s[j]) != sg:
            return None
        if j == 0 or np.sign(s[j - 1]) != sg:
            return n - 1 - j
    return None


def analyse(sym: str, t, lookback_bars: int = 1, cap: float | None = None, edges: dict | None = None) -> tuple[str, list[Candidate], dict]:
    """State card, candidates and state for one symbol as of t. `lookback_bars`: a shock signal that started within
    that many 1h bars and is still held is a NEW signal (1 = only on the last bar; 4 for the 4-hourly decision session,
    entered at the current price on the sleeve's structural levels). `cap`: the stop cap in per-bar sigmas (default the
    applied H-STOP1 cap at t)."""
    from ai_trader.risk.stop_posterior import applied_cap

    from ai_trader.risk.edge_posterior import applied_edges

    if cap is None:
        cap = applied_cap(t)[0]
    edges = {**PRIOR_EDGE, **(applied_edges(t)[0] if edges is None else edges)}
    min_sig = float(load().risk["placement"]["min_stop_distance_sigma"])
    f = label(market_frame(sym, "1h", t), 1.0)
    f4 = market_frame(sym, "4h", t, funding=False, oi=False)
    last = f.tail(1).row(0, named=True)
    px = float(last["close"])
    cal = add_calendar(f.tail(1)).row(0, named=True)
    weekend = bool(cal["tradfi_weekend"])
    har = _daily_har(f)
    sig_risk_h = float(last["sigma_risk_next"])
    cands: list[Candidate] = []

    # --- shock sleeve: act only on a NEW signal (entry at this bar), with the strategy's own levels
    shock = REGISTRY["shock_regime"]()
    s = shock(f).to_numpy()
    shock_txt = "none in last 72h"
    recent = f.tail(72).filter(pl.col("z").abs() >= 4)
    if recent.height:
        r = recent.tail(1).row(0, named=True)
        mode = "continuation" if r["vol_state"] == "low" else "reversal"
        shock_txt = f"{r['z']:+.1f}σ at {r['event_ts']:%a %H:%M} in {r['cell']} ({r['vol_state']} vol) -> {mode}"
        age = _new_signal_age(s, lookback_bars)
        if age is not None:
            d = 1 if s[-1] > 0 else -1
            if mode == "reversal":
                tp = r["open"]
                stop = (r["low"] - 0.5 * r["sigma_next"] * r["close"]) if d > 0 else (r["high"] + 0.5 * r["sigma_next"] * r["close"])
            else:
                stop = r["open"]
                tp = px * math.exp(d * 2 * abs(math.log(px / stop)))  # bracket TP at 2R; trailing handled by the sleeve
            if (tp - px) * d > 0 and (px - stop) * d > 0:
                pn = p_first(px, stop, tp)
                why = f"{mode} after {r['z']:+.1f}σ shock (P1, H-S2)" + (f", signal {age}h old" if age else "")
                c = Candidate(f"shk-{sym[:4]}", sym, "shock_regime", "long" if d > 0 else "short", "market", px, stop, tp, pn,
                              p_with_edge(pn, edges["shock_regime"]), abs(math.log(tp / px)) / abs(math.log(px / stop)), None,
                              why, CLUSTER.get(sym, "other"), str(r["event_ts"]), sigma_per_bar=sig_risk_h, freq="1h")
                cands.append(cap_stop(c, cap, min_sig, 2.0 if mode == "continuation" else None, edges["shock_regime"]))

    # --- trend sleeve (4h ema_stack): small tilt; stop 1.5 x HAR daily sigma, TP at 3R
    tr = float(REGISTRY["ema_stack"]()(f4)[-1]) if f4.height >= 300 else 0.0  # needs real 4h history
    if abs(tr) >= 0.3 and np.isfinite(har):
        d = 1 if tr > 0 else -1
        stop = px * math.exp(-d * 1.5 * har)
        tp = px * math.exp(d * 4.5 * har)
        pn = p_first(px, stop, tp)
        blocked = "TradFi weekend: no new directional entries (S4)" if weekend else None
        sig4 = float(f4["sigma_risk_next"][-1]) if "sigma_risk_next" in f4.columns and f4.height else float("nan")
        c = Candidate(f"trd-{sym[:4]}", sym, "trend_4h", "long" if d > 0 else "short", "limit", px, stop, tp, pn,
                      p_with_edge(pn, edges["trend_4h"]), 3.0, blocked, f"4h ema_stack {tr:+.2f} in {last['cell']} (P3, weak)", CLUSTER.get(sym, "other"),
                      sigma_per_bar=sig4 if np.isfinite(sig4) else None, freq="4h")
        cands.append(cap_stop(c, cap, min_sig, 3.0, edges["trend_4h"]))

    flow_txt, flow = flow_context(sym, t)
    liq_txt, liq = liq_context(sym, t)
    card = (f"{sym} {px:,.4f}  {last['event_ts']:%a %H:%M} UTC  {'[TRADFI WEEKEND] ' if weekend else ''}{last['cell']} {CELL_LABELS.get(last['cell'], '')}"
            f"\n  vol: risk σ/h {sig_risk_h:.2%}, central σ/h {last['sigma_rb_next']:.2%} (m {last['wk_m']:.2f}), HAR day {har:.2%}"
            f"\n  shock: {shock_txt} | shock_regime signal {s[-1]:+.2f} | 4h trend {tr:+.2f}"
            f"\n  {flow_txt}"
            f"\n  {liq_txt}")
    state = {"symbol": sym, "price": px, "cell": last["cell"], "weekend": weekend, "har_day": har, "sigma_risk_h": sig_risk_h, "trend_4h": tr, "shock_signal": float(s[-1]), "flow": flow, "liquidations": liq,
             "bar_ts": str(last["event_ts"]), "data_max_ts": str(f["known_ts"].max()), "data_max_ts_4h": str(f4["known_ts"].max()) if f4.height else None,
             "label": CELL_LABELS.get(last["cell"], ""), "sigma_rb_h": float(last["sigma_rb_next"]), "wk_m": float(last["wk_m"]), "shock": shock_txt, "stop_cap": cap, "edges": edges}
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
    # A shock that hits many symbols in the same bar is ONE market bet, not N independent ones: shock candidates
    # from the same bar on >= 3 symbols share a cluster, so the risk engine's per-cluster cap keeps only the best one.
    by_bar: dict[str, list[Candidate]] = {}
    for c in cands:
        if c.shock_ts:
            by_bar.setdefault(c.shock_ts, []).append(c)
    for ts, group in by_bar.items():
        if len(group) >= 3:
            for c in group:
                c.cluster = f"market-shock-{ts[:13]}"
    j.write("snapshot", {"states": states})
    (out / "cards.txt").write_text("\n\n".join(cards), encoding="utf-8")
    (out / "candidates.json").write_text(json.dumps([asdict(c) for c in cands], indent=1), encoding="utf-8")
    gate = load().risk["min_net_expected_r"]
    for c in cands:
        if not c.blocked and net_er(c) < gate:
            c.blocked = f"net E[R] {net_er(c):+.3f} after costs < gate {gate} (risk engine would reject)"
    live = [c for c in cands if not c.blocked]
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
        print(f"  {c.id:10s} {c.sleeve:13s} {c.side:5s} {c.symbol:9s} entry {c.entry:,.4f} SL {c.stop:,.4f} TP {c.take_profit:,.4f}  "
              f"RR {c.rr:.1f} p_null {c.p_null:.2f} p {c.p_quant:.2f} net E[R] {net_er(c):+.3f}  {'BLOCKED: ' + c.blocked if c.blocked else 'proposed'}  | {c.rationale}")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("symbols", nargs="*", default=DEFAULT)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    run(a.symbols, Path(a.out) if a.out else load().root / "data" / "sessions")
