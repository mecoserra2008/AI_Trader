"""Decision session: the tool layer Claude calls each session (research/PLAYBOOK.md).

    PYTHONIOENCODING=utf-8 uv run python -m ai_trader.agent.session [SYMBOLS ...] [--out DIR]

1. State cards for every symbol (regime, robust / HAR / weekend sigma, shock context, sleeve signals).
2. Candidates from validated sleeves only:
     shock_regime  (P1: continuation in calm cells, fade in volatile cells) - forward test from 2026-11-08
     trend_4h      (P3: small ema_stack tilt) - low weight
   Structural placement: stops/targets from the sleeve's own rules and the validated sigma (risk sigma, never
   weekend-shrunk); p_quant = driftless first-passage null + a small documented prior edge per sleeve.
3. Playbook gates: TradFi weekend => no new directional entries except shock sleeves (S4); stance per scenario.
4. Writes data/sessions/<id>/{cards.txt, candidates.json, proposals.json} and journals session_start / snapshot /
   decision. proposals.json is what `atx propose --file ... [--submit]` sends through the Rust risk engine.
Claude's views are applied on top as bounded log-odds tilts (ARCHITECTURE §2.4) before submission.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
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


def _daily_har(f: pl.DataFrame) -> float:
    d = (f.with_columns(pl.col("event_ts").dt.date().alias("d")).group_by("d")
         .agg((pl.col("ret").fill_nan(None).fill_null(0) ** 2).sum().alias("rv"), pl.len().alias("n")).filter(pl.col("n") == 24).sort("d"))
    fc = har_forecast(d["rv"].to_numpy())
    return float(np.sqrt(fc[-1])) if len(fc) and np.isfinite(fc[-1]) else float("nan")


def net_er(c: "Candidate") -> float:
    """Net expected R exactly as the Rust risk engine computes it (at-risk/src/lib.rs): round-trip costs
    (entry fee maker/taker by entry type, taker exit, slippage on market legs) added to the stop distance."""
    k = load().risk["costs"]
    entry_bps = (k["taker_bps"] + k["slippage_bps"]) if c.entry_type == "market" else k["maker_bps"]
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


def analyse(sym: str, t) -> tuple[str, list[Candidate], dict]:
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
        if s[-1] != 0 and (len(s) < 2 or np.sign(s[-2]) != np.sign(s[-1])):
            d = 1 if s[-1] > 0 else -1
            if mode == "reversal":
                tp = r["open"]
                stop = (r["low"] - 0.5 * r["sigma_next"] * r["close"]) if d > 0 else (r["high"] + 0.5 * r["sigma_next"] * r["close"])
            else:
                stop = r["open"]
                tp = px * math.exp(d * 2 * abs(math.log(px / stop)))  # bracket TP at 2R; trailing handled by the sleeve
            if (tp - px) * d > 0 and (px - stop) * d > 0:
                pn = p_first(px, stop, tp)
                cands.append(Candidate(f"shk-{sym[:4]}", sym, "shock_regime", "long" if d > 0 else "short", "market", px, stop, tp, pn,
                                       min(0.95, pn + PRIOR_EDGE["shock_regime"]), abs(math.log(tp / px)) / abs(math.log(px / stop)), None,
                                       f"{mode} after {r['z']:+.1f}σ shock (P1, H-S2)", CLUSTER.get(sym, "other"), str(r["event_ts"])))

    # --- trend sleeve (4h ema_stack): small tilt; stop 1.5 x HAR daily sigma, TP at 3R
    tr = float(REGISTRY["ema_stack"]()(f4)[-1]) if f4.height >= 300 else 0.0  # needs real 4h history
    if abs(tr) >= 0.3 and np.isfinite(har):
        d = 1 if tr > 0 else -1
        stop = px * math.exp(-d * 1.5 * har)
        tp = px * math.exp(d * 4.5 * har)
        pn = p_first(px, stop, tp)
        blocked = "TradFi weekend: no new directional entries (S4)" if weekend else None
        cands.append(Candidate(f"trd-{sym[:4]}", sym, "trend_4h", "long" if d > 0 else "short", "limit", px, stop, tp, pn,
                               min(0.95, pn + PRIOR_EDGE["trend_4h"]), 3.0, blocked, f"4h ema_stack {tr:+.2f} in {last['cell']} (P3, weak)", CLUSTER.get(sym, "other")))

    flow_txt, flow = flow_context(sym, t)
    liq_txt, liq = liq_context(sym, t)
    card = (f"{sym} {px:,.4f}  {last['event_ts']:%a %H:%M} UTC  {'[TRADFI WEEKEND] ' if weekend else ''}{last['cell']} {CELL_LABELS.get(last['cell'], '')}"
            f"\n  vol: risk σ/h {sig_risk_h:.2%}, central σ/h {last['sigma_rb_next']:.2%} (m {last['wk_m']:.2f}), HAR day {har:.2%}"
            f"\n  shock: {shock_txt} | shock_regime signal {s[-1]:+.2f} | 4h trend {tr:+.2f}"
            f"\n  {flow_txt}"
            f"\n  {liq_txt}")
    state = {"symbol": sym, "price": px, "cell": last["cell"], "weekend": weekend, "har_day": har, "sigma_risk_h": sig_risk_h, "trend_4h": tr, "shock_signal": float(s[-1]), "flow": flow, "liquidations": liq}
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
