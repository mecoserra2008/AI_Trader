"""Markdown report for the STRESS block (research/reports/stress_test_2026-11-26.md) from results.json.

The plain-words summary and the hand-written caveats are read from data/research/stress/plain_words.md and
data/research/stress/caveats.md when present (written after reading the results); everything else is generated."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from ai_trader.research import stress as S

ORDER = ["tsmom", "ema_stack", "donchian_breakout", "compression_breakout", "geo_struct", "geo_pivot_bo", "geo_flag_bo", "zoo_ensemble", "trend_4h",
         "zscore_reversion", "rsi_reversion", "setup_c_proxy", "funding_contrarian", "oi_price_divergence", "hrb2", "shock_regime", "shock_desk",
         "hs2_fade_blackout", "weekend_shock_follow", "weekend_drift_fade", "cme_gap_fill", "flow_momentum", "cvd_divergence"]


def f(x: Any, nd: int = 3, sign: bool = True) -> str:
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return "n/a"
    if isinstance(x, bool):
        return "yes" if x else "no"
    if isinstance(x, int):
        return str(x)
    return f"{x:+.{nd}f}" if sign else f"{x:.{nd}f}"


def pct(x: Any, nd: int = 1) -> str:
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return "n/a"
    return f"{100 * x:.{nd}f}%"


def mm(d: dict | None, n: bool = False) -> str:
    if not d or not d.get("n"):
        return "-"
    s = f(d.get("mean"))
    return f"{s} ({d['n']})" if n else s


def table(head: list[str], rows: list[list[str]]) -> str:
    out = ["| " + " | ".join(head) + " |", "|" + "|".join("---" for _ in head) + "|"]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(out)


def _units(res: dict) -> list[str]:
    return [u for u in ORDER if u in res] + [u for u in res if u not in ORDER]


def write_report(out: dict, path: Path) -> None:
    res = out["units"]
    U = _units(res)
    pw = S.OUT / "plain_words.md"
    cv = S.OUT / "caveats.md"
    L: list[str] = []
    L.append("# Stress test of every deterministic strategy in every scenario, 2026-11-26\n")
    L.append("Agent STRESS. Research only: no orders, no venue calls. The LLM decision layer is not tested (CLAUDE.md principle 1); "
             "these are the deterministic signal generators, the desk setups as coded or specified, and the live risk engine limits "
             "(config/risk.yaml `live_small`) used as the sizing layer. Design and verdict rules were registered before any run "
             "(research/hypotheses.yaml block STRESS, H-STRESS0..10; diary 2026-11-25 \"STRESS design frozen before results\"). "
             f"Data as of {out['design_as_of'][:16]} UTC. R = one exchange-stop loss including round-trip costs; 1 R = 1.30 USD live.\n")
    if pw.exists():
        L.append(pw.read_text(encoding="utf-8").strip() + "\n")

    # ---- scorecard
    L.append("## Scorecard\n")
    L.append("Mean net R per trade, base trade layer, universe L pooled (23 crypto perps; flow units BTCUSDT only; setup_c_proxy BTC/ETH). "
             "`*` = data-generated unit (designed after seeing its evidence: historical rows are not evidence).\n")
    rows = []
    for u in U:
        r = res[u]
        b = r.get("base") or {}
        sg = r.get("segments") or {}
        tag = "*" if (r.get("unit") or {}).get("data_generated") else ""
        rows.append([f"{u}{tag}", f"**{r.get('verdict')}**", f(b.get("n")) if b.get("n") else "0",
                     f"{f(b.get('mean'))} [{f(b.get('lo'))}, {f(b.get('hi'))}]" if b.get("n") else "-",
                     mm(sg.get("llm_in_sample"), True), mm(sg.get("clean_holdout"), True), "; ".join(r.get("fired") or []) or "-"])
    L.append(table(["unit", "verdict", "n", "base mean R [95% CI]", "llm_in_sample (n)", "clean_holdout (n)", "rules fired"], rows) + "\n")

    L.append("### Costs, fills and latency (mean net R per trade)\n")
    rows = []
    for u in U:
        r = res[u]
        c, e = r.get("costs") or {}, r.get("exec") or {}
        sgp = r.get("stop_gaps") or {}
        rows.append([u, mm(r.get("base")), mm(c.get("fees_x1.5")), mm(c.get("fees_x2")), mm(c.get("slip+5")), mm(c.get("slip+10")), mm(c.get("slip+25")),
                     mm(e.get("gap")), mm(e.get("stop_next_open")), mm(e.get("no_stop")), mm(e.get("lat1")), mm(e.get("lat5")), mm(e.get("lat15")),
                     f"{sgp.get('gapped', 0)}/{sgp.get('stops', 0)}, worst {f(sgp.get('worst_gap_R'), 2)}" if sgp else "-"])
    L.append(table(["unit", "base", "fees x1.5", "fees x2", "+5 bp", "+10 bp", "+25 bp", "gap fill", "stop next open", "no stop", "lat 1m", "lat 5m", "lat 15m",
                    "gapped stops, worst gap (R)"], rows) + "\n")

    L.append("### Parameters, synthetic paths, correlation, liquidity, crises, live engine\n")
    rows = []
    for u in U:
        r = res[u]
        p = r.get("perturb") or {}
        sy = r.get("synthetic") or {}
        e = r.get("engine") or {}
        er = e.get("R") or {}
        rows.append([u, f(p.get("base_mean")), f(p.get("worst_pm20")), f(p.get("worst_pm50")), ", ".join(p.get("cliff_keys") or []) or ("-" if p else "n/a"),
                     pct(p.get("joint_share_pos"), 0), f((sy.get("sbb") or {}).get("median_mean")), f((sy.get("rs") or {}).get("median_mean")),
                     mm(r.get("corr")), mm((r.get("costs") or {}).get("drought")), f(r.get("crisis_engine_min"), 1),
                     f"{e.get('n', 0)}, {f(er.get('mean'))}" if e.get("n") else "0"])
    L.append(table(["unit", "D8 base", "worst +-20%", "worst +-50%", "cliff keys (+-20%)", "joint draws > 0", "SBB median path", "RS median path",
                    "co-stop (S7)", "drought (S8)", "worst crisis, engine (R)", "engine n, mean R"], rows) + "\n")

    # ---- risk engine
    L.append("## What the live risk limits imply (S9, live_small on each unit's pooled stream)\n")
    L.append("Registered S9: 1.30 USD per R, 2 USD worst case, 20 entries per UTC day, 10 USD daily budget, 3 open, 2 same side, 400 USD gross, "
             "tier caps, EV gate >= 0.10 where a probability is defined (bracket setups: first-passage null + prior edge). "
             "USD figures are strategy P&L under these limits, not account values. P(loss >= x) = probability that the cumulative P&L "
             "falls to -x USD at some point within 91 days (stationary bootstrap of days).\n")
    rows = []
    for u in U:
        e = (res[u].get("engine") or {})
        if not e.get("n"):
            ref = (e.get("refused") or {})
            rows.append([u, "0", "-", "-", "-", "-", "-", "-", "-", "-", "-", "-", ", ".join(f"{k} {v}" for k, v in ref.items()) or "-"])
            continue
        ref = e.get("refused") or {}
        rows.append([u, str(e["n"]), f(e.get("usd_per_trade")), f(e.get("usd_total"), 0), pct(e.get("p_daily_stop_hit"), 2), pct(e.get("p_daily_budget_refusal"), 2),
                     f(e.get("worst_day_usd"), 2), str(e.get("max_loss_run")), f"{pct(e.get('p5_losers_3m_empirical'), 0)} / {pct(e.get('p5_losers_3m_iid'), 0)}",
                     f"{f(e.get('mean_uw_spell_days'), 1, False)} d (max {e.get('max_uw_spell_days')})",
                     f"{pct(e.get('p_loss_ge_25'), 0)} / {pct(e.get('p_loss_ge_50'), 0)} / {pct(e.get('p_loss_ge_100'), 0)}",
                     f(e.get("q05_3m_usd"), 1), ", ".join(f"{k} {v}" for k, v in sorted(ref.items(), key=lambda x: -x[1])[:4]) or "-"])
    L.append(table(["unit", "trades", "USD/trade", "USD total", "P(day stop hit)", "P(budget refusal day)", "worst day USD", "max losing run",
                    "P(>=5 losers in 3m) emp / iid", "mean time under water", "P(3m loss >= 25 / 50 / 100 USD)", "3m 5th pct USD", "main refusals"], rows) + "\n")
    rows = []
    for u in U:
        e = res[u].get("engine_nogate") or {}
        if e.get("n") and (res[u].get("unit") or {}).get("kind") == "bracket":
            rows.append([u, str(e["n"]), f(e.get("usd_per_trade")), f(e.get("usd_total"), 0), pct(e.get("p_daily_stop_hit"), 2),
                         f"{pct(e.get('p_loss_ge_25'), 0)} / {pct(e.get('p_loss_ge_50'), 0)} / {pct(e.get('p_loss_ge_100'), 0)}", f(res[u].get("crisis_engine_nogate_min"), 1)])
    if rows:
        L.append("Bracket setups without the EV gate (the gate is the part a calibrated probability or an LLM view changes):\n")
        L.append(table(["unit", "trades", "USD/trade", "USD total", "P(day stop hit)", "P(3m loss >= 25 / 50 / 100)", "worst crisis (R)"], rows) + "\n")

    # ---- regimes and slices
    L.append("## Regime cells (S1): mean net R per trade by v0 cell at entry, pooled (n)\n")
    L.append("`+`/`-` after the number: sign in llm_in_sample and clean_holdout agree (both >= 10 trades); `?` they disagree.\n")
    rows = []
    for u in U:
        cells = res[u].get("cells") or {}
        row = [u]
        for c in [f"R{i}" for i in range(1, 10)]:
            v = cells.get(c) or {}
            p = v.get("pooled") or {}
            if not p.get("n"):
                row.append("-")
                continue
            a, b = v.get("llm_in_sample") or {}, v.get("clean_holdout") or {}
            mark = ""
            if a.get("n", 0) >= 10 and b.get("n", 0) >= 10:
                mark = "+" if (a["mean"] > 0) == (b["mean"] > 0) else "?"
            row.append(f"{f(p['mean'], 2)}{mark} ({p['n']})")
        rows.append(row)
    L.append(table(["unit"] + [f"R{i}" for i in range(1, 10)], rows) + "\n")
    rows = []
    for u in U:
        s = res[u].get("slices") or {}
        rows.append([u] + [mm(s.get(k), True) for k in ("weekday", "weekend", "nyse_holiday", "off_funding", "funding_window")])
    L.append("### Calendar slices\n")
    L.append(table(["unit", "weekday", "TradFi weekend", "NYSE holiday", "off funding", "+-1 h of funding"], rows) + "\n")

    # ---- crises
    L.append("## Crisis windows (S2)\n")
    W = list(out["windows"])
    L.append("Sum of net R of trades entered in the window: live-engine stream (raw universe stream in brackets). The 2020-03 window uses Deribit "
             "BTC-PERPETUAL 1h (Bybit history starts 2020-03-25), BTC only.\n")
    L.append("Windows: " + "; ".join(f"{w} = {a}..{b}" for w, (a, b) in out["windows"].items()) + "\n")
    rows = []
    for u in U:
        cr = res[u].get("crises") or {}
        row = [u]
        for w in W:
            v = cr.get(w) or {}
            e, rw = v.get("engine") or {}, v.get("raw") or {}
            row.append(f"{f(e.get('sum'), 1) if e.get('n') else '0'} ({f(rw.get('sum'), 0) if rw.get('n') else '0'})")
        rows.append(row)
    L.append(table(["unit"] + [w.split(" (")[0] for w in W], rows) + "\n")
    rows = []
    for u in U:
        cr = res[u].get("crises") or {}
        worst = sorted(((w, v["raw"]) for w, v in cr.items() if (v.get("raw") or {}).get("n")), key=lambda x: x[1]["sum"])[:1]
        if worst:
            w, v = worst[0]
            no = (cr.get(w) or {}).get("next_open") or {}
            rows.append([u, w, f(v["sum"], 1), f(v["max_dd"], 1), f(v["worst"], 2), str(v["n"]), str(v.get("stops", 0)),
                         f(no.get("mean_slip_R"), 2), f(no.get("worst_slip_R"), 2), f(no.get("worst_trade"), 2), f(no.get("sum"), 1)])
    L.append("Worst window per unit on the raw universe stream. Base stops fill at the stop price, and 1h bars of 24/7 perps open where the "
             "previous bar closed, so open-gap fills never differ from it (0 gapped stops in every unit). The slippage columns therefore use "
             "the stop_next_open variant (a stop touched in bar k fills at the open of bar k+1), the realistic bound in a fast market:\n")
    L.append(table(["unit", "window", "sum R", "max DD R", "worst trade R", "trades", "stops", "next-open slip mean R", "worst slip R",
                    "worst trade R (next open)", "sum R (next open)"], rows) + "\n")

    # ---- perturbation detail
    L.append("## Parameter surfaces (S5, D8 universe, mean net R per trade)\n")
    rows = []
    for u in U:
        p = res[u].get("perturb") or {}
        for k, d in (p.get("surface") or {}).items():
            rows.append([u, k] + [f(d.get(str(x))) for x in S.PERTURB_FACTORS] + [f(p.get("base_mean"))])
    L.append(table(["unit", "parameter", "x0.5", "x0.8", "x1.2", "x1.5", "base"], rows) + "\n")

    # ---- synthetic
    L.append("## Synthetic paths (S6)\n")
    L.append("One-year 1h paths after a 2000-bar warm-up; per path the unit's mean net R per trade and total R. Historical comparator: median over "
             "symbols of total R per symbol-year and max drawdown per symbol-year on real data.\n")
    rows = []
    for u in U:
        sy = res[u].get("synthetic") or {}
        if not sy:
            continue
        h = res[u].get("hist_symbol_year") or {}
        for g in ("sbb", "rs"):
            v = sy.get(g)
            if not v:
                continue
            rows.append([u, g.upper(), str(v["paths"]), f(v["median_mean"]), pct(v["p_mean_le0"], 0), f(v["median_total"], 1), f(v["q05_total"], 1), f(v["q05_max_dd"], 1),
                         f(h.get("total_per_year_median"), 1), f(h.get("max_dd_per_year_median"), 1)])
    L.append(table(["unit", "gen", "paths", "median mean R", "P(mean <= 0)", "median total R", "5th pct total R", "5th pct max DD", "hist total R / sym-yr",
                    "hist max DD / sym-yr"], rows) + "\n")

    # ---- latency check
    lc = {u: res[u].get("latcheck") for u in U if res[u].get("latcheck")}
    if lc:
        L.append("## Latency model check (BTCUSDT, 1h units): Brownian-bridge fills vs exact 5m-bar fills\n")
        rows = []
        for u, d in lc.items():
            for k, v in d.items():
                rows.append([u, k, str(v.get("n")), f(v.get("base")), f(v.get("bridge")), f(v.get("exact"))])
        L.append(table(["unit", "delay", "trades", "base", "bridge", "exact 5m"], rows) + "\n")

    # ---- MM
    if out.get("mm"):
        L.append("## Weekend MM quoters (existing replay results, mm_weekend_2026-11-21; not re-run)\n")
        L.append("Held mean net USD per symbol-day [day-clustered 95% CI] under the replay's own stress cells (queue model x latency). "
                 "No verdict: the MM quoters are outside the registered verdict rules (they are not R-denominated trade streams).\n")
        rows = []
        for k, v in out["mm"].items():
            c = v["cells"]
            rows.append([k] + [f"{f(c[x]['mean'], 4)} [{f(c[x]['ci'][0], 4)}, {f(c[x]['ci'][1], 4)}]" if x in c else "-" for x in
                               ("prorata_150ms", "prorata_300ms", "pessimistic_150ms", "pessimistic_300ms")] + [f(v["worst_cell_mean"], 4)])
        L.append(table(["quoter", "pro-rata 150 ms", "pro-rata 300 ms", "FIFO-pess. 150 ms", "FIFO-pess. 300 ms", "worst cell"], rows) + "\n")

    # ---- rule candidates
    L.append("## Rule candidates (suggestions only; research/rules.yaml is not edited)\n")
    rc = out.get("rule_candidates") or []
    if rc:
        L.append("Cells where a unit loses in both segments (>= 10 trades each, pooled >= 50, pooled mean <= -0.05 R): candidates for \"do not take unit X in cell Y\".\n")
        L.append(table(["unit", "cell", "pooled mean R", "n", "llm_in_sample", "clean_holdout", "n holdout"],
                       [[x["unit"], x["cell"], f(x["pooled"]), str(x["n"]), f(x["is"]), f(x["ho"]), str(x["n_ho"])] for x in rc]) + "\n")
    else:
        L.append("None met the registered-in-the-report screen.\n")

    # ---- hypotheses
    L.append("## Hypotheses (recorded in research/hypotheses_results.jsonl)\n")
    rows = []
    for h, v in out["hypotheses"].items():
        extra = {k: x for k, x in v.items() if k not in ("verdict", "detail", "delta", "holdout", "hits")}
        rows.append([h, f"**{v['verdict']}**", ", ".join(f"{k}={x}" for k, x in extra.items())[:220]])
    L.append(table(["id", "verdict", "summary"], rows) + "\n")

    if cv.exists():
        L.append("## Caveats and what went wrong\n")
        L.append(cv.read_text(encoding="utf-8").strip() + "\n")
    L.append(METHOD)
    path.write_text("\n".join(L), encoding="utf-8")


METHOD = """## Method

- **Units.** 17 registered strategies at their configured frequency (1h; geo_* on 4h) plus `zoo_ensemble` (4h), through one generic trade
  layer; desk setups through a bracket layer: `trend_4h` (agent.session 4h ema_stack sleeve), `shock_desk` (agent.session shock candidates,
  market entry), `hs2_fade_blackout` (reversal leg, entry one 1h bar after the shock = the R-FLOW-2 60-minute blackout), `hrb2`
  (post_cascade_study H-RB2, re-implemented with parameters), `setup_c_proxy` (setup C without its spot-absorption and Coinbase-premium
  confirmations, which have no history). Weekend MM quoters: replay results only.
- **Trade layer.** Enter at the bar close when flat and |s| >= 0.3; exit at the first close with s x side <= 0; exchange stop 4.0 x
  sigma_risk_next per bar (the applied stop cap) from the fill, walked on 1h bars; no re-entry after a stop until the signal is flat or
  flips. R = (d (exit - entry) - c_in entry - c_out exit) / (|entry - stop| + entry (c_in + c_out at base costs)). Base costs = live: entry
  pegged limit 2.0 + 1.3 bp (shock units market 5.5 + 2.0), exits taker 5.5 + 2.0 bp, stops filled at the stop price.
- **Scenarios.** S1 slices of base trades; S2 crisis windows by entry date on the raw stream and on the S9 stream; S3 fees x1.5 / x2, +5 /
  +10 / +25 bp per side, gap fills at the open, stop filled at the next bar open, no exchange stop; S4 1 / 5 / 15 minute delays on entries and
  signal exits, filled at the Brownian-bridge conditional mean inside the next 1h bar (checked against BTC 5m bars above); S5 x0.5 / 0.8 /
  1.2 / 1.5 one at a time on every key parameter plus theta_in and stop_k, and 30 joint U(0.5, 1.5) draws, on D8; S6 1000 one-year 1h paths
  per unit (500 stationary block bootstrap with Politis-White block length on |r|, 500 regime-switching on the v0 cell chain; sources
  BTC ETH SOL XRP DOGE LINK); S7 in BTC down states (v0 R7 / R8 / R9) a stop-out closes every concurrent same-side trade of the unit at its
  stop; S8 mid / illiquid tiers (point-in-time trailing 30-day median turnover < 100M USD) pay spread x3 and stop slippage sqrt(2) x the
  tier's stop_slip_bps; S9 live_small applied to the pooled stream, first come first served, ties broken by universe order.
- **Statistics.** Mean R per trade with a stationary bootstrap of entry days (mean block 5 days, 2000 draws), which keeps the cross-symbol
  clustering of same-day trades; max drawdown on the realised curve in exit order; CVaR5% = mean of the worst 5% trades / active days;
  time under water on the daily realised curve. Segments split by entry date at research.knowledge_cutoff (2026-07-01).
- **Verdicts (frozen).** FAILS: base mean <= 0 or fees x1.5 mean <= 0. FRAGILE: +-20% parameter cliff (D8), 5-minute latency mean <= 0,
  a crisis window < -20 R on the engine stream, +5 bp mean <= 0, clean_holdout mean <= 0 with >= 20 trades, base CI includes 0, or the total
  without the 5 best trades <= 0. Else SURVIVES. Other scenarios are reported, not part of the verdict.
- **Reproduce.** `uv run python -m ai_trader.research.stress design | run | perturb | synth | deribit | latcheck | analyse [--record]`;
  outputs in data/research/stress/. A new strategy: `ai_trader.research.stress.stress_signal(fn, symbols, freq)`.
"""
