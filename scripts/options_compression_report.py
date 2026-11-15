"""Writes research/reports/options_compression_2026-11-14.md from the study JSON plus a narrative file.

    uv run python scripts/options_compression_report.py <narrative.json>

narrative.json: {"summary": md, "opx1": md, "opx2": md, "greeks": md, "opx3": md, "opx4": md, "plan": md, "caveats": md,
                 "rules_yaml": "<yaml text of rule candidates>"}
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "research" / "reports" / "options_compression_2026-11-14"


def f(x, nd=3, pct=False):
    if x is None:
        return "n/a"
    if isinstance(x, str):
        return x
    if isinstance(x, bool):
        return str(x)
    if isinstance(x, int):
        return str(x)
    return f"{x * (100 if pct else 1):.{nd}f}"


def ci(c, nd=3, pct=False):
    return f"[{f(c[0], nd, pct)}, {f(c[1], nd, pct)}]" if c else "n/a"


def table(head, rows):
    out = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def build(o: dict, n: dict) -> str:
    d = o["data"]
    me = o["model_error"]
    L = ["# Options after a volatility compression, the user's call + short deep-ITM put, and extreme-profit continuation",
         "",
         "_Agent OPX, 2026-11-14. Hypotheses H-OPX1..4 (registered before any data). Research only: no orders anywhere. "
         "Code: `src/ai_trader/research/options_compression.py`, `options_download.py`, `scripts/options_compression_study.py`; "
         "tests `tests/test_options_compression.py`; design frozen in the diary (2026-11-14 16:57 UTC) before any result._",
         "", "## In short", "", n["summary"], "",
         "## Data", "",
         f"- Deribit BTC option trades from the public history API, {d['first']}..{d['last']} ({d['months_complete']} complete months"
         f"{', partial ' + ', '.join(d['partial_months']) if d['partial_months'] else ''}): **{d['trades']:,} trades**, aggregated per "
         f"instrument-hour into {d['rows']:,} rows, **{d['mb_on_disk']} MB** on disk (`data/research/options/`, manifest + one parquet per month; "
         "raw responses never written). Block trades excluded.",
         "- Lake: Bybit BTCUSDT / ETHUSDT 1h bars and funding, the deribit_svi daily smile summary (ATM / RR25 / BF25 at 7/30/90 days, to "
         "2026-10-09) and DVOL, read point-in-time.",
         f"- Compression events: {len(o['events'])} after declustering ({o['trigger_days']} trigger days). Entry days with a full trade: "
         + ", ".join(f"{k} {v}" for k, v in o["n_entry_days"].items()) + ".",
         "", "## Pricing and the fallback model", "",
         "Each option leg at a 08:00 fixing is priced from a real trade when the instrument traded between 05:00 and 11:00 (its trade IV, "
         "moved to the 08:00 forward with Black-76), otherwise by Black-76 on a 3-point smile rebuilt from ATM / RR25 / BF25 (quadratic in "
         "log-moneyness through the 25-delta put, ATM and 25-delta call, straight lines outside, ATM variance interpolated linearly in time).",
         "", "Model error against every real 08:00 quote (7-120 days to expiry, |delta| 0.05-0.95):", "",
         table(["subset", "n", "bias (vol pts)", "median abs (vol pts)", "RMSE (vol pts)", "median abs price error", "median abs (BTC)"],
               [[k.replace("|d|", "abs delta "), v.get("n"), f(v.get("bias_vol_pts"), 2), f(v.get("median_abs_vol_pts"), 2), f(v.get("rmse_vol_pts"), 2),
                 f(v.get("median_abs_price_err_pct"), 1) + "%", f(v.get("median_abs_err_btc"), 4)]
                for k, v in [("all", me["all"]), ("strikes like the short put", me["strikes_like_the_short_put"])]
                + sorted(me["by_bucket"].items()) + sorted(me["by_tenor"].items()) + sorted(me["by_year"].items())]),
         "", f"![model error](../figures/options_compression_2026-11-14/model_error.png)", ""]
    # ---- H-OPX1
    L += ["## H-OPX1: does long volatility pay after a compression?", "", n["opx1"], ""]
    rows = []
    for seg in ("llm_in_sample", "clean_holdout"):
        r = o["H-OPX1"][seg]
        for v in ("hedged", "unhedged", "btc_hedged", "btc_unhedged"):
            b = r[v]
            rows.append([seg, v.replace("btc_", "BTC-settled, "), b.get("n_event"), b.get("n_all"), f(b.get("mean_event"), 3), f(b.get("mean_all"), 3),
                         f(b.get("diff"), 3), ci(b.get("ci95"))])
    L += [table(["segment", "variant", "events", "all days", "mean (events)", "mean (all)", "difference", "95% CI"], rows), "",
          "Units: net P&L / premium paid (1.0 = doubled the premium). " +
          " ".join(f"{seg}: verdict **{o['H-OPX1'][seg]['verdict']}**; median realised/implied vol ratio events {f(o['H-OPX1'][seg]['rv_iv_event_median'], 2)} vs all "
                   f"{f(o['H-OPX1'][seg]['rv_iv_all_median'], 2)}; entry legs priced from real trades {f(o['H-OPX1'][seg]['share_entry_legs_real'], 2)}."
                   for seg in ("llm_in_sample", "clean_holdout")),
          "", "![straddle](../figures/options_compression_2026-11-14/h_opx1_straddle.png)", ""]
    # ---- Greeks
    g = o["greeks"]
    L += ["## What the short deep-ITM put actually does (Greeks at entry)", "", n["greeks"], ""]
    for name in ("events", "all_days"):
        if name not in g:
            continue
        x = g[name]
        L += [f"**{name.replace('_', ' ')}** (n = {x['n']}; mean call delta {f(x['mean_call_delta'], 2)}, mean put delta {f(x['mean_put_delta'], 2)}):", "",
              table(["structure", "net delta", "vega (% of spot per vol pt)", "theta (% of spot per day)", "net premium (% of spot)", "vega per unit delta"],
                    [[k.replace("_", " "), f(x[k]["net_delta"], 3), f(x[k]["vega_pct_S_per_volpt"], 4), f(x[k]["theta_pct_S_per_day"], 4),
                      f(x[k]["net_premium_pct_S"], 2), f(x[k]["vega_per_unit_delta_pct_S"], 4)]
                     for k in ("call_alone", "call_plus_perp", "user_structure", "short_put_alone")]), "",
              f"Short put: premium {f(x['short_put_premium_pct_S'], 2)}% of spot, of which time value {f(x['short_put_time_value_pct_S'], 2)}% "
              f"(time-value share mean {f(x['short_put_time_value_share']['mean'], 3)}, median {f(x['short_put_time_value_share']['median'], 3)}); "
              f"the rest is intrinsic value you pay back. Put vega / call vega (median) {f(x['put_vega_over_call_vega_median'], 2)}.", ""]
    L += ["![greeks](../figures/options_compression_2026-11-14/greeks.png)", ""]
    # ---- H-OPX2
    L += ["## H-OPX2: short deep-ITM put vs a perp with the same delta", "", n["opx2"], ""]
    rows = []
    for seg in ("llm_in_sample", "clean_holdout"):
        r = o["H-OPX2"][seg]
        for k in ("events_14d", "all_days_14d", "events_to_expiry", "all_days_to_expiry"):
            b = r.get(k) or {}
            rows.append([seg, k.replace("_", " "), b.get("n"), f(b.get("mean"), 3), ci(b.get("ci95")), f(b.get("median"), 3), f(b.get("share_pos"), 2)])
    L += [table(["segment", "sample", "n", "mean diff (% of spot)", "95% CI", "median", "share > 0"], rows), "",
          " ".join(f"{seg}: verdict **{o['H-OPX2'][seg]['verdict']}**." for seg in ("llm_in_sample", "clean_holdout")), ""]
    for seg in ("llm_in_sample", "clean_holdout"):
        lg = o["H-OPX2"][seg].get("legs_events")
        if lg:
            L += [f"Leg means at events, {seg} (% of spot, 14-day hold): structure {f(lg['struct_pct'], 2)}, call + perp {f(lg['comp_pct'], 2)}, "
                  f"call {f(lg['call_pct'], 2)}, short put {f(lg['shortput_pct'], 2)}, perp {f(lg['perp_pct'], 2)} (of which funding paid {f(lg['perp_fund_pct'], 2)}, "
                  f"costs {f(lg['perp_cost_pct'], 2)}), put costs {f(lg['put_cost_pct'], 2)}; put priced from real trades at entry {f(lg['share_put_real_entry'], 2)}, "
                  f"at exit {f(lg['share_put_real_exit'], 2)}.", ""]
    bu = o["btc_vs_usd"]
    L += ["### BTC-settled vs USD P&L (Deribit options are inverse: premium and P&L are in BTC)", "",
          table(["sample", "straddle unhedged USD", "straddle unhedged BTC", "straddle hedged USD", "straddle hedged BTC",
                 "user structure USD (% spot)", "user structure BTC-settled, in USD (% spot)"],
                [[k, f(v["straddle_unhedged_usd_pct_prem"], 1) + "%", f(v["straddle_unhedged_btc_pct_prem"], 1) + "%", f(v["straddle_hedged_usd_pct_prem"], 1) + "%",
                  f(v["straddle_hedged_btc_pct_prem"], 1) + "%", f(v["user_struct_usd_pct_S"], 2), f(v["user_struct_btc_settled_in_usd_pct_S"], 2)] for k, v in bu.items()]),
          "", "Straddle columns are % of premium (USD premium for USD, BTC premium for BTC). The BTC-settled number keeps the premium and the "
          "P&L in BTC: USD P&L - BTC-settled P&L x S_exit = (net BTC premium) x (S_exit - S_entry), i.e. paying a premium in BTC adds a "
          "long-BTC exposure equal to it, and receiving the short put's large premium in BTC adds a short-BTC exposure of the same size.", ""]
    # ---- H-OPX3
    L += ["## H-OPX3: are extreme unexpected moves followed by more of the same?", "", n["opx3"], ""]
    for sym in ("BTCUSDT", "ETHUSDT"):
        rows = []
        for H in ("1", "3", "7"):
            for seg in ("llm_in_sample", "clean_holdout"):
                t = o["H-OPX3"][sym][H][seg]
                for q in ("50", "75", "90", "95", "99"):
                    b = t["by_q"][q]
                    rows.append([H, seg, f"q{q}", b["n"], f(b["mean_z"], 3), f(b["t_cluster"], 2), f(b["t_nw"], 2), f(b["mean_ret_pct"], 2),
                                 f(b["hit"], 2), f(b["mean_z_up"], 3), f(b["mean_z_down"], 3)])
                rows.append([H, seg, "slope", "", f(t["slope"], 4), ci(t["slope_ci90"], 4), "", "", "", "", f"call: {t['call']}"])
        L += [f"**{sym}** (verdict llm_in_sample **{o['H-OPX3'][sym]['verdict_llm_in_sample']}**, clean_holdout "
              f"**{o['H-OPX3'][sym]['verdict_clean_holdout']}**{'; ETH is robustness only, not recorded separately' if sym == 'ETHUSDT' else ''}):", "",
              table(["H (days)", "segment", "beyond", "n", "mean next-H z (same dir.)", "cluster t", "NW t", "mean next-H move (%)", "share > 0",
                     "z after up moves", "z after down moves"], rows), ""]
        cal = {H: o["H-OPX3"][sym][H]["calibration_two_sided_exceedance"] for H in ("1", "3", "7")}
        L += ["Calibration of the ex-ante distribution (share of origins beyond each two-sided percentile; ideal = 2 x (1 - q)): "
              + "; ".join(f"H={H}: " + ", ".join(f"q{q} {f(v, 3)}" for q, v in c.items()) for H, c in cal.items()), ""]
    L += ["![tails](../figures/options_compression_2026-11-14/h_opx3_tail_continuation.png)", ""]
    # ---- H-OPX4
    t4 = o["H-OPX4"]
    L += ["## H-OPX4: scaling out in thirds with limit orders after an extreme profit", "", n["opx4"], ""]
    rows = []
    for seg in ("llm_in_sample", "clean_holdout"):
        r = t4["trend"][seg]
        rows.append([seg, r["n_trades"], f(r["mean_R0"], 4), f(r["mean_R1"], 4), f(r["diff"]["mean"], 4), ci(r["diff"]["ci95"], 4),
                     f(r["share_triggered"], 3), f(r["mean_fills_when_triggered"], 2),
                     f((r.get("diff_when_triggered") or {}).get("mean"), 4), r["verdict"]])
    L += ["Trend book (23 crypto perps x 8 trend strategies, own exits, net R per trade in daily risk units):", "",
          table(["segment", "trades", "own exit R", "scaled R", "diff", "95% CI", "share triggered", "fills per triggered trade", "diff when triggered",
                 "verdict"], rows), ""]
    rows = []
    for seg in ("llm_in_sample", "clean_holdout"):
        for k, b in (t4["options"].get(seg) or {}).items():
            rows.append([seg, k.replace("_", " "), b["n"], f(b["mean_R0"], 3), f(b["mean_R1"], 3), f(b["mean"], 3), ci(b["ci95"]),
                         f(b["share_triggered"], 2), f(b["mean_leg_fills_when_triggered"], 2)])
    L += ["Option trades (net P&L in % of spot, 14-day hold vs thirds at the first three daily marks beyond the ex-ante q95 P&L; each leg's "
          "third fills only if a real trade prints through the mark within 24 h):", "",
          table(["segment", "book", "n", "hold", "scaled", "diff", "95% CI", "share triggered", "leg fills when triggered"], rows), ""]
    bs = t4["trend"]["llm_in_sample"]["by_strategy"]
    L += ["Trend book by strategy (llm_in_sample): " + "; ".join(f"{k} n={v['n']}, diff {f(v['diff_mean'], 4)}, trig {f(v['share_trig'], 2)}"
                                                         for k, v in sorted(bs.items())), ""]
    # ---- plan
    dp = o["downside_plan"]
    s = dp["summary"]
    L += ["## The downside plan (DESCRIPTIVE ONLY, not a registered hypothesis)", "", n["plan"], ""]
    if dp["events"]:
        L += [table(["entry", "segment", "day-1 move %", "14-day move %", "structure", "protection held", "protection with plan",
                     "structure + plan", "spot hedge day", "fills"],
                    [[e["entry"], e["seg"], f(e["first_day_ret_pct"], 2), f(e["ret_14d_pct"], 2), f(e["struct_pct"], 2), f(e["prot_hold_pct"], 2),
                      f(e["prot_plan_pct"], 2), f(e["struct_plus_plan_pct"], 2), e["spot_hedge_day"] if e["spot_hedge_day"] is not None else "-", e["n_fills"]]
                     for e in dp["events"]]), "",
              f"All in % of spot at entry. Mean structure {f(s.get('mean_struct_pct'), 2)} -> with plan {f(s.get('mean_struct_plus_plan_pct'), 2)}; "
              f"worst {f(s.get('worst_struct_pct'), 2)} -> {f(s.get('worst_struct_plus_plan_pct'), 2)}; best {f(s.get('best_struct_pct'), 2)} -> "
              f"{f(s.get('best_struct_plus_plan_pct'), 2)}; the spot hedge triggered on {s.get('n_spot_hedged')} of {s.get('n')} events.", ""]
    L += ["![detector](../figures/options_compression_2026-11-14/detector.png)", "",
          "## Caveats", "", n["caveats"], "",
          "## Rule candidates", "", "```yaml", n["rules_yaml"].strip(), "```", ""]
    return "\n".join(L)


if __name__ == "__main__":
    o = json.loads(REPORT.with_suffix(".json").read_text(encoding="utf-8"))
    n = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    REPORT.with_suffix(".md").write_text(build(o, n), encoding="utf-8")
    print(REPORT.with_suffix(".md"))
