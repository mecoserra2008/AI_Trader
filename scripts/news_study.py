"""H-NEWS1 / H-NEWS2 for BTC (agent F3): builds the point-in-time feature tables, runs the pre-registered tests and
writes research/reports/btc_news_<date>.{json,md}. Design: src/ai_trader/news/study.py (fixed in the diary first).

    PYTHONIOENCODING=utf-8 uv run python scripts/news_study.py [--as-of 2026-11-13] [--record]

--record appends one result per hypothesis and segment to research/hypotheses_results.jsonl, skipping any
(hypothesis, segment) that already has a result.
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, date, datetime, timedelta

import numpy as np
import polars as pl

from ai_trader.core.config import find_root
from ai_trader.news import events, features, study
from ai_trader.news.store import Manifest

SEGS = ("llm_in_sample", "clean_holdout")


def coverage(man: Manifest) -> dict:
    out = {}
    for k in man.keys():
        e = man.get(k)
        parts = k.split("/")
        grp = "/".join(parts[:3]) if parts[0] == "gdelt" else "/".join(parts[:2])
        g = out.setdefault(grp, {"chunks": 0, "rows": 0, "first": None, "last": None, "status": {}})
        g["chunks"] += 1
        g["rows"] += e.get("rows", 0)
        g["status"][e.get("status")] = g["status"].get(e.get("status"), 0) + 1
        for a, b in (("first", "first"), ("last", "last")):
            v = e.get(b)
            if v:
                g[a] = v if g[a] is None or (v < g[a] if a == "first" else v > g[a]) else g[a]
    out["_bytes"] = man.size_bytes()
    return out


def fmt(x, nd=3):
    return "n/a" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.{nd}f}" if isinstance(x, float) else str(x)


def _ci(x: dict) -> str:
    c = x.get("ci95")
    return f"[{c[0]:+.4f}, {c[1]:+.4f}]" if c else "n/a"


def rule_candidates(h1: dict, h2: dict) -> list[dict]:
    p = h1["variants"]["primary"]
    v1 = {s: p[s]["verdict"] for s in SEGS}
    rules = []
    if all(v == "supports" for v in v1.values()):
        rules.append({"id": "R-NEWS-1", "when": "next_day_vol for BTC", "rule": "add the news / event / on-chain block (news.features.BLOCK) to log-HAR x weekday",
                      "evidence": "H-NEWS1 supports in both segments", "confidence": "B"})
    else:
        rules.append({"id": "R-NEWS-1", "when": "next_day_vol for BTC",
                      "rule": "keep log-HAR x weekday; do not add GDELT news intensity, tone, event flags or on-chain activity as daily vol regressors",
                      "evidence": f"H-NEWS1 in-sample {v1['llm_in_sample']} (QLIKE diff {fmt(p['llm_in_sample']['diff'], 4)}, NW t {fmt(p['llm_in_sample']['nw_t'], 2)}); "
                                  f"holdout {v1['clean_holdout']} (diff {fmt(p['clean_holdout']['diff'], 4)}, t {fmt(p['clean_holdout']['nw_t'], 2)})",
                      "confidence": "B" if v1["llm_in_sample"] == "contradicts" else "C"})
    ev = h1["variants"]["events_only"]
    c = h1["coefs_at_cutoff"].get("ev_macro_next", {})
    if c:
        rules.append({"id": "R-NEWS-2", "when": "a scheduled CPI / NFP / FOMC release falls on the forecast day",
                      "rule": f"read the vol forecast with the macro-day multiplier exp(b) = {np.exp(c['coef']):.2f} from the in-sample fit (OLS t {c['t_ols']:.1f}) as context, not as a regressor unless the events-only variant beats the baseline",
                      "evidence": f"events-only QLIKE diff in-sample {fmt(ev['llm_in_sample']['diff'], 4)} (t {fmt(ev['llm_in_sample']['nw_t'], 2)}), holdout {fmt(ev['clean_holdout']['diff'], 4)} (t {fmt(ev['clean_holdout']['nw_t'], 2)})",
                      "confidence": "C"})
    s2 = h2["segments"]
    rules.append({"id": "R-NEWS-3", "when": "any BTC direction call",
                  "rule": "never use news volume, tone, event flags or on-chain activity as a directional signal",
                  "evidence": f"H-NEWS2 Brier skill in-sample {fmt(s2['llm_in_sample'].get('skill'), 4)} CI {_ci(s2['llm_in_sample'])}, "
                              f"holdout {fmt(s2['clean_holdout'].get('skill'), 4)} CI {_ci(s2['clean_holdout'])}",
                  "confidence": "B"})
    rules.append({"id": "R-NEWS-4", "when": "reading GDELT counts after 2026-09",
                  "rule": "use log share of monitored volume and its trailing z, never raw article counts; GDELT's monitored total fell about 7x between July and November 2026",
                  "evidence": "manifest: bitcoin norm monthly mean 146k (2026-07) to 21k (2026-11)", "confidence": "A"})
    return rules


def to_yaml(rules: list[dict]) -> str:
    lines = ["rules:"]
    for r in rules:
        lines.append(f"  - id: {r['id']}")
        for k in ("when", "rule", "evidence", "confidence"):
            v = str(r[k]).replace('"', "'")
            lines.append(f'    {k}: "{v}"')
    return "\n".join(lines)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--as-of", default=None)
    ap.add_argument("--record", action="store_true")
    ap.add_argument("--allow-partial", action="store_true", help="record even if some GDELT daily query/mode is missing")
    a = ap.parse_args(argv)
    as_of = datetime.fromisoformat(a.as_of).replace(tzinfo=UTC) if a.as_of else datetime.now(UTC).replace(hour=0, minute=0, second=0, microsecond=0)
    root = find_root()
    from ai_trader.core.config import load

    cutoff = date.fromisoformat(str(load().settings["research"]["knowledge_cutoff"]))
    man = Manifest()
    ev = events.load_or_build(man)
    rvd = study.btc_daily(as_of)
    start = datetime.combine(rvd["date"].min(), datetime.min.time(), tzinfo=UTC)
    d, h = features.build_all(man, ev, start, as_of)
    df = study.frame(rvd, d)
    h1 = study.run_h1(df, cutoff)
    h2 = study.run_h2(df, cutoff)
    fc = h1.pop("_forecasts")
    real = h1.pop("_realised")
    # per-day series for the dashboard / audits
    pl.DataFrame({"date": df["date"], "rv_next": real, **{f"f_{k}": v for k, v in fc.items()}}).write_parquet(man.root / "features" / "h1_forecasts.parquet")
    cov = coverage(man)
    from ai_trader.news import gdelt as G

    missing = [f"{q}/{m}" for q in G.QUERIES for m in G.MODES if not any(man.get(k).get("status") == "ok" and man.get(k).get("kind", "day") == "day"
                                                                       for k in man.keys(f"gdelt/{m}/{q}/"))]
    nz = d.select([pl.col(c).is_not_null().mean().alias(c) for c in features.BLOCK]).row(0, named=True)
    cut_ts = datetime.combine(cutoff, datetime.min.time(), tzinfo=UTC)
    seg_null = {s: d.filter((pl.col("asof") >= start + timedelta(days=study.MIN_DAYS)) & ((pl.col("asof") >= cut_ts) if s == "clean_holdout" else (pl.col("asof") < cut_ts)))
                .select(pl.col("att_btc").is_null().mean()).item() for s in SEGS}
    rules = rule_candidates(h1, h2)
    stamp = as_of.date().isoformat()
    rep = {"as_of": as_of.isoformat(), "cutoff": cutoff.isoformat(), "symbol": "BTCUSDT", "n_days": df.height,
           "date_range": [str(df["date"].min()), str(df["date"].max())], "coverage": cov,
           "status": "provisional (GDELT daily missing: " + ", ".join(missing) + ")" if missing else "final", "gdelt_missing": missing, "block_nonnull_share": nz, "att_btc_null_share_by_segment": seg_null,
           "hourly_table": {"rows": h.height, "cols": h.columns}, "H-NEWS1": h1, "H-NEWS2": h2, "rules": rules}
    out = root / "research" / "reports" / f"btc_news_{stamp}"
    out.with_suffix(".json").write_text(json.dumps(rep, indent=1, default=str), encoding="utf-8")

    p = h1["variants"]
    L = [f"# BTC news, events and on-chain vs next-day volatility ({stamp})", "",
         "Agent F3, BTC field program. Pre-registered H-NEWS1 / H-NEWS2 (research/hypotheses.yaml), design fixed in the diary before the run "
         "(src/ai_trader/news/study.py docstring). Point-in-time: every feature component carries known_ts <= its use time (tests/test_news.py).", "",
         (f"**Status: PROVISIONAL, not recorded.** GDELT daily series missing ({len(missing)} of {len(G.QUERIES) * len(G.MODES)}): {', '.join(missing)}. "
          "The GDELT API returned HTTP 429 / timeouts continuously from 2026-11-13 21:31 UTC; the missing columns enter the block as zeros "
          "(att_ind, tone_ind empty). Re-run `scripts/news_fetch.py gdelt-daily` then `scripts/news_study.py --record` to finalise." if missing else "**Status: final.**"), "",
         "## Data", "",
         "| source | coverage | notes |", "|---|---|---|"]
    for k, g in cov.items():
        if k.startswith("_"):
            continue
        L.append(f"| {k} | {g['first']} .. {g['last']} | {g['chunks']} chunks, {g['rows']} rows, {g['status']} |")
    L += ["", f"Store size: {cov['_bytes'] / 1e6:.1f} MB. BTC days {df['date'].min()} .. {df['date'].max()}.", "",
          "GDELT's monitored total (`norm`) is not stationary (bitcoin-query monthly mean ~0.7 M articles/day in 2017, ~0.15 M in 2023-26, ~0.08 M Oct 2026, "
          "~0.02 M Nov 2026), so all news features are log shares relative to a trailing 60-day mean. GDELT days missing (feature set to 0): "
          f"in-sample {100 * seg_null['llm_in_sample']:.1f}%, holdout {100 * seg_null['clean_holdout']:.1f}% of decision days.", "",
          "## H-NEWS1: next-day QLIKE, augmented minus log-HAR x weekday (negative = news helps)", "",
          "| variant | segment | n | QLIKE base | QLIKE aug | diff | rel | NW t |", "|---|---|---|---|---|---|---|---|"]
    for name, r in p.items():
        for s in SEGS:
            x = r[s]
            L.append(f"| {name} | {s} | {x['n']} | {fmt(x['qlike_base'], 4)} | {fmt(x['qlike_aug'], 4)} | {fmt(x['diff'], 4)} | {fmt(x['rel'], 3)} | {fmt(x['nw_t'], 2)} |")
    L += ["", f"Verdicts (primary): in-sample **{p['primary']['llm_in_sample']['verdict']}**, holdout **{p['primary']['clean_holdout']['verdict']}**.", "",
          "Block coefficients (OLS on ln RV_{t+1}, fit on in-sample days only; t is plain OLS, descriptive):", "",
          "| term | coef | t |", "|---|---|---|"]
    for k, v in h1["coefs_at_cutoff"].items():
        if k in features.BLOCK:
            L.append(f"| {k} | {v['coef']:+.4f} | {v['t_ols']:+.2f} |")
    s2 = h2["segments"]
    L += ["", "## H-NEWS2: direction (null)", "", "| segment | n | Brier model | Brier base | skill | 95% CI | hit rate | verdict |", "|---|---|---|---|---|---|---|---|"]
    for s in SEGS:
        x = s2[s]
        ci = x.get("ci95") or [float("nan")] * 2
        L.append(f"| {s} | {x['n']} | {fmt(x.get('brier_model'), 4)} | {fmt(x.get('brier_base'), 4)} | {fmt(x.get('skill'), 4)} | [{ci[0]:+.4f}, {ci[1]:+.4f}] | {fmt(x.get('hit_rate'), 3)} | **{x['verdict']}** |")
    L += ["", "## Rule candidates", "", "```yaml", to_yaml(rules), "```", "",
          "## Caveats", "",
          "- GDELT timelines fetched on 2026-11-13 are today's vintage (late-crawled articles back-filled); only increments fetched later keep a first-seen vintage.",
          "- CPI / NFP dates 2020-2025 were entered by hand (FRED unreachable, BLS refuses scripted access); 2026 dates and all FOMC dates come from fetched pages.",
          "- The holdout has ~4 months and, inside it, GDELT's coverage collapses; the holdout news features are noisier than in-sample.",
          "- No scheduled crypto-regulatory decision is recorded for 2025-2026; the crypto event flag is effectively in-sample only.",
          "- RSS headline counts start 2026-11-13 (forward only); they are in the hourly table, not in the test."]
    out.with_suffix(".md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print(out.with_suffix(".md").read_text(encoding="utf-8"))

    if a.record and missing and not a.allow_partial:
        print(f"NOT recorded: GDELT daily data incomplete ({len(missing)} missing). Finish the fetch or pass --allow-partial.")
    elif a.record:
        from ai_trader.research import hypotheses as H

        have = {(r["id"], r["segment"]) for r in H.results() if r["id"] in ("H-NEWS1", "H-NEWS2")}
        rel = f"research/reports/btc_news_{stamp}.json"
        for s in SEGS:
            x = p["primary"][s]
            if ("H-NEWS1", s) not in have:
                H.record("H-NEWS1", s, x["verdict"], {"qlike_diff": x["diff"], "nw_t": x["nw_t"], "rel": x["rel"], "n": x["n"],
                                                      "news_l1_t": p["news_l1"][s]["nw_t"], "events_only_t": p["events_only"][s]["nw_t"],
                                                      "onchain_only_t": p["onchain_only"][s]["nw_t"], "news_only_t": p["news_only"][s]["nw_t"]},
                         note="log-HAR x weekday + 13-col news/event/on-chain block, expanding OLS, refit 30 d", data=rel)
                print("recorded H-NEWS1", s, x["verdict"])
            y = s2[s]
            if ("H-NEWS2", s) not in have:
                H.record("H-NEWS2", s, y["verdict"], {"skill": y.get("skill"), "ci95": y.get("ci95"), "n": y["n"], "hit_rate": y.get("hit_rate")},
                         note="walk-forward L2 logistic on the news block, Brier skill vs training base rate, 10-day block bootstrap", data=rel)
                print("recorded H-NEWS2", s, y["verdict"])


if __name__ == "__main__":
    main()
