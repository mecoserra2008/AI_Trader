"""H-NEWS3 / H-NEWS4 for BTC (agent W): non-GDELT news attention (Wikipedia pageviews + edits, Hacker News, Fear and
Greed) + the event calendar + on-chain block vs log-HAR x weekday, and the direction null. Writes
research/reports/btc_news_alt_<date>.{json,md}.

    PYTHONIOENCODING=utf-8 uv run python scripts/news_alt_study.py [--as-of 2026-11-13] [--pilot] [--record]

Design (fixed before the run; mechanics of study.py, which ran H-NEWS1/2):
  decision  asof = t + 1 00:00 UTC (end of day t), target = ln RV of day t + 1; segment by the target day.
  block     H-NEWS3 primary = news.alt.ALT_COLS (8) + features.EVENT_COLS (3) + features.ONCHAIN_COLS (4) = 15 columns,
            each taken from the latest source day with known_ts <= asof (strict: HN / edits of day t are known at
            t + 1 00:30, so the 00:00 decision uses day t - 1; pageviews day t - 1; Fear and Greed of day t + 1, stamped
            00:00). Missing values -> 0 (= no abnormal attention). Guardian refused the test key -> not in the block.
  H-NEWS3   expanding OLS, refit every 30 d, first fit after 365 d; paired QLIKE (aug - base), NW t; supports if
            diff < 0 and t < -2; contradicts if diff >= 0; else inconclusive; per segment.
  H-NEWS4   walk-forward L2 logistic (C = 1) on the standardised 15-column block; Brier skill vs the training up-share;
            95% circular 10-day block bootstrap; supports if the CI upper bound <= 0.005, contradicts if the lower
            bound > 0.005, else inconclusive.
  robustness (reported, not recorded): alt_only, wiki_only (pageviews + edits), hn_only, fng_only, events_onchain.
  pilot     --pilot runs the same pipeline on target days < 2022-06-01 (about one year out of sample after the
            365-day warm-up), prints it and records nothing.
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, date, datetime, timedelta

import numpy as np
import polars as pl

from ai_trader.core.config import find_root, load
from ai_trader.news import alt, events, features, study
from ai_trader.news.store import Manifest

SEGS = ("llm_in_sample", "clean_holdout")
BLOCK = list(alt.ALT_COLS + features.EVENT_COLS + features.ONCHAIN_COLS)
VARIANTS = {"alt_only": list(alt.ALT_COLS), "wiki_only": ["wp_btc", "wp_ind", "wp_max", "wp_edits"], "hn_only": ["hn_story", "hn_comment"],
            "fng_only": ["fng", "fng_chg"], "events_onchain": list(features.EVENT_COLS + features.ONCHAIN_COLS)}
PILOT_END = date(2022, 6, 1)


def fmt(x, nd=3):
    return "n/a" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.{nd}f}" if isinstance(x, float) else str(x)


def coverage(am: Manifest) -> dict:
    out: dict = {}
    for k in am.keys():
        e = am.get(k)
        src, *rest = k.split("/")
        ser = rest[0] if rest else ""
        g = out.setdefault(src, {}).setdefault(ser, {"chunks": 0, "rows": 0, "first": None, "last": None, "status": e.get("status")})
        g["chunks"] += 1
        g["rows"] += e.get("rows", 0)
        for a in ("first", "last"):
            v = e.get(a)
            if v and (g[a] is None or (v < g[a] if a == "first" else v > g[a])):
                g[a] = v
        for m in ("requests", "approx_days", "items", "edits"):
            if m in e:
                g[m] = g.get(m, 0) + e[m]
    out["_bytes"] = am.size_bytes()
    return out


def build_frame(as_of: datetime) -> tuple[pl.DataFrame, pl.DataFrame]:
    man = Manifest()
    ev = events.load_or_build(man)
    rvd = study.btc_daily(as_of)
    start = datetime.combine(rvd["date"].min(), datetime.min.time(), tzinfo=UTC)
    d = features.daily_table(man, ev, start, as_of)
    features.assert_point_in_time(d)
    keep = ["asof", *[c for c in features.EVENT_COLS + features.ONCHAIN_COLS if c in d.columns], "kt_onchain", "kt_events"]
    d = d.select([c for c in keep if c in d.columns])
    a = alt.alt_table(d["asof"])
    alt.assert_point_in_time(a)
    feats = d.join(a, on="asof", how="left")
    return study.frame(rvd, feats), feats


def rule_candidates(h3: dict, h4: dict) -> list[dict]:
    p = h3["variants"]["primary"]
    v = {s: p[s]["verdict"] for s in SEGS}
    s4 = h4["segments"]
    ev3 = "H-NEWS3"
    rules = []
    if all(x == "supports" for x in v.values()):
        rules.append({"id": "R-NEWS-5", "kind": "prefer",
                      "rule": "WHEN forecasting next-day BTC volatility THEN prefer log-HAR x weekday augmented with the news-attention block (abnormal Wikipedia / HN attention, Fear and Greed, events, on-chain)",
                      "when": "next-day BTC vol forecast", "then": "use news.alt.daily_alt_features + events + on-chain as extra regressors",
                      "effect": f"QLIKE diff in-sample {fmt(p['llm_in_sample']['diff'], 4)} (t {fmt(p['llm_in_sample']['nw_t'], 2)}), holdout {fmt(p['clean_holdout']['diff'], 4)} (t {fmt(p['clean_holdout']['nw_t'], 2)})",
                      "confidence": "B", "status": "active"})
    else:
        rules.append({"id": "R-NEWS-5", "kind": "avoid",
                      "rule": "WHEN forecasting next-day BTC volatility THEN do not add daily news-attention counts (Wikipedia, Hacker News, Fear and Greed) or event / on-chain flags to log-HAR x weekday",
                      "when": "next-day BTC vol forecast", "then": "keep log-HAR x weekday; read attention spikes as context only",
                      "effect": f"QLIKE diff in-sample {fmt(p['llm_in_sample']['diff'], 4)} (NW t {fmt(p['llm_in_sample']['nw_t'], 2)}, {v['llm_in_sample']}); "
                                f"holdout {fmt(p['clean_holdout']['diff'], 4)} (t {fmt(p['clean_holdout']['nw_t'], 2)}, {v['clean_holdout']})",
                      "confidence": "B" if v["llm_in_sample"] == "contradicts" and v["clean_holdout"] != "supports" else "C",
                      "status": "active" if v["llm_in_sample"] == "contradicts" and v["clean_holdout"] != "supports" else "provisional"})
    # best robustness sub-block in-sample (reported only: a hint, never a rule above grade C)
    sub = {k: h3["variants"][k] for k in VARIANTS}
    best = min(sub, key=lambda k: sub[k]["llm_in_sample"]["nw_t"] if np.isfinite(sub[k]["llm_in_sample"]["nw_t"]) else 9)
    b = sub[best]
    if b["llm_in_sample"]["diff"] < 0 and b["llm_in_sample"]["nw_t"] < -2:
        rules.append({"id": "R-NEWS-6", "kind": "expect",
                      "rule": f"WHEN the {best.replace('_', ' ')} attention measures are abnormally high THEN expect next-day BTC volatility somewhat above the log-HAR x weekday forecast",
                      "when": f"news.alt {best} sub-block elevated", "then": "treat as a mild upward vol adjustment, context only",
                      "effect": f"sub-block QLIKE diff in-sample {fmt(b['llm_in_sample']['diff'], 4)} (t {fmt(b['llm_in_sample']['nw_t'], 2)}), holdout {fmt(b['clean_holdout']['diff'], 4)} (t {fmt(b['clean_holdout']['nw_t'], 2)}); robustness variant, 5 tried",
                      "confidence": "C", "status": "provisional"})
    vv = {s: s4[s]["verdict"] for s in SEGS}
    ci = {s: s4[s].get("ci95") or [float("nan")] * 2 for s in SEGS}
    rules.append({"id": "R-NEWS-7", "kind": "avoid",
                  "rule": "WHEN making a BTC direction call THEN do not use news attention, Fear and Greed level or change, event flags or on-chain activity as a directional signal",
                  "when": "any next-day BTC direction view", "then": "direction from these inputs is no better than the base rate",
                  "effect": f"H-NEWS4 Brier skill in-sample {fmt(s4['llm_in_sample'].get('skill'), 4)} CI [{ci['llm_in_sample'][0]:+.4f}, {ci['llm_in_sample'][1]:+.4f}] ({vv['llm_in_sample']}); "
                            f"holdout {fmt(s4['clean_holdout'].get('skill'), 4)} CI [{ci['clean_holdout'][0]:+.4f}, {ci['clean_holdout'][1]:+.4f}] ({vv['clean_holdout']})",
                  "confidence": "B" if all(x == "supports" for x in vv.values()) else "C",
                  "status": "active" if all(x == "supports" for x in vv.values()) else "provisional"})
    for r in rules:
        r["scope"] = {"assets": ["BTCUSDT"], "horizon": "1d", "sides": "both"}
        r["evidence"] = ["H-NEWS4"] if r["id"] == "R-NEWS-7" else [ev3]
        r["review"] = "re-run scripts/news_alt_study.py when the clean holdout has >= 250 days; flip if the holdout verdict changes"
    return rules


def to_yaml(rules: list[dict]) -> str:
    import yaml

    order = ("id", "kind", "rule", "when", "then", "scope", "effect", "confidence", "evidence", "review", "status")
    return yaml.safe_dump({"rules": [{k: r[k] for k in order} for r in rules]}, sort_keys=False, allow_unicode=True, width=200).rstrip()


def reading(h3: dict, h4: dict) -> str:
    """Plain-language summary generated from the numbers (so a re-run never leaves stale text)."""
    p = h3["variants"]["primary"]
    out = []
    for s, name in (("llm_in_sample", "Before the model's knowledge cutoff (llm_in_sample)"), ("clean_holdout", "After the cutoff (clean_holdout)")):
        x = p[s]
        if not x["n"]:
            out.append(f"- {name}: no forecast days.")
            continue
        better = x["diff"] < 0
        out.append(f"- {name}, {x['n']} days: adding the 15-column block made the volatility forecast "
                   f"{'better' if better else 'worse'} by {abs(100 * (x['rel'] or 0)):.1f}% of the baseline QLIKE (NW t {x['nw_t']:+.2f}); verdict **{x['verdict']}**.")
    subs = [(k, h3["variants"][k]["llm_in_sample"]) for k in VARIANTS]
    good = [f"{k} (t {v['nw_t']:+.2f})" for k, v in subs if v["n"] and v["diff"] < 0 and v["nw_t"] < -2]
    out.append("- Sub-blocks in-sample: " + (("these alone beat the baseline at |t| > 2: " + ", ".join(good) + " (a robustness check among 5 tried; not a recorded result).")
                                           if good else "none of the five sub-blocks beats the baseline at |t| > 2 on its own."))
    s4 = h4["segments"]
    d = []
    for s in SEGS:
        x = s4[s]
        if x.get("skill") is None:
            d.append(f"{s}: too few days")
            continue
        d.append(f"{s} Brier skill {x['skill']:+.4f} (95% CI {x['ci95'][0]:+.4f} .. {x['ci95'][1]:+.4f}, hit rate {100 * x['hit_rate']:.1f}%) -> {x['verdict']}")
    out.append("- Direction (H-NEWS4, registered as the null 'no skill'): " + "; ".join(d) + ". 'supports' here means the attention block does NOT predict direction.")
    return "\n".join(out)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--as-of", default=None)
    ap.add_argument("--pilot", action="store_true")
    ap.add_argument("--record", action="store_true")
    a = ap.parse_args(argv)
    as_of = datetime.fromisoformat(a.as_of).replace(tzinfo=UTC) if a.as_of else datetime.now(UTC).replace(hour=0, minute=0, second=0, microsecond=0)
    cutoff = date.fromisoformat(str(load().settings["research"]["knowledge_cutoff"]))
    df, feats = build_frame(as_of)
    if a.pilot:
        df = df.filter(pl.col("date") < PILOT_END)
    h3 = study.run_h1(df, cutoff, block=BLOCK, variants=VARIANTS)
    h4 = study.run_h2(df, cutoff, cols=BLOCK)
    h3.pop("_forecasts")
    h3.pop("_realised")
    cut_ts = datetime.combine(cutoff, datetime.min.time(), tzinfo=UTC)
    tgt = df.with_columns((pl.col("date") + pl.duration(days=1)).alias("_tgt")).filter(pl.col("_tgt") >= df["date"].min() + timedelta(days=study.MIN_DAYS))
    nonnull = {s: tgt.filter((pl.col("_tgt") >= cutoff) if s == "clean_holdout" else (pl.col("_tgt") < cutoff))
               .select([pl.col(c).is_not_null().mean().alias(c) for c in BLOCK]).row(0, named=True) for s in SEGS}
    desc = feats.filter(pl.col("asof") >= cut_ts).select(alt.ALT_COLS).describe() if feats.height else None
    if a.pilot:
        print(json.dumps({"pilot_days": df.height, "H-NEWS3": h3["variants"], "H-NEWS4": h4["segments"], "nonnull": nonnull}, indent=1, default=str))
        return
    am = alt.alt_manifest()
    cov = coverage(am)
    rules = rule_candidates(h3, h4)
    stamp = as_of.date().isoformat()
    root = find_root()
    out = root / "research" / "reports" / f"btc_news_alt_{stamp}"
    rep = {"as_of": as_of.isoformat(), "cutoff": cutoff.isoformat(), "symbol": "BTCUSDT", "n_days": df.height,
           "date_range": [str(df["date"].min()), str(df["date"].max())], "block": BLOCK, "coverage": cov, "block_nonnull_share": nonnull,
           "guardian": am.get("guardian/status"), "H-NEWS3": h3, "H-NEWS4": h4, "rules": rules}
    out.with_suffix(".json").write_text(json.dumps(rep, indent=1, default=str), encoding="utf-8")

    p = h3["variants"]
    s4 = h4["segments"]
    L = [f"# BTC news attention (non-GDELT) vs next-day volatility and direction ({stamp})", "",
         "Agent W. Pre-registered H-NEWS3 / H-NEWS4 (research/hypotheses.yaml). GDELT rate-limited the history (3 of 20 series), so these "
         "tests replace H-NEWS1/2 with sources whose history can be downloaded for free. Design: scripts/news_alt_study.py docstring; "
         "features: src/ai_trader/news/alt.py; point-in-time tests: tests/test_news_alt.py.", "",
         "## What was measured, in plain words", "",
         "Every day we ask: given what was public at midnight UTC, does knowing how much attention bitcoin and the crypto industry got "
         "(Wikipedia page views and edits, Hacker News stories and comments, the Fear and Greed index), plus scheduled macro events and "
         "on-chain activity, make tomorrow's BTC volatility forecast better than the standard model (log-HAR with a day-of-week effect)? "
         "And does it say anything about whether BTC goes up or down tomorrow?", "",
         "## Data obtained", "", "| source | series | days | first .. last | notes |", "|---|---|---|---|---|"]
    for src in ("wiki", "edits", "hn", "fng", "guardian"):
        for ser, g in sorted((cov.get(src) or {}).items()):
            extra = ", ".join(f"{m} {g[m]}" for m in ("requests", "approx_days", "items", "edits") if m in g)
            L.append(f"| {src} | {ser} | {g['rows']} | {g['first']} .. {g['last']} | {g['status']}{'; ' + extra if extra else ''} |")
    L += ["", f"Store: {cov['_bytes'] / 1e6:.1f} MB under data/research/news/alt/. The Guardian refused the public `test` key (HTTP 401), "
          "so Guardian counts are absent. 'Bitcoin ETF' has no en.wikipedia article. HN days with more than 1000 matches (only possible "
          "for a single day) use the search engine's nbHits estimate (approx_days above).", "",
          f"BTC days {df['date'].min()} .. {df['date'].max()}; first out-of-sample forecast after a 365-day warm-up.", "",
          "Share of decision days with a non-missing value (missing -> 0 in the regression):", "",
          "| column | " + " | ".join(SEGS) + " |", "|---|---|---|"]
    for c in BLOCK:
        L.append(f"| {c} | " + " | ".join(f"{100 * (nonnull[s][c] or 0):.0f}%" for s in SEGS) + " |")
    L += ["", "## H-NEWS3: next-day QLIKE, block-augmented minus log-HAR x weekday (negative = the block helps)", "",
          "| variant | segment | n | QLIKE base | QLIKE aug | diff | rel | NW t |", "|---|---|---|---|---|---|---|---|"]
    for name, r in p.items():
        for s in SEGS:
            x = r[s]
            L.append(f"| {name} | {s} | {x['n']} | {fmt(x['qlike_base'], 4)} | {fmt(x['qlike_aug'], 4)} | {fmt(x['diff'], 4)} | {fmt(x['rel'], 3)} | {fmt(x['nw_t'], 2)} |")
    L += ["", f"Verdicts (primary, recorded): llm_in_sample **{p['primary']['llm_in_sample']['verdict']}**, clean_holdout **{p['primary']['clean_holdout']['verdict']}**. "
          "Variants are robustness checks (5 tried), not recorded.", "",
          "Block coefficients (OLS on ln RV_{t+1}, in-sample days only; plain OLS t, descriptive):", "", "| term | coef | t |", "|---|---|---|"]
    for k, v in h3["coefs_at_cutoff"].items():
        if k in BLOCK:
            L.append(f"| {k} | {v['coef']:+.4f} | {v['t_ols']:+.2f} |")
    L += ["", "## H-NEWS4: direction (registered as the null)", "",
          "| segment | n | Brier model | Brier base | skill | 95% CI | hit rate | verdict |", "|---|---|---|---|---|---|---|---|"]
    for s in SEGS:
        x = s4[s]
        ci = x.get("ci95") or [float("nan")] * 2
        L.append(f"| {s} | {x['n']} | {fmt(x.get('brier_model'), 4)} | {fmt(x.get('brier_base'), 4)} | {fmt(x.get('skill'), 4)} | [{ci[0]:+.4f}, {ci[1]:+.4f}] | {fmt(x.get('hit_rate'), 3)} | **{x['verdict']}** |")
    L += ["", "## Plain-language reading", "", reading(h3, h4), "",
          "## Caveats", "",
          "- Strict timing costs a day: HN and edit counts for day t are only complete at t + 1 00:30 and Wikipedia pageviews at t + 2, so the midnight forecast uses attention from day t - 1. Same-day attention may carry more, but it is not knowable at the decision time.",
          "- Wikipedia pageviews are today's vintage of history (Wikimedia does not revise daily counts materially); HN counts are today's index (deleted items are gone).",
          "- 'crypto' on HN also matches cryptography talk (exact word only, no prefix matching); the abnormal transform removes its slow-moving level.",
          "- The clean holdout is about four months; a holdout t of -2 needs a large effect.",
          "- Fear and Greed is itself built from price volatility and momentum, so part of it overlaps with the HAR terms.", "",
          "## Rule candidates (research/rules.yaml schema)", "", "```yaml", to_yaml(rules), "```"]
    out.with_suffix(".md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print(out.with_suffix(".md").read_text(encoding="utf-8"))
    if desc is not None:
        print(desc)

    if a.record:
        from ai_trader.research import hypotheses as H

        have = {(r["id"], r["segment"]) for r in H.results() if r["id"] in ("H-NEWS3", "H-NEWS4")}
        rel = f"research/reports/btc_news_alt_{stamp}.json"
        for s in SEGS:
            x = p["primary"][s]
            if ("H-NEWS3", s) not in have:
                H.record("H-NEWS3", s, x["verdict"], {"qlike_diff": x["diff"], "nw_t": x["nw_t"], "rel": x["rel"], "n": x["n"],
                                                      **{f"{k}_t": p[k][s]["nw_t"] for k in VARIANTS}, "n_variants": len(VARIANTS)},
                         note="log-HAR x weekday + 15-col block (wiki pageviews/edits, HN, F&G, events, on-chain), expanding OLS refit 30 d; Guardian refused", data=rel)
                print("recorded H-NEWS3", s, x["verdict"])
            y = s4[s]
            if ("H-NEWS4", s) not in have:
                H.record("H-NEWS4", s, y["verdict"], {"skill": y.get("skill"), "ci95": y.get("ci95"), "n": y["n"], "hit_rate": y.get("hit_rate")},
                         note="walk-forward L2 logistic on the 15-col block, Brier skill vs training base rate, 10-day block bootstrap", data=rel)
                print("recorded H-NEWS4", s, y["verdict"])


if __name__ == "__main__":
    main()
