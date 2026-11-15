"""Report for H-MINE1 (agent E): research/reports/micro_mining_2026-11-12.{json,md}.

Reads data/research/mining/{discovery,discoveries_frozen,confirmation}.json and the cached discovery-day state tables
(descriptives only). Rule candidates are passed in by the caller (scripts/micro_mining_report.py) after the
confirmation, and only for replicated effects.

    PYTHONIOENCODING=utf-8 uv run python -m ai_trader.research.micro_mining_report
"""

from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

import polars as pl

from ai_trader.research import micro_mining as mm

REPORT_STEM = mm.ROOT / "research" / "reports" / "micro_mining_2026-11-12"
FAMILY_TEXT = {
    "F1": "own state cross: class x flow quintile x 15-min vol tercile x hour block (cell vs symbol-day mean)",
    "F2": "flow contrast: top minus bottom flow quintile within class x vol tercile x hour block",
    "F3": "spread regime (L2-book symbols): spread (tick-bound / free) x flow quintile x vol tercile",
    "F3c": "spread contrast: free minus tick-bound within flow quintile x vol tercile",
    "F4a": "leader state for alts: BTC / ETH / SOL same-minute flow quintile x vol tercile",
    "F4b": "leader x own flow for alts: leader flow quintile x own flow quintile",
    "F5": "5-minute table: class x 5-min flow quintile x 30-min vol tercile x hour block (next-5m targets)",
}
TARGET_TEXT = {"a1": "next-1m |return|", "a5": "next-5m |return|", "s1": "next-1m return", "s5": "next-5m return"}
HOUR_TEXT = {0: "00-06 UTC", 1: "06-12 UTC", 2: "12-16 UTC", 3: "16-24 UTC"}
FLOW_TEXT = {0: "most net selling (Q1)", 1: "Q2", 2: "Q3 (balanced)", 3: "Q4", 4: "most net buying (Q5)"}
VOL_TEXT = {0: "low", 1: "mid", 2: "high"}


def plain(eid: str, size: float | None = None) -> str:
    """Plain-words reading of an effect id."""
    parts = eid.split(":")
    fam, tg = parts[0], parts[1]
    kv = {}
    for p in parts[2:]:
        for pre in ("lf", "lv", "sp", "f", "v", "h"):
            if p.startswith(pre) and p[len(pre):].isdigit():
                kv[pre] = int(p[len(pre):])
                break
        else:
            kv["who"] = p
    bits = []
    if fam in ("F1", "F2", "F5"):
        bits.append({"alt": "alts", "major": "majors", "tradfi": "TradFi perps"}[kv["who"]])
    elif fam in ("F3", "F3c"):
        bits.append("L2-book symbols")
    else:
        bits.append(f"alts given {kv['who']}")
    if "lf" in kv:
        bits.append(f"{kv['who']} flow {FLOW_TEXT[kv['lf']]}")
    if "lv" in kv:
        bits.append(f"{kv['who']} vol {VOL_TEXT[kv['lv']]}")
    if "sp" in kv:
        bits.append("free spread" if kv["sp"] else "tick-bound spread")
    if "f" in kv:
        bits.append(f"own flow {FLOW_TEXT[kv['f']]}")
    if "v" in kv:
        bits.append(f"recent vol {VOL_TEXT[kv['v']]}")
    if "h" in kv:
        bits.append(HOUR_TEXT[kv["h"]])
    if fam == "F2":
        bits.append("flow Q5 minus Q1")
    if fam == "F3c":
        bits.append("free minus tick-bound")
    s = ", ".join(bits) + f": {TARGET_TEXT[tg]}"
    if size is not None and math.isfinite(size):
        if tg.startswith("a"):
            s += f" {size * 100:+.0f}% of the symbol-day mean |r|" + (" (contrast)" if fam in ("F2", "F3c") else "")
        else:
            s += f" {size:+.3f} sd" + (" (contrast)" if fam in ("F2", "F3c") else "")
    return s


def _load(name: str) -> dict[str, Any] | None:
    p = mm.MINING_DIR / name
    return json.loads(p.read_text(encoding="utf-8")) if p.is_file() else None


def descriptives() -> dict[str, Any]:
    """Marginal state descriptives on the discovery days (day-mean of the demeaned targets, then mean over days)."""
    t1 = pl.concat([pl.read_parquet(mm.CACHE / f"{d}_t1.parquet") for d in mm.DISCOVERY_DAYS], how="vertical_relaxed")

    def dm(keys, tg, filt=None):
        d = t1 if filt is None else t1.filter(filt)
        for k in keys:
            d = d.filter(pl.col(k) >= 0) if d.schema[k] != pl.Utf8 else d
        g = d.group_by(["day", *keys]).agg(pl.col(tg).mean().alias("m"))
        g = g.group_by(keys).agg(pl.col("m").mean().alias("mean"), (pl.col("m").std() / pl.len().sqrt()).alias("se"),
                                 pl.len().alias("days")).sort(keys)
        return g.to_dicts()

    x = t1.sort(["day", "sym", "minute"]).with_columns(pl.col("vol_t").shift(1).over(["day", "sym"]).alias("pv"),
                                                       pl.col("minute").shift(1).over(["day", "sym"]).alias("pm"))
    pers = x.filter((pl.col("pv") >= 0) & (pl.col("pm") == pl.col("minute") - 1)).select((pl.col("vol_t") == pl.col("pv")).mean()).item()
    spread = t1.filter(pl.col("has_book") == 1).group_by("sym").agg(pl.col("spread").mean().alias("free_share"),
                                                                     pl.len().alias("rows")).sort("free_share").to_dicts()
    nt = t1.group_by("cls").agg(pl.len().alias("rows"), pl.col("sym").n_unique().alias("symbols")).sort("cls").to_dicts()
    return {"rows": t1.height, "by_class": nt, "a1_by_vol": dm(["vol_t"], "a1"), "a1_by_hour": dm(["hour"], "a1"),
            "a1_by_flow": dm(["flow_q"], "a1"), "s1_by_class_flow": dm(["cls", "flow_q"], "s1"),
            "alt_s1_by_btc_flow": dm(["BTC_flow_q"], "s1", pl.col("cls") == "alt"),
            "alt_s5_by_btc_flow": dm(["BTC_flow_q"], "s5", pl.col("cls") == "alt"),
            "vol_tercile_persistence_1m": pers, "spread_free_share": spread}


def _fmt(v, nd=3):
    if v is None or (isinstance(v, float) and not math.isfinite(v)):
        return "n/a"
    if isinstance(v, float):
        return f"{v:.{nd}f}"
    return str(v)


def build(rules: list[dict[str, Any]] | None = None, rule_notes: str = "", extra_md: str = "") -> dict[str, Any]:
    disc = _load("discovery.json")
    fz = _load("discoveries_frozen.json")
    conf = _load("confirmation.json")
    if disc is None or fz is None:
        raise FileNotFoundError("discovery.json / discoveries_frozen.json missing")
    fam = mm.family_counts()
    eff = disc["effects"]
    desc = descriptives()
    out: dict[str, Any] = {"hypothesis": "H-MINE1", "family": fam, "family_text": FAMILY_TEXT, "discovery_days": disc["days"],
                           "confirmation_days_planned": list(mm.CONFIRMATION_DAYS), "frozen_at": fz["frozen_at"],
                           "n_testable": sum(r["testable"] for r in eff), "n_discoveries": fz["n_discoveries"],
                           "discoveries_by_family_kind": {f"{k[0]}:{k[1]}": v for k, v in
                                                         sorted(Counter((x["family"], x["kind"]) for x in fz["discoveries"]).items())},
                           "descriptives": desc, "rules": rules or []}
    if conf is not None:
        out["confirmation_days_used"] = conf["days"]
        out["summary"] = conf["summary"]
        by_f = {}
        for f in FAMILY_TEXT:
            for k in ("vol", "direction"):
                sel = [r for r in conf["effects"] if r["family"] == f and r["kind"] == k]
                if sel:
                    by_f[f"{f}:{k}"] = {"discoveries": len(sel), "replicated": sum(r["pooled"]["replicated"] for r in sel)}
        out["replication_by_family"] = by_f
        out["replicated"] = [{"id": r["id"], "kind": r["kind"], "family": r["family"], "direction": r["direction"],
                              "disc_size": next(x["size"] for x in fz["discoveries"] if x["id"] == r["id"]),
                              "conf_size": r["pooled"]["mean"], "conf_t": r["pooled"]["t"], "conf_p": r["pooled"]["p"],
                              "conf_days": r["pooled"]["n_days"],
                              "holdout_replicated": r["clean_holdout"]["replicated"], "insample_replicated": r["llm_in_sample"]["replicated"],
                              "plain": plain(r["id"], r["pooled"]["mean"])}
                             for r in conf["effects"] if r["pooled"]["replicated"]]
        # shrinkage of replicated sizes
        sz = [(x["conf_size"] / x["disc_size"]) for x in out["replicated"] if x["disc_size"]]
        out["replicated_size_ratio_median"] = float(sorted(sz)[len(sz) // 2]) if sz else None
        out["confirmation_effects"] = [{"id": r["id"], "kind": r["kind"], "direction": r["direction"],
                                        **{s: {k: r[s][k] for k in ("n_days", "mean", "t", "p", "replicated")}
                                           for s in ("pooled", "llm_in_sample", "clean_holdout")}} for r in conf["effects"]]
    REPORT_STEM.parent.mkdir(parents=True, exist_ok=True)
    from ai_trader.research.micro_days import jsonable

    REPORT_STEM.with_suffix(".json").write_text(json.dumps(jsonable(out), indent=1), encoding="utf-8")
    REPORT_STEM.with_suffix(".md").write_text(markdown(out, disc, fz, conf, rule_notes, extra_md), encoding="utf-8")
    return out


def markdown(out, disc, fz, conf, rule_notes: str, extra_md: str) -> str:
    L: list[str] = []
    fam = out["family"]
    L.append("# FDR-controlled mining of discretised micro states (H-MINE1)\n")
    L.append("_Program 2026-11-12, agent E. Code: `src/ai_trader/research/micro_mining.py`, `src/ai_trader/research/micro_mining_report.py`, "
             "`scripts/micro_mining.py`; tests: `tests/test_micro_mining.py`. Data: agent A's aggregates `data/research/micro_days/` "
             "(read only). Frozen discoveries: `data/research/mining/discoveries_frozen.json`. These are deterministic research tools; "
             "the LLM decision layer is not tested here._\n")
    if conf is not None:
        s = out["summary"]
        p = s["pooled"]
        L.append("## Summary\n")
        L.append(f"- **Effect family:** {fam['total']} effects ({fam['vol']} vol, {fam['direction']} direction), all declared before any statistic was computed. {out['n_testable']} of them could be tested on the 16 discovery days.")
        L.append(f"- **Discoveries:** BH at FDR 10% kept **{out['n_discoveries']}**. They were frozen at {out['frozen_at']}, before any confirmation day existed.")
        L.append(f"- **Confirmation:** {len(conf['days'])} of the 15 planned days were used. Each discovery was tested once, one-sided in its discovered sign, with a day-clustered t-test at p < 0.05.")
        L.append(f"- **Replication rate** {p['all']['replicated']}/{p['all']['discoveries']} = **{p['all']['rate']:.2f}**:")
        L.append(f"  - vol effects {p['vol']['replicated']}/{p['vol']['discoveries']} = **{p['vol']['rate']:.2f}**")
        dr = p["direction"]["rate"]
        L.append(f"  - direction effects {p['direction']['replicated']}/{p['direction']['discoveries']} = **{dr:.2f}**")
        c = s["criterion"]
        L.append(f"- **H-MINE1 criterion** (rate >= 50% and vol rate > direction rate): **{'met' if c['met'] else 'not met'}** "
                 f"(rate >= 50%: {c['rate_ge_50']}; vol > direction: {c['vol_gt_direction']}).")
        if out.get("replicated_size_ratio_median") is not None:
            L.append(f"- **Size on confirmation:** replicated effects kept a median {out['replicated_size_ratio_median']:.2f}x of their discovery size.")
        L.append("")
    L.append("## Protocol\n")
    L.append("Each step was fixed in advance:\n")
    L.append("1. The family was declared in code and in the diary (12:33 UTC).")
    L.append("2. Discovery ran on the 16 days of batches 1-3 plus the pilot.")
    L.append("3. BH at FDR 10% was applied over the whole family; effects that could not be tested count with p = 1.")
    L.append("4. The discovery list was frozen in JSON and in the diary (12:34 UTC) before any confirmation day was produced. "
             "`build_day` refuses a confirmation date while the frozen file is missing.")
    L.append("5. Confirmation ran once.\n")
    L.append("**States.** Every state is point in time within the symbol-day: a mid-rank among the same symbol-day's earlier minutes, "
             "after a warm-up of 60 values.\n")
    L.append("- flow quintile: notional imbalance (buy - sell) / (buy + sell) of the last minute (5 minutes in T5);")
    L.append("- vol tercile: log RV of 5 s last-trade returns over the last 15 minutes (30 in T5);")
    L.append("- spread regime (L2 symbols only): time-average spread over the minute, tick-bound if <= 1.5 ticks, free otherwise;")
    L.append("- hour block: 00-06, 06-12, 12-16 or 16-24 UTC;")
    L.append("- class: major, alt or TradFi;")
    L.append("- for alts, the BTC / ETH / SOL flow quintile and vol tercile of the same minute.\n")
    L.append("**Targets.** Next-1m and next-5m |return| (vol) and signed return (direction), from last-trade closes. Each is normalised "
             "per symbol-day (|r| / mean, r / sd) and demeaned. Units:\n")
    L.append("- vol effects are a fraction of the symbol-day mean |r| (+0.30 = 30% above the day's average minute);")
    L.append("- direction effects are in symbol-day sd units.\n")
    L.append("**Inference.**\n")
    L.append("- One estimate per day, pooled over the day's symbols; a day-cell needs at least 30 rows.")
    L.append("- t-test over days with df = D - 1, two-sided for discovery.")
    L.append("- An effect is testable if at least 10 of the 16 discovery days give an estimate.")
    L.append("- Confirmation needs at least 5 days; an effect that can't be tested there counts as not replicated.")
    L.append("- A symbol-day enters only if at least 75% of its minutes have a print.\n")
    L.append("### Effect family\n")
    L.append("| family | definition | effects | testable on discovery | discoveries (vol / direction) |")
    L.append("|---|---|---|---|---|")
    for f, txt in FAMILY_TEXT.items():
        n = fam["by_family"][f]
        te = sum(r["testable"] for r in disc["effects"] if r["family"] == f)
        dv = sum(1 for x in fz["discoveries"] if x["family"] == f and x["kind"] == "vol")
        dd = sum(1 for x in fz["discoveries"] if x["family"] == f and x["kind"] == "direction")
        L.append(f"| {f} | {txt} | {n} | {te} | {dv} / {dd} |")
    L.append(f"| total | | {fam['total']} | {out['n_testable']} | "
             f"{sum(x['kind'] == 'vol' for x in fz['discoveries'])} / {sum(x['kind'] == 'direction' for x in fz['discoveries'])} |\n")
    L.append("Untestable effects are mostly F5 cells (one row per 5 minutes is too few rows per day-cell) and TradFi cells on weekend days.\n")
    d = out["descriptives"]
    L.append("## Discovery-day descriptives (marginals, not part of the tested family)\n")
    L.append(f"{d['rows']:,} T1 rows; " + ", ".join(f"{x['cls']} {x['rows']:,} rows / {x['symbols']} symbols" for x in d["by_class"]) + ". "
             f"The vol tercile stays the same from one minute to the next {d['vol_tercile_persistence_1m']:.0%} of the time.\n")
    L.append("| state | next-1m \\|r\\| vs day mean (day-clustered SE) |")
    L.append("|---|---|")
    for x in d["a1_by_vol"]:
        L.append(f"| vol tercile {VOL_TEXT[x['vol_t']]} | {x['mean']:+.3f} ({x['se']:.3f}) |")
    for x in d["a1_by_hour"]:
        L.append(f"| hour {HOUR_TEXT[x['hour']]} | {x['mean']:+.3f} ({x['se']:.3f}) |")
    for x in d["a1_by_flow"]:
        L.append(f"| flow {FLOW_TEXT[x['flow_q']]} | {x['mean']:+.3f} ({x['se']:.3f}) |")
    L.append("")
    L.append("- **Extreme flow minutes are quiet.** Next-minute |r| is about 9% below the day mean after both flow extremes, because a "
             "one-sided minute is usually a minute with few prints.")
    L.append("- **Vol and hour carry the vol structure.** The 15-minute vol tercile spans about -24% to +28%, and 12-16 UTC is +25%.\n")
    L.append("| next-1m return (sd units) | Q1 | Q2 | Q3 | Q4 | Q5 |")
    L.append("|---|---|---|---|---|---|")
    for c in ("major", "alt", "tradfi"):
        xs = {x["flow_q"]: x for x in d["s1_by_class_flow"] if x["cls"] == c}
        L.append(f"| {c}, own flow | " + " | ".join(f"{xs[q]['mean']:+.3f}" if q in xs else "n/a" for q in range(5)) + " |")
    xs = {x["BTC_flow_q"]: x for x in d["alt_s1_by_btc_flow"]}
    L.append("| alts, BTC flow (same minute) | " + " | ".join(f"{xs[q]['mean']:+.3f}" if q in xs else "n/a" for q in range(5)) + " |")
    xs = {x["BTC_flow_q"]: x for x in d["alt_s5_by_btc_flow"]}
    L.append("| alts, BTC flow -> next-5m | " + " | ".join(f"{xs[q]['mean']:+.3f}" if q in xs else "n/a" for q in range(5)) + " |\n")
    L.append("Share of minutes with a free spread (L2 symbols): " + ", ".join(f"{x['sym'].replace('USDT', '')} {x['free_share']:.2f}" for x in d["spread_free_share"]) + ".\n")
    L.append("## Discoveries (frozen)\n")
    top = sorted(fz["discoveries"], key=lambda x: x["p"])
    L.append(f"{len(top)} discoveries; the largest BH-rejected p is {max(x['p'] for x in top):.4f}. The strongest 12 vol and 12 direction discoveries:\n")
    L.append("| id | reading | size | t (16 days) |")
    L.append("|---|---|---|---|")
    for k in ("vol", "direction"):
        for x in [y for y in top if y["kind"] == k][:12]:
            L.append(f"| {x['id']} | {plain(x['id'])} | {x['size']:+.3f} | {x['t']:.1f} |")
    L.append("")
    if conf is not None:
        L.append("## Confirmation\n")
        segs = conf["segments"]
        L.append(f"Days used ({len(conf['days'])}): {', '.join(conf['days'])}. llm_in_sample {len(segs['llm_in_sample'])}, clean_holdout "
                 f"{len(segs['clean_holdout'])}." + ("" if len(conf["days"]) == 15 else
                                                     f" Missing: {', '.join(d for d in mm.CONFIRMATION_DAYS if d not in conf['days'])}.") + "\n")
        L.append("| segment | kind | discoveries | testable | replicated | rate |")
        L.append("|---|---|---|---|---|---|")
        for s in ("pooled", "llm_in_sample", "clean_holdout"):
            for k in ("all", "vol", "direction"):
                x = out["summary"][s][k]
                L.append(f"| {s} | {k} | {x['discoveries']} | {x['testable']} | {x['replicated']} | {_fmt(x['rate'], 2)} |")
        L.append("\n(The segment rows test each discovery on that segment's days only, which have fewer days and so less power; "
                 "pooled is the registered test.)\n")
        L.append("| family:kind | discoveries | replicated (pooled) |")
        L.append("|---|---|---|")
        for k, v in out["replication_by_family"].items():
            L.append(f"| {k} | {v['discoveries']} | {v['replicated']} |")
        L.append("")
        rep = out["replicated"]
        L.append("### Replicated effects\n")
        for k in ("vol", "direction"):
            xs = sorted([x for x in rep if x["kind"] == k], key=lambda x: x["conf_p"])
            L.append(f"**{k}** ({len(xs)}):\n")
            if xs:
                L.append("| id | reading (confirmation size) | discovery size | confirmation t | days | holdout alone |")
                L.append("|---|---|---|---|---|---|")
                for x in xs[: (40 if k == "vol" else 200)]:
                    L.append(f"| {x['id']} | {x['plain']} | {x['disc_size']:+.3f} | {x['conf_t']:.1f} | {x['conf_days']} | "
                             f"{'yes' if x['holdout_replicated'] else 'no'} |")
                if k == "vol" and len(xs) > 40:
                    L.append(f"\n({len(xs) - 40} more vol effects in the JSON.)")
            L.append("")
    if extra_md:
        L.append(extra_md.rstrip() + "\n")
    if out["rules"]:
        L.append("## Rule candidates\n")
        if rule_notes:
            L.append(rule_notes.rstrip() + "\n")
        L.append("```yaml")
        for r in out["rules"]:
            L.append(f"- id: \"{r['id']}\"")
            L.append(f"  kind: \"{r['kind']}\"")
            L.append(f"  rule: \"{r['rule']}\"")
            L.append(f"  when: \"{r['when']}\"")
            L.append(f"  then: \"{r['then']}\"")
            sc = r["scope"]
            L.append("  scope:")
            L.append(f"    assets: \"{sc['assets']}\"")
            L.append(f"    horizon: \"{sc['horizon']}\"")
            L.append(f"    sides: \"{sc['sides']}\"")
            L.append(f"  effect: \"{r['effect']}\"")
            L.append(f"  confidence: \"{r['confidence']}\"")
            L.append("  evidence: [\"H-MINE1\"]")
            L.append(f"  review: \"{r['review']}\"")
            L.append("  status: \"provisional\"")
        L.append("```\n")
    return "\n".join(L)


if __name__ == "__main__":
    build()
