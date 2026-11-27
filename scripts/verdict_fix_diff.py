"""D7a dry run: which recorded clean_holdout verdicts would change under the fixed holdout rule (nothing is recorded).

    PYTHONIOENCODING=utf-8 uv run python scripts/verdict_fix_diff.py      # writes research/reports/verdict_fix_diff_<date>.md

The fixed rule (src/ai_trader/research/verdicts.py, 2026-11-27): a clean_holdout "supports" must pass a declared one-sided
test of the registered direction at HOLDOUT_ALPHA (5%), not just keep the sign; MIN_HOLDOUT stays; the wrong sign stays
"contradicts"; the right sign that fails the test becomes "inconclusive".

Part A re-judges, from the recorded metrics, every holdout row produced by a judge that goes through verdicts._judge
(run_research.py: H-W1..H-W5b, H-R1, H-R2, H-S1; rv_study: H-RV1..3; vol_fund_study: H-VCL2..5, H-FU1, H-FU2, H-C1).
Part B applies the same rule to every other recorded clean_holdout "supports" (their study scripts have their own
judges; most are sign-only on the holdout), using one primary p per row extracted by WS01's frozen rules
(ai_trader.research.audit_ws01.extract), halved to one-sided in the observed (= registered, for a supports) direction.
Read-only: the registry, the results file and the rules are only read.
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from ai_trader.research import verdicts as V
from ai_trader.research.audit_ws01 import claim_type, extract, p_from_t, simes
from ai_trader.research.hypotheses import registry, results

ROOT = Path(__file__).resolve().parents[1]
STRICT = 0.025   # sensitivity: the one-sided equivalent of the registry's usual in-sample bar (t > 2)


def _f(x, nd: int = 3) -> str:
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "-"
    return f"{x:.{nd}g}" if isinstance(x, float) else str(x)


def _new(n: float, sign_ok: bool, p: float, alpha: float) -> str:
    return V._judge("clean_holdout", False, bool(sign_ok), int(n) if n is not None and np.isfinite(n) else 0, p, alpha)


def rejudge(hid: str, m: dict) -> tuple[str, float, bool, float, str] | None:
    """(statistic text, n, sign_ok, one-sided p, test) for a verdicts._judge hypothesis, or None if not recomputable."""
    g = lambda d, *k: d.get(k[0]) if len(k) == 1 else g(d.get(k[0]) or {}, *k[1:])  # noqa: E731
    if hid == "H-W1":
        n, frac, med = m["n_weekends"], m["frac_below_1"], m["median_rv_ratio"]
        k = int(round(frac * n))
        return f"median {med:.3f}, {k}/{n} weekends < 1", n, med < 1, V.p_one_sided_binom(k, n), "sign test"
    if hid in ("H-W2", "H-W3"):
        want = -1 if hid == "H-W2" else +1
        mm, t, n = m["mean_signed_post_z"], m["t"], m["n_weekends"]
        return f"mean {mm:+.3f} (t {t:.2f})", n, np.sign(mm) == want, V.p_one_sided_t(t, n, want), "one-sided t"
    if hid == "H-W4":
        mm, t, n = m["excess"], m["t"], m["n_weekends"]
        return f"excess {mm:+.3f} (t {t:.2f})", n, mm > 0, V.p_one_sided_t(t, n, +1), "one-sided t"
    if hid in ("H-W5", "H-W5b"):
        r, (lo, hi), n = m["ratio"], m["ci95"], m["n_weekends"]
        return (f"ratio {r:.2f} [{lo:.2f}, {hi:.2f}]", n, r > 1, V.p_one_sided_ci(r, lo, hi, 1.0, +1, log=True),
                "from the 95% CI (log scale; approximates the bootstrap p)")
    if hid == "H-S1":
        cells = {c: m[c] for c in ("R3", "R9", "R4", "R5") if isinstance(m.get(c), dict)}
        if len(cells) < 4:
            return None
        sign_ok = cells["R3"]["mean"] > 0 and cells["R9"]["mean"] > 0 and cells["R4"]["mean"] < 0 and cells["R5"]["mean"] < 0
        ps = [V.p_one_sided_t(cells[c]["t"], cells[c]["n"], +1 if c in ("R3", "R9") else -1) for c in cells]
        txt = ", ".join(f"{c} {cells[c]['mean']:+.2f} (t {cells[c]['t']:.2f}, n {cells[c]['n']})" for c in cells)
        return txt, sum(cells[c]["n"] for c in cells), sign_ok, max(ps), "intersection-union of one-sided t"
    if hid == "H-RV1":
        x = m["net"]
        return f"net {x['mean']:+.4f} (t_nw {x['t_nw']:.2f})", x["n_clusters"], x["mean"] > 0, V.p_one_sided_t(x["t_nw"], x["n_clusters"], +1), "one-sided t"
    if hid in ("H-RV2", "H-RV3"):
        x = m["net_sd"]
        return f"net {x['mean']:+.3f} sd (t {x['t']:.2f})", x["n"], x["mean"] > 0, V.p_one_sided_t(x["t"], x["n"], +1), "one-sided t"
    if hid in ("H-VCL2", "H-VCL3"):
        return f"dQLIKE {m['dqlike']:+.4f} (t {m['t']:.2f})", m["n_origins"], m["dqlike"] < 0, V.p_one_sided_t(m["t"], m["n_origins"], -1), "one-sided t"
    if hid == "H-VCL4":
        x = m["dvol"]
        return f"dQLIKE {x['dqlike']:+.4f} (t {x['t']:.2f})", x["n_origins"], x["dqlike"] < 0, V.p_one_sided_t(x["t"], x["n_origins"], -1), "one-sided t"
    if hid == "H-VCL5":
        med, (lo, hi) = m["median_ratio_events"], m["ci95"]
        return (f"median ratio {med:.2f} [{lo:.2f}, {hi:.2f}]", m["n_event_dates"], med > 1, V.p_one_sided_ci(med, lo, hi, 1.0, +1, log=True),
                "from the 95% CI (log scale)")
    if hid in ("H-FU1", "H-FU2"):
        x = m["btc"]
        return f"BTC slope {x['slope']:+.3f} (t {x['t']:.2f})", x["n"], x["slope"] > 0, V.p_one_sided_t(x["t"], x["n"], +1), "one-sided t"
    if hid == "H-C1":
        x = m["net"]
        return (f"net {x['ann_return']:+.3f}/yr (NW t {x['nw_t']:.2f})", m["n_weeks"], x["ann_return"] > 0, V.p_one_sided_t(x["nw_t"], m["n_weeks"], +1),
                "one-sided t")
    if hid in ("H-R1", "H-R2"):
        return None
    return None


T_KEYS2 = ("t_hln", "t_dm", "dm_t", "dm_t_hln", "t_nw", "t", "z", "t_stat")
N_KEYS2 = ("n", "n_dates", "n_days", "n_obs", "n_trades", "n_events", "n_clusters")


def fallback_p(m: dict, depth: int = 0) -> list[float]:
    """Two-sided p values found in nested metric dicts the WS01 extractor does not read (t_hln-style keys, mean / lo /
    hi CIs); combined by Simes ('some cell') like WS01's cell_combined rows."""
    out: list[float] = []
    if not isinstance(m, dict) or depth > 3:
        return out
    t = next((m[k] for k in T_KEYS2 if isinstance(m.get(k), (int, float)) and not isinstance(m.get(k), bool) and np.isfinite(m[k])), None)
    n = next((m[k] for k in N_KEYS2 if isinstance(m.get(k), (int, float)) and not isinstance(m.get(k), bool) and m[k] >= 3), None)
    if t is not None:
        out.append(p_from_t(float(t), n))
    elif all(isinstance(m.get(k), (int, float)) for k in ("mean", "lo", "hi")) and m["hi"] > m["lo"]:
        se = (m["hi"] - m["lo"]) / (2 * 1.959964)
        out.append(float(2 * __import__("scipy").stats.norm.sf(abs(m["mean"] / se))) if se > 0 else 1.0)
    for v in m.values():
        if isinstance(v, dict):
            out += fallback_p(v, depth + 1)
    return out


PART_A = ("H-W1", "H-W2", "H-W3", "H-W4", "H-W5", "H-W5b", "H-R1", "H-R2", "H-S1", "H-RV1", "H-RV2", "H-RV3",
          "H-VCL2", "H-VCL3", "H-VCL4", "H-VCL5", "H-FU1", "H-FU2", "H-C1")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default=datetime.now(UTC).strftime("%Y-%m-%d"))
    args = ap.parse_args()
    reg, res = registry(), results()
    rules = __import__("yaml").safe_load((ROOT / "research" / "rules.yaml").read_text(encoding="utf-8"))["rules"]
    cites: dict[str, list[str]] = {}
    for r in rules:
        if r["status"] == "retired":
            continue
        for e in r["evidence"]:
            if isinstance(e, str) and e.startswith("H-"):
                cites.setdefault(e, []).append(f"{r['id']} ({r['confidence']})")

    # ---------------------------------------------------------------- Part A
    A = []
    for row in res:
        if row["id"] not in PART_A or row.get("segment") != "clean_holdout":
            continue
        rj = rejudge(row["id"], row.get("metrics") or {})
        if rj is None:
            A.append({"id": row["id"], "ts": row.get("ts", "")[:16], "old": row["verdict"], "stat": "per-cell t not recorded", "n": None, "p": float("nan"),
                      "new": row["verdict"] if row["verdict"] != "supports" else "inconclusive", "new_strict": row["verdict"], "test": "-",
                      "note": "holdout n_ts below the 500 floor or no per-cell t: the fixed rule cannot add a supports"})
            continue
        stat, n, sign_ok, p, test = rj
        A.append({"id": row["id"], "ts": row.get("ts", "")[:16], "old": row["verdict"], "stat": stat, "n": n, "p": p, "test": test,
                  "new": _new(n, sign_ok, p, V.HOLDOUT_ALPHA), "new_strict": _new(n, sign_ok, p, STRICT), "note": ""})

    # ---------------------------------------------------------------- Part B
    B = []
    for row in res:
        if row.get("segment") != "clean_holdout" or row.get("verdict") != "supports" or row["id"] in PART_A:
            continue
        x = extract(row)
        if x.get("source") == "none":
            fb = fallback_p(row.get("metrics") or {})
            if fb:
                x = {"p": simes(fb), "source": f"fallback:simes({len(fb)})", "flag": "nested keys", "n": None}
        p2, src = x.get("p"), x.get("source")
        p1 = float(p2) / 2 if src != "none" and p2 is not None and np.isfinite(p2) else float("nan")
        n = x.get("n")
        small = n is not None and n < V.MIN_HOLDOUT
        null = claim_type(reg.get(row["id"]) or {}) == "null"
        if null:
            new = new_s = "null claim (needs an equivalence / MDE test, not D7a's one-sided effect test)"
        elif src == "none":
            new = new_s = "inconclusive (no statistic recorded)"
        elif small:
            new = new_s = "inconclusive (n < MIN_HOLDOUT)"
        else:
            new = "supports" if p1 <= V.HOLDOUT_ALPHA else "inconclusive"
            new_s = "supports" if p1 <= STRICT else "inconclusive"
        B.append({"id": row["id"], "ts": row.get("ts", "")[:16], "p1": p1, "src": src, "flag": x.get("flag", ""), "n": n, "new": new, "new_strict": new_s,
                  "title": str((reg.get(row["id"]) or {}).get("title", ""))[:90], "rules": cites.get(row["id"], [])})

    # ---------------------------------------------------------------- report
    chA = [a for a in A if a["new"] != a["old"]]
    nb = len(B)
    stay = [b for b in B if b["new"] == "supports"]
    drop = [b for b in B if b["new"] == "inconclusive"]
    nostat = [b for b in B if "no statistic" in b["new"]]
    nulls = [b for b in B if b["new"].startswith("null claim")]
    small = [b for b in B if "MIN_HOLDOUT" in b["new"]]
    stay_s = [b for b in B if b["new_strict"] == "supports"]
    lose = [b for b in B if not b["new"].startswith(("supports", "null claim"))]
    hyp_drop = sorted({b["id"] for b in lose} - {b["id"] for b in B if b["new"] == "supports"})
    ab_rules = sorted({r for b in lose for r in b["rules"] if r.endswith("(A)") or r.endswith("(B)")})
    ab_rules_test = sorted({r for b in B if b["new"] == "inconclusive" for r in b["rules"] if r.endswith("(A)") or r.endswith("(B)")})
    L = [f"# Verdict-fix dry run (D7a), {args.date}", "",
         "_Agent A4 RULES. Dry run only: no result row was recorded, rewritten or deleted; research/hypotheses_results.jsonl was read only. "
         "Generated by `scripts/verdict_fix_diff.py`._", "",
         "## The fix", "",
         "`src/ai_trader/research/verdicts.py` (D7a, user-approved 2026-11-27): a clean_holdout **supports** now needs the right sign **and** a declared "
         f"one-sided test of the registered direction at alpha = {V.HOLDOUT_ALPHA} (the registry's 5% level; a hypothesis that registers its own level "
         f"passes it as `alpha`). `MIN_HOLDOUT` = {V.MIN_HOLDOUT} is kept; the wrong sign stays **contradicts**; the right sign that fails the test becomes "
         "**inconclusive**. The tests per judge: a sign test (H-W1), one-sided t over weekends (H-W2..H-W4), a one-sided bootstrap p (H-W5, H-W5b), "
         "intersection-union of one-sided per-cell t (H-R1, H-R2, H-S1). The external callers of `verdicts._judge` (rv_study, vol_fund_study, including "
         f"vol_fund_study's inline H-C1 holdout rule) now pass their one-sided p. A sensitivity column uses alpha = {STRICT} (the one-sided equivalent of "
         "the registry's usual in-sample bar t > 2).", "",
         "## Summary", "",
         f"- **Part A** (judges that go through `verdicts._judge`): {len(A)} recorded holdout rows, **{len(chA)} would change** "
         f"({', '.join(sorted({a['id'] for a in chA})) or 'none'}).",
         f"- **Part B** (every other recorded clean_holdout supports, judged by its own study script): {nb} rows. Under the same rule "
         f"**{len(stay)} stay supports**, **{len(drop)} become inconclusive** (one-sided p > {V.HOLDOUT_ALPHA}), **{len(nostat)} carry no recorded "
         f"statistic** (cannot pass a declared test: inconclusive until re-run), {len(small)} have n < MIN_HOLDOUT, and **{len(nulls)} are null "
         f"claims** ('no effect' hypotheses: a one-sided effect test is the wrong test; they need an equivalence / MDE check, WS01). "
         f"At alpha {STRICT}: {len(stay_s)} stay.",
         f"- Hypotheses that would lose every holdout supports row: {len(hyp_drop)}.",
         f"- Active / provisional **A or B rules** citing at least one effect-claim row that would lose its supports: {len(ab_rules)} "
         f"({', '.join(ab_rules) if ab_rules else 'none'}); of these, citing a row whose statistic fails the test (rather than a row with no "
         f"statistic): {len(ab_rules_test)} ({', '.join(ab_rules_test) if ab_rules_test else 'none'}). A rule can still stand on its in-sample "
         "result or on other evidence (most of these also cite FDR-surviving or newer rows); these are the grades to re-read, not automatic downgrades.", "",
         "Nothing here is recorded. Re-running a study under the fixed code would record a new row (a new, labelled trial); the old rows stay as they are.", "",
         "## Part A: judges through verdicts._judge (exact re-judgement from the recorded metrics)", "",
         f"| id | recorded at | recorded | statistic | n | one-sided p | test | fixed rule (alpha {V.HOLDOUT_ALPHA}) | alpha {STRICT} |",
         "|---|---|---|---|---|---|---|---|---|"]
    for a in A:
        mark = " **(changes)**" if a["new"] != a["old"] else ""
        L.append(f"| {a['id']} | {a['ts']} | {a['old']} | {a['stat']} | {_f(a['n'])} | {_f(a['p'])} | {a['test']} | {a['new']}{mark} | {a['new_strict']} |")
    L += ["", "H-R1 / H-R2: the recorded holdout rows are inconclusive (the 500-observation floor); their per-cell t are not in the metrics, so the "
          "intersection-union test cannot be re-run from the record, and an inconclusive row cannot become supports under the stricter rule.", "",
          "## Part B: every other recorded clean_holdout supports under the same rule", "",
          "One primary p per row by WS01's frozen extraction rules (raw p, else t with n - 1 df, else a CI, else an adjusted p, else Simes over cells); "
          "for rows those rules cannot read (the 2026-11-27 program's nested keys such as t_hln or mean / lo / hi), a labelled fallback takes Simes "
          "over every nested t or CI (`fallback`); "
          "the two-sided p is halved (the supports row has the registered sign). `simes` / `adjusted_only` rows are approximate. Rules: active or "
          "provisional rules that cite the hypothesis (grade in brackets).", "",
          f"| id | recorded at | one-sided p | source | n | fixed rule (alpha {V.HOLDOUT_ALPHA}) | alpha {STRICT} | rules citing it | title |",
          "|---|---|---|---|---|---|---|---|---|"]
    order = {"inconclusive (no statistic recorded)": 2, "inconclusive (n < MIN_HOLDOUT)": 1, "inconclusive": 0, "supports": 4}
    order.update({b["new"]: 3 for b in nulls})
    for b in sorted(B, key=lambda x: (order.get(x["new"], 9), x["id"])):
        src = b["src"] + (f" ({b['flag']})" if b["flag"] else "")
        L.append(f"| {b['id']} | {b['ts']} | {_f(b['p1'])} | {src} | {_f(b['n'])} | {b['new']} | {b['new_strict']} | {', '.join(b['rules']) or '-'} | {b['title']} |")
    L += ["", "## Caveats", "",
          "- Part B is an approximation of what each study's judge would say under the D7a rule: those judges are not in verdicts.py and some "
          "register a different statistic or a multi-cell success rule; the extracted p is WS01's primary p (one per row), not necessarily the "
          "statistic the registration names. Rows without any recorded statistic (WS01: 217 of 596 rows overall) cannot pass a declared test.",
          "- One-sided p from a two-sided p assumes the observed direction is the registered one, which a supports row implies.",
          "- H-W5 / H-W5b / H-VCL5 p values come from the recorded 95% CI on the log scale (the bootstrap draws were not stored).",
          "- WS01's own caveats apply: several rows store a study-level p shared by segments; bootstrap p floors cap some rows.", ""]
    out = ROOT / "research" / "reports" / f"verdict_fix_diff_{args.date}.md"
    out.write_text("\n".join(L), encoding="utf-8")
    print(json.dumps({"part_a_rows": len(A), "part_a_changed": [a["id"] for a in chA], "part_b_rows": nb, "stay": len(stay), "drop": len(drop),
                      "no_stat": len(nostat), "small": len(small), "null_claims": len(nulls), "stay_strict": len(stay_s), "hyp_lose_all": len(hyp_drop), "ab_rules": ab_rules, "ab_rules_failing_test": ab_rules_test}, indent=1))
    print(out)


if __name__ == "__main__":
    main()
