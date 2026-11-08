"""Render research/reports/SCOREBOARD.md: latest verdict per hypothesis and segment, from the registry
(research/hypotheses.yaml) and the append-only results (research/hypotheses_results.jsonl).

    PYTHONIOENCODING=utf-8 uv run python scripts/scoreboard.py
"""

from __future__ import annotations

import yaml

from ai_trader.core.config import load
from ai_trader.research import hypotheses as H

ICON = {"supports": "✅ supports", "contradicts": "❌ contradicts", "inconclusive": "… inconclusive", None: "· untested"}


def main() -> None:
    root = load().root
    reg = yaml.safe_load((root / "research" / "hypotheses.yaml").read_text(encoding="utf-8"))["hypotheses"]
    latest: dict[tuple[str, str], dict] = {}
    for r in H.results():
        latest[(r["id"], r["segment"])] = r  # results are chronological: last one wins
    lines = ["# Hypothesis scoreboard", "", "_Generated from research/hypotheses.yaml + hypotheses_results.jsonl. "
             "A hypothesis is supported only if it holds in-sample AND keeps its sign in the clean holdout (docs/RESEARCH_PROTOCOL.md)._", "",
             "| id | family | title | prior | in-sample | holdout | overall |", "|---|---|---|---|---|---|---|"]
    counts: dict[str, int] = {}
    for hid, h in reg.items():
        ins = latest.get((hid, "llm_in_sample"), {}).get("verdict")
        hol = latest.get((hid, "clean_holdout"), {}).get("verdict")
        if h.get("data_generated"):
            overall = {"supports": "✅ supported (forward)", "contradicts": "❌ rejected"}.get(hol, "… awaiting forward data")
        elif ins == "supports" and hol == "supports":
            overall = "✅ SUPPORTED"
        elif ins == "contradicts" or hol == "contradicts":
            overall = "❌ not supported"
        elif ins is None and hol is None:
            overall = "· untested"
        else:
            overall = "… open"
        counts[overall] = counts.get(overall, 0) + 1
        flag = " (data-generated)" if h.get("data_generated") else ""
        lines.append(f"| {hid} | {h.get('family', '')} | {h['title']}{flag} | {h['prior']:.2f} | {ICON.get(ins, ins)} | {ICON.get(hol, hol)} | {overall} |")
    lines += ["", "**Totals:** " + ", ".join(f"{k}: {v}" for k, v in sorted(counts.items()))]
    out = root / "research" / "reports" / "SCOREBOARD.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"{out}\n" + lines[-1])


if __name__ == "__main__":
    main()
