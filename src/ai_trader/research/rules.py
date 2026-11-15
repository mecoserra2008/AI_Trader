"""Decision rules: the research record compressed into conditions the LLM can apply (research/rules.yaml).

A rule is a generalisation of one or more registered, tested hypotheses:

    - id: R-VOL-4
      kind: expect | prefer | avoid
      rule: "WHEN <condition> THEN <expectation or preference>"     one plain sentence
      when: "<condition in tool terms>"                             what the state card / tools must show
      then: "<what to expect or prefer>"                            never a position size (the risk engine sizes)
      scope: {assets: crypto|tradfi|all|[...], horizon: "...", sides: long|short|both}
      effect: "<numbers with uncertainty>"
      confidence: A | B | C                                         BELIEFS.md grades
      evidence: [H-..., <report path>]                              >= 1 hypothesis with a recorded result
      review: "<what would change it>"
      status: active | provisional | retired

Rules change only through registered tests (research protocol); `validate()` enforces the format and the evidence
links, `render_md()` writes the human view research/RULES.md, `for_prompt()` gives the compact text the decision
session loads.

    uv run python -m ai_trader.research.rules            # validate + write research/RULES.md
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import yaml

from ai_trader.core.config import load
from ai_trader.research.hypotheses import registry, results

KINDS = ("expect", "prefer", "avoid")
GRADES = ("A", "B", "C")
STATUSES = ("active", "provisional", "retired")
REQUIRED = ("id", "kind", "rule", "when", "then", "scope", "effect", "confidence", "evidence", "review", "status")
ID_RE = re.compile(r"^R-[A-Z]+-\d+$")
H_RE = re.compile(r"^H-[A-Z]+\d+[a-z]?$")
# Sizing belongs to the risk engine (CLAUDE.md principle 4): rules may not prescribe it.
SIZING_RE = re.compile(r"position size|size the|sizing|risk per trade|% of (equity|capital|the account)|\blots?\b|\bnotional of\b", re.I)


def path() -> Path:
    return load().root / "research" / "rules.yaml"


def load_rules(p: Path | None = None) -> list[dict[str, Any]]:
    return yaml.safe_load((p or path()).read_text(encoding="utf-8")).get("rules", [])


def validate(rules: list[dict[str, Any]]) -> list[str]:
    """Returns a list of problems (empty = valid)."""
    errs: list[str] = []
    reg = registry()
    recorded = {r["id"] for r in results()}
    seen: set[str] = set()
    for r in rules:
        rid = r.get("id", "?")
        missing = [k for k in REQUIRED if k not in r]
        if missing:
            errs.append(f"{rid}: missing {missing}")
            continue
        if not ID_RE.match(rid):
            errs.append(f"{rid}: bad id")
        if rid in seen:
            errs.append(f"{rid}: duplicate id")
        seen.add(rid)
        if r["kind"] not in KINDS:
            errs.append(f"{rid}: kind {r['kind']!r} not in {KINDS}")
        if r["confidence"] not in GRADES:
            errs.append(f"{rid}: confidence {r['confidence']!r} not in {GRADES}")
        if r["status"] not in STATUSES:
            errs.append(f"{rid}: status {r['status']!r} not in {STATUSES}")
        if not str(r["rule"]).startswith("WHEN ") or " THEN " not in str(r["rule"]):
            errs.append(f"{rid}: rule must read 'WHEN ... THEN ...'")
        for field in ("rule", "then"):
            if SIZING_RE.search(str(r[field])):
                errs.append(f"{rid}: {field} prescribes sizing (the risk engine sizes)")
        hids = [e for e in r["evidence"] if isinstance(e, str) and e.startswith("H-")]
        if not hids:
            errs.append(f"{rid}: no hypothesis in evidence")
        for h in hids:
            if not H_RE.match(h) or h not in reg:
                errs.append(f"{rid}: {h} is not a registered hypothesis")
            elif h not in recorded:
                errs.append(f"{rid}: {h} has no recorded result")
        if not isinstance(r["scope"], dict) or "assets" not in r["scope"]:
            errs.append(f"{rid}: scope needs at least 'assets'")
    return errs


def for_prompt(rules: list[dict[str, Any]] | None = None, include_provisional: bool = True) -> str:
    """Compact rule text for the decision session: one line per rule, grade first."""
    rules = load_rules() if rules is None else rules
    keep = [r for r in rules if r["status"] == "active" or (include_provisional and r["status"] == "provisional")]
    keep.sort(key=lambda r: (GRADES.index(r["confidence"]), r["id"]))
    return "\n".join(f"[{r['confidence']}] {r['id']} ({r['kind']}): {r['rule']}  ({r['effect']})" for r in keep)


def render_md(rules: list[dict[str, Any]]) -> str:
    lines = [
        "# Decision rules",
        "",
        "_Generated from [rules.yaml](rules.yaml) by `uv run python -m ai_trader.research.rules`; do not edit by hand. Each rule "
        "generalises registered, tested hypotheses ([hypotheses.yaml](hypotheses.yaml), results in `hypotheses_results.jsonl`). "
        "Grades follow [BELIEFS.md](BELIEFS.md): A = supported before and after the cutoff with a large effect, B = small effect or "
        "forward-confirmed once, C = plausible, forward test running. Rules never size positions: the risk engine does._",
        "",
    ]
    areas: dict[str, list[dict[str, Any]]] = {}
    for r in rules:
        areas.setdefault(r["id"].split("-")[1], []).append(r)
    names = {"VOL": "Volatility", "LEV": "Asymmetry and leverage", "PX": "Price behaviour", "FLOW": "Order flow",
             "SIG": "Signals", "EXEC": "Execution", "MM": "Market making", "CARRY": "Carry", "AVOID": "Don't spend capital on",
             "BUDGET": "Decision budget"}
    for area, rs in areas.items():
        lines += [f"## {names.get(area, area)}", "", "| id | grade | rule | effect | evidence | status |", "|---|---|---|---|---|---|"]
        for r in rs:
            ev = ", ".join(str(e) for e in r["evidence"])
            lines.append(f"| {r['id']} | {r['confidence']} | {r['rule']} | {r['effect']} | {ev} | {r['status']} |")
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    rules = load_rules()
    errs = validate(rules)
    if errs:
        raise SystemExit("rules.yaml invalid:\n  " + "\n  ".join(errs))
    out = load().root / "research" / "RULES.md"
    out.write_text(render_md(rules), encoding="utf-8")
    print(f"{len(rules)} rules valid -> {out}")


if __name__ == "__main__":
    main()
