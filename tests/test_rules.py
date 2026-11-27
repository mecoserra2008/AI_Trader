from ai_trader.research.rules import for_prompt, load_rules, render_md, validate


def test_rules_file_is_valid():
    rules = load_rules()
    assert rules, "research/rules.yaml has no rules"
    assert validate(rules) == []


def test_validator_rejects_sizing_and_missing_evidence():
    bad = {"id": "R-VOL-999", "kind": "prefer", "rule": "WHEN vol is high THEN cut the position size in half", "when": "x", "then": "use 2 lots",
           "scope": {"assets": "all"}, "effect": "-", "confidence": "A", "evidence": ["H-NOPE1"], "review": "-", "status": "active"}
    errs = validate([bad])
    assert any("sizing" in e for e in errs) and any("not a registered hypothesis" in e for e in errs)


def _rule(rid: str, evidence: list[str]) -> dict:
    return {"id": rid, "kind": "expect", "rule": "WHEN x THEN y", "when": "x", "then": "y", "scope": {"assets": "all"}, "effect": "-",
            "confidence": "C", "evidence": evidence, "review": "-", "status": "provisional"}


def test_validator_accepts_program_ids_and_rejects_malformed_ones():
    # specification-program ids (AGENT_RULES §4): R-WS04-1 style rules, H-WS05-11b style hypotheses
    assert validate([_rule("R-WS04-99", ["H-WS05-11b", "H-WS04-3"])]) == []
    for rid in ("R-ws04-1", "R-WS04", "R-04-1", "WS04-1"):
        assert any("bad id" in e for e in validate([_rule(rid, ["H-W1"])])), rid
    for hid in ("H-WS05-", "H-ws05-1", "H-WS05-1-2"):
        assert any("not a registered hypothesis" in e for e in validate([_rule("R-VOL-998", [hid])])), hid


def test_prompt_text_lists_strongest_rules_first():
    text = for_prompt()
    lines = text.splitlines()
    assert lines and lines[0].startswith("[A]")


def test_d6_merge_decisions_hold():
    """D6 (2026-11-27, user-approved): the grade / status decisions of the specification-program merge."""
    by = {r["id"]: r for r in load_rules()}
    assert by["R-LEV-1"]["confidence"] == "C" and "daily" in by["R-LEV-1"]["when"]
    assert by["R-PNN-1"]["status"] == "retired" and by["R-MAP-5"]["status"] == "retired"
    for rid in ("R-MAP-4", "R-MAP-6", "R-MAP-7"):          # cell-specific claims are grade C (WS07)
        assert by[rid]["confidence"] == "C" and by[rid]["status"] == "provisional", rid
    assert by["R-MAP-3"]["confidence"] == "B"             # 1h mean reversion fails in every cell (strategy-level)
    for rid in ("R-STRESS-2", "R-STRESS-3"):             # D5: shock fade and setup C retired as live strategies
        assert by[rid]["kind"] == "avoid" and by[rid]["status"] == "active", rid
    assert "intensity" in by["R-FLOW-2"]["then"] and "2 hours" in by["R-FLOW-2"]["rule"]
    assert by["R-WS05-1"]["kind"] == "avoid" and "retired" in by["R-WS05-1"]["rule"]
    # merged candidates do not exist as near-copies
    for gone in ("R-GEO-1", "R-GEO-4", "R-WS04-3", "R-WS05-3", "R-WS12-3", "R-WS14-2"):
        assert gone not in by, gone
    # the bias gate still finds its phrases in R-AVOID-1 (execution.gate.AVOID1_PHRASES)
    t = by["R-AVOID-1"]["rule"].lower()
    assert all(p in t for p in ("funding contrarian", "oi-price divergence", "pivot-geometry trend"))


def test_render_md_names_program_areas():
    md = render_md(load_rules())
    for title in ("Stress screens", "Intraday geometry", "Trend sleeve", "Volatility specification (WS04)"):
        assert f"## {title}" in md
