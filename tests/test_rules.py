from ai_trader.research.rules import for_prompt, load_rules, validate


def test_rules_file_is_valid():
    rules = load_rules()
    assert rules, "research/rules.yaml has no rules"
    assert validate(rules) == []


def test_validator_rejects_sizing_and_missing_evidence():
    bad = {"id": "R-VOL-999", "kind": "prefer", "rule": "WHEN vol is high THEN cut the position size in half", "when": "x", "then": "use 2 lots",
           "scope": {"assets": "all"}, "effect": "-", "confidence": "A", "evidence": ["H-NOPE1"], "review": "-", "status": "active"}
    errs = validate([bad])
    assert any("sizing" in e for e in errs) and any("not a registered hypothesis" in e for e in errs)


def test_prompt_text_lists_strongest_rules_first():
    text = for_prompt()
    lines = text.splitlines()
    assert lines and lines[0].startswith("[A]")
