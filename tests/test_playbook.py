"""Playbook cards and the rules audit: short cards in grade order from live rules only, state filtering, and conflict
detection on a synthetic rules list."""

from __future__ import annotations

import re

from ai_trader.research import playbook as pb
from ai_trader.research.rules import GRADES


def _rule(rid, rule, grade="B", status="active", kind="expect", when=None, then="see rule", assets="crypto", horizon="1d", effect="x"):
    return {"id": rid, "kind": kind, "rule": rule, "when": when or rule.split(" THEN ")[0].removeprefix("WHEN "), "then": then,
            "scope": {"assets": assets, "horizon": horizon, "sides": "both"}, "effect": effect, "confidence": grade, "evidence": ["H-W1"],
            "review": "-", "status": status}


SYN = [
    _rule("R-VOL-91", "WHEN the forecast day is Saturday THEN expect next-day realised variance about 40% below the per-asset log-HAR forecast", "A"),
    _rule("R-VOL-92", "WHEN the forecast day is Saturday THEN expect next-day realised variance about 30% above the per-asset log-HAR forecast", "C",
          status="provisional"),
    _rule("R-VOL-93", "WHEN the forecast day is Monday THEN expect next-day realised variance about 30% above the per-asset log-HAR forecast", "B"),
    _rule("R-VOL-94", "WHEN forecasting next-day variance for a crypto perp THEN prefer the per-asset log-HAR (vol.har) over EWMA", "B", kind="prefer"),
    _rule("R-VOL-95", "WHEN forecasting next-day variance for a crypto perp THEN expect daily variance to be higher than the HAR says", "A",
          status="retired"),
    _rule("R-VOL-96", "WHEN forecasting next-day crypto variance THEN don't call jump-variation or range-estimator (Parkinson / GK) inputs", "C",
          kind="avoid", status="provisional"),
]
RULE_LINE = re.compile(r"^\[([ABC])\] (R-[A-Z]+-\d+) ")


def _rules_in(card: str) -> list[tuple[str, str]]:
    return [m.groups() for m in (RULE_LINE.match(x) for x in card.splitlines()) if m]


def test_card_short_live_and_grade_ordered_on_real_rules():
    live = {r["id"] for r in pb.load_rules() if r["status"] in ("active", "provisional")}
    for ctx in pb.CONTEXTS:
        for state in (None, {"asset_class": "crypto", "weekday": "sat", "vol_state": "high"}):
            got = _rules_in(pb.card_for(ctx, state))
            assert len(got) <= pb.MAX_RULES, ctx
            assert state is not None or got, ctx
            assert all(rid in live for _, rid in got)
            g = [GRADES.index(x) for x, _ in got]
            assert g == sorted(g), (ctx, got)


def test_card_excludes_retired_and_orders_grades():
    card = pb.card_for("next_day_vol", None, SYN)
    got = _rules_in(card)
    ids = [rid for _, rid in got]
    assert "R-VOL-95" not in ids                       # retired
    assert len(got) <= pb.MAX_RULES
    g = [GRADES.index(x) for x, _ in got]
    assert g == sorted(g)                              # A before B before C
    assert "Rules never size positions" in card


def test_state_filters_machine_checkable_conditions():
    sat = [rid for _, rid in _rules_in(pb.card_for("next_day_vol", {"weekday": "Saturday", "asset_class": "crypto"}, SYN))]
    mon = [rid for _, rid in _rules_in(pb.card_for("next_day_vol", {"weekday": "mon", "asset_class": "crypto"}, SYN))]
    assert "R-VOL-91" in sat and "R-VOL-93" not in sat
    assert "R-VOL-93" in mon and "R-VOL-91" not in mon
    tradfi = _rules_in(pb.card_for("next_day_vol", {"asset_class": "tradfi"}, SYN))
    assert tradfi == []                                # every synthetic rule is crypto-only
    # as_of near midnight: the forecast day of a day-ahead card is that UTC date (2026-11-14 is a Saturday)
    assert pb.select("next_day_vol", {"as_of": "2026-11-14T00:00:00+00:00"}, SYN)["state"]["weekday"] == "sat"
    assert pb.select("next_day_vol", {"as_of": "2026-11-13T20:00:00+00:00"}, SYN)["state"]["weekday"] == "sat"


def test_contradicting_pair_detected():
    flags = pb.pair_conflicts(SYN)
    pairs = {(p["a"], p["b"]): p["type"] for p in flags}
    assert pairs.get(("R-VOL-91", "R-VOL-92")) == "contradiction"
    assert ("R-VOL-91", "R-VOL-93") not in pairs       # Saturday vs Monday: conditions exclude each other
    assert not any("R-VOL-95" in k for k in pairs)     # retired rules are not audited as live conflicts


def test_tools_from_rules():
    s = pb.select("next_day_vol", {"asset_class": "crypto"}, SYN)
    assert any(n.startswith("vol.har") for n, _ in s["call"])
    assert any("range estimators" in n for n, _ in s["skip"])
    assert not any("EWMA" in n for n, _ in s["call"])  # "prefer X over EWMA": EWMA is not a tool to call


def test_context_aliases():
    assert pb.context_key("Next-day vol") == "next_day_vol"
    assert pb.context_key("TradFi session open") == "tradfi_open"
    assert pb.context_key("liquidation regime") == "liquidation"
