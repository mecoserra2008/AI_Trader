"""Compile research/rules.yaml into the decision playbook and the rules audit (read only on rules.yaml).

    PYTHONIOENCODING=utf-8 uv run python scripts/playbook.py [--card next_day_vol --state '{"weekday": "sat", "vol_state": "high"}']

Writes research/PLAYBOOK.md, data/research/playbook.json and research/reports/rules_audit_<UTC date>.md (+ .json).
With --card it only prints that card (for a quick look at what the decision session would load).
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime

from ai_trader.core.config import load
from ai_trader.research import playbook as pb

EXAMPLES = {
    "next-day vol, crypto perp, Saturday, high vol": ("next_day_vol", {"asset_class": "crypto", "weekday": "sat", "vol_state": "high"}),
    "next-day vol, crypto perp, Monday, low vol, v0 trend up": ("next_day_vol", {"asset_class": "crypto", "weekday": "mon", "vol_state": "low", "dir_state": "up"}),
    "next-hour vol, TradFi equity perp at the NYSE open": ("next_hour_vol", {"asset_class": "tradfi", "session": "nyse_open"}),
    "next-day direction, crypto perp, v0 cell R6 (whipsaw)": ("next_day_direction", {"asset_class": "crypto", "cell": "R6", "vol_state": "high"}),
}

NOTES = """**Reviewer's reading (agent F, 2026-11-12).** The automatic pass is keyword-based; these are the flags that hold up on reading:

1. **Monday factor (real conflict).** R-VOL-9 (C) says "no blanket Monday multiplier"; R-ML-1 (B) applies Monday ~1.3x and R-DS-2 (C) +34% vs log-HAR
   for crypto next-day vol. R-VOL-9 rests on H-DOW2 (BTC Monday 1.3x in-sample only); the newer cell and ML evidence sizes it in both segments.
   Recommendation: narrow R-VOL-9 to TradFi / oil (Monday US open, Sunday CME reopen) and let R-ML-1 / R-DS-2 own crypto next-day vol; H-DS4 decides forward.
2. **Shock fade (real tension).** R-PX-1 (B) expects reversal after a |z| >= 4 hourly shock in a volatile market; R-PX-2 (B) forbids fading >= 2 sigma hourly
   deviations. Both match the same bar. Recommendation: until H-S3 (forward) is recorded, the card should apply R-PX-2 (no discretionary fade) and leave the
   reversal leg to the deterministic shock_regime sleeve (R-BUDGET-4).
3. **R-SIM-13 should be downgraded or retired**: its own multi-day test H-MD6 is recorded 'contradicts' (per-day v1 calibration does not beat v0).
4. **Eight A rules have segment-flipping evidence** (R-AVOID-1, R-AVOID-2, R-PD-2, R-PX-3, R-PX-4, R-PX-5, R-PX-6, R-PX-9): a cited test failed in-sample
   but is 'supports' in the holdout (or the reverse), e.g. H-P1 funding contrarian, H-VC1 vector-candle continuation, H-OPT2/3 weekend options, H-W2 weekend
   drift reversion. BELIEFS grade A needs both segments to agree: re-grade to B or split the components that agree; and watch the flipped ideas, because the
   holdout is the only segment Claude has not memorised.
5. **The weekend discount is stated five times** (R-VOL-1 A, R-ML-1 B, R-DS-1 C, R-SP-3 C, R-ML-2 C) with different baselines (Friday sigma, log-HAR,
   ML baseline). Not contradictory, but a card can show three of them: merge into one family with R-VOL-1 as headline.
6. **Next-day crypto vol has six 'prefer' rules** (R-VOL-4, R-VOL-20, R-ML-1, R-SP-1, R-DS-5, R-ML-4). They are compatible as a pipeline - level (log-HAR) ->
   weekday factor -> (C, only if validated) XGB or LAT1 cells - which is the order the playbook card prints; R-VOL-4 (level HAR) is superseded by R-VOL-20.
7. B rules with no decisive registry verdict: R-MAP-1, R-MAP-4 (H-RC1 inconclusive; they rest on the context table), R-SIG-1 (H-VB1 inconclusive).
8. Flags read as **not conflicts**: R-VOL-9 vs R-VOL-27 (hour-of-week profile vs a daily factor). R-DS-4 vs R-SP-3 is real but narrow (calm up-trend Fridays).
9. No rule is recommended for outright retirement on current evidence except R-SIM-13 (and R-VOL-9's crypto clause); 14 provisional micro rules have their
   "more days" trigger met by the 21 random micro days, and their re-tests (H-MD1..H-MD5) are registered and not yet recorded."""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--card", default="")
    ap.add_argument("--state", default="")
    a = ap.parse_args()
    if a.card:
        print(pb.card_for(a.card, json.loads(a.state) if a.state else None))
        return
    root = load().root
    rules = pb.load_rules()
    fwd_p = root / "data" / "research" / "forward_tests" / "status.json"
    fwd = json.loads(fwd_p.read_text(encoding="utf-8")) if fwd_p.is_file() else None
    now = datetime.now(UTC)
    au = pb.audit(rules, fwd, now)
    stem = f"rules_audit_{now:%Y-%m-%d}"
    (root / "research" / "reports" / f"{stem}.md").write_text(pb.render_audit(au, NOTES), encoding="utf-8")
    (root / "research" / "reports" / f"{stem}.json").write_text(json.dumps(au, indent=1, default=str), encoding="utf-8")
    book = pb.playbook(rules)
    book["examples"] = {t: {"context": c, "state": s, "card": pb.card_for(c, s, rules)} for t, (c, s) in EXAMPLES.items()}
    out = root / "data" / "research" / "playbook.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(book, indent=1, default=str), encoding="utf-8")
    (root / "research" / "PLAYBOOK.md").write_text(pb.render_playbook(book, {t: v["card"] for t, v in book["examples"].items()}), encoding="utf-8")
    print(f"{len(rules)} rules, {len(book['contexts'])} contexts -> research/PLAYBOOK.md, data/research/playbook.json")
    print(f"audit: {len(au['pairs'])} pair flags, {len(au['evidence'])} evidence flags, {sum(x['fired'] for x in au['reviews'])} fired review triggers "
          f"-> research/reports/{stem}.md")


if __name__ == "__main__":
    main()
