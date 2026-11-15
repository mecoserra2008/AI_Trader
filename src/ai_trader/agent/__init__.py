"""LLM decision layer (phase 4, forward only, paper book).

- session: state cards, candidates from the validated sleeves (shock_regime, trend_4h), structural placement with
  the H-STOP1 stop cap
- decision: the 4-hourly harness (scripts/decision_session.py prepare / decide / status): PACKET.md, validated
  take / skip answers with bounded log-odds tilts (ARCHITECTURE 2.4), OrderIntents -> Rust risk engine -> paper book
- scorecard: AI vs RULE vs RANDOM vs ALL on the same candidates, H-AI1 / H-AI2 running statistics
"""
