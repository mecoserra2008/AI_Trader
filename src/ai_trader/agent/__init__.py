"""LLM decision layer (phase 4, forward only, paper book).

- trend_sleeves: the sleeve config (D4 / D5, 2026-11-27): WS16 spec A trend_zoo_ew and spec B trend_tsmom30 (no
  target, rule exit at s x side <= 0), the unconstrained stream and the H-WS16-15 kill monitor; shock_regime and
  setup C retired, trend_4h legacy (open positions only)
- session: state cards, candidates from the trend sleeves, stop placement with the H-STOP1 stop cap
- decision: the 4-hourly harness (scripts/decision_session.py prepare / decide / status): PACKET.md, validated
  take / skip answers with bounded log-odds tilts (ARCHITECTURE 2.4), OrderIntents -> Rust risk engine -> paper book
- scorecard: AI vs RULE vs RANDOM vs ALL on the same candidates, H-AI1 / H-AI2 running statistics
"""
