"""LLM decision layer (phase 4). Planned contents:

- session runner: build market + candidate cards from `snapshots`, run the pinned model with
  tool use, enforce the tool-call budget and the mandatory per-candidate checklist
- view schema: bounded Δ log-odds views (ARCHITECTURE §2.4), validated before reaching the combiner
- hand-off to the Rust risk engine and broker via JSON proposals

Nothing here is wired yet; the deterministic layers come first.
"""
