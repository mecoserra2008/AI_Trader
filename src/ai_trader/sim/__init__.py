"""Simulation and generated-data tools for deterministic strategies (never for the LLM layer, CLAUDE.md principle 1).

  resample     resampling generators of synthetic 1h price paths (B5, H-SYN1, H-SYN2)

Execution simulator v0 (B4, research/hypotheses.yaml block SIM, H-SIM1..H-SIM4):
  data         point-in-time flow_3s reads per symbol, complete 3 s grids with explicit gaps, live mm records
  fill_model   passive-fill model: 3 s trade-through x queue-survival factor, MLE on live quotes (H-SIM1)
  reaction     reaction functions of aggressive flow: regressions and VAR per symbol-day (H-SIM3)
  flow_sim     generator of 3 s flow, trades, spread and mid returns, with `rescale` to other markets (H-SIM2)
  replay       market / limit execution engine on 3 s paths + a faithful port of the `atx mm` quote rule (H-SIM4)

Every SIM result is clean_holdout (flow_3s starts 2026-11-07, after the knowledge cutoff).
"""
