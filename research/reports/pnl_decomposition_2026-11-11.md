# PnL decomposition of the strategy book (H-PD1..H-PD4) — 2026-11-11

Agent B1, Program 2 ([plan](../plans/2026-11-11_strategy_program.md)); methods fixed in the docstring of `ai_trader.research.pnl_decomposition_study` before running. Input: the strategy panel (see [strategy_map_2026-11-11.md](strategy_map_2026-11-11.md)). Units: daily-sigma units (pnl_d). Segments split at 2026-07-01.

## Headline

- H-PD1 (few bets): llm_in_sample **contradicts**, clean_holdout **contradicts**: the correlation matrix has ~8 eigen-bets (L), but the equal-weight book is ~1-2.5 Meucci bets (extension): one trend exposure dominates.
- H-PD2 (factors explain P&L, no alpha): **contradicts** / **contradicts**: no sleeve has alpha in both segments, but the median R^2 is only ~0.25 (< 0.4). Trend sleeves are TSMOM + long convexity; 1h mean reversion is short convexity / short vol-change; OI divergence is market beta.
- H-PD3 (IPCA beats trailing Sharpe): **supports** / **supports**: largely because trailing-Sharpe ranks carry NEGATIVE information week to week; the IPCA IC itself is small (0.10 in-sample t 3.2, 0.13 holdout t 1.2).
- H-PD4 (costs + exit leakage > 50%): **supports** / **supports**: ~70-75% of the favourable excursion is lost, almost all of it to exits (costs ~5-9% of MFE; MFE is a bar high/low upper bound).

## H-PD1: how many independent bets?

| segment | sleeves | days | top-3 share [95% CI] | ENB [95% CI] | mean abs corr | verdict |
|---|---|---|---|---|---|---|
| llm_in_sample/L | 16 | 2204 | 0.605 [0.590, 0.638] | 8.32 [7.74, 8.62] | 0.272 | contradicts |
| clean_holdout/LT | 30 | 132 | 0.480 [0.462, 0.532] | 13.78 [11.51, 14.03] | 0.184 | contradicts |
| clean_holdout/L | 16 | 132 | 0.655 [0.590, 0.740] | 7.54 [6.07, 8.62] | 0.263 | (robustness) |

Extension (not registered): Meucci ENB of the actual equal-weight book: llm_in_sample|L 1.15; clean_holdout|LT 1.99; clean_holdout|L 2.54.

Extension: registered ENB per v0 cell (universe L; sleeve P&L per date over the symbols in that cell):

| segment | cell | days | sleeves | top-3 share | ENB | Meucci ENB (EW book) |
|---|---|---|---|---|---|---|
| clean_holdout | R3 | 34 | 16 | 0.703 | 6.41 | 2.01 |
| clean_holdout | R4 | 64 | 16 | 0.663 | 7.32 | 2.28 |
| clean_holdout | R5 | 67 | 16 | 0.557 | 9.29 | 2.44 |
| clean_holdout | R6 | 48 | 14 | 0.656 | 7.23 | 1.93 |
| clean_holdout | R7 | 39 | 15 | 0.698 | 6.39 | 1.25 |
| clean_holdout | R8 | 33 | 15 | 0.673 | 7.07 | 2.37 |
| llm_in_sample | R1 | 369 | 16 | 0.596 | 8.97 | 2.19 |
| llm_in_sample | R2 | 368 | 16 | 0.601 | 8.82 | 2.13 |
| llm_in_sample | R3 | 347 | 16 | 0.668 | 7.33 | 1.76 |
| llm_in_sample | R4 | 808 | 16 | 0.563 | 9.84 | 1.18 |
| llm_in_sample | R5 | 752 | 16 | 0.500 | 10.98 | 1.80 |
| llm_in_sample | R6 | 565 | 16 | 0.585 | 9.37 | 1.28 |
| llm_in_sample | R7 | 426 | 15 | 0.733 | 5.97 | 1.23 |
| llm_in_sample | R8 | 383 | 16 | 0.662 | 6.93 | 1.24 |
| llm_in_sample | R9 | 317 | 16 | 0.657 | 6.75 | 1.19 |

In-sample loadings (correlation-scaled) on the first three components, universe L:

| sleeve | PC1 | PC2 | PC3 |
|---|---|---|---|
| funding_contrarian/L | -0.414 | 0.027 | -0.093 |
| donchian_breakout/L | 0.856 | 0.197 | 0.125 |
| ema_stack/L | 0.845 | 0.259 | 0.104 |
| geo_flag_bo/L | 0.754 | -0.538 | -0.053 |
| oi_price_divergence/L | 0.116 | 0.262 | 0.137 |
| shock_regime/L | 0.17 | 0.357 | 0.087 |
| zscore_reversion/L | -0.683 | -0.496 | -0.133 |
| geo_struct/L | 0.674 | -0.568 | -0.067 |
| weekend_drift_fade/L | -0.121 | -0.271 | 0.657 |
| geo_pivot_bo/L | 0.882 | -0.321 | -0.034 |
| weekend_shock_follow/L | 0.135 | 0.132 | -0.579 |
| zoo_ensemble/L | 0.917 | -0.205 | 0.004 |
| rsi_reversion/L | -0.405 | -0.493 | -0.3 |
| compression_breakout/L | 0.591 | -0.4 | -0.066 |
| cme_gap_fill/L | -0.379 | -0.313 | 0.741 |
| tsmom/L | 0.813 | 0.394 | 0.136 |

## H-PD2: what each strategy really is (factor attribution)

### llm_in_sample (factors MKT, TSMOM, VOL, CARRY, LIQ; median R^2 0.246, 16 sleeves)

| sleeve | R^2 | alpha/yr | alpha t | b MKT (t) | b TSMOM (t) | b VOL (t) | b CARRY (t) | b LIQ (t) | raw Sharpe |
|---|---|---|---|---|---|---|---|---|---|
| cme_gap_fill/L | 0.02 | -0.49 | -0.9 | 0.00 (0.1) | -0.01 (-2.9) | -0.00 (-1.0) | -0.01 (-1.7) | 0.00 (0.5) | -0.54 |
| compression_breakout/L | 0.26 | -5.23 | -1.6 | -0.23 (-4.4) | 0.18 (3.5) | -0.05 (-2.3) | 0.07 (1.7) | -0.07 (-1.6) | -0.26 |
| donchian_breakout/L | 0.28 | 0.91 | 0.3 | -0.04 (-1.0) | 0.37 (8.0) | 0.07 (3.9) | 0.02 (0.4) | 0.02 (0.6) | 0.42 |
| ema_stack/L | 0.33 | 0.11 | 0.0 | -0.08 (-3.7) | 0.24 (8.7) | 0.06 (5.4) | 0.01 (0.6) | 0.03 (1.3) | 0.36 |
| funding_contrarian/L | 0.14 | -0.43 | -0.8 | 0.01 (1.5) | -0.04 (-4.9) | 0.00 (0.5) | 0.02 (1.9) | -0.01 (-1.1) | -0.42 |
| geo_flag_bo/L | 0.67 | -3.06 | -2.2 | -0.10 (-6.5) | 0.33 (16.6) | -0.03 (-3.7) | -0.00 (-0.3) | 0.00 (0.1) | -0.03 |
| geo_pivot_bo/L | 0.71 | -1.36 | -0.6 | -0.13 (-5.7) | 0.61 (20.2) | -0.01 (-0.8) | -0.01 (-0.7) | 0.02 (0.6) | 0.38 |
| geo_struct/L | 0.39 | -3.01 | -1.8 | -0.03 (-1.6) | 0.25 (10.4) | -0.03 (-4.0) | 0.00 (0.0) | -0.03 (-1.4) | -0.18 |
| oi_price_divergence/L | 0.41 | -1.83 | -4.5 | 0.05 (18.6) | 0.01 (1.5) | 0.00 (2.1) | -0.00 (-0.7) | 0.01 (2.3) | -1.58 |
| rsi_reversion/L | 0.08 | -2.46 | -4.8 | 0.00 (0.1) | -0.02 (-2.6) | -0.02 (-5.1) | 0.00 (0.0) | 0.02 (1.7) | -1.91 |
| shock_regime/L | 0.03 | 1.88 | 2.5 | -0.00 (-0.6) | 0.01 (1.1) | 0.02 (3.5) | -0.01 (-0.8) | -0.01 (-1.3) | 1.02 |
| tsmom/L | 0.24 | -5.86 | -1.9 | -0.07 (-2.5) | 0.25 (7.5) | 0.11 (7.1) | 0.04 (1.3) | 0.04 (1.2) | -0.41 |
| weekend_drift_fade/L | 0.00 | -0.12 | -0.5 | 0.00 (0.5) | -0.00 (-0.5) | -0.00 (-0.0) | 0.00 (0.5) | 0.00 (1.5) | -0.22 |
| weekend_shock_follow/L | 0.02 | -0.05 | -0.6 | 0.00 (1.4) | 0.00 (1.7) | 0.00 (0.6) | 0.00 (1.8) | -0.00 (-0.2) | -0.14 |
| zoo_ensemble/L | 0.68 | -0.06 | -0.0 | -0.03 (-1.8) | 0.46 (17.6) | 0.02 (1.7) | 0.01 (0.5) | 0.03 (1.3) | 0.52 |
| zscore_reversion/L | 0.19 | -7.36 | -4.3 | 0.04 (2.3) | -0.11 (-4.6) | -0.07 (-6.7) | 0.01 (0.6) | -0.00 (-0.1) | -1.70 |

### clean_holdout (factors MKT, TSMOM, VOL, CARRY, LIQ, QQQ, GOLD, OIL; median R^2 0.224, 30 sleeves)

| sleeve | R^2 | alpha/yr | alpha t | b MKT (t) | b TSMOM (t) | b VOL (t) | b CARRY (t) | b LIQ (t) | b QQQ (t) | b GOLD (t) | b OIL (t) | raw Sharpe |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| cme_gap_fill/L | 0.14 | 0.41 | 0.5 | -0.00 (-1.2) | -0.01 (-1.6) | 0.00 (0.7) | -0.01 (-1.9) | -0.01 (-0.9) | -0.00 (-1.4) | 0.00 (0.1) | -0.00 (-1.6) | 0.36 |
| cme_gap_fill/T | 0.08 | -0.98 | -1.7 | 0.00 (0.8) | 0.00 (0.6) | -0.00 (-1.4) | 0.01 (1.8) | 0.00 (1.2) | 0.00 (0.8) | 0.00 (1.2) | -0.00 (-0.6) | -2.59 |
| compression_breakout/L | 0.68 | -0.22 | -0.0 | -0.23 (-7.0) | 0.09 (3.5) | 0.01 (0.8) | 0.08 (3.6) | -0.11 (-2.6) | -0.01 (-0.4) | 0.00 (0.1) | 0.01 (0.4) | -0.56 |
| compression_breakout/T | 0.49 | -1.09 | -0.2 | 0.04 (1.3) | 0.12 (2.5) | -0.01 (-0.5) | 0.04 (0.6) | 0.03 (0.5) | -0.16 (-5.2) | 0.02 (0.6) | 0.10 (3.9) | 0.89 |
| donchian_breakout/L | 0.46 | -16.81 | -1.8 | 0.40 (2.9) | 0.08 (0.9) | 0.08 (1.5) | 0.10 (1.4) | 0.16 (1.6) | -0.10 (-2.1) | 0.03 (0.8) | 0.06 (1.9) | -0.36 |
| donchian_breakout/T | 0.10 | -4.71 | -0.5 | 0.04 (1.2) | -0.01 (-0.2) | -0.02 (-0.5) | -0.02 (-0.3) | 0.08 (1.1) | -0.11 (-2.6) | 0.04 (0.6) | 0.02 (0.4) | -0.21 |
| ema_stack/L | 0.21 | -10.59 | -1.6 | 0.11 (1.5) | 0.07 (1.2) | 0.04 (1.0) | 0.03 (0.6) | 0.13 (2.0) | -0.04 (-1.2) | 0.03 (1.2) | 0.03 (1.4) | -1.34 |
| ema_stack/T | 0.12 | -14.53 | -1.6 | 0.04 (1.3) | 0.05 (1.1) | -0.04 (-1.0) | -0.02 (-0.4) | 0.06 (0.9) | -0.09 (-2.4) | 0.02 (0.5) | 0.03 (0.8) | -1.79 |
| funding_contrarian/L | 0.15 | -0.13 | -0.1 | 0.01 (1.9) | 0.01 (1.8) | -0.01 (-2.0) | 0.01 (1.4) | -0.01 (-1.6) | -0.00 (-0.5) | -0.00 (-1.1) | -0.00 (-1.4) | 0.40 |
| funding_contrarian/T | 0.04 | -3.39 | -1.9 | -0.01 (-1.1) | 0.00 (0.3) | -0.01 (-1.8) | -0.01 (-0.6) | -0.01 (-1.0) | -0.00 (-0.8) | 0.00 (0.7) | 0.00 (0.4) | -3.25 |
| geo_flag_bo/L | 0.57 | -2.77 | -0.7 | -0.11 (-5.0) | 0.23 (7.1) | -0.02 (-1.4) | -0.10 (-3.2) | -0.04 (-1.4) | 0.02 (1.2) | -0.02 (-1.4) | -0.00 (-0.1) | -2.28 |
| geo_flag_bo/T | 0.44 | -13.84 | -2.4 | 0.03 (0.8) | 0.10 (1.9) | 0.02 (0.8) | 0.03 (0.6) | -0.05 (-1.2) | -0.05 (-2.2) | -0.17 (-4.4) | 0.02 (0.7) | -1.98 |
| geo_pivot_bo/L | 0.63 | -5.31 | -0.8 | 0.01 (0.2) | 0.46 (9.6) | -0.01 (-0.5) | -0.09 (-2.1) | 0.06 (1.0) | -0.00 (-0.2) | -0.01 (-0.3) | 0.01 (0.5) | -1.19 |
| geo_pivot_bo/T | 0.20 | -20.70 | -2.3 | 0.03 (0.4) | 0.20 (2.2) | 0.03 (0.7) | 0.04 (0.7) | -0.07 (-1.4) | -0.09 (-2.1) | -0.05 (-0.8) | 0.03 (0.7) | -2.26 |
| geo_struct/L | 0.39 | -2.45 | -0.6 | -0.11 (-2.3) | 0.18 (4.0) | -0.04 (-1.9) | -0.09 (-2.2) | -0.06 (-1.2) | 0.04 (1.9) | -0.02 (-1.3) | -0.00 (-0.2) | -2.13 |
| geo_struct/T | 0.39 | -13.00 | -1.8 | -0.01 (-0.3) | 0.11 (2.2) | -0.00 (-0.0) | 0.05 (1.5) | -0.07 (-2.2) | -0.02 (-0.6) | -0.14 (-3.8) | 0.03 (0.9) | -2.17 |
| oi_price_divergence/L | 0.61 | -0.99 | -0.6 | 0.07 (6.6) | 0.01 (1.1) | 0.01 (1.9) | -0.01 (-1.3) | -0.01 (-0.6) | -0.01 (-3.7) | 0.00 (1.2) | 0.00 (1.2) | 0.54 |
| rsi_reversion/L | 0.44 | -0.49 | -0.4 | -0.06 (-3.8) | 0.01 (0.7) | -0.02 (-2.1) | 0.00 (0.0) | 0.01 (1.1) | 0.02 (2.9) | -0.00 (-0.6) | -0.00 (-0.8) | -1.87 |
| rsi_reversion/T | 0.08 | -7.56 | -2.9 | -0.01 (-1.2) | -0.00 (-0.1) | -0.02 (-1.3) | 0.00 (0.2) | 0.02 (0.8) | 0.02 (2.1) | 0.01 (0.5) | 0.02 (1.6) | -4.34 |
| shock_regime/L | 0.41 | -3.44 | -1.5 | 0.07 (4.3) | -0.06 (-3.2) | -0.02 (-1.5) | -0.02 (-1.1) | -0.05 (-1.5) | -0.02 (-2.6) | -0.01 (-1.2) | 0.00 (0.2) | -0.75 |
| shock_regime/T | 0.07 | -3.92 | -1.0 | 0.02 (1.6) | -0.02 (-0.8) | -0.00 (-0.3) | -0.00 (-0.1) | -0.03 (-1.2) | -0.01 (-0.5) | -0.04 (-1.6) | -0.01 (-1.0) | -1.37 |
| tsmom/L | 0.28 | -18.27 | -2.0 | 0.23 (2.4) | 0.08 (0.9) | 0.06 (1.0) | 0.03 (0.5) | 0.08 (1.0) | -0.09 (-2.5) | 0.05 (1.4) | 0.04 (1.1) | -1.34 |
| tsmom/T | 0.12 | -29.65 | -2.6 | 0.04 (0.7) | 0.11 (1.6) | -0.03 (-0.6) | 0.05 (0.5) | 0.14 (1.5) | -0.12 (-2.2) | 0.03 (0.5) | 0.04 (0.6) | -2.51 |
| weekend_drift_fade/L | 0.04 | 0.02 | 0.1 | -0.00 (-0.9) | -0.00 (-0.9) | -0.00 (-0.6) | -0.00 (-1.0) | 0.00 (0.5) | 0.00 (0.4) | 0.00 (1.0) | -0.00 (-0.6) | -0.29 |
| weekend_drift_fade/T | 0.08 | -0.13 | -1.4 | -0.00 (-0.9) | 0.00 (0.9) | -0.00 (-1.3) | -0.00 (-1.1) | -0.00 (-1.2) | -0.00 (-1.0) | 0.00 (1.0) | 0.00 (1.4) | -2.53 |
| weekend_shock_follow/L | 0.16 | -0.20 | -1.2 | 0.00 (1.0) | 0.00 (0.3) | -0.00 (-1.3) | 0.00 (1.8) | 0.00 (1.9) | 0.00 (1.0) | 0.00 (1.0) | 0.00 (1.3) | -1.54 |
| zoo_ensemble/L | 0.70 | -3.37 | -0.8 | 0.05 (1.2) | 0.34 (10.0) | 0.01 (0.3) | -0.00 (-0.1) | 0.04 (1.0) | -0.02 (-1.2) | -0.00 (-0.1) | 0.01 (0.5) | -0.48 |
| zoo_ensemble/T | 0.24 | -13.90 | -2.0 | 0.05 (1.3) | 0.10 (1.9) | 0.01 (0.3) | 0.03 (0.7) | -0.02 (-0.4) | -0.09 (-2.6) | -0.04 (-0.9) | 0.03 (1.3) | -1.79 |
| zscore_reversion/L | 0.54 | -1.28 | -0.2 | -0.36 (-3.0) | 0.05 (0.9) | -0.10 (-2.4) | -0.10 (-2.0) | -0.08 (-1.0) | 0.10 (2.9) | -0.02 (-0.8) | -0.04 (-1.7) | -2.08 |
| zscore_reversion/T | 0.17 | -8.38 | -1.7 | -0.02 (-0.6) | -0.02 (-0.9) | 0.01 (0.5) | 0.01 (0.3) | -0.03 (-0.9) | 0.08 (3.6) | -0.02 (-0.4) | 0.02 (0.7) | -2.56 |

### llm_in_sample_ext (EXTENSION, not registered) (factors MKT, TSMOM, VOL, CARRY, LIQ, GOLD; median R^2 0.266, 16 sleeves)

| sleeve | R^2 | alpha/yr | alpha t | b MKT (t) | b TSMOM (t) | b VOL (t) | b CARRY (t) | b LIQ (t) | b GOLD (t) | raw Sharpe |
|---|---|---|---|---|---|---|---|---|---|---|
| cme_gap_fill/L | 0.01 | 2.45 | 2.5 | 0.00 (0.9) | -0.00 (-0.4) | 0.00 (1.0) | 0.00 (0.4) | 0.00 (0.0) | -0.00 (-0.2) | 2.02 |
| compression_breakout/L | 0.26 | -1.91 | -0.4 | -0.14 (-2.4) | 0.15 (2.5) | -0.04 (-1.6) | 0.01 (0.3) | -0.04 (-0.4) | -0.00 (-0.3) | 0.11 |
| donchian_breakout/L | 0.36 | -1.95 | -0.3 | -0.05 (-0.8) | 0.38 (6.6) | 0.07 (2.6) | -0.11 (-2.0) | 0.10 (1.0) | -0.02 (-0.9) | -0.13 |
| ema_stack/L | 0.40 | -2.70 | -0.7 | -0.06 (-2.1) | 0.25 (6.1) | 0.06 (3.3) | -0.07 (-1.9) | 0.15 (3.2) | -0.01 (-1.1) | -0.24 |
| funding_contrarian/L | 0.22 | -0.04 | -0.1 | 0.02 (5.5) | -0.00 (-0.9) | 0.00 (0.1) | 0.00 (0.3) | -0.01 (-2.4) | 0.00 (1.1) | -0.39 |
| geo_flag_bo/L | 0.73 | 0.77 | 0.3 | -0.15 (-4.9) | 0.31 (9.3) | -0.02 (-1.2) | -0.00 (-0.1) | 0.06 (1.2) | -0.01 (-1.1) | 0.65 |
| geo_pivot_bo/L | 0.76 | 2.00 | 0.5 | -0.22 (-5.6) | 0.53 (11.0) | 0.02 (0.9) | -0.02 (-0.5) | 0.08 (2.3) | -0.01 (-0.9) | 0.69 |
| geo_struct/L | 0.55 | -0.12 | -0.0 | -0.14 (-3.8) | 0.25 (5.3) | -0.01 (-0.6) | 0.00 (0.1) | -0.03 (-0.8) | -0.00 (-0.1) | 0.49 |
| oi_price_divergence/L | 0.48 | -1.90 | -2.3 | 0.05 (9.4) | 0.00 (0.6) | 0.00 (1.4) | -0.02 (-2.4) | 0.01 (1.1) | 0.00 (0.2) | -2.22 |
| rsi_reversion/L | 0.12 | -3.54 | -3.2 | 0.01 (0.4) | -0.02 (-1.3) | -0.02 (-2.8) | 0.01 (1.3) | 0.02 (0.6) | 0.00 (0.8) | -2.50 |
| shock_regime/L | 0.05 | 1.86 | 1.3 | -0.00 (-0.4) | -0.01 (-0.6) | 0.02 (1.7) | -0.04 (-2.7) | -0.01 (-0.7) | 0.01 (1.2) | 0.98 |
| tsmom/L | 0.28 | -10.73 | -1.8 | -0.04 (-0.9) | 0.25 (4.5) | 0.09 (3.6) | -0.08 (-1.7) | 0.20 (3.0) | -0.02 (-1.3) | -1.23 |
| weekend_drift_fade/L | 0.01 | 0.77 | 1.5 | 0.00 (0.6) | 0.00 (0.2) | 0.00 (0.7) | 0.00 (0.7) | 0.00 (0.1) | 0.00 (0.1) | 1.20 |
| weekend_shock_follow/L | 0.01 | -0.31 | -2.1 | 0.00 (0.3) | -0.00 (-0.2) | -0.00 (-1.2) | 0.00 (1.1) | -0.00 (-0.8) | -0.00 (-0.5) | -1.79 |
| zoo_ensemble/L | 0.73 | -0.98 | -0.3 | -0.07 (-2.7) | 0.42 (11.5) | 0.03 (1.9) | -0.01 (-0.4) | 0.13 (3.6) | -0.01 (-1.1) | 0.20 |
| zscore_reversion/L | 0.20 | -8.25 | -2.3 | 0.01 (0.2) | -0.14 (-2.8) | -0.07 (-3.4) | 0.07 (2.3) | -0.02 (-0.2) | 0.01 (0.5) | -1.66 |

### llm_in_sample_convex (EXTENSION, not registered) (factors MKT, TSMOM, VOL, CARRY, LIQ, CONVEX; median R^2 0.357, 16 sleeves)

| sleeve | R^2 | alpha/yr | alpha t | b MKT (t) | b TSMOM (t) | b VOL (t) | b CARRY (t) | b LIQ (t) | b CONVEX (t) | raw Sharpe |
|---|---|---|---|---|---|---|---|---|---|---|
| cme_gap_fill/L | 0.02 | -0.49 | -0.9 | 0.00 (0.1) | -0.01 (-3.1) | -0.00 (-0.9) | -0.01 (-1.7) | 0.00 (0.5) | -0.00 (-0.2) | -0.54 |
| compression_breakout/L | 0.26 | -5.25 | -1.6 | -0.23 (-4.6) | 0.19 (3.6) | -0.04 (-2.1) | 0.07 (1.7) | -0.08 (-1.7) | -0.04 (-0.9) | -0.26 |
| donchian_breakout/L | 0.34 | 1.05 | 0.3 | -0.02 (-0.7) | 0.35 (8.4) | 0.00 (0.3) | 0.01 (0.4) | 0.04 (1.2) | 0.23 (7.0) | 0.42 |
| ema_stack/L | 0.39 | 0.21 | 0.1 | -0.07 (-3.6) | 0.23 (9.5) | 0.02 (1.9) | 0.01 (0.5) | 0.04 (1.8) | 0.16 (7.6) | 0.36 |
| funding_contrarian/L | 0.15 | -0.44 | -0.8 | 0.01 (1.4) | -0.04 (-4.9) | 0.01 (1.7) | 0.02 (1.9) | -0.01 (-1.2) | -0.01 (-2.4) | -0.42 |
| geo_flag_bo/L | 0.67 | -3.07 | -2.2 | -0.10 (-6.8) | 0.33 (16.8) | -0.02 (-3.6) | -0.00 (-0.3) | 0.00 (0.1) | -0.02 (-1.1) | -0.03 |
| geo_pivot_bo/L | 0.71 | -1.35 | -0.6 | -0.13 (-5.6) | 0.61 (20.3) | -0.02 (-1.9) | -0.01 (-0.7) | 0.02 (0.7) | 0.03 (1.3) | 0.38 |
| geo_struct/L | 0.40 | -3.03 | -1.9 | -0.04 (-1.8) | 0.26 (10.7) | -0.03 (-3.2) | 0.00 (0.0) | -0.03 (-1.5) | -0.03 (-1.7) | -0.18 |
| oi_price_divergence/L | 0.43 | -1.82 | -4.4 | 0.05 (20.0) | 0.00 (1.1) | -0.00 (-0.9) | -0.00 (-0.7) | 0.02 (2.7) | 0.02 (5.3) | -1.58 |
| rsi_reversion/L | 0.20 | -2.49 | -4.9 | -0.00 (-0.6) | -0.01 (-2.4) | -0.01 (-2.2) | 0.00 (0.1) | 0.02 (1.8) | -0.04 (-5.9) | -1.91 |
| shock_regime/L | 0.07 | 1.90 | 2.5 | -0.00 (-0.1) | 0.00 (0.7) | 0.01 (1.3) | -0.01 (-0.9) | -0.01 (-1.1) | 0.04 (4.7) | 1.02 |
| tsmom/L | 0.37 | -5.67 | -1.9 | -0.04 (-2.2) | 0.22 (8.4) | 0.02 (1.7) | 0.03 (1.3) | 0.06 (2.2) | 0.30 (11.5) | -0.41 |
| weekend_drift_fade/L | 0.01 | -0.11 | -0.5 | 0.00 (0.7) | -0.00 (-0.7) | -0.00 (-0.5) | 0.00 (0.5) | 0.00 (1.6) | 0.00 (1.7) | -0.22 |
| weekend_shock_follow/L | 0.02 | -0.05 | -0.6 | 0.00 (1.4) | 0.00 (1.9) | 0.00 (0.5) | 0.00 (1.8) | -0.00 (-0.2) | -0.00 (-0.0) | -0.14 |
| zoo_ensemble/L | 0.68 | -0.04 | -0.0 | -0.03 (-1.7) | 0.46 (17.6) | 0.00 (0.5) | 0.01 (0.4) | 0.03 (1.5) | 0.04 (2.2) | 0.52 |
| zscore_reversion/L | 0.40 | -7.49 | -4.5 | 0.03 (2.2) | -0.09 (-5.1) | -0.01 (-1.2) | 0.01 (0.9) | -0.02 (-1.0) | -0.21 (-12.8) | -1.70 |

### clean_holdout_convex (EXTENSION, not registered) (factors MKT, TSMOM, VOL, CARRY, LIQ, QQQ, GOLD, OIL, CONVEX; median R^2 0.299, 30 sleeves)

| sleeve | R^2 | alpha/yr | alpha t | b MKT (t) | b TSMOM (t) | b VOL (t) | b CARRY (t) | b LIQ (t) | b QQQ (t) | b GOLD (t) | b OIL (t) | b CONVEX (t) | raw Sharpe |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| cme_gap_fill/L | 0.21 | 0.64 | 0.7 | -0.01 (-2.3) | -0.01 (-1.5) | -0.00 (-0.7) | -0.01 (-2.2) | -0.01 (-0.8) | -0.00 (-0.4) | 0.00 (0.1) | -0.00 (-1.6) | 0.01 (2.4) | 0.36 |
| cme_gap_fill/T | 0.09 | -1.03 | -1.8 | 0.00 (0.9) | 0.00 (0.4) | -0.00 (-1.0) | 0.01 (1.9) | 0.00 (1.3) | 0.00 (0.0) | 0.00 (1.1) | -0.00 (-0.6) | -0.00 (-0.9) | -2.59 |
| compression_breakout/L | 0.70 | 0.84 | 0.2 | -0.26 (-11.8) | 0.09 (3.5) | -0.00 (-0.3) | 0.07 (3.1) | -0.11 (-2.6) | 0.01 (0.7) | 0.00 (0.0) | 0.01 (0.5) | 0.06 (2.3) | -0.56 |
| compression_breakout/T | 0.52 | 0.93 | 0.1 | -0.02 (-0.5) | 0.12 (2.7) | -0.05 (-1.4) | 0.03 (0.4) | 0.03 (0.7) | -0.14 (-4.3) | 0.02 (0.6) | 0.10 (3.9) | 0.12 (3.1) | 0.89 |
| donchian_breakout/L | 0.60 | -10.83 | -1.3 | 0.24 (3.7) | 0.09 (1.4) | -0.03 (-0.9) | 0.07 (1.2) | 0.17 (2.6) | -0.01 (-0.4) | 0.03 (0.9) | 0.07 (2.4) | 0.37 (5.2) | -0.36 |
| donchian_breakout/T | 0.10 | -4.59 | -0.4 | 0.04 (1.1) | -0.01 (-0.2) | -0.02 (-0.5) | -0.02 (-0.3) | 0.08 (1.1) | -0.11 (-2.1) | 0.04 (0.5) | 0.02 (0.4) | 0.01 (0.1) | -0.21 |
| ema_stack/L | 0.34 | -7.20 | -1.2 | 0.01 (0.3) | 0.08 (1.5) | -0.02 (-0.6) | 0.01 (0.2) | 0.14 (3.0) | 0.01 (0.4) | 0.03 (1.2) | 0.04 (1.7) | 0.21 (4.9) | -1.34 |
| ema_stack/T | 0.13 | -13.33 | -1.5 | 0.01 (0.1) | 0.05 (1.2) | -0.06 (-1.2) | -0.03 (-0.5) | 0.06 (1.0) | -0.07 (-1.6) | 0.02 (0.5) | 0.03 (0.8) | 0.07 (1.1) | -1.79 |
| funding_contrarian/L | 0.17 | -0.28 | -0.3 | 0.01 (2.5) | 0.01 (1.7) | -0.01 (-1.2) | 0.01 (1.6) | -0.01 (-1.6) | -0.00 (-1.2) | -0.00 (-1.1) | -0.00 (-1.4) | -0.01 (-1.7) | 0.40 |
| funding_contrarian/T | 0.04 | -3.46 | -1.9 | -0.00 (-0.6) | 0.00 (0.3) | -0.01 (-1.2) | -0.01 (-0.5) | -0.01 (-1.1) | -0.01 (-0.9) | 0.00 (0.7) | 0.00 (0.4) | -0.00 (-0.5) | -3.25 |
| geo_flag_bo/L | 0.58 | -3.12 | -0.7 | -0.10 (-3.0) | 0.23 (7.3) | -0.01 (-1.1) | -0.10 (-3.1) | -0.04 (-1.5) | 0.01 (1.0) | -0.02 (-1.4) | -0.00 (-0.1) | -0.02 (-0.7) | -2.28 |
| geo_flag_bo/T | 0.47 | -12.02 | -2.0 | -0.02 (-0.6) | 0.10 (2.2) | -0.01 (-0.8) | 0.02 (0.4) | -0.04 (-1.2) | -0.02 (-1.0) | -0.17 (-4.5) | 0.02 (0.9) | 0.11 (2.8) | -1.98 |
| geo_pivot_bo/L | 0.64 | -4.32 | -0.6 | -0.02 (-0.3) | 0.47 (8.9) | -0.03 (-1.5) | -0.09 (-2.1) | 0.06 (1.1) | 0.01 (0.4) | -0.01 (-0.3) | 0.01 (0.5) | 0.06 (1.0) | -1.19 |
| geo_pivot_bo/T | 0.25 | -17.83 | -2.1 | -0.05 (-0.7) | 0.21 (2.5) | -0.03 (-0.9) | 0.02 (0.4) | -0.06 (-1.2) | -0.05 (-1.1) | -0.05 (-0.8) | 0.03 (0.8) | 0.18 (2.9) | -2.26 |
| geo_struct/L | 0.46 | -4.18 | -0.8 | -0.06 (-1.5) | 0.17 (4.4) | -0.01 (-0.4) | -0.08 (-2.0) | -0.06 (-1.6) | 0.01 (0.9) | -0.02 (-1.5) | -0.00 (-0.3) | -0.11 (-2.5) | -2.13 |
| geo_struct/T | 0.42 | -11.41 | -1.7 | -0.06 (-1.4) | 0.12 (2.4) | -0.03 (-1.4) | 0.04 (1.1) | -0.07 (-2.1) | 0.01 (0.2) | -0.14 (-3.8) | 0.03 (1.0) | 0.10 (3.1) | -2.17 |
| oi_price_divergence/L | 0.72 | -0.06 | -0.0 | 0.05 (10.9) | 0.02 (1.6) | -0.00 (-0.7) | -0.02 (-2.0) | -0.01 (-0.7) | 0.00 (0.2) | 0.00 (1.6) | 0.01 (1.9) | 0.06 (7.1) | 0.54 |
| rsi_reversion/L | 0.62 | -1.42 | -1.3 | -0.03 (-4.5) | 0.01 (0.7) | -0.00 (-0.3) | 0.01 (0.5) | 0.01 (1.0) | 0.00 (0.6) | -0.00 (-0.7) | -0.00 (-1.6) | -0.06 (-7.3) | -1.87 |
| rsi_reversion/T | 0.08 | -7.63 | -2.9 | -0.01 (-1.0) | -0.00 (-0.1) | -0.01 (-1.1) | 0.01 (0.3) | 0.02 (0.9) | 0.02 (1.7) | 0.01 (0.5) | 0.02 (1.5) | -0.00 (-0.2) | -4.34 |
| shock_regime/L | 0.41 | -3.29 | -1.4 | 0.06 (4.0) | -0.06 (-3.2) | -0.02 (-1.8) | -0.02 (-1.1) | -0.05 (-1.5) | -0.02 (-2.4) | -0.01 (-1.2) | 0.00 (0.2) | 0.01 (0.6) | -0.75 |
| shock_regime/T | 0.08 | -3.55 | -0.9 | 0.01 (0.7) | -0.02 (-0.8) | -0.01 (-0.6) | -0.00 (-0.1) | -0.03 (-1.2) | -0.00 (-0.1) | -0.04 (-1.6) | -0.01 (-1.0) | 0.02 (1.1) | -1.37 |
| tsmom/L | 0.49 | -12.20 | -1.6 | 0.06 (1.4) | 0.10 (1.5) | -0.05 (-1.2) | -0.01 (-0.1) | 0.09 (1.8) | -0.01 (-0.2) | 0.05 (1.7) | 0.04 (1.6) | 0.37 (6.9) | -1.34 |
| tsmom/T | 0.12 | -29.60 | -2.5 | 0.04 (0.5) | 0.11 (1.6) | -0.04 (-0.5) | 0.05 (0.5) | 0.14 (1.5) | -0.12 (-1.9) | 0.03 (0.5) | 0.04 (0.6) | 0.00 (0.0) | -2.51 |
| weekend_drift_fade/L | 0.06 | 0.07 | 0.2 | -0.00 (-1.6) | -0.00 (-0.8) | -0.00 (-1.5) | -0.00 (-1.4) | 0.00 (0.5) | 0.00 (0.7) | 0.00 (1.0) | -0.00 (-0.6) | 0.00 (1.8) | -0.29 |
| weekend_drift_fade/T | 0.08 | -0.13 | -1.4 | -0.00 (-0.9) | 0.00 (0.9) | -0.00 (-1.2) | -0.00 (-1.1) | -0.00 (-1.2) | -0.00 (-1.1) | 0.00 (0.9) | 0.00 (1.4) | 0.00 (0.5) | -2.53 |
| weekend_shock_follow/L | 0.25 | -0.24 | -1.4 | 0.00 (1.5) | 0.00 (0.0) | 0.00 (0.0) | 0.00 (1.9) | 0.00 (2.1) | -0.00 (-0.4) | 0.00 (1.0) | 0.00 (1.4) | -0.00 (-1.7) | -1.54 |
| zoo_ensemble/L | 0.74 | -1.63 | -0.4 | 0.01 (0.2) | 0.35 (10.4) | -0.03 (-1.7) | -0.01 (-0.4) | 0.04 (1.4) | 0.00 (0.0) | -0.00 (-0.1) | 0.01 (0.6) | 0.11 (3.0) | -0.48 |
| zoo_ensemble/T | 0.25 | -12.62 | -1.8 | 0.01 (0.4) | 0.11 (2.1) | -0.02 (-0.6) | 0.02 (0.5) | -0.01 (-0.3) | -0.07 (-1.8) | -0.04 (-0.9) | 0.03 (1.3) | 0.08 (1.7) | -1.79 |
| zscore_reversion/L | 0.76 | -7.09 | -1.2 | -0.20 (-4.8) | 0.04 (1.1) | 0.01 (0.2) | -0.07 (-1.9) | -0.09 (-2.0) | 0.02 (1.0) | -0.02 (-0.9) | -0.04 (-3.2) | -0.36 (-6.8) | -2.08 |
| zscore_reversion/T | 0.17 | -8.58 | -1.7 | -0.01 (-0.4) | -0.02 (-0.9) | 0.01 (0.6) | 0.01 (0.3) | -0.03 (-0.9) | 0.08 (2.7) | -0.02 (-0.4) | 0.02 (0.7) | -0.01 (-0.3) | -2.56 |

- Sleeves with a Holm-significant alpha t > 2 in both segments: none.
- **Verdicts: llm_in_sample contradicts, clean_holdout contradicts.**

Factor returns (daily, NW t of the mean):

| factor | segment | n | mean | t | ann. Sharpe |
|---|---|---|---|---|---|
| MKT | llm_in_sample | 2254 | 0.0177 | 0.9 | 0.37 |
| TSMOM | llm_in_sample | 2244 | 0.0398 | 2.6 | 1.05 |
| VOL | llm_in_sample | 2254 | 0.0002 | 0.0 | 0.00 |
| CARRY | llm_in_sample | 1844 | 0.0193 | 2.0 | 0.89 |
| LIQ | llm_in_sample | 1899 | 0.0044 | 0.5 | 0.22 |
| GOLD | llm_in_sample | 495 | 0.0878 | 1.7 | 1.46 |
| OIL | llm_in_sample | 48 | -0.2108 | -1.9 | -5.01 |
| MKT | clean_holdout | 132 | 0.0619 | 0.7 | 1.41 |
| TSMOM | clean_holdout | 132 | -0.0042 | -0.1 | -0.14 |
| VOL | clean_holdout | 132 | 0.0026 | 0.1 | 0.07 |
| CARRY | clean_holdout | 132 | 0.0630 | 1.6 | 2.59 |
| LIQ | clean_holdout | 132 | -0.0244 | -0.6 | -0.96 |
| QQQ | clean_holdout | 132 | -0.0659 | -0.7 | -1.07 |
| GOLD | clean_holdout | 132 | 0.0203 | 0.2 | 0.35 |
| OIL | clean_holdout | 132 | 0.0528 | 0.6 | 0.98 |

### Families from hierarchical clustering of in-sample standardised loadings (k = 3)

| family | dominant loading | members | centre (std loadings MKT, TSMOM, VOL, CARRY, LIQ) |
|---|---|---|---|
| F2 | -TSMOM | cme_gap_fill, funding_contrarian, rsi_reversion, shock_regime, weekend_drift_fade, weekend_shock_follow, zscore_reversion | +0.04, -0.11, -0.04, +0.02, +0.01 |
| F1 | +TSMOM | compression_breakout, donchian_breakout, ema_stack, geo_flag_bo, geo_pivot_bo, geo_struct, tsmom, zoo_ensemble | -0.18, +0.55, +0.03, +0.01, +0.00 |
| F3 | +MKT | oi_price_divergence | +0.65, +0.05, +0.04, -0.02, +0.08 |

Context table by cluster family (universe L; shrunk mean pnl_d x100 per active symbol-day; grade as in the strategy map):

| family | cell | IS x100 | IS t | HO x100 | HO t | grade |
|---|---|---|---|---|---|---|
| F1 | R1 | -0.71 | -0.7 | -3.55 | -0.2 | C |
| F1 | R2 | -0.80 | 0.7 | -4.84 | -0.3 | C |
| F1 | R3 | 2.47 | 0.4 | 7.02 | 1.0 | C |
| F1 | R4 | -0.45 | -1.1 | -0.97 | -1.6 | C |
| F1 | R5 | -0.63 | 0.4 | -2.24 | -1.2 | C |
| F1 | R6 | 0.59 | -0.3 | -0.19 | -2.1 | - |
| F1 | R7 | 3.55 | 1.7 | -3.06 | -1.0 | - |
| F1 | R8 | 2.12 | 0.9 | -4.97 | -1.8 | - |
| F1 | R9 | -1.91 | -0.2 | -12.32 | -2.7 | C |
| F2 | R1 | -0.76 | -3.1 | -1.33 | -2.3 | A |
| F2 | R2 | -0.63 | -3.1 | -1.09 | -1.4 | B |
| F2 | R3 | -0.10 | -1.0 | -4.10 | -0.6 | C |
| F2 | R4 | -0.78 | -4.5 | -1.76 | -1.5 | B |
| F2 | R5 | -0.25 | -0.7 | -0.95 | -0.1 | C |
| F2 | R6 | -0.19 | -1.6 | -0.20 | -1.1 | C |
| F2 | R7 | -1.22 | -4.4 | -0.60 | -1.0 | C |
| F2 | R8 | -1.07 | -2.6 | 0.45 | 0.9 | - |
| F2 | R9 | -0.43 | -0.7 | 2.84 | 0.9 | - |
| F3 | R1 | -1.38 | -1.5 | -0.23 | -0.3 | C |
| F3 | R2 | 0.33 | 0.3 | -1.11 | -0.3 | - |
| F3 | R3 | -1.06 | -1.1 | 3.22 | 0.3 | - |
| F3 | R4 | -2.22 | -2.6 | 1.09 | 0.6 | - |
| F3 | R5 | -2.02 | -4.0 | 0.44 | 0.0 | - |
| F3 | R6 | -0.34 | -0.5 | 0.50 | 1.1 | - |
| F3 | R7 | -4.64 | -5.6 | -3.23 | -1.3 | B |
| F3 | R8 | -2.79 | -2.0 | 1.32 | 0.8 | - |
| F3 | R9 | -1.48 | -2.1 | 1.10 | -0.0 | - |

## H-PD3: does a conditional factor model (IPCA) rank strategies better than trailing Sharpe?

- 90512 strategy x symbol weeks, 368 units, 325 weeks (2020-08-17 to 2026-11-02), 56 instruments; first forecast after 52 training weeks; K chosen in-sample = **3**.

| K | segment | weeks | IC IPCA (t) | IC trailing Sharpe (t) | diff (NW t) | unit-level IC |
|---|---|---|---|---|---|---|
| 1 | llm_in_sample | 255 | -0.065 (-2.2) | -0.039 (-1.9) | -0.026 (-0.9) | -0.069 |
| 1 | clean_holdout | 18 | -0.159 (-1.6) | -0.185 (-2.3) | 0.026 (0.4) | -0.106 |
| 2 | llm_in_sample | 255 | 0.059 (1.8) | -0.039 (-1.9) | 0.098 (2.5) | 0.031 |
| 2 | clean_holdout | 18 | 0.136 (1.3) | -0.185 (-2.3) | 0.321 (1.8) | 0.101 |
| 3 | llm_in_sample | 255 | 0.097 (3.2) | -0.039 (-1.9) | 0.136 (3.3) | 0.078 |
| 3 | clean_holdout | 18 | 0.134 (1.2) | -0.185 (-2.3) | 0.319 (1.8) | 0.123 |

Extension (descriptive): chosen-K IPCA vs trailing-30d / trailing-365d sleeve Sharpe:

| segment | window | weeks | IC trailing Sharpe (t) | IPCA minus (NW t) |
|---|---|---|---|---|
| llm_in_sample | 30d | 255 | -0.032 (-1.3) | 0.129 (3.1) |
| llm_in_sample | 365d | 255 | -0.045 (-1.9) | 0.142 (2.8) |
| clean_holdout | 30d | 18 | 0.006 (0.0) | 0.128 (0.9) |
| clean_holdout | 365d | 18 | 0.015 (0.2) | 0.118 (0.8) |

- **Verdicts: llm_in_sample supports, clean_holdout supports.**

Model-implied expected weekly pnl_d by family under stylised contexts (full in-sample fit, chosen K; x100):

| context | trend | mean_reversion | volatility | positioning | calendar | shock |
|---|---|---|---|---|---|---|
| neutral | 189.41 | -76.02 | 1083.52 | -24.31 | -12.41 | 63.31 |
| trend_up_lowvol | -332.80 | 69.95 | -1253.33 | 32.78 | 9.51 | -89.08 |
| trend_up_highvol | -1149.95 | 315.08 | -253.13 | 27.07 | 10.80 | -93.34 |
| range_lowvol | 554.07 | -204.80 | 284.15 | -44.62 | -13.45 | 87.12 |
| range_highvol | -263.08 | 40.34 | 1284.35 | -50.33 | -12.16 | 82.86 |
| trend_down_highvol | 1483.83 | -151.59 | 3065.05 | -119.45 | -12.75 | 121.59 |
| trend_down_lowvol | 1580.25 | -520.00 | 561.91 | -44.34 | -18.42 | 190.15 |
| funding_hot | 213.99 | -123.35 | 1111.15 | -26.30 | -12.22 | 85.09 |
| vol_pct_high | 23.00 | -35.14 | 976.93 | -28.66 | -13.28 | 48.99 |

## H-PD4: where does the gross edge go? (trade level)

| segment | subset | trades | sum MFE | kept | exit leak | dir loss | costs | net | share (C+E)/MFE | alt (C+E_win)/kept | capture | hit gross |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| llm_in_sample | all | 143467 | 66845.2 | 23471.0 | 43374.2 | 20982.0 | 3229.0 | -739.9 | 0.697 | 1.181 | 0.351 | 0.380 |
| llm_in_sample | 1h | 132471 | 50548.8 | 17920.7 | 32628.1 | 16796.5 | 3021.1 | -1897.0 | 0.705 | 1.183 | 0.355 | 0.395 |
| llm_in_sample | 4h | 10996 | 16296.4 | 5550.4 | 10746.1 | 4185.5 | 207.8 | 1157.1 | 0.672 | 1.176 | 0.341 | 0.200 |
| clean_holdout | all | 16875 | 6490.6 | 2233.2 | 4257.4 | 2568.0 | 592.8 | -927.6 | 0.747 | 1.231 | 0.344 | 0.379 |
| clean_holdout | 1h | 15619 | 5418.4 | 1955.9 | 3462.5 | 2104.5 | 563.5 | -712.1 | 0.743 | 1.191 | 0.361 | 0.397 |
| clean_holdout | 4h | 1256 | 1072.1 | 277.3 | 794.9 | 463.5 | 29.3 | -215.5 | 0.769 | 1.514 | 0.259 | 0.166 |

- llm_in_sample: share 0.697, 95% CI [0.680, 0.713] (316 entry weeks); verdict **supports**.

- clean_holdout: share 0.747, 95% CI [0.686, 0.811] (20 entry weeks); verdict **supports**.

Per strategy (llm_in_sample | clean_holdout): share, capture, costs / kept gross:

| strategy | IS share | IS capture | IS cost/kept | HO share | HO capture | HO cost/kept |
|---|---|---|---|---|---|---|
| cme_gap_fill | 0.512 | 0.542 | 0.099 | 0.642 | 0.428 | 0.164 |
| compression_breakout | 1.028 | 0.004 | 7.854 | 1.133 | 0.003 | 40.442 |
| donchian_breakout | 0.701 | 0.335 | 0.105 | 0.696 | 0.368 | 0.173 |
| ema_stack | 0.700 | 0.334 | 0.103 | 0.763 | 0.304 | 0.221 |
| funding_contrarian | 0.607 | 0.494 | 0.203 | 0.619 | 0.563 | 0.323 |
| geo_flag_bo | 0.691 | 0.317 | 0.028 | 0.776 | 0.243 | 0.080 |
| geo_pivot_bo | 0.642 | 0.368 | 0.028 | 0.809 | 0.214 | 0.104 |
| geo_struct | 0.665 | 0.349 | 0.040 | 0.685 | 0.345 | 0.086 |
| oi_price_divergence | 0.659 | 0.439 | 0.222 | 0.625 | 0.480 | 0.219 |
| rsi_reversion | 0.622 | 0.604 | 0.374 | 0.762 | 0.561 | 0.576 |
| shock_regime | 0.623 | 0.428 | 0.121 | 0.631 | 0.469 | 0.214 |
| tsmom | 0.763 | 0.305 | 0.223 | 0.824 | 0.289 | 0.392 |
| weekend_drift_fade | 0.526 | 0.555 | 0.146 | 0.750 | 0.328 | 0.240 |
| weekend_shock_follow | 0.759 | 0.262 | 0.081 | 0.859 | 0.209 | 0.326 |
| zoo_ensemble | 0.699 | 0.318 | 0.053 | 0.777 | 0.257 | 0.133 |
| zscore_reversion | 0.452 | 0.655 | 0.164 | 0.471 | 0.680 | 0.222 |

## Multiple testing

- llm_in_sample: Holm over m = 19 PD tests; adjusted: H-PD4 0.0095, H-PD3 0.0103, H-PD1 1.0000
- clean_holdout: Holm over m = 19 PD tests; adjusted: H-PD4 0.0095, H-PD3 0.7112, H-PD1 1.0000

## Caveats

- Factor returns are built from the same 23 coins the strategies trade: a high R^2 on TSMOM says 'this is trend', not that an external risk premium was found. Survivorship: L is today's survivors.
- TradFi factors (QQQ, gold, oil) exist only from 2025-2026, so the in-sample attribution is crypto-only (PAXG gold from 2025 as robustness).
- IPCA characteristics are context of the symbol plus the strategy's own trailing Sharpe; the panel has 15-16 strategies, so a weekly cross-sectional IC is noisy (each IC uses ~16 points).
- MFE uses bar highs and lows (1h or 4h), so it is an upper bound on what any exit could capture; exit leakage is therefore an upper bound too.
- shock_regime is data-generated; calendar strategies trade rarely. T sleeves are holdout-only.

## Rule candidates

```yaml
- id: R-PD-1
  kind: expect
  rule: "WHEN judging how diversified the registered deterministic strategy book is THEN expect the equal-weight book to carry close to one bet (crypto trend), even though the strategy correlation matrix has about 8 eigen-bets"
  when: "equal-weight book of the 16 bar-runnable strategies on crypto universe L (or with the TradFi sleeves added)"
  then: "count trend variants as one exposure; adding strategies does not add independent risk unless they load on something other than TSMOM"
  scope: {assets: crypto, horizon: "1d", sides: both}
  effect: "registered ENB (eigenvalue entropy) 8.3 [7.7, 8.6] in-sample, 7.5 [6.1, 8.6] holdout L, 13.8 with T sleeves; top-3 share 0.61 / 0.65 L (H-PD1 contradicted: more than 6 eigen-bets); but Meucci ENB of the equal-weight book 1.15 in-sample, 2.5 holdout (L), 2.0 (L+T)"
  confidence: B
  evidence: [H-PD1, H-TR1, research/reports/pnl_decomposition_2026-11-11.md]
  review: "Meucci ENB of the book > 3 in a 6-month holdout window"
  status: provisional
- id: R-PD-2
  kind: expect
  rule: "WHEN a signal comes from the trend family (tsmom, ema_stack, donchian_breakout, geo_struct, geo_pivot_bo, geo_flag_bo, compression_breakout, the zoo ensemble) THEN read it as a crypto time-series-momentum bet that is also long convexity, not as an independent edge"
  when: "any trend-family strategy on crypto perps (cluster F1 of the PD report)"
  then: "expect its P&L to follow the TSMOM factor and big absolute market days; don't stack several trend variants as if they were separate opinions"
  scope: {assets: crypto, horizon: "1d", sides: both}
  effect: "standardised TSMOM loading 0.36-0.79 (t 7-20) in-sample; zoo ensemble 0.79 (t 17.6) in-sample, 0.77 (t 10.0) holdout, R^2 0.68 / 0.70; convexity extension: tsmom +0.41 (t 11.5) / +0.59 (t 6.9), ema_stack +0.28 / +0.47, donchian +0.28 / +0.48"
  confidence: A
  evidence: [H-PD2, H-PD1, H-TR1, research/reports/pnl_decomposition_2026-11-11.md]
  review: "a trend-family sleeve with TSMOM loading < 0.2 in a holdout half-year"
  status: provisional
- id: R-PD-3
  kind: expect
  rule: "WHEN running 1h mean reversion (zscore_reversion, rsi_reversion) THEN expect it to be short convexity: it loses on large absolute market days and when realised vol rises, so it does not hedge the trend book in stress"
  when: "1h fade strategies active on crypto perps"
  then: "expect losses concentrated on big |market| days and vol-up days; treat its negative correlation with trend as a short-straddle exposure, not as diversification"
  scope: {assets: crypto, horizon: "1d", sides: both}
  effect: "convexity extension (not registered): zscore_reversion CONVEX loading -0.51 (t -12.8) in-sample, -0.61 (t -6.8) holdout; rsi_reversion -0.38 (t -5.9) / -0.55 (t -7.3); VOL loading -0.20 (t -5 to -7) in-sample (registered fit)"
  confidence: B
  evidence: [H-PD2, research/reports/pnl_decomposition_2026-11-11.md]
  review: "re-fit after 6 more holdout months; the convexity factor is an extension, so register it before promoting"
  status: provisional
- id: R-PD-4
  kind: expect
  rule: "WHEN oi_price_divergence (rising open interest) gives a long tilt THEN read it as plain long crypto market beta, not an independent positioning signal"
  when: "oi_price_divergence active on crypto perps"
  then: "expect P&L to track the equal-weight crypto market return of the day"
  scope: {assets: crypto, horizon: "1d", sides: long}
  effect: "standardised MKT loading 0.65 (t 18.6) in-sample, 0.77 (t 6.6) holdout; R^2 0.41 / 0.61; alpha t -4.5 in-sample, -0.6 holdout"
  confidence: A
  evidence: [H-PD2, H-P2, research/reports/pnl_decomposition_2026-11-11.md]
  review: "MKT loading < 0.3 over a holdout half-year"
  status: provisional
- id: R-PD-5
  kind: expect
  rule: "WHEN a registered strategy shows a positive raw Sharpe THEN expect it to be factor exposure (TSMOM, market, carry) plus noise, not residual alpha; observable factors explain only about a quarter of daily strategy variance"
  when: "any registered strategy sleeve, daily P&L"
  then: "don't credit a strategy with alpha from its raw track record; no sleeve has a Holm-significant positive alpha in both segments"
  scope: {assets: all, horizon: "1d", sides: both}
  effect: "median R^2 0.25 in-sample (5 crypto factors), 0.22 holdout (8 factors), 0.36 / 0.30 with the convexity extension (H-PD2 contradicted on R^2 < 0.4); largest in-sample alpha shock_regime t 2.5 (data-generated), holdout t -1.5"
  confidence: B
  evidence: [H-PD2, research/reports/pnl_decomposition_2026-11-11.md]
  review: "any sleeve with Holm-significant alpha t > 2 in a new holdout year"
  status: provisional
- id: R-PD-6
  kind: prefer
  rule: "WHEN looking for improvement in the directional strategies THEN prefer work on exits over work on entries or fees: trades give back about two thirds of their favourable excursion, while taker costs take about 5% of it"
  when: "directional registered strategies at 1h and 4h (trade = maximal same-sign run)"
  then: "prioritise exit design (targets, trailing rules, time-in-trade) and maker execution only second; direction is right on about 38% of trades by gross P&L"
  scope: {assets: all, horizon: "1h-1w", sides: both}
  effect: "(costs + exit leakage) / sum MFE 0.70 [0.68, 0.71] in-sample (143k trades), 0.75 [0.69, 0.81] holdout (17k); capture of MFE 35% / 34%; costs 4.8% / 9.1% of MFE (14% / 27% of kept gross); 1h 0.71 / 0.74, 4h 0.67 / 0.77"
  confidence: B
  evidence: [H-PD4, research/reports/pnl_decomposition_2026-11-11.md]
  review: "MFE from bar highs/lows is an upper bound, so leakage is overstated; confirm with a registered exit-rule test before acting"
  status: provisional
- id: R-PD-7
  kind: prefer
  rule: "WHEN ranking registered strategy types for next week THEN prefer a context-conditional ranking (IPCA on regime shares, vol percentile, funding z, trend strength, drawdown) over last-quarter performance; trailing Sharpe ranks have negative information week to week"
  when: "weekly cross-strategy ranking on crypto universe L"
  then: "don't promote a strategy because its trailing 30-365d Sharpe is high; strategy performance mean-reverts at the weekly horizon; the conditional ranking is weak but positive"
  scope: {assets: crypto, horizon: "1w", sides: both}
  effect: "IPCA (K=3, chosen in-sample) IC 0.10 (t 3.2) in-sample, 0.13 (t 1.2, 18 weeks) holdout; trailing-90d Sharpe IC -0.04 (t -1.9) / -0.19 (t -2.3); difference +0.14 (t 3.3, Holm p 0.01) / +0.32 (t 1.8)"
  confidence: C
  evidence: [H-PD3, research/reports/pnl_decomposition_2026-11-11.md]
  review: "holdout IPCA IC > 0 with t > 2 after 52 weeks; K selection among 3 in-sample variants"
  status: provisional
```
