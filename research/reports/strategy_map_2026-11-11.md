# Strategy map by regime (H-RC1, H-RC2) and the context-ranking table — 2026-11-11

Agent B1, Program 2 ([plan](../plans/2026-11-11_strategy_program.md)). Hypotheses registered 2026-11-11 before testing.

## Data and methods

- Panel `data/priors/strategy_panel_daily.parquet` (692705 rows; built by `ai_trader.research.strategy_panel`), as of 2026-11-10T23:37:59.166818+00:00.
- Universe L: 23 crypto perps; universe T: 15 TradFi perps with >= 120 days (XAUUSDT, XAGUSDT, CLUSDT, BZUSDT, TSLAUSDT, INTCUSDT, GOOGLUSDT, AAPLUSDT, MSTRUSDT, NVDAUSDT, MUUSDT, QQQUSDT, SOXLUSDT, PAXGUSDT, XAUTUSDT); skipped symbols: {'SPCXUSDT': '118 days < 120'}. 4h built from 1h for: TSLAUSDT, INTCUSDT, GOOGLUSDT, AAPLUSDT, MSTRUSDT, NVDAUSDT, MUUSDT, QQQUSDT, SOXLUSDT.
- Strategies (L): cme_gap_fill, compression_breakout, donchian_breakout, ema_stack, funding_contrarian, geo_flag_bo, geo_pivot_bo, geo_struct, oi_price_divergence, rsi_reversion, shock_regime, tsmom, weekend_drift_fade, weekend_shock_follow, zoo_ensemble, zscore_reversion. Skipped: flow_momentum (needs flow_bars: only BTCUSDT from 2026-05-11 in the lake), cvd_divergence (needs flow_bars: only BTCUSDT from 2026-05-11 in the lake); on T also oi_price_divergence (Bybit TradFi open interest only from 2026-11-05).
- 1h-parameterised strategies on 1h bars; geo_* and zoo_ensemble on 4h. evaluate.strategy_pnl (risk units, sized at entry, 7.5 bps per side). Daily P&L dated by the start of each holding interval (UTC 00-24h); `pnl_d` = daily sum / sqrt(bars per day) (daily-sigma units, scale-free across frequencies). Regime = v0 cell of the symbol at 00:00 UTC (1h labeller).
- by_regime: empirical-Bayes shrinkage kappa = 10 pseudo-episodes toward the strategy's pooled active mean; effective n = episodes.
- Segments: llm_in_sample < 2026-07-01 <= clean_holdout. `shock_regime` is data-generated: shown, never evidence.

## H-RC1: are per-regime rankings stable across the cutoff?

- Pairs (strategy x cell, >= 5 episodes in both segments): **144** over 16 strategies.
- Spearman(in-sample shrunk mean, holdout shrunk mean) = **0.085**, 95% CI [-0.108, 0.210] (episode bootstrap, 2000 draws), one-sided p 0.2719, Holm p 0.2719.
- Sign agreement over pairs: 0.528. Within-strategy (strategy level removed) Spearman: 0.133: does the *cell ordering inside a strategy* persist?
- **Verdict: inconclusive** (recorded once, segment pooled).

| strategy | cells | within-strategy Spearman |
|---|---|---|
| cme_gap_fill | 9 | 0.650 |
| compression_breakout | 9 | 0.683 |
| donchian_breakout | 9 | 0.450 |
| ema_stack | 9 | 0.300 |
| funding_contrarian | 9 | 0.083 |
| geo_flag_bo | 9 | -0.483 |
| geo_pivot_bo | 9 | 0.067 |
| geo_struct | 9 | 0.000 |
| oi_price_divergence | 9 | 0.000 |
| rsi_reversion | 9 | 0.583 |
| shock_regime | 9 | 0.033 |
| tsmom | 9 | -0.033 |
| weekend_drift_fade | 9 | 0.167 |
| weekend_shock_follow | 9 | -0.209 |
| zoo_ensemble | 9 | 0.450 |
| zscore_reversion | 9 | -0.417 |

## H-RC2: does a regime-conditional allocator beat equal weight out of sample?

- Unconditional set (in-sample pooled mean > 0): donchian_breakout, ema_stack, geo_flag_bo, geo_pivot_bo, geo_struct, shock_regime, zoo_ensemble.
- Conditional sets per cell (in-sample shrunk mean > 0): R1: ema_stack, shock_regime, weekend_drift_fade, weekend_shock_follow; R2: funding_contrarian, geo_pivot_bo, oi_price_divergence, weekend_shock_follow, zoo_ensemble; R3: cme_gap_fill, donchian_breakout, ema_stack, funding_contrarian, geo_flag_bo, geo_pivot_bo, geo_struct, weekend_drift_fade, zoo_ensemble; R4: compression_breakout, donchian_breakout, ema_stack, geo_pivot_bo, shock_regime, weekend_shock_follow, zoo_ensemble; R5: geo_pivot_bo, geo_struct, shock_regime, zoo_ensemble; R6: donchian_breakout, geo_flag_bo, geo_pivot_bo, geo_struct, shock_regime, weekend_drift_fade, zoo_ensemble; R7: compression_breakout, donchian_breakout, ema_stack, geo_flag_bo, geo_pivot_bo, geo_struct, shock_regime, tsmom, weekend_drift_fade, zoo_ensemble; R8: compression_breakout, donchian_breakout, ema_stack, geo_flag_bo, geo_pivot_bo, geo_struct, shock_regime, tsmom, weekend_shock_follow, zoo_ensemble; R9: funding_contrarian, shock_regime

| segment | days | Sharpe cond | Sharpe EW | Sharpe uncond | cond-EW [95% CI] | cond-uncond [95% CI] |
|---|---|---|---|---|---|---|
| llm_in_sample | 2184 | 1.46 | 0.68 | 0.92 | 0.78 [0.45, 1.14] | 0.54 [0.27, 0.85] |
| clean_holdout | 132 | -0.77 | -2.48 | -1.56 | 1.71 [0.78, 2.65] | 0.79 [0.06, 1.99] |

Sensitivity (not the registered test): without the data-generated shock_regime (its design saw holdout evidence):

- llm_in_sample: Sharpe cond 1.48, EW 0.66, uncond 0.90; cond-EW 0.82 [0.46, 1.20], cond-uncond 0.58 [0.28, 0.92]
- clean_holdout: Sharpe cond -0.80, EW -2.40, uncond -1.49; cond-EW 1.60 [0.69, 2.63], cond-uncond 0.69 [-0.19, 2.11]
- H-RC1 Spearman without shock_regime pairs: 0.084.

- Holm (RC block): {"H-RC2_vs_eq": 0.006, "H-RC2_vs_uncond": 0.038, "H-RC1": 0.2719}.
- **Verdict: clean_holdout supports**; llm_in_sample inconclusive by construction (selection in-sample).

## Context-ranking table (family x v0 cell)

Shrunk mean net P&L per active symbol-day in daily-sigma units (x100 = % of one daily sigma), in-sample | holdout, with date-clustered NW t and grade. A: same sign, t >= 2 in both segments; B: same sign, t >= 2 in-sample and >= 1 holdout; C: same sign otherwise; '-': signs disagree. t = NW t of the per-date mean over symbols (clustered by time), counted in the direction of the shrunk mean.

| family | cell | IS shrunk x100 | IS t | IS eps | HO shrunk x100 | HO t | HO eps | sign agree | grade |
|---|---|---|---|---|---|---|---|---|---|
| calendar | R1 quiet trend up | 0.38 | 0.4 | 2079 | -0.81 | n/a | 128 | no | **-** |
| calendar | R2 trend up | -0.86 | 0.1 | 2195 | -1.30 | -0.4 | 167 | yes | **C** |
| calendar | R3 momentum burst | 1.14 | 0.7 | 1832 | 1.21 | 1.1 | 183 | yes | **C** |
| calendar | R4 compression | -3.15 | -2.3 | 4340 | -2.93 | -2.0 | 336 | yes | **B** |
| calendar | R5 balanced range | -2.23 | -1.4 | 4429 | -1.37 | n/a | 342 | yes | **C** |
| calendar | R6 whipsaw | 0.63 | 0.2 | 3533 | 0.70 | n/a | 321 | yes | **C** |
| calendar | R7 grind down | -0.40 | -2.5 | 2513 | 0.09 | 2.0 | 210 | no | **-** |
| calendar | R8 trend down | -2.32 | -1.0 | 2422 | -1.52 | n/a | 231 | yes | **C** |
| calendar | R9 liquidation | -3.46 | -1.6 | 2175 | 3.69 | n/a | 155 | no | **-** |
| mean_reversion | R1 quiet trend up | -3.42 | -3.2 | 2079 | -3.46 | -1.7 | 128 | yes | **B** |
| mean_reversion | R2 trend up | -2.02 | -2.8 | 2195 | -2.50 | -1.1 | 167 | yes | **B** |
| mean_reversion | R3 momentum burst | -0.56 | -0.7 | 1832 | -17.35 | -0.9 | 183 | yes | **C** |
| mean_reversion | R4 compression | -3.02 | -2.8 | 4340 | -8.01 | -1.3 | 336 | yes | **B** |
| mean_reversion | R5 balanced range | -0.53 | 0.5 | 4429 | -1.36 | 0.5 | 342 | yes | **C** |
| mean_reversion | R6 whipsaw | -1.12 | -1.7 | 3533 | -1.93 | -0.8 | 321 | yes | **C** |
| mean_reversion | R7 grind down | -6.11 | -3.7 | 2513 | -1.04 | -0.7 | 210 | yes | **C** |
| mean_reversion | R8 trend down | -3.91 | -2.4 | 2422 | 2.59 | 0.5 | 231 | no | **-** |
| mean_reversion | R9 liquidation | -1.96 | -0.5 | 2175 | 8.62 | 0.5 | 155 | no | **-** |
| positioning | R1 quiet trend up | -0.77 | -2.1 | 2079 | -0.24 | -0.3 | 128 | yes | **C** |
| positioning | R2 trend up | 0.34 | 0.6 | 2195 | -0.48 | -0.1 | 167 | no | **-** |
| positioning | R3 momentum burst | -0.30 | -1.4 | 1832 | 1.46 | 0.3 | 183 | no | **-** |
| positioning | R4 compression | -1.58 | -3.3 | 4340 | 0.91 | 0.8 | 336 | no | **-** |
| positioning | R5 balanced range | -1.44 | -3.3 | 4429 | -1.63 | -1.3 | 342 | yes | **B** |
| positioning | R6 whipsaw | -0.71 | -1.2 | 3533 | 0.32 | 1.0 | 321 | no | **-** |
| positioning | R7 grind down | -2.12 | -4.4 | 2513 | -1.34 | -1.2 | 210 | yes | **B** |
| positioning | R8 trend down | -1.85 | -2.2 | 2422 | 1.38 | 1.9 | 231 | no | **-** |
| positioning | R9 liquidation | 0.47 | 0.1 | 2175 | 2.60 | -0.1 | 155 | yes | **C** |
| shock | R1 quiet trend up | 2.40 | 0.5 | 2079 | -7.38 | -1.8 | 128 | no | **-** |
| shock | R2 trend up | -2.25 | -3.3 | 2195 | -7.28 | -2.4 | 167 | yes | **A** |
| shock | R3 momentum burst | -1.12 | -0.4 | 1832 | 5.86 | 2.4 | 183 | no | **-** |
| shock | R4 compression | 2.23 | 0.2 | 4340 | 2.35 | 0.0 | 336 | yes | **C** |
| shock | R5 balanced range | 0.29 | -0.6 | 4429 | -5.73 | 0.0 | 342 | no | **-** |
| shock | R6 whipsaw | 3.13 | 0.4 | 3533 | 3.20 | 0.6 | 321 | yes | **C** |
| shock | R7 grind down | 5.19 | 1.2 | 2513 | -5.28 | -0.2 | 210 | no | **-** |
| shock | R8 trend down | 2.10 | 0.0 | 2422 | -3.72 | -0.5 | 231 | no | **-** |
| shock | R9 liquidation | 1.49 | -0.6 | 2175 | 0.58 | 0.7 | 155 | yes | **C** |
| trend | R1 quiet trend up | -0.53 | -0.1 | 2079 | -3.64 | -0.1 | 128 | yes | **C** |
| trend | R2 trend up | -0.62 | 0.8 | 2195 | -5.28 | -0.3 | 167 | yes | **C** |
| trend | R3 momentum burst | 2.89 | 0.5 | 1832 | 8.02 | 0.7 | 183 | yes | **C** |
| trend | R4 compression | -0.51 | -0.7 | 4340 | -1.21 | -2.0 | 336 | yes | **C** |
| trend | R5 balanced range | -0.70 | 0.1 | 4429 | -3.10 | -2.1 | 342 | yes | **C** |
| trend | R6 whipsaw | 0.72 | -0.7 | 3533 | -0.19 | -2.3 | 321 | no | **-** |
| trend | R7 grind down | 3.80 | 2.0 | 2513 | -4.00 | -1.3 | 210 | no | **-** |
| trend | R8 trend down | 2.28 | 1.2 | 2422 | -5.52 | -1.8 | 231 | no | **-** |
| trend | R9 liquidation | -1.99 | -0.2 | 2175 | -11.18 | -2.0 | 155 | yes | **C** |
| volatility | R1 quiet trend up | -2.15 | -2.0 | 2079 | -3.10 | -0.9 | 128 | yes | **C** |
| volatility | R2 trend up | -2.50 | -0.1 | 2195 | -1.99 | 0.1 | 167 | yes | **C** |
| volatility | R3 momentum burst | -0.64 | -0.4 | 1832 | 0.01 | 1.2 | 183 | no | **-** |
| volatility | R4 compression | 0.00 | -1.3 | 4340 | 0.81 | 0.8 | 336 | yes | **C** |
| volatility | R5 balanced range | -0.17 | 1.0 | 4429 | 4.33 | 0.5 | 342 | no | **-** |
| volatility | R6 whipsaw | -0.41 | 0.7 | 3533 | -0.22 | 0.6 | 321 | yes | **C** |
| volatility | R7 grind down | 1.98 | -0.2 | 2513 | 3.71 | 0.7 | 210 | yes | **C** |
| volatility | R8 trend down | 1.23 | -0.5 | 2422 | -1.27 | -0.7 | 231 | no | **-** |
| volatility | R9 liquidation | -1.72 | -0.4 | 2175 | -22.04 | -2.7 | 155 | yes | **C** |

### Universe T (TradFi perps, holdout only, robustness)

| family | cell | HO shrunk x100 | HO t | HO eps |
|---|---|---|---|---|
| calendar | R1 | 6.21 | n/a | 101 |
| calendar | R2 | -3.96 | n/a | 84 |
| calendar | R3 | -3.96 | n/a | 51 |
| calendar | R4 | -5.65 | n/a | 147 |
| calendar | R5 | -3.74 | n/a | 172 |
| calendar | R6 | -3.96 | n/a | 100 |
| calendar | R7 | -2.89 | n/a | 100 |
| calendar | R8 | -3.96 | n/a | 102 |
| calendar | R9 | -10.02 | n/a | 79 |
| mean_reversion | R1 | -7.80 | -2.0 | 101 |
| mean_reversion | R2 | -0.56 | 1.0 | 84 |
| mean_reversion | R3 | 0.00 | 1.0 | 51 |
| mean_reversion | R4 | -11.68 | -2.2 | 147 |
| mean_reversion | R5 | -4.49 | -0.2 | 172 |
| mean_reversion | R6 | -2.75 | -0.8 | 100 |
| mean_reversion | R7 | -3.00 | -0.5 | 100 |
| mean_reversion | R8 | 0.16 | -0.5 | 102 |
| mean_reversion | R9 | 0.48 | 0.8 | 79 |
| positioning | R1 | -28.09 | -2.2 | 101 |
| positioning | R2 | 0.05 | n/a | 84 |
| positioning | R3 | -25.67 | n/a | 51 |
| positioning | R4 | -6.54 | -0.9 | 147 |
| positioning | R5 | -9.35 | -1.1 | 172 |
| positioning | R6 | 2.38 | 0.5 | 100 |
| positioning | R7 | -16.08 | n/a | 100 |
| positioning | R8 | 8.37 | n/a | 102 |
| positioning | R9 | -13.14 | n/a | 79 |
| shock | R1 | -2.81 | -1.6 | 101 |
| shock | R2 | -2.76 | -0.4 | 84 |
| shock | R3 | -4.79 | -1.1 | 51 |
| shock | R4 | 9.04 | 1.2 | 147 |
| shock | R5 | -6.11 | -1.8 | 172 |
| shock | R6 | -9.94 | -2.4 | 100 |
| shock | R7 | 1.44 | -0.1 | 100 |
| shock | R8 | -9.75 | -1.7 | 102 |
| shock | R9 | -9.82 | -1.6 | 79 |
| trend | R1 | 0.79 | -0.1 | 101 |
| trend | R2 | -6.61 | -1.3 | 84 |
| trend | R3 | -5.87 | -2.1 | 51 |
| trend | R4 | -0.52 | -0.8 | 147 |
| trend | R5 | -0.17 | -0.7 | 172 |
| trend | R6 | -3.25 | -0.4 | 100 |
| trend | R7 | 0.40 | -0.3 | 100 |
| trend | R8 | -6.39 | -0.0 | 102 |
| trend | R9 | -2.42 | -1.7 | 79 |
| volatility | R1 | -9.10 | -1.3 | 101 |
| volatility | R2 | -0.76 | -1.3 | 84 |
| volatility | R3 | -6.18 | -0.8 | 51 |
| volatility | R4 | 4.85 | 0.4 | 147 |
| volatility | R5 | 2.65 | 0.0 | 172 |
| volatility | R6 | 1.68 | -0.3 | 100 |
| volatility | R7 | -13.81 | -0.6 | 100 |
| volatility | R8 | -1.04 | 0.4 | 102 |
| volatility | R9 | 33.34 | 2.3 | 79 |

### Per-strategy cells graded A or B (universe L)

| strategy | cell | IS shrunk x100 | IS t | HO shrunk x100 | HO t | grade |
|---|---|---|---|---|---|---|
| cme_gap_fill | R4 | -7.04 | -2.3 | -7.47 | -1.6 | B |
| ema_stack | R5 | -0.62 | -2.4 | -0.66 | -1.4 | B |
| ema_stack | R6 | -0.66 | -2.0 | -2.01 | -2.3 | A |
| geo_struct | R4 | -2.56 | -3.0 | -2.58 | -2.6 | A |
| oi_price_divergence | R7 | -4.64 | -5.6 | -3.23 | -1.3 | B |
| rsi_reversion | R1 | -2.09 | -4.0 | -2.92 | -1.7 | B |
| rsi_reversion | R4 | -1.67 | -2.5 | -4.13 | -2.2 | A |
| rsi_reversion | R7 | -3.72 | -3.3 | -2.59 | -2.1 | A |
| shock_regime | R2 | -2.25 | -3.3 | -7.28 | -2.4 | A |
| tsmom | R5 | -3.92 | -4.4 | -0.91 | -1.8 | B |
| tsmom | R6 | -3.56 | -3.4 | -3.61 | -1.6 | B |
| weekend_drift_fade | R4 | -6.17 | -2.3 | -2.35 | -1.7 | B |
| zscore_reversion | R2 | -3.77 | -2.4 | -7.66 | -1.2 | B |

## Caveats

- Universe L is today's survivors (survivorship bias in any cross-sectional statement).
- The holdout is ~4 months: holdout cell means rest on a handful of episodes per symbol, pooled across 23 co-moving coins.
- Cells and strategies are both functions of past prices, so a strategy's P&L by cell partly restates its construction (trend strategies are long in up cells by design); the question is whether the *net* P&L ordering persists.
- 1h strategy parameters are the pre-registered config values, not tuned; families with one member (volatility, shock) are single strategies.
- shock_regime is data-generated (H-S2/H-S3 seen in both segments) and is not evidence; the calendar strategies trade only around weekends.

## Rule candidates

```yaml
- id: R-MAP-1
  kind: expect
  rule: "WHEN choosing a strategy because of the current v0 regime cell THEN don't expect the in-sample per-cell ranking of the registered strategies to carry over; the regime map is not a reliable selector of winners"
  when: "any v0 cell (R1-R9) on crypto universe L; strategy x cell shrunk mean net P&L from research/reports/strategy_map_2026-11-11.md"
  then: "treat per-cell 'this strategy pays here' claims as unproven; no family x cell has a positive net mean graded A or B; only negative cells (where strategies lose) are stable"
  scope: {assets: crypto, horizon: "1d", sides: both}
  effect: "Spearman of in-sample vs holdout shrunk cell means 0.09, 95% CI [-0.11, 0.21] (144 pairs, 16 strategies, episode bootstrap); sign agreement 53%; within-strategy cell ordering 0.13"
  confidence: B
  evidence: [H-RC1, research/reports/strategy_map_2026-11-11.md]
  review: "re-test when the holdout reaches 12 months; becomes A-negative if the CI stays around 0, retired if Spearman > 0 with CI excluding 0"
  status: provisional
- id: R-MAP-2
  kind: prefer
  rule: "WHEN combining the registered deterministic strategies THEN prefer dropping, per v0 cell, the strategies whose in-sample shrunk cell mean is negative over holding all of them equally, as loss avoidance rather than an edge"
  when: "book of the 16 bar-runnable strategies on crypto universe L; start-of-day v0 cell per symbol"
  then: "expect the cell-filtered book to lose less than the equal-weight book; don't expect it to make money (all three books lost in Jul-Nov 2026)"
  scope: {assets: crypto, horizon: "1d", sides: both}
  effect: "holdout annualised Sharpe -0.77 conditional vs -2.48 equal weight (diff +1.71, CI [0.78, 2.65], Holm p 0.006) and -1.56 unconditional (diff +0.79, CI [0.06, 1.99], Holm p 0.04); without shock_regime +1.60 [0.69, 2.63] vs EW but +0.69 [-0.19, 2.11] vs unconditional; books correlate 0.94"
  confidence: C
  evidence: [H-RC2, research/reports/strategy_map_2026-11-11.md]
  review: "re-test at 12 holdout months; the gain vs the unconditional selection must survive without shock_regime"
  status: provisional
- id: R-MAP-3
  kind: avoid
  rule: "WHEN the v0 cell is R1 (quiet trend up), R2 (trend up), R4 (compression) or R7 (grind down) THEN don't run 1h mean reversion (zscore_reversion, rsi_reversion); it loses net in both segments"
  when: "start-of-day v0 cell in {R1, R2, R4, R7}; 1h fades of 2-sigma EMA deviations or RSI 30/70"
  then: "expect negative net P&L from fading; consistent with R-PX-2 (don't fade hourly deviations)"
  scope: {assets: crypto, horizon: "1h-1d", sides: both}
  effect: "family shrunk mean x100 daily-sigma per active symbol-day: R1 -3.4 / -3.5 (IS t -3.2, HO t -1.7), R2 -2.0 / -2.5, R4 -3.0 / -8.0; rsi_reversion R4 -1.7 / -4.1 (t -2.5 / -2.2, grade A), R7 -3.7 / -2.6 (t -3.3 / -2.1, A)"
  confidence: B
  evidence: [H-RC1, H-T1, research/reports/strategy_map_2026-11-11.md]
  review: "a holdout cell mean > 0 with t > 1 in two consecutive quarters"
  status: provisional
- id: R-MAP-4
  kind: avoid
  rule: "WHEN the v0 cell is R5 (balanced range) or R6 (whipsaw) THEN don't expect 1h trend-following (tsmom, ema_stack) to pay; it loses net in both segments"
  when: "start-of-day v0 cell in {R5, R6}; 1h-parameterised trend signals"
  then: "expect negative net P&L from 1h trend entries in range cells"
  scope: {assets: crypto, horizon: "1h-1d", sides: both}
  effect: "ema_stack R6 -0.66 / -2.01 x100 (t -2.0 / -2.3, A), R5 -0.62 / -0.66 (B); tsmom R5 -3.9 / -0.9 (t -4.4 / -1.8, B), R6 -3.6 / -3.6 (t -3.4 / -1.6, B)"
  confidence: B
  evidence: [H-RC1, research/reports/strategy_map_2026-11-11.md]
  review: "holdout t > 0 for either strategy in R5/R6 over 12 months"
  status: provisional
- id: R-MAP-5
  kind: avoid
  rule: "WHEN the v0 cell is R5 (balanced range) or R7 (grind down) THEN don't expect funding-contrarian or OI-divergence positioning signals to pay"
  when: "start-of-day v0 cell in {R5, R7}; funding_contrarian or oi_price_divergence active"
  then: "expect negative net P&L (in line with R-AVOID-1: these signals fail overall)"
  scope: {assets: crypto, horizon: "1d", sides: both}
  effect: "positioning family x100: R5 -1.44 / -1.63 (t -3.3 / -1.3), R7 -2.12 / -1.34 (t -4.4 / -1.2); oi_price_divergence R7 -4.6 / -3.2 (t -5.6 / -1.3)"
  confidence: B
  evidence: [H-RC1, H-P1, H-P2, research/reports/strategy_map_2026-11-11.md]
  review: "holdout cell mean > 0 with t > 1"
  status: provisional
- id: R-MAP-6
  kind: avoid
  rule: "WHEN the v0 cell is R4 (compression) THEN don't expect swing-structure trend (geo_struct) or weekend fades (cme_gap_fill, weekend_drift_fade) to pay"
  when: "start-of-day v0 cell R4"
  then: "expect negative net P&L from geo_struct entries and from fading the weekend move at the CME reopen"
  scope: {assets: crypto, horizon: "1d-3d", sides: both}
  effect: "geo_struct R4 -2.56 / -2.58 x100 (t -3.0 / -2.6, A); calendar family R4 -3.2 / -2.9 (t -2.3 / -2.0, B); cme_gap_fill R4 -7.0 / -7.5 (B)"
  confidence: B
  evidence: [H-RC1, H-GT1, H-W4, research/reports/strategy_map_2026-11-11.md]
  review: "holdout cell mean > 0 with t > 1"
  status: provisional
- id: R-MAP-7
  kind: avoid
  rule: "WHEN shock_regime fires its reversal leg in v0 cell R2 (trend up, normal vol) THEN don't expect the fade to pay"
  when: "shock_regime signal with vol_state normal and start-of-day cell R2"
  then: "expect negative net P&L; a candidate veto for the forward test (shock_regime is data-generated, so this is context only)"
  scope: {assets: crypto, horizon: "1h-1d", sides: both}
  effect: "shock_regime R2 -2.25 / -7.28 x100 (t -3.3 / -2.4); R2 was outside the cells whose evidence shaped the design"
  confidence: C
  evidence: [H-RC1, H-S2, research/reports/strategy_map_2026-11-11.md]
  review: "forward R2 shock trades after 2026-11-08 (H-S3 record)"
  status: provisional
```
