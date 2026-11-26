# Stress test of every deterministic strategy in every scenario, 2026-11-26

Agent STRESS. Research only: no orders, no venue calls. The LLM decision layer is not tested (CLAUDE.md principle 1); these are the deterministic signal generators, the desk setups as coded or specified, and the live risk engine limits (config/risk.yaml `live_small`) used as the sizing layer. Design and verdict rules were registered before any run (research/hypotheses.yaml block STRESS, H-STRESS0..10; diary 2026-11-25 "STRESS design frozen before results"). Data as of 2026-11-25T23:07 UTC. R = one exchange-stop loss including round-trip costs; 1 R = 1.30 USD live.

## In plain words

**Only one unit passes every registered stress: `zoo_ensemble` (the 4h trend-model zoo).** Everything else either loses money outright (FAILS) or has a positive average that some stress, the clean holdout or plain sampling noise can erase (FRAGILE). `geo_pivot_bo` also passes, but only because 16 of its positions are still open today and are marked at the last close; on closed trades its holdout is -0.16 R and it is FRAGILE. SURVIVES means "passed the screens", not "has a proven edge": even zoo_ensemble's closed-trade holdout is a modest +0.10 R per trade on 131 trades.

| verdict | units |
|---|---|
| **SURVIVES** | zoo_ensemble; geo_pivot_bo (only with today's open positions marked to market) |
| **FRAGILE** | tsmom, ema_stack, donchian_breakout, trend_4h, geo_struct, geo_flag_bo (positive, CI includes 0); shock_regime (holdout < 0); hrb2 (CI includes 0, negative on the D8 majors); compression_breakout (only through 17 open positions; on closed trades it FAILS at -0.95 R); flow_momentum (186 trades, BTC only); cvd_divergence (1 trade: no information) |
| **FAILS** | zscore_reversion, rsi_reversion, funding_contrarian, oi_price_divergence, weekend_shock_follow, weekend_drift_fade, cme_gap_fill, shock_desk (the paper shock sleeve), hs2_fade_blackout (H-S2 fade with the R-FLOW-2 blackout), setup_c_proxy |

**Why the trend family is fragile, not failed.** Every trend unit is positive in both segments, but the 1h versions earn only +0.03 to +0.09 R per trade, so their 95% intervals include zero, and +5 bp per side (tsmom) or +10 bp (ema_stack) wipes them out; the liquidity drought (spread x3 and wider stop slippage on mid-tier alts) turns all three 1h trend units negative. The 4h versions (zoo_ensemble, geo_*, trend_4h) earn more per trade and stay positive even at +25 bp per side. No trend unit has a parameter cliff: every +-20% change and all 30 joint draws keep the D8 mean positive; at +-50% only geo_struct turns negative. A 5-minute delay costs less than 0.02 R per trade for every unit except `shock_desk` (-0.036 R).

**The desk sleeves.**
- `shock_desk` (the paper shock sleeve with its structural bracket) **fails**: -0.003 R overall, -0.14 R in the holdout, and the exact 5m fills on BTC are worse than the model at 5 minutes. The registered strategy `shock_regime` (its own trailing / invalidation exits) is +0.06 R overall but -0.03 R in the holdout. Both are data-generated, so the history is not evidence either way, and the forward record is what counts.
- `hs2_fade_blackout` **fails** (-0.08 R): waiting the R-FLOW-2 hour before fading a shock leaves nothing to fade.
- `trend_4h` is positive (+0.10 R, holdout +0.43 R on 266 closed trades) and cost-robust, but its interval includes 0.
- `hrb2` (short the failed post-flush rebound) is +0.05 R on 23 symbols but -0.03 R on the 8 majors, with only 30 holdout trades.
- `setup_c_proxy` **fails** (-0.11 R): the bar skeleton of setup C (sweep, wait an hour, reclaim) loses. Its two order-flow confirmations could not be tested, so this does not rule out the user's setup, only the version without them.
- Weekend MM quoters (replay only): at most a few cents per symbol-day, and only the H-MMW2 BTC-pull quoter (+0.7 to +0.9 cents) is above zero in all four queue / latency cells; the join-short-queues variant loses 1.1 to 1.25 USD per symbol-day in every cell.

**What to switch off where (suggestions for the rules owner; research/rules.yaml was not edited).**
- All mean-reversion and calendar units, everywhere: they lose in every regime cell and in almost every crisis window.
- `shock_desk` in R8 (trend down, -0.27 R, both segments) and R5 (balanced range, -0.19 R); `hs2_fade_blackout` in R8 (-0.24 R); `shock_regime` in R8 / R9.
- `tsmom` and `donchian_breakout` in R6 (whipsaw: -0.11 / -0.15 R, both segments); `ema_stack` in R9.
- 1h trend entries during the TradFi weekend: tsmom -0.11 R vs +0.08 R on weekdays, donchian -0.12 vs +0.13, zoo_ensemble -0.20 vs +0.48. This supports keeping the S4 weekend block that trend_4h already has (a slice, not a registered test).

**What the live risk limits imply.**
- **The 10 USD daily stop is practically never hit** (0 of the active days for every unit). With 3 open positions at 1.30 USD each, the worst day of any unit was -9.09 USD (zscore_reversion) and -7.21 USD for tsmom; the daily budget rule refused an entry on at most 0.17% of days.
- **Losing runs are certain; ruin is not.** Trend units win 25-35% of trades, so a run of 5 losers happens in essentially every 3-month window (tsmom's longest run is 31 trades, zoo_ensemble's 12). Time under water averages 70-170 days per spell (geo_struct 370). For units with a positive base, the probability of a 3-month loss of 50 USD or more is at most 2% (tsmom 1.6%); a 25 USD loss has 10-23% probability for the 1h trend units and about 0% for the 4h units.
- **The limits change which trades are taken.** With 3 open and 2 on one side, the live stream accepts only 15-22% of the 1h trend units' candidates (tsmom 3,610 of 23,649), first come first served; the accepted trades' mean is similar or better (donchian +0.18 R vs +0.09 R).
- **The EV gate is decisive for brackets.** With no calibrated edge, the 0.10 gate refuses every hrb2 and setup C candidate, 48% of trend_4h candidates and 70% of shock_desk candidates. Without the gate, trend_4h makes +0.25 USD per trade and hrb2 +0.14.
- **Correlation is the real tail.** If every concurrent same-side position stops together in down markets (rho -> 1), every trend unit's mean turns negative on the raw stream (tsmom -0.23 R, zoo_ensemble -0.34 R): the trend book is one bet (R-PD-1). Under the live limits (at most 2 same side) the daily CVaR5% worsens by 23-35% for the trend units, but by 53-56% for compression_breakout and geo_flag_bo (H-STRESS5 inconclusive).
- **History understates drawdowns.** In the bad resampled years (5th percentile of 1000 one-year paths per unit) the drawdown is 3-7 times that of a typical historical symbol-year (zoo_ensemble -15 R vs -3 R, tsmom -47 R vs -10 R). Resampled paths also keep the trend edge only when month-long blocks of real history are kept (SBB); on regime-switching paths every trend unit's median year is negative (H-STRESS8 contradicted).
- **Crises are bounded by the limits.** No unit lost more than 20 R on the live stream in any crisis window (worst: oi_price_divergence -14 R in the November 2026 drop, zscore_reversion -14 R in December 2021). On the unconstrained 23-symbol stream the worst windows were far larger (shock_desk -80 R in the November 2026 drop, rsi_reversion -64 R in the February 2026 drop).

## Scorecard

Mean net R per trade, base trade layer, universe L pooled (23 crypto perps; flow units BTCUSDT only; setup_c_proxy BTC/ETH). `*` = data-generated unit (designed after seeing its evidence: historical rows are not evidence).

| unit | verdict | n | base mean R [95% CI] | llm_in_sample (n) | clean_holdout (n) | rules fired | open at data end: n, sum R | closed trades only: verdict (holdout mean) |
|---|---|---|---|---|---|---|---|---|
| tsmom | **FRAGILE** | 23649 | +0.035 [-0.026, +0.103] | +0.035 (21734) | +0.031 (1915) | G4 +5 bp per side mean <= 0; G6 base 95% CI includes 0 | 13, +0.5 | FRAGILE (+0.031) |
| ema_stack | **FRAGILE** | 12488 | +0.064 [-0.018, +0.154] | +0.063 (11495) | +0.067 (993) | G6 base 95% CI includes 0 | 16, -2.2 | FRAGILE (+0.071) |
| donchian_breakout | **FRAGILE** | 6279 | +0.085 [-0.056, +0.249] | +0.085 (5774) | +0.090 (505) | G6 base 95% CI includes 0 | 0, +0.0 | FRAGILE (+0.090) |
| compression_breakout | **FRAGILE** | 947 | +0.046 [-0.706, +1.407] | +0.012 (893) | +0.606 (54) | G4 +5 bp per side mean <= 0; G6 base 95% CI includes 0; G7 total without the 5 best trades <= 0 | 17, +928.9 | FAILS (-0.960) |
| geo_struct | **FRAGILE** | 1377 | +0.116 [-0.087, +0.347] | +0.110 (1279) | +0.196 (98) | G6 base 95% CI includes 0; G7 total without the 5 best trades <= 0 | 8, +26.8 | FRAGILE (-0.085) |
| geo_pivot_bo | **SURVIVES** | 1081 | +0.438 [+0.092, +0.835] | +0.441 (994) | +0.401 (87) | - | 16, +45.9 | FRAGILE (-0.155) |
| geo_flag_bo | **FRAGILE** | 888 | +0.280 [-0.003, +0.620] | +0.255 (816) | +0.564 (72) | G6 base 95% CI includes 0 | 13, +41.8 | FRAGILE (-0.020) |
| zoo_ensemble | **SURVIVES** | 2190 | +0.354 [+0.035, +0.712] | +0.331 (2039) | +0.664 (151) | - | 20, +87.4 | SURVIVES (+0.099) |
| trend_4h | **FRAGILE** | 3710 | +0.099 [-0.024, +0.219] | +0.068 (3421) | +0.468 (289) | G6 base 95% CI includes 0 | 23, +20.4 | FRAGILE (+0.432) |
| zscore_reversion | **FAILS** | 9127 | -0.094 [-0.129, -0.061] | -0.101 (8391) | -0.016 (736) | F1 base mean <= 0; F2 fees x1.5 mean <= 0 | 0, +0.0 | FAILS (-0.016) |
| rsi_reversion | **FAILS** | 8452 | -0.052 [-0.078, -0.026] | -0.055 (7734) | -0.017 (718) | F1 base mean <= 0; F2 fees x1.5 mean <= 0 | 0, +0.0 | FAILS (-0.017) |
| setup_c_proxy | **FAILS** | 676 | -0.110 [-0.172, -0.049] | -0.127 (597) | +0.013 (79) | F1 base mean <= 0; F2 fees x1.5 mean <= 0 | 0, +0.0 | FAILS (+0.013) |
| funding_contrarian | **FAILS** | 5171 | -0.021 [-0.046, +0.003] | -0.022 (4764) | -0.013 (407) | F1 base mean <= 0; F2 fees x1.5 mean <= 0 | 0, +0.0 | FAILS (-0.013) |
| oi_price_divergence | **FAILS** | 8789 | -0.019 [-0.048, +0.012] | -0.024 (8130) | +0.050 (659) | F1 base mean <= 0; F2 fees x1.5 mean <= 0 | 3, +2.3 | FAILS (+0.047) |
| hrb2* | **FRAGILE** | 588 | +0.046 [-0.109, +0.202] | +0.005 (558) | +0.819 (30) | G6 base 95% CI includes 0 | 0, +0.0 | FRAGILE (+0.819) |
| shock_regime* | **FRAGILE** | 6722 | +0.058 [+0.010, +0.108] | +0.067 (6103) | -0.026 (619) | G5 clean_holdout mean <= 0 | 1, +0.3 | FRAGILE (-0.027) |
| shock_desk* | **FAILS** | 6561 | -0.003 [-0.064, +0.062] | +0.011 (5951) | -0.136 (610) | F1 base mean <= 0; F2 fees x1.5 mean <= 0 | 1, +0.4 | FAILS (-0.137) |
| hs2_fade_blackout* | **FAILS** | 1497 | -0.082 [-0.183, +0.025] | -0.087 (1323) | -0.046 (174) | F1 base mean <= 0; F2 fees x1.5 mean <= 0 | 0, +0.0 | FAILS (-0.046) |
| weekend_shock_follow | **FAILS** | 135 | -0.051 [-0.291, +0.218] | -0.028 (126) | -0.379 (9) | F1 base mean <= 0; F2 fees x1.5 mean <= 0 | 0, +0.0 | FAILS (-0.379) |
| weekend_drift_fade | **FAILS** | 451 | -0.007 [-0.199, +0.192] | -0.016 (406) | +0.066 (45) | F1 base mean <= 0; F2 fees x1.5 mean <= 0 | 0, +0.0 | FAILS (+0.066) |
| cme_gap_fill | **FAILS** | 1137 | -0.115 [-0.267, +0.047] | -0.128 (1020) | +0.001 (117) | F1 base mean <= 0; F2 fees x1.5 mean <= 0 | 0, +0.0 | FAILS (+0.001) |
| flow_momentum | **FRAGILE** | 186 | +0.085 [-0.160, +0.400] | +0.067 (41) | +0.090 (145) | G6 base 95% CI includes 0; G7 total without the 5 best trades <= 0 | 0, +0.0 | FRAGILE (+0.090) |
| cvd_divergence | **FRAGILE** | 1 | +0.590 [n/a, n/a] | - | +0.590 (1) | G1 +-20% parameter cliff; G6 base 95% CI includes 0; G7 total without the 5 best trades <= 0 | 0, +0.0 | FRAGILE (+0.590) |

### Costs, fills and latency (mean net R per trade)

| unit | base | fees x1.5 | fees x2 | +5 bp | +10 bp | +25 bp | gap fill | stop next open | no stop | lat 1m | lat 5m | lat 15m | gapped stops, worst gap (R) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| tsmom | +0.035 | +0.021 | +0.006 | -0.003 | -0.041 | -0.154 | +0.035 | +0.034 | +0.031 | +0.034 | +0.034 | +0.033 | 0/4940, worst +0.00 |
| ema_stack | +0.064 | +0.050 | +0.036 | +0.026 | -0.011 | -0.123 | +0.064 | +0.063 | +0.089 | +0.064 | +0.065 | +0.066 | 0/4724, worst -0.00 |
| donchian_breakout | +0.085 | +0.073 | +0.060 | +0.052 | +0.018 | -0.083 | +0.085 | +0.101 | +0.156 | +0.087 | +0.091 | +0.084 | 0/3721, worst +0.00 |
| compression_breakout | +0.046 | +0.028 | +0.010 | -0.003 | -0.052 | -0.199 | +0.046 | +0.108 | -0.129 | +0.047 | +0.048 | +0.052 | 0/738, worst -0.00 |
| geo_struct | +0.116 | +0.110 | +0.103 | +0.099 | +0.083 | +0.033 | +0.116 | +0.138 | +0.112 | +0.117 | +0.105 | +0.106 | 0/865, worst +0.00 |
| geo_pivot_bo | +0.438 | +0.432 | +0.425 | +0.421 | +0.405 | +0.355 | +0.438 | +0.468 | +0.606 | +0.437 | +0.437 | +0.434 | 0/717, worst -0.00 |
| geo_flag_bo | +0.280 | +0.273 | +0.267 | +0.263 | +0.246 | +0.197 | +0.280 | +0.309 | +0.334 | +0.279 | +0.283 | +0.277 | 0/580, worst -0.00 |
| zoo_ensemble | +0.354 | +0.347 | +0.340 | +0.335 | +0.316 | +0.261 | +0.354 | +0.379 | +0.458 | +0.352 | +0.350 | +0.343 | 0/1196, worst -0.00 |
| trend_4h | +0.099 | +0.093 | +0.086 | +0.082 | +0.064 | +0.010 | +0.099 | +0.121 | - | +0.099 | +0.096 | +0.089 | 0/2649, worst -0.00 |
| zscore_reversion | -0.094 | -0.106 | -0.117 | -0.125 | -0.156 | -0.249 | -0.094 | -0.088 | -0.145 | -0.095 | -0.097 | -0.098 | 0/3992, worst -0.00 |
| rsi_reversion | -0.052 | -0.064 | -0.076 | -0.084 | -0.116 | -0.213 | -0.052 | -0.045 | -0.045 | -0.052 | -0.055 | -0.060 | 0/1906, worst -0.00 |
| setup_c_proxy | -0.110 | -0.145 | -0.180 | -0.203 | -0.296 | -0.575 | -0.110 | -0.098 | -0.119 | -0.113 | -0.119 | -0.084 | 0/255, worst +0.00 |
| funding_contrarian | -0.021 | -0.033 | -0.045 | -0.054 | -0.086 | -0.184 | -0.021 | -0.019 | -0.009 | -0.021 | -0.022 | -0.023 | 0/906, worst +0.00 |
| oi_price_divergence | -0.019 | -0.032 | -0.044 | -0.053 | -0.087 | -0.190 | -0.019 | -0.009 | -0.014 | -0.019 | -0.017 | -0.018 | 0/1855, worst +0.00 |
| hrb2 | +0.046 | +0.039 | +0.032 | +0.028 | +0.010 | -0.045 | +0.046 | +0.035 | +0.078 | +0.046 | +0.044 | +0.042 | 0/235, worst -0.00 |
| shock_regime | +0.058 | +0.045 | +0.033 | +0.024 | -0.009 | -0.109 | +0.058 | +0.064 | +0.068 | +0.057 | +0.055 | +0.045 | 0/884, worst -0.00 |
| shock_desk | -0.003 | -0.035 | -0.066 | -0.061 | -0.118 | -0.292 | -0.003 | +0.005 | - | -0.011 | -0.039 | -0.083 | 0/4361, worst -0.00 |
| hs2_fade_blackout | -0.082 | -0.109 | -0.135 | -0.130 | -0.179 | -0.325 | -0.082 | -0.047 | - | -0.086 | -0.100 | -0.127 | 0/883, worst +0.00 |
| weekend_shock_follow | -0.051 | -0.059 | -0.066 | -0.071 | -0.091 | -0.150 | -0.051 | -0.039 | -0.068 | -0.053 | -0.065 | -0.082 | 0/42, worst +0.00 |
| weekend_drift_fade | -0.007 | -0.023 | -0.039 | -0.050 | -0.092 | -0.218 | -0.007 | -0.031 | -0.028 | -0.007 | -0.014 | +0.009 | 0/117, worst +0.00 |
| cme_gap_fill | -0.115 | -0.128 | -0.141 | -0.150 | -0.184 | -0.287 | -0.115 | -0.129 | -0.085 | -0.117 | -0.119 | -0.119 | 0/559, worst +0.00 |
| flow_momentum | +0.085 | +0.057 | +0.030 | +0.011 | -0.063 | -0.284 | +0.085 | +0.094 | +0.091 | +0.085 | +0.085 | +0.096 | 0/37, worst +0.00 |
| cvd_divergence | +0.590 | +0.568 | +0.547 | +0.532 | +0.475 | +0.301 | +0.590 | +0.590 | +0.590 | +0.585 | +0.566 | +0.518 | 0/0, worst +0.00 |

### Parameters, synthetic paths, correlation, liquidity, crises, live engine

| unit | D8 base | worst +-20% | worst +-50% | cliff keys (+-20%) | joint draws > 0 | SBB median path | RS median path | co-stop (S7) | drought (S8) | worst crisis, engine (R) | engine n, mean R |
|---|---|---|---|---|---|---|---|---|---|---|---|
| tsmom | +0.066 | +0.057 | +0.038 | - | 100% | +0.026 | -0.079 | -0.234 | -0.093 | -5.6 | 3610, +0.038 |
| ema_stack | +0.090 | +0.077 | +0.041 | - | 100% | +0.045 | -0.079 | -0.349 | -0.089 | -4.3 | 2599, +0.069 |
| donchian_breakout | +0.141 | +0.141 | +0.079 | - | 100% | +0.048 | -0.117 | -0.341 | -0.084 | -7.6 | 1382, +0.184 |
| compression_breakout | -0.951 | -0.989 | -1.000 | - | 0% | -0.500 | -0.678 | -0.954 | -0.188 | -1.0 | 130, +3.427 |
| geo_struct | +0.192 | +0.115 | -0.093 | - | 100% | -0.075 | -0.121 | -0.460 | +0.032 | -3.0 | 335, +0.153 |
| geo_pivot_bo | +0.689 | +0.536 | +0.317 | - | 100% | +0.009 | -0.235 | -0.683 | +0.355 | -0.8 | 239, +0.557 |
| geo_flag_bo | +0.535 | +0.265 | +0.274 | - | 100% | -0.167 | -0.257 | -0.729 | +0.194 | -1.0 | 211, +0.358 |
| zoo_ensemble | +0.538 | +0.500 | +0.321 | - | 100% | +0.145 | -0.107 | -0.337 | +0.264 | -2.1 | 371, +0.470 |
| trend_4h | +0.153 | +0.085 | +0.035 | - | 100% | +0.028 | -0.016 | -0.307 | +0.006 | -4.0 | 326, +0.235 |
| zscore_reversion | -0.126 | -0.144 | -0.181 | - | 0% | -0.105 | -0.018 | -0.201 | -0.232 | -13.6 | 3768, -0.090 |
| rsi_reversion | -0.045 | -0.059 | -0.077 | - | 0% | -0.045 | -0.021 | -0.109 | -0.166 | -13.0 | 4283, -0.059 |
| setup_c_proxy | -0.110 | -0.142 | -0.140 | - | 0% | n/a | n/a | -0.121 | -0.110 | n/a | 0 |
| funding_contrarian | -0.016 | -0.036 | -0.072 | - | 7% | n/a | n/a | -0.062 | -0.125 | -9.0 | 3239, -0.036 |
| oi_price_divergence | -0.022 | -0.030 | -0.034 | - | 33% | n/a | n/a | -0.048 | -0.136 | -14.2 | 5328, -0.033 |
| hrb2 | -0.031 | -0.058 | -0.062 | - | 43% | n/a | n/a | -0.075 | -0.020 | n/a | 0 |
| shock_regime | +0.051 | +0.021 | +0.005 | - | 97% | +0.001 | -0.057 | +0.004 | -0.040 | -10.3 | 3052, -0.001 |
| shock_desk | -0.032 | -0.070 | -0.126 | - | 47% | -0.090 | -0.128 | -0.162 | -0.286 | -10.0 | 1062, -0.042 |
| hs2_fade_blackout | -0.150 | -0.179 | -0.178 | - | 0% | -0.205 | -0.134 | -0.205 | -0.301 | -3.0 | 177, -0.076 |
| weekend_shock_follow | -0.030 | -0.303 | -0.269 | - | 3% | n/a | n/a | -0.116 | -0.149 | -2.0 | 81, -0.051 |
| weekend_drift_fade | -0.059 | -0.135 | -0.145 | - | 24% | n/a | n/a | -0.170 | -0.182 | -2.0 | 233, -0.148 |
| cme_gap_fill | -0.202 | -0.228 | -0.243 | - | 0% | n/a | n/a | -0.247 | -0.294 | -2.0 | 454, -0.102 |
| flow_momentum | +0.085 | +0.023 | -0.044 | - | 67% | n/a | n/a | +0.085 | +0.085 | -0.8 | 186, +0.088 |
| cvd_divergence | +0.590 | -0.405 | -0.037 | window | 45% | n/a | n/a | +0.590 | +0.590 | n/a | 1, +0.590 |

## What the live risk limits imply (S9, live_small on each unit's pooled stream)

Registered S9: 1.30 USD per R, 2 USD worst case, 20 entries per UTC day, 10 USD daily budget, 3 open, 2 same side, 400 USD gross, tier caps, EV gate >= 0.10 where a probability is defined (bracket setups: first-passage null + prior edge). USD figures are strategy P&L under these limits, not account values. P(loss >= x) = probability that the cumulative P&L falls to -x USD at some point within 91 days (stationary bootstrap of days).

| unit | trades | USD/trade | USD total | P(day stop hit) | P(budget refusal day) | worst day USD | max losing run | P(>=5 losers in 3m) emp / iid | mean time under water | P(3m loss >= 25 / 50 / 100 USD) | 3m 5th pct USD | main refusals |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| tsmom | 3610 | +0.050 | +179 | 0.00% | 0.17% | -7.21 | 31 | 100% / 100% | 73.2 d (max 644) | 23% / 2% / 0% | -36.1 | max_open 11621, same_side 7926, tier 335, stop_distance 148 |
| ema_stack | 2599 | +0.089 | +232 | 0.00% | 0.00% | -5.51 | 19 | 100% / 100% | 70.3 d (max 736) | 13% / 0% / 0% | -28.4 | same_side 6189, max_open 3407, tier 205, stop_distance 88 |
| donchian_breakout | 1382 | +0.239 | +330 | 0.00% | 0.00% | -6.18 | 17 | 96% / 99% | 93.0 d (max 681) | 10% / 0% / 0% | -26.9 | same_side 3476, max_open 1282, tier 106, stop_distance 33 |
| compression_breakout | 130 | +4.455 | +579 | 0.00% | 0.00% | -2.60 | 127 | 44% / 89% | 2179.0 d (max 2179) | 0% / 0% / 0% | -13.0 | max_open 756, same_side 59, tier 2 |
| geo_struct | 335 | +0.199 | +67 | 0.00% | 0.00% | -3.90 | 17 | 61% / 64% | 370.8 d (max 1039) | 0% / 0% / 0% | -14.4 | max_open 459, same_side 305, stop_distance 261, tier 17 |
| geo_pivot_bo | 239 | +0.725 | +173 | 0.00% | 0.00% | -2.60 | 11 | 41% / 38% | 169.0 d (max 423) | 0% / 0% / 0% | -9.7 | same_side 356, max_open 291, stop_distance 177, tier 18 |
| geo_flag_bo | 211 | +0.466 | +98 | 0.00% | 0.00% | -2.60 | 9 | 35% / 37% | 143.2 d (max 727) | 0% / 0% / 0% | -9.5 | same_side 322, max_open 185, stop_distance 161, tier 9 |
| zoo_ensemble | 371 | +0.611 | +227 | 0.00% | 0.00% | -2.60 | 12 | 54% / 65% | 89.9 d (max 353) | 0% / 0% / 0% | -13.2 | same_side 842, max_open 719, stop_distance 228, tier 30 |
| trend_4h | 326 | +0.306 | +100 | 0.00% | 0.00% | -3.90 | 18 | 76% / 56% | 91.5 d (max 714) | 0% / 0% / 0% | -11.1 | ev_gate 1771, same_side 1129, stop_distance 434, tier 50 |
| zscore_reversion | 3768 | -0.117 | -441 | 0.00% | 0.12% | -9.09 | 10 | 90% / 82% | 1194.5 d (max 2355) | 35% / 2% / 0% | -40.8 | same_side 4732, max_open 331, tier 163, stop_distance 127 |
| rsi_reversion | 4283 | -0.076 | -326 | 0.00% | 0.00% | -7.79 | 10 | 65% / 52% | 1196.5 d (max 2359) | 14% / 0% / 0% | -30.4 | same_side 3859, tier 151, stop_distance 150, max_open 9 |
| setup_c_proxy | 0 | - | - | - | - | - | - | - | - | - | - | ev_gate 676 |
| funding_contrarian | 3239 | -0.047 | -151 | 0.00% | 0.00% | -7.16 | 13 | 82% / 87% | 764.3 d (max 2114) | 5% / 0% / 0% | -23.3 | same_side 1737, stop_distance 134, tier 53, max_open 8 |
| oi_price_divergence | 5328 | -0.043 | -227 | 0.00% | 0.00% | -7.48 | 15 | 96% / 100% | 378.3 d (max 2029) | 31% / 2% / 0% | -40.2 | same_side 3218, tier 162, stop_distance 81 |
| hrb2 | 0 | - | - | - | - | - | - | - | - | - | - | ev_gate 449, stop_distance 135, tier 4 |
| shock_regime | 3052 | -0.001 | -3 | 0.00% | 0.00% | -6.50 | 17 | 97% / 100% | 87.5 d (max 1244) | 6% / 0% / 0% | -22.7 | same_side 1749, max_open 1737, tier 101, stop_distance 83 |
| shock_desk | 1062 | -0.055 | -59 | 0.00% | 0.00% | -6.05 | 24 | 95% / 96% | 538.5 d (max 1915) | 8% / 0% / 0% | -25.3 | ev_gate 4598, same_side 723, tier 98, max_open 66 |
| hs2_fade_blackout | 177 | -0.099 | -18 | 0.00% | 0.00% | -2.60 | 19 | 26% / 27% | 398.0 d (max 1052) | 0% / 0% / 0% | -8.6 | ev_gate 1177, same_side 110, stop_distance 19, tier 14 |
| weekend_shock_follow | 81 | -0.066 | -5 | 0.00% | 0.00% | -2.60 | 6 | 1% / 0% | 340.8 d (max 701) | 0% / 0% / 0% | -4.0 | same_side 40, tier 7, stop_distance 7 |
| weekend_drift_fade | 233 | -0.192 | -45 | 0.00% | 0.00% | -3.00 | 13 | 45% / 41% | 1071.5 d (max 2038) | 0% / 0% / 0% | -8.7 | same_side 208, max_open 6, tier 4 |
| cme_gap_fill | 454 | -0.132 | -60 | 0.00% | 0.00% | -3.81 | 12 | 49% / 40% | 1176.5 d (max 2045) | 0% / 0% / 0% | -12.7 | same_side 611, max_open 33, tier 20, stop_distance 19 |
| flow_momentum | 186 | +0.114 | +21 | 0.00% | 0.00% | -2.60 | 8 | 100% / 99% | 16.2 d (max 70) | 6% / 0% / 0% | -22.8 | - |
| cvd_divergence | 1 | +0.767 | +1 | 0.00% | 0.00% | +0.77 | 0 | n/a / 0% | 0.0 d (max 0) | n/a / n/a / n/a | n/a | - |

Bracket setups without the EV gate (the gate is the part a calibrated probability or an LLM view changes):

| unit | trades | USD/trade | USD total | P(day stop hit) | P(3m loss >= 25 / 50 / 100) | worst crisis (R) |
|---|---|---|---|---|---|---|
| trend_4h | 566 | +0.245 | +138 | 0.00% | 0% / 0% / 0% | -2.4 |
| setup_c_proxy | 676 | -0.140 | -94 | 0.00% | 5% / 0% / 0% | -8.8 |
| hrb2 | 243 | +0.140 | +34 | 0.00% | 0% / 0% / 0% | -3.7 |
| shock_desk | 2990 | -0.069 | -206 | 0.00% | 28% / 2% / 0% | -19.5 |
| hs2_fade_blackout | 930 | -0.165 | -153 | 0.00% | 2% / 0% / 0% | -4.6 |

## Regime cells (S1): mean net R per trade by v0 cell at entry, pooled (n)

`+`/`-` after the number: sign in llm_in_sample and clean_holdout agree (both >= 10 trades); `?` they disagree.

| unit | R1 | R2 | R3 | R4 | R5 | R6 | R7 | R8 | R9 |
|---|---|---|---|---|---|---|---|---|---|
| tsmom | +1.12 (46) | +0.52 (15) | +0.36 (7) | +0.11+ (10481) | +0.02+ (6935) | -0.10+ (5076) | -0.35+ (62) | -0.12 (19) | -0.57 (15) |
| ema_stack | +0.12+ (1120) | +0.09? (1042) | +0.29+ (1181) | +0.14? (2249) | -0.07? (1630) | -0.07? (1218) | +0.08? (1351) | +0.00? (1113) | -0.07+ (1033) |
| donchian_breakout | +0.31? (697) | +0.04? (402) | +0.24+ (355) | +0.27+ (1357) | -0.05? (913) | -0.15+ (589) | +0.11? (898) | -0.10+ (488) | -0.16? (294) |
| compression_breakout | -0.78+ (123) | -0.97 (24) | -1.00 (3) | -0.45? (400) | -0.93 (81) | -0.86 (7) | +0.03? (207) | -0.92 (20) | - |
| geo_struct | +0.16 (117) | +0.98 (112) | +0.51? (148) | -0.07+ (147) | -0.05? (227) | -0.18? (269) | +0.11 (93) | -0.04 (100) | +0.01 (80) |
| geo_pivot_bo | +2.69 (93) | -0.07 (140) | +1.11+ (212) | +0.74 (32) | -0.01 (44) | -0.35 (47) | +0.19 (118) | +0.09+ (140) | -0.21? (181) |
| geo_flag_bo | +2.13 (67) | -0.16 (110) | +0.64+ (162) | +0.84 (19) | -0.28 (33) | +0.16 (56) | +0.24 (105) | +0.13+ (145) | -0.26 (138) |
| zoo_ensemble | +0.30+ (308) | +0.71? (296) | +1.24+ (288) | +0.50 (107) | +0.26 (90) | -0.08 (51) | +0.11+ (347) | +0.12+ (313) | -0.09? (288) |
| trend_4h | +0.11 (202) | +0.08+ (274) | +0.23+ (689) | +0.17? (243) | +0.08+ (283) | +0.07+ (566) | +0.16+ (358) | +0.18+ (334) | -0.07? (761) |
| zscore_reversion | -0.11+ (968) | -0.04+ (751) | -0.08+ (687) | -0.18+ (1424) | -0.12+ (1114) | -0.02? (1155) | -0.17? (1188) | -0.00? (894) | +0.05? (539) |
| rsi_reversion | -0.08+ (1119) | -0.05? (878) | -0.05+ (952) | -0.09+ (1033) | -0.08+ (552) | -0.06? (374) | -0.08? (1268) | +0.02? (998) | -0.02? (883) |
| setup_c_proxy | -0.22 (9) | +0.66 (3) | -0.61 (3) | -0.13? (115) | -0.05? (82) | +0.06 (76) | -0.24+ (102) | -0.11+ (123) | -0.13 (163) |
| funding_contrarian | -0.04 (133) | -0.09 (248) | +0.00? (526) | -0.03? (603) | -0.02+ (636) | -0.04+ (705) | -0.08+ (572) | -0.03+ (641) | +0.05+ (918) |
| oi_price_divergence | -0.01? (863) | -0.02? (1084) | +0.02+ (1558) | +0.01+ (1103) | -0.01? (917) | +0.05? (857) | -0.14? (770) | -0.07+ (646) | -0.08? (638) |
| hrb2 | +0.22 (3) | -0.45 (2) | -0.07 (7) | -0.02 (47) | +0.14+ (141) | +0.06+ (324) | -0.39 (12) | -0.07 (23) | -0.17 (29) |
| shock_regime | +0.09+ (840) | -0.09? (443) | +0.04+ (339) | +0.16? (1932) | -0.04? (776) | +0.08+ (276) | +0.08? (1266) | -0.08+ (536) | -0.08+ (314) |
| shock_desk | +0.07? (825) | -0.22? (445) | -0.02? (338) | +0.11? (1865) | -0.19+ (763) | +0.08+ (275) | +0.08? (1198) | -0.26+ (537) | -0.04? (315) |
| hs2_fade_blackout | - | -0.14? (233) | -0.06+ (204) | - | -0.06? (379) | +0.07+ (194) | - | -0.24+ (304) | +0.01? (183) |
| weekend_shock_follow | -1.00 (1) | +0.84 (5) | -0.01 (62) | - | - | -0.46 (5) | - | +0.35 (3) | -0.06 (49) |
| weekend_drift_fade | +0.18 (75) | -0.07 (27) | +0.51 (34) | -0.24 (44) | -0.46 (12) | +0.74 (10) | -0.07 (112) | -0.08 (56) | -0.16? (64) |
| cme_gap_fill | +0.00 (136) | -0.21? (102) | +0.15+ (199) | -0.12 (85) | -0.07 (37) | +0.05 (68) | -0.06+ (149) | -0.30 (91) | -0.39? (206) |
| flow_momentum | -0.14 (20) | +1.04 (6) | -0.27 (10) | +0.28 (52) | -0.18 (37) | +0.03 (20) | +0.66 (17) | -0.33 (16) | +0.12 (8) |
| cvd_divergence | - | - | - | - | - | +0.59 (1) | - | - | - |

### Calendar slices

| unit | weekday | TradFi weekend | NYSE holiday | off funding | +-1 h of funding |
|---|---|---|---|---|---|
| tsmom | +0.081 (17910) | -0.111 (5739) | +0.072 (684) | +0.021 (13773) | +0.054 (9876) |
| ema_stack | +0.103 (9408) | -0.057 (3080) | -0.208 (308) | +0.080 (7801) | +0.036 (4687) |
| donchian_breakout | +0.129 (5179) | -0.120 (1100) | +0.142 (139) | +0.119 (3599) | +0.041 (2680) |
| compression_breakout | +0.481 (429) | -0.313 (518) | -0.969 (31) | +0.439 (579) | -0.572 (368) |
| geo_struct | +0.070 (1126) | +0.321 (251) | -0.302 (26) | +0.042 (679) | +0.188 (698) |
| geo_pivot_bo | +0.477 (851) | +0.293 (230) | -0.990 (19) | +0.415 (552) | +0.462 (529) |
| geo_flag_bo | +0.228 (686) | +0.456 (202) | -0.492 (18) | +0.084 (451) | +0.481 (437) |
| zoo_ensemble | +0.494 (1664) | -0.092 (526) | +0.242 (38) | +0.504 (1065) | +0.211 (1125) |
| trend_4h | +0.099 (3710) | - | +0.105 (71) | +0.087 (1463) | +0.108 (2247) |
| zscore_reversion | -0.103 (7640) | -0.048 (1487) | -0.016 (170) | -0.088 (5278) | -0.102 (3849) |
| rsi_reversion | -0.054 (6520) | -0.044 (1932) | -0.111 (192) | -0.033 (4903) | -0.077 (3549) |
| setup_c_proxy | -0.067 (562) | -0.324 (114) | +0.407 (12) | -0.073 (474) | -0.197 (202) |
| funding_contrarian | -0.015 (3682) | -0.037 (1489) | +0.027 (150) | -0.013 (879) | -0.023 (4292) |
| oi_price_divergence | -0.020 (6539) | -0.014 (2250) | -0.039 (221) | -0.032 (4702) | -0.004 (4087) |
| hrb2 | +0.078 (438) | -0.047 (150) | +0.455 (12) | +0.100 (289) | -0.006 (299) |
| shock_regime | +0.079 (5413) | -0.031 (1309) | -0.055 (141) | +0.053 (3660) | +0.064 (3062) |
| shock_desk | +0.013 (5293) | -0.070 (1268) | -0.241 (141) | +0.030 (3561) | -0.041 (3000) |
| hs2_fade_blackout | -0.089 (1230) | -0.049 (267) | +0.057 (16) | -0.061 (847) | -0.110 (650) |
| weekend_shock_follow | -0.051 (135) | - | - | - | -0.051 (135) |
| weekend_drift_fade | -0.007 (451) | - | - | - | -0.007 (451) |
| cme_gap_fill | -0.115 (1137) | - | - | - | -0.115 (1137) |
| flow_momentum | +0.168 (132) | -0.118 (54) | +0.580 (3) | +0.222 (114) | -0.133 (72) |
| cvd_divergence | +0.590 (1) | - | - | +0.590 (1) | - |

## Crisis windows (S2)

Sum of net R of trades entered in the window: live-engine stream (raw universe stream in brackets). The 2020-03 window uses Deribit BTC-PERPETUAL 1h (Bybit history starts 2020-03-25), BTC only.

Windows: 2020-03 covid crash = 2020-03-09..2020-03-20; 2021-05-19 crash = 2021-05-12..2021-05-26; 2021-12 deleveraging = 2021-12-01..2021-12-31; 2022-05 LUNA = 2022-05-05..2022-05-20; 2022-06 3AC/Celsius = 2022-06-10..2022-06-25; 2022-11 FTX = 2022-11-06..2022-11-20; 2023-03 SVB/USDC = 2023-03-08..2023-03-20; 2024-08-05 yen carry = 2024-08-01..2024-08-10; 2026-11-21 flush = 2026-11-19..2026-11-25; BTC 14d drop to 2026-02-05 (-35.3%) = 2026-01-22..2026-02-08; BTC 14d drop to 2026-11-18 (-24.4%) = 2026-11-04..2026-11-21; BTC 14d drop to 2026-06-06 (-23.2%) = 2026-05-23..2026-06-09

| unit | 2020-03 covid crash | 2021-05-19 crash | 2021-12 deleveraging | 2022-05 LUNA | 2022-06 3AC/Celsius | 2022-11 FTX | 2023-03 SVB/USDC | 2024-08-05 yen carry | 2026-11-21 flush | BTC 14d drop to 2026-02-05 | BTC 14d drop to 2026-11-18 | BTC 14d drop to 2026-06-06 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| tsmom | +1.4 (+1) | +11.6 (+13) | +2.2 (+50) | +12.6 (+82) | -5.4 (+15) | +8.6 (+48) | +4.6 (+8) | -2.3 (+17) | -5.6 (-27) | +34.1 (+130) | +0.5 (+82) | +6.7 (+85) |
| ema_stack | -1.6 (-2) | +11.1 (+24) | +0.2 (+22) | +8.6 (+64) | +2.6 (+36) | -3.5 (+13) | -3.3 (-23) | -4.3 (+16) | +2.4 (+7) | +9.5 (+125) | +14.2 (+109) | +25.5 (+71) |
| donchian_breakout | +0.3 (+0) | +5.2 (+3) | -7.6 (-10) | +3.7 (+71) | +6.3 (+38) | +5.0 (+19) | +0.8 (-18) | -1.8 (+7) | -3.0 (-10) | +15.1 (+64) | +2.6 (+1) | +1.1 (+27) |
| compression_breakout | 0 (0) | 0 (0) | -0.9 (-13) | 0 (0) | -1.0 (-3) | 0 (-4) | 0 (0) | 0 (-1) | 0 (0) | 0 (+27) | +12.1 (+23) | 0 (-2) |
| geo_struct | 0 (0) | 0 (-2) | +3.1 (+15) | -0.1 (+26) | 0 (-4) | +0.6 (-10) | -1.0 (-8) | -0.1 (-5) | 0 (-0) | 0 (+29) | +2.0 (+7) | -3.0 (+5) |
| geo_pivot_bo | +0.9 (+1) | +2.0 (+3) | -0.8 (-3) | 0 (+35) | 0 (-1) | +4.3 (+12) | -0.5 (-8) | +0.2 (-4) | 0 (-1) | -0.1 (+17) | 0 (+20) | -0.3 (+4) |
| geo_flag_bo | 0 (0) | 0 (-2) | -1.0 (-5) | 0 (+1) | 0 (-0) | +4.6 (+6) | -0.2 (-6) | -1.0 (+0) | 0 (-4) | 0 (+14) | +2.8 (+14) | +5.8 (+8) |
| zoo_ensemble | 0 (0) | +2.0 (+3) | -2.1 (+15) | 0 (0) | 0 (+3) | +4.4 (+6) | -1.1 (-8) | +2.0 (+5) | 0 (0) | +0.2 (+6) | 0 (+14) | +2.4 (+6) |
| trend_4h | 0 (0) | 0 (+1) | -1.0 (+0) | -3.0 (+81) | 0 (-15) | -1.0 (-13) | +2.4 (-24) | -2.0 (-2) | 0 (-1) | -1.0 (+43) | 0 (+78) | -4.0 (+33) |
| zscore_reversion | -1.0 (-0) | -4.7 (-8) | -13.6 (-29) | -8.8 (-30) | -6.8 (-14) | -1.4 (-14) | -6.1 (-23) | -5.0 (-15) | -0.1 (-4) | -9.6 (-36) | -11.1 (-6) | -0.6 (-10) |
| rsi_reversion | -2.0 (-2) | -0.6 (+1) | -11.7 (-19) | -0.3 (+6) | -4.9 (-11) | -0.5 (+14) | -0.5 (-2) | +0.2 (-12) | +2.1 (+12) | -13.0 (-64) | -0.7 (+8) | -0.7 (+4) |
| setup_c_proxy | 0 (0) | 0 (0) | 0 (0) | 0 (0) | 0 (0) | 0 (0) | 0 (0) | 0 (-3) | 0 (0) | 0 (-9) | 0 (-2) | 0 (-6) |
| funding_contrarian | 0 (0) | +0.4 (+6) | -0.8 (+0) | -7.3 (-12) | -4.1 (-11) | -0.6 (-10) | -2.0 (-2) | -8.1 (-7) | -0.2 (-2) | -9.0 (-17) | -8.0 (-13) | -2.5 (+0) |
| oi_price_divergence | 0 (0) | -3.5 (-4) | -4.2 (-5) | -13.8 (-25) | -5.6 (-9) | -12.6 (-16) | -0.1 (+7) | -9.3 (-7) | +2.8 (+5) | -10.2 (-28) | -14.2 (-16) | -11.9 (+4) |
| hrb2 | 0 (0) | 0 (+1) | 0 (+1) | 0 (+6) | 0 (-3) | 0 (+6) | 0 (-2) | 0 (-12) | 0 (+2) | 0 (+11) | 0 (+16) | 0 (+2) |
| shock_regime | 0 (0) | -3.5 (-8) | +5.4 (+13) | +9.7 (+19) | +4.5 (+21) | +8.1 (+10) | -4.4 (-12) | -6.9 (-17) | +0.8 (-7) | +3.2 (-8) | -10.3 (-52) | +3.8 (+3) |
| shock_desk | 0 (0) | -5.0 (-10) | -3.8 (-2) | +0.8 (+29) | +3.7 (+20) | +8.0 (+23) | -0.3 (-29) | -10.0 (-30) | -4.6 (-46) | -0.9 (+7) | -6.8 (-80) | -2.0 (+21) |
| hs2_fade_blackout | 0 (0) | -3.0 (-4) | -0.1 (+6) | -1.7 (-12) | +2.4 (+5) | +2.3 (+16) | +1.6 (-11) | -2.0 (-2) | -2.0 (-15) | -2.0 (-25) | -1.2 (-16) | 0 (+7) |
| weekend_shock_follow | 0 (0) | 0 (0) | -0.9 (-1) | 0 (0) | +4.3 (+9) | 0 (0) | 0 (0) | +2.2 (+2) | 0 (0) | -2.0 (-5) | +0.2 (+0) | -1.0 (-1) |
| weekend_drift_fade | 0 (0) | -2.0 (-2) | 0 (0) | -1.9 (-3) | -0.8 (-2) | -1.4 (-2) | 0 (0) | -2.0 (-4) | 0 (0) | +2.7 (+17) | -1.1 (+10) | +0.6 (+1) |
| cme_gap_fill | 0 (0) | -2.0 (-3) | +1.0 (-15) | -2.0 (-5) | -1.7 (-20) | -2.0 (-4) | -1.8 (-1) | -2.0 (-16) | 0 (0) | +2.3 (+19) | -1.2 (+4) | +0.3 (+0) |
| flow_momentum | 0 (0) | 0 (0) | 0 (0) | 0 (0) | 0 (0) | 0 (0) | 0 (0) | 0 (0) | 0 (0) | 0 (0) | -0.8 (-1) | +8.5 (+8) |
| cvd_divergence | 0 (0) | 0 (0) | 0 (0) | 0 (0) | 0 (0) | 0 (0) | 0 (0) | 0 (0) | 0 (0) | 0 (0) | 0 (0) | 0 (0) |

Worst window per unit on the raw universe stream. Base stops fill at the stop price, and 1h bars of 24/7 perps open where the previous bar closed, so open-gap fills never differ from it (0 gapped stops in every unit). The slippage columns therefore use the stop_next_open variant (a stop touched in bar k fills at the open of bar k+1), the realistic bound in a fast market:

| unit | window | sum R | max DD R | worst trade R | trades | stops | next-open slip mean R | worst slip R | worst trade R (next open) | sum R (next open) |
|---|---|---|---|---|---|---|---|---|---|---|
| tsmom | 2026-11-21 flush | -26.6 | -29.5 | -1.00 | 62 | 18 | -0.07 | -0.86 | -1.87 | -27.8 |
| ema_stack | 2023-03 SVB/USDC | -23.5 | -26.7 | -1.00 | 64 | 24 | +0.09 | -0.63 | -1.63 | -21.3 |
| donchian_breakout | 2023-03 SVB/USDC | -18.2 | -21.1 | -1.00 | 35 | 15 | +0.03 | -0.49 | -1.49 | -17.7 |
| compression_breakout | 2021-12 deleveraging | -13.3 | -13.3 | -1.00 | 14 | 12 | +0.18 | -0.54 | -1.54 | -11.2 |
| geo_struct | 2022-11 FTX | -9.6 | -9.6 | -1.00 | 17 | 12 | +0.05 | -0.29 | -1.29 | -8.9 |
| geo_pivot_bo | 2023-03 SVB/USDC | -8.5 | -9.0 | -1.00 | 10 | 9 | -0.04 | -0.23 | -1.23 | -8.8 |
| geo_flag_bo | 2023-03 SVB/USDC | -6.2 | -8.0 | -1.00 | 10 | 7 | +0.01 | -0.14 | -1.14 | -6.1 |
| zoo_ensemble | 2023-03 SVB/USDC | -8.3 | -9.3 | -1.00 | 11 | 9 | -0.06 | -1.09 | -2.09 | -8.8 |
| trend_4h | 2023-03 SVB/USDC | -24.0 | -34.0 | -1.00 | 40 | 36 | -0.02 | -0.29 | -1.29 | -24.5 |
| zscore_reversion | BTC 14d drop to 2026-02-05 (-35.3%) | -35.6 | -38.4 | -1.00 | 89 | 61 | +0.01 | -1.07 | -2.07 | -34.7 |
| rsi_reversion | BTC 14d drop to 2026-02-05 (-35.3%) | -63.9 | -64.4 | -1.00 | 105 | 69 | -0.01 | -1.07 | -2.07 | -64.9 |
| setup_c_proxy | BTC 14d drop to 2026-02-05 (-35.3%) | -8.9 | -8.9 | -1.00 | 20 | 12 | -0.01 | -1.57 | -2.57 | -9.0 |
| funding_contrarian | BTC 14d drop to 2026-02-05 (-35.3%) | -16.6 | -19.8 | -1.00 | 99 | 22 | +0.14 | -1.03 | -2.02 | -13.6 |
| oi_price_divergence | BTC 14d drop to 2026-02-05 (-35.3%) | -28.1 | -36.4 | -1.00 | 99 | 39 | +0.11 | -0.45 | -1.45 | -23.8 |
| hrb2 | 2024-08-05 yen carry | -11.6 | -13.4 | -1.00 | 15 | 13 | +0.05 | -0.21 | -1.21 | -10.9 |
| shock_regime | BTC 14d drop to 2026-11-18 (-24.4%) | -52.4 | -52.4 | -1.00 | 108 | 32 | +0.25 | -0.29 | -1.29 | -44.4 |
| shock_desk | BTC 14d drop to 2026-11-18 (-24.4%) | -79.8 | -79.8 | -1.00 | 109 | 98 | +0.18 | -2.91 | -3.91 | -62.2 |
| hs2_fade_blackout | BTC 14d drop to 2026-02-05 (-35.3%) | -25.0 | -25.9 | -1.00 | 28 | 26 | -0.28 | -0.95 | -1.95 | -32.2 |
| weekend_shock_follow | BTC 14d drop to 2026-02-05 (-35.3%) | -4.7 | -4.7 | -1.00 | 5 | 4 | +0.11 | -0.02 | -1.02 | -4.2 |
| weekend_drift_fade | 2024-08-05 yen carry | -4.0 | -4.0 | -1.00 | 4 | 4 | -0.36 | -0.58 | -1.58 | -5.4 |
| cme_gap_fill | 2022-06 3AC/Celsius | -19.7 | -20.0 | -1.00 | 21 | 20 | -0.01 | -0.29 | -1.29 | -19.9 |
| flow_momentum | BTC 14d drop to 2026-11-18 (-24.4%) | -0.8 | -1.0 | -1.00 | 5 | 1 | -0.39 | -0.39 | -1.39 | -1.2 |

## Parameter surfaces (S5, D8 universe, mean net R per trade)

| unit | parameter | x0.5 | x0.8 | x1.2 | x1.5 | base |
|---|---|---|---|---|---|---|
| tsmom | horizons | +0.038 | +0.076 | +0.112 | +0.158 | +0.066 |
| tsmom | scale | +0.042 | +0.061 | +0.076 | +0.069 | +0.066 |
| tsmom | _theta | +0.041 | +0.061 | +0.077 | +0.069 | +0.066 |
| tsmom | _stopk | +0.077 | +0.072 | +0.057 | +0.044 | +0.066 |
| ema_stack | fast | +0.077 | +0.088 | +0.106 | +0.115 | +0.090 |
| ema_stack | mid | +0.041 | +0.077 | +0.115 | +0.153 | +0.090 |
| ema_stack | slow | +0.116 | +0.090 | +0.083 | +0.089 | +0.090 |
| ema_stack | k | +0.101 | +0.088 | +0.107 | +0.126 | +0.090 |
| ema_stack | _theta | +0.102 | +0.090 | +0.113 | +0.131 | +0.090 |
| ema_stack | _stopk | +0.051 | +0.091 | +0.113 | +0.093 | +0.090 |
| donchian_breakout | _stopk | +0.182 | +0.152 | +0.141 | +0.132 | +0.141 |
| donchian_breakout | entry_window | +0.114 | +0.155 | +0.186 | +0.260 | +0.141 |
| donchian_breakout | exit_window | +0.079 | +0.149 | +0.144 | +0.193 | +0.141 |
| donchian_breakout | scale | +0.141 | +0.141 | +0.141 | +0.141 | +0.141 |
| donchian_breakout | _theta | +0.141 | +0.141 | +0.141 | +0.141 | +0.141 |
| compression_breakout | ratio | -0.946 | -0.941 | -0.946 | -0.965 | -0.951 |
| compression_breakout | _stopk | -1.000 | -0.989 | -0.888 | -0.777 | -0.951 |
| compression_breakout | scale | -0.951 | -0.951 | -0.951 | -0.951 | -0.951 |
| compression_breakout | _theta | -0.951 | -0.951 | -0.951 | -0.951 | -0.951 |
| compression_breakout | history | -0.948 | -0.952 | -0.956 | -0.946 | -0.951 |
| compression_breakout | window | -0.810 | -0.902 | -0.970 | -0.983 | -0.951 |
| geo_struct | max_retr | +0.192 | +0.192 | +0.192 | +0.192 | +0.192 |
| geo_struct | _stopk | +0.479 | +0.271 | +0.192 | +0.120 | +0.192 |
| geo_struct | vol_hl | +0.167 | +0.137 | +0.126 | +0.301 | +0.192 |
| geo_struct | _theta | +0.445 | +0.192 | +0.192 | -0.093 | +0.192 |
| geo_struct | ks | +0.262 | +0.224 | +0.115 | +0.379 | +0.192 |
| geo_pivot_bo | _theta | +0.658 | +0.689 | +0.689 | +0.692 | +0.689 |
| geo_pivot_bo | vol_hl | +0.622 | +0.655 | +0.643 | +0.656 | +0.689 |
| geo_pivot_bo | ks | +0.317 | +0.722 | +0.867 | +0.368 | +0.689 |
| geo_pivot_bo | max_retr | +0.689 | +0.689 | +0.689 | +0.689 | +0.689 |
| geo_pivot_bo | _stopk | +0.764 | +0.835 | +0.536 | +0.425 | +0.689 |
| geo_flag_bo | vol_hl | +0.473 | +0.440 | +0.319 | +0.329 | +0.535 |
| geo_flag_bo | ks | +0.274 | +0.295 | +0.265 | +0.716 | +0.535 |
| geo_flag_bo | _stopk | +0.429 | +0.596 | +0.514 | +0.415 | +0.535 |
| geo_flag_bo | _theta | +0.431 | +0.535 | +0.535 | +0.444 | +0.535 |
| geo_flag_bo | max_retr | +0.538 | +0.650 | +0.638 | +0.549 | +0.535 |
| zoo_ensemble | bpd | +0.321 | +0.515 | +0.500 | +0.749 | +0.538 |
| zoo_ensemble | _theta | +0.524 | +0.575 | +0.603 | +0.598 | +0.538 |
| zoo_ensemble | _stopk | +0.869 | +0.598 | +0.512 | +0.457 | +0.538 |
| trend_4h | k | +0.112 | +0.122 | +0.094 | +0.091 | +0.153 |
| trend_4h | fast | +0.111 | +0.130 | +0.142 | +0.117 | +0.153 |
| trend_4h | stop_mult | +0.035 | +0.085 | +0.143 | +0.128 | +0.153 |
| trend_4h | tp_r | +0.066 | +0.111 | +0.138 | +0.182 | +0.153 |
| trend_4h | mid | +0.123 | +0.142 | +0.148 | +0.154 | +0.153 |
| trend_4h | thr | +0.115 | +0.125 | +0.092 | +0.095 | +0.153 |
| trend_4h | slow | +0.103 | +0.119 | +0.119 | +0.114 | +0.153 |
| zscore_reversion | exit | -0.136 | -0.129 | -0.125 | -0.119 | -0.126 |
| zscore_reversion | scale | -0.126 | -0.126 | -0.126 | -0.126 | -0.126 |
| zscore_reversion | _theta | -0.126 | -0.126 | -0.126 | -0.126 | -0.126 |
| zscore_reversion | entry | -0.056 | -0.107 | -0.137 | -0.181 | -0.126 |
| zscore_reversion | span | -0.105 | -0.112 | -0.124 | -0.128 | -0.126 |
| zscore_reversion | _stopk | -0.151 | -0.144 | -0.110 | -0.096 | -0.126 |
| zscore_reversion | std_window | -0.100 | -0.128 | -0.106 | -0.103 | -0.126 |
| rsi_reversion | period | -0.043 | -0.039 | -0.056 | -0.077 | -0.045 |
| rsi_reversion | band | -0.061 | -0.055 | -0.059 | -0.066 | -0.045 |
| rsi_reversion | full | -0.035 | -0.039 | -0.052 | -0.062 | -0.045 |
| rsi_reversion | _theta | -0.035 | -0.039 | -0.052 | -0.062 | -0.045 |
| rsi_reversion | _stopk | -0.071 | -0.050 | -0.038 | -0.033 | -0.045 |
| setup_c_proxy | level_hours | -0.128 | -0.121 | -0.142 | -0.140 | -0.110 |
| setup_c_proxy | wait_min | -0.136 | -0.106 | -0.113 | -0.097 | -0.110 |
| setup_c_proxy | reclaim_hours | -0.109 | -0.115 | -0.102 | -0.098 | -0.110 |
| setup_c_proxy | buffer_sigma | -0.117 | -0.117 | -0.113 | -0.113 | -0.110 |
| setup_c_proxy | time_stop_hours | -0.097 | -0.100 | -0.122 | -0.124 | -0.110 |
| funding_contrarian | _stopk | -0.072 | -0.036 | -0.016 | -0.013 | -0.016 |
| funding_contrarian | window | -0.010 | -0.013 | -0.024 | -0.037 | -0.016 |
| funding_contrarian | threshold | -0.034 | -0.032 | -0.003 | -0.008 | -0.016 |
| funding_contrarian | scale | -0.016 | -0.016 | -0.016 | -0.016 | -0.016 |
| funding_contrarian | extension | -0.021 | -0.017 | -0.030 | -0.035 | -0.016 |
| funding_contrarian | _theta | -0.016 | -0.016 | -0.016 | -0.016 | -0.016 |
| oi_price_divergence | horizon | -0.015 | -0.027 | -0.010 | -0.020 | -0.022 |
| oi_price_divergence | window | -0.016 | -0.015 | -0.017 | -0.007 | -0.022 |
| oi_price_divergence | threshold | -0.034 | -0.027 | -0.009 | +0.012 | -0.022 |
| oi_price_divergence | scale | -0.031 | -0.027 | -0.016 | +0.006 | -0.022 |
| oi_price_divergence | squeeze_weight | +0.008 | -0.013 | -0.025 | -0.030 | -0.022 |
| oi_price_divergence | _theta | -0.030 | -0.030 | -0.009 | +0.023 | -0.022 |
| oi_price_divergence | _stopk | -0.031 | -0.030 | -0.019 | -0.019 | -0.022 |
| hrb2 | time_stop_hours | -0.048 | -0.024 | -0.025 | -0.059 | -0.031 |
| hrb2 | target_r | +0.068 | -0.000 | -0.058 | -0.039 | -0.031 |
| hrb2 | buffer_sigma | -0.024 | -0.052 | -0.027 | -0.024 | -0.031 |
| hrb2 | ema_span | +0.010 | -0.045 | +0.035 | +0.030 | -0.031 |
| hrb2 | r48_pct | +0.054 | -0.020 | -0.036 | -0.026 | -0.031 |
| hrb2 | oi_drop | -0.062 | -0.017 | +0.010 | +0.021 | -0.031 |
| hrb2 | rb2_window_bars | +0.015 | -0.049 | -0.032 | -0.039 | -0.031 |
| hrb2 | rebound_sigma | -0.020 | -0.031 | -0.032 | -0.039 | -0.031 |
| shock_regime | z_min | +0.005 | +0.021 | +0.071 | +0.082 | +0.051 |
| shock_regime | z_full | +0.051 | +0.051 | +0.051 | +0.051 | +0.051 |
| shock_regime | decluster_bars | +0.042 | +0.048 | +0.059 | +0.062 | +0.051 |
| shock_regime | trail_sigma | +0.030 | +0.038 | +0.065 | +0.097 | +0.051 |
| shock_regime | invalid_sigma | +0.053 | +0.050 | +0.051 | +0.053 | +0.051 |
| shock_regime | _theta | +0.051 | +0.051 | +0.051 | +0.051 | +0.051 |
| shock_regime | _stopk | +0.092 | +0.066 | +0.048 | +0.038 | +0.051 |
| shock_desk | z_min | -0.126 | -0.070 | -0.001 | +0.029 | -0.032 |
| shock_desk | decluster_bars | -0.036 | -0.034 | -0.022 | -0.022 | -0.032 |
| shock_desk | rev_buffer | -0.025 | -0.020 | -0.012 | -0.006 | -0.032 |
| shock_desk | cont_tp_r | -0.071 | -0.048 | -0.010 | -0.001 | -0.032 |
| hs2_fade_blackout | rev_buffer | -0.143 | -0.141 | -0.131 | -0.108 | -0.150 |
| hs2_fade_blackout | z_min | -0.178 | -0.179 | -0.076 | -0.108 | -0.150 |
| hs2_fade_blackout | blackout_bars | -0.150 | -0.150 | -0.150 | -0.080 | -0.150 |
| hs2_fade_blackout | decluster_bars | -0.166 | -0.156 | -0.137 | -0.132 | -0.150 |
| weekend_shock_follow | z_min | -0.060 | -0.203 | -0.303 | -0.269 | -0.030 |
| weekend_shock_follow | z_full | -0.030 | -0.030 | -0.030 | -0.030 | -0.030 |
| weekend_shock_follow | vol_ratio_min | -0.021 | -0.119 | -0.045 | +0.013 | -0.030 |
| weekend_shock_follow | _theta | -0.030 | -0.030 | -0.030 | -0.030 | -0.030 |
| weekend_shock_follow | _stopk | -0.078 | +0.030 | -0.050 | -0.033 | -0.030 |
| weekend_drift_fade | z_min | -0.101 | -0.101 | -0.051 | +0.182 | -0.059 |
| weekend_drift_fade | z_full | -0.059 | -0.059 | -0.043 | -0.081 | -0.059 |
| weekend_drift_fade | vol_ratio_max | -0.127 | -0.004 | -0.123 | -0.131 | -0.059 |
| weekend_drift_fade | _theta | -0.059 | -0.059 | -0.043 | -0.081 | -0.059 |
| weekend_drift_fade | _stopk | -0.145 | -0.135 | -0.002 | -0.011 | -0.059 |
| cme_gap_fill | z_min | -0.202 | -0.202 | -0.202 | -0.194 | -0.202 |
| cme_gap_fill | _stopk | -0.194 | -0.228 | -0.153 | -0.125 | -0.202 |
| cme_gap_fill | _theta | -0.185 | -0.185 | -0.183 | -0.243 | -0.202 |
| cme_gap_fill | z_full | -0.185 | -0.185 | -0.183 | -0.243 | -0.202 |
| flow_momentum | scale | +0.011 | +0.057 | +0.079 | +0.131 | +0.085 |
| flow_momentum | _theta | +0.010 | +0.054 | +0.084 | +0.119 | +0.085 |
| flow_momentum | window | -0.044 | +0.023 | +0.079 | +0.075 | +0.085 |
| flow_momentum | _stopk | +0.192 | +0.114 | +0.062 | +0.054 | +0.085 |
| cvd_divergence | window | -0.037 | -0.405 | +0.680 | n/a | +0.590 |
| cvd_divergence | scale | +0.196 | +0.006 | n/a | n/a | +0.590 |
| cvd_divergence | _stopk | +1.114 | +0.727 | +0.496 | +0.400 | +0.590 |
| cvd_divergence | _theta | +0.196 | +0.006 | n/a | n/a | +0.590 |

## Synthetic paths (S6)

One-year 1h paths after a 2000-bar warm-up; per path the unit's mean net R per trade and total R. Historical comparator: median over symbols of total R per symbol-year and max drawdown per symbol-year on real data.

| unit | gen | paths | median mean R | P(mean <= 0) | median total R | 5th pct total R | 5th pct max DD | hist total R / sym-yr | hist max DD / sym-yr |
|---|---|---|---|---|---|---|---|---|---|
| tsmom | SBB | 500 | +0.026 | 44% | +5.2 | -35.1 | -46.9 | +6.1 | -10.5 |
| tsmom | RS | 500 | -0.079 | 77% | -15.4 | -48.5 | -56.4 | +6.1 | -10.5 |
| ema_stack | SBB | 500 | +0.045 | 42% | +4.6 | -27.2 | -35.3 | +6.5 | -7.0 |
| ema_stack | RS | 500 | -0.079 | 66% | -8.3 | -32.8 | -36.6 | +6.5 | -7.0 |
| donchian_breakout | SBB | 500 | +0.048 | 45% | +2.4 | -23.6 | -27.5 | +2.2 | -6.2 |
| donchian_breakout | RS | 500 | -0.117 | 69% | -6.4 | -25.7 | -30.1 | +2.2 | -6.2 |
| compression_breakout | SBB | 414 | -0.500 | 58% | -5.0 | -26.7 | -28.4 | -6.2 | -8.6 |
| compression_breakout | RS | 429 | -0.678 | 65% | -5.4 | -33.0 | -34.2 | -6.2 | -8.6 |
| geo_struct | SBB | 500 | -0.075 | 55% | -1.0 | -10.5 | -10.9 | +0.1 | -3.1 |
| geo_struct | RS | 500 | -0.121 | 59% | -1.4 | -10.3 | -11.1 | +0.1 | -3.1 |
| geo_pivot_bo | SBB | 500 | +0.009 | 50% | +0.1 | -8.8 | -9.5 | +1.6 | -2.3 |
| geo_pivot_bo | RS | 500 | -0.235 | 61% | -2.1 | -9.8 | -10.2 | +1.6 | -2.3 |
| geo_flag_bo | SBB | 499 | -0.167 | 57% | -1.4 | -7.8 | -8.2 | +1.3 | -1.8 |
| geo_flag_bo | RS | 500 | -0.257 | 61% | -1.9 | -7.9 | -8.4 | +1.3 | -1.8 |
| zoo_ensemble | SBB | 500 | +0.145 | 39% | +2.7 | -12.4 | -14.8 | +5.7 | -3.1 |
| zoo_ensemble | RS | 500 | -0.107 | 58% | -2.2 | -15.5 | -16.9 | +5.7 | -3.1 |
| trend_4h | SBB | 500 | +0.028 | 46% | +0.6 | -11.6 | -15.4 | +3.8 | -3.8 |
| trend_4h | RS | 500 | -0.016 | 52% | -0.4 | -15.1 | -17.8 | +3.8 | -3.8 |
| zscore_reversion | SBB | 500 | -0.105 | 88% | -8.4 | -20.5 | -22.2 | -7.0 | -7.9 |
| zscore_reversion | RS | 500 | -0.018 | 57% | -1.4 | -14.5 | -17.1 | -7.0 | -7.9 |
| rsi_reversion | SBB | 500 | -0.045 | 70% | -3.3 | -15.2 | -16.5 | -3.2 | -4.4 |
| rsi_reversion | RS | 500 | -0.021 | 63% | -1.5 | -10.1 | -12.2 | -3.2 | -4.4 |
| shock_regime | SBB | 500 | +0.001 | 49% | +0.1 | -12.8 | -16.7 | +3.8 | -2.4 |
| shock_regime | RS | 500 | -0.057 | 71% | -4.1 | -16.1 | -18.1 | +3.8 | -2.4 |
| shock_desk | SBB | 500 | -0.090 | 67% | -5.7 | -25.5 | -31.0 | +2.6 | -5.5 |
| shock_desk | RS | 500 | -0.128 | 79% | -9.1 | -27.0 | -30.6 | +2.6 | -5.5 |
| hs2_fade_blackout | SBB | 500 | -0.205 | 72% | -3.2 | -11.0 | -11.5 | -1.1 | -2.3 |
| hs2_fade_blackout | RS | 500 | -0.134 | 65% | -2.2 | -9.7 | -11.1 | -1.1 | -2.3 |

## Latency model check (BTCUSDT, 1h units): Brownian-bridge fills vs exact 5m-bar fills

| unit | delay | trades | base | bridge | exact 5m |
|---|---|---|---|---|---|
| tsmom | lat5 | 1209 | +0.121 | +0.124 | +0.119 |
| tsmom | lat15 | 1209 | +0.121 | +0.124 | +0.119 |
| ema_stack | lat5 | 687 | +0.059 | +0.060 | +0.071 |
| ema_stack | lat15 | 687 | +0.059 | +0.062 | +0.062 |
| donchian_breakout | lat5 | 324 | +0.378 | +0.389 | +0.382 |
| donchian_breakout | lat15 | 324 | +0.378 | +0.403 | +0.365 |
| compression_breakout | lat5 | 0 | n/a | n/a | n/a |
| compression_breakout | lat15 | 0 | n/a | n/a | n/a |
| zscore_reversion | lat5 | 496 | -0.161 | -0.163 | -0.143 |
| zscore_reversion | lat15 | 496 | -0.161 | -0.152 | -0.164 |
| rsi_reversion | lat5 | 518 | -0.149 | -0.148 | -0.143 |
| rsi_reversion | lat15 | 518 | -0.149 | -0.152 | -0.143 |
| funding_contrarian | lat5 | 226 | -0.046 | -0.045 | -0.040 |
| funding_contrarian | lat15 | 226 | -0.046 | -0.037 | -0.013 |
| oi_price_divergence | lat5 | 529 | -0.043 | -0.044 | -0.033 |
| oi_price_divergence | lat15 | 529 | -0.043 | -0.045 | -0.037 |
| shock_regime | lat5 | 477 | +0.028 | +0.024 | -0.003 |
| shock_regime | lat15 | 477 | +0.028 | +0.017 | -0.014 |
| shock_desk | lat5 | 457 | -0.147 | -0.179 | -0.180 |
| shock_desk | lat15 | 445 | -0.147 | -0.218 | -0.170 |
| hs2_fade_blackout | lat5 | 85 | -0.433 | -0.440 | -0.419 |
| hs2_fade_blackout | lat15 | 83 | -0.433 | -0.452 | -0.402 |
| weekend_shock_follow | lat5 | 3 | -0.190 | -0.197 | -0.221 |
| weekend_shock_follow | lat15 | 3 | -0.190 | -0.210 | -0.261 |
| weekend_drift_fade | lat5 | 16 | +0.193 | +0.185 | +0.158 |
| weekend_drift_fade | lat15 | 16 | +0.193 | +0.171 | +0.085 |
| cme_gap_fill | lat5 | 37 | -0.232 | -0.235 | -0.235 |
| cme_gap_fill | lat15 | 37 | -0.232 | -0.241 | -0.256 |

## Weekend MM quoters (existing replay results, mm_weekend_2026-11-21; not re-run)

Held mean net USD per symbol-day [day-clustered 95% CI] under the replay's own stress cells (queue model x latency). No verdict: the MM quoters are outside the registered verdict rules (they are not R-denominated trade streams).

| quoter | pro-rata 150 ms | pro-rata 300 ms | FIFO-pess. 150 ms | FIFO-pess. 300 ms | worst cell |
|---|---|---|---|---|---|
| B0 | +0.0158 [-0.0013, +0.0330] | +0.0196 [+0.0047, +0.0352] | +0.0093 [-0.0066, +0.0257] | +0.0091 [-0.0064, +0.0246] | +0.0091 |
| B0h | +0.0393 [-0.0410, +0.1611] | +0.0434 [-0.0391, +0.1689] | +0.0047 [-0.0868, +0.1266] | +0.0031 [-0.0877, +0.1224] | +0.0031 |
| H-MMW1:W1_A2 | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | +0.0000 |
| H-MMW2:W2_B0_k2_c10 | +0.0079 [+0.0015, +0.0167] | +0.0085 [+0.0019, +0.0180] | +0.0066 [+0.0008, +0.0137] | +0.0072 [+0.0011, +0.0150] | +0.0066 |
| H-MMW3:W3_flow30_t0.3_widen | +0.0089 [-0.0074, +0.0218] | +0.0089 [-0.0074, +0.0218] | +0.0072 [-0.0078, +0.0192] | +0.0072 [-0.0078, +0.0192] | +0.0072 |
| H-MMW4:W4_q0.25 | -1.2331 [-1.3354, -1.1394] | -1.2464 [-1.3500, -1.1478] | -1.1144 [-1.2423, -0.9997] | -1.1348 [-1.2426, -1.0331] | -1.2464 |

## Rule candidates (suggestions only; research/rules.yaml is not edited)

Cells where a unit loses in both segments (>= 10 trades each, pooled >= 50, pooled mean <= -0.05 R): candidates for "do not take unit X in cell Y".

| unit | cell | pooled mean R | n | llm_in_sample | clean_holdout | n holdout |
|---|---|---|---|---|---|---|
| compression_breakout | R1 | -0.779 | 123 | -0.755 | -0.978 | 13 |
| tsmom | R7 | -0.350 | 62 | -0.272 | -0.713 | 11 |
| shock_desk | R8 | -0.265 | 537 | -0.254 | -0.323 | 81 |
| setup_c_proxy | R7 | -0.237 | 102 | -0.280 | -0.006 | 16 |
| hs2_fade_blackout | R8 | -0.236 | 304 | -0.212 | -0.367 | 48 |
| shock_desk | R5 | -0.191 | 763 | -0.193 | -0.178 | 83 |
| zscore_reversion | R4 | -0.181 | 1424 | -0.178 | -0.219 | 104 |
| donchian_breakout | R6 | -0.149 | 589 | -0.121 | -0.408 | 59 |
| zscore_reversion | R5 | -0.120 | 1114 | -0.127 | -0.065 | 124 |
| setup_c_proxy | R8 | -0.113 | 123 | -0.110 | -0.143 | 14 |
| zscore_reversion | R1 | -0.112 | 968 | -0.114 | -0.071 | 37 |
| tsmom | R6 | -0.105 | 5076 | -0.099 | -0.156 | 492 |
| donchian_breakout | R8 | -0.101 | 488 | -0.028 | -0.605 | 61 |
| rsi_reversion | R4 | -0.085 | 1033 | -0.080 | -0.145 | 72 |
| shock_regime | R8 | -0.085 | 536 | -0.078 | -0.123 | 81 |
| zscore_reversion | R3 | -0.084 | 687 | -0.085 | -0.079 | 55 |
| funding_contrarian | R7 | -0.082 | 572 | -0.088 | -0.031 | 61 |
| shock_regime | R9 | -0.080 | 314 | -0.075 | -0.124 | 34 |
| rsi_reversion | R5 | -0.078 | 552 | -0.073 | -0.109 | 77 |
| rsi_reversion | R1 | -0.075 | 1119 | -0.075 | -0.088 | 52 |
| geo_struct | R4 | -0.070 | 147 | -0.025 | -0.685 | 10 |
| ema_stack | R9 | -0.067 | 1033 | -0.070 | -0.041 | 97 |
| oi_price_divergence | R8 | -0.066 | 646 | -0.068 | -0.047 | 62 |
| cme_gap_fill | R7 | -0.059 | 149 | -0.037 | -0.337 | 11 |
| hs2_fade_blackout | R3 | -0.056 | 204 | -0.027 | -0.240 | 28 |
| rsi_reversion | R3 | -0.051 | 952 | -0.049 | -0.062 | 152 |

## Hypotheses (recorded in research/hypotheses_results.jsonl)

| id | verdict | summary |
|---|---|---|
| H-STRESS1 | **supports** | met=7, of=9 |
| H-STRESS2 | **supports** | fails=5 |
| H-STRESS3 | **supports** | exceed=[] |
| H-STRESS4 | **supports** |  |
| H-STRESS5 | **inconclusive** | n_units=20 |
| H-STRESS6 | **supports** |  |
| H-STRESS7 | **supports** | share_no_cliff=0.9090909090909091 |
| H-STRESS8 | **contradicts** | share=0.3333333333333333 |
| H-STRESS9 | **contradicts** |  |
| H-STRESS10 | **inconclusive** | is_survivors=['shock_regime', 'geo_pivot_bo', 'zoo_ensemble'] |

## Caveats and what went wrong

- **The trade layer changes some strategies.** The live engine needs an exchange stop on every entry, so every signal strategy was given the widest stop the engine allows (4 sigma of one bar at its own frequency) and the 0.3 entry threshold of the trend_4h sleeve. This is right for "what would the live engine do with this signal", but it is not the strategy as researched. `compression_breakout` is the extreme case: it holds a breakout until the close crosses the old box midpoint, often for months, so a 4-sigma hourly stop ends almost every trade at -1 R and the re-arm rule (no re-entry until the signal is flat) keeps it out afterwards. Its `no stop` column is the native strategy. The 0.3 threshold also trims `rsi_reversion` to its extreme readings.
- **No open gaps on hourly 24/7 bars.** A 1h perp bar opens where the previous one closed, so the registered gap-fill variant never differs from the stop-price fill (0 gapped stops in every unit). Real stop slippage in a crash happens inside the bar. The `stop next open` variant (fill at the next bar's open) and the drought scenario's stop slippage are the usable bounds; the first one often *helps* trend units, because stops are swept by wicks that close back.
- **Stops are checked on 1h bars.** Brackets take the stop first when stop and target are inside the same bar (conservative). Intrabar paths are unknown except for BTC 5m (latency check only).
- **Latency is modelled, not replayed.** Delayed fills use the Brownian-bridge conditional mean inside the next 1h bar. On BTC the bridge and exact 5m fills agree to within about 0.01-0.03 R per trade for most units; for `shock_regime` and `shock_desk` the exact fills are worse than the bridge at 5 minutes (post-shock drift is not Brownian), so S4 understates their latency cost.
- **The S5 perturbation run used the first walker**, which dropped positions still open at the end of the data (diary "mistake" entry, fixed before the final base run). Base and variants drop the same, at most one trade per symbol and variant, so the sign comparisons are internally consistent, but the D8 base means in the S5 table are not exactly the S0 numbers.
- **S5 is on D8, the rest on 23 symbols.** For `hrb2` the D8 base is negative while the 23-symbol base is positive, so the cliff rule (defined only when the D8 base is > 0) could not fire; that itself says its edge lives in the smaller alts.
- **Synthetic paths keep a lot of structure.** The Politis-White block length on |r| hits the cap (about 30 days of 1h bars) for every source, so stationary-bootstrap paths keep month-long stretches of real history: trends shorter than a month survive. They are a test of "does the unit survive other orderings of real months", not a test against a driftless null. Calendar, positioning and flow units were left out of S6 (resampling breaks the weekend alignment, and funding / OI / flow are not generated); `hrb2` and `setup_c_proxy` need OI or 15m bars.
- **Setup C is a proxy.** Its two order-flow confirmations (spot absorption, Coinbase premium) have no history, the user's chart levels are replaced by the prior-72h low, and the target by the nearest 15m pivot high. A loss here does not test the user's discretionary setup, only its bar skeleton.
- **The live engine simulation is simplified.** First come first served with ties broken by universe order; the minimum lot, partial fills, the 15-minute peg window and margin are not modelled; tiers use turnover only (no historical spreads; the engine also requires the spread condition). The EV gate is applied only where a probability exists (bracket setups); with the null probability `hrb2` and `setup_c_proxy` are refused entirely, and `trend_4h` / `shock_desk` lose most candidates to it. "P(day stop hit)" is about zero for every unit because 3 open positions x 1.30 USD cannot reach 10 USD in a day without several rounds of stops; the daily budget rule refuses an entry before the stop can be hit.
- **Universe survivorship.** Universe L is today's liquid survivors; LUNA, FTT and other delisted coins are not in it, so the crisis windows understate what an alt-heavy stream would have lost.
- **The 2020-03 window is BTC on Deribit only**, with about 70 days of warm-up from 2020-01-01: `zoo_ensemble` (69-day lookback) is barely warm and the regime labels are short-history.
- **Crisis windows overlap**: "BTC 14d drop to 2026-11-18" contains most of the "2026-11-21 flush" window.
- **Multiple testing.** 23 units x about 30 scenario cells: the verdicts are robustness screens with frozen thresholds, not evidence of edge. Data-generated units (`shock_regime`, `shock_desk`, `hs2_fade_blackout`, `hrb2`) were designed after their in-sample evidence was seen; only their forward results count.

## Method

- **Units.** 17 registered strategies at their configured frequency (1h; geo_* on 4h) plus `zoo_ensemble` (4h), through one generic trade
  layer; desk setups through a bracket layer: `trend_4h` (agent.session 4h ema_stack sleeve), `shock_desk` (agent.session shock candidates,
  market entry), `hs2_fade_blackout` (reversal leg, entry one 1h bar after the shock = the R-FLOW-2 60-minute blackout), `hrb2`
  (post_cascade_study H-RB2, re-implemented with parameters), `setup_c_proxy` (setup C without its spot-absorption and Coinbase-premium
  confirmations, which have no history). Weekend MM quoters: replay results only.
- **Trade layer.** Enter at the bar close when flat and |s| >= 0.3; exit at the first close with s x side <= 0; exchange stop 4.0 x
  sigma_risk_next per bar (the applied stop cap) from the fill, walked on 1h bars; no re-entry after a stop until the signal is flat or
  flips. R = (d (exit - entry) - c_in entry - c_out exit) / (|entry - stop| + entry (c_in + c_out at base costs)). Base costs = live: entry
  pegged limit 2.0 + 1.3 bp (shock units market 5.5 + 2.0), exits taker 5.5 + 2.0 bp, stops filled at the stop price.
- **Scenarios.** S1 slices of base trades; S2 crisis windows by entry date on the raw stream and on the S9 stream; S3 fees x1.5 / x2, +5 /
  +10 / +25 bp per side, gap fills at the open, stop filled at the next bar open, no exchange stop; S4 1 / 5 / 15 minute delays on entries and
  signal exits, filled at the Brownian-bridge conditional mean inside the next 1h bar (checked against BTC 5m bars above); S5 x0.5 / 0.8 /
  1.2 / 1.5 one at a time on every key parameter plus theta_in and stop_k, and 30 joint U(0.5, 1.5) draws, on D8; S6 1000 one-year 1h paths
  per unit (500 stationary block bootstrap with Politis-White block length on |r|, 500 regime-switching on the v0 cell chain; sources
  BTC ETH SOL XRP DOGE LINK); S7 in BTC down states (v0 R7 / R8 / R9) a stop-out closes every concurrent same-side trade of the unit at its
  stop; S8 mid / illiquid tiers (point-in-time trailing 30-day median turnover < 100M USD) pay spread x3 and stop slippage sqrt(2) x the
  tier's stop_slip_bps; S9 live_small applied to the pooled stream, first come first served, ties broken by universe order.
- **Statistics.** Mean R per trade with a stationary bootstrap of entry days (mean block 5 days, 2000 draws), which keeps the cross-symbol
  clustering of same-day trades; max drawdown on the realised curve in exit order; CVaR5% = mean of the worst 5% trades / active days;
  time under water on the daily realised curve. Segments split by entry date at research.knowledge_cutoff (2026-07-01).
- **Verdicts (frozen).** FAILS: base mean <= 0 or fees x1.5 mean <= 0. FRAGILE: +-20% parameter cliff (D8), 5-minute latency mean <= 0,
  a crisis window < -20 R on the engine stream, +5 bp mean <= 0, clean_holdout mean <= 0 with >= 20 trades, base CI includes 0, or the total
  without the 5 best trades <= 0. Else SURVIVES. Other scenarios are reported, not part of the verdict.
- **Reproduce.** `uv run python -m ai_trader.research.stress design | run | perturb | synth | deribit | latcheck | analyse [--record]`;
  outputs in data/research/stress/. A new strategy: `ai_trader.research.stress.stress_signal(fn, symbols, freq)`.
