# Multi-day microstructure, batch 2: H-MD1..H-MD6 interim (overnight program 2026-11-12)

_data: data/research/micro_days (11 verified days: 2025-08-31, 2025-09-05, 2025-11-19, 2025-12-18, 2026-07-12, 2026-07-13, 2026-07-22, 2026-08-17, 2026-09-23, 2026-10-25, 2026-11-10) | segments: llm_in_sample ['2025-08-31', '2025-09-05', '2025-11-19', '2025-12-18'], clean_holdout ['2026-07-12', '2026-07-13', '2026-07-22', '2026-08-17', '2026-09-23', '2026-10-25', '2026-11-10'] | code: src/ai_trader/research/micro_days.py, src/ai_trader/research/multiday_micro_study.py, scripts/micro_batch.py | tests: tests/test_micro_days.py_

## Changes since batch 1 (11 days; read this first)

- **Batch 2 processing.** 5 new days, all verified, raw deleted day by day; peak raw on disk was about 4.8 GB. With the fixed scheduler each day was verified 20-40 min after its downloads finished. Lake plus research is 2.51 GB; micro_days is 1.26 GB.
- **Regimes.** The new holdout days are quiet: 2026-07-22, 2026-08-17 and 2026-10-25 (a Sunday) have 0.5-1.0 major events/s and DY totals of 41-77. The new in-sample days are 2025-08-31 (a Sunday, quiet) and 2025-11-19 (busy, 5.6 events/s, BTC sigma_day 247 bp). In-sample days are still busier on average.
- **H-MD1.** Unchanged: gamma 0.29, acf1 0.30, zeta 0.23, mixture weights 0.19. Day-to-day variation dominates.
- **H-MD2.**
  - Rate scaling remains very stable (slope about 0.54, LOO R^2 0.92) and impact LOO R^2 is 0.63.
  - The day x log W interaction is still significant pooled (Wald p 0.0003 for rate, 0.026 for impact). This part keeps failing.
- **H-MD3.**
  - The share dropped below the 0.8 line: 350/438 = 0.799 pooled and 0.783 in holdout. The quiet days have fewer symbols where a power-law ACF is preferred.
  - Hurst is 0.542 [0.532, 0.553], comfortably above 0.5.
- **H-MD4.** Weakened: 20 qualifying symbol-days, gain > 0 on only 10/20 at both 5 and 15 min (0/3 in-sample, 10/17 holdout). The pilot's free-spread pattern does not generalise so far.
- **H-MD5.** ETH and SOL are net transmitters on 11/11 days.
- **H-MD6.** v1 beats v0 on 18/29 class-days (0.62, under 0.7), but majors reach 4/5 on 0/11 days. The majors part is far from its 50% requirement but not yet locked.
- **Nothing recorded.** No share criterion is lost for good with 20 planned days left; final verdicts follow the last batch.

## Status of each hypothesis

Interim numbers only: a verdict is recorded when a share criterion is already lost for good given the remaining pre-drawn days, or after the last batch (with Holm over the six). Not final.

| id | segment | criterion so far | summary p | planned days left | recorded |
|---|---|---|---|---|---|
| H-MD1 | llm_in_sample | not met | 0.999 | 8 | not recorded |
| H-MD1 | clean_holdout | not met | 1.000 | 12 | not recorded |
| H-MD1 | pooled | not met | 1.000 | 20 | not recorded |
| H-MD2 | llm_in_sample | met | 0.064 | 8 | not recorded |
| H-MD2 | clean_holdout | not met | 0.107 | 12 | not recorded |
| H-MD2 | pooled | not met | 0.067 | 20 | not recorded |
| H-MD3 | llm_in_sample | met | 0.197 | 8 | not recorded |
| H-MD3 | clean_holdout | not met | 0.790 | 12 | not recorded |
| H-MD3 | pooled | not met | 0.547 | 20 | not recorded |
| H-MD4 | llm_in_sample | not met | 1.000 | 8 | not recorded |
| H-MD4 | clean_holdout | not met | 0.895 | 12 | not recorded |
| H-MD4 | pooled | not met | 0.983 | 20 | not recorded |
| H-MD5 | llm_in_sample | met | 0.410 | 8 | not recorded |
| H-MD5 | clean_holdout | met | 0.210 | 12 | not recorded |
| H-MD5 | pooled | met | 0.086 | 20 | not recorded |
| H-MD6 | llm_in_sample | not met | 1.000 | 8 | not recorded |
| H-MD6 | clean_holdout | not met | 1.000 | 12 | not recorded |
| H-MD6 | pooled | not met | 1.000 | 20 | not recorded |

Pooled summary: MD1 ICC gamma 0.29, acf1 0.30, zeta 0.23, mix_weights 0.19; MD2 LOO R2 rate 0.92 impact 0.63 Wald p 0.000/0.026; MD3 350/438 power-law, Hurst 0.542 [0.532, 0.553]; MD4 qualifying 20, gain>0 10/20 (5m) 10/20 (15m); MD5 net>0 ETH 11/11 SOL 11/11; MD6 v1>v0 18/29 class-days, majors>=4/5 0/11

## Processing and storage

- **batch days**: 2025-08-31 (verified True, raw deleted True), 2025-11-19 (verified True, raw deleted True), 2026-07-22 (verified True, raw deleted True), 2026-08-17 (verified True, raw deleted True), 2026-10-25 (verified True, raw deleted True)
- **storage before (GB)**: {"lake_gb": 1.139, "research_gb": 0.83, "micro_days_gb": 0.744, "raw_gb": 0.399, "lake_plus_research_gb": 1.969, "trades_raw_gb": 0.432}
- **storage after (GB)**: {"lake_gb": 1.159, "research_gb": 1.349, "micro_days_gb": 1.264, "raw_gb": 0.399, "lake_plus_research_gb": 2.508, "trades_raw_gb": 0.432}

## H-MD1: persistence of trade-level traits (ICC, symbol as group)

| segment | trait (TEST half) | ICC | CI lo | CI hi | symbols | symbol-days | mean | sd between | sd within |
|---|---|---|---|---|---|---|---|---|---|
| llm_in_sample | gamma | 0.348 | 0.158 | 0.504 | 38 | 150 | 0.750 | 0.183 | 0.208 |
| llm_in_sample | acf1 | 0.737 | 0.604 | 0.814 | 38 | 150 | 0.410 | 0.086 | 0.049 |
| llm_in_sample | zeta | 0.227 | 0.048 | 0.403 | 38 | 150 | 1.973 | 0.361 | 0.494 |
| llm_in_sample | w1 | 0.340 | 0.158 | 0.464 | 38 | 150 | 0.298 | 0.057 | 0.066 |
| llm_in_sample | w2 | -0.097 | -0.186 | -0.013 | 38 | 150 | 0.322 | 0.045 | 0.111 |
| llm_in_sample | w3 | 0.053 | -0.089 | 0.177 | 38 | 150 | 0.380 | 0.073 | 0.133 |
| llm_in_sample | mix_weights | 0.099 | 0.010 | 0.163 | n/a | n/a | n/a | n/a | n/a |
| clean_holdout | gamma | 0.369 | 0.258 | 0.465 | 51 | 355 | 0.728 | 0.191 | 0.227 |
| clean_holdout | acf1 | 0.427 | 0.256 | 0.572 | 51 | 357 | 0.382 | 0.070 | 0.075 |
| clean_holdout | zeta | 0.297 | 0.144 | 0.416 | 51 | 357 | 1.640 | 0.334 | 0.449 |
| clean_holdout | w1 | 0.443 | 0.152 | 0.610 | 51 | 357 | 0.274 | 0.066 | 0.069 |
| clean_holdout | w2 | 0.235 | 0.125 | 0.316 | 51 | 357 | 0.294 | 0.064 | 0.096 |
| clean_holdout | w3 | 0.208 | 0.079 | 0.320 | 51 | 357 | 0.432 | 0.074 | 0.118 |
| clean_holdout | mix_weights | 0.295 | 0.184 | 0.368 | n/a | n/a | n/a | n/a | n/a |
| pooled | gamma | 0.287 | 0.191 | 0.377 | 51 | 506 | 0.735 | 0.170 | 0.235 |
| pooled | acf1 | 0.298 | 0.180 | 0.398 | 51 | 508 | 0.390 | 0.058 | 0.083 |
| pooled | zeta | 0.228 | 0.112 | 0.347 | 51 | 508 | 1.738 | 0.322 | 0.486 |
| pooled | w1 | 0.317 | 0.082 | 0.480 | 51 | 508 | 0.281 | 0.054 | 0.075 |
| pooled | w2 | 0.127 | 0.045 | 0.195 | 51 | 508 | 0.302 | 0.055 | 0.106 |
| pooled | w3 | 0.128 | 0.023 | 0.227 | 51 | 508 | 0.417 | 0.063 | 0.128 |
| pooled | mix_weights | 0.190 | 0.103 | 0.257 | n/a | n/a | n/a | n/a | n/a |

- robustness H-MD1_fit_half: gamma 0.240 [0.148, 0.325], acf1 0.285 [0.168, 0.380], zeta 0.196 [0.100, 0.295], mix_weights 0.191 [0.112, 0.256]
- robustness H-MD1_ge20k: gamma 0.301 [0.193, 0.399], acf1 0.322 [0.185, 0.435], zeta 0.176 [0.065, 0.311], mix_weights 0.220 [0.104, 0.295]

## H-MD2: activity scaling with W (FIT half, day fixed effects)

| segment | y | slope on log W | SE (cl. symbol) | LOO R^2 | LOO CI | Wald p (day x log W) | rows | symbols | days |
|---|---|---|---|---|---|---|---|---|---|
| llm_in_sample | rate | 0.526 | 0.021 | 0.910 | [0.613, 0.945] | 0.144 | 151 | 39 | 4 |
| llm_in_sample | impact | -0.511 | 0.129 | 0.641 | [0.447, 0.821] | 0.480 | 151 | 39 | 4 |
| llm_in_sample | depth | 0.708 | 0.119 | 0.773 | [-0.531, 0.972] | 0.054 | 28 | 7 | 4 |
| clean_holdout | rate | 0.531 | 0.017 | 0.914 | [0.798, 0.941] | 6.55e-04 | 357 | 51 | 7 |
| clean_holdout | impact | -0.510 | 0.084 | 0.614 | [0.431, 0.767] | 0.054 | 356 | 51 | 7 |
| clean_holdout | depth | 0.793 | 0.158 | 0.658 | [0.235, 0.908] | 0.153 | 63 | 9 | 7 |
| pooled | rate | 0.530 | 0.015 | 0.920 | [0.87, 0.95] | 3.25e-04 | 508 | 51 | 11 |
| pooled | impact | -0.510 | 0.086 | 0.633 | [0.446, 0.768] | 0.026 | 507 | 51 | 11 |
| pooled | depth | 0.767 | 0.127 | 0.734 | [0.081, 0.905] | 0.000 | 91 | 9 | 11 |

Per-day slopes (pilot 2026-11-10 one-day values: rate ~W^0.54, depth ~W^0.68, impact ~W^-0.75):

| day | rate slope | impact slope | depth slope | rate SE | impact SE |
|---|---|---|---|---|---|
| 2025-08-31 | 0.516 | -0.487 | n/a | 0.021 | 0.177 |
| 2025-09-05 | 0.529 | -0.503 | n/a | 0.026 | 0.179 |
| 2025-11-19 | 0.525 | -0.556 | n/a | 0.042 | 0.089 |
| 2025-12-18 | 0.546 | -0.521 | n/a | 0.033 | 0.184 |
| 2026-07-12 | 0.522 | -0.599 | 1.023 | 0.027 | 0.108 |
| 2026-07-13 | 0.524 | -0.651 | 0.802 | 0.022 | 0.084 |
| 2026-07-22 | 0.522 | -0.551 | 0.798 | 0.020 | 0.082 |
| 2026-08-17 | 0.522 | -0.477 | 0.880 | 0.028 | 0.102 |
| 2026-09-23 | 0.582 | -0.399 | 1.350 | 0.032 | 0.143 |
| 2026-10-25 | 0.525 | -0.486 | 1.130 | 0.028 | 0.129 |
| 2026-11-10 | 0.540 | -0.487 | 0.759 | 0.032 | 0.086 |

## H-MD3: order-sign long memory

| segment | power law & 0<gamma<1 | share | mean Hurst 1 s signed flow | CI lo (days) | CI hi | days |
|---|---|---|---|---|---|---|
| llm_in_sample | 119/143 | 0.832 | 0.548 | 0.535 | 0.560 | 4 |
| clean_holdout | 231/295 | 0.783 | 0.539 | 0.527 | 0.554 | 7 |
| pooled | 350/438 | 0.799 | 0.542 | 0.532 | 0.553 | 11 |

| day | k/n | share | mean Hurst |
|---|---|---|---|
| 2025-08-31 | 25/34 | 0.735 | 0.536 |
| 2025-09-05 | 31/34 | 0.912 | 0.534 |
| 2025-11-19 | 27/37 | 0.730 | 0.562 |
| 2025-12-18 | 36/38 | 0.947 | 0.559 |
| 2026-07-12 | 29/38 | 0.763 | 0.530 |
| 2026-07-13 | 34/41 | 0.829 | 0.517 |
| 2026-07-22 | 33/43 | 0.767 | 0.544 |
| 2026-08-17 | 28/37 | 0.757 | 0.531 |
| 2026-09-23 | 36/48 | 0.750 | 0.575 |
| 2026-10-25 | 27/40 | 0.675 | 0.536 |
| 2026-11-10 | 44/48 | 0.917 | 0.531 |

## H-MD4: spread -> next 5 / 15 min volatility on free-spread symbol-days

| segment | h (min) | qualifying | gain > 0 | share | median gain | sign-test p | non-qualifying share > 0 | non-qual. median gain |
|---|---|---|---|---|---|---|---|---|
| llm_in_sample | 5 | 3 | 0/3 | 0.000 | -0.021 | 1.000 | 0.515 | 5.92e-04 |
| llm_in_sample | 15 | 3 | 0/3 | 0.000 | -0.070 | 1.000 | 0.576 | 0.002 |
| clean_holdout | 5 | 17 | 10/17 | 0.588 | 0.005 | 0.315 | 0.358 | -0.003 |
| clean_holdout | 15 | 17 | 10/17 | 0.588 | 0.004 | 0.315 | 0.358 | -0.002 |
| pooled | 5 | 20 | 10/20 | 0.500 | 0.001 | 0.588 | 0.410 | -9.65e-04 |
| pooled | 15 | 20 | 10/20 | 0.500 | 0.001 | 0.588 | 0.430 | -0.002 |

| day | symbol | FIT share > 1 tick | qualifying | gain 5m | gain 15m | t TEST 5m |
|---|---|---|---|---|---|---|
| 2025-08-31 | DOGEUSDT | 0.002 | no | 8.12e-04 | 0.008 | 0.672 |
| 2025-08-31 | SOLUSDT | 0.001 | no | 0.014 | 0.029 | 1.993 |
| 2025-08-31 | BTCUSDT | 6.02e-04 | no | 0.002 | 0.003 | 3.248 |
| 2025-08-31 | ETHUSDT | 0.004 | no | -0.003 | 0.002 | 1.629 |
| 2025-08-31 | HYPEUSDT | 0.125 | no | 0.003 | -0.002 | 1.820 |
| 2025-08-31 | XRPUSDT | 0.001 | no | 0.025 | 0.144 | 3.497 |
| 2025-08-31 | KASUSDT | 0.099 | no | -0.010 | -0.010 | 0.597 |
| 2025-08-31 | ATOMUSDT | 4.86e-04 | no | -0.016 | 0.007 | -1.124 |
| 2025-08-31 | CARVUSDT | 0.086 | no | 5.92e-04 | -0.017 | 1.460 |
| 2025-09-05 | DOGEUSDT | 0.002 | no | -0.010 | -0.011 | 0.598 |
| 2025-09-05 | SOLUSDT | 0.002 | no | -0.008 | -0.005 | -0.299 |
| 2025-09-05 | BTCUSDT | 0.001 | no | 0.007 | 0.005 | 1.750 |
| 2025-09-05 | ETHUSDT | 0.003 | no | 0.012 | 0.008 | 1.396 |
| 2025-09-05 | HYPEUSDT | 0.185 | no | -0.002 | -0.005 | 2.162 |
| 2025-09-05 | XRPUSDT | 8.80e-04 | no | -8.42e-04 | -0.007 | -0.363 |
| 2025-09-05 | KASUSDT | 0.177 | no | 0.003 | -0.014 | 2.236 |
| 2025-09-05 | ATOMUSDT | 3.24e-04 | no | -0.004 | -0.011 | -1.894 |
| 2025-09-05 | CARVUSDT | 0.134 | no | 0.002 | 0.003 | 1.273 |
| 2025-11-19 | XRPUSDT | 0.001 | no | 0.061 | 0.172 | 0.312 |
| 2025-11-19 | SOLUSDT | 9.72e-04 | no | -0.002 | 0.001 | -0.227 |
| 2025-11-19 | ETHUSDT | 0.003 | no | 0.011 | 0.002 | 1.802 |
| 2025-11-19 | BTCUSDT | 0.002 | no | 0.002 | 6.71e-04 | -0.226 |
| 2025-11-19 | DOGEUSDT | 9.72e-04 | no | 0.006 | 0.020 | 0.551 |
| 2025-11-19 | HYPEUSDT | 0.207 | no | -0.043 | -0.026 | 0.291 |
| 2025-11-19 | KASUSDT | 0.293 | yes | -0.028 | -0.094 | 2.065 |
| 2025-11-19 | ATOMUSDT | 7.18e-04 | no | -0.082 | -0.090 | -1.022 |
| 2025-11-19 | CARVUSDT | 0.306 | yes | -0.021 | -0.070 | 0.094 |
| 2025-12-18 | XRPUSDT | 5.56e-04 | no | -0.027 | -0.050 | -1.097 |
| 2025-12-18 | ETHUSDT | 0.001 | no | 0.018 | 0.015 | 2.222 |
| 2025-12-18 | BTCUSDT | 0.001 | no | -4.11e-04 | 0.002 | 1.101 |
| 2025-12-18 | SOLUSDT | 4.86e-04 | no | 0.060 | 0.085 | -1.077 |
| 2025-12-18 | DOGEUSDT | 3.94e-04 | no | -0.161 | -0.629 | -1.281 |
| 2025-12-18 | HYPEUSDT | 0.094 | no | 0.018 | 0.037 | -1.403 |
| 2025-12-18 | ATOMUSDT | 2.55e-04 | no | -0.012 | 0.003 | -1.239 |
| 2025-12-18 | KASUSDT | 0.241 | no | -0.067 | -0.036 | -0.671 |
| 2025-12-18 | CARVUSDT | 0.443 | yes | -0.003 | -0.016 | 0.767 |
| 2026-07-12 | HYPEUSDT | 0.063 | no | 0.002 | 0.007 | -0.748 |
| 2026-07-12 | ETHUSDT | 6.71e-04 | no | -6.30e-04 | 3.29e-04 | -0.750 |
| 2026-07-12 | BTCUSDT | 5.79e-04 | no | 4.87e-04 | -9.93e-04 | -0.249 |
| 2026-07-12 | XRPUSDT | 0.000 | no | 0.005 | 0.027 | 1.650 |
| 2026-07-12 | DOGEUSDT | 6.94e-05 | no | -0.046 | -0.031 | 0.177 |
| 2026-07-12 | SOLUSDT | 2.31e-05 | no | -0.121 | -0.013 | 3.143 |
| 2026-07-12 | ATOMUSDT | 0.242 | no | -0.018 | -0.025 | 0.242 |
| 2026-07-12 | XAUUSDT | 0.000 | no | -16.971 | -126.516 | 4.519 |
| 2026-07-12 | NVDAUSDT | 0.024 | no | -0.003 | -0.035 | 2.915 |
| 2026-07-12 | QQQUSDT | 0.535 | yes | 0.016 | 0.004 | 2.902 |
| 2026-07-12 | CARVUSDT | 0.778 | yes | -0.001 | 0.003 | 1.158 |
| 2026-07-12 | KASUSDT | 0.106 | no | 0.002 | 0.007 | -1.539 |
| 2026-07-13 | HYPEUSDT | 0.053 | no | -0.038 | -0.071 | 0.886 |
| 2026-07-13 | ETHUSDT | 0.002 | no | -0.005 | -0.004 | -1.300 |
| 2026-07-13 | BTCUSDT | 0.002 | no | 0.002 | 0.002 | 1.797 |
| 2026-07-13 | SOLUSDT | 4.63e-05 | no | -3.75e-04 | 0.006 | -1.542 |
| 2026-07-13 | XRPUSDT | 6.94e-05 | no | -0.075 | -0.120 | -1.214 |
| 2026-07-13 | XAUUSDT | 5.09e-04 | no | 2.11e-04 | -7.94e-04 | 1.060 |
| 2026-07-13 | DOGEUSDT | 4.63e-05 | no | -0.010 | -0.041 | -1.350 |
| 2026-07-13 | NVDAUSDT | 0.165 | no | 0.003 | 0.004 | -1.409 |
| 2026-07-13 | ATOMUSDT | 0.428 | yes | -0.008 | 0.011 | -0.114 |
| 2026-07-13 | QQQUSDT | 0.739 | yes | 0.005 | 0.009 | -0.431 |
| 2026-07-13 | KASUSDT | 0.138 | no | -0.002 | -7.60e-05 | 1.314 |
| 2026-07-13 | CARVUSDT | 0.594 | yes | 0.006 | 0.027 | 1.667 |
| 2026-07-22 | HYPEUSDT | 0.003 | no | -0.013 | -0.018 | 2.537 |
| 2026-07-22 | ETHUSDT | 4.86e-04 | no | -6.98e-05 | -0.007 | -0.244 |
| 2026-07-22 | BTCUSDT | 4.17e-04 | no | -6.01e-04 | -0.001 | -0.257 |
| 2026-07-22 | XRPUSDT | 2.31e-05 | no | -0.012 | -0.010 | -1.531 |
| 2026-07-22 | SOLUSDT | 4.63e-05 | no | -8.79e-04 | -1.26e-04 | -0.707 |
| 2026-07-22 | DOGEUSDT | 6.94e-05 | no | -0.021 | -0.041 | -1.272 |
| 2026-07-22 | XAUUSDT | 4.63e-04 | no | -0.003 | 3.46e-04 | 0.515 |
| 2026-07-22 | NVDAUSDT | 0.114 | no | 0.002 | 0.003 | 1.496 |
| 2026-07-22 | ATOMUSDT | 0.192 | no | -0.001 | -0.002 | 0.689 |
| 2026-07-22 | QQQUSDT | 0.532 | yes | -0.023 | -0.034 | 1.257 |
| 2026-07-22 | KASUSDT | 0.073 | no | -0.008 | 1.76e-04 | 1.033 |
| 2026-07-22 | CARVUSDT | 0.425 | yes | 0.011 | -0.002 | 1.761 |
| 2026-08-17 | XRPUSDT | 0.000 | no | -0.022 | -0.035 | -0.902 |
| 2026-08-17 | ETHUSDT | 3.47e-04 | no | -0.023 | -0.025 | 0.748 |
| 2026-08-17 | BTCUSDT | 2.31e-04 | no | 1.94e-05 | -0.004 | 4.486 |
| 2026-08-17 | HYPEUSDT | 1.000 | yes | 0.022 | 0.048 | 0.545 |
| 2026-08-17 | DOGEUSDT | 0.000 | no | -0.111 | -0.216 | -0.820 |
| 2026-08-17 | SOLUSDT | 2.31e-05 | no | -0.010 | -0.002 | -0.490 |
| 2026-08-17 | XAUUSDT | 3.94e-04 | no | 8.71e-04 | 0.002 | 5.381 |
| 2026-08-17 | NVDAUSDT | 0.074 | no | 6.59e-04 | 0.010 | 1.625 |
| 2026-08-17 | QQQUSDT | 0.216 | no | -0.010 | -0.015 | -0.708 |
| 2026-08-17 | ATOMUSDT | 0.012 | no | -0.011 | -0.001 | -2.864 |
| 2026-08-17 | KASUSDT | 0.106 | no | -0.004 | -0.009 | -1.516 |
| 2026-08-17 | CARVUSDT | 0.789 | yes | -0.001 | -6.21e-04 | 1.039 |
| 2026-09-23 | XRPUSDT | 8.80e-04 | no | -0.306 | -0.454 | -1.865 |
| 2026-09-23 | ETHUSDT | 8.57e-04 | no | 0.036 | 0.144 | 2.192 |
| 2026-09-23 | BTCUSDT | 9.03e-04 | no | 0.007 | 0.003 | 1.027 |
| 2026-09-23 | DOGEUSDT | 5.79e-04 | no | -0.479 | -0.568 | -1.774 |
| 2026-09-23 | SOLUSDT | 2.55e-04 | no | -0.028 | -0.074 | -1.209 |
| 2026-09-23 | HYPEUSDT | 1.000 | yes | 0.031 | 0.057 | 1.470 |
| 2026-09-23 | XAUUSDT | 0.002 | no | -0.013 | 0.019 | -0.434 |
| 2026-09-23 | ATOMUSDT | 0.053 | no | -0.004 | -0.008 | -0.964 |
| 2026-09-23 | NVDAUSDT | 0.026 | no | -5.34e-04 | 0.001 | 0.358 |
| 2026-09-23 | QQQUSDT | 0.300 | yes | -0.008 | -0.004 | -0.527 |
| 2026-09-23 | KASUSDT | 0.882 | yes | 0.003 | 0.010 | 0.870 |
| 2026-09-23 | CARVUSDT | 0.994 | yes | 0.014 | 0.010 | 2.655 |
| 2026-10-25 | HYPEUSDT | 0.006 | no | -0.131 | -0.216 | 2.130 |
| 2026-10-25 | XRPUSDT | 1.39e-04 | no | -0.032 | -0.095 | 3.981 |
| 2026-10-25 | SOLUSDT | 2.31e-05 | no | -0.132 | -0.186 | 1.836 |
| 2026-10-25 | ETHUSDT | 4.17e-04 | no | 0.016 | 0.027 | 2.181 |
| 2026-10-25 | BTCUSDT | 2.08e-04 | no | -0.059 | -0.151 | 1.069 |
| 2026-10-25 | DOGEUSDT | 2.31e-05 | no | -0.062 | -0.243 | 3.534 |
| 2026-10-25 | XAUUSDT | 9.26e-05 | no | 2.69e-04 | -0.002 | 0.392 |
| 2026-10-25 | NVDAUSDT | 0.008 | no | 0.019 | 0.082 | 0.546 |
| 2026-10-25 | QQQUSDT | 0.006 | no | 0.001 | -0.002 | 2.615 |
| 2026-10-25 | ATOMUSDT | 0.004 | no | 0.005 | -0.005 | -0.594 |
| 2026-10-25 | KASUSDT | 0.303 | yes | -0.003 | -0.006 | -1.479 |
| 2026-10-25 | CARVUSDT | 0.909 | yes | -0.035 | -0.041 | 2.872 |
| 2026-11-10 | BTCUSDT | 5.32e-04 | no | 0.004 | -2.03e-04 | -1.466 |
| 2026-11-10 | ETHUSDT | 4.40e-04 | no | 5.91e-04 | 0.001 | -1.610 |
| 2026-11-10 | SOLUSDT | 1.16e-04 | no | -0.012 | -0.143 | -0.485 |
| 2026-11-10 | XRPUSDT | 2.08e-04 | no | -0.080 | -0.100 | -0.570 |
| 2026-11-10 | DOGEUSDT | 4.63e-05 | no | 0.111 | 0.277 | -0.515 |
| 2026-11-10 | HYPEUSDT | 0.016 | no | 0.059 | 0.068 | -0.022 |
| 2026-11-10 | CARVUSDT | 0.986 | yes | 0.014 | 0.007 | 3.479 |
| 2026-11-10 | KASUSDT | 0.589 | yes | 0.034 | -0.003 | 0.882 |
| 2026-11-10 | ATOMUSDT | 0.010 | no | 0.015 | 0.018 | 2.456 |
| 2026-11-10 | NVDAUSDT | 0.227 | no | -6.20e-04 | 0.002 | 0.016 |
| 2026-11-10 | QQQUSDT | 0.058 | no | 0.004 | 0.001 | -2.032 |
| 2026-11-10 | XAUUSDT | 0.013 | no | -0.043 | -0.026 | -3.297 |

## H-MD5: Diebold-Yilmaz net transmitters (log 1 m RV, full day)

| day | ETH net | SOL net | DOGE net | BTC net | ETH CI | SOL CI | total | ETH net TEST half | SOL net TEST half | symbols |
|---|---|---|---|---|---|---|---|---|---|---|
| 2025-08-31 | 21.944 | 9.412 | 21.122 | -23.136 | [15.4, 27.5] | [0.4, 16.9] | 69.874 | 15.965 | 8.291 | 13 |
| 2025-09-05 | 11.991 | 9.476 | 10.696 | -3.833 | [3.1, 19.3] | [-2.2, 17.5] | 81.257 | 10.120 | 11.343 | 13 |
| 2025-11-19 | 24.799 | 0.522 | 29.675 | 7.347 | [16.2, 30.6] | [-5.7, 8.3] | 83.010 | 19.133 | -1.083 | 15 |
| 2025-12-18 | 30.771 | 12.008 | 20.331 | 12.524 | [22.9, 38.6] | [3.4, 21.1] | 81.522 | 22.353 | -2.000 | 12 |
| 2026-07-12 | 25.631 | 16.551 | 26.341 | -6.392 | [18.3, 33.8] | [8.5, 23.7] | 59.113 | 23.870 | 13.897 | 12 |
| 2026-07-13 | 37.224 | 24.278 | 12.386 | 10.989 | [28.2, 44.8] | [15.3, 31.8] | 73.136 | 23.242 | 21.449 | 13 |
| 2026-07-22 | 23.619 | 18.455 | 15.047 | 9.891 | [15.1, 31.3] | [9.2, 25.8] | 59.341 | 28.480 | 19.697 | 11 |
| 2026-08-17 | 8.476 | 0.884 | n/a | 1.475 | [3.0, 14.8] | [-4.9, 5.5] | 40.699 | 16.925 | -0.017 | 6 |
| 2026-09-23 | 39.182 | 35.127 | 27.555 | 18.035 | [31.3, 46.8] | [27.0, 42.8] | 72.014 | 28.162 | 27.094 | 15 |
| 2026-10-25 | 40.147 | 25.042 | 10.151 | -22.161 | [30.4, 46.8] | [17.9, 31.1] | 76.613 | 41.600 | 22.476 | 15 |
| 2026-11-10 | 26.820 | 28.461 | 21.195 | 7.852 | [17.8, 32.0] | [19.5, 34.8] | 83.947 | 25.715 | 22.694 | 15 |

| segment | ETH net > 0 | SOL net > 0 | DOGE net > 0 | BTC net > 0 | mean BTC - ETH | CI (days) |
|---|---|---|---|---|---|---|
| llm_in_sample | 4/4 | 4/4 | 4/4 | 2/4 | -24.151 | [-38.2, -16.4] |
| clean_holdout | 7/7 | 7/7 | 6/6 | 5/7 | -25.916 | [-39.0, -15.4] |
| pooled | 11/11 | 11/11 | 10/10 | 7/11 | -25.274 | [-35.2, -17.2] |

## H-MD6: simulator v1 vs v0 on held-out half-days (12 L2 symbols)

| day | class | symbols | v1 inside (of 5) | v0 inside | v1 facts inside | v0 facts inside |
|---|---|---|---|---|---|---|
| 2025-08-31 | major | 5 | 1 | 1 | impact | impact |
| 2025-08-31 | alt | 4 | 2 | 1 | kurtosis, flow_acf | impact |
| 2025-09-05 | major | 5 | 2 | 0 | flow_acf, impact | - |
| 2025-09-05 | alt | 4 | 1 | 2 | flow_acf | flow_acf, impact |
| 2025-11-19 | major | 5 | 0 | 1 | - | impact |
| 2025-11-19 | alt | 4 | 2 | 0 | kurtosis, impact | - |
| 2025-12-18 | major | 5 | 1 | 2 | impact | impact, spread |
| 2025-12-18 | alt | 4 | 3 | 1 | kurtosis, flow_acf, impact | impact |
| 2026-07-12 | major | 5 | 1 | 1 | flow_acf | spread |
| 2026-07-12 | alt | 4 | 2 | 1 | flow_acf, impact | impact |
| 2026-07-12 | tradfi | 3 | 2 | 2 | kurtosis, spread | flow_acf, impact |
| 2026-07-13 | major | 5 | 2 | 0 | flow_acf, impact | - |
| 2026-07-13 | alt | 4 | 1 | 1 | flow_acf | flow_acf |
| 2026-07-13 | tradfi | 3 | 3 | 2 | kurtosis, flow_acf, impact | kurtosis, flow_acf |
| 2026-07-22 | major | 5 | 2 | 1 | flow_acf, spread | spread |
| 2026-07-22 | alt | 4 | 3 | 1 | kurtosis, flow_acf, impact | flow_acf |
| 2026-07-22 | tradfi | 3 | 3 | 1 | kurtosis, flow_acf, impact | kurtosis |
| 2026-08-17 | major | 5 | 3 | 1 | flow_acf, impact, spread | spread |
| 2026-08-17 | alt | 4 | 0 | 1 | - | flow_acf |
| 2026-08-17 | tradfi | 3 | 2 | 0 | kurtosis, flow_acf | - |
| 2026-09-23 | major | 5 | 0 | 2 | - | kurtosis, impact |
| 2026-09-23 | alt | 4 | 3 | 2 | kurtosis, flow_acf, impact | kurtosis, impact |
| 2026-09-23 | tradfi | 3 | 2 | 1 | kurtosis, flow_acf | flow_acf |
| 2026-10-25 | major | 5 | 3 | 2 | flow_acf, impact, spread | impact, spread |
| 2026-10-25 | alt | 4 | 1 | 2 | kurtosis | flow_acf, impact |
| 2026-10-25 | tradfi | 3 | 3 | 2 | flow_acf, impact, spread | flow_acf, impact |
| 2026-11-10 | major | 5 | 3 | 2 | kurtosis, flow_acf, impact | kurtosis, impact |
| 2026-11-10 | alt | 4 | 2 | 2 | kurtosis, flow_acf | flow_acf, impact |
| 2026-11-10 | tradfi | 3 | 5 | 0 | kurtosis, absr_acf, flow_acf, impact, spread | - |

| segment | v1 > v0 class-days | share | majors v1 >= 4/5 | mean v1 - v0 |
|---|---|---|---|---|
| llm_in_sample | 4/8 | 0.500 | 0/4 | 0.500 |
| clean_holdout | 14/21 | 0.667 | 0/7 | 0.905 |
| pooled | 18/29 | 0.621 | 0/11 | 0.793 |

Share of class-days with the fact inside the envelope: kurtosis v1 0.48 / v0 0.17, absr_acf v1 0.03 / v0 0.00, flow_acf v1 0.72 / v0 0.34, impact v1 0.55 / v0 0.52, spread v1 0.21 / v0 0.17

## Day profiles and differences from the 2026-11-10 pilot

| day | segment | symbols | major events/s TEST | major TEST/FIT events | BTC sigma_day (bp) | BTC notional (USD, FIT x2) | median gamma | median sign ACF1 | median zeta (prints) | median Hurst | major prints/order | median burst weight | DY total |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2025-08-31 | llm_in_sample | 37 | 1.953 | 0.964 | 104.397 | 4.05e+09 | 0.841 | 0.403 | 2.000 | 0.528 | 4.020 | 0.266 | 69.874 |
| 2025-09-05 | llm_in_sample | 37 | 4.949 | 1.822 | 143.915 | 7.34e+09 | 0.805 | 0.403 | 2.110 | 0.529 | 3.764 | 0.375 | 81.257 |
| 2025-11-19 | llm_in_sample | 38 | 5.551 | 1.834 | 247.433 | 8.10e+09 | 0.677 | 0.414 | 1.791 | 0.552 | 4.401 | 0.266 | 83.010 |
| 2025-12-18 | llm_in_sample | 39 | 7.009 | 3.670 | 177.934 | 3.80e+09 | 0.624 | 0.390 | 1.685 | 0.555 | 3.812 | 0.297 | 81.522 |
| 2026-07-12 | clean_holdout | 51 | 0.743 | 0.849 | 114.011 | 2.26e+09 | 0.668 | 0.396 | 1.734 | 0.517 | 4.124 | 0.277 | 59.113 |
| 2026-07-13 | clean_holdout | 51 | 1.244 | 0.966 | 212.754 | 5.26e+09 | 0.748 | 0.343 | 1.729 | 0.520 | 4.350 | 0.283 | 73.136 |
| 2026-07-22 | clean_holdout | 51 | 0.980 | 1.387 | 142.111 | 2.64e+09 | 0.668 | 0.395 | 1.676 | 0.529 | 4.572 | 0.286 | 59.341 |
| 2026-08-17 | clean_holdout | 51 | 0.536 | 0.926 | 125.470 | 3.43e+09 | 0.627 | 0.357 | 1.595 | 0.528 | 3.885 | 0.217 | 40.699 |
| 2026-09-23 | clean_holdout | 51 | 4.342 | 1.564 | 164.161 | 5.35e+09 | 0.683 | 0.391 | 1.432 | 0.580 | 4.939 | 0.320 | 72.014 |
| 2026-10-25 | clean_holdout | 51 | 0.817 | 1.118 | 95.562 | 1.69e+09 | 0.793 | 0.420 | 1.436 | 0.524 | 3.438 | 0.230 | 76.613 |
| 2026-11-10 | clean_holdout | 51 | 2.509 | 2.610 | 177.012 | 2.81e+09 | 0.640 | 0.376 | 1.756 | 0.535 | 4.041 | 0.249 | 83.947 |

Hayashi-Yoshida lead-lag (TEST half, crypto targets): share of peaks at lag 0 and in (0.1, 30] s:

| day | ETH peak at 0 | SOL peak at 0 | BTC peak at 0 | ETH positive peak | SOL positive peak | BTC positive peak | ETH median rho(0) |
|---|---|---|---|---|---|---|---|
| 2025-08-31 | 0.722 | 0.667 | 0.194 | 0.194 | 0.333 | 0.111 | 0.507 |
| 2025-09-05 | 0.889 | 0.917 | 0.889 | 0.083 | 0.083 | 0.056 | 0.473 |
| 2025-11-19 | 0.919 | 0.946 | 0.405 | 0.054 | 0.054 | 0.000 | 0.552 |
| 2025-12-18 | 0.868 | 0.895 | 0.658 | 0.079 | 0.079 | 0.079 | 0.576 |
| 2026-07-12 | 0.769 | 0.692 | 0.564 | 0.154 | 0.308 | 0.103 | 0.357 |
| 2026-07-13 | 0.821 | 0.718 | 0.718 | 0.103 | 0.231 | 0.103 | 0.442 |
| 2026-07-22 | 0.615 | 0.692 | 0.436 | 0.205 | 0.231 | 0.128 | 0.355 |
| 2026-08-17 | 0.564 | 0.487 | 0.538 | 0.231 | 0.308 | 0.205 | 0.339 |
| 2026-09-23 | 0.923 | 0.949 | 0.769 | 0.026 | 0.026 | 0.051 | 0.313 |
| 2026-10-25 | 0.821 | 0.846 | 0.564 | 0.077 | 0.128 | 0.077 | 0.611 |
| 2026-11-10 | 0.974 | 0.923 | 0.897 | 0.000 | 0.000 | 0.000 | 0.296 |

| day | free-spread L2 symbols (FIT share > 1 tick >= 25%) | missing trade archives (404) | missing ob200 (404) |
|---|---|---|---|
| 2025-08-31 | - | GRAMUSDT, MONUSDT, ASTERUSDT, CLUSDT, XAUUSDT, BZUSDT, XAGUSDT, QQQUSDT, NVDAUSDT, MSTRUSDT, TSLAUSDT, CRWVUSDT, COINUSDT, GOOGLUSDT | NVDAUSDT, XAUUSDT, QQQUSDT |
| 2025-09-05 | - | GRAMUSDT, MONUSDT, ASTERUSDT, CLUSDT, BZUSDT, XAUUSDT, XAGUSDT, NVDAUSDT, QQQUSDT, TSLAUSDT, MSTRUSDT, COINUSDT, CRWVUSDT, GOOGLUSDT | NVDAUSDT, QQQUSDT, XAUUSDT |
| 2025-11-19 | KASUSDT, CARVUSDT | GRAMUSDT, MONUSDT, CLUSDT, BZUSDT, XAUUSDT, XAGUSDT, QQQUSDT, NVDAUSDT, TSLAUSDT, COINUSDT, MSTRUSDT, CRWVUSDT, GOOGLUSDT | NVDAUSDT, QQQUSDT, XAUUSDT |
| 2025-12-18 | CARVUSDT | GRAMUSDT, CLUSDT, BZUSDT, XAUUSDT, XAGUSDT, QQQUSDT, NVDAUSDT, TSLAUSDT, MSTRUSDT, CRWVUSDT, COINUSDT, GOOGLUSDT | NVDAUSDT, QQQUSDT, XAUUSDT |
| 2026-07-12 | QQQUSDT, CARVUSDT | - | - |
| 2026-07-13 | ATOMUSDT, QQQUSDT, CARVUSDT | - | - |
| 2026-07-22 | QQQUSDT, CARVUSDT | - | - |
| 2026-08-17 | HYPEUSDT, CARVUSDT | - | - |
| 2026-09-23 | HYPEUSDT, QQQUSDT, KASUSDT, CARVUSDT | - | - |
| 2026-10-25 | KASUSDT, CARVUSDT | - | - |
| 2026-11-10 | CARVUSDT, KASUSDT | - | - |

## Aggregate schema (data/research/micro_days, read-only for other agents)

- **layout**: data/research/micro_days/<YYYY-MM-DD>/{trades_1s,book_1s,orders}/<SYMBOL>.parquet, laws/<SYMBOL>.json, sim/<SYMBOL>.json, day_stats.json, qa.json; a day is usable only if qa.json has verified.ok = true (micro_days.processed_days()); FIT = 00:00-12:00 UTC, TEST = 12:00-24:00 UTC; segment = llm_in_sample if date < 2026-07-01 else clean_holdout
- **trades_1s**: one row per UTC second WITH at least one print (sparse: missing seconds had no trade; reindex to 86400 rows yourself). ts Int64 = second OPEN in epoch ms (the bar covers [ts, ts + 1000)); open/high/low/close Float64 = first/max/min/last print price; vwap Float64; buy_vol/sell_vol Float64 = taker-buy / taker-sell base quantity; buy_notional/sell_notional Float64 = USD (price x size); n_trades/n_buy/n_sell Int32 = prints; n_orders/n_buy_orders/n_sell_orders Int32 = taker orders (consecutive same-ms same-side prints merged, counted in the second of the order's first print). Signed flow = buy_notional - sell_notional. Point in time: a bar is known at ts + 1000
- **book_1s**: L2 symbols only (BTC ETH SOL XRP DOGE HYPE CARV KAS ATOM NVDA QQQ XAU; NVDA/QQQ/XAU absent before their listing); one row per second, dense from the first message (forward-filled when no update). ts Datetime(ms, UTC) = END of the 1 s bin of the push time: the book state as of ts (known at ts); cts Int64 = matching-engine ms of the last applied message; bid1, ask1, bid1_size, ask1_size, mid, rel_spread_bp = (ask1 - bid1) / mid x 1e4; depth_{bid,ask}_{5,10,25} = USD notional within 5/10/25 bp of mid per side; imb_5, imb_10 = (bid - ask) / (bid + ask) depth at 5 / 10 bp; span_bid_bp, span_ask_bp = distance of the deepest of the 200 levels from mid (a depth band wider than the span is TRUNCATED: BTC ~2.7 bp, so BTC 5/10/25 bp depths are lower bounds); n_updates Int32 = messages in the bin; symbol Utf8. Join to trades_1s with book ts = trades ts + 1000 (state at the end of the trade second)
- **orders**: one row per taker order: ms Int64, side Int8 (+1 buy / -1 sell), size, notional (USD), n_prints Int32, first_price, last_price
- **laws/<SYM>.json**: halves{fit,test} in the schema of data/models/microstructure_laws/params_2026-11-10.json; tests (held-out comparisons); activity{fit,test} (W, sigma_day, orders_per_s, Kyle lambda, ...); tick (minimum price increment of the day)

## Methods

- **segments**: day < 2026-07-01 -> llm_in_sample, else clean_holdout; pooled = all days; the 2026-11-10 pilot day is clean_holdout
- **H-MD1**: per symbol-day TEST-half (12-24 UTC) estimates from laws/<SYM>.json tests: order-sign ACF power-law gamma, order-sign lag-1 ACF, print-size CSN tail index zeta, event-duration expmix3 weights (components ordered fastest first); ICC(1) one-way random effects with symbol as the group (unbalanced n0), symbols with >= 2 days; mixture trait = mean of the three weight ICCs; 95% CI by a bootstrap over symbols (B = 2000); a trait passes if ICC > 0.5 and CI lower > 0.3; supports = >= 3 of 4 traits; p = 3rd smallest of the trait p's (P*(ICC <= 0.5)); robustness: FIT half, symbol-days with >= 20k prints
- **H-MD2**: FIT-half activity rows (research.population_simulator_study._half_measures): W = sigma_day x P V_day; rate = log orders/s, impact = log Kyle lambda (mid proxy, bp per USD 1k; rows with lambda <= 0 dropped), depth = log median 25 bp book depth (L2 symbols, symbol-days whose 25 bp band is truncated in > 50% of the 1 s states dropped; descriptive); model y = 1 + log W + alt + tradfi + day FE; LOO R^2 leaving out all days of one symbol; bootstrap over symbols (B = 300) for P*(LOO R^2 <= 0.5); interaction model adds day x log W, Wald chi2 of the interactions with CR1 covariance clustered by symbol; supports = LOO R^2 > 0.5 for rate and impact and Wald p > 0.05 for both; p = max of the two LOO bootstrap p's
- **H-MD3**: symbol-days with >= 20k prints: order-sign ACF power law preferred on held-out TEST lags (fit on FIT, as H-MS3) and 0 < gamma_TEST < 1; share vs 0.8 (one-sided binomial); pooled Hurst = mean over the same symbol-days of the DFA-1 Hurst of 1 s signed notional (TEST half), 95% CI by a bootstrap over days (B = 2000); supports = share >= 0.8 and CI lower > 0.5; p = max(binomial p, P*(mean H <= 0.5))
- **H-MD4**: L2 symbol-days whose FIT-half 1 s states have spread > 1.5 ticks in >= 25% (qualifying, known before TEST); gain = OOS R^2 (TEST) of the FIT-estimated per-symbol regression with log spread minus without (micro_days.METHOD spread_vol), h = 5 and 15 min; supports = >= 10 qualifying symbol-days and gain > 0 on >= 70% of them at both horizons; p = max over h of the one-sided binomial p vs 0.7 (sign test vs 0.5 reported)
- **H-MD5**: daily Diebold-Yilmaz net directional connectedness (full UTC day, primary; halves reported) of log 1 m RV for the fixed H-MS6 token set; supports = ETH net > 0 and SOL net > 0 on >= 80% of days; p = max of the binomial p's vs 0.8; BTC vs ETH: mean of net_BTC - net_ETH over days with a day bootstrap CI (reported, not in the success line)
- **H-MD6**: L2 symbols; real 3 s grid from book_1s (mid at candle close, spread = mean of the 3 one-second states, 25 bp depth at close) and trades_1s (buy / sell notional, prints, high / low); facts as H-SIM2 / H-MS8 on the TEST half, envelope = [2.5, 97.5]% of the class median under a 1 h block bootstrap (B = 300); v1 = flow_sim_v1 built and calibrated on the FIT half exactly as population_simulator_study.stage_symbol_one (laws = that day's FIT laws, spread / tick regime / shape targets from the FIT-half grid instead of pre-TEST flow_3s days, trade statistics from the order stream); v0 = flow_sim.FlowSim.fit on the FIT-half grid; 10 paths each of the TEST ok-candle count; class-day = (day, class) with >= 2 symbols; count = facts inside (kurtosis, |r| ACF, flow ACF, impact, spread median AND p90); supports = v1 > v0 on >= 70% of class-days and majors v1 >= 4/5 on >= 50% of days; p = max of the binomial p's vs 0.7 and 0.5
- **holm**: Holm over the six MD summary p's (only at the final batch); supports needs the criterion and Holm p < 0.05
- **recording**: early record only for a share criterion already lost given the remaining planned days ('contradicts'); everything else after the last batch
- **pipeline source**: public.bybit.com/trading (every linear print; timestamp s -> ms = round(ts * 1000) as at-bybit archive.rs; stable sort by ms keeps the archive order within a millisecond) and quote-saver.bycsi.com ob200 (snapshot + delta JSON lines, sim.book)
- **pipeline orders**: order = maximal run of consecutive prints with the same ms and taker side (pilot C1 definition)
- **pipeline laws**: research.microstructure_laws_study blocks (rates, durations event/print, ACD, size print/order, signs, volume) with the module's day constants set to this day (FIT 00-12, TEST 12-24 UTC; no CPI exclusion window: test_ex == test); CSN bootstrap B = 50 on TEST prints only (pilot 200 / 50); no AFT
- **pipeline activity**: research.population_simulator_study._half_measures on the half's prints (W = sigma_day x P V_day, orders/s, Kyle lambda of the trade mid proxy, 1-min sigma, ...)
- **pipeline book_1s**: sim.book.build_series(bin_ms = 1000): state at the end of each 1 s bin of the push ts; pilot: last 100 ms state of each second from book_100ms (identical by construction), n_updates summed
- **pipeline hy**: book_spillover_study.hy_blocks / lead_stats on per-ms VWAP log prices, lags 0, +-0.1..60 s, per half, no bootstrap
- **pipeline dy**: VAR (BIC lag <= 4) on log 1 m RV (12 x 5 s last-trade returns from the 1 s bars), generalized FEVD H = 10 (book_spillover_study.dy_table); fixed pilot token set, a symbol is dropped on a day if missing or its nonzero-RV minute coverage < 0.98; full day (primary, 30-minute block bootstrap B = 200) and halves
- **pipeline spread_vol**: per L2 symbol-day as book_spillover_study.ms5_per_symbol: y = log RV over (T, T+h], h = 5, 15 min (1 s mid returns), x = log TWA spread over minute m (1 s states), controls log RV last 5 / 60 min, log depth10, intercept (no hour-of-week factor); OLS on FIT rows (T >= 01:00, T + h <= 12:00), scored on TEST rows; gain = OOS R^2 full - restricted; qualifying (H-MD4) = share of FIT 1 s states with spread > 1.5 ticks >= 0.25 (tick = minimum price increment seen in the day's prints)
