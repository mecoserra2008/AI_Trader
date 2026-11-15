# Multi-day microstructure, batch 3: H-MD1..H-MD6 interim (overnight program 2026-11-12)

_data: data/research/micro_days (16 verified days: 2025-08-31, 2025-09-05, 2025-11-19, 2025-12-18, 2026-05-11, 2026-06-21, 2026-07-03, 2026-07-12, 2026-07-13, 2026-07-22, 2026-08-16, 2026-08-17, 2026-09-23, 2026-10-02, 2026-10-25, 2026-11-10) | segments: llm_in_sample ['2025-08-31', '2025-09-05', '2025-11-19', '2025-12-18', '2026-05-11', '2026-06-21'], clean_holdout ['2026-07-03', '2026-07-12', '2026-07-13', '2026-07-22', '2026-08-16', '2026-08-17', '2026-09-23', '2026-10-02', '2026-10-25', '2026-11-10'] | code: src/ai_trader/research/micro_days.py, src/ai_trader/research/multiday_micro_study.py, scripts/micro_batch.py | tests: tests/test_micro_days.py_

## Changes since batch 2 (16 days; read this first)

- **Processing.**
  - The job was killed by a machine sleep (~08:02-11:05 UTC) partway through 2026-07-03. 2026-05-11 and 2026-06-21 had already been verified and their raw files deleted before the interruption, and they were not touched again.
  - The rerun at 11:10 UTC used the new resume mode (micro_days.resumed_task): it reused 16 finished symbol-days of 2026-07-03 and the cached, atomically downloaded archives, then finished 07-03, 08-16 and 10-02. All 5 days are verified and data/raw/batch_3 is deleted.
  - The 'batch days' line under 'Processing and storage' lists only the 3 days of the resumed run. The full log is logs/micro_batch_3.log.
  - Storage: lake plus research is 3.07 GB; micro_days is 1.79 GB.
- **New universe.** 2026-05-11 is missing GRAM, BZ and CRWV (not yet listed) and 2026-06-21 is missing GRAM. NVDA, QQQ and XAU books exist on both days.
- **H-MD6 recorded: contradicts (clean_holdout and pooled).** Majors reach 4/5 facts inside the envelope on 0/10 holdout days and 0/16 days pooled. Even if every remaining planned day passed, the share would be at most 9/19 = 0.47 and 15/31 = 0.48, below the registered 50%. v1 still beats v0 on 28/44 class-days (0.64). llm_in_sample is not locked (0/6 majors with 6 days left -> at most 0.5), so it waits.
- **H-MD2.** The in-sample segment no longer meets its criterion (the interaction Wald test now rejects there too). Pooled Wald p is 0.0003 for rate and 0.005 for impact; the rate LOO R^2 is 0.93.
- **H-MD3.** Pooled share is 502/637 = 0.788 (holdout 0.80, in-sample 0.766). Hurst is 0.541 [0.534, 0.549].
- **H-MD4.** Gain > 0 on 16/32 qualifying symbol-days at both horizons (in-sample 1/9 at 5 min). It is close to a coin flip, but not yet locked.
- **H-MD5.** ETH and SOL are net transmitters on 15/15 days.
- **H-MD1.** ICCs are 0.23-0.31; none passes.

## Status of each hypothesis

Interim numbers only: a verdict is recorded when a share criterion is already lost for good given the remaining pre-drawn days, or after the last batch (with Holm over the six). Not final.

| id | segment | criterion so far | summary p | planned days left | recorded |
|---|---|---|---|---|---|
| H-MD1 | llm_in_sample | not met | 1.000 | 6 | not recorded |
| H-MD1 | clean_holdout | not met | 1.000 | 9 | not recorded |
| H-MD1 | pooled | not met | 1.000 | 15 | not recorded |
| H-MD2 | llm_in_sample | not met | 0.053 | 6 | not recorded |
| H-MD2 | clean_holdout | not met | 0.137 | 9 | not recorded |
| H-MD2 | pooled | not met | 0.060 | 15 | not recorded |
| H-MD3 | llm_in_sample | not met | 0.911 | 6 | not recorded |
| H-MD3 | clean_holdout | met | 0.529 | 9 | not recorded |
| H-MD3 | pooled | not met | 0.790 | 15 | not recorded |
| H-MD4 | llm_in_sample | not met | 1.000 | 6 | not recorded |
| H-MD4 | clean_holdout | not met | 0.945 | 9 | not recorded |
| H-MD4 | pooled | not met | 0.995 | 15 | not recorded |
| H-MD5 | llm_in_sample | met | 0.262 | 6 | not recorded |
| H-MD5 | clean_holdout | met | 0.134 | 9 | not recorded |
| H-MD5 | pooled | met | 0.035 | 15 | not recorded |
| H-MD6 | llm_in_sample | not met | 1.000 | 6 | not recorded |
| H-MD6 | clean_holdout | not met | 1.000 | 9 | contradicts |
| H-MD6 | pooled | not met | 1.000 | 15 | contradicts |

Pooled summary: MD1 ICC gamma 0.31, acf1 0.31, zeta 0.23, mix_weights 0.24; MD2 LOO R2 rate 0.93 impact 0.62 Wald p 0.000/0.005; MD3 502/637 power-law, Hurst 0.541 [0.534, 0.549]; MD4 qualifying 32, gain>0 16/32 (5m) 16/32 (15m); MD5 net>0 ETH 15/15 SOL 15/15; MD6 v1>v0 28/44 class-days, majors>=4/5 0/16

## Processing and storage

- **batch days**: 2026-08-16 (verified True, raw deleted True), 2026-07-03 (verified True, raw deleted True), 2026-10-02 (verified True, raw deleted True)
- **storage before (GB)**: {"lake_gb": 1.183, "research_gb": 1.644, "micro_days_gb": 1.559, "raw_gb": 3.119, "lake_plus_research_gb": 2.827, "trades_raw_gb": 0.432}
- **storage after (GB)**: {"lake_gb": 1.195, "research_gb": 1.878, "micro_days_gb": 1.793, "raw_gb": 0.399, "lake_plus_research_gb": 3.073, "trades_raw_gb": 0.432}

## H-MD1: persistence of trade-level traits (ICC, symbol as group)

| segment | trait (TEST half) | ICC | CI lo | CI hi | symbols | symbol-days | mean | sd between | sd within |
|---|---|---|---|---|---|---|---|---|---|
| llm_in_sample | gamma | 0.242 | 0.099 | 0.368 | 46 | 243 | 0.776 | 0.191 | 0.255 |
| llm_in_sample | acf1 | 0.346 | 0.207 | 0.470 | 48 | 247 | 0.413 | 0.071 | 0.086 |
| llm_in_sample | zeta | 0.154 | 0.036 | 0.301 | 48 | 247 | 1.819 | 0.322 | 0.517 |
| llm_in_sample | w1 | 0.288 | 0.080 | 0.447 | 48 | 247 | 0.289 | 0.059 | 0.074 |
| llm_in_sample | w2 | 0.006 | -0.077 | 0.091 | 48 | 247 | 0.311 | 0.054 | 0.112 |
| llm_in_sample | w3 | 0.049 | -0.073 | 0.158 | 48 | 247 | 0.400 | 0.063 | 0.133 |
| llm_in_sample | mix_weights | 0.114 | 0.026 | 0.184 | n/a | n/a | n/a | n/a | n/a |
| clean_holdout | gamma | 0.352 | 0.254 | 0.433 | 51 | 506 | 0.715 | 0.178 | 0.224 |
| clean_holdout | acf1 | 0.383 | 0.221 | 0.526 | 51 | 510 | 0.381 | 0.067 | 0.079 |
| clean_holdout | zeta | 0.300 | 0.153 | 0.421 | 51 | 510 | 1.618 | 0.319 | 0.443 |
| clean_holdout | w1 | 0.418 | 0.161 | 0.581 | 51 | 510 | 0.276 | 0.063 | 0.070 |
| clean_holdout | w2 | 0.273 | 0.140 | 0.384 | 51 | 510 | 0.292 | 0.063 | 0.093 |
| clean_holdout | w3 | 0.250 | 0.128 | 0.354 | 51 | 510 | 0.432 | 0.073 | 0.112 |
| clean_holdout | mix_weights | 0.314 | 0.206 | 0.389 | n/a | n/a | n/a | n/a | n/a |
| pooled | gamma | 0.314 | 0.232 | 0.388 | 51 | 753 | 0.734 | 0.172 | 0.235 |
| pooled | acf1 | 0.309 | 0.185 | 0.421 | 51 | 759 | 0.392 | 0.060 | 0.087 |
| pooled | zeta | 0.229 | 0.120 | 0.353 | 51 | 759 | 1.683 | 0.291 | 0.473 |
| pooled | w1 | 0.353 | 0.120 | 0.503 | 51 | 759 | 0.280 | 0.056 | 0.075 |
| pooled | w2 | 0.185 | 0.092 | 0.290 | 51 | 759 | 0.299 | 0.056 | 0.102 |
| pooled | w3 | 0.177 | 0.073 | 0.273 | 51 | 759 | 0.421 | 0.063 | 0.121 |
| pooled | mix_weights | 0.238 | 0.144 | 0.307 | n/a | n/a | n/a | n/a | n/a |

- robustness H-MD1_fit_half: gamma 0.273 [0.181, 0.349], acf1 0.306 [0.188, 0.394], zeta 0.173 [0.098, 0.246], mix_weights 0.208 [0.126, 0.269]
- robustness H-MD1_ge20k: gamma 0.344 [0.241, 0.443], acf1 0.362 [0.215, 0.487], zeta 0.182 [0.068, 0.316], mix_weights 0.285 [0.185, 0.351]

## H-MD2: activity scaling with W (FIT half, day fixed effects)

| segment | y | slope on log W | SE (cl. symbol) | LOO R^2 | LOO CI | Wald p (day x log W) | rows | symbols | days |
|---|---|---|---|---|---|---|---|---|---|
| llm_in_sample | rate | 0.520 | 0.015 | 0.924 | [0.844, 0.948] | 0.003 | 249 | 50 | 6 |
| llm_in_sample | impact | -0.495 | 0.100 | 0.652 | [0.471, 0.801] | 0.887 | 249 | 50 | 6 |
| llm_in_sample | depth | 0.706 | 0.106 | 0.800 | [0.423, 0.967] | 0.008 | 48 | 10 | 6 |
| clean_holdout | rate | 0.530 | 0.016 | 0.921 | [0.818, 0.944] | 8.07e-04 | 510 | 51 | 10 |
| clean_holdout | impact | -0.496 | 0.082 | 0.603 | [0.418, 0.733] | 0.051 | 507 | 51 | 10 |
| clean_holdout | depth | 0.737 | 0.172 | 0.580 | [-0.104, 0.938] | 1.000 | 90 | 9 | 10 |
| pooled | rate | 0.527 | 0.014 | 0.925 | [0.886, 0.945] | 2.35e-05 | 759 | 51 | 16 |
| pooled | impact | -0.505 | 0.081 | 0.619 | [0.457, 0.766] | 0.005 | 756 | 51 | 16 |
| pooled | depth | 0.720 | 0.139 | 0.697 | [0.098, 0.919] | 1.000 | 138 | 10 | 16 |

Per-day slopes (pilot 2026-11-10 one-day values: rate ~W^0.54, depth ~W^0.68, impact ~W^-0.75):

| day | rate slope | impact slope | depth slope | rate SE | impact SE |
|---|---|---|---|---|---|
| 2025-08-31 | 0.516 | -0.487 | n/a | 0.021 | 0.177 |
| 2025-09-05 | 0.529 | -0.503 | n/a | 0.026 | 0.179 |
| 2025-11-19 | 0.525 | -0.556 | n/a | 0.042 | 0.089 |
| 2025-12-18 | 0.546 | -0.521 | n/a | 0.033 | 0.184 |
| 2026-05-11 | 0.488 | -0.513 | 0.758 | 0.031 | 0.092 |
| 2026-06-21 | 0.530 | -0.455 | 0.687 | 0.021 | 0.111 |
| 2026-07-03 | 0.528 | -0.491 | 0.787 | 0.027 | 0.126 |
| 2026-07-12 | 0.522 | -0.599 | 1.023 | 0.027 | 0.108 |
| 2026-07-13 | 0.524 | -0.651 | 0.802 | 0.022 | 0.084 |
| 2026-07-22 | 0.522 | -0.551 | 0.798 | 0.020 | 0.082 |
| 2026-08-16 | 0.479 | -0.526 | 0.560 | 0.044 | 0.143 |
| 2026-08-17 | 0.522 | -0.477 | 0.880 | 0.028 | 0.102 |
| 2026-09-23 | 0.582 | -0.399 | 1.350 | 0.032 | 0.143 |
| 2026-10-02 | 0.563 | -0.447 | 0.750 | 0.026 | 0.084 |
| 2026-10-25 | 0.525 | -0.486 | 1.130 | 0.028 | 0.129 |
| 2026-11-10 | 0.540 | -0.487 | 0.759 | 0.032 | 0.086 |

## H-MD3: order-sign long memory

| segment | power law & 0<gamma<1 | share | mean Hurst 1 s signed flow | CI lo (days) | CI hi | days |
|---|---|---|---|---|---|---|
| llm_in_sample | 170/222 | 0.766 | 0.546 | 0.537 | 0.556 | 6 |
| clean_holdout | 332/415 | 0.800 | 0.539 | 0.530 | 0.550 | 10 |
| pooled | 502/637 | 0.788 | 0.541 | 0.534 | 0.549 | 16 |

| day | k/n | share | mean Hurst |
|---|---|---|---|
| 2025-08-31 | 25/34 | 0.735 | 0.536 |
| 2025-09-05 | 31/34 | 0.912 | 0.534 |
| 2025-11-19 | 27/37 | 0.730 | 0.562 |
| 2025-12-18 | 36/38 | 0.947 | 0.559 |
| 2026-05-11 | 24/41 | 0.585 | 0.533 |
| 2026-06-21 | 27/38 | 0.711 | 0.550 |
| 2026-07-03 | 30/41 | 0.732 | 0.536 |
| 2026-07-12 | 29/38 | 0.763 | 0.530 |
| 2026-07-13 | 34/41 | 0.829 | 0.517 |
| 2026-07-22 | 33/43 | 0.767 | 0.544 |
| 2026-08-16 | 27/31 | 0.871 | 0.527 |
| 2026-08-17 | 28/37 | 0.757 | 0.531 |
| 2026-09-23 | 36/48 | 0.750 | 0.575 |
| 2026-10-02 | 44/48 | 0.917 | 0.550 |
| 2026-10-25 | 27/40 | 0.675 | 0.536 |
| 2026-11-10 | 44/48 | 0.917 | 0.531 |

## H-MD4: spread -> next 5 / 15 min volatility on free-spread symbol-days

| segment | h (min) | qualifying | gain > 0 | share | median gain | sign-test p | non-qualifying share > 0 | non-qual. median gain |
|---|---|---|---|---|---|---|---|---|
| llm_in_sample | 5 | 9 | 1/9 | 0.111 | -0.006 | 0.998 | 0.412 | -8.42e-04 |
| llm_in_sample | 15 | 9 | 3/9 | 0.333 | -0.003 | 0.910 | 0.510 | 1.57e-04 |
| clean_holdout | 5 | 23 | 15/23 | 0.652 | 0.006 | 0.105 | 0.392 | -0.001 |
| clean_holdout | 15 | 23 | 13/23 | 0.565 | 0.003 | 0.339 | 0.340 | -0.004 |
| pooled | 5 | 32 | 16/32 | 0.500 | 3.74e-04 | 0.570 | 0.399 | -0.001 |
| pooled | 15 | 32 | 16/32 | 0.500 | 5.84e-04 | 0.570 | 0.399 | -0.002 |

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
| 2026-05-11 | XRPUSDT | 1.16e-04 | no | -0.011 | -0.006 | 1.436 |
| 2026-05-11 | HYPEUSDT | 0.048 | no | -0.019 | -0.020 | -0.747 |
| 2026-05-11 | ETHUSDT | 9.72e-04 | no | -0.001 | -0.005 | 2.370 |
| 2026-05-11 | SOLUSDT | 9.26e-05 | no | -0.457 | -1.005 | 4.042 |
| 2026-05-11 | DOGEUSDT | 2.55e-04 | no | -0.010 | -0.007 | 3.198 |
| 2026-05-11 | BTCUSDT | 0.001 | no | -1.68e-04 | -7.62e-04 | 0.104 |
| 2026-05-11 | ATOMUSDT | 0.086 | no | -0.029 | 0.012 | -0.800 |
| 2026-05-11 | XAUUSDT | 0.002 | no | -0.003 | 5.39e-04 | 1.211 |
| 2026-05-11 | KASUSDT | 0.377 | yes | -0.023 | -0.046 | 4.179 |
| 2026-05-11 | NVDAUSDT | 0.634 | yes | -3.16e-04 | 0.015 | -0.488 |
| 2026-05-11 | CARVUSDT | 0.970 | yes | -0.006 | -0.003 | 0.855 |
| 2026-05-11 | QQQUSDT | 0.935 | yes | -0.079 | 0.022 | -1.198 |
| 2026-06-21 | HYPEUSDT | 0.074 | no | -4.32e-04 | -0.003 | 0.146 |
| 2026-06-21 | ETHUSDT | 3.01e-04 | no | -3.91e-04 | 0.001 | 0.698 |
| 2026-06-21 | BTCUSDT | 3.70e-04 | no | 0.001 | 8.01e-04 | 0.714 |
| 2026-06-21 | XRPUSDT | 4.63e-05 | no | -0.433 | -0.766 | 0.913 |
| 2026-06-21 | SOLUSDT | 0.000 | no | -0.445 | -0.588 | 3.153 |
| 2026-06-21 | DOGEUSDT | 4.63e-05 | no | 0.042 | 0.095 | 5.422 |
| 2026-06-21 | ATOMUSDT | 0.140 | no | 0.012 | 1.57e-04 | -1.474 |
| 2026-06-21 | XAUUSDT | 2.31e-05 | no | -0.011 | -0.001 | 0.573 |
| 2026-06-21 | KASUSDT | 0.141 | no | -0.004 | -0.001 | 1.021 |
| 2026-06-21 | QQQUSDT | 0.698 | yes | -8.77e-04 | -0.003 | 1.140 |
| 2026-06-21 | CARVUSDT | 0.697 | yes | 0.015 | 0.002 | 1.533 |
| 2026-06-21 | NVDAUSDT | 0.018 | no | 0.005 | 0.034 | 1.284 |
| 2026-07-03 | ETHUSDT | 5.56e-04 | no | 0.002 | -0.003 | -0.033 |
| 2026-07-03 | BTCUSDT | 6.94e-04 | no | 7.40e-04 | -2.71e-04 | 3.631 |
| 2026-07-03 | HYPEUSDT | 0.101 | no | -5.69e-04 | -0.007 | -0.629 |
| 2026-07-03 | XRPUSDT | 4.63e-05 | no | 0.058 | 0.155 | -1.258 |
| 2026-07-03 | SOLUSDT | 4.63e-05 | no | -0.046 | -0.122 | 1.739 |
| 2026-07-03 | DOGEUSDT | 9.26e-05 | no | 0.032 | -0.030 | -0.398 |
| 2026-07-03 | XAUUSDT | 3.94e-04 | no | 8.61e-04 | 0.003 | 0.258 |
| 2026-07-03 | ATOMUSDT | 0.099 | no | -0.023 | -0.013 | 1.386 |
| 2026-07-03 | NVDAUSDT | 0.017 | no | -0.001 | -0.004 | 1.796 |
| 2026-07-03 | QQQUSDT | 0.425 | yes | 0.013 | -0.012 | 3.767 |
| 2026-07-03 | KASUSDT | 0.272 | yes | 0.001 | -0.014 | -0.492 |
| 2026-07-03 | CARVUSDT | 0.633 | yes | 0.006 | 0.015 | -0.285 |
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
| 2026-08-16 | DOGEUSDT | 0.000 | no | -0.468 | -0.531 | -0.143 |
| 2026-08-16 | XRPUSDT | 2.31e-05 | no | 0.024 | 0.051 | -0.203 |
| 2026-08-16 | ETHUSDT | 6.94e-05 | no | 9.98e-04 | -0.001 | -0.070 |
| 2026-08-16 | BTCUSDT | 4.63e-05 | no | 6.69e-04 | 0.002 | -0.150 |
| 2026-08-16 | HYPEUSDT | 9.26e-05 | no | 0.007 | 0.023 | -1.827 |
| 2026-08-16 | SOLUSDT | 0.000 | no | 0.020 | 0.245 | -1.372 |
| 2026-08-16 | XAUUSDT | 0.000 | no | -33.812 | -59.909 | -0.114 |
| 2026-08-16 | ATOMUSDT | 0.016 | no | 2.32e-05 | -0.001 | 0.176 |
| 2026-08-16 | KASUSDT | 0.123 | no | 0.014 | 0.013 | 1.604 |
| 2026-08-16 | NVDAUSDT | 0.006 | no | -0.010 | -0.179 | 0.994 |
| 2026-08-16 | CARVUSDT | 0.811 | yes | 0.011 | 0.002 | 1.478 |
| 2026-08-16 | QQQUSDT | 0.016 | no | -0.021 | 0.008 | 3.441 |
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
| 2026-10-02 | XRPUSDT | 4.63e-04 | no | -0.009 | -0.026 | -2.060 |
| 2026-10-02 | ETHUSDT | 9.26e-04 | no | -0.016 | -0.004 | -3.171 |
| 2026-10-02 | DOGEUSDT | 2.78e-04 | no | 0.005 | -0.015 | -1.516 |
| 2026-10-02 | BTCUSDT | 0.001 | no | -0.008 | -0.011 | -0.109 |
| 2026-10-02 | SOLUSDT | 2.31e-04 | no | -0.001 | -0.041 | -1.954 |
| 2026-10-02 | HYPEUSDT | 5.32e-04 | no | -0.001 | -0.006 | -2.471 |
| 2026-10-02 | XAUUSDT | 0.002 | no | -0.004 | -0.004 | -1.533 |
| 2026-10-02 | QQQUSDT | 0.118 | no | -0.002 | -0.006 | 1.638 |
| 2026-10-02 | NVDAUSDT | 0.218 | no | 0.004 | 0.005 | 3.079 |
| 2026-10-02 | ATOMUSDT | 0.054 | no | -0.004 | -0.015 | -2.028 |
| 2026-10-02 | KASUSDT | 0.684 | yes | -0.028 | -0.010 | -0.292 |
| 2026-10-02 | CARVUSDT | 0.985 | yes | 0.011 | 0.006 | -2.517 |
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
| 2026-05-11 | 29.951 | 12.398 | 19.814 | 13.607 | [21.9, 38.1] | [7.9, 22.0] | 74.536 | 22.849 | 11.620 | 15 |
| 2026-06-21 | 26.378 | 11.880 | 13.256 | -5.435 | [17.7, 34.4] | [6.0, 18.1] | 69.236 | 35.038 | 10.928 | 13 |
| 2026-07-03 | 38.671 | 13.381 | 19.696 | 21.044 | [30.7, 48.0] | [7.6, 18.7] | 63.188 | 37.322 | 13.428 | 14 |
| 2026-07-12 | 25.631 | 16.551 | 26.341 | -6.392 | [18.3, 33.8] | [8.5, 23.7] | 59.113 | 23.870 | 13.897 | 12 |
| 2026-07-13 | 37.224 | 24.278 | 12.386 | 10.989 | [28.2, 44.8] | [15.3, 31.8] | 73.136 | 23.242 | 21.449 | 13 |
| 2026-07-22 | 23.619 | 18.455 | 15.047 | 9.891 | [15.1, 31.3] | [9.2, 25.8] | 59.341 | 28.480 | 19.697 | 11 |
| 2026-08-17 | 8.476 | 0.884 | n/a | 1.475 | [3.0, 14.8] | [-4.9, 5.5] | 40.699 | 16.925 | -0.017 | 6 |
| 2026-09-23 | 39.182 | 35.127 | 27.555 | 18.035 | [31.3, 46.8] | [27.0, 42.8] | 72.014 | 28.162 | 27.094 | 15 |
| 2026-10-02 | 29.675 | 32.968 | 12.325 | -1.995 | [15.9, 41.8] | [20.8, 42.7] | 78.116 | 18.505 | 28.867 | 15 |
| 2026-10-25 | 40.147 | 25.042 | 10.151 | -22.161 | [30.4, 46.8] | [17.9, 31.1] | 76.613 | 41.600 | 22.476 | 15 |
| 2026-11-10 | 26.820 | 28.461 | 21.195 | 7.852 | [17.8, 32.0] | [19.5, 34.8] | 83.947 | 25.715 | 22.694 | 15 |

| segment | ETH net > 0 | SOL net > 0 | DOGE net > 0 | BTC net > 0 | mean BTC - ETH | CI (days) |
|---|---|---|---|---|---|---|
| llm_in_sample | 6/6 | 6/6 | 6/6 | 3/6 | -24.127 | [-33.6, -16.8] |
| clean_holdout | 9/9 | 9/9 | 8/8 | 6/9 | -25.634 | [-36.0, -17.1] |
| pooled | 15/15 | 15/15 | 14/14 | 9/15 | -25.031 | [-32.3, -18.7] |

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
| 2026-05-11 | major | 5 | 2 | 1 | flow_acf, impact | impact |
| 2026-05-11 | alt | 4 | 1 | 1 | flow_acf | flow_acf |
| 2026-05-11 | tradfi | 3 | 3 | 1 | kurtosis, flow_acf, impact | impact |
| 2026-06-21 | major | 5 | 2 | 1 | impact, spread | spread |
| 2026-06-21 | alt | 4 | 2 | 0 | flow_acf, impact | - |
| 2026-06-21 | tradfi | 3 | 2 | 2 | kurtosis, spread | flow_acf, impact |
| 2026-07-03 | major | 5 | 2 | 0 | flow_acf, impact | - |
| 2026-07-03 | alt | 4 | 1 | 0 | flow_acf | - |
| 2026-07-03 | tradfi | 3 | 2 | 0 | flow_acf, impact | - |
| 2026-07-12 | major | 5 | 1 | 1 | flow_acf | spread |
| 2026-07-12 | alt | 4 | 2 | 1 | flow_acf, impact | impact |
| 2026-07-12 | tradfi | 3 | 2 | 2 | kurtosis, spread | flow_acf, impact |
| 2026-07-13 | major | 5 | 2 | 0 | flow_acf, impact | - |
| 2026-07-13 | alt | 4 | 1 | 1 | flow_acf | flow_acf |
| 2026-07-13 | tradfi | 3 | 3 | 2 | kurtosis, flow_acf, impact | kurtosis, flow_acf |
| 2026-07-22 | major | 5 | 2 | 1 | flow_acf, spread | spread |
| 2026-07-22 | alt | 4 | 3 | 1 | kurtosis, flow_acf, impact | flow_acf |
| 2026-07-22 | tradfi | 3 | 3 | 1 | kurtosis, flow_acf, impact | kurtosis |
| 2026-08-16 | major | 5 | 3 | 2 | flow_acf, impact, spread | kurtosis, spread |
| 2026-08-16 | alt | 3 | 1 | 1 | flow_acf | kurtosis |
| 2026-08-16 | tradfi | 3 | 1 | 1 | spread | flow_acf |
| 2026-08-17 | major | 5 | 3 | 1 | flow_acf, impact, spread | spread |
| 2026-08-17 | alt | 4 | 0 | 1 | - | flow_acf |
| 2026-08-17 | tradfi | 3 | 2 | 0 | kurtosis, flow_acf | - |
| 2026-09-23 | major | 5 | 0 | 2 | - | kurtosis, impact |
| 2026-09-23 | alt | 4 | 3 | 2 | kurtosis, flow_acf, impact | kurtosis, impact |
| 2026-09-23 | tradfi | 3 | 2 | 1 | kurtosis, flow_acf | flow_acf |
| 2026-10-02 | major | 5 | 1 | 2 | impact | kurtosis, impact |
| 2026-10-02 | alt | 4 | 3 | 2 | kurtosis, flow_acf, impact | flow_acf, impact |
| 2026-10-02 | tradfi | 3 | 3 | 2 | kurtosis, flow_acf, impact | flow_acf, impact |
| 2026-10-25 | major | 5 | 3 | 2 | flow_acf, impact, spread | impact, spread |
| 2026-10-25 | alt | 4 | 1 | 2 | kurtosis | flow_acf, impact |
| 2026-10-25 | tradfi | 3 | 3 | 2 | flow_acf, impact, spread | flow_acf, impact |
| 2026-11-10 | major | 5 | 3 | 2 | kurtosis, flow_acf, impact | kurtosis, impact |
| 2026-11-10 | alt | 4 | 2 | 2 | kurtosis, flow_acf | flow_acf, impact |
| 2026-11-10 | tradfi | 3 | 5 | 0 | kurtosis, absr_acf, flow_acf, impact, spread | - |

| segment | v1 > v0 class-days | share | majors v1 >= 4/5 | mean v1 - v0 |
|---|---|---|---|---|
| llm_in_sample | 8/14 | 0.571 | 0/6 | 0.714 |
| clean_holdout | 20/30 | 0.667 | 0/10 | 0.867 |
| pooled | 28/44 | 0.636 | 0/16 | 0.818 |

Share of class-days with the fact inside the envelope: kurtosis v1 0.41 / v0 0.18, absr_acf v1 0.02 / v0 0.00, flow_acf v1 0.73 / v0 0.34, impact v1 0.59 / v0 0.48, spread v1 0.23 / v0 0.16

Symbol-days without a usable v1/v0 run: [('2026-08-16', 'CARVUSDT')]

## Day profiles and differences from the 2026-11-10 pilot

| day | segment | symbols | major events/s TEST | major TEST/FIT events | BTC sigma_day (bp) | BTC notional (USD, FIT x2) | median gamma | median sign ACF1 | median zeta (prints) | median Hurst | major prints/order | median burst weight | DY total |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2025-08-31 | llm_in_sample | 37 | 1.953 | 0.964 | 104.397 | 4.05e+09 | 0.841 | 0.403 | 2.000 | 0.528 | 4.020 | 0.266 | 69.874 |
| 2025-09-05 | llm_in_sample | 37 | 4.949 | 1.822 | 143.915 | 7.34e+09 | 0.805 | 0.403 | 2.110 | 0.529 | 3.764 | 0.375 | 81.257 |
| 2025-11-19 | llm_in_sample | 38 | 5.551 | 1.834 | 247.433 | 8.10e+09 | 0.677 | 0.414 | 1.791 | 0.552 | 4.401 | 0.266 | 83.010 |
| 2025-12-18 | llm_in_sample | 39 | 7.009 | 3.670 | 177.934 | 3.80e+09 | 0.624 | 0.390 | 1.685 | 0.555 | 3.812 | 0.297 | 81.522 |
| 2026-05-11 | llm_in_sample | 48 | 2.488 | 1.154 | 163.918 | 3.70e+09 | 0.854 | 0.428 | 1.660 | 0.533 | 4.360 | 0.253 | 74.536 |
| 2026-06-21 | llm_in_sample | 50 | 1.494 | 1.514 | 117.507 | 1.92e+09 | 0.718 | 0.414 | 1.448 | 0.545 | 4.121 | 0.272 | 69.236 |
| 2026-07-03 | clean_holdout | 51 | 1.507 | 1.210 | 170.706 | 2.31e+09 | 0.784 | 0.373 | 1.653 | 0.531 | 4.426 | 0.261 | 63.188 |
| 2026-07-12 | clean_holdout | 51 | 0.743 | 0.849 | 114.011 | 2.26e+09 | 0.668 | 0.396 | 1.734 | 0.517 | 4.124 | 0.277 | 59.113 |
| 2026-07-13 | clean_holdout | 51 | 1.244 | 0.966 | 212.754 | 5.26e+09 | 0.748 | 0.343 | 1.729 | 0.520 | 4.350 | 0.283 | 73.136 |
| 2026-07-22 | clean_holdout | 51 | 0.980 | 1.387 | 142.111 | 2.64e+09 | 0.668 | 0.395 | 1.676 | 0.529 | 4.572 | 0.286 | 59.341 |
| 2026-08-16 | clean_holdout | 51 | 0.464 | 1.726 | 37.159 | 5.34e+08 | 0.612 | 0.347 | 1.404 | 0.523 | 2.962 | 0.246 | n/a |
| 2026-08-17 | clean_holdout | 51 | 0.536 | 0.926 | 125.470 | 3.43e+09 | 0.627 | 0.357 | 1.595 | 0.528 | 3.885 | 0.217 | 40.699 |
| 2026-09-23 | clean_holdout | 51 | 4.342 | 1.564 | 164.161 | 5.35e+09 | 0.683 | 0.391 | 1.432 | 0.580 | 4.939 | 0.320 | 72.014 |
| 2026-10-02 | clean_holdout | 51 | 2.661 | 1.325 | 202.426 | 5.97e+09 | 0.647 | 0.407 | 1.415 | 0.541 | 3.806 | 0.287 | 78.116 |
| 2026-10-25 | clean_holdout | 51 | 0.817 | 1.118 | 95.562 | 1.69e+09 | 0.793 | 0.420 | 1.436 | 0.524 | 3.438 | 0.230 | 76.613 |
| 2026-11-10 | clean_holdout | 51 | 2.509 | 2.610 | 177.012 | 2.81e+09 | 0.640 | 0.376 | 1.756 | 0.535 | 4.041 | 0.249 | 83.947 |

Hayashi-Yoshida lead-lag (TEST half, crypto targets): share of peaks at lag 0 and in (0.1, 30] s:

| day | ETH peak at 0 | SOL peak at 0 | BTC peak at 0 | ETH positive peak | SOL positive peak | BTC positive peak | ETH median rho(0) |
|---|---|---|---|---|---|---|---|
| 2025-08-31 | 0.722 | 0.667 | 0.194 | 0.194 | 0.333 | 0.111 | 0.507 |
| 2025-09-05 | 0.889 | 0.917 | 0.889 | 0.083 | 0.083 | 0.056 | 0.473 |
| 2025-11-19 | 0.919 | 0.946 | 0.405 | 0.054 | 0.054 | 0.000 | 0.552 |
| 2025-12-18 | 0.868 | 0.895 | 0.658 | 0.079 | 0.079 | 0.079 | 0.576 |
| 2026-05-11 | 0.737 | 0.737 | 0.632 | 0.105 | 0.263 | 0.105 | 0.415 |
| 2026-06-21 | 0.816 | 0.789 | 0.632 | 0.105 | 0.158 | 0.105 | 0.457 |
| 2026-07-03 | 0.821 | 0.769 | 0.718 | 0.103 | 0.179 | 0.103 | 0.438 |
| 2026-07-12 | 0.769 | 0.692 | 0.564 | 0.154 | 0.308 | 0.103 | 0.357 |
| 2026-07-13 | 0.821 | 0.718 | 0.718 | 0.103 | 0.231 | 0.103 | 0.442 |
| 2026-07-22 | 0.615 | 0.692 | 0.436 | 0.205 | 0.231 | 0.128 | 0.355 |
| 2026-08-16 | 0.333 | 0.282 | 0.179 | 0.179 | 0.231 | 0.205 | 0.235 |
| 2026-08-17 | 0.564 | 0.487 | 0.538 | 0.231 | 0.308 | 0.205 | 0.339 |
| 2026-09-23 | 0.923 | 0.949 | 0.769 | 0.026 | 0.026 | 0.051 | 0.313 |
| 2026-10-02 | 0.949 | 0.974 | 0.897 | 0.026 | 0.026 | 0.026 | 0.375 |
| 2026-10-25 | 0.821 | 0.846 | 0.564 | 0.077 | 0.128 | 0.077 | 0.611 |
| 2026-11-10 | 0.974 | 0.923 | 0.897 | 0.000 | 0.000 | 0.000 | 0.296 |

| day | free-spread L2 symbols (FIT share > 1 tick >= 25%) | missing trade archives (404) | missing ob200 (404) |
|---|---|---|---|
| 2025-08-31 | - | GRAMUSDT, MONUSDT, ASTERUSDT, CLUSDT, XAUUSDT, BZUSDT, XAGUSDT, QQQUSDT, NVDAUSDT, MSTRUSDT, TSLAUSDT, CRWVUSDT, COINUSDT, GOOGLUSDT | NVDAUSDT, XAUUSDT, QQQUSDT |
| 2025-09-05 | - | GRAMUSDT, MONUSDT, ASTERUSDT, CLUSDT, BZUSDT, XAUUSDT, XAGUSDT, NVDAUSDT, QQQUSDT, TSLAUSDT, MSTRUSDT, COINUSDT, CRWVUSDT, GOOGLUSDT | NVDAUSDT, QQQUSDT, XAUUSDT |
| 2025-11-19 | KASUSDT, CARVUSDT | GRAMUSDT, MONUSDT, CLUSDT, BZUSDT, XAUUSDT, XAGUSDT, QQQUSDT, NVDAUSDT, TSLAUSDT, COINUSDT, MSTRUSDT, CRWVUSDT, GOOGLUSDT | NVDAUSDT, QQQUSDT, XAUUSDT |
| 2025-12-18 | CARVUSDT | GRAMUSDT, CLUSDT, BZUSDT, XAUUSDT, XAGUSDT, QQQUSDT, NVDAUSDT, TSLAUSDT, MSTRUSDT, CRWVUSDT, COINUSDT, GOOGLUSDT | NVDAUSDT, QQQUSDT, XAUUSDT |
| 2026-05-11 | KASUSDT, NVDAUSDT, CARVUSDT, QQQUSDT | GRAMUSDT, BZUSDT, CRWVUSDT | - |
| 2026-06-21 | QQQUSDT, CARVUSDT | GRAMUSDT | - |
| 2026-07-03 | QQQUSDT, KASUSDT, CARVUSDT | - | - |
| 2026-07-12 | QQQUSDT, CARVUSDT | - | - |
| 2026-07-13 | ATOMUSDT, QQQUSDT, CARVUSDT | - | - |
| 2026-07-22 | QQQUSDT, CARVUSDT | - | - |
| 2026-08-16 | CARVUSDT | - | - |
| 2026-08-17 | HYPEUSDT, CARVUSDT | - | - |
| 2026-09-23 | HYPEUSDT, QQQUSDT, KASUSDT, CARVUSDT | - | - |
| 2026-10-02 | KASUSDT, CARVUSDT | - | - |
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
