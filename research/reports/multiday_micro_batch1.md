# Multi-day microstructure, batch 1: H-MD1..H-MD6 interim (overnight program 2026-11-12)

_data: data/research/micro_days (6 verified days: 2025-09-05, 2025-12-18, 2026-07-12, 2026-07-13, 2026-09-23, 2026-11-10) | segments: llm_in_sample ['2025-09-05', '2025-12-18'], clean_holdout ['2026-07-12', '2026-07-13', '2026-09-23', '2026-11-10'] | code: src/ai_trader/research/micro_days.py, src/ai_trader/research/multiday_micro_study.py, scripts/micro_batch.py | tests: tests/test_micro_days.py_

## What differs from the 2026-11-10 pilot (read this first)

- **Pipeline reproduces the pilot on its own day.** Rebuilt from aggregates, 2026-11-10 gives 44/48 power-law sign ACFs and a mean Hurst exponent of 0.531, as H-MS3 did. DY net on the TEST half is ETH +25.7, SOL +22.7 and BTC +10.6, against the pilot's +25.0, +21.6 and +10.2. The small DY gap comes from using the last trade per 5 s rather than the per-ms VWAP.
- **Regimes differ.**
  - 2026-11-10 was the CPI sell-off day: majors had 2.6x more arrival events in the TEST half than in the FIT half.
  - Both llm_in_sample days are also US macro-release days: 2025-09-05 (August payrolls, TEST/FIT 1.8x, majors 4.9 events/s in TEST) and 2025-12-18 (the delayed November CPI release, TEST/FIT 3.7x, 7.0 events/s). Both are busier than the pilot.
  - 2026-07-12 (Sunday) and 2026-07-13 (Monday) are quiet: majors at 0.7-1.2 events/s with TEST/FIT about 0.85-0.97. The DY total there is 59-73, against 81-84 on the macro days.
  - 2026-09-23 is busy (4.3 events/s) with the most free-spread L2 symbols: HYPE, QQQ, KAS and CARV.
  - So the day draw is not 'normal days vs a CPI day'. The in-sample segment is two macro days.
- **Universe changes with the date.** On 2025-09-05 and 2025-12-18, all 11 TradFi perps (and GRAM; MON and ASTER on 2025-09-05) have no archive (404 = not listed), and neither do the NVDA, QQQ and XAU books. H-MD6 has no TradFi class-day in llm_in_sample, and H-MD4 has only 1 qualifying symbol-day there.
- **H-MD1 (persistence).** Day-to-day variation is as large as between-symbol variation. Pooled ICCs: gamma 0.29, lag-1 sign ACF 0.30, size tail 0.17, mixture weights 0.19, so none passes. Only on the two in-sample days does the lag-1 sign ACF reach 0.62 [0.35, 0.78]. The pilot's per-symbol traits should not be treated as stable calibrations (interim).
- **H-MD2 (scaling).**
  - The rate exponent replicates almost exactly. The pooled slope is 0.54 (pilot 0.54), with per-day slopes of 0.52-0.58, and the LOO R^2 is 0.92.
  - Impact is flatter than in the pilot: -0.52 pooled (per day -0.40 to -0.65) against the pilot's -0.75, with LOO R^2 0.61.
  - The day x log W interaction is significant (Wald p = 0.003 for rate, 0.031 for impact, pooled). Even small slope differences, about 0.05, are detectable with about 50 symbols per day. This is the part of the criterion currently failing.
- **H-MD3 (sign memory).** Holds on every day: 75-95% of the >= 20k-print symbol-days, and per-day mean Hurst 0.517-0.580. The pooled CI is [0.527, 0.560]. The holdout share is 0.817, close to the 0.8 line: the two quiet July days and 2026-09-23 sit at 0.75-0.83.
- **H-MD4 (spread -> vol).** 12 qualifying symbol-days (11 in holdout), so the minimum of >= 10 is met only pooled. Gain > 0 on 8/12 at 5 min and 9/12 at 15 min. The 5-min share (0.67) is under 0.7, so the criterion is not met pooled. Non-qualifying (tick-bound) symbol-days show no gain (share 0.41-0.46), the same pattern as the pilot.
- **H-MD5 (DY).** ETH and SOL are net transmitters on 6/6 days, and DOGE too, on the full day. On the TEST half of 2025-12-18, SOL is -2.0. BTC is a smaller transmitter than ETH on every day (mean BTC - ETH = -22 [-27, -18]) and is a net receiver on 2/6 days. The pilot's ranking generalises so far.
- **H-MD6 (simulator).**
  - v1 beats v0 on 9/16 class-days, but majors never reach 4/5. The best major day is the pilot day (3/5, as in H-MS8), and 2026-09-23 gets 0/5.
  - v1 wins on flow ACF (inside on 81% of class-days vs 38%) and on kurtosis (50% vs 25%).
  - Both miss the |r| ACF (6% / 0%) and the spread (12% / 12%) almost always.
  - Here v0 is refitted on each day's FIT half. This is fairer than the pilot's v0, which was fitted on 2026-11-08/09.
- **HY lead-lag.** Cross-asset peaks sit at lag 0 on most targets on every day: 56-97% of crypto targets. 0-31% of targets show a positive peak (0.1-30 s), with the highest share on the quiet July days (SOL 31% on 2026-07-12). There is still no tradable sub-second lead from ETH, SOL or BTC.
- **Nothing recorded.** No H-MD share criterion is already lost given the 25 remaining planned days, and the ICC, regression and Holm parts need the full set. The verdict for every hypothesis is deferred.

## Status of each hypothesis

Interim numbers only: a verdict is recorded when a share criterion is already lost for good given the remaining pre-drawn days, or after the last batch (with Holm over the six). Not final.

| id | segment | criterion so far | summary p | planned days left | recorded |
|---|---|---|---|---|---|
| H-MD1 | llm_in_sample | not met | 0.999 | 10 | not recorded |
| H-MD1 | clean_holdout | not met | 1.000 | 15 | not recorded |
| H-MD1 | pooled | not met | 1.000 | 25 | not recorded |
| H-MD2 | llm_in_sample | met | 0.091 | 10 | not recorded |
| H-MD2 | clean_holdout | not met | 0.131 | 15 | not recorded |
| H-MD2 | pooled | not met | 0.097 | 25 | not recorded |
| H-MD3 | llm_in_sample | met | 0.002 | 10 | not recorded |
| H-MD3 | clean_holdout | met | 0.324 | 15 | not recorded |
| H-MD3 | pooled | met | 0.026 | 25 | not recorded |
| H-MD4 | llm_in_sample | not met | 1.000 | 10 | not recorded |
| H-MD4 | clean_holdout | met | 0.570 | 15 | not recorded |
| H-MD4 | pooled | not met | 0.724 | 25 | not recorded |
| H-MD5 | llm_in_sample | met | 0.640 | 10 | not recorded |
| H-MD5 | clean_holdout | met | 0.410 | 15 | not recorded |
| H-MD5 | pooled | met | 0.262 | 25 | not recorded |
| H-MD6 | llm_in_sample | not met | 1.000 | 10 | not recorded |
| H-MD6 | clean_holdout | not met | 1.000 | 15 | not recorded |
| H-MD6 | pooled | not met | 1.000 | 25 | not recorded |

Pooled summary: MD1 ICC gamma 0.29, acf1 0.30, zeta 0.16, mix_weights 0.19; MD2 LOO R2 rate 0.92 impact 0.61 Wald p 0.003/0.031; MD3 210/247 power-law, Hurst 0.542 [0.527, 0.560]; MD4 qualifying 12, gain>0 8/12 (5m) 9/12 (15m); MD5 net>0 ETH 6/6 SOL 6/6; MD6 v1>v0 9/16 class-days, majors>=4/5 0/6

## Processing and storage

- **batch days**: 2026-07-12 (verified True, raw deleted True), 2026-07-13 (verified True, raw deleted True), 2025-09-05 (verified True, raw deleted True), 2025-12-18 (verified True, raw deleted True), 2026-09-23 (verified True, raw deleted True)
- **storage before (GB)**: {"lake_gb": 1.129, "research_gb": 0.086, "micro_days_gb": 0.0, "raw_gb": 1.3, "lake_plus_research_gb": 1.215, "trades_raw_gb": 0.432}
- **storage after (GB)**: {"lake_gb": 1.136, "research_gb": 0.827, "micro_days_gb": 0.741, "raw_gb": 0.399, "lake_plus_research_gb": 1.963, "trades_raw_gb": 0.432}

## H-MD1: persistence of trade-level traits (ICC, symbol as group)

| segment | trait (TEST half) | ICC | CI lo | CI hi | symbols | symbol-days | mean | sd between | sd within |
|---|---|---|---|---|---|---|---|---|---|
| llm_in_sample | gamma | 0.490 | 0.230 | 0.663 | 37 | 74 | 0.713 | 0.201 | 0.168 |
| llm_in_sample | acf1 | 0.621 | 0.351 | 0.784 | 37 | 74 | 0.408 | 0.086 | 0.060 |
| llm_in_sample | zeta | 0.126 | -0.202 | 0.365 | 37 | 74 | 2.000 | 0.490 | 0.619 |
| llm_in_sample | w1 | 0.112 | -0.225 | 0.386 | 37 | 74 | 0.328 | 0.060 | 0.077 |
| llm_in_sample | w2 | -0.095 | -0.467 | 0.187 | 37 | 74 | 0.367 | 0.050 | 0.078 |
| llm_in_sample | w3 | 0.463 | 0.182 | 0.673 | 37 | 74 | 0.305 | 0.078 | 0.068 |
| llm_in_sample | mix_weights | 0.160 | -0.038 | 0.301 | n/a | n/a | n/a | n/a | n/a |
| clean_holdout | gamma | 0.360 | 0.212 | 0.491 | 51 | 203 | 0.732 | 0.190 | 0.214 |
| clean_holdout | acf1 | 0.394 | 0.185 | 0.555 | 51 | 204 | 0.377 | 0.072 | 0.077 |
| clean_holdout | zeta | 0.295 | 0.145 | 0.408 | 51 | 204 | 1.697 | 0.371 | 0.458 |
| clean_holdout | w1 | 0.466 | 0.124 | 0.630 | 51 | 204 | 0.287 | 0.068 | 0.065 |
| clean_holdout | w2 | 0.170 | 0.017 | 0.311 | 51 | 204 | 0.312 | 0.058 | 0.088 |
| clean_holdout | w3 | 0.176 | 0.001 | 0.329 | 51 | 204 | 0.400 | 0.072 | 0.108 |
| clean_holdout | mix_weights | 0.270 | 0.132 | 0.362 | n/a | n/a | n/a | n/a | n/a |
| pooled | gamma | 0.289 | 0.162 | 0.395 | 51 | 279 | 0.727 | 0.167 | 0.218 |
| pooled | acf1 | 0.300 | 0.156 | 0.417 | 51 | 280 | 0.386 | 0.063 | 0.082 |
| pooled | zeta | 0.165 | 0.058 | 0.289 | 51 | 280 | 1.777 | 0.340 | 0.532 |
| pooled | w1 | 0.334 | 0.088 | 0.511 | 51 | 280 | 0.298 | 0.058 | 0.072 |
| pooled | w2 | 0.103 | -0.013 | 0.215 | 51 | 280 | 0.327 | 0.050 | 0.090 |
| pooled | w3 | 0.125 | -0.025 | 0.262 | 51 | 280 | 0.375 | 0.062 | 0.113 |
| pooled | mix_weights | 0.188 | 0.077 | 0.270 | n/a | n/a | n/a | n/a | n/a |

- robustness H-MD1_fit_half: gamma 0.249 [0.136, 0.354], acf1 0.283 [0.146, 0.398], zeta 0.219 [0.113, 0.337], mix_weights 0.221 [0.129, 0.297]
- robustness H-MD1_ge20k: gamma 0.321 [0.173, 0.451], acf1 0.326 [0.167, 0.470], zeta 0.088 [-0.006, 0.207], mix_weights 0.186 [0.066, 0.263]

## H-MD2: activity scaling with W (FIT half, day fixed effects)

| segment | y | slope on log W | SE (cl. symbol) | LOO R^2 | LOO CI | Wald p (day x log W) | rows | symbols | days |
|---|---|---|---|---|---|---|---|---|---|
| llm_in_sample | rate | 0.536 | 0.024 | 0.905 | [0.56, 0.942] | 0.365 | 76 | 39 | 2 |
| llm_in_sample | impact | -0.509 | 0.146 | 0.623 | [0.393, 0.79] | 0.785 | 76 | 39 | 2 |
| llm_in_sample | depth | 0.716 | 0.202 | 0.668 | [-0.928, 0.98] | 0.298 | 14 | 7 | 2 |
| clean_holdout | rate | 0.539 | 0.018 | 0.917 | [0.847, 0.947] | 7.42e-04 | 204 | 51 | 4 |
| clean_holdout | impact | -0.520 | 0.084 | 0.595 | [0.406, 0.768] | 0.013 | 204 | 51 | 4 |
| clean_holdout | depth | 0.746 | 0.217 | 0.540 | [0.026, 0.922] | 0.254 | 36 | 9 | 4 |
| pooled | rate | 0.538 | 0.016 | 0.917 | [0.784, 0.946] | 0.003 | 280 | 51 | 6 |
| pooled | impact | -0.518 | 0.088 | 0.614 | [0.443, 0.77] | 0.031 | 280 | 51 | 6 |
| pooled | depth | 0.737 | 0.185 | 0.625 | [0.048, 0.965] | 0.273 | 50 | 9 | 6 |

Per-day slopes (pilot 2026-11-10 one-day values: rate ~W^0.54, depth ~W^0.68, impact ~W^-0.75):

| day | rate slope | impact slope | depth slope | rate SE | impact SE |
|---|---|---|---|---|---|
| 2025-09-05 | 0.529 | -0.503 | n/a | 0.026 | 0.179 |
| 2025-12-18 | 0.546 | -0.521 | n/a | 0.033 | 0.184 |
| 2026-07-12 | 0.522 | -0.599 | 1.023 | 0.027 | 0.108 |
| 2026-07-13 | 0.524 | -0.651 | 0.802 | 0.022 | 0.084 |
| 2026-09-23 | 0.582 | -0.399 | 1.350 | 0.032 | 0.143 |
| 2026-11-10 | 0.540 | -0.487 | 0.759 | 0.032 | 0.086 |

## H-MD3: order-sign long memory

| segment | power law & 0<gamma<1 | share | mean Hurst 1 s signed flow | CI lo (days) | CI hi | days |
|---|---|---|---|---|---|---|
| llm_in_sample | 67/72 | 0.931 | 0.547 | 0.534 | 0.559 | 2 |
| clean_holdout | 143/175 | 0.817 | 0.540 | 0.521 | 0.564 | 4 |
| pooled | 210/247 | 0.850 | 0.542 | 0.527 | 0.560 | 6 |

| day | k/n | share | mean Hurst |
|---|---|---|---|
| 2025-09-05 | 31/34 | 0.912 | 0.534 |
| 2025-12-18 | 36/38 | 0.947 | 0.559 |
| 2026-07-12 | 29/38 | 0.763 | 0.530 |
| 2026-07-13 | 34/41 | 0.829 | 0.517 |
| 2026-09-23 | 36/48 | 0.750 | 0.575 |
| 2026-11-10 | 44/48 | 0.917 | 0.531 |

## H-MD4: spread -> next 5 / 15 min volatility on free-spread symbol-days

| segment | h (min) | qualifying | gain > 0 | share | median gain | sign-test p | non-qualifying share > 0 | non-qual. median gain |
|---|---|---|---|---|---|---|---|---|
| llm_in_sample | 5 | 1 | 0/1 | 0.000 | -0.003 | 1.000 | 0.412 | -8.42e-04 |
| llm_in_sample | 15 | 1 | 0/1 | 0.000 | -0.016 | 1.000 | 0.471 | -0.005 |
| clean_holdout | 5 | 11 | 8/11 | 0.727 | 0.006 | 0.113 | 0.405 | -6.30e-04 |
| clean_holdout | 15 | 11 | 9/11 | 0.818 | 0.009 | 0.033 | 0.459 | -2.03e-04 |
| pooled | 5 | 12 | 8/12 | 0.667 | 0.006 | 0.194 | 0.407 | -7.36e-04 |
| pooled | 15 | 12 | 9/12 | 0.750 | 0.008 | 0.073 | 0.463 | -4.98e-04 |

| day | symbol | FIT share > 1 tick | qualifying | gain 5m | gain 15m | t TEST 5m |
|---|---|---|---|---|---|---|
| 2025-09-05 | DOGEUSDT | 0.002 | no | -0.010 | -0.011 | 0.598 |
| 2025-09-05 | SOLUSDT | 0.002 | no | -0.008 | -0.005 | -0.299 |
| 2025-09-05 | BTCUSDT | 0.001 | no | 0.007 | 0.005 | 1.750 |
| 2025-09-05 | ETHUSDT | 0.003 | no | 0.012 | 0.008 | 1.396 |
| 2025-09-05 | HYPEUSDT | 0.185 | no | -0.002 | -0.005 | 2.162 |
| 2025-09-05 | XRPUSDT | 8.80e-04 | no | -8.42e-04 | -0.007 | -0.363 |
| 2025-09-05 | KASUSDT | 0.177 | no | 0.003 | -0.014 | 2.236 |
| 2025-09-05 | ATOMUSDT | 3.24e-04 | no | -0.004 | -0.011 | -1.894 |
| 2025-09-05 | CARVUSDT | 0.134 | no | 0.002 | 0.003 | 1.273 |
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
| 2025-09-05 | 11.991 | 9.476 | 10.696 | -3.833 | [3.1, 19.3] | [-2.2, 17.5] | 81.257 | 10.120 | 11.343 | 13 |
| 2025-12-18 | 30.771 | 12.008 | 20.331 | 12.524 | [22.9, 38.6] | [3.4, 21.1] | 81.522 | 22.353 | -2.000 | 12 |
| 2026-07-12 | 25.631 | 16.551 | 26.341 | -6.392 | [18.3, 33.8] | [8.5, 23.7] | 59.113 | 23.870 | 13.897 | 12 |
| 2026-07-13 | 37.224 | 24.278 | 12.386 | 10.989 | [28.2, 44.8] | [15.3, 31.8] | 73.136 | 23.242 | 21.449 | 13 |
| 2026-09-23 | 39.182 | 35.127 | 27.555 | 18.035 | [31.3, 46.8] | [27.0, 42.8] | 72.014 | 28.162 | 27.094 | 15 |
| 2026-11-10 | 26.820 | 28.461 | 21.195 | 7.852 | [17.8, 32.0] | [19.5, 34.8] | 83.947 | 25.715 | 22.694 | 15 |

| segment | ETH net > 0 | SOL net > 0 | DOGE net > 0 | BTC net > 0 | mean BTC - ETH | CI (days) |
|---|---|---|---|---|---|---|
| llm_in_sample | 2/2 | 2/2 | 2/2 | 1/2 | -17.035 | [-18.2, -15.8] |
| clean_holdout | 4/4 | 4/4 | 4/4 | 3/4 | -24.593 | [-29.3, -20.1] |
| pooled | 6/6 | 6/6 | 6/6 | 4/6 | -22.074 | [-26.6, -18.1] |

## H-MD6: simulator v1 vs v0 on held-out half-days (12 L2 symbols)

| day | class | symbols | v1 inside (of 5) | v0 inside | v1 facts inside | v0 facts inside |
|---|---|---|---|---|---|---|
| 2025-09-05 | major | 5 | 2 | 0 | flow_acf, impact | - |
| 2025-09-05 | alt | 4 | 1 | 2 | flow_acf | flow_acf, impact |
| 2025-12-18 | major | 5 | 1 | 2 | impact | impact, spread |
| 2025-12-18 | alt | 4 | 3 | 1 | kurtosis, flow_acf, impact | impact |
| 2026-07-12 | major | 5 | 1 | 1 | flow_acf | spread |
| 2026-07-12 | alt | 4 | 2 | 1 | flow_acf, impact | impact |
| 2026-07-12 | tradfi | 3 | 2 | 2 | kurtosis, spread | flow_acf, impact |
| 2026-07-13 | major | 5 | 2 | 0 | flow_acf, impact | - |
| 2026-07-13 | alt | 4 | 1 | 1 | flow_acf | flow_acf |
| 2026-07-13 | tradfi | 3 | 3 | 2 | kurtosis, flow_acf, impact | kurtosis, flow_acf |
| 2026-09-23 | major | 5 | 0 | 2 | - | kurtosis, impact |
| 2026-09-23 | alt | 4 | 3 | 2 | kurtosis, flow_acf, impact | kurtosis, impact |
| 2026-09-23 | tradfi | 3 | 2 | 1 | kurtosis, flow_acf | flow_acf |
| 2026-11-10 | major | 5 | 3 | 2 | kurtosis, flow_acf, impact | kurtosis, impact |
| 2026-11-10 | alt | 4 | 2 | 2 | kurtosis, flow_acf | flow_acf, impact |
| 2026-11-10 | tradfi | 3 | 5 | 0 | kurtosis, absr_acf, flow_acf, impact, spread | - |

| segment | v1 > v0 class-days | share | majors v1 >= 4/5 | mean v1 - v0 |
|---|---|---|---|---|
| llm_in_sample | 2/4 | 0.500 | 0/2 | 0.500 |
| clean_holdout | 7/12 | 0.583 | 0/4 | 0.833 |
| pooled | 9/16 | 0.562 | 0/6 | 0.750 |

Share of class-days with the fact inside the envelope: kurtosis v1 0.50 / v0 0.25, absr_acf v1 0.06 / v0 0.00, flow_acf v1 0.81 / v0 0.38, impact v1 0.56 / v0 0.56, spread v1 0.12 / v0 0.12

## Day profiles and differences from the 2026-11-10 pilot

| day | segment | symbols | major events/s TEST | major TEST/FIT events | BTC sigma_day (bp) | BTC notional (USD, FIT x2) | median gamma | median sign ACF1 | median zeta (prints) | median Hurst | major prints/order | median burst weight | DY total |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2025-09-05 | llm_in_sample | 37 | 4.949 | 1.822 | 143.915 | 7.34e+09 | 0.805 | 0.403 | 2.110 | 0.529 | 3.764 | 0.375 | 81.257 |
| 2025-12-18 | llm_in_sample | 39 | 7.009 | 3.670 | 177.934 | 3.80e+09 | 0.624 | 0.390 | 1.685 | 0.555 | 3.812 | 0.297 | 81.522 |
| 2026-07-12 | clean_holdout | 51 | 0.743 | 0.849 | 114.011 | 2.26e+09 | 0.668 | 0.396 | 1.734 | 0.517 | 4.124 | 0.277 | 59.113 |
| 2026-07-13 | clean_holdout | 51 | 1.244 | 0.966 | 212.754 | 5.26e+09 | 0.748 | 0.343 | 1.729 | 0.520 | 4.350 | 0.283 | 73.136 |
| 2026-09-23 | clean_holdout | 51 | 4.342 | 1.564 | 164.161 | 5.35e+09 | 0.683 | 0.391 | 1.432 | 0.580 | 4.939 | 0.320 | 72.014 |
| 2026-11-10 | clean_holdout | 51 | 2.509 | 2.610 | 177.012 | 2.81e+09 | 0.640 | 0.376 | 1.756 | 0.535 | 4.041 | 0.249 | 83.947 |

Hayashi-Yoshida lead-lag (TEST half, crypto targets): share of peaks at lag 0 and in (0.1, 30] s:

| day | ETH peak at 0 | SOL peak at 0 | BTC peak at 0 | ETH positive peak | SOL positive peak | BTC positive peak | ETH median rho(0) |
|---|---|---|---|---|---|---|---|
| 2025-09-05 | 0.889 | 0.917 | 0.889 | 0.083 | 0.083 | 0.056 | 0.473 |
| 2025-12-18 | 0.868 | 0.895 | 0.658 | 0.079 | 0.079 | 0.079 | 0.576 |
| 2026-07-12 | 0.769 | 0.692 | 0.564 | 0.154 | 0.308 | 0.103 | 0.357 |
| 2026-07-13 | 0.821 | 0.718 | 0.718 | 0.103 | 0.231 | 0.103 | 0.442 |
| 2026-09-23 | 0.923 | 0.949 | 0.769 | 0.026 | 0.026 | 0.051 | 0.313 |
| 2026-11-10 | 0.974 | 0.923 | 0.897 | 0.000 | 0.000 | 0.000 | 0.296 |

| day | free-spread L2 symbols (FIT share > 1 tick >= 25%) | missing trade archives (404) | missing ob200 (404) |
|---|---|---|---|
| 2025-09-05 | - | GRAMUSDT, MONUSDT, ASTERUSDT, CLUSDT, BZUSDT, XAUUSDT, XAGUSDT, NVDAUSDT, QQQUSDT, TSLAUSDT, MSTRUSDT, COINUSDT, CRWVUSDT, GOOGLUSDT | NVDAUSDT, QQQUSDT, XAUUSDT |
| 2025-12-18 | CARVUSDT | GRAMUSDT, CLUSDT, BZUSDT, XAUUSDT, XAGUSDT, QQQUSDT, NVDAUSDT, TSLAUSDT, MSTRUSDT, CRWVUSDT, COINUSDT, GOOGLUSDT | NVDAUSDT, QQQUSDT, XAUUSDT |
| 2026-07-12 | QQQUSDT, CARVUSDT | - | - |
| 2026-07-13 | ATOMUSDT, QQQUSDT, CARVUSDT | - | - |
| 2026-09-23 | HYPEUSDT, QQQUSDT, KASUSDT, CARVUSDT | - | - |
| 2026-11-10 | CARVUSDT, KASUSDT | - | - |

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
