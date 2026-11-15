# Discretised observation space: how many cells carry the volatility signal (H-DS1 to H-DS3)

_Program 2026-11-12 (acceleration, agent C). Data as of 2026-11-12; knowledge cutoff 2026-07-01. Code: `src/ai_trader/ml/discretize.py` (lattices, k-means codebooks, honest trees, empirical-Bayes cells) and `src/ai_trader/research/discrete_states_study.py`; tests `tests/test_discretize.py`; artefacts `data/models/ml/discrete/`. Deterministic research tools only; the LLM decision layer is not trained or backtested here (principle 1)._

## Summary

| hypothesis | in-sample | holdout | pooled | headline |
|---|---|---|---|---|
| H-DS1: <= 64 cells recover >= 80% of the XGB vol gain | inconclusive | inconclusive | **inconclusive** | best (LAT1, chosen in-sample) -6.0% QLIKE (t -3.01) in-sample, -7.4% (t -2.13) holdout; share of the XGB gain 1.34 [-5.96, 7.55] / 0.47 [0.17, 1.02] |
| H-DS2: the cell ordering is stable across the cutoff | - | - | **inconclusive** | Spearman 0.76 [0.46, 0.88] over 30 of 36 cells |
| H-DS3: no discrete state predicts direction stably (null) | inconclusive | inconclusive | **inconclusive** | best in-sample direction cells KM64: holdout Brier skill 0.0048 [-0.0002, 0.0087]; return ordering Spearman 0.00 [-0.13, 0.41] |

Holm within DS (p): H-DS1 1.000, H-DS2 1.000, H-DS3 1.000. Common sample 35527 rows; the re-fitted log-HAR matches H-NN1's to a max relative difference of 0.0e+00.

**How much do a few cells carry?** The best discretisation chosen in-sample is LAT1 (weekday group x vol tercile x v0 direction (36)): log-HAR times one shrunk factor per cell. It cuts next-day QLIKE by -6.0% (t -3.01) in-sample and -7.4% (t -2.13) in the holdout. That beats XGB in-sample, where the XGB gain itself is noise (-4.5%, t -0.98), but recovers only 0.47 of XGB's holdout gain (-15.7%); the share CI [0.17, 1.02] neither clears nor rules out the 0.8 bar, so H-DS1 is inconclusive. 9 of 11 discretisations clear the bar in-sample, 0 in the holdout. The holdout shortfall is concentrated on weekends: on Saturday / Sunday forecast days the cells give -15.8% against XGB's -26.4%.

**What the cells say (cell table below).** Almost all of the cell signal is the forecast weekday, which log-HAR lacks, modulated by the v0 trend state. Relative to log-HAR: Saturday with a range or down trend -49.3% in-sample / -57.7% holdout; Monday +44.8% / +24.6%; Tue-Fri in a v0 up-trend +20.4% / +28.0%. An up-trend shrinks the Saturday discount (-27.8% in-sample, +9.7% on 111 holdout rows). Within a weekday the vol tercile matters much less than in the H-NN3 attributions, because log-HAR already carries the level.

**Do the cells keep their meaning?** Over 30 cells with at least 30 rows in both segments, the frozen LAT1 cell ordering has Spearman 0.76 (bootstrap CI [0.46, 0.88], Fisher [0.55, 0.88]), so H-DS2 is inconclusive against the bar of 0.8. The weekday-level structure is stable; the order within a weekday group is noisier (holdout cells have 20-120 rows).

**Direction.** No cell table forecasts the sign of the next-day return. The best in-sample direction cells (KM64) have in-sample Brier skill 0.0003 and holdout skill 0.0048 [-0.0002, 0.0087]. For most partitions the empirical-Bayes estimate puts the between-cell variance of returns at zero, which shrinks every cell to the base rate (the ordering is then defined as rho = 0). The return ordering of KM64 has Spearman 0.00 across segments. H-DS3 (the null) is inconclusive.

## Methods

- **Panel and protocol** = H-NN1: universe L (23 crypto perps), 52 point-in-time daily features, complete rows, decision at 00:00 UTC of D+1, expanding monthly refits 2022-04 to 2026-11 with training labels ending at least one day before the test month. The XGB forecasts are H-NN1's own walk-forward forecasts on the same rows.
- **Cell value**: in each refit, u = log(RV_{D+1} / log-HAR fitted value) on the training rows; the cell value is the empirical-Bayes shrunk mean of u - mean(u): date-clustered sampling variances, one pooled kappa = sigma^2 / tau^2 per refit (moment estimator), weight n / (n + kappa), shrink toward 0 (= plain log-HAR). Forecast = log-HAR x exp(cell value).
- **Discretisations** (fitted on training rows only, <= 64 cells): WD7 weekday; V0 the 9 v0 regime cells; LAT1 weekday group (Mon / Tue-Fri / Sat / Sun) x vol_pct_1y tercile x v0 direction (36); LAT2 weekday group x lrv_w tercile x market-day tercile (36); LAT3 weekday x vol tercile x v0 direction (63); KM8-KM64 k-means on the top-10 H-NN3 TreeSHAP inputs + dow_cos; HT3 / HT6 honest trees (split on alternating training weeks, leaf values from the other weeks). Direction uses the same partitions with return trees HT3r / HT6r.
- **Selection**: the best vol discretisation has the lowest walk-forward llm_in_sample QLIKE; the best direction one the highest in-sample Brier skill. H-DS2 / H-DS3 orderings use the partition frozen at the 2026-07 refit (labels before the cutoff), with out-of-sample HAR residuals, cells with >= 30 rows in both segments, and a 10-day date-block bootstrap within each segment.

## Results

### H-DS1: QLIKE vs per-asset log-HAR, and share of the XGB gain

| discretisation | cells | in-sample rel | t | share of XGB | holdout rel | t | share of XGB |
|---|---|---|---|---|---|---|---|
| XGB (H-NN1 reference) | - | -4.5% | -0.98 | - | -15.7% | -2.40 | - |
| MLP (H-NN1) | - | -8.0% | -2.92 | - | -6.2% | -0.89 | - |
| log-HAR + weekday dummies (H-NN1 exploratory) | - | -5.5% | -2.92 | - | -2.3% | -0.57 | - |
| WD7: weekday (7 cells) | 7 | -5.4% | -2.96 | 1.21 | -1.6% | -0.37 | 0.10 |
| V0: v0 regime (9) | 9 | -0.3% | -1.64 | 0.06 | -2.5% | -1.93 | 0.16 |
| LAT1: weekday group x vol tercile x v0 direction (36) **(best)** | 36 | -6.0% | -3.01 | 1.34 | -7.4% | -2.13 | 0.47 |
| LAT2: weekday group x 7d level tercile x market-day tercile (36) | 36 | -4.9% | -1.88 | 1.10 | -2.8% | -0.75 | 0.18 |
| LAT3: weekday x vol tercile x v0 direction (63) | 63 | -5.5% | -2.67 | 1.24 | -6.2% | -2.18 | 0.39 |
| KM8: k-means 8 | 8 | -5.5% | -3.37 | 1.23 | -3.2% | -0.87 | 0.20 |
| KM16: k-means 16 | 16 | -5.0% | -3.15 | 1.12 | -5.3% | -1.44 | 0.34 |
| KM32: k-means 32 | 32 | -4.7% | -3.12 | 1.06 | -6.5% | -1.55 | 0.41 |
| KM64: k-means 64 | 64 | -5.9% | -3.12 | 1.32 | -5.9% | -1.97 | 0.38 |
| HT3: honest tree depth 3 | 8 | -4.7% | -2.57 | 1.06 | -5.1% | -1.93 | 0.33 |
| HT6: honest tree depth 6 | 27 | -5.8% | -2.51 | 1.29 | -1.8% | -0.51 | 0.11 |

XGB gain (diff) -0.0174 in-sample (t -0.98), -0.0646 holdout (t -2.40), on 1552 / 133 dates.

Best (LAT1) against the references (diff, NW t): vs XGB -0.0059 (-0.51) in-sample, +0.0340 (1.67) holdout; vs log-HAR + weekday -0.0020 (-0.54) / -0.0212 (-1.27). Passing the H-DS1 bar (share >= 0.8 and t < -2): in-sample WD7, LAT1, LAT3, KM8, KM16, KM32, KM64, HT3, HT6; holdout none.

| segment | forecast day | best rel vs HAR | t | XGB rel vs HAR | t |
|---|---|---|---|---|---|
| llm_in_sample | Mon-Fri | -4.4% | -3.02 | -5.4% | -1.59 |
| llm_in_sample | Sat-Sun | -9.3% | -1.66 | -2.4% | -0.20 |
| clean_holdout | Mon-Fri | -3.2% | -0.67 | -10.3% | -1.28 |
| clean_holdout | Sat-Sun | -15.8% | -2.46 | -26.4% | -3.74 |

### H-DS2: is the cell ordering stable across the cutoff? (frozen partitions, Spearman over cells with >= 30 rows in both segments)

| discretisation | rho | bootstrap 95% CI | Fisher 95% CI | cells used |
|---|---|---|---|---|
| WD7 | 0.89 | [0.25, 0.96] | [0.43, 0.98] | 7 |
| V0 | 0.75 | [0.00, 0.88] | [0.17, 0.94] | 9 |
| LAT1 **(best, the test)** | 0.76 | [0.46, 0.88] | [0.55, 0.88] | 30 |
| LAT2 | 0.70 | [0.32, 0.87] | [0.45, 0.85] | 29 |
| LAT3 | 0.62 | [0.30, 0.71] | [0.41, 0.77] | 47 |
| KM8 | 0.69 | [0.48, 0.95] | [-0.03, 0.94] | 8 |
| KM16 | 0.82 | [0.57, 0.94] | [0.54, 0.93] | 16 |
| KM32 | 0.74 | [0.52, 0.82] | [0.52, 0.87] | 30 |
| KM64 | 0.78 | [0.41, 0.79] | [0.63, 0.87] | 48 |
| HT3 | 0.43 | [-0.03, 1.00] | [-0.59, 0.92] | 6 |
| HT6 | 0.77 | [0.56, 0.91] | [0.50, 0.90] | 21 |

Spread of the shrunk cell effects (sd over the cells used): 0.319 in-sample, 0.276 holdout log points; segment kappas 39 / 27.

### Cell table: LAT1 (weekday group x vol tercile x v0 direction (36)), partition frozen at the 2026-07 refit

Effect = EB-shrunk next-day RV relative to the per-asset log-HAR forecast (exp(cell value) - 1), relative to the segment's average residual. 'train' is the value the frozen refit actually used (estimated on its own training rows). z = raw cell mean / date-clustered SE.

| cell | state in plain words | rows in | effect in | z in | rows ho | effect ho | z ho | train |
|---|---|---|---|---|---|---|---|---|
| 19 | forecast day Saturday, vol (own 1y percentile) in the bottom third (< 0.256), v0 range | 702 | -54.0% | -11.1 | 75 | -44.2% | -5.3 | -52.7% |
| 22 | forecast day Saturday, vol (own 1y percentile) in the middle third (0.256 to 0.607), v0 range | 728 | -49.9% | -13.6 | 56 | -53.0% | -5.4 | -48.8% |
| 18 | forecast day Saturday, vol (own 1y percentile) in the bottom third (< 0.256), v0 trend down | 328 | -47.5% | -5.8 | 51 | -37.7% | -6.6 | -45.2% |
| 21 | forecast day Saturday, vol (own 1y percentile) in the middle third (0.256 to 0.607), v0 trend down | 468 | -45.0% | -6.8 | 23 | -38.0% | -4.4 | -42.6% |
| 25 | forecast day Saturday, vol (own 1y percentile) in the top third (>= 0.607), v0 range | 850 | -43.4% | -8.3 | 91 | -44.9% | -3.8 | -43.2% |
| 24 | forecast day Saturday, vol (own 1y percentile) in the top third (>= 0.607), v0 trend down | 425 | -38.9% | -4.4 | 30 | -38.6% | -3.5 | -36.8% |
| 20 | forecast day Saturday, vol (own 1y percentile) in the bottom third (< 0.256), v0 trend up | 297 | -32.6% | -5.0 | 16 | -11.0% | -1.4 | -32.1% |
| 23 | forecast day Saturday, vol (own 1y percentile) in the middle third (0.256 to 0.607), v0 trend up | 391 | -26.5% | -4.7 | 30 | -8.6% | -0.6 | -27.2% |
| 26 | forecast day Saturday, vol (own 1y percentile) in the top third (>= 0.607), v0 trend up | 451 | -19.4% | -2.7 | 65 | +24.9% | 0.7 | -21.4% |
| 31 | forecast day Sunday, vol (own 1y percentile) in the middle third (0.256 to 0.607), v0 range | 795 | -17.1% | -4.0 | 55 | -1.7% | -0.1 | -18.1% |
| 30 | forecast day Sunday, vol (own 1y percentile) in the middle third (0.256 to 0.607), v0 trend down | 363 | -16.7% | -2.1 | 46 | +10.8% | 0.4 | -16.4% |
| 27 | forecast day Sunday, vol (own 1y percentile) in the bottom third (< 0.256), v0 trend down | 376 | -15.1% | -1.7 | 48 | -12.5% | -1.7 | -13.7% |
| 28 | forecast day Sunday, vol (own 1y percentile) in the bottom third (< 0.256), v0 range | 964 | -14.9% | -3.1 | 80 | -14.4% | -1.4 | -15.9% |
| 32 | forecast day Sunday, vol (own 1y percentile) in the middle third (0.256 to 0.607), v0 trend up | 353 | -6.6% | -1.3 | 33 | -0.4% | -0.0 | -7.3% |
| 34 | forecast day Sunday, vol (own 1y percentile) in the top third (>= 0.607), v0 range | 669 | -5.0% | -0.8 | 47 | -8.5% | -0.8 | -6.6% |
| 16 | forecast day Tue-Fri, vol (own 1y percentile) in the top third (>= 0.607), v0 range | 3113 | -3.3% | -1.1 | 289 | +1.8% | 0.2 | -4.1% |
| 33 | forecast day Sunday, vol (own 1y percentile) in the top third (>= 0.607), v0 trend down | 337 | -0.4% | -0.0 | 21 | +3.1% | 0.2 | -1.0% |
| 35 | forecast day Sunday, vol (own 1y percentile) in the top third (>= 0.607), v0 trend up | 451 | +0.6% | 0.1 | 73 | +5.2% | 0.4 | -0.2% |
| 13 | forecast day Tue-Fri, vol (own 1y percentile) in the middle third (0.256 to 0.607), v0 range | 3102 | +1.5% | 0.5 | 258 | +9.0% | 1.3 | +1.7% |
| 29 | forecast day Sunday, vol (own 1y percentile) in the bottom third (< 0.256), v0 trend up | 336 | +1.5% | 0.3 | 34 | -14.3% | -2.7 | +1.6% |
| 10 | forecast day Tue-Fri, vol (own 1y percentile) in the bottom third (< 0.256), v0 range | 3050 | +4.3% | 1.2 | 397 | +11.6% | 1.4 | +6.1% |
| 9 | forecast day Tue-Fri, vol (own 1y percentile) in the bottom third (< 0.256), v0 trend down | 1643 | +10.2% | 1.7 | 186 | -3.1% | -0.3 | +12.5% |
| 12 | forecast day Tue-Fri, vol (own 1y percentile) in the middle third (0.256 to 0.607), v0 trend down | 1676 | +13.7% | 2.9 | 136 | +23.9% | 2.0 | +12.4% |
| 11 | forecast day Tue-Fri, vol (own 1y percentile) in the bottom third (< 0.256), v0 trend up | 1079 | +16.7% | 3.5 | 106 | +19.9% | 1.3 | +17.9% |
| 15 | forecast day Tue-Fri, vol (own 1y percentile) in the top third (>= 0.607), v0 trend down | 1824 | +19.0% | 3.3 | 97 | +3.6% | 0.2 | +17.2% |
| 14 | forecast day Tue-Fri, vol (own 1y percentile) in the middle third (0.256 to 0.607), v0 trend up | 1421 | +19.0% | 5.3 | 129 | +35.4% | 2.0 | +18.9% |
| 17 | forecast day Tue-Fri, vol (own 1y percentile) in the top third (>= 0.607), v0 trend up | 1630 | +22.6% | 6.0 | 150 | +14.3% | 2.0 | +21.5% |
| 7 | forecast day Monday, vol (own 1y percentile) in the top third (>= 0.607), v0 range | 680 | +28.8% | 4.5 | 60 | -6.3% | -0.6 | +26.1% |
| 3 | forecast day Monday, vol (own 1y percentile) in the middle third (0.256 to 0.607), v0 trend down | 301 | +33.8% | 3.4 | 26 | +5.7% | 1.5 | +34.6% |
| 5 | forecast day Monday, vol (own 1y percentile) in the middle third (0.256 to 0.607), v0 trend up | 359 | +34.3% | 5.3 | 26 | +15.7% | 3.5 | +33.5% |
| 0 | forecast day Monday, vol (own 1y percentile) in the bottom third (< 0.256), v0 trend down | 450 | +34.6% | 4.1 | 33 | +13.4% | 1.8 | +36.2% |
| 4 | forecast day Monday, vol (own 1y percentile) in the middle third (0.256 to 0.607), v0 range | 859 | +38.6% | 6.0 | 64 | +15.3% | 1.8 | +35.6% |
| 1 | forecast day Monday, vol (own 1y percentile) in the bottom third (< 0.256), v0 range | 1029 | +41.9% | 6.2 | 117 | +31.6% | 2.8 | +38.1% |
| 8 | forecast day Monday, vol (own 1y percentile) in the top third (>= 0.607), v0 trend up | 336 | +42.7% | 6.3 | 60 | +31.9% | 4.7 | +40.4% |
| 2 | forecast day Monday, vol (own 1y percentile) in the bottom third (< 0.256), v0 trend up | 330 | +46.8% | 4.1 | 32 | +27.0% | 3.1 | +43.7% |
| 6 | forecast day Monday, vol (own 1y percentile) in the top third (>= 0.607), v0 trend down | 302 | +98.7% | 4.3 | 19 | -13.8% | -0.9 | +97.0% |

Kappa (pseudo-rows): in-sample 39, holdout 27, frozen training 40.

**Merged cells (post hoc, defined after reading the table; raw centred mean log(RV / log-HAR), date-clustered z):**

| merged cells | rows in | effect in | z in | rows ho | effect ho | z ho |
|---|---|---|---|---|---|---|
| Saturday, v0 range or trend down | 3501 | -49.3% | -13.6 | 326 | -57.7% | -7.2 |
| Saturday, v0 trend up | 1139 | -27.8% | -5.1 | 111 | +9.7% | 0.3 |
| Sunday | 4644 | -10.5% | -3.2 | 437 | -6.7% | -0.6 |
| Monday | 4646 | +44.8% | 9.8 | 437 | +24.6% | 2.4 |
| Monday, vol in the top third and v0 trend up | 336 | +48.7% | 6.3 | 60 | +49.2% | 4.7 |
| Tue-Fri, v0 trend up | 4130 | +20.4% | 6.9 | 385 | +28.0% | 2.2 |
| Tue-Fri, v0 range | 9265 | +0.8% | 0.3 | 944 | +8.4% | 1.3 |
| Tue-Fri, v0 trend down | 5143 | +14.8% | 3.7 | 419 | +8.0% | 0.7 |

### H-DS3: direction from cells (Brier skill vs the training base rate; cell ordering by next-day vol-scaled return)

| discretisation | in-sample skill | 95% CI | holdout skill | 95% CI | return ordering rho | bootstrap 95% CI | Fisher 95% CI |
|---|---|---|---|---|---|---|---|
| WD7 | 0.0000 | [-0.0008, 0.0007] | -0.0005 | [-0.0079, 0.0042] | 0.00 | [-0.54, 0.68] | [-0.75, 0.75] |
| V0 | -0.0002 | [-0.0005, 0.0000] | 0.0000 | [-0.0000, 0.0000] | 0.00 | [-0.58, 0.30] | [-0.66, 0.66] |
| LAT1 | -0.0000 | [-0.0010, 0.0011] | 0.0001 | [-0.0003, 0.0005] | 0.00 | [-0.37, 0.43] | [-0.36, 0.36] |
| LAT2 | -0.0005 | [-0.0018, 0.0007] | -0.0012 | [-0.0039, 0.0011] | -0.24 | [-0.53, 0.23] | [-0.56, 0.14] |
| LAT3 | -0.0002 | [-0.0024, 0.0016] | 0.0019 | [-0.0056, 0.0073] | 0.00 | [-0.20, 0.41] | [-0.29, 0.29] |
| KM8 | -0.0002 | [-0.0006, 0.0001] | 0.0000 | [-0.0000, 0.0000] | 0.00 | [-0.41, 0.62] | [-0.70, 0.70] |
| KM16 | -0.0001 | [-0.0004, 0.0002] | -0.0000 | [-0.0000, 0.0000] | 0.00 | [-0.22, 0.64] | [-0.50, 0.50] |
| KM32 | -0.0001 | [-0.0010, 0.0008] | 0.0029 | [0.0001, 0.0062] | 0.00 | [0.00, 0.60] | [-0.36, 0.36] |
| KM64 **(best in-sample)** | 0.0003 | [-0.0013, 0.0019] | 0.0048 | [-0.0002, 0.0087] | 0.00 | [-0.13, 0.41] | [-0.28, 0.28] |
| HT3r | -0.0003 | [-0.0024, 0.0015] | -0.0006 | [-0.0022, 0.0013] | 0.00 | [-0.60, 1.00] | [-0.88, 0.88] |
| HT6r | -0.0012 | [-0.0032, 0.0008] | -0.0001 | [-0.0096, 0.0082] | 0.00 | [-0.32, 0.43] | [-0.45, 0.45] |

P*(return ordering rho > 0.3) 0.144; P*(holdout skill > 0) 0.969.

### Verdicts (after Holm within DS)

| hypothesis | llm_in_sample | clean_holdout | pooled | Holm p |
|---|---|---|---|---|
| H-DS1 | inconclusive | inconclusive | inconclusive | 1.000 |
| H-DS2 | - | - | inconclusive | 1.000 |
| H-DS3 | inconclusive | inconclusive | inconclusive | 1.000 |

## Caveats

- The holdout is 133 dates (2026-07 to 2026-11). The XGB holdout gain that H-DS1 is measured against rests on those dates and is concentrated on weekends.
- 'Best' was chosen on in-sample QLIKE. Other discretisations (LAT3, KM64) come close in the holdout; see the H-DS1 table.
- The k-means inputs are the top in-sample H-NN3 TreeSHAP features, computed on all in-sample walk-forward rows. Early in-sample months therefore carry mild feature-selection look-ahead; the holdout is clean. The lattices use only calendar, own-vol percentile, v0 trend and market-day inputs, fixed in advance.
- Percentile bootstrap intervals of a Spearman between two noisy cell tables are biased toward 0: resampling adds a second layer of noise, so the bootstrap median sits below the point estimate. Verdicts use the registered bootstrap; Fisher-z intervals are shown for reference.
- Cell effects are relative to each segment's average log-HAR residual, and are geometric (mean log ratio). Under QLIKE the arithmetic scale differs by roughly s^2/2 per cell, which matters little for the ordering but does matter for an exact multiplier.
- The merged groups behind the expect rules were defined after the cell table was seen. They are rule candidates, not tested hypotheses, and should be registered as forward hypotheses before being promoted.
- Mistake (fixed before recording): with a between-cell variance of 0, EB weights of about 1e-12 left a float-residue ordering that the Spearman read as signal. Cell values are now exactly 0 and the ordering is defined as rho = 0 in that case (diary 'mistake' entry). The walk-forward forecasts were produced before the fix; the difference is below 1e-8 in relative terms.

## Rule candidates

```yaml
- id: "R-DS-1"
  kind: "expect"
  rule: "WHEN the forecast day is Saturday and the v0 trend state is range or down THEN expect next-day realised variance about 54% below the per-asset log-HAR forecast"
  when: "daily (1d) variance expectation for a universe-L crypto perp; merged cells of the frozen LAT1 partition (weekday group x vol tercile x v0 direction (36)): Saturday, v0 range or trend down"
  then: "central next-day variance expectation = log-HAR x 0.46; stops and risk keep the unshrunk risk sigma (R-VOL-2)"
  scope:
    assets: "crypto perps (universe L)"
    horizon: "1d"
    sides: "both"
  effect: "next-day RV vs log-HAR (relative to the segment average) -49.3% in-sample (3501 rows, 219 dates, z -13.6), -57.7% holdout (326 rows, 18 dates, z -7.2); cell group defined after the cell table was seen; grade B needs |z| > 2 in both segments and >= 15 holdout dates"
  confidence: "B"
  evidence: ["H-DS1", "H-DS2"]
  review: "register the merged cell as a forward hypothesis; demote if the effect on dates after 2026-11-12 changes sign or falls below half"
  status: "provisional"
- id: "R-DS-2"
  kind: "expect"
  rule: "WHEN the forecast day is Monday THEN expect next-day realised variance about 34% above the per-asset log-HAR forecast"
  when: "daily (1d) variance expectation for a universe-L crypto perp; merged cells of the frozen LAT1 partition (weekday group x vol tercile x v0 direction (36)): Monday"
  then: "central next-day variance expectation = log-HAR x 1.34; stops and risk keep the unshrunk risk sigma (R-VOL-2)"
  scope:
    assets: "crypto perps (universe L)"
    horizon: "1d"
    sides: "both"
  effect: "next-day RV vs log-HAR (relative to the segment average) +44.8% in-sample (4646 rows, 222 dates, z 9.8), +24.6% holdout (437 rows, 19 dates, z 2.4); cell group defined after the cell table was seen; grade B needs |z| > 2 in both segments and >= 15 holdout dates"
  confidence: "B"
  evidence: ["H-DS1", "H-DS2"]
  review: "register the merged cell as a forward hypothesis; demote if the effect on dates after 2026-11-12 changes sign or falls below half"
  status: "provisional"
- id: "R-DS-3"
  kind: "expect"
  rule: "WHEN the forecast day is Monday, vol is in the top third of its own 1-year range and the v0 trend state is up THEN expect next-day realised variance about 49% above the per-asset log-HAR forecast"
  when: "daily (1d) variance expectation for a universe-L crypto perp; merged cells of the frozen LAT1 partition (weekday group x vol tercile x v0 direction (36)): Monday, vol in the top third and v0 trend up"
  then: "central next-day variance expectation = log-HAR x 1.49; stops and risk keep the unshrunk risk sigma (R-VOL-2)"
  scope:
    assets: "crypto perps (universe L)"
    horizon: "1d"
    sides: "both"
  effect: "next-day RV vs log-HAR (relative to the segment average) +48.7% in-sample (336 rows, 108 dates, z 6.3), +49.2% holdout (60 rows, 10 dates, z 4.7); cell group defined after the cell table was seen; grade B needs |z| > 2 in both segments and >= 15 holdout dates"
  confidence: "C"
  evidence: ["H-DS1", "H-DS2"]
  review: "register the merged cell as a forward hypothesis; demote if the effect on dates after 2026-11-12 changes sign or falls below half"
  status: "provisional"
- id: "R-DS-4"
  kind: "expect"
  rule: "WHEN the forecast day is Tuesday to Friday and the v0 trend state is up THEN expect next-day realised variance about 24% above the per-asset log-HAR forecast"
  when: "daily (1d) variance expectation for a universe-L crypto perp; merged cells of the frozen LAT1 partition (weekday group x vol tercile x v0 direction (36)): Tue-Fri, v0 trend up"
  then: "central next-day variance expectation = log-HAR x 1.24; stops and risk keep the unshrunk risk sigma (R-VOL-2)"
  scope:
    assets: "crypto perps (universe L)"
    horizon: "1d"
    sides: "both"
  effect: "next-day RV vs log-HAR (relative to the segment average) +20.4% in-sample (4130 rows, 652 dates, z 6.9), +28.0% holdout (385 rows, 63 dates, z 2.2); cell group defined after the cell table was seen; grade B needs |z| > 2 in both segments and >= 15 holdout dates"
  confidence: "B"
  evidence: ["H-DS1", "H-DS2"]
  review: "register the merged cell as a forward hypothesis; demote if the effect on dates after 2026-11-12 changes sign or falls below half"
  status: "provisional"
- id: "R-DS-5"
  kind: "prefer"
  rule: "WHEN a central next-day variance expectation is needed and no validated nonlinear tool is available THEN apply the LAT1 cell adjustment to the per-asset log-HAR forecast"
  when: "daily (1d) variance expectation for a universe-L crypto perp"
  then: "forecast = log-HAR x exp(EB-shrunk cell value of the cell table); fall back to plain log-HAR for cells with fewer than 30 training rows"
  scope:
    assets: "crypto perps (universe L)"
    horizon: "1d"
    sides: "both"
  effect: "QLIKE vs log-HAR -6.0% (t -3.01) in-sample, -7.4% (t -2.13) holdout; share of the XGB gain 1.34 / 0.47; H-DS1 inconclusive, H-DS2 rho 0.76"
  confidence: "C"
  evidence: ["H-DS1", "H-DS2"]
  review: "forward test against log-HAR + weekday and the frozen XGB tool for 3 months; demote if the gain is < 2% or t > -2"
  status: "provisional"
- id: "R-DS-6"
  kind: "avoid"
  rule: "WHEN a discrete market state (weekday, vol tercile, trend state, k-means or tree cell) is known THEN don't infer next-day direction from it"
  when: "next-24h direction for crypto perps, from any cell table of standard features"
  then: "treat direction as a coin flip; cell tables are for variance expectations only"
  scope:
    assets: "crypto perps (universe L)"
    horizon: "1d"
    sides: "both"
  effect: "best in-sample direction cells (KM64): holdout Brier skill 0.0048 [-0.0002, 0.0087]; cell ordering by next-day return Spearman 0.00 between segments; H-DS3 inconclusive"
  confidence: "C"
  evidence: ["H-DS3", "H-NN2"]
  review: "a pre-registered direction cell with holdout Brier skill CI lower bound > 0"
  status: "provisional"
```
