# Volatility-model tournament, 2026-11-10 (H-VM1..H-VM5)

**Agent:** A2 (vol-tournament), program `research/plans/2026-11-10_volatility_program.md`.

**Code:**
- `src/ai_trader/features/vol_panel.py`: the forecasters, SV Kalman filter, MCS and Holm.
- `src/ai_trader/research/vol_tournament_study.py`: build and evaluate.
- `tests/test_vol_panel.py`: 10 tests.

**Data:**
- Forecast cache: `data/research/vol_tournament/forecasts.parquet`, 46,065 symbol-days, 38 symbols.
- Machine-readable results: `research/reports/vol_tournament_2026-11-10.json`.
- Registry rows: `research/hypotheses_results.jsonl`.

## Summary

| hypothesis | llm_in_sample | clean_holdout | key numbers (pooled L and T, date-averaged QLIKE, NW t) |
|---|---|---|---|
| H-VM1: panel HAR beats per-asset HAR | **supports** (Holm p 0.003) | inconclusive (Holm p 0.60) | level panel vs level HAR: h1 -1.6% (t -3.42), h5 -3.0% (t -3.09) in-sample; h1 -0.7% (t -0.85), h5 -1.2% (t -1.11) holdout. The gain sits in assets with 1-2 years of history; none at >= 2 years. The log panel adds nothing over per-asset log-HAR (+0.7%, t 1.5) |
| H-VM2: HAR-Q and log-HAR beat HAR | **supports** (Holm p 0.002) | inconclusive (Holm p 0.67) | HAR-Q passes in both segments (per-asset -5.1% t -6.8 / -4.7% t -3.31; panel -4.2% t -7.9 / -3.7% t -4.1). Log-HAR: -11.9% (t -5.0) in-sample, -4.7% with t only -0.42 in the holdout (heavy-tailed date losses) |
| H-VM3: range (GK/PK) inputs beat RV inputs | **contradicts** | **contradicts** | GK-HAR is worse than RV-HAR: +5.4% (t 4.9) / +15.7% (t 4.8) at 1d, +6.6% / +12.6% at 5d. RV+GK is no better than RV (in-sample +5.9%, t 2.7) |
| H-VM4: common crypto vol factor | inconclusive | inconclusive | Median alt R^2 of log RV on the factor: 0.72 in-sample (leave-one-out 0.70), 0.68 holdout (0.65), which passes. The factor adds nothing to the panel log-HAR forecast for alts: -0.1% (t -0.58) / -1.0% (t -1.64), which fails |
| H-VM5: in-sample winner in the holdout 90% MCS (L, 1d and 5d) | (selection) | **supports**, low power | 1d winner: per-asset log-HAR (in-sample MCS = {loghar, p_loghar_f, p_loghar, sv_rv}). It is in the holdout MCS (p 0.17, rank 6 of 25). 5d winner: panel log-HAR, in the holdout MCS (p 0.29, rank 14). But the holdout MCS keeps 22 of 25 models (1d) and 25 of 25 (5d): 131 holdout days cannot separate the models. TR statistic: same membership |

T (reported only; H-VM5 is decided on L):
- 1d: the in-sample winner `loghar` (from PAXG and XAUT only) is in T's holdout MCS (p 0.15, rank 3 of 25).
- 5d: the winner `p_loghar_f` is in it too (p 0.18, rank 16).

Holm within the VM block (per segment, intersection-union p):
- in-sample: VM2 0.002, VM1 0.003, VM4 0.56, VM3 1.0;
- holdout: VM4 0.20, VM1 0.60, VM2 0.67, VM3 1.0.

In-sample support for VM1 and VM2 is textbook (BPQ 2016/2018) and contaminated (plan section 7). Every primary holdout difference keeps its in-sample sign. Among secondary comparisons, log-HAR vs HAR at 5d for L turns positive (+1.3%, t 0.06).

## Recommended default model for daily risk

The selection rule was fixed in code (`recommend()`) before the evaluation ran:
- **Crypto (L):** the in-sample QLIKE winner, provided it is in the holdout 90% MCS.
- **TradFi (T):** the crypto pick if it is in T's holdout MCS on the short-history set. Otherwise the lowest-loss panel
  member of that MCS. This second branch selects on the holdout, so it is flagged.

| asset class | 1d (next-day RV) | 5d (mean daily RV over the next 5 days) |
|---|---|---|
| **crypto, >= 180 days of history** | **per-asset log-HAR** (`loghar`, which is the existing `vol.har`). QLIKE 0.373 in-sample = -12.1% vs level HAR, -17.6% vs robust_bc x 24; holdout 0.417 = -4.5% vs HAR, -18% vs robust_bc x 24 | **panel log-HAR, direct 5d target** (`p_loghar`). QLIKE 0.261 in-sample = -15.5% vs HAR (robust_bc x 24 flat = 1.51 x HAR). Holdout 0.310 (= HAR; robust_bc = 2.27 x HAR; holdout best p_harq -4.7%) |
| **crypto, < 180 days** | panel log-HAR (`p_loghar`): -27.5% vs robust_bc x 24 in the holdout (t -5.2) | `p_loghar` (-74% vs robust_bc, t -5.5) |
| **TradFi perps** (all short history) | **panel HAR-Q pooled over L and T** (`p_harq`): holdout QLIKE 0.653, -30% vs robust_bc x 24 (15 symbols, 131 days). `p_loghar` / `p_loghar_f` are statistically tied (0.705 / 0.703 x robust_bc). **Selected on holdout: confidence C** | **level panel HAR** (rule pick `p_shar`, QLIKE 0.203; `p_har` / `p_harq` tied at 0.208-0.217 x robust_bc, i.e. -79%). `p_loghar` falls just outside this MCS (0.236). **Confidence C** |

**Operational simplification (suggestion, not a test result).**
- Crypto: one model, panel log-HAR at both horizons, costs about 0.7% QLIKE at 1d versus per-asset log-HAR (t 1.5, not
  significant) and covers new listings after about 60 days.
- TradFi: `p_harq` sits in both T holdout MCSs (1d best; 5d 0.217, in the MCS).
- In every universe, horizon and segment, robust_bc scaled to a day (x 24) or flat over 5 days is the worst or
  near-worst contestant. Keep it as the per-bar sigma (V4) only.

## Methods

**Data.** Bybit 1h bars through `research.frame.market_frame` (price sanity, listing burn-in), aggregated by
`features.realized.daily_measures` into complete UTC days (24 bars): RV, RS+, RS-, RQ, BV, JV, Parkinson (PK),
Garman-Klass (GK) and the close-to-close daily return. A day's measures are known at the next 00:00 UTC (`known_ts`).
Series sit on a gap-aware calendar (a missing day is NaN, never shifted). As of 2026-11-10 00:00 UTC.

- Universe L: the 23 crypto perps of `trend_zoo_study.UNIVERSE`.
- Universe T: TradFi perps with at least 120 complete days. SPCXUSDT (117 days) is excluded, which leaves 15.
- Segments: forecasts made before 2026-07-01 00:00 UTC are `llm_in_sample`, later ones `clean_holdout`. T is holdout
  only except for PAXG and XAUT, which have in-sample history from 2025.

**Targets.** RV_{t+1} (h = 1) and mean(RV_{t+1..t+5}) (h = 5). Every forecast is made at the end of day t.

**Contestants** (25; `*` marks an extension beyond the registered list, reported but flagged).

| family | models | definition |
|---|---|---|
| recursive | `ewma` | EWMA(0.94) of daily r^2 (flat over 5d) |
| recursive | `robust_bc` | 24 x sigma_risk_next^2 at the day's hour-23 bar (H-V1b model, risk version, flat over 5d) |
| recursive | `robust_bc_c`* | the tool's central version: sigma_risk_next^2 x (weekday hours + m^2 x TradFi-weekend hours) of the target window |
| GARCH | `garch`, `gjr`, `egarch` | (1,1)-t with constant mean (`arch` 8.0, fit on 100 x r). Refit every 30 days: warm and cold starts, the best sane fit is kept. Analytic 5d means for GARCH/GJR; EGARCH 5d by Monte Carlo (1000 standardised-t paths), so nothing was cut |
| SV | `sv_r2` | log r^2 (Fuller transform, c = 0.02) = h + log chi2 (mean -1.27, var pi^2/2); h is an AR(1). QML Kalman filter, forecast exp(m_k + P_k/2) |
| SV | `sv_r2_cal`* | `sv_r2` x the training ratio mean(r^2) / mean(forecast). The log-chi2 constant assumes Gaussian returns |
| SV | `sv_rv`* | the same state space with log RV as the measurement (free measurement variance H) |
| HAR (levels, OLS) | `har`, `harq`, `shar`, `harj`, `har_gk`, `har_pk`*, `har_rvgk`* | Corsi 1/7/30-day HAR. `harq` adds RV_d x sqrt(RQ_d) (BPQ 2016); `shar` splits RV_d into RS+/RS-; `harj` adds JV_d; `har_gk`/`har_pk` use GK/PK inputs (d/w/m); `har_rvgk` uses both. The target is always RV |
| HAR (logs) | `loghar` | log RV regressors (mean of logs over 7/30 days), log target, back-transform exp(fit + s^2/2). This is the repo's validated `vol.har` (H-VCL1) |
| panel | `p_har`, `p_harq`, `p_shar`, `p_harj`, `p_har_gk`, `p_har_rvgk`, `p_loghar` | pooled slopes over L and T with symbol fixed effects (within estimator). Level panels divide each symbol's target and regressors by c_s^p, where c_s is the symbol's training-mean target |
| panel + factor | `p_loghar_f` | `p_loghar` plus the crypto factor f_t (cross-sectional mean log RV of L) at d/w/m lags. The factor terms are interacted with asset class (crypto / TradFi) |

**Estimation.**
- Expanding window, refit every 30 days.
- A fit at day d uses only pairs whose target is complete by d (j + h <= d).
- Per-asset models need 180 days of own history.
- Panel models refit on a global 30-day grid. A symbol enters once it has 30 complete training pairs, roughly 60-90
  days after listing.
- Level-model insanity filter (BPQ 2016): a forecast outside the training target range is replaced by the training mean.
- No-look-ahead, one-asset panel = per-asset, scale invariance and SV-recovery checks are in `tests/test_vol_panel.py`.

**Evaluation.**
- Losses: QLIKE (primary) and MSE of log variance.
- Paired differences are taken on the common (symbol, day) sample, averaged across symbols per date, with a Newey-West
  t on the date series (clustered by date, which handles 5d overlap).
- MCS (Hansen, Lunde and Nason 2011): Tmax statistic (TR as a robustness check), 90%, circular block bootstrap (block
  10 days, 1000 replications). It runs on date-averaged losses over the common sample of all 25 models, per universe,
  horizon and segment.
- A second T leaderboard, the "short set", contains only models that need no 180-day own history (EWMA, robust_bc, the
  panels). This is where most TradFi symbol-days live.
- Holm within the VM block per segment, using intersection-union p-values (the max over each hypothesis's required parts).

**Verdict rule (fixed before evaluation).**
- supports: every required part has diff < 0 with t < -2;
- contradicts: any part has diff > 0 with t > 2, or no part has diff < 0;
- inconclusive: otherwise.

H-VM4 also needs the median R^2 >= 0.5. H-VM5 is decided on the holdout MCS of L at both horizons.

## Results by hypothesis

### H-VM1

**llm_in_sample: supports** (Holm p 0.003)

| comparison | mean dQLIKE (a - b) | rel. to b | NW t | dates | obs | loss a | loss b |
|---|---|---|---|---|---|---|---|
| h1 | -0.0068 | -1.6% | -3.42 | 2093 | 36047 | 0.4192 | 0.4260 |
| h5 | -0.0093 | -3.0% | -3.09 | 2093 | 36047 | 0.2983 | 0.3076 |

**clean_holdout: inconclusive** (Holm p 0.596)

| comparison | mean dQLIKE (a - b) | rel. to b | NW t | dates | obs | loss a | loss b |
|---|---|---|---|---|---|---|---|
| h1 | -0.0032 | -0.7% | -0.85 | 131 | 3359 | 0.4541 | 0.4572 |
| h5 | -0.0036 | -1.2% | -1.11 | 127 | 3247 | 0.2879 | 0.2915 |

<details><summary>secondary comparisons</summary>

llm_in_sample

| comparison | mean dQLIKE (a - b) | rel. to b | NW t | dates | obs | loss a | loss b |
|---|---|---|---|---|---|---|---|
| L / p_har-har / h1 | -0.0071 | -1.7% | -3.91 | 2093 | 35471 | 0.4149 | 0.4219 |
| L / p_har-har / h5 | -0.0094 | -3.1% | -3.21 | 2093 | 35471 | 0.2962 | 0.3056 |
| L / p_loghar-loghar / h1 | 0.0018 | 0.5% | 1.07 | 2093 | 35471 | 0.3742 | 0.3724 |
| L / p_loghar-loghar / h5 | -0.0029 | -1.1% | -1.21 | 2093 | 35471 | 0.2551 | 0.2579 |
| T / p_har-har / h1 | 0.0175 | 2.5% | 0.27 | 334 | 576 | 0.7210 | 0.7035 |
| T / p_har-har / h5 | 0.0110 | 2.7% | 0.23 | 334 | 576 | 0.4141 | 0.4031 |
| T / p_loghar-loghar / h1 | 0.0621 | 10.6% | 2.61 | 334 | 576 | 0.6484 | 0.5863 |
| T / p_loghar-loghar / h5 | -0.0087 | -3.0% | -0.37 | 334 | 576 | 0.2801 | 0.2888 |
| LT / p_har-har / h1 | -0.0068 | -1.6% | -3.42 | 2093 | 36047 | 0.4192 | 0.4260 |
| LT / p_har-har / h5 | -0.0093 | -3.0% | -3.09 | 2093 | 36047 | 0.2983 | 0.3076 |
| LT / p_loghar-loghar / h1 | 0.0026 | 0.7% | 1.50 | 2093 | 36047 | 0.3780 | 0.3755 |
| LT / p_loghar-loghar / h5 | -0.0030 | -1.2% | -1.25 | 2093 | 36047 | 0.2560 | 0.2590 |

llm_in_sample, by history length at the forecast origin

| comparison | mean dQLIKE (a - b) | rel. to b | NW t | dates | obs | loss a | loss b |
|---|---|---|---|---|---|---|---|
| <180d / p_har-har / h1 | n/a | n/a | n/a | 0 | 0 | n/a | n/a |
| <180d / p_loghar-loghar / h1 | n/a | n/a | n/a | 0 | 0 | n/a | n/a |
| <180d / p_loghar-robust_bc / h1 | -0.0266 | -6.1% | -0.81 | 1134 | 2944 | 0.4114 | 0.4380 |
| <180d / p_har-ewma / h1 | 0.0043 | 1.0% | 0.18 | 1134 | 2944 | 0.4368 | 0.4325 |
| <180d / p_har-har / h5 | n/a | n/a | n/a | 0 | 0 | n/a | n/a |
| <180d / p_loghar-loghar / h5 | n/a | n/a | n/a | 0 | 0 | n/a | n/a |
| <180d / p_loghar-robust_bc / h5 | -0.1279 | -29.5% | -2.86 | 1104 | 2824 | 0.3056 | 0.4334 |
| <180d / p_har-ewma / h5 | 0.0243 | 9.2% | 1.00 | 1104 | 2824 | 0.2901 | 0.2657 |
| 180-364d / p_har-har / h1 | -0.0069 | -1.7% | -0.35 | 1449 | 4625 | 0.4034 | 0.4103 |
| 180-364d / p_loghar-loghar / h1 | 0.0094 | 2.8% | 1.38 | 1449 | 4625 | 0.3483 | 0.3389 |
| 180-364d / p_loghar-robust_bc / h1 | -0.0913 | -20.8% | -5.94 | 1449 | 4625 | 0.3483 | 0.4396 |
| 180-364d / p_har-ewma / h1 | -0.0229 | -5.4% | -0.86 | 1449 | 4625 | 0.4034 | 0.4263 |
| 180-364d / p_har-har / h5 | -0.0036 | -1.2% | -0.20 | 1449 | 4625 | 0.2913 | 0.2949 |
| 180-364d / p_loghar-loghar / h5 | -0.0179 | -7.6% | -2.33 | 1449 | 4625 | 0.2174 | 0.2353 |
| 180-364d / p_loghar-robust_bc / h5 | -0.1885 | -46.4% | -7.47 | 1449 | 4625 | 0.2174 | 0.4059 |
| 180-364d / p_har-ewma / h5 | 0.0065 | 2.3% | 0.19 | 1449 | 4625 | 0.2913 | 0.2849 |
| 365-729d / p_har-har / h1 | -0.0069 | -1.9% | -2.78 | 1738 | 8412 | 0.3610 | 0.3679 |
| 365-729d / p_loghar-loghar / h1 | 0.0064 | 2.0% | 2.85 | 1738 | 8412 | 0.3345 | 0.3281 |
| 365-729d / p_loghar-robust_bc / h1 | -0.0711 | -17.5% | -6.42 | 1738 | 8412 | 0.3345 | 0.4056 |
| 365-729d / p_har-ewma / h1 | -0.0389 | -9.7% | -2.51 | 1738 | 8412 | 0.3610 | 0.3999 |
| 365-729d / p_har-har / h5 | -0.0111 | -4.5% | -3.15 | 1738 | 8412 | 0.2381 | 0.2492 |
| 365-729d / p_loghar-loghar / h5 | 0.0032 | 1.5% | 1.30 | 1738 | 8412 | 0.2121 | 0.2089 |
| 365-729d / p_loghar-robust_bc / h5 | -0.1722 | -44.8% | -8.24 | 1738 | 8412 | 0.2121 | 0.3843 |
| 365-729d / p_har-ewma / h5 | -0.0290 | -10.9% | -1.51 | 1738 | 8412 | 0.2381 | 0.2671 |
| >=730d / p_har-har / h1 | 0.0040 | 0.9% | 1.61 | 1543 | 23010 | 0.4516 | 0.4476 |
| >=730d / p_loghar-loghar / h1 | 0.0030 | 0.7% | 0.88 | 1543 | 23010 | 0.4133 | 0.4103 |
| >=730d / p_loghar-robust_bc / h1 | -0.0962 | -18.9% | -7.01 | 1543 | 23010 | 0.4133 | 0.5095 |
| >=730d / p_har-ewma / h1 | -0.0594 | -11.6% | -2.62 | 1543 | 23010 | 0.4516 | 0.5109 |
| >=730d / p_har-har / h5 | 0.0068 | 2.2% | 1.95 | 1543 | 23010 | 0.3153 | 0.3085 |
| >=730d / p_loghar-loghar / h5 | 0.0024 | 0.9% | 0.86 | 1543 | 23010 | 0.2744 | 0.2720 |
| >=730d / p_loghar-robust_bc / h5 | -0.2536 | -48.0% | -6.51 | 1543 | 23010 | 0.2744 | 0.5280 |
| >=730d / p_har-ewma / h5 | -0.0541 | -14.7% | -1.90 | 1543 | 23010 | 0.3153 | 0.3694 |

llm_in_sample, loss = MSE of log variance

| comparison | mean dQLIKE (a - b) | rel. to b | NW t | dates | obs | loss a | loss b |
|---|---|---|---|---|---|---|---|
| LT / p_har-har / h1 | -0.0176 | -1.7% | -4.91 | 2093 | 36047 | 1.0218 | 1.0394 |
| LT / p_har-har / h5 | -0.0234 | -3.4% | -5.68 | 2093 | 36047 | 0.6727 | 0.6960 |
| LT / p_loghar-loghar / h1 | -0.0078 | -1.1% | -4.77 | 2093 | 36047 | 0.7237 | 0.7315 |
| LT / p_loghar-loghar / h5 | -0.0123 | -2.5% | -4.04 | 2093 | 36047 | 0.4746 | 0.4868 |

clean_holdout

| comparison | mean dQLIKE (a - b) | rel. to b | NW t | dates | obs | loss a | loss b |
|---|---|---|---|---|---|---|---|
| L / p_har-har / h1 | -0.0030 | -0.7% | -0.76 | 131 | 3013 | 0.4329 | 0.4359 |
| L / p_har-har / h5 | -0.0043 | -1.4% | -1.26 | 127 | 2921 | 0.3056 | 0.3099 |
| L / p_loghar-loghar / h1 | -0.0027 | -0.7% | -0.40 | 131 | 3013 | 0.4138 | 0.4165 |
| L / p_loghar-loghar / h5 | -0.0039 | -1.2% | -0.37 | 127 | 2921 | 0.3099 | 0.3138 |
| T / p_har-har / h1 | -0.0029 | -0.4% | -0.74 | 131 | 346 | 0.6680 | 0.6709 |
| T / p_har-har / h5 | 0.0016 | 1.4% | 0.54 | 127 | 326 | 0.1099 | 0.1084 |
| T / p_loghar-loghar / h1 | -0.0136 | -2.1% | -0.44 | 131 | 346 | 0.6379 | 0.6515 |
| T / p_loghar-loghar / h5 | 0.0759 | 89.7% | 6.13 | 127 | 326 | 0.1605 | 0.0846 |
| LT / p_har-har / h1 | -0.0032 | -0.7% | -0.85 | 131 | 3359 | 0.4541 | 0.4572 |
| LT / p_har-har / h5 | -0.0036 | -1.2% | -1.11 | 127 | 3247 | 0.2879 | 0.2915 |
| LT / p_loghar-loghar / h1 | -0.0052 | -1.2% | -0.75 | 131 | 3359 | 0.4303 | 0.4355 |
| LT / p_loghar-loghar / h5 | 0.0049 | 1.7% | 0.54 | 127 | 3247 | 0.2937 | 0.2888 |

clean_holdout, by history length at the forecast origin

| comparison | mean dQLIKE (a - b) | rel. to b | NW t | dates | obs | loss a | loss b |
|---|---|---|---|---|---|---|---|
| <180d / p_har-har / h1 | n/a | n/a | n/a | 0 | 0 | n/a | n/a |
| <180d / p_loghar-loghar / h1 | n/a | n/a | n/a | 0 | 0 | n/a | n/a |
| <180d / p_loghar-robust_bc / h1 | -0.2598 | -27.5% | -5.20 | 131 | 1212 | 0.6844 | 0.9442 |
| <180d / p_har-ewma / h1 | -0.0795 | -10.6% | -2.54 | 131 | 1212 | 0.6725 | 0.7520 |
| <180d / p_har-har / h5 | n/a | n/a | n/a | 0 | 0 | n/a | n/a |
| <180d / p_loghar-loghar / h5 | n/a | n/a | n/a | 0 | 0 | n/a | n/a |
| <180d / p_loghar-robust_bc / h5 | -0.7390 | -74.0% | -5.54 | 127 | 1172 | 0.2595 | 0.9985 |
| <180d / p_har-ewma / h5 | -0.0149 | -5.9% | -1.07 | 127 | 1172 | 0.2390 | 0.2539 |
| 180-364d / p_har-har / h1 | -0.0161 | -2.8% | -1.81 | 33 | 84 | 0.5691 | 0.5852 |
| 180-364d / p_loghar-loghar / h1 | -0.1961 | -26.4% | -2.20 | 33 | 84 | 0.5457 | 0.7418 |
| 180-364d / p_loghar-robust_bc / h1 | -0.2691 | -33.0% | -2.41 | 33 | 84 | 0.5457 | 0.8148 |
| 180-364d / p_har-ewma / h1 | -0.0542 | -8.7% | -1.78 | 33 | 84 | 0.5691 | 0.6233 |
| 180-364d / p_har-har / h5 | 0.0194 | 19.9% | 1.90 | 29 | 72 | 0.1171 | 0.0977 |
| 180-364d / p_loghar-loghar / h5 | 0.0702 | 80.6% | 2.98 | 29 | 72 | 0.1573 | 0.0871 |
| 180-364d / p_loghar-robust_bc / h5 | -0.7331 | -82.3% | -2.44 | 29 | 72 | 0.1573 | 0.8904 |
| 180-364d / p_har-ewma / h5 | -0.0232 | -16.5% | -0.67 | 29 | 72 | 0.1171 | 0.1403 |
| 365-729d / p_har-har / h1 | 0.0062 | 1.1% | 0.61 | 131 | 393 | 0.5796 | 0.5734 |
| 365-729d / p_loghar-loghar / h1 | 0.0045 | 0.8% | 0.25 | 131 | 393 | 0.5511 | 0.5466 |
| 365-729d / p_loghar-robust_bc / h1 | -0.2603 | -32.1% | -4.11 | 131 | 393 | 0.5511 | 0.8114 |
| 365-729d / p_har-ewma / h1 | -0.0493 | -7.8% | -1.63 | 131 | 393 | 0.5796 | 0.6289 |
| 365-729d / p_har-har / h5 | 0.0097 | 6.1% | 1.13 | 127 | 381 | 0.1686 | 0.1590 |
| 365-729d / p_loghar-loghar / h5 | 0.0479 | 32.9% | 4.40 | 127 | 381 | 0.1934 | 0.1455 |
| 365-729d / p_loghar-robust_bc / h5 | -0.5958 | -75.5% | -5.44 | 127 | 381 | 0.1934 | 0.7893 |
| 365-729d / p_har-ewma / h5 | -0.0106 | -5.9% | -0.30 | 127 | 381 | 0.1686 | 0.1793 |
| >=730d / p_har-har / h1 | -0.0040 | -0.9% | -1.26 | 131 | 2882 | 0.4341 | 0.4382 |
| >=730d / p_loghar-loghar / h1 | -0.0027 | -0.7% | -0.40 | 131 | 2882 | 0.4149 | 0.4177 |
| >=730d / p_loghar-robust_bc / h1 | -0.0927 | -18.3% | -2.51 | 131 | 2882 | 0.4149 | 0.5076 |
| >=730d / p_har-ewma / h1 | -0.0727 | -14.3% | -0.58 | 131 | 2882 | 0.4341 | 0.5068 |
| >=730d / p_har-har / h5 | -0.0059 | -1.9% | -2.18 | 127 | 2794 | 0.3066 | 0.3124 |
| >=730d / p_loghar-loghar / h5 | -0.0038 | -1.2% | -0.33 | 127 | 2794 | 0.3120 | 0.3158 |
| >=730d / p_loghar-robust_bc / h5 | -0.3988 | -56.1% | -1.97 | 127 | 2794 | 0.3120 | 0.7108 |
| >=730d / p_har-ewma / h5 | -0.1481 | -32.6% | -0.87 | 127 | 2794 | 0.3066 | 0.4547 |

clean_holdout, loss = MSE of log variance

| comparison | mean dQLIKE (a - b) | rel. to b | NW t | dates | obs | loss a | loss b |
|---|---|---|---|---|---|---|---|
| LT / p_har-har / h1 | 0.0061 | 0.4% | 1.34 | 131 | 3359 | 1.3531 | 1.3471 |
| LT / p_har-har / h5 | -0.0091 | -1.2% | -1.93 | 127 | 3247 | 0.7546 | 0.7637 |
| LT / p_loghar-loghar / h1 | -0.0455 | -4.6% | -3.46 | 131 | 3359 | 0.9493 | 0.9947 |
| LT / p_loghar-loghar / h5 | 0.0180 | 3.7% | 4.74 | 127 | 3247 | 0.5063 | 0.4883 |

</details>

### H-VM2

**llm_in_sample: supports** (Holm p 0.002)

| comparison | mean dQLIKE (a - b) | rel. to b | NW t | dates | obs | loss a | loss b |
|---|---|---|---|---|---|---|---|
| harq-har | -0.0218 | -5.1% | -6.80 | 2093 | 36047 | 0.4043 | 0.4260 |
| loghar-har | -0.0506 | -11.9% | -4.99 | 2093 | 36047 | 0.3755 | 0.4260 |
| p_harq-p_har | -0.0193 | -4.2% | -7.86 | 2213 | 38991 | 0.4346 | 0.4538 |
| p_loghar-p_har | -0.0389 | -8.6% | -3.34 | 2213 | 38991 | 0.4149 | 0.4538 |

**clean_holdout: inconclusive** (Holm p 0.673)

| comparison | mean dQLIKE (a - b) | rel. to b | NW t | dates | obs | loss a | loss b |
|---|---|---|---|---|---|---|---|
| harq-har | -0.0213 | -4.7% | -3.31 | 131 | 3359 | 0.4359 | 0.4572 |
| loghar-har | -0.0217 | -4.7% | -0.42 | 131 | 3359 | 0.4355 | 0.4572 |
| p_harq-p_har | -0.0190 | -3.7% | -4.06 | 131 | 4571 | 0.4926 | 0.5117 |
| p_loghar-p_har | -0.0191 | -3.7% | -0.49 | 131 | 4571 | 0.4926 | 0.5117 |

<details><summary>secondary comparisons</summary>

llm_in_sample

| comparison | mean dQLIKE (a - b) | rel. to b | NW t | dates | obs | loss a | loss b |
|---|---|---|---|---|---|---|---|
| L / harq-har / h1 | -0.0230 | -5.4% | -7.33 | 2093 | 35471 | 0.3990 | 0.4219 |
| L / harq-har / h5 | -0.0124 | -4.0% | -6.45 | 2093 | 35471 | 0.2932 | 0.3056 |
| L / loghar-har / h1 | -0.0496 | -11.7% | -4.88 | 2093 | 35471 | 0.3724 | 0.4219 |
| L / loghar-har / h5 | -0.0477 | -15.6% | -5.49 | 2093 | 35471 | 0.2579 | 0.3056 |
| L / p_harq-p_har / h1 | -0.0190 | -4.2% | -7.87 | 2213 | 38149 | 0.4300 | 0.4490 |
| L / p_harq-p_har / h5 | -0.0102 | -3.2% | -5.81 | 2213 | 38029 | 0.3112 | 0.3214 |
| L / p_loghar-p_har / h1 | -0.0386 | -8.6% | -3.32 | 2213 | 38149 | 0.4104 | 0.4490 |
| L / p_loghar-p_har / h5 | -0.0156 | -4.9% | -0.81 | 2213 | 38029 | 0.3058 | 0.3214 |
| T / harq-har / h1 | 0.0744 | 10.6% | 1.39 | 334 | 576 | 0.7779 | 0.7035 |
| T / harq-har / h5 | 0.0193 | 4.8% | 0.85 | 334 | 576 | 0.4224 | 0.4031 |
| T / loghar-har / h1 | -0.1172 | -16.7% | -2.09 | 334 | 576 | 0.5863 | 0.7035 |
| T / loghar-har / h5 | -0.1143 | -28.4% | -1.62 | 334 | 576 | 0.2888 | 0.4031 |
| T / p_harq-p_har / h1 | -0.0332 | -5.1% | -1.00 | 443 | 842 | 0.6155 | 0.6487 |
| T / p_harq-p_har / h5 | -0.0060 | -1.7% | -0.35 | 443 | 842 | 0.3548 | 0.3608 |
| T / p_loghar-p_har / h1 | -0.0575 | -8.9% | -0.80 | 443 | 842 | 0.5912 | 0.6487 |
| T / p_loghar-p_har / h5 | -0.1042 | -28.9% | -1.29 | 443 | 842 | 0.2566 | 0.3608 |

llm_in_sample, loss = MSE of log variance

| comparison | mean dQLIKE (a - b) | rel. to b | NW t | dates | obs | loss a | loss b |
|---|---|---|---|---|---|---|---|
| harq-har | -0.1018 | -9.8% | -10.91 | 2093 | 36047 | 0.9376 | 1.0394 |
| loghar-har | -0.3079 | -29.6% | -11.73 | 2093 | 36047 | 0.7315 | 1.0394 |
| p_harq-p_har | -0.0758 | -6.4% | -11.11 | 2213 | 38991 | 1.1001 | 1.1759 |
| p_loghar-p_har | -0.2675 | -22.7% | -5.72 | 2213 | 38991 | 0.9084 | 1.1759 |

clean_holdout

| comparison | mean dQLIKE (a - b) | rel. to b | NW t | dates | obs | loss a | loss b |
|---|---|---|---|---|---|---|---|
| L / harq-har / h1 | -0.0254 | -5.8% | -3.41 | 131 | 3013 | 0.4106 | 0.4359 |
| L / harq-har / h5 | -0.0143 | -4.6% | -2.83 | 127 | 2921 | 0.2956 | 0.3099 |
| L / loghar-har / h1 | -0.0194 | -4.5% | -0.33 | 131 | 3013 | 0.4165 | 0.4359 |
| L / loghar-har / h5 | 0.0039 | 1.3% | 0.06 | 127 | 2921 | 0.3138 | 0.3099 |
| L / p_harq-p_har / h1 | -0.0214 | -5.0% | -4.12 | 131 | 3013 | 0.4115 | 0.4329 |
| L / p_harq-p_har / h5 | -0.0102 | -3.4% | -2.96 | 127 | 2921 | 0.2953 | 0.3056 |
| L / p_loghar-p_har / h1 | -0.0191 | -4.4% | -0.34 | 131 | 3013 | 0.4138 | 0.4329 |
| L / p_loghar-p_har / h5 | 0.0043 | 1.4% | 0.06 | 127 | 2921 | 0.3099 | 0.3056 |
| T / harq-har / h1 | 0.0255 | 3.8% | 3.25 | 131 | 346 | 0.6964 | 0.6709 |
| T / harq-har / h5 | -0.0085 | -7.9% | -4.27 | 127 | 326 | 0.0998 | 0.1084 |
| T / loghar-har / h1 | -0.0195 | -2.9% | -1.16 | 131 | 346 | 0.6515 | 0.6709 |
| T / loghar-har / h5 | -0.0238 | -21.9% | -2.02 | 127 | 326 | 0.0846 | 0.1084 |
| T / p_harq-p_har / h1 | -0.0108 | -1.6% | -1.88 | 131 | 1558 | 0.6525 | 0.6633 |
| T / p_harq-p_har / h5 | 0.0089 | 4.4% | 2.31 | 127 | 1498 | 0.2126 | 0.2037 |
| T / p_loghar-p_har / h1 | -0.0024 | -0.4% | -0.10 | 131 | 1558 | 0.6608 | 0.6633 |
| T / p_loghar-p_har / h5 | 0.0277 | 13.6% | 3.05 | 127 | 1498 | 0.2314 | 0.2037 |

clean_holdout, loss = MSE of log variance

| comparison | mean dQLIKE (a - b) | rel. to b | NW t | dates | obs | loss a | loss b |
|---|---|---|---|---|---|---|---|
| harq-har | -0.1024 | -7.6% | -5.73 | 131 | 3359 | 1.2446 | 1.3471 |
| loghar-har | -0.3523 | -26.2% | -5.29 | 131 | 3359 | 0.9947 | 1.3471 |
| p_harq-p_har | -0.0966 | -6.0% | -6.27 | 131 | 4571 | 1.5095 | 1.6061 |
| p_loghar-p_har | -0.4333 | -27.0% | -6.86 | 131 | 4571 | 1.1729 | 1.6061 |

</details>

### H-VM3

**llm_in_sample: contradicts** (Holm p 1.000)

| comparison | mean dQLIKE (a - b) | rel. to b | NW t | dates | obs | loss a | loss b |
|---|---|---|---|---|---|---|---|
| har_gk-har / h1 | 0.0231 | 5.4% | 4.87 | 2093 | 36047 | 0.4491 | 0.4260 |
| har_gk-har / h5 | 0.0203 | 6.6% | 5.01 | 2093 | 36047 | 0.3279 | 0.3076 |
| har_rvgk-har / h1 | 0.0249 | 5.9% | 2.68 | 2093 | 36047 | 0.4510 | 0.4260 |
| har_rvgk-har / h5 | 0.0371 | 12.1% | 2.86 | 2093 | 36047 | 0.3447 | 0.3076 |
| har_rvgk-har_gk / h1 | 0.0019 | 0.4% | 0.20 | 2093 | 36047 | 0.4510 | 0.4491 |
| har_rvgk-har_gk / h5 | 0.0169 | 5.1% | 1.26 | 2093 | 36047 | 0.3447 | 0.3279 |

**clean_holdout: contradicts** (Holm p 1.000)

| comparison | mean dQLIKE (a - b) | rel. to b | NW t | dates | obs | loss a | loss b |
|---|---|---|---|---|---|---|---|
| har_gk-har / h1 | 0.0720 | 15.7% | 4.80 | 131 | 3359 | 0.5292 | 0.4572 |
| har_gk-har / h5 | 0.0366 | 12.6% | 2.35 | 127 | 3247 | 0.3281 | 0.2915 |
| har_rvgk-har / h1 | 0.0002 | 0.0% | 0.17 | 131 | 3359 | 0.4574 | 0.4572 |
| har_rvgk-har / h5 | 0.0010 | 0.3% | 0.89 | 127 | 3247 | 0.2924 | 0.2915 |
| har_rvgk-har_gk / h1 | -0.0718 | -13.6% | -4.63 | 131 | 3359 | 0.4574 | 0.5292 |
| har_rvgk-har_gk / h5 | -0.0356 | -10.9% | -2.19 | 127 | 3247 | 0.2924 | 0.3281 |

<details><summary>secondary comparisons</summary>

llm_in_sample

| comparison | mean dQLIKE (a - b) | rel. to b | NW t | dates | obs | loss a | loss b |
|---|---|---|---|---|---|---|---|
| L / har_pk-har / h1 | 0.0163 | 3.9% | 4.13 | 2093 | 35471 | 0.4382 | 0.4219 |
| L / har_pk-har / h5 | 0.0140 | 4.6% | 4.29 | 2093 | 35471 | 0.3195 | 0.3056 |
| L / p_har_gk-p_har / h1 | 0.0278 | 6.2% | 5.84 | 2213 | 38149 | 0.4768 | 0.4490 |
| L / p_har_gk-p_har / h5 | 0.0212 | 6.6% | 6.60 | 2213 | 38029 | 0.3426 | 0.3214 |
| L / p_har_rvgk-p_har / h1 | 0.0279 | 6.2% | 2.91 | 2213 | 38149 | 0.4769 | 0.4490 |
| L / p_har_rvgk-p_har / h5 | 0.0136 | 4.2% | 1.56 | 2213 | 38029 | 0.3350 | 0.3214 |
| L / har_gk-har / h1 | 0.0240 | 5.7% | 5.08 | 2093 | 35471 | 0.4459 | 0.4219 |
| L / har_gk-har / h5 | 0.0188 | 6.1% | 5.10 | 2093 | 35471 | 0.3243 | 0.3056 |
| T / har_pk-har / h1 | -0.0083 | -1.2% | -0.20 | 334 | 576 | 0.6952 | 0.7035 |
| T / har_pk-har / h5 | 0.1296 | 32.1% | 1.94 | 334 | 576 | 0.5326 | 0.4031 |
| T / p_har_gk-p_har / h1 | 0.0274 | 4.2% | 0.87 | 443 | 842 | 0.6761 | 0.6487 |
| T / p_har_gk-p_har / h5 | 0.0043 | 1.2% | 0.08 | 443 | 842 | 0.3651 | 0.3608 |
| T / p_har_rvgk-p_har / h1 | -0.0044 | -0.7% | -0.91 | 443 | 842 | 0.6444 | 0.6487 |
| T / p_har_rvgk-p_har / h5 | 0.0131 | 3.6% | 1.22 | 443 | 842 | 0.3739 | 0.3608 |
| T / har_gk-har / h1 | 0.0236 | 3.4% | 0.50 | 334 | 576 | 0.7271 | 0.7035 |
| T / har_gk-har / h5 | 0.2336 | 58.0% | 2.09 | 334 | 576 | 0.6367 | 0.4031 |
| LT / har_pk-har / h1 | 0.0153 | 3.6% | 3.88 | 2093 | 36047 | 0.4413 | 0.4260 |
| LT / har_pk-har / h5 | 0.0149 | 4.8% | 4.39 | 2093 | 36047 | 0.3224 | 0.3076 |
| LT / p_har_gk-p_har / h1 | 0.0270 | 6.0% | 5.82 | 2213 | 38991 | 0.4808 | 0.4538 |
| LT / p_har_gk-p_har / h5 | 0.0205 | 6.3% | 6.30 | 2213 | 38871 | 0.3438 | 0.3233 |
| LT / p_har_rvgk-p_har / h1 | 0.0269 | 5.9% | 2.92 | 2213 | 38991 | 0.4807 | 0.4538 |
| LT / p_har_rvgk-p_har / h5 | 0.0129 | 4.0% | 1.56 | 2213 | 38871 | 0.3363 | 0.3233 |
| LT / har_gk-har / h1 | 0.0231 | 5.4% | 4.87 | 2093 | 36047 | 0.4491 | 0.4260 |
| LT / har_gk-har / h5 | 0.0203 | 6.6% | 5.01 | 2093 | 36047 | 0.3279 | 0.3076 |

clean_holdout

| comparison | mean dQLIKE (a - b) | rel. to b | NW t | dates | obs | loss a | loss b |
|---|---|---|---|---|---|---|---|
| L / har_pk-har / h1 | 0.0602 | 13.8% | 4.70 | 131 | 3013 | 0.4962 | 0.4359 |
| L / har_pk-har / h5 | 0.0304 | 9.8% | 2.27 | 127 | 2921 | 0.3404 | 0.3099 |
| L / p_har_gk-p_har / h1 | 0.1117 | 25.8% | 5.58 | 131 | 3013 | 0.5446 | 0.4329 |
| L / p_har_gk-p_har / h5 | 0.0635 | 20.8% | 3.68 | 127 | 2921 | 0.3691 | 0.3056 |
| L / p_har_rvgk-p_har / h1 | 0.0005 | 0.1% | 1.03 | 131 | 3013 | 0.4334 | 0.4329 |
| L / p_har_rvgk-p_har / h5 | 0.0005 | 0.2% | 1.14 | 127 | 2921 | 0.3060 | 0.3056 |
| L / har_gk-har / h1 | 0.0789 | 18.1% | 4.83 | 131 | 3013 | 0.5148 | 0.4359 |
| L / har_gk-har / h5 | 0.0415 | 13.4% | 2.46 | 127 | 2921 | 0.3514 | 0.3099 |
| T / har_pk-har / h1 | -0.0043 | -0.6% | -0.77 | 131 | 346 | 0.6666 | 0.6709 |
| T / har_pk-har / h5 | -0.0153 | -14.1% | -4.06 | 127 | 326 | 0.0931 | 0.1084 |
| T / p_har_gk-p_har / h1 | 0.0590 | 8.9% | 2.81 | 131 | 1558 | 0.7223 | 0.6633 |
| T / p_har_gk-p_har / h5 | 0.0094 | 4.6% | 0.71 | 127 | 1498 | 0.2131 | 0.2037 |
| T / p_har_rvgk-p_har / h1 | 0.0004 | 0.1% | 0.41 | 131 | 1558 | 0.6637 | 0.6633 |
| T / p_har_rvgk-p_har / h5 | 0.0003 | 0.1% | 0.47 | 127 | 1498 | 0.2040 | 0.2037 |
| T / har_gk-har / h1 | -0.0020 | -0.3% | -0.28 | 131 | 346 | 0.6689 | 0.6709 |
| T / har_gk-har / h5 | -0.0180 | -16.6% | -3.66 | 127 | 326 | 0.0904 | 0.1084 |
| LT / har_pk-har / h1 | 0.0548 | 12.0% | 4.64 | 131 | 3359 | 0.5120 | 0.4572 |
| LT / har_pk-har / h5 | 0.0267 | 9.1% | 2.16 | 127 | 3247 | 0.3181 | 0.2915 |
| LT / p_har_gk-p_har / h1 | 0.1007 | 19.7% | 5.50 | 131 | 4571 | 0.6124 | 0.5117 |
| LT / p_har_gk-p_har / h5 | 0.0515 | 18.1% | 3.62 | 127 | 4419 | 0.3354 | 0.2839 |
| LT / p_har_rvgk-p_har / h1 | 0.0005 | 0.1% | 0.84 | 131 | 4571 | 0.5122 | 0.5117 |
| LT / p_har_rvgk-p_har / h5 | 0.0004 | 0.2% | 1.02 | 127 | 4419 | 0.2844 | 0.2839 |
| LT / har_gk-har / h1 | 0.0720 | 15.7% | 4.80 | 131 | 3359 | 0.5292 | 0.4572 |
| LT / har_gk-har / h5 | 0.0366 | 12.6% | 2.35 | 127 | 3247 | 0.3281 | 0.2915 |

</details>

### H-VM4

| segment | verdict | median R^2 (inclusive) | median R^2 (leave-one-out) | min R^2 | alts |
|---|---|---|---|---|---|
| llm_in_sample | inconclusive | 0.723 | 0.695 | 0.429 | 22 |
| clean_holdout | inconclusive | 0.678 | 0.646 | 0.192 | 22 |

llm_in_sample (Holm p 0.563)

| comparison | mean dQLIKE (a - b) | rel. to b | NW t | dates | obs | loss a | loss b |
|---|---|---|---|---|---|---|---|
| alts h1: p_loghar_f - p_loghar | -0.0004 | -0.1% | -0.58 | 1853 | 35546 | 0.3769 | 0.3773 |
| L / h1 | -0.0005 | -0.1% | -0.75 | 1853 | 37399 | 0.3784 | 0.3789 |
| L / h5 | 0.0109 | 4.2% | 2.44 | 1853 | 37279 | 0.2732 | 0.2623 |
| T / h1 | -0.0043 | -0.7% | -0.39 | 443 | 842 | 0.5869 | 0.5912 |
| T / h5 | 0.0359 | 14.0% | 1.66 | 443 | 842 | 0.2925 | 0.2566 |
| alts / h5 | 0.0115 | 4.4% | 2.44 | 1853 | 35426 | 0.2767 | 0.2652 |

clean_holdout (Holm p 0.200)

| comparison | mean dQLIKE (a - b) | rel. to b | NW t | dates | obs | loss a | loss b |
|---|---|---|---|---|---|---|---|
| alts h1: p_loghar_f - p_loghar | -0.0040 | -1.0% | -1.64 | 131 | 2882 | 0.4065 | 0.4105 |
| L / h1 | -0.0038 | -0.9% | -1.67 | 131 | 3013 | 0.4100 | 0.4138 |
| L / h5 | -0.0013 | -0.4% | -0.49 | 127 | 2921 | 0.3086 | 0.3099 |
| T / h1 | -0.0025 | -0.4% | -0.33 | 131 | 1558 | 0.6584 | 0.6608 |
| T / h5 | -0.0205 | -8.8% | -1.45 | 127 | 1498 | 0.2110 | 0.2314 |
| alts / h5 | -0.0012 | -0.4% | -0.46 | 127 | 2794 | 0.3107 | 0.3119 |

### H-VM5

| universe, horizon | in-sample winner | in holdout MCS (Tmax) | holdout MCS p | holdout rank | in holdout MCS (TR) | holdout best | holdout MCS |
|---|---|---|---|---|---|---|---|
| L h1 | loghar | True | 0.167 | 6 | True | p_loghar_f | p_loghar_f, harq, p_harq, p_loghar, sv_rv, loghar, p_harj, p_shar, p_har, p_har_rvgk, shar, harj, egarch, har, har_rvgk, gjr, garch, har_pk, ewma, robust_bc, har_gk, robust_bc_c |
| L h5 | p_loghar | True | 0.293 | 14 | True | p_harq | p_harq, harq, gjr, p_harj, egarch, p_har, p_shar, p_har_rvgk, shar, p_loghar_f, harj, garch, har, p_loghar, har_rvgk, loghar, sv_rv, har_pk, sv_r2, har_gk, p_har_gk, sv_r2_cal, ewma, robust_bc, robust_bc_c |
| T h1 | loghar | True | 0.150 | 3 | None | p_loghar_f | p_loghar_f, p_loghar, loghar, p_harq, p_harj, har_pk, p_har, p_shar, p_har_rvgk, har_gk, har, shar, har_rvgk, harj, sv_r2, harq, sv_rv, p_har_gk, ewma, robust_bc_c |
| T h5 | p_loghar_f | True | 0.178 | 16 | None | sv_r2 | sv_r2, loghar, har_gk, har_pk, harq, p_har_gk, har, harj, p_harj, shar, p_shar, p_har, p_har_rvgk, ewma, p_loghar_f |

## Leaderboards (QLIKE, date-averaged, common sample of all models)

`*` = extension beyond the registered list.

### L, h = 1d, llm_in_sample

common sample: 1853 dates, 34931 symbol-days, 23 symbols; ratio vs `har`; MCS Tmax 90%

| rank | model | mean loss | ratio | in MCS | MCS p |
|---|---|---|---|---|---|
| 1 | loghar | 0.3728 | 0.879 | **yes** | 0.387 |
| 2 | p_loghar_f | 0.3752 | 0.884 | **yes** | 0.387 |
| 3 | p_loghar | 0.3755 | 0.885 | **yes** | 0.387 |
| 4 | sv_rv * | 0.3779 | 0.891 | **yes** | 0.387 |
| 5 | p_harq | 0.3959 | 0.933 | no | 0.053 |
| 6 | harq | 0.3996 | 0.942 | no | 0.018 |
| 7 | p_shar | 0.4160 | 0.980 | no | 0.008 |
| 8 | p_harj | 0.4167 | 0.982 | no | 0.008 |
| 9 | p_har | 0.4175 | 0.984 | no | 0.008 |
| 10 | shar | 0.4238 | 0.999 | no | 0.008 |
| 11 | harj | 0.4239 | 0.999 | no | 0.008 |
| 12 | har | 0.4243 | 1.000 | no | 0.008 |
| 13 | p_har_rvgk | 0.4344 | 1.024 | no | 0.008 |
| 14 | garch | 0.4365 | 1.029 | no | 0.008 |
| 15 | gjr | 0.4411 | 1.040 | no | 0.005 |
| 16 | har_rvgk * | 0.4455 | 1.050 | no | 0.008 |
| 17 | har_pk * | 0.4466 | 1.053 | no | 0.004 |
| 18 | robust_bc | 0.4523 | 1.066 | no | 0.008 |
| 19 | egarch | 0.4541 | 1.070 | no | 0.000 |
| 20 | har_gk | 0.4559 | 1.074 | no | 0.000 |
| 21 | p_har_gk | 0.4564 | 1.076 | no | 0.000 |
| 22 | ewma | 0.4667 | 1.100 | no | 0.008 |
| 23 | robust_bc_c * | 0.4741 | 1.117 | no | 0.008 |
| 24 | sv_r2 | 0.5389 | 1.270 | no | 0.000 |
| 25 | sv_r2_cal * | 0.5529 | 1.303 | no | 0.000 |

### L, h = 1d, clean_holdout

common sample: 131 dates, 3013 symbol-days, 23 symbols; ratio vs `har`; MCS Tmax 90%

| rank | model | mean loss | ratio | in MCS | MCS p |
|---|---|---|---|---|---|
| 1 | p_loghar_f | 0.4100 | 0.940 | **yes** | 0.167 |
| 2 | harq | 0.4106 | 0.942 | **yes** | 0.167 |
| 3 | p_harq | 0.4115 | 0.944 | **yes** | 0.167 |
| 4 | p_loghar | 0.4138 | 0.949 | **yes** | 0.167 |
| 5 | sv_rv * | 0.4141 | 0.950 | **yes** | 0.167 |
| 6 | loghar | 0.4165 | 0.955 | **yes** | 0.167 |
| 7 | p_harj | 0.4311 | 0.989 | **yes** | 0.167 |
| 8 | p_shar | 0.4321 | 0.991 | **yes** | 0.167 |
| 9 | p_har | 0.4329 | 0.993 | **yes** | 0.167 |
| 10 | p_har_rvgk | 0.4334 | 0.994 | **yes** | 0.167 |
| 11 | shar | 0.4337 | 0.995 | **yes** | 0.167 |
| 12 | harj | 0.4352 | 0.998 | **yes** | 0.167 |
| 13 | egarch | 0.4357 | 0.999 | **yes** | 0.167 |
| 14 | har | 0.4359 | 1.000 | **yes** | 0.167 |
| 15 | har_rvgk * | 0.4363 | 1.001 | **yes** | 0.167 |
| 16 | gjr | 0.4414 | 1.013 | **yes** | 0.167 |
| 17 | garch | 0.4473 | 1.026 | **yes** | 0.167 |
| 18 | har_pk * | 0.4962 | 1.138 | **yes** | 0.167 |
| 19 | ewma | 0.5045 | 1.157 | **yes** | 0.167 |
| 20 | robust_bc | 0.5077 | 1.165 | **yes** | 0.167 |
| 21 | har_gk | 0.5148 | 1.181 | **yes** | 0.167 |
| 22 | robust_bc_c * | 0.5360 | 1.230 | **yes** | 0.167 |
| 23 | sv_r2 | 0.5393 | 1.237 | no | 0.051 |
| 24 | p_har_gk | 0.5446 | 1.249 | no | 0.072 |
| 25 | sv_r2_cal * | 0.5965 | 1.368 | no | 0.002 |

### L, h = 5d, llm_in_sample

common sample: 1853 dates, 34931 symbol-days, 23 symbols; ratio vs `har`; MCS Tmax 90%

| rank | model | mean loss | ratio | in MCS | MCS p |
|---|---|---|---|---|---|
| 1 | p_loghar | 0.2606 | 0.845 | **yes** | 0.950 |
| 2 | loghar | 0.2617 | 0.849 | **yes** | 0.950 |
| 3 | sv_rv * | 0.2618 | 0.849 | **yes** | 0.950 |
| 4 | p_loghar_f | 0.2719 | 0.882 | no | 0.079 |
| 5 | p_harq | 0.2913 | 0.945 | no | 0.029 |
| 6 | harq | 0.2946 | 0.955 | no | 0.029 |
| 7 | p_harj | 0.3002 | 0.974 | no | 0.029 |
| 8 | p_har | 0.3007 | 0.975 | no | 0.029 |
| 9 | p_shar | 0.3007 | 0.975 | no | 0.029 |
| 10 | shar | 0.3063 | 0.993 | no | 0.029 |
| 11 | harj | 0.3073 | 0.996 | no | 0.018 |
| 12 | har | 0.3083 | 1.000 | no | 0.014 |
| 13 | garch | 0.3089 | 1.002 | no | 0.009 |
| 14 | gjr | 0.3112 | 1.009 | no | 0.004 |
| 15 | p_har_rvgk | 0.3157 | 1.024 | no | 0.029 |
| 16 | har_pk * | 0.3238 | 1.050 | no | 0.002 |
| 17 | p_har_gk | 0.3243 | 1.052 | no | 0.002 |
| 18 | har_gk | 0.3295 | 1.069 | no | 0.001 |
| 19 | har_rvgk * | 0.3298 | 1.070 | no | 0.004 |
| 20 | ewma | 0.3441 | 1.116 | no | 0.029 |
| 21 | sv_r2 | 0.3519 | 1.141 | no | 0.002 |
| 22 | egarch | 0.3634 | 1.179 | no | 0.000 |
| 23 | sv_r2_cal * | 0.3661 | 1.187 | no | 0.000 |
| 24 | robust_bc | 0.4669 | 1.514 | no | 0.000 |
| 25 | robust_bc_c * | 0.5022 | 1.629 | no | 0.000 |

### L, h = 5d, clean_holdout

common sample: 127 dates, 2921 symbol-days, 23 symbols; ratio vs `har`; MCS Tmax 90%

| rank | model | mean loss | ratio | in MCS | MCS p |
|---|---|---|---|---|---|
| 1 | p_harq | 0.2953 | 0.953 | **yes** | 0.293 |
| 2 | harq | 0.2956 | 0.954 | **yes** | 0.293 |
| 3 | gjr | 0.3042 | 0.982 | **yes** | 0.293 |
| 4 | p_harj | 0.3045 | 0.983 | **yes** | 0.293 |
| 5 | egarch | 0.3052 | 0.985 | **yes** | 0.293 |
| 6 | p_har | 0.3056 | 0.986 | **yes** | 0.293 |
| 7 | p_shar | 0.3059 | 0.987 | **yes** | 0.293 |
| 8 | p_har_rvgk | 0.3060 | 0.988 | **yes** | 0.293 |
| 9 | shar | 0.3081 | 0.994 | **yes** | 0.293 |
| 10 | p_loghar_f | 0.3086 | 0.996 | **yes** | 0.293 |
| 11 | harj | 0.3089 | 0.997 | **yes** | 0.293 |
| 12 | garch | 0.3093 | 0.998 | **yes** | 0.293 |
| 13 | har | 0.3099 | 1.000 | **yes** | 0.293 |
| 14 | p_loghar | 0.3099 | 1.000 | **yes** | 0.293 |
| 15 | har_rvgk * | 0.3100 | 1.000 | **yes** | 0.293 |
| 16 | loghar | 0.3138 | 1.013 | **yes** | 0.293 |
| 17 | sv_rv * | 0.3182 | 1.027 | **yes** | 0.293 |
| 18 | har_pk * | 0.3404 | 1.098 | **yes** | 0.293 |
| 19 | sv_r2 | 0.3438 | 1.109 | **yes** | 0.293 |
| 20 | har_gk | 0.3514 | 1.134 | **yes** | 0.293 |
| 21 | p_har_gk | 0.3691 | 1.191 | **yes** | 0.293 |
| 22 | sv_r2_cal * | 0.4004 | 1.292 | **yes** | 0.293 |
| 23 | ewma | 0.4487 | 1.448 | **yes** | 0.293 |
| 24 | robust_bc | 0.7039 | 2.271 | **yes** | 0.293 |
| 25 | robust_bc_c * | 0.7518 | 2.426 | **yes** | 0.293 |

### T, h = 1d, llm_in_sample

common sample: 334 dates, 516 symbol-days, 2 symbols; ratio vs `har`; MCS Tmax 90%

| rank | model | mean loss | ratio | in MCS | MCS p |
|---|---|---|---|---|---|
| 1 | loghar | 0.5722 | 0.827 | **yes** | 0.313 |
| 2 | p_loghar_f | 0.6071 | 0.878 | **yes** | 0.313 |
| 3 | p_loghar | 0.6270 | 0.907 | **yes** | 0.313 |
| 4 | garch | 0.6351 | 0.918 | **yes** | 0.313 |
| 5 | shar | 0.6412 | 0.927 | **yes** | 0.313 |
| 6 | p_harq | 0.6685 | 0.967 | **yes** | 0.313 |
| 7 | sv_rv * | 0.6691 | 0.968 | **yes** | 0.313 |
| 8 | p_harj | 0.6704 | 0.969 | **yes** | 0.313 |
| 9 | ewma | 0.6733 | 0.973 | **yes** | 0.313 |
| 10 | har_pk * | 0.6781 | 0.980 | **yes** | 0.313 |
| 11 | har | 0.6916 | 1.000 | **yes** | 0.313 |
| 12 | p_har_rvgk | 0.6940 | 1.003 | **yes** | 0.313 |
| 13 | p_shar | 0.7093 | 1.026 | **yes** | 0.313 |
| 14 | p_har | 0.7102 | 1.027 | **yes** | 0.313 |
| 15 | har_gk | 0.7112 | 1.028 | **yes** | 0.313 |
| 16 | har_rvgk * | 0.7164 | 1.036 | **yes** | 0.313 |
| 17 | p_har_gk | 0.7234 | 1.046 | **yes** | 0.313 |
| 18 | egarch | 0.7237 | 1.046 | **yes** | 0.313 |
| 19 | gjr | 0.7474 | 1.081 | **yes** | 0.313 |
| 20 | harq | 0.7735 | 1.118 | **yes** | 0.313 |
| 21 | harj | 0.8197 | 1.185 | **yes** | 0.313 |
| 22 | sv_r2_cal * | 0.8296 | 1.200 | **yes** | 0.313 |
| 23 | robust_bc | 0.8915 | 1.289 | **yes** | 0.313 |
| 24 | robust_bc_c * | 0.9660 | 1.397 | **yes** | 0.313 |
| 25 | sv_r2 | 1.0136 | 1.466 | **yes** | 0.313 |

### T, h = 1d, clean_holdout

common sample: 131 dates, 346 symbol-days, 5 symbols; ratio vs `har`; MCS Tmax 90%

| rank | model | mean loss | ratio | in MCS | MCS p |
|---|---|---|---|---|---|
| 1 | p_loghar_f | 0.6279 | 0.936 | **yes** | 0.150 |
| 2 | p_loghar | 0.6379 | 0.951 | **yes** | 0.150 |
| 3 | loghar | 0.6515 | 0.971 | **yes** | 0.150 |
| 4 | p_harq | 0.6541 | 0.975 | **yes** | 0.150 |
| 5 | p_harj | 0.6653 | 0.992 | **yes** | 0.150 |
| 6 | har_pk * | 0.6666 | 0.994 | **yes** | 0.150 |
| 7 | p_har | 0.6680 | 0.996 | **yes** | 0.150 |
| 8 | p_shar | 0.6682 | 0.996 | **yes** | 0.150 |
| 9 | p_har_rvgk | 0.6687 | 0.997 | **yes** | 0.150 |
| 10 | har_gk | 0.6689 | 0.997 | **yes** | 0.150 |
| 11 | har | 0.6709 | 1.000 | **yes** | 0.150 |
| 12 | shar | 0.6724 | 1.002 | **yes** | 0.150 |
| 13 | har_rvgk * | 0.6726 | 1.003 | **yes** | 0.150 |
| 14 | harj | 0.6926 | 1.032 | **yes** | 0.150 |
| 15 | sv_r2 | 0.6956 | 1.037 | **yes** | 0.150 |
| 16 | harq | 0.6964 | 1.038 | **yes** | 0.150 |
| 17 | sv_rv * | 0.7034 | 1.048 | **yes** | 0.150 |
| 18 | p_har_gk | 0.7038 | 1.049 | **yes** | 0.150 |
| 19 | ewma | 0.7207 | 1.074 | **yes** | 0.150 |
| 20 | garch | 0.7578 | 1.130 | no | 0.021 |
| 21 | sv_r2_cal * | 0.7824 | 1.166 | no | 0.020 |
| 22 | robust_bc_c * | 0.7887 | 1.176 | **yes** | 0.150 |
| 23 | gjr | 0.7966 | 1.187 | no | 0.021 |
| 24 | egarch | 0.8169 | 1.218 | no | 0.003 |
| 25 | robust_bc | 0.9764 | 1.455 | no | 0.021 |

### T, h = 5d, llm_in_sample

common sample: 334 dates, 516 symbol-days, 2 symbols; ratio vs `har`; MCS Tmax 90%

| rank | model | mean loss | ratio | in MCS | MCS p |
|---|---|---|---|---|---|
| 1 | p_loghar_f | 0.2622 | 0.665 | **yes** | 0.193 |
| 2 | p_loghar | 0.2741 | 0.695 | **yes** | 0.193 |
| 3 | loghar | 0.2824 | 0.716 | **yes** | 0.193 |
| 4 | garch | 0.2962 | 0.751 | **yes** | 0.193 |
| 5 | egarch | 0.3282 | 0.832 | **yes** | 0.193 |
| 6 | ewma | 0.3319 | 0.842 | **yes** | 0.193 |
| 7 | gjr | 0.3646 | 0.924 | **yes** | 0.193 |
| 8 | sv_rv * | 0.3833 | 0.972 | **yes** | 0.193 |
| 9 | shar | 0.3897 | 0.988 | **yes** | 0.193 |
| 10 | har | 0.3944 | 1.000 | **yes** | 0.193 |
| 11 | p_harj | 0.3960 | 1.004 | **yes** | 0.193 |
| 12 | p_harq | 0.3982 | 1.010 | **yes** | 0.193 |
| 13 | harj | 0.4068 | 1.031 | **yes** | 0.193 |
| 14 | harq | 0.4106 | 1.041 | **yes** | 0.193 |
| 15 | p_har | 0.4127 | 1.046 | **yes** | 0.193 |
| 16 | p_shar | 0.4135 | 1.048 | **yes** | 0.193 |
| 17 | p_har_gk | 0.4233 | 1.073 | **yes** | 0.193 |
| 18 | p_har_rvgk | 0.4301 | 1.091 | **yes** | 0.193 |
| 19 | sv_r2_cal * | 0.4548 | 1.153 | **yes** | 0.193 |
| 20 | har_pk * | 0.5160 | 1.308 | **yes** | 0.193 |
| 21 | har_rvgk * | 0.5789 | 1.468 | **yes** | 0.193 |
| 22 | har_gk | 0.6162 | 1.562 | **yes** | 0.193 |
| 23 | sv_r2 | 0.6702 | 1.699 | **yes** | 0.193 |
| 24 | robust_bc | 0.7862 | 1.993 | no | 0.099 |
| 25 | robust_bc_c * | 0.9083 | 2.303 | no | 0.019 |

### T, h = 5d, clean_holdout

common sample: 127 dates, 326 symbol-days, 5 symbols; ratio vs `har`; MCS Tmax 90%

| rank | model | mean loss | ratio | in MCS | MCS p |
|---|---|---|---|---|---|
| 1 | sv_r2 | 0.0840 | 0.775 | **yes** | 0.178 |
| 2 | loghar | 0.0846 | 0.781 | **yes** | 0.178 |
| 3 | har_gk | 0.0904 | 0.834 | **yes** | 0.178 |
| 4 | har_pk * | 0.0931 | 0.859 | **yes** | 0.178 |
| 5 | harq | 0.0998 | 0.921 | **yes** | 0.178 |
| 6 | p_har_gk | 0.1051 | 0.970 | **yes** | 0.178 |
| 7 | har | 0.1084 | 1.000 | **yes** | 0.178 |
| 8 | harj | 0.1091 | 1.007 | **yes** | 0.178 |
| 9 | p_harj | 0.1092 | 1.008 | **yes** | 0.178 |
| 10 | shar | 0.1096 | 1.011 | **yes** | 0.178 |
| 11 | p_shar | 0.1097 | 1.013 | **yes** | 0.178 |
| 12 | p_har | 0.1099 | 1.014 | **yes** | 0.178 |
| 13 | p_har_rvgk | 0.1112 | 1.026 | **yes** | 0.178 |
| 14 | ewma | 0.1152 | 1.063 | **yes** | 0.178 |
| 15 | har_rvgk * | 0.1190 | 1.098 | no | 0.067 |
| 16 | p_loghar_f | 0.1194 | 1.102 | **yes** | 0.178 |
| 17 | p_harq | 0.1199 | 1.107 | no | 0.067 |
| 18 | garch | 0.1602 | 1.478 | no | 0.062 |
| 19 | p_loghar | 0.1605 | 1.481 | no | 0.047 |
| 20 | sv_rv * | 0.1744 | 1.609 | no | 0.014 |
| 21 | sv_r2_cal * | 0.1776 | 1.639 | no | 0.000 |
| 22 | gjr | 0.1858 | 1.714 | no | 0.067 |
| 23 | egarch | 0.2027 | 1.871 | no | 0.001 |
| 24 | robust_bc | 0.9277 | 8.561 | no | 0.000 |
| 25 | robust_bc_c * | 1.0250 | 9.458 | no | 0.000 |

## TradFi short-history set (models that need no 180-day own history), QLIKE

### T, h = 1d, llm_in_sample

common sample: 443 dates, 842 symbol-days, 4 symbols; ratio vs `robust_bc`; MCS Tmax 90%

| rank | model | mean loss | ratio | in MCS | MCS p |
|---|---|---|---|---|---|
| 1 | p_loghar_f | 0.5869 | 0.687 | **yes** | 0.113 |
| 2 | p_loghar | 0.5912 | 0.692 | **yes** | 0.113 |
| 3 | p_harq | 0.6155 | 0.721 | **yes** | 0.113 |
| 4 | ewma | 0.6164 | 0.722 | **yes** | 0.113 |
| 5 | p_harj | 0.6174 | 0.723 | **yes** | 0.113 |
| 6 | p_har_rvgk | 0.6444 | 0.755 | **yes** | 0.113 |
| 7 | p_shar | 0.6482 | 0.759 | **yes** | 0.113 |
| 8 | p_har | 0.6487 | 0.760 | **yes** | 0.113 |
| 9 | p_har_gk | 0.6761 | 0.792 | **yes** | 0.113 |
| 10 | robust_bc | 0.8538 | 1.000 | no | 0.035 |
| 11 | robust_bc_c * | 0.8968 | 1.050 | no | 0.044 |

### T, h = 1d, clean_holdout

common sample: 131 dates, 1558 symbol-days, 15 symbols; ratio vs `robust_bc`; MCS Tmax 90%

| rank | model | mean loss | ratio | in MCS | MCS p |
|---|---|---|---|---|---|
| 1 | p_harq | 0.6525 | 0.696 | **yes** | 0.822 |
| 2 | p_loghar_f | 0.6584 | 0.703 | **yes** | 0.822 |
| 3 | p_loghar | 0.6608 | 0.705 | **yes** | 0.822 |
| 4 | p_harj | 0.6625 | 0.707 | **yes** | 0.822 |
| 5 | p_har | 0.6633 | 0.708 | **yes** | 0.822 |
| 6 | p_shar | 0.6634 | 0.708 | **yes** | 0.822 |
| 7 | p_har_rvgk | 0.6637 | 0.708 | **yes** | 0.822 |
| 8 | p_har_gk | 0.7223 | 0.771 | no | 0.035 |
| 9 | ewma | 0.7348 | 0.784 | no | 0.035 |
| 10 | robust_bc_c * | 0.8242 | 0.880 | no | 0.035 |
| 11 | robust_bc | 0.9370 | 1.000 | no | 0.000 |

### T, h = 5d, llm_in_sample

common sample: 443 dates, 842 symbol-days, 4 symbols; ratio vs `robust_bc`; MCS Tmax 90%

| rank | model | mean loss | ratio | in MCS | MCS p |
|---|---|---|---|---|---|
| 1 | p_loghar | 0.2566 | 0.327 | **yes** | 0.524 |
| 2 | ewma | 0.2738 | 0.349 | **yes** | 0.524 |
| 3 | p_loghar_f | 0.2925 | 0.373 | **yes** | 0.524 |
| 4 | p_harj | 0.3521 | 0.449 | **yes** | 0.524 |
| 5 | p_harq | 0.3548 | 0.453 | **yes** | 0.524 |
| 6 | p_har | 0.3608 | 0.460 | **yes** | 0.524 |
| 7 | p_shar | 0.3615 | 0.461 | **yes** | 0.524 |
| 8 | p_har_gk | 0.3651 | 0.466 | **yes** | 0.524 |
| 9 | p_har_rvgk | 0.3739 | 0.477 | **yes** | 0.524 |
| 10 | robust_bc | 0.7841 | 1.000 | no | 0.002 |
| 11 | robust_bc_c * | 0.8890 | 1.134 | no | 0.000 |

### T, h = 5d, clean_holdout

common sample: 127 dates, 1498 symbol-days, 15 symbols; ratio vs `robust_bc`; MCS Tmax 90%

| rank | model | mean loss | ratio | in MCS | MCS p |
|---|---|---|---|---|---|
| 1 | p_shar | 0.2035 | 0.208 | **yes** | 0.829 |
| 2 | p_har | 0.2037 | 0.208 | **yes** | 0.829 |
| 3 | p_har_rvgk | 0.2040 | 0.208 | **yes** | 0.829 |
| 4 | p_harj | 0.2040 | 0.208 | **yes** | 0.829 |
| 5 | p_loghar_f | 0.2110 | 0.215 | **yes** | 0.829 |
| 6 | p_harq | 0.2126 | 0.217 | **yes** | 0.829 |
| 7 | p_har_gk | 0.2131 | 0.217 | **yes** | 0.829 |
| 8 | ewma | 0.2143 | 0.219 | **yes** | 0.829 |
| 9 | p_loghar | 0.2314 | 0.236 | no | 0.002 |
| 10 | robust_bc | 0.9806 | 1.000 | no | 0.000 |
| 11 | robust_bc_c * | 1.0923 | 1.114 | no | 0.000 |

## MSE of log variance (top 8 and MCS)

### L, h = 1d, llm_in_sample

common sample: 1853 dates, 34931 symbol-days, 23 symbols; ratio vs `har`; MCS Tmax 90%

| rank | model | mean loss | ratio | in MCS | MCS p |
|---|---|---|---|---|---|
| 1 | robust_bc_c * | 0.6879 | 0.639 | **yes** | 1.000 |
| 2 | p_loghar | 0.7226 | 0.672 | no | 0.089 |
| 3 | p_loghar_f | 0.7239 | 0.673 | no | 0.089 |
| 4 | robust_bc | 0.7248 | 0.674 | no | 0.089 |
| 5 | loghar | 0.7268 | 0.676 | no | 0.059 |
| 6 | sv_rv * | 0.7660 | 0.712 | no | 0.000 |
| 7 | ewma | 0.8607 | 0.800 | no | 0.000 |
| 8 | harq | 0.9591 | 0.892 | no | 0.000 |

MCS: robust_bc_c

### L, h = 1d, clean_holdout

common sample: 131 dates, 3013 symbol-days, 23 symbols; ratio vs `har`; MCS Tmax 90%

| rank | model | mean loss | ratio | in MCS | MCS p |
|---|---|---|---|---|---|
| 1 | robust_bc_c * | 0.7181 | 0.603 | **yes** | 0.222 |
| 2 | robust_bc | 0.7804 | 0.655 | **yes** | 0.222 |
| 3 | loghar | 0.7917 | 0.665 | **yes** | 0.222 |
| 4 | ewma | 0.8057 | 0.677 | **yes** | 0.222 |
| 5 | p_loghar_f | 0.8085 | 0.679 | **yes** | 0.222 |
| 6 | p_loghar | 0.8128 | 0.683 | **yes** | 0.222 |
| 7 | sv_rv * | 0.8555 | 0.719 | no | 0.002 |
| 8 | egarch | 1.0651 | 0.895 | no | 0.000 |

MCS: robust_bc_c, robust_bc, loghar, ewma, p_loghar_f, p_loghar

### L, h = 5d, llm_in_sample

common sample: 1853 dates, 34931 symbol-days, 23 symbols; ratio vs `har`; MCS Tmax 90%

| rank | model | mean loss | ratio | in MCS | MCS p |
|---|---|---|---|---|---|
| 1 | p_loghar | 0.4880 | 0.668 | **yes** | 0.338 |
| 2 | loghar | 0.4991 | 0.683 | **yes** | 0.338 |
| 3 | sv_rv * | 0.5042 | 0.690 | **yes** | 0.338 |
| 4 | p_loghar_f | 0.5244 | 0.718 | **yes** | 0.338 |
| 5 | ewma | 0.5323 | 0.728 | **yes** | 0.338 |
| 6 | robust_bc | 0.6164 | 0.843 | no | 0.000 |
| 7 | robust_bc_c * | 0.6244 | 0.854 | no | 0.000 |
| 8 | p_harq | 0.6741 | 0.922 | no | 0.000 |

MCS: p_loghar, loghar, sv_rv, p_loghar_f, ewma

### L, h = 5d, clean_holdout

common sample: 127 dates, 2921 symbol-days, 23 symbols; ratio vs `har`; MCS Tmax 90%

| rank | model | mean loss | ratio | in MCS | MCS p |
|---|---|---|---|---|---|
| 1 | ewma | 0.4586 | 0.563 | **yes** | 0.117 |
| 2 | loghar | 0.5216 | 0.641 | **yes** | 0.117 |
| 3 | p_loghar_f | 0.5241 | 0.644 | **yes** | 0.117 |
| 4 | p_loghar | 0.5249 | 0.645 | **yes** | 0.117 |
| 5 | sv_rv * | 0.5659 | 0.695 | **yes** | 0.117 |
| 6 | egarch | 0.6403 | 0.787 | **yes** | 0.117 |
| 7 | robust_bc | 0.6684 | 0.821 | **yes** | 0.117 |
| 8 | robust_bc_c * | 0.6835 | 0.840 | **yes** | 0.117 |

MCS: ewma, loghar, p_loghar_f, p_loghar, sv_rv, egarch, robust_bc, robust_bc_c, gjr, harq, p_harq, p_harj, p_har, p_shar, p_har_rvgk

### T, h = 1d, llm_in_sample

common sample: 334 dates, 516 symbol-days, 2 symbols; ratio vs `har`; MCS Tmax 90%

| rank | model | mean loss | ratio | in MCS | MCS p |
|---|---|---|---|---|---|
| 1 | robust_bc_c * | 1.2691 | 0.703 | **yes** | 0.155 |
| 2 | p_loghar | 1.5662 | 0.868 | **yes** | 0.155 |
| 3 | p_loghar_f | 1.6048 | 0.889 | **yes** | 0.155 |
| 4 | loghar | 1.7170 | 0.952 | **yes** | 0.155 |
| 5 | har_pk * | 1.7372 | 0.963 | **yes** | 0.155 |
| 6 | har_gk | 1.7621 | 0.977 | **yes** | 0.155 |
| 7 | harq | 1.7782 | 0.986 | **yes** | 0.155 |
| 8 | shar | 1.7865 | 0.990 | **yes** | 0.155 |

MCS: robust_bc_c, p_loghar, p_loghar_f, loghar, har_pk, har_gk, harq, shar, p_harq, har, p_har_rvgk, p_harj, p_shar, p_har, p_har_gk, har_rvgk, sv_rv, robust_bc, sv_r2, harj, gjr

### T, h = 1d, clean_holdout

common sample: 131 dates, 346 symbol-days, 5 symbols; ratio vs `har`; MCS Tmax 90%

| rank | model | mean loss | ratio | in MCS | MCS p |
|---|---|---|---|---|---|
| 1 | robust_bc_c * | 1.3494 | 0.478 | **yes** | 1.000 |
| 2 | p_loghar | 2.2405 | 0.793 | no | 0.004 |
| 3 | p_loghar_f | 2.2819 | 0.808 | no | 0.004 |
| 4 | har_pk * | 2.7511 | 0.974 | no | 0.004 |
| 5 | har_gk | 2.7516 | 0.974 | no | 0.004 |
| 6 | p_harq | 2.7633 | 0.979 | no | 0.004 |
| 7 | loghar | 2.7954 | 0.990 | no | 0.004 |
| 8 | har_rvgk * | 2.8074 | 0.994 | no | 0.004 |

MCS: robust_bc_c

### T, h = 5d, llm_in_sample

common sample: 334 dates, 516 symbol-days, 2 symbols; ratio vs `har`; MCS Tmax 90%

| rank | model | mean loss | ratio | in MCS | MCS p |
|---|---|---|---|---|---|
| 1 | loghar | 0.4022 | 0.783 | **yes** | 0.122 |
| 2 | p_loghar | 0.4374 | 0.851 | **yes** | 0.122 |
| 3 | p_loghar_f | 0.4618 | 0.899 | **yes** | 0.122 |
| 4 | har | 0.5138 | 1.000 | **yes** | 0.122 |
| 5 | shar | 0.5172 | 1.007 | **yes** | 0.122 |
| 6 | harq | 0.5186 | 1.009 | **yes** | 0.122 |
| 7 | p_harj | 0.5205 | 1.013 | **yes** | 0.122 |
| 8 | harj | 0.5250 | 1.022 | **yes** | 0.122 |

MCS: loghar, p_loghar, p_loghar_f, har, shar, harq, p_harj, harj, p_har_gk, p_har, p_shar, p_harq, gjr, p_har_rvgk, egarch, garch, sv_rv, ewma, har_pk, sv_r2_cal, sv_r2, har_gk, har_rvgk

### T, h = 5d, clean_holdout

common sample: 127 dates, 326 symbol-days, 5 symbols; ratio vs `har`; MCS Tmax 90%

| rank | model | mean loss | ratio | in MCS | MCS p |
|---|---|---|---|---|---|
| 1 | loghar | 0.1712 | 0.649 | **yes** | 0.111 |
| 2 | sv_r2 | 0.1881 | 0.712 | **yes** | 0.111 |
| 3 | har_gk | 0.2132 | 0.808 | **yes** | 0.111 |
| 4 | har_pk * | 0.2201 | 0.834 | **yes** | 0.111 |
| 5 | harq | 0.2423 | 0.918 | **yes** | 0.111 |
| 6 | p_har_gk | 0.2611 | 0.989 | **yes** | 0.111 |
| 7 | har | 0.2640 | 1.000 | **yes** | 0.111 |
| 8 | p_harj | 0.2645 | 1.002 | no | 0.089 |

MCS: loghar, sv_r2, har_gk, har_pk, harq, p_har_gk, har, p_shar, ewma, harj, p_loghar_f


## Weekend seasonality (descriptive, no hypothesis): the largest loss term nobody models

Next-day forecasts by target day: median RV relative to Tue-Fri, and mean log(F / RV) as the bias. Every contestant is
calendar-blind except `robust_bc_c`.

| universe, segment | Sat RV / Tue-Fri | log-HAR bias, Sat | log-HAR bias, Sun | log-HAR bias, Mon | log-HAR bias, Tue-Fri |
|---|---|---|---|---|---|
| L, in-sample | 0.53 | +0.91 | +0.45 | -0.02 | +0.25 |
| L, holdout | 0.46 | +0.98 | +0.43 | +0.14 | +0.24 |
| T, in-sample (PAXG, XAUT) | 0.10 | +2.65 | +0.87 | -0.29 | +0.06 |
| T, holdout | 0.05 | +3.71 | +1.55 | +0.05 | +0.15 |

For TradFi this dominates the QLIKE: Saturday's next-day QLIKE is 2-3 versus 0.2-0.4 on weekdays.

**For risk this is the safe direction.** Weekends are jump-prone (V3, H-V2: shrinking σ at weekends worsened tail
loss), so this is **not** a reason to shrink weekend risk. It is a reason to compare quote widths and central forecasts
against a weekday-aware model.

**Proposed follow-up, not registered:** H-VM6, a HAR with day-of-week (or weekend-share) terms in the log panel, tested
on QLIKE and on tail coverage.

## Caveats

- **RV from 1h bars is a noisy proxy.** 24 returns a day, and no microstructure correction is needed at 1h. QLIKE is
  robust to proxy noise in ranking (Patton 2011), but the level of every loss includes the proxy noise. The GK/PK
  result (H-VM3) is specific to 1h ranges summed over the day, and may differ with 5m ranges.
- **The holdout is short.** 131 days (1d), 127 (5d). The MCS keeps nearly every model, so H-VM5 "supports" is weak
  evidence. Only robust_bc and the SV-on-r^2 variants are excluded at 1d.
- **TradFi is mostly holdout.** The T common sample for all 25 models holds 5 symbols (PAXG, XAUT, plus 20-35 days of
  XAU, XAG, CL). The "short set" board covers all 15 TradFi symbols but only for models needing no 180-day own history.
  T recommendations were selected on the holdout, so they count as in-sample by protocol: confidence C.
- **The pooled panel mixes asset classes.** It is fit on L and T together. T contributes about 6% of the training
  pairs, so its slopes are essentially crypto slopes with TradFi fixed effects.
- **Corrections after inspection.** Two estimator fixes were found in a one-symbol smoke test, and one in a first
  (unrecorded) evaluation pass:
  - the EGARCH degenerate optima;
  - the HAR-Q collapse onto the clip floor, which led to the BPQ insanity filter;
  - EGARCH variance-path collapse.

  All three are generic and are logged in the diary as `mistake` entries. No model specification, sample or
  evaluation choice was changed afterwards. EGARCH 5d was not cut: Monte Carlo was cheap.
- **Survivorship.** Today's L is a survivor list.
- **No transaction costs.** This is a loss tournament, not P&L.

## Rule candidates

Ids are numbered from 20 so they don't collide with `research/rules.yaml`; the orchestrator renumbers when merging. All six pass `ai_trader.research.rules.validate`.

```yaml
- id: R-VOL-20
  kind: prefer
  rule: "WHEN forecasting next-day variance for a crypto perp with at least 180 days of history THEN prefer the per-asset log-HAR (vol.har) over robust_bc x 24, EWMA, GARCH-family and SV-on-r^2 forecasts"
  when: "asset_class == crypto and complete daily history >= 180 days; horizon next UTC day"
  then: "use log-HAR (lognormal back-transform) as the daily risk forecast; robust_bc stays the per-bar sigma"
  scope: {assets: crypto, horizon: "1d", sides: both}
  effect: "QLIKE -12.1% vs level HAR (t -5.0) and -17.6% vs robust_bc x 24 in-sample; holdout -4.5% vs HAR and -18% vs robust_bc x 24; in-sample winner, in the holdout 90% MCS (p 0.17)"
  confidence: B
  evidence: [H-VM5, H-VM2, H-VCL1, research/reports/vol_tournament_2026-11-10.md]
  review: "holdout doubles and the holdout MCS shrinks enough to exclude log-HAR, or the HAR-Q/log-HAR ordering flips in the holdout"
  status: provisional
- id: R-VOL-21
  kind: avoid
  rule: "WHEN a 5-day (or multi-day) variance forecast is needed THEN don't scale the 1h robust_bc sigma flat over the horizon; use a direct 5d HAR (panel log-HAR for crypto)"
  when: "horizon >= 2 days, any asset"
  then: "use the direct 5d panel log-HAR (crypto) or level panel HAR (TradFi); expect robust_bc x 24 per day to be badly calibrated"
  scope: {assets: all, horizon: "5d", sides: both}
  effect: "robust_bc x 24 flat has 1.51x (in-sample) / 2.27x (holdout) the QLIKE of level HAR for crypto, and 8.6x for TradFi (holdout); panel log-HAR is the crypto in-sample 5d winner (-15.5% vs HAR) and in the holdout MCS"
  confidence: B
  evidence: [H-VM5, H-VM1, research/reports/vol_tournament_2026-11-10.md]
  review: "a weekday-aware robust_bc term structure closes the gap"
  status: provisional
- id: R-VOL-22
  kind: prefer
  rule: "WHEN forecasting daily risk for a short-history perp (TradFi perps, or any listing with < 180 days) THEN prefer the panel HAR pooled over crypto and TradFi (HAR-Q or log form, symbol fixed effects) over robust_bc and EWMA"
  when: "complete daily history < 180 days, or asset_class == tradfi; panel needs >= 30 own training pairs (~60-90 days)"
  then: "use the panel HAR-Q (p_harq) for TradFi 1d and the level panel HAR for TradFi 5d; panel log-HAR for young crypto listings"
  scope: {assets: tradfi, horizon: "1d|5d", sides: both}
  effect: "TradFi holdout QLIKE 0.70x robust_bc x 24 at 1d and 0.21x at 5d (15 symbols, 131 days); crypto < 180 days: panel log-HAR -27.5% vs robust_bc (t -5.2) holdout"
  confidence: C
  evidence: [H-VM1, H-VM5, research/reports/vol_tournament_2026-11-10.md]
  review: "TradFi model choice was made on the holdout: re-test when the TradFi sample doubles; a weekday-aware HAR (proposed H-VM6) may replace it"
  status: provisional
- id: R-VOL-23
  kind: prefer
  rule: "WHEN a level (non-log) HAR is used THEN prefer the HAR-Q variant (RV_d coefficient falling with sqrt(RQ_d)) over plain HAR"
  when: "model family = level HAR, per-asset or panel"
  then: "add the RV_d x sqrt(RQ_d) term; expect smaller over-reaction after noisy, jumpy days"
  scope: {assets: all, horizon: "1d|5d", sides: both}
  effect: "next-day QLIKE -5.1% (t -6.8) in-sample, -4.7% (t -3.3) holdout per asset; panel -4.2% (t -7.9) / -3.7% (t -4.1); HAR-Q is the crypto 5d holdout leader"
  confidence: B
  evidence: [H-VM2, research/reports/vol_tournament_2026-11-10.md]
  review: "sign flips in a larger holdout"
  status: provisional
- id: R-VOL-24
  kind: avoid
  rule: "WHEN choosing inputs for a daily variance forecast from 1h bars THEN don't expect range estimators (Garman-Klass, Parkinson) to beat 1h realised variance"
  when: "only 1h OHLC available"
  then: "keep RV (sum of squared 1h returns) as the HAR input; GK/PK add nothing"
  scope: {assets: all, horizon: "1d|5d", sides: both}
  effect: "GK-HAR QLIKE +5.4% (t 4.9) in-sample, +15.7% (t 4.8) holdout vs RV-HAR at 1d; RV+GK no better than RV"
  confidence: A
  evidence: [H-VM3, research/reports/vol_tournament_2026-11-10.md]
  review: "re-test with 5m ranges once enough history exists"
  status: provisional
- id: R-VOL-25
  kind: expect
  rule: "WHEN reading alt volatility THEN expect it to move mostly with the market-wide crypto vol factor, but don't expect the factor to improve the alt's own HAR forecast"
  when: "factor = cross-sectional mean log RV over the 23-perp crypto universe"
  then: "use the factor to describe the market vol state (one regime for all alts); forecast each alt from its own panel log-HAR"
  scope: {assets: crypto, horizon: "1d", sides: both}
  effect: "median alt R^2 of log RV on the factor 0.72 in-sample / 0.68 holdout (leave-one-out 0.70 / 0.65); forecast gain -0.1% (t -0.6) / -1.0% (t -1.6)"
  confidence: B
  evidence: [H-VM4, H-VCL3, research/reports/vol_tournament_2026-11-10.md]
  review: "holdout gain reaches t < -2 as the holdout grows"
  status: provisional
```
