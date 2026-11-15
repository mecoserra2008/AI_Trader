# Population extrapolation and event-level simulator v1 (H-MS7, H-MS8)

_Agent C3, microstructure pilot, 2026-11-11. Data: trades_raw 2026-11-10 (51 Bybit perps, every print), FIT 00-12 UTC, TEST 12-24 UTC (CPI at 13:30, about twice the activity); flow_3s TEST half for the real stylised facts; flow_3s 2026-11-07..09 (before TEST) for spread, depth and the book-mid shape targets. Segment: clean_holdout. Code: `src/ai_trader/sim/population.py`, `src/ai_trader/sim/flow_sim_v1.py`, `src/ai_trader/research/population_simulator_study.py`; tests `tests/test_population.py`, `tests/test_flow_sim_v1.py`; population model `data/models/population/population_2026-11-10.json`; machine-readable results `research/reports/population_simulator_2026-11-11.json`. Generated data is for deterministic strategies and the execution engine only, never for the LLM layer._

## Summary

- **H-MS7 contradicts.** Order rate (LOO R^2 0.85), Kyle lambda (0.64) and depth (0.66) are functions of trading activity W, but the relative spread is not (LOO R^2 0.28 on log W and class): 39 of 51 perps are tick-bound, so their spread is tick / price, an exchange choice. Spread becomes predictable once the tick enters (free-spread level on log W and log tick: LOO R^2 0.85, exploratory). The shape exponents are not universal either: the between-symbol spread of the order size tail index is 2.4x its FIT-vs-TEST noise variance (p 0.03) and of the sign-memory gamma 4.9x (p < 0.001).
- **H-MS8 contradicts, narrowly.** v1 beats v0 in every class (facts inside the TEST envelopes: majors 3 vs 2, alts 4 vs 0, TradFi 3 vs 1), but the majors need 4 of 5 and get 3 (impact 0.42 vs an envelope up to 0.41; spread 0.915 vs 0.922-0.936 bp, a tick / price discreteness effect). In 1000 symbol-bootstrap replicates the majors reach 4/5 in 23% of draws (3/5 in 57%) and the whole criterion holds in 21%. The signed-flow ACF, v0's main failure, is now inside in every class (majors 0.014 vs real 0.021; v0 0.0003).
- **Extrapolation works for levels, not for regimes.** Leave-one-out, params_for(W, class, tick) predicts each symbol's order rate within a median factor of 1.29, its 1-min volatility within 1.23 and the median order notional within 1.39; the spread regime is right for 38 of 51. The population-only generator lands 2/5 (majors), 1/5 (alts), 2/5 (TradFi) facts inside the TEST envelopes, about as many as v0 that was fitted on each symbol's own data.
- **MM replay (descriptive).** On v1 paths the 36-set grid ranking by net bp per turnover agrees with real TEST replays at Spearman 0.30 (v0: -0.09) and turnover / fill counts at 0.995; the live mm_multi set makes 8.2 bp on the real TEST half and -10.3 bp on v1, the replay-best set 11.8 vs -1.0 bp. A driftless generator still cannot rank skewed quoting on a sell-off day.

## Verdicts

| id | registered success criterion | key numbers | p / Holm (MS7, MS8) | verdict |
|---|---|---|---|---|
| H-MS7 | LOO R^2 > 0.5 for rate, spread, impact; zeta and gamma dispersion not significantly above FIT-vs-TEST noise | LOO R^2 rate 0.848 [0.499, 0.905], spread 0.282 [-0.141, 0.482], impact 0.642 [0.330, 0.777]; D(zeta) 2.43 (p_disp 0.026), D(gamma) 4.87 (p_disp 0.000) | 0.976 / 1.000 | **contradicts** |
| H-MS8 | v1 facts inside the TEST envelopes > v0 in every class and v1 >= 4/5 for majors | v1 / v0: major 3/2, alt 4/0, tradfi 3/1 | 0.794 / 1.000 | **contradicts** |

p for H-MS7 = the largest bootstrap P*(LOO R^2 <= 0.5) of the three liquidity parameters (the universality part is a pass/fail condition: p_disp >= 0.05); p for H-MS8 = share of 1000 symbol-bootstrap replicates (envelopes and simulated medians resampled together) in which the criterion fails. A criterion that is not met gives contradicts; met with Holm p >= 0.05 gives inconclusive. Holm over the whole MS block with the recorded p values of H-MS1..6, 9: H-MS1 1.000, H-MS2 1.000, H-MS3 0.224, H-MS4 1.000, H-MS5 1.000, H-MS6 1.000, H-MS7 1.000, H-MS8 1.000, H-MS9 1.000.

## Methods (fixed before the tests)

- **data**: trades_raw 2026-11-10, 51 symbols (config/mm_multi.yaml + BTCUSDT); orders = same-ms same-side prints merged (C1); FIT = 00-12 UTC, TEST = 12-24 UTC; flow_3s TEST half for the real facts (12-24 UTC; 6.3k-9.9k ok 3 s candles per symbol, gaps never filled); flow_3s 2026-11-07..09 (before TEST) for spread / depth / tick regime; C1 FIT-half laws for sizes, signs, prints per order
- **classes**: major = BTC ETH SOL XRP DOGE; tradfi = CL BZ XAU XAG QQQ NVDA TSLA MSTR COIN CRWV GOOGL; alt = the rest (incl. BNB) (MS block)
- **W**: W = sigma_day x P V_day from FIT trades: sigma_day = std of 1-min log returns of the trade mid proxy ((last buy + last sell) / 2) x sqrt(1440); P V_day = 2 x FIT notional (USD); illiquidity 1/L = (sigma_day^2 / P V_day)^(1/3) (Kyle-Obizhaeva 2016)
- **parameters**: per symbol, FIT half: log orders/s (rate); deseasonalised expmix3 duration weights and log component means (mean-1 units); order size tail zeta (C1 size.order) and log-normal body mu, sigma; order sign gamma (C1 signs.order.acf_power) and lag-1 ACF; log median relative spread (bp, pre-TEST flow_3s TWA) of all symbols; tick regime (logistic on log tick/price and log W); log Kyle lambda = OLS slope of 1 s trade-mid-proxy log return (bp) on 1 s signed notional (USD 1k) (trade-price slope reported)
- **H-MS7**: OLS of each parameter on [1, log W, alt, tradfi] over the 51 symbols; leave-one-symbol-out R^2 = 1 - PRESS / TSS; 95% CI and p = P*(R2_LOO <= 0.5) by a bootstrap over symbols (B = 2000; LOO on the resample with all copies of the held-out symbol dropped); universality of zeta (order size tail) and gamma (order sign ACF): D = Var((F+T)/2) / (Var(F-T)/4) across symbols (F, T = FIT and TEST estimates; D = 1 when the between-symbol spread is all estimation noise; D > 1 iff the halves covary), p_disp = bootstrap P*(D <= 1); supports = LOO R^2 > 0.5 for rate, spread and impact AND p_disp >= 0.05 for zeta and gamma, with Holm p < 0.05; summary p = max of the three R^2 bootstrap p values; criterion met with Holm p >= 0.05 -> inconclusive; not met -> contradicts
- **v1**: sim/flow_sim_v1.py fitted per symbol on FIT: intensity two-component AR of log order counts (bins with mean >= 10 orders), expmix3 on durations x local rate (centered 5-bin window) rescaled to mean 1, C1 size law (capped at the FIT maximum order), C1 TranslatedSignModel with p_buy = 0.5 (no drift), prints-per-order pmf rank-coupled to notional (FIT Spearman), square-root impact per order with permanent share g_inf = 0.6 and tau = 30 s (fixed, not fitted), noise variance prop. to activity, jumps from FIT 1-min exceedances (> 6 robust sd), spread latent AR(1) in log ticks censored at 1 tick (pre-TEST flow_3s quantiles, 3 s lag-1 ACF), elasticity 0.07 to log activity; k_imp, sigma0 and jump rate by indirect inference (2 pilots x 4 h, 5 iterations) so that the same trade-based estimators on simulated trades match FIT lambda, 1-min sigma and exceedance rate
- **H-MS8**: facts as H-SIM2 (excess kurtosis of 3 s mid returns, mean ACF |r| lags 1-60, mean ACF signed flow lags 1-20, corr(r_t, F_t), spread median AND p90) on the TEST half; envelope = [2.5%, 97.5%] of the class median under a 1 h block bootstrap of each symbol's TEST facts (B = 300), the v0 method on one period; sim value = class median over symbols of each symbol's median over 10 paths of the symbol's TEST ok-candle count; v0 = the H-SIM2 FlowSim models (fitted 2026-11-08/09) with the same path count; supports = v1 count > v0 count in every class and v1 >= 4/5 for majors, with Holm p < 0.05; p = share of 1000 symbol-bootstrap replicates (resampling symbols within class for envelope and sims together) where the criterion fails; level-vs-shape: v1 refitted with the TEST levels (order rate, 1-min sigma, lambda, spread location) as an oracle diagnostic
- **holm**: Holm over H-MS7 and H-MS8 (the registered family is the whole MS block; the block-wide adjustment with the other recorded MS p values is reported)
- **replay**: descriptive only (H-SIM4 / H-MM3 are the registered tests): sim.replay run_mm with the H-SIM1 touch fill model, decisions every 9 s, real = flow_3s TEST half of the 50 mm_multi symbols, v1 = 3 paths per symbol (5 h warm-up + TEST length), 3 fill seeds; sets: live config/mm_multi.yaml and the replay-best a0.2_b0.9_tp0.9_rq0.15 (plus the 36-set grid for a rank comparison)

## H-MS7: what scales with liquidity

W = sigma_day x dollar volume per day, both from the FIT trades (sigma_day from 1-min returns of the trade mid proxy). It spans five orders of magnitude (CARV, MAVIA about 1e4 USD/day; BTC, ETH about 1e8-1e9). What scales with W (LOO R^2 >= 0.5): order rate (slope 0.54 per log W), depth within 25 bp (0.68), Kyle lambda (-0.49; in invariance form lambda / sigma^2 slopes -0.75 against the predicted -2/3, t -1.9, so consistent), the burst-component duration in mean-duration units (0.48: bursts stay at a few ms in clock time while the mean gap shrinks), prints per order (0.22) and the FIT |r| ACF (0.013 per log W). Weak or no scaling (LOO R^2 below 0.35): spread on W alone, median order notional (0.12, LOO 0.33), daily sigma, sign lag-1 ACF and gamma, size tail index, mixture weights, 3 s flow-return correlation and signed-flow ACF.

The parameters that do not scale are not universal either. For the size tail index and gamma the FIT and TEST estimates covary across symbols (D 2.4 and 4.9; within class still 2.4 and 3.3), and the sign lag-1 ACF is strongly symbol-specific (D 20). These are persistent symbol traits (order splitting styles, lot structure) that neither W nor class explains: the population model can only carry them as class medians with a wide band. TradFi perps have lower gamma (class effect -0.30: longer sign memory), as C1 found.

Invariance exponents are flatter than Kyle-Obizhaeva for counts and sizes: orders per day slope 0.52 (prediction 2/3, t -6.0), median order notional 0.20 (1/3, t -2.9); prints per day 0.61 is closer, so part of the gap is that one Bybit taker order is not one bet. Spreads fall faster than invariance: spread / sigma slope -0.55 overall and -0.75 for free-spread symbols (prediction -1/3), and spread vs illiquidity 1.92 (prediction 1; v0 found 1.69 on flow_3s). Tick-size floors and the thin books of the illiquid tail both push in that direction.

The spread-regime logistic (log tick in bp +0.53, log W +0.71: a large tick and high activity both mean tick-bound) has a leave-one-out accuracy of 0.75, no better than calling everything tick-bound (base rate 0.76); its Brier score is 0.17. The regime is better read off the instrument's own book when one exists.

Cross-sectional OLS on [1, log W, alt, tradfi], 51 symbols (FIT half). Slopes are per unit of log W; class effects are relative to majors. LOO R^2 = 1 - PRESS / TSS with a symbol bootstrap 95% CI.

| parameter | n | slope on log W (HC1 SE) | alt / tradfi effect | R^2 | LOO R^2 | LOO 95% CI | LOO R^2 (log W only) |
|---|---|---|---|---|---|---|---|
| log orders / s (primary) | 51 | 0.540 (0.032) | 0.17 / 0.24 | 0.865 | 0.848 | [0.499, 0.905] | 0.853 |
| log median relative spread (bp) (primary) | 51 | -0.447 (0.105) | 0.60 / -0.50 | 0.414 | 0.282 | [-0.141, 0.482] | 0.268 |
| log Kyle lambda (mid proxy, bp per $1k) (primary) | 51 | -0.487 (0.086) | 1.76 / 0.15 | 0.698 | 0.642 | [0.330, 0.777] | 0.447 |
| log median order notional | 51 | 0.124 (0.044) | -0.88 / -0.62 | 0.437 | 0.327 | [-0.299, 0.566] | 0.244 |
| size body mu (log USD) | 51 | -0.114 (0.225) | -1.37 / -0.61 | 0.071 | -0.156 | [-0.680, 0.345] | -0.137 |
| log daily sigma | 51 | 0.123 (0.043) | 0.69 / 0.33 | 0.314 | 0.164 | [-0.750, 0.364] | -0.026 |
| log depth within 25 bp | 51 | 0.682 (0.116) | -1.82 / -0.95 | 0.709 | 0.661 | [0.293, 0.793] | 0.593 |
| order sign ACF lag 1 | 51 | 0.011 (0.007) | -0.01 / -0.04 | 0.072 | -0.139 | [-0.747, 0.057] | -0.045 |
| order sign gamma | 51 | -0.010 (0.018) | 4.61e-03 / -0.30 | 0.290 | 0.189 | [-0.100, 0.378] | -0.084 |
| order size tail zeta | 51 | -0.009 (0.039) | -0.06 / -0.19 | 0.019 | -0.162 | [-0.618, -0.067] | -0.097 |
| prints per order | 51 | 0.218 (0.054) | -0.07 / -1.16 | 0.665 | 0.594 | [0.256, 0.727] | 0.275 |
| burst weight (mixture) | 51 | 0.017 (0.007) | -0.02 / -0.08 | 0.331 | 0.115 | [-0.638, 0.408] | 0.109 |
| middle weight | 51 | 0.004 (0.006) | 0.01 / 0.08 | 0.122 | -0.043 | [-0.361, 0.058] | -0.057 |
| log burst mean (x mean duration) | 51 | 0.480 (0.049) | 0.02 / 0.22 | 0.792 | 0.763 | [0.219, 0.869] | 0.767 |
| log middle mean | 51 | 0.337 (0.075) | 0.21 / 0.35 | 0.364 | 0.271 | [-0.215, 0.572] | 0.302 |
| log slow mean | 51 | 0.033 (0.013) | -0.04 / -0.05 | 0.216 | -0.013 | [-0.758, 0.290] | 0.103 |
| log Kyle lambda (trade price) | 51 | -0.598 (0.093) | 1.73 / 0.17 | 0.741 | 0.693 | [0.359, 0.810] | 0.539 |
| 3 s corr(r, F), pre-TEST book | 51 | 0.004 (0.007) | -0.01 / -0.15 | 0.486 | 0.348 | [-0.298, 0.606] | -0.085 |
| 3 s signed-flow ACF (FIT) | 51 | 0.001 (6.78e-04) | -0.01 / -8.85e-03 | 0.281 | 0.108 | [-0.542, 0.454] | 0.092 |
| 3 s |r| ACF of the mid proxy (FIT) | 51 | 0.013 (0.002) | -4.05e-03 / 0.06 | 0.682 | 0.615 | [0.277, 0.767] | 0.255 |
| log calibrated impact k | 51 | -0.212 (0.066) | 0.80 / -0.02 | 0.520 | 0.435 | [0.033, 0.655] | 0.282 |

Universality of the shape exponents (D = Var((F+T)/2) / (Var(F-T)/4); D = 1 when the between-symbol spread equals estimation noise; p_disp = bootstrap P*(D <= 1)):

| parameter | n | mean | between-symbol sd | noise sd (FIT-TEST) | D | D 95% CI | p_disp | Spearman FIT vs TEST | D within class | p_disp within class |
|---|---|---|---|---|---|---|---|---|---|---|
| zeta_order | 51 | 1.332 | 0.322 | 0.207 | 2.430 | [1.00, 6.18] | 0.026 | 0.341 | 2.366 | 0.031 |
| sign_gamma | 51 | 0.673 | 0.207 | 0.094 | 4.867 | [2.89, 8.30] | 0.000 | 0.634 | 3.256 | 0.000 |
| zeta_print | 51 | 1.704 | 0.379 | 0.308 | 1.514 | [0.83, 2.88] | 0.102 | 0.271 | 1.163 | 0.307 |
| sign_acf1 | 51 | 0.365 | 0.098 | 0.022 | 20.241 | [10.67, 40.78] | 0.000 | 0.904 | 19.860 | 0.000 |

Gamma on the C1 eligibility set (>= 20k prints, n = 48): D 5.44 [3.44, 8.78], p_disp 0.000.

Kyle-Obizhaeva invariance (log-log slopes on log W, HC1 SE):

| relation | n | slope (SE) | invariance | t vs invariance | R^2 |
|---|---|---|---|---|---|
| orders per day vs W (bets ~ W^2/3) | 51 | 0.524 (0.024) | 0.667 | -6.042 | 0.862 |
| prints per day vs W | 51 | 0.611 (0.023) | 0.667 | -2.377 | 0.902 |
| median order notional vs W (bet size ~ W^1/3) | 51 | 0.197 (0.048) | 0.333 | -2.855 | 0.314 |
| spread / sigma vs W (~ W^-1/3) | 51 | -0.551 (0.083) | -0.333 | -2.623 | 0.492 |
| spread vs illiquidity (sigma^2 / PV)^1/3 | 51 | 1.924 (0.205) | 1.000 | 4.513 | 0.636 |
| lambda / sigma^2 vs W (~ W^-2/3) | 51 | -0.752 (0.046) | -0.667 | -1.859 | 0.848 |
| lambda (trade price) / sigma^2 vs W | 51 | -0.861 (0.041) | -0.667 | -4.742 | 0.894 |
| spread / sigma vs W, free-spread symbols only | 12 | -0.749 (0.077) | -0.333 | -5.390 | 0.768 |

Spread regime: tick-bound = at least 90% of pre-TEST 3 s candles below 1.5 ticks; 39 tick-bound, 12 free (CARVUSDT, COINUSDT, CRWVUSDT, GOOGLUSDT, KASUSDT, MAVIAUSDT, MSTRUSDT, NVDAUSDT, RENDERUSDT, TAOUSDT, TSLAUSDT, ZECUSDT). Logistic on [1, log tick_bps, log W]: coefficients -7.94, 0.53, 0.71; accuracy in sample 0.824, leave-one-out 0.745 (base rate 0.765), LOO Brier 0.169. Free-spread level (log bp on [1, log W, log tick_bps], n = 12): coefficients 2.827, -0.193, 1.060, LOO R^2 0.853.

## H-MS8: simulator v1 against the TEST stylised facts

v1 is calibrated per symbol from FIT trades (order rate, intensity persistence, duration mixture, sizes, signs, prints per order, signed-flow and size ACF, 1-min volatility, jump rate) and pre-TEST book mids (3 s flow-return correlation, variance ratios, spread). Each path has the symbol's TEST ok-candle count; the value per symbol is the median over 10 paths and the class value the median over symbols. The calibration itself is close for correlation, flow ACF and variance ratios; volatility and the |r| ACF of majors come out about 20-35% low on the pilots.

What v1 fixes: the signed-flow ACF (inside in all classes; v0 had about zero) and the kurtosis (inside for majors and alts; v0 was 4-9x too high). What it misses: alts' |r| ACF (0.019 vs real 0.088: the quiet FIT morning gives a target of 0.023, the CPI afternoon has a volatility regime the stationary generator does not contain; with TEST levels it overshoots to 0.096), TradFi kurtosis (15.9 vs 9.4-15.0, marginal) and the spread of tick-bound majors and TradFi (discreteness and price level).

Level versus shape: handing v1 the TEST levels (twice the order rate, TEST 1-min sigma, TEST price and spread location) does not rescue it; it breaks facts that passed (majors |r| ACF 0.136, alts kurtosis 14.5) because impact, noise and coupling were calibrated jointly with the FIT level. Most failures are therefore shape or regime failures, not a simple level shift; the CPI afternoon is a different state, not a scaled morning.

Per-symbol own envelopes (the stricter, descriptive check) show the same pattern: v1 passes kurtosis for 43-64% of symbols (v0 0-45%) and flow ACF for 55-100%, impact for 40-73%, and the spread for few symbols in any model (0-36%).

**major** (5 symbols): v1 3/5, v0 2/5, v1 with TEST levels 1/5, population v1 (leave-one-out) 2/5

| fact | real TEST | envelope | v1 | in | v0 | in | v1 TEST levels | in | population v1 | in | v1 failure |
|---|---|---|---|---|---|---|---|---|---|---|---|
| excess kurtosis of 3 s returns | 10.977 | [7.939, 14.370] | 8.329 | yes | 59.030 | no | 7.373 | no | 6.440 | no | pass |
| mean ACF of |r|, lags 1-60 | 0.086 | [0.052, 0.118] | 0.079 | yes | 0.058 | yes | 0.136 | no | 0.082 | yes | pass |
| mean ACF of signed flow, lags 1-20 | 0.021 | [0.008, 0.033] | 0.014 | yes | 2.89e-04 | no | 0.009 | yes | 0.016 | yes | pass |
| impact corr(r_t, F_t) | 0.373 | [0.306, 0.414] | 0.423 | no | 0.399 | yes | 0.274 | no | 0.423 | no | shape |
| spread median (bp) | 0.936 | [0.922, 0.936] | 0.915 | no | 0.881 | no | 0.943 | no | 0.867 | no | shape |
| spread p90 (bp) | 0.950 | [0.950, 0.950] | 0.922 | no | 0.908 | no | 0.950 | no | 0.867 | no | shape |
| continuation ratio (extra) | 0.545 | [0.537, 0.568] | 0.500 | no | 0.499 | no | 0.498 | no | 0.504 | no |  |

**alt** (35 symbols): v1 4/5, v0 0/5, v1 with TEST levels 0/5, population v1 (leave-one-out) 1/5

| fact | real TEST | envelope | v1 | in | v0 | in | v1 TEST levels | in | population v1 | in | v1 failure |
|---|---|---|---|---|---|---|---|---|---|---|---|
| excess kurtosis of 3 s returns | 11.949 | [8.477, 12.534] | 11.124 | yes | 110.113 | no | 14.474 | no | 12.586 | no | pass |
| mean ACF of |r|, lags 1-60 | 0.088 | [0.062, 0.089] | 0.019 | no | 0.040 | no | 0.096 | no | 0.026 | no | shape |
| mean ACF of signed flow, lags 1-20 | 0.005 | [0.002, 0.006] | 0.004 | yes | -1.47e-04 | no | 0.007 | no | 0.004 | yes | pass |
| impact corr(r_t, F_t) | 0.343 | [0.320, 0.364] | 0.348 | yes | 0.417 | no | 0.291 | no | 0.371 | no | pass |
| spread median (bp) | 1.977 | [1.947, 1.977] | 1.947 | yes | 2.007 | no | 2.007 | no | 2.150 | no | pass |
| spread p90 (bp) | 2.167 | [1.977, 2.268] | 1.977 | yes | 2.070 | no | 2.101 | no | 2.167 | no | pass |
| continuation ratio (extra) | 0.535 | [0.528, 0.545] | 0.510 | no | 0.501 | no | 0.511 | no | 0.520 | no |  |

**tradfi** (11 symbols): v1 3/5, v0 1/5, v1 with TEST levels 1/5, population v1 (leave-one-out) 2/5

| fact | real TEST | envelope | v1 | in | v0 | in | v1 TEST levels | in | population v1 | in | v1 failure |
|---|---|---|---|---|---|---|---|---|---|---|---|
| excess kurtosis of 3 s returns | 12.604 | [9.359, 15.003] | 15.947 | no | 48.695 | no | 16.030 | no | 16.288 | no | shape |
| mean ACF of |r|, lags 1-60 | 0.134 | [0.102, 0.154] | 0.108 | yes | 0.051 | no | 0.192 | no | 0.127 | yes | pass |
| mean ACF of signed flow, lags 1-20 | 0.004 | [-6.74e-04, 0.008] | 0.007 | yes | 2.36e-04 | yes | 0.010 | no | 0.004 | yes | pass |
| impact corr(r_t, F_t) | 0.206 | [0.171, 0.237] | 0.218 | yes | 0.304 | no | 0.218 | yes | 0.253 | no | pass |
| spread median (bp) | 1.107 | [1.026, 1.124] | 0.804 | no | 1.010 | no | 1.107 | no | 0.791 | no | shape |
| spread p90 (bp) | 1.195 | [1.177, 1.330] | 1.177 | no | 1.141 | no | 1.392 | no | 1.042 | no | shape |
| continuation ratio (extra) | 0.511 | [0.505, 0.521] | 0.479 | no | 0.499 | no | 0.495 | no | 0.503 | no |  |

Share of symbols whose own simulated fact lies inside their own TEST envelope (v1 / v0):

| class | kurtosis | absr_acf | flow_acf | impact | spread |
|---|---|---|---|---|---|
| major | 0.60 / 0.00 | 1.00 / 1.00 | 1.00 / 0.40 | 0.40 / 0.80 | 0.00 / 0.00 |
| alt | 0.43 / 0.14 | 0.31 / 0.51 | 0.66 / 0.57 | 0.57 / 0.31 | 0.17 / 0.17 |
| tradfi | 0.64 / 0.45 | 0.45 / 0.27 | 0.55 / 0.73 | 0.73 / 0.18 | 0.36 / 0.09 |

Calibration check (FIT / pre-TEST target vs simulated value on the calibration pilots, class medians):

| class | sigma 1 min (bp) | corr(r, F) 3 s | flow ACF 3 s | |r| ACF 3 s (proxy) | VR 3 s -> 1 min | VR 1 -> 5 min |
|---|---|---|---|---|---|---|
| major | 6.481 / 5.293 | 0.384 / 0.396 | 0.024 / 0.011 | 0.093 / 0.062 | 0.959 / 1.047 | 1.105 / 1.139 |
| alt | 8.903 / 8.566 | 0.354 / 0.331 | 0.004 / 0.005 | 0.023 / 0.022 | 1.050 / 1.071 | 1.049 / 1.042 |
| tradfi | 6.309 / 5.989 | 0.204 / 0.200 | 0.003 / 0.005 | 0.079 / 0.063 | 1.037 / 1.001 | 0.963 / 0.955 |

## The extrapolated instrument: TAOUSDT built only from params_for(W, class, tick)

TAOUSDT is the alt with the median W. Its generator was built from the other 50 symbols only (params_for with its W, class, tick and FIT price; impact, noise and coupling re-calibrated to the population-predicted targets). It reproduces the spread median (0.447 vs 0.457 bp), the impact correlation (0.35 vs 0.30, inside its own envelope) and the kurtosis (11.6, inside), like its own fitted v1; neither reproduces its |r| ACF (0.04 vs 0.12) or its negative flow ACF. The regime is predicted wrong (tick-bound; TAO is free-spread 14% of the time), which shows in the spread p90 (0.45 vs 0.91 bp). The order rate is under-predicted by a factor 2 (log rate -1.48 vs -0.77): TAO trades more than its W suggests.

| fact | real TEST | own envelope | own v1 (fitted on its FIT half) | population v1 (never saw the symbol) | v0 |
|---|---|---|---|---|---|
| excess kurtosis of 3 s returns | 16.709 | [5.428, 39.314] | 8.762 | 11.642 | 25.955 |
| mean ACF of |r|, lags 1-60 | 0.122 | [0.077, 0.166] | 0.045 | 0.037 | 0.090 |
| mean ACF of signed flow, lags 1-20 | -0.004 | [-0.007, -0.002] | 0.002 | 0.003 | 7.32e-04 |
| impact corr(r_t, F_t) | 0.303 | [0.221, 0.411] | 0.352 | 0.346 | 0.373 |
| spread median (bp) | 0.457 | [0.457, 0.464] | 0.457 | 0.447 | 0.524 |
| spread p90 (bp) | 0.908 | [0.630, 0.922] | 0.756 | 0.450 | 0.611 |
| continuation ratio (extra) | 0.573 | [0.556, 0.588] | 0.501 | 0.526 | 0.511 |

| parameter | actual (FIT) | predicted from the other 50 |
|---|---|---|
| log_rate | -0.773 | -1.484 |
| log_spread_bps | -0.794 | -0.798 |
| regime | free | tick_bound |
| log_sigma_day | -3.510 | -3.460 |
| log_v_ref | 3.328 | 4.389 |
| sign_gamma | 0.988 | 0.750 |
| zeta_order | 1.257 | 1.420 |

Leave-one-out prediction errors over all 51 symbols (predicted minus actual, log units; median absolute error and the factor it implies): log_rate 0.25 (x1.29); log_spread_bps 9.04e-04 (x1.00); log_sigma_day 0.21 (x1.23); log_v_ref 0.33 (x1.39); spread regime right for 38/51.

## Market-making replay on v1 paths (descriptive, not a verdict)

Same engine and fill model as H-SIM4 (touch model, theta0 0.45), 50 mm_multi symbols, real = flow_3s TEST half, v1 = 3 paths per symbol with the instrument's tick at the TEST price. v1 reproduces the activity side of quoting almost exactly (turnover and fills rank at 0.995; fills within about 20% of real) and moves the P&L ranking from -0.09 (v0) to 0.30, but the levels are off: every set loses on v1, because the generator has no drift and the real TEST half was a sell-off in which bid-wide / ask-tight quoting earned. Not a verdict; H-SIM4 and H-MM3 remain the registered tests.

| parameters | real bp of turnover | v1 bp | v0 bp (H-SIM4) | real P&L USD | v1 P&L USD | real turnover | v1 turnover | real fills | v1 fills | real symbols > 0 | v1 symbols > 0 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| live mm_multi | 8.239 | -10.303 | -6.113 | 0.408 | -0.372 | 494.702 | 360.716 | 83.000 | 65.667 | 30 | 26 |
| replay-best a0.2_b0.9_tp0.9_rq0.15 | 11.812 | -1.027 | -4.468 | 4.182 | -0.286 | 3540.302 | 2786.616 | 502.000 | 416.333 | 44 | 25 |

Over the 36-set grid (50 symbols, 3 seeds): Spearman(real, v1) of net bp per turnover 0.301 (v0 in H-SIM4: -0.090), of turnover 0.995, of fill counts 0.996; the real-best set ranks 5 of 36 on v1.

## What a multi-day pilot would most improve

- Separate the regime from the parameters. With one day the FIT-vs-TEST difference mixes estimation noise with the CPI regime break, so the universality test (D) and every 'level' failure are confounded. Five to ten pilot days give a symbol x day panel: between-symbol variance, within-symbol day-to-day variance and estimation noise become three separate components, and D can be tested against the day-to-day variance that a generator must reproduce.
- Book data for every symbol, every day. Only 12 of 51 symbols have an ob200 book, and none on the FIT half of the shape targets: the price-process calibration had to borrow book-mid shape from flow_3s of 2026-11-07..09 and the spread from the same days. Recording ob200 (or at least best bid / ask with sizes) for all quoted symbols removes the trade mid proxy, whose staleness was the main calibration problem.
- An intraday activity profile. One day cannot identify an hour-of-day shape, so the FIT-calibrated generator has the quiet morning's level and the afternoon (CPI, US open) has twice the activity. Several days give an hour-of-week intensity profile (features.seasonality) that the generator can carry as a deterministic multiplier of m(t).
- A wider cross-section for the population model: the 51 symbols include only 5 majors and 13 free-spread perps, so the class effects for majors and the free-spread level model rest on a handful of points. Adding the remaining ~50 collector symbols' trades for the same days would roughly double the free-spread sample.
- Day-to-day envelopes. The H-SIM2 envelope was a range over three days; here it is a single half-day bootstrap interval, which is much tighter (spread p90 of tick-bound majors is a single histogram bin). Multi-day envelopes would test the generator against the variability a strategy will actually face.

## Caveats

- One day (the CPI day), one market state (a crypto sell-off). Every number is a first measurement, confidence C.
- No FIT-half book for 39 of 51 symbols and no FIT-half flow_3s at all (the collector was down 00-12 UTC on 2026-11-10): the level of the price process comes from FIT trades (1-min sigma of the trade mid proxy), its shape (3 s flow-return correlation, variance ratios) and the spread from flow_3s 2026-11-07..09, which is before TEST but not the FIT half. v0 was calibrated on 2026-11-08/09 alone, so the comparison is between two pre-TEST calibrations, not two FIT-half calibrations.
- The v1 design was iterated on five symbols (SOL, CARV, AVAX, NVDA, BTC) with their TEST facts visible (diary 'mistake' entry). The bias runs toward support; the verdict as registered is still applied.
- The registered envelope (95% bootstrap interval of the TEST class median over one half-day) is narrow for discrete facts: the spread median and p90 of tick-bound majors sit in one or two histogram bins (p90 envelope [0.950, 0.950] bp), so a 2-4% difference in price level (tick / price, the sell-off lowered prices) fails the fact. The TEST-level variant moves the majors' median from 0.915 to 0.943 bp, across the envelope, so the binary level/shape rule labels it 'shape'; read it as discreteness.
- Signs are generated with p_buy = 0.5 (no drift) on purpose: the generator is stationary. A sell-off day's directional flow (TEST net selling) is outside its scope, which matters for skewed quoting rules (H-SIM4 finding).
- The impact exponent is fixed at 0.5 and the jump profile at 20 s; kurtosis is not a calibration target. Per-order impact-curve exponents measured on FIT trades are 0.1-0.7 and are reported, not used.
- Kyle-Obizhaeva slopes use orders (same-ms same-side prints merged), not bets; metaorders are not identified, so invariance is a rough check.
- The universality statistic D uses FIT vs TEST as two independent estimates. If parameters shift with the regime by different amounts per symbol, the 'noise' is overstated and D understated; the observed D is well above 1 nonetheless.
- The replay comparison is descriptive (H-SIM4 and H-MM3 are the registered tests). Generated paths have no drift by construction; the real TEST half was a sell-off.

## Rule candidates

```yaml
- id: R-SIM-10
  kind: expect
  rule: "WHEN estimating the order arrival rate, book depth or price impact of a Bybit perp that has not been recorded THEN expect them to follow trading activity W (sigma_day x dollar volume): rate ~ W^0.54, depth within 25 bp ~ W^0.68, lambda / sigma^2 ~ W^-0.75, with about a 1.3x typical error"
  when: "instrument with known daily volatility and dollar volume, no trade or book history"
  then: "use sim.population.Population.params_for(W, class, tick) for the rate, depth and impact level; treat them as +-30% estimates"
  scope: {assets: all, horizon: "1s-1d", sides: both}
  effect: "LOO R^2 rate 0.85, impact 0.64, depth 0.66 over 51 perps; LOO median error rate x1.29, 1-min sigma x1.23"
  confidence: C
  evidence: [H-MS7]
  review: "refit on >= 5 pilot days; LOO R^2 > 0.5 for rate and impact on each day"
  status: provisional
- id: R-SIM-11
  kind: avoid
  rule: "WHEN simulating order-sign memory or the order-size tail for a specific perp THEN avoid a universal gamma or tail index; use that perp's own estimates, and the class median only as a wide prior"
  when: "building or transferring a flow generator"
  then: "measure gamma and the order size tail index per symbol (C1 params); class medians carry an uncertainty of about +-0.2 (gamma) and +-0.3 (tail index)"
  scope: {assets: all, horizon: "1-1000 orders", sides: both}
  effect: "between-symbol dispersion 4.9x (gamma) and 2.4x (tail index) the FIT-vs-TEST noise; neither predicted by W or class (LOO R^2 0.19, -0.16)"
  confidence: C
  evidence: [H-MS7, H-MS2, H-MS3]
  review: "multi-day panel: D against day-to-day variance"
  status: provisional
- id: R-SIM-12
  kind: prefer
  rule: "WHEN predicting the quoted spread of a perp THEN prefer the tick size first (most perps sit at one tick) and, for free-spread perps, a model in log W and log tick; avoid predicting it from activity alone"
  when: "spread input for a simulator, a cost model or quote widths of a new symbol"
  then: "spread = about one tick when the tick is large relative to the natural spread; otherwise log spread ~ -0.19 log W + 1.06 log tick"
  scope: {assets: all, horizon: "tick", sides: both}
  effect: "spread on log W and class LOO R^2 0.28; free-spread level on log W and log tick LOO R^2 0.85 (12 symbols); 39 of 51 perps tick-bound"
  confidence: C
  evidence: [H-MS7, H-MS5]
  review: "book data for all quoted symbols on >= 5 days"
  status: provisional
- id: R-SIM-13
  kind: prefer
  rule: "WHEN testing a deterministic execution or quoting rule on generated order flow THEN prefer the event-level generator v1 over v0, and use it for activity questions (fills, turnover, inventory) rather than for net P&L rankings"
  when: "synthetic flow for sim.replay or strategy research (never for the LLM layer)"
  then: "sim.flow_sim_v1 paths; rank parameter sets by fills / turnover on v1, confirm P&L on real replays"
  scope: {assets: all, horizon: "3s-1d", sides: both}
  effect: "TEST facts inside: v1 3-4/5 vs v0 0-2/5 per class; replay rank agreement with real: turnover 0.995, net bp 0.30 (v0 -0.09)"
  confidence: C
  evidence: [H-MS8, H-SIM2, H-SIM4]
  review: "a generator with an intraday profile passing >= 4/5 in every class on multi-day envelopes"
  status: provisional
- id: R-SIM-14
  kind: avoid
  rule: "WHEN a generator is calibrated on a quiet session THEN avoid using it to judge behaviour in a news session; its volatility clustering and tails stay at the quiet level, and feeding it the busy session's levels does not fix the shape"
  when: "FIT window and target window differ in regime (news, US open, CPI)"
  then: "calibrate per session or regime; evaluate news-session rules on real replays"
  scope: {assets: all, horizon: "3s-1h", sides: both}
  effect: "alts |r| ACF 0.019 simulated vs 0.088 real on the CPI half; with TEST levels the majors' |r| ACF overshoots (0.136 vs <= 0.118) and alts' kurtosis 14.5 vs <= 12.5"
  confidence: C
  evidence: [H-MS8, H-MS1]
  review: "hour-of-week intensity profile from >= 5 days"
  status: provisional
```
