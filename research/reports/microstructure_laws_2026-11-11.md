# Trade-level laws of trading activity: durations, sizes, sign memory, volume, ML durations (H-MS1, H-MS2, H-MS3, H-MS4, H-MS9)

_data: trades_raw 2026-11-10 (every Bybit linear print, 51 symbols) | FIT 00:00-12:00 UTC, TEST 12:00-24:00 UTC | segment: clean_holdout | code: src/ai_trader/sim/laws.py, src/ai_trader/sim/l2_features.py, src/ai_trader/research/microstructure_laws_study.py | tests: tests/test_laws.py | params: data/models/microstructure_laws/params_2026-11-10.json (+ .parquet, aft_2026-11-10/)_

Every print of one full UTC day (2026-11-10, the CPI day) for 51 Bybit perps: 6.9k (CARV) to 2.1M (BTC) prints. Laws are fitted on 00-12 UTC and scored on 12-24 UTC, which is a harder half: it holds the 13:30 CPI release and about twice the activity (majors 1.3 -> 2.5 arrival events per second). This is one day, so every number is a first measurement (confidence C).

In short: arrivals are bursty and far from Poisson. A 3-component exponential mixture (or Burr XII) is the best marginal law, but deseasonalised ACD(1,1) residuals stay far from exponential, so H-MS1 fails on its third part. Trade-size tails are heavy (tail index 1.4-1.8) but curved: CSN rejects the pure power law and the log-normal is never worse (H-MS2 fails). Order signs have long memory: a power-law ACF with gamma 0.45-0.75, and the 1 s signed-flow Hurst exponent is 0.53 with a CI above 0.5. That meets the H-MS3 criterion, but it is inconclusive after Holm. Volume is log-normal, persistent and tied to |returns|, but the power-law ACF part fails at 1 minute for too many symbols (H-MS4 just misses: 78% vs > 80%). The xgboost AFT beats the best ACD only for majors (H-MS9 fails).

## Verdicts

| id | registered success criterion | key numbers (TEST half) | p summary / Holm | verdict |
|---|---|---|---|---|
| H-MS1 | exponential rejected for every symbol; non-exponential family wins TEST log-lik for > 80%; ACD residual KS <= half the raw KS for > 80% | KS reject 51/51; non-exp wins 51/51; KS halved 2/51 (median ratio 0.78) | 1.000 / 1.000 | **contradicts** |
| H-MS2 | tail index in [1, 2] and CSN not rejected for > 70%; power law preferred to log-normal (Vuong) for > 50% | in range and not rejected 4/51 (in range 35, not rejected 6); preferred 0/51; median zeta 1.76 | 1.000 / 1.000 | **contradicts** |
| H-MS3 | power law preferred and 0 < gamma < 1 for > 80% of symbols with >= 20k prints; pooled Hurst CI above 0.5 | 44/48 symbols; Hurst (1 s signed notional) 0.531 [0.522, 0.541] | 0.025 / 0.124 | **inconclusive** |
| H-MS4 | heavy law beats gamma, power-law ACF, Spearman(volume, |r|) > 0, at 10 s and 1 m, for > 80% | 40/51 symbols pass all six | 0.685 / 1.000 | **contradicts** |
| H-MS9 | AFT minus best ACD per-duration log-lik > 0 with CI excluding 0 in >= 2 of 3 classes | major +0.071 [+0.063, +0.078]; alt -0.012 [-0.030, +0.012]; tradfi -0.027 [-0.034, -0.019] | 0.851 / 1.000 | **contradicts** |

p summary: intersection-union p of the criterion's parts (one-sided binomial tests of each share against its threshold; bootstrap p for Hurst and AFT), Holm over the five. Supports needs the literal criterion and Holm p < 0.05.

## Data and levels

| class | symbols | median prints | median orders | median events | prints per order (FIT) | events/s FIT | events/s TEST | ms with both sides |
|---|---|---|---|---|---|---|---|---|
| major | 5 | 6.89e+05 | 1.65e+05 | 1.64e+05 | 4.041 | 1.278 | 2.509 | 0.010 |
| alt | 35 | 1.23e+05 | 36814.000 | 36770.000 | 3.130 | 0.294 | 0.602 | 0.003 |
| tradfi | 11 | 77062.000 | 38506.000 | 38461.000 | 1.867 | 0.240 | 0.492 | 0.002 |

Merging matters. A Bybit taker order is printed as one trade per maker it hits, all with the same millisecond and side: majors average 4 prints per order and TradFi perps 1.9. Treating prints as independent orders creates mechanical zero durations and a print-level sign ACF of 0.7-0.84 at lag 1 that comes from splitting, not from behaviour. Only 0.2-1% of milliseconds carry both sides, so arrival events (distinct ms) and orders are nearly the same series.

## Methods

- **data**: trades_raw 2026-11-10 (Bybit linear public archive): event_ts (ms), side (taker), price, size; 51 symbols; DuckDB read of the lake parquet, archive row order kept within a millisecond (file_row_number)
- **split**: FIT = durations / prints with start and end in [00:00, 12:00) UTC; TEST = [12:00, 24:00); robustness TEST-exCPI drops durations or prints starting in [13:00, 16:00); registered verdicts use the full TEST half
- **levels**: order = maximal run of consecutive prints with the same ms and taker side; arrival event = distinct ms; primary levels: H-MS1 events, H-MS2 prints, H-MS3 orders; print durations of 0 ms set to 0.5 ms
- **classes**: major = BTC ETH SOL XRP DOGE; tradfi = CL BZ XAU XAG QQQ NVDA TSLA MSTR COIN CRWV GOOGL; alt = the rest (incl. BNB)
- **H-MS1**: part 1: KS of raw TEST durations vs an exponential with the sample mean, p < 0.01, for EVERY symbol (Anderson-Darling at 1% reported); part 2: exponential, weibull, gamma, lognormal, gengamma, burr12, loglogistic (MLE, loc 0) and 2-3 component exponential mixtures (EM) fitted on FIT, winner = highest TEST log-likelihood (AIC on TEST reported), holds if a non-exponential family wins for > 80% of symbols; part 3: x = d / s with s the causal trailing 30-min mean duration, ACD(1,1) with exponential / Weibull / generalized-gamma unit-mean innovations fitted on FIT (after a 30-min warm-up), filtered through the whole day; residuals e = x / psi on TEST; holds if KS(e vs exponential) <= 0.5 KS(raw TEST durations vs exponential) for > 80% of symbols, using the ACD with the best FIT log-likelihood; supports = all three parts; p = max of the three one-sided binomial p-values (part 1 and 3 vs 0.8, part 2 vs 0.8)
- **H-MS2**: notional = price x size (USD); CSN: x_min minimises KS over 80 candidates (tail sizes geometric from the upper half to 50 points), MLE alpha_pdf, zeta = alpha_pdf - 1; bootstrap p (B = 200 on TEST prints, 50 elsewhere), power law not rejected if p > 0.1 (CSN); Vuong vs a log-normal truncated at x_min, preferred if R > 0 and two-sided p < 0.05; Hill on the same tail; per symbol on TEST prints; supports = zeta in [1, 2] and not rejected for > 70% of symbols AND preferred for > 50%; p = max of binomial p vs 0.7 and 0.5
- **H-MS3**: order signs (+1 buy, -1 sell) per half; ACF lags 1-1000 averaged in log bins (8 per decade); power law = OLS of log ACF on log lag, exponential = OLS of log ACF on lag, over bins whose ACF exceeds the 2/sqrt(n) noise band of that half (rule added after the CARV/KAS smoke test, before the full run: near-zero noisy bins dominated the log MSE); both fitted on FIT, held-out MSE of log ACF on the TEST bins above the TEST band; gamma = power-law fit on TEST; per symbol with >= 20k prints in the day: power preferred (lower held-out MSE) and 0 < gamma < 1; PACF lags 1-50 (Durbin-Levinson) with +-1.96/sqrt(n) bands, count of significant positive lags 11-50 (reported, not in the success line); Hurst = DFA-1 of 1 s signed notional on TEST (scales 10-4320 s), pooled = mean over the >= 20k symbols, 95% CI by bootstrap over symbols; supports = share > 0.8 and the CI excludes 0.5 from above; p = max(binomial p vs 0.8, bootstrap p of mean H <= 0.5)
- **H-MS4**: notional volume per 10 s and per 60 s clock bin; positive bins: gamma, gengamma, lognormal fitted on FIT, TEST log-likelihood (zero bins share reported, identical for all families); distribution holds if max(gengamma, lognormal) > gamma; ACF of log(1 + v / median positive FIT v), lags 1-360 (10 s) / 1-120 (60 s), log bins, power vs exponential as in H-MS3 (FIT fit, TEST held-out MSE); co-movement: Spearman of v and |log return| (last price per bin, carried forward) on TEST > 0; a symbol passes if all three hold at both intervals; supports = > 80% of symbols; p = binomial p vs 0.8
- **H-MS9**: arrival-event durations; xgboost survival:aft (hist, depth 6, eta 0.05, subsample 0.8, early stopping on the last 20% of FIT after warm-up) with distributions normal / logistic / extreme x training scale {0.8, 1.5}; sigma of the density re-estimated by MLE on the validation part; chosen config = best validation log-likelihood, refitted on all of FIT with 1.25x the best rounds; features at the start of each duration: log of the last 5 durations, log trailing 30-min scale, trailing 10 s / 60 s counts, |log return| over the last 10 / 100 events and 10 s / 60 s, signed-notional imbalance of the last 10 events and 60 s, last event sign / log notional / log prints, effective spread proxy |last buy - last sell| / price, L2 spread / touch imbalance / 5 bp depth imbalance / log 5 bp depth (12 ob200 symbols, 250 ms grid, point in time), hour of day; comparator = the ACD(1,1) with the best TEST log-likelihood (conservative); per-duration log-lik difference in seconds units, class value = mean over symbols of the per-symbol mean, 95% CI by a 5-minute block bootstrap shared across symbols (B = 2000); supports = difference > 0 with CI excluding 0 in >= 2 of 3 classes; p = second-smallest class bootstrap p
- **holm**: Holm over the five summary p-values; supports only if the literal criterion holds and the Holm-adjusted p < 0.05; criterion met but Holm p >= 0.05 -> inconclusive; criterion not met -> contradicts

## H-MS1: inter-arrival times

Durations have a coefficient of variation of about 2.1-2.3 (an exponential has 1), and the exponential is rejected for every symbol (median KS 0.33). Out of sample, a non-exponential family wins for every symbol. The 3-component exponential mixture wins for 30 of 51 (mean rank 2.4), then Burr XII (3.0) and Weibull (3.3). The mixture reads as three regimes: about 25-32% of durations are bursts with a mean of about 5 ms, 35-45% have a mean of 0.2-0.4 s, and 25-38% are quiet gaps with a mean of 1.2-4.8 s (class medians, TEST fit). The generalized gamma nests Weibull and gamma but ranks only 5.0 out of sample: its extra shape overfits the FIT half.

Part 3 fails clearly. Causal deseasonalisation (trailing 30-min mean duration) cuts the KS distance only from 0.33 to 0.29. The ACD residuals reach 0.22-0.26: a median ratio of 0.78 (exponential innovations 0.77, Weibull 0.78, generalized gamma 0.78), and only 2 of 51 symbols halve it (0 of 51 without 13-16 UTC). The fitted ACD sits at the integrated boundary (median alpha + beta = 0.99995, alpha about 0.1-0.2), and the unit-mean innovations stay strongly over-dispersed (Weibull k about 0.47-0.61). Conditional means absorb the slow clustering but not the millisecond bursts. A generator needs a burst component (a self-exciting or mixture innovation), not only a moving conditional mean.

| class | n | KS rejects exp | non-exp wins | KS halved | TEST winners | KS raw | KS deseasonalised | KS ACD resid | ratio | CV of durations |
|---|---|---|---|---|---|---|---|---|---|---|
| major | 5 | 5/5 | 5/5 | 1/5 | burr12 2, expmix3 2, lognormal 1 | 0.329 | 0.292 | 0.222 | 0.674 | 2.256 |
| alt | 35 | 35/35 | 35/35 | 0/35 | expmix3 22, gamma 5, lognormal 3, burr12 3, weibull 2 | 0.328 | 0.297 | 0.259 | 0.809 | 2.064 |
| tradfi | 11 | 11/11 | 11/11 | 1/11 | expmix3 6, lognormal 2, burr12 2, weibull 1 | 0.311 | 0.280 | 0.238 | 0.755 | 2.304 |

Mean held-out rank of each family over all symbols (1 = best TEST log-likelihood, event level): expmix3 2.37, burr12 2.96, weibull 3.25, lognormal 3.86, gamma 4.67, gengamma 4.98, loglogistic 6.49, expmix2 7.41, exponential 9.00.

Robustness: without 13:00-16:00 the non-exponential winner share is 51/51 and the KS-halved share 0/51. Print level (0 ms gaps set to 0.5 ms): exponential rejected 51/51, non-exponential winner 51/51 (print-level fits are dominated by the 0.5 ms atom and are not usable as a continuous law; simulate orders, then split them into prints with the prints-per-order pmf).

<details><summary>Per symbol (event level, TEST half)</summary>

| symbol | class | TEST durations | KS raw vs exp | winner | 2nd, 3rd | log-lik gain vs exp per duration | ACD (best FIT) | alpha | beta | innovation shape | KS resid | ratio |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1000PEPEUSDT | alt | 91837 | 0.298 | lognormal | expmix3, burr12 | 0.804 | gengamma | 0.091 | 0.909 | a 0.355, c 1.120 | 0.177 | 0.596 |
| AAVEUSDT | alt | 44384 | 0.355 | expmix3 | burr12, weibull | 1.014 | gengamma | 0.120 | 0.880 | a 0.396, c 0.829 | 0.281 | 0.793 |
| ADAUSDT | alt | 19698 | 0.335 | expmix3 | weibull, burr12 | 1.117 | gengamma | 0.189 | 0.697 | a 0.284, c 1.164 | 0.259 | 0.774 |
| ARBUSDT | alt | 70656 | 0.351 | expmix3 | lognormal, weibull | 1.001 | gengamma | 0.095 | 0.905 | a 0.390, c 0.884 | 0.290 | 0.827 |
| ASTERUSDT | alt | 15661 | 0.357 | expmix3 | weibull, burr12 | 0.982 | gengamma | 0.161 | 0.753 | a 0.390, c 0.892 | 0.308 | 0.862 |
| ATOMUSDT | alt | 6182 | 0.236 | gamma | burr12, weibull | 0.729 | gengamma | 0.071 | 0.905 | a 0.184, c 1.475 | 0.189 | 0.801 |
| AVAXUSDT | alt | 46698 | 0.337 | expmix3 | weibull, burr12 | 0.944 | gengamma | 0.076 | 0.914 | a 0.296, c 1.072 | 0.270 | 0.802 |
| BNBUSDT | alt | 34191 | 0.397 | expmix3 | lognormal, weibull | 1.406 | gengamma | 0.070 | 0.912 | a 0.216, c 1.295 | 0.346 | 0.872 |
| CARVUSDT | alt | 2526 | 0.355 | expmix3 | gamma, gengamma | 1.195 | gengamma | 0.054 | 0.878 | a 0.131, c 1.624 | 0.317 | 0.894 |
| CRVUSDT | alt | 14482 | 0.329 | burr12 | weibull, lognormal | 1.246 | gengamma | 0.083 | 0.917 | a 0.239, c 1.209 | 0.229 | 0.695 |
| DOTUSDT | alt | 29592 | 0.341 | lognormal | weibull, burr12 | 0.941 | gengamma | 0.059 | 0.937 | a 0.234, c 1.374 | 0.251 | 0.737 |
| ENAUSDT | alt | 129195 | 0.363 | expmix3 | lognormal, gengamma | 0.943 | gengamma | 0.032 | 0.968 | a 1.133, c 0.440 | 0.309 | 0.851 |
| ETCUSDT | alt | 12310 | 0.316 | weibull | burr12, gamma | 1.059 | gengamma | 0.063 | 0.908 | a 0.142, c 1.904 | 0.253 | 0.802 |
| FILUSDT | alt | 25998 | 0.319 | expmix3 | burr12, weibull | 0.992 | gengamma | 0.062 | 0.926 | a 0.192, c 1.528 | 0.249 | 0.779 |
| GRAMUSDT | alt | 14724 | 0.328 | expmix3 | burr12, weibull | 0.881 | gengamma | 0.180 | 0.819 | a 0.561, c 0.714 | 0.268 | 0.816 |
| HYPEUSDT | alt | 119666 | 0.309 | expmix3 | lognormal, gengamma | 0.751 | gengamma | 0.725 | 0.275 | a 1.795, c 0.316 | 0.280 | 0.906 |
| INJUSDT | alt | 48499 | 0.301 | expmix3 | weibull, burr12 | 0.827 | gengamma | 0.035 | 0.955 | a 0.195, c 1.518 | 0.259 | 0.863 |
| KASUSDT | alt | 11202 | 0.401 | lognormal | burr12, weibull | 1.344 | gengamma | 0.094 | 0.801 | a 0.198, c 1.470 | 0.332 | 0.828 |
| LINKUSDT | alt | 42427 | 0.378 | expmix3 | lognormal, burr12 | 1.235 | gengamma | 0.071 | 0.924 | a 0.264, c 1.080 | 0.281 | 0.743 |
| LTCUSDT | alt | 18833 | 0.304 | burr12 | weibull, expmix3 | 1.016 | gengamma | 0.159 | 0.790 | a 0.254, c 1.290 | 0.225 | 0.741 |
| MAVIAUSDT | alt | 2583 | 0.406 | expmix3 | gamma, gengamma | 1.757 | gengamma | 0.113 | 0.802 | a 0.128, c 1.696 | 0.370 | 0.911 |
| MONUSDT | alt | 11843 | 0.257 | gamma | weibull, burr12 | 0.891 | gengamma | 0.076 | 0.869 | a 0.146, c 2.079 | 0.214 | 0.833 |
| NEARUSDT | alt | 54082 | 0.260 | burr12 | weibull, lognormal | 0.765 | gengamma | 0.114 | 0.870 | a 0.296, c 1.261 | 0.188 | 0.724 |
| ONDOUSDT | alt | 30817 | 0.264 | expmix3 | gamma, burr12 | 0.761 | gengamma | 0.067 | 0.920 | a 0.260, c 1.417 | 0.187 | 0.708 |
| RENDERUSDT | alt | 8829 | 0.234 | gamma | gengamma, expmix3 | 0.835 | gengamma | 0.100 | 0.839 | a 0.196, c 1.633 | 0.207 | 0.882 |
| SUIUSDT | alt | 44683 | 0.294 | expmix3 | lognormal, weibull | 0.795 | gengamma | 0.110 | 0.883 | a 0.323, c 1.233 | 0.196 | 0.667 |
| TAOUSDT | alt | 48892 | 0.337 | expmix3 | lognormal, weibull | 1.049 | gengamma | 0.045 | 0.946 | a 0.202, c 1.474 | 0.275 | 0.817 |
| TRUMPUSDT | alt | 14253 | 0.266 | expmix3 | gamma, gengamma | 0.837 | gengamma | 0.099 | 0.859 | a 0.271, c 1.242 | 0.231 | 0.866 |
| TRXUSDT | alt | 4931 | 0.218 | gamma | gengamma, burr12 | 0.531 | gengamma | 0.251 | 0.749 | a 0.752, c 0.653 | 0.178 | 0.814 |
| UNIUSDT | alt | 47873 | 0.276 | weibull | burr12, lognormal | 0.728 | gengamma | 0.067 | 0.926 | a 0.290, c 1.222 | 0.208 | 0.752 |
| WIFUSDT | alt | 7601 | 0.272 | gamma | burr12, weibull | 0.988 | gengamma | 0.155 | 0.845 | a 0.312, c 1.009 | 0.170 | 0.625 |
| WLDUSDT | alt | 30250 | 0.268 | expmix3 | gamma, weibull | 0.747 | gengamma | 0.077 | 0.904 | a 0.259, c 1.468 | 0.180 | 0.671 |
| WLFIUSDT | alt | 10344 | 0.402 | expmix3 | burr12, weibull | 1.437 | gengamma | 0.052 | 0.911 | a 0.223, c 1.123 | 0.371 | 0.923 |
| XLMUSDT | alt | 13057 | 0.363 | expmix3 | weibull, burr12 | 1.291 | gengamma | 0.070 | 0.874 | a 0.206, c 1.331 | 0.294 | 0.809 |
| ZECUSDT | alt | 311427 | 0.382 | expmix3 | lognormal, gengamma | 1.081 | gengamma | 0.496 | 0.504 | a 4.980, c 0.184 | 0.321 | 0.839 |
| BTCUSDT | major | 431420 | 0.522 | burr12 | expmix3, lognormal | 1.924 | gengamma | 0.900 | 0.100 | a 1.50e+09, c 1.22e-05 | 0.408 | 0.781 |
| DOGEUSDT | major | 45405 | 0.319 | lognormal | expmix3, burr12 | 0.976 | gengamma | 0.124 | 0.860 | a 0.337, c 1.151 | 0.209 | 0.655 |
| ETHUSDT | major | 343217 | 0.478 | burr12 | expmix3, lognormal | 1.597 | gengamma | 0.801 | 0.199 | a 30.312, c 0.078 | 0.393 | 0.822 |
| SOLUSDT | major | 90290 | 0.305 | expmix3 | lognormal, burr12 | 0.793 | gengamma | 0.106 | 0.894 | a 0.497, c 0.961 | 0.135 | 0.444 |
| XRPUSDT | major | 108376 | 0.329 | expmix3 | lognormal, weibull | 0.845 | gengamma | 0.077 | 0.923 | a 0.376, c 1.059 | 0.222 | 0.674 |
| BZUSDT | tradfi | 20571 | 0.291 | burr12 | weibull, gamma | 0.559 | gengamma | 0.089 | 0.900 | a 0.307, c 1.301 | 0.228 | 0.786 |
| CLUSDT | tradfi | 33453 | 0.276 | expmix3 | burr12, weibull | 0.672 | gengamma | 0.220 | 0.780 | a 0.911, c 0.579 | 0.225 | 0.818 |
| COINUSDT | tradfi | 9128 | 0.336 | burr12 | weibull, lognormal | 1.174 | gengamma | 0.104 | 0.883 | a 0.262, c 1.213 | 0.239 | 0.711 |
| CRWVUSDT | tradfi | 33956 | 0.385 | expmix3 | gengamma, weibull | 1.327 | gengamma | 0.161 | 0.839 | a 0.773, c 0.604 | 0.301 | 0.782 |
| GOOGLUSDT | tradfi | 18953 | 0.314 | expmix3 | burr12, weibull | 1.154 | gengamma | 0.052 | 0.948 | a 0.205, c 1.573 | 0.242 | 0.773 |
| MSTRUSDT | tradfi | 43409 | 0.348 | lognormal | weibull, burr12 | 1.205 | gengamma | 0.069 | 0.931 | a 0.232, c 1.333 | 0.263 | 0.755 |
| NVDAUSDT | tradfi | 94042 | 0.286 | lognormal | weibull, burr12 | 0.905 | gengamma | 0.097 | 0.903 | a 0.367, c 1.006 | 0.206 | 0.720 |
| QQQUSDT | tradfi | 21275 | 0.306 | expmix3 | weibull, burr12 | 0.938 | gengamma | 0.169 | 0.831 | a 0.628, c 0.721 | 0.227 | 0.742 |
| TSLAUSDT | tradfi | 18333 | 0.311 | weibull | burr12, gamma | 0.988 | gengamma | 0.123 | 0.877 | a 0.425, c 0.846 | 0.238 | 0.763 |
| XAGUSDT | tradfi | 13040 | 0.271 | expmix3 | gamma, gengamma | 0.684 | gengamma | 0.160 | 0.840 | a 0.661, c 0.817 | 0.114 | 0.419 |
| XAUUSDT | tradfi | 45482 | 0.414 | expmix3 | lognormal, gengamma | 1.155 | gengamma | 0.325 | 0.675 | a 1.602, c 0.410 | 0.277 | 0.668 |

</details>

## H-MS2: trade-size tails

Print notional has a heavy upper tail. The CSN tail index zeta (CCDF exponent; CSN pdf alpha = zeta + 1) has medians of 1.84 for majors, 1.76 for alts and 1.39 for TradFi, and 35 of 51 symbols fall in [1, 2]. At order level the tails are heavier (zeta 1.1-1.4), because merging the prints of one order rebuilds large orders. But the pure power law is rejected by the bootstrap for 45 of 51 symbols (p <= 0.1). The Vuong test never prefers it to a log-normal truncated at x_min (0 of 51); the log-normal is significantly better or tied. With tens of thousands of tail points, any curvature in log-log space is detected. Lot-size discreteness adds to the KS distance (the five most common sizes are 6-8% of the tail for liquid symbols, 22% for KAS) but does not explain the Vuong result. The index is also unstable across halves (Spearman 0.27 between FIT and TEST zeta, median absolute change 0.33). For simulation, use the spliced law (log-normal body, Pareto tail above x_min) per symbol and half. Treat zeta as roughly 1.3-1.8 with a large uncertainty, not as a universal 1.5.

| class | n | zeta (print, TEST) | Hill | tail share | zeta in [1, 2] | CSN p > 0.1 | Vuong prefers PL | zeta (order) | order: in range and not rejected | order: PL preferred |
|---|---|---|---|---|---|---|---|---|---|---|
| major | 5 | 1.844 | 1.843 | 0.007 | 4/5 | 1/5 | 0/5 | 1.376 | 0/5 | 0/5 |
| alt | 35 | 1.759 | 1.759 | 0.033 | 23/35 | 3/35 | 0/35 | 1.289 | 4/35 | 0/35 |
| tradfi | 11 | 1.391 | 1.391 | 0.044 | 8/11 | 2/11 | 0/11 | 1.111 | 1/11 | 0/11 |

FIT-vs-TEST stability (print level): Spearman of zeta across symbols 0.271, median |zeta_FIT - zeta_TEST| 0.332.

<details><summary>Per symbol</summary>

| symbol | class | zeta TEST | zeta FIT | zeta ex-CPI | x_min USD | n tail | KS | CSN p | Vuong R | Vuong p | Hill | zeta order | CSN p order | median USD | p99 USD |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1000PEPEUSDT | alt | 2.012 | 1.906 | 0.961 | 7271.620 | 1438 | 0.046 | 0.000 | 1.77e-04 | 0.744 | 2.011 | 1.375 | 0.000 | 13.185 | 5549.675 |
| AAVEUSDT | alt | 1.528 | 1.454 | 1.539 | 1989.120 | 1557 | 0.060 | 0.000 | -8.932 | 0.016 | 1.528 | 1.004 | 0.000 | 19.602 | 2270.752 |
| ADAUSDT | alt | 1.846 | 1.527 | 0.977 | 8729.432 | 162 | 0.082 | 0.000 | -0.019 | 0.897 | 1.835 | 1.340 | 0.000 | 17.346 | 5904.032 |
| ARBUSDT | alt | 1.705 | 1.824 | 1.694 | 654.388 | 5962 | 0.051 | 0.000 | -2.944 | 0.295 | 1.705 | 1.389 | 0.020 | 7.508 | 1208.368 |
| ASTERUSDT | alt | 1.586 | 1.926 | 1.638 | 275.440 | 4486 | 0.060 | 0.000 | -0.129 | 0.772 | 1.584 | 1.289 | 0.520 | 12.289 | 1001.009 |
| ATOMUSDT | alt | 2.592 | 1.091 | 2.281 | 1501.731 | 396 | 0.049 | 0.010 | -0.065 | 0.860 | 2.591 | 0.873 | 0.000 | 15.994 | 1985.544 |
| AVAXUSDT | alt | 1.733 | 2.020 | 1.795 | 623.484 | 8759 | 0.033 | 0.000 | -52.091 | 3.11e-06 | 1.732 | 1.100 | 0.000 | 20.565 | 1701.223 |
| BNBUSDT | alt | 1.457 | 1.844 | 1.539 | 1013.371 | 3915 | 0.026 | 0.000 | -0.239 | 0.658 | 1.457 | 1.078 | 0.000 | 6.870 | 2872.399 |
| CARVUSDT | alt | 1.698 | 1.354 | 1.960 | 297.032 | 86 | 0.061 | 0.270 | -0.264 | 0.679 | 1.696 | 1.726 | 0.260 | 35.384 | 509.405 |
| CRVUSDT | alt | 1.976 | 2.762 | 2.094 | 2002.024 | 201 | 0.054 | 0.085 | 2.75e-05 | 0.869 | 1.975 | 1.305 | 0.040 | 12.869 | 1509.172 |
| DOTUSDT | alt | 1.759 | 2.220 | 2.201 | 807.935 | 928 | 0.045 | 0.000 | -4.429 | 0.075 | 1.759 | 1.280 | 0.000 | 7.536 | 825.770 |
| ENAUSDT | alt | 2.283 | 2.608 | 2.944 | 6654.468 | 756 | 0.037 | 0.015 | -4.384 | 0.033 | 2.278 | 0.957 | 0.000 | 7.619 | 1831.442 |
| ETCUSDT | alt | 1.582 | 1.066 | 1.441 | 188.326 | 2844 | 0.041 | 0.000 | 3.62e-04 | 0.310 | 1.582 | 1.253 | 0.000 | 14.755 | 837.303 |
| FILUSDT | alt | 1.343 | 1.659 | 1.421 | 187.488 | 3493 | 0.039 | 0.000 | -19.585 | 1.83e-04 | 1.343 | 1.332 | 0.020 | 7.499 | 680.869 |
| GRAMUSDT | alt | 1.592 | 2.418 | 1.337 | 1129.930 | 2229 | 0.036 | 0.000 | -8.588 | 0.008 | 1.591 | 1.254 | 0.000 | 7.725 | 2245.058 |
| HYPEUSDT | alt | 1.756 | 1.379 | 2.180 | 5493.990 | 3387 | 0.030 | 0.000 | -4.265 | 0.067 | 1.755 | 0.987 | 0.000 | 18.460 | 4687.554 |
| INJUSDT | alt | 1.264 | 1.320 | 1.300 | 398.245 | 8079 | 0.044 | 0.000 | -15.651 | 2.09e-04 | 1.264 | 1.021 | 0.000 | 8.083 | 1632.148 |
| KASUSDT | alt | 2.452 | 1.522 | 1.692 | 620.880 | 818 | 0.076 | 0.000 | -7.975 | 0.215 | 2.446 | 1.515 | 0.000 | 10.374 | 1040.703 |
| LINKUSDT | alt | 1.690 | 1.808 | 1.782 | 1018.067 | 5765 | 0.046 | 0.000 | -10.519 | 0.017 | 1.690 | 1.091 | 0.000 | 10.419 | 2480.887 |
| LTCUSDT | alt | 1.877 | 2.489 | 2.008 | 996.268 | 3580 | 0.040 | 0.000 | -4.442 | 0.141 | 1.877 | 1.259 | 0.000 | 20.582 | 2803.193 |
| MAVIAUSDT | alt | 2.114 | 2.349 | 2.154 | 109.816 | 606 | 0.056 | 0.000 | -0.508 | 0.539 | 2.112 | 1.414 | 0.040 | 32.914 | 434.370 |
| MONUSDT | alt | 1.852 | 2.627 | 2.049 | 672.561 | 1384 | 0.046 | 0.000 | -6.202 | 0.230 | 1.852 | 1.040 | 0.000 | 19.538 | 1547.744 |
| NEARUSDT | alt | 2.164 | 1.493 | 2.002 | 15103.058 | 354 | 0.039 | 0.255 | 3.96e-05 | 0.918 | 2.158 | 1.303 | 0.000 | 15.428 | 8646.331 |
| ONDOUSDT | alt | 2.461 | 2.380 | 2.696 | 4111.866 | 377 | 0.062 | 0.000 | -0.577 | 0.559 | 2.456 | 1.841 | 0.000 | 14.944 | 3295.444 |
| RENDERUSDT | alt | 3.209 | 1.554 | 3.241 | 1589.362 | 497 | 0.053 | 0.000 | -0.968 | 0.291 | 3.205 | 1.456 | 0.000 | 15.455 | 1947.427 |
| SUIUSDT | alt | 1.991 | 1.756 | 1.767 | 1475.264 | 6370 | 0.038 | 0.000 | -3.946 | 0.180 | 1.991 | 1.456 | 0.020 | 16.116 | 3229.776 |
| TAOUSDT | alt | 1.199 | 1.607 | 1.201 | 166.635 | 17777 | 0.058 | 0.000 | -177.178 | 1.25e-26 | 1.199 | 1.167 | 0.000 | 7.543 | 1289.246 |
| TRUMPUSDT | alt | 1.914 | 1.658 | 1.993 | 1640.742 | 1543 | 0.068 | 0.000 | -10.715 | 0.025 | 1.910 | 1.637 | 0.140 | 31.300 | 3208.134 |
| TRXUSDT | alt | 2.876 | 0.829 | 0.751 | 3478.784 | 432 | 0.071 | 0.000 | -0.399 | 0.685 | 2.876 | 0.547 | 0.000 | 80.069 | 5129.907 |
| UNIUSDT | alt | 2.168 | 1.167 | 1.239 | 4446.288 | 320 | 0.050 | 0.010 | 3.96e-05 | 0.783 | 2.157 | 1.381 | 0.000 | 8.653 | 2458.003 |
| WIFUSDT | alt | 0.913 | 1.020 | 0.981 | 161.659 | 3900 | 0.056 | 0.000 | -49.684 | 3.48e-11 | 0.913 | 1.514 | 0.000 | 10.208 | 2813.148 |
| WLDUSDT | alt | 2.197 | 1.576 | 1.726 | 6285.005 | 241 | 0.037 | 0.680 | -0.197 | 0.692 | 2.189 | 1.761 | 0.540 | 7.532 | 3677.440 |
| WLFIUSDT | alt | 1.389 | 1.286 | 1.483 | 241.275 | 2338 | 0.055 | 0.000 | -6.826 | 0.063 | 1.388 | 1.072 | 0.000 | 14.969 | 1222.806 |
| XLMUSDT | alt | 1.262 | 1.476 | 1.349 | 900.564 | 2534 | 0.051 | 0.000 | -48.143 | 3.01e-11 | 1.255 | 2.447 | 0.100 | 28.820 | 4197.043 |
| ZECUSDT | alt | 1.153 | 1.924 | 2.138 | 1994.726 | 35021 | 0.037 | 0.000 | -110.169 | 1.52e-19 | 1.153 | 0.909 | 0.000 | 31.321 | 6393.426 |
| BTCUSDT | major | 1.420 | 1.580 | 1.478 | 18426.971 | 40866 | 0.024 | 0.000 | -10.583 | 0.003 | 1.420 | 0.819 | 0.000 | 219.489 | 36095.352 |
| DOGEUSDT | major | 1.930 | 2.118 | 1.647 | 11861.241 | 275 | 0.045 | 0.145 | 3.30e-05 | 0.804 | 1.930 | 1.376 | 0.020 | 10.266 | 5377.258 |
| ETHUSDT | major | 1.844 | 2.219 | 1.980 | 65557.065 | 1816 | 0.025 | 0.000 | 2.20e-04 | 0.646 | 1.843 | 1.250 | 0.000 | 20.931 | 22733.630 |
| SOLUSDT | major | 2.047 | 1.715 | 2.170 | 17928.064 | 2675 | 0.019 | 0.020 | -0.217 | 0.705 | 2.047 | 1.665 | 0.000 | 16.968 | 15003.065 |
| XRPUSDT | major | 1.779 | 1.769 | 1.937 | 6655.788 | 7578 | 0.045 | 0.000 | -20.349 | 0.028 | 1.778 | 1.604 | 0.020 | 9.425 | 8721.167 |
| BZUSDT | tradfi | 1.178 | 1.210 | 1.293 | 1015.084 | 2356 | 0.022 | 0.000 | -0.041 | 0.856 | 1.178 | 1.124 | 0.000 | 39.987 | 4343.230 |
| CLUSDT | tradfi | 1.810 | 2.383 | 2.011 | 24936.336 | 126 | 0.064 | 0.155 | -0.570 | 0.426 | 1.808 | 1.387 | 0.100 | 33.916 | 8970.609 |
| COINUSDT | tradfi | 0.974 | 1.437 | 1.011 | 148.781 | 1385 | 0.028 | 0.000 | -5.695 | 0.017 | 0.974 | 0.927 | 0.000 | 13.004 | 1939.056 |
| CRWVUSDT | tradfi | 2.117 | 1.085 | 0.912 | 3683.720 | 452 | 0.062 | 0.000 | -0.398 | 0.568 | 2.088 | 1.928 | 0.000 | 40.842 | 2571.233 |
| GOOGLUSDT | tradfi | 1.477 | 1.001 | 1.446 | 1733.303 | 1301 | 0.048 | 0.000 | -2.264 | 0.145 | 1.476 | 1.019 | 0.000 | 95.806 | 4311.262 |
| MSTRUSDT | tradfi | 1.308 | 1.425 | 1.407 | 1000.933 | 3930 | 0.039 | 0.000 | -12.885 | 0.003 | 1.308 | 1.061 | 0.000 | 15.954 | 2970.922 |
| NVDAUSDT | tradfi | 1.089 | 1.176 | 1.111 | 2053.220 | 9681 | 0.048 | 0.000 | -62.713 | 1.52e-13 | 1.089 | 1.289 | 0.000 | 43.512 | 8742.785 |
| QQQUSDT | tradfi | 0.776 | 0.892 | 0.785 | 726.840 | 5462 | 0.066 | 0.000 | -58.876 | 1.76e-11 | 0.776 | 0.827 | 0.000 | 101.272 | 15079.811 |
| TSLAUSDT | tradfi | 1.770 | 1.046 | 1.358 | 8702.545 | 206 | 0.048 | 0.270 | -0.047 | 0.867 | 1.766 | 1.108 | 0.000 | 48.918 | 5101.478 |
| XAGUSDT | tradfi | 1.462 | 1.360 | 1.452 | 2828.544 | 2776 | 0.088 | 0.000 | -6.190 | 0.082 | 1.461 | 1.325 | 0.240 | 18.167 | 9900.945 |
| XAUUSDT | tradfi | 1.391 | 1.570 | 1.491 | 3957.977 | 4261 | 0.033 | 0.000 | -0.466 | 0.581 | 1.391 | 1.111 | 0.000 | 64.619 | 11916.408 |

</details>

## H-MS3: sign memory and signed flow

Order signs persist strongly: ACF at lag 1 is 0.33-0.39 in every class, about 0.04-0.09 at lag 10 and 0.01-0.05 at lag 100. A power law predicts the held-out correlogram better than an exponential for every eligible symbol (48 of 48). The fitted gamma is 0.64 for majors, 0.74 for alts and 0.45 for TradFi (longest memory). It is in (0, 1) for 44 of 48. The four failures (AAVE, DOT, FIL, ZEC) have gamma 1.00-1.09, just outside. 39 of 48 symbols have 4 or more significant positive PACF lags in 11-50, so the memory is not an AR(1). In clock time the memory is diluted by size heterogeneity: 1 s signed notional has a lag-1 ACF of only 0.03-0.04, but DFA gives Hurst 0.531 [0.522, 0.541] for notional and 0.579 [0.568, 0.590] for signed counts (every symbol above 0.5). The literal criterion is met (44/48 = 0.92 > 0.8; Hurst CI above 0.5). The share is not significantly above 0.8 (binomial p 0.025, Holm 0.12), so the pre-set rule gives inconclusive. Without 13-16 UTC the share drops to 37 of 48 (more gammas just above 1), while at print level it is 47 of 48.

| class | n (>= 20k prints) | sign ACF 1 | ACF 10 | ACF 100 | ACF 1000 | gamma | power preferred | criterion | PACF + lags 11-50 | print ACF 1 | print gamma | Hurst 1 s notional | Hurst 1 s count | 1 s flow ACF 1 | order notional ACF 1 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| major | 5 | 0.329 | 0.041 | 0.013 | 0.003 | 0.640 | 5/5 | 5/5 | 16.000 | 0.841 | 0.570 | 0.583 | 0.585 | 0.036 | 0.042 |
| alt | 33 | 0.391 | 0.044 | 0.009 | 0.001 | 0.739 | 33/33 | 29/33 | 6.000 | 0.803 | 0.655 | 0.534 | 0.573 | 0.028 | 0.046 |
| tradfi | 10 | 0.358 | 0.087 | 0.046 | 0.012 | 0.447 | 10/10 | 10/10 | 21.000 | 0.699 | 0.605 | 0.526 | 0.568 | 0.042 | 0.051 |

Pooled Hurst (DFA of 1 s signed notional, TEST, 48 symbols): mean 0.531, 95% CI [0.522, 0.541], share above 0.5 0.83; signed COUNT: mean 0.579 [0.568, 0.590], share above 0.5 1.00. Ex-CPI mean Hurst 0.534. Order-sign criterion without 13-16 UTC: 37/48; at print level: 47/48. PACF with >= 4 significant positive lags in 11-50 (binomial 2% under white noise): 39/48.

<details><summary>Per symbol (order signs, TEST)</summary>

| symbol | class | TEST orders | ACF 1 | ACF 10 | ACF 100 | gamma TEST | gamma FIT | held-out MSE power | MSE exp | power preferred | PACF + 11-50 | print gamma | Hurst notional | Hurst count |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1000PEPEUSDT | alt | 92991 | 0.283 | 0.052 | 0.009 | 0.625 | 0.705 | 0.023 | 0.372 | yes | 23 | 0.551 | 0.549 | 0.635 |
| AAVEUSDT | alt | 44610 | 0.479 | 0.052 | 0.008 | 1.030 | 0.639 | 0.554 | 1.498 | yes | 4 | 0.590 | 0.550 | 0.589 |
| ADAUSDT | alt | 19902 | 0.202 | 0.057 | 0.008 | 0.518 | 0.592 | 0.088 | 3.488 | yes | 5 | 0.732 | 0.517 | 0.536 |
| ARBUSDT | alt | 70882 | 0.470 | 0.050 | 0.013 | 0.941 | 0.988 | 0.190 | 7.564 | yes | 3 | 0.649 | 0.561 | 0.623 |
| ASTERUSDT | alt | 15759 | 0.429 | 0.090 | 0.043 | 0.418 | 0.545 | 0.178 | 1.046 | yes | 10 | 0.615 | 0.522 | 0.549 |
| ATOMUSDT | alt | 6197 | 0.249 | 0.057 | 0.011 | 0.413 | 0.586 | 0.118 | 2.922 | yes | 6 | 0.710 | 0.522 | 0.583 |
| AVAXUSDT | alt | 46916 | 0.418 | 0.033 | 0.015 | 0.837 | 0.770 | 0.115 | 0.793 | yes | 7 | 0.576 | 0.540 | 0.637 |
| BNBUSDT | alt | 34306 | 0.514 | 0.072 | -0.003 | 0.829 | 0.809 | 0.140 | 1.962 | yes | 3 | 0.771 | 0.501 | 0.557 |
| CARVUSDT | alt | 2529 | 0.378 | 0.083 | -8.46e-04 | 0.488 | 0.998 | 0.669 | 24.389 | yes | 5 | 0.562 | 0.459 | 0.567 |
| CRVUSDT | alt | 14525 | 0.339 | 0.032 | 0.015 | 0.639 | 0.995 | 0.391 | 28.024 | yes | 6 | 0.655 | 0.488 | 0.547 |
| DOTUSDT | alt | 29693 | 0.444 | 0.026 | 0.007 | 1.087 | 1.090 | 0.066 | 0.724 | yes | 2 | 0.727 | 0.511 | 0.553 |
| ENAUSDT | alt | 129772 | 0.474 | 0.028 | 0.018 | 0.499 | 0.494 | 0.517 | 1.125 | yes | 6 | 0.631 | 0.579 | 0.629 |
| ETCUSDT | alt | 12323 | 0.398 | 0.037 | 0.020 | 0.881 | 1.113 | 0.206 | 15.119 | yes | 3 | 0.627 | 0.492 | 0.568 |
| FILUSDT | alt | 26034 | 0.464 | 0.044 | 0.006 | 1.057 | 0.975 | 0.051 | 0.379 | yes | 3 | 0.747 | 0.548 | 0.601 |
| GRAMUSDT | alt | 14887 | 0.139 | 0.028 | 0.014 | 0.568 | 0.353 | 0.324 | 0.451 | yes | 7 | 0.658 | 0.535 | 0.549 |
| HYPEUSDT | alt | 120517 | 0.403 | 0.071 | 0.006 | 0.859 | 0.757 | 0.140 | 0.785 | yes | 11 | 0.551 | 0.540 | 0.608 |
| INJUSDT | alt | 48653 | 0.401 | 0.030 | 0.005 | 0.807 | 0.912 | 0.176 | 2.200 | yes | 5 | 0.645 | 0.511 | 0.577 |
| KASUSDT | alt | 11215 | 0.509 | 0.242 | 0.137 | 0.305 | 0.552 | 2.577 | 113.073 | yes | 17 | 0.566 | 0.482 | 0.619 |
| LINKUSDT | alt | 42745 | 0.465 | 0.054 | 0.017 | 0.934 | 1.062 | 0.132 | 3.434 | yes | 2 | 0.536 | 0.553 | 0.578 |
| LTCUSDT | alt | 18882 | 0.376 | 0.049 | 0.002 | 0.640 | 0.841 | 0.136 | 20.847 | yes | 7 | 0.613 | 0.542 | 0.567 |
| MAVIAUSDT | alt | 2589 | 0.403 | 0.047 | -0.017 | 1.019 | 0.951 | 0.198 | 0.192 | no | 2 | 0.683 | 0.528 | 0.524 |
| MONUSDT | alt | 11886 | 0.224 | 0.033 | 0.042 | 0.454 | 0.543 | 0.173 | 1.297 | yes | 5 | 0.651 | 0.502 | 0.583 |
| NEARUSDT | alt | 54804 | 0.255 | 0.040 | 0.007 | 0.724 | 0.679 | 0.162 | 0.451 | yes | 9 | 0.614 | 0.562 | 0.615 |
| ONDOUSDT | alt | 31092 | 0.293 | 0.018 | 0.003 | 0.739 | 0.650 | 0.741 | 1.116 | yes | 5 | 0.643 | 0.543 | 0.548 |
| RENDERUSDT | alt | 8846 | 0.226 | 0.037 | 0.005 | 0.859 | 0.662 | 0.113 | 0.216 | yes | 4 | 0.874 | 0.546 | 0.541 |
| SUIUSDT | alt | 44886 | 0.397 | 0.029 | 0.007 | 0.995 | 0.916 | 0.186 | 0.684 | yes | 2 | 0.771 | 0.520 | 0.566 |
| TAOUSDT | alt | 49004 | 0.477 | 0.069 | 0.009 | 0.718 | 0.988 | 0.410 | 18.600 | yes | 6 | 0.732 | 0.534 | 0.617 |
| TRUMPUSDT | alt | 14317 | 0.210 | 0.048 | 0.007 | 0.455 | 0.427 | 0.271 | 0.580 | yes | 7 | 0.710 | 0.516 | 0.536 |
| TRXUSDT | alt | 4955 | 0.238 | 0.053 | -0.003 | 0.604 | 0.340 | 0.924 | 1.010 | yes | 2 | 0.763 | 0.543 | 0.511 |
| UNIUSDT | alt | 48048 | 0.391 | 0.027 | 0.013 | 0.764 | 0.823 | 0.149 | 1.334 | yes | 10 | 0.686 | 0.546 | 0.597 |
| WIFUSDT | alt | 7620 | 0.206 | 0.014 | -0.004 | 0.868 | 0.526 | 0.432 | 0.589 | yes | 4 | 0.641 | 0.490 | 0.539 |
| WLDUSDT | alt | 30500 | 0.249 | 0.048 | 0.017 | 0.512 | 0.733 | 0.134 | 3.946 | yes | 16 | 0.665 | 0.497 | 0.573 |
| WLFIUSDT | alt | 10361 | 0.343 | 0.039 | 0.032 | 0.779 | 1.191 | 0.458 | 35.901 | yes | 2 | 0.876 | 0.519 | 0.517 |
| XLMUSDT | alt | 13100 | 0.290 | 0.043 | 0.021 | 0.620 | 0.420 | 0.255 | 0.464 | yes | 6 | 0.713 | 0.471 | 0.558 |
| ZECUSDT | alt | 315058 | 0.469 | 0.086 | 0.003 | 1.003 | 0.786 | 0.508 | 2.365 | yes | 11 | 0.660 | 0.549 | 0.660 |
| BTCUSDT | major | 439507 | 0.605 | 0.260 | 0.007 | 0.817 | 0.958 | 0.510 | 4.123 | yes | 15 | 0.693 | 0.583 | 0.546 |
| DOGEUSDT | major | 45942 | 0.297 | 0.038 | 0.010 | 0.640 | 0.743 | 0.084 | 1.927 | yes | 14 | 0.539 | 0.542 | 0.614 |
| ETHUSDT | major | 350410 | 0.441 | 0.093 | 0.017 | 0.587 | 0.822 | 0.425 | 1.389 | yes | 25 | 0.678 | 0.611 | 0.596 |
| SOLUSDT | major | 92302 | 0.242 | 0.041 | 0.019 | 0.548 | 0.535 | 0.251 | 0.494 | yes | 28 | 0.570 | 0.607 | 0.585 |
| XRPUSDT | major | 109863 | 0.329 | 0.040 | 0.013 | 0.704 | 0.528 | 0.344 | 1.033 | yes | 16 | 0.535 | 0.555 | 0.548 |
| BZUSDT | tradfi | 20598 | 0.337 | 0.162 | 0.082 | 0.263 | 0.466 | 1.032 | 5.544 | yes | 21 | 0.184 | 0.542 | 0.676 |
| CLUSDT | tradfi | 33493 | 0.362 | 0.163 | 0.096 | 0.334 | 0.426 | 0.137 | 3.265 | yes | 26 | 0.564 | 0.566 | 0.632 |
| COINUSDT | tradfi | 9133 | 0.378 | 0.057 | -0.004 | 0.641 | 0.510 | 0.295 | 0.577 | yes | 2 | 0.864 | 0.536 | 0.540 |
| CRWVUSDT | tradfi | 34072 | 0.444 | 0.115 | 0.076 | 0.244 | 0.438 | 0.453 | 0.494 | yes | 27 | 0.385 | 0.512 | 0.507 |
| GOOGLUSDT | tradfi | 18971 | 0.341 | 0.065 | 0.016 | 0.497 | 0.446 | 0.049 | 0.623 | yes | 6 | 0.587 | 0.481 | 0.577 |
| MSTRUSDT | tradfi | 43503 | 0.414 | 0.087 | 0.033 | 0.451 | 0.474 | 0.028 | 1.770 | yes | 21 | 0.756 | 0.524 | 0.609 |
| NVDAUSDT | tradfi | 94261 | 0.348 | 0.086 | 0.039 | 0.443 | 0.472 | 0.105 | 0.639 | yes | 35 | 0.566 | 0.576 | 0.636 |
| QQQUSDT | tradfi | 21310 | 0.361 | 0.075 | 0.009 | 0.652 | 0.419 | 1.120 | 1.422 | yes | 5 | 0.734 | 0.528 | 0.551 |
| TSLAUSDT | tradfi | 18357 | 0.354 | 0.074 | 0.053 | 0.425 | 0.471 | 0.118 | 1.180 | yes | 10 | 0.630 | 0.523 | 0.558 |
| XAGUSDT | tradfi | 13119 | 0.146 | 0.066 | 0.012 | 0.481 | 0.360 | 0.181 | 0.285 | yes | 4 | 0.623 | 0.435 | 0.544 |
| XAUUSDT | tradfi | 45855 | 0.437 | 0.244 | 0.064 | 0.537 | 0.461 | 0.162 | 0.434 | yes | 23 | 0.700 | 0.544 | 0.549 |

</details>

## H-MS4: volume per interval

Volume per 10 s and per minute is log-normal-like. Out of sample the log-normal beats the gamma by 0.18-0.65 nats per bin (all 51 at 10 s, 49 at 1 m), and the generalized gamma does nearly as well. The co-movement with |return| in the same bin holds for every symbol (Spearman 0.40-0.55). Log volume is persistent (lag-1 ACF 0.35-0.64) and decays slowly. The power law is the better held-out decay for 46 of 51 at 10 s but only 42 of 51 at 1 minute, where 12 h give only 720 bins and a correlogram truncated at 2 h. 40 of 51 symbols pass all six parts (78%, below the registered 80%), and 31 of 51 without 13-16 UTC. The failures are about ACF shape, not about distribution or co-movement.

| class | bin | n | zero share | log-normal minus gamma (per bin) | gen. gamma minus gamma | heavy beats gamma | ACF 1 of log volume | ACF gamma | power ACF preferred | Spearman vol, |r| | Spearman > 0 | all six |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| major | 10s | 5 | 0.004 | 0.401 | 0.371 | 5/5 | 0.512 | 0.204 | 4/5 | 0.469 | 5/5 | 3/5 |
| major | 60s | 5 | 0.000 | 0.648 | 0.624 | 5/5 | 0.635 | 0.319 | 3/5 | 0.459 | 5/5 | 3/5 |
| alt | 10s | 35 | 0.127 | 0.216 | 0.209 | 35/35 | 0.346 | 0.239 | 34/35 | 0.527 | 35/35 | 30/35 |
| alt | 60s | 35 | 0.000 | 0.270 | 0.266 | 34/35 | 0.482 | 0.310 | 31/35 | 0.396 | 35/35 | 30/35 |
| tradfi | 10s | 11 | 0.155 | 0.216 | 0.214 | 11/11 | 0.387 | 0.162 | 8/11 | 0.551 | 11/11 | 7/11 |
| tradfi | 60s | 11 | 0.001 | 0.178 | 0.182 | 10/11 | 0.580 | 0.244 | 8/11 | 0.457 | 11/11 | 7/11 |

Without 13-16 UTC: 31/51 symbols pass all six.

## H-MS9: machine-learned conditional durations (xgboost AFT) vs ACD

The AFT model is the ML law of the next arrival: log T = mu(x) + sigma Z. For majors it beats the best ACD(1,1) by 0.071 nats per duration [0.063, 0.078] (still 0.063 without 13-16 UTC). The extreme-value (log-Weibull) error distribution wins validation for most symbols. The most useful features are trades in the last 10 s and 60 s, the last durations, and the number of prints in the last order (a big sweep is followed by fast follow-on trades). For alts the difference is -0.012 [-0.030, +0.012], and for TradFi -0.027 [-0.034, -0.019]: with a few thousand to tens of thousands of FIT durations, trees fitted on the quiet morning do not transfer to the busier afternoon better than a 4-parameter ACD. The registered hour-of-day feature is outside its FIT support in TEST; dropping it helps a little (alts -0.003, TradFi -0.016) and changes no class verdict. The three TradFi L2 symbols (NVDA, QQQ, XAU) gain +0.017 [0.010, 0.025], a hint that book features help where trades are sparse. The criterion holds in 1 of 3 classes, so H-MS9 is contradicted.

| class | variant | symbols | mean per-duration log-lik gain (nats) | 95% CI (5-min blocks) | bootstrap p (gain <= 0) |
|---|---|---|---|---|---|
| major | primary | 5 | 0.071 | [+0.0632, +0.0777] | 0.000 |
| major | no_tod | 5 | 0.082 | [+0.0718, +0.0929] | 0.000 |
| major | ex_cpi | 5 | 0.063 | [+0.0553, +0.0690] | 0.000 |
| major | l2_only | 5 | 0.071 | [+0.0635, +0.0776] | 0.000 |
| alt | primary | 35 | -0.012 | [-0.0297, +0.0120] | 0.851 |
| alt | no_tod | 35 | -0.003 | [-0.0223, +0.0228] | 0.708 |
| alt | ex_cpi | 35 | -0.027 | [-0.0310, -0.0220] | 1.000 |
| alt | l2_only | 4 | -0.039 | [-0.0642, -0.0088] | 0.994 |
| tradfi | primary | 11 | -0.027 | [-0.0336, -0.0189] | 1.000 |
| tradfi | no_tod | 11 | -0.016 | [-0.0213, -0.0087] | 1.000 |
| tradfi | ex_cpi | 11 | -0.019 | [-0.0250, -0.0128] | 1.000 |
| tradfi | l2_only | 3 | 0.017 | [+0.0101, +0.0247] | 0.000 |

<details><summary>Per symbol</summary>

| symbol | class | TEST durations | AFT dist | sigma | best ACD | ACD ll/obs | AFT ll/obs | gain | gain no tod | L2 | top features (gain) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1000PEPEUSDT | alt | 91837 | extreme | 1.818 | gengamma | 0.526 | 0.523 | -0.003 | 0.019 | no | cnt_10s, log_d_lag1, cnt_60s, log_d_lag3 |
| AAVEUSDT | alt | 44384 | extreme | 2.249 | gengamma | 0.015 | 0.026 | 0.011 | 0.013 | no | cnt_10s, log_d_lag1, cnt_60s, absret_10s |
| ADAUSDT | alt | 19698 | extreme | 2.053 | weibull | -0.781 | -0.755 | 0.026 | 0.019 | no | cnt_10s, log_d_lag1, log_d_lag2, last_log_prints |
| ARBUSDT | alt | 70656 | extreme | 1.971 | gengamma | 0.378 | 0.388 | 0.010 | 0.018 | no | cnt_10s, log_d_lag1, cnt_60s, absret_10s |
| ASTERUSDT | alt | 15661 | extreme | 1.863 | weibull | -1.027 | -1.012 | 0.015 | 0.038 | no | cnt_10s, log_d_lag1, log_d_lag2, cnt_60s |
| ATOMUSDT | alt | 6182 | extreme | 2.243 | gengamma | -2.202 | -2.260 | -0.058 | -0.056 | yes | cnt_60s, log_d_lag1, absret_60s, last_sign |
| AVAXUSDT | alt | 46698 | extreme | 1.938 | gengamma | -0.016 | 0.012 | 0.028 | 0.029 | no | cnt_10s, log_d_lag1, last_log_notional, absret_10s |
| BNBUSDT | alt | 34191 | extreme | 2.338 | gengamma | -0.064 | -0.048 | 0.016 | 0.017 | no | cnt_10s, log_d_lag1, cnt_60s, last_log_notional |
| CARVUSDT | alt | 2526 | normal | 3.638 | gengamma | -2.646 | -2.739 | -0.093 | -0.090 | yes | log_d_lag1, last_log_notional, l2_obi_5, cnt_60s |
| CRVUSDT | alt | 14482 | extreme | 2.459 | gengamma | -1.033 | -1.078 | -0.045 | -0.047 | no | cnt_10s, absret_10s, cnt_60s, log_d_lag1 |
| DOTUSDT | alt | 29592 | extreme | 2.093 | gengamma | -0.467 | -0.491 | -0.023 | -0.011 | no | cnt_10s, cnt_60s, log_d_lag1, last_sign |
| ENAUSDT | alt | 129195 | extreme | 1.739 | gengamma | 0.809 | 0.833 | 0.024 | 0.030 | no | log_d_lag1, last_sign, last_log_notional, cnt_60s |
| ETCUSDT | alt | 12310 | extreme | 2.440 | gengamma | -1.309 | -1.406 | -0.097 | -0.064 | no | cnt_10s, last_log_notional, log_d_lag1, cnt_60s |
| FILUSDT | alt | 25998 | extreme | 2.219 | gengamma | -0.618 | -0.651 | -0.033 | -0.027 | no | cnt_60s, cnt_10s, last_log_prints, log_d_lag1 |
| GRAMUSDT | alt | 14724 | extreme | 2.206 | gengamma | -1.107 | -1.062 | 0.045 | 0.050 | no | cnt_10s, cnt_60s, log_d_lag1, absret_10s |
| HYPEUSDT | alt | 119666 | normal | 2.316 | weibull | 0.677 | 0.705 | 0.027 | 0.051 | yes | log_d_lag1, last_sign, last_log_notional, log_d_lag2 |
| INJUSDT | alt | 48499 | extreme | 2.119 | gengamma | -0.172 | -0.174 | -0.002 | 7.05e-04 | no | log_d_lag1, cnt_10s, cnt_60s, absret_10s |
| KASUSDT | alt | 11202 | extreme | 2.389 | weibull | -1.328 | -1.360 | -0.033 | -0.022 | yes | cnt_10s, log_d_lag1, cnt_60s, last_log_notional |
| LINKUSDT | alt | 42427 | extreme | 2.240 | gengamma | 0.052 | 0.051 | -0.001 | 0.006 | no | cnt_10s, log_d_lag1, last_log_notional, cnt_60s |
| LTCUSDT | alt | 18833 | extreme | 1.887 | gengamma | -0.874 | -0.879 | -0.005 | 0.010 | no | cnt_10s, log_d_lag1, absret_10s, log_d_lag3 |
| MAVIAUSDT | alt | 2583 | extreme | 2.849 | gengamma | -2.127 | -2.164 | -0.037 | -0.034 | no | cnt_60s, absret_60s, log_d_lag1, last_log_prints |
| MONUSDT | alt | 11843 | extreme | 2.193 | gengamma | -1.427 | -1.516 | -0.089 | -0.064 | no | absret_10s, log_d_lag1, cnt_60s, last_log_prints |
| NEARUSDT | alt | 54082 | extreme | 1.765 | weibull | -0.101 | -0.078 | 0.023 | 0.028 | no | cnt_10s, log_d_lag1, log_d_lag2, absret_10s |
| ONDOUSDT | alt | 30817 | extreme | 1.777 | gengamma | -0.576 | -0.608 | -0.033 | -0.031 | no | cnt_10s, log_d_lag1, log_d_lag3, log_d_lag2 |
| RENDERUSDT | alt | 8829 | extreme | 2.008 | gengamma | -1.867 | -1.951 | -0.084 | -0.080 | no | cnt_10s, log_d_lag1, cnt_60s, absret_10s |
| SUIUSDT | alt | 44683 | extreme | 1.845 | gengamma | -0.165 | -0.168 | -0.003 | -0.002 | no | cnt_10s, log_d_lag1, cnt_60s, log_d_lag2 |
| TAOUSDT | alt | 48892 | extreme | 2.102 | gengamma | 0.015 | -0.009 | -0.025 | -0.004 | no | cnt_10s, cnt_60s, log_d_lag1, absret_60s |
| TRUMPUSDT | alt | 14253 | extreme | 1.953 | gengamma | -1.290 | -1.272 | 0.018 | 0.024 | no | absret_10s, log_d_lag1, cnt_10s, cnt_60s |
| TRXUSDT | alt | 4931 | extreme | 1.735 | gengamma | -2.597 | -2.571 | 0.027 | 0.028 | no | cnt_10s, log_d_lag1, log_d_lag2, cnt_60s |
| UNIUSDT | alt | 47873 | extreme | 1.905 | gengamma | -0.236 | -0.240 | -0.004 | 0.005 | no | cnt_10s, log_d_lag1, cnt_60s, log_d_lag3 |
| WIFUSDT | alt | 7601 | extreme | 2.321 | gengamma | -1.803 | -1.890 | -0.087 | -0.085 | no | cnt_10s, absret_60s, log_d_lag1, cnt_60s |
| WLDUSDT | alt | 30250 | extreme | 1.749 | gengamma | -0.568 | -0.530 | 0.038 | 0.043 | no | cnt_10s, cnt_60s, log_d_lag1, log_d_lag2 |
| WLFIUSDT | alt | 10344 | normal | 3.109 | gengamma | -1.068 | -1.078 | -0.009 | 3.11e-05 | no | log_d_lag1, last_log_prints, last_log_notional, hour_of_day |
| XLMUSDT | alt | 13057 | extreme | 2.422 | gengamma | -1.029 | -1.015 | 0.014 | 0.024 | no | cnt_10s, last_log_notional, cnt_60s, last_sign |
| ZECUSDT | alt | 311427 | normal | 2.301 | gengamma | 1.745 | 1.766 | 0.021 | 0.044 | no | log_d_lag1, last_log_notional, last_log_prints, eff_spread_bps |
| BTCUSDT | major | 431420 | logistic | 1.114 | gengamma | 2.572 | 2.737 | 0.165 | 0.169 | yes | log_d_lag1, last_log_prints, log_d_lag2, last_log_notional |
| DOGEUSDT | major | 45405 | extreme | 1.710 | weibull | -0.042 | -0.020 | 0.021 | 0.031 | yes | cnt_10s, log_d_lag1, cnt_60s, log_d_lag4 |
| ETHUSDT | major | 343217 | normal | 1.982 | gengamma | 2.098 | 2.196 | 0.098 | 0.133 | yes | log_d_lag1, last_log_prints, log_d_lag2, cnt_10s |
| SOLUSDT | major | 90290 | extreme | 1.577 | gengamma | 0.632 | 0.651 | 0.019 | 0.024 | yes | cnt_10s, log_d_lag1, log_d_lag5, cnt_60s |
| XRPUSDT | major | 108376 | extreme | 1.726 | gengamma | 0.770 | 0.821 | 0.051 | 0.052 | yes | cnt_10s, log_d_lag1, last_log_prints, last_log_notional |
| BZUSDT | tradfi | 20571 | extreme | 1.857 | weibull | -1.084 | -1.076 | 0.008 | 0.013 | no | cnt_10s, cnt_60s, log_d_lag1, log_d_lag2 |
| CLUSDT | tradfi | 33453 | extreme | 1.807 | gengamma | -0.472 | -0.453 | 0.019 | 0.031 | no | cnt_10s, log_d_lag1, log_d_lag4, cnt_60s |
| COINUSDT | tradfi | 9128 | extreme | 2.313 | gengamma | -1.493 | -1.571 | -0.078 | -0.056 | no | cnt_60s, cnt_10s, log_d_lag1, log_d_lag2 |
| CRWVUSDT | tradfi | 33956 | logistic | 1.244 | weibull | -0.154 | -0.241 | -0.087 | -0.072 | no | log_d_lag1, cnt_60s, cnt_10s, last_log_prints |
| GOOGLUSDT | tradfi | 18953 | extreme | 2.035 | gengamma | -0.774 | -0.824 | -0.050 | -0.028 | no | cnt_60s, cnt_10s, hour_of_day, log_scale30m |
| MSTRUSDT | tradfi | 43409 | extreme | 2.301 | gengamma | 0.023 | -0.025 | -0.049 | -0.020 | no | cnt_10s, cnt_60s, log_d_lag1, log_d_lag2 |
| NVDAUSDT | tradfi | 94042 | extreme | 1.939 | gengamma | 0.465 | 0.473 | 0.009 | 0.011 | yes | cnt_10s, cnt_60s, log_d_lag1, log_d_lag3 |
| QQQUSDT | tradfi | 21275 | extreme | 2.139 | gengamma | -0.770 | -0.787 | -0.017 | -0.016 | yes | cnt_10s, log_d_lag1, cnt_60s, l2_obi_l1 |
| TSLAUSDT | tradfi | 18333 | extreme | 2.114 | gengamma | -0.863 | -0.890 | -0.026 | -0.022 | no | cnt_10s, cnt_60s, log_d_lag1, last_sign |
| XAGUSDT | tradfi | 13040 | extreme | 1.733 | gengamma | -1.154 | -1.237 | -0.083 | -0.076 | no | cnt_10s, cnt_60s, log_d_lag1, absret_10s |
| XAUUSDT | tradfi | 45482 | logistic | 1.110 | gengamma | 0.361 | 0.420 | 0.059 | 0.062 | yes | log_d_lag1, log_d_lag2, cnt_10s, log_d_lag3 |

</details>

## What the generator must reproduce

Targets for the event-level simulator v1 (agent C3). Each line is a held-out fact with its class range, and the parameter file has the per-symbol values.

1. **Arrivals in event time, by order, not by print.** CV of durations about 2.2. The marginal is a 3-component exponential mixture (about 30% bursts at about 5 ms, about 40% at 0.2-0.4 s, about 30% gaps at 1-5 s). The intensity moves slowly (trailing 30-min level, doubling from FIT to TEST) and clustering is integrated (ACD alpha + beta near 1). Plain ACD leaves the burst component unexplained (KS ratio 0.78), so add a self-exciting burst term or draw innovations from the fitted mixture. The fitted ACDs are FILTERS, not generators: E[(alpha eps + beta)^2] (`acd.models.*.m2_factor`) is >= 1 for all 51 Weibull fits (median 1.20) and for 28 of 51 exponential / generalized-gamma fits (median 1.00), so simulated ACD paths explode or collapse (CV 3-35 instead of about 2). Generate arrivals from the mixture marginal on a slow intensity path (or a Hawkes process), and use the ACD as a one-step forecast only.
2. **Prints per order.** Mean 1.9 (TradFi) to 4.0 (majors), heavy-tailed pmf (`rates.prints_per_order_pmf`). Split simulated orders into prints at the same ms.
3. **Sizes.** Log-normal body plus a heavy tail: print zeta about 1.4-1.8, order zeta about 1.1-1.5 (`size.*.size_law`). Do not rely on a universal exponent.
4. **Order signs.** Lag-1 ACF 0.33-0.39 and a power-law decay with gamma 0.45 (TradFi) to 0.75 (alts), still about 0.01-0.05 at lag 100, plus positive PACF beyond lag 10. `TranslatedSignModel` (`signs.order.sign_model`) reproduces the empirical short lags and the power-law tail (tests/test_laws.py). The pure fGn generator cannot reach a lag-1 ACF of 0.35 when gamma is above about 0.5. The v0 generator had near-zero flow autocorrelation, and this is the missing piece.
5. **Clock-time signed flow.** Lag-1 ACF of 1 s signed notional only 0.03-0.04 but Hurst 0.53 (notional) and 0.58 (counts). Long sign memory combined with heavy-tailed sizes must give both.
6. **Volume per bin.** Log-normal positive part, zero share 13-16% of 10 s bins for alts and TradFi (none for majors), lag-1 ACF of log volume 0.35-0.64 with slow decay, Spearman with |return| 0.40-0.55.
7. **Conditional arrival model.** For majors, the saved xgboost AFT models (`aft_2026-11-10/<SYM>.json`) give a better next-arrival density than ACD. For alts and TradFi, use the ACD.

Class medians of the fitted parameters (each half fitted on itself):

| parameter | major fit | major test | alt fit | alt test | tradfi fit | tradfi test |
|---|---|---|---|---|---|---|
| events_per_s | 1.278 | 2.509 | 0.294 | 0.602 | 0.240 | 0.492 |
| prints_per_order_mean | 4.041 | 3.873 | 3.130 | 3.061 | 1.867 | 2.134 |
| weibull_c_event | 0.507 | 0.466 | 0.433 | 0.444 | 0.437 | 0.433 |
| lognormal_s_event | 2.534 | 2.363 | 2.996 | 2.733 | 2.837 | 2.731 |
| acd_weibull_alpha | 0.200 | 0.092 | 0.207 | 0.098 | 0.185 | 0.141 |
| acd_weibull_beta | 0.800 | 0.908 | 0.793 | 0.902 | 0.815 | 0.859 |
| acd_weibull_k | 0.540 | 0.611 | 0.463 | 0.538 | 0.515 | 0.536 |
| size_zeta_print | 1.769 | 1.844 | 1.658 | 1.759 | 1.210 | 1.391 |
| size_zeta_order | 1.544 | 1.376 | 1.281 | 1.289 | 1.188 | 1.111 |
| sign_gamma_order | 0.743 | 0.640 | 0.757 | 0.739 | 0.461 | 0.451 |
| sign_c_order | 0.343 | 0.258 | 0.354 | 0.284 | 0.287 | 0.277 |
| sign_acf1_order | 0.352 | 0.329 | 0.388 | 0.391 | 0.357 | 0.361 |
| hurst_dfa_1s | 0.581 | 0.583 | 0.520 | 0.528 | 0.515 | 0.528 |
| vol10s_p_zero | 0.005 | 0.004 | 0.257 | 0.127 | 0.225 | 0.155 |
| vol60s_lognormal_s | 0.992 | 1.194 | 1.223 | 1.333 | 1.359 | 1.671 |

## Parameter file schema (for the population / simulator agent)

- `file`: data/models/microstructure_laws/params_2026-11-10.json (+ params_2026-11-10.parquet: tidy rows symbol, class, half, path, value, text)
- `top`: schema_version, date, created, method (METHOD dict), units, classes, symbols{SYM}, class_medians{class}
- `symbols.SYM`: class, n_prints, n_orders, n_events, halves{fit, test}: every law below is fitted on THAT half (fit = 00-12 UTC, test = 12-24 UTC)
- `rates`: prints_per_s, orders_per_s, events_per_s, prints_per_order_mean, prints_per_order_pmf{1..10, >10}, prints_per_order_tail_zeta (Hill on n >= 2), p_buy_order, two_sided_ms_share, notional_usd, median_print_usd, hourly_events[12]
- `durations.{event|print}.laws.FAMILY`: Law dict {family, params, n_fit, loglik_fit, k}: params use scipy names with loc = 0: exponential{scale}, weibull{c, scale} = weibull_min, gamma{a, scale}, lognormal{s, scale} = lognorm, gengamma{a, c, scale}, burr12{c, d, scale}, loglogistic{c, scale} = fisk, expmixK{w1..wK, rate1..rateK (per second)}; durations in seconds; rebuild with ai_trader.sim.laws.law_from_dict(d).sample(n)
- `acd`: deseason{kind: trailing_mean_duration, window_s: 1800} (x = d / s, s = mean event duration over the trailing 30 min); models{exponential|weibull|gengamma}: {omega, alpha, beta, innov, shape{k} or {a, c} (unit-mean innovation), psi0}; duration = s * psi * eps; ai_trader.sim.laws.ACD.from_dict(d).sample(n, scale=s) (most fits are near-integrated, so sample() caps alpha + beta at 0.99 and anchors at psi0 unless max_persistence=1); m2_factor = E[(alpha eps + beta)^2]: only fits with m2_factor < 1 are stable generators, the others are one-step filters
- `size.{print|order}`: powerlaw{xmin (USD), alpha (pdf exponent), zeta = alpha - 1 (CCDF tail index), n_tail, n, ks, p_tail}, size_law{kind: size, mu, sigma (log-normal body below xmin), xmin, alpha, p_tail} -> SizeLaw.sample, hill_zeta, vuong{R, z, p, ln_mu, ln_sigma}, p_boot (CSN), median, p99, max
- `signs.{order|print}`: acf_power{c, gamma} (ACF ~ c k^-gamma, log-binned lags 1-1000 above the noise band), acf_exp{c, tau}, sign_model{kind: translated_sign, c, gamma, p_buy, short{lag: ACF for lags 1-20}} -> law_from_dict(d).sample(n) gives +1/-1 signs matching the empirical short lags and the power-law tail (USE THIS ONE); fgn_sign{c, gamma, p_buy, H_latent, w} (pure power law; cannot reach the lag-1 ACF of ~0.35 when gamma > ~0.5), p_buy, acf{lag: value}, pacf_1_20
- `signed_flow`: order_signed_notional (per order, trade time), clock_1s / clock_10s (signed USD per clock bin): acf_power, acf_exp, acf, pacf_1_20; clock_1s also hurst_dfa (signed notional) and hurst_dfa_count (signed count)
- `volume.{10s|60s}`: p_zero, laws_positive{gamma|gengamma|lognormal} (USD per bin, positive bins), best_own_half, acf_log1p_power / acf_log1p_exp (ACF of log(1 + v / log1p_divisor_usd)), spearman_absret{rho, p, n}
- `aft (fit half)`: model_file (xgboost JSON, survival:aft; predict(output_margin=True) = mu, log T = mu + sigma Z), best{dist, sigma, train_scale, best_iter}, rounds, features (ordered names, see aft_features), importance_gain

Tidy parquet: 57074 rows. Example: `ai_trader.sim.laws.law_from_dict(P['symbols']['BTCUSDT']['halves']['fit']['durations']['event']['laws']['weibull']).sample(1000)`.

## Caveats

- One day, one market state (a crypto sell-off on CPI day). The TEST half is about twice as active as FIT, so every FIT-fitted scale is off in TEST: this penalises all families alike and favours models with an adaptive level (ACD, AFT count features).
- Deseasonalisation is a causal trailing 30-minute mean. It absorbs slow intensity swings, not a true time-of-day profile (one day cannot identify one). With more days, use an hour-of-week profile (features.seasonality) instead.
- ACD persistence sits at the integrated bound for most symbols. The model is close to IACD, so its unconditional mean is not defined, and simulating it raw drifts to near-zero durations. `ACD.sample` therefore caps alpha + beta at 0.99 and anchors at the in-sample mean (a simulation-time regularisation, not a fitted quantity). Even capped, fits with m2_factor >= 1 stay far too dispersed.
- Print-level duration fits are dominated by the 0.5 ms atom and are reported for completeness only.
- CSN with 10^3-10^5 tail points rejects any curvature. Sizes are partly discrete (round lots).
- The ACF noise-floor rule (bins above 2/sqrt(n) only) was set after a two-symbol smoke test (CARV, KAS), before the full run. A generalized gamma / Burr XII optimiser bug (stuck below the families they nest) was found and fixed before any verdict was recorded (diary 'mistake'). Duration and volume laws were refitted for all symbols.
- The Hurst CI bootstraps symbols as if independent. Crypto flows co-move, so the true CI is wider, but 40 of 48 symbols are individually above 0.5 for notional and all 48 for counts.
- The H-MS9 comparator is the ACD that is best on TEST (conservative toward the AFT). Spread and depth imbalance come from ob200 for 12 symbols only; the others use a trade-based effective-spread proxy. Hour of day is outside its FIT support (no-tod variant reported).
- Holm runs over the five summary p-values. Hypotheses that fail get p = 1, which makes the step-down strict for H-MS3.
- Classes: BNB is an alt here (task definition); in simulator v0 it was a major.

## Rule candidates

```yaml
- id: R-MS-1
  kind: expect
  rule: "WHEN reasoning about or simulating aggressive trade arrivals on a Bybit perp THEN expect bursts, not a Poisson stream: durations have a coefficient of variation of about 2 and roughly a quarter to a third of order gaps are a few milliseconds"
  when: "inter-arrival times of taker orders (same-ms same-side prints merged)"
  then: "use a 3-component exponential mixture or Burr XII marginal with a slowly moving intensity; never an exponential law"
  scope: {assets: all, horizon: "1ms-1m", sides: both}
  effect: "exponential rejected 51/51 (median KS 0.33); mixture or Burr XII best held-out for 37/51; burst component weight 0.24-0.32 at ~5 ms"
  confidence: C
  evidence: [H-MS1]
  review: "a week of trades_raw with the same winners in >= 80% of symbol-days"
  status: provisional
- id: R-MS-2
  kind: avoid
  rule: "WHEN building a duration model for a generator or a fill-timing estimate THEN avoid relying on ACD(1,1) alone to remove clustering; its residuals keep most of the non-exponential shape"
  when: "conditional duration modelling, event time"
  then: "add a burst / self-exciting component or draw innovations from the fitted mixture; check residual KS"
  scope: {assets: all, horizon: "1ms-1m", sides: both}
  effect: "residual KS / raw KS median 0.78; halved for 2/51 symbols; ACD persistence ~1 (integrated); second-moment factor >= 1 for 51/51 Weibull fits, so they are filters, not generators"
  confidence: C
  evidence: [H-MS1]
  review: "a Hawkes or mixture-innovation ACD that halves the KS on >= 80% of symbols out of sample"
  status: provisional
- id: R-MS-3
  kind: expect
  rule: "WHEN taker orders have been mostly on one side THEN expect the next orders to stay on that side more often than not, with memory decaying slowly (power law) over hundreds of orders"
  when: "order-level trade signs (prints of one order merged)"
  then: "treat order flow as persistent in event time: lag-1 sign autocorrelation about 0.35, still 0.01-0.05 at lag 100; longest memory in TradFi perps"
  scope: {assets: all, horizon: "1-1000 orders", sides: both}
  effect: "power law beats exponential held-out for 48/48; gamma 0.64 majors, 0.74 alts, 0.45 TradFi; 1 s signed-flow Hurst 0.53 (CI 0.52-0.54)"
  confidence: C
  evidence: [H-MS3]
  review: "a week of trades_raw; gamma in (0, 1) for >= 80% of symbol-days"
  status: provisional
- id: R-MS-4
  kind: avoid
  rule: "WHEN counting prints as trades (signs, durations, sizes) THEN avoid treating same-millisecond same-side prints as separate orders"
  when: "any statistic built from Bybit public trades"
  then: "merge same-ms same-side prints into one order first; print-level sign autocorrelation and zero durations are artefacts of matching against several makers"
  scope: {assets: all, horizon: "tick", sides: both}
  effect: "1.9-4.0 prints per order; print-level sign ACF(1) 0.70-0.84 vs 0.33-0.39 for orders"
  confidence: C
  evidence: [H-MS1, H-MS3]
  review: "confirm on another venue or day"
  status: provisional
- id: R-MS-5
  kind: avoid
  rule: "WHEN modelling the tail of trade or order notional THEN avoid a pure Pareto with a universal exponent; prefer a log-normal body spliced to a heavy tail fitted per symbol, with the tail index treated as uncertain"
  when: "size distributions for simulation, impact or liquidation scenarios"
  then: "use the per-symbol size_law; expect tail index 1.3-1.8 (prints) and about 1.1-1.5 (orders) but not stable between half-days"
  scope: {assets: all, horizon: "tick", sides: both}
  effect: "CSN rejects the power law for 45/51; Vuong never prefers it to a log-normal (0/51); FIT-TEST Spearman of zeta 0.27"
  confidence: C
  evidence: [H-MS2]
  review: "multi-day tails with a stable index (Spearman > 0.7 between days)"
  status: provisional
- id: R-MS-6
  kind: expect
  rule: "WHEN traded volume in a 10 s or 1 min interval is high THEN expect the absolute return in the same interval to be large and the elevated volume to persist for several intervals"
  when: "notional volume per clock bin"
  then: "read volume bursts as volatility bursts; quote distances based on calm-period volume are too tight during them"
  scope: {assets: all, horizon: "10s-1h", sides: both}
  effect: "Spearman(volume, |return|) 0.40-0.55, positive for 51/51; lag-1 ACF of log volume 0.35-0.64; log-normal beats gamma for 49-51/51"
  confidence: C
  evidence: [H-MS4]
  review: "a week of trades_raw"
  status: provisional
- id: R-MS-7
  kind: prefer
  rule: "WHEN predicting the time to the next trade on BTC, ETH, SOL, XRP or DOGE THEN prefer the gradient-boosted AFT model (recent trade counts, last durations, prints in the last order) over ACD; for alts and TradFi perps keep the ACD"
  when: "next-arrival density, e.g. fill-timing or quote-refresh logic"
  then: "use data/models/microstructure_laws/aft_2026-11-10/<SYM>.json for majors only"
  scope: {assets: [BTCUSDT, ETHUSDT, SOLUSDT, XRPUSDT, DOGEUSDT], horizon: "1ms-1m", sides: both}
  effect: "+0.071 nats per duration [0.063, 0.078] for majors; alts -0.012 [-0.030, 0.012]; TradFi -0.027 [-0.034, -0.019]"
  confidence: C
  evidence: [H-MS9]
  review: "refit on several days; the majors gain must survive a day-ahead test"
  status: provisional
```
