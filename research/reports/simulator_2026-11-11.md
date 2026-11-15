# Execution simulator v0: fill model, reaction functions, flow generator, replay (H-SIM1..H-SIM4)

_as of 2026-11-10 23:00 UTC (flow_3s read point in time at this timestamp) | segment: clean_holdout only | code: src/ai_trader/sim/, src/ai_trader/research/simulator_study.py | tests: tests/test_sim.py_

All data is after the knowledge cutoff (2026-07-01), so every verdict is clean_holdout. It is a short first sample: about three days of collector data with gaps (one market state: a crypto sell-off), and our own live quotes from about 15 hours on 2026-11-08/09. Treat every number as a first measurement, not a settled property.

## Verdicts

| id | test (as registered) | key numbers | verdict |
|---|---|---|---|
| H-SIM1 | fill rate and 60 s markout of held-out live quotes inside their 95% CIs | variant touch: predicted fill rate 1.02% vs observed 0.90% CI [0.37, 1.48]; markout 2.45 bp vs -0.12 bp CI [-11.27, 7.56]; test fills 23 | **inconclusive** |
| H-SIM2 | >= 4 of 5 stylised facts inside the day-to-day envelope, each class | facts inside: major 4/5, alt 0/5, tradfi 2/5 | **contradicts** |
| H-SIM3 | sign stable on >= 80% of symbol-days and equal scaled coefficients within >= 2 of 3 classes | sign shares: momentum 0.77, obi 0.70, spread 0.72 (358 symbol-days); classes not rejected 0/3 | **contradicts** |
| H-SIM4 | Spearman > 0.5 with CI excluding 0 (36-set grid) | rho -0.09, CI [-0.73, 0.44], 50 symbols | **contradicts** |

## Data and coverage

flow_3s candles per symbol and UTC day (a full day is 28,800 three-second candles); collector gaps are never filled:

| day | symbols | median candles | max candles | share of a full day (median / max) |
|---|---|---|---|---|
| 2026-11-07 | 50 | 7394 | 7394 | 26% / 26% |
| 2026-11-08 | 104 | 3417 | 24384 | 12% / 85% |
| 2026-11-09 | 104 | 15923 | 15923 | 55% / 55% |
| 2026-11-10 | 104 | 10186 | 10186 | 35% / 35% |

On 2026-11-08 the 50 original symbols have about 85% of the day; the 54 symbols added to the collector at about 21:00 UTC have 3 hours.

Live records: 5 overlapping pulls deduplicated by orderId / execId: 5128 mm quotes (161 filled), 162 maker executions; 42 post-only rejects dropped (they never rested). Quote coverage by flow_3s: 5086 of 5086 usable (100%; reasons {'usable': 5086}). The collector covered both quoting sessions (2026-11-08 02:20-11:21 on SOL/XRP/ADA, 2026-11-08 22:59 to 2026-11-09 08:15 on up to 50 symbols).

## Methods

- **data**: flow_3s Bybit linear, Lake.read_asof per symbol with column projection at a fixed as_of; complete 3 s grid; candles missing (collector gaps) or without a valid book (mid / spread null) are not ok and never zero-filled; lags and ACFs never cross a gap
- **classes**: major = BTC, ETH, SOL, XRP, BNB, DOGE; tradfi = instrument symbol_type in ETF/stock/commodity or the TradFi stream list (incl. XAUT, PAXG); alt = the rest
- **H-SIM1**: live mm quotes (orderLinkId 'mm...', deduplicated across the 5 pulls by orderId) with at-risk candles [created, cancel) or [created, fill candle]; usable when >= 95% of at-risk weight is in the lake, the fill candle is present and the 60 s markout mid is known; train = first half of usable quotes by creation time, test = second half; MLE of (theta0, b, eps) per variant (touch / strict); primary variant = lower train AIC; predicted fill rate = compensator sum of hazards / quotes; predicted markout = hazard-weighted; observed CIs: 30-min block bootstrap of the test half (B = 2000); markout = side * (mid at the close of the candle 60 s after the fill candle - quote price) / price in bps
- **H-SIM3**: per symbol-day OLS of y = signed taker notional / day median depth_25 on lagged returns (sigma units; lags 1, 2-5, 6-20), obi_5, flow lags and recent vol; intensity log(1+trades) on lagged log spread, vol and activity lags; momentum = b1 + 4 b2 + 15 b3; 5-minute clock clusters; sign stability = share of symbol-days with the modal sign for each primary response (momentum, obi, spread); equality: per symbol pooled over days, stacked cluster influence across symbols, Wald H0 equal (momentum, obi, spread) within class; supports = all three sign shares >= 0.80 and no rejection at 5% in >= 2 of 3 classes; symbol-days with < 2000 usable 3 s rows dropped
- **H-SIM2**: generator (sim/flow_sim.py) fitted per symbol on 2026-11-08 + 2026-11-09, 10 paths of the 2026-11-10 ok-candle count each; facts: excess kurtosis of 3 s mid returns, mean ACF of |r| lags 1-60, mean ACF of signed flow lags 1-20, standardised impact slope corr(r_t, F_t), spread median and p90 (both must pass); class value = median over symbols (sim: of each symbol's median over paths); envelope = [min over the 3 real days of the 2.5% bootstrap quantile, max of the 97.5%] of the class median, 1 h block bootstrap within each symbol-day (B = 300); supports = >= 4 of 5 facts inside for every class; contradicts = some class has <= 2 inside
- **H-SIM4**: atx mm quote rule ported (sim/replay.py) with config/mm_multi.yaml values, one symbol at a time, decisions every 3 candles (9 s), no structural levels; grid ask {0.2, 0.45, 0.9} x bid {0.2, 0.45, 0.9} x take-profit {0.3, 0.9} x requote {0.15, 0.5} sigma = 36 sets; replay on the real 2026-11-10 paths of the mm_multi symbols (warm-up from earlier candles) with the H-SIM1 primary fill model, 3 fill seeds; simulator: the H-SIM2 model of each symbol, 3 paths of the same ok-candle count, same fill model; metric = pooled net P&L / turnover (bps) per parameter set; Spearman of the 36 rankings, bootstrap over symbols (B = 2000); supports = rho > 0.5 with the 95% CI excluding 0
- **fees**: maker 2 bps, taker 5.5 bps; market orders pay half-spread + 10 bps * sqrt(notional / depth_25)

## H-SIM1: passive fill model

Split at 2026-11-09T01:42:44 UTC (median creation time of usable quotes). Train: 2543 quotes, 138 fills; test: 2543 quotes, 23 fills (14 half-hour blocks).

| variant | theta0 (touch) | b per bp | eps | train AIC | test log-lik | train fill rate pred / obs | test fill rate pred | test markout pred (bp) | naive theta=1 fill rate | inside both CIs |
|---|---|---|---|---|---|---|---|---|---|---|
| touch | 0.454 | 20.1 | 1.9e-13 | 175.0 | -20.5 | 5.43% / 5.43% | 1.02% | 2.45 | 1.67% | True |
| strict | 1.000 | 20.1 | 3.6e-04 | 973.9 | -148.7 | 5.41% / 5.43% | 2.89% | 69.73 | 0.48% | False |

Observed on the test half: fill rate 0.90% (95% CI [0.37, 1.48]), 60 s markout -0.12 bp (CI [-11.27, 7.56]).

Reading: a strict trade-through always reaches our order (b at its upper bound: any penetration of a fraction of a bp fills), and a candle that only touches our price fills us with probability theta0 = 0.45 (queue position). The touch variant explains the six-fold drop in fill rate between the halves (5.4% to 0.9%) from the 3 s candles alone. The naive model (every touch fills) over-predicts the held-out fill rate, outside the CI. The strict variant is rejected by AIC: touch-only fills are common, and its leak term puts fill probability on candles far from the quote. Self-print check: in 0.6% of fill candles the aggressive notional equalled our own fill, so our own prints rarely create the touch that the model then explains.

Verdict: **inconclusive**. Both predictions are inside the observed CIs (the literal success criterion), but the held-out half has only 23 fills, below the 30-fill minimum fixed in the study code before the run; the markout CI is about 19 bp wide, so the markout test has little power.

## H-SIM3: reaction functions of aggressive flow (implicit trader types)

358 symbol-days (3 s regressions, 103 symbols, days 2026-11-07, 2026-11-08, 2026-11-09, 2026-11-10); classes: major 6, alt 60, tradfi 38. Median R^2: direction 0.006, intensity 0.156.

Sign stability of the primary responses (share of symbol-days with the modal sign):

| response | all (3 s) | modal sign | major | alt | TradFi | all (1 m) |
|---|---|---|---|---|---|---|
| momentum | 0.77 | + | 0.92 (+) | 0.86 (+) | 0.58 (+) | 0.66 (+) |
| obi | 0.70 | - | 0.58 (-) | 0.71 (-) | 0.72 (-) | 0.72 (-) |
| spread | 0.72 | - | 0.88 (+) | 0.72 (-) | 0.84 (-) | 0.51 (+) |

Reaction-function table by class (3 s; median coefficient, share positive, share with t > 2 / t < -2). Flow is in units of the day's depth within 25 bp, returns in units of the day's 3 s sigma:

| component | class | median | share + | t>2 | t<-2 |
|---|---|---|---|---|---|
| momentum 3 s (lag 1) | major | 1.49e-04 | 0.92 | 0.42 | 0.00 |
| momentum lags 2-5 | major | 2.96e-05 | 0.79 | 0.42 | 0.00 |
| momentum lags 6-20 | major | 2.59e-05 | 0.83 | 0.42 | 0.00 |
| book imbalance obi_5 | major | -5.34e-05 | 0.42 | 0.04 | 0.08 |
| flow persistence lag 1 | major | 3.88e-02 | 0.79 | 0.25 | 0.00 |
| flow persistence lags 2-5 | major | 9.64e-03 | 0.75 | 0.21 | 0.00 |
| signed flow on recent vol | major | 9.17e-08 | 0.50 | 0.04 | 0.12 |
| intensity on log spread | major | 1.03e+00 | 0.88 | 0.54 | 0.04 |
| intensity on recent vol | major | 7.90e-02 | 0.92 | 0.21 | 0.00 |
| VAR: cum. flow response to a 1-sigma return | major | 2.27e-04 | 0.92 | n/a | n/a |
| VAR: cum. return response to a unit flow shock | major | -1.45e+00 | 0.46 | n/a | n/a |
| VAR: cum. flow response to a flow shock | major | 1.10e-01 | 0.88 | n/a | n/a |
| momentum 3 s (lag 1) | alt | 1.83e-04 | 0.77 | 0.27 | 0.02 |
| momentum lags 2-5 | alt | 7.57e-05 | 0.81 | 0.29 | 0.01 |
| momentum lags 6-20 | alt | 3.51e-05 | 0.79 | 0.31 | 0.01 |
| book imbalance obi_5 | alt | -3.42e-04 | 0.29 | 0.07 | 0.22 |
| flow persistence lag 1 | alt | 1.64e-02 | 0.70 | 0.11 | 0.05 |
| flow persistence lags 2-5 | alt | 7.08e-04 | 0.52 | 0.07 | 0.06 |
| signed flow on recent vol | alt | -1.42e-04 | 0.38 | 0.02 | 0.13 |
| intensity on log spread | alt | -1.50e-01 | 0.28 | 0.12 | 0.50 |
| intensity on recent vol | alt | 1.59e-01 | 0.87 | 0.60 | 0.03 |
| VAR: cum. flow response to a 1-sigma return | alt | 5.86e-04 | 0.80 | n/a | n/a |
| VAR: cum. return response to a unit flow shock | alt | -7.74e-01 | 0.40 | n/a | n/a |
| VAR: cum. flow response to a flow shock | alt | 3.13e-02 | 0.70 | n/a | n/a |
| momentum 3 s (lag 1) | tradfi | 9.46e-06 | 0.52 | 0.09 | 0.07 |
| momentum lags 2-5 | tradfi | 8.25e-06 | 0.59 | 0.11 | 0.06 |
| momentum lags 6-20 | tradfi | 5.29e-06 | 0.65 | 0.14 | 0.02 |
| book imbalance obi_5 | tradfi | -1.48e-04 | 0.28 | 0.04 | 0.29 |
| flow persistence lag 1 | tradfi | 1.62e-02 | 0.73 | 0.23 | 0.03 |
| flow persistence lags 2-5 | tradfi | 3.63e-03 | 0.61 | 0.18 | 0.07 |
| signed flow on recent vol | tradfi | 1.68e-05 | 0.54 | 0.06 | 0.02 |
| intensity on log spread | tradfi | -7.11e-02 | 0.16 | 0.03 | 0.69 |
| intensity on recent vol | tradfi | 8.38e-02 | 0.88 | 0.60 | 0.06 |
| VAR: cum. flow response to a 1-sigma return | tradfi | 3.91e-05 | 0.56 | n/a | n/a |
| VAR: cum. return response to a unit flow shock | tradfi | 7.59e-01 | 0.59 | n/a | n/a |
| VAR: cum. flow response to a flow shock | tradfi | 4.56e-02 | 0.70 | n/a | n/a |

Equality of scaled coefficients within class (per symbol pooled over days; stacked 5-minute-cluster influence functions; Wald):

| class | symbols | joint W (df), p | momentum p | obi p | spread p | momentum median [IQR] | obi median [IQR] | spread median [IQR] |
|---|---|---|---|---|---|---|---|---|
| major | 6 | 198 (15), 6.7e-34 | 6.6e-06 | 5.9e-03 | 5.5e-11 | 1.61e-03 [1.0e-03, 1.7e-03] | -4.80e-04 [-6.4e-04, -7.4e-06] | 0.744 [0.27, 1.24] |
| alt | 60 | 3258 (177), 0.0e+00 | 4.8e-62 | 2.2e-60 | 6.3e-196 | 1.81e-03 [9.6e-04, 3.0e-03] | -4.48e-04 [-9.0e-04, -1.1e-04] | -0.110 [-0.19, 0.13] |
| tradfi | 37 | 3201 (108), 0.0e+00 | 2.4e-19 | 1.2e-23 | 1.3e-292 | 2.67e-04 [-7.1e-05, 5.5e-04] | -1.93e-04 [-7.2e-04, -1.5e-05] | -0.070 [-0.12, -0.02] |

Trader types read from the stable signs:

- **Short-horizon momentum takers** (flow follows the last 3 s to 1 min of returns): stable in majors and alts, weak in TradFi perps. This is the most transferable component across crypto.
- **Order splitting / herding** (signed flow persists at lag 1): positive on most symbol-days at 3 s, gone or reversed at 1 m.
- **Liquidity-seeking takers** (obi_5 has a NEGATIVE coefficient): takers sell into the deeper bid side and buy the deeper ask. They do not chase book imbalance; book imbalance is where liquidity is consumed.
- **Activity followers** (trade intensity rises with recent volatility): the most stable component of all.
- **Spread response**: intensity falls when the spread widens in alts and TradFi (liquidity-sensitive takers) but rises in majors, where a wider TWA spread is the footprint of a burst of aggressive trading (reverse causality at 3 s).

Signs are mostly stable at the class level, but the magnitudes are not interchangeable: with about 580 five-minute clusters the equal-coefficient Wald test rejects in every class, and the within-class IQRs span a factor of 2-10. A transfer to another market therefore needs that market's own coefficients (or at least its class median with an explicit dispersion), not a universal scaled reaction function.

### Market-microstructure invariance check (Kyle-Obizhaeva 2016; descriptive)

Cross-symbol log-log slopes (HC1 SE); W = daily sigma x daily dollar volume; illiquidity = (sigma^2 / dollar volume)^(1/3). Prints are not bets (a bet is split into many prints), so the trade-level slopes are only indicative.

| relation | slope (SE) | invariance prediction | R^2 | n |
|---|---|---|---|---|
| median trade notional vs W | 0.31 (0.07) | 0.33 | 0.49 | 104 |
| trades per day vs W | 0.66 (0.02) | 0.67 | 0.92 | 104 |
| median spread (bp) vs illiquidity | 1.69 (0.13) | 1.00 | 0.70 | 104 |
| model impact of a 10k USD order (bp) vs illiquidity | 1.44 (0.05) | 1.00 | 0.93 | 104 |
| depth within 25 bp vs W | 0.66 (0.10) | - | 0.47 | 104 |

Reading: the activity exponents match invariance closely (trade notional about W^(1/3), number of prints about W^(2/3)), so W is a good single scale for transferring trade size and intensity to a market we have not recorded. Spreads and model impact fall faster with liquidity than invariance predicts (slopes about 1.4-1.7 vs 1): tick-size floors keep liquid perps at one tick, and illiquid ones pay more. Use per-class intercepts and these empirical slopes, not the theoretical ones, when rescaling the generator to a new market.

## H-SIM2: flow generator vs held-out stylised facts

Fitted per symbol on 2026-11-08, 2026-11-09, validated on 2026-11-10 (104 symbols; skipped: 0). Class value = median over symbols; the envelope is the range over the three real days of the 95% bootstrap interval of the class median.

**major** (6 symbols): 4/5 registered facts inside; jump variant 2/5 (median jump probability per nonzero-return candle 0.0004).

| fact | simulated | jump variant | envelope | real by day (11-08, 11-09, 11-10) | inside |
|---|---|---|---|---|---|
| excess kurtosis of 3 s returns | 62.760 | 159.793 | [4.142, 491.261] | 369.846, 5.648, 10.864 | True |
| mean ACF of |r|, lags 1-60 | 0.060 | 0.048 | [0.053, 0.364] | 0.316, 0.186, 0.084 | True |
| mean ACF of signed flow, lags 1-20 | 0.001 | 0.001 | [0.006, 0.044] | 0.026, 0.011, 0.020 | False |
| impact slope corr(r_t, F_t) | 0.401 | 0.484 | [0.286, 0.480] | 0.433, 0.307, 0.358 | True |
| spread median (bp) | 0.513 | 0.513 | [0.499, 0.542] | 0.505, 0.527, 0.542 | True |
| spread p90 (bp) | 0.529 | 0.529 | [0.505, 0.549] | 0.513, 0.535, 0.549 | True |
| continuation ratio (extra, not registered) | 0.501 | 0.500 | [0.540, 0.592] | 0.570, 0.546, 0.550 | False |

**alt** (60 symbols): 0/5 registered facts inside; jump variant 0/5 (median jump probability per nonzero-return candle 0.0003).

| fact | simulated | jump variant | envelope | real by day (11-08, 11-09, 11-10) | inside |
|---|---|---|---|---|---|
| excess kurtosis of 3 s returns | 106.431 | 141.973 | [6.908, 51.099] | 61.985, 9.127, 10.377 | False |
| mean ACF of |r|, lags 1-60 | 0.040 | 0.036 | [0.070, 0.251] | 0.249, 0.107, 0.093 | False |
| mean ACF of signed flow, lags 1-20 | -0.000 | -0.000 | [0.002, 0.011] | 0.011, 0.004, 0.003 | False |
| impact slope corr(r_t, F_t) | 0.444 | 0.469 | [0.287, 0.439] | 0.413, 0.299, 0.325 | False |
| spread median (bp) | 2.544 | 2.544 | [2.303, 2.412] | 2.393, 2.339, 2.321 | False |
| spread p90 (bp) | 2.808 | 2.808 | [2.766, 3.125] | 3.054, 2.895, 2.852 | True |
| continuation ratio (extra, not registered) | 0.502 | 0.501 | [0.528, 0.544] | 0.538, 0.533, 0.535 | False |

**tradfi** (38 symbols): 2/5 registered facts inside; jump variant 2/5 (median jump probability per nonzero-return candle 0.0005).

| fact | simulated | jump variant | envelope | real by day (11-08, 11-09, 11-10) | inside |
|---|---|---|---|---|---|
| excess kurtosis of 3 s returns | 50.831 | 72.087 | [11.955, 73.340] | 61.459, 22.043, 15.145 | True |
| mean ACF of |r|, lags 1-60 | 0.031 | 0.025 | [0.106, 0.204] | 0.192, 0.151, 0.136 | False |
| mean ACF of signed flow, lags 1-20 | 0.000 | -0.000 | [0.000, 0.009] | 0.005, 0.002, 0.006 | False |
| impact slope corr(r_t, F_t) | 0.299 | 0.323 | [0.186, 0.258] | 0.211, 0.204, 0.229 | False |
| spread median (bp) | 2.123 | 2.123 | [1.847, 2.711] | 1.889, 2.339, 2.393 | True |
| spread p90 (bp) | 3.222 | 3.222 | [2.677, 4.308] | 2.941, 3.532, 4.039 | True |
| continuation ratio (extra, not registered) | 0.498 | 0.500 | [0.493, 0.517] | 0.504, 0.497, 0.512 | True |

Share of symbols whose own simulated fact lies inside their own envelope:

| class | excess kurtosis of 3 s returns | mean ACF of |r|, lags 1-60 | mean ACF of signed flow, lags 1-20 | impact slope corr(r_t, F_t) | spread median (bp) | spread p90 (bp) | continuation ratio (extra, not registered) |
|---|---|---|---|---|---|---|---|
| major | 1.00 | 1.00 | 0.83 | 0.67 | 1.00 | 0.50 | 0.00 |
| alt | 0.53 | 0.68 | 0.93 | 0.63 | 0.67 | 0.80 | 0.35 |
| tradfi | 0.89 | 0.58 | 0.95 | 0.82 | 0.82 | 0.89 | 0.79 |

Reading: the generator gets the level facts (spread distribution, and kurtosis in majors and TradFi) and, for majors, the volatility clustering and impact, but it fails the dependence facts everywhere: signed-flow autocorrelation comes out at about 0 (real 0.002-0.044) because the heavy-tailed notional of each candle is drawn independently of the previous candle's direction (only the buy share q is persistent, and F = q V is dominated by V); |r| autocorrelation is too low for alts and TradFi; impact is too strong for alts and TradFi (the fit on days 1-2 is not stable into day 3); and the continuation ratio of price changes is 0.50 (real 0.53-0.57 in crypto), the price-level footprint of the momentum takers of H-SIM3 that the generator does not carry into the mid. The compound-Poisson jump variant raises kurtosis and does not help any class. A v1 should model events (marked Hawkes on prints with the reaction functions as cross-excitation) rather than 3 s aggregates.

Fitted generator parameters (class medians):

| class | sigma | impact_lin | impact_sqrt | ret_ar | dir_mom1 | dir_mom2_20 | dir_q1 | dir_obi | ret_df | spread_rho |
|---|---|---|---|---|---|---|---|---|---|---|
| major | 2.100 | 4.367 | 32.131 | -0.019 | 0.002 | -0.001 | 0.129 | 0.154 | 17.105 | 0.838 |
| alt | 2.966 | 0.000 | 40.312 | -0.009 | 0.003 | -0.005 | 0.156 | 0.314 | 12.123 | 0.604 |
| tradfi | 1.974 | 0.000 | 24.788 | -0.007 | -0.034 | -0.007 | 0.165 | 0.126 | 7.935 | 0.780 |

## H-SIM4: quoting-parameter rankings, replay vs simulator

50 mm_multi symbols, 36 parameter sets, fill model touch (theta0 0.45). Spearman of the pooled net P&L per turnover rankings: **-0.09** (95% CI over symbols [-0.73, 0.44]); by class: major 0.20 (5 symbols), alt -0.21 (34 symbols), tradfi 0.36 (11 symbols). Rank agreement on turnover 0.98, on fill counts 0.98.

Best and worst parameter sets on the real 2026-11-10 replay (net of fees, flattened at the end of the path):

| parameters (ask, bid, take-profit, requote; sigma units) | real bp of turnover | sim bp | real P&L USD | real turnover USD | real fills | symbols with P&L > 0 |
|---|---|---|---|---|---|---|
| a0.2_b0.9_tp0.9_rq0.15 | 14.2 | -4.5 | 3.84 | 2702 | 395 | 43 |
| a0.45_b0.9_tp0.9_rq0.5 | 12.6 | -5.8 | 1.60 | 1269 | 185 | 33 |
| a0.2_b0.9_tp0.9_rq0.5 | 10.5 | -4.2 | 4.59 | 4393 | 576 | 39 |
| a0.2_b0.9_tp0.3_rq0.15 | 8.3 | -3.3 | 3.16 | 3814 | 606 | 43 |
| a0.2_b0.9_tp0.3_rq0.5 | 5.5 | -3.9 | 3.05 | 5535 | 760 | 36 |
| a0.9_b0.45_tp0.9_rq0.15 | -54.7 | 2.2 | -1.22 | 223 | 36 | 9 |
| a0.9_b0.9_tp0.9_rq0.5 | -58.9 | -7.3 | -0.13 | 22 | 5 | 0 |
| a0.9_b0.45_tp0.3_rq0.15 | -62.1 | 1.7 | -1.38 | 223 | 36 | 10 |
| live mm_multi (a0.45_b0.9_tp0.6_rq0.3) | 7.5 | -6.1 | 0.21 | 283 | 48 | 17 |

Reading: the simulator ranks ACTIVITY almost perfectly (turnover and fill counts, rank correlation about 0.98) but not P&L. Generated paths have no drift by construction, while 2026-11-10 was a sell-off: on the real day the sets with a tight ask and a wide bid (the bearish distance skew) earn, and wide asks lose; on generated paths every set loses a few bp and the order is noise. P&L rankings of a skewed quoter are a bet on the day's drift, which a stationary generator cannot reproduce. The generator also fills about twice as often as the real paths (flatter, more mean-reverting sim paths touch the quotes more).

## Finding for the live quoter (crates/atx/src/mm.rs, not edited)

The fast sigma never activates: `sigma_1h(&closes[closes.len() - sigma_fast_min..])` passes the last 30 closes (29 returns) to a function that returns the floor when it has fewer than 30 returns. With sigma_fast_min = 30 the fast estimate is always the 20 bp floor, so `max(slow, fast)` is the 4 h sigma (or the floor) and quotes do not widen on a 30-minute volatility burst as the config comment intends. The Python port reproduces this behaviour on purpose. A fix (pass 31 closes, or lower the minimum) is the user's call.

## Caveats

- Three days of collector data with gaps and one market state (a crypto sell-off; 2026-11-08 is a Sunday for TradFi perps). Every verdict is clean_holdout and a first sample.
- The fill model sees 3 s candle highs and lows, not the queue: theta0 is an average queue position for our minimum-size quotes; larger quotes would queue differently. Held-out fills: 23.
- The replay is one symbol at a time (portfolio net / gross caps never bind), without structural levels, with the book top rebuilt from the 3 s mid and TWA spread, and with decisions every 9 s (the live loop over 50 symbols was slower). Our own historical quotes are in the real data, but at minimum size their footprint is negligible (self-print check above).
- The generator has no cross-symbol dependence, no intraday seasonality and no feedback from returns to trade activity; activity persistence is capped at a 1 h half-life, so day-level regime shifts are not simulated within a path.
- Reaction-function regressions are reduced-form: lagged flow and returns are correlated with unobserved order splitting, so 'trader types' are statistical components, not identified agents.
- The Spearman test pools P&L over symbols; a handful of volatile symbols dominate turnover.

## What data would make v1 much better

The v0 bottlenecks are (1) the fill model sees candles, not queues; (2) 23 held-out live fills; (3) three gappy days. Proposed v1 data, for the user to decide (storage budget 30 GB, lake today 0.84 GB):

| item | scope | estimate for 4 weeks |
|---|---|---|
| raw trades (trades_raw: ts, side, price, size, id) | the 50 mm_multi symbols, 7.6 M prints/day at 25 B/print (measured on BTC trades_raw) | 5.3 GB |
| L2 book deltas (orderbook.50 or .200 updates, levels within 25-100 bp) | 12 representative symbols (4 majors, 4 alts, 4 TradFi: BTCUSDT, ETHUSDT, SOLUSDT, XRPUSDT, HYPEUSDT, ENAUSDT, MAVIAUSDT, WIFUSDT, TSLAUSDT, NVDAUSDT, XAUUSDT, CLUSDT), 5.6 M update messages/day (the collector's book_updates count) at an assumed 40-120 B/message | 6.3-19.0 GB |
| flow_3s as today (all symbols) | ongoing, 0.16 GB/day | 4.5 GB |
| total incl. the current lake | | 17-30 GB |

With raw trades and book deltas the fill model can track queue position (our size vs the size ahead at our price, depletion by prints at that price, cancels), the generator can be a proper marked Hawkes on events instead of 3 s aggregates, and reaction functions can be estimated on event time. The book-delta byte cost is an assumption: measure it with a one-day pilot on 2-3 symbols before committing, and keep the deltas on a rolling retention (like trades_raw) after the derived features are stored. A further cheap gain: run the live quoter with a few deliberate parameter variants (paired, same symbols) to collect hundreds of fills per variant, which is what H-SIM1 and H-SIM4 lack most.


## Rule candidates

```yaml
- id: R-SIM-1
  kind: expect
  rule: "WHEN estimating whether a resting post-only quote at the venue minimum quantity gets filled THEN expect a fill on almost every 3 s candle that trades strictly through its price and on only about 45% of candles that just touch it"
  when: "passive quote at the minimum quantity on a Bybit linear perp; 3 s candle low <= bid (high >= ask)"
  then: "fill probability per candle ~1 for a strict trade-through, ~0.45 for a touch (queue position); never assume every touch fills"
  scope: {assets: crypto and TradFi perps quoted by atx mm, horizon: "3s-5m", sides: both}
  effect: "touch model predicts 1.02% held-out fill rate vs 0.90% observed (CI 0.37-1.48%); every-touch-fills predicts 1.67%"
  confidence: C
  evidence: [H-SIM1, research/reports/simulator_2026-11-11.md]
  review: ">= 150 held-out live fills, or raw trades + book deltas allowing a queue-position model"
  status: candidate
- id: R-SIM-2
  kind: expect
  rule: "WHEN a crypto perp (major or alt) has just moved over the last 3 s to 1 min THEN expect aggressive flow to keep following that move (momentum takers), so a resting quote on the side the price moved toward is the one likely to be hit next"
  when: "sign of the last 3 s - 1 min mid return on BTC/ETH/SOL/XRP/BNB/DOGE or alts"
  then: "expect signed taker flow in the same direction over the next 3 s - 1 min; treat fills on that side as adverse-selection prone"
  scope: {assets: crypto, horizon: "3s-1m", sides: both}
  effect: "momentum coefficient > 0 on 92% of major and 86% of alt symbol-days (3 s); TradFi perps only 58%"
  confidence: C
  evidence: [H-SIM3, research/reports/simulator_2026-11-11.md]
  review: "a month of flow_3s with the sign share staying >= 80% for majors and alts"
  status: candidate
- id: R-SIM-3
  kind: expect
  rule: "WHEN the top-of-book depth within 5 bp is imbalanced THEN expect the next aggressive flow to hit the DEEPER side (sell into a deep bid, buy a deep ask), not to chase the imbalance"
  when: "obi_5 = (bid - ask) / (bid + ask) depth within 5 bp, latest 3 s candle"
  then: "do not read book imbalance as a signal of taker direction; positive obi_5 precedes net taker selling"
  scope: {assets: all, horizon: "3s", sides: both}
  effect: "obi coefficient < 0 on 70% of 358 symbol-days (all classes, 3 s); t < -2 on about a quarter"
  confidence: C
  evidence: [H-SIM3, research/reports/simulator_2026-11-11.md]
  review: "event-time estimates on raw book deltas; a month of flow_3s keeping the negative sign on >= 70% of symbol-days"
  status: candidate
- id: R-SIM-4
  kind: avoid
  rule: "WHEN carrying flow reaction functions or quoting calibrations from one market to another THEN avoid assuming one universal coefficient after scaling by sigma, depth and median trade notional; use the target market's own estimates or its class median with the class dispersion"
  when: "transferring simulator or quoting parameters across symbols"
  then: "re-estimate per symbol (or per class with the IQR as uncertainty); signs transfer within crypto, magnitudes do not"
  scope: {assets: all, horizon: "3s-1m", sides: both}
  effect: "Wald test of equal scaled coefficients rejects in every class (p < 1e-5); within-class IQRs span a factor of 2-10"
  confidence: C
  evidence: [H-SIM3, research/reports/simulator_2026-11-11.md]
  review: "a longer sample where the within-class equality is not rejected for >= 2 classes"
  status: candidate
- id: R-SIM-5
  kind: avoid
  rule: "WHEN choosing market-making parameters THEN avoid ranking them on paths from the v0 flow generator; rank them on replays of real 3 s paths with the calibrated fill model"
  when: "parameter search for atx mm (offsets, take-profit, requote threshold)"
  then: "use real-path replay (sim/replay.py) for rankings until a generator passes H-SIM2 and H-SIM4"
  scope: {assets: all, horizon: "1d", sides: both}
  effect: "generator failed H-SIM2 (alt 0/5, TradFi 2/5 facts) and its ranking agreement with replay is Spearman -0.09 (CI -0.73 to 0.44)"
  confidence: C
  evidence: [H-SIM2, H-SIM4, research/reports/simulator_2026-11-11.md]
  review: "a v1 generator (event-level data) that passes H-SIM2 in all classes and reaches Spearman > 0.5 with CI excluding 0"
  status: candidate
- id: R-SIM-6
  kind: expect
  rule: "WHEN recent 1-minute volatility on a perp is high THEN expect more aggressive prints in the next seconds and therefore more fills (and more trade-throughs) on resting quotes"
  when: "mean |3 s return| over the last minute above its usual level"
  then: "expect trade intensity to rise; quotes left at the same sigma distance will be reached more often"
  scope: {assets: all, horizon: "3s-1m", sides: both}
  effect: "intensity on recent vol > 0 on 88% of symbol-days (t > 2 on about 60%), all classes"
  confidence: C
  evidence: [H-SIM3, research/reports/simulator_2026-11-11.md]
  review: "a month of flow_3s"
  status: candidate
- id: R-SIM-7
  kind: expect
  rule: "WHEN replaying the atx mm quoter on real 3 s paths of a sell-off day THEN expect net results per turnover to depend mostly on the quote distances, with the best grid set (a0.2_b0.9_tp0.9_rq0.15) at 14.2 bp of turnover vs 7.5 bp for the live mm_multi settings"
  when: "real-path replay, 2026-11-10, 50 mm_multi symbols, minimum order quantities"
  then: "treat the replay ranking as one day's evidence; confirm any change with paired live variants before adopting it"
  scope: {assets: all, horizon: "1d", sides: both}
  effect: "best set 14.2 bp (3.84 USD on 2702 USD turnover); live settings 7.5 bp"
  confidence: C
  evidence: [H-SIM4, research/reports/simulator_2026-11-11.md]
  review: "replays on >= 10 more days and paired live variants"
  status: candidate
```
