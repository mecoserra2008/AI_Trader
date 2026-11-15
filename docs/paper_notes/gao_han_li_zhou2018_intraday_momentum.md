# Market intraday momentum (Gao, Han, Li & Zhou, 2018)

- **Citation:** Lei Gao, Yufeng Han, Sophia Zhengzi Li, Guofu Zhou (2018). *Journal of Financial Economics* 129(2), 394-414. DOI [10.1016/j.jfineco.2018.05.009](https://doi.org/10.1016/j.jfineco.2018.05.009)
- **Verified:** Crossref and the full RePEc abstract, 2026-11-10. Numbers marked (m) come from memory.
- **Topic:** 2. TradFi session effects. Hypothesis: H-TF4 (and H-NS5 for the cross-asset version).

## Question
Does the market's return early in the day predict its return at the end of the day?

## Data and method
- SPY high-frequency data, 1993-2013, in half-hour intervals.
- r_first = return from the previous close to 10:00 ET (it includes the overnight gap); r_last = return over the last half hour (15:30-16:00).
- Predictive regression r_last = a + b r_first (+ b2 r_12th half-hour) + e, OLS with robust SEs. Out-of-sample R-squared vs the historical mean; a market-timing strategy (long / short in the last half hour by the sign of r_first).
- Splits by volatility, volume, recessions and macro-news days; 10 other ETFs.

## Key results
- The first half-hour return predicts the last half-hour return, statistically and economically (abstract). Predictive R-squared is ~1.6% in-sample for this one regressor (m), large for an intraday return regression.
- Stronger on high-vol, high-volume, recession and major macro-news days (abstract).
- Present in 10 other liquid ETFs (abstract).
- Proposed mechanisms: infrequent portfolio rebalancing (Bogousslavsky 2016) and late-informed trading near the close. Baltussen et al. (2021) later attribute it to gamma hedging (see [baltussen_da_lammers_martens2021_hedging_intraday_momentum.md](baltussen_da_lammers_martens2021_hedging_intraday_momentum.md)).

## Limitations / transfer to Bybit perps
- The predictor includes the overnight gap. On equity perps, "previous close to first hour" is a traded period, so the information content differs.
- The mechanism (dealers rebalancing hedges and leveraged ETFs into the close) operates in the cash market. The perp tracks the underlying, so the effect could transmit through the index price, but the perp's own flow is crypto-native.
- With 1h bars we can only test first hour -> last hour (H-TF4 as registered); half-hour resolution needs 3s flow data (flow_3s covers the TradFi perps but only from 2026-11-07).
- Several later papers report weakening after 2013 in the US (not verified here; treat as possible decay).

## How we use it here
- B2 H-TF4 as registered, plus two conditioning splits the paper says matter: high-vol days (top tercile of the HAR-forecast sigma) and macro-news days (CPI / NFP / FOMC from `research/calendar/`).
- If H-TF4 is supported, the strategy is mechanical: at 15:00 ET, take the sign of the first-hour return and hold one hour. Taker costs (7.5 bp per side) are likely larger than the edge, so test a maker entry using the H-SIM1 fill model.

## Evidence quality
**Strong** in its sample (JFE, replicated across ETFs, mechanism-backed). **Medium-weak** for Bybit perps.
