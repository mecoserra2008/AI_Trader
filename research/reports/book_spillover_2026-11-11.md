# Order books and spillovers: spread vs own future volatility (H-MS5), ETH / SOL transmission (H-MS6)

_Agent C2, microstructure pilot, 2026-11-11. Data: one UTC day, 2026-11-10 (CPI day). Every result is segment **clean_holdout** because the whole day is after the knowledge cutoff. FIT = 00:00-12:00 UTC, TEST = 12:00-24:00 UTC. TEST contains the 13:30 UTC CPI release, so robustness rows drop 13:00-16:00. This is one day and one market state (a crypto sell-off), so treat every number as a first measurement._

Code: `src/ai_trader/sim/book.py` (ob200 reconstruction, 100 ms series, trade validation), `src/ai_trader/research/book_spillover_study.py` (tests and tables). Tests: `tests/test_book.py`. Book series: `data/raw/pilot_2026-11-10/book_100ms/<SYMBOL>.parquet` (12 symbols, 864,001 rows each, plus `_qa.json`). Machine-readable results: `research/reports/book_spillover_2026-11-11.json`.

## Summary

- **H-MS5 contradicts.** On FIT the spread predicts future RV: log spread coefficient +0.25 (t 2.74) at 5 min and +0.19 (t 2.74) at 15 min. On TEST the sign flips: -0.17 (t -1.60) at 5 min and -0.32 (t -3.00) at 15 min. Adding the spread to the FIT model lowers the out-of-sample R^2 at both horizons (gain -0.0016 / -0.0014). Dropping 13:00-16:00 makes it worse (-0.16, t -2.75 / -0.25, t -4.94). The reason is mechanical: 8 of the 12 L2 symbols sit at one tick more than 96% of the time (BTC, ETH, SOL, XRP, DOGE, HYPE, ATOM, XAU). For them the relative spread is tick / mid, a price-level variable rather than a liquidity signal. Where the spread does move (CARV, KAS, NVDA, an exploratory row that was not registered) the pooled coefficient is +0.12 (t 1.05) at 5 min and -0.10 (t -0.87) at 15 min on TEST. Its out-of-sample R^2 gain is positive for all three symbols (+0.0086 / +0.0056, DM t 3.4 / 4.0).
- **H-MS6 contradicts.** The return-lead part fails. The Hayashi-Yoshida cross-correlation peaks at lag 0 for 38 of 39 crypto targets for ETH and 36 of 39 for SOL on TEST. **No** target has its peak in [0.1, 30] s (0/39 for both; the threshold was > 70%). A fine-lag check (10-500 ms) puts the peak within ±20 ms of zero. The asymmetry at ±0.1-0.5 s even leans slightly *toward the target* (median LLR 0.76 for ETH, 0.93 for SOL). The volatility part holds. In the Diebold-Yilmaz table of log 1 m RV on TEST, ETH (net +25.0, CI [13.1, 33.3]) and SOL (+21.6, CI [10.9, 31.4]) are net transmitters, along with DOGE (+27.0). BTC is +10.2 with a CI that includes 0. The registered verdict needs both parts, so H-MS6 is contradicted. Holm over the two hypotheses: both adjusted p = 1.0.
- **The 1 s Granger "lead" is a stale-price artefact.** ETH Granger-causes 35 of 39 targets on TEST (robust p < 0.01), against 2 of 39 in the reverse direction. HY, which handles asynchronous trading, shows no lead. Illiquid tokens' last-trade prices catch up with a move that has already happened in their quotes. Book mid-quotes (no bounce, 100 ms resolution) also peak at 0-0.1 s.
- **Spillover map** (TEST): for prices it is a synchronous common factor. Same-ms HY correlation with ETH is 0.08-0.46 on trade prices, and the HY correlation of 100 ms book-mid changes with ETH is 0.37-0.80 for crypto. Volatility flows from ETH, SOL and DOGE (then BTC, HYPE) to the rest. Total connectedness is 88% on TEST against 67% on FIT, so the CPI half is one vol factor. The weakest receivers and transmitters are INJ, ARB and ZEC. ETH and SOL each explain about 7-9% of every other token's 10-step vol forecast-error variance.
- **TradFi perps.** Crypto-linked equities (MSTR, COIN, CRWV) move synchronously with crypto (HY rho0 0.16-0.51). QQQ and TSLA peak at +0.1-0.3 s behind ETH, SOL and BTC (LLR 0.9-1.6), a small lag. Gold, silver and oil are nearly uncorrelated at these horizons (rho < 0.13; oil's +1.5-2 s peaks have rho of about 0.02 to 0.04). In the vol-connectedness table MSTR is a net transmitter (+34.5) and XAU, XAG and CRWV are net receivers.
- **Book data quality.** No u gaps, no crossed books, 2 snapshots per file. Trade prices lie outside the 100 ms book's [bid1, ask1] for 1-19% of trades if the state before *or* after counts (loose check), and for 0.2-4% outside multi-price same-ms sweeps. 81-94% of the violations are sweeps. The strict check, which uses only the state before the trade, fails for 11-64%. The feed is a 100 ms conflation, so trades inside a push interval routinely move the touch.

## Verdicts (clean_holdout, Holm over H-MS5 and H-MS6)

| id | registered success | key numbers (TEST) | p (one-sided) | p Holm | verdict |
|---|---|---|---|---|---|
| H-MS5 | coef > 0 with t > 2 on TEST and OOS R^2 gain > 0 (5 and 15 min) | coef -0.174 (t -1.60) / -0.323 (t -3.00); gain -0.0016 / -0.0014 (DM t -1.74 / -1.85) | 0.999 | 1.000 | **contradicts** |
| H-MS6 | ETH and SOL: positive HY peak (0.1-30 s) for > 70% of targets, and net DY > 0 on TEST | positive peaks 0/39 (ETH), 0/39 (SOL); net +25.0 [13.1, 33.3] / +21.6 [10.9, 31.4] | 1.000 | 1.000 | **contradicts** (lead part) |

Verdict rules were fixed in the module before the first full run (`METHOD` in the module). H-MS5 contradicts when the TEST coefficient has t < -1.645 or the gain's DM p > 0.95 at either horizon. H-MS6 contradicts when either leader's positive-peak share is <= 0.5 or its net CI lies entirely below 0. The p for H-MS5 is the maximum over horizons of the coefficient and DM p values. The p for H-MS6 is the maximum over ETH and SOL of the binomial p (positive peaks vs 0.5) and P*(net <= 0). The registry's Holm family is the whole MS block; the other MS hypotheses belong to another agent, and with two contradicted hypotheses here the adjustment changes nothing.

## Methods

- **split**: FIT 2026-11-10 00:00-12:00 UTC, TEST 12:00-24:00 UTC; robustness: TEST without 13:00-16:00 (CPI 13:30)
- **book**: sim/book.py: ob200 snapshot + deltas rebuilt per message; state at the end of each 100 ms bin of the push ts; depth = notional within x bp of mid
- **ms5_y**: log RV over (T, T+h], h = 5, 15 min, sum of squared 1 s mid log returns; zero RV floored at the symbol's 1st percentile of positive values
- **ms5_x**: log time-averaged relative spread (bp) over the minute ending at T; robustness: spread level in bp
- **ms5_controls**: log RV last 5 min, log RV last 60 min, log(depth_bid_10 + depth_ask_10) averaged over the last minute, log hour-of-week factor (profile_asof 2026-11-10 00:00, u = r^2 / trailing-168h mean r^2 on 1h last bars; pooled crypto-major profile for symbols with < 8 weeks of bars)
- **ms5_model**: pooled OLS, symbol fixed effects, CR1 SE clustered by 5-minute calendar block (robustness: 30-minute blocks); OOS R^2 gain of the FIT model on TEST; Diebold-Mariano t on 5-minute block sums of the squared-error differential
- **ms6_prices**: trade-time log price = per-ms VWAP (trades_raw stores no within-ms order); 1 s and 5 s grids = last trade price at or before the grid time
- **ms6_hy**: Hayashi-Yoshida rho(l) = corr(leader at t, target at t+l), lags 0, +-0.1..60 s (37); peak = argmax rho; positive peak = peak in [0.1, 30] s; 5-minute block bootstrap B = 1000 of HY contributions (peak-lag distribution, LLR = sum_{l>0} rho^2 / sum_{l<0} rho^2, p = P*(LLR <= 1))
- **ms6_granger**: bivariate VAR on 1 s last-trade log returns, BIC lag <= 10; HC0-robust Wald F of the other series' lags (classical F reported too), both directions
- **ms6_dy**: VAR (BIC lag <= 4) on log 1 m RV (12 x 5 s returns), generalized FEVD (Pesaran-Shin) H = 10, row-normalised; net = to - from (percent); 30-minute moving-block bootstrap B = 500 with the lag fixed
- **ms6_universe**: crypto = the 51-symbol trades set minus TradFi (sim.flow_data.asset_class); DY set = BTC, ETH, SOL + crypto with nonzero 1 m RV in >= 98% of the minutes of both halves, top by trade count, at most 15
- **verdict_ms5**: supports = TEST coefficient > 0 with t > 2 and OOS R^2 gain > 0, both horizons; contradicts = coefficient <= 0 or gain <= 0 at a horizon with the one-sided p of the opposite > 0.95 (clearly wrong sign); else inconclusive
- **verdict_ms6**: supports = share of crypto targets with a positive peak > 0.70 for ETH and SOL and both DY net > 0 on TEST; contradicts = a share <= 0.50 or a net CI entirely below 0; else inconclusive
- **holm**: Holm over H-MS5, H-MS6; p(H-MS5) = max over horizons of the one-sided coefficient p and the DM p; p(H-MS6) = max over ETH, SOL of the binomial p (positive peaks vs 0.5) and the bootstrap P*(net <= 0)

## Book reconstruction and validation against trades

The parser streams each zip line by line. A snapshot clears both sides; in a delta, a level with size 0 is removed and any other size replaces the level. The state at the end of every 100 ms bin of the push time `ts` is stored with `ts` = bin end (known at `ts`), and empty bins carry the previous state forward with `n_updates = 0`. Depth is quoted notional within 5 / 10 / 25 bp of the mid. `span_*_bp` is the distance of the deepest level held: a 200-level feed truncates any band wider than the span (BTC: 4.1 bp, so all three BTC bands equal the whole visible book; ETH and XAU: 25 bp truncated).

| symbol | messages | snapshots | u gaps | crossed bins | trades | strict outside [bid1, ask1] | loose (before or after) | sweep share | loose ex sweeps | violations in sweeps | wrong side |
|---|---|---|---|---|---|---|---|---|---|---|---|
| BTCUSDT | 863808 | 2 | 0 | 0 | 2098300 | 61.13% | 7.58% | 53.7% | 2.58% | 84% | 4.17% |
| ETHUSDT | 863046 | 2 | 0 | 0 | 1790465 | 62.23% | 11.48% | 59.3% | 4.01% | 86% | 6.18% |
| SOLUSDT | 820711 | 2 | 0 | 0 | 564853 | 32.97% | 6.16% | 21.0% | 1.45% | 81% | 2.67% |
| XRPUSDT | 847011 | 2 | 0 | 0 | 689236 | 38.59% | 6.24% | 22.3% | 1.38% | 83% | 1.96% |
| DOGEUSDT | 714308 | 2 | 0 | 0 | 236133 | 27.96% | 5.25% | 18.3% | 1.11% | 83% | 1.24% |
| HYPEUSDT | 823712 | 2 | 0 | 0 | 648142 | 63.84% | 19.38% | 62.1% | 3.49% | 93% | 3.35% |
| CARVUSDT | 109365 | 2 | 0 | 0 | 6905 | 28.01% | 2.38% | 26.8% | 0.40% | 88% | 0.17% |
| KASUSDT | 267166 | 2 | 0 | 0 | 33063 | 33.59% | 3.77% | 30.3% | 0.66% | 88% | 0.24% |
| ATOMUSDT | 285722 | 2 | 0 | 0 | 32285 | 10.81% | 1.17% | 8.1% | 0.18% | 86% | 0.37% |
| NVDAUSDT | 672538 | 2 | 0 | 0 | 285059 | 41.40% | 8.71% | 32.9% | 1.53% | 88% | 2.08% |
| QQQUSDT | 616688 | 2 | 0 | 0 | 62969 | 37.63% | 11.32% | 35.5% | 1.14% | 94% | 2.04% |
| XAUUSDT | 672623 | 2 | 0 | 0 | 157179 | 29.70% | 7.90% | 28.8% | 1.80% | 84% | 3.84% |

Strict = the book state with matching-engine time `cts` <= trade time. Loose = inside the union of the states before and after. Sweep = same-ms, same-side trades at more than one price. Wrong side = a buy below bid1 or a sell above ask1 of the state before. The loose violation rate is the reconstruction check: 1-19% overall and 0.2-4% outside sweeps, so the book is right to within the 100 ms conflation of the feed. For the simulator: a 100 ms ob200 state is not the book a trade met. Fill and markout models need the trade itself (side, price) and should never infer the touch at trade time from the last push.

## H-MS5: spread and the symbol's own future volatility

Pooled 1 m panel, 12 symbols, symbol fixed effects, SE clustered by 5-minute block. The hour-of-week factor uses each symbol's own 1h profile, except CARV and KAS, which have 98 bars and use the pooled BTC/ETH/SOL/XRP/DOGE profile.

| spec | h | n FIT / TEST | FIT coef (t) | TEST coef (t) | OOS R^2 full | OOS R^2 without spread | gain | DM t |
|---|---|---|---|---|---|---|---|---|
| primary (log spread) | 5 | 7852 / 8561 | 0.252 (2.74) | -0.174 (-1.60) | 0.753 | 0.755 | -0.0016 | -1.74 |
| primary (log spread) | 15 | 7732 / 8441 | 0.192 (2.74) | -0.323 (-3.00) | 0.696 | 0.698 | -0.0014 | -1.85 |
| cluster30 | 15 | 7732 / 8441 | 0.192 (2.18) | -0.323 (-2.15) | 0.696 | 0.698 | -0.0014 | -1.04 |
| crypto_only | 15 | 5794 / 6323 | 0.003 (0.05) | -0.106 (-0.75) | 0.363 | 0.363 | 0.0001 | 6.55 |
| ex_cpi | 15 | 7732 / 6114 | 0.192 (2.74) | -0.253 (-4.94) | 0.836 | 0.838 | -0.0027 | -3.49 |
| level_spread | 15 | 7732 / 8441 | -0.010 (-1.03) | -0.024 (-1.40) | 0.697 | 0.698 | -0.0002 | -0.97 |
| not_tick_bound | 15 | 1918 / 2087 | 0.108 (1.71) | -0.099 (-0.87) | 0.399 | 0.393 | 0.0056 | 4.01 |
| cluster30 | 5 | 7852 / 8561 | 0.252 (2.19) | -0.174 (-1.61) | 0.753 | 0.755 | -0.0016 | -1.11 |
| crypto_only | 5 | 5884 / 6413 | 0.049 (0.53) | 0.100 (0.61) | 0.500 | 0.498 | 0.0018 | 5.91 |
| ex_cpi | 5 | 7852 / 6354 | 0.252 (2.74) | -0.160 (-2.75) | 0.809 | 0.812 | -0.0029 | -3.18 |
| level_spread | 5 | 7852 / 8561 | -0.010 (-0.80) | -0.012 (-0.53) | 0.754 | 0.755 | -0.0003 | -1.20 |
| not_tick_bound | 5 | 1948 / 2117 | 0.240 (2.56) | 0.122 (1.05) | 0.434 | 0.426 | 0.0086 | 3.40 |

The rows `crypto_only` and `not_tick_bound` (CARV, KAS and NVDA, whose spread is at one tick less than 90% of the time) are exploratory and were not registered. `level_spread` uses the spread in bp instead of its log. `ex_cpi` drops TEST rows whose forecast window touches 13:00-16:00.

Controls (coefficient, t):

| control | FIT h5 | TEST h5 | FIT h15 | TEST h15 |
|---|---|---|---|---|
| lrv5p | 0.339 (7.59) | 0.522 (10.53) | 0.236 (5.56) | 0.377 (8.07) |
| lrv60p | 0.274 (5.72) | 0.168 (1.63) | 0.278 (7.10) | 0.103 (0.94) |
| ldepth10 | 0.048 (1.33) | -0.095 (-1.21) | 0.052 (1.76) | -0.075 (-1.27) |
| lhow | 0.215 (1.80) | 0.782 (2.85) | 0.194 (1.72) | 1.378 (4.10) |

Per symbol (same regression without fixed effects; the OOS gain is the pooled FIT model's gain on that symbol's TEST rows):

| symbol | h5 FIT coef (t) | h5 TEST coef (t) | h15 FIT coef (t) | h15 TEST coef (t) | OOS gain h5 (pooled model) | OOS gain h15 |
|---|---|---|---|---|---|---|
| BTCUSDT | 0.191 (0.14) | 0.058 (0.40) | 0.549 (0.45) | -0.297 (-1.36) | -0.0015 | -0.0022 |
| ETHUSDT | 6.436 (3.07) | -0.842 (-1.23) | 6.569 (3.68) | -1.107 (-1.65) | -0.0069 | -0.0070 |
| SOLUSDT | 9.088 (0.83) | -2.041 (-0.63) | 23.239 (1.90) | -7.468 (-1.55) | -0.0100 | -0.0096 |
| XRPUSDT | 5.898 (1.14) | -2.437 (-0.63) | 11.746 (2.63) | -3.149 (-0.80) | -0.0076 | -0.0062 |
| DOGEUSDT | 22.825 (2.14) | -1.264 (-0.44) | 27.294 (2.94) | -4.731 (-1.13) | -0.0114 | -0.0110 |
| HYPEUSDT | 1.280 (2.15) | 0.185 (0.50) | 1.559 (2.78) | -0.070 (-0.22) | 0.0150 | 0.0134 |
| CARVUSDT | 0.197 (1.24) | 0.641 (3.60) | 0.103 (0.87) | 0.374 (2.28) | 0.0121 | 0.0061 |
| KASUSDT | 0.098 (0.85) | 0.162 (0.84) | -0.009 (-0.14) | 0.122 (0.71) | 0.0122 | 0.0177 |
| ATOMUSDT | 1.116 (1.38) | 1.350 (2.50) | 0.863 (1.26) | 0.887 (2.17) | 0.0037 | 0.0011 |
| NVDAUSDT | 0.406 (1.86) | -0.004 (-0.02) | 0.138 (0.51) | 0.067 (0.27) | 0.0086 | 0.0116 |
| QQQUSDT | -0.097 (-0.66) | -0.303 (-1.99) | -0.025 (-0.18) | -0.431 (-2.98) | -0.0073 | -0.0058 |
| XAUUSDT | 0.320 (2.38) | -0.436 (-3.11) | 0.256 (2.14) | -0.346 (-2.74) | -0.0220 | -0.0160 |

Reading: past RV carries the forecast (R^2 OOS 0.70-0.75), and the hour-of-week factor matters most in the CPI half (coefficient 1.38 at 15 min, t 4.1). The spread adds nothing out of sample. In the tick-bound majors and XAU the "spread" is 1 tick / mid. On a falling-price day a wider relative spread means a lower price, not more expected volatility: XAU -0.44 (t -3.1) and QQQ -0.30 (t -2.0) on TEST. Where the spread is free to move it does lead volatility on TEST: CARV +0.64 (t 3.6) and ATOM +1.35 (t 2.5) at 5 min, and the alts KAS, CARV and HYPE have positive per-symbol OOS gains. With 12 symbols and one day this is a hypothesis for the forward record, not a finding.

## H-MS6: lead-lag of returns

### Summary by leader

| leader | half | targets | peak at 0 | peak in [0.1, 30] s | peak < 0 | median LLR | LLR > 1, P*(LLR<=1) < 0.05 | LLR < 1, P*(LLR<=1) > 0.95 | Granger L->T p < 0.01 | T->L p < 0.01 |
|---|---|---|---|---|---|---|---|---|---|---|
| ETHUSDT -> crypto | fit | 39 | 24 | 0 | 15 | 0.85 | 4 | 22 | 38 | 21 |
| ETHUSDT -> crypto | test | 39 | 38 | 0 | 1 | 0.76 | 2 | 25 | 35 | 2 |
| ETHUSDT -> crypto | test_ex_cpi | 39 | 23 | 1 | 15 | 0.79 | 2 | 25 | 38 | 16 |
| SOLUSDT -> crypto | fit | 39 | 29 | 5 | 5 | 0.95 | 10 | 12 | 37 | 36 |
| SOLUSDT -> crypto | test | 39 | 36 | 0 | 3 | 0.93 | 6 | 18 | 36 | 23 |
| SOLUSDT -> crypto | test_ex_cpi | 39 | 35 | 1 | 3 | 0.95 | 9 | 17 | 37 | 32 |
| BTCUSDT -> crypto | fit | 39 | 13 | 0 | 26 | 0.85 | 4 | 22 | 38 | 22 |
| BTCUSDT -> crypto | test | 39 | 35 | 0 | 4 | 0.78 | 3 | 25 | 38 | 3 |
| BTCUSDT -> crypto | test_ex_cpi | 39 | 18 | 1 | 20 | 0.83 | 2 | 25 | 38 | 12 |
| ETHUSDT -> tradfi | fit | 11 | 6 | 3 | 2 | 0.94 | 4 | 4 | 11 | 5 |
| ETHUSDT -> tradfi | test | 11 | 4 | 4 | 3 | 1.05 | 2 | 4 | 8 | 1 |
| SOLUSDT -> tradfi | fit | 11 | 7 | 2 | 1 | 0.92 | 4 | 4 | 10 | 11 |
| SOLUSDT -> tradfi | test | 11 | 4 | 5 | 2 | 1.01 | 1 | 4 | 8 | 5 |
| BTCUSDT -> tradfi | fit | 11 | 6 | 3 | 2 | 0.93 | 4 | 5 | 11 | 6 |
| BTCUSDT -> tradfi | test | 11 | 6 | 4 | 1 | 0.94 | 2 | 4 | 8 | 1 |

"peak" = argmax over the 37-lag grid of HY rho(l) = corr(leader at t, target at t + l). LLR = sum over positive lags of rho^2 / sum over negative lags; LLR > 1 means the leader leads. Bootstrap: 5-minute blocks, B = 1000. Granger: robust Wald F of the other series' lags, 1 s last-trade returns, BIC lag <= 10. `test_ex_cpi` drops the 13:00-16:00 blocks for HY and uses 16:00-24:00 for Granger.

### Fine lags (diagnostic, not on the registered grid)

| leader | half | median rho over crypto targets at lag -500 / -200 / -100 / -50 / -20 / -10 / 0 / 10 / 20 / 50 / 100 / 200 / 500 ms |
|---|---|---|
| ETHUSDT | fit | 0.433 / 0.449 / 0.483 / 0.499 / 0.514 / 0.514 / 0.509 / 0.505 / 0.501 / 0.493 / 0.485 / 0.476 / 0.432 |
| ETHUSDT | test | 0.160 / 0.214 / 0.234 / 0.252 / 0.276 / 0.285 / 0.296 / 0.271 / 0.250 / 0.237 / 0.218 / 0.190 / 0.149 |
| SOLUSDT | fit | 0.284 / 0.307 / 0.326 / 0.341 / 0.346 / 0.343 / 0.351 / 0.357 / 0.352 / 0.338 / 0.330 / 0.327 / 0.307 |
| SOLUSDT | test | 0.192 / 0.236 / 0.270 / 0.282 / 0.305 / 0.313 / 0.329 / 0.317 / 0.301 / 0.289 / 0.266 / 0.244 / 0.187 |
| BTCUSDT | fit | 0.479 / 0.497 / 0.532 / 0.539 / 0.539 / 0.539 / 0.534 / 0.530 / 0.528 / 0.518 / 0.511 / 0.489 / 0.448 |
| BTCUSDT | test | 0.238 / 0.317 / 0.359 / 0.370 / 0.394 / 0.409 / 0.406 / 0.376 / 0.359 / 0.340 / 0.323 / 0.281 / 0.212 |

The cross-correlation is flat within ±20 ms and decays on both sides. On FIT the maximum is at -10 to -20 ms for ETH and BTC, which is why 15 of 39 (ETH) and 26 of 39 (BTC) FIT peaks sit at small negative lags on the coarse grid. Ten milliseconds is inside the timestamp and matching jitter between books. There is no exploitable sub-minute lead in either direction.

### Book mid-quotes (100 ms, no bid-ask bounce), L2 symbols

| pair (book mid changes, 100 ms) | FIT peak s / rho / LLR | TEST peak s / rho / LLR |
|---|---|---|
| ETHUSDT->BTCUSDT | 0.0 / 0.773 / 0.89 | 0.0 / 0.682 / 0.53 |
| ETHUSDT->SOLUSDT | 0.0 / 0.804 / 0.73 | 0.0 / 0.769 / 0.64 |
| ETHUSDT->XRPUSDT | 0.0 / 0.719 / 0.77 | 0.0 / 0.677 / 0.83 |
| ETHUSDT->DOGEUSDT | 0.0 / 0.826 / 0.70 | 0.0 / 0.713 / 0.70 |
| ETHUSDT->HYPEUSDT | 0.0 / 0.495 / 1.05 | 0.0 / 0.562 / 1.06 |
| ETHUSDT->CARVUSDT | 0.3 / 0.273 / 1.01 | 0.1 / 0.372 / 0.99 |
| ETHUSDT->KASUSDT | 0.0 / 0.477 / 0.82 | 0.1 / 0.447 / 0.99 |
| ETHUSDT->ATOMUSDT | -0.5 / 0.550 / 0.42 | 0.0 / 0.640 / 0.37 |
| ETHUSDT->NVDAUSDT | 0.0 / 0.336 / 1.05 | 0.0 / 0.256 / 1.14 |
| ETHUSDT->QQQUSDT | 0.0 / 0.493 / 0.89 | 0.0 / 0.353 / 0.82 |
| ETHUSDT->XAUUSDT | -0.1 / 0.257 / 0.70 | -0.2 / 0.191 / 0.54 |
| SOLUSDT->BTCUSDT | -0.1 / 0.793 / 1.19 | -0.1 / 0.716 / 1.02 |
| SOLUSDT->ETHUSDT | 0.0 / 0.804 / 1.36 | 0.0 / 0.769 / 1.57 |
| SOLUSDT->XRPUSDT | -0.1 / 0.777 / 1.00 | 0.0 / 0.765 / 1.19 |
| SOLUSDT->DOGEUSDT | -0.1 / 0.870 / 0.89 | 0.0 / 0.808 / 1.05 |
| SOLUSDT->HYPEUSDT | 0.1 / 0.536 / 1.35 | 0.0 / 0.619 / 1.66 |
| SOLUSDT->CARVUSDT | 0.2 / 0.311 / 1.16 | 0.2 / 0.402 / 1.17 |
| SOLUSDT->KASUSDT | 0.3 / 0.528 / 1.07 | 0.1 / 0.505 / 1.25 |
| SOLUSDT->ATOMUSDT | -0.8 / 0.563 / 0.49 | 0.0 / 0.692 / 0.46 |
| SOLUSDT->NVDAUSDT | 0.1 / 0.336 / 1.24 | 0.0 / 0.306 / 1.65 |
| SOLUSDT->QQQUSDT | 0.0 / 0.509 / 1.12 | 0.0 / 0.426 / 1.30 |
| SOLUSDT->XAUUSDT | -0.1 / 0.265 / 0.81 | 0.0 / 0.196 / 0.76 |
| BTCUSDT->ETHUSDT | 0.0 / 0.773 / 1.13 | 0.0 / 0.682 / 1.88 |
| BTCUSDT->SOLUSDT | 0.1 / 0.793 / 0.84 | 0.1 / 0.716 / 0.98 |
| BTCUSDT->XRPUSDT | 0.0 / 0.707 / 0.88 | 0.1 / 0.693 / 1.22 |
| BTCUSDT->DOGEUSDT | 0.0 / 0.834 / 0.77 | 0.1 / 0.770 / 0.97 |
| BTCUSDT->HYPEUSDT | 0.0 / 0.492 / 1.08 | 0.0 / 0.558 / 1.77 |
| BTCUSDT->CARVUSDT | 0.3 / 0.304 / 1.03 | 0.2 / 0.411 / 1.04 |
| BTCUSDT->KASUSDT | 0.2 / 0.489 / 0.95 | 0.1 / 0.470 / 1.07 |
| BTCUSDT->ATOMUSDT | -0.8 / 0.566 / 0.45 | 0.1 / 0.682 / 0.44 |
| BTCUSDT->NVDAUSDT | 0.0 / 0.335 / 1.09 | 0.1 / 0.273 / 1.99 |
| BTCUSDT->QQQUSDT | 0.0 / 0.490 / 0.95 | 0.0 / 0.407 / 1.20 |
| BTCUSDT->XAUUSDT | -0.1 / 0.273 / 0.74 | 0.0 / 0.222 / 0.74 |

On mid-quotes BTC and SOL lead ETH slightly (BTC->ETH LLR 1.88 on TEST, SOL->ETH 1.57), but every peak is at 0-0.1 s, so the lead is within one feed interval.

### Per target (TEST)

Peak s / peak rho / rho at 0 / LLR, the bootstrap probability that the peak is in [0.1, 30] s, and robust Granger F (p):

| target | n ms (day) | ETH peak s / rho / rho0 / LLR | ETH boot P(+peak) | ETH->T F (p) | T->ETH F (p) | SOL peak s / rho / rho0 / LLR | SOL boot P(+peak) | SOL->T F (p) | T->SOL F (p) | BTC peak s / LLR | BTC->T F (p) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| BTCUSDT | 596722 | 0.0 / 0.187 / 0.187 / 0.84 | 0.00 | 0.74 (0.683) | 1.37 (0.186) | 0.0 / 0.293 / 0.293 / 1.26 | 0.00 | 2.16 (0.027) | 14.14 (0.000) | (self) |  |
| ETHUSDT | 467355 | (self) |  |  |  | 0.0 / 0.217 / 0.217 / 1.22 | 0.00 | 1.89 (0.056) | 3.43 (0.001) | 0.0 / 1.19 | 1.37 (0.186) |
| ZECUSDT | 423796 | 0.0 / 0.164 / 0.164 / 0.24 | 0.00 | 2.50 (0.007) | 1.19 (0.294) | 0.0 / 0.231 / 0.231 / 0.69 | 0.00 | 2.40 (0.014) | 11.19 (0.000) | 0.0 / 0.24 | 4.91 (0.002) |
| XRPUSDT | 163571 | 0.0 / 0.221 / 0.221 / 0.84 | 0.00 | 2.86 (0.001) | 1.78 (0.058) | 0.0 / 0.256 / 0.256 / 1.08 | 0.00 | 4.81 (0.000) | 7.56 (0.000) | 0.0 / 0.77 | 6.12 (0.000) |
| HYPEUSDT | 187863 | 0.0 / 0.261 / 0.261 / 1.07 | 0.00 | 3.41 (0.000) | 3.23 (0.001) | 0.0 / 0.297 / 0.297 / 1.21 | 0.00 | 4.73 (0.000) | 15.73 (0.000) | 0.0 / 0.93 | 14.00 (0.000) |
| SOLUSDT | 131884 | 0.0 / 0.217 / 0.217 / 0.82 | 0.00 | 3.43 (0.001) | 1.89 (0.056) | (self) |  |  |  | 0.0 / 0.80 | 14.14 (0.000) |
| 1000PEPEUSDT | 129586 | 0.0 / 0.255 / 0.255 / 0.54 | 0.00 | 4.93 (0.000) | 1.04 (0.407) | 0.0 / 0.276 / 0.276 / 0.77 | 0.00 | 5.90 (0.000) | 6.12 (0.000) | 0.0 / 0.51 | 16.78 (0.000) |
| ENAUSDT | 182239 | 0.0 / 0.254 / 0.254 / 0.28 | 0.00 | 2.04 (0.031) | 2.87 (0.002) | 0.0 / 0.300 / 0.300 / 0.68 | 0.00 | 2.97 (0.003) | 20.84 (0.000) | 0.0 / 0.28 | 5.83 (0.000) |
| ARBUSDT | 110079 | 0.0 / 0.248 / 0.248 / 0.76 | 0.00 | 3.11 (0.002) | 1.18 (0.308) | 0.0 / 0.296 / 0.296 / 1.03 | 0.00 | 5.13 (0.000) | 3.10 (0.001) | 0.0 / 0.77 | 11.58 (0.000) |
| NEARUSDT | 73708 | 0.0 / 0.296 / 0.296 / 0.67 | 0.00 | 6.34 (0.000) | 1.73 (0.077) | 0.0 / 0.339 / 0.339 / 0.82 | 0.00 | 11.88 (0.000) | 9.21 (0.000) | 0.0 / 0.67 | 27.10 (0.000) |
| AVAXUSDT | 75428 | 0.0 / 0.273 / 0.273 / 0.65 | 0.00 | 3.62 (0.000) | 1.11 (0.349) | 0.0 / 0.330 / 0.330 / 0.92 | 0.00 | 4.63 (0.000) | 5.10 (0.000) | 0.0 / 0.72 | 15.82 (0.000) |
| DOGEUSDT | 60637 | 0.0 / 0.293 / 0.293 / 0.89 | 0.00 | 5.85 (0.000) | 1.23 (0.270) | 0.0 / 0.355 / 0.355 / 1.01 | 0.00 | 14.88 (0.000) | 4.40 (0.000) | 0.0 / 0.95 | 35.62 (0.000) |
| INJUSDT | 72374 | 0.0 / 0.276 / 0.276 / 0.53 | 0.00 | 7.24 (0.000) | 1.16 (0.314) | 0.0 / 0.312 / 0.312 / 0.79 | 0.00 | 10.78 (0.000) | 4.65 (0.000) | 0.0 / 0.53 | 25.61 (0.000) |
| UNIUSDT | 68695 | 0.0 / 0.300 / 0.300 / 0.57 | 0.00 | 3.96 (0.000) | 2.24 (0.017) | 0.0 / 0.345 / 0.345 / 0.83 | 0.00 | 8.38 (0.000) | 10.13 (0.000) | 0.0 / 0.61 | 15.37 (0.000) |
| AAVEUSDT | 73932 | 0.0 / 0.297 / 0.297 / 0.98 | 0.00 | 3.24 (0.001) | 1.41 (0.176) | 0.0 / 0.329 / 0.329 / 1.01 | 0.00 | 4.07 (0.000) | 6.20 (0.000) | 0.0 / 0.95 | 9.96 (0.000) |
| SUIUSDT | 62935 | 0.0 / 0.333 / 0.333 / 0.42 | 0.00 | 2.05 (0.030) | 0.80 (0.616) | 0.0 / 0.365 / 0.365 / 0.74 | 0.00 | 4.99 (0.000) | 12.05 (0.000) | 0.0 / 0.54 | 12.10 (0.000) |
| LINKUSDT | 59253 | 0.0 / 0.305 / 0.305 / 0.69 | 0.00 | 3.81 (0.000) | 0.95 (0.477) | 0.0 / 0.314 / 0.314 / 0.90 | 0.00 | 8.01 (0.000) | 3.25 (0.000) | 0.0 / 0.73 | 19.80 (0.000) |
| TAOUSDT | 68822 | 0.0 / 0.352 / 0.352 / 0.97 | 0.00 | 5.67 (0.000) | 0.99 (0.441) | 0.0 / 0.396 / 0.396 / 1.02 | 0.00 | 8.61 (0.000) | 4.65 (0.000) | 0.0 / 0.95 | 27.75 (0.000) |
| ONDOUSDT | 43507 | 0.0 / 0.329 / 0.329 / 0.67 | 0.00 | 3.89 (0.000) | 1.42 (0.172) | 0.0 / 0.360 / 0.360 / 0.88 | 0.00 | 6.20 (0.000) | 4.97 (0.000) | 0.0 / 0.68 | 20.34 (0.000) |
| WLDUSDT | 43065 | 0.0 / 0.339 / 0.339 / 0.60 | 0.00 | 2.34 (0.009) | 1.25 (0.255) | 0.0 / 0.366 / 0.366 / 0.72 | 0.00 | 4.57 (0.000) | 1.47 (0.145) | 0.0 / 0.57 | 15.32 (0.000) |
| GRAMUSDT | 28024 | 0.0 / 0.153 / 0.153 / 0.88 | 0.00 | 9.97 (0.000) | 1.23 (0.268) | 0.0 / 0.137 / 0.137 / 1.04 | 0.01 | 13.60 (0.000) | 1.44 (0.173) | 0.0 / 0.91 | 20.68 (0.000) |
| DOTUSDT | 43229 | 0.0 / 0.287 / 0.287 / 1.06 | 0.00 | 3.98 (0.000) | 0.86 (0.572) | 0.0 / 0.314 / 0.314 / 1.18 | 0.00 | 5.25 (0.000) | 3.00 (0.001) | 0.0 / 0.97 | 17.37 (0.000) |
| BNBUSDT | 49226 | 0.0 / 0.305 / 0.305 / 1.14 | 0.00 | 5.41 (0.000) | 0.85 (0.571) | 0.0 / 0.341 / 0.341 / 1.41 | 0.00 | 10.66 (0.000) | 3.12 (0.001) | 0.0 / 1.20 | 38.42 (0.000) |
| ADAUSDT | 26886 | 0.0 / 0.289 / 0.289 / 1.10 | 0.00 | 9.67 (0.000) | 1.41 (0.176) | 0.0 / 0.306 / 0.306 / 1.11 | 0.00 | 17.74 (0.000) | 1.61 (0.097) | 0.0 / 1.14 | 37.81 (0.000) |
| FILUSDT | 36770 | 0.0 / 0.351 / 0.351 / 0.74 | 0.00 | 2.19 (0.016) | 0.94 (0.493) | 0.0 / 0.382 / 0.382 / 0.99 | 0.00 | 4.76 (0.000) | 1.57 (0.109) | 0.0 / 0.86 | 11.55 (0.000) |
| LTCUSDT | 25990 | 0.0 / 0.397 / 0.397 / 0.80 | 0.00 | 7.75 (0.000) | 0.69 (0.717) | 0.0 / 0.403 / 0.403 / 0.94 | 0.00 | 11.85 (0.000) | 3.92 (0.000) | 0.0 / 0.80 | 31.99 (0.000) |
| TRUMPUSDT | 23777 | 0.0 / 0.274 / 0.274 / 0.90 | 0.00 | 6.23 (0.000) | 1.61 (0.097) | 0.0 / 0.268 / 0.268 / 0.93 | 0.00 | 9.99 (0.000) | 3.25 (0.000) | 0.0 / 0.93 | 21.16 (0.000) |
| ASTERUSDT | 25429 | 0.0 / 0.257 / 0.257 / 0.82 | 0.00 | 9.16 (0.000) | 1.51 (0.129) | 0.0 / 0.249 / 0.249 / 0.98 | 0.00 | 10.48 (0.000) | 2.02 (0.027) | 0.0 / 0.86 | 8.72 (0.000) |
| WLFIUSDT | 19690 | 0.0 / 0.200 / 0.200 / 1.03 | 0.00 | 5.20 (0.000) | 0.97 (0.461) | 0.0 / 0.199 / 0.199 / 1.00 | 0.01 | 8.38 (0.000) | 0.66 (0.725) | 0.0 / 0.98 | 23.56 (0.000) |
| CRVUSDT | 19092 | 0.0 / 0.329 / 0.329 / 0.74 | 0.00 | 4.89 (0.000) | 0.60 (0.799) | 0.0 / 0.342 / 0.342 / 0.86 | 0.00 | 6.70 (0.000) | 1.59 (0.101) | 0.0 / 0.78 | 16.05 (0.000) |
| XLMUSDT | 19102 | 0.0 / 0.351 / 0.351 / 0.95 | 0.00 | 12.62 (0.000) | 0.64 (0.784) | 0.0 / 0.333 / 0.333 / 1.01 | 0.00 | 25.14 (0.000) | 0.72 (0.706) | 0.0 / 1.02 | 38.55 (0.000) |
| MONUSDT | 16891 | 0.0 / 0.275 / 0.275 / 0.75 | 0.00 | 5.96 (0.000) | 0.96 (0.479) | 0.0 / 0.273 / 0.273 / 0.89 | 0.00 | 9.85 (0.000) | 1.56 (0.110) | -0.1 / 0.76 | 16.15 (0.000) |
| RENDERUSDT | 12347 | 0.0 / 0.396 / 0.396 / 0.69 | 0.00 | 4.85 (0.000) | 0.92 (0.508) | -0.1 / 0.385 / 0.382 / 0.73 | 0.00 | 7.89 (0.000) | 1.29 (0.229) | 0.0 / 0.74 | 27.24 (0.000) |
| ETCUSDT | 17199 | 0.0 / 0.461 / 0.461 / 0.75 | 0.00 | 8.83 (0.000) | 0.52 (0.880) | 0.0 / 0.467 / 0.467 / 0.85 | 0.00 | 16.70 (0.000) | 1.52 (0.124) | 0.0 / 0.77 | 26.10 (0.000) |
| KASUSDT | 14395 | 0.0 / 0.346 / 0.346 / 0.59 | 0.00 | 21.09 (0.000) | 1.57 (0.107) | 0.0 / 0.333 / 0.333 / 0.68 | 0.00 | 28.81 (0.000) | 2.46 (0.006) | -0.1 / 0.61 | 34.07 (0.000) |
| ATOMUSDT | 9676 | 0.0 / 0.390 / 0.390 / 0.92 | 0.00 | 8.34 (0.000) | 1.33 (0.210) | 0.0 / 0.371 / 0.371 / 0.93 | 0.00 | 15.53 (0.000) | 1.58 (0.105) | 0.0 / 0.92 | 24.95 (0.000) |
| WIFUSDT | 10507 | 0.0 / 0.419 / 0.419 / 0.76 | 0.00 | 6.73 (0.000) | 0.88 (0.545) | 0.0 / 0.419 / 0.419 / 0.82 | 0.00 | 9.93 (0.000) | 1.71 (0.072) | 0.0 / 0.78 | 24.60 (0.000) |
| TRXUSDT | 7990 | -0.2 / 0.087 / 0.080 / 0.88 | 0.09 | 2.44 (0.009) | 1.69 (0.085) | 0.0 / 0.068 / 0.068 / 1.12 | 0.35 | 2.59 (0.006) | 1.85 (0.055) | -0.2 / 1.03 | 4.99 (0.000) |
| MAVIAUSDT | 4712 | 0.0 / 0.387 / 0.387 / 0.68 | 0.00 | 6.58 (0.000) | 1.24 (0.259) | -0.1 / 0.394 / 0.390 / 0.79 | 0.00 | 10.61 (0.000) | 1.57 (0.127) | -0.1 / 0.75 | 18.82 (0.000) |
| CARVUSDT | 4418 | 0.0 / 0.362 / 0.362 / 0.87 | 0.00 | 5.46 (0.000) | 1.29 (0.232) | -0.2 / 0.322 / 0.321 / 0.88 | 0.28 | 6.37 (0.000) | 1.47 (0.145) | 0.0 / 0.80 | 6.54 (0.000) |

### TradFi perps (descriptive)

| target | n ms (day) | ETH peak s / rho / rho0 / LLR | ETH boot P(+peak) | ETH->T F (p) | T->ETH F (p) | SOL peak s / rho / rho0 / LLR | SOL boot P(+peak) | SOL->T F (p) | T->SOL F (p) | BTC peak s / LLR | BTC->T F (p) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| NVDAUSDT | 126272 | 0.0 / 0.173 / 0.173 / 0.69 | 0.00 | 7.32 (0.000) | 3.01 (0.001) | 0.0 / 0.184 / 0.184 / 0.75 | 0.00 | 8.04 (0.000) | 7.71 (0.000) | 0.0 / 0.70 | 17.74 (0.000) |
| XAUUSDT | 77105 | 0.0 / 0.059 / 0.059 / 4.65 | 0.35 | 4.01 (0.000) | 1.10 (0.360) | 0.3 / 0.053 / 0.050 / 1.94 | 0.63 | 5.25 (0.000) | 2.07 (0.035) | 0.0 / 3.92 | 4.39 (0.000) |
| CLUSDT | 65592 | 1.5 / 0.018 / -0.003 / 4.22 | 0.59 | 0.77 (0.646) | 1.73 (0.076) | 2.0 / 0.021 / -0.001 / 1.36 | 0.74 | 1.55 (0.135) | 2.17 (0.027) | 2.0 / 3.97 | 1.82 (0.140) |
| MSTRUSDT | 58408 | 0.0 / 0.390 / 0.390 / 0.51 | 0.00 | 4.72 (0.000) | 1.79 (0.064) | 0.0 / 0.387 / 0.387 / 0.55 | 0.00 | 6.73 (0.000) | 9.21 (0.000) | 0.0 / 0.49 | 34.33 (0.000) |
| CRWVUSDT | 43561 | -0.1 / 0.161 / 0.158 / 0.67 | 0.00 | 10.97 (0.000) | 1.38 (0.190) | -0.1 / 0.151 / 0.145 / 0.66 | 0.00 | 14.22 (0.000) | 3.58 (0.000) | 0.0 / 0.63 | 34.83 (0.000) |
| BZUSDT | 38461 | 1.5 / 0.030 / -0.003 / 1.61 | 0.83 | 1.40 (0.183) | 1.65 (0.095) | 2.0 / 0.036 / -0.005 / 1.86 | 0.98 | 2.03 (0.039) | 2.66 (0.006) | 2.0 / 2.82 | 0.93 (0.463) |
| XAGUSDT | 22879 | -0.1 / 0.114 / 0.106 / 0.91 | 0.01 | 1.74 (0.075) | 1.21 (0.282) | 0.0 / 0.125 / 0.125 / 0.91 | 0.05 | 2.22 (0.023) | 1.70 (0.092) | 0.0 / 0.92 | 1.35 (0.204) |
| QQQUSDT | 31623 | 0.3 / 0.252 / 0.244 / 1.60 | 0.80 | 10.53 (0.000) | 1.30 (0.223) | 0.2 / 0.255 / 0.251 / 1.41 | 0.62 | 15.63 (0.000) | 1.98 (0.037) | 0.2 / 1.50 | 25.30 (0.000) |
| GOOGLUSDT | 26052 | 0.0 / 0.229 / 0.229 / 1.15 | 0.01 | 4.54 (0.000) | 1.97 (0.038) | 0.0 / 0.224 / 0.224 / 1.10 | 0.32 | 5.26 (0.000) | 2.59 (0.008) | 0.0 / 1.12 | 10.30 (0.000) |
| TSLAUSDT | 27465 | 0.2 / 0.274 / 0.265 / 1.05 | 0.59 | 8.53 (0.000) | 1.54 (0.127) | 0.1 / 0.276 / 0.274 / 1.01 | 0.52 | 11.99 (0.000) | 2.19 (0.025) | 0.3 / 0.94 | 14.58 (0.000) |
| COINUSDT | 12472 | -0.1 / 0.523 / 0.508 / 0.69 | 0.00 | 8.93 (0.000) | 1.54 (0.119) | -0.1 / 0.488 / 0.480 / 0.71 | 0.00 | 20.82 (0.000) | 1.90 (0.055) | -0.1 / 0.71 | 32.24 (0.000) |

## H-MS6: volatility connectedness (Diebold-Yilmaz)

Set: BTC, ETH and SOL plus the 12 crypto tokens with the most trades among those with nonzero 1 m RV in >= 98% of minutes in both halves. That is ZEC, XRP, HYPE, 1000PEPE, ENA, ARB, NEAR, AVAX, DOGE, INJ, UNI and AAVE. Variables are log 1 m realised volatility (12 x 5 s last-trade returns). The VAR uses a BIC lag (1 in every table). FEVD is generalized and row-normalised with H = 10. CIs are a 30-minute moving-block bootstrap with B = 500.

TEST (registered):

| symbol | to others | from others | net | net 95% CI | P*(net <= 0) |
|---|---|---|---|---|---|
| BTCUSDT | 99.0 | 88.8 | 10.2 | [-2.7, 23.7] | 0.052 |
| ETHUSDT | 113.7 | 88.6 | 25.0 | [13.1, 33.3] | 0.000 |
| SOLUSDT | 110.7 | 89.1 | 21.6 | [10.9, 31.4] | 0.000 |
| ZECUSDT | 71.5 | 85.1 | -13.7 | [-25.7, 2.5] | 0.956 |
| XRPUSDT | 86.9 | 88.6 | -1.8 | [-13.6, 6.7] | 0.766 |
| HYPEUSDT | 95.4 | 87.7 | 7.6 | [-7.7, 23.2] | 0.200 |
| 1000PEPEUSDT | 79.0 | 87.4 | -8.3 | [-16.0, 4.9] | 0.854 |
| ENAUSDT | 87.4 | 88.6 | -1.2 | [-8.9, 12.0] | 0.372 |
| ARBUSDT | 63.3 | 86.8 | -23.5 | [-34.4, -9.8] | 1.000 |
| NEARUSDT | 81.8 | 89.1 | -7.3 | [-14.8, 3.6] | 0.878 |
| AVAXUSDT | 81.2 | 87.9 | -6.7 | [-17.6, 7.9] | 0.838 |
| DOGEUSDT | 115.8 | 88.8 | 27.0 | [15.0, 32.9] | 0.000 |
| INJUSDT | 58.9 | 86.0 | -27.1 | [-39.8, -14.7] | 1.000 |
| UNIUSDT | 82.6 | 88.9 | -6.3 | [-20.4, 8.2] | 0.808 |
| AAVEUSDT | 92.7 | 88.3 | 4.4 | [-6.8, 13.7] | 0.266 |
| total connectedness | 88.0% | VAR lag 1 | H 10 | [85.8, 89.3] | |

Pairwise, TEST (percent of the target's 10-step forecast-error variance explained by the source):

| from \ to (TEST, % of target's 10-step FEV) | BTC | ETH | SOL | ZEC | XRP | HYPE | 1000PEPE | ENA | ARB | NEAR | AVAX | DOGE | INJ | UNI | AAVE |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| BTCUSDT | - | 8.7 | 8.2 | 6.0 | 7.9 | 7.0 | 6.8 | 6.5 | 7.0 | 6.6 | 7.5 | 7.1 | 6.2 | 6.8 | 6.5 |
| ETHUSDT | 9.8 | - | 9.1 | 6.9 | 9.0 | 8.3 | 7.6 | 7.6 | 7.9 | 7.8 | 8.6 | 8.3 | 7.2 | 7.8 | 7.8 |
| SOLUSDT | 8.9 | 9.0 | - | 7.0 | 8.5 | 8.3 | 7.6 | 7.5 | 7.5 | 7.8 | 8.0 | 8.4 | 7.3 | 7.5 | 7.5 |

FIT (reference):

| symbol | to others | from others | net | net 95% CI | P*(net <= 0) |
|---|---|---|---|---|---|
| BTCUSDT | 88.5 | 77.7 | 10.8 | [2.5, 18.5] | 0.008 |
| ETHUSDT | 107.7 | 77.9 | 29.8 | [23.3, 40.2] | 0.000 |
| SOLUSDT | 102.9 | 76.3 | 26.7 | [16.4, 34.5] | 0.000 |
| ZECUSDT | 66.0 | 69.1 | -3.1 | [-10.7, 5.4] | 0.788 |
| XRPUSDT | 87.3 | 75.4 | 12.0 | [3.7, 18.7] | 0.002 |
| HYPEUSDT | 48.7 | 60.0 | -11.3 | [-19.0, -0.0] | 0.976 |
| 1000PEPEUSDT | 33.5 | 52.4 | -18.9 | [-25.1, -13.4] | 1.000 |
| ENAUSDT | 88.2 | 75.1 | 13.1 | [2.1, 23.6] | 0.004 |
| ARBUSDT | 36.7 | 57.9 | -21.2 | [-28.4, -13.4] | 1.000 |
| NEARUSDT | 66.6 | 70.5 | -3.9 | [-11.0, 4.2] | 0.840 |
| AVAXUSDT | 51.7 | 63.1 | -11.3 | [-17.5, -2.0] | 0.988 |
| DOGEUSDT | 91.0 | 76.8 | 14.2 | [6.9, 21.1] | 0.000 |
| INJUSDT | 28.1 | 46.7 | -18.5 | [-25.7, -12.1] | 1.000 |
| UNIUSDT | 72.2 | 72.2 | 0.1 | [-7.6, 8.1] | 0.542 |
| AAVEUSDT | 35.4 | 53.7 | -18.4 | [-26.5, -11.4] | 1.000 |
| total connectedness | 67.0% | VAR lag 1 | H 10 | [61.8, 72.4] | |

TEST without 13:00-16:00 (lags never cross the gap):

| symbol | to others | from others | net | net 95% CI | P*(net <= 0) |
|---|---|---|---|---|---|
| BTCUSDT | 93.4 | 87.3 | 6.1 | [-7.7, 21.5] | 0.154 |
| ETHUSDT | 110.2 | 86.9 | 23.3 | [12.3, 28.5] | 0.000 |
| SOLUSDT | 106.4 | 87.5 | 18.9 | [5.8, 29.8] | 0.000 |
| ZECUSDT | 86.7 | 82.4 | 4.3 | [-17.1, 22.9] | 0.468 |
| XRPUSDT | 80.2 | 85.7 | -5.5 | [-19.4, 4.4] | 0.880 |
| HYPEUSDT | 94.3 | 84.8 | 9.5 | [-6.5, 22.3] | 0.140 |
| 1000PEPEUSDT | 72.6 | 84.1 | -11.5 | [-20.7, 3.3] | 0.918 |
| ENAUSDT | 91.2 | 86.3 | 4.9 | [-5.2, 22.6] | 0.108 |
| ARBUSDT | 60.3 | 83.1 | -22.8 | [-34.1, -6.9] | 0.992 |
| NEARUSDT | 81.9 | 86.9 | -5.1 | [-14.7, 8.5] | 0.710 |
| AVAXUSDT | 77.8 | 86.2 | -8.4 | [-20.5, 7.9] | 0.848 |
| DOGEUSDT | 115.0 | 87.3 | 27.8 | [14.3, 36.9] | 0.000 |
| INJUSDT | 50.0 | 80.9 | -30.9 | [-42.8, -18.3] | 1.000 |
| UNIUSDT | 75.5 | 86.8 | -11.3 | [-27.2, 7.0] | 0.864 |
| AAVEUSDT | 87.0 | 86.2 | 0.7 | [-11.9, 13.3] | 0.530 |
| total connectedness | 85.5% | VAR lag 1 | H 10 | [81.8, 87.7] | |

With TradFi perps (TEST, descriptive; TradFi with >= 90% nonzero 1 m RV, top 8 by trades):

| symbol | to others | from others | net | net 95% CI | P*(net <= 0) |
|---|---|---|---|---|---|
| BTCUSDT | 90.0 | 76.8 | 13.2 | [-4.7, 30.0] | 0.085 |
| ETHUSDT | 101.3 | 77.5 | 23.8 | [8.0, 32.6] | 0.000 |
| SOLUSDT | 89.0 | 76.9 | 12.1 | [-6.4, 24.4] | 0.080 |
| NVDAUSDT | 54.0 | 68.4 | -14.4 | [-31.1, 1.3] | 0.955 |
| XAUUSDT | 32.1 | 61.9 | -29.8 | [-37.4, -11.7] | 1.000 |
| CLUSDT | 65.7 | 68.8 | -3.1 | [-16.4, 7.2] | 0.790 |
| MSTRUSDT | 109.1 | 74.6 | 34.5 | [19.5, 47.1] | 0.000 |
| CRWVUSDT | 11.9 | 27.3 | -15.5 | [-24.5, -6.6] | 0.995 |
| BZUSDT | 59.6 | 67.8 | -8.1 | [-21.4, 3.6] | 0.925 |
| XAGUSDT | 9.0 | 31.7 | -22.7 | [-31.8, -13.9] | 1.000 |
| QQQUSDT | 79.5 | 69.6 | 9.9 | [-1.4, 38.6] | 0.060 |
| total connectedness | 63.8% | VAR lag 1 | H 10 | [59.5, 69.1] | |

Reading: ETH and SOL are net vol transmitters in all three windows (FIT +29.8 / +26.7, TEST +25.0 / +21.6, ex-CPI +23.3 / +18.9), with CIs above 0. DOGE is equally strong on TEST. BTC's net is positive but smaller, and its CI includes 0 on TEST. Shares are nearly uniform (7-10% per source), which is a common-factor pattern rather than a hub. INJ and ARB are the clearest receivers. Connectedness rises from 67% to 88% in the CPI half.

## Descriptive statistics for the simulator

Spread and depth (100 ms states over the whole day; depth = bid + ask notional within the band):

| symbol | tick bp | spread bp p5 / p50 / p95 / p99 | mean | 1-tick share | depth5 USD p50 | depth10 USD p5 / p50 / p95 | depth25 USD p50 | book span bp p50 | 10 / 25 bp band truncated | top level USD p50 | updates/s |
|---|---|---|---|---|---|---|---|---|---|---|---|
| BTCUSDT | 0.013 | 0.013 / 0.013 / 0.014 / 0.014 | 0.014 | 99.8% | 15,717,163 | 13,151,428 / 15,735,591 / 18,852,185 | 15,735,735 | 4.1 | 100% / 100% | 289,026 | 10.0 |
| ETHUSDT | 0.048 | 0.047 / 0.048 / 0.049 / 0.050 | 0.049 | 99.8% | 5,840,787 | 9,567,720 / 15,794,286 / 20,011,496 | 17,716,952 | 10.5 | 5% / 100% | 104,001 | 10.0 |
| SOLUSDT | 1.183 | 1.157 / 1.181 / 1.217 / 1.222 | 1.184 | 100.0% | 1,172,792 | 2,151,833 / 3,260,220 / 4,091,643 | 10,091,572 | 235.6 | 0% / 0% | 49,815 | 9.5 |
| XRPUSDT | 0.929 | 0.908 / 0.927 / 0.957 / 0.961 | 0.930 | 100.0% | 608,013 | 1,340,134 / 2,041,887 / 2,426,548 | 7,939,697 | 184.9 | 0% / 0% | 23,862 | 9.8 |
| DOGEUSDT | 1.452 | 1.428 / 1.441 / 1.490 / 1.497 | 1.453 | 100.0% | 210,861 | 470,087 / 741,963 / 875,664 | 3,169,827 | 287.6 | 0% / 0% | 13,175 | 8.3 |
| HYPEUSDT | 0.136 | 0.133 / 0.136 / 0.141 / 0.545 | 0.148 | 96.8% | 215,853 | 481,640 / 687,207 / 855,153 | 2,276,355 | 28.7 | 0% / 0% | 1,196 | 9.5 |
| CARVUSDT | 3.646 | 7.223 / 14.793 / 25.257 / 29.102 | 15.517 | 1.6% | 0 | 0 / 273 / 1,068 | 4,180 | 5486.7 | 0% / 0% | 118 | 1.3 |
| KASUSDT | 3.346 | 3.318 / 6.671 / 10.076 / 10.171 | 6.057 | 34.4% | 899 | 3,578 / 7,799 / 13,943 | 46,587 | 710.3 | 0% / 0% | 494 | 3.1 |
| ATOMUSDT | 7.233 | 7.166 / 7.233 / 7.318 / 14.451 | 7.344 | 98.5% | 6,496 | 2,919 / 6,589 / 13,566 | 71,839 | 1442.0 | 0% / 0% | 3,295 | 3.3 |
| NVDAUSDT | 0.568 | 0.549 / 0.588 / 1.191 / 1.766 | 0.736 | 75.5% | 78,935 | 168,758 / 272,754 / 403,466 | 1,037,962 | 139.5 | 0% / 0% | 661 | 7.8 |
| QQQUSDT | 0.148 | 0.147 / 0.148 / 0.442 / 0.746 | 0.179 | 91.8% | 473,338 | 1,126,897 / 1,543,105 / 1,866,970 | 2,760,995 | 45.0 | 0% / 0% | 820 | 7.1 |
| XAUUSDT | 0.024 | 0.024 / 0.025 / 0.025 / 0.319 | 0.032 | 98.0% | 858,784 | 1,556,472 / 2,348,827 / 3,066,492 | 3,146,464 | 13.7 | 1% / 100% | 9,245 | 7.8 |

Dynamics and spread-volatility elasticity (log 1 m time-averaged spread on log same-minute RV of 1 s mid returns, 5-minute-block clustered t):

| symbol | spread ACF 1 s, lags 1 / 10 / 60 / 300 s | spread ACF 1 m, lags 1 / 5 / 15 / 60 min | elasticity d log spread / d log RV (t) | Spearman(spread, RV), 1 m | spread p50 FIT / TEST | depth10 p50 FIT / TEST |
|---|---|---|---|---|---|---|
| BTCUSDT | 0.08 / 0.01 / 0.00 / 0.00 | 0.07 / 0.01 / 0.01 / 0.01 | 0.021 (2.55) | 0.55 | 0.013 / 0.014 | 15,276,323 / 16,291,030 |
| ETHUSDT | 0.13 / 0.06 / 0.02 / 0.00 | 0.11 / 0.01 / 0.01 / 0.01 | 0.026 (2.27) | 0.58 | 0.047 / 0.049 | 17,412,403 / 14,109,259 |
| SOLUSDT | 0.55 / 0.59 / 0.53 / 0.52 | 0.87 / 0.85 / 0.83 / 0.75 | 0.008 (9.22) | 0.50 | 1.163 / 1.202 | 3,657,987 / 2,848,477 |
| XRPUSDT | 0.16 / 0.10 / 0.10 / 0.10 | 0.54 / 0.52 / 0.51 / 0.45 | 0.009 (4.17) | 0.33 | 0.913 / 0.943 | 2,151,926 / 1,888,430 |
| DOGEUSDT | 0.54 / 0.31 / 0.26 / 0.26 | 0.68 / 0.67 / 0.65 / 0.60 | 0.007 (7.39) | 0.53 | 1.433 / 1.475 | 795,722 / 656,350 |
| HYPEUSDT | 0.12 / 0.08 / 0.05 / 0.02 | 0.44 / 0.19 / 0.08 / 0.16 | 0.079 (6.28) | 0.63 | 0.134 / 0.138 | 645,825 / 717,612 |
| CARVUSDT | 0.96 / 0.75 / 0.29 / 0.09 | 0.48 / 0.15 / 0.05 / -0.01 | 0.037 (5.81) | 0.17 | 14.482 / 14.870 | 280 / 271 |
| KASUSDT | 0.87 / 0.47 / 0.16 / 0.13 | 0.46 / 0.27 / 0.17 / 0.20 | 0.072 (12.26) | 0.36 | 6.633 / 6.696 | 8,424 / 7,246 |
| ATOMUSDT | 0.62 / 0.15 / 0.03 / 0.02 | 0.14 / 0.09 / 0.08 / 0.06 | 0.010 (5.06) | 0.42 | 7.228 / 7.233 | 5,898 / 7,459 |
| NVDAUSDT | 0.41 / 0.25 / 0.16 / 0.11 | 0.63 / 0.48 / 0.32 / 0.40 | 0.097 (11.83) | 0.59 | 0.555 / 0.593 | 281,803 / 259,397 |
| QQQUSDT | 0.52 / 0.31 / 0.15 / 0.05 | 0.39 / 0.15 / 0.06 / 0.06 | 0.052 (3.86) | 0.38 | 0.147 / 0.148 | 1,503,600 / 1,608,686 |
| XAUUSDT | 0.52 / 0.29 / 0.14 / 0.06 | 0.34 / 0.12 / 0.03 / 0.00 | 0.060 (2.45) | 0.40 | 0.024 / 0.025 | 2,289,531 / 2,424,478 |

Simulator notes:

- **Tick-bound regime.** BTC, ETH, SOL, XRP, DOGE, ATOM, XAU and HYPE spend more than 96% of the time at one tick. Their spread should be simulated as a two-state process (1 tick, with rare short widenings; p99 at most 2 ticks, except HYPE (p99 4 ticks) and XAU (13 ticks)). The "spread ACF" of SOL, XRP and DOGE (0.5-0.9 at 1 m) only tracks the price level through tick / mid.
- **Free-spread regime.** CARV (median 14.8 bp, 4 ticks; 1-tick share 1.6%), KAS (6.7 bp, 34% at one tick) and NVDA (0.59 bp, 76%) have a spread that moves. Spread ACF decays within about a minute (CARV 0.96 at 1 s, 0.29 at 60 s). Elasticity to same-minute RV is 0.04-0.10, so the spread scales roughly as vol^0.1-0.2. That is weak but highly significant (t 6-12).
- **Depth.** Within 10 bp the median ranges from $15.7M (BTC, truncated at 4 bp) through $0.69M (HYPE) to $273 (CARV). Depth fell from FIT to TEST for ETH, SOL, XRP and DOGE (-12% to -22%) and was flat or up for the rest. Top-of-book notional is $289k (BTC), $104k (ETH) and $50k (SOL), down to about $100 (CARV).
- **Feed.** Pushes arrive at 10/s for the majors and 1.3-3/s for CARV, KAS and ATOM, so a 100 ms book is complete for the majors and a step function for the illiquid tail.

## Caveats

- One day, one market state (sell-off plus CPI). The FIT/TEST split is also a regime split, and the TEST half has the release.
- ob200 shows 200 levels. For BTC the visible book spans about 4 bp, so "depth within 10 / 25 bp" is the whole visible book, not the true depth. The same holds for ETH and XAU at 25 bp.
- Relative spread in tick-bound symbols is a price-level variable. H-MS5 as registered mixes 8 tick-bound and 4 free-spread symbols, so its pooled coefficient mostly measures the former.
- Trade-time prices are per-ms VWAPs because trades_raw keeps no within-ms order. Bid-ask bounce lowers the level of HY correlations (0.15-0.45) but not the location of the peak. Mid-quote HY gives the same peak location.
- The Granger tests reject in both directions for many pairs (SOL: 36 of 39 forward, 23 of 39 back). With 43,200 one-second returns, tiny stale-price effects are significant. Granger significance here is not a tradeable lead.
- The DY bootstrap concatenates 30-minute blocks with the VAR lags reset at block edges. Connectedness estimates from 720 minutes and a 15-variable VAR(1) are noisy, and the CIs are wide (±10 points).
- The hour-of-week factor is coarse within a single day. For CARV and KAS it is the pooled majors' profile.

## Rule candidates

```yaml
- id: R-MS-1
  kind: expect
  rule: WHEN a perp's spread sits at one tick most of the time (BTC, ETH, SOL, XRP, DOGE, HYPE, ATOM, XAU on Bybit) THEN do not read its relative spread as a volatility signal; forecast short-horizon vol from past RV and the hour-of-week factor
  when: 1-tick share of the 100 ms book > 95%
  then: vol forecast and quote width from RV (last 5 / 60 min) and the hour-of-week profile; spread ignored as an input
  scope:
    assets: tick-bound crypto majors and XAU perps
    horizon: 5-15 min
    sides: both
  effect: pooled TEST coef of log spread -0.17 (t -1.6) / -0.32 (t -3.0), OOS R^2 gain -0.0016 / -0.0014; past RV gives OOS R^2 0.70-0.75
  confidence: C
  evidence: [H-MS5]
  review: re-run on >= 10 days of ob200 books; a free-spread subset (CARV, KAS, NVDA) is the open lead
  status: provisional
- id: R-MS-2
  kind: avoid
  rule: WHEN ETH, SOL or BTC moves THEN do not expect a sub-minute follow-through in other crypto tokens to trade on; prices move together within about 20 ms, and 1 s Granger leads are stale last-trade prices
  when: any move in a crypto major
  then: no lead-lag entry or exit timed off a major's tick in another token; treat the move as a common shock
  scope:
    assets: Bybit crypto perps (40 tokens of the pilot)
    horizon: 0.1-60 s
    sides: both
  effect: HY peak at lag 0 for 38/39 (ETH) and 36/39 (SOL) targets, 0/39 positive peaks; fine-lag maximum within +-20 ms; mid-quote peaks 0-0.1 s
  confidence: C
  evidence: [H-MS6, H-OF5]
  review: re-run on more days of trades and books; a lag > 100 ms on >= 3 days would revive it
  status: provisional
- id: R-MS-3
  kind: expect
  rule: WHEN realised vol jumps in ETH, SOL or DOGE THEN expect the vol of the other crypto tokens to rise with it over the next minutes, and widen or pause quotes on the whole crypto book together rather than symbol by symbol
  when: 1 m RV of ETH, SOL or DOGE well above its recent level
  then: crypto-wide quote widening or pause; alts (INJ, ARB, ZEC) are the receivers
  scope:
    assets: Bybit crypto perps
    horizon: 1-10 min
    sides: both
  effect: DY net connectedness on TEST ETH +25.0 [13.1, 33.3], SOL +21.6 [10.9, 31.4], DOGE +27.0; total 88% (TEST) vs 67% (FIT)
  confidence: C
  evidence: [H-MS6]
  review: rolling DY on >= 10 days; check that the transmitter ranking persists outside news days
  status: provisional
- id: R-MS-4
  kind: prefer
  rule: WHEN modelling fills or markouts from ob200 books THEN use the trade records for the touch a trade met, not the last 100 ms book push
  when: simulator or replay built on the 100 ms book series
  then: trades define executed prices; the 100 ms book defines depth and quote context only
  scope:
    assets: all ob200 symbols
    horizon: sub-second
    sides: both
  effect: 11-64% of trades are outside the pre-trade 100 ms [bid1, ask1]; 1-19% outside the before-or-after union; 81-94% of those are sweeps
  confidence: C
  evidence: [H-MS5]
  review: compare with a tick-by-tick (ob50 / ob1) feed when one is recorded
  status: provisional
```
