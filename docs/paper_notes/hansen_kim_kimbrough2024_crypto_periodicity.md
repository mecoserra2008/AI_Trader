# Periodicity in Cryptocurrency Volatility and Liquidity (Hansen, Kim & Kimbrough, 2024)

- **Citation:** Peter Reinhard Hansen, Chan Kim, Wade Kimbrough (2024). *Journal of Financial Econometrics* 22(1), 224-251 (online 2022). DOI [10.1093/jjfinec/nbac034](https://doi.org/10.1093/jjfinec/nbac034). arXiv [2109.12142](https://arxiv.org/abs/2109.12142).
- **Verified:** arXiv API, Crossref, and the full text (arXiv v2 PDF, read in the scratchpad), 2026-11-10.
- **Topic:** 2. Crypto intraday seasonality and funding-time effects. Hypotheses: H-NS4 (hour-of-week vol profile), H-NS1 (funding settlement), H-VM6 / V17 (Saturday over-forecast).

## Question
Are there recurring day-of-week, hour-of-day and minute-of-hour patterns in crypto volatility and volume, and do they improve volatility models?

## Data and method
- BTC/USD and ETH/USD (Coinbase Pro), BTC/USDT and ETH/USDT (Binance), WETH/USDC (Uniswap V2). 1-minute OHLCV and trade data; second-by-second data for an extended sample from Jan 2019 to May 2021.
- Relative measures: |r| (and volume) in a given day / hour / minute divided by its trailing-window average, so the persistent level of vol is removed. Averaged over weeks, with 95% CIs, and shown as polar "Nightingale" plots.
- Forecasting: GARCH, EGARCH and EGARCH-X (with log RV from 5-minute returns), each with and without multiplicative day-of-week factors (log h_t = kappa_{d(t)} + log h*_t).

## Key results
- **Day of week:** vol and volume are lower on weekends, **most pronounced on Saturdays**; volume varies more than vol.
- **Hour of day:** vol is high around **16 UTC** and lowest near **05 UTC**. There are **spikes in volume and vol at 00h and 16h, and smaller ones at 08h UTC, coinciding with perp funding times** (Binance funding 00/08/16 UTC; BitMEX 04/12/20).
- **Minute of hour:** a large vol burst in the **first minute of each hour**, medium bursts after minutes 15, 30 and 45, and small ones every 5 minutes. These are algorithmic-trading footprints, strongest every fourth hour.
- The patterns have **grown stronger over time**.
- The periodic EGARCH-X tracks realised variance better out of sample than the non-periodic one (weekly pattern in h_t).

## Limitations / transfer to Bybit perps
- Pre-2022 spot data. Bybit perps share the 00/08/16 UTC funding schedule (some symbols now settle hourly / 4-hourly; check per symbol in the data). The patterns plausibly transfer and should be stronger in perps, where funding is paid.
- The paper studies vol and volume, not **returns**, around funding times. H-NS1 (directional pressure) is not supported or refuted by it.
- The minute-of-hour bursts matter for our 3s flow and for quoting: quotes resting through hh:00:00-hh:01:00 face more adverse moves.

## How we use it here
- B3 `features/seasonality.py` / H-NS4: fit a multiplicative hour-of-week factor (168 cells, smoothed, expanding window), plus dummies for funding hours and minute-of-hour for the 3s / 1m data. The paper's own result gives H-NS4 a high prior (0.8 seems right).
- `atx mm` quoting (via B4's replay): widen or pull quotes in the first minute of each hour and around funding timestamps, and test it with the H-SIM1 fill model. Draft hypothesis H-LIT-SIM5 (literature report).
- The Saturday dip matches V17 / H-VM6 (all models over-forecast Saturdays): this is independent confirmation.

## Evidence quality
**Strong** (J. Financial Econometrics, simple robust descriptive method, consistent across 5 markets and years).
