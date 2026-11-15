# Can Day Trading Really Be Profitable? (Zarattini & Aziz, 2023): 5-minute opening-range breakout on QQQ

- **Citation:** Carlo Zarattini, Andrew Aziz (2023). "Can Day Trading Really Be Profitable? Evidence of Sustainable Long-term Profits from Opening Range Breakout (ORB) Day Trading Strategy vs. Benchmark in the US Stock Market." SSRN working paper. DOI [10.2139/ssrn.4416622](https://doi.org/10.2139/ssrn.4416622)
- **Related (same authors):** Zarattini, Barbon & Aziz (2024), "A Profitable Day Trading Strategy For The U.S. Equity Market" (stocks in play), SSRN [4729284](https://doi.org/10.2139/ssrn.4729284); Zarattini, Aziz & Barbon (2024), "Beat the Market: An Effective Intraday Momentum Strategy for S&P500 ETF (SPY)", SSRN [4824172](https://doi.org/10.2139/ssrn.4824172). Crossref also lists a 2026 independent replication and audit of the "Beat the Market" paper (SSRN 7323419, Delgado). It was **not read**, so its findings are not reported here.
- **Verified:** Crossref metadata; results from the CXO Advisory summary and author pages, 2026-11-10. The paper itself was not read.
- **Topic:** 2. TradFi session effects. Hypothesis: H-TF3.

## Question
Does a simple, fully mechanical opening-range breakout beat buy-and-hold on QQQ after costs?

## Data and method
- QQQ (and TQQQ for leverage), 1-minute / 5-minute bars, January 2016 - February 2023.
- Rule: if the first 5-minute candle after the 09:30 open is up, buy at the open of the second candle; if it is down, sell short. No trade if open is about equal to close.
- Stop at the opposite extreme of the first candle; profit target at 10R; otherwise exit at the close.
- Sizing: risk 1% of equity per trade, capped by a leverage limit.

## Key results (per the CXO summary)
- 33.2% CAGR vs 14.9% for buy-and-hold QQQ, over 1,795 trades, 2016 - Feb 2023. **Slippage assumed zero** (commissions included per the authors).
- The stocks-in-play follow-up (top-20 relative-volume US stocks, 5-minute ORB, stop at 10% of ATR, exit at the close) reports a 41.6% annual return, **Sharpe 2.81** vs 0.78 for the S&P 500, 2016-2023.

## Limitations / transfer to Bybit perps
- Practitioner papers on SSRN, not peer-reviewed. Short sample (7 years), mostly a bull market, zero slippage. The 10R target gives a fat-right-tail payoff that depends on a few trend days.
- An independent replication on "five indices" posted on a trading blog (MQL5) claims "gross reproduced, net zero". Unreviewed; it is mentioned only as a reason for caution.
- On Bybit equity perps the cash open is not the perp's open: the perp traded all night, so the opening range carries less "price discovery" information. Taker costs of 7.5 bp per side on a 5-minute stop distance are a large fraction of R.
- With 1h bars H-TF3 uses a first-hour range, which differs from the 5-minute ORB. The 5-minute version needs flow_3s (from 2026-11-07, holdout only).

## How we use it here
- B2 H-TF3 as registered (1h ORB), with the random-entry baseline. Report the payoff skew and the share of P&L from the top 5% of days: if it is above 100%, the strategy is a lottery on trend days.
- If the user extends TradFi flow_3s collection, add a 5-minute ORB variant on QQQ / NVDA / TSLA perps.
- Context link: ORB is intraday trend following, so expect it to load on Fung-Hsieh convexity (B1 attribution).

## Evidence quality
**Weak-medium**: practitioner working papers, zero slippage, short sample, popular and therefore probably crowded.
