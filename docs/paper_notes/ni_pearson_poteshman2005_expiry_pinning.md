# Stock price clustering on option expiration dates (Ni, Pearson & Poteshman, 2005)

- **Citation:** Sophie Xiaoyan Ni, Neil D. Pearson, Allen M. Poteshman (2005). *Journal of Financial Economics* 78(1), 49-87. DOI [10.1016/j.jfineco.2004.08.005](https://doi.org/10.1016/j.jfineco.2004.08.005). SSRN 519044.
- **Verified:** Crossref and RePEc metadata (no abstract on RePEc). The summary below combines a web search snippet (HKBU repository / econpapers) with memory, marked (m).
- **Topic:** 2. Expiry / structural-flow effects. Hypothesis: H-NS2 (Deribit expiry compresses then releases vol).

## Question
Do stock prices "pin" to option strike prices at expiration, and who causes it?

## Data and method
- US optionable vs non-optionable stocks, expiration Fridays vs other days (1996-2002 era, CBOE / ISE option data (m)).
- Statistic: the frequency with which closing prices fall near a strike on expiration Fridays vs non-expiration days (and vs non-optionable stocks).
- Mechanism tests: changes in net option positions of market makers (delta-hedge rebalancing when long gamma) vs firm proprietary traders (manipulation when short options).

## Key results
- On expiration dates, closing prices of stocks with listed options **cluster at strike prices** (verified summary).
- Both hedge rebalancing by option market makers and stock-price manipulation by firm proprietary traders contribute (verified summary).
- Economic size: average shifts of ~16.5 bp in optionable stock prices, i.e. market-cap shifts of the order of $9 billion per expiration (m; the $9bn figure also appears in the search summary).

## Limitations / transfer to Bybit perps
- Pinning needs **dealers long gamma** near a heavily open-interest strike and a liquid hedging market. For BTC / ETH, Deribit dealers' net gamma sign is uncertain: crypto customers both buy and sell options (covered calls, structured products sell vol).
- Deribit expiry is 08:00 UTC Friday (a TWAP of the index over the last 30 minutes before 08:00). H-NS2 tests the vol implication (compression before, release after), not pinning itself. A direct pinning test is cheap: the distance of the 08:00 index price to the max-OI strike vs placebo days.
- Bybit perps track an index across spot venues; the pinning pressure acts on spot / perp hedges, so it should show in the perp too if it exists.

## How we use it here
- B3 / H-NS2: add the pinning test as a companion diagnostic. |log(P_0800 / K_maxOI)| on expiry Fridays vs the same statistic on non-expiry Fridays (K taken from the same OI snapshot), with a permutation p-value. Needs a point-in-time Deribit OI-by-strike snapshot. If we don't collect it, use the nearest round-number strike (1000 USD BTC, 50 USD ETH) as a weaker proxy.
- If pinning holds, it is a **quoting** input: tighter quotes / mean-reversion around the pin strike in the 08:00 window. Otherwise expiry is just a vol-calendar event.

## Evidence quality
**Strong** for 1990s-2000s US equities (JFE). **Weak** transfer to crypto (unknown dealer gamma sign).
