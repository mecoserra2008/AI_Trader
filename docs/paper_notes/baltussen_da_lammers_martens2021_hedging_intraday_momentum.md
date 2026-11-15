# Hedging demand and market intraday momentum (Baltussen, Da, Lammers & Martens, 2021)

- **Citation:** Guido Baltussen, Zhi Da, Sten Lammers, Martin Martens (2021). *Journal of Financial Economics* 142(1), 377-403. DOI [10.1016/j.jfineco.2021.04.029](https://doi.org/10.1016/j.jfineco.2021.04.029)
- **Verified:** Crossref and the full RePEc abstract, 2026-11-10.
- **Topic:** 2. Structural flows (gamma hedging) and session effects. Hypotheses: H-TF4, H-NS2 (expiry / gamma), and possible crypto analogues via Deribit dealer gamma.

## Question
Is market intraday momentum caused by hedging flows from agents who are short gamma (options market makers, leveraged ETFs)?

## Data and method
- Intraday returns of **over 60 futures** on equities, bonds, commodities and currencies, **1974-2020** (abstract).
- Predictor: the return from the previous close to 30 minutes before the close (the "rest of the day"). Target: the last 30 minutes.
- Proxies for net dealer gamma: estimated from options open interest (e.g. S&P 500 index options), plus leveraged-ETF rebalancing demand. Intraday momentum is interacted with the sign and size of net gamma.

## Key results
- Strong market intraday momentum **everywhere**, across asset classes (abstract).
- The last-30-minute return is positively predicted by the rest-of-day return, and this **reverts over the next days** (abstract), consistent with temporary price pressure, not information.
- The effect is linked to gamma-hedging demand: stronger when dealers are net short gamma (abstract).

## Limitations / transfer to Bybit perps
- Crypto options are concentrated on Deribit (BTC, ETH). Dealer gamma positioning can be estimated from Deribit open interest by strike, with the usual "customers buy options" sign assumption, but that assumption is weaker in crypto.
- 24/7 trading has no "close" for crypto. The closest analogues are the Friday 08:00 UTC Deribit expiry (gamma unwinds) and the daily 08:00 UTC Deribit settlement.
- For TradFi perps the underlying's close is the hedging event, so a 1h test of "rest of day -> last hour" on equity perps is a direct transplant (a variant of H-TF4).

## How we use it here
- B2: run H-TF4's variant with the paper's predictor (previous close -> 1 hour before the close) and add the **next-day reversal** check. The reversal is the paper's signature of non-informational pressure.
- B3 / H-NS2: the gamma-pinning-then-release story for Deribit expiries. The sign of the expected effect on realised vol depends on dealer gamma (long gamma pins and dampens; short gamma amplifies). If a Deribit OI-by-strike snapshot is available point in time, H-NS2 should be conditioned on estimated dealer gamma. That data source is the user's decision (see the user-guides-data memory).

## Evidence quality
**Strong** (JFE, 60+ markets over 46 years, mechanism tested). **Medium** transfer to equity perps; **weak** to crypto until dealer-gamma data exist.
