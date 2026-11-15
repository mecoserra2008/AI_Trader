# The Risk in Hedge Fund Strategies: Theory and Evidence from Trend Followers (Fung & Hsieh, 2001)

- **Citation:** William Fung, David A. Hsieh (2001). *Review of Financial Studies* 14(2), 313-341. DOI [10.1093/rfs/14.2.313](https://doi.org/10.1093/rfs/14.2.313)
- **Verified:** Crossref metadata, 2026-11-10 (~1,250 citations). Numbers marked (m) come from memory of the paper and were not re-checked.
- **Topic:** 1. P&L decomposition with option-like (non-linear) factors. Hypotheses: H-PD2, H-PD1.

## Question
Trend followers' returns look uncorrelated with asset indices. Is that alpha, or is their risk a non-linear factor that linear style analysis misses?

## Data and method
- Theory: over a period, a perfect trend follower earns the payoff of a **lookback straddle** (buy at the period minimum, sell at the maximum: payoff = max - min). Rolling standard straddles replicate it.
- The authors build "primitive trend-following strategies" (PTFS): returns of rolling lookback straddles on five markets (bonds, currencies, commodities, short-term interest rates, stock indices), constructed from exchange-traded options.
- They regress trend-following (CTA) fund returns (TASS database, 1990s (m)) on the PTFS returns and compare with regressions on standard asset indices.
- Fung & Hsieh (2004, FAJ) later extended this into the 7-factor hedge-fund benchmark (not covered here).

## Key results
- Standard asset-class regressions explain little of trend followers' returns. The PTFS straddle factors explain far more: roughly half the variance of a trend-follower index (m).
- Trend followers have **long-volatility, convex** payoffs. They earn in the underlying markets' extreme months, both up and down, which is why they look like "crisis alpha".

## Limitations / transfer to Bybit perps
- The straddle factors need listed option prices. On crypto we can build the lookback-straddle return synthetically from price paths (max - min over the holding period, minus a premium proxy from DVOL or realised vol). Alternatively, use the squared or absolute market return as a convexity factor (Treynor-Mazuy / Henriksson-Merton).
- Our trend zoo is ~1.9 independent bets and lost money in the holdout (BELIEFS P7). This paper says its natural factor is market **convexity**, so its expected sign depends on the regime: it needs large moves.

## How we use it here
- In B1's `pnl_decomposition_study.py`, add two non-linear factors to H-PD2:
  - (a) the synthetic lookback-straddle return of the crypto market factor over the strategy's typical holding period;
  - (b) |market return| or the squared market return.
- A trend strategy that loads heavily on (a) with zero selection is "a straddle", not alpha.
- Context rule candidate: trend sleeves pay in expanding-range states. Combine this with V9 (high-vol regimes over-forecast vol) to judge when the straddle is cheap.

## Evidence quality
**Strong**: seminal and widely replicated. The convexity of trend following is a robust stylised fact.
