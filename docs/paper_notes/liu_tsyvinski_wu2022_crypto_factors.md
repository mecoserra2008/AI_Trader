# Common Risk Factors in Cryptocurrency (Liu, Tsyvinski & Wu, 2022)

- **Citation:** Yukun Liu, Aleh Tsyvinski, Xi Wu (2022). *Journal of Finance* 77(2), 1133-1177. DOI [10.1111/jofi.13119](https://doi.org/10.1111/jofi.13119). NBER w25882.
- **Verified:** Crossref and the RePEc abstract, 2026-11-10. Numbers marked (m) come from memory.
- **Topic:** 1. Observable crypto factors for P&L attribution. Hypothesis: H-PD2.

## Question
Is there a small set of factors that prices the cross-section of cryptocurrency returns, analogous to the equity factor models?

## Data and method
- Weekly returns of a large coin universe (more than a thousand coins with a market-cap filter, roughly 2014-2018 (m)).
- They build crypto counterparts of the equity price- and market-based predictors (size, momentum, volume, volatility, etc.) and form long-short quintile strategies.
- Candidate factors: crypto market (CMKT), size (CSMB) and momentum (CMOM). Each characteristic strategy is regressed on them and the alphas are tested.

## Key results
- Three factors (crypto market, size, momentum) capture cross-sectional expected crypto returns (abstract).
- Ten crypto characteristics form significant long-short strategies, and all ten are explained by the three-factor model (abstract).
- Momentum works at short (1-4 week) horizons in crypto (m).

## Limitations / transfer to Bybit perps
- The sample predates perps dominating volume and ends in 2018. Small-cap coins drive the size effect, and they are not in our 23-symbol liquid perp universe.
- For a liquid perp universe, size is nearly constant and the useful factors are **market** and **momentum**, plus perp-specific ones the paper doesn't have: funding carry and basis (see [schmeling_schrimpf_todorov2026_crypto_carry.md](schmeling_schrimpf_todorov2026_crypto_carry.md)) and a vol-change factor.
- Short-selling costs in spot don't apply to perps, but funding does.

## How we use it here
- B1 / H-PD2 factor set: CMKT = equal-weight L-universe return; CMOM = long top-tercile / short bottom-tercile by trailing 3-week return, rebalanced weekly, all built point in time from bars L. Add a funding-carry factor (long low funding / short high funding) and a BTC-dominance factor (alt basket minus BTC).
- Expected outcome: most trend sleeves load on CMOM plus convexity (Fung-Hsieh), which tells the LLM that "five trend strategies agreeing" is one bet.

## Evidence quality
**Strong** (JF, widely cited). Transfer is **medium**: liquid perps and the post-2020 regime differ.
