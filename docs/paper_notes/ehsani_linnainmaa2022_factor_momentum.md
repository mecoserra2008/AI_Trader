# Factor Momentum and the Momentum Factor (Ehsani & Linnainmaa, 2022)

- **Citation:** Sina Ehsani, Juhani T. Linnainmaa (2022). *Journal of Finance* 77(3), 1877-1919. DOI [10.1111/jofi.13131](https://doi.org/10.1111/jofi.13131). NBER w25551.
- **Verified:** Crossref and the RePEc abstract, 2026-11-10. Numbers marked (m) come from memory.
- **Topic:** 1. Ranking strategies by context: is the trailing performance of a strategy predictive? This is the baseline in H-PD3 and relevant to H-RC1 / H-RC2.

## Question
Are factor (strategy) returns autocorrelated, so that a factor's own past performance predicts its future performance?

## Data and method
- About 20 US and international equity factors (m), monthly.
- Time-series factor momentum: hold each factor long if its trailing 12-month return is positive, short (or out) if negative. Compare with cross-sectional factor momentum and with individual-stock momentum.
- They test whether factor momentum subsumes stock momentum and whether it concentrates in high-eigenvalue PC factors.

## Key results
- Most factors are positively autocorrelated. The average factor earns **6 bp a month after a losing year and 51 bp after a winning year** (abstract).
- Factor momentum concentrates in the factors that explain more of the cross-section, i.e. high-eigenvalue PCs. Momentum in those PC factors subsumes most forms of individual-stock momentum (abstract).
- Interpretation: momentum is not a separate risk factor; it **times** other factors.

## Limitations / transfer to Bybit perps
- Monthly horizons over decades. Our strategy panel has ~5 years in-sample and ~4 months of holdout, so the number of independent "years" is tiny.
- Factor-return autocorrelation can come from slow-moving risk premia that also exist in crypto (leverage cycles, as in carry decaying from 57%/yr in 2021 to negative in 2026, BELIEFS C1). That is a reason to expect some strategy momentum.
- Trailing-Sharpe ranking is exactly what H-RC2's "unconditional selection" and H-PD3's baseline do. This paper says that baseline is **hard to beat**, so H-PD3 (IPCA beats trailing Sharpe) has a low prior.

## How we use it here
- B1: implement strategy momentum as the explicit baseline. Sign of the trailing 90 / 180 / 365-day net P&L of each strategy sleeve; evaluate next-month P&L conditional on the sign (the 6 bp vs 51 bp test, transplanted).
- Apply it to the PCs of the strategy panel too (H-PD1 output): momentum in PC1 vs in the residual sleeves.
- Draft hypothesis H-LIT-PD5 (see the literature report): strategy-level time-series momentum.

## Evidence quality
**Strong** for equity factors (JF). **Medium** as a prior for our strategy sleeves.
