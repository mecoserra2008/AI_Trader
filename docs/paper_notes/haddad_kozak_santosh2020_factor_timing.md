# Factor Timing (Haddad, Kozak & Santosh, 2020)

- **Citation:** Valentin Haddad, Serhiy Kozak, Shrihari Santosh (2020). *Review of Financial Studies* 33(5), 1980-2018. DOI [10.1093/rfs/hhaa017](https://doi.org/10.1093/rfs/hhaa017). NBER w26708.
- **Verified:** Crossref and the RePEc abstract, 2026-11-10. (RePEc also lists the editor Stijn Van Nieuwerburgh in the author field; Crossref and the NBER version list three authors.) Numbers marked (m) come from memory.
- **Topic:** 1. Dimensionality reduction plus conditional ranking: time the dominant PCs of a strategy set with a few predictors.

## Question
Can we time a large set of factor (anomaly) portfolios without overfitting, and how much does timing add over static factor investing?

## Data and method
- 50 well-known US equity anomaly long-short portfolios, monthly (m).
- Reduce to the top ~5 principal components of anomaly returns, a dimensionality restriction that guards against overfitting.
- Predict each PC's next-month return with that PC's own valuation ratio (its book-to-market spread), using univariate predictive regressions with shrinkage. Combine the predicted means and covariances into a mean-variance factor-timing portfolio, which they show equals the SDF.
- Out-of-sample: split the sample, estimate the PCs and predictive coefficients in the first half, and evaluate in the second (m).

## Key results
- Market-neutral equity factors are strongly and robustly predictable (abstract).
- Timing gives a substantial improvement over static factor investing (abstract). Out-of-sample monthly R-squared for the dominant PCs is a few percent (m).
- Key design lesson: **predict a few dominant components, not each of the many strategies separately**. Restricting to the high-variance PCs is what makes the timing robust.

## Limitations / transfer to Bybit perps
- Their predictor (book-to-market spread of the factor) has no crypto analogue. Plausible substitutes per strategy-panel PC: funding level / crowding for carry-like PCs, realised-vs-implied vol spread for vol-like PCs, trailing dispersion for momentum-like PCs.
- With ~5 years of daily data the effective sample is small; use monthly or weekly horizons and few predictors.

## How we use it here
- B1: after H-PD1 (PCA of the strategy panel), time PC1-PC3 with 1-2 state variables each: weekly predictive regressions, expanding window, NW SEs. Compare with IPCA (H-PD3) and with strategy momentum.
- It is the simplest credible "rank strategy types by context" model, a natural rung between "trailing Sharpe" and IPCA in a complexity ladder.

## Evidence quality
**Strong** (RFS). Transfer is **medium-weak**, because the predictor has no direct crypto analogue.
