# Characteristics are covariances: A unified model of risk and return (Kelly, Pruitt & Su, 2019)

- **Citation:** Bryan T. Kelly, Seth Pruitt, Yinan Su (2019). *Journal of Financial Economics* 134(3), 501-524. DOI [10.1016/j.jfineco.2019.05.001](https://doi.org/10.1016/j.jfineco.2019.05.001). NBER w24540.
- **Verified:** Crossref and the RePEc abstract, 2026-11-10. Numbers marked (m) come from memory of the paper.
- **Topic:** 1. Conditional latent factors (IPCA). Hypothesis: H-PD3.

## Question
Are characteristic-sorted return premia compensation for exposure to latent risk factors (with loadings that vary with the characteristics), or "anomaly" alpha?

## Data and method
- Model: r_{i,t+1} = alpha_{i,t} + beta_{i,t}' f_{t+1} + e_{i,t+1}, with beta_{i,t} = z_{i,t}' Gamma_beta and alpha_{i,t} = z_{i,t}' Gamma_alpha.
  - z_{i,t} is an L-vector of lagged characteristics plus a constant;
  - Gamma_beta is L x K;
  - f holds K latent factors.
- Estimation by alternating least squares (ALS):
  - given Gamma, f_{t+1} = (Gamma' Z_t' Z_t Gamma)^-1 Gamma' Z_t' r_{t+1}, a cross-sectional regression each period;
  - given f, Gamma comes from a pooled regression of r on kron(z, f);
  - iterate to convergence.
- Characteristics are rank-normalised to [-0.5, 0.5] across the cross-section each period.
- Tests:
  - "characteristics are risk": a bootstrap Wald-type test of Gamma_alpha = 0;
  - each characteristic's contribution: a test that its row of Gamma_beta is 0.
- Data: US stocks, 36 characteristics, 1962-2014 (m). The paper reports both total R-squared (fit of realised returns) and predictive R-squared (fit of expected returns via beta_{i,t}' lambda).
- Code: Kelly's website and a community `ipca` Python package (not audited here).

## Key results
- Five IPCA factors explain the cross-section of average returns better than existing factor models, with small and insignificant anomaly intercepts (abstract).
- Only 10 of the characteristics are significant at 1%, and they account for nearly all of the model's accuracy (abstract).
- Predictive R-squared is small in absolute terms: under 1% per month for individual stocks (m), the norm for return prediction.

## Limitations / transfer to Bybit perps
- IPCA is built for a large cross-section (thousands of stocks). Ours is ~15-30 strategies x 23 symbols. With N that small, K must be 1-3 and L (context characteristics) about 5-8, or the L x K Gamma matrix overfits.
- In H-PD3 the "assets" are **strategies** and the "characteristics" are **context**: regime cell, vol percentile, funding z, trend strength, drawdown flag, weekend. All strategies on the same symbol-day share that context.
  - To identify Gamma, the instruments must also vary across strategies. Interact context with strategy-family dummies or strategy traits (holding period, direction bias, stop width).
  - Otherwise the model collapses to a time-series factor model.
- Weekly refits over 4-5 holdout months give few independent weeks, so the IC difference in H-PD3 will be noisy.

## How we use it here
- In B1's `pnl_decomposition_study.py` (H-PD3), implement IPCA by ALS in numpy (~60 lines) rather than adding a dependency:
  - z = [1, context x family interactions], with K in {1, 2, 3};
  - compare its next-week rank IC with trailing-90-day Sharpe. That baseline is "factor momentum" (see [ehsani_linnainmaa2022_factor_momentum.md](ehsani_linnainmaa2022_factor_momentum.md)).
- The fitted Gamma_beta directly gives the "strategy family x context -> expected loading" table the orchestrator wants for the context-ranking table.

## Evidence quality
**Strong** for the method and the US-equity result (JFE, highly cited, replicated). The prior for our use is **weak** because the cross-section is tiny.
