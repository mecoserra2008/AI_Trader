# Hawkes processes in finance (Bacry, Mastromatteo & Muzy, 2015)

- **Citation:** Emmanuel Bacry, Iacopo Mastromatteo, Jean-Francois Muzy (2015). *Market Microstructure and Liquidity* 1(1), 1550005. DOI [10.1142/S2382626615500057](https://doi.org/10.1142/S2382626615500057). arXiv [1502.04592](https://arxiv.org/abs/1502.04592).
- **Verified:** arXiv API and Crossref, 2026-11-10. Review article; the statements about cited results (marked (m)) come from memory.
- **Topic:** 3. Self-exciting point processes for order flow, the generator family in H-SIM2. Also our jump work (R-EXEC-3 / H-JD2) and the negative result R-SIG-4 / H-VB4.

## Question
Survey: how are (multivariate, marked) Hawkes processes defined, estimated and used in high-frequency finance?

## Data and method (what to implement)
- Intensity lambda_i(t) = mu_i + sum_j integral phi_ij(t - s) dN_j(s). Kernels: exponential (fast MLE, Markovian), sum of exponentials, power law (long memory). Stationarity requires the spectral radius of the branching matrix ||phi|| to be < 1.
- Estimation:
  - **parametric MLE** (closed-form likelihood recursion for exponential kernels);
  - **non-parametric** kernel estimation via the Wiener-Hopf / conditional-expectation method (Bacry-Muzy);
  - EM.
- Goodness of fit: the time-rescaling theorem (compensator residuals should be unit-exponential; QQ plot / KS test).
- Applications covered: the signature plot / microstructure noise for volatility estimation, endogeneity ("reflexivity"), order-book event modelling, market impact, optimal execution, and contagion.

## Key results (as surveyed)
- Branching ratios estimated on futures are high, often 0.7-0.9 with exponential kernels (Filimonov-Sornette), and close to 1 with power-law kernels (Hardiman, Bercot & Bouchaud 2013, EPJ B; existence verified via Crossref). Much of the trading activity is endogenous. Estimates are very sensitive to kernel choice and to non-stationarity of the baseline mu(t) (intraday seasonality) (m).
- Bivariate Hawkes on up / down mid moves reproduces the signature plot (microstructure mean reversion). Buy / sell market-order Hawkes reproduce order-flow clustering (m).

## Limitations / transfer to Bybit perps
- Our data are 3s candles, not event times: a Hawkes model has to be fitted on **binned counts** (Hawkes-INAR / discretised likelihood) or on events reconstructed from trade counts.
- **Seasonality must be removed first** (hour-of-day, minute-of-hour; see [hansen_kim_kimbrough2024_crypto_periodicity.md](hansen_kim_kimbrough2024_crypto_periodicity.md)). Otherwise the branching ratio is inflated toward 1.
- Our own negative result (R-SIG-4: no exceedance clustering beyond what the HAR + Student-t forecast gives, once returns are scaled by robust sigma) shows Hawkes excitation partly proxies for slow vol. Fit Hawkes on **vol-standardised** event definitions.

## How we use it here
- B4 `sim/flow_sim.py` (H-SIM2): a marked bivariate Hawkes for buy / sell taker volume per 3s bin, with an exponential-kernel MLE on a deseasonalised baseline mu(t) = mu x s(hour, minute). Marks are Student-t / lognormal trade sizes. Validate with the time-rescaling QQ plot plus the H-SIM2 stylised facts.
- Report the branching ratio per symbol class as a descriptive result. Compare it with the subcritical ~0.1-0.2 reported for the Oct 2025 cascade ([garcia_seuma2026_liquidation_cascade_ews.md](garcia_seuma2026_liquidation_cascade_ews.md)).

## Evidence quality
**Strong** as a review of a mature literature. Hawkes models are a standard, well-validated tool.
