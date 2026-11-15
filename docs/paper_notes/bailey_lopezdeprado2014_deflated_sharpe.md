# The Deflated Sharpe Ratio: Correcting for Selection Bias, Backtest Overfitting and Non-Normality (Bailey & Lopez de Prado, 2014)

- **Citation:** David H. Bailey, Marcos Lopez de Prado (2014). *Journal of Portfolio Management* 40(5), 94-107. SSRN [2460551](https://doi.org/10.2139/ssrn.2460551). (JPM volume / pages from memory (m); Crossref returned the SSRN record and a JPM "Practical Applications" summary, DOI 10.3905/pa.2015.2.3.089.)
- **Verified:** Crossref, 2026-11-10. Formulas are the standard ones and match the repo's implementation (`research/trend_zoo_study.py::deflated_sharpe`).
- **Topic:** 4. Overfitting-adjusted significance of a selected strategy. Already in use (P7: best zoo variant DSR 0.72 < 0.95; R-AVOID-3 Cipher DSR 0.00).

## Question
Given that a strategy was selected as the best of N trials, with non-normal returns and a finite sample, what is the probability that its true Sharpe ratio is above zero?

## Method
- Probabilistic Sharpe ratio: PSR(SR*) = Phi( (SR-hat - SR*) sqrt(T - 1) / sqrt(1 - g3 SR-hat + (g4 - 1)/4 SR-hat^2) ), where g3 is the skewness and g4 the kurtosis of returns, with SR at the sampling frequency.
- Expected maximum Sharpe among N independent trials under the null: SR0 = sqrt(V[SR_n]) x ( (1 - gamma) Phi^-1(1 - 1/N) + gamma Phi^-1(1 - 1/(N e)) ), with gamma = 0.5772 (Euler-Mascheroni) and V[SR_n] the cross-trial variance of the Sharpe estimates.
- DSR = PSR(SR0). Accept at DSR > 0.95.
- N should be the **effective** number of independent trials (e.g. from clustering trials or from the eigenvalue spectrum), not the raw count.

## Key results
- Many published "significant" backtests fail once the number of trials and fat tails are accounted for. The haircut grows with N and with negative skew / excess kurtosis.

## Limitations / transfer to Bybit perps
- The effective N is the critical, hard-to-measure input. For the trend zoo we have ~1.9 independent bets (P7), which supports using an effective N from the H-PD1 eigenvalue entropy instead of 21.
- Assumes i.i.d. returns for the variance of SR; with autocorrelated strategy P&L, adjust T (Lo 2002 autocorrelation correction) or use block bootstrap.

## How we use it here
- Already implemented, including an effective-N variant: `trend_zoo_study.dsr_block` reports DSR at the raw N, at N_eff (the eigenvalue participation ratio) and at N including other frequencies. Proposed small refinement for B1 / B5: also report it at the eigenvalue-entropy N_eff (H-PD1's statistic), and apply the same three-N reporting to every grid, not only the trend zoo.
- Pair it with PBO ([bailey_borwein_lopezdeprado_zhu2017_pbo.md](bailey_borwein_lopezdeprado_zhu2017_pbo.md)): DSR judges the winner, PBO judges the selection process.

## Evidence quality
**Strong** (standard, peer-reviewed, already part of our protocol).
