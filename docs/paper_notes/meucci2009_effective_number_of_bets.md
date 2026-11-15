# Managing Diversification (Meucci, 2009)

- **Citation:** Attilio Meucci (2009). "Managing Diversification." *Risk* (May 2009), 74-79. A working-paper version is on SSRN; its exact id was not verified.
- **Verified:** existence, venue and the effective-number-of-bets (ENB) definition, via web sources on 2026-11-10 (portfoliooptimizer.io's ENB write-up and later papers that cite Meucci 2009). The Risk magazine text itself was not read.
- **Topic:** 1. Statistical factors / effective number of bets. Hypothesis: H-PD1.

## Question
How many truly independent bets does a portfolio of correlated positions contain?

## Data and method
1. Take the covariance matrix Sigma of strategy returns and eigen-decompose it: Sigma = E Lambda E'.
2. Express the portfolio weights w in principal-portfolio coordinates: w~ = E' w.
3. The risk contribution of principal portfolio n is p_n = (w~_n^2 lambda_n) / sum_k (w~_k^2 lambda_k). These sum to 1.
4. ENB = exp( - sum_n p_n ln p_n ), the exponential of the Shannon entropy. ENB = 1 when all risk loads on one factor and ENB = N when risk is spread evenly over N uncorrelated factors.
- Variant (not reviewed here): Meucci, Santangelo & Deguest's "minimum-torsion" bets (Crossref shows an SSRN 2013 paper on optimised uncorrelated factors) use the uncorrelated factors closest to the original assets instead of PCs, which avoids the arbitrary signs and rotation of PCs.

## Key results
- A portfolio of many assets can have an ENB near 1-3 when they share a dominant factor.
- ENB is a single, interpretable diversification number and a target for risk budgeting (maximise ENB).

## Limitations / transfer to Bybit perps
- PC-based ENB depends on the estimated covariance. With short histories (TradFi perps, ~5 months), use shrinkage (Ledoit-Wolf, already in `rel.factors`) and bootstrap the ENB.
- H-PD1 as registered uses exp(entropy of the normalised eigenvalues) of the correlation matrix. That is the **equal-weight** special case: it ignores portfolio weights. Report both:
  - eigenvalue entropy, the structure of the strategy universe;
  - Meucci ENB at the actual book weights, the diversification of what we hold.
- Crypto correlations jump toward 1 in liquidation cascades (see [garcia_seuma2026_liquidation_cascade_ews.md](garcia_seuma2026_liquidation_cascade_ews.md)), so ENB depends on the state. Compute it per regime.

## How we use it here
- B1: in `pnl_decomposition_study.py`, add `effective_bets(weights, cov)` next to the eigenvalue-entropy version for H-PD1, and report ENB per v0 regime cell.
- Risk engine (Rust, later): a book-level ENB floor is a portfolio-constraint candidate (e.g. warn when ENB < 2). This is consistent with R-MM-2 (many crypto fills are one bet).

## Evidence quality
**Medium-strong** as a method: widely used in practice, though not an empirical claim. Unverified: the SSRN id and the exact page range.
