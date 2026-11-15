# The Probability of Backtest Overfitting (Bailey, Borwein, Lopez de Prado & Zhu, 2017)

- **Citation:** David H. Bailey, Jonathan M. Borwein, Marcos Lopez de Prado, Qiji Jim Zhu (2017). "The probability of backtest overfitting." *Journal of Computational Finance* 20(4), 39-69 (online 2016). DOI [10.21314/JCF.2016.322](https://doi.org/10.21314/JCF.2016.322). SSRN [2326253](https://doi.org/10.2139/ssrn.2326253).
- **Verified:** Crossref (JCF DOI and SSRN records), 2026-11-10. The volume / issue / pages (20(4), 39-69) come from memory (m). The method below is the paper's CSCV algorithm.
- **Topic:** 4. Backtest overfitting, measured by CSCV / PBO. Hypotheses: H-SYN1, H-SYN2; applies to every strategy-selection result (trend zoo, breakout grids, MM parameter grids in H-SIM4).

## Question
When we pick the best of N strategy configurations on a backtest, what is the probability that the chosen one is below the median out of sample?

## Data and method: CSCV (combinatorially symmetric cross-validation)
1. Build the T x N matrix M of per-period returns (or P&L) for the N configurations tried. **Include every configuration tried, not just the survivors.**
2. Split the rows into S contiguous, equal blocks (S even, e.g. 16).
3. For each of the C(S, S/2) combinations (12,870 for S = 16), take the chosen S/2 blocks as in-sample J and the complement as out-of-sample J-bar.
4. Compute a performance statistic (e.g. Sharpe) for every configuration on J and on J-bar. Let n* = argmax in-sample. Let omega-bar = the relative rank of n* out of sample, in (0, 1).
5. Logit lambda = ln( omega-bar / (1 - omega-bar) ).
6. **PBO = fraction of combinations with lambda <= 0**, i.e. the in-sample winner is at or below the OOS median.
- Companion diagnostics: the regression of OOS performance on IS performance across combinations (a negative slope means overfitting), the probability of OOS loss, and stochastic dominance of the selected strategy's OOS distribution over the median.
- Model-free and non-parametric, but it assumes the blocks are exchangeable; it ignores serial dependence across block boundaries.

## Key results
- Shows analytically and by simulation that with enough configurations, a high in-sample Sharpe is easy to get by chance and PBO approaches 1 for pure-noise strategies. With a real signal, PBO is low.
- Gives a single, interpretable number to report next to any "best of N" claim.

## Limitations / transfer to Bybit perps
- With our short holdout, CSCV is applied **within `llm_in_sample`** (2019 - mid 2026). The clean holdout stays separate and untouched.
- Blocks must be long relative to strategy holding periods and vol-regime persistence (regime episodes last 5-7 days, L1). Use S = 10-16 blocks of ~5 months on ~6 years.
- PBO measures the **selection process**, not one strategy. Report it for each grid (trend zoo 21 variants, Cipher 4 variants, MM parameter grid).

## How we use it here
- New utility `research/evaluate.py::pbo_cscv(M, S=16, stat=sharpe)` (deterministic, testable; ~40 lines). Not present today: the repo has `trend_zoo_study.deflated_sharpe` but no PBO.
- Report PBO for: the trend zoo (P7), H-SYN1 / H-SYN2 (with real-path vs generated-path selection: does generated-data selection lower PBO?) and H-SIM4's parameter grid.
- Draft hypothesis H-LIT-SYN3 (literature report): PBO of the trend-zoo selection is > 0.5.

## Evidence quality
**Strong** as a method (widely adopted, peer-reviewed).
