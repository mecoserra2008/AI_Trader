# Backtest overfitting in the machine learning era: A comparison of out-of-sample testing methods in a synthetic controlled environment (Arian, Norouzi Mobarekeh & Seco, 2024)

- **Citation:** Hamid Arian, Daniel Norouzi Mobarekeh, Luis Seco (2024). *Knowledge-Based Systems* 305, 112477. DOI [10.1016/j.knosys.2024.112477](https://doi.org/10.1016/j.knosys.2024.112477). SSRN 4686376 / 4778909.
- **Verified:** Crossref and a web summary of the abstract, 2026-11-10. The paper text was not read; details marked (m) are inferred from the abstract summary.
- **Topic:** 4. Which out-of-sample scheme limits overfitting best: walk-forward, K-fold, purged K-fold or combinatorial purged CV (CPCV). This bears on our house rule "purged walk-forward".

## Question
In a controlled synthetic environment where the true edge is known, which cross-validation / backtest scheme gives the most reliable estimate of out-of-sample performance and the fewest false discoveries?

## Data and method
- A synthetic market environment with financial-data features (non-stationarity, autocorrelation, regime shifts; e.g. stochastic vol with jumps and regime switching (m)), where strategies with known true performance are generated.
- Compared: walk-forward (WF), K-fold, purged K-fold, and **CPCV** (Lopez de Prado 2018: split into N groups, test on all C(N, k) combinations of k test groups with purging and embargo, which yields many backtest paths).
- Criteria: PBO (CSCV), the deflated Sharpe test statistic, and the variability / stationarity of the OOS estimates. Then an empirical check on S&P 500 data.

## Key results (per the abstract summary)
- **CPCV is markedly superior** at mitigating overfitting: lower PBO and a better DSR statistic.
- **Walk-forward has notable shortcomings in preventing false discoveries**: higher temporal variability and weaker stationarity of its estimates.
- The S&P 500 empirical check is consistent.

## Limitations / transfer to Bybit perps
- The synthetic environment is chosen by the authors; the conclusions could depend on it. Single paper, journal outside the core finance field.
- CPCV uses future data in training folds (purged). That is fine for **deterministic** strategy parameter selection inside `llm_in_sample`, but the LLM-layer principle is untouched (it is never backtested). The clean holdout remains the final judge.
- WF has one real advantage that CPCV lacks: it mimics deployment (parameters re-estimated only on the past). For **model refit cadence** questions (R-BUDGET-2), WF stays the right tool.

## How we use it here
- **Tension with the house rule** "purged walk-forward" (ARCHITECTURE §1): for **selecting among parameter grids** in-sample (trend zoo, breakout grids, MM parameters), CPCV gives more backtest paths and a less noisy selection. Proposal: keep purged WF for forecast-model evaluation and deployment simulation; add CPCV (with purge and embargo of at least the label horizon) for parameter selection, then report PBO. Changing the house rule is the user's call.
- B5: include "CPCV-selected" as a third selector in H-SYN1 / H-SYN2 (historical single path vs generated paths vs CPCV). A cheap and informative comparison.

## Evidence quality
**Medium-weak** (one simulation study, not read in full; its conclusions agree with Lopez de Prado's arguments, but the author overlap of ideas is high and independent replication is unknown).
