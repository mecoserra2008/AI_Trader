# Empirical Asset Pricing via Machine Learning (Gu, Kelly & Xiu, 2020)

- **Citation:** Shihao Gu, Bryan Kelly, Dacheng Xiu (2020). *Review of Financial Studies* 33(5), 2223-2273. DOI [10.1093/rfs/hhaa009](https://doi.org/10.1093/rfs/hhaa009). NBER w25398.
- **Verified:** Crossref and the RePEc abstract, 2026-11-10. Numbers marked (m) come from memory.
- **Topic:** 1. Ranking strategies / predictors in a high-dimensional setting; complexity budget.

## Question
Which ML methods best measure expected returns, and where do their gains come from?

## Data and method
- ~30,000 US stocks, 1957-2016 (m). Predictors: 94 stock characteristics, 8 macro predictors and their interactions, plus industry dummies (~900 predictors (m)).
- Methods: OLS (all predictors / 3 predictors), elastic net, PCR, PLS, generalised linear models with splines, random forests, gradient-boosted trees, and neural nets with 1-5 hidden layers (NN1-NN5).
- Validation:
  - the sample is split in time: an expanding training window, a rolling 12-year validation window, annual refits;
  - hyperparameters are tuned on the validation window only;
  - out-of-sample R-squared is computed monthly against a **zero** forecast, not the historical mean.
- Variable importance: the drop in R-squared when one predictor is removed (drop-one).

## Key results
- Trees and neural nets are best, and their gains come from **non-linear interactions** among predictors (abstract).
- All methods agree on the same dominant signals: variants of momentum, liquidity and volatility (abstract).
- Monthly out-of-sample R-squared is small, well under 1% for individual stocks (m). Economic gains are still large because they compound in a long-short portfolio across thousands of names. Shallow nets beat deep ones (m).

## Limitations / transfer to Bybit perps
- The gains come from the size of the cross-section, and we have 23 crypto + ~15 TradFi perps. Our own result (D1 / R-BUDGET-1: more inputs make next-day direction worse) is exactly the small-N regime where this paper's gains don't appear.
- The paper is useful mainly as a **protocol**: time-ordered train / validation / test, R-squared against zero, drop-one importance. That matches our house rules (purged walk-forward, linear-first complexity budget).

## How we use it here
- As a protocol reference, not as a model, for any context-ranking model in B1 (H-PD3) and for R-BUDGET: report out-of-sample R-squared against a zero forecast and drop-one importance, and try linear / ridge first.
- It also predicts that if anything works, the inputs will be momentum, liquidity and volatility state variables, which are already on the market-state card.

## Evidence quality
**Strong** for US equities. Transfer to our small universe is **weak**; treat it as methodology.
