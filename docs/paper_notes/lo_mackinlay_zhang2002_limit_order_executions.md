# Econometric models of limit-order executions (Lo, MacKinlay & Zhang, 2002)

- **Citation:** Andrew W. Lo, A. Craig MacKinlay, June Zhang (2002). *Journal of Financial Economics* 65(1), 31-71. DOI [10.1016/S0304-405X(02)00134-4](https://doi.org/10.1016/S0304-405X(02)00134-4). NBER w6257 (1997).
- **Verified:** Crossref and Semantic Scholar (~255 citations), 2026-11-10. No abstract was retrievable (RePEc shows a mismatched abstract), so the method and results come from memory (m) plus the Vyetrenko et al. (2020) catalogue, which cites "generalized gamma distribution with accelerated failure time" for time-to-fill, as read in that paper.
- **Topic:** 3. Fill-probability / time-to-fill models for limit orders. Hypothesis: H-SIM1.

## Question
How long do limit orders take to execute, what determines it, and can hypothetical "first-passage" fills (price touches the limit) stand in for real executions?

## Data and method (m)
- Actual limit orders (submission, cancellation and execution times) from an institutional trading dataset (ITG) for US stocks in the mid-1990s.
- **Survival analysis**: time-to-first-fill and time-to-completion modelled with a **generalized gamma accelerated-failure-time (AFT)** model. log T = x' beta + sigma W, with W generalized-gamma distributed. Covariates: limit price distance from the mid (in ticks / vol units), order size, side, spread, recent volatility, market depth, recent trading activity. **Cancellations are treated as right-censored** observations.
- Comparison with first-passage-time models: execution = the first time the price process (geometric Brownian motion) crosses the limit.

## Key results (m)
- Time-to-execution is very sensitive to the limit price (distance from the mid) and to market conditions, and much less sensitive to order size.
- **Hypothetical first-passage executions are poor proxies for actual executions:** a trade-through / touch is not a fill, because of queue priority and adverse selection. Models that assume a fill at touch overstate fill rates and speed.

## Limitations / transfer to Bybit perps
- 1990s US equities with decimal-era tick sizes not yet in place. Crypto perp queues are faster and cancellations more frequent.
- Exactly our situation, though: we have ~250 live fills **plus all unfilled / cancelled quotes** from `atx mm`. The survival framing (cancel = censored) uses the unfilled quotes as information instead of discarding them.

## How we use it here
- B4 `sim/fill_model.py` (H-SIM1): fit a parametric AFT survival model (Weibull or generalized gamma; `lifelines` or a hand-written MLE). Covariates: quote distance from mid in robust-sigma units, distance in ticks, 3s taker volume on our side, spread, minute-of-hour / funding-hour dummies. Cancelled / requoted quotes are right-censored. Evaluate on the time-split holdout with the **right-censored log-likelihood** and calibration of P(fill within 60 s) (see [arroyo_cartea_morenopino_zohren2024_fill_survival.md](arroyo_cartea_morenopino_zohren2024_fill_survival.md)).
- Use the paper's warning as a design rule: "trade-through" fill models must include a survival / queue factor, as the plan already does. Also report the touch-but-no-fill rate in our live logs.

## Evidence quality
**Strong** (JFE, the canonical reference for limit-order survival models). Details not re-verified here.
