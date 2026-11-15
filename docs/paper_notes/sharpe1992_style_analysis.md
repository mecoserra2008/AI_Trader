# Asset allocation: Management style and performance measurement (Sharpe, 1992)

- **Citation:** William F. Sharpe (1992). "Asset allocation: Management style and performance measurement." *Journal of Portfolio Management* 18(2), 7-19. DOI [10.3905/jpm.1992.409394](https://doi.org/10.3905/jpm.1992.409394)
- **Verified:** Crossref metadata, 2026-11-10 (Semantic Scholar: ~1,000 citations). Paper text not re-read; the method below is the standard, widely reproduced specification.
- **Topic:** 1. P&L decomposition with observable factors. Program-2 hypothesis: H-PD2.

## Question
Can a fund's "style" (its effective exposure to a set of asset-class indices) be inferred from its returns alone, and how much of its return is style versus selection?

## Data and method
- Returns-based style analysis: regress the fund return on K asset-class index returns with weights constrained to be non-negative and sum to 1. It is a quadratic programme that minimises the variance of the tracking error.
- Sharpe used 12 asset classes (bills, several bond types, value / growth / small-cap US stocks, non-US bonds, European and Japanese stocks) and rolling 60-month windows, so style drift is visible.
- "Selection" return = fund return minus the style-benchmark return. Its mean and t-stat measure skill.
- Out-of-sample discipline: estimate the weights on months t-60..t-1 and measure selection on month t.

## Key results
- For many US mutual funds, a few asset-class exposures explain most of the return variance (high style R-squared). The residual selection return was small and often negative after fees. Per-fund R-squared values were not re-checked here.
- Styles drift over time, so the estimation must roll.

## Limitations / transfer to Bybit perps
- The long-only constraint (weights >= 0 summing to 1) is wrong for long/short or directional strategies. For our strategies, use unconstrained OLS with HAC SEs, or allow signed weights.
- Linear exposures miss option-like payoffs: trend following is long a straddle, not long an index (see [fung_hsieh2001_trend_followers.md](fung_hsieh2001_trend_followers.md)).
- Our "asset classes" must be built point in time from the panel: the crypto market (equal / cap-weighted L universe), BTC, alt-minus-BTC, a TSMOM factor, a funding-carry factor, a vol-change factor, and the QQQ, gold and oil perps.

## How we use it here
- `research/pnl_decomposition_study.py` (B1), H-PD2: run rolling 90-day style regressions of each strategy's daily net P&L on the observable factors. Report style weights, R-squared and the selection residual's mean with NW t, per segment (`llm_in_sample` / `clean_holdout`).
- This feeds a "what each strategy really is" line for the decision layer, e.g. "donchian_4h = 0.6 x TSMOM + 0.2 x crypto market, selection ~0".

## Evidence quality
**Strong** as a measurement tool (decades of use). It is descriptive, not a source of alpha.
