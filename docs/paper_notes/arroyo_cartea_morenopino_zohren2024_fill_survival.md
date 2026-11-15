# Deep attentive survival analysis in limit order books: estimating fill probabilities with convolutional-transformers (Arroyo, Cartea, Moreno-Pino & Zohren, 2024)

- **Citation:** Alvaro Arroyo, Alvaro Cartea, Fernando Moreno-Pino, Stefan Zohren (2024). *Quantitative Finance* 24(1), 35-57. DOI [10.1080/14697688.2023.2286351](https://doi.org/10.1080/14697688.2023.2286351). arXiv [2306.05479](https://arxiv.org/abs/2306.05479).
- **Verified:** arXiv API, Crossref, and the full text (arXiv PDF, tables read in the scratchpad), 2026-11-10.
- **Topic:** 3. Fill-probability models for passive orders. Hypothesis: H-SIM1.

## Question
What is the distribution of the time-to-fill of a limit order placed at a given LOB level, conditional on the recent state of the book, and how should such models be scored?

## Data and method
- LOBSTER message data, Nasdaq, **9 stocks (AAPL, AMZN, BIDU, COST, CSCO, DELL, GOOG, INTC, MSFT), 1 Oct - 27 Dec 2022**. Both small-tick (e.g. COST, mean spread 30.9 ticks) and large-tick (INTC 1.13, CSCO 1.15, AAPL 1.44 ticks) stocks.
- Survival framing: the time from placement to fill is the event. **Cancellation, or end of day, is right-censoring.** Two datasets:
  - observed limit orders;
  - hypothetical **pegged** orders at a fixed depth, "filled" when the queue ahead is consumed.
- Baselines: Kaplan-Meier (unconditional), DeepSurv (Cox-type), DeepHit. Proposed model: a convolutional-Transformer encoder over the recent window of LOB features, with a **monotonic neural-network decoder** that guarantees a valid survival function.
- Scoring: **right-censored log-likelihood (RCLL)**, a proper scoring rule (Rindt et al. 2022). The authors note that C-index and Brier-type scores commonly used in survival analysis are not proper for this purpose.
- Interpretability: feature attributions over the lookback window.

## Key results
- Unconditional fill probabilities are low at the best level (**~4-9%** over the horizon studied, e.g. AAPL 5.4%, AMZN 9.2%) and rise for orders placed inside the spread (AAPL 13% at 1 tick inside, 36% at 2 ticks, 44% at 5 ticks inside). Fill times are sub-second to a few seconds for active stocks (AAPL 1.3 s at the best level) and 4-7 s for less active ones.
- For large-tick stocks, the fill probability of in-spread orders rises sharply at very short horizons, then falls back to best-level levels. Interpretation: fast fills mean the price moved favourably (**adverse-selection signature**); otherwise the order ends up deeper in the book.
- The conv-Transformer improves RCLL over an MLP baseline by **~62-86% on average** across stocks, versus ~17-27% for DeepSurv / DeepHit.

## Limitations / transfer to Bybit perps
- Requires the full message stream (queue position). We have ~250 own fills and our quote logs plus 3s trade candles, far too few for a deep model.
- The protocol transfers fully: survival framing with censoring, a proper scoring rule (RCLL), Kaplan-Meier by depth as the first descriptive step, and separate calibration of P(fill within h).
- The adverse-selection signature (fast fills are bad fills) is directly testable on our markouts.

## How we use it here
- B4 `sim/fill_model.py` (H-SIM1):
  - (1) Kaplan-Meier curves of our quotes by distance from mid (ticks and sigma units);
  - (2) a parametric AFT / Cox model with few covariates ([lo_mackinlay_zhang2002_limit_order_executions.md](lo_mackinlay_zhang2002_limit_order_executions.md));
  - (3) score models by RCLL on the time-split holdout, not by fill-rate error alone.
- Add the **fill-time vs markout** relation to the H-SIM1 report: if fills within 3 s have worse 60 s markouts than slower fills, the quoter should not chase. That would be a rule candidate for `atx mm` (R-MM family).

## Evidence quality
**Medium-strong** (Quantitative Finance; strong protocol; equity LOB only; a deep model unsuitable for our sample sizes).
