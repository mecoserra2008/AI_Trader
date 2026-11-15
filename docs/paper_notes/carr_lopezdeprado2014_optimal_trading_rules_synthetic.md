# Determining Optimal Trading Rules without Backtesting (Carr & Lopez de Prado, 2014)

- **Citation:** Peter P. Carr, Marcos Lopez de Prado (2014). "Determining Optimal Trading Rules without Backtesting." arXiv [1408.1159](https://arxiv.org/abs/1408.1159); SSRN [2658641](https://doi.org/10.2139/ssrn.2658641). The same procedure appears as "Backtesting on synthetic data" in Lopez de Prado (2018), *Advances in Financial Machine Learning*, ch. 13 (book chapter, from memory).
- **Verified:** arXiv API (abstract) and Crossref, 2026-11-10. Details marked (m) come from memory.
- **Topic:** 4. Synthetic data for choosing exit rules (profit-take / stop-loss) without overfitting a single historical path. Hypotheses: H-SYN1; H-PD4 (exit leakage).

## Question
Can we choose profit-taking and stop-loss thresholds from a fitted price process instead of searching them on the historical backtest?

## Data and method
- Fit a discrete Ornstein-Uhlenbeck process to the (spread / price) series the rule trades: P_t = (1 - phi) E[P] + phi P_{t-1} + sigma eps_t. Estimate phi (equivalently the half-life) and sigma by OLS on historical data.
- Simulate many synthetic paths (e.g. 100,000) from the fitted process. On a mesh of (profit-take, stop-loss) pairs, with a maximum holding period, compute the Sharpe of the rule on synthetic paths. Pick the optimal trading rule (OTR) on the mesh.
- Repeat across (forecast E[P], half-life, sigma) to map how the OTR depends on the process parameters.

## Key results
- Optimal rules exist and can be found numerically for OU processes (abstract). The authors conjecture a closed form exists but don't derive it.
- Qualitatively (m): with strong mean reversion (short half-life), tight profit-takes and wide stops are optimal. As the process approaches a random walk (phi -> 1, long half-life), the Sharpe surface flattens and **no rule has an edge**, so exits should not be "optimised" at all.

## Limitations / transfer to Bybit perps
- OU fits a mean-reverting spread. Our directional strategies trade near-random-walk prices with fat tails and jumps, where the paper's own result says the surface is flat. For them, the value is negative knowledge: don't tune exits.
- It applies better to quantities with genuine mean reversion: market-making inventory (R-MM-1: passive exits), the XAUT-XAU basis (reverts ~28%/24h, below costs, R-AVOID-1), funding-rate deviations.
- The rule's optimum is only as good as the fitted model. Add jumps (our `features/jump_diffusion.py`) and Student-t innovations, otherwise stops look too safe (V3).

## How we use it here
- B5: for **exit parameters only** (stop multiple, take-profit, max holding) of the deterministic strategies, choose them on paths simulated from the fitted jump-diffusion / HAR model, and compare holdout degradation with historical-path selection (a sub-case of H-SYN1).
- B4: choose the MM take-profit / inventory exit on simulated OU-plus-jump paths of the mid around our inventory (complements the H-SIM4 grid).
- H-PD4 interpretation: if exit leakage is large but the simulated Sharpe surface is flat, leakage is unavoidable noise, not a fixable design flaw.

## Evidence quality
**Medium** (sound method, simulation-only evidence, not peer-reviewed in this form).
