# Quant GANs: deep generation of financial time series (Wiese, Knobloch, Korn & Kretschmer, 2020)

- **Citation:** Magnus Wiese, Robert Knobloch, Ralf Korn, Peter Kretschmer (2020). *Quantitative Finance* 20(9), 1419-1440. DOI [10.1080/14697688.2020.1730426](https://doi.org/10.1080/14697688.2020.1730426). arXiv [1907.06673](https://arxiv.org/abs/1907.06673).
- **Verified:** arXiv API and Crossref, 2026-11-10. Details marked (m) come from memory.
- **Topic:** 4. Neural market generators for synthetic return paths. Hypotheses: H-SYN1, H-SYN2 (an alternative generator to the bootstrap).

## Question
Can a GAN with temporal convolutional networks (TCNs) generate return series that reproduce heavy tails, volatility clustering and the leverage effect?

## Data and method
- S&P 500 daily log returns, roughly 2009-2018 (m).
- Preprocessing (m): standardise, apply an inverse Lambert-W transform to reduce heavy tails (Gaussianise), then rolling-window normalisation. After generation, invert the transforms.
- Generator: a TCN with dilated causal convolutions (receptive field of a few hundred days) that maps noise to returns. The construction gives a "stochastic volatility neural network" that allows a transition to the risk-neutral measure (abstract). Discriminator: TCN.
- Evaluation (m): distributional metrics (Wasserstein-1 / earth mover's distance, DY metric) at several aggregation lags; scores for the ACF of r, |r| and r^2; the leverage-effect correlation. Benchmark: GARCH(1,1).

## Key results
- Distributional properties at small and large lags, plus vol clustering, leverage effects and serial autocorrelations, are reproduced "in high fidelity", better than GARCH on the reported metrics (abstract; GARCH comparison (m)).

## Limitations / transfer to Bybit perps
- A single index with ~2,400 daily observations; the evaluation is in-sample to the training series. GANs memorise and mode-collapse, so the generated paths may be near-copies of training windows, which **defeats** the purpose for selection robustness.
- Crypto has no leverage effect (V11 / R-LEV-1), so fitting it would be wrong. The facts that matter for our strategies (weekend vol ratio, jumps, regime episodes) are not targeted.
- Compute and maintenance cost are high relative to a bootstrap / HAR-t simulator. Our complexity budget says linear first.

## How we use it here
- **Not now.** B5 should first show that the transparent generators (stationary bootstrap, regime-switching resampling, HAR-t + jumps simulation) help (H-SYN1). Only if they fail on a stylised fact that matters would a TCN generator be worth testing.
- Borrow the **evaluation protocol** for B5's generator check: Wasserstein-1 of the return distribution at 1h / 1d / 1w aggregation, plus ACF scores for r, |r| and r^2, against block-bootstrap envelopes.

## Evidence quality
**Medium** (Quantitative Finance, well cited). Evidence for **strategy-selection** usefulness: none in this paper.
