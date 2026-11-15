# Simulating and analyzing order book data: The queue-reactive model (Huang, Lehalle & Rosenbaum, 2015)

- **Citation:** Weibing Huang, Charles-Albert Lehalle, Mathieu Rosenbaum (2015). *Journal of the American Statistical Association* 110(509), 107-122. DOI [10.1080/01621459.2014.982278](https://doi.org/10.1080/01621459.2014.982278). arXiv [1312.0563](https://arxiv.org/abs/1312.0563).
- **Verified:** arXiv API, Crossref, and the full text (arXiv PDF, read in the scratchpad), 2026-11-10.
- **Topic:** 3. State-dependent (reactive) LOB simulator: the reference design for "trader reaction functions". Hypotheses: H-SIM2, H-SIM3, H-SIM4.

## Question
Can an order-book simulator in which order-flow intensities depend on the current book state reproduce both microstructure and lower-frequency stylised facts, and serve as a TCA / execution testbed?

## Data and method
- Cheuvreux LOB database, **Euronext Paris, Jan 2010 - Mar 2012**, 5 levels per side, every book change; first and last trading hours removed. Large-tick stocks **France Telecom** (159,250 orders / day, 7,282 trades / day, mean spread 1.43 ticks) and **Alcatel-Lucent**.
- Between changes of a reference price p_ref, the LOB is a **Markov queueing system** whose limit / cancel / market-order intensities at queue Q_{+-i} depend on the **queue's own size** (Model I) and on neighbouring queues (Models IIa / IIb). Intensities are estimated non-parametrically by binning the queue size and counting events per unit time in each bin.
- Model III adds price dynamics. When a best queue depletes, p_ref moves by a tick with probability theta. With probability theta_reinit, the whole book is redrawn from its invariant distribution after a price change (exogenous information). theta and theta_reinit are calibrated to match 10-minute volatility and the mean-reversion ratio eta = N_continuations / (2 N_alternations).
- Applications: P(execution of a buy order at Q_{-1} before the mid moves) as a function of the initial best-queue sizes, by Monte Carlo; order-placement analysis.

## Key results
- State-dependent intensities reproduce the invariant distributions of queue sizes well.
- **A purely book-driven model (theta_reinit = 0) cannot generate enough volatility.** Its maximal "mechanical" volatility is **5 bp vs 14 bp empirical** (10-minute, France Telecom), and its mean-reversion ratio is **eta = 0.08 vs 0.39 empirical**. Exogenous information (theta_reinit > 0) is needed to match both.
- Fill probability of a passive order depends strongly on both best-queue sizes (the opposite queue matters as much as one's own).

## Limitations / transfer to Bybit perps
- Requires L2 event data, which we don't store (only flow_3s plus top of book). Large-tick assumption: fine for BTC / ETH perps, less so for alts with wide spreads in ticks.
- Paris equities from 2010-2012; crypto books have more cancellations and much more HFT quoting, so intensities must be re-estimated, but the **structure** transfers.

## How we use it here
- **Key design lesson for B4's `sim/flow_sim.py` (H-SIM2):** a generator driven only by endogenous flow (Hawkes / VAR on own lags) will under-produce volatility and over-produce mean reversion. Include an **exogenous price-jump / reset component** (e.g. driven by BTC moves for alts, or by our jump model `features/jump_diffusion.py`) and calibrate its rate to match 1-minute RV and the continuation / alternation ratio eta. Add eta to the H-SIM2 stylised-fact list; it is cheap to compute from 3s mid changes.
- Reaction functions (H-SIM3): estimate taker-flow intensity as a non-parametric function of state bins (spread, best-queue imbalance, recent return), exactly as the paper bins queue size. This is the "implicit trader types" decomposition the plan asks for.
- Fill model (H-SIM1): Monte Carlo fill probability conditional on (own queue ahead, opposite queue). It needs at least a top-of-book size snapshot at quote time; check whether the `atx mm` logs have it.

## Evidence quality
**Strong** (JASA, influential, open method; the vol and eta numbers verified in the text).
