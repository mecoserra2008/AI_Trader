# State-dependent Hawkes processes and their application to limit order book modelling (Morariu-Patrichi & Pakkanen, 2022)

- **Citation:** Maxime Morariu-Patrichi, Mikko S. Pakkanen (2022). *Quantitative Finance* 22(3), 563-583. DOI [10.1080/14697688.2021.1983199](https://doi.org/10.1080/14697688.2021.1983199). arXiv [1809.08060](https://arxiv.org/abs/1809.08060).
- **Verified:** arXiv API, Crossref, and parts of the full text (arXiv PDF, read in the scratchpad), 2026-11-10.
- **Topic:** 3. Coupling order flow with book state, a formal version of "trader reaction functions". Hypotheses: H-SIM2, H-SIM3.

## Question
How can a self- and cross-exciting point process be fully coupled with a discrete state process (spread, queue imbalance), so that excitation depends on the state and events change the state?

## Data and method
- State-dependent Hawkes (sdHawkes): event types e with intensities lambda_e(t) = nu_e + sum over past events (e', x') of k_{e',e}(t - s; x'). The kernel depends on the state x' at the time of the past event. Each event moves the state via a transition matrix phi_e(x, x') that is estimated from the data.
- Existence and uniqueness proofs, a simulation algorithm, and parametric MLE (exponential kernels with state-dependent amplitudes). Goodness of fit by residual (time-rescaled) QQ plots.
- Data: tick-by-tick level-I LOB data for **Intel (INTC) on Nasdaq, June 2017 - May 2018**. State variable = bid-ask spread or queue imbalance (binned).
- Python implementation released by the authors (`mpoints`, per the paper; not checked).

## Key results
- Excitation effects in order flow are **strongly state-dependent**. The endogeneity of order flow (the magnitude of excitation) is also state-dependent, **more pronounced in disequilibrium states** of the book: wide spread, extreme imbalance (abstract).
- The model captures the feedback loop between order flow and book shape, which standard Hawkes models miss.

## Limitations / transfer to Bybit perps
- Needs event-level data with the state at each event. With flow_3s we can approximate the state per 3s bin (spread proxy from high-low or the stored book top, if any; flow imbalance), but not per event.
- Many parameters (kernel amplitude per event-type pair per state): use a coarse state (3 bins).

## How we use it here
- H-SIM3 (reaction functions): estimate a **discrete-time analogue**, a VAR / Poisson-regression of 3s buy and sell taker counts on lagged counts **interacted with state bins** (spread tercile, imbalance tercile, recent-return sign). The sign stability and liquidity-scaled equality tests in H-SIM3 then apply per state. This is the cheapest faithful translation of sdHawkes to our data.
- H-SIM2: if the plain Hawkes generator fails the flow-autocorrelation fact, upgrade to state-dependent amplitudes (the paper's main empirical finding says it should help in stressed states).

## Evidence quality
**Medium-strong** (Quantitative Finance, rigorous; one stock, one year).
