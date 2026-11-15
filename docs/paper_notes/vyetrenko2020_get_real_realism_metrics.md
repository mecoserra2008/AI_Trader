# Get Real: Realism Metrics for Robust Limit Order Book Market Simulations (Vyetrenko et al., 2020)

- **Citation:** Svitlana Vyetrenko, David Byrd, Nick Petosa, Mahmoud Mahfouz, Danial Dervovic, Manuela Veloso, Tucker Hybinette Balch (2020). *Proceedings of the First ACM International Conference on AI in Finance (ICAIF '20)*, 1-8. DOI [10.1145/3383455.3422561](https://doi.org/10.1145/3383455.3422561). arXiv [1912.04941](https://arxiv.org/abs/1912.04941) (NeurIPS 2019 workshop version).
- **Verified:** arXiv API, Crossref, and the full text (arXiv PDF, read in the scratchpad), 2026-11-10.
- **Topic:** 3. Validation checklist for market simulators. Hypothesis: H-SIM2 (the stylised-fact battery).

## Question
What metrics should a LOB simulator reproduce to be considered realistic, and how do current agent-based simulators score?

## Data and method
- A catalogue of stylised facts with formulas, grouped as:
  - **Return distributions:** absence of return autocorrelation beyond ~20 min; heavy tails with aggregational normality (kurtosis by horizon); intermittency; volatility clustering (ACF of r^2); long-range dependence of |r| (ACF decaying ~ tau^-beta with beta in [0.2, 0.4]); gain / loss asymmetry (skew); volume-volatility positive correlation; return-volatility negative correlation; asymmetric causal flow (coarse vol predicts fine vol better than the reverse).
  - **Volumes and order flow:** best-queue volumes ~ Gamma; order sizes power-law (limit ~ x^-2, market ~ x^-(2.3..2.7)) with round-number clustering; order counts per window ~ Gamma / lognormal; interarrival times exponential / lognormal / **Weibull**; new-order price distance power law (1 + mu ~ 1.6); order lifetimes power law (1 + mu ~ 1.3-1.6); time-to-fill with a generalized-gamma AFT model; time correlation of order flow.
  - **Non-stationary patterns:** U-shaped intraday volume; event / holiday effects; negative volume-spread correlation.
  - **Market impact:** mid-price move as a function of the participation imbalance P = |V_buy - V_sell| / (V_buy + V_sell), binned (concave impact).
  - **Cross-asset correlations.**
- Applied to ABIDES configurations (`sparse_zi_100`, `rmsc01`) vs real data: minutely returns of 50 random US stocks per day in 2011; NASDAQ JPM order book, June 2019.

## Key results
- The more diverse agent population (`rmsc01`) is closer to the real data, but **significant discrepancies remain**:
  - both configurations fail the zero-autocorrelation fact;
  - volume-volatility correlation is too weak;
  - intraday volume is **inverted** (low at open and close) in rmsc01;
  - interarrival times are dominated by near-zero values (poor Weibull fit).
- **Time correlation of order flow and intraday patterns are the hardest facts to reproduce.**

## Limitations / transfer to Bybit perps
- The catalogue is equity-centric (the U-shape becomes hour-of-week periodicity in crypto). Several facts need event data we don't store (order lifetimes, new-order prices).
- With flow_3s we can test the return facts, flow ACF, the volume-vol correlation, the impact curve on participation imbalance, and 3s-bin count distributions.

## How we use it here
- B4 / H-SIM2: adopt this as the **pre-registered metric battery**, extended from the 5 registered facts to the computable subset, so the generator's realism is reported as a scorecard (pass / fail per fact per class) rather than one cherry-picked number:
  - kurtosis by horizon (3s, 1m, 15m);
  - ACF of r and |r|, with the |r| power-law exponent;
  - flow ACF;
  - volume-vol correlation;
  - impact vs participation imbalance;
  - hour-of-day profile;
  - continuation / alternation ratio eta (from [huang_lehalle_rosenbaum2015_queue_reactive.md](huang_lehalle_rosenbaum2015_queue_reactive.md)).
- H-SIM2's registered success criterion (>= 4 of 5) stays as registered; the extended scorecard is descriptive.

## Evidence quality
**Medium-strong** as a validation framework (peer-reviewed conference, a well-cited catalogue). Its own empirical comparison is small.
