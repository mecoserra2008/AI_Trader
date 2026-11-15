# A Stochastic Model for Order Book Dynamics (Cont, Stoikov & Talreja, 2010)

- **Citation:** Rama Cont, Sasha Stoikov, Rishi Talreja (2010). *Operations Research* 58(3), 549-563. DOI [10.1287/opre.1090.0780](https://doi.org/10.1287/opre.1090.0780). SSRN 1273160.
- **Verified:** Crossref and Semantic Scholar (~520 citations), 2026-11-10. Paper text not re-read; numbers / data marked (m) come from memory.
- **Topic:** 3. Zero-intelligence queueing LOB model, the baseline simulator and fill-probability calculator. Hypotheses: H-SIM1, H-SIM2.

## Question
Can a simple continuous-time Markov model of order arrivals and cancellations at each price level reproduce LOB behaviour and give analytic short-horizon probabilities (mid moves up, a limit order executes before the mid moves)?

## Data and method
- The LOB is a vector of queue sizes at price levels i = 1..n ticks from the opposite best quote. Independent Poisson flows:
  - limit orders at distance i with rate lambda(i), fitted as a power law lambda(i) = k / i^alpha;
  - market orders at rate mu;
  - cancellations at rate theta(i) per resting order (proportional to queue size).
- Estimation from order-book event data by simple counts (rates = number of events / time; theta from cancellations per resting share). Calibrated on Tokyo Stock Exchange stocks (m).
- Using Laplace transforms and birth-death first-passage theory, it computes conditional probabilities given the current queue sizes: P(next mid move is up), P(a bid order executes before the ask moves), P(making the spread).

## Key results
- The model reproduces the average depth profile (a hump shape) and gives conditional probabilities close to empirical frequencies (m).
- Short-horizon dynamics are largely captured by **queue sizes at the best quotes** (queue imbalance predicts the next mid move).

## Limitations / transfer to Bybit perps
- Poisson flows are state-independent: no reaction to imbalance or recent trades (fixed in Huang-Lehalle-Rosenbaum 2015). No self-excitation (fixed by Hawkes models).
- Needs L2 event data (book deltas). We store only 3s candles of trades / flow (flow_3s), so we can't estimate lambda(i) and theta(i) directly. Raw book-delta collection is the user's decision (plan: "a better simulator needs longer L2 data").
- Large-tick vs small-tick: Bybit BTCUSDT is large-tick (spread usually 1 tick), alts vary. The model fits large-tick books best.

## How we use it here
- As the **analytic baseline** for `sim/fill_model.py` (H-SIM1). With only top-of-book snapshots plus 3s trades, a reduced CST model is feasible: the best-queue size Q, a market-order arrival rate mu (from 3s taker volume), and a cancellation rate theta (unknown, so fit it as the free "queue-survival" parameter on our 250 fills). P(fill before the mid moves) then follows from the birth-death formulas.
- Compare its fill-probability predictions with the "trade-through x survival" heuristic in the plan.

## Evidence quality
**Strong** (Operations Research, a canonical model, many extensions).
