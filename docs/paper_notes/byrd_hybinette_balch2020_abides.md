# ABIDES: Towards High-Fidelity Multi-Agent Market Simulation (Byrd, Hybinette & Balch, 2020)

- **Citation:** David Byrd, Maria Hybinette, Tucker Hybinette Balch (2020). "ABIDES: Towards High-Fidelity Multi-Agent Market Simulation." *Proceedings of the 2020 ACM SIGSIM Conference on Principles of Advanced Discrete Simulation (PADS)*, 11-22. DOI [10.1145/3384441.3395986](https://doi.org/10.1145/3384441.3395986). arXiv [1904.12066](https://arxiv.org/abs/1904.12066) ("ABIDES: Towards High-Fidelity Market Simulation for AI Research").
- **Verified:** arXiv API and Crossref, 2026-11-10. Only the abstract and the descriptions in Vyetrenko et al. (2020) and Coletta et al. (2021) were read.
- **Topic:** 3. Agent-based LOB simulator (an open-source reference implementation). Hypotheses: H-SIM2, H-SIM4.

## Question
Provide an open, discrete-event, multi-agent market simulator with realistic exchange mechanics and latencies for AI / trading research.

## Data and method
- Discrete-event kernel. An exchange agent with a NASDAQ-like matching engine (message design after ITCH / OUCH). Tens of thousands of trading agents with configurable pairwise latencies.
- Background agent types (from the Vyetrenko et al. 2020 description): zero-intelligence (ZI) agents, momentum agents, market makers, and noise agents with a fundamental value following a mean-reverting process with jumps. Configurations such as `rmsc01` (a mixed population) and `sparse_zi_100`.
- Example use: a market-impact experiment.

## Key results
- A working platform. Fidelity is **not** established in this paper: the follow-up Vyetrenko et al. (2020) shows several stylised facts are not reproduced (U-shaped intraday volume, return autocorrelation, interarrival times).

## Limitations / transfer to Bybit perps
- Python and slow at scale. Agent parameters are not calibrated from data (the "calibration gap" that Coletta et al. address with a GAN "world agent").
- Equity-day structure (open, close); crypto is 24/7.

## How we use it here
- **Don't adopt** as a dependency. Borrow two design ideas for B4's `sim/flow_sim.py` / `sim/replay.py`:
  - (1) the experimental agent (our quoter) interacts with a background flow model, so our orders can **impact** the simulated market, which plain replay cannot capture;
  - (2) latency per agent. Our real latency to Bybit (requote delay) belongs in the replay.
- The background flow should come from our calibrated reaction functions (H-SIM3), not from uncalibrated agent zoos.

## Evidence quality
**Medium** as infrastructure (peer-reviewed conference, widely used). **Weak** as evidence of realism.
