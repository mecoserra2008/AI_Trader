# Towards Realistic Market Simulations: a Generative Adversarial Networks Approach (Coletta et al., 2021)

- **Citation:** Andrea Coletta, Matteo Prata, Michele Conti, Emanuele Mercanti, Novella Bartolini, Aymeric Moulin, Svitlana Vyetrenko, Tucker Balch (2021). *Proceedings of the Second ACM International Conference on AI in Finance (ICAIF '21)*, 1-9. DOI [10.1145/3490354.3494411](https://doi.org/10.1145/3490354.3494411). arXiv [2110.13287](https://arxiv.org/abs/2110.13287).- **Verified:** arXiv API, Crossref, and parts of the full text (arXiv PDF, read in the scratchpad), 2026-11-10.
- **Topic:** 3. A calibrated, **responsive** market simulator (the world reacts to our orders). Hypotheses: H-SIM2, H-SIM4.

## Question
Replay-based backtests can't capture the market's response to our own orders, and agent-based models can't be calibrated without agent-level data. Can a conditional generator trained on aggregate historical data act as a responsive "world agent"?

## Data and method
- **LOBSTER** Nasdaq data, 20 levels. The order stream plus book state.
- A conditional Wasserstein GAN (gradient penalty). Generator input: noise plus a condition vector built from the recent market state (best bid / ask prices and volumes, recent order history, mid-price changes), normalised. Output: the next order as a 4-tuple **(price relative to mid, volume, direction, inter-arrival time)**.
- Integrated into ABIDES as a "world agent" that keeps generating orders conditioned on the **current** simulated state. If an experimental agent trades, the state changes and the world reacts.
- Evaluation: the stylised facts of Vyetrenko et al. (2020) plus **market-impact** response to an experimental agent's orders, compared with the IABS agent configurations.

## Key results
- The CGAN world agent reproduces the stylised facts better than prior agent-based configurations, and shows a **market response** (impact) to experimental orders that pure replay can't (abstract and experiments).

## Limitations / transfer to Bybit perps
- Needs order-level (L3-like) data. We have 3s aggregates. A GAN on 3s aggregates is possible but data-hungry (3 days of flow is far too little).
- GAN training instability; realism measured on in-sample stocks / days.
- Follow-up work by the same group on conditional generators (e.g. arXiv 2306.12806) was not reviewed.

## How we use it here
- **Concept to adopt, model not yet:** the generator must be **conditional on state**, so that our quoter's fills and requotes feed back. That is the property H-SIM4 needs (ranking quoting parameters is meaningless if the market can't react).
- Concrete v0 for B4: the conditional generator is the H-SIM3 reaction-function VAR (taker flow conditioned on state, including our resting quotes' effect on the visible top of book). Deep conditional generators wait until the user decides on longer L2 collection.

## Evidence quality
**Medium-weak** (peer-reviewed conference, one data source, GAN evaluation is soft).
