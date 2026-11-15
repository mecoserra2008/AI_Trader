# Where does the criticality live? Early-warning signals across seven crypto-perpetual liquidation cascades (Garcia Seuma, 2026)

- **Citation:** Ramon Marc Garcia Seuma (2026). "Where does the criticality live? Early-warning signals are event-heterogeneous across seven crypto-perpetual liquidation cascades." arXiv [2607.27070](https://arxiv.org/abs/2607.27070) (29 Jul 2026). Companion: same author, "Measuring the engine of a liquidation cascade: subcritical branching inside a first-order transition," arXiv [2608.03616](https://arxiv.org/abs/2608.03616) (4 Aug 2026).
- **Verified:** arXiv API (titles, author, abstracts), 2026-11-10. Only the abstracts were read. Single author, not peer-reviewed, and posted **after** Claude's training cutoff.
- **Topic:** 2. Liquidation cascades in crypto perps (structural flow). Relevant to H-OF3 / H-OF7 (liquidation clusters did not revert, R-FLOW-2), R-PX-8 (after a liquidation regime, expect whipsaw), R-MM-2 (correlated fills).

## Question
Do crypto perpetual crashes (liquidation cascades) show a reproducible early-warning signature of a critical transition, and in which variable?

## Data and method
- Seven major BTC liquidation cascades, 2022-2025, including the record ~19B USD event of 10 Oct 2025. Minute-level price data and 5-minute leverage / order-flow data.
- Early-warning indicators on detrended residuals: rolling variance and lag-1 autocorrelation, with the pre-cascade trend tested by Kendall tau. 39 analysis configurations per variable per event (a specification sweep). A 300-onset placebo test.
- The companion paper measures the cascade's branching ratio from a transparent on-chain venue's fill log (October 2025 event), plus inter-asset coupling and liquidity measures at onset across the seven events.

## Key results (abstracts)
- **No early-warning variable is event-invariant.** Price shows critical slowing down in 5 of 7 events but not in the two sudden-news (tariff) shocks. This suggests two types: endogenous build-up vs exogenous shock.
- The only regularity across all events with data is a **compression of taker order-flow variance** before onset. It passes a 300-onset placebo test (Fisher-combined p ~ 5e-6), but it is a population-level precursor, **not a per-event alarm**.
- Companion: the October 2025 cascade ran **subcritical** (branching ratio ~0.1-0.2). 88% of post-onset forced selling landed within 30 minutes. At onset, mean inter-asset coupling jumps 1.6-4.4 baseline sd into a near-fully-ordered phase. Open interest clears 25-70%. Severity is set by shock size x map-in-path x liquidity withdrawal, not by a diverging multiplier.

## Limitations / transfer to Bybit perps
- n = 7 events with a 39-configuration sweep is a garden of forking paths. The author says single-event claims are fragile. Not peer-reviewed.
- Directly relevant to us, though: the cascade dynamics (88% of forced selling within 30 minutes, coupling jumping toward 1) are consistent with our H-OF3 result (clusters continued over 60 minutes rather than reverting) and with R-MM-2 (fills across contracts are one bet).

## How we use it here
- Risk / quoting (B4 replay): during a detected liquidation cluster (R-FLOW-2 trigger), treat cross-contract correlation as ~1 for inventory limits and expect the forced flow to be mostly done within ~30 minutes. Candidate rule: suspend fading for 30 minutes after cluster onset.
- Flow-variance compression as a **weak, population-level** state feature for H-OF6 (selective detector): test whether low rolling variance of 1m signed taker flow raises the base rate of a >3 sigma 1h move. Draft hypothesis H-LIT-NS7 (literature report).

## Evidence quality
**Weak** (single author, unrefereed, n = 7, specification sweep). Kept because it is the only crypto-perp-specific cascade study found and it matches our own data.
