# The Overnight Drift (Boyarchenko, Larsen & Whelan, 2023), with the 2026 follow-up "The Disappearing Overnight Drift"

- **Citation:** Nina Boyarchenko, Lars C. Larsen, Paul Whelan (2023). "The Overnight Drift." *Review of Financial Studies* 36(9), 3502-3547. DOI [10.1093/rfs/hhad020](https://doi.org/10.1093/rfs/hhad020). SSRN 3546173.
- **Follow-up:** Boyarchenko, Larsen, Whelan (1 July 2026). "The Disappearing Overnight Drift." *Liberty Street Economics*, Federal Reserve Bank of New York. DOI [10.59576/lse.20260701](https://doi.org/10.59576/lse.20260701); [web](https://libertystreeteconomics.newyorkfed.org/2026/07/the-disappearing-overnight-drift/)
- **Verified:** Crossref and the RePEc abstract (RFS paper); the blog post was fetched and read, 2026-11-10. The blog post is dated after Claude's training cutoff, so its content is not memorised.
- **Topic:** 2. TradFi session / structural-flow effects. Hypotheses: H-TF2, H-TP1; general anomaly-decay lesson.

## Question
When during the 24-hour cycle do US equity index futures earn their returns, and why?

## Data and method
- E-mini S&P 500 futures, intraday, 1998-2020 (RFS paper). Returns by hour of the 24h cycle.
- The return in the 2:00-3:00 ET window (European open) is regressed on the previous US close's order imbalance (from closing-auction / end-of-day data). Asymmetry is tested after sell-offs vs rallies.
- Model: market makers absorb end-of-day imbalances and unload inventory when European liquidity arrives, with time-varying risk-bearing capacity.

## Key results
- U.S. equity returns are large and positive during the **opening hours of European markets** (abstract). Per the 2026 post, the 2:00-3:00 ET window earned **~3.7% per year** in 1998-2020, over 60% of the E-mini's 5.9% annualised close-to-close return.
- The return is strongly related to **order imbalances at the previous US close** (inventory risk). Sell-offs generate strong positive overnight reversals; rallies only weak ones (abstract).
- **2026 follow-up:** since 2021 the window has averaged **close to zero**. The proposed cause is that the dispersion of end-of-day imbalances fell by more than half (sd from 6.5% to 2.9%), as algorithms slice orders more finely. Two ETFs launched in June 2022 to harvest overnight returns closed within 14 months.

## Limitations / transfer to Bybit perps
- The mechanism (inventory left over from the cash close and unloaded at a later liquidity event) has crypto analogues. One is the funding-settlement hours (00/08/16 UTC), where position flows bunch; another is the NYSE open for equity perps. But the effect itself is now dead in its home market.
- This is a clear case of a published, mechanism-backed anomaly decaying once it is known and the frictions shrink. Treat literature priors for session effects as **upper bounds**.

## How we use it here
- B2 (H-TF2, H-TP1): when testing the night effect and the open re-anchoring on equity perps, condition on the sign of the prior US-session move. The asymmetric-reversal prediction (stronger after sell-offs) is a sharper test than the unconditional mean.
- Crypto analogue for B3: a "settlement-hour inventory" test. Flow imbalance in the last hour before a funding timestamp predicts the next hour's return with the opposite sign (complements H-NS1).
- Lesson for the research protocol: the `clean_holdout` is the only defence against decayed anomalies; in-sample support for a classic effect is weak evidence.

## Evidence quality
**Strong** (RFS plus an author-run update), but the effect has **disappeared** since 2021. Expected value of harvesting it directly is low.
