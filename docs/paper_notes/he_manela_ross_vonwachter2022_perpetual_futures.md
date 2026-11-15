# Fundamentals of Perpetual Futures (He, Manela, Ross & von Wachter, 2022, arXiv v7)

- **Citation:** Songrun He, Asaf Manela, Omri Ross, Victor von Wachter (2022, revised to v7). "Fundamentals of Perpetual Futures." arXiv [2212.06888](https://arxiv.org/abs/2212.06888). Working paper; no journal reference in the arXiv record.
- **Verified:** arXiv API (title, authors, abstract) and the arXiv HTML (v5), 2026-11-10. Numbers below are from v5 (June 2024).
- **Topic:** 2. Funding mechanics / perp basis. Relevant to H-C1 / H-C1b (carry), H-NS1 (settlement pressure) and to decomposing carry-strategy P&L.

## Question
What is the no-arbitrage price of a perpetual future given its funding rule, how far do crypto perps deviate from it, and is the deviation tradable?

## Data and method
- Binance BTC, ETH, BNB, DOGE, ADA (73% of crypto market cap, Mar 2024). Hourly prices, 8h funding. Jan 2020 - Mar 2024. Aave stablecoin lending / borrowing rates as the arbitrageur's funding cost.
- Theory: with funding proportional to (F - S) with intensity kappa, the frictionless no-arbitrage price is F_t = kappa / (kappa - (r - r')) x S_t. Here r is the cash rate and r' the asset's holding (staking / lending) rate. With trading costs, deviations are bounded: |F_t - lambda S_t| <= C, where C is the round-trip cost.
- "Random-maturity arbitrage": go long spot / short perp (or the reverse) when the deviation exceeds the cost bound, and unwind when it closes. Its return splits into convergence and funding receipts.

## Key results (v5)
- The mean deviation is about zero, but the mean **absolute** deviation is 57-88% per year (annualised), much larger than in FX forwards.
- Arbitrage Sharpe ratios even with high trading costs: BTC 1.80, ETH 2.55, BNB 4.84, DOGE 3.58, ADA 2.68. Without costs, annualised returns are BTC 13.7%, ETH 21.2%, BNB 54.8%.
- Absolute deviations **decline ~11% per year**. Convergence contributes 64% of the arbitrage return and funding 36%, the funding share declining.
- Deviations comove across coins (a common factor). Trailing 120-day returns explain over 50% of spread variation (momentum-driven leveraged demand); volatility explains little.

## Limitations / transfer to Bybit perps
- Requires a spot leg (Bybit spot or another venue) and capital on both legs; our system trades perps only. As a **signal**, the perp-spot deviation is a crowding gauge, consistent with H-LEV5 / R-LEV-5 (funding z points to higher next-day vol).
- The decay (~11% a year) matches our own carry finding (C1: 57%/yr in 2021 -> negative in 2026 H1). The literature and our data agree that perp-basis arbitrage is a fading, leverage-cycle premium.
- The momentum link (trailing 120d return explains spreads) means basis / funding is partly a **trend-crowding** factor. Carry and trend sleeves should be expected to correlate in mania phases.

## How we use it here
- B1 factor set: add a "basis" factor, the cross-sectional mean of (perp mark - index)/index or the funding rate, as a crowding factor for attributing trend and carry sleeves.
- Rule-candidate support for R-CARRY-1: the premium is regime-dependent and decaying (now two independent sources).
- B3 H-NS1: the funding-settlement mechanics in the paper (8h discrete funding approximating the average spread) suggest the pre-settlement flow should scale with |funding| x position size. Use OI x funding as the regressor, not just sign(funding).

## Evidence quality
**Medium-strong**: a well-known working paper by credible authors, with clear theory and replicable data; not yet peer-reviewed per arXiv.
