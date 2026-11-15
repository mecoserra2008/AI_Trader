# Crypto Carry (Schmeling, Schrimpf & Todorov, Management Science 2026)

- **Citation:** Maik Schmeling, Andreas Schrimpf, Karamfil Todorov. "Crypto Carry." *Management Science* (2026, forthcoming / online). DOI [10.1287/mnsc.2024.05069](https://doi.org/10.1287/mnsc.2024.05069). Earlier versions: SSRN 4268371 (2022) and a BIS working paper.
- **Verified:** Crossref (Management Science record) and the Semantic Scholar abstract, 2026-11-10. Numbers marked (m) come from memory of the BIS / SSRN version.
- **Topic:** 2. Funding mechanics / perp basis; carry as a leverage-demand premium. Relevant to H-C1 / H-C1b, R-CARRY-1, H-LEV5 / H-LEV6.

## Question
How large is crypto carry (futures-spot basis / perp funding), what drives it, and what does it predict?

## Data and method
- Futures-spot basis on major crypto venues (CME and offshore futures; perp funding), BTC and ETH, 2018 to the early 2020s (m).
- Decomposition of carry into a convenience (inconvenience) yield. Regressions of carry on trend-chasing retail demand proxies (past returns, retail flows) and on arbitrage-capital proxies (margin, regulatory frictions). Carry is then used to predict future returns, crash risk and liquidations (m).

## Key results
- Carry can reach exceptionally high levels, **sometimes over 40% per year**, and varies a lot over time (abstract).
- It reflects a large, volatile inconvenience yield of holding spot vs futures (abstract).
- Drivers: (i) demand from smaller, **trend-chasing investors** seeking leveraged exposure; (ii) **limited arbitrage capital** due to regulatory and margin frictions (abstract).
- High carry predicts higher crash / liquidation risk for carry trades (m).

## Limitations / transfer to Bybit perps
- Venue mix and period differ (pre-2024, including CME). Since spot ETFs (2024) arbitrage capital has grown, which per the paper's own mechanism should compress carry. That is consistent with our C1 decay finding.
- It is about the level of carry, not its intraday mechanics.

## How we use it here
- Supports the existing belief C1 / R-CARRY-1 with an independent peer-reviewed mechanism: carry is a leverage-mania premium that shrinks as arbitrage capital grows.
- Supports a **vol / crash-risk** use of funding (H-LEV5's funding-z main effect, H-LEV6) rather than a direction use (H-P1 funding contrarian failed).
- B1 attribution: the funding-carry factor and the trend factor should co-move in mania phases (both driven by trend-chasing demand). Test their correlation by regime.

## Evidence quality
**Strong** (Management Science; BIS authors; consistent with He et al. 2022 and with our own data).
