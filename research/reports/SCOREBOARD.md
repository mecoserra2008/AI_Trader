# Hypothesis scoreboard

_Generated from research/hypotheses.yaml + hypotheses_results.jsonl. A hypothesis is supported only if it holds in-sample AND keeps its sign in the clean holdout (docs/RESEARCH_PROTOCOL.md)._

| id | family | title | prior | in-sample | holdout | overall |
|---|---|---|---|---|---|---|
| H-W1 | calendar | Weekend volatility is lower than weekday volatility | 0.85 | ✅ supports | ✅ supports | ✅ SUPPORTED |
| H-W2 | calendar | Weekend drift without activity reverts by Monday | 0.45 | ❌ contradicts | ✅ supports | ❌ not supported |
| H-W3 | calendar | Weekend information shocks continue into the TradFi session | 0.40 | ❌ contradicts | … inconclusive | ❌ not supported |
| H-W4 | calendar | CME gaps fill more often than a random walk implies | 0.30 | ❌ contradicts | ✅ supports | ❌ not supported |
| H-W5 | calendar | The Monday NYSE open is a volatility and inflection point after eventful weekends | 0.55 | ❌ contradicts | … inconclusive | ❌ not supported |
| H-W5b | calendar | Monday NYSE-open moves after eventful weekends, measured in pre-weekend sigma | 0.55 | ✅ supports | … inconclusive | … open |
| H-R1 | regime | Trend strategies earn only in trend cells | 0.60 | ❌ contradicts | … inconclusive | ❌ not supported |
| H-R2 | regime | Mean reversion earns in range cells and loses in trends | 0.50 | ❌ contradicts | … inconclusive | ❌ not supported |
| H-S1 | shock | Large price shocks continue in high-vol trend cells and revert in ranges | 0.40 | ❌ contradicts | ❌ contradicts | ❌ not supported |
| H-S2 | shock | Shocks into calm markets continue; shocks in liquidation regimes reverse (data-generated) | 0.50 | · untested | ✅ supports | ✅ supported (forward) |
| H-T1 | regime | Hourly deviations from trend continue rather than revert (data-generated) | 0.60 | · untested | ✅ supports | ✅ supported (forward) |
| H-S3 | shock | Vol state at shock time decides continuation vs reversal (data-generated) | 0.45 | · untested | · untested | … awaiting forward data |
| H-F1 | flow | L1 order-flow imbalance explains contemporaneous and predicts short-horizon mid moves | 0.55 | · untested | · untested | · untested |
| H-F2 | flow | Depth imbalance beyond the touch adds information over L1 | 0.40 | · untested | · untested | · untested |
| H-F3 | flow | Spot-vs-perp taker-flow divergence predicts the perp | 0.45 | · untested | · untested | · untested |
| H-F4 | flow | Flow regimes differ on weekends (TradFi closed) | 0.60 | · untested | · untested | · untested |
| H-W6 | calendar | Weekend crypto moves confirmed by weekend-traded TradFi perps continue | 0.40 | … inconclusive | … inconclusive | … open |
| H-X1 | execution | Close-confirmed structural stops beat resting stops on 1h | 0.50 | ❌ contradicts | … inconclusive | ❌ not supported |
| H-X2 | regime | Trend edge lives at 4h/1d, not 1h | 0.60 | ❌ contradicts | ✅ supports | ❌ not supported |
| H-X3 | regime | High trailing realised vol predicts weaker trend returns | 0.45 | ❌ contradicts | ❌ contradicts | ❌ not supported |
| H-W7 | calendar | Weekend variance risk premium: implied vol overprices the TradFi weekend | 0.65 | ✅ supports | ✅ supports | ✅ SUPPORTED |
| H-S4 | shock | Post-jump volatility contracts quickly | 0.55 | ✅ supports | ✅ supports | ✅ SUPPORTED |
| H-F5 | execution | Order-flow signals pay as entry timing, not as standalone taker strategies | 0.45 | · untested | · untested | · untested |
| H-TP1 | calendar | Stock perps drift off-hours and revert at the US cash open | 0.45 | · untested | · untested | · untested |
| H-X2b | regime | Trend edge lives at 4h/1d, not 1h (scale-free re-test) | 0.60 | ❌ contradicts | ❌ contradicts | ❌ not supported |
| H-X2c | regime | 4h is the trend sweet spot (beats 1h); 1d is not better (data-generated) | 0.60 | · untested | · untested | … awaiting forward data |
| H-P1 | positioning | Crowded funding fades work where crowding unwinds (funding_contrarian) | 0.35 | ❌ contradicts | ✅ supports | ❌ not supported |
| H-P2 | positioning | Rising open interest predicts upward pressure (oi_price_divergence) | 0.30 | ❌ contradicts | ❌ contradicts | ❌ not supported |
| H-V1 | vol | Jump-robust EWMA forecasts variance better than plain EWMA | 0.65 | ❌ contradicts | ❌ contradicts | ❌ not supported |
| H-V2 | vol | Weekend scaling of sigma improves variance forecasts that span TradFi weekends | 0.75 | ❌ contradicts | ❌ contradicts | ❌ not supported |
| H-V1b | vol | Bias-corrected robust EWMA forecasts variance better than plain EWMA | 0.55 | ✅ supports | ✅ supports | ✅ SUPPORTED |
| H-VC1 | microstructure | Vector candles continue in their direction | 0.45 | ❌ contradicts | ✅ supports | ❌ not supported |
| H-VC2 | microstructure | Swept zones get revisited more than a random walk implies | 0.35 | ❌ contradicts | ✅ supports | ❌ not supported |
| H-VC3 | microstructure | Swept zones act as support/resistance on the first revisit | 0.35 | ❌ contradicts | ❌ contradicts | ❌ not supported |
| H-VC5 | microstructure | Vector candles backed by one-sided aggressor flow continue; balanced ones revert | 0.45 | ❌ contradicts | ❌ contradicts | ❌ not supported |
| H-VC4 | vol | Vol clusters after vector candles (unlike pure jumps) | 0.50 | ❌ contradicts | ❌ contradicts | ❌ not supported |
| H-LC2 | vol | Illiquidity spikes predict vol that the forecast misses | 0.45 | ❌ contradicts | … inconclusive | ❌ not supported |
| H-VCL1 | vol | Multi-scale HAR beats the single-scale robust EWMA | 0.55 | ✅ supports | ✅ supports | ✅ SUPPORTED |
| H-GT1 | trend | Pivot-geometry trend models earn positive net Sharpe at 4h | 0.35 | ❌ contradicts | ❌ contradicts | ❌ not supported |
| H-GT2 | trend | Geometry adds over slow trend | 0.25 | ❌ contradicts | ❌ contradicts | ❌ not supported |
| H-GT3 | levels | Directional-change swing levels act as support/resistance | 0.40 | ❌ contradicts | ❌ contradicts | ❌ not supported |
| H-VC6 | microstructure | Thin vector candles revert (data-generated) | 0.45 | · untested | · untested | … awaiting forward data |
| H-VC7 | microstructure | The first 1-sigma move after a vector candle goes against it (data-generated) | 0.45 | · untested | · untested | … awaiting forward data |
| H-VZ1 | levels | Active vector-open levels hold on the first touch | 0.40 | ❌ contradicts | ❌ contradicts | ❌ not supported |
| H-VZ2 | levels | Vector opens are better levels than ordinary candle opens | 0.35 | ❌ contradicts | ❌ contradicts | ❌ not supported |
| H-VZ3 | levels | Taken levels are switched off | 0.40 | ❌ contradicts | ❌ contradicts | ❌ not supported |
| H-VZ4 | levels | Untaken vector opens act as magnets | 0.40 | ❌ contradicts | ✅ supports | ❌ not supported |
| H-AU1 | auction | Acceptance outside the prior value area leads to continuation (imbalance) | 0.45 | ❌ contradicts | ✅ supports | ❌ not supported |
| H-AU2 | auction | Failed auction outside value rotates to the other side (the 80 percent rule) | 0.35 | ❌ contradicts | ❌ contradicts | ❌ not supported |
| H-AU3 | auction | Balance vs imbalance regimes decide whether mean reversion or trend works | 0.40 | ❌ contradicts | ✅ supports | ❌ not supported |
| H-AU4 | auction | The prior day's POC is a magnet in balance | 0.45 | ❌ contradicts | ❌ contradicts | ❌ not supported |
| H-RG1 | regime | Variance ratio detects trend vs mean-reversion regimes | 0.45 | ❌ contradicts | ❌ contradicts | ❌ not supported |
| H-VCL2 | vol | Downside variance clusters more (leverage effect) in crypto | 0.50 | ❌ contradicts | ❌ contradicts | ❌ not supported |
| H-VCL3 | vol | Vol spills over from BTC to alts | 0.55 | ❌ contradicts | ❌ contradicts | ❌ not supported |
| H-VCL4 | vol | Implied vol adds information beyond HAR | 0.60 | … inconclusive | ✅ supports | … open |
| H-VCL5 | vol | Vol compression underpredicts the expansion | 0.40 | ❌ contradicts | ❌ contradicts | ❌ not supported |
| H-FU1 | fundamentals | Stablecoin liquidity growth predicts crypto returns | 0.30 | … inconclusive | ❌ contradicts | ❌ not supported |
| H-FU2 | fundamentals | A high variance risk premium predicts positive returns | 0.35 | ❌ contradicts | ✅ supports | ❌ not supported |
| H-C1 | carry | Perp funding carry is a harvestable premium | 0.55 | ✅ supports | ✅ supports | ✅ SUPPORTED |
| H-AS1 | adverse_selection | Large-OFI candles are toxic: mid keeps moving in their direction | 0.65 | · untested | · untested | · untested |
| H-AS2 | adverse_selection | Liquidity withdrawal precedes moves (book-based crunch) | 0.55 | · untested | · untested | · untested |
| H-C1b | carry | Low-turnover carry overlay that scales with the funding regime (data-generated) | 0.35 | · untested | · untested | … awaiting forward data |
| H-OPT1 | options | Selling the weekend straddle earns a premium net of costs | 0.60 | ❌ contradicts | ❌ contradicts | ❌ not supported |
| H-OPT2 | options | The weekend straddle beats the same-length weekday straddle | 0.60 | ❌ contradicts | ✅ supports | ❌ not supported |
| H-OPT3 | options | The weekend straddle's tail is survivable at small size | 0.40 | ❌ contradicts | ✅ supports | ❌ not supported |
| H-OPT4 | options | Sell the weekend at the US close, not at Friday 08:00 (data-generated) | 0.35 | · untested | · untested | … awaiting forward data |
| H-RV1 | relative_value | Alt residuals vs a BTC/ETH factor model revert | 0.40 | ❌ contradicts | ❌ contradicts | ❌ not supported |
| H-RV2 | relative_value | The Brent-WTI spread on Bybit perps mean-reverts | 0.45 | … inconclusive | ❌ contradicts | ❌ not supported |
| H-RV3 | relative_value | Tokenised gold vs gold perp basis mean-reverts | 0.50 | ❌ contradicts | ❌ contradicts | ❌ not supported |
| H-RV4 | relative_value | Perp-spot basis extremes revert within the hour (forward) | 0.60 | · untested | · untested | · untested |
| H-TR1 | trend | An equal-weight ensemble of simple trend models earns positive net Sharpe | 0.45 | ✅ supports | ❌ contradicts | ❌ not supported |
| H-TR2 | trend | Walk-forward selection of the best trend models beats the ensemble | 0.30 | ❌ contradicts | ❌ contradicts | ❌ not supported |
| H-TR3 | regime | A walk-forward regime classifier tells trend from mean-reversion periods | 0.30 | ❌ contradicts | ✅ supports | ❌ not supported |
| H-TR4 | regime | Directional efficiency predicts trend profitability (single feature) | 0.40 | ❌ contradicts | ✅ supports | ❌ not supported |

**Totals:** · untested: 9, … awaiting forward data: 6, … open: 3, ✅ SUPPORTED: 6, ✅ supported (forward): 2, ❌ not supported: 48
