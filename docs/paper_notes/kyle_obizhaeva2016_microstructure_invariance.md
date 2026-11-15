# Market Microstructure Invariance: Empirical Hypotheses (Kyle & Obizhaeva, 2016)

- **Citation:** Albert S. Kyle, Anna A. Obizhaeva (2016). *Econometrica* 84(4), 1345-1404. DOI [10.3982/ECTA10486](https://doi.org/10.3982/ECTA10486). SSRN 2722524.
- **Verified:** Crossref and Semantic Scholar (~170 citations), 2026-11-10. Paper text not re-read; the formulas below are the paper's well-known invariance predictions, and numbers marked (m) come from memory.
- **Topic:** 3. **Transfer across markets by liquidity scaling**, the theoretical basis for H-SIM3's "scale by sigma, depth, trade size".

## Question
Is there a scaling law that makes trading-game quantities (bet size, number of bets, transaction costs) identical across assets once measured in units of "business time" risk?

## Data and method
- Define trading activity W = P x V x sigma: dollar volume times returns volatility (daily). Invariance hypothesis: the distribution of the dollar risk transferred by a "bet" per unit of business time is the same across assets and time.
- Predictions, relative to a benchmark asset with activity W*:
  - number of bets per day gamma proportional to W^(2/3);
  - mean bet size as a fraction of daily volume, |Q| / V, proportional to W^(-2/3);
  - transaction cost in basis points for a bet of given relative size proportional to sigma x W^(-1/3) (spread and impact scale together); the cost function C(Q) = sigma W^(-1/3) f(|Q| / V W^(2/3)) has an invariant shape f.
- Test: large dataset of **portfolio-transition orders** (proxies for bets) in US stocks (hundreds of thousands of orders, early 2000s (m)). Regress log(order size / volume) on log W, estimate the exponent, and test the predicted -2/3. Also implementation-shortfall costs.

## Key results (m)
- Estimated exponent for order size / volume close to the predicted **-2/3**. The distribution of scaled order sizes is approximately **log-normal** and invariant across activity groups.
- Transaction costs (spread plus impact) scale approximately as predicted, giving a one-parameter-family cost model calibrated on liquid stocks and transferred to illiquid ones.

## Limitations / transfer to Bybit perps
- Calibrated on US equities. Crypto perps have different participant mixes (retail leverage, HFT market makers), and "bets" are unobservable. Our proxy would be trade clusters / metaorders, but we can't identify metaorders from anonymous trades.
- The **testable** part with our data is the scaling of trade-level quantities: median trade size / volume, trades per 3s, spread in bp, and impact per unit of signed flow, as functions of W = notional volume x sigma across our 23 + 15 perps.
- Pre-registering the 2/3 exponents gives H-SIM3 a sharp, falsifiable form instead of "statistically indistinguishable after scaling".

## How we use it here
- B4 / H-SIM3 transfer step: rescale each symbol's reaction-function coefficients using invariance units. Time in "business time" (proportional to W^(2/3)), size in units of V x W^(-2/3), price in units of sigma x W^(-1/3) (for impact / spread). Then test equality across symbols. Plain sigma / depth / median-size scaling is the alternative to compare.
- Draft hypothesis H-LIT-SIM6 (literature report): across Bybit perps, log(trades per day) has slope 2/3 on log W and log(spread bp / sigma) has slope -1/3.
- Cost model for the risk engine / candidate generation: a spread-plus-impact cost curve per symbol from W, instead of the flat 7.5 bp, for illiquid alts and TradFi perps (`config/mm_illiq20.yaml` suggests these are being quoted).

## Evidence quality
**Strong** (Econometrica, theory plus large-sample tests, independent confirmations in other markets (m)).
