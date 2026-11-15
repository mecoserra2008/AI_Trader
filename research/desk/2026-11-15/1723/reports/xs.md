# XS 20261115T1723 (as_of 17:23 UTC)
## Regime
Broad R9: 17/60 lake tokens in R9, 5 in R8, 3 in R6, 3 in R4. Crypto EW basket (n=46) -162bp 4h / -36bp 24h /
-1479bp 7d: the weekend crash is still unwinding, not reversing. TradFi basket (n=14) is flat-to-up (24h +65bp) on
thin weekend liquidity. Funding on BTC/ETH/SOL/XRP/DOGE/WLD/AVAX sits simultaneously at the 0th-5th percentile of
its own 90d window (shorts paying the most in 90 days), but OI is split underneath: BTC and XRP are rebuilding OI
off a 72h low (new shorts into the drop), ETH/SOL/AVAX are deleveraging (OI falling with price). No sector is
decoupled from the move -- this is beta, not rotation.

## Ideas
### 1. BTCUSDT XRPUSDT ETHUSDT SOLUSDT positioning none (conviction 0.35)
- Thesis: BTC and XRP match the R-RB-4 pattern exactly -- funding below the 90d 10th percentile while OI has
  rebuilt >=3% off its 72h low after the flush. That pattern tested as no evidence for a squeeze (next 24h if
  anything weaker). With R-RB-1 (don't buy the rebound in R9) on top, there is no tested long or short edge, only
  a crowded, split-positioning posture to watch.
- Evidence: funding.pct BTC 0.039 / XRP 0.009 (xs); OI from 72h low BTC +3.15% / XRP +6.63% (tape players.oi); OI
  24h ETH -2.28% / SOL -0.74% (xs, deleveraging contrast)
- Invalidation: BTC funding.pct closes back above 0.50 (90d median) or BTC OI falls back within 1% of its 72h low
- Rules: R-RB-4 (funding<p10 + OI rebuild is not squeeze evidence), R-RB-1 (don't buy the R9 rebound), R-RB-2,
  R-AVOID-1, R-MAP-5, R-BUDGET-1
- Why this rank: direct, cited rule match on the two most liquid majors in the universe, not an inference.

### 2. SOPHUSDT watch none (conviction 0.25)
- Thesis: the lake's largest dislocation -- -5332bp (-2.21sd) 24h vs a basket at -36bp -- with OI +26.4% into the
  drop (new shorts), funding at p5 and ann -3393%, and a -54.5bp mark/index basis past the 25bp flag threshold.
  Classic funding-contrarian / OI-divergence shape that R-AVOID-1 says has no tested edge; book is thin (15.5bp
  spread, ~$330/$91 top-of-book) even against this desk's $50-200 tier.
- Evidence: rs_sigma.24h -2.21, oi.chg_pct.24h +26.37%, funding.ann_pct_last -3393.4 (p 0.048), basis_bp.mark_index
  -54.5 (xs); book.spread_bp 15.48 (tape)
- Invalidation: basis_bp.mark_index back inside +-25bp and oi.chg_pct.24h turns negative (covering, not building)
- Rules: R-AVOID-1, R-PD-4, R-BUDGET-1
- Why this rank: most multiply-flagged single name this cycle, but a thin-book name with no tested dislocation
  edge, so it sits below idea 1.

### 3. VVVUSDT watch none (conviction 0.20)
- Thesis: +1436bp (+2.53sd) 24h vs basket while its own sector (ai, n=4, includes VVV) averages only +3.5bp 24h --
  genuine idiosyncratic dispersion, not sector beta -- in a liquid, tight book (2.98bp spread, ~$1.8-2.2k depth at
  10bp), unlike SOPH/KAVA/INIT. Still no tested reversion or continuation edge for a sigma outlier: R-XS-2/R-XS-3
  cover momentum ranks specifically and both rank backwards or lose net of costs.
- Evidence: rs_sigma.24h 2.53 (xs); ai sector rs_bp_mean.24h 3.5, n 4 (xs sectors[8]); book.spread_bp 2.98,
  depth_usd.10bp.bid 1761 (tape)
- Invalidation: rs_sigma.24h reverts back inside +-1.5
- Rules: R-XS-2, R-XS-3, R-BUDGET-1
- Why this rank: cleanest textbook dislocation (liquid, sector-flat) but still untested either direction --
  listed for awareness only.

## No trade / watch
No relative-value trade this cycle. All three items above are posture/watch, not entries: the majors' positioning
extreme matches a pattern that tested as no squeeze evidence (R-RB-4); the two single-name dislocations (SOPH,
VVV) have no tested reversion or continuation edge (R-XS-2/3, R-AVOID-1).

## Data gaps
- SOPH/INIT/KAVA/ASTER/ZRO regime.cell is "not in lake" or "short history": their v0 cell, and so R-MAP-*/R-RB-*
  applicability, is unknown despite being flagged on rs_sigma/funding/OI/basis.
- xs.json only has oi.chg_pct at 1h/4h/24h/7d; the 72h-from-low figure for the R-RB-4 screen comes from
  tape.json players.oi, which covers only the ~25 tape tokens, not the full 60-token universe.
- account.json is still pending agent LIVE this cycle.

## Honesty
- direction_claim: false. A random entry ahead of any stated invalidation has roughly zero expected edge net of
  costs per this desk's own tests (R-XS-2/3 quintile spreads lose after costs both segments; R-RB-4 CIs include 0;
  R-AVOID-1 funding-contrarian/OI-divergence failed its test).
- Conflicts: R-AVOID-1, R-MAP-5, R-XS-2, R-XS-3, R-RB-1, R-RB-4, R-BUDGET-1, R-PD-4.
- What would change my mind: a v0 cell move out of R9 into R6 (the modal next cell per R-RB-3, ~80%) together
  with majors' funding normalizing above its 90d median; or SOPH/VVV reverting inside flag thresholds alongside a
  cell/sector confirmation rather than noise.
