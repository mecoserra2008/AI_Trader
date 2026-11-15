# GEOMETRY 20261115T1723 (as_of 17:23 UTC)
## Regime
Sunday 17:23 UTC, post-liquidation; CME reopens 23:00 UTC. BTC, ETH and SOL are all in the R9 liquidation cell
(xs.json regime.cell) and have bounced back inside their 24h value areas after the flush.
- BTC 64,784: on its 24h POC (64,795, +0.0A), 65,000 round +0.3A above; unswept 15m low 64,235 (-0.7A) and day
  open 64,122 (-0.8A) rest just below, not yet released.
- ETH 1,813.84: 0.5A under its 24h POC (1,828.72), 1.0A under the VAH (1,841.10); unswept 15m low 1,801.85 (-0.4A)
  stacked on round 1,800 just beneath.
- SOL 71.27: pinned on its 24h POC (71.11, -0.1A) with round 71.00 and two unswept 15m lows (70.94, 70.91) in the
  same 0.3A band.
## Ideas
### 1. BTCUSDT level none (conviction 0.52)
- Thesis: BTC (64,784) sits almost exactly on its 24h POC (64,795, +0.0A) with the 65,000 round stacked +0.3A
  above. Below, an unswept 15m swing low at 64,235 (-0.7A) rests just above the day open 64,122 (-0.8A);
  traders who bought that 16:15Z low still have stops resting under it, not yet released.
- Evidence: nearest.above.0.price 64,794.79 / nearest.above.1.price 65,000 / nearest.below.0.price 64,234.6 /
  nearest.below.0.dist_atr_1h -0.71 (geometry.json)
- Invalidation: 64,234.6 - 1h close below the unswept 15m swing low releases those stops toward the day open /
  64,000 round
- Rules: R-PX-4, R-PX-10, R-EXEC-1, R-FLD-5
- Why this rank: tightest, cleanest two-sided level stack of the three majors.
### 2. ETHUSDT level none (conviction 0.50)
- Thesis: ETH (1,813.84) trades inside value, 0.5A under the POC (1,828.72), 1.0A under the VAH (1,841.10). The
  nearest pool is an unswept 15m low at 1,801.85 (-0.4A) stacked on round 1,800. The modeled long-liq cluster
  sits far lower at 1,664 (-8.6%), consistent with R-RB-2: the flush low is usually still ahead.
- Evidence: nearest.above.0.price 1,828.72 / nearest.below.0.price 1,801.85 / nearest.below.1.price 1,800 /
  liq_map.long_liq_below.0.price 1,664.13 (geometry.json)
- Invalidation: 1,801.85 - 1h close below sweeps that pool, no further live level until 1,772.02 (-1.5A)
- Rules: R-PX-4, R-EXEC-1, R-RB-2, R-AVOID-1
- Why this rank: same pattern as BTC but the pool is thinner and the liq cluster below is a model, not a level.
### 3. SOLUSDT level none (conviction 0.48)
- Thesis: SOL (71.27) is pinned on its 24h POC (71.11, -0.1A) with round 71.00 and two unswept 15m lows (70.94,
  70.91) in the same 0.3A band - four kinds in a tight box. Above, 72.00 round + 24h VAH 72.45 sit near +1.0A,
  then a 15m+1h swing-high confluence at 73.23 (+1.7A).
- Evidence: nearest.below.0.price 71.11 / nearest.below.1.price 71.00 / nearest.above.2.price 73.23 /
  nearest.above.2.dist_atr_1h 1.67 (geometry.json)
- Invalidation: 70.91 - 1h close below the stacked round/POC/swing-low zone opens air toward the 24h VAL
- Rules: R-PX-4, R-EXEC-1, R-MAP-6
- Why this rank: tightest confluence of the three majors, but SOL's book is thinner (tape.md depth25 ~3.5M/3.7M).
### 4. SOPHUSDT positioning none (conviction 0.45)
- Thesis: SOPH is flagged (rs24h -2.2sd, funding p5 ann -3393%, OI24h +26.4%, basis -54bp); tape.md reads OI
  rising into the drop (new shorts). Those shorts' stops sit at the overhead confluence of the 0.0013 round
  (+0.1A) and the 7d POC 0.001327 (+0.2A); a separate 3-way confluence sits -0.2A below.
- Evidence: nearest.above.0.price 0.0013 / nearest.above.1.price 0.0013273 / nearest.below.2.price 0.001228
  (geometry.json); oi.chg_pct.24h 26.37 / funding.ann_pct_last -3393.4 (xs.json)
- Invalidation: 0.0013 - 1h close above the round/7d-POC confluence stops out shorts opened into the drop
- Rules: R-EXEC-1, R-PX-4, R-VOL-3
- Why this rank: specific trapped-party story, but book is thin (bid $329/ask $91 at 10bp); structure, not size.
### 5. INITUSDT positioning none (conviction 0.42)
- Thesis: INIT is flagged (OI24h +23.1%, basis +27bp) and pays extreme funding (ann +1205%, p95) - crowded
  longs rebuilding after a drop. Price (0.1131) sits above a thin cushion at the 7d POC 0.1114 (-0.2A) and the
  0.11 round (-0.4A), below the reclaimed prior week low 0.1172 (+0.5A), now role-reversed resistance.
- Evidence: nearest.above.0.price 0.1172 / nearest.below.0.price 0.11139 / nearest.below.2.price 0.10787
  (geometry.json); funding.ann_pct_last 1204.9 / oi.chg_pct.24h 23.1 (xs.json)
- Invalidation: 0.1114 - 1h close below the POC/round cushion opens air to the prior day high 0.1079
- Rules: R-EXEC-1, R-PX-4, R-RB-2
- Why this rank: extreme flag, but the 60m move already round-tripped (-2.4%, tape.md) on thin liquidity.
## No trade / watch
- XRP, DOGE, AVAX, UNI, WLD are flagged (funding/OI) but their nearest stacks are single-kind and wider than
  0.5A; watch only, no idea raised this cycle.
## Data gaps
- tape.json flow is "not collected" for SOPHUSDT/INITUSDT - can't cross-check the OI/funding story with taker flow.
- research/live/LATEST.md: ls_ratio stale since 2026-11-07T20:00 (one-off backfill; no live updater).
- tape.md marks BTCUSDT depth25 with "*" (book did not reach 25bp): a lower bound, not the full picture.
- SOPHUSDT book is thin (l1_usd bid $329/ask $91): the cited confluence is a place, not a level with size behind it.
## Honesty
- direction_claim: false
- base_rate: R-PX-4 - an approached level breaks slightly more often than it holds (52-55%); flat coin-flip prior.
- conflicts: R-AVOID-1/R-MAP-6 (swing/pivot trend ideas have failed); R-RB-2 (ETH's modeled long-liq cluster sits
  -8.6%, well below the cited pool); SOPH/INIT theses lean on tape.json's players.pattern label, not an observed
  position book.
- what_would_change_my_mind: the next snapshot showing a cited swing flip to swept=true, or a liq cluster firing
  (R-FLOW-2) at one of these prices.
