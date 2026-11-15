# TRADER 20261115T1723 (as_of 17:23 UTC)
## Regime
BTC / ETH / SOL / XRP all in v0 R9 (liquidation, dir down, vol high; xs.json). BTC 64,784 sits on its 24h POC 64,795
inside the 24h VA 63,338-65,462, -0.99% 24h after the 61,899 flush low (15m/1h/4h swing, unswept, -3.7A). Longs were
the liquidated side (BTC liq 24h L $60.0M / S $14.4M; ETH $30.8M / $3.9M). Funding is negative everywhere that matters
(pct_90d BTC 0.039, ETH 0.043, SOL 0.002, XRP 0.009), DVOL 44.4 (p99), F&G 10. No liq cluster live on the majors
(cluster_60m fired: false). Sunday: R-PX-9 window open until the CME reopen at 23:00 UTC. account.json is
`pending agent LIVE`: I assume nothing is open.
## Ideas
No proposal this cycle (no proposals.json). The game-theory checklist, by symbol:
- **BTC positioning**: OI +3.15% off its 72h low (players.oi.from_low_pct), funding last -0.0012% / predicted ann
  -3.2%: new shorts added into the drop (R-RB-4 pattern, also XRP +6.63%). Shorts pay little (ann -1.4% last), so
  holding them hurts little: there is no forced short cover.
- **BTC trapped**: longs from 68,281 (prior day open), 74,455 (7d POC) and 74,817 (week open) are mostly gone (they
  were the $60M flushed). Shorts who sold 61.9k-63.3k are offside +2-4.6%. Shorts added since the low sit near the
  POC, about 0A from price: not trapped yet.
- **BTC stops**: below, 64,235 (15m, -0.7A), 64,122 day open, 64,000, 63,734 (1h, -1.36A), VAL 63,338, then the
  big shelf 61,899 flush low + July 4h/1d lows 61,778 / 61,500 / 61,272 (all unswept) + est. long liq 61,705
  ($27.2M). Above, 65,000, VAH 65,462, 65,778 (15m+1h), 66,133 (1h) / 66,179 (15m), then 67,333 / 67,496. Est.
  short liq is far (74,042 / 75,536).
- **Forced flows**: funding 20:00 (22 perps) and 00:00 (60); CME reopen 23:00; Deribit 08:00 Nov 16 BTC $343.8M,
  P/C 1.99, max pain 68,000, top puts 60k / 67k / 65k (R-VOL-28: no pin edge); NYSE 14:30 Mon (ETF outflow streak);
  ARB unlock 13:00 Nov 16. Nothing is forced between now and 23:00 except two small funding prints.
- **BTC payoff** (validator functions): a rebound long at 64,799.1 with the stop beyond the shelf (61,650) and the
  target before the shorts' stops (65,700) gives R:R 0.28, p_tp 0.78. That is a bad shape, and R-RB-1 says it loses
  (-0.19 / -0.34 R). A failed-rebound short at 64,799.2 (stop 66,200, beyond 66,133/66,179; target 61,950, before
  the 61,899 stops) gives R:R 2.10, p_tp 0.32 driftless. R-RB-5 tests that family at about breakeven (-0.00 R in
  sample), so at no edge E[R] = -costs. R-RB-2 (low still ahead, BTC median -5.4% at 36h) is a path description
  with no net weekly drift, not an edge. Taking it now would be a weekend-direction trade (R-PX-9) on a
  regime-cell tilt (R-PX-6).
- **BTC next moves / pain**: longs must hold 64,122-64,235. Below VAL 63,338 they capitulate toward 61,899 /
  61,272. Shorts' pain starts above VAH 65,462 and 65,778, and they are squeezed through 66,133-66,179 toward
  67,333. Their low funding cost means they can wait.
- **ETH**: deleveraged, not crowded. OI -2.28% 24h / -7.2% 72h, only +0.12% off its low; funding p4. Price is
  0.5A under the POC 1,829. Stops at 1,801.85 / 1,800 / 1,793 / 1,772 (1h) / VAL 1,762 / flush low 1,709. Above:
  1,841-1,844, then 1,859. No trapped short and no forced flow. Max pain 1,880 (Nov 16).
- **SOL**: funding p0 (ann -21.3%), the only real time-pain on shorts, but OI is flat to down (-0.74% 24h), so
  shorts are not piling in. Price is pinned on the POC 71.11, with the 70.94 / 70.91 stops 0.3A below and
  72.00 / 72.45 / 73.23 above. A long at POC (stop 69.40, target 73.10) gives R:R 0.92: no asymmetry.
  Funding-contrarian is R-AVOID-1.
- **Equilibrium**: two-sided balance inside the 24h VA. The liquidated longs are out. New shorts are at the POC
  and paying almost nothing. Both stop pools sit 0.7-1.8A outside the VA, and only 2 small funding prints come
  before 23:00. Nobody is forced until CME prices the weekend, so the equilibrium holds until new information
  arrives. Neither side's asymmetry beats costs with a tested edge.
## No trade / watch
No trade. Why: no forced or trapped actor before 23:00. The only R:R >= 1.5 shape (BTC failed-rebound short) is
R-RB-5 breakeven, weekend direction (R-PX-9) and a regime tilt (R-PX-6, R-BUDGET-1). Long rebounds are R-RB-1.
Watch list (conditional, not proposals):
1. BTC failed rebound, after 23:00 only: an R-RB-5 trigger (4h close back below the 4h EMA20 after a >= 1.5 sigma
   rebound off 61,899) plus rejection of 65,462-65,778. Then a short at the touch, stop above 66,180, target
   61,950 (R:R 2.85 / p 0.26 from the VAH). This is still breakeven-tested, so only with a fresh snapshot and the
   orchestrator's call.
2. BTC sweep of 61,899 into the 61,778-61,272 shelf with a liq cluster: R-FLOW-2 says no fade for 60 min, and
   R-RB-1 says no long at the next 4h close. Watch for the R9 -> R6 hand-off (R-RB-3, ~80%): whipsaw, not trend.
3. CME reopen 23:00: first futures print versus the weekend range 61.9k-65.5k. A break of 63,338 or 65,462
   in the first hour is context for item 1, not an entry (R-PX-4: levels break slightly more often than they hold).
4. ARB unlock Nov 16 13:00: predicted funding ann -20.7%, OI 4h -5.32%, so it is pre-hedged. No tested edge
   (R-NEWS-7). Check the innovation-zone status first.
5. SOL / XRP: if OI rebuilds while funding stays at p0-p1, that is the R-RB-4 pattern, not a squeeze.
6. Deribit Nov 16 08:00 (max pain 68,000, P/C 1.99): context only (R-VOL-28, daily expiry untested).
## Data gaps
- No CME Friday close / weekend gap level in the snapshot, so the 23:00 gap can't be sized.
- 4h EMA20 and the R-RB-1 flush-event timestamp are not in the snapshot (needed for watch item 1).
- account.json pending agent LIVE. The tape is as of 17:23 and was not refreshed (no proposal, so not needed).
- ls_ratio stale since 11-07. The BTC book is truncated at 5.1 bp (depth figures are lower bounds). liq_map is
  a model.
## Honesty
- direction_claim: false. Base rate: driftless p_tp, E[R] = -costs. The desk has no validated edge (H-LT1 forward).
- Conflicts: R-RB-2 / R-RB-5 lean short-path while R-PX-9, R-PX-6, R-BUDGET-1 and R-NEWS-7 forbid acting on it. The
  max pain 68k and negative funding look squeeze-ish, but R-VOL-28, R-RB-4 and R-AVOID-1 say no.
- What would make me trade: a post-CME R-RB-5 trigger at the VAH with a stop beyond 66,179, or a trapped side that
  becomes forced (a short liq cluster through 65,778 is not a fade either: R-FLOW-2).
