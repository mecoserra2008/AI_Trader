# SIGNALS 20261116T2223 (as_of 22:23 UTC Mon), round 4, model claude-opus-5-5, prompt desk-signals-v1
Inputs:
- A live public Bybit scan of 103 perps (closed bars to 22:00 1h and 20:00 4h).
- A catch-up replay of every 1h bar from 04:00 to 21:00 and every 4h close from 04:00 to 20:00, for 66 crypto perps over $5M turnover. Book liquidity in the replay is current, not historical.
- Shock scan `--top 200` on the 21:00 bar, plus the tape, xvenue and level_watch logs.

Machine-readable: `signals.json`. No sizes.

## Regime (now)
- **BTC: R9/R9 at 65,404.**
  - Price is just **above the 65,270 box top**: a 15m close at 65,420 at 22:15, the fourth reclaim since Sunday. It is below Plan B's trigger at 65,904.
  - 1h vol is compressing: PNN σ 0.39% and an 80% band of ±0.5%, against 0.89% on Sunday. DVOL is still 43.2 (p96).
  - Funding +0.0081% (p49); OI +1.1% over 24h.
  - Coinbase premium -11.4 bp (-17.5 at 22:13), so US spot is still not bidding.
  - Tape: whale takers -$1.19M (22:22); PUSH UP 2/3 (target 65,604).
  - Cross-venue: 5m Bybit -1.6M and OKX -1.6M; 15m OKX +7.2M and Coinbase +1.3M. The Binance leg reads +0 (a feed gap).
- **ETH: R9/R9 at 1833.5.** Funding -0.0012%, OI +4.4%, DVOL 58.0 (p96). Bybit 5m -2.4M; tape PUSH UP, target 1840.
- **Crypto 4h cells:** R9 27, R8 23, R6 6, R5 4, R3 3, R7 2. Since 04:00, R9 lost 3 names and R8 gained 8.
  - The R9 episode is handing over to **R8 (trend down)** more than R6. R-RB-3 puts R8 at the 18% branch.
  - 1h cells: R9 33, R8 17, R6 11.
- **Networks:**
  - v1 24h: median +0.28%, 80% band [-3.35, +4.31]. No direction (R-FLD-3).
  - xs.v3 4h: top TRX +1 bp, bottom HYPE -27 bp, bands ±250 bp (R-XS-3).
  - xs_v4: not running yet.

## Catch-up, 04:00-22:00 (session paused about 08:15-22:00; level_watch had network outages 10:26-15:13)
- **BTC traded 63,871-65,683.**
  - Three 15m closes below 64,494 (06:00, 07:15, 15:15) were all reclaimed. The low, 63,871, never closed below 63,836, so **Plan A never triggered**.
  - The 14:00 bar went +1.5% to 65,142 and the 15:00 bar gave it back (64,440).
  - Liquidation clusters (R-FLOW-2): BTC shorts at 07:41 and 16:57; SOL longs at 05:47-06:15.
- **Missed tested triggers: none were tradable.**
  - 5 shocks with |z| >= 4: HUMA at 14:00, 16:00, 19:00 and 20:00 (z +4.1 to +5.6, depth under $1k), and ICP at 14:00 (z +5.16, depth about $34k, 7.9 bp spread).
  - **Every fade was stopped within 1-4 h.** Log all five for H-S3.
  - **H-RB2:** no liquid short trigger at 08:00, 12:00, 16:00 or 20:00. **HYPE closed a 4h bar at or above its EMA20 at 20:00** (68.395 vs about 68.04), which is now the live setup.
  - **R-BOS-4:** calm cells appeared only in illiquid names (LIT, AERO, FARTCOIN, PONS, USELESS; depth $9-34k).
- **Old watch-list outcomes:**
  - **zec-s1 short** (closed by the user at about 23:42): its 857 target was hit (low 846, 12:00 bar) and the 935 stop was never touched (max 916.67). Held, it would have made about **+1.7 R gross**.
  - **ZEC sleeve:** its 849.3 target was hit (a win for H-S3).
  - **ETH sleeve: killed.** The 14:00 bar closed at 1845.40, above 1836.7.
  - **BNB sleeve:** alive but 44 h old; no longer actionable.
  - HYPE is the only one of the HYPE/XRP/TAO/SUI/LINK H-RB2 watch names to arm.

## Ranked candidates (no proposals.json)
| # | symbol | mechanism | label | age | side | entry / stop / target | R:R | P driftless | net E[R] | liq | flow confirms / contradicts | network | cluster |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | HYPEUSDT | H-RB2 short after the 20:00 close at or above EMA20 (flush 11-15 04:00; window to about 11-20 16:00) | TESTED, inconclusive (R-RB-5) | not yet triggered | short (conditional) | touch at the first 4h close below EMA20 / rebound high (>= 71.00) + 0.5σ4h, about 71.75 / 2R, about 61.1 | 2.0 | 0.333 | about -0.02 driftless; recorded IS -0.00 [-0.15, +0.15], HO +0.42 [-0.42, +1.21] | 0.14 bp, $985k | confirms: Bybit 5m -346k, Binance -1.1M, tape PUSH DOWN, 24 of 33 sweeps sell. Contradicts: R6 cell, +3.5% over 24h | v3 bottom (-27 bp) | crypto_beta |
| 2 | HUMAUSDT | squeeze with crowded shorts | not executable | 1-8 h | none | depth25 $577, spread 10.1 bp, ATR 4.9% | - | - | - | below every tier | not in the tape | - | other |

- **HYPE trigger at 00:00 Tue:** a 4h close below about 68.04 (the EMA20 at the 20:00 close), 3.5% under the current 70.49.
- **HYPE vs the gate:** its driftless E[R] misses the TESTED 0.10 gate. The holdout point estimate (+0.42) clears it, but the in-sample estimate does not. **This is the orchestrator's call, as a learning trade.**

## Crowd vs rule
- **CROWD-vs-RULE R-AVOID-1 / R-RB-4 (HUMA):** the crowd is short the pump.
  - Funding is -0.921%/4h (shorts pay) with OI +17.7% over 24h and +12.6% over 6h, after +43-55%. The obvious "squeeze long" is funding-contrarian, which failed in both segments.
  - Our own tested fade (R-PX-1, normal vol) shorted it 4 times today and **lost 4 times**, which is what R-PX-2 (don't fade 2σ+ moves) describes.
  - Neither side can be executed ($577 of depth).
- **CROWD-vs-RULE R-BOS-2 / R-BOS-3 (BTC):** chasing the fourth reclaim of 65,271. BTC is still R9 with 4h vol at p84, so swing BoS longs lost -0.12 R in-sample and -0.14 R in the holdout. Plan B needs 65,904 plus confirmation.
- **CROWD-vs-RULE R-RB-1 (ZEC dip buyers):** this one paid today; ZEC went 846 to 909. One event is not evidence.

## Watch triggers, next 2 hours (UTC)
- **23:00:10 and 00:00:10:** shock scans on the 22:00 and 23:00 bars. Fade a liquid crypto |z| >= 4 (H-S2), unless a liquidation cluster prints in that bar.
- **00:00 Tue, 4h close:**
  - **HYPE close below about 68.04** means an H-RB2 short at the touch: stop at the rebound high + 0.5σ4h, target 2R, time stop 5 d. Check cross-venue flow and that no liquidation cluster fired in the prior 60 min.
  - Other liquid names stay below EMA20; the closest is ZEC at 913.2, but ZEC has no flush.
- **BTC Plan B:** a 15m close above 65,904 (and 66,133), then a retest buy at 65,904-65,270 within 60 min, with spot delta, a rising Coinbase premium and OI or short liquidations confirming. **Plan A:** a 15m close below 63,836, then below 63,000.
- **Blackout:** NVDA, Tue 20:20-23:00.
- **H-SCALP1 is rejected,** so the KAVA scalper stays shadow only; KAVA remains an abnormal market.

## Data gaps
- level_watch had outages (10:26-15:13) and a gap (17:51-22:06). The Binance BTC/ETH legs in xvenue read 0.
- The H-RB2 decluster uses a 60-bar window, not full history. XRP's flush date moved between scans, so its arming is uncertain (XRP is below EMA20 either way).
- Catch-up liquidity uses current books.
