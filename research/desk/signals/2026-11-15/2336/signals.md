# SIGNALS 20261115T2336 (as_of 23:36 UTC Sun), round 1, model claude-opus-5-5, prompt desk-signals-v1
Inputs: public Bybit scan of 89 perps (24h turnover >= 5M USD, no innovation zone, no SPY, plus TradFi context), using closed bars only (1h to 23:00, 4h to 20:00). Also: shock scan `--top 200`, lake liquidations / flow / DVOL, level_watch 23:29, xvenue_tape 23:36 and LATEST.md nn-20261115T23. Machine-readable output: `signals.json`. No sizes.

## Regime
- **BTC: R9 on 1h and 4h.** 4h vol is at p92 of 1y; DVOL 44.2 (p98).
- **Price:** 65,037. The CME-reopen poke reached 65,327 (23:00 cross of 65,271) and failed within 7 min. BTC is back **inside the 63,836-65,270 chop box**.
- **Funding:** predicted +0.0038%/8h, settles 00:00. OI is -2.3% over 24h on Bybit's 4h series; level_watch OI rose from 69,459 to 69,832 since 19:55.
- **Liquidations:** 4h long $260k / short $364k; 15m long $202k. **No cluster** (R-FLOW-2 is not fired for BTC, ETH, SOL, XRP or DOGE).
- **Players:** Bybit perp takers are selling (imbalance -0.15 over 15m, -0.17 over 60m) while Bybit spot takers are buying (+0.31 / +0.22). The Coinbase premium is -9.1 bp, so US spot is not bidding. Large and whale takers are mixed: BTC was net buying in the last 35 s; ETH had a $572k whale sell sweep at 1826.8-1827.0 into $1.2M of bids at 1824.6-1825.3.
- **ETH: R9/R9,** DVOL 57.3 (p93). ETH is +1.98% vs BTC over 24h and -2.0% over 7d. On cross-venue 5m flow, Binance is buying +909k and Coinbase is selling -909k.
- **Crypto cells.** 1h: R9 28, R6 22, R3 7, R8 5, R5 4. 4h: R9 35, R8 18, R6 5, R3 5, R7 1, R2 1, R5 1.
  - There is **no calm cell** in liquid crypto: only LIT has 4h vol at or below p33, and its depth is $12k. R-BOS-4 and the shock-continuation leg are off everywhere.
  - The R9 episode is about 2 days past the BTC flush (4h, 11-13 16:00 in this scan). Expect R6 whipsaw next (R-RB-3, R-PX-8).
- **Sectors** (median log return vs BTC, 4h / 24h / 7d, %): ai +1.28 / +4.04 / +9.22 (NEAR, TAO, DOT, ICP, VVV); exchange +0.29 / +4.42 / -4.37 (HYPE, UNI, BNB, ASTER); rwa +0.76 / +3.91 / +1.49 (LINK, ONDO, INJ, XLM, QNT); l2 -0.23 / +3.74 / -3.06; meme +0.34 / +2.97 / -4.40; l1 (17 names) +0.18 / +2.33 / +0.08; privacy (ZEC, XMR) -0.07 / +5.17 / -10.04.
  Alts are leading BTC on falling OI (24h: ETH -4.6%, XRP -4.5%, UNI -8.7%, AVAX -11.3%, ZEC -9.5%). This is short covering, not new longs.
- **TradFi** (report only: out of the plan's scope):
  - QQQ: new 24h high 683.83, +1.1% above its 4h EMA20 676.15. The NAS100 CFD is 28,097 against its 27,841 open.
  - NVDA: 165.8; longs pay a predicted +0.140%/8h at 00:00 (SOXL +0.236%, KORU +0.306%).
  - XAU: 3887.7, inside the 3881-3899.5 confluence. XAUUSD CFD 3881.97 against its 3880.28 open.
  - CL and BZ: R4 compression; the 23:00 bar z is -0.19. EURUSD 1.1022 is flat, so the dollar is flat.
- **Networks:**
  - v1 1h median +0.06%, 80% [-1.12, +1.17]; v2.1 1h 80% [-1.32, +1.36]; 24h ACI under-covers (8 alarms). No direction signal (R-FLD-3). xs.v3 4h: all relative medians within ±20 bp, bands span 0 (top ETH +0, bottom NEAR -20): no relative call (R-XS-3). xs_v4: not running yet.

## Ranked candidates (none qualifies; no proposals.json)

How to read the table: P is driftless, ln(e/s)/ln(t/s). Net E[R] = P·RR - (1-P) - 9.5 bp / risk. Every row is in the `crypto_beta` cluster (168h correlation to BTC 0.64-0.93), so at most one of them could ever be proposed.

| # | symbol | mechanism | label | signal age | side | entry zone | structural stop | target | R:R | P driftless / recorded | net E[R] | liq tier | flow confirms / contradicts | network |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | HYPEUSDT | H-RB2 failed-rebound short (flush 11-15 04:00, armed) | TESTED, inconclusive (R-RB-5) | not triggered | short (conditional) | touch at the first 4h close < EMA20 after a close >= 68.747 | rebound high 68.986 + 0.5σ4h (~69.89) | 2R (~66.53) | 2.00 | 0.333 / IS ~0.33, HO higher | -0.06 driftless; recorded IS -0.00, HO +0.42 | 0.1 bp, $627k | + Bybit/Binance 15m net sell (-41k / -143k); - whale buy prints +39k; ask wall 67.70 $313k | v3 4h bottom group (rel -13 bp); no call |
| 2 | XRPUSDT | H-RB2 (flush 11-14 16:00, armed) | TESTED, inconclusive | not triggered | short (conditional) | same, with arm close >= 0.98828 | ~1.0035 | 2R (~0.9585) | 2.00 | 0.333 | -0.06 / IS -0.00 | 1.0 bp, $2.9M | - taker net buy (+28.8k large), bid walls 0.9666-0.9691 ($1.67M); + Binance 5m -373k | no call |
| 3 | ZECUSDT | zec-s1 level short (2056 desk), stale sleeve short from 910.76 | PATTERN-ONLY | 22 h (shock), 3 h (idea) | short | 905-925 (touch 905.6 = bottom edge) | 935 | 857 | 1.73 | 0.367 | -0.03 | 0.9 bp, $1.15M | + takers selling (Bybit 5m -208k, Binance -382k; whale -64k, 3 × $31.7k sell sweeps 904.4-905.1); - CB premium +2.3 bp, bid 902.96 $92k | not in v3 |
| 4 | BNBUSDT | sleeve fade (02:00 +4.58σ) | PATTERN-ONLY (late) | 22 h | short | touch 593.8 | 598.2 (0.78 × ATR) | 571.16 | 5.27 | 0.160 | -0.13 | 0.2 bp, $652k | n/a (not tape-checked) | v3 BNB -1 bp |
| 5 | ETHUSDT | sleeve fade (02:00 +4.39σ) | PATTERN-ONLY (late) | 22 h | short | touch 1829.85 | 1836.7 = **0.30 × ATR** (fails the 0.5 ATR floor) | 1729.64 | 15.07 | 0.062 | -0.25 | 0.1 bp, $5.3M | + whale sell sweep; - Binance buying, $1.2M of bids | v3 top (+0 bp) |

Why nothing is proposed:
- **No fresh tested trigger.** The 22:00 bar has 0 shocks of |z| >= 4 across 200 perps (no errors).
- **No H-RB2 cross before 04:00** in liquid crypto. Every armed liquid name is 1.0-6.4% below its EMA20 (INIT crossed at 20:00, but its depth is $5k).
- **BTC sits inside the box,** which bars pattern-only trades. Rows 3-5 also fail the TESTED E[R] >= 0.10 gate.
- Learning slots left today: not counted here; that is the orchestrator's call.

## Crowd vs rule
- **CROWD-vs-RULE R-RB-1/R-RB-2: alts are being bought off the flush lows.** Alts are up 5-10% over 24h, beating BTC by a median of 2-4%. The crowd is short covering (OI 24h -2 to -11%, funding around 0 or negative), with spot bid on Bybit (+0.22 over 60m) but not on Coinbase (-9.1 bp).
  - Rule evidence (R-RB-1): buying the next 4h close after a flush returned -0.19 R in-sample [-0.36, -0.00] and -0.34 R in the holdout.
  - Rule evidence (R-RB-2): the 7-day low is usually still ahead, a median -8.7% about 54 h later (BTC: -5.4%, 36 h).
  - The edge, if any, is on the short side after the rebound fails (H-RB2), not in fading the bounce now.
- **CROWD-vs-RULE R-BOS-2/R-BOS-3: chasing the CME-reopen breakout above 65,271.** It failed in 7 min with perp takers net selling (-0.17 over 60m). Swing BoS in R9 lost -0.25 R (shorts) and -0.12 R (longs) in-sample.
- **CROWD-vs-RULE R-RB-4/R-AVOID-1: the negative-funding "squeeze" story** (SOL predicted -0.0056%, AVAX -0.0113%, ADA -0.0089%). Negative funding plus OI rebuilding after a flush gave a weaker next 24h (-0.14σ); funding-contrarian trading failed both segments.
- **TradFi (no rule; note only):** semis and NVDA longs pay +0.14 to +0.31%/8h going into NVDA results. Carry is untested (HC-1) and out of scope.

## What changed since ALPHA 22:00 / ALTS 20:56
- BTC made a CME-reopen poke to 65,327 and lost it. It is back in the box, so Plans A and B are both untriggered.
- **ZEC** slid from 910.97 to 905.4, now the bottom edge of the zec-s1 touch zone, with sellers on both venues.
- **Shock scan:** the 22:00 bar is clean across 200 names, and the scanner's earlier KeyErrors are gone.
- **TradFi:** QQQ made a new 24h high at the reopen.
- **ETH flush window disagrees.** This scan dates ETH's last flush to 11-09 04:00, which leaves 3 4h bars (alpha said 11), so ETH H-RB2 has effectively expired.
- **BTC funding:** predicted +0.0070% at 22:00, +0.0038% now.

## Watch triggers, next hour (UTC)
- **00:00:10, 1h close of the 23:00 CME-reopen bar.** Run `alt_shock_scan.py --top 200`.
  - Trigger: a liquid name (spread <= 3 bp, depth25 >= $100k, ATR1h <= 3%) with |z| >= 4.
  - Action: fade it (all crypto vol is normal or high; skip R2, R-MAP-7). Peg at the touch within 60 min; stop at the extreme + 0.5σ1h; target the shock open (R-PX-1, R-BUDGET-4: TESTED H-S2). Three or more names = one bet.
- **00:00, 4h close (H-RB2 arming, R-RB-5).**
  - HYPE close >= 68.747, XRP >= 0.98828, LINK >= 9.514, SUI >= 0.7574, TAO >= 200.54. Each sets "above EMA"; the short triggers at the FIRST later 4h close below EMA20 (04:00 earliest).
  - BTC needs >= 67,290 and is out of reach. QNT needs >= 157.56, but its depth is $15k.
- **Sleeve kills on a 1h close:** ETH > 1836.7, BNB > 598.2, ZEC > 927.07 (targets 1729.64 / 571.16 / 849.3). **ZEC W3:** a 1h close > 929.51 with OI 1h rising kills the short idea (no long, R-BOS-3). A 15m close < 876.71 means no chase.
- **BTC Plan A:** a 15m close < 63,836, then < 63,000; retest short with stop 64,494 and target 61,950. **Plan B:** a 15m close > 65,904 (24h high 66,133); retest long with stop 65,270 and target 67,376.
- **Funding settles at 00:00:** BTC +0.0038%, NVDA +0.140%, SOXL +0.236% (HC-1 data).
- **Liquidation cluster** on any name: no fade for 60 min (R-FLOW-2). **Monday:** no entries 07:00-09:00 (Deribit expiry) or 07:00-12:00 for commodities (R-TF-6).

## Data gaps
- **The `atx tape` labeller is not up** (no logs/tape.log), so footprints come from recent-trade snapshots and xvenue_tape.
- **Scan approximations.** Cells, z, EMA and flush state come from this agent's public-kline scan (1h vol window about 167 d vs 180 d; H-RB2 σ = EWMA sigma_next, not sigma_risk_next). These are not snapshot values.
- **Liquidations** exist only for BTC, ETH, SOL, XRP and DOGE. There is no DXY on cTrader; EURUSD is the proxy.
