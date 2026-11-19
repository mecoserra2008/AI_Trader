# SIGNALS 20261119T2350 (as_of 23:45 UTC Thu), round 7: every Bybit linear USDT perp, model claude-opus-5-5
Inputs:
- **Universe:** 828 Bybit linear USDT perps (406 crypto, 110 innovation, 243 stocks, 61 ETFs, 4 commodities, 4 FX). I scanned 167: crypto with 24h turnover >= $5M, every TradFi perp >= $1M, plus PAXG, FX, COIN, CRCL and ANT. The rest are below every liquidity tier.
- **Bars:** closed 1h to 23:00, 4h to 20:00.
- **Shock scan `--top 300`:** 0 shocks on the 22:00 bar; the full scan found 0 with |z| >= 4 on the 23:00 bar.
- **Logs:** tape and xvenue 23:37, level_watch 23:33.
- **Note:** returns are log; "σ" = the move divided by the asset's own 1h σ (168h) × √h.

## Regime by asset class
**Crypto majors.**
- **BTC: R9/R9 at 61,774.** A lower low (61,388), below the 61,899 low of 11-15.
  - 4h -1.15% (-0.9σ), 24h -0.1%, 7d -13.9%. DVOL 45.6 (p99); ETH DVOL 61.0 (p100).
  - Funding +0.0028%; OI -1.7% over 24h and +2.3% over 72h, so **not a flush by our definition** (OI has not fallen 5%).
  - Long liquidation cascades at 23:08 ($188k) and 23:32 ($136k). **R-FLOW-2: no fade until 00:32.**
  - Tape: a $490k bid iceberg at 61,769 and a $405k ask iceberg at 61,897; 98 of 141 sweeps are sells.
  - Cross-venue 15m: Binance +61.6M against Bybit -9.7M, Bitstamp -8.3M and OKX -6.8M. The Coinbase premium is -13.4 bp.
- **ETH:** R9/R9, 1748, rel BTC +0.3% over 24h.
- **SOL and XRP:** R6/R6. **BNB:** R8/R9. **DOGE:** R8/R8. All are within ±1.1% of BTC over 24h.
- **Crypto cells.** 1h: R8 24, R6 15, R9 12, R5 12. 4h: R8 20, R6 16, R9 10, R5 9, R3 8, R7 7, R4 2. The liquidation phase has moved into **trend down (R8)**, R-RB-3's 18% branch.

**Alt sectors** (median log return vs BTC, 4h / 24h):

| sector | 4h | 24h | 4h cells | note |
|---|---|---|---|---|
| infra | +0.75 | +14.6 | | ICNT, illiquid |
| unmapped | +1.84 | +5.9 | R3 / R6 | small caps |
| other | +0.05 | +3.6 | | |
| ai | -0.15 | +1.2 | | |
| meme | +0.05 | +0.6 | | |
| rwa | +0.25 | +0.4 | | |
| l1 | -0.17 | +0.3 | R8 | |
| l2 | +0.22 | +0.0 | | |
| privacy | +0.15 | -0.8 | | |
| exchange | -0.03 | -1.2 | | |
| gaming | +2.12 | -3.1 | | |

No sector has a 24h move above 1.5σ of its own vol, except infra.

**TradFi equities and ETFs** (1h/4h cell, 24h log return and own-σ, rel BTC 24h):

| name | cells | 24h | rel BTC | other |
|---|---|---|---|---|
| NVDA | R9/R9 | -3.05% (-0.7σ) | -2.9 | 7d -13.3 |
| AMD | R6/R3 | -8.7% (-1.4σ) | | $18k depth: illiquid |
| MU | R6/R6 | -1.8% | | |
| SNDK | R9/R6 | -1.6% | | |
| SOXL | R6/R6 | -7.1% | | 7d -38.5 |
| SOXS | R6/R6 | +6.4% | | 7d +33.8 |
| QQQ | R9/R6 | -2.2% (-1.2σ) | | OI +54% |
| TQQQ | | -6.7% | | |
| TSLA | R6/R2 | -2.3% | | |
| MSTR | R9/R9 | -3.05% | | 7d -27.8 |
| COIN | R9/R8 | -1.6% | | |
| CRCL | R8/R8 | -2.8% | | |
| ANT | R6/R6 | -3.4% (-0.7σ) | | OI +673%: post-IPO listing |
| SK hynix | R5/R5 | +0.4% | | |
| SKHY | | -1.6% | | |
| Samsung | | -2.0% | | Samsung EM +5.6%, illiquid |
| KORU | | -3.4% | | 15.8 bp spread |
| EWY | | -1.0% | | |
| DRAM | | -1.5% | | |
| GOOGL | | -3.0% | | |
| META | | -3.9% (-2.0σ) | | |
| AAPL | | +0.1% | | |
| UVXY | | +15.6% | | |

- TradFi cells. 1h: R9 30, R6 28. 4h: R6 31, R9 10. 17 names have too short a history for a cell.

**Commodities.**

| name | cells | 4h | 24h | 7d | notes |
|---|---|---|---|---|---|
| XAU | R9/R9 | -1.3% (-2.2σ) | -2.6% (-1.8σ) | -6.6% | 3601.6; OI -11.1% |
| PAXG | | -2.66% (24h) | | | tracks XAU within 0.1% |
| XAUT | | -2.65% (24h) | | | tracks XAU within 0.1% |
| XAG | R9/R9 | -3.1% (-2.6σ) | -4.5% | -15.0% | **new flush at 12:00** |
| CL | R5/R7 | +0.3% | +1.3% | | 4h vol p14: **calm** |
| BZ | R5/R7 | +0.4% | +1.6% | | 4h vol p17: **calm** |

- There is no NG perp.

**FX:** EURUSD -0.03%, GBPUSD -0.53% (-1.4σ), USDJPY +0.23% over 24h. Spreads are 3.5-9.5 bp and depth is $8-17k (illiquid). USDBRL has too short a history.

## Tested triggers, live or in the next 12 h (any instrument)
- **Shock fades (H-S2):** none at 22:00 or 23:00. Next scans 00:00:10 to 11:00:10. Skip 07:00-09:00 (Deribit expiry, Fri 08:00, $2.2B, P/C 0.61 per the orchestrator).
- **H-RB2: the orchestrator's "8 alts at 16:00" are mostly re-crosses.** Only the FIRST cross after the arm is the tested trade.
  - First crosses at 16:00 Thu: HOOD (TradFi, liquid, now 4.7% below its 71.47 entry, so late; needs a plan extension), ADA (6.6 bp spread: fails) and BMNR ($8k depth).
  - **Trade book of first-cross H-RB2 trades:**

| status | trades |
|---|---|
| resolved: target | UNI (11-19 15:00, so `uni-rb2-s1` would have made +2R), QQQ, XAU |
| resolved: stop | SOL, MNT, TAO |
| open: HYPE | entry 68.20, last 66.23, target 61.83 |
| open: LINK (`link-rb2-s1`) | entry 9.215, last 8.968, target 8.744 |
| open: SUI | entry 0.729, last 0.7087 |
| open: DOGE | entry 0.0593, last 0.0581 |
| open: DOT | entry 0.732, last 0.713 |

  - **Next 12 h:** no liquid armed name sits above its EMA20 waiting for a cross. APT, AVAX (closest, +2.2% to its EMA20 of 7.136), BTC, MSTR and OP need a close at or above EMA20 first. The earliest new trigger is 04:00; 12:00 is likelier, since 08:00 falls in the Deribit blackout.
  - **XAG flushed at 12:00:** no long at the next 4h closes (R-RB-1 analog; untested on commodities). Its H-RB2 clock has started.
- **R-BOS-4 calm cells:** CL and BZ (4h vol p14/p17, cells R5/R7). Long-side calm is R1/R4/R7; R7 is a down-trend cell. This is a forward-test candidate only and needs a 1h Donchian-20 CHoCH up. TradFi: needs a plan extension.
- **R-TF after the US close:**
  - Friday 14:30 cash open: no fade of a > 1 overnight σ move in the first 2 h (R-TF-1).
  - Expect the previous close to be revisited only 22-28% of the time (R-TF-2).
  - No opening-range breakouts (R-TF-3); no first-hour to last-hour follow (R-TF-4); no night premium (R-TF-5).
- **Weekend from Fri 22:00 UTC:** no direction from weekend moves (R-PX-9). Expect 0.6-0.8× vol (R-VOL-1), but stops use the unshrunk σ (R-VOL-2). The Sun 22:00 to Mon 02:00 variance spike is expected (R-VOL-27).

## Cross-asset relative value (log %, 24h / 7d; descriptive only, R-XS-3)
- **BTC/QQQ** +2.10 / -7.4, and **BTC/XAU** +2.46 / -7.3. BTC fell the least today but is the 7-day laggard.
- **MSTR/BTC** -2.93 / -13.9. The equity proxy is underperforming its asset.
- **ANT/NVDA** -0.35 / +4.7. After the IPO pop, ANT is moving with NVDA.
- **Memory vs NVDA:** MU +1.25 / -2.5, SNDK +1.47 / -9.3, SK hynix +3.48 / +1.1, DRAM +1.55 / -0.7. Memory held up better than NVDA today, after the beat was sold.
- **PAXG/XAU** -0.08: no basis.

## Proposals: none
- There is no live tested trigger on any liquid instrument.
- BTC is inside R-FLOW-2's no-fade window to 00:32.
- HOOD is TradFi and 4.7% late.
- Pattern-only shorts after a lower low would be late continuation shorts (R-RB-5 about breakeven), with BTC cross-venue flow split (Binance +61.6M against everyone else selling).

## Crowd vs rule
- **CROWD-vs-RULE R-RB-1 / R-FLOW-2 / R-OPX-5: "capitulation, buy the lower low."**
  - The crowd: Binance BTC +61.6M over 15m, a $490k bid iceberg at 61,769, and a perp-led squeeze to 62.4k that fully reversed into two long cascades.
  - The rules: no fade within 60 min of a cluster (-1.85σ over 60m). BTC moves beyond the 90th percentile of the 3-day distribution continue (+0.15 to +0.31σ). Flush lows usually come later (median 54 h).
- **CROWD-vs-RULE R-TF-1: "buy the NVDA/AMD dip at Friday's open."** The holdout fade of a > 1σ overnight move lost 67 bp per trade (t -2.5). SOXS is +34% and UVXY +38% over 7d: the crowd is hedged, not capitulating.
- **CROWD-vs-RULE R-TF-3 / R-TF-4 (ANT):** chasing the IPO. Opening-range breakouts and first-hour follow-through both lost in the holdout.
- **Gold dip-buying after a -7.4% slide and the XAG flush** (R-RB-1 analog, untested on commodities): real yields and hike odds (Waller floated 50 bp) are against it, and XAU funding of +0.019% shows longs are still paying.
