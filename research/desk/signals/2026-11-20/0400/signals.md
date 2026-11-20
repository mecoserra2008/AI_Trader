# SIGNALS 20261120T0400 (as_of 04:16 UTC Fri), round 8: H-RB2 check at the 04:00 close (research only)
**Definition used:** the frozen post_cascade_study definition. Two deviations:
- σ is EWMA σ_next rather than σ_risk_next.
- The flush search covers the last 60 4h bars.

The scan covers crypto perps with 24h turnover >= $5M, excluding innovation zone, TradFi and SPY.

## Regime
**BTC: R9/R9.** The 04:00 4h close was 62,123; price is now 61,821.
- A 15m close below 61,899 printed at 04:15.
- 24h -1.7%; funding +0.0011%; OI -0.6%.
- Cross-venue 5m is net buying (Bybit +4.2M, Binance +6.1M, OKX +3.5M), while 15m Binance is -39.6M.

**Crypto cells.** 1h: R8 28, R9 21, R6 16. 4h: R8 23, R6 15, R9 13.

**TradFi, 24h** (report only):
- SOXL -16.0%, ANT -8.4%, XAG -7.4%, MSTR -7.1%, NVDA -5.1%, XAU -4.7%, QQQ -3.7%.
- CL +1.3% (R5/R4).

**Shocks on the 03:00 bar:** MEGA, BARD and PUFFER, all illiquid.

## H-RB2 result for the 00:00-04:00 bar: NO first trigger on any symbol
**HYPE.** Its single tested trigger was **11-17 16:00 at 68.197**. That trade is open: stop 71.62, target 61.83, last 65.95.
- HYPE closed above EMA20 at 08:00 and 12:00 Thu, then below it from 16:00 Thu through 04:00. Those are re-crosses, not a new H-RB2 trade.

## Armed liquid names waiting for a first cross

| symbol | flush (bars ago) | needs a 4h close >= EMA20 | gap now | bars left | spread / depth25 |
|---|---|---|---|---|---|
| BTCUSDT | 11-13 16:00 (38) | 63,670 | -2.54σ | 4 (expires after 20:00) | 0.02 bp / $6.9M |
| AVAXUSDT | 11-14 16:00 (32) | 7.100 | -1.95σ | 10 | 1.45 bp / $361k |
| OPUSDT | 11-15 04:00 (29) | 0.0772 | -3.04σ | 13 | 1.37 bp / $111k |

- The illiquid ones are APT, VIRTUAL, TIA and ICNT (already above its EMA20).
- **The earliest possible first trigger is the 12:00 close:** above EMA20 at 08:00, then below it at 12:00. The 08:00 close sits inside the Deribit blackout.

## Tested trade book (first crosses only)

| status | trades |
|---|---|
| open | HYPE 68.197, LINK 9.215, SUI 0.729, DOGE 0.0593, DOT 0.732 |
| hit target | UNI, QQQ, XAU |
| stopped | SOL, MNT, TAO |

## Correlation flag
The desk is **long BTC.** Any liquid alt short is a BTC-beta hedge: 168h correlation ETH 0.94, SOL 0.91, XRP 0.82, HYPE 0.76.
- A short would pull the book toward flat rather than add an independent bet (R-MM-2, R-PD-1).
- The plan forbids hedging a losing position with a correlated token.
