# SIGNALS 20261116T0014 (as_of 00:14 UTC Mon), round 2, model claude-opus-5-5, prompt desk-signals-v1
Inputs: a public Bybit scan of 90 perps (closed bars through the 00:00 1h and 4h closes), shock scan `--top 200` (23:00 bar), `atx tape` (logs/tape.log 00:08), xvenue_tape 00:08, and level_watch 00:06. Machine-readable: `signals.json`. No sizes.

## Regime
- **BTC: R9 on both 1h and 4h.** Last price 65,071, inside the 63,836-65,270 box.
  - The 00:00 4h bar closed at 65,317, 2.14σ below its EMA20 of 67,102.
  - Two reclaims of 65,271 failed: up at 23:58 and down at 00:02, up at 00:04 and down at 00:06.
  - Funding is about 0 (+0.0001%).
- **Players.**
  - `atx tape`, 00:07: whale takers net -$7.03M in one minute; 00:08: PUSH DOWN 2/3, target 64,914. There is a $30k bid iceberg at 65,187 and no liquidations.
  - Cross-venue, 15m: Binance -7.6M, Coinbase -1.4M, OKX -1.2M, Bitstamp +1.4M. The Coinbase premium fell from -5.5 to -17.7 bp at 00:06, so US spot is selling.
- **ETH: R9/R9 at 1827.** Its 1h close of 1834.35 is below its sleeve kill.
  - On Bybit, 12 of 14 spoofs are pulled bid walls, a footprint that implies a push down (descriptive only).
  - Binance 5m is -2.0M against Bybit +649k.
- **Crypto cells.** 1h: R9 26, R6 23, R3 7, R8 5, R5 4. 4h: R9 34, R8 16, R6 5, R3 5, R5 3.
  - There is no calm cell, so R-BOS-4 and continuation legs are off.
  - Sectors are unchanged from round 1: alts lead BTC on 24h while OI falls (short covering).
- **KAVA: abnormal market** (exploit; Upbit/Coinbase trade 30-35% above Binance/Bybit). Excluded.
- **Networks:** v1/v2.1 1h bands are about ±1.2-1.4% (no direction, R-FLD-3). xs.v3 has no relative call (R-XS-3). xs_v4 is not yet running.

## Triggers checked at 00:00
- **Shock scan, 23:00 bar:** 200 scanned, 1 shock, 0 tradable. The shock is AXPUSDT (z +4.34), a TradFi stock with a 28 bp spread and $104 of ask depth. No crypto shock, so no H-S2 fade.
- **H-RB2 at the 00:00 4h close: no arm cross.** Every liquid armed crypto name closed below its EMA20.

| name | close | EMA20 = arm close needed at 04:00 | gap | depth25 |
|---|---|---|---|---|
| HYPE | 68.096 | 68.685 | -0.38σ | $619k |
| XRP | 0.9735 | 0.98688 | -0.67σ | $2.5M |
| TAO | 194.87 | 200.00 | -1.14σ | $176k |
| SUI | 0.737 | 0.7554 | -1.27σ | $437k |
| LINK | 9.287 | 9.4928 | -1.29σ | $594k |
| SOL | 72.72 | 75.158 | -1.76σ | |
| DOGE | 0.05984 | 0.061625 | | |
| ETH | 1834.35 | 1882.9 | | |
| BTC | 65,317 | 67,102 | | |

  - Already above EMA20 are only illiquid or TradFi names (AKE, MEGA, INIT; NVDA, XAU, QQQ). TradFi is out of scope and H-RB2 is untested on TradFi.
  - **The earliest H-RB2 short trigger is now the 08:00 close.** That needs a close >= EMA20 at 04:00 and then a close below it at 08:00, but 07:00-09:00 is the Deribit blackout, so in practice 12:00.
- **Sleeves (1h close at 00:00): all alive, none killed.**
  - ETH 1834.35 vs kill 1836.7 (2.35 away, 0.13%).
  - BNB 596.35 vs 598.2.
  - ZEC 915.67 vs 927.07.

## Ranked candidates (nothing qualifies; no proposals.json)
| # | symbol | mechanism | label | age | side | entry / stop / target | R:R | P driftless | net E[R] | liq | flow confirms / contradicts | cluster |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | HYPEUSDT | H-RB2 failed rebound | TESTED, inconclusive (R-RB-5) | not armed | short (conditional) | touch at the trigger close / rebound high 68.986 + 0.5σ4h / 2R | 2.0 | 0.333 | -0.06 (recorded IS -0.00, HO +0.42) | 0.15 bp, $619k | confirms: PUSH DOWN, 29/34 sweeps sell (00:07), Binance -302k and OKX -306k 5m. Contradicts: nothing | crypto_beta |
| 2 | XRPUSDT | H-RB2 | TESTED, inconclusive | not armed | short (conditional) | touch / ~1.0035 / 2R | 2.0 | 0.333 | -0.06 | 1.0 bp, $2.5M | confirms: PUSH DOWN, Bybit 5m -183k. Contradicts: Coinbase 15m +107k | crypto_beta |
| 3 | ETHUSDT | stale sleeve short (02:00 shock, 22 h) | PATTERN-ONLY | 22 h | short | touch 1827.5 / 1836.7 / 1729.64 | ~11 | ~0.09 | about -0.2; stop 0.40x ATR is under the 0.5 ATR floor | 0.05 bp, $5.9M | confirms: whale sell, bid-spoof pulls. Contradicts: Bybit cross-venue 15m +1.2M | crypto_beta |
| 4 | ZECUSDT | the user's long view (zec-s1 closed 23:42) | PATTERN-ONLY | n/a | long | no structural setup | n/a | n/a | n/a | 0.1 bp | mixed: Binance 5m +634k but 15m -233k; Bybit 12 bid-spoof pulls; Coinbase premium +4.2. BTC cross-venue contradicts (see below) | crypto_beta |

The orchestrator's new rule (cross-venue flow in BTC AND the token must not contradict a pattern-only entry): BTC's flow (Binance -7.6M and Coinbase -1.4M over 15m, Coinbase premium -17.7 bp) contradicts **every pattern-only long right now**, including ZEC. It would confirm shorts, but BTC is in the box, which bars pattern-only trades anyway.

## Crowd vs rule
- **CROWD-vs-RULE R-BOS-2/R-BOS-3:** three CME-week reclaim attempts of 65,271 (23:00, 23:45, 23:58/00:04), each faded by whale sell sweeps. Swing BoS in R9 lost -0.12 R (longs) in-sample and -0.14 R in the holdout. Funding is about 0, so the chasers are not paying up.
- **CROWD-vs-RULE R-RB-1, ZEC long after the bounce:** the user's bullish read. ZEC's 4h cell is R8, still below its EMA20 of 928.1 (the 4h R8 makes the flush rule not apply literally). The relevant evidence is R-BOS-3 (R6 1h, longs -0.11 R) if it is bought on a break of 929.51. A long after a 4h close back above 928 with OI rising and positive taker flow is the only version not contradicted, and it is still untested.
- **R-RB-1 / R-RB-4 (unchanged):** alt bounce on falling OI and negative funding is not a squeeze signal.

## What changed since round 1 (23:36)
- The user closed the ZEC short at about 23:42; there are no bot positions.
- BTC made two more failed reclaims; the Coinbase premium went from -9.1 to -17.7 bp.
- `atx tape` is live: whale delta -7.0M at 00:07.
- The 23:00 shock bar is clean in crypto.
- **00:00 4h: no H-RB2 arm.** HYPE narrowed from -1.02σ to -0.38σ below EMA20, and XRP from -1.38σ to -0.67σ.
- Sleeves alive; ETH came within 2.35 of its kill.
- KAVA was flagged as an abnormal market.

## Watch triggers, next hour (UTC)
- **01:00:10, shock scan on the 00:00 bar:** a liquid crypto |z| >= 4 means a fade candidate (H-S2), entered within 60 min. The first post-reopen bar has the most variance (R-VOL-27).
- **ETH sleeve kill:** a 1h close above 1836.7 at 01:00 ends the sleeve; log it for H-S3, it is not a long signal.
- **BTC Plan A:** a 15m close below 63,836, then below 63,000. **Plan B:** a 15m close above 65,904. The tape's PUSH target is 64,914; 64,494 is the next level down.
- **Any `LIQ_CASCADE` line in tape.log:** no fade for 60 min (R-FLOW-2).
- **04:00 4h close:** H-RB2 arms if it closes at or above the EMA20 levels in the table (HYPE 68.685 and XRP 0.98688 are closest).

## Data gaps
- Cells, EMA, flush state and z come from this agent's public scan, not a snapshot.
- Tape labels are descriptive, not tested.
- There is no DXY on cTrader.
