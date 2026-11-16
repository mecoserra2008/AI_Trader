# SIGNALS 20261116T0407 (as_of 04:06 UTC Mon), round 3, model claude-opus-5-5, prompt desk-signals-v1
Inputs: a public Bybit scan of 89 perps (closed bars through the 04:00 1h/4h closes), shock scan `--top 200` (03:00 bar), `atx tape` 04:06, xvenue_tape 04:06, level_watch 04:03. Machine-readable: `signals.json`. No sizes.

## Regime
- **BTC: R9/R9 at ~64,930, inside the 63,836-65,270 box.**
  - The 04:00 4h bar closed at 65,079, 2.27σ below its EMA20 (66,909).
  - Funding is -0.0047%; OI is 70,915 on level_watch (rising); Coinbase premium -8.6 bp (it hit -17.3 at 04:01).
  - The tape shows PUSH DOWN 3/3 (target 64,672) with 66 of 90 sweeps on the sell side. The 03:57-04:00 buy sweeps of about $4.5M were absorbed.
  - Cross-venue 5m: Binance -10.3M, Bybit -2.9M, OKX -1.7M.
- **ETH: R9/R9 at 1820,** with 4h vol at p93 and funding -0.0036%.
  - Binance 5m is **-21.8M** and OKX -3.6M, but OKX 15m is +3.1M against Binance -15.7M.
  - Bid icebergs held at 1814.6, 1815.7 and 1818.3; longs were liquidated at 1816-1819 at 02:44 ($30k, small).
- **Crypto cells.** 1h: R9 28, R6 22, R3 6, R8 5, R5 2. 4h: R9 30, R8 15, R6 6, R5 6, R3 4.
  - There is no calm cell. R9 on 4h is slowly shrinking (35 to 30 since 23:36) as names move to R5/R6, as R-RB-3 expects.
- **KAVA is an abnormal market** and is excluded.
- **Networks:** none has a direction (R-FLD-3, R-XS-3). xs_v4 is not running yet.

## Checks at 04:00
- **Shock scan (03:00 bar): 200 scanned, 0 shocks.** No crypto shock means no H-S2 fade.
- **H-RB2 arm at the 04:00 4h close: no liquid crypto name crossed.** Every one closed further below its EMA20 than at 00:00:

| name | 04:00 close | EMA20 | gap |
|---|---|---|---|
| HYPE | 66.545 | 68.481 | -1.28σ |
| XRP | 0.9632 | 0.98462 | -1.11σ |
| TAO | 192.44 | 199.28 | -1.57σ |
| SUI | 0.7299 | 0.75299 | -1.64σ |
| LINK | 9.207 | 9.4655 | -1.67σ |
| SOL / ETH / BTC | 72.12 / 1826.79 / 65,079 | 74.869 / 1877.54 / 66,909 | |

  - **No crypto H-RB2 short can trigger at 08:00 or at 12:00.** The earliest path is an arm close at or above EMA20 at 08:00, then a cross below at 12:00 or later.
  - TradFi names did cross back below EMA20 at 04:00: NVDA 162.86 (EMA 163.53), XAU 3879.2 (3884.5), QQQ 676.56 (676.8). This is report only: out of scope, H-RB2 is untested on TradFi, and NVDA has its Tuesday blackout. AKE also triggered but has $4k of depth.
- **Sleeves: all alive.** 1h closes 00-03 UTC: ETH 1813.4 / 1804.5 / 1824.4 / 1826.8 (max 1834.35 vs the 1836.7 kill); BNB max 594.75 vs 598.2; ZEC max 899.4 vs 927.07.

## The ETH long question
**No TESTED mechanism supports an ETH long now.** The recorded evidence points the other way:
- **R-RB-1 (H-RB1):** buying flush rebounds lost -0.19 R in-sample and -0.34 R in the holdout.
- **A 4h close back above EMA20 (1877.5) is not a tested long entry.** It is H-RB2's arm condition, the setup for a later short. No registered hypothesis tests an EMA20-reclaim long. ETH's own H-RB2 window (flush 11-09 04:00) also ends after the 08:00 bar.
- **R-BOS-2/R-BOS-3:** with ETH 4h vol at p93 and cell R9, swing BoS longs lost (R9L -0.12 R in-sample, -0.14 R in the holdout).
- **R-RB-4 and R-AVOID-1:** negative funding with rising OI is not a squeeze signal.
- **Iceberg defence is descriptive only.** The tape labels are untested (config/tape.yaml), and R-FLOW-1 says flow extremes are not direction signals.

**The only conditions that would make an ETH long TESTED:**
1. **shock_regime fade (H-S2 supports in the holdout; R-PX-1, R-BUDGET-4).**
   - Trigger: an ETH 1h close at z <= -4 against the prior 168h σ (0.787%). That is about **1770 or lower** from a 1826.8 base, a -3.1% hour.
   - Conditions: vol state normal or high (true now), and the cell is not R2 (R-MAP-7).
   - Entry: PostOnly at the touch within 60 min of the close.
   - Stop: the shock low - 0.5σ1h, at least 0.5 × the 1h ATR. Target: the shock open.
   - **If a liquidation cluster prints in that bar, skip it.** R-FLOW-2 bars a fade for 60 min, and the edge's half-life is about 1.2 h.
   - Expected value: about -0.15 R driftless at 5:1 (P 0.167). The recorded R9 edge is small (grade C), so this is a learning trade unless the edge-adjusted E[R] reaches 0.10.
2. **R-BOS-4 calm-regime long (forward-test candidate only).** It needs ETH's 4h vol in the bottom tercile (now p93) or cell R1/R4/R7, then a 1h Donchian-20 CHoCH up. That is days away, and its payoff is right-tail only.

Any pattern-only ETH long also needs BTC and ETH cross-venue flow that does not contradict it. Right now Binance-led selling (BTC -10.3M, ETH -21.8M per 5m) **contradicts** it.

## Ranked candidates (nothing qualifies; no proposals.json)
| # | symbol | mechanism | label | side | status | net E[R] | flow | cluster |
|---|---|---|---|---|---|---|---|---|
| 1 | HYPEUSDT | H-RB2 | TESTED, inconclusive (R-RB-5) | short (conditional) | needs an 08:00 close >= 68.48, then a cross below (12:00 or later) | about -0.06 driftless; recorded IS -0.00, HO +0.42 | confirms a later short (Binance and OKX selling) | crypto_beta |
| 2 | XRPUSDT | H-RB2 | TESTED, inconclusive | short (conditional) | needs an 08:00 close >= 0.98462 | about -0.06 | confirms | crypto_beta |
| 3 | ETHUSDT | shock fade long | TESTED (H-S2) once triggered | long (conditional) | needs a 1h close <= ~1770 | -0.15 driftless | n/a until triggered | crypto_beta |
| 4 | ETH, BNB, ZEC | stale sleeve shorts | PATTERN-ONLY | short | alive; BTC is in the box | -0.13 to -0.25 | BTC flow confirms a short; ETH stop is under the ATR floor | crypto_beta |

## Crowd vs rule
- **CROWD-vs-RULE R-RB-1/R-RB-2: buying the defended 1815 ETH bid.** Icebergs refilled four times and funding is slightly negative, so dip buyers are present. The rule's evidence: the 7-day low usually comes later (median -8.7%, about 54 h after the flush), and rebound longs lost in both segments.
- **CROWD-vs-RULE R-BOS-2/R-BOS-3:** five failed pushes toward 65,271 this session, each faded by sell sweeps over $1M.
- **R-RB-4:** OI is rising with funding negative; the obvious read is a squeeze, but the evidence is a weaker next 24h.

## What changed since round 2 (00:14)
- **No arm:** every armed crypto name moved further below its EMA20 (HYPE -0.38σ to -1.28σ, XRP -0.67σ to -1.11σ). TradFi H-RB2 triggers printed on NVDA, XAU and QQQ (report only). Binance turned into a heavy seller at 04:05.

## Watch triggers, next hour (UTC)
- **05:00:10:** shock scan on the 04:00 bar. ETH <= ~1770 or BTC <= ~63,530 (z -4) is a fade candidate, unless a liquidation cluster prints in that bar.
- **BTC:** the tape's PUSH target is 64,672 and the next level is 64,494. Plan A: a 15m close below 63,836, then below 63,000 (retest short, stop 64,494, target 61,950).
- **Any LIQ_CASCADE line:** no fade for 60 min (R-FLOW-2).
- **Monday:** the 08:00 Deribit blackout runs 07:00-09:00.
