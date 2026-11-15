# ALPHA 2026-11-15 22:00 UTC: tested edges live across Bybit USDT perps

Agent ALPHA, model claude-opus-5-5, prompt desk-alpha-v1. Data: public Bybit REST, closed bars only (1h to the 21:00 bar, 4h to the 16:00 bar), lake liquidations, Deribit DVOL. Universe: 125 perps scanned, 52 liquid (spread <= 3 bp, depth25 >= $50k both sides): 34 crypto, 18 TradFi. Machine-readable: `alpha.json`. No sizes anywhere.

**Bottom line: no tested edge is live in the plan scope right now, so there is no proposals.json** (CONTRACT section 5: no file when there is no trade). `zec-s1` stays the only crypto-beta short. The next places a tested edge can switch on are the 00:00 UTC 1h close (CME-reopen bar, shock scan) and the 00:00 / 04:00 4h closes (H-RB2).

## Regime
- **BTC: v0 R9 (liquidation) on both 1h and 4h; 4h vol in its top 1y tercile. ETH is the same.** Across the 34 liquid crypto names the 1h cells are R9 x25, R6 x4, R8 x4, R5 x1, with every vol state normal or high. No calm crypto cell exists. The R9 episode is about 1-2 days past the flush; expect R6 whipsaw next, not a trend (R-RB-3, R-PX-8).
- **Funding** is about 0. BTC +0.0070%/8h, up from -0.0012% at 16:00 (0.72 rank in its last 200 prints); ETH +0.0014%; SOL -0.0019%. Nothing in liquid crypto is extreme.
- **Open interest:** BTC -0.3% over 24h and -4.0% below its 72h high; ETH -2.0% and -4.6%. The deleveraging is ending quietly.
- **Liquidations** (lake, 4h to 21:57). BTC: shorts $486k vs longs $38k. ETH: shorts $233k vs longs $71k. The last 1h is small (BTC shorts $54k). These are short-covering bursts with no cluster (R-FLOW-2 not fired). Over 24h (snapshot 20:56), longs dominated: BTC $49.1M vs $13.7M.
- **DVOL:** BTC 44.3 (p98 of 90d, median 37.9), ETH 57.7 (p94). Implied vol is high while realised vol fades after the shock (R-VOL-3).
- **Basis:** BTC -4.8 bp, ETH -5.7 bp; perps sit at a small discount.
- **Who is doing what:** forced sellers are done; shorts are covering into the weekend bounce; nobody is adding leverage.
- **Calendar:** weekend rules apply until the 23:00 CME reopen (R-PX-9). Sun 22:00-Mon 02:00 is a high-variance window (R-VOL-27).

## Ranked live edges (none tested AND fresh AND in scope)
| # | symbol | mechanism | rule / hyp | tested? | signal age | side | entry zone | stop (structural) | target | R:R | P(tgt) driftless / with edge | net E[R] | liquidity | cluster |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | ETHUSDT | shock_regime fade sleeve, still short from the 02:00 +4.39 sigma shock (R9 high vol) | R-PX-1, R-BUDGET-4 / H-S2, H-S3 | mechanism tested (C, data-generated); **late entry = PATTERN-ONLY** | 20 h | short | touch 1818.25 | 1836.7 (shock high + 0.5 sigma_1h; 0.87x 1h ATR) | 1729.64 (shock open) | 4.80 | 0.168 / 0.168 (R(20h) ~ 0) | **-0.12** | 0.05 bp, $6.0M/$7.9M ($200 tier) | crypto_beta |
| 2 | BNBUSDT | the same sleeve (02:00 +4.58 sigma) | R-PX-1, R-BUDGET-4 | PATTERN-ONLY (stale) | 20 h | short | touch 593.04 | 598.2 (1.06x ATR) | 571.16 | 4.24 | 0.187 / 0.187 | **-0.13** | 0.17 bp, $1.05M/$0.78M | crypto_beta |
| 3 | H-RB2 basket: 19 armed crypto names | failed-rebound short after a flush | R-RB-5 / H-RB2 | **TESTED, about breakeven**; **NOT TRIGGERED** | arms at 00:00 at the earliest | short (conditional) | first 4h close back below EMA20 after a close >= EMA20 | rebound high + 0.5 sigma_4h (cap 4 sigma) | 2R | 2.0 | 0.33 / ~0.33-0.47 | IS -0.00 [-0.15,+0.15], HO +0.42 [-0.42,+1.21] | see W2 | crypto_beta |
| 4 | NVDA (SOXL, SKHY, SKHYNIX, MSTR, MU) | funding capture: NVDA predicted +0.265%/8h (above all 200 recent prints), settles 00:00; SOXL +0.223%, basis +35 bp | none (R-CARRY-1/2 are BTC/crypto) | **UNTESTED** (HC-1); TradFi, **needs plan extension** | live until 00:00 | short | - | - | - | - | - | carry 26.5 bp/settlement (~+0.18 R at a 1.5% stop, gross) | NVDA 0.61 bp, $500k | tradfi_equity |
| 5 | CL / BZ | shock_regime continuation leg (1h vol low, R4) if the CME-reopen 23:00 bar is a \|z\| >= 4 shock | R-PX-1 vs **R-TF-6** | mechanism tested, conflicts with R-TF-6; TradFi, **needs plan extension** | not fired | follow | - | - | - | - | - | R-TF-6 follow: net -32 bp | CL 1.14 bp, $692k | tradfi_commodity |

How to read the table:
- E[R] = P x RR - (1 - P) - cost/risk. The cost is 9.5 bp round trip: a 2 bp maker entry (R-EXE-1/2), plus a 5.5 bp taker exit and 2 bp slippage (config/fees.yaml).
- Rows 1-2 pass the learning gate (>= -0.15), but they are not proposed:
  - one bet per cluster (R-MM-2, R-PD-1): `zec-s1` (-0.07) already holds the crypto-beta short slot and is the better number;
  - ETH is 1% from its sleeve invalidation.
- ETH is inside the user's plan scope (BTC/ETH), and ZEC is strictly not. If the orchestrator voids `zec-s1` for scope, the ETH sleeve short is the substitute: same mechanism, pattern-only, E[R] -0.12.
- Recorded effect for the shock family (strategy map): R6 +3.1 / +3.2 x100 daily sigma per active symbol-day, R9 +1.5 / +0.6, all grade C. The edge(0) has t 1.05 and decays with a half-life of about 1.2 h (R-BUDGET-4). Even a fresh fade is therefore about 0 to +0.1 R net. It is the best mechanism we have, not a strong one.

## What is NOT alpha today, and why
- **There is no fresh shock.** At the 21:00 bar no liquid name has |z| >= 4 (the largest is XMR -1.07). ASTER -9.80 sigma (reversing its 20:00 +8.29 sigma) fails liquidity: 3.86 bp spread, $29k / $27k depth, ATR 5.8%. Every other shock is 19 h or older (R-BUDGET-4).
- **No flush rebound longs** on the 19 names inside a flush window. H-RB1 contradicts in both segments, and the low usually comes later (R-RB-1, R-RB-2).
- **No calm-vol plays in crypto.** No crypto 4h vol sits in its low tercile, so the swing BoS and shock-continuation legs are off. BoS loses in R9 and R6 (R-BOS-2, R-BOS-3, R-BOS-4).
- **No crypto carry or funding trade.** Funding is about 0; funding contrarian fails; carry has decayed; negative funding plus rebuilding OI is not a squeeze (R-AVOID-1, R-CARRY-1, R-CARRY-2, R-RB-4).
- **Rankings are not entries.** The 24h leaders and laggards say nothing on their own (R-XS-2, R-XS-3).
- **No weekend or Monday direction** (R-PX-9, R-PX-6).
- **No TradFi open fades, ORB or gap fills** (R-TF-1..4). The TF-1..4 ideas in tradfi_week.md are pattern-only.
- **Limit entry is a cost saving, not alpha:** +0.010 R per signal (R-EXE-1). It applies to everything.

## Watch list: triggers that switch on a tested edge in the next 24h
- **W1 shock scan, every hour at HH:00:10.** Run `scripts/alt_shock_scan.py` plus the majors. Its 22:00 run threw 10 KeyErrors: SOL, HYPE, HBAR, DOT and OP were covered here, while ADA, ONDO, PUMPFUN, SOPH and USELESS were not.
  - Trigger: a liquid crypto name (spread <= 3 bp, depth25 >= $50k, ATR1h <= 3%) closes a 1h bar with |z| >= 4.
  - Action (R-PX-1, R-BUDGET-4, R-MM-2):
    - fade it, since every crypto vol state is normal or high (follow only in low vol); skip cell R2 (R-MAP-7);
    - pegged entry at the touch within 60 min of the close;
    - stop at the shock extreme + 0.5 sigma_1h; target the shock open;
    - three or more names in the same bar count as one bet.
  - Timing:
    - the 22:00 bar closes 23:00, inside the 22:45-23:30 blackout, so its entry would be 23:30+ with most of the edge gone;
    - the **00:00 close (the CME-reopen bar) is the first clean one**;
    - Deribit 08:00 expiry: no entry 07:00-09:00 Monday.
- **W2 H-RB2, at the 4h closes 00:00, 04:00, 08:00, 12:00, 16:00 and 20:00.**
  - Arm: a 4h close >= the name's EMA20. Then the FIRST later 4h close below EMA20 is a short at the touch: stop above the rebound high + 0.5 sigma_4h, 2R target, 5-day time stop. Expect about 0 R: a learning trade only (R-RB-5).
  - The EMA20 moves each bar. Closest names (EMA20, gap from the 16:00 close in sigma_4h, % gap):
    - XRP 0.9883 (1.33 sigma, +2.8%);
    - LINK 9.514 (1.66, +3.0%);
    - ETH 1888.0 (1.81, +3.5%; only 11 bars of its window left);
    - SUI 0.7574 (2.16);
    - DOGE 0.06181 (2.34);
    - SOL 75.41 (2.66);
    - BTC 67,290 (2.70, +3.6%).
  - Also armed: SEI (spread now 5 bp: fails), ARB, TIA, TAO, UNI, DOT, VIRTUAL, OP, AVAX, PEPE, MNT, APT.
  - BTC's arm level is above the user's Plan B trigger (65,904): an H-RB2 short would come after a failed reclaim, not during it.
- **W3 sleeves.**
  - A 1h close above ETH 1836.7, BNB 598.2 or ZEC 927.07 kills that sleeve.
  - A touch of 1729.64 / 571.16 / 849.3 is its target.
  - Log both for H-S3. These are not entry signals.
- **W4 new flush.**
  - SHIB1000 arms at the 00:00 close if it is >= 0.003785.
  - Any new flush (r48 below its 1y p5, OI <= -5% vs the 72h max, R9) means no long at the next 4h close (R-RB-1); start the H-RB2 clock.
- **W5 TradFi at the reopen (report only).**
  - A CL/BZ 23:00 bar with |z| >= 4 in low 1h vol is a shock-continuation candidate, but R-TF-6 says the move fades by Monday 07-12Z.
  - The NVDA/SOXL funding prints at 00:00 are data for HC-1.
- **W6 liquidation cluster** on any traded name: no fade for 60 min (R-FLOW-2).

## Hypothesis candidates (for the orchestrator to register BEFORE any test)
- **HC-1:** On Bybit TradFi perps, when predicted funding is >= its 90d p95 within 2 h of settlement, a short held through the settlement earns the funding without an offsetting adverse drift, net of the 4.75 bp exit.
  - Test: all TradFi settlements since listing; signed return from T-2h to T+1h vs funding received; split llm_in_sample / clean_holdout.
- **HC-2:** For commodity perps in low 1h vol, the shock_regime continuation leg at the Sunday CME-reopen bar beats R-TF-6's no-continuation over 4h.
  - Test: H-TF5 events split by vol_state; forward only (data-generated).

## Data gaps and honesty
- Cells, z and flush state come from public klines (this agent's scratch scan, not the snapshot). They match the lake method but are not snapshot values. The 4h OI is Bybit's 4h series, so the flush timing can differ by one bar from the study (BTC 12:00 here vs 16:00 in the report).
- No public liquidation REST exists: liquidations are lake data for BTC, ETH, SOL, XRP and DOGE only.
- **Direction claim: none.** The base rate is that a random short in R9/R6 loses about 0.16 R (H-RB2 baseline). What would change this view: a fresh |z| >= 4 1h close on a liquid name at 00:00, or the first H-RB2 cross.
