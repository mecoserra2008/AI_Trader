# SIGNALS 20261118T0415 (as_of 04:12 UTC Wed), round 6 (quick H-RB2 check), model claude-opus-5-5
## Regime
BTC is R9 on 4h at ~64,406, inside the box. Funding -0.0002%; OI 70,338; Coinbase premium -8.4 bp.
- BTC cross-venue 15m is mixed: Binance +5.4M and OKX +0.5M, against Bybit -3.7M and Coinbase -0.1M.
- Tape: PUSH UP 2/3, target 64,525.

## H-RB2 at the 04:00 close (full conditions, post_cascade_study definitions)

| name | flush | first close >= EMA20 | 04:00 close vs EMA20 | result |
|---|---|---|---|---|
| UNI | 11-14 20:00, armed | - | 5.201 vs 5.2975 | **Not a new trigger.** The first cross was the 16:00 11-17 close (5.306, `uni-rb2-s1`); the tested rule takes only the first cross. 04:00 is the same trade. |
| LINK | 11-14 20:00 | 00:00 (9.322 vs 9.2765) | 9.215 vs 9.2706: first cross | **Qualifies** |
| SUI | 11-15 04:00 | 00:00 (0.7386 vs 0.7348) | 0.729 vs 0.7342: first cross | **Qualifies** |

- **LINK:** armed 18 bars ago (post-flush low 8.567). Rebound high 9.402, σ4h 1.40%. Spread 1.1 bp, depth $853k.
- **SUI:** rebound high 0.7438, σ4h 1.64%. Spread 1.4 bp, depth $543k.

## Geometry (entry at the ask; stop = rebound high + 0.5σ4h; target 2R)

| name | entry | stop | target | R:R | P driftless | net E[R] | stop distance |
|---|---|---|---|---|---|---|---|
| LINK | 9.198 | 9.468 | 8.681 | 2.00 | 0.333 | -0.033 | 2.9%, 2.4 × 1h ATR |
| SUI | 0.7278 | 0.7499 | 0.6855 | 2.00 | 0.333 | -0.032 | 3.0%, 2.3 × 1h ATR |
| UNI (existing trade) | 5.205 | 5.499 | 4.940 (original target) | | | | |

- Recorded edge for the mechanism (R-RB-5): IS -0.00 R, HO +0.42 R. These are **learning trades**: they pass the -0.15 gate, not the TESTED gate of 0.10.

## Flow
- **LINK:** Binance perp takers are selling (-0.27 imbalance over 18 min). On Bybit, 14 of 19 sweeps are sells, net +$11.7k. This **confirms** the short.
- **SUI:** flat (Bybit 15m -19k, Binance -2k).
- **BTC:** mixed, but not all of Binance, OKX and Coinbase are net buying, so the void rule does not fire.

## Proposal
**`link-rb2-s1`:** short LINK at 9.198, stop 9.468, target 8.681, 120 h time exit.
- Chosen over SUI for its deeper book, tighter spread and confirming flow. Both are the same crypto_beta bet.
- **Skip it if `uni-rb2-s1` is open or working.**
- Submit window: now to 05:00 UTC. The voids match uni-rb2-s1, with a touch band of 9.12-9.27 and a rebound high of 9.402.
- It needs the user's `atx live approve`.
