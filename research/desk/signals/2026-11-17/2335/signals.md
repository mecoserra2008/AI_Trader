# SIGNALS 20261117T2335 (as_of 23:33 UTC Tue), round 5, model claude-opus-5-5, prompt desk-signals-v1
Inputs:
- A public Bybit scan of 112 perps (closed bars to the 23:00 1h close and 20:00 4h close).
- A catch-up replay from 11-16 22:00 to 11-17 23:00 (71 crypto perps; the desk was stopped 08:15-20:15).
- Shock scan `--top 200` on the 22:00 bar.
- tape, xvenue and level_watch at 23:30, and LATEST.md nn-20261117T23.

Machine-readable: `signals.json`; **`proposals.json` holds 1 learning trade (validated: ok, 0 warnings)**. No sizes.

## Regime
- **BTC: R9/R9 at 64,939**, back inside the 63,836-65,270 box.
  - Funding +0.0043%, OI -1.7% over 24h, DVOL 43.7 (p97). Next-hour σ is 0.45%, 80% band ±0.6%.
  - Players: three short squeezes after the NVDA print, each met by sell sweeps ($3.35M, $6.14M). Spot absorption fired at 21:00 and 22:52 (Bybit spot +306-409 BTC over 60m).
  - Cross-venue 15m is still net selling: Binance -11.5M, OKX -4.4M, Coinbase -0.8M, Bybit -1.5M. The 5m window is mixed: Bybit +4.7M, Binance -3.7M.
  - The Coinbase premium has swung between -18.6 and +1.5 bp; it is -9.8 now. The tape shows a run of $130-195k buy sweeps at 23:29-23:30.
- **The R9 episode is ending.**
  - Crypto 1h cells: R6 24, R5 22, R9 13, R8 6, R3 5.
  - Crypto 4h cells: R8 17, R9 14, R6 14, R5 12, R7 7. The first R7 and R5 cells appeared, mostly in mid-caps.
  - ETH: R9/R9, OI -4.8%. SOL: R6/R6, +3.2% over 24h.
- **The NVDA read did not carry into crypto.**
  - TradFi semis over 4h (log): SOXL +12.6%, MU +5.7%, SNDK +5.7%, SKHYNIX +5.7%, NVDA +5.3%, TSM +5.2% (TSM was a +4.66σ shock at 22:00), AVGO +3.4%, QQQ +1.7%.
  - BTC +1.5% over 4h. The crypto AI sector relative to BTC: 4h median -0.88% (NEAR +3.5, TAO +1.4, DOT +0.3, ICP -1.0, VVV -0.9, PIEVERSE -2.2, AKE -5.5); 24h +2.6%.
  - Strongest relative sectors: infra +6.9% (ICNT +12.2) and meme +1.4%.
- **Networks:** v1 24h median +0.53%, 80% band [-3.18, +4.98] (no direction, R-FLD-3). xs.v3: no relative call (R-XS-3). xs_v4: not yet running.

## Catch-up (11-16 22:00 to 11-17 23:00)
- **BTC Plan A step 1 fired but step 2 never did.** BTC closed 15m bars below 63,836 from 16:30 to 19:45 (low 63,300), but never below 63,000. No close above 65,904, so Plan B is not triggered either.
- **Shocks: 14 in crypto, none in the NVDA hours.**
  - Fades: 6 hit the target, 8 were stopped.
  - The two liquid by today's turnover were BREV (z 13, 20.9 and 5.7; all 3 fades stopped, +42% 4h) and MMT (2 of 3 won). Their books at the time are unknown.
  - The 22:00 bar had only COOKIE ($436 of depth) and TSM (TradFi).
- **H-RB2 short triggers missed while the desk was stopped.** Liquid names: SOL, UNI and HYPE at the 16:00 close, and MNT at 20:00. INJ, ICP and ASTER also triggered but are illiquid.
  - **SOL is now stopped:** last 75.2 against a 74.88 stop, a -1R outcome.
  - **HYPE** (the round-4 watch): the 00:00 close of 69.74 held above EMA20. It triggered at 16:00 (68.197 < 68.543). It is now 69.75 (+2.3% vs the trigger), stop 71.69 not hit; the 20:00 close was 67.846 against EMA20 68.48.
- **R-BOS-4 calm long side:** 4h R7 in ENA, WLD, PUMPFUN, AERO, CRV and USELESS. ENA is liquid ($622k). It is a forward-test candidate only and needs a 1h Donchian-20 CHoCH up; none has printed.

## Ranked candidates
| # | symbol | mechanism | label | age | side | entry / stop / target | R:R | P driftless / recorded | net E[R] | liq | flow confirms / contradicts | network | cluster |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | UNIUSDT `uni-rb2-s1` | H-RB2 failed rebound. Flush 11-14 20:00; first 4h close below EMA20 at 16:00 (5.306); 20:00 close 5.238 still below | TESTED, inconclusive (R-RB-5): **learning trade** | 7.5 h late, price back at the trigger | short | touch 5.307 / 5.499 (rebound high 5.447 + 0.5σ4h) / 4.940 (2R) | 2.02 | 0.331 / HO point +0.42 R, IS -0.00 | **-0.03** (passes the >= -0.15 gate, not the TESTED gate) | 1.9 bp, $238k | confirms: BTC 15m Binance/OKX selling; $560k ask walls at 5.323-5.33. Contradicts: Bybit UNI taker +57k over 11 min (Binance neutral, -0.035) | v3 n/a | crypto_beta |
| 2 | HYPEUSDT | H-RB2 (16:00 trigger) | learning | 7.5 h | short | 69.75 / 71.69 / 61.72 | 4.47 | 0.183 | -0.035 | 0.14 bp, $1.08M | **contradicts:** 5m Binance +485k, Bybit +321k, Hyperliquid +141k | v3 bottom | crypto_beta |
| 3 | MNTUSDT | H-RB2 (20:00 trigger) | learning | 3.5 h | short | 0.3732 / 0.3948 (capped at 4σ) / 0.3199 | 2.73 | 0.268 | -0.017 | 2.7 bp, $438k | stop 5.6% = 4.3 × ATR (too wide for the $2 cap at $50 notional) | - | crypto_beta |
| 4 | SOLUSDT | H-RB2 | stopped | - | - | - | - | - | -1R | - | - | - | - |

- **Only one crypto_beta proposal is allowed, and UNI is the pick.** Its entry equals the trigger, so the tested geometry is unchanged. Its flow is not contradicting, and its regime is R5/R5, outside the BTC box.
- **UNI submit window and voids.**
  - Submit 23:45-00:30 UTC.
  - Void if: the touch is outside 5.24-5.36; a UNI liquidation cluster prints; BTC closes a 15m bar above 65,904; Binance, OKX and Coinbase 15m all turn net buying; or a 1h close above 5.447 before the fill.
  - **The 00:00 4h close does not void it:** the tested exits are stop, target and the 120 h time limit only.
- **UNI data gap:** it has no lake liquidations and no xvenue coverage; its flow comes from Bybit plus Binance aggTrades.

## Crowd vs rule
- **CROWD-vs-RULE R-PX-12: buying crypto on the NVDA beat.**
  - The crowd: three BTC short squeezes, spot absorption and buy sweeps; funding is normal, so it is spot- and short-cover-led.
  - The rule: QQQ first-hour moves co-move with BTC in the same hour only (beta 0.29); the 3h-ahead slope is -0.05 (t -0.63). Our history of 8 NVDA prints shows BTC up over 1-12 h in 7 of 8 but -0.29% at 24h. That history is unregistered, 8 events.
  - Today: semis rose 5-13% over 4h, BTC only 1.5%, and the crypto AI sector lagged BTC. Every squeeze met a larger seller.
- **CROWD-vs-RULE R-RB-1/R-RB-2:** "the R9 low is in" (spot absorption at 63.3-63.9k). The rule's evidence puts the 7-day low a median 54 h after the flush; BTC's flush was 11-13 16:00 and it has not undercut 61,899. The R9 to R5/R6/R7 handover matches R-RB-3; it is not a direction.
- **Hot retail sales (control +0.6%) raise the odds of a December hike,** which is negative for crypto beta, against the NVDA euphoria. This is theses (a)/(d) context only (R-MAC-3).

## Watch triggers, next 4 h (UTC)
- **00:00:10 to 03:00:10:** hourly shock scans. Fade a liquid crypto |z| >= 4 in a normal or high vol state, not cell R2, unless a liquidation cluster prints (H-S2). The first post-NVDA cash-session bar is 14:30 Wed, outside this window.
- **00:00 4h close (H-RB2):** a new first close below EMA20 after a close above it.
  - Watch SOL (its trade is stopped, but a new rebound high resets it), XRP 0.975, LINK, SUI and TAO.
  - Below the cap, only liquid names count (spread <= 3 bp, depth >= $100k).
- **BTC Plan B:** a 15m close above 65,904 with spot, the Coinbase premium and OI confirming; it also voids `uni-rb2-s1` if unfilled. **Plan A:** a 15m close below 63,000.
- **Wed:** FOMC minutes blackout 18:00-20:00. Any LIQ_CASCADE line: no fade for 60 min. KAVA stays abnormal and shadow-only (H-SCALP1 rejected).
