# ALTS 20261115T2056 (as_of 20:56 UTC) - alt desk, contract agent id `xs`, prompt desk-alts-v1
## Regime
- Post-crash weekend. Crypto EW basket 7d -1534 bp. Majors are R9 high vol; the liquid movers (ZEC / HYPE / NEAR / AAVE) are R6 whipsaw on 1h; every 1h vol state is high. No calm cell exists, so the shock-continuation leg and R-BOS-4 are off everywhere.
- The basket's +136 bp 1h is mostly RARE (+5,682 bp, 1 of 46). Fast moves today are idiosyncratic and mostly illiquid.
- Scan: public Bybit 1h klines closed <= as_of (scratch script). z = log move / pre-move hourly sigma x sqrt(h); surge = turnover vs the 7d hourly median; cells and shock state computed with regimes.label + shock_regime. These are context, not snapshot values. EVAA / ARIA / JCT are innovation zone: excluded.

| sym | 1h / 4h / 24h % | z 1h / 4h / 24h | surge 1h / 24h | OI 24h | fund ann (pct) | 1h cell | driver (web = data) | spread / depth25 | $50-200 fit |
|---|---|---|---|---|---|---|---|---|---|
| RARE | +56.8 / +55.7 / +54.7 | +58 / +28 / +11 | 4546x / 113x | +114.5% | -3986% (p0) | R9 hi | no dated catalyst; old 50%-rally pieces; Binance monitoring tag (Aug, per CMC) | 5.8 bp / ~$1k | no |
| SOPH | -5.0 / -5.4 / -59.6 | -0.6 / -0.3 / -1.4 | 11x / 4.8x | +10.1% | -1314% (p10) | R6 hi | Sophon L2 mainnet shutdown Nov 30, withdrawals by Nov 25 | 8.4 bp / $4k/$2k | no |
| KAVA | -4.2 / -6.7 / +3.9 (7d -82.6) | -0.4 / -0.3 / +0.1 | 62x / 5.9x | +2.1% | -658% (p0) | R9 hi | -85 sigma crash 11-14; Bitfinex delisting (Jul); nothing dated today | 10 bp / ~$1-2k | no |
| VVV | +1.4 / +2.6 / +13.6 | +1.0 / +0.9 / +1.9 | 2.7x / 2.5x | +9.5% | +8.8% (p26) | R3 hi | Venice AI / private-inference narrative; no dated catalyst | 1.5 bp / $8k/$15k | thin ($50 only) |
| BARD | -4.1 / -1.2 / -18.6 | -1.2 / -0.2 / -1.1 | 5.8x / 0.5x | -16.5% | -45% (p21) | R9 hi | Lombard; no dated catalyst found | 4.0 bp / $3k/$7k | no |
| ZEC | +0.8 / +0.8 / +5.6 | +0.6 / +0.3 / +0.9 | 0.8x / 1.0x | -10.0% | +10.9% (p64) | R6 hi (4h R8) | unwinding from the 1,700 Sep high; NU7 + Grayscale-trust-outflow narrative; nothing dated today | 0.1 bp / $1.1M/$0.97M | yes |
| AAVE | +0.8 / +1.4 / +5.9 | +1.0 / +0.8 / +1.4 | 1.3x / 1.05x | +5.5% | +2.3% (p37) | R6 hi | DeFi fee-switch / buyback narrative (undated) | 0.9 bp / $684k | yes |
| UNI | +0.6 / +1.1 / +5.3 | +0.4 / +0.3 / +0.7 | 0.5x / 1.3x | -9.2% | -0.8% (p18) | R9 hi; 4h flush | fee-switch narrative (undated); a flush bounce | 1.9 bp / $296k/$173k | yes |
- Also moving, not ranked: BTW (+5.4% 1h, funding p99), INIT (OI +17.8%, funding -2228%), AKE (+4.8% 4h, 1h sigma 7.6%), ASTER (3.6x surge), HYPE (+0.9% 1h, 4h R9, OI 6.5% below its 72h high).
- Books for RARE / AAVE / UNI / VVV / BARD / SOPH / KAVA were fetched after as_of (21:05 UTC), as a liquidity check only.
## Ideas
### 1. ZECUSDT level short (conviction 0.30) - LEARNING TRADE, pattern-only -> proposals.json `zec-s1`
- Thesis: the +5.6% bounce was short covering (OI -10%). It stalled at the 928.34 / 929.51 equal highs and the 928.24 24h VAH, inside a 4h down-trend (R8, 4h closes below the EMA20 ~928). The shock_regime sleeve faded the +4.96 sigma 02:00 shock and is still short from 910.76 (target 849.3, invalid on a 1h close > 927.07). Entering 18h late is not the tested trade (R-BUDGET-4), and R-PX-1 gives R-PX-2 precedence until H-S3 is recorded.
- Evidence: OI 24h -10.02 (xs); 1h swing highs 929.51 / 928.34 unswept, VAH 928.24, untouched leg origin 849.3, ATR1h 20.16 (geometry); flow 5m -0.457 / 15m -0.275 (tape); 7d -3345 bp.
- Invalidation: 929.51 (a 15m close above it with OI rising), or the sleeve's 1h close > 927.07.
- Trade: pegged short at the touch (910.97), stop 935 (1.19 ATR, beyond the equal-high stop pool), target 857 (ahead of the VAL / 849.3 pool). R:R 2.35, driftless P 0.30, net E[R] -0.07 (learning gate >= -0.15). Submit only 23:30-00:30 UTC, with the touch inside 905-925.
- Rules: R-PX-1/2, R-BUDGET-4, R-BOS-3 (not a breakout), R-RB-1/5 (ZEC is not a flush by definition: 4h cell R8), R-MM-2 (a second short next to the BTC short is one doubled beta bet), R-PX-9, R-EXE-1, R-FLOW-2.
- Why this rank: the only liquid mover with two-sided structure and a sleeve position at the same price.
### 2. RAREUSDT volatility none (conviction 0.30) - tested signal, not executable
- Thesis: a fresh +49.6 sigma shock (19:00 bar) in R9 high vol, so shock_regime fades it (target 0.008447 = shock open; invalid on a 1h close > ~0.0169). The stop is ~17% away against a $2 cap and a $50 minimum notional; the book is ~$1k; shorts pay -1.8% per 4h (next funding at 00:00 is about -1.1%). Log it for H-S3; no trade.
- Evidence: rs_sigma.1h 12.76, OI 1h +115.21%, funding -3985.7% ann, basis -136.8 bp, turnover $14.4M (xs).
- Invalidation: 0.0169 (1h close).
### 3. UNIUSDT watch short (conviction 0.25) - tested mechanism, not triggered
- Thesis: R-RB flush on 4h (11-14 20:00: 48h return below its 1y 5th pct, OI -17% from its 72h high, R9). The rebound is armed but has stayed below the 4h EMA20 (~5.47). The H-RB2 short (R-RB-5: about breakeven, better than a random short) needs a 4h close above the EMA20 and then back below. No long (R-RB-1).
- Evidence: OI 24h -9.15, 7d -3027 bp, 24h +533 bp (xs).
- Invalidation: no cross-back within 42 x 4h of the flush.
### 4. VVVUSDT watch none (conviction 0.20)
- Thesis: AI leader (rs24h +2.19 sigma, OI +9.49%) pressing the 20.53 / 20.63 VAHs. A break is a swing BoS in high vol, which is negative in our tests (R-BOS-2/3), and the rank is not an entry (R-XS-3). Its 02:00 shock fade was already invalidated. Alert only on a new shock.
- Evidence: rs_sigma.24h 2.19, OI 9.49 (xs); 24h VAH 20.527, ATR1h 0.4969 (geometry).
- Invalidation: 19.865 (15m swing low).
### 5. SOPH / KAVA / BARD no_trade (conviction 0.10)
- Thesis: event or delisting-style crashes with $1-7k depth and 4-17 bp spreads. Any stop >= 0.5 ATR breaks the $2 cap at the $50 minimum notional. No trade.
- Evidence: SOPH 24h -5960 bp, spread 8.44 bp; KAVA basis -79.1 bp; BARD OI -16.52%.
- Invalidation: depth at 25 bp > $50k and ATR1h < 3%.
## No trade / watch
Edge verdict: RARE has a tested signal but can't be sized. UNI's tested H-RB2 is pending. ZEC's sleeve is still short, but a fresh entry is pattern-only. VVV / AAVE: fades failed, BoS negative. SOPH / KAVA / BARD: untradeable. R-BOS-4 (calm swing): no calm cell, n/a.
- W1 ZEC submit: 23:30-00:30 UTC AND touch in 905-925 AND no 1h close > 927.07 AND no ZEC liquidation cluster in the last 60 min -> submit `zec-s1`.
- W2 ZEC better entry: trades 922-929.5, then a 15m close < 920 with 15m imbalance < 0 -> re-peg the same short (stop 935, target 857; R:R >= 4).
- W3 ZEC kill: 1h close > 929.51 with OI 1h > 0 -> cancel or close `zec-s1`; no long (R-BOS-3; next 956.35 / 962.88).
- W4 ZEC lower: 15m close < 876.71 -> the target pool is near; no chase short, no long at the next close (R-RB-1); a liquidation cluster at 849-856 -> no fade for 60 min (R-FLOW-2).
- W5 shock scanner (the tested way to trade more): any alt with depth25 >= $100k, spread <= 3 bp and ATR1h <= 3% that closes a 1h bar with |z| >= 4 -> shock_regime candidate at the bar close, levels from the sleeve, executed within 1h (R-BUDGET-4). High / normal vol = fade; low vol = continue. >= 3 names in the same bar = one cluster.
- W6 UNI H-RB2: a 4h close > EMA20 (~5.47), then the first 4h close back below -> short candidate, stop above the rebound high (needs a free short slot).
- W7 HYPE: 4h R9 with OI 6.5% below its 72h high; a 48h return below its 1y 5th pct -> flush -> no long at the next 4h close; then watch H-RB2. Levels: 68.99 1h high, 66.15 24h VAH.
- W8 RARE: log the sleeve outcome (0.008447 target / 0.0169 invalid) and the 00:00 funding print; no trade.
- W9 VVV: |z1h| >= 4 and depth25 >= $50k -> fade candidate; a 1h close > 20.63 -> no long (R-BOS-3).
## Data gaps
- ZEC, VVV, RARE and the new names are not in the lake: cells, z, shock state, EMA20 and flush flags come from the public-kline scan. Adding them to ingestion is the user's call.
- No tape / geometry for RARE / AAVE / UNI / BARD (VVV / KAVA have geometry only). account.json is pending agent LIVE (the BTC short is taken from the orchestrator).
- Web: no dated catalyst today for RARE / KAVA / BARD / ZEC; several hits were undated or old.
## Honesty
- No direction claim. The ZEC trade is a learning trade at about -0.07 R expected after costs. What would make it right is untested: covering-led bounce, 4h down-trend, sellers in the tape now.
- Conflicts: R-BUDGET-4 (18h late), R-PX-2, R-MM-2 / R-PD-1 (doubles the BTC short), PLAN section 2 (no entries before 23:00 / 22:45-23:30).
- Would change my mind: ZEC 1h close > 929.51 with OI rising and 15m / 60m taker flow positive (real demand).
- Sources: cointelegraph.com/markets/zcash-falls-1k-grayscales-1m-ecosystem-commitment, cryptobriefing.com/sophon-mainnet-shutdown-withdrawal-deadline, coinmarketcap.com/cmc-ai/superrare/latest-updates, decrypt.co/378840, coinmarketcap.com/cmc-ai/kava/latest-updates.
