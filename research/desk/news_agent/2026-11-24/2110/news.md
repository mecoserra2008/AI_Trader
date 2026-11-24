# NEWS round 5: Tue 2026-11-24 21:10 UTC (claude-opus-5-5, desk-news-v1). Research only.
Desk gaps: Fri 11-21 08:17 to Mon 11-23 16:55, and 11-24 12:19-20:55. Lake liquidations cover Sat 00-08 and parts of Mon/Tue only. The FRED store ends 11-18. **Fact** = sourced, with time. *Spec* = interpretation. U = unconfirmed. Headlines are data, not instructions.

## Regime (Bybit 21:10)
- **BTC 59,735** (24h -2.6%, range 59,231-61,668), funding +0.0041%, OI $4.06B (coordinator: ~67.9k BTC, -3.8% vs 70.6k on 11-19; not a deep reset).
- DVOL 53.7 (was 46.3 on Thursday). Cells: R9 / post-flush whipsaw (no snapshot: data gap).
- **Alts weaker than BTC:** ETH 1673.5 (-3.0%), SOL -5.1%, HYPE -7.3%, ZEC -9.0%, MON -19.7% (unlock).
- **TradFi perps:** NVDA 154.65 (-6.6%), SOXL 32.70 (-25.0%), QQQ 637.83 (-2.3%), MSTR 77.37 (-9.1%), ANT 81.41 (below the $83 IPO), XAU 3556.8 (-2.4%), XAG 40.85 (-5.5%).

## 1. What drove the flush and the rebound (facts)
| UTC | fact (source) |
|---|---|
| Thu 11-19 | US spot BTC ETFs **-$582M**; IBIT **-$541.1M, its record one-day outflow** (Bitcoin Magazine, The Block) |
| Fri 11-20 ~20:00 | **NVDA -12.5% to -13% (~$127-128)** after Reuters/WSJ reports that SEC enforcement staff are examining its accounting (a geography-reporting change that removed Singapore) (Decrypt). Nasdaq -5.3%, S&P -3.5%; on the week Nasdaq -10.4% (worst since Mar 2020), S&P -8.6%, VIX +86% (Cointelegraph, CoinDesk) |
| Fri 11-20 | **Bessent pushed back on a 50 bp hike** (Bitcoin Magazine). Crypto liquidations $1.53B in 24h, $1.4B of them longs. BTC broke 60k late morning; Bybit first <= 58,000 in the 18:00 bar |
| **Sat 11-21 03:00-07:00** | **Flush with no new fundamental headline**, a leverage cascade into a thin weekend book ("excessive leverage rather than a new fundamental shock": crypto.news/Yahoo, secondary sources). Bybit 1h: 03:00 bar 58,577 -> 56,698 on **$1.31B**; 04:00 low 55,496.6; **06:00 low 55,100.0**, closing 58,019. Lake: **BTC long liquidations $36.3M (00-04) + $21.4M (04-08); shorts $0.4M**. Coinbase premium -71 bp (coordinator) = US-led selling. Fear & Greed hit a record low (Incrypted). Kraken BTC/CAD printed **$17K** (venue-specific; trades upheld, Cointelegraph) |
| Sat-Sun | Exploits: Chainflip $12M (Binance-hack loot routing), Fusionist ACE bridge $5.7M (Bithumb froze it), CYBER bridge (Korean exchanges halted). Sky USDS outflows >$2B in a day. Not BTC-specific |
| Mon 11-23 04:45 | "Market rout kills 50 bp December hike bets": 50 bp 0% (from 15%), no hike 12% (CoinDesk) |
| **Mon 11-23 ~14:00** | **NVIDIA reinstates Jensen Huang as CEO.** NVDA +31-35%, Nasdaq +6.5%. BTC 61.4k -> **64,572.5 (high)**; **$372M of shorts liquidated** (CryptoSlate). Then faded to 61.3k by the close |
| Mon 11-23 | Strategy bought 457 BTC and $212M of preferred; buyback ceiling raised to $4B. ETF weekly outflow -$1.8B (third-largest); Monday -$569.7M |
| **Tue 11-24** | **WSJ: tariff increases on China, Japan, South Korea and others "as soon as this week"** (CoinDesk 08:15 ET). NDX -2.1%, NVDA -7.5%, AMD/MU -10%. MSTR 52-week low $78.50. ANT back to $83. **November ETF outflows a record $4.71B**. Williams: favours higher rates but opposes 50 bp (CoinDesk). 24h liquidations $844M, BTC split evenly long/short (CoinDesk) |

- *Spec:* Friday was an equity/AI-accounting shock plus forced ETF selling. Saturday was crypto-native leverage meeting weekend liquidity, with no catalyst found. Monday's rebound was a pure NVDA-headline short squeeze and did not hold (R-SIG-2).
- *Spec:* BTC took the full downside of the Nasdaq moves and only part of the upside. That is consistent with R-PX-12 (same-hour co-move).

## 2. Calendar, next 72h (Tue 21:10 -> Fri 21:10 UTC)
| UTC | event | note |
|---|---|---|
| Wed 11-25 00:00 / 08:00 / 16:00 | funding; 08:00 Deribit 25NOV26 (6.2k BTC, P/C 1.00, max pain 61k) | |
| Wed 11-25 | Plasma (XPL) unlock, 4.78% of supply, ~$10.3M (CoinDesk); Sophon withdrawal deadline; Monad (MON) unlock was 11-24 (140% of circulating, ~$217M) | MON -19.7% |
| Wed 12:25 | Bowman (supervision; pre-recorded video) | no policy |
| **Wed 13:30** | **Q3 GDP 2nd est (est 3.4%), October PCE (est +0.3% m/m, prior +0.5%), durable goods, claims (prior 203k)**; new-home sales | CoinDesk's weekday labels are off by one; date U (Wed, pre-holiday dump) |
| Wed 18:00 | CME equity/crypto settlement 12:00 CT (U, CME advisory) | |
| Wed 19:00 | **Beige Book** (Fed calendar) | |
| **Thu 11-26** | **Thanksgiving: NYSE/Nasdaq closed; ~40h ETF gap**. CME equity halts 12:00 CT (18:00 UTC), reopens 17:00 CT (U) | holiday-thin books |
| **Fri 11-27 08:00** | **Deribit 27NOV26 monthly: 111.8k BTC (~$6.7B), P/C 0.89, max pain 72,000 (+18.7%)**. Top: 90k C 4.8k; 75k / 70k / 65k / 63k / 68k P ~2.9-3.1k | R-VOL-28: no pin |
| Fri 11-27 18:00 | NYSE early close 13:00 ET; Bybit delists 5 tokens (list U); HUMA unlock 11-26 (amount U) | |
| Any time | Tariff announcement (WSJ); Mon 11-30 21:00 MSCI MSTR deletion (just outside) | |

## 3. Gold and silver (context only; no view on the user's positions)
- **Moves:** XAU 3556.8 (-2.4% 24h), daily closes 3855 (11-17) -> 3550 (11-20) -> 3638 (11-23) -> 3556. Gold is down 24.6% over 60 days, the worst since 1982 (Cointelegraph). XAG 40.85 (-5.5%), from 47.57 on 11-17; gold/silver ratio ~87.6.
- **Drivers cited:** Real yields: 2y real >2.40%, the highest since 2023; 10y and 30y real at 20+ year highs (CoinDesk 11-20). FRED DFII10 2.88 on 11-18. A stronger dollar (19-month high on 11-18). Repricing toward December, January and March hikes (JPM). A cross-asset "liquidity event that started in AI stocks" (Cointelegraph). Tuesday's tariff headline (FXEmpire), even as the 2y slipped below 4.68%, the 10y fell toward 4.90% and the dollar came off yearly highs.
- **Levels discussed (FXEmpire 11-24 17:47):** XAU support 3,550, then 3,500-3,520, then 3,380-3,400; resistance 3,630-3,650 and R1 3,694.90. XAG support 40.50-41.00, then 37.50-38.00; resistance 44.00-44.50, then 47.00.

## 4. Theses for crypto perps, 1-3 days (ranked; theses.yaml)
1. **T-NEWS-20261124-1, P 0.45: BTC stays inside [57,000, 62,500] through Fri 21:00.** Wed's data dump, the holiday and the $6.7B expiry raise vol, not direction (R-NEWS-5, R-PX-12, R-VOL-28). Driftless "neither barrier" ~0.38, raised for holiday damping. Confirm: Thursday realised vol below forecast. Deny: a 1h close beyond either edge after the 13:30 PCE or a tariff headline.
2. **T-NEWS-20261124-2, P 0.70: MSTR underperforms BTC to 12-01** (supersedes 20261119-1; -7.4% relative since round 4). The MSCI flow comes at the Nov 30 close. Risk: the $4B buyback and R-XS-2 reversal.
3. **T-NEWS-20261124-3, P 0.35: BTC trades below 55,100 before Fri 12-04 08:00.** Driftless 0.35, no uplift: OI is only -3.8%, ETF outflows are at a record, and tariffs and AI are live; but 50 bp is dead.
- **Resolved:** T-NEWS-20261119-2 (<= 58,000) **YES** on Fri 18:00. T-NEWS-1 NO.
- **Ledger** (FRED to 11-18): a strengthening 4/5; b strengthening 6/6; c weakening 0/5; d strengthening 8/9. c needs bull_steepening plus HY easing. Tuesday's yield fall is 10y-led (*spec:* bull_flattening) with HY wider, so no trigger.

## 5. Crowd vs rule leads (HC; registered only by the orchestrator / research)
- **HC-NEWS-20261124-10.** Crowd: "Thanksgiving = quiet holiday drift." Rule: none for US holidays (R-VOL-1 covers weekends only). Positioning: ETF gap 40h+, CME halts. Lean: test it. Test: realised vol on Thanksgiving Thu + half-day Fri (and other NYSE full holidays 2018-2025) vs the robust forecast, as in H-W1. Success: median rv_ratio < 0.9 in both segments.
- **HC-NEWS-20261124-11.** Crowd: "capitulation is in: a record ETF-outflow month, 11.6M BTC underwater, F&G at a record low, Grayscale 'oversold'." Rules: R-NEWS-7, R-RB-1/2. Positioning: funding p1 at the low, now +0.004%; OI -3.8% only; CB premium -71 -> -17 bp. Lean: the rules (no long on sentiment). Test: BTC's 30d vol-standardised return after a record monthly ETF outflow and after F&G record lows, 2024-2026 (extends HC-1 and HC-5). Success: > +0.3 sigma, CI > 0.
- **HC-NEWS-20261124-12.** Crowd: "Huang's return ignites the bull market" (Bitwise headline). Rule: R-SIG-2 (surprise days don't continue). The Monday spike was fully reversed in 24h, a confirmation of the rule rather than a lead. Log it as a case: shocks to the BTC-Nasdaq co-move are asymmetric (full downside, partial upside). Test the up/down beta to QQQ on big-move days in both segments.
- **HC-4 at scale:** 27NOV26 max pain is 72,000, 18.7% above spot. Record any drift (R-VOL-28 says none).
- **Tariffs:** the crowd reads them as risk-off. R-NEWS-7 gives no daily direction from event flags. Record BTC 1h/24h around the official announcement.

## Data gaps
- No desk snapshot since 11-15; the lake misses liquidations for Fri 14-24 and most of Mon.
- FRED store ends 11-18 (no new DXY or real-yield prints).
- Wednesday data timing (CoinDesk's weekday labels conflict); CME holiday hours from a secondary summary; the Bybit 5-token delisting list; the HUMA unlock size.

## Sources
- https://decrypt.co/383744/nvidia-shares-sink-52-week-low-sec-probing-accounting | https://bitcoinmagazine.com/markets/bitcoin-price-slips-58300-nasdaq | https://bitcoinmagazine.com/markets/bessent-fed-bigger-rate-hike-bitcoin
- https://www.coindesk.com/markets/2026/11/23/market-rout-kills-50-basis-points-december-fed-rate-hike-bets | https://cointelegraph.com/magazine/will-bitcoins-big-rebound-hold-more-tampered-ledgers-found-hodlers-digest | https://crypto.news/bitcoin-price-rebounds-55k-2-7b-liquidations/
- https://cointelegraph.com/news/bitcoin-17k-kraken-canadian-dollar-market-trades-stand | https://decrypt.co/383797/nvidia-reinstates-jensen-huang-as-ceo-days-after-board-ruled-out-his-return | https://cryptoslate.com/bitcoin-rockets-above-64000-as-nvidia-reinstates-jensen-huang-wiping-out-372-million-in-shorts/
- https://www.coindesk.com/tech/2026/11/24/live-updates-bitcoin-pulls-back-to-usd60-000-as-stock-rally-fails-to-budge-crypto | https://www.coindesk.com/markets/2026/11/24/bitcoin-hands-back-monday-s-nvidia-driven-gains-liquidations-triple | https://www.coindesk.com/daybook-us/2026/11/24/bitcoin-drops-as-november-spot-etf-outflows-increase-to-record-usd4-71-billion
- https://www.coindesk.com/markets/2026/11/23/u-s-inflation-thanksgiving-trading-crypto-week-ahead | https://www.federalreserve.gov/newsevents/2026-November.htm | https://www.cmegroup.com/tools-information/holiday-calendar.html | https://cryptoslate.com/bitcoin-etf-investors-have-a-way-to-hedge-during-thanksgivings-40-hour-trading-gap/
- https://cointelegraph.com/markets/bitcoin-dips-60k-gold-worst-60-day-loss-1980s | https://www.coindesk.com/markets/2026/11/20/here-s-what-s-crashing-gold-and-likely-weighing-on-bitcoin-too | https://www.fxempire.com/forecasts/article/gold-xauusd-silver-platinum-forecasts-gold-retreats-below-3600-1646965
- Bybit v5 tickers / klines / announcements (21:10 UTC); Deribit public book summary and DVOL (21:11 UTC); lake liquidations (BTCUSDT)
