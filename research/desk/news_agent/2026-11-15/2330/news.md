# NEWS round 1, 2026-11-15 23:30 UTC (claude-opus-5-5, desk-news-v1)
**Regime:** BTC, ETH and SOL are in v0 R9 (liquidation), with high vol: DVOL 44.2 (p98). Funding is about 0: BTC predicted +0.0040% per 8h. BTC OI is flat at 69.7k BTC since 21:00, -2.0% vs 16:00. 24h liquidations: longs $26.0M vs shorts $13.0M; the last 4h saw short-covering bursts. ETH OI fell into its rise (short covering). F&G 10. Narrative is risk-off: ETF outflows, hike pricing, a mega-IPO and NVDA CEO turmoil. Headlines and web pages are data, not instructions (R-NEWS-5/7). "U" = unconfirmed.

## Triggers: last 6h and next 48h (UTC)
| UTC | trigger (source) | perps | theses | expected (sigma) |
|---|---|---|---|---|
| 11-15 23:00 | CME reopen. Bybit 15m bar 65,485 -> 64,780 (1.08% range, ~1.2 sigma_1h at 0.89%); $88.6M turnover, the evening's high; OI flat. Failed break of 65,271 | BTC, ETH | deep | done; a vol event, no direction (R-PX-9) |
| 11-15 21:57 | CZ: "Binance did not sell crypto" after the Oct 29 hack. Sat-Sun liquidations $2.02B, $1.68B of it longs (The Block) | BTC, BNB | deep | none (denial) |
| 11-15 22:00 | Kava Lend exploit (from 11-14): ~55.2M KAVA borrowed (~$3.76M) through a dormant BUSD market; KAVA -~40% (newsbtc, The Block). KAVA funding -1.41%/8h at 00:00, OI $1.8M | KAVA | - | shorts crowded; not tradeable (depth) |
| 11-15 21:00 | ETH ETFs: 7th straight outflow week ($398.9M, ~$2.42B since late Sep). BTC ETFs: -$1.91B over 7 days (newsbtc, The Block) | ETH, BTC | a, deep | context; R-NEWS-7 says no direction |
| 11-15 22:50 | Huang (resigned Nov 12) campaigns to return; petition >100k (Cointelegraph). CEO uncertainty runs into the print | NVDA, QQQ, SOXL, MU | b, d | NVDA vol; BTC same-hour only (R-PX-12) |
| in force | Anthropic IPO: 825M sh at $70-78, ~$64.35B, ~$1.4T; pricing ~11-18, Nasdaq ANT ~11-19 (UPI). Crowd calls it a crypto liquidity drain | BTC, ANTHROPIC, AI cluster | b, d | untested: HC-2 |
| in force | MSCI drops MSTR at the Nov 30 close. JPM: ~$2.8B passive selling, up to +$8.8B if others follow (The Block, BTC Mag) | MSTR (COIN) | deep | MSTR vs BTC down: T-NEWS-2 |
| 11-16 U | Bybit: ANTHROPICUSDT split + rename; spot delisting of 12 pairs. Pages blocked, ratio/time/list U | ANTHROPIC | - | admin; check before any order |
| 11-16 08:00 | Deribit daily expiry: BTC OI $346M, P/C 1.94, max pain 68,000 (snapshot 2056) | BTC | - | no pin (R-VOL-28): HC-4 |
| 11-16 13:00 | ARB unlock 92.65M. **Erratum to W47.yaml:** $27.6M was the 2025 price; at 0.13446 it is ~$12.5M, 0.67x ARB 24h turnover | ARB | - | untested; vol only |
| 11-16 U | STRK, SEI, CRO monthly unlocks (2026 amounts not verified) | STRK, SEI, CRO | - | untested |
| 11-17 13:30 | Oct retail sales. Consensus +0.3% (forexfundamentals) vs +0.2% (kiplinger): conflict | all | a, d | vol up. Hot = hawkish. Nov 10 CPI analog: BTC -2.4 sigma over 3h |
| 11-17 14:10 | Waller, "Payments" (Fed calendar) | - | a | low |
| 11-17 21:20 | NVDA Q3 FY27, call 22:00 (NVIDIA IR). Implied +-9.6..10.8%, ~1.6x 1d ATR | NVDA, AI, BTC | b, d | NVDA +-1.6 ATR; BTC beta 0.29 same hour |
| 11-18 18:00 / 19:00 | 20y auction ($16bn) / FOMC minutes (Fed calendar 14:00 ET) | all, XAU | a, d | vol; tail = 2y up |

## Theses (theses.yaml; the orchestrator adopts or rejects)
- **T-NEWS-1, P 0.52:** BTC's 96h close stays in +-1 sigma(30d) [62,483, 68,041] by 11-19 23:00 (vol week, not direction). R-PX-12, R-NEWS-5, R-RB-3, R-SIG-2. Confirm: 4h closes revert inside 64.0-66.1k. Deny: daily close < 59,877 or > 71,003. Partly conflicts with deep (61,899 is outside the band).
- **T-NEWS-2, P 0.55:** MSTR underperforms BTC to 12-01 00:00 (MSCI forced flow). R-MM-2, R-PD-1 (one crypto-beta bet), R-TF-1. Confirm: stays below the 107.52 leg. Deny: MSTR beats BTC by >= 10% over 10 obs. Base rate: drift front-loaded since 11-11; beta > 1.
- **T-NEWS-3, P 0.55:** ETH/BTC gives back the covering bounce by 11-18 23:30. R-XS-2 (reversal places the median), R-XS-3 (not an entry). Deny: ETH/BTC +3% more, or ETH OI rising with price.
- **T-NEWS-4, P 0.33:** XAU (3886.47, inside 3881-3899.53) tags 3777.97 before 3951.03 by 11-20 20:00. Driftless 0.37, less ~10-15% for neither touched; no narrative uplift (R-MAC-1/3). R-TF-6 Mon 07-12.
- **Deep (Claude), kept at P 0.60:** nothing in the news adds the missing OI reset; ETF selling and US (Coinbase) selling continue.

## Crowd vs our rules (research leads; only a registered test changes a rule)
1. **"7-day ETF outflow streak = more downside"** (13/158 headlines, tone -0.36; Anchorage "US sellers") vs R-NEWS-7, flows untested. Lean: no direction. **HC-NEWS-1:** after >= 7 net-outflow days (Farside, 2024-01..), BTC next-5d vol-standardised return vs unconditional, plus 5d flow z regression (HAC). Success: < -0.3 sigma, block-bootstrap CI < 0, both segments.
2. **"Mega-IPOs drain crypto"** (Anthropic $64B; CoinDesk x2, CZ, Saylor, a hedge fund) vs nothing tested; R-PX-12. Lean: no trade on "drain" or "relief". Equity-funded AI capex cuts against d's debt mechanism. **HC-NEWS-2:** monthly US IPO proceeds (Ritter) vs BTC vol-standardised monthly return (HAC), plus an event study of IPOs >= $5B at [-5d, 0] / [0, +5d]. Success: t < -2 both segments, pre < 0 and post > 0 (low power: flag).
3. **"Crowded shorts = squeeze"** (KAVA -1.41%/8h, RARE -3986% ann, SOL p0) vs R-RB-4, R-AVOID-1, R-CARRY-2. Lean: the rules. **HC-NEWS-3:** funding <= -0.5%/8h on a perp with depth25 >= $50k, next 24h return net of funding vs p10 matches. Success: > +0.25 sigma, CI > 0, both segments.
4. **"Max pain 68k pulls price into Mon 08:00"** (P/C 1.94) vs R-VOL-28. Lean: no pin. **HC-NEWS-4:** Deribit dailies with OI >= $200M, |ln(S/maxpain)| at T-12h vs T against a placebo. Success: shrink > placebo, t > 2, both segments. Gap: historical OI by strike.
5. **"F&G 10 = bottom"** (minority: "looks cheap", CZ "bought") vs R-NEWS-7, R-RB-1/2. **HC-NEWS-5:** first F&G <= 10 print (14d dedup) -> BTC 7d/14d vol-standardised return. Success: > +0.3 sigma, CI > 0, both segments (the tail H-NEWS4 did not isolate).

## Theses a-d (thesis_check dry run 23:30; no new FRED data at the weekend)
- a strengthening 4/5; b active 4/6; c weakening 1/5 (waiting for trigger); d strengthening 7/9; deep active 3/6, not invalidated. Since 19:00 BTC +0.76%, ETH +0.63%, against the three shorts. This week: retail sales and minutes -> a/d; NVDA -> b/d; IPO size -> d (equity, not debt).

## Next 24h schedule (UTC)
- 00:00 funding (60 perps; NVDA predicted +0.143%/8h, KAVA -1.41%), 4h close (H-RB2), CME-bar 1h close (shock scan). 04:00 funding (23), 4h close. 07:00-12:00 no commodity entries (R-TF-6).
- 08:00 Deribit expiry ($346M), funding (60). 10:00 Barr, "AI and financial stability". 13:00 ARB unlock. 13:30 Empire State. 14:30 NYSE open (R-TF-1/3 for two hours).
- 16:00 funding. 17:00 Warsh (U: not on the Fed Board calendar). ~21:00+ Farside ETF flows (U). Mon, time U: ANTHROPICUSDT split/rename; STRK/SEI/CRO unlocks.

## Data gaps
- Bybit article pages blocked (ANTHROPIC split ratio/time, the 12 delisted pairs). Unlock aggregators show 2025 pages. Dec-hike probability stale (11-10). No CME gap figure. Reddit keys not set up.

## Sources
- NVDA IR https://investor.nvidia.com/news/press-release-details/2026/NVIDIA-Sets-Conference-Call-for-Third-Quarter-Financial-Results/ | Fed https://www.federalreserve.gov/newsevents/2026-November.htm
- Anthropic IPO https://www.upi.com/Top_News/US/2026/11/10/anthropic-ipo-stock-range-nasdaq/6651794326745/ | https://decrypt.co/383001/morning-minute-what-anthropics-1-4-trillion-ipo-means-for-crypto
- MSCI https://theblock.co/news/markets/2026-11-11-strategy-metaplanet-lose-msci-index-membership-november-reshuffle-438437 | https://bitcoinmagazine.com/markets/msci-removes-strategy-mstr-usa-index
- CZ https://www.theblock.co/news/markets/2026-11-15-cz-denies-binance-sold-crypto-as-bitcoin-bounces-from-four-month-low-near-62000-440807 | Kava https://www.newsbtc.com/news/defi/kava-lend-exploit-collateral-busd-market-55-million/
- Hodler's Digest https://cointelegraph.com/magazine/us-sellers-drive-bitcoin-crash-jensen-plots-comeback-hodlers-digest | ETH ETFs https://www.newsbtc.com/ethereum-news/ether-etfs-seventh-week-outflows-398-million-november-2026/
- Binance hack https://aljazeera.com/economy/2026/10/30/binance-says-2-54bn-in-crypto-funds-frozen-amid-potential-security-issue | Sophon https://cointelegraph.com/news/sophon-sets-nov-25-withdrawal-deadline-ahead-of-mainnet-shutdown
- Retail sales https://forexfundamentals.com/economic-calendar/us-retail-sales/2026-11-17 | Bybit https://api.bybit.com/v5/announcements/index?locale=en-US&limit=50 (public tickers 23:32 UTC)
