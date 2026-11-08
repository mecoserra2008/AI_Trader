# Bybit instrument catalogue — snapshot 2026-11-07 17:37 UTC

Source: Bybit v5 public REST (`/v5/market/instruments-info`, `/v5/market/tickers`), mainnet, snapshot time 2026-11-07 17:37 UTC (server ms 1794073047577). Note: Saturday snapshot — rolling 24h windows cover Fri 17:37 → Sat 17:37 UTC, so 24h turnover understates weekday activity; tiers use 7-day averages.
Cross-market base matching normalises multiplier prefixes/suffixes (1000PEPE→PEPE, SHIB1000→SHIB) and aliases PUMPFUN→PUMP.
Data file: `data/catalog/bybit_instruments_2026-11-07.parquet` (9464 rows incl. Closed/delisted). Turnover in USD: linear = `turnover24h` (quote USDT/USDC); inverse = `volume24h` (1-USD contracts); spot = `turnover24h` × quote-coin USD price; options = `turnover24h` as reported. Inverse OI USD = `openInterest`; option OI USD = contracts × underlyingPrice.

**API behaviour observed:** linear/inverse `instruments-info` without `status` returns only `Trading`; `PreLaunch` and `Closed` need explicit `status`. Spot ignores `status` (always 521 rows). Option without `baseCoin` returns **BTC only** (664); other underlyings require `baseCoin` — discovered by probing every live base coin: BTC, DOGE, ETH, HYPE, MNT, MU, NVDA, QQQ, SKHY, SNDK, SOL, SOXL, SPCX, TSLA, XAUT, XRP. Option tickers also returned 52 HYPE 7NOV26 contracts (expired 08:00 UTC today) absent from instruments-info; excluded.

## 1. Summary

| category | status | type | n |
|---|---|---|---|
| inverse | Closed | crypto | 50 |
| inverse | Trading | crypto | 26 |
| linear | Closed | crypto | 960 |
| linear | Closed | innovation | 93 |
| linear | PendingOpen | crypto | 5 |
| linear | PreLaunch | stock | 5 |
| linear | PreLaunch | crypto | 2 |
| linear | Trading | crypto | 514 |
| linear | Trading | stock | 238 |
| linear | Trading | innovation | 111 |
| linear | Trading | ETF | 59 |
| linear | Trading | commodity | 4 |
| linear | Trading | forex | 4 |
| option | PreLaunch | stock | 518 |
| option | PreLaunch | ETF | 252 |
| option | Trading | crypto | 3052 |
| option | Trading | stock | 2146 |
| option | Trading | ETF | 904 |
| spot | Trading | crypto | 507 |
| spot | Trading | xstocks | 11 |
| spot | Trading | adventure | 3 |

| category | live_instruments | turnover24h_usd | oi_usd |
|---|---|---|---|
| linear | 937 | $4.49B | $12.12B |
| spot | 521 | $1.20B | $0 |
| option | 6872 | $590.1M | $1.97B |
| inverse | 26 | $69.8M | $1.03B |

Live = status Trading or PreLaunch.

## 2. Linear

### 2.1 Crypto USDT perpetuals — top 60 by 24h turnover (of 519)

| rank | symbol | turnoverUsd | turnover7dAvgUsd | oiUsd | fundingRate | fundingInterval | launchTime | maxLeverage |
|---|---|---|---|---|---|---|---|---|
| 1 | BTCUSDT | $1.16B | $4.54B | $5.27B | 0.0100% | 8h | 2020-03-15 | 150.0 |
| 2 | ETHUSDT | $634.2M | $2.09B | $1.71B | 0.0100% | 8h | 2021-03-15 | 150.0 |
| 3 | ZECUSDT | $304.0M | $395.2M | $208.1M | 0.0100% | 8h | 2021-11-24 | 75.0 |
| 4 | XRPUSDT | $243.5M | $281.7M | $279.0M | -0.0027% | 8h | 2021-05-13 | 100.0 |
| 5 | SOLUSDT | $163.1M | $473.9M | $549.8M | 0.0084% | 8h | 2021-10-15 | 100.0 |
| 6 | AKEUSDT | $155.5M | $72.0M | $10.8M | 0.0072% | 4h | 2025-09-26 | 25.0 |
| 7 | NEARUSDT | $72.8M | $199.5M | $124.9M | 0.0000% | 8h | 2021-10-11 | 75.0 |
| 8 | HYPEUSDT | $72.1M | $271.8M | $252.5M | 0.0070% | 8h | 2024-12-04 | 75.0 |
| 9 | 1000PEPEUSDT | $56.7M | $104.1M | $56.0M | -0.0210% | 8h | 2023-05-03 | 50.0 |
| 10 | UAIUSDT | $44.4M | $22.6M | $2.9M | -0.0993% | 4h | 2025-11-17 | 20.0 |
| 11 | ICNTUSDT | $41.8M | $14.3M | $5.9M | 0.0050% | 4h | 2025-07-03 | 20.0 |
| 12 | ENAUSDT | $37.8M | $84.8M | $56.6M | 0.0050% | 4h | 2024-04-02 | 75.0 |
| 13 | LITUSDT | $28.5M | $36.3M | $61.5M | 0.0050% | 4h | 2025-12-30 | 75.0 |
| 14 | PUMPFUNUSDT | $25.5M | $30.8M | $31.7M | 0.0041% | 4h | 2025-07-10 | 75.0 |
| 15 | DOGEUSDT | $25.0M | $72.9M | $108.6M | -0.0065% | 8h | 2021-06-02 | 75.0 |
| 16 | SUIUSDT | $23.7M | $54.9M | $65.8M | -0.0102% | 8h | 2023-05-02 | 75.0 |
| 17 | WLDUSDT | $23.1M | $46.7M | $38.7M | -0.0082% | 8h | 2023-07-24 | 75.0 |
| 18 | JCTUSDT | $21.2M | $32.2M | $5.1M | 0.0033% | 4h | 2025-11-17 | 20.0 |
| 19 | BIRBUSDT | $20.7M | $17.2M | $4.1M | 0.0050% | 4h | 2026-01-28 | 25.0 |
| 20 | ADAUSDT | $18.6M | $40.4M | $43.5M | -0.0086% | 8h | 2021-01-15 | 75.0 |
| 21 | TRUMPUSDT | $18.6M | $54.1M | $19.2M | 0.0050% | 4h | 2025-01-18 | 75.0 |
| 22 | INJUSDT | $17.7M | $22.9M | $17.6M | 0.0100% | 8h | 2022-08-17 | 75.0 |
| 23 | UNIUSDT | $16.7M | $46.3M | $58.1M | -0.0148% | 8h | 2021-11-30 | 75.0 |
| 24 | LYNUSDT | $16.4M | $10.9M | $2.5M | 0.0044% | 4h | 2025-10-06 | 10.0 |
| 25 | CYBERUSDT | $15.9M | $6.1M | $2.6M | -0.0408% | 4h | 2023-08-15 | 25.0 |
| 26 | DGAIUSDT | $14.9M | $14.8M | $1.6M | 0.0040% | 4h | 2026-09-04 | 20.0 |
| 27 | LINKUSDT | $14.8M | $38.1M | $62.0M | -0.0041% | 8h | 2018-01-01 | 75.0 |
| 28 | ARIAUSDT | $14.8M | $28.9M | $6.9M | 0.1522% | 4h | 2025-09-03 | 25.0 |
| 29 | XMRUSDT | $14.5M | $32.3M | $44.9M | 0.0100% | 8h | 2021-12-22 | 75.0 |
| 30 | ACTUSDT | $14.3M | $1.2M | $1.6M | -0.0640% | 4h | 2024-11-11 | 25.0 |
| 31 | AAVEUSDT | $14.2M | $37.6M | $48.6M | -0.0067% | 8h | 2021-05-13 | 75.0 |
| 32 | CAPUSDT | $12.5M | $12.2M | $10.0M | 0.0050% | 4h | 2026-06-29 | 10.0 |
| 33 | STRKUSDT | $12.4M | $16.8M | $14.0M | 0.0050% | 4h | 2024-02-20 | 50.0 |
| 34 | BCHUSDT | $12.1M | $14.8M | $32.1M | -0.0036% | 8h | 2018-01-01 | 75.0 |
| 35 | ESPORTSUSDT | $11.5M | $38.5M | $3.5M | 0.0050% | 4h | 2025-07-29 | 10.0 |
| 36 | ARBUSDT | $11.0M | $18.6M | $25.6M | -0.0048% | 8h | 2023-03-23 | 75.0 |
| 37 | TIAUSDT | $10.8M | $13.5M | $9.6M | -0.0011% | 4h | 2023-10-31 | 50.0 |
| 38 | BNBUSDT | $10.2M | $37.7M | $92.6M | -0.0174% | 8h | 2021-06-29 | 75.0 |
| 39 | ONDOUSDT | $10.0M | $27.3M | $42.5M | 0.0019% | 4h | 2024-01-22 | 75.0 |
| 40 | XLMUSDT | $9.4M | $17.2M | $30.0M | 0.0026% | 8h | 2021-08-12 | 75.0 |
| 41 | QNTUSDT | $9.4M | $12.2M | $16.0M | -0.0022% | 8h | 2023-06-19 | 75.0 |
| 42 | HBARUSDT | $9.3M | $11.6M | $25.2M | -0.0027% | 8h | 2021-10-25 | 75.0 |
| 43 | AVAXUSDT | $8.9M | $24.9M | $56.0M | 0.0100% | 8h | 2021-09-15 | 75.0 |
| 44 | BTWUSDT | $8.7M | $11.3M | $6.2M | 0.0050% | 4h | 2026-06-05 | 10.0 |
| 45 | CTUSDT | $8.7M | $3.0M | $2.6M | 0.0050% | 4h | 2026-10-01 | 20.0 |
| 46 | TAOUSDT | $8.6M | $17.9M | $31.9M | 0.0050% | 4h | 2024-01-30 | 75.0 |
| 47 | LTCUSDT | $8.0M | $15.4M | $57.9M | 0.0049% | 8h | 2018-01-01 | 75.0 |
| 48 | XAUTUSDT | $7.7M | $32.8M | $149.4M | -0.0053% | 4h | 2025-04-03 | 75.0 |
| 49 | PONSUSDT | $7.1M | $10.5M | $9.9M | 0.0050% | 4h | 2026-08-31 | 20.0 |
| 50 | XPLUSDT | $6.8M | $25.2M | $21.7M | 0.0050% | 4h | 2025-08-22 | 75.0 |
| 51 | ZROUSDT | $6.6M | $22.3M | $26.5M | 0.0050% | 4h | 2024-06-20 | 50.0 |
| 52 | CARVUSDT | $6.4M | $461k | $1.1M | -0.0541% | 4h | 2024-10-16 | 20.0 |
| 53 | LSKUSDT | $6.2M | $29.2M | $3.6M | -0.0975% | 4h | 2024-01-02 | 25.0 |
| 54 | STXUSDT | $6.1M | $10.4M | $8.1M | 0.0100% | 8h | 2021-11-30 | 75.0 |
| 55 | WLFIUSDT | $6.0M | $14.3M | $25.4M | -0.0141% | 4h | 2025-08-23 | 75.0 |
| 56 | DOTUSDT | $5.9M | $9.6M | $17.7M | 0.0093% | 8h | 2021-03-18 | 75.0 |
| 57 | FARTCOINUSDT | $5.7M | $12.4M | $14.5M | 0.0050% | 4h | 2024-12-20 | 50.0 |
| 58 | TRXUSDT | $5.7M | $28.0M | $59.9M | -0.0016% | 8h | 2021-08-12 | 75.0 |
| 59 | OPUSDT | $5.5M | $11.3M | $11.3M | 0.0100% | 8h | 2022-06-01 | 75.0 |
| 60 | ASTERUSDT | $5.1M | $7.4M | $47.3M | 0.0042% | 4h | 2025-09-19 | 75.0 |

Turnover distribution (crypto USDT perps): ≥$1.00B: 1; ≥$100.0M: 6; ≥$50.0M: 9; ≥$20.0M: 19; ≥$10.0M: 38

### 2.2 USDC perpetuals (68)

| symbol | baseCoin | turnoverUsd | oiUsd | fundingRate | launchTime |
|---|---|---|---|---|---|
| BTCPERP | BTC | $15.4M | $100.2M | 0.0100% | 2021-01-14 |
| ETHPERP | ETH | $5.5M | $31.6M | 0.0088% | 2022-08-05 |
| XRPPERP | XRP | $621k | $5.2M | 0.0059% | 2022-08-05 |
| MNTPERP | MNT | $448k | $3.6M | 0.0100% | 2024-04-24 |
| BCHPERP | BCH | $369k | $937k | 0.0050% | 2025-04-30 |
| SOLPERP | SOL | $223k | $8.9M | -0.0003% | 2022-08-05 |
| 1000PEPEPERP | 1000PEPE | $210k | $987k | -0.0216% | 2024-04-24 |
| KASPERP | KAS | $182k | $227k | 0.0050% | 2026-03-20 |
| ICNTPERP | ICNT | $177k | $242k | 0.0050% | 2025-09-16 |
| MERLPERP | MERL | $167k | $106k | 0.0050% | 2026-03-19 |
| HYPEPERP | HYPE | $152k | $2.9M | 0.0014% | 2025-04-30 |
| XPLPERP | XPL | $90k | $155k | -0.0139% | 2026-03-19 |
| STRKPERP | STRK | $79k | $193k | -0.0075% | 2024-05-08 |
| AVNTPERP | AVNT | $73k | $123k | -0.1013% | 2026-03-20 |
| PUMPFUNPERP | PUMPFUN | $65k | $246k | 0.0050% | 2025-08-13 |
| INJPERP | INJ | $64k | $272k | 0.0050% | 2025-07-22 |
| SUIPERP | SUI | $64k | $1.1M | 0.0024% | 2024-04-26 |
| UNIPERP | UNI | $61k | $1.4M | 0.0050% | 2025-04-30 |
| TRXPERP | TRX | $59k | $698k | -0.0292% | 2026-03-19 |
| XLMPERP | XLM | $58k | $1.4M | 0.0100% | 2024-11-28 |
| ARBPERP | ARB | $50k | $515k | -0.0014% | 2024-04-22 |
| WLDPERP | WLD | $44k | $848k | -0.0309% | 2024-05-10 |
| ENAPERP | ENA | $42k | $843k | 0.0100% | 2024-11-28 |
| LINKPERP | LINK | $34k | $1.7M | 0.0056% | 2024-11-28 |
| DOTPERP | DOT | $32k | $229k | 0.0100% | 2024-11-28 |
| DOGEPERP | DOGE | $31k | $976k | -0.0197% | 2024-04-22 |
| WLFIPERP | WLFI | $29k | $261k | 0.0050% | 2025-09-16 |
| AAVEPERP | AAVE | $29k | $661k | -0.0023% | 2025-04-30 |
| XAUTPERP | XAUT | $27k | $1.2M | -0.0331% | 2026-01-20 |
| ONDOPERP | ONDO | $27k | $766k | 0.0100% | 2024-05-08 |
| MOODENGPERP | MOODENG | $26k | $60k | 0.0050% | 2025-07-31 |
| POLPERP | POL | $25k | $140k | 0.0100% | 2024-09-18 |
| TRUMPPERP | TRUMP | $21k | $150k | 0.0050% | 2025-04-30 |
| SHIB1000PERP | SHIB1000 | $20k | $33k | 0.0100% | 2024-05-10 |
| ICPPERP | ICP | $19k | $106k | 0.0050% | 2026-01-21 |
| ORDIPERP | ORDI | $19k | $119k | -0.0121% | 2024-04-22 |
| SEIPERP | SEI | $16k | $170k | 0.0050% | 2025-07-31 |
| 1000NEIROCTOPERP | 1000NEIROCTO | $15k | $15k | 0.0050% | 2025-07-31 |
| ASTERPERP | ASTER | $14k | $100k | 0.0050% | 2026-01-20 |
| WIFPERP | WIF | $14k | $284k | 0.0100% | 2024-05-10 |
| BNBPERP | BNB | $11k | $3.1M | 0.0074% | 2022-09-29 |
| PENGUPERP | PENGU | $10k | $117k | 0.0050% | 2025-07-22 |
| ETHFIPERP | ETHFI | $10k | $200k | 0.0100% | 2024-04-24 |
| FARTCOINPERP | FARTCOIN | $10k | $68k | 0.0050% | 2025-07-22 |
| XMRPERP | XMR | $8k | $598k | 0.0050% | 2026-01-20 |
| PAXGPERP | PAXG | $7k | $304k | 0.0051% | 2026-01-21 |
| LTCPERP | LTC | $5k | $771k | 0.0009% | 2025-04-30 |
| CRVPERP | CRV | $5k | $149k | 0.0050% | 2025-04-30 |
| SAHARAPERP | SAHARA | $5k | $46k | 0.0050% | 2025-07-31 |
| POPCATPERP | POPCAT | $5k | $112k | 0.0100% | 2024-09-18 |
| ETCPERP | ETC | $4k | $386k | 0.0100% | 2022-09-01 |
| EIGENPERP | EIGEN | $4k | $66k | 0.0050% | 2026-01-21 |
| OPPERP | OP | $4k | $170k | 0.0100% | 2022-09-01 |
| 1000BONKPERP | 1000BONK | $3k | $182k | 0.0080% | 2024-11-28 |
| NIGHTPERP | NIGHT | $3k | $37k | 0.0050% | 2026-03-19 |
| TWTPERP | TWT | $3k | $38k | 0.0050% | 2026-03-20 |
| HBARPERP | HBAR | $3k | $172k | 0.0050% | 2025-07-22 |
| TIAPERP | TIA | $3k | $170k | 0.0100% | 2024-05-10 |
| NOTPERP | NOT | $2k | $23k | 0.0100% | 2024-09-18 |
| AEVOPERP | AEVO | $2k | $31k | 0.0100% | 2024-05-08 |
| TAOPERP | TAO | $2k | $259k | 0.0050% | 2025-07-22 |
| FILPERP | FIL | $2k | $95k | 0.0050% | 2026-01-21 |
| SPXPERP | SPX | $2k | $88k | 0.0050% | 2025-07-31 |
| PENDLEPERP | PENDLE | $669 | $121k | 0.0050% | 2025-09-16 |
| VIRTUALPERP | VIRTUAL | $662 | $233k | 0.0050% | 2025-07-22 |
| PNUTPERP | PNUT | $618 | $30k | 0.0050% | 2025-07-22 |
| RESOLVPERP | RESOLV | $334 | $49k | 0.0050% | 2025-07-31 |
| HPERP | H | $322 | $100k | 0.0050% | 2025-09-16 |

### 2.3 Linear dated futures (40)

| symbol | baseCoin | settleCoin | deliveryTime | markPrice | turnoverUsd | oiUsd | annBasisVsPerp |
|---|---|---|---|---|---|---|---|
| BTCUSDT-13NOV26 | BTC | USDT | 2026-11-13 | 75922.9 | $239k | $11.9M | 7.69% |
| BTCUSDT-20NOV26 | BTC | USDT | 2026-11-20 | 76088.1 | $0 | $1.8M | 9.73% |
| BTCUSDT-27NOV26 | BTC | USDT | 2026-11-27 | 76134.5 | $561k | $5.3M | 7.39% |
| BTCUSDT-25DEC26 | BTC | USDT | 2026-12-25 | 76461.71 | $433k | $13.8M | 6.35% |
| BTCUSDT-29JAN27 | BTC | USDT | 2027-01-29 | 76924.2 | $0 | $1.0M | 6.36% |
| BTCUSDT-26MAR27 | BTC | USDT | 2027-03-26 | 77537.2 | $233 | $10.6M | 5.92% |
| BTCUSDT-25JUN27 | BTC | USDT | 2027-06-25 | 78521.2 | $27k | $5.5M | 5.63% |
| BTCUSDT-24SEP27 | BTC | USDT | 2027-09-24 | 79627.2 | $300k | $2.5M | 5.70% |
| DOGEUSDT-13NOV26 | DOGE | USDT | 2026-11-13 | 0.07192 | $58k | $65k | 10.90% |
| DOGEUSDT-20NOV26 | DOGE | USDT | 2026-11-20 | 0.07194 | $248k | $18k | 5.65% |
| DOGEUSDT-27NOV26 | DOGE | USDT | 2026-11-27 | 0.07205 | $507k | $54k | 6.48% |
| DOGEUSDT-25DEC26 | DOGE | USDT | 2026-12-25 | 0.07223 | $1k | $384 | 4.59% |
| ETHUSDT-13NOV26 | ETH | USDT | 2026-11-13 | 2237.89 | $19k | $5.8M | 6.79% |
| ETHUSDT-20NOV26 | ETH | USDT | 2026-11-20 | 2241.09 | $606 | $2.5M | 7.17% |
| ETHUSDT-27NOV26 | ETH | USDT | 2026-11-27 | 2241.86 | $1.6M | $4.8M | 5.25% |
| ETHUSDT-25DEC26 | ETH | USDT | 2026-12-25 | 2250.4 | $183k | $11.8M | 5.09% |
| ETHUSDT-29JAN27 | ETH | USDT | 2027-01-29 | 2259.93 | $122k | $358k | 4.82% |
| ETHUSDT-26MAR27 | ETH | USDT | 2027-03-26 | 2276.6 | $47k | $11.3M | 4.83% |
| ETHUSDT-25JUN27 | ETH | USDT | 2027-06-25 | 2304.58 | $17k | $4.5M | 4.91% |
| ETHUSDT-24SEP27 | ETH | USDT | 2027-09-24 | 2332.15 | $11k | $1.8M | 4.92% |
| HYPEUSDT-13NOV26 | HYPE | USDT | 2026-11-13 | 77.71 | $416k | $121k | 13.87% |
| HYPEUSDT-20NOV26 | HYPE | USDT | 2026-11-20 | 77.91 | $606k | $53k | 13.64% |
| HYPEUSDT-27NOV26 | HYPE | USDT | 2026-11-27 | 77.94 | $704k | $553k | 9.49% |
| HYPEUSDT-25DEC26 | HYPE | USDT | 2026-12-25 | 78.53 | $80 | $256 | 9.74% |
| MNTUSDT-13NOV26 | MNT | USDT | 2026-11-13 | 0.4526 | $0 | $1k | 1.44% |
| MNTUSDT-20NOV26 | MNT | USDT | 2026-11-20 | 0.4531 | $0 | $1k | 3.84% |
| MNTUSDT-27NOV26 | MNT | USDT | 2026-11-27 | 0.4536 | $0 | $2k | 4.53% |
| MNTUSDT-25DEC26 | MNT | USDT | 2026-12-25 | 0.4557 | $656 | $498 | 5.42% |
| SOLUSDT-13NOV26 | SOL | USDT | 2026-11-13 | 89.94 | $315k | $56k | 14.53% |
| SOLUSDT-20NOV26 | SOL | USDT | 2026-11-20 | 89.93 | $317k | $299 | 6.13% |
| SOLUSDT-27NOV26 | SOL | USDT | 2026-11-27 | 90.0 | $2.4M | $644k | 5.40% |
| SOLUSDT-25DEC26 | SOL | USDT | 2026-12-25 | 90.39 | $0 | $118 | 5.55% |
| XAUTUSDT-13NOV26 | XAUT | USDT | 2026-11-13 | 4062.9 | $0 | $22k | 3.24% |
| XAUTUSDT-20NOV26 | XAUT | USDT | 2026-11-20 | 4068.3 | $0 | $1k | 5.29% |
| XAUTUSDT-27NOV26 | XAUT | USDT | 2026-11-27 | 4074.6 | $0 | $20k | 6.29% |
| XAUTUSDT-25DEC26 | XAUT | USDT | 2026-12-25 | 4097.4 | $0 | $0 | 6.90% |
| XRPUSDT-13NOV26 | XRP | USDT | 2026-11-13 | 1.164 | $341k | $110k | 11.22% |
| XRPUSDT-20NOV26 | XRP | USDT | 2026-11-20 | 1.164 | $777k | $13k | 4.99% |
| XRPUSDT-27NOV26 | XRP | USDT | 2026-11-27 | 1.165 | $1.8M | $25k | 4.81% |
| XRPUSDT-25DEC26 | XRP | USDT | 2026-12-25 | 1.1697 | $41 | $20 | 5.08% |

annBasisVsPerp = (future mark / USDT-perp mark − 1) annualised to delivery.

### 2.4 TradFi perpetuals (310; symbolType stock/ETF/commodity/forex — `innovation` is a crypto risk zone and is counted as crypto)

| symbolType | n | t |
|---|---|---|
| ETF | 59 | $260.9M |
| stock | 243 | $210.7M |
| commodity | 4 | $91.4M |
| forex | 4 | $218k |

307/310 TradFi perps show non-zero 24h price change and turnover in this Saturday snapshot (i.e. they trade through the weekend).

**ETF** (59)

| symbol | status | turnoverUsd | oiUsd | fundingRate | price24hPcnt | maxLeverage | launchTime |
|---|---|---|---|---|---|---|---|
| SOXLUSDT | Trading | $182.5M | $74.4M | 0.0000% | 3.43% | 100.0 | 2026-05-19 |
| SOXSUSDT | Trading | $27.2M | $9.2M | -0.0058% | -2.44% | 50.0 | 2026-07-16 |
| KORUUSDT | Trading | $18.2M | $16.5M | 0.0000% | 3.38% | 50.0 | 2026-07-15 |
| QQQUSDT | Trading | $7.5M | $27.1M | 0.0000% | 0.22% | 50.0 | 2026-05-07 |
| SNXXUSDT | Trading | $7.3M | $7.0M | 0.1599% | 0.92% | 50.0 | 2026-07-14 |
| SPYUSDT | Trading | $3.0M | $10.3M | 0.0000% | 0.02% | 100.0 | 2026-05-14 |
| DRAMUSDT | Trading | $2.8M | $7.2M | 0.0290% | 1.65% | 100.0 | 2026-05-19 |
| MUUUSDT | Trading | $2.2M | $1.9M | 0.0675% | 0.36% | 25.0 | 2026-07-16 |
| TQQQUSDT | Trading | $1.9M | $5.1M | 0.0000% | 0.92% | 25.0 | 2026-06-24 |
| EWYUSDT | Trading | $1.6M | $6.3M | 0.0000% | 1.17% | 100.0 | 2026-05-07 |
| XLFUSDT | Trading | $1.0M | $72k | 0.0000% | -0.17% | 25.0 | 2026-07-21 |
| UVXYUSDT | Trading | $983k | $1.5M | 0.0000% | 1.43% | 25.0 | 2026-06-25 |
| SQQQUSDT | Trading | $853k | $1.7M | 0.0000% | -1.15% | 25.0 | 2026-06-30 |
| IBITUSDT | Trading | $525k | $365k | 0.0000% | 0.21% | 25.0 | 2026-09-16 |
| INTWUSDT | Trading | $293k | $1.1M | 0.0000% | 2.70% | 25.0 | 2026-07-14 |
| MVLLUSDT | Trading | $273k | $708k | 0.0000% | 5.42% | 25.0 | 2026-06-30 |
| SOXXUSDT | Trading | $265k | $1.1M | 0.0000% | 1.25% | 25.0 | 2026-07-21 |
| XLKUSDT | Trading | $223k | $79k | 0.0000% | 0.38% | 25.0 | 2026-07-21 |
| SPCHUSDT | Trading | $208k | $150k | 0.0000% | 5.29% | 25.0 | 2026-08-20 |
| XLVUSDT | Trading | $199k | $120k | 0.0000% | 0.02% | 25.0 | 2026-07-21 |
| SKDDUSDT | Trading | $154k | $180k | 0.0000% | -5.01% | 25.0 | 2026-08-10 |
| EWTUSDT | Trading | $130k | $263k | 0.0000% | 0.40% | 25.0 | 2026-06-04 |
| CSOPSKHYNIX2LUSDT | Trading | $129k | $219k | 0.0000% | 4.41% | 20.0 | 2026-08-11 |
| TLTUSDT | Trading | $126k | $657k | 0.0000% | -0.42% | 25.0 | 2026-09-18 |
| SMHUSDT | Trading | $121k | $505k | 0.0000% | 1.35% | 25.0 | 2026-07-15 |
| LYTEUSDT | Trading | $105k | $180k | 0.0000% | 1.76% | 25.0 | 2026-08-13 |
| MSTUUSDT | Trading | $101k | $756k | 0.0000% | 1.44% | 25.0 | 2026-08-26 |
| RAMUSDT | Trading | $101k | $442k | 0.0000% | 2.98% | 25.0 | 2026-08-25 |
| SSPCUSDT | Trading | $100k | $105k | 0.0000% | -6.89% | 25.0 | 2026-07-29 |
| EWJUSDT | Trading | $43k | $263k | -0.0276% | 0.07% | 25.0 | 2026-05-07 |
| NCLDUSDT | Trading | $43k | $132k | 0.0000% | 1.31% | 25.0 | 2026-08-13 |
| BWETUSDT | Trading | $41k | $158k | 0.0000% | 1.00% | 25.0 | 2026-10-09 |
| SKUUUSDT | Trading | $41k | $168k | 0.0200% | 4.98% | 25.0 | 2026-08-10 |
| TMFUSDT | Trading | $40k | $890k | 0.0000% | -0.75% | 25.0 | 2026-07-29 |
| TZAUSDT | Trading | $39k | $289k | 0.0000% | -0.98% | 25.0 | 2026-07-16 |
| KSTRUSDT | Trading | $37k | $1.2M | 0.0000% | 1.39% | 25.0 | 2026-07-14 |
| XBIUSDT | Trading | $34k | $219k | 0.0000% | 0.61% | 25.0 | 2026-07-15 |
| EWZUSDT | Trading | $33k | $981k | 0.0000% | 0.22% | 25.0 | 2026-08-31 |
| URNMUSDT | Trading | $33k | $51k | 0.0000% | 0.91% | 25.0 | 2026-09-22 |
| CONLUSDT | Trading | $33k | $206k | 0.0000% | 2.21% | 25.0 | 2026-09-16 |
| SPXLUSDT | Trading | $31k | $110k | 0.0000% | 0.18% | 25.0 | 2026-08-26 |
| IWMUSDT | Trading | $29k | $532k | 0.0000% | 0.26% | 25.0 | 2026-05-14 |
| NVDLUSDT | Trading | $24k | $240k | 0.0000% | 0.76% | 25.0 | 2026-07-21 |
| CSOPSAMSUNG2LUSDT | Trading | $20k | $78k | 0.0000% | 2.85% | 20.0 | 2026-08-11 |
| ZCSHUSDT | Trading | $18k | $100k | 0.0000% | -5.14% | 25.0 | 2026-10-13 |
| XLUUSDT | Trading | $12k | $91k | 0.0000% | -0.30% | 25.0 | 2026-08-27 |
| ARKKUSDT | Trading | $10k | $175k | 0.0000% | 1.56% | 25.0 | 2026-07-20 |
| GDXUSDT | Trading | $10k | $405k | 0.0000% | 1.14% | 25.0 | 2026-07-27 |
| XLPUSDT | Trading | $10k | $130k | 0.0000% | -1.60% | 25.0 | 2026-08-27 |
| TSLLUSDT | Trading | $9k | $162k | 0.0000% | 0.11% | 25.0 | 2026-07-20 |
| XLIUSDT | Trading | $8k | $11k | 0.0000% | -0.52% | 25.0 | 2026-10-30 |
| TBTUSDT | Trading | $7k | $156k | 0.0000% | 0.64% | 25.0 | 2026-07-29 |
| MSFUUSDT | Trading | $6k | $394k | 0.0000% | -1.13% | 25.0 | 2026-09-16 |
| XLEUSDT | Trading | $4k | $173k | 0.0000% | -0.26% | 25.0 | 2026-07-15 |
| BMNUUSDT | Trading | $3k | $40k | 0.0000% | 4.46% | 25.0 | 2026-10-09 |
| KODEX200USDT | Trading | $3k | $13k | 0.0000% | -1.08% | 25.0 | 2026-08-14 |
| EWZSUSDT | Trading | $2k | $25k | 0.0000% | -1.37% | 25.0 | 2026-10-30 |
| MAGSUSDT | Trading | $2k | $36k | 0.0000% | 0.00% | 25.0 | 2026-08-19 |
| BITOUSDT | Trading | $2k | $21k | 0.0000% | -0.13% | 25.0 | 2026-07-27 |

**stock** (243)

| symbol | status | turnoverUsd | oiUsd | fundingRate | price24hPcnt | maxLeverage | launchTime |
|---|---|---|---|---|---|---|---|
| SNDKUSDT | Trading | $43.0M | $41.1M | 0.0891% | 1.22% | 100.0 | 2026-04-28 |
| MUUSDT | Trading | $30.0M | $27.1M | 0.0065% | 0.15% | 100.0 | 2026-05-07 |
| SPCXUSDT | Trading | $22.3M | $41.2M | -0.0156% | 2.48% | 100.0 | 2026-06-15 |
| INTCUSDT | Trading | $11.4M | $41.5M | 0.0000% | 1.57% | 50.0 | 2026-04-22 |
| SKHYNIXUSDT | Trading | $11.3M | $31.2M | 0.0011% | 2.81% | 100.0 | 2026-06-04 |
| NVDAUSDT | Trading | $8.0M | $19.6M | 0.0000% | 0.36% | 100.0 | 2026-05-06 |
| SKHYUSDT | Trading | $7.9M | $11.7M | 0.0029% | 2.63% | 100.0 | 2026-07-11 |
| MSTRUSDT | Trading | $6.0M | $22.9M | -0.0233% | 0.78% | 50.0 | 2026-04-29 |
| NBISUSDT | Trading | $5.7M | $13.2M | 0.0000% | 2.38% | 50.0 | 2026-05-26 |
| CRCLUSDT | Trading | $4.1M | $14.9M | 0.0000% | 1.25% | 50.0 | 2026-05-06 |
| GOOGLUSDT | Trading | $3.9M | $37.2M | 0.0000% | -0.29% | 100.0 | 2026-04-27 |
| TSLAUSDT | Trading | $3.6M | $17.0M | 0.0000% | 0.06% | 100.0 | 2026-04-21 |
| ANTHROPICUSDT | PreLaunch | $3.3M | $2.9M | 0.0000% | 2.22% | 25.0 | 2026-07-13 |
| SAMSUNGUSDT | Trading | $3.1M | $6.4M | 0.0000% | 0.85% | 100.0 | 2026-06-04 |
| ORCLUSDT | Trading | $3.1M | $3.2M | 0.0000% | 5.55% | 50.0 | 2026-05-06 |
| MRVLUSDT | Trading | $2.7M | $4.8M | 0.0000% | 3.37% | 50.0 | 2026-05-13 |
| AAPLUSDT | Trading | $2.0M | $14.4M | 0.0000% | -0.30% | 100.0 | 2026-04-28 |
| AMDSTOCKUSDT | Trading | $1.9M | $3.6M | 0.0000% | 1.62% | 50.0 | 2026-05-28 |
| CRWVUSDT | Trading | $1.7M | $3.2M | 0.0000% | 4.15% | 25.0 | 2026-05-12 |
| FLYUSDT | Trading | $1.7M | $497k | 0.0000% | 6.47% | 25.0 | 2026-07-20 |
| COHRUSDT | Trading | $1.4M | $715k | 0.0000% | 2.29% | 25.0 | 2026-06-16 |
| DELLUSDT | Trading | $1.4M | $2.2M | 0.0000% | 0.63% | 50.0 | 2026-06-02 |
| ADBEUSDT | Trading | $1.3M | $214k | 0.0000% | -1.08% | 25.0 | 2026-06-29 |
| AAOIUSDT | Trading | $1.3M | $1.5M | -0.0353% | 2.64% | 25.0 | 2026-06-09 |
| HOODUSDT | Trading | $1.1M | $4.8M | -0.0208% | 1.62% | 25.0 | 2026-04-22 |
| MSFTUSDT | Trading | $1.0M | $10.1M | 0.0000% | -0.25% | 100.0 | 2026-04-27 |
| IRENUSDT | Trading | $961k | $2.1M | 0.0204% | 5.31% | 25.0 | 2026-06-09 |
| CBRSUSDT | Trading | $916k | $4.5M | 0.0000% | 3.25% | 25.0 | 2026-05-19 |
| LITEUSDT | Trading | $858k | $3.7M | 0.0000% | 2.92% | 25.0 | 2026-05-12 |
| QNTXUSDT | Trading | $827k | $354k | 0.0000% | 3.04% | 25.0 | 2026-06-09 |
| BEUSDT | Trading | $815k | $1.9M | 0.0000% | 1.48% | 25.0 | 2026-05-28 |
| XOMUSDT | Trading | $776k | $599k | 0.0000% | -0.28% | 50.0 | 2026-08-03 |
| MRNAUSDT | Trading | $775k | $1.3M | 0.0000% | -1.18% | 25.0 | 2026-08-20 |
| COINUSDT | Trading | $747k | $5.5M | 0.0000% | 1.19% | 50.0 | 2026-04-29 |
| OPENAIUSDT | PreLaunch | $741k | $893k | 0.0000% | 1.87% | 25.0 | 2026-07-13 |
| AVGOUSDT | Trading | $730k | $4.7M | 0.0139% | 1.42% | 50.0 | 2026-05-12 |
| WDCUSDT | Trading | $720k | $2.2M | 0.0315% | 1.09% | 25.0 | 2026-05-28 |
| METAUSDT | Trading | $709k | $6.5M | 0.0000% | -0.75% | 100.0 | 2026-04-27 |
| APPSTOCKUSDT | Trading | $628k | $1.2M | -0.0540% | 0.21% | 25.0 | 2026-07-16 |
| ASTSUSDT | Trading | $552k | $1.2M | 0.0000% | 2.64% | 25.0 | 2026-06-04 |
| QCOMUSDT | Trading | $525k | $1.2M | 0.0000% | 1.31% | 25.0 | 2026-05-26 |
| RKLBUSDT | Trading | $515k | $1.5M | 0.0000% | 5.62% | 25.0 | 2026-05-20 |
| AMZNUSDT | Trading | $461k | $3.7M | 0.0000% | -0.68% | 100.0 | 2026-04-28 |
| CSCOUSDT | Trading | $448k | $306k | 0.0000% | -0.20% | 25.0 | 2026-05-20 |
| BMNRUSDT | Trading | $430k | $1.4M | -0.1206% | 1.26% | 25.0 | 2026-05-14 |
| ARMUSDT | Trading | $398k | $774k | 0.0000% | 0.90% | 25.0 | 2026-05-13 |
| UNITREEUSDT | Trading | $360k | $3.6M | 0.0000% | 0.03% | 25.0 | 2026-08-03 |
| ALABUSDT | Trading | $354k | $379k | 0.0000% | 3.50% | 25.0 | 2026-06-29 |
| PLTRUSDT | Trading | $339k | $2.6M | 0.0000% | 0.11% | 25.0 | 2026-04-29 |
| CRWDUSDT | Trading | $332k | $215k | 0.0000% | -0.48% | 25.0 | 2026-07-03 |
| SHAZUSDT | Trading | $316k | $766k | -0.0349% | -1.13% | 25.0 | 2026-07-08 |
| HUTUSDT | Trading | $280k | $173k | 0.0000% | 5.14% | 25.0 | 2026-09-21 |
| AXTIUSDT | Trading | $280k | $851k | 0.0000% | 2.54% | 25.0 | 2026-06-16 |
| TSMUSDT | Trading | $278k | $2.5M | 0.0000% | 1.17% | 50.0 | 2026-04-22 |
| ANETUSDT | Trading | $263k | $231k | 0.0000% | 0.47% | 25.0 | 2026-08-20 |
| AEHRUSDT | Trading | $261k | $196k | 0.0000% | 2.84% | 25.0 | 2026-07-16 |
| IONQUSDT | Trading | $251k | $734k | 0.0000% | 4.64% | 25.0 | 2026-07-17 |
| STXXUSDT | Trading | $245k | $491k | 0.0495% | 1.94% | 25.0 | 2026-06-25 |
| PENGSTOCKUSDT | Trading | $244k | $280k | 0.0000% | -0.83% | 25.0 | 2026-07-22 |
| TTWOUSDT | Trading | $234k | $2.1M | 0.0000% | 0.36% | 25.0 | 2026-07-06 |
| DDOGUSDT | Trading | $232k | $156k | 0.0000% | -1.03% | 25.0 | 2026-08-18 |
| PURRUSDT | Trading | $228k | $624k | 0.0000% | -0.02% | 25.0 | 2026-07-10 |
| BBXUSDT | Trading | $223k | $281k | 0.0000% | -1.14% | 25.0 | 2026-06-23 |
| OKLOUSDT | Trading | $222k | $456k | 0.0000% | 3.07% | 20.0 | 2026-09-22 |
| ONDSUSDT | Trading | $220k | $409k | 0.0000% | 4.00% | 25.0 | 2026-06-09 |
| LLYUSDT | Trading | $214k | $1.1M | 0.0000% | -0.46% | 25.0 | 2026-05-14 |
| AMATUSDT | Trading | $208k | $464k | 0.0000% | 0.02% | 25.0 | 2026-06-05 |
| CRDOUSDT | Trading | $203k | $847k | 0.0000% | 4.66% | 25.0 | 2026-06-23 |
| RDWUSDT | Trading | $195k | $280k | 0.0000% | 7.67% | 25.0 | 2026-07-16 |
| OKTAUSDT | Trading | $190k | $444k | 0.0000% | -2.18% | 25.0 | 2026-08-31 |
| DJTUSDT | Trading | $176k | $119k | 0.0000% | 1.10% | 25.0 | 2026-08-25 |
| VRTUSDT | Trading | $176k | $429k | 0.0000% | 1.50% | 25.0 | 2026-07-10 |
| ASMLUSDT | Trading | $173k | $834k | 0.0000% | 0.80% | 25.0 | 2026-06-26 |
| RIOTUSDT | Trading | $141k | $190k | 0.0000% | 2.74% | 25.0 | 2026-08-12 |
| FWDIUSDT | Trading | $138k | $179k | 0.0000% | 0.87% | 25.0 | 2026-07-16 |
| TSEMUSDT | Trading | $137k | $236k | 0.0000% | 1.94% | 25.0 | 2026-07-17 |
| NETUSDT | Trading | $131k | $494k | 0.0000% | -1.91% | 25.0 | 2026-08-06 |
| CIFRUSDT | Trading | $129k | $169k | 0.0000% | 5.50% | 25.0 | 2026-08-07 |
| CLSKUSDT | Trading | $127k | $149k | 0.0000% | 5.80% | 25.0 | 2026-08-04 |
| FLEXUSDT | Trading | $127k | $107k | 0.0000% | 3.47% | 25.0 | 2026-07-03 |
| BOTUSDT | Trading | $125k | $269k | 0.0000% | -0.37% | 25.0 | 2026-07-01 |
| JDUSDT | Trading | $118k | $110k | 0.0000% | 0.21% | 25.0 | 2026-08-21 |
| UBERUSDT | Trading | $116k | $515k | 0.0000% | -0.71% | 25.0 | 2026-08-06 |
| GEVUSDT | Trading | $111k | $159k | 0.0000% | 0.60% | 25.0 | 2026-07-10 |
| CYPHUSDT | Trading | $110k | $125k | 0.0000% | 1.11% | 25.0 | 2026-09-21 |
| BNCUSDT | Trading | $110k | $456k | 0.0000% | 2.34% | 25.0 | 2026-07-16 |
| MRKUSDT | Trading | $108k | $129k | 0.0000% | -0.60% | 25.0 | 2026-08-21 |
| BABAUSDT | Trading | $108k | $1.3M | 0.0000% | 0.62% | 25.0 | 2026-05-13 |
| USARUSDT | Trading | $106k | $456k | -0.0210% | 2.87% | 25.0 | 2026-06-23 |
| JPMUSDT | Trading | $105k | $319k | 0.0000% | -0.78% | 25.0 | 2026-07-31 |
| TXNUSDT | Trading | $97k | $207k | 0.0000% | 1.60% | 25.0 | 2026-07-02 |
| SMCIUSDT | Trading | $92k | $417k | 0.0000% | 2.93% | 25.0 | 2026-06-24 |
| FUTUUSDT | Trading | $88k | $239k | 0.0000% | -0.12% | 25.0 | 2026-08-20 |
| POETUSDT | Trading | $87k | $201k | 0.0000% | 5.48% | 25.0 | 2026-07-16 |
| OUSTUSDT | Trading | $85k | $68k | 0.0000% | 5.03% | 25.0 | 2026-08-10 |
| BSPUSDT | Trading | $85k | $268k | 0.0000% | 0.85% | 25.0 | 2026-07-06 |
| MARAUSDT | Trading | $84k | $441k | 0.0000% | 5.68% | 25.0 | 2026-07-17 |
| DKNGUSDT | Trading | $84k | $141k | 0.0000% | 6.97% | 25.0 | 2026-07-20 |
| PFEUSDT | Trading | $81k | $113k | 0.0000% | 0.51% | 25.0 | 2026-11-03 |
| ISRGUSDT | Trading | $79k | $309k | -0.1241% | -1.02% | 25.0 | 2026-08-18 |
| LGELECTRONICSUSDT | Trading | $79k | $129k | 0.0000% | -0.15% | 25.0 | 2026-08-14 |
| SMICUSDT | Trading | $77k | $74k | 0.0000% | 1.37% | 20.0 | 2026-08-12 |
| KLACUSDT | Trading | $76k | $249k | 0.0000% | 1.76% | 25.0 | 2026-06-26 |
| ZHIPUUSDT | Trading | $75k | $489k | 0.0000% | 1.77% | 25.0 | 2026-07-28 |
| GMESTOCKUSDT | Trading | $72k | $205k | 0.0000% | 2.00% | 25.0 | 2026-07-27 |
| SNOWUSDT | Trading | $72k | $213k | 0.0000% | 0.38% | 25.0 | 2026-07-10 |
| NFLXUSDT | Trading | $70k | $1.2M | 0.0000% | -0.61% | 50.0 | 2026-07-06 |
| CVXSTOCKUSDT | Trading | $70k | $98k | 0.0000% | -0.86% | 25.0 | 2026-09-03 |
| CIENUSDT | Trading | $70k | $195k | 0.0000% | 2.43% | 25.0 | 2026-06-29 |
| GLWUSDT | Trading | $69k | $488k | 0.0000% | 0.62% | 25.0 | 2026-06-24 |
| HANMIUSDT | Trading | $68k | $81k | 0.0000% | -0.02% | 25.0 | 2026-08-14 |
| BXUSDT | Trading | $67k | $513k | 0.0000% | -0.72% | 25.0 | 2026-07-17 |
| DUOLUSDT | Trading | $66k | $19k | 0.0000% | -1.57% | 25.0 | 2026-11-06 |
| TERUSDT | Trading | $65k | $543k | 0.0000% | 1.41% | 25.0 | 2026-07-02 |
| FNUSDT | Trading | $65k | $31k | 0.0000% | 2.09% | 25.0 | 2026-11-06 |
| INFQUSDT | Trading | $61k | $72k | 0.0000% | 1.83% | 25.0 | 2026-08-26 |
| WENSTOCKUSDT | Trading | $60k | $229k | 0.0000% | 0.78% | 25.0 | 2026-07-20 |
| LRCXUSDT | Trading | $53k | $329k | 0.0000% | 0.26% | 25.0 | 2026-06-24 |
| AALUSDT | Trading | $53k | $97k | 0.0000% | 0.23% | 25.0 | 2026-07-23 |
| MNSTUSDT | Trading | $52k | $25k | 0.0000% | -2.31% | 25.0 | 2026-08-18 |
| EBAYUSDT | Trading | $50k | $83k | 0.0000% | -0.33% | 25.0 | 2026-08-07 |
| VSTUSDT | Trading | $49k | $415k | 0.0000% | 0.66% | 25.0 | 2026-08-13 |
| SMRUSDT | Trading | $48k | $59k | 0.0000% | 0.67% | 25.0 | 2026-10-30 |
| TEAMUSDT | Trading | $45k | $115k | 0.0000% | -1.08% | 25.0 | 2026-09-03 |
| TEMUSDT | Trading | $43k | $345k | 0.0000% | 1.61% | 25.0 | 2026-08-31 |
| AMGNUSDT | Trading | $43k | $84k | 0.0000% | -0.34% | 25.0 | 2026-07-31 |
| IBKRUSDT | Trading | $43k | $31k | 0.0000% | 0.44% | 25.0 | 2026-10-23 |
| HPEUSDT | Trading | $43k | $95k | 0.0000% | 0.83% | 25.0 | 2026-06-26 |
| MCDUSDT | Trading | $41k | $426k | 0.0000% | -0.19% | 25.0 | 2026-08-27 |
| MEITUUSDT | Trading | $39k | $43k | 0.0000% | -0.15% | 25.0 | 2026-09-11 |
| GSUSDT | Trading | $39k | $217k | 0.0000% | -0.95% | 25.0 | 2026-07-23 |
| PANWUSDT | Trading | $39k | $198k | 0.0000% | -1.25% | 25.0 | 2026-07-22 |
| GPROUSDT | Trading | $38k | $329k | 0.0000% | 2.10% | 25.0 | 2026-09-02 |
| USDEXUSDT | Trading | $36k | $237k | 0.0000% | -2.91% | 25.0 | 2026-10-09 |
| PYPLUSDT | Trading | $35k | $214k | 0.0000% | -0.92% | 25.0 | 2026-07-30 |
| GILDUSDT | Trading | $35k | $98k | 0.0000% | -0.45% | 25.0 | 2026-07-31 |
| FISVUSDT | Trading | $32k | $14k | 0.0000% | -0.25% | 25.0 | 2026-11-06 |
| NOWUSDT | Trading | $32k | $572k | 0.0000% | -0.06% | 25.0 | 2026-06-04 |
| BACUSDT | Trading | $31k | $202k | 0.0000% | -0.32% | 25.0 | 2026-08-04 |
| BRKBUSDT | Trading | $30k | $194k | 0.0000% | -0.09% | 25.0 | 2026-07-30 |
| HDUSDT | Trading | $30k | $61k | 0.0000% | -0.73% | 25.0 | 2026-08-19 |
| COSTUSDT | Trading | $30k | $267k | 0.0000% | -0.91% | 25.0 | 2026-07-08 |
| ZHONGJIUSDT | Trading | $29k | $648k | 0.0000% | 1.40% | 25.0 | 2026-08-14 |
| MXLUSDT | Trading | $27k | $20k | 0.0000% | 3.31% | 25.0 | 2026-10-22 |
| HIMSUSDT | Trading | $27k | $222k | -0.0038% | 2.16% | 25.0 | 2026-07-03 |
| CGNXUSDT | Trading | $27k | $47k | 0.0000% | -0.48% | 25.0 | 2026-09-10 |
| SOFIUSDT | Trading | $27k | $177k | 0.0000% | 0.19% | 25.0 | 2026-07-22 |
| CATSTOCKUSDT | Trading | $27k | $301k | 0.0000% | 0.29% | 25.0 | 2026-07-02 |
| IBMUSDT | Trading | $26k | $787k | 0.0000% | 0.79% | 50.0 | 2026-06-05 |
| SONYUSDT | Trading | $26k | $105k | 0.0000% | 0.04% | 25.0 | 2026-07-08 |
| VKTXUSDT | Trading | $24k | $20k | 0.0000% | 0.94% | 25.0 | 2026-10-09 |
| AGPUUSDT | Trading | $23k | $107k | 0.0000% | -0.37% | 25.0 | 2026-10-26 |
| MDBUSDT | Trading | $23k | $56k | 0.0000% | 0.00% | 25.0 | 2026-09-03 |
| HUAHONGUSDT | Trading | $22k | $16k | 0.0000% | -0.87% | 25.0 | 2026-09-07 |
| ONUSDT | Trading | $20k | $87k | 0.0000% | -0.05% | 50.0 | 2026-07-21 |
| UMCUSDT | Trading | $20k | $26k | 0.0000% | 2.62% | 25.0 | 2026-10-09 |
| GEUSDT | Trading | $20k | $104k | 0.0000% | 0.45% | 25.0 | 2026-07-31 |
| GTLBUSDT | Trading | $19k | $29k | 0.0000% | -0.29% | 25.0 | 2026-09-18 |
| SPOTUSDT | Trading | $18k | $151k | 0.0000% | -0.75% | 25.0 | 2026-08-03 |
| MINIMAXUSDT | Trading | $18k | $356k | 0.0000% | 0.92% | 25.0 | 2026-07-28 |
| YOFCUSDT | Trading | $18k | $18k | 0.0000% | -3.91% | 25.0 | 2026-09-02 |
| KOUSDT | Trading | $17k | $286k | 0.0000% | -0.64% | 25.0 | 2026-07-30 |
| FLNCUSDT | Trading | $17k | $377k | 0.0000% | 2.12% | 25.0 | 2026-06-16 |
| APLDUSDT | Trading | $17k | $132k | 0.0000% | 2.90% | 25.0 | 2026-07-17 |
| RDDTUSDT | Trading | $17k | $316k | 0.0000% | 0.69% | 25.0 | 2026-08-04 |
| XIAOMIUSDT | Trading | $17k | $225k | -0.0244% | -0.13% | 25.0 | 2026-08-12 |
| SECZUSDT | Trading | $16k | $861k | 0.0000% | 2.70% | 25.0 | 2026-09-18 |
| ACNUSDT | Trading | $16k | $51k | 0.0000% | -0.15% | 25.0 | 2026-10-08 |
| NKEUSDT | Trading | $14k | $245k | 0.0000% | 0.42% | 25.0 | 2026-10-08 |
| RUMUSDT | Trading | $14k | $29k | 0.0000% | -0.20% | 25.0 | 2026-09-29 |
| CRMUSDT | Trading | $14k | $128k | 0.0000% | -0.33% | 25.0 | 2026-07-23 |
| ZMUSDT | Trading | $12k | $50k | 0.0000% | 0.32% | 25.0 | 2026-07-17 |
| PDDUSDT | Trading | $12k | $568k | 0.0000% | -0.16% | 25.0 | 2026-08-21 |
| PEPUSDT | Trading | $12k | $118k | 0.0000% | 0.28% | 25.0 | 2026-07-30 |
| AMCUSDT | Trading | $11k | $61k | 0.0000% | 1.13% | 25.0 | 2026-07-27 |
| VISTUSDT | Trading | $11k | $49k | 0.0000% | 0.96% | 25.0 | 2026-10-12 |
| RIVNUSDT | Trading | $11k | $111k | 0.0000% | 1.11% | 25.0 | 2026-07-20 |
| YPFUSDT | Trading | $11k | $77k | 0.0000% | -0.65% | 25.0 | 2026-10-12 |
| UNHUSDT | Trading | $10k | $126k | 0.0000% | 0.15% | 25.0 | 2026-07-31 |
| EIDOUSDT | Trading | $10k | $38k | 0.0000% | 0.10% | 25.0 | 2026-10-27 |
| CXMTUSDT | Trading | $10k | $486k | 0.0000% | 0.57% | 20.0 | 2026-07-27 |
| AKAMUSDT | Trading | $9k | $46k | 0.0000% | -1.03% | 25.0 | 2026-11-03 |
| NOKIAUSDT | Trading | $9k | $187k | 0.0000% | 0.06% | 25.0 | 2026-06-05 |
| REGNUSDT | Trading | $9k | $67k | 0.0000% | 0.61% | 25.0 | 2026-07-31 |
| CORZUSDT | Trading | $9k | $33k | 0.0000% | 0.94% | 25.0 | 2026-09-18 |
| TWSTUSDT | Trading | $9k | $145k | 0.0000% | 1.67% | 25.0 | 2026-09-29 |
| LENOVOUSDT | Trading | $9k | $283k | 0.0000% | 0.41% | 25.0 | 2026-09-07 |
| HYUNDAIUSDT | Trading | $9k | $313k | 0.0000% | 0.17% | 25.0 | 2026-06-04 |
| POLYMARKETUSDT | PreLaunch | $8k | $242k | 0.0000% | -0.83% | 10.0 | 2026-09-23 |
| MAUSDT | Trading | $8k | $105k | 0.0000% | -1.17% | 25.0 | 2026-07-30 |
| XPUSDT | Trading | $8k | $48k | 0.0000% | 0.60% | 25.0 | 2026-10-30 |
| SAMSUNGEMUSDT | Trading | $8k | $118k | 0.0000% | -0.76% | 25.0 | 2026-08-14 |
| PBRUSDT | Trading | $8k | $61k | 0.0000% | -0.43% | 25.0 | 2026-10-12 |
| VRTXSTOCKUSDT | Trading | $8k | $28k | 0.0000% | 0.61% | 25.0 | 2026-08-05 |
| MPUSDT | Trading | $8k | $55k | 0.0000% | 0.22% | 25.0 | 2026-10-09 |
| JNJUSDT | Trading | $8k | $213k | 0.0000% | 0.82% | 25.0 | 2026-08-03 |
| WMTUSDT | Trading | $8k | $462k | 0.0000% | -0.68% | 50.0 | 2026-07-30 |
| TTMIUSDT | Trading | $8k | $39k | 0.0000% | 3.98% | 25.0 | 2026-10-26 |
| SNPSUSDT | Trading | $7k | $21k | 0.0000% | 1.29% | 25.0 | 2026-10-13 |
| EQTUSDT | Trading | $7k | $26k | 0.0000% | -1.60% | 25.0 | 2026-10-27 |
| VUSDT | Trading | $7k | $343k | 0.0000% | 0.03% | 25.0 | 2026-07-27 |
| ADIUSDT | Trading | $7k | $110k | 0.0000% | 0.39% | 25.0 | 2026-08-19 |
| GRALUSDT | Trading | $6k | $88k | 0.0000% | -1.74% | 25.0 | 2026-10-13 |
| HORIZONUSDT | Trading | $6k | $71k | 0.0000% | -1.55% | 25.0 | 2026-09-11 |
| LMTUSDT | Trading | $6k | $18k | 0.0000% | 0.03% | 25.0 | 2026-10-22 |
| GIGADEVICEUSDT | Trading | $5k | $1.1M | 0.0000% | 0.83% | 25.0 | 2026-07-28 |
| CRMLUSDT | Trading | $5k | $22k | 0.0000% | 2.60% | 25.0 | 2026-10-23 |
| SHOPUSDT | Trading | $5k | $217k | 0.0000% | -0.51% | 25.0 | 2026-08-05 |
| INTUUSDT | Trading | $5k | $263k | 0.0000% | -1.16% | 25.0 | 2026-08-25 |
| BYDUSDT | Trading | $5k | $219k | 0.0000% | 0.11% | 25.0 | 2026-09-07 |
| ABBVUSDT | Trading | $5k | $21k | 0.0000% | 0.23% | 25.0 | 2026-10-26 |
| SDGRUSDT | Trading | $4k | $19k | 0.0000% | -0.95% | 20.0 | 2026-09-22 |
| PGUSDT | Trading | $4k | $61k | 0.0000% | -0.73% | 25.0 | 2026-10-22 |
| BKNGUSDT | Trading | $4k | $41k | 0.0000% | -1.50% | 25.0 | 2026-10-30 |
| FORMSTOCKUSDT | Trading | $4k | $11k | 0.0000% | 1.45% | 25.0 | 2026-09-10 |
| BTDRUSDT | Trading | $3k | $39k | 0.0000% | -0.17% | 25.0 | 2026-09-10 |
| AXPUSDT | Trading | $3k | $38k | 0.0000% | -0.79% | 25.0 | 2026-10-23 |
| POPMARTUSDT | Trading | $3k | $571k | 0.0000% | -0.06% | 25.0 | 2026-07-28 |
| MELIUSDT | Trading | $3k | $81k | 0.0000% | 0.40% | 25.0 | 2026-10-12 |
| BAUSDT | Trading | $3k | $49k | 0.0000% | 0.56% | 25.0 | 2026-10-23 |
| NVOUSDT | Trading | $3k | $576k | 0.0000% | -0.20% | 25.0 | 2026-08-06 |
| MEITUANUSDT | Trading | $3k | $59k | 0.0000% | 0.00% | 25.0 | 2026-08-11 |
| TENCENTUSDT | Trading | $3k | $398k | 0.0000% | 0.07% | 25.0 | 2026-07-28 |
| CEGUSDT | Trading | $3k | $11k | 0.0000% | 0.66% | 25.0 | 2026-11-03 |
| NUSTOCKUSDT | Trading | $3k | $91k | 0.0000% | -0.50% | 25.0 | 2026-10-12 |
| SENSETIMEUSDT | Trading | $2k | $8k | 0.0000% | 2.98% | 25.0 | 2026-09-11 |
| SHEINUSDT | Trading | $2k | $319k | 0.0000% | 0.58% | 25.0 | 2026-08-27 |
| BIIBUSDT | Trading | $2k | $10k | 0.0000% | 0.04% | 25.0 | 2026-08-05 |
| AMZUUSDT | Trading | $2k | $25k | 0.0000% | -1.07% | 25.0 | 2026-09-18 |
| MOUSDT | Trading | $2k | $16k | 0.0000% | -0.30% | 25.0 | 2026-08-07 |
| MDTSTOCKUSDT | Trading | $2k | $24k | 0.0000% | -0.95% | 25.0 | 2026-10-23 |
| STMUSDT | Trading | $2k | $37k | 0.0000% | 1.28% | 25.0 | 2026-10-27 |
| PATHUSDT | Trading | $2k | $21k | 0.0000% | -1.52% | 25.0 | 2026-09-21 |
| ZSUSDT | Trading | $1k | $25k | 0.0000% | -2.88% | 25.0 | 2026-09-18 |
| NAVERUSDT | Trading | $1k | $94k | 0.0000% | 0.01% | 25.0 | 2026-08-14 |
| CVNAUSDT | Trading | $1k | $70k | 0.0000% | -0.47% | 25.0 | 2026-09-29 |
| OXYUSDT | Trading | $802 | $31k | 0.0000% | -0.46% | 25.0 | 2026-10-30 |
| LULUUSDT | Trading | $713 | $34k | 0.0000% | -0.37% | 25.0 | 2026-10-08 |
| KUAISHOUUSDT | Trading | $697 | $104k | 0.0000% | 0.27% | 25.0 | 2026-08-11 |
| MOONSHOTUSDT | PreLaunch | $570 | $231k | 0.0000% | 0.16% | 10.0 | 2026-08-07 |
| CSTOCKUSDT | Trading | $506 | $9k | 0.0000% | -0.76% | 25.0 | 2026-10-23 |
| OURAUSDT | PreLaunch | $373 | $677k | 0.0000% | 0.02% | 10.0 | 2026-09-22 |
| LAOPUUSDT | Trading | $138 | $71k | 0.0000% | 0.20% | 25.0 | 2026-09-02 |

**commodity** (4)

| symbol | status | turnoverUsd | oiUsd | fundingRate | price24hPcnt | maxLeverage | launchTime |
|---|---|---|---|---|---|---|---|
| CLUSDT | Trading | $45.2M | $31.2M | 0.0184% | 0.58% | 150.0 | 2026-03-24 |
| XAUUSDT | Trading | $29.6M | $127.0M | 0.0000% | 0.39% | 150.0 | 2026-03-09 |
| BZUSDT | Trading | $9.6M | $12.8M | 0.0000% | 0.58% | 100.0 | 2026-05-13 |
| XAGUSDT | Trading | $7.0M | $49.7M | 0.0000% | 0.66% | 150.0 | 2026-03-09 |

**forex** (4)

| symbol | status | turnoverUsd | oiUsd | fundingRate | price24hPcnt | maxLeverage | launchTime |
|---|---|---|---|---|---|---|---|
| EURUSDUSDT | Trading | $86k | $509k | 0.0000% | -0.01% | 100.0 | 2026-09-08 |
| USDJPYUSDT | Trading | $83k | $481k | -0.0126% | 0.08% | 100.0 | 2026-09-08 |
| GBPUSDUSDT | Trading | $35k | $424k | 0.0000% | -0.02% | 100.0 | 2026-09-08 |
| USDBRLUSDT | Trading | $13k | $120k | 0.0000% | 0.36% | 100.0 | 2026-11-03 |

## 3. Inverse

| symbol | contractType | deliveryTime | turnoverUsd | oiUsd | fundingRate | launchTime |
|---|---|---|---|---|---|---|
| ETHUSDZ26 | InverseFutures | 2026-12-25 | $136k | $5.4M | – | 2026-06-12 |
| BTCUSDZ26 | InverseFutures | 2026-12-25 | $86k | $7.4M | – | 2026-06-12 |
| ETHUSDH27 | InverseFutures | 2027-03-26 | $17k | $2.9M | – | 2026-09-11 |
| BTCUSDH27 | InverseFutures | 2027-03-26 | $5k | $3.2M | – | 2026-09-11 |
| BTCUSD | InversePerpetual | – | $45.5M | $773.4M | 0.0100% | 2018-11-14 |
| XRPUSD | InversePerpetual | – | $10.0M | $41.3M | -0.0092% | 2019-04-22 |
| ETHUSD | InversePerpetual | – | $9.3M | $152.1M | 0.0077% | 2019-01-01 |
| SOLUSD | InversePerpetual | – | $1.7M | $15.3M | 0.0034% | 2024-04-10 |
| ADAUSD | InversePerpetual | – | $719k | $3.0M | -0.0132% | 2022-03-24 |
| MNTUSD | InversePerpetual | – | $690k | $2.6M | 0.0100% | 2025-03-07 |
| DOTUSD | InversePerpetual | – | $248k | $3.9M | 0.0100% | 2021-10-13 |
| DOGEUSD | InversePerpetual | – | $198k | $1.9M | -0.0068% | 2025-03-05 |
| AAVEUSD | InversePerpetual | – | $179k | $1.2M | -0.0447% | 2025-03-07 |
| ETCUSD | InversePerpetual | – | $172k | $578k | 0.0100% | 2025-03-07 |
| SUIUSD | InversePerpetual | – | $168k | $2.0M | 0.0100% | 2025-03-06 |
| NEARUSD | InversePerpetual | – | $162k | $2.4M | 0.0100% | 2025-03-03 |
| XLMUSD | InversePerpetual | – | $123k | $1.0M | 0.0100% | 2025-03-06 |
| FILUSD | InversePerpetual | – | $99k | $386k | 0.0100% | 2025-03-05 |
| LTCUSD | InversePerpetual | – | $88k | $4.2M | 0.0100% | 2022-03-31 |
| UNIUSD | InversePerpetual | – | $58k | $2.9M | 0.0100% | 2025-03-06 |
| HYPEUSD | InversePerpetual | – | $38k | $919k | 0.0050% | 2026-07-07 |
| LINKUSD | InversePerpetual | – | $29k | $1.7M | 0.0100% | 2025-03-05 |
| BCHUSD | InversePerpetual | – | $11k | $1.7M | 0.0100% | 2025-03-05 |
| AVAXUSD | InversePerpetual | – | $10k | $2.2M | 0.0100% | 2025-03-05 |
| APTUSD | InversePerpetual | – | $6k | $437k | 0.0100% | 2025-03-07 |
| OPUSD | InversePerpetual | – | $6k | $295k | 0.0100% | 2025-03-06 |

## 4. Spot

| quoteCoin | pairs | t |
|---|---|---|
| USDT | 389 | $773.6M |
| USDC | 82 | $419.0M |
| EUR | 18 | $2.8M |
| BTC | 7 | $6.7M |
| AED | 4 | $18k |
| USD1 | 4 | $1.8M |
| USDE | 3 | $130k |
| RLUSD | 3 | $231k |
| USD | 3 | $165k |
| PLN | 3 | $90k |
| BRL | 2 | $354k |
| ETH | 1 | $48k |
| XUSD | 1 | $123 |
| SOL | 1 | $61k |

| symbolType | pairs | t |
|---|---|---|
| crypto | 507 | $1.20B |
| xstocks | 11 | $3.3M |
| adventure | 3 | $326k |

Top 40 spot pairs by USD turnover:

| rank | symbol | quoteCoin | symbolType | turnoverUsd | lastPrice |
|---|---|---|---|---|---|
| 1 | BTCUSDT | USDT |  | $256.4M | 75849.8 |
| 2 | BTCUSDC | USDC |  | $194.3M | 75772.0 |
| 3 | ETHUSDC | USDC |  | $179.3M | 2233.87 |
| 4 | USDCUSDT | USDT |  | $117.6M | 1.001 |
| 5 | ETHUSDT | USDT |  | $107.4M | 2236.39 |
| 6 | SOLUSDC | USDC |  | $39.6M | 89.67 |
| 7 | LINKUSDT | USDT |  | $25.7M | 10.905 |
| 8 | XRPUSDT | USDT |  | $24.0M | 1.1626 |
| 9 | SOLUSDT | USDT |  | $19.6M | 89.76 |
| 10 | HYPEUSDT | USDT |  | $14.4M | 77.594 |
| 11 | USD1USDT | USDT |  | $12.1M | 0.9991 |
| 12 | AAVEUSDT | USDT |  | $11.3M | 131.65 |
| 13 | ENAUSDT | USDT |  | $10.6M | 0.1585 |
| 14 | LITUSDT | USDT |  | $7.4M | 3.2217 |
| 15 | USDEUSDT | USDT |  | $6.3M | 1.0 |
| 16 | WLDUSDT | USDT |  | $6.1M | 0.3361 |
| 17 | BIRBUSDT | USDT |  | $5.7M | 0.09403 |
| 18 | PEPEUSDT | USDT |  | $5.6M | 4.879e-06 |
| 19 | MNTUSDT | USDT |  | $5.1M | 0.4528 |
| 20 | RLUSDUSDT | USDT |  | $4.8M | 1.0012 |
| 21 | ICNTUSDT | USDT |  | $4.7M | 0.313 |
| 22 | XRPBTC | BTC |  | $4.6M | 1.532e-05 |
| 23 | NEARUSDT | USDT |  | $4.6M | 3.662 |
| 24 | ASTERUSDT | USDT |  | $4.4M | 0.6782 |
| 25 | CAPUSDT | USDT |  | $4.1M | 0.10323 |
| 26 | XAUTUSDT | USDT |  | $3.4M | 4061.88 |
| 27 | KORUSDT | USDT |  | $3.3M | 0.002096 |
| 28 | PUMPUSDT | USDT |  | $3.1M | 0.003657 |
| 29 | CCUSDT | USDT |  | $2.9M | 0.09517 |
| 30 | UNIUSDT | USDT |  | $2.6M | 7.093 |
| 31 | ZIGUSDT | USDT |  | $2.5M | 0.0391 |
| 32 | XRPUSDC | USDC |  | $2.4M | 1.1612 |
| 33 | BNBUSDT | USDT |  | $2.4M | 694.93 |
| 34 | PIEVERSEUSDT | USDT |  | $1.9M | 1.404 |
| 35 | LTCUSDT | USDT |  | $1.9M | 52.31 |
| 36 | GRAMUSDT | USDT |  | $1.9M | 1.082 |
| 37 | DATAUSDT | USDT |  | $1.8M | 0.1153 |
| 38 | GRASSUSDT | USDT |  | $1.8M | 0.5101 |
| 39 | SPCXXUSDT | USDT | xstocks | $1.8M | 134.7 |
| 40 | ELSAUSDT | USDT |  | $1.7M | 0.07626 |

## 5. Options

| baseCoin | contracts | expiries | strikes | next | last | oi_contracts | oiUsd | turnoverUsd |
|---|---|---|---|---|---|---|---|---|
| BTC | 664 | 11 | 90 | 2026-11-08 08:00 | 2027-09-24 | 16,974.6 | $1.30B | $410.5M |
| ETH | 582 | 11 | 94 | 2026-11-08 08:00 | 2027-09-24 | 288,205.7 | $647.3M | $167.7M |
| XAUT | 368 | 7 | 56 | 2026-11-08 08:00 | 2026-12-25 | 2,756.7 | $11.2M | $6.5M |
| SOL | 312 | 7 | 52 | 2026-11-08 08:00 | 2026-12-25 | 98,610.0 | $8.9M | $4.1M |
| XRP | 276 | 7 | 40 | 2026-11-08 08:00 | 2026-12-25 | 2,145,830.0 | $2.5M | $287k |
| HYPE | 356 | 7 | 53 | 2026-11-08 08:00 | 2026-12-25 | 30,433.0 | $2.4M | $330k |
| NVDA | 264 | 5 | 52 | 2026-11-09 21:00 | 2026-11-20 | 9,783.0 | $1.8M | $241k |
| QQQ | 508 | 5 | 102 | 2026-11-09 21:00 | 2026-11-20 | 2,274.0 | $1.5M | $108k |
| SPCX | 324 | 5 | 75 | 2026-11-09 21:00 | 2026-11-20 | 6,328.0 | $852k | $102k |
| TSLA | 370 | 5 | 60 | 2026-11-09 21:00 | 2026-11-20 | 1,598.0 | $573k | $70k |
| DOGE | 212 | 7 | 31 | 2026-11-08 08:00 | 2026-12-25 | 6,420,000.0 | $462k | $66k |
| SNDK | 774 | 5 | 195 | 2026-11-09 21:00 | 2026-11-20 | 258.8 | $216k | $54k |
| MU | 636 | 5 | 139 | 2026-11-09 21:00 | 2026-11-20 | 276.4 | $191k | $13k |
| SKHY | 296 | 5 | 55 | 2026-11-09 21:00 | 2026-11-20 | 1,216.0 | $157k | $2k |
| SOXL | 648 | 5 | 161 | 2026-11-09 21:00 | 2026-11-20 | 3,190.0 | $145k | $122k |
| MNT | 282 | 7 | 43 | 2026-11-08 08:00 | 2026-12-25 | 259,980.0 | $118k | $11k |

Next three expiries per underlying (OI USD):

- **BTC**: 08Nov26 (72 contracts, OI $172.9M); 09Nov26 (50 contracts, OI $51.3M); 10Nov26 (44 contracts, OI $5.7M)
- **ETH**: 08Nov26 (74 contracts, OI $66.6M); 09Nov26 (50 contracts, OI $30.8M); 10Nov26 (46 contracts, OI $6.4M)
- **XAUT**: 08Nov26 (60 contracts, OI $4.3M); 09Nov26 (54 contracts, OI $1.1M); 10Nov26 (54 contracts, OI $197k)
- **SOL**: 08Nov26 (46 contracts, OI $1.6M); 09Nov26 (32 contracts, OI $237k); 10Nov26 (32 contracts, OI $67k)
- **XRP**: 08Nov26 (44 contracts, OI $260k); 09Nov26 (34 contracts, OI $48k); 10Nov26 (28 contracts, OI $4k)
- **HYPE**: 08Nov26 (56 contracts, OI $116k); 09Nov26 (40 contracts, OI $40k); 10Nov26 (36 contracts, OI $4k)
- **NVDA**: 09Nov26 (54 contracts, OI $44k); 10Nov26 (50 contracts, OI $376); 11Nov26 (46 contracts, OI $0)
- **QQQ**: 09Nov26 (114 contracts, OI $13k); 10Nov26 (120 contracts, OI $0); 11Nov26 (118 contracts, OI $0)
- **SPCX**: 09Nov26 (60 contracts, OI $21k); 10Nov26 (50 contracts, OI $0); 11Nov26 (48 contracts, OI $0)
- **TSLA**: 09Nov26 (72 contracts, OI $20k); 10Nov26 (76 contracts, OI $0); 11Nov26 (74 contracts, OI $0)
- **DOGE**: 08Nov26 (36 contracts, OI $213k); 09Nov26 (22 contracts, OI $53k); 10Nov26 (22 contracts, OI $39k)
- **SNDK**: 09Nov26 (160 contracts, OI $59k); 10Nov26 (174 contracts, OI $0); 11Nov26 (174 contracts, OI $0)
- **MU**: 09Nov26 (116 contracts, OI $20k); 10Nov26 (120 contracts, OI $0); 11Nov26 (122 contracts, OI $0)
- **SKHY**: 09Nov26 (62 contracts, OI $5k); 10Nov26 (52 contracts, OI $129); 11Nov26 (54 contracts, OI $0)
- **SOXL**: 09Nov26 (106 contracts, OI $9k); 10Nov26 (136 contracts, OI $2k); 11Nov26 (134 contracts, OI $0)
- **MNT**: 08Nov26 (32 contracts, OI $14k); 09Nov26 (24 contracts, OI $231); 10Nov26 (26 contracts, OI $1k)

## 6. Cross-market coverage — top 60 crypto bases by linear turnover

| # | base | linear 24h T/O | USDT perp | USDC perp | inv perp | dated fut | spot USDT | spot USDC | options | spot/perp T/O |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | BTC | $1.18B | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | 0.22 |
| 2 | ETH | $641.8M | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | 0.17 |
| 3 | ZEC | $304.0M | ✓ | · | · | · | · | · | · | 0.00 |
| 4 | XRP | $247.1M | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | 0.10 |
| 5 | SOL | $166.3M | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | 0.12 |
| 6 | AKE | $155.5M | ✓ | · | · | · | · | · | · | 0.00 |
| 7 | HYPE | $74.0M | ✓ | ✓ | ✓ | ✓ | ✓ | · | ✓ | 0.20 |
| 8 | NEAR | $72.8M | ✓ | · | ✓ | · | ✓ | ✓ | · | 0.06 |
| 9 | PEPE | $56.9M | ✓ | ✓ | · | · | ✓ | ✓ | · | 0.10 |
| 10 | UAI | $44.4M | ✓ | · | · | · | · | · | · | 0.00 |
| 11 | ICNT | $41.9M | ✓ | ✓ | · | · | ✓ | · | · | 0.11 |
| 12 | ENA | $37.8M | ✓ | ✓ | · | · | ✓ | ✓ | · | 0.28 |
| 13 | LIT | $28.5M | ✓ | · | · | · | ✓ | · | · | 0.26 |
| 14 | DOGE | $25.9M | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | 0.06 |
| 15 | PUMP | $25.5M | ✓ | ✓ | · | · | ✓ | ✓ | · | 0.12 |
| 16 | SUI | $23.7M | ✓ | ✓ | ✓ | · | ✓ | ✓ | · | 0.06 |
| 17 | WLD | $23.2M | ✓ | ✓ | · | · | ✓ | ✓ | · | 0.26 |
| 18 | JCT | $21.2M | ✓ | · | · | · | · | · | · | 0.00 |
| 19 | BIRB | $20.7M | ✓ | · | · | · | ✓ | · | · | 0.28 |
| 20 | ADA | $18.6M | ✓ | · | ✓ | · | ✓ | ✓ | · | 0.06 |
| 21 | TRUMP | $18.6M | ✓ | ✓ | · | · | ✓ | ✓ | · | 0.09 |
| 22 | INJ | $17.8M | ✓ | ✓ | · | · | ✓ | ✓ | · | 0.07 |
| 23 | UNI | $16.8M | ✓ | ✓ | ✓ | · | ✓ | ✓ | · | 0.16 |
| 24 | LYN | $16.4M | ✓ | · | · | · | · | · | · | 0.00 |
| 25 | CYBER | $15.9M | ✓ | · | · | · | ✓ | · | · | 0.07 |
| 26 | DGAI | $14.9M | ✓ | · | · | · | · | · | · | 0.00 |
| 27 | LINK | $14.9M | ✓ | ✓ | ✓ | · | ✓ | ✓ | · | 1.73 |
| 28 | ARIA | $14.8M | ✓ | · | · | · | · | · | · | 0.00 |
| 29 | XMR | $14.5M | ✓ | ✓ | · | · | · | · | · | 0.00 |
| 30 | ACT | $14.3M | ✓ | · | · | · | · | · | · | 0.00 |
| 31 | AAVE | $14.2M | ✓ | ✓ | ✓ | · | ✓ | ✓ | · | 0.80 |
| 32 | STRK | $12.5M | ✓ | ✓ | · | · | ✓ | ✓ | · | 0.07 |
| 33 | BCH | $12.5M | ✓ | ✓ | ✓ | · | ✓ | ✓ | · | 0.13 |
| 34 | CAP | $12.5M | ✓ | · | · | · | ✓ | · | · | 0.33 |
| 35 | ESPORTS | $11.5M | ✓ | · | · | · | · | · | · | 0.00 |
| 36 | ARB | $11.0M | ✓ | ✓ | · | · | ✓ | ✓ | · | 0.12 |
| 37 | TIA | $10.8M | ✓ | ✓ | · | · | ✓ | ✓ | · | 0.05 |
| 38 | BNB | $10.2M | ✓ | ✓ | · | · | ✓ | ✓ | · | 0.23 |
| 39 | ONDO | $10.0M | ✓ | ✓ | · | · | ✓ | ✓ | · | 0.16 |
| 40 | XLM | $9.5M | ✓ | ✓ | ✓ | · | ✓ | ✓ | · | 0.12 |
| 41 | QNT | $9.4M | ✓ | · | · | · | ✓ | · | · | 0.15 |
| 42 | HBAR | $9.3M | ✓ | ✓ | · | · | ✓ | ✓ | · | 0.17 |
| 43 | AVAX | $8.9M | ✓ | · | ✓ | · | ✓ | ✓ | · | 0.15 |
| 44 | BTW | $8.7M | ✓ | · | · | · | · | · | · | 0.00 |
| 45 | CT | $8.7M | ✓ | · | · | · | · | · | · | 0.00 |
| 46 | TAO | $8.6M | ✓ | ✓ | · | · | · | · | · | 0.00 |
| 47 | LTC | $8.0M | ✓ | ✓ | ✓ | · | ✓ | ✓ | · | 0.24 |
| 48 | XAUT | $7.7M | ✓ | ✓ | · | ✓ | ✓ | · | ✓ | 0.44 |
| 49 | PONS | $7.1M | ✓ | · | · | · | ✓ | · | · | 0.16 |
| 50 | XPL | $6.9M | ✓ | ✓ | · | · | ✓ | · | · | 0.10 |
| 51 | ZRO | $6.6M | ✓ | · | · | · | ✓ | ✓ | · | 0.21 |
| 52 | CARV | $6.4M | ✓ | · | · | · | ✓ | · | · | 0.15 |
| 53 | LSK | $6.2M | ✓ | · | · | · | · | · | · | 0.00 |
| 54 | STX | $6.1M | ✓ | · | · | · | ✓ | ✓ | · | 0.11 |
| 55 | WLFI | $6.0M | ✓ | ✓ | · | · | ✓ | · | · | 0.13 |
| 56 | DOT | $5.9M | ✓ | ✓ | ✓ | · | ✓ | ✓ | · | 0.13 |
| 57 | TRX | $5.7M | ✓ | ✓ | · | · | ✓ | ✓ | · | 0.27 |
| 58 | FARTCOIN | $5.7M | ✓ | ✓ | · | · | · | · | · | 0.00 |
| 59 | OP | $5.6M | ✓ | ✓ | ✓ | · | ✓ | ✓ | · | 0.07 |
| 60 | ASTER | $5.1M | ✓ | ✓ | · | · | ✓ | · | · | 0.86 |

45/60 have both USDT perp and USDT spot (spot-vs-perp flow studies); 7 have dated futures (basis term structure); 7 have options; 18 have inverse perps (coin-margined vs USDT-margined flow).

## 7. Liquidity tiers for a 3-second order-flow collector (30 GB budget)

Ranking uses the **7-day average daily turnover** (7 complete daily klines up to 2026-11-07 00:00 UTC) rather than the Saturday 24h snapshot, which understates weekday activity. Stablecoin perps (USDCUSDT etc.) excluded.

Rules: **Tier A** = top 20 crypto USDT perps (by 7d avg turnover) that have a USDT/USDC spot pair with ≥ $3M 7d-avg turnover; spot leg = the more liquid of USDT/USDC spot, plus the second one if it also does ≥ $20M/day. **Tier B** = next 40 USDT perps, any symbolType (perp L2 + trades). **Tier C** = remaining Trading USDT perps with ≥ $20M (7d avg or 24h).

- Tier A: 20 perps + 23 spot books; perp 7d-avg turnover $8.56B/day, spot $1.53B/day.
- Tier B: 7d-avg turnover $2.64B/day; range $17.9M – $615.7M.
- Tier C: 2 symbols ≥ $20M ($31.5M/day). The $20M floor is already crossed inside Tier B (Tier B minimum $17.9M), so Tier C is small; an extended Tier C′ (trades only, ≥ $5M/day) would add 61 symbols ($581.8M/day, ≈ 0.01 GB/day).
- Coverage: A+B+C = $11.24B of $12.16B non-stable USDT-perp 7d-avg turnover.

Storage estimate (assumptions: 12 MB/day per book for 50-level L2 at 3 s, 25 B/trade compressed, mean trade $1,500): A ≈ 0.68 GB/day, B ≈ 0.52 GB/day, C ≈ 0.00 GB/day → **1.21 GB/day, ~25 days** to fill 30 GB (~25 days with C′). For a longer horizon, cut Tier B L2 to 20 levels or 15 s first; trades are the cheaper, higher-value stream.

| tier | perp | 7d avg perp T/O | 24h perp T/O | spot leg(s) | 7d avg spot T/O |
|---|---|---|---|---|---|
| A | BTCUSDT | $4.54B | $1.16B | BTCUSDT, BTCUSDC | $910.2M |
| A | ETHUSDT | $2.09B | $634.2M | ETHUSDT, ETHUSDC | $291.7M |
| A | SOLUSDT | $473.9M | $163.1M | SOLUSDT, SOLUSDC | $88.2M |
| A | XRPUSDT | $281.7M | $243.5M | XRPUSDT | $35.8M |
| A | HYPEUSDT | $271.8M | $72.1M | HYPEUSDT | $42.5M |
| A | NEARUSDT | $199.5M | $72.8M | NEARUSDT | $16.7M |
| A | 1000PEPEUSDT | $104.1M | $56.7M | PEPEUSDT | $16.6M |
| A | ENAUSDT | $84.8M | $37.8M | ENAUSDT | $20.0M |
| A | DOGEUSDT | $72.9M | $25.0M | DOGEUSDT | $5.0M |
| A | SUIUSDT | $54.9M | $23.7M | SUIUSDT | $6.6M |
| A | TRUMPUSDT | $54.1M | $18.6M | TRUMPUSDT | $6.1M |
| A | WLDUSDT | $46.7M | $23.1M | WLDUSDT | $7.3M |
| A | UNIUSDT | $46.3M | $16.7M | UNIUSDT | $9.2M |
| A | LINKUSDT | $38.1M | $14.8M | LINKUSDT | $20.4M |
| A | BNBUSDT | $37.7M | $10.2M | BNBUSDT | $8.2M |
| A | AAVEUSDT | $37.6M | $14.2M | AAVEUSDT | $15.7M |
| A | LITUSDT | $36.3M | $28.5M | LITUSDT | $12.3M |
| A | XAUTUSDT | $32.8M | $7.7M | XAUTUSDT | $11.0M |
| A | BEAMUSDT | $31.4M | $4.6M | BEAMUSDT | $3.3M |
| A | PUMPFUNUSDT | $30.8M | $25.5M | PUMPUSDT | $5.5M |
| B | SOXLUSDT (ETF) | $615.7M | $182.5M | – | – |
| B | ZECUSDT | $395.2M | $304.0M | – | – |
| B | XAUUSDT (commodity) | $148.0M | $29.6M | – | – |
| B | CLUSDT (commodity) | $146.7M | $45.2M | – | – |
| B | KORUUSDT (ETF) | $100.2M | $18.2M | – | – |
| B | SNDKUSDT (stock) | $91.0M | $43.0M | – | – |
| B | AKEUSDT | $72.0M | $155.5M | – | – |
| B | SKHYNIXUSDT (stock) | $70.1M | $11.3M | – | – |
| B | SOXSUSDT (ETF) | $62.7M | $27.2M | – | – |
| B | MUUSDT (stock) | $60.9M | $30.0M | – | – |
| B | SPCXUSDT (stock) | $60.2M | $22.3M | – | – |
| B | XAGUSDT (commodity) | $45.2M | $7.0M | – | – |
| B | XOMUSDT (stock) | $42.5M | $776k | – | – |
| B | ADAUSDT | $40.4M | $18.6M | – | – |
| B | MSTRUSDT (stock) | $39.2M | $6.0M | – | – |
| B | ESPORTSUSDT | $38.5M | $11.5M | – | – |
| B | SKHYUSDT (stock) | $36.4M | $7.9M | – | – |
| B | QQQUSDT (ETF) | $34.0M | $7.5M | – | – |
| B | INTCUSDT (stock) | $34.0M | $11.4M | – | – |
| B | XMRUSDT | $32.3M | $14.5M | – | – |
| B | JCTUSDT | $32.2M | $21.2M | – | – |
| B | LSKUSDT | $29.2M | $6.2M | – | – |
| B | ARIAUSDT | $28.9M | $14.8M | – | – |
| B | TRXUSDT | $28.0M | $5.7M | – | – |
| B | ONDOUSDT | $27.3M | $10.0M | – | – |
| B | CLANKERUSDT | $25.9M | $4.2M | – | – |
| B | XPLUSDT | $25.2M | $6.8M | – | – |
| B | AVAXUSDT | $24.9M | $8.9M | – | – |
| B | NVDAUSDT (stock) | $23.8M | $8.0M | – | – |
| B | SNXXUSDT (ETF) | $23.5M | $7.3M | – | – |
| B | TSLAUSDT (stock) | $23.4M | $3.6M | – | – |
| B | INJUSDT | $22.9M | $17.7M | – | – |
| B | BZUSDT (commodity) | $22.7M | $9.6M | – | – |
| B | UAIUSDT | $22.6M | $44.4M | – | – |
| B | ZROUSDT | $22.3M | $6.6M | – | – |
| B | GOOGLUSDT (stock) | $21.6M | $3.9M | – | – |
| B | AAPLUSDT (stock) | $19.7M | $2.0M | – | – |
| B | DRAMUSDT (ETF) | $18.8M | $2.8M | – | – |
| B | ARBUSDT | $18.6M | $11.0M | – | – |
| B | TAOUSDT | $17.9M | $8.6M | – | – |
| C | BIRBUSDT | $17.2M | $20.7M | – | – |
| C | ICNTUSDT | $14.3M | $41.8M | – | – |

**Tier A**: BTCUSDT + BTCUSDT/BTCUSDC, ETHUSDT + ETHUSDT/ETHUSDC, SOLUSDT + SOLUSDT/SOLUSDC, XRPUSDT + XRPUSDT, HYPEUSDT + HYPEUSDT, NEARUSDT + NEARUSDT, 1000PEPEUSDT + PEPEUSDT, ENAUSDT + ENAUSDT, DOGEUSDT + DOGEUSDT, SUIUSDT + SUIUSDT, TRUMPUSDT + TRUMPUSDT, WLDUSDT + WLDUSDT, UNIUSDT + UNIUSDT, LINKUSDT + LINKUSDT, BNBUSDT + BNBUSDT, AAVEUSDT + AAVEUSDT, LITUSDT + LITUSDT, XAUTUSDT + XAUTUSDT, BEAMUSDT + BEAMUSDT, PUMPFUNUSDT + PUMPUSDT

**Tier B**: SOXLUSDT, ZECUSDT, XAUUSDT, CLUSDT, KORUUSDT, SNDKUSDT, AKEUSDT, SKHYNIXUSDT, SOXSUSDT, MUUSDT, SPCXUSDT, XAGUSDT, XOMUSDT, ADAUSDT, MSTRUSDT, ESPORTSUSDT, SKHYUSDT, QQQUSDT, INTCUSDT, XMRUSDT, JCTUSDT, LSKUSDT, ARIAUSDT, TRXUSDT, ONDOUSDT, CLANKERUSDT, XPLUSDT, AVAXUSDT, NVDAUSDT, SNXXUSDT, TSLAUSDT, INJUSDT, BZUSDT, UAIUSDT, ZROUSDT, GOOGLUSDT, AAPLUSDT, DRAMUSDT, ARBUSDT, TAOUSDT

**Tier C**: BIRBUSDT, ICNTUSDT

**Tier C′ (≥ $5M, optional)**: XLMUSDT, STRKUSDT, QUSDT, LTCUSDT, NBISUSDT, DGAIUSDT, BCHUSDT, CRCLUSDT, WLFIUSDT, TQQQUSDT, TIAUSDT, SAMSUNGUSDT, FARTCOINUSDT, MRVLUSDT, QNTUSDT, CAPUSDT, ELSAUSDT, HBARUSDT, VVVUSDT, BTWUSDT, OPUSDT, EVAAUSDT, LYNUSDT, PONSUSDT, STXUSDT, SPYUSDT, DOTUSDT, USELESSUSDT, APTUSDT, MAGMAUSDT, LITEUSDT, CASHCATUSDT, AEROUSDT, SOPHUSDT, APRUSDT, MUUUSDT, EWYUSDT, PENGUUSDT, ASTERUSDT, AMDSTOCKUSDT, GRAMUSDT, IRYSUSDT, BMTUSDT, RUNEUSDT, MNTUSDT, CRWVUSDT, ATOMUSDT, CYBERUSDT, INITUSDT, GRASSUSDT, CRVUSDT, METAUSDT, AXISUSDT, MSFTUSDT, FILUSDT, WIFUSDT, VIRTUALUSDT, HUMAUSDT, SANDUSDT, PAXGUSDT, DASHUSDT

**New listings (< 30 days, launched after 2026-10-08): 46 USDT perps** — volume on fresh listings decays fast, treat tier membership as provisional:

| symbol | status | symbolType | launchTime | turnoverUsd | turnover7dAvgUsd | oiUsd | fundingRate | tier |
|---|---|---|---|---|---|---|---|---|
| AXISUSDT | Trading | innovation | 2026-10-21 | $1.7M | $5.9M | $1.8M | 0.0050% | C′ |
| DRVUSDT | Trading |  | 2026-10-16 | $1.3M | $1.6M | $865k | 0.0050% | – |
| DNAUSDT | Trading | innovation | 2026-10-16 | $638k | $1.3M | $963k | 0.0383% | – |
| PFEUSDT | Trading | stock | 2026-11-03 | $81k | – | $113k | 0.0000% | – |
| PREDICTUSDT | PreLaunch |  | 2026-10-26 | $68k | – | $936k | 0.0050% | – |
| DUOLUSDT | Trading | stock | 2026-11-06 | $66k | – | $19k | 0.0000% | – |
| FNUSDT | Trading | stock | 2026-11-06 | $65k | – | $31k | 0.0000% | – |
| SMRUSDT | Trading | stock | 2026-10-30 | $48k | – | $59k | 0.0000% | – |
| IBKRUSDT | Trading | stock | 2026-10-23 | $43k | – | $31k | 0.0000% | – |
| BWETUSDT | Trading | ETF | 2026-10-09 | $41k | – | $158k | 0.0000% | – |
| USDEXUSDT | Trading | stock | 2026-10-09 | $36k | – | $237k | 0.0000% | – |
| FISVUSDT | Trading | stock | 2026-11-06 | $32k | – | $14k | 0.0000% | – |
| MXLUSDT | Trading | stock | 2026-10-22 | $27k | – | $20k | 0.0000% | – |
| VKTXUSDT | Trading | stock | 2026-10-09 | $24k | – | $20k | 0.0000% | – |
| AGPUUSDT | Trading | stock | 2026-10-26 | $23k | – | $107k | 0.0000% | – |
| UMCUSDT | Trading | stock | 2026-10-09 | $20k | – | $26k | 0.0000% | – |
| ZCSHUSDT | Trading | ETF | 2026-10-13 | $18k | – | $100k | 0.0000% | – |
| USDBRLUSDT | Trading | forex | 2026-11-03 | $13k | – | $120k | 0.0000% | – |
| VISTUSDT | Trading | stock | 2026-10-12 | $11k | – | $49k | 0.0000% | – |
| YPFUSDT | Trading | stock | 2026-10-12 | $11k | – | $77k | 0.0000% | – |
| EIDOUSDT | Trading | stock | 2026-10-27 | $10k | – | $38k | 0.0000% | – |
| AKAMUSDT | Trading | stock | 2026-11-03 | $9k | – | $46k | 0.0000% | – |
| XPUSDT | Trading | stock | 2026-10-30 | $8k | – | $48k | 0.0000% | – |
| XLIUSDT | Trading | ETF | 2026-10-30 | $8k | – | $11k | 0.0000% | – |
| PBRUSDT | Trading | stock | 2026-10-12 | $8k | – | $61k | 0.0000% | – |
| MPUSDT | Trading | stock | 2026-10-09 | $8k | – | $55k | 0.0000% | – |
| TTMIUSDT | Trading | stock | 2026-10-26 | $8k | – | $39k | 0.0000% | – |
| SNPSUSDT | Trading | stock | 2026-10-13 | $7k | – | $21k | 0.0000% | – |
| EQTUSDT | Trading | stock | 2026-10-27 | $7k | – | $26k | 0.0000% | – |
| GRALUSDT | Trading | stock | 2026-10-13 | $6k | – | $88k | 0.0000% | – |
| LMTUSDT | Trading | stock | 2026-10-22 | $6k | – | $18k | 0.0000% | – |
| CRMLUSDT | Trading | stock | 2026-10-23 | $5k | – | $22k | 0.0000% | – |
| ABBVUSDT | Trading | stock | 2026-10-26 | $5k | – | $21k | 0.0000% | – |
| PGUSDT | Trading | stock | 2026-10-22 | $4k | – | $61k | 0.0000% | – |
| BKNGUSDT | Trading | stock | 2026-10-30 | $4k | – | $41k | 0.0000% | – |
| BMNUUSDT | Trading | ETF | 2026-10-09 | $3k | – | $40k | 0.0000% | – |
| AXPUSDT | Trading | stock | 2026-10-23 | $3k | – | $38k | 0.0000% | – |
| MELIUSDT | Trading | stock | 2026-10-12 | $3k | – | $81k | 0.0000% | – |
| BAUSDT | Trading | stock | 2026-10-23 | $3k | – | $49k | 0.0000% | – |
| CEGUSDT | Trading | stock | 2026-11-03 | $3k | – | $11k | 0.0000% | – |
| NUSTOCKUSDT | Trading | stock | 2026-10-12 | $3k | – | $91k | 0.0000% | – |
| EWZSUSDT | Trading | ETF | 2026-10-30 | $2k | – | $25k | 0.0000% | – |
| MDTSTOCKUSDT | Trading | stock | 2026-10-23 | $2k | – | $24k | 0.0000% | – |
| STMUSDT | Trading | stock | 2026-10-27 | $2k | – | $37k | 0.0000% | – |
| OXYUSDT | Trading | stock | 2026-10-30 | $802 | – | $31k | 0.0000% | – |
| CSTOCKUSDT | Trading | stock | 2026-10-23 | $506 | – | $9k | 0.0000% | – |


## 8. Where to look next

**Pre-listing perps (`isPreListing=true`, 7)** — perps on assets with no spot market (incl. private companies):

| symbol | status | symbolType | launchTime | lastPrice | turnoverUsd | oiUsd | fundingRate |
|---|---|---|---|---|---|---|---|
| ANTHROPICUSDT | PreLaunch | stock | 2026-07-13 | 1391.59 | $3.3M | $2.9M | 0.0000% |
| OPENAIUSDT | PreLaunch | stock | 2026-07-13 | 1305.0 | $741k | $893k | 0.0000% |
| PREDICTUSDT | PreLaunch |  | 2026-10-26 | 1.2135 | $68k | $936k | 0.0050% |
| BPUSDT | PreLaunch |  | 2026-03-16 | 0.95179 | $36k | $761k | 0.0050% |
| POLYMARKETUSDT | PreLaunch | stock | 2026-09-23 | 23.96 | $8k | $242k | 0.0000% |
| MOONSHOTUSDT | PreLaunch | stock | 2026-08-07 | 62.93 | $570 | $231k | 0.0000% |
| OURAUSDT | PreLaunch | stock | 2026-09-22 | 48.46 | $373 | $677k | 0.0000% |

**Extreme funding (crypto USDT perps with ≥ $5M 24h turnover, |annualised| ≥ 20%):**

| symbol | fundingRate | fundingInterval | fundAnn | turnoverUsd | oiUsd |
|---|---|---|---|---|---|
| UAIUSDT | -0.0993% | 4h | -217% | $44.4M | $2.9M |
| LSKUSDT | -0.0975% | 4h | -214% | $6.2M | $3.6M |
| ACTUSDT | -0.0640% | 4h | -140% | $14.3M | $1.6M |
| CARVUSDT | -0.0541% | 4h | -118% | $6.4M | $1.1M |
| CYBERUSDT | -0.0408% | 4h | -89% | $15.9M | $2.6M |
| WLFIUSDT | -0.0141% | 4h | -31% | $6.0M | $25.4M |
| 1000PEPEUSDT | -0.0210% | 8h | -23% | $56.7M | $56.0M |
| ARIAUSDT | 0.1522% | 4h | 333% | $14.8M | $6.9M |

**Highest OI / 24h turnover (≥ $5M turnover)** — crowded, sticky positioning:

| symbol | oiToT | oiUsd | turnoverUsd | fundingRate |
|---|---|---|---|---|
| XAUTUSDT | 19.47 | $149.4M | $7.7M | -0.0053% |
| TRXUSDT | 10.59 | $59.9M | $5.7M | -0.0016% |
| ASTERUSDT | 9.29 | $47.3M | $5.1M | 0.0042% |
| BNBUSDT | 9.09 | $92.6M | $10.2M | -0.0174% |
| LTCUSDT | 7.28 | $57.9M | $8.0M | 0.0049% |
| AVAXUSDT | 6.30 | $56.0M | $8.9M | 0.0100% |
| BTCUSDT | 4.54 | $5.27B | $1.16B | 0.0100% |
| DOGEUSDT | 4.34 | $108.6M | $25.0M | -0.0065% |
| WLFIUSDT | 4.26 | $25.4M | $6.0M | -0.0141% |
| ONDOUSDT | 4.26 | $42.5M | $10.0M | 0.0019% |

**Spot-only (no perp) — top by turnover:** KORUSDT $3.3M, ZIGUSDT $2.5M, METHUSDT $846k, DEGENUSDT $626k, FETUSDT $577k, GROVEUSDT $394k, PYUSDUSDT $374k, STETHUSDT $358k, LAVAUSDT $310k, NYMUSDT $227k

**Perp-only crypto (no Bybit spot in any quote) — 217 perps; top:** ZECUSDT $304.0M, AKEUSDT $155.5M, UAIUSDT $44.4M, JCTUSDT $21.2M, LYNUSDT $16.4M, DGAIUSDT $14.9M, ARIAUSDT $14.8M, XMRUSDT $14.5M, ACTUSDT $14.3M, ESPORTSUSDT $11.5M, BTWUSDT $8.7M, CTUSDT $8.7M, TAOUSDT $8.6M, LSKUSDT $6.2M, FARTCOINUSDT $5.7M

**Tokenised-stock spot (xStocks) vs stock perps** — same underlying, two venues/instruments: NVDAXUSDT ($547k, 188.53) ↔ NVDAUSDT ($8.0M, 188.06); COINXUSDT ($242k, 134.61) ↔ COINUSDT ($747k, 134.48); AAPLXUSDT ($60k, 343.86) ↔ AAPLUSDT ($2.0M, 342.55); CRCLXUSDT ($260k, 67.37) ↔ CRCLUSDT ($4.1M, 67.41); METAXUSDT ($21k, 613.98) ↔ METAUSDT ($709k, 612.65); HOODXUSDT ($62k, 80.64) ↔ HOODUSDT ($1.1M, 80.75); AMZNXUSDT ($34k, 259.49) ↔ AMZNUSDT ($461k, 259.83); GOOGLXUSDT ($75k, 285.5) ↔ GOOGLUSDT ($3.9M, 285.03); MCDXUSDT ($6k, 273.47) ↔ MCDUSDT ($41k, 276.7); TSLAXUSDT ($193k, 358.28) ↔ TSLAUSDT ($3.6M, 358.39); SPCXXUSDT ($1.8M, 134.7) ↔ SPCXUSDT ($22.3M, 134.77)

**Equity/ETF/commodity options:** non-crypto option underlyings XAUT, NVDA, QQQ, SPCX, TSLA, SNDK, MU, SKHY, SOXL — combine with the matching TradFi perp for vol/funding/basis studies outside crypto.
