# Data Sources

_Catalogue (machine-readable): [config/datasources.yaml](../config/datasources.yaml). This page explains why each source is there and its point-in-time (PIT) risks. Status as of 2026-11-07; connectivity is verified with `atx check`._

## How to read the PIT column

- **native**: stamped at publication and never revised. Safe for historical validation.
- **vintage**: the source keeps its revision history (ALFRED, Glassnode PIT). Each vintage is stored as its own row, which makes it truly point-in-time.
- **revisable**: values can change silently after the fact (back-fills, re-normalisation, entity re-labelling). History from the vendor is **not** point-in-time. Only our own live archive, starting from the day we begin ingesting, is trustworthy. Tools built on these sources must be validated on self-archived data only, or treated as forward-only.

## P0: build now (free)

| Source | What | PIT | Status | Why it matters |
|---|---|---|---|---|
| **Bybit klines** (last/mark/index/premium) | OHLCV since listing | native | ✅ | price, vol, structure; mark/index for stops and basis |
| **Bybit funding** | realised funding | native | ✅ | positioning / crowding (`flow.derivs`) |
| **Bybit open interest** | OI 5 min–1 d | native | ✅ | leverage build-up, squeeze risk |
| **Bybit long/short ratio** | account ratio | native | ✅ | retail positioning. Short vendor history, so self-archive |
| **Bybit trade archive** | every trade, daily files since ~2019 | native | ✅ | **exact aggressor-side flow** (CVD, imbalance, VPIN) without paying for tick data |
| **Bybit instruments** | contract specs | revisable | ✅ | delisted symbols disappear from the API. **Our daily snapshots prevent survivorship bias** |
| **Bybit WebSocket** | kline, trades, book, tickers, liquidations | native | ✅ client | live state. Liquidations exist only here |
| **Binance Data Vision** | bulk klines, aggTrades, funding, OI, top-trader L/S | native | planned | largest venue: cross-venue confirmation of flow and positioning |
| **Deribit DVOL** | BTC/ETH implied vol index | native | ✅ | implied vs realised vol, risk appetite |
| **Fear & Greed** | daily sentiment composite since 2018 | native | ✅ | slow sentiment regime |
| **DefiLlama stablecoins** | total stablecoin supply | revisable | ✅ | crypto liquidity proxy |
| **FRED / ALFRED** | rates, dollar, Fed balance sheet, RRP, TGA, HY spreads | **vintage** | ✅ (needs free key) | macro liquidity regime; ALFRED makes it truly PIT |
| **GDELT** | global news volume/tone at 15-min resolution since 2017 | native | planned | attention shocks. Rate-limited (~1 req / 5 s) |
| **CryptoPanic** | crypto news, tagged by coin, with votes | native | planned (free key) | coin-level news flow (`sent.news`) |
| **Own RSS archive** | CoinDesk, The Block, Decrypt, Cointelegraph | native by construction | planned | headlines we timestamp ourselves |

## P1: next

| Source | What | PIT | Note |
|---|---|---|---|
| Deribit options book | skew, put/call OI | native | no free history, so start self-archiving early |
| DefiLlama TVL / fees | chain & protocol fundamentals | revisable | self-archive |
| Coin Metrics community | active addresses, tx counts, MVRV | revisable | free, partial coverage |
| Token unlocks (Tokenomist / DefiLlama emissions) | supply events | revisable | `known_ts` = first seen; schedules change |
| Bybit announcements | listings, delistings, maintenance | native | `/v5/announcements/index` |
| Macro calendar | FOMC, CPI, NFP times | native | official calendars |
| Polymarket | event probabilities (macro, regulation) | native | prediction-market priors for event risk |
| Spot ETF flows | BTC/ETH ETF net flows | revisable | published after the US close, so `known_ts` = next day |
| TradFi prices | SPX, NDX, DXY, VIX, gold, US10Y | native | cross-asset regime (also available as Bybit TradFi perps) |
| CoinDesk Data | news + social stats | native | the claude.ai CoinDesk connector needs authorising in claude.ai settings |

## P2: paid, only if a free proxy proves the signal

| Source | What | Why wait |
|---|---|---|
| Tardis.dev | tick + L2 history, all venues | needed only for order-book tools (OFI history) |
| Coinglass | aggregated OI / liquidations / funding | free single-venue data first |
| Glassnode | on-chain, **with PIT variants** | the only on-chain vendor with explicit PIT |
| Santiment / LunarCrush | social volume, dev activity | revisable; validate a free proxy first |
| Kaiko / Amberdata | institutional market data | redundant at our frequency |

## Known traps

1. **Survivorship.** `atx universe` uses *today's* listings and turnover. A historical universe must be rebuilt as of each date from our instrument snapshots plus historical turnover from bars. It must never come from today's list.
2. **TradFi perps.** Bybit lists stock, ETF and commodity perpetuals (`symbolType` = stock / ETF / commodity). They're excluded by `universe.symbol_types: [""]` and can be added back as cross-asset context.
3. **LLM-scored text in backtests.** Any modern-LLM sentiment score is contaminated for historical validation (ARCHITECTURE §3.7). Use lexicon or pre-cutoff models historically. LLM scoring is forward-only.
4. **Google Trends** re-normalises every query window. It's only usable if self-archived with identical queries.
5. **Rate limits.** Keep `rest_rate_limit_per_s` well under venue limits. GDELT returns 429 quickly.
