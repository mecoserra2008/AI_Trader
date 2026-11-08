# Data audit: Parquet lake, 2026-11-07

Scope: `data/lake` (Bybit USDT perps plus alt_series), 23-symbol research universe. The audit used a frozen snapshot of `bars` taken at about 17:15 UTC while the backfill was still running. Raw rows were read with DuckDB, and point-in-time reads were cross-checked with `Lake().bars(...)`. Nothing under `data/` was modified.

## Summary

| # | Finding | Severity |
|---|---|---|
| 4 | **BCHUSDT 1d 2020-12-14** has `open=0`, `low=0`: log return and range are infinite | **critical** |
| 3 | 1,037 exact-duplicate PK rows in raw parquet (BTC 1h ×903, BTC 1d ×67, ETH 1d ×67), all identical incl. `known_ts`. `read_asof` hides them; raw DuckDB scans double-count | warning |
| 1 | `launch_ts` = 2018-01-01 placeholder for LINK/LTC/BCH; first klines 2020-10-21 / 2020-10-21 / 2020-12-14. BTC +10 d, ADA +62 d, TRX/XLM +19 d after launch_ts | warning |
| 6 | 1h→4h/1d mismatches in 2020–2021 only: 4h has 84 price and 4,236 volume mismatches (3 above 10%, max 27×); 1d has 17 price and 1,580 volume. Exact from 2022-01-01 | warning |
| 8 | First 1d bar of every symbol is a partial listing day; LINK/LTC first daily open is 6.8% / 12.7% away from the first intraday open | warning |
| 7 | 2025-10-10 cascade: real but extreme low wicks (SUI −84% intraday, DOT/ENA/FIL/UNI/AVAX >70%) distort range-vol estimators | warning |
| 5 | known_ts is synthetic (close+2 s) on a backfill that is today's vintage; revisions are possible (see #6) | info |
| 2 | **No gaps** in any series (1h/4h/1d), and no zero-volume or flat bars | info |
| 4 | High/low envelope 0 violations; open_t = close_{t-1} exact in 99.97%, max 0.62%; 10 1h bars (2021) with VWAP outside [L,H] | info |
| 9 | alt_series clean; F&G misses 3 d after 2018-04-13 and 1 d after 2024-10-25; stablecoins tiny before 2019-04-27 | info |
| — | funding / OI / ls_ratio / mark / flow_bars: BTC only, short windows | info |
| 10 | Survivorship: universe = today's survivors | info |

## 1. Coverage
In the snapshot, 1d and 4h are complete for all 23 symbols (last 1d = 2026-11-06, last 4h = 2026-11-07 12:00). 1h was present for 12 symbols (BTC, ETH, SOL, XRP, DOGE, ADA, LINK, LTC, BCH, AVAX, DOT, BNB), each complete from its first bar through 2026-11-07 16:00. The other 11 were still pending. The first 1d bar falls on the `launch_ts` date for 16 symbols. The exceptions are LINK and LTC (1,024 d after launch_ts), BCH (1,078 d), BTC (2020-03-25, 10 d), ADA (2021-03-18, 62 d), and TRX and XLM (2021-08-31, 19 d). The first 1h/4h bar usually starts a few hours into the first day.

The partial tables:

| table | contents | window |
|---|---|---|
| funding | BTC only, 203 rows | 2026-09-01 to 2026-11-07 |
| open_interest / ls_ratio 1h | BTC only, 904 rows each | 2026-10-01 to 2026-11-07 |
| mark 1h | BTC only, 903 rows | same window |
| flow_bars 15m / 1h | BTC only, 192 / 48 rows | 2026-11-04 to 2026-11-05 |
| trades_raw | 4.85M rows | 2 days |

None of these tables has duplicates or gaps. Their lag over known_ts matches the rules in docs/SCHEMA.md: +2 s for funding, mark and flow_bars, and +3,602 s for OI and ls_ratio. Flow-bar volume matches 1h bar volume to within 1e-6.

## 2. Gaps
After dedup, consecutive bars differ by exactly one period in every symbol and frequency: 0 gaps, 0 missing bars, and 0 misaligned `event_ts`. There are also no zero-volume or o=h=l=c bars. This means Bybit's kline endpoint returns no outage holes for these symbols, so outage detection would need a mark/index comparison. No exchange-wide or symbol-specific gaps were found.

## 3. Duplicates
There are 1,037 duplicate primary keys, each with exactly two copies:

| symbol | freq | duplicate keys | window |
|---|---|---|---|
| BTCUSDT | 1h | 903 | 2026-10-01 to 2026-11-07 14:00 |
| BTCUSDT | 1d | 67 | 2026-09-01 to 2026-11-06 |
| ETHUSDT | 1d | 67 | 2026-09-01 to 2026-11-06 |

All copies are identical in OHLCV and `known_ts`, so there are **0 conflicting duplicates**. They come from earlier ingest runs (`part-1794066908*`, `part-1794071146*` …) that overlap the backfill. `Lake.bars` returns 2,418 unique BTC 1d bars and 58,039 unique BTC 1h bars, which is correct. A future re-fetch with *different* values but the same synthetic `known_ts` would tie under `ORDER BY known_ts DESC`, and the choice of row would be non-deterministic.

## 4. OHLC integrity
- No bars have `high < max(o,c)`, `low > min(o,c)`, NaN values, negative volume or null volume.
- There is one non-positive price: **BCHUSDT 1d 2020-12-14 has open=0 and low=0** (h=280.35, c=276.95). The same day's 4h and 1h bars open at 272.95. This value comes from Bybit itself: `f()` in market.rs returns NaN, not 0, when parsing fails.
- Continuity. Of 897k pairs, `open_t == close_{t-1}` holds exactly in all but 166. |open/prev_close − 1| has a p99.99 of 1.9e-4 (1h), 1.8e-4 (4h) and 6.6e-4 (1d). All 10 cases above 10 bp fall in Feb–Apr 2021, with a max of 0.62% (LTC 1h 2021-02-10 02:00). None exceeds 1%.
- Turnover ≈ volume × price. For the implied VWAP (turnover/volume) against typical price, the median deviation is 0.07% (1h) and 0.39% (1d). 10 1h bars, all in 2021, have VWAP outside [low, high], for example ADA 2021-10-15 23:00 (VWAP 2.298 vs high 2.224) and ADA 2021-06-27 03:00. The largest 1d deviations other than BCH are the 2025-10-10 wick days, where the VWAP is legitimately far from the typical price.

## 5. Bitemporal
Across 897,392 deduped bars:

| check | result |
|---|---|
| `close_ts − event_ts` | equals the frequency exactly in every bar |
| `known_ts < close_ts` | 0 bars |
| `known_ts − close_ts` | always exactly 2,000 ms |

No incomplete bar is stored: `MarketData::klines` filters on `open_ts + dur <= now`. An as-of read at 2026-11-06 23:00 correctly returns 2026-11-05 as the last daily bar.

One caveat. Historical bars carry a synthetic `known_ts`, but their values are what Bybit served on 2026-11-07. Check 6 shows that Bybit's 2020–21 history is internally inconsistent, so it was probably revised at some point. The data is point-in-time by rule, not by vintage.

## 6. Cross-frequency (1h aggregated vs stored 4h / 1d; all 12 1h symbols, complete buckets only)
- From 2022-01-01 onward the match is exact: max relative difference ≤ 7e-16 on every OHLCV field.
- **Price mismatches** in 2020–2021: 84 4h bars and 17 1d bars, the last on 2021-11-29. They mostly affect close (max 0.19%, BCH 4h) and high (DOT 2021-11-29 high 37.535 vs 37.505). Open never mismatches.
- **Volume mismatches** in 2020–2021: 4,236 4h bars and 1,580 1d bars. Most are small; only 5 4h bars exceed 1%. Three exceed 10%:

| symbol | bar | stored volume | sum of 1h volume | ratio |
|---|---|---|---|---|
| BTCUSDT | 4h 2020-11-02 04:00 | 28.5 | 809.8 | 27× |
| LTCUSDT | 4h 2021-01-20 00:00 | 2,575 | 28,723 | 10× |
| LINKUSDT | 4h 2021-01-20 00:00 | 46,390 | 184,471 | 3× |

  The matching 1d bars are 17%, 21% and 11% low.

## 7. Return outliers
There are 7 1h bars with |log r| > 25%, 19 1d bars with |log r| > 40%, and 6 4h bars with |log r| > 30%.

| date | event | symbols affected |
|---|---|---|
| 2021-05-19 | crash | BCH, LTC, ADA, DOT, LINK at 1h (DOT −41%); 6 symbols at 1d |
| 2022-05-11 | LUNA | AVAX |
| 2022-11-09 / 11-10 | FTX | SOL −81% / +50% |
| 2023-07-13 | XRP ruling | XRP +55%, XLM +48% |
| 2024-02-23 | — | UNI +44% |
| 2024-03-02 | — | BCH +46% |
| 2024-11 | — | XLM |
| 2024-12-03 | — | TRX +67% |
| 2025-03-02 | US crypto reserve announcement | ADA +54% |
| 2025-11-07 | — | FIL +58% |

All of these are real, market-wide or news-driven events. They show partial reversals but no single-bar snap-back, so none looks like a bad tick. The 2025-10-10 20:00–22:00 cascade produced the widest ranges in the dataset: SUI 1d low 0.5608 against open 3.406, ln(H/L) = 1.80, plus DOT, ENA, FIL, UNI, AVAX and others. These wicks are real on Bybit, but range-based volatility estimators will treat them as extremes.

## 8. Early-listing behaviour (1d, range relative to BTC, normalised by days 61–150)
- **All 22 alts:**

| days after first bar | median range ratio | median turnover ratio |
|---|---|---|
| 2–3 | 1.29 | 0.4–0.7 |
| 4–14 | about 1.1 | 0.5–0.8 |
| 15–30 | 1.04 | 0.82 |
| 31–60 | 1.0 | 0.96 |

  The p90 range ratio stays around 1.8–2.6 in week 1. Most of these coins already had liquid markets elsewhere when Bybit listed them, so their burn-in is mild.
- **Native token launches (ENA, SUI, HYPE):** the range ratio is 1.4–5.9× in days 1–14 (ENA day 2 is 5.9×; HYPE day 10 is 4.8×). ENA turnover is 6–16× its later level (airdrop churn). HYPE turnover is only 0.02–0.26× its later level, because the market was illiquid early and grew later. The 7-day median range ratio first falls below 1.5 on day 16 (ENA), day 14 (SUI) and day 18 (HYPE).
- The listing day itself is always a partial day (e.g. AVAX's first 1h bar is at 08:00) and should be dropped.

## 9. alt_series

| series | rows | span | pit_quality | known_ts lag |
|---|---|---|---|---|
| fear_greed | 3,198 | 2018-02-01 to 2026-11-07 | native | +1 h |
| defillama stablecoins_total_usd | 3,265 | 2017-11-29 to 2026-11-06 | revisable | +24 h |
| deribit dvol_close BTC | 903 | 2026-10 to 2026-11-07 | native | +1.0006 h |

- Dates are strictly monotone and midnight-aligned for the daily series. There are no nulls, no negative values and no duplicates.
- fear_greed has 2 gaps: 2018-04-14 to 04-16 (3 days) and 2024-10-26 (1 day). Its range is 5–95.
- In the stablecoins series, values are negligible until 2019-04-27, when the total first exceeds $1 bn. Before that the series starts at about $110k, with a +27,280% jump on 2018-01-11. Only one vintage is stored, so its "revisable" flag cannot be tested.

## 10. Survivorship
The 23-symbol universe is made of today's survivors and liquid names. Any cross-sectional result is biased upward. Delisted perps are not in the bars table, although instrument snapshots do exist for them.

## Recommendations
1. **Exclude or repair BCHUSDT 1d 2020-12-14** (open=0). A simpler rule: drop the first 1d bar of every symbol, since the listing day is partial. In `crates/at-bybit/src/market.rs::MarketData::klines`, reject or flag bars where any price is ≤ 0 or NaN. In `crates/at-store/src/writer.rs::validate`, enforce `low>0 && high>=max(o,c) && low<=min(o,c)` for the `bars` table.
2. **Burn-in:** drop listing day + 14 days for all symbols. Use 30 days for native token launches (ENA, SUI, HYPE, and similar future additions). Also gate eligibility on trailing turnover, not on age alone.
3. **Cross-frequency:** before 2022-01-01, build 4h and 1d bars from 1h bars, or prefer the 1h-derived volume. Blacklist BTC 4h 2020-11-02 04:00, LTC and LINK 4h 2021-01-20 00:00, and their 1d bars.
4. **Duplicates:** have `atx klines` (`crates/atx/src/cmd.rs::klines`) skip keys that already exist, or run a compaction job. Make `read_asof` tie-break deterministically, for example by `known_ts DESC, filename DESC`. If a re-fetch's values differ from what is stored, stamp it with `known_ts = fetch time` so the revision is visible.
5. **Coverage:** do not use `launch_ts` as a "tradable since" date (LINK/LTC/BCH show 2018-01-01). Use the first bar instead.
6. **Volatility estimators:** winsorise or clip ln(H/L) on 2025-10-10 and other cascade days, or use close-to-close estimators alongside range-based ones.
7. Re-run checks 2–6 on the remaining 11 1h symbols, and on funding/OI/ls_ratio/flow_bars, once the backfill finishes.
