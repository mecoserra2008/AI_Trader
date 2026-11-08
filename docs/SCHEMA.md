# Lake Schema

_Source of truth: [config/schema.yaml](../config/schema.yaml). This page explains the rules; the YAML defines the columns._

## Layout

```
data/lake/<table>/<partition>=<value>/.../part-<written_ms>-<id>.parquet
```

- **Parquet, zstd level 3.** Timestamps are `timestamp[ms, tz=UTC]`.
- **Hive partitions.** Partition columns live in the path, not in the file. `year`, `month` and `date` are derived from `event_ts`.
- **Append-only, atomic.** Each write produces new files (tmp, then rename). Nothing is updated in place.
- **Writers:** Rust (`crates/at-store`) for ingestion, Python (`ai_trader.storage.lake.Lake.write`) for Python-produced tables. Both validate against the same YAML.
- **Reader:** Python/DuckDB only (`Lake.read_asof`). DuckDB runs in-process with no server. Memory is capped by `data.duckdb.memory_limit`.

## Bitemporal rule (the look-ahead firewall)

Every table has two timestamps:

| column | meaning |
|---|---|
| `event_ts` | what the row refers to: bar open, funding time, observation period, publish time |
| `known_ts` | the earliest moment the system could have known the value |

**Every read is an as-of read:** `known_ts <= as_of`. Within each primary key, only the row with the greatest `known_ts` is kept. Revisions (for example a new FRED vintage, or a corrected bar) are **new rows with a later `known_ts`**, so a read as of yesterday returns what was known yesterday.

Guards in code:
- Writers reject rows with `known_ts < event_ts`.
- `ToolResult` refuses `data_max_ts > as_of`.
- The reader re-checks its own output.
- `tests/test_lake.py::test_tool_output_is_invariant_to_future_data` checks that deleting all data after `as_of` changes no output.

## known_ts rules per source

Conservative by design: being late costs a little freshness, being early is look-ahead.

| table / source | `event_ts` | `known_ts` |
|---|---|---|
| bars | bar open | bar close + 2 s |
| flow_bars | bar open | bar close + 2 s |
| trades_raw | trade time | + 0.5 s |
| funding | settlement time | + 2 s |
| open_interest, ls_ratio | snapshot timestamp | timestamp **+ one period** + 2 s (Bybit may update the current bucket) |
| instruments | fetch time | fetch time |
| Fear & Greed | day start | + 1 h |
| Deribit DVOL | candle open | candle close + 2 s |
| DefiLlama stablecoins | day | + 1 day (`pit_quality = revisable`) |
| FRED / ALFRED | observation date | max(vintage date, observation date) + 1 day (`pit_quality = vintage`) |
| flow_3s | candle open (exchange time) | max(local finalisation, candle end) |
| flow_1m | minute open | max known_ts of its 3 s candles |
| text_items (live) | published | ingestion time |

Latencies are configurable: `data.known_ts_latency_ms` (settings.yaml) and `defaults.known_ts_latency_ms` (datasources.yaml).

## Tables

| table | partitions | primary key | contents |
|---|---|---|---|
| `instruments` | venue | venue, symbol | tick/lot size, leverage, status, `symbol_type` (crypto vs TradFi perp). Snapshots kept forever, to avoid survivorship bias |
| `bars` | venue, price_type, freq, year | venue, price_type, freq, symbol, event_ts | OHLCV; `price_type` ∈ last, mark, index, premium |
| `flow_bars` | venue, freq, year | venue, freq, symbol, event_ts | aggressor buy/sell volume and turnover, trade counts, VWAP, max print |
| `trades_raw` | venue, symbol, date | venue, symbol, trade_id | raw prints. **Rolling retention** (`data.retention.trades_raw_days`) |
| `funding` | venue, year | venue, symbol, event_ts | realised funding rate |
| `open_interest` | venue, interval, year | venue, interval, symbol, event_ts | OI in base units |
| `ls_ratio` | venue, period, year | venue, period, symbol, event_ts | share of accounts long/short |
| `liquidations` | venue, year, month | venue, symbol, event_ts, side, price, size | from WebSocket only (no REST history on Bybit) |
| `alt_series` | source, year | source, series_id, entity, event_ts | generic numeric series + `vintage` + `pit_quality` |
| `text_items` | source, year, month | source, item_id | news/social metadata and scores; **no article bodies** |
| `events_calendar` | source, year | source, event_id | scheduled events; `known_ts` = when announced / first seen |
| `flow_3s` | venue, category, date | venue, category, symbol, event_ts | **base order-flow layer**: 3 s candles (taker buy/sell, TWA spread / microprice / OBI bands / depth, L1 OFI, book quality). Budget-pruned |
| `flow_1m` | venue, category, date | venue, category, symbol, event_ts | exact rollup of flow_3s; long-term tier |
| `snapshots` | year, month | session_id, symbol, tool | what the LLM saw each session (precomputed tool outputs) |

## Measured footprint (2026-11-07 sample)

| data | size |
|---|---|
| BTCUSDT raw trades, 1 day | ~2.4M rows, ~50 MB: hence rolling retention |
| BTCUSDT 1h flow bars, 1 day | 24 rows, a few KB |
| 900 hourly bars | < 100 KB |

Long-term storage is dominated by bars and flow bars, which stay in single-digit GB per year for ~150 symbols. Raw trades are a cache, not an archive.
