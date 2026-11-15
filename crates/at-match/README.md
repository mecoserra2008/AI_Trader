# at-match

A deterministic limit order book that follows Bybit v5 USDT-perp semantics, plus `atx match-sim`, a JSONL driver around it. It is a research simulator only: there is no network or exchange code, and no clocks (every time comes from the input).

- **Contract:** [docs/MATCHING_ENGINE.md](../../docs/MATCHING_ENGINE.md).
- **Full semantics and assumptions:** the crate docs in `src/lib.rs`.

## What it does

- **Priority.** Price-time priority over FIFO levels: `BTreeMap<tick, Level>` with an intrusive linked list per level in a slab, so a cancel is O(1). A partial fill keeps the order's queue position. An amend that changes the price or increases the qty loses priority; an amend that decreases the qty keeps it.
- **Order types.**
  - Limit GTC / IOC / FOK / PostOnly.
  - Market orders run as an IOC at a protected price: `ask1 x (1 + pct/100)` or `ask1 + n ticks`, and symmetrically from `bid1` for a sell. The default band is 1%; it can be set per run or per order.
- **Reduce-only** works against a per-owner one-way position ledger:
  - rejected when nothing would be reduced;
  - a qty above the position is cut to it;
  - resting reduce-only orders are trimmed, furthest from the market first, whenever the position changes;
  - a reduce-only order can never flip a position.
- **Rejections are named.** Tick, step, min/max qty, min notional, duplicate id, unknown order, PostOnly cross, FOK not fillable.
- **Fees.** Maker and taker rates come from `config/fees.yaml`; the aggressor pays taker.
- **Outputs.**
  - Trade prints with the aggressor side.
  - Order updates.
  - Rejects.
  - `orderbook.200`-shaped snapshot and delta messages (`u` +1 per message, `seq` = book version). `src/ai_trader/sim/book.py` reads them unchanged.

## CLI

```
atx match-sim --events in.jsonl --symbol BTCUSDT --out DIR [--snapshot-every-ms 100] [--depth 200]
              [--rules rules.json] [--fees config/fees.yaml] [--fee-class crypto|tradfi]
              [--market-band-pct 1 | --market-band-ticks N] [--orders]
```

Input, one JSON object per line. `id` and `owner` are integers or strings; `price` and `qty` are numbers or strings:

```
{"ts_ms":1731456000000,"kind":"new","id":17,"owner":"mm1","side":"Buy","type":"Limit","tif":"GTC","price":75812.3,"qty":0.01,"reduce_only":false}
{"ts_ms":1731456000005,"kind":"amend","id":17,"price":75812.5,"qty":0.02}
{"ts_ms":1731456000009,"kind":"cancel","id":17}
{"ts_ms":1731456000010,"kind":"new","id":18,"side":"Sell","type":"Market","qty":0.5,"slippage_type":"Percent","slippage":0.05}
```

Defaults and rules:
- `type` defaults to Limit and `tif` to GTC; a market order is always IOC.
- An amend `qty` is the total order qty after the amendment, including the filled part (as in Bybit).
- `ts_ms` must be non-decreasing.

Outputs, written to `DIR/`:

| File | Content |
|---|---|
| `trades.jsonl` | `{"ts","id","price","size","side" (aggressor),"taker","maker","taker_owner","maker_owner","taker_fee","maker_fee"}` |
| `book.jsonl` | `{"topic":"orderbook.200.SYM","type":"snapshot"\|"delta","ts","data":{"s","b","a","u","seq"},"cts"}`. `ts` is the end of the 100 ms bin; `cts` is the time of the last event applied. Levels leaving or entering the top-200 window are sent as `"0"` or with their size. |
| `rejects.jsonl` | `{"ts","op","id","owner","reason","line"[, "detail"]}` |
| `orders.jsonl` | Written with `--orders`. `{"ts","id","owner","side","price","qty","leaves","cum","status","reason"}` |
| `stats.json` | Counts, volumes, reject and close reasons, final book, per-owner position / cash / fees, rules, fees, and config. `perf` is the only field that is not deterministic. |

Where the inputs come from:
- **Rules:** `--rules`, or the lake's `instruments` table, read point in time at the first event when it is a real epoch-ms value, else the latest.
- **Fees:** the class comes from `--fee-class`, or from the lake symbol type.

## Rust API

```rust
let mut book = OrderBook::new(&rules, fees)?;                  // at_core InstrumentRules, at_risk FeeRates
let ev: Vec<Event> = book.submit(&NewOrder::limit(ts, id, owner, Side::Buy, Tif::GTC, 75812.3, 0.01));
book.cancel(ts, id); book.amend(ts, id, Some(price), Some(qty));
let l2 = book.snapshot(200);                                   // (ticks, lots), best first
// hot path: submit_into / cancel_into / amend_into(&mut Vec<Event>)
```

- **Queue position:** `order(id)` returns `queue_ahead`, and `queue(side, tick)` returns the queue at a level. Both are useful for queue-reactive models.
- **Seeding from a real book:** `Ob200Replayer` turns ob200 snapshot and delta messages into new / cancel / amend events. Growth joins the back of the queue; shrinkage comes off the back.

## Tests

`cargo test -p at-match`: 23 unit tests, 5 integration tests and 1 doctest.

- **Unit tests:**
  - every rule: priority, partial-fill priority, IOC, FOK (including the reduce-only cap pre-check), PostOnly, market band, filters, cancel, amend priority loss and keep, amend crossing as taker;
  - reduce-only: reject, cut, trim and never-flip;
  - fees and ledger, self-match, determinism;
  - the publisher's windowed deltas;
  - the driver's ids, bins and rejects.
- **`tests/invariants.rs`:**
  - 12 random flows with `check_invariants()` after every operation: never crossed, level totals equal the queue sums, FIFO priority, the index, positions net to zero, reduce-only total never above the position;
  - per-order conservation (cum equals the trades, open orders equal the book);
  - 6 flows compared trade-for-trade, and on the final book, against a naive reference matcher.
- **`tests/replay.rs`:**
  - a 200-level snapshot plus 600 deltas (one minute at 100 ms, including touch moves) replayed as events: the engine book equals the message-implied book after every message, with no trades and no rejects;
  - the same check on a real Bybit file when `AT_MATCH_OB200=<unzipped ob200 .data file>` is set (first snapshot plus 60 s). No ob200 file was on disk when this was written, so that case is skipped by default.
- **`tests/roundtrip_python.rs`:** `book.jsonl` from a 30k-event run is parsed by `ai_trader.sim.book` (`iter_messages`, `Book.apply`, `build_from_messages`). The Python book equals the engine's top-200 window, with no `u` gaps and no crossed bins. Runs through `uv run --no-sync`.

## Benchmark

`cargo run --release -p at-match --example bench [-- N]` uses a synthetic flow with these shares: 40% passive GTC, 5% PostOnly, 35% cancel, 8% amend, 7% aggressive limits, 4% market, 1% reduce-only. The book holds about 270 levels per side and about 80k resting orders. Results from one core of this machine, measured 2026-11-13:

| Path | Throughput |
|---|---|
| Engine only (`submit_into` / `cancel_into` / `amend_into`) | **3.79 M order events/s** (264 ns/event; 1M events, 193k trades) |
| JSONL driver (parse + match + render trades / book@100 ms / rejects into memory) | 0.62 M lines/s |

The contract target is >= 1M order events/s for a 200-level book. JSON parsing dominates the driver.
