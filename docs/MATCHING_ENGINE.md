# at-match: Bybit-semantics matching engine (contract for agents F4 and F5, 2026-11-13)

**Purpose:** a deterministic, fast limit order book that matches like Bybit's USDT linear perpetuals, so generated order flow (F5) and strategies can be simulated at the order level.

**What "exact" covers:** the documented semantics (Bybit v5 API docs: order types, time-in-force, post-only, reduce-only, price and qty filters, fees, the shape of public trade and orderbook messages). It does **not** cover Bybit internals that are not public: the risk and liquidation engine, ADL, mark-price computation, internal latency, hidden queue priority tweaks. These are out of scope and must be stated in the report.

## Semantics to implement (crates/at-match)

- **Priority:** price-time (FIFO within a price level); partial fills keep their queue position; amendments that change price or increase qty lose priority (Bybit amend semantics).
- **Order types:**
  - **Limit.** Time-in-force:
    - GTC;
    - IOC (fill what crosses, cancel the rest);
    - FOK (all or nothing);
    - PostOnly (rejected with `EC_PostOnlyWillTakeLiquidity` if it would cross).
  - **Market.** Executes as an IOC limit at a protected price band, slippage-capped like Bybit's market order price protection; make the band a config parameter.
- **Order constraints:**
  - **reduce_only:** an order may only reduce an existing position. Excess quantity is cancelled or rejected per the docs, and the behaviour is documented.
  - **Instrument filters:** tick size, qty step, min order qty and min notional. Rejections are named.
- **Fees:** maker / taker from `config/fees.yaml` (per symbol class). Taker side = aggressor.
- **Outputs:**
  - trade prints (id, ts, price, size, taker side);
  - L2 snapshot and delta messages in the same JSON shape as Bybit's `orderbook.200` stream (`type` snapshot/delta, `b`/`a` levels, `u`, `seq`). The existing parser `src/ai_trader/sim/book.py` must read them unchanged.
- **Determinism:** the same event stream gives the same output (no clocks inside the engine; event timestamps come from input).
- **Performance target:** >= 1M order events per second on one core for a 200-level book.

## Interfaces

- **Rust API:** `OrderBook::new(rules, fees)`, `submit(order) -> Vec<Event>`, `cancel(id)`, `amend(id, ...)`, `snapshot(depth)`.
- **CLI:** `atx match-sim --events <in.jsonl> --symbol BTCUSDT --out <dir> [--snapshot-every-ms 100] [--depth 200]`.
  - Input lines: `{"ts_ms", "kind": "new|cancel|amend", "id", "side", "type", "tif", "price", "qty", "reduce_only", "owner"}`.
  - Outputs: `trades.jsonl`, `book.jsonl` (ob200-shaped), `rejects.jsonl`, `stats.json`.
- **Python:** F5 drives the CLI through files or a pipe. A pyo3 binding is not required.

## Tests

- Unit tests for every rule above, including PostOnly rejection, IOC/FOK, reduce-only, priority after a partial fill, and amend priority loss.
- A property test: book invariants (no crossed book after matching, conserved quantity).
- A round-trip test: book.jsonl parsed by `sim/book.py`.
- A replay test: replay a real ob200 snapshot plus the deltas of one minute as add/cancel events, and check the engine's book equals the real book.

## Changes

Note any interface change here, dated.

### 2026-11-13 (F4, first implementation: `crates/at-match`, `atx match-sim`)

**Rust API**
- `OrderBook::new(&InstrumentRules, FeeRates) -> Result<OrderBook, String>`. Use `with_config` to set the market band.
- Calls that change the book take the event time: `cancel(ts, id)` and `amend(ts, id, price: Option<f64>, qty: Option<f64>)`. There are no clocks inside the engine.
- Hot-path variants append to a reused buffer: `submit_into`, `cancel_into` and `amend_into`.
- `snapshot(depth) -> L2` returns (ticks, lots), best level first. `Grid` converts ticks and lots to and from decimals.
- Queue position: `order(id).queue_ahead` and `queue(side, tick)`.

**Input lines**
- `id` and `owner` may be integers or strings. Use one id type per file: polars infers one type per column.
- `price` and `qty` may be numbers or strings.
- `type` defaults to Limit and `tif` to GTC. A market order is always IOC, and PostOnly on a market order is rejected.
- Optional `slippage_type` (`Percent` | `TickSize`) and `slippage` set a market band per order, like Bybit `slippageToleranceType` / `slippageTolerance`.
- In an amend, `qty` is the total order qty after the amendment, including the filled part (Bybit semantics).
- `ts_ms` must be non-decreasing; a line that goes back in time is rejected as `BadInput`.
- An id must be unique among live orders. It may be reused once its order is closed.

**CLI flags beyond the contract**
- `--rules <json>`. Without it, rules come from the lake's instruments table.
- `--fees <yaml>` and `--fee-class`.
- `--market-band-pct` or `--market-band-ticks`. The default band is 1%, which is an assumption.
- `--orders` also writes `orders.jsonl` with every order update.

**Output shapes**
- `trades.jsonl`: `ts`, `id`, `price`, `size`, `side` (aggressor), `taker`, `maker`, `taker_owner`, `maker_owner`, `taker_fee`, `maker_fee`. Price and size are JSON numbers.
- `rejects.jsonl`: `ts`, `op`, `id`, `owner`, `reason`, `line`.
- `book.jsonl`:
  - `ts` is the end of the publication bin and `cts` the last event time applied;
  - the first message is a snapshot with `u = 1`, and `u` then rises by 1 per message;
  - `seq` is the book version;
  - a message is sent only when the top-`depth` window changed;
  - levels leaving the window are sent with size `"0"`.
- `stats.json`: `perf` is its only non-deterministic field.

**Reject and cancel reason names**
- Bybit `EC_*` and `CancelBy*` names where they exist.
- Engine names otherwise: `QtyBelowMinOrderQty`, `QtyAboveMaxOrderQty`, `OrderValueBelowMinNotional`, `ReduceOnlyRuleNotSatisfied`, `OrderNotModified`, `AmendQtyNotAboveFilled`, `IocRemainderCancelled`, `ReduceOnlyQtyAdjusted`, `BadInput`.
- The full list and the assumptions are in the crate docs (`crates/at-match/src/lib.rs`).

**Replay**
- `at_match::Ob200Replayer` converts ob200 snapshots and deltas into new / cancel / amend events. Growth joins the back of the queue and shrinkage comes off the back (LIFO).
- The real-data replay test runs when `AT_MATCH_OB200` points at an unzipped ob200 file. No ob200 file was on disk when this was written, so only the synthetic one-minute replay ran.

### 2026-11-13 (F5)

- The real-data replay test ran on the 2026-09-13 BTC ob200 file (`AT_MATCH_OB200`) before the file was deleted, and passed.
- `src/ai_trader/sim/ob_gen.py` drives `atx match-sim`: over 31 held-out days the engine's prints matched the generator's own book, with 0 rejects.

### 2026-11-14 (agent M, `atx mm-replay`)

- `OrderBook::level_qty(side, tick)`: open lots at one price level (0 when empty).
- `OrderBook::worst_price(side)`: the deepest level of a side (the edge of a depth-limited replayed view).
- `atx mm-replay` (crates/atx/src/mm_replay.rs) feeds real ob200 messages through `Ob200Replayer` into an `OrderBook` holding others' liquidity only (the market book is built with min qty = qty step, no min notional, no max qty, so every replayed level is accepted) and merges the public trade tape by time (trade ts <= message `cts` first). Our quotes are virtual and never enter the book: their queue position is tracked beside it (see the module docs: three cancel-allocation models, 150 ms latency).
- Real-data findings: on 80 symbol-days the replayer's book equalled the engine's at every check and the engine produced 0 trades and 0 rejects. TRXUSDT prints on 2026-10-29 / 10-31 include sub-tick prices while every ob200 level stays on the 0.0001 grid; the BTCUSDT archive is a 100 ms feed (about 59% of BTC prints fall outside the pre-trade touch; best alignment at `cts`).
