# Order-Flow Data: Collection, Storage, Derivation, Generative Modelling

_v0.1, 2026-11-07. Code: `crates/at-flow` (collector), `src/ai_trader/storage/flow.py` (lifecycle and derivations). Schema: `flow_3s` / `flow_1m` in [config/schema.yaml](../config/schema.yaml)._

## 1. Principle: store the minimal sufficient base, derive everything else

Raw order-book deltas run to tens of GB per day for a broad universe, far beyond a 30 GB budget. So the collector **aggregates in memory and stores only 3-second candles**. Each candle keeps:

| group | columns | aggregation inside the 3 s |
|---|---|---|
| price | open, high, low, close, vwap, mid | trades (close carried forward through quiet candles); mid at close |
| taker flow ("OHLCVDBS") | buy_vol, sell_vol, buy_notional, sell_notional, n_buy, n_sell, max_trade | sums; **V = buy + sell and Delta = buy − sell are derived, not stored** |
| book shape | spread_bps, micro_bps, obi_l1, obi_{5,10,25,50,100}, depth_{bid,ask}_{25,100} | **time-weighted** over the candle (a state that lived 5 ms barely counts) |
| book dynamics | ofi | L1 order-flow imbalance (Cont–Kukanov–Stoikov) summed over every update |
| quality | book_span_bps, book_cov, book_updates | span = how deep the received book reaches (bands beyond it are truncated); cov = share of the candle with a valid book |

Everything else is a **deterministic function of this base**, recomputed on read:
- `flow_1m`: an exact rollup (sums; coverage-weighted means; OHLC); `known_ts = max(latest 3 s known_ts, minute end)`, so a partial minute at the edge of an outage is never known before it ends;
- SMA, EMA and **EMA-of-SMA stacks** for every feature over any set of windows (`flow.derive`);
- vol- and depth-standardised z-scores;
- cross-asset aggregates (sector or market flow).

Only the base is stored, never the derived views, so the stored size scales with the number of symbols and does not grow with the number of features or windows used.

## 2. Collection

`atx collect` (Rust, always on):
- one websocket per 25 symbols, subscribed to `publicTrade` + `orderbook.{depth}` for every stream in `flow.streams`;
- books are rebuilt from snapshot + deltas;
- a reconnect invalidates only that connection's books, and coverage records the gap;
- candles are bucketed on **exchange time** and close on a **trade + book watermark** (since 2026-11-27, D8): once both
  the trade stream and the book stream of the symbol have delivered an event past the candle end; or once the wall clock
  is `grace_ms` (1.5 s) past the end and every stream that has not passed it has been silent for `grace_ms` (quiet
  symbols); or at the latest `max_hold_ms` (30 s) past the end while a stream lags. Before, the first later book message
  closed the candle and trades still in flight were booked one candle late (WS02, H-WS02-9: per-3 s trade counts matched
  the public archive 57% of the time). A trade that arrives after its candle closed (lag > 30 s) goes into the oldest
  open candle, so totals stay exact, and is counted (`late_trades` in the heartbeat). Rows before 2026-11-27 keep the old
  timing: use the WS02 archive fills or minute aggregates for sub-minute trade timing on those days;
- `known_ts = max(local finalisation time, candle end)`, because Bybit's clock runs 0.3-2.9 s ahead of ours;
- a connection that delivers no data for `silence_secs` (30 s) is torn down and reconnected (websocket stalls, dead
  TCP sessions after Modern Standby or sleep). Repeated silence right after a reconnect (a quiet market, e.g. TradFi
  perps off hours) doubles the threshold, up to 8 min;
- every `heartbeat_secs` (60 s) each connection logs one `heartbeat` line: messages, trades, book updates,
  liquidations, symbols without a trade, mean / max trade lag (receive − exchange time), late trades, open candles,
  reconnects. It warns `trade feed trickling` when trades fall below 20% of their trailing rate while book messages
  keep flowing (the `trade_deficit` hours WS02 found after the fact);
- rows are buffered and flushed to the lake every 5 minutes.

**Book depth matters.** Measured 2026-11-07:

| book | BTC perp span | BTC spot span | alts span |
|---|---|---|---|
| depth 200 | ~4 bps (every band truncated) | ~16 bps | 200+ bps |
| depth 1000 | ~28 bps (bands ≤ 25 bps valid) | ~93 bps | n/a |

So the most liquid names use depth 1000 and the rest use 200. Any band wider than `book_span_bps` must be treated as truncated.

## 3. Storage lifecycle and the 30 GB budget

`uv run python -m ai_trader.storage.flow maintain` (daily, part of `ai_trader.storage.daily`) does three things:
1. compacts each **cold** day of `flow_3s` (before today UTC and no file written in the last 2 h, so a flush after a
   resume from standby is never raced) from its flush files into one sorted zstd-6 file, dropping exact duplicate rows.
   This is reversible: the old files move to `data/lake_backup/<run>/` with a `manifest.jsonl` (rows in / out, a
   multiset check of the new file against the old rows, sha256); `python -m ai_trader.storage.repair undo|purge --run R`;
2. rolls it up exactly into `flow_1m`, one symbol at a time (memory: one symbol-day);
3. prunes the oldest **already-rolled-up** `flow_3s` days, never younger than `storage.flow_3s_min_days`, until the whole lake fits `storage.budget_gb`.

It never ran before 2026-11-27 (it needed an OS scheduler entry, which the desk does not use; days had up to 1,734
flush files). `scripts/ops/desk_start.sh` now runs it daily as the `daily_ops` loop.

This gives two tiers:
- a **rolling high-resolution window** (3 s, as many days as the budget allows), for microstructure and for training sequence models;
- a **permanent 1-minute history**, about 1/20 of the size, for regimes, priors and cross-sectional panels.

Sizing is measured, not assumed: see §6.

## 4. From flow data to priors, copulas and the latent manifold

The plan below is staged by how much data it needs. Nothing is fitted until the data exists. Every step is deterministic or seeded, and is validated out of sample by day.

**Stage 1: standardised marginals (≥ 2 weeks of 1 m data).** Per symbol, standardise each feature, e.g. delta by trailing turnover, OFI by trailing L1 depth, OBI as-is. Fit Student-t / skew-t marginals per regime cell with hierarchical shrinkage (asset → cluster → global), as in ARCHITECTURE §1.2 and §2.1. That gives *prior distributions of flow states per regime*.

**Stage 2: dependence via copulas (≥ 1 month).** Transform features to uniforms through their fitted marginals (the PIT transform). Then fit two kinds of copula per regime cell:
- **within-asset:** a t-copula, or a vine copula if tail dependence is asymmetric, on (delta z, OFI z, OBI_10, micro_bps, spread, forward return);
- **cross-asset:** a factor copula (market flow factor + idiosyncratic) for the panel.

Copulas separate *how features co-move* (and how that changes by regime and on weekends, H-F4) from their marginal scales. That's what stress testing needs.

**Stage 3: generative scenario engine (≥ 1–2 months).** Chain the factorial-HMM regime transitions with regime-conditional copula draws and marginal inverse-CDFs. This yields synthetic multi-asset flow-and-return paths, including regime sequences and shock combinations that never occurred historically. Uses:
- stress-test strategies and the risk engine (liquidation cascades, weekend shocks in thin books);
- build priors for rare cells (R3, R9) where real episodes are few.

Synthetic data must pass three checks before use:
- matching marginals and tail indices;
- matching autocorrelation of |returns| and flow;
- a classifier two-sample test: a discriminator shouldn't beat chance.

**Stage 4: latent manifold (≥ 3 months of 3 s data; it will exceed the 3 s window, so train incrementally).** A masked sequence model (small transformer encoder or temporal convolutional autoencoder) on windows of the multi-asset 3 s / 1 m feature tensor learns an embedding of market state. Uses:
- regimes as regions or trajectories in latent space (compare with the 9-cell grid);
- **drift monitoring**: distance of today's embeddings from the training manifold flags novel conditions where every prior is suspect;
- nearest-neighbour analogues in latent space (`pred.analog`, CANDIDATE_GENERATION G8);
- conditional generation as an alternative to Stage 3.

The transformer here is a *tool*: deterministic once trained, versioned, and validated walk-forward. It isn't the LLM decision layer, and it's trained only on our own post-cutoff collected data, so it carries no look-ahead contamination.

Hypotheses for this data were registered before collection started (H-F1 to H-F4 in research/hypotheses.yaml). All flow data is therefore clean holdout.

## 5. Universe for collection

See [research/reports/bybit_instruments_2026-11-07.md](../research/reports/bybit_instruments_2026-11-07.md) for the full catalogue. On Bybit as of 2026-11-07:
- 937 linear instruments, including 310 TradFi perps and 7 pre-listing perps;
- 521 spot pairs;
- 6,872 options on 16 underlyings;
- 26 inverse instruments.

Streams are configured in `settings.yaml → flow.streams`.

## 6. Measured footprint (15-minute run, 32 streams, 2026-11-07)

| layer | bytes per row | per stream-day |
|---|---|---|
| flow_3s, 5-min flush files | 88 | 2.5 MB |
| flow_3s, after daily compaction (zstd 6, sorted) | **59** | **1.7 MB** |
| flow_1m (small sample, so overhead-heavy) | ≤ 160 | ≤ 0.23 MB |

Projection within the 30 GB budget:

| streams | 3 s per day | days of 3 s retained | 1 m per year |
|---|---|---|---|
| 32 | 54 MB | ~480 | 2.7 GB |
| **70 (current config)** | **~120 MB** | **~200** | **~6 GB** |
| 100 | 169 MB | ~120 | 8.4 GB |

Collector load: 32 streams ≈ 200 websocket messages per second (~94% book updates). The process is single-threaded and uses a few tens of MB of RAM.

## 7. Operations

The desk processes (collector, updater, daily_ops, ...) are started by `scripts/ops/desk_start.sh` and are NOT
persisted in Task Scheduler or the Startup folder (a standing user constraint): **after a reboot, run
`bash scripts/ops/desk_start.sh`** (the 2026-11-11 Windows-Update reboot cost 25 h of flow data). Watch the collector with
`grep heartbeat logs/collector_$(date -u +%Y%m%d).log | tail`.

```bash
powershell -ExecutionPolicy Bypass -File scripts\collector.ps1     # stand-alone alternative: collector with auto-restart, private binary copy
uv run python -m ai_trader.storage.flow status                       # lake size per table vs budget
uv run python -m ai_trader.storage.flow maintain [--dry-run]         # compact + rollup + prune to budget (cold days only)
uv run python -m ai_trader.storage.daily [--dry-run]                 # the daily jobs: maintain + flow_bars + Deribit surface
atx collect --minutes 2 --category spot --symbols BTCUSDT --depth 1000   # ad-hoc test run
```
