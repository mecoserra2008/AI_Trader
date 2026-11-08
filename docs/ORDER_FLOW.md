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
- `flow_1m`: an exact rollup (sums; coverage-weighted means; OHLC);
- SMA, EMA and **EMA-of-SMA stacks** for every feature over any set of windows (`flow.derive`);
- vol- and depth-standardised z-scores;
- cross-asset aggregates (sector or market flow).

Only the base is stored, never the derived views, so the stored size scales with the number of symbols and does not grow with the number of features or windows used.

## 2. Collection

`atx collect` (Rust, always on):
- one websocket per 25 symbols, subscribed to `publicTrade` + `orderbook.{depth}` for every stream in `flow.streams`;
- books are rebuilt from snapshot + deltas;
- a reconnect invalidates only that connection's books, and coverage records the gap;
- candles close on **exchange time**: when a later event arrives, or on wall clock + 1.5 s grace for quiet symbols;
- `known_ts = max(local finalisation time, candle end)`, because Bybit's clock runs about 0.3 s ahead of ours;
- rows are buffered and flushed to the lake every 5 minutes.

**Book depth matters.** Measured 2026-11-07:

| book | BTC perp span | BTC spot span | alts span |
|---|---|---|---|
| depth 200 | ~4 bps (every band truncated) | ~16 bps | 200+ bps |
| depth 1000 | ~28 bps (bands ≤ 25 bps valid) | ~93 bps | n/a |

So the most liquid names use depth 1000 and the rest use 200. Any band wider than `book_span_bps` must be treated as truncated.

## 3. Storage lifecycle and the 30 GB budget

`uv run python -m ai_trader.storage.flow maintain` (daily) does three things:
1. compacts each finished day of `flow_3s` (288 flush files into one sorted zstd file);
2. rolls it up exactly into `flow_1m`;
3. prunes the oldest **already-rolled-up** `flow_3s` days, never younger than `storage.flow_3s_min_days`, until the whole lake fits `storage.budget_gb`.

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

```bash
powershell -ExecutionPolicy Bypass -File scripts\collector.ps1     # always-on collector, auto-restart, private binary copy
uv run python -m ai_trader.storage.flow status                       # lake size per table vs budget
uv run python -m ai_trader.storage.flow maintain [--dry-run]         # daily: compact + rollup + prune to budget
atx collect --minutes 2 --category spot --symbols BTCUSDT --depth 1000   # ad-hoc test run
```
