# Overnight loop 2026-11-12: multi-day microstructure on random days, and neural-network / tree signal generators with interpretability

User request (01:10 UTC): at today's scope, take 5 random days per token and rerun the tests, or more tests; loop
through the night with 2 agents, orchestrated by Claude. Process at most one week of raw data at a time, then delete the
raw data and keep only the lighter aggregates; rinse and repeat. Create more hypotheses and keep updating the knowledge.
Experiment with neural networks as signal generators on nonlinear data, and use mechanistic interpretability to learn
what forecasts returns and volatility (trees and networks). The aim is to inform how tools are called in the future.

Registered before any new data or model: H-MD1-6 (multi-day microstructure) and H-NN1-6 (models and interpretability),
in `research/hypotheses.yaml`.

## Random days (seed 20261112, drawn before any download; no repeats; 2 pre-cutoff + 3 post-cutoff per batch)

| batch | days |
|---|---|
| 1 | 2025-09-05 2025-12-18 2026-07-12 2026-07-13 2026-09-23 |
| 2 | 2025-08-31 2025-11-19 2026-07-22 2026-08-17 2026-10-25 |
| 3 | 2026-05-11 2026-06-21 2026-07-03 2026-08-16 2026-10-02 |
| 4 | 2026-05-20 2026-06-14 2026-07-07 2026-07-18 2026-10-09 |
| 5 | 2025-11-21 2026-01-25 2026-07-17 2026-08-04 2026-08-18 |
| 6 | 2025-09-02 2025-09-10 2026-10-16 2026-11-02 2026-11-08 |

The day pool runs from 2025-08-01 to 2026-11-09, because Bybit's ob200 archives start in mid-2025. TradFi perps exist
only after their listing (for example NVDA from 2026-05), so on earlier days they are simply absent. The 2026-11-10
pilot day joins the multi-day set from the aggregates already built for it.

## Scope per day

- **Trades** (`public.bybit.com/trading`): the 50 symbols of `config/mm_multi.yaml` plus BTCUSDT.
- **ob200 books** (`quote-saver.bycsi.com`): the 12 pilot symbols (BTC ETH SOL XRP DOGE HYPE CARV KAS ATOM NVDA QQQ XAU).
- **Raw volume:** about 1.3 GB per day at the 2026-11-10 scope, so a batch is about 6.5 GB. Only one batch of raw data
  is on disk at a time.

## Pipeline (agent A, `scripts/micro_batch.py` + `ai_trader.research.micro_days`)

1. **Download** the batch into `data/raw/batch_<n>/`. This goes straight from the public archives and does not write to
   the lake.
2. **Process each symbol-day**, then aggregate to `data/research/micro_days/<date>/`:
   - 1 s trade bars: OHLC, buy/sell volume and notional, trade and order counts (same-ms same-side prints merged).
   - 1 s book series for the L2 symbols: bid1/ask1, sizes, relative spread, depth at 5/10/25 bp, imbalance.
   - Per symbol-day fitted laws, using `sim.laws` on the raw trades before deletion: duration mixture, ACD, size tail,
     sign ACF, gamma, Hurst, volume laws, AFT where cheap.
   - Per-day cross-asset stats: lead-lag peaks, connectedness.
   - Activity and liquidity numbers for the population model.
   - A QA json.
3. **Verify** the aggregates (row counts, no NaN blow-ups), then **delete** the batch's raw files and log the storage.
4. **After each batch,** rerun the multi-day tests on all aggregated days so far. Record an H-MD row only once
   enough days exist. Final verdicts come after the last batch, or when the criterion is decidable.

## Models and interpretability (agent B, `ai_trader.ml`, torch CPU via the optional `ml` extra)

**Daily panel** (universe L, point-in-time features):
- MLP and XGBoost forecasters for next-day RV (H-NN1) and next-day direction (H-NN2), against log-HAR and the base
  rate, with expanding-window monthly refits.
- TreeSHAP main effects and interactions for the trees (H-NN3).
- For the MLP: a sparse autoencoder (dictionary learning) on the hidden layer, with probes against named state
  variables and feature ablation (H-NN4).
- Also linear probes, integrated gradients and activation patching, as descriptive tools.

**Micro data** (agent A's 1 s aggregates, once at least 2 batches exist): a 1D CNN for next-minute |return| (H-NN5) and
direction (H-NN6), evaluated leave-days-out.

**Output:** a "what to call when" note. For each forecasting target, which inputs carry the signal, which nonlinear
structure matters, and therefore which tools the decision layer should call, and in what order.

## Orchestration loop (Claude)

Self-paced wake-ups, typically every 20-40 minutes. Each iteration:
1. **Check the live NVDA micro market maker** (read-only): heartbeat, P&L, fills, trend-guard events. Never stop it
   unless a predefined rule fires or the user asks.
2. **Check both agents.** When A finishes a batch, hand it the next batch. When B finishes an experiment, hand it the
   next one, or the micro-data CNN once data exists.
3. **Merge new results into knowledge:** recorded results, new forward or registered hypotheses (always before
   testing), `research/rules.yaml` (validated), BELIEFS.md, the diary.
4. **Stop the loop** when all 6 batches and the NN program are done, or the storage budget would be exceeded, or the
   user says so.

## Storage rule

The lake plus `data/research` must stay under the 30 GB `storage.budget_gb`. Raw batch folders are deleted after
verified aggregation. The 2026-11-10 ob200 zips (923 MB) are deleted once agent A has rebuilt the 1 s book series from
`book_100ms`. The lake's `trades_raw` partitions follow the lake's own retention mechanism, not ad hoc deletion.

## Acceleration (user, 2026-11-12 ~12:20 UTC)

The user netted the NVDA exposure. The program is research only from here, and three more agents join (5 in total):

- **C, discretised state spaces** (H-DS1-3). Fixed quantile lattices, k-means codebooks and honest shallow trees on the
  daily panel, with empirical-Bayes cell estimates. Tests how much of the nonlinear vol signal a small number of cells
  captures, and whether the cells keep their meaning across the cutoff.
- **D, sparsity** (H-SP1-3). Lottery-ticket pruning of the vol network, stability selection over a wide feature
  library, and per-neuron analysis of the surviving circuit: GAM fits and ablation.
- **E, FDR-controlled mining** (H-MINE1). A pre-declared family of discretised micro-state effects is tested with
  Benjamini-Hochberg control on 16 discovery days (batches 1-3 plus the pilot). The discoveries are then tested once on
  the confirmation days (batches 4-6), which E may not look at until its discovery list is frozen.

A keeps processing batches 4-6. B keeps the 1 s CNN study (H-NN5/6).
