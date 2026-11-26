# Market participants with machine learning (PART), 2026-11-26

Agent PART. Hypotheses H-PART1..H-PART8 were registered and the design was frozen (diary 2026-11-25 22:36 UTC) before
any fit. H-PART9 was registered afterwards as a forward-only follow-up. Code: `src/ai_trader/research/participants.py`
and `crates/at-part`. Outputs: `data/research/participants/` (schema note `SCHEMA.md`). Data: 5 Bybit USDT perps (BTC,
ETH, SOL, XRP, DOGE) over 62 UTC days, 81.7 M taker orders (313 M prints) and 1 s order books with maker-side flows.

## Verdicts in plain words

1. **Who trades: seven taker types, and they are recognisable.** Clustering every taker order on size, aggression,
   timing and child-order traits gives seven types (table below). Two of them account for 84% of the notional:
   *multi-level burst sweepers* (type 6, 20% of orders, 55% of notional) and *regular precise-size takers* (type 4,
   37% / 28%). The other five are small: tiny round-lot repeaters, small clipped bursts, mid-size repeated clips and
   isolated round-size takers. The split is stable on new days (shares move by at most 3.9 pp), but a second method
   (k-means) does not reproduce it (ARI 0.14). So H-PART1 is **inconclusive**: the types are real descriptions of the
   tape, not natural clusters.
2. **Who is informed: clearly answered.** After 60 s, the mid has moved *for* the sweepers by +0.84 bp more than for the
   average order (HELD +0.94 bp, t ≈ 13-19), and slightly for the regular precise-size takers (+0.31 / +0.35 bp). It
   has moved *against* the tiny round-lot repeaters (-1.26 / -1.48 bp), the tiny same-side runs (-0.83 / -0.86) and the
   isolated round-size takers (-0.93 / -1.12). The same signs hold before and after the knowledge cutoff. H-PART2
   **supports**. This is the classic informed-vs-noise split: the flow a maker wants to trade against is the isolated,
   round-size, single-print flow, and the flow to avoid is the multi-level bursts.
3. **Knowing who trades does not predict returns.** Adding each type's signed flow and the flow-state HMM to OFI,
   aggregate taker imbalance and lagged returns changes held-out R² by +0.00003 (1 min), -0.00005 (5 min) and -0.0011
   (60 min), all with CIs around zero. Calibrated direction probabilities stay between 0.46 and 0.52 and have negative
   Brier skill against the baseline. In-sample the participant block is "significant" (Wald p ≈ 1e-23), which shows how
   little that means on 250 k overlapping rows. H-PART3 is **inconclusive by the letter of the rule, a null in substance**.
4. **It predicts volatility a little, but not robustly.** On HELD, the participant features improve ln RV forecasts by
   +0.0034 (5 min) and +0.0114 (60 min) of variance on top of a baseline R² of 0.74-0.76, on every symbol. The
   registered gate (positive in both FIT segments) fails at each horizon, so H-PART4 is **inconclusive**. A post-hoc
   ablation shows most of the 60-min gain came from the HMM state, whose inputs drift with the price level (a mistake,
   see Caveats). The type mix alone gives +0.0013 at 5 min, positive in every segment. That is registered as forward
   test H-PART9, not claimed.
5. **It does not predict a maker's adverse selection on new days.** Out-of-sample CV on FIT looked good (+0.018 R² for the
   10 s touch-maker loss). On HELD it was -0.0013 (10 s) and -0.00005 (60 s), and the registered gradient-boosting
   challenger was worse. H-PART5 **contradicts** on the clean holdout.
6. **Nothing is executable.** Every pre-registered variant loses money on HELD after costs. Taker on the 5-min signal:
   -11.5 bp/trade. Taker on the 60-min signal: -5.3 bp (gross +5.7, n = 832). Maker entry: -9.1 bp (5 min) and -2.2 bp
   (60 min, pegged cost case -1.3). The quote filter that skips minutes with high predicted adverse selection improves
   a touch maker's result by 0.23 bp per fill (-2.33 vs -2.56 bp after the 2 bp fee), so the fee is the whole story.
   H-PART6 **contradicts**. Capacity is not the constraint: median L1 is about $219 k (BTC), $67 k (ETH), $49 k (SOL),
   $23 k (XRP) and $13 k (DOGE), against our $2-risk trades of a few hundred dollars notional.
7. **Cross-venue flow: no answer.** The xvenue tracker lost Binance flow from 2026-11-23 and has no file for 11-22, so only
   2 of the 5 registered test days could be scored (dR2 +0.0005, CI includes 0). H-PART7 **inconclusive**.
8. **The atx tape labels agree with the types in direction only.** Sweepers are 1.24x over-represented near `SWEEP`
   alerts, small clipped bursts 1.12x inside same-side `ALGO` windows (criterion 2x), and tiny round-lot orders are
   under-represented near both. H-PART8 **inconclusive**. Descriptively, orders matched to lake liquidation events
   (±5 ms, same side, 2026-11-08..11-24, 69 k matches) are 2.9x over-represented in type 3 (mid-size repeated clips). The
   liquidation engine looks like a clipping algo, not a sweeper.

**What this means for the desk.** The participant taxonomy works as a *description* and as an *informedness* map: who
to fade and who to avoid. It is not a return signal, and at retail fees it is not a market-making edge. No rule
candidates for `research/rules.yaml` come out of this study. Three suggestions:

- (a) If the desk ever quotes, treat type-6 bursts as toxic and type-0/1/5 flow as benign. The measured gap is
  ~2 bp of 60 s markout, the same size as the maker fee, so it only matters with a rebate.
- (b) Use the type mix only as a volatility input, after H-PART9 passes forward.
- (c) Do not use any `tape` / participant label as a directional trigger.

## The taxonomy

Medians per type on FIT days (trait sample of 400 orders per symbol-day). Shares are FIT / HELD. Markout = 60 s
mid-to-mid relative to the all-order mean of the same symbol-days (t over days). "rel size" is the order notional over
the symbol-day median order notional.

| type | name (given after description) | orders % | notional % | median $ BTC / ETH / DOGE | rel size | prints | multi-level | gap any / same side (ms) | orders in prior 1 s | exact size repeat | sig. digits | rel 60 s markout bp, FIT (t) / HELD (t) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | tiny round-lot repeaters | 4.6 / 3.9 | 0.2 / 0.2 | 168 / 69 / 6 | 0.10 | 1 | 0% | 318 / 1,636 | 2 | 100% | 1 | -1.26 (-23) / -1.48 (-22) |
| 1 | tiny round-lot same-side runs | 4.0 / 2.0 | 0.1 / 0.1 | - / 176 / 6 | 0.07 | 1 | 0% | 320 / 320 | 2 | 99% | 1 | -0.83 (-14) / -0.86 (-14) |
| 2 | small clipped bursts | 17.0 / 20.9 | 0.4 / 0.6 | 163 / 50 / 21 | 0.18 | 1 | 0% | 11 / 11 | 8 | 92% | 1 | +0.03 (+1) / -0.11 (-2) |
| 3 | mid-size repeated clips | 4.7 / 4.5 | 1.7 / 1.7 | 1,481 / 517 / 100 | 1.04 | 2 | 0% | 43 / 67 | 6 | 91% | 2 | -0.06 (-1) / -0.26 (-2) |
| 4 | regular precise-size takers | 37.0 / 34.9 | 28.1 / 28.8 | 2,405 / 1,270 / 135 | 1.99 | 2 | 0% | 73 / 74 | 3 | 0% | 3 | +0.31 (+21) / +0.35 (+22) |
| 5 | isolated round-size takers | 13.2 / 11.5 | 13.9 / 13.6 | 3,546 / 1,556 / 122 | 2.32 | 2 | 0% | 336 / 1,821 | 2 | 0% | 2 | -0.93 (-22) / -1.12 (-20) |
| 6 | multi-level burst sweepers | 19.6 / 22.2 | 55.5 / 55.1 | 944 / 372 / 2,077 | 2.25 | 6 | 100% | 6 / 6 | 8 | 23% | 2 | +0.84 (+19) / +0.94 (+13) |

- All types peak at 14 UTC except type 3, which peaks at 07 UTC (Asia/Europe morning).
- Absolute 60 s mid markouts run from -0.29 bp (type 0) to +1.61 bp (type 6). The all-order mean is positive (~+1 bp)
  because each order moves the mid it hits.
- The mix depends on the symbol. Types 2 and 6 make up 25% / 24% of BTC orders and 23% / 28% of ETH orders, but only
  1-7% on the alts. On BTC and ETH the tick is tiny relative to price, so multi-level prints and same-size clip bursts
  are common there. Types 4 and 5 make up 48-59% and 17-20% of orders on SOL, XRP and DOGE.
- The 12 pre-cutoff days and the 50 post-cutoff days give the same types and the same informedness ranking.

Flow states (HMM, K = 3). No K reached the ARI ≥ 0.80 stability bar, so the rule took the most stable one (K = 3, ARI
0.65). The states are: average; tight-spread quiet book; active (more notional, larger |r|, more updates and refills).
They are persistent (stay probabilities 0.90-0.97). They are **not stable across price regimes**: state 2 holds 25% of
FIT minutes and 78% of HELD minutes. Treat them as descriptive only.

## Caveats

- **Takers only.** Public prints show the aggressor. Makers are seen only through book-state flows (near-touch adds,
  removals and refills), so "participants" means taker behaviours plus a book-state model. The types are behavioural
  footprints, not identities. One firm can produce several types, and one type can mix firms.
- **The HMM inputs drift with price (mistake, diary 2026-11-26).** `h_spread` is ln spread in bp. For 1-tick markets
  that equals tick / price, so it rose as BTC fell from ~110 k (2025) to ~62 k (Nov 2026). `h_cxl` and `h_refill` use a
  5 bp band whose width in ticks also scales with price. An ADF/KPSS + PSI check (house rule 1.3) would have caught this
  before the fit. It inflated the H-PART4 60-min HELD gain; the ablation is in `supervised/ablation_no_hmm.json`.
- **GMM on discrete traits.** Exact-size repeats, significant digits and print counts are discrete, so GMM components
  partly sit on point masses. That is why k-means disagrees.
- **Book resolution and depth.** The book is the 1 s state at or before the order, so up to 1 s stale, and the 200-level
  feed truncates BTC depth beyond ~3 bp. Depth-relative sizes are approximate, especially for BTC (sweepers show a
  median size of 4% of the previous-second L1 yet cross 3 ticks).
- **Executable test is approximate.** Latency is ≥ 1 s (the book resolution), harsher than the 150-300 ms target.
  Maker fills require a print strictly through the limit (conservative). The quote filter assumes our 1-lot fills on
  every touch order (optimistic). The at-match replay (`atx mm-replay`) was not run because no signal survived to
  justify it.
- **Day sample.** Pre-cutoff days are 12 scattered dates (2025-08 .. 2026-06). The post-cutoff block is contiguous, so
  FIT-CV blocks mix regimes. HELD (2026-10-30 .. 11-24) contains the late-November sell-off.
- **Multiple testing.** 8 registered hypotheses. H-PART3..5 cover 7 targets with Holm within each family, plus 5
  executable variants (Holm), plus 4 post-hoc ablation trials (descriptive, not verdicts).
- **Liquidation matching is loose.** Liquidation-stream timestamps match a print exactly only 8% of the time; ±5 ms
  matches 95%, so some matches are coincidental.

## Method

**Data.** 62 UTC days:

- the 12 pre-cutoff micro_days dates: 2025-08-31, 09-02, 09-05, 09-10, 11-19, 11-21, 12-18, 2026-01-25, 05-11, 05-20,
  06-14 and 06-21 (llm_in_sample);
- every day 2026-10-06 .. 2026-11-24 (clean_holdout).

Raw public.bybit.com prints and quote-saver ob200 archives were downloaded to `data/raw/participants/`, reduced, and
deleted. Transient raw peaked at ~2.1 GB once, during a restart, and was otherwise ≤ 1.9 GB. FIT = the pre-cutoff days
+ 2026-10-06 .. 10-29 (36 days); HELD = 2026-10-30 .. 11-24 (26 days), read once.

**Reductions.**

- *Taker orders.* Same-ms, same-side runs of prints (the micro_days definition).
- *Book.* `crates/at-part` (`part-book`) rebuilds the ob200 book and emits, per 1 s bin, the end-of-bin state plus the
  bin's near-touch (≤ 5 bp) added and removed notional, best-level refills, L1 Cont-Kukanov-Stoikov OFI and best-price
  moves. It is parity-exact with the Python reference `FlowBook` on BTCUSDT 2025-08-31 (861 k messages) and about 10x
  faster.

**Order features.** 11 features, all point in time: the book is the 1 s state at or before the order, and only earlier
orders are used.

- log10 notional;
- log notional / opposite L1;
- log notional / opposite 10 bp depth;
- log1p ticks crossed;
- log1p prints;
- log1p ms since the previous order, and since the previous same-side order;
- log1p orders in the prior 1 s and 10 s;
- log1p exact-size repeats among the previous 20 same-side orders;
- significant digits of the size.

Features are robust-standardised per symbol on a FIT sample of 3,000 orders per symbol-day (540 k orders). The GMM is
full covariance. k ∈ 2..10, chosen by the frozen rule: bootstrap ARI ≥ 0.80, then minimum BIC. The ARIs were 0.98,
0.61, 0.97, 0.85, 0.93, 0.86, 0.76, 0.66 and 0.67 for k = 2..10, which gave **k = 7**. Types are ordered by mean size.

**Minute panel.** One row per symbol-minute; features are known at the minute end.

- *Baselines.* OFI 1/5 min over the trailing-60-min L1, aggregate taker imbalance 1/5 min and standardised lagged
  returns for the returns family; lagged ln RV etc. for vol; lagged touch-maker loss, spread and RV for AS.
- *Participant block.* Per-type signed flow 1/5 min, per-type notional and count shares, and per-type |signed flow|,
  plus the forward-filtered HMM probabilities and a state-change flag.
- *Model and inference.* Pooled OLS with symbol intercepts. Features are clipped at the train 0.1/99.9 percentiles.
  Coefficients use day-clustered SEs.
- *Validation.* FIT uses purged blocked CV (6 chronological blocks, 1-day embargo) with out-of-sample dR2 by segment.
  HELD is fit on all FIT days and scored once. CIs come from a day-block bootstrap (2000 resamples, seed 20261125),
  with Holm within each family.
- *Challengers.* Logistic + Platt for direction. LightGBM (depth 3, 300 trees) ran only where the FIT-CV gate passed
  (AS10, AS60).
- *Correction made before any fit.* The AS lags use the minute before last, so they are fully known at t (diary
  2026-11-25).

**Results files.** `data/research/participants/`:

- `taxonomy/`: taxonomy.json, type_description.json, informedness.json, tape_sanity.json, liquidation_mix.json;
- `hmm/hmm.json`;
- `supervised/`: supervised.json, exec.json, xvenue.json, ablation_no_hmm.json.

The reusable feature set is `features_1m/<date>.parquet`. Join with an asof on `known_ts`; columns are in `SCHEMA.md`.

**Reproduce.** `uv run python -m ai_trader.research.participants <cmd>` with cmd in turn: reduce, taxonomy, assign,
describe, informedness, hmm, states, supervised, exec, xvenue, tape, liq, export. The Rust binary is built with
`CARGO_TARGET_DIR=target/agent_part cargo build --release -p at-part` and copied to
`target/agent_part/bin/part-book.exe`.
