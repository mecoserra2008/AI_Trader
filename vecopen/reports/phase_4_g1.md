# Phase 4: gate G1 (leak audit)

Date: 2026-11-14 (UTC). Leak auditor (fresh context; SPEC section 8, role 3). Status: **G1 PASS on all 20 crypto
assets.** Leak tests 1, 2 and 3 of SPEC section 6 pass on every asset, and so does every construction-rule check.
**CFDs are not covered: no CFD data exists yet.**

## Reproduce (single command, from the repo root)

```
uv run --extra vecopen python vecopen/tests/leak/run_g1.py
```

The command does the following, one asset at a time:

1. Audits the 20 crypto assets of `config/baseline.yaml`.
2. Checks that no CFD research data exists.
3. Runs the matching-timing check (covariates and timestamps only).
4. Runs the fast `vecopen/tests/leak` pytest suite.
5. Writes `reports/phase_4_g1_results.json` and `reports/phase_4_g1_tables.md`.
6. Appends one registry line (phase 4, kind `audit`).

It reads `data/research` only.

**Blindness for G2.** No label table is read. No placebo touch count is computed, stored or printed: successful checks
are counted on treated outputs only, and placebo outputs report failures only.

**Resources.**
- Runtime: 12,115 s (3.4 h) on this machine, which was shared with other heavy apps. That is 470 to 866 s per asset.
- Peak working set: at most 549 MB for the parent plus 374 MB for its worker.
- Threads: polars uses 2 threads in the parent and 1 in the worker. During the streaming loop the parent does no
  polars work, so the two processes together use at most 2 compute threads.

**Registry line 12 (`seq 11`).** `kind audit`, phase 4, `config_sha256 9e78f831…ae14a`, `git_commit "uncommitted"`,
`diff_sha256 3682a38e1513…`, `line_sha256 58d81d50bcdc6e1d59a293f54a95c600e219a98e96906dde5a527fcc3681e46e`. The
metrics hold:
- pass or fail per test per asset;
- the harness sha256 `ebf5491ece4023ed3989a217a8633b9ac9231229c30f12d6845925fee7b7aeb1`;
- the results file sha256 `30d4eb758fd8bae5…`;
- the tables sha256 `aa262fbc061e9205…`;
- the pytest counts.

`python -m vecopen verify` reports `registry OK: 12 lines`.

The full-history tests also exist as pytest tests, parametrised over the 20 assets, but they run only with
`VECOPEN_LEAK_FULL=1`. `vecopen/tests` is run by every phase's reproduce command, and these tests take about 10
minutes per asset.

## Verdict

| asset | test 1 truncation (200 t; literal run) | test 2 perturbation (200 t) | test 3 timestamp audit | construction rules |
|---|---|---|---|---|
| BTCUSDT | PASS (200/200; literal pass) | PASS (200/200) | PASS | PASS |
| ETHUSDT | PASS (200/200; literal pass) | PASS (200/200) | PASS | PASS |
| SOLUSDT | PASS (200/200; literal pass) | PASS (200/200) | PASS | PASS |
| XRPUSDT | PASS (200/200; literal pass) | PASS (200/200) | PASS | PASS |
| BNBUSDT | PASS (200/200; literal pass) | PASS (200/200) | PASS | PASS |
| DOGEUSDT | PASS (200/200; literal pass) | PASS (200/200) | PASS | PASS |
| ADAUSDT | PASS (200/200; literal pass) | PASS (200/200) | PASS | PASS |
| AVAXUSDT | PASS (200/200; literal pass) | PASS (200/200) | PASS | PASS |
| LINKUSDT | PASS (200/200; literal pass) | PASS (200/200) | PASS | PASS |
| LTCUSDT | PASS (200/200; literal pass) | PASS (200/200) | PASS | PASS |
| BCHUSDT | PASS (200/200; literal pass) | PASS (200/200) | PASS | PASS |
| DOTUSDT | PASS (200/200; literal pass) | PASS (200/200) | PASS | PASS |
| TRXUSDT | PASS (200/200; literal pass) | PASS (200/200) | PASS | PASS |
| ATOMUSDT | PASS (200/200; literal pass) | PASS (200/200) | PASS | PASS |
| ETCUSDT | PASS (200/200; literal pass) | PASS (200/200) | PASS | PASS |
| FILUSDT | PASS (200/200; literal pass) | PASS (200/200) | PASS | PASS |
| XLMUSDT | PASS (200/200; literal pass) | PASS (200/200) | PASS | PASS |
| NEARUSDT | PASS (200/200; literal pass) | PASS (200/200) | PASS | PASS |
| UNIUSDT | PASS (200/200; literal pass) | PASS (200/200) | PASS | PASS |
| AAVEUSDT | PASS (200/200; literal pass) | PASS (200/200) | PASS | PASS |

**No failures, so there is no minimal reproduction to give.**

**CFDs: not audited, because none exists.** None of the 16 CFD candidates has research data (no cTrader keys yet,
phase 1).
- G1 must be re-run when CFD data lands. `test_no_cfd_research_data_yet` fails on purpose as soon as a CFD research
  file appears.
- Three CFD-only paths have never run on real data:
  - the non-UTC sessions (`fx_ny17`, `cme_ny18`, ...);
  - the label horizon in `events.base_bars_per_tau`, which still raises for non-24h sessions;
  - the Panama roll columns.

`vecopen/tests/leak` pytest suite: 21 passed, 0 failed, 20 skipped (the env-gated full-history tests).

## Design, and why it is equivalent to the spec's tests

### What is audited

The phase 3 event step, exactly:
- base 5m bars, and the closed bars of the 8 primary timeframes from phase 1's `resample` with the asset's session;
- merged by `levels.bar_stream`;
- fed one bar at a time to four `TouchDetector`s (treated, `p1_nonorigin`, `p2_displaced`, `p3_candle`), each over
  its own `LevelRegistry`, and to the `PlaceboGenerator`.

`leakkit.Pipeline` replicates the `phase3.event_step` loop line for line. Two checks tie it to production:
- `test_harness_pipeline_equals_phase3_event_step`: identical events and level rows on 35 days of ETHUSDT.
- On every asset, the matching covariates of every treated and every family-1 event in the stored phase 3 events
  table equal the audited stream's values (tables, section 7).

What "exists at t", and is therefore compared:
- **The level registry and detector state:** every live level with its zone, state, polarity and flips.
- **The pending leg candidates.**
- **The accumulators:** ATR, sigma, regime and volume baseline.
- **The placebo generator:** its pending family-1 candidates and its RNG state.
- **Every output emitted so far:** `TouchEvent`, `LevelInfo`, the registry's births, fills, flips and expiries, and
  the placebo specs.
- **Features:** `features.py` is still a phase 6 stub, so the only features are the event and birth covariates.
- **Signals:** no signal generator exists yet (`models.py` and `backtest.py` are stubs). Rule M0's decision event is
  the `TouchEvent` with its side g, which is covered.

Labels are excluded, because they look forward by design. That no label field feeds anything else is checked
separately (see the rule table below).

### Test 1: truncation invariance

**Sampling.** 200 timestamps t per asset, drawn without replacement among the base-bar closes.
- The generator is `default_rng([config.seed, crc32(asset), crc32("vecopen-g1-leak-clean")])`.
- Each t gets a warm-start point s = t − W, with W log-uniform in [1, 30] days on the 5m grid. The median is 4.6 to
  6.7 days, depending on the asset.
- One seeded t per asset gets s = none: a literal from-scratch run.

**One full run per asset, in the parent process.** The parent:
- pickles the complete pipeline at every s;
- digests every bar fed, and every output emitted, inside each window (s, t];
- digests the complete state at every t with `leakkit.canon`, an exact object-graph walk (floats by `float.hex`,
  arrays by their bytes, dict, list and heap order, and aliasing).

**Every truncated run reuses that full run.** Each one runs in a separate worker process (a fresh interpreter, with
its own globals):
1. Truncate the research frame at t (`ts_close <= t`).
2. Re-run `resample` for all 8 timeframes on the truncated frame.
3. Require every resampled frame, and the base frame, to equal the full run's on `ts_close <= t`: all rows and all
   columns, byte for byte.
4. Restore the snapshot taken at s, or start fresh for the literal run.
5. Replay the truncated stream over (s, t].
6. Require these to equal the full run's: the digest of the bars fed, the digest of the outputs, and the digest of
   the complete state at t.
7. Require the replayed state to hold no timestamp after t.

Over all assets this replayed 24.3 M stream bars for test 1.

**Why this is equivalent to the literal test.** The event step is a deterministic state machine,
S_k = F(S_{k−1}, x_k). A run on data truncated at t can differ from the full run at or before t in only two ways.

(a) **Its input prefix differs.** This is tested exactly, for every t and every row at or before t (step 3).

(b) **F reads something other than (S_{k−1}, x_k).** Four checks exclude this:
- **The state is closed.** `canon` raises on any object that is not plain data or one of vecopen's own classes: a
  frame, an iterator, a file or a path.
- **No future timestamp in the state.** The full-run state at every t and s holds no timestamp after its instant,
  and neither does the replayed state.
- **No hidden channel in the source.** Static AST tests show that the four streaming modules import no data, frame,
  IO or label module, call no file or loader function, and keep no mutable module state.
- **No hidden channel at run time.** Each replay runs in a fresh process from pickled bytes, so any hidden channel
  would surface as a digest mismatch.

Any look-ahead inside F would have to appear in the outputs or in the complete state at t, which are compared in full.

Given (a) and (b), the literal run reaches the full run's state at s, and the window replay shows that it then
reaches the same outputs and the same state at t.

**Validation.** The one literal from-scratch run per asset checks this equivalence directly on real data: no snapshot,
first bar to t. It passed on all 20 assets.

**Reduced coverage, explicitly.**
- 199 of the 200 timestamps per asset are replayed from a warm start rather than from the first bar. The argument
  above is why this is equivalent.
- Snapshot-restore exactness (the restored state digests like the parent's state at s) is checked on every 10th
  timestamp. This check is diagnostic only: an inexact restore can only produce a false failure at t, never a false
  pass.

### Test 2: future perturbation

At each of the same 200 timestamps:
- every price after t (o, h, l, c) is multiplied by a factor f_p;
- every volume after t is multiplied by a factor f_v. This goes beyond the spec's "all prices", so the test is
  stricter;
- f is log-uniform in [0.5, 2] with |ln f| ≥ 0.05, seeded per timestamp.

The worker then:
1. Re-resamples the perturbed data.
2. Requires every frame to be unchanged on `ts_close <= t`.
3. Requires the perturbed stream's bars at or before t to equal the truncated stream's, bar for bar (by digest).
4. Continues the same pipeline, whose state at t was just compared in test 1, through one day of perturbed bars:
   1.9 M bars over all assets.
5. Counts outputs stamped at or before t but emitted after t. There must be none, and there were none.

Separately, the full run checks that every output is stamped with the instant of the bar that emitted it (test 3).

**Reduced coverage, explicitly.**
- **The perturbed dataset is the data up to t + 1 day, not the whole remaining history.** It is the spec's test on
  the dataset D[: t + 1 day], with every price and volume after t multiplied. Rows more than one day after t cannot
  reach outputs at or before t: test 1 shows those outputs are identical with no data after t at all. The day after t
  contains every intraday bucket that straddles t, and the first day of the 1d, 1w and 1M buckets that straddle it.
- **The outputs at or before t of the perturbed run are not recomputed from a second snapshot.** Their inputs are
  verified identical bar for bar, and F is deterministic, so they are test 1's outputs, already compared in full.

### Test 3: timestamp audit

This runs for every output of every family in the full run: 4.45 M treated outputs counted, and every placebo
output checked as well. It has four parts.

- **Provenance of every resampled bar** (17.07 M bars).
  - The same resampler runs on a frame whose prices are the base bars' own close timestamps. Each bucket's `h` and `l`
    are then its maximum and minimum source timestamps.
  - Every bar has maximum source ≤ its `ts_close`, minimum source ≥ its `ts_open` + 5m, and source count = `n_base`.
  - Its keys equal the real frame's.
  - Result: 0 bars with a source after their close.
- **A timestamp-tracking proxy on the stream.**
  - Each bar carries its provenance.
  - The high-water mark of source timestamps fed so far must be at or before the stamp of every output.
  - Every output must be stamped with the bar that emitted it.
  - Every timestamp field inside every output must be at or before its stamp.
- **The explicit inputs of every event** (349,259 treated events, and every placebo event):
  - the touch bar is the base bar at t₀, and `prev_close` is the bar before it;
  - the level's confirmation bar is at or before t₀, and so is its provenance;
  - the polarity bar is the last trade τ-bar closing at or before t₀, with its provenance;
  - the maximum source timestamp is ≤ t₀.
- **Confluence inputs.** Every level counted in confluence (719,237 treated counts) was born at or before t₀ and is
  still active. The registry's `FirstFill.overlapping` ids are read through a class-level spy on
  `LevelRegistry.update` that returns the original result unchanged. `src` is not edited.

### Construction rules and the other requested checks

Every recomputation is the auditor's own, from the bars: Wilder ATR, mean TR, sigma, volume baselines, trade-bar
counts and the leg rule. Exact float equality is required.

| Request | How it is checked | Result |
|---|---|---|
| Rule 1: multi-timeframe joins use close time | Stream order: increasing close, with higher timeframes before the base bar at the same close. At every base bar t (10.35 M), the number of bars fed per timeframe equals the number with `ts_close <= t`. Provenance of every bar (above). Polarity at the fill comes from the last τ close at or before t₀ (349,259 treated). Non-vacuity: the join on open time fails tests 1, 2 and 3 | pass, 0 failures |
| Rule 2: rolling statistics shifted by one bar | Recomputed exactly for every level of every family: ATR_{c−1}, not ATR_c; regime = ln(ATR_{c−1} / mean TR of the bars before c); `sigma_birth` and `sigma_t0` exclude the bar at their instant; `leg_vol`'s baseline uses earlier sessions only. 440,504 treated levels, 349,259 treated events | pass, 0 failures |
| Non-vacuity of rule 2 | The unshifted ATR never equals the one used (0 of 440,504). The unshifted `sigma_birth` equals the shifted one in 11,635 of 440,504, and `sigma_t0` in 5,373 of 349,259. Those are bars whose entering and leaving squared changes are equal, typically both 0 on coarse-tick assets, so the shift check discriminates for at least 97% of levels. The running mean TR agrees with an exact `fsum` to a relative 1.9e-12 | discriminating |
| Rule 3: levels enter at the confirmation close, with the zone frozen | For every level: born at the close of trade bar `origin_index + k*`, on its own timeframe's bar. The leg rule is re-checked: threshold, strict trade-back, first k ≤ n, candle direction and ω. Zone = price ± κ·ATR_{c−1}. At every t, every active level's price, zone, ATR and birth time equal its birth record (343,704 treated level-instants). No fill on the birth bar (349,259) | pass, 0 failures |
| Rule 4: session anchors from the config | `sessions.yaml` sha256 `9add7ce4…a2607` equals the hash registered in phase 1. All 20 assets are `utc_24h`, as is the baseline's `assets.crypto.session`. 0 bars misaligned: 15m to 4h aligned to 00:00 UTC; 1d opens at 00:00, 1w on Monday 00:00, 1M on the 1st; all full length. Non-vacuity: the `fx_ny17` anchor moves every daily open | pass |
| Placebo RNG draws cannot depend on future data | Every family-2 displacement must equal the next draw of an independent `default_rng([seed, crc32(asset)])` (the D-P2 formula) in treated-birth order; 0 failures. At all 4,000 sampled instants, the generator's bit-generator state equals the reference's: draws are consumed only at treated births, so they depend only on the number of past births. Tests 1 and 2 also compare the RNG state as part of the complete state | pass |
| Matching covariates use only information at or before the event | `age` = trade τ-bars in (birth, t₀], for 349,259 treated events and every family-1 event. `departure_atr` and `regime` are taken at birth: birth ≤ t₀, and the regime is shifted as in rule 2. The stored events table that `match_all` reads is identical to the audited stream on all 20 assets. Matching's use of timing: see finding 1 | pass |
| No label field feeds anything else | Only `phase3.py` imports `vecopen.labels` (AST check of every src module). `event_step`, `event_row`, `level_row` and `match_all` reference no label object or table, and `match_all` reads `events.parquet` with explicit covariate columns. No label column is in `events.parquet`. `vecopen.labels` is never imported during the audit run. Running the label step leaves every event object unchanged | pass |

## Results

Every number below comes from `reports/phase_4_g1_tables.md`, which the reproduce command generated.

### Checks summed over the 20 assets (treated successes, failures of every family)

| check | ok (treated) | fail (all families) |
|---|---|---|
| `t3_bar_provenance` (bars) | 17,071,839 | 0 |
| `t3_output_stamp_is_now`, `t3_output_fields_le_stamp`, `t3_hwm_le_stamp` (outputs) | 4,450,203 each | 0 |
| `t3_event_at_base_close`, `t3_event_level_is_birth_record`, `t3_max_source_ts_le_t0`, `t3_touch_bar_is_bar_at_t0`, `t3_prev_close_is_bar_before_t0` | 349,259 each | 0 |
| `t3_confluence_inputs_le_t0` | 719,237 | 0 |
| `r1_asof_close_join` (base bars) | 10,347,088 | 0 |
| `r1_polarity_from_last_tau_close_le_t0`, `r2_sigma_t0_excludes_touch_bar`, `r3_no_fill_on_birth_bar`, `cov_age_counts_tau_bars_in_birth_t0` | 349,259 each | 0 |
| `r2_atr_shifted`, `r2_regime_shifted`, `r2_sigma_birth_shifted`, `r2_leg_vol_baseline_shifted`, `r3_*` birth checks | 440,504 each | 0 |
| `r3_zone_frozen_since_birth` | 343,704 | 0 |
| `rng_state_is_fixed_sequence_position` | 4,000 | 0 |
| family-2 and family-3 checks (`rng_p2_draw_is_next_fixed_draw`, `r3_p2_born_with_source`, `r3_p3_candle_prices`) | placebo-only (not counted) | 0 |

### Provenance of resampled bars (all assets)

| tf | 15m | 30m | 1h | 2h | 4h | 1d | 1w | 1M |
|---|---|---|---|---|---|---|---|---|
| bars | 3,449,031 | 1,724,516 | 862,258 | 431,132 | 215,569 | 35,933 | 5,133 | 1,179 |
| with a source after their close | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| assets with a partial first bucket | 4 | 4 | 4 | 10 | 11 | 17 | 20 | 17 |

On every asset the only partial bucket is the first bar. The per-asset coverage, runtime and memory tables, and the
stored-table comparison, are in `reports/phase_4_g1_tables.md`, sections 1, 2 and 7.

## Findings

No leak was found in the level, event, placebo or resampling code. The points below are observations for later
phases, not G1 failures.

1. **Family-1 matching is an offline construct that uses whole-sample covariate information.** This concerns
   `placebo.match_events` and `phase3.match_all`.
   - They scale the covariates by their SD over all treated and pool events of the cell, over the whole research
     sample.
   - The nearest neighbour may be a placebo event whose t₀ is later than the treated one's, within the same calendar
     year. That is the case for **0.475 of all matched pairs** (intraday 0.475, daily-and-higher 0.447). Both t₀ are
     in the same calendar year in 100% of pairs.
   - They exclude family-1 levels flagged `late_origin`. That flag is set after the placebo's birth, when its candle
     later becomes a treated origin. It is **0 on every asset**, so it has no effect today.

   Each covariate value is known at its own event's t₀, and no outcome enters matching. Nothing downstream of
   matching feeds a feature, signal or label. So this is a design point for G2 (which comparison sample), not a
   look-ahead in the decision path. If a later phase uses matched pairs inside a walk-forward or CPCV fold, the pool
   and the SDs must be restricted to the training window.
2. **`levels.parquet` carries outcome columns.** `fill_ts` and `expire_ts` are attached after the stream ends. With
   the births they are H2 outcomes, so they play the role of labels and must never be used as covariates.
   `match_all` does not read `levels.parquet`. The builder's new balance path reads `matches.parquet` only.
3. **"Up to t₀" is inclusive, and the code uses information at t₀.**
   - The polarity at a fill uses the close of the level's own τ-bar when one closes at t₀, and that bar contains the
     touch bar.
   - Confluence counts levels born at t₀.

   SPEC section 2 allows both. It follows that any entry must be at or after the close of the touch bar. The labels'
   next-bar-open entry already respects this.
4. **Partial first buckets.** For assets listed mid-period, the first bar of a timeframe can be a partial bucket (see
   the table above). Its open is the first traded open, not the period's. This is not a look-ahead, since it uses
   past data only. It affects only that timeframe's ATR seed (TR₁), because the first 14 bars cannot be candidates.
5. **`resample(..., complete_only=False)` would leak.** It emits a partial last bucket keyed at its nominal close.
   Every production caller uses the default (`True`). The non-vacuity test shows the audit catches this class of
   leak (the open-time join).
6. **Family-2 draws follow the treated-birth sequence.** The draw assigned to a level depends on its position in that
   sequence. That is past data only, and it was verified exactly. But any change to past data (a repair, or a CFD
   re-download) reshuffles every later draw, so family-2 placebos are reproducible only on frozen data.
7. **When G1 must be re-run.** SPEC requires G1 again after phase 6 (`FeatureBuilder`, `ContextScore`: features,
   macro vintages, filtered HMM) and after phase 8. When CFD data lands, re-run this command. The CFD-specific checks
   to add are the session anchors of the non-UTC conventions and the Panama columns.

## Non-vacuity (the tests can fail)

All of these run in `vecopen/tests/leak`, on 30 days of BTCUSDT used as the full data, unless noted.
- **The forbidden join on open time** (a bar visible at its first base close) fails test 1 and test 2 at every
  timestamp. It also fails test 3: the provenance check and the high-water mark.
- **A one-bar look-ahead** (each base bar's high includes the next bar's high) fails tests 1 and 2.
- **A non-closed state** (a polars frame or an iterator reachable from it) raises the closure check.
- **Bad stamps and keys:** an output stamped one bar late, an output holding a future timestamp, and a bar keyed
  before its data each fail the matching test 3 check.
- **On real data, the shift checks discriminate** (rule 2 row above).

## Files

All new; no file outside these was edited, apart from the appended registry line.
- `vecopen/tests/leak/leakkit.py`: the harness. It holds the pipeline, the canonical state digest and closure audit,
  provenance, the independent recomputation, the worker and `run_asset`.
- `vecopen/tests/leak/run_g1.py`: the reproduce command.
- `vecopen/tests/leak/test_leak_g1.py`: the pytest suite (21 fast tests plus 20 env-gated full-history tests).
- `vecopen/reports/phase_4_g1.md` (this report), `vecopen/reports/phase_4_g1_tables.md` and
  `vecopen/reports/phase_4_g1_results.json`.
- `vecopen/trials/registry.jsonl` and `registry.head.json`: line 12 appended.

## Disclosures

- **A pooled count I saw during development.** One smoke run of the harness printed per-check success counts pooled
  over all families, for a 60-day BTCUSDT slice (2021-02-04 to 2021-04-05).
  - Read together with the treated counts printed beside them, those totals give the pooled placebo birth and
    first-fill counts for that slice: a crude, unmatched touch rate for one asset, an H2-type aggregate.
  - Only I saw it. I changed the harness at once so that successes are counted on treated outputs only.
  - The numbers are not repeated in this report, the results file, the tables or the registry.
  - No audit design choice depends on them, and I have no role at G2.
- **Holdout.** `data/holdout` was never listed, read or sampled. The harness reads `data/research/<ASSET>.parquet`
  (through `load_bars`) and the covariate and timestamp columns of the phase 3 tables under `data/research/events/`.
  `test_holdout_is_never_a_harness_input` checks the harness source.
- **Labels.** No label table was opened, so no treated-only outcome was read either.
- **No edits elsewhere.** I did not touch `SPEC.md`, the config, `src` or `data`, apart from appending the registry
  line.
  - The instruments are applied from outside: a class-level spy on `LevelRegistry.update` that returns the original
    result unchanged, inside a context manager; pickling; and my own recomputation.
  - In variant mode only, the non-vacuity variants wrap `resample` and the bar stream.
- **Processes.** Only my own python processes ran: the parent and one spawned worker per asset, each worker stopped
  at the end of its asset. None remains. No scheduled task was created.
- **The builder's parallel work.** I did not touch its `phase3.py` balance-only path. My static test on `match_all`
  reads the current source.
- **Python 3.13** under uv, the deviation already recorded in SPEC.
