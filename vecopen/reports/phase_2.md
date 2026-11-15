# Phase 2: levels

Date: 2026-11-14 (UTC). Builder: phase 2 subagent. Status: **acceptance met.**

## Reproduce (single command, from the repo root)

```
uv run --extra vecopen python -m vecopen phase2
```

The command runs these steps:

1. The BTCUSDT smoke run: `load_bars`, phase 1's `resample` to the 8 primary timeframes with the asset's session, the
   as-of merge, and one `LevelRegistry.update` call per bar.
2. It writes `reports/phase_2_tables.md`.
3. It runs `vecopen/tests`.
4. It appends one registry line (phase 2, kind `build`).

`--dry-run` skips the append and `--asset X` picks another asset. It reads `data/research` only.

Registry line 7 (`seq 6`):

| field | value |
|---|---|
| kind / phase | `build`, phase 2 |
| `config_sha256` | `9e78f831…ae14a` (baseline unchanged) |
| `git_commit` | `"uncommitted"` |
| `diff_sha256` | `eb9b9fc1…ba57d3` |
| `line_sha256` | `ca19763cc698350b0d9e47f5d36de1af43c99d66d2586e8fc192011dba4f747c` |

The metrics hold the test counts, the full smoke statistics and `tables_sha256`. The events digest
`a5e9ae6759a35407…` is a sha256 over every birth and first fill. It is identical across repeated runs. The tables
hash is not, because it includes the runtime.

## Acceptance (SPEC section 8, phase 2)

| Criterion | Result |
|---|---|
| `LegOriginDetector.update(bar)`, `LevelRegistry.update(bar)` per section 2, streaming only | `vecopen/src/vecopen/legs.py`, `levels.py`. No function takes a frame and returns per-bar outputs. `levels.run` and the smoke run are loops over `update`; `bar_stream` only orders bars |
| 12 hand-built synthetic cases pass | C01 to C12 below, in `tests/test_vo_levels_cases.py`. They cover birth at the confirmation bar, one level per leg, zone freeze, first fill, polarity flip, expiry and confluence |
| Property tests | `tests/test_vo_levels_properties.py` (hypothesis); listed below |
| Smoke run, one asset, all primary timeframes | BTCUSDT, 541,728 5m bars plus 352,077 higher-timeframe bars. Loop 15.0 s, peak RAM 332 MB |
| `vecopen/tests` and root `pytest` green | 139 passed (112 earlier + 27 new); root 571 passed |

## The 12 cases

The 12 cases run on the frozen baseline (theta 2.0, n 6, omega 0.25, ATR 14, kappa 0.10, A_max 200; weights 15m 2,
1h 4, 4h 6), and a test pins those values. Fourteen warm-up bars with TR = 1 make ATR_13 = 1.0 exactly. Each case's
comments give the arithmetic, and each expected output is written by hand.

| case | sequence (abridged) | expected output |
|---|---|---|
| C01 birth at confirmation, k* = 2 | Candidate at bar 14 (open 100, 0.5 ATR). Bar 15 high 99.9 < 100. Bar 16 closes at 98 (2.0 ATR) | No leg at bars 14 and 15. One leg at bar 16: origin 14, k* 2, birth_ts = close of bar 16, size 2.0 ATR. The registry has no level before the 15m close of bar 16, then zone [99.9, 100.1] |
| C02 k* = 0, ATR shifted | Bar 14 (100, 100, 84, 97.5) | Born at its own close, size 2.5. The unshifted ATR_14 = 29/14 would give a 4.14 threshold and reject it. The same bar during warm-up gives nothing |
| C03 trade back to the open | Bar 15 high exactly 100.0 = the open | No leg (the condition is strict). With high 99.99 the leg is confirmed at bar 16 |
| C04 window n | Confirmation at k = 6; a second sequence reaches 2 ATR only at k = 7 | Leg with k* = 6; no leg at all |
| C05 one level per leg | Candidates at 14 and 15. Bar 16 confirms 14; bar 16 would qualify by itself (2.1 ≥ 2·ATR_15). Bar 17 starts a new leg | Legs only at 16 (origin 14, price 100) and 17 (origin 17, price 97). Candidate 15 and bar 16 never become levels |
| C06 wick filter | ω = 0.25 exactly; then ω = 0.286 | Leg; no leg |
| C07 zone freeze | 20 wide bars push ATR above 4.5 | The zone stays [99.9, 100.1] and rebinding it raises. High 99.85 (inside a zone resized with today's ATR) is not a fill; high = 99.9 is |
| C08 first fill | Bars 15 and 16 reach into the future zone before birth; then base bars at T17−10m (high 99.5), T17−5m (high 99.9), T17 (high 100.05) | Exactly one `FirstFill`: t0 = T17−5m, age 0, resistance, confluence 0. No event for the pre-birth touches or the second touch |
| C09 multi-timeframe as-of join | 5m bars resampled to 15m by phase 1's `resample`, streamed by `bar_stream` | At every base bar t the registry has seen exactly the 15m bars with close ≤ t. The level is born between the 5m bars closing at T−5m and T. The 5m bar closing at T (in the zone) is not a fill; the fill is at T+10m. A 15m bar sent after the base bar of the same close raises |
| C10 polarity flip | Reference level; 15m closes 100.05 (inside), 100.5, 100.0 (inside), 99.5 | Flip to support at bar 19 and to resistance at bar 21 (n_flips 2); none at bars 18 and 20 |
| C11 expiry | Untouched level, 200 15m bars after birth | Age 199 at bar 215; `LevelExpired(age 200, untouched)` at bar 216; a later base bar into the zone gives nothing |
| C12 confluence | Six levels on 15m, 1h and 4h; one is expired, one is a reference level, one does not overlap | Fills at 16h: E 4, R 2 (filled by the same bar, so each counts the other). Fills at 62h: L15 = 4+6+4 = 14, L4 = 4+2+4 = 10, L1 = 4+2+6 = 12. The expired E and the non-overlapping N are excluded |

Extra unit tests:
- no-trade bars are invisible: a flat bar at the open with v = 0 neither counts in the window nor kills the
  candidate, whereas with v = 1 it kills it; a flat base bar with v = 0 does not fill;
- the opposite direction is not locked;
- a reference level expires too;
- a gap over the zone is not a fill and the level stays live;
- the feed order is enforced;
- parameters come from the config (theta 3.0 rejects C02; kappa 0.25 widens the zone);
- Wilder and simple ATR match hand values;
- `leg_vol` matches a hand value;
- the volume slots follow the session's DST.

## Property tests (hypothesis)

- **Detector equals a direct oracle** (150 examples × 2 configs). The oracle evaluates the section 2 formulas per
  candidate, finding k* by direct search, then applies one level per leg greedily in confirmation order.
- **Registry equals a brute-force model after every bar** (150 × 2). The model checks these properties:
  - a level never exists before its confirmation close, and births come only from the bar of their own timeframe at
    the confirmation close;
  - the zone width never changes after birth: zone and ATR equal their birth values at every step, and the width is
    2κ·ATR;
  - first fills fire exactly when a base bar is the first after birth to intersect the zone of a live untouched level,
    so the heap index equals a full scan;
  - confluence and overlapping ids equal a full scan;
  - age, initial polarity, flips and expiry follow the state machine;
  - no-trade bars change nothing.
- **Truncation and future-perturbation invariance** (60 × 2). All outputs with ts ≤ cut are equal on the full data,
  on the data truncated at the cut, and on data whose prices after the cut are multiplied by a random factor.
- **Non-vacuity.** A seeded stream produces all four event kinds.
- **Real data.** BTCUSDT research bars from 2021-02-01 to 2021-03-18 on 6 timeframes, with the same brute-force model.
  The slice contains the 11-bar venue_flat run of 2021-03-02.
- **Generator.** It produces exact retests of earlier opens, wick spikes, gaps over zones and flat bars.
- **Mutation check.** I injected 9 one-line bugs into `src` one at a time, and each was caught: birth-bar fill,
  non-strict trade-back, dropped jump-over, open zone edge, ω < ω̄, window n−1, expiry off by one, wrong flip edge,
  and no-trade handling removed in either the detector or the registry. Exact-equality edges with a non-grid ATR
  (zone edge, ω = ω̄) are caught by the hand cases.

## Test output

```
uv run --extra vecopen pytest vecopen/tests     139 passed in 85.34s   (112 phase 0/1 + 27 phase 2: 22 cases/unit, 5 property)
uv run pytest (root)                            571 passed, 2 warnings in 473.07s   (vecopen/tests is not collected)
python -m vecopen phase2 (tests step)           {'tests': 139, 'passed': 139, 'failed': 0, 'errors': 0, 'skipped': 0}
python -m vecopen verify                        registry OK: 7 lines
```

## Smoke run: BTCUSDT research sample (2020-01-01 to 2025-02-24, session utc_24h)

Copied from `reports/phase_2_tables.md`.

Levels born per timeframe per year (by birth close, UTC; 2025 is partial):

| tf | 2020 | 2021 | 2022 | 2023 | 2024 | 2025 | total |
|---|---|---|---|---|---|---|---|
| 15m | 1,978 | 2,164 | 2,015 | 2,089 | 2,370 | 338 | 10,954 |
| 30m | 951 | 1,057 | 1,011 | 981 | 1,158 | 167 | 5,325 |
| 1h | 488 | 515 | 511 | 476 | 560 | 83 | 2,633 |
| 2h | 262 | 287 | 261 | 248 | 296 | 41 | 1,395 |
| 4h | 133 | 134 | 131 | 130 | 140 | 23 | 691 |
| 1d | 21 | 28 | 22 | 25 | 29 | 3 | 128 |
| 1w | 5 | 8 | 1 | 4 | 7 | 0 | 25 |
| 1M | 0 | 2 | 1 | 0 | 3 | 0 | 6 |

First fills and lifecycle:

| tf | tau-bars | no-trade | births | first fills | filled / born | median age at first fill (tau-bars) | median age (hours) | median confluence | flips | expired (untouched) | live at end (untouched / reference) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 15m | 180,576 | 16 | 10,954 | 8,704 | 0.795 | 17 | 4.3 | 5 | 49,405 | 10,942 (2,245) | 5 / 7 |
| 30m | 90,288 | 5 | 5,325 | 4,176 | 0.784 | 18 | 9.2 | 7 | 22,075 | 5,313 (1,145) | 4 / 8 |
| 1h | 45,144 | 1 | 2,633 | 2,064 | 0.784 | 19 | 19.8 | 9 | 10,561 | 2,620 (566) | 3 / 10 |
| 2h | 22,572 | 0 | 1,395 | 1,068 | 0.766 | 17 | 35.5 | 10 | 5,504 | 1,384 (324) | 3 / 8 |
| 4h | 11,286 | 0 | 691 | 510 | 0.738 | 17 | 71.8 | 10 | 2,279 | 680 (178) | 3 / 8 |
| 1d | 1,881 | 0 | 128 | 90 | 0.703 | 18 | 444 | 6 | 437 | 112 (32) | 6 / 10 |
| 1w | 269 | 0 | 25 | 15 | 0.600 | 19 | 3,311 | 0 | 62 | 9 (3) | 7 / 9 |
| 1M | 61 | 0 | 6 | 3 | 0.500 | 4 | 3,175 | 23 | 10 | 0 (0) | 3 / 3 |

Total: 21,157 levels and 16,630 first fills. The median k* is 2 on every timeframe except 1M (0).

| step | value |
|---|---|
| load_bars + resample to 8 timeframes | 0.59 s |
| streaming loop, 893,805 `update` calls (541,728 base + 352,077 higher) | 15.0 s (12.5 to 19.2 s across runs, depending on machine load) |
| peak RAM (process peak working set) | 332 MB |

**Performance.** The loop is workable as it is:
- Untouched levels sit in two price-keyed heaps, so a base bar costs O(1) when nothing is touched.
- ATR and the volume baselines are incremental.
- Expiry is a FIFO per timeframe.
- Ages are counters, not per-bar loops.
- Frames are converted to bars in 50k-row slices. That cut the peak from 544 MB to 332 MB; the remaining peak is
  polars inside `resample`.

**Observations, not tests.**
- The median age at first fill is 17 to 19 tau-bars on every timeframe from 15m to 1w.
- Between 70% and 80% of intraday and daily levels are filled before expiry.
- Monthly levels can only appear from 2021-03 onwards, because the first 14 months are ATR warm-up, and 200 months
  is longer than the whole sample. Weekly and monthly counts are tiny (25 and 6), as phase 1 already warned: crypto
  history is too short for them.
- No result about H1 or H2 is implied. Placebos and labels are phase 3.

## Decisions where SPEC is silent (each marked D)

- **D1 ATR.**
  - Wilder (RMA) of length `legs.atr_len`: `TR_1 = H − L`, then the true range; seeded with the simple mean of the
    first 14 TRs; incremental.
  - Shifted by one bar: bar t sees ATR_{t−1}. The first 14 bars of a timeframe cannot be candidates.
  - `legs.atr_method` is absent from the frozen baseline, so the default `wilder` applies. `simple` is implemented
    but unused. `baseline.yaml` was not edited.
- **D2 Which ATR sizes the zone.** The zone is `O_t ± κ·ATR_{t−1}`: the origin's shifted ATR, the same one as the
  detection threshold and the leg size. Using one unit keeps the leg size and the zone consistent. It is frozen at
  birth: `Level` refuses to rebind zone, ATR or price.
- **D3 Lockout scope.**
  - When a direction-s leg confirms at t+k*, every pending direction-s candidate is discarded, including earlier
    origins not yet confirmed. No bar with index ≤ t+k* can start one.
  - If several confirm on the same bar, the earliest origin wins. The opposite direction is independent.
  - This reading follows from "no new origin of the same direction … until after t+k*" and "one level per leg, at
    its earliest qualifying candle".
- **D4 Counting bars.**
  - k, age and expiry count the trade bars that exist on that timeframe; missing bars (CFD weekends later) and
    no-trade bars are not counted.
  - The strict `>` in the trade-back condition is SPEC's own formula: a high or low equal to the open kills the
    candidate.
- **D5 No-trade bars.**
  - A bar with `h == l` and `v == 0` (or no volume column) is invisible on every timeframe. It creates no candidate,
    does not advance or kill a window, and stays out of the ATR and the volume baseline. It fills nothing, ages
    nothing and flips nothing.
  - The rule is decided from the bar alone, not from `gaps.json`, which is a whole-sample file. So it covers
    venue_flat and symbol_flat alike.
  - BTCUSDT has 63 such 5m bars (44 venue_flat + 19 symbol_flat; every zero-range bar there has zero volume), plus 16,
    5 and 1 fully flat 15m, 30m and 1h bars.
  - A zero-range bar with volume is a real trade and can fill.
- **D6 Fill eligibility.**
  - Only base bars closing strictly after `birth_ts` can fill. The base bar closing at the birth instant traded
    before the level was known.
  - Intersection uses closed intervals (`l ≤ zone_hi` and `h ≥ zone_lo`).
  - Several fills on one bar are emitted in level-id order.
- **D7 Feed order at one instant.**
  - Higher-timeframe bars closing at t are processed before the base bar closing at t. This is the backward as-of
    join on close ≤ t, and `update` enforces it.
  - Consequences: a level born at t counts in the confluence of a fill at t but cannot be filled by that bar; an
    expiry at t takes effect before the base bar at t.
  - The order among higher timeframes at the same instant is irrelevant, because they interact only through
    confluence, which is computed at base bars.
- **D8 Initial polarity.** At the fill, the last close of the level's own timeframe as of t0 sets it: below the zone
  gives resistance, above gives support. If that close is inside the zone, which cannot happen for an untouched level
  on gap-free data, the leg side decides (a down-leg gives resistance). Untouched levels have no polarity.
- **D9 Expiry.** Age is counted from birth in trade tau-bars. The level expires at the close of the A_max-th tau-bar
  after birth, from either live state. An expired level is dropped from memory and from every index.
- **D10 Confluence.**
  - Counts other levels in the registry's timeframes that are active (untouched or reference, birth ≤ t0, not
    expired), the same timeframe included.
  - Overlap uses closed intervals. Levels filled by the same base bar count for each other.
  - The weights come from `levels.timeframe_weights`. 5m levels count only if 5m is in the registry's timeframes; it
    is not in the primary list.
- **D11 Relative leg volume (`leg_vol`).**
  - Defined as Σv over the leg bars t..t+k* divided by Σbaseline over the same bars.
  - Intraday baseline: the mean volume of the same intraday slot (position in the trade day, from the session,
    DST-aware) over the previous 20 sessions that printed the slot. Daily, weekly and monthly: the mean of the
    previous 20 bars.
  - It is `None` without volume, before the baseline is full, or when the baseline is zero.
  - It is stored as a feature only; detection never uses volume.
- **D12 Data types.** Bars are a `NamedTuple` `Bar(tf, ts_close, o, h, l, c, v, ts_open)` with epoch-millisecond UTC
  timestamps. Events are frozen dataclasses: `LevelBorn`, `FirstFill`, `PolarityFlip`, `LevelExpired`.
- **D13 Higher-timeframe bars in the smoke run.**
  - They come from phase 1's `resample` on the research frame, then they are streamed. Each resampled bar depends
    only on base bars inside its bucket (phase 1 property test) and is fed only at its close.
  - A streaming resampler for live use (phase 10) is not built.
  - The smoke run drops `buy_v` and `sell_v` before resampling (not used by levels).

## Files

- `vecopen/src/vecopen/legs.py` (was a stub): `Bar`, `is_no_trade`, `ATR`, `Leg`, `LegOriginDetector`.
- `vecopen/src/vecopen/levels.py` (was a stub): `Level`, the four event types, `LevelRegistry`, and the batch helpers
  `bar_stream` and `run`, which loop over `update`.
- `vecopen/src/vecopen/phase2.py` (new): the reproduce command, smoke run, tables and registry line.
- `vecopen/src/vecopen/__main__.py`: registers `phase2` (and one docstring paragraph).
- `vecopen/tests/test_vo_levels_cases.py` and `vecopen/tests/test_vo_levels_properties.py` (new).
- `vecopen/reports/phase_2.md`, `vecopen/reports/phase_2_tables.md`; `vecopen/trials/registry.jsonl` and
  `registry.head.json` (one appended line).

## Notes for later phases

- **Phase 3.** The placebo levels must pass through the identical code path. That needs a small `LevelRegistry`
  entry to register an externally given price as a level, through the same `_birth`, heaps, state machine and
  confluence. It is not built here, because it is phase 3's deliverable.
- **H2.** `registry.untouched()` gives the live untouched levels at any instant for the H2 distance sampling.
- **Labels.** `FirstFill` carries `atr` (ATR_{t−1} of the origin, D2). Phase 3 decides which ATR scales its barriers.

## Disclosures

- My first directory listing (`Get-ChildItem -Recurse` on `vecopen/`) tried to descend into `vecopen/data/holdout`
  and was refused with `PermissionDenied`. Nothing inside it was listed or read, and every later listing avoided it.
  No holdout file was opened.
- `SPEC.md`, `baseline.yaml`, `sessions.yaml`, `.env`, keys, `config/schema.yaml` and `bin/` were not touched.
- No scheduled task was created. The only processes run were my own pytest and python invocations.
- During mutation testing I edited `legs.py` and `levels.py` temporarily, from backups in the scratchpad, and
  restored them byte-identically (checked with `diff`). The registry line and the test runs above used the restored
  files.
- No diary entry was written; per `vecopen/CLAUDE.md` that is the orchestrator's job.
