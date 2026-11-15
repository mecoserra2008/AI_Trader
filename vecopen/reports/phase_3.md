# Phase 3: events and labels

Date: 2026-11-14 (UTC). Builder: phase 3 subagent. Status: **built and run on all 20 crypto assets and every primary
timeframe.**

| Criterion | Status |
|---|---|
| Stop-first tie rule tested | met |
| Identical registry code path proven by tests | met |
| Matched-sample SMD below 0.1 | met on 15m to 1w and overall. **Not met on 1M:** `departure_atr` has SMD 0.177 with 31 pairs (see "Acceptance") |

No statistic comparing treated and placebo outcomes was computed or printed.

## Reproduce (single command, from the repo root)

```
uv run --extra vecopen python -m vecopen phase3
```

Steps, per asset and one asset at a time:

1. **Event step** (streaming; no bar after the one being processed is ever read).
   - `load_bars`, then phase 1's `resample` to the 8 primary timeframes with the asset's session.
   - `levels.bar_stream` merges the bars into the as-of feed order.
   - Each bar goes into four `TouchDetector`s, one per family (treated, `p1_nonorigin`, `p2_displaced`,
     `p3_candle`). Each detector wraps its own plain `LevelRegistry`.
   - After every tau-bar, `PlaceboGenerator.update(bar, treated levels born at that bar)` returns placebo levels.
     They are registered in the placebo registries before the base bar of the same close.
2. **Label step** (forward-looking, separate). A second pass over the base bars feeds `labels.LabelStream`. Each event
   is added at its t0 and labelled from the bars after it.
3. **Write.** `data/research/events/<ASSET>/{levels,events,labels}.parquet`.

After the last asset:

4. **Family-1 matching** (`phase3.match_all`) reads covariate columns only and writes `<ASSET>/matches.parquet`.
5. The balance tables.
6. `reports/phase_3_tables.md` and `data/research/events/manifest.json` (sha256 per file).
7. `vecopen/tests`.
8. One registry line (phase 3, kind `build`).

Options: `--assets A,B` runs a subset, `--dry-run` skips the append and `--skip-tests` skips the tests. The command
reads `data/research` only. The full run takes 1,404 s; the process peak working set is 956 MB (each asset stays at
about 700 MB; the peak is the cross-asset matching step). POLARS/OMP use 2 threads.

Registry lines of this phase (`config_sha256 9e78f831…ae14a`, baseline unchanged):

| line (seq) | `line_sha256` | status |
|---|---|---|
| 9 (8) | `50d886626fb13408831e8abc042aba361198fd56032ce8ccff5627a9a5fa8e77` | **authoritative**. `diff_sha256 5e398fc4…01ac`, `tables_sha256 e18e7f28…bf3`, `manifest_sha256 e0d94c6f…2b92` |
| 8 (7) | `dcf5c7dec5728d5362cd30ecf84feb0b2070845488c31404f5b47b0b80e54dbc` | **superseded**. See "Disclosures" |

`python -m vecopen verify` reports `registry OK: 9 lines`.

## Acceptance (SPEC section 8, phase 3)

| Criterion | Result |
|---|---|
| `TouchDetector`, `label_event()` triple barrier, `PlaceboGenerator` for the three families | Built in `events.py`, `labels.py` and `placebo.py`. The detector and the generator are streaming `update(bar)`; labels are a separate step |
| Stop-first tie rule tested | `test_vo_labels.py`: `test_stop_first_tie_rule_fade_short` and `test_stop_first_tie_rule_both_labels_and_both_sides` cover both labels and both sides. `test_tie_only_when_both_barriers_are_in_the_same_bar` and `test_open_through_a_barrier_exits_at_the_open_and_is_not_a_tie` show what is not a tie. The mutation "target wins a tie" is caught |
| Placebo levels go through the identical registry code path, proven by a test | `test_registered_levels_follow_the_identical_state_machine` (hypothesis, 60 examples × 2 configs) and `test_identical_code_path_non_vacuous_and_on_real_bars` (random stream plus 45 days of BTCUSDT on 6 timeframes): each feeds the treated registry's own legs into a `LevelRegistry(family="probe")` through `register()`. The event stream is identical to the treated one, bar by bar, except for the family tag. `test_every_placebo_birth_goes_through_the_same_birth_method` spies on `LevelRegistry._birth`: every level of every family is born there, in a plain `LevelRegistry`, and only the treated registry has detectors |
| Matched-sample balance: SMD below 0.1 on the matching covariates | Met on every timeframe from 15m to 1w (max abs SMD 0.078) and overall (max 0.018). **1M: SMD 0.177 on `departure_atr`, with 31 pairs.** The SMD's own noise at n = 31 is about sqrt(2/31) = 0.25, but the criterion is not met there. By (asset, timeframe) cell, all 100 cells with at least 300 pairs are below 0.1 (max 0.084). One of the 20 cells with 30 to 299 pairs is above: DOGEUSDT 1d, 81 pairs, 0.111 |
| Analytic baseline inputs per event (a, b, d, σ, T) | Events table: `a_fade_edge`, `b_fade_edge`, `a_follow_edge` and `b_follow_edge` (near-edge entry), plus `d_edge`, `sigma_birth`, `sigma_t0` and `h_base` (= T, in base bars). Labels table: `fade_a`, `fade_b`, `follow_a` and `follow_b` from the actual entry. Levels table: `d_edge`, `sigma_birth` and `h_base` at birth. Definitions are in D-E3 and D-L6 |
| `vecopen/tests` and root `pytest` green | 168 passed (139 earlier + 29 new); root suite in "Test output" |

## Decisions where SPEC is silent (each marked)

### Labels (`labels.py`)

- **D-L1 ATR unit.** The barriers use the level's frozen ATR at birth: ATR_{t−1} of the origin candle, the unit that
  also sizes the zone (phase 2 D2). It is not re-measured at t0.
- **D-L2 Barrier geometry, anchored at the level ℓ.** A = the unit; g = the side at birth (D-E1).
  - Resolution band: ℓ ± k_pt·A.
  - Fade (direction g):
    - target ℓ + g·k_pt·A;
    - stop k_sl·A beyond the far zone edge: `zone_hi + k_sl A` when g = −1, `zone_lo − k_sl A` when g = +1.
  - Follow-through (direction −g):
    - target ℓ − g·k_pt·A;
    - stop k_sl·A beyond the approach-side edge, which is the edge opposite to the follow trade.
  - So both targets are exactly the band edges. With κ = 0.10, k_pt = 1.0, k_sl = 0.5 and the entry at the near edge,
    that gives:

    | label | a | b |
    |---|---|---|
    | fade | 0.9 A | 0.7 A |
    | follow | 1.1 A | 0.5 A |
- **D-L3 Entry and time.**
  - The entry E is the open of the first trade bar after t0, normally bar j = 1. No-trade bars (`legs.is_no_trade`)
    are skipped for the entry and for barrier checks.
  - The horizon counts every base bar received, j = 1..h_base, so it is wall-clock time on a 24-hour session.
- **D-L4 Horizon.** `h_base = 6 × base bars per tau-bar`:

  | tf | 15m | 30m | 1h | 2h | 4h | 1d | 1w | 1M |
  |---|---|---|---|---|---|---|---|---|
  | base bars per tau-bar | 3 | 6 | 12 | 24 | 48 | 288 | 2,016 | 8,766 |
  | h_base | 18 | 36 | 72 | 144 | 288 | 1,728 | 12,096 | 52,596 |

  The 1M value is 365.25 / 12 days. This is defined for 24-hour sessions only; other sessions raise until CFD data
  exists.
- **D-L5 Execution.**
  - An open at or through a barrier exits there, at the open. This is a gap: the order is known, so it is not a tie.
  - Otherwise, a bar reaching both barriers is a stop (SPEC tie rule; `tie = True`).
  - Exits are at the barrier price. The time barrier exits at the close of bar h_base.
- **D-L6 Void and R.**
  - If E is already at or through a barrier (a ≤ 0 or b ≤ 0), the directional label is void
    (`reason = void_target` / `void_stop`, label and R None). No position with that target and stop can be opened at
    E. Treated 15m void rates are 12% for the fade and 16% for the follow.
  - R = dir(exit − E)/b, per SPEC. `pnl_atr` = dir(exit − E)/A is stored next to it.
  - Phase 5 may pre-register a different mapping, for example void_stop → 0 / −1 R.
- **D-L7 Resolution and MAE.**
  - Resolution: the band is left when a high reaches ℓ + k_pt·A or a low reaches ℓ − k_pt·A.
  - `res_side` is +1 (up), −1 (down) or 0 (both in the first resolving bar).
  - MAE = max(max H − ℓ, ℓ − min L)/A over the whole window, from the entry bar.
- **D-L8 End times and censoring.**
  - `res_end_ts`: the first band exit, else the horizon end.
  - `fade_end_ts` / `follow_end_ts`: the exit bar.
  - `mae_end_ts` = `horizon_end_ts`: the close of bar h_base.
  - If the data ends first, every undecided label is None and `censored` is True.

### Events (`events.py`)

- **D-E1 Side.** `g = sign(ref_close − price)`, where `ref_close` is the tau close at birth (`Leg.confirm_close`); the
  fallback is `Leg.direction`.
  - For treated levels, g = s_t exactly; `_born` asserts it.
  - The same rule gives the side of every placebo, including those with no leg direction.
- **D-E2 Departure distance and regime at birth.**
  - `departure_atr = |ref_close − price|/A`. For treated levels this is `leg_size_atr`, also asserted.
  - `regime = ln(ATR_{c−1} / mean TR_{c−1})` at the source candle c. The mean TR is over up to `a_max_bars` (200)
    trade tau-bars and at least `atr_len` (14). Both terms are shifted by one bar.
  - c is the origin candle for treated levels and families 2 and 3, and the candle itself for family 1.
  - The ATR is a replica of the detector's (same class); `_born` asserts that it equals the level's frozen ATR,
    which also proves that the bar indices agree.
- **D-E3 Touch-baseline inputs.**
  - `d_edge = max(|ref_close − price| − κA, 0)`: the distance to the near zone edge, in price units.
  - σ = the RMS of the 5m close-to-close price changes over the last h_base changes. It is driftless and shifted:
    the touch bar is not included in `sigma_t0`.
  - T = h_base.
- **D-E4 Approach side.** `sign(prev base close − ℓ)`. All 349,259 treated events approach from the birth side (0
  mismatches), because no gap jumped a zone.
- **D-E5 Confluence** in placebo registries is computed among levels of the same family (one registry per family).
  It is not comparable with treated confluence; phase 6 owns features.

### Registry entry point (`levels.py`, minimal)

- **D-R1.** These additions are the only changes to `levels.py`:
  - `LevelRegistry(family=...)`; only `family="treated"`, the default, builds detectors.
  - `register(leg)` calls the unchanged `_birth` with the same tau-bar count and last close as a detected leg.
  - `register` refuses unless it is called at `leg.birth_ts`, after that timeframe's bar and before the base bar of
    the same close. That is the birth point of a detected leg (phase 2 D7).
  - A `family` field on `Level`, `LevelBorn` and `FirstFill`.

  The state machine, zone, touch and confluence code is untouched, and all phase 2 tests still pass. For a placebo
  `Leg`, `origin_o` is the placebo price and `direction` is its g.

### Placebos (`placebo.py`)

- **D-P1 Family 1 (non-origin opens).**
  - **Candidates.** Any trade bar c that is not a leg origin, with ATR_{c−1} > 0.
  - **Birth.** It becomes a placebo level at the departure point of the leg rule: the first k in [0, n] with
    |C_{c+k} − O_c| ≥ θ_L·ATR_{c−1}, where no bar c+1..c+k trades back to the open (strict test, in the departure
    direction).
  - **Not applied:** the candle direction, the wick test and the lockout. These are what make an origin.
  - **Result.** Departure at birth has the same support as treated levels (≥ 2 ATR).
  - **Origin check.** Done at the departure bar, against all origins confirmed so far. An origin in the opposite
    direction could only appear later through a gap; such levels are flagged `late_origin` and excluded (0 found).
  - **Pool.** This is the matching pool.
- **D-P2 Family 2 (displaced).** Both signs are always used (variants `up` and `down`): 440,504 each.
  - The two upsilon values are drawn independently from U[1, 3] (`stats.placebo_displacement_atr`), up before down,
    in stream order.
  - The generator is `numpy.random.default_rng([seed, crc32(asset)])`, so draws do not depend on the asset order.
  - Born with the treated level: same instant, ATR and zone width.
- **D-P3 Family 3 (same candle).** The prices are the close, the midpoint (H+L)/2 and the far extreme in the leg
  direction: the low of a down-leg candle, the high of an up-leg one.
  - The open-side extreme lies within ω̄ of the range from the open and equals the open when ω = 0. That happens for
    50,293 of 440,504 treated levels (11.4%), so it cannot be a distinct placebo.
  - Born with the treated level.
- **D-P4 Matching unit and covariates.**
  - The unit is the first-fill event: "age" only exists at the touch.
  - Covariates, all known at t0:
    - `age_log = ln(1 + age in tau-bars)`;
    - `departure_atr` at birth;
    - `regime` at birth.
  - Distance: Euclidean, each covariate divided by its SD over the treated and pool events of the cell.
  - **Pre-declared ratio: 1:1 nearest neighbour with replacement**, exact on the calendar year of t0. The year
    constraint keeps the by-year analysis of gate G2 within-year.
  - The pool excludes `late_origin` levels.
- **D-P5 Matching cells.** One (asset, timeframe) cell for 15m to 4h. For 1d, 1w and 1M the cell is one timeframe
  across all 20 assets, because one asset has about 150, 20 and 7 family-1 levels there in five years (BTCUSDT).
  - The covariates are scale-free, so cross-asset matches are comparable.
  - 1,367 of the 1,714 1d matches, 174 of 254 1w and 28 of 31 1M are to another asset. `matches.parquet` stores
    `placebo_asset`.
  - I took this decision after a covariate-only diagnostic on BTCUSDT, where within-asset 1w matching gave an SMD of
    −0.77 on 15 pairs. No placebo outcome was seen; see "Disclosures".
- **D-P6 What is not reported, to keep G2 blind.** Placebo first-fill counts, event and label row counts, and
  pre-matching balance of age are withheld. With the births, the first two give touch rates (H2), and age at fill is
  the time to touch for unmatched levels. Pre-matching balance is shown for birth covariates only.

## Tables (from `reports/phase_3_tables.md`, generated by the reproduce command)

### Treated first-fill events per asset and timeframe

| asset | 15m | 30m | 1h | 2h | 4h | 1d | 1w | 1M | total |
|---|---|---|---|---|---|---|---|---|---|
| BTCUSDT | 8,704 | 4,176 | 2,064 | 1,068 | 510 | 90 | 15 | 3 | 16,630 |
| ETHUSDT | 8,909 | 4,312 | 2,155 | 1,113 | 561 | 101 | 18 | 3 | 17,172 |
| SOLUSDT | 8,809 | 4,253 | 2,030 | 1,018 | 498 | 77 | 18 | 3 | 16,706 |
| XRPUSDT | 8,899 | 4,273 | 2,040 | 990 | 509 | 78 | 15 | 3 | 16,807 |
| BNBUSDT | 9,876 | 4,610 | 2,247 | 1,129 | 556 | 85 | 11 | 5 | 18,519 |
| DOGEUSDT | 8,363 | 4,031 | 1,888 | 930 | 462 | 81 | 7 | 1 | 15,763 |
| ADAUSDT | 9,520 | 4,551 | 2,199 | 1,121 | 541 | 88 | 12 | 1 | 18,033 |
| AVAXUSDT | 8,908 | 4,320 | 2,115 | 1,027 | 521 | 87 | 14 | 2 | 16,994 |
| LINKUSDT | 9,749 | 4,602 | 2,204 | 1,163 | 550 | 86 | 13 | 1 | 18,368 |
| LTCUSDT | 9,367 | 4,417 | 2,149 | 1,046 | 536 | 103 | 15 | 1 | 17,634 |
| BCHUSDT | 9,484 | 4,441 | 2,168 | 1,092 | 534 | 96 | 12 | 1 | 17,828 |
| DOTUSDT | 8,893 | 4,241 | 2,047 | 982 | 493 | 79 | 15 | 0 | 16,750 |
| TRXUSDT | 10,346 | 4,959 | 2,445 | 1,238 | 653 | 95 | 10 | 3 | 19,749 |
| ATOMUSDT | 10,338 | 4,850 | 2,356 | 1,115 | 571 | 93 | 13 | 1 | 19,337 |
| ETCUSDT | 9,274 | 4,342 | 2,054 | 1,016 | 517 | 86 | 13 | 1 | 17,303 |
| FILUSDT | 8,071 | 3,749 | 1,814 | 865 | 446 | 65 | 9 | 0 | 15,019 |
| XLMUSDT | 10,463 | 4,882 | 2,231 | 1,112 | 550 | 90 | 9 | 0 | 19,337 |
| NEARUSDT | 8,883 | 4,210 | 2,040 | 1,015 | 477 | 74 | 16 | 0 | 16,715 |
| UNIUSDT | 9,183 | 4,356 | 2,167 | 1,045 | 521 | 84 | 11 | 1 | 17,368 |
| AAVEUSDT | 9,148 | 4,300 | 2,113 | 1,068 | 513 | 76 | 8 | 1 | 17,227 |
| **all** | **185,187** | **87,875** | **42,526** | **21,153** | **10,519** | **1,714** | **254** | **31** | **349,259** |

BTCUSDT equals phase 2's smoke run exactly (16,630 first fills).

### Placebo counts: levels born per family and timeframe (all 20 assets)

| family | 15m | 30m | 1h | 2h | 4h | 1d | 1w | 1M | total |
|---|---|---|---|---|---|---|---|---|---|
| treated | 231,959 | 111,210 | 54,035 | 27,051 | 13,541 | 2,281 | 368 | 59 | 440,504 |
| p1_nonorigin (pool) | 262,470 | 125,456 | 61,771 | 30,554 | 15,451 | 2,656 | 375 | 60 | 498,793 |
| p2_displaced | 463,918 | 222,420 | 108,070 | 54,102 | 27,082 | 4,562 | 736 | 118 | 881,008 |
| p3_candle | 695,877 | 333,630 | 162,105 | 81,153 | 40,623 | 6,843 | 1,104 | 177 | 1,321,512 |

The per-asset split is in `phase_3_tables.md`. Families 2 and 3 are exactly 2× and 3× treated per asset.

### Family 1 matching (1:1 nearest neighbour with replacement, exact on year)

| tf | cell | treated events | matched | unmatched | distinct placebo events used | max reuse | matched to another asset | median distance (SD units) |
|---|---|---|---|---|---|---|---|---|
| 15m | asset | 185,187 | 185,187 | 0 | 117,313 | 9 | 0 | 0.125 |
| 30m | asset | 87,875 | 87,875 | 0 | 55,858 | 9 | 0 | 0.159 |
| 1h | asset | 42,526 | 42,526 | 0 | 27,345 | 8 | 0 | 0.204 |
| 2h | asset | 21,153 | 21,153 | 0 | 13,600 | 9 | 0 | 0.257 |
| 4h | asset | 10,519 | 10,519 | 0 | 6,842 | 7 | 0 | 0.322 |
| 1d | all assets | 1,714 | 1,714 | 0 | 1,084 | 6 | 1,367 | 0.179 |
| 1w | all assets | 254 | 254 | 0 | 147 | 5 | 174 | 0.237 |
| 1M | all assets | 31 | 31 | 0 | 16 | 4 | 28 | 0.460 |

### Balance table: matched sample, treated first fills vs matched family-1 first fills

SMD = (mean_T − mean_C)/sqrt((var_T + var_C)/2) over the pairs. A placebo used twice counts twice.

| tf | pairs | SMD age_log | SMD departure_atr | SMD regime |
|---|---|---|---|---|
| 15m | 185,187 | -0.001 | 0.012 | 0.003 |
| 30m | 87,875 | -0.001 | 0.017 | 0.006 |
| 1h | 42,526 | -0.004 | 0.024 | 0.005 |
| 2h | 21,153 | 0.001 | 0.038 | 0.007 |
| 4h | 10,519 | -0.005 | 0.052 | 0.020 |
| 1d | 1,714 | -0.000 | 0.025 | 0.016 |
| 1w | 254 | -0.009 | 0.078 | 0.018 |
| 1M | 31 | -0.033 | **0.177** | -0.036 |
| **all** | 349,259 | -0.001 | 0.018 | 0.005 |

The means and SDs are in `phase_3_tables.md`, section 4. By (asset, timeframe) cell:

| cell size (pairs) | cells | with any abs SMD ≥ 0.1 | max abs SMD |
|---|---|---|---|
| < 30 | 36 | 20 | 0.891 |
| 30 to 299 | 20 | 1 (DOGEUSDT 1d: 0.111 on departure) | 0.111 |
| ≥ 300 | 100 | 0 | 0.084 |

Before matching, at birth, treated levels vs the family-1 pool (birth covariates only): abs SMD ≤ 0.052 on 15m to
4h. On 1d to 1M the regime SMD is 0.137 (1d) and 0.143 (1w), and the departure SMD is 0.392 on 1M. Matching removes
these differences, except on 1M.

### Labels summary (TREATED events only; placebo labels are written but not summarised)

| tf | events | censored | resolution rate | mean MAE (ATR) | fade n | fade target-first | fade mean PnL (ATR) | follow n | follow target-first | follow mean PnL (ATR) |
|---|---|---|---|---|---|---|---|---|---|---|
| 15m | 185,187 | 10 | 0.925 | 2.843 | 162,452 | 0.459 | 0.013 | 154,617 | 0.311 | -0.028 |
| 30m | 87,875 | 12 | 0.922 | 2.840 | 79,716 | 0.453 | 0.008 | 76,524 | 0.303 | -0.020 |
| 1h | 42,526 | 10 | 0.919 | 2.838 | 39,351 | 0.452 | 0.008 | 37,893 | 0.299 | -0.014 |
| 2h | 21,153 | 8 | 0.921 | 2.844 | 19,768 | 0.445 | -0.001 | 19,165 | 0.300 | -0.002 |
| 4h | 10,519 | 5 | 0.923 | 2.892 | 9,866 | 0.450 | 0.019 | 9,694 | 0.289 | -0.027 |
| 1d | 1,714 | 6 | 0.930 | 3.078 | 1,626 | 0.441 | 0.007 | 1,592 | 0.308 | -0.001 |
| 1w | 254 | 12 | 0.937 | 3.208 | 247 | 0.514 | 0.137 | 248 | 0.242 | -0.100 |
| 1M | 31 | 9 | 0.966 | 4.340 | 26 | 0.423 | -0.009 | 28 | 0.214 | -0.205 |

Mean R, max R, voids, ties and time exits are in `phase_3_tables.md`, section 6. The tie rule fired on 2,011 fade
and 2,412 follow labels on 15m.

**Mean R is not robust.** The follow label on 2h has a max R of 14,799. Its b is 6.6e-7 on a price of 0.41: the entry
sits a fraction of a tick inside the stop. That one label moves the 2h mean R to 0.797. Use `pnl_atr`, or R after the
tick-based cost model of phase 7.

## Tables saved for phase 5

`vecopen/data/research/events/<ASSET>/` (zstd parquet, 811 MB in total; sha256 per file in `manifest.json` and in
registry line 9):

- **`levels.parquet`**: one row per level, all families.
  - `family`, `level_id`, `event_key`, `tf`, `birth_ts`, `price`, `zone_lo`, `zone_hi`, `atr`, `g`, `leg_direction`;
  - `origin_index`, `origin_ts_close`, `k_star`, `leg_size_atr`, `leg_vol`, `omega`;
  - `ref_close`, `departure_atr`, `d_edge`, `regime`, `sigma_birth`, `h_base`;
  - `source_level_id`, `variant`, `displacement_atr`;
  - `fill_ts`, `expire_ts`, `late_origin`.

  This is the input for H2: untouched intervals are birth to fill or expiry.
- **`events.parquet`**: one row per first fill, all families.
  - The level columns, plus `t0`, `year`, `age`, `age_log`, `age_hours`, `polarity`, `confluence` (within family),
    `approach_side` and `prev_close`;
  - `sigma_t0` and the touch bar's `touch_o/h/l/c`;
  - `a_fade_edge`, `b_fade_edge`, `a_follow_edge`, `b_follow_edge`.
- **`labels.parquet`**: one row per event, key `(asset, event_key)`.
  - `t0`, `h_base`, `horizon_end_ts`, `entry_ts`, `entry`, `bars_seen`, `censored`;
  - `res`, `res_side`, `res_end_ts`, `mae_atr`, `mae_end_ts`;
  - per side (`fade_`, `follow_`): `dir`, `target`, `stop`, `a`, `b`, `label`, `r`, `pnl_atr`, `exit`, `reason`,
    `end_ts`, `tie`.
- **`matches.parquet`**: `asset`, `tf`, `year`, `treated_key`, `placebo_asset`, `placebo_key`, `distance`, `cell`.

  Families 2 and 3 are paired to their treated level by `source_level_id` (`variant`); they need no matching.

## Test output

```
uv run --extra vecopen pytest vecopen/tests     168 passed in 83.71s (139 earlier + 29 new)
python -m vecopen phase3 (tests step, line 9)   {'tests': 168, 'passed': 168, 'failed': 0, 'errors': 0, 'skipped': 0}
uv run pytest (root)                            583 passed, 2 warnings in 205.59s (vecopen/tests is not collected there)
python -m vecopen verify                        registry OK: 9 lines
uvx ruff check --select F,E9 (new/changed files) All checks passed!
```

The 29 new tests:

- **`test_vo_labels.py` (13).**
  - Frozen parameters and barrier geometry for both sides.
  - The tie rule: three tests, plus the gap case that is not a tie.
  - Time barrier: at most h bars are read.
  - Voids, no-trade entry, resolution and MAE, censoring.
  - h_base per timeframe and the non-24h refusal.
  - `LabelStream` == `label_event` (hypothesis, 60 examples).
- **`test_vo_events_placebo.py` (16).**
  - `TouchDetector` fields by hand on phase 2's LEG case: g, departure 2.0, regime ln(1/1) = 0, d_edge 1.9, σ at
    birth and at t0, approach side, age, polarity.
  - `side_at_birth`.
  - The whole event step (treated + 3 placebo families + generator) is invariant to truncation and to a perturbation
    of all prices after the cut (hypothesis, 60 examples × 2 configs).
  - Every family is produced.
  - The identical code path: three tests, described in "Acceptance".
  - `register` guards.
  - Family 1 by hand: a non-origin candle with wick 0.4 is born at its departure, with k = 1, price 100, departure
    2.0. Strict trade-back; origin exclusion.
  - Families 2 and 3 by hand: seeded draws, up before down, the far extreme for a down-leg and an up-leg.
  - `match_events` by hand: exact year, with replacement, a missing covariate.
  - SMD by hand and pooled moments.
  - No event, level, placebo or feature module imports `labels`.
  - Real ETHUSDT slice: treated touch events equal the plain registry's first fills, every event is labelled, and the
    entry is at the open after t0.

**Mutation check.** I injected 11 one-line bugs into `src`, one at a time, and the new tests caught each one:

- the tie resolved as a target;
- gap-through stop ignored;
- horizon off by one;
- wrong stop side;
- entry at the close;
- `register` with count + 1;
- ATR replica not updated;
- touch bar included in σ;
- non-strict trade-back in family 1;
- origin exclusion removed;
- wrong extreme.

The sources were restored from backups and checked byte-identical before the registry runs.

## Notes for later phases

- **SPEC arithmetic, for the orchestrator.** SPEC section 3 says that "even a driftless random walk gives a rejection
  probability above one half". With the frozen κ = 0.10, k_pt = 1.0 and k_sl = 0.5, and the entry at the near edge,
  b/(a+b) is below one half under D-L2:
  - fade: 0.7/1.6 = 0.4375;
  - follow: 0.5/1.6 = 0.3125.

  This is parameter arithmetic, not data. Phase 5 should use the per-event a and b, which are stored.
- **1M** has 31 treated events in total and its balance criterion fails. Gate G2 should pre-register how 1M is
  handled: pooled into a daily-plus timeframe group, or excluded from the family-1 comparison. The data cannot give
  more: crypto history is 6.1 to 6.9 years.
- **Voids** (D-L6) are 12 to 16% of treated directional labels on 15m and fewer on higher timeframes. A rule for
  voids belongs in the G2 pre-registration.
- **Families 2 and 3** are born at the treated birth instant. Some sit inside the current range, so they can be
  touched by the next base bar. That is their definition, not a bug.
- **Cross-asset matches** on 1d, 1w and 1M: a BTCUSDT daily event can be matched to an ETHUSDT placebo. Phase 5's
  cluster bootstrap should cluster on the treated event's calendar week, as SPEC says.
- **Phase 6.** Event confluence in the placebo registries is within-family (D-E5). The treated-level confluence of a
  placebo touch, if wanted, is a feature to build there.
- **Phase 10.** The streaming classes are ready for live bars, except the resampler, as in phase 2 D13.

## Files

- **`vecopen/src/vecopen/events.py`** (was a stub): `TouchDetector`, `TouchEvent`, `LevelInfo`, `TfVol`, `Sigma`,
  `side_at_birth`, `horizon_base_bars` and `base_bars_per_tau`.
- **`vecopen/src/vecopen/labels.py`** (was a stub): `label_event`, `LabelStream`, `LabelState`, `Label` and
  `BarrierOutcome`.
- **`vecopen/src/vecopen/placebo.py`** (was a stub): `PlaceboGenerator`, `PlaceboSpec`, `rng_for`, `match_events`,
  `smd`, `Moments` and `smd_moments`.
- **`vecopen/src/vecopen/levels.py`**: the minimal entry point (D-R1) and docstring.
- **`vecopen/src/vecopen/phase3.py`** (new): the reproduce command, with `event_step`, `label_step`, `match_all`, the
  tables and the registry line.
- **`vecopen/src/vecopen/__main__.py`**: registers `phase3`.
- **Tests:** `vecopen/tests/test_vo_labels.py` and `vecopen/tests/test_vo_events_placebo.py` (new).
- **Reports and registry:** `vecopen/reports/phase_3.md` and `vecopen/reports/phase_3_tables.md`;
  `vecopen/trials/registry.jsonl` and `registry.head.json` (lines 8 and 9).
- **Data:** `vecopen/data/research/events/` (80 parquet files plus `manifest.json`).

## Disclosures

- **Superseded registry line 8.** The first full run (line 8) wrote a tables file whose section 7 listed event and
  label row counts per asset, summed over all families, and a total label-row line.
  - Combined with the level counts, these numbers give a pooled placebo touch count, which can be read as a crude
    touch rate. That is an H2-type outcome aggregate.
  - I saw that table before noticing.
  - I removed those counts from the tables, the manifest and the stored file records, added `pnl_atr`, and reran.
    Line 9 is the authoritative run; its outputs overwrote line 8's.
  - Line 8's metrics never contained those counts; they held only the tables hash.
  - No placebo-vs-treated outcome statistic was computed, and no design choice was changed after that table was seen.
    The two changes after line 8 were the leak removal and the `pnl_atr` column; neither touches matching.
- **Treated-only outcomes in diagnostics.** Before the full run, smoke runs printed the treated-only label summary
  (BTCUSDT; BTCUSDT + ETHUSDT on 2 to 4 months). D-P5 was decided from covariate balance alone.
- **One extra read of treated labels.** After line 8, I read the treated labels once (max R) to diagnose the R tail.
- **Holdout.** My first listings of `vecopen/` and `vecopen/data/` showed the name of the `holdout` folder as a
  directory entry. Nothing inside it was listed, read or sampled; all reads used explicit `data/research` paths.
- **Mutation testing.** I edited `labels.py`, `levels.py`, `events.py` and `placebo.py` temporarily, from scratchpad
  backups, and restored them byte-identically (checked with `diff`) before the registry runs.
- **Untouched.** `SPEC.md`, `baseline.yaml`, `sessions.yaml`, `.env`, keys, `config/schema.yaml` and `bin/`.
- **Processes.** No scheduled task. Only my own pytest and python processes ran.
- **Diary.** No diary entry was written; that is the orchestrator's job (`vecopen/CLAUDE.md`).

## Amendment, 2026-11-14 (UTC): balance per timeframe group

**Why.** After the 1M balance failure, the user decided (SPEC section 9, 2026-11-14 ~17:50 UTC; registry line 10,
kind `decision`) that phase 3 balance and the G2 test family are judged per timeframe group:

- **intraday:** 15m, 30m, 1h, 2h, 4h;
- **daily_plus:** 1d, 1w, 1M.

**What was done.**
- `python -m vecopen phase3-balance` (new; `phase3.py`) reads the stored outputs of the authoritative run (line 9)
  and nothing else:
  - `matches.parquet`;
  - from `events.parquet`, only `asset`, `event_key`, `family` and the three covariates.
- No `labels.parquet` or `levels.parquet` is opened, no event is recomputed, and matching and parameters are
  unchanged.
- Before reading, it checks every file it reads against line 9's `manifest.json`: the manifest sha256 equals line 9's
  `manifest_sha256`, and each `events.parquet` and `matches.parquet` equals its manifest sha256.
- SMDs are pooled with `placebo.Moments` / `smd_moments` over every asset and timeframe of the group. Each
  (asset, timeframe) cell is accumulated once and merged into its timeframe, its group and the overall row.
- **Cross-check.** The per-timeframe breakdown reproduces line 9's balance table to 1.3e-15, with identical pair
  counts.
- The only count shown is matched pairs, which equals the treated first-fill count already published. No outcome,
  label, touch count, placebo event count or row count was computed or printed.

**Group balance (acceptance: max |SMD| < 0.1 in each group): met in both groups.**

| group | timeframes | pairs | SMD age_log | SMD departure_atr | SMD regime | max abs SMD | below 0.1 |
|---|---|---|---|---|---|---|---|
| intraday | 15m, 30m, 1h, 2h, 4h | 347,260 | -0.001 | 0.017 | 0.005 | 0.017 | yes |
| daily_plus | 1d, 1w, 1M | 1,999 | -0.002 | 0.037 | 0.016 | 0.037 | yes |

Breakdown per timeframe (unchanged from line 9):

| tf | group | pairs | SMD age_log | SMD departure_atr | SMD regime | max abs SMD | below 0.1 |
|---|---|---|---|---|---|---|---|
| 15m | intraday | 185,187 | -0.001 | 0.012 | 0.003 | 0.012 | yes |
| 30m | intraday | 87,875 | -0.001 | 0.017 | 0.006 | 0.017 | yes |
| 1h | intraday | 42,526 | -0.004 | 0.024 | 0.005 | 0.024 | yes |
| 2h | intraday | 21,153 | 0.001 | 0.038 | 0.007 | 0.038 | yes |
| 4h | intraday | 10,519 | -0.005 | 0.052 | 0.020 | 0.052 | yes |
| 1d | daily_plus | 1,714 | -0.000 | 0.025 | 0.016 | 0.025 | yes |
| 1w | daily_plus | 254 | -0.009 | 0.078 | 0.018 | 0.078 | yes |
| 1M | daily_plus | 31 | -0.033 | 0.177 | -0.036 | 0.177 | **no** |
| **all** | | 349,259 | -0.001 | 0.018 | 0.005 | 0.018 | yes |

1M still fails on its own (31 pairs). Within daily_plus it is 31 of 1,999 pairs. Means and SDs per group are in
`reports/phase_3_group_balance.md`, which the command generates.

**Reproduce (single command, from the repo root):**

```
uv run --extra vecopen python -m vecopen phase3-balance
```

`--dry-run` skips the registry append; `--skip-tests` skips the tests. The run takes 84 s with the tests. Peak working
set 437 MB, 2 threads.

**Registry.** Line 11 (seq 10), phase 3, kind `build`:
- `line_sha256 0c8c88f38a9d09808e9941d7f43cf94cdb94f77b7e2929bb2664f5843cda528f`;
- `tables_sha256 888dee48…bf3c9`;
- metrics: `group_balance`, `by_tf`, `all`, `threshold`, `acceptance_met = true`, `source_line_sha256` (line 9),
  `manifest_sha256` and `tests`.

`python -m vecopen verify` reports `registry OK: 11 lines`.

**Tests.**
- `vecopen/tests/test_vo_group_balance.py` (7 new):
  - the groups partition the primary timeframes and match SPEC section 9;
  - pooled group SMDs equal `smd` on the concatenated pairs;
  - imbalanced or empty groups fail;
  - an unknown timeframe raises;
  - the cross-asset placebo lookup works and unmatched rows are dropped;
  - a pair with a missing covariate raises;
  - the manifest and file hash checks work;
  - the reader code names no label or level table.
- `uv run --extra vecopen pytest vecopen/tests`: 175 passed (168 earlier + 7 new).

**Files.**
- `vecopen/src/vecopen/phase3.py`: new functions only, after `_jsonable`, plus the `phase3-balance` subparser. The
  `phase3` command and its logic are unchanged.
- `vecopen/src/vecopen/__main__.py`: docstring.
- `vecopen/tests/test_vo_group_balance.py`.
- `vecopen/reports/phase_3_group_balance.md` (generated).
- `SPEC.md`, the config, `data/holdout/` and `tests/leak/` were not touched.
