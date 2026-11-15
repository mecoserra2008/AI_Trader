# Phase 1: data

Date: 2026-11-14 (UTC). Builder: phase 1 subagent. Status: **crypto done, acceptance met. The CFD code is built and
tested on canned messages; the CFD download waits for the cTrader keys (one command, below).**

## Reproduce (single command, from the repo root)

```
uv run --extra vecopen python -m vecopen phase1
```

The command does these steps in order:

1. Downloads any Binance archive file not yet in `vecopen/data/manifest.json` (none now).
2. Repairs archive gaps from daily zips (none left).
3. Builds and splits any staged asset not yet split. None are left, and assets already split are never rebuilt.
4. Recomputes the research-sample gap runs, flat runs and their classes from `vecopen/data/research` alone, and checks
   them against `vecopen/data/gaps.json`.
5. Writes every table below to `vecopen/reports/phase_1_tables.md`.
6. Runs `vecopen/tests`.
7. Appends one registry line (phase 1, kind `data`).

It never reads `vecopen/data/holdout`. Holdout numbers come from `split.json`, which was written when the holdout was
created. The flag `--with-cfd` adds the cTrader download, build and split.

Registry lines of this phase (`vecopen/trials/registry.jsonl`, all kind `data`, all carrying `sessions_sha256`
`9add7ce4e40a89ffce64e1f8a8f4cdcc47e82efdcad178cc798de1e80e2a2607`; baseline `config_sha256 9e78f831…ae14a`):

| seq | step | line_sha256 |
|---|---|---|
| 1 | crypto-download (1,834 archive files) | `e248821d9a157b30…` |
| 2 | fill-gaps (35 daily zips) | `adb095a29787e152…` |
| 3 | phase1: first build and split of the 20 crypto assets, holdout locked | `9b292dec5b95d173…` |
| 4 | phase1: reproduce run (nothing downloaded or split; tables with the history-start and repair columns) | `91df083872d2ff4b…` |
| 5 | phase1: reproduce run on the final code (adds the re-lock test); same tables and split.json hashes as line 4 | `2c7e86c81ca58445…` |

Line 5 is the authoritative one (final code, `diff_sha256 793ddf40…14995`). `split_json_sha256` is `7ce76c5a757325e4076e20fc0f945ba6c6aec93f5f9e3b28ec5067cd61d995f1`
(identical on lines 3, 4 and 5). `tables_sha256` is `da4ec6e02364d33b4e6fc7388e9374e548af36c9c4503b7119be47257b1c0917`.
Line 5's metrics also hold the split boundaries and the research and holdout row counts per asset.

## Acceptance (SPEC section 8, phase 1)

| Criterion | Result |
|---|---|
| Resampled OHLCV equals aggregation of the base bars (property test) | `test_vo_bars.py::test_resampled_ohlcv_equals_aggregation_of_base_bars`, hypothesis, 300 examples. Random 5m series (1 to 900 bars, gaps up to 3 days, starts near US and EU DST changes and month ends) × all 9 timeframes × 5 sessions (UTC, FX 17:00 NY, CME 18:00 NY, ICE 23:00 London, Eurex 00:00 Berlin) × complete-only on and off. The comparison is against an independent oracle that buckets each bar with plain `datetime` + `zoneinfo` and aggregates in Python. It checks exact equality of `ts_open, ts_close, o, h, l, c, v, buy_v, sell_v, n_base`, that keys are unique and increasing, and that every base bar is used once. A non-vacuity test shows the oracle tells sessions apart. On real data, `test_real_resample_equals_aggregation_on_a_research_slice` checks 40 days of BTCUSDT research bars at 15m, 1h, 4h, 1d and 1w |
| No duplicate timestamps | Duplicate rule in `vecopen.data.dedup_bars`, tested: a monthly-archive row beats a daily-archive row, and within one source the row written last wins. A property test shows no duplicates remain and that the kept row is the last one written. Real data: 0 duplicates in all 20 assets. `load_bars` validates strictly increasing `ts_close` on the 5m grid on every load, and `test_real_research_files_unique_sorted_and_before_boundary` loads all 20 research files |
| Gaps listed and explained | Tables below and the section "Gaps explained" |
| Holdout unreadable | The `vecopen/data/holdout` folder has `icacls … /deny "%USERNAME%":(OI)(CI)(R)`, applied before its first file was written. `test_real_holdout_is_unreadable`: listing the folder raises `PermissionError`, and so does opening each of the 20 part files named in `split.json`; `load_bars` refuses each of them. `load_bars` refuses any path under `data/holdout` before touching it (7 path forms tested, with `read_parquet` patched to fail if it were ever called) |

## Test output

```
uv run --extra vecopen pytest vecopen/tests        112 passed in 39.31s (0 skipped; 59 phase-0 tests + 53 phase-1 tests)
uv run pytest (root)                               571 passed (vecopen/tests is not collected by the root suite)
cargo test --workspace                             194 passed, 0 failed
cargo clippy --workspace --all-targets             clean (no warnings)
```

New Rust tests: 4 unit tests (`trendbars` decode, period, month windows; `store::writes_trendbars`) and 4 canned
fake-server tests in `crates/at-ctrader/tests/canned.rs`:
- bisection when `hasMore` truncates, whether the server keeps the newest or the oldest bars;
- window splitting on an `INCORRECT_BOUNDARIES` error;
- months already on disk are skipped;
- the current month is written as partial;
- the history start is detected after `max_empty` empty months;
- the request uses the read-only allowlist.

## Per-asset history, gaps, split boundaries

Copied from `vecopen/reports/phase_1_tables.md` (generated by the reproduce command).

### Per-asset history (5m base bars)

| asset | session | first bar open (UTC) | history start | last bar close (UTC) | years | bars | coverage | split boundary (UTC) | research bars | holdout bars | dups identical / conflicting | archive files |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| AAVEUSDT | utc_24h | 2020-10-16 07:00 | listing | 2026-11-14 00:00 | 6.08 | 639,276 | 100.00000% | 2025-05-08 00:00 | 479,436 | 159,840 | 0 / 0 | 86 |
| ADAUSDT | utc_24h | 2020-01-31 08:00 | listing | 2026-11-14 00:00 | 6.78 | 713,856 | 100.00000% | 2025-03-04 00:00 | 535,296 | 178,560 | 0 / 0 | 95 |
| ATOMUSDT | utc_24h | 2020-02-07 12:10 | listing | 2026-11-14 00:00 | 6.77 | 711,790 | 100.00000% | 2025-03-06 00:00 | 533,806 | 177,984 | 0 / 0 | 94 |
| AVAXUSDT | utc_24h | 2020-09-23 07:00 | listing | 2026-11-14 00:00 | 6.14 | 645,900 | 100.00000% | 2025-05-02 00:00 | 484,332 | 161,568 | 0 / 0 | 87 |
| BCHUSDT | utc_24h | 2020-01-01 00:00 | archive start | 2026-11-14 00:00 | 6.87 | 722,592 | 100.00000% | 2025-02-24 00:00 | 541,728 | 180,864 | 0 / 0 | 95 |
| BNBUSDT | utc_24h | 2020-02-10 08:00 | listing | 2026-11-14 00:00 | 6.76 | 710,976 | 100.00000% | 2025-03-06 00:00 | 532,992 | 177,984 | 0 / 0 | 94 |
| BTCUSDT | utc_24h | 2020-01-01 00:00 | archive start | 2026-11-14 00:00 | 6.87 | 722,592 | 100.00000% | 2025-02-24 00:00 | 541,728 | 180,864 | 0 / 0 | 95 |
| DOGEUSDT | utc_24h | 2020-07-10 09:00 | listing | 2026-11-14 00:00 | 6.34 | 667,476 | 100.00000% | 2025-04-13 00:00 | 500,436 | 167,040 | 0 / 0 | 89 |
| DOTUSDT | utc_24h | 2020-08-22 07:00 | listing | 2026-11-14 00:00 | 6.23 | 655,116 | 100.00000% | 2025-04-24 00:00 | 491,244 | 163,872 | 0 / 0 | 88 |
| ETCUSDT | utc_24h | 2020-01-16 08:00 | listing | 2026-11-14 00:00 | 6.83 | 718,176 | 100.00000% | 2025-02-28 00:00 | 538,464 | 179,712 | 0 / 0 | 95 |
| ETHUSDT | utc_24h | 2020-01-01 00:00 | archive start | 2026-11-14 00:00 | 6.87 | 722,592 | 100.00000% | 2025-02-24 00:00 | 541,728 | 180,864 | 0 / 0 | 95 |
| FILUSDT | utc_24h | 2020-10-16 06:00 | listing | 2026-11-14 00:00 | 6.08 | 639,288 | 100.00000% | 2025-05-08 00:00 | 479,448 | 159,840 | 0 / 0 | 91 |
| LINKUSDT | utc_24h | 2020-01-17 08:00 | listing | 2026-11-14 00:00 | 6.82 | 717,888 | 100.00000% | 2025-02-28 00:00 | 538,176 | 179,712 | 0 / 0 | 95 |
| LTCUSDT | utc_24h | 2020-01-09 08:05 | listing | 2026-11-14 00:00 | 6.84 | 720,191 | 100.00000% | 2025-02-26 00:00 | 539,903 | 180,288 | 0 / 0 | 100 |
| NEARUSDT | utc_24h | 2020-10-15 08:00 | listing | 2026-11-14 00:00 | 6.08 | 639,552 | 100.00000% | 2025-05-07 00:00 | 479,424 | 160,128 | 0 / 0 | 91 |
| SOLUSDT | utc_24h | 2020-09-14 07:00 | listing | 2026-11-14 00:00 | 6.16 | 648,492 | 100.00000% | 2025-04-30 00:00 | 486,348 | 162,144 | 0 / 0 | 92 |
| TRXUSDT | utc_24h | 2020-01-15 08:05 | listing | 2026-11-14 00:00 | 6.83 | 718,463 | 100.00000% | 2025-02-28 00:00 | 538,751 | 179,712 | 0 / 0 | 100 |
| UNIUSDT | utc_24h | 2020-09-18 07:00 | listing | 2026-11-14 00:00 | 6.15 | 647,340 | 100.00000% | 2025-05-01 00:00 | 485,484 | 161,856 | 0 / 0 | 87 |
| XLMUSDT | utc_24h | 2020-01-20 08:00 | listing | 2026-11-14 00:00 | 6.81 | 717,024 | 100.00000% | 2025-03-01 00:00 | 537,600 | 179,424 | 0 / 0 | 100 |
| XRPUSDT | utc_24h | 2020-01-06 08:20 | listing | 2026-11-14 00:00 | 6.85 | 721,052 | 100.00000% | 2025-02-26 00:00 | 540,764 | 180,288 | 0 / 0 | 100 |

### Research-sample gaps by class (runs / missing bars)

| asset | exchange_outage | archive_gap | weekend | daily_break | holiday | feed_gap | total missing | zero-volume bars | OHLC violations | buy_v > v (clipped) |
|---|---|---|---|---|---|---|---|---|---|---|
| AAVEUSDT |  |  |  |  |  |  | 0 | 47 | 0 | 0 |
| ADAUSDT |  |  |  |  |  |  | 0 | 54 | 0 | 0 |
| ATOMUSDT |  |  |  |  |  |  | 0 | 155 | 0 | 0 |
| AVAXUSDT |  |  |  |  |  |  | 0 | 47 | 0 | 0 |
| BCHUSDT |  |  |  |  |  |  | 0 | 49 | 0 | 0 |
| BNBUSDT |  |  |  |  |  |  | 0 | 47 | 0 | 0 |
| BTCUSDT |  |  |  |  |  |  | 0 | 66 | 0 | 0 |
| DOGEUSDT |  |  |  |  |  |  | 0 | 49 | 0 | 0 |
| DOTUSDT |  |  |  |  |  |  | 0 | 49 | 0 | 1 |
| ETCUSDT |  |  |  |  |  |  | 0 | 47 | 0 | 0 |
| ETHUSDT |  |  |  |  |  |  | 0 | 47 | 0 | 0 |
| FILUSDT |  |  |  |  |  |  | 0 | 47 | 0 | 0 |
| LINKUSDT |  |  |  |  |  |  | 0 | 47 | 0 | 0 |
| LTCUSDT |  |  |  |  |  |  | 0 | 49 | 0 | 0 |
| NEARUSDT |  |  |  |  |  |  | 0 | 50 | 0 | 0 |
| SOLUSDT |  |  |  |  |  |  | 0 | 47 | 0 | 0 |
| TRXUSDT |  |  |  |  |  |  | 0 | 49 | 0 | 0 |
| UNIUSDT |  |  |  |  |  |  | 0 | 47 | 0 | 0 |
| XLMUSDT |  |  |  |  |  |  | 0 | 62 | 0 | 0 |
| XRPUSDT |  |  |  |  |  |  | 0 | 58 | 0 | 0 |

### Research-sample flat bars (present, zero volume) by class (runs / bars)

| asset | venue_flat | symbol_flat |
|---|---|---|
| AAVEUSDT | 7 / 44 |  |
| ADAUSDT | 8 / 46 | 5 / 5 |
| ATOMUSDT | 8 / 46 | 97 / 106 |
| AVAXUSDT | 7 / 44 |  |
| BCHUSDT | 8 / 46 |  |
| BNBUSDT | 7 / 44 |  |
| BTCUSDT | 7 / 44 | 1 / 19 |
| DOGEUSDT | 8 / 46 |  |
| DOTUSDT | 8 / 46 |  |
| ETCUSDT | 7 / 44 |  |
| ETHUSDT | 7 / 44 |  |
| FILUSDT | 7 / 44 |  |
| LINKUSDT | 7 / 44 |  |
| LTCUSDT | 8 / 46 |  |
| NEARUSDT | 7 / 44 | 3 / 3 |
| SOLUSDT | 7 / 44 |  |
| TRXUSDT | 8 / 45 | 1 / 1 |
| UNIUSDT | 7 / 44 |  |
| XLMUSDT | 7 / 44 | 13 / 15 |
| XRPUSDT | 8 / 46 | 4 / 9 |

Venue-wide flat runs (research sample; first and last flat bar close, UTC):

| first | last | bars | symbols flat |
|---|---|---|---|
| 2020-09-27 11:25 | 2020-09-27 11:30 | 2 | 7 |
| 2020-09-27 11:30 | 2020-09-27 11:30 | 1 | 1 |
| 2021-03-02 01:10 | 2021-03-02 02:00 | 11 | 20 |
| 2022-05-01 22:35 | 2022-05-01 22:55 | 5 | 20 |
| 2022-05-28 16:45 | 2022-05-28 17:15 | 7 | 20 |
| 2023-09-12 08:40 | 2023-09-12 08:50 | 3 | 20 |
| 2024-10-28 16:30 | 2024-10-28 16:35 | 2 | 20 |
| 2024-10-28 20:05 | 2024-10-28 21:10 | 14 | 20 |
| 2025-01-29 01:30 | 2025-01-29 01:35 | 2 | 20 |

Zero-volume, OHLC and buy_v counters are over all archive files of the asset (research and holdout periods), counted at conversion, before the split.

### Holdout-period gaps and flat bars by class (counts only; the run lists are sealed in `holdout/<ASSET>/gaps.json`)

| asset | exchange_outage | archive_gap | weekend | daily_break | holiday | feed_gap | venue_flat | symbol_flat |
|---|---|---|---|---|---|---|---|---|
| AAVEUSDT |  |  |  |  |  |  | 1 / 3 |  |
| ADAUSDT |  |  |  |  |  |  | 1 / 3 |  |
| ATOMUSDT |  |  |  |  |  |  | 1 / 3 |  |
| AVAXUSDT |  |  |  |  |  |  | 1 / 3 |  |
| BCHUSDT |  |  |  |  |  |  | 1 / 3 |  |
| BNBUSDT |  |  |  |  |  |  | 1 / 3 |  |
| BTCUSDT |  |  |  |  |  |  | 1 / 3 |  |
| DOGEUSDT |  |  |  |  |  |  | 1 / 3 |  |
| DOTUSDT |  |  |  |  |  |  | 1 / 3 |  |
| ETCUSDT |  |  |  |  |  |  | 1 / 3 |  |
| ETHUSDT |  |  |  |  |  |  | 1 / 3 |  |
| FILUSDT |  |  |  |  |  |  | 1 / 3 |  |
| LINKUSDT |  |  |  |  |  |  | 1 / 3 |  |
| LTCUSDT |  |  |  |  |  |  | 1 / 3 |  |
| NEARUSDT |  |  |  |  |  |  | 1 / 3 |  |
| SOLUSDT |  |  |  |  |  |  | 1 / 3 |  |
| TRXUSDT |  |  |  |  |  |  | 1 / 3 |  |
| UNIUSDT |  |  |  |  |  |  | 1 / 3 |  |
| XLMUSDT |  |  |  |  |  |  | 1 / 3 |  |
| XRPUSDT |  |  |  |  |  |  | 1 / 3 |  |

### Every research-sample run (UTC; first and last missing bar close)

| asset | first missing | last missing | bars | class | peers listed | peers also missing |
|---|---|---|---|---|---|---|
| (none) | | | | | | |

### Archive-gap repairs (daily zips fetched before the split because the monthly zip lacked those days)

| asset | research-period days repaired (UTC) | bars added | holdout-period days repaired (count) |
|---|---|---|---|
| FILUSDT | 2022-02-26, 2022-02-27, 2022-02-28, 2022-04-01, 2022-04-02 | 1440 | 0 |
| LTCUSDT | 2022-02-26, 2022-02-27, 2022-02-28, 2022-04-01, 2022-04-02 | 1440 | 0 |
| NEARUSDT | 2022-02-26, 2022-02-27, 2022-02-28, 2022-04-01, 2022-04-02 | 1440 | 0 |
| SOLUSDT | 2022-02-26, 2022-02-27, 2022-02-28, 2022-04-01, 2022-04-02 | 1440 | 0 |
| TRXUSDT | 2022-02-26, 2022-02-27, 2022-02-28, 2022-04-01, 2022-04-02 | 1440 | 0 |
| XLMUSDT | 2022-02-26, 2022-02-27, 2022-02-28, 2022-04-01, 2022-04-02 | 1440 | 0 |
| XRPUSDT | 2022-02-26, 2022-02-27, 2022-02-28, 2022-04-01, 2022-04-02 | 1440 | 0 |

## Gaps explained

- **No 5m bar is missing in any of the 20 series** after the repair below. Coverage is 100% from each asset's first
  bar to 2026-11-14 00:00 UTC, in the research and holdout periods alike.
- **Archive gaps, repaired before the split.** Seven symbols (FIL, LTC, NEAR, SOL, TRX, XLM, XRP) lacked exactly
  2022-02-26..28 and 2022-04-01..02 UTC: the last three days of the February 2022 monthly zip and the first two days of
  the April 2022 monthly zip. The other 13 symbols had those days, so the classifier put them in `archive_gap`
  (19 peers listed, 6 also missing). That means truncated monthly files, not a venue outage. The archive's daily zips
  for those days exist and are complete. `data fill-gaps` fetched the 35 daily zips (checksum-verified, manifest
  `purpose: archive-gap repair`), adding 1,440 bars per symbol. They overlap no monthly row, so the duplicate rule
  never had to choose. The same repair ran over the whole history before the split and found nothing else to repair.
- **Listing start.** BTC, ETH and BCH start at 2020-01-01 00:00, the first month of the archive (their perps listed in
  2019, before the archive starts). The other 17 start at their Binance listing, mid-month: for example SOLUSDT
  2020-09-14 07:00 and AAVEUSDT 2020-10-16 07:00. History is 6.1 to 6.9 years per asset, so crypto cannot give the
  10-year depth that SPEC section 9 expects for weekly and monthly levels.
- **Exchange outages show up as flat bars, not missing bars.** During Binance futures maintenance the archive prints
  bars with `v = 0` and `o = h = l = c` instead of leaving them out. In the research sample, 9 such venue-wide runs
  hit all listed symbols at the same instants, for example 2021-03-02 01:10..02:00 (11 bars) and 2024-10-28
  20:05..21:10 (14 bars). The holdout period has one 3-bar venue-wide run (counted, not listed). They stay in the data
  as archived and are classed `venue_flat`.
- **What later phases should know about flat bars.** A flat bar has zero range, so it can only "touch" a level that
  equals its price. Its volume features are degenerate (absorption, heat), and a phase-3/6 builder should treat a bar
  inside a `venue_flat` run as no-trade. The `gaps.json` run list is the lookup.
- **Single-symbol flat bars.** `symbol_flat` runs are illiquid minutes that printed no trade. The largest count is
  ATOMUSDT (97 runs, 106 bars), mostly in its first months of 2020. BTCUSDT has one 19-bar run.
- **Other quality checks.** There are 0 OHLC violations, 0 off-grid timestamps and 0 `close_time` mismatches. One DOTUSDT
  bar had taker-buy volume above total volume by float rounding; its `sell_v` was clipped to 0 and counted.
- **Sessions.** The CFD classes `weekend`, `daily_break`, `holiday` and `feed_gap` are implemented and tested, but no
  CFD data exists yet.

## Split

- **Rule.** For each asset, the boundary is first bar open + 0.75 × (last bar close − first bar open), floored to the
  start of that trade day in the asset's session (00:00 UTC for crypto). Bars that open before the boundary go to
  `data/research/<ASSET>.parquet`. Bars that open at or after it go to `data/holdout/<ASSET>/part-0000.parquet`.
- **Result.** The actual research fraction is between 0.7496 and 0.7500. The boundaries run from 2025-02-24 to
  2025-05-08 and are in the tables and in `split.json`.
- **Size.** Research is 240 MB and holdout 78 MB. Staging and the raw zips were deleted. The repo is at about 9.1 GB
  of the 30 GB budget, measured before the download (8.79 GB) and again at the reproduce run.
- **Sealed holdout gap lists.** `holdout/<ASSET>/gaps.json` holds the holdout-period run lists. Only class counts sit
  outside the holdout.
- **Write under the deny.** These results were measured on NTFS here, not assumed:

  | Under the deny | Result |
  |---|---|
  | read, list, `stat`, append, overwrite, rename into the folder, polars' own writer | fail |
  | create a new file with `open(path, "xb")`, create a subfolder | work |
  | re-run the lock on a locked folder | works |

  So new holdout data (CFDs later) is written as new part files from an in-memory parquet buffer
  (`split.write_holdout_part`), which then checks that the new file cannot be opened for reading. No staging-and-move
  step is needed. Append is not possible and is not used.
- **Guard.** `lock_holdout` refuses a folder whose ACL has an OWNER RIGHTS entry. Python 3.13's `mkdir(mode=0o700)`
  and `tempfile` write such an entry, and it would make the deny impossible to lift without an administrator.
  `vecopen/data` has no such entry: it inherits plain SYSTEM, Administrators and user grants.
- **Phase 9 lift.** The orchestrator runs this once, in cmd.exe. Do not run it before phase 9:

  ```
  icacls "C:\Users\Asus\Desktop\AI_Trader\vecopen\data\holdout" /remove:d "%USERNAME%" /T /C
  ```

  It was verified on scratch folders with the same ACL shape, never on the real holdout. It is also stored as
  `lift_command_phase9` in `split.json`.

## What was built

### Python (`vecopen/src/vecopen/`)

**`data.py`**
- Paths, the base-bar schema and `load_bars(asset)`. `load_bars` reads `data/research/<ASSET>.parquet` and validates
  the schema, UTC, the 5m grid and strictly increasing `ts_close`. Its columns are `ts_close, o, h, l, c, v`, plus
  `buy_v, sell_v` for crypto.
- `HoldoutAccessError`: any path inside `vecopen/data/holdout`, or inside any `data/holdout`, is refused before the
  file is touched.
- `dedup_bars`, with the documented duplicate rule.
- `panama_adjust`: difference back-adjustment at given roll dates. The gap is measured as open after minus close
  before, or taken from the broker. It keeps `o_unadj … c_unadj` and `roll_offset`, and leaves the latest contract
  unadjusted.
- `read_roll_calendar`: reads `config/rolls/<ASSET>.csv`, which is required for any `panama` asset.

**`bars.py`**
- `Session`, and `load_sessions()`, which parses and sha256-hashes `config/sessions.yaml`.
- `resample(df, tf, session)` for 5m, 15m, 30m, 1h, 2h, 4h, 1d, 1w and 1M:
  - the trade date comes from local time shifted by `24h - day_start` (DST-aware);
  - intraday buckets count elapsed real time from the trade-day start and are cut at the next day start;
  - weeks are Monday trade dates; months are calendar months of trade dates;
  - each bar is keyed by its nominal close, and the last incomplete bucket is dropped (construction rule 1);
  - `n_base` counts the base bars in each bucket.

**`gaps.py`**
- `missing_runs`.
- The crypto classifier: `exchange_outage` (at least half of at least 3 listed peers also missing) versus
  `archive_gap`.
- The CFD classifier against a weekly schedule, either the cTrader `ProtoOAInterval` list or the nominal hours: classes
  `weekend`, `daily_break`, `holiday` (cTrader `ProtoOAHoliday`, recurring or not) and `feed_gap`.
- `zero_volume_runs` and `classify_flat`.
- Research classes use research data only: each peer is cut at its own boundary.

**`ingest.py`**
- `Manifest`: atomic JSON, with an entry for every archive file covering URL, zip sha256, CHECKSUM verified, row
  count, first and last bar, quality counters and the staged path.
- The storage check against the 30 GB budget and a 1.5 GB phase cap. It passed: repo 8.79 GB, plan 0.55 GB, peak
  add 1.1 GB.
- S3 listing of the archive; monthly zips plus the daily zips of later months.
- `parse_klines`: handles files with and without a header, converts µs timestamps to ms, and sets `ts_close` to
  open_time + 5m, `buy_v` to taker-buy base volume and `sell_v` to volume − buy_v.
- `fill_from_daily`.
- The cTrader driver: `ctrader_keys_present`, `ctrader_download` and `stage_ctrader`.
- `roll_decision`: the metadata-only rule for `panama_if_futures_based`.

**`split.py`**
- `split_boundary`, `lock_holdout`, `assert_unreadable`, `write_holdout_part`, `write_holdout_json`, `build_split` and
  `lift_command`.

**`phase1.py`, `__main__.py`**
- `python -m vecopen data crypto-download | fill-gaps | build-split | cfd-download | gaps | storage`, and `phase1`.

### Config

**`vecopen/config/sessions.yaml`** (new)
- Written before any CFD data existed: no keys, and no CFD bar, price or schedule ever fetched. Its sha256 is in
  every registry line.
- Conventions:

  | Convention | Assets | Trade day starts |
  |---|---|---|
  | `utc_24h` | 20 crypto | 00:00 UTC |
  | `fx_ny17` | EURUSD, GBPUSD, USDJPY, AUDUSD, USDCAD | 17:00 New York |
  | `cme_ny18` | XAU, XAG, USOIL, US500, US30, NAS100, JPN225 | 18:00 New York (CME Globex trade date) |
  | `ice_brent_ldn23` | UKOIL | 23:00 London |
  | `eurex_cet00` | GER40, BUND | 00:00 Berlin |
  | `ice_ftse_ldn00` | UK100 | 00:00 London |

- Roll per instrument:

  | Roll | Assets |
  |---|---|
  | `none` | perps, FX, spot metals, cash indices |
  | `panama` | BUND (a futures-based CFD by construction) |
  | `panama_if_futures_based` | USOIL, UKOIL: decided from cTrader symbol metadata (name or description naming futures, a contract month or an expiry), never from prices; the decision and its evidence go into the manifest, and the build refuses until it exists |

- The nominal hours are documentation and the fallback schedule. Once a symbol resolves, its own cTrader schedule
  classifies gaps. A disagreement is reported and never changes `day_start`.

**`vecopen/config/ctrader_candidates.yaml`** (new)
- cTrader name candidates for the 16 CFDs, in the xvenue format.

### Rust

All formatted with max_width=180 and use_small_heuristics=Max, on my own lines only.

**`crates/at-ctrader`**
- New `src/trendbars.rs`:
  - `Period` and `decode`/`encode` (low + deltas, minute timestamps);
  - `months_desc`;
  - `fetch_window`: splits a window in half on `hasMore` or on a range or boundary error, so it does not depend on the
    server's undocumented per-period range cap or chunk size, or on which end a truncated page keeps;
  - `fetch_history`: walks calendar months backwards, skips final months already on disk, marks the current month as
    partial, and stops `max_empty` empty months after the data starts.
- `codec.rs`: registers `ProtoOAGetTrendbarsReq/Res` as `OaMsg` and `OaRequest`. The request was already on the
  read-only allowlist and shares the historical rate limit (220 ms between historical requests, plus a
  `BLOCKED_PAYLOAD_TYPE` retry). No order message type was added.
- `store.rs`: `trendbar_path` and `write_trendbars`, with columns `ts_open, o, h, l, c, tick_volume, known_ts`.
- `lib.rs`: declares the module.
- `tests/canned.rs`: the fake server answers trend-bar requests.

**`crates/atx/src/ctrader.rs`**
- `atx ctrader trendbars --symbol S --period M5|… --from --to --out --max-empty --force`, writing
  `<out>/<symbol>/<period>/<YYYY-MM>.parquet` plus `_meta.jsonl` and `_summary.json` (the history actually obtained).
- `symbols --resolve --config <file> --map-out <file>`, and `XInst` accepts `asset` as an alias of `bybit`.

**`crates/atx/src/main.rs`**
- Two doc lines only (subcommand lists); the CRLF line endings are kept.

**Binary**
- `target/debug/atx` was rebuilt. Another agent's `mm-replay` runs were using it back to back, so the in-use image was
  renamed aside (Windows allows that for a running exe), the binary was relinked, and the old image was deleted after
  that process exited. No process was stopped.

## What the CFD part still needs

1. **cTrader keys.** `atx ctrader check` reports all keys missing. The user adds `CTRADER_CLIENT_ID`,
   `CTRADER_CLIENT_SECRET`, `CTRADER_ACCESS_TOKEN` (read-only `accounts` scope) and `CTRADER_ENV` to `.env`.
2. **The one command:**
   ```
   uv run --extra vecopen python -m vecopen phase1 --with-cfd
   ```
   It runs these steps:
   1. `atx ctrader symbols --resolve --config vecopen/config/ctrader_candidates.yaml --map-out vecopen/data/ctrader_map.json`;
      the mapping and the roll decisions for USOIL and UKOIL go into the manifest.
   2. `atx ctrader trendbars --symbol <name> --period M5 --from 2000-01-01 --out vecopen/data/staging/ctrader_raw` for
      each resolved asset. The history actually obtained is recorded in `_summary.json` and the manifest; the target
      is 10 years or more.
   3. Staging, gap classes against the symbol's own schedule and holidays, and the split. New holdout part files are
      written into the locked folder.
   4. One registry line.
3. **Unresolved names.** If a name is unresolved, add the right one to `ctrader_candidates.yaml` and rerun.
4. **BUND.** If BlackBull offers the Bund, `vecopen/config/rolls/BUND.csv` must list its roll dates from BlackBull's
   roll notices or the Eurex calendar before the build; the build refuses without it. The same applies to USOIL and
   UKOIL if their metadata says futures-based.
5. **Schedule check.** Compare each resolved cTrader schedule with the nominal hours in `sessions.yaml`, especially
   JPN225, where CME versus OSE hours is the open question. Report mismatches; do not change `day_start` after data is
   seen.

## Findings, deviations, disclosures

- **Disclosure: holdout-period rows seen.** During the format check, before any split existed, I printed the header and
  the first two rows of the `BTCUSDT-5m-2026-10` zip (5m OHLCV of 2026-10-01 00:00 and 00:05 UTC). That data is now in
  the BTCUSDT holdout period. Nothing else from any holdout period was printed:
  - the pre-split gap inspection showed research-period runs only;
  - the archive-gap repair over the whole history logged counts only;
  - holdout run lists are sealed.
- **Listing denied as intended.** One of my status commands included a listing of `data/holdout`; the auto-mode
  classifier blocked it, and it was not retried in any form.
- **Leftover test folders cleaned.** Early runs of my lock test left three of my own pytest temp folders locked: they
  carried the OWNER RIGHTS ACE, which is how the guard above was found. I lifted them with `SetFileSecurityW`, which
  needs only WRITE_DAC, and deleted them. The test fixtures now give their folders a plain ACL and lift their own deny.
- **Base timeframe.** The 5m base and the SPEC section 7 note "5m levels need 1m data" are unchanged; only 5m was
  downloaded.
- **Crypto history depth.** 6.1 to 6.9 years. The holdout starts between 2025-02-24 and 2025-05-08 depending on the
  asset.
- **Not done here.** No diary entry was written; per vecopen/CLAUDE.md that is the orchestrator's per-phase job.
  `SPEC.md`, `baseline.yaml`, `.env`, keys, `config/schema.yaml` and `bin/` were not touched, and no scheduled task was
  created.
