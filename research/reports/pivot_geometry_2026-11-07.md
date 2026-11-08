# Pivot geometry (DC pivots): H-GT1, H-GT2, H-GT3 (2026-11-07)

Port of the TradingView indicator "Pivot geometry trend (DC pivots)" (`features/dc_pivots.py`), registered as
`geo_struct`, `geo_pivot_bo`, `geo_flag_bo` (`strategies/structure.py`, v0.1.0). Study: `research/geo_study.py`.
Data: bybit 4h, 23-symbol universe, `market_frame(funding=False, oi=False)`. Signals are computed on the full
history and then split at the knowledge cutoff (2026-07-01). The in-sample numbers and verdicts were recorded
before the holdout was run.

## Port decisions
- bar_index is the frame row index. sigma_t is the EWMA std of close changes (price points, half-life 60), and
  delta_t = K·sigma_{t-1}. `ready` = bar_index > 60 and sigma > 0. Signals are 0 when not ready.
- When delta is invalid (NaN or ≤ 0, which only happens in the first bars) or the close is non-finite, the
  engine and all three vote states hold their values for that bar. The Pine source does not say what to do here.
- A reversal confirms at exactly K·sigma (`>=`). In a run, the reversal is tested before the extreme is updated.
  In warm-up, hiR/loR are updated first and then tested.
- FLAG_BO retracement filter is inclusive (≤ 0.618). A zero or negative denominator blocks entry. There is no
  re-entry (either side) on an exit bar.

## H-GT1 / H-GT2: mean over symbols of annualised daily net Sharpe

| model | in-sample | holdout | exposure | corr w/ ema_stack |
|---|---:|---:|---:|---:|
| geo_struct | −0.04 | −0.90 | 0.86 | 0.46 |
| geo_pivot_bo | 0.31 | −0.60 | 0.99 | 0.75 |
| geo_flag_bo | 0.06 | −1.16 | 0.80 | 0.68 |
| geo_avg | 0.16 | −1.03 | 0.96 | 0.72 |
| ema_stack | 0.24 | −0.43 | 0.69 | 1 |
| donchian_breakout | 0.28 | 0.02 | 0.63 | |
| tsmom | 0.36 | −0.56 | 0.98 | |

All 23 symbols have more than 30 days in both segments (the holdout has about 129 days).
- **H-GT1: contradicts in both segments.** geo_struct is below 0 in-sample, and all three models are negative in the holdout.
- **H-GT2: contradicts in both segments.** geo_avg is below ema_stack in both segments (0.16 vs 0.24; −1.03 vs −0.43).

## H-GT3: DC swing levels as support/resistance (K = 6)

| segment | resolved | reject rate | binom z | highs | lows | random-walk null | z vs null |
|---|---:|---:|---:|---:|---:|---:|---:|
| in-sample | 2088 | 0.574 | 6.78 | 0.533 | 0.612 | 0.568 | 0.85 |
| holdout | 158 | 0.570 | 1.75 | 0.522 | 0.632 | 0.549 | 0.78 |

The registered verdict is **supports** in both segments: in-sample the rate is > 0.55 with z > 2, and in the holdout it is > 0.5 with n ≥ 8. **The
verdict is not evidence of support/resistance, though.** The race starts at the touch bar's close. That close is
usually back on the near side of the level, because only the wick touched it (median offset 0.18 sigma toward
rejection). A driftless walk from that start rejects about 0.57 of the time. Compared with that null the excess is
z ≈ 0.8 in both segments. The difference between highs and lows matches the in-sample upward drift.

Resolved events per symbol (in-sample / holdout): about 31 (HYPE) to 124 (BTC) per symbol in-sample, and 4 to 12 per symbol in the holdout (see `geo_study` output).

## FLAG_BO stop distance at entry, in units of sigma_next·close
- All K: n = 3638, median 6.0, p10 3.6, p90 12.3, 84% wider than 4 sigma. In units of K·sigma_ind the median is 1.5.
- K = 6: median 8.2, p10 5.3, p90 14.3, 98% wider than 4 sigma.

## Read
- In-sample, the geo models are a noisier copy of slow trend: corr 0.7 with ema_stack, and lower Sharpe than
  ema_stack, donchian or tsmom. In the holdout they lose more than the baselines. This repeats the prior project's
  negative result.
- The structural stops sit far outside a 4-sigma cap. Capping them would turn FLAG_BO into a different strategy.
- Tested in the race design that controls for where it starts, the confirmed DC levels carry no detectable
  support/resistance information. A future level test should start the race at a symmetric point, for example
  the first close on either side of the level.
