# Side study (2026-11-13): volatility break of structure as a trend-following filter, with up/down semivolatility and market structure

**User idea:** a simpler model. A volatility break of structure (vol BoS) is a good filter for trend following with market-structure awareness. Decompose vol into upside and downside parts, track which one crosses its threshold, find the threshold, and train the hyperparameters smartly. Trend following has worked historically, but maybe not the way we have tested it.

**Registered before testing:** H-VBOS1-4 (`research/hypotheses.yaml`).

## What we already know (do not rediscover)

- Simple trend models are thin edges, positive before the cutoff and negative after (P3 / P7).
- Selecting among them per symbol is a coin flip (R-SYN-2).
- Trend profits come from rare large moves, and tight trailing exits destroy them (EX1, H-PD6).
- Squeeze breakouts (H-VB3) and daily vol-surprise continuation (H-VB2) failed.
- 1h trend loses in range and whipsaw regimes (R-MAP-4).
- There is no leverage-type vol asymmetry for forecasting (H-LEV2). Semivariance for direction is untested.

## Building blocks (fix all families and grids before looking at any result; write them in a diary entry)

1. **Vol BoS detectors.** A few families, each with a small grid:
   - (a) Range break: short-window log RV (5-20 d, or 24-120 h) above the max of its trailing range (60-180 d).
   - (b) Changepoint: CUSUM, or a Bayesian online changepoint on log RV.
   - (c) Persistent HAR surprise: RV above k x the log-HAR forecast for m consecutive days.
   - (d) Vol-of-vol break.
2. **Semivolatility.** Use realised semivariances RS+ and RS- from `features/realized.py`, daily and 4h. Each side breaks against its own trailing quantile threshold. Dominance: the RS+ / RS- ratio, or which side broke first.
3. **Market structure.** Fractal swing highs and lows (k bars, as in `research/levels`/`dc_pivots`). A structure break is a close beyond the last swing high or low. CHoCH (change of character) is the first break against the prior structure.
4. **Strategy.** Enter in the direction of the confirmed structure break, when a vol BoS is active and the dominant semivariance agrees. Exits are the trend strategies' own exits or the opposite structure break (R-EXIT-1). There is no tight trailing stop. P&L is in risk units, via `evaluate.strategy_pnl` (7.5 bps per side). Sizing belongs to the risk engine.

## How to train the hyperparameters without fooling ourselves

1. **Nested purged walk-forward inside `llm_in_sample` only.**
   - The inner loop chooses hyperparameters on the past.
   - The outer loop evaluates the next block.
   - Purge and embargo the forward-return horizon.
2. **Small grids and a pre-stated objective:**
   - Objective: net Sharpe, or the event-study t for VBOS1/2.
   - Thresholds as quantiles, for example 0.80 / 0.90 / 0.95, not raw levels, so they transfer across assets.
   - Pool across the 23 crypto perps; only BTC also gets a per-symbol look.
3. **Report the selection penalty:** the deflated Sharpe over the full grid, plus PBO by CSCV. Use `pbo_cscv` in `research/synthetic_robustness_study.py`.
4. **Then a single look at `clean_holdout`** with the selected configuration. TradFi perps serve as holdout robustness.
5. **Threshold stability (VBOS4):** compare the objective surfaces across segments. The holdout surface may be described but never used to choose.

## Outputs

- **Code:** `research/vol_bos_study.py` plus `features/vol_bos.py`.
- **Tests:** `tests/test_vol_bos.py`, covering causality and detector behaviour on synthetic data.
- **Report:** `research/reports/vol_bos_2026-11-13.{json,md}` with figures (matplotlib through the `viz` extra), and rule candidates R-VBOS-n.
- **Resources:** at most 2 threads and about 1 GB of RAM. The work runs on daily and 4h data, so it is light.
