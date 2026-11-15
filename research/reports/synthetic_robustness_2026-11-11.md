# Synthetic robustness of trend-parameter selection (H-SYN1, H-SYN2)

_Program 2, workstream B5 ([plan](../plans/2026-11-11_strategy_program.md)). Code: `src/ai_trader/sim/resample.py`,
`src/ai_trader/research/synthetic_robustness_study.py`; tests `tests/test_resample.py`, `tests/test_synthetic_robustness.py`.
Machine-readable results: [synthetic_robustness_2026-11-11.json](synthetic_robustness_2026-11-11.json). Generated data is used here only to choose
parameters of deterministic strategies. It never trains or backtests the LLM layer (CLAUDE.md principle 1)._

## Bottom line

1. **H-SYN1 is recorded as `supports` under the literal criterion, but the effect is mechanical.** Picking the variant with the best
   median Sharpe over generated paths loses 0.59 less Sharpe from in-sample to holdout than the in-sample champion (CI [-1.05, -0.15]).
   A **uniformly random variant loses 0.59 less too** (CI [-0.90, -0.31]). Any choice other than the champion starts from a lower in-sample
   Sharpe, so it has less to lose. The result also **does not hold across seeds**: seeds 1 and 2 give -0.41 [-0.84, +0.01] and
   -0.34 [-0.78, +0.07].
2. **H-SYN2 is `inconclusive`.** The zoo variant with the best 10th-percentile Sharpe beats the in-sample champion in the holdout by
   +0.18 Sharpe (CI [-0.53, +0.91]). The sign flips with the seed (-0.37 and -0.06 for seeds 1 and 2).
3. **Selection inside these grids is a coin flip.** PBO (CSCV, S = 16) has a median of 0.44 for the zoo and 0.65 for the breakout grid.
   In the breakout grid, the in-sample champion does *worse* in the holdout than a random pick (+0.38 Sharpe for the random pick,
   CI [+0.04, +0.69]). This agrees with P7 and H-TR2: per-symbol parameter choice adds nothing, and fixed choices or ensembles are the
   honest default.
4. **Generator quality.** The stationary block bootstrap (Politis-White block on |r|, median about 580 h) reproduces the fat tails and the
   |r| autocorrelation out to one day, and regime durations. It partly loses clustering at one week (0.72x). The regime-switching
   resampler **as designed destroys clustering beyond about half a day** (|r| ACF at 24 h 0.24x, at 168 h 0x). That was a design mistake
   (diary, `mistake`).

## Methods (fixed before running; diary 2026-11-10 "B5 start")

**Data.** The universe is `trend_zoo_study.UNIVERSE` (23 Bybit perps), `market_frame(funding=False, oi=False)`. The `knowledge_cutoff`
is 2026-07-01. The holdout runs 2026-07-01 to 2026-11-10 (133 days).

**Historical P&L.** Native 4h frames, with signals computed on the full history. The rules follow `evaluate.strategy_pnl`: leverage
s/sigma fixed at entry, a 0.2 rebalance band, taker + slippage of 7.5 bps per side, risk units. As in `trend_zoo_study`, evaluation starts
on the first full UTC date after `trend_zoo.lookback_bars(6)`. Daily P&L is dated by the bar that *realises* it (bar t's P&L uses
close_t -> close_{t+1}), so no holdout return enters an in-sample Sharpe. Sharpe = mean / sd x sqrt(365) of daily P&L. A vectorised
engine reproduces `trend_zoo` + `strategy_pnl` to within 5e-13 (signals) and 6e-12 (P&L) on every symbol's real frame. The unit test
`test_vectorised_zoo_matches_production` checks the same thing.

**Generator (`sim/resample.py`)**, fed with llm_in_sample 1h bars only:
- *(c) Bars* are resampled as whole relative-OHLC tuples: r = log(c_t/c_{t-1}), gap g = log(o_t/c_{t-1}), upper wick
  u = log(h/max(o,c)) >= 0, lower wick d = log(min(o,c)/l) >= 0. A path is rebuilt as c_t = c_{t-1}e^r, o = c_{t-1}e^g,
  h = max(o,c)e^u, l = min(o,c)e^{-d}, so high >= max(open, close) and low <= min(open, close) hold by construction. Wicks keep their
  joint law with the bar's own return. Paths are then aggregated 1h -> 4h. Each path has the in-sample length (whole days).
- *(a) Stationary block bootstrap* (Politis-Romano): geometric blocks with circular wrap. The mean block is the Politis-White/PPW
  length on |r_1h| (arch `optimal_block_length`, which caps at min(3 sqrt n, n/3); BTC and LTC hit the cap). Median 580 h, range
  268 h (HYPE) to 701 h (BTC).
- *(b) Regime-switching*: v0 cells (`regimes.label`, 1h) on the in-sample bars. The hourly 9x9 transition matrix is fitted from
  consecutive labelled bars, a cell path is simulated, and each simulated cell run is filled with stationary blocks from the pool of
  real bars in that cell. A source bar carries its contemporaneous cell. A new block starts at every cell change, and otherwise with
  probability 1/b.
- Per symbol and seed: 200 paths = 100 SBB + 100 RS. Seed 0 gives the verdicts; seeds 1 and 2 measure stability. Runtime was 44-320 s
  per symbol and seed with 4 workers, so **200 paths were kept** (no reduction to 100).

**Grids.** *zoo* = the 21 `trend_zoo` variants (4h build). *bo* = Donchian breakout with entry N in {5, 10, 20, 40} days and exit
M in {0.25 N, 0.5 N}, scale 3.0 (8 configurations, fixed in the docstring and diary before running).

**Selection (per symbol and grid, in-sample information only).** *hist* = argmax of historical in-sample Sharpe. *med* = argmax of
median generated Sharpe. *q10* = argmax of the 10th-percentile generated Sharpe.

**Tests (registry, literal).**
- H-SYN1: deg(v) = Sharpe_IS - Sharpe_HO. Per symbol, d = deg(med) - deg(hist), averaged over the two grids. Take the mean over
  symbols, with a 95% CI from 10,000 symbol-bootstrap draws. Success: mean < 0 and the CI excludes 0.
- H-SYN2: zoo grid, e = Sharpe_HO(q10) - Sharpe_HO(hist). Success: mean > 0 and the CI excludes 0.
- Verdict map: the success criterion met -> supports; point estimate on the wrong side -> contradicts; right sign but the CI covers 0
  -> inconclusive. The verdict is recorded on `clean_holdout`. The `llm_in_sample` rows are `inconclusive` by construction and carry
  the in-sample diagnostics: the hypothesis concerns the holdout outcome of an in-sample selection.

## Results (seed 0, 23 symbols)

| statistic (Sharpe units) | mean | 95% CI (symbols) | share < 0 | share identical choice |
|---|---|---|---|---|
| **H-SYN1** d, both grids | **-0.59** | [-1.05, -0.15] | 0.57 | 0.22 |
| d, zoo only | -0.48 | [-1.11, +0.07] | 0.43 | 0.30 |
| d, breakout only | -0.69 | [-1.15, -0.27] | 0.48 | 0.48 |
| d using 100 SBB paths only | -0.21 | [-0.65, +0.15] | | |
| d using 100 RS paths only | -0.57 | [-0.98, -0.20] | | |
| **null: random variant** d, both grids | **-0.59** | [-0.90, -0.31] | 0.83 | |
| **H-SYN2** e, zoo | **+0.18** | [-0.53, +0.91] | 0.39 | 0.17 |
| e, breakout (secondary) | +0.51 | [+0.02, +1.01] | 0.26 | 0.13 |
| holdout gain of the *med* pick, zoo | +0.31 | [-0.25, +0.93] | | 0.30 |
| holdout gain of the *med* pick, breakout | +0.58 | [+0.19, +1.00] | | 0.48 |
| null: random variant holdout gain, zoo | +0.02 | [-0.43, +0.51] | | |
| null: random variant holdout gain, breakout | +0.38 | [+0.04, +0.69] | | |

Day-block bootstrap of the holdout (days resampled jointly across symbols, mean block 5 days, so the common crypto factor is kept):
H-SYN1 CI [-1.17, -0.14], H-SYN2 CI [-0.99, +1.31].

Mean Sharpe of each choice:

| grid | choice | in-sample | holdout |
|---|---|---|---|
| zoo | hist (champion) | +0.76 | -0.37 |
| zoo | med | +0.59 | -0.06 |
| zoo | q10 | +0.41 | -0.19 |
| breakout | hist | +0.53 | -0.67 |
| breakout | med | +0.42 | -0.10 |
| breakout | q10 | +0.28 | -0.17 |

**Seed sensitivity of the headline statistics** (same historical Sharpes, new generated paths):

| seed | H-SYN1 d | H-SYN2 e | literal verdicts |
|---|---|---|---|
| 0 (verdict) | -0.59 [-1.05, -0.15] | +0.18 [-0.53, +0.91] | supports / inconclusive |
| 1 | -0.41 [-0.84, +0.01] | -0.37 [-0.85, +0.11] | inconclusive / contradicts |
| 2 | -0.34 [-0.78, +0.07] | -0.06 [-0.57, +0.46] | inconclusive / contradicts |

**Reading.** The headline H-SYN1 number equals the random-pick null, the zoo part alone does not clear 0, and two of three seeds miss.
Generated-path selection therefore regularises only in the trivial sense of "not picking the champion". The one consistent pattern is
in the breakout grid: the champion is a fast (5-10 day) channel in 13 of 23 symbols, and it does worse out of sample than slower
channels. The *med* pick moves towards 20-40 day channels (which also have lower turnover), and the random pick captures most of that
gain too.

### Per symbol (seed 0; Sharpe in-sample / holdout)

| symbol | zoo hist | zoo med | zoo q10 | d zoo | e zoo | bo hist | bo med | d bo |
|---|---|---|---|---|---|---|---|---|
| BTCUSDT | cfg_tsmom (+1.10/-1.78) | cfg_donchian_breakout (+1.04/-0.70) | mac_7_30 (+0.63/-1.83) | -1.14 | -0.05 | bo_5d_x0.5 (+0.99/-1.98) | bo_40d_x0.5 (+0.99/-0.98) | -1.00 |
| ETHUSDT | tsmom_30d (+0.96/+0.76) | donch_14 (+0.73/+0.62) | donch_14 (+0.73/+0.62) | -0.09 | -0.14 | bo_10d_x0.25 (+0.87/-2.80) | same | 0 |
| SOLUSDT | tsmom_7d (+1.04/+0.70) | cfg_ema_stack (+0.59/+2.54) | cfg_ema_stack | -2.29 | +1.84 | bo_5d_x0.25 (+0.82/+0.77) | bo_20d_x0.25 (+0.59/+1.99) | -1.45 |
| XRPUSDT | slope_7d (+0.68/-0.81) | same | same | 0 | 0 | bo_10d_x0.25 (+0.45/-0.51) | same | 0 |
| DOGEUSDT | tsmom_14d (+1.03/+0.46) | mac_1_7 (+0.83/-1.77) | cfg_ema_stack (+0.40/-1.93) | +2.03 | -2.40 | bo_20d_x0.5 (+0.77/+0.71) | bo_5d_x0.5 (+0.59/-0.86) | +1.39 |
| ADAUSDT | mac_1_7 (+1.01/+0.68) | same | mac_7_30 (+0.60/-1.96) | 0 | -2.64 | bo_10d_x0.5 (+0.68/-2.23) | same | 0 |
| LINKUSDT | slope_30d (+0.39/+0.42) | same | same | 0 | 0 | bo_40d_x0.25 (+0.14/-0.19) | bo_20d_x0.5 (+0.11/+0.42) | -0.64 |
| LTCUSDT | mac_14_60 (+0.26/+0.59) | same | same | 0 | 0 | bo_5d_x0.5 (+0.11/-0.73) | same | 0 |
| BCHUSDT | tsmom_30d (+0.41/-3.29) | slope_7d (+0.30/+1.36) | slope_7d | -4.76 | +4.65 | bo_20d_x0.5 (+0.47/-1.75) | bo_5d_x0.5 (+0.17/+0.85) | -2.91 |
| AVAXUSDT | tsmom_14d (+1.15/-0.17) | ema_4_16_64 (+0.71/-1.53) | cfg_ema_stack (+0.64/+0.11) | +0.93 | +0.27 | bo_20d_x0.5 (+0.96/+1.65) | same | 0 |
| DOTUSDT | tsmom_14d (+0.80/-1.31) | ema_4_16_64 (+0.58/-0.64) | slope_30d (+0.46/+1.37) | -0.89 | +2.68 | bo_5d_x0.5 (+0.49/-0.31) | bo_40d_x0.5 (+0.11/+0.44) | -1.13 |
| BNBUSDT | tsmom_30d (+0.56/+1.50) | mac_7_30 (+0.41/+1.99) | ema_4_16_64 (+0.31/-1.35) | -0.64 | -2.85 | bo_10d_x0.25 (+0.25/-1.95) | bo_20d_x0.5 (+0.02/+1.27) | -3.45 |
| TRXUSDT | mac_7_30 (+1.07/-2.56) | same | same | 0 | 0 | bo_20d_x0.25 (+0.66/-4.26) | same | 0 |
| NEARUSDT | tsmom_7d (+0.97/+0.69) | mac_1_7 (+0.81/+0.27) | ema_4_16_64 (+0.49/+1.35) | +0.26 | +0.66 | bo_5d_x0.5 (+0.93/+0.79) | bo_10d_x0.5 (+0.53/+1.65) | -1.25 |
| ATOMUSDT | slope_30d (+0.56/+1.21) | ema_4_16_64 (+0.43/+0.71) | ema_4_16_64 | +0.36 | -0.50 | bo_40d_x0.5 (+0.36/+0.18) | same | 0 |
| ETCUSDT | donch_7 (+0.37/+0.45) | mac_14_60 (+0.30/+0.04) | cfg_ema_stack (+0.14/-0.92) | +0.33 | -1.37 | bo_20d_x0.25 (+0.36/-1.11) | same | 0 |
| FILUSDT | mac_1_7 (+0.66/-2.83) | ema_4_16_64 (+0.34/+0.26) | ema_4_16_64 | -3.42 | +3.09 | bo_5d_x0.5 (+0.59/-2.02) | same | 0 |
| UNIUSDT | slope_30d (+0.69/+1.08) | same | ema_4_16_64 (-0.07/+1.67) | 0 | +0.60 | bo_20d_x0.5 (+0.21/+1.58) | bo_5d_x0.25 (+0.01/+2.62) | -1.23 |
| AAVEUSDT | mac_1_7 (+0.19/-0.27) | same | slope_30d (-0.08/-0.49) | 0 | -0.22 | bo_20d_x0.5 (-0.20/+0.61) | same | 0 |
| XLMUSDT | mac_1_7 (+0.75/-1.12) | slope_7d (+0.65/+0.09) | mac_14_60 (-0.37/+0.28) | -1.31 | +1.39 | bo_5d_x0.25 (+0.69/-1.08) | bo_20d_x0.5 (+0.45/+0.79) | -2.11 |
| SUIUSDT | ema_4_16_64 (+1.20/-1.10) | slope_7d (+0.67/-0.40) | slope_7d | -1.23 | +0.70 | bo_5d_x0.5 (+0.63/-0.99) | same | 0 |
| ENAUSDT | tsmom_30d (+1.16/+0.25) | ema_4_16_64 (+0.40/-1.48) | ema_4_16_64 | +0.97 | -1.73 | bo_5d_x0.25 (+1.04/+1.05) | bo_40d_x0.5 (+0.94/+1.29) | -0.34 |
| HYPEUSDT | donch_60 (+0.48/-1.97) | mac_14_60 (+0.46/-1.79) | mac_14_60 | -0.20 | +0.18 | bo_40d_x0.25 (-0.10/-0.95) | bo_20d_x0.5 (-0.43/+0.52) | -1.80 |

One symbol's 133-day holdout Sharpe has a standard error of about 1.7, which is why single rows swing by +-4.

## Generator diagnostics (llm_in_sample, real vs generated)

The table gives the median across symbols of: the real value; the generated median (over paths); the ratio generated / real; and the
share of symbols whose real value lies inside the 5-95% generated band. Regime statistics are re-labelled with `regimes.label` on 10
generated paths per generator.

| moment | real | SBB gen | SBB ratio | SBB in band | RS gen | RS ratio | RS in band |
|---|---|---|---|---|---|---|---|
| excess kurtosis 1h | 29.7 | 28.4 | 0.96 | 1.00 | 28.3 | 0.97 | 0.91 |
| excess kurtosis 4h | 13.6 | 14.8 | 1.02 | 1.00 | 15.3 | 0.94 | 0.78 |
| ACF abs r, 1h lag 1 | 0.280 | 0.275 | 0.99 | 1.00 | 0.244 | 0.88 | 0.30 |
| ACF abs r, 1h lag 24 | 0.168 | 0.160 | 0.94 | 1.00 | 0.040 | 0.24 | 0.00 |
| ACF abs r, 1h lag 168 | 0.115 | 0.074 | 0.72 | 0.17 | 0.000 | 0.00 | 0.00 |
| ACF abs r, 4h lag 1 | 0.218 | 0.217 | 0.99 | 1.00 | 0.143 | 0.67 | 0.09 |
| ACF abs r, 4h lag 6 | 0.176 | 0.160 | 0.91 | 0.96 | 0.040 | 0.25 | 0.00 |
| ACF r, 1h lag 1 | -0.011 | -0.011 | 0.99 | 1.00 | -0.011 | 0.90 | 0.96 |
| sd 1h | 0.011 | 0.010 | 0.99 | 1.00 | 0.010 | 0.97 | 0.83 |
| v0 cell episode, median h | 9.0 | 9.8 | 1.00 | 1.00 | 9.0 | 1.00 | 0.83 |
| v0 cell episode, mean h | 13.8 | 14.2 | 1.02 | 0.61 | 12.8 | 0.93 | 0.04 |
| direction episode, mean h | 24.7 | 24.7 | 1.01 | 0.96 | 26.1 | 1.06 | 0.30 |
| hourly stay probability | 0.927 | 0.930 | 1.00 | 0.61 | 0.922 | 0.99 | 0.04 |

The total-variation distance between generated and real cell frequencies is 0.05 for SBB and 0.07 for RS.

- **SBB** is a good generator up to about one day. Fat tails, 4h aggregation tails, short-lag clustering and regime durations are all
  reproduced. Long memory decays beyond the ~3.5-week mean block (lag 168 h at 0.72x).
- **RS** keeps tails and the regime chain, but **loses vol clustering**. v0 cells change every ~13 h, and each new run starts a block at
  an arbitrary date in that cell's pool. v0 vol states are relative to the trailing 180 days, so the absolute vol level jumps at every
  cell switch. RS also never draws the unlabelled warm-up bars (the first ~100 days of each symbol), which under-represents volatile
  listing periods. RS paths are therefore "less clustered" scenarios, not faithful replicas. In the selection, RS-only picks (d -0.57)
  look better than SBB-only picks (d -0.21) only because they disagree with the champion more often (identical choice 0% vs 26%).

**Selection stability.**
- Across the 3 seeds, the zoo *med* pick has a mean modal share of 0.78 and all seeds agree for 43% of symbols. For the zoo *q10* pick
  the figures are 0.57 and 13%. For the breakout grid: *med* 0.68 / 17%, *q10* 0.55 / 4%.
- Resampling the 200 paths of seed 0 keeps the same *med* pick 68% of the time (zoo) and 54% (breakout); *q10* 47% / 53%.
- The rank correlation between the in-sample Sharpe ranking and the generated-median ranking is 0.61 (zoo) and 0.41 (breakout).
- The q10 criterion is the least stable: the 10th percentile of ~200 noisy Sharpes is a noisy order statistic.

## Overfitting diagnostics (descriptive, orchestrator request; not registered)

**PBO by CSCV** (Bailey, Borwein, Lopez de Prado & Zhu): `pbo_cscv(perf, n_blocks=16)`, C(16,8) = 12,870 splits, run on each symbol's
llm_in_sample daily P&L matrix.

| | zoo (21) | breakout (8) |
|---|---|---|
| PBO median / mean over symbols | 0.44 / 0.50 | 0.65 / 0.62 |
| share of symbols with PBO > 0.5 | 0.39 | 0.65 |
| Spearman(PBO, H-SYN1 d) | -0.10 | +0.12 |
| Spearman(PBO, H-SYN2 e) | -0.20 | +0.20 |
| Spearman(selection instability, d) | +0.24 | -0.18 |
| mean d, high- vs low-PBO half | -0.59 vs -0.36 | -0.56 vs -0.84 |
| mean e, high- vs low-PBO half | +0.22 vs +0.14 | +0.53 vs +0.48 |
| CPCV pick: holdout Sharpe minus champion | -0.15 [-0.67, +0.40] | +0.24 [-0.11, +0.61] |
| CPCV pick: degradation minus champion | +0.09 [-0.47, +0.61] | -0.28 [-0.66, +0.07] |
| CPCV pick = champion | 0.48 | 0.57 |

- Per-symbol selection overfits about as often as not (zoo), and more often than not for breakouts.
- Koshiyama et al. report that generated-path selection helps mainly where PBO is high. We do **not** see that here: the correlations
  have mixed signs and are small with 23 symbols, and the zoo difference between the PBO halves (-0.59 vs -0.36) has the expected sign
  but is far inside the noise.
- The CPCV comparator scores each candidate by its median test-split Sharpe over C(6,2) = 15 splits with a 5-day embargo. For unfitted
  rules the training folds carry no information, so this degenerates towards full-sample selection. It matches the champion half the
  time and is no better.

## Caveats

- **Holdout power.** The holdout is 133 days. One symbol's Sharpe has a standard error of about 1.7, and the 23 symbols share the same
  days (crypto factor), so the symbol bootstrap overstates the independent information. The day-block bootstrap gives similar
  intervals here.
- **Mechanical component of H-SYN1.** Degradation falls for any non-champion choice. The registered statistic cannot separate
  regularisation from "choosing something worse in-sample", and the random-pick null shows it is the latter.
- **Seed fragility.** The pre-specified seed-0 verdict for H-SYN1 does not survive seeds 1 and 2.
- **Generated data cannot add information** (it is a reshuffle of the same in-sample path). Blocks of ~3.5 weeks keep in-sample trend
  episodes almost intact, so generated Sharpes are a smoothed copy of the in-sample ranking (rank correlation 0.61), not an independent
  view.
- **Regime-switching generator.** As designed, it destroys clustering beyond ~half a day (see the diagnostics and the diary
  `mistake`). A better generator needs a new hypothesis id.
- **Universe.** The universe is today's survivors. The bootstrap treats symbols as exchangeable.
- **Multiplicity.** There are 2 grids, 3 selection rules, 2 generators, 3 seeds and 2 hypotheses. Only the registered statistic gives
  the verdict; everything else is descriptive.

## Recorded results

- `H-SYN1` clean_holdout **supports**: literal criterion, seed 0. The note states the random-null and seed caveats.
- `H-SYN1` llm_in_sample **inconclusive**: diagnostics only.
- `H-SYN2` clean_holdout **inconclusive**.
- `H-SYN2` llm_in_sample **inconclusive**: diagnostics only.

## Rule candidates

```yaml
- id: R-SYN-1
  kind: avoid
  rule: "WHEN choosing trend-zoo or Donchian-breakout parameters per symbol THEN don't expect selection on block-bootstrap or regime-resampled paths (median or 10th-percentile Sharpe) to beat a fixed or random choice in the holdout"
  when: "per-symbol parameter choice among the 21 zoo variants or an 8-point Donchian grid at 4h, using paths resampled from in-sample 1h bars"
  then: "use fixed pre-registered parameters or the equal-weight ensemble; treat generated-path rankings as a smoothed copy of the in-sample ranking (rank corr 0.6), not new evidence"
  scope: {assets: crypto, horizon: "4h-1d bars, multi-week holding", sides: both}
  effect: "degradation reduction -0.59 Sharpe = the random-pick null (-0.59, CI [-0.90, -0.31]); not seed-robust (-0.41, -0.34, CIs cover 0); q10 pick vs champion holdout +0.18 [-0.53, +0.91], sign flips across seeds"
  confidence: C
  evidence: [H-SYN1, H-SYN2, research/reports/synthetic_robustness_2026-11-11.md]
  review: "a generated-path selection that beats the random-pick null with a CI excluding 0 after >= 12 holdout months"
  status: provisional
- id: R-SYN-2
  kind: expect
  rule: "WHEN a per-symbol trend or breakout champion is picked from its in-sample record THEN expect a coin-flip chance that it is overfit (PBO about 0.45 for the zoo, 0.65 for breakouts) and a loss of about 1.1 Sharpe out of sample"
  when: "best-in-sample variant per symbol among the trend zoo (21) or the Donchian grid (8), 4h bars"
  then: "zoo champion in-sample +0.76 vs holdout -0.37; the breakout champion (often a 5-10 day channel) does worse in the holdout than a random grid point; don't promote a per-symbol champion"
  scope: {assets: crypto, horizon: "4h-1d bars", sides: both}
  effect: "PBO (CSCV S=16) median 0.44 zoo / 0.65 breakout; random pick beats the breakout champion by +0.38 Sharpe (CI [+0.04, +0.69]) in the holdout"
  confidence: B
  evidence: [H-SYN1, H-TR2, research/reports/synthetic_robustness_2026-11-11.md]
  review: "holdout champion Sharpe within 0.2 of in-sample over 12 months, or PBO < 0.2 on a refreshed in-sample window"
  status: provisional
- id: R-SYN-3
  kind: prefer
  rule: "WHEN generating synthetic crypto price paths for robustness checks of deterministic rules THEN prefer a stationary block bootstrap of whole 1h relative-OHLC bars with a Politis-White block on |r| (about 3-4 weeks), and don't use cell-by-cell v0 regime resampling, which destroys vol clustering beyond about half a day"
  when: "building generated paths for parameter or stress checks (never for the LLM layer)"
  then: "SBB reproduces 1h/4h kurtosis (0.96x/1.02x), |r| ACF to 24h (0.94x) and v0 episode durations (1.02x); expect it to understate clustering at one week (0.72x)"
  scope: {assets: crypto, horizon: "1h-1w", sides: both}
  effect: "regime-switching resampler: |r| ACF 24h 0.24x, 168h 0x real; real value inside the generated 5-95% band for 0% of symbols"
  confidence: B
  evidence: [H-SYN1, research/reports/synthetic_robustness_2026-11-11.md]
  review: "a vol-level-matched regime resampler (new hypothesis id) that reproduces |r| ACF at 24h and 168h"
  status: provisional
```
