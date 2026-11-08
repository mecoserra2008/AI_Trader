# Trend vs mean-reversion regime switch: H-TR4 and H-TR3 (2026-11-08)

Code: `features/regime_features.py` (tests in `tests/test_regime_features.py`), `research/regime_switch_study.py`
(tests in `tests/test_regime_switch_study.py`), live tool `tools/regime_tf.py` (`regime.trend_vs_mr` 0.1.0).
Data: Bybit 4h bars for the 23-symbol universe, `market_frame(funding=False, oi=False)`. The trend leg is the
trend-zoo equal-weight ensemble from `data/priors/trend_zoo_daily_4h.parquet` (variant `ensemble`, written by the
trend-zoo study). The MR leg is the equal-weight signal of `zscore_reversion` and `rsi_reversion` (as configured) on
4h bars, run through `evaluate.strategy_pnl` (risk units, 7.5 bps per side). Daily P&L is summed by the UTC date of
`event_ts` (the H-X2b convention, the same one the zoo file uses).
Signals, features and P&L are all computed on the full history and split by segment afterwards. The in-sample run
was recorded (00:32:12 UTC) before the holdout was computed (00:32:41). Nothing was changed between the two runs.

## Design (fixed before results)
- **Features** (one row per UTC day D, known at 00:00 on D+1; bars sit on a regular 4h grid, so a missing bar gives a
  missing return rather than one that spans the gap): ER 7d/14d/30d; DVR = (RS+ − RS−)/(RS+ + RS−) over 7d/30d;
  trend t-stat 7d/30d; Lo–MacKinlay VR(q = 6, i.e. 1d vs 4h) over 30d; lag-1 autocorrelation of 4h and of 1d
  returns over 30d; percentile of the robust vol (`sigma_risk_next`) within the trailing year; vol-of-vol = sd of
  log daily RV over 30d. Unit tests cover truncation invariance (the frame and the vol model are recomputed on
  truncated bars), ER = 1 on a straight line, the DVR sign on zero-drift asymmetric returns, AC/VR on AR(1) data,
  and gap handling.
- **Sample:** symbol-weeks. The decision is at Monday 00:00 UTC using Sunday's features. The outcome is the P&L
  dated Monday through Sunday (all 7 days present for both legs), so it never overlaps the feature window. A week
  is in-sample if D+8 < 2026-07-01 and holdout if D+1 ≥ 2026-07-01; the week that straddles the cutoff is dropped.
  This gives 5416 in-sample symbol-weeks (314 weeks) and 391 holdout (17 weeks).
- **H-TR4:** er_14d terciles. The cut points are the in-sample quantiles (0.067 and 0.147), and the holdout reuses
  them. Statistic: each week, the mean next-7d trend P&L of top-tercile symbols minus bottom-tercile symbols; mean
  and NW t over weeks.
- **H-TR3:** a logistic regression on the 11 registry features (er_14d is not an input). It is refit at the start
  of every month on all symbol-weeks with D + 9 ≤ month start, with at least 52 weeks of training data. Features are
  standardised on the training window, and a ridge of 1 (negligible at n ≈ 5000) keeps the fit numerically stable.
  Target = 1{trend7 − MR7 > 0}. The first out-of-sample week is 2021-10-31, and there are 61 refits up to 2026-11-01.
  IC = pooled Spearman of p vs realised (trend7 − MR7), with t clustered by week. Gated = the trend ensemble when
  p > 0.5, flat otherwise. Sharpe = annualised daily Sharpe per symbol on the out-of-sample days, averaged over
  symbols.
- **Checks:** a random-gate null (500 draws with the same exposure); a walk-forward placebo (100 runs with outcomes
  permuted across whole weeks); coefficient paths; and separate fits on each calendar year.

## H-TR4: does the 14-day efficiency ratio predict trend P&L?

| next-7d net P&L per symbol-week (risk units) | bottom ER14 | middle | top ER14 |
|---|---:|---:|---:|
| in-sample: trend ensemble | **+0.34** | +0.29 | **−0.11** |
| in-sample: MR ensemble | −0.16 | −0.15 | −0.00 |
| in-sample: trend − MR | +0.50 | +0.44 | −0.11 |
| holdout: trend ensemble | −0.32 | +0.36 | −0.20 |
| holdout: trend − MR | +0.05 | +0.81 | −0.01 |

| segment | weeks | top − bottom (weekly, registered) | NW t | pooled slope, week-clustered t | within-week terciles (diagnostic) |
|---|---:|---:|---:|---:|---:|
| in-sample | 217 | **−0.49** | **−2.33** | −0.45 (t −1.71) | −0.26 (t −1.94) |
| holdout | 16 | +0.27 | 0.71 | +0.11 (t 0.23) | +0.61 (t 2.00) |

**Verdict: in-sample contradicts; holdout supports (sign only, t 0.7). Overall: not supported.** In-sample, the sign
is the *opposite* of the hypothesis. After the most efficient two-week moves, the trend ensemble did worst the next
week (late entries into moves that then reversed). The holdout flips the sign back. Neither direction is stable,
and the reversed in-sample sign was not registered, so it cannot be claimed.

## H-TR3: walk-forward classifier

| segment | OOS symbol-weeks (weeks) | trend wins (base rate) | mean p | share p > 0.5 | **IC (pooled Spearman)** | **t (week-clustered)** | weekly cross-sectional IC (NW t) | accuracy |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| in-sample | 4909 (243) | 36.6% | 0.38 | 5.3% | **−0.019** | **−0.55** | −0.040 (−2.03) | 62.3% |
| holdout | 391 (17) | 39.6% | 0.37 | 1.5% | **+0.026** | **0.30** | +0.049 (0.86) | 59.8% |

| segment | ungated trend Sharpe | **gated Sharpe** | gated − ungated (t over symbols) | exposure | random-gate null: mean / 95% / p | MR ensemble Sharpe |
|---|---:|---:|---:|---:|---:|---:|
| in-sample | **+0.27** | **+0.20** | −0.07 (−0.78) | 5.3% | −0.01 / +0.16 / **0.028** | −0.42 |
| holdout | **−0.31** | **−0.07** | +0.24 (0.83) | 1.5% | −0.12 / +0.14 / 0.44 | −1.03 |

Walk-forward placebo with week-permuted outcomes (100 runs): the mean IC t is −0.35 in-sample (sd 0.91) and −0.11
in the holdout (sd 1.12), and 1% of runs exceed t = 2. The pipeline has no leakage, and t-stats are on the right scale.

**Verdict: in-sample contradicts (IC < 0, and gated < ungated); holdout supports by the registered sign-only rule.
Overall: not supported.** The holdout "pass" is trivial. The gate was on for 6 of 391 symbol-weeks, the trend
ensemble lost money over the holdout, and being flat 98.5% of the time beats losing. The random-gate null shows the
same thing (p = 0.44). In the 6 weeks the gate did pick, trend − MR was −0.52 against +0.28 in the other weeks.

Two things I noticed but cannot use, because they are post hoc:
- *In-sample, the few weeks with p > 0.5 were better than random weeks:* trend − MR +0.54 vs +0.26, and gated Sharpe
  beat the random-gate null with p = 0.028. This did not repeat in the holdout.
- *A design weakness in the registered test.* Trend P&L is positively skewed: the trend ensemble wins only ~37% of
  symbol-weeks, but by more than it loses on average. A classifier on the *sign* with a p > 0.5 gate is therefore
  almost always flat. So "gated vs ungated" mostly tests "flat vs trend", and the gate can only win in periods when
  trend loses. A regression on the size of (trend − MR), or a gate at the base rate, would be a fairer test. That
  would be a new hypothesis and an extra trial. It was not run here.

### Are the coefficients stable?

Standardised logistic coefficients. "Monthly path" = the 61 expanding-window refits. "Yearly fits" = separate fits on
each calendar year's symbol-weeks (disjoint samples, which is the honest test of stability).

| feature | first refit (2021-11) | last refit (2026-11) | z at last refit | share of refits with the median's sign | yearly fits 2022 / 23 / 24 / 25 / 26 |
|---|---:|---:|---:|---:|---|
| er_7d | −0.15 | −0.04 | −1.2 | 0.54 | +0.11 / +0.05 / −0.22 / −0.19 / +0.05 |
| er_30d | −0.32 | +0.05 | +1.6 | 0.79 | +0.07 / +0.05 / +0.02 / −0.03 / +0.02 |
| dvr_7d | +0.04 | +0.14 | +2.2 | 0.67 | −0.41 / +0.30 / +0.27 / +0.49 / −0.14 |
| dvr_30d | +1.09 | −0.02 | −0.4 | 0.92 | +0.12 / +0.34 / −0.32 / −0.22 / −0.21 |
| tstat_7d | +0.16 | −0.15 | −2.3 | 0.67 | +0.39 / −0.05 / −0.08 / −0.58 / −0.29 |
| tstat_30d | −0.61 | +0.02 | +0.3 | 0.90 | −0.15 / −0.32 / +0.37 / +0.24 / +0.19 |
| vr_30d | +0.50 | +0.12 | +2.8 | 0.89 | −0.04 / +0.08 / +0.37 / +0.04 / +0.21 |
| ac1_4h_30d | −0.19 | −0.07 | −1.8 | 0.70 | +0.16 / −0.11 / −0.33 / +0.11 / −0.28 |
| ac1_1d_30d | +0.21 | +0.01 | +0.2 | 0.56 | −0.22 / +0.09 / +0.18 / −0.09 / −0.10 |
| vol_pct_1y | −0.28 | −0.15 | **−4.8** | 0.93 | +0.06 / −0.27 / −0.18 / −0.23 / +0.03 |
| vol_of_vol_30d | +0.37 | +0.05 | +1.8 | 0.54 | −0.03 / −0.04 / +0.10 / +0.21 / +0.00 |

**No, apart from weak hints.** The expanding-window path looks smooth only because each refit shares most of its
data with the previous one. Across disjoint years, most features change sign, and the trend and directional
features (dvr, tstat, ac1) flip hardest. The only persistent hint is **vol_pct_1y < 0**: trend is less likely to
beat MR when vol is high relative to the past year (z −4.8 in the last fit, negative in 3 of 5 years). vr_30d > 0
is positive in 4 of 5 years. The model-based z ignores cross-symbol correlation within a week, so it overstates
precision, and 11 coefficients were examined. Neither hint is a finding.

## Current regime (the live table, data to 2026-11-07, model fitted for 2026-11-01)

`regime.trend_vs_mr` returns these numbers for one symbol. Training base rate: 0.37.

| symbol | P(trend beats MR, next 7d) | ER7 | ER14 | ER30 | DVR30 | t-stat 30d | VR30 | vol pct 1y |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| BTCUSDT | 0.33 | 0.10 | 0.15 | 0.08 | −0.32 | −0.69 | 0.84 | 0.57 |
| ETHUSDT | 0.34 | 0.02 | 0.09 | 0.08 | −0.22 | −0.66 | 0.87 | 0.58 |
| SOLUSDT | 0.32 | 0.17 | 0.19 | 0.13 | −0.30 | −1.21 | 0.65 | 0.49 |
| ADAUSDT | 0.33 | 0.13 | 0.16 | 0.13 | −0.29 | −1.17 | 0.72 | 0.48 |
| AVAXUSDT | 0.31 | 0.09 | 0.10 | 0.10 | −0.22 | −0.94 | 0.69 | 0.62 |
| LINKUSDT | 0.29 | 0.17 | 0.07 | 0.08 | −0.25 | −0.73 | 0.65 | 0.77 |
| XRPUSDT | 0.33 | 0.22 | 0.18 | 0.13 | −0.30 | −1.15 | 0.95 | 0.59 |
| DOGEUSDT | 0.33 | 0.15 | 0.12 | 0.11 | −0.33 | −0.96 | 0.69 | 0.42 |
| others | 0.27 (UNI) – 0.34 (BNB) | | | | | | | |

All 23 symbols are in the range 0.27–0.34, just below the base rate. The model nudges P down because 30-day VR is
below 1 everywhere, the 7-day downside-vol share is high, and vol percentiles are mid-to-high. The table is
descriptive of the market's current shape (slow, choppy decline, 30d DVR negative everywhere). It is not a
validated switch.

## Verdicts recorded (`research/hypotheses_results.jsonl`)

| id | in-sample | holdout | overall |
|---|---|---|---|
| H-TR4 | contradicts (−0.49, t −2.33: wrong sign) | supports (sign only, +0.27, t 0.7, 16 weeks) | not supported |
| H-TR3 | contradicts (IC −0.02, t −0.55; gated 0.20 < ungated 0.27) | supports (sign only; trivial at 1.5% exposure, null p 0.44) | not supported |

Trial count: 2 hypotheses × 2 segments, one fixed model spec, plus the diagnostics labelled above. Nothing is close
to a deflated threshold. `config/tools.yaml`: `regime.trend_vs_mr` is `planned` (descriptive).

## Plain-language read

**This is the third failed attempt to time trend vs mean reversion, and the most thorough one.** H-RG1 used a
single variance ratio. H-AU3 used value-area overlap. This test combined 11 features, including the vol and
directional-vol ones those designs lacked, and judged them strictly out of sample. The classifier has no rank
skill (IC ≈ 0 in both segments). Its coefficients change sign from year to year. The simplest "is it trending?"
measure (ER14) points the wrong way in-sample and the right way in the holdout.

What does hold up is the level, not the timing. In-sample, the 4h trend ensemble beats the MR ensemble on average
(+0.28 risk units per symbol-week overall; it falls behind only in the top ER14 tercile), and the MR ensemble loses after costs (Sharpe −0.42 in-sample,
−1.03 in the holdout). The holdout was a bad period for trend (Sharpe −0.31), and no feature saw it coming.
Practical use for the decision session: show `regime.trend_vs_mr` as context (how efficient, one-sided and
volatile each market is right now), but do not let P(trend wins) switch strategies or size positions. If a
regime overlay is ever retried, it should predict the size of trend P&L or gate at the base rate (see the design
weakness above), and it would need to be registered as a new hypothesis.
