# Volatility-based signals (H-VB1..H-VB4): 2026-11-10

Agent A3, program `research/plans/2026-11-10_volatility_program.md`. Module `src/ai_trader/research/vol_signal_study.py`,
tests `tests/test_vol_signal_study.py` (13 pass), full numbers in `vol_signals_2026-11-10.json`. Data as of
2026-11-10 22:49 UTC. Design fixed before results (diary 2026-11-10, "A3 vol-signals: design fixed before results").

**Bottom line:** none of the four signals is supported. Variance management adds nothing beyond risk-unit sizing (VB1).
Daily vol surprises don't continue (VB2). Squeeze-release breakouts don't follow through net of costs in crypto (VB3).
A Hawkes model of 2-sigma hourly exceedances loses to the plain HAR-plus-Student-t rate (VB4). All four become
"don't expect" rules.

## Data and shared definitions

- **Universes.** L is the 23 crypto perps of `trend_zoo_study.UNIVERSE`. T is the 16 TradFi perps, with 115 to 555
  complete days each, almost all of them after the cutoff.
- **Bars and daily measures.** Bars come from `market_frame` (1h and 4h, funding and OI off). Daily RV and returns come
  from `features.realized.daily_measures` on the same 1h bars (identical to `daily_panel`).
- **Segments.** Each result is assigned by its decision timestamp, compared with `knowledge_cutoff` (2026-07-01). The
  holdout is 132 days for daily studies.
- **Daily variance forecast F.**
  - L: per-symbol `har_forecast_x` (log-HAR, expanding window, refit every 30 days, first forecast at day 180).
  - T: pooled log-HAR (`pooled_har_forecast`: common slopes, refit every 30 days, first fit after 300 pooled pairs),
    plus target-day weekend and Monday dummies. TradFi perps have roughly 5x lower RV at weekends.
- **Costs.** 7.5 bps per side throughout.
- **Multiple testing.** Holm across the four VB primaries in each segment. Every adjusted p-value is 1.0, except
  H-VB2 in the holdout at 0.51.

## H-VB1 Variance-managed exposure (Moreira & Muir 2017)

**Method**

- **Factor level, as in the paper.** The base series are:
  - the EW long-only index (EW 1h simple returns, compounded hourly);
  - the EW trend-zoo ensemble (4h, risk units).
- **Ensemble replica check.** I recomputed the ensemble with the zoo and the `strategy_pnl` execution policy (the
  `held_leverage` test proves the policy is identical). It equals `data/priors/trend_zoo_daily_4h.parquet` on every
  symbol-day except each symbol's first evaluation day (≤ 0.031) and the parquet's partial last day.
- **Weight.** w_D = c / F_D, where F_D is the HAR forecast of the index's RV for day D, made at the end of D-1.
- **Costs.** 7.5 bps × |Δw| for long-only. For the ensemble, 7.5 bps × |Δ(w·leverage)| per 4h bar.
- **Test.** Spanning regression managed = a + b·base, HAC t.
- **Choice of c.** c matches the full-sample vol, as in the paper. A constant c leaves the alpha t unchanged; only the
  cost scales with it. The real-time c variant (expanding window) gives almost the same results.
- **Variants.** These don't drive the verdict: real-time c, MM-native σ² (sum of the previous 22 squared daily base
  returns) and per-symbol HAR.

**Primary results (`har_fullc`).** Alpha is annualised: log return for long-only, risk units for the ensemble.

| universe | base | segment | n days | alpha/yr | t | SR managed | SR base |
|---|---|---|---|---|---|---|---|
| L | long-only | llm_in_sample | 1738 | −0.228 | −1.24 | −0.49 | −0.19 |
| L | ensemble | llm_in_sample | 1738 | +4.53 | 1.00 | 0.62 | 0.44 |
| L | long-only | clean_holdout | 132 | −0.282 | −0.56 | 0.89 | 1.37 |
| L | ensemble | clean_holdout | 132 | +1.05 | 0.14 | −0.19 | −0.27 |
| T | long-only | clean_holdout | 132 | −0.108 | −0.25 | −0.75 | −0.77 |
| T | ensemble | clean_holdout | 132 | −15.9 | −0.49 | −1.07 | −0.61 |

**Variants**

- **MM-native 22-day RV on the ensemble.** In-sample alpha is +12.5 (t 2.08, SR 1.02 vs 0.50). In the holdout it is
  +3.45 (t 0.16). The deflated Sharpe over the 10 in-sample variant series is 0.005, so this is the best of 10, not
  evidence.
- **Per-symbol HAR on the ensemble.** In-sample t is 1.50; the holdout alpha is −5.4.
- **T weights.** With weekend-aware forecasts, w swings weekly: p95 is 8x and turnover is 1.7 per day.

**Verdict.** Inconclusive in llm_in_sample: the best base, the ensemble, has alpha > 0 but t 1.0 < 2. Inconclusive in
clean_holdout: L is positive for the ensemble, T is negative for both bases. Pooled: inconclusive.

## H-VB2 Daily vol surprise continuation

**Method**

- **Event.** RV_t > 1.5 F_t and |ret_t| > √F_t, where F_t is made at the end of t-1. The median event has RV/F = 2.3.
- **Declustering.** 3 days per symbol.
- **Outcome.** The signed return over the next 1 or 3 days, divided by √(h·F_{t+1|t}), net of 2 × 7.5 bps.
- **Statistics.** The estimate is the mean of per-date means; the t-stat is NW over the date series. L and T are
  pooled.

| segment | n events / dates | 1d net (t) | 3d net (t) | 3d gross (t) |
|---|---|---|---|---|
| llm_in_sample | 2620 / 763 | −0.014 (−0.39) | −0.008 (−0.25) | +0.013 (0.41) |
| clean_holdout | 345 / 92 | −0.118 (−1.24) | +0.150 (1.14) | +0.188 (1.43) |

**Splits**

- **By v0 vol state at the start of day t (in-sample, 3d net).** Low: +0.064 (t 1.21, n 1355). Normal: −0.018. High:
  −0.059 (t −1.18). The ordering matches P1 but is far from significant. In the holdout, low is −0.023.
- **Holdout by universe.** L: −0.10 (t −0.86). T: +0.37 (t 2.11, 148 events, holdout-only data).
- **Baseline (all non-event days signed by their own return).** 1d net is −0.031 (t −2.08) in-sample, a mild daily
  reversal after costs.

**Verdict.** Contradicts in llm_in_sample (estimate ≤ 0). In clean_holdout the criterion is literally met (positive
sign), but t is only 1.1 and the in-sample test failed. Pooled: contradicts.

## H-VB3 Squeeze-release breakouts (4h)

**Method**

- **Squeeze.** BB(20, 2, population sd) inside KC(EMA20 ± 1.5 × SMA20 of true range).
- **Release and event.** Release is the first non-squeeze bar after at least 6 squeeze bars. The event is the first
  close outside BB at or after release, before a new squeeze starts and within 20 bars of release.
- **Comparison set.** Fresh BB breakouts (previous close inside), not in a squeeze, with no release in the last 20
  bars.
- **Declustering.** 6 bars per symbol and class.
- **Outcome.** sign × log(c_{t+6}/c_t) / (σ_next·√6), net of costs.
- **Statistics.**
  - Means: NW on the date means.
  - Difference: OLS on a squeeze dummy, SE clustered by date.
  - Random entry: 200 count-matched draws per symbol, sign = side of BB mid.

| segment | squeeze n | squeeze net (t) | plain net (t) | diff (t) | random entry mean [5%, 95%] |
|---|---|---|---|---|---|
| llm_in_sample | 2262 | −0.079 (−2.07) | −0.068 (−2.90) | +0.023 (0.48) | −0.020 [−0.057, 0.022] |
| clean_holdout | 213 | +0.022 (0.17) | −0.127 (−1.56) | +0.305 (1.82) | −0.057 [−0.183, 0.077] |

**Details**

- **Event mean vs date mean.** The in-sample event mean is +0.009: days with many simultaneous crypto breakouts carry
  more weight in it than in the date mean.
- **Gross squeeze.** In-sample date mean is −0.028 (t −0.74).
- **T squeezes.** Pooled over segments: +0.29 (t 1.42, 69 events). These T z-scores are inflated by low weekend σ.

**Verdict.** Contradicts in llm_in_sample: the squeeze mean is ≤ 0, t is −2.07, and the difference has t 0.48. In
clean_holdout the signs are as hypothesised, so the literal criterion is met, but neither t reaches 2. Pooled:
contradicts. Breakouts in general lose money net of costs at 4h in crypto.

## H-VB4 Hawkes vs HAR-t for 1h 2-sigma exceedances

**Method**

- **Exceedance.** |r_t| > 2σ_t, where σ_t = `sigma_risk_next` of bar t-1 (robust EWMA).
- **Hawkes model**
  - Fit: per symbol, exponential kernel, the `fit_hawkes` continuous-time likelihood evaluated on a bar grid. The test
    checks it equals `jump_diffusion._hawkes_nll`. Expanding window, refit on the first of each month.
  - Intensity: from `hawkes_intensity`.
  - 24h expectation: the closed form m·H + (λ₀ − m)(1 − e^{−(β−α)H})/(β−α), with m = μβ/(β−α).
- **HAR-t mapping**
  - Per-bar variance is F_{D+1}/24 (flat intraday).
  - Bar returns are unit-variance Student-t, with ν = 3.35 fitted by MLE on L llm_in_sample hourly returns divided by
    √(F/24).
  - p = P(|t_ν| > 2σ₀/√(F/24)), with σ₀ the threshold σ at the decision bar (00:00 UTC). The rate is 24p.
- **Score.** Poisson log score of the next-24h count. Paired difference per symbol-day, averaged per date, NW t.

| segment | symbol-days / dates | Hawkes − HAR-t (t) | Hawkes − constant (t) | HAR-t − constant (t) | corr(N, λ) Hawkes / HAR-t | mean N / λ_HAR-t / λ_Hawkes |
|---|---|---|---|---|---|---|
| llm_in_sample | 35744 / 2068 | −0.0137 (−2.14) | −0.000 (−0.12) | +0.014 (2.13) | 0.03 / 0.31 | 1.32 / 1.47 / 1.33 |
| clean_holdout | 4591 / 132 | −0.0188 (−0.70) | +0.004 (4.51) | +0.023 (0.84) | 0.09 / 0.48 | 1.45 / 1.69 / 1.40 |

**Details**

- **Fitted Hawkes (median of the last fit per symbol).** Branching ratio 0.05, half-life 2.1 h, LR 10.7.
- **Why Hawkes adds so little.** Once returns are standardised by a robust EWMA σ, 2σ exceedances barely
  self-excite. The robust σ has already absorbed the clustering.
- **Where the information is.** It sits in the ratio of the daily HAR forecast to the current hourly σ, which HAR-t
  uses.
- **HAR-t calibration.** It over-predicts counts by about 11% in-sample and 16% in the holdout. Even so, it still
  wins on log score.

**Verdict.** Contradicts in llm_in_sample (diff < 0, t −2.14) and in clean_holdout (diff < 0). Pooled: contradicts.

## Verdicts recorded (`research/hypotheses_results.jsonl`)

| id | llm_in_sample | clean_holdout | pooled |
|---|---|---|---|
| H-VB1 | inconclusive | inconclusive | inconclusive |
| H-VB2 | contradicts | supports (sign only, t 1.1) | contradicts |
| H-VB3 | contradicts | supports (signs only, t 0.2 / 1.8) | contradicts |
| H-VB4 | contradicts | contradicts | contradicts |

## Caveats

- **"Supports" in the holdout only means the sign held.** For VB2 and VB3, the holdout rows meet the literal
  registered criterion (same sign), but t < 2 and the in-sample test failed. They are not evidence for the
  hypotheses.
- **Universe T is holdout-only and short.** T has at most 132 forecast days, 2 to 16 symbols, and a pooled HAR whose
  first fit came on 2026-06-24. TradFi weekend vol is about 5x lower, so:
  - VB1 weights swing weekly;
  - σ-unit returns from VB3 and VB4 events at weekends are inflated.

  Confidence for T stays at C.
- **The VB1 ensemble base is already vol-scaled.** It is sized in risk units at 1/σ̂ EWMA, so management on top of it
  tests the residual timing value only. The paper's equity factors are unscaled.
- **VB4's HAR-t baseline ignores intraday seasonality** and uses σ₀ for all 24 bars. A better-calibrated baseline would
  only widen its lead over Hawkes.
- **The universe is today's survivors.** In-sample crypto results are textbook-contaminated (H-VB1, plan §7).
- **Bug caught and fixed before recording.** The first run used the event mean with the date-mean t; see the
  2026-11-10 diary "mistake" entry. The fix flipped the H-VB3 in-sample estimate from +0.009 to −0.079.

## Rule candidates

```yaml
- id: R-VSIG-1
  rule: "WHEN sizing crypto long exposure or the trend-zoo ensemble THEN don't expect variance-managed scaling (exposure x c / HAR forecast variance) to add risk-adjusted return beyond fixed-risk sizing"
  when: "asset_class crypto (universe L) or TradFi perps; exposure scaled by 1 / next-day HAR variance of the EW index or of the symbol"
  then: "keep the risk engine's fixed-risk sizing; treat vol-timing overlays as unproven (the MM 22d-RV ensemble variant is best of 10 trials, DSR 0.005)"
  scope: {assets: all, horizon: "1d", sides: both}
  effect: "spanning alpha t: long-only -1.24 (L in-sample), +1.00 ensemble (L in-sample); holdout t -0.56 / +0.14 (L), -0.25 / -0.49 (T)"
  confidence: B
  evidence: [H-VB1, research/reports/vol_signals_2026-11-10.md]
  review: "holdout ensemble alpha t > 2 with the in-sample-fixed variant, or a registered test of a capped/real-time variant"
- id: R-VSIG-2
  rule: "WHEN a day's realised variance exceeds 1.5x the HAR forecast and the day moved more than 1 sigma THEN don't expect the move to continue over the next 1-3 days; the surprise informs volatility, not direction"
  when: "RV_t / HAR F_t > 1.5 and |ret_t| > sqrt(F_t), daily, any v0 vol state"
  then: "no directional preference after a daily vol surprise; use the raised vol forecast for stops and expectations"
  scope: {assets: all, horizon: "1d", sides: both}
  effect: "signed next-3d return net -0.008 sigma (t -0.25, 2620 events) in-sample; holdout +0.15 sigma (t 1.14, 345 events); low-vol-state subset +0.06 (t 1.2)"
  confidence: B
  evidence: [H-VB2, research/reports/vol_signals_2026-11-10.md]
  review: "holdout 3d mean stays > 0 with t > 2 as n doubles (driven by TradFi, t 2.1), or a registered low-vol-state variant (P1 analogue) passes"
- id: R-VSIG-3
  rule: "WHEN a 4h Bollinger/Keltner squeeze releases with a close outside the bands THEN don't expect directional follow-through over the next 24h net of costs; plain Bollinger breakouts lose too"
  when: "4h, BB(20,2) inside KC(EMA20, 1.5 ATR20) for >= 6 bars, first close outside BB within 20 bars of release"
  then: "don't treat squeeze releases or BB closes as entry signals; compression is not a directional edge (with H-VCL5: not a vol edge either)"
  scope: {assets: crypto, horizon: "1d", sides: both}
  effect: "signed next-6-bar return net -0.079 sigma (t -2.07, 2262 events) in-sample vs plain -0.068 (t -2.90); diff +0.02 (t 0.48); holdout diff +0.30 (t 1.82)"
  confidence: B
  evidence: [H-VB3, research/reports/vol_signals_2026-11-10.md]
  review: "holdout squeeze-minus-plain diff reaches t > 2 with n >= 400 squeeze events; TradFi squeezes (+0.29 sigma, t 1.4, 69 events) need a weekend-robust sigma before re-test"
- id: R-VSIG-4
  rule: "WHEN judging the chance of more 2-sigma hourly bars in the next 24h THEN prefer the HAR daily variance mapped through a Student-t (df ~3.4) against the current hourly sigma over a self-exciting (Hawkes) count; don't expect exceedances to cluster once returns are scaled by the robust sigma"
  when: "1h bars, threshold |r| > 2 x sigma_risk (robust EWMA, known at t-1), decision at 00:00 UTC"
  then: "expected next-24h count = 24 x P(|t_3.4| > 2 sigma_0 / sqrt(F/24)), shaded down ~10% (it over-predicts); a recent exceedance adds little (branching ~0.05, half-life ~2h)"
  scope: {assets: all, horizon: "1d", sides: both}
  effect: "log score Hawkes - HAR-t -0.014 nats/symbol-day (t -2.14) in-sample, -0.019 (t -0.70) holdout; corr(count, rate) 0.31 / 0.48 HAR-t vs 0.03 / 0.09 Hawkes"
  confidence: B
  evidence: [H-VB4, research/reports/vol_signals_2026-11-10.md]
  review: "a Hawkes on raw (unstandardised) exceedances or a HAR-t with intraday seasonality is registered and tested; holdout sign flips with t > 2"
```
