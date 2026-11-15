# Backtests on generated markets: a parameter-uncertain 1h generator (H-GEN1, H-GEN2, H-GEN3)

_Agent N, 2026-11-13. Code: `src/ai_trader/sim/market_gen.py` (generator), `src/ai_trader/research/market_gen_study.py` (calibration,
runs, evaluation); tests `tests/test_market_gen.py` (9 pass). Machine-readable: [market_gen_2026-11-13.json](market_gen_2026-11-13.json).
Generated data is for the deterministic strategies only. It never trains or backtests the LLM layer (CLAUDE.md principle 1)._

## Bottom line

1. **H-GEN1 is `inconclusive`.** The parameter-uncertain generator covers the realised clean_holdout Sharpe of 11/12
   family x universe pairs (coverage 0.92, Wilson CI [0.65, 0.99]). That meets the
   80% bar. The fixed-parameter version scores exactly the same (11/12), so the registered "higher than fixed"
   condition fails on a tie. The B5-style stationary bootstrap covers 10/12. At the strategy level the
   counts are 29/30, 29/30 and 26/30. Parameter uncertainty widens the 5-95% Sharpe band by only about 7% (median width
   5.92 vs 5.54). Over 132 days, path noise dominates parameter noise: the band is
   about +/-3 Sharpe wide whichever generator is used. Coverage is therefore easy and says little.
2. **H-GEN2 `supports`: 8/10 facts lie inside the envelopes** (universe L). The two misses: the holdout's hourly kurtosis (34.8 vs a 95th
   percentile of 28.4) and a lower cross-asset return factor share (0.61 vs a 5th percentile of 0.63; correlations fell in the holdout). Both misses also showed
   up in the in-sample design check. The median regime duration sits on the band edge (9 h, and the generated 5th percentile is 9 h). The fixed generator
   also gets 8/10. The bootstrap gets 10/10, because it replays real bars.
3. **H-GEN3 `contradicts`: the generator does not rank strategies the way the holdout did.** Spearman of median generated Sharpe vs realised
   = +0.24 (CI [-0.48, +0.83], 12 pairs). At strategy level it is +0.09. The resampling bootstrap ranks better
   (+0.64, CI [+0.01, +0.93]) because it carries the real in-sample drift and microstructure, which the
   zero-drift parametric model removes by design.
4. **The one clear failure is positioning on L.** In the holdout, funding_contrarian and oi_price_divergence earned +0.63 Sharpe. In generated data they
   earn -3.4 (median), and every generator rejects the realised value (q 0.96-0.99). Synthetic funding and OI series (AR(1) with t shocks)
   fire the z-score triggers far more often than real funding does, since real funding sits at the 0.01% clamp most of the time. **Do not judge the
   positioning family on generated data.**

## Design (fixed in the diary before any holdout comparison)

- **Calibration** used llm_in_sample only (`market_frame(as_of=2026-07-01)`, rows known before the cutoff; max known_ts 2026-06-30T23:00:02+00:00).
  Universe L: 23 symbols, 2275 days. Universe T: 15 symbols, 516 days, but 13 of them have only 12-80 in-sample days. **T is feasible but weak:**
  it is calibrated mostly on PAXG / XAUT, its SMM objective is 101 (vs 4.7 for L), and no parameter-drift estimate was possible.
- **Daily layer.** Log bipower variance = per-symbol level (trailing-365-day mean; an in-sample check showed it beats the full-history mean for the next 132
  days, MSE 0.31 vs 0.38) + per-symbol weekday effect + a common log-HAR factor (b1/b5/b22 = 0.39, 0.38, 0.13, Student-t
  shocks, nu 7.1) + a pooled idiosyncratic log-HAR with v0-cell shifts. The cell shifts follow latent daily chains from the pooled 9x9
  transition matrix.
- **Hourly layer.** Hour-of-week shares x AR(1) log-vol (common + idiosyncratic; simulated moments give sig_h 0.37, phi
  0.67, psi 0.77) x Student-t returns with a per-cluster common shock (lam ~ 0.90 for crypto; T splits
  into metals / energy / equities clusters). The hours are normalised to the day's bipower variance. The regime-dependent hourly and daily AR(1) terms are
  EB-shrunk and all close to -0.03; drift is otherwise zero.
- **Jumps.** Kou jumps (p 0.135/h in z units) with per-symbol Hawkes excitation (branching about 0 after vol standardisation), a common-jump share
  0.63 matched to the in-sample co-jump phi 0.33, and a GPD tail above 6 sd.
- **Other series.** Wicks from a Brownian bridge (kappa_w 1.27), plus turnover, funding (8h AR) and OI (4h AR).
- **Parameter uncertainty.** 500 moving-block (20-day) bootstrap replicates of every estimator, asymptotic covariances for Kou and Hawkes, and
  in-sample between-window drift of the hour-of-week shares, the cross-asset correlation and the jump intensity. Each path gets one fresh draw.
- **Paths.** 500 param, 200 fixed and 200 boot per universe, starting from the real state at the cutoff (2026-06-30 23:00 to 2026-11-09 23:00, 132 holdout days).
  They run through the unchanged strategy-panel engines (`strategy_specs` / `strategy_fn` / `evaluate.strategy_pnl`, 7.5 bps per side) with a 40-day
  real prefix on 1h and an 80-day prefix on 4h. Family book Sharpe is computed exactly as from `strategy_panel_daily.parquet`. `shock_regime` is
  data-generated (the panel caveat applies).
- **Pre-registered verdict maps** (study docstring):
  - H-GEN1 supports iff coverage >= 0.8 and strictly greater than fixed.
  - H-GEN2 supports iff >= 8/10 facts are inside the envelopes.
  - H-GEN3 supports iff rho > 0.3 and the CI is above 0; it contradicts if rho <= 0.3.

## Results

| generator | covered / pairs | coverage | Wilson 95% | median band width (Sharpe) | strategy-level coverage |
|---|---|---|---|---|---|
| param | 11/12 | 0.92 | [0.65, 0.99] | 5.92 | 29/30 = 0.97 |
| fixed | 11/12 | 0.92 | [0.65, 0.99] | 5.54 | 29/30 = 0.97 |
| boot | 10/12 | 0.83 | [0.55, 0.95] | 4.97 | 26/30 = 0.87 |

| universe | family | realised holdout Sharpe | param p05 / p50 / p95 (q) | fixed p05 / p95 (q) | boot p05 / p95 (q) |
|---|---|---|---|---|---|
| L | calendar | -0.02 | -2.98 / -0.28 / +2.63 (0.56) | -3.18 / +2.49 (0.49) | -3.33 / +1.68 (0.67) |
| L | mean_reversion | -2.09 | -2.75 / -0.21 / +2.79 (0.12) | -2.57 / +2.80 (0.10) | -4.03 / +0.09 (0.49) |
| L | positioning | +0.63 | -6.26 / -3.39 / -0.30 (0.99) **out** | -6.45 / -0.50 (0.99) **out** | -4.27 / +0.43 (0.96) **out** |
| L | shock | -0.75 | -4.48 / -1.28 / +1.42 (0.58) | -3.84 / +1.58 (0.63) | -1.59 / +2.79 (0.14) |
| L | trend | -1.55 | -4.04 / -0.80 / +2.01 (0.32) | -4.13 / +1.69 (0.41) | -2.60 / +2.31 (0.17) |
| L | volatility | -0.55 | -3.89 / -0.71 / +2.47 (0.53) | -3.60 / +2.23 (0.64) | -2.43 / +2.69 (0.35) |
| T | calendar | -2.98 | -3.22 / -0.70 / +2.22 (0.09) | -3.27 / +1.82 (0.09) | -4.64 / +0.83 (0.28) |
| T | mean_reversion | -3.47 | -5.73 / -2.49 / +0.34 (0.32) | -4.98 / +0.12 (0.29) | -4.34 / +0.05 (0.25) |
| T | positioning | -3.24 | -5.83 / -2.96 / -0.02 (0.44) | -5.49 / -0.43 (0.46) | -7.32 / -2.30 (0.85) |
| T | shock | -1.37 | -5.15 / -2.38 / +0.83 (0.69) | -4.99 / +0.43 (0.69) | -4.65 / +0.29 (0.64) |
| T | trend | -2.41 | -4.75 / -1.00 / +2.15 (0.26) | -4.35 / +2.39 (0.23) | -0.49 / +6.00 (0.01) **out** |
| T | volatility | +0.89 | -3.16 / -0.32 / +2.72 (0.76) | -3.68 / +2.41 (0.81) | +0.12 / +6.75 (0.12) |

| fact (universe L, holdout) | realised | generated p05 / p50 / p95 | q | inside |
|---|---|---|---|---|
| kurt_daily | 4.29 | 1.36 / 2.81 / 6.635 | 0.82 | yes |
| kurt_hourly | 34.82 | 7.862 / 12.23 / 28.43 | 0.97 | no |
| acf_abs_1h | 0.1085 | 0.0852 / 0.1183 / 0.173 | 0.35 | yes |
| acf_abs_1d | 0.02182 | -0.002092 / 0.02753 / 0.09773 | 0.41 | yes |
| weekday_profile | distance 0.271 | gen p95 distance 0.517 (pointwise inside 0.86) | - | yes |
| how_profile | distance 34.890 | gen p95 distance 50.782 (pointwise inside 0.92) | - | yes |
| ret_factor_share | 0.614 | 0.6273 / 0.6944 / 0.7472 | 0.03 | no |
| vol_factor_share | 0.6452 | 0.6038 / 0.6847 / 0.7807 | 0.24 | yes |
| jump_freq | 0.005627 | 0.004336 / 0.005435 / 0.006727 | 0.59 | yes |
| regime_median_hours | 9 | 9 / 9 / 10 | 0.01 | yes |
| **total** | | | | **8/10** |
| (fixed generator) | | | | 8/10 |
| (boot generator) | | | | 10/10 |

| ranking statistic | Spearman | 95% CI | pairs |
|---|---|---|---|
| family x universe, param median vs realised | +0.24 | [-0.48, +0.83] | 12 |
| strategy x universe (secondary) | +0.09 | [-0.33, +0.46] | 30 |
| family x universe, fixed median vs realised | +0.22 | [-0.52, +0.79] | 12 |
| family x universe, boot median vs realised | +0.64 | [+0.01, +0.93] | 12 |

## What the generator still misses

- **Funding and open-interest microstructure.** Real funding sits at the 0.01% clamp most of the time and moves in rare bursts. Generated funding is a
  continuous AR process. As a result, synthetic positioning P&L is badly biased (point 4).
- **Return drift and the direction of trend.** The model is a martingale by design (BELIEFS P8). The realised ordering of families (trend and
  mean reversion lost, positioning won) is mostly path luck that no zero-drift generator can rank.
- **Tail depth and correlation regimes.** Hourly kurtosis above about 30 and periods where cross-asset correlation falls (the factor share dropped to 0.61)
  sit outside the envelopes. The return correlation is not tied to the vol state.
- **T.** Too little in-sample history; the equity-perp session structure comes mostly from 5-6 weeks of data shrunk toward gold.

## Rule candidates

```yaml
- id: "R-GEN-1"
  kind: "avoid"
  rule: "WHEN judging the positioning family (funding_contrarian, oi_price_divergence) THEN do not use generated paths; use real funding / OI history only"
  when: "evaluating or ranking positioning strategies"
  then: "exclude synthetic funding / OI backtests from the evidence"
  scope: "universe L (and T funding_contrarian)"
  effect: "synthetic median Sharpe -3.4 vs realised +0.63; every generator rejects the realised value (q 0.96-0.99)"
  confidence: "B"
  evidence: ["H-GEN1", "H-GEN3"]
  review: "after a funding generator with the clamp / burst structure"
  status: "candidate"
- id: "R-GEN-2"
  kind: "expect"
  rule: "WHEN a deterministic strategy's 132-day Sharpe is read THEN expect a 5-95% range about +/-3 Sharpe wide from path noise alone"
  when: "reading any 4-5 month holdout or forward Sharpe of a single family book"
  then: "treat |Sharpe| < 3 over 132 days as uninformative about the edge"
  scope: "families x universes L, T"
  effect: "median band width 5.9 (param), 5.5 (fixed), 5.0 (bootstrap); 11/12 realised values inside"
  confidence: "B"
  evidence: ["H-GEN1"]
  review: "2027-01 with a longer holdout"
  status: "candidate"
- id: "R-GEN-3"
  kind: "avoid"
  rule: "WHEN choosing between strategy families THEN do not rank them by median Sharpe on parametric generated paths"
  when: "strategy selection or allocation"
  then: "use real-data evidence; generated paths serve only as stress / variance envelopes"
  scope: "families x universes L, T"
  effect: "Spearman +0.24 (CI -0.48..+0.83) family level, +0.09 strategy level"
  confidence: "B"
  evidence: ["H-GEN3"]
  review: "if a generator with estimated drift or flow structure is built"
  status: "candidate"
- id: "R-GEN-4"
  kind: "prefer"
  rule: "WHEN a synthetic stress test of daily / hourly risk statistics is needed THEN prefer the parametric generator (it reproduces 8 of 10 held-out stylised facts and creates unseen states), but widen hourly tails and allow correlation drops"
  when: "stress-testing vol, clustering, weekday / hour-of-week risk, jump frequency"
  then: "use sim.market_gen param paths; add a separate tail / decorrelation scenario"
  scope: "universe L"
  effect: "8/10 facts inside; misses hourly kurtosis (34.8 vs p95 28.4) and return factor share (0.61 vs p05 0.63)"
  confidence: "C"
  evidence: ["H-GEN2"]
  review: "next holdout window"
  status: "candidate"
- id: "R-GEN-5"
  kind: "expect"
  rule: "WHEN parameter uncertainty is added to a market generator THEN expect little extra Sharpe dispersion over a few months; path noise dominates"
  when: "designing robustness checks on generated data"
  then: "do not count parameter-uncertain coverage as extra evidence beyond a fixed-parameter generator"
  scope: "132-day windows, L and T"
  effect: "band width +7%; coverage identical (11/12 vs 11/12)"
  confidence: "C"
  evidence: ["H-GEN1"]
  review: "with longer horizons"
  status: "candidate"
```
