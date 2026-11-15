# Trend / breakout / state-space block: H-BO1, H-MC1, H-KF1, H-KF2 (agent A6, 2026-11-10)

Code: `src/ai_trader/research/breakout_cipher_kalman_study.py` (the module docstring is the full pre-run spec),
`src/ai_trader/features/state_space.py` (LLT Kalman filter, steady-state gain, concentrated diffuse MLE),
`src/ai_trader/features/cipher.py` (WaveTrend, money flow, dots, trades). Tests: `tests/test_state_space.py`,
`tests/test_cipher.py` (14 pass; they cover slope recovery on simulated data, MLE ratio recovery, and causality of the
filter, WaveTrend, MF and trades under truncation and future-bar perturbation). Raw numbers:
`research/reports/trend_state_space_2026-11-10.json`. Results are recorded in `research/hypotheses_results.jsonl`
(8 rows).

Run: `uv run python -m ai_trader.research.breakout_cipher_kalman_study --workers 4`, as of 2026-11-10 ~22:50 UTC.

## Data and conventions

- Bybit last-price bars via `market_frame(funding=False, oi=False)`. Signals and events are computed on the full
  history, and results are split by decision date at 2026-07-01. Costs are 7.5 bps per side (taker 5.5 + slippage 2).
- **Universe L:** the 23 crypto perps of `trend_zoo_study.UNIVERSE`.
- **Universe T:** 15 TradFi perps. SPCXUSDT was dropped by the pre-set ">= 120 days" rule (118 days after burn-in).
  T starts Mar-Jun 2026, so its `llm_in_sample` slice is only ~2-3 months.
- The 10 equity perps have native 4h bars only from 2026-11-04. For them, 4h bars are aggregated from complete
  UTC-aligned groups of four 1h bars, with `known_ts` = the max of the four. This was validated on BTC against native
  4h: identical highs, sigma difference < 3e-4. The first run silently dropped these symbols. Nothing was recorded
  from it, and it is logged as a diary "mistake".
- **Holm:** across the block's 12 primary tests per segment. A "t > 2" requirement needs Holm p < 0.05.
- Two of these hypotheses are registered as NULLS:
  - H-BO1: pivots are not special;
  - H-MC1: Cipher dots have no edge.
  For them, `supports` means "no edge found".

## Verdicts

| hypothesis | llm_in_sample | clean_holdout | overall reading |
|---|---|---|---|
| H-BO1 (pivots not special, null) | **supports** | **supports** | R1/S1 breakouts behave like any level at the same sigma distance. Neither shows continuation: gross ~0, net negative. |
| H-MC1 (Cipher dots no edge, null) | **supports** | **supports** | All 4 variants have DSR ~0. The best in-sample variant (4h, MF on) loses in the holdout (-0.60 R-units per trade, t -2.8). |
| H-KF1 (adaptive-gain Kalman beats fixed) | **contradicts** | supports (sign only) | Adaptive is better in all 3 q in-sample, but t 1.5-1.8, below the bar (Holm p 0.6-0.9). The holdout sign is positive at q=1e-3 (t 1.3). **Not supported**. |
| H-KF2 (LLT q_hat predicts trend P&L) | **contradicts** | **contradicts** | Mean IC is -0.02 (t -0.6) in-sample and -0.15 over 4 holdout months. q_hat sits at its zero bound in 65% of fits. |

## H-BO1: floor-pivot breakouts vs placebo levels

Event: the first 1h close beyond R1 (long) or S1 (short) of the prior UTC day's floor pivots. The return is signed,
net of 15 bps round trip, in units of sigma_1h * sqrt(h).

Placebo: one per real event, with the same symbol, segment and side. It uses the same distance k from the prior
close, measured in the 1h sigma known at that close, but is re-anchored on a random other day. Every real event got
a placebo except 1. The SE is clustered by event date.

| univ | segment | h | n events | dates | pivot net (t) | pivot gross | placebo net | **pivot - placebo (t)** | k median |
|---|---|---|---|---|---|---|---|---|---|
| L | llm_in_sample | 4h | 26272 | 2133 | -0.093 (-5.1) | +0.019 | -0.098 | +0.005 (0.48) | 3.35 |
| L | llm_in_sample | 12h | 26272 | 2133 | -0.033 (-1.8) | +0.032 | -0.048 | +0.015 (1.51) | 3.35 |
| L | clean_holdout | 4h | 2024 | 132 | -0.194 (-3.2) | -0.046 | -0.148 | -0.046 (-1.27) | 3.31 |
| L | clean_holdout | 12h | 2024 | 132 | -0.089 (-1.2) | -0.004 | -0.029 | -0.060 (-1.78) | 3.31 |
| T | llm_in_sample | 4h | 1013 | 351 | -0.275 (-3.9) | +0.081 | -0.264 | -0.011 (-0.20) | 3.12 |
| T | llm_in_sample | 12h | 1013 | 351 | -0.090 (-1.3) | +0.115 | -0.118 | +0.027 (0.48) | 3.12 |
| T | clean_holdout | 4h | 1417 | 120 | -0.230 (-3.5) | +0.056 | -0.266 | +0.036 (0.60) | 3.07 |
| T | clean_holdout | 12h | 1417 | 120 | -0.060 (-0.9) | +0.105 | -0.135 | +0.074 (1.44) | 3.07 |

- **Every |t| < 2**, so the null holds in both segments and both universes. All Holm-adjusted p are >= 0.6.
- **Breakouts don't continue.** By the time the first 1h close is beyond R1 or S1, price has already moved ~3.3
  sigma_1h from the prior close. Gross continuation is ~0 (+0.02 sigma at 4h for L in-sample), so costs (~0.1 sigma for
  crypto, ~0.3 sigma for TradFi) make it negative. The hit rate is 0.36-0.45. This matches P4: levels are neither
  barriers nor launch pads.
- The mirrored diagnostic (same day, same distance, opposite side) shows pivot minus mirror of +0.5 to +1.0 sigma
  (t 5-21). **This is a selection artefact, not a pivot effect.** The mirror exists only on days that also had a real
  event, i.e. wide two-sided days, and on those days the second extreme mean-reverts. It is reported for completeness
  and not used.

## H-MC1: Market Cipher B dots (WaveTrend 9/12/4, +/-53, VuManChu MF proxy)

Trades enter at the dot's close and exit at the next opposite cross or after 12 bars. Returns are net trade returns
in risk units (/ sigma_next at entry). The random baseline uses 1000 draws with the same symbol, segment, side and
holding period.

Universe L:

| segment | variant | trades | mean net (NW t) | gross | hit | random mean [5%, 95%] | p(random >= real) | daily port Sharpe | DSR (N=4) |
|---|---|---|---|---|---|---|---|---|---|
| llm_in_sample | 1h MF on | 9208 | -0.114 (-2.14) | +0.114 | 0.36 | -0.221 [-0.276, -0.166] | 0.001 | -0.84 | 0.000 |
| llm_in_sample | 1h MF off | 42123 | -0.296 (-9.68) | -0.068 | 0.36 | -0.238 [-0.264, -0.212] | 1.000 | -3.32 | 0.000 |
| llm_in_sample | 4h MF on | 2399 | +0.038 (0.34) | +0.147 | 0.39 | -0.091 [-0.202, 0.016] | 0.022 | 0.26 | 0.000 |
| llm_in_sample | 4h MF off | 10489 | -0.122 (-2.01) | -0.015 | 0.39 | -0.118 [-0.170, -0.065] | 0.534 | -0.53 | 0.000 |
| clean_holdout | 1h MF on | 726 | -0.494 (-3.23) | -0.193 | 0.31 | -0.317 [-0.508, -0.126] | 0.934 | -4.05 | 0.000 |
| clean_holdout | 1h MF off | 3243 | -0.435 (-4.90) | -0.149 | 0.34 | -0.309 [-0.403, -0.221] | 0.990 | -5.46 | 0.000 |
| clean_holdout | 4h MF on | 208 | -0.601 (-2.82) | -0.454 | 0.33 | -0.199 [-0.551, 0.179] | 0.967 | -3.65 | 0.000 |
| clean_holdout | 4h MF off | 777 | -0.156 (-0.83) | -0.017 | 0.37 | -0.157 [-0.356, 0.042] | 0.495 | -1.00 | 0.044 |

Universe T (robustness; holdout is what counts) in clean_holdout:

| variant | trades | mean net (t) | p(random >= real) |
|---|---|---|---|
| 1h MF on | 482 | -0.78 (-4.4) | 0.94 |
| 1h MF off | 2072 | -0.72 (-7.7) | 0.99 |
| 4h MF on | 133 | -0.14 (-0.3) | 0.40 |
| 4h MF off | 480 | +0.01 (0.0) | 0.05 |

- The DSR threshold (expected max Sharpe over 4 trials) is 1.6 per year in-sample. No variant comes close, and the
  best daily Sharpe is +0.26.
- The MF filter did help in-sample: 1h MF on beat its random entries (p = 0.001), and 4h MF on was the best variant.
  **That advantage reversed in the holdout:** -0.60 per trade, worse than random 97% of the time.
- 1h dots are cost-dominated. Unfiltered 1h dots are also worse than random in both segments.
- Matching P2, the oscillator's mean-reversion entries have no net edge here.

## H-KF1: vol-adaptive vs fixed-gain local-linear-trend slope (4h, universe L)

Model: a smooth trend (level noise 0) with q = Var(zeta)/Var(eps).

- **Adaptive:** R_t = sigma_risk_next(t-1)^2 with Q = q * Rref_t, where Rref_t is the trailing 180-day median of R.
  The steady-state gain is identical to the constant filter whenever R_t = Rref_t.
- **Constant:** R = 1, Q = q.
- Steady-state gains (level, slope):

  | q | level gain | slope gain |
  |---|---|---|
  | 1e-4 | 0.13 | 0.009 |
  | 1e-3 | 0.22 | 0.028 |
  | 1e-2 | 0.36 | 0.080 |

- Signal: tanh(slope / trailing-180d sd).
- Correlation between the adaptive and constant signals is 0.955-0.959.

| segment | q | days | adaptive - constant, daily (NW t) | Sharpe adaptive / constant (mean over symbols) | portfolio Sharpe a / c | share of symbols adaptive better |
|---|---|---|---|---|---|---|
| llm_in_sample | 1e-4 | 2094 | +0.0124 (1.83) | 0.39 / 0.29 | 0.89 / 0.73 | 0.83 |
| llm_in_sample | **1e-3** | 2094 | **+0.0106 (1.49)** | 0.41 / 0.32 | 0.76 / 0.61 | 0.78 |
| llm_in_sample | 1e-2 | 2094 | +0.0107 (1.69) | 0.16 / 0.05 | 0.29 / 0.11 | 0.91 |
| clean_holdout | 1e-4 | 133 | -0.0004 (-0.02) | -0.00 / -0.01 | 0.11 / 0.12 | 0.48 |
| clean_holdout | **1e-3** | 133 | **+0.0218 (1.26)** | -0.09 / -0.33 | 0.04 / -0.29 | 0.83 |
| clean_holdout | 1e-2 | 133 | +0.0128 (0.78) | -0.37 / -0.47 | -0.20 / -0.40 | 0.61 |

- The registered test fails: t 1.49 < 2 at the median q, and the Holm p is 0.92.
- The direction is consistent, though: positive in 5 of 6 cells, and in-sample 78-91% of symbols are better adaptive.
- Holdout q=1e-3 is positive, which gives the "supports" row (sign only).
- Read: shrinking the trend filter's gain when vol is above its median is a small, unproven improvement, worth
  ~+0.1 Sharpe on a sleeve that is itself thin. Both versions are negative in the holdout at q >= 1e-3, as P3/P7 say
  for trend.

## H-KF2: rolling LLT signal-to-noise q_hat vs next-30-day trend-ensemble P&L

Fits: 1272 monthly 180-day MLE fits (full LLT, concentrated diffuse likelihood) on 23 symbols, from 2021 to Oct 2026.
q_hat = Var(zeta)/Var(eta).

- **The q_hat median is at the lower bound** (exp(-16)) and 65% of fits sit there; the 90th percentile is 1.8e-5.
  This is the classic pile-up of a small state variance's MLE at zero: 180 days of 4h prices can't tell a stochastic
  slope from a random walk.

| segment | months | mean Spearman IC | NW t | share of months IC > 0 | pooled Spearman |
|---|---|---|---|---|---|
| llm_in_sample | 55 | -0.023 | -0.63 | 0.38 | -0.066 |
| clean_holdout | 4 | -0.150 | n/a (n < 10; plain t -1.85) | 0.25 | -0.162 |

- The holdout monthly ICs are +0.05, -0.33, -0.21 and -0.11.
- The sign is, if anything, negative: symbols with an estimated stochastic slope did *worse* with trend afterwards.
  This is H-TR4's state-space twin, and it fails the same way.

## Holm table (primary tests)

| segment | test | t | p | p Holm |
|---|---|---|---|---|
| llm_in_sample | BO1 L 4h / 12h | 0.48 / 1.51 | 0.63 / 0.13 | 1.0 / 0.92 |
| llm_in_sample | BO1 T 4h / 12h | -0.20 / 0.48 | 0.84 / 0.63 | 1.0 / 1.0 |
| llm_in_sample | MC1 1h on / 1h off / 4h on / 4h off | -2.14 / -9.68 / 0.34 / -2.01 | | 0.36 / <1e-4 / 1.0 / 0.44 |
| llm_in_sample | KF1 q 1e-4 / 1e-3 / 1e-2 | 1.83 / 1.49 / 1.69 | | 0.60 / 0.92 / 0.73 |
| llm_in_sample | KF2 IC | -0.63 | 0.53 | 1.0 |
| clean_holdout | BO1 L 4h / 12h | -1.27 / -1.78 | | 1.0 / 0.61 |
| clean_holdout | BO1 T 4h / 12h | 0.60 / 1.44 | | 1.0 / 1.0 |
| clean_holdout | MC1 1h on / 1h off / 4h on / 4h off | -3.23 / -4.90 / -2.82 / -0.83 | | 0.013 / <1e-4 / 0.043 / 1.0 |
| clean_holdout | KF1 q 1e-4 / 1e-3 / 1e-2 | -0.02 / 1.26 / 0.78 | | 1.0 / 1.0 / 1.0 |

## Caveats

- **The TradFi sample is short.** There are 15 symbols, about 120 holdout dates, an in-sample slice of only ~2-3
  months, and 4h bars aggregated from 1h for the equities. Treat T results as confidence C.
- **The holdouts are small.**
  - H-KF2 has 4 holdout months, so no NW t.
  - H-KF1 has 133 holdout days.
  - H-MC1's 4h MF-on holdout has 208 trades.
- **The placebo is a modelling choice**, fixed before the run: the same sigma distance on a random other day, from the
  same symbol and segment. It controls for "moved k sigma from the prior close" but not for intraday timing.
  Pivot-event hours and placebo hours can differ (both are "first crossing").
- **The R-unit mapping is rough.** BO1 returns are in sigma*sqrt(h); MC1 trades are in risk units of one entry-bar
  sigma, not stop R.
- **The 1h variants pay ~0.15 risk units per round trip.** Any 1h oscillator result is mostly a cost statement.
- **The KF1 adaptive/constant contrast depends on the matching rule** (trailing-median reference variance). Another
  reference, such as the EWMA of R, could change the size of the effect but likely not the conclusion, since the
  signal correlation is 0.96.
- **Survivorship:** universe L is today's survivors.
- **Contamination:** Cipher and pivots are retail lore and Kalman trend is textbook, so in-sample results are weak
  evidence either way. H-MC1's in-sample "best variant" failing out of sample is the most informative single fact here.

## Rule candidates

```yaml
- id: R-TS-1
  rule: "WHEN price makes its first 1h close beyond the prior UTC day's floor pivot R1 or S1 THEN don't expect continuation beyond a generic k-sigma move, and don't treat R1/S1 as support or resistance"
  when: "first 1h close beyond R1/S1 (P=(H+L+C)/3 of the prior UTC day); typical distance ~3.3 sigma_1h from the prior close"
  then: "expect ~0 gross signed drift over the next 4-12h (net of taker costs it is negative); pivots carry no information beyond the sigma distance; don't enter breakouts or fades on pivots alone"
  scope: {assets: all, horizon: "4h-12h", sides: both}
  effect: "pivot minus placebo +0.005 sigma (t 0.5) at 4h in-sample, 26k events; holdout -0.05 (t -1.3) crypto, +0.04 (t 0.6) TradFi; pivot gross +0.02 sigma at 4h, net -0.09 (t -5.1)"
  confidence: A
  evidence: [H-BO1, research/reports/trend_state_space_2026-11-10.md]
  review: "a pivot-minus-placebo t > 2 in the growing holdout (TradFi n doubles), or a registered session-timing variant"
- id: R-TS-2
  rule: "WHEN a WaveTrend / Market Cipher B green or red dot prints (with or without the money-flow filter) THEN don't expect a positive net trade; don't use Cipher dots as entry or timing signals"
  when: "LazyBear WaveTrend(9,12,4) cross with wt2 <= -53 (green) or >= 53 (red), 1h or 4h bars"
  then: "expect net trade return <= 0; the 1h variants lose more than random entries with the same side and holding"
  scope: {assets: all, horizon: "12h-2d (12 bars of 1h or 4h)", sides: both}
  effect: "in-sample DSR 0.00 for all 4 variants (best daily Sharpe 0.26 vs SR0 1.6); best in-sample variant (4h + MF) holdout -0.60 risk units per trade (t -2.8, worse than random p 0.97); 1h unfiltered -0.30 (t -9.7) in-sample, -0.44 (t -4.9) holdout"
  confidence: A
  evidence: [H-MC1, research/reports/trend_state_space_2026-11-10.md]
  review: "any variant with holdout DSR > 0.95 or a positive holdout mean with t > 2 after n >= 500 trades"
- id: R-TS-3
  rule: "WHEN reading a Kalman / local-linear-trend slope THEN prefer the vol-adaptive gain (measurement variance = robust sigma^2) over a fixed gain only as a tie-breaker; don't expect it to turn trend into an edge"
  when: "4h trend sleeve built from an LLT slope (q 1e-4..1e-2); adaptive R_t = sigma_risk_next^2, Q scaled to the trailing-median variance"
  then: "expect at most ~+0.1 annual Sharpe vs the fixed-gain filter; both are negative in Jul-Nov 2026 at q >= 1e-3, consistent with P3/P7"
  scope: {assets: crypto, horizon: "1d", sides: both}
  effect: "paired daily diff +0.011 risk units (t 1.5) in-sample at q=1e-3, 78% of symbols better; holdout +0.022 (t 1.3); signal correlation 0.96"
  confidence: C
  evidence: [H-KF1, research/reports/trend_state_space_2026-11-10.md]
  review: "re-test after 12 holdout months; becomes B if the holdout diff has t > 2"
- id: R-TS-4
  rule: "WHEN asking 'is this market trending enough for trend-following?' THEN don't use a rolling MLE signal-to-noise ratio (LLT slope variance) to answer it"
  when: "q_hat = var(slope noise)/var(level noise) from a 180-day 4h LLT fit, ranked across symbols monthly"
  then: "expect no cross-sectional information about the next 30 days of trend-ensemble P&L (if anything a negative sign); q_hat is at its zero bound in ~2/3 of fits"
  scope: {assets: crypto, horizon: "30d", sides: both}
  effect: "mean Spearman IC -0.02 (t -0.6, 55 months) in-sample, -0.15 over 4 holdout months; 65% of fits at q_hat = 0"
  confidence: B
  evidence: [H-KF2, research/reports/trend_state_space_2026-11-10.md]
  review: "a registered variant with a longer window or daily bars (where q_hat is identifiable) showing IC > 0 with t > 2"
```
