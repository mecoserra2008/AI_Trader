# Decision budget: inputs, history and latency (H-IC1, H-IC2, H-IC3), 2026-11-10

Agent A7, program 2026-11-10. Code: `src/ai_trader/research/decision_budget_study.py`. Tests: `tests/test_decision_budget.py`.
Full numbers: `research/reports/decision_budget_2026-11-10.json`. H-IC4, the forward-only LLM A/B, is **not** run here
(principle 1). This study covers only the deterministic question underneath it: how fast information saturates.

## 1. Methods (fixed before testing; diary entry "H-IC1 feature list fixed before testing")

- **Universe and sample.** Universe L: 23 crypto perps on Bybit. One decision per symbol per UTC day, at 00:00 UTC
  (the close of the last 1h bar). 38,424 symbol-days have all 32 features and the target: 35,388 `llm_in_sample` and
  3,036 `clean_holdout`. Out-of-sample predictions start in 2021-11, giving 34,434 in-sample OOS rows over 1,703 dates
  and 3,036 holdout rows over 132 dates.
- **Target.** y = 1{next-24h log return > 0}. This is the sign of the vol-scaled return.
- **The 32 features, ordered simple to complex.** Model k uses features 1..k.
  - 1-4: past returns in σ units over 1d, 3d, 7d and 30d.
  - 5-11: volatility: log σ, RV surprise, RV term ratio, vol percentile over 1y, semivariance ratio, jump share, vol of vol.
  - 12-20: regime features (efficiency ratio, trend t-stat, directional-vol ratio, VR, autocorrelations).
  - 21-24: funding level and z-score, OI change over 1d and 7d.
  - 25-26: calendar (next day is a weekend, next day is a Monday).
  - 27-30: range position over 30d, 90d drawdown in σ units, day range in σ units, BTC's 1d return.
  - 31-32: two interactions.
  - Every input has event_ts < T. Funding and OI count only if known_ts ≤ T.
- **Walk-forward.** Refit monthly. Training labels must end ≤ test start − 1 day: a 24h purge plus a 1-day embargo,
  checked by a unit test. The first test month is the first one with ≥ 365 training dates. Features are standardised on
  the training sample. The model is a ridge logistic (λ = 1, Newton).
- **Brier skill and CIs.** Brier skill is measured against the base rate of each refit's training sample. CIs come
  from a moving-block bootstrap over dates (blocks of 10 days, 1,000 draws).
- **H-IC2.** Uses k = 1, the in-sample argmax from H-IC1. Training windows are 90, 180, 365 and 730 days plus
  expanding, applied to two models:
  - the logistic model, scored on Brier loss;
  - a pooled panel log-HAR (log RV_d, log mean RV over 5d, log mean RV over 22d), scored on QLIKE for the next day's RV.

  Both are compared on the same rows across windows, using paired per-date loss differences vs expanding with NW t.
- **H-IC3.**
  - `shock_regime` runs as configured on 1h bars.
  - The 4h trend-zoo ensemble is computed on 4h bars and applied to the 1h return path. The position held over the 1h
    bar after close c is the latest 4h signal available (4h bar close) at or before c − d.
  - d ∈ {0, 1, 2, 4, 8} h. P&L follows `evaluate.strategy_pnl`: risk units, net of 7.5 bps per side.
  - edge(d) is the mean over dates of the cross-symbol mean daily P&L, and R(d) = edge(d)/edge(0).
  - The test statistic is g(d) = edge(d) − 0.5·edge(0), with a 10-day block bootstrap CI.
- **Multiple testing.** Holm across the three hypotheses within each segment.

## 2. H-IC1: OOS Brier skill vs number of features k

| k | in-sample skill | 95% CI | holdout skill | 95% CI |
|---|---|---|---|---|
| 1 | **+0.0004** | [−0.0036, +0.0045] | +0.0026 | [−0.0031, +0.0083] |
| 2 | +0.0001 | [−0.0044, +0.0044] | +0.0031 | [−0.0041, +0.0119] |
| 4 | −0.0008 | [−0.0055, +0.0035] | +0.0029 | [−0.0054, +0.0117] |
| 8 | −0.0034 | [−0.0089, +0.0022] | **+0.0069** | [−0.0010, +0.0144] |
| 16 | −0.0068 | [−0.0133, −0.0004] | +0.0041 | [−0.0062, +0.0117] |
| 32 | −0.0147 | [−0.0229, −0.0068] | +0.0024 | [−0.0090, +0.0121] |

The comparison that decides the verdict is Δ = skill(32) − max skill over k ≤ 8:

| segment | Δ | 95% CI |
|---|---|---|
| in-sample | −0.0152 | [−0.022, −0.009] |
| holdout | −0.0045 | [−0.013, +0.004] |

Accuracy is 0.51-0.54 everywhere.

**Reading.** No feature count produces skill distinguishable from zero for next-day direction. Adding features
monotonically *destroys* skill in-sample. The 16- and 32-feature models are significantly worse than the base rate.
In the holdout, k = 8 is the (insignificant) best and every CI contains 0.

## 3. H-IC2: training-window length

Mean OOS loss per window; t is the NW t of the paired difference vs expanding (negative = better than expanding):

| window | logit Brier (IS) | t | HAR QLIKE (IS) | t | logit Brier (HO) | t | HAR QLIKE (HO) | t |
|---|---|---|---|---|---|---|---|---|
| 90d | 0.25122 | +1.55 | 0.3964 | **+3.10** | 0.25225 | +1.90 | 0.4181 | +1.17 |
| 180d | 0.25018 | +0.28 | 0.3872 | +1.60 | 0.25221 | +1.74 | 0.4211 | +1.14 |
| 365d | **0.24944** | −1.72 | 0.3834 | +0.48 | 0.25266 | +1.74 | 0.4100 | +0.74 |
| 730d | 0.24979 | −1.26 | **0.3827** | −0.45 | 0.25109 | +1.15 | 0.4061 | +0.71 |
| expanding | 0.25005 | - | 0.3829 | - | **0.24963** | - | **0.4052** | - |

**Reading.**
- No shorter window beats expanding with t < −2 for either model in either segment.
- Short windows hurt: 90d HAR is +3.4% QLIKE, t +3.1.
- Gains flatten beyond about one year (logit) or two years (HAR). The 730d window and expanding are statistically tied.
- More history never hurt significantly, but past one to two years it stops helping.

## 4. H-IC3: latency decay

Edge in risk units per day for the cross-symbol mean; R = edge(d)/edge(0):

| d (h) | trend IS edge | R [95% CI] | shock IS edge | R [95% CI] | trend HO edge | shock HO edge |
|---|---|---|---|---|---|---|
| 0 | 0.1116 (t 2.40) | 1 | 0.0104 (t 1.05) | 1 | −0.0384 | −0.0180 |
| 1 | 0.1050 | 0.94 [0.74, 1.01] | 0.0063 | 0.61 [−2.8, 2.2] | −0.0399 | −0.0169 |
| 2 | 0.1112 | 1.00 [0.90, 1.11] | 0.0010 | **0.10** [−8.3, 1.3] | −0.0292 | +0.0062 |
| 4 | 0.1083 | **0.97** [0.79, 1.09] | 0.0003 | 0.03 [−8.1, 1.4] | −0.0152 | +0.0109 |
| 8 | 0.0979 | 0.88 [0.49, 1.05] | −0.0057 | −0.55 [−17, 0.9] | −0.0283 | +0.0117 |

**Trend ensemble.**
- Half-life > 8 h. g(4) CI is [+0.009, +0.100], so the trend leg holds: more than half the edge survives a 4h delay.
- The annualised Sharpe of the cross-symbol mean is 1.00 at d = 0 and 0.93 at d = 8.

**shock_regime.**
- The point half-life is about 1.2 h, and 90% of the in-sample edge is gone at d = 2.
- But edge(0) itself is not significant (t 1.05; 15% of bootstrap draws have edge(0) ≤ 0). The g(2) CI is
  [−0.016, +0.008] and includes 0, so the shock leg is inconclusive.

**Holdout.** Both edges at d = 0 are negative, so the ratios are undefined and both legs are inconclusive. The trend
result is consistent with P7.

## 5. Verdicts (recorded in `research/hypotheses_results.jsonl`)

| hyp | llm_in_sample | clean_holdout | pooled | key numbers |
|---|---|---|---|---|
| H-IC1 | **supports** (Holm p 0.003) | inconclusive (all CIs contain 0) | inconclusive | argmax k = 1 IS, k = 8 HO; Δ(32 vs ≤8) −0.015 IS, −0.0045 HO |
| H-IC2 | inconclusive | inconclusive | inconclusive | no shorter-window win with t < −2; best IS window 365d (logit) and 730d (HAR); 90d HAR worse, t +3.1 |
| H-IC3 | inconclusive | inconclusive | inconclusive | trend R(4) 0.97 [0.79, 1.09], half-life > 8h (leg holds); shock R(2) 0.10 but CI includes 0.5 (leg undecided); holdout edges ≤ 0 |

The H-IC1 in-sample "supports" applies the registered rule literally: not every CI contains 0, because the large-k
CIs are significantly *negative*. It means "saturation at k ≤ 8" only in the sense that **no k carries directional
skill and more inputs make it worse**.

## 6. What this means for the LLM decision layer

1. **Inputs.** For next-day direction, 32 standard state features hold no usable information. Adding them to a fitted
   model costs about 1.5% Brier skill, and the 16- and 32-feature models are significantly worse than the base rate.
   - A reader that weighs many weak cues, LLM or logistic, should be expected to overfit noise in the same way.
   - So show a short card: about 8 lines or fewer, built from the tested signals with known effects (vol forecasts,
     shock context, regime cell), not a dump of every tool output.
   - The card is for risk and context, not for calling direction.
   - Whether this transfers to the LLM is exactly H-IC4, which is forward-only.
2. **History.** For estimated parameters, 1-2 years of daily history is enough. Short windows (≤ 6 months) are
   significantly worse for vol forecasting, and nothing is gained by going longer.
   - For the LLM: summaries of the recent 90 days should not override parameters estimated on 1-2 years.
   - The card should show long-window estimates with the recent state as a deviation from them, not recent-only
     statistics.
3. **Deliberation time.** It depends on the signal type.
   - **Slow 4h trend signals:** decay is negligible. 97% of the edge remains after 4 h, and 88% (CI down to 49%) after
     8 h. Hours of deliberation, or waiting for the next session, cost little.
   - **Shock signals:** point estimates say most of the edge is gone within 1-2 h. That is not yet significant, but it
     is consistent with V4 (jump half-life 2-3 h).
   - Shock-type candidates should therefore be executed mechanically by the deterministic layer at the bar close. The
     LLM can veto them but should not hold them for review.

## 7. Caveats

- **The direction target is close to unpredictable** (base rate ≈ 0.49-0.51). H-IC1 measures "more features add
  noise", not "where real information saturates". A vol target would have a non-flat skill curve, but it was not
  registered.
- **The model is linear.** Ridge λ = 1 is almost no shrinkage at n ≈ 30k. Stronger ridge or Lasso could flatten the
  large-k penalty, but they were not registered and were not tuned.
- **Short holdout.** 132 dates, so every holdout leg is low-powered. The trend and shock edges at d = 0 are negative
  in the holdout, which makes the latency ratios undefined there.
- **shock_regime was designed after seeing its in-sample evidence** (strategy docstring), so its in-sample edge is
  contaminated as well as weak (t 1.05).
- **The trend ensemble here is the 4h signal traded on the 1h path with 1h-σ sizing at entry.** The risk-unit scale
  differs from the native 4h panel. The ratio is scale-free; absolute edges are not comparable with
  trend_zoo_2026-11-08.
- **The universe is today's survivors.** Decisions are taken at 00:00 UTC only. The study assumes execution at the bar
  close, with no queueing or slippage beyond 7.5 bps.

## Rule candidates

```yaml
- id: R-IC-1
  rule: "WHEN judging next-day direction from the standard state features THEN don't expect any of them (or more of them) to help; prefer a short card and never add inputs to make a directional call"
  when: "daily decision, crypto universe L, features = returns in sigma, vol state, regime features, funding/OI, calendar, range"
  then: "treat next-24h direction as ~50/50 from these features; show <= 8 tested context lines; more inputs lowered OOS skill"
  scope: {assets: crypto, horizon: "1d", sides: both}
  effect: "Brier skill k=1 +0.0004 [-0.004, +0.005], k=32 -0.015 [-0.023, -0.007] in-sample; holdout all CIs contain 0 (best k=8 +0.007)"
  confidence: B
  evidence: [H-IC1, research/reports/decision_budget_2026-11-10.md]
  review: "H-IC4 forward A/B (compact vs full card); a holdout k with CI > 0; a registered vol-target version of H-IC1"
- id: R-IC-2
  rule: "WHEN estimating model parameters (vol HAR, classifiers) THEN prefer 1-2 years or expanding history over recent-only windows; don't let a 90-day summary override them"
  when: "any fitted tool parameter; daily data; crypto"
  then: "use 730d-or-expanding fits; read recent state as a deviation from the long-window estimate"
  scope: {assets: crypto, horizon: "1d", sides: both}
  effect: "panel HAR QLIKE 90d window +0.0129 vs expanding (t 3.1) in-sample; 730d vs expanding -0.0002 (t -0.4); logit 365d -0.0006 (t -1.7)"
  confidence: B
  evidence: [H-IC2, research/reports/decision_budget_2026-11-10.md]
  review: "a shorter window winning with t < -2 in the holdout as it grows; a regime break making expanding fits worse"
- id: R-IC-3
  rule: "WHEN the candidate is a slow (4h) trend signal THEN deliberation of several hours costs little edge; there is no need to rush execution"
  when: "trend_zoo ensemble / 4h trend candidates"
  then: "latency up to 4-8h is acceptable; expect >= ~90% of the edge to remain at 4h"
  scope: {assets: crypto, horizon: "1d", sides: both}
  effect: "edge ratio R(4h) 0.97 [0.79, 1.09], R(8h) 0.88 [0.49, 1.05], half-life > 8h (llm_in_sample); holdout edge(0) < 0 so undefined"
  confidence: B
  evidence: [H-IC3, research/reports/decision_budget_2026-11-10.md]
  review: "holdout edge(0) turning positive with R(4) < 0.5; re-run on a longer holdout"
- id: R-IC-4
  rule: "WHEN the candidate is a shock signal (shock_regime) THEN don't delay it for LLM review; the deterministic layer executes at the bar close and the LLM may only veto"
  when: "shock_regime entry on a 1h bar (|z| >= 4, declustered)"
  then: "expect most of the edge to be gone within ~1-2h of the shock-bar close"
  scope: {assets: crypto, horizon: "1h", sides: both}
  effect: "point estimates R(1h) 0.61, R(2h) 0.10, half-life ~1.2h, but edge(0) t 1.05 and CIs include 0.5 (in-sample, contaminated: strategy designed after seeing it)"
  confidence: C
  evidence: [H-IC3, research/reports/decision_budget_2026-11-10.md]
  review: "forward shock_regime record (H-S3) with edge(0) t > 2; re-run H-IC3 when the forward record has >= 100 shocks"
```
