# ML signals round 2: next-hour volatility (H-NN8) and a wider network for the SAE (H-NN9)

_Program 2026-11-12, agent B, round 2. Code: `src/ai_trader/ml/hourly.py`, `src/ai_trader/research/ml_signals_round2_study.py`; tests: `tests/test_ml_round2.py`; artifacts: `data/models/ml/round2/`. These are deterministic research tools. The LLM layer is not trained or backtested here. H-NN7 and H-LEV7 are forward-only and are not tested here._

## Summary

| hypothesis | in-sample | holdout | pooled |
|---|---|---|---|
| H-NN8 | supports | supports | **supports** |
| H-NN9 | contradicts | contradicts | **contradicts** |

**H-NN8.** The next-hour QLIKE difference against the champion (robust_bc x hour-of-week factor) is -0.04045 (t -5.11) for XGB, -0.03617 (t -4.82) for the MLP and -0.02453 (t -4.28) for the linear reference in-sample. In the holdout it is -0.06493 (t -2.65), -0.06455 (t -2.94) and -0.02869 (t -3.07). For scale, the champion itself beats the bare robust sigma by -0.04776 (t -8.84) in-sample and -0.07595 (t -4.49) in the holdout.

**H-NN9.** With a (64, 64) network, the SAE picked by the in-sample rule (dictionary 128, L1 0.3) has 0 of 10 most-used features tied to a named state on the holdout (0 in-sample). Ablating the top 3 changes holdout QLIKE by +0.0015 (t 0.10). The 64-unit ensemble against the 16-unit H-NN1 MLP: QLIKE -0.0002 (t -0.08) in-sample, -0.0022 (t -0.77) in the holdout.

## Methods

**H-NN8.** The cached H-NS4 1h frames for universe L (23 symbols) are placed on a regular hourly grid, so no return spans a gap. There are 19 point-in-time features (`ml.hourly`):

- log robust per-bar variance s2;
- log causal hour-of-week factor;
- funding-hour flags;
- log r^2/s2 at 1, 2, 3, 6, 12 and 24 h lags, plus 6 h and 24 h means;
- 24 h jump flag (|z| > 4) and hours since the last jump;
- daily HAR level (log RV of the last complete day, 7 d and 30 d means) relative to 24 s2;
- weekday.

The target is log(r^2/s2) and the forecast is s2 exp(f + c). The champion is s2 x factor (R-VOL-26). The walk-forward is monthly and expanding from 2021-07 (65 refits, 35 CPU-minutes) with a 1-day embargo.

**Runtime choice:** the models train on a deterministic random 1/3 of the bars (329,164 rows at the last refit). Every bar of each test month is scored (969,177 test bars: 896,336 in-sample, 72,841 holdout). A fixed every-3rd-hour rule was rejected because it would alias with the 24 h cycle.

Protocol as H-NN1: purged validation block, early stopping, validation-block bias constant. Mean early-stopped size: 4.2 epochs and 304 trees. Configs come from equal 4-config searches on pre-2021-07 rows only: MLP [64, 32], wd 0.01, batch 2048; XGB depth 3, mcw 200. The loss is the per-bar QLIKE difference r^2 (1/Fa - 1/Fb) + log(Fa/Fb), averaged per UTC date, with NW t.

**H-NN9.** The H-NN1 walk-forward is repeated with only the network changed: hidden (64, 64), wd 0.01, 3 seeds (56 refits, mean 6.5 epochs). The frozen model is the seed-0 member for 2026-07, trained on labels up to 2026-06-30.

SAE selection, all in-sample:

- dictionary size 64 or 128;
- for each size, L1 is the largest value in the grid with FVU <= 0.10 on the last 20% of in-sample dates;
- the size is then chosen by its tie count on that in-sample block.

- size 64: L1 0.3, validation FVU 0.067, L0 21.1, in-sample validation ties 0/10
- size 128: L1 0.3, validation FVU 0.054, L0 28.9, in-sample validation ties 0/10

Chosen: 128, L1 0.3. Holdout FVU 0.047, L0 29.1, 128 alive features.

## Results

### H-NN8 next-hour QLIKE difference vs the champion

| segment | model | diff | NW t | dates | bars |
|---|---|---|---|---|---|
| llm_in_sample | mlp | -0.03617 | -4.82 | 1826 | 896,336 |
| llm_in_sample | xgb | -0.04045 | -5.11 | 1826 | 896,336 |
| llm_in_sample | lin | -0.02453 | -4.28 | 1826 | 896,336 |
| llm_in_sample | champion vs bare s2 (context) | -0.04776 | -8.84 | 1826 | 896,336 |
| clean_holdout | mlp | -0.06455 | -2.94 | 132 | 72,841 |
| clean_holdout | xgb | -0.06493 | -2.65 | 132 | 72,841 |
| clean_holdout | lin | -0.02869 | -3.07 | 132 | 72,841 |
| clean_holdout | champion vs bare s2 (context) | -0.07595 | -4.49 | 132 | 72,841 |

### H-NN8 what the hourly trees use (TreeSHAP mean |phi| share) and group ablation (dQLIKE when a group is set to its training mean)

| group | SHAP share in | SHAP share ho | ablation xgb in | ablation xgb ho | ablation mlp in | ablation mlp ho |
|---|---|---|---|---|---|---|
| base_sigma | 2.1% | 1.0% | -0.00082 (-1.3) | -0.00021 (-0.4) | -0.00229 (-1.8) | -0.00283 (-0.7) |
| seasonal | 22.4% | 24.3% | 0.03172 (10.9) | 0.05907 (4.3) | 0.03136 (9.5) | 0.06283 (4.1) |
| funding_hour | 6.7% | 4.3% | -0.00102 (-1.3) | -0.00243 (-1.8) | -0.00299 (-1.9) | -0.00176 (-1.2) |
| recent_returns | 37.6% | 41.0% | 0.03274 (4.8) | 0.06695 (1.8) | 0.03131 (4.7) | 0.07455 (1.7) |
| jump | 3.6% | 3.7% | 0.00108 (1.5) | 0.00351 (0.8) | 0.00007 (0.1) | -0.00506 (-1.0) |
| daily_har | 21.6% | 20.6% | 0.02159 (8.0) | 0.02798 (3.2) | 0.02709 (10.3) | 0.02787 (3.1) |
| weekday | 6.0% | 5.0% | 0.00373 (4.3) | 0.00318 (1.2) | 0.00321 (2.6) | 0.00361 (1.2) |

Top features (holdout SHAP share): lfac 24.3%; lu_1 12.8%; har_w 9.6%; har_m 8.0%; lu_m6 7.9%; lu_m24 6.3%; lu_2 5.2%; lu_24 4.8%

### H-NN9 SAE features (holdout)

| feature | freq in | freq ho | best state | abs rho | ties | top inputs | d log RV per unit |
|---|---|---|---|---|---|---|---|
| 65 | 0.30 | 0.45 | vol_level | 0.50 | no | log_sigma -0.63, lrv_d -0.62, lrv_w -0.62 | -0.153 |
| 14 | 0.30 | 0.31 | drawdown | 0.15 | no | log_sigma +0.20, lrv_q +0.19, vol_pct_1y +0.18 | +0.073 |
| 32 | 0.30 | 0.23 | trend_strength | 0.12 | no | next_is_monday +0.20, x_ret1_fundz +0.19, neg_share_d -0.17 | +0.061 |
| 8 | 0.30 | 0.40 | vol_level | 0.22 | no | dvr_30d +0.28, day_range_z +0.25, abs_ret_1d_z +0.25 | -0.005 |
| 59 | 0.28 | 0.28 | weekend | 0.30 | no | next_is_weekend +0.30, dow_sin -0.28, next_wk_hours +0.27 | +0.027 |
| 99 | 0.28 | 0.13 | drawdown | 0.20 | no | mkt_lrv_d +0.27, log_sigma +0.25, lrv_d +0.23 | +0.081 |
| 86 | 0.27 | 0.14 | vol_level | 0.40 | no | lrv_m +0.40, lrv_q +0.38, lrv_w +0.35 | +0.145 |
| 40 | 0.27 | 0.30 | drawdown | 0.22 | no | mkt_ret_1d_z -0.31, vr_30d -0.29, dvr_30d -0.29 | +0.020 |
| 17 | 0.27 | 0.30 | vol_level | 0.25 | no | lrv_m +0.25, lrv_d +0.25, lrv_q +0.24 | +0.049 |
| 89 | 0.27 | 0.23 | vol_level | 0.43 | no | lrv_m +0.43, lrv_q +0.40, lrv_w +0.36 | +0.100 |

| segment | ablation | dQLIKE | NW t | intact QLIKE |
|---|---|---|---|---|
| clean_holdout | top3 | +0.00151 | 0.10 | 0.3590 |
| clean_holdout | top10 | -0.00001 | -0.00 | 0.3590 |
| llm_in_sample | top3 | +0.00281 | 1.20 | 0.2993 |
| llm_in_sample | top10 | +0.00511 | 1.56 | 0.2993 |

Linear probes (holdout R^2):

| layer | vol_level | vol_of_vol | recent_jump | weekend | trend_strength | funding_z | drawdown |
|---|---|---|---|---|---|---|---|
| input | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| hidden1 | 0.92 | 0.73 | 0.70 | 0.93 | 0.71 | 0.74 | 0.82 |
| hidden2 | 0.88 | 0.51 | 0.40 | 0.80 | 0.54 | 0.59 | 0.71 |
| sae_codes | 0.88 | 0.39 | 0.43 | 0.82 | 0.53 | 0.52 | 0.71 |

Last-layer geometry: 0 dead units of 64. Participation ratio (effective dimension) 4.7, against 3.1 for the 16-unit model. The frozen 64-unit model's holdout QLIKE is 0.3590.

### H-NN9 the 64-unit MLP's own forecasts (walk-forward ensemble, daily QLIKE)

| segment | 64 - 16 unit | 64-unit - log-HAR | 64-unit - XGB |
|---|---|---|---|
| llm_in_sample | -0.0002 (-0.1%, t -0.08) | -0.0317 (-8.1%, t -2.71) | -0.0143 (-3.8%, t -1.49) |
| clean_holdout | -0.0022 (-0.6%, t -0.77) | -0.0276 (-6.7%, t -1.04) | 0.0370 (+10.7%, t 0.97) |

### Verdicts (after Holm within round 2)

| hypothesis | in-sample | holdout | pooled | Holm p (in / ho) |
|---|---|---|---|---|
| H-NN8 | supports | supports | supports | 0.000 / 0.007 |
| H-NN9 | contradicts | contradicts | contradicts | 1.000 / 1.000 |

## What this changes for tool calls

- **Next hour:** a nonlinear model (mlp, xgb) beats the robust sigma x hour-of-week factor in both segments. The groups it relies on most are recent_returns (41%), seasonal (24%), daily_har (21%). The intraday champion leaves usable structure, so the decision layer should prefer such a tool once it is built and forward-tested.
- **How much is nonlinear:** a ridge-linear model on the same 19 inputs already gets -0.02453 (t -4.28) in-sample and -0.02869 (t -3.07) in the holdout. Trees add -0.01592 (t -4.22) and -0.03624 (t -1.93) on top. So roughly half of the gain is just extra inputs and half is nonlinearity, unlike the daily panel, where the gain was mostly calendar.
- **Which inputs to add to the intraday forecast:** ablation matters most for recent |returns| (significant in-sample, t 1.8 in the holdout) (XGB +0.0327, t 4.8 in-sample; +0.0669, t 1.8 holdout), for the daily HAR level (+0.0216, t 8.0; +0.0280, t 3.2) and for the seasonal factor itself. Funding-hour flags, jump flags and the base sigma level beyond the offset do not matter (|t| < 2, or the wrong sign). The robust 1h sigma reacts too slowly and too smoothly: the last 1-6 hours of |returns| and the daily HAR level carry information it lacks.
- **Interpretability:** with 64 units the SAE result is contradicts (0/10 named features on the holdout). Width does not make the network's features nameable. The 64-unit last layer has an effective dimension (participation ratio) of only 4.7, against 3.1 for 16 units. The network represents a handful of mixed directions, mostly vol level and weekend, so there is no superposition of many concepts for an SAE to separate. Keep explaining model forecasts with grouped TreeSHAP (R-ML-6).
- **Network width:** going from 16 to 64 units changes daily QLIKE by -0.0022 (t -0.77) in the holdout and -0.0002 (t -0.08) in-sample. Capacity is not what limits the daily vol forecast.

## Caveats

- The hourly models train on a random third of the bars to keep runtime sane. All test bars are scored.
- The search window before the first hourly test month is small (BTC/ETH-heavy, about one year). Configs were frozen from it as registered.
- The holdout has 132 dates for H-NN8 and 133 dates for the daily H-NN9 comparison.
- The H-NN9 interpretation rests on one frozen seed-0 network. Refit noise across seeds and months is large (round 1).

## Rule candidates

```yaml
- id: R-ML-7
  kind: prefer
  rule: WHEN a central next-hour variance expectation is needed for a crypto perp and a validated hourly ML tool exists THEN prefer it over robust_bc x the hour-of-week factor, for expectations only, never for stops
  when: horizon 1h, quoting or intraday expectations
  then: use the hourly model (features of ml.hourly) with its in-sample bias; fall back to R-VOL-26 if it is stale
  scope:
    assets: crypto perps (universe L)
    horizon: 1h
    sides: both
  effect: next-hour QLIKE diff vs champion: XGB -0.04045 (t -5.11) in-sample / -0.06493 (t -2.65) holdout; MLP -0.03617 (t -4.82) / -0.06455 (t -2.94)
  confidence: C
  evidence: [H-NN8, H-NS4]
  review: forward test for 1 month against R-VOL-26 before use; demote if the forward QLIKE gain is not significant
  status: provisional
- id: R-ML-8
  kind: expect
  rule: WHEN considering a wider neural network for the daily crypto vol forecast THEN don't expect extra capacity to improve it; the forecast is limited by information (vol level, weekday), not by network width
  when: choosing or retraining a daily vol model
  then: keep small networks or trees; spend effort on inputs and validation, not width
  scope:
    assets: crypto perps (universe L)
    horizon: 1d
    sides: both
  effect: 64-unit vs 16-unit last layer, daily QLIKE -0.0002 (t -0.08) in-sample, -0.0022 (t -0.77) holdout
  confidence: B
  evidence: [H-NN9, H-NN1]
  review: a wider model with a QLIKE gain at t < -2 in both segments
  status: active
- id: R-ML-9
  kind: avoid
  rule: WHEN explaining a daily vol forecast THEN don't cite sparse-autoencoder features of the network, even of a wider one; they are not stable named concepts. Use grouped TreeSHAP (R-ML-6)
  when: a model-based vol forecast is shown with an explanation
  then: grouped TreeSHAP (vol level, calendar, context) in log-RV units
  scope:
    assets: crypto perps (universe L)
    horizon: 1d
    sides: both
  effect: 64-unit layer, SAE dictionary 128: 0/10 features named on holdout (16-unit H-NN4: 1/10); top-3 ablation +0.0015 QLIKE (t 0.10)
  confidence: B
  evidence: [H-NN9, H-NN4]
  review: an SAE on a model with more concepts in play (micro data, H-NN5) that yields >= 6/10 named features
  status: active
- id: R-ML-10
  kind: prefer
  rule: WHEN forming a next-hour variance expectation for a crypto perp THEN adjust the robust sigma x hour-of-week forecast for the last 1-6 hours of absolute returns and the daily HAR level; ignore funding-hour and jump flags for this purpose
  when: horizon 1h, central expectation or quote width; not stops
  then: recent |returns| above the robust sigma and a daily HAR level above 24 x sigma^2 both point to a higher next-hour variance than the champion gives
  scope:
    assets: crypto perps (universe L)
    horizon: 1h
    sides: both
  effect: group ablation in the hourly XGB raises QLIKE by +0.0327 (t 4.8) / +0.0669 (t 1.8) for recent returns and +0.0216 (t 8.0) / +0.0280 (t 3.2) for the daily HAR level (in-sample / holdout); funding-hour and jump groups |t| < 2; linear model on the same inputs -0.02453 (t -4.28) / -0.02869 (t -3.07) vs the champion
  confidence: B
  evidence: [H-NN8, H-NS4, H-S4]
  review: build vol.hourly_x (linear first, R-ML-7 trees second) and forward test for 1 month against R-VOL-26
  status: provisional
```
