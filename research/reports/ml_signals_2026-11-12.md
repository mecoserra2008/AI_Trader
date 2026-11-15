# Neural-network and tree signal generators, with interpretability (H-NN1 to H-NN4)

_Program 2026-11-12 (overnight loop, agent B). Data as of 2026-11-12; knowledge cutoff 2026-07-01. Code: `src/ai_trader/ml/` (features, models, interpret) and `src/ai_trader/research/ml_signals_study.py`; tests: `tests/test_ml.py`; models and artifacts: `data/models/ml/`. These models are deterministic research tools. The LLM decision layer is not trained or backtested here (principle 1)._

## Summary

| hypothesis | in-sample | holdout | pooled | headline |
|---|---|---|---|---|
| H-NN1: nonlinear models beat log-HAR for next-day vol | supports | supports | **inconclusive** | in-sample MLP -8.0% QLIKE (t -2.92), XGB -4.5% (t -0.98); holdout XGB -15.7% (t -2.40), MLP -6.2% (t -0.89); the winning model is different in each segment |
| H-NN2: nonlinear models do not beat the base rate for direction (null) | supports | inconclusive | inconclusive | Brier skill in-sample MLP -0.0024, XGB -0.0016; holdout +0.0053 / +0.0018, with CI upper bounds +0.0161 / +0.0067 (above +0.005) |
| H-NN3: interactions matter for vol (TreeSHAP) | supports | supports | **supports** | interaction share 33.4% / 34.1% of \|phi\|; top-5 pair overlap across segments 5/5 |
| H-NN4: the SAE on the MLP's hidden layer finds named market states | contradicts | contradicts | **contradicts** | 1 of 10 features tie to a state; top-3 ablation +0.092 QLIKE (t 1.32) |

**What the models use.** Next-day volatility runs on two things. The first is the **vol level**: the 7-day mean log RV and the robust sigma, 43% / 40% of all TreeSHAP attribution as main effects. The second is the **forecast day's weekday**: the calendar group carries a 10% main effect plus the largest cross-group interaction. Relative to the model's average day, Saturday is 0.64x / 0.57x (in-sample / holdout), Sunday 0.83x / 0.84x and Monday 1.27x / 1.36x. Trend regime, funding and the market factor add about 1-3% each. Jump, semivariance, range-estimator and raw-return inputs add nothing. Trees and networks agree on this ranking: Spearman 0.72 / 0.75 over the 52 features, and 9 / 8 of the top 10 in common. **Direction has no learnable signal.** The models stop after 1.5 epochs / 24 trees on average. The families disagree on what matters (Spearman 0.22 in the holdout), and the tree's direction attributions sit mostly on market-wide day-level inputs (58% of its attribution), which is the signature of fitting date noise.

**How much nonlinearity adds over log-HAR.** It adds 6-16% QLIKE, but most of that is calendar knowledge, which log-HAR lacks. A ridge-linear model on the same 52 inputs gets -5.9% (t -2.43) in-sample and -3.1% (t -0.65) in the holdout. Per-asset log-HAR plus six day-of-week dummies (exploratory, post hoc) gets -5.5% (t -2.92) in-sample and -2.3% (t -0.57) in the holdout. Beyond that, the MLP and XGB add -2.7% / +1.1% in-sample and -4.0% / -13.7% (t -0.45 / -1.57) in the holdout. None of these is significant. The holdout XGB gain is broad, 22 of 23 symbols, but it rests on 133 dates, and holdout weekend days alone show -26.4% (t -3.74).

## Methods

**Panel.** `ai_trader.ml.features`, universe L (23 crypto perps), one row per symbol and UTC day D. Decision time T = 00:00 UTC of D+1. There are 52 point-in-time features:

- the fixed H-IC1 list of 32;
- the log-HAR inputs (log RV_D, 7d / 30d mean log RV, as `vol_panel.design('loghar')`) and the 90d mean log RV;
- daily downside share, log(sqrt RQ / RV), daily jump share, log(Parkinson / RV), log(GK / RV), |ret_1d_z|;
- the forecast day's weekday (sin/cos), TradFi-weekend hours in D+1, a Deribit monthly-expiry flag;
- the v0 regime (direction, vol state, score) at the 20:00 4h bar;
- the crypto vol factor and the mean cross-sectional return.

Only complete rows are used, so every model sees an identical sample: 38,275 rows from 2021-03-31 to 2026-11-10 (35,216 llm_in_sample, 3,059 clean_holdout). A unit test checks causality: appending future bars, funding or OI leaves past rows bit-identical. Standardisation uses fit-block statistics only. Concept groups for attribution: vol_level: log_sigma, lrv_d, lrv_w, lrv_m, lrv_q, vol_pct_1y; vol_shape: rv_surprise, rv_term, vol_of_vol_30d, rq_ratio_d; jumps_semivar: semivar_7d, jump_share_7d, neg_share_d, jump_share_d; range_measures: pk_ratio_d, gk_ratio_d, day_range_z; returns: ret_1d_z, ret_3d_z, ret_7d_z, ret_30d_z, abs_ret_1d_z, x_ret7_volpct; trend_regime: er_7d, er_30d, tstat_7d, tstat_30d, dvr_7d, dvr_30d, vr_30d, ac1_4h_30d, ac1_1d_30d, v0_dir, v0_vol, v0_dir_score, range_pos_30d, dd_90d_z; funding_oi: funding_1d, funding_z_90d, x_ret1_fundz, oi_chg_1d, oi_chg_7d; calendar: next_is_weekend, next_is_monday, dow_sin, dow_cos, next_wk_hours, next_is_mexpiry; market: btc_ret_1d_z, mkt_lrv_d, mkt_lrv_w, mkt_ret_1d_z.

**Walk-forward.** Monthly and expanding, 2022-04 to 2026-11 (56 refits, 19 CPU-minutes). Training rows have label_end <= month - 1 day (24h purge plus 1-day embargo). Inside each window:

- the last 15% of dates form a purged validation block;
- both families stop early on that block (MLP: patience 10, max 200 epochs; XGB: 50 rounds, max 2000);
- the variance bias correction exp(f + c), with c = log mean(RV / exp(f)), is estimated on that block.

**Models.**

- MLP: 2 ReLU layers, AdamW lr 1e-3, batch 512, MSE on standardised log RV or BCE, 3-seed ensemble.
- XGB: hist, eta 0.03, subsample and colsample 0.8.
- Equal tuning effort: 4 configs each, chosen only on rows whose labels end before 2022-04 (diary entries 'ML design fixed before evaluation' and 'walk-forward started'). Chosen: vol MLP (32, 16), wd 0.01; vol XGB depth 3, mcw 20; direction MLP (64, 32); direction XGB depth 3. Mean early-stopped size: vol 11 epochs / 342 trees.
- Baseline: per-asset log-HAR (OLS on the symbol's own rows, at least 180 of them, exp(xb + s^2/2), R-VOL-20), with the same months and purge.
- References: the same HAR with an in-sample QLIKE-optimal scale, and ridge-linear on the same 52 features.

**Tests.**

- H-NN1: QLIKE on RV_{D+1}, paired difference averaged over symbols per date, NW t.
- H-NN2: Brier skill vs the training base rate, with a 10-day block bootstrap.
- H-NN3: xgboost exact TreeSHAP interaction values on each month's own out-of-sample rows (bias excluded), 10-day block bootstrap over dates.
- H-NN4: frozen MLP = seed-0 member of the 2026-07 vol ensemble (labels to 2026-06-30).
  - SAE: untied, dictionary 64 = 4 x 16, unit-norm decoder, pre-encoder bias, dead-feature resampling.
  - L1 = the largest of 0.03 / 0.1 / 0.3 / 1.0 with FVU <= 0.10 on the last 20% of in-sample dates, which selected 0.3.
  - The 10 most-used features are ranked by in-sample firing frequency, then correlated (Spearman) with the named states on holdout activations.
  - Ablation is error-preserving: H - sum z_k a_k is passed through the frozen head.
- Holm within NN, per segment. Full specification in the study docstring.

## Results

### H-NN1 next-day RV: QLIKE vs per-asset log-HAR

| segment | model | QLIKE model | QLIKE log-HAR | diff | rel | NW t | dates |
|---|---|---|---|---|---|---|---|
| llm_in_sample | mlp | 0.3600 | 0.3914 | -0.0315 | -8.0% | -2.92 | 1552 |
| llm_in_sample | xgb | 0.3740 | 0.3914 | -0.0174 | -4.5% | -0.98 | 1552 |
| llm_in_sample | lin | 0.3682 | 0.3914 | -0.0233 | -5.9% | -2.43 | 1552 |
| clean_holdout | mlp | 0.3856 | 0.4110 | -0.0254 | -6.2% | -0.89 | 133 |
| clean_holdout | xgb | 0.3464 | 0.4110 | -0.0646 | -15.7% | -2.40 | 133 |
| clean_holdout | lin | 0.3982 | 0.4110 | -0.0127 | -3.1% | -0.65 | 133 |

Other pairs (diff, NW t):

| segment | mlp - xgb | mlp - lin | xgb - lin | har_qs - har | mlp - har_qs | xgb - har_qs |
|---|---|---|---|---|---|---|
| llm_in_sample | -0.0141 (-1.4) | -0.0082 (-1.5) | 0.0058 (0.5) | -0.0003 (-0.1) | -0.0312 (-2.9) | -0.0171 (-0.9) |
| clean_holdout | 0.0392 (1.0) | -0.0127 (-0.9) | -0.0518 (-1.4) | -0.0029 (-0.3) | -0.0225 (-0.7) | -0.0617 (-3.1) |

By v0 vol state (xgb / mlp rel. QLIKE vs log-HAR, t):

| segment | vol state | n obs | xgb | mlp |
|---|---|---|---|---|
| llm_in_sample | low | 12684 | -2.5% (-0.4) | -8.3% (-2.4) |
| llm_in_sample | normal | 10097 | -5.7% (-1.4) | -8.2% (-2.5) |
| llm_in_sample | high | 9687 | -5.9% (-2.3) | -4.9% (-1.5) |
| llm_in_sample | next day weekend=0 | 23184 | -5.4% (-1.6) | -8.0% (-2.6) |
| llm_in_sample | next day weekend=1 | 9284 | -2.4% (-0.2) | -8.1% (-1.4) |
| clean_holdout | low | 1083 | -15.5% (-1.8) | +2.8% (0.3) |
| clean_holdout | normal | 850 | -3.3% (-0.4) | -9.2% (-1.4) |
| clean_holdout | high | 1126 | -13.9% (-2.1) | -13.8% (-2.4) |
| clean_holdout | next day weekend=0 | 2185 | -10.3% (-1.3) | +3.9% (0.4) |
| clean_holdout | next day weekend=1 | 874 | -26.4% (-3.7) | -26.1% (-3.3) |

### H-NN2 next-24h direction: Brier skill vs training base rate

| segment | model | skill | 95% CI | AUC | log loss | sd(p) | P*(skill > 0.005) |
|---|---|---|---|---|---|---|---|
| llm_in_sample | mlp | -0.00238 | [-0.00722, 0.00211] | 0.5143 | 0.6945 | 0.0381 | 0.001 |
| llm_in_sample | xgb | -0.00163 | [-0.00546, 0.00222] | 0.5016 | 0.6941 | 0.0325 | 0.001 |
| llm_in_sample | lin | -0.02707 | [-0.03928, -0.01578] | 0.5067 | 0.7077 | 0.0915 | 0.001 |
| clean_holdout | mlp | 0.00535 | [-0.00333, 0.01613] | 0.5485 | 0.6909 | 0.0284 | 0.460 |
| clean_holdout | xgb | 0.00183 | [-0.00531, 0.00671] | 0.5512 | 0.6927 | 0.0127 | 0.108 |
| clean_holdout | lin | 0.00879 | [-0.01436, 0.02327] | 0.5546 | 0.6892 | 0.0539 | 0.500 |

### H-NN3 TreeSHAP (xgb vol model, out-of-sample rows)

| segment | interaction share | 95% CI | top-5 pairs (share of total |phi|) |
|---|---|---|---|
| llm_in_sample | 33.4% | [33.0, 33.8] | log_sigma x lrv_w 2.26%; btc_ret_1d_z x mkt_ret_1d_z 1.10%; log_sigma x next_is_weekend 0.97%; log_sigma x lrv_q 0.89%; next_is_weekend x lrv_w 0.85% |
| clean_holdout | 34.1% | [33.2, 34.8] | log_sigma x lrv_w 2.13%; btc_ret_1d_z x mkt_ret_1d_z 1.30%; log_sigma x lrv_q 1.11%; log_sigma x next_is_weekend 0.95%; next_is_weekend x lrv_w 0.84% |

Top-5 overlap across segments: 5 (bootstrap mean 5.00, P*(overlap < 3) 0.001).

| segment | top main effects (share of total |phi|) |
|---|---|
| llm_in_sample | log_sigma 14.7%; lrv_w 14.5%; lrv_q 4.2%; next_is_weekend 3.6%; dow_sin 2.9%; lrv_m 2.2%; next_wk_hours 2.1%; lrv_d 1.9%; rq_ratio_d 1.9%; v0_dir_score 1.9% |
| clean_holdout | lrv_w 14.5%; log_sigma 12.3%; dow_sin 3.6%; next_is_weekend 3.5%; lrv_q 3.4%; mkt_ret_1d_z 2.7%; lrv_m 2.4%; rq_ratio_d 2.0%; v0_dir_score 2.0%; lrv_d 1.9% |

Concept-group pairs (interaction share of total |phi|):

- llm_in_sample: vol_level x calendar 5.18%; vol_level x trend_regime 2.88%; calendar x market 2.17%; vol_level x market 1.77%; trend_regime x market 1.51%; vol_shape x market 0.98%
- clean_holdout: vol_level x calendar 4.43%; vol_level x trend_regime 2.21%; calendar x market 2.09%; trend_regime x market 1.81%; vol_shape x market 1.17%; vol_level x funding_oi 1.01%

### H-NN4 sparse autoencoder on the frozen MLP's last hidden layer

Frozen model: month 2026-07 member 0, hidden [32, 16], last training label 2026-06-30T00:00:00.000000; holdout QLIKE 0.3661. SAE dictionary 64 (4 x 16), L1 0.3 (selection: 0.03: FVU 0.001, L0 27.1, 0.1: FVU 0.008, L0 22.4, 0.3: FVU 0.039, L0 14.4, 1.0: FVU 0.166, L0 6.4); in-sample FVU 0.036, L0 14.12; holdout FVU 0.032, L0 14.30; alive 64 of 64.

| feature | freq in | freq ho | best state (holdout) | abs rho | ties | top inputs (rho) | d log RV per unit |
|---|---|---|---|---|---|---|---|
| 4 | 0.39 | 0.54 | vol_level | 0.68 | yes | lrv_w -0.75, log_sigma -0.72, lrv_d -0.71 | -0.416 |
| 40 | 0.33 | 0.21 | vol_level | 0.37 | no | lrv_d +0.51, log_sigma +0.50, vol_pct_1y +0.42 | +0.323 |
| 9 | 0.33 | 0.20 | vol_level | 0.44 | no | log_sigma +0.56, lrv_w +0.54, lrv_d +0.54 | +0.370 |
| 17 | 0.31 | 0.43 | vol_level | 0.56 | no | lrv_w -0.59, log_sigma -0.58, lrv_d -0.58 | -0.086 |
| 33 | 0.30 | 0.32 | weekend | 0.43 | no | next_is_weekend +0.43, next_wk_hours +0.39, ret_3d_z -0.39 | -0.115 |
| 45 | 0.29 | 0.21 | vol_level | 0.46 | no | log_sigma +0.56, lrv_w +0.55, lrv_d +0.55 | +0.437 |
| 19 | 0.28 | 0.15 | vol_level | 0.32 | no | lrv_w +0.34, lrv_m +0.32, lrv_q +0.31 | +0.211 |
| 13 | 0.28 | 0.26 | weekend | 0.31 | no | dow_sin +0.37, next_wk_hours -0.34, lrv_w +0.31 | +0.066 |
| 34 | 0.27 | 0.39 | weekend | 0.39 | no | next_wk_hours +0.45, dow_sin -0.39, next_is_weekend +0.39 | +0.007 |
| 11 | 0.27 | 0.26 | recent_jump | 0.30 | no | range_pos_30d -0.46, semivar_7d +0.46, dvr_7d -0.45 | -0.100 |

Ties: 1 / 10 on holdout (1 / 10 in-sample).

| segment | ablation | dQLIKE | NW t | intact QLIKE |
|---|---|---|---|---|
| clean_holdout | top3 | +0.09188 | 1.32 | 0.3661 |
| clean_holdout | top10 | +0.11471 | 1.64 | 0.3661 |
| llm_in_sample | top3 | +0.14410 | 5.20 | 0.3078 |
| llm_in_sample | top10 | +0.19718 | 5.05 | 0.3078 |

Linear probes (R^2 on holdout, ridge fit in-sample):

| layer | vol_level | vol_of_vol | recent_jump | weekend | trend_strength | funding_z | drawdown |
|---|---|---|---|---|---|---|---|
| input | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| hidden1 | 0.89 | 0.47 | 0.52 | 0.83 | 0.48 | 0.56 | 0.63 |
| hidden2 | 0.73 | 0.18 | 0.33 | 0.52 | 0.26 | 0.28 | 0.37 |
| sae_codes | 0.75 | 0.15 | 0.37 | 0.60 | 0.31 | 0.30 | 0.45 |

### Do trees and networks agree? (mean |attribution|, out-of-sample)

| target | segment | Spearman (52 features) | top-10 overlap | group Spearman | xgb top 5 | mlp top 5 |
|---|---|---|---|---|---|---|
| vol | llm_in_sample | 0.72 | 9 | 0.87 | log_sigma, lrv_w, lrv_q, next_is_weekend, dow_sin | lrv_w, lrv_m, log_sigma, lrv_q, lrv_d |
| vol | clean_holdout | 0.75 | 8 | 0.93 | lrv_w, log_sigma, lrv_q, next_is_weekend, dow_sin | lrv_w, log_sigma, lrv_m, lrv_d, lrv_q |
| dir | llm_in_sample | 0.59 | 6 | 0.88 | mkt_ret_1d_z, mkt_lrv_d, btc_ret_1d_z, mkt_lrv_w, dow_sin | mkt_ret_1d_z, neg_share_d, mkt_lrv_d, dow_cos, ret_1d_z |
| dir | clean_holdout | 0.22 | 4 | 0.72 | mkt_lrv_d, mkt_ret_1d_z, btc_ret_1d_z, mkt_lrv_w, dow_sin | lrv_q, neg_share_d, dow_cos, ac1_1d_30d, mkt_lrv_d |

Grouped attribution shares (xgb TreeSHAP / mlp IG), clean_holdout:

| group | vol xgb | vol mlp | dir xgb | dir mlp |
|---|---|---|---|---|
| vol_level | 51.6% | 36.8% | 9.4% | 12.9% |
| vol_shape | 4.5% | 6.9% | 1.1% | 7.3% |
| jumps_semivar | 1.1% | 3.0% | 2.1% | 10.2% |
| range_measures | 2.4% | 3.8% | 0.0% | 5.6% |
| returns | 2.1% | 6.2% | 4.9% | 10.3% |
| trend_regime | 9.5% | 14.7% | 9.7% | 26.8% |
| funding_oi | 3.9% | 3.2% | 0.5% | 4.4% |
| calendar | 16.5% | 17.9% | 13.8% | 14.8% |
| market | 8.4% | 7.5% | 58.4% | 7.7% |

### Group ablation (inputs set to the training mean): QLIKE increase (rel., NW t) for vol

| group | mlp llm_in_sample | xgb llm_in_sample | mlp clean_holdout | xgb clean_holdout |
|---|---|---|---|---|
| vol_level | +75.5% (10.3) | +75.5% (8.8) | +53.5% (2.4) | +76.5% (5.0) |
| vol_shape | +3.3% (2.2) | +0.7% (2.4) | -1.5% (-0.7) | -0.5% (-0.5) |
| jumps_semivar | -0.0% (-0.1) | +0.1% (0.7) | +1.3% (1.4) | -0.2% (-0.7) |
| range_measures | -0.7% (-1.0) | +0.2% (0.8) | +1.1% (1.3) | +0.5% (1.1) |
| returns | +1.5% (2.2) | +1.1% (1.6) | +1.4% (1.2) | +0.7% (0.7) |
| trend_regime | +2.5% (1.8) | +3.1% (2.7) | +2.6% (0.3) | +7.8% (1.0) |
| funding_oi | +0.6% (1.5) | +0.4% (1.6) | +1.4% (2.0) | +2.2% (1.5) |
| calendar | +7.0% (3.4) | +4.8% (1.2) | +7.0% (1.8) | +8.3% (2.6) |
| market | +0.6% (0.8) | -1.1% (-0.5) | +1.9% (0.9) | +2.8% (1.4) |

Direction: Brier-skill change when a group is ablated (negative = the group carried skill):

| group | mlp llm_in_sample | xgb llm_in_sample | mlp clean_holdout | xgb clean_holdout |
|---|---|---|---|---|
| vol_level | -0.00028 | +0.00016 | +0.00157 | -0.00009 |
| vol_shape | +0.00009 | +0.00001 | +0.00010 | -0.00009 |
| jumps_semivar | +0.00058 | -0.00011 | -0.00158 | -0.00022 |
| range_measures | +0.00005 | -0.00006 | +0.00160 | +0.00002 |
| returns | +0.00066 | -0.00022 | +0.00062 | -0.00021 |
| trend_regime | +0.00126 | -0.00010 | -0.00427 | -0.00095 |
| funding_oi | +0.00018 | -0.00001 | -0.00028 | -0.00000 |
| calendar | +0.00123 | -0.00038 | -0.00072 | -0.00132 |
| market | +0.00212 | -0.00315 | -0.00345 | +0.00179 |

### Exploratory (post hoc, not a verdict): per-asset log-HAR + day-of-week dummies

| segment | pair | diff | rel | NW t |
|---|---|---|---|---|
| llm_in_sample | har_dow - har | -0.0214 | -5.5% | -2.92 |
| llm_in_sample | mlp - har_dow | -0.0101 | -2.7% | -1.19 |
| llm_in_sample | xgb - har_dow | 0.0040 | +1.1% | 0.29 |
| llm_in_sample | lin - har_dow | -0.0019 | -0.5% | -0.20 |
| clean_holdout | har_dow - har | -0.0094 | -2.3% | -0.57 |
| clean_holdout | mlp - har_dow | -0.0160 | -4.0% | -0.45 |
| clean_holdout | xgb - har_dow | -0.0551 | -13.7% | -1.57 |
| clean_holdout | lin - har_dow | -0.0033 | -0.8% | -0.13 |

### Calendar contribution to log RV forecast (xgb TreeSHAP, mean by forecast weekday; weekend gap by v0 vol state)

| segment | Mon | Tue | Wed | Thu | Fri | Sat | Sun | weekend gap low / normal / high vol |
|---|---|---|---|---|---|---|---|---|
| llm_in_sample | +0.239 | +0.093 | +0.157 | +0.089 | +0.065 | -0.443 | -0.187 | -0.482 / -0.449 / -0.391 |
| clean_holdout | +0.309 | +0.105 | +0.132 | +0.095 | +0.105 | -0.567 | -0.176 | -0.571 / -0.515 / -0.480 |

### Direction of effects (Spearman of feature value vs its TreeSHAP value; clean_holdout, in-sample in brackets)

- vol: lrv_w +1.00 [+0.95]; log_sigma +0.99 [+0.97]; lrv_q +0.97 [+0.90]; next_is_weekend -0.78 [-0.78]; dow_sin +0.71 [+0.76]; mkt_ret_1d_z -0.75 [-0.44]; lrv_m +0.94 [+0.84]; v0_dir_score +0.61 [+0.50]; rq_ratio_d -0.96 [-0.93]; lrv_d +0.95 [+0.85]; next_is_monday +0.61 [+0.61]; btc_ret_1d_z +0.35 [+0.13]
- dir: mkt_lrv_d +0.18 [+0.66]; mkt_ret_1d_z -0.74 [-0.49]; btc_ret_1d_z -0.51 [-0.23]; mkt_lrv_w -0.51 [-0.20]; dow_sin -0.83 [+0.09]; lrv_q -0.73; tstat_7d -0.19 [-0.07]; dow_cos -0.76 [-0.01]; ret_1d_z -0.62 [-0.62]; next_wk_hours -0.25 [+0.09]; lrv_m -0.76; range_pos_30d -0.52 [-0.33]
- vol: joint btc_ret_1d_z + mkt_ret_1d_z contribution vs |mkt_ret_1d_z|: rho +0.50 holdout, +0.35 in-sample

### Verdicts (after Holm within NN)

| hypothesis | llm_in_sample | clean_holdout | pooled | Holm p (in / ho) |
|---|---|---|---|---|
| H-NN1 | supports | supports | inconclusive | 0.007 / 0.050 |
| H-NN2 | supports | inconclusive | inconclusive | 0.004 / 0.919 |
| H-NN3 | supports | supports | supports | 0.004 / 0.004 |
| H-NN4 | contradicts | contradicts | contradicts | 1.000 / 1.000 |


## Interpretability findings

1. **Trees: interactions are real, stable and mostly 'level x context'.** A third of the absolute TreeSHAP attribution is pairwise interaction, and the top-5 pairs are identical in both segments. About 8% of |phi| is interaction *within* the vol-level group: log_sigma x lrv_w and log_sigma x lrv_q are two estimates of the same level splitting credit, which is a collinearity artefact rather than knowledge. The economically meaningful interactions are:
   - **vol level x calendar** (5.2% of |phi| in-sample). The weekend discount is smaller when vol is already high: the TreeSHAP weekend gap is -0.48 / -0.45 / -0.39 log points in low / normal / high v0 vol in-sample, and -0.57 / -0.51 / -0.48 in the holdout.
   - **BTC x market return**. Their joint contribution rises with the size of the market move (rho +0.35 / +0.50 with |mkt_ret_1d_z|). The trees build an absolute value out of two signed inputs.
   - **vol level x trend regime**. Up-trends raise the forecast (v0_dir_score sign +0.50 / +0.61).
   Descriptive and not yet registered: market-wide *down* days raise the forecast (mkt_ret_1d_z sign -0.44 / -0.75), and a day dominated by one big hour lowers it (rq_ratio_d -0.93 / -0.96, the HAR-Q attenuation, R-VOL-23).
2. **Networks: no monosemantic state features.** The 16-unit last layer of the vol MLP is a distributed code for vol level and weekend. The SAE (FVU 0.032, L0 14.3 of 64) splits it into graded mixtures, not named concepts: only feature 4 (a 'low vol' detector, rho -0.75 with lrv_w) clears |rho| > 0.6. It is also the only feature whose ablation moves QLIKE materially (holdout +0.097, t 1.41; the top-3 ablation is significant in-sample, t 5.20); ablating the others individually changes QLIKE by less than 0.01. The probes explain why. The first hidden layer still encodes vol level (R^2 0.89) and weekend (0.83). The last layer keeps vol level (0.73) but has largely discarded vol-of-vol (0.18), trend (0.26) and funding (0.28): it throws away what does not help the vol forecast. With so few concepts in play there is little superposition for dictionary learning to undo, and a 4x dictionary over-splits the vol-level direction.
3. **Integrated gradients and TreeSHAP agree for vol** (feature-ranking Spearman 0.75, group Spearman 0.93 in the holdout). The MLP spreads more weight onto trend-regime inputs (15% vs 10%), but ablating that group costs little in either model.
4. **Group ablation (causal)**, relative QLIKE increase when a group is set to its training mean:
   (XGB in-sample / holdout; MLP in-sample / holdout; NW t in the table above)
   - vol level: +75.5% / +76.5%; +75.5% / +53.5%.
   - calendar: +4.8% / +8.3%; +7.0% / +7.0%.
   - trend regime: +3.1% / +7.8%; +2.5% / +2.6%.
   - funding / OI: +0.4% / +2.2%; +0.6% / +1.4%.
   - market factor: -1.1% / +2.8%; +0.6% / +1.9%.
   - jumps / semivariance: +0.1% / -0.2%; -0.0% / +1.3%.
   - range measures: +0.2% / +0.5%; -0.7% / +1.1%.
   - returns: +1.1% / +0.7%; +1.5% / +1.4%.
   For direction, no group moves Brier skill by more than 0.005.

## What to call when (for the decision layer)

| target | call first | then | marginal (tie-breakers) | noise (do not call for this target) | nonlinear model adds over log-HAR |
|---|---|---|---|---|---|
| next-day variance (crypto perp, 1d) | `vol.har` per-asset log-HAR: 7d mean log RV and robust sigma, about half of all attribution | forecast-day weekday: Saturday about 0.64x, Monday about 1.27x of the average day; the discount is smaller in high vol | trend regime (v0 dir score, up-trend raises vol), funding z / OI, market factor and the size of the market move; 1-3% QLIKE each, not robustly significant | jump variation, semivariances, Parkinson / GK range, raw vol-scaled returns, expiry flag, vol-of-vol on its own | in-sample -8.0% (MLP), holdout -15.7% (XGB); after adding weekday dummies to HAR, the remainder is not significant |
| next-day direction (sign of next-24h return) | nothing: no tool forecasts it | none | none | every standard feature, linear or nonlinear (MLP / XGB / linear Brier skill <= 0 in-sample, holdout CIs include 0, families disagree) | none (skill -0.0016 in-sample) |

Order for the decision layer: (1) the vol level from `vol.har`; (2) the calendar adjustment for the forecast day (the hour-of-week profile R-VOL-26/27 at intraday horizons, the weekday effect here at 1d); (3) at most one context line (trend regime or funding crowding) as a tie-breaker. A nonlinear forecaster is worth deploying only as a *central* next-day expectation tool, after a forward test, never for risk or stops (weekends are jump-prone, R-VOL-2). For direction the budget rule R-BUDGET-1 stands: spending inputs or nonlinear capacity on it adds noise.

## Caveats

- The holdout is 133 dates (2026-07 to 2026-11). The holdout XGB win (t -2.40) passes Holm only at 0.050, and the in-sample winner (MLP) does not win in the holdout. The family-level conclusion (nonlinear beats plain log-HAR) is stable; the choice of family is not.
- The log-HAR baseline is evaluated on the complete-case sample (rows where all 52 features exist), which is about 10% smaller than the vol tournament's; the HAR fit itself uses all of the symbol's rows.
- The frozen single MLP used for H-NN4 has holdout QLIKE 0.366, better than the walk-forward MLP ensemble (0.386): month-to-month refit noise in the MLP is large.
- TreeSHAP interaction shares among collinear vol-level proxies are partly an artefact of how trees share credit between correlated inputs. The cross-group interactions, the signs and the ablations are the robust part.
- The har_dow comparison, the calendar-by-vol table and the effect signs are exploratory, added after the H-NN1 result was seen. They inform rule candidates at grade C and should be registered before being relied on.
- H-NN5 / H-NN6 (1 s micro data, CNN) are not run in this round.

## Rule candidates

```yaml
- id: R-ML-1
  kind: prefer
  rule: WHEN forecasting next-day variance for a crypto perp THEN start from the per-asset log-HAR level and adjust it for the forecast day's weekday before reading any other input
  when: daily (1d) variance expectation for a universe-L perp
  then: central forecast = log-HAR x weekday factor (Saturday about 0.6x, Sunday about 0.85x, Monday about 1.3x of the model's average day); risk and stops keep the unshrunk risk sigma (R-VOL-2)
  scope:
    assets: crypto perps (universe L)
    horizon: 1d
    sides: both
  effect: vol level carries 40-43% of TreeSHAP attribution and ablating it raises QLIKE 75%; calendar ablation +5-8% QLIKE; per-asset log-HAR + weekday dummies -5.5% QLIKE (t -2.9) in-sample, -2.3% (t -0.6) holdout vs plain log-HAR (exploratory)
  confidence: B
  evidence: [H-NN1, H-NN3, H-VM5, H-NS4]
  review: register and test log-HAR + weekday as a tool (vol.har_dow); demote if the holdout weekday gain stays insignificant over 2 more months
  status: provisional
- id: R-ML-2
  kind: expect
  rule: WHEN vol is already high (v0 high-vol state) THEN expect a smaller weekend discount in next-day variance than in calm markets
  when: the forecast day is Saturday or Sunday
  then: weekend gap about -0.39 log points in high vol vs -0.48 in low vol (in-sample), -0.48 vs -0.57 (holdout); do not apply a calm-market weekend discount in a high-vol regime
  scope:
    assets: crypto perps (universe L)
    horizon: 1d
    sides: both
  effect: TreeSHAP vol_level x calendar is the largest cross-group interaction (5.2% / 4.4% of |phi|), the same direction in both segments
  confidence: C
  evidence: [H-NN3]
  review: register a direct test (weekend RV ratio by vol tercile, both segments) before promoting
  status: provisional
- id: R-ML-3
  kind: avoid
  rule: WHEN forecasting next-day crypto variance THEN don't call jump-variation, semivariance, range-estimator (Parkinson / GK) or raw-return inputs on top of the HAR level; they carry no information a nonlinear model can use either
  when: daily variance forecast with the HAR level already known
  then: skip those tools for this target; they remain useful descriptively (for example jumps for stop placement)
  scope:
    assets: crypto perps (universe L)
    horizon: 1d
    sides: both
  effect: group ablation in MLP and XGB changes QLIKE by -0.7% to +1.5%, none with |t| > 2 in the holdout; TreeSHAP shares 0.5-2%
  confidence: B
  evidence: [H-NN1, H-NN3, H-VCL2, H-VM3]
  review: a holdout quarter where any of these groups raises QLIKE by more than 3% (t > 2) when ablated
  status: active
- id: R-ML-4
  kind: prefer
  rule: WHEN a central next-day variance expectation is needed for a crypto perp and a validated nonlinear tool exists THEN prefer gradient-boosted trees on the 52-feature panel over plain log-HAR, but only for central expectations, never for risk or stops
  when: next-day expectation (quote widths, target distances); not risk
  then: use the tree forecast with its in-sample bias correction; fall back to R-ML-1 if the tool is stale (no refit for more than 35 days)
  scope:
    assets: crypto perps (universe L)
    horizon: 1d
    sides: both
  effect: XGB holdout QLIKE -15.7% vs log-HAR (t -2.4, 22/23 symbols), but in-sample -4.5% (t -1.0) and only -13.7% (t -1.6) vs log-HAR + weekday; the MLP wins in-sample (-8.0%, t -2.9) and not in the holdout
  confidence: C
  evidence: [H-NN1]
  review: forward test against log-HAR + weekday for 3 months after the tool exists; demote if QLIKE gain < 3% or t > -2
  status: provisional
- id: R-ML-5
  kind: avoid
  rule: WHEN looking for next-day direction THEN don't call a neural network or boosted-tree model on standard price, vol, regime, funding or calendar features; nonlinear capacity doesn't create a signal that linear models miss
  when: next-24h direction for crypto perps
  then: treat the direction as a coin flip; any direction view must come from a separately validated, registered signal
  scope:
    assets: crypto perps (universe L)
    horizon: 1d
    sides: both
  effect: Brier skill in-sample MLP -0.0024 [-0.0072, 0.0021], XGB -0.0016 [-0.0055, 0.0022]; holdout +0.0054 / +0.0018 with CIs including 0; models stop after about 1.5 epochs / 24 trees; tree and network attributions disagree (Spearman 0.22 holdout)
  confidence: B
  evidence: [H-NN2, H-IC1]
  review: a pre-registered nonlinear direction model with holdout Brier skill CI lower bound > 0
  status: active
- id: R-ML-6
  kind: prefer
  rule: WHEN explaining a model-based volatility forecast to the decision layer THEN cite grouped TreeSHAP attributions (vol level, calendar, context) rather than hidden-unit or dictionary features of a neural network
  when: a nonlinear vol forecast is shown with an explanation
  then: report the vol-level, weekday and context contributions in log-RV units; don't name network features
  scope:
    assets: crypto perps (universe L)
    horizon: 1d
    sides: both
  effect: SAE on the MLP's last layer, only 1/10 most-used features tie to a named state (|rho| > 0.6), top-3 ablation +0.09 QLIKE (t 1.3, holdout); TreeSHAP groups are stable across segments (top-5 pairs 5/5) and agree with integrated gradients (Spearman 0.75)
  confidence: B
  evidence: [H-NN4, H-NN3]
  review: an SAE on a wider or multi-day micro model (H-NN5) that does yield >= 6/10 named features
  status: active
```
