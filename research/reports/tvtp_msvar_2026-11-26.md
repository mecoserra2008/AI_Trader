# TVTP Markov-switching VAR: review, validation, gold/silver and crypto lead-lag (2026-11-26)

Agent TVTP. Methodology and original code by the user (`tvtp_msvar.py` at the repo root, left unchanged and untracked). Reviewed copy: `src/ai_trader/research/tvtp_msvar.py`. Drivers: `tvtp_mc.py` (synthetic), `tvtp_study.py` (real data). Tests: `tests/test_tvtp_msvar.py` (26). Outputs: `data/research/tvtp/`. Pre-registration: `research/hypotheses.yaml` block TVTP (H-TVTP0..6c, plus H-TVTP1s registered post hoc), frozen in the diary before any real-data fit. Research only: nothing here touches orders, accounts or the user's XAU / XAG positions, and nothing here is advice on them.

## In plain words

**Does the methodology work as coded?** Yes, on its own terms. The EM, Hamilton filter, Kim smoother, TVTP logit M-step, marginal effects and implied cross-correlations are all correct. I checked each against an independent reference: the log-space original, finite differences, and a 200,000-row simulation. On the user's own two-regime synthetic check, the model recovers the lag matrices, covariances and the sign of every driver coefficient. Its out-of-sample regime accuracy is 0.947, against 0.948 for an oracle that knows the true parameters. With three regimes it works up to the limit of what the data identifies: accuracy is 0.69 against an oracle of 0.71.

**What was wrong and what was fixed:**

- **Look-ahead (the important one).** Smoothed probabilities, the in-sample predictive log score and the CUSUM all used parameters fit on the whole sample, and the smoothed probabilities also used future data. The copy adds a causal walk-forward. Parameters are fit on rows before each origin, the filtered one-step regime forecast is used, and regimes are aligned across refits by Hungarian matching.
- **Wrong error-correction value in the forecast.** `expected_corr_next` used ect at T-1 to forecast T+1. It now requires ect at T explicitly.
- **Degenerate regimes.** A regime with near-zero weight got a near-zero covariance, which makes the likelihood unbounded. The copy shrinks it toward the pooled covariance and adds a Cholesky jitter ladder.
- **Bookkeeping and speed.** The reported log-likelihood belonged to the previous iteration, and the BIC missed the initial-state parameters. The filter now runs in scaled probability space, 29x faster with identical numbers.
- **Additions.** A Student-t option, out-of-sample CUSUM, CCF stability check, Wald SEs and simulation.

**Does TVTP add predictive value over simpler models?** On gold/silver hourly, yes, but much less than the headline suggests.

- Against plain and rolling VARs it wins by about 0.31 nats per hour. Most of that is the weekend: Bybit's XAU/XAG perps trade 24/7, and on weekends returns shrink to about 0.16 of their EWMA scale. One of the three regimes is simply "the weekend".
- A VAR that just rescales by hour-of-week volatility closes about three quarters of the gap. On weekdays the regime model is no better than that seasonal VAR (-0.006 nats/row, t -0.6).
- What the time-varying transitions themselves add over constant transitions is small but significant out of sample: +0.0075 nats/row (t 2.9) in the clean holdout, +0.0085 on weekdays.
- Its covariance forecasts beat rolling, EWMA and DCC by a wide margin (QLIKE 0.24 vs 1.14–1.19 in the holdout), again mostly because the baselines cannot see the weekend collapse.
- Daily data is too short to say anything (148 test days).
- At 1-minute resolution, both for gold/silver and for BTC to ETH/SOL/DOGE, the TVTP drivers make forecasts *worse* than constant transitions (t -2.7 to -6.7).

**The gold/silver "lead-lag regimes" found.** There is no leadership switching at 1h or 1m. The three hourly regimes are session and volatility states:

| Regime | Scale | Correlation | Duration |
|---|---|---|---|
| Weekend | ~0.16 | 0.4–0.57 | 60–90 h |
| Normal | ~0.55 | 0.80 | 6–10 h |
| Event / high-vol | ~1.2–1.4 | 0.85 | 4–6 h |

First-lag cross-correlation asymmetries between gold and silver are below 0.04 after June in every regime.

**What drives switching.**
- The price-based realised-vol state is a stable, significant driver in both subperiods. A higher gold vol state lowers the hazard of leaving the normal regime (about -0.085 per sd in both subperiods) and the weekend regime (-0.03 to -0.05).
- Real yields (DFII10) matter only in the holdout, with opposite signs in two regimes and nothing in-sample.
- The broad dollar and breakevens never matter.
- Adding the macro drivers to the hourly model lowers its out-of-sample log score.

**Is any rule executable after costs?** No.

- **Gold/silver 1h.** The pre-registered lead-lag rule never found a threshold that was net positive on fit data, in any fold, so it never traded. Every fixed threshold loses 0.26–0.36 R per trade in the holdout, on a gross edge of about zero.
- **Gold/silver 1m.** There is a small real gross edge (+0.03 to +0.04 R per trade, t 2–3), but costs are about 50 times larger (net -2.2 R).
- **Crypto 1m.** BTC does lead SOL and DOGE inside 2-minute burst regimes. But the rule's gross edge is ≤ 0 even at zero latency, and taker costs make it -1.2 to -1.4 R per trade. Here costs are not even the binding problem: on 1-minute bars there is no gross edge to give up.

## Verdicts (recorded in research/hypotheses_results.jsonl)

| id | segment | verdict | key numbers |
|---|---|---|---|
| H-TVTP0 synthetic recovery | pooled | inconclusive | K=2 passes all criteria; K=3 accuracy 0.69 (oracle 0.71, the 0.85 bar was unattainable), CCF error 0.070 > 0.05 |
| H-TVTP1 log score M1h | llm_in_sample | inconclusive | vs CTP +0.0055 (t 1.26), vs VAR +0.339 (t 8.2), vs RVAR +0.350 (t 8.2) |
| H-TVTP1 | clean_holdout | supports | vs CTP +0.0075 (t 2.89), vs VAR +0.308 (t 11.4), vs RVAR +0.317 (t 11.4) |
| H-TVTP1s session-aware baseline (post hoc) | clean_holdout | supports | vs SVAR +0.080 (t 4.45) overall: weekday -0.006 (t -0.58), weekend +0.280 (t 6.1); TVTP-CTP weekday +0.0085 (t 2.89) |
| H-TVTP1d daily | clean_holdout | inconclusive | 148 rows; vs CTP +0.013 (t 0.18), vs VAR -0.044 (t -0.56) |
| H-TVTP2 covariance | llm_in_sample | supports | QLIKE 0.553 vs rolling 1.335 / EWMA 1.473 / DCC 1.517 (t -2.4 to -3.2) |
| H-TVTP2 | clean_holdout | supports | QLIKE 0.240 vs CTP 0.325 (t -3.4), rolling 1.194, EWMA 1.138, DCC 1.152 (t -8.8 to -10.8); cov-element MSE lower (t -6 to -7.5) |
| H-TVTP3 lead-lag rule 1h | clean_holdout | inconclusive | zero trades selected; fixed theta net -0.26..-0.36 R (t <= -9.7), gross -0.02..+0.03 R |
| H-TVTP3b rule 1m metals | clean_holdout | inconclusive | zero trades selected; fixed theta 0.5: gross +0.041 R (t 3.2) at 0 latency, +0.030 at 3 s, net -2.19 R |
| H-TVTP4 macro drivers 1h | clean_holdout | inconclusive | DFII10 CIs exclude 0 only in holdout, opposite signs (-0.078 / +0.057); DTWEXBGS never |
| H-TVTP4d macro drivers daily | pooled | contradicts | all CIs include 0 (about 240 rows, weak by construction) |
| H-TVTP5 CUSUM | llm_in_sample | inconclusive | 5 episodes: hit 0.40 vs 0.20, false 8.5 vs 12.8 / 1000, binomial p 0.26 |
| H-TVTP5 | clean_holdout | inconclusive | 13 episodes: hit 0.31 both, delay 1.25 h vs 19 h, false 7.4 vs 10.5 / 1000, binomial p vs chance 0.17 |
| H-TVTP6a crypto log score | clean_holdout | contradicts | TVTP-CTP: ETH -0.0037 (t -3.3), SOL -0.0095 (t -6.7), DOGE -0.0133 (t -5.9) |
| H-TVTP6b crypto rule | clean_holdout | inconclusive | zero trades selected; fixed theta taker net -1.17..-1.40 R, gross <= 0 at zero latency (DOGE -0.053 R, t -2.7) |
| H-TVTP6c cross-venue | clean_holdout | inconclusive | not testable as registered (877 design rows, no test rows); no substitute run |

Trials: 13 registered hypotheses + 1 post hoc, 8 specs on M1h, 4 per 1m dataset, 5 thresholds per rule (DSR used 40 / 20 trials; moot, since no rule traded).

## Rule candidates (suggestions only; research/rules.yaml not edited)

1. **Weekend scale for metals perps (risk, not signal).** On Bybit XAUUSDT/XAGUSDT, weekend hourly returns run at about 0.16 of the EWMA sigma and the regime lasts 60–90 h. Any sigma-based sizing or stop distance on these perps that uses a plain EWMA over-states weekend risk and lags the Monday reopen. Candidate: an hour-of-week scale factor (as in SVAR) or the TVTP mixture covariance for metals risk estimates. Evidence: H-TVTP1s, H-TVTP2, both segments.
2. **No gold/silver or BTC to alt lead-lag trading on bars of 1 minute or longer.** No leadership switching was found at 1h or 1m. Where a gross edge exists (metals 1m) it is about 1/50 of costs. Crypto lead-lag at 1m has no gross edge after the bar closes.
3. **Realised-vol state as the switching driver.** If a regime-switching model is used for metals at all, use the price-based rv state (stable in both subperiods). Macro daily drivers (real yields, dollar, breakevens) did not help and lowered the log score.

## Caveats

- **Short history.** Bybit XAU/XAG starts 2026-03-09. The cTrader read-only MCP (127.0.0.1:9876) refused connections, so there is no longer daily history and the daily tests are underpowered. Re-run M1d / H-TVTP4d when cTrader daily trendbars are available.
- **Spread measurement.** Spreads come from flow_3s medians for 2026-11-07..26 (XAU 0.026 bp, XAG 2.09 bp) and are applied to the whole hourly sample.
- **1m sample and data deviations.** The 1m data covers about 8.9 days of collector coverage, split by gaps of up to 56 h. Alt OI exists only at 4h in the lake, so it was used instead of the registered 1h (diary mistake entry).
- **One spec selected once.** The spec was chosen once on pre-origin data. K=3 t5 was chosen by validation log score, whereas BIC would have picked K=2 t5.
- **Session structure drives the main result.** The H-TVTP1/H-TVTP2 "supports" are dominated by session structure. They say the mixture density is a good density forecaster, not that there are leadership regimes. H-TVTP1s was registered after seeing H-TVTP1 and is an extra trial.
- **Zero-trade verdicts.** With zero selected trades, the trading verdicts are inconclusive by rule. Read the fixed-threshold variants for the economics.
- **CCF meaning.** The implied CCF assumes the regime persists forever. With 2–6 h regimes, it overstates within-regime dynamics.
- **Standard errors.** Wald SEs on the TVTP coefficients under-cover badly (0.46–0.84 coverage at 95%). Use the bootstrap.
- **Bootstrap limits.** The parametric bootstrap holds z and ect at observed paths and uses 30 reps, so its CIs are rough.

## Method

**Model.** The model is the user's: K-regime VAR(p) with regime-specific intercepts, lags, error-correction loading and covariance. Transitions are softmax(W[i]·[1, z_{t-1}]) with reference category K-1. EM uses L2 = 1.0 on W, covariance shrinkage of 2 pseudo-observations, 3 random block starts plus a warm start per refit, max_iter 200 and tol 1e-6.

**Fixes F1–F13.** These are listed in the module docstring, one regression test each:

- **F1** causal walk_forward / filter_design
- **F2** exog_now
- **F3** Hungarian alignment and W permutation with reference re-normalisation
- **F4** covariance prior
- **F5** Cholesky jitter
- **F6** block initialisation (the original Dirichlet init gave the same accuracy in 10-rep comparisons)
- **F7** log-likelihood at the final parameters; BIC adds K-1
- **F8** K=1 and Z=None baselines
- **F9** out-of-sample CUSUM with burn-in and reset
- **F10** CCF stability
- **F11** scaled filter
- **F12** Student-t (ECM)
- **F13** Wald SEs and simulation

With init=dirichlet and no covariance prior, the copy reproduces the original exactly: ll -10592.16 and identical parameters.

**Synthetic validation (tvtp_mc.py).** 50 reps for each K in {2, 3} and T in {1000, 4000}; data/research/tvtp/mc_summary.json, mc_oracle.json, mc_cusum.json.

| | out-of-sample acc. | oracle | smoothed (in-sample) | A RMSE | Sigma RMSE | W RMSE | slope sign | Wald cover | CCF err (median) | sec |
|---|---|---|---|---|---|---|---|---|---|---|
| K=2 T=1000 | 0.947 | 0.952 | 0.959 | 0.04–0.085 | 0.05–0.07 | 0.65–1.0 | 100% | 0.84–0.90 | 0.074 | 3 |
| K=2 T=4000 | 0.947 | 0.948 | 0.965 | 0.02–0.036 | 0.022–0.037 | 0.33–0.72 | 100% | 0.74–0.84 | 0.037 | 8 |
| K=3 T=1000 | 0.647 | 0.716 | 0.734 | 0.07–0.125 | 0.08–0.19 | 0.9–3.0 | 78–88% | 0.40–0.58 | 0.159 | 12 |
| K=3 T=4000 | 0.693 | 0.709 | 0.783 | 0.02–0.048 | 0.04–0.076 | 0.37–0.80 | 98–100% | 0.46–0.68 | 0.070 | 31 |

- Bias is within ±0.03 on A and Sigma everywhere.
- No degenerate fits; all runs converged.
- At K=3 the start-to-start log-likelihood spread averaged 3.7, so multiple starts are needed.
- Synthetic CUSUM: an unmodelled correlation-flip regime was injected for 200 rows, 30 reps. CUSUM hit 100% with a median delay of 2.5 rows and 3.6 false alarms per 1000 rows. The rolling-correlation detector hit 97% with a delay of 19 rows and 5.1 false alarms per 1000.

**Data.**
- **M1h:** Bybit 1h last-price XAUUSDT / XAGUSDT, 2026-03-09..11-25, from the lake.
- **M1d:** Bybit 1d.
- **Macro:** FRED DFII10 / DTWEXBGS / T10YIE (data/research/macro, known_ts = obs + 1 day; d5 changes, expanding z-scores since 2003).
- **1m:** flow_3s 3 s candles rolled to 1m mids. The latency fill is the mid of the first 3 s candle after the close (about 3 s, more conservative than 150–300 ms). Measured spreads come from the same candles.
- **Crypto drivers:** Bybit funding and OI from the lake (known_ts ≤ t).
- **xvenue:** data/journal/xvenue snapshots.

**Features.**
- y = log return / EWMA sigma_{t|t-1} (halflife 24 h / 10 d / 60 m), clipped at ±8 for the model; P&L uses unclipped values.
- ect: rolling 336-row OLS of log XAU on log XAG (expanding for daily), z-scored by residual sd.
- rv: trailing z-score of the next-bar log EWMA vol.
- Design rows never cross a data gap.

**Walk-forward.**
- M1h: expanding, first origin 2026-05-01, refit every 336 rows, 15 folds; test rows 1464 in llm_in_sample and 3552 in clean_holdout.
- M1d: from 2026-07-01, refit every 14 rows.
- 1m: rolling 4320-row window, daily refits, 6 folds of about 8100 test rows.
- Baselines: CTP (z dropped), VAR (K=1), RVAR (K=1 rolling: 720 / 60 / 1440 rows), and SVAR (hour-of-week-scaled t5 VAR, Jacobian-adjusted).
- Covariance baselines: rolling 120, EWMA 0.97, and DCC(1,1) by QMLE on each fold's fit rows.
- Tests: Diebold-Mariano with Newey-West HAC t on per-row differences.
- **Spec grid:** K {2,3} × p {1,2} × {Gaussian, t5}, chosen by validation log score on the last 30% of pre-origin rows.

| K, p, dist | validation log score | BIC |
|---|---|---|
| 2,1,G | -1.861 | 2067 |
| 2,1,t5 | -1.681 | 1873 |
| 2,2,G | -1.877 | 2091 |
| 2,2,t5 | -1.699 | 1893 |
| 3,1,G | -1.725 | 1969 |
| **3,1,t5** | **-1.659** | 1949 |
| 3,2,G | -1.726 | 2036 |
| 3,2,t5 | -1.670 | 1997 |

**Trading rule.**
- Leader regimes come from the first-lag CCF asymmetry of each fold's fit.
- Trade the lagger for one bar in the sign of A[lagger, leader] × the leader's last move when the predictive regime probability exceeds theta.
- Theta is chosen per fold on fit data, with the last fit row purged.
- Net R = sign × r / sigma_{t+1|t} − cost / sigma.
- Costs:
  - desk (pegged entry 2.0 + 1.3 bp, taker exit 5.5 bp + half spread): XAG 9.85 bp, XAU 8.8 bp
  - taker both sides: 11 bp + spread
  - crypto taker: 11 bp + measured half spreads
- Comparators: a regime-free pooled-VAR lead-lag rule and random signs (2000 draws). Neither was evaluated in the end, because no rule traded.

**CUSUM.**
- Settings: burn 168, halflife 168, kappa 0.5, h 5, refractory 24.
- Input: walk-forward TVTP log scores.
- Comparison detector: |Fisher z(corr_24) − Fisher z(corr_168)| / sqrt(1/21 + 1/165) > 3.
- Episodes: CPI / NFP / FOMC from data/research/news/events (24 h windows).

**Runtime and storage.**

| Job | Runtime |
|---|---|
| Monte Carlo | 53 min of fits (240 fits) |
| M1h | 17 min |
| M1h-macro walk-forward | 12 min |
| H-TVTP4 bootstrap | 15 min |
| M1d | 1 min |
| 1m runs | about 5 min per pair, run twice |
| xvenue | 1 min |

At most 2 processes, 1 BLAS thread each. data/research/tvtp is 1.5 MB. The lake, research and raw total about 14.7 GB, under the 30 GB budget.
