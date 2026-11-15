# Working beliefs (v1, 2026-11-07)

_What the evidence currently supports, how confident we are, and how each belief is used. Every line traces to [hypotheses.yaml](hypotheses.yaml) + [SCOREBOARD](reports/SCOREBOARD.md). Beliefs change only through registered tests, and holdout data after 2026-07-01 is the only evidence Claude can't have memorised._

Confidence:
- **A**: supported both before and after the cutoff, large effect;
- **B**: supported but small effect, or forward-confirmed once;
- **C**: plausible, forward test running.

## 1. Volatility: what we believe and how we use it

| # | belief | conf. | evidence | how it's used |
|---|---|---|---|---|
| V1 | Weekends (Fri 17:00 → Sun 18:00 ET) are quiet: realised vol ≈ 0.6–0.8× the Friday σ̂ | A | H-W1 | central σ is weekend-scaled (`sigma_rb_next`, m ≈ 0.57 for BTC) |
| V2 | Weekend realised variance is far below what a 7d ATM implied vol charges per calendar hour… | A | H-W7 (t −17 / −7.7) | descriptive truth, but **not harvestable as tested**: Friday 3-day options already price the weekend ~20% cheaper, the Fri 08:00 → Mon 08:00 window includes the loud Friday US session and Sunday reopen, and hedging costs exceed the edge (H-OPT1–3 failed both segments). Only a US-close entry with band hedging remains: H-OPT4, forward-only |
| V3 | …but weekends are jump-prone: shrinking σ for *risk* worsens tail loss | A | H-V2 failed on QLIKE | stops and sizing use `sigma_risk_next` (no weekend shrink) |
| V4 | Jumps decay fast (excess-variance half-life 2–3 h); the plain EWMA over-forecasts post-jump vol by ~27% | A | H-S4 | jump-robust, bias-corrected EWMA (`vol.robust_bc`, H-V1b, QLIKE −0.05 both segments) |
| V5 | Vol clustering is multi-scale: HAR (1d / 7d / 30d) beats the single-λ EWMA at 24h | A | H-VCL1 (QLIKE −0.08 both segments) | daily risk forecast = `vol.har`; per-bar σ = `vol.robust_bc` |
| V6 | Vector candles and Amihud illiquidity spikes do *not* predict extra vol beyond the robust forecast | A (negative) | H-VC4, H-LC2 | no special vol regime after them |
| V7 | Implied vol (DVOL) adds a little to HAR for 7d BTC/ETH variance | B | H-VCL4 (−3% in-sample weak, −19% holdout) | add DVOL as a feature in BTC/ETH risk sizing |
| V8 | No leverage-effect asymmetry, no BTC→alt spillover beyond own HAR, no compression-expansion edge | A (negative) | H-VCL2, H-VCL3, H-VCL5 | HAR stays plain |
| V9 | **High-vol regimes over-forecast next-week vol**: in the v0 high-vol state, next-7d realised vol comes in ~13–31% further below the EWMA forecast than in low-vol states, crypto and TradFi alike | A | H-RL2 (t −10.9 in-sample, −5.0 holdout, 9 assets) | don't size or quote off plain EWMA after spikes (use robust_bc / HAR); vol-selling / liquidity provision after spikes has the wind at its back; tails still apply (V3) |
| V10 | Monday variance: BTC Mondays ~1.3× Tue–Fri in-sample (CI 1.01–1.68) but not in the holdout (0.82); SPX/NDX Mondays are *not* more volatile per trading day (0.93–0.97×) though the weekend gap carries 27% of Monday's S&P variance vs 17% on other days; TLT/DXY Mondays are quieter (0.72–0.80×); WTI Mondays 2.3× in the holdout (war weekends) | B / C | H-DOW2 (inconclusive / contradicts) | widen risk at the Monday US open and the Sunday CME reopen for oil; no blanket Monday vol multiplier |

## 2. Price behaviour

| # | belief | conf. | evidence | how it's used |
|---|---|---|---|---|
| P1 | **Shocks into calm markets continue; shocks into volatile markets reverse** (vol state at shock time decides) | B / C | H-S2 forward-supported; H-S3 forward test from 2026-11-08 | **first strategy to forward-test: `shock_regime`** |
| P2 | Hourly moves away from trend keep going; fading 2σ hourly deviations loses even before costs | B | H-T1 forward-supported | mean-reversion strategies stay off at intraday horizons |
| P3 | Slow trend has a weak edge (Sharpe ≈ 0.2–0.3 before the cutoff), negative in Jul–Nov 2026; 4h > 1h, 1d not better | B / C | H-X2b, H-X2c forward test | trend is a low-weight baseline sleeve, not a core edge |
| P4 | Structural "levels" (vector-candle opens and closes, DC swing pivots, CME gaps, swept zones) do **not** hold more than a random walk; price goes *through* them slightly more often (break ≈ 0.52–0.55) | A (negative) | H-VZ1–4, H-VC2/3/7, H-GT3, H-W4 | level-chain prior: π_break ≥ 0.5 by default; levels are where to *expect* the move to continue, not to fade it |
| P7 | Simple trend models (21-variant zoo) earned thin edges before the cutoff (best tsmom_14d +0.50, ensemble +0.33; deflated Sharpe of the winner 0.72 < 0.95) and are negative across the board Jul–Nov 2026; walk-forward model selection loses to the ensemble; 'is it trending?' isn't forecastable (H-TR3/H-TR4); zoo is ~1.9 independent bets | A (negative for deployment) | H-TR1–4 | trend stays paper-only (ensemble tracked forward); net-E[R] gate stays at 0.10 |
| P6 | Neither auction balance (value-area overlap) nor the variance ratio says when to trend-follow vs mean-revert; value-area / POC levels behave like random-walk levels | A (negative) | H-AU1–4, H-RG1 | no regime switch on these; 1h MR stays off everywhere |
| P8 | **No Monday drift** in any of 9 assets (crypto and TradFi, Holm-adjusted); regime cells carry **no directional information** about the next week (up − down 0.03σ, t 1.0, 37k obs) | A (negative) | H-DOW1, H-RL1 | no day-of-week or regime-direction tilts |
| P9 | After a < −1σ week, TradFi Mondays lean up (+0.12σ pooled, t 2.3; SPX +0.24σ, TLT +0.14σ); crypto shows nothing (−0.08σ, t −0.4) | C | H-DOW3 in-sample only, contaminated (classic literature), ~6 events a year, holdout too small | context only; not deployable |
| L1 | Regime lifecycle (daily v0 cells, 9 assets): median episode 5–7 days, daily stay probability 0.84–0.89; older episodes are more likely to end (duration dependence, z 4.2 in-sample, +0.18 holdout z 1.5; partly mechanical from the 3-bar hysteresis). Typical exits: R7 grind-down → R4 compression 46% / R8 28%; R9 liquidation → R6 whipsaw 62% / R3 18% / R8 16% | B | H-RL3 + descriptive transition matrix | describes the *path of the state* (after a liquidation regime, expect high-vol chop), not returns |
| P5 | Weekend direction isn't predictable from price/volume; Bybit TradFi perps freeze on weekends and carry no information | A (negative) | H-W2/3/6 | no weekend directional trades; Monday NYSE-open vol is higher after eventful weekends (H-W5b, in-sample) → widen risk there |

## 3. Things that don't work here (don't spend capital on them)

| idea | evidence |
|---|---|
| funding contrarian, OI-price divergence as standalone 4h signals | H-P1, H-P2 |
| market-profile rules (acceptance, 80% rule, POC magnet), variance-ratio regime switch | H-AU1–4, H-RG1 |
| relative value at taker costs: alt residuals vs BTC/ETH, Brent–WTI spread, XAUT–XAU basis (reverts ~28%/24h but sd 9–14 bps vs 30 bps costs) | H-RV1–3 |
| stablecoin growth, VRP as return predictors | H-FU1, H-FU2 |
| Fear & Greed contrarian | prior project (no effect) |
| pivot-geometry trend (STRUCT / PIVOT_BO / FLAG_BO) | H-GT1/2: a noisier copy of slow trend |
| vector-candle continuation, zone revisits, zone S/R | H-VC1–5, H-VZ1–4 |
| close-confirmed stops instead of resting stops | H-X1 (worst −1.9R vs −1.0R) |
| continuous-signal trend at 1h | costs eat the whole gross edge |

## 3b. Carry

- **C1: funding carry is real but regime-dependent and has decayed** (H-C1: net Sharpe 2.8 in-sample, but 57%/yr in 2021 → 0.5% in 2025 → −4.8% in 2026 H1). It's a leverage-mania premium, not a standing one. Forward test: H-C1b, an always-on overlay scaled by trailing 30d funding.

## 4. Execution and risk beliefs

- **Keep resting stops** (Bybit position SL, MarkPrice trigger) (H-X1).
- **Size on `sigma_risk_next`** (robust_bc, no weekend shrink); daily risk budget from HAR.
- **Open design conflict:** structural stops for breakout / structure strategies sit a median **6–8σ** from entry (donchian, FLAG_BO), but `placement.max_stop_distance_sigma = 4`. Either raise the cap (the position just gets smaller; risk per trade stays fixed) or don't run structure-stop strategies. **Decision needed from the user.**
- Order flow: as **entry timing**, not standalone alpha (prior project + H-F5). Untested until ≥ 30 days of `flow_3s`.

## 5. What to exploit first (initial forward-test book)

| priority | sleeve | status | what gets measured forward |
|---|---|---|---|
| 1 | `shock_regime` (P1), robust σ sizing, resting stops | built; forward from 2026-11-08 | H-S3 + strategy P&L vs RANDOM_N on the same shock events |
| 2 | Weekend vol (V2): short-vol study on Bybit BTC/ETH options (weekly, Friday entry, delta-hedged) | to research | premium vs weekend jump losses, net of option spreads |
| 3 | 4h trend baseline (ema_stack / donchian), small weight | built | H-X2c; correlation with sleeve 1 |
| 4 | Order-flow timing for entries | collecting data | H-F1–F5 after ≥ 30 days |

**The LLM's role on top of this book:** read the state cards (regime cell, robust and HAR σ, weekend context, shock context, flow), and express bounded views on these candidates only. A belief changes only through a registered test.

## 6. v2 additions (2026-11-10, program research/plans/2026-11-10_volatility_program.md)

_Each line is generalised into a rule in [rules.yaml](rules.yaml) / [RULES.md](RULES.md), the form the decision layer loads._

| # | belief | conf. | evidence | rule |
|---|---|---|---|---|
| V11 | **Crypto has no leverage effect**: down days don't raise next-day vol more than up days of the same RV (0/23 GJR gammas significant; panel SHAR diff t −0.28) | A | H-LEV2 | R-LEV-1 |
| V12 | TradFi equity perps show **no textbook leverage effect** on Bybit (holdout, 132 days; sign flips with spec); drawdowns create no asymmetry; crowded funding doesn't amplify down days | C / B | H-LEV1, H-LEV3, H-LEV5 | R-LEV-2/3/5 |
| V13 | **Best daily vol model for crypto = per-asset log-HAR** (QLIKE −18% vs robust_bc scaled to a day, both segments); 5d = direct panel log-HAR; never scale the 1h σ flat to multi-day | B | H-VM2, H-VM5, H-VM1 | R-VOL-20/21 |
| V14 | Short-history and TradFi perps: **panel HAR pooled across assets** (HAR-Q level form) | C | H-VM1, H-VM5 | R-VOL-22/23 |
| V15 | Range estimators from 1h bars (GK, Parkinson) are **worse** HAR inputs than 1h RV (+5% / +16% QLIKE) | A (negative) | H-VM3 | R-VOL-24 |
| V16 | Alt vol is mostly one crypto vol factor (median R² 0.7) but the factor doesn't improve forecasts | B | H-VM4 | R-VOL-25 |
| V17 | All models over-forecast Saturdays (crypto 0.5×, TradFi 0.05–0.1× Tue–Fri RV): expectations only, never risk (V3) | C (forward H-VM6) | tournament diagnostic | — |
| P10 | Vol-based signals carry no direction: variance-managed exposure, vol-surprise continuation, squeeze breakouts, Hawkes exceedance clustering all fail | B (negative) | H-VB1–4 | R-SIG-1..4 |
| P11 | Floor pivots, Market Cipher B dots, Kalman trend "trendiness" carry no edge; the adaptive Kalman gain is at most a tie-breaker | A / B (negative) | H-BO1, H-MC1, H-KF1, H-KF2 | R-PX-10/11, R-AVOID-3, R-SIG-5 |
| F1 | Order flow (first 3 days): `|z| > 4` imbalance is an activity gauge, not a vol or direction signal (fires on 22% of minutes); no BTC lead; liquidation clusters did **not** revert (continued, forward test H-OF7); a selective top-1% detector is the lead (H-OF6) | C | H-OF1–5 | R-FLOW-1..4 |
| D1 | **Decision budget**: next-day direction from standard features ≈ coin flip and more inputs make it worse → show a short card (≤ ~8 tested lines); fit on 1–2y or expanding history (90d windows hurt); 4h trend tolerates hours of latency, shock signals decay in ~1–2 h (execute at bar close, LLM veto only) | B / C | H-IC1–3 | R-BUDGET-1..4 |
| H1 | Funding crowding alone tilts next-day vol up (+0.07× mean RV per z, t 2.4 / 2.7) | C (forward H-LEV6) | exploratory in H-LEV5 | — |

## 7. v3 additions (2026-11-11, Program 2: research/plans/2026-11-11_strategy_program.md)

| # | belief | conf. | evidence | rule |
|---|---|---|---|---|
| S1 | **The regime map does not pick winners; it only tells where strategies lose.** Per-cell rankings don't persist across the cutoff (Spearman 0.085, CI incl. 0); every family x cell graded A/B is negative (e.g. 1h mean reversion in trend/compression/grind-down cells, 1h trend in range/whipsaw cells) | B | H-RC1, H-RC2 | R-MAP-1..7 |
| S2 | **The deterministic book is about one trend bet** (equal-weight book 1.15 effective bets in-sample, 2.5 holdout); the trend family = crypto TSMOM factor + long convexity; 1h mean reversion = short convexity; OI divergence = market beta; no residual alpha anywhere | A / B | H-PD1, H-PD2 | R-PD-1..5 |
| S3 | **Edge leaks through exits, not fees**: costs + exit leakage = 70-75% of the favourable excursion, costs only 5-9% of it | B | H-PD4 | R-PD-6 (forward work: H-PD6) |
| S4 | Context-conditional ranking (IPCA K=3) beats trailing Sharpe, mostly because strategy returns reverse week to week | C | H-PD3 | R-PD-7 (H-PD5 forward) |
| T1 | **Bybit equity perps trade 24/7** (overnight 0.25-0.5x turnover, weekends 0.1-0.15x; last within ~5 bp of index); classic session effects don't survive: no gap fill, no night premium, ORB loses net, no intraday momentum, no CME-reopen continuation; fading big overnight moves at the open loses (-67 bp net) | B / C | H-TP1, H-TF1-5 | R-TF-1..7, R-PX-9 (corrected: not frozen) |
| T2 | **Hour-of-week vol profile is real and large**: QLIKE -0.04/-0.08 crypto, -0.29/-0.38 TradFi, both segments; vol peaks Sunday 22-02 UTC and at the US cash open | A | H-NS4 | R-VOL-26/27 |
| T3 | No funding-settlement flow beyond generic funding drift; no expiry pin in weekly expiries; no session return seasonality; QQQ's first hour doesn't lead crypto | A / B (negative) | H-NS1, H-NS2, H-NS3, H-NS5 | R-CARRY-2, R-VOL-28, R-AVOID-4, R-PX-12 |
| G1 | **Per-symbol parameter selection is a coin flip** (PBO 0.44 zoo, 0.65 breakouts; breakout champions do worse than a random pick in the holdout); generated paths reshuffle the in-sample data and do not fix it; block bootstrap reproduces tails and clustering up to a day, regime-switching resampling does not | B | H-SYN1, H-SYN2 | R-SYN-1..3 |
| M1 | Simulator v0: fill model consistent with live fills (23 test fills, inconclusive); flow generator fails stylised facts (alts 0/5); trader types exist (momentum takers stable in crypto; takers hit the deeper side of the book; activity follows vol) but magnitudes don't transfer across markets; Kyle-Obizhaeva invariance holds for trade size (W^0.31) and count (W^0.66); MM parameters must be ranked on real 3s replays, not generated paths | C | H-SIM1-4 | R-SIM-1..7 (H-MM3, H-NS7 forward) |
| X1 | **Live quoter bug fixed 2026-11-11**: the 30-minute fast sigma always returned the floor (29 returns vs a 30-return minimum), so quotes never widened on 30-minute bursts (`sigma_last` in mm.rs, test added) | - | simulator study | - |

## 8. Microstructure pilot (one day, 2026-11-10, CPI day; all grade C until more days)

| # | belief | evidence | rule |
|---|---|---|---|
| M2 | Trade arrivals are bursty, never Poisson: a 3-component exponential mixture fits best (about 30% bursts of ~5 ms, 40% gaps of 0.2-0.4 s, 30% gaps of 1-5 s); ACD(1,1) leaves most of the excess and is unstable as a generator; an XGBoost AFT model beats ACD only for majors | H-MS1, H-MS9 | R-MS-1, R-MS-2, R-MS-7 |
| M3 | Order signs have long memory: lag-1 ACF 0.33-0.39, power-law decay (gamma 0.64 majors, 0.74 alts, 0.45 TradFi); Hurst of 1 s signed flow 0.53; merge same-ms same-side prints into orders (1.9-4 prints per order) | H-MS3 | R-MS-3, R-MS-4 |
| M4 | Sizes: log-normal body with a Pareto-type tail (index 1.76 alts, 1.84 majors, 1.39 TradFi), not a pure power law; volume per 10 s / 1 min is log-normal and co-moves with |return| (Spearman 0.40-0.55) | H-MS2, H-MS4 | R-MS-5, R-MS-6 |
| M5 | Most perps sit at one tick (39 of 51; 8 of 12 L2 symbols > 96% of the time), so the spread carries no vol information there; only free-spread perps (CARV, KAS, NVDA) showed a spread-to-future-vol gain (forward test H-MS5b) | H-MS5 | R-MS-11, R-SIM-12 |
| M6 | No sub-minute return lead from ETH, SOL or BTC (peaks within +/-20 ms); ETH, SOL and DOGE do transmit volatility (net +25 / +22 / +27); connectedness 67% -> 88% into the CPI afternoon | H-MS6 | R-MS-12, R-MS-13 |
| M7 | Population extrapolation works for activity, not for character: order rate (LOO R^2 0.85, ~W^0.54), depth (~W^0.68) and impact (~W^-0.75, close to Kyle-Obizhaeva) scale with trading activity W; sign memory, size tail and arrival mixture are persistent per-symbol traits (neither scaled nor universal) | H-MS7 | R-SIM-10, R-SIM-11 |
| M8 | Simulator v1 (fitted laws) beats v0 in every class (majors 3 vs 2, alts 4 vs 0, TradFi 3 vs 1 of 5 facts) and fixes signed-flow autocorrelation, but misses majors' impact and tick-bound spreads; a quiet-morning calibration does not describe a news afternoon; MM rankings v1 vs real replay 0.30 (v0 -0.09) | H-MS8 | R-SIM-13, R-SIM-14 |
| F2 | Fees: TradFi perps (NVDA, QQQ) have 0% maker / 2.75 bp taker on this account; crypto 2 bp maker / 5.5 bp taker (fee-rate API, 2026-11-11) | account data | — |

## 9. Overnight program (2026-11-12), running additions

| # | belief | conf. | evidence | rule |
|---|---|---|---|---|
| N1 | Nonlinear models (MLP / XGBoost) beat log-HAR for next-day crypto vol in both segments (MLP -8.0% QLIKE in-sample t -2.9; XGB -15.7% holdout t -2.4, 22/23 symbols), but the winning family flips between segments and most of the gain is calendar structure (log-HAR + weekday dummies gets -5.5% alone; ML beyond that not significant) | B | H-NN1 | R-ML-1, R-ML-4 |
| N2 | What vol models use: the vol level (7d log RV, robust sigma; ~40% of attribution, +75% QLIKE when ablated) and the forecast weekday (Sat ~0.6x, Mon ~1.3x); trend, funding, market add 1-3% each; jumps, semivariances, range estimators, raw returns add nothing; interactions are ~1/3 of attribution with identical top-5 pairs in both segments; trees and nets agree on vol drivers (rank corr 0.75) but not on direction (0.22) | B | H-NN3 | R-ML-3, R-ML-6 |
| N3 | No ML direction skill (Brier skill about 0); SAE on a 16-unit layer did not yield named concepts (1/10) - redo with a wider layer (H-NN9) | B (negative) / C | H-NN2, H-NN4 | R-ML-5 |
| N4 | **Next-hour vol: nonlinear models beat robust sigma x hour-of-week in both segments** (XGB QLIKE -0.040 / -0.065, MLP -0.036 / -0.065; linear on the same inputs -0.025 / -0.029); about half from extra inputs (recent \|r\| at 1-24 h 41%, daily HAR level 21%, hour-of-week 24% of attribution), half from nonlinearity; funding-hour and jump flags add nothing | B | H-NN8 | R-ML-7, R-ML-10 (forward H-NN10) |
| N5 | Wider networks add nothing for daily vol and their last layers are low-dimensional (effective dim 4.7 of 64); sparse autoencoders find no named concepts there - attribution (TreeSHAP groups, ablation) is the useful interpretability tool for these models, not dictionary learning | B (negative) | H-NN9, H-NN4 | R-ML-8, R-ML-9 |
| MD1 | (interim, 16 random days) Per-symbol microstructure traits (sign memory gamma, sign ACF1, size tail, arrival mixture) are NOT stable per symbol: ICC 0.23-0.31, day-to-day variation as large as between-symbol differences -> calibrate per day, not once per symbol | C (interim) | H-MD1 | - |
| MD2 | ETH and SOL are net volatility transmitters on 15 of 15 random days (BTC smaller than ETH) | C (interim, recording at the end) | H-MD5 | - |
| MD3 | Simulator v1 per-day calibration never reproduces >= 4/5 facts for majors (0/16 days): contradicts | B (negative) | H-MD6 | R-SIM-13 stays provisional |
| N6 | Next-minute volatility from 1 s data: a dilated CNN beats a linear model of 10/60/300 s activity means by only 0.8% QLIKE (wins 72% of holdout symbol-days, 51% for L2-book symbols); both beat trailing variance by ~10x that margin; the signal is the activity level over the last 1-5 minutes (log squared returns, orders, volume, range), not signed flow, imbalance, spread or depth | C | H-NN5 | R-ML-11, R-ML-12 |
| N7 | No next-minute direction from 1 s flow, linear or CNN (AUC 0.505-0.507) | B (negative) | H-NN6 | R-ML-13 |
| DS1 | A 36-cell discretisation (weekday group x own vol tercile x v0 trend) beats log-HAR for next-day vol by 6-7% in both segments but recovers only ~half of XGBoost's holdout gain (misses weekend depth); cells: Saturday + range/down trend ~-50%, Monday ~+25-45%, Tue-Fri + up-trend ~+20-28%; the weekday drives the cells, the v0 trend is second; no cell predicts direction | C (forward H-DS4) | H-DS1, H-DS2, H-DS3 | R-DS-1..6 (capped at C) |
| SP1 | The daily vol network is tiny underneath: 95% of weights can be pruned (100 of 2,192) with no loss in-sample (QLIKE ratio 0.986) and none detectable in the holdout (0.974, CI to 1.027); it degrades past ~97% | B / C | H-SP1 | R-SP-1 |
| SP2 | Minimal input set for next-day vol = ONE vol-level measure (robust sigma or 7d log RV, interchangeable) + the calendar (weekend / TradFi-weekend hours); funding, OI, jumps, range estimators, lags and products are pruned first; single vol columns are not individually stable (they substitute), the vol-level cluster is (forward H-SP4) | B | H-SP2 | R-SP-1, R-SP-2 |
| SP3 | Surviving neurons are mostly simple 2-input functions (71% R^2 > 0.8): the strongest are calm-weekend dampers (low 7d/30d vol x Fri-Sun lowers the forecast); others raise it for high vol x weekend, high vol x falling market, high vol x high trend; the circuit is redundant (removing any one unit changes QLIKE -5% to +6%, removing the top 3 x1.4) | C | H-SP3 | R-SP-3 |
| MN1 | FDR mining on the daily panel (495 effects, discovery before the cutoff, confirmation after) mostly re-finds known rules: 90 discoveries, 74 restate existing rules (53/74 replicate); only 2/16 novel ones replicate and both are marginal (forward H-MINE2a/b, not rules); return effects replicate at chance (1/11) - the rule set already captures the stable daily structure | B | H-MINE2 | - |
| N8 | Next-hour vol for TradFi perps: nonlinear models beat robust sigma x hour-of-week in all segments (holdout MLP -0.176 t -6.9, linear -0.131, XGB -0.102; MLP wins on 15/15 symbols); equities improve only with the MLP | C (TradFi mostly holdout) | H-NN11 | R-ML-20.. |
| N9 | Cross-asset inputs (BTC / ETH / QQQ recent activity) add nothing to next-hour vol forecasts beyond each asset's own inputs (XGB ~0, MLP worse); BTC carries about half of the cross-asset attribution for every class, but it is not a forecasting gain | B (null) | H-NN12 | R-ML-2x |
| RA1 | Rules audit 2026-11-12 (agent F): two conflicts resolved (R-VOL-9 narrowed to TradFi/oil; R-PX-1 defers to R-PX-2 until H-S3), R-SIM-13 retired, 8 A rules whose evidence flips across segments re-graded to B; the playbook (research/PLAYBOOK.md) is the compact per-context card the decision layer loads | - | audit | - |
| EX1 | Tight trailing exits (k = 1.5-3.5 one-bar sigma from the best close) do NOT fix the trend book's exit leakage: in-sample net P&L falls at every k (t about -2.2; Sharpe 0.86 -> -1.13..0.27); the trail just cuts exposure (0.86 -> 0.10-0.25) and gives up the big trend days (difference corr -0.99 with the book, skew -2); the share of the favourable excursion kept does not improve in-sample. Trend profits come from rare large moves - exits must leave room | B (negative) | H-PD6 | R-EXIT-1..3 |

## 10. Errata (2026-11-13, from the consistency check in research/reports/SCOREBOARD_2026-11-12.md)

Earlier lines are kept as written; where they conflict with later evidence, the later line wins:

| earlier | superseded by | current reading |
|---|---|---|
| P5 "Bybit TradFi perps freeze on weekends" | T1 (H-TP1) | they trade 24/7 at 0.1-0.15x weekday turnover; weekend direction is still uninformative (R-PX-9 corrected) |
| V10 "no blanket Monday multiplier" | N2, DS1 (H-NN1, H-DS1) | holds for TradFi equities; crypto next-day vol does carry a weekday factor (Mon ~1.3x, Sat ~0.5-0.6x) (R-VOL-9 narrowed) |
| M7 "per-symbol microstructure traits are persistent" (one day) | MD1 (H-MD1 interim, 16 days) | traits vary as much day to day as across symbols: calibrate per day |
| P3 "4h beats 1h for trend" | H-X2b (contradicts in both segments) | no reliable frequency edge; trend is a thin, segment-dependent bet |
| S3 "edge leaks through exits" (implied: fix the exits) | EX1 (H-PD6) | the leakage is the price of keeping the convex tail days; tight trailing exits make it worse |
| SP2 / SP3 graded on tests that contradicted their registered criteria | H-SP2, H-SP3 | descriptive findings only (grade C); the minimal-input reading rests on H-SP1 and TreeSHAP (H-NN3) |
| Section 5 'what to exploit first' (2026-11-07) | the research-only decision of 2026-11-12 | no live sleeves; the forward record runs through tools/forward_tests and the paper pipeline (docs/EXECUTION_PIPELINE.md) |
| priors | SCOREBOARD 2026-11-12 calibration | our priors ranked hypotheses well (AUC 0.75) but were ~2.3x too optimistic (mean prior 0.45 vs hit rate 0.19; direction/lore hypotheses ~0.07 vs priors ~0.38): shade new priors down accordingly |

## 11. Multi-day microstructure and mining, final (2026-11-13; 31 random days 2025-08 .. 2026-11, raw deleted after aggregation)

| # | belief | conf. | evidence | rule |
|---|---|---|---|---|
| MD4 | Micro traits (sign memory, size tail, arrival mixture) are day-specific, not symbol constants (ICC 0.23-0.34); activity scales with W but the exponent drifts by day; sign long memory on ~76% of symbol-days; the spread->vol gain for free-spread perps does not hold on new days (~40%); simulator v1 never reaches 4/5 facts for majors | B | H-MD1-4, H-MD6 | R-MD-1..3, R-SIM-13 retired |
| MD5 | ETH and SOL are net volatility transmitters on >= 80% of 31 days (criterion met; Holm-adjusted significance short) | C | H-MD5 | R-MD-4 |
| MN2 | FDR micro mining replicates on 15 unseen days: 454/550 (vol 90%, direction 28%); vol effects mostly restate known vol persistence and intraday seasonality; novel replicated effects are tiny: leader (BTC/ETH/SOL) extreme flow -> alts' next minute leans the same way (0.02-0.05 sd), TradFi own-flow continuation (partly bounce), extreme one-sided minutes are quieter | C | H-MINE1 | R-MINE-1..3 |
| GEN1 | A parametric, parameter-uncertain market generator (calibrated before the cutoff) covers the realised holdout Sharpe of 11/12 strategy families (but so does a fixed-parameter one: 132-day Sharpe noise is about +/-3, so coverage is easy and uninformative), reproduces 8/10 stylised facts (misses deep hourly tails and correlation drops), and does NOT rank strategies like the holdout (Spearman +0.24, CI incl. 0; the bootstrap ranks better at +0.64). Use generated paths for vol/jump stress tests, not for choosing strategies; funding/positioning cannot be simulated this way | B / C | H-GEN1-3 | R-GEN-1..5 |

## 12. BTC field program and side studies (2026-11-13 onward)

| # | belief | conf. | evidence | rule |
|---|---|---|---|---|
| VB1 | A volatility break of structure does NOT improve trend following: no detector family helps (Holm p 1), semivariance dominance is mostly a momentum proxy (matches the trailing return 69-80% of the time) and carries no direction, and the vol + structure filter loses to the plain ensemble (in-sample Sharpe 0.35 vs 0.44, DSR 0.22, PBO 0.84; holdout timing = random-shift median). The robust pattern is the opposite: trend earns LESS for 5-20 days after a vol range break (t -2.4 in-sample, -3.0 holdout) - forward test H-VBOS5 | B (negative) / C | H-VBOS1-4 | R-VBOS-1..4 |
| PN1 | Small networks (~500 params) beat log-HAR-t for BTC 1h return quantiles in both segments (pinball t -7.3 / -2.95), but KAN-style polynomial neurons add nothing over an equal-size MLP; at 1d nothing beats log-HAR-t. Adaptive conformal keeps 1h coverage within 0.2 pp but bands are 2-3% wider than log-HAR-t's (already near-calibrated). The latent state is highly predictable (R^2 gain 0.14 at 1h, 0.89 at 1d) yet feeding it back adds nothing. Extracted edge polynomials restate known structure: hour-of-week and last-hour |r| at 1h, a weekend clock at 1d | B (negative for PNN structure) | H-PNN1-3 | R-PNN-1..4 |
| FL1 | BTC probability field (Fokker-Planck with jumps, sparse polynomials, fit at the cutoff on 51k hours): variance = amplified hour-of-week profile x mean-reverting log-vol (half-life 107 h, symmetric in the sign of moves), fast-vol overshoot corrected, smaller intraday swing when long-run vol is high, more variance and jump risk when 24h momentum and trend agree (forward H-FLD4); jumps ~1.6%/h on a Student-t (df 4.3) core; drift = constant. It beats log-HAR-t + constant jumps (1h +0.033/+0.051 log score, 4h +0.027/+0.062) but only through hour-of-week; vs log-HAR-t x hour-of-week it ties at 1h and is worse at 4h/1d; online updating adds nothing (coefficients stable) | B | H-FLD1-3 | R-FLD-1.. |
| NW1 | Daily news attention (Wikipedia pageviews and edits, Hacker News counts, Fear and Greed) plus the event calendar and on-chain activity does NOT help next-day BTC vol forecasts: QLIKE worse by 3-4% vs log-HAR x weekday in both segments (t +1.5); no sub-block beats the baseline (HN alone t -0.8/-0.7). No direction skill either (Brier skill -0.018 both segments). GDELT (H-NEWS1/2) not run: rate-limited | B (negative) | H-NEWS3-4 | R-NEWS-5, R-NEWS-7 |
| OB1 | Generated BTC order books (queue-reactive generator calibrated on 3 ob200 days, executed by the Rust Bybit-semantics engine at-match) reproduce 6 of 8 held-out book and trade facts (7/8 in-sample, 6/8 holdout, 31 days); signed-flow ACF is too weak (0.003 vs 0.011) and median taker size too small. Generated twin days do not help a next-minute vol model (QLIKE ratio 1.003) and hurt when real data is cut to 25% (1.006, CI above 1). The engine replays a real ob200 day exactly | B | H-OBG1-2 | R-OBG-1, R-OBG-2 |
| MR1 | Market making replayed on real Bybit ob200 books (atx mm-replay, live quoting code, 80 symbol-days, BTC + TRX/ETC/XLM) reproduces our live fills roughly (6/6 P&L sign, fill-count error 0.25-0.75; thin data). Tight quoting within ~25 bp of mid loses: adverse selection within 1 s (-0.60 on BTC) exceeds the spread captured (+0.24) before fees. Wide quoting (40-50 bp) breaks even at best (BTC +0.039 USD/day CI [-0.11, +0.18]; illiquid +0.002). Micro-sigma widths lose to the best fixed width; the field's 1h band ties it | B (negative) | H-MMR0-2 | R-MMR-1..4 |
| OX1 | BTC options after a volatility compression (54 events, Deribit trades 2020-2026, 24 M trades): long ATM straddles lose less than on other days (delta-hedged -3.9% vs -6.4% of premium; realised/implied 0.85 vs 0.82) but still lose, so options are cheaper there, not mispriced. The user's call + short deep-ITM put structure is a ~1.1-delta long future (only 5.6% of the put premium is time value); it cuts vega by ~74% vs the call alone but loses vs a same-delta perp at events (-0.33% of spot per 14 d, crash losses). Tail moves show mild continuation at 3 days (q90-q99 +0.15 to +0.31 sigma, t ~2) and no reversal; scaling out after a q95 profit loses (trend book -0.004 R/trade, both segments) | B (negative) / C | H-OPX1-4 | R-OPX-1..5 |
| EX2 | Entering the paper sleeves' signals with a limit pegged to the touch beats a market order by +0.010 R per signal (CI +0.009/+0.012; holdout +0.014), all 275 signals filled within 37 s (no missed winners at paper sizes on BTC/ETH/SOL); the gain is the maker-taker fee gap (+0.015 R) minus a worse fill price (-0.005 R). Quoting only in quiet sideways phases (H-MMR3) removes most adverse selection but also almost all fills: still no edge after the 2 bp maker fee | B | H-EXE1, H-MMR3 | R-EXE-1, R-EXE-2, R-MMR-5, R-MMR-6 |
| RB1 | After a long-liquidation flush (48h return < 1y q05, OI -5% from its 72h high, cell R9; 649 events on 23 perps 2020-2026), buying the next 4h close loses (-0.19 R in-sample, -0.34 R holdout, worse than random R9 longs): price usually makes a further low first (median -8.7% below the event close, 70% go >= 5% lower, median 54 h later, bimodal), then rebounds about +15% off the low, with the 7-day return near zero. Shorting the first failed rebound is about breakeven; negative funding + rebuilding OI is not a squeeze signal. R9 hands over to R6 about 80% of the time (median 36 h more of R9 first) | B (negative) / C | H-RB1-3 | R-RB-1..5 |
| XS1 | A cross-sectional network over 23 perps (shared encoder, ~4k params, 18 purged quarterly refits) beats panel log-HAR-t x hour-of-week for 1h vol (pinball -1.4% both segments, every token; 4h gain vanishes in the holdout) and ranks relative returns with IC 0.05-0.08 beating momentum/reversal, but that IC is mostly a free low-vol ranking (holdout v3 minus low-vol +0.013 at 4h, 0 at 24h) and the top-minus-bottom quintile spread is negative after costs. Cross-token attention adds nothing out of sample (H-XS3 contradicts); BTC and TRX, not ETH/SOL, are the attention hubs. Polynomial (KAN) edges again add nothing over an MLP | B | H-XS1-3 | R-XS-1..4 |
| AN1 | Multi-scale shape + state analogs of BTC (2020-2026, 5 scales, declustered) do not forecast: their next-24h/72h move-size forecast is ~16% worse than log-HAR-t (QLIKE t +6.3 in-sample) and their sign calls are below the base rate. Descriptively, the 20 analogs of the 2026-11-15 weekend crash split into catalyst-resolved rebounds (SVB backstop, Fed cut signal, forced seller exhausted), relief rallies, and further legs down while forced selling continued (Jun 2022, Nov 2025, Feb 2026); the 7d low came a median 46 h later and 7d drawdown tails are deeper than normal | B (negative) | H-ANA1 | R-ANA-1..3 |
| MC1 | Yield-curve regimes (2y/10y 5-day changes, 2020-2026) don't separate BTC's forward returns: bear flattening -0.08 z at 5d (Holm p 0.63), bull steepening +0.15 z (p 0.34), wrong signs at 20d. Bull steepening with easing credit beats bull steepening with widening credit by +0.32 z at 20d (CI -0.12/+0.77, 2023-11+ only, HY data limit): the most promising macro lead. On 2026-11-12 the curve was in a young bear flattening (2y +28 bp in 5d after the hot CPI); IG/BBB spreads at p92 and widening; Sahm 0.00 | C | H-MAC1-2 | R-MAC-1..3 |
