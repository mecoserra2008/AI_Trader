# Knowledge book

_Built 2026-11-12 by `scripts/knowledge_book.py` from 236 registered hypotheses (412 result rows), 170 rules (hash 1bf8bb6c45), 12 forward tests and 240 diary entries. Do not edit by hand; rebuild. Machine-readable twin: `data/research/knowledge.json`. Per-hypothesis verdicts, prior calibration and the consistency check: [reports/SCOREBOARD_2026-11-12.md](reports/SCOREBOARD_2026-11-12.md). Grades follow [BELIEFS.md](BELIEFS.md): A = both segments and a large effect, B = small effect or confirmed once, C = plausible, being tested._

**Verdict codes** are in-sample / holdout[/pooled]: S supports, X contradicts, ? inconclusive, - not recorded. **Status**: supported (holds out of sample, no contradiction), refuted (contradicted), flip (one segment supports, the other contradicts; not supported under the protocol), open (results, nothing decisive), forward (collecting), untested.

Registry today: 30 supported, 87 refuted, 25 segment flips, 42 open, 12 collecting forward, 40 untested.

## The ten things to remember

1. Vol is forecastable; direction is not. Every working tool forecasts variance; every direction test is at coin-flip level (H-IC1, H-NN2, H-MINE2).
2. A daily vol forecast is the vol level plus the calendar: log-HAR level, then the forecast weekday (Saturday about 0.5-0.6x, Monday about 1.3x for crypto), then nothing else of note (H-NN3, H-SP1, H-DS1).
3. Per bar, use robust_bc; after a jump, excess variance halves in 2-3 h; after spikes, forecasts run hot (H-S4, H-V1b, H-RL2).
4. Expectations and risk differ: weekends and Saturdays are quiet in expectation but jump-prone, so stops use the unshrunk sigma (H-W1, H-V2).
5. Intraday vol has a large hour-of-week shape (peaks Sunday 22-02 UTC and at the US open); recent |returns| refine it (H-NS4, H-NN8).
6. No leverage effect in crypto; forecast from total RV, not the sign of the move (H-LEV2).
7. Levels, indicators, sessions and positioning lore do not predict direction; levels break slightly more often than they hold (H-VZ1, H-AU1, H-MC1, H-TF1).
8. The deterministic book is about one crypto-trend bet with a thin, currently negative edge; its P&L leaks through exits, not fees, and tight trails do not fix that (H-PD1, H-PD4, H-PD6, H-TR1).
9. Regimes say where strategies lose and how long a state lasts, not who wins next (H-RC1, H-RL3).
10. Microstructure and order flow are first measurements (C): bursty arrivals, long sign memory, activity scaling; traits vary by day; flow is an activity gauge (H-MS1, H-MS3, H-MD1, H-OF1).

## 1. Volatility I: forecasting models

**What we know**

- Volatility is what we forecast well, with simple models. Daily crypto variance: per-asset log-HAR (1d/7d/30d) beats level HAR by about 12% and robust_bc x 24 by about 18% QLIKE in-sample; multi-scale HAR beat the single-lambda EWMA by 0.08 QLIKE in both segments (H-VCL1, H-VM2, H-VM5; R-VOL-20).
- Per bar, use the jump-robust bias-corrected EWMA (robust_bc). Excess variance after a jump has a half-life of 2-3 h, so a plain EWMA over-forecasts post-jump vol by about 27%; robust_bc wins by 0.05 QLIKE in both segments, the uncorrected robust EWMA lost (H-S4, H-V1b, H-V1).
- After spikes forecasts run hot: in the v0 high-vol state next-week RV comes in 13-31% further below the EWMA forecast than in calm states (t -10.9 in-sample, -5.0 holdout, 9 assets) (H-RL2; R-VOL-8).
- Short-history and TradFi perps: pool across assets (panel log-HAR / HAR-Q; TradFi holdout QLIKE 0.70x robust_bc x 24 at 1d, 0.21x at 5d); for 2+ years of history the panel adds nothing (H-VM1; R-VOL-22). Multi-day: a direct 5d HAR, never sqrt(h) scaling of the 1h sigma (R-VOL-21).
- No help to HAR from: 1h range estimators (GK-HAR +5% / +16% QLIKE worse), BTC spillover, compression terms, vector candles, illiquidity spikes, or the common crypto vol factor (it explains ~70% of alt log-RV but not the alt's forecast error) (H-VM3, H-VCL3, H-VCL5, H-VC4, H-LC2, H-VM4). DVOL helps a little at 7d for BTC/ETH (-19% on 17 holdout origins; weak in-sample) (H-VCL4). Density: Student-t (df ~3.4) on HAR variance; a Kou jump mixture did not beat it (H-JD1).

**Confidence.** A for the core (both segments; 23 crypto perps over years plus ~132 holdout days). Refinements are B/C: log-HAR's holdout gain has t -0.42 and the panel HAR is inconclusive in the holdout. Registry: 4 supported, 7 refuted, 5 open, 1 untested.

**Rules** (\* = provisional): A: R-VOL-3, R-VOL-4, R-VOL-5, R-VOL-7, R-VOL-8, R-VOL-24; B: R-VOL-6, R-VOL-20, R-VOL-21, R-VOL-23, R-VOL-25, R-ML-1, R-ML-3, R-ML-10; C: R-VOL-22*, R-FLD-5*.

**Still open**

- Registered, no result: H-AN1 (titles in the scoreboard).

**Do not believe**

- *A jump-robust EWMA is better on its own*: lost until bias-corrected (H-V1 vs H-V1b).
- *Range estimators are more efficient, so they forecast better*: worse in both segments (H-VM3).
- *BTC vol spills into alts; compression predicts expansion; vector candles start a vol regime*: no gain in either segment (H-VCL3, H-VCL5, H-VC4).
- Our own priors >= 0.5 that failed: H-V1 (0.65, X/X), H-VCL3 (0.55, X/X), H-VC4 (0.50, X/X), H-VM3 (0.55, X/X).

## 2. Volatility II: calendar (weekends, Mondays, hour of week, expiries)

**What we know**

- Weekends are quiet: TradFi-weekend RV is a median 0.77x the Friday sigma (326 weekends; holdout 0.82x, 18 weekends); Saturday is the quietest forecast day (~0.5-0.6x Tue-Fri for crypto, 0.05-0.1x for TradFi perps) (H-W1; R-VOL-1).
- Weekends are also jump-prone: shrinking the risk sigma for them worsened QLIKE in both segments. Expectations use the weekend-scaled sigma; stops and risk use the unshrunk sigma_risk_next (H-V2; R-VOL-2).
- Hour of week is large and stable: robust sigma^2 x an expanding hour-of-week factor cuts next-hour QLIKE by 0.040 / 0.076 (crypto) and 0.29 / 0.38 (TradFi), in-sample / holdout. Peaks: Sunday 22-02 UTC and the US cash open (13-16 UTC); trough: Saturday 03-11 UTC (H-NS4; R-VOL-26, R-VOL-27).
- Mondays: in the crypto panel next-day variance is ~1.3x Tue-Fri once the vol level is known (cells +45% vs log-HAR in-sample, +25% holdout), but the per-asset test failed (BTC 1.33x in-sample, 0.82x holdout). SPX/NDX Mondays are not more volatile per trading day; WTI Mondays were 2.3x in the holdout (war weekends) (H-DOW2, H-DS1; R-DS-2, R-VOL-9).
- Implied vol overprices the weekend (BTC weekend RV/IV 0.35 vs 0.84 on weekdays, t -17; holdout 0.20, t -7.7), yet selling the Friday 08:00 straddle lost net in both segments: Friday options already price the weekend ~20% cheaper and the window holds the loud Friday US session and Sunday reopen (H-W7, H-OPT1-3; R-AVOID-2). Only a US-close entry remains, forward-only (H-OPT4). No weekly-expiry pin either (H-NS2; R-VOL-28).

**Confidence.** A for the weekend discount and hour of week (both segments, hundreds of weekends, two asset classes). The Monday factor is B/C: panel cells and ML attribution support it, the per-asset test failed, and its forward tests (H-DS4, H-VM6) are at 0-1 of 120 dates. Registry: 3 supported, 5 refuted, 2 flip, 1 open, 2 forward, 1 untested.

**Rules** (\* = provisional): A: R-VOL-1, R-VOL-2, R-VOL-26; B: R-AVOID-2, R-VOL-27, R-VOL-28, R-ML-1, R-ML-10, R-FLD-1, R-FLD-2; C: R-VOL-9*, R-ML-7*, R-SP-3*, R-ML-20*.

**Still open**

- Forward tests: H-VM6 3/120 forward forecast dates (min of L, T) (interim 0.0026); H-NN7 1/120 forward forecast dates.
- Registered, no result: H-OPT4 (forward-registered, but no runner collects them) (titles in the scoreboard).
- Segment flips to watch: H-OPT2 X/S, H-OPT3 X/S.

**Do not believe**

- *Selling weekend vol from Friday morning is free money*: the premium is real, the trade lost (H-W7 vs H-OPT1).
- *Shrink the risk sigma over weekends*: prior 0.75, refuted in both segments (H-V2).
- *One Monday multiplier for every asset*: TradFi Mondays 0.93-0.97x; BTC's own effect vanished in the holdout (H-DOW2).
- *The Monday NYSE open is an inflection point after eventful weekends*: H-W5 failed; H-W5b holds only in-sample.
- Our own priors >= 0.5 that failed: H-V2 (0.75, X/X), H-DOW2 (0.55, ?/X), H-W5 (0.55, X/?), H-OPT1 (0.60, X/X), H-OPT2 (0.60, X/S).

## 3. Volatility III: asymmetry and the leverage effect

**What we know**

- Crypto has no leverage effect: a down day raises next-day vol no more than an up day of the same RV (b(RS-) - b(RS+) = -0.03, t -0.28, 2,252 dates in-sample; -0.20, t -1.57 holdout; 0 of 23 GJR gammas significant). Forecast from total RV, not the sign of the move (H-LEV2, H-VCL2; R-LEV-1).
- Bybit equity/index perps show no textbook effect either in the 132-day holdout (sign flips with the specification; our prior was 0.85). Drawdowns create no asymmetry and SHAR does not beat plain (log) HAR for either class (H-LEV1, H-LEV3, H-LEV4; R-LEV-2, R-LEV-3, R-LEV-4).
- Crowded funding plus a down day does not amplify vol (H-LEV5, all segments); crowding alone tilted next-day vol up ~+0.07x mean RV per z (t 2.4 / 2.7, exploratory), now the forward test H-LEV6. A market-wide down-day effect is H-LEV7 (forward).

**Confidence.** Crypto A (both segments, 23 assets, years). TradFi C (holdout only, 132 days, 10 equity perps). Registry: 1 supported, 5 refuted, 2 forward.

**Rules** (\* = provisional): A: R-VOL-7, R-LEV-1; B: R-LEV-3, R-LEV-4, R-ML-3, R-FLD-2; C: R-LEV-2*, R-LEV-5*.

**Still open**

- Forward tests: H-LEV6 3/120 forward decision dates (interim -0.0804); H-LEV7 1/120 forward forecast dates.

**Do not believe**

- *Down moves raise crypto vol more than up moves*: refuted in both segments (H-VCL2, H-LEV2).
- *Crowded longs plus a red day means liquidation vol*: refuted in all segments (H-LEV5).
- Our own priors >= 0.5 that failed: H-VCL2 (0.50, X/X), H-LEV1 (0.85, ?/X/X).

## 4. Volatility IV: intraday (next hour, next minute, jumps)

**What we know**

- Next hour: the baseline robust_bc x hour of week beats bare robust sigma by 0.048 / 0.076 QLIKE. Nonlinear models beat the baseline in both segments for crypto (XGB -0.040 / -0.065, MLP -0.036 / -0.065; linear on the same inputs -0.025 / -0.029). Half the gain is inputs, half nonlinearity; the inputs that matter are recent |r| over 1-24 h (41% of attribution), the daily HAR level (21%) and hour of week (24%); funding-hour and jump flags add nothing (H-NN8; R-ML-10, R-ML-7).
- TradFi perps gain more (holdout MLP -0.176, t -6.9, 15 of 15 symbols; linear -0.131; XGB -0.102), mostly on holdout data; equities improve only with the MLP (H-NN11; R-ML-20).
- Cross-asset activity (BTC, ETH, QQQ) adds nothing at 1 h (XGB ~0, MLP worse), though BTC carries half the cross-asset attribution (H-NN12; R-ML-21, R-ML-22). Token-to-token vol transmission is a seconds-to-minutes effect (H-MS6), not an hourly one.
- Next minute (1 s data, 11 days, 347 symbol-days): the signal is the activity level of the last 1-5 minutes. A linear model of 10/60/300 s aggregates is ~10x better than trailing variance; a CNN adds only 0.8% QLIKE; signed flow, imbalance, spread and depth add nothing (H-NN5; R-ML-11, R-ML-12).
- Jumps cluster, but barely usefully (Hawkes vs constant rate: Brier skill +0.004 in-sample, +0.0002 holdout); for 2-sigma bar counts HAR + Student-t beats Hawkes (H-JD2, H-VB4; R-EXEC-3, R-SIG-4). The frozen hourly XGB (tools.vol_ml) is for expectations only until H-NN10 passes; never for stops.

**Confidence.** B for crypto hourly (both segments, 23 symbols, two model families agree); C for TradFi hourly (mostly holdout) and for next-minute (11 days). H-NN10 is at 0 of 60 forward days. Registry: 3 supported, 1 refuted, 2 open, 1 forward.

**Rules** (\* = provisional): B: R-EXEC-3, R-SIG-4, R-ML-10, R-PNN-1; C: R-ML-7*, R-ML-11*, R-ML-12*, R-ML-20*, R-ML-21*, R-ML-22*, R-ML-23*, R-ML-24*.

**Still open**

- Forward tests: H-NN10 2/60 forward days of logged hourly forecasts (interim -0.324).

**Do not believe**

- *Cross-asset leaders improve own-vol forecasts at 1 h*: no; the MLP gets worse (H-NN12).
- *A deep net on raw 1 s flow is needed for next-minute vol*: +0.8% over a linear activity model (H-NN5).

## 5. Microstructure: trade-level laws

**What we know**

- Trades arrive in bursts, never Poisson: a 3-component exponential mixture fits best (~30% bursts of ~5 ms, 40% gaps of 0.2-0.4 s, 30% gaps of 1-5 s); ACD(1,1) leaves most clustering in its residuals; a boosted AFT duration model beats ACD only for majors (H-MS1, H-MS9; R-MS-1, R-MS-2, R-MS-7).
- Order signs have long memory (lag-1 ACF 0.33-0.39, power-law decay gamma 0.45-0.75, 1 s signed-flow Hurst 0.53-0.54); on 16 random days 79% of symbol-days prefer a power-law ACF, just under the registered 80%. Merge same-ms same-side prints into one order (H-MS3, H-MD3 interim; R-MS-3, R-MS-4).
- Sizes: log-normal body with a heavy tail (index 1.4-1.8), not a pure power law. Volume per interval is log-normal, persistent and co-moves with |return| (Spearman 0.40-0.55) (H-MS2, H-MS4; R-MS-5, R-MS-6).
- Activity scales, character does not: order rate (~W^0.54), depth (~W^0.68) and impact (~W^-0.75, near Kyle-Obizhaeva) follow trading activity W, but per-symbol traits (sign gamma, ACF1, size tail, mixture weights) are not stable across 16 days (ICC 0.23-0.31): calibrate per day (H-MS7, H-MD1 and H-MD2 interim; R-SIM-10).
- 39 of 51 perps sit at one tick, so their spread is tick/price and says nothing about vol; for free-spread perps (CARV, KAS, NVDA) the gain is a coin flip so far (16 of 32 symbol-days). No sub-minute return lead from BTC/ETH/SOL (peaks within 20 ms), but ETH and SOL are net vol transmitters on 15 of 15 days (H-MS5, H-MD4, H-MS6, H-MD5; R-MS-11, R-MS-12, R-MS-13).
- Simulator v1 beats v0 in every class but never reproduces 4 of 5 facts for majors on any of 16 days, so R-SIM-13 is retired (H-MS8, H-MD6). Micro-state FDR mining froze 550 discoveries (nearly all activity effects); confirmation pending (H-MINE1).

**Confidence.** C throughout: one pilot day (the 2026-11-10 CPI sell-off) plus 16 of 30 planned random days; final H-MD verdicts after batches 4-6. Many pilot 'contradicts' are one failed clause of a composite criterion (e.g. H-MS1 on ACD residuals): read the numbers. Registry: 1 supported, 13 refuted, 2 open, 1 forward.

**Rules** (\* = provisional): B: R-ML-13, R-MD-1, R-MD-2; C: R-MS-1*, R-MS-2*, R-MS-3*, R-MS-4*, R-MS-5*, R-MS-6*, R-MS-7*, R-MS-11*, R-MS-12*, R-MS-13*, R-MS-14*, R-SIM-10*, R-SIM-11*, R-SIM-12*, R-SIM-14*, R-ML-12*, R-MINE-1*, R-MINE-2*, R-MINE-3*, R-MD-3*, R-MD-4*; retired: R-SIM-13.

**Still open**

- Forward tests: H-MS5b 0/10 free-spread symbol-days after 2026-11-10.

**Do not believe**

- *Trades are Poisson, or ACD removes the clustering*: both fail (H-MS1).
- *Each perp has fixed microstructure traits (one universal or per-symbol exponent)*: ICC 0.23-0.31 across days (H-MD1 interim, H-MS7).
- *The spread predicts vol*: on tick-bound perps the sign flipped out of sample (H-MS5).
- Our own priors >= 0.5 that failed: H-MS1 (0.80, -/X), H-MS2 (0.60, -/X), H-MS4 (0.75, -/X), H-MS5 (0.55, -/X), H-MS6 (0.50, -/X), H-MS7 (0.50, -/X), H-MS9 (0.50, -/X), H-MD1 (0.55, X/X/X), H-MD2 (0.55, X/X/X), H-MD3 (0.80, X/X/X), H-MD6 (0.50, X/X/X).

## 6. Direction: what fails, and why

**What we know**

- Next-day direction from standard features is a coin flip: Brier skill +0.0004 with one feature, -0.015 with 32 (34k symbol-days). ML has no direction skill, 1 s flow gives next-minute AUC 0.505-0.507, no discrete state predicts direction, and mined return effects replicate at chance (1 of 11) while mined vol effects replicate 54 of 79 (H-IC1, H-NN2, H-NN6, H-DS3, H-MINE2; R-BUDGET-1, R-ML-5, R-DS-6).
- Levels are not walls: vector-candle opens, swing pivots, CME gaps, swept zones, value-area/POC and floor pivots behave like random-walk levels, and price breaks through slightly more often than it holds (~0.52-0.55) (H-VZ1-4, H-VC2, H-VC3, H-GT3, H-W4, H-AU1-4, H-BO1; R-PX-4, R-PX-5, R-PX-10).
- Indicators and positioning lore have no net edge: Market Cipher B dots, squeeze releases, vol-surprise continuation, variance management, funding contrarian, OI divergence, stablecoin growth, VRP as direction (H-MC1, H-VB1-3, H-P1, H-P2, H-FU1, H-FU2; R-AVOID-1, R-AVOID-3).
- Calendar and sessions carry no direction: no Monday drift (9 assets), no session seasonality (F 1.86, p 0.13; ranking flips), no weekend-to-Monday information, and no TradFi session effect on Bybit perps (gap fill, night premium, ORB, intraday momentum, CME-reopen continuation); fading >1-sigma overnight moves at the open lost 67 bp net per trade in the holdout (H-DOW1, H-NS3, H-W2, H-W3, H-W6, H-TF1-5, H-TP1; R-PX-6, R-PX-9, R-AVOID-4, R-TF-1).
- Cross-section and relative value: sector momentum, within-sector outliers, residual momentum and low idio-vol fail in-sample (survivorship-biased universe); RV spreads revert (XAUT-XAU ~28% per 24 h) but their 9-14 bp sd is below the ~30 bp round-trip cost (H-SEC1-3, H-PAN1, H-PAN2, H-RV1-3).
- Why: our features describe the vol state and vol has no sign; most lore was learned on the period Claude has read about; the thin effects that exist are smaller than 7.5 bp per side. Holdout-only 'wins' (segment flips listed below) are watched, not traded.

**Confidence.** A/B for the negatives: tens of thousands of observations, both segments, Holm control in the newer blocks. The holdout is ~4 months, so small positive effects cannot be excluded; flips are the open risk. Registry: 3 supported, 28 refuted, 13 flip, 8 open, 1 forward, 11 untested.

**Rules** (\* = provisional): A: R-PX-10, R-AVOID-3, R-PD-4, R-AVOID-4; B: R-PX-4, R-PX-5, R-PX-6, R-PX-9, R-AVOID-1, R-SIG-1, R-SIG-2, R-SIG-3, R-MAP-5, R-MAP-6, R-TF-1, R-TF-3, R-TF-4, R-TF-7, R-CARRY-2, R-ML-5, R-ML-13, R-SP-4; C: R-PX-7*, R-TF-2*, R-TF-5*, R-TF-6*, R-PX-12*, R-DS-6*.

**Still open**

- Forward tests: H-TP1b 4/150 forward equity symbol-sessions with |on_z| > 1 (interim -336 net bp per trade).
- Registered, no result: H-SEC4, H-PAN3, H-PAN4, H-PAN6, H-RV4, H-AN2, H-PM1, H-PM2, H-MINE2b; H-VC6, H-VC7 (forward-registered, but no runner collects them) (titles in the scoreboard).
- Segment flips to watch: H-W2 X/S, H-W4 X/S, H-P1 X/S, H-VC1 X/S, H-VC2 X/S, H-VZ4 X/S, H-AU1 X/S, H-AU3 X/S, H-FU2 X/S, H-RL1 X/S, H-SEC2 X/S, H-VB2 X/S/X, H-VB3 X/S/X.

**Do not believe**

- *CME gaps always fill*: random-walk rate in-sample; the holdout 'support' is a flip (H-W4).
- *Levels hold, taken levels switch off, untaken opens are magnets; the 80% rule; the POC magnet*: all at random-walk rates (H-VZ1-4, H-VC3, H-AU1-4).
- *Mondays drift, sessions have a direction, BTC bounces after a down CPI day*: no (H-DOW1, H-NS3, H-EV2).
- *Fade crowded funding; rising OI is bullish*: no; OI divergence is market beta (H-P1, H-P2; R-PD-4).
- *The regime cell tells next week's direction*: up minus down 0.03 sigma, t 1.0 (H-RL1).
- Our own priors >= 0.5 that failed: H-RV3 (0.50, X/X).

## 7. Strategies and regimes

**What we know**

- Trend is the only family with an edge, and it is thin: in the 21-variant zoo the best variant had Sharpe +0.50 and the ensemble +0.33 before the cutoff (deflated Sharpe of the winner 0.72 < 0.95); every variant is negative Jul-Nov 2026 (H-TR1, H-X2b; R-PX-3).
- Selecting variants does not help: walk-forward selection loses to the ensemble; per-symbol parameter choice is a coin flip (PBO 0.44 zoo, 0.65 breakouts; holdout breakout champions did worse than a random pick); generated paths do not fix it (H-TR2, H-SYN1, H-SYN2; R-SYN-1, R-SYN-2).
- 'Is it trending?' is not reliably forecastable: classifiers, directional efficiency, variance ratio, auction balance and Kalman SNR fail in-sample; the 'trend moment' probit/XGB show holdout skill only (Brier skill +0.10 / +0.16, CI includes 0) (H-TR3-6, H-RG1, H-KF2; R-PX-11).
- The regime map tells where strategies lose, not who wins: per-cell rankings do not persist (Spearman 0.085, CI includes 0); robust losers are 1h mean reversion in trend/compression/grind-down cells and 1h trend in range/whipsaw cells; dropping in-sample-negative cells beat equal weight in the holdout but stayed negative (Sharpe -0.77 vs -2.48) (H-RC1, H-RC2, H-R1, H-R2; R-MAP-1-7).
- Hourly deviations from trend continue; fading 2-sigma hourly moves loses even before costs (H-T1; R-PX-2). 'Calm shocks continue, volatile shocks reverse' is data-generated and weakly tested (holdout t 1.6 on 152 R4 events, n 13 in R9); the registered H-S1 failed and the jump version H-JD3 flipped. shock_regime is a forward test (H-S3), not knowledge (H-S2; R-PX-1).
- Regimes describe the vol path: median episode 5-7 days, daily stay 0.84-0.89, older episodes end sooner; after a liquidation regime expect high-vol whipsaw (62%) (H-RL3; R-PX-8). Funding carry was real (net Sharpe 2.8 in-sample) but decayed (57%/yr 2021, 0.5% 2025, -4.8% 2026 H1; holdout net t 1.3): a leverage-mania premium (H-C1; R-CARRY-1; forward H-C1b).

**Confidence.** B: trend facts in both segments but a 132-day holdout; the regime-map grades rest on a context table whose rank test (H-RC1) is inconclusive; shock and carry claims are C until their forward tests report. Registry: 4 supported, 11 refuted, 9 flip, 5 open, 4 untested.

**Rules** (\* = provisional): B: R-PX-1, R-PX-2, R-PX-3, R-PX-5, R-PX-8, R-CARRY-1, R-AVOID-1, R-PX-11, R-MAP-1, R-MAP-3, R-MAP-4, R-MAP-5, R-MAP-6, R-PD-1, R-PD-2, R-SYN-2, R-SYN-3, R-VBOS-2, R-VBOS-3; C: R-SIG-5*, R-MAP-2*, R-MAP-7*, R-SYN-1*, R-VBOS-1*, R-VBOS-4*.

**Still open**

- Registered, no result: H-VBOS5; H-S3, H-X2c, H-C1b (forward-registered, but no runner collects them) (titles in the scoreboard).
- Segment flips to watch: H-VBOS2 X/S/X, H-X2 X/S/?, H-TR1 S/X, H-TR3 X/S, H-TR4 X/S, H-TR5 X/S, H-TR6 X/S, H-KF1 X/S, H-JD3 S/X.

**Do not believe**

- *Trend earns only in trend cells; mean reversion earns in ranges*: both refuted (H-R1, H-R2).
- *4h/1d is the trend sweet spot*: the scale-free re-test failed in both segments (H-X2b); H-X2c was never evaluated.
- *High trailing vol kills trend; pivot geometry adds over slow trend*: refuted (H-X3, H-GT1, H-GT2).
- *Funding carry is a standing premium*: it decayed to about zero (H-C1).
- Our own priors >= 0.5 that failed: H-R1 (0.60, X/?), H-R2 (0.50, X/?), H-X2 (0.60, X/S/?), H-X2b (0.60, X/X).

## 8. PnL structure of the deterministic book

**What we know**

- The book is roughly one bet: ~8 eigen-bets in the 16-sleeve correlation matrix, but 1.15 effective (Meucci) bets in the equal-weight book in-sample and 2.5 in the holdout; crypto trend dominates (H-PD1; R-PD-1).
- Trend = crypto TSMOM factor (loading 0.36-0.79, t 7-20) + long convexity; 1h mean reversion = short convexity (loading -0.51, t -12.8), so it does not hedge trend in stress; OI divergence = market beta. No sleeve has alpha in both segments; factors explain ~25% of daily variance (H-PD2; R-PD-2, R-PD-3, R-PD-4, R-PD-5).
- Edge leaks through exits, not fees: ~70-75% of the favourable excursion is lost, costs are 5-9% of it (H-PD4; R-PD-6). But that leakage is the price of the convex tail days: k-sigma close trails (k 1.5-3.5) cut in-sample P&L at every k (t -2.1 to -2.3; Sharpe 0.86 -> -1.13 to 0.27) by cutting exposure from 0.86 to 0.10-0.25; their holdout 'gain' is an exposure cut on a losing book (H-PD6; R-EXIT-1, R-EXIT-2, R-EXIT-3).
- Strategy returns reverse week to week, so trailing-Sharpe ranks carry negative information and a context IPCA ranking beats them (IC 0.10, t 3.2 / 0.13, t 1.2) (H-PD3; R-PD-7; forward H-PD5).

**Confidence.** B: H-PD1-4 decided in both segments; MFE uses bar highs/lows, so 'exit leakage' is an upper bound; H-PD5's first forward week pair is 2026-11-30. Registry: 2 supported, 2 refuted, 1 flip, 1 forward.

**Rules** (\* = provisional): A: R-PD-4; B: R-PD-1, R-PD-2, R-PD-3, R-PD-5, R-PD-6, R-EXIT-1, R-EXIT-3; C: R-PD-7*, R-EXIT-2*.

**Still open**

- Forward tests: H-PD5 0/26 forward week pairs (trailing, next).
- Segment flips to watch: H-PD6 X/S/X.

**Do not believe**

- *A 16-strategy book is diversified; a positive raw Sharpe is alpha*: 1-2.5 bets; factor exposure plus noise (H-PD1, H-PD2).
- *Fees are the main drag; back last week's winner*: exits are; trailing ranks reverse (H-PD4, H-PD3).
- *Tighter trailing stops recover the exit leakage*: they cut the trend tails and lose in-sample (H-PD6; R-EXIT-1).
- Our own priors >= 0.5 that failed: H-PD1 (0.70, X/X), H-PD2 (0.55, X/X).

## 9. Execution, risk mechanics and market making

**What we know**

- Keep resting exchange stops (MarkPrice trigger): close-confirmed stops had a worst trade of -1.9R vs -1.0R (H-X1; R-EXEC-1). Jump-aware first-passage odds are better calibrated for brackets than a Gaussian (Brier -0.0022 / -0.0014, CIs below 0) (H-JD4; R-EXEC-2).
- Live MM (H-MM2, 2 of 10 run-days): passive quote-and-clear made money (passive exits 84-85% winners, +0.55 USD over 43 trips); market-order exits gave it back (-0.54 USD over 42 trips). ~50 crypto contracts are one correlated bet (5m correlations 0.85-0.96) (R-MM-1, R-MM-2).
- NVDA micro MM (0% maker): quoting captured ~+9 cents per NVDA traded, but a bullish inventory on a falling NVDA lost more (total ~-0.16 USD): inventory direction, not spread capture, decided P&L. During a machine sleep, resting quotes stayed live and unmanaged (no disconnect-cancel-all) (diary 2026-11-12).
- Simulator v0: fills consistent with 23 live fills (inconclusive); generated flow fails stylised facts; reaction functions do not transfer across markets; generated paths rank MM parameters by activity, not P&L (Spearman -0.09). Rank on real 3 s replays (one sell-off day: ask 0.2 / bid 0.9 sigma earned +14.2 bp of turnover vs +7.5 bp live) (H-SIM1-4; R-SIM-1-7; paired live check H-MM3 blocked).
- Fees: crypto 2 bp maker / 5.5 bp taker; TradFi perps 0% maker / 2.75 bp taker; research assumes 7.5 bp per side. Open design conflict for the user: breakout structural stops sit 6-8 sigma from entry vs a 4-sigma cap (BELIEFS F2, section 4).

**Confidence.** C for market making (2 live run-days, 1 replay day, 23 test fills); B for stops (H-X1) and bracket odds (H-JD4). Registry: 1 supported, 4 refuted, 2 open, 6 untested.

**Rules** (\* = provisional): B: R-EXEC-1, R-EXEC-2; C: R-MM-1*, R-MM-2*, R-SIM-1*, R-SIM-2*, R-SIM-3*, R-SIM-4*, R-SIM-5*, R-SIM-6*, R-SIM-7*; retired: R-SIM-13.

**Still open**

- Registered, no result: H-F5, H-MM1, H-AS1, H-AS2; H-MM3 (needs paired live market-making runs (exchange records)); H-NS7 (needs paired live runs or post-registration 3 s replays of the live quoter) (titles in the scoreboard).

**Do not believe**

- *Market-order exits are fine for MM inventory; many symbols diversify*: they gave back the edge; one correlated bet (R-MM-1, R-MM-2).
- *A flow generator can rank MM parameters*: it ranks activity, not P&L (H-SIM4).
- Our own priors >= 0.5 that failed: H-X1 (0.50, X/?).

## 10. Order flow

**What we know**

- Flow data exists only after the cutoff (flow_3s from 2026-11-07, liquidations from 2026-11-08): all clean holdout, about 3-5 usable days, one market state.
- The registered |z| > 4 imbalance detector fires on 22% of minutes (signed flow is far heavier-tailed than its MAD scale; a symbol's top 1% starts near |z| 56). Afterwards |return| is only 1.14x at 5 m, 1.05x at 15 m, sign rate 0.513 (H-OF1, H-OF2; R-FLOW-1). A selective top-1% detector shows 1.48x (CI 0.93-1.63, 309 events): forward H-OF6.
- Liquidation clusters continued, not reverted (-1.85 sigma at 60 m in the liquidation direction, t -1.9, n 142): forward H-OF7 (interim +1.05 on 2 of 40 cluster-hours) (H-OF3; R-FLOW-2). BTC flow does not lead alts; a multivariate score is no better than the imbalance z (H-OF5, H-OF4; R-FLOW-3, R-FLOW-4).
- At 1 s, signed flow and book imbalance add nothing for next-minute vol or direction (H-NN5, H-NN6): read flow as an activity gauge. L1-OFI, depth, spot-perp and weekend-flow tests (H-F1-5) wait for 30+ days of flow_3s.

**Confidence.** C: about 3-5 days, one market state (includes the CPI sell-off), collector gaps. Registry: 3 refuted, 2 open, 2 forward, 5 untested.

**Rules** (\* = provisional): B: R-ML-13; C: R-FLOW-1*, R-FLOW-2*, R-FLOW-3*, R-FLOW-4*, R-MS-12*, R-MINE-3*.

**Still open**

- Forward tests: H-OF6 1/14 forward UTC days of flow (interim 1.25); H-OF7 23/40 independent cluster-hours (interim 0.386).
- Registered, no result: H-F1, H-F2, H-F3, H-F4, H-PAN5 (titles in the scoreboard).

**Do not believe**

- *A big taker imbalance tells direction*: sign rate 0.51 (H-OF2).
- *Fade liquidation cascades; BTC flow leads alts by minutes*: they continued; no lead (H-OF3, H-OF5).
- Our own priors >= 0.5 that failed: H-OF1 (0.75, -/X), H-OF5 (0.50, -/X).

## 11. ML, discrete states and interpretability

**What we know**

- Daily vol: MLP and XGB beat log-HAR in both segments (MLP -8.0% in-sample, t -2.9; XGB -15.7% holdout, t -2.4, 22 of 23 symbols), but the winning family flips between segments and most of the gain is calendar: log-HAR + weekday dummies gets -5.5% alone (H-NN1; R-ML-1, R-ML-4).
- What models use: the vol level (7d log RV, robust sigma; ~40% of attribution; +75% QLIKE when ablated) and the forecast weekday; trend, funding, market 1-3% each; jumps, semivariances, ranges, raw returns nothing. Interactions are ~1/3 of attribution with the same top-5 pairs in both segments; trees and nets agree on vol drivers (rank corr 0.75), not on direction (0.22) (H-NN3; R-ML-3, R-ML-6).
- The vol net is tiny underneath: 95% of weights prune away (100 of 2,192) with no loss (QLIKE ratio 0.986 in-sample, 0.974 holdout, CI to 1.027). Minimal inputs: one vol-level measure + the calendar; single vol columns substitute for each other, so column-level stability selection found nothing (H-SP2 failed). Survivors are mostly simple 2-input units (17 of 24), the strongest being calm-weekend dampers; the circuit is redundant (H-SP1, H-SP2, H-SP3; R-SP-1, R-SP-2, R-SP-3).
- A 36-cell table (weekday group x vol tercile x v0 trend) beats log-HAR by 6-7% in both segments but recovers only ~half of XGB's holdout gain (it misses weekend depth); cell ordering Spearman 0.76 (H-DS1, H-DS2; R-DS-1-5; forward H-DS4).
- Sparse autoencoders (16 and 64 units) find no named concepts (1 and 0 of 10); explain forecasts with grouped TreeSHAP and ablation; wider nets add nothing (H-NN4, H-NN9; R-ML-6, R-ML-8, R-ML-9). Daily FDR mining (495 effects) re-found known rules: 74 of 90 discoveries were restatements; 2 of 16 novel ones replicate, both marginal (H-MINE2; forward H-MINE2a, H-MINE2b).

**Confidence.** B for 'vol level + calendar is the signal' (both segments; two model families and three methods agree). Specific ML tools are C until forward tests pass (H-NN10 0/60, H-DS4 0/120, H-SP4 0/3). Registry: 3 supported, 4 refuted, 3 open, 2 forward, 1 untested.

**Rules** (\* = provisional): B: R-ML-1, R-ML-3, R-ML-6, R-ML-8, R-ML-9, R-SP-1, R-SP-2, R-SP-4; C: R-ML-2*, R-ML-4*, R-DS-1*, R-DS-2*, R-DS-3*, R-DS-4*, R-DS-5*, R-SP-3*.

**Still open**

- Forward tests: H-DS4 1/120 forward forecast dates; H-SP4 0/3 monthly refits with training end after 2026-11-12.
- Registered, no result: H-MINE2a (titles in the scoreboard).

**Do not believe**

- *Bigger networks or more features help; ML finds direction*: no (H-NN9, H-IC1, H-NN2).
- *SAE features are market concepts; an FDR-controlled novel discovery is probably real*: 1/10 and 0/10; 2 of 16 replicated (H-NN4, H-NN9, H-MINE2).
- Our own priors >= 0.5 that failed: H-SP2 (0.50, X/X/X).

## 12. BTC field program: probability field, polynomial networks, news, generated markets and books, live interpretation

**What we know**

- The BTC probability field (Fokker-Planck with jumps, sparse polynomial drift / diffusion / jump intensity) beats log-HAR-t with constant jumps only through hour-of-week; against log-HAR-t x hour-of-week it ties at 1h and loses at 4h and 1d; online coefficient updating adds nothing; drift is a constant. Momentum x trend agreement raising variance is a forward test (H-FLD1, H-FLD2, H-FLD3, H-FLD4).
- Small networks (~500 parameters) beat log-HAR-t for 1h BTC return quantiles in both segments, but polynomial (KAN-style) neurons add nothing over an equal-size MLP; adaptive conformal bands keep coverage within 0.2 pp; the latent state is predictable yet feeding it back adds nothing (H-PNN1, H-PNN2, H-PNN3).
- News attention (Wikipedia pageviews and edits, Hacker News counts, Fear and Greed), the event calendar and on-chain activity make next-day BTC vol forecasts 3-4% worse and carry no direction skill; GDELT was not testable (H-NEWS1, H-NEWS2, H-NEWS3, H-NEWS4).
- Generated BTC order books run through the Bybit-semantics matching engine reproduce 6 of 8 held-out book facts (signed-flow persistence and taker size miss), but generated twin days do not improve a next-minute vol model and hurt when real data is scarce (H-OBG1, H-OBG2). The daily market generator reproduces stylised facts yet cannot rank strategies (H-GEN2, H-GEN3).
- Live: the frozen network runs hourly and Claude logs its own interpreted forecasts before each window opens; whether interpretation beats the network is forward-only (H-LIVE1, H-LIVE2).

**Confidence.** B for the negatives (both segments agree); C for everything forward (H-FLD4, H-LIVE1, H-LIVE2). Registry: 4 supported, 4 refuted, 7 open, 3 untested.

**Rules** (\* = provisional): B: R-PNN-1, R-FLD-1, R-FLD-2, R-FLD-3, R-NEWS-5, R-NEWS-7, R-OBG-1, R-OBG-2; C: R-GEN-1*, R-GEN-2*, R-GEN-3*, R-GEN-4*, R-GEN-5*, R-PNN-2*, R-PNN-3*, R-PNN-4*, R-FLD-4*, R-FLD-5*.

**Still open**

- Registered, no result: H-FLD4, H-LIVE1, H-LIVE2 (titles in the scoreboard).

**Do not believe**

- *Richer structure (fields, polynomial neurons, latent feedback, news) beats a calendar-aware vol baseline*: not out of sample (H-FLD1, H-PNN1, H-PNN3, H-NEWS3).
- *Generated data can stand in for real data*: not for training or strategy ranking (H-OBG2, H-GEN3).

## 13. Decision budget: what to look at, how much history, how fast

**What we know**

- Direction skill peaks at ~1 feature and falls as features are added (Brier skill +0.0004 at k = 1, -0.015 at k = 32, 1,703 dates); the holdout peaks at k = 8 with every CI including 0 (H-IC1; R-BUDGET-1).
- Fit on 1-2 years or expanding history; 90-day windows are worse (Brier +0.0012 / +0.0026, t 1.6 / 1.9) and no shorter window wins anywhere (H-IC2; R-BUDGET-2).
- Latency: the 4h trend keeps 97% of its edge after 4 h of delay (88% after 8 h); shock edges keep 61% after 1 h, 10% after 2 h. Shock signals execute at the bar close with an LLM veto only (H-IC3; R-BUDGET-3, R-BUDGET-4).
- So: short cards (8 or fewer tested lines), no extra inputs for directional calls, deliberate on slow signals, never delay fast ones. Whether the compact card beats a full tool dump for the LLM is forward-only (H-IC4).

**Confidence.** B/C: H-IC1-3 are formally inconclusive (the holdout lacks power) although the in-sample samples are large (34k symbol-days). Registry: 3 open, 1 untested.

**Rules** (\* = provisional): B: R-BUDGET-1, R-ML-5; C: R-BUDGET-2*, R-BUDGET-3*, R-BUDGET-4*.

**Still open**

- Registered, no result: H-IC4 (titles in the scoreboard).

**Do not believe**

- *More context makes better calls; recent data is more relevant*: skill falls with features; short windows lose (H-IC1, H-IC2).

## 14. Data and infrastructure lessons

**What we know**

- Point-in-time must be enforced in code: inputs need event_ts < T, funding and OI count only if known_ts <= T, and known_ts was clamped because the exchange clock ran ahead of local time (diary 2026-11-07).
- Most errors were test design: wrong nulls (flat 0.5 instead of the random-walk race, H-GT3), unreachable criteria (H-VCL5), standardisation that absorbed the effect (H-W5), metrics not scale-free across frequencies (H-X2), an extra verdict conjunct (H-LEV2). Corrections are appended, never overwritten.
- Silent failures recur: Int8 overflow, numpy NaN passing a polars is_not_null filter (use is_finite), optimisers that overflow or collapse, universes that drop symbols (10 of 16 TradFi perps lacked native 4h history).
- Storage and ops: only 3 s flow candles are stored (~1.7 MB per stream-day, ~200 days in 30 GB); raw micro batches are deleted after verified aggregation (lake + research 3.07 GB after batch 3). Loops die with their owning session (updater and collector were down from 2026-11-11 02:46), a machine sleep froze agents for 3 h, and long pipelines need resume and atomic downloads.
- Samples are small where it matters: survivorship-biased universe, TradFi perps listed Mar-Jun 2026 (2-3 months in-sample), a ~132-day daily holdout. Verdict labels compress: 'contradicts' can be one failed clause (H-MS1, H-SP2), 'SUPPORTED' can hide decay (H-C1).

**Diary mistakes by class** (40): test design / definitions 14, pipeline / operations 12, numerics / silent NaN 8, point-in-time / look-ahead 6.

**Do not believe**

- *An in-sample 'supports' is new evidence*: that period is what Claude has read about; only the clean holdout is unmemorised (CLAUDE.md principle 6).

## 15. How the decision layer should use this

- **Load the card, not the book.** At decision time call `ai_trader.research.playbook.card_for(context, state)` ([PLAYBOOK.md](PLAYBOOK.md)): at most 8 tested rules for the context, already filtered by the market state (R-BUDGET-1). This book explains why the cards say what they say; it is not a list of extra inputs.
- **Order of questions.** Vol first (level, then calendar, then intraday shape), risk second (sigma_risk_next, resting stops), direction last, and only through the registered deterministic candidates. Never add inputs to make a directional call.
- **Expectations vs risk.** Weekend, Saturday, cell and ML adjustments change expectations only; stops and risk always use the unshrunk sigma_risk_next (R-VOL-2). Position size comes from the risk engine, never from a rule.
- **Grades.** Apply A and B rules as stated, treat C rules as context, and never trade a segment flip or an interim forward estimate.
- **Shocks.** shock_regime executes at the bar close without waiting for the LLM, which may only veto (R-BUDGET-4); never fade a 2-sigma hourly move on discretion (R-PX-2).

Call order per context (from `data/research/playbook.json`, rules hash 1bf8bb6c45):

| context | call, in order | skip |
|---|---|---|
| `next_day_vol` | vol.har -> vol.robust_bc -> vol level: robust sigma or 7d mean log RV -> calendar: forecast weekday / weekend hours -> LAT1 cell adjustment -> HAR variance through a Student-t -> sigma_risk_next | BTC spillover / compression terms; jump variation as a vol input; plain / single-lambda EWMA; range estimators; raw-return inputs; semivariance / SHAR terms |
| `next_day_vol_tradfi` | vol.robust_bc -> vol.har -> panel log-HAR -> calendar: forecast weekday / weekend hours -> HAR variance through a Student-t -> HAR-Q -> sigma_risk_next | range estimators; semivariance / SHAR terms |
| `next_hour_vol` | vol.robust_bc -> vol.har -> seasonality.seasonal_factor -> recent \|r\| (1-6h) + daily HAR level adjustment -> tools.vol_ml hourly XGB -> sigma_risk_next | - |
| `next_day_direction` | (no tool; rules only) | cell tables for direction |
| `weekend` | vol level: robust sigma or 7d mean log RV -> vol.robust_bc -> vol.har -> calendar: forecast weekday / weekend hours -> sigma_risk_next | range estimators; semivariance / SHAR terms |
| `monday` | vol.har -> calendar: forecast weekday / weekend hours -> sigma_risk_next | - |
| `tradfi_open` | vol.robust_bc -> seasonality.seasonal_factor -> calendar: forecast weekday / weekend hours -> tools.vol_ml hourly XGB -> sigma_risk_next | - |
| `high_vol` | vol.robust_bc -> vol.har -> calendar: forecast weekday / weekend hours -> sigma_risk_next | - |
| `liquidation` | (no tool; rules only) | Mahalanobis flow score; flow imbalance \|z\| > 4 |
| `quoting` | vol.robust_bc -> seasonality.seasonal_factor -> linear 10/60/300 s activity model -> sigma_risk_next | ML direction models |
| `stops_and_risk` | vol.robust_bc -> vol.har -> calendar: forecast weekday / weekend hours -> seasonality.seasonal_factor -> XGB daily vol tool -> sigma_risk_next -> jump-aware first-passage p_tp | flat sqrt(h) scaling of the 1h sigma; jump variation as a vol input; plain / single-lambda EWMA; range estimators; raw-return inputs; semivariance / SHAR terms |
| `strategy_choice` | (no tool; rules only) | semivariance / SHAR terms |

Audit note: for `next_day_vol` the intended pipeline is log-HAR level (R-VOL-20) -> forecast-weekday factor (R-ML-1) -> (C, only once validated forward) LAT1 cells (R-DS-5) or the nonlinear tool (R-ML-4); R-VOL-4 is superseded by R-VOL-20.
