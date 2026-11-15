# Less obvious strategies: funding settlement, options expiry, sessions, hour-of-week volatility, QQQ open (H-NS1..H-NS5)

_Program 2 (research/plans/2026-11-11_strategy_program.md), agent B3. Code: `src/ai_trader/features/seasonality.py`,
`src/ai_trader/research/new_strategies_study.py` (the docstring fixes every control and verdict rule; they were written
before any result was seen and logged in the diary at the start). Numbers: `new_strategies_2026-11-11.json`. as_of
2026-11-10 00:00 UTC; split at the knowledge cutoff 2026-07-01. Tests: `tests/test_seasonality.py`._

## Summary

| id | test | llm_in_sample | clean_holdout | Holm p (primary) | reading |
|---|---|---|---|---|---|
| H-NS1 funding-settlement pressure | pre b < 0 and post b > 0 | **inconclusive** (b_pre −0.010, t −1.4; b_post +0.006, t 0.7) | supports (signs only: −0.010, t −0.5; +0.048, t 2.3) | 0.70 | no settlement-specific flow: the same pre-hour lean appears at mid-interval hours (placebo b −0.020, t −2.4) |
| H-NS2 options expiry pin and release | d_pre < 0 and d_post > 0 | **contradicts** (d_pre −0.007, t −0.2; d_post −0.20, t −4.2) | contradicts (+0.11, t 0.7; −0.31, t −1.2) | 1.00 | no weekly pin; the post window is quieter, not louder (partly the Saturday-morning bias). Lead: monthly pre-window −0.23 log (t −2.4) |
| H-NS3 session seasonality | F p < 0.01; holdout rank ρ > 0.5 and tilt > 0 | **contradicts** (F 1.86, p 0.13) | contradicts (ρ −0.40; tilt −0.89 R/day, t −3.1) | 0.53 | sessions don't differ; the tilt loses even before costs in the holdout |
| H-NS4 hour-of-week vol profile | QLIKE diff < 0, t < −2, L and T | **supports** (L −0.040, t −8.1; T −0.29, t −9.5) | **supports** (L −0.076, t −4.5; T −0.38, t −15.8) | 1e-15 | large, robust gain: the one result to deploy |
| H-NS5 QQQ first NYSE hour leads crypto | slope > 0, t > 2 pooled | contradicts (16 days) | contradicts (92 days) | 1.00 | pooled slope −0.05 (t −0.6, 108 days); QQQ and crypto co-move in the same hour (β 0.29, t 3.2), no lead after it |

Holm (5 primary p-values) changed no verdict. Only H-NS4 survives. "R" in tables means risk units (P&L per unit of
σ̂ at entry, as in `evaluate.strategy_pnl`).

## Common definitions

- 1h bars from `research.frame.market_frame` (price sanity, listing burn-in). Vol-scaled return z_t = r_t / s_t, with
  s_t = `sigma_risk_next` of bar t−1h (robust, bias-corrected, no weekend shrink; known at the open of bar t).
- Universe L = `trend_zoo_study.UNIVERSE` (23 crypto perps). Universe T = the 15 TradFi perps of the vol tournament with
  ≥ 120 complete UTC days (XAU, XAG, CL, BZ, TSLA, INTC, GOOGL, AAPL, MSTR, NVDA, MU, QQQ, SOXL, PAXG, XAUT; SPCX
  excluded at 116 days).
- Date-clustered SE: OLS with SEs clustered by UTC date. NW t: `evaluate.newey_west_t` on a date (or week) series.
- Verdicts: in-sample rows need the predicted sign with |t| > 2 on every part; holdout rows where the criterion asks
  only for "same signs" need the signs only; any wrong sign with |t| > 2, or every part wrong, is a contradiction.

## H-NS1 Funding-settlement pressure (universe L)

**Settlement mapping.** I read the lake's `funding` table (point-in-time) and mapped each symbol's actual settlement
times. Every L symbol settles every 8h at 00/08/16 UTC, with two exceptions: ENA has been on 4h since 2025-01-15
(3,981 prints) and SOL ran a 2h interval from 2022-11-10 to 2022-12-20 (480 prints). Only settlements reached after
exactly 480 minutes at 00/08/16 UTC are used (`seasonality.funding_settlements`). Prints are known 2 s after
settlement.

**Design.** For settlement T, the origin is T−1h. fz is the z of the latest print known at the origin (the T−8h
settlement) against prints in the trailing 90 days (≥ 90 prints, clipped ±5; the H-LEV5 definition), and
x = sign(rate)·|fz|. The pre return is z of [T−1h, T) and the post return is z of [T, T+1h).

| variant | segment | events | dates | b_pre | t | b_post | t |
|---|---|---|---|---|---|---|---|
| **registered x** | llm_in_sample | 119,844 | 2,259 | −0.0095 | −1.38 | +0.0058 | 0.73 |
| | clean_holdout | 8,712 | 132 | −0.0105 | −0.47 | +0.0480 | 2.34 |
| | pooled | 128,556 | 2,391 | −0.0096 | −1.45 | +0.0083 | 1.09 |
| scale-only x = rate / RMS (diag.) | llm_in_sample | | | −0.0133 | −1.62 | +0.0044 | 0.49 |
| | clean_holdout | | | −0.0205 | −0.90 | +0.0550 | 2.56 |
| placebo at 04/12/20 UTC (no settlement) | llm_in_sample | 118,951 | | **−0.0199** | **−2.40** | +0.0003 | 0.05 |
| | clean_holdout | 9,108 | | +0.0004 | 0.02 | +0.0181 | 0.87 |

The signs point the way the hypothesis predicts, but the effect is tiny: 0.01 σ per unit of x, about 0.5 bp on BTC.
It isn't significant in-sample, and it isn't specific to settlement. The mid-interval placebo shows a pre-hour lean
twice as large, so what there is is a generic funding-contrarian drift at the 1h horizon (H-P1 failed at 4h), not
flow that clusters at settlement. The holdout post coefficient (t 2.3) comes from the x ≥ 1 bucket (post z +0.18,
n 1,197) in four months, and it doesn't appear in-sample (+0.04). Even at face value, 0.05 σ per hour (about 2–3 bp
on BTC) is far below the 15 bp round-trip cost (7.5 bp per side).

## H-NS2 Options expiry (BTC, ETH)

**Design.** I used 24h windows anchored at 08:00 UTC (the Deribit expiry time), with RV built from 1h bars and a
per-asset log-HAR (`vol_panel`: lags 1/7/30 windows, expanding, refit every 30) fitted on that 08:00-anchored series.
The residual is e = log RV − log F. Every Friday is an expiry, so the control can't be "a non-expiry Friday". The
control, defined before running:
- **C1 (verdict):** pre = Thu 08→Fri 08 and post = Fri 08→Sat 08, each minus the mean of the Tue 08→Wed 08 and
  Wed 08→Thu 08 windows of the same week. Mon→Tue is excluded because its HAR lags hold the quiet weekend. Values
  are averaged over BTC and ETH per week, with NW t over weeks. Known bias: the post window contains Saturday
  00–08 UTC, which the HAR over-forecasts, so d_post is biased against the hypothesis.
- **C2 (the "strongest at monthly" modifier, not in the verdict):** e at monthly (incl. quarterly) Fridays minus e
  at other Fridays (same weekday, so weekday-clean), with Welch t over weeks.

| segment | weeks | d_pre (C1) | t | d_post (C1) | t | monthly − other, pre | t | monthly − other, post | t |
|---|---|---|---|---|---|---|---|---|---|
| llm_in_sample | 298 | −0.007 | −0.16 | **−0.201** | **−4.19** | **−0.232** | **−2.38** | +0.119 | 1.27 |
| clean_holdout | 19 | +0.110 | 0.65 | −0.311 | −1.19 | −0.007 (4 monthly) | −0.01 | −0.423 | −1.21 |
| pooled | 317 | +0.000 | 0.01 | −0.207 | −4.28 | −0.220 | −2.29 | +0.090 | 0.98 |

Quarterly − other (in-sample): pre −0.09 (t −0.6), post +0.03 (t 0.2), with 23 quarterlies. BTC and ETH agree (BTC
d_post −0.20, t −3.9; ETH −0.22, t −4.0).

There is no weekly pin-and-release. The 24h before a Friday expiry is as loud as Tue–Thu relative to the HAR, and the
24h after is quieter (log −0.2, about 18% less variance). That is at least partly the Saturday-morning bias stated up
front, so the post leg says "no release", not "anti-release". The one interesting piece is post-hoc: in-sample, the
24h before a **monthly** expiry is 21% quieter than before a weekly one (t −2.4). The holdout has 4 monthly expiries,
so this is proposed below as a new hypothesis (H-NS6) for forward testing, not claimed.

## H-NS3 Session seasonality (universe L)

Sessions use the bar's open hour: Asia 00–08, Europe 08–13, US 13–20, late 20–24 UTC. z is averaged over symbols per
timestamp, then per (date, session), and the test is a HAC Wald test of equal means.

| segment | dates | Asia | Europe | US | late | F | p |
|---|---|---|---|---|---|---|---|
| llm_in_sample | 2,273 | −0.003 (t −0.6) | −0.000 (0.0) | −0.008 (−1.0) | +0.018 (2.0) | 1.86 | 0.133 |
| clean_holdout | 132 | −0.005 (−0.3) | +0.042 (1.4) | +0.013 (0.4) | −0.043 (−1.1) | 1.31 | 0.273 |

The Spearman correlation of session ranks, in-sample vs holdout, is **−0.40**. The tilt (long the in-sample best,
"late"; short the worst, "US"; 7.5 bp per side) loses −0.35 R/day net in-sample (Sharpe −1.8; gross only +0.13), and
in the holdout −0.89 R/day net (t −3.1, Sharpe −5.1; gross −0.28). The lore ("US sells, Asia buys") isn't in the
data. The in-sample hour-of-day means peak at 20–22 UTC (+0.04 σ) and dip at 13–14 UTC (−0.04 σ), and both flip in
the holdout.

## H-NS4 Hour-of-week volatility profile (universes L and T)

The factor comes from `seasonality.seasonal_factor` (expanding, κ 20, a 3h triangular kernel, cap 50, floor 0.1,
≥ 8 weeks of history). It multiplies s_t². The paired per-bar QLIKE difference is averaged per date, with NW t over
dates.

| universe | segment | dates | bars | ΔQLIKE (registered: expanding vs sigma_risk_next) | t | trailing 26w | t | base sigma_rb_next | t |
|---|---|---|---|---|---|---|---|---|---|
| L | llm_in_sample | 2,218 | 919,974 | **−0.040** | **−8.12** | −0.032 | −6.87 | −0.026 | −6.39 |
| L | clean_holdout | 132 | 72,841 | **−0.076** | **−4.49** | −0.058 | −2.31 | −0.059 | −3.78 |
| T | llm_in_sample | 459 | 21,350 | **−0.293** | **−9.47** | −0.278 | −9.51 | −0.125 | −7.25 |
| T | clean_holdout | 132 | 40,650 | **−0.384** | **−15.81** | −0.369 | −15.20 | −0.239 | −13.51 |

**Where the gain comes from** (descriptive, bar means). For crypto it is mostly weekday intraday shape: ΔQLIKE is
−0.060 on non-weekend bars vs −0.019 on TradFi-weekend bars in-sample. So it doesn't conflict with R-VOL-2 (no weekend
shrink for stops). For TradFi perps it is both (−0.33 weekday, −0.31 weekend in-sample), because their weekends are
frozen. The expanding factor beats the trailing 26-week one, so the shape is stable.

**The profile known at as_of** (mean of the per-symbol factors; 1 = average hour):

| UTC hour | 00 | 01 | 02 | 03–07 | 08 | 09–11 | 12 | 13 | 14 | 15 | 16 | 17 | 18–23 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| L | 1.12 | 1.14 | 0.97 | 0.75–0.84 | 0.91 | 0.80–0.87 | 1.08 | 1.35 | 1.52 | 1.44 | 1.22 | 1.05 | 0.94–0.97 |
| T | 1.08 | 1.02 | 0.84 | 0.70–0.85 | 0.90 | 0.77–0.94 | 1.50 | 2.07 | 1.83 | 1.23 | 0.92 | 0.82 | 0.82–1.04 |

Day-of-week mean: L Mon 1.24 … Sat 0.71; T Mon 1.46 … Sat 0.52. The loudest hours of the week for L are Mon 00–01 and
Sun 22–23 UTC (1.8–2.1×: the CME reopen and the weekly open), then the US open hours (Mon/Tue/Fri 14 UTC, about
1.75–1.8×). For T, Mon 13 UTC is 3.6× and Sun 22–23 is 3.3×. The quietest are Saturday 03–11 UTC (0.57–0.65× L,
0.47× T). Funding hours show only a small bump (08 UTC 0.91 vs 0.84/0.87 on either side; 00 UTC 1.12 vs 0.97 at 23),
so the 1h bars can't separate it from the Asia/Europe opens (see H-NS10 below).

## H-NS5 QQQ first NYSE hour leads BTC and ETH

The signal bar is the 1h bar containing 09:30 America/New_York (13:00 UTC under EDT, 14:00 UTC under EST), with NYSE
holidays dropped. x is QQQ z, and y is the next 3h of BTC (ETH) log return divided by σ̂·√3. The QQQ perp has 1h
bars from 2026-06-06 (after listing burn-in), so the sample runs 2026-06-08 to 2026-11-09.

| segment | days | obs | slope | t | same-hour β (BTC/ETH z on QQQ z) | t |
|---|---|---|---|---|---|---|
| llm_in_sample | 16 | 32 | −0.075 | −0.81 | 0.81 | 5.0 |
| clean_holdout | 92 | 184 | −0.047 | −0.50 | 0.19 | 3.2 |
| **pooled (primary)** | 108 | 216 | **−0.050** | **−0.63** | 0.29 | 3.2 |

BTC alone: −0.052 (t −0.66); ETH alone: −0.049 (t −0.59). The open hour is shared, and nothing carries over into the
next 3h. With about 100 days, slopes above +0.15 are excluded at about 2 SE.

## IDEAS: draft hypotheses for the orchestrator (not tested, not recorded)

Each one comes from data seen in this study or from a lake-coverage check (Binance 1m bars: ETHUSDT only,
2023-01 → 2026-07; Bybit has no 1m bars but has 5m; liquidations: 3 days, 2026-11-08 → 11-10; Deribit perp 1h bars and
perp funding to 2026-10-10; DVOL hourly from 2021-03). Ideas built on in-sample observations from this run (H-NS6,
H-NS10) are post-hoc, so only holdout and forward data can count for them.

```yaml
H-NS6:
  title: "Monthly Deribit expiry compresses BTC/ETH vol in the 24h before it (post-hoc from H-NS2 C2)"
  statement: "log(RV / log-HAR) over Thu 08:00 -> Fri 08:00 UTC is lower before monthly (last-Friday) expiries than before weekly ones."
  test: "08:00-anchored windows as H-NS2; monthly minus weekly e_pre, Welch t over weeks; holdout and forward only (in-sample -0.23, t -2.4 was the discovery)."
  success: "difference < 0 in the clean holdout once >= 12 monthly expiries have accrued, one-sided p < 0.10."
  prior: 0.30
H-NS7:
  title: "Hour-of-week-scaled quote widths improve market-making markout"
  statement: "Scaling the atx mm half-spread by sqrt(hour-of-week factor) (features.seasonality) raises markout-adjusted P&L per quote-hour versus flat widths at the same average width."
  test: "B4 replay (sim/replay.py) of target_quotes on real 3s paths, flat vs seasonal width, same mean width; paired per-hour difference, block bootstrap."
  success: "paired P&L difference > 0 with the bootstrap CI excluding 0 on held-out days."
  prior: 0.50
H-NS8:
  title: "Seasonal stop distances cut noise stop-outs without worse losses"
  statement: "For 1h strategies (shock_regime, 4h trend entries), stops at k * sigma * sqrt(f_how) rather than k * sigma lower the stop-out rate in loud hours (US open, Sunday reopen) without raising the mean loss per stopped trade."
  test: "deterministic replay on 1h bars L, same k, both segments; stop-out rate and mean R per trade; NW t."
  success: "stop-out rate lower and mean R not lower (t > -1) in both segments."
  prior: 0.40
H-NS9:
  title: "TradFi perp Sunday-reopen hour corrects the weekend perp drift"
  statement: "On Bybit TradFi perps, the return over the first hour after the CME/futures reopen (Sun 18:00 ET) is negatively related to the perp's Fri-close-to-reopen weekend drift (thin weekend quotes overshoot)."
  test: "universe T, weekends with the reopen bar, OLS of reopen-hour z on weekend z, date-clustered SE; holdout-dominated."
  success: "slope < 0 with t < -2 pooled, same sign in the holdout."
  prior: 0.30
H-NS10:
  title: "Funding settlement minutes carry a short vol spike worth widening quotes for"
  statement: "In flow_3s, |return| in the 5 minutes after 00/08/16 UTC settlements exceeds the same minutes one hour earlier and later by > 20%."
  test: "flow_3s 1m rollups, universe L, paired by settlement, bootstrap CI over settlements; holdout only (collector data)."
  success: "ratio > 1.2 with CI excluding 1 after >= 30 days of flow_3s."
  prior: 0.45
H-NS11:
  title: "Implied vol times the hour-of-week shape beats the robust sigma times the shape for BTC/ETH next-hour vol"
  statement: "Replacing the base sigma with DVOL (hourly, converted to a per-hour variance) or a blend of DVOL and robust sigma, both multiplied by the hour-of-week factor, lowers next-hour QLIKE further."
  test: "as H-NS4 for BTC/ETH; paired QLIKE vs the H-NS4 model; NW t over dates; segments."
  success: "difference < 0 with t < -2 in both segments."
  prior: 0.35
H-NS12:
  title: "Liquidation intensity peaks at hour-of-week vol peaks more than vol alone implies"
  statement: "Bybit liquidation notional per hour, divided by the hour-of-week vol factor, is still higher in the US-open and Sunday-reopen hours."
  test: "liquidations table, universe L, Poisson/NB regression of hourly counts on log factor plus hour-group dummies; forward only once >= 30 days are collected."
  success: "hour-group coefficients > 0 with t > 2 after controlling for the factor."
  prior: 0.30
H-NS13:
  title: "Binance ETH leads Bybit ETH at 5 minutes"
  statement: "The Binance ETHUSDT 5m return (built from 1m bars) predicts the next Bybit ETHUSDT 5m return beyond Bybit's own lag."
  test: "2023-01 to 2026-07 overlap, OLS of Bybit r_{t+1} on Binance r_t and Bybit r_t, date-clustered SE; segments."
  success: "Binance coefficient > 0 with t > 2 in both segments, and gross edge > 2 x 7.5 bp per signal."
  prior: 0.10
H-NS14:
  title: "Deribit-Bybit funding differential predicts the Bybit 8h return"
  statement: "When Bybit's latest 8h funding exceeds Deribit's perp funding (lake alt_series deribit perp_funding_8h) by > 2 trailing SDs, the next 8h Bybit BTC/ETH return is negative (cross-venue crowding)."
  test: "BTC/ETH, settlements to 2026-10-10, OLS of next-8h z on the standardised differential, date-clustered SE; segments."
  success: "slope < 0 with t < -2 in-sample, same sign in the holdout."
  prior: 0.20
```

## Caveats

- **Contamination.** Session lore, expiry pinning (Ni, Pearson & Poteshman 2005) and funding-settlement lore are all
  in the training data. Only H-NS4's holdout and H-NS5 (almost all holdout) are clean evidence.
- **Holdout samples are short:** 132 days, 19 expiry weeks (4 monthly), 92 QQQ open days. H-NS1's holdout "supports"
  meets only the sign criterion, and its in-sample row is inconclusive, so the hypothesis as a whole is not supported.
- **H-NS2 control.** Every Friday is an expiry, so C1 compares weekdays, and weekday vol seasonality (Thursday US data,
  Saturday morning) is in the contrast. The post-window bias is documented in the module docstring. C2 is clean but
  was not the registered test.
- **H-NS4** measures central-forecast quality (QLIKE) at 1h. It says nothing about tail loss across weekends, where
  R-VOL-2 still governs stops. The floor (0.1) and cap (50) are untuned defaults. TradFi in-sample covers only
  ~2026-01 to 06 (PAXG/XAUT longer).
- **Survivorship:** universe L is today's survivors.
- **Bug found before recording** (diary, kind mistake). NaN vs null in polars dropped NS2 weeks with one missing asset
  and blanked the NS1 bucket table. It was fixed before the single recorded run, and the test design did not change.

## Rule candidates

```yaml
- id: R-VOL-26
  kind: prefer
  rule: WHEN forecasting next-hour variance for quoting or intraday expectations THEN multiply the robust per-bar sigma squared by the expanding hour-of-week factor (features.seasonality.seasonal_factor)
  when: horizon <= a few hours; central expectation or quote width
  then: sigma_1h_expected = sigma_risk_next * sqrt(f_how), f_how known at the bar open; stops across weekends still follow R-VOL-2
  scope:
    assets: crypto perps (universe L) and TradFi perps (universe T)
    horizon: 1h
    sides: both
  effect: next-hour QLIKE -0.040 (t -8.1) in-sample / -0.076 (t -4.5) holdout for crypto; -0.29 (t -9.5) / -0.38 (t -15.8) for TradFi perps; crypto gain mostly from weekday intraday shape
  confidence: A
  evidence: [H-NS4]
  review: a holdout quarter where the expanding factor no longer lowers 1h QLIKE (t > -2), or a quoting replay (H-NS7) showing no markout gain
  status: active
- id: R-VOL-27
  kind: expect
  rule: WHEN it is Sunday 22:00 to Monday 02:00 UTC or the US cash open hours (13:00-16:00 UTC) THEN expect hourly variance well above the weekly average, and WHEN it is Saturday 03:00-11:00 UTC THEN expect it well below
  when: hour of week in the loud or quiet set of the current profile
  then: crypto about 1.75-2.1x the average hour at the Sunday reopen and the US open, about 0.6x on Saturday mornings; TradFi perps about 3.3-3.6x at Monday 13 UTC and the Sunday reopen, about 0.5x on Saturdays
  scope:
    assets: crypto perps and TradFi perps
    horizon: 1h
    sides: both
  effect: profile known at 2026-11-10 from the H-NS4 model (expanding, all history)
  confidence: B
  evidence: [H-NS4]
  review: refresh the profile monthly; demote if the top or bottom hours change set
  status: active
- id: R-AVOID-4
  kind: avoid
  rule: WHEN tempted to tilt direction by trading session (Asia, Europe, US, late UTC) THEN do not; session means are equal and the ranking flips out of sample
  when: any session-of-day directional tilt
  then: no session tilt
  scope:
    assets: crypto perps (universe L)
    horizon: 1h-8h
    sides: both
  effect: in-sample F 1.86 (p 0.13); rank correlation in-sample vs holdout -0.40; tilt net -0.35 R/day in-sample, -0.89 R/day (t -3.1) holdout
  confidence: A
  evidence: [H-NS3]
  review: none planned
  status: active
- id: R-PX-12
  kind: avoid
  rule: WHEN the QQQ perp makes a large move in the first NYSE hour THEN do not expect BTC or ETH to follow over the next 3 hours; the co-move is same-hour only
  when: bar containing 09:30 ET on NYSE days
  then: no directional follow-through trade on BTC/ETH from the QQQ open hour
  scope:
    assets: BTCUSDT, ETHUSDT
    horizon: 3h
    sides: both
  effect: pooled slope -0.05 (t -0.63, 108 days); same-hour beta 0.29 (t 3.2)
  confidence: C
  evidence: [H-NS5]
  review: re-run with >= 250 open days
  status: provisional
- id: R-CARRY-2
  kind: avoid
  rule: WHEN a funding settlement is about an hour away THEN do not expect price to lean against funding into it or rebound after it beyond the generic hourly funding drift
  when: 1h before or after a 00/08/16 UTC settlement
  then: no settlement-timed directional trade
  scope:
    assets: crypto perps (universe L, 8h funding)
    horizon: 1h
    sides: both
  effect: pre b -0.010 (t -1.4), post +0.006 (t 0.7) in-sample; mid-interval placebo pre b -0.020 (t -2.4); 0.01 sigma per unit x is far below 7.5 bp costs
  confidence: B
  evidence: [H-NS1]
  review: a holdout year with post b > 0 at t > 2 and a null placebo
  status: provisional
- id: R-VOL-28
  kind: expect
  rule: WHEN a weekly Deribit Friday 08:00 UTC expiry approaches THEN do not expect pinned (lower) realised vol into it or a release after it in BTC or ETH
  when: Thu 08:00 to Sat 08:00 UTC
  then: use the ordinary vol forecast; a monthly-expiry pre-window compression is under forward test (draft H-NS6)
  scope:
    assets: BTCUSDT, ETHUSDT
    horizon: 24h
    sides: both
  effect: d_pre -0.007 (t -0.2), d_post -0.20 (t -4.2) in-sample; holdout +0.11 / -0.31 (19 weeks)
  confidence: B
  evidence: [H-NS2]
  review: H-NS6 forward result after 12 monthly expiries
  status: active
```
