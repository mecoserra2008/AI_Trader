# Vol extensions, fundamentals and funding carry: H-VCL2..5, H-FU1, H-FU2, H-C1 (2026-11-07)

Code: `src/ai_trader/features/vol_models_x.py` (`har_forecast_x`: HAR-X with extra regressors and a direct
h-day horizon; equals `har_forecast` when extra=None, h=1; tests in `tests/test_vol_models_x.py`),
`src/ai_trader/research/vol_fund_study.py` (panels, verdicts; `python -m ai_trader.research.vol_fund_study insample|holdout`).
All verdicts are recorded in `research/hypotheses_results.jsonl`.

## Design (fixed before results)

- Daily RV = Σ squared 1h log returns over UTC days with 24 bars (H-VCL1 construction), 23-symbol universe,
  `market_frame(sym, '1h', now(), funding=False, oi=False)`, everything computed on the full history.
- HAR on log RV (daily, 7d, 30d means), expanding OLS refit every 30 days from day 180, lognormal mean correction.
  Multi-day horizons are direct (target log Σ RV over h days), fitted only on windows fully observed at d and spanning
  h consecutive calendar days.
- Decision time for day d = (d+1) 00:05 UTC. DVOL = latest hourly close known by then (the 23:00 print, at most 3h
  stale); SVI ATM IV = latest 08:00 fit (at most 48h stale). Units: DVOL in vol points, SVI in decimals; both ->
  daily variance IV²/365, the RV unit (the 7d regressor is log(7·IV²/365)).
- Segments by forecast origin; observations whose outcome window straddles the 2026-07-01 cutoff are dropped from both.
- QLIKE = V/F − log(V/F) − 1; paired difference (variant − HAR) averaged across symbols per origin, Newey–West t.
- Fixed choices: H-VCL2 regressor = log max(RS⁻, 1% RV); H-VCL3 regressors = BTC log RV_d and BTC 7d mean log RV
  (joined by date); H-VCL4 primary = DVOL, secondary = SVI `iv_atm_7`, Monday origins (non-overlapping 7d windows),
  HAR and HAR+IV both need 180 fitting pairs; H-VCL5 compression = RV20 ≤ 10th percentile of the previous 365 daily
  RV20 values (≥ 300 finite), outcome realised 5d RV / direct HAR-5d forecast, 95% CI by calendar-week cluster
  bootstrap (2000 reps), signed-return t on the per-date cross-symbol mean of the 5d log return.
- H-FU1: DefiLlama total stablecoins, value usable at its known_ts date; x = 30d log change, z vs trailing 365d
  (≥ 300); Monday 00:00 samples; outcome next-30d log return of BTC and of the equal-weight universe (symbols
  available at the time). H-FU2: VRP = (DVOL/100)² − direct HAR-30d forecast × 365/30, BTC, Monday samples.
  Both: OLS slope with HAC t, lag = max(NW auto, 5) for the overlapping 30d outcomes.
- H-C1: per unit of notional per leg; short perp receives the realised Bybit funding rate × notional at each
  settlement (pays when negative); **the spot leg is assumed to hedge price exactly** (no basis, no spot borrow /
  availability, no margin or capital cost: a real book needs ~1 unit of spot capital plus perp margin per unit of
  notional, so return on capital is lower than the figures here; 217 Bybit perps lack spot, our 23 mostly have it).
  Rebalance Mondays 00:05 UTC; signal = mean funding settled in (t−7d, t]; equal weight among symbols with signal > 0;
  P&L = settlements in (t, t+7d]; cost = 10 bps × 2 legs × Σ|Δw| (initial entry included). Listing burn-in from
  settings (14d, 30d for 2023+ listings) applied to each symbol's first settlement. Annualisation ×52, Sharpe on
  weekly returns, drawdown on the cumulative (uncompounded) sum. An always-on equal-weight book is shown as
  unregistered context only.
- Verdict rule (`verdicts._judge`): in-sample supports if the criterion holds, contradicts if the sign is wrong,
  inconclusive otherwise; holdout supports/contradicts on sign only, inconclusive with fewer than 8 observations.

## In-sample results (`llm_in_sample`, recorded 2026-11-07 22:51 UTC, before the holdout was computed)

| id | metric | value | t | n | verdict |
|---|---|---|---|---|---|
| H-VCL2 | dQLIKE (HAR+RS⁻ − HAR) | +0.0011 (0.3742 vs 0.3730) | +1.34 | 2093 days | contradicts |
| H-VCL3 | dQLIKE (HAR+BTC − HAR), 22 alts | +0.0022 (0.3754 vs 0.3731) | +1.82 | 1883 days | contradicts |
| H-VCL4 | dQLIKE HAR-7d+DVOL − HAR-7d (primary) | −0.0051 (0.1778 vs 0.1829) | −1.44 | 248 weeks | inconclusive |
| H-VCL4 | dQLIKE HAR-7d+SVI7 − HAR-7d (secondary) | −0.0027 | −0.75 | 293 weeks | (variant) |
| H-VCL5 | median realised/HAR-5d after compression | 0.713, CI [0.650, 0.784]; all days 0.711 | ret5 t −1.27 | 5638 events / 927 dates | contradicts |
| H-FU1 | slope next-30d BTC on stablecoin z | +0.026 (corr 0.17) | +1.63 | 321 weeks | inconclusive |
| H-FU1 | slope next-30d EW universe | +0.022 (corr 0.12) | +1.22 | 321 | |
| H-FU2 | slope next-30d BTC on VRP | −0.143 (corr −0.12) | −1.52 | 271 weeks | contradicts |
| H-C1 | net Sharpe / net ann. return | 2.76 / 14.3% (gross 3.51 / 17.7%) | NW t 3.45 | 323 weeks | supports |

H-VCL5 note: the median ratio is below 1 for all days (0.71) because the HAR forecast is a mean and 5-day RV is
right-skewed; the compression subset is indistinguishable from all days (mean log ratio −0.270 vs −0.280), so
there is no conditional under-prediction either.

H-C1 in-sample detail: max drawdown −3.9% net (−1.6% gross), worst week −1.64% net (2022-11-07, FTX: SOL funding
−2.5% per settlement), share of weeks with negative gross carry 5.0% (net 31.0%), mean 14.1 active symbols,
turnover 16.8×/yr, cost 3.4%/yr.

| year | weeks | gross ann. | net ann. | net Sharpe | worst week (net) | neg. gross weeks | active |
|---|---|---|---|---|---|---|---|
| 2020 | 37 | 26.0% | 24.5% | 5.2 | −0.31% | 2.7% | 1.6 |
| 2021 | 52 | 59.5% | 57.3% | 6.0 | −0.19% | 5.8% | 11.2 |
| 2022 | 52 | 3.6% | −2.7% | −1.5 | −1.64% | 7.7% | 13.3 |
| 2023 | 52 | 9.6% | 7.0% | 5.5 | −0.11% | 0.0% | 17.3 |
| 2024 | 53 | 13.1% | 11.4% | 5.6 | −0.14% | 1.9% | 19.2 |
| 2025 | 52 | 4.5% | 0.5% | 0.5 | −0.37% | 5.8% | 18.0 |
| 2026 H1 | 25 | 1.3% | −4.8% | −5.9 | −0.44% | 16.0% | 15.1 |

Context (unregistered): always-on equal weight across all live symbols, in-sample gross 15.8% / net 15.6%,
Sharpe 3.06, max DD −3.0%.

## Holdout results (`clean_holdout`, origins ≥ 2026-07-01, computed once after the in-sample record)

| id | metric | value | t | n | verdict |
|---|---|---|---|---|---|
| H-VCL2 | dQLIKE (HAR+RS⁻ − HAR) | +0.0031 (0.4159 vs 0.4128) | +1.23 | 129 days | contradicts |
| H-VCL3 | dQLIKE (HAR+BTC − HAR) | +0.0047 (0.4143 vs 0.4095) | +1.39 | 129 days | contradicts |
| H-VCL4 | dQLIKE HAR-7d+DVOL − HAR-7d (primary) | −0.056 (0.242 vs 0.297) | −1.98 | 17 weeks | supports (sign) |
| H-VCL4 | dQLIKE HAR-7d+SVI7 − HAR-7d (secondary) | −0.032 | −3.77 | 14 weeks (SVI ends 2026-10-09) | (variant) |
| H-VCL5 | median realised/HAR-5d after compression | 0.589, CI [0.49, 1.21]; all days 0.675 | ret5 t +0.60 | 631 events / 66 dates | contradicts |
| H-FU1 | slope next-30d BTC on stablecoin z | −0.045 (corr −0.33) | −1.17 | 14 weeks | contradicts |
| H-FU1 | slope next-30d EW universe | −0.012 | −0.17 | 14 | |
| H-FU2 | slope next-30d BTC on VRP | +1.49 (corr +0.51) | +6.45 | 14 weeks | supports (sign) |
| H-C1 | net ann. return / net Sharpe | 1.2% / 3.07 (gross 3.6%, Sharpe 15.3) | NW t 1.33 | 17 weeks | supports (sign) |

H-C1 holdout detail: max DD −0.16% net (0 gross), worst week −0.11% net, 0% of weeks with negative gross carry
(41% net), mean 19.9 active symbols, turnover 11.9×/yr, cost 2.4%/yr (two thirds of the gross carry).
Context (unregistered): always-on book 2.65% gross = net, Sharpe 10.3.

## Overall reading

| id | in-sample | holdout | status |
|---|---|---|---|
| H-VCL2 | contradicts | contradicts | rejected: RS⁻ adds noise to log-HAR (wrong sign in both segments) |
| H-VCL3 | contradicts | contradicts | rejected: BTC RV adds nothing to an alt's own HAR |
| H-VCL4 | inconclusive (t −1.44) | supports sign (t −1.98, 17 wks) | not supported yet; consistent small gain |
| H-VCL5 | contradicts | contradicts | rejected: compression does not lead to under-forecast |
| H-FU1 | inconclusive (t +1.63) | contradicts | rejected (and data are backfilled) |
| H-FU2 | contradicts (t −1.52) | supports sign on 14 obs | rejected: the in-sample sign is wrong |
| H-C1 | supports (net Sharpe 2.76) | supports (net +1.2%/yr) | supported, but the premium has decayed |

Which results can be traded:

- **H-C1 (carry).** The premium is real and very steady (gross carry negative in only 5% of weeks, worst week −1.6% at
  FTX), but most of it was earned in 2020–21 (25–60%/yr gross). It has shrunk: 2025 gross 4.5% / net 0.5%,
  2026 H1 net −4.8%, holdout net 1.2%/yr per unit of notional. Turnover from the 7d-sign filter (12–17× per year,
  2.4–3.4%/yr of cost) uses up most of today's carry, and the filter adds nothing: the unregistered always-on book has
  the same or better net (15.6% in-sample, 2.65% holdout). Once you count capital (spot fully funded plus perp
  margin, roughly 1.3–2× notional), the current return on capital is below cash yields. So in the current regime
  it does not pay as a stand-alone book. It could pay as a low-turnover, regime-dependent overlay (scale up when
  trailing funding is high, as in 2021 and 2024), but that is a new hypothesis and needs its own registration.
  Unmodelled risks: basis and hedge slippage in crashes, spot borrow, exchange risk, and funding-interval changes.
- **H-VCL4 (implied vol).** DVOL lowers 7d QLIKE in both segments (−3% in-sample, −19% holdout), but the
  in-sample t is only −1.44, so the effect is not established. It is worth keeping as a feature in BTC/ETH risk
  forecasts (sizing, stops). It is not a trade by itself: a forecast gain of this size says little about whether
  implied vol is mispriced against realised.
- **H-VCL5 (compression).** No edge. Compressed-vol days have the same realised/forecast distribution as all days
  (mean log ratio −0.270 vs −0.280 in-sample), and direction is unpredictable as stated, so there is no
  straddle-style trade on this signal. Note: the registered median-ratio > 1.1 threshold is structurally hard to
  meet against a mean forecast (the median of all days is 0.71), so a future version should compare against the
  all-days baseline.
- **H-FU2.** The holdout t of +6.5 comes from 14 overlapping 30-day outcomes (about 3.5 independent observations)
  and has the opposite sign to 5 years of in-sample data. Read it as noise, not as a reversal.

Trials: 7 hypotheses × 2 segments, plus 2 variants (SVI for H-VCL4, always-on context for H-C1). No parameters were
changed after results were seen. Diary entries were not written in this session (file ownership); the results
are in `research/hypotheses_results.jsonl`.
