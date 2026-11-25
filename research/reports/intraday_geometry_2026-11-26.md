# Intraday structure and market geometry (H-GEO1..H-GEO17), 2026-11-26

Agent GEO. Research only: no orders, no private or trading endpoints.

- **Registration.** The hypotheses were registered before any fit: research/hypotheses.yaml, block GEO, commit 6fcc062.
- **Design freeze.** The design was frozen in the diary (2026-11-25 22:25 UTC) and in `data/research/intraday_geometry/design.json`.
- **Clarifications.** Implementation clarifications were logged before inference (diary, kind test).
- **Code.** `src/ai_trader/research/intraday_geometry.py`, with tests in `tests/test_intraday_geometry.py` (9).
- **Outputs.** `data/research/intraday_geometry/`: map, events, grid, infer, hetero, rules and liq results.
- **Records.** All 33 verdicts are recorded with `ai_trader.research.hypotheses.record`.

## Verdicts

| hypothesis | llm_in_sample | clean_holdout | key numbers |
| --- | --- | --- | --- |
| H-GEO1 hour-of-week vol / volume shape is stable | supports | inconclusive | Spearman of the in-sample vs holdout 168-cell profile: 0.77 for variance (needed 0.8), 0.81 for volume. Months with rho >= 0.5: 93% / 96% in-sample, 80% / 80% holdout. |
| H-GEO2 tradable hour-of-week direction | inconclusive | **contradicts** | Joint chi2 185.7 on 168 df, p 0.17. 8 cells have \|t\| > 2 (chance: 7.6). In-sample vs holdout rank rho 0.03. Tilt: gross -0.025 R/h against a cost of 0.21 R/h. |
| H-GEO3 release-minute variance spike | **supports** | **supports** | First 5m after CPI / NFP / FOMC: 13.6x baseline [9.4, 18.8] in-sample, 16.5x [4.6, 36.5] holdout (n 13). 30-60 min after: 2.9x / 1.7x. |
| H-GEO4 funding-settlement 5m variance | inconclusive | **contradicts** | 08+16 UTC vs neighbouring top-of-hour bars: 1.20x [0.99, 1.51] in-sample, 0.71x [0.53, 0.97] holdout. |
| H-GEO5 drift against the funding payer into the print | inconclusive | inconclusive | Pre-settlement signed z: +0.018 (t 1.8) in-sample, +0.014 (t 0.4) holdout. |
| H-GEO6 session-open vol step, US follows NY time | inconclusive | supports | US cash open, 30 min after / before: 2.9x in-sample, 3.0x [2.5, 3.7] holdout. The off-clock check passes in EDT (1.08) but fails in EST (1.75, the 08:30 ET slot). Asia 2.7x, London 1.8x. |
| H-GEO7 month-end fix | **contradicts** | **contradicts** | 15-16 UTC on the last day vs other weekdays: 0.77x in-sample, 0.42x holdout. Direction t 0.5. |
| H-GEO8 spreads track vol by hour | - | **contradicts** | Spread vs variance rho -0.01, depth vs variance rho -0.39 (18 days). Spreads are widest in Asia hours, when variance is lowest. |
| H-GEO9 a geometry cell beats placebo levels | **supports** | **supports** | 15 of 156 cells survive BH 10% in-sample and 4 pass the holdout BH. All four are continuation after the first touch or sweep of the prior day's or week's high/low. |
| H-GEO10 PDH/PDL sweep-and-reclaim reverses | **contradicts** | inconclusive | +0.026 (t 1.0) in-sample, the wrong sign. -0.063 (t -0.7) holdout. |
| H-GEO11 30-min opening-range breakout continues | **contradicts** | inconclusive | -0.009 (t -0.3) in-sample. London +0.11 (t 2.0) in-sample, -0.09 holdout. |
| H-GEO12 first session-VWAP cross continues | inconclusive | **contradicts** | +0.072 (t 1.6) in-sample, -0.128 (t -1.1) holdout. |
| H-GEO13 round numbers hold | **contradicts** | **contradicts** | Round numbers break more than placebo levels: +0.050 (t 2.9) in-sample, +0.038 holdout. |
| H-GEO14 Asia-box sweep in the London morning reverses | **contradicts** | inconclusive | +0.022 (t 0.6) in-sample, -0.063 (t -0.7) holdout. |
| H-GEO15 regime / session dependence persists | inconclusive | **contradicts** | Wald p < 0.01 in 1 of 30 cell x grouping units. Holdout rank rho <= 0 in 11 of 18 testable units. |
| H-GEO16 an executable geometry rule | **contradicts** | **contradicts** | All 13 rules lose in the walk-forward (-0.008 to -0.104 R/trade) and in the holdout (-0.07 to -0.22 R). DSR 0.00 against 2,576 trials. |
| H-GEO17 (exploratory) liquidation-cluster levels | - | contradicts | Touch -0.07 (t -1.0) over 16 days. A sweep-and-reclaim reverses -0.30 (t -2.7), single segment only. |

Effects are in vol units. Event effects are the real level minus the placebo mean of the continuation-signed forward return, divided by sigma_slow x sqrt(bars). t-stats are day-clustered (Newey-West lag 2 over days). Each verdict applies its registered success rule mechanically. Treat |t| ~ 2 as interesting, not proven.

## In plain words

### 1. The intraday map: volatility is forecastable, direction is not

**Volatility.** This part is real and stable. It confirms and sharpens R-VOL-26 / R-VOL-27.

**Hour-of-day variance** (24 perps pooled, mean = 1). These are the same numbers as in the table below:
- The day is quietest from 03:00 to 11:00 UTC, at 0.69-0.85x.
- Variance starts rising at 12:00 UTC (1.06x).
- It peaks at 14:00 UTC (1.6x in-sample, 1.76x holdout) and stays high through 16:00 (1.37x).
- There is a smaller bump at 00:00-01:00 UTC (1.1-1.3x).

**Hour of week.**
- The loudest cells are 14:00-15:00 UTC on every weekday: 1.8-2.05x the average hour.
- The quietest are Saturday and Sunday mornings, 05:00-11:00 UTC: 0.47-0.53x.
- Volume has the same shape.
- The shape holds month by month: 93% of in-sample months and 80% of holdout months correlate at rho >= 0.5 with the rest of the sample.
- It only just misses the registered 0.8 cross-cutoff bar for variance (0.77). That miss is mostly because the holdout's 00:00 UTC hour is quieter and 15:00 is louder.

**Scheduled US data is the single largest intraday volatility event.**
- The first 5 minutes after CPI carry about 22x the variance of the same minute on a normal day. FOMC carries about 17x and NFP about 5x (in-sample).
- The holdout shows the same: 16.5x pooled over 13 releases.
- The excess halves within 5-15 minutes and is about 2-3x by 30-60 minutes.
- The 30 minutes before a release are already 1.3-1.4x.

**The US cash open (09:30 New York) roughly triples 5m variance.**
- The 30 minutes after the open carry about 3x the variance of the 30 minutes before (median-type ratio about 1.9x).
- This holds in both segments.
- The step moves with US daylight saving: 13:30 UTC in summer, 14:30 UTC in winter.
- In winter the off-clock 13:30 UTC slot also jumps, at 08:30 ET. That is the US data-release minute: only CPI and NFP were excluded, and other 08:30 ET releases print there. So the registered DST check fails on a technicality. The economics point the same way: crypto's intraday vol clock is the New York clock.

**Other session effects.**
- The 00:00 UTC step (2.7x) mixes the Asia open, the funding print and the daily candle roll.
- London's step is modest (1.5-1.8x mean of ratios, 1.15-1.2x ratio of means).

**What is *not* a vol event.**
- **Funding settlements.** The 08:00 and 16:00 UTC 5m bars are no louder than the neighbouring top-of-hour bars (holdout 0.7x).
- **The month-end London fix.** Quieter than other days, not louder.

**Spreads.** In the 18 days of book data, spreads are widest in Asia hours, when variance is lowest. They are tightest in Europe and US hours (about 09:00-16:00 UTC), when it is highest.
- Spreads track participation, not inventory risk.
- Practical consequence: a pegged limit is cheapest to work in the loud European/US hours.
- Taker stops are relatively more expensive in quiet Asia hours, because the spread is wider there while the expected move is smaller.

**Direction: noise, as expected (R-PX-6, R-AVOID-4 confirmed at hour x weekday resolution).**
- The 168 hour-of-week mean returns are jointly indistinguishable from zero (p 0.17).
- The in-sample ranking does not predict the holdout ranking (rho 0.03).
- Holding the 8 "significant" hours would have lost 0.24 R per held hour after costs.
- No funding-window drift either (t 1.8 in-sample, 0.4 holdout).
- No month-end drift.

### 2. Geometry: which objects matter, and how

**The only geometry that matters is the prior day's and the prior week's high and low.** What they do is break, not hold.
- **Prior-day high/low.** The first touch carries on further than the same touch of a placebo level placed ±0.5-1.5 daily sigma away:
  - 15 min: +0.18 sigma in-sample (t 7.0) and +0.15 holdout (t 2.2);
  - 1 h: +0.10 sigma (t 5.6) and +0.06 holdout (t 1.3).
- **Prior-week high/low.**
  - Touch at 1 h: +0.33 sigma in-sample (t 6.3), +0.26 holdout (t 2.6).
  - Sweep, 1 h: +0.21 / +0.31.
- These four cells are the H-GEO9 holdout survivors.
- In plain terms: when price reaches yesterday's or last week's extreme, stops and breakout orders resting there push it through. This matches R-PX-4 (levels break slightly more often than they hold).

**The effect is a tail, not a hit rate.**
- The continuation probability after a PDH/PDL touch is 0.46-0.49 at 15 min-1 h, about the same as for placebo levels.
- The positive mean comes from the occasional run, not from a higher win rate.

**How big is it?** About 5-8 bp over 15 min-1 h on an average 71 bp hourly sigma.

**A confound checked and ruled out (post-hoc, not registered).** Placebo levels sit further from the price, so they are touched later in the day. Matching real and placebo touches on time-into-window leaves the effect unchanged (15 min: 0.177 -> 0.177 in-sample, 0.146 -> 0.151 holdout).

**Round numbers.** They break slightly more than placebo levels in-sample (+0.05 sigma at 1 h). They do not hold, so the Osler-style "round numbers as support" claim (H-GEO13) is contradicted.

**What does not matter, in both segments:**
- **Session opening ranges** (15/30/60 min of Asia, London and US). All cells are within ±0.07 sigma and none keeps its sign. The London OR30 breakout looked good in-sample (t 2.0) and flipped in the holdout.
- **Asia box.** Neither its sweep (the "Judas swing") nor its breakout works. The in-sample "retest reverses" survivor (-0.06 at 4 h) has the opposite sign in the holdout (+0.07).
- **Session VWAP and anchored VWAP.** Positive in-sample at t ~ 1.5, negative in the holdout (-0.13 to -0.17 sigma, t ~ -1.1 to -1.5). VWAP is not a line the market respects.
- **Prior-day value-area edges and POC.** All within ±0.05 sigma at every horizon and event. This confirms R-PX-5 on 5m data and for alts.
- **4h pivot zones (scripts/levels.py).** Within ±0.03 sigma. This is consistent with their current use in the market maker: only to push quotes beyond zones, never as directional evidence.
- **Sweep-and-reclaim of the prior-day high/low** (the stop-run fade, H-GEO10). It does not reverse.
- **Regime cell and session.** They do not reliably change any of this (H-GEO15). The in-sample cell differences do not carry over (11 of 18 holdout rank correlations are <= 0).

### 3. Executable edge after costs: none

Every registered rule loses money net of realistic costs, in the walk-forward and in the holdout.
- **The rules.** 8 rules were built from the in-sample survivors and 5 from the classic setups.
- **Exit variants.** Each was tested over 16 exit variants:
  - stop 1 or 2 hourly sigma;
  - a 2R limit target or no target;
  - a time stop of 1 h or 4 h;
  - a flat or hour-of-week-scaled stop.
- **Costs.** Pegged entry 2.0 + 1.3 bp, taker exits 5.5 + 2.0 bp, gapped stops, funding, and worst-case intrabar ordering.

| rule (direction) | in-sample signals | best variant (stop, target, time, scale) | walk-forward net R [95% CI] (n) | holdout net R [95% CI] (n) | +5 min latency (in / ho) | vs regime-matched random (in-sample) |
| --- | --- | --- | --- | --- | --- | --- |
| PD_HL touch, continuation (stop entry) | 32,352 | 2σ, 2R, 1h, scaled | -0.104 [-0.124, -0.082] (25,617) | -0.113 [-0.155, -0.068] (3,176) | -0.094 / -0.113 | -0.018 [-0.034, -0.001] |
| PW_HL touch, continuation | 4,569 | 2σ, none, 4h, scaled | -0.088 [-0.154, -0.021] (3,637) | -0.221 [-0.384, -0.055] (440) | -0.066 / -0.221 | -0.032 [-0.105, +0.045] |
| PW_HL sweep, continuation | 1,708 | 2σ, none, 4h, flat | -0.008 [-0.108, +0.087] (1,389) | -0.156 [-0.328, +0.033] (169) | -0.003 / -0.171 | +0.054 [-0.059, +0.168] |
| ROUND touch, continuation | 44,044 | 2σ, 2R, 1h, scaled | -0.100 [-0.114, -0.087] (34,455) | -0.123 [-0.154, -0.090] (4,120) | -0.093 / -0.123 | -0.023 [-0.034, -0.013] |
| ROUND accept, continuation | 41,584 | 2σ, 2R, 1h, flat | -0.082 [-0.100, -0.064] (32,550) | -0.076 [-0.109, -0.043] (3,879) | -0.072 / -0.069 | -0.003 [-0.015, +0.008] |
| PD_HL accept, continuation | 29,966 | 2σ, none, 1h, flat | -0.063 [-0.085, -0.041] (23,742) | -0.097 [-0.143, -0.051] (2,941) | -0.063 / -0.097 | +0.012 [-0.007, +0.033] |
| PD_C touch, continuation | 31,075 | 2σ, 2R, 4h, flat | -0.091 [-0.114, -0.068] (24,663) | -0.075 [-0.138, -0.004] (3,032) | -0.073 / -0.075 | -0.009 [-0.030, +0.013] |
| ASIA_BOX retest, reversal | 15,374 | 2σ, 2R, 4h, flat | -0.067 [-0.092, -0.042] (12,407) | -0.114 [-0.174, -0.051] (1,442) | -0.064 / -0.121 | +0.039 [+0.016, +0.062] |
| H-GEO10 PDH/PDL sweep fade | 10,548 | 2σ, 2R, 4h, scaled | -0.083 [-0.110, -0.054] (8,390) | -0.166 [-0.238, -0.094] (1,083) | -0.068 / -0.152 | +0.028 [-0.000, +0.057] |
| H-GEO11 OR30 breakout | 115,627 | 2σ, none, 1h, flat | -0.078 [-0.093, -0.062] (92,569) | -0.073 [-0.103, -0.039] (11,319) | -0.073 / -0.081 | +0.006 [-0.002, +0.015] |
| H-GEO12 VWAP first cross | 34,545 | 2σ, 2R, 4h, flat | -0.038 [-0.061, -0.014] (27,403) | -0.095 [-0.158, -0.029] (3,319) | -0.038 / -0.098 | +0.037 [+0.016, +0.059] |
| H-GEO13 round-number fade (limit at level) | 40,061 | 2σ, 2R, 4h, scaled | -0.087 [-0.103, -0.071] (31,671) | -0.111 [-0.155, -0.068] (3,744) | -0.085 / -0.111 | -0.001 [-0.015, +0.013] |
| H-GEO14 Asia-box Judas fade | 4,648 | 2σ, 2R, 4h, flat | -0.096 [-0.128, -0.063] (3,725) | -0.110 [-0.209, -0.003] (511) | -0.078 / -0.143 | +0.019 [-0.023, +0.058] |

**Why the PDH/PDL edge does not pay.** Take the strongest holdout-confirmed effect, the PDH/PDL touch continuation.
- Traded as a stop entry at the level, it is worth only +0.01 to +0.02 R per trade before costs. That uses the best exit (2-sigma stop, 2R target, 1 h).
- Costs on that bracket are about 0.10 R.
- With a 1-sigma stop it is -0.09 R even before costs: touches usually retrace before the tail pays.
- A best-case fill-bar ordering does not change this.

**The cost structure dominates everything else.**
- The best exit is the 2-sigma stop in 12 of 13 rules, because a wider stop dilutes the ~11 bp round trip.
- 5-minute latency barely matters (±0.01-0.03 R).
- Adding each alt's half-spread to the taker legs costs another 0.01-0.03 R.
- This is R-BOS-1 again. At 5m-1h horizons on these levels, the move a level buys you is a few bp, and the round trip is about 11 bp.

**What survives the random-entry control?** Two rules beat regime-matched random entries in-sample by a few hundredths of an R. Both are still net negative, and neither difference holds in the holdout:
- **VWAP first cross:** +0.037 R in-sample, +0.008 holdout.
- **Asia-box retest:** +0.039 R in-sample, -0.016 holdout.

**No ob200 replay was run.** It was only meant for rules that survived the bar-level test, and none did. The pegged-entry cost itself comes from the ob200 replay of R-EXE-1/2.

### Rule candidates (suggestions only; research/rules.yaml not edited)

- **R-GEO-1 (expect, B).**
  - WHEN price makes its first touch of the prior UTC day's or prior week's high or low, THEN expect it to break further more often than to hold, over 15 min-1 h (+0.15 sigma at 15 min vs placebo levels, both segments).
  - Do not fade these levels, and do not place a resting entry just inside them expecting a bounce.
  - Do not trade the breakout either: worth +0.01-0.02 R gross, about -0.10 R net (H-GEO9, H-GEO16).
  - Use it for stop placement: stops just beyond PDH/PDL/PWH/PWL are where the extension runs. Put a protective stop beyond them by a margin, not at them.
- **R-GEO-2 (avoid, B).**
  - WHEN considering an intraday rule on opening ranges, the Asia box, session or anchored VWAP, value-area edges/POC, 4h pivot zones or round numbers, THEN avoid it.
  - None has a direction effect beyond placebo levels that survives the holdout. Every tested rule loses 0.04-0.22 R/trade after costs (H-GEO9..16).
- **R-GEO-3 (expect, A).**
  - WHEN a CPI, FOMC or NFP release is within 30 minutes, THEN expect 5m variance about 15x normal in the first 5 minutes after the print. In-sample that is CPI ~22x, FOMC ~17x, NFP ~5x; the holdout pools all three at ~16x.
  - Expect it to decay to about 2-3x by 30-60 minutes, and the 30 minutes before to be about 1.3x.
  - Widen or pull quotes, and do not enter pegged limits in the release minute (H-GEO3).
- **R-GEO-4 (expect, B).**
  - WHEN it is the 30 minutes after 09:30 New York time on an NYSE day, THEN expect 5m variance about 3x the preceding 30 minutes. That is at 13:30 UTC in summer and 14:30 UTC in winter.
  - The intraday vol peak is 14:00-16:00 UTC; weekend mornings 05:00-11:00 UTC are about 0.5x. Follow the NY clock, not a fixed UTC hour (H-GEO6, H-GEO1; refines R-VOL-27).
- **R-GEO-5 (avoid, B).**
  - WHEN tempted to expect extra volatility or a drift at the funding print (00/08/16 UTC) or the month-end London fix, THEN do not. Neither is louder than the neighbouring hours in the holdout, and neither has a direction (H-GEO4, H-GEO5, H-GEO7).
- **R-GEO-6 (expect, C, 18 days).**
  - WHEN working entries by hour, THEN expect the widest spreads and thinnest books in 00:00-08:00 UTC and the tightest at 09:00-16:00 UTC.
  - Spreads track participation, not variance (H-GEO8).
  - Taker legs cost the most in quiet Asia hours. Review after 60 days of flow_3s.

## Caveats

- **Placebo design.** The placebo levels (±0.5-1.5 daily sigma away) are reached less often than real levels near the price.
  - Example: opening-range accepts are 115k real vs 30k placebo.
  - So the two event populations differ in how far price had to travel.
  - The time-matched diagnostic says this does not drive the PD/PW result. It is not a full distance match.
- **Survivorship.** The universe is today's survivors (23 perps). 5m data starts in 2022, so the in-sample period is mostly a 2022 bear market and the 2023-2025 bull market.
- **Holdout size.** About 4.8 months, 13 releases, 4 month-ends, and 18 days of book data (H-GEO8) and 16 days of liquidations (H-GEO17). These verdicts are weak.
- **Bar resolution.**
  - The rule tests use 5m bars. Fills inside a bar are approximated: limit fills require a 1 bp trade-through, stops are worst-case, and gaps fill at the open.
  - Latency was tested as 0 or 5 minutes (next or next-but-one bar open), not 1 minute.
  - Pegged entries are assumed to fill at the next open + 1.3 bp. That is R-EXE-2's measured drift for BTC/ETH/SOL at paper order values; it is optimistic for thin alts.
- **Off-clock check.** It failed only because other 08:30 ET releases (claims, PPI, retail sales, GDP) are not in the event calendar. Adding them would be a new registration.
- **One placebo draw.** All events come from one seeded placebo draw (2 placebos per level).
- **Overlapping events.** Many events overlap across levels and symbols. Day clustering handles the cross-sectional correlation, not overlapping horizons across days (4 h at most).
- **Implementation clarifications** (diary, before inference):
  - the approach side is the first arming, which matters only for levels starting at the price;
  - the funding sign is the previous settled rate;
  - NYSE holidays are excluded.
- **Disclosure.** A pipeline smoke test printed pooled BTC real-level means before inference. Nothing was changed after it.
- **PART features.** None appeared in `data/research/participants/` during the run, so no participant conditioning was used.

## Method

- **Bars.**
  - Bybit linear 5m klines for universe L: the BOS cache 2022-01..2026-11-15, plus a public REST extension to 2026-11-25 22:00 UTC (1.7 MB).
  - Bars are placed on a complete 5m grid, with listing burn-in as in `bos_regimes_study`.
  - sigma_slow = sqrt(trailing 2,016-bar mean of r^2), shifted one bar.
- **Part 1 (the map).**
  - Per symbol: hourly RV, turnover and return, each divided by its trailing 168-hour mean or sigma. These are pooled over symbols per hour.
  - The 5m standardised r^2 and z are pooled per 5m timestamp.
  - Event windows use the pooled 5m series:
    - releases: same UTC minute on the 20 previous non-event weekdays, bootstrap over events;
    - funding: neighbouring top-of-hour bars, day bootstrap;
    - session opens: 30 min after / before, per-day ratios, day bootstrap;
    - month-end: hour 15 UTC vs other weekdays, month bootstrap.
  - Spread and depth come from lake flow_3s rolled to 5m. Per-symbol hour-of-day profiles are z-scored and the median is taken across symbols; day bootstrap.
- **Part 2 (geometry).**
  - Levels (point-in-time):
    - PD_HL, PD_C and PW_HL;
    - OR15/30/60 for Asia 00:00 UTC, London 08:00 local and US 09:30 ET (NYSE days);
    - the Asia box (00-08 UTC);
    - VWAP_S from 00:00 UTC, and VWAP_A from the last confirmed 1h fractal (k = 3);
    - VA edges and POC from `features.profile`;
    - PIV4H, a vectorised scripts/levels.py with pivots used only after confirmation, tested equal to the script;
    - ROUND.
  - The event machine is a vectorised first test per level window:
    - arm at 0.5 hourly sigma;
    - touch, measured from the level;
    - sweep: >= 0.1 sigma through and a close back, before acceptance;
    - accept: first close beyond;
    - retest: within 24 bars, holding.
  - Outcomes: forward 3/12/48-bar log returns signed by the approach direction, divided by sigma_slow x sqrt(h).
  - Two placebo levels per instance.
  - Inference: real minus placebo with per-day influence functions and a Newey-West lag of 2 over days. BH 10% over 156 cells in-sample, then a one-sided holdout BH among the survivors.
  - Regime: the v0 cell from lake 4h bars (latest 4h close <= the 5m bar close). Sessions follow `research.calendar`. Cell effects are shrunk with w = days / (days + 10).
- **Part 3 (rules).**
  - Survivors are deduplicated over horizons, plus the 5 named setups.
  - Entry by event: pegged at the next open (accept / sweep / retest), a limit at the level (touch fade), or a stop entry at the level (touch continuation).
  - 16 exit variants.
  - Purged expanding walk-forward with test years 2023, 2024, 2025 and 2026H1 (1-day embargo).
  - The holdout uses the variant chosen on all of llm_in_sample.
  - Statistics: CSCV PBO (16 blocks) on daily R per variant, and the deflated Sharpe of the walk-forward daily R against 2,576 trials.
  - Control: 10 random entries per trade, matched on symbol, v0 cell, session and side within ±30 days, with the same exits.
- **Runtime.**
  - map 1.5 min; events about 31 min on 2 workers; infer 5 s; rules 10.6 min; evaluate 11 s; liq 48 s; plus diagnostics about 5 min.
- **Storage.** `data/research/intraday_geometry` is 245 MB (event tables 95 MB, rule trades). The total stays well under the 30 GB budget; no raw files.
