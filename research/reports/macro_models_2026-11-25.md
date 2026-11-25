# Macro models: inflation, policy path, recession risk, and whether 2026 looks like 2018 or 2008 (2026-11-25)

Block MACRO, hypotheses H-MAC3..H-MAC9 (registered 2026-11-25 ~00:35 UTC, commit b83dcf7, before any fit). Code:
`src/ai_trader/research/macro_models.py`, tests `tests/test_macro_models.py`, outputs `data/research/macro_models/`,
forward files `research/macro/forecasts/`. Research only: no orders, no trading endpoints, no rule changes.

Context read for risk assets (desk line `research/live/LATEST.md`, as_of 2026-11-25T00:03Z): BTC in v0 cell **R6 whipsaw**,
-2.4% over 24h, funding +0.0034% (p37, normal), OI +2.0% / 72h -0.6% (rebuilding on the way down, so new shorts), 24h
liquidations 6.9M long vs 1.0M short (a long flush), DVOL 53.1 (p97, implied vol high). ETH and SOL are in the same cell.

## 1. Verdicts in plain words

**Is this 2018 or 2008?** Mostly late 2018, not 2008, but with less room to pivot than in 2018.
- The curve, the equity drawdown, VIX, the labour market and loose financial conditions all look like December 2018.
- None of the 2008 signs are present yet. In mid-2008 the Baa spread was above 3 pp, NFCI was positive (tight), the Sahm
  indicator had triggered (0.73-1.10), payrolls were negative and the Fed had already cut to 2%.
- The real difference from 2018 is inflation. Core PCE is 3.1% now against 1.8% then, and the Taylor (1993) rule
  prescribes 4.5% against the actual 4.0%. In 2018 Powell could pivot on Jan 4 at no inflation cost, because inflation was
  below target. A 2026 pivot would mean easing with inflation 1.1 pp above target, so it is costlier and slower. If credit
  breaks, the "too late" risk is closer to 2008 than to 2018.
- The 10-year real yield is 2.79%, higher than in every comparison date. That is the main tightening channel now.

**Recession / policy-mistake probability (P(NBER recession within 12 months)).** The selected probit uses the term spread
plus the Baa-10y spread level and its 6-month change.
- With November to date it is **19%**. The 80% band is 11-25% and the 90% band 10-28% (24-month block bootstrap). With the
  October month-end inputs it is 13%.
- The same real-time model said **65% in December 2018** and 57% in September 2007. In mid-2008 it said 28%, which was
  wrong: the recession had already started, because the steep post-cut curve hid it. September 2008 also read 29%.
- The term-spread-only probit says 28% now (42% in Dec 2018). The richer model with the real rate and Sahm, which was not
  selected, says 7%.
- **Conditional risk is where the policy-mistake story sits** (selected model):

| scenario (from November-to-date inputs) | P(recession 12m) | 80% |
|---|---|---|
| now | 0.19 | 0.11-0.25 |
| Dec +25 bp, 10y unchanged (curve flattens 25 bp) | 0.23 | 0.15-0.32 |
| Dec + Jan +50 bp, 10y unchanged | 0.28 | 0.20-0.40 |
| +50 bp and Baa spread +50 bp | **0.51** | 0.40-0.65 |
| +50 bp and Baa spread +100 bp (2008-like credit) | **0.74** | 0.60-0.87 |
| pivot (bills -50 bp) with Baa +50 bp | 0.29 | 0.17-0.34 |

  Hikes alone barely move the probability. A hike combined with credit widening doubles to triples it. The variable to
  watch is the credit spread. HY OAS widened 51 bp in the week to Nov 20 (to about 3.63-3.67%), and Baa-10y is only 1.59 pp.

**What the models expect for the upcoming releases, against consensus:**

| event | model median | 80% interval | model P of prints | consensus |
|---|---|---|---|---|
| Oct core PCE m/m (today 13:30 UTC) | **0.24%** | 0.18-0.31 (frozen); 0.16-0.33 (recalibrated) | 0.2: 51%, 0.3: 43%, 0.4: 2% | 0.3 |
| Oct core PCE y/y | **3.19%** | 3.11-3.27 | 3.1: 25%, 3.2: 55%, 3.3: 17% | 3.2 |
| Q3 real GDP 2nd est. (today) | **3.47% SAAR** | 2.68-4.26 | | 3.5 |
| Nov core CPI m/m (Dec 10) | 0.28% | 0.16-0.39 | 0.2: 30%, 0.3: 42%, 0.4: 17% | not yet published |
| Nov core CPI y/y | 2.96% | | | |
| Nov core PCE m/m | 0.25% | 0.14-0.35 | | |
| Nov core PCE y/y | 3.28% | 3.13-3.44 | | |
| Dec core PCE m/m / y/y | 0.23% / 3.20% | 0.12-0.34 / 2.98-3.41 | | |
| FOMC Dec 9: P(hike 25 bp) | 0.33 reaction function; 0.59 bill-proxy model; 0.34 persistence | | | CME FedWatch 0.93-0.96 |
| FOMC path (reaction function) | upper 4.08 after Dec, 4.16 after Jan, 4.23 after Mar (mean) | p10-p90 4.00-4.50 by Mar | | |
| ECB Dec 17: P(hike 25 bp) | 0.29 reaction function; 0.32 persistence | | | +25 bp to 2.75% expected |
| ECB path | DFR 2.57 after Dec, 2.71 after Mar (mean) | 2.50-3.00 | | |

- Today the model leans **softer than consensus on core PCE m/m**: a 0.24% median, with a 0.2 print slightly more likely
  than 0.3. On y/y it agrees with the 3.2% consensus. The bridge reads October core CPI at 0.30% and core finished-goods PPI.
  It does not have the PPI portfolio-management component (FRED's series ends 2022-12), which is often the swing item.
- **The policy probabilities from the models are low compared with the market.** Two reasons, both documented below.
  First, the monthly design counts non-meeting months as holds, which dilutes meeting-month probabilities by about 8/12.
  Second, the Fed reaction function lost to climatology out of sample (H-MAC5 contradicts). For the Dec 9 decision, trust
  market pricing. The bill-proxy model, the best performer out of sample, also leans towards a hike (0.59 monthly).

**How hawkish and dovish surprises map to risk assets (M4, BTC, 2020-04..2026-11, 156 CPI + PCE releases).** The
expectation is the model's own pre-release forecast, not consensus.
- **Core CPI surprises matter.** BTC's 1-day log return falls 0.32 daily sd per 1 sd of surprise (0.18 pp m/m), NW t -2.0
  in-sample. That is about **-0.5% per +0.1 pp hawkish core CPI surprise** at today's ~2.8% daily vol (DVOL 53). The 5-day
  effect is about -0.9%. The holdout since July has 5 CPI releases, all with the same sign and a larger effect.
- **Core PCE surprises have no reliable sign** (t +1.25): PCE is mostly known from CPI and PPI before release day.
  Pooled, the registered test is negative but not significant (t -1.5 / -1.6), so H-MAC8 is inconclusive.
- Combined with M3, the policy read is this. A hawkish inflation surprise is a modest, short-lived negative for BTC. A
  hawkish Fed combined with credit widening is the non-linear risk: recession probability rises from 19% to 50-75% in the
  scenarios. Those are the paths in which risk assets and BTC fall together, the 2008 path. A dovish turn without a
  credit break would look like the January 2019 pivot. That needs inflation to cooperate first: at least two soft core
  prints, i.e. 0.2 or below.

### The comparison table (values as known on each date; ALFRED vintages for revised series)

| feature | 2007-09-17 (pre first cut) | 2008-06-30 | 2008-09-12 (pre Lehman) | 2018-12-18 (pre Powell hike) | 2019-01-03 (pre pivot) | **now 2026-11-24** |
|---|---|---|---|---|---|---|
| Fed upper target, % | 5.25 | 2.00 | 2.00 | 2.25 | 2.50 | **4.00** |
| change over 3 months, bp | 0 | -25 | 0 | +25 | +25 | **+25** |
| 10y - 3m, pp | 0.33 | 2.09 | 2.25 | 0.43 | 0.15 | **0.55** |
| 10y - 2y, pp | 0.40 | 1.36 | 1.51 | 0.17 | 0.17 | **0.29** |
| 10y nominal, % | 4.48 | 3.99 | 3.74 | 2.82 | 2.56 | **4.92** |
| 10y TIPS real, % | 2.21 | 1.48 | 1.79 | 1.01 | 0.88 | **2.79** |
| 10y breakeven, % | 2.27 | 2.51 | 1.95 | 1.81 | 1.68 | **2.12** |
| Baa - 10y, pp | 2.10 | 3.05 | 3.43 | 2.26 | 2.48 | **1.59** |
| HY OAS, % | n/a on FRED | n/a | n/a | n/a | n/a | **3.63** |
| VIX | 26.5 | 24.0 | 25.7 | 25.6 | 25.5 | **26.6** |
| core PCE y/y, % (latest known month) | 1.91 (Jul) | 2.14 (May) | 2.43 (Jul) | 1.78 (Oct) | 1.88 (Nov) | **3.12 (Sep)** |
| headline CPI y/y, % | 2.38 | 4.08 | 5.52 | 2.21 | 2.21 | **3.95** |
| real policy rate (upper - core PCE y/y) | 3.34 | -0.14 | -0.43 | 0.47 | 0.62 | **0.88** |
| unemployment, % | 4.6 | 5.5 | 6.1 | 3.7 | 3.7 | **4.1** |
| unemployment - CBO NROU, pp | -0.40 | +0.50 | +1.10 | -0.91 | -0.91 | **-0.29** |
| real-time Sahm, pp | 0.10 | 0.73 | 1.10 | -0.07 | -0.07 | **0.00** |
| payrolls, 3-month avg, k | 44 | -55 | -81 | 170 | 170 | **137** |
| NFCI (neg = loose) | 0.06 | 0.44 | 0.71 | -0.76 | -0.70 | **-0.48** |
| NASDAQ log drawdown from 1y high, % | -5.2 | -22.1 | -23.5 | -17.9 | -22.7 | **-14.5** |
| broad dollar, 6m log change, % | -4.5 | -3.3 | +5.7 | +2.4 | +2.2 | **+3.7** |
| WTI, 6m log change, % | +35.2 | +37.8 | -8.4 | -34.5 | -44.3 | **-8.7** |
| inflation gap (core PCE - 2), pp | -0.09 | +0.14 | +0.43 | -0.22 | -0.12 | **+1.12** |
| Taylor 1993 rule (core PCE, Okun gap), % | 2.77 | 2.21 | 2.04 | 3.08 | 3.23 | **4.47** |
| rule minus actual, pp (room to hike if > 0) | -2.48 | +0.21 | +0.04 | +0.83 | +0.73 | **+0.47** |
| P(recession 12m), real-time probit (TS + Baa) | 0.57 | 0.28 | 0.29 | 0.65 | | **0.19 (0.11-0.25)** |
| P(recession 12m), term spread only | 0.39 | 0.10 | 0.07 | 0.42 | | **0.28** |

How to read it:
- **The curve and the drawdown say 2018.**
- **Credit says neither**: it is tighter than both, but it is the variable that moves fastest, and it moved last week.
- **The labour market says 2018** (a mild +0.4 pp drift in unemployment over the cycle, Sahm 0).
- **Inflation, real yields and reaction-function room say neither.** Inflation is above target, so the Fed is hiking
  "by the rule", not hiking into a below-target-inflation slowdown as in 2018. The 2018 "mistake" was cheap to undo; this
  one is not.

## 2. Out-of-sample scores (test once, after selection inside the training window)

### M1 core PCE (first-release log m/m x100; test origins 2015-01..2026-09, COVID target months 2020-03..2021-12 excluded)

| h | selected AR(12) | best VAR (p 6, lambda 0.2) | random walk | 12m mean | DM selected vs better bench (p, one-sided) | 80% / 90% coverage (selected) | COVID RMSE sel / mean12 |
|---|---|---|---|---|---|---|---|
| 0 | 0.0995 | 0.0962 | 0.1296 | 0.0996 | ratio 0.999, p 0.49 | 0.75 / 0.85 | 0.230 / 0.216 |
| 1 | 0.0993 | 0.0983 | 0.1397 | 0.0993 | ratio 1.000, p 0.50 | 0.72 / 0.82 | 0.241 / 0.230 |
| 2 | 0.0990 | 0.0984 | 0.1350 | 0.1007 | ratio 0.983, p 0.35 | 0.74 / 0.82 | 0.237 / 0.235 |
| 3 | 0.0984 | 0.0969 | 0.1274 | 0.1011 | ratio 0.974, p 0.27 | 0.75 / 0.83 | 0.234 / 0.236 |

- **H-MAC3: inconclusive.** The models beat the random walk by 25-30% RMSE but only tie the 12-month mean (the classic
  Atkeson-Ohanian result). Their intervals are fine, slightly narrow.
- **Bridge nowcast (h = 0, H-MAC4)**, set S2 (core CPI + core PPI + lagged PCE), selected on 2010-2014: RMSE **0.0753 vs
  AR(12) 0.0995** (ratio 0.76, DM p 0.002), and RW 0.130. COVID RMSE 0.124 vs 0.230.
- The frozen 80% interval covered only **0.67** (90%: 0.75), which fails the registered [0.70, 0.90] clause, so the verdict
  is **inconclusive**. The point skill is large; the frozen intervals are too narrow. The other sets scored S1 0.0757,
  S3 0.0816 and S4 0.0774.
- The pre-release file therefore also carries a recalibrated interval (scale refit on all ex-COVID errors through the
  origin; df 30, scale 0.067). It is a forward convenience and is never used to score H-MAC4.
- Implied y/y at h = 0: test RMSE 0.10 pp, 80% coverage 0.68, also too narrow. At h = 1-3 it is 0.15 / 0.19 / 0.24 pp with
  coverage 0.69-0.76.
- clean_holdout (target months >= 2026-07, n = 3): AR(12) RMSE 0.041 vs RW 0.089. Recorded as inconclusive (n tiny).
- Q3 GDP second estimate = advance (3.37%) + mean advance-to-second revision (+0.10). The t-fit on 1992-2014 revisions has
  df 9 and scale 0.57. On the 2015-2025 test (ex 2020-21, n 36) it has RMSE 0.39 and 80% coverage 0.92, so it is too wide
  if anything.

### M2 policy (monthly ordered logit; test decision months 2015-01..2026-10 ex 2020-03..2021-12)

| | selected | test Brier model | climatology | persistence | market proxy (3m bill - target mid) | verdict |
|---|---|---|---|---|---|---|
| Fed (target > 0.25 at prior month end, n 105, 19 hikes) | C: inflation gap, u gap, NFCI, level, last move | 0.461 | 0.434 | 0.411 | **0.319** | **H-MAC5 contradicts** (worse than climatology, diff +0.027, CI -0.032..+0.044) |
| Fed ZLB months (n 15, scored separately) | | 0.301 | 0.300 | 0.249 | 0.181 | |
| ECB (n 120, 12 hikes) | A: core HICP gap, last move | 0.292 | 0.333 | **0.288** | n/a | **H-MAC6 inconclusive** (vs clim -0.040, CI -0.111..+0.005; vs persistence +0.004) |

A free market proxy (the bill rate against the target) beats the Taylor-type reaction function by a wide margin. The
reaction function's coefficients have the right signs (inflation gap +0.17, unemployment gap +0.08 per sd, NFCI -0.22),
but "last move" (+1.66) dominates, so it is mostly a persistence model.

### M3 recession probit (pseudo out-of-sample, expanding window, labels purged 24 months; test origins 1990-01..2024-10)

| candidate | validation log score (common origins 1977-1989) | test Brier ex COVID | test AUC ex COVID |
|---|---|---|---|
| TS (term spread) | -0.620 | 0.159 | 0.61 |
| **TS + Baa spread (level, 6m change), selected** | **-0.448** | **0.158** | **0.80** |
| TS + Sahm | -0.501 | 0.162 | 0.72 |
| TS + spread + real FF + Sahm | -1.381 | 0.132 | 0.81 |
| + NFCI | -2.883 | 0.143 | 0.80 |

- **H-MAC7: inconclusive.** The selected model ranks far better (AUC 0.80 vs 0.61) but its Brier is not significantly
  lower: diff -0.003, 95% CI -0.069..+0.048, n 396, 62 positives.
- The 4-variable model has the best test Brier. It was not selected, because it overfit the 1970s-80s validation, and
  re-selecting after the test is not allowed. Its 7% current probability is shown only for transparency.

### M4 BTC response (registered pooled test; llm_in_sample 2020-04..2026-06, clean_holdout 2026-07..)

| sample | 1d slope (sd per surprise sd) | t | 5d slope | t | n |
|---|---|---|---|---|---|
| pooled, llm_in_sample | -0.149 | -1.50 | -0.123 | -1.58 | 147 |
| pooled, clean_holdout | -1.10 | -3.88 | -0.52 | -2.42 | 9 |
| core CPI only, in-sample (sub-analysis, extra trial) | -0.32 | -2.03 | -0.28 | -2.30 | 73 |
| core PCE only, in-sample | +0.14 | +1.25 | +0.14 | +1.50 | 74 |

**H-MAC8: inconclusive** in both segments (in-sample t > -2).

## 3. Forward test

- **Pre-release nowcast committed and pushed at 2026-11-25T00:44:23Z** (commit 0c73a21), 12 h 46 min before the 13:30 UTC
  release: `research/macro/forecasts/2026-11-25_pre_release.json`. It holds October core PCE m/m 0.243 (log x100), 80%
  0.175-0.311, and y/y 3.189%, 80% 3.105-3.274, plus Q3 GDP second estimate 3.47%, 80% 2.68-4.26.
- Forecasts for Nov CPI, the Dec 9 FOMC, the Dec 17 ECB and Nov/Dec PCE are in `research/macro/forecasts/2026-11-25_forward.json`
  (commit 03fd4ee).
- **Scoring after the release** (writes a separate file and never edits the pre-release one):
  `target/debug/atx alt fred --series PCEPILFE,GDPC1`, then
  `PYTHONIOENCODING=utf-8 uv run python -m ai_trader.research.macro_models score`. This writes
  `research/macro/forecasts/2026-11-25_score.json` (absolute error vs consensus, interval hits, the model's probability
  of the actual print).
- H-MAC9 is evaluated after 12 or more scored events.

## 4. Rule candidates (suggestions only; research/rules.yaml not edited)

1. *Credit-gated macro risk flag*: raise the desk's macro-risk flag when Baa-10y rises by 0.5 pp or more over 6 months, or
   when HY OAS widens 75 bp or more in 4 weeks, while the Fed is hiking. In the M3 scenarios this is the step from about
   20% to about 50% recession probability. To be registered and tested before use.
2. *CPI-day sizing*: on US CPI days, treat the 13:30 UTC print as a scheduled jump. M4 gives about -0.5% BTC per +0.1 pp
   core surprise (in-sample t about -2; holdout n 5). This only says to use the existing jump/stop rules around the print;
   it is not a direction signal.
3. *Do not use the Fed reaction function for meeting odds*; use market pricing (H-MAC5 contradicts).

## 5. Caveats and data gaps

- **Consensus history is not available point-in-time** (paywalled). H-MAC4 and M4 use model expectations; today's consensus
  is the one given in the brief (web), not verified.
- **Pseudo real time before first vintages.** ALFRED vintages start late for several inputs: PPI components and WPSFD4131
  2015, CPI components 2011, NFCI 2011, NROU 2011, SAHMREALTIME 2019, euro HICP 2015. Before those dates the first
  vintage's values are used with a conservative release-lag known_ts. Validation windows (2003-2014, 1972-1989) therefore
  partly use revised data. The bridge's 2010-2014 validation is mostly pseudo real time, which may explain why its
  intervals were too narrow on the true real-time test.
- **PPI portfolio management** (PCU523920523920) ends 2022-12 on FRED and was dropped from bridge set S4.
- **Oct 2025 CPI is missing** (shutdown); interior gaps are interpolated geometrically. The 2025-26 release calendar was
  irregular (combined releases); release dates come from the vintages themselves.
- **Daily series with more than 2000 vintages** (DGS*, T10Y3M, DFEDTARU/L, NASDAQCOM) are rejected by the ALFRED API.
  They are unrevised market data, so they come from the FRED CSV with known_ts = obs + 1 day.
- **HY OAS is on FRED only from 2023-11** (ICE licence), so the Baa-10y spread is its historical proxy. HY values for
  2007, 2008 and 2018 are not in the table.
- **M2 is monthly**: non-meeting months count as holds in training, so meeting-month probabilities are diluted (roughly
  x 8/12). FOMC/ECB 2027 meeting months are assumed (Fed Jan and Mar; ECB Feb and Mar). The policy path holds unemployment
  and NFCI flat and uses M1's inflation path.
- **Recession labels** for origins after about 2024-10 depend on NBER not having called a 2025-26 recession (USREC = 0 so
  far). The 24-month purge is conservative but means the current model learns nothing from 2023-2024.
- **M3 sample**: 8 recessions in 1960-2024, so intervals are wide and the "mid-2008 = 28%" miss shows what the model cannot see.
- All p-values are single tests within a block of 7 hypotheses, plus sub-analyses (core CPI-only M4) that count as extra
  trials.
- The comparison table uses the latest vintage known on each date. The Taylor rule uses r* = 0.5 (Taylor 1993 with a
  2% inflation target) and an Okun gap of -2 x (u - NROU); this is an illustration, not a fitted rule.

## 6. Method

**Data.**
- FRED/ALFRED all vintages via `atx alt fred --series ...` into lake `alt_series` (known_ts = vintage date + 1 day,
  `config/datasources.yaml`). Series: PCEPILFE, PCEPI, CPILFESL, CPIAUCSL, CPI components (SAH1, SACL1E, SASLE, SAM2,
  SETG01), PPI (WPSFD4131, PPIFES, PPIFIS, PPIFGS, PCU622110622110, PCU621111621111, PCU481111481111), AHETPI,
  CES0500000003, UNRATE, PAYEMS, NROU, GDPC1, GDPPOT, A191RL1Q225SBEA, GDPNOW, NFCI, ANFCI, USREC, GS10, TB3MS, BAA10YM,
  BAA, AAA, SAHMREALTIME, FEDFUNDS, DFEDTAR, ECBDFR, CP0000EZ19M086NEST, 00XEFDEZ19M086NEST, HY/CCC OAS, DTWEXBGS. About
  23 MB.
- Daily unrevised series come from `ai_trader.narrative.macro`; this commit added the target range, DFF, DTB3, DGS3MO,
  T10Y3M, T10YIE, ECBDFR, BAA10Y and WTI.
- `Vintages.asof(sid, T)` returns the latest value per observation with known_ts <= T. It is unit-tested, including that
  a revision is not visible before its vintage.

**M1.**
- Origin T = 00:00 UTC on the PCE release date of month m (the first vintage that contains m). The target is the
  first-release log m/m of m+h.
- Grid: AR(p), p 1..12. Minnesota VAR(p), p 1..6, lambda {0.05, 0.1, 0.2, 0.5, 1, OLS}, theta 0.5, prior mean 0, on core
  PCE, core CPI, core PPI and AHE m/m plus the UNRATE change, estimated from 1985 on the vintage at T. Bridge OLS from
  1995 on month-m component prints plus lagged PCE.
- Selection is by mean validation RMSE over h = 1..3 (origins 2003-2014, targets <= 2014-12). For the bridge it is the
  h = 0 RMSE on 2010-2014 origins.
- Intervals: Student-t (location 0) fit to the validation errors per horizon. DM uses squared loss, Newey-West with lag
  max(h-1, 1) and the Harvey correction.

**M2.** Month-end Fed upper target (DFEDTAR before 2008-12-16) and ECB DFR.
- Features are known at the first day of the decision month: core PCE y/y gap, UNRATE - NROU, NFCI 4-week mean, level,
  and the sign of the last move within 6 months. For the ECB: core and headline HICP y/y gaps.
- Ordered logit (own MLE, standardised features), expanding window. Selection by validation Brier; block bootstrap with
  12-month blocks.

**M3.**
- Monthly features read at day 12 of the next month: GS10 - TB3MS, BAA10YM and its 6m change, FEDFUNDS - core PCE y/y
  (vintage), SAHMREALTIME, NFCI. Label: USREC = 1 in t+1..t+12. At origin t, training rows are s <= t - 24.
- Probit by Newton MLE. Bands use a 24-month moving-block bootstrap of the training rows (500 refits).
- November 2026 to date uses the DGS10 and DTB3 averages and the BAA10Y average.

**M4.**
- Release day = first vintage date. BTC is the UTC-day close (Deribit perp before 2020-03-25, then Bybit). Returns run from
  the close before release day to release day (1d) and to the close of the 4th day after it (5d), vol-scaled by the sd of
  30 daily log returns.
- Surprise = first release minus the pre-release model forecast: the bridge for PCE, AR(2) for core CPI (p chosen on
  2003-2014). Standardised slope, Newey-West SEs.
