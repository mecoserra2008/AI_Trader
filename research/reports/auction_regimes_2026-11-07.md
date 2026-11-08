# Auction equilibrium and regime detection: H-AU1..H-AU4, H-RG1 (2026-11-07)

Code: `features/profile.py` (daily volume profile, tests in `tests/test_profile.py`) and `research/auction_study.py`.
Data: Bybit BTCUSDT 5m (2020-04-01 to 2026-09-27) for the primary sample. Binance ETHUSDT 1m (2023-01 to 2026-07-31)
is the second sample. ETH has only about a month after the cutoff, so it is reported as **in-sample replication only**,
and its numbers are attached to the BTC in-sample records. H-RG1 uses the Bybit 1h 23-symbol universe.
Profiles, 1h bars, strategy signals and outcomes are computed on the full history and then split at the knowledge
cutoff (2026-07-01). An event counts as in-sample only if its outcome is fully known before the cutoff, and as holdout
if the decision is at or after the cutoff. Events that straddle the cutoff are dropped. The in-sample numbers below
were written to file and recorded before the holdout was computed.

## Profile definition (`features/profile.py`)
- Session = UTC day. Each 5m (1m) bar's volume is spread uniformly in price over [low, high]. The grid is a global
  geometric grid with 0.05% bins, so it is the same every day. POC = the max-volume bin. VA = grown from the POC,
  adding the larger adjacent bin at each step (both on an exact tie) until it holds at least 70% of the volume.
  Only complete days (at least 95% of bars) are used.
- Causality: the profile of day D uses only day-D bars and is stamped with the last bar's `known_ts`, which falls at
  the end of D. "Prior VA" for decisions on day D is the profile of D−1. Unit tests check volume conservation, POC and
  VA on known histograms, the 70% share and minimality, that truncating or perturbing later days leaves earlier
  profiles unchanged, and the as-of join.
- BTC in-sample: the median VA width is 1.8% of price (IQR 1.1–2.9%), and the mean VA share is 71.0%.
- Decision bars are 1h bars aggregated from the same intraday bars (complete hours only). σ = `sigma_risk_next`
  (hourly, robust bias-corrected) at the decision close. The first 30 days are skipped while the vol model warms up.

## Results

### H-AU1: acceptance outside the prior VA leads to continuation
Event: the first two consecutive 1h closes beyond VAH (or below VAL) within day D, at most one per side per day.
Outcome: the signed 24h return in σ√24.

| sample | events / days | per-day mean (registered stat) | NW t | driftless null of that stat | **excess** | **t** | first event of day (tradeable) |
|---|---:|---:|---:|---:|---:|---:|---:|
| BTC in-sample | 2277 / 1995 | +0.131 | 5.20 | +0.091 | **+0.040** | **1.58** | +0.042 (t 1.46) |
| ETH in-sample (replication) | 1264 / 1097 | +0.127 | 3.78 | +0.094 | +0.032 | 0.97 | +0.049 (t 1.25) |
| BTC holdout | 87 / 77 | +0.326 | 1.56 | +0.082 | **+0.244** | 1.17 | +0.252 (t 1.21) |

**Correction (recorded before the holdout was run).** The first in-sample row tested the per-day mean against 0 and
said "supports". That is wrong. Averaging the events of a day weights each event by 1/(events that day), and whether a
day ends up with one or two events depends on the rest of that day's path, which overlaps the 24h outcome window. One-sided
days are mostly days where the first acceptance did *not* reverse. Their events average +0.25σ (t 9.6), while events
on two-sided days average −0.62σ (t −18.8). That split is look-ahead, not edge. Gaussian random walks started at the
real 00:00 closes, using the real VA and σ, give this statistic a mean of **+0.091σ** under no drift. Measured
against that null, the in-sample excess is +0.040σ (t 1.58). The unbiased tradeable versions (event-weighted, or
first event of the day) agree: about +0.04σ, t ≈ 1.5.
**Verdict: in-sample contradicts; holdout supports (sign only, t 1.2).** Overall: not supported. Even taken at face
value, +0.04σ over 24h is roughly 12 bps on BTC, which is less than one taker round trip (~15 bps). The holdout
effect comes from the upside (+0.50, n 49), and the downside is negative (−0.11, n 38).

### H-AU2: failed auction rotates to the other side (the "80% rule")
Event: the day opens outside the prior VA, then two consecutive 1h closes come back inside. Race on 1h closes: the
opposite VA edge vs the re-entry edge ∓ 0.5σ. Per-event null b/(a+b).

| sample | events | hit rate | null b/(a+b) | **excess** | **t** | null with discrete-monitoring correction |
|---|---:|---:|---:|---:|---:|---:|
| BTC in-sample | 530 | 0.343 | 0.363 | **−0.019** | −1.00 | 0.384 |
| ETH in-sample (replication) | 285 | 0.358 | 0.350 | +0.008 | 0.29 | 0.374 |
| BTC holdout | 14 | 0.286 | 0.303 | −0.017 | −0.15 | 0.337 |

The opposite edge is reached about 34% of the time, not 80%, and no more often than a random walk predicts. The
median distances are 2.8σ to the target and 1.3σ to the barrier. Races were capped at 720 bars and none in-sample
was left unresolved. The race uses closes, so the continuous b/(a+b) null is slightly generous to the hypothesis. The
Broadie–Glasserman–Kou correction (last column) raises the null and makes the shortfall bigger.
**Verdict: contradicts in both segments.**

### H-AU3: balance vs imbalance decides mean reversion vs trend
Overlap = share of VA(D−1) lying inside VA(D−2). The registry does not fix the denominator, so the prior day's VA
width is used. Balance means overlap ≥ 0.70 (24% of days); imbalance means ≤ 0.30 (50%). Daily net P&L (risk
units, 24 bars) is the P&L earned on day D. DiD = slope of (MR − trend) on 1{balance}, HAC t.

| BTC daily net P&L | balance (549 d) | middle (584 d) | imbalance (1119 d) |
|---|---:|---:|---:|
| zscore_reversion | −0.122 | −0.028 | −0.248 |
| rsi_reversion | −0.041 | −0.029 | −0.072 |
| ema_stack | +0.052 | −0.010 | +0.097 |
| donchian_breakout | +0.217 | −0.019 | +0.082 |
| MR − trend | −0.216 | −0.014 | −0.249 |

| sample | n bal / imb | DiD | t |
|---|---:|---:|---:|
| BTC in-sample | 549 / 1119 | **+0.033** | **0.15** |
| ETH in-sample (replication) | 316 / 591 | +0.106 | 0.38 |
| BTC holdout | 20 / 45 | +0.99 | 0.94 |

In-sample, trend beats MR by the same margin whether the prior day was balanced or imbalanced. Both MR strategies
lose net of costs in every bucket. In the holdout, MR beat trend after balance days (+0.47) and lost after imbalance
days (−0.52). That is the hypothesised pattern, but it rests on 20 balance days with t 0.9.
**Verdict: in-sample contradicts; holdout supports (sign only).** Overall: not supported.

### H-AU4: the prior day's POC is a magnet after a balance day
Touch of POC(D−1) during day D (1h highs and lows) vs 2(1−Φ(d/√24)), with d measured from the 00:00 close in hourly σ.

| sample | days | touch rate | null | **excess** | **t** | context: all days | after imbalance |
|---|---:|---:|---:|---:|---:|---:|---:|
| BTC in-sample | 549 | 0.794 | 0.783 | **+1.1pp** | **0.77** | +0.2pp (t 0.28) | −1.4pp (t −1.18) |
| ETH in-sample (replication) | 316 | 0.794 | 0.769 | +2.5pp | 1.33 | −1.0pp | −3.4pp (t −2.13) |
| BTC holdout | 20 | 0.750 | 0.825 | −7.5pp | −0.87 | −0.6pp | +0.5pp |

The POC is touched about 79% of the time, which is exactly what its distance at the open predicts for a random walk.
**Verdict: contradicts in both segments.**

### H-RG1: the variance ratio detects trend vs mean-reversion regimes
Decision at each Monday 00:00 UTC close. VR(24) is Lo–MacKinlay with unbiased variances, computed from the trailing
168 1h log returns (overlapping 24h sums). The outcome is the next 168 bars' net P&L. DiD = slope of (trend − MR) on
1{VR > 1.2} over the obs with VR > 1.2 or VR < 0.8, with SE clustered by week.

| in-sample, per symbol-week net P&L | VR > 1.2 (1128) | middle (1972) | VR < 0.8 (2519) |
|---|---:|---:|---:|
| zscore_reversion | −0.59 | −0.62 | −0.96 |
| rsi_reversion | −0.24 | −0.22 | −0.24 |
| ema_stack | +0.28 | +0.17 | +0.18 |
| donchian_breakout | +0.36 | +0.22 | +0.21 |
| trend − MR | +0.73 | +0.61 | +0.80 |

| segment | obs (weeks) | n hi / lo | DiD | t |
|---|---:|---:|---:|---:|
| in-sample | 5619 (323) | 1128 / 2519 | **−0.06** | **−0.09** |
| holdout | 391 (17) | 90 / 166 | −4.37 | −1.10 |

**Correction (recorded before the holdout was run).** The first in-sample row came from a bug: the VR was divided by
q twice, so every VR was about 1/24 and the high bucket was empty. The corrected function has a unit test (mean VR of
1.00 on iid noise, > 1.5 on an AR(1) process with φ = 0.3). Note that 168 hourly returns estimate VR(24) with a
standard error of about 0.4, so the buckets are mostly noise. In the holdout, trend strategies lost in the VR > 1.2
and middle buckets and only donchian won when VR < 0.8, which is the opposite of the hypothesis.
**Verdict: contradicts in both segments.**

## Verdicts recorded (`research/hypotheses_results.jsonl`)

| id | in-sample | holdout | overall |
|---|---|---|---|
| H-AU1 | contradicts (correction row; the original row said "supports") | supports (n 87, t 1.2) | not supported |
| H-AU2 | contradicts | contradicts (n 14) | not supported |
| H-AU3 | contradicts | supports (20 / 45 days, t 0.9) | not supported |
| H-AU4 | contradicts | contradicts (n 20) | not supported |
| H-RG1 | contradicts (correction row after the VR bug) | contradicts | not supported |

Trial count: 5 hypotheses × 2 segments, plus 2 corrections and the labelled diagnostics. Nothing here is close to a
deflated threshold.

## Plain-language read: is there a rule for "trend-follow vs mean-revert right now"?

**No.** Neither detector picks the right style.
- Value-area overlap (H-AU3) and the 7-day variance ratio (H-RG1) are the two candidate switches. Across 6 years of
  BTC and 23 coins, the P&L gap between trend and mean reversion is the same in "balanced/reverting" states as in
  "imbalanced/trending" states. The DiD t-stats are 0.15 and −0.09.
- The one stable in-sample fact is not a regime fact. With the configured parameters and costs, the 1h
  mean-reversion strategies (zscore, rsi) lose in every bucket, and trend (ema_stack, donchian) is the better choice
  everywhere. The holdout (2026-07 to 2026-11) was bad for trend in almost every bucket too, and neither detector saw
  that coming. Treat that as a regime the tools cannot see, not as a sign that the switch works.
- The auction-theory levels behave like random-walk levels. Price reaches the opposite VA edge (H-AU2) and the prior
  POC (H-AU4) exactly as often as its distance and volatility predict. "Acceptance outside value" (H-AU1) shows at
  most a small continuation, about +0.04σ over 24h in-sample. That is less than costs, and it looks strong only when
  the outcome-dependent per-day averaging is used.
- Practical use: VA/POC levels are fine as *descriptive* reference prices (where the volume was). They carry no
  measured edge as targets, magnets or regime switches. The style decision should come from evidence that is already
  validated elsewhere (the trend-strategy priors in the playbook), not from profile overlap or VR.
