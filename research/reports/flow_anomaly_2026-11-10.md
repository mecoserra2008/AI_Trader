# Order-flow anomaly detectors: H-OF1 to H-OF5

_Agent A4 (program 2026-11-10). as_of **2026-11-10T22:30:00+00:00** (fixed for reproducibility; the collector flushes every 5 min). Segment **clean_holdout** only: every row is after the knowledge cutoff. This is a short first sample (about 3 days of flow_3s, about 2.9 days of liquidations, one market state that includes the 2026-11-10 CPI sell-off), and the forward record continues._

Code: `src/ai_trader/features/flow_anomaly.py` (detectors, `latest_anomalies`), `src/ai_trader/research/flow_anomaly_study.py` (tests). Tests: `tests/test_flow_anomaly.py`. Fitted model: `data/models/flow_anomaly/mahalanobis_v1.json`. Machine-readable: `research/reports/flow_anomaly_2026-11-10.json`.

## Summary

- **The registered detector isn't selective.** |robust z| > 4 on 1m signed taker imbalance fires on 21.6% of scored minutes (median symbol 19.5%, p10-p90 11.7%-32.7%). Signed flow is far heavier-tailed than its 24h MAD scale suggests: a symbol's own top 1% of |z| begins at |z| ~ 56 (median across symbols). That makes H-OF1, H-OF2 and H-OF5 weak tests of the underlying idea; the verdicts below apply to the detector as registered.
- **H-OF1 contradicts.** Forward |return| after an anomaly is 1.14x the matched minutes at 5m (CI [0.99, 1.40]) and 1.05x at 15m (CI [0.93, 1.21]). Both CIs exclude the registered 1.5, and no symbol with >= 10 events reaches 1.5 at 15m.
- **H-OF2 contradicts.** The sign rate is 0.513 at 5m (CI [0.495, 0.529]) and 0.505 at 15m. Mean signed returns are 1.6 / 2.5 bp (t 1.17 / 1.00).
- **H-OF3 is inconclusive, with the sign opposite to the hypothesis.** After liquidation clusters (declustered, n 142, 140 of 164 long liquidations), price **continued** in the liquidation direction: -1.85 sigma at 60m (t -1.92, 17 hour clusters), -82 bp (t -3.01). There was no reversal.
- **H-OF4 is inconclusive.** The robust Mahalanobis test AUC is 0.583, against 0.603 for the best single feature (imbalance): difference -0.021, CI [-0.065, 0.073]. All detectors are weak (AUC 0.50-0.60) on the test half.
- **H-OF5 contradicts.** BTC-to-alt lift is 1.32 (CI [1.16, 1.49]) and alt-to-BTC is 1.28 (CI [1.14, 1.44]). The co-occurrence is strongest in the same minute (lift 1.71) and symmetric, so it looks like common shocks rather than a BTC lead.
- Holm across the five: no hypothesis survives (smallest adjusted p 0.62). The original-spec sensitivity (min_valid 720, no scale floor) gives: H-OF1 contradicts, H-OF2 contradicts, H-OF3 contradicts, H-OF4 inconclusive, H-OF5 contradicts.

## Data and point in time

- flow_3s, Bybit **linear perps** only (spot streams excluded; liquidations are perp events): 104 symbols, of which 37 are TradFi perps. 52 symbols start 2026-11-07 17:46 UTC and 52 start 2026-11-08 21:09 UTC. Liquidations start 2026-11-08 00:46 UTC.
- Every read is `Lake.read_asof(..., as_of=2026-11-10T22:30:00+00:00)`. 3 s candles finalised more than 60 s after their end (reconnect artefacts, known_ts up to ~10 h late) are dropped. Only complete minutes (minute end <= as_of) are kept.
- **Collector outages:** 168070 of 389657 symbol-minutes (43%) have no flow candle. These are collector-wide; the longest runs from about 2026-11-09 23:00 to 2026-11-10 12:00 UTC. Gap minutes are null (never zero) for flow and liquidations, and forward returns across gaps are null.
- **Warm-up:** the first 24 h of each symbol has no trailing window and is dropped. So are minutes whose trailing 24 h holds fewer than 360 valid minutes. Scored: 95887 symbol-minutes, first 2026-11-08T17:53:00+00:00, last 2026-11-10T22:29:00+00:00.

## Methods

- **z**: robust z vs trailing 1440 1m bars (t-1440..t-1), >= 360 valid (collector gaps); scale 1.4826*MAD, fallback 1.2533*meanAD when MAD = 0, floored at 1% of |median|; 3 s candles finalised > 60 s late dropped
- **features**: imbalance = buy_notional - sell_notional; trades = n_buy + n_sell; volume = buy+sell notional; spread = TWA spread_bps; depth = depth_bid_25 + depth_ask_25; abs_ret = |1m log mid return|; liq = 1m liquidation notional
- **anomaly**: |z_imbalance| > 4 (H-OF1, H-OF2, H-OF5)
- **forward**: log(mid_{t+h} / mid_t), mid at the close of minute t (entry after the anomaly minute is known)
- **sigma**: sigma_h = sqrt(h * trailing-1440 mean of squared 1m mid returns)
- **warmup**: first 1440 minutes of each symbol dropped (no trailing window); liquidation features need 1440 minutes after 2026-11-08 00:46 (collection start); minutes in collector outages are null (never zero) for flow and liquidations
- **blocks**: calendar-hour blocks pooled across symbols (cross-sectional dependence kept inside a block), B = 2000
- **universe**: Bybit linear perps in flow_3s (spot streams excluded); asset class TradFi = the TradFi stream groups in settings.flow.streams
- **verdict**: supports = registered criterion met and Holm-adjusted p < 0.05; contradicts = criterion fails and the 95% CI excludes the required effect; else inconclusive; < 30 events -> inconclusive
- **H-OF4 split**: the median timestamp of the pooled complete-feature rows; the model is fitted on rows before it and scored on rows after it. The fit uses MCD-style C-steps (h = 75%) from a coordinate-median start, a Ledoit-Wolf-shrunk covariance of the retained subset, and chi2(7) consistency rescaling. z values are clipped to +-20. numpy only.
- **p values for Holm** (one-sided): H-OF1, the max over horizons of the bootstrap P(ratio <= 1); H-OF2, the max over horizons of p from the hour-clustered t; H-OF3, p from the 60m t; H-OF4, the bootstrap P(diff <= 0); H-OF5, the max of P(lift <= 1) and P(diff <= 0).

## Sample

| item | value |
|---|---|
| symbols | 104 |
| symbols_scored | 104 |
| tradfi_symbols | 37 |
| minutes_total | 389657 |
| minutes_scored | 95887 |
| gap_minutes | 168070 |
| first_minute | 2026-11-07T17:46:00+00:00 |
| first_scored | 2026-11-08T17:53:00+00:00 |
| last_minute | 2026-11-10T22:29:00+00:00 |
| imbalance_anomalies | 20722 |
| imbalance_anomaly_share | 0.2161 |
| inf_z_imbalance | 0 |
| liq_clusters | 418 |

Share of scored minutes with |robust z| > 4, by feature: imbalance 0.216, trades 0.145, volume 0.205, spread 0.126, depth 0.047, abs_ret 0.121

## Verdicts (clean_holdout; Holm over the 5 OF hypotheses)

| id | primary statistic | p raw | p Holm | verdict |
|---|---|---|---|---|
| H-OF1 | 5m ratio 1.14 CI [0.99, 1.40] (n 20191); 15m ratio 1.05 CI [0.93, 1.21] (n 19466) | 0.2274 | 0.6822 | **contradicts** |
| H-OF2 | 5m sign 0.513, mean 1.63 bp (t 1.17); 15m sign 0.505, mean 2.48 bp (t 1.00) | 0.1589 | 0.6358 | **contradicts** |
| H-OF3 | 60m signed -1.853 sigma (t -1.92, n 142); 15m -1.782 (t -1.55) | 0.9727 | 1.0000 | **inconclusive** |
| H-OF4 | AUC maha 0.583 vs best single (imbalance) 0.603; diff -0.021 CI [-0.065, 0.073] | 0.7363 | 1.0000 | **inconclusive** |
| H-OF5 | BTC->alt lift 1.32 CI [1.16, 1.49]; alt->BTC 1.28 CI [1.14, 1.44]; diff CI [-0.03, 0.10] | 0.1234 | 0.6172 | **contradicts** |

Sensitivity, original spec (min_valid 720, rel_floor 0.0): H-OF1 contradicts (p Holm 0.564), H-OF2 contradicts (p Holm 0.681), H-OF3 contradicts (p Holm 1.000), H-OF4 inconclusive (p Holm 1.000), H-OF5 contradicts (p Holm 0.112).

## H-OF1: forward |return| after |z_imbalance| > 4 vs matched minutes

| horizon | class | ratio | 95% CI | n | median event ratio |
|---|---|---|---|---|---|
| 5m | pooled | 1.14 | [0.99, 1.40] | 20191 | 0.86 |
| 5m | crypto | 1.17 | [0.97, 1.49] | 12464 | |
| 5m | tradfi | 1.10 | [1.01, 1.23] | 7727 | |
| 15m | pooled | 1.05 | [0.93, 1.21] | 19466 | 0.86 |
| 15m | crypto | 1.06 | [0.90, 1.26] | 11960 | |
| 15m | tradfi | 1.04 | [0.96, 1.15] | 7506 | |

Per symbol, 15m (top 15 by events; all in the JSON):

| symbol | n | ratio | 95% CI |
|---|---|---|---|
| QQQUSDT | 649 | 1.21 | [0.94, 1.60] |
| NVDAUSDT | 579 | 1.12 | [0.98, 1.27] |
| AAPLUSDT | 560 | 1.23 | [0.90, 1.98] |
| TSLAUSDT | 535 | 1.24 | [0.88, 1.65] |
| XAGUSDT | 526 | 1.12 | [0.86, 1.60] |
| GOOGLUSDT | 478 | 1.22 | [0.90, 1.65] |
| XAUTUSDT | 463 | 0.93 | [0.83, 1.07] |
| SPCXUSDT | 425 | 1.02 | [0.89, 1.13] |
| TRXUSDT | 420 | 1.11 | [0.95, 1.28] |
| XAUUSDT | 417 | 1.06 | [0.91, 1.26] |
| INTCUSDT | 397 | 0.99 | [0.87, 1.12] |
| CLUSDT | 382 | 1.17 | [0.86, 1.52] |
| MSTRUSDT | 354 | 1.15 | [0.95, 1.39] |
| ETHUSDT | 352 | 1.21 | [0.82, 1.88] |
| HYPEUSDT | 342 | 1.05 | [0.82, 1.33] |

Symbols with >= 10 events: 103; 15m ratio > 1 in 56, >= 1.5 in 0; median 1.01.

## H-OF2: direction after |z_imbalance| > 4

| horizon | sign rate | 95% CI | zero-return share | mean bp | t (hour-clustered) | mean sigma | t |
|---|---|---|---|---|---|---|---|
| 5m | 0.513 | [0.495, 0.529] | 0.032 | 1.63 | 1.17 | 0.157 | 1.09 |
| 15m | 0.505 | [0.484, 0.526] | 0.016 | 2.48 | 1.00 | 0.183 | 1.30 |

| horizon | class | n | sign rate | mean bp | t |
|---|---|---|---|---|---|
| 5m | crypto | 12464 | 0.518 | 2.74 | 1.38 |
| 5m | tradfi | 7853 | 0.504 | -0.14 | -0.24 |
| 15m | crypto | 11960 | 0.505 | 4.10 | 1.14 |
| 15m | tradfi | 7632 | 0.505 | -0.05 | -0.05 |

Per symbol, 15m (top 15 by events):

| symbol | n | sign rate | mean bp | t |
|---|---|---|---|---|
| QQQUSDT | 649 | 0.512 | 1.82 | 1.16 |
| XAGUSDT | 646 | 0.511 | 0.41 | 0.28 |
| NVDAUSDT | 579 | 0.502 | -4.04 | -1.09 |
| AAPLUSDT | 560 | 0.523 | 2.28 | 1.20 |
| TSLAUSDT | 535 | 0.493 | -1.59 | -0.73 |
| GOOGLUSDT | 478 | 0.553 | 5.58 | 3.05 |
| XAUTUSDT | 463 | 0.513 | -0.13 | -0.22 |
| SPCXUSDT | 425 | 0.470 | -2.18 | -0.91 |
| TRXUSDT | 420 | 0.447 | -0.69 | -1.34 |
| XAUUSDT | 417 | 0.536 | 1.31 | 0.82 |
| INTCUSDT | 397 | 0.482 | -1.72 | -0.58 |
| CLUSDT | 382 | 0.538 | 3.06 | 1.32 |
| MSTRUSDT | 354 | 0.511 | 3.42 | 0.70 |
| ETHUSDT | 352 | 0.523 | 14.89 | 1.36 |
| HYPEUSDT | 342 | 0.472 | 0.36 | 0.09 |

## H-OF3: liquidation clusters (declustered 30 min), return signed against the liquidated side

Raw cluster minutes 418, declustered events 164, by side {'short': 24, 'long': 140}.

| horizon | mean sigma | t | n | hour clusters | mean bp | t bp | hit rate |
|---|---|---|---|---|---|---|---|
| 15m | -1.782 | -1.55 | 153 | 19 | -39.04 | -1.74 | 0.405 |
| 60m | -1.853 | -1.92 | 142 | 17 | -82.02 | -3.01 | 0.303 |

| horizon | liquidated side | mean sigma | t | n |
|---|---|---|---|---|
| 15m | long | -2.117 | -1.75 | 131 |
| 15m | short | 0.210 | 0.36 | 22 |
| 60m | long | -2.176 | -2.22 | 121 |
| 60m | short | 0.005 | 0.01 | 21 |

Per symbol, 60m (top 15 by events):

| symbol | n | mean sigma | mean bp |
|---|---|---|---|
| ETHUSDT | 10 | -4.492 | -100.94 |
| BTCUSDT | 9 | -2.390 | -29.46 |
| SOXLUSDT | 8 | -1.219 | -231.57 |
| XRPUSDT | 7 | -2.103 | -83.81 |
| ZECUSDT | 7 | -1.547 | -221.08 |
| NVDAUSDT | 6 | 0.196 | 13.07 |
| HYPEUSDT | 5 | -2.974 | -126.54 |
| SOLUSDT | 5 | -2.625 | -74.16 |
| 1000PEPEUSDT | 4 | -0.059 | 9.01 |
| ARBUSDT | 4 | -0.434 | -54.66 |
| NEARUSDT | 4 | -0.857 | -80.70 |
| XAUUSDT | 4 | -1.354 | -7.54 |
| ADAUSDT | 3 | -2.373 | -139.53 |
| AVAXUSDT | 3 | -2.105 | -82.26 |
| CLUSDT | 3 | 4.598 | 106.01 |

## H-OF4: multivariate robust Mahalanobis vs single features (AUC for |r_15m| > 2 sigma_15m)

Fit on minutes before 2026-11-09T14:10:00+00:00 (n 38038, base rate 0.149); test n 38105 (base rate 0.091, 104 symbols, 12 hour blocks). Ledoit-Wolf shrinkage 0.001.

| score | test AUC | 95% CI | train AUC |
|---|---|---|---|
| imbalance | 0.603 | [0.537, 0.653] | 0.658 |
| mahalanobis | 0.583 | [0.536, 0.661] | 0.725 |
| trades | 0.580 | [0.507, 0.627] | 0.729 |
| volume | 0.576 | [0.521, 0.617] | 0.709 |
| abs_ret | 0.559 | [0.499, 0.604] | 0.636 |
| spread | 0.553 | [0.501, 0.677] | 0.636 |
| depth | 0.525 | [0.471, 0.611] | 0.648 |
| liq | 0.504 | [0.499, 0.512] | 0.534 |

Difference vs the best single feature on the test half (imbalance): -0.021 CI [-0.065, 0.073]; vs the best per replicate: CI [-0.065, 0.012]; vs the feature best on the training half (trades): 0.003 CI [-0.027, 0.097].

## H-OF5: BTC-led vs alt-led flow anomalies (next 3 minutes)

BTC anomalies 320, alt anomalies 11945 over 65 alts, 78 hour blocks.

| direction | P(cond) | P(uncond) | lift | 95% CI | pairs |
|---|---|---|---|---|---|
| btc_to_alt | 0.514 | 0.388 | 1.32 | [1.16, 1.49] | 12625 |
| alt_to_btc | 0.489 | 0.382 | 1.28 | [1.14, 1.44] | 11774 |

Lift difference (BTC->alt minus alt->BTC): 0.045 CI [-0.034, 0.103].

By lag (BTC anomaly at t, alt anomaly at exactly t+k): k=0: lift 1.71, k=1: lift 1.49, k=2: lift 1.35, k=3: lift 1.40.

Per-alt lifts (alts with >= 20 BTC events): median 1.34, >= 2 in 1 of 65.

## EXPLORATORY: per-symbol quantile thresholds (not a registered test, not recorded as a verdict)

Anomaly = |z_imbalance| above the symbol's own 99% / 99.9% quantile **over the whole scored sample**. This uses future data, so it isn't causal. It is here only to show whether a more selective detector looks different; any rule from it needs a new registered hypothesis and a forward test.

| threshold | median abs z cut | OF1 5m ratio [CI] (n) | OF1 15m ratio [CI] | OF2 5m sign [CI] | OF2 15m mean bp (t) | OF5 BTC->alt lift [CI] (BTC events) |
|---|---|---|---|---|---|---|
| top 1.0% | 56 | 1.51 [0.96, 2.33] (915) | 1.22 [0.95, 1.58] | 0.532 [0.480, 0.570] | 6.2 (1.16) | 7.03 [0.00, 12.61] (14) |
| top 0.1% | 190 | 2.28 [0.65, 3.33] (50) | 1.54 [0.51, 2.12] | 0.460 [0.362, 0.739] | -6.0 (-0.58) | 0.00 [0.00, 0.00] (1) |

Reading: at the top 1% the 5m volatility ratio rises to about 1.5, but its CI includes 1, and direction stays at a coin flip. The top 0.1% has 50 events and one BTC anomaly, which is too few to read. A forward, pre-registered test of a selective (quantile or |z| >> 4) detector is the natural follow-up.

## Firing rate of |z_imbalance| > 4 by symbol

Pooled 0.216; median symbol 0.195; p10 0.117; p90 0.327. All 104 values are in the JSON (`sample.firing_rate_per_symbol`).

| lowest 10 | rate | highest 10 | rate |
|---|---|---|---|
| USDJPYUSDT | 0.023 | XAGUSDT | 0.463 |
| EURUSDUSDT | 0.037 | QQQUSDT | 0.458 |
| IWMUSDT | 0.044 | NVDAUSDT | 0.408 |
| SMHUSDT | 0.044 | AAPLUSDT | 0.393 |
| XLEUSDT | 0.049 | TSLAUSDT | 0.377 |
| TLTUSDT | 0.058 | ICPUSDT | 0.374 |
| GDXUSDT | 0.060 | AMDSTOCKUSDT | 0.339 |
| CORZUSDT | 0.070 | GOOGLUSDT | 0.338 |
| ASTERUSDT | 0.088 | XAUTUSDT | 0.337 |
| WLFIUSDT | 0.092 | MAVIAUSDT | 0.337 |

## API for the market-state report

```python
from ai_trader.features.flow_anomaly import latest_anomalies
latest_anomalies(symbols: Sequence[str] | None, as_of: datetime, lookback_h: int = 24, *, z_thresh: float = 4.0,
                 lake=None, category='linear', model=None, use_model=True, maha_quantile='0.999') -> dict[str, dict]
# -> {symbol: {'counts': {type: n}, 'n_total': n, 'minutes': n_scored,
#              'latest': {'type', 'z', 'time', 'side'} | None, 'latest_by_type': {type: {...}}}}
# types: imbalance (side buy/sell), trades / volume (high/low), spread (widen/tighten), depth (thick/thin),
#        abs_ret (side = up/down), liq_cluster (side = liquidated side long/short), multivariate (Mahalanobis > training q99.9)
```

It reads point in time (lookback + 24 h warm-up), in symbol batches of 12 rolled up to 1 minute at once, so memory is bounded. Rolling statistics are computed only for the lookback minutes. Live callers should pass as_of <= now - 5 min (flush interval).
Timing on this machine (shared with other jobs): 8 symbols 7 s, all 104 linear symbols 85 s (lookback 24 h, as_of 2026-11-10 22:30 UTC; was ~45 s for 8 symbols and a MemoryError at 57 before batching and lookback-only scoring).
Because the registered |z| > 4 rule fires about 20% of the time, the counts are an activity gauge, not an alarm. The JSON has the per-symbol firing rates to put them in context.

## Caveats

- **3 days, one market state.** The sample covers a weekend, a Monday, and the 2026-11-10 CPI sell-off (long liquidations dominate: 140 long vs 24 short clusters). H-OF3's continuation is essentially one or two cascade episodes (17 hour clusters). Everything here is confidence C at most.
- **Collector outages** remove 43% of symbol-minutes, including most of 2026-11-10 00:00-12:00 UTC. With the 360-minute rule, the trailing windows right after long gaps are thin. The H-OF4 test half has only 12 hour blocks, so its bootstrap CIs are optimistic.
- **A non-selective detector makes H-OF1, H-OF2 and H-OF5 weaker tests.** About 1 in 5 minutes counts as an 'anomaly', so these are tests of the registered threshold, not of extreme flow. The exploratory quantile table hints that a much higher threshold behaves differently (5m vol ratio about 1.5) but is not significant and not causal.
- **Spec changes after a 6-symbol debug run** (diary 2026-11-10, kind mistake): min_valid 720 -> 360, a 1%-of-median scale floor, and dropping late-finalised candles. Under the original spec, every statistic keeps its sign. H-OF3 turns **contradicts** (60m -2.17 sigma, t -2.06), and the H-OF5 BTC-minus-alt lift difference gets a CI just above 0 ([0.001, 0.135]) while the lift stays well below 2 (1.46). The recorded verdicts use the primary spec; the sensitivity is in the JSON and in the metrics of each recorded row.
- The matched baseline for H-OF1 is held fixed in the bootstrap (it averages many more minutes than the events). Pooled bp means are dominated by volatile small caps; sigma units are reported alongside.
- TradFi perps trade 24/7 on Bybit while their underlying is closed most of this sample; 37 of 104 symbols are TradFi.

## Rule candidates

```yaml
- id: R-FLOW-1
  rule: "WHEN 1m signed taker imbalance has |robust z| > 4 (trailing 24h median/MAD) THEN do not read it as a volatility or direction signal; it fires on about 1 minute in 5"
  when: "flow_anomaly z_imbalance |z| > 4 (latest_anomalies type imbalance)"
  then: "expect next 5-15m |return| about the same as a normal minute of that hour and a coin-flip direction; use it as an activity gauge only"
  scope: {assets: all, horizon: "15m", sides: both}
  effect: "fwd |ret| ratio 5m 1.14 (CI 0.99-1.40), 15m 1.05 (CI 0.93-1.21); sign rate 5m 0.513, 15m 0.505; firing rate 21.6%"
  confidence: C
  evidence: [H-OF1, H-OF2, research/reports/flow_anomaly_2026-11-10.md]
  review: "a forward month of flow data, or a pre-registered selective detector (per-symbol top-1% |z|) showing a ratio >= 1.5 with CI excluding 1"
- id: R-FLOW-2
  rule: "WHEN a liquidation cluster prints (1m liquidation notional > trailing-24h 99th percentile and >= 3 prints) THEN do not fade it expecting a reversal within 60 minutes"
  when: "latest_anomalies type liq_cluster (side = liquidated side)"
  then: "prefer no counter-trend entry on the cluster; in the first sample the move continued in the liquidation direction"
  scope: {assets: all, horizon: "1h", sides: both}
  effect: "signed (reversal-positive) 60m return -1.85 sigma (t -1.92, n 142, 17 hour clusters), -82 bp; mostly long liquidations in the 2026-11-10 sell-off"
  confidence: C
  evidence: [H-OF3, research/reports/flow_anomaly_2026-11-10.md]
  review: "holdout reaches >= 300 declustered clusters across >= 10 separate sell-offs/squeezes, or short-liquidation clusters show the opposite sign"
- id: R-FLOW-3
  rule: "WHEN BTC shows a flow anomaly THEN do not expect alt flow anomalies to follow with a 1-3 minute lead; flow shocks hit BTC and alts together"
  when: "BTCUSDT latest_anomalies imbalance anomaly"
  then: "watch the cross-section at the same minute rather than BTC first; no BTC-leads-alts timing edge"
  scope: {assets: crypto, horizon: "3m", sides: both}
  effect: "lift BTC->alt 1.32 (CI 1.16-1.49) vs alt->BTC 1.28 (CI 1.14-1.44); same-minute lift 1.71"
  confidence: C
  evidence: [H-OF5, research/reports/flow_anomaly_2026-11-10.md]
  review: "a selective detector (top 1% |z|) with >= 100 BTC events shows BTC->alt lift >= 2 and > alt->BTC"
- id: R-FLOW-4
  rule: "WHEN the multivariate flow score (robust Mahalanobis over 7 flow z) is high THEN treat it as no better than the imbalance z for anticipating a 2-sigma 15m move"
  when: "latest_anomalies type multivariate (distance > training q99.9)"
  then: "descriptive flag only; do not raise vol expectations on it beyond what the vol model already says"
  scope: {assets: all, horizon: "15m", sides: both}
  effect: "test AUC 0.583 vs imbalance 0.603 (diff -0.021, CI -0.065 to +0.073)"
  confidence: C
  evidence: [H-OF4, research/reports/flow_anomaly_2026-11-10.md]
  review: "refit on >= 2 weeks of gap-free flow; AUC difference CI excluding 0 in a forward window"
```
