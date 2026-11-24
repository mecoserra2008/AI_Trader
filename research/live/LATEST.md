# BTC live network: bar 2026-11-24 21:00 UTC (line nn-20261124T21)
as_of 2026-11-24T21:08:28Z, newest bar closed 21:00, close 59,752.1. Flags: rolling_day. Research only, no direction call.

## Next hour [11-24T21:00 -> 11-24T22:00], % log return
- PNN median +0.01, 80% [-0.64, +0.64], 90% [-0.89, +0.88], sigma 0.50%
- ACI 80% [-0.66, +0.66] alpha_t 0.1940; 90% [-1.07, +1.06] alpha_t 0.0745; 95% [-1.56, +1.55] alpha_t 0.0373
- H-NN10 vol (bar 07:00): ML 0.645% vs baseline 0.650% (ratio 0.99) - not this bar's line

## Next 24h [11-24T21:00 -> 11-25T21:00], % log return
- PNN median +0.61, 80% [-3.58, +5.85], 90% [-6.01, +7.65], sigma 3.68%
- ACI 80% [-3.47, +5.74] alpha_t 0.2050; 90% [-6.65, +8.29] alpha_t 0.0916; 95% [-9.76, +11.40] alpha_t 0.0405
- 24h net: rolling day, shift 21 h

## v2.1 shadow (ACI gamma 0.02, issue alpha >= nominal/4, jump-cluster floor; cfg 2d4ba801a6363fe9; v21-20261124T21)
- 1h: 80% [-0.58, +0.58]; 90% [-0.97, +0.96]; 95% [-1.45, +1.43]; alpha_t (unclipped) 80/90/95 0.2420/0.0935/0.0418; floor inactive
- 24h: 80% [-3.59, +5.86]; 90% [-6.87, +8.51]; 95% [-10.08, +11.72]; alpha_t 0.1975/0.0854/0.0369
- running coverage (issued, 72 / 168): 1h 80% 0.875/0.816, 90% 0.931/0.905, 95% 0.972/0.952 | 24h 80% 0.819/0.726, 90% 0.931/0.839, 95% 0.986/0.917
- v1 vs v2 since 11-13T20:00 (same bars; v2 mostly replay): 1h n 127: cov80 0.756 vs 0.780, cov95 0.929 vs 0.945, band pinball 0.0965 vs 0.0965; 24h n 117: cov80 0.675 vs 0.692, cov95 0.889 vs 0.897, band pinball 0.5063 vs 0.5104

## Regime and players (information only, changes no forecast; as_of 2026-11-24T21:08Z)
| | v0 cell | 24h px | funding (90d pct) | OI 24h / 72h | liq 4h long/short | liq 24h long/short | DVOL (90d pct) |
|---|---|---|---|---|---|---|---|
| BTC | R9 liquidation | -2.4% | +0.0034% (p37) | +2.0% / -0.8% | 0k / 0k | 6.8M / 0.9M | 53.7 (p97) |
| ETH | R9 liquidation | -2.9% | +0.0022% (p36) | +1.2% / -0.4% | 0k / 0k | 2.8M / 0.3M | 67.0 (p97) |
| SOL | R9 liquidation | -5.0% | -0.0028% (p26) | +2.8% / -1.0% | 0k / 0k | 0.7M / 44k | no index |
- Pattern: BTC long flush, funding normal, OI rebuilding after a drop while price fell (new shorts), implied vol high; ETH long flush, funding normal, OI rebuilding after a drop while price fell (new shorts), implied vol high; SOL long flush, funding negative, OI rebuilding after a drop while price fell (new shorts)
- Long/short ratio omitted: lake ls_ratio last known 2026-11-07T20:00 (one-off backfill of 11-07; the live updater has no ls_ratio stream)

## Cross-section v3 (context only, not a signal: R-XS-2/R-XS-3; xs3-20261124T21, 23/23 tokens)
- 4h relative median vs EW basket, bp [80% ACI]: top BTC +1 [-82, +86], ETH +1 [-61, +64], XLM +1 [-76, +80], TRX -0 [-194, +192], HYPE -0 [-112, +116]
- bottom NEAR -5 [-124, +128], ATOM -6 [-129, +128], FIL -7 [-118, +117], LTC -7 [-82, +80], ETC -9 [-81, +81]
- mostly low vol: rank corr with -lsig2 4h +0.53 / 24h +0.57; top 5 shared with the low-vol top 5: 3/5 (TRX, BNB, ETH, SOL, BTC)
- BTC 1h |r|: v3 q50 0.29%, q90 0.79% (log-HAR-t q90 0.93%) vs v1 90% half-width 0.88% (ACI90 1.06%)
- running (scored lines): 4h IC v3 +0.110 vs low-vol +0.056 (n 87); 24h IC v3 +0.216 vs low-vol +0.164 (n 77); 1h vol pinball v3/HAR 0.954 (n 87)
- 24h ACI band under-covers (holdout 0.749 at 80%); the decision sessions do not read this block

## Field card (btc_field_2026-11-13.json, regime R9)
- 1h: 90% [-0.82, +0.84] med +0.01; touch +1/-1 5/5%, +2/-2 1/1%, +3/-3 0/0%; P(jump) 2%
- 4h: 90% [-1.71, +1.79] med +0.04; touch +1/-1 31/28%, +2/-2 7/6%, +3/-3 2/2%; P(jump) 6%
- 24h: 90% [-5.45, +5.50] med +0.31; touch +1/-1 75/69%, +2/-2 52/45%, +3/-3 33/29%; P(jump) 32%
- variance drivers: s -0.18, x2 -0.14, m24*dir +0.09; jump drivers: m24*dir +0.05

## Latent (1h net, 4 dims)
- z [0.3, 0.49, -0.43, -0.45], predicted next [0.37, 0.61, -0.41, -0.39]
- last hour: model error 0.22 vs persistence 0.47 (latent sd); 168h R^2 gain +0.197

## News
- alt: wp_btc -8.02, wp_ind -6.20, wp_max -4.83
- RSS items last 1h 44, 24h 129

## Running coverage (ACI, last 72 / 168 resolved)
- 1h: 80% 0.875 / 0.821; 90% 0.917 / 0.893; 95% 0.958 / 0.952 (n 168)
- 24h: 80% 0.792 / 0.714; 90% 0.917 / 0.833; 95% 0.972 / 0.911 (n 168)

## Scored this run
- nn-20261124T08 1h: r -0.290% (z -0.51), in ACI80 True, pinball 0.0827

## Alarms
- coverage 24h ACI 80% over last 168: 0.714 vs 0.80
- coverage 24h ACI 90% over last 168: 0.833 vs 0.90

## Changed vs previous line (nn-20261124T08)
- 1h sigma 0.57 -> 0.50%; 24h sigma 3.80 -> 3.68%; 1h alpha80 0.1960 -> 0.1940; field 1h P(jump) 1.6 -> 1.6%
