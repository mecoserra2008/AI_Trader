# BTC live network: bar 2026-11-15 16:00 UTC (line nn-20261115T16)
as_of 2026-11-15T16:03:05Z, newest bar closed 16:00, close 64,720.7. Flags: rolling_day. Research only, no direction call.

## Next hour [11-15T16:00 -> 11-15T17:00], % log return
- PNN median +0.01, 80% [-0.76, +0.78], 90% [-1.06, +1.06], sigma 0.60%
- ACI 80% [-0.92, +0.93] alpha_t 0.1530; 90% [-1.43, +1.43] alpha_t 0.0640; 95% [-2.08, +2.08] alpha_t 0.0270
- H-NN10 vol (bar 15:00): ML 0.831% vs baseline 1.041% (ratio 0.80) - not this bar's line

## Next 24h [11-15T16:00 -> 11-16T16:00], % log return
- PNN median +0.84, 80% [-2.76, +5.24], 90% [-4.75, +6.72], sigma 3.12%
- ACI 80% [-2.71, +5.19] alpha_t 0.2045; 90% [-5.23, +7.20] alpha_t 0.0909; 95% [-7.60, +9.57] alpha_t 0.0399
- 24h net: rolling day, shift 16 h

## v2.1 shadow (ACI gamma 0.02, issue alpha >= nominal/4, jump-cluster floor; cfg 2d4ba801a6363fe9; v21-20261115T16)
- 1h: 80% [-1.15, +1.17]; 90% [-1.98, +1.99]; 95% [-2.76, +2.76]; alpha_t (unclipped) 80/90/95 0.0980/0.0315/0.0008; floor inactive
- 24h: 80% [-2.84, +5.32]; 90% [-5.49, +7.46]; 95% [-8.12, +10.09]; alpha_t 0.1940/0.0816/0.0337
- running coverage (issued, 72 / 168): 1h 80% 0.778/0.768, 90% 0.861/0.875, 95% 0.917/0.941 | 24h 80% 0.569/0.720, 90% 0.694/0.804, 95% 0.819/0.887
- v1 vs v2 since 11-13T20:00 (same bars; v2 mostly replay): 1h n 44: cov80 0.636 vs 0.727, cov95 0.864 vs 0.909, band pinball 0.1245 vs 0.1170; 24h n 21: cov80 0.000 vs 0.000, cov95 0.476 vs 0.476, band pinball 1.1615 vs 1.1615

## Regime and players (information only, changes no forecast; as_of 2026-11-15T16:03Z)
| | v0 cell | 24h px | funding (90d pct) | OI 24h / 72h | liq 4h long/short | liq 24h long/short | DVOL (90d pct) |
|---|---|---|---|---|---|---|---|
| BTC | R9 liquidation | -1.5% | -0.0018% (p3) | +1.6% / +2.4% | 1.6M / 0.2M | 61.2M / 14.9M | 44.3 (p99) |
| ETH | R9 liquidation | -1.7% | -0.0012% (p11) | -2.6% / +1.9% | 0.9M / 0.2M | 34.0M / 3.8M | 57.8 (p95) |
| SOL | R9 liquidation | -3.7% | -0.0027% (p21) | -2.2% / -5.8% | 0.2M / 0.1M | 7.7M / 1.1M | no index |
- Pattern: BTC long flush, funding negative, OI rising into the drop (new shorts), long-liquidation burst (4h), implied vol high; ETH long flush, funding negative, OI falling with price (deleveraging), long-liquidation burst (4h), implied vol high; SOL long flush, funding negative, OI falling with price (deleveraging)
- Long/short ratio omitted: lake ls_ratio last known 2026-11-07T20:00 (one-off backfill of 11-07; the live updater has no ls_ratio stream)

## Cross-section v3 (context only, not a signal: R-XS-2/R-XS-3; xs3-20261115T16, 23/23 tokens)
- 4h relative median vs EW basket, bp [80% ACI]: top BTC +4 [-127, +137], TRX +2 [-304, +287], ETC -0 [-76, +81], ETH -1 [-113, +121], DOGE -1 [-79, +87]
- bottom ATOM -10 [-188, +184], UNI -10 [-189, +182], ENA -11 [-148, +137], NEAR -12 [-162, +150], AAVE -15 [-167, +164]
- mostly low vol: rank corr with -lsig2 4h +0.58 / 24h +0.65; top 5 shared with the low-vol top 5: 2/5 (TRX, BTC, BNB, LTC, ATOM)
- BTC 1h |r|: v3 q50 0.48%, q90 1.34% (log-HAR-t q90 1.35%) vs v1 90% half-width 1.06% (ACI90 1.43%)
- running (scored lines): 4h IC n/a; 24h IC n/a; 1h vol pinball v3/HAR 0.995 (n 2)
- 24h ACI band under-covers (holdout 0.749 at 80%); the decision sessions do not read this block

## Field card (btc_field_2026-11-13.json, regime R9)
- 1h: 90% [-1.47, +1.51] med +0.01; touch +1/-1 27/26%, +2/-2 4/3%, +3/-3 1/1%; P(jump) 2%
- 4h: 90% [-3.10, +3.26] med +0.09; touch +1/-1 59/55%, +2/-2 27/24%, +3/-3 11/10%; P(jump) 6%
- 24h: 90% [-10.28, +10.03] med +0.90; touch +1/-1 87/83%, +2/-2 74/66%, +3/-3 62/54%; P(jump) 31%
- variance drivers: x1 -0.37, s +0.30, x2 -0.13; jump drivers: m24*dir +0.03

## Latent (1h net, 4 dims)
- z [0.88, 1.11, -0.88, -0.1], predicted next [0.71, 0.93, -0.67, -0.25]
- last hour: model error 0.90 vs persistence 0.89 (latent sd); 168h R^2 gain +0.126

## News
- alt: wp_btc -8.02, wp_ind -6.27, wp_max -4.92, wp_edits +0.00, hn_story +0.00, hn_comment -0.58, fng -0.80, fng_chg -0.06
- RSS items last 1h 0, 24h 0 (polling stale)

## Running coverage (ACI, last 72 / 168 resolved)
- 1h: 80% 0.694 / 0.732; 90% 0.806 / 0.851; 95% 0.875 / 0.923 (n 168)
- 24h: 80% 0.569 / 0.720; 90% 0.694 / 0.804; 95% 0.819 / 0.887 (n 168)

## Scored this run
- nn-20261114T16 24h: r -1.538% (z -1.03), in ACI80 False, pinball 0.3787
- nn-20261115T15 1h: r -0.027% (z -0.04), in ACI80 True, pinball 0.0751

## Alarms (8)
- coverage (band level/window cov): 1h 90%/72 0.806; 1h 95%/72 0.875; 24h 80%/72 0.569; 24h 80%/168 0.720; 24h 90%/72 0.694; 24h 90%/168 0.804; 24h 95%/72 0.819; 24h 95%/168 0.887

## Changed vs previous line (nn-20261115T15)
- 1h sigma 0.65 -> 0.60%; 24h sigma 3.41 -> 3.12%; 1h alpha80 0.1520 -> 0.1530; field 1h P(jump) 1.6 -> 1.6%
- new alarms: 2
