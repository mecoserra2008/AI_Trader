# BTC live network: bar 2026-11-15 17:00 UTC (line nn-20261115T17)
as_of 2026-11-15T17:03:03Z, newest bar closed 17:00, close 64,673.9. Flags: rolling_day. Research only, no direction call.

## Next hour [11-15T17:00 -> 11-15T18:00], % log return
- PNN median +0.02, 80% [-0.69, +0.72], 90% [-0.97, +0.99], sigma 0.55%
- ACI 80% [-0.83, +0.86] alpha_t 0.1540; 90% [-1.32, +1.35] alpha_t 0.0645; 95% [-1.91, +1.93] alpha_t 0.0273
- H-NN10 vol (bar 16:00): ML 0.860% vs baseline 1.027% (ratio 0.84) - not this bar's line

## Next 24h [11-15T17:00 -> 11-16T17:00], % log return
- PNN median +0.76, 80% [-2.99, +4.97], 90% [-5.00, +6.50], sigma 3.10%
- ACI 80% [-2.92, +4.89] alpha_t 0.2045; 90% [-5.48, +6.98] alpha_t 0.0909; 95% [-7.85, +9.35] alpha_t 0.0399
- 24h net: rolling day, shift 17 h

## v2.1 shadow (ACI gamma 0.02, issue alpha >= nominal/4, jump-cluster floor; cfg 2d4ba801a6363fe9; v21-20261115T17)
- 1h: 80% [-1.04, +1.08]; 90% [-1.82, +1.85]; 95% [-2.61, +2.64]; alpha_t (unclipped) 80/90/95 0.1020/0.0335/0.0018; floor inactive
- 24h: 80% [-3.06, +5.04]; 90% [-5.75, +7.25]; 95% [-8.37, +9.87]; alpha_t 0.1942/0.0817/0.0337
- running coverage (issued, 72 / 168): 1h 80% 0.778/0.774, 90% 0.861/0.881, 95% 0.917/0.946 | 24h 80% 0.583/0.720, 90% 0.708/0.804, 95% 0.833/0.887
- v1 vs v2 since 11-13T20:00 (same bars; v2 mostly replay): 1h n 45: cov80 0.644 vs 0.733, cov95 0.867 vs 0.911, band pinball 0.1234 vs 0.1165; 24h n 22: cov80 0.045 vs 0.045, cov95 0.500 vs 0.500, band pinball 1.1156 vs 1.1156

## Regime and players (information only, changes no forecast; as_of 2026-11-15T17:03Z)
| | v0 cell | 24h px | funding (90d pct) | OI 24h / 72h | liq 4h long/short | liq 24h long/short | DVOL (90d pct) |
|---|---|---|---|---|---|---|---|
| BTC | R9 liquidation | -1.0% | -0.0012% (p4) | +1.6% / +2.4% | 2.0M / 0.4M | 60.0M / 14.4M | 44.4 (p99) |
| ETH | R9 liquidation | -0.6% | -0.0050% (p4) | -2.6% / +1.9% | 0.9M / 0.2M | 30.8M / 3.9M | 57.7 (p95) |
| SOL | R9 liquidation | -1.4% | -0.0195% (p0) | -2.2% / -5.8% | 0.2M / 76k | 6.0M / 1.1M | no index |
- Pattern: BTC long flush, funding negative, OI rising into the drop (new shorts), long-liquidation burst (4h), implied vol high; ETH long flush, funding negative, OI falling with price (deleveraging), long-liquidation burst (4h), implied vol high; SOL long flush, funding negative, OI falling with price (deleveraging), long-liquidation burst (4h)
- Long/short ratio omitted: lake ls_ratio last known 2026-11-07T20:00 (one-off backfill of 11-07; the live updater has no ls_ratio stream)

## Cross-section v3 (context only, not a signal: R-XS-2/R-XS-3; xs3-20261115T17, 23/23 tokens)
- 4h relative median vs EW basket, bp [80% ACI]: top BTC +4 [-118, +126], DOGE +1 [-72, +79], TRX +1 [-278, +262], ETC +1 [-72, +77], XRP -0 [-135, +144]
- bottom FIL -10 [-164, +163], NEAR -12 [-157, +146], UNI -12 [-186, +178], AAVE -12 [-141, +135], ENA -14 [-147, +134]
- mostly low vol: rank corr with -lsig2 4h +0.54 / 24h +0.65; top 5 shared with the low-vol top 5: 3/5 (TRX, BTC, BNB, LTC, ETC)
- BTC 1h |r|: v3 q50 0.42%, q90 1.15% (log-HAR-t q90 1.29%) vs v1 90% half-width 0.98% (ACI90 1.34%)
- running (scored lines): 4h IC n/a; 24h IC n/a; 1h vol pinball v3/HAR 1.007 (n 3)
- 24h ACI band under-covers (holdout 0.749 at 80%); the decision sessions do not read this block

## Field card (btc_field_2026-11-13.json, regime R9)
- 1h: 90% [-1.39, +1.43] med +0.01; touch +1/-1 24/23%, +2/-2 3/3%, +3/-3 1/1%; P(jump) 2%
- 4h: 90% [-3.21, +3.37] med +0.12; touch +1/-1 60/56%, +2/-2 28/25%, +3/-3 12/11%; P(jump) 6%
- 24h: 90% [-10.43, +9.85] med +0.84; touch +1/-1 87/83%, +2/-2 74/66%, +3/-3 61/54%; P(jump) 31%
- variance drivers: x1 -0.36, s +0.26, x2 -0.13; jump drivers: m24*dir +0.02

## Latent (1h net, 4 dims)
- z [0.84, 1.08, -0.6, -0.44], predicted next [0.71, 0.86, -0.59, -0.42]
- last hour: model error 0.41 vs persistence 0.58 (latent sd); 168h R^2 gain +0.128

## News
- alt: wp_btc -8.02, wp_ind -6.27, wp_max -4.92, wp_edits +0.00, hn_story +0.00, hn_comment -0.58, fng -0.80, fng_chg -0.06
- RSS items last 1h 23, 24h 23

## Running coverage (ACI, last 72 / 168 resolved)
- 1h: 80% 0.694 / 0.738; 90% 0.806 / 0.857; 95% 0.875 / 0.929 (n 168)
- 24h: 80% 0.583 / 0.720; 90% 0.708 / 0.804; 95% 0.833 / 0.887 (n 168)

## Scored this run
- nn-20261114T17 24h: r -0.990% (z -0.63), in ACI80 True, pinball 0.2618
- nn-20261115T16 1h: r -0.072% (z -0.12), in ACI80 True, pinball 0.0703

## Alarms (8)
- coverage (band level/window cov): 1h 90%/72 0.806; 1h 95%/72 0.875; 24h 80%/72 0.583; 24h 80%/168 0.720; 24h 90%/72 0.708; 24h 90%/168 0.804; 24h 95%/72 0.833; 24h 95%/168 0.887

## Changed vs previous line (nn-20261115T16)
- 1h sigma 0.60 -> 0.55%; 24h sigma 3.12 -> 3.10%; 1h alpha80 0.1530 -> 0.1540; field 1h P(jump) 1.6 -> 1.6%
- new alarms: 3
