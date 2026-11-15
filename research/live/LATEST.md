# BTC live network: bar 2026-11-15 19:00 UTC (line nn-20261115T19)
as_of 2026-11-15T19:03:08Z, newest bar closed 19:00, close 64,708.8. Flags: rolling_day. Research only, no direction call.

## Next hour [11-15T19:00 -> 11-15T20:00], % log return
- PNN median -0.01, 80% [-0.62, +0.59], 90% [-0.87, +0.82], sigma 0.47%
- ACI 80% [-0.75, +0.72] alpha_t 0.1560; 90% [-1.18, +1.13] alpha_t 0.0655; 95% [-1.76, +1.71] alpha_t 0.0278
- H-NN10 vol (bar 17:00): ML 0.763% vs baseline 0.980% (ratio 0.78) - not this bar's line

## Next 24h [11-15T19:00 -> 11-16T19:00], % log return
- PNN median +0.64, 80% [-3.16, +4.74], 90% [-5.14, +6.33], sigma 3.08%
- ACI 80% [-3.10, +4.69] alpha_t 0.2044; 90% [-5.62, +6.81] alpha_t 0.0910; 95% [-7.98, +9.17] alpha_t 0.0400
- 24h net: rolling day, shift 19 h

## v2.1 shadow (ACI gamma 0.02, issue alpha >= nominal/4, jump-cluster floor; cfg 2d4ba801a6363fe9; v21-20261115T19)
- 1h: 80% [-0.93, +0.89]; 90% [-1.65, +1.59]; 95% [-2.42, +2.37]; alpha_t (unclipped) 80/90/95 0.1100/0.0375/0.0038; floor inactive
- 24h: 80% [-3.23, +4.82]; 90% [-5.88, +7.07]; 95% [-8.42, +9.61]; alpha_t 0.1937/0.0818/0.0338
- running coverage (issued, 72 / 168): 1h 80% 0.778/0.780, 90% 0.861/0.881, 95% 0.917/0.946 | 24h 80% 0.583/0.714, 90% 0.708/0.804, 95% 0.833/0.887
- v1 vs v2 since 11-13T20:00 (same bars; v2 mostly replay): 1h n 47: cov80 0.660 vs 0.745, cov95 0.872 vs 0.915, band pinball 0.1208 vs 0.1151; 24h n 24: cov80 0.083 vs 0.083, cov95 0.542 vs 0.542, band pinball 1.0389 vs 1.0389

## Regime and players (information only, changes no forecast; as_of 2026-11-15T19:03Z)
| | v0 cell | 24h px | funding (90d pct) | OI 24h / 72h | liq 4h long/short | liq 24h long/short | DVOL (90d pct) |
|---|---|---|---|---|---|---|---|
| BTC | R9 liquidation | -1.3% | -0.0012% (p4) | +1.6% / +2.4% | 0.5M / 0.2M | 59.9M / 13.6M | 44.3 (p99) |
| ETH | R9 liquidation | -0.8% | -0.0050% (p4) | -2.6% / +1.9% | 61k / 0.2M | 30.6M / 3.7M | 57.5 (p94) |
| SOL | R9 liquidation | -1.5% | -0.0195% (p0) | -2.2% / -5.8% | 21k / 46k | 5.9M / 1.1M | no index |
- Pattern: BTC long flush, funding negative, OI rising into the drop (new shorts), long-liquidation burst (4h), implied vol high; ETH long flush, funding negative, OI falling with price (deleveraging), short-liquidation burst (4h), implied vol high; SOL long flush, funding negative, OI falling with price (deleveraging), short-liquidation burst (4h)
- Long/short ratio omitted: lake ls_ratio last known 2026-11-07T20:00 (one-off backfill of 11-07; the live updater has no ls_ratio stream)

## Cross-section v3 (context only, not a signal: R-XS-2/R-XS-3; xs3-20261115T19, 23/23 tokens)
- 4h relative median vs EW basket, bp [80% ACI]: top TRX +3 [-258, +245], BTC +3 [-117, +120], ETC +1 [-71, +74], DOGE -0 [-72, +74], ETH -1 [-103, +106]
- bottom UNI -9 [-177, +170], ATOM -10 [-170, +162], ENA -12 [-137, +127], NEAR -14 [-153, +142], AAVE -15 [-148, +143]
- mostly low vol: rank corr with -lsig2 4h +0.58 / 24h +0.68; top 5 shared with the low-vol top 5: 3/5 (TRX, BTC, BNB, LTC, ETC)
- BTC 1h |r|: v3 q50 0.37%, q90 1.04% (log-HAR-t q90 1.29%) vs v1 90% half-width 0.85% (ACI90 1.16%)
- running (scored lines): 4h IC v3 -0.127 vs low-vol -0.045 (n 2); 24h IC n/a; 1h vol pinball v3/HAR 0.982 (n 5)
- 24h ACI band under-covers (holdout 0.749 at 80%); the decision sessions do not read this block

## Field card (btc_field_2026-11-13.json, regime R9)
- 1h: 90% [-1.44, +1.49] med +0.01; touch +1/-1 26/25%, +2/-2 4/3%, +3/-3 1/1%; P(jump) 2%
- 4h: 90% [-4.05, +4.26] med +0.14; touch +1/-1 69/65%, +2/-2 40/36%, +3/-3 21/19%; P(jump) 6%
- 24h: 90% [-9.89, +9.65] med +0.78; touch +1/-1 87/82%, +2/-2 73/66%, +3/-3 60/53%; P(jump) 31%
- variance drivers: s +0.38, x1 -0.33, x2 -0.13; jump drivers: m24*dir +0.03

## Latent (1h net, 4 dims)
- z [0.94, 0.73, -0.48, -0.31], predicted next [0.81, 0.77, -0.44, -0.31]
- last hour: model error 0.29 vs persistence 0.64 (latent sd); 168h R^2 gain +0.118

## News
- alt: wp_btc -8.02, wp_ind -6.27, wp_max -4.92, wp_edits +0.00, hn_story +0.00, hn_comment -0.58, fng -0.80, fng_chg -0.06
- RSS items last 1h 0, 24h 23

## Running coverage (ACI, last 72 / 168 resolved)
- 1h: 80% 0.694 / 0.744; 90% 0.806 / 0.857; 95% 0.875 / 0.929 (n 168)
- 24h: 80% 0.583 / 0.714; 90% 0.708 / 0.804; 95% 0.833 / 0.887 (n 168)

## Scored this run
- nn-20261114T19 24h: r -1.264% (z -0.82), in ACI80 True, pinball 0.3211
- nn-20261115T18 1h: r +0.264% (z +0.57), in ACI80 True, pinball 0.0683

## Alarms (8)
- coverage (band level/window cov): 1h 90%/72 0.806; 1h 95%/72 0.875; 24h 80%/72 0.583; 24h 80%/168 0.714; 24h 90%/72 0.708; 24h 90%/168 0.804; 24h 95%/72 0.833; 24h 95%/168 0.887

## Changed vs previous line (nn-20261115T18)
- 1h sigma 0.46 -> 0.47%; 24h sigma 3.13 -> 3.08%; 1h alpha80 0.1550 -> 0.1560; field 1h P(jump) 1.6 -> 1.6%
- new alarms: 1
