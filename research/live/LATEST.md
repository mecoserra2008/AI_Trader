# BTC live network: bar 2026-11-15 20:00 UTC (line nn-20261115T20)
as_of 2026-11-15T20:03:07Z, newest bar closed 20:00, close 64,919.9. Flags: rolling_day. Research only, no direction call.

## Next hour [11-15T20:00 -> 11-15T21:00], % log return
- PNN median +0.01, 80% [-0.73, +0.75], 90% [-1.03, +1.03], sigma 0.58%
- ACI 80% [-0.85, +0.87] alpha_t 0.1570; 90% [-1.31, +1.32] alpha_t 0.0660; 95% [-1.89, +1.89] alpha_t 0.0280
- H-NN10 vol (bar 19:00): ML 0.746% vs baseline 0.970% (ratio 0.77) - not this bar's line

## Next 24h [11-15T20:00 -> 11-16T20:00], % log return
- PNN median +0.60, 80% [-3.14, +4.59], 90% [-5.07, +6.15], sigma 3.02%
- ACI 80% [-3.09, +4.53] alpha_t 0.2045; 90% [-5.55, +6.63] alpha_t 0.0910; 95% [-7.92, +9.00] alpha_t 0.0400
- 24h net: rolling day, shift 20 h

## v2.1 shadow (ACI gamma 0.02, issue alpha >= nominal/4, jump-cluster floor; cfg 2d4ba801a6363fe9; v21-20261115T20)
- 1h: 80% [-1.00, +1.02]; 90% [-1.73, +1.73]; 95% [-2.54, +2.55]; alpha_t (unclipped) 80/90/95 0.1140/0.0395/0.0048; floor inactive
- 24h: 80% [-3.22, +4.66]; 90% [-5.81, +6.90]; 95% [-8.35, +9.44]; alpha_t 0.1938/0.0819/0.0339
- running coverage (issued, 72 / 168): 1h 80% 0.778/0.780, 90% 0.861/0.881, 95% 0.917/0.946 | 24h 80% 0.583/0.714, 90% 0.708/0.804, 95% 0.833/0.887
- v1 vs v2 since 11-13T20:00 (same bars; v2 mostly replay): 1h n 48: cov80 0.667 vs 0.750, cov95 0.875 vs 0.917, band pinball 0.1195 vs 0.1143; 24h n 25: cov80 0.120 vs 0.120, cov95 0.560 vs 0.560, band pinball 1.0032 vs 1.0033

## Regime and players (information only, changes no forecast; as_of 2026-11-15T20:03Z)
| | v0 cell | 24h px | funding (90d pct) | OI 24h / 72h | liq 4h long/short | liq 24h long/short | DVOL (90d pct) |
|---|---|---|---|---|---|---|---|
| BTC | R9 liquidation | -1.2% | -0.0012% (p4) | +1.1% / -3.4% | 0.5M / 0.6M | 59.8M / 13.9M | 44.2 (p98) |
| ETH | R9 liquidation | -0.6% | -0.0050% (p4) | -2.7% / -7.2% | 0.1M / 0.3M | 30.6M / 3.8M | 57.5 (p94) |
| SOL | R9 liquidation | -0.8% | -0.0195% (p0) | -1.0% / -5.7% | 30k / 49k | 5.8M / 1.1M | no index |
- Pattern: BTC long flush, funding negative, OI rebuilding after a drop while price fell (new shorts), implied vol high; ETH long flush, funding negative, OI falling with price (deleveraging), short-liquidation burst (4h), implied vol high; SOL long flush, funding negative, OI flat
- Long/short ratio omitted: lake ls_ratio last known 2026-11-07T20:00 (one-off backfill of 11-07; the live updater has no ls_ratio stream)

## Cross-section v3 (context only, not a signal: R-XS-2/R-XS-3; xs3-20261115T20, 23/23 tokens)
- 4h relative median vs EW basket, bp [80% ACI]: top TRX +2 [-259, +244], BTC +2 [-116, +118], BNB -0 [-114, +113], ETH -0 [-106, +108], DOGE -2 [-76, +75]
- bottom HYPE -9 [-169, +167], UNI -10 [-186, +183], ENA -13 [-144, +134], AAVE -15 [-143, +139], NEAR -17 [-168, +160]
- mostly low vol: rank corr with -lsig2 4h +0.71 / 24h +0.71; top 5 shared with the low-vol top 5: 3/5 (TRX, BTC, BNB, LTC, ETC)
- BTC 1h |r|: v3 q50 0.41%, q90 1.14% (log-HAR-t q90 1.39%) vs v1 90% half-width 1.03% (ACI90 1.32%)
- running (scored lines): 4h IC v3 -0.193 vs low-vol -0.177 (n 3); 24h IC n/a; 1h vol pinball v3/HAR 0.970 (n 6)
- 24h ACI band under-covers (holdout 0.749 at 80%); the decision sessions do not read this block

## Field card (btc_field_2026-11-13.json, regime R9)
- 1h: 90% [-1.63, +1.68] med +0.01; touch +1/-1 32/31%, +2/-2 6/5%, +3/-3 1/1%; P(jump) 2%
- 4h: 90% [-4.45, +4.70] med +0.15; touch +1/-1 72/68%, +2/-2 44/40%, +3/-3 25/22%; P(jump) 6%
- 24h: 90% [-9.59, +9.63] med +0.83; touch +1/-1 87/82%, +2/-2 72/65%, +3/-3 59/51%; P(jump) 31%
- variance drivers: s +0.62, x1 -0.31, x2 -0.13; jump drivers: m24*dir +0.03

## Latent (1h net, 4 dims)
- z [0.56, 0.83, -0.36, -0.34], predicted next [0.53, 0.81, -0.38, -0.32]
- last hour: model error 0.28 vs persistence 0.44 (latent sd); 168h R^2 gain +0.119

## News
- alt: wp_btc -8.02, wp_ind -6.27, wp_max -4.92, wp_edits +0.00, hn_story +0.00, hn_comment -0.58, fng -0.80, fng_chg -0.06
- RSS items last 1h 1, 24h 24

## Running coverage (ACI, last 72 / 168 resolved)
- 1h: 80% 0.694 / 0.744; 90% 0.806 / 0.857; 95% 0.875 / 0.929 (n 168)
- 24h: 80% 0.583 / 0.714; 90% 0.708 / 0.804; 95% 0.833 / 0.887 (n 168)

## Scored this run
- nn-20261114T20 24h: r -1.188% (z -0.79), in ACI80 True, pinball 0.2962
- nn-20261115T19 1h: r +0.326% (z +0.69), in ACI80 True, pinball 0.0801

## Alarms (8)
- coverage (band level/window cov): 1h 90%/72 0.806; 1h 95%/72 0.875; 24h 80%/72 0.583; 24h 80%/168 0.714; 24h 90%/72 0.708; 24h 90%/168 0.804; 24h 95%/72 0.833; 24h 95%/168 0.887

## Changed vs previous line (nn-20261115T19)
- 1h sigma 0.47 -> 0.58%; 24h sigma 3.08 -> 3.02%; 1h alpha80 0.1560 -> 0.1570; field 1h P(jump) 1.6 -> 1.6%
