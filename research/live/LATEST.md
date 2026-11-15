# BTC live network: bar 2026-11-15 18:00 UTC (line nn-20261115T18)
as_of 2026-11-15T18:03:06Z, newest bar closed 18:00, close 64,538.3. Flags: rolling_day. Research only, no direction call.

## Next hour [11-15T18:00 -> 11-15T19:00], % log return
- PNN median +0.00, 80% [-0.59, +0.59], 90% [-0.83, +0.82], sigma 0.46%
- ACI 80% [-0.72, +0.73] alpha_t 0.1550; 90% [-1.17, +1.16] alpha_t 0.0650; 95% [-1.75, +1.73] alpha_t 0.0275
- H-NN10 vol (bar 17:00): ML 0.763% vs baseline 0.980% (ratio 0.78) - not this bar's line

## Next 24h [11-15T18:00 -> 11-16T18:00], % log return
- PNN median +0.62, 80% [-3.20, +4.82], 90% [-5.16, +6.44], sigma 3.13%
- ACI 80% [-3.15, +4.76] alpha_t 0.2044; 90% [-5.64, +6.92] alpha_t 0.0909; 95% [-8.01, +9.29] alpha_t 0.0399
- 24h net: rolling day, shift 18 h

## v2.1 shadow (ACI gamma 0.02, issue alpha >= nominal/4, jump-cluster floor; cfg 2d4ba801a6363fe9; v21-20261115T18)
- 1h: 80% [-0.91, +0.92]; 90% [-1.65, +1.64]; 95% [-2.43, +2.41]; alpha_t (unclipped) 80/90/95 0.1060/0.0355/0.0028; floor inactive
- 24h: 80% [-3.28, +4.90]; 90% [-5.91, +7.19]; 95% [-8.53, +9.81]; alpha_t 0.1935/0.0818/0.0338
- running coverage (issued, 72 / 168): 1h 80% 0.778/0.774, 90% 0.861/0.881, 95% 0.917/0.946 | 24h 80% 0.569/0.714, 90% 0.708/0.804, 95% 0.833/0.887
- v1 vs v2 since 11-13T20:00 (same bars; v2 mostly replay): 1h n 46: cov80 0.652 vs 0.739, cov95 0.870 vs 0.913, band pinball 0.1221 vs 0.1159; 24h n 23: cov80 0.043 vs 0.043, cov95 0.522 vs 0.522, band pinball 1.0775 vs 1.0775

## Regime and players (information only, changes no forecast; as_of 2026-11-15T18:03Z)
| | v0 cell | 24h px | funding (90d pct) | OI 24h / 72h | liq 4h long/short | liq 24h long/short | DVOL (90d pct) |
|---|---|---|---|---|---|---|---|
| BTC | R9 liquidation | -2.0% | -0.0012% (p4) | +1.6% / +2.4% | 2.1M / 0.3M | 60.0M / 13.6M | 44.4 (p99) |
| ETH | R9 liquidation | -1.6% | -0.0050% (p4) | -2.6% / +1.9% | 1.0M / 0.2M | 30.7M / 3.7M | 57.5 (p94) |
| SOL | R9 liquidation | -1.9% | -0.0195% (p0) | -2.2% / -5.8% | 0.2M / 53k | 6.0M / 1.1M | no index |
- Pattern: BTC long flush, funding negative, OI rising into the drop (new shorts), long-liquidation burst (4h), implied vol high; ETH long flush, funding negative, OI falling with price (deleveraging), long-liquidation burst (4h), implied vol high; SOL long flush, funding negative, OI falling with price (deleveraging), long-liquidation burst (4h)
- Long/short ratio omitted: lake ls_ratio last known 2026-11-07T20:00 (one-off backfill of 11-07; the live updater has no ls_ratio stream)

## Cross-section v3 (context only, not a signal: R-XS-2/R-XS-3; xs3-20261115T18, 23/23 tokens)
- 4h relative median vs EW basket, bp [80% ACI]: top BTC +5 [-115, +124], ETC +0 [-72, +76], TRX +0 [-273, +256], ETH -0 [-104, +109], XRP -1 [-131, +139]
- bottom ATOM -10 [-182, +175], UNI -11 [-173, +162], AAVE -12 [-135, +128], NEAR -13 [-153, +139], ENA -14 [-144, +130]
- mostly low vol: rank corr with -lsig2 4h +0.58 / 24h +0.64; top 5 shared with the low-vol top 5: 3/5 (TRX, BTC, BNB, LTC, ETC)
- BTC 1h |r|: v3 q50 0.39%, q90 1.10% (log-HAR-t q90 1.24%) vs v1 90% half-width 0.82% (ACI90 1.16%)
- running (scored lines): 4h IC v3 -0.053 vs low-vol +0.104 (n 1); 24h IC n/a; 1h vol pinball v3/HAR 0.988 (n 4)
- 24h ACI band under-covers (holdout 0.749 at 80%); the decision sessions do not read this block

## Field card (btc_field_2026-11-13.json, regime R9)
- 1h: 90% [-1.36, +1.41] med +0.01; touch +1/-1 23/22%, +2/-2 3/3%, +3/-3 1/1%; P(jump) 2%
- 4h: 90% [-3.56, +3.71] med +0.13; touch +1/-1 65/61%, +2/-2 34/30%, +3/-3 16/15%; P(jump) 6%
- 24h: 90% [-10.35, +9.83] med +0.76; touch +1/-1 87/83%, +2/-2 74/66%, +3/-3 61/54%; P(jump) 31%
- variance drivers: x1 -0.34, s +0.22, x2 -0.13; jump drivers: m24*dir +0.05

## Latent (1h net, 4 dims)
- z [1.15, 1.05, -0.67, -0.35], predicted next [0.93, 0.85, -0.63, -0.38]
- last hour: model error 0.55 vs persistence 0.33 (latent sd); 168h R^2 gain +0.119

## News
- alt: wp_btc -8.02, wp_ind -6.27, wp_max -4.92, wp_edits +0.00, hn_story +0.00, hn_comment -0.58, fng -0.80, fng_chg -0.06
- RSS items last 1h 0, 24h 23

## Running coverage (ACI, last 72 / 168 resolved)
- 1h: 80% 0.694 / 0.738; 90% 0.806 / 0.857; 95% 0.875 / 0.929 (n 168)
- 24h: 80% 0.569 / 0.714; 90% 0.708 / 0.804; 95% 0.833 / 0.887 (n 168)

## Scored this run
- nn-20261114T18 24h: r -2.041% (z -1.34), in ACI80 False, pinball 0.5579
- nn-20261115T17 1h: r -0.210% (z -0.38), in ACI80 True, pinball 0.0753

## Alarms (8)
- coverage (band level/window cov): 1h 90%/72 0.806; 1h 95%/72 0.875; 24h 80%/72 0.569; 24h 80%/168 0.714; 24h 90%/72 0.708; 24h 90%/168 0.804; 24h 95%/72 0.833; 24h 95%/168 0.887

## Changed vs previous line (nn-20261115T17)
- 1h sigma 0.55 -> 0.46%; 24h sigma 3.10 -> 3.13%; 1h alpha80 0.1540 -> 0.1550; field 1h P(jump) 1.6 -> 1.6%
- new alarms: 2
