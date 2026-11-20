# BTC live network: bar 2026-11-20 08:00 UTC (line nn-20261120T08)
as_of 2026-11-20T08:03:03Z, newest bar closed 08:00, close 61,821.1. Flags: rolling_day. Research only, no direction call.

## Next hour [11-20T08:00 -> 11-20T09:00], % log return
- PNN median +0.02, 80% [-0.59, +0.61], 90% [-0.82, +0.82], sigma 0.47%
- ACI 80% [-0.64, +0.66] alpha_t 0.1750; 90% [-0.96, +0.97] alpha_t 0.0750; 95% [-1.24, +1.25] alpha_t 0.0450
- H-NN10 vol (bar 07:00): ML 0.587% vs baseline 0.607% (ratio 0.97) - not this bar's line

## Next 24h [11-20T08:00 -> 11-21T08:00], % log return
- PNN median +0.27, 80% [-3.57, +4.81], 90% [-5.51, +6.22], sigma 3.27%
- ACI 80% [-3.42, +4.66] alpha_t 0.2067; 90% [-5.89, +6.59] alpha_t 0.0925; 95% [-8.55, +9.25] alpha_t 0.0407
- 24h net: rolling day, shift 8 h

## v2.1 shadow (ACI gamma 0.02, issue alpha >= nominal/4, jump-cluster floor; cfg 2d4ba801a6363fe9; v21-20261120T08)
- 1h: 80% [-0.70, +0.72]; 90% [-1.10, +1.10]; 95% [-1.11, +1.11]; alpha_t (unclipped) 80/90/95 0.1460/0.0555/0.0528; floor inactive
- 24h: 80% [-3.45, +4.69]; 90% [-6.04, +6.74]; 95% [-8.82, +9.52]; alpha_t 0.2045/0.0881/0.0370
- running coverage (issued, 72 / 168): 1h 80% 0.778/0.809, 90% 0.889/0.905, 95% 0.958/0.958 | 24h 80% 0.972/0.774, 90% 1.000/0.851, 95% 1.000/0.911
- v1 vs v2 since 11-13T20:00 (same bars; v2 mostly replay): 1h n 109: cov80 0.743 vs 0.771, cov95 0.936 vs 0.945, band pinball 0.0895 vs 0.0889; 24h n 99: cov80 0.737 vs 0.758, cov95 0.889 vs 0.889, band pinball 0.4718 vs 0.4784

## Regime and players (information only, changes no forecast; as_of 2026-11-20T08:03Z)
| | v0 cell | 24h px | funding (90d pct) | OI 24h / 72h | liq 4h long/short | liq 24h long/short | DVOL (90d pct) |
|---|---|---|---|---|---|---|---|
| BTC | R9 liquidation | -3.0% | +0.0026% (p26) | -0.6% / -3.7% | 0.4M / 22k | 4.8M / 1.0M | 48.6 (p100) |
| ETH | R9 liquidation | -2.9% | +0.0053% (p66) | -1.1% / -3.4% | 0.8M / 0.3M | 3.2M / 0.4M | 61.1 (p99) |
| SOL | R9 liquidation | -3.1% | -0.0029% (p23) | -0.5% / -3.1% | 0.2M / 2k | 0.8M / 11k | no index |
- Pattern: BTC long flush, funding normal, OI flat, long-liquidation burst (4h), implied vol high; ETH long flush, funding normal, OI falling with price (deleveraging), long-liquidation burst (4h), implied vol high; SOL long flush, funding negative, OI flat, long-liquidation burst (4h)
- Long/short ratio omitted: lake ls_ratio last known 2026-11-07T20:00 (one-off backfill of 11-07; the live updater has no ls_ratio stream)

## Cross-section v3 (context only, not a signal: R-XS-2/R-XS-3; xs3-20261120T08, 23/23 tokens)
- 4h relative median vs EW basket, bp [80% ACI]: top ETC +2 [-48, +48], DOGE +2 [-48, +49], SUI +1 [-42, +43], BTC +0 [-77, +74], TRX +0 [-187, +183]
- bottom XRP -4 [-107, +102], NEAR -6 [-146, +141], UNI -8 [-127, +122], ENA -10 [-170, +172], FIL -12 [-163, +156]
- mostly low vol: rank corr with -lsig2 4h +0.54 / 24h +0.73; top 5 shared with the low-vol top 5: 2/5 (TRX, ATOM, BNB, BTC, XRP)
- BTC 1h |r|: v3 q50 0.27%, q90 0.75% (log-HAR-t q90 0.91%) vs v1 90% half-width 0.82% (ACI90 0.96%)
- running (scored lines): 4h IC v3 +0.091 vs low-vol +0.030 (n 66); 24h IC v3 +0.172 vs low-vol +0.142 (n 59); 1h vol pinball v3/HAR 0.940 (n 69)
- 24h ACI band under-covers (holdout 0.749 at 80%); the decision sessions do not read this block

## Field card (btc_field_2026-11-13.json, regime R9)
- 1h: 90% [-0.98, +0.99] med +0.01; touch +1/-1 10/9%, +2/-2 1/1%, +3/-3 0/0%; P(jump) 2%
- 4h: 90% [-2.01, +2.06] med +0.06; touch +1/-1 37/34%, +2/-2 10/9%, +3/-3 3/3%; P(jump) 7%
- 24h: 90% [-5.36, +5.06] med +0.38; touch +1/-1 74/68%, +2/-2 50/44%, +3/-3 31/28%; P(jump) 33%
- variance drivers: m24*dir +0.26, x1 -0.14, x2 -0.11; jump drivers: m24*dir +0.14

## Latent (1h net, 4 dims)
- z [0.2, 0.53, -0.59, -0.27], predicted next [0.29, 0.66, -0.36, -0.32]
- last hour: model error 0.44 vs persistence 0.55 (latent sd); 168h R^2 gain +0.162

## News
- alt: wp_btc -8.02, wp_ind -6.24, wp_max -4.91, wp_edits +0.69, hn_story -0.37, hn_comment -0.67, fng -0.76, fng_chg +0.04
- RSS items last 1h 0, 24h 54

## Running coverage (ACI, last 72 / 168 resolved)
- 1h: 80% 0.806 / 0.792; 90% 0.889 / 0.887; 95% 0.972 / 0.952 (n 168)
- 24h: 80% 0.944 / 0.762; 90% 1.000 / 0.851; 95% 1.000 / 0.911 (n 168)

## Scored this run
- nn-20261119T08 24h: r -3.017% (z -0.85), in ACI80 True, pinball 0.7140
- nn-20261120T07 1h: r -0.320% (z -0.69), in ACI80 True, pinball 0.0789

## Alarms
- coverage 24h ACI 80% over last 72: 0.944 vs 0.80
- coverage 24h ACI 90% over last 72: 1.000 vs 0.90

## Changed vs previous line (nn-20261120T07)
- 1h sigma 0.46 -> 0.47%; 24h sigma 3.24 -> 3.27%; 1h alpha80 0.1740 -> 0.1750; field 1h P(jump) 1.7 -> 1.8%
