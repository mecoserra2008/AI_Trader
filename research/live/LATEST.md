# BTC live network: bar 2026-11-20 04:00 UTC (line nn-20261120T04)
as_of 2026-11-20T04:03:04Z, newest bar closed 04:00, close 62,123.1. Flags: rolling_day. Research only, no direction call.

## Next hour [11-20T04:00 -> 11-20T05:00], % log return
- PNN median +0.02, 80% [-0.56, +0.59], 90% [-0.79, +0.81], sigma 0.45%
- ACI 80% [-0.61, +0.64] alpha_t 0.1760; 90% [-0.91, +0.93] alpha_t 0.0780; 95% [-1.21, +1.23] alpha_t 0.0440
- H-NN10 vol (bar 03:00): ML 0.799% vs baseline 0.576% (ratio 1.39) - not this bar's line

## Next 24h [11-20T04:00 -> 11-21T04:00], % log return
- PNN median +0.13, 80% [-3.93, +4.59], 90% [-5.92, +6.13], sigma 3.33%
- ACI 80% [-3.79, +4.45] alpha_t 0.2065; 90% [-6.32, +6.53] alpha_t 0.0924; 95% [-9.01, +9.22] alpha_t 0.0407
- 24h net: rolling day, shift 4 h

## v2.1 shadow (ACI gamma 0.02, issue alpha >= nominal/4, jump-cluster floor; cfg 2d4ba801a6363fe9; v21-20261120T04)
- 1h: 80% [-0.67, +0.70]; 90% [-0.97, +0.99]; 95% [-0.96, +0.98]; alpha_t (unclipped) 80/90/95 0.1500/0.0675/0.0688; floor inactive
- 24h: 80% [-3.82, +4.48]; 90% [-6.48, +6.69]; 95% [-9.29, +9.50]; alpha_t 0.2038/0.0878/0.0368
- running coverage (issued, 72 / 168): 1h 80% 0.792/0.809, 90% 0.903/0.905, 95% 0.972/0.964 | 24h 80% 0.972/0.774, 90% 1.000/0.851, 95% 1.000/0.911
- v1 vs v2 since 11-13T20:00 (same bars; v2 mostly replay): 1h n 105: cov80 0.743 vs 0.771, cov95 0.933 vs 0.952, band pinball 0.0901 vs 0.0894; 24h n 95: cov80 0.726 vs 0.747, cov95 0.884 vs 0.884, band pinball 0.4773 vs 0.4839

## Regime and players (information only, changes no forecast; as_of 2026-11-20T04:03Z)
| | v0 cell | 24h px | funding (90d pct) | OI 24h / 72h | liq 4h long/short | liq 24h long/short | DVOL (90d pct) |
|---|---|---|---|---|---|---|---|
| BTC | R9 liquidation | -1.7% | +0.0026% (p26) | -0.5% / +1.3% | 4.0M / 0.6M | 4.4M / 1.6M | 46.4 (p99) |
| ETH | R9 liquidation | -2.7% | +0.0053% (p65) | -1.8% / -0.9% | 2.3M / 0.1M | 2.4M / 0.5M | 61.3 (p99) |
| SOL | R6 whipsaw | -1.8% | -0.0029% (p23) | +1.7% / -3.3% | 0.7M / 5k | 0.7M / 0.1M | no index |
- Pattern: BTC long flush, funding normal, OI flat, long-liquidation burst (4h), implied vol high; ETH long flush, funding normal, OI falling with price (deleveraging), long-liquidation burst (4h), implied vol high; SOL long flush, funding negative, OI rebuilding while price fell (new shorts), long-liquidation burst (4h)
- Long/short ratio omitted: lake ls_ratio last known 2026-11-07T20:00 (one-off backfill of 11-07; the live updater has no ls_ratio stream)

## Cross-section v3 (context only, not a signal: R-XS-2/R-XS-3; xs3-20261120T04, 23/23 tokens)
- 4h relative median vs EW basket, bp [80% ACI]: top ETC +4 [-44, +46], LTC +4 [-69, +72], DOGE +1 [-50, +49], DOT +1 [-72, +72], SUI +1 [-44, +43]
- bottom BTC -4 [-104, +96], NEAR -5 [-164, +156], SOL -5 [-89, +83], BCH -6 [-115, +108], XRP -10 [-141, +137]
- mostly low vol: rank corr with -lsig2 4h +0.04 / 24h +0.53; top 5 shared with the low-vol top 5: 0/5 (TRX, BNB, BTC, ATOM, XRP)
- BTC 1h |r|: v3 q50 0.31%, q90 0.80% (log-HAR-t q90 0.73%) vs v1 90% half-width 0.80% (ACI90 0.92%)
- running (scored lines): 4h IC v3 +0.086 vs low-vol +0.023 (n 62); 24h IC v3 +0.152 vs low-vol +0.123 (n 55); 1h vol pinball v3/HAR 0.945 (n 65)
- 24h ACI band under-covers (holdout 0.749 at 80%); the decision sessions do not read this block

## Field card (btc_field_2026-11-13.json, regime R9)
- 1h: 90% [-0.65, +0.67] med +0.00; touch +1/-1 2/2%, +2/-2 0/0%, +3/-3 0/0%; P(jump) 2%
- 4h: 90% [-1.55, +1.60] med +0.05; touch +1/-1 26/24%, +2/-2 4/4%, +3/-3 1/1%; P(jump) 7%
- 24h: 90% [-4.91, +4.94] med +0.33; touch +1/-1 73/67%, +2/-2 48/43%, +3/-3 28/26%; P(jump) 32%
- variance drivers: s -0.50, m24*dir +0.14, x1 -0.12; jump drivers: m24*dir +0.07

## Latent (1h net, 4 dims)
- z [0.21, 0.51, -0.24, -0.38], predicted next [0.32, 0.62, -0.2, -0.35]
- last hour: model error 0.52 vs persistence 1.00 (latent sd); 168h R^2 gain +0.163

## News
- alt: wp_btc -8.02, wp_ind -6.24, wp_max -4.91, wp_edits +0.69, hn_story -0.37, hn_comment -0.67, fng -0.76, fng_chg +0.04
- RSS items last 1h 2, 24h 54

## Running coverage (ACI, last 72 / 168 resolved)
- 1h: 80% 0.819 / 0.786; 90% 0.903 / 0.887; 95% 0.972 / 0.946 (n 168)
- 24h: 80% 0.944 / 0.762; 90% 1.000 / 0.851; 95% 1.000 / 0.911 (n 168)

## Scored this run
- nn-20261119T04 24h: r -1.688% (z -0.46), in ACI80 True, pinball 0.5065
- nn-20261120T03 1h: r -0.200% (z -0.34), in ACI80 True, pinball 0.0804

## Alarms
- coverage 24h ACI 80% over last 72: 0.944 vs 0.80
- coverage 24h ACI 90% over last 72: 1.000 vs 0.90

## Changed vs previous line (nn-20261120T03)
- 1h sigma 0.59 -> 0.45%; 24h sigma 3.20 -> 3.33%; 1h alpha80 0.1750 -> 0.1760; field 1h P(jump) 1.6 -> 1.6%
- new alarms: 1
