# BTC live network: bar 2026-11-28 13:00 UTC (line nn-20261128T13)
as_of 2026-11-28T13:03:35Z, newest bar closed 13:00, close 59,175.2. Flags: rolling_day. Research only, no direction call.

## Next hour [11-28T13:00 -> 11-28T14:00], % log return
- PNN median +0.00, 80% [-0.43, +0.43], 90% [-0.61, +0.62], sigma 0.34%
- ACI 80% [-0.43, +0.43] alpha_t 0.2070; 90% [-0.69, +0.69] alpha_t 0.0885; 95% [-0.86, +0.86] alpha_t 0.0493
- H-NN10 vol (bar 13:00): ML 0.348% vs baseline 0.308% (ratio 1.13)

## Next 24h [11-28T13:00 -> 11-29T13:00], % log return
- PNN median -0.15, 80% [-2.58, +2.78], 90% [-3.90, +4.04], sigma 2.09%
- ACI 80% [-2.42, +2.63] alpha_t 0.2070; 90% [-4.30, +4.43] alpha_t 0.0927; 95% [-6.93, +7.07] alpha_t 0.0410
- 24h net: rolling day, shift 13 h

## v2.1 shadow (ACI gamma 0.02, issue alpha >= nominal/4, jump-cluster floor; cfg 2d4ba801a6363fe9; v21-20261128T13)
- 1h: 80% [-0.41, +0.40]; 90% [-0.56, +0.57]; 95% [-0.73, +0.74]; alpha_t (unclipped) 80/90/95 0.2340/0.1295/0.0698; floor inactive
- 24h: 80% [-2.43, +2.63]; 90% [-4.42, +4.56]; 95% [-7.04, +7.18]; alpha_t 0.2058/0.0896/0.0390
- running coverage (issued, 72 / 168): 1h 80% 0.792/0.833, 90% 0.931/0.923, 95% 0.958/0.964 | 24h 80% 0.833/0.786, 90% 0.931/0.869, 95% 0.986/0.929
- v1 vs v2 since 11-13T20:00 (same bars; v2 mostly replay): 1h n 182: cov80 0.775 vs 0.775, cov95 0.951 vs 0.956, band pinball 0.0824 vs 0.0820; 24h n 167: cov80 0.772 vs 0.784, cov95 0.922 vs 0.928, band pinball 0.4619 vs 0.4669

## Regime and players (information only, changes no forecast; as_of 2026-11-28T13:03Z)
| | v0 cell | 24h px | funding (90d pct) | OI 24h / 72h | liq 4h long/short | liq 24h long/short | DVOL (90d pct) |
|---|---|---|---|---|---|---|---|
| BTC | R9 liquidation | -4.0% | +0.0006% (p15) | -0.9% / -3.0% | 0.6M / 0.1M | 2.4M / 0.7M | 53.3 (p96) |
| ETH | R9 liquidation | -4.6% | -0.0017% (p13) | -0.8% / -0.2% | 0.1M / 95k | 1.0M / 0.2M | 65.7 (p95) |
| SOL | R5 balanced range | -3.2% | +0.0008% (p47) | +0.4% / -13.8% | 71k / 3k | 0.2M / 13k | no index |
- Pattern: BTC long flush, funding normal, OI flat, long-liquidation burst (4h), implied vol high; ETH long flush, funding negative, OI flat, implied vol high; SOL long flush, funding normal, OI flat, long-liquidation burst (4h)
- Long/short accounts: BTC 0.72 long (1h), ETH 0.77 long (4h), SOL 0.79 long (4h)

## Cross-section v3 (context only, not a signal: R-XS-2/R-XS-3; xs3-20261128T13, 23/23 tokens)
- 4h relative median vs EW basket, bp [80% ACI]: top ETH +0 [-47, +48], AVAX +0 [-46, +50], BTC -0 [-51, +50], DOT -1 [-53, +56], XLM -2 [-57, +57]
- bottom ATOM -7 [-101, +102], ENA -8 [-97, +105], UNI -8 [-82, +86], HYPE -8 [-105, +107], LTC -9 [-77, +86]
- mostly low vol: rank corr with -lsig2 4h +0.65 / 24h +0.81; top 5 shared with the low-vol top 5: 2/5 (TRX, BNB, BTC, ETH, ETC)
- BTC 1h |r|: v3 q50 0.17%, q90 0.48% (log-HAR-t q90 0.47%) vs v1 90% half-width 0.62% (ACI90 0.69%)
- running (scored lines): 4h IC v3 +0.067 vs low-vol +0.031 (n 140); 24h IC v3 +0.112 vs low-vol +0.079 (n 128); 1h vol pinball v3/HAR 0.966 (n 143)
- 24h ACI band under-covers (holdout 0.749 at 80%); the decision sessions do not read this block

## Field card (btc_field_2026-11-13.json, regime R9)
- 1h: 90% [-0.51, +0.51] med +0.00; touch +1/-1 1/1%, +2/-2 0/0%, +3/-3 0/0%; P(jump) 2%
- 4h: 90% [-1.10, +1.14] med +0.03; touch +1/-1 13/11%, +2/-2 1/2%, +3/-3 0/0%; P(jump) 7%
- 24h: 90% [-3.22, +3.16] med +0.21; touch +1/-1 58/52%, +2/-2 26/24%, +3/-3 10/11%; P(jump) 32%
- variance drivers: s -0.70, m24*dir +0.30, x1 +0.15; jump drivers: m24*dir +0.16

## Latent (1h net, 4 dims)
- z [0.92, 1.33, 0.29, 0.26], predicted next [0.55, 1.1, 0.27, 0.14]
- last hour: model error 0.83 vs persistence 0.90 (latent sd); 168h R^2 gain +0.174

## News
- alt: wp_btc -8.07, wp_ind -6.19, wp_max -4.86, wp_edits +0.00, hn_story -0.23, hn_comment +0.04, fng -0.80, fng_chg -0.04
- RSS items last 1h 0, 24h 28

## Running coverage (ACI, last 72 / 168 resolved)
- 1h: 80% 0.833 / 0.851; 90% 0.944 / 0.923; 95% 0.972 / 0.964 (n 168)
- 24h: 80% 0.833 / 0.774; 90% 0.917 / 0.863; 95% 0.972 / 0.923 (n 168)

## Scored this run
- nn-20261128T12 1h: r -0.054% (z -0.18), in ACI80 True, pinball 0.0353

## Alarms
- none

## Changed vs previous line (nn-20261128T12)
- 1h sigma 0.31 -> 0.34%; 24h sigma 2.14 -> 2.09%; 1h alpha80 0.2060 -> 0.2070; field 1h P(jump) 1.8 -> 1.8%
