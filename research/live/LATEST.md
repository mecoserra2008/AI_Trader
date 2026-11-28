# BTC live network: bar 2026-11-28 08:00 UTC (line nn-20261128T08)
as_of 2026-11-28T08:03:23Z, newest bar closed 08:00, close 59,797.4. Flags: rolling_day. Research only, no direction call.

## Next hour [11-28T08:00 -> 11-28T09:00], % log return
- PNN median -0.00, 80% [-0.35, +0.34], 90% [-0.49, +0.48], sigma 0.27%
- ACI 80% [-0.34, +0.34] alpha_t 0.2120; 90% [-0.57, +0.57] alpha_t 0.0860; 95% [-0.75, +0.75] alpha_t 0.0480
- H-NN10 vol (bar 07:00): ML 0.340% vs baseline 0.304% (ratio 1.12) - not this bar's line

## Next 24h [11-28T08:00 -> 11-29T08:00], % log return
- PNN median +0.30, 80% [-2.15, +3.57], 90% [-3.64, +5.02], sigma 2.23%
- ACI 80% [-1.98, +3.40] alpha_t 0.2070; 90% [-4.06, +5.45] alpha_t 0.0927; 95% [-6.90, +8.29] alpha_t 0.0410
- 24h net: rolling day, shift 8 h

## v2.1 shadow (ACI gamma 0.02, issue alpha >= nominal/4, jump-cluster floor; cfg 2d4ba801a6363fe9; v21-20261128T08)
- 1h: 80% [-0.30, +0.30]; 90% [-0.46, +0.46]; 95% [-0.64, +0.64]; alpha_t (unclipped) 80/90/95 0.2540/0.1195/0.0648; floor inactive
- 24h: 80% [-1.99, +3.40]; 90% [-4.20, +5.58]; 95% [-7.02, +8.41]; alpha_t 0.2058/0.0896/0.0390
- running coverage (issued, 72 / 168): 1h 80% 0.806/0.839, 90% 0.917/0.923, 95% 0.958/0.964 | 24h 80% 0.833/0.786, 90% 0.931/0.869, 95% 0.986/0.929
- v1 vs v2 since 11-13T20:00 (same bars; v2 mostly replay): 1h n 177: cov80 0.780 vs 0.780, cov95 0.949 vs 0.955, band pinball 0.0836 vs 0.0833; 24h n 167: cov80 0.772 vs 0.784, cov95 0.922 vs 0.928, band pinball 0.4619 vs 0.4669

## Regime and players (information only, changes no forecast; as_of 2026-11-28T08:03Z)
| | v0 cell | 24h px | funding (90d pct) | OI 24h / 72h | liq 4h long/short | liq 24h long/short | DVOL (90d pct) |
|---|---|---|---|---|---|---|---|
| BTC | R6 whipsaw | -3.0% | +0.0046% (p50) | -3.0% / -2.7% | 1.6M / 0.1M | 1.8M / 0.5M | 52.6 (p95) |
| ETH | R9 liquidation | -4.6% | +0.0005% (p24) | +1.1% / +2.1% | 0.5M / 14k | 0.9M / 78k | 65.2 (p93) |
| SOL | R6 whipsaw | -2.8% | +0.0087% (p78) | +1.0% / -8.5% | 0.1M / 0k | 0.1M / 10k | no index |
- Pattern: BTC long flush, funding normal, OI falling with price (deleveraging), long-liquidation burst (4h), implied vol high; ETH long flush, funding normal, OI rising into the drop (new shorts), long-liquidation burst (4h), implied vol high; SOL long flush, funding normal, OI rebuilding while price fell (new shorts), long-liquidation burst (4h)
- Long/short accounts: BTC 0.72 long (1h), ETH 0.77 long (4h), SOL 0.79 long (4h)

## Cross-section v3 (context only, not a signal: R-XS-2/R-XS-3; xs3-20261128T08, 23/23 tokens)
- 4h relative median vs EW basket, bp [80% ACI]: top ETH -1 [-49, +48], XLM -1 [-54, +56], BTC -1 [-55, +52], XRP -2 [-61, +62], AVAX -2 [-47, +49]
- bottom SUI -8 [-56, +61], LTC -8 [-74, +77], ENA -9 [-106, +114], HYPE -11 [-127, +126], NEAR -11 [-114, +123]
- mostly low vol: rank corr with -lsig2 4h +0.80 / 24h +0.80; top 5 shared with the low-vol top 5: 2/5 (TRX, BNB, BTC, ETH, ETC)
- BTC 1h |r|: v3 q50 0.17%, q90 0.44% (log-HAR-t q90 0.43%) vs v1 90% half-width 0.49% (ACI90 0.57%)
- running (scored lines): 4h IC v3 +0.065 vs low-vol +0.029 (n 135); 24h IC v3 +0.112 vs low-vol +0.079 (n 128); 1h vol pinball v3/HAR 0.966 (n 138)
- 24h ACI band under-covers (holdout 0.749 at 80%); the decision sessions do not read this block

## Field card (btc_field_2026-11-13.json, regime R6)
- 1h: 90% [-0.40, +0.40] med +0.00; touch +1/-1 1/1%, +2/-2 0/0%, +3/-3 0/0%; P(jump) 2%
- 4h: 90% [-0.88, +0.93] med +0.02; touch +1/-1 7/6%, +2/-2 1/1%, +3/-3 0/0%; P(jump) 7%
- 24h: 90% [-2.71, +2.80] med +0.15; touch +1/-1 53/46%, +2/-2 21/19%, +3/-3 7/7%; P(jump) 32%
- variance drivers: s -1.01, x1 +0.17, m24*dir +0.16; jump drivers: m24*dir +0.09

## Latent (1h net, 4 dims)
- z [0.9, 1.05, -0.11, 0.14], predicted next [0.64, 0.95, -0.02, 0.03]
- last hour: model error 0.58 vs persistence 0.62 (latent sd); 168h R^2 gain +0.175

## News
- alt: wp_btc -8.07, wp_ind -6.19, wp_max -4.86, wp_edits +0.00, hn_story -0.23, hn_comment +0.04, fng -0.80, fng_chg -0.04
- RSS items last 1h 1, 24h 27

## Running coverage (ACI, last 72 / 168 resolved)
- 1h: 80% 0.847 / 0.857; 90% 0.931 / 0.923; 95% 0.972 / 0.964 (n 168)
- 24h: 80% 0.833 / 0.774; 90% 0.917 / 0.863; 95% 0.972 / 0.923 (n 168)

## Scored this run
- nn-20261127T08 24h: r -3.004% (z -0.96), in ACI80 True, pinball 0.6074
- nn-20261128T07 1h: r +0.132% (z +0.46), in ACI80 True, pinball 0.0418

## Alarms
- none

## Changed vs previous line (nn-20261128T07)
- 1h sigma 0.29 -> 0.27%; 24h sigma 2.22 -> 2.23%; 1h alpha80 0.2110 -> 0.2120; field 1h P(jump) 1.7 -> 1.7%
