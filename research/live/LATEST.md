# BTC live network: bar 2026-11-28 15:00 UTC (line nn-20261128T15)
as_of 2026-11-28T15:03:06Z, newest bar closed 15:00, close 59,288.3. Flags: rolling_day. Research only, no direction call.

## Next hour [11-28T15:00 -> 11-28T16:00], % log return
- PNN median -0.02, 80% [-0.46, +0.38], 90% [-0.65, +0.56], sigma 0.33%
- ACI 80% [-0.45, +0.38] alpha_t 0.2090; 90% [-0.73, +0.63] alpha_t 0.0895; 95% [-0.88, +0.79] alpha_t 0.0498
- H-NN10 vol (bar 15:00): ML 0.367% vs baseline 0.315% (ratio 1.17)

## Next 24h [11-28T15:00 -> 11-29T15:00], % log return
- PNN median +0.10, 80% [-2.10, +2.96], 90% [-3.40, +4.23], sigma 1.98%
- ACI 80% [-1.94, +2.81] alpha_t 0.2070; 90% [-3.80, +4.63] alpha_t 0.0927; 95% [-6.52, +7.35] alpha_t 0.0410
- 24h net: rolling day, shift 15 h

## v2.1 shadow (ACI gamma 0.02, issue alpha >= nominal/4, jump-cluster floor; cfg 2d4ba801a6363fe9; v21-20261128T15)
- 1h: 80% [-0.42, +0.35]; 90% [-0.60, +0.51]; 95% [-0.77, +0.67]; alpha_t (unclipped) 80/90/95 0.2420/0.1335/0.0718; floor inactive
- 24h: 80% [-1.95, +2.81]; 90% [-3.94, +4.77]; 95% [-6.63, +7.46]; alpha_t 0.2058/0.0896/0.0390
- running coverage (issued, 72 / 168): 1h 80% 0.806/0.839, 90% 0.931/0.929, 95% 0.958/0.970 | 24h 80% 0.833/0.786, 90% 0.931/0.869, 95% 0.986/0.929
- v1 vs v2 since 11-13T20:00 (same bars; v2 mostly replay): 1h n 184: cov80 0.777 vs 0.777, cov95 0.951 vs 0.957, band pinball 0.0819 vs 0.0815; 24h n 167: cov80 0.772 vs 0.784, cov95 0.922 vs 0.928, band pinball 0.4619 vs 0.4669

## Regime and players (information only, changes no forecast; as_of 2026-11-28T15:03Z)
| | v0 cell | 24h px | funding (90d pct) | OI 24h / 72h | liq 4h long/short | liq 24h long/short | DVOL (90d pct) |
|---|---|---|---|---|---|---|---|
| BTC | R9 liquidation | +0.2% | +0.0006% (p15) | -0.9% / -3.0% | 0.5M / 65k | 3.0M / 0.7M | 54.0 (p97) |
| ETH | R9 liquidation | -0.4% | -0.0017% (p13) | -0.8% / -0.2% | 0.9M / 0.1M | 1.8M / 0.2M | 66.5 (p96) |
| SOL | R5 balanced range | -0.5% | +0.0008% (p47) | +0.4% / -13.8% | 0.1M / 5k | 0.4M / 15k | no index |
- Pattern: BTC long flush, funding normal, OI flat, long-liquidation burst (4h), implied vol high; ETH long flush, funding negative, OI flat, long-liquidation burst (4h), implied vol high; SOL long flush, funding normal, OI flat, long-liquidation burst (4h)
- Long/short accounts: BTC 0.72 long (1h), ETH 0.77 long (4h), SOL 0.79 long (4h)

## Cross-section v3 (context only, not a signal: R-XS-2/R-XS-3; xs3-20261128T15, 23/23 tokens)
- 4h relative median vs EW basket, bp [80% ACI]: top ETH +0 [-47, +50], AVAX +0 [-49, +52], XLM -0 [-55, +57], SOL -1 [-45, +48], DOT -1 [-58, +61]
- bottom ENA -7 [-100, +106], NEAR -7 [-101, +107], UNI -8 [-90, +94], HYPE -9 [-116, +117], LTC -10 [-94, +108]
- mostly low vol: rank corr with -lsig2 4h +0.68 / 24h +0.83; top 5 shared with the low-vol top 5: 1/5 (TRX, BNB, BTC, ETH, ETC)
- BTC 1h |r|: v3 q50 0.17%, q90 0.45% (log-HAR-t q90 0.48%) vs v1 90% half-width 0.61% (ACI90 0.68%)
- running (scored lines): 4h IC v3 +0.069 vs low-vol +0.036 (n 142); 24h IC v3 +0.112 vs low-vol +0.079 (n 128); 1h vol pinball v3/HAR 0.967 (n 145)
- 24h ACI band under-covers (holdout 0.749 at 80%); the decision sessions do not read this block

## Field card (btc_field_2026-11-13.json, regime R9)
- 1h: 90% [-0.46, +0.47] med +0.00; touch +1/-1 1/1%, +2/-2 0/0%, +3/-3 0/0%; P(jump) 2%
- 4h: 90% [-1.05, +1.12] med +0.02; touch +1/-1 12/10%, +2/-2 1/1%, +3/-3 0/0%; P(jump) 6%
- 24h: 90% [-3.05, +3.14] med +0.15; touch +1/-1 57/51%, +2/-2 26/22%, +3/-3 10/9%; P(jump) 31%
- variance drivers: s -0.58, x1 +0.16, x2 -0.11; jump drivers: m24*dir -0.00

## Latent (1h net, 4 dims)
- z [0.57, 0.54, 0.38, 0.07], predicted next [0.41, 0.83, 0.39, 0.03]
- last hour: model error 0.91 vs persistence 0.90 (latent sd); 168h R^2 gain +0.170

## News
- alt: wp_btc -8.07, wp_ind -6.19, wp_max -4.86, wp_edits +0.00, hn_story -0.23, hn_comment +0.04, fng -0.80, fng_chg -0.04
- RSS items last 1h 1, 24h 31

## Running coverage (ACI, last 72 / 168 resolved)
- 1h: 80% 0.847 / 0.857; 90% 0.944 / 0.929; 95% 0.972 / 0.970 (n 168)
- 24h: 80% 0.833 / 0.774; 90% 0.917 / 0.863; 95% 0.972 / 0.923 (n 168)

## Scored this run
- nn-20261128T14 1h: r +0.363% (z +0.98), in ACI80 True, pinball 0.0956

## Alarms
- none

## Changed vs previous line (nn-20261128T14)
- 1h sigma 0.37 -> 0.33%; 24h sigma 2.06 -> 1.98%; 1h alpha80 0.2080 -> 0.2090; field 1h P(jump) 1.8 -> 1.5%
