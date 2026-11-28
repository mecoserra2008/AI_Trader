# BTC live network: bar 2026-11-28 04:00 UTC (line nn-20261128T04)
as_of 2026-11-28T04:03:31Z, newest bar closed 04:00, close 59,998.1. Flags: rolling_day. Research only, no direction call.

## Next hour [11-28T04:00 -> 11-28T05:00], % log return
- PNN median -0.02, 80% [-0.37, +0.30], 90% [-0.53, +0.45], sigma 0.26%
- ACI 80% [-0.37, +0.29] alpha_t 0.2130; 90% [-0.63, +0.54] alpha_t 0.0840; 95% [-0.85, +0.76] alpha_t 0.0470
- H-NN10 vol (bar 03:00): ML 0.368% vs baseline 0.319% (ratio 1.16) - not this bar's line

## Next 24h [11-28T04:00 -> 11-29T04:00], % log return
- PNN median +0.50, 80% [-2.00, +3.80], 90% [-3.59, +5.42], sigma 2.26%
- ACI 80% [-1.82, +3.62] alpha_t 0.2069; 90% [-4.04, +5.87] alpha_t 0.0926; 95% [-7.06, +8.88] alpha_t 0.0410
- 24h net: rolling day, shift 4 h

## v2.1 shadow (ACI gamma 0.02, issue alpha >= nominal/4, jump-cluster floor; cfg 2d4ba801a6363fe9; v21-20261128T04)
- 1h: 80% [-0.32, +0.25]; 90% [-0.52, +0.43]; 95% [-0.73, +0.64]; alpha_t (unclipped) 80/90/95 0.2580/0.1115/0.0608; floor inactive
- 24h: 80% [-1.84, +3.64]; 90% [-4.19, +6.02]; 95% [-7.18, +9.01]; alpha_t 0.2052/0.0893/0.0388
- running coverage (issued, 72 / 168): 1h 80% 0.806/0.839, 90% 0.903/0.917, 95% 0.958/0.958 | 24h 80% 0.833/0.780, 90% 0.931/0.869, 95% 0.986/0.929
- v1 vs v2 since 11-13T20:00 (same bars; v2 mostly replay): 1h n 173: cov80 0.780 vs 0.780, cov95 0.948 vs 0.954, band pinball 0.0848 vs 0.0845; 24h n 163: cov80 0.767 vs 0.779, cov95 0.920 vs 0.926, band pinball 0.4655 vs 0.4706

## Regime and players (information only, changes no forecast; as_of 2026-11-28T04:03Z)
| | v0 cell | 24h px | funding (90d pct) | OI 24h / 72h | liq 4h long/short | liq 24h long/short | DVOL (90d pct) |
|---|---|---|---|---|---|---|---|
| BTC | R6 whipsaw | -2.3% | +0.0046% (p50) | -3.3% / -0.6% | 70k / 0.4M | 0.3M / 1.1M | 52.2 (p95) |
| ETH | R6 whipsaw | -3.7% | +0.0005% (p24) | +1.7% / +2.9% | 44k / 63k | 0.4M / 0.5M | 64.8 (p93) |
| SOL | R6 whipsaw | -1.4% | +0.0087% (p78) | -1.4% / -4.8% | 8k / 9k | 17k / 0.1M | no index |
- Pattern: BTC short squeeze, funding normal, OI falling with price (deleveraging), short-liquidation burst (4h), implied vol high; ETH two-sided liquidations, funding normal, OI rising into the drop (new shorts), implied vol high; SOL short squeeze, funding normal, OI falling with price (deleveraging)
- Long/short ratio omitted: lake ls_ratio last known 2026-11-07T20:00 (one-off backfill of 11-07; the live updater has no ls_ratio stream)

## Cross-section v3 (context only, not a signal: R-XS-2/R-XS-3; xs3-20261128T04, 23/23 tokens)
- 4h relative median vs EW basket, bp [80% ACI]: top TRX -0 [-137, +134], BTC -2 [-58, +55], XLM -2 [-55, +56], FIL -2 [-84, +85], ETH -2 [-55, +54]
- bottom DOGE -7 [-61, +63], LTC -8 [-74, +77], ENA -8 [-103, +108], HYPE -10 [-125, +125], NEAR -13 [-144, +159]
- mostly low vol: rank corr with -lsig2 4h +0.77 / 24h +0.79; top 5 shared with the low-vol top 5: 3/5 (TRX, BNB, BTC, ETC, ETH)
- BTC 1h |r|: v3 q50 0.17%, q90 0.45% (log-HAR-t q90 0.48%) vs v1 90% half-width 0.49% (ACI90 0.58%)
- running (scored lines): 4h IC v3 +0.058 vs low-vol +0.017 (n 131); 24h IC v3 +0.106 vs low-vol +0.072 (n 124); 1h vol pinball v3/HAR 0.963 (n 134)
- 24h ACI band under-covers (holdout 0.749 at 80%); the decision sessions do not read this block

## Field card (btc_field_2026-11-13.json, regime R6)
- 1h: 90% [-0.40, +0.41] med +0.00; touch +1/-1 1/1%, +2/-2 0/0%, +3/-3 0/0%; P(jump) 2%
- 4h: 90% [-0.87, +0.93] med +0.02; touch +1/-1 7/6%, +2/-2 1/1%, +3/-3 0/0%; P(jump) 6%
- 24h: 90% [-2.71, +2.73] med +0.14; touch +1/-1 52/46%, +2/-2 20/18%, +3/-3 7/7%; P(jump) 32%
- variance drivers: s -0.96, x1 +0.15, x2 -0.11; jump drivers: m24*dir +0.05

## Latent (1h net, 4 dims)
- z [1.09, 0.76, 0.18, -0.42], predicted next [0.89, 0.74, 0.0, -0.26]
- last hour: model error 0.61 vs persistence 0.62 (latent sd); 168h R^2 gain +0.181

## News
- alt: wp_btc -8.07, wp_ind -6.19, wp_max -4.86, wp_edits +0.00, hn_story -0.23, hn_comment +0.04, fng -0.80, fng_chg -0.04
- RSS items last 1h 0, 24h 33 (polling stale)

## Running coverage (ACI, last 72 / 168 resolved)
- 1h: 80% 0.847 / 0.857; 90% 0.917 / 0.917; 95% 0.972 / 0.958 (n 168)
- 24h: 80% 0.833 / 0.768; 90% 0.917 / 0.863; 95% 0.972 / 0.923 (n 168)

## Scored this run
- nn-20261127T04 24h: r -2.289% (z -0.67), in ACI80 True, pinball 0.4928
- nn-20261128T03 1h: r +0.104% (z +0.33), in ACI80 True, pinball 0.0418

## Alarms
- none

## Changed vs previous line (nn-20261128T03)
- 1h sigma 0.31 -> 0.26%; 24h sigma 2.20 -> 2.26%; 1h alpha80 0.2120 -> 0.2130; field 1h P(jump) 1.6 -> 1.6%
