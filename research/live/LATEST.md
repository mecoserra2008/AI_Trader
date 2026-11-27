# BTC live network: bar 2026-11-27 08:00 UTC (line nn-20261127T08)
as_of 2026-11-27T18:10:50Z, newest bar closed 08:00, close 61,621.0. Flags: replay, rolling_day, stale. Research only, no direction call.

## Next hour [11-27T08:00 -> 11-27T09:00], % log return
- PNN median +0.00, 80% [-0.53, +0.52], 90% [-0.74, +0.72], sigma 0.41%
- ACI 80% [-0.53, +0.52] alpha_t 0.2080; 90% [-0.82, +0.80] alpha_t 0.0840; 95% [-1.02, +1.00] alpha_t 0.0470
- H-NN10 vol (bar 07:00): ML 0.466% vs baseline 0.403% (ratio 1.15) - not this bar's line

## Next 24h [11-27T08:00 -> 11-28T08:00], % log return
- PNN median -0.54, 80% [-4.25, +3.74], 90% [-6.04, +5.30], sigma 3.12%
- ACI 80% [-4.09, +3.59] alpha_t 0.2064; 90% [-6.51, +5.76] alpha_t 0.0924; 95% [-9.57, +8.83] alpha_t 0.0409
- 24h net: rolling day, shift 8 h

## v2.1 shadow (ACI gamma 0.02, issue alpha >= nominal/4, jump-cluster floor; cfg 2d4ba801a6363fe9; v21-20261127T08, replay)
- 1h: 80% [-0.48, +0.47]; 90% [-0.69, +0.67]; 95% [-0.92, +0.90]; alpha_t (unclipped) 80/90/95 0.2580/0.1315/0.0608; floor inactive
- 24h: 80% [-4.12, +3.61]; 90% [-6.65, +5.91]; 95% [-9.72, +8.97]; alpha_t 0.2032/0.0883/0.0383
- running coverage (issued, 72 / 168): 1h 80% 0.806/0.833, 90% 0.931/0.923, 95% 0.972/0.952 | 24h 80% 0.833/0.756, 90% 0.931/0.851, 95% 0.986/0.923
- v1 vs v2 since 11-13T20:00 (same bars; v2 mostly replay): 1h n 167: cov80 0.778 vs 0.784, cov95 0.946 vs 0.952, band pinball 0.0866 vs 0.0863; 24h n 156: cov80 0.756 vs 0.769, cov95 0.917 vs 0.923, band pinball 0.4709 vs 0.4760

## Regime and players (information only, changes no forecast; as_of 2026-11-27T18:10Z)
| | v0 cell | 24h px | funding (90d pct) | OI 24h / 72h | liq 4h long/short | liq 24h long/short | DVOL (90d pct) |
|---|---|---|---|---|---|---|---|
| BTC | R6 whipsaw | -1.0% | +0.0008% (p16) | -4.8% / -0.7% | 0k / 0k | 0.3M / 5.0M | 51.9 (p94) |
| ETH | R6 whipsaw | -1.3% | +0.0041% (p57) | +0.9% / +3.3% | 0k / 0k | 0.3M / 2.1M | 65.1 (p93) |
| SOL | R6 whipsaw | +1.4% | +0.0061% (p72) | -0.8% / -6.3% | 0k / 0k | 30k / 0.6M | no index |
- Pattern: BTC short squeeze, funding normal, OI falling with price (deleveraging), implied vol high; ETH short squeeze, funding normal, OI flat, implied vol high; SOL short squeeze, funding normal, OI flat
- Long/short ratio omitted: lake ls_ratio last known 2026-11-07T20:00 (one-off backfill of 11-07; the live updater has no ls_ratio stream)

## Cross-section v3 (context only, not a signal: R-XS-2/R-XS-3; xs3-20261127T18, 23/23 tokens)
- 4h relative median vs EW basket, bp [80% ACI]: top TRX -1 [-195, +190], BTC -2 [-75, +72], ETH -3 [-69, +68], AVAX -3 [-63, +65], XLM -4 [-70, +69]
- bottom HYPE -12 [-160, +166], NEAR -12 [-122, +124], ATOM -13 [-156, +153], ENA -14 [-140, +140], UNI -15 [-131, +139]
- mostly low vol: rank corr with -lsig2 4h +0.84 / 24h +0.86; top 5 shared with the low-vol top 5: 3/5 (TRX, BNB, BTC, ETC, ETH)
- BTC 1h |r|: v3 q50 0.26%, q90 0.71% (log-HAR-t q90 0.87%) vs v1 90% half-width 0.73% (ACI90 0.81%)
- running (scored lines): 4h IC v3 +0.058 vs low-vol +0.017 (n 128); 24h IC v3 +0.132 vs low-vol +0.096 (n 116); 1h vol pinball v3/HAR 0.964 (n 128)
- 24h ACI band under-covers (holdout 0.749 at 80%); the decision sessions do not read this block

## Field card (btc_field_2026-11-13.json, regime R5)
- 1h: 90% [-0.63, +0.65] med +0.00; touch +1/-1 2/2%, +2/-2 0/0%, +3/-3 0/0%; P(jump) 2%
- 4h: 90% [-1.27, +1.36] med +0.03; touch +1/-1 19/16%, +2/-2 3/2%, +3/-3 1/1%; P(jump) 6%
- 24h: 90% [-3.35, +3.58] med +0.13; touch +1/-1 60/56%, +2/-2 31/27%, +3/-3 15/12%; P(jump) 31%
- variance drivers: x1 +0.20, x2 -0.12, m24*dir +0.03; jump drivers: m24*dir +0.01

## Latent (1h net, 4 dims)
- z [-0.35, 0.59, 0.17, 0.06], predicted next [-0.23, 0.8, 0.29, 0.0]
- last hour: model error 0.42 vs persistence 0.32 (latent sd); 168h R^2 gain +0.174

## News
- alt: wp_btc -8.11, wp_ind -6.21, wp_max -4.95, wp_edits +1.10, hn_story +0.00, hn_comment +0.23, fng -0.76, fng_chg +0.02
- RSS items last 1h 0, 24h 19 (polling stale)

## Running coverage (ACI, last 72 / 168 resolved)
- 1h: 80% 0.833 / 0.839; 90% 0.931 / 0.911; 95% 0.986 / 0.952 (n 168)
- 24h: 80% 0.806 / 0.744; 90% 0.917 / 0.845; 95% 0.972 / 0.917 (n 168)

## Alarms
- stale data: newest bar closed 611 min before as_of

## Changed vs previous line (nn-20261127T07)
- 1h sigma 0.40 -> 0.41%; 24h sigma 3.16 -> 3.12%; 1h alpha80 0.2070 -> 0.2080; field 1h P(jump) 1.6 -> 1.6%
- new alarms: 1
