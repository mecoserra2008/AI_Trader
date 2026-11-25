# BTC live network: bar 2026-11-25 08:00 UTC (line nn-20261125T08)
as_of 2026-11-25T21:10:20Z, newest bar closed 08:00, close 58,706.9. Flags: replay, rolling_day, stale. Research only, no direction call.

## Next hour [11-25T08:00 -> 11-25T09:00], % log return
- PNN median +0.01, 80% [-0.65, +0.67], 90% [-0.93, +0.93], sigma 0.52%
- ACI 80% [-0.67, +0.69] alpha_t 0.1950; 90% [-1.07, +1.07] alpha_t 0.0750; 95% [-1.44, +1.44] alpha_t 0.0400
- H-NN10 vol (bar 07:00): ML 0.508% vs baseline 0.543% (ratio 0.94) - not this bar's line

## Next 24h [11-25T08:00 -> 11-26T08:00], % log return
- PNN median +0.56, 80% [-4.00, +6.18], 90% [-6.47, +8.10], sigma 3.97%
- ACI 80% [-3.88, +6.05] alpha_t 0.2054; 90% [-7.09, +8.72] alpha_t 0.0919; 95% [-10.23, +11.86] alpha_t 0.0406
- 24h net: rolling day, shift 8 h

## v2.1 shadow (ACI gamma 0.02, issue alpha >= nominal/4, jump-cluster floor; cfg 2d4ba801a6363fe9; v21-20261125T08, replay)
- 1h: 80% [-0.60, +0.62]; 90% [-0.98, +0.98]; 95% [-1.21, +1.21]; alpha_t (unclipped) 80/90/95 0.2460/0.0955/0.0528; floor inactive
- 24h: 80% [-3.95, +6.12]; 90% [-7.28, +8.91]; 95% [-10.43, +12.06]; alpha_t 0.1992/0.0863/0.0373
- running coverage (issued, 72 / 168): 1h 80% 0.861/0.821, 90% 0.917/0.905, 95% 0.972/0.952 | 24h 80% 0.819/0.738, 90% 0.931/0.845, 95% 0.986/0.923
- v1 vs v2 since 11-13T20:00 (same bars; v2 mostly replay): 1h n 138: cov80 0.761 vs 0.783, cov95 0.935 vs 0.949, band pinball 0.0930 vs 0.0930; 24h n 127: cov80 0.701 vs 0.717, cov95 0.898 vs 0.906, band pinball 0.4950 vs 0.4995

## Regime and players (information only, changes no forecast; as_of 2026-11-25T21:10Z)
| | v0 cell | 24h px | funding (90d pct) | OI 24h / 72h | liq 4h long/short | liq 24h long/short | DVOL (90d pct) |
|---|---|---|---|---|---|---|---|
| BTC | R9 liquidation | -2.8% | +0.0003% (p14) | +1.2% / +1.1% | 0k / 0k | 3.4M / 0.6M | 54.4 (p98) |
| ETH | R9 liquidation | -3.4% | +0.0022% (p36) | +2.0% / -1.7% | 0k / 0k | 2.6M / 0.2M | 67.0 (p97) |
| SOL | R6 whipsaw | -5.3% | -0.0078% (p9) | -0.6% / +0.9% | 0k / 0k | 0.4M / 39k | no index |
- Pattern: BTC long flush, funding normal, OI rebuilding after a drop while price fell (new shorts), implied vol high; ETH long flush, funding normal, OI rebuilding after a drop while price fell (new shorts), implied vol high; SOL long flush, funding negative, OI flat
- Long/short ratio omitted: lake ls_ratio last known 2026-11-07T20:00 (one-off backfill of 11-07; the live updater has no ls_ratio stream)

## Cross-section v3 (context only, not a signal: R-XS-2/R-XS-3; xs3-20261125T08, 23/23 tokens)
- 4h relative median vs EW basket, bp [80% ACI]: top AVAX +3 [-66, +67], ETH +2 [-69, +69], SOL +2 [-73, +73], XLM +2 [-78, +77], BTC +2 [-77, +75]
- bottom HYPE -1 [-120, +118], TRX -2 [-213, +204], BCH -4 [-103, +102], ATOM -7 [-133, +130], ETC -8 [-91, +87]
- mostly low vol: rank corr with -lsig2 4h -0.12 / 24h +0.18; top 5 shared with the low-vol top 5: 1/5 (TRX, BNB, BTC, XRP, LINK)
- BTC 1h |r|: v3 q50 0.35%, q90 0.96% (log-HAR-t q90 0.88%) vs v1 90% half-width 0.93% (ACI90 1.07%)
- running (scored lines): 4h IC v3 +0.105 vs low-vol +0.048 (n 95); 24h IC v3 +0.244 vs low-vol +0.212 (n 87); 1h vol pinball v3/HAR 0.957 (n 98)
- 24h ACI band under-covers (holdout 0.749 at 80%); the decision sessions do not read this block; flags: replay, stale

## Field card (btc_field_2026-11-13.json, regime R9)
- 1h: 90% [-0.83, +0.85] med +0.01; touch +1/-1 6/5%, +2/-2 1/1%, +3/-3 0/0%; P(jump) 2%
- 4h: 90% [-1.74, +1.81] med +0.05; touch +1/-1 31/29%, +2/-2 7/6%, +3/-3 2/2%; P(jump) 6%
- 24h: 90% [-5.33, +5.36] med +0.36; touch +1/-1 74/68%, +2/-2 50/45%, +3/-3 31/28%; P(jump) 32%
- variance drivers: m24*dir +0.16, x2 -0.13, x1 +0.08; jump drivers: m24*dir +0.09

## Latent (1h net, 4 dims)
- z [-0.11, 0.7, -0.17, -0.02], predicted next [-0.07, 0.76, -0.02, -0.07]
- last hour: model error 0.55 vs persistence 0.91 (latent sd); 168h R^2 gain +0.190

## News
- alt: wp_btc +0.17, wp_ind +0.51, wp_max +2.90, wp_edits +0.00, hn_story +0.14, hn_comment +0.36, fng -0.74, fng_chg +0.04
- RSS items last 1h 0, 24h 16 (polling stale)

## Running coverage (ACI, last 72 / 168 resolved)
- 1h: 80% 0.861 / 0.821; 90% 0.903 / 0.893; 95% 0.958 / 0.952 (n 168)
- 24h: 80% 0.792 / 0.726; 90% 0.917 / 0.839; 95% 0.972 / 0.917 (n 168)

## Alarms
- stale data: newest bar closed 790 min before as_of
- coverage 24h ACI 90% over last 168: 0.839 vs 0.90

## Changed vs previous line (nn-20261125T07)
- 1h sigma 0.40 -> 0.52%; 24h sigma 3.67 -> 3.97%; 1h alpha80 0.1990 -> 0.1950; field 1h P(jump) 1.6 -> 1.7%
- new alarms: 1
