# BTC live network: bar 2026-11-17 08:00 UTC (line nn-20261117T08)
as_of 2026-11-17T08:03:06Z, newest bar closed 08:00, close 64,700.0. Flags: rolling_day. Research only, no direction call.

## Next hour [11-17T08:00 -> 11-17T09:00], % log return
- PNN median +0.00, 80% [-0.63, +0.61], 90% [-0.87, +0.83], sigma 0.48%
- ACI 80% [-0.68, +0.66] alpha_t 0.1730; 90% [-0.97, +0.94] alpha_t 0.0790; 95% [-1.37, +1.34] alpha_t 0.0370
- H-NN10 vol (bar 07:00): ML 0.515% vs baseline 0.509% (ratio 1.01) - not this bar's line

## Next 24h [11-17T08:00 -> 11-18T08:00], % log return
- PNN median +0.37, 80% [-3.45, +4.66], 90% [-5.44, +6.31], sigma 3.16%
- ACI 80% [-3.35, +4.55] alpha_t 0.2058; 90% [-5.84, +6.70] alpha_t 0.0917; 95% [-8.27, +9.13] alpha_t 0.0403
- 24h net: rolling day, shift 8 h

## v2.1 shadow (ACI gamma 0.02, issue alpha >= nominal/4, jump-cluster floor; cfg 2d4ba801a6363fe9; v21-20261117T08)
- 1h: 80% [-0.67, +0.65]; 90% [-1.00, +0.97]; 95% [-1.31, +1.27]; alpha_t (unclipped) 80/90/95 0.1780/0.0715/0.0408; floor inactive
- 24h: 80% [-3.43, +4.63]; 90% [-6.06, +6.92]; 95% [-8.67, +9.53]; alpha_t 0.1990/0.0849/0.0354
- running coverage (issued, 72 / 168): 1h 80% 0.819/0.804, 90% 0.931/0.887, 95% 0.972/0.958 | 24h 80% 0.639/0.726, 90% 0.736/0.809, 95% 0.833/0.887
- v1 vs v2 since 11-13T20:00 (same bars; v2 mostly replay): 1h n 74: cov80 0.743 vs 0.797, cov95 0.919 vs 0.946, band pinball 0.1004 vs 0.0991; 24h n 61: cov80 0.623 vs 0.623, cov95 0.820 vs 0.820, band pinball 0.5668 vs 0.5725

## Regime and players (information only, changes no forecast; as_of 2026-11-17T08:03Z)
| | v0 cell | 24h px | funding (90d pct) | OI 24h / 72h | liq 4h long/short | liq 24h long/short | DVOL (90d pct) |
|---|---|---|---|---|---|---|---|
| BTC | R9 liquidation | -0.4% | +0.0067% (p68) | +2.6% / +3.5% | 0.8M / 0.2M | 2.0M / 0.9M | 43.6 (p97) |
| ETH | R9 liquidation | +0.6% | -0.0012% (p11) | +4.3% / +3.6% | 0.3M / 0.1M | 0.5M / 0.3M | 57.5 (p93) |
| SOL | R9 liquidation | +2.2% | -0.0082% (p7) | -0.1% / -4.9% | 0.1M / 32k | 0.2M / 0.1M | no index |
- Pattern: BTC long flush, funding normal, OI rising into the drop (new shorts), long-liquidation burst (4h), implied vol high; ETH two-sided liquidations, funding negative, OI rebuilding, long-liquidation burst (4h), implied vol high; SOL two-sided liquidations, funding negative, OI flat, long-liquidation burst (4h)
- Long/short ratio omitted: lake ls_ratio last known 2026-11-07T20:00 (one-off backfill of 11-07; the live updater has no ls_ratio stream)

## Cross-section v3 (context only, not a signal: R-XS-2/R-XS-3; xs3-20261117T08, 23/23 tokens)
- 4h relative median vs EW basket, bp [80% ACI]: top ETC -0 [-42, +42], BTC -1 [-77, +77], ETH -1 [-63, +63], XRP -1 [-75, +75], LTC -1 [-52, +52]
- bottom ENA -7 [-129, +136], HYPE -8 [-141, +142], FIL -9 [-110, +110], AAVE -11 [-107, +107], NEAR -12 [-150, +152]
- mostly low vol: rank corr with -lsig2 4h +0.73 / 24h +0.80; top 5 shared with the low-vol top 5: 3/5 (TRX, BNB, BTC, LTC, ETC)
- BTC 1h |r|: v3 q50 0.24%, q90 0.66% (log-HAR-t q90 0.80%) vs v1 90% half-width 0.85% (ACI90 0.96%)
- running (scored lines): 4h IC v3 +0.025 vs low-vol -0.035 (n 30); 24h IC v3 -0.026 vs low-vol -0.001 (n 20); 1h vol pinball v3/HAR 0.925 (n 33)
- 24h ACI band under-covers (holdout 0.749 at 80%); the decision sessions do not read this block

## Field card (btc_field_2026-11-13.json, regime R9)
- 1h: 90% [-0.73, +0.74] med +0.00; touch +1/-1 4/3%, +2/-2 0/0%, +3/-3 0/0%; P(jump) 2%
- 4h: 90% [-1.39, +1.45] med +0.04; touch +1/-1 21/19%, +2/-2 3/3%, +3/-3 1/1%; P(jump) 6%
- 24h: 90% [-4.53, +4.32] med +0.34; touch +1/-1 69/63%, +2/-2 43/38%, +3/-3 23/21%; P(jump) 31%
- variance drivers: x2 -0.12, m24*dir +0.04, x1 -0.02; jump drivers: m24*dir +0.02

## Latent (1h net, 4 dims)
- z [0.09, 0.66, -0.45, -0.11], predicted next [0.12, 0.76, -0.27, -0.12]
- last hour: model error 0.51 vs persistence 0.60 (latent sd); 168h R^2 gain +0.140

## News
- alt: wp_btc -8.02, wp_ind -6.26, wp_max -4.92, wp_edits +0.00, hn_story +0.14, hn_comment +0.63, fng -0.76, fng_chg +0.04
- RSS items last 1h 3, 24h 68

## Running coverage (ACI, last 72 / 168 resolved)
- 1h: 80% 0.764 / 0.768; 90% 0.889 / 0.869; 95% 0.958 / 0.941 (n 168)
- 24h: 80% 0.639 / 0.726; 90% 0.736 / 0.809; 95% 0.833 / 0.887 (n 168)

## Scored this run
- nn-20261116T08 24h: r -0.399% (z -0.14), in ACI80 True, pinball 0.3542
- nn-20261117T07 1h: r +0.281% (z +0.75), in ACI80 True, pinball 0.0666

## Alarms (5)
- coverage (band level/window cov): 24h 80%/72 0.639; 24h 90%/72 0.736; 24h 90%/168 0.809; 24h 95%/72 0.833; 24h 95%/168 0.887

## Changed vs previous line (nn-20261117T07)
- 1h sigma 0.38 -> 0.48%; 24h sigma 3.15 -> 3.16%; 1h alpha80 0.1720 -> 0.1730; field 1h P(jump) 1.6 -> 1.6%
