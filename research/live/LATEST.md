# BTC live network: bar 2026-11-19 08:00 UTC (line nn-20261119T08)
as_of 2026-11-19T08:03:03Z, newest bar closed 08:00, close 63,714.4. Flags: rolling_day. Research only, no direction call.

## Next hour [11-19T08:00 -> 11-19T09:00], % log return
- PNN median -0.04, 80% [-0.61, +0.55], 90% [-0.86, +0.78], sigma 0.46%
- ACI 80% [-0.65, +0.59] alpha_t 0.1810; 90% [-0.95, +0.87] alpha_t 0.0830; 95% [-1.31, +1.23] alpha_t 0.0390
- H-NN10 vol (bar 07:00): ML 0.564% vs baseline 0.503% (ratio 1.12) - not this bar's line

## Next 24h [11-19T08:00 -> 11-20T08:00], % log return
- PNN median -0.01, 80% [-4.58, +4.55], 90% [-6.61, +6.45], sigma 3.56%
- ACI 80% [-4.46, +4.43] alpha_t 0.2063; 90% [-7.00, +6.85] alpha_t 0.0923; 95% [-9.69, +9.53] alpha_t 0.0406
- 24h net: rolling day, shift 8 h

## v2.1 shadow (ACI gamma 0.02, issue alpha >= nominal/4, jump-cluster floor; cfg 2d4ba801a6363fe9; v21-20261119T08)
- 1h: 80% [-0.67, +0.61]; 90% [-0.90, +0.82]; 95% [-1.16, +1.08]; alpha_t (unclipped) 80/90/95 0.1700/0.0875/0.0488; floor inactive
- 24h: 80% [-4.48, +4.45]; 90% [-7.17, +7.02]; 95% [-9.96, +9.81]; alpha_t 0.2027/0.0872/0.0365
- running coverage (issued, 72 / 168): 1h 80% 0.806/0.809, 90% 0.917/0.893, 95% 0.972/0.946 | 24h 80% 0.889/0.768, 90% 0.986/0.851, 95% 1.000/0.911
- v1 vs v2 since 11-13T20:00 (same bars; v2 mostly replay): 1h n 98: cov80 0.755 vs 0.786, cov95 0.929 vs 0.949, band pinball 0.0911 vs 0.0906; 24h n 88: cov80 0.705 vs 0.727, cov95 0.875 vs 0.875, band pinball 0.4873 vs 0.4939

## Regime and players (information only, changes no forecast; as_of 2026-11-19T08:03Z)
| | v0 cell | 24h px | funding (90d pct) | OI 24h / 72h | liq 4h long/short | liq 24h long/short | DVOL (90d pct) |
|---|---|---|---|---|---|---|---|
| BTC | R9 liquidation | -1.8% | +0.0033% (p33) | +0.2% / -0.5% | 30k / 0.7M | 2.1M / 1.8M | 45.9 (p99) |
| ETH | R9 liquidation | -0.7% | +0.0019% (p32) | +0.6% / +2.0% | 2k / 0.4M | 82k / 0.6M | 58.5 (p96) |
| SOL | R6 whipsaw | -0.3% | -0.0009% (p34) | +0.9% / -2.7% | 20k / 0.1M | 94k / 0.2M | no index |
- Pattern: BTC two-sided liquidations, funding normal, OI rebuilding after a drop while price fell (new shorts), short-liquidation burst (4h), implied vol high; ETH short squeeze, funding normal, OI flat, short-liquidation burst (4h), implied vol high; SOL short squeeze, funding negative, OI flat, short-liquidation burst (4h)
- Long/short ratio omitted: lake ls_ratio last known 2026-11-07T20:00 (one-off backfill of 11-07; the live updater has no ls_ratio stream)

## Cross-section v3 (context only, not a signal: R-XS-2/R-XS-3; xs3-20261119T08, 23/23 tokens)
- 4h relative median vs EW basket, bp [80% ACI]: top ETH +2 [-58, +60], TRX +1 [-179, +176], BTC +1 [-67, +66], ETC +1 [-41, +41], DOT -0 [-60, +61]
- bottom AAVE -4 [-88, +87], ADA -5 [-45, +44], FIL -5 [-112, +110], HYPE -6 [-112, +108], NEAR -13 [-168, +173]
- mostly low vol: rank corr with -lsig2 4h +0.71 / 24h +0.76; top 5 shared with the low-vol top 5: 2/5 (TRX, BNB, ATOM, BTC, LTC)
- BTC 1h |r|: v3 q50 0.25%, q90 0.68% (log-HAR-t q90 0.76%) vs v1 90% half-width 0.82% (ACI90 0.91%)
- running (scored lines): 4h IC v3 +0.060 vs low-vol +0.000 (n 55); 24h IC v3 +0.107 vs low-vol +0.102 (n 48); 1h vol pinball v3/HAR 0.939 (n 58)
- 24h ACI band under-covers (holdout 0.749 at 80%); the decision sessions do not read this block

## Field card (btc_field_2026-11-13.json, regime R9)
- 1h: 90% [-0.75, +0.76] med +0.00; touch +1/-1 4/4%, +2/-2 0/1%, +3/-3 0/0%; P(jump) 2%
- 4h: 90% [-1.53, +1.60] med +0.04; touch +1/-1 25/23%, +2/-2 4/4%, +3/-3 1/1%; P(jump) 6%
- 24h: 90% [-4.87, +4.75] med +0.36; touch +1/-1 72/65%, +2/-2 47/41%, +3/-3 27/25%; P(jump) 30%
- variance drivers: s -0.17, m24*dir +0.13, x2 -0.11; jump drivers: m24*dir +0.07

## Latent (1h net, 4 dims)
- z [0.18, 0.74, -0.35, 0.14], predicted next [0.15, 0.77, -0.1, -0.04]
- last hour: model error 0.70 vs persistence 0.74 (latent sd); 168h R^2 gain +0.161

## News
- alt: wp_btc -8.02, wp_ind -6.25, wp_max -4.91, wp_edits +0.00, hn_story +0.00, hn_comment -0.31, fng -0.80, fng_chg -0.02
- RSS items last 1h 0, 24h 59

## Running coverage (ACI, last 72 / 168 resolved)
- 1h: 80% 0.833 / 0.786; 90% 0.931 / 0.875; 95% 0.972 / 0.929 (n 168)
- 24h: 80% 0.861 / 0.756; 90% 0.986 / 0.851; 95% 1.000 / 0.911 (n 168)

## Scored this run
- nn-20261118T08 24h: r -1.773% (z -0.60), in ACI80 True, pinball 0.5391
- nn-20261119T07 1h: r -0.015% (z -0.03), in ACI80 True, pinball 0.0526

## Alarms
- none

## Changed vs previous line (nn-20261119T07)
- 1h sigma 0.47 -> 0.46%; 24h sigma 3.44 -> 3.56%; 1h alpha80 0.1800 -> 0.1810; field 1h P(jump) 1.6 -> 1.6%
