# BTC live network: bar 2026-11-16 00:00 UTC (line nn-20261116T00)
as_of 2026-11-16T00:12:46Z, newest bar closed 00:00, close 65,317.0. Flags: none. Research only, no direction call.

## Next hour [11-16T00:00 -> 11-16T01:00], % log return
- PNN median -0.02, 80% [-1.15, +1.08], 90% [-1.60, +1.59], sigma 0.87%
- ACI 80% [-1.24, +1.16] alpha_t 0.1610; 90% [-1.84, +1.83] alpha_t 0.0680; 95% [-2.36, +2.35] alpha_t 0.0290
- H-NN10 vol (bar 23:00): ML 0.781% vs baseline 1.256% (ratio 0.62) - not this bar's line

## Next 24h [11-16T00:00 -> 11-17T00:00], % log return
- PNN median +0.38, 80% [-3.33, +4.20], 90% [-5.15, +5.85], sigma 2.94%
- ACI 80% [-3.26, +4.13] alpha_t 0.2046; 90% [-5.61, +6.31] alpha_t 0.0911; 95% [-7.91, +8.61] alpha_t 0.0400
- 24h net: rolling day, shift 0 h (= training construction)

## v2.1 shadow (ACI gamma 0.02, issue alpha >= nominal/4, jump-cluster floor; cfg 2d4ba801a6363fe9; v21-20261116T00)
- 1h: 80% [-1.33, +1.25]; 90% [-2.05, +2.04]; 95% [-2.95, +2.95]; alpha_t (unclipped) 80/90/95 0.1300/0.0475/0.0088; floor inactive
- 24h: 80% [-3.40, +4.27]; 90% [-5.85, +6.55]; 95% [-8.34, +9.04]; alpha_t 0.1945/0.0823/0.0340
- running coverage (issued, 72 / 168): 1h 80% 0.778/0.786, 90% 0.861/0.887, 95% 0.917/0.952 | 24h 80% 0.583/0.714, 90% 0.708/0.804, 95% 0.833/0.887
- v1 vs v2 since 11-13T20:00 (same bars; v2 mostly replay): 1h n 52: cov80 0.692 vs 0.769, cov95 0.885 vs 0.923, band pinball 0.1167 vs 0.1130; 24h n 29: cov80 0.241 vs 0.241, cov95 0.621 vs 0.621, band pinball 0.8895 vs 0.8897

## Regime and players (information only, changes no forecast; as_of 2026-11-16T00:12Z)
| | v0 cell | 24h px | funding (90d pct) | OI 24h / 72h | liq 4h long/short | liq 24h long/short | DVOL (90d pct) |
|---|---|---|---|---|---|---|---|
| BTC | R9 liquidation | +1.8% | +0.0033% (p32) | -2.3% / -4.5% | 0.3M / 0.4M | 26.0M / 12.1M | 44.0 (p98) |
| ETH | R9 liquidation | +3.7% | +0.0004% (p20) | -4.6% / -3.9% | 64k / 0.5M | 6.7M / 3.4M | 57.1 (p93) |
| SOL | R9 liquidation | +3.5% | -0.0061% (p10) | -2.1% / -6.2% | 2k / 46k | 1.4M / 0.9M | no index |
- Pattern: BTC long flush, funding normal, OI falling into the rise (short covering), implied vol high; ETH two-sided liquidations, funding normal, OI falling into the rise (short covering), short-liquidation burst (4h), implied vol high; SOL two-sided liquidations, funding negative, OI falling into the rise (short covering), short-liquidation burst (4h)
- Long/short ratio omitted: lake ls_ratio last known 2026-11-07T20:00 (one-off backfill of 11-07; the live updater has no ls_ratio stream)

## Cross-section v3 (context only, not a signal: R-XS-2/R-XS-3; xs3-20261116T00, 23/23 tokens)
- 4h relative median vs EW basket, bp [80% ACI]: top TRX +0 [-273, +256], ETH -0 [-97, +94], BTC -1 [-119, +111], BNB -2 [-112, +105], ETC -4 [-76, +70]
- bottom SUI -11 [-90, +85], HYPE -12 [-182, +176], ENA -13 [-159, +155], NEAR -13 [-174, +172], AAVE -14 [-146, +150]
- mostly low vol: rank corr with -lsig2 4h +0.72 / 24h +0.71; top 5 shared with the low-vol top 5: 3/5 (TRX, BTC, BNB, LTC, ATOM)
- BTC 1h |r|: v3 q50 0.38%, q90 1.04% (log-HAR-t q90 1.68%) vs v1 90% half-width 1.59% (ACI90 1.84%)
- running (scored lines): 4h IC v3 -0.088 vs low-vol -0.201 (n 7); 24h IC n/a; 1h vol pinball v3/HAR 0.878 (n 10)
- 24h ACI band under-covers (holdout 0.749 at 80%); the decision sessions do not read this block

## Field card (btc_field_2026-11-13.json, regime R9)
- 1h: 90% [-2.12, +2.20] med +0.01; touch +1/-1 46/45%, +2/-2 13/12%, +3/-3 3/3%; P(jump) 1%
- 4h: 90% [-3.63, +3.93] med +0.12; touch +1/-1 66/63%, +2/-2 34/31%, +3/-3 16/14%; P(jump) 6%
- 24h: 90% [-7.85, +7.87] med +0.53; touch +1/-1 84/80%, +2/-2 66/59%, +3/-3 50/43%; P(jump) 30%
- variance drivers: s +1.32, L*s +0.26, x1 -0.25; jump drivers: m24*dir -0.03

## Latent (1h net, 4 dims)
- z [0.08, 1.64, 0.27, 0.59], predicted next [-0.08, 1.25, 0.21, 0.37]
- last hour: model error 1.23 vs persistence 0.91 (latent sd); 168h R^2 gain +0.138

## News
- alt: wp_btc -8.02, wp_ind -6.26, wp_max -4.92, wp_edits +0.00, hn_story +0.00, hn_comment -0.58, fng -0.80, fng_chg -0.06
- RSS items last 1h 1, 24h 27

## Running coverage (ACI, last 72 / 168 resolved)
- 1h: 80% 0.694 / 0.750; 90% 0.806 / 0.863; 95% 0.875 / 0.934 (n 168)
- 24h: 80% 0.583 / 0.714; 90% 0.708 / 0.804; 95% 0.833 / 0.887 (n 168)

## Scored this run
- nn-20261115T00 24h: r +1.846% (z +1.00), in ACI80 True, pinball 0.3763
- nn-20261115T23 1h: r +0.175% (z +0.20), in ACI80 True, pinball 0.1034

## Alarms (8)
- coverage (band level/window cov): 1h 90%/72 0.806; 1h 95%/72 0.875; 24h 80%/72 0.583; 24h 80%/168 0.714; 24h 90%/72 0.708; 24h 90%/168 0.804; 24h 95%/72 0.833; 24h 95%/168 0.887

## Changed vs previous line (nn-20261115T23)
- 1h sigma 0.89 -> 0.87%; 24h sigma 3.13 -> 2.94%; 1h alpha80 0.1600 -> 0.1610; field 1h P(jump) 1.5 -> 1.5%
