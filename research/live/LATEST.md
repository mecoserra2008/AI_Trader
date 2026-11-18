# BTC live network: bar 2026-11-18 08:00 UTC (line nn-20261118T08)
as_of 2026-11-18T08:03:06Z, newest bar closed 08:00, close 64,854.2. Flags: rolling_day. Research only, no direction call.

## Next hour [11-18T08:00 -> 11-18T09:00], % log return
- PNN median +0.01, 80% [-0.52, +0.55], 90% [-0.73, +0.74], sigma 0.42%
- ACI 80% [-0.56, +0.58] alpha_t 0.1770; 90% [-0.82, +0.83] alpha_t 0.0810; 95% [-1.16, +1.18] alpha_t 0.0380
- H-NN10 vol (bar 07:00): ML 0.470% vs baseline 0.486% (ratio 0.97) - not this bar's line

## Next 24h [11-18T08:00 -> 11-19T08:00], % log return
- PNN median +0.41, 80% [-3.00, +4.52], 90% [-4.88, +6.10], sigma 2.94%
- ACI 80% [-2.89, +4.41] alpha_t 0.2063; 90% [-5.25, +6.46] alpha_t 0.0920; 95% [-7.72, +8.93] alpha_t 0.0405
- 24h net: rolling day, shift 8 h

## v2.1 shadow (ACI gamma 0.02, issue alpha >= nominal/4, jump-cluster floor; cfg 2d4ba801a6363fe9; v21-20261118T08)
- 1h: 80% [-0.56, +0.59]; 90% [-0.82, +0.83]; 95% [-1.07, +1.08]; alpha_t (unclipped) 80/90/95 0.1740/0.0795/0.0448; floor inactive
- 24h: 80% [-2.92, +4.45]; 90% [-5.44, +6.65]; 95% [-8.13, +9.34]; alpha_t 0.2013/0.0861/0.0360
- running coverage (issued, 72 / 168): 1h 80% 0.861/0.816, 90% 0.944/0.893, 95% 0.986/0.952 | 24h 80% 0.722/0.738, 90% 0.806/0.821, 95% 0.889/0.887
- v1 vs v2 since 11-13T20:00 (same bars; v2 mostly replay): 1h n 87: cov80 0.759 vs 0.793, cov95 0.931 vs 0.954, band pinball 0.0923 vs 0.0913; 24h n 75: cov80 0.693 vs 0.693, cov95 0.853 vs 0.853, band pinball 0.5179 vs 0.5244

## Regime and players (information only, changes no forecast; as_of 2026-11-18T08:03Z)
| | v0 cell | 24h px | funding (90d pct) | OI 24h / 72h | liq 4h long/short | liq 24h long/short | DVOL (90d pct) |
|---|---|---|---|---|---|---|---|
| BTC | R9 liquidation | +0.2% | +0.0046% (p46) | -3.3% / -2.0% | 31k / 0.4M | 0.5M / 2.0M | 44.0 (p97) |
| ETH | R9 liquidation | -0.8% | +0.0058% (p66) | -2.9% / -1.0% | 16k / 61k | 0.6M / 1.4M | 57.8 (p94) |
| SOL | R6 whipsaw | +3.0% | -0.0070% (p9) | -3.5% / -6.3% | 10k / 29k | 54k / 0.6M | no index |
- Pattern: BTC short squeeze, funding normal, OI falling into the rise (short covering), short-liquidation burst (4h), implied vol high; ETH short squeeze, funding normal, OI falling with price (deleveraging), short-liquidation burst (4h), implied vol high; SOL short squeeze, funding negative, OI falling into the rise (short covering), short-liquidation burst (4h)
- Long/short ratio omitted: lake ls_ratio last known 2026-11-07T20:00 (one-off backfill of 11-07; the live updater has no ls_ratio stream)

## Cross-section v3 (context only, not a signal: R-XS-2/R-XS-3; xs3-20261118T08, 23/23 tokens)
- 4h relative median vs EW basket, bp [80% ACI]: top ETH +1 [-58, +60], BTC -0 [-66, +67], BNB -1 [-67, +67], TRX -1 [-186, +182], BCH -1 [-65, +64]
- bottom ETC -8 [-72, +69], FIL -9 [-117, +113], XLM -12 [-121, +124], NEAR -12 [-152, +159], XRP -17 [-150, +152]
- mostly low vol: rank corr with -lsig2 4h +0.70 / 24h +0.66; top 5 shared with the low-vol top 5: 3/5 (TRX, BNB, BTC, ATOM, LTC)
- BTC 1h |r|: v3 q50 0.22%, q90 0.61% (log-HAR-t q90 0.72%) vs v1 90% half-width 0.73% (ACI90 0.83%)
- running (scored lines): 4h IC v3 +0.030 vs low-vol -0.009 (n 44); 24h IC v3 +0.038 vs low-vol +0.019 (n 35); 1h vol pinball v3/HAR 0.941 (n 47)
- 24h ACI band under-covers (holdout 0.749 at 80%); the decision sessions do not read this block

## Field card (btc_field_2026-11-13.json, regime R9)
- 1h: 90% [-0.67, +0.68] med +0.00; touch +1/-1 3/2%, +2/-2 0/0%, +3/-3 0/0%; P(jump) 2%
- 4h: 90% [-1.38, +1.45] med +0.04; touch +1/-1 22/20%, +2/-2 3/3%, +3/-3 1/1%; P(jump) 6%
- 24h: 90% [-4.21, +4.29] med +0.28; touch +1/-1 68/62%, +2/-2 41/36%, +3/-3 22/19%; P(jump) 30%
- variance drivers: x2 -0.11, s -0.03, x1 +0.01; jump drivers: m24*dir -0.00

## Latent (1h net, 4 dims)
- z [0.01, 0.5, -0.3, 0.02], predicted next [0.04, 0.69, -0.07, -0.07]
- last hour: model error 0.55 vs persistence 0.51 (latent sd); 168h R^2 gain +0.161

## News
- alt: wp_btc -8.02, wp_ind -6.26, wp_max -4.92, wp_edits +0.00, hn_story +0.00, hn_comment -0.37, fng -0.78, fng_chg -0.02
- RSS items last 1h 2, 24h 62

## Running coverage (ACI, last 72 / 168 resolved)
- 1h: 80% 0.861 / 0.786; 90% 0.944 / 0.875; 95% 0.986 / 0.934 (n 168)
- 24h: 80% 0.722 / 0.738; 90% 0.806 / 0.821; 95% 0.889 / 0.887 (n 168)

## Scored this run
- nn-20261117T08 24h: r +0.238% (z +0.08), in ACI80 True, pinball 0.3487
- nn-20261118T07 1h: r +0.424% (z +1.19), in ACI80 True, pinball 0.0914

## Alarms (3)
- coverage (band level/window cov): 24h 90%/72 0.806; 24h 90%/168 0.821; 24h 95%/168 0.887

## Changed vs previous line (nn-20261118T07)
- 1h sigma 0.36 -> 0.42%; 24h sigma 2.95 -> 2.94%; 1h alpha80 0.1760 -> 0.1770; field 1h P(jump) 1.5 -> 1.5%
- new alarms: 1
