# xs.v4 shadow line (not LATEST.md)

_SHADOW, context only, not a signal (R-XS-2 / R-XS-3); no orders; the decision sessions do not read it. H-XS4 forward test (rank IC vs the low-vol ranking). Updated 2026-11-18T08:02 UTC, forecast hour S = 2026-11-18T08:00 UTC._

- universe now: 26 perps pass 24h turnover >= 20 M USD and spread <= 5 bp; present in the panel: 26
- absent: none
- model: trained 2026-11-16T00:35, data to 2026-11-16T00:00, weights 787be7ca6d62addf

| token | sector | 24h rel q10 / q50 / q90 (bp) | rank 24h | rank 4h | low-vol rank | 1h |r| q50 / q90 % (v4) | log-HAR-t q50 / q90 % |
|---|---|---|---|---|---|---|---|
| AKEUSDT | ai | -3156 / +90 / +4426 | 1 | 2 | 24 | 2.19 / 6.31 | 2.90 / 8.25 |
| XAUTUSDT | other | -415 / +10 / +478 | 2 | 4 | 1 | 0.16 / 0.43 | 0.14 / 0.40 |
| BTCUSDT | btc | -391 / +6 / +394 | 3 | 3 | 2 | 0.27 / 0.75 | 0.24 / 0.69 |
| AVAXUSDT | l1 | -339 / -2 / +366 | 4 | 8 | 8 | 0.45 / 1.23 | 0.40 / 1.14 |
| SOLUSDT | l1 | -258 / -3 / +294 | 5 | 6 | 7 | 0.40 / 1.08 | 0.36 / 1.02 |
| HYPEUSDT | exchange | -305 / -6 / +314 | 6 | 5 | 11 | 0.45 / 1.20 | 0.40 / 1.14 |
| ETHUSDT | l1 | -343 / -7 / +331 | 7 | 7 | 3 | 0.35 / 0.94 | 0.34 / 0.98 |
| DOGEUSDT | meme | -305 / -11 / +312 | 8 | 9 | 4 | 0.41 / 1.12 | 0.38 / 1.09 |
| LINKUSDT | oracle | -284 / -18 / +275 | 9 | 11 | 5 | 0.43 / 1.17 | 0.40 / 1.14 |
| AAVEUSDT | defi | -355 / -19 / +345 | 10 | 10 | 14 | 0.50 / 1.38 | 0.51 / 1.45 |
| ASTERUSDT | exchange | -575 / -19 / +549 | 11 | 1 | 23 | 0.82 / 2.12 | 1.05 / 2.98 |
| XRPUSDT | payments | -305 / -19 / +381 | 12 | 15 | 6 | 0.49 / 1.26 | 0.36 / 1.04 |
| SUIUSDT | l1 | -264 / -22 / +264 | 13 | 12 | 9 | 0.46 / 1.25 | 0.40 / 1.15 |
| 1000PEPEUSDT | meme | -512 / -24 / +541 | 14 | 17 | 12 | 0.60 / 1.71 | 0.49 / 1.39 |
| ONDOUSDT | rwa | -308 / -24 / +294 | 15 | 13 | 10 | 0.49 / 1.35 | 0.44 / 1.24 |
| UNIUSDT | defi | -385 / -26 / +399 | 16 | 16 | 15 | 0.58 / 1.60 | 0.53 / 1.51 |
| WLDUSDT | ai | -381 / -32 / +386 | 17 | 18 | 13 | 0.53 / 1.51 | 0.46 / 1.32 |
| INJUSDT | l1 | -390 / -35 / +398 | 18 | 22 | 20 | 0.68 / 1.82 | 0.65 / 1.84 |
| ZECUSDT | privacy | -353 / -37 / +364 | 19 | 20 | 18 | 0.67 / 1.84 | 0.63 / 1.78 |
| LITUSDT | other | -574 / -38 / +590 | 20 | 21 | 21 | 0.76 / 2.02 | 0.70 / 1.99 |
| NEARUSDT | l1 | -392 / -40 / +412 | 21 | 14 | 17 | 0.64 / 1.74 | 0.60 / 1.72 |
| ENAUSDT | defi | -502 / -42 / +518 | 22 | 23 | 16 | 0.67 / 1.85 | 0.63 / 1.79 |
| PUMPFUNUSDT | meme | -521 / -45 / +547 | 23 | 19 | 19 | 0.71 / 1.96 | 0.66 / 1.87 |
| PIEVERSEUSDT | ai | -903 / -83 / +1112 | 24 | 24 | 22 | 1.19 / 3.24 | 0.86 / 2.46 |
| BREVUSDT | other | -1856 / -105 / +2557 | 25 | 25 | 25 | 2.29 / 6.42 | 1.57 / 4.47 |
| MMTUSDT | other | -3355 / -356 / +5142 | 26 | 26 | 26 | 4.40 / 12.28 | 2.43 / 6.90 |

Top 5 by 24h median: AKEUSDT, XAUTUSDT, BTCUSDT, AVAXUSDT, SOLUSDT. Bottom 5: ENAUSDT, PUMPFUNUSDT, PIEVERSEUSDT, BREVUSDT, MMTUSDT. Calmest 5 (low-vol): XAUTUSDT, BTCUSDT, ETHUSDT, DOGEUSDT, LINKUSDT.

Forward scores (H-XS4; evaluated once at 60 daily origins):

- 4h origins: n 8, mean IC v4 +0.052, low-vol +0.111, diff -0.059
- 24h origins: n 2, mean IC v4 +0.037, low-vol +0.053, diff -0.016
- 1h vol pinball: n 34, v4 +0.104, log-HAR-t +0.107
