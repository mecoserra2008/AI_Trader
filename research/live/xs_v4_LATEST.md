# xs.v4 shadow line (not LATEST.md)

_SHADOW, context only, not a signal (R-XS-2 / R-XS-3); no orders; the decision sessions do not read it. H-XS4 forward test (rank IC vs the low-vol ranking). Updated 2026-11-27T08:02 UTC, forecast hour S = 2026-11-27T08:00 UTC._

- universe now: 24 perps pass 24h turnover >= 20 M USD and spread <= 5 bp; present in the panel: 24
- absent: none
- model: trained 2026-11-16T00:35, data to 2026-11-16T00:00, weights 787be7ca6d62addf

| token | sector | 24h rel q10 / q50 / q90 (bp) | rank 24h | rank 4h | low-vol rank | 1h |r| q50 / q90 % (v4) | log-HAR-t q50 / q90 % |
|---|---|---|---|---|---|---|---|
| XAUTUSDT | other | -294 / +1 / +337 | 1 | 1 | 1 | 0.16 / 0.41 | 0.17 / 0.49 |
| BTCUSDT | btc | -214 / -4 / +244 | 2 | 3 | 2 | 0.25 / 0.66 | 0.24 / 0.69 |
| XRPUSDT | payments | -249 / -15 / +303 | 3 | 7 | 6 | 0.34 / 0.90 | 0.33 / 0.94 |
| AAVEUSDT | defi | -242 / -15 / +249 | 4 | 5 | 8 | 0.39 / 1.05 | 0.40 / 1.15 |
| NEARUSDT | l1 | -253 / -16 / +273 | 5 | 8 | 10 | 0.45 / 1.21 | 0.52 / 1.48 |
| ETHUSDT | l1 | -195 / -16 / +212 | 6 | 6 | 4 | 0.29 / 0.79 | 0.32 / 0.90 |
| DOGEUSDT | meme | -198 / -19 / +226 | 7 | 9 | 5 | 0.31 / 0.84 | 0.32 / 0.91 |
| XMRUSDT | privacy | -662 / -20 / +712 | 8 | 2 | 17 | 0.60 / 1.63 | 0.78 / 2.21 |
| SOLUSDT | l1 | -227 / -23 / +257 | 9 | 10 | 3 | 0.32 / 0.90 | 0.32 / 0.91 |
| SUIUSDT | l1 | -202 / -26 / +198 | 10 | 11 | 7 | 0.34 / 0.96 | 0.38 / 1.08 |
| ZROUSDT | infra | -352 / -30 / +387 | 11 | 12 | 14 | 0.59 / 1.57 | 0.65 / 1.86 |
| ENAUSDT | defi | -301 / -30 / +379 | 12 | 16 | 11 | 0.48 / 1.30 | 0.53 / 1.52 |
| LINKUSDT | oracle | -236 / -36 / +289 | 13 | 19 | 9 | 0.44 / 1.17 | 0.44 / 1.26 |
| HYPEUSDT | exchange | -276 / -37 / +295 | 14 | 15 | 15 | 0.58 / 1.50 | 0.67 / 1.91 |
| 1000PEPEUSDT | meme | -276 / -40 / +283 | 15 | 17 | 13 | 0.49 / 1.30 | 0.57 / 1.61 |
| MOVEUSDT | other | -515 / -43 / +539 | 16 | 4 | 20 | 0.75 / 1.98 | 0.80 / 2.27 |
| UNIUSDT | defi | -285 / -46 / +330 | 17 | 20 | 12 | 0.52 / 1.36 | 0.50 / 1.41 |
| BIGTIMEUSDT | other | -1017 / -49 / +1244 | 18 | 14 | 24 | 1.66 / 4.22 | 1.88 / 5.35 |
| MEUSDT | other | -530 / -51 / +536 | 19 | 18 | 22 | 0.83 / 2.26 | 0.95 / 2.70 |
| ZECUSDT | privacy | -399 / -55 / +391 | 20 | 13 | 19 | 0.76 / 1.99 | 0.99 / 2.82 |
| VVVUSDT | ai | -466 / -57 / +533 | 21 | 21 | 18 | 0.70 / 1.84 | 0.89 / 2.54 |
| PIEVERSEUSDT | ai | -987 / -63 / +1368 | 22 | 23 | 21 | 1.28 / 3.48 | 1.22 / 3.47 |
| SEIUSDT | l1 | -348 / -64 / +372 | 23 | 22 | 16 | 0.62 / 1.69 | 0.65 / 1.84 |
| INITUSDT | l1 | -1452 / -129 / +2028 | 24 | 24 | 23 | 1.64 / 4.56 | 2.06 / 5.85 |

Top 5 by 24h median: XAUTUSDT, BTCUSDT, XRPUSDT, AAVEUSDT, NEARUSDT. Bottom 5: ZECUSDT, VVVUSDT, PIEVERSEUSDT, SEIUSDT, INITUSDT. Calmest 5 (low-vol): XAUTUSDT, BTCUSDT, SOLUSDT, ETHUSDT, DOGEUSDT.

Forward scores (H-XS4; evaluated once at 60 daily origins):

- 4h origins: n 29, mean IC v4 +0.036, low-vol +0.067, diff -0.031
- 24h origins: n 9, mean IC v4 +0.116, low-vol +0.103, diff +0.013
- 1h vol pinball: n 111, v4 +0.121, log-HAR-t +0.124
