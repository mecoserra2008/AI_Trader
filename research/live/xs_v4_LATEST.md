# xs.v4 shadow line (not LATEST.md)

_SHADOW, context only, not a signal (R-XS-2 / R-XS-3); no orders; the decision sessions do not read it. H-XS4 forward test (rank IC vs the low-vol ranking). Updated 2026-11-24T08:02 UTC, forecast hour S = 2026-11-24T08:00 UTC._

- universe now: 28 perps pass 24h turnover >= 20 M USD and spread <= 5 bp; present in the panel: 28
- absent: none
- model: trained 2026-11-16T00:35, data to 2026-11-16T00:00, weights 787be7ca6d62addf

| token | sector | 24h rel q10 / q50 / q90 (bp) | rank 24h | rank 4h | low-vol rank | 1h |r| q50 / q90 % (v4) | log-HAR-t q50 / q90 % |
|---|---|---|---|---|---|---|---|
| AKEUSDT | ai | -3380 / +130 / +4284 | 1 | 1 | 27 | 2.02 / 5.32 | 3.09 / 8.78 |
| AAVEUSDT | defi | -279 / +2 / +294 | 2 | 3 | 15 | 0.50 / 1.30 | 0.47 / 1.33 |
| SOLUSDT | l1 | -259 / +2 / +279 | 3 | 6 | 5 | 0.34 / 0.93 | 0.32 / 0.91 |
| BTCUSDT | btc | -240 / +1 / +253 | 4 | 4 | 4 | 0.30 / 0.81 | 0.30 / 0.85 |
| XAUTUSDT | other | -336 / +1 / +382 | 5 | 10 | 1 | 0.17 / 0.46 | 0.19 / 0.53 |
| AVAXUSDT | l1 | -228 / -2 / +233 | 6 | 8 | 7 | 0.38 / 0.99 | 0.36 / 1.01 |
| ETHUSDT | l1 | -208 / -2 / +209 | 7 | 5 | 6 | 0.33 / 0.89 | 0.33 / 0.93 |
| TAOUSDT | ai | -239 / -3 / +252 | 8 | 11 | 11 | 0.41 / 1.07 | 0.41 / 1.17 |
| XRPUSDT | payments | -309 / -5 / +326 | 9 | 12 | 10 | 0.36 / 0.99 | 0.41 / 1.17 |
| BNBUSDT | exchange | -242 / -5 / +245 | 10 | 9 | 2 | 0.29 / 0.77 | 0.31 / 0.88 |
| PIEVERSEUSDT | ai | -1343 / -9 / +1458 | 11 | 2 | 28 | 1.61 / 4.03 | 1.83 / 5.21 |
| ENAUSDT | defi | -281 / -10 / +278 | 12 | 15 | 20 | 0.55 / 1.40 | 0.62 / 1.76 |
| ONDOUSDT | rwa | -226 / -10 / +221 | 13 | 13 | 13 | 0.43 / 1.11 | 0.44 / 1.24 |
| LTCUSDT | payments | -255 / -16 / +287 | 14 | 24 | 9 | 0.39 / 1.00 | 0.38 / 1.09 |
| LINKUSDT | oracle | -264 / -16 / +276 | 15 | 20 | 3 | 0.36 / 0.98 | 0.33 / 0.94 |
| HYPEUSDT | exchange | -274 / -17 / +286 | 16 | 21 | 12 | 0.44 / 1.16 | 0.43 / 1.22 |
| ARBUSDT | l2 | -262 / -18 / +258 | 17 | 17 | 19 | 0.55 / 1.38 | 0.58 / 1.65 |
| DOGEUSDT | meme | -245 / -18 / +245 | 18 | 19 | 8 | 0.35 / 0.94 | 0.32 / 0.91 |
| ZECUSDT | privacy | -317 / -18 / +307 | 19 | 7 | 23 | 0.58 / 1.53 | 0.63 / 1.80 |
| PUMPFUNUSDT | meme | -297 / -19 / +314 | 20 | 22 | 21 | 0.58 / 1.48 | 0.60 / 1.70 |
| SUIUSDT | l1 | -227 / -19 / +204 | 21 | 14 | 14 | 0.44 / 1.15 | 0.45 / 1.29 |
| INJUSDT | l1 | -296 / -19 / +281 | 22 | 16 | 22 | 0.57 / 1.48 | 0.62 / 1.75 |
| UNIUSDT | defi | -267 / -25 / +255 | 23 | 23 | 17 | 0.52 / 1.34 | 0.48 / 1.36 |
| LITUSDT | other | -374 / -27 / +352 | 24 | 18 | 26 | 0.64 / 1.66 | 0.73 / 2.07 |
| WLDUSDT | ai | -309 / -33 / +321 | 25 | 26 | 16 | 0.53 / 1.38 | 0.52 / 1.48 |
| NEARUSDT | l1 | -318 / -35 / +337 | 26 | 25 | 18 | 0.58 / 1.51 | 0.57 / 1.63 |
| VVVUSDT | ai | -549 / -48 / +675 | 27 | 28 | 25 | 0.77 / 1.99 | 0.61 / 1.73 |
| 1000PEPEUSDT | meme | -407 / -64 / +438 | 28 | 27 | 24 | 0.65 / 1.71 | 0.62 / 1.75 |

Top 5 by 24h median: AKEUSDT, AAVEUSDT, SOLUSDT, BTCUSDT, XAUTUSDT. Bottom 5: LITUSDT, WLDUSDT, NEARUSDT, VVVUSDT, 1000PEPEUSDT. Calmest 5 (low-vol): XAUTUSDT, BNBUSDT, LINKUSDT, BTCUSDT, SOLUSDT.

Forward scores (H-XS4; evaluated once at 60 daily origins):

- 4h origins: n 19, mean IC v4 +0.045, low-vol +0.073, diff -0.028
- 24h origins: n 6, mean IC v4 +0.252, low-vol +0.187, diff +0.065
- 1h vol pinball: n 73, v4 +0.127, log-HAR-t +0.131
