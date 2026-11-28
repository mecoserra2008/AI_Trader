# xs.v4 shadow line (not LATEST.md)

_SHADOW, context only, not a signal (R-XS-2 / R-XS-3); no orders; the decision sessions do not read it. H-XS4 forward test (rank IC vs the low-vol ranking). Updated 2026-11-28T15:02 UTC, forecast hour S = 2026-11-28T15:00 UTC._

- universe now: 19 perps pass 24h turnover >= 20 M USD and spread <= 5 bp; present in the panel: 19
- absent: none
- model: trained 2026-11-16T00:35, data to 2026-11-16T00:00, weights 787be7ca6d62addf

| token | sector | 24h rel q10 / q50 / q90 (bp) | rank 24h | rank 4h | low-vol rank | 1h |r| q50 / q90 % (v4) | log-HAR-t q50 / q90 % |
|---|---|---|---|---|---|---|---|
| INITUSDT | l1 | -1261 / +27 / +1496 | 1 | 1 | 18 | 1.22 / 3.26 | 1.68 / 4.79 |
| BLURUSDT | other | -1786 / +4 / +2232 | 2 | 9 | 19 | 2.09 / 5.51 | 2.00 / 5.69 |
| USD1USDT | other | -592 / -7 / +695 | 3 | 3 | 13 | 0.37 / 1.01 | 0.46 / 1.32 |
| BTCUSDT | btc | -219 / -11 / +225 | 4 | 2 | 1 | 0.21 / 0.60 | 0.16 / 0.46 |
| ETHUSDT | l1 | -194 / -19 / +187 | 5 | 4 | 2 | 0.25 / 0.73 | 0.20 / 0.58 |
| 1000PEPEUSDT | meme | -274 / -22 / +272 | 6 | 6 | 10 | 0.44 / 1.25 | 0.49 / 1.39 |
| XRPUSDT | payments | -249 / -24 / +258 | 7 | 7 | 3 | 0.30 / 0.88 | 0.26 / 0.75 |
| SOLUSDT | l1 | -211 / -26 / +199 | 8 | 5 | 4 | 0.31 / 0.90 | 0.28 / 0.79 |
| DOGEUSDT | meme | -230 / -27 / +226 | 9 | 8 | 5 | 0.33 / 0.94 | 0.32 / 0.91 |
| HYPEUSDT | exchange | -267 / -35 / +273 | 10 | 10 | 12 | 0.45 / 1.22 | 0.45 / 1.29 |
| LINKUSDT | oracle | -256 / -36 / +256 | 11 | 11 | 7 | 0.36 / 1.00 | 0.30 / 0.86 |
| ENAUSDT | defi | -288 / -38 / +284 | 12 | 12 | 11 | 0.44 / 1.26 | 0.43 / 1.22 |
| SUIUSDT | l1 | -249 / -41 / +254 | 13 | 13 | 6 | 0.38 / 1.11 | 0.30 / 0.87 |
| NEARUSDT | l1 | -300 / -43 / +306 | 14 | 14 | 8 | 0.47 / 1.35 | 0.45 / 1.29 |
| MNTUSDT | l2 | -613 / -46 / +881 | 15 | 18 | 14 | 0.80 / 2.02 | 0.57 / 1.61 |
| UNIUSDT | defi | -267 / -46 / +256 | 16 | 15 | 9 | 0.45 / 1.25 | 0.41 / 1.17 |
| XMRUSDT | privacy | -573 / -55 / +666 | 17 | 17 | 16 | 0.69 / 1.81 | 0.77 / 2.18 |
| ZECUSDT | privacy | -385 / -68 / +409 | 18 | 16 | 15 | 0.64 / 1.74 | 0.66 / 1.87 |
| PIEVERSEUSDT | ai | -1224 / -72 / +1703 | 19 | 19 | 17 | 1.55 / 4.21 | 1.61 / 4.57 |

Top 5 by 24h median: INITUSDT, BLURUSDT, USD1USDT, BTCUSDT, ETHUSDT. Bottom 5: MNTUSDT, UNIUSDT, XMRUSDT, ZECUSDT, PIEVERSEUSDT. Calmest 5 (low-vol): BTCUSDT, ETHUSDT, XRPUSDT, SOLUSDT, DOGEUSDT.

Forward scores (H-XS4; evaluated once at 60 daily origins):

- 4h origins: n 33, mean IC v4 +0.063, low-vol +0.076, diff -0.013
- 24h origins: n 10, mean IC v4 +0.117, low-vol +0.105, diff +0.012
- 1h vol pinball: n 129, v4 +0.118, log-HAR-t +0.120
