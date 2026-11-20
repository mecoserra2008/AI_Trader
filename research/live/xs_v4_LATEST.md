# xs.v4 shadow line (not LATEST.md)

_SHADOW, context only, not a signal (R-XS-2 / R-XS-3); no orders; the decision sessions do not read it. H-XS4 forward test (rank IC vs the low-vol ranking). Updated 2026-11-20T08:02 UTC, forecast hour S = 2026-11-20T08:00 UTC._

- universe now: 26 perps pass 24h turnover >= 20 M USD and spread <= 5 bp; present in the panel: 26
- absent: none
- model: trained 2026-11-16T00:35, data to 2026-11-16T00:00, weights 787be7ca6d62addf

| token | sector | 24h rel q10 / q50 / q90 (bp) | rank 24h | rank 4h | low-vol rank | 1h |r| q50 / q90 % (v4) | log-HAR-t q50 / q90 % |
|---|---|---|---|---|---|---|---|
| XAUTUSDT | other | -317 / -7 / +344 | 1 | 5 | 1 | 0.23 / 0.61 | 0.17 / 0.50 |
| BTCUSDT | btc | -196 / -9 / +199 | 2 | 1 | 2 | 0.35 / 0.95 | 0.32 / 0.90 |
| XRPUSDT | payments | -253 / -24 / +273 | 3 | 10 | 3 | 0.41 / 1.12 | 0.36 / 1.03 |
| SOLUSDT | l1 | -211 / -24 / +198 | 4 | 2 | 7 | 0.45 / 1.24 | 0.42 / 1.18 |
| LTCUSDT | payments | -236 / -25 / +229 | 5 | 3 | 10 | 0.52 / 1.44 | 0.45 / 1.27 |
| XMRUSDT | privacy | -629 / -25 / +686 | 6 | 6 | 21 | 0.63 / 1.77 | 0.77 / 2.19 |
| SUIUSDT | l1 | -189 / -25 / +169 | 7 | 8 | 8 | 0.46 / 1.25 | 0.46 / 1.30 |
| DOGEUSDT | meme | -185 / -26 / +162 | 8 | 4 | 5 | 0.43 / 1.20 | 0.41 / 1.15 |
| ETHUSDT | l1 | -214 / -30 / +199 | 9 | 9 | 4 | 0.43 / 1.17 | 0.38 / 1.09 |
| AVAXUSDT | l1 | -217 / -30 / +203 | 10 | 11 | 6 | 0.46 / 1.26 | 0.43 / 1.24 |
| AAVEUSDT | defi | -272 / -31 / +286 | 11 | 12 | 11 | 0.54 / 1.54 | 0.49 / 1.39 |
| LINKUSDT | oracle | -203 / -32 / +170 | 12 | 7 | 9 | 0.47 / 1.31 | 0.47 / 1.35 |
| INITUSDT | l1 | -1159 / -33 / +1439 | 13 | 19 | 23 | 1.31 / 3.63 | 1.24 / 3.52 |
| 1000PEPEUSDT | meme | -329 / -37 / +333 | 14 | 13 | 17 | 0.66 / 1.76 | 0.68 / 1.94 |
| PUMPFUNUSDT | meme | -348 / -37 / +380 | 15 | 14 | 16 | 0.71 / 1.96 | 0.67 / 1.91 |
| HYPEUSDT | exchange | -306 / -46 / +294 | 16 | 15 | 13 | 0.57 / 1.61 | 0.60 / 1.71 |
| UNIUSDT | defi | -286 / -48 / +286 | 17 | 18 | 12 | 0.69 / 1.88 | 0.54 / 1.54 |
| SKYUSDT | other | -1004 / -48 / +1149 | 18 | 21 | 25 | 1.16 / 3.22 | 1.17 / 3.33 |
| ENAUSDT | defi | -360 / -52 / +436 | 19 | 22 | 15 | 0.77 / 2.18 | 0.64 / 1.83 |
| ZECUSDT | privacy | -356 / -54 / +349 | 20 | 16 | 19 | 0.76 / 2.07 | 0.69 / 1.96 |
| INJUSDT | l1 | -358 / -56 / +364 | 21 | 20 | 22 | 0.77 / 1.97 | 0.80 / 2.26 |
| NEARUSDT | l1 | -358 / -58 / +352 | 22 | 17 | 18 | 0.74 / 1.99 | 0.70 / 2.00 |
| WLDUSDT | ai | -326 / -65 / +337 | 23 | 23 | 14 | 0.70 / 1.96 | 0.67 / 1.91 |
| STRKUSDT | l2 | -408 / -83 / +452 | 24 | 24 | 20 | 0.87 / 2.39 | 0.68 / 1.94 |
| BARDUSDT | other | -757 / -108 / +946 | 25 | 25 | 24 | 1.30 / 3.76 | 1.10 / 3.13 |
| PIEVERSEUSDT | ai | -1713 / -142 / +2146 | 26 | 26 | 26 | 1.90 / 5.23 | 1.73 / 4.93 |

Top 5 by 24h median: XAUTUSDT, BTCUSDT, XRPUSDT, SOLUSDT, LTCUSDT. Bottom 5: NEARUSDT, WLDUSDT, STRKUSDT, BARDUSDT, PIEVERSEUSDT. Calmest 5 (low-vol): XAUTUSDT, BTCUSDT, XRPUSDT, ETHUSDT, DOGEUSDT.

Forward scores (H-XS4; evaluated once at 60 daily origins):

- 4h origins: n 14, mean IC v4 +0.057, low-vol +0.020, diff +0.037
- 24h origins: n 4, mean IC v4 +0.262, low-vol +0.200, diff +0.061
- 1h vol pinball: n 56, v4 +0.118, log-HAR-t +0.121
