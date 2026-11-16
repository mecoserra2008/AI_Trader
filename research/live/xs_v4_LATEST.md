# xs.v4 shadow line (not LATEST.md)

_SHADOW, context only, not a signal (R-XS-2 / R-XS-3); no orders; the decision sessions do not read it. H-XS4 forward test (rank IC vs the low-vol ranking). Updated 2026-11-16T00:36 UTC, forecast hour S = 2026-11-16T00:00 UTC._

- universe now: 26 perps pass 24h turnover >= 20 M USD and spread <= 5 bp; present in the panel: 26
- absent: none
- model: trained 2026-11-16T00:35, data to 2026-11-16T01:00, weights 787be7ca6d62addf

| token | sector | 24h rel q10 / q50 / q90 (bp) | rank 24h | rank 4h | low-vol rank | 1h |r| q50 / q90 % (v4) | log-HAR-t q50 / q90 % |
|---|---|---|---|---|---|---|---|
| AKEUSDT | ai | -4518 / +337 / +6451 | 1 | 1 | 26 | 3.52 / 10.26 | 4.50 / 12.79 |
| INITUSDT | l1 | -1575 / +135 / +2292 | 2 | 3 | 25 | 2.60 / 7.13 | 2.83 / 8.06 |
| XAUTUSDT | other | -247 / +17 / +325 | 3 | 4 | 1 | 0.30 / 0.79 | 0.38 / 1.07 |
| ASTERUSDT | exchange | -922 / +6 / +927 | 4 | 2 | 24 | 2.12 / 5.23 | 2.22 / 6.31 |
| XRPUSDT | payments | -255 / -1 / +258 | 5 | 10 | 9 | 0.62 / 1.59 | 0.80 / 2.27 |
| BNBUSDT | exchange | -236 / -3 / +204 | 6 | 6 | 3 | 0.44 / 1.15 | 0.60 / 1.70 |
| BTCUSDT | btc | -238 / -4 / +196 | 7 | 7 | 2 | 0.45 / 1.21 | 0.66 / 1.87 |
| ETHUSDT | l1 | -231 / -5 / +193 | 8 | 5 | 6 | 0.60 / 1.57 | 0.86 / 2.45 |
| LTCUSDT | payments | -211 / -11 / +170 | 9 | 8 | 4 | 0.50 / 1.34 | 0.74 / 2.09 |
| SOLUSDT | l1 | -235 / -11 / +202 | 10 | 9 | 10 | 0.63 / 1.65 | 0.84 / 2.39 |
| AVAXUSDT | l1 | -243 / -17 / +200 | 11 | 12 | 11 | 0.68 / 1.75 | 0.91 / 2.59 |
| 1000PEPEUSDT | meme | -327 / -21 / +267 | 12 | 14 | 14 | 0.80 / 2.03 | 1.05 / 2.98 |
| DOGEUSDT | meme | -248 / -25 / +199 | 13 | 15 | 8 | 0.67 / 1.74 | 0.84 / 2.38 |
| LINKUSDT | oracle | -210 / -25 / +160 | 14 | 13 | 5 | 0.59 / 1.54 | 0.85 / 2.42 |
| ZECUSDT | privacy | -360 / -27 / +325 | 15 | 21 | 20 | 0.97 / 2.57 | 1.08 / 3.08 |
| INJUSDT | l1 | -302 / -27 / +245 | 16 | 11 | 21 | 0.79 / 1.97 | 1.08 / 3.07 |
| HYPEUSDT | exchange | -290 / -31 / +237 | 17 | 16 | 16 | 0.69 / 1.78 | 0.85 / 2.43 |
| ONDOUSDT | rwa | -229 / -33 / +172 | 18 | 17 | 12 | 0.73 / 1.86 | 0.97 / 2.74 |
| AAVEUSDT | defi | -299 / -36 / +252 | 19 | 19 | 7 | 0.73 / 1.89 | 0.76 / 2.17 |
| UNIUSDT | defi | -295 / -37 / +228 | 20 | 20 | 17 | 0.87 / 2.22 | 1.03 / 2.93 |
| SUIUSDT | l1 | -244 / -39 / +192 | 21 | 18 | 13 | 0.75 / 1.91 | 0.95 / 2.71 |
| ENAUSDT | defi | -310 / -45 / +250 | 22 | 22 | 15 | 0.85 / 2.19 | 0.96 / 2.74 |
| NEARUSDT | l1 | -308 / -48 / +265 | 23 | 23 | 19 | 0.90 / 2.32 | 1.06 / 3.03 |
| WLDUSDT | ai | -329 / -63 / +281 | 24 | 25 | 18 | 0.90 / 2.28 | 0.99 / 2.81 |
| PUMPFUNUSDT | meme | -348 / -65 / +289 | 25 | 24 | 22 | 0.91 / 2.39 | 1.05 / 2.98 |
| PIEVERSEUSDT | ai | -863 / -86 / +863 | 26 | 26 | 23 | 1.61 / 4.15 | 1.62 / 4.60 |

Top 5 by 24h median: AKEUSDT, INITUSDT, XAUTUSDT, ASTERUSDT, XRPUSDT. Bottom 5: ENAUSDT, NEARUSDT, WLDUSDT, PUMPFUNUSDT, PIEVERSEUSDT. Calmest 5 (low-vol): XAUTUSDT, BTCUSDT, BNBUSDT, LTCUSDT, LINKUSDT.

Forward scores (H-XS4; evaluated once at 60 daily origins):

- 4h origins: n 0, mean IC v4 n/a, low-vol n/a, diff n/a
- 24h origins: n 0, mean IC v4 n/a, low-vol n/a, diff n/a
- 1h vol pinball: n 0, v4 n/a, log-HAR-t n/a
