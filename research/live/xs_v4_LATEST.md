# xs.v4 shadow line (not LATEST.md)

_SHADOW, context only, not a signal (R-XS-2 / R-XS-3); no orders; the decision sessions do not read it. H-XS4 forward test (rank IC vs the low-vol ranking). Updated 2026-11-28T04:02 UTC, forecast hour S = 2026-11-28T04:00 UTC._

- universe now: 24 perps pass 24h turnover >= 20 M USD and spread <= 5 bp; present in the panel: 24
- absent: none
- model: trained 2026-11-16T00:35, data to 2026-11-16T00:00, weights 787be7ca6d62addf

| token | sector | 24h rel q10 / q50 / q90 (bp) | rank 24h | rank 4h | low-vol rank | 1h |r| q50 / q90 % (v4) | log-HAR-t q50 / q90 % |
|---|---|---|---|---|---|---|---|
| INITUSDT | l1 | -1748 / +82 / +2128 | 1 | 4 | 23 | 1.46 / 3.87 | 1.80 / 5.12 |
| MNTUSDT | l2 | -262 / +19 / +323 | 2 | 1 | 8 | 0.29 / 0.76 | 0.30 / 0.86 |
| XAUTUSDT | other | -224 / +4 / +274 | 3 | 3 | 1 | 0.10 / 0.25 | 0.07 / 0.21 |
| SSVUSDT | other | -706 / +2 / +823 | 4 | 2 | 24 | 1.43 / 3.67 | 1.38 / 3.92 |
| BTCUSDT | btc | -202 / -1 / +222 | 5 | 6 | 2 | 0.18 / 0.50 | 0.14 / 0.40 |
| ETHUSDT | l1 | -153 / -1 / +160 | 6 | 5 | 3 | 0.22 / 0.59 | 0.18 / 0.52 |
| AAVEUSDT | defi | -198 / -10 / +194 | 7 | 7 | 10 | 0.30 / 0.82 | 0.30 / 0.86 |
| LINKUSDT | oracle | -179 / -11 / +172 | 8 | 8 | 12 | 0.34 / 0.90 | 0.31 / 0.88 |
| WLDUSDT | ai | -202 / -11 / +206 | 9 | 11 | 9 | 0.33 / 0.90 | 0.34 / 0.97 |
| LTCUSDT | payments | -195 / -13 / +213 | 10 | 12 | 11 | 0.33 / 0.87 | 0.31 / 0.89 |
| XRPUSDT | payments | -192 / -13 / +196 | 11 | 9 | 4 | 0.25 / 0.68 | 0.23 / 0.67 |
| SOLUSDT | l1 | -180 / -16 / +176 | 12 | 10 | 5 | 0.26 / 0.73 | 0.26 / 0.75 |
| SUIUSDT | l1 | -166 / -18 / +148 | 13 | 13 | 7 | 0.29 / 0.78 | 0.27 / 0.76 |
| UNIUSDT | defi | -205 / -18 / +212 | 14 | 16 | 13 | 0.39 / 1.03 | 0.39 / 1.11 |
| HYPEUSDT | exchange | -222 / -19 / +240 | 15 | 17 | 17 | 0.45 / 1.21 | 0.46 / 1.32 |
| DOGEUSDT | meme | -180 / -19 / +172 | 16 | 14 | 6 | 0.28 / 0.75 | 0.25 / 0.71 |
| ENAUSDT | defi | -223 / -21 / +214 | 17 | 15 | 16 | 0.40 / 1.07 | 0.40 / 1.15 |
| 1000PEPEUSDT | meme | -244 / -26 / +244 | 18 | 18 | 15 | 0.40 / 1.08 | 0.38 / 1.07 |
| XMRUSDT | privacy | -620 / -33 / +698 | 19 | 20 | 18 | 0.66 / 1.77 | 0.64 / 1.81 |
| BIGTIMEUSDT | other | -738 / -44 / +831 | 20 | 19 | 22 | 1.25 / 3.32 | 1.33 / 3.78 |
| NEARUSDT | l1 | -305 / -55 / +351 | 21 | 21 | 14 | 0.47 / 1.30 | 0.42 / 1.20 |
| PIEVERSEUSDT | ai | -1012 / -89 / +1283 | 22 | 23 | 21 | 1.31 / 3.59 | 1.55 / 4.40 |
| ZECUSDT | privacy | -420 / -92 / +450 | 23 | 22 | 19 | 0.69 / 1.85 | 0.70 / 1.99 |
| BTWUSDT | other | -1402 / -101 / +1881 | 24 | 24 | 20 | 1.31 / 3.40 | 1.91 / 5.44 |

Top 5 by 24h median: INITUSDT, MNTUSDT, XAUTUSDT, SSVUSDT, BTCUSDT. Bottom 5: BIGTIMEUSDT, NEARUSDT, PIEVERSEUSDT, ZECUSDT, BTWUSDT. Calmest 5 (low-vol): XAUTUSDT, BTCUSDT, ETHUSDT, XRPUSDT, SOLUSDT.

Forward scores (H-XS4; evaluated once at 60 daily origins):

- 4h origins: n 31, mean IC v4 +0.043, low-vol +0.074, diff -0.031
- 24h origins: n 10, mean IC v4 +0.117, low-vol +0.105, diff +0.012
- 1h vol pinball: n 118, v4 +0.120, log-HAR-t +0.122
