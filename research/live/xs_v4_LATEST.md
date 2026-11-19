# xs.v4 shadow line (not LATEST.md)

_SHADOW, context only, not a signal (R-XS-2 / R-XS-3); no orders; the decision sessions do not read it. H-XS4 forward test (rank IC vs the low-vol ranking). Updated 2026-11-19T08:02 UTC, forecast hour S = 2026-11-19T08:00 UTC._

- universe now: 19 perps pass 24h turnover >= 20 M USD and spread <= 5 bp; present in the panel: 19
- absent: none
- model: trained 2026-11-16T00:35, data to 2026-11-16T00:00, weights 787be7ca6d62addf

| token | sector | 24h rel q10 / q50 / q90 (bp) | rank 24h | rank 4h | low-vol rank | 1h |r| q50 / q90 % (v4) | log-HAR-t q50 / q90 % |
|---|---|---|---|---|---|---|---|
| XMRUSDT | privacy | -424 / +11 / +470 | 1 | 1 | 6 | 0.40 / 1.09 | 0.41 / 1.16 |
| XAUTUSDT | other | -339 / +3 / +355 | 2 | 4 | 1 | 0.17 / 0.46 | 0.17 / 0.48 |
| BTCUSDT | btc | -217 / -0 / +222 | 3 | 2 | 2 | 0.28 / 0.73 | 0.26 / 0.73 |
| XRPUSDT | payments | -209 / -3 / +229 | 4 | 5 | 5 | 0.35 / 0.92 | 0.32 / 0.91 |
| ETHUSDT | l1 | -177 / -8 / +170 | 5 | 3 | 4 | 0.32 / 0.86 | 0.32 / 0.90 |
| AAVEUSDT | defi | -233 / -13 / +224 | 6 | 6 | 12 | 0.42 / 1.13 | 0.44 / 1.24 |
| SOLUSDT | l1 | -187 / -15 / +178 | 7 | 7 | 7 | 0.37 / 0.98 | 0.34 / 0.97 |
| HYPEUSDT | exchange | -258 / -16 / +281 | 8 | 12 | 10 | 0.45 / 1.19 | 0.41 / 1.17 |
| DOGEUSDT | meme | -190 / -16 / +183 | 9 | 9 | 3 | 0.34 / 0.90 | 0.30 / 0.85 |
| ENAUSDT | defi | -329 / -18 / +355 | 10 | 8 | 13 | 0.57 / 1.54 | 0.52 / 1.49 |
| SUIUSDT | l1 | -183 / -18 / +170 | 11 | 10 | 8 | 0.40 / 1.05 | 0.38 / 1.07 |
| UNIUSDT | defi | -239 / -21 / +230 | 12 | 11 | 11 | 0.47 / 1.25 | 0.44 / 1.24 |
| BREVUSDT | other | -1104 / -45 / +1186 | 13 | 13 | 19 | 1.34 / 3.30 | 1.64 / 4.65 |
| PUMPFUNUSDT | meme | -382 / -46 / +434 | 14 | 14 | 17 | 0.66 / 1.77 | 0.59 / 1.69 |
| 1000PEPEUSDT | meme | -313 / -47 / +328 | 15 | 15 | 9 | 0.54 / 1.45 | 0.45 / 1.29 |
| INJUSDT | l1 | -400 / -52 / +461 | 16 | 17 | 16 | 0.69 / 1.79 | 0.69 / 1.96 |
| ZECUSDT | privacy | -350 / -62 / +368 | 17 | 16 | 14 | 0.60 / 1.62 | 0.60 / 1.70 |
| NEARUSDT | l1 | -405 / -63 / +475 | 18 | 18 | 15 | 0.63 / 1.70 | 0.60 / 1.72 |
| ZKCUSDT | other | -885 / -83 / +1015 | 19 | 19 | 18 | 1.24 / 3.28 | 1.30 / 3.71 |

Top 5 by 24h median: XMRUSDT, XAUTUSDT, BTCUSDT, XRPUSDT, ETHUSDT. Bottom 5: 1000PEPEUSDT, INJUSDT, ZECUSDT, NEARUSDT, ZKCUSDT. Calmest 5 (low-vol): XAUTUSDT, BTCUSDT, DOGEUSDT, ETHUSDT, XRPUSDT.

Forward scores (H-XS4; evaluated once at 60 daily origins):

- 4h origins: n 11, mean IC v4 +0.018, low-vol +0.018, diff -0.000
- 24h origins: n 3, mean IC v4 +0.213, low-vol +0.228, diff -0.015
- 1h vol pinball: n 45, v4 +0.111, log-HAR-t +0.114
