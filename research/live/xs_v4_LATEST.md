# xs.v4 shadow line (not LATEST.md)

_SHADOW, context only, not a signal (R-XS-2 / R-XS-3); no orders; the decision sessions do not read it. H-XS4 forward test (rank IC vs the low-vol ranking). Updated 2026-11-28T13:02 UTC, forecast hour S = 2026-11-28T13:00 UTC._

- universe now: 22 perps pass 24h turnover >= 20 M USD and spread <= 5 bp; present in the panel: 22
- absent: none
- model: trained 2026-11-16T00:35, data to 2026-11-16T00:00, weights 787be7ca6d62addf

| token | sector | 24h rel q10 / q50 / q90 (bp) | rank 24h | rank 4h | low-vol rank | 1h |r| q50 / q90 % (v4) | log-HAR-t q50 / q90 % |
|---|---|---|---|---|---|---|---|
| BLURUSDT | other | -1987 / +9 / +2527 | 1 | 15 | 21 | 2.02 / 5.36 | 1.99 / 5.66 |
| BTCUSDT | btc | -203 / +0 / +223 | 2 | 1 | 1 | 0.21 / 0.59 | 0.16 / 0.45 |
| INITUSDT | l1 | -1109 / -4 / +1424 | 3 | 7 | 19 | 1.29 / 3.49 | 1.53 / 4.34 |
| 1000PEPEUSDT | meme | -233 / -4 / +237 | 4 | 2 | 13 | 0.41 / 1.12 | 0.41 / 1.15 |
| HYPEUSDT | exchange | -227 / -6 / +251 | 5 | 4 | 15 | 0.46 / 1.22 | 0.45 / 1.29 |
| ETHUSDT | l1 | -182 / -7 / +188 | 6 | 3 | 2 | 0.25 / 0.70 | 0.19 / 0.53 |
| PORTALUSDT | other | -3491 / -11 / +4986 | 7 | 19 | 22 | 4.84 / 13.06 | 3.96 / 11.26 |
| XRPUSDT | payments | -213 / -13 / +223 | 8 | 10 | 3 | 0.30 / 0.82 | 0.26 / 0.73 |
| DOGEUSDT | meme | -185 / -14 / +171 | 9 | 5 | 5 | 0.30 / 0.82 | 0.28 / 0.80 |
| SUIUSDT | l1 | -175 / -15 / +160 | 10 | 6 | 6 | 0.32 / 0.90 | 0.29 / 0.83 |
| SOLUSDT | l1 | -193 / -16 / +188 | 11 | 8 | 4 | 0.29 / 0.81 | 0.24 / 0.67 |
| ENAUSDT | defi | -235 / -16 / +233 | 12 | 9 | 14 | 0.42 / 1.15 | 0.41 / 1.16 |
| MNTUSDT | l2 | -381 / -18 / +491 | 13 | 18 | 10 | 0.44 / 1.11 | 0.39 / 1.10 |
| LTCUSDT | payments | -248 / -21 / +275 | 14 | 12 | 7 | 0.34 / 0.91 | 0.31 / 0.88 |
| WLDUSDT | ai | -226 / -22 / +226 | 15 | 14 | 9 | 0.36 / 1.00 | 0.35 / 1.00 |
| NEARUSDT | l1 | -236 / -24 / +223 | 16 | 13 | 11 | 0.42 / 1.15 | 0.42 / 1.18 |
| LINKUSDT | oracle | -199 / -25 / +189 | 17 | 11 | 8 | 0.32 / 0.87 | 0.30 / 0.85 |
| UNIUSDT | defi | -229 / -31 / +224 | 18 | 16 | 12 | 0.41 / 1.12 | 0.43 / 1.23 |
| ZECUSDT | privacy | -343 / -39 / +338 | 19 | 17 | 16 | 0.64 / 1.73 | 0.68 / 1.94 |
| XMRUSDT | privacy | -717 / -76 / +937 | 20 | 20 | 17 | 0.77 / 2.01 | 0.71 / 2.02 |
| SSVUSDT | other | -849 / -110 / +1007 | 21 | 21 | 20 | 1.39 / 3.63 | 1.33 / 3.80 |
| PIEVERSEUSDT | ai | -1207 / -117 / +1724 | 22 | 22 | 18 | 1.70 / 4.64 | 1.65 / 4.70 |

Top 5 by 24h median: BLURUSDT, BTCUSDT, INITUSDT, 1000PEPEUSDT, HYPEUSDT. Bottom 5: UNIUSDT, ZECUSDT, XMRUSDT, SSVUSDT, PIEVERSEUSDT. Calmest 5 (low-vol): BTCUSDT, ETHUSDT, XRPUSDT, SOLUSDT, DOGEUSDT.

Forward scores (H-XS4; evaluated once at 60 daily origins):

- 4h origins: n 33, mean IC v4 +0.063, low-vol +0.076, diff -0.013
- 24h origins: n 10, mean IC v4 +0.117, low-vol +0.105, diff +0.012
- 1h vol pinball: n 127, v4 +0.118, log-HAR-t +0.120
