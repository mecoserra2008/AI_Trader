# xs.v4 shadow line (not LATEST.md)

_SHADOW, context only, not a signal (R-XS-2 / R-XS-3); no orders; the decision sessions do not read it. H-XS4 forward test (rank IC vs the low-vol ranking). Updated 2026-11-28T08:02 UTC, forecast hour S = 2026-11-28T08:00 UTC._

- universe now: 24 perps pass 24h turnover >= 20 M USD and spread <= 5 bp; present in the panel: 24
- absent: none
- model: trained 2026-11-16T00:35, data to 2026-11-16T00:00, weights 787be7ca6d62addf

| token | sector | 24h rel q10 / q50 / q90 (bp) | rank 24h | rank 4h | low-vol rank | 1h |r| q50 / q90 % (v4) | log-HAR-t q50 / q90 % |
|---|---|---|---|---|---|---|---|
| XAUTUSDT | other | -227 / +7 / +293 | 1 | 1 | 1 | 0.10 / 0.24 | 0.08 / 0.23 |
| ETHUSDT | l1 | -128 / +3 / +126 | 2 | 2 | 3 | 0.20 / 0.53 | 0.16 / 0.47 |
| BTCUSDT | btc | -159 / +3 / +170 | 3 | 4 | 2 | 0.18 / 0.48 | 0.14 / 0.40 |
| XRPUSDT | payments | -132 / +2 / +132 | 4 | 3 | 4 | 0.22 / 0.59 | 0.24 / 0.67 |
| WLDUSDT | ai | -180 / -2 / +193 | 5 | 7 | 11 | 0.33 / 0.87 | 0.35 / 0.99 |
| SOLUSDT | l1 | -127 / -3 / +116 | 6 | 5 | 5 | 0.23 / 0.63 | 0.23 / 0.64 |
| XMRUSDT | privacy | -520 / -3 / +616 | 7 | 12 | 19 | 0.65 / 1.71 | 0.69 / 1.96 |
| DOGEUSDT | meme | -138 / -7 / +123 | 8 | 6 | 6 | 0.25 / 0.67 | 0.23 / 0.66 |
| LTCUSDT | payments | -184 / -8 / +200 | 9 | 8 | 8 | 0.29 / 0.75 | 0.28 / 0.78 |
| LINKUSDT | oracle | -160 / -11 / +162 | 10 | 9 | 10 | 0.32 / 0.85 | 0.28 / 0.81 |
| 1000PEPEUSDT | meme | -208 / -15 / +201 | 11 | 10 | 15 | 0.38 / 1.01 | 0.36 / 1.03 |
| UNIUSDT | defi | -194 / -16 / +202 | 12 | 13 | 14 | 0.40 / 1.06 | 0.38 / 1.07 |
| MNTUSDT | l2 | -332 / -17 / +409 | 13 | 16 | 12 | 0.43 / 1.08 | 0.37 / 1.06 |
| AAVEUSDT | defi | -190 / -22 / +172 | 14 | 11 | 9 | 0.32 / 0.87 | 0.30 / 0.84 |
| ENAUSDT | defi | -219 / -22 / +229 | 15 | 15 | 16 | 0.42 / 1.13 | 0.42 / 1.20 |
| HYPEUSDT | exchange | -232 / -26 / +251 | 16 | 18 | 17 | 0.47 / 1.24 | 0.47 / 1.33 |
| SUIUSDT | l1 | -178 / -31 / +155 | 17 | 17 | 7 | 0.31 / 0.85 | 0.29 / 0.81 |
| NEARUSDT | l1 | -240 / -33 / +237 | 18 | 19 | 13 | 0.40 / 1.09 | 0.38 / 1.07 |
| BLURUSDT | other | -1827 / -38 / +1985 | 19 | 14 | 23 | 1.56 / 4.17 | 1.57 / 4.48 |
| PORTALUSDT | other | -4466 / -38 / +6578 | 20 | 24 | 24 | 5.72 / 15.76 | 4.58 / 13.03 |
| PIEVERSEUSDT | ai | -858 / -41 / +985 | 21 | 20 | 20 | 1.20 / 3.30 | 1.36 / 3.86 |
| INITUSDT | l1 | -1467 / -52 / +1848 | 22 | 22 | 21 | 1.52 / 4.09 | 1.66 / 4.71 |
| ZECUSDT | privacy | -312 / -54 / +283 | 23 | 21 | 18 | 0.59 / 1.56 | 0.65 / 1.84 |
| CFGUSDT | other | -906 / -122 / +1051 | 24 | 23 | 22 | 1.47 / 4.04 | 1.45 / 4.13 |

Top 5 by 24h median: XAUTUSDT, ETHUSDT, BTCUSDT, XRPUSDT, WLDUSDT. Bottom 5: PORTALUSDT, PIEVERSEUSDT, INITUSDT, ZECUSDT, CFGUSDT. Calmest 5 (low-vol): XAUTUSDT, BTCUSDT, ETHUSDT, XRPUSDT, SOLUSDT.

Forward scores (H-XS4; evaluated once at 60 daily origins):

- 4h origins: n 32, mean IC v4 +0.059, low-vol +0.077, diff -0.018
- 24h origins: n 10, mean IC v4 +0.117, low-vol +0.105, diff +0.012
- 1h vol pinball: n 122, v4 +0.120, log-HAR-t +0.122
