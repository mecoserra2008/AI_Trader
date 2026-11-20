# xs.v4 shadow line (not LATEST.md)

_SHADOW, context only, not a signal (R-XS-2 / R-XS-3); no orders; the decision sessions do not read it. H-XS4 forward test (rank IC vs the low-vol ranking). Updated 2026-11-20T04:02 UTC, forecast hour S = 2026-11-20T04:00 UTC._

- universe now: 26 perps pass 24h turnover >= 20 M USD and spread <= 5 bp; present in the panel: 26
- absent: none
- model: trained 2026-11-16T00:35, data to 2026-11-16T00:00, weights 787be7ca6d62addf

| token | sector | 24h rel q10 / q50 / q90 (bp) | rank 24h | rank 4h | low-vol rank | 1h |r| q50 / q90 % (v4) | log-HAR-t q50 / q90 % |
|---|---|---|---|---|---|---|---|
| XAUTUSDT | other | -383 / -7 / +432 | 1 | 6 | 1 | 0.26 / 0.72 | 0.17 / 0.49 |
| AKEUSDT | ai | -2660 / -10 / +3611 | 2 | 26 | 23 | 1.93 / 5.44 | 2.08 / 5.91 |
| AAVEUSDT | defi | -298 / -17 / +324 | 3 | 1 | 10 | 0.58 / 1.59 | 0.52 / 1.47 |
| XMRUSDT | privacy | -505 / -18 / +584 | 4 | 14 | 5 | 0.49 / 1.37 | 0.39 / 1.10 |
| DOGEUSDT | meme | -229 / -20 / +232 | 5 | 3 | 6 | 0.43 / 1.20 | 0.35 / 0.99 |
| AVAXUSDT | l1 | -264 / -23 / +280 | 6 | 5 | 7 | 0.46 / 1.27 | 0.38 / 1.08 |
| SUIUSDT | l1 | -219 / -24 / +209 | 7 | 2 | 8 | 0.47 / 1.31 | 0.41 / 1.17 |
| ETHUSDT | l1 | -280 / -28 / +289 | 8 | 10 | 4 | 0.45 / 1.23 | 0.38 / 1.07 |
| BTCUSDT | btc | -318 / -30 / +348 | 9 | 19 | 2 | 0.42 / 1.12 | 0.26 / 0.74 |
| UNIUSDT | defi | -357 / -33 / +368 | 10 | 7 | 11 | 0.68 / 1.89 | 0.59 / 1.68 |
| XRPUSDT | payments | -339 / -38 / +396 | 11 | 22 | 3 | 0.46 / 1.26 | 0.32 / 0.90 |
| ENAUSDT | defi | -426 / -39 / +475 | 12 | 16 | 13 | 0.70 / 1.98 | 0.60 / 1.69 |
| 1000PEPEUSDT | meme | -440 / -40 / +433 | 13 | 4 | 21 | 0.69 / 1.96 | 0.65 / 1.84 |
| SOLUSDT | l1 | -288 / -41 / +295 | 14 | 18 | 9 | 0.52 / 1.43 | 0.38 / 1.08 |
| INJUSDT | l1 | -384 / -42 / +388 | 15 | 12 | 19 | 0.68 / 1.87 | 0.65 / 1.86 |
| WLDUSDT | ai | -368 / -44 / +350 | 16 | 8 | 14 | 0.65 / 1.82 | 0.57 / 1.62 |
| HYPEUSDT | exchange | -316 / -44 / +304 | 17 | 15 | 12 | 0.59 / 1.65 | 0.49 / 1.41 |
| LITUSDT | other | -466 / -44 / +481 | 18 | 17 | 18 | 0.78 / 2.15 | 0.70 / 1.99 |
| INITUSDT | l1 | -1096 / -45 / +1268 | 19 | 20 | 22 | 1.06 / 3.13 | 1.00 / 2.86 |
| ZECUSDT | privacy | -324 / -46 / +302 | 20 | 9 | 16 | 0.73 / 1.97 | 0.63 / 1.78 |
| PUMPFUNUSDT | meme | -391 / -47 / +392 | 21 | 11 | 15 | 0.68 / 1.89 | 0.60 / 1.71 |
| STRKUSDT | l2 | -343 / -48 / +318 | 22 | 13 | 17 | 0.68 / 1.92 | 0.65 / 1.84 |
| NEARUSDT | l1 | -389 / -56 / +382 | 23 | 21 | 20 | 0.77 / 2.07 | 0.73 / 2.08 |
| SKYUSDT | other | -1093 / -62 / +1206 | 24 | 23 | 24 | 1.13 / 3.15 | 1.09 / 3.11 |
| MMTUSDT | other | -2153 / -135 / +2647 | 25 | 25 | 26 | 2.93 / 7.91 | 2.38 / 6.76 |
| PIEVERSEUSDT | ai | -1691 / -169 / +1955 | 26 | 24 | 25 | 1.91 / 5.16 | 1.95 / 5.54 |

Top 5 by 24h median: XAUTUSDT, AKEUSDT, AAVEUSDT, XMRUSDT, DOGEUSDT. Bottom 5: STRKUSDT, NEARUSDT, SKYUSDT, MMTUSDT, PIEVERSEUSDT. Calmest 5 (low-vol): XAUTUSDT, BTCUSDT, XRPUSDT, ETHUSDT, XMRUSDT.

Forward scores (H-XS4; evaluated once at 60 daily origins):

- 4h origins: n 13, mean IC v4 +0.025, low-vol +0.008, diff +0.017
- 24h origins: n 4, mean IC v4 +0.262, low-vol +0.200, diff +0.061
- 1h vol pinball: n 52, v4 +0.115, log-HAR-t +0.117
