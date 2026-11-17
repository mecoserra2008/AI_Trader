# xs.v4 shadow line (not LATEST.md)

_SHADOW, context only, not a signal (R-XS-2 / R-XS-3); no orders; the decision sessions do not read it. H-XS4 forward test (rank IC vs the low-vol ranking). Updated 2026-11-17T08:02 UTC, forecast hour S = 2026-11-17T08:00 UTC._

- universe now: 20 perps pass 24h turnover >= 20 M USD and spread <= 5 bp; present in the panel: 20
- absent: none
- model: trained 2026-11-16T00:35, data to 2026-11-16T00:00, weights 787be7ca6d62addf

| token | sector | 24h rel q10 / q50 / q90 (bp) | rank 24h | rank 4h | low-vol rank | 1h |r| q50 / q90 % (v4) | log-HAR-t q50 / q90 % |
|---|---|---|---|---|---|---|---|
| XMRUSDT | privacy | -448 / +14 / +484 | 1 | 7 | 11 | 0.46 / 1.17 | 0.51 / 1.46 |
| SOLUSDT | l1 | -155 / +8 / +164 | 2 | 2 | 4 | 0.33 / 0.84 | 0.33 / 0.95 |
| AVAXUSDT | l1 | -172 / +7 / +169 | 3 | 4 | 6 | 0.36 / 0.93 | 0.37 / 1.05 |
| XRPUSDT | payments | -191 / +4 / +179 | 4 | 3 | 2 | 0.33 / 0.84 | 0.38 / 1.08 |
| ETHUSDT | l1 | -172 / -2 / +144 | 5 | 1 | 3 | 0.30 / 0.79 | 0.33 / 0.93 |
| BTCUSDT | btc | -241 / -3 / +198 | 6 | 5 | 1 | 0.24 / 0.62 | 0.24 / 0.70 |
| DOGEUSDT | meme | -162 / -6 / +135 | 7 | 6 | 5 | 0.32 / 0.83 | 0.32 / 0.91 |
| UNIUSDT | defi | -240 / -11 / +245 | 8 | 11 | 10 | 0.46 / 1.19 | 0.41 / 1.16 |
| 1000PEPEUSDT | meme | -302 / -12 / +307 | 9 | 14 | 13 | 0.51 / 1.31 | 0.51 / 1.44 |
| WLDUSDT | ai | -211 / -14 / +190 | 10 | 9 | 7 | 0.45 / 1.16 | 0.42 / 1.20 |
| SUIUSDT | l1 | -168 / -16 / +130 | 11 | 8 | 8 | 0.40 / 1.01 | 0.42 / 1.18 |
| AAVEUSDT | defi | -185 / -16 / +156 | 12 | 10 | 9 | 0.41 / 1.05 | 0.43 / 1.23 |
| ENAUSDT | defi | -301 / -20 / +276 | 13 | 12 | 12 | 0.55 / 1.44 | 0.54 / 1.53 |
| HYPEUSDT | exchange | -276 / -28 / +244 | 14 | 13 | 14 | 0.48 / 1.21 | 0.53 / 1.52 |
| ZECUSDT | privacy | -291 / -29 / +267 | 15 | 15 | 15 | 0.60 / 1.55 | 0.61 / 1.73 |
| ASTERUSDT | exchange | -425 / -43 / +443 | 16 | 16 | 19 | 0.74 / 1.90 | 0.92 / 2.61 |
| NEARUSDT | l1 | -301 / -46 / +271 | 17 | 17 | 16 | 0.59 / 1.51 | 0.62 / 1.76 |
| INJUSDT | l1 | -384 / -84 / +377 | 18 | 18 | 17 | 0.76 / 1.90 | 0.72 / 2.06 |
| LITUSDT | other | -541 / -87 / +628 | 19 | 19 | 18 | 0.83 / 2.22 | 0.96 / 2.72 |
| RAREUSDT | other | -2219 / -250 / +2981 | 20 | 20 | 20 | 3.03 / 7.83 | 3.08 / 8.76 |

Top 5 by 24h median: XMRUSDT, SOLUSDT, AVAXUSDT, XRPUSDT, ETHUSDT. Bottom 5: ASTERUSDT, NEARUSDT, INJUSDT, LITUSDT, RAREUSDT. Calmest 5 (low-vol): BTCUSDT, XRPUSDT, ETHUSDT, SOLUSDT, DOGEUSDT.

Forward scores (H-XS4; evaluated once at 60 daily origins):

- 4h origins: n 5, mean IC v4 +0.028, low-vol +0.051, diff -0.022
- 24h origins: n 1, mean IC v4 +0.036, low-vol +0.206, diff -0.170
- 1h vol pinball: n 22, v4 +0.105, log-HAR-t +0.109
