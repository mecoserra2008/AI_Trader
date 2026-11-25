# xs.v4 shadow line (not LATEST.md)

_SHADOW, context only, not a signal (R-XS-2 / R-XS-3); no orders; the decision sessions do not read it. H-XS4 forward test (rank IC vs the low-vol ranking). Updated 2026-11-25T08:02 UTC, forecast hour S = 2026-11-25T08:00 UTC._

- universe now: 20 perps pass 24h turnover >= 20 M USD and spread <= 5 bp; present in the panel: 20
- absent: none
- model: trained 2026-11-16T00:35, data to 2026-11-16T00:00, weights 787be7ca6d62addf

| token | sector | 24h rel q10 / q50 / q90 (bp) | rank 24h | rank 4h | low-vol rank | 1h |r| q50 / q90 % (v4) | log-HAR-t q50 / q90 % |
|---|---|---|---|---|---|---|---|
| BTCUSDT | btc | -185 / +9 / +203 | 1 | 1 | 2 | 0.32 / 0.82 | 0.29 / 0.83 |
| SOLUSDT | l1 | -184 / +4 / +190 | 2 | 2 | 8 | 0.41 / 1.05 | 0.39 / 1.10 |
| XAUTUSDT | other | -356 / +3 / +405 | 3 | 15 | 1 | 0.20 / 0.55 | 0.17 / 0.48 |
| AAVEUSDT | defi | -225 / +3 / +220 | 4 | 4 | 7 | 0.45 / 1.21 | 0.45 / 1.29 |
| ETHUSDT | l1 | -162 / +2 / +155 | 5 | 3 | 5 | 0.36 / 0.94 | 0.37 / 1.05 |
| ENAUSDT | defi | -245 / +1 / +247 | 6 | 8 | 11 | 0.50 / 1.36 | 0.51 / 1.44 |
| AVAXUSDT | l1 | -173 / -1 / +167 | 7 | 5 | 6 | 0.39 / 1.03 | 0.38 / 1.08 |
| XRPUSDT | payments | -188 / -2 / +193 | 8 | 10 | 3 | 0.36 / 0.95 | 0.34 / 0.96 |
| SUIUSDT | l1 | -168 / -3 / +169 | 9 | 7 | 9 | 0.41 / 1.08 | 0.40 / 1.14 |
| DOGEUSDT | meme | -175 / -4 / +171 | 10 | 9 | 4 | 0.38 / 1.00 | 0.38 / 1.09 |
| WLDUSDT | ai | -232 / -4 / +241 | 11 | 11 | 13 | 0.48 / 1.25 | 0.47 / 1.34 |
| HYPEUSDT | exchange | -239 / -4 / +234 | 12 | 13 | 10 | 0.46 / 1.21 | 0.40 / 1.12 |
| UNIUSDT | defi | -211 / -6 / +192 | 13 | 6 | 12 | 0.48 / 1.26 | 0.53 / 1.51 |
| NEARUSDT | l1 | -254 / -6 / +267 | 14 | 12 | 14 | 0.54 / 1.43 | 0.53 / 1.50 |
| INJUSDT | l1 | -325 / -7 / +341 | 15 | 14 | 16 | 0.61 / 1.60 | 0.57 / 1.61 |
| ZECUSDT | privacy | -319 / -18 / +322 | 16 | 16 | 15 | 0.61 / 1.64 | 0.55 / 1.58 |
| PIEVERSEUSDT | ai | -934 / -21 / +1127 | 17 | 17 | 20 | 1.22 / 3.30 | 1.28 / 3.64 |
| VVVUSDT | ai | -524 / -48 / +617 | 18 | 20 | 18 | 0.79 / 2.13 | 0.65 / 1.85 |
| ZROUSDT | infra | -467 / -48 / +461 | 19 | 18 | 19 | 0.89 / 2.33 | 0.78 / 2.22 |
| 1000PEPEUSDT | meme | -350 / -59 / +308 | 20 | 19 | 17 | 0.66 / 1.75 | 0.62 / 1.76 |

Top 5 by 24h median: BTCUSDT, SOLUSDT, XAUTUSDT, AAVEUSDT, ETHUSDT. Bottom 5: ZECUSDT, PIEVERSEUSDT, VVVUSDT, ZROUSDT, 1000PEPEUSDT. Calmest 5 (low-vol): XAUTUSDT, BTCUSDT, XRPUSDT, DOGEUSDT, ETHUSDT.

Forward scores (H-XS4; evaluated once at 60 daily origins):

- 4h origins: n 22, mean IC v4 +0.018, low-vol +0.075, diff -0.057
- 24h origins: n 7, mean IC v4 +0.220, low-vol +0.211, diff +0.009
- 1h vol pinball: n 85, v4 +0.123, log-HAR-t +0.126
