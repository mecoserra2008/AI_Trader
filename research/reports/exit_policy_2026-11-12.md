# Exit policies for the trend family (H-PD6) — 2026-11-12

Agent I, Program 2 follow-up. Design fixed in the docstring of `ai_trader.research.exit_policy_study` and in the diary (2026-11-12, 'H-PD6 exit-policy test: design fixed before evaluation') before any result was computed. Units: pnl_d (daily-sigma units). Segments split at 2026-07-01; dates 2020-06-18 .. 2026-11-09; symbols L 23, T 15.

Rule: at each close, exit if the close has retraced more than k x sigma_t from the best close since entry (sigma_t = sigma_risk_next, the robust one-bar stop sigma at the strategy's bar size, known at that close); after such an exit, no re-entry until the raw signal turns flat or flips. k in {1.5, 2.5, 3.5} (registered; no other value was run). Costs 7.5 bps per side (evaluate.strategy_pnl).

Baseline check vs the stored strategy panel: 347544 of 347544 trend rows matched, max |diff| 0.00e+00, only-panel 0, only-recomputed 0.

## Verdicts

- llm_in_sample: **contradicts**; clean_holdout: **supports**; pooled: **contradicts** (registered test on k = 2.5).
- In-sample the trail lowers the trend book's net P&L at every k: k 1.5 -0.0208/day (t -2.31), k 2.5 -0.0195/day (t -2.30), k 3.5 -0.0167/day (t -2.14); Sharpe 0.86 -> -1.13 / -0.27 / 0.27.
- In the holdout it raises it, not significantly: k 1.5 +0.0177 (t 1.42), k 2.5 +0.0187 (t 1.63), k 3.5 +0.0186 (t 1.75), while the own-exit trend book itself had Sharpe -2.21; the trail mostly cuts exposure to a losing book.
- Decomposition: the share of the favourable excursion kept barely moves in-sample (capture 0.316 -> 0.318/0.320/0.323; holdout 0.293 -> 0.354/0.352/0.350). The trail shortens trades (mean hold 70 h -> 9/15/21 h) and shrinks the excursion pool (sum MFE 55021 -> 15669/23664/30834), so exit leakage falls in absolute terms but not as a share; costs per unit of kept gross rise (0.12 -> 0.25/0.19/0.16).
- Mean exposure (share of bars in a position, trend symbol-days): in-sample 0.86 -> 0.10/0.17/0.25, holdout 0.85 -> 0.11/0.18/0.26; the daily difference is almost the negative of the own-exit book (corr -0.99 in-sample at k 2.5), so the paired test mostly asks whether the trend book made money in that segment.
- Trade counts are almost unchanged (86146 -> 86159/86159/86159 in-sample) because re-entry needs a fresh signal change; 55%/47%/40% of trades end on the trail.

## Paired daily book difference (trailing - own exits), equal-weight trend sleeves L+T

corr and skew (labelled extension, descriptive): correlation of the daily difference with the own-exit book, and the skewness of the difference.

| segment | days | k | diff/day | NW t | Holm p (one-sided) | Sharpe own -> trail | days better / worse | corr(diff, own) | skew(diff) |
|---|---|---|---|---|---|---|---|---|---|
| llm_in_sample | 2204 | 1.5 | -0.0208 | -2.31 | 1.000 | 0.86 -> -1.13 | 0.54 / 0.46 | -1.00 | -2.31 |
| llm_in_sample | 2204 | 2.5 | -0.0195 | -2.30 | 1.000 | 0.86 -> -0.27 | 0.53 / 0.47 | -0.99 | -2.04 |
| llm_in_sample | 2204 | 3.5 | -0.0167 | -2.14 | 1.000 | 0.86 -> 0.27 | 0.51 / 0.49 | -0.96 | -2.00 |
| clean_holdout | 132 | 1.5 | +0.0177 | 1.42 | 0.121 | -2.21 -> -2.83 | 0.55 / 0.45 | -0.99 | -0.22 |
| clean_holdout | 132 | 2.5 | +0.0187 | 1.63 | 0.121 | -2.21 -> -1.31 | 0.55 / 0.45 | -0.97 | -0.04 |
| clean_holdout | 132 | 3.5 | +0.0186 | 1.75 | 0.121 | -2.21 -> -0.87 | 0.57 / 0.43 | -0.91 | -0.01 |
| pooled | 2336 | 1.5 | -0.0186 | -2.17 | 1.000 | 0.77 -> -1.21 | 0.54 / 0.46 | -1.00 | -2.35 |
| pooled | 2336 | 2.5 | -0.0173 | -2.15 | 1.000 | 0.77 -> -0.32 | 0.53 / 0.47 | -0.99 | -2.08 |
| pooled | 2336 | 3.5 | -0.0147 | -1.97 | 1.000 | 0.77 -> 0.22 | 0.51 / 0.49 | -0.96 | -2.04 |

Robustness books (diff/day, NW t):

| book | segment | days | k1.5 | k2.5 | k3.5 | Sharpe own |
|---|---|---|---|---|---|---|
| L | llm_in_sample | 2204 | -0.0206 (-2.3) | -0.0193 (-2.3) | -0.0166 (-2.1) | 0.86 |
| L | clean_holdout | 132 | +0.0148 (1.1) | +0.0171 (1.4) | +0.0181 (1.7) | -1.65 |
| L | pooled | 2336 | -0.0186 (-2.2) | -0.0172 (-2.1) | -0.0146 (-2.0) | 0.78 |
| T | llm_in_sample | 13 | -0.1203 (-1.2) | -0.1108 (-1.2) | -0.0903 (-1.2) | - |
| T | clean_holdout | 132 | +0.0207 (1.2) | +0.0204 (1.3) | +0.0190 (1.2) | -2.01 |
| T | pooled | 145 | +0.0080 (0.5) | +0.0087 (0.5) | +0.0092 (0.6) | -1.01 |
| ex_compression | llm_in_sample | 2204 | -0.0187 (-2.2) | -0.0173 (-2.2) | -0.0140 (-2.0) | 0.78 |
| ex_compression | clean_holdout | 132 | +0.0215 (1.5) | +0.0225 (1.8) | +0.0223 (2.0) | -2.47 |
| ex_compression | pooled | 2336 | -0.0164 (-2.0) | -0.0150 (-2.0) | -0.0119 (-1.8) | 0.67 |

## H-PD4 decomposition on trend-family trades, by variant

| segment | variant | trades | sum MFE | kept | exit leak | dir loss | costs | net | (C+E)/MFE | capture | hit gross | mean / median hold (h) | closed by trail |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| llm_in_sample | orig | 86146 | 55021.0 | 17373.4 | 37647.6 | 14619.2 | 2085.8 | 668.3 | 0.722 | 0.316 | 0.201 | 69.7 / 11 | 0.00 |
| llm_in_sample | k1.5 | 86159 | 15668.5 | 4981.2 | 10687.3 | 4622.3 | 1250.1 | -891.2 | 0.762 | 0.318 | 0.243 | 8.9 / 5 | 0.55 |
| llm_in_sample | k2.5 | 86159 | 23663.7 | 7573.2 | 16090.5 | 6717.6 | 1438.7 | -583.1 | 0.741 | 0.320 | 0.226 | 14.8 / 7 | 0.47 |
| llm_in_sample | k3.5 | 86159 | 30834.1 | 9949.2 | 20884.8 | 8501.5 | 1581.9 | -134.3 | 0.729 | 0.323 | 0.218 | 21.4 / 9 | 0.40 |
| clean_holdout | orig | 10257 | 5195.5 | 1523.2 | 3672.3 | 1806.9 | 397.7 | -681.4 | 0.783 | 0.293 | 0.196 | 50.3 / 11 | 0.00 |
| clean_holdout | k1.5 | 10470 | 1989.1 | 703.7 | 1285.4 | 585.6 | 255.1 | -137.1 | 0.774 | 0.354 | 0.259 | 9.3 / 5 | 0.56 |
| clean_holdout | k2.5 | 10455 | 2959.4 | 1041.6 | 1917.9 | 855.2 | 290.1 | -103.8 | 0.746 | 0.352 | 0.237 | 15.2 / 7 | 0.48 |
| clean_holdout | k3.5 | 10446 | 3812.2 | 1335.9 | 2476.3 | 1106.2 | 318.3 | -88.7 | 0.733 | 0.350 | 0.227 | 21.3 / 9 | 0.40 |
| pooled | orig | 96403 | 60216.6 | 18896.6 | 41320.0 | 16426.2 | 2483.5 | -13.1 | 0.727 | 0.314 | 0.200 | 67.7 / 11 | 0.00 |
| pooled | k1.5 | 96629 | 17657.6 | 5684.9 | 11972.7 | 5207.9 | 1505.2 | -1028.3 | 0.763 | 0.322 | 0.245 | 8.9 / 5 | 0.55 |
| pooled | k2.5 | 96614 | 26623.2 | 8614.8 | 18008.4 | 7572.8 | 1728.9 | -686.9 | 0.741 | 0.324 | 0.227 | 14.9 / 7 | 0.47 |
| pooled | k3.5 | 96605 | 34646.2 | 11285.1 | 23361.1 | 9607.8 | 1900.3 | -222.9 | 0.729 | 0.326 | 0.219 | 21.4 / 9 | 0.40 |

By strategy (llm_in_sample / clean_holdout): net (sum pnl_d over trades), capture, mean hold (h), trades:

| strategy | variant | IS net | IS capture | IS hold h | IS trades | HO net | HO capture | HO hold h | HO trades |
|---|---|---|---|---|---|---|---|---|---|
| tsmom | orig | -434.3 | 0.305 | 17 | 53939 | -196.2 | 0.289 | 17 | 6369 |
| tsmom | k1.5 | -563.3 | 0.308 | 5 | 53939 | -79.4 | 0.345 | 6 | 6394 |
| tsmom | k2.5 | -511.4 | 0.313 | 8 | 53939 | -69.8 | 0.344 | 9 | 6392 |
| tsmom | k3.5 | -357.6 | 0.318 | 11 | 53939 | -60.3 | 0.343 | 11 | 6390 |
| donchian_breakout | orig | 304.5 | 0.335 | 106 | 5679 | -40.2 | 0.368 | 96 | 705 |
| donchian_breakout | k1.5 | -112.1 | 0.328 | 8 | 5679 | -21.2 | 0.369 | 8 | 725 |
| donchian_breakout | k2.5 | -61.5 | 0.333 | 15 | 5679 | 3.3 | 0.381 | 16 | 723 |
| donchian_breakout | k3.5 | -67.1 | 0.331 | 24 | 5679 | 25.1 | 0.401 | 25 | 723 |
| ema_stack | orig | 256.1 | 0.334 | 44 | 14714 | -116.8 | 0.304 | 42 | 1784 |
| ema_stack | k1.5 | -243.6 | 0.321 | 8 | 14714 | -36.5 | 0.337 | 9 | 1809 |
| ema_stack | k2.5 | -166.5 | 0.320 | 15 | 14714 | -43.9 | 0.329 | 15 | 1807 |
| ema_stack | k3.5 | -47.1 | 0.326 | 22 | 14714 | -50.3 | 0.332 | 22 | 1807 |
| compression_breakout | orig | -615.1 | 0.004 | 482 | 818 | -112.8 | 0.003 | 80 | 143 |
| compression_breakout | k1.5 | -34.5 | 0.348 | 7 | 831 | -10.1 | 0.377 | 9 | 158 |
| compression_breakout | k2.5 | -8.7 | 0.348 | 14 | 831 | -5.1 | 0.356 | 15 | 158 |
| compression_breakout | k3.5 | -22.9 | 0.323 | 21 | 831 | -6.6 | 0.348 | 22 | 158 |
| geo_struct | orig | 26.9 | 0.349 | 420 | 1927 | -52.4 | 0.345 | 286 | 213 |
| geo_struct | k1.5 | 30.6 | 0.368 | 33 | 1927 | -6.0 | 0.349 | 32 | 239 |
| geo_struct | k2.5 | 36.5 | 0.351 | 60 | 1927 | -7.9 | 0.344 | 58 | 235 |
| geo_struct | k3.5 | 73.5 | 0.357 | 91 | 1927 | -9.1 | 0.326 | 83 | 233 |
| geo_pivot_bo | orig | 610.9 | 0.368 | 707 | 1310 | -70.5 | 0.214 | 458 | 139 |
| geo_pivot_bo | k1.5 | 41.6 | 0.356 | 30 | 1310 | 8.7 | 0.401 | 28 | 177 |
| geo_pivot_bo | k2.5 | 78.3 | 0.339 | 59 | 1310 | -2.2 | 0.351 | 53 | 176 |
| geo_pivot_bo | k3.5 | 166.7 | 0.343 | 99 | 1310 | -7.3 | 0.329 | 88 | 175 |
| geo_flag_bo | orig | 89.8 | 0.317 | 664 | 1160 | -50.2 | 0.243 | 381 | 117 |
| geo_flag_bo | k1.5 | 3.0 | 0.325 | 29 | 1160 | -3.4 | 0.351 | 32 | 147 |
| geo_flag_bo | k2.5 | 26.4 | 0.331 | 58 | 1160 | 1.1 | 0.387 | 59 | 145 |
| geo_flag_bo | k3.5 | 46.4 | 0.325 | 95 | 1160 | -0.4 | 0.347 | 94 | 143 |
| zoo_ensemble | orig | 429.5 | 0.318 | 141 | 6599 | -42.4 | 0.257 | 110 | 787 |
| zoo_ensemble | k1.5 | -13.0 | 0.273 | 26 | 6599 | 10.8 | 0.366 | 25 | 821 |
| zoo_ensemble | k2.5 | 23.7 | 0.286 | 40 | 6599 | 20.7 | 0.385 | 38 | 819 |
| zoo_ensemble | k3.5 | 73.9 | 0.291 | 56 | 6599 | 20.3 | 0.356 | 52 | 817 |

## Per sleeve (diff/day x100, NW t)

| sleeve | segment | k1.5 | k2.5 | k3.5 | Sharpe own -> k2.5 |
|---|---|---|---|---|---|
| compression_breakout|L | llm_in_sample | -3.66 (-2.0) | -3.57 (-2.0) | -3.63 (-2.0) | 0.88 -> 0.08 |
| compression_breakout|L | clean_holdout | +0.71 (0.4) | +0.81 (0.4) | +0.76 (0.4) | -0.55 -> 0.79 |
| compression_breakout|T | clean_holdout | -2.45 (-0.9) | -2.31 (-0.9) | -2.29 (-0.9) | 0.89 -> -1.55 |
| donchian_breakout|L | llm_in_sample | -2.22 (-1.8) | -2.14 (-1.8) | -2.14 (-1.9) | 0.59 -> -0.39 |
| donchian_breakout|L | clean_holdout | +0.19 (0.0) | +1.17 (0.3) | +2.20 (0.9) | -0.36 -> 0.36 |
| donchian_breakout|T | clean_holdout | +0.66 (0.3) | +0.71 (0.3) | +0.27 (0.1) | -0.21 -> 0.33 |
| ema_stack|L | llm_in_sample | -1.65 (-2.1) | -1.47 (-2.1) | -1.00 (-1.8) | 0.55 -> -0.50 |
| ema_stack|L | clean_holdout | +1.11 (0.5) | +1.13 (0.7) | +1.18 (1.1) | -1.34 -> -1.49 |
| ema_stack|T | clean_holdout | +2.32 (1.1) | +2.14 (1.2) | +1.55 (1.1) | -1.78 -> -1.65 |
| geo_flag_bo|L | llm_in_sample | -0.80 (-1.1) | -0.76 (-1.1) | -0.67 (-1.0) | 0.46 -> 0.21 |
| geo_flag_bo|L | clean_holdout | +2.39 (1.3) | +2.38 (1.3) | +2.16 (1.2) | -2.27 -> 0.63 |
| geo_flag_bo|T | clean_holdout | +2.74 (1.2) | +3.04 (1.2) | +3.08 (1.3) | -1.98 -> -0.80 |
| geo_pivot_bo|L | llm_in_sample | -3.74 (-2.6) | -3.71 (-2.6) | -3.15 (-2.2) | 1.12 -> 0.28 |
| geo_pivot_bo|L | clean_holdout | +2.66 (1.0) | +2.39 (0.9) | +2.43 (1.0) | -1.19 -> 0.30 |
| geo_pivot_bo|T | clean_holdout | +4.34 (1.5) | +4.50 (1.5) | +4.08 (1.4) | -2.25 -> 0.02 |
| geo_struct|L | llm_in_sample | -1.01 (-1.4) | -0.81 (-1.1) | -0.67 (-1.0) | 0.64 -> 0.94 |
| geo_struct|L | clean_holdout | +1.99 (1.1) | +1.98 (1.0) | +2.02 (1.1) | -2.12 -> -1.32 |
| geo_struct|T | clean_holdout | +3.07 (1.2) | +3.04 (1.2) | +2.78 (1.0) | -2.16 -> -1.30 |
| tsmom|L | llm_in_sample | -1.01 (-1.0) | -0.96 (-1.0) | -0.39 (-0.5) | -0.03 -> -0.90 |
| tsmom|L | clean_holdout | +1.59 (0.5) | +2.44 (1.1) | +2.47 (2.1) | -1.33 -> -0.28 |
| tsmom|T | clean_holdout | +3.55 (1.3) | +2.61 (1.1) | +3.32 (1.9) | -2.50 -> -3.67 |
| zoo_ensemble|L | llm_in_sample | -2.39 (-2.5) | -2.01 (-2.3) | -1.62 (-2.0) | 1.02 -> 0.40 |
| zoo_ensemble|L | clean_holdout | +1.22 (0.7) | +1.35 (0.8) | +1.27 (0.8) | -0.48 -> 1.12 |
| zoo_ensemble|T | clean_holdout | +2.31 (1.1) | +2.62 (1.3) | +2.44 (1.2) | -1.79 -> -0.08 |

## Caveats

- Close-based trail: no intrabar stop orders, so fills are at the bar close (1h or 4h) after the retracement; a resting stop would exit earlier and at a different price.
- The distance is one-bar sigma, not scaled with time in trade; on 1h bars a 1.5-3.5 sigma retracement is reached within hours, so the rule mostly truncates trades early.
- Re-entry needs a sign change of the raw signal; continuous signals (tsmom, ema_stack conviction) rarely pass through zero, so after a trailing exit those strategies can stay flat for a long time (lower exposure).
- MFE is a bar high/low upper bound (as in H-PD4); capture and leakage are diagnostics, the test is paired net P&L.
- The holdout is short (about 4.5 months) and T sleeves exist almost only in the holdout.

## Rule candidates

```yaml
- id: "R-EXIT-1"
  kind: "avoid"
  rule: "WHEN a trend-family trade is open THEN do not replace the strategy's own exit with a close-based trail of 1.5-3.5 one-bar sigma from the best close; over six years it lowered the trend book's net P&L at every k"
  when: "a trend-family position (tsmom, donchian_breakout, ema_stack, compression_breakout, geo_*, zoo_ensemble) is open on 1h or 4h bars"
  then: "keep the strategy's own exit; treat a k-sigma trail on the best close as a cost to the trend book, not an edge"
  scope:
    assets: "23 crypto perps (L) and 15 Bybit TradFi perps (T)"
    horizon: "1h-4h bars, trades of hours to weeks"
    sides: "both"
  effect: "paired book diff per day over k = 1.5/2.5/3.5: in-sample -0.0208 to -0.0167 (NW t -2.31 to -2.14; Sharpe 0.86 -> -1.13, -0.27, 0.27); holdout +0.0177 to +0.0187 (t +1.42 to +1.75, Holm p 0.12; Sharpe -2.21 -> -2.83, -1.31, -0.87); 2204 / 132 days"
  confidence: "B"
  evidence: ["H-PD6", "H-PD4", "research/reports/exit_policy_2026-11-12.md"]
  review: "re-run when the holdout passes 180 days or the trend book turns profitable in the holdout; a sqrt(time-in-trade) distance or a profit-only trail needs its own registration"
  status: "provisional"
- id: "R-EXIT-2"
  kind: "expect"
  rule: "WHEN a tight trailing exit is laid over the trend book THEN expect it to act mainly as an exposure cut with short convexity: small gains on most days, losses on the large trend days, so it helps only while the trend book itself is losing"
  when: "comparing a trailed trend book with the untrailed one, day by day"
  then: "read a trailing exit's holdout gain as a hedge of a losing trend stretch, not as better exits; do not switch it on from a few months of trend losses"
  scope:
    assets: "23 crypto perps (L) and 15 Bybit TradFi perps (T)"
    horizon: "1h-4h bars, trades of hours to weeks"
    sides: "both"
  effect: "exposure in-sample 0.86 -> 0.10/0.17/0.25; in-sample corr(diff, own-exit book) -1.00 to -0.96, skew of diff -2.31 to -2.00, diff > 0 on 0.51-0.54 of days yet negative on average; holdout corr -0.99 to -0.91 while the own-exit book had Sharpe -2.21"
  confidence: "C"
  evidence: ["H-PD6", "H-PD4", "research/reports/exit_policy_2026-11-12.md"]
  review: "re-run when the holdout passes 180 days or the trend book turns profitable in the holdout; a sqrt(time-in-trade) distance or a profit-only trail needs its own registration"
  status: "provisional"
- id: "R-EXIT-3"
  kind: "expect"
  rule: "WHEN trying to recover the H-PD4 exit leakage THEN expect a k-sigma trail on closes not to raise the share of favourable excursion kept in-sample; it shortens trades and shrinks the excursion pool instead"
  when: "exit designs evaluated with the H-PD4 MFE decomposition on trend-family trades"
  then: "rank exit rules on paired net P&L; read capture and leakage only as diagnostics, and look at time-in-trade or profit-only rules (each needs its own registration)"
  scope:
    assets: "23 crypto perps (L) and 15 Bybit TradFi perps (T)"
    horizon: "1h-4h bars, trades of hours to weeks"
    sides: "both"
  effect: "capture of MFE in-sample 0.316 -> 0.318/0.320/0.323, holdout 0.293 -> 0.354/0.352/0.350; (C+E)/MFE in-sample 0.722 -> 0.762/0.741/0.729; sum MFE in-sample 55021 -> 15669/23664/30834; mean hold 70 h -> 9/15/21 h; costs / kept gross 0.12 -> 0.25/0.19/0.16"
  confidence: "B"
  evidence: ["H-PD6", "H-PD4", "research/reports/exit_policy_2026-11-12.md"]
  review: "re-run when the holdout passes 180 days or the trend book turns profitable in the holdout; a sqrt(time-in-trade) distance or a profit-only trail needs its own registration"
  status: "provisional"
```
