# Daily-panel hypothesis mining with FDR control (H-MINE2)

_Program 2026-11-12 (full-speed round, agent H). Data as of 2026-11-12; knowledge cutoff 2026-07-01. Code: `src/ai_trader/research/daily_mining.py`; tests `tests/test_daily_mining.py`; artefacts `data/research/mining/daily_{family,discovery,discoveries_frozen,confirmation}.json`. Deterministic research only; the LLM decision layer is not tested here (principle 1)._

## Summary

- **Family:** 495 effects declared before any statistic (165 vol, 330 return; 495 testable). Targets: vr1 = log(RV_next / log-HAR), r1 = next-day vol-scaled return, r5 = next-5-day vol-scaled return.
- **Discovery (llm_in_sample, BH-FDR 10%, 10-day-block clustered t):** 90 discoveries (vr1 79, r1 5, r5 6); 74 of them restate existing rules (mostly the weekday and v0-trend vol effects).
- **Confirmation (clean_holdout, once, one-sided p < 0.05):** replicated 55/90 = 61% [51, 71]; vol 54/79 = 68% [57, 78], return 1/11 = 9% [2, 38]. Sign agreement 86/90 = 96% [89, 98]. Untestable on the holdout (too few rows / clusters): 0.
- **H-MINE2 verdict (pooled): supports.** Registry criterion: rate >= 50% and vol rate > return rate.
- **Novel replicated effects:** 2; replicated restatements of known rules: 53.
- **Read with care:** the 'supports' rests on restated vol rules (53/74 replicate); discoveries that are not restatements replicate 2/16. Return targets: 1/11 replicated against 0.55 expected by chance at one-sided 5%, so there is no evidence of a replicating return effect. Both novel replications lose significance with date-clustered SEs (secondary), and one rests on 9 holdout dates.

## Protocol

- Panel: universe-L complete daily rows (H-NN1 sample, 52 point-in-time features), decision at 00:00 UTC of D+1. vr1 uses H-NN1's walk-forward per-asset log-HAR forecast (monthly purged refits, from 2022-04). r1 / r5 are winsorised at +-5 sigma; r5 needs 5 consecutive days and its own segment rule (in-sample iff D + 6 <= cutoff).
- States (edges fitted on llm_in_sample rows only, frozen and re-used on the holdout): F1 terciles of the top-10 H-NN3 TreeSHAP features + top-bottom contrast; F2 quintiles of log_sigma, lrv_w, lrv_q + Q5-Q1; F3 the 10 tercile pairs of the top-5; F4 weekday; F5 v0 cell; F6 funding-z tercile + contrast; F7 deep-drawdown flag (dd_90d_z bottom fifth) + contrast; F8 BTC v0 cell (alts).
- Effect = cell mean minus the state's population mean (or top minus bottom). SE from the influence function, clustered by 10-day blocks, t with df = blocks touched - 1; date-clustered SE shown as a secondary check. BH at q = 0.10 over the whole family; untestable effects (< 100 rows or < 30 dates) keep p = 1.
- The discovery list and edges were frozen (diary 'H-MINE2 discovery list frozen', sha256 `d7e0e93b34af5602...`) before any holdout statistic; `target_rows(clean_holdout)` raises without the frozen file.

### Family by state family and target

| family | effects | | target | effects |
|---|---|---|---|---|
| F1 | 117 | | r1 | 165 |
| F2 | 54 | | r5 | 165 |
| F3 | 228 | | vr1 | 165 |
| F4 | 21 | |  |  |
| F5 | 27 | |  |  |
| F6 | 12 | |  |  |
| F7 | 9 | |  |  |
| F8 | 27 | |  |  |

## Replication

| split | discoveries | replicated (Wilson 95%) |
|---|---|---|
| overall | 90 | 55/90 = 61% [51, 71] |
| kind vol | 79 | 54/79 = 68% [57, 78] |
| kind return | 11 | 1/11 = 9% [2, 38] |
| target vr1 | 79 | 54/79 = 68% [57, 78] |
| target r1 | 5 | 1/5 = 20% [4, 62] |
| target r5 | 6 | 0/6 = 0% [0, 39] |
| family F1 | 17 | 14/17 = 82% [59, 94] |
| family F2 | 3 | 0/3 = 0% [0, 56] |
| family F3 | 55 | 34/55 = 62% [49, 73] |
| family F4 | 5 | 2/5 = 40% [12, 77] |
| family F5 | 5 | 2/5 = 40% [12, 77] |
| family F6 | 2 | 2/2 = 100% [34, 100] |
| family F8 | 3 | 1/3 = 33% [6, 79] |
| tagged known | 74 | 53/74 = 72% [60, 81] |
| tagged contradicts | 12 | 2/12 = 17% [5, 45] |
| tagged novel | 4 | 0/4 = 0% [0, 49] |
| overall, date-clustered SE (secondary) | 90 | 58/90 = 64% [54, 74] |

## Discoveries and their holdout test

Effect: vr1 as a ratio to the average log-HAR residual (exp(d) - 1); r1 / r5 in sigma units. t = 10-day-block clustered. Holdout p is one-sided in the discovered sign.

| id | state | effect in | t in | p_bh | rows in | effect ho | t ho | p ho | rows ho | replicated | tag |
|---|---|---|---|---|---|---|---|---|---|---|---|
| r1|Q5:lrv_q|c0 | 90-day mean log RV in the bottom fifth (<= -7.3) | +0.067 sigma | 2.9 | 0.027 | 7044 | +0.037 sigma | 1.47 | 0.0823 | 1049 | no | contradicts |
| r1|T3xT3:lrv_qxdow_sin|c0 | 90-day mean log RV in the bottom third (<= -6.99), forecast day Fri-Sun | +0.100 sigma | 2.9 | 0.03 | 5010 | +0.157 sigma | 1.91 | 0.0403 | 734 | **yes** | contradicts |
| r1|T3xT3:lrv_wxlrv_q|c6 | 7-day mean log RV in the top third (> -6.4), 90-day mean log RV in the bottom third (<= -6.99) | +0.104 sigma | 2.7 | 0.056 | 1009 | -0.126 sigma | -1.04 | 0.831 | 137 | no | contradicts |
| r1|btc_v0cell|c2 | BTC v0 cell R3 momentum burst [alts only] | +0.138 sigma | 2.6 | 0.068 | 2748 | +0.064 sigma | 0.30 | 0.389 | 286 | no | contradicts |
| r1|T3xT3:log_sigmaxlrv_q|c6 | robust 1h sigma (log) in the top third (> -3.11), 90-day mean log RV in the bottom third (<= -6.99) | +0.095 sigma | 2.5 | 0.085 | 1552 | +0.125 sigma | 1.37 | 0.102 | 163 | no | contradicts |
| r5|T3xT3:lrv_wxlrv_q|c6 | 7-day mean log RV in the top third (> -6.4), 90-day mean log RV in the bottom third (<= -6.99) | +0.275 sigma | 3.9 | 0.002 | 1005 | -0.225 sigma | -1.03 | 0.828 | 137 | no | contradicts |
| r5|T3xT3:log_sigmaxlrv_q|c6 | robust 1h sigma (log) in the top third (> -3.11), 90-day mean log RV in the bottom third (<= -6.99) | +0.204 sigma | 3.5 | 0.0047 | 1544 | -0.051 sigma | -0.29 | 0.612 | 163 | no | contradicts |
| r5|Q5:lrv_q|c0 | 90-day mean log RV in the bottom fifth (<= -7.3) | +0.160 sigma | 3.1 | 0.017 | 7016 | +0.058 sigma | 1.03 | 0.16 | 1029 | no | contradicts |
| r5|T3xT3:log_sigmaxlrv_q|c3 | robust 1h sigma (log) in the middle third (-3.52 to -3.11), 90-day mean log RV in the bottom third (<= -6.99) | +0.124 sigma | 2.5 | 0.072 | 2939 | +0.106 sigma | 0.68 | 0.255 | 424 | no | contradicts |
| r5|btc_v0cell|c2 | BTC v0 cell R3 momentum burst [alts only] | +0.221 sigma | 2.5 | 0.088 | 2748 | +0.051 sigma | 0.30 | 0.389 | 286 | no | contradicts |
| r5|T3xT3:log_sigmaxlrv_q|c5 | robust 1h sigma (log) in the middle third (-3.52 to -3.11), 90-day mean log RV in the top third (> -6.45) | -0.136 sigma | -2.4 | 0.098 | 4015 | -0.253 sigma | -1.75 | 0.0519 | 111 | no | contradicts |
| vr1|T3:next_is_weekend|hi-lo | forecast day Sat/Sun minus forecast day Mon-Fri | -38.9% | -16.8 | 6.1e-34 | 9284 | -38.2% | -5.06 | 0.00014 | 874 | **yes** | known |
| vr1|T3:next_is_weekend|c2 | forecast day Sat/Sun | -29.7% | -16.5 | 7.5e-34 | 9284 | -29.1% | -4.57 | 0.00032 | 874 | **yes** | known |
| vr1|T3:next_wk_hours|c2 | forecast day Sat/Sun (TradFi weekend) | -29.7% | -16.5 | 7.5e-34 | 9284 | -29.1% | -4.57 | 0.00032 | 874 | **yes** | known |
| vr1|T3xT3:next_is_weekendxdow_sin|c6 | forecast day Sat/Sun, forecast day Fri-Sun | -29.7% | -16.5 | 7.5e-34 | 9284 | -29.1% | -4.57 | 0.00032 | 874 | **yes** | known |
| vr1|T3:next_wk_hours|hi-lo | forecast day Sat/Sun (TradFi weekend) minus forecast day Mon-Thu (no TradFi-weekend hours) | -39.3% | -16.3 | 2.4e-33 | 9284 | -39.4% | -4.65 | 0.000281 | 874 | **yes** | known |
| vr1|weekday|c5 | forecast day Saturday | -44.7% | -16.1 | 6.2e-33 | 4640 | -46.1% | -5.76 | 4.49e-05 | 437 | **yes** | known |
| vr1|T3:next_is_weekend|c0 | forecast day Mon-Fri | +15.1% | 15.7 | 5e-32 | 23184 | +14.7% | 6.09 | 1.41e-05 | 2185 | **yes** | known |
| vr1|T3xT3:lrv_wxnext_is_weekend|c2 | 7-day mean log RV in the bottom third (<= -7.13), forecast day Sat/Sun | -36.7% | -13.6 | 2.6e-26 | 3262 | -33.5% | -6.90 | 8.28e-06 | 439 | **yes** | known |
| vr1|T3xT3:lrv_qxnext_is_weekend|c5 | 90-day mean log RV in the middle third (-6.99 to -6.45), forecast day Sat/Sun | -28.7% | -13.2 | 2e-25 | 3152 | -24.8% | -4.02 | 0.000855 | 274 | **yes** | known |
| vr1|T3xT3:log_sigmaxnext_is_weekend|c2 | robust 1h sigma (log) in the bottom third (<= -3.52), forecast day Sat/Sun | -33.1% | -13.0 | 7.2e-25 | 3668 | -29.4% | -4.96 | 0.000166 | 452 | **yes** | known |
| vr1|T3xT3:lrv_qxnext_is_weekend|c2 | 90-day mean log RV in the bottom third (<= -6.99), forecast day Sat/Sun | -34.3% | -12.7 | 6.7e-24 | 3266 | -32.5% | -4.98 | 0.000161 | 485 | **yes** | known |
| vr1|T3:dow_sin|c0 | forecast day Fri-Sun | -17.9% | -12.1 | 1.5e-22 | 13924 | -18.9% | -3.05 | 0.00501 | 1311 | **yes** | known |
| vr1|T3xT3:log_sigmaxnext_is_weekend|c5 | robust 1h sigma (log) in the middle third (-3.52 to -3.11), forecast day Sat/Sun | -31.6% | -12.1 | 1.6e-22 | 2893 | -39.7% | -4.01 | 0.000863 | 248 | **yes** | known |
| vr1|T3xT3:lrv_wxnext_is_weekend|c5 | 7-day mean log RV in the middle third (-7.13 to -6.4), forecast day Sat/Sun | -28.7% | -12.1 | 1.6e-22 | 3108 | -23.4% | -1.83 | 0.0461 | 258 | **yes** | known |
| vr1|T3:next_wk_hours|c0 | forecast day Mon-Thu (no TradFi-weekend hours) | +15.9% | 11.2 | 2.8e-20 | 18544 | +17.0% | 3.59 | 0.00147 | 1748 | **yes** | known |
| vr1|weekday|c0 | forecast day Monday | +44.8% | 11.0 | 1.5e-19 | 4646 | +24.6% | 2.19 | 0.0238 | 437 | **yes** | known |
| vr1|T3xT3:lrv_qxnext_is_weekend|c3 | 90-day mean log RV in the middle third (-6.99 to -6.45), forecast day Mon-Fri | +15.3% | 10.6 | 1.1e-18 | 7844 | +16.1% | 4.46 | 0.000272 | 652 | **yes** | known |
| vr1|T3xT3:lrv_wxnext_is_weekend|c3 | 7-day mean log RV in the middle third (-7.13 to -6.4), forecast day Mon-Fri | +14.1% | 10.3 | 8.4e-18 | 7685 | +14.3% | 4.49 | 0.000252 | 643 | **yes** | known |
| vr1|T3xT3:lrv_qxdow_sin|c0 | 90-day mean log RV in the bottom third (<= -6.99), forecast day Fri-Sun | -22.7% | -10.3 | 8.8e-18 | 4928 | -21.9% | -3.15 | 0.00418 | 734 | **yes** | known |
| vr1|T3xT3:log_sigmaxnext_is_weekend|c0 | robust 1h sigma (log) in the bottom third (<= -3.52), forecast day Mon-Fri | +20.8% | 10.2 | 1.8e-17 | 7703 | +18.0% | 4.23 | 0.000416 | 1041 | **yes** | known |
| vr1|T3xT3:lrv_wxdow_sin|c0 | 7-day mean log RV in the bottom third (<= -7.13), forecast day Fri-Sun | -23.7% | -9.6 | 4.6e-16 | 4907 | -24.2% | -3.36 | 0.00282 | 657 | **yes** | known |
| vr1|T3xT3:lrv_qxnext_is_weekend|c0 | 90-day mean log RV in the bottom third (<= -6.99), forecast day Mon-Fri | +14.5% | 9.3 | 3.7e-15 | 8275 | +12.5% | 5.76 | 2.48e-05 | 1245 | **yes** | known |
| vr1|T3xT3:lrv_qxnext_is_weekend|c6 | 90-day mean log RV in the top third (> -6.45), forecast day Mon-Fri | +15.6% | 9.3 | 5.6e-15 | 7065 | +21.8% | 4.62 | 0.000198 | 288 | **yes** | known |
| vr1|T3xT3:lrv_qxnext_is_weekend|c8 | 90-day mean log RV in the top third (> -6.45), forecast day Sat/Sun | -25.0% | -9.3 | 5.7e-15 | 2866 | -23.9% | -2.65 | 0.0106 | 115 | **yes** | known |
| vr1|T3:dow_sin|c1 | forecast day Mon or Thu | +23.4% | 9.1 | 1e-14 | 9272 | +24.3% | 3.08 | 0.00436 | 874 | **yes** | known |
| vr1|T3xT3:next_is_weekendxdow_sin|c1 | forecast day Mon-Fri, forecast day Mon or Thu | +23.4% | 9.1 | 1e-14 | 9272 | +24.3% | 3.08 | 0.00436 | 874 | **yes** | known |
| vr1|T3xT3:log_sigmaxdow_sin|c0 | robust 1h sigma (log) in the bottom third (<= -3.52), forecast day Fri-Sun | -23.3% | -9.0 | 1.2e-14 | 5045 | -22.3% | -3.57 | 0.00194 | 639 | **yes** | known |
| vr1|T3xT3:log_sigmaxdow_sin|c1 | robust 1h sigma (log) in the bottom third (<= -3.52), forecast day Mon or Thu | +30.8% | 9.0 | 1.9e-14 | 3475 | +32.1% | 3.48 | 0.00205 | 428 | **yes** | known |
| vr1|T3xT3:lrv_wxnext_is_weekend|c0 | 7-day mean log RV in the bottom third (<= -7.13), forecast day Mon-Fri | +16.8% | 8.7 | 6.9e-14 | 8197 | +16.7% | 4.42 | 0.000289 | 1083 | **yes** | known |
| vr1|T3:rq_ratio_d|hi-lo | day-D quarticity ratio (one-big-hour share) in the top third (> 0.167) minus day-D quarticity ratio (one-big-hour share) in the bottom third (<= -0.0103) | -20.8% | -8.7 | 7.7e-14 | 10818 | -17.7% | -3.97 | 0.000703 | 957 | **yes** | known |
| vr1|T3xT3:log_sigmaxdow_sin|c3 | robust 1h sigma (log) in the middle third (-3.52 to -3.11), forecast day Fri-Sun | -19.3% | -8.6 | 1.1e-13 | 4461 | -23.1% | -3.11 | 0.00449 | 373 | **yes** | known |
| vr1|T3:dow_sin|hi-lo | forecast day Tue or Wed minus forecast day Fri-Sun | +32.5% | 8.5 | 2.2e-13 | 9272 | +35.6% | 2.62 | 0.0112 | 874 | **yes** | known |
| vr1|T3:rq_ratio_d|c2 | day-D quarticity ratio (one-big-hour share) in the top third (> 0.167) | -12.0% | -8.4 | 4.8e-13 | 10818 | -8.5% | -4.03 | 0.000624 | 1131 | **yes** | known |
| vr1|T3xT3:lrv_qxdow_sin|c3 | 90-day mean log RV in the middle third (-6.99 to -6.45), forecast day Fri-Sun | -16.6% | -8.0 | 5.3e-12 | 4722 | -15.8% | -3.09 | 0.00468 | 406 | **yes** | known |
| vr1|T3xT3:lrv_wxnext_is_weekend|c8 | 7-day mean log RV in the top third (> -6.4), forecast day Sat/Sun | -22.0% | -7.9 | 9.7e-12 | 2914 | -25.6% | -2.28 | 0.0228 | 177 | **yes** | known |
| vr1|T3xT3:log_sigmaxnext_is_weekend|c3 | robust 1h sigma (log) in the middle third (-3.52 to -3.11), forecast day Mon-Fri | +11.3% | 7.8 | 1.1e-11 | 7866 | +11.6% | 3.20 | 0.00318 | 669 | **yes** | known |
| vr1|T3:rq_ratio_d|c0 | day-D quarticity ratio (one-big-hour share) in the bottom third (<= -0.0103) | +11.1% | 7.8 | 1.4e-11 | 10896 | +11.2% | 3.37 | 0.00227 | 957 | **yes** | known |
| vr1|T3xT3:lrv_wxdow_sin|c3 | 7-day mean log RV in the middle third (-7.13 to -6.4), forecast day Fri-Sun | -16.8% | -7.6 | 3.8e-11 | 4647 | -11.2% | -1.19 | 0.129 | 383 | no | known |
| vr1|T3xT3:lrv_qxdow_sin|c4 | 90-day mean log RV in the middle third (-6.99 to -6.45), forecast day Mon or Thu | +22.2% | 7.5 | 5.5e-11 | 3133 | +27.6% | 3.55 | 0.00179 | 257 | **yes** | known |
| vr1|T3xT3:lrv_wxdow_sin|c1 | 7-day mean log RV in the bottom third (<= -7.13), forecast day Mon or Thu | +26.3% | 7.0 | 8.4e-10 | 3279 | +31.6% | 3.48 | 0.00204 | 433 | **yes** | known |
| vr1|T3xT3:lrv_wxnext_is_weekend|c6 | 7-day mean log RV in the top third (> -6.4), forecast day Mon-Fri | +14.3% | 7.0 | 1.1e-09 | 7302 | +10.8% | 1.62 | 0.0652 | 459 | no | known |
| vr1|T3xT3:lrv_wxdow_sin|c4 | 7-day mean log RV in the middle third (-7.13 to -6.4), forecast day Mon or Thu | +21.4% | 6.9 | 1.4e-09 | 3078 | +17.9% | 2.01 | 0.0329 | 263 | **yes** | known |
| vr1|T3:v0_dir_score|c1 | v0 trend score in the middle third (-0.368 to 0.275) | -9.3% | -6.8 | 2.6e-09 | 10852 | -11.2% | -2.45 | 0.014 | 1063 | **yes** | known |
| vr1|T3xT3:lrv_qxdow_sin|c1 | 90-day mean log RV in the bottom third (<= -6.99), forecast day Mon or Thu | +25.1% | 6.7 | 5.5e-09 | 3309 | +22.0% | 2.52 | 0.0128 | 501 | **yes** | known |
| vr1|T3xT3:lrv_qxdow_sin|c7 | 90-day mean log RV in the top third (> -6.45), forecast day Mon or Thu | +22.9% | 6.5 | 1.4e-08 | 2830 | +27.7% | 3.12 | 0.00406 | 116 | **yes** | known |
| vr1|T3xT3:log_sigmaxnext_is_weekend|c6 | robust 1h sigma (log) in the top third (> -3.11), forecast day Mon-Fri | +13.6% | 6.2 | 4.4e-08 | 7615 | +12.1% | 2.54 | 0.0118 | 475 | **yes** | known |
| vr1|T3xT3:lrv_wxdow_sin|c7 | 7-day mean log RV in the top third (> -6.4), forecast day Mon or Thu | +22.4% | 6.1 | 1.1e-07 | 2915 | +17.3% | 2.04 | 0.0333 | 178 | **yes** | known |
| vr1|T3xT3:lrv_qxdow_sin|c6 | 90-day mean log RV in the top third (> -6.45), forecast day Fri-Sun | -13.3% | -6.1 | 1.3e-07 | 4274 | -12.6% | -1.67 | 0.0604 | 171 | no | known |
| vr1|T3xT3:log_sigmaxnext_is_weekend|c8 | robust 1h sigma (log) in the top third (> -3.11), forecast day Sat/Sun | -22.5% | -6.0 | 1.6e-07 | 2723 | -9.6% | -0.38 | 0.358 | 174 | no | known |
| vr1|T3xT3:log_sigmaxdow_sin|c4 | robust 1h sigma (log) in the middle third (-3.52 to -3.11), forecast day Mon or Thu | +17.7% | 5.8 | 3.4e-07 | 3029 | +14.6% | 1.20 | 0.125 | 269 | no | known |
| vr1|T3xT3:lrv_wxdow_sin|c6 | 7-day mean log RV in the top third (> -6.4), forecast day Fri-Sun | -11.9% | -4.8 | 3.1e-05 | 4370 | -15.8% | -2.63 | 0.0116 | 271 | **yes** | known |
| vr1|v0cell|c2 | v0 cell R3 momentum burst | +15.2% | 4.5 | 0.00012 | 2598 | +28.0% | 3.57 | 0.00194 | 402 | **yes** | known |
| vr1|T3xT3:log_sigmaxdow_sin|c7 | robust 1h sigma (log) in the top third (> -3.11), forecast day Mon or Thu | +21.0% | 4.5 | 0.00015 | 2768 | +21.5% | 2.40 | 0.0167 | 177 | **yes** | known |
| vr1|T3xT3:log_sigmaxdow_sin|c2 | robust 1h sigma (log) in the bottom third (<= -3.52), forecast day Tue or Wed | +14.3% | 4.4 | 0.00021 | 2851 | +14.4% | 1.46 | 0.0827 | 426 | no | known |
| vr1|v0cell|c4 | v0 cell R5 balanced range | -6.6% | -4.1 | 0.0005 | 5115 | -3.0% | -0.55 | 0.295 | 437 | no | known |
| vr1|btc_v0cell|c5 | BTC v0 cell R6 whipsaw [alts only] | -15.8% | -4.2 | 0.00084 | 4271 | -19.9% | -2.20 | 0.0396 | 198 | **yes** | contradicts |
| vr1|T3:v0_dir_score|c2 | v0 trend score in the top third (> 0.275) | +7.0% | 4.0 | 0.00099 | 10799 | +19.3% | 3.72 | 0.00114 | 1040 | **yes** | known |
| vr1|T3xT3:lrv_wxdow_sin|c2 | 7-day mean log RV in the bottom third (<= -7.13), forecast day Tue or Wed | +11.1% | 3.9 | 0.0011 | 3273 | +12.6% | 1.19 | 0.126 | 432 | no | known |
| vr1|T3:dow_sin|c2 | forecast day Tue or Wed | +8.8% | 3.9 | 0.0013 | 9272 | +10.1% | 1.48 | 0.0806 | 874 | no | known |
| vr1|T3xT3:next_is_weekendxdow_sin|c2 | forecast day Mon-Fri, forecast day Tue or Wed | +8.8% | 3.9 | 0.0013 | 9272 | +10.1% | 1.48 | 0.0806 | 874 | no | known |
| vr1|v0cell|c5 | v0 cell R6 whipsaw | -9.0% | -3.8 | 0.0016 | 4588 | -13.3% | -2.28 | 0.0202 | 542 | **yes** | known |
| vr1|T3xT3:log_sigmaxdow_sin|c6 | robust 1h sigma (log) in the top third (> -3.11), forecast day Fri-Sun | -9.7% | -3.6 | 0.0039 | 4418 | -4.7% | -0.26 | 0.401 | 299 | no | known |
| vr1|weekday|c6 | forecast day Sunday | -10.5% | -3.5 | 0.0048 | 4644 | -6.7% | -0.77 | 0.228 | 437 | no | known |
| vr1|T3:funding_z_90d|c2 | funding z (90d) in the top third (> 0.419) | +5.5% | 3.5 | 0.0048 | 11161 | +9.5% | 2.94 | 0.00533 | 1323 | **yes** | known |
| vr1|T3xT3:lrv_qxdow_sin|c5 | 90-day mean log RV in the middle third (-6.99 to -6.45), forecast day Tue or Wed | +9.4% | 3.3 | 0.0083 | 3141 | +10.5% | 1.38 | 0.0943 | 263 | no | known |
| vr1|T3xT3:lrv_qxdow_sin|c2 | 90-day mean log RV in the bottom third (<= -6.99), forecast day Tue or Wed | +8.7% | 3.3 | 0.0087 | 3304 | +7.8% | 1.08 | 0.148 | 495 | no | known |
| vr1|weekday|c2 | forecast day Wednesday | +13.1% | 3.2 | 0.013 | 4626 | +23.2% | 1.76 | 0.0511 | 437 | no | known |
| vr1|v0cell|c1 | v0 cell R2 trend up | +8.8% | 3.1 | 0.016 | 2287 | +18.7% | 1.57 | 0.0709 | 170 | no | known |
| vr1|T3:funding_z_90d|hi-lo | funding z (90d) in the top third (> 0.419) minus funding z (90d) in the bottom third (<= -0.375) | +9.1% | 3.1 | 0.016 | 10318 | +25.7% | 3.29 | 0.00268 | 871 | **yes** | known |
| vr1|T3:next_wk_hours|c1 | forecast day Fri (TradFi weekend starts) | +12.0% | 3.0 | 0.024 | 4640 | +6.1% | 0.45 | 0.33 | 437 | no | known |
| vr1|T3xT3:next_is_weekendxdow_sin|c0 | forecast day Mon-Fri, forecast day Fri-Sun | +12.0% | 3.0 | 0.024 | 4640 | +6.1% | 0.45 | 0.33 | 437 | no | known |
| vr1|weekday|c4 | forecast day Friday | +12.0% | 3.0 | 0.024 | 4640 | +6.1% | 0.45 | 0.33 | 437 | no | known |
| vr1|T3xT3:lrv_wxdow_sin|c5 | 7-day mean log RV in the middle third (-7.13 to -6.4), forecast day Tue or Wed | +7.6% | 2.8 | 0.038 | 3068 | +7.8% | 1.26 | 0.114 | 255 | no | known |
| vr1|T3xT3:log_sigmaxlrv_q|c4 | robust 1h sigma (log) in the middle third (-3.52 to -3.11), 90-day mean log RV in the middle third (-6.99 to -6.45) | -4.2% | -2.7 | 0.052 | 4424 | +0.5% | 0.09 | 0.533 | 364 | no | novel |
| vr1|Q5:log_sigma|c4 | robust 1h sigma (log) in the top fifth (> -2.91) | +6.5% | 2.6 | 0.056 | 6167 | +14.1% | 1.25 | 0.116 | 346 | no | novel |
| vr1|T3xT3:lrv_qxdow_sin|c8 | 90-day mean log RV in the top third (> -6.45), forecast day Tue or Wed | +8.4% | 2.6 | 0.062 | 2827 | +18.8% | 2.93 | 0.00552 | 116 | **yes** | known |
| vr1|v0cell|c0 | v0 cell R1 quiet trend up | +8.5% | 2.6 | 0.063 | 2549 | +10.5% | 0.73 | 0.241 | 182 | no | known |
| vr1|T3xT3:log_sigmaxlrv_q|c0 | robust 1h sigma (log) in the bottom third (<= -3.52), 90-day mean log RV in the bottom third (<= -6.99) | -4.1% | -2.5 | 0.073 | 7130 | -1.1% | -0.34 | 0.368 | 1136 | no | novel |
| vr1|T3:log_sigma|c1 | robust 1h sigma (log) in the middle third (-3.52 to -3.11) | -2.4% | -2.5 | 0.085 | 10759 | -5.5% | -1.59 | 0.0666 | 917 | no | novel |

## Replicated effects in plain words

### Novel (or contradicting an existing rule)

- When BTC v0 cell R6 whipsaw [alts only]: next-day RV relative to the log-HAR forecast about -16% in-sample / -20% holdout, relative to the average row. holdout 198 rows on 9 dates (6 ten-day blocks), t -2.20, one-sided p 0.0396; date-clustered p 0.0718; tag contradicts (R-VOL-7, R-VOL-25 (no BTC vol spillover beyond own HAR)).
- When 90-day mean log RV in the bottom third (<= -6.99), forecast day Fri-Sun: next-day vol-scaled return +0.100 / +0.157 sigma (in-sample / holdout), relative to the average row. holdout 734 rows on 57 dates (13 ten-day blocks), t 1.91, one-sided p 0.0403; date-clustered p 0.0586; tag contradicts (R-DS-6, R-BUDGET-1, R-ML-5, R-PX-6 (no direction from states / weekday / regime), R-PX-9 (no weekend direction)).

### Already known (restate an existing rule), grouped by rule

- **R-VOL-1, R-VOL-27** (44 replicated): When 7-day mean log RV in the bottom third (<= -7.13), forecast day Sat/Sun: next-day RV relative to the log-HAR forecast about -37% in-sample / -33% holdout, relative to the average row; When forecast day Mon-Fri: next-day RV relative to the log-HAR forecast about +15% in-sample / +15% holdout, relative to the average row; When 90-day mean log RV in the bottom third (<= -6.99), forecast day Mon-Fri: next-day RV relative to the log-HAR forecast about +15% in-sample / +12% holdout, relative to the average row.
- **R-DS-4, R-VOL-8** (4 replicated): When v0 trend score in the top third (> 0.275): next-day RV relative to the log-HAR forecast about +7% in-sample / +19% holdout, relative to the average row; When v0 cell R3 momentum burst: next-day RV relative to the log-HAR forecast about +15% in-sample / +28% holdout, relative to the average row; When v0 trend score in the middle third (-0.368 to 0.275): next-day RV relative to the log-HAR forecast about -9% in-sample / -11% holdout, relative to the average row.
- **R-VOL-23 (HAR-Q: a day dominated by one big hour gets less weight), R-VOL-3 / BELIEFS V4 (spike excess variance decays within hours)** (3 replicated): When day-D quarticity ratio (one-big-hour share) in the top third (> 0.167): next-day RV relative to the log-HAR forecast about -12% in-sample / -8% holdout, relative to the average row; day-D quarticity ratio (one-big-hour share) in the top third (> 0.167) minus day-D quarticity ratio (one-big-hour share) in the bottom third (<= -0.0103): next-day RV relative to the log-HAR forecast about -21% in-sample / -18% holdout (contrast); When day-D quarticity ratio (one-big-hour share) in the bottom third (<= -0.0103): next-day RV relative to the log-HAR forecast about +11% in-sample / +11% holdout, relative to the average row.
- **R-LEV-5, BELIEFS H1 (funding crowding tilts next-day vol up)** (2 replicated): funding z (90d) in the top third (> 0.419) minus funding z (90d) in the bottom third (<= -0.375): next-day RV relative to the log-HAR forecast about +9% in-sample / +26% holdout (contrast); When funding z (90d) in the top third (> 0.419): next-day RV relative to the log-HAR forecast about +5% in-sample / +9% holdout, relative to the average row.

Known-rule discoveries that did **not** replicate (21): forecast day Wednesday (+13.1% in, +23.2% ho, p 0.05); 90-day mean log RV in the top third (> -6.45), forecast day Fri-Sun (-13.3% in, -12.6% ho, p 0.06); 7-day mean log RV in the top third (> -6.4), forecast day Mon-Fri (+14.3% in, +10.8% ho, p 0.07); v0 cell R2 trend up (+8.8% in, +18.7% ho, p 0.07); forecast day Tue or Wed (+8.8% in, +10.1% ho, p 0.08); forecast day Mon-Fri, forecast day Tue or Wed (+8.8% in, +10.1% ho, p 0.08); robust 1h sigma (log) in the bottom third (<= -3.52), forecast day Tue or Wed (+14.3% in, +14.4% ho, p 0.08); 90-day mean log RV in the middle third (-6.99 to -6.45), forecast day Tue or Wed (+9.4% in, +10.5% ho, p 0.09); 7-day mean log RV in the middle third (-7.13 to -6.4), forecast day Tue or Wed (+7.6% in, +7.8% ho, p 0.11); robust 1h sigma (log) in the middle third (-3.52 to -3.11), forecast day Mon or Thu (+17.7% in, +14.6% ho, p 0.13); 7-day mean log RV in the bottom third (<= -7.13), forecast day Tue or Wed (+11.1% in, +12.6% ho, p 0.13); 7-day mean log RV in the middle third (-7.13 to -6.4), forecast day Fri-Sun (-16.8% in, -11.2% ho, p 0.13); 90-day mean log RV in the bottom third (<= -6.99), forecast day Tue or Wed (+8.7% in, +7.8% ho, p 0.15); forecast day Sunday (-10.5% in, -6.7% ho, p 0.23); v0 cell R1 quiet trend up (+8.5% in, +10.5% ho, p 0.24); v0 cell R5 balanced range (-6.6% in, -3.0% ho, p 0.30); forecast day Fri (TradFi weekend starts) (+12.0% in, +6.1% ho, p 0.33); forecast day Mon-Fri, forecast day Fri-Sun (+12.0% in, +6.1% ho, p 0.33); forecast day Friday (+12.0% in, +6.1% ho, p 0.33); robust 1h sigma (log) in the top third (> -3.11), forecast day Sat/Sun (-22.5% in, -9.6% ho, p 0.36); robust 1h sigma (log) in the top third (> -3.11), forecast day Fri-Sun (-9.7% in, -4.7% ho, p 0.40).

### Overlap with agent C's cells (discrete_states_2026-11-12)

74 of 90 discoveries use a state agent C's LAT1 lattice also uses (weekday / calendar, v0 trend, own vol level); 49 of them replicated. Discoveries outside that lattice: 16, replicated 6.

## Caveats

- The holdout is about 133 dates (2026-07-01 to 2026-11-10), i.e. 13-14 ten-day blocks; one-sided tests with df <= 13 have little power for small effects, so a non-replication is weak evidence of absence.
- Effects within a state are strongly dependent (cells of one partition, the same feature in F1 / F2 / F3, three vol-level copies), so the effective number of independent discoveries is much smaller than the count. BH is valid under positive dependence; the replication rate counts dependent effects separately.
- The top-10 features come from H-NN3's in-sample TreeSHAP ranking, computed on in-sample walk-forward rows; the choice of states is therefore informed by in-sample data (mild look-ahead within discovery, none in the holdout).
- Effects are relative to each segment's own population mean, so a level shift between segments (e.g. a lower average log-HAR residual) does not count; the vr1 effect is geometric (mean log ratio).
- Known-rule tags are a mechanical mapping (KNOWN / known_tag in the code), fixed before discovery; they can over- or under-state novelty. One correction was made after freezing and before any holdout statistic (diary 'H-MINE2 known-rule tags reviewed'): the three day-D quarticity-ratio vol effects are the HAR-Q correction (R-VOL-23), so they are tagged known; the mechanical tag is kept as known_mechanical in the JSON.
- vr1 uses H-NN1's walk-forward log-HAR forecast, so the HAR-Q and weekday effects are relative to plain log-HAR; against log-HAR + weekday dummies (H-NN1 exploratory) the calendar cells would mostly vanish by construction.
- The BTC-whipsaw cell (R-MINE-20) is a date-level state: its holdout evidence is 9 dates in 6 blocks, and alts' vol then is one correlated bet (R-VOL-25).
- Rule candidates below are mined, replicated once on a short holdout, and graded C; they must be registered as forward hypotheses before promotion.

## Rule candidates

Novel replicated effects only (known-rule restatements are excluded).

```yaml
- id: "R-MINE-20"
  kind: "expect"
  rule: "WHEN the perp is an alt (not BTC) and BTC's v0 cell is R6 whipsaw THEN expect next-day RV relative to the log-HAR forecast -15.8% vs the average (in-sample), -19.9% holdout"
  when: "daily (1d) decision at 00:00 UTC for a universe-L crypto perp; H-MINE2 state btc_v0cell (edges fitted on llm_in_sample)"
  then: "central next-day variance expectation = log-HAR x 0.84 relative to the average residual; stops and risk keep the unshrunk risk sigma (R-VOL-2)"
  scope:
    assets: "crypto perps (universe L)"
    horizon: "1d"
    sides: "both"
  effect: "vr1 effect -0.1714 (t -4.16, 4271 rows, 211 dates) in-sample, -0.2213 (t -2.20, one-sided p 0.0396, 198 rows, 9 dates) holdout; date-clustered holdout p 0.0718; BH-FDR 10% discovery among 495 effects"
  confidence: "C"
  evidence: ["H-MINE2"]
  review: "register as a forward hypothesis; demote if the effect on dates after 2026-11-12 changes sign or falls below half; it contradicts R-VOL-7, R-VOL-25 (no BTC vol spillover beyond own HAR), which stay in force until a forward test confirms this"
  status: "provisional"
- id: "R-MINE-21"
  kind: "expect"
  rule: "WHEN 90-day mean log RV in the bottom third (<= -6.99), forecast day Fri-Sun THEN expect next-day vol-scaled return +0.100 sigma vs the average (in-sample), +0.157 sigma holdout"
  when: "daily (1d) decision at 00:00 UTC for a universe-L crypto perp; H-MINE2 state T3xT3:lrv_qxdow_sin (edges fitted on llm_in_sample)"
  then: "treat the next-day vol-scaled return as tilted by about +0.10 sigma; context for the decision layer only, not a reason to change risk limits"
  scope:
    assets: "crypto perps (universe L)"
    horizon: "1d"
    sides: "both"
  effect: "r1 effect +0.1005 (t 2.87, 5010 rows, 710 dates) in-sample, +0.1567 (t 1.91, one-sided p 0.0403, 734 rows, 57 dates) holdout; date-clustered holdout p 0.0586; BH-FDR 10% discovery among 495 effects"
  confidence: "C"
  evidence: ["H-MINE2"]
  review: "register as a forward hypothesis; demote if the effect on dates after 2026-11-12 changes sign or falls below half; it contradicts R-DS-6, R-BUDGET-1, R-ML-5, R-PX-6 (no direction from states / weekday / regime), R-PX-9 (no weekend direction), which stay in force until a forward test confirms this"
  status: "provisional"
```
