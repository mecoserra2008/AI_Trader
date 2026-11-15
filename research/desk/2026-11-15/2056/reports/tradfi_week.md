# TradFi perps, week of Mon 2026-11-16 (W47): context, levels, scenarios, candidates
_Built 2026-11-15 21:01 UTC on snapshot 2056 (as_of 20:56). Levels: the repo's geometry code on Bybit public data at 21:01 (`tradfi_week_inputs.json`). Friday-close anchors: cTrader CFDs (read-only). Machine-readable: `tradfi_week.json`. (web) = web source. No sizes. Every candidate is pattern-only; the ledger theses are narrative context, not an edge (R-MAC-3)._

## 1. Context
**Rates, Fed, dollar.**
- 2y 4.86%, at the 98th percentile of its 1-year range. 10y 5.10% (FRED, Nov 12); 5.15% on Nov 13 (web).
- The curve has been bear-flattening since the hot October CPI (Nov 10).
- Real 10y 2.86% (p96). Broad dollar at p99; DXY 102.86 (web).
- Fed funds 3.75-4.00%. A 25 bp hike on Dec 9 is fully priced; 50 bp has 13.5% (CME, web).
- Credit: IG/BBB spreads wide (p92) and widening; HY narrowed; term premium -20 bp.

**Equities, AI concentration.**
- S&P 7561.67, 3.3% off its high; the user's 7000 is 7.7% (log) below. VIX 18.7.
- 20 days vs the S&P: NDX -3.1%, NVDA -27.1%, MU -12.4%, semis -16.2%.
- **This cash week was a rotation, not a flush.** MU 676 -> 779, SNDK 820 -> 886, SOXL 44.06 -> 58.40, QQQ 654.8 -> 680.4.
- **NVDA alone fell**, 179.89 -> 163.11 (low 149.39): Jensen Huang resigned on Nov 12, CFO Kress is interim CEO (NVIDIA 8-K, web).
- Crypto-equities broke with BTC over 7 days: MSTR -23.7%, COIN -10.9%, CRCL -14.1%.

**Gold, oil.**
- XAU -7.5% over 28 days, cell R8 (trend down), with real yields and the dollar near 1-year highs. Longs pay +28% annualised funding (p88).
- CL 102.74 (Nov open) -> 87.70 as the Iran-war supply shock fades (Gulf exports normalising, Iran talks; web). This lowers the inflation impulse but hasn't turned the front end. CL is in R4 compression at 87.3-88.2.

**Calendar (UTC)** (newsquawk, kiplinger, stocktitan, treasury.gov):

| Day | Events |
|---|---|
| Mon 16 | 13:30 Empire State; 14:30 NYSE open; **17:00 Warsh** (SF Fed conference, no guidance expected); PDD, BIDU before the open |
| Tue 17 | **13:30 retail sales** (consensus +0.2% m/m, control +0.4%; kiplinger says Monday); 14:15 industrial production; HD before the open; **NVDA results ~21:20, call 22:00** (1:20 pm PT per NVIDIA: earlier than the plan's 22:00) |
| Wed 18 | 13:30 housing starts; 15:30 EIA (standard time); **18:00 20y auction, $16bn**; **19:00 FOMC minutes**; TGT, LOW before the open; PANW after the close |
| Thu 19 / Fri 20 | 12:00 WMT; 13:30 claims, Philly Fed / 15:00 UMich final |

Fed speakers: Barr, Williams, Barkin, Musalem, Waller, Hammack (times unconfirmed). No other semis earnings this week.

**Blackouts (no new entries):**
- **Tue 20:20-23:00 (NVDA)** and **Wed 18:00-20:00 (FOMC minutes)**.
- Sun 22:45-23:30 (CME reopen).
- Mon 07:00-12:00 for commodities (R-TF-6).
- First two NYSE hours: no fades and no gap fills (R-TF-1), no opening-range breakouts (R-TF-3).

**Theses (ledger) and what moves them this week:**
- **(a) strengthening 4/5** and **(d) strengthening 7/9.** Pushed further by hot retail sales, hawkish minutes, a tailing 20y auction, or a bigger NVDA capex guide. (a) dies on a 2y move of -30 bp or more over 20 obs.
- **(b) active 4/6.** NVDA is the test. A miss that spreads to QQQ strengthens it. It dies on the S&P back within 0.5% of its high with the NDX leading, or VIX < 14.
- **(c) weakening 1/5.** Only an oil-led drop in the front end would start it.
- **Claude's lower-low thesis: active.** BTC 64.1k vs the 61,899 low.

**Weekend.**
- The perps traded through Sunday 01:00 UTC's BTC flush.
- Weekend moves carry no Monday information (R-PX-9, R-TF-7). Don't buy the crypto-equity gaps back at the open (R-TF-1). The Friday close is revisited only ~22-28% of the time (R-TF-2).
- NVDA's mark is 48 bp over a sticky weekend index.
- Predicted funding (longs pay if it settles; last settled 0 / -65 / 0 / 0): NVDA +302%, SOXL +288%, KORU +261%, CRWV +255% annualised.

## 2. Technicals
How to read the table:
- **Tier**: risk.yaml tier on weekend turnover (re-tiered at order time); spread in bp.
- **Drift**: Bybit Friday close (21:00 UTC; 22:00 for commodities) -> last.
- **ATR**: Wilder on 24/7 bars.
- **Week Nov 9-15**: the prior week from Monday.
- **Funding**: last / predicted annualised % (90-day percentile); basis = mark vs index.
- **Levels**: unswept 4h/1d swing high / low (SH/SL), untouched leg origins (leg), period, profile and Friday (Fri) levels, in daily ATRs.

| sym | tier, spread | Fri -> last (drift) | ATR 1d (1h) | Nov 9-15 O / H / L | 7d POC [VAL-VAH] | funding; basis | above (1d ATR) | below (1d ATR) |
|---|---|---|---|---|---|---|---|---|
| NVDA | mid, 0.6 | 163.11 -> 165.33 (+1.35%, +0.22A) | 10.17 (1.54) | 179.89 / 183.52 / 149.39 | 161.09 [157.94-174.29] | 0 / +302, p66; +48 bp | 171.84 4h SH +0.64; 175.82 4h leg +1.03; 179.02-179.89 +1.35 | 161.09 POC -0.42; 156.15 4h SL -0.90; 149.39 1d SL -1.57 |
| MU | mid, 0.13 | 779.42 -> 765.41 (-1.81%) | 55.3 (5.6) | 676.0 / 828.6 / 662.6 | 765.2 [710.8-807.8] | 0 / +60; 0 | 779.42 Fri +0.25; 815.39 4h leg +0.90; 828.59 SH +1.14 | 740.42 4h SL -0.45; 714.81 4h leg -0.91; 700.86 SL -1.17 |
| SNDK | mid, 0.12 | 885.89 -> 863.92 (-2.51%) | 95.9 (7.3) | 820.1 / 1012.3 / 791.4 | 861.8 [832.4-943.2] | 0 / +53; 0 | 885.89 Fri +0.23; 943.85 4h leg +0.83; 967.95 SH +1.08 | 830.58-828.34 4h SL -0.35; 802.27 4h leg -0.64; 791.38 1d SL |
| SKHYNIX | illiquid, 0.11 | 897.68 -> 887.32 (-1.16%) | 75.3 (6.7) | 838.9 / 999.8 / 822.8 | 896.1 [840.3-924.5] | +7 / +81; +4 | 897.68 Fri +0.14; 924.54 VAH +0.49; 987.61 SH +1.33 | 853.53 4h SL -0.45; 825.28 4h leg -0.82; 822.82 1d SL -0.86 |
| SOXL | liquid, 1.77 | 58.40 -> 56.48 (-3.34%) | 11.14 (0.93) | 44.06 / 69.77 / 41.35 | 55.99 [52.07-64.62] | -65 / +288, p3; -2 | 58.40 Fri +0.17; 64.62 VAH +0.73; 67.93 4h leg +1.03 | 52.67-52.44 4h SL -0.35; 51.04 4h SL -0.49; 44.06 -1.11 |
| SOXS | mid, 1.9 | 50.83 -> 52.53 (+3.29%) | 8.67 (0.79) | 74.15 / 78.42 / 41.84 | 47.88 [43.73-55.51] | +59 / -220, p97; +2 | 55.51 VAH +0.34; 56.02 SH +0.40 | 51.87 Nov open -0.08; 50.83 Fri -0.20 |
| QQQ | illiquid, 0.15 | 680.43 -> 675.73 (-0.69%) | 15.9 (1.91) | 654.8 / 695.7 / 649.3 | 674.94 [668.73-687.02] | 0 / 0; +6 | 680.43 Fri +0.30; 685.51 Oct low +0.62; 690.13 4h leg +0.91 | 668.73 VAL -0.44; 667.08 4h SL -0.54; 654.80 -1.32; 649.32 1d SL -1.66 |
| MSTR | mid, 0.99 | 111.77 -> 100.65 (-10.48%, -1.27A) | 8.75 (1.42) | 124.68 / 129.55 / 95.53 | 122.74 [107.86-129.38] | -90 / +83, p2.5; +15 | 107.52 4h leg +0.79; 111.77 Fri +1.27; 114.76 4h leg +1.61 | 95.53 4h SL -0.59; 92.24-89.40 four 1d SL -0.96..-1.29 |
| COIN | illiquid, 4.18 | 127.79 -> 119.87 (-6.40%) | 8.49 (1.11) | 131.39 / 139.87 / 113.81 | 131.31 [120.76-139.88] | -26 / +28, p2; +8 | 124.09 4h leg +0.50; 127.79 Fri +0.93 | 113.81 4h SL -0.71 |
| CRCL | mid, 3.43 | 62.29 -> 58.32 (-6.59%) | 4.63 (0.70) | 66.24 / 69.90 / 54.89 | 65.97 [60.16-69.75] | 0 / +28; -31 | 60.60 4h leg +0.49; 62.29 Fri +0.86 | 54.89 4h SL -0.74 |
| CRWV | none (2.6M < 3M floor), 5.76 | 53.17 -> 52.12 (-1.99%) | 5.77 (0.48) | 45.62 / 62.12 / 44.13 | 56.78 [49.90-58.23] | 0 / +255; +23 | 55.04 Nov open +0.51; 56.73 4h leg +0.80 | 51.33 Oct low -0.14; 49.34 4h SL -0.48; 47.38 4h leg -0.82 |
| TSLA | illiquid, 0.26 | 390.65 -> 389.60 (-0.27%) | 15.1 (1.31) | 355.5 / 404.7 / 354.0 | 389.3 [377.9-404.4] | 0 / +7; +7 | 399.60 Nov open +0.66; 400.54 4h leg +0.73; 404.68 SH +1.00 | 382.20 4h SL -0.49; 381.44 4h leg -0.54; 370.78 SL -1.25 |
| KORU | illiquid, 13.5 | 7.67 -> 7.43 (-3.18%) | 1.38 (0.13) | 6.65 / 8.96 / 6.22 | 7.46 [6.82-8.23] | 0 / +261; +81 | 7.67 Fri +0.17; 8.23 VAH +0.58; 8.96 SH +1.11 | 6.80 4h SL -0.46; 6.27 4h leg -0.84; 6.22 1d SL -0.88 |
| XAU | mid, 0.03 | 3892.42 -> 3859.05 (-0.86%, -0.43A) | 77.0 (12.5) | 4063.0 / 4156.6 / 3777.7 | 3884.2 [3778.0-4053.9] | +28 / +66, p88; +1 | 3881-3899.53 confluence +0.28..+0.53; 3951.03 4h leg +1.19 | 3777.97 VAL / 3777.66 4h SL -1.06 |
| XAG | mid, 2.08 | 48.84 -> 48.05 (-1.63%) | 1.82 (0.26) | 51.73 / 53.83 / 46.15 | 50.16 [49.31-53.73] | 0 / 0; +2 | 48.74-48.98 cluster +0.38..+0.51; 49.46 Nov open +0.77 | 46.15 4h SL -1.04 |
| CL | mid, 1.14 | 87.66 -> 87.70 (+0.05%) | 3.43 (0.26) | 90.23 / 91.23 / 83.48 | 87.32 [84.89-88.22] | 0 / 0; -6 | 88.19 / 88.56 4h SH +0.14 / +0.25; 90.23 +0.74; 91.00 4h leg +0.96 | 86.91 Oct low -0.23; 86.71 -0.29; 84.33 4h SL -0.98 |
| BZ | illiquid, 1.07 | 93.38 -> 93.61 (+0.25%) | 3.72 (0.26) | 97.08 / 97.85 / 88.28 | 93.37 [90.64-94.52] | 0 / 0; -6 | 94.61 4h SH +0.27; 96.56 Oct low +0.79 | 92.58 4h SL -0.28; 90.45 4h leg -0.85; 89.59 SL -1.08 |

**Cross-venue anchor: cTrader CFDs (BlackBull).**
- Friday close = the CFD's h1 bar ending 21:00 UTC, the NYSE close (22:00 for metals and oil).
- Drift = 100 ln(Bybit last / CFD Friday close).
- Premium = Bybit Friday close vs CFD Friday close.

| CFD | Fri close | drift vs CFD | Bybit premium at the Friday close | cash week Nov 9-13 O / H / L | Nov open; Oct low |
|---|---|---|---|---|---|
| NVDA | 162.69 | +1.61% (+0.26A) | +26 bp | 173.99 / 177.52 / 148.80 | 207.61; 200.92 |
| MU | 778.11 | -1.65% | +17 bp | 717.81 / 827.43 / 699.67 | 890.04; 849.33 |
| QQQ | 679.66 | -0.58% | +11 bp | 672.81 / 695.55 / **667.06** | 695.36; 683.45 |
| MSTR | 111.55 | -10.28% (-1.25A) | +20 bp | 121.86 / 126.26 / 108.25 | 131.89; 124.27 |
| COIN / TSLA | 127.55 / 390.11 | -6.21% / -0.13% | +19 / +14 bp | 132.31 / 138.84 / 125.37; 370.93 / 404.25 / 370.93 | 140.26; 136.88 / 404.05; 349.33 |
| XAUUSD / XAGUSD (spot) | 3880.28 / 48.70 | -0.55% / -1.34% | +31 / +29 bp (futures carry) | 4138.1 / 4146.5 / 3852.4; 53.39 / 53.71 / 48.48 | 3882.57; 3870.08 / 49.33; 48.85 |
| WTI / BRENT (cash) | 89.26 / 98.42 | not comparable | -181 / -526 bp: a different reference | 86.86 / 90.01 / 84.37 | 103.55; 88.75 |

What the CFD anchors show:
- **The weekend drift is confirmed:** within 0.3 pp of the Bybit-based figure for every stock. The equity perps closed Friday 11-26 bp rich to cash.
- **QQQ:** the CFD cash-week low 667.06 equals the Bybit 4h swing low 667.08, so TF-1's level is the same on both venues.
- **XAU:** the spot anchors (Friday close 3880.28, Nov open 3882.57, Oct low 3870.08) sit at the bottom of the perp's 3881-3899.53 zone.
- **Cash ranges are narrower than 24/7 ranges** (MSTR cash low 108.25 vs perp 95.53).
- **NAS100:** 27,944.35 at the NYSE close; 27,841.25 is the later CFD session close (the coordinator's figure). Week Nov 9-13: 27,970.6 / 28,597.35 / 27,415.15; Nov open 28,503.35.

## 3. Scenarios
**NVDA (the week's pivot).** No entry before the print; the perp sits inside 7d value [157.94-174.29], POC 161.09.
- **Implied move:** options price ±9.6% to ±10.8% (web, Nov 12), against a 10-year average of ±7.8%. August moved +8.7% against ±5.9% implied. Deribit lists no NVDA options.
- **Bands:** from the Bybit Friday close 163.11, ±9.6% = **147.45 / 178.77** and ±10.8% = **145.49 / 180.73**. From the CFD close 162.69: 147.07 / 178.31.
- **Bear** (a miss, a guide cut, or the CEO succession not settled on the call): 156.15, then the 149.39 stop pool. Below that, no snapshot levels: price discovery. R-SIG-2: big surprise days don't continue on average.
- **Bull** (a beat and continuity): 171.84, then 174.29 / 175.82, then 178.50-179.89 (Nov 2-8 low, 4h leg, Nov 9 open), then 183.52. No thesis supports this side, and longs pay the weekend premium.

**The rest.**
- **AI cluster** (SOXL, MU, SNDK, SKHYNIX, KORU, CRWV, QQQ): one bet; the print sets its direction. Memory's rebound against NVDA's fall can close either way.
  - Before the print QQQ is in R5 balance at 668.73-687.02.
  - Below 667.08 it opens 654.80, then 649.32. Above 685.51 it points to 695.11-695.72.
- **Crypto-equities** follow BTC:
  - BTC below 63k points MSTR to 95.53, then 92.24-89.40.
  - A BTC reclaim of 65.9k points it to 107.52-107.86.
  - No opening fade (R-TF-1).
- **Metals** ride real yields via retail sales, the 20y auction and the minutes.
  - XAU rejection at 3881-3899.53 points to 3777.97 / 3777.66; acceptance above 3899.53 opens 3951.03.
  - XAG's matching zone is 48.74-48.98.
- **Oil:** compression. A 1h close above 88.56 or below 86.71 ends it (EIA Wednesday, Iran headlines). No trade inside the box.

## 4. Candidates
- Constraints: at most one per cluster and one per symbol; all are shorts, and the risk engine allows 2 same-side positions.
- Execution: pegged limit at the touch, resting MarkPrice stop.
- RR and the driftless P(target first) come from the desk validator.

| id | trade, window | trigger | entry zone | stop | target | RR (P driftless) | thesis; rules |
|---|---|---|---|---|---|---|---|
| TF-1 | **QQQ short**, Mon 16:30 to Tue 20:15 UTC (time stop before the NVDA blackout) | 1h close below 667.08 (unswept 4h SL under the 7d VAL: out of the R5 balance); retest from below within 60 min | 667.08-668.73 | 674.94 (7d POC) | 654.80 (ahead of 651.02 / 649.32) | 1.59-2.28 (0.39-0.31) | b, d; R-TF-1/3, R-PX-4, R-PX-12, R-MAC-3 |
| TF-2 | **NVDA short after the print**, Wed 16:30-18:00, then Wed 20:00 to Fri 20:00 UTC; only if TF-1 is flat | 1h close below 156.15 after the first two cash hours; void if 149.39 has already traded (no chase) | 156.15-157.94 | 161.09 (24h + 7d POC) | 150.00 (ahead of the 149.39 pool) | 1.29-2.61 (0.44-0.28); **only at RR ≥ 1.5** | b, d; R-TF-1/2, R-SIG-2, R-VOL-3, R-EXE-1 |
| TF-3 | **XAU short**, Mon 12:00 to Fri 20:00 UTC | trades into 3881-3899.53, then a 15m close back below 3881 | 3881.00-3888.34 | 3951.03 (untouched 4h leg) | 3777.97 (7d VAL, ahead of 3777.66) | 1.51-1.80 (0.40-0.36) | a, d; R-TF-6, R-PX-4, R-MAC-1/3, R-AVOID-1 |
| TF-4 | **MSTR short instead of BTC Plan A** (same bet, never both), NYSE hours | BTC 15m close below 63,000 **and** MSTR 1h close below 95.53 | 95.53-97.60 | 98.56 (1h swing low) | 92.24 (ahead of the four 1d lows) | 1.12-5.77 (0.47-0.15); **skip at the 95.53 touch** | deep, a; R-RB-5, R-TF-1, R-MM-2, R-PD-1 |

- **All four are pattern-only.** The tested rules only say what to avoid (R-TF-1/2/3/4/6, R-PX-9). With no edge, E[R] = cost < 0 at the driftless P.
- **What makes each case better than that:** a trapped crowd behind a structure break (TF-1, TF-2); late reclaim longs paying +28% funding into a 4-level confluence (TF-3); the BTC trigger (TF-4).
- **What would make them wrong:**
  - Levels break slightly more often than they hold (R-PX-4).
  - Post-flush continuation shorts are about breakeven (R-RB-5).
  - MSTR shorts already pay funding (p2.5).
  - A soft retail sales print squeezes gold.
- **Not proposed:** NVDA long (no thesis; longs pay the premium), CL / BZ (chop), KORU / CRWV / COIN (illiquid or no tier), anything at the weekend (R-PX-9).
- **Scope.** PLAN_2026-11-15 covers BTC / ETH only. TradFi entries need the user's explicit plan extension, which makes a new plan version.

## 5. Watch list (all 30 alerts are in `tradfi_week.json`)
- **QQQ:** 1h close below 667.08 (TF-1); above 685.51 (TF-1 void); below 649.32 (thesis b).
- **NVDA (after the print):**
  - 1h close below 156.15 (TF-2); crosses 149.39 (TF-2 void before the fill).
  - 1h close above 171.84 (bull case); crosses 179.02.
  - Exits the ±9.6% band 147.45 / 178.77.
- **XAU:** enters 3881-3899.53 (arms TF-3); 15m close below 3881 (TF-3); 4h close above 3899.53 (invalidation); crosses 3777.66.
- **MSTR + BTC:** 1h close below 95.53 together with BTC 15m below 63,000 (TF-4).
- **Friday-close revisits** (context, R-TF-2): MSTR 111.77, COIN 127.79, CRCL 62.29.
- **AI cluster:** SOXL below 51.04 / above 64.62; MU below 740.42 / above 828.59; SNDK below 828.34 (1h closes).
- **Oil, silver:** CL 1h close above 88.56 / below 86.71; XAG crosses 46.15.
- **Funding / basis:**
  - Any TradFi perp settling beyond ±100% annualised. First settlement Mon 00:00 UTC; NVDA, SOXL, KORU and CRWV are predicted +255..+302%.
  - MSTR funding at or below p5.
  - Mark vs index beyond 25 bp after 14:30 UTC.
- **Time:** Tue 20:15 (TF-1 time stop), 20:20 (cancel resting entries for the NVDA blackout); Wed 18:00 (FOMC blackout).

## Data gaps
- Retail sales date conflict between sources.
- Fed speaker, EIA and auction times not confirmed.
- The NVDA implied move dates from Nov 12.
- No overnight-sigma tool was run, so R-TF-1 is applied conservatively.
- The 1d ATR on 24/7 bars understates cash ranges.
- The alt desk's ZEC short (`proposals.json`) competes with TF-4 for the crypto-beta bet.
