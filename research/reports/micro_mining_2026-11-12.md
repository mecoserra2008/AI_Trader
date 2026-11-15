# FDR-controlled mining of discretised micro states (H-MINE1)

_Program 2026-11-12, agent E. Code: `src/ai_trader/research/micro_mining.py`, `src/ai_trader/research/micro_mining_report.py`, `scripts/micro_mining.py`; tests: `tests/test_micro_mining.py`. Data: agent A's aggregates `data/research/micro_days/` (read only). Frozen discoveries: `data/research/mining/discoveries_frozen.json`. These are deterministic research tools; the LLM decision layer is not tested here._

## Protocol

Each step was fixed in advance:

1. The family was declared in code and in the diary (12:33 UTC).
2. Discovery ran on the 16 days of batches 1-3 plus the pilot.
3. BH at FDR 10% was applied over the whole family; effects that could not be tested count with p = 1.
4. The discovery list was frozen in JSON and in the diary (12:34 UTC) before any confirmation day was produced. `build_day` refuses a confirmation date while the frozen file is missing.
5. Confirmation ran once.

**States.** Every state is point in time within the symbol-day: a mid-rank among the same symbol-day's earlier minutes, after a warm-up of 60 values.

- flow quintile: notional imbalance (buy - sell) / (buy + sell) of the last minute (5 minutes in T5);
- vol tercile: log RV of 5 s last-trade returns over the last 15 minutes (30 in T5);
- spread regime (L2 symbols only): time-average spread over the minute, tick-bound if <= 1.5 ticks, free otherwise;
- hour block: 00-06, 06-12, 12-16 or 16-24 UTC;
- class: major, alt or TradFi;
- for alts, the BTC / ETH / SOL flow quintile and vol tercile of the same minute.

**Targets.** Next-1m and next-5m |return| (vol) and signed return (direction), from last-trade closes. Each is normalised per symbol-day (|r| / mean, r / sd) and demeaned. Units:

- vol effects are a fraction of the symbol-day mean |r| (+0.30 = 30% above the day's average minute);
- direction effects are in symbol-day sd units.

**Inference.**

- One estimate per day, pooled over the day's symbols; a day-cell needs at least 30 rows.
- t-test over days with df = D - 1, two-sided for discovery.
- An effect is testable if at least 10 of the 16 discovery days give an estimate.
- Confirmation needs at least 5 days; an effect that can't be tested there counts as not replicated.
- A symbol-day enters only if at least 75% of its minutes have a print.

### Effect family

| family | definition | effects | testable on discovery | discoveries (vol / direction) |
|---|---|---|---|---|
| F1 | own state cross: class x flow quintile x 15-min vol tercile x hour block (cell vs symbol-day mean) | 720 | 680 | 203 / 22 |
| F2 | flow contrast: top minus bottom flow quintile within class x vol tercile x hour block | 144 | 136 | 1 / 5 |
| F3 | spread regime (L2-book symbols): spread (tick-bound / free) x flow quintile x vol tercile | 120 | 112 | 46 / 8 |
| F3c | spread contrast: free minus tick-bound within flow quintile x vol tercile | 60 | 52 | 6 / 2 |
| F4a | leader state for alts: BTC / ETH / SOL same-minute flow quintile x vol tercile | 180 | 180 | 71 / 7 |
| F4b | leader x own flow for alts: leader flow quintile x own flow quintile | 300 | 300 | 112 / 19 |
| F5 | 5-minute table: class x 5-min flow quintile x 30-min vol tercile x hour block (next-5m targets) | 360 | 150 | 44 / 4 |
| total | | 1884 | 1610 | 483 / 67 |

Untestable effects are mostly F5 cells (one row per 5 minutes is too few rows per day-cell) and TradFi cells on weekend days.

## Discovery-day descriptives (marginals, not part of the tested family)

926,839 T1 rows; alt 707,719 rows / 35 symbols, major 108,880 rows / 5 symbols, tradfi 110,240 rows / 11 symbols. The vol tercile stays the same from one minute to the next 92% of the time.

| state | next-1m \|r\| vs day mean (day-clustered SE) |
|---|---|
| vol tercile low | -0.236 (0.017) |
| vol tercile mid | -0.086 (0.017) |
| vol tercile high | +0.278 (0.017) |
| hour 00-06 UTC | -0.108 (0.029) |
| hour 06-12 UTC | -0.131 (0.022) |
| hour 12-16 UTC | +0.251 (0.056) |
| hour 16-24 UTC | +0.038 (0.033) |
| flow most net selling (Q1) | -0.084 (0.007) |
| flow Q2 | +0.073 (0.009) |
| flow Q3 (balanced) | +0.046 (0.009) |
| flow Q4 | +0.059 (0.008) |
| flow most net buying (Q5) | -0.095 (0.011) |

- **Extreme flow minutes are quiet.** Next-minute |r| is about 9% below the day mean after both flow extremes, because a one-sided minute is usually a minute with few prints.
- **Vol and hour carry the vol structure.** The 15-minute vol tercile spans about -24% to +28%, and 12-16 UTC is +25%.

| next-1m return (sd units) | Q1 | Q2 | Q3 | Q4 | Q5 |
|---|---|---|---|---|---|
| major, own flow | +0.011 | +0.007 | -0.009 | -0.007 | -0.003 |
| alt, own flow | +0.008 | +0.003 | -0.011 | -0.001 | -0.000 |
| tradfi, own flow | -0.028 | -0.012 | +0.004 | +0.009 | +0.025 |
| alts, BTC flow (same minute) | -0.019 | -0.003 | -0.009 | +0.003 | +0.028 |
| alts, BTC flow -> next-5m | -0.010 | -0.004 | +0.009 | +0.004 | -0.000 |

Share of minutes with a free spread (L2 symbols): SOL 0.00, XRP 0.00, DOGE 0.00, BTC 0.01, ETH 0.01, XAU 0.02, ATOM 0.11, NVDA 0.17, HYPE 0.22, KAS 0.35, QQQ 0.41, CARV 0.66.

## Discoveries (frozen)

550 discoveries; the largest BH-rejected p is 0.0289. The strongest 12 vol and 12 direction discoveries:

| id | reading | size | t (16 days) |
|---|---|---|---|
| F1:a1:alt:f0:v0:h3 | alts, own flow most net selling (Q1), recent vol low, 16-24 UTC: next-1m |return| | -0.228 | -16.8 |
| F3:a1:sp0:f2:v2 | L2-book symbols, tick-bound spread, own flow Q3 (balanced), recent vol high: next-1m |return| | +0.367 | 14.6 |
| F3:a1:sp0:f0:v0 | L2-book symbols, tick-bound spread, own flow most net selling (Q1), recent vol low: next-1m |return| | -0.273 | -13.2 |
| F1:a1:alt:f4:v0:h3 | alts, own flow most net buying (Q5), recent vol low, 16-24 UTC: next-1m |return| | -0.240 | -13.1 |
| F1:a1:alt:f3:v2:h3 | alts, own flow Q4, recent vol high, 16-24 UTC: next-1m |return| | +0.305 | 13.0 |
| F5:a5:alt:f2:v2:h3 | alts, own flow Q3 (balanced), recent vol high, 16-24 UTC: next-5m |return| | +0.275 | 12.8 |
| F1:a5:alt:f3:v2:h3 | alts, own flow Q4, recent vol high, 16-24 UTC: next-5m |return| | +0.265 | 12.8 |
| F3:a1:sp0:f2:v0 | L2-book symbols, tick-bound spread, own flow Q3 (balanced), recent vol low: next-1m |return| | -0.268 | -12.8 |
| F3:a1:sp0:f4:v0 | L2-book symbols, tick-bound spread, own flow most net buying (Q5), recent vol low: next-1m |return| | -0.271 | -12.7 |
| F3:a5:sp0:f4:v0 | L2-book symbols, tick-bound spread, own flow most net buying (Q5), recent vol low: next-5m |return| | -0.230 | -12.2 |
| F3:a5:sp0:f2:v0 | L2-book symbols, tick-bound spread, own flow Q3 (balanced), recent vol low: next-5m |return| | -0.222 | -12.2 |
| F1:a1:alt:f0:v0:h1 | alts, own flow most net selling (Q1), recent vol low, 06-12 UTC: next-1m |return| | -0.246 | -12.1 |
| F4b:s1:BTC:lf4:f0 | alts given BTC, BTC flow most net buying (Q5), own flow most net selling (Q1): next-1m return | +0.049 | 5.8 |
| F4b:s1:SOL:lf0:f2 | alts given SOL, SOL flow most net selling (Q1), own flow Q3 (balanced): next-1m return | -0.054 | -5.3 |
| F5:s5:alt:f3:v2:h3 | alts, own flow Q4, recent vol high, 16-24 UTC: next-5m return | -0.134 | -5.0 |
| F4b:s1:BTC:lf0:f2 | alts given BTC, BTC flow most net selling (Q1), own flow Q3 (balanced): next-1m return | -0.038 | -4.9 |
| F4b:s1:SOL:lf4:f0 | alts given SOL, SOL flow most net buying (Q5), own flow most net selling (Q1): next-1m return | +0.064 | 4.9 |
| F4b:s1:ETH:lf0:f2 | alts given ETH, ETH flow most net selling (Q1), own flow Q3 (balanced): next-1m return | -0.042 | -4.7 |
| F4b:s1:ETH:lf0:f4 | alts given ETH, ETH flow most net selling (Q1), own flow most net buying (Q5): next-1m return | -0.054 | -4.2 |
| F1:s5:alt:f0:v1:h1 | alts, own flow most net selling (Q1), recent vol mid, 06-12 UTC: next-5m return | +0.060 | 4.2 |
| F4b:s1:SOL:lf4:f4 | alts given SOL, SOL flow most net buying (Q5), own flow most net buying (Q5): next-1m return | +0.042 | 3.9 |
| F4a:s1:BTC:lf4:lv0 | alts given BTC, BTC flow most net buying (Q5), BTC vol low: next-1m return | +0.044 | 3.9 |
| F1:s1:tradfi:f3:v0:h0 | TradFi perps, own flow Q4, recent vol low, 00-06 UTC: next-1m return | +0.041 | 4.1 |
| F1:s5:alt:f2:v1:h1 | alts, own flow Q3 (balanced), recent vol mid, 06-12 UTC: next-5m return | +0.053 | 3.7 |
