# Break-of-structure breakout and momentum by regime, timeframe and execution (H-BOS1-5), 2026-11-15

Agent BOS. Research only: no orders and no trading calls.
- **Registration and design.** The hypotheses were registered before any test (research/hypotheses.yaml, block BOS). The grid, execution arms, regimes and multiple-testing plan were frozen in the diary (2026-11-15 18:05 UTC, kind decision) and in `data/research/bos/design.json` before any outcome was computed.
- **Code.**
  - `src/ai_trader/research/bos_download.py` (5m klines);
  - `bos_regimes_study.py` (trade simulation, per symbol and timeframe, checkpointed);
  - `bos_regimes_analysis.py` (cell statistics);
  - `bos_regimes_infer.py` (BH, DSR/PBO, tails, stability, allocator, execution);
  - `bos_regimes_report.py` (figures, verdicts, records).
- **Tests:** `tests/test_bos_regimes.py` (5).
- **Machine-readable results:** `research/reports/bos_regimes_2026-11-15.json` (full: `data/research/bos/results.json`).

## Verdicts

| hypothesis | llm_in_sample | clean_holdout | pooled | key numbers |
| --- | --- | --- | --- | --- |
| H-BOS1: some (variant, arm, regime) cell has an edge net of costs | contradicts | inconclusive | contradicts | **0 of 17,460** tested in-sample cells survive BH 10% on the positive side (0 of 15,185 in the holdout). 8,210 survive on the **negative** side, and 84% of those are still negative in the holdout. Best cell: 1h Donchian-20 CHoCH, time exit, pegged, calmest vol tercile, +0.234 R/trade (z 3.25, n 8,103), DSR 0.00 against 17,460 trials. |
| H-BOS2: explosive in some regimes, detrimental in others | supports | supports | supports | **Explosive in both segments (top-5% share > 100%):** swing R1L, R3L, R4L, R7L and vol_T1 (market). **Detrimental in both segments (z < -2):** intraday R3S, R4S, R8L, R9L, vol_T1, vol_T3, trend_S1/S2. The explosive cells are not significantly positive (holdout z < 1). Within swing, the best-minus-worst regime gap (R4S - R9S, +0.41 R, z 3.98 in-sample) reverses in the holdout (-0.16 R). |
| H-BOS3: unstable over time | contradicts | inconclusive | contradicts | Among the 10 in-sample-best cells, **0 flip sign in >= 1/3 of half-years** (flip share 0.18-0.20). 6 of 10 show a 5% break (CUSUM or sup-F), dated 2023-06-26 (R1S cells) or 2022-07-10. Yet **7 of 10 flip sign in the holdout**. |
| H-BOS4: Bayesian on/off allocator profits from the instability | contradicts | inconclusive | contradicts | Pegged arm, walk-forward 2023-2026H1: allocator Sharpe -0.38 vs static best +0.25 vs equal weight -4.60. Allocator minus static t -0.69; DSR 0.004. Holdout: -0.90 vs -1.11 vs -4.90; vs static t 0.34, vs EW t 2.82. |
| H-BOS5: retest limit beats pegged and market | supports | inconclusive | inconclusive | **Intraday in-sample:** retest - market +0.068 R [+0.057, +0.078] and retest - pegged +0.018 [+0.007, +0.027]. **Holdout:** retest - pegged CI includes 0. **Swing:** no difference in either segment. The retest misses 24% of signals and 35-39% of the market arm's top-5% trades. |

Trials: 240 variants x 3 arms x 25 regime cells = 18,000 cells per segment. 17,460 cells met the n >= 30 / weeks >= 10 bar in-sample and 15,185 in the holdout; the sensitivity arms (retest N = 24, retest best case) are not counted in BH. Each verdict is recorded with `ai_trader.research.hypotheses.record` for llm_in_sample, clean_holdout and pooled. Treat |z| ~ 2 as interesting, not proven.

## In plain words

**The short answer: no break-of-structure variant has an edge in any regime that survives multiple testing.**
- That holds over 240 variants, 3 ways of entering and 25 regime cells.
- Intraday (5m/15m), every regime loses after costs.
- Swing (1h/4h) is a near-zero bet overall. It earns modestly and noisily in calm regimes, where a few large trades carry the average, and it bleeds clearly in high-volatility, whipsaw and liquidation regimes.

The user's instinct is right in one sense: the family is not uniformly useless. Its sign depends on the regime, and that pattern is stable in its broad outline. But nothing in the positive regimes is strong enough to call an edge.

**Where it has some power (swing only, never significant after multiple testing):**
- **Calm volatility (4h sigma in the bottom tercile of its trailing year).** Each of the top 20 in-sample cells by z is a swing cell that is either calm-vol or a short in a quiet trend (R1S, R7S).
  - Swing pool, pegged entry: +0.111 R per trade in-sample (z 2.25), +0.064 in the holdout (z 0.20).
  - The 1h Donchian-20 CHoCH with a time exit earns +0.23 R in both segments (n 8,103 / 740) and +0.14 R on the TradFi perps in the holdout (n 96).
  - It is the best of 17,460 cells, so its deflated Sharpe is 0.00. It is a candidate for a forward test, not a finding.
- **Long breakouts in quiet trends and compression (R1L, R4L, R7L).** Positive in both segments: +0.155 / +0.154, +0.120 / +0.671 and +0.051 / +0.361 R. Holdout z is below 1 everywhere.
- **Time exits and the 4h/1h timeframes.** Pooled over regimes, the time-only exit is the best exit on every timeframe (1h +0.043, 4h +0.048 R in-sample). The 2.5-sigma trail is among the worst exits (R-EXIT-1 again: trend profits need room).

**Where it is explosive (the average is made by the top 5% of trades):**
- Almost every positive swing cell is "explosive" by the registered definition: the other 95% of trades lose, and the cell is positive only because of its top 5%. Calm vol (T1) has a top-5% share of 3.3x the cell total in-sample and 7.1x in the holdout. R1L, R3L, R4L and R7L are 2.4-20x.
- Hit rates are 20-35% and payoff ratios 2-5. This is the classic trend payoff: many small stop-outs, rare large runs.
- **Explosive does not mean profitable.** The tail is large enough to make the mean positive, but not large or regular enough to make it significant.

**Where it is detrimental:**
- **All intraday cells.**
  - 5m/15m pooled, market entry: -0.173 R per trade in-sample (z -12.3) and -0.191 in the holdout (z -3.2).
  - Pegged entry: -0.123 / -0.130.
  - 6,088 of 9,000 tested intraday cells have z < -2 in-sample; not one has z > 1.64.
  - Costs are the main reason. A 4-sigma-of-1h stop is small next to a 15 bp round trip, and the 5m/15m signals have no edge to pay for it.
- **Swing in high volatility (T3).** -0.121 R in-sample (z -3.47), -0.104 in the holdout.
- **Swing in liquidation (R9), whipsaw (R6) and shorts inside a momentum burst (R3S).** -0.11 to -0.25 R in-sample (z -2 to -3.4), negative again in the holdout but with small samples.
- **Shorts in R4/R7 flipped.** Swing shorts in compression (R4S) and grind-down (R7S) were positive in-sample (+0.175, +0.065) and are clearly negative in the holdout: -0.288 (z -2.79) and -0.357 (z -2.60).

**Are the relationships stable?**
- **The broad regime shape is stable:** calm/quiet-trend longs positive, high-vol / whipsaw / liquidation negative, intraday negative, in both segments.
- **The cell-level picks are not.**
  - The 10 best in-sample cells were steady across half-years: none flipped in a third of them, so H-BOS3 contradicts as registered.
  - But 6 of 10 show a structural break, and 7 of 10 flip sign in the holdout. Those are the R1S and R7S short cells: +0.58 R in-sample, -0.46 R in the holdout.
  - The full-series Bai-Perron breaks fall on two dates: 2023-06-26 (four R1S cells, 16 days before a BTC v0 direction shift on 2023-07-12) and 2022-07-10 (one cell, within 30 days of Celsius/3AC). Of the 5 cell breaks, 4 lie within 30 days of a BTC regime shift and 1 of a frozen event. The chance rates are 40% and 35% of all days, so with two distinct dates the alignment is anecdotal.
- So the instability is in which narrow cell wins, not in the broad regime map. This matches R-SYN-2 and R-MAP-1: the regime map tells you where strategies lose, not which variant wins.

**Can we profit from the instability?** Not with the registered allocator.
- It turned variant x regime cells on and off weekly by a Bayesian posterior of their recent net R.
- It beat equal weight by a mile (EW is dragged down by the intraday variants), but it lost to simply holding the in-sample-best variant: Sharpe -0.38 vs +0.25 in-sample, -0.90 vs -1.11 in the holdout (paired t -0.69 / +0.34).
- Its inner selection always picked no forgetting (phi = 1.0): the data preferred the longest memory, which is the opposite of what an "exploit instability" story needs.
- DSR 0.004 over its own 5-point grid.

**Limit orders.**
- **Pegged limit at the touch:** better than market everywhere, by the fee gap: +0.013 R per signal on swing, +0.05 R intraday, CIs above zero in both segments.
- **Retest limit at the broken level:**
  - It misses about a quarter of signals.
  - It misses 35-39% of the market arm's top-5% trades, the explosive winners (46% on 4h in the holdout).
  - On swing it is no better than market (-0.002 R) and slightly worse than pegged (-0.015 R, CI includes 0).
  - Intraday it beats market by +0.07 R and pegged by +0.02 R in-sample. Those are fee savings and a better price on a book that still loses -0.105 R per signal.
  - Intrabar ordering does not matter here: the best-case retest (no stop on the fill bar) differs from the worst case by < 0.002 R.

## Selection steps: deflated Sharpe and PBO

| step | selected | trials | DSR | PBO (CSCV 16 blocks) | holdout of the pick |
| --- | --- | --- | --- | --- | --- |
| 1. best cell (variant x arm x regime) | 1h don20 CHoCH time, pegged, vol_T1 (weekly SR 1.39 ann.) | 17,460 | 0.00 (SR0 6.07) | 0.036 | +0.230 R, z 0.60, n 740 |
| 2. static best variant (ALL regimes) | 4h frac2 CHoCH time, pegged (SR 0.70 ann.) | 720 | 0.00 (SR0 9.13) | 0.015 | +0.011 R, z 0.04, n 852 |
| 3. allocator forgetting factor | phi = 1.0 in 8/8 blocks (pegged) | 5 | 0.004 | 0.19 | Sharpe -0.90 |
| 4. execution arm (EW book of all variants) | pegged, both classes | 3 | swing 0.50, intraday 0.00 | 0.00 | pegged best in the holdout too |

How to read the low PBO at steps 1-2:
- Most cells are intraday losers, so the in-sample winner almost always lands above the out-of-sample median. That is why PBO is low.
- The DSR of 0.00 is the binding number. The winner's Sharpe is far below what the best of 17,460 or 720 noise draws would reach (SR0).
- The grid's weekly Sharpe quantiles (0/10/50/90/100%) are -16.5 / -5.7 / -0.87 / +0.28 / +0.70, and only 23% of variant x arm books have a positive in-sample mean.

## TradFi perps (robustness, 1h)

TradFi history is short (mostly 2026) and its terciles need a year of history, so this is a weak check.
- **vol_T1 1h Donchian-20 CHoCH time-exit cells:** positive on T in both segments.
  - Pegged: in-sample +1.55 R (n 51), holdout +0.14 R (n 96).
  - Retest: in-sample +1.98 (n 34), holdout +0.32 (n 72).
- **R1S cells:** negative on T in-sample (n 8-17) and slightly positive in the holdout.

## Figures (research/figures/bos_regimes_2026-11-15/)

- `heatmap_variant_regime.png`: mean net R per trade, 60 variant groups (timeframe x structure x signal, mean over the 4 exits) x 25 regimes, pegged entry, both segments. Blue is positive, red negative, gray about zero. Intraday rows are red almost everywhere. Swing rows are blue in the calm and quiet-trend long columns and red in R9, R6, R3S and vol_T3.
- `tail_decomposition.png`: per regime, the per-trade contribution of the top 5% of trades (blue) against the other 95% (orange), and the cell mean (dot), swing and intraday, both segments.
- `rolling_stability.png`: half-year means of the four most distinct in-sample-best cells, with Bai-Perron breaks (dashed), frozen events (gray) and the holdout (aqua).
- `allocator_equity.png`: cumulative walk-forward P&L of the allocator vs the static best variant vs equal weight, pegged and market. Each book is scaled by its training-window daily sd, so early blocks with few active trades have large scaled swings.
- `execution_arms.png`: mean net R per signal by arm with 95% week-bootstrap CIs, intraday and swing, both segments.

## How a surviving cell would become a desk candidate generator (note only; nothing was built)

No cell survived, so nothing should feed the desk now. If a (variant, regime, execution) cell passes a pre-registered forward test (R-BOS-4's candidate is the obvious one), it would be wired in like this:

1. **Deterministic generator.** A function `bos_candidates(as_of)` in `ai_trader.strategies` or `tools`.
   - It reads closed bars through `Lake.read_asof` and the 4h regime through `regimes.label`, so nothing is timestamped after `as_of`.
   - It emits a candidate only when the frozen cell's conditions hold at the bar close: structure break of the stated type, regime cell and vol tercile.
   - Each candidate carries: side, the broken level, the structural stop level (clipped to 0.5-4 sigma units as here), the exit rule (e.g. time 10 days) and the execution tag (pegged).
2. **Its own pinned entry in `config/tools.yaml`.** Changing the cell's parameters starts a new track record.
3. **The trader agent decides; the risk engine sizes.** The candidate goes onto the trader agent's list like any other sleeve signal. The agent reads the regime card and the R-BOS rules and may take or skip it, but it never sets size. at-risk sizes from the stop distance and fixed risk per trade, applies the net-E[R] gate and the limits, and the pegged entry uses the existing `paper_entry` mechanics (R-EXE-1).
4. **Logged and compared forward.** Each candidate is journaled with the generator version and compared forward against the same cell's mechanical (no-LLM) outcome and the paired baselines, per CLAUDE.md principle 1.

## Caveats and what could be wrong

- **Bar-level fills.** No 1m bars were used. Stops are intrabar on the bar's low/high and gap-filled at the open. The only same-bar ambiguity (retest fill and stop in one bar) is bounded: best case minus worst case < 0.002 R per signal.
- **Pegged fills are assumed (R-EXE-1).** The 275 replayed signals filled within 37 s on BTC/ETH/SOL. On thin alts or fast breakouts, fills may be worse.
- **Retest queue.** The retest needs a 1 bp trade-through, which is a pessimistic queue assumption. A touch-only fill would raise the fill rate and lower the price advantage.
- **Overlapping trades.** Every signal is its own trade (no position overlap logic). Same-direction signals stack in the per-day books, and the variants overlap heavily (CHoCH and BoS+mom are subsets of BoS). The week-clustered bootstrap handles the dependence within a week, not across overlapping variants. The "family pool" n counts variant-trades, not independent trades.
- **Survivorship.** The universe is today's survivors (DATA_SOURCES.md).
- **Short holdout.** 4.5 months; most holdout z are below 1.
- **Data gaps and burn-in.** 5m data from 2022 only (Bybit public kline); HYPE 2024 is missing 11 bars; listing burn-in follows settings.research.
- **Regime labels.** They come from the symbol's own 4h frame. A BTC-wide regime might sort differently.
- **Stop cap.** The 4-sigma cap binds for most swing signals (median stop 4.0 sigma units for fractal k >= 3 and Donchian). The structural stop is therefore often the cap, not the structure.

## What was tested (frozen before any result)

The design was frozen in the diary (2026-11-15 18:05 UTC, kind decision) and in `data/research/bos/design.json` before any outcome was computed. Nothing was changed afterwards.

- **Universe and data.**
  - Swing: Bybit 1h and 4h bars from the lake, universe L (23 crypto perps), 2020-03 to 2026-11-15.
  - Intraday: Bybit public 5m klines for the same 23 symbols, 2022-01-01 to 2026-11-15, downloaded for this study (`data/research/bos/bars/5m`, manifest included, about 0.1 GB). 15m bars are built from the 5m bars.
  - TradFi perps (16 symbols, 1h; 4h only for PAXG/XAUT) are a robustness check. Their history is short (mostly 2026).
  - No vecopen data was read.
- **240 variants** = 4 timeframes x 5 structures x 3 signals x 4 exits.
  - Structures: fractal swings with k = 2, 3, 5 bars each side (the same swing and break logic as `features/vol_bos.structure`), and Donchian channels n = 20, 55 (first bar of a break only).
  - Signals:
    - BoS: every close beyond the last unbroken swing, traded in the break direction, long or short.
    - CHoCH: a BoS against the previous BoS.
    - BoS+mom: a BoS whose 20-bar vol-scaled return in the trade direction is above 1 sigma.
  - Initial stop: the opposite structure level, its distance clipped to 0.5-4 sigma units. A sigma unit is the 1h-equivalent sigma intraday and the 4h-equivalent sigma for swing. **1 R** is the distance from the signal close to that stop, and every execution arm of a signal uses the same 1 R.
  - Exits, each with the hard stop and a maximum hold (intraday 1 day, swing 10 days):
    - opposite structure break (close-based);
    - close-based trail of 2.5 or 4.0 sigma units from the best close;
    - time only.
- **Three execution arms.** Costs come from `config/risk.yaml`, and funding is charged.
  - **Market:** next open +2 bp, taker 5.5 bp.
  - **Pegged:** a limit joining the touch (R-EXE-1): next open +1.3 bp, maker 2.0 bp, always filled.
  - **Retest:** a limit at the broken level, live for 6 bars. It fills only if price trades through the level by at least 1 bp, and pays maker 2.0 bp. A missed fill counts as 0 R per signal.
  - Exits are taker plus 2 bp in every arm.
  - Two sensitivity arms: retest live for 24 bars, and a best-case retest that ignores the stop on the fill bar. The main retest assumes stop-first when the fill bar also touched the stop.
- **Regimes known at the signal**, all from the symbol's last closed 4h bar:
  - the v0 cell (R1..R9) crossed with the trade side (L = long, S = short);
  - the 4h sigma tercile within its trailing year (vol_T1 = calmest third);
  - the |v0 direction score| tercile within its trailing year (trend_S3 = strongest trend third);
  - ALL.

  That makes 25 regime cells.
- **Statistics.**
  - Each cell (variant x arm x regime) gets a mean net R per trade, with a week-clustered multiplier bootstrap (400 draws). All symbols in a week are resampled together.
  - A cell is tested only with at least 30 trades and at least 10 weeks.
  - BH at 10% within each segment.
  - DSR and PBO (CSCV, 16 blocks) at every selection step, on the common 2022-01 to 2026-06 weekly window.
  - Segments: llm_in_sample = signals before 2026-07-01, with trades exiting after the cutoff purged. clean_holdout = 2026-07-01 to 2026-11-15.


## H-BOS3 detail: the 10 in-sample-best cells

| variant | arm | regime | n in / ho | mean in | mean ho | half-years with flipped sign | CUSUM p | sup-F (in) | sup-F winsorised | full-series breaks | near event / BTC regime shift |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1h|don20|choch|time | pegged | vol_T1 | 8103 / 740 | +0.234 | +0.230 | 0.18 (11 halves) | 0.139 | 5.4 | 6.7 | - | - |
| 1h|don55|choch|time | pegged | R1S | 319 / 22 | +0.584 | -0.464 | 0.20 (10 halves) | 0.022 | 9.6 | 9.0 | 2023-06-26 | 2023-06-26: - / 2023-07-12 |
| 1h|don55|choch|time | market | R1S | 319 / 22 | +0.575 | -0.479 | 0.20 (10 halves) | 0.021 | 9.7 | 9.1 | 2023-06-26 | 2023-06-26: - / 2023-07-12 |
| 1h|don20|choch|time | market | vol_T1 | 8103 / 740 | +0.219 | +0.210 | 0.18 (11 halves) | 0.155 | 5.2 | 6.7 | - | - |
| 1h|don55|bos|time | pegged | R1S | 449 / 36 | +0.537 | -0.458 | 0.20 (10 halves) | 0.001 | 16.0 | 15.4 | 2023-06-26 | 2023-06-26: - / 2023-07-12 |
| 1h|don55|choch|tr40 | pegged | R1S | 319 / 22 | +0.350 | -0.187 | 0.20 (10 halves) | 0.152 | 6.2 | 6.0 | - | - |
| 1h|don55|bos|time | market | R1S | 449 / 36 | +0.528 | -0.474 | 0.20 (10 halves) | 0.001 | 16.1 | 15.5 | 2023-06-26 | 2023-06-26: - / 2023-07-12 |
| 1h|don20|choch|time | retest | vol_T1 | 5955 / 559 | +0.193 | +0.279 | 0.20 (10 halves) | 0.041 | 8.2 | 7.4 | - | - |
| 1h|don55|choch|tr40 | market | R1S | 319 / 22 | +0.341 | -0.202 | 0.20 (10 halves) | 0.151 | 6.2 | 6.0 | - | - |
| 1h|don55|bos|tr40 | pegged | R1S | 449 / 36 | +0.314 | -0.220 | 0.20 (10 halves) | 0.085 | 10.5 | 12.0 | 2022-07-10 | 2022-07-10: Celsius/3AC 2022-06-13 / - |

BTC regime shifts (zero crossings of the 90-day up-minus-down v0 share on BTC 4h, declustered 60 days): 2021-05-23, 2021-08-15, 2022-01-06, 2022-03-30, 2023-01-10, 2023-04-29, 2023-07-12, 2023-10-24, 2024-06-13, 2024-09-21, 2025-03-05, 2025-08-27, 2026-05-02, 2026-08-28. Sup-F critical value (Andrews, 15% trimming, 5%): 8.85.

## H-BOS4 detail: walk-forward blocks (pegged entries)

| test block | phi chosen in training | static best variant (training) | share of variant x cell units ON | Sharpe allocator | Sharpe static | Sharpe EW |
|---|---|---|---|---|---|---|
| 2023-01-01..2023-06-30 | 1.0 | 4h|don20|choch|time | 0.168 | -1.07 | +1.00 | -4.63 |
| 2023-07-01..2023-12-31 | 1.0 | 4h|frac2|bos|time | 0.148 | -2.44 | +0.59 | -8.30 |
| 2024-01-01..2024-06-30 | 1.0 | 4h|frac2|bos|time | 0.141 | +0.04 | -0.39 | -7.32 |
| 2024-07-01..2024-12-31 | 1.0 | 4h|frac2|bos|time | 0.167 | +1.72 | +3.71 | -2.41 |
| 2025-01-01..2025-06-30 | 1.0 | 4h|frac3|bos|opp | 0.207 | -0.87 | -5.80 | -2.22 |
| 2025-07-01..2025-12-31 | 1.0 | 4h|don20|choch|time | 0.215 | +0.64 | -0.41 | -4.84 |
| 2026-01-01..2026-06-30 | 1.0 | 4h|don20|choch|time | 0.210 | -0.71 | -1.14 | -4.51 |
| 2026-07-01..2026-11-16 | 1.0 | 4h|frac2|bos|time | 0.189 | -0.90 | -1.11 | -4.90 |

llm_in_sample: Sharpe allocator -0.38, static +0.25, EW -4.60; allocator - static -0.0318/day (NW t -0.69), allocator - EW +0.3271 (t +9.97), 1277 days; fixed-phi Sharpes 0.9: -2.69, 0.95: -1.87, 0.98: -0.98, 0.99: -0.87, 1.0: -0.38

clean_holdout: Sharpe allocator -0.90, static -1.11, EW -4.90; allocator - static +0.0398/day (NW t +0.34), allocator - EW +0.2730 (t +2.82), 139 days; fixed-phi Sharpes 0.9: -2.64, 0.95: -3.48, 0.98: -2.66, 0.99: -0.58, 1.0: -0.90

DSR (in-sample WF series, 5 trials) 0.004; PBO over the 5 fixed-phi books 0.19.

## H-BOS4 detail: walk-forward blocks (market entries)

| test block | phi chosen in training | static best variant (training) | share of variant x cell units ON | Sharpe allocator | Sharpe static | Sharpe EW |
|---|---|---|---|---|---|---|
| 2023-01-01..2023-06-30 | 1.0 | 4h|don20|choch|time | 0.144 | -1.84 | +0.94 | -6.52 |
| 2023-07-01..2023-12-31 | 1.0 | 4h|frac2|bos|time | 0.121 | -2.12 | +0.43 | -10.81 |
| 2024-01-01..2024-06-30 | 0.99 | 4h|don20|choch|time | 0.102 | -1.32 | -0.33 | -9.53 |
| 2024-07-01..2024-12-31 | 1.0 | 4h|don20|choch|time | 0.139 | +1.59 | +2.66 | -4.22 |
| 2025-01-01..2025-06-30 | 1.0 | 4h|frac3|bos|opp | 0.173 | -1.14 | -5.98 | -3.48 |
| 2025-07-01..2025-12-31 | 0.99 | 4h|don20|choch|time | 0.173 | -1.20 | -0.49 | -6.87 |
| 2026-01-01..2026-06-30 | 1.0 | 4h|don20|choch|time | 0.172 | -0.65 | -1.24 | -6.55 |
| 2026-07-01..2026-11-16 | 1.0 | 4h|frac2|bos|time | 0.156 | +0.05 | -1.26 | -7.11 |

llm_in_sample: Sharpe allocator -0.81, static +0.04, EW -6.47; allocator - static -0.0279/day (NW t -0.73), allocator - EW +0.4533 (t +12.79), 1277 days; fixed-phi Sharpes 0.9: -2.80, 0.95: -1.84, 0.98: -1.36, 0.99: -0.97, 1.0: -0.47

clean_holdout: Sharpe allocator +0.05, static -1.26, EW -7.11; allocator - static +0.0759/day (NW t +0.73), allocator - EW +0.4333 (t +4.07), 139 days; fixed-phi Sharpes 0.9: -3.49, 0.95: -4.34, 0.98: -3.67, 0.99: -0.69, 1.0: +0.05

DSR (in-sample WF series, 5 trials) 0.001; PBO over the 5 fixed-phi books 0.13.

## H-BOS5 detail: per-signal net R by arm (missed retest = 0), 95% week-bootstrap CI

| scope | segment | market | pegged | retest N6 | retest N6 best case | retest N24 | retest - market | retest - pegged | pegged - market | missed signals | missed winners | missed top-5% |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| intraday | llm_in_sample | -0.173 [-0.198, -0.143] | -0.123 [-0.148, -0.094] | -0.105 [-0.121, -0.086] | -0.105 [-0.121, -0.086] | -0.120 [-0.138, -0.098] | +0.068 [+0.057, +0.078] | +0.018 [+0.007, +0.027] | +0.050 [+0.048, +0.052] | 24% | 36% | 37% |
| intraday | clean_holdout | -0.191 [-0.300, -0.072] | -0.130 [-0.237, -0.013] | -0.112 [-0.182, -0.035] | -0.112 [-0.182, -0.034] | -0.127 [-0.207, -0.039] | +0.079 [+0.031, +0.122] | +0.019 [-0.025, +0.058] | +0.060 [+0.054, +0.067] | 25% | 37% | 38% |
| 5m | llm_in_sample | -0.194 [-0.220, -0.164] | -0.138 [-0.163, -0.109] | -0.118 [-0.134, -0.098] | -0.118 [-0.134, -0.098] | -0.134 [-0.153, -0.112] | +0.077 [+0.065, +0.087] | +0.021 [+0.010, +0.030] | +0.056 [+0.054, +0.058] | 25% | 36% | 38% |
| 5m | clean_holdout | -0.216 [-0.325, -0.098] | -0.149 [-0.253, -0.033] | -0.126 [-0.194, -0.055] | -0.126 [-0.194, -0.054] | -0.144 [-0.221, -0.060] | +0.090 [+0.040, +0.134] | +0.023 [-0.023, +0.063] | +0.067 [+0.060, +0.074] | 26% | 38% | 39% |
| 15m | llm_in_sample | -0.106 [-0.133, -0.076] | -0.075 [-0.101, -0.044] | -0.066 [-0.083, -0.047] | -0.066 [-0.083, -0.047] | -0.074 [-0.093, -0.053] | +0.041 [+0.029, +0.050] | +0.009 [-0.002, +0.019] | +0.032 [+0.030, +0.033] | 23% | 35% | 35% |
| 15m | clean_holdout | -0.112 [-0.225, +0.019] | -0.073 [-0.186, +0.058] | -0.066 [-0.149, +0.019] | -0.066 [-0.149, +0.020] | -0.074 [-0.162, +0.017] | +0.045 [+0.001, +0.083] | +0.006 [-0.037, +0.042] | +0.039 [+0.035, +0.043] | 24% | 36% | 35% |
| swing | llm_in_sample | -0.007 [-0.064, +0.057] | +0.006 [-0.051, +0.070] | -0.010 [-0.048, +0.031] | -0.009 [-0.048, +0.031] | -0.011 [-0.052, +0.033] | -0.002 [-0.026, +0.019] | -0.015 [-0.039, +0.007] | +0.013 [+0.013, +0.014] | 24% | 37% | 35% |
| swing | clean_holdout | -0.020 [-0.302, +0.389] | -0.004 [-0.283, +0.405] | -0.022 [-0.199, +0.229] | -0.024 [-0.201, +0.231] | -0.031 [-0.216, +0.238] | -0.002 [-0.160, +0.104] | -0.019 [-0.175, +0.087] | +0.017 [+0.015, +0.018] | 24% | 37% | 39% |
| 1h | llm_in_sample | -0.012 [-0.065, +0.049] | +0.002 [-0.051, +0.063] | -0.011 [-0.049, +0.028] | -0.011 [-0.049, +0.029] | -0.013 [-0.053, +0.029] | +0.001 [-0.022, +0.021] | -0.013 [-0.036, +0.007] | +0.014 [+0.014, +0.015] | 24% | 36% | 34% |
| 1h | clean_holdout | -0.016 [-0.292, +0.378] | +0.002 [-0.273, +0.395] | -0.014 [-0.195, +0.219] | -0.015 [-0.195, +0.221] | -0.025 [-0.214, +0.227] | +0.002 [-0.140, +0.100] | -0.016 [-0.158, +0.081] | +0.018 [+0.016, +0.020] | 24% | 36% | 38% |
| 4h | llm_in_sample | +0.013 [-0.066, +0.088] | +0.022 [-0.058, +0.097] | -0.003 [-0.052, +0.048] | -0.001 [-0.050, +0.049] | -0.002 [-0.053, +0.049] | -0.016 [-0.049, +0.016] | -0.024 [-0.057, +0.007] | +0.008 [+0.008, +0.009] | 24% | 39% | 39% |
| 4h | clean_holdout | -0.038 [-0.385, +0.496] | -0.027 [-0.374, +0.507] | -0.057 [-0.255, +0.246] | -0.058 [-0.262, +0.250] | -0.055 [-0.265, +0.273] | -0.019 [-0.252, +0.135] | -0.030 [-0.263, +0.125] | +0.011 [+0.010, +0.012] | 23% | 42% | 46% |

## Appendix: per-regime family pools and the top / bottom cells

Label rule (frozen): explosive = mean > 0 and top-5% share > 100%; power = mean > 0 and z > 2; detrimental = mean < 0 and z < -2. n counts variant-trades (variants overlap).

#### intraday, market entry: every variant of the class pooled, per regime

| regime | in-sample n | mean R | z | top-5% share | label | holdout n | mean R | z | top-5% share | label |
|---|---|---|---|---|---|---|---|---|---|---|
| ALL | 31175516 | -0.173 | -12.29 | n/a | detrimental | 2728632 | -0.191 | -3.22 | n/a | detrimental |
| R1L | 1268732 | -0.170 | -2.89 | n/a | detrimental | 70916 | -0.289 | -1.04 | n/a | neutral |
| R1S | 1257068 | -0.186 | -3.68 | n/a | detrimental | 78788 | -0.248 | -1.65 | n/a | neutral |
| R2L | 1105792 | -0.241 | -7.16 | n/a | detrimental | 75780 | +0.262 | +0.61 | 2.02 | explosive |
| R2S | 1098836 | -0.181 | -4.55 | n/a | detrimental | 73180 | -0.157 | -0.91 | n/a | neutral |
| R3L | 1209932 | -0.140 | -3.02 | n/a | detrimental | 174116 | -0.110 | -0.75 | n/a | neutral |
| R3S | 1194600 | -0.241 | -6.35 | n/a | detrimental | 160084 | -0.236 | -2.12 | n/a | detrimental |
| R4L | 3285158 | -0.169 | -3.61 | n/a | detrimental | 274012 | -0.173 | -0.85 | n/a | neutral |
| R4S | 3216767 | -0.152 | -3.64 | n/a | detrimental | 269884 | -0.320 | -3.48 | n/a | detrimental |
| R5L | 2391091 | -0.217 | -6.58 | n/a | detrimental | 200952 | -0.216 | -1.68 | n/a | neutral |
| R5S | 2385256 | -0.120 | -3.10 | n/a | detrimental | 201608 | -0.041 | -0.28 | n/a | neutral |
| R6L | 2152709 | -0.193 | -6.66 | n/a | detrimental | 232488 | -0.208 | -1.45 | n/a | neutral |
| R6S | 2122041 | -0.225 | -6.96 | n/a | detrimental | 236456 | -0.139 | -1.14 | n/a | neutral |
| R7L | 1575296 | -0.208 | -3.78 | n/a | detrimental | 132472 | -0.101 | -0.60 | n/a | neutral |
| R7S | 1530166 | -0.073 | -1.15 | n/a | neutral | 117804 | -0.480 | -4.63 | n/a | detrimental |
| R8L | 1284645 | -0.254 | -5.55 | n/a | detrimental | 118555 | -0.399 | -2.87 | n/a | detrimental |
| R8S | 1279691 | -0.054 | -0.77 | n/a | neutral | 128063 | -0.075 | -0.34 | n/a | neutral |
| R9L | 1260788 | -0.131 | -2.68 | n/a | detrimental | 92022 | -0.347 | -2.32 | n/a | detrimental |
| R9S | 1185376 | -0.211 | -4.16 | n/a | detrimental | 91452 | -0.073 | -0.33 | n/a | neutral |
| vol_T1 | 12270569 | -0.172 | -7.85 | n/a | detrimental | 1150712 | -0.260 | -3.35 | n/a | detrimental |
| vol_T2 | 9311110 | -0.175 | -9.41 | n/a | detrimental | 780216 | -0.105 | -1.41 | n/a | neutral |
| vol_T3 | 8311769 | -0.184 | -12.20 | n/a | detrimental | 797704 | -0.175 | -3.53 | n/a | detrimental |
| trend_S1 | 10041945 | -0.187 | -11.54 | n/a | detrimental | 931288 | -0.195 | -2.86 | n/a | detrimental |
| trend_S2 | 9935684 | -0.183 | -11.69 | n/a | detrimental | 933353 | -0.241 | -4.51 | n/a | detrimental |
| trend_S3 | 9915819 | -0.159 | -8.53 | n/a | detrimental | 863991 | -0.132 | -1.78 | n/a | neutral |

#### intraday, pegged entry: every variant of the class pooled, per regime

| regime | in-sample n | mean R | z | top-5% share | label | holdout n | mean R | z | top-5% share | label |
|---|---|---|---|---|---|---|---|---|---|---|
| ALL | 31175516 | -0.123 | -9.00 | n/a | detrimental | 2728632 | -0.130 | -2.26 | n/a | detrimental |
| R1L | 1268732 | -0.110 | -1.89 | n/a | neutral | 70916 | -0.201 | -0.73 | n/a | neutral |
| R1S | 1257068 | -0.128 | -2.54 | n/a | detrimental | 78788 | -0.166 | -1.11 | n/a | neutral |
| R2L | 1105792 | -0.194 | -5.81 | n/a | detrimental | 75780 | +0.318 | +0.74 | 1.67 | explosive |
| R2S | 1098836 | -0.136 | -3.42 | n/a | detrimental | 73180 | -0.102 | -0.60 | n/a | neutral |
| R3L | 1209932 | -0.107 | -2.33 | n/a | detrimental | 174116 | -0.076 | -0.53 | n/a | neutral |
| R3S | 1194600 | -0.209 | -5.52 | n/a | detrimental | 160084 | -0.202 | -1.82 | n/a | neutral |
| R4L | 3285158 | -0.102 | -2.21 | n/a | detrimental | 274012 | -0.090 | -0.44 | n/a | neutral |
| R4S | 3216767 | -0.085 | -2.06 | n/a | detrimental | 269884 | -0.235 | -2.60 | n/a | detrimental |
| R5L | 2391091 | -0.166 | -5.08 | n/a | detrimental | 200952 | -0.159 | -1.25 | n/a | neutral |
| R5S | 2385256 | -0.070 | -1.83 | n/a | neutral | 201608 | +0.015 | +0.10 | 22.54 | explosive |
| R6L | 2152709 | -0.155 | -5.38 | n/a | detrimental | 232488 | -0.160 | -1.13 | n/a | neutral |
| R6S | 2122041 | -0.187 | -5.83 | n/a | detrimental | 236456 | -0.091 | -0.73 | n/a | neutral |
| R7L | 1575296 | -0.150 | -2.74 | n/a | detrimental | 132472 | -0.025 | -0.15 | n/a | neutral |
| R7S | 1530166 | -0.016 | -0.25 | n/a | neutral | 117804 | -0.401 | -3.86 | n/a | detrimental |
| R8L | 1284645 | -0.208 | -4.58 | n/a | detrimental | 118555 | -0.347 | -2.49 | n/a | detrimental |
| R8S | 1279691 | -0.008 | -0.12 | n/a | neutral | 128063 | -0.024 | -0.11 | n/a | neutral |
| R9L | 1260788 | -0.098 | -2.01 | n/a | detrimental | 92022 | -0.294 | -1.96 | n/a | neutral |
| R9S | 1185376 | -0.177 | -3.54 | n/a | detrimental | 91452 | -0.019 | -0.09 | n/a | neutral |
| vol_T1 | 12270569 | -0.107 | -4.99 | n/a | detrimental | 1150712 | -0.180 | -2.34 | n/a | detrimental |
| vol_T2 | 9311110 | -0.128 | -6.95 | n/a | detrimental | 780216 | -0.051 | -0.70 | n/a | neutral |
| vol_T3 | 8311769 | -0.150 | -10.15 | n/a | detrimental | 797704 | -0.136 | -2.80 | n/a | detrimental |
| trend_S1 | 10041945 | -0.132 | -8.37 | n/a | detrimental | 931288 | -0.130 | -1.92 | n/a | neutral |
| trend_S2 | 9935684 | -0.130 | -8.46 | n/a | detrimental | 933353 | -0.177 | -3.40 | n/a | detrimental |
| trend_S3 | 9915819 | -0.114 | -6.26 | n/a | detrimental | 863991 | -0.080 | -1.12 | n/a | neutral |

#### intraday, retest entry: every variant of the class pooled, per regime

| regime | in-sample n | mean R | z | top-5% share | label | holdout n | mean R | z | top-5% share | label |
|---|---|---|---|---|---|---|---|---|---|---|
| ALL | 23612746 | -0.139 | -11.90 | n/a | detrimental | 2041730 | -0.149 | -3.01 | n/a | detrimental |
| R1L | 957520 | -0.148 | -3.04 | n/a | detrimental | 52892 | -0.229 | -0.99 | n/a | neutral |
| R1S | 943388 | -0.132 | -3.07 | n/a | detrimental | 56140 | -0.212 | -1.75 | n/a | neutral |
| R2L | 856804 | -0.207 | -6.83 | n/a | detrimental | 56360 | +0.240 | +0.65 | 2.01 | explosive |
| R2S | 834204 | -0.141 | -3.95 | n/a | detrimental | 54344 | -0.125 | -0.84 | n/a | neutral |
| R3L | 957576 | -0.111 | -2.67 | n/a | detrimental | 136960 | -0.039 | -0.26 | n/a | neutral |
| R3S | 939346 | -0.193 | -5.93 | n/a | detrimental | 125276 | -0.174 | -1.61 | n/a | neutral |
| R4L | 2398450 | -0.146 | -3.80 | n/a | detrimental | 197348 | -0.156 | -1.04 | n/a | neutral |
| R4S | 2360212 | -0.110 | -2.98 | n/a | detrimental | 193360 | -0.250 | -3.00 | n/a | detrimental |
| R5L | 1811166 | -0.187 | -6.74 | n/a | detrimental | 150060 | -0.174 | -1.54 | n/a | neutral |
| R5S | 1805416 | -0.093 | -2.76 | n/a | detrimental | 149532 | -0.060 | -0.48 | n/a | neutral |
| R6L | 1666475 | -0.158 | -6.10 | n/a | detrimental | 176312 | -0.164 | -1.26 | n/a | neutral |
| R6S | 1650350 | -0.185 | -6.31 | n/a | detrimental | 180792 | -0.112 | -1.00 | n/a | neutral |
| R7L | 1162597 | -0.163 | -3.37 | n/a | detrimental | 96256 | -0.083 | -0.54 | n/a | neutral |
| R7S | 1130658 | -0.052 | -0.97 | n/a | neutral | 87792 | -0.367 | -3.49 | n/a | detrimental |
| R8L | 974865 | -0.204 | -5.04 | n/a | detrimental | 89939 | -0.360 | -2.92 | n/a | detrimental |
| R8S | 985007 | -0.022 | -0.36 | n/a | neutral | 97590 | -0.028 | -0.13 | n/a | neutral |
| R9L | 965412 | -0.103 | -2.32 | n/a | detrimental | 69213 | -0.292 | -2.03 | n/a | detrimental |
| R9S | 926964 | -0.165 | -3.68 | n/a | detrimental | 71564 | -0.027 | -0.14 | n/a | neutral |
| vol_T1 | 9003271 | -0.137 | -7.79 | n/a | detrimental | 838948 | -0.210 | -3.46 | n/a | detrimental |
| vol_T2 | 7125304 | -0.139 | -8.84 | n/a | detrimental | 585048 | -0.083 | -1.30 | n/a | neutral |
| vol_T3 | 6508887 | -0.148 | -11.31 | n/a | detrimental | 617734 | -0.130 | -2.74 | n/a | detrimental |
| trend_S1 | 7530350 | -0.153 | -11.34 | n/a | detrimental | 691752 | -0.166 | -2.99 | n/a | detrimental |
| trend_S2 | 7524976 | -0.145 | -11.32 | n/a | detrimental | 694841 | -0.187 | -4.08 | n/a | detrimental |
| trend_S3 | 7582136 | -0.125 | -8.05 | n/a | detrimental | 655137 | -0.092 | -1.40 | n/a | neutral |

#### swing, market entry: every variant of the class pooled, per regime

| regime | in-sample n | mean R | z | top-5% share | label | holdout n | mean R | z | top-5% share | label |
|---|---|---|---|---|---|---|---|---|---|---|
| ALL | 2618630 | -0.007 | -0.24 | n/a | neutral | 203198 | -0.020 | -0.11 | n/a | neutral |
| R1L | 143192 | +0.141 | +0.98 | 2.68 | explosive | 6936 | +0.132 | +0.24 | 4.25 | explosive |
| R1S | 86740 | +0.143 | +1.49 | 2.32 | explosive | 5608 | -0.163 | -1.02 | n/a | neutral |
| R2L | 108932 | -0.035 | -0.35 | n/a | neutral | 7800 | +0.400 | +0.77 | 0.86 | neutral |
| R2S | 73594 | +0.004 | +0.06 | 68.37 | explosive | 4337 | -0.079 | -0.38 | n/a | neutral |
| R3L | 107211 | +0.069 | +0.52 | 4.97 | explosive | 15652 | +0.004 | +0.02 | 64.02 | explosive |
| R3S | 71054 | -0.180 | -3.41 | n/a | detrimental | 9374 | -0.194 | -1.07 | n/a | neutral |
| R4L | 293946 | +0.103 | +1.22 | 3.89 | explosive | 21580 | +0.648 | +0.77 | 1.18 | explosive |
| R4S | 289353 | +0.158 | +1.88 | 2.12 | explosive | 21964 | -0.310 | -3.00 | n/a | detrimental |
| R5L | 192244 | -0.131 | -2.29 | n/a | detrimental | 15796 | -0.193 | -0.86 | n/a | neutral |
| R5S | 194632 | +0.001 | +0.02 | 268.54 | explosive | 15720 | -0.059 | -0.32 | n/a | neutral |
| R6L | 159699 | -0.124 | -1.85 | n/a | neutral | 14764 | -0.112 | -0.37 | n/a | neutral |
| R6S | 153043 | -0.178 | -3.02 | n/a | detrimental | 16823 | -0.153 | -0.91 | n/a | neutral |
| R7L | 100212 | +0.034 | +0.29 | 11.10 | explosive | 9180 | +0.339 | +0.59 | 1.72 | explosive |
| R7S | 161850 | +0.052 | +0.57 | 5.54 | explosive | 10311 | -0.376 | -2.74 | n/a | detrimental |
| R8L | 77658 | -0.141 | -2.07 | n/a | detrimental | 6800 | +0.182 | +0.34 | 2.57 | explosive |
| R8S | 118959 | -0.099 | -1.17 | n/a | neutral | 10628 | -0.289 | -1.87 | n/a | neutral |
| R9L | 76465 | -0.134 | -2.25 | n/a | detrimental | 4284 | -0.156 | -0.39 | n/a | neutral |
| R9S | 88722 | -0.253 | -3.46 | n/a | detrimental | 5641 | -0.145 | -0.63 | n/a | neutral |
| vol_T1 | 983231 | +0.094 | +1.89 | 3.89 | explosive | 87874 | +0.041 | +0.13 | 11.04 | explosive |
| vol_T2 | 735926 | -0.035 | -0.89 | n/a | neutral | 58435 | -0.021 | -0.14 | n/a | neutral |
| vol_T3 | 624953 | -0.129 | -3.71 | n/a | detrimental | 56889 | -0.114 | -0.94 | n/a | neutral |
| trend_S1 | 699239 | -0.010 | -0.33 | n/a | neutral | 62439 | -0.005 | -0.02 | n/a | neutral |
| trend_S2 | 732021 | -0.019 | -0.60 | n/a | neutral | 65184 | -0.052 | -0.25 | n/a | neutral |
| trend_S3 | 912850 | +0.007 | +0.15 | 39.87 | explosive | 75575 | -0.006 | -0.03 | n/a | neutral |

#### swing, pegged entry: every variant of the class pooled, per regime

| regime | in-sample n | mean R | z | top-5% share | label | holdout n | mean R | z | top-5% share | label |
|---|---|---|---|---|---|---|---|---|---|---|
| ALL | 2618630 | +0.006 | +0.18 | 54.84 | explosive | 203198 | -0.004 | -0.02 | n/a | neutral |
| R1L | 143192 | +0.155 | +1.08 | 2.44 | explosive | 6936 | +0.154 | +0.27 | 3.66 | explosive |
| R1S | 86740 | +0.160 | +1.68 | 2.07 | explosive | 5608 | -0.138 | -0.86 | n/a | neutral |
| R2L | 108932 | -0.023 | -0.24 | n/a | neutral | 7800 | +0.413 | +0.80 | 0.84 | neutral |
| R2S | 73594 | +0.017 | +0.25 | 15.31 | explosive | 4337 | -0.063 | -0.31 | n/a | neutral |
| R3L | 107211 | +0.077 | +0.58 | 4.48 | explosive | 15652 | +0.012 | +0.06 | 20.69 | explosive |
| R3S | 71054 | -0.171 | -3.24 | n/a | detrimental | 9374 | -0.184 | -1.01 | n/a | neutral |
| R4L | 293946 | +0.120 | +1.43 | 3.34 | explosive | 21580 | +0.671 | +0.80 | 1.14 | explosive |
| R4S | 289353 | +0.175 | +2.08 | 1.92 | explosive + power | 21964 | -0.288 | -2.79 | n/a | detrimental |
| R5L | 192244 | -0.118 | -2.06 | n/a | detrimental | 15796 | -0.176 | -0.78 | n/a | neutral |
| R5S | 194632 | +0.014 | +0.22 | 19.10 | explosive | 15720 | -0.044 | -0.24 | n/a | neutral |
| R6L | 159699 | -0.114 | -1.70 | n/a | neutral | 14764 | -0.099 | -0.33 | n/a | neutral |
| R6S | 153043 | -0.168 | -2.86 | n/a | detrimental | 16823 | -0.140 | -0.82 | n/a | neutral |
| R7L | 100212 | +0.051 | +0.44 | 7.35 | explosive | 9180 | +0.361 | +0.63 | 1.62 | explosive |
| R7S | 161850 | +0.065 | +0.71 | 4.44 | explosive | 10311 | -0.357 | -2.60 | n/a | detrimental |
| R8L | 77658 | -0.127 | -1.87 | n/a | neutral | 6800 | +0.198 | +0.38 | 2.36 | explosive |
| R8S | 118959 | -0.089 | -1.05 | n/a | neutral | 10628 | -0.277 | -1.79 | n/a | neutral |
| R9L | 76465 | -0.124 | -2.08 | n/a | detrimental | 4284 | -0.138 | -0.35 | n/a | neutral |
| R9S | 88722 | -0.246 | -3.36 | n/a | detrimental | 5641 | -0.133 | -0.58 | n/a | neutral |
| vol_T1 | 983231 | +0.111 | +2.25 | 3.29 | explosive + power | 87874 | +0.064 | +0.20 | 7.14 | explosive |
| vol_T2 | 735926 | -0.023 | -0.58 | n/a | neutral | 58435 | -0.007 | -0.04 | n/a | neutral |
| vol_T3 | 624953 | -0.121 | -3.47 | n/a | detrimental | 56889 | -0.104 | -0.86 | n/a | neutral |
| trend_S1 | 699239 | +0.006 | +0.19 | 57.01 | explosive | 62439 | +0.014 | +0.07 | 25.84 | explosive |
| trend_S2 | 732021 | -0.005 | -0.16 | n/a | neutral | 65184 | -0.033 | -0.16 | n/a | neutral |
| trend_S3 | 912850 | +0.019 | +0.39 | 15.79 | explosive | 75575 | +0.007 | +0.04 | 42.49 | explosive |

#### swing, retest entry: every variant of the class pooled, per regime

| regime | in-sample n | mean R | z | top-5% share | label | holdout n | mean R | z | top-5% share | label |
|---|---|---|---|---|---|---|---|---|---|---|
| ALL | 1982984 | -0.013 | -0.47 | n/a | neutral | 154511 | -0.029 | -0.20 | n/a | neutral |
| R1L | 107632 | +0.062 | +0.52 | 5.43 | explosive | 5672 | +0.120 | +0.23 | 4.58 | explosive |
| R1S | 65028 | +0.126 | +1.46 | 2.64 | explosive | 3968 | -0.133 | -0.88 | n/a | neutral |
| R2L | 85900 | -0.018 | -0.19 | n/a | neutral | 5564 | +0.240 | +0.57 | 1.33 | explosive |
| R2S | 56594 | +0.025 | +0.43 | 10.31 | explosive | 3171 | -0.122 | -0.98 | n/a | neutral |
| R3L | 83373 | +0.080 | +0.63 | 4.23 | explosive | 11960 | +0.002 | +0.01 | 144.56 | explosive |
| R3S | 56728 | -0.134 | -2.72 | n/a | detrimental | 7348 | -0.152 | -1.07 | n/a | neutral |
| R4L | 215576 | +0.061 | +0.82 | 6.10 | explosive | 15408 | +0.384 | +0.58 | 1.89 | explosive |
| R4S | 206770 | +0.130 | +1.70 | 2.57 | explosive | 16112 | -0.260 | -3.23 | n/a | detrimental |
| R5L | 146503 | -0.131 | -2.54 | n/a | detrimental | 12336 | -0.135 | -0.64 | n/a | neutral |
| R5S | 144196 | -0.002 | -0.04 | n/a | neutral | 11593 | -0.026 | -0.14 | n/a | neutral |
| R6L | 124904 | -0.080 | -1.19 | n/a | neutral | 12176 | -0.069 | -0.24 | n/a | neutral |
| R6S | 122801 | -0.131 | -2.40 | n/a | detrimental | 12730 | -0.096 | -0.58 | n/a | neutral |
| R7L | 74656 | +0.010 | +0.10 | 34.73 | explosive | 6776 | +0.385 | +0.67 | 1.60 | explosive |
| R7S | 117563 | +0.041 | +0.49 | 7.03 | explosive | 8207 | -0.360 | -3.44 | n/a | detrimental |
| R8L | 60662 | -0.123 | -1.86 | n/a | neutral | 5508 | +0.154 | +0.33 | 2.85 | explosive |
| R8S | 90776 | -0.091 | -1.23 | n/a | neutral | 8374 | -0.276 | -1.95 | n/a | neutral |
| R9L | 59437 | -0.106 | -2.03 | n/a | detrimental | 3384 | -0.073 | -0.19 | n/a | neutral |
| R9S | 70337 | -0.211 | -2.97 | n/a | detrimental | 4224 | -0.044 | -0.19 | n/a | neutral |
| vol_T1 | 733013 | +0.062 | +1.46 | 5.63 | explosive | 66499 | +0.002 | +0.01 | 208.46 | explosive |
| vol_T2 | 561454 | -0.039 | -1.12 | n/a | neutral | 44256 | -0.020 | -0.16 | n/a | neutral |
| vol_T3 | 479013 | -0.095 | -2.94 | n/a | detrimental | 43756 | -0.086 | -0.81 | n/a | neutral |
| trend_S1 | 532349 | -0.008 | -0.31 | n/a | neutral | 47419 | -0.029 | -0.19 | n/a | neutral |
| trend_S2 | 557612 | -0.028 | -0.99 | n/a | neutral | 49750 | -0.027 | -0.14 | n/a | neutral |
| trend_S3 | 683519 | -0.004 | -0.09 | n/a | neutral | 57342 | -0.032 | -0.25 | n/a | neutral |

#### Top 20 tested in-sample cells by z (none survives BH unless marked)

| variant | arm | regime | n | weeks | mean R | z | p | hit | payoff | top-5% share | holdout n | holdout mean | holdout z |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1h|don20|choch|time | pegged | vol_T1 | 8103 | 255 | +0.234 | +3.25 | 5.8e-04 | 0.27 | 3.56 | 2.00 | 740 | +0.230 | +0.60 |
| 1h|don55|choch|time | pegged | R1S | 319 | 123 | +0.584 | +3.09 | 9.9e-04 | 0.52 | 2.09 | 0.49 | 22 | -0.464 | n/a |
| 1h|don55|choch|time | market | R1S | 319 | 123 | +0.575 | +3.05 | 1.2e-03 | 0.52 | 2.06 | 0.50 | 22 | -0.479 | n/a |
| 1h|don20|choch|time | market | vol_T1 | 8103 | 255 | +0.219 | +3.04 | 1.2e-03 | 0.27 | 3.49 | 2.14 | 740 | +0.210 | +0.55 |
| 1h|don55|bos|time | pegged | R1S | 449 | 126 | +0.537 | +2.97 | 1.5e-03 | 0.51 | 2.08 | 0.54 | 36 | -0.458 | n/a |
| 1h|don55|choch|tr40 | pegged | R1S | 319 | 123 | +0.350 | +2.95 | 1.6e-03 | 0.51 | 1.84 | 0.51 | 22 | -0.187 | n/a |
| 1h|don55|bos|time | market | R1S | 449 | 126 | +0.528 | +2.92 | 1.7e-03 | 0.51 | 2.05 | 0.55 | 36 | -0.474 | n/a |
| 1h|don20|choch|time | retest | vol_T1 | 5955 | 248 | +0.193 | +2.91 | 1.8e-03 | 0.24 | 4.11 | 2.36 | 559 | +0.279 | +0.69 |
| 1h|don55|choch|tr40 | market | R1S | 319 | 123 | +0.341 | +2.87 | 2.0e-03 | 0.51 | 1.80 | 0.52 | 22 | -0.202 | n/a |
| 1h|don55|bos|tr40 | pegged | R1S | 449 | 126 | +0.314 | +2.86 | 2.1e-03 | 0.50 | 1.86 | 0.57 | 36 | -0.220 | n/a |
| 1h|frac5|bos|time | pegged | vol_T1 | 17453 | 262 | +0.210 | +2.83 | 2.3e-03 | 0.25 | 3.84 | 2.40 | 1589 | +0.140 | +0.38 |
| 1h|frac3|choch|time | pegged | vol_T1 | 12944 | 258 | +0.232 | +2.82 | 2.4e-03 | 0.20 | 5.05 | 2.74 | 1182 | +0.141 | +0.36 |
| 4h|frac2|choch|time | pegged | vol_T1 | 3755 | 244 | +0.209 | +2.81 | 2.5e-03 | 0.29 | 3.13 | 2.00 | 366 | +0.291 | +0.52 |
| 1h|frac2|choch|time | pegged | vol_T1 | 16804 | 258 | +0.218 | +2.79 | 2.7e-03 | 0.18 | 5.70 | 3.09 | 1539 | +0.130 | +0.32 |
| 1h|don55|bos|tr40 | market | R1S | 449 | 126 | +0.306 | +2.78 | 2.8e-03 | 0.50 | 1.82 | 0.58 | 36 | -0.236 | n/a |
| 4h|frac2|bos|time | pegged | vol_T1 | 7594 | 251 | +0.223 | +2.73 | 3.2e-03 | 0.29 | 3.17 | 1.89 | 701 | +0.062 | +0.16 |
| 4h|don20|choch|tr25 | pegged | R7S | 246 | 103 | +0.394 | +2.70 | 3.5e-03 | 0.49 | 2.44 | 0.62 | 16 | -0.402 | n/a |
| 4h|don20|choch|tr40 | pegged | R7S | 246 | 103 | +0.439 | +2.70 | 3.5e-03 | 0.46 | 2.53 | 0.62 | 16 | -0.364 | n/a |
| 1h|don20|choch|tr40 | pegged | vol_T1 | 8111 | 255 | +0.122 | +2.67 | 3.8e-03 | 0.34 | 2.39 | 2.52 | 741 | +0.107 | +0.41 |
| 1h|don20|bos|time | pegged | vol_T1 | 22855 | 263 | +0.206 | +2.65 | 4.1e-03 | 0.28 | 3.30 | 2.13 | 2013 | +0.114 | +0.30 |

#### Bottom 20 tested in-sample cells by z

| variant | arm | regime | n | weeks | mean R | z | p | hit | payoff | top-5% share | holdout n | holdout mean | holdout z |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 5m|frac2|choch|opp | market | trend_S2 | 155513 | 236 | -0.323 | -35.07 | 0.0e+00 | 0.26 | 1.54 | n/a | 14387 | -0.382 | -15.27 |
| 5m|frac2|choch|opp | retest | ALL | 363257 | 236 | -0.240 | -34.42 | 0.0e+00 | 0.25 | 1.78 | n/a | 31332 | -0.284 | -12.95 |
| 5m|frac2|choch|opp | market | ALL | 485274 | 236 | -0.301 | -34.31 | 0.0e+00 | 0.26 | 1.56 | n/a | 42266 | -0.365 | -14.04 |
| 5m|frac2|choch|opp | market | vol_T2 | 145450 | 236 | -0.295 | -33.60 | 0.0e+00 | 0.26 | 1.55 | n/a | 12001 | -0.313 | -9.26 |
| 5m|frac2|bos|opp | retest | ALL | 735511 | 236 | -0.241 | -33.52 | 0.0e+00 | 0.24 | 1.82 | n/a | 64189 | -0.281 | -12.12 |
| 5m|frac2|choch|opp | retest | vol_T2 | 109962 | 236 | -0.237 | -33.20 | 0.0e+00 | 0.25 | 1.76 | n/a | 8932 | -0.249 | -7.96 |
| 5m|frac2|bos|opp | retest | trend_S2 | 235311 | 236 | -0.258 | -33.08 | 0.0e+00 | 0.24 | 1.79 | n/a | 21783 | -0.297 | -11.79 |
| 5m|frac2|choch|opp | retest | trend_S2 | 116677 | 236 | -0.253 | -33.02 | 0.0e+00 | 0.24 | 1.75 | n/a | 10689 | -0.294 | -12.52 |
| 5m|frac2|bos|opp | market | trend_S2 | 312355 | 236 | -0.324 | -32.72 | 0.0e+00 | 0.25 | 1.57 | n/a | 29191 | -0.388 | -13.35 |
| 5m|frac2|bos|opp | market | ALL | 977977 | 236 | -0.301 | -32.02 | 0.0e+00 | 0.26 | 1.59 | n/a | 85966 | -0.362 | -12.44 |
| 5m|frac2|choch|opp | market | trend_S1 | 155457 | 236 | -0.329 | -31.73 | 0.0e+00 | 0.25 | 1.55 | n/a | 14463 | -0.406 | -14.27 |
| 5m|frac2|choch|opp | retest | trend_S1 | 115587 | 236 | -0.266 | -31.36 | 0.0e+00 | 0.24 | 1.76 | n/a | 10582 | -0.329 | -13.49 |
| 5m|frac2|choch|opp | retest | vol_T1 | 137857 | 229 | -0.297 | -30.90 | 0.0e+00 | 0.23 | 1.79 | n/a | 12801 | -0.362 | -14.90 |
| 5m|frac2|bos|opp | retest | vol_T2 | 221826 | 236 | -0.237 | -30.84 | 0.0e+00 | 0.24 | 1.81 | n/a | 18306 | -0.237 | -7.82 |
| 5m|frac2|bos|opp | market | vol_T2 | 291980 | 236 | -0.295 | -30.23 | 0.0e+00 | 0.26 | 1.60 | n/a | 24413 | -0.296 | -7.47 |
| 5m|frac2|bos|opp | retest | trend_S1 | 233241 | 236 | -0.266 | -30.18 | 0.0e+00 | 0.23 | 1.80 | n/a | 21452 | -0.313 | -10.86 |
| 5m|frac2|choch|opp | market | vol_T1 | 190290 | 229 | -0.370 | -29.74 | 0.0e+00 | 0.24 | 1.56 | n/a | 17719 | -0.461 | -15.56 |
| 5m|frac2|choch|opp | pegged | trend_S2 | 155513 | 236 | -0.239 | -29.65 | 0.0e+00 | 0.27 | 1.64 | n/a | 14387 | -0.281 | -12.99 |
| 5m|frac3|choch|opp | market | trend_S2 | 117211 | 236 | -0.277 | -29.50 | 0.0e+00 | 0.26 | 1.62 | n/a | 10882 | -0.339 | -15.05 |
| 5m|frac2|choch|opp | pegged | ALL | 485274 | 236 | -0.222 | -29.17 | 0.0e+00 | 0.28 | 1.66 | n/a | 42266 | -0.270 | -11.90 |

## Rule candidates

Validated with `ai_trader.research.rules.validate` (no errors). Not merged into research/rules.yaml by this agent.

```yaml
- id: R-BOS-1
  kind: avoid
  rule: WHEN any regime and the timeframe is intraday (5m or 15m bars, hold up to 1 day) THEN avoid break-of-structure breakout variants (fractal k 2-5 or Donchian 20/55; BoS, CHoCH or BoS+momentum; any exit) with any execution, market, pegged or retest
  when: a BoS / CHoCH / Donchian-breakout candidate built on 5m or 15m bars of a universe L crypto perp, in any v0 cell, vol tercile or trend tercile
  then: skip it; after fees and slippage every intraday regime cell lost money in both segments, and no execution arm turned it positive
  scope:
    assets: crypto
    horizon: 5m-15m bars, hold up to 1 day
    sides: both
  effect: 'family pool, market entry: in-sample -0.173 R per trade (z -12.3, 31.2M variant-trades), holdout -0.191 (z -3.2); pegged -0.123 (z -9.0) / -0.130 (z -2.3); 646 of 9,000 tested intraday in-sample cells had a positive mean, none with z > 1.64; 6,088 had z < -2; across all timeframes 8,210 cells were negative at BH 10% in-sample and 84% of them stayed negative in the holdout'
  confidence: B
  evidence:
  - H-BOS1
  - H-BOS2
  - research/reports/bos_regimes_2026-11-15.md
  review: a fee tier where the round trip falls below about 5 bp, or a registered intraday BoS variant with positive net R in a 3-month forward window
  status: active
- id: R-BOS-2
  kind: avoid
  rule: WHEN the 4h volatility is in its top tercile of the trailing year THEN avoid swing break-of-structure breakouts (1h/4h fractal or Donchian BoS, any exit) with any execution
  when: vol tercile T3 (4h sigma_next percentile >= 2/3 within the trailing 365 days, known at the signal), 1h or 4h BoS signal, universe L
  then: skip the breakout; in high-volatility regimes BoS entries are stopped out without a compensating right tail
  scope:
    assets: crypto
    horizon: 1h-4h bars, hold up to 10 days
    sides: both
  effect: 'swing family pool, pegged entry: in-sample -0.121 R per trade (z -3.47, 625k variant-trades), holdout -0.104 (z -0.86, 57k); market -0.121 / -0.104; same sign in both segments, holdout not significant on its own'
  confidence: C
  evidence:
  - H-BOS2
  - research/reports/bos_regimes_2026-11-15.md
  review: holdout reaches z < -2 for the T3 pool (keep, upgrade to B) or turns positive over 6 months (retire)
  status: provisional
- id: R-BOS-3
  kind: avoid
  rule: 'WHEN the v0 cell is R9 (liquidation) or R6 (whipsaw), or a short in R3 (momentum burst) THEN avoid swing break-of-structure breakouts with any execution: they lost in both segments'
  when: v0 cell of the last closed 4h bar in {R9, R6} for either side, or R3 with a short (break of a swing low inside an up-burst), 1h/4h BoS signals, universe L
  then: skip the breakout; read the break as noise inside a violent range or against a burst
  scope:
    assets: crypto
    horizon: 1h-4h bars, hold up to 10 days
    sides: both
  effect: 'swing pool, pegged, in-sample / holdout mean net R per trade: R9S -0.246 (z -3.36) / -0.133; R9L -0.124 (z -2.08) / -0.138; R6S -0.168 (z -2.86) / -0.140; R6L -0.114 (z -1.70) / -0.099; R3S -0.171 (z -3.24) / -0.184; holdout z between -1.0 and -0.3 (small samples)'
  confidence: C
  evidence:
  - H-BOS2
  - research/reports/bos_regimes_2026-11-15.md
  review: any of these pools positive over the next 6 months of holdout
  status: provisional
- id: R-BOS-4
  kind: expect
  rule: WHEN a swing BoS breakout is taken in a calm regime (4h volatility in the bottom tercile, or v0 cells R1/R4/R7 on the long side) THEN expect a right-tail-driven payoff, positive on average only because of the top 5% of trades, and no reliable edge; prefer the 1h Donchian-20 CHoCH with a time exit and pegged entry only as a forward-test candidate
  when: vol tercile T1 or v0 cell R1/R4/R7 with a long, 1h/4h BoS, universe L
  then: treat it as a lottery-like trend bet whose average is carried by about 5% of trades; it did not survive BH and its DSR is about 0 against the 35,625 cells searched
  scope:
    assets: crypto
    horizon: 1h-4h bars, hold up to 10 days
    sides: both
  effect: 'swing pool pegged: T1 in-sample +0.111 (z 2.25, top-5% share 3.3x the total), holdout +0.064 (z 0.20, 7.1x); R1L +0.155 / +0.154, R4L +0.120 / +0.671, R7L +0.051 / +0.361, all with top-5% share > 100% in both segments and holdout z < 1; best cell 1h don20 CHoCH time pegged T1 +0.234 R (z 3.25, n 8103) in-sample, +0.230 (z 0.60, n 740) holdout, TradFi holdout +0.14 (n 96); DSR 0.00, not a BH survivor'
  confidence: C
  evidence:
  - H-BOS1
  - H-BOS2
  - research/reports/bos_regimes_2026-11-15.md
  review: a pre-registered forward test of the 1h don20 CHoCH time-exit cell in T1 with z > 2 after 6 months
  status: provisional
- id: R-BOS-5
  kind: prefer
  rule: WHEN entering any break-of-structure signal THEN prefer a pegged limit at the touch over a market order; it was better in every timeframe, regime pool and segment
  when: a BoS / CHoCH / Donchian breakout signal on a universe L crypto perp, VIP0 fees (maker 2.0 bp, taker 5.5 bp)
  then: enter with the pegged limit (R-EXE-1 mechanics); expect about +0.01 R per signal on swing and +0.03 to +0.07 R intraday, from the fee gap only
  scope:
    assets: crypto
    horizon: 5m-4h bars
    sides: both
  effect: 'pegged minus market per signal (week-bootstrap 95% CI): swing in-sample +0.013 [+0.0125, +0.0135], holdout +0.017 [+0.015, +0.018]; intraday +0.050 [+0.048, +0.052] / +0.060 [+0.054, +0.067]'
  confidence: B
  evidence:
  - H-BOS5
  - H-EXE1
  - research/reports/bos_regimes_2026-11-15.md
  review: a fee tier where the maker-taker gap narrows below about 1.5 bp, or live fills of pegged BoS entries worse than 1.3 bp beyond the touch
  status: active
- id: R-BOS-6
  kind: expect
  rule: WHEN a BoS breakout is entered with a limit at the broken level (retest) THEN expect to miss about a quarter of the signals and more than a third of the biggest winners; on swing it is no better than market or pegged entries
  when: retest limit at the broken swing / channel level, live 6 bars, trade-through of 1 bp, universe L, 5m-4h
  then: do not expect the retest to rescue a breakout; on swing prefer the pegged entry, and read the intraday gain over market as fee savings on a losing base
  scope:
    assets: crypto
    horizon: 5m-4h bars
    sides: both
  effect: 'missed signals 23-26%; missed share of the market arm top-5% trades 35-39% (swing 4h holdout 46%), of its winners 35-42%; retest minus market swing in-sample -0.002 R [-0.026, +0.020], holdout -0.002 [-0.160, +0.104]; retest minus pegged swing -0.015 [-0.039, +0.007]; intraday retest minus market +0.068 [+0.057, +0.078] in-sample, +0.079 [+0.032, +0.122] holdout, all arms negative intraday'
  confidence: B
  evidence:
  - H-BOS5
  - research/reports/bos_regimes_2026-11-15.md
  review: a retest rule with a different placement (a fraction of the breakout bar) registered and tested the same way
  status: active
- id: R-BOS-7
  kind: avoid
  rule: WHEN choosing which BoS variant to run THEN avoid switching variants on and off by a Bayesian posterior of their recent regime-conditional edge; it did not beat the static in-sample pick and its own Sharpe was negative
  when: walk-forward on/off allocator over 240 BoS variants x 18 regime cells, conjugate normal posterior, forgetting factor chosen in training windows
  then: don't expect adaptive regime switching to harvest the instability; it mainly switches off the losing intraday variants, which a static rule (R-BOS-1) does more simply
  scope:
    assets: crypto
    horizon: weekly decisions, 5m-4h bars
    sides: both
  effect: 'pegged entries, walk-forward 2023-2026H1: allocator Sharpe -0.38 vs static best +0.25 and equal weight -4.60; paired diff vs static -0.032 (NW t -0.69); holdout -0.90 vs -1.11 / -4.90, vs static +0.040 (t 0.34), vs EW +0.273 (t 2.82); DSR 0.004 over the 5-point forgetting grid; PBO 0.19'
  confidence: C
  evidence:
  - H-BOS4
  - research/reports/bos_regimes_2026-11-15.md
  review: a registered allocator with a different unit (regime-only, not variant x regime) that beats the static pick with DSR > 0.95 in-sample
  status: provisional
```
