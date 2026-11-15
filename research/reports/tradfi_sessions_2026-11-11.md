# TradFi session effects on Bybit TradFi perps (block TF: H-TP1, H-TF1..H-TF5)

_Program 2, agent B2, run 2026-11-10 on data to 2026-11-10 22:00 UTC. Module `src/ai_trader/research/tradfi_session_study.py`, tests `tests/test_tradfi_session.py`, numbers in [tradfi_sessions_2026-11-11.json](tradfi_sessions_2026-11-11.json). The design was fixed before any result (diary 2026-11-10, "B2 TradFi session block: design fixed before results")._

## Summary

- **Off-hours:** Bybit equity perps trade 24/7. Overnight they run at about 0.25-0.5x regular-hours turnover and 0.4-0.65x regular-hours |r| per hour. At weekends they run at about 0.1-0.15x turnover and 0.25-0.33x |r|. Weekends are quiet but **not frozen**: the index price moves on weekends too, which corrects BELIEFS P5's "freeze". So the overnight "gap" is a traded drift on the perp, not a jump.
- **None of the six hypotheses is supported.** In the clean holdout (93 NYSE sessions, 9 equity perps; 19 weekends for the commodities), five are **contradicted** (estimate has the wrong sign) and H-TF2 is formally inconclusive at about zero (+3 bp, t 0.14). The llm_in_sample slice is only 16 sessions (11 weekends) and is inconclusive everywhere.
- **No automatable effect survives 7.5 bp per side.** The one robust negative finding is that fading overnight moves larger than 1σ at the open loses: -67 bp net per trade in the holdout, t -2.5, and it does worse than 99.8% of matched random entries. Those moves continued instead. That is post-hoc, it has the opposite sign in-sample, and it is proposed below only as a forward-only hypothesis.

## 1. How the perps behave when the underlying is closed (measured first)

The table gives median turnover and mean |1h log return| relative to regular hours (10:00-15:00 ET bars) = 1. All hours, Apr-Nov 2026, last price; index-price columns where stated. Buckets are in America/New_York time; "weekend" runs Fri 20:00 ET to Sun 23:59 ET.

| symbol | pre 04-08 ET | first cash bar 09 ET | regular 10-15 ET | post 16-19 ET | overnight 20-03 ET | weekend | weekend zero-return hours (last / index) | median abs(last/index basis), bp |
|---|---|---|---|---|---|---|---|---|
| TSLA | 0.55 / 0.54 | 4.60 / 2.46 | 1 / 1 | 0.23 / 0.38 | 0.26 / 0.42 | 0.12 / 0.23 | 0.019 / 0.062 | 4.6 |
| INTC | 0.80 / 0.73 | 4.71 / 2.32 | 1 / 1 | 0.28 / 0.60 | 0.47 / 0.63 | 0.15 / 0.30 | 0.027 / 0.110 | 6.4 |
| GOOGL | 0.69 / 0.62 | 4.94 / 2.20 | 1 / 1 | 0.30 / 0.55 | 0.29 / 0.45 | 0.11 / 0.27 | 0.019 / 0.053 | 7.2 |
| AAPL | 0.60 / 0.54 | 5.15 / 2.25 | 1 / 1 | 0.25 / 0.44 | 0.26 / 0.38 | 0.14 / 0.28 | 0.029 / 0.068 | 4.9 |
| MSTR | 0.53 / 0.63 | 3.31 / 1.85 | 1 / 1 | 0.31 / 0.43 | 0.30 / 0.52 | 0.15 / 0.32 | 0.013 / 0.072 | 5.6 |
| NVDA | 0.69 / 0.61 | 3.17 / 1.85 | 1 / 1 | 0.33 / 0.43 | 0.37 / 0.48 | 0.14 / 0.26 | 0.032 / 0.054 | 5.3 |
| MU | 0.66 / 0.74 | 5.17 / 2.24 | 1 / 1 | 0.32 / 0.56 | 0.44 / 0.66 | 0.12 / 0.29 | 0.004 / 0.014 | 5.4 |
| QQQ | 0.61 / 0.65 | 3.24 / 1.68 | 1 / 1 | 0.28 / 0.51 | 0.40 / 0.62 | 0.13 / 0.33 | 0.017 / 0.020 | 5.0 |
| SOXL | 0.89 / 0.70 | 4.17 / 1.93 | 1 / 1 | 0.23 / 0.49 | 0.47 / 0.64 | 0.09 / 0.27 | 0.010 / 0.020 | 5.6 |
| CL | 0.93 / 1.04 | 1.50 / 1.30 | 1 / 1 | 0.42 / 0.50 | 0.48 / 0.66 | 0.19 / 0.43 | 0.026 / 0.004 | 7.6 |
| BZ | 1.16 / 1.02 | 1.42 / 1.26 | 1 / 1 | 0.48 / 0.46 | 0.54 / 0.69 | 0.27 / 0.47 | 0.027 / 0.048 | 7.0 |
| XAU | 1.10 / 0.97 | 2.45 / 1.58 | 1 / 1 | 0.39 / 0.55 | 0.91 / 0.99 | 0.17 / 0.29 | 0.005 / 0.004 | 4.7 |
| XAG | 0.94 / 0.97 | 2.34 / 1.74 | 1 / 1 | 0.39 / 0.48 | 0.82 / 1.07 | 0.10 / 0.29 | 0.094 / 0.164 | 5.6 |

Each cell is turnover / |r|. Reading the table:
- **A bar exists for every hour, weekends included.** Pooled hour-of-week profiles show spikes at the underlying's session boundaries: the 04:00 ET pre-market open, the 09:00 ET bar that contains the cash open (about 4x turnover), and Sunday 18:00-20:00 ET (the futures and overnight-equity reopen).
- **Weekends are quiet, not frozen.** Weekend hours carry about a quarter to a third of the regular-hours |r| per hour. Only 1-3% of hours have a zero return on last price and 2-11% on the index. The index keeps moving, and last price stays within about 5 bp of it at all hours.
- **The overnight move is real price discovery on the perp.** About half of the regular-hours |r| per hour, spread over 17 hours on weekdays and 65 hours over weekends, adds up to a large traded move. There is no jump at the cash open, only a volatility spike: the 09:00 ET bar carries 1.7-2.5x the regular-hours |r|.
- **What "gap" means here:** the drift between the NYSE close and the open price. It is a traded move, so a "fill" means price trading back to the previous close.

## 2. Methods (as registered; implementation choices fixed in advance)

- **Universe.**
  - Equity subset: TSLA INTC GOOGL AAPL MSTR NVDA MU QQQ SOXL.
  - Commodities: CL BZ XAU XAG.
  - Excluded: SPCX (under 120 days) and the equity perps listed 2026-11-05 (SPY, MSFT, META, AVGO, TSM, ORCL, HOOD and others), which have too little history.
  - Bars come from `research.frame.market_frame`, which applies price sanity checks and a 30-day listing burn-in.
- **Calendar.**
  - NYSE regular session is 09:30-16:00 America/New_York: 13:30-20:00 UTC under EDT and 14:30-21:00 UTC under EST. DST 2026 runs Mar 8 to Nov 1.
  - The 2026 holidays and the 13:00 ET early closes (Nov 27, Dec 24) are hard-coded from the NYSE 2026-2028 holiday calendar release. Any other year raises an error.
  - The CME Sunday reopen is 18:00 ET (22:00 UTC under EDT, 23:00 UTC under EST).
- **Prices and the half-hour offset.**
  - P(T) is the close of the 1h bar ending at T. The NYSE close (16:00 ET) falls on a bar boundary.
  - The **first cash hour** is the 1h bar containing 09:30 ET, i.e. the 09:00-10:00 ET bar. The "open" price is P(09:00 ET).
  - So the 30 pre-open minutes (09:00-09:30) count as cash session:
    - overnight = previous close (16:00 ET, or 13:00 ET after an early close) to 09:00 ET;
    - intraday = 09:00 to 16:00 ET, which is 7 bars (4 on early closes).
  - Robustness check: open at 10:00 ET.
- **Point-in-time sigmas** (both known at the open):
  - sigma_on: RMS of the previous 20 overnight returns (at least 10), weekends and holidays included.
  - sigma_bar: square root of the mean per-bar regular-session realised variance over the previous 20 sessions.
  - Strategy P&L is in **risk units (RU)**: net log return / sigma_bar, with the position fixed at 1/sigma at entry.
- **Tests:**
  - H-TP1: signed reversal = -sign(overnight) × ret(09:00→11:00 ET) / (sigma_bar·√2). Averaged across symbols per date, Newey-West t over dates.
  - H-TF1: events are sessions with |overnight| > 1 sigma_on.
    - Filled = some cash-session bar's [low, high] contains the previous close.
    - Baseline = `calendar.gap_fill_baseline(|gap|/sigma_bar, n_bars)`, the H-W4 reflection principle.
    - Test statistic: date-clustered z of (filled − baseline).
  - H-TF2: overnight − intraday log return per symbol-day, date-averaged, NW t.
  - H-TF3: entry on the first 1h close of the 10:00-14:00 ET bars beyond the 09:00 ET bar's high or low, in that direction, held to the NYSE close. 15 bp round-trip cost.
    - Random-entry baseline: 1000 draws, each replaying every trade with the same symbol, side, entry hour and holding, on a random session of the same segment and length.
  - H-TF4: last bar (15:00 ET) on (previous close → 10:00 ET), each scaled by its trailing 20-session RMS. Pooled OLS with CR1 standard errors clustered by date.
  - H-TF5: sign(Sun 18:00 ET → Mon 07:00 UTC) × (Mon 07:00 → 12:00 UTC) / (sigma·√5), where sigma is the RMS of the prior Mon-Fri 1h returns. Clustered by weekend.
    - Variant: the literal fixed "Sunday 23:00 UTC" start, which equals the DST-aware start under EST.
- **Segments** are split by session open (TF5: Monday 07:00 UTC) at 2026-07-01.
  - A verdict needs at least 30 dates (10 weekends for TF5).
  - "supports" = the registered criterion is met and the Holm-adjusted p < 0.05, Holm taken over the six TF-block one-sided p-values per segment.
  - "contradicts" = the point estimate has the wrong sign.
  - Anything else is "inconclusive".
- **Coverage:**

  | segment | sessions | dates | weekends (TF5) |
  |---|---|---|---|
  | llm_in_sample (Jun 8-Jun 30) | 83 | 16 | 11 |
  | clean_holdout (Jul 1-Nov 10) | 834 | 93 | 19 |
  | pooled | 917 | 109 | 30 |

## 3. Verdicts

| hypothesis | segment | statistic | t / z | dates | p raw | p Holm | verdict |
|---|---|---|---|---|---|---|---|
| H-TP1 open reversal (σ) | llm_in_sample | +0.274 | 1.88 | 16 | 0.040 | 0.241 | inconclusive (insufficient) |
| | clean_holdout | −0.088 | −0.89 | 93 | 0.81 | 1 | **contradicts** |
| | pooled | −0.035 | −0.39 | 109 | 0.65 | 1 | contradicts |
| H-TF1 fill − baseline | llm_in_sample | +1.5 pp | 0.17 | 13 | 0.43 | 1 | inconclusive (insufficient) |
| | clean_holdout | −5.4 pp (22.4% vs 27.8%) | −1.76 | 81 | 0.96 | 1 | **contradicts** |
| | pooled | −4.6 pp | −1.60 | 94 | 0.94 | 1 | contradicts |
| H-TF2 night − day (log) | llm_in_sample | +96 bp | 1.46 | 16 | 0.083 | 0.414 | inconclusive (insufficient) |
| | clean_holdout | +3 bp | 0.14 | 93 | 0.44 | 1 | inconclusive (≈ 0) |
| | pooled | +17 bp | 0.77 | 109 | 0.22 | 1 | inconclusive |
| H-TF3 ORB net (RU) | llm_in_sample | −0.09 | −0.35 | 16 | 0.63 | 1 | inconclusive (insufficient) |
| | clean_holdout | −0.18 (−13.9 bp/trade) | −2.44 | 93 | 0.99 | 1 | **contradicts** |
| | pooled | −0.17 | −2.37 | 109 | 0.99 | 1 | contradicts |
| H-TF4 slope | llm_in_sample | +0.07 | 0.51 | 16 | 0.31 | 1 | inconclusive (insufficient) |
| | clean_holdout | −0.09 | −0.92 | 93 | 0.82 | 1 | **contradicts** |
| | pooled | −0.08 | −0.85 | 109 | 0.80 | 1 | contradicts |
| H-TF5 continuation (σ) | llm_in_sample | +0.09 | 0.49 | 11 wk | 0.32 | 1 | inconclusive |
| | clean_holdout | −0.13 | −0.92 | 19 wk | 0.82 | 1 | **contradicts** |
| | pooled | −0.05 | −0.46 | 30 wk | 0.68 | 1 | contradicts |

Details and robustness (holdout unless stated):
- **H-TP1:**
  - About half the moves revert (49.3%).
  - Overnight moves larger than 1σ continue rather than revert: reversal −0.22σ (t −1.5, 246 events).
  - Post-weekend sessions: −0.24σ (t −1.2). Weekday overnights: −0.05σ.
  - In-sample, weekday overnights reverted: +0.51σ, t 3.0, 12 dates. The sign flipped after the cutoff, which is what memorised lore looks like.
- **H-TF1:**
  - Average event gap: 1.8 sigma_on.
  - With an ex-post baseline that uses the day's own realised session variance, the excess is −1.0 pp (z −0.3).
  - So fills happen at about random-walk rates. The trailing-variance baseline is slightly too high because realised variance is right-skewed (mean > typical).
- **H-TF2:**
  - Overnight mean −14 bp, intraday −18 bp. Both are negative over a falling holdout.
  - Vol-scaled difference −0.01 (t −0.1). With the open at 10:00 ET: +16 bp (t 0.7).
  - Only AAPL has a significant difference, and with the opposite sign: −39 bp, t −2.5.
- **H-TF3:**
  - 621 trades, gross +1.1 bp per trade, net −13.9 bp, hit rate 41%.
  - Beats matched random entries in only 61% of draws (the criterion needs > 95%).
  - AAPL (t −4.7) and QQQ (t −3.0) are clearly negative.
- **H-TF4:**
  - Sign agreement 44%.
  - Per symbol: AAPL −0.40 (t −3.4) and INTC −0.25 (t −2.2) mean-revert into the close. No symbol has a positive t > 1.3.
- **H-TF5:**
  - Per symbol: BZ −0.29σ (t −1.2), CL −0.35σ (t −1.4), XAG +0.13σ, XAU −0.03σ.
  - The literal 23:00 UTC start gives +0.06σ (t 0.4).
  - Weekends with |R1| > 1σ: +0.09σ (12 weekends).

### Per symbol, clean holdout

| symbol | H-TP1 reversal σ (t) | H-TF1 events, fill / baseline | H-TF2 night − day bp (t) | overnight / intraday bp | H-TF3 trades, net bp (t) | H-TF4 slope (t) |
|---|---|---|---|---|---|---|
| AAPL | −0.11 (−0.7) | 29, 0.45 / 0.47 | −38.9 (−2.5) | −9.4 / +29.5 | 68, −20.2 (−4.7) | −0.40 (−3.4) |
| GOOGL | −0.21 (−1.6) | 23, 0.26 / 0.33 | +6.5 (0.3) | −8.5 / −15.0 | 66, −10.3 (−1.2) | −0.09 (−1.1) |
| INTC | −0.02 (−0.2) | 28, 0.14 / 0.23 | +6.3 (0.1) | −43.9 / −50.2 | 71, −5.6 (−0.2) | −0.25 (−2.2) |
| MSTR | −0.16 (−1.0) | 23, 0.30 / 0.30 | −22.6 (−0.5) | +6.5 / +29.1 | 70, −24.7 (−0.6) | +0.13 (1.2) |
| MU | +0.23 (1.4) | 28, 0.25 / 0.22 | +8.5 (0.2) | −20.6 / −29.2 | 62, +6.0 (0.4) | +0.05 (0.3) |
| NVDA | −0.10 (−0.7) | 33, 0.15 / 0.27 | +37.5 (1.5) | +9.7 / −27.8 | 74, −4.9 (−0.6) | −0.02 (−0.1) |
| QQQ | −0.16 (−0.9) | 27, 0.15 / 0.17 | +11.0 (1.0) | +0.9 / −10.1 | 76, −17.6 (−3.0) | +0.01 (0.0) |
| SOXL | −0.00 (−0.0) | 30, 0.07 / 0.14 | +38.3 (0.4) | −50.2 / −88.5 | 69, −28.5 (−0.5) | +0.06 (0.2) |
| TSLA | −0.26 (−1.5) | 25, 0.28 / 0.38 | −17.7 (−0.7) | −14.3 / +3.4 | 65, −17.7 (−1.6) | −0.04 (−0.4) |

## 4. Automatable-strategy view (7.5 bp per side, random-entry baselines)

Every rule below is fully mechanical from 1h bars and the NYSE calendar. The random baseline replays each trade with the same symbol, side, entry hour and holding on a random session of the same segment and length. For TF2 the baseline is random windows of the same length; for TF5 it is random weekdays at 07:00-12:00 UTC.

| strategy (rule) | segment | trades | gross bp | net bp | net RU (t) | hit rate | random mean RU | beats random |
|---|---|---|---|---|---|---|---|---|
| TP1 fade: at 09:00 ET take the side opposite the overnight move, exit 11:00 ET | in-sample | 83 | +68.8 | +53.8 | +0.22 (1.1) | 0.54 | −0.11 | 91% |
| | holdout | 834 | −4.6 | −19.6 | −0.33 (−2.4) | 0.45 | −0.21 | 6% |
| TP1 fade, only if \|overnight\| > 1 sigma_on | in-sample | 32 | +56.7 | +41.7 | +0.20 (1.0) | 0.62 | −0.14 | 75% |
| | holdout | 246 | −52.1 | −67.1 | −0.52 (−2.5) | 0.41 | −0.20 | **0.2%** |
| TF1 gap fade: if \|gap\| > 1 sigma_on, at 09:00 ET fade the gap, take profit at the previous close, else exit at the close | in-sample | 32 | −5.4 | −20.4 | −0.30 (−1.5) | 0.56 | −0.27 | 46% |
| | holdout | 246 | −48.7 | −63.7 | −0.44 (−1.8) | 0.50 | −0.26 | 1% |
| TF2 long overnight: buy at the 16:00 ET close, sell at 09:00 ET | in-sample | 83 | +41.5 | +26.5 | +0.31 (1.0) | 0.55 | −0.22 | 84% |
| | holdout | 834 | −14.3 | −29.3 | −0.36 (−2.1) | 0.48 | −0.35 | 45% |
| TF3 ORB to close | in-sample | 57 | +7.5 | −7.5 | −0.09 (−0.3) | 0.54 | −0.12 | 30% |
| | holdout | 621 | +1.1 | −13.9 | −0.18 (−2.4) | 0.41 | −0.21 | 61% |
| TF4: at 15:00 ET take the side of (previous close → 10:00 ET), exit 16:00 ET | in-sample | 82 | −12.0 | −27.0 | −0.26 (−2.7) | 0.41 | −0.19 | 47% |
| | holdout | 834 | −4.3 | −19.3 | −0.27 (−4.2) | 0.32 | −0.21 | 3% |
| TF5: at Monday 07:00 UTC take the side of (Sunday reopen → 07:00), exit 12:00 UTC (CL BZ XAU XAG) | in-sample | 33 | +8.6 | −6.4 | −0.16 (−0.4) | 0.55 | −0.41 | 75% |
| | holdout | 76 | −17.3 | −32.3 | −0.67 (−2.0) | 0.39 | −0.38 | 15% |

TF2's other leg, long intraday (09:00→16:00 ET), nets −32 bp in the holdout. The overnight and intraday legs are equally bad over a falling sample: there is no night premium to harvest.

**No effect passes or nearly passes**, so there is no strategy to promote. The only rows that separate from random are **negative**: fading overnight moves larger than 1σ at the open, and fading gaps to the previous close.

**Post-hoc observation, not registered, not tested.**
- What was seen: mirroring the "fade if > 1σ" row gives a "follow the overnight move for the first 2 cash hours" rule. In the holdout it would have made +52 bp gross and about +37 bp net per trade, 246 trades.
- Why it is not a result: in-sample it had the opposite sign (fade +57 bp gross, 32 trades), and it was found by looking at the holdout. It is therefore in-sample evidence only and counts as an extra trial.
- Proposal for the orchestrator to register: **H-TP1b**, "Overnight moves > 1 sigma_on in Bybit equity perps continue over 09:00-11:00 ET". It would be tested forward only, on sessions after 2026-11-11, with a 15 bp round-trip cost and the same random baseline.

## 5. Caveats

- **Short history.**
  - Five months of TradFi perps, with only 16 in-sample sessions after the burn-in and sigma warm-up. Every verdict rests on the holdout: 93 sessions and 19 weekends.
  - The holdout was a falling market for most of these names (mean overnight −14 bp, mean intraday −18 bp a day).
- **1h granularity.**
  - The true 09:30 open price is not observed. The overnight leg ends at 09:00 ET, and the first 30 minutes of the "cash" window are pre-open.
  - The 10:00 ET variant of TF2 does not change the conclusion.
  - Gap fills use bar high/low, i.e. continuous monitoring, which is the right match for the reflection baseline. But a fill on touch is optimistic for a resting take-profit (queue position).
- **H-TP1's registered test asked for flow_3s / index vs last.**
  - flow_3s for the TradFi perps covers only days, so last-price 1h bars were used.
  - Index and last differ by about 5 bp at all hours, so the choice of price series is immaterial at this horizon.
- **H-TF1 baseline.** It uses trailing mean realised variance, which is right-skewed and so puts the baseline slightly too high. The ex-post realised-vol baseline gives about 0 excess (−1 pp). Neither is close to the +5 pp needed.
- **H-TF5.** The DST-aware CME reopen was used, because the title says "after the Sunday CME reopen" and the brief asked for DST handling. The literal fixed 23:00 UTC variant, which equals it under EST, is also reported, and neither passes. Only 19 holdout weekends.
- **Survivorship.** These are the perps Bybit lists today.
- **Multiple testing.** Holm was applied over the six TF-block tests. Variants (the > 1σ subsets, 10:00 ET open, ex-post baseline, 23:00 UTC) are reported for context and were never used for a verdict.
- **Process.** One bug was fixed before recording: the TF5 random-baseline comparison statistic (diary "mistake" entry). No primary statistic was affected.

## Rule candidates

```yaml
- id: R-TF-1
  kind: avoid
  rule: WHEN a Bybit equity perp has moved more than 1 overnight sigma between the NYSE close and 09:00 ET THEN do not fade the move at the cash open, neither for the first two hours nor toward the previous close
  when: tradfi_session.abs_overnight_z > 1 and time in [09:00, 11:00] ET on an NYSE session
  then: no opening fade or gap-fill trades on equity perps
  scope:
    assets: TradFi equity perps (TSLA INTC GOOGL AAPL MSTR NVDA MU QQQ SOXL)
    horizon: 2h to the NYSE close
    sides: both
  effect: holdout fade first 2h net -67 bp per trade (-0.52 RU, t -2.5, 246 trades), worse than 99.8% of matched random entries; gap fade to the previous close net -64 bp (t -1.8), worse than 99% of random; reversal of >1 sigma overnight moves -0.22 sigma (t -1.5)
  confidence: B
  evidence: [H-TP1, H-TF1, research/reports/tradfi_sessions_2026-11-11.md]
  review: proposed H-TP1b (continuation of >1 sigma overnight moves), forward only
  status: provisional
- id: R-TF-2
  kind: expect
  rule: WHEN an equity perp opens more than 1 overnight sigma away from the previous NYSE close THEN expect the previous close to be revisited during the session no more often than a driftless random walk implies (about 22-28% for a typical 1.8 sigma gap)
  when: tradfi_session.abs_overnight_z > 1 at 09:00 ET
  then: use calendar.gap_fill_baseline with session sigma for the fill probability; no gap-fill premium
  scope:
    assets: TradFi equity perps
    horizon: intraday (to the NYSE close)
    sides: both
  effect: holdout fill rate 22.4% vs reflection baseline 27.8% (z -1.8); vs ex-post realised-vol baseline -1.0 pp (z -0.3)
  confidence: C
  evidence: [H-TF1, H-W4]
  review: re-run at 200 holdout sessions
  status: provisional
- id: R-TF-3
  kind: avoid
  rule: WHEN the first 1h close after the first cash hour breaks that hour's high or low in an equity perp THEN do not trade the opening-range breakout to the close
  when: 1h close in 10:00-14:00 ET beyond the high/low of the 09:00 ET bar
  then: no ORB entries on 1h bars
  scope:
    assets: TradFi equity perps
    horizon: intraday (to the NYSE close)
    sides: both
  effect: holdout gross +1.1 bp, net -13.9 bp per trade (-0.18 RU, t -2.4, 621 trades); beats matched random entries in only 61% of draws
  confidence: B
  evidence: [H-TF3]
  review: '-'
  status: provisional
- id: R-TF-4
  kind: avoid
  rule: WHEN the first cash hour of an equity perp (previous close to 10:00 ET) moved strongly THEN do not expect the last hour before the NYSE close to follow it
  when: time 15:00-16:00 ET on an NYSE session
  then: no last-hour momentum trades keyed on the first hour
  scope:
    assets: TradFi equity perps
    horizon: 1h (last cash hour)
    sides: both
  effect: holdout slope -0.09 (t -0.9), sign agreement 44%; follow strategy net -19 bp per trade (t -4.2), worse than 97% of random; AAPL -0.40 (t -3.4)
  confidence: B
  evidence: [H-TF4]
  review: '-'
  status: provisional
- id: R-TF-5
  kind: expect
  rule: WHEN choosing between holding an equity perp overnight (NYSE close to open) or intraday THEN expect no night premium; the overnight and intraday legs earn the same on Bybit perps
  when: holding-period choice for TradFi equity perps
  then: no overnight-only or intraday-only tilt
  scope:
    assets: TradFi equity perps
    horizon: 1 day
    sides: long
  effect: holdout overnight minus intraday +3 bp per day (t 0.14, 93 dates); long-overnight net -29 bp vs long-intraday -32 bp per day; in-sample +96 bp (t 1.5, 16 dates only)
  confidence: C
  evidence: [H-TF2]
  review: re-run at 200 holdout sessions
  status: provisional
- id: R-TF-6
  kind: avoid
  rule: WHEN a commodity perp (CL BZ XAU XAG) moves between the Sunday CME reopen (18:00 ET) and Monday 07:00 UTC THEN do not expect that move to continue into the Monday Europe morning (07:00-12:00 UTC)
  when: Monday 07:00 UTC after a weekend
  then: no continuation entries keyed on the Sunday-night move; oil leans to slight reversal
  scope:
    assets: CLUSDT BZUSDT XAUUSDT XAGUSDT
    horizon: 5h
    sides: both
  effect: holdout continuation -0.13 sigma (t -0.9, 19 weekends), CL -0.35 sigma, BZ -0.29 sigma; follow strategy net -32 bp (t -2.0)
  confidence: C
  evidence: [H-TF5]
  review: re-run at 40 holdout weekends
  status: provisional
- id: R-TF-7
  kind: expect
  rule: WHEN outside NYSE hours THEN expect Bybit equity perps to keep trading (about 0.25-0.5x regular-hours turnover and 0.4-0.65x the hourly abs return overnight, about 0.1-0.15x turnover and 0.25-0.33x abs return at weekends) and the 09:00 ET bar to carry 1.7-2.5x the regular-hours abs return; weekends are quiet, not frozen, and the index moves too
  when: calendar.nyse_open == false
  then: treat overnight and weekend moves as traded price discovery (no gap at the open); keep risk sigma unshrunk across the 09:00 ET bar
  scope:
    assets: TradFi equity perps
    horizon: 1h
    sides: both
  effect: 9 equity perps, Apr-Nov 2026, last and index 1h bars; basis to index about 5 bp at all hours
  confidence: B
  evidence: [H-TP1, research/reports/tradfi_sessions_2026-11-11.md]
  review: qualifies BELIEFS P5 / R-PX-9 wording ("freeze")
  status: provisional
```
