# Weekend options study: H-OPT1 / H-OPT2 / H-OPT3 (2026-11-07)

Code: `src/ai_trader/research/options_study.py` (Black-76 pricer, quote extraction, hedge simulation, verdicts);
tests: `tests/test_options_study.py`. Run `uv run python -m ai_trader.research.options_study insample|holdout`.
Verdicts are in `research/hypotheses_results.jsonl` (6 rows, data + note fields filled).

**Bottom line.** Neither sleeve is deployable. The delta-hedged Fri→Mon short ATM daily straddle lost money
net of costs on both underlyings, in both segments: in-sample −0.17 % (BTC) and −0.27 % (ETH) of notional per
weekend, holdout −0.10 % and −0.13 %. Before costs the edge is only +0.09 to +0.25 % of notional. Costs are about
0.35 %, and three quarters of that is the hourly hedge. The large H-W7 weekend variance discount doesn't reach this
trade, for two reasons. First, the Friday 3-day daily IV is already about 20 % below the Monday IV. Second, the
Friday 08:00→21:00 UTC session realises 2–2.8× its pro-rata implied variance and eats the premium.

## 1. Sample

| | BTC weekend | BTC weekday | ETH weekend | ETH weekday |
|---|---|---|---|---|
| llm_in_sample trades (2025-08-22 … 2026-06-26) | 45 | 44 | 45 | 44 |
| clean_holdout trades (2026-07-03 … 2026-10-09) | 15 | 14 | 15 | 14 |
| skipped 2025 (expiry 3 days out had no trades / was not listed) | 33 | 33 | 33 | 33 |
| dropped: Mon 2026-06-29 → Thu 07-02 straddles the cutoff | – | 1 | – | 1 |

- The raw-trade files cover 2020-01-01 … 2026-10-09. Before 2025-08-22, no Friday file has a single trade in the
  Monday daily expiry, and no Monday file has one in the Thursday expiry. Deribit only listed dailies up to two days
  ahead then. So 2020 to mid-2025 can't be tested (about 290 Fridays per underlying, not counted as skips).
- From 2025-08-22 there were **no skips**:
  - every week had ≥ 3 trades within ±2 strikes of ATM;
  - the near-money fallback IV was used for 1 in-sample trade (ETH weekday, 2025-12-22) and 8 holdout trades (1 ETH
    weekend, 3 BTC and 4 ETH weekday controls), because the Monday-entry books were thinner after July;
  - no hedge windows were missing.
- The holdout has 14–15 weeks, above the 8-week minimum. In-sample is under one year: 19 weekends in 2025 and 26 in
  2026 H1.

## 2. Judgement calls (fixed before P&L was computed)

1. **Entry time and forward.**
   - Entry is at 08:00 UTC, and T = 72 h / 8760 h exactly.
   - F0 is the index_price of the last trade in the 04:00–08:00 window; r = 0.
2. **ATM strike.**
   - The strike step is the smallest gap between traded strikes of that expiry within ±10 % of F0.
   - K = round(F0/step)·step.
   - "±2 strikes" means |K_i − K| ≤ 2·step.
3. **Entry IV.**
   - Primary: the median trade iv within ±2 strikes, if there are ≥ 3 such trades.
   - Fallback: the amount-weighted iv of trades with |ln K/F0| ≤ 5 %.
   - All trades are included: buys, sells, blocks and combos.
4. **Half-spread.** Median |price − mark_price| × index_price, over the trades that set the IV. It is charged on both
   legs at entry only.
5. **Fees.**
   - Fee per leg = min(0.03 % F0, 12.5 % × leg price). In practice the 0.03 % rate always bound: 0.06 % per straddle.
   - Following the registry, no settlement fee is charged. This is mildly optimistic: Deribit and Bybit charge a
     0.015 % delivery fee.
6. **Hedge.**
   - Hedge instrument: the Bybit perp, using lake 1h bars. The perp stands in for the Deribit index; the basis
     between them is ignored.
   - Rebalanced 72 times at hourly closes.
   - Delta is Black-76 at the **entry IV**, with F = the perp price.
   - Cost is 5 bps on notional changed, including the opening trade and the unwind.
7. **Settlement.**
   - Settlement price = the open of the 08:00 UTC bar on the expiry date. Deribit actually settles on a 30-min index
     TWAP.
   - P&L is computed as for a linear USD option; coin margining of the inverse contracts is ignored.
8. **Segments and statistics.**
   - Segment is set by entry date; a trade straddling the cutoff is dropped from both segments.
   - The primary unit is net P&L in % of notional. Premium units are also reported, and H-OPT2 uses premium units, as
     registered.
   - t-statistics are Newey–West.
   - H-OPT1 requires both underlyings to pass.
   - H-OPT2's primary test averages BTC and ETH per week (clustering by time); per-underlying results are also shown.
   - H-OPT3 is computed per underlying and requires both to pass.
9. **Process.**
   - One dry run of the in-sample segment was made to check for bugs. It was inspected against raw trades for
     2025-08-22 BTC and 2026-05-22 ETH; no code or parameter changed after it.
   - In-sample verdicts were recorded before the holdout was computed.
   - No diary entry was written, because this task was restricted to the files I own.

## 3. Results

### H-OPT1: does the weekend straddle earn net of costs? Verdict: contradicts (in-sample), contradicts (holdout)

Per weekend, short 1 straddle. Columns are net P&L in % of notional (NW t) and in premium units, then the share of
weekends with a loss.

| | in-sample % notional | in-sample premium units | in-sample loss share | holdout % notional | holdout premium units | holdout loss share |
|---|---|---|---|---|---|---|
| BTC hedged | **−0.168 (t −1.61)** | −0.071 | 56 % | **−0.099 (t −0.93)** | −0.036 | 47 % |
| ETH hedged | **−0.272 (t −1.35)** | −0.090 | 47 % | **−0.130 (t −0.61)** | −0.024 | 47 % |
| BTC unhedged (descriptive) | +0.016 (t 0.06) | +0.000 | 49 % | +0.469 (t 1.60) | +0.166 | 20 % |
| ETH unhedged (descriptive) | +0.130 (t 0.32) | +0.030 | 36 % | +0.562 (t 1.74) | +0.178 | 27 % |

By year (hedged, % of notional, plain t):

| | BTC 2025 (n=19) | BTC 2026 H1 (n=26) | BTC holdout (n=15) | ETH 2025 (n=19) | ETH 2026 H1 (n=26) | ETH holdout (n=15) |
|---|---|---|---|---|---|---|
| net P&L | −0.19 (t −1.17) | −0.15 (t −1.23) | −0.10 | −0.41 (t −1.15) | −0.17 (t −0.89) | −0.13 |

Decomposition of the weekend trade (means, % of notional):

| | premium | gross hedged P&L | spread (2 legs) | fees | hedge cost | net |
|---|---|---|---|---|---|---|
| BTC in-sample | 2.76 | +0.18 | 0.037 | 0.06 | 0.256 | −0.17 |
| ETH in-sample | 3.91 | +0.09 | 0.049 | 0.06 | 0.257 | −0.27 |
| BTC holdout | 2.20 | +0.25 | 0.029 | 0.06 | 0.263 | −0.10 |
| ETH holdout | 2.91 | +0.23 | 0.032 | 0.06 | 0.267 | −0.13 |

Hourly re-hedging a 3-day ATM straddle turns over about 5 units of notional, which costs 7–12 % of the premium. That
alone is larger than the gross edge.

### H-OPT2: weekend vs the same-length weekday straddle (Mon→Thu). Verdict: contradicts (in-sample), supports (holdout)

The difference is the paired Fri−Mon net P&L in premium units, same ISO week.

| | in-sample mean diff | in-sample NW t | in-sample n | holdout mean diff | holdout NW t | holdout n |
|---|---|---|---|---|---|---|
| pooled (primary) | −0.018 | −0.33 | 44 | +0.070 | +0.67 | 14 |
| BTC | +0.032 | +0.67 | 44 | +0.146 | +1.63 | 14 |
| ETH | −0.069 | −1.04 | 44 | −0.007 | −0.06 | 14 |

- The weekday control is also negative: in-sample −0.30 % (BTC, t −3.06) and −0.02 % (ETH); holdout −0.44 % (BTC)
  and +0.01 % (ETH).
- The holdout "supports" is a sign-only pass on 14 weeks. Because in-sample failed, the hypothesis is **not
  supported**.
- In-sample, the weekend is not better per unit of premium. The market already charges about 20 % less IV for the
  weekend expiry: the median Fri/Mon IV ratio is 0.80 (BTC) and 0.77 (ETH).

### H-OPT3: is the tail survivable at small size? Verdict: contradicts (in-sample), "supports" on its registered holdout criterion

Size is set so the in-sample q99 loss equals 1 % of capital.

| | q99 loss (% notional) | notional / capital | in-sample ann. return | in-sample worst loss | holdout ann. return | holdout worst loss |
|---|---|---|---|---|---|---|
| BTC | 1.91 % | 0.52× | **−4.6 %** | 1.02 % | −2.7 % | 0.67 % |
| ETH | 3.95 % | 0.25× | **−3.6 %** | 1.01 % | −1.7 % | 0.51 % |

The tail is survivable but nothing is being paid for bearing it. The registered holdout criterion is worst loss
< 3 %, which passes trivially; annualised return is negative in both segments. Unhedged at the same size
(descriptive): in-sample +0.4 % (BTC) and +1.7 % (ETH) per year; holdout +12.7 % and +7.4 % on 15 weeks.

## 4. Realised / implied variance vs H-W7

| | pooled RV/IV, weekend trade | median RV/IV, weekend trade | pooled RV/IV, weekday control |
|---|---|---|---|
| BTC in-sample | 1.01 | 0.89 | 1.01 |
| ETH in-sample | 1.12 | 0.90 | 0.96 |
| BTC holdout | 0.90 | 0.64 | 1.12 |
| ETH holdout | 1.04 | 0.75 | 1.08 |

The 3-day implied variance is spread evenly over the 72 h. Splitting realised variance by sub-window gives:

| | Fri 08–21 UTC (13 h) | TradFi weekend, Fri 21 → Sun 22 (49 h) | Sun 22 → Mon 08 (10 h) |
|---|---|---|---|
| BTC in-sample | 2.00× | **0.59×** | 1.74× |
| ETH in-sample | 2.15× | **0.74×** | 1.66× |
| BTC holdout | 2.81× | **0.34×** | 1.14× |
| ETH holdout | 2.53× | **0.58×** | 1.36× |

The TradFi weekend itself is cheap, close to H-W7's range. But a Deribit daily can't isolate it: the window has to
include Friday's US session and the Sunday-night/Asia reopen.

## 5. Tail: 10 worst weekends (hedged)

Columns:
- **sub-window returns**: returns for Fri 08–21 / TradFi weekend / Sun 22 → Mon 08 UTC.
- **biggest 6h move**: the largest move over any 6-hour stretch, with its start time.

In-sample:

| u | Friday | IV % | Fri→Mon return % | max up / max down % | sub-window returns % | biggest 6h move | RV/IV | net % notional | net premium units | unhedged % |
|---|---|---|---|---|---|---|---|---|---|---|
| ETH | 2025-08-22 | 64 | +5.9 | +13.2 / −1.9 | +11.4 / −1.4 / −4.1 | +12.1 % from Fri 13:00 | 2.7 | −3.97 | −0.86 | −1.61 |
| ETH | 2025-11-28 | 49 | −5.5 | +2.4 / −6.2 | +1.8 / −1.1 / −6.2 | −7.2 % from Sun 23:00 | 2.5 | −3.91 | −1.11 | −1.96 |
| ETH | 2026-05-22 | 25 | −0.6 | +0.8 / −4.9 | −2.6 / −0.1 / +2.2 | +4.0 % from Sat 15:00 | 4.9 | −3.35 | −1.83 | +1.24 |
| BTC | 2026-06-05 | 50 | +0.3 | +0.8 / −6.1 | −2.4 / +0.3 / +2.4 | −4.2 % from Fri 13:00 | 2.1 | −1.95 | −0.54 | +3.06 |
| ETH | 2025-10-10 | 49 | −3.4 | +0.7 / −15.7 | −11.2 / +6.7 / +1.0 | −10.1 % from Fri 15:00 | 5.3 | −1.92 | −0.54 | +0.01 |
| BTC | 2026-02-27 | 40 | −2.5 | +0.4 / −6.8 | −3.1 / +0.1 / +0.5 | −4.2 % from Sat 01:00 | 2.8 | −1.87 | −0.64 | +0.60 |
| ETH | 2026-02-27 | 56 | −4.6 | +0.0 / −9.8 | −5.5 / +0.3 / +0.6 | −4.7 % from Sat 01:00 | 2.3 | −1.85 | −0.45 | −0.17 |
| ETH | 2026-06-05 | 68 | −0.3 | +1.1 / −9.8 | −4.6 / +1.9 / +2.4 | −6.5 % from Fri 13:00 | 1.9 | −1.76 | −0.36 | +4.64 |
| BTC | 2025-11-28 | 34 | −4.9 | +1.6 / −5.8 | +0.3 / +0.0 / −5.2 | −6.1 % from Sun 23:00 | 2.5 | −1.75 | −0.72 | −2.63 |
| BTC | 2025-08-22 | 36 | −1.4 | +3.5 / −1.4 | +3.5 / −3.5 / −1.4 | +4.1 % from Fri 12:00 | 1.5 | −1.45 | −0.56 | +1.16 |

- Five of the ten in-sample tails come from the **Friday US session** (biggest 6h move starting 12:00–15:00 UTC):
  - 2025-08-22 was the Jackson Hole speech day;
  - 2025-10-10 was the Friday-evening liquidation crash: ETH −15.7 % intra-window, but the hedged loss was only
    −1.9 % of notional because the hourly hedge followed the trend.
- Two come from the hours just after the TradFi close (2026-02-27: biggest move starts Sat 01:00 UTC).
- Two come from the **Sunday-night reopen** (2025-11-28).
- One is a weekend whipsaw: 2026-05-22 ETH, IV only 25 %, RV/IV 4.9.
- Several are **hedged losses on weekends that were profitable unhedged** (a V-shaped path, so the hedge bought high
  and sold low).
- The holdout worst weekends are all small. The worst are ETH 2026-08-21 at −2.0 % and ETH 2026-09-11 at −1.96 %.
  The worst BTC weekend is 2026-08-21 at −1.29 %. Each was a Friday-session rally of +5 to +6 %.

## 6. Execution feasibility: Bybit options

Bybit public API snapshot, Sat 2026-11-07 ≈ 23:16 UTC (`/v5/market/instruments-info`, `/v5/market/tickers`,
category=option).

**Listings.**
- BTC and ETH both have USDT-settled options (664 and 582 instruments).
- Daily expiries at 08:00 UTC are listed three days ahead (Sun 8, Mon 9, Tue 10 Nov), then weeklies on Fridays. A
  Friday-08:00 sale of the Monday daily is therefore listable on Bybit.
- The delivery fee is 0.015 %.

**Near-ATM quotes.**

| | half-spread on 9-NOV (Mon) dailies | half-spread on 10-NOV dailies | quoted size |
|---|---|---|---|
| BTC | 1 tick (5 USDT) = 0.33 bp of underlying, about 0.5 % of option mid | same | about 1–8 BTC at top of book |
| ETH | 1 tick (0.1 USDT) = 0.22 bp of underlying, 0.2–0.3 % of mid | same | about 7–60 ETH at top of book |

**Comparison with Deribit.**
- The Deribit trade-vs-mark half-spread estimate used here is 1.5–2.4 bp of underlying per leg.
- That is about 5–10× wider than Bybit's quoted ATM touch. Bybit is cheaper on the option leg, at small size.
- This doesn't rescue the trade. The whole option spread (0.03–0.05 % of notional) is small next to the 0.26 % hedge
  cost and the negative gross edge.

**Term structure in the snapshot.** ATM mark IV is 18 % for Sun 8, 28 % for Mon 9, 34 % for Tue 10 and 39 % for
Fri 13 Nov (BTC). The market prices the weekend lull explicitly.

## 7. Read

- **Not a deployable sleeve at any size**, as registered. The hedged trade has a negative expectation in both
  segments, and the weekday control is no better.
- The unhedged straddle was positive in the holdout: +0.47 % and +0.56 % of notional, t ≈ 1.6–1.7, n = 15. It was
  flat in-sample (t 0.06 / 0.32). That is a new idea, not evidence. It would need a new registration, for example
  H-OPT4: unhedged, or hedged daily or on a band, possibly on Bybit's tighter book. It would be read forward only,
  because this holdout has now been seen.
- If someone insists on a token position: about 0.25× (ETH) to 0.5× (BTC) notional per unit of capital, which caps
  the q99 loss at 1 %. On these numbers it would cost roughly 2–5 % per year.
