# Relative value / fair value: H-RV1, H-RV2, H-RV3 (2026-11-07)

Code: `src/ai_trader/features/relative_value.py` (causal `rolling_ols`, `factor_residuals`, `spread_z`,
`ar1_half_life`, `basis` / `basis_frame` / `basis_excursions` for H-RV4; tests in `tests/test_relative_value.py`),
`src/ai_trader/research/rv_study.py` (`python -m ai_trader.research.rv_study insample|holdout`),
live tools `src/ai_trader/tools/relval.py` (`relval.factor_gap`, `relval.pair`). Verdicts are in
`research/hypotheses_results.jsonl`. H-RV4 is forward-only: the measurement (`basis_frame` + `basis_excursions`)
is built; flow_3s has ~6 h of data, so nothing is tested.

## Design (fixed before results; full text in the rv_study docstring)

- Bybit 1h `last` closes, legs synchronous (inner join on event_ts; a 1h return only between rows exactly 1 h apart).
  Computed on full history, then split: in-sample = outcome window ends <= 2026-07-01, holdout = decided >= cutoff,
  straddlers dropped from both. Decision at the close of bar t, outcome over the next 24 h. One event per instrument
  per 24 h (greedy). Costs 7.5 bps x 4 legs = 30 bps per event (as registered).
- H-RV1: per alt, OLS (with intercept) of 1h log returns on BTC and ETH returns over the previous 720 h (>= 576 obs);
  residual at t uses the fit through t-1; z = 72h residual sum / (resid std x sqrt 72); event |z| >= 2. Trade: fade
  the residual with a static dollar hedge (betas fit through t); P&L = -sign(z) (R_alt - b_BTC R_BTC - b_ETH R_ETH),
  simple 24h returns, minus 30 bps. Stats: per-origin mean across alts, Newey-West t on the time-ordered cluster
  means. Simulated null: same code on r* = a + b F + s e (full-sample OLS residuals, iid random signs), 5 seeds.
- H-RV2 (BZ-CL) / H-RV3 (XAUT-XAU): s = ln A - ln B, z vs mean/std of the previous 720 synchronous hours;
  event |z| >= 2; primary = (toward-mean 24h change - 30 bps) / trailing std; plain t over events. H-RV3 weekend
  split by whether the decision time is in the TradFi weekend (Fri 17:00 -> Sun 18:00 ET); "faster" = share of
  the deviation reverted after 4 h, weekend vs weekday (reported, not part of the verdict).
- Data caveat found before testing: XAUUSDT starts 2026-03-09 (not 2025-01 as the registry assumed for XAUT), so
  H-RV3 in-sample is 2026-04-08 -> 2026-06-30 only. BZUSDT starts 2026-05-13: one in-sample event for H-RV2.

## In-sample results (`llm_in_sample`, recorded 2026-11-07 23:41 UTC, before the holdout was computed)

| id | metric | value | t (NW) | n | verdict |
|---|---|---|---|---|---|
| H-RV1 | mean net reversion, % of alt notional | **-0.41%** | -3.32 (plain -4.10) | 3394 events / 2934 timestamps | contradicts |
| H-RV1 | gross reversion % | -0.11% | -0.89 | | |
| H-RV1 | residual units (24h residual sum / sd sqrt24) | +0.039 | +0.97 | | |
| H-RV1 | net with beta-weighted hedge costs | -0.43% | -3.47 | mean abs beta sum 1.13 | |
| H-RV1 | hit rate (net > 0) | 50.2% | | | |
| H-RV1 null | sign-flipped residuals, 5 seeds: gross %; res units | -0.04 / +0.14 / +0.09 / -0.00 / -0.08 %; +0.04..+0.10 | | ~3700 each | |
| H-RV2 | net reversion / trailing std | +0.69 (1 event, 2026-06-18) | n/a | 1 | inconclusive (registry: verdict pooled) |
| H-RV3 | net reversion / trailing std | **-1.51** | -5.58 | 18 | contradicts |
| H-RV3 | gross reversion / std; bps | +0.64 sd (t 3.26); +11.3 bps (t 3.01) | | 18 | |
| H-RV3 | share of deviation reverted 4h / 24h | 0.11 / 0.28 | 2.57 / 3.15 | | |
| H-RV3 | trailing spread std (median); mean abs deviation at entry | 13.8 bps; 33.5 bps | | | |
| H-RV3 weekend | gross / net (sd); share reverted 4h / 24h | +1.37 / -0.59; 0.12 / 0.56 | net t -1.11 | 5 | |
| H-RV3 weekday | gross / net (sd); share reverted 4h / 24h | +0.36 / -1.86; 0.10 / 0.18 | net t -7.05 | 13 | |
| H-RV3 | 4h reversion weekend - weekday | +0.02 | Welch 0.15 | | |

Per-alt H-RV1 net means range from -1.46% (XLM) to +0.26% (DOT, ATOM); only DOT, ATOM and AVAX are positive,
none was selected on. In-sample read:
- **H-RV1: no reversion at all, before costs.** The gross factor-hedged return after a 2-sigma 72h residual is
  -0.11% (t -0.9), indistinguishable from zero, so 30 bps of costs make it reliably negative. The small positive
  residual-unit figure (+0.04) is inside the simulated null's range (+0.02..+0.10), i.e. it is a by-product of the
  rolling fit (the event window sits inside the next fit's intercept), not reversion.
- **H-RV3: the spread does revert gross** (about a third of a 2-sigma deviation over 24 h, t ~3), and faster when
  the event starts in the TradFi weekend at 24 h (0.56 vs 0.18) though not at 4 h. But the spread's std is ~14 bps
  and the move back is ~11 bps against 30 bps of round-trip costs on two legs: untradeable at taker costs.

## Holdout results (`clean_holdout`, events decided on/after 2026-07-01; recorded 2026-11-07 23:43 UTC)

| id | segment | metric | value | t | n | verdict |
|---|---|---|---|---|---|---|
| H-RV1 | holdout | mean net reversion % | **-0.26%** (gross +0.04%) | NW -0.91 (gross +0.16) | 299 events / 258 ts | contradicts |
| H-RV1 | holdout | residual units; hit rate | +0.085; 52.8% | +0.77 | | |
| H-RV1 null | holdout | gross % (5 seeds); res units | +0.06 / +0.01 / -0.35 / +0.28 / +0.00 %; -0.12..+0.17 | | ~270 | |
| H-RV1 | pooled (context) | net / gross % | -0.40% / -0.10% | -3.45 / -0.85 | 3694 | (not recorded) |
| H-RV2 | holdout | net reversion / trailing std | **-0.51** (gross -0.11) | -3.53 (gross -0.77) | 33 | contradicts |
| H-RV2 | pooled | net / std (registered criterion) | **-0.47** (gross -0.08); net -40 bps, toward -10 bps | -3.28 | 34 | contradicts |
| H-RV2 | pooled | trailing std (median); mean abs dev at entry | 67 bps; 248 bps | | | |
| H-RV3 | holdout | net reversion / std | **-2.54** (gross +0.64, t 4.63) | -13.9 | 43 | contradicts |
| H-RV3 | holdout | toward bps; share reverted 4h / 24h | +6.4 bps; 0.09 / 0.28 | 4.93 | | |
| H-RV3 | holdout | trailing std (median) | 9.4 bps | | | |
| H-RV3 | holdout weekend / weekday | gross sd; share reverted 4h | +0.65 / +0.64; 0.14 / 0.08 (Welch 0.52) | | 9 / 34 | |
| H-RV3 | pooled (context) | gross sd; net sd; toward bps | +0.64 (t 5.8); -2.24; +7.8 bps | | 62 | (not recorded) |

## Read

- **H-RV1 (contradicted, both segments).** After a 2-sigma 72h residual vs the BTC/ETH factor model, the hedged alt
  does not revert over 24 h: gross ~0 in-sample (-0.11%, 3394 events) and in the holdout (+0.04%), against 30 bps
  of costs. The residual-unit figure is the same size as the simulated null's, i.e. a fitting artefact. Alt
  idiosyncratic moves at the 3-day scale behave like random walks (or, if anything, continue slightly). The
  fair-value gap is a useful *descriptor* (how much of a coin's move is not explained by BTC/ETH) but not a fade
  signal at this horizon.
- **H-RV2 (contradicted, pooled).** BZ-CL on Bybit perps shows no reversion at all at 24 h (gross -0.08 sd, median
  ~0). The spread's 30-day std is ~67 bps and it trends (contract roll / genuine Brent-WTI moves, 30-day AR(1)
  half-life ~70 h live), so a 30-day z is not an anchor. 34 events, all but one in the holdout.
- **H-RV3 (contradicted net; the gross effect is real).** XAUT-XAU reverts reliably: ~28% of a 2-sigma deviation
  closes within 24 h (t 5.8 pooled), and in-sample weekend events reverted more by 24 h (0.56 vs 0.18) though the
  4h comparison is a wash (Welch t 0.15 IS, 0.52 holdout) and the holdout shows no weekend difference. But the
  spread's std is only 9-14 bps; the average reversion is 6-11 bps against 30 bps of taker costs on 4 legs. It only
  becomes interesting with maker fills on both legs (unvalidated: H-F5) or as an *execution* hint (which of the two
  gold instruments is cheap right now), not as a standalone trade.
- Nothing here is tradeable at taker costs. Tool status stays `planned` (descriptive use only). H-RV4 (perp-spot
  basis, flow_3s) is built and waits for 14 days of data; it is the one RV idea whose horizon (minutes) and leg
  structure (same asset) give it a chance against costs.

## Live readings (as of 2026-11-07 23:44 UTC, data through the 22:00 bar)

| symbol | beta_BTC | beta_ETH | 72h residual z | gap vs 720h factor fair value | R2 |
|---|---|---|---|---|---|
| ETHUSDT (vs BTC only) | 1.25 | | -0.21 | -0.6% | 0.82 |
| SOLUSDT | 0.68 | 0.58 | +1.53 | -8.7% | 0.81 |
| ADAUSDT | 0.59 | 0.88 | +1.08 | -9.5% | 0.73 |
| AVAXUSDT | 0.36 | 0.97 | +1.37 | -6.0% | 0.66 |
| LINKUSDT | 0.50 | 0.88 | +1.40 | -2.8% | 0.72 |
| XRPUSDT | 0.61 | 0.51 | -1.27 | -5.9% | 0.72 |
| DOGEUSDT | 0.47 | 0.72 | +0.91 | -4.4% | 0.79 |

Pairs: BZ/CL spread +741 bps vs 30d mean +864 (sd 142): z -0.87, AR(1) half-life 71 h. XAUT/XAU -45.7 bps vs mean
-31.5 (sd 12.5): z -1.14, half-life 26 h. No reading is at the |z| >= 2 event level, and per the verdicts above none
would be a trade if it were.
