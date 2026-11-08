# What the prior projects teach AI_Trader

_2026-11-07. Sources: [ETH_BMNR_PROJECT.md](ETH_BMNR_PROJECT.md), [BACKTEST_TREND_FOLLOWING.md](BACKTEST_TREND_FOLLOWING.md) (agent reviews, read-only). This note is the synthesis: what we adopt, what we import, what we now test._

## 1. The headline: two serious projects, almost no deployable edge

- **ETH_BMNR:** 15 strategy families, 0 deployable (its own verdict). The mNAV premium had no edge, BMNR was too illiquid, pairs had 0/10,080 configs surviving deflation, and order-book alpha was real but smaller than costs.
- **TREND_FOLLOWING:** the champion's Sharpe of 0.8 came from roughly 300–500 configs across 13 LLM-agent rounds that reused the "OOS" half every round. The exact live rule on truly unseen 2020–22 data has Sharpe **−0.12**. Only a pre-registered, hedged options strangle survives (deflated Sharpe 0.57, CI includes 0).

Both prior projects reached the same conclusion AI_Trader was built around: **most backtested edge is selection bias and cost underestimation.** Our in-sample trend Sharpe of 0.2–0.3 (H-X2b) is in line with theirs.

## 2. Process lessons, now rules (added to RESEARCH_PROTOCOL)

1. **An automated research loop consumes its holdout on every selection round.** A segment used to choose between variants is in-sample, whatever it's called. Our holdout is touched once per registered hypothesis, and agent loops may only select on `llm_in_sample`.
2. **Count trials and deflate.** Any result chosen from more than 5 variants reports the deflated Sharpe.
3. **Multi-leg instruments need synchronous data.** The BZ/CL "99.3% win rate" came from combining Brent and WTI extremes that didn't happen at the same time, filling on favourable intrabar order, and using an ATR that included the current bar. Our paper broker already assumes the worst intrabar order. Spreads must be built from synchronous legs.
4. **Fill simulation is optimistic by default.** Simulated maker fills flattered results by about 3 bps; an order-book simulator cut Sharpe from 3.15 to 0.78. Execution assumptions get their own tests (H-F5).
5. **Three days of microstructure prove nothing.** Our flow hypotheses need 30+ days (H-F*).

## 3. Evidence that changes our priors

| finding (source, label) | effect on AI_Trader |
|---|---|
| 1h trend: ~0 net, costs 7–10%/yr (TF, IS+OOS) | matches our 1h results → **H-X2b/H-X2c**: 4h beats 1h in both segments |
| resting stops hurt, 1h touches revert within the hour (TF, pre-registered OOS, p=0.01) | **our bracket orders use resting stops** → H-X1 (close-confirmed stop + catastrophe stop) |
| Friday-sold weeklies beat other days (TF, pre-registered, t 3.1/3.6) | → **H-W7: strongly supported in both segments** (§5) |
| OBI IC 0.2 at 1 s, unprofitable as taker or maker (EB, 3 days OOS) | flow as entry *timing*, not standalone alpha → H-F5 |
| vol forecasts ≈ persistence (EB, OOS R² 0.439 vs 0.389) | HAR/GARCH-t remain the bar any vol model must beat |
| high trailing RV → weaker trend returns (TF, OOS IC −0.19) | → H-X3 |
| post-jump vol half-life ~5 h (TF jump work) | → H-S4; consistent with our H-S2/S3 (high-vol shocks revert) |
| Fear & Greed has no contrarian effect (TF, IS) | downweight `sent.*` built on F&G |
| stock-perp funding: mean +31%/yr, median 0 (EB) | descriptive. Stock perps are being collected → H-TP1 |

## 4. Data imported (scripts/import_prior_projects.py, validated)

| dataset | lake location | validation |
|---|---|---|
| Bybit BTCUSDT 5m, 2020-04 → 2026-09 (683k) | bars venue=bybit freq=5m | 5m→1h matches our bars **exactly from 2022**; 2021 has 134 venue-vintage mismatch hours (same as the data audit) |
| Binance ETHUSDT 1h (2017→) and 1m (2023→, 1.9M) | bars venue=binance | spacing checked (29 historical exchange gaps) |
| Deribit BTC/ETH-PERPETUAL 1h, 2020→2026-10 | bars venue=deribit | spacing exact |
| Deribit funding 8h/1h, 2019→ | alt_series source=deribit | known_ts = observation + 2 s |
| Deribit SVI surfaces: ATM / RR25 / BF25 at 7/30/90 d, term slopes, basis, 2,474 days | alt_series source=deribit_svi | fitted on trades up to 08:00 UTC, so known_ts = 08:00 + 1 min |

**Not imported:**
- TF's 50-coin bars (a survivor universe; 1h covers only one year);
- `results/live_snapshot.json` (contains account metadata, **do not import**);
- the BMNR L2 files (3 days only);
- EDGAR fundamentals and Yahoo commodities (P1: import when a hypothesis needs them).

## 5. First results on the new hypotheses

- **H-W7 (weekend variance risk premium): SUPPORTED in both segments.** Weekend realised variance against variance implied by Friday's 7-day ATM IV over the same calendar time:

  | | weekend | weekday | difference (t) |
  |---|---|---|---|
  | BTC, in-sample | 0.35× | 0.84× | t −17 |
  | BTC, holdout (15 weeks) | 0.20× | 0.97× | t −7.7 |
  | ETH, in-sample | 0.45× | 0.95× | t −15 |
  | ETH, holdout | 0.27× | 0.95× | t −5.3 |

  Options price calendar time; TradFi-closed weekends deliver a fraction of it. Implications:
  1. weekend σ for stops, sizing and first passage should be roughly 0.5–0.6× the Friday σ̂ (vol units);
  2. a candidate **weekend short-vol sleeve** (Bybit lists BTC/ETH options), which needs its own cost and tail-risk study;
  3. the sharpest prior yet for the weekend layer.
- **H-X2 → H-X2b → H-X2c:** the first metric wasn't scale-free (invalidated), and the second run restarted indicator warm-up in the holdout (corrected). Corrected result: "contradicts" as registered (1d is not better), but **4h > 1h holds in both segments**, registered as H-X2c for forward testing. Trend Sharpe is about 0.2–0.3 in-sample and negative in the holdout (Jul–Nov 2026).

## 6. Reusable code (not yet adopted)

- ETH_BMNR: Rust crates and the BZ/CL grid bot (`trader/`), and Tardis L2 parsing.
- TREND_FOLLOWING: `live_engine/` (Python signal → Rust execution on Bybit), the order-book execution simulator, the Deribit options fetchers and the SVI fitter, `gdelt_fetch.py`.

Adopt case by case once a hypothesis needs them. The SVI fitter and options fetchers are the first candidates if a weekend short-vol sleeve goes ahead.

## 7. Second round of tests (2026-11-07 evening)

| hypothesis | verdict | key number |
|---|---|---|
| H-S4 post-jump vol contracts | **supported** both segments | excess-variance half-life 2–3 h; realised/EWMA forecast 0.73 |
| H-X1 close-confirmed stops | not supported | B−A −0.05R (ema_stack n=299); worst −1.9R vs −1.0R → **keep resting stops** |
| H-X3 RV percentile → trend returns | not supported | cross-sectional IC ≈ 0 |
| H-P1 / H-P2 positioning strategies | not supported | funding_contrarian −1.7 (in-sample); OI ≈ buy-and-hold |
| H-W6 TradFi-perp weekend confirmation | untestable | Bybit stock perps freeze on weekends (RV 0.2×, turnover 0.1–0.2×) |

Live scoreboard: [research/reports/SCOREBOARD.md](../reports/SCOREBOARD.md) (`scripts/scoreboard.py`).

**What the evidence says to build next:**
1. **Weekend-aware σ (H-W1, H-W7):** scale σ̂ over TradFi weekends (prior about 0.6× Friday σ̂ for realised vol), used in first passage, stops and sizing.
2. **Jump-robust σ (H-S4):** stop single jumps from inflating the EWMA.
3. **Shock playbook (H-S2/S3):** continuation after shocks in calm cells, reversal in normal and high-vol cells.

All three are deterministic vol and regime tools, and all are backtestable.
