# ETH_BMNR_PROJECT — post-mortem for AI_Trader (read-only review, 2026-11-07)

Source `C:\Users\Asus\Documents\ETH_BMNR_PROJECT` (work 2026-08-29→09-01); paths relative to it.

## 1. Purpose & thesis
BMNR = BitMine Immersion Technologies, "the largest public Ethereum treasury"; Bybit lists a **BMNRUSDT stock perp** (launched 2026-05-14). README: *"BMNR's perp trades at a time-varying premium/discount (mNAV) to the market value of its ETH holdings. When mNAV strays from a fair band — driven by the staking compound yield, costs and dilution disclosed in SEC filings — we long/short BMNR vs a delta-hedged ETH perp and harvest the reversion."* Fair value `ETH·eth_per_share·exp(net_drift)`, OU z-bands. After it failed, the repo pivoted to ETH order-flow alpha, ETH market making, cross-asset pairs (BZ/CL, XAU/XAG, KO/PEP, 90-coin universe), vol selling, ETH regimes/jumps, and a live Rust BZ/CL ATR-grid bot.

## 2. Empirical process
- **Data**: Bybit V5 (1h klines, 8h funding, trades→1 s buy/sell vol); Tardis free **first-of-month** L2 (3 days) → 250 ms L5; SEC EDGAR (CIK 0001829311); Binance klines (1h 2017→, 1m 2023→) + 30 s bookDepth; Yahoo/IBKR dailies; cTrader 5 m crude; one Deribit IV snapshot.
- **Sample**: BMNR 2,579 hourly bars (~3.5 mo); microstructure 3 days; pairs up to 26 y daily.
- **Methods/validation**: expanding PIT OU, latency_bars=1, side-applied funding, L2-replay liquidity model; purged/embargoed k-fold IC; tick-replay queue fills; t-GARCH, KMeans/GMM, Lee-Mykland jumps; PSR/DSR, Newey-West HAC-t on daily blocks, BH-FDR, project-wide trial counts (up to 10,080).
- **Self-audits** (`results/lookahead_audit.md`, `docs/alpha_leak_audit.md`, `docs/sim2real_review.md`) fixed: 8-K stamps at date 00:00 (→ EDGAR acceptance time), latest share count on history (→ PIT `shares_asof`), DSR n_trials=1, PSR on bars not bets (~29× t), MM Sharpe √(1.26e8), price-blind 1 s fills, microprice marking, full-sample thresholds/VPIN/KMeans, sim calibrated on test days.
- **Missed/under-weighted (my flags)**: (1) **BZ/CL daily "edge" likely artifact** — `backtest/atr_grid_bot.py` builds spread high/low from non-simultaneous cross-extremes (`sh = B_high − β·W_low`), fills limits and take-profits on that synthetic range with favourable intrabar ordering, and uses ATR including the current bar; Yahoo BZ=F/CL=F settle at different times with unaligned rolls. Win rate 99.3% and HAC-t 1.7→25 after a "fix" are red flags. (2) 90-coin universe = current listings (survivorship). (3) Microstructure n = 3 non-random days. (4) `staking_yield_annual` constant 0.032 (not PIT). (5) Yahoo dailies adjusted/revisable, date-stamped not close-stamped.

## 3. Conclusions with numbers
| Result | Value | IS/OOS | Source |
|---|---|---|---|
| mNAV reversion | 4 trades, net SR −0.61, −6.8%, maxDD −17%, half-life 223 h, DSR 0.006 | PIT, no power | `backtest_summary.json` |
| mNAV best of 24 | net SR 0.89, DSR 0.043 | IS select | `sensitivity.json` |
| Book-level arb | 1 cost-clearing dislocation /1,036,799 snaps, net $9.79 | descr. | `arb_backtest.json` |
| BMNR liquidity | spread 8.76 bps, touch $31, book $17–29k/side, 2,134 bps/$M, max clip ≈$4.1k; ETH 0.054 bps, 1.62 bps/$M | measured | `liquidity_report.md` |
| BMNR funding | +2.89 bps/8h mean (+31.6%/yr), **median 0**, p99 40 bps; ETH +3.2%/yr; ex-top-10 spikes +20%/yr | descr. | `cost_report.md` |
| Funding overlay ×5 | best +$109, SR 0.26, DSR 0.020; others −$167…−$724 | IS select | `funding_overlay.json` |
| ETH OBI→fwd ret | IC 0.198@1 s, 0.134@5 s, 0.066@30 s; purged OOS IC 0.131, R² 0.017; BMNR IC ≈0 | **OOS** (3 d) | `signal_study.md` |
| OBI taker | RT cost 11.1 bps vs 0.8–4.8 bps moves | — | same |
| ETH MM tick-accurate | break-even maker fee −1.40 bps; with VPIN brake+A-S −1.21 (adverse sel. −46%); 1 s model ~10× optimistic | 3 d | `market_making.md` |
| Leak-free MM bench | best −0.073 bps/notional, DSR 0.005, 0/8 profitable | 3 d | `strategy_bench.json` |
| Sim→real MM | sim +2.72 bps vs real −0.07…−0.81 | real OOS | `sim2real.json` |
| Binance 30 s depth imb. | OOS IC 0.008–0.024 | OOS | `binance_depth_edge.json` |
| 1-min money-flow/RSI/λ | OOS IC 0.005 | OOS | `return_model.json` |
| ETH 24 h vol classifier | OOS R² 0.439, AUC 0.81; persistence 0.389 → +0.050 | **OOS** | `vol_selling.json` |
| ETH t-GARCH daily | α 0.10, β 0.90, ν 3.2; ACF(1) −0.06, \|r\| 0.17 | IS | `return_dynamics.json` |
| ETH 1m jumps | 3.27/day, Fano 1.36, count ACF 0.27; 74% → vol contraction, median HL 322 min | descr. | `jump_cycle.json` |
| ETH regimes | k=2: calm-bull (vol 66%) vs crash (vol 116%, skew −0.79) | full-sample | `eth_regimes.json` |
| Pairs universe | 147/4,005 BH-FDR<5%; 10,080 configs → best SR 1.26, HAC-t 1.48, **0 survivors** | OOS+deflated | `pairs_enum.json` |
| Basket RV | best 353 bps, DSR 0.95, HAC-t<2 → 0 survivors | OOS 1 y | `basket_rv.json` |
| BZ/CL, XAU/XAG, KO/PEP, SPCE/TSLA | all DSR ≤0.19; BZ/CL 19 y ADF p 0.36; Au/Ag HL 682 d | daily | `cross_strategy_leaderboard.json` |
| BZ/CL intraday (192 cfg) | 158 HAC-t≥2, **0** cointegrated (ADF 0.70–0.80); RW null reproduces it | — | `bzcl_ctrader_sweep.json` |
| BZ/CL daily grid (claimed) | win 99.3%, HAC-t 25, ~22%/yr @5% DD | IS, **suspect** | `bzcl_daily_verdict.json` |
| σ-band martingale | size to 99th pct → −100% (12.8σ tail); size-at-stop maxDD −6.5% | IS | `bzcl_martingale.json` |
| Vol selling | var-swap SR 4.24 **mechanical** (assumed VRP); hedged straddle SR 0.81, HAC-t 2.23, DSR 0.61 | — | `vol_selling.json` |
| Master | 15 families, **0 deployable** | — | `master_strategy_ranking.json` |

## 4. Data inventory (`data/`)
Int64 ns timestamps; kline `ts_ns` = bar **open** (AI_Trader known_ts = close + latency).

| Path | Content | Range / rows | PIT | Reuse |
|---|---|---|---|---|
| `raw/tardis/bybit/incremental_book_L2/{BMNR,ETH}USDT/` | raw L2 deltas, ~50 lvls | 2026-06-01, 07-01, 08-01 | exch+local ts | **Yes**, new L2 table |
| `processed/book_snapshots_2026-0{6,7,8}-01.parquet` | 250 ms L5 BMNR+ETH | 345,600/day | derived | Yes |
| `processed/trades_1s_{ETH,BMNR}USDT.parquet` | 1 s price, buy/sell vol | 05-14→08-28; 6.31M / 210k | aggressor, exch ts | **Yes** → `flow_bars` |
| `processed/funding_{BMNR,ETH}USDT.parquet` | 8h funding | 05-14→08-29; 323 | settlement ts | Yes |
| `processed/{klines_BMNRUSDT_60m,mnav,fair_bands}.parquet` | BMNR 1h, NAV, mNAV, bands | 05-14→08-29; 2,579 | PIT shares; const. yield | `alt_series` (revisable) |
| `processed/fundamentals.json`, `raw/edgar/companyfacts.json` | 40 filings: ETH 3.4M→5.85M, shares 6.2M→603M | 2025-11→2026-08 | acceptance time = good known_ts | **Yes** |
| `processed/binance_klines_ETHUSDT_{1h,1m}.parquet` | Binance OHLCV | 1h 2017-08→2026-07 (78k); 1m 2023→ (1.88M) | native | **Yes** (2nd venue, 1m) |
| `processed/binance_bookdepth_ETHUSDT.parquet` | ±0.2–5% depth, 30 s | 05-20→08-20; 264k | native | Yes |
| `processed/klines_bybit_*_1d.parquet` (90) | Bybit daily alts | 2022-08→2026-08 | survivorship | Low (lake has) |
| `processed/klines_{bybit,binance}_{BZ,CL,XAU,XAG}USDT_60` | TradFi perps 1h | 2026-03/05→08-30 | native | Medium |
| `processed/klines_yahoo_{BZ=F,CL=F,GC=F,SI=F}_1d` | continuous futures | 2000/07→2026-08, ~5–6.5k | roll gaps | `alt_series` revisable |
| `processed/klines_{yahoo,ibkr}_{KO,PEP,TSLA,SPCE}_*`, `klines_ctrader_*`, `ctrader/*.csv` | equities; 2023 crude CFD 5 m | — | adjusted | Low |
| `raw/{bybit_ws,binance_ws}`, `ws_*.parquet` | minutes of WS tests | tiny | local ts | No |

Lake lacks (present here): L2 history, Binance, 1m bars, SEC treasury fundamentals, Yahoo commodity dailies. Absent in both: BMNR NYSE prices (intraday), ETF flows, on-chain staking/treasury data, peer treasuries (MSTR, SBET).

## 5. Reusable components
- `crates/bybit/` — V5 REST/WS client + HMAC signing (copy code only, no key files).
- `crates/execution/{oms,risk,client}.rs` — OMS, leverage resolver, equity kill switch; `trader/src/main.rs` + `deploy/` — live loop on `signal.json`, systemd.
- `crates/grid/tests/golden.rs` + `scripts/gen_grid_golden.py`; `crates/contracts/tests/parity.rs` — Python⇄Rust golden-vector parity pattern.
- `src/eth_bmnr/book/{reconstruct,matching}.py` — Tardis L2 replay + taker matching; `backtest/tick_fill.py` — price-conditioned queue fills.
- `backtest/bench.py`, `evaluation/{statistical_tests,cross_validation}.py` — daily-block PnL, HAC-t, PSR/DSR, purged k-fold.
- `ingest/{edgar,binance_data,bybit_universe,bybit_commodities,deribit,yahoo,ibkr,tardis,ws_capture}.py` — fetchers.
- `signal/calm_classifier.py`, `stats/{jump_diffusion,regime_cluster,return_dynamics,price_impact,pairs_enumerator}.py`, `microstructure/features.py` (VPIN sigma not causal).

## 6. Lessons & pitfalls
1. Count independent bets: 3.5 mo with a 6–9-day half-life ≈ 4 bets — nothing can pass.
2. The illiquid leg decides capacity (BMNR ~$4k/clip).
3. New/stock-perp funding is a spiky carry; harvesting it means holding directional spread risk.
4. Real micro signals (OBI IC 0.2) lie below taker cost and retail maker fees.
5. Coarse fills ~10× optimistic; honest fills are adverse-selected; default back-of-queue.
6. Sims calibrated on test data flatter makers ~3 bps.
7. Harness bugs flipped verdicts (fees on spread price, ~250× under-charge; phantom first-day return deflating HAC-t) — golden-test the harness.
8. HAC-t≥2 is not sufficient: add cointegration, random-walk null, era stability.
9. Size to the stop/tail, not the 99th percentile (12.8σ tail wiped out the account).
10. Bar-based limit fills need simultaneous prices; cross-extreme spread OHLC fabricates fills.

## 7. Candidate hypotheses for AI_Trader
- **H-EB1 Stock-perp funding skew.** Bybit `symbol_type=stock` perps show significantly positive mean funding with median ≈0, above crypto perps of similar OI. *Evidence*: BMNR +31.6%/yr, robust ex-spikes (IS, 1 symbol). *Test*: lake `funding` × `instruments.symbol_type`, bootstrap CI, clean holdout. *Prior 0.6.*
- **H-EB2 Elevated-funding hedged short.** Shorting a stock perp (hedged) only when trailing funding >p80 nets carry > spread loss + costs. *Evidence*: only positive overlay, DSR 0.02 (IS). *Test*: 1h bars + funding, PIT thresholds, liquidity-capped size. *Prior 0.2.*
- **H-EB3 Post-jump vol contraction.** After a >5σ 15m/1h jump, RV decays (HL ~5 h) and the next 4–24 h favour mean-reversion/short-vol cells. *Evidence*: 74% contraction, but jumps self-excite (descriptive, Binance 1m). *Test*: causal detector on lake bars vs `shock_study` priors. *Prior 0.5.*
- **H-EB4 Persistence dominates vol regimes.** HAR/GARCH-t forward vol captures ≥85% of the OOS R² of any feature-rich calm classifier. *Evidence*: 0.439 vs 0.389 (OOS). *Test*: purged CV vs `models/vol.garch_t`. *Prior 0.8.*
- **H-EB5 OBI-timed passive entries.** Posting entries only on the OBI-favoured side cuts 5–60 s markout adverse selection ≥30%. *Evidence*: −46% adverse selection (3 days). *Test*: import Tardis days + start L5 recorder; per-fill markouts. *Prior 0.45.*
- **H-EB6 Stock-perp off-hours premium.** Bybit stock perps drift from the last US cash close over weekends/overnight and partly revert at the open. *Evidence*: untested here; structural 24/7-vs-closed mismatch. *Test*: import underlying NYSE dailies/intraday as `alt_series`; gap-at-open regression. *Prior 0.45.*
