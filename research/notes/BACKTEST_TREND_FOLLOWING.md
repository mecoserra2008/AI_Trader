# Prior project review: BACKTEST_TREND_FOLLOWING ("GTF")

Source: `C:\Users\Asus\Documents\BACKTEST_TREND_FOLLOWING` (read-only, reviewed 2026-11-07). Key write-ups: `paper/gtf_paper.tex`, `results/research_journal.jsonl`, `results/trend_drawdown_report.md` (the most honest one), `results/btc_options_SYNTHESIS.md`.

## 1. Purpose & thesis
- **GTF = Geometric Trend Following.** A bond-futures spec ported to Bybit BTC/ETH/SOL perps. Directional-change pivots (δ = K·σ_EWMA) feed STRUCT, PIVOT_BO and FLAG_BO signals, as an ensemble over K∈{3..12}. Benchmarks: TSMOM(30d) and Donchian(20d). Inference: White RC, SPA_c, NW.
- **Engine v2 (`gtf_fast` + Rust).** Next-bar-open fills, stop-first intrabar, 0.5% risk per trade. Regime switch between trend (chandelier trail), BOS breakout (fixed RR) and range fade (lag-1 AC < 0, ER < 0.4). Run as 9 cells (3 symbols × 1h/4h/1d).
- **Alt-data (`ALTDATA_DESIGN.md`).** Causal gates on funding z, OI and Fear&Greed, plus funding charged as P&L. A GDELT fetcher exists, but **no GDELT data is on disk**.
- **Options and other sleeves.** First a DVOL-anchored condor sleeve, later a pre-registered Deribit-history options programme (SVI). Also XS-momentum, CVaR sizing, a CLOB simulator and a Python→Rust live engine.

## 2. Empirical process
- **Data.** Bybit klines. 1d from 2022-05; **4h covers only 730d and 1h only 365d** (the paper's "4.4 yr" claim is wrong). The 50-coin universe is the *current* top-50 by turnover, so it is survivorship-biased with no delisted coins.
- **Validation.** 60/40 split per cell, objective min(IS,OOS). **The OOS half was used in selection every round, so there is no holdout.**
- **Search size.** 5 workflow versions and 13 synthesis rounds. Each round = 3 LLM agents × 5–10 experiments, plus ablations and fine sweeps. **That is ≈300–500 configs**, plus 30 faithful cells and 50 coins. Final knobs are knife-edge (chand_mult 4.12; fng_greed 74). **Snooping risk is severe.**
- **Costs.** 7.5 bp/side + 2 bp slippage; funding was added later. Option fees were swept 3–12.5% × spread 1–5× (`cost_audit.py`).
- **Bias risks:**
  - `fetch_klines` keeps the still-forming bar, so live signals use partial bars.
  - Live vol target 0.25 vs 0.12 in the backtest; TREND_K was calibrated on the last 12 months.
  - A $250 per-instrument cap silently clipped 10.6% of target hours.
  - Later studies (options, drawdown) were pre-registered with DSR.

## 3. Conclusions with numbers
| Result | Label | Value |
|---|---|---|
| Faithful 1m (45d) | IS | STRUCT Sharpe −28.7, TSMOM −39.4 |
| STRUCT/TSMOM/TRB by TF | IS, 30 cells | 1h 1.55/0.56/1.07 (1y only); 4h −0.10/1.00/0.93; 1d 0.58/0.97/0.62. No HAC alpha vs slow trend |
| Robust champion | snooped | IS 0.82 / "OOS" 0.85, PF 1.76, 98 trades. Without alt gates: 0.61. Funding and OI gates alone were falsified |
| **Exact live rule replayed hourly** | **clean OOS 2020-01→2022-04** | **Sharpe −0.12**; IS 0.30; FULL 0.17; maxDD −47%; last 12m 1.69 (the calibration window) |
| Attribution, FULL | diagnostic | 1h cells = 74% of variance, ~0 net; 1h BOS −27%; costs 7–10%/yr; turnover ~90×/yr |
| By TF, OOS/IS | post-hoc | 1h −0.09/0.03; 4h −0.31/0.78; 1d 0.27/−0.17 |
| 7 pre-registered DD fixes | OOS, N=8 | None passed. Vol targeting +0.11 (P=.78); cov cap +0.10; resting stops **−0.24 (P=.01)** |
| IC vs next-10d book return | FULL | rv20 −0.19 (p=.003); cell flips −0.14; ER −0.14. Funding/F&G not significant |
| Champion on 50 coins | OOS, survivors | 54% profitable, median 0.13; EW Sharpe 0.40 |
| XS-mom 4h | CLOB MC | Frictionless 1.18 → taker median −1.48 (0% of paths > 0); liquid-only 0.02 |
| Deployable book | synthetic MC | 3.15 frictionless → 0.78 median (p5 −0.45). Superseded by the drawdown study |
| F&G quintiles, fwd 20d | IS | Not contrarian (fear is worst); HAC t ≤ 1.3 |
| Alt index by BTC 1d regime | IS | Trend-up: +9.0%/20d (t 2.3), +4.5% vs BTC (t 1.8) |
| BTC options | pre-registered OOS 2023→ | Long vol, puts, naked strangles, jump gating and surface arbitrage all fail. Hedged 30D 25Δ strangle: Sharpe 0.87 [−0.18, 1.88], ~2%/yr, DSR 0.57 (N=17), shrinking. Funding carry ≈0 since 2025 |
| Friday-sold weekly | pre-registered | BTC +15.7 bp/wk (t 3.1), ETH +23.2 (t 3.6); IV −2.5 vol pts Fri→Sat |
| Live | real | ≈$1.2k account, one smoke order, otherwise dry-run. **No track record** |

## 4. Data inventory (`BACKTEST_TREND_FOLLOWING/data/`)
| Path | Content | Range / size | PIT | Value for AI_Trader |
|---|---|---|---|---|
| `btc_options/`, `eth_options/raw_trades/` | Deribit option trades sampled before 08:00 UTC daily | 2020-01→2026-10; 63/46 MB | native | **High**: no options history in AI_Trader |
| `*_options/surface_daily_svi_v2.csv`, `surface_expiries_svi_v2.csv.gz` | Daily ATM/25Δ/10Δ IV at 7/14/30D, SVI | 2,474 days | causal | **High**: skew, term structure |
| `btc_options/{dvol_1h,deribit_funding_1h,deribit_perp_1h}.csv`, `futures_1h/` (78 expiries) | DVOL 1h, Deribit funding/perp, dated futures | 2019-06/2020→ | native | **High**: second venue, basis |
| `btc_jump_regime/bybit_btcusdt_5m.csv` | Bybit BTC 5m OHLCV | 2020-04→2026-09; 683k rows, no gaps | native | **High** |
| `btc_jump_regime/{daily_decomposition,lm_jump_events,merton_fits}.csv`, `deribit_opt/` | RV/BV/jumps, jump events, ~4.7M option trades | 2021→2026-09 | causal | **High**: jump-regime labels |
| `btc_options/{binance_spot_5m.csv.gz,rv_daily,rf_dtb3}.csv` | Spot 5m, RV, T-bill | 2020→ | native / revisable | Medium |
| `options_hist/` | Deribit USDC option trades (SOL/XRP 2024-08→, BTC/ETH 2025-08→, HYPE…), surfaces | 325 MB | native | Medium: alt skew |
| `bybit_chains/` | Bybit chain snapshots and BTC option tape | 9–13 days (2026-09/10) | native | Low; collector reusable |
| `trend_long/`, `trend_dd_long/` | BTC/ETH/SOL with Binance spot spliced before perp listing | 2020→ | mixed venue | Low |
| `{SYM}_{1d,4h,1h}.csv`, `altdata/*_funding.csv` (50 syms) | Bybit bars, 8h funding | 1d 2022-05→, 1h 1y | last bar may be partial | Low–medium: ~27 extra alts (survivors) |
| `altdata/fng.csv`, OI, DVOL 1d, 1m/15m | — | small | — | Low (already in AI_Trader) |
- **Missing:** GDELT, delisted coins, other perp venues.
- **Do not import:** `results/live_snapshot.json` and `source_of_truth.html` (account/API metadata).
- **Mapping:** surfaces and DVOL go to `alt_series` (pit_quality = native); option trades and chains go to `snapshots`/trades.

## 5. Reusable code
- `gtf/pivots.py`: EWMA σ plus the directional-change pivot machine, with confirmation timestamps (structural TP/SL).
- `gtf/inference.py`, `research/trend_drawdown/stats.py`: RC, SPA_c, NW, stationary bootstrap, PSR/DSR, paired ΔSharpe.
- `research/trend_drawdown/{livebook,parity_check}.py`: hourly replay of the live rule; parity harness.
- `gtf_fast/rustcore/` (PyO3 kernels, `execsim.rs`/`clob.rs`): fast backtester and CLOB taker/maker simulator.
- `altdata/features.py` (`align_series_causal`), `altdata/gdelt.py`/`gdelt_fetch.py`: causal alignment; GDELT fetcher.
- `portfolio/{bootstrap,allocation,tail_sizing}.py`: block MC, risk parity, CVaR tail sizing.
- `research/btc_options/{harness/,fetch_data.py}`: SVI surface, Bybit fees, WW hedger; resumable Deribit pulls.
- `research/bybit_chain_snapshotter.py`: forward option-chain and tape collector.
- `live_engine/src/{risk,oms,reconcile}.rs`, `bybit/*.rs`, `deploy/`: Rust Bybit v5 OMS, 3-switch gate, KILL file, loss breaker, 92 tests, systemd units.
- `build_dashboard.py`: offline HTML dashboard.

## 6. Lessons & pitfalls
1. An agent search under min(IS,OOS) produced balanced 0.8 Sharpes; clean OOS was −0.12. A reused OOS is IS.
2. The 1h/4h cells were "validated" on 1–2 years. Equal capital ≠ equal risk: tight 1h stops created 74% of the variance and most of the costs.
3. 1h breakouts fail in both directions in crashes and whipsaw; range fades got run over in crashes.
4. Resting stops lose to close-based exits, because touches revert within the hour.
5. Realistic execution cut Sharpe ~4×. Alt XS-momentum only pays as a maker.
6. The static smile overstated short-dated ATM IV by ~3 vol pts, which created a fake 1.4-Sharpe short.
7. VRP and carry are shrinking since 2024. Strong numbers sat in calibration windows.

## 7. Candidate hypotheses for AI_Trader
1. **Slow trend survives costs; 1h structure does not.** 1d/4h TSMOM/Donchian has net Sharpe > 0.3, while 1h structural entries ≈ 0 after 9.5 bp.
   - *Evidence:* TSMOM 1d 0.97 (single pass); 1h ≈ 0 over 6.7 years.
   - *Test:* AI_Trader bars 2020→, matched vol, holdout after 2026-07-01.
   - *Prior:* 0.40.
2. **1h BOS breakouts have negative net expectancy in range/high-flip regimes.**
   - *Evidence:* −27% over 6.7 years; "no-BOS" OOS +0.42 is post-hoc.
   - *Test:* per-cell expectancy by regime.
   - *Prior:* 0.60.
3. **High trailing 20d RV predicts lower trend returns.**
   - *Evidence:* IC −0.19 (partly-IS diagnosis).
   - *Test:* non-overlapping IC vs forward 10d cell P&L.
   - *Prior:* 0.45.
4. **1h structural-stop touches revert within the hour.** A close-confirmed stop beats a resting stop.
   - *Evidence:* pre-registered, ΔSharpe −0.24 for resting stops.
   - *Test:* AI_Trader's 180d flow bars. Post-touch return conditional on flow imbalance.
   - *Prior:* 0.50.
5. **Short-dated BTC/ETH IV is marked down over weekends and weekend RV < Friday IV.** This supports H-W1.
   - *Evidence:* pre-registered, t 3.1–3.6.
   - *Test:* imported Deribit surfaces/DVOL.
   - *Prior:* 0.65.
6. **F&G and funding extremes add no out-of-sample value as entry gates.** This is a null to confirm first.
   - *Evidence:* the gate "wins" came from the ~300–500-config snooped search; F&G quintile |t| ≤ 1.3.
   - *Test:* holdout ablation.
   - *Prior that the gates help:* 0.15.
