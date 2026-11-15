# Program 2 (2026-11-11): strategies by regime, TradFi session effects, PnL decomposition, a calibrated execution simulator, robustness through generated data

Follows [2026-11-10_volatility_program.md](2026-11-10_volatility_program.md), whose results are now in
[BELIEFS.md](../BELIEFS.md) §6 and [rules.yaml](../rules.yaml) (53 rules). 22 hypotheses were registered before testing:
H-RC1-2, H-TF1-5 (plus the untested H-TP1), H-PD1-4, H-NS1-5, H-SIM1-4 and H-SYN1-2.

## What we learned that shapes this program

Volatility is forecastable, and we now know the best models (log-HAR, panel HAR). Direction is not forecastable from
standard features, from vol signals, from pivots, from indicators like Market Cipher, or from order-flow z-scores. Edge
is therefore most likely in:
1. **conditioning**: which strategy, in which state;
2. **structural flows and calendars**: session gaps, funding settlement, expiries;
3. **execution**: maker vs taker, exits.

We also need to know what each strategy really is, i.e. which factor it rides. Program 2 attacks those four directly,
builds the tools to simulate execution realistically, and uses generated data to make parameter choices robust.

## Workstreams

| agent | hypotheses | builds | key data |
|---|---|---|---|
| B1 strategy map and PnL decomposition | H-RC1, H-RC2, H-PD1-4 | `research/strategy_panel.py`, the daily net P&L panel of every registered strategy plus the zoo ensemble (`data/priors/strategy_panel_daily.parquet`); `research/pnl_decomposition_study.py` (PCA / effective bets, factor attribution, IPCA conditional ranking, trade-level decomposition) | bars L and T; regimes v0 |
| B2 TradFi sessions | H-TP1, H-TF1-5 | `research/tradfi_session_study.py`, an NYSE / CME session calendar with DST | 1h bars of TradFi perps (4h built from 1h where native 4h is missing) |
| B3 new strategies | H-NS1-5 | `research/new_strategies_study.py`, `features/seasonality.py` | 1h bars, funding, Deribit expiry calendar |
| B4 simulator | H-SIM1-4 | `sim/fill_model.py`, `sim/reaction.py` (trader reaction functions), `sim/flow_sim.py` (generator), `sim/replay.py` (quoter replay on real 3s paths), `research/simulator_study.py` | flow_3s, liquidations, our live mainnet fills (`data/mm/analysis/*`) |
| B5 synthetic robustness | H-SYN1-2 | `sim/resample.py` (stationary block bootstrap, regime-switching resampling), `research/synthetic_robustness_study.py` | 1h bars L |
| B6 literature | none (proposals) | `docs/paper_notes/*.md`, `research/reports/literature_2026-11-11.md` | papers (web / arXiv) |

## Methods

**PnL decomposition, from several angles** (B1, informed by B6):
1. **Statistical factors.** PCA on the strategy-return correlation matrix gives the effective number of bets and the
   loadings.
2. **Observable factors.** Returns-based style analysis (Sharpe 1992; Fung-Hsieh 2001) on crypto market, TSMOM, vol
   change, carry, liquidity, QQQ, gold and oil.
3. **Conditional factors.** IPCA (Kelly, Pruitt & Su 2019): loadings are a linear function of context
   characteristics, so the model directly produces "which strategy type pays in this context". That is the ranking
   the user asked for.
4. **Trade-level attribution.** Direction, favourable-excursion capture, exit leakage and costs (MFE/MAE).
5. **Families.** Hierarchical clustering on factor loadings, so rules are stated per strategy family, not per
   strategy.

**Simulator** (B4):
1. **Fill model for passive quotes.** A quote fills when the 3s candle trades through it, times a queue-survival
   factor. It is validated on our ~250 live fills, trained on the first half in time and tested on the second.
2. **Reaction functions.** Regressions / VAR of 3s signed taker flow on lagged returns (momentum vs contrarian
   traders), book imbalance (liquidity-takers reacting to the book), spread and vol. Flow is decomposed into
   components with stable signs, which act as the implicit trader types.
3. **Generator.** Marked Hawkes or VAR flow with heavy-tailed innovations, plus an impact function and spread
   dynamics. It is fitted on days 1-2 and validated on day 3 against stylised facts.
4. **Transfer to other markets.** Scale by σ, depth and median trade size so one calibration can drive another market
   with matched quantities, volatility and liquidity.
5. **Replay.** The `atx mm` quoting rules (`target_quotes`), ported to Python faithfully, run on real 3s paths and on
   generated paths. Parameter rankings are compared between the two.

**Generated data** (B5): it is used only to regularise choices for deterministic strategies (principle 1: the LLM
layer is never trained or backtested). The resampling preserves vol clustering and regimes, and the test is whether it
reduces the in-sample to holdout degradation.

## Limits stated up front

- Order flow covers 3 days with collector gaps, so the simulator is a v0 and every SIM result is holdout-only.
- TradFi perps have about 5 months of history, so the TF results are mostly holdout.
- Our live fills number about 250, from one market state.
- A better simulator needs longer L2 data. Collecting it (raw trades and book deltas, storage budget 30 GB) is the
  user's decision.

## Outputs to the decision layer

Every agent ends its report with rule candidates in the rules.yaml schema. The orchestrator merges them, updates
BELIEFS.md, and adds a context-ranking table: strategy family × regime/context → expected sign and grade, read from
B1. The market-state card (tools/market_state.py) gives the context; the rules give the expectations.
