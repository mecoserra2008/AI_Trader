# Specification program (launched 2026-11-27)

User request (2026-11-26 ~23:20 UTC): many agents on different areas; every modelling and hypothesis decision justified by the
variable being modelled; for each variable, why this specification, which alternatives, how their conclusions differ, which
to trust, tested properly; foundation and machine-learning models with deep mechanistic interpretability; audit what has and
has not been done; improve the hypotheses; test everything; controlled use of RAM and storage; up to 20 agents at once.

State of knowledge at launch (read the reports): short-horizon direction does not pay after costs (geometry, participants,
TVTP, MM weekend); volatility and covariance are forecastable (intraday clock, release minutes, TVTP covariance, HAR family);
only the 4h trend family survives stress tests (zoo_ensemble SURVIVES, trend_4h FRAGILE-positive); mean-reversion, calendar,
shock fades and setup C fail; the macro models are weak beyond the CPI/PPI bridge nowcast. Registry ~336 hypotheses.

Every agent follows research/programs/AGENT_RULES.md (safety, governor, storage, locks, dossier, protocol, report).

## Workstreams (id, quota, scope)

WS01 AUDIT (100 MB). Method audit of the whole registry and results: per family, the target variable, the specification
used, its distributional/dependence assumptions, the test used and whether it was valid (overlap, HAC, look-ahead,
multiple testing, power). Registry-wide multiple-testing control (Benjamini-Hochberg / Benjamini-Yekutieli over all
recorded p-values or CIs; which "supports" survive). Output: a research-debt ledger (what is untested, underpowered,
mis-specified) ranked by decision value, handed to WS19.

WS02 DATA (1 GB). Data quality and point-in-time audit of the lake and research inputs: coverage gaps (PC suspends,
2026-11-20/21/23/24/26), duplicates, timestamp alignment (known_ts), survivorship (delistings), vintages, sampling
frequency limits; data cards per dataset; fill gaps from public archives where possible. Output: data catalogue + a
gap map every other workstream can use.

WS03 DIST (300 MB). Marginal return distributions by asset (BTC, ETH, liquid alts, XAU/XAG), horizon (5m..1d) and regime:
Normal, Student-t, skew-t (Hansen, Fernandez-Steel), GED, NIG / generalised hyperbolic, alpha-stable, mixture models;
EVT tails (POT/GPD, Hill with threshold selection), tail-index stability; standardised by each vol model. Decision link:
stop distances (the 4 sigma cap prior), VaR/ES of the live risk unit, the Student-t house default.

WS04 VOL (300 MB). Volatility specifications: GARCH, GJR, EGARCH, APARCH, FIGARCH, component GARCH, Realized GARCH, HAR /
log-HAR / HARQ / HAR-J, SV, rough-vol forecasts, regime-switching GARCH; innovations Normal vs t vs skew-t; realised
measures (RV, bipower, realised kernel); losses QLIKE and MSE; Model Confidence Set across 1h/4h/1d horizons and assets;
weekend / session effects (link the intraday clock). Decision link: sigma_risk used by stops and the EV gate.

WS05 JUMPS (300 MB). Jumps and clustering: BNS / Lee-Mykland / Ait-Sahalia-Jacod tests, jump-diffusion vs pure diffusion,
Hawkes self-exciting intensity (univariate and cross-asset, with liquidation and release-minute marks), forecast of jump
probability next hour; validity of the field P(jump) used in the desk. Decision link: blackouts, stop gaps, R-FLOW-2.

WS06 DEPEND (300 MB). Dependence: rolling vs EWMA vs DCC / ADCC / cDCC, factor (PCA) structure of crypto perps, copulas
(Gaussian, t, Clayton, Gumbel, vine), tail dependence, regime-dependent correlation, the "one bet" problem quantified
(effective number of bets, co-stopping probability). Decision link: open-slot / same-side limits, cluster caps (R-PD-1).

WS07 REGIME (300 MB). Regime specifications compared on the same data: v0 heuristic cells, Gaussian and t HMMs, MS-AR /
MS-GARCH, TVTP (src/ai_trader/research/tvtp_msvar.py), change-point (PELT, BOCPD), clustering of state features. Agreement
(adjusted Rand), persistence, stability across refits (label alignment), and USEFULNESS (does conditioning on each improve
held-out vol / return / strategy outcomes vs unconditional?). Decision link: the cell gates used by rules.

WS08 PREDICT (300 MB). Return predictability with correct inference: predictive regressions with persistent predictors
(funding, basis, OI change, ETF flows, stablecoin supply, DVOL, skew, Coinbase premium, macro), Stambaugh bias, IVX
(Kostakis-Magdalinos-Stamatogiannis), bootstrap; horizons 1h..30d with overlapping-observation corrections; Clark-West /
Giacomini-White for nested out-of-sample comparisons; multiple-testing across predictors (Romano-Wolf). Decision link:
whether any directional predictor deserves a strategy.

WS09 XSECT (300 MB). Cross-sectional pricing across Bybit perps: momentum, reversal, size, liquidity, beta, idio-vol,
funding carry, basis, OI growth; portfolio sorts and Fama-MacBeth with Shanken correction, GRS, net of costs and turnover,
capacity at our size; survivorship (delistings). Decision link: alt selection inside the trend sleeve, universe choice.

WS10 MICRO (1 GB). Microstructure model specifications: price impact (Kyle lambda, square-root law, propagator / transient
impact), Hasbrouck information share and Gonzalo-Granger across venues, VPIN / PIN validity, queue-reactive model, spread
decomposition (Huang-Stoll / Roll); calibrate on the lake (flow_3s, trades, ob200 days) and the live fills. Decision
link: the cost model (WS17) and passive vs aggressive execution.

WS11 FOUNDATION (1.5 GB incl. weights). Time-series foundation models on CPU within RAM: Chronos / Chronos-Bolt (tiny,
mini, small), TimesFM (smallest), Moirai-small, Lag-Llama, Toto if small enough; zero-shot vs light fine-tuning; targets
log returns (quantiles) and realised vol at 1h / 4h / 1d; strict contamination control (evaluate only after each model's
training-data cutoff; report the cutoff); compare with HAR / GARCH-t / empirical quantiles on proper scores and DM / MCS.
Decision link: whether any foundation model earns a place in the live network.

WS12 DEEP (500 MB). Deep sequence models trained here (LSTM, GRU, TCN, small Transformer, N-BEATS, TFT-lite, DeepAR-style)
with distributional heads (quantile, mixture density, Student-t), honest purged walk-forward, early stopping on validation
only, seeds x 5, calibration; vs the best classical spec from WS04 / WS03. Overfitting diagnostics (PBO, deflated Sharpe,
learning curves). Hand the trained models to WS14.

WS13 TREES (300 MB). Gradient boosting (LightGBM), random forests, regularised linear (ridge / lasso / elastic net) on
engineered point-in-time features for vol, direction (as a sanity check), and trade outcome (meta-labelling of trend
trades); purged CV; SHAP with stability across folds and seeds; monotone constraints where theory implies them. Hand the
models to WS14.

WS14 INTERP (300 MB). Mechanistic interpretability of the networks: the live PNN (v1, v2.1), xs_v3 / xs_v4, and the models
from WS11-WS13 as they land. Probing (linear probes for vol, regime, flow), activation patching / causal tracing,
ablations of units and inputs, attribution with sanity checks (integrated gradients vs random-weight baseline), monotonicity
and invariance tests, distillation into a simple surrogate (is the network a scaled HAR?), failure modes by regime.
Decision link: whether we understand and can trust what the live network does.

WS15 GENERATIVE (500 MB). Synthetic market generators judged on stylised facts (fat tails, vol clustering, leverage,
aggregational Gaussianity, cross-correlation, jump clustering, intraday seasonality): stationary block bootstrap, GARCH-t /
HAR sim, regime-switching sim, Hawkes-driven sim, the agent-based ob_gen (src/ai_trader/sim), a small diffusion or
TimeGAN-lite if CPU-feasible. Output: a validated generator library the stress harness (src/ai_trader/research/stress.py)
can call.

WS16 TREND (300 MB). The surviving family in depth: trend signal definitions (TSMOM sign / scaled, EMA stacks, Donchian,
regression slope, ensemble weights), lookbacks, vol targeting vs fixed risk, regime filters, entry timing (intraday vol
clock), exits (signal flat, trailing, time, chandelier), stop placement relative to prior-day / week extremes (R-GEO
suggestions), capacity; all through the stress harness (stress_signal) with SURVIVES / FRAGILE / FAILS rules; a forward
paper plan for the best one or two.

WS17 COSTS (300 MB). The cost model: fees by tier, spread by hour (GEO map), impact (from WS10), funding, stop slippage on
gaps, maker fill probability for pegged entries and its adverse selection; calibrate on our live fills, paper fills and
replays; a distribution, not a point. Re-price the main strategies with it. Decision link: every net-R number we report.

WS18 MACRO (500 MB). Cross-asset and macro: BTC / alts vs real yields, dollar, equities (NDX), credit spreads with
time-varying betas (rolling, Kalman, DCC); options-implied information (DVOL level / changes, 25-delta skew, term
structure) as predictors of realised vol and drawdowns; TradFi perps (gold, silver, indices) session structure; the
2018 / 2008 analog work extended with proper inference. Decision link: risk-on / risk-off gates.

WS19 REFINE (300 MB). Research debt: take the inconclusive and underpowered results (start from WS01's ledger as it lands,
and the reports since 2026-11-14), compute power / minimum detectable effects, redesign the tests that could matter
(more data, better specification, sharper hypotheses), pre-register the improved versions on data never used for them,
and run the ones that are cheap. Never re-use a holdout that was already looked at.

WS20 REDTEAM (100 MB). Adversarial review: as other workstreams commit reports, audit them for look-ahead, leakage,
overlapping-observation inference, multiple-testing blind spots, selection after seeing results, cost omissions, and
claims that exceed the evidence; re-run key numbers independently where cheap. Also review the last two weeks' reports
(geometry, stress, participants, TVTP, macro, MM weekend). Output: findings per report with severity; the orchestrator
decides what is withdrawn.

## Orchestration
- Up to 20 agents at once; heavy compute is serialised by the governor (4 heavy slots, 8 light, RAM headroom 1.5 GB).
- Agents do not wait on each other unless stated (WS14 picks up WS11-13 models as they land; WS19 picks up WS01's ledger;
  WS20 reviews as reports land; WS17 uses WS10 when available).
- The orchestrator (main session) keeps the desk running, watches storage and the governor log, relays decisions to the
  user, and writes the synthesis when the workstreams report.

## Compaction (2026-11-26 ~23:40 UTC, user: "compact the 20 agents into 10 ... until 3:10 am")
Ten agents were stopped and their scopes folded, trimmed, into the survivors; all agents work lean (highest decision-value
questions first), commit a partial report by 02:45 UTC, and use private scratch folders.
- WS01 AUDIT + REFINE (top ~5 debts: power, redesign, cheapest re-tests) + REDTEAM (stress, geometry, participants, TVTP reports)
- WS02 DATA (unchanged; gap map and PIT checks first)
- WS04 VOL + DIST (innovation distributions, Student-t df, 4-sigma cap probability, VaR/ES of the risk unit)
- WS05 JUMPS + DEPEND ("tail risk": co-jumps, tail dependence, DCC vs EWMA, effective bets, co-stopping, live limits)
- WS07 REGIME (unchanged; usefulness of filtered regimes first)
- WS08 PREDICT + XSECT (one cross-sectional test, alt selection in the trend sleeve) + MACRO (time-varying betas, DVOL/skew)
- WS10 MICRO + COSTS (cost distribution validated on real fills; re-price zoo_ensemble, trend_4h, H-RB2)
- WS12 DEEP + FOUNDATION (Chronos-Bolt zero-shot after its cutoff) + TREES (LightGBM meta-labelling of trend trades)
- WS14 INTERP (live PNN surrogate and invariance tests first)
- WS16 TREND + GENERATIVE (correct block bootstrap + one parametric generator for the synthetic stress)
Stopped: WS03, WS06, WS09, WS11, WS13, WS15, WS17, WS18, WS19, WS20 (they had only read files; no jobs left running).
