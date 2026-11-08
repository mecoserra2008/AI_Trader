# AI_Trader — Architecture

_v0.2, 2026-11-07. Companion to [PLAN.md](PLAN.md) (goals, evaluation, roadmap). This document covers **how the system works**: statistical conventions, the belief state and its priors, the tool catalogue, tool selection, and lean compute/storage._

---

## 0. The one-line architecture

> **Deterministic quant layer builds a calibrated, point-in-time *belief state* (priors → posteriors) for every asset. The LLM reads that state, drills into tools when needed, and expresses bounded *views* that a deterministic combiner merges into the final posterior. The risk engine turns the posterior into sized trades.**

```
 raw data ─► bars/aggregates ─► FEATURES (stationary, vol-standardised)
                                     │
             ┌───────────────────────┼─────────────────────────────┐
             ▼                       ▼                             ▼
      STATE / LABEL tools     PREDICTIVE tools              RELATIONAL tools
      regime, vol, trend,     P(r>k), quantiles,            clusters, factors,
      timing, S/R, flow,      triple-barrier P(TP≻SL),      betas, cointegration,
      sentiment               event hazards, vol fcst       cross-sectional rank
             └───────────────────────┼─────────────────────────────┘
                                     ▼
                       BELIEF STATE per asset (priors → posteriors)
                       hierarchical: global → cluster → asset
                                     │
                     compact STATE CARDS (≈150 tokens/asset)
                                     ▼
              LLM: read cards → optional drill-down tool calls (budgeted)
                   → bounded VIEWS (log-odds adjustments + reasons)
                                     ▼
              COMBINER (deterministic) → final P, E[R], uncertainty
                                     ▼
              RISK ENGINE → rank, filter, size (fixed R), ≤ N trades/day
```

Everything above the LLM box can be backtested with walk-forward validation, and **must** be before it ships. The LLM box is evaluated forward only, as the *increment* of the final posterior over the quant-only posterior.

---

## 1. Statistical house rules

These apply to every tool. A tool that breaks one doesn't ship.

### 1.1 Returns and standardisation
- Work in **log returns** `r_t = ln(P_t / P_{t-1})`.
- **Vol-standardise everything:** `z_t = r_t / σ̂_{t|t-1}`, where `σ̂` is the one-step-ahead conditional vol (EWMA or GARCH, §3.1). This takes care of volatility clustering at the feature level. It also makes assets comparable cross-sectionally and pushes features toward stationarity.
- Horizons are expressed in **bars** and in **vol units**, e.g. "+1.5σ within 5 days", not "+3%".

### 1.2 Fat tails: Student-t by default
- Default error distribution is **Student-t** with ν estimated per asset (MLE on standardised residuals), floored at ν ≥ 3 and shrunk toward the cluster ν.
- Applies to: GARCH innovations, HMM emissions, regression errors (t-likelihood, or Huber loss as the cheap robust stand-in), and every tail probability reported to the LLM.
- Estimators default to robust versions: median/MAD scale, Huber regression, Ledoit–Wolf shrunk covariances.

### 1.3 Stationarity
- Every feature entering a model must pass **ADF (reject unit root) and KPSS (fail to reject stationarity)** on its training window. Using both tests guards against the weak power of either one alone.
- If it fails, transform in this order: (1) difference or ratio, (2) rolling z-score, (3) cross-sectional rank, (4) **fractional differencing** with the minimal d that passes ADF (keeps level memory for price-like features such as distance to S/R).
- **Drift monitoring:** Population Stability Index (PSI) per feature, train window vs last 20 bars. PSI > 0.25 flags the feature as stale in the state card.

### 1.4 Volatility clustering and nonlinearity, without black boxes
Nonlinearity enters only through these channels, from simplest to most complex:
1. **Conditional variance** (GARCH/EWMA): the dominant nonlinearity in returns.
2. **Regime conditioning:** coefficients interact with regime posteriors, `β(s) = Σ_k p(s=k)·β_k`. This is piecewise-linear and interpretable.
3. **Thresholds** (TAR-style): e.g. a different slope when |z| > 2.
4. **Quantile regression:** asymmetric and tail behaviour without distributional assumptions.
5. **Shallow GBM** (depth ≤ 3, monotone constraints) as a *challenger only*. It's admitted only if it beats the linear/quantile baseline out of sample by a pre-registered margin (e.g. ΔBrier ≥ 1% across purged walk-forward folds).

### 1.5 Inference and validation
- Overlapping horizon labels → **Newey–West / HAC** standard errors.
- Panels → **double-clustered** standard errors (by date and asset).
- Validation is **walk-forward only**, with **purging + embargo** (López de Prado) whenever labels span multiple bars.
- Every probability output is **calibrated** (Platt or isotonic on a held-out fold) and reports Brier, log-loss and a reliability summary. Uncalibrated probabilities never reach the LLM.
- **Multiple testing:** every feature or model variant tried goes into `research/trials.jsonl`. Selection uses the deflated Sharpe ratio and Holm/BH adjustments.

### 1.6 Complexity budget
Default model per tool: **OLS/Huber/logistic** with ≤ 8 features. A tool may add complexity only when it demonstrates out-of-sample improvement over its own simpler version. The trial log records the comparison.

---

## 2. The belief state: priors and updating

This is the core of the architecture. For every asset `i`, the system keeps a small, structured **belief state** `B_i(t)`, a set of probability distributions rather than point estimates:

| Component | Representation | Prior source | Update mechanism | Update cadence |
|---|---|---|---|---|
| Regime | **9-cell posterior** `p(d_t, v_t)`: direction {up, range, down} × volatility {low, normal, high} (§3.2) | factorial HMM / jump-model fit, Dirichlet priors on both 3×3 transition matrices | **forward filter** (2 × O(3²) per bar) | every bar |
| Volatility | `σ̂_{t+1}`, `σ̂_{t+h}` + ν | GARCH(1,1)-t params | GARCH recursion | every bar |
| Trend | slope μ_t, its s.e. | local-linear-trend state space | **Kalman filter** | every bar |
| Return distribution | location, scale, ν per horizon | regression/quantile fit | **conjugate update** (Normal–Inverse-Gamma with forgetting factor λ) | daily |
| Event probabilities | `P(event within h)` | logistic/hazard fit | Beta–Binomial update on recent hit rate | daily |
| Levels (S/R) | list of `(price, strength, last_touch)` | pivot clustering | add/decay/merge on new pivots | every bar |
| Relations | cluster id, factor loadings, β to market | clustering/PCA refit | Kalman on β; clusters refit weekly | daily / weekly |
| Sentiment, flow | z-scores + surprise vs own history | EWMA baselines | EWMA | per arrival / bar |

### 2.1 Hierarchical priors (shrinkage)
Parameters are estimated at three levels, then shrunk:

```
θ_asset = w · θ̂_asset + (1 − w) · θ̂_cluster ,   w = τ²_cluster / (τ²_cluster + SE²_asset)
θ_cluster = w' · θ̂_cluster + (1 − w') · θ̂_global
```

This is empirical Bayes (James–Stein style). New or illiquid assets borrow strength from their cluster automatically. It's also the formal version of "a predefined view for every asset". The prior is a function of where the asset sits in the cross-section, and it is refined as the asset's own data accumulates.

### 2.2 Fit slow, filter fast
- **Weekly / monthly (structural refit):** cluster membership, factor loadings, HMM parameters, regression coefficients, GARCH parameters, calibration maps. These *are* the priors.
- **Daily:** conjugate updates of return-distribution parameters (with forgetting factor λ ≈ 0.98–0.995), Beta–Binomial updates of event rates, shrinkage weights recomputed, state snapshot written.
- **Intraday (bar close):** filtering only (HMM forward step, Kalman step, GARCH step, EWMA flow/sentiment, S/R proximity). Every step is **O(1) memory and time per asset**: no refits intraday.

### 2.3 Combining tools into one posterior
Each predictive tool `j` emits a calibrated probability `p_j` for a target, e.g. `P(TP before SL)`. They're pooled in log-odds space:

```
logit(p_quant) = logit(p_base) + Σ_j  w_j(regime) · [logit(p_j) − logit(p_base)]
```

- `p_base` = the unconditional/cluster base rate (the prior).
- `w_j(regime)` = tool weights per regime, estimated by walk-forward stacking (constrained ≥ 0) and **shrunk toward equal weights**. This is backtestable.
- The result is the **quant posterior**. On its own it's also the `RULE_N` baseline strategy from PLAN.md §5.

### 2.4 The LLM as a "view" provider (Black–Litterman analogue)
The LLM doesn't output a probability from scratch. It outputs **views relative to the quant posterior**:

```json
{"symbol": "XYZ", "target": "tp_before_sl", "view_logodds": +0.4, "confidence": "medium",
 "reason_codes": ["sentiment_divergence", "level_confluence"], "evidence": ["call_17", "call_22"]}
```

- `view_logodds` is **capped** (e.g. |Δ| ≤ 1.0 ≈ moving 50% to 73%). The combiner applies it scaled by confidence (low 0.25, medium 0.5, high 1.0).
- Final posterior: `logit(p_final) = logit(p_quant) + c · Δ_LLM`.
- **Evaluation falls straight out of this:** forward Brier and log-loss of `p_final` vs `p_quant`, on every candidate. That number *is* the LLM's value-add, measured on all M candidates per session, not just the N traded.

---

## 3. Tool catalogue

Every tool shares one interface (§4). They're grouped by the question they answer.

### 3.1 "What is the volatility state?" (feeds standardisation everywhere)
| Tool | Model | Output | Notes |
|---|---|---|---|
| `vol.ewma` | RiskMetrics EWMA, λ per asset | σ̂ next bar | cheapest; fallback |
| `vol.garch_t` | GARCH(1,1) with t innovations | σ̂_{t+1..h}, persistence α+β, ν | main conditional vol |
| `vol.range` | Yang–Zhang / Garman–Klass from OHLC | realised vol | more efficient than close-close |
| `vol.har` | HAR-RV (daily/weekly/monthly RV, OLS) | σ̂ forecast at h | linear, robust, strong benchmark |
| `vol.regime_pct` | percentile of σ̂ vs own 1y history | 0–100 | label: compressed / normal / expanded |

**Validated σ (2026-11-07):** `vol.robust_bc` is the jump-robust, bias-corrected EWMA. H-V1b is supported: QLIKE −0.05 in-sample and −0.04 in the holdout vs plain EWMA, with post-jump calibration 0.92 vs 0.73. It comes in two outputs:
- **`sigma_risk_next`** for **stops and sizing**;
- **`sigma_rb_next`**, which applies the weekend multiplier m (about 0.57–0.6), for **central estimates** such as first-passage and TP probabilities.

Weekends are quiet in the median but jump-prone. Shrinking σ for risk worsened tail loss (H-V2 failed on QLIKE), so the risk σ is never weekend-shrunk.

### 3.2 "What regime are we in?" — always 9 regimes

The system always models **9 regimes**, as a **3 × 3 grid: direction × volatility**.

|  | **Low vol** | **Normal vol** | **High vol** |
|---|---|---|---|
| **Up** | R1 quiet trend up | R2 trend up | R3 momentum burst / blow-off |
| **Range** | R4 compression (pre-breakout) | R5 balanced range | R6 whipsaw / chop |
| **Down** | R7 grind down | R8 trend down | R9 liquidation / crisis |

**Why a factorised 3×3 and not a free 9-state HMM.** A free 9-state HMM has 72 transition parameters plus 9 emission distributions, and fit on noisy returns it produces unstable, unlabelled states. Factorising into two 3-state chains (a factorial HMM; Ghahramani & Jordan 1997) keeps the 9 regimes but cuts the transition parameters to 2 × 6 = 12. Every state also has a fixed, interpretable meaning, so "R4" means the same thing for every asset and every refit.

| Tool | Model | Output |
|---|---|---|
| `regime.direction` | 3-state jump model / HMM with t emissions on trend features (Kalman slope t-stat, multi-horizon tsmom, efficiency ratio) | p(up), p(range), p(down) |
| `regime.vol` | 3-state model on log σ̂ (GARCH-t) and its percentile vs own history | p(low), p(normal), p(high) |
| `regime.grid` | joint 9-cell posterior. Start with the product of the two marginals; add an **empirical coupling correction** (joint 9-cell transition counts with a Dirichlet prior centred on the product) only if it improves out-of-sample log-likelihood | 9-vector, most likely cell, expected duration, entropy |
| `regime.trend_range` | variance ratio (Lo–MacKinlay) + Hurst (DFA) as **diagnostics** for the direction axis | test stats |
| `regime.market` | the same 9-cell grid on the market proxy (BTC / SPY / equal-weight index) | market cell |
| `regime.consensus` | agreement between HMM, jump model and rule-based labels | **disagreement is information**: transition risk |

**Persistence.** Jump-model penalty plus hysteresis: a cell switch requires posterior > 0.6 for ≥ k confirmation bars. This prevents flickering between cells.

**Overfitting defence: regime-specific parameters are shrunk, never free.** Any coefficient that depends on the regime is written additively and shrunk hierarchically:

```
β(d, v) = β₀ + a_d + b_v + (ab)_{dv}
          global  direction  vol    cell interaction (strongest shrinkage toward 0)
```

Rare cells (R3, R9) borrow almost everything from their row and column effects.

**Effective sample size is the binding constraint.** Regimes are persistent, so the effective number of independent observations per cell is roughly the **number of episodes** in that cell, not the number of bars. Pooling across assets doesn't help much, because crypto/equity regimes are largely market-wide (episodes are shared). Every regime-conditional model must report `n_episodes` per cell. A cell with fewer than ~10 episodes uses the shrunk estimate only.

The asset's 9-cell regime and the market's 9-cell regime are **not** crossed into 81 cells. The market cell enters asset models as a feature (or a main effect), never as a cross-product.

### 3.3 "What is the direction / trend?"
| Tool | Model | Output |
|---|---|---|
| `trend.tsmom` | sign and vol-scaled magnitude of past returns at 3 horizons (Moskowitz–Ooi–Pedersen) | score ∈ [−1, 1] per horizon |
| `trend.kalman_llt` | local-linear-trend state space | slope, slope t-stat, slope change |
| `trend.ma_structure` | EMA stack (fast > mid > slow), distance in ATR units | categorical + continuous |
| `trend.breakout` | Donchian / range break, vol-adjusted | bars since break, magnitude in σ |

### 3.4 "Where is price relative to structure?" (support / resistance / technicals)
| Tool | Model | Output |
|---|---|---|
| `levels.pivots` | swing detection with ATR-scaled zigzag threshold (non-repainting: a pivot confirms only after k bars) | confirmed swing highs/lows |
| `levels.cluster` | 1-D clustering of pivot prices (KDE peaks or DBSCAN with ε = 0.5·ATR) | levels with strength = touches × recency decay |
| `levels.volume_profile` | volume binned by price over window | POC, value-area high/low, low-volume nodes |
| `levels.proximity` | distance to nearest support/resistance in ATR & σ units | the feature the predictive models use |
| `ta.oscillators` | RSI, stochastic, z-score of price vs mean, Bollinger %B | as **features**, all stationary by construction |

The **non-repainting rule** is mandatory. A level or pivot is only known after its confirmation bar, and each level carries `known_at`. Repainting is the most common hidden look-ahead in technical analysis.

### 3.5 "What timing is this?"

**Principle: elapsed time is not an entry or exit signal.** Time tools describe *context*: which session, what's scheduled, how volume compares with normal for this hour. No trade is opened or closed because "N bars have passed". The **only exception** is a strategy explicitly tagged `seasonal` (§3.12), whose edge *is* the calendar effect.

| Tool | Output |
|---|---|
| `time.session` | session (Asia / EU / US / overlap), time-of-day bucket, day-of-week: a context feature |
| `time.seasonality` | intraday vol & volume curve per asset (expected vs actual: volume *surprise*); seasonal return patterns for `seasonal`-tagged strategies |
| `time.calendar` | upcoming scheduled events: macro (CPI, FOMC, NFP), earnings, token unlocks, expiries, funding times. These are **risk context** (gap/jump risk across a stop), not exits |
| `time.since` | bars since last regime change, breakout, level touch: a feature only (e.g. a level not tested for a long time behaves differently), never a trigger |

### 3.6 "What is order flow telling us?" (availability depends on asset class)
| Tool | Model | Output |
|---|---|---|
| `flow.imbalance` | signed volume via tick rule / bulk volume classification; cumulative volume delta | CVD z-score, divergence vs price |
| `flow.ofi` | order-flow imbalance from L1 best bid/ask changes (Cont–Kukanov–Stoikov) | OFI z-score per bar |
| `flow.derivs` | funding rate, open-interest change, basis, liquidations (crypto/futures); put/call, skew (equities) | positioning / crowding z-scores |
| `flow.vpin` | volume-synchronised probability of informed trading | toxicity percentile |

Crypto has the richest free flow data (trades, funding, OI). For equities, L1/L2 data is costly, so flow tools reduce to volume, options and short interest.

### 3.7 "What does sentiment say?"
| Tool | Model | Output |
|---|---|---|
| `sent.news` | per-item score → per-asset EWMA; **mention-volume surprise** and **dispersion** | level z, change z, attention z |
| `sent.social` | same pipeline on social data | same |
| `sent.divergence` | sentiment trend vs price trend | divergence flag + magnitude |

⚠️ **Look-ahead trap inside the tool layer.** A modern LLM used as the sentiment scorer can't be backtested honestly; it has the same contamination problem as the decision layer. So:
- For **historical validation** of sentiment tools, use a contamination-light scorer: the Loughran–McDonald lexicon, or a small model whose training cutoff predates the validation window.
- **Live**, an LLM scorer can be added. It's a new tool version with a fresh forward track record.

### 3.8 "How does this asset relate to the others?" (cross-sectional and relational)
| Tool | Model | Output |
|---|---|---|
| `rel.beta` | Kalman time-varying β to market and sector/cluster factor | β, β change, idiosyncratic vol |
| `rel.factors` | PCA on standardised returns (k = 3–5) on a Ledoit–Wolf covariance | loadings; explained variance (high = "everything moves together" regime) |
| `rel.clusters` | hierarchical clustering on correlation distance `d = √(2(1−ρ))`, or GMM on (factor loadings, vol, trend) | cluster id, membership stability |
| `rel.pairs` | Engle–Granger / Johansen cointegration within cluster | spread z, half-life, test p-value |
| `rel.leadlag` | lagged cross-correlation / Granger on standardised returns | leaders of each cluster |
| `rel.xs_rank` | cross-sectional ranks of momentum, vol, flow, sentiment | percentile per date |

### 3.9 "What is likely to happen?" (predictive, probabilistic)
These are the models the posterior is built from. Targets are defined in **vol units** and aligned with how trades are actually managed.

| Tool | Target | Model |
|---|---|---|
| `pred.ols_ret` | forward vol-standardised return, h bars | OLS/Huber with t errors + HAC; regime interactions |
| `pred.quantiles` | 10/25/50/75/90% quantiles of forward return | linear quantile regression |
| `pred.threshold` | `P(r_{t→t+h} > k·σ)` and `< −k·σ` | logistic + calibration |
| `pred.barrier` | **`P(TP before SL)`**: two-barrier first passage, **no time barrier**, with TP/SL placed by structure × volatility | level-chain model, §3.11 |
| `pred.event` | P(breakout within h), P(vol spike), P(regime switch) | discrete-time hazard (logistic on bars-at-risk) |
| `pred.analog` | empirical forward-return distribution of the **k nearest historical states** (or same GMM cluster) | kNN on standardised state vector; outputs distribution and n_eff |
| `pred.vol` | σ over the trade horizon | from `vol.garch_t` / `vol.har`; feeds stop and target sizing |

`pred.barrier` is the most important one. Its target is literally the trade outcome in R-multiples under the risk engine's own rules, so the probability maps directly to expected value:
`E[R] = p_TP · (|TP − entry| / |entry − SL|) − (1 − p_TP)` (minus costs in R).

### 3.10 Panel / cross-sectional analysis (point in time *and* through time)
| Tool | Model | What it tells us |
|---|---|---|
| `panel.fama_macbeth` | each date: cross-sectional regression of forward returns on characteristics; then time-series mean and NW t-stat of the coefficients | which characteristics are priced *now* and on average |
| `panel.coef_drift` | rolling/EWMA of the Fama–MacBeth coefficient series + Kalman smoothing | **how relationships move over time**: e.g. "momentum premium turning negative" is itself a regime signal |
| `panel.ic` | rolling rank IC of every feature, overall and per regime | feeds tool selection (§5) |
| `panel.fe` | pooled panel with asset fixed effects, double-clustered SE | effects net of persistent asset differences |

### 3.11 Take-profit and stop-loss: structure × volatility (no time barrier)

TP and SL are **placed at market structure and measured in volatility units**. The question the model answers is: *"How likely is price to reach this level before that one, given how far each is from entry, how strong the levels in between are, and the current context?"*

**Layer 1: volatility-only prior (analytic first passage).** Work on log price, with per-bar drift μ and volatility σ (from `vol.garch_t`). For distances `a` up to TP and `b` down to SL:

```
P(TP first) = (1 − e^{2μb/σ²}) / (e^{−2μa/σ²} − e^{2μb/σ²})        →  b / (a + b)  when μ = 0
```

Time doesn't appear: absorption probabilities don't depend on the clock. **Key implication:** with μ = 0 this gives E[R] = 0 exactly, whatever the reward-to-risk ratio. A wide TP doesn't create edge by itself. **Edge can only come from drift (trend/regime) or from structure** (levels breaking or holding differently than a random walk would).

**Layer 2: the level chain (structure).** The levels between SL and TP are the states of a small absorbing Markov chain:

```
 SL ◄── s₂ ◄── s₁ ◄── [entry] ──► r₁ ──► r₂ ──► TP(=before r₃)
         ↑ hold/break       ↑ hold/break    ↑ break/reject    ↑ break/reject
```

- Moving between adjacent levels uses the Layer 1 first-passage probability for that segment (distance in σ units, drift from the regime cell).
- On arriving at a level ℓ, price **breaks** with probability `π_ℓ` (continues to the next level) or **rejects** with `1 − π_ℓ` (travels back toward the previous one).
- `P(TP first)` is the absorption probability. It solves an (≤ ~8)-state linear system: microseconds, no simulation.

**The break/reject model `π_ℓ` (`levels.break_prob`).** This is where market structure and context enter. It is trained on *level-arrival events* (price comes within 0.25σ of a confirmed level). Each event resolves as a break (moves δσ beyond) or a reject (moves δσ back). Because the label is price-defined, there is almost no censoring and no time limit is needed.
- Features:
  - **level:** strength (recency-decayed touch count), type (swing pivot / volume POC / value-area edge / round number), number of prior tests (levels weaken with repeated tests), higher-timeframe confluence;
  - **approach:** speed in σ units, distance travelled since the previous level, volume surprise, flow imbalance (CVD/OFI z) on approach;
  - **context:** regime cell as shrunk direction and volatility main effects, the market cell, and sentiment.
- Model: logistic, panel-pooled with hierarchical shrinkage (asset → cluster → asset class), calibrated, purged walk-forward validation.

**Placement rules (pre-registered, structural, few free parameters).**
- **SL** = beyond the nearest valid opposing level, plus a buffer. Buffer = 80th percentile of **overshoot** (how far price pierces a level that ultimately holds), in σ units, estimated per volatility state. If that level is < 0.5σ away, use the next one. If the nearest valid level is > X σ away, it's no trade (or the vol-only fallback). A wide stop is fine in itself: fixed risk R means a wider stop gives a smaller position, not more risk.
- **TP candidates** = just in front of each level in the trade direction, using the 20th percentile of **undershoot** (price tends to turn before the level). Compute `E[R]_k` for each of the ≤ 4 nearest candidates with the chain. Choose the argmax, then **shrink it for selection** (subtract ≈ √(2 ln K)·se). Picking the best of K candidates is optimistic.
- **No structure** (price discovery, all-time highs, new listings): fall back to a vol-only target with drift from the regime cell. The trade is flagged `no_structure` and its performance tracked separately.

**Calibration (Layer 3).** Predicted `P(TP first)` vs realised outcome on historical panel replays of the deterministic placement rules, isotonic-calibrated by vol state. Since the placement rules and the chain are deterministic, all of this is backtestable.

### 3.12 Position management: no holding-period limits

A position closes only for a **price or state reason**, never because time has passed:

| Exit | Rule | Deterministic? |
|---|---|---|
| Stop loss | price hits SL | yes |
| Take profit | price hits TP | yes |
| Structural trail | when a new pivot **confirms** in the trade direction, the stop moves beyond it (plus the overshoot buffer) | yes |
| Thesis invalidation | regime posterior turns against the trade with hysteresis (e.g. long and p(down row) > 0.6 for k bars) | yes |
| Slot replacement | at max open positions, a new candidate replaces the weakest open one only if `E[R]_new − E[R]_remaining > hurdle` (costs + uncertainty) | yes |
| LLM exit view | at each session the LLM reviews open positions and may propose an exit, with reasons | **forward-evaluated** vs the counterfactual of holding |
| Seasonal exit | **only** for strategies tagged `seasonal`, where the calendar pattern defines entry and exit | yes; separate track record |

Consequences of having no time limit:
- **Open positions are re-scored every session.** The chain is recomputed from the current price, giving `E[R]_remaining` and the current `P(TP first)`.
- **Max open positions becomes the binding constraint** in place of a time stop. Hence the slot-replacement rule, which compares expected value, not age.
- **Evaluation:** open trades are marked to market in R for interim reporting. Shadow outcomes for untaken candidates are also price-resolved, so they take longer to resolve; the evaluation reports resolved vs open counts.

---

## 4. Tool interface

Every tool, whether called by the batch pipeline or by the LLM, follows one contract:

```python
@tool(name="pred.barrier", version="0.3.1", cadence="daily", cost="low")
def barrier(asset: str, as_of: datetime, side: str, entry: float) -> ToolResult: ...
```

```json
{
  "tool": "pred.barrier", "version": "0.3.1", "asset": "XYZ",
  "as_of": "2026-11-09T13:00Z", "data_max_ts": "2026-11-09T12:59Z",
  "value": {"p_tp_first": 0.42, "p_tp_vol_only": 0.34, "sl": 98.9, "tp": 104.5, "rr": 1.9, "exp_R": 0.18},
  "uncertainty": {"n_eff": 412, "ci90_exp_R": [0.05, 0.55]},
  "quality": {"brier_oos": 0.208, "calibration_err": 0.02, "regime_ic": 0.06, "stale_features": []},
  "summary": "TP before SL 42% vs 34% vol-only; edge from weak resistance at 104.8 (break π=.38) in R2."
}
```

- `data_max_ts <= as_of` is **asserted** at runtime. A violation raises an error and never returns silently.
- `quality` travels with every output, so the LLM always knows how much to trust a tool *right now*.
- `summary` is a one-line natural-language rendering, so the LLM doesn't spend tokens parsing JSON.
- A **tool registry** (`tools/registry.yaml`) records each tool's name, version, cadence, cost, applicable asset classes, regimes where it has validated skill, and its status (`active` / `probation` / `retired`).

---

## 5. Tool selection: which tools at which point in time

There are three levels, and only the last involves the LLM.

**Level 1: deterministic availability gating.** A tool runs for an asset only if its data exists and is fresh, the asset class is supported, and its status is `active`.

**Level 2: skill-based weighting (backtestable).** The `panel.ic` / calibration tracker maintains, per tool × regime, a **posterior of skill**: a Normal posterior on IC, or a Beta posterior on hit rate, with a shrinkage prior centred at zero skill. Then:
- Combiner weights `w_j(regime)` (§2.3) are proportional to the posterior mean skill, floored at 0.
- A tool whose `P(skill > 0) < 0.6` for 60 days moves to `probation` (shown to the LLM, weight 0). After 120 days it is `retired`.
- This all happens automatically, with no LLM judgment and no hand-tuning.

**Level 3: LLM drill-down (forward-evaluated).** The LLM always receives the **precomputed** market card and candidate cards (§6.4), which already contain the Level 2 output. It may then call drill-down tools with a **budget**, e.g. ≤ 25 calls per session and ≤ 4 per candidate:
- A **mandatory checklist** per candidate applies regardless of budget: regime, vol, nearest levels, `pred.barrier`, calendar events.
- **Discretionary calls** are things like `rel.pairs`, `pred.analog`, `sent.divergence` and `flow.*` detail. The journal records which were called and whether they changed the view. Over time this shows which drill-downs actually add value. If a drill-down consistently adds value, it gets promoted into the precomputed card.

---

## 6. Lean compute and storage

Target: the whole system runs on one laptop-class machine, needs < 4 GB RAM at peak, and keeps a year of data for ~200 assets in single-digit GB.

### 6.1 Storage stack: no database server
| Data | Format | Engine |
|---|---|---|
| Bars, flow aggregates, features | **Parquet**, zstd, `float32`, partitioned `asset_class/freq/year/month` | **DuckDB** (in-process, out-of-core) for queries; **Polars** (lazy/streaming) for compute |
| Model parameters / filter states | one small `.npz`/JSON per (model, asset, version) | file system |
| State snapshots (belief state) | one Parquet file per snapshot time, all assets as rows | DuckDB |
| Journal (prompts, tool calls, decisions, fills) | append-only **JSONL**, compacted to Parquet daily | DuckDB |
| Text (news/social) | **no article bodies**: hash, ts, source, entity, scores, (optional) float16 384-d embedding | Parquet |

### 6.2 Point-in-time correctness is built into the schema (bitemporal)
Every table has `event_ts` (when it happened) and `known_ts` (when we could have known it, e.g. the bar close, pivot confirmation or data-vendor publish time). All reads go through one function:

```sql
SELECT ... FROM t WHERE known_ts <= :as_of
```

Revisions are new rows with a new `known_ts`. Nothing is updated in place. This one rule eliminates most look-ahead in the tool layer.

### 6.3 Data tiering and sizing
| Tier | Retention | Size estimate (200 assets) |
|---|---|---|
| Raw trades / L1 book | rolling 7–30 days, then aggregated and dropped | GBs/day for crypto: never kept long-term |
| 1-min bars + flow aggregates | 1 year, only for active intraday universe | ~105M rows/yr → ~1–1.5 GB compressed |
| 15-min / 1-h bars | multi-year | ~7M / 1.75M rows/yr: trivial |
| Daily bars, features, snapshots | full history | trivial |

### 6.4 Compute and memory discipline
- **Precompute, don't recompute.** A batch job at each decision bar computes every Level 1/2 tool and writes a snapshot. Most LLM tool calls become **lookups** (milliseconds, no RAM spike). Only a few drill-downs (analogs, pairs) compute on demand, from the snapshot.
- **Recursive (online) estimators everywhere intraday:** EWMA mean, variance and covariance, GARCH recursion, Kalman steps, HMM forward step. State per asset is a few hundred bytes.
- **Covariance via factor model** for large universes: store `B` (N×k) + specific variances instead of the full N×N matrix.
- **Process asset by asset** with Polars streaming. Never load the full panel into memory. Panel tools (Fama–MacBeth) work date by date: one cross-section at a time.
- **LLM context is memory too.** Each asset becomes a ~150-token **state card**; the market card is ~300 tokens. With 20 candidates that's ≈ 3.5k tokens before drill-downs. That keeps sessions cheap and leaves room for reasoning.

Example state card (what the LLM actually sees):
```
XYZ | cluster: L1-majors (stable 94%) | β_mkt 1.21↑
REGIME R2 trend-up/normal-vol p=0.71 (dir up .82, vol normal .86) — consensus HIGH | mkt R1
VOL   σ_d 3.1% (62nd pct, expanding), ν=4.2, GARCH persist .96
TREND tsmom +0.6/+0.4/+0.2 (short/med/long), kalman slope t=2.9
LEVELS R 104.8 (1.1 ATR, str 0.8) | S 99.6 (0.9 ATR, str 0.6) | POC 101.0
FLOW  CVD z +1.4, funding z +2.1 (crowded long ⚠), OI +8%
SENT  news z +0.3, attention z +1.8, divergence: none
EVENTS token unlock in 3d
PLAN  SL 98.9 (below S 99.6 + overshoot .3σ) | TP 104.5 (front of R 104.8) | RR 1.9
PRIOR P(TP first)=0.42 (vol-only .34; R 104.8 break π=.38) E[R]=+0.18 [−0.05,+0.39]
```

---

## 7. Session flow (one decision cycle)

1. **T−5 min: batch.** Ingest up to `as_of`, update filters, compute all precomputed tools, write the snapshot, build cards. Deterministic and reproducible.
2. **Candidate channels.** Main list: rank the universe by `E[R]` from the quant posterior plus liquidity filters, then take the top M (M ≈ 15–20). Side channels (anomaly, cross-sectional, event, transition, relational) produce short lists that the LLM may promote. Every candidate carries its channel tag. See [CANDIDATE_GENERATION.md](CANDIDATE_GENERATION.md).
3. **LLM session** (pinned model, prompt version, tool versions):
   1. Read the market card → state the market regime view (can disagree, with reasons).
   2. For each candidate: mandatory checklist → optional drill-downs → **view** (Δ log-odds, confidence, reason codes) or "no view" (Δ = 0).
   3. Rank all M candidates; mark up to N as "take".
4. **Combiner** → `p_final`, `E[R]_final` for all M.
5. **Risk engine** (hard rules): drop anything with `E[R]_final < threshold`; enforce N/day, exposure, correlation (max one per cluster unless explicitly paired) and the daily loss halt. Size: `qty = R_equity / |entry − stop|`.
6. **Execution:** bracket orders (entry, SL and TP from structural placement §3.11; no time stop). Open positions are re-scored and managed per §3.12.
7. **Journal:** everything, including the baselines (`RANDOM_N`, `RULE_N` = quant posterior only) for the same session.

---

## 8. Code layout

Implemented 2026-11-07 (see README for commands). **Rust** handles the hot path and everything touching money. **Python** handles fitting, research and LLM orchestration.

```
config/              settings, risk, schema (lake contract), datasources, tools registry
crates/
  at-core            time/freq, look-ahead guard, domain types, config + secrets, exact tick/lot rounding
  at-store           bitemporal Parquet writer validated against schema.yaml
  at-bybit           REST (signed, paced, retried), WebSocket, market data, trade archive -> flow bars, trading
  at-alt             Fear & Greed, Deribit DVOL, DefiLlama stablecoins, FRED/ALFRED vintages
  at-filters         EWMA, GARCH-t filter/forecast, range vol, first passage (§3.11 L1), level chain (§3.11 L2)
  at-risk            fixed-R sizing incl. costs, caps, halts, net-E[R] gate, slot replacement
  at-exec            Broker trait; BybitBroker (bracket orders), PaperBroker (conservative, for baselines)
  atx                CLI
src/ai_trader/
  core/              config, time, ToolResult, tool registry
  storage/           Lake.read_asof (DuckDB), Lake.write
  features/          returns, vol standardisation, ADF+KPSS, fracdiff, PSI
  tools/             vol.ewma, vol.garch_t (fit -> data/models/vol.garch_t/*.json -> Rust filter)
  journal/           append-only JSONL journal
  agent/             phase 4
tests/               look-ahead invariance, PIT revisions, Rust->Python reads, features, GARCH recovery
```

The **look-ahead test** in `tests/` is the most important test in the repo. For random `t`, compute each tool with the full dataset and `as_of=t`, then again with all data after `t` physically deleted. Outputs must be identical.

---

## 8b. Strategy & research layer (2026-11-07)

- **Strategies** (`src/ai_trader/strategies/`) are deterministic signal generators in [-1, 1]. They never size positions. Each declares a falsifiable hypothesis and the regime cells where it is *expected* to work: the prior.
  - Families: trend (tsmom, donchian_breakout, ema_stack), mean reversion (zscore_reversion, rsi_reversion), volatility (compression_breakout), positioning (funding_contrarian, oi_price_divergence), flow (flow_momentum, cvd_divergence), calendar (weekend_shock_follow, weekend_drift_fade, cme_gap_fill).
  - Parameters are pre-registered in `config/strategies.yaml`. `assert_causal` (truncation invariance) is tested for every strategy.
- **Regime labeller v0** (`research/regimes.py`): causal, rule-based 3×3 grid (trend score vs ±θ; σ̂ vs its trailing terciles; hysteresis). It's the baseline the factorial HMM must beat.
- **Calendar / weekend context** (`research/calendar.py`), in New York time:
  - TradFi weekend = Fri 17:00 → Sun 18:00 ET (CME closure);
  - weekend stats in Friday-σ units (wk_z, rv_ratio, vol_ratio);
  - the absorption window ends at Monday's NYSE close;
  - the gap-fill null is the reflection principle.
- **Evaluation** (`research/evaluate.py`, `studies.py`):
  - P&L in risk units, sized at entry, net of costs;
  - per-cell stats as cell-conditional equal-weight portfolios clustered by time, Newey–West t-stats;
  - effective N = episodes, empirical-Bayes shrinkage;
  - forward-path priors (quantiles, MFE/MAE in σ) per cell;
  - post-shock continuation per cell;
  - every output split at the LLM knowledge cutoff.
- **Playbook priors** are written to `data/priors/*.parquet`. They're the starting point for "which strategy in which cell" and "what path to expect", to be updated as the holdout grows.

## 9. Build order (refines PLAN.md phases 1–4)

1. `core/` (as_of reader, ToolResult, registry) + look-ahead test harness
2. `data/` for **one** asset class and **one** frequency (daily) + `features/`
3. Vol tools → regime tools → trend tools → levels: the minimum viable state card
4. `pred.barrier` + `pred.threshold` + calibration + walk-forward validation → first quant posterior
5. `belief/` combiner + hierarchical shrinkage; `rel.clusters` / `rel.factors` (needed for cluster priors)
6. Cards + agent dry runs (no orders)
7. Flow, sentiment and panel tools, each added only after it passes validation and improves the quant posterior's out-of-sample Brier
8. Intraday frequency, if the chosen timeframe needs it

---

## 10. Open design questions

- ~~Barrier parameters~~ → resolved: structural × vol placement, no time barrier (§3.11–3.12).
- ~~Number of regimes~~ → resolved: always 9, as a factorised 3 × 3 direction × vol grid (§3.2).
- **Placement constants** (δ for break/reject resolution, overshoot/undershoot quantiles, max stop distance X σ): pre-register them, then validate by walk-forward.
- **Forgetting factor λ:** asset-class specific; choose by walk-forward log-loss.
- **Candidates outside the screener's list:** analysed in [CANDIDATE_GENERATION.md](CANDIDATE_GENERATION.md).
- **Asset class** (still open, PLAN.md §7). It determines which flow tools exist at all.
