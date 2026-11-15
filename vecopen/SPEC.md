# Vector-Open Level Signal Generator: Framework and Agent Build Plan

Nov 14, 2026 · Américo Cabral Abrantes Serra. Exported verbatim from `Vector-Open Level Signal Generator Framework and Agent Build Plan.pdf` (13 pages) for the agents. **Read-only for agents.** A change requires the user's written approval and a new registry entry. Decisions the user took on 2026-11-14 for section 9 are recorded at the end and in `config/baseline.yaml`.

## 1. Hypotheses and architecture

The observation splits into two separate, falsifiable claims. The system is only built if at least one survives a placebo test. Treating them as one claim is the first route to overfitting: a level can attract price without repelling it, and the trade is different in each case.

**H1 (decision point).** When price first returns to an untouched leg-origin open, it resolves. Within h bars it leaves a band of ±k ATR around the level, by rejection or by acceptance through it, more often and by more than at matched placebo levels. The level is a saddle point, not a wall, so the primary test is on the size of the move and not its sign. Which branch occurs is the job of the state and context layers. This is the entry claim.

**H2 (magnet).** Conditional on an untouched leg-origin open sitting at distance d from price, the probability that price touches it within h bars is higher than for matched placebo levels at the same distance. It is also higher than the driftless diffusion baseline. This is the exit-target claim.

The architecture keeps four layers independent, so each can be tested and discarded alone:

1. **Level layer** (structural, rule-based): detects leg origins on each timeframe and maintains a registry of active open levels. It has no fitted parameters.
2. **State layer** (local price action at the touch): absorption, trapped flow, volatility-volume heat. It measures how hot the market is, not where it goes.
3. **Context layer** (direction prior): point-in-time macro and cross-asset state, compressed into one score. It only gates or signs trades.
4. **Decision layer:** a rule first, then a regularised meta-model, and only if the rule alone has an edge over placebo.

The trade thesis in one sentence: at the first return to an untouched leg-origin open, take the branch that state and context indicate (fade on rejection, follow on acceptance), and use untouched opens as profit targets for existing positions.

## 2. Point-in-time definitions

Every object below is computable from bars that have already closed. A bar on timeframe τ is indexed by its close timestamp; its values do not exist before that instant.

**Leg origin.** This is the detection rule chosen against the simulator and the Bund 4h example: the open of the candle that starts a displacement leg, at a threshold of 2.0 ATR.

- A bar t on timeframe τ is a candidate when it is directional and its open sits near its extreme, measured by the open-side wick ω.
- It is confirmed at the first bar t+k, with k between 0 and n, whose close is at least θ_L·ATR away from the open in the candle's direction.
- No bar in between may trade back through that open.
- ATR is taken at t−1.

```
s_t = \operatorname{sign}(C_t - O_t), \qquad
\omega_t = \begin{cases} (H_t - O_t)/(H_t - L_t) & s_t = -1 \\ (O_t - L_t)/(H_t - L_t) & s_t = +1 \end{cases}, \qquad \omega_t \le \bar{\omega}
```
```
k^* = \min\Big\{ k \in [0, n] : s_t\,(C_{t+k} - O_t) \ge \theta_L\,\text{ATR}_{t-1} \ \text{ and } \ s_t\,(X_{t+j} - O_t) > 0 \ \ \forall j \in [1, k] \Big\}, \qquad X = \begin{cases} H & s_t = -1 \\ L & s_t = +1 \end{cases}
```

Once a leg is confirmed, no new origin of the same direction is accepted on that timeframe until after bar t+k*. So each leg yields one level, at its earliest qualifying candle. Volume is not part of detection, because the rule has to work on instruments without it. Where volume exists, the leg's relative volume is stored as a feature.

**Level.** A confirmed origin creates the level ℓ = O_t. A down-leg leaves the level above price, and an up-leg leaves it below. The level is born at the close of the confirmation bar t+k*, never at bar t: before that instant it is not known to exist, and using it earlier is look-ahead. Its zone is ℓ ± κ·ATR, frozen at birth.

**Level state machine.**
- A level is `untouched` from birth until the first base bar trades into its zone. That first fill is the decision event.
- After the fill, the level stays alive as a `reference` with a polarity: resistance while τ-bars close below the zone, support while they close above it. The polarity flips each time a τ-bar closes beyond the zone on the other side.
- It is `expired` after A_max τ-bars.

The primary tests use first fills only. Later touches of reference levels are a secondary, pre-registered variant.

**Touch event.** On the base timeframe (5 minutes), an event fires at the close of the first bar whose range intersects the zone. No separate departure condition is needed, because a level is only born once price is θ_L·ATR away from it. The event timestamp t₀ is that bar's close, and every feature uses information up to t₀ only.

**Confluence.** The confluence score at t₀ is the weighted count of other active levels, across all timeframes, whose zones overlap the touched zone. Weights rise with timeframe in a fixed order (5m < 15m < 30m < 1h < 2h < 4h < D < W < M) and are set a priori, not fitted.

## 3. Labels and null tests

No model is trained until the raw level effect beats placebo. This section is the gate that decides whether the rest of the build happens.

**Labels.** Each first fill gets three outcomes, all measured from the open of the base bar after t₀.
- The non-directional outcome is resolution: whether price leaves a band of ±k_pt·ATR around ℓ within h base bars, and the maximum absolute excursion in ATR units.
- The two directional outcomes are triple-barrier labels:
  - one for the fade, trading in the direction g = s_t of the origin leg;
  - one for the follow-through, trading against it, through the level.
- Each label has its target at k_pt·ATR, its stop at k_sl·ATR beyond the opposite edge of the zone, and time barrier h.
- Each label is 1 if its target is hit first and 0 otherwise, with the realised return in R multiples stored alongside.
- If both barriers fall inside one base bar, the stop is assumed to be hit first.

**Why a placebo is mandatory.** The barriers are asymmetric by construction, so even a driftless random walk gives a rejection probability above one half. With the touch at the near edge of the zone, target distance a and stop distance b from entry:

```
P_{\text{RW}}(\text{target first}) = \frac{b}{a+b}, \qquad
P_{\text{RW}}(\text{touch level at distance } d \text{ within } T) = 2\,\Phi\!\left(-\frac{d}{\sigma\sqrt{T}}\right)
```

These two formulas are the analytic baselines for H1 and H2. They are necessary but not sufficient, because real returns have volatility clustering and intraday seasonality. So the binding comparison is against empirical placebo levels drawn from the same data.

**Placebo levels.** Three families are generated with the same registry code and the same state machine:

1. Opens of candles that are not leg origins, on the same timeframe, matched to each treated level by nearest neighbour on age, departure distance and ATR regime.
2. The treated level displaced by ±υ·ATR, with υ drawn uniformly from [1, 3].
3. Other prices of the same origin candle: its close, its midpoint and its extreme. This family answers whether the open specifically matters, or just any price inside the origin candle.

**Statistics.**
- For H1, the statistic is the difference between treated and placebo events in resolution rate and in mean absolute excursion. The fade and follow target-first rates and the mean R are reported as secondary.
- For H2, it is the difference in touch rate within h bars at matched distance.
- Confidence intervals come from a cluster bootstrap with 10,000 resamples, which absorbs the dependence between overlapping events. Clusters are trading days for intraday levels and weeks for daily and higher.
- The family of primary tests is one per asset and timeframe group, corrected with Holm.

**Gate G2.** The build continues only if, on the research sample, treated beats all three placebo families with a Holm-adjusted p below 0.05 for H1 or H2, and the sign is the same in at least two thirds of calendar years. If only H2 passes, the system is built as a target and exit engine, not an entry engine.

## 4. State features and the context layer

The state layer is a fixed list of 12 features, all scale-free (ATR units, ratios or lagged z-scores), so that they are comparable across assets and volatility regimes. The list is closed; adding a feature is a logged trial.

| Feature | Definition at t₀ | Proxy for |
|---|---|---|
| `tf_rank` | Ordinal timeframe of the level | Importance of the level |
| `leg_size`, `leg_vol` | Leg size in ATR at confirmation; relative volume over the leg, where volume exists | Strength of the original displacement |
| `age` | τ-bars since birth, log scale | Decay of resting interest |
| `confluence` | Weighted count of overlapping active levels | Multi-timeframe agreement |
| `approach_speed` | Return over the last m base bars / ATR | Momentum into the level |
| `approach_vol` | Relative volume over the last m base bars | Participation on the approach |
| `absorption` | z-score of volume per unit range on the touch bar | Heavy volume, little progress |
| `trap_wick` | Wick beyond ℓ / bar range, with close back inside the zone | Trapped breakout traders |
| `delta_div` | Sign disagreement between cumulative delta and price over m bars | Aggression failing to move price |
| `heat` | Realised-vol z-score × relative volume | How hot the market is |
| `tod` | Session bucket (Asia, Europe, US, overlap) | Liquidity regime |

Delta features use true aggressor volume where the feed has it, and bulk volume classification otherwise. If neither is reliable for an asset, `delta_div` is dropped for that asset, not imputed.

**Context layer.** Context produces one number c ∈ [−1, 1] per asset per day, known before the session opens. It is used only to gate (trade when sign(c) = g) or to sign the exposure. It has three pre-specified components, averaged with equal weights:

1. **Higher-timeframe structure:** the sign of the last weekly and daily close relative to the nearest active weekly and daily leg-origin opens.
2. **Macro state:** the first two principal components of a small panel (2-year yield change, curve slope change, broad dollar, credit spread, equity-implied volatility, a data-surprise index). Loadings are estimated on an expanding window and mapped to the asset by a rolling regression with a one-day lag.
3. **Regime:** the filtered (never smoothed) state probability from a 2-state Gaussian HMM on the asset's daily returns, re-estimated monthly on an expanding window.

Macro series enter by release timestamp and first-release vintage only. Revised series are a look-ahead and are rejected at load. The Fed reaction function and its spillovers are represented by the latent factors, not modelled structurally. A structural block is a later extension and a separate set of trials.

## 5. Model ladder and validation

Complexity is added one rung at a time. Each rung must beat the one below out of sample, net of costs, or it is dropped.

| Rung | Decision rule | Fitted parameters |
|---|---|---|
| M0 | Fade every first fill in the direction g of the origin leg | 0 |
| M1 | Fade when sign(c) = g, follow through when sign(c) = −g | 0 in the decision; context loadings estimated on expanding windows |
| M2 | L2 logistic regression on the 12 state features over M1 events; trade when p̂ exceeds break-even, size proportional to the excess | 13 plus one penalty |
| M3 (challenger) | Gradient-boosted trees, depth ≤ 3, monotone constraints matching the sign expected a priori | Kept only if CPCV log loss beats M2 by more than one standard error |

M2 is meta-labelling: the rule decides the side, and the model decides whether to act and how much. Its break-even probability, with target a, stop b and round-trip cost c in the same units, is:

```
p_{\text{be}} = \frac{b + c}{a + b}
```

**Sample design.**
- The last 25% of the calendar is a locked holdout, stored in a separate directory that no agent reads before the final phase. The first 75% is the research sample.
- Inside the research sample, M0 and M1 are evaluated by plain walk-forward.
- M2 and M3 use combinatorial purged cross-validation with 10 groups and 2 test groups (45 splits, 9 backtest paths). Every training event whose label window overlaps a test group is purged, and h base bars are embargoed after each test group.
- Events are weighted by average label uniqueness.
- M2 requires at least 1,000 events. Below that, assets are pooled with asset fixed effects, or M2 is not fitted.

**Acceptance statistics.** The Sharpe ratio is deflated for the number of trials N recorded in the registry, the variance of Sharpe across trials, and the skewness γ₃ and kurtosis γ₄ of returns:

```
\text{SR}_0 = \sqrt{\mathbb{V}[\widehat{\text{SR}}_n]}\left[(1-\gamma)\,\Phi^{-1}\!\left(1-\tfrac{1}{N}\right) + \gamma\,\Phi^{-1}\!\left(1-\tfrac{1}{Ne}\right)\right], \qquad
\text{DSR} = \Phi\!\left(\frac{(\widehat{\text{SR}} - \text{SR}_0)\sqrt{T-1}}{\sqrt{1 - \gamma_3\widehat{\text{SR}} + \tfrac{\gamma_4 - 1}{4}\widehat{\text{SR}}^2}}\right)
```

**Gate G3.** A rung is accepted when all four hold:
- DSR exceeds 0.95;
- the probability of backtest overfitting from CPCV is below 0.20;
- net Sharpe is positive in at least two thirds of walk-forward folds;
- the result survives doubling the cost assumption.

The holdout is run once, on the single accepted configuration, and its result is reported whatever it is.

**Assumptions and how each is respected.**

| Method | Assumption | Control |
|---|---|---|
| Event-study inference | Independent observations | Cluster bootstrap by day or week; uniqueness weights |
| Logistic regression | Stationary, comparable features; no separation | ATR-unit and lagged z-score features; L2 penalty; variance inflation check |
| Cross-validation | No information shared across folds | Purging and embargo sized to the label horizon |
| PCA factors | Stable loadings | Expanding-window estimation; sign alignment to the previous fit |
| HMM regime | Probabilities usable in real time | Filtered probabilities only; monthly re-estimation on past data |
| Sharpe-based selection | A single trial | Trial registry and DSR |
| Barrier labels | Fills are achievable | Next-bar-open entry, stop-first tie rule, explicit costs |

## 6. Look-ahead and overfitting controls

Leakage is prevented by construction rules and then proven by automated tests. A failing test blocks every later phase (gate G1).

**Construction rules.**
1. Every higher-timeframe bar is keyed by its close timestamp and joined to the base timeframe with a backward as-of join on `close_time <= t`. Joining on the bar's open time is the most common leak in multi-timeframe work and is forbidden.
2. Every rolling statistic (ATR, median, MAD, volume baseline, z-score) is shifted by one bar before use.
3. A level enters the registry at the close of its confirmation bar, not of its origin candle, and its zone width is frozen at birth.
4. Bar boundaries for 1h and above follow one session convention per asset, fixed in the config before any result is seen, because moving the anchor moves every open.
5. Futures use difference (Panama) back-adjustment, so that level-to-price distances in points are preserved. Ratio adjustment is rejected because it distorts them. Each level also stores the unadjusted contract price for live use.
6. PCA, HMM, scalers and the logistic model are fitted on training data only, inside each fold.

**Automated leak tests.**
1. **Truncation invariance:** for 200 random timestamps t, the level registry, features and signals computed on data truncated at t equal, exactly, those computed on the full data at t.
2. **Future perturbation:** multiplying all prices after t by a random factor leaves every output at or before t unchanged.
3. **Timestamp audit:** for every event, the maximum source timestamp of any input is at or before t₀.
4. **Shuffled-label control:** M2 trained on permuted labels gives a CPCV AUC within 0.50 ± 0.02.
5. **Random-walk control:** the full pipeline run on a simulated driftless GARCH series calibrated to the asset's returns gives no significant H1 or H2 effect and a DSR below 0.5.

**Overfitting controls.**
1. All thresholds are fixed a priori in one config file (section 7) and hashed; the hash is written to every result.
2. Each run with a different config is a trial, appended to an append-only registry with its hash, date and metrics. The registry's count is the N in the DSR.
3. Robustness is examined on a coarse 3 × 3 × 3 grid around the three leg-origin parameters, reported as a full surface. The baseline config is never replaced by the grid's best cell. A result is credible only if neighbouring cells agree in sign.
4. Results are broken down by asset, year, timeframe and session. A result driven by one cell of that breakdown is treated as a failure.
5. The holdout directory is unreadable until phase 9 and is used once.

## 7. Frozen parameters

These values are chosen before any data is examined and go into `config/baseline.yaml` unchanged. They are a priori choices from the user's description, not estimates. After phase 5, every edit is a counted trial.

| Parameter | Symbol | Baseline | Robustness grid |
|---|---|---|---|
| Leg size threshold (user's choice, 14 Nov) | θ_L | 2.0 ATR_τ | 1.5, 2.0, 2.5 |
| Confirmation window | n | 6 τ-bars | 4, 6, 8 |
| Maximum open-side wick of the origin candle | ω̄ | 0.25 of range | 0.15, 0.25, 0.35 |
| Volume baseline sessions (feature only) | N | 20 | none |
| ATR length | | 14 | none |
| Zone half-width | κ | 0.10 ATR_τ | none |
| Level expiry | A_max | 200 τ-bars | none |
| Resolution band and profit target | k_pt | 1.0 ATR_τ | none |
| Stop beyond opposite zone edge | k_sl | 0.5 ATR_τ | none |
| Time barrier | h | 6 τ-bars, in base bars | none |
| Approach lookback | m | 6 base bars | none |
| Timeframe weights | | 1 to 9 for 5m, 15m, 30m, 1h, 2h, 4h, D, W, M | none |
| Base timeframe | | 5m; levels from 15m upward in the primary test | 5m levels need 1m data |
| Cost per side | c | 1 tick plus commission, per asset | doubled in the stress run |
| Research / holdout split | | 75% / 25% by calendar | none |

Barriers and horizon scale with the level's own timeframe, so a weekly open is judged over weeks and a 15-minute open over hours. The grid has 27 cells; with the baseline, that is 27 logged trials for the robustness surface.

## 8. Claude Code agent plan

The build is ten phases with three hard gates. One orchestrator session runs it, delegating each phase to a builder subagent and each gate to an independent auditor with a fresh context. Sections 1 to 7 of this document are the specification.

**Stack and layout.** Python (polars, numpy, scipy, scikit-learn, hmmlearn, lightgbm, pytest, hypothesis). The PDF names Python 3.11; this repo runs 3.13 under uv, recorded as a deviation.

```
vecopen/
  SPEC.md                  # this document, read-only for agents
  CLAUDE.md                # standing rules below, verbatim
  config/baseline.yaml     # section 7, hashed with sha256
  data/research/           # first 75% by calendar
  data/holdout/            # last 25%, unreadable until phase 9
  src/vecopen/
    data.py bars.py legs.py levels.py events.py labels.py placebo.py
    features.py context.py stats.py cv.py models.py backtest.py registry.py
  tests/                   # unit tests by builders; tests/leak/ by the auditor only
  trials/registry.jsonl    # append-only
  reports/                 # one Markdown report per phase
```

**Roles.**
1. **Orchestrator (main session):** runs phases in order, never writes code in `src/`, and stops at a failed gate and reports to the user.
2. **Builder (one subagent per phase):** implements exactly the deliverable of its phase and its unit tests.
3. **Leak auditor (fresh context at G1, and again after phases 6 and 8):** writes and runs `tests/leak/`; may not edit `src/`.
4. **Stats reviewer (fresh context at G2 and G3):** reads only `reports/` and `trials/registry.jsonl`, recomputes the gate statistics from the saved event tables, and returns pass or fail with the numbers.

**Standing rules for `CLAUDE.md`.** See `vecopen/CLAUDE.md`.

**Phases.**

| Phase | Deliverable | Acceptance criteria |
|---|---|---|
| 0 Scaffold | Repo, `load_config()` returning config plus sha256, `registry.append()` | `pytest` green; hash printed; registry refuses edits to past lines |
| 1 Data | `load_bars(asset)` with columns `ts_close` (UTC), o, h, l, c, v, optional buy_v and sell_v; `resample(df, tf, session)`; roll adjustment; split script | Resampled OHLCV equals aggregation of base bars in a property test; no duplicate timestamps; gaps listed and explained; holdout unreadable |
| 2 Levels | `LegOriginDetector.update(bar)`, `LevelRegistry.update(bar)` per section 2 | 12 hand-built synthetic cases pass, covering birth at the confirmation bar, one level per leg, zone freeze, first fill, polarity flip, expiry and confluence |
| 3 Events and labels | `TouchDetector`, `label_event()` triple barrier, `PlaceboGenerator` for the three families | Stop-first tie rule tested; placebo levels pass through the identical registry code path; the matched-sample balance table shows standardised mean differences below 0.1 |
| 4 Gate G1 | `tests/leak/` tests 1 to 3 of section 6 | All pass on every asset; otherwise return to the failing phase |
| 5 Gate G2 | `reports/event_study.md`: H1 and H2 versus the two analytic baselines and the three placebo families, with cluster-bootstrap intervals, Holm-adjusted p-values, the by-year table and the random-walk control | Criteria of gate G2 in section 3; the stats reviewer reproduces the figures |
| 6 Features and context | `FeatureBuilder.update()` for the 12 features; `ContextScore` with expanding-window PCA and filtered HMM; vintage-aware macro loader | Leak tests re-run and pass; no feature has more than 5% missing; context known before each session open |
| 7 M0 and M1 | Event-driven backtester with costs; walk-forward report; 27-cell robustness surface | Baseline and stress-cost results; breakdown by asset, year, timeframe and session; every grid run in the registry |
| 8 Gate G3 | M2, optional M3, CPCV with purging and embargo, uniqueness weights, PBO, DSR; shuffled-label control | Criteria of gate G3 in section 5; leak test 4 passes; the stats reviewer reproduces DSR and PBO |
| 9 Holdout | One run of the single accepted configuration on `data/holdout/`; `reports/final.md` | Run exactly once; result reported unchanged next to the research-sample result |
| 10 Signal service (after approval) | The same streaming classes fed by live bars, emitting one JSON record per signal: asset, time, level, timeframe, side, zone, target, stop, p̂, context | Replay of historical bars through the service reproduces the backtest signals exactly |

**Kickoff prompt.** "Read SPEC.md and CLAUDE.md in full. You are the orchestrator. Execute phases 0 to 9 of SPEC.md section 8 in order. For each phase, spawn one builder subagent with the phase row, the relevant SPEC sections and the standing rules; require it to return the files changed, the test output and the single reproduce command. At phases 4, 5 and 8, spawn the auditor or reviewer with a fresh context as specified. If an acceptance criterion or gate fails, stop, write reports/phase_<n>_failure.md with the numbers, and wait for me. Do not start phase 10."

## 9. Inputs needed before phase 1 (decided by the user, 2026-11-14)

- **Asset universe and data source:** crypto plus BlackBull Markets CFDs together.
  - Crypto: liquid USDT-M perpetuals, 5m bars from the Binance public archive, with true aggressor (taker-buy) volume.
  - CFDs: FX, indices, metals, energy and bonds (Bund if offered) from cTrader trendbars on the user's BlackBull account, once the read-only Open API token exists.
  - The weekly and monthly levels need 10 or more years, which the CFDs provide and crypto only partly.
- **Aggressor-signed volume:** yes for crypto (taker-buy volume), so `delta_div` is kept. CFDs have tick volume only, so `delta_div` is dropped for them, not imputed (section 4).
- **Session convention for 1h to daily bars:** 24-hour UTC for crypto; the exchange session for CFDs (e.g. the 17:00 New York roll for FX), fixed per asset in `config/baseline.yaml` before any result is seen.
- **Settled:** the detection rule is the leg-origin definition at 2.0 ATR, chosen against the simulator and the Bund 4h example.
- **Timeframe groups for balance and G2 (user decision, 2026-11-14 ~17:50 UTC, after the phase 3 balance failure on 1M):** balance (phase 3) and the G2 test family (section 3, "one per asset and timeframe group") are judged per timeframe group. There are two groups:
  - **intraday:** 15m, 30m, 1h, 2h and 4h, clustered by day;
  - **daily-and-higher:** 1d, 1w and 1M, clustered by week.

  No frozen parameter changes. Per-timeframe tables are still reported as breakdowns. The decision used covariate balance only, never outcomes. It is recorded in vecopen/reports/phase_3_failure.md and in the registry.
