# Candidate Generation: Beyond the Screener's List

_v0.1, 2026-11-07. Answers: should the LLM be able to trade assets outside the screener's top-M list? If yes, in which situations, and which models should generate those candidates?_

---

## 1. Conclusion first

**Yes, candidates may come from outside the main list, but never from the LLM's memory.** Every candidate must arrive through a **deterministic generator ("channel")** with a provenance tag. The LLM may:
1. **promote** candidates from any channel's short list into full analysis, and
2. issue **hypothesis requests** drawn from a fixed whitelist of query templates (e.g. "laggards in cluster X vs its leader"), which a generator executes.

Free recall ("I think SOL looks good") is **not tradeable**. It's logged in a shadow-only channel so its value can be measured forward without risking capital.

This keeps three properties the evaluation depends on:
- **Provenance:** every trade can be traced to a channel.
- **A control per channel:** "random pick from the same channel" is always computable.
- **Point-in-time reproducibility:** each generator is deterministic code over `as_of` data.

---

## 2. When would an off-list candidate matter? (situations)

| # | Situation | Why the main screener misses it |
|---|---|---|
| S1 | **Screener blind spot**: asset just became liquid, recently listed, or failed a filter for a stale reason | Filters and rankings are built on history the asset doesn't have yet |
| S2 | **Relational spillover**: a move in a leader or cluster implies a move in a laggard or related asset | The screener scores assets one at a time; relations aren't in the ranking |
| S3 | **Catalyst before price**: news, attention or flow shock before the price has moved | A price/structure-based ranking only reacts after the move |
| S4 | **Regime transition**: e.g. R4 compression about to break, or the market cell flipping | Ranking weights are tuned to the *current* cell; the asset matters under the *next* one |
| S5 | **Better expression of a view**: the thesis is on a sector or factor, and a different instrument (leader, index proxy, pair spread) expresses it more cleanly | The screener ranks single instruments, not views |
| S6 | **Scheduled events**: unlocks, earnings, expiries, macro releases | Events are known in advance but don't affect the price-based score yet |
| S7 | **Cross-sectional dislocation**: asset extreme vs its own cluster on a priced characteristic | Absolute ranking misses *relative* extremes in quiet clusters |
| S8 | **Open-position management**: a held asset dropped out of today's list | Not a new candidate, but it must still be re-scored (ARCHITECTURE §3.12) |

S8 is already handled by re-scoring open positions every session. S1–S7 need generators.

---

## 3. Candidate generator models: assessed and ranked

### 3.1 Scoring dimensions (1 = poor, 5 = excellent)

- **OF**: resistance to overfitting. Free parameters vs effective sample size, multiple-testing exposure, and whether a target is fitted at all.
- **EFF**: computational and memory efficiency within the lean budget (ARCHITECTURE §6).
- **DATA**: fit to our data constraints: short histories, few regime episodes per cell (9 regimes, §3.2), point-in-time availability.
- **NS**: robustness to non-stationarity and regime shifts.
- **AUD**: auditability. Can it be validated historically, and can a same-channel random control be built?

Composite = 0.35·OF + 0.15·EFF + 0.15·DATA + 0.20·NS + 0.15·AUD. Overfitting gets the largest weight because it's the failure mode we can least detect in a slow forward test.

### 3.2 Ranking

| Rank | Generator | Situations | OF | EFF | DATA | NS | AUD | **Score** | Verdict |
|---|---|---|---|---|---|---|---|---|---|
| 1 | **G3 Surprise/anomaly detection**: robust z-scores of residual return (vs factor model), volume, attention, flow; t-tail probabilities | S1, S3, S7 | 5 | 5 | 4 | 4 | 5 | **4.65** | **Adopt** |
| 2 | **G1 Rule screens**: vol-adjusted momentum rank, breakout, level proximity, liquidity | S1, S4 | 5 | 5 | 5 | 3 | 5 | **4.60** | **Adopt** (already the base filter) |
| 3 | **G2 Quant-posterior ranking**: E[R] from `pred.barrier` (level chain) | main list | 4 | 4 | 3 | 4 | 5 | **4.00** | **Adopt** (= main list) |
| 4 | **G4 Cross-sectional factor score**: IC-weighted composite / Fama–MacBeth fitted score, within cluster | S7, S5 | 4 | 5 | 3 | 3 | 5 | **3.95** | **Adopt** |
| 5 | **G6 Event/catalyst hazard**: pooled discrete-time hazard on scheduled events | S6 | 3 | 5 | 3 | 3 | 4 | **3.45** | **Probation** (needs a point-in-time event calendar) |
| 6 | **G7 Regime-transition screen**: assets with high posterior of leaving the current cell (esp. R4 → R1/R2/R3, R5 → R6) | S4 | 2 | 5 | 3 | 3 | 4 | **3.10** | **Probation** (transitions are rare, so small n_episodes) |
| 7 | **G5 Relational propagation (constrained)**: lead-lag and cointegration **within clusters only**, BH-FDR across pairs, stable in 2 non-overlapping windows | S2, S5 | 3 | 3 | 3 | 2 | 4 | **2.95** | **Probation**, disabled in R9 (correlations → 1) |
| 8 | **G8 Analog / kNN state matching**: assets whose current state (≤ 6 dims) resembles states with strong historical outcomes | S3, S4 | 3 | 3 | 3 | 2 | 4 | **2.95** | **Research only** |
| 9 | **G12 Text-embedding / co-mention graph** | S2, S3 | 3 | 3 | 2 | 3 | 2 | **2.70** | **Research only** (pretrained embeddings are contaminated for historical validation) |
| 10 | **G9 Unsupervised clustering of current states** (GMM) → members of "good" clusters | S4, S7 | 2 | 4 | 3 | 2 | 3 | **2.60** | Research only (unstable clusters, label switching) |
| 11 | **G13 LLM free recall** | all | 1 | 4 | 5 | 3 | 1 | **2.45** | **Shadow only**: no trades, forward-measured |
| 12 | **G10 Gradient-boosted panel ranker** (LambdaMART) | all | 2 | 3 | 2 | 2 | 3 | **2.30** | Reject for now (data-hungry; revisit after ≥ 2y of point-in-time panel) |
| 13 | **G11 Deep sequence models** (LSTM/Transformer) | all | 1 | 1 | 1 | 1 | 2 | **1.15** | Reject |

### 3.3 Reasoning behind the scores

- **G3 ranks first because it fits no target.** It flags *where something unusual is happening* (heavy-tail-aware, using t-distribution tail probabilities on standardised residuals) and leaves *direction and edge* to the quant posterior and the LLM. With no outcome fitted, there is almost nothing to overfit. Its EWMA baselines also adapt to drift.
- **G1 and G2 are robust but backward-looking.** They miss S2/S3/S6 by construction. That's exactly why extra channels are justified.
- **G4 depends on cross-section size.** It needs roughly ≥ 30 assets per cross-section for stable coefficients. Applied *within cluster*, it captures relative dislocations (S7). `panel.coef_drift` guards against factor-premium rotation.
- **G6/G7 are useful but sample-starved.** Events and regime transitions are rare. Spread over 9 regime cells, per-cell episode counts get small fast. They're usable only with pooled panels and the additive shrinkage of ARCHITECTURE §3.2.
- **G5 is the classic overfitting trap.** Searching N² pairs for lead-lag or cointegration guarantees spurious hits. It's only admissible when restricted to within-cluster pairs, FDR-controlled and stability-checked. In R9 (liquidation), correlations converge to 1 and relational signals stop working, so the channel is switched off there.
- **G8/G9 have hidden hyperparameters:** the distance metric, k, feature choice and cluster count. A curse of dimensionality also sets in beyond about 6 dimensions. They can be revisited once the deterministic layer has years of point-in-time history.
- **G13 fails on auditability, not intelligence.** The LLM's recall isn't reproducible, can't be controlled against a random baseline, and leans toward popular, heavily discussed names (popularity bias). Its value is real but **unmeasurable in the trading record**. Hence shadow mode: it's logged and outcome-tracked, but never traded.
- **G10/G11:** with non-stationary, fat-tailed returns and a limited effective sample, they fit noise. They also break the "linear first" complexity budget (ARCHITECTURE §1.6).

> The scores are a pre-registered prior, not a fact. Each channel's realised skill (shadow outcomes, §5) updates it. The same skill posterior that manages tools (ARCHITECTURE §5) promotes or retires channels.

---

## 4. Resulting design

```
                ┌── C-MAIN  : G1 filter → G2 E[R] ranking ............ top M (≈15–20), full cards
 universe ──────┼── C-ANOM  : G3 surprise detection .................. top 5, one-line cards
 (as_of)        ├── C-XS    : G4 within-cluster factor dislocation .... top 5, one-line cards
                ├── C-EVENT : G6 catalyst hazard        [probation] ... top 3
                ├── C-TRANS : G7 regime-transition      [probation] ... top 3
                ├── C-REL   : G5 constrained relational [probation] ... top 3 (off in R9)
                └── C-LLM   : G13 free recall           [shadow only] . logged, never traded
                                         │
                        LLM: full analysis of C-MAIN
                             + may PROMOTE ≤ 3 off-list candidates to full analysis
                             + may issue ≤ 2 HYPOTHESIS REQUESTS (whitelisted templates)
                                         │
                        every candidate carries {channel, generator_version, rank_in_channel}
```

**Hypothesis-request templates (initial whitelist):**
- `laggards(cluster, leader, min_gap_sigma)`: cluster members lagging the leader's move by ≥ x σ
- `in_regime(cells=[...], min_posterior)`: assets currently in given regime cells
- `near_level(type, max_dist_sigma, break_prob_range)`: assets approaching strong levels
- `event_within(type, days)`: assets with a scheduled event
- `xs_extreme(feature, cluster, pct)`: cross-sectional extremes within a cluster

Each template is deterministic and versioned. Its results count as candidates of channel `C-REQ:<template>`.

**Promotion budget.** At most 3 off-list promotions and 2 hypothesis requests per session. This keeps token cost bounded and limits the "garden of forking paths" inside a session.

---

## 5. Evaluation of channels

- **Shadow outcomes for every channel candidate** (traded or not), via the same structural TP/SL placement and level chain. This gives each channel a forward record of its own.
- **Per-channel random control:** traded off-list candidates are compared against random picks from the same channel.
- **LLM selection skill per channel** = Brier/log-loss of `p_final` vs `p_quant` on that channel's candidates. Does the LLM add value on off-list names, or only on the main list?
- **C-LLM (shadow):** compare free-recall candidates' outcomes to the C-MAIN and C-ANOM baselines. If free recall consistently beats them, design a deterministic generator that captures what it was seeing. Don't let it trade directly.
- **Multiple testing:** each channel and each template counts as a trial in `research/trials.jsonl`. Promotion thresholds use FDR-adjusted skill.
- **Regime dependence:** channel skill is tracked per regime cell with additive shrinkage (direction + vol main effects), because channels are expected to differ by cell (e.g. C-ANOM in R4/R6, C-REL off in R9).
