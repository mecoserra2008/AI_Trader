# Research program 2026-11-10: volatility, leverage, order-flow anomalies, market state, rules

Owner: Claude (orchestrator) with 7 research agents. Registered hypotheses: `research/hypotheses.yaml`, block
"Program 2026-11-10" (27 new: H-LEV1-5, H-VM1-5, H-VB1-4, H-OF1-5, H-BO1, H-MC1, H-KF1-2, H-IC1-4). Everything
below follows CLAUDE.md principles 1-8 and docs/RESEARCH_PROTOCOL.md.

## 0. Why this program

The live work (H-MM1, H-MM2) showed that what we need most during the week is a reliable read of *how much* each
market can move and *when* it is about to: quote widths, stop distances, sizing and the LLM's "is something
happening?" judgement all depend on it. Direction has been hard (P3, P4, P6, P7, P8 are negative), volatility has been
forecastable (V1-V10). So the program goes deep where the evidence is strong (volatility, its asymmetry and its
event structure), adds detectors on the new order-flow data, closes the open trend/breakout questions from 2026-11-09
quickly, and turns every result into rules the decision layer can apply.

## 1. Data (fixed, no new ingestion; the user leads data decisions)

| set | coverage | used for |
|---|---|---|
| Bybit 1h / 4h / 1d bars | universe L: 23 crypto perps 2020-2024 to now (both segments); universe T: 16 TradFi perps from Mar-Jun 2026 (mostly holdout); ~55 contracts from 2026-11-05 (too short, excluded) | LEV, VM, VB, BO, MC, KF, IC |
| `ai_trader.features.realized` | daily RV, RS-, RS+, RQ, BV, JV, Parkinson, Garman-Klass from 1h bars; complete UTC days; known at day end | every vol study (one shared definition) |
| funding, open interest | lake, as of | LEV5, market state |
| flow_3s (3 s order-flow candles) | 106 symbols, 2026-11-07 17:46 to now | OF1-5, market state |
| liquidations | 99 symbols, 2026-11-08 to now | OF3, market state |
| alt_series | whatever is already in the lake (FRED etc.) | LEV1 robustness only |

Segments: `llm_in_sample` before 2026-07-01, `clean_holdout` after. TradFi perps and all order-flow data are
holdout-only: say so in every result row.

## 2. Workstreams and agents

| agent | hypotheses | builds | outputs |
|---|---|---|---|
| A1 vol-leverage | H-LEV1-5 | `research/leverage_study.py` | `research/reports/leverage_2026-11-10.{json,md}` |
| A2 vol-tournament | H-VM1-5 | `features/vol_panel.py` (panel HAR family, SV state-space), `research/vol_tournament_study.py` | `vol_tournament_2026-11-10.{json,md}`; the recommended default vol model |
| A3 vol-signals | H-VB1-4 | `research/vol_signal_study.py` | `vol_signals_2026-11-10.{json,md}` |
| A4 flow-anomaly | H-OF1-5 | `features/flow_anomaly.py` (online robust z, multivariate score, liquidation clusters), `research/flow_anomaly_study.py` | `flow_anomaly_2026-11-10.{json,md}` |
| A5 market-state | none (tooling) | `tools/market_state.py`, `scripts/market_report.py` | `research/reports/market_state/YYYY-MM-DD.{md,json}`, one line per token in `logs/market_state.log`, diary entry |
| A6 trend-state-space | H-BO1, H-MC1, H-KF1-2 | `research/breakout_cipher_kalman_study.py` | `trend_state_space_2026-11-10.{json,md}` |
| A7 decision-budget | H-IC1-3 (H-IC4 is forward-only, not run) | `research/decision_budget_study.py` | `decision_budget_2026-11-10.{json,md}` |
| orchestrator | synthesis | `research/RULES.md`, `research/rules.yaml`, `tests/test_rules.py`, BELIEFS.md v2 | scoreboard update, diary |

All agents run in parallel. Shared state is append-only (results JSONL, diary). Nobody edits
`research/hypotheses.yaml`, `research/BELIEFS.md`, `config/*`, Rust crates, or live processes.

## 3. Methods (house rules applied)

**Volatility models.** Targets are daily RV from 1h returns (next day, next 5 days). Loss is QLIKE (robust to noise in
the RV proxy, Patton 2011) plus MSE of log variance. Expanding windows refit every 30 days, with no look-ahead.
Contestants:
- EWMA(0.94) and the current `vol.robust_bc`;
- GARCH(1,1)-t, GJR-t and EGARCH-t (`arch`);
- HAR, log-HAR, HAR-Q, SHAR (semivariances), HAR-J (jumps) and range-HAR (GK/PK inputs);
- panel versions of each HAR: pooled slopes with symbol-specific levels, plus the crypto common vol factor;
- a stochastic-volatility state-space model: log r² = log σ² + ε, log σ² an AR(1), QML Kalman filter.

Comparison uses paired loss differences with NW t clustered by date (Diebold-Mariano), the Hansen-Lunde-Nason Model
Confidence Set (90%, block bootstrap), and Holm within each block.

**Leverage.** GJR-GARCH-t γ per asset with robust SE, and a pooled SHAR with symbol fixed effects and date-clustered
SE: b(RS⁻) − b(RS⁺). State dependence via drawdown interactions; real leverage via RS⁻ × funding z.

**Signals.** Event studies are declustered and use signed returns in σ units net of 7.5 bps per side, with SE
clustered by date and a random-entry baseline matched in count and holding period. Strategy variants are fixed before
testing, with a deflated Sharpe over the variants.

**Order flow.**
- Single-feature detectors are online: trailing 24h median and MAD, so they are causal.
- The multivariate detector is a robust Mahalanobis distance with a shrunk covariance, fitted on the first half of
  the sample and scored on the second. There's no sklearn, so it's written in numpy.
- Ratios and AUCs get block-bootstrap CIs (1h blocks).
- Every OF result is `clean_holdout`, and the forward record keeps growing.

**Decision budget.** Purged and embargoed walk-forward. The feature lists and windows are fixed in the registry.

## 4. From results to rules (what the LLM reads)

Every agent ends its report with **rule candidates**, in this schema (YAML):

```yaml
- id: R-<AREA>-<n>              # e.g. R-VOL-3
  rule: "one plain sentence: WHEN <condition> THEN <what to expect / what to prefer>"
  when: "condition in tool terms (asset_class, regime cell, sigma percentile, flow z, funding z, ...)"
  then: "expectation or preference; never a position size (sizes come from the risk engine)"
  scope: {assets: crypto|tradfi|all|<list>, horizon: "1h|1d|5d", sides: long|short|both}
  effect: "numbers with uncertainty, e.g. next-day vol +18% (t 4.1)"
  confidence: A|B|C               # BELIEFS.md grades
  evidence: [H-..., report path]
  review: "what would change it (e.g. holdout n doubles, sign flips)"
```

The orchestrator merges candidates into `research/rules.yaml`, the machine-readable source the agent prompt loads,
and `research/RULES.md`, the human view. `tests/test_rules.py` checks that every rule cites a registered hypothesis
with a recorded result, uses a known confidence grade and never states a size. Negative results become rules too
("WHEN … THEN don't expect …"). A rule is retired only through a registered test.

## 5. Market state report (per token, every day, logged)

For each token in the active universes: price and returns (1h / 24h / 7d in σ units), the v0 regime cell, vol
forecasts (current robust_bc, HAR, and the program's winner once known), vol percentile vs 1y, the latest vol surprise
(RV / forecast), the asymmetry state (drawdown flag, RS⁻ share), funding and its z-score, OI change, 24h liquidations
by side, flow anomaly flags (last 24h count and the latest), the jump / exceedance intensity, and the nearest 4h zones.
All of it is point in time, as of a timestamp. Output is Markdown and JSON in `research/reports/market_state/`, one
line per token in `logs/market_state.log`, and a diary observation.

## 6. Sequencing and checkpoints

1. Done before launch: shared realized-measures module (tests pass) and the 27 hypotheses registered.
2. Wave 1: A1-A7 in parallel. Each records every hypothesis per segment via `hypotheses.record`, including failures,
   and writes diary entries.
3. Synthesis: rules.yaml, RULES.md, the test, BELIEFS v2, the scoreboard and a summary to the user.
4. Wave 2, after the user reviews: plug the winning vol model into `tools/vol.py` and the risk engine (Rust port per
   "fit in Python, filter in Rust"), a forward test of the market-state card in decision sessions (H-IC4), and the
   order-flow forward record.

## 7. Risks to the conclusions

- **Short TradFi and flow samples:** results stay at confidence C until the holdout grows.
- **In-sample contamination:** H-LEV1, H-VM1 and H-VB1 are textbook results Claude has read, so in-sample support is
  weak evidence.
- **Multiple testing:** Holm within blocks, plus deflated Sharpe for strategy variants.
- **RV from 1h bars is noisy:** this is why QLIKE and HAR-Q are used.
