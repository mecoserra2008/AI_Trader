# AI_Trader — System Plan

_Started 2026-11-07. Living document: update it as decisions get made._

## 1. The problem in one paragraph

The decision-maker is an LLM (Claude). An LLM has two problems that make backtesting it meaningless:

1. **Non-determinism.** The same inputs can give different decisions, so a historical run can't be reproduced.
2. **Contamination by its training data.** The model has seen what happened after any historical date before its knowledge cutoff. Any "backtest" before the cutoff is look-ahead biased, even when prices are masked or names anonymised. Several papers below show that anonymisation only partly helps.

So the AI's decisions can only be evaluated **forward, in live (paper or real) conditions, on dates after the model's knowledge cutoff.** The rest of the system is built to make that forward evidence clean, auditable and statistically useful as fast as possible.

## 2. Core design principle: split deterministic code from judgment

| Layer | What it is | Backtestable? |
|---|---|---|
| **Data layer** | Point-in-time market and alternative data, every record stamped with `as_of` | Yes |
| **Tool layer** | Deterministic functions the AI calls: technical analysis, regime detection, alt-data models, screeners | **Yes, and they must be validated historically** |
| **Judgment layer** | The LLM: reads tool outputs, forms a thesis, ranks candidates, picks trades | **No. Forward only.** |
| **Risk and execution engine** | Fixed per-trade risk, trade caps, stops, sizing, order routing | Yes, as deterministic replay |
| **Journal and evaluation** | Every prompt, tool call, output, decision and fill; scored against baselines | n/a |

The LLM is never the source of numbers. It **interprets** numbers that the tools produce from point-in-time data. The LLM never sizes positions and never overrides risk rules.

## 3. Architecture

> **Detailed design lives in [ARCHITECTURE.md](ARCHITECTURE.md)**: statistical rules, belief state & priors, tool catalogue, tool selection, lean storage. The sketch below is the high-level flow only.

```
            ┌────────────────────────── scheduled session (e.g. pre-open, fixed times) ─────────────┐
            │                                                                                       │
 Data layer ─► Screener (deterministic) ─► candidate list (top M)                                    │
 (as_of)         │                              │                                                    │
                 ▼                              ▼                                                    │
            Tool layer  ◄──── tool calls ────  LLM agent (Claude, pinned model + prompt version)     │
            - TA / levels                       1. regime assessment (market + asset)                │
            - regime model                      2. per-candidate analysis via tools                  │
            - alt-data models                   3. rank ALL candidates, pick ≤ N                     │
            - news/sentiment (timestamped)      4. emit Decision JSON (schema-validated)             │
                                                │                                                    │
                                                ▼                                                    │
                                   Risk engine (hard, deterministic)                                 │
                                   - fixed risk R per trade → size = R / stop distance               │
                                   - max N new trades/day, max open positions                        │
                                   - exposure / correlation / sector caps                            │
                                   - daily loss limit → halt for the day                             │
                                   - mandatory stop, no averaging down                               │
                                                │                                                    │
                                                ▼                                                    │
                                   Execution (broker API, bracket orders) ─► fills                  │
                                                │                                                    │
                                                ▼                                                    │
                                   Journal ─► Evaluation vs baselines (same candidate list)         │
            └───────────────────────────────────────────────────────────────────────────────────────┘
```

### Decision JSON (draft contract between the AI and the engine)

```json
{
  "session_id": "2026-11-09T13:00Z",
  "model_id": "claude-opus-5-5", "prompt_version": "v0.1", "tool_versions": {...},
  "regime": {"label": "trend_up", "confidence": 0.7, "evidence_tool_calls": ["..."]},
  "candidates_ranked": [
    {"symbol": "XYZ", "score": 0.82, "side": "long", "setup": "pullback_in_trend",
     "entry": {"type": "limit", "price": 101.2}, "invalidation": 97.8,
     "thesis": "...", "key_evidence": ["tool_call_id", "..."], "take": true}
  ]
}
```

The AI ranks **every** candidate, not only the ones it takes. That matters for evaluation (section 5).

## 4. Handling non-determinism

- **Pin everything.** Model ID, temperature, system-prompt hash and tool versions are logged per session. Changing any of them starts a **new track record**, because it is a different strategy.
- **Log the full transcript.** Store every tool input and output and the raw model response, so any decision can be audited later.
- **Optional self-consistency.** Run the decision K times and only take trades the majority agrees on. Treat disagreement as low confidence, which means no trade. This costs K× the API spend, so it's a later experiment.
- **Fixed schedule.** Decisions happen at pre-defined times, never ad hoc. This stops the operator from cherry-picking sessions.

## 5. Evaluation: getting evidence fast in a forward-only world

Forward testing is slow, so the evaluation design has to squeeze maximum information out of each session.

1. **Paired baselines on the same candidate list.** Each session, deterministic strategies trade the **same shortlist** under the **same risk engine**, on paper:
   - `RANDOM_N`: N random picks from the shortlist. This is the key control: AI minus random equals the AI's selection skill.
   - `RULE_N`: top N by the screener's own score. This shows whether the LLM adds anything beyond the rule.
   - `NO_TRADE` / buy-and-hold benchmark.

   Because the baselines share the same market noise, the paired difference has far lower variance than raw P&L.
2. **Shadow outcomes for untaken candidates.** Every ranked candidate gets a hypothetical outcome, measured in R-multiples from the proposed entry and invalidation. This gives M data points per session instead of N. AI ranking skill can then be measured as the rank correlation (IC) between the AI's score and the realised R.
3. **Pre-registration.** Before going live, write down the metrics, minimum sample size, success threshold and kill criteria. Don't move the goalposts afterwards.
4. **Sample-size reality check.** With per-trade R-multiple std σ ≈ 1.2R, detecting a mean edge μ at t ≈ 2 needs n ≈ (2σ/μ)² trades:

   | Edge per trade | Trades needed | At 3 trades/day |
   |---|---|---|
   | 0.30R | ~64 | ~1 month |
   | 0.20R | ~144 | ~2.5 months |
   | 0.10R | ~576 | ~9 months |

   Shadow outcomes plus paired baselines are how this gets shortened.

## 6. Roadmap

| Phase | Goal | Exit criterion |
|---|---|---|
| **0. Research and decisions** | Reading list through the paper pipeline; settle the open decisions (section 7) | Universe, timeframe, broker and risk params fixed |
| **1. Data layer + journal schema** | Point-in-time store; every query takes `as_of` | No query can return data timestamped after `as_of` (tested) |
| **2. Tool layer** | TA, regime model, screener, first alt-data model | Each tool unit-tested and **historically validated**, since tools are deterministic |
| **3. Risk and execution engine** | Sizing, caps, halts, bracket orders, paper broker | Deterministic replay passes; baselines run daily on paper |
| **4. Agent integration** | Claude API tool use (or an MCP server exposing the tools); Decision JSON validation | 2+ weeks of dry runs: decisions logged, no orders |
| **5. Forward paper trading** | AI plus baselines live on paper | Pre-registered sample reached; evaluate |
| **6. Small live capital** | Only if phase 5 passes | Ongoing |

## 7. Open decisions (yours to make)

1. ~~Asset universe~~ → **decided:** Bybit USDT linear perpetuals, crypto only (`universe.symbol_types`). TradFi perps optional as context.
2. **Timeframe.** Intraday or swing (multi-day)? LLM latency and cost favour swing or daily decisions. Intraday raises cost and execution complexity.
3. ~~Broker~~ → **decided:** Bybit. Forward paper testing on **Bybit Demo Trading** (mainnet prices); testnet only for connectivity.
4. **Risk parameters.** Risk per trade (e.g. 0.25–1% of equity), max trades/day N, max open positions, daily loss limit.
5. **Shortlist size M.** Bigger M gives more shadow data but more tokens per session.
6. **API budget per day.** This sets how many tool calls and how much self-consistency are affordable.
7. **Alternative data.** Which sources: news, filings, on-chain, options flow, sentiment? Each needs a point-in-time archive and a model.

## 8. Known risks to the plan itself

- **Tool-output leakage.** A tool that uses revised data (macro revisions, survivorship-biased universes, back-adjusted fundamentals) reintroduces look-ahead even in the tool layer. Point-in-time discipline is enforced in code, not by convention.
- **Model deprecation or upgrade.** It resets the track record. Plan for it: run the old and new model in parallel during transitions.
- **Operator interference.** Manual overrides contaminate the record. Log any manual action as an explicit event.
- **Regime dependence.** A few months of forward data covers few regimes. Report results conditioned on the regime label.
