# Research Protocol

_How hypotheses are formed, tested, accepted and logged. Applies to Claude and to humans._

## 1. The contamination problem, applied to research

Claude can backtest *deterministic* strategies. But the **ideas** Claude proposes are shaped by its training data: papers, forum lore and post-mortems about the very periods we test on. If Claude "discovers" that weekend gaps fill in 2021–2025 data, that may only be recall of what traders wrote about 2021–2025.

So every result is reported in two segments, split at `research.knowledge_cutoff` (2026-07-01):

| segment | meaning | weight as evidence |
|---|---|---|
| `llm_in_sample` | before the cutoff: Claude may have read about this period | weak; necessary but not sufficient |
| `clean_holdout` | after the cutoff: Claude cannot know what happened | strong; grows every day |

A hypothesis is **supported** only if it holds in-sample *and* keeps its sign in the holdout. Holdout samples are small at first, so most early verdicts will be "inconclusive", and that's the honest answer.

## 2. Lifecycle

1. **Register first.** Add the hypothesis to [research/hypotheses.yaml](../research/hypotheses.yaml): statement, rationale, test, success criterion, prior probability. Do this *before* looking at the result.
2. **Test.** Use `ai_trader.research.studies` / `evaluate`. Deterministic code only; parameters come from `config/strategies.yaml` and are never tuned on the full sample.
3. **Record.** `ai_trader.research.hypotheses.record(id, segment, verdict, metrics)` appends to `research/hypotheses_results.jsonl`. **Failures are recorded too.** An unrecorded failed test is data snooping.
4. **Diary.** Log what was tried, what was surprising and what it changes (§4).
5. **Promote.** A supported hypothesis can become a candidate tool or strategy prior (playbook), still subject to the forward evaluation of PLAN.md §5.

Changing a hypothesis after seeing results means registering a **new** id (e.g. H-W2b) and counting it as an additional trial.

## 3. Statistical standards (ARCHITECTURE §1)

- P&L is measured in **risk units** (position = signal / σ̂, fixed at entry), net of costs.
- t-stats are **Newey–West**. Cross-symbol pooling averages across symbols *per timestamp* (clustering by time), because crypto moves together.
- Effective sample size = **regime episodes**, not bars. Cell estimates are shrunk toward the pooled mean: `w = episodes / (episodes + κ)`.
- Null models must be explicit (e.g. the reflection-principle baseline for gap fills, not "50%").
- Survivorship: the research universe is today's survivors, so cross-sectional claims carry that bias (docs/DATA_SOURCES.md).
- Multiple testing: count every hypothesis × segment × variant tested. Treat |t| ≈ 2 as interesting, not proven.

## 3b. Lessons from prior projects (research/notes/SYNTHESIS_prior_projects.md)

- **A holdout used for selection is in-sample.** Automated or agent research loops may select variants only on `llm_in_sample`. `clean_holdout` is read once per registered hypothesis.
- **Deflate anything chosen from more than 5 variants** (deflated Sharpe, trial count in the record).
- **Compute signals on the full history, then split P&L by segment.** Truncating the frame first restarts indicator warm-up and silently empties the holdout.
- **Metrics compared across frequencies must be scale-free** (Sharpe, IC), never raw P&L in bar-risk units.
- **Multi-leg instruments need synchronous legs and worst-case intrabar ordering.**
- **Execution simulation is optimistic by default.** Treat maker fills and resting stops as hypotheses (H-F5, H-X1), not assumptions.

## 4. The diary

`research/diary/YYYY-MM-DD.md` (readable) + `research/diary/index.jsonl` (machine). Written through `ai_trader.research.diary.entry()` or

```bash
uv run python -m ai_trader.research.diary add --kind observation --title "..." --body "..." --tags weekend --hyp H-W2
```

Entry kinds:

| kind | use it for |
|---|---|
| observation | something seen in the data, before interpreting it |
| hypothesis | a new idea (then register it) |
| test | what was run, with what parameters |
| result | what came out, with numbers and segment |
| decision | a design or trading choice and the reason for it |
| trade | (live phase) the thesis, the evidence and what would invalidate it, written *before* the outcome |
| reflection | after outcomes: what was right, what was luck, what to change |
| mistake | bugs, bad assumptions, look-ahead found. These are the most valuable entries |
| question | open issues for the human |

Rules: write the entry when the thing happens, not afterwards. Record numbers with their segment. Never edit past entries: add a new one that corrects the old.
