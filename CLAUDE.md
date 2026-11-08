# AI_Trader

A forward-only trading system where Claude is the decision-maker. See [docs/PLAN.md](docs/PLAN.md) for goals/evaluation/roadmap, [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the system design, and [docs/reading_list.md](docs/reading_list.md) for research.

## Non-negotiable principles

1. **The LLM decision layer is never backtested.** Its performance is measured only forward, after the model's knowledge cutoff, against paired baselines on the same candidate list.
2. **Everything else is deterministic and must be testable historically**: data, tools, screener and risk engine.
3. **Point-in-time everything.** Every data and tool call takes `as_of`. Returning data timestamped after `as_of` is a bug.
4. **The LLM interprets; it doesn't compute or size.** Numbers come from tools. Position size comes from the risk engine (fixed risk per trade). The LLM can't override risk limits.
5. **Stats house rules** (ARCHITECTURE.md §1): log returns, vol-standardised, Student-t default, ADF+KPSS stationarity, HAC/clustered SEs, purged walk-forward, calibrated probabilities, linear-first complexity budget.
6. **Research protocol** ([docs/RESEARCH_PROTOCOL.md](docs/RESEARCH_PROTOCOL.md)): register hypotheses in `research/hypotheses.yaml` *before* testing; report `llm_in_sample` vs `clean_holdout` (split at `research.knowledge_cutoff`, Claude's training cutoff); record every result, including failures.
7. **Keep the diary.** Log observations, tests, decisions, mistakes and (later) trade theses in `research/diary/` via `ai_trader.research.diary` as they happen. Never edit past entries.
8. **Pin and log.** Model ID, prompt version and tool versions are logged per session; changing any of them starts a new track record. Full transcripts are journaled.

## Layout

- `config/`: `settings.yaml`, `risk.yaml`, `schema.yaml` (lake contract shared by Rust and Python), `datasources.yaml`, `tools.yaml`. Secrets only in `.env`.
- `crates/` (Rust, hot path): at-core, at-store, at-bybit, at-alt, at-filters, at-risk, at-exec, at-flow (order-flow collector), `atx` CLI.
- Order flow: only 3 s candles (`flow_3s`) are stored; 1 m rollups and SMA/EMA stacks are derived ([docs/ORDER_FLOW.md](docs/ORDER_FLOW.md)). The lake must stay under `storage.budget_gb` (30 GB). The collector runs from `bin/atx-collector.exe` (a copy), so builds don't hit a locked binary.
- Never edit `config/schema.yaml` while ingestion jobs are running: older binaries re-read it per command.
- `src/ai_trader/` (Python): `storage` (Lake.read_asof), `research` (frame, regimes v0, calendar/weekend, evaluate, studies, verdicts, hypotheses, diary), `strategies` (13 deterministic signal generators), `tools`, `journal`, `agent` (phase 4).
- `research/`: hypotheses registry + results, diary, reports. `scripts/run_research.py` regenerates the playbook and verdicts.
- `docs/`: plan, architecture, candidate generation, schema, data sources, reading list, `paper_notes/`.
- `papers/`, `reports/`, `bucket.json`: paper-critique pipeline output.
- `data/` (git-ignored): lake, journal, model states, raw cache.

## Language split

Rust for anything always-on, per-bar or touching money: ingestion, websockets, recursive filters, level chain, risk, execution. Python for fitting and research (statsmodels/arch), point-in-time reads (DuckDB) and LLM orchestration. Pattern: **fit in Python, filter in Rust**, with parameters handed over as JSON in `data/models/<tool>/`.

## Commands

- `cargo test --workspace` and `cargo clippy --workspace --all-targets` must stay clean
- `uv run pytest` (on Windows set `PYTHONIOENCODING=utf-8` when printing tool summaries)
- `target/debug/atx check` verifies all connectivity
