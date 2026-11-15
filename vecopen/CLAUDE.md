# vecopen: standing rules (from SPEC.md section 8, verbatim)

1. `SPEC.md` and `config/baseline.yaml` are read-only. A change to either requires the user's written approval and a new registry entry.
2. `LevelRegistry`, the feature builder and the signal generator expose a streaming API only: `update(bar) -> outputs`. Batch runs are loops over that API. No function may take a full future-inclusive frame and return per-bar outputs.
3. Every run appends one line to `trials/registry.jsonl` with config hash, git commit, UTC time, phase and metrics. Lines are never edited or deleted.
4. `data/holdout/` is not read, listed or sampled before phase 9.
5. No threshold, feature, barrier or model hyperparameter is tuned on test folds. Hyperparameters are chosen inside training folds only.
6. A failed gate ends the run. Do not adjust anything to make a gate pass; write the failure report.
7. Every number in a report is produced by a script in the repo and reproducible from a single command stated in that report.
8. Random seeds are fixed in the config.

## Also binding here (from the parent repo, ../CLAUDE.md)

- Never open, print or copy secrets: `.env`, `*.pem`, `keys/`. cTrader credentials are read at runtime by `atx ctrader` (crates/at-ctrader) and are read-only scope; no order of any kind is sent from this project.
- Point-in-time everything; a timestamp after `as_of` is a bug.
- Storage: the whole repo's lake plus research data stays under 30 GB (about 6 GB used on 2026-11-14). Keep raw downloads transient and record them in a manifest so nothing is downloaded twice.
- Resources: at most 2 threads and about 1 GB of RAM per agent (`POLARS_MAX_THREADS=2`, `OMP_NUM_THREADS=2`); the user runs other heavy apps.
- Never touch `config/schema.yaml` of the parent repo, the mainnet settings, `bin/` executables, or SPYUSDT.
- Kill only your own processes, by PID. No scheduled tasks.
- Gate verdicts (G2: H1/H2) are also recorded in the parent research registry as H-VO1 / H-VO2 (`ai_trader.research.hypotheses.record`) by the orchestrator, and a diary entry is written per phase via `ai_trader.research.diary`.
- Environment: run with `uv run` from the repo root, using the `vecopen` optional dependencies (`uv run --extra vecopen ...`). Python is 3.13, not the 3.11 named in the PDF; this deviation is recorded in SPEC.md.
- "Small improvements" during the build mean engineering (speed, tests, clarity, data quality) or a logged trial with its own registry line. They never mean retuning a frozen parameter, and never adjusting anything after seeing a gate result (rule 6).
