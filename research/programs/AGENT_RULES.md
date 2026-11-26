# Rules for every research agent of the specification program (2026-11-27)

Read this, CLAUDE.md, docs/RESEARCH_PROTOCOL.md, docs/ARCHITECTURE.md §1 and research/programs/2026-11-27_specification_program.md
(your workstream section) before doing anything. Up to 20 agents run at the same time on one machine: follow every rule.

## 1. Hard safety rules (no exceptions)
- Research only. Never place, amend or cancel orders; never call Bybit private/trading endpoints; never run `atx live`, `atx mm`,
  `atx scalp --mode live`; never touch SPYUSDT; never open .env, keys/, *.pem or any secret (loaders read keys at runtime).
- Never edit config/schema.yaml, config/risk.yaml, research/rules.yaml (rule candidates go in your report as suggestions).
- Never kill, restart or touch processes you did not start. Desk processes (collector, updater, level_watch, tape, xvenue,
  watchdog, live_nn, vol_ml, xs_v4, dashboard, desk_collect) and the user's STOCK SCANNER / Chrome / TradingView are off limits.
- The cTrader MCP only through src/ai_trader/ctrader/mcp_ro.py (data). No X/Twitter/Facebook scraping; FinancialJuice must not be scraped.
- vecopen holdout (vecopen/data/holdout) must never be read. Never write account balances anywhere.
- The LLM decision layer is never backtested (CLAUDE.md principle 1).

## 2. Compute: always through the governor
- The machine has 31.6 GB RAM but typically only 3-6 GB free (the user's apps run). CPU 20 threads shared.
- Every job that runs longer than ~30 s or may use more than ~300 MB MUST run through the governor:
  `.venv/Scripts/python.exe scripts/ops/governor.py run --mb <cap> --slots heavy --name <WS id> -- <command>`
  (light pool for tests and small scripts: `--slots light`). Caps: at most 4000 MB and 4 threads per job; default 2 threads.
  The governor waits for RAM and a slot, caps BLAS/Polars/Torch threads and hard-limits memory (allocations past the cap fail).
- Design for the cap: stream / chunk data (DuckDB / Polars lazy, per-symbol, per-day), float32 where it is safe, no full
  in-memory copies of the lake. Ask for more memory only when you have measured the need.
- Expect to wait in the queue; do other work (reading, writing, design) while a job waits. Never bypass the governor.
- Rust: avoid unless essential. If you must build, use CARGO_TARGET_DIR=target/agent_<ws> and build through the governor (--mb 3000).
- Deep learning on CPU only; small models; set torch.set_num_threads(int(os.environ.get("TORCH_NUM_THREADS", 2))).

## 3. Storage
- Total budget for data/ (lake + research + raw + models) is 30 GB; about 15 GB is free. Your quota is in your workstream
  section (default 300 MB). Outputs go under data/research/<ws_id_lowercase>/. Raw downloads go under data/raw/<ws_id>/ and
  are deleted after processing (transient raw <= 1 GB per agent). Model weights (e.g. foundation models) go under
  data/models/hf_cache/ and count against your quota. Check your usage (du -sh) before and after each large step.
- Reuse existing data before downloading. Never duplicate the lake.

## 4. Shared files: always under a named lock
- Registry: write your hypotheses block to research/hypotheses_pending/<WS>.yaml (same format as research/hypotheses.yaml,
  ids H-<WS>-1, H-<WS>-2, ...), then append it with
  `.venv/Scripts/python.exe scripts/ops/governor.py lock registry -- .venv/Scripts/python.exe scripts/ops/append_hypotheses.py research/hypotheses_pending/<WS>.yaml`
  BEFORE any fit on the data that tests them. The script refuses duplicate ids.
- Results: `hypotheses.record(...)` (ai_trader.research.hypotheses) and diary entries (ai_trader.research.diary.entry) must run
  inside `governor.py lock results -- ...` and `governor.py lock diary -- ...` respectively (or call them from one small
  script run under the lock).
- Git: stage only your own files and commit/push under the repo lock:
  `.venv/Scripts/python.exe scripts/ops/governor.py lock repo -- bash -c "git add <your files> && git commit -m '...' && git push -q origin main"`.
  Messages end with "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>". Scan your staged diff for secrets / keys /
  PII / balances first. Never commit raw data, model weights or large outputs (> 5 MB).
- Do not edit other workstreams' modules, reports or data folders. Your code lives in src/ai_trader/research/<ws_module>.py
  (+ tests/test_<ws_module>.py). Keep `uv run pytest` passing for your tests (run them through the light pool).

## 5. Method: the specification dossier (the core deliverable)
For every target variable you model, write a dossier section with:
1. The variable and why we model it: what decision or risk it feeds (stops, sizing inputs, regime gates, entries, costs).
2. Its empirical properties, measured point-in-time: stationarity (ADF + KPSS), distribution (moments, tails, QQ), serial
   dependence (ACF / Ljung-Box on levels and squares), heteroskedasticity, regimes / breaks, sampling and missing data.
3. Candidate specifications, each with its assumptions stated explicitly (distribution, dependence, stationarity, linearity,
   parameter stability), why it is plausible for THIS variable, and what it would imply if true.
4. Assumption checks for each spec on fit data (residual diagnostics, PIT / probability integral transform, Berkowitz,
   KS/AD, tail tests, stability tests such as CUSUM / Andrews sup-F / rolling estimates).
5. Out-of-sample comparison with proper tests: proper scoring rules (log score, CRPS, pinball, QLIKE for variance),
   Diebold-Mariano with HAC (and the small-sample correction), the Model Confidence Set (Hansen-Lunde-Nason) when comparing
   many specs, Hansen SPA / White Reality Check / Romano-Wolf for many strategy variants, calibration (coverage, PIT
   uniformity), deflated Sharpe and PBO for any selection from a grid.
6. How different the conclusions are across specs (does the decision change? by how much?), and which specs can be trusted
   and why (tests passed, stability, simplicity, out-of-sample record). Say clearly when the data cannot distinguish them.
7. Power: the minimum detectable effect for the tests you ran; say when a null result is just underpowered.

## 6. Protocol (CLAUDE.md principles 3, 5, 6)
- Point-in-time always (known_ts <= t; ALFRED vintages for revised macro series). Purged / embargoed walk-forward splits.
- Register hypotheses with success rules BEFORE testing; freeze the design in a diary entry ("<WS> design frozen before
  results"); record every result, including failures and inconclusive ones; segments llm_in_sample vs clean_holdout split at
  research.knowledge_cutoff where both exist. Any change after seeing results is a new, labelled, extra trial.
- Complexity budget: linear / classical first; a complex model must beat the best simple one out of sample with a proper test.
- Costs: crypto maker 2.0 bp, taker 5.5 bp, pegged-limit entry 2.0 + 1.3 bp, stop slippage; any "edge" is reported net.

## 7. Report
research/reports/<ws_id_lowercase>_2026-11-27.md: plain-words verdicts first (what we now believe, what changed, what to
use and what to stop using), then the dossiers, then caveats, then method. Final reply to the orchestrator: the verdicts with
numbers and tests, which specs to trust, what is still unknown, data used, storage before/after, runtime and governor waits,
and everything that went wrong. Do not substitute a different design silently.
