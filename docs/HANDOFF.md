# AI_Trader: handoff for projects building on top

State as of **2026-11-08 13:30 UTC**. This file is the entry point for a new project (or a new agent session) that reuses this codebase, data layout and research record. Read it together with `CLAUDE.md` (non-negotiable rules), `docs/ARCHITECTURE.md` and `research/BELIEFS.md`.

---

## 1. What this is

A forward-only crypto trading system on **Bybit** where an LLM (Claude) is the decision-maker and everything else is deterministic:

- **LLM layer**: reads state cards, picks from deterministic candidates, writes theses. Never backtested; measured only forward, after its knowledge cutoff (2026-07-01, `research.knowledge_cutoff`), against paired baselines (RULE / RANDOM books, `src/ai_trader/agent/scorecard.py`).
- **Deterministic layer**: data lake, tools, strategies, regime labeller, risk engine, execution. All point-in-time (`as_of`) and testable on history.
- **Language split**: Rust for anything always-on or touching money (ingestion, websockets, filters, risk, execution); Python for fitting, research, point-in-time reads (DuckDB/Polars) and orchestration. Pattern: fit in Python, filter in Rust, parameters handed over as JSON.

Rules that every extension must keep (full list in `CLAUDE.md`): point-in-time everything; the LLM interprets but never sizes (fixed risk per trade from the risk engine, which the LLM can't override); register hypotheses in `research/hypotheses.yaml` *before* testing; report `llm_in_sample` vs `clean_holdout`; record every result including failures; keep the append-only diary.

---

## 2. Status snapshot

| Area | State |
|---|---|
| Venue | Bybit **Demo** for the decision book (equity ~177.8k USDT, flat). Mainnet used once, for the weekend market-making experiment (below). `execution.allow_mainnet` is **false** again. |
| Live data | Order-flow collector: 70 streams (linear + spot, depth 200/1000), 3 s candles (`flow_3s`) + liquidations since 2026-11-08 00:43 UTC, 99%+ completeness. Bar updater: bars / funding / OI / DVOL / F&G every few minutes, 0 stale streams. |
| Lake | 456 MB (budget 30 GB): bars 161 MB (39 symbols, 1h back to 2017 for BTC, 2020-21 for majors), flow_3s 154 MB, trades_raw 115 MB, alt_series 5.8 MB, funding / OI / LS ratio / liquidations / instruments. |
| Decision loop | Hourly session (`ai_trader.agent.session`) builds cards + candidates (shock_regime, trend_4h), gates by the risk engine's net E[R] >= 0.10. Currently nothing clears the gate; no discretionary trades taken. |
| Research | 78 hypotheses registered, 125 result lines. Validated beliefs and refutations in `research/BELIEFS.md`, `research/PLAYBOOK.md`, `research/reports/SCOREBOARD.md`. |
| Market making | `atx mm` built and run once on mainnet (2026-11-08 02:20-12:00 UTC): SOL/XRP/ADA, minimum lots, caps 60 USD gross. Result **+0.018 USD on 142.69 USD traded (+1.27 bp), 20 maker fills, flat**. One run decides nothing (H-MM1 needs >= 4 weekends, >= 200 fills). |
| Context | Regime + narrative boards for the week of 2026-11-09 (section 8). |

Not in git (see `.gitignore`): `.env` (API keys: never commit, never print), `data/` (lake, journals, sessions, model states), `logs/`, `bin/`, `target/`, `.venv/`, `.claude/`. A new project must recreate `.env` from its own keys and re-ingest or copy `data/lake`.

---

## 3. Directory map

```
config/            settings.yaml (env, streams, live symbols, research cutoff, regimes v0), risk.yaml (risk engine),
                   schema.yaml (lake contract shared by Rust and Python; never edit while ingestion runs),
                   datasources.yaml, tools.yaml, strategies.yaml, mm.yaml (market maker)
crates/            Rust workspace
  at-core            config (two-key mainnet guard), time/Freq, Decimal rounding (format_step), shared types
  at-store           Parquet writer validated against schema.yaml (hive partitions, zstd)
  at-bybit           REST signing, websocket, market data, public trade archive, trading endpoints, lake writers
  at-alt             Fear & Greed, Deribit DVOL, DefiLlama stablecoins, FRED/ALFRED (bug: see section 9)
  at-filters         recursive filters: EWMA, GARCH-t, range vol, first passage, level chain, robust vol
  at-risk            RiskEngine: fixed-risk sizing incl. costs, daily caps, cluster cap, loss halt, net E[R] gate
  at-exec            Broker trait; BybitBroker (bracket orders, position-level TP/SL), PaperBroker
  at-flow            order-book + trade aggregation into 3 s candles (OHLCV, delta, OBI by band, OFI), collector
  atx                CLI: check, universe, instruments, klines, funding, oi, lsr, trades, alt, account, stream,
                     collect, update, propose [--submit], positions [--manage], flatten, mm, order-roundtrip
src/ai_trader/     Python package
  core               config loader, tool registry, ToolResult schema, time
  storage            Lake.read_asof (bitemporal, known_ts <= as_of, latest revision), flow rollups/maintenance
  features           transforms, vol models (EWMA / GARCH-t / robust bias-corrected / HAR), candles, profiles,
                     regime features, relative value, DC pivots
  research           frame (market_frame), regimes v0 (9 cells), calendar/weekend, evaluate (risk-unit P&L),
                     studies, verdicts, hypotheses registry, diary, and one module per study (vector, auction,
                     rv, options, vol_fund, regime_switch, trend_zoo, geo, stops)
  strategies         13 registered deterministic signal generators (+ trend_zoo.py, 21 variants, not registered)
  tools              vol.*, relval.*, regime.trend_vs_mr (registered tools the LLM calls)
  journal            JSONL journal writer (same envelope as the Rust journal)
  agent              session.py (cards, candidates, proposals), scorecard.py (forward books)
scripts/           run_research.py (playbook + verdicts), state_now.py (state cards), regime_context.py,
                   narratives.py (GDELT tracker), ai_structure.py, scoreboard.py, freshness.py, eth_week_forecast.py,
                   import_prior_projects.py, backfill_research.sh, collector.ps1 / updater.ps1 (daemons), mm_run.sh
research/          hypotheses.yaml (registry), hypotheses_results.jsonl (append-only results), BELIEFS.md, PLAYBOOK.md,
                   diary/ (append-only, one file per UTC day + index.jsonl), reports/ (one file per study),
                   notes/ (lessons from the prior ETH_BMNR and BACKTEST_TREND_FOLLOWING projects),
                   narratives/ (events.yaml labelled event log, STATE.md narrative board)
docs/              PLAN, ARCHITECTURE, CANDIDATE_GENERATION, SCHEMA, DATA_SOURCES, ORDER_FLOW, RESEARCH_PROTOCOL,
                   EXECUTION, reading_list, paper_notes/, HANDOFF (this file)
tests/             pytest suite (features, lake, flow, strategies, studies, session, stops, ...); Rust tests in crates
```

---

## 4. How to run

```bash
# setup
uv sync                                   # Python env (.venv)
cargo build --release                     # Rust; daemons run from bin/ copies so rebuilds don't hit a locked exe
cp .env.example .env  # (create it: BYBIT_DEMO_API_KEY/SECRET, BYBIT_MAINNET_API_KEY/SECRET, FRED_API_KEY, ...)

# checks
cargo test --workspace && cargo clippy --workspace --all-targets
PYTHONIOENCODING=utf-8 uv run pytest
target/release/atx check                  # connectivity + credentials for the active environment

# daemons (Windows: scripts/collector.ps1, scripts/updater.ps1; ideally Task Scheduler)
bin/atx-collector.exe collect             # order flow -> flow_3s (+ liquidations)
bin/atx-updater.exe update --loop         # bars / funding / OI / DVOL / F&G
uv run python -m ai_trader.storage.flow maintain   # daily compaction / budget

# decision session (hourly) and execution
uv run python -m ai_trader.agent.session  # cards + candidates + proposals -> data/sessions/<id>/
target/release/atx propose --file data/sessions/<id>/proposals.json [--submit]   # --submit: demo/testnet only
target/release/atx positions [--manage --breakeven]
uv run python -m ai_trader.agent.scorecard

# research and context
uv run python scripts/run_research.py     # regenerate playbook + verdicts
uv run python scripts/state_now.py BTCUSDT ETHUSDT
uv run python scripts/regime_context.py   # cross-asset regime + analogs -> research/reports/regime_context_<date>.*
uv run python -u scripts/narratives.py    # GDELT intensity/tone snapshot -> data/narratives/<stamp>/
uv run python scripts/ai_structure.py     # AI trade structure + crypto-AI betas

# market making (H-MM1)
bash scripts/mm_run.sh 2026-11-15T12:00:00Z                       # demo (settings environment)
AI_TRADER_ENVIRONMENT=mainnet AI_TRADER_ALLOW_MAINNET=1 bash scripts/mm_run.sh <RFC3339> --yes   # needs allow_mainnet: true
```

Windows notes: print-heavy Python needs `PYTHONIOENCODING=utf-8`; DuckDB 2.0 removed `.pl()` (use `pl.from_arrow(con.sql(...).to_arrow_table())`); to stop a market-making run kill `atx-mm.exe` (killing the wrapper orphans it; the wrapper flattens only if the child exits).

---

## 5. Data and point-in-time semantics

- Every lake row has `event_ts` and `known_ts`; reads return rows with `known_ts <= as_of`, latest revision per primary key (`Lake.read_asof`). Returning data after `as_of` raises `LookAheadError`.
- Parquet + zstd, hive partitions, DuckDB reader, Polars compute. Tables and keys: `config/schema.yaml`, `docs/SCHEMA.md`.
- Exchange clock runs ~1.1 s ahead of local: `known_ts` is clamped to >= candle end and event time.
- Order flow: only 3 s candles are stored (~59 B per compacted candle, ~1.7 MB per stream-day); 1 m rollups and SMA/EMA stacks are derived (`docs/ORDER_FLOW.md`).
- Sources in use: Bybit (REST + WS + public trade archive), Deribit DVOL, alternative.me F&G, DefiLlama stablecoins, options IV / funding history imported from the prior ETH_BMNR project, FRED (public CSV, latest vintage; market series are not revised), GDELT (snapshots archived per run), Google News RSS (headlines only, no archive), Yahoo chart API (equity closes for studies; not in the lake).
- TradFi perps on Bybit (XAU, XAG, CL, BZ, QQQ, NVDA, MSTR, ...) have 1h history only from March-June 2026.

---

## 6. Research record (what is believed, what failed)

**Supported / validated** (details and numbers in `research/BELIEFS.md`, `research/reports/`):
- Weekend (CME closed, Fri 17:00 ET to Sun 18:00 ET) vol is lower (H-W1); weekend realised variance is 0.2-0.45x implied (H-W7), but not tradeable as tested (H-OPT1-3 failed: Friday options already discount it, hedge costs exceed the edge).
- Post-jump vol half-life 2-3 h (H-S4). Shock playbook (`shock_regime`): on |z| >= 4, continuation if the vol state was low, reversal otherwise (H-S2/H-T1 forward-supported); simultaneous multi-coin shocks are one bet.
- Vol stack: robust winsorised EWMA x slow bias factor (`sigma_risk_next`, H-V1b) for stops and sizing; HAR on log daily RV beats EWMA at the daily horizon (H-VCL1); DVOL adds a little at 7 d (H-VCL4).
- Carry is real but decayed (H-C1; forward H-C1b).

**Not supported** (do not rebuild these without a new reason): vector candles and vector-open zones (H-VC1-5, H-VZ1-4), pivot geometry (H-GT1-3), auction / market profile (H-AU1-4), single-feature and multi-feature trend-vs-MR regime switches (H-RG1, H-TR3, H-TR4), trend model zoo out of sample (H-TR1 holdout ensemble −0.42 Sharpe; H-TR2 selection worse), relative value at taker costs (H-RV1-3), fundamentals (H-FU1/2), positioning (H-P1/2), several vol add-ons (H-VCL2/3/5, H-LC2), stop variants (H-X1: keep resting stops, H-X3).

**Forward-only, open**: H-OPT4 (US-close options entry), H-C1b, H-MM1 (market making), H-AN1/H-AN2 (historical analogs for vol / direction), H-EV1 (CPI surprise sign vs BTC window), H-F1-5 / H-AS1-2 / H-RV4 (order-flow hypotheses, testable after >= 14 days of flow_3s).

**Method lessons** (`docs/RESEARCH_PROTOCOL.md` section 3b): a holdout used for selection is in-sample; compute signals on full history, then split P&L; use scale-free metrics; per-event first-passage nulls, never a flat 0.5; run simulated nulls through the same selection code; compare ratios against the unconditional baseline; multi-leg tests need synchronous legs; size at entry with a no-trade band (sizing on sigma drift fakes costs).

---

## 7. Execution and risk

- Environments: `demo` (default; mainnet prices, simulated fills), `testnet`, `mainnet`. Mainnet needs **both** `execution.allow_mainnet: true` (settings) **and** `AI_TRADER_ALLOW_MAINNET=1`. `atx propose --submit` refuses mainnet outright; `atx flatten` and `atx mm` work on mainnet behind the guard. Mainnet state and journal are separate (`data/state/exec_day_mainnet.json`, `data/journal_mainnet/`).
- Risk engine (`config/risk.yaml`, `crates/at-risk`): 0.5% of equity at risk per trade including costs; 3 new trades/day, 5 open positions, 1 per cluster; daily loss halt 1.5%; gross <= 3x; stop <= 15%; net E[R] >= 0.10 R. Costs: taker 5.5 bp, maker 2 bp, slippage 2 bp. Minimum lots reject BTC/ETH on a 100 USD account.
- Market maker (`crates/atx/src/mm.rs`, `config/mm.yaml`): post-only minimum-lot quotes; ask 0.35 sigma_1h / bid 0.80 sigma_1h around a reservation price shifted by inventory and 5-min order flow; own-fill 60 s markouts widen or pause; quotes pushed beyond nearby 15m swing levels; requote only on > 0.3 sigma drift after 60 s; profitable inventory cleared at the touch; 45-min passive wind-down; caps 30 / 12 / 60 USD; loss limit 1.5 USD on a fill ledger (wallet equity is not a P&L measure; the execution feed lags positions by seconds).
- Keys: read + trade only, never withdrawal. Never open, copy or print `.env`, `*.pem`, key files, or the prior project's `results/live_snapshot.json`.

---

## 8. Market context at handoff (week of 2026-11-09)

- **Regime label**: equity-led cross-asset de-risking (AI/semis) with crypto in a grind-down and liquidity draining. Crypto majors R7 (1h) / R8-R9 (4h); BTC 75.9k, −8% 30 d; nearly all alts below their 30 d mean; stablecoin supply −4.6% in 30 d; DVOL 40 (not stressed); F&G 28; funding flat. `research/reports/regime_context_2026-11-08.md`.
- **Narratives** (`research/narratives/STATE.md`, `events.yaml`): AI valuations / financing peaking (VXN 38, 1y high; neocloud credit stress, CoreWeave bonds ~16%); Iran war / Hormuz mine strikes escalating in news while oil prices fall; Fed: December hike ~77% priced after +165k payrolls and claims at a 57-year low; Democrats took both chambers (resolved); yen/BOJ emerging.
- **AI structure** (`research/reports/ai_structure_2026-11-08.md`): NVDA / SMH / NDX broke weekly structure (week of Oct 16); SMH −30% with lower highs since July. Bitcoin miners turned AI landlords (CoreWeave, IREN, Core Scientific) now carry AI-equity beta; "AI tokens" are pure BTC beta.
- **Calendar**: CPI Tue Nov 10 13:30 UTC; Veterans Day Wed; claims Thu; Cisco/AMAT Thu after close, CoreWeave this week; PPI Fri; Nvidia Nov 17 after close.

---

## 9. Known issues and next steps

- `atx alt fred` requests all ALFRED vintages and fails (DGS10: 5146 > 2000 cap); needs a vintage window. Studies currently use FRED's public CSV.
- GDELT rate-limits aggressively (HTTP 429 after a few calls); `narratives.py` backs off but a run can take a long time. No archive exists yet (first snapshot pending).
- Automation is session-only: hourly decision session, MM health checks, daily narrative refresh and the CPI capture were Claude Code cron jobs that die with the session. Move the collector, updater, flow maintenance and session to Windows Task Scheduler.
- Pending decisions: the 4-sigma stop cap; a second MM run (low-vol / wide-spread book: XLM, GRAM, ADA) as a new H-MM1 config version, avoiding US-open and earnings windows.
- Optional data upgrades: live 3 s snapshot file, ADA/AVAX spot, deeper books, live OI/funding on cards, Bybit option quotes (H-OPT4), USD/JPY and equity closes in the lake, ETF flows, token unlocks / fundamentals.
- After >= 14 days of flow_3s: test H-F1-5, H-AS1-2, H-RV4; later a liquidation-conditioned shock hypothesis.

---

## 10. How the project got here (chat timeline, 2026-11-07 to 2026-11-08)

1. **Plan and architecture**: fixed-risk engine, daily trade caps, tool catalogue, priors and Bayesian updating, Student-t, stationarity, panel data, 9-regime grid, vol- and structure-driven TP/SL, no time exits, lean storage. Majority in Rust at the user's request.
2. **Build**: Bybit connectivity and execution (demo), lake schema, data sources, config, 13 strategies, research protocol and diary; up to 2 agents at a time (later 3).
3. **Order flow**: 3 s candles (OHLCV, delta, OBI by band, OFI) with derived rollups under a 30 GB budget; instrument inventory.
4. **Prior projects** (ETH_BMNR, BACKTEST_TREND_FOLLOWING) mined for lessons and data (`research/notes/`).
5. **Vol tools** (weekend-aware and jump-robust sigma, shock playbook), then a long hypothesis campaign: vector candles, the user's Pivot Geometry indicator (dropped), 4h vector-open levels, vol clustering, adverse selection, trend vs MR, auction equilibrium, fundamentals; most refuted, belief set v1 written.
6. **Visualisation**: an ETH week-ahead forecast artifact; weekend options study; live bar updater.
7. **Forward demo test**: hourly sessions on demo with a scorecard against RULE / RANDOM baselines; data-stream audit (regime, flow, OBI).
8. **Trend vs MR split** and a ranked trend model zoo (failed out of sample); collector upgraded with liquidations.
9. **Shorts and market making**: no short cleared the gate; the user chose a bearish-skewed weekend market maker on mainnet with 100 USD. Built `atx mm`, found two bugs in demo smoke tests, ran it live 02:20-12:00 UTC: flat, +0.018 USD.
10. **Universe scan** for market making (t-GARCH vol, spread net of fees, bounce): wide-spread moderate-vol tokens (XLM, GRAM, ADA) beat the lowest-vol ones.
11. **Context**: cross-asset regime label + historical analogs; narrative tracking (AI, war/oil, Fed, labour, politics, crypto); impartial macro read (strong economy feeding hike odds); AI structure break and crypto-AI linkage.

---

## 11. Building on top: conventions

- New data: add the table to `config/schema.yaml` (with ingestion stopped), write through `at-store` or `Lake.write`, keep `known_ts`. Archive anything revisable as snapshots.
- New ideas: register in `research/hypotheses.yaml` first (with prior, test, success criterion, segments), then test, then append to `hypotheses_results.jsonl` and the diary, including failures.
- New tools for the LLM: register in `ai_trader.tools` with a version; changing a tool version starts a new track record.
- New execution paths: go through the risk engine; never let model output set size; keep the two-key mainnet guard; journal every venue action.
- Keep `cargo test --workspace`, `cargo clippy --workspace --all-targets` and `uv run pytest` clean.
