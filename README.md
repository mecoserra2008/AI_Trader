# AI_Trader

A forward-only trading system in which Claude is the decision-maker. Everything except the LLM is deterministic and validated on history. The LLM is evaluated only live, against paired baselines.

Design: [docs/PLAN.md](docs/PLAN.md) · [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) · [docs/CANDIDATE_GENERATION.md](docs/CANDIDATE_GENERATION.md) · [docs/SCHEMA.md](docs/SCHEMA.md) · [docs/DATA_SOURCES.md](docs/DATA_SOURCES.md)

## Layout

```
config/            settings.yaml, risk.yaml, schema.yaml (lake contract), datasources.yaml, tools.yaml
.env               secrets (copy from .env.example; git-ignored)
crates/            Rust: hot path
  at-core          time/freq, look-ahead guard, domain types, config + secrets, exact tick/lot rounding
  at-store         bitemporal Parquet lake writer (validates against schema.yaml)
  at-bybit         Bybit v5: REST (signed), WebSocket, market data, trade archive -> flow bars, trading
  at-alt           alt data: Fear & Greed, Deribit DVOL, DefiLlama stablecoins, FRED/ALFRED vintages
  at-filters       O(1) filters: EWMA, GARCH-t, range vol, first passage, level chain
  at-risk          deterministic risk engine: fixed R incl. costs, caps, halts, net E[R] gate
  at-exec          Broker trait: Bybit broker + conservative paper broker (baselines)
  at-flow          live order-flow collector: L2 books + trades -> 3 s candles (delta, OBI by band, OFI)
  atx              command line
src/ai_trader/     Python: research, fitting, point-in-time reads, LLM layer
  core             config, time, ToolResult contract, tool registry
  storage          Lake.read_asof (DuckDB), Lake.write
  features         returns, vol standardisation, ADF+KPSS, fractional differencing, PSI
  tools            vol.ewma, vol.garch_t (fit -> JSON params for the Rust filter)
  journal          append-only session journal
  agent            (phase 4)
tests/             Python tests (look-ahead invariance, PIT revisions, cross-language reads, features)
data/              lake, journal, model states, raw cache (git-ignored)
```

## Setup

```bash
uv sync                      # Python 3.13 env in .venv
cargo build --release        # Rust workspace -> target/release/atx
cp .env.example .env         # then fill in keys (all optional for public data)
```

Keys worth getting now (free): **Bybit Demo Trading** API key (`BYBIT_DEMO_*`), **FRED** (`FRED_API_KEY`), **CryptoPanic**. Bybit keys need read + trade permissions only, **never withdrawal**.

## Commands

```bash
atx check                                              # every endpoint, credentials, lake
atx universe                                           # current crypto universe (live, not PIT)
atx instruments                                        # snapshot contract specs (run daily)
atx klines  --symbols BTCUSDT,ETHUSDT --freq 1h --start 2024-01-01
atx klines  --symbols universe --freq 1d --price-type mark
atx funding --symbols BTCUSDT --start 2022-01-01
atx oi      --symbols BTCUSDT --interval 1h --start 2024-01-01
atx lsr     --symbols BTCUSDT --period 1h
atx trades  --symbol BTCUSDT --from 2026-11-01 --to 2026-11-05 [--raw] [--cache]
atx alt fng | atx alt stablecoins | atx alt dvol --currencies BTC,ETH | atx alt fred
atx account                                            # needs keys for settings.environment
atx stream  --symbols BTCUSDT,ETHUSDT --seconds 30
atx order-roundtrip --symbol BTCUSDT --yes             # demo/testnet only: real order path smoke test
atx collect [--minutes N] [--category spot] [--symbols A,B] [--depth 1000]   # live order flow -> flow_3s
atx update [--once | --loop]                           # live bars/funding/OI/DVOL/F&G -> lake (config: live)
```

Execution environment: `environment: demo` in settings.yaml (or `AI_TRADER_ENVIRONMENT`). Mainnet requires **both** `execution.allow_mainnet: true` and `AI_TRADER_ALLOW_MAINNET=1`.

## Live data

Two always-on daemons. Each runs a private copy of `atx` from `bin/` (so `cargo build` never hits a locked binary) with a restart loop and a daily log in `logs/`:

| daemon | runner | binary | writes |
|---|---|---|---|
| order-flow collector (`atx collect`) | `scripts\collector.ps1` | `bin/atx-collector.exe` | `flow_3s` (trades + L2 books as 3 s candles); config `flow:` |
| market-data updater (`atx update`) | `scripts\updater.ps1` | `bin/atx-updater.exe` | `bars` 1h/4h/1d last+mark, `funding`, `open_interest` 4h, DVOL + Fear & Greed (`alt_series`); config `live:` |

Start with `powershell -ExecutionPolicy Bypass -File scripts\<runner>.ps1` (Task Scheduler "At startup" makes it permanent). The updater polls every `live.poll_secs` and fetches only closed bars/periods newer than its cursors (`data/state/updater_cursor.json`, advanced only after a successful lake write). It logs a heartbeat line per cycle. Run `PYTHONIOENCODING=utf-8 uv run python scripts/freshness.py` to see the age of the newest knowable row per symbol and stream.

Order flow daily job: `uv run python -m ai_trader.storage.flow maintain`. Design and budget: [docs/ORDER_FLOW.md](docs/ORDER_FLOW.md). Bybit instrument catalogue: [research/reports/bybit_instruments_2026-11-07.md](research/reports/bybit_instruments_2026-11-07.md).

## Execution (demo)

Proposals (ranked best-first JSON, schema in [docs/EXECUTION.md](docs/EXECUTION.md)) go through the deterministic risk engine. Only then do they reach the Bybit demo or testnet account.

```bash
atx propose --file proposals.json                 # dry run: sizes, caps, net E[R], rejections (journaled)
atx propose --file proposals.json --submit        # demo/testnet only: bracket orders, link id "<session>-<id>"
atx positions                                     # open positions in R + open orders
atx positions --manage --breakeven                # stop -> entry once unrealised >= 1R (off by default)
atx flatten --symbol BTCUSDT --yes                # demo/testnet only: cancel orders + close (manual_action)
```

`--submit` and `flatten` refuse mainnet outright, and the two-key guard still applies on top. The daily caps read `data/state/exec_day.json`, which resets at the UTC day change. Every step appends to the same JSONL journal as Python (`data/journal/YYYY-MM-DD/{risk_decision,order,position_update,manual_action}.jsonl`). No position is ever closed on time.

## Research

```bash
bash scripts/backfill_research.sh                                      # 23-symbol research backfill
PYTHONIOENCODING=utf-8 uv run python scripts/run_research.py --freq 1h --record   # playbook + verdicts
uv run python -m ai_trader.research.diary tail                         # Claude's research diary
```

Protocol: [docs/RESEARCH_PROTOCOL.md](docs/RESEARCH_PROTOCOL.md). Hypotheses: [research/hypotheses.yaml](research/hypotheses.yaml). Beliefs: [research/BELIEFS.md](research/BELIEFS.md). Scoreboard: [research/reports/SCOREBOARD.md](research/reports/SCOREBOARD.md) (`uv run python scripts/scoreboard.py`). Reports: `research/reports/`.

## Tests

```bash
cargo test --workspace       # 27 tests: signing vectors, rounding, lake writer, filters, chain, risk, paper broker
uv run pytest                # 12 tests: as-of revisions, look-ahead invariance, Rust->Python reads, features, GARCH recovery
```
