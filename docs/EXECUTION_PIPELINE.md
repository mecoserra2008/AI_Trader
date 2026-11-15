# Strategy execution pipeline (contract for agents K and L, 2026-11-13)

**Goal:** run every registered deterministic strategy through one pipeline, with risk managed properly and the tested
biases (`research/rules.yaml`, `research/PLAYBOOK.md`) as the pillar of the edge. **Default mode is paper or replay.**
Mainnet stays off (`execution.allow_mainnet: false`), and the user has asked for research only, so nothing in this
pipeline may send live orders. Live use is a separate, later decision for the user.

```
strategies (REGISTRY, point-in-time signals)
   -> [K] bias gate: playbook rules + vol forecasts + calendar + placement -> OrderIntent (JSONL)
   -> [L] risk engine (Rust, at-risk): fixed-risk sizing + limits          -> RiskDecision (JSONL)
   -> [L] execution (at-exec PaperBroker; replay or paper)                 -> fills, positions, P&L, journal
```

Ownership:
- **K (Python)** owns `src/ai_trader/execution/`: the signal service, the bias gate and intent writing, plus
  `scripts/strategy_runner.py`.
- **L (Rust)** owns the at-risk and at-exec extensions and a new `atx exec-intents` subcommand.

Each owns only these files. The JSON formats below are the contract. Change them only by writing a note to this file,
dated, under a "Changes" heading.

## OrderIntent (one JSON object per line, written by K)

```json
{"intent_id": "uuid", "ts_ms": 0, "as_of": "ISO8601", "mode": "replay|paper",
 "strategy": "tsmom", "family": "trend|mean_reversion|breakout|positioning|calendar|shock|carry",
 "symbol": "ETHUSDT", "asset_class": "crypto|tradfi", "side": "long|short|flat",
 "signal": 0.73, "entry": {"type": "market|limit", "price": 2130.5},
 "stop": 2061.2, "targets": [2250.0],
 "sigma": {"per_bar": 0.0061, "daily_logHAR": 0.031, "weekday_factor": 0.62, "source": "vol.har|vol_ml"},
 "p_tp": 0.41, "net_er_r": 0.18,
 "gate": {"decision": "pass|block|reduce", "rules_applied": ["R-MAP-3", "R-PX-2"], "reasons": ["..."]},
 "context": {"regime_cell": "R4", "weekday": "Sat", "vol_pctile": 0.83, "card": "next-day vol / crypto perp"}}
```

Rules for intents:
- An intent never carries a size. Sizing belongs to the risk engine (CLAUDE.md principle 4).
- The stop is a structural or vol-based price. L checks it against the placement constants in `config/risk.yaml`.

## RiskDecision (one JSON object per line, written by L)

```json
{"intent_id": "uuid", "decision": "accept|reject", "qty": 0.012, "notional_usd": 25.6, "risk_usd": 0.5,
 "risk_r": 1.0, "rejections": ["max_positions_per_cluster", "daily_loss_limit", "stop_too_far", "net_er_below_min"],
 "cluster": "trend", "portfolio": {"gross_x": 0.4, "open_positions": 2, "family_counts": {"trend": 1}}}
```

## Risk rules L must enforce

These come from `config/risk.yaml` plus the evidence:
- Fixed risk per trade (`risk_per_trade_pct`).
- Daily loss limit, maximum open positions and maximum new trades per day.
- Maximum gross exposure.
- Placement: stop distance in sigma between `min_stop_distance_sigma` and `max_stop_distance_sigma`, plus
  `max_stop_distance_pct`.
- The `min_net_expected_r` gate.
- **Cluster caps by strategy family** (`max_positions_per_cluster`). R-PD-1 says the equal-weight book is about one
  trend bet, so families are the clusters.
- A kill switch: a file flag `data/exec/KILL` cancels and flattens in paper mode.

## Bias gate K must apply

The gate selects rules through `research/playbook.py` (`card_for` and the playbook JSON) and applies the
machine-checkable ones:
- **Regime-cell avoid lists:** R-MAP-3..6, for example no 1h mean reversion in R1/R2/R4/R7, and no 1h trend in R5/R6.
- **Direction tilts:** never from calendar or regime cells (R-PX-6, R-SIG-*, R-ML-5).
- **No fading:** never fade 2-sigma hourly moves (R-PX-2).
- **Exits:** the strategy's own exit (R-EXIT-1).
- **Stops:** `sigma_risk_next`, with no weekend shrink (R-VOL-2).
- **Expectations:** log-HAR times the weekday factor (R-ML-1, R-DS-*).
- **Reward:** jump-aware `p_tp` (R-EXEC-2) and `net_er`.
- **Trace:** each intent records the rules it applied, so every decision is traceable.

Rules that are not machine-checkable are listed in the intent's context for the LLM layer, which is phase 4 and
forward-only.

## Changes

### 2026-11-13 (L): how the risk engine and paper broker read the contract

Additive only: no contract field was renamed or removed.

- **Rust types.** Intents parse into `at_risk::OrderIntent`. Decisions are `at_risk::IntentDecision` (the name
  `RiskDecision` was already taken by the batch `propose` API). Each decision line has the nine contract fields,
  plus `action` (open|close|none), `symbol`, `strategy`, `side`, `entry_type`, tick-rounded `entry`/`stop`/`target`,
  `stop_pct`, `stop_sigma`, `cost_r`, `net_er_r` (the value the engine checked), `fee_class`, `details`
  (rejection -> numbers), and, in the file, `decided_at`, `decided_at_ms`, `mode` and `run`. `portfolio` is the
  state after the decision.
- **Rejection names.** `kill_switch`, `gate_blocked`, `invalid_intent`, `no_equity`, `unknown_instrument`,
  `duplicate_symbol`, `daily_loss_limit`, `max_new_trades_per_day`, `max_open_positions`,
  `max_positions_per_cluster`, `max_gross_exposure`, `exceeds_max_leverage`, `stop_too_far`,
  `stop_inside_min_sigma`, `stop_beyond_max_sigma`, `missing_sigma`, `net_er_below_min`, `net_er_missing`,
  `below_min_qty`, `below_min_notional`, `no_position_to_close`, `stale_intent` and `duplicate_intent`. A
  rejected intent lists every rule it broke, not only the first.
- **Gate.** `block` produces `gate_blocked`. `reduce` sizes at 0.5 of the fixed risk (`risk_r` is about 0.5).
- **Clusters.** `family` is the cluster (R-PD-1). Resting entries count as positions for slots, symbols,
  clusters and gross exposure. Each symbol holds one position at a time, so there is no stacking and no flipping;
  to reverse, send `flat` first.
- **Exits.** `side: flat` closes what that `strategy` holds in the symbol (position plus resting entries) at the
  next bar open. Exits are never blocked by entry limits or by KILL.
- **Placement.**
  - `sigma.per_bar` is required: a log-return sigma for one bar of the strategy's own frequency.
  - The stop distance in sigma is `|ln(entry/stop)| / per_bar`. It must lie within `placement.min/max_stop_distance_sigma`.
  - `weekday_factor` is not applied to placement (R-VOL-2).
  - The engine rounds to the tick conservatively: the stop away from entry, the target toward it.
- **Expected value.**
  - `net_er_r` checked = min(engine recomputation, intent `net_er_r`).
  - The engine recomputes from `p_tp` and `targets[0]`, net of the `config/fees.yaml` round trip.
  - At least one of the two must be present. Strategies that use their own exit (no target) must send `net_er_r`.
  - Only `targets[0]` is used, as a resting take-profit.
- **Entry price.** A market entry with no `entry.price` is sized at the latest close.
- **Fees.** New `config/fees.yaml`. Crypto: maker 2, taker 5.5 and slippage 2 bp. TradFi perps: maker 0,
  taker 2.75 and slippage 2 bp. The class comes from, in order: the symbol override, then the lake instruments
  `symbol_type`, then the intent's `asset_class`, then the default (crypto).
- **Timing.**
  - Replay: the decision time is `ts_ms` rounded up to the `--freq` bar grid, and fills start at that bar's open.
  - Paper: the book first advances over completed bars, then every new intent is decided at now. Intents older
    than `--max-intent-age-bars` are rejected as `stale_intent`.
  - Limit entries expire after `--limit-ttl-bars` bars (default 3).
- **Refusals.** An intent with `mode: "live"` refuses the whole file. `--mode live` is refused, and so is a run
  while the environment is mainnet.
- **Kill switch.** `data/exec/KILL` cancels resting orders and flattens the paper book at the last close. Every
  new entry is then rejected with `kill_switch` until the file is removed.

### 2026-11-13 (K): what the bias gate writes

Additive only; matches L's note above.

- **Families.** `flow` is added to the enum (flow_momentum, cvd_divergence). Because `family` is the risk cluster,
  every R-PD-2 trend-cluster strategy (tsmom, ema_stack, donchian_breakout, geo_*, compression_breakout,
  zoo_ensemble) is sent as `trend`. K never emits `reduce`.
- **Exits.** `side: flat` intents carry `stop: null`, `targets: []`, `p_tp: null` and `net_er_r: null`. In replay,
  K does not write the exit that follows an entry the gate blocked, and the run summary counts these.
- **Targets.** Every registered strategy exits on its own signal (R-EXIT-1), so entries carry `targets: []` and always
  carry `net_er_r`. The 3R reference target that `p_tp` and `net_er_r` are computed against is in
  `context.reference_target`. With `BiasGate(resting_target=True)`, it is sent as `targets[0]` instead.
- **Stops.** The stop is `STOP_K` = 1.5 daily risk sigmas (`sigma_risk_next`, unshrunk, R-VOL-2), converted to
  per-bar sigmas of the strategy's frequency (`sigma.per_bar`, L's unit). It is clamped 1% inside
  `[min, max]_stop_distance_sigma` to allow for tick rounding, then capped at `max_stop_distance_pct`.
  - For 1h strategies the result is 3.96 sigma_1h, about 0.8 daily sigma (flagged `stop_clamped_max_sigma`).
  - For 4h strategies it is 3.67 sigma_4h, which is the full 1.5 daily sigma.
  - The cap-versus-wide-stop conflict in BELIEFS.md is still the user's decision.
- **p_tp.**
  - `p_tp` = P(reference target before stop), from `jump_diffusion.simulate_bracket`. The model is Kou + Hawkes,
    fitted monthly on the 2 years of 1h bars before the month. The run is 1344 1h bars, effectively no time barrier.
    The diffusion sigma is the 5-day log-HAR level times the mean R-ML-1 weekday factor.
  - TradFi falls back to `agent.session.p_first`.
  - A prior edge is added only for the `agent.session.PRIOR_EDGE` sleeves: shock_regime +0.05 and zoo_ensemble +0.03.
- **Extra keys.**
  - `sigma`: `risk_daily`, `freq` and `central_1h`.
  - `gate`: `blocked_by` and `stop_sigma` (per-bar units).
  - `context`: `cell_now`, `vol_state`, `dir_state`, `cluster`, `llm_rules` (playbook rules for the LLM that are not
    machine-checked), `rules_hash`, `p_source`, `p_null`, `edge_prior`, `flags`, `exit_policy`, `target_role` and
    `entry_gate` (on exits).
- **Intent ids.** `intent_id` is a deterministic uuid5 of (mode, strategy, symbol, decision time, kind, side), so
  re-runs are idempotent.

### 2026-11-14 (D): stop-cap posterior, `--as-of`, and the phase-4 decision harness

Additive only: no contract field was renamed or removed.

- **Stop cap.** `placement.max_stop_distance_sigma` (4.0) is now the PRIOR. When `data/models/risk/stop_cap.json` exists,
  at-risk enforces its `applied_cap_sigma` instead (`at_risk::stop_cap`), after checking it lies inside the new hard bounds
  `placement.stop_cap_bounds_sigma: [2.5, 6.0]`. A file that fails a check is an error (fail closed). The only writer is the
  weekly H-STOP1 update `ai_trader.risk.stop_posterior` (Monday 00:00 UTC, closed paper trades). `execution.gate.BiasGate` and
  `agent.session` place stops against the same value. The LLM never sets it.
- **`atx exec-intents --as-of <RFC 3339>`.** The pass runs as of that time instead of now: bars are read only if known by
  then, and stale-intent ages are measured from it. It is refused in the future, and refused if the paper book has already
  processed a bar that closes after it (a book never moves backwards). `summary.json` gains `as_of`, `as_of_override` and
  `stop_cap_source`.
- **Phase-4 book.** `scripts/decision_session.py` (agent.decision) writes OrderIntents for Claude's takes with
  `strategy` = sleeve (shock_regime | trend_4h), `family` = shock | trend, `targets` = [the sleeve's structural target]
  (resting TP), `p_tp` = min(p_quant, p_final) (a positive LLM tilt never loosens the E[R] gate), and `context` carrying the
  session, candidate, reason and tilt. Early closes are `side: flat` intents for the position's strategy. Book:
  `--out-dir data/exec/paper_book --run ai` (pipeline tests: `--run pipeline_test`).

### 2026-11-14 ~13:30 UTC (D): paper EV gate and per-sleeve edge posteriors (user decision, option 1)

Additive only.

- **Paper EV gate.** `config/risk.yaml` gains `min_net_expected_r_paper: 0.0`, an exclusive bound: net E[R] after costs
  must be > 0. `min_net_expected_r: 0.10` (inclusive) stays the gate for every other mode.
  - `at_risk::DecideConfig::for_mode(mode, mainnet)` swaps in the paper value only for mode paper or replay and only off
    mainnet; `atx exec-intents` calls it.
  - The rejection name is still `net_er_below_min`. `summary.json` `risk` now reports `min_net_expected_r` and
    `net_er_exclusive`.
  - This applies to strategy_runner paper/replay runs too. The 15 null-edge strategies are still rejected there, because
    their net E[R] = −costs < 0.
- **Edge posteriors.** `ai_trader.risk.edge_posterior` turns `agent.session.PRIOR_EDGE` (shock_regime +0.05,
  trend_4h +0.03) into the priors of a conjugate normal model worth 30 pseudo-trades.
  - **Data:** the AI book's closed trades of each sleeve (kill exits excluded). Each trade's net R is put on the edge scale
    as y = (R − E0[R]) / (1 + g), using `context.p_null` and `context.g_net` of its intent. Those two fields are now
    written by the decision harness, with `er_null`.
  - **Schedule:** weekly, Monday 00:00 UTC.
  - **Output:** `data/models/risk/edge_posterior.json`.
  - **Use:** `agent.session.analyse` uses the posterior mean in force at as_of for p_quant and net E[R]. A posterior that
    takes net E[R] to <= 0 blocks that sleeve again.
  - The bias gate's `PRIOR_EDGE` use (strategy_runner) is unchanged.

### 2026-11-15 (D): paper entries pegged to the touch (H-EXE1, R-EXE-1 / R-EXE-2)

Additive only. Tool-version change: the session packet is now `compact-2` and exec is `pegged-limit-v1`. The AI book
had 0 trades, so no record was lost.

- **Config.** `config/risk.yaml` `paper_entry: {mode: pegged_limit, slip_bps: 1.3, window_min: 15}`. The alternative mode
  `as_signal` keeps the sleeves' own entry types.
- **Intent.** `entry` may carry `peg: "touch"` (with `type: limit`). `price` is then the decision-time reference. The
  decision harness sends every shock_regime and trend_4h entry this way when the mode is pegged_limit.
- **Risk engine (at-risk).**
  - A pegged entry is costed as maker + `slip_bps`, plus the usual taker + slippage exit
    (`FeeRates::round_trip_frac_pegged`).
  - A peg on a market entry, an unknown peg, or a missing / non-pegged `paper_entry` is `invalid_intent`.
  - The decision line gains `peg_slip_bps`.
- **Paper fill (at-exec PaperBook).**
  - `BookOrder.peg_slip_bps` (serde default None) fills at the first 1h bar open after submission, moved `slip_bps`
    against the trade, as maker.
  - The pre-fill cancels match a market order's: stop or target gapped through.
  - **Approximation:** no 1m bars, so the 15-minute re-peg window lies inside that first bar. The fill is taken even if
    the bar range never reaches the peg price. Such a bar moved toward the resting order, which would have filled at
    least as well. Dropping those fills would keep only the entries that went our way first. H-EXE1 filled 275 of 275
    replayed entries within 37 s.
  - Exits are unchanged.
- **Python.**
  - `agent.session.entry_cost_bps` / `net_er` cost `pegged_limit` entries the same way.
  - `agent.scorecard` shadow R now uses each candidate's entry-type round trip. The old fixed market round trip remains
    for unknown types.
