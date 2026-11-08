# Execution bridge (demo / testnet)

How decision proposals become orders. Code: `crates/atx/src/exec.rs` (CLI), `crates/at-risk` (sizing and caps, unchanged), `crates/at-exec/src/bybit.rs` (bracket orders).

```
proposals.json --> atx propose --> RiskEngine::evaluate --> table + risk_decision.jsonl
                        | --submit (demo/testnet only)
                        v
                  BybitBroker::submit (entry + position TP/SL) --> order.jsonl, exec_day.json
atx positions [--manage --breakeven] --> stop moves --> position_update.jsonl
atx flatten --symbol SYM --yes       --> cancel + close --> manual_action.jsonl
```

## 1. Proposal file

```json
{
  "session_id": "s20261107-1200",
  "clusters": {"ETHUSDT": "L1"},
  "meta": {"model": "...", "prompt_version": "..."},
  "proposals": [
    {"id": "p1", "symbol": "BTCUSDT", "side": "long", "entry_type": "limit",
     "entry": 85000.0, "stop": 83300.0, "take_profit": 88400.0,
     "p_tp": 0.45, "cluster": "L1", "channel": "C-MAIN"}
  ]
}
```

| field | required | meaning |
|---|---|---|
| `session_id` | yes, unless `--session-id` is given (the flag wins) | journal session; prefix of every order link id |
| `proposals` | yes, non-empty | `TradeProposal`s, **ranked best-first**: earlier proposals consume slots first |
| `proposals[].id` | yes, unique in the file | proposal id; link id = `<session>-<id>` |
| `proposals[].symbol` | yes | Bybit linear symbol (upper-cased on load) |
| `proposals[].side` | yes | `long` \| `short` |
| `proposals[].entry_type` | yes | `market` \| `limit` |
| `proposals[].entry`, `stop`, `take_profit` | yes | prices from structural placement; rounded to tick by the risk engine (stop away from entry, target toward entry) |
| `proposals[].p_tp` | yes | posterior P(TP before SL), in [0, 1] |
| `proposals[].cluster`, `channel` | yes (may be `""`) | cluster cap key; candidate channel |
| `clusters` | no | `symbol -> cluster` for positions **already open** (the venue does not know clusters) |
| `meta` | no | free-form provenance, journaled, never interpreted |

Unknown top-level fields are an error (a size field is never part of a proposal: size comes from the risk engine).

## 2. `atx propose --file F [--submit] [--session-id ID]`

Inputs assembled for `RiskEngine::evaluate`:

- **Instrument rules**: Bybit instruments-info of the execution environment for the proposal symbols (status `Trading` only; anything else is rejected as `unknown_instrument`).
- **Account**: wallet `totalEquity`; open positions. Their cluster comes from `clusters`, else from the journaled order for that symbol, else `""`.
- **Pending entries**: open, non-reduce-only orders on symbols without a position count as positions. A re-run of the same file therefore rejects with `duplicate_symbol` and cannot double up on slots, clusters or gross exposure.
- **Day state** `data/state/exec_day.json`: `{date, day_start_equity, trades_opened_today}`. The state resets at the UTC day change: on the first call of a UTC day, `day_start_equity` is set to the current equity and the count goes back to 0. `trades_opened_today` goes up once per **successfully submitted** entry order, whether or not it fills. A corrupt file is an error, not a silent reset.
- **Marks**: tickers `markPrice` (last price as fallback), used for gross exposure.

The command prints the accepted orders (qty, notional, risk incl. costs, R:R, net E[R]) and the rejections with reasons. It writes one `risk_decision` journal line for every run, dry or not (`"submit": true|false`).

With `--submit`:

1. Refused unless `environment` is `demo` or `testnet`. Mainnet is refused **outright**, and the existing two-key guard (`execution.allow_mainnet` + `AI_TRADER_ALLOW_MAINNET=1`) is applied on top in `BybitBroker::new`.
2. Each accepted order goes through `BybitBroker::submit`: leverage set once per symbol, an entry order (Limit GTC or Market), and position-level TP/SL (`tpslMode=Full`, SL trigger `MarkPrice`, TP trigger `LastPrice`).
3. `orderLinkId = "<session>-<proposal id>"`, limited to `[A-Za-z0-9_-]`. If it is over 36 chars, it becomes the first 27 chars + `-` + an 8-hex FNV-1a hash of the full id, so ids stay distinct. Bybit rejects a reused link id, so re-submitting the same session + proposal is a no-op at the venue.
4. One `order` journal line per order, with status `submitted` (and `venue_order_id`) or `error`. Any failure gives a non-zero exit.

## 3. `atx positions [--manage [--breakeven]] [--session-id ID]`

This lists open positions (side, qty, entry, mark, stop, TP, uPnL, **unrealised R**) and open orders. R = `(mark − entry)·sign / d`. Here `d` is the stop distance at entry, taken from the latest journaled `submitted` order for the symbol (`|entry − stop|` as sized). If there is no such order, `d` falls back to the position's current stop. The `R from` column says which source was used.

`--manage` applies the deterministic exit rules that need active management (ARCHITECTURE §3.12). For now:

- **structural trail**: not implemented yet;
- **breakeven** (`--breakeven`, off by default): once unrealised R ≥ 1, the stop moves to the entry price, rounded to the tick on the protective side. The move happens only if it tightens the stop and the new stop stays on the correct side of mark. Every change (applied or failed) goes to `position_update.jsonl`.

Nothing here closes a position on time.

## 4. `atx flatten --symbol SYM --yes [--session-id ID]`

This works on demo and testnet only. It cancels all open orders for the symbol, closes the position with a reduce-only market order, and verifies the result. It writes a `manual_action` line (cancelled orders, closed position, residuals). Operator interventions contaminate the track record, which is why they are always journaled.

## 5. Journal

The journal uses the same layout and envelope as `ai_trader.journal.writer.Journal`: `<paths.journal>/YYYY-MM-DD/<type>.jsonl`, one JSON object per line, `{"ts": "<UTC ISO-8601, +00:00>", "session_id": ..., "type": ..., ...payload}`. The event types are the Python `EVENT_TYPES`. Files are only ever appended to.

| type | written by | key payload fields |
|---|---|---|
| `risk_decision` | `propose` | `environment`, `submit`, `account`, `proposals`, `accepted` (SizedOrder), `rejected` (`id`, `rejection`, `detail`), `halted` |
| `order` | `propose --submit` | `broker`, `link_id`, `proposal_id`, `symbol`, `side`, `entry_type`, `qty`, `entry`, `stop`, `take_profit`, `risk_amount`, `cluster`, `status`, `venue_order_id` \| `error` |
| `position_update` | `positions --manage` | `action`, `rule`, `symbol`, `old_stop`, `new_stop`, `unrealised_r`, `r_distance`, `r_source`, `status` |
| `manual_action` | `flatten` | `action`, `symbol`, `cancelled_orders`, `closed_position`, `position_still_open`, `orders_remaining` |

Fills (`fill`) are not journaled by the CLI yet. `Trading::executions` is in place for a reconciler.
