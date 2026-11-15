# Confluence log (`confluence_v1`)

The user trades by confluence: proportions and unfilled liquidity points, on-chain narratives, charts / volume / delta /
book / flow imbalance, momentum, structure breaks, vol that comes with thinner liquidity, and direction that comes
with the narrative. That process is hard to turn into model execution. The confluence log is the first step: every
desk cycle, the trader writes **one signed reading per layer per candidate asset**, with a pointer to the number it
read. Readings are logged *before* outcomes. Once >= 300 cycles exist, `scripts/confluence_fit.py` measures, walk-
forward, which layers carried information about forward returns and move size. Until then they are a record, not a
model. Readings never size: the risk engine sizes inside its hard limits.

Code: `src/ai_trader/narrative/confluence.py` (validate, log, rows); tests `tests/test_narrative.py`.

## Layers and scale

Each reading is a number in **[-1, 1]**: the sign is the direction that layer favours for the asset (+ = up,
- = down), the magnitude how clearly (0.2 faint, 0.5 clear, 0.8+ unusually strong). 0 = looked, no view. `null` =
could not read it, and needs a `reason`.

| layer | what it reads | snapshot inputs (evidence sources) |
|---|---|---|
| `levels` | unfilled levels and stop pools: untouched leg opens, unswept swing highs / lows, POC / VA, round numbers, estimated liquidation clusters; + when the nearer / larger pool is above (price is drawn to it) | `geometry.json` `legs`, `swings.*.swept`, `profile`, `nearest`, `liq_map` (an estimate) |
| `flow` | taker delta / flow imbalance over 5 / 15 / 60 min | `tape.json` `flow.windows.*.imbalance` |
| `book` | depth and asymmetry (bids vs asks at 10 / 25 / 100 bp), spread | `tape.json` `book` |
| `structure` | break of structure / change of character on 15m / 1h / 4h (BoS up = +) | `geometry.json` `swings`, BoS research (H-BOS) |
| `momentum` | returns and relative strength at 1h / 4h / 24h / 7d | `xs.json` `ret_bp`, `rs_sigma` |
| `vol_liq` | vol / liquidity state; sign = whether thin-book vol favours continuation (+ for the prevailing direction) or exhaustion; magnitude = how stressed | `tape.json` `book.depth`, `players.dvol`, `xs.json` `regime` |
| `positioning` | funding, OI change, liquidations: who is crowded and who is forced | `xs.json` `funding`, `oi.chg_pct`; `tape.json` `liq` |
| `narrative` | the active theses in the ledger: direction x status for this asset | `theses.json` (`<thesis id>.status`, `<thesis id>.net_support`) |
| `onchain` | placeholder until a source exists (whale / exchange-flow alerts, see the on-chain memo in research/reports/macro_narrative_2026-11-15.md) | none yet: log `null`, reason `no source` |

## Record (`<snapshot>/reports/confluence.json`, written by the trader)

The example uses real paths and values from the 2026-11-15 17:23 snapshot (the readings themselves are illustrative).

```json
{
  "contract": "confluence_v1",
  "cycle_id": "20261115T1723", "as_of": "2026-11-15T17:23:34.695628+00:00", "snapshot": "research/desk/2026-11-15/1723",
  "model": "<model id>", "prompt_version": "desk-trader-v1",
  "assets": [
    {"symbol": "BTCUSDT",
     "readings": {
       "levels":      {"value": 0.3,  "evidence": [{"source": "geometry.json", "metric": "nearest.above.2.price", "value": 65462.18}], "note": "24h VAH 1.2% above"},
       "flow":        {"value": 0.1,  "evidence": [{"source": "tape.json", "metric": "flow.windows.15m.imbalance", "value": 0.08}]},
       "book":        {"value": -0.2, "evidence": [{"source": "tape.json", "metric": "book.depth_usd.25bp.imbalance", "value": -0.234}]},
       "structure":   {"value": -0.5, "evidence": [{"source": "geometry.json", "metric": "swings.1h.highs.0.swept", "value": false}]},
       "momentum":    {"value": -0.3, "evidence": [{"source": "xs.json", "metric": "rs_sigma.24h", "value": -0.25}]},
       "vol_liq":     {"value": -0.4, "evidence": [{"source": "tape.json", "metric": "players.dvol.level", "value": 44.38}]},
       "positioning": {"value": 0.2,  "evidence": [{"source": "xs.json", "metric": "funding.pct", "value": 0.039}]},
       "narrative":   {"value": -0.4, "evidence": [{"source": "theses.json", "metric": "T-USER-20261115-a.net_support", "value": 0.6}]},
       "onchain":     {"value": null, "reason": "no source"}
     }}
  ]
}
```

Rules (enforced by `python -m ai_trader.narrative.confluence validate FILE --snapshot DIR`):
- every candidate asset the trader discussed (proposed or not) gets all nine layers; `null` needs a `reason`;
- every non-null reading has >= 1 evidence pointer whose `source` is a snapshot file and whose `metric` is a dotted
  path in that symbol's record (the same paths as CONTRACT section 3; `theses.json` uses `<thesis id>.<field>`);
  numeric evidence must match the snapshot (1% / printed rounding), as in the desk contract;
- no size, quantity, notional or leverage keys;
- one record per cycle; `log` appends it to `data/journal/confluence/<YYYY-MM>.jsonl` and refuses a second record
  for the same `cycle_id` (append-only; corrections are a note in the next cycle, never an edit).

## Honesty

- A reading is the trader's read of a tool output, not a new number: never compute a level or a probability.
- Readings are scored only in aggregate and only forward (`confluence_fit.py`, after >= 300 cycles, under a
  hypothesis registered before the first fit). Do not tune readings to past outcomes.
- Layers are correlated (momentum, structure and flow often agree); agreement is not independent evidence, and the
  fit's weights will show how much each adds once the others are known.
- Changing the layer list or the scale starts a new log version (`confluence_v2`).
