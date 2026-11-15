"""Desk data layer: deterministic, point-in-time inputs for the LLM analyst agents (research/desk/CONTRACT.md).

Modules
  public     GET-only clients for Bybit /v5/market/* and Deribit public endpoints (no keys, no trading calls)
  xs         cross-section: log-ratio relative strength vs the equal-weight basket, funding, OI, basis, sectors, v3 context
  geometry   swings, prior day / week / month levels, vecopen leg-origin opens, volume profile, estimated liquidation map
  tape       short moves, book spread / depth, liquidations, taker imbalance (flow_3s), the players.py read
  news       RSS headlines (titles kept) collector, scheduled events, Fear and Greed, Deribit expiries
  sentiment  Reddit (official OAuth API) + RSS mentions / tone collector, hourly panel, trend z-scores
  contract   report and proposal schemas + validator (research/desk/CONTRACT.md)
  snapshot   orchestration: research/desk/<date>/<HHMM>/{xs,geometry,news,sentiment,tape,account}.json + digests

Nothing here trades, sizes or forecasts direction. Every compute function takes `as_of` and ignores data after it.
"""

DESK_VERSION = "desk_v1.0"
