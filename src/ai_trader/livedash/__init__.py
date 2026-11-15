"""Local, read-only live dashboard for the frozen BTC networks (research only).

    PYTHONIOENCODING=utf-8 uv run --extra ml python scripts/live_dashboard.py      ->  http://127.0.0.1:8765/

  pct          percentile of a move inside a 7-quantile predictive distribution (probit interpolation; mirrored in
               research/live/dashboard/pct.js and tested equal)
  activations  the frozen PNN forward pass of one journal line, decomposed per edge (node values, edge outputs,
               latent, predicted latent, quantile heads) plus per-input attributions
  store        journal readers (live_nn v1 / v2, live_claude, scores, LATEST.md) and the /api/state payload
  thinking     plain-language sentences from a line and its activations
  server       the HTTP server (127.0.0.1 only), static files, JSON endpoints and the Bybit public kline proxy

Never writes to the journals or models; never places orders.
"""
