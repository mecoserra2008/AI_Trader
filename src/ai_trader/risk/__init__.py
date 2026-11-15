"""Deterministic risk-side models fitted in Python and enforced in Rust (pattern: fit in Python, filter in Rust).

- stop_posterior: the H-STOP1 stop-distance cap (prior 4 sigma, weekly conjugate update from closed paper trades),
  written to data/models/risk/stop_cap.json and read by crates/at-risk (stop_cap.rs) and agent.session placement.
"""
