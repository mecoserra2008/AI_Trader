## 8. Thesis ledger and confluence log (addendum, agent MACRO 2026-11-15; for the orchestrator to merge)

After `desk_snapshot.py`, the orchestrator runs `uv run python scripts/thesis_check.py`, which evaluates
`research/desk/theses.yaml` point in time and writes `<snapshot>/theses.json` and `theses.md` (<= 30 lines: each
thesis's status, proxy support and forward score). Analysts and the trader read `theses.md` as narrative context;
a thesis is never an entry signal by itself (H-MAC1 / H-MAC2 inconclusive). The trader also writes
`<snapshot>/reports/confluence.json` (`confluence_v1`, research/desk/CONFLUENCE.md): for every candidate asset it
discussed, one signed reading in [-1, 1] per layer (levels, flow, book, structure, momentum, vol_liq, positioning,
narrative, onchain), each with an evidence pointer into the snapshot or `theses.json` (`onchain` stays `null`,
reason `no source`, until a source exists). The orchestrator validates and logs it with
`uv run python -m ai_trader.narrative.confluence log <snapshot>/reports/confluence.json --snapshot <snapshot>`
(appended to `data/journal/confluence/`); a failed validation is returned to the trader once. The readings carry no
size and do not change proposal rules; `scripts/confluence_fit.py` will fit them walk-forward only after >= 300
logged cycles and a registered hypothesis.
