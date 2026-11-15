# XS v3 live phase: design note (not built)

_Agent X3, 2026-11-15. Research artifact: `data/models/xs_v3/frozen/` (spec.json, v3_seed{0,1,2}.pt / .json, hashes). Evidence: [research/reports/xs_network_2026-11-15.md](../reports/xs_network_2026-11-15.md). Nothing here is running. Building it needs the user's go-ahead (research-only phase since 2026-11-12)._

## What a live runner would do, once per hour

1. **When.** At hh:00:05 UTC, after the 1h bar of hour t-1 is in the lake and its known_ts (close + 2 s) has passed. The row it computes is the decision row for hour t.
2. **Inputs.** For each universe-L symbol, read about 120 days of 1h bars (the 90-day funding z-score and the 8-week hour-of-week warm-up need them) plus funding and 4h OI with `market_frame(sym, "1h", now, funding=True, oi=True)`. Then build the row with the research code itself, `xs_network_study.token_rows` and `assemble`. Re-implementing the features is how train/serve skew gets in. Keep the last row (hour t) only. Absent tokens (listing warm-up, gaps, stale feeds) are masked exactly as in training. If fewer than 5 tokens are present, write no output.
3. **Model.** Load `spec.json`, check `code_sha256` and `files_sha256` against the working tree, standardise with `spec.standardisation`, clip to +/- 6 and run the three seeds. Average their quantiles. The cost is about 4k parameters x 23 tokens, well under a millisecond on CPU, with torch at 2 threads.
4. **Outputs, per token**, appended to `data/models/xs_v3/live/v3_<YYYY-MM>.parquet`:
   - ts, symbol and present;
   - vol quantiles q05..q95 at 1h, 4h and 24h, in return units (multiply by sqrt(h) s_t);
   - rel quantiles q10/q50/q90 at 4h and 24h, in return units;
   - the ACI-adjusted 80% rel band (state from `spec.aci.state`, updated when each origin's outcome is known: 4h origins at 00/04/.., 24h origins at 00 UTC);
   - cross-sectional rank and percentile of the rel median;
   - the attention row (top 3 keys and the market-token share).

   Also log the model version (`spec.version`), git commit, data as_of and the code hashes. Changing any of them starts a new track record (CLAUDE.md principle 8).
5. **Scoring.** The forward evaluation of the registered tests (H-XS1/2/3), run on the live rows once outcomes are known. Reuse `evaluate`'s maths on the appended rows, never on refitted predictions.

## Dashboard

- **v3 line per token.** On the token's chart: the 4h and 24h vol bands (q10-q90 shaded, q05/q95 dotted) around the last close. Beside it, a relative-strength line: the cumulative rel median forecast against the realised relative return.
- **Cross-sectional panel.** One table, refreshed hourly, sorted by the 24h rel median. Columns:
  - token and sector;
  - 24h and 4h rel median, in bp;
  - the 80% ACI band;
  - P(beats BTC) and P(beats ETH) from `pairwise_prob`;
  - the vol 24h q50 relative to its 30-day median;
  - the top attention keys.

  Next to the table:
  - a pairwise heatmap P(i beats j);
  - the PCA token map refreshed weekly;
  - a forward IC tracker (rolling 30-day mean IC of v3 vs the best momentum baseline, with its t).

  The panel is descriptive context for the decision session. It carries no sizes: position size comes from the risk engine (CLAUDE.md principle 4).

## Refits and guards

- Quarterly refits, the same protocol as the study:
  - expanding window;
  - purge 24 h + 1 day;
  - warm start from the previous refit;
  - 3 seeds.

  Each refit is a new version, frozen with `xs_network_study freeze`.
- Freshness guard: if any input bar is older than 2 h, mask that token. Above 5 masked tokens, drop the hour.
- Drift guard: if the 30-day rolling vol-pinball ratio v3 / log-HAR-t goes above 1.02, or the ACI alpha of a token leaves [0.05, 0.40], flag it on the dashboard and fall back to log-HAR-t bands for that token.
