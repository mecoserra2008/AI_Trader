# Phase 3 failure report (orchestrator, 2026-11-14)

**Status:** phase 3 was built and run (registry line 9, `50d88662…`). One acceptance criterion failed, so the build is stopped and waits for the user, per the kickoff prompt. No later phase has started.

**Failed criterion.** "Matched-sample balance table shows standardised mean differences below 0.1."
- **1M:** 31 matched pairs, SMD on `departure_atr` = **0.177**. The other two covariates are `age_log` −0.033 and `regime` −0.036.
- **Every other timeframe passes:** 15m 0.012, 30m 0.017, 1h 0.024, 2h 0.038, 4h 0.052, 1d 0.025, 1w 0.078.
- **Pooled:** all 349,259 pairs, maximum 0.018.
- **By (asset, timeframe) cell:** all 100 cells with ≥ 300 pairs are below 0.1 (max 0.084). One small cell, DOGEUSDT 1d with 81 pairs, is at 0.111.
- **Cause:** crypto has at most 6.9 years, so there are only 31 monthly first fills and 16 distinct monthly placebo events to match from. The matching was not retuned after the balance was seen.

**What the builder did correctly.** It reported the failure and did not adjust the matching (standing rule 6).

**Disclosures from phase 3**
1. **H2 outcome leak, since fixed.**
   - The first full run (registry line 8, superseded) printed event and label row counts per asset summed over families. Together with the level counts, those give a crude pooled placebo touch rate, an H2 outcome. Only the builder subagent saw it.
   - The counts were removed from the tables, the manifest and the stored records, and line 9 re-ran.
   - Nothing was retuned. The orchestrator and the future stats reviewer never saw the numbers, and the builder has no further role.
2. **Treated-only label summaries** (resolution rate, fade/follow target-first rates) were printed during smoke runs. They are treated-only, not treated-vs-placebo, so they are not a G2 statistic.
3. **SPEC section 3 arithmetic.** Under the frozen parameters, with targets at the resolution-band edges, the random-walk target-first probability from the near zone edge is b/(a+b) = 0.4375 for the fade and 0.3125 for the follow. That is below one half, not above as section 3 states. The placebo comparison is unaffected, and the analytic baseline inputs are stored per event.
4. **Void labels.** Where the next open is already through a barrier, the directional label is void: 12% of fade and 16% of follow labels on 15m. **Mean R is unstable** when entry falls a fraction of a tick inside the stop, with one R of 14,799. `pnl_atr` is stored alongside R for later phases.

**Options for the user**
- **A (orchestrator's recommendation).** Evaluate balance and G2 by timeframe group, as section 3 already words the family ("one per asset and timeframe group"): an intraday group (15m–4h) and a daily-and-higher group (1d, 1w, 1M). The daily-and-higher group is clustered by week, as the spec says. Balance would be judged per group. This changes no parameter, and the decision rests only on covariates, never on outcomes.
- **B.** Exclude 1M from the primary test and keep it as descriptive only.
- **C.** Wait for the BlackBull CFD data (10+ years, more monthly levels). Then re-run phase 3 balance on crypto plus CFDs before deciding between A and B.

Reproduce phase 3: `uv run --extra vecopen python -m vecopen phase3`.
