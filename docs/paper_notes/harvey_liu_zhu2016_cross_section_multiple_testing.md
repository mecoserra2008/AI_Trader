# ... and the Cross-Section of Expected Returns (Harvey, Liu & Zhu, 2016)

- **Citation:** Campbell R. Harvey, Yan Liu, Heqing Zhu (2016). *Review of Financial Studies* 29(1), 5-68. DOI [10.1093/rfs/hhv059](https://doi.org/10.1093/rfs/hhv059). NBER w20592.
- **Verified:** Crossref, 2026-11-10. Numbers marked (m) come from memory of this well-known paper.
- **Topic:** 4. Multiple-testing thresholds for strategy / factor discovery. Relevant to our Holm-within-block rule and to every hypothesis block.

## Question
Given hundreds of published factors, what t-statistic hurdle should a new factor clear?

## Data and method
- A catalogue of **316 factors** published in top journals up to 2012 (m), with their t-stats.
- Multiple-testing frameworks:
  - family-wise error rate: Bonferroni, Holm;
  - false discovery rate: Benjamini-Hochberg-Yekutieli (BHY), which allows arbitrary dependence.
- A model of unpublished (missing) tests is used to infer the total number of tests tried.
- They derive the time path of the hurdle as the number of tried factors grows.

## Key results (m)
- A new factor should clear **t > 3.0** (rather than 2.0). Many published factors are likely false discoveries.
- BHY is a reasonable default when tests are correlated; Holm is more conservative.

## Limitations / transfer to Bybit perps
- Calibrated to the academic equity-factor literature. Our "family" is our own registry. The repo's protocol already applies Holm within each block. HLZ suggests also tracking the **cumulative** number of tests across the whole registry (we have well over 100 hypotheses), since program-level false-discovery control matters for the beliefs file.
- In-sample here is "llm_in_sample", contaminated by Claude's knowledge of the literature. The clean holdout is a stronger defence than any t-hurdle, but it is short.

## How we use it here
- `research/verdicts.py` / SCOREBOARD (B1 or the orchestrator): add a registry-wide BHY q-value for in-sample results next to the per-block Holm, and flag beliefs graded A or B that rest on a single in-sample t between 2 and 3.
- For new strategy proposals from the literature (e.g. H-TF3 ORB, H-TF4), require in-sample t > 3 **or** holdout confirmation before deployment.

## Evidence quality
**Strong** (RFS, highly cited, a standard reference).
