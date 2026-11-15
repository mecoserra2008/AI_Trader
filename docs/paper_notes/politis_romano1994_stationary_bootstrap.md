# The Stationary Bootstrap (Politis & Romano, 1994)

- **Citation:** Dimitris N. Politis, Joseph P. Romano (1994). *Journal of the American Statistical Association* 89(428), 1303-1313. DOI [10.1080/01621459.1994.10476870](https://doi.org/10.1080/01621459.1994.10476870)
- **Verified:** Crossref, 2026-11-10. The method is standard.
- **Topic:** 4. Resampling for generated paths and inference. Hypotheses: H-SYN1, H-SYN2 (B5's `sim/resample.py`), and the block-bootstrap CIs used across the registry.

## Question
How do we bootstrap a weakly dependent stationary time series so that the resampled series is itself stationary?

## Method
- Resample blocks of consecutive observations with **random block lengths**, geometric with mean 1/p. Start each block at a uniformly random index and wrap around circularly. The resulting pseudo-series is strictly stationary (unlike fixed-block moving-block bootstrap).
- Choice of p: automatic block-length selection (Politis & White 2004, corrected by Patton, Politis & White 2009; not reviewed here) based on the autocorrelation of the statistic of interest. For vol clustering the relevant dependence is in |r| or r^2, which decays slowly, so long mean blocks are needed (days to weeks for 1h data).
- It is consistent for means and smooth functions of means, and the basis of White's Reality Check and Hansen's SPA bootstrap.

## Key results
- Stationarity of the pseudo-series and validity for weakly dependent data; less sensitive to the block-length choice than the fixed-block bootstrap.

## Limitations / transfer to Bybit perps
- It preserves dependence only **within** blocks. Long-memory vol (HAR-type, V5), regime episodes of 5-7 days (L1) and the weekly cycle (weekend vol 0.6-0.8x, V1) are broken at block joins. Mitigations:
  - (a) resample **vol-standardised** returns z = r / sigma-hat and re-impose a vol path from the fitted HAR / regime model;
  - (b) align blocks to the hour-of-week (resample whole weeks or day-of-week-matched days), so seasonality survives;
  - (c) regime-conditional block resampling (blocks drawn from the same v0 cell).
- **It can't create information:** generated paths only regularise selection (as the plan states); they don't validate a strategy.

## How we use it here
- B5 `sim/resample.py`: stationary bootstrap with mean block length chosen from the ACF of |z|, with week-aligned and regime-conditional variants. Report which variant reproduces the stylised facts (kurtosis, |r| ACF, weekend ratio) before using it in H-SYN1 / H-SYN2.
- Inference: prefer the stationary bootstrap (random lengths) over fixed 10-day blocks in paired-Sharpe tests. Minor, but it removes the block-length artefact.

## Evidence quality
**Strong** (a JASA classic).
