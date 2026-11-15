# Generative adversarial networks for financial trading strategies fine-tuning and combination (Koshiyama, Firoozye & Treleaven, 2021)

- **Citation:** Adriano Koshiyama, Nick Firoozye, Philip Treleaven (2021). *Quantitative Finance* 21(5), 797-813 (online 2020). DOI [10.1080/14697688.2020.1790635](https://doi.org/10.1080/14697688.2020.1790635). arXiv [1901.01751](https://arxiv.org/abs/1901.01751).
- **Verified:** arXiv API and Crossref, 2026-11-10. Details marked (m) come from memory.
- **Topic:** 4. Using generated data to **calibrate and combine** strategies, the closest published analogue of H-SYN1 / H-SYN2.

## Question
Does choosing strategy hyperparameters, and ensembling strategies, on cGAN-generated paths beat standard time-series validation methods out of sample?

## Data and method
- **579 assets** (abstract): equity indices, FX, commodities, rates, daily (m).
- Train a conditional GAN per asset (conditioned on recent history) and generate many alternative paths.
- (i) Calibration: pick the strategy hyperparameters that maximise average performance across generated paths.
- (ii) Combination: weight strategies by their performance across generated samples (an ensemble).
- Compare with time-series validation (e.g. sliding-window / block CV, stationary bootstrap (m)) and with standard ensembling, on held-out real data.

## Key results
- cGANs are a suitable alternative for calibration and combination. They **outperform when the traditional techniques fail to generate any alpha** (abstract). In other cases the gains are mixed and not uniform (m).

## Limitations / transfer to Bybit perps
- The benefit is conditional ("when traditional methods fail"), which reads as a regularisation effect, not new information. Results vary by asset and strategy; no deflated / multiple-testing adjustment of the comparison (m).
- GAN training per asset is costly and fragile. The stationary bootstrap (one of their baselines) is the cheap competitor and should be tested first.

## How we use it here
- Supports H-SYN1 / H-SYN2's design (select on many generated paths; robust selection by percentile). It also suggests the effect is **conditional**: expect improvement mainly where historical-path selection overfits badly (high PBO). That gives a sharper test: regress the degradation reduction on the in-sample PBO of each grid (draft hypothesis H-LIT-SYN4).
- Combination idea for B1 / B5: weight the trend-zoo ensemble by median performance across generated paths vs equal weight. Cheap to add.

## Evidence quality
**Medium** (Quantitative Finance, large asset panel; conditional, non-uniform gains; no multiple-testing adjustment).
