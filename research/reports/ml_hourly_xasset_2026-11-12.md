# Hourly volatility ML for TradFi perps (H-NN11) and cross-asset activity inputs (H-NN12)

_Program 2026-11-12, agent G. Code: `src/ai_trader/ml/hourly_xasset.py`, `src/ai_trader/research/ml_hourly_xasset_study.py`, `src/ai_trader/research/ml_hourly_xasset_report.py`; tests: `tests/test_ml_hourly_xasset.py`; artifacts and per-refit checkpoints: `data/models/ml/xasset/`. Extends H-NN8 (`ml_signals_round2_study`): same features, refits, models, bias and loss. Deterministic research tools; the LLM layer is not trained or backtested here. Design fixed in the diary (2026-11-12 'H-NN11 / H-NN12 design fixed') before any model was fitted._

## Summary

| hypothesis | in-sample | holdout | pooled | raw (in / ho / pooled) | Holm p (in / ho) |
|---|---|---|---|---|---|
| H-NN11 | supports | supports | **supports** | supports / supports / supports | 0.049 / 0.000 |
| H-NN12 | inconclusive | inconclusive | **inconclusive** | inconclusive / inconclusive / inconclusive | 0.917 / 0.826 |

**H-NN11 (universe T, next-hour QLIKE vs robust_bc x hour-of-week).** Holdout: XGB -0.10192 (t -3.87), MLP -0.17556 (t -6.88), linear -0.13142 (t -4.86) over 132 dates and 15 symbols. In-sample (short, mostly gold tokens): XGB -0.10495 (t -2.25), MLP -0.07584 (t -1.29) over 91 dates. For scale, the champion beats bare robust sigma by -0.38380 (t -15.81) in the holdout.

**H-NN12 (with minus without the BTC/ETH/QQQ block, same family).** Crypto (L minus BTC/ETH): XGB -0.00018 (t -0.10) in-sample, -0.00050 (t -0.22) holdout; MLP 0.00673 (t 2.39) / 0.05430 (t 1.90). TradFi (T): XGB -0.01327 (t -1.48), MLP 0.06278 (t 5.63) in the holdout (-0.01069 (t -0.72) / 0.00681 (t 0.31) in-sample, reported only).

## Methods

- **Universe T**: 15 TradFi perps with >= 120 complete UTC days in the H-NS4 1h cache: TSLAUSDT, INTCUSDT, GOOGLUSDT, AAPLUSDT, MSTRUSDT, NVDAUSDT, MUUSDT, QQQUSDT, SOXLUSDT, XAUUSDT, XAGUSDT, CLUSDT, BZUSDT, PAXGUSDT, XAUTUSDT. Excluded: SPCXUSDT (116 days). Rows need the 8-week hour-of-week factor and the 30-day HAR level, so equities enter the test set only from mid-July to mid-August 2026 (after the cutoff).
- **Universe L**: the H-NN8 panel unchanged (23 crypto perps).
- **Features**: the 19 H-NN8 features (`ml.hourly`). T adds `cash_sess` for equities/ETFs (NYSE trading day, the 09:00 ET bar through the last bar before the 16:00 / 13:00 ET close) and `cme_sess` for commodities and the gold tokens (CME Globex Sun 18:00 - Fri 17:00 ET, daily break 17-18 ET). Both use America/New_York with DST. No class dummy, no symbol id.
- **Cross-asset block** (21 columns): for each of BTC, ETH and QQQ, log r^2/s2 at lags 1, 2, 3 and 6 h and the 1-6 h mean (in the leader's own base units), the log robust variance, and an availability flag. The inputs are built on the leader's own gap-safe grid from leader bars t-1 and earlier. The target's own copy is excluded. Missing values are set to the training mean (QQQ exists only from June 2026).
- **Walk-forward**: monthly, expanding, 1-day purge/embargo. First test month = the first with >= 365 pooled training dates: T 2026-04 to 2026-11 (8 refits), L 2021-07 to 2026-11 (65 refits). L trains on the H-NN8 1/3 subsample (same seed), T on all rows (58,415 at the last refit). Every test bar is scored.
- **Models**: frozen H-NN8 configs (MLP [64, 32], wd 0.01, 3 seeds; XGB depth 3, mcw 200) plus a ridge-linear reference. There was no new search. With and without the block are fitted on identical rows in every refit. Mean early-stopped XGB size: 301 / 296 trees (L, base / block), 196 / 194 (T).
- **Loss**: per-bar QLIKE difference r^2 (1/Fa - 1/Fb) + log(Fa/Fb), averaged per UTC date, NW t over dates. Segment = bar open before / after 2026-07-01. Holm per segment over the two hypotheses.
- **Runtime and machine**: <= 2 threads, memory-mapped arrays, pause if free RAM < 6 GB (shared machine). Total CPU time 46 min.

## H-NN11: TradFi next-hour QLIKE vs the champion

| segment | model | diff vs champion | dates | symbols |
|---|---|---|---|---|
| llm_in_sample | xgb | -0.10495 (t -2.25) | 91 | 5 |
| llm_in_sample | mlp | -0.07584 (t -1.29) | 91 | 5 |
| llm_in_sample | lin | -0.13033 (t -2.66) | 91 | 5 |
| llm_in_sample | champion vs bare s2 (context) | -0.40135 (t -7.05) | | |
| llm_in_sample | xgb vs linear | 0.02538 (t 1.33) | | |
| llm_in_sample | mlp vs linear | 0.05449 (t 2.75) | | |
| clean_holdout | xgb | -0.10192 (t -3.87) | 132 | 15 |
| clean_holdout | mlp | -0.17556 (t -6.88) | 132 | 15 |
| clean_holdout | lin | -0.13142 (t -4.86) | 132 | 15 |
| clean_holdout | champion vs bare s2 (context) | -0.38380 (t -15.81) | | |
| clean_holdout | xgb vs linear | 0.02950 (t 2.87) | | |
| clean_holdout | mlp vs linear | -0.04414 (t -4.58) | | |

By class (diff vs champion):

| segment | class | xgb | mlp | lin | champion vs s2 |
|---|---|---|---|---|---|
| llm_in_sample | commodity | -0.27426 (t -3.51) | -0.28171 (t -4.49) | -0.32562 (t -4.63) | -0.34730 (t -2.61) |
| llm_in_sample | gold_token | -0.08503 (t -1.81) | -0.05742 (t -0.99) | -0.11276 (t -2.31) | -0.44728 (t -6.75) |
| clean_holdout | commodity | -0.16061 (t -5.49) | -0.19773 (t -6.71) | -0.17325 (t -6.31) | -0.31932 (t -10.56) |
| clean_holdout | equity | -0.04272 (t -0.96) | -0.14504 (t -3.79) | -0.08325 (t -1.74) | -0.39222 (t -15.71) |
| clean_holdout | gold_token | -0.11788 (t -3.33) | -0.17586 (t -4.73) | -0.15175 (t -3.89) | -0.47904 (t -10.49) |

Per symbol, holdout: XGB beats the champion on 13/15 symbols (t < -2 on 9); MLP on 15/15 (t < -2 on 13).

| symbol | xgb diff (t) | mlp diff (t) | dates |
|---|---|---|---|
| AAPLUSDT | -0.02766 (t -0.24) | -0.19042 (t -2.15) | 109 |
| BZUSDT | -0.15340 (t -3.32) | -0.16887 (t -3.72) | 94 |
| CLUSDT | -0.04030 (t -0.77) | -0.08697 (t -2.16) | 132 |
| GOOGLUSDT | -0.10314 (t -1.73) | -0.27470 (t -3.65) | 110 |
| INTCUSDT | 0.00373 (t 0.06) | -0.11119 (t -1.55) | 115 |
| MSTRUSDT | 0.01412 (t 0.34) | -0.09574 (t -2.55) | 108 |
| MUUSDT | -0.08758 (t -2.01) | -0.15233 (t -3.54) | 100 |
| NVDAUSDT | -0.13823 (t -3.14) | -0.15156 (t -2.58) | 101 |
| PAXGUSDT | -0.13541 (t -3.74) | -0.18833 (t -5.00) | 132 |
| QQQUSDT | -0.09299 (t -2.36) | -0.13849 (t -4.05) | 100 |
| SOXLUSDT | -0.08467 (t -1.88) | -0.09466 (t -1.76) | 88 |
| TSLAUSDT | -0.12015 (t -2.33) | -0.25018 (t -4.25) | 116 |
| XAGUSDT | -0.23796 (t -4.78) | -0.27319 (t -5.46) | 132 |
| XAUTUSDT | -0.10035 (t -2.78) | -0.16338 (t -4.25) | 132 |
| XAUUSDT | -0.20987 (t -5.11) | -0.25600 (t -6.18) | 132 |

Group ablation in the base T models (dQLIKE when the group is set to its training mean; > 0 = the group helps):

| group | xgb in | xgb ho | mlp in | mlp ho |
|---|---|---|---|---|
| base_sigma | -0.00130 (-0.2) | -0.00762 (-1.8) | -0.01919 (-2.2) | -0.00203 (-0.9) |
| seasonal | +0.12788 (3.0) | +0.23717 (11.3) | +0.20275 (2.5) | +0.30446 (13.1) |
| funding_hour | -0.00041 (-0.5) | +0.00001 (0.1) | -0.01346 (-2.2) | +0.00024 (0.2) |
| recent_returns | +0.03203 (0.7) | +0.14155 (7.1) | -0.00134 (-0.0) | +0.12437 (5.2) |
| jump | +0.00009 (0.0) | -0.00014 (-0.1) | +0.00668 (1.8) | -0.00184 (-1.2) |
| daily_har | +0.01519 (1.5) | +0.00995 (1.9) | +0.01015 (0.5) | +0.02739 (3.6) |
| weekday | +0.01494 (2.4) | +0.01328 (4.3) | -0.00126 (-0.2) | +0.00799 (2.4) |
| session | +0.27731 (7.3) | +0.07213 (2.6) | -0.02730 (-0.9) | -0.00207 (-0.1) |

TreeSHAP group shares of the base T XGB (holdout):

| class | rows | seasonal | recent_returns | daily_har | session | base_sigma | weekday | other |
|---|---|---|---|---|---|---|---|---|
| commodity | 7123 | 28.3% | 24.2% | 9.9% | 28.5% | 4.9% | 2.9% | 1.2% |
| equity | 12065 | 23.6% | 23.8% | 6.8% | 36.3% | 4.6% | 3.6% | 1.2% |
| gold_token | 3948 | 33.0% | 24.0% | 9.0% | 25.9% | 4.5% | 2.6% | 1.0% |

## H-NN12: with vs without the cross-asset block

| component | segment | xgb | mlp | lin (reference) | xa XGB vs champion | dates | symbols | verdict |
|---|---|---|---|---|---|---|---|---|
| crypto (L minus BTC, ETH) | llm_in_sample | -0.00018 (t -0.10) | 0.00673 (t 2.39) | 0.00022 (t 0.18) | -0.04132 (t -4.95) | 1826 | 21 | inconclusive |
| crypto (L minus BTC, ETH) | clean_holdout | -0.00050 (t -0.22) | 0.05430 (t 1.90) | 0.00019 (t 0.06) | -0.06667 (t -2.41) | 132 | 21 | inconclusive |
| TradFi (T) | llm_in_sample | -0.01069 (t -0.72) | 0.00681 (t 0.31) | -0.00026 (t -0.02) | -0.11563 (t -2.67) | 91 | 5 | inconclusive |
| TradFi (T) | clean_holdout | -0.01327 (t -1.48) | 0.06278 (t 5.63) | 0.05289 (t 2.86) | -0.11519 (t -4.36) | 132 | 15 | inconclusive |

Descriptive subsets (with minus without):

| segment | subset | xgb | mlp |
|---|---|---|---|
| llm_in_sample | btc | 0.01291 (t 5.03) | 0.01552 (t 3.37) |
| llm_in_sample | eth | 0.00348 (t 1.19) | 0.01018 (t 3.38) |
| llm_in_sample | tradfi_ex_qqq | -0.01069 (t -0.72) | 0.00681 (t 0.31) |
| llm_in_sample | commodity | -0.00975 (t -0.51) | 0.01573 (t 0.22) |
| llm_in_sample | gold_token | -0.00749 (t -0.48) | 0.02142 (t 0.84) |
| clean_holdout | btc | 0.00953 (t 3.00) | 0.05830 (t 2.17) |
| clean_holdout | eth | 0.00151 (t 0.20) | 0.04318 (t 1.60) |
| clean_holdout | tradfi_ex_qqq | -0.01417 (t -1.53) | 0.06497 (t 5.63) |
| clean_holdout | qqq | 0.00436 (t 0.25) | 0.02106 (t 1.45) |
| clean_holdout | equity | -0.03559 (t -2.15) | 0.06170 (t 3.48) |
| clean_holdout | commodity | 0.00023 (t 0.03) | 0.04561 (t 2.93) |
| clean_holdout | gold_token | 0.01346 (t 2.23) | 0.07624 (t 7.71) |

Holdout, crypto: the block lowers XGB QLIKE on 8/21 symbols (t < -2 on 0, t > 2 on 2).
Holdout, TradFi: the block lowers XGB QLIKE on 8/15 symbols (t < -2 on 2, t > 2 on 1).

## Who leads whom: grouped TreeSHAP and leader ablation

Grouped TreeSHAP of the block XGB (mean |sum of phi| per group, as a share of all groups). 'block' = the three leader groups together. 'within block' splits the block among the leaders.

| universe | segment | class | rows | block share | BTC | ETH | QQQ (within block) |
|---|---|---|---|---|---|---|---|
| L | llm_in_sample | btc | 6138 | 13.4% | 39% | 61% | 0% |
| L | llm_in_sample | crypto_alt | 107731 | 18.4% | 54% | 46% | 0% |
| L | llm_in_sample | eth | 6131 | 12.5% | 85% | 15% | 0% |
| L | clean_holdout | btc | 433 | 8.0% | 23% | 77% | 0% |
| L | clean_holdout | crypto_alt | 9099 | 15.9% | 54% | 46% | 0% |
| L | clean_holdout | eth | 468 | 8.7% | 91% | 9% | 0% |
| T | llm_in_sample | commodity | 1551 | 9.8% | 54% | 46% | 0% |
| T | llm_in_sample | gold_token | 4368 | 10.1% | 53% | 47% | 0% |
| T | clean_holdout | commodity | 7123 | 10.3% | 49% | 33% | 18% |
| T | clean_holdout | equity | 12065 | 10.2% | 52% | 29% | 19% |
| T | clean_holdout | gold_token | 3948 | 9.4% | 48% | 35% | 17% |

Leader ablation (one leader's block set to its training mean; dQLIKE vs the intact block model, > 0 = that leader helps):

| segment | class | xgb BTC | xgb ETH | xgb QQQ | mlp BTC | mlp ETH | mlp QQQ |
|---|---|---|---|---|---|---|---|
| llm_in_sample | L:crypto_alt | +0.00507 (3.2) | +0.00135 (1.1) | +0.00000 (0.0) | +0.00014 (0.1) | -0.00006 (-0.0) | -0.00043 (-1.0) |
| llm_in_sample | L:btc | +0.00000 (0.0) | +0.00483 (3.4) | +0.00000 (0.0) | -0.00474 (-2.7) | +0.00241 (0.9) | -0.00070 (-0.9) |
| llm_in_sample | L:eth | +0.01066 (4.8) | +0.00000 (0.0) | +0.00000 (0.0) | +0.00248 (1.4) | -0.00002 (-0.0) | -0.00078 (-1.3) |
| llm_in_sample | T:commodity | -0.01265 (-0.6) | +0.01217 (0.9) | +0.00000 (0.0) | -0.04408 (-0.9) | +0.04648 (1.9) | -0.00892 (-0.2) |
| llm_in_sample | T:gold_token | -0.01209 (-0.9) | +0.00423 (0.4) | +0.00000 (0.0) | -0.02969 (-1.2) | +0.02523 (1.2) | -0.03269 (-1.8) |
| clean_holdout | L:crypto_alt | +0.00017 (0.0) | +0.00491 (0.5) | +0.00000 (0.0) | -0.00206 (-0.1) | +0.00181 (0.2) | -0.05262 (-1.9) |
| clean_holdout | L:btc | +0.00000 (0.0) | -0.00183 (-0.6) | +0.00000 (0.0) | +0.00579 (1.0) | -0.00443 (-0.7) | -0.05692 (-2.4) |
| clean_holdout | L:eth | +0.00618 (1.4) | +0.00000 (0.0) | +0.00000 (0.0) | +0.01271 (1.2) | +0.00734 (1.4) | -0.05061 (-2.1) |
| clean_holdout | T:equity | +0.03120 (2.9) | -0.00290 (-0.7) | -0.00250 (-0.4) | -0.01221 (-1.1) | +0.00718 (0.5) | +0.00193 (0.1) |
| clean_holdout | T:commodity | -0.00798 (-0.9) | -0.00119 (-0.3) | +0.00448 (1.9) | -0.00513 (-1.0) | -0.01244 (-1.4) | -0.02244 (-1.3) |
| clean_holdout | T:gold_token | -0.00770 (-1.0) | +0.00005 (0.0) | +0.00114 (0.6) | -0.00664 (-1.0) | -0.01180 (-2.7) | -0.05189 (-3.8) |

**Reading.** In the holdout - crypto_alt: block 16% of attribution, led by BTC (54% of the block); no leader ablation with t > 2; btc: block 8% of attribution, led by ETH (77% of the block); no leader ablation with t > 2; eth: block 9% of attribution, led by BTC (91% of the block); no leader ablation with t > 2; equity: block 10% of attribution, led by BTC (52% of the block); significant ablations: BTC (xgb +0.0312, t 2.9); commodity: block 10% of attribution, led by BTC (49% of the block); no leader ablation with t > 2; gold_token: block 9% of attribution, led by BTC (48% of the block); no leader ablation with t > 2.

## Replication check against H-NN8

Our base L run against H-NN8's own walk-forward (`wf8_preds`), on the same rows (969,177): log-forecast correlation XGB 0.999, MLP 1.000. The H-NN8 run's diff vs the champion on these rows: XGB -0.04045 (t -5.11) / -0.06493 (t -2.65) (in / ho); ours: XGB -0.04043 (t -5.16) / -0.06599 (t -2.64), MLP -0.03606 (t -4.82) / -0.06185 (t -2.90). The differences are refit noise only (seeds, threads), so H-NN8 replicates.

## What this changes for tool calls

- **TradFi next hour:** xgb and mlp beat robust sigma x hour-of-week out of sample (XGB -0.10192 (t -3.87), MLP -0.17556 (t -6.88)). The gain is 2-3x the crypto gain of H-NN8, on top of a champion that already beats bare sigma by -0.38380 (t -15.81). Both segments support it after Holm.
- **How much is nonlinear (TradFi):** the ridge-linear model on the same 21 inputs gets -0.13142 (t -4.86). It beats XGB (0.02950 (t 2.87) XGB minus linear) and loses to the MLP (-0.04414 (t -4.58) MLP minus linear). On ~60k training rows the extra inputs carry most of the gain, and depth-3 trees add nothing. The MLP is the only family that is significant for equities (MLP -0.14504 (t -3.79), XGB -0.04272 (t -0.96)). Commodities and gold tokens gain with every family.
- **What the TradFi model uses:** the hour-of-week factor (ablation XGB +0.237, t 11.3) and recent |returns| (+0.142, t 7.1) dominate. The session flags matter to XGB (+0.072, t 2.6) but not to the MLP (-0.0021), which reads the session from the factor. Funding-hour and jump flags add nothing, as in crypto.
- **Cross-asset block:** no forecast gain anywhere. For crypto alts XGB is -0.00018 (t -0.10) / -0.00050 (t -0.22) (in / ho), effectively zero. For TradFi XGB is -0.01327 (t -1.48) in the holdout. The MLP gets worse with the block (0.00673 (t 2.39) / 0.05430 (t 1.90) crypto, 0.06278 (t 5.63) TradFi holdout). In the crypto holdout most of that harm comes from QQQ: dropping QQQ's block improves the MLP by 0.053 (t 1.9) for alts. QQQ is a newly listed input, so the network extrapolates on it. The ETH/SOL vol transmission seen at 1 s (H-MS6, H-MD5) does not survive at the 1 h horizon once each asset's own last 1-24 h are known: the leaders' bursts are already in the target's own recent returns.
- **Who leads whom (descriptive; the block is ~8-18% of attribution and adds no loss gain):** BTC carries about half the block for every class: alts (BTC 54% / ETH 46%), ETH (BTC 91%), equities (BTC 52%, ETH 29%, QQQ 19%), commodities (49/33/18) and gold tokens (48/35/17). For BTC itself ETH leads (77%). QQQ does not drive TradFi more than BTC does. The only holdout leader ablation with t > 2 is BTC for equities in the XGB (+0.0312, t 2.9). In-sample: BTC for alts (+0.0051, t 3.2), ETH for BTC and BTC for ETH. These effects are 0.5-3% of QLIKE, and for alts they vanish in the holdout.

## Caveats

- TradFi history is short. The in-sample T test set (91 dates) is gold tokens plus XAU/XAG/CL from June. Equities appear only after the cutoff, so H-NN11 rests on the holdout, as registered. The holdout has 132 dates.
- The first T refits train almost only on PAXG/XAUT. For equities, the first holdout refits have seen few or no equity rows, so the cash-session flag is constant in their training data and cannot be used. This is conservative for H-NN11.
- The CME flag ignores CME holiday schedules. The NYSE calendar is hard-coded for 2026, with early closes.
- QQQ exists only from June 2026. In the crypto in-sample segment the QQQ block is missing (set to the training mean) everywhere, so the in-sample crypto result is about BTC and ETH only.
- L trains on a random 1/3 of bars, as H-NN8. All test bars are scored. Threads were capped at 2 and memory at ~1.5 GB per process (shared machine). This changes runtime only, not the design.
- The block is stored as float32, the base features as float64. Refit noise (seeds, threads) means our base L run is not bit-identical to H-NN8's wf8 (replication check above).

## Rule candidates

```yaml
- id: R-ML-20
  kind: "prefer"
  rule: "WHEN a central next-hour variance expectation is needed for a Bybit TradFi perp and a validated hourly ML tool exists THEN prefer it (H-NN8 inputs plus cash/CME session flags) over robust_bc x the hour-of-week factor, for expectations only, never for stops"
  when: "horizon 1h, TradFi perps (equities, ETFs, commodities, gold tokens), quoting or intraday expectations"
  then: "use the pooled TradFi hourly model with its validation bias; fall back to R-VOL-26 if it is stale"
  scope:
    assets: "Bybit TradFi perps (universe T)"
    horizon: "1h"
    sides: "both"
  effect: "next-hour QLIKE diff vs champion, holdout: XGB -0.10192 (t -3.87), MLP -0.17556 (t -6.88), linear -0.13142 (t -4.86); 132 dates, 15 symbols"
  confidence: "C"
  evidence: ["H-NN11", "H-NN8", "H-NS4"]
  review: "forward test for 1 month against R-VOL-26; demote if the forward QLIKE gain is not significant"
  status: "provisional"
- id: R-ML-21
  kind: "expect"
  rule: "WHEN forming a next-hour variance expectation for a crypto alt perp THEN don't expect BTC/ETH/QQQ recent activity to add much beyond the alt's own recent returns, daily HAR level and hour-of-week factor"
  when: "horizon 1h, crypto alts"
  then: "keep the alt's own inputs (R-ML-10); treat leader bursts as context, not as a forecast adjustment"
  scope:
    assets: "crypto perps (universe L minus BTC, ETH)"
    horizon: "1h"
    sides: "both"
  effect: "with minus without block, crypto alts: XGB -0.00018 (t -0.10) / -0.00050 (t -0.22), MLP 0.00673 (t 2.39) / 0.05430 (t 1.90) (in / ho)"
  confidence: "C"
  evidence: ["H-NN12", "H-NN8"]
  review: "rerun with a richer block (leader hour-of-week-adjusted surprises) as a new hypothesis"
  status: "provisional"
- id: R-ML-22
  kind: "expect"
  rule: "WHEN forming a next-hour variance expectation for a Bybit TradFi perp THEN don't expect recent BTC/ETH/QQQ activity to improve it beyond the perp's own inputs and session calendar"
  when: "horizon 1h, TradFi perps"
  then: "use the perp's own inputs and the session calendar only"
  scope:
    assets: "Bybit TradFi perps (universe T)"
    horizon: "1h"
    sides: "both"
  effect: "with minus without block, TradFi holdout: XGB -0.01327 (t -1.48), MLP 0.06278 (t 5.63)"
  confidence: "C"
  evidence: ["H-NN12", "H-NN11"]
  review: "rerun when the TradFi holdout has >= 250 dates"
  status: "provisional"
- id: R-ML-23
  kind: "expect"
  rule: "WHEN citing cross-asset context for a next-hour variance expectation THEN expect BTC to be the main leader for alts, ETH, equities, commodities and gold tokens (ETH for BTC itself), and QQQ to matter less than BTC even for TradFi perps; treat it as explanation only, since the block does not improve the forecast"
  when: "explaining or sanity-checking an hourly vol forecast"
  then: "mention BTC's last hours first; don't adjust the forecast for leader bursts beyond what the target's own recent returns show"
  scope:
    assets: "crypto perps (universe L) and Bybit TradFi perps (universe T)"
    horizon: "1h"
    sides: "both"
  effect: "holdout within-block TreeSHAP share BTC / ETH / QQQ: alts 54/46/0, equities 52/29/19, commodities 49/33/18, gold tokens 48/35/17, ETH 91/9/0, BTC -/77/0 (own copy excluded); only significant holdout ablation: BTC for equities (XGB +0.0312, t 2.9)"
  confidence: "C"
  evidence: ["H-NN12"]
  review: "recompute on the first 3 forward refits; demote if the per-class leader changes"
  status: "provisional"
- id: R-ML-24
  kind: "avoid"
  rule: "WHEN extending an hourly vol network with inputs that have a short history (a series listed months ago, such as QQQ for the crypto model) THEN avoid adding them to the MLP; the network extrapolates and next-hour QLIKE gets worse"
  when: "choosing inputs for a retrained hourly vol MLP"
  then: "keep the MLP on the target's own inputs; test new cross-asset inputs in trees or linear models first, as a new hypothesis"
  scope:
    assets: "crypto perps (universe L) and Bybit TradFi perps (universe T)"
    horizon: "1h"
    sides: "both"
  effect: "MLP with minus without block: crypto 0.00673 (t 2.39) / 0.05430 (t 1.90) (in / ho), TradFi holdout 0.06278 (t 5.63); removing the QQQ block improves the crypto-alt MLP by 0.0526 (t 1.9) in the holdout"
  confidence: "C"
  evidence: ["H-NN12"]
  review: "repeat when QQQ has >= 12 months of history"
  status: "provisional"
```
