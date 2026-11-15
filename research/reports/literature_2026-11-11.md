# Literature cycle 2026-11-11 (agent B6): P&L decomposition, less obvious strategies, execution simulators, generated data and overfitting

Supports [Program 2](../plans/2026-11-11_strategy_program.md). 38 papers, one note each in [docs/paper_notes/](../../docs/paper_notes/).

**Verification.** Every citation was checked on 2026-11-10 against the arXiv API and/or Crossref (DOI, venue, volume, pages). Abstracts came from RePEc, Semantic Scholar or arXiv. Six papers were read in full in a scratch directory, and the numbers quoted from them are verified: Hansen-Kim-Kimbrough, Huang-Lehalle-Rosenbaum, Vyetrenko et al., Coletta et al., Morariu-Patrichi-Pakkanen and Arroyo et al. The Liberty Street post (overnight drift) and the He et al. HTML (perps) were also read. Numbers marked **(m)** in the notes come from memory of the paper and are flagged as unverified.

**Status.** The draft hypotheses in §5 are **proposals only**: nothing here is registered in `hypotheses.yaml`, and no result was recorded.

## 0. Index of notes

| topic | note | evidence |
|---|---|---|
| 1 decomposition | [sharpe1992_style_analysis](../../docs/paper_notes/sharpe1992_style_analysis.md) | strong (method) |
| 1 | [fung_hsieh2001_trend_followers](../../docs/paper_notes/fung_hsieh2001_trend_followers.md) | strong |
| 1 | [meucci2009_effective_number_of_bets](../../docs/paper_notes/meucci2009_effective_number_of_bets.md) | medium-strong (method) |
| 1 | [kelly_pruitt_su2019_ipca](../../docs/paper_notes/kelly_pruitt_su2019_ipca.md) | strong (method); weak for our N |
| 1 | [gu_kelly_xiu2020_ml_asset_pricing](../../docs/paper_notes/gu_kelly_xiu2020_ml_asset_pricing.md) | strong (equities); protocol only |
| 1 | [liu_tsyvinski_wu2022_crypto_factors](../../docs/paper_notes/liu_tsyvinski_wu2022_crypto_factors.md) | strong |
| 1 | [ehsani_linnainmaa2022_factor_momentum](../../docs/paper_notes/ehsani_linnainmaa2022_factor_momentum.md) | strong |
| 1 | [haddad_kozak_santosh2020_factor_timing](../../docs/paper_notes/haddad_kozak_santosh2020_factor_timing.md) | strong; weak transfer |
| 2 strategies | [lou_polk_skouras2019_overnight_intraday](../../docs/paper_notes/lou_polk_skouras2019_overnight_intraday.md) | strong; weak transfer |
| 2 | [boyarchenko_larsen_whelan2023_overnight_drift](../../docs/paper_notes/boyarchenko_larsen_whelan2023_overnight_drift.md) | strong, but **decayed since 2021** |
| 2 | [gao_han_li_zhou2018_intraday_momentum](../../docs/paper_notes/gao_han_li_zhou2018_intraday_momentum.md) | strong; medium-weak transfer |
| 2 | [baltussen_da_lammers_martens2021_hedging_intraday_momentum](../../docs/paper_notes/baltussen_da_lammers_martens2021_hedging_intraday_momentum.md) | strong |
| 2 | [zarattini_aziz2023_opening_range_breakout](../../docs/paper_notes/zarattini_aziz2023_opening_range_breakout.md) | weak-medium (practitioner, zero slippage) |
| 2 | [lucca_moench2015_pre_fomc_drift](../../docs/paper_notes/lucca_moench2015_pre_fomc_drift.md) | strong, but **decayed after 2015** |
| 2 | [ni_pearson_poteshman2005_expiry_pinning](../../docs/paper_notes/ni_pearson_poteshman2005_expiry_pinning.md) | strong; weak transfer |
| 2 | [he_manela_ross_vonwachter2022_perpetual_futures](../../docs/paper_notes/he_manela_ross_vonwachter2022_perpetual_futures.md) | medium-strong |
| 2 | [schmeling_schrimpf_todorov2026_crypto_carry](../../docs/paper_notes/schmeling_schrimpf_todorov2026_crypto_carry.md) | strong |
| 2 | [hansen_kim_kimbrough2024_crypto_periodicity](../../docs/paper_notes/hansen_kim_kimbrough2024_crypto_periodicity.md) | strong |
| 2 | [garcia_seuma2026_liquidation_cascade_ews](../../docs/paper_notes/garcia_seuma2026_liquidation_cascade_ews.md) | weak (unrefereed, n = 7) |
| 3 simulators | [cont_stoikov_talreja2010_queue_model](../../docs/paper_notes/cont_stoikov_talreja2010_queue_model.md) | strong |
| 3 | [huang_lehalle_rosenbaum2015_queue_reactive](../../docs/paper_notes/huang_lehalle_rosenbaum2015_queue_reactive.md) | strong |
| 3 | [bacry_mastromatteo_muzy2015_hawkes_finance](../../docs/paper_notes/bacry_mastromatteo_muzy2015_hawkes_finance.md) | strong (review) |
| 3 | [morariu_patrichi_pakkanen2022_state_dependent_hawkes](../../docs/paper_notes/morariu_patrichi_pakkanen2022_state_dependent_hawkes.md) | medium-strong |
| 3 | [lo_mackinlay_zhang2002_limit_order_executions](../../docs/paper_notes/lo_mackinlay_zhang2002_limit_order_executions.md) | strong |
| 3 | [arroyo_cartea_morenopino_zohren2024_fill_survival](../../docs/paper_notes/arroyo_cartea_morenopino_zohren2024_fill_survival.md) | medium-strong |
| 3 | [kyle_obizhaeva2016_microstructure_invariance](../../docs/paper_notes/kyle_obizhaeva2016_microstructure_invariance.md) | strong |
| 3 | [byrd_hybinette_balch2020_abides](../../docs/paper_notes/byrd_hybinette_balch2020_abides.md) | medium (infrastructure) |
| 3 | [vyetrenko2020_get_real_realism_metrics](../../docs/paper_notes/vyetrenko2020_get_real_realism_metrics.md) | medium-strong (framework) |
| 3 | [coletta2021_cgan_market_simulation](../../docs/paper_notes/coletta2021_cgan_market_simulation.md) | medium-weak |
| 4 overfitting | [bailey_borwein_lopezdeprado_zhu2017_pbo](../../docs/paper_notes/bailey_borwein_lopezdeprado_zhu2017_pbo.md) | strong (method) |
| 4 | [bailey_lopezdeprado2014_deflated_sharpe](../../docs/paper_notes/bailey_lopezdeprado2014_deflated_sharpe.md) | strong (in use) |
| 4 | [harvey_liu_zhu2016_cross_section_multiple_testing](../../docs/paper_notes/harvey_liu_zhu2016_cross_section_multiple_testing.md) | strong |
| 4 | [politis_romano1994_stationary_bootstrap](../../docs/paper_notes/politis_romano1994_stationary_bootstrap.md) | strong |
| 4 | [carr_lopezdeprado2014_optimal_trading_rules_synthetic](../../docs/paper_notes/carr_lopezdeprado2014_optimal_trading_rules_synthetic.md) | medium |
| 4 | [wiese2020_quant_gans](../../docs/paper_notes/wiese2020_quant_gans.md) | medium (no selection evidence) |
| 4 | [buehler2020_market_generator_small_data](../../docs/paper_notes/buehler2020_market_generator_small_data.md) | weak-medium |
| 4 | [koshiyama_firoozye_treleaven2021_gans_strategy_tuning](../../docs/paper_notes/koshiyama_firoozye_treleaven2021_gans_strategy_tuning.md) | medium |
| 4 | [arian_norouzi_seco2024_cpcv_vs_walkforward](../../docs/paper_notes/arian_norouzi_seco2024_cpcv_vs_walkforward.md) | medium-weak |

Checked and verified but not given their own note:
- Baur, Cahill, Godfrey & Liu (2019), *Finance Research Letters* 31, 78-92. Bitcoin time-of-day / day-of-week effects in returns: "time-specific anomalies but no persistent effects", per the abstract.
- Kurov, Wolfe & Gilbert (2021), *FRL* 40, 101781 (pre-FOMC drift disappeared).
- Hardiman, Bercot & Bouchaud (2013), *EPJ B* 86 (near-critical Hawkes branching).
- Bondarenko & Muravyev (2023), *JFQA* 58(3), 939-967 (market return around the clock).
- Ackerer, Hugonnier & Jermann, "Perpetual Futures Pricing", arXiv 2310.11771.
- Hansen (2005) SPA test, *JBES* 23(4), 365-380.
- Garcia Seuma, arXiv 2608.03616 (companion cascade paper).

Found but **not read**, so not used: a 2026 independent audit of Zarattini et al.'s SPY intraday-momentum paper (SSRN 7323419) and an AFA conference paper on spot-market quality over the funding cycle (title and authors not verified).

---

## 1. Decomposing strategy P&L into factors, hidden factors and context

### Synthesis
- **Agreement.** Strategy returns are mostly exposure to a few factors, and the residual "selection" is small:
  - Sharpe's style analysis does this with observable linear factors;
  - Fung-Hsieh adds option-like (convex) factors, without which trend followers look like alpha;
  - Liu-Tsyvinski-Wu find that three factors (market, size, momentum) explain crypto's cross-section of characteristic strategies;
  - KPS find that latent factors with characteristic-driven loadings absorb the "anomaly" intercepts.

  Our own P7 (the trend zoo is ~1.9 bets) and R-MM-2 (MM fills are one bet) are local instances.
- **Dimensionality reduction first, then conditioning.** Haddad-Kozak-Santosh time only the dominant PCs, because predicting each of many strategies overfits. Ehsani-Linnainmaa show the momentum in factor returns lives in the high-eigenvalue PCs. KPS make loadings a function of characteristics. All three say the same thing: **rank in a low-dimensional factor space, not strategy by strategy**.
- **Disagreement / open.** Whether conditioning beats a naive baseline. Factor momentum (trailing performance) is itself a strong, simple conditioner, and GKX show that ML gains need large cross-sections. Our universe is tiny, and R-BUDGET-1 already found more inputs hurt. Expect IPCA (H-PD3) to struggle; strategy momentum and PC timing are the more credible conditioners.
- **Crypto-specific factors** missing from the equity literature: **funding / basis** (He et al.; Schmeling-Schrimpf-Todorov) and its link to trend-chasing demand. Carry and trend should co-move in mania phases.

### Methods to adopt (ranked)

| rank | method | where | effort | expected value |
|---|---|---|---|---|
| 1 | PCA of the strategy panel + Meucci ENB at actual book weights, per regime cell | B1 `pnl_decomposition_study.py` (H-PD1) | 0.5 day | high: tells the LLM how many bets the book holds, by state |
| 2 | Style regressions with **convexity factors** (synthetic lookback straddle, market^2) plus CMKT, CMOM, funding-carry, basis, QQQ, gold, oil | B1 (H-PD2) | 1 day | high: says what each strategy "is" and in which states it pays |
| 3 | Strategy time-series momentum (sign of trailing 90/180/365d net P&L) as the explicit baseline for every conditional ranking | B1 (H-PD3 baseline, H-RC2) | 0.5 day | medium-high: the literature's strongest simple conditioner |
| 4 | Time PC1-PC3 of the strategy panel with 1-2 state variables each (HKS-style) | B1 | 1 day | medium |
| 5 | IPCA with context x strategy-family instruments, K <= 2-3 | B1 (H-PD3) | 2 days | low-medium (small N); keep as registered |

---

## 2. Less obvious, automatable strategies (TradFi sessions, calendars, expiries, funding, cascades)

### Synthesis
- **TradFi session effects are real but fragile.** Three mechanism-backed effects decayed after publication or after a market-structure change:
  - the S&P overnight drift (RFS 2023), near zero since 2021 per the authors' own 2026 update;
  - the pre-FOMC drift (JF 2015), "disappearing" per Kurov et al. 2021;
  - US intraday momentum, reported weaker after 2013 (unverified).

  Durable effects are the ones tied to **mechanical hedging flows**: gamma hedging and intraday momentum across 60+ futures over 46 years (Baltussen et al.), and expiry pinning by delta-hedgers (Ni et al.). Even those depend on the dealer gamma sign.
- **Transfer to 24/5 perps is the weak link.** Overnight on an equity perp is a traded, crypto-native period, so the clientele and inventory mechanisms may not carry over. Tests are holdout-dominated (5 months).
- **Crypto structural effects with strong evidence are about volatility and volume, not direction.**
  - Hansen-Kim-Kimbrough document hour-of-day, Saturday and minute-of-hour patterns, including **spikes at funding times 00/08/16 UTC and in the first minute of every hour**, growing stronger over time.
  - Funding / basis is a decaying leverage-demand premium (He et al.: deviations shrink ~11% a year; Schmeling et al.: driven by trend-chasing demand and limited arbitrage capital). This **agrees** with our C1 / R-CARRY-1.
  - Liquidation cascades are fast (88% of forced selling within 30 minutes) and correlation-ordering, consistent with our R-FLOW-2 (clusters continued) and R-MM-2.
- **Disagreement.** The practitioner ORB / intraday-momentum papers (Zarattini et al.) claim Sharpe 1.3-2.8 with zero slippage. An unread 2026 audit and a blog replication suggest net-of-cost results vanish. Weight them low.
- **Net.** The most automatable edges for us are **execution- and risk-side uses of calendars** (seasonal vol for quoting and stops, pulling quotes at hh:00 and funding minutes), not directional calendar bets. That fits BELIEFS (direction is not forecastable; edge is in conditioning, flows and execution).

### Methods to adopt (ranked)

| rank | method | where | effort | expected value |
|---|---|---|---|---|
| 1 | Hour-of-week x funding-hour x minute-of-hour vol / volume seasonal factors (multiplicative, expanding window) | B3 `features/seasonality.py` (H-NS4), B4 quoting | 1 day | high: improves sigma for stops and quotes; strong prior |
| 2 | Event-study template with **asymmetry and next-day-reversal checks** (BLW sell-off asymmetry; Baltussen reversal) for every session hypothesis | B2 `tradfi_session_study.py` (H-TF2/4, H-TP1) | 0.5 day | medium: sharper tests than unconditional means |
| 3 | Basis / funding x OI as a crowding factor for attribution and vol risk (not direction) | B1 factors, H-LEV6 | 0.5 day | medium |
| 4 | Expiry pinning diagnostic (distance of the 08:00 Deribit index to the max-OI or round strike, vs non-expiry Fridays) | B3 (H-NS2 companion) | 0.5 day + data decision | low-medium |
| 5 | Pre-FOMC / macro-event windows as risk context in `research/calendar/` | calendar tool | 0.5 day | low (decayed), but cheap risk context |

---

## 3. Market simulators for execution research

### Synthesis
- **Model ladder agreed in the literature:**
  1. zero-intelligence Poisson queues (CST 2010), with analytic fill probabilities;
  2. state-dependent intensities (queue-reactive, HLR 2015; state-dependent Hawkes, MPP 2022), where "reaction functions" are intensities as functions of book state;
  3. self-exciting flow (Hawkes; BMM 2015);
  4. agent-based or learned responsive generators (ABIDES; Coletta's CGAN world agent).

  Realism is judged on a stylised-fact battery (Vyetrenko et al.), and current ABMs still fail several facts (flow autocorrelation, intraday patterns).
- **Two lessons directly change B4's design:**
  - (a) **endogenous flow alone under-produces volatility and over-produces mean reversion.** HLR: purely book-driven max vol 5 bp vs 14 bp empirical; continuation ratio eta 0.08 vs 0.39. The generator needs an exogenous price-information component.
  - (b) **a touch is not a fill** (LMZ 2002). Fill models should be survival models with cancellations as right-censoring, scored with a proper scoring rule (RCLL; Arroyo et al.). Fast fills carry an adverse-selection signature.
- **Transfer across markets.** Kyle-Obizhaeva invariance gives a pre-specified scaling (trades proportional to W^(2/3), bet size / volume proportional to W^(-2/3), cost proportional to sigma W^(-1/3), with W = dollar volume x sigma). That is a sharper version of H-SIM3's "scale by sigma, depth and trade size".
- **Disagreement / gap.** Deep generators (Coletta, Arroyo) need L3 message data and far more than 3 days. With flow_3s, the faithful choices are discrete-time analogues: a VAR / Poisson regression with state interactions, binned Hawkes, and parametric survival.

### Methods to adopt (ranked)

| rank | method | where | effort | expected value |
|---|---|---|---|---|
| 1 | Parametric survival fill model (Weibull / generalized-gamma AFT; covariates: distance in sigma and ticks, own-side 3s taker volume, spread, minute-of-hour / funding dummies), cancels censored, scored by RCLL + calibration; Kaplan-Meier by depth first | B4 `sim/fill_model.py` (H-SIM1) | 1 day | high: every maker / MM test depends on it |
| 2 | Reaction functions as state-binned intensities (VAR / Poisson regression of buy and sell taker counts on lags x {spread, imbalance, recent-return} bins) | B4 `sim/reaction.py` (H-SIM3) | 1-2 days | high |
| 3 | Exogenous jump / reset component in the generator, calibrated to 1m RV and the eta continuation / alternation ratio; add eta to the H-SIM2 battery | B4 `sim/flow_sim.py` (H-SIM2) | 1 day | high: avoids a known failure mode |
| 4 | Invariance-based liquidity scaling (W = notional x sigma) for transfer and for per-symbol cost curves | B4 (H-SIM3), risk / costs | 0.5 day | medium-high |
| 5 | Vyetrenko stylised-fact scorecard (computable subset) as the descriptive realism report | B4 | 0.5 day | medium |
| 6 | Responsive replay: our quotes alter the simulated top of book (ABIDES / Coletta idea), plus latency | B4 `sim/replay.py` (H-SIM4) | 2 days | medium |

---

## 4. Generated data, robust selection and overfitting

### Synthesis
- **Agreement:**
  - Selecting the best of N backtests inflates performance;
  - deflate the winner (DSR), measure the selection process (PBO via CSCV), and raise the hurdle under multiple testing (HLZ: t > 3; BHY / Holm);
  - resampling must preserve dependence: the stationary bootstrap, with vol clustering and regimes handled via standardised returns or regime-conditional blocks.
- **Generated data regularises selection; it does not add information.** Evidence that it helps selection is thin and conditional:
  - Koshiyama et al. find cGAN-path calibration helps mainly "when traditional techniques fail";
  - Carr-LdP's synthetic OU paths show exits can be "optimised" only when there is real mean reversion; near a random walk the surface is flat;
  - Quant GANs and signature VAEs reproduce stylised facts but offer **no** out-of-sample selection evidence.

  H-SYN1 / H-SYN2's priors (0.45 / 0.40) look reasonable, if slightly generous.
- **Disagreement.** Walk-forward vs combinatorial purged CV. Arian et al. (2024, synthetic controlled study) find CPCV gives lower PBO and better DSR than walk-forward, which "exhibits notable shortcomings in false discovery prevention". Our house rule is purged walk-forward. Both have a role: WF mimics deployment and refits, while CPCV gives more paths for **parameter selection**.

### Methods to adopt (ranked)

| rank | method | where | effort | expected value |
|---|---|---|---|---|
| 1 | `pbo_cscv(M, S)` utility; report PBO for every grid (trend zoo, breakout, Cipher, MM grid) | `research/evaluate.py` (new function), B5 / B1 / B4 | 0.5 day | high: missing today; cheap; protects every selection claim |
| 2 | Registry-wide BHY q-values next to per-block Holm; flag A / B beliefs resting on in-sample 2 < t < 3 | `research/verdicts.py` / SCOREBOARD | 0.5 day | medium-high |
| 3 | Stationary bootstrap of vol-standardised returns with week-aligned and regime-conditional variants; validated on kurtosis, |r| ACF, weekend ratio and a path-feature MMD | B5 `sim/resample.py` (H-SYN1/2) | 1 day | medium |
| 4 | CPCV as a third selector in H-SYN1 / H-SYN2 (historical path vs generated paths vs CPCV) | B5 | 1 day | medium |
| 5 | Exit-parameter selection on simulated jump-diffusion / OU paths (Carr-LdP) | B5, B4 (MM exits) | 1 day | low-medium |
| - | GAN / TCN / signature-VAE generators | not now | high | low until transparent generators are shown to fail |

---

## 5. Draft hypotheses for future registration (proposals, NOT registered)

IDs use an `H-LIT-` prefix so they can't collide with the registry. Segments are `llm_in_sample` / `clean_holdout` as per protocol; Holm within each block.

### Block PD (decomposition)

**H-LIT-PD5: Strategy sleeves have time-series momentum**
- **Statement:** For the strategy panel (all registered strategies x universe L, daily net P&L), the next-30-day net P&L of sleeves whose trailing-180-day net P&L is > 0 exceeds that of sleeves with trailing P&L <= 0.
- **Test:** pooled panel regression of next-30d P&L (risk units) on sign(trailing 180d), with time-clustered SEs. Robustness at 90d and 365d lookbacks.
- **Success:** coefficient > 0 with t > 2 in-sample and > 0 in the holdout.
- **Prior:** 0.45. Factor momentum is strong in equities (6 vs 51 bp a month), but our sleeves' history is short and the trend sleeves flipped sign at the cutoff.

**H-LIT-PD6: Trend sleeves are long convexity, with no residual alpha**
- **Statement:** Regressing each trend / breakout sleeve's weekly net P&L on the crypto market return and a synthetic lookback-straddle return of the market (max - min over the week, minus its trailing mean) gives a positive straddle loading with t > 2, and the intercept is insignificant.
- **Test:** per sleeve and pooled; NW SEs; segments.
- **Success:** pooled straddle loading > 0 with t > 2 in both segments, and |alpha t| < 2.
- **Prior:** 0.60. Fung-Hsieh is a robust stylised fact, but crypto trend is short-horizon and noisy.

### Block NS / TF (strategies and calendars)

**H-LIT-NS6: Funding-minute and top-of-hour volatility bursts on Bybit perps**
- **Statement:** Mean 1m |r| / robust sigma in the first minute after 00:00, 08:00 and 16:00 UTC exceeds that in the first minute of other hours, which in turn exceeds minutes 2-14.
- **Test:** flow_3s / 1m bars, ratio with a date-clustered bootstrap CI, per symbol class.
- **Success:** both ratios > 1 with CIs excluding 1 for >= 2 of 3 classes.
- **Prior:** 0.80. Hansen-Kim-Kimbrough document exactly this on spot venues, growing over time.

**H-LIT-NS7: Compression of taker-flow variance precedes large moves (population level)**
- **Statement:** When the rolling 60-minute variance of 1m signed taker flow is in its bottom decile (vs trailing 7 days), the probability of a > 3 sigma move in the next 60 minutes is higher than the unconditional base rate.
- **Test:** flow_3s, logistic / binomial test with a block bootstrap; holdout only.
- **Success:** lift > 1.3 with CI excluding 1 after >= 30 days of flow.
- **Prior:** 0.25. One unrefereed n = 7 study; plausible but fragile.

**H-LIT-TF6: Equity-perp off-hours moves revert more after US-session sell-offs than after rallies**
- **Statement:** For equity perps, the signed reversal from the NYSE close to the next NYSE open is larger when the preceding NYSE session fell by more than 1 sigma than when it rose by more than 1 sigma.
- **Test:** event study clustered by date, difference of means.
- **Success:** difference > 0 with t > 2 pooled (holdout-dominated).
- **Prior:** 0.25. The BLW inventory asymmetry is strong in its home market, but the overnight drift itself has disappeared since 2021.

**H-LIT-TF7: Intraday momentum in equity perps reverts the next day**
- **Statement:** The last-hour return before the NYSE close is positively predicted by the previous-close-to-1h-before-close return, and the last-hour return is followed by a partial reversal over the next session.
- **Test:** two predictive regressions with date-clustered SEs.
- **Success:** both slopes with the predicted sign and |t| > 2 pooled.
- **Prior:** 0.25. Strong across 60+ futures (Baltussen), but the perp's flow is crypto-native and history is short.

**H-LIT-TF8: There is no pre-FOMC drift in BTC, ETH or the QQQ perp**
- **Statement (stated as the null-like prediction):** The mean vol-scaled return over the 24h before scheduled FOMC announcements is not different from other 24h windows, for BTC and ETH (2019-2026) and the QQQ perp.
- **Test:** event dummy regression with NW SEs.
- **Success (for the "no drift" statement):** |t| < 2 in both segments.
- **Prior:** 0.75. The drift decayed after 2015 in equities, and crypto has no strong reason to carry it.

**H-LIT-NS8: Deribit expiry pins BTC / ETH at 08:00 UTC**
- **Statement:** On expiry Fridays, |log(index_0800 / nearest major strike)| is smaller than on non-expiry Fridays.
- **Test:** permutation test over Fridays; monthly expiries separately.
- **Success:** median distance ratio < 0.85 with permutation p < 0.05 in-sample and the same direction in the holdout.
- **Prior:** 0.25. Pinning needs dealers long gamma, which is uncertain in crypto. The round-number proxy is weak without OI data.

### Block SIM (execution)

**H-LIT-SIM5: Pulling quotes over funding minutes and the first minute of each hour improves MM markouts**
- **Statement:** In replay with the H-SIM1 fill model, a quoter that cancels resting quotes from hh:59:50 to hh:01:00 (and +-2 minutes around funding times) has a higher net P&L per notional than the same quoter without the rule.
- **Test:** paired difference over symbol-days, bootstrap CI.
- **Success:** difference > 0 with CI excluding 0.
- **Prior:** 0.40. The bursts are documented, but lost spread capture may offset the avoided adverse selection.

**H-LIT-SIM6: Microstructure invariance scaling holds across Bybit perps**
- **Statement:** Across Bybit perps (crypto and TradFi), the cross-sectional slope of log(trades per day) on log W (W = daily notional volume x daily sigma) is in [0.55, 0.80], and the slope of log(median trade notional / daily notional volume) on log W is in [-0.80, -0.55].
- **Test:** cross-sectional OLS on 30-day averages, bootstrap over symbols, per segment.
- **Success:** both CIs overlap the stated ranges and exclude 0.
- **Prior:** 0.45. Robust in equities, but crypto trade-size distributions are shaped by retail and bots.

**H-LIT-SIM7: Fast passive fills are adversely selected**
- **Statement:** Among our live post-only fills, those filled within 3 s of placement have a worse 60 s markout (in bp, signed for us) than fills that took longer.
- **Test:** difference of mean markouts, bootstrap CI.
- **Success:** difference < 0 with CI excluding 0.
- **Prior:** 0.55. Arroyo et al. show the fill-time / adverse-selection signature on Nasdaq. The sample is ~250 fills.

**H-LIT-SIM8: A purely endogenous flow generator under-produces volatility**
- **Statement:** A Hawkes / VAR flow generator plus an impact function **without** an exogenous jump / reset component, fitted on days 1-2, produces day-3 1m realised vol < 0.7x the real value and a continuation / alternation ratio eta below the real value. Adding the exogenous component brings both inside the day-to-day envelope.
- **Test:** simulated vs real, envelope as in H-SIM2.
- **Success:** both parts hold for >= 2 of 3 classes.
- **Prior:** 0.55. HLR: 5 bp vs 14 bp and eta 0.08 vs 0.39 in a pure book-driven model.

### Block SYN (overfitting)

**H-LIT-SYN3: Trend-zoo selection is overfit (PBO > 0.5)**
- **Statement:** CSCV (S = 16 blocks of `llm_in_sample`) on the 21-variant trend zoo, per symbol and pooled, gives PBO > 0.5.
- **Test:** pbo_cscv; bootstrap over symbols.
- **Success:** pooled PBO > 0.5 with CI excluding 0.5.
- **Prior:** 0.60. Our DSR 0.72, the ~1.9 bets and the holdout reversal all point to overfit selection.

**H-LIT-SYN4: Generated-path selection helps more when historical selection is more overfit**
- **Statement:** Across parameter grids x symbols, the H-SYN1 degradation reduction (historical-path minus generated-path degradation) increases with the grid's in-sample PBO.
- **Test:** Spearman over grid-symbol units, bootstrap CI.
- **Success:** rho > 0 with CI excluding 0.
- **Prior:** 0.40. Koshiyama et al. report gains "when traditional techniques fail"; this makes that conditional claim testable.

**H-LIT-SYN5: CPCV parameter selection degrades less than walk-forward selection**
- **Statement:** For the trend-zoo and breakout grids, the parameter chosen by CPCV (N = 10 groups, k = 2, purge and embargo >= the holding horizon) on `llm_in_sample` loses less Sharpe in `clean_holdout` than the parameter chosen by purged walk-forward (the last-window winner).
- **Test:** paired difference over symbols, bootstrap CI.
- **Success:** mean difference < 0 with CI excluding 0.
- **Prior:** 0.40. One synthetic study supports CPCV; our holdout is short and regime-shifted.

---

## 6. Where the literature contradicts or qualifies our current beliefs and rules

1. **House rule "purged walk-forward"** (CLAUDE.md §5 / ARCHITECTURE §1) is questioned for **parameter selection** by Arian et al. 2024: CPCV gives lower PBO and better DSR. This doesn't affect forecast evaluation or refit cadence, where WF remains the right design. Proposal: keep WF and add CPCV + PBO for selection. Changing the rule is the user's decision (test H-LIT-SYN5 first).
2. **H-SIM2's generator design** ("marked Hawkes or VAR flow ... plus an impact function") lacks an exogenous-information component. HLR show that such endogenous models under-produce vol by ~3x and over-produce mean reversion. Expect H-SIM2 to fail on kurtosis and |r| ACF unless a jump / reset channel is added (H-LIT-SIM8).
3. **H-TF2 (night effect, prior 0.35)** probably over-rates the effect. Its strongest index-level form, the overnight drift, has been ~0 since 2021 per the original authors (2026), and the pre-FOMC drift also decayed. Classic session effects should be treated as upper bounds. ORB / intraday-momentum practitioner papers (H-TF3) assume zero slippage.
4. **H-PD1's "effective number of bets"** uses the equal-weight eigenvalue entropy. Meucci's ENB at actual weights is the decision-relevant one. This is not a contradiction, but the two can differ a lot; report both.
5. **H-PD3 (IPCA beats trailing Sharpe)**: the literature makes trailing performance (factor momentum) a strong baseline, and IPCA needs a large cross-section. The prior of 0.30 may be generous.
6. **Consistent with our beliefs.** No contradiction: crypto carry is a decaying leverage premium (C1 / R-CARRY-1; He et al., Schmeling et al.). Liquidation clusters continue rather than revert within ~30-60 minutes (R-FLOW-2; Garcia Seuma). Hawkes clustering is largely seasonality and slow vol (R-SIG-4; BMM caveats). Saturday vol is lowest (V17; Hansen et al.). Touch-based fills overstate fills (H-SIM1 already includes survival; LMZ).
7. **The literature supports R-BUDGET-1**: GKX's ML gains come from large cross-sections, and in small universes more inputs don't help.
