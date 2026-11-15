"""Build the consolidated knowledge book, an updated scoreboard and a consistency check.

    PYTHONIOENCODING=utf-8 uv run python scripts/knowledge_book.py [--date 2026-11-12]

Reads only (never edits): research/hypotheses.yaml, research/hypotheses_results.jsonl, research/rules.yaml,
research/BELIEFS.md, data/research/playbook.json, data/research/forward_tests/status.json, research/diary/index.jsonl.

Writes:
  research/KNOWLEDGE.md                         the knowledge book (topic by topic; plain language)
  data/research/knowledge.json                  the same, machine-readable, plus calibration and consistency data
  research/reports/SCOREBOARD_<date>.md         every registered hypothesis, latest verdict per segment, prior
                                                calibration, and the consistency check (recommendations only)

What is machine-generated: hypothesis verdicts and statuses, rule ids per topic (a rule belongs to every topic whose
hypotheses it cites as evidence), forward-test status, registered-but-untested lists, refuted high-prior hypotheses,
prior calibration, duplicate results, bookkeeping checks and the playbook call order. What is curated (in this file):
the 'what we know' sentences, the confidence reading, the 'do not believe' lore and the BELIEFS / rule contradictions.
Every curated claim cites hypothesis or rule ids; the build fails if a cited id does not exist, so the text cannot
silently drift from the registry. Rebuild after new results are recorded and re-read the curated text against the
regenerated verdicts.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
SEGS = ("llm_in_sample", "clean_holdout", "pooled")
CODE = {"supports": "S", "contradicts": "X", "inconclusive": "?", None: "-"}
HID = re.compile(r"\bH-[A-Z]+\d+[a-z]?\b")
RID = re.compile(r"\bR-[A-Z]+-\d+\b")

# ----------------------------------------------------------------------------------------------------------------------
# Curated content. Each topic: hypotheses it owns (for the scoreboard and status counts), what we know, the confidence
# reading, and what not to believe. Ranges such as "H-OPT1-3" cite H-OPT1 to H-OPT3.
# ----------------------------------------------------------------------------------------------------------------------

TOPICS: list[dict] = [
    {
        "key": "vol_models",
        "title": "Volatility I: forecasting models",
        "hyps": "H-V1 H-V1b H-S4 H-VCL1 H-VCL3 H-VCL4 H-VCL5 H-VC4 H-LC2 H-VM1 H-VM2 H-VM3 H-VM4 H-VM5 H-RL2 H-JD1 H-AN1",
        "know": [
            "Volatility is what we forecast well, with simple models. Daily crypto variance: per-asset log-HAR (1d/7d/30d) beats level HAR by about 12% and robust_bc x 24 by about 18% QLIKE in-sample; multi-scale HAR beat the single-lambda EWMA by 0.08 QLIKE in both segments (H-VCL1, H-VM2, H-VM5; R-VOL-20).",
            "Per bar, use the jump-robust bias-corrected EWMA (robust_bc). Excess variance after a jump has a half-life of 2-3 h, so a plain EWMA over-forecasts post-jump vol by about 27%; robust_bc wins by 0.05 QLIKE in both segments, the uncorrected robust EWMA lost (H-S4, H-V1b, H-V1).",
            "After spikes forecasts run hot: in the v0 high-vol state next-week RV comes in 13-31% further below the EWMA forecast than in calm states (t -10.9 in-sample, -5.0 holdout, 9 assets) (H-RL2; R-VOL-8).",
            "Short-history and TradFi perps: pool across assets (panel log-HAR / HAR-Q; TradFi holdout QLIKE 0.70x robust_bc x 24 at 1d, 0.21x at 5d); for 2+ years of history the panel adds nothing (H-VM1; R-VOL-22). Multi-day: a direct 5d HAR, never sqrt(h) scaling of the 1h sigma (R-VOL-21).",
            "No help to HAR from: 1h range estimators (GK-HAR +5% / +16% QLIKE worse), BTC spillover, compression terms, vector candles, illiquidity spikes, or the common crypto vol factor (it explains ~70% of alt log-RV but not the alt's forecast error) (H-VM3, H-VCL3, H-VCL5, H-VC4, H-LC2, H-VM4). DVOL helps a little at 7d for BTC/ETH (-19% on 17 holdout origins; weak in-sample) (H-VCL4). Density: Student-t (df ~3.4) on HAR variance; a Kou jump mixture did not beat it (H-JD1).",
        ],
        "confidence": "A for the core (both segments; 23 crypto perps over years plus ~132 holdout days). Refinements are B/C: log-HAR's holdout gain has t -0.42 and the panel HAR is inconclusive in the holdout.",
        "not_believe": [
            ("A jump-robust EWMA is better on its own", "lost until bias-corrected (H-V1 vs H-V1b)."),
            ("Range estimators are more efficient, so they forecast better", "worse in both segments (H-VM3)."),
            ("BTC vol spills into alts; compression predicts expansion; vector candles start a vol regime", "no gain in either segment (H-VCL3, H-VCL5, H-VC4)."),
        ],
    },
    {
        "key": "vol_calendar",
        "title": "Volatility II: calendar (weekends, Mondays, hour of week, expiries)",
        "hyps": "H-W1 H-W7 H-V2 H-DOW2 H-W5 H-W5b H-NS2 H-NS4 H-OPT1 H-OPT2 H-OPT3 H-OPT4 H-VM6 H-NN7",
        "know": [
            "Weekends are quiet: TradFi-weekend RV is a median 0.77x the Friday sigma (326 weekends; holdout 0.82x, 18 weekends); Saturday is the quietest forecast day (~0.5-0.6x Tue-Fri for crypto, 0.05-0.1x for TradFi perps) (H-W1; R-VOL-1).",
            "Weekends are also jump-prone: shrinking the risk sigma for them worsened QLIKE in both segments. Expectations use the weekend-scaled sigma; stops and risk use the unshrunk sigma_risk_next (H-V2; R-VOL-2).",
            "Hour of week is large and stable: robust sigma^2 x an expanding hour-of-week factor cuts next-hour QLIKE by 0.040 / 0.076 (crypto) and 0.29 / 0.38 (TradFi), in-sample / holdout. Peaks: Sunday 22-02 UTC and the US cash open (13-16 UTC); trough: Saturday 03-11 UTC (H-NS4; R-VOL-26, R-VOL-27).",
            "Mondays: in the crypto panel next-day variance is ~1.3x Tue-Fri once the vol level is known (cells +45% vs log-HAR in-sample, +25% holdout), but the per-asset test failed (BTC 1.33x in-sample, 0.82x holdout). SPX/NDX Mondays are not more volatile per trading day; WTI Mondays were 2.3x in the holdout (war weekends) (H-DOW2, H-DS1; R-DS-2, R-VOL-9).",
            "Implied vol overprices the weekend (BTC weekend RV/IV 0.35 vs 0.84 on weekdays, t -17; holdout 0.20, t -7.7), yet selling the Friday 08:00 straddle lost net in both segments: Friday options already price the weekend ~20% cheaper and the window holds the loud Friday US session and Sunday reopen (H-W7, H-OPT1-3; R-AVOID-2). Only a US-close entry remains, forward-only (H-OPT4). No weekly-expiry pin either (H-NS2; R-VOL-28).",
        ],
        "confidence": "A for the weekend discount and hour of week (both segments, hundreds of weekends, two asset classes). The Monday factor is B/C: panel cells and ML attribution support it, the per-asset test failed, and its forward tests (H-DS4, H-VM6) are at 0-1 of 120 dates.",
        "not_believe": [
            ("Selling weekend vol from Friday morning is free money", "the premium is real, the trade lost (H-W7 vs H-OPT1)."),
            ("Shrink the risk sigma over weekends", "prior 0.75, refuted in both segments (H-V2)."),
            ("One Monday multiplier for every asset", "TradFi Mondays 0.93-0.97x; BTC's own effect vanished in the holdout (H-DOW2)."),
            ("The Monday NYSE open is an inflection point after eventful weekends", "H-W5 failed; H-W5b holds only in-sample."),
        ],
    },
    {
        "key": "vol_asymmetry",
        "title": "Volatility III: asymmetry and the leverage effect",
        "hyps": "H-VCL2 H-LEV1 H-LEV2 H-LEV3 H-LEV4 H-LEV5 H-LEV6 H-LEV7",
        "know": [
            "Crypto has no leverage effect: a down day raises next-day vol no more than an up day of the same RV (b(RS-) - b(RS+) = -0.03, t -0.28, 2,252 dates in-sample; -0.20, t -1.57 holdout; 0 of 23 GJR gammas significant). Forecast from total RV, not the sign of the move (H-LEV2, H-VCL2; R-LEV-1).",
            "Bybit equity/index perps show no textbook effect either in the 132-day holdout (sign flips with the specification; our prior was 0.85). Drawdowns create no asymmetry and SHAR does not beat plain (log) HAR for either class (H-LEV1, H-LEV3, H-LEV4; R-LEV-2, R-LEV-3, R-LEV-4).",
            "Crowded funding plus a down day does not amplify vol (H-LEV5, all segments); crowding alone tilted next-day vol up ~+0.07x mean RV per z (t 2.4 / 2.7, exploratory), now the forward test H-LEV6. A market-wide down-day effect is H-LEV7 (forward).",
        ],
        "confidence": "Crypto A (both segments, 23 assets, years). TradFi C (holdout only, 132 days, 10 equity perps).",
        "not_believe": [
            ("Down moves raise crypto vol more than up moves", "refuted in both segments (H-VCL2, H-LEV2)."),
            ("Crowded longs plus a red day means liquidation vol", "refuted in all segments (H-LEV5)."),
        ],
    },
    {
        "key": "vol_intraday",
        "title": "Volatility IV: intraday (next hour, next minute, jumps)",
        "hyps": "H-NN8 H-NN10 H-NN11 H-NN12 H-NN5 H-JD2 H-VB4",
        "know": [
            "Next hour: the baseline robust_bc x hour of week beats bare robust sigma by 0.048 / 0.076 QLIKE. Nonlinear models beat the baseline in both segments for crypto (XGB -0.040 / -0.065, MLP -0.036 / -0.065; linear on the same inputs -0.025 / -0.029). Half the gain is inputs, half nonlinearity; the inputs that matter are recent |r| over 1-24 h (41% of attribution), the daily HAR level (21%) and hour of week (24%); funding-hour and jump flags add nothing (H-NN8; R-ML-10, R-ML-7).",
            "TradFi perps gain more (holdout MLP -0.176, t -6.9, 15 of 15 symbols; linear -0.131; XGB -0.102), mostly on holdout data; equities improve only with the MLP (H-NN11; R-ML-20).",
            "Cross-asset activity (BTC, ETH, QQQ) adds nothing at 1 h (XGB ~0, MLP worse), though BTC carries half the cross-asset attribution (H-NN12; R-ML-21, R-ML-22). Token-to-token vol transmission is a seconds-to-minutes effect (H-MS6), not an hourly one.",
            "Next minute (1 s data, 11 days, 347 symbol-days): the signal is the activity level of the last 1-5 minutes. A linear model of 10/60/300 s aggregates is ~10x better than trailing variance; a CNN adds only 0.8% QLIKE; signed flow, imbalance, spread and depth add nothing (H-NN5; R-ML-11, R-ML-12).",
            "Jumps cluster, but barely usefully (Hawkes vs constant rate: Brier skill +0.004 in-sample, +0.0002 holdout); for 2-sigma bar counts HAR + Student-t beats Hawkes (H-JD2, H-VB4; R-EXEC-3, R-SIG-4). The frozen hourly XGB (tools.vol_ml) is for expectations only until H-NN10 passes; never for stops.",
        ],
        "confidence": "B for crypto hourly (both segments, 23 symbols, two model families agree); C for TradFi hourly (mostly holdout) and for next-minute (11 days). H-NN10 is at 0 of 60 forward days.",
        "not_believe": [
            ("Cross-asset leaders improve own-vol forecasts at 1 h", "no; the MLP gets worse (H-NN12)."),
            ("A deep net on raw 1 s flow is needed for next-minute vol", "+0.8% over a linear activity model (H-NN5)."),
        ],
    },
    {
        "key": "microstructure",
        "title": "Microstructure: trade-level laws",
        "hyps": "H-MS1 H-MS2 H-MS3 H-MS4 H-MS5 H-MS5b H-MS6 H-MS7 H-MS8 H-MS9 H-MD1 H-MD2 H-MD3 H-MD4 H-MD5 H-MD6 H-MINE1",
        "know": [
            "Trades arrive in bursts, never Poisson: a 3-component exponential mixture fits best (~30% bursts of ~5 ms, 40% gaps of 0.2-0.4 s, 30% gaps of 1-5 s); ACD(1,1) leaves most clustering in its residuals; a boosted AFT duration model beats ACD only for majors (H-MS1, H-MS9; R-MS-1, R-MS-2, R-MS-7).",
            "Order signs have long memory (lag-1 ACF 0.33-0.39, power-law decay gamma 0.45-0.75, 1 s signed-flow Hurst 0.53-0.54); on 16 random days 79% of symbol-days prefer a power-law ACF, just under the registered 80%. Merge same-ms same-side prints into one order (H-MS3, H-MD3 interim; R-MS-3, R-MS-4).",
            "Sizes: log-normal body with a heavy tail (index 1.4-1.8), not a pure power law. Volume per interval is log-normal, persistent and co-moves with |return| (Spearman 0.40-0.55) (H-MS2, H-MS4; R-MS-5, R-MS-6).",
            "Activity scales, character does not: order rate (~W^0.54), depth (~W^0.68) and impact (~W^-0.75, near Kyle-Obizhaeva) follow trading activity W, but per-symbol traits (sign gamma, ACF1, size tail, mixture weights) are not stable across 16 days (ICC 0.23-0.31): calibrate per day (H-MS7, H-MD1 and H-MD2 interim; R-SIM-10).",
            "39 of 51 perps sit at one tick, so their spread is tick/price and says nothing about vol; for free-spread perps (CARV, KAS, NVDA) the gain is a coin flip so far (16 of 32 symbol-days). No sub-minute return lead from BTC/ETH/SOL (peaks within 20 ms), but ETH and SOL are net vol transmitters on 15 of 15 days (H-MS5, H-MD4, H-MS6, H-MD5; R-MS-11, R-MS-12, R-MS-13).",
            "Simulator v1 beats v0 in every class but never reproduces 4 of 5 facts for majors on any of 16 days, so R-SIM-13 is retired (H-MS8, H-MD6). Micro-state FDR mining froze 550 discoveries (nearly all activity effects); confirmation pending (H-MINE1).",
        ],
        "confidence": "C throughout: one pilot day (the 2026-11-10 CPI sell-off) plus 16 of 30 planned random days; final H-MD verdicts after batches 4-6. Many pilot 'contradicts' are one failed clause of a composite criterion (e.g. H-MS1 on ACD residuals): read the numbers.",
        "not_believe": [
            ("Trades are Poisson, or ACD removes the clustering", "both fail (H-MS1)."),
            ("Each perp has fixed microstructure traits (one universal or per-symbol exponent)", "ICC 0.23-0.31 across days (H-MD1 interim, H-MS7)."),
            ("The spread predicts vol", "on tick-bound perps the sign flipped out of sample (H-MS5)."),
        ],
    },
    {
        "key": "direction",
        "title": "Direction: what fails, and why",
        "hyps": ("H-W2 H-W3 H-W4 H-W6 H-S1 H-P1 H-P2 H-VC1 H-VC2 H-VC3 H-VC5 H-VC6 H-VC7 H-VZ1 H-VZ2 H-VZ3 H-VZ4 H-GT3 H-AU1 H-AU2 "
                 "H-AU3 H-AU4 H-FU1 H-FU2 H-DOW1 H-DOW3 H-RL1 H-SEC1 H-SEC2 H-SEC3 H-SEC4 H-PAN1 H-PAN2 H-PAN3 H-PAN4 H-PAN6 "
                 "H-EV1 H-EV2 H-RV1 H-RV2 H-RV3 H-RV4 H-BO1 H-MC1 H-VB1 H-VB2 H-VB3 H-NS1 H-NS3 H-NS5 H-TP1 H-TP1b H-TF1 H-TF2 "
                 "H-TF3 H-TF4 H-TF5 H-AN2 H-PM1 H-PM2 H-NN2 H-NN6 H-DS3 H-MINE2b"),
        "know": [
            "Next-day direction from standard features is a coin flip: Brier skill +0.0004 with one feature, -0.015 with 32 (34k symbol-days). ML has no direction skill, 1 s flow gives next-minute AUC 0.505-0.507, no discrete state predicts direction, and mined return effects replicate at chance (1 of 11) while mined vol effects replicate 54 of 79 (H-IC1, H-NN2, H-NN6, H-DS3, H-MINE2; R-BUDGET-1, R-ML-5, R-DS-6).",
            "Levels are not walls: vector-candle opens, swing pivots, CME gaps, swept zones, value-area/POC and floor pivots behave like random-walk levels, and price breaks through slightly more often than it holds (~0.52-0.55) (H-VZ1-4, H-VC2, H-VC3, H-GT3, H-W4, H-AU1-4, H-BO1; R-PX-4, R-PX-5, R-PX-10).",
            "Indicators and positioning lore have no net edge: Market Cipher B dots, squeeze releases, vol-surprise continuation, variance management, funding contrarian, OI divergence, stablecoin growth, VRP as direction (H-MC1, H-VB1-3, H-P1, H-P2, H-FU1, H-FU2; R-AVOID-1, R-AVOID-3).",
            "Calendar and sessions carry no direction: no Monday drift (9 assets), no session seasonality (F 1.86, p 0.13; ranking flips), no weekend-to-Monday information, and no TradFi session effect on Bybit perps (gap fill, night premium, ORB, intraday momentum, CME-reopen continuation); fading >1-sigma overnight moves at the open lost 67 bp net per trade in the holdout (H-DOW1, H-NS3, H-W2, H-W3, H-W6, H-TF1-5, H-TP1; R-PX-6, R-PX-9, R-AVOID-4, R-TF-1).",
            "Cross-section and relative value: sector momentum, within-sector outliers, residual momentum and low idio-vol fail in-sample (survivorship-biased universe); RV spreads revert (XAUT-XAU ~28% per 24 h) but their 9-14 bp sd is below the ~30 bp round-trip cost (H-SEC1-3, H-PAN1, H-PAN2, H-RV1-3).",
            "Why: our features describe the vol state and vol has no sign; most lore was learned on the period Claude has read about; the thin effects that exist are smaller than 7.5 bp per side. Holdout-only 'wins' (segment flips listed below) are watched, not traded.",
        ],
        "confidence": "A/B for the negatives: tens of thousands of observations, both segments, Holm control in the newer blocks. The holdout is ~4 months, so small positive effects cannot be excluded; flips are the open risk.",
        "not_believe": [
            ("CME gaps always fill", "random-walk rate in-sample; the holdout 'support' is a flip (H-W4)."),
            ("Levels hold, taken levels switch off, untaken opens are magnets; the 80% rule; the POC magnet", "all at random-walk rates (H-VZ1-4, H-VC3, H-AU1-4)."),
            ("Mondays drift, sessions have a direction, BTC bounces after a down CPI day", "no (H-DOW1, H-NS3, H-EV2)."),
            ("Fade crowded funding; rising OI is bullish", "no; OI divergence is market beta (H-P1, H-P2; R-PD-4)."),
            ("The regime cell tells next week's direction", "up minus down 0.03 sigma, t 1.0 (H-RL1)."),
        ],
    },
    {
        "key": "strategies_regimes",
        "title": "Strategies and regimes",
        "hyps": ("H-VBOS1 H-VBOS2 H-VBOS3 H-VBOS4 H-VBOS5 H-R1 H-R2 H-S2 H-S3 H-T1 H-X2 H-X2b H-X2c H-X3 H-TR1 H-TR2 H-TR3 H-TR4 H-TR5 H-TR6 H-GT1 H-GT2 H-KF1 H-KF2 "
                 "H-RG1 H-RC1 H-RC2 H-RL3 H-C1 H-C1b H-SYN1 H-SYN2 H-JD3"),
        "know": [
            "Trend is the only family with an edge, and it is thin: in the 21-variant zoo the best variant had Sharpe +0.50 and the ensemble +0.33 before the cutoff (deflated Sharpe of the winner 0.72 < 0.95); every variant is negative Jul-Nov 2026 (H-TR1, H-X2b; R-PX-3).",
            "Selecting variants does not help: walk-forward selection loses to the ensemble; per-symbol parameter choice is a coin flip (PBO 0.44 zoo, 0.65 breakouts; holdout breakout champions did worse than a random pick); generated paths do not fix it (H-TR2, H-SYN1, H-SYN2; R-SYN-1, R-SYN-2).",
            "'Is it trending?' is not reliably forecastable: classifiers, directional efficiency, variance ratio, auction balance and Kalman SNR fail in-sample; the 'trend moment' probit/XGB show holdout skill only (Brier skill +0.10 / +0.16, CI includes 0) (H-TR3-6, H-RG1, H-KF2; R-PX-11).",
            "The regime map tells where strategies lose, not who wins: per-cell rankings do not persist (Spearman 0.085, CI includes 0); robust losers are 1h mean reversion in trend/compression/grind-down cells and 1h trend in range/whipsaw cells; dropping in-sample-negative cells beat equal weight in the holdout but stayed negative (Sharpe -0.77 vs -2.48) (H-RC1, H-RC2, H-R1, H-R2; R-MAP-1-7).",
            "Hourly deviations from trend continue; fading 2-sigma hourly moves loses even before costs (H-T1; R-PX-2). 'Calm shocks continue, volatile shocks reverse' is data-generated and weakly tested (holdout t 1.6 on 152 R4 events, n 13 in R9); the registered H-S1 failed and the jump version H-JD3 flipped. shock_regime is a forward test (H-S3), not knowledge (H-S2; R-PX-1).",
            "Regimes describe the vol path: median episode 5-7 days, daily stay 0.84-0.89, older episodes end sooner; after a liquidation regime expect high-vol whipsaw (62%) (H-RL3; R-PX-8). Funding carry was real (net Sharpe 2.8 in-sample) but decayed (57%/yr 2021, 0.5% 2025, -4.8% 2026 H1; holdout net t 1.3): a leverage-mania premium (H-C1; R-CARRY-1; forward H-C1b).",
        ],
        "confidence": "B: trend facts in both segments but a 132-day holdout; the regime-map grades rest on a context table whose rank test (H-RC1) is inconclusive; shock and carry claims are C until their forward tests report.",
        "not_believe": [
            ("Trend earns only in trend cells; mean reversion earns in ranges", "both refuted (H-R1, H-R2)."),
            ("4h/1d is the trend sweet spot", "the scale-free re-test failed in both segments (H-X2b); H-X2c was never evaluated."),
            ("High trailing vol kills trend; pivot geometry adds over slow trend", "refuted (H-X3, H-GT1, H-GT2)."),
            ("Funding carry is a standing premium", "it decayed to about zero (H-C1)."),
        ],
    },
    {
        "key": "pnl_structure",
        "title": "PnL structure of the deterministic book",
        "hyps": "H-PD1 H-PD2 H-PD3 H-PD4 H-PD5 H-PD6",
        "know": [
            "The book is roughly one bet: ~8 eigen-bets in the 16-sleeve correlation matrix, but 1.15 effective (Meucci) bets in the equal-weight book in-sample and 2.5 in the holdout; crypto trend dominates (H-PD1; R-PD-1).",
            "Trend = crypto TSMOM factor (loading 0.36-0.79, t 7-20) + long convexity; 1h mean reversion = short convexity (loading -0.51, t -12.8), so it does not hedge trend in stress; OI divergence = market beta. No sleeve has alpha in both segments; factors explain ~25% of daily variance (H-PD2; R-PD-2, R-PD-3, R-PD-4, R-PD-5).",
            "Edge leaks through exits, not fees: ~70-75% of the favourable excursion is lost, costs are 5-9% of it (H-PD4; R-PD-6). But that leakage is the price of the convex tail days: k-sigma close trails (k 1.5-3.5) cut in-sample P&L at every k (t -2.1 to -2.3; Sharpe 0.86 -> -1.13 to 0.27) by cutting exposure from 0.86 to 0.10-0.25; their holdout 'gain' is an exposure cut on a losing book (H-PD6; R-EXIT-1, R-EXIT-2, R-EXIT-3).",
            "Strategy returns reverse week to week, so trailing-Sharpe ranks carry negative information and a context IPCA ranking beats them (IC 0.10, t 3.2 / 0.13, t 1.2) (H-PD3; R-PD-7; forward H-PD5).",
        ],
        "confidence": "B: H-PD1-4 decided in both segments; MFE uses bar highs/lows, so 'exit leakage' is an upper bound; H-PD5's first forward week pair is 2026-11-30.",
        "not_believe": [
            ("A 16-strategy book is diversified; a positive raw Sharpe is alpha", "1-2.5 bets; factor exposure plus noise (H-PD1, H-PD2)."),
            ("Fees are the main drag; back last week's winner", "exits are; trailing ranks reverse (H-PD4, H-PD3)."),
            ("Tighter trailing stops recover the exit leakage", "they cut the trend tails and lose in-sample (H-PD6; R-EXIT-1)."),
        ],
    },
    {
        "key": "execution_mm",
        "title": "Execution, risk mechanics and market making",
        "hyps": "H-X1 H-JD4 H-F5 H-MM1 H-MM2 H-MM3 H-NS7 H-SIM1 H-SIM2 H-SIM3 H-SIM4 H-AS1 H-AS2",
        "know": [
            "Keep resting exchange stops (MarkPrice trigger): close-confirmed stops had a worst trade of -1.9R vs -1.0R (H-X1; R-EXEC-1). Jump-aware first-passage odds are better calibrated for brackets than a Gaussian (Brier -0.0022 / -0.0014, CIs below 0) (H-JD4; R-EXEC-2).",
            "Live MM (H-MM2, 2 of 10 run-days): passive quote-and-clear made money (passive exits 84-85% winners, +0.55 USD over 43 trips); market-order exits gave it back (-0.54 USD over 42 trips). ~50 crypto contracts are one correlated bet (5m correlations 0.85-0.96) (R-MM-1, R-MM-2).",
            "NVDA micro MM (0% maker): quoting captured ~+9 cents per NVDA traded, but a bullish inventory on a falling NVDA lost more (total ~-0.16 USD): inventory direction, not spread capture, decided P&L. During a machine sleep, resting quotes stayed live and unmanaged (no disconnect-cancel-all) (diary 2026-11-12).",
            "Simulator v0: fills consistent with 23 live fills (inconclusive); generated flow fails stylised facts; reaction functions do not transfer across markets; generated paths rank MM parameters by activity, not P&L (Spearman -0.09). Rank on real 3 s replays (one sell-off day: ask 0.2 / bid 0.9 sigma earned +14.2 bp of turnover vs +7.5 bp live) (H-SIM1-4; R-SIM-1-7; paired live check H-MM3 blocked).",
            "Fees: crypto 2 bp maker / 5.5 bp taker; TradFi perps 0% maker / 2.75 bp taker; research assumes 7.5 bp per side. Open design conflict for the user: breakout structural stops sit 6-8 sigma from entry vs a 4-sigma cap (BELIEFS F2, section 4).",
        ],
        "confidence": "C for market making (2 live run-days, 1 replay day, 23 test fills); B for stops (H-X1) and bracket odds (H-JD4).",
        "not_believe": [
            ("Market-order exits are fine for MM inventory; many symbols diversify", "they gave back the edge; one correlated bet (R-MM-1, R-MM-2)."),
            ("A flow generator can rank MM parameters", "it ranks activity, not P&L (H-SIM4)."),
        ],
    },
    {
        "key": "order_flow",
        "title": "Order flow",
        "hyps": "H-F1 H-F2 H-F3 H-F4 H-OF1 H-OF2 H-OF3 H-OF4 H-OF5 H-OF6 H-OF7 H-PAN5",
        "know": [
            "Flow data exists only after the cutoff (flow_3s from 2026-11-07, liquidations from 2026-11-08): all clean holdout, about 3-5 usable days, one market state.",
            "The registered |z| > 4 imbalance detector fires on 22% of minutes (signed flow is far heavier-tailed than its MAD scale; a symbol's top 1% starts near |z| 56). Afterwards |return| is only 1.14x at 5 m, 1.05x at 15 m, sign rate 0.513 (H-OF1, H-OF2; R-FLOW-1). A selective top-1% detector shows 1.48x (CI 0.93-1.63, 309 events): forward H-OF6.",
            "Liquidation clusters continued, not reverted (-1.85 sigma at 60 m in the liquidation direction, t -1.9, n 142): forward H-OF7 (interim +1.05 on 2 of 40 cluster-hours) (H-OF3; R-FLOW-2). BTC flow does not lead alts; a multivariate score is no better than the imbalance z (H-OF5, H-OF4; R-FLOW-3, R-FLOW-4).",
            "At 1 s, signed flow and book imbalance add nothing for next-minute vol or direction (H-NN5, H-NN6): read flow as an activity gauge. L1-OFI, depth, spot-perp and weekend-flow tests (H-F1-5) wait for 30+ days of flow_3s.",
        ],
        "confidence": "C: about 3-5 days, one market state (includes the CPI sell-off), collector gaps.",
        "not_believe": [
            ("A big taker imbalance tells direction", "sign rate 0.51 (H-OF2)."),
            ("Fade liquidation cascades; BTC flow leads alts by minutes", "they continued; no lead (H-OF3, H-OF5)."),
        ],
    },
    {
        "key": "ml",
        "title": "ML, discrete states and interpretability",
        "hyps": "H-NN1 H-NN3 H-NN4 H-NN9 H-DS1 H-DS2 H-DS4 H-SP1 H-SP2 H-SP3 H-SP4 H-MINE2 H-MINE2a",
        "know": [
            "Daily vol: MLP and XGB beat log-HAR in both segments (MLP -8.0% in-sample, t -2.9; XGB -15.7% holdout, t -2.4, 22 of 23 symbols), but the winning family flips between segments and most of the gain is calendar: log-HAR + weekday dummies gets -5.5% alone (H-NN1; R-ML-1, R-ML-4).",
            "What models use: the vol level (7d log RV, robust sigma; ~40% of attribution; +75% QLIKE when ablated) and the forecast weekday; trend, funding, market 1-3% each; jumps, semivariances, ranges, raw returns nothing. Interactions are ~1/3 of attribution with the same top-5 pairs in both segments; trees and nets agree on vol drivers (rank corr 0.75), not on direction (0.22) (H-NN3; R-ML-3, R-ML-6).",
            "The vol net is tiny underneath: 95% of weights prune away (100 of 2,192) with no loss (QLIKE ratio 0.986 in-sample, 0.974 holdout, CI to 1.027). Minimal inputs: one vol-level measure + the calendar; single vol columns substitute for each other, so column-level stability selection found nothing (H-SP2 failed). Survivors are mostly simple 2-input units (17 of 24), the strongest being calm-weekend dampers; the circuit is redundant (H-SP1, H-SP2, H-SP3; R-SP-1, R-SP-2, R-SP-3).",
            "A 36-cell table (weekday group x vol tercile x v0 trend) beats log-HAR by 6-7% in both segments but recovers only ~half of XGB's holdout gain (it misses weekend depth); cell ordering Spearman 0.76 (H-DS1, H-DS2; R-DS-1-5; forward H-DS4).",
            "Sparse autoencoders (16 and 64 units) find no named concepts (1 and 0 of 10); explain forecasts with grouped TreeSHAP and ablation; wider nets add nothing (H-NN4, H-NN9; R-ML-6, R-ML-8, R-ML-9). Daily FDR mining (495 effects) re-found known rules: 74 of 90 discoveries were restatements; 2 of 16 novel ones replicate, both marginal (H-MINE2; forward H-MINE2a, H-MINE2b).",
        ],
        "confidence": "B for 'vol level + calendar is the signal' (both segments; two model families and three methods agree). Specific ML tools are C until forward tests pass (H-NN10 0/60, H-DS4 0/120, H-SP4 0/3).",
        "not_believe": [
            ("Bigger networks or more features help; ML finds direction", "no (H-NN9, H-IC1, H-NN2)."),
            ("SAE features are market concepts; an FDR-controlled novel discovery is probably real", "1/10 and 0/10; 2 of 16 replicated (H-NN4, H-NN9, H-MINE2)."),
        ],
    },
    {
        "key": "btc_field",
        "title": "BTC field program: probability field, polynomial networks, news, generated markets and books, live interpretation",
        "hyps": "H-FLD1 H-FLD2 H-FLD3 H-FLD4 H-PNN1 H-PNN2 H-PNN3 H-NEWS1 H-NEWS2 H-NEWS3 H-NEWS4 H-OBG1 H-OBG2 H-GEN1 H-GEN2 H-GEN3 H-LIVE1 H-LIVE2",
        "know": [
            "The BTC probability field (Fokker-Planck with jumps, sparse polynomial drift / diffusion / jump intensity) beats log-HAR-t with constant jumps only through hour-of-week; against log-HAR-t x hour-of-week it ties at 1h and loses at 4h and 1d; online coefficient updating adds nothing; drift is a constant. Momentum x trend agreement raising variance is a forward test (H-FLD1, H-FLD2, H-FLD3, H-FLD4).",
            "Small networks (~500 parameters) beat log-HAR-t for 1h BTC return quantiles in both segments, but polynomial (KAN-style) neurons add nothing over an equal-size MLP; adaptive conformal bands keep coverage within 0.2 pp; the latent state is predictable yet feeding it back adds nothing (H-PNN1, H-PNN2, H-PNN3).",
            "News attention (Wikipedia pageviews and edits, Hacker News counts, Fear and Greed), the event calendar and on-chain activity make next-day BTC vol forecasts 3-4% worse and carry no direction skill; GDELT was not testable (H-NEWS1, H-NEWS2, H-NEWS3, H-NEWS4).",
            "Generated BTC order books run through the Bybit-semantics matching engine reproduce 6 of 8 held-out book facts (signed-flow persistence and taker size miss), but generated twin days do not improve a next-minute vol model and hurt when real data is scarce (H-OBG1, H-OBG2). The daily market generator reproduces stylised facts yet cannot rank strategies (H-GEN2, H-GEN3).",
            "Live: the frozen network runs hourly and Claude logs its own interpreted forecasts before each window opens; whether interpretation beats the network is forward-only (H-LIVE1, H-LIVE2).",
        ],
        "confidence": "B for the negatives (both segments agree); C for everything forward (H-FLD4, H-LIVE1, H-LIVE2).",
        "not_believe": [
            ("Richer structure (fields, polynomial neurons, latent feedback, news) beats a calendar-aware vol baseline", "not out of sample (H-FLD1, H-PNN1, H-PNN3, H-NEWS3)."),
            ("Generated data can stand in for real data", "not for training or strategy ranking (H-OBG2, H-GEN3)."),
        ],
    },
    {
        "key": "decision_budget",
        "title": "Decision budget: what to look at, how much history, how fast",
        "hyps": "H-IC1 H-IC2 H-IC3 H-IC4",
        "know": [
            "Direction skill peaks at ~1 feature and falls as features are added (Brier skill +0.0004 at k = 1, -0.015 at k = 32, 1,703 dates); the holdout peaks at k = 8 with every CI including 0 (H-IC1; R-BUDGET-1).",
            "Fit on 1-2 years or expanding history; 90-day windows are worse (Brier +0.0012 / +0.0026, t 1.6 / 1.9) and no shorter window wins anywhere (H-IC2; R-BUDGET-2).",
            "Latency: the 4h trend keeps 97% of its edge after 4 h of delay (88% after 8 h); shock edges keep 61% after 1 h, 10% after 2 h. Shock signals execute at the bar close with an LLM veto only (H-IC3; R-BUDGET-3, R-BUDGET-4).",
            "So: short cards (8 or fewer tested lines), no extra inputs for directional calls, deliberate on slow signals, never delay fast ones. Whether the compact card beats a full tool dump for the LLM is forward-only (H-IC4).",
        ],
        "confidence": "B/C: H-IC1-3 are formally inconclusive (the holdout lacks power) although the in-sample samples are large (34k symbol-days).",
        "not_believe": [
            ("More context makes better calls; recent data is more relevant", "skill falls with features; short windows lose (H-IC1, H-IC2)."),
        ],
    },
]

DATA_INFRA = {
    "key": "data_infra",
    "title": "Data and infrastructure lessons",
    "know": [
        "Point-in-time must be enforced in code: inputs need event_ts < T, funding and OI count only if known_ts <= T, and known_ts was clamped because the exchange clock ran ahead of local time (diary 2026-11-07).",
        "Most errors were test design: wrong nulls (flat 0.5 instead of the random-walk race, H-GT3), unreachable criteria (H-VCL5), standardisation that absorbed the effect (H-W5), metrics not scale-free across frequencies (H-X2), an extra verdict conjunct (H-LEV2). Corrections are appended, never overwritten.",
        "Silent failures recur: Int8 overflow, numpy NaN passing a polars is_not_null filter (use is_finite), optimisers that overflow or collapse, universes that drop symbols (10 of 16 TradFi perps lacked native 4h history).",
        "Storage and ops: only 3 s flow candles are stored (~1.7 MB per stream-day, ~200 days in 30 GB); raw micro batches are deleted after verified aggregation (lake + research 3.07 GB after batch 3). Loops die with their owning session (updater and collector were down from 2026-11-11 02:46), a machine sleep froze agents for 3 h, and long pipelines need resume and atomic downloads.",
        "Samples are small where it matters: survivorship-biased universe, TradFi perps listed Mar-Jun 2026 (2-3 months in-sample), a ~132-day daily holdout. Verdict labels compress: 'contradicts' can be one failed clause (H-MS1, H-SP2), 'SUPPORTED' can hide decay (H-C1).",
    ],
    "not_believe": [
        ("An in-sample 'supports' is new evidence", "that period is what Claude has read about; only the clean holdout is unmemorised (CLAUDE.md principle 6)."),
    ],
}

# Curated consistency findings (recommendations only; nothing is edited). Belief ids are checked against BELIEFS.md.
BELIEF_CONFLICTS = [
    ("P5", "T1", "P5 (v1) says Bybit TradFi perps freeze on weekends; T1 (v3) measured that they trade 24/7 (weekends 0.1-0.15x turnover, the index moves). R-PX-9 was corrected, but P5 still states the freeze.",
     "Annotate P5 as superseded by T1."),
    ("V10", "N2", "V10 (v1) says 'no blanket Monday vol multiplier' (BTC 1.3x in-sample, 0.82x holdout); N2 and DS1 (v9) size a crypto Monday factor of about 1.3x (+25-45% vs log-HAR in both segments). The audit narrowed R-VOL-9 to TradFi and oil, but V10 still reads as a general statement.",
     "Split V10: crypto next-day vol uses the weekday factor (B/C, decided forward by H-DS4 / H-VM6); TradFi and oil keep the V10 wording."),
    ("M7", "MD1", "M7 (v8, one day) calls sign memory, size tail and arrival mixture 'persistent per-symbol traits'; MD1 (v9, 16 random days) finds them NOT stable per symbol (ICC 0.23-0.31). R-SIM-11 still says to use each perp's own estimates.",
     "When H-MD1 is recorded, rewrite M7 and R-SIM-11 to per-day calibration; until then mark M7 as contested."),
    ("P3", "P7", "P3 (v1) says 4h beats 1h for trend; the scale-free re-test H-X2b contradicts that in both segments, and the 4h claim rests on H-X2c, which is data-generated and not collected by the forward runner. R-PX-3 repeats 'better at 4h than 1h'.",
     "Drop the 4h-superiority clause from P3 and R-PX-3, or add H-X2c to the forward runner."),
    ("V2", "V2", "Section 5 (v1) still lists 'weekend short-vol on Bybit options' as priority 2 'to research', while V2 (updated) and H-OPT1-3 say the trade is not harvestable as tested.",
     "Mark section 5 as the historical initial book; the current book is shock_regime (forward), the trend baseline and research-only MM."),
    ("V13", "N1", "V13 (v6) names the per-asset log-HAR the best daily crypto model; N1 (v9) finds MLP and XGB beat it in both segments, though the gain is mostly calendar structure. The playbook still lists R-VOL-4 (level HAR vs EWMA) tools first, which the audit calls superseded by R-VOL-20.",
     "State one pipeline in BELIEFS: log-HAR level, then the weekday factor, then (C, only once validated forward) cells or ML."),
    ("SP2", "SP3", "SP2 (grade B) and SP3 (grade C) state findings whose registered tests (H-SP2, H-SP3) are recorded 'contradicts' in every segment; the beliefs rest on an unregistered re-reading (cluster-level stability, descriptive unit fits).",
     "Cap SP2 at C until H-SP4 (forward) reports; keep SP3 as explanation only."),
    ("P1", "P2", "P1 (v1, B/C) says shocks into volatile markets reverse; P2 says never fade 2-sigma hourly moves. Later evidence weakened P1: H-S1 failed in both segments, the jump-detected version H-JD3 flipped (in-sample supports, holdout contradicts), and the holdout support of H-S2 is t 1.6 (n 152) and n 13 in R9.",
     "Grade P1 / R-PX-1 C until H-S3 is recorded, and add H-S3 to the forward runner (it is not collected today)."),
    ("S3", "S3", "S3 (v3) says the edge leaks through exits and names H-PD6 as the forward work; H-PD6 is now recorded (2026-11-13): trailing exits lose in-sample, so 'work on exits' did not pay in its first registered form.",
     "Append to S3 that tight trails failed (H-PD6) and that the leakage is the cost of the convex tail days."),
    ("F1", "F1", "Section 4 (v1) says order flow is 'untested until 30 days of flow_3s'; F1 (v6) already reports a 3-day first measurement. Not a contradiction, but section 4 is stale.",
     "Point section 4 to F1 and the forward tests H-OF6 / H-OF7."),
    ("M6", "N9", "M6 (one day) says ETH / SOL / DOGE transmit volatility and R-MS-13 says widen or pause quotes; N9 says cross-asset activity adds nothing to next-hour forecasts. Reconciled by horizon (seconds to minutes vs one hour), but neither line says so.",
     "Add the horizon to both lines."),
]

RULE_FINDINGS = [
    ("R-SIM-11", "Rests on H-MS7 (one day); the multi-day interim H-MD1 (16 days, ICC 0.23-0.31) says the opposite: traits are not persistent per symbol.", "Rewrite to per-day calibration when H-MD1 is recorded."),
    ("R-PX-1", "Rests on H-S2 (data-generated; holdout t 1.6 in R4, n 13 in R9). The registered H-S1 failed in both segments and the later H-JD3 flipped (holdout contradicts).", "Downgrade to C until H-S3 reports; H-S3 is not in the forward runner."),
    ("R-PX-3", "Its clause 'better at 4h than 1h' is contradicted by H-X2b in both segments; the support (H-X2c) was never evaluated.", "Remove the clause or evaluate H-X2c."),
    ("R-SP-1", "Grade B, but H-SP1 is inconclusive in the holdout and H-SP2 (cited) is contradicts everywhere; the 'vol level + calendar' claim stands on H-NN3 and the cluster re-reading.", "Keep B only through H-NN3; cite H-SP4 as the deciding test."),
    ("R-SP-3", "Cites H-SP3, recorded contradicts in all segments (the ablation criterion failed).", "Keep as explanation only (it is C / provisional); never as a reason to change risk."),
    ("R-SIM-10", "Activity scaling holds, but H-MD2 interim finds a significant day x log W interaction (exponents move across days, Wald p 0.0003 for rate).", "Note 'exponents vary by day' when H-MD2 is recorded."),
    ("R-MS-3", "Long memory holds, but H-MD3 interim share is 0.788, just under the registered 0.8, after quiet days were added.", "Wait for the final H-MD3; wording 'on most symbol-days'."),
    ("R-ML-4", "Says 'prefer gradient-boosted trees'; under H-NN1 the winning family flips (MLP in-sample, XGB holdout), and H-DS1 found the XGB in-sample gain to be noise (-4.5%, t -0.98).", "Name 'the nonlinear tool that passes its forward test', not XGB."),
    ("R-VOL-4", "Superseded by R-VOL-20 (audit item 6), yet the playbook still lists its tools first for next_day_vol.", "Mark superseded in its review field so the card leads with log-HAR."),
    ("R-TF-1", "'Do not fade' rests on a post-hoc holdout finding with the opposite sign in-sample; its forward twin H-TP1b (continuation) has an interim negative estimate (4 of 150 symbol-sessions, far too few to read).", "Keep, but note that H-TP1b decides it."),
    ("R-PD-6", "Says 'prefer work on exits'; its registered follow-up H-PD6 is contradicts in-sample (k-sigma trails cut P&L at every k) and its holdout pass is an exposure cut; R-EXIT-1..3 now encode this, but R-PD-6 does not cite H-PD6.",
     "Add H-PD6 to R-PD-6's evidence and narrow it to 'exit designs other than tight close trails, each registered first'."),
    ("R-EXEC-3", "H-JD2 is 'supports' in both segments, but the holdout Brier skill is +0.0002 with a CI including 0.", "Consider B -> C."),
]

# ----------------------------------------------------------------------------------------------------------------------


def load_sources() -> dict:
    reg = yaml.safe_load((ROOT / "research" / "hypotheses.yaml").read_text(encoding="utf-8"))["hypotheses"]
    rows = [json.loads(x) for x in (ROOT / "research" / "hypotheses_results.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
    rules = yaml.safe_load((ROOT / "research" / "rules.yaml").read_text(encoding="utf-8"))["rules"]
    beliefs = (ROOT / "research" / "BELIEFS.md").read_text(encoding="utf-8")
    fwd = json.loads((ROOT / "data" / "research" / "forward_tests" / "status.json").read_text(encoding="utf-8"))
    pb = json.loads((ROOT / "data" / "research" / "playbook.json").read_text(encoding="utf-8"))
    diary = [json.loads(x) for x in (ROOT / "research" / "diary" / "index.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
    return {"reg": reg, "rows": rows, "rules": rules, "beliefs": beliefs, "fwd": fwd, "pb": pb, "diary": diary}


def latest_verdicts(rows: list[dict]) -> tuple[dict, dict, dict]:
    lat: dict[str, dict] = defaultdict(dict)
    first: dict[str, str] = {}
    count: Counter = Counter()
    for r in rows:  # chronological: the last row per (id, segment) wins
        lat[r["id"]][r["segment"]] = r["verdict"]
        first.setdefault(r["id"], r["ts"][:10])
        count[(r["id"], r["segment"])] += 1
    return lat, first, count


def status_of(v: dict, h: dict, fwd_ids: set, hid: str) -> str:
    ins, hol, pool = (v.get(s) for s in SEGS)
    vals = [x for x in (ins, hol, pool) if x]
    if not vals:
        return "forward" if hid in fwd_ids else "untested"
    if {ins, hol} == {"supports", "contradicts"}:
        return "flip"
    if "contradicts" in vals:
        return "refuted"
    if hol == "supports" and ins in ("supports", None):
        return "supported"
    if pool == "supports" and ins in ("supports", None) and hol in ("supports", None):
        return "supported"
    return "open"


def outcome_strict(st: str) -> int | None:
    return {"supported": 1, "refuted": 0, "flip": 0}.get(st)


def outcome_lenient(v: dict) -> int | None:
    for s in ("clean_holdout", "pooled", "llm_in_sample"):
        if v.get(s) == "supports":
            return 1
        if v.get(s) == "contradicts":
            return 0
    return None


def code(v: dict) -> str:
    c = "/".join(CODE[v.get(s)] for s in SEGS[:2])
    return c + (f"/{CODE[v['pooled']]}" if v.get("pooled") else "")


def expand(ids: str) -> list[str]:
    return ids.split()


def brier_block(items: list[tuple[float, int]]) -> dict:
    n = len(items)
    if not n:
        return {"n": 0}
    p = [a for a, _ in items]
    o = [b for _, b in items]
    rate = sum(o) / n
    bs = sum((a - b) ** 2 for a, b in items) / n
    clim = rate * (1 - rate)
    pos = [a for a, b in items if b == 1]
    neg = [a for a, b in items if b == 0]
    auc = (sum((x > y) + 0.5 * (x == y) for x in pos for y in neg) / (len(pos) * len(neg))) if pos and neg else None
    return {"n": n, "mean_prior": round(sum(p) / n, 3), "hit_rate": round(rate, 3), "brier": round(bs, 3),
            "auc": None if auc is None else round(auc, 3),
            "brier_base_rate": round(clim, 3), "brier_half": 0.25, "skill_vs_half": round(1 - bs / 0.25, 3)}


def calibration(reg: dict, stat: dict, lat: dict, first: dict, topic_of: dict) -> dict:
    strict = [(reg[h]["prior"], outcome_strict(stat[h]), h) for h in reg if outcome_strict(stat[h]) is not None]
    lenient = [(reg[h]["prior"], outcome_lenient(lat.get(h, {})), h) for h in reg if outcome_lenient(lat.get(h, {})) is not None]
    out = {"definition": {
        "strict": "1 = supported (holdout supports and in-sample supports or untested; or pooled supports with no contradiction); "
                  "0 = any segment contradicts (segment flips count as 0, as the protocol does); open / untested excluded",
        "lenient": "the most out-of-sample decisive verdict: clean_holdout, else pooled, else llm_in_sample (supports = 1, contradicts = 0)"},
        "strict": brier_block([(a, b) for a, b, _ in strict]), "lenient": brier_block([(a, b) for a, b, _ in lenient])}
    bins = [(0.0, 0.3), (0.3, 0.45), (0.45, 0.6), (0.6, 1.01)]
    out["bins"] = []
    for lo, hi in bins:
        sel = [(a, b) for a, b, _ in strict if lo <= a < hi]
        out["bins"].append({"prior_range": f"[{lo:.2f}, {min(hi, 1):.2f})", **brier_block(sel)})
    fam: dict[str, list] = defaultdict(list)
    top: dict[str, list] = defaultdict(list)
    wave: dict[str, list] = defaultdict(list)
    for a, b, h in strict:
        fam[reg[h]["family"]].append((a, b))
        top[topic_of.get(h, "other")].append((a, b))
        d = first.get(h, "")
        w = "1: 11-07/08" if d <= "2026-11-08" else "2: 11-09/10" if d <= "2026-11-10" else "3: 11-11" if d <= "2026-11-11" else "4: 11-12"
        wave[w].append((a, b))
    out["by_family"] = {k: brier_block(v) for k, v in sorted(fam.items(), key=lambda kv: -len(kv[1]))}
    out["by_topic"] = {k: brier_block(v) for k, v in sorted(top.items(), key=lambda kv: -len(kv[1]))}
    out["by_wave"] = {k: brier_block(v) for k, v in sorted(wave.items())}
    surprises = sorted([(a, h) for a, b, h in strict if b == 0 and a >= 0.6], reverse=True)
    pleasant = sorted([(a, h) for a, b, h in strict if b == 1 and a <= 0.4])
    out["confident_misses"] = [{"id": h, "prior": a, "title": reg[h]["title"]} for a, h in surprises]
    out["unexpected_hits"] = [{"id": h, "prior": a, "title": reg[h]["title"]} for a, h in pleasant]
    return out


def cell(x: str) -> str:
    """Escape pipes inside a markdown table cell."""
    return str(x).replace("|", r"\|")


def classify_mistake(title: str) -> str:
    t = title.lower()
    if any(k in t for k in ("look-ahead", "full-sample", "known_ts", "clock", "leak", "test facts", "pre-walk-forward")):
        return "point-in-time / look-ahead"
    if any(k in t for k in ("overflow", "nan", "mle", "collapse", "int8", "-inf", "float residue", "stuck", "estimator")):
        return "numerics / silent NaN"
    if any(k in t for k in ("null", "criterion", "biased", "wrong", "measured", "design", "mapping", "weighting", "compared",
                            "baseline", "unreachable", "gross", "proxy", "destroys", "validated model", "dropped weeks", "correction")):
        return "test design / definitions"
    return "pipeline / operations"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default="2026-11-12")
    args = ap.parse_args()
    S = load_sources()
    reg, rules, fwd = S["reg"], S["rules"], S["fwd"]
    lat, first, count = latest_verdicts(S["rows"])
    fwd_tests = {t["hypothesis"]: t for t in fwd["tests"]}
    blocked = {b["hypothesis"]: b["reason"] for b in fwd.get("blocked", [])}
    stat = {h: status_of(lat.get(h, {}), reg[h], set(fwd_tests), h) for h in reg}
    rule_by_id = {r["id"]: r for r in rules}

    # ---- topic ownership and validation
    topic_of: dict[str, str] = {}
    for t in TOPICS:
        for h in expand(t["hyps"]):
            if h not in reg:
                raise SystemExit(f"topic {t['key']}: {h} is not registered")
            if h in topic_of:
                raise SystemExit(f"{h} assigned to two topics ({topic_of[h]}, {t['key']})")
            topic_of[h] = t["key"]
    unassigned = [h for h in reg if h not in topic_of]
    errors = []
    texts = []
    for t in [*TOPICS, DATA_INFRA]:
        texts += t["know"] + [t.get("confidence", "")] + [a + " " + b for a, b in t["not_believe"]]
    texts += [x[2] + x[3] for x in BELIEF_CONFLICTS] + [x[1] + x[2] for x in RULE_FINDINGS]
    for txt in texts:
        for h in HID.findall(txt):
            if h not in reg:
                errors.append(f"unknown hypothesis {h} in: {txt[:80]}")
        for r in RID.findall(txt):
            if r not in rule_by_id:
                errors.append(f"unknown rule {r} in: {txt[:80]}")
    for a, b, _, _ in BELIEF_CONFLICTS:
        for bid in (a, b):
            if not re.search(rf"\|\s*{re.escape(bid)}\s*\|", S["beliefs"]):
                errors.append(f"belief {bid} not found in BELIEFS.md")
    for rid, *_ in RULE_FINDINGS:
        if rid not in rule_by_id:
            errors.append(f"rule {rid} not found")
    if errors:
        raise SystemExit("validation failed:\n" + "\n".join(errors))

    # ---- rules per topic (a rule belongs to every topic whose hypotheses it cites)
    rules_of: dict[str, list[dict]] = defaultdict(list)
    for r in rules:
        ev = [e for e in r.get("evidence", []) if e in reg]
        for k in sorted({topic_of.get(e, "other") for e in ev}):
            rules_of[k].append(r)

    calib = calibration(reg, stat, lat, first, topic_of)
    try:
        from ai_trader.research.playbook import rules_hash
        live_hash = rules_hash()
    except Exception:  # noqa: BLE001 - the book still builds without the package
        live_hash = None
    pb_hash = S["pb"].get("rules_hash")
    pb_stale = bool(live_hash and pb_hash and live_hash != pb_hash)

    # ---- consistency (automatic parts)
    dups = []
    by_key: dict[tuple, list] = defaultdict(list)
    for r in S["rows"]:
        by_key[(r["id"], r["segment"])].append(r)
    for (h, s), rs in by_key.items():
        if len(rs) > 1:
            verdicts = [x["verdict"] for x in rs]
            kind = ("correction (verdict changed)" if len(set(verdicts)) > 1 else
                    "correction (same verdict)" if any("CORRECT" in (x.get("note") or "") for x in rs) else "identical re-run")
            dups.append({"id": h, "segment": s, "n": len(rs), "verdicts": verdicts, "kind": kind,
                         "ts": [x["ts"][:16] for x in rs]})
    auto_rule_flags = []
    for r in rules:
        if r.get("status") == "retired":
            continue
        ev = [e for e in r.get("evidence", []) if e in reg]
        if not ev:
            continue
        sts = {e: stat[e] for e in ev}
        flags = []
        if any(s == "flip" for s in sts.values()):
            flags.append("segment flip in " + ", ".join(e for e, s in sts.items() if s == "flip"))
        if all(s in ("open", "untested", "forward") for s in sts.values()):
            flags.append("no decisive evidence")
        changed = [e for e in ev if any(len({x["verdict"] for x in by_key[(e, s)]}) > 1 for s in SEGS if (e, s) in by_key)]
        if changed:
            flags.append("evidence verdict changed after first record: " + ", ".join(changed))
        if flags and r["confidence"] in ("A", "B"):
            auto_rule_flags.append({"rule": r["id"], "grade": r["confidence"], "status": r["status"], "flags": flags,
                                    "evidence": {e: code(lat.get(e, {})) for e in ev}})
    stale_status = Counter(reg[h]["status"] for h in reg if stat[h] in ("supported", "refuted", "flip", "open"))
    tracked_dg = [h for h in reg if (reg[h].get("data_generated") or reg[h].get("test_from")) and stat[h] == "untested"]
    old_sb = (ROOT / "research" / "reports" / "SCOREBOARD.md")
    old_sb_rows = len([x for x in old_sb.read_text(encoding="utf-8").splitlines() if x.startswith("| H-")]) if old_sb.exists() else 0
    rules_on_untracked = sorted({r["id"] for r in rules if r.get("status") != "retired" and any(
        e in tracked_dg for e in [*r.get("evidence", []), *HID.findall(str(r.get("review", ""))), *HID.findall(str(r.get("effect", "")))])})

    mistakes = [d for d in S["diary"] if d["kind"] == "mistake"]
    mclass = Counter(classify_mistake(m["title"]) for m in mistakes)

    # ---- knowledge.json
    topics_out = []
    for t in TOPICS:
        hs = expand(t["hyps"])
        cnt = Counter(stat[h] for h in hs)
        fw = [{"id": h, "n": fwd_tests[h].get("n"), "need": fwd_tests[h].get("need"), "unit": fwd_tests[h].get("unit"),
               "interim": fwd_tests[h].get("est"), "interim_unit": fwd_tests[h].get("unit_est", ""), "status": fwd_tests[h].get("status")} for h in hs if h in fwd_tests]
        untested = [{"id": h, "title": reg[h]["title"], "blocked": blocked.get(h),
                     "forward_registered": bool(reg[h].get("data_generated") or reg[h].get("test_from"))}
                    for h in hs if stat[h] == "untested"]
        refuted_hi = [{"id": h, "prior": reg[h]["prior"], "title": reg[h]["title"], "verdicts": code(lat.get(h, {}))}
                      for h in hs if stat[h] in ("refuted", "flip") and reg[h]["prior"] >= 0.5]
        rl = rules_of.get(t["key"], [])
        topics_out.append({
            "key": t["key"], "title": t["title"], "know": t["know"], "confidence": t["confidence"],
            "status_counts": dict(cnt), "hypotheses": {h: {"verdicts": code(lat.get(h, {})), "status": stat[h], "prior": reg[h]["prior"]} for h in hs},
            "rules": [{"id": r["id"], "grade": r["confidence"], "status": r["status"]} for r in rl],
            "forward": fw, "untested": untested, "refuted_high_prior": refuted_hi,
            "not_believe": [{"claim": a, "evidence": b} for a, b in t["not_believe"]],
        })
    contexts = S["pb"]["contexts"]
    knowledge = {
        "version": "1.0", "built": args.date, "rules_hash": S["pb"].get("rules_hash"),
        "sources": {"hypotheses": len(reg), "result_rows": len(S["rows"]), "rules": len(rules),
                    "forward_tests": len(fwd_tests), "diary_entries": len(S["diary"])},
        "topics": topics_out,
        "data_infra": {"know": DATA_INFRA["know"], "not_believe": [{"claim": a, "evidence": b} for a, b in DATA_INFRA["not_believe"]],
                       "mistakes_by_class": dict(mclass)},
        "call_order": {k: {"title": c["title"], "call": [x[0] for x in c.get("call", [])], "skip": [x[0] for x in c.get("skip", [])],
                           "rules": c.get("rules", [])} for k, c in contexts.items()},
        "calibration": calib,
        "consistency": {"belief_conflicts": [{"a": a, "b": b, "issue": i, "recommendation": rec} for a, b, i, rec in BELIEF_CONFLICTS],
                        "rule_findings": [{"rule": r, "issue": i, "recommendation": rec} for r, i, rec in RULE_FINDINGS],
                        "auto_rule_flags": auto_rule_flags, "duplicate_results": dups,
                        "untracked_forward_hypotheses": tracked_dg, "rules_on_untracked": rules_on_untracked,
                        "stale_registry_status": dict(stale_status), "old_scoreboard_rows": old_sb_rows},
        "unassigned_hypotheses": unassigned,
    }
    out_json = ROOT / "data" / "research" / "knowledge.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(knowledge, indent=1, ensure_ascii=False), encoding="utf-8")

    # ---- KNOWLEDGE.md
    tot = Counter(stat.values())
    L = ["# Knowledge book", "",
         f"_Built {args.date} by `scripts/knowledge_book.py` from {len(reg)} registered hypotheses ({len(S['rows'])} result rows), "
         f"{len(rules)} rules (hash {S['pb'].get('rules_hash')}), {len(fwd_tests)} forward tests and {len(S['diary'])} diary entries. "
         "Do not edit by hand; rebuild. Machine-readable twin: `data/research/knowledge.json`. Per-hypothesis verdicts, prior "
         "calibration and the consistency check: [reports/SCOREBOARD_" + args.date + ".md](reports/SCOREBOARD_" + args.date + ".md). "
         "Grades follow [BELIEFS.md](BELIEFS.md): A = both segments and a large effect, B = small effect or confirmed once, C = plausible, being tested._", "",
         "**Verdict codes** are in-sample / holdout[/pooled]: S supports, X contradicts, ? inconclusive, - not recorded. **Status**: "
         "supported (holds out of sample, no contradiction), refuted (contradicted), flip (one segment supports, the other contradicts; "
         "not supported under the protocol), open (results, nothing decisive), forward (collecting), untested.", "",
         f"Registry today: {tot['supported']} supported, {tot['refuted']} refuted, {tot['flip']} segment flips, {tot['open']} open, "
         f"{tot['forward']} collecting forward, {tot['untested']} untested.", "",
         "## The ten things to remember", "",
         "1. Vol is forecastable; direction is not. Every working tool forecasts variance; every direction test is at coin-flip level (H-IC1, H-NN2, H-MINE2).",
         "2. A daily vol forecast is the vol level plus the calendar: log-HAR level, then the forecast weekday (Saturday about 0.5-0.6x, Monday about 1.3x for crypto), then nothing else of note (H-NN3, H-SP1, H-DS1).",
         "3. Per bar, use robust_bc; after a jump, excess variance halves in 2-3 h; after spikes, forecasts run hot (H-S4, H-V1b, H-RL2).",
         "4. Expectations and risk differ: weekends and Saturdays are quiet in expectation but jump-prone, so stops use the unshrunk sigma (H-W1, H-V2).",
         "5. Intraday vol has a large hour-of-week shape (peaks Sunday 22-02 UTC and at the US open); recent |returns| refine it (H-NS4, H-NN8).",
         "6. No leverage effect in crypto; forecast from total RV, not the sign of the move (H-LEV2).",
         "7. Levels, indicators, sessions and positioning lore do not predict direction; levels break slightly more often than they hold (H-VZ1, H-AU1, H-MC1, H-TF1).",
         "8. The deterministic book is about one crypto-trend bet with a thin, currently negative edge; its P&L leaks through exits, not fees, and tight trails do not fix that (H-PD1, H-PD4, H-PD6, H-TR1).",
         "9. Regimes say where strategies lose and how long a state lasts, not who wins next (H-RC1, H-RL3).",
         "10. Microstructure and order flow are first measurements (C): bursty arrivals, long sign memory, activity scaling; traits vary by day; flow is an activity gauge (H-MS1, H-MS3, H-MD1, H-OF1).", ""]

    def short(title: str, n: int = 48) -> str:
        title = re.sub(r"\s*\((forward|forward confirmation|data-generated)\)\s*$", "", title)
        return title if len(title) <= n else title[:n].rsplit(" ", 1)[0] + "..."

    def fmt_rules(rl: list[dict]) -> str:
        parts = []
        for g in "ABC":
            ids = [r["id"] + ("*" if r["status"] == "provisional" else "") for r in rl if r["confidence"] == g and r["status"] != "retired"]
            if ids:
                parts.append(f"{g}: " + ", ".join(ids))
        ret = [r["id"] for r in rl if r["status"] == "retired"]
        if ret:
            parts.append("retired: " + ", ".join(ret))
        return "; ".join(parts) if parts else "none"

    for i, t in enumerate(topics_out, 1):
        L += [f"## {i}. {t['title']}", "", "**What we know**", ""]
        L += [f"- {s}" for s in t["know"]]
        c = t["status_counts"]
        cnt_txt = ", ".join(f"{c[k]} {k}" for k in ("supported", "refuted", "flip", "open", "forward", "untested") if c.get(k))
        L += ["", f"**Confidence.** {t['confidence']} Registry: {cnt_txt}.", "",
              f"**Rules** (\\* = provisional): {fmt_rules(rules_of.get(t['key'], []))}.", ""]
        L += ["**Still open**", ""]
        if t["forward"]:
            L.append("- Forward tests: " + "; ".join(
                f"{f['id']} {f['n']}/{f['need']} {f['unit']}" + (f" (interim {f['interim']:.3g}{' ' + f['interim_unit'] if f['interim_unit'] else ''})" if isinstance(f["interim"], (int, float)) else "")
                for f in t["forward"]) + ".")
        ut = t["untested"]
        if ut:
            plain = [u["id"] for u in ut if not u["blocked"] and not (u["forward_registered"] and u["id"] not in fwd_tests)]
            orphan = [u["id"] for u in ut if u["forward_registered"] and u["id"] not in fwd_tests and not u["blocked"]]
            blk = [f"{u['id']} ({u['blocked'].split(':')[0].split(';')[0]})" for u in ut if u["blocked"]]
            parts = ([", ".join(plain)] if plain else []) + ([", ".join(orphan) + " (forward-registered, but no runner collects them)"] if orphan else [])                 + (["; ".join(blk)] if blk else [])
            L.append("- Registered, no result: " + "; ".join(parts) + " (titles in the scoreboard).")
        op = [h for h, x in t["hypotheses"].items() if x["status"] == "flip"]
        if op:
            L.append("- Segment flips to watch: " + ", ".join(f"{h} {t['hypotheses'][h]['verdicts']}" for h in op) + ".")
        if not (t["forward"] or ut or op):
            L.append("- nothing registered.")
        L += ["", "**Do not believe**", ""]
        L += [f"- *{x['claim']}*: {x['evidence']}" for x in t["not_believe"]]
        if t["refuted_high_prior"]:
            L.append("- Our own priors >= 0.5 that failed: " + ", ".join(
                f"{x['id']} ({x['prior']:.2f}, {x['verdicts']})" for x in t["refuted_high_prior"]) + ".")
        L.append("")
    n = len(topics_out) + 1
    L += [f"## {n}. {DATA_INFRA['title']}", "", "**What we know**", ""]
    L += [f"- {s}" for s in DATA_INFRA["know"]]
    L += ["", f"**Diary mistakes by class** ({len(mistakes)}): " + ", ".join(f"{k} {v}" for k, v in mclass.most_common()) + ".", "",
          "**Do not believe**", ""]
    L += [f"- *{a}*: {b}" for a, b in DATA_INFRA["not_believe"]]
    L += ["", f"## {n + 1}. How the decision layer should use this", "",
          "- **Load the card, not the book.** At decision time call `ai_trader.research.playbook.card_for(context, state)` "
          "([PLAYBOOK.md](PLAYBOOK.md)): at most 8 tested rules for the context, already filtered by the market state (R-BUDGET-1). "
          "This book explains why the cards say what they say; it is not a list of extra inputs.",
          "- **Order of questions.** Vol first (level, then calendar, then intraday shape), risk second (sigma_risk_next, resting stops), "
          "direction last, and only through the registered deterministic candidates. Never add inputs to make a directional call.",
          "- **Expectations vs risk.** Weekend, Saturday, cell and ML adjustments change expectations only; stops and risk always use "
          "the unshrunk sigma_risk_next (R-VOL-2). Position size comes from the risk engine, never from a rule.",
          "- **Grades.** Apply A and B rules as stated, treat C rules as context, and never trade a segment flip or an interim forward estimate.",
          "- **Shocks.** shock_regime executes at the bar close without waiting for the LLM, which may only veto (R-BUDGET-4); "
          "never fade a 2-sigma hourly move on discretion (R-PX-2).", "",
          "Call order per context (from `data/research/playbook.json`, rules hash " + str(pb_hash) + "):" +
          (f" **the playbook is stale**: rules.yaml now hashes to {live_hash} ({len(rules)} rules); rerun `scripts/playbook.py`." if pb_stale else ""), "",
          "| context | call, in order | skip |", "|---|---|---|"]
    for k, c in knowledge["call_order"].items():
        call = [re.sub(r"\s*\(.*\)$", "", x) for x in c["call"]]
        skip = [re.sub(r"\s*\(.*\)$", "", x) for x in c["skip"]]
        L.append(f"| `{k}` | {cell(' -> '.join(call)) or '(no tool; rules only)'} | {cell('; '.join(skip)) or '-'} |")
    L += ["", "Audit note: for `next_day_vol` the intended pipeline is log-HAR level (R-VOL-20) -> forecast-weekday factor (R-ML-1) -> "
          "(C, only once validated forward) LAT1 cells (R-DS-5) or the nonlinear tool (R-ML-4); R-VOL-4 is superseded by R-VOL-20.", ""]
    (ROOT / "research" / "KNOWLEDGE.md").write_text("\n".join(L), encoding="utf-8")

    # ---- SCOREBOARD_<date>.md
    B = [f"# Hypothesis scoreboard ({args.date})", "",
         f"_Generated by `scripts/knowledge_book.py` from research/hypotheses.yaml ({len(reg)} hypotheses) and "
         f"hypotheses_results.jsonl ({len(S['rows'])} rows); latest row per hypothesis and segment wins. Supersedes "
         f"[SCOREBOARD.md](SCOREBOARD.md), which covers {old_sb_rows} hypotheses and ignores the pooled segment. Status rules are "
         "stated in [KNOWLEDGE.md](../KNOWLEDGE.md). Forward n is from data/research/forward_tests/status.json._", "",
         "**Totals:** " + ", ".join(f"{k} {v}" for k, v in tot.most_common()), "",
         "## Prior calibration", ""]
    cs, cl = calib["strict"], calib["lenient"]
    B += [f"- Strict outcome ({cs['n']} decided hypotheses): mean prior {cs['mean_prior']}, hit rate {cs['hit_rate']}; Brier of priors "
          f"{cs['brier']} vs {cs['brier_base_rate']} for always forecasting the realised base rate and 0.25 for always saying 0.5 "
          f"(skill vs 0.5: {cs['skill_vs_half']}); ranking skill of priors (AUC) {cs['auc']}.",
          f"- Lenient outcome ({cl['n']}): mean prior {cl['mean_prior']}, hit rate {cl['hit_rate']}, Brier {cl['brier']} "
          f"(base rate {cl['brier_base_rate']}); AUC {cl['auc']}.",
          f"- Definitions: strict = {calib['definition']['strict']}; lenient = {calib['definition']['lenient']}.", "",
          "| prior range | n | mean prior | hit rate | Brier |", "|---|---|---|---|---|"]
    hits = [b["hit_rate"] for b in calib["bins"] if b["n"]]
    mono = all(x <= y for x, y in zip(hits, hits[1:]))
    B.insert(len(B) - 2, f"**Reading.** On the protocol outcome our priors were about {cs['mean_prior'] / max(cs['hit_rate'], 1e-9):.1f}x too optimistic "
             f"(mean prior {cs['mean_prior']} vs realised {cs['hit_rate']}) and lose to a constant base-rate forecast on Brier, but they ranked "
             f"hypotheses usefully (AUC {cs['auc']}; the hit rate {'rises monotonically' if mono else 'does not rise monotonically'} across prior bins). "
             f"On the holdout-first outcome they look calibrated on average ({cl['mean_prior']} vs {cl['hit_rate']}) with almost no ranking skill "
             f"(AUC {cl['auc']}): holdout-only verdicts, including segment flips, are close to coin flips. By topic, direction and lore claims "
             "realised about 0.07 against priors near 0.38, volatility-structure and ML-vol claims about 0.3-0.45 against 0.45-0.6: set priors "
             "near 0.1 for direction / lore and near 0.35-0.4 for vol-structure claims, and expect composite criteria (microstructure) to fail more often.")
    B.insert(len(B) - 2, "")
    for b in calib["bins"]:
        if b["n"]:
            B.append(f"| {b['prior_range']} | {b['n']} | {b['mean_prior']} | {b['hit_rate']} | {b['brier']} |")
    B += ["", "By registration wave (date of first recorded result; strict outcome):", "", "| wave | n | mean prior | hit rate | Brier |", "|---|---|---|---|---|"]
    for k, b in calib["by_wave"].items():
        B.append(f"| {k} | {b['n']} | {b['mean_prior']} | {b['hit_rate']} | {b['brier']} |")
    B += ["", "By topic (strict):", "", "| topic | n | mean prior | hit rate | Brier |", "|---|---|---|---|---|"]
    for k, b in calib["by_topic"].items():
        B.append(f"| {k} | {b['n']} | {b['mean_prior']} | {b['hit_rate']} | {b['brier']} |")
    B += ["", "By registry family (strict; families with at least 3 decided):", "", "| family | n | mean prior | hit rate | Brier |", "|---|---|---|---|---|"]
    small = []
    for k, b in calib["by_family"].items():
        if b["n"] >= 3:
            B.append(f"| {k} | {b['n']} | {b['mean_prior']} | {b['hit_rate']} | {b['brier']} |")
        else:
            small.append(f"{k} {b['n']}")
    B += ["", "Families with fewer than 3 decided: " + ", ".join(small) + ".", "",
          "Confident misses (prior >= 0.6, outcome 0): " + "; ".join(f"{x['id']} ({x['prior']:.2f}) {short(x['title'], 60)}" for x in calib["confident_misses"]) + ".", "",
          "Unexpected hits (prior <= 0.4, outcome 1): " + "; ".join(f"{x['id']} ({x['prior']:.2f}) {short(x['title'], 60)}" for x in calib["unexpected_hits"]) + ".", "",
          "## All registered hypotheses", "",
          "| id | topic | family | prior | in-sample | holdout | pooled | status | rows | first result | title |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    ICON = {"supports": "S", "contradicts": "X", "inconclusive": "?", None: "-"}
    for h, d in reg.items():
        v = lat.get(h, {})
        st = stat[h]
        if st == "forward":
            f = fwd_tests[h]
            st = f"forward {f.get('n')}/{f.get('need')}"
        elif h in blocked:
            st = "blocked"
        rows_n = sum(count[(h, s)] for s in SEGS)
        dg = " (data-generated)" if d.get("data_generated") else ""
        B.append(f"| {h} | {topic_of.get(h, 'other')} | {d['family']} | {d['prior']:.2f} | {ICON[v.get('llm_in_sample')]} | "
                 f"{ICON[v.get('clean_holdout')]} | {ICON[v.get('pooled')]} | {st} | {rows_n} | {first.get(h, '')} | {cell(d['title'])}{dg} |")

    B += ["", "## Consistency check (recommendations only; nothing was edited)", "",
          "### 1. BELIEFS.md lines that disagree with later lines", "", "| lines | issue | recommendation |", "|---|---|---|"]
    for a, b, i, rec in BELIEF_CONFLICTS:
        B.append(f"| {a} vs {b} | {cell(i)} | {cell(rec)} |")
    B += ["", "### 2. Rules whose evidence was weakened or contradicted later", "", "Read by hand:", "",
          "| rule | grade | issue | recommendation |", "|---|---|---|---|"]
    for rid, i, rec in RULE_FINDINGS:
        r = rule_by_id[rid]
        B.append(f"| {rid} | {r['confidence']} ({r['status']}) | {cell(i)} | {cell(rec)} |")
    B += ["", "Automatic flags (A/B rules only; heuristic, an 'avoid' rule built on a refuted idea is not a problem):", "",
          "| rule | grade | flags | evidence verdicts |", "|---|---|---|---|"]
    for f in auto_rule_flags:
        B.append(f"| {f['rule']} | {f['grade']} | {'; '.join(f['flags'])} | {', '.join(f'{k} {v}' for k, v in f['evidence'].items())} |")
    B += ["", "### 3. Hypotheses recorded more than once per segment", "",
          "| id | segment | rows | verdicts | kind | timestamps |", "|---|---|---|---|---|---|"]
    for d in dups:
        B.append(f"| {d['id']} | {d['segment']} | {d['n']} | {' -> '.join(d['verdicts'])} | {d['kind']} | {', '.join(d['ts'])} |")
    B += ["", "Recommendation: the 'latest row wins' rule is correct for the two corrections that changed a verdict (H-GT3, H-AU1). "
          "The triple rows for H-SEC1-3 and H-PAN1-2 come from three runs of scripts/sector_study.py within six minutes; make the study "
          "refuse to record an (id, segment) that already has a row unless `--supersede` is passed, and add a `supersedes` field to corrections.", "",
          "### 4. Bookkeeping", "",
          f"- `status` in hypotheses.yaml is never updated: of the hypotheses with results, the registry says {dict(stale_status)}. Derive status from results (as this script does) or update it on record.",
          f"- The old SCOREBOARD.md covers {old_sb_rows} of {len(reg)} hypotheses and ignores the pooled segment (e.g. H-MINE2 is only pooled), so it is stale.",
          "- Forward-registered hypotheses that no runner collects (data_generated or test_from, no result, not in status.json): "
          + ", ".join(tracked_dg) + ". Rules that cite them as evidence or as their review trigger: " + (", ".join(rules_on_untracked) or "none") +
          ". H-S3 is the forward test of the first sleeve (shock_regime) and is not being measured.",
          "- Registered forward hypotheses without a test_from and not collected: " + ", ".join(
              h for h in reg if stat[h] == "untested" and h not in tracked_dg and h not in blocked and "forward" in reg[h]["title"].lower()) + ".",
          *([f"- data/research/playbook.json (rules hash {pb_hash}) predates the current rules.yaml (hash {live_hash}, {len(rules)} rules, e.g. R-EXIT-1..3): rerun scripts/playbook.py."] if pb_stale else []),
          "- The weekend discount is stated five times in rules (R-VOL-1, R-ML-2, R-DS-1, R-SP-3, R-VOL-2 context) with different baselines (audit item 5): merge into one family with R-VOL-1 as the headline.",
          "- 'SUPPORTED' for H-C1 hides the decay (holdout net t 1.3 over 17 weeks; 2026 H1 negative): add a decay / effect-size column to verdicts that rest on a premium.", ""]
    sb = ROOT / "research" / "reports" / f"SCOREBOARD_{args.date}.md"
    sb.write_text("\n".join(B), encoding="utf-8")

    words = len(" ".join(L).split())
    print(f"KNOWLEDGE.md {words} words; {sb.name}; knowledge.json")
    print("status totals", dict(tot))
    print("calibration strict", calib["strict"], "lenient", calib["lenient"])
    if unassigned:
        print("unassigned hypotheses:", unassigned)


if __name__ == "__main__":
    main()
