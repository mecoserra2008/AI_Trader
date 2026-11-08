# AI trade: structure, crypto linkage, and what is bearish (2026-11-08)

Script: `scripts/ai_structure.py` (Yahoo daily closes for equities, Bybit daily for AI tokens; weekly bars). Descriptive; no trading claim.

## 1. Break of structure
Definition: weekly close below the last confirmed weekly swing low (2-bar fractal), after a >= 15% run-up, first break in 13 weeks.

| | Now | Current break | Earlier 2026 breaks |
|---|---|---|---|
| NVDA | 186.64, −20.2% from 52w high, −8.4% vs 40w MA | week of Oct 16 | Feb 13 and Jun 26 (both recovered to new highs in 7-10 weeks) |
| SMH (semis) | 463.96, **−29.7%** from 52w high, −11.5% vs 40w MA | week of Oct 16 | **Jul 17, no new high since**: two breaks in a row = lower highs, a semis bear market |
| Nasdaq-100 | 26,990, −12.6%, −3.6% vs 40w MA | week of Oct 16 | Feb 6 and Jul 17 (recovered) |

History (2015-2026, n = 20-23 breaks each): 13 weeks after a break the median return was NVDA +17% (70% up), SMH +6.8% (68%), NDX +8.7% (75%); median further drawdown 4-8%; a new high came within a median of 5-11 weeks; only ~10% failed to make a new high within 18 months. Caveat: this sample is a secular tech bull market.

The exceptions cluster in Fed-tightening windows (post hoc, n small): 2018-10 (NDX −13% at 13w, −18% further), 2021-12 / 2022-01 / 2022-04 (−8% to −26% at 13w, −27% to −47% further). Late in the cycle (2022-09) breaks marked the bottom. Today a new hike is 77% priced, which is why the 2022 analog matters, though inflation is far milder (4.0% headline, 2.8% core, 10y breakeven 2.23 vs 7-9% CPI in 2022).

BTC after AI-index breaks (2018+, n = 27): 13-week median +2.6% (55% up) vs +8.6% unconditional; in the tightening windows −10% to −48%.

## 2. Is AI using crypto? Weekly betas over 52 weeks (r = a + b_BTC·BTC + b_AI·NVDA⊥BTC)

| Asset | β BTC (t) | β AI (t) | R² (BTC only) | Since Oct 15 |
|---|---|---|---|---|
| CoreWeave (ex-miner) | 0.38 (1.5) | **1.40 (4.1)** | 0.28 (0.03) | −33% |
| IREN (miner → AI cloud) | 0.97 (3.7) | **1.18 (3.3)** | 0.33 (0.18) | −29% |
| Core Scientific | 0.34 (1.7) | **0.70 (2.6)** | 0.17 (0.05) | −19% |
| Hut 8 | 0.41 (1.6) | 0.69 (1.9) | 0.11 (0.04) | −31% |
| MSTR | **1.39 (8.5)** | 0.38 (1.7) | 0.60 (0.58) | −15% |
| COIN | **1.07 (7.3)** | 0.21 (1.1) | 0.53 (0.52) | −27% |
| TAO / RENDER / VIRTUAL / NEAR ("AI tokens") | **1.43-1.67 (5-8)** | −0.5 to 0.0 (n.s.) | 0.33-0.60 | −18% to −30% |
| ETH | 1.26 (21.8) | −0.16 (−2.1) | 0.91 | −8% |

- Miners that became AI landlords now trade on AI (AI beta 0.7-1.4, BTC explains almost nothing for CoreWeave/Core Scientific/Hut 8). This is the real bridge: hashrate fell in Q1 2026 for the first time since 2020 as miners retrofit sites for AI; IREN, Core Scientific and TeraWulf are heading for ~70% of revenue from AI.
- "AI tokens" have no AI-equity beta: they are high-beta crypto (1.4-1.7x BTC).
- AI agents do use crypto rails, mostly stablecoins: x402 processed > 600M USD across ~500k agent wallets by early 2026; ~176M on-chain agent payments in the 12 months to April 2026, mostly USDC. Real but small next to a 294 bn USD stablecoin float, and it doesn't route through BTC.
- BTC~NDX correlation of weekly returns: 0.20 over the last 26 weeks (0.51 in 2025, 0.54 in 2022), but hourly correlation 0.65 in the shock week: BTC couples to AI in stress, not day to day.

## 3. What is bearish in AI, and how alarming
- **Financing, not demand.** Neoclouds: ~30 bn USD revenue vs ~80 bn USD capex (JPMorgan); CoreWeave's 2031 bonds above 16% yield, CDS-implied ~63% 5-year default probability (reported), a 2026-28 refinancing wall; tech bond issuance > 500 bn USD in 2026. Payback doubts on ~900 bn USD of 2026 AI spend triggered Nov 5 (AMD −12%, Micron −10%, SanDisk −14%).
- **Rates join in.** A December hike raises the cost of the most leveraged part of the buildout.
- **Not systemic (yet).** Listed tech generates ~2.2 trn USD of cash flow vs 1.8 trn USD capex (JPMorgan): the hyperscalers self-fund; the stress sits in the leveraged edge (neoclouds, miners-turned-AI, high-multiple semis). HY OAS 3.48% (1y wide, far from the 5-6% of credit stress); VXN 38 (high, below the 2022 and Apr-2025 peaks).
- **Verdict.** Alarming for the leveraged AI edge and the crypto-equities tied to it; not yet a systemic or macro event. Escalation markers: HY OAS > 4%, CoreWeave earnings / bond yields > 20%, VXN > 45, SMH below its 2026 March low, and Nvidia's guidance on **Nov 17** (after the close).
