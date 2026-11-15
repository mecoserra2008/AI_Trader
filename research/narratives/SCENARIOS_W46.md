# Week of 2026-11-09: story, positioning, geopolitics, scenarios

Written Sun 2026-11-08 ~20:45 UTC, before the CME reopen. Sources: CFTC COT as of Tue Nov 3 (`scripts/cftc_positioning.py`, snapshot `data/cftc/2026-11-03.parquet`), Polymarket (`scripts/polymarket_geo.py`, snapshot `data/polymarket/20261108T2025.json`), Bybit lake (funding, OI, long/short accounts, liquidations), FRED, and the boards in `STATE.md` / `events.yaml`. Scenario weights are **judgment** anchored on market-implied numbers, not model output.

## Story of the week

1. **Election (Nov 3-4):** Democrats took both chambers. Crypto rallied (BTC +3.9%, ETH +6.6%), oil fell 9%.
2. **AI financing scare (Nov 5):** neocloud credit (CoreWeave bonds ~16%) met a crowded positioning: leveraged funds were net long Nasdaq-100 minis at the 97th percentile of 3 years on Nov 3, and had added VIX shorts over 4 weeks. Nasdaq −4% into correction, NVDA −10%, VIX 16.6 -> 28.2, BTC −292 bp in the 15:00 UTC hour.
3. **Strong jobs (Nov 6):** +165k payrolls, claims at a 57-year low; December hike priced at 76.5% on Polymarket (70.5% a week earlier; no cut in 2026: 98%). Equities bounced; crypto didn't (spot BTC ETF outflows ~430M USD).
4. **Weekend:** three tankers exploded in Hormuz, a US strike on a bulk carrier, Houthis targeting Riyadh, a US blockade of Iran still in place. Yet prediction markets moved *toward* containment: Israel-Iran ceasefire holding through Nov 15 80.5% -> 90.5% in a day.

**One line:** a rates-and-AI deleveraging shock inside a chronic Gulf-chokepoint regime, with crypto bleeding from weak institutional demand and crowded retail longs.

## Positioning

| Market | Who | Net (% of OI) | 3y pct | Read |
|---|---|---|---|---|
| BTC CME | leveraged funds | −14.8% | **99th** (least short in 3y) | basis trade unwound (CME OI −18% in 4 weeks), the mechanical source of ETF outflows |
| BTC CME | asset managers | +13.7% | **7th** | institutions near 3-year-low longs: no real-money bid |
| ETH CME | leveraged funds | −16.2% | 86th | same unwind (OI −17%) |
| Nasdaq-100 mini | leveraged funds | +3.4% | **97th** | crowded long into the AI selloff (pre-crash report): forced deleveraging fuel |
| S&P e-mini | leveraged / asset mgrs | −18.7% / +46.7% | 30th / 63rd | unremarkable |
| VIX futures | leveraged funds | −8.5% | 38th | added ~37k short in 4 weeks: short-vol caught by the VIX spike |
| UST 10Y | asset managers | +51.0% | **99th** | record duration long: vulnerable to a hot CPI / weak auctions |
| UST 2Y | leveraged funds | −32.5% | 91st | |
| Japanese yen | asset managers | +20.9% | 77th | long yen with USD/JPY near 160: intervention / BOJ bets |
| WTI (NYMEX) | managed money | +1.1% | **9th** | speculators very light on oil despite the war: asymmetric upside if escalation |
| Brent / gold / silver | managed money | +2.2% / +26% / +3.5% | 28th / 34th / 8th | light |

Crypto perps (Bybit, now): 71-82% of accounts long on the majors while prices are down 11-33% in 30 days; BTC funding +9.4%/yr (7d +8.5%); BTC OI +24% in 30 days while price −11.5% (leveraged longs averaging down); since the liquidation feed started (Sun 00:43 UTC) longs liquidated 2.5M USD vs shorts 1.2M USD on a quiet day. Read: retail long, institutions out, leverage building against the trend: downside fragility.

## Geopolitics on Polymarket (implied probability; change vs 7 days ago)

| Market | Now | 7d ago | 30d ago |
|---|---|---|---|
| Israel-Iran ceasefire holds through Nov 15 | 90.5% | 68.5% | n/a |
| ... through Nov 30 | 73.5% | 55.0% | 63.0% |
| ... through Dec 31 | 59.0% | 37.5% | 57.5% |
| US announces end of Iranian blockade by Nov 15 / Nov 30 / Dec 31 | 4.4% / 13.5% / 36.3% | 12.0% / 26.5% / 41.5% | n/a / 37.5% / 48.0% |
| Hormuz traffic normal by Nov 30 / Dec 31 | 2.6% / 7.5% | 5.5% / 11.5% | 9.5% / 18.5% |
| Military action against the Saudi East-West pipeline by Nov 30 | **36.5%** | 35.5% (21.0% 1d ago) | 67.0% |
| Bab el-Mandeb effectively closed by Dec 31 | 9.0% | 11.5% | 15.5% |
| Iran full airspace closure by Dec 31 | 24.5% | 36.0% | 23.5% |
| US invades Iran before 2027 | 9.5% | 17.5% | 15.5% |
| Crude oil new all-time high by Dec 31 | 7.5% | 16.5% | 10.5% |
| US reissues Iran oil sanction relief by Dec 31 | 31.5% | 55.0% | n/a |
| Fed +25 bp on Dec 9 / no change | 76.5% / 22.5% | 70.5% / 28.5% | 74.5% / 22.5% |
| US recession by end of 2026 | 3.8% | 5.5% | 6.5% |

**Read: chronic chokepoint, receding escalation.** The big-war tail is shrinking (invasion, airspace closure, oil ATH all down sharply) while disruption is expected to persist (blockade and Hormuz normalisation odds falling). The live tail is infrastructure strikes (Saudi pipeline 36.5% this month).

**Calibration check.** Polymarket's BTC and WTI price-level markets match driftless volatility touch probabilities (BTC < 70k by year end 53.5% vs 55-59% at 36-40 vol; WTI <= 85 in November 61.5% vs 60-70% at 45-60 vol). They carry no directional view: last week's swing in oil odds (WTI <= 85: 19% -> 61.5%) came from the price falling to ~90, not from a forecast. What they do carry is tail premia: BTC <= 60k 17.5% vs 8-12% from vol (crash premium), WTI >= 120 7% vs 1-5% (war premium).

## Thesis: "Chokepoint without escalation"

Base case: supply friction persists (blockade on, Hormuz not normal) but escalation stays contained (ceasefire holds). Oil oscillates with headline spikes that fade, as it did all last week; the macro driver of the week is rates (CPI, auctions) and AI credit, not the war. Escalation is a fat tail with outsized impact because speculative oil positioning is light (9th percentile) and duration longs are crowded (99th): an oil spike would hit headline inflation, lock the hike and squeeze duration and long-duration assets at once.

Signals that the thesis is wrong: ceasefire-through-Nov-30 below 60%; Saudi-pipeline strike odds above 50% or an actual strike; Bab el-Mandeb odds above 20%; blockade-end-by-Nov-30 above 30% (de-escalation version). Registered as H-PM1 (forward).

## Scenarios for Nov 9-13 (weights are judgment)

| | Weight | Triggers | Path | Crypto | Cross-asset |
|---|---|---|---|---|---|
| **A. Chop under a priced hike** | ~50% | status-quo war; CPI in line (core 0.2-0.3); AI earnings mixed | soft Monday open, then range trading around the data | BTC ~72-80k, vol above its 30d level, alts keep lagging; crowded longs bled by small liquidations | oil 85-100 drifting with headlines; VIX mid-20s |
| **B. Hawkish / duration squeeze** | ~20% | core CPI >= 0.4; weak 10y (Tue) / 30y (Thu) auctions | record duration longs and NQ longs unwind | BTC breaks 72k toward 70k (Polymarket < 70k by YE already 53.5%); long-liquidation cascade | yields up, Nasdaq retests its lows, USD/JPY toward 160+ (intervention risk) |
| **C. Relief** | ~20% | soft core (<= 0.1-0.2) and a credible CoreWeave financing story | hike odds below ~60%, short-vol rebuilds | BTC 79-83k led by ETH / high beta; alts lag even in the bounce | semis and neoclouds squeeze; long end rallies further |
| **D. Gulf escalation** | ~10% | ceasefire break (Polymarket: ~10% by Nov 15), Saudi pipeline strike, Bab el-Mandeb | oil +10-20% from light positioning; breakevens up; hike locked | BTC down with equities, then the shock playbook applies | energy and gold up; duration down |

**On a weak Monday:** for it in crypto: crowded retail longs with positive funding, long liquidations already dominating, no institutional bid (CME asset-manager longs at the 7th percentile, ETF outflows), weekend escalation headlines. Against: ceasefire odds rose sharply today, oil specs are light (a reopen gap could go either way), and equities closed Friday on a bounce. Weakness is more likely concentrated in crypto and alts than broad, unless equities gap down at the reopen. There is no validated day-of-week edge in our research, so this is context, not a signal.

**Validated rule in force:** on a >= 4-sigma hourly BTC move, fade it if the vol state was high, follow it if low (H-S2/S3). Models: P(trend week) ~ base rate 30%; P(vol expansion) ~42% (H-TR5/H-TR6).

## Update 21:20 UTC: day-of-week and regime-lifecycle evidence (from BELIEFS V10 / P8 / P9 / L1, added by a parallel session)

- **No Monday drift** in any of 9 assets, crypto or TradFi (P8, confidence A, Holm-adjusted). A "weak Monday" has no statistical support as a directional base rate.
- **After a < −1σ week, TradFi Mondays have leaned *up*** (+0.12σ pooled, SPX +0.24σ, TLT +0.14σ); crypto shows nothing (P9, confidence C: in-sample only, ~6 events a year, contaminated by the literature). This week qualifies (QQQ −5%, NVDA −9%): mild evidence *against* a weak Monday for equities, none either way for crypto.
- **Monday variance:** BTC Mondays ~1.3x Tue-Fri in-sample but not in the holdout; S&P Mondays carry 27% of their variance in the weekend gap; **WTI Mondays 2.3x in the holdout (war weekends)** (V10). So the Sunday 23:00 UTC oil reopen is where to widen risk, not equities or crypto.
- **Regime lifecycle (L1):** daily R7 grind-down episodes (BTC and most alts now) usually exit to R4 compression (46%) or R8 trend-down (28%), median 5-7 days; after R9 liquidation, 62% go to R6 whipsaw.
- **Net effect on the scenarios:** keep the weights; reword the Monday view to "Monday volatility, not Monday direction". Crypto weakness, if it comes, would come from positioning (crowded longs, no institutional bid), not from a day-of-week effect.
