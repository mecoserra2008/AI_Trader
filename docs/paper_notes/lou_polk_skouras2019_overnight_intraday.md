# A tug of war: Overnight versus intraday expected returns (Lou, Polk & Skouras, 2019)

- **Citation:** Dong Lou, Christopher Polk, Spyros Skouras (2019). *Journal of Financial Economics* 134(1), 192-213. DOI [10.1016/j.jfineco.2019.03.011](https://doi.org/10.1016/j.jfineco.2019.03.011)
- **Verified:** Crossref and the full RePEc abstract, 2026-11-10. Numbers marked (m) come from memory.
- **Topic:** 2. TradFi session effects. Hypotheses: H-TF2 (night effect) and the decomposition of strategy P&L into overnight / intraday (H-PD).

## Question
Do the overnight (close-to-open) and intraday (open-to-close) components of stock returns behave differently, and do known strategies earn their profits in one component?

## Data and method
- CRSP US stocks with open prices (1990s-2010s (m)). Split each daily return into overnight = log(open_t / close_{t-1}) and intraday = log(close_t / open_t).
- Firm-level persistence: cross-sectional regressions of future overnight (intraday) returns on past overnight (intraday) returns, plus cross-period effects.
- 14 trading strategies (momentum variants, reversal, size, value, profitability, investment, etc.) decomposed into overnight and intraday legs.

## Key results
- Strong overnight and intraday firm-level return continuation, with an offsetting cross-period reversal, lasting for years (abstract).
- For all 14 strategies, profits are earned **either entirely overnight** (reversal and a variety of momentum strategies) **or entirely intraday**, typically with opposite signs across the two components (abstract).
- The smoothed spread between a strategy's overnight and intraday components forecasts its close-to-close performance (abstract).

## Limitations / transfer to Bybit perps
- Bybit equity perps trade 24/5 (frozen on weekends, BELIEFS P5). "Overnight" there is a traded period with crypto-native participants, not a no-trading gap, so the clientele story (institutions trade intraday, individuals at the open) may not carry over.
- History is ~5 months for TradFi perps, so tests are holdout-dominated and low-power.
- The persistent firm-level component needs a cross-section of stocks; we have ~9 equity perps.
- The best-known aggregate version, the S&P overnight drift, **disappeared after 2020** (see [boyarchenko_larsen_whelan2023_overnight_drift.md](boyarchenko_larsen_whelan2023_overnight_drift.md)). The H-TF2 prior (0.35) looks, if anything, high.

## How we use it here
- B2 `research/tradfi_session_study.py` (H-TF2): compute the overnight / intraday split per equity perp from 1h bars using the NYSE calendar with DST, and test mean(overnight - intraday) with NW SEs clustered by date.
- Decomposition tool: for any strategy trading TradFi perps, split its P&L into NYSE-session vs off-session legs (B1). A strategy whose edge sits entirely off-session is exposed to the weekend freeze and to gap risk at the US open (R-VOL-9).
- Crypto analogue: split crypto strategy P&L by "US cash session open" vs "closed". That is cheap and informative given V1 / V10.

## Evidence quality
**Strong** for US stocks (JFE). **Weak** transfer to 24h perps.
