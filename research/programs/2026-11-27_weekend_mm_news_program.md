# Weekend market making with selection and news (program WMM2, launched 2026-11-27 ~21:50 UTC)

User request (2026-11-27 ~21:35 UTC, paraphrased closely): this weekend should be read as a low-volatility weekend (US
Thanksgiving and other conditions); go into it market making with a smaller book on illiquid names; tell the names that
trend from the ones that do not and make markets only on names with a high mean-reversion index on calm weekends; track
breaks of structure; there is a general buyer bias but altcoins still suffer and a few thrive and trend hard, so be
selective; follow the news to find adverse selection from information as well as from flow; collect past weekends; model
the impact of a news item on single tokens and on the whole market; test the hypotheses and build the models to choose the
assets; quote skew (price-skewed or size-skewed), inventory management, and the filtering and time decay of news and of the
volatility spikes that news drives.

## What is already known (read before designing)
- research/reports/mm_weekend_2026-11-21.md (H-MMW1-6, 4 held weekends, 7 alts): touch-join loses 1.11-1.25 USD per
  symbol-day (adverse selection inside 1 s is ten times the spread captured); the live mm_illiq20 width (B0) is about
  +0.01 USD per symbol-day and not significant; the BTC-jump pull passes the frozen rule only by removing fills; gates
  mostly stop trading (R-MMR-5/6); the liquidation proxy did not discriminate.
- research/reports/mm_replay_2026-11-14.md (R-MMR-1..6); WS10 (research/reports/ws10_2026-11-27.md): makers at the touch
  lose about 0.7 bp of realised spread before fees; pegged entries cost 3.1-3.8 bp; cost table data/research/ws10/cost_inputs.parquet.
- WS04 (loghar_cr is the best 1h/4h sigma; weekend and night sigma is ~10% lower than robust_bc says); WS05 (Hawkes
  jump intensity, liquidation cascades raise jump risk for ~2 h; release minutes 76x jump intensity); R-VOL-1/2 and
  R-PX-9 in research/rules.yaml (weekend rules).
- News data so far: src/ai_trader/news (GDELT daily, rate-limited; Wikipedia, Hacker News, Fear and Greed; macro events
  table; RSS archive since 2026-11-13 stores hashes and timestamps only, no titles). research/reports/btc_news*_2026-11-13.md.

## Workstreams (3 agents; ids H-WMS-n, H-WMN-n, H-WMQ-n)
- **WMS SELECT** (1 GB). Which names to quote and when. Friday-known mean-reversion indices (variance ratio, AR(1) of
  5m returns, OU half-life, Hurst / DFA, efficiency ratio) and trend / break-of-structure measures; do they persist from
  the week into the weekend and predict weekend round-trip behaviour and MM outcomes? A Friday-known "calm weekend"
  classifier (holiday weekends incl. Thanksgiving as a labelled subset, honest about the tiny count). Weekend drift: is
  there a buyer bias in BTC, do alts bleed, how concentrated are the weekend trenders, and can they be seen early
  (breaks of structure, news)? Output: a frozen selection rule and per-name scores that WMQ uses.
- **WMN NEWS** (500 MB). Information arrival. A point-in-time token-level event set from free, ToS-compliant sources with
  history (exchange announcements via official APIs, listing / delisting, hacks, token unlocks, Hacker News, GDELT, and an
  endogenous idiosyncratic-jump detector validated against them). Event studies of abnormal return, volume and
  volatility for tokens and the market; the decay of news-driven volatility (power law vs exponential, exogenous vs
  endogenous shocks; Hawkes with exogenous marks); informational adverse selection: markouts of passive fills around
  news vs flow-only toxicity. Output: a time-decayed news-intensity gate (and widening rule) WMQ can test.
- **WMQ QUOTE** (3 GB incl. order books). How to quote. Replayed weekend books (atx mm-replay on ob200 + trades):
  Avellaneda-Stoikov / Gueant-Lehalle-Fernandez-Tapia reservation-price skew vs size skew vs both, depth placement away
  from the touch scaled by sigma, intensity calibration (fill rate vs distance), inventory caps and inventory exits,
  BTC pull, and the WMS selection and WMN gate as registered conditional tests. Output: whether any variant earns a
  positive held net after fees and adverse selection, with paired tests against B0.

## Forward test
Every weekend after the designs are frozen is forward data (the Bybit archive publishes books and trades a day or two
later). The first is 2026-11-28/29. A live pilot is out of scope: it needs a variant that passes FIT, the clean holdout and
at least the first forward weekends, then explicit live limits signed by the user (the 3-trade delegation expired 2026-11-21).

Every agent follows research/programs/AGENT_RULES.md.
