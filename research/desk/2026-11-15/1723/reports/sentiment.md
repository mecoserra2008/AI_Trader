# SENTIMENT 20261115T1723 (as_of 17:23 UTC)
## Regime
Fear & Greed is 10.0 today, down from 28.0-32.0 a week ago and 13.0 yesterday (news.json). RSS tone on BTC/ETH/XRP
is uniformly negative (ETF-outflow and "worst asset" headlines) while BTC/ETH/SOL/XRP all sit in xs.json's R9
(liquidation) cell with funding at the bottom of their 90d range (p0-p4). The sentiment panel is brand new
(collecting since 16:21Z, ~1h of history, 0h baseline): every ticker's trend is "insufficient" - this is a
snapshot of raw tone on 1-7 headlines per ticker, not a measured spike. Web search below only sanity-checks
direction; it does not add a number the panel can't yet support, and sources disagreed on exact index/price levels.
## Ideas
### 1. BTCUSDT/ETHUSDT/SOLUSDT/XRPUSDT sentiment none (conviction 0.20)
- Thesis: Mood is at a fresh extreme (F&G 10.0) and all four majors are in R9; RSS tone on BTC/ETH/XRP is uniformly
  negative, matching the ETF-outflow headlines, but the panel has 0h baseline and 1-7 headlines per ticker - this
  confirms what the tape already shows rather than adding something new.
- Evidence: fear_greed.value 10.0 (news), counts_24h.by_ticker.BTC 48 (news), BTC tone_24h -0.338 (sentiment), BTC
  mentions_24h 7 (sentiment), BTC funding.pct 0.039 (xs); web: 8 distinct outlets (cointelegraph, decrypt,
  bitcoinist, forklog, bravenewcoin, sentimentrader, kucoin, mexc) ran "extreme fear/capitulation" pieces in 48h.
- Invalidation: BTC - F&G back above 30 and BTC tone turns net positive (>= +0.2 avg, n >= 10), collection uninterrupted.
- Rules: R-NEWS-7 (tone/F&G didn't predict BTC direction or vol), R-NEWS-5, R-BUDGET-1.
- Why this rank: best-evidenced read available (F&G + regime cells are snapshot facts), but it's confirmation, not signal.
### 2. SOLUSDT watch none (conviction 0.15)
- Thesis: SOL funding is at the bottom of its 90d range (p0, -21.3% ann.), the most stressed of the four majors,
  yet attention is near zero (1 headline, tone 0.00; news.json's fuller 24h count is only 4). Web coverage frames
  SOL as the one major with net ETF inflows (~$269M cumulative, nexo.com) against that funding stress - a quiet
  divergence worth flagging, not trading: H-SENT1 is untested and the sample is too small to size on.
- Evidence: funding.pct 0.002, funding.ann_pct_last -21.3% (xs); mentions_24h 1, tone_24h 0.0 (sentiment);
  counts_24h.by_ticker.SOL 4 (news); web: SOL ETF cumulative inflow ~$269M (nexo.com Dispatch #269).
- Invalidation: SOL - mentions_24h rises past 10 with trend no longer "insufficient" while funding normalizes above p20.
- Rules: R-AVOID-1 (low funding percentile alone is not a reason to act), R-BUDGET-1, R-XS-3.
- Why this rank: a genuine "quiet asset" gap, capped by the tiny sample and SOL's own insufficient trend.
### 3. ETHUSDT watch none (conviction 0.10)
- Thesis: ETH's tone_24h (-0.74) is the panel's most negative, but it rests on one headline ("worst asset of all
  time") - n=1, lexicon noise, not a measured shift. Web search shows ETH ETFs in their worst monthly outflow
  (~$728.6M weekly, third-highest on record, Blockworks/Sherwood coverage); if the RSS sample grows and stays this
  negative it would agree with that real number - right now it's one headline next to it.
- Evidence: tone_24h -0.743, mentions_24h 1 (sentiment); counts_24h.by_ticker.ETH 7 (news); oi.chg_pct.24h -2.28%,
  funding.pct 0.043 (xs); web: ETH ETF weekly outflow ~$728.6M (coverage of Blockworks/Sherwood ETF-flow data).
- Invalidation: ETH - news_24h sample grows to >= 5 headlines while average tone stays <= -0.5.
- Rules: R-NEWS-7, R-BUDGET-1.
- Why this rank: lowest conviction - a sanity check on the news agent's ETH tone read, not a standalone call.
## No trade / watch
No idea here is a trade: all three are `sentiment`/`watch` with `direction: none`. H-SENT1 (sentiment as a decision
input) is proposed but untested, so nothing above may stand alone as a reason for the trader to act; at most it is
context next to xs/tape facts (R-AVOID-1, R-BUDGET-1).
## Data gaps
Reddit keys missing (0/147 items are Reddit, RSS-only). Collection since 16:21Z (~1h): baseline_hours = 0 and trend
= "insufficient" for every ticker. sentiment.json mentions_24h undercounts vs news.json's fuller 24h tally (e.g.
SOL 1 vs 4) because the collector's window is only ~1h old. Web search gave inconsistent exact F&G values (10/14/15/5)
and BTC price levels ($63k-$98k) across outlets; only the shared qualitative direction is used, no single number.
## Honesty
direction_claim: false. base_rate: absent an edge, ~50/50 odds of target before stop (R-BUDGET-1); H-SENT1 carries
no weight on its own. Conflicts: R-NEWS-7 (tone/F&G didn't predict BTC direction/vol before), R-AVOID-1
(funding-contrarian failed), R-BUDGET-1 (more inputs don't make direction predictable), and every ticker's trend
being "insufficient" (0h baseline) means none of this is a confirmed spike. What would change my mind: Reddit keys
provisioned plus >=24h (ideally the 60 days H-SENT1 needs) of continuous collection showing a real z_mentions/z_tone
spike that agrees or cleanly disagrees with the same-day funding/OI/price move.
