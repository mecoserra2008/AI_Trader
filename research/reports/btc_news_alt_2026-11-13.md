# BTC news attention (non-GDELT) vs next-day volatility and direction (2026-11-13)

Agent W. Pre-registered H-NEWS3 / H-NEWS4 (research/hypotheses.yaml). GDELT rate-limited the history (3 of 20 series), so these tests replace H-NEWS1/2 with sources whose history can be downloaded for free. Design: scripts/news_alt_study.py docstring; features: src/ai_trader/news/alt.py; point-in-time tests: tests/test_news_alt.py.

## What was measured, in plain words

Every day we ask: given what was public at midnight UTC, does knowing how much attention bitcoin and the crypto industry got (Wikipedia page views and edits, Hacker News stories and comments, the Fear and Greed index), plus scheduled macro events and on-chain activity, make tomorrow's BTC volatility forecast better than the standard model (log-HAR with a day-of-week effect)? And does it say anything about whether BTC goes up or down tomorrow?

## Data obtained

| source | series | days | first .. last | notes |
|---|---|---|---|---|
| wiki | pv_Binance | 4153 | 2015-07-01 .. 2026-11-12 | ok |
| wiki | pv_Bitcoin | 4153 | 2015-07-01 .. 2026-11-12 | ok |
| wiki | pv_Blockchain | 4153 | 2015-07-01 .. 2026-11-12 | ok |
| wiki | pv_Coinbase | 4153 | 2015-07-01 .. 2026-11-12 | ok |
| wiki | pv_Cryptocurrency | 4153 | 2015-07-01 .. 2026-11-12 | ok |
| wiki | pv_Cryptocurrency_bubble | 4153 | 2015-07-01 .. 2026-11-12 | ok |
| wiki | pv_Cryptocurrency_exchange | 4153 | 2015-07-01 .. 2026-11-12 | ok |
| wiki | pv_Dogecoin | 4153 | 2015-07-01 .. 2026-11-12 | ok |
| wiki | pv_Ethereum | 4153 | 2015-07-01 .. 2026-11-12 | ok |
| wiki | pv_FTX | 4153 | 2015-07-01 .. 2026-11-12 | ok |
| wiki | pv_FTX_(company) | 4153 | 2015-07-01 .. 2026-11-12 | ok |
| wiki | pv_MicroStrategy | 4153 | 2015-07-01 .. 2026-11-12 | ok |
| wiki | pv_Satoshi_Nakamoto | 4153 | 2015-07-01 .. 2026-11-12 | ok |
| wiki | pv_Stablecoin | 4153 | 2015-07-01 .. 2026-11-12 | ok |
| wiki | pv_Strategy_(company) | 4153 | 2015-07-01 .. 2026-11-12 | ok |
| wiki | pv_Tether_(cryptocurrency) | 4153 | 2015-07-01 .. 2026-11-12 | ok |
| edits | Bitcoin | 4334 | 2015-01-01 .. 2026-11-12 | ok; requests 16, edits 7653 |
| edits | Cryptocurrency | 4334 | 2015-01-01 .. 2026-11-12 | ok; requests 6, edits 2800 |
| hn | bitcoin_comment | 3603 | 2017-01-01 .. 2026-11-12 | ok; requests 275, approx_days 0, items 119195 |
| hn | bitcoin_story | 3603 | 2017-01-01 .. 2026-11-12 | ok; requests 119, approx_days 0, items 14749 |
| hn | crypto_comment | 3603 | 2017-01-01 .. 2026-11-12 | ok; requests 287, approx_days 0, items 130423 |
| hn | crypto_story | 3603 | 2017-01-01 .. 2026-11-12 | ok; requests 119, approx_days 0, items 15944 |
| hn | ethereum_comment | 3603 | 2017-01-01 .. 2026-11-12 | ok; requests 119, approx_days 0, items 22399 |
| hn | ethereum_story | 3603 | 2017-01-01 .. 2026-11-12 | ok; requests 119, approx_days 0, items 4176 |
| fng | index | 3204 | 2018-02-01 .. 2026-11-13 | ok |
| guardian | status | 0 | None .. None | refused |

Store: 0.5 MB under data/research/news/alt/. The Guardian refused the public `test` key (HTTP 401), so Guardian counts are absent. 'Bitcoin ETF' has no en.wikipedia article. HN days with more than 1000 matches (only possible for a single day) use the search engine's nbHits estimate (approx_days above).

BTC days 2020-04-09 .. 2026-11-11; first out-of-sample forecast after a 365-day warm-up.

Share of decision days with a non-missing value (missing -> 0 in the regression):

| column | llm_in_sample | clean_holdout |
|---|---|---|
| wp_btc | 100% | 100% |
| wp_ind | 100% | 100% |
| wp_max | 100% | 100% |
| wp_edits | 100% | 100% |
| hn_story | 100% | 100% |
| hn_comment | 100% | 100% |
| fng | 100% | 100% |
| fng_chg | 100% | 100% |
| ev_macro_next | 100% | 100% |
| ev_crypto_next | 100% | 100% |
| ev_macro_today | 100% | 100% |
| oc_ntx | 100% | 100% |
| oc_fees | 100% | 100% |
| oc_mempool | 100% | 100% |
| oc_hash | 100% | 100% |

## H-NEWS3: next-day QLIKE, block-augmented minus log-HAR x weekday (negative = the block helps)

| variant | segment | n | QLIKE base | QLIKE aug | diff | rel | NW t |
|---|---|---|---|---|---|---|---|
| primary | llm_in_sample | 1908 | 0.3538 | 0.3659 | 0.0121 | 0.034 | 1.50 |
| primary | clean_holdout | 134 | 0.3548 | 0.3701 | 0.0153 | 0.043 | 1.46 |
| alt_only | llm_in_sample | 1908 | 0.3538 | 0.3550 | 0.0013 | 0.004 | 0.29 |
| alt_only | clean_holdout | 134 | 0.3548 | 0.3587 | 0.0038 | 0.011 | 0.57 |
| wiki_only | llm_in_sample | 1908 | 0.3538 | 0.3541 | 0.0003 | 0.001 | 0.14 |
| wiki_only | clean_holdout | 134 | 0.3548 | 0.3560 | 0.0011 | 0.003 | 0.31 |
| hn_only | llm_in_sample | 1908 | 0.3538 | 0.3514 | -0.0024 | -0.007 | -0.79 |
| hn_only | clean_holdout | 134 | 0.3548 | 0.3491 | -0.0058 | -0.016 | -0.67 |
| fng_only | llm_in_sample | 1908 | 0.3538 | 0.3567 | 0.0029 | 0.008 | 1.06 |
| fng_only | clean_holdout | 134 | 0.3548 | 0.3625 | 0.0077 | 0.022 | 1.41 |
| events_onchain | llm_in_sample | 1908 | 0.3538 | 0.3600 | 0.0062 | 0.018 | 1.63 |
| events_onchain | clean_holdout | 134 | 0.3548 | 0.3663 | 0.0115 | 0.032 | 1.92 |

Verdicts (primary, recorded): llm_in_sample **contradicts**, clean_holdout **contradicts**. Variants are robustness checks (5 tried), not recorded.

Block coefficients (OLS on ln RV_{t+1}, in-sample days only; plain OLS t, descriptive):

| term | coef | t |
|---|---|---|
| wp_btc | +0.1905 | +2.09 |
| wp_ind | +0.3539 | +2.08 |
| wp_max | -0.1015 | -2.28 |
| wp_edits | -0.0264 | -1.12 |
| hn_story | +0.1707 | +3.21 |
| hn_comment | -0.0566 | -1.95 |
| fng | -0.0417 | -1.06 |
| fng_chg | -0.2291 | -1.64 |
| ev_macro_next | +0.2187 | +3.48 |
| ev_crypto_next | +0.1835 | +0.62 |
| ev_macro_today | -0.0099 | -0.16 |
| oc_ntx | -0.0712 | -0.65 |
| oc_fees | -0.0121 | -0.29 |
| oc_mempool | +0.0372 | +1.58 |
| oc_hash | -0.0469 | -0.34 |

## H-NEWS4: direction (registered as the null)

| segment | n | Brier model | Brier base | skill | 95% CI | hit rate | verdict |
|---|---|---|---|---|---|---|---|
| llm_in_sample | 1908 | 0.2560 | 0.2515 | -0.0181 | [-0.0305, -0.0055] | 0.491 | **supports** |
| clean_holdout | 134 | 0.2540 | 0.2495 | -0.0180 | [-0.0364, +0.0007] | 0.440 | **supports** |

## Plain-language reading

- Before the model's knowledge cutoff (llm_in_sample), 1908 days: adding the 15-column block made the volatility forecast worse by 3.4% of the baseline QLIKE (NW t +1.50); verdict **contradicts**.
- After the cutoff (clean_holdout), 134 days: adding the 15-column block made the volatility forecast worse by 4.3% of the baseline QLIKE (NW t +1.46); verdict **contradicts**.
- Sub-blocks in-sample: none of the five sub-blocks beats the baseline at |t| > 2 on its own.
- Direction (H-NEWS4, registered as the null 'no skill'): llm_in_sample Brier skill -0.0181 (95% CI -0.0305 .. -0.0055, hit rate 49.1%) -> supports; clean_holdout Brier skill -0.0180 (95% CI -0.0364 .. +0.0007, hit rate 44.0%) -> supports. 'supports' here means the attention block does NOT predict direction.

## Caveats

- Strict timing costs a day: HN and edit counts for day t are only complete at t + 1 00:30 and Wikipedia pageviews at t + 2, so the midnight forecast uses attention from day t - 1. Same-day attention may carry more, but it is not knowable at the decision time.
- Wikipedia pageviews are today's vintage of history (Wikimedia does not revise daily counts materially); HN counts are today's index (deleted items are gone).
- 'crypto' on HN also matches cryptography talk (exact word only, no prefix matching); the abnormal transform removes its slow-moving level.
- The clean holdout is about four months; a holdout t of -2 needs a large effect.
- Fear and Greed is itself built from price volatility and momentum, so part of it overlaps with the HAR terms.

## Rule candidates (research/rules.yaml schema)

```yaml
rules:
- id: R-NEWS-5
  kind: avoid
  rule: WHEN forecasting next-day BTC volatility THEN do not add daily news-attention counts (Wikipedia, Hacker News, Fear and Greed) or event / on-chain flags to log-HAR x weekday
  when: next-day BTC vol forecast
  then: keep log-HAR x weekday; read attention spikes as context only
  scope:
    assets:
    - BTCUSDT
    horizon: 1d
    sides: both
  effect: QLIKE diff in-sample 0.0121 (NW t 1.50, contradicts); holdout 0.0153 (t 1.46, contradicts)
  confidence: B
  evidence:
  - H-NEWS3
  review: re-run scripts/news_alt_study.py when the clean holdout has >= 250 days; flip if the holdout verdict changes
  status: active
- id: R-NEWS-7
  kind: avoid
  rule: WHEN making a BTC direction call THEN do not use news attention, Fear and Greed level or change, event flags or on-chain activity as a directional signal
  when: any next-day BTC direction view
  then: direction from these inputs is no better than the base rate
  scope:
    assets:
    - BTCUSDT
    horizon: 1d
    sides: both
  effect: H-NEWS4 Brier skill in-sample -0.0181 CI [-0.0305, -0.0055] (supports); holdout -0.0180 CI [-0.0364, +0.0007] (supports)
  confidence: B
  evidence:
  - H-NEWS4
  review: re-run scripts/news_alt_study.py when the clean holdout has >= 250 days; flip if the holdout verdict changes
  status: active
```
