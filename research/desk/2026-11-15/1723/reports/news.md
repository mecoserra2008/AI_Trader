# NEWS 20261115T1723 (as_of 17:23 UTC)
## Regime
Risk-off, deepening but off the lows. BTC fell ~68k -> four-month low ~62k over Nov 13-15 on a 7th
straight day of spot-ETF outflows, the Nov 10 hot-CPI hangover and AI-capex doubt, then rebounded to
~64.8k by this snapshot (options.json underlying). F&G 10 (extreme fear, vs 28 a week ago). BTCUSDT v0
cell R9 (liquidation, dir down, vol high, 24h ret -0.99%); funding negative (ann -3.2%, z -1.78): shorts
paying longs, a flush signature, not fresh conviction. Active narratives (research/narratives/STATE.md):
inflation_fed, ai (BTC~NVDA same-day co-move ~0.53), crypto-specific bleed (ETF outflows, CLARITY Act
stalled in the lame-duck Senate). No new network-wide hack/insolvency found beyond the feed (Kava Lend
hack -$3.8M/-40% KAVA Nov 14; Coinone breach warning Nov 15). Per R-NEWS-5/7 none of this is a direction
or next-day-vol call.
## Ideas
### 1. NVDAUSDT, BTCUSDT event none (conviction 0.35)
- Thesis: Nvidia Q3 FY2027 reports Tue Nov 17 22:00 UTC (calendar 2026-W46.yaml, importance 3), the next
  AI-capex test after the Nov 5 selloff (BTC -5.5%, NVDA -10.2%); both now sit in v0 cell R9. A weak read
  risks a same-session spillover; R-PX-12 caps the reaction to the same hour.
- Evidence: counts_24h.by_ticker.NVDA 14 (news), NVDAUSDT regime.ret_24h_pct +1.13% (xs), NVDA reaction
  -10.2% on E-2026-11-05-ai-selloff (narratives/events.yaml)
- Invalidation: NVDAUSDT, 1h close back inside the pre-print 24h range 4h after the call
- Rules: R-PX-12, R-NEWS-7, R-AVOID-4
- Why this rank: the only scheduled, in-universe catalyst with a sized precedent this window.
### 2. BTCUSDT, ETHUSDT volatility none (conviction 0.30)
- Thesis: three daily Deribit expiries in-window (08:00 UTC Nov 16/17/18): notional 343.8M/146.1M/59.5M,
  P/C 1.99/1.52/1.36, max pain 68,000/66,000/65,000 vs 64,843 underlying -- put-heavy, max pain above
  spot, converging by the 18th. R-VOL-28 found no pin/release edge for the weekly Friday case; context
  only here (these are daily).
- Evidence: options.BTC.expiries.0.max_pain 68000 (news), options.BTC.expiries.0.put_call_oi 1.99 (news),
  options.BTC.underlying 64842.78 (news)
- Invalidation: BTCUSDT, trades through 68,000 before the Nov 16 08:00 UTC expiry
- Rules: R-VOL-28, R-NEWS-7
- Why this rank: clean snapshot data, but the tested rule found no edge even for the larger weekly case.
### 3. BTCUSDT event none (conviction 0.25)
- Thesis: FOMC minutes (Oct 27-28 meeting) release Wed Nov 18 19:00 UTC, the second Fed read since the
  Nov 10 hot CPI (BTC -1.60%, z -2.4 in the 13:00-16:00 UTC window, per events.yaml). Lower amplitude than
  CPI; vol event, not direction (R-NEWS-5).
- Evidence: events.E-2026-11-10-cpi-october-hot.reaction.BTC -1.60 (narratives/events.yaml),
  release_time_et 1400 (web:financecalendar.com)
- Invalidation: BTCUSDT, no change in Polymarket Dec-hike odds around the release
- Rules: R-NEWS-5, R-NEWS-7, R-VOL-27
- Why this rank: web-sourced time, not in this snapshot's calendar or scheduled_24h (empty).
### 4. BTCUSDT flow none (conviction 0.20)
- Thesis: spot BTC ETFs on a 7th straight outflow day (~2.1B USD in November, Decrypt; already in our
  feed), CLARITY Act lame-duck odds ~3% (Polymarket, after a 49-50 cloture failure) removing a
  regulatory tailwind while outflows continue. Forced-flow context, not a direction call (R-NEWS-7).
- Evidence: counts_24h.items 147 (news), etf_outflow_november_usd 2.1B (web:decrypt.co),
  clarity_act_2026_passage_odds_pct 3 (web:analyticsinsight.net)
- Invalidation: BTCUSDT, a single day of net ETF inflows breaks the streak
- Rules: R-NEWS-7, R-NEWS-5
- Why this rank: already in this snapshot's own headlines, included for completeness only.
### 5. ARBUSDT flow none (conviction 0.15)
- Thesis: Arbitrum unlocks ~92.65M ARB (~27.6M USD, 2.33% of supply) 2026-11-16 13:00 UTC (web,
  Tokenomist-sourced). ARBUSDT's own 24h turnover here is ~25.8M USD: the unlock is roughly a full day's
  turnover, a real in-universe supply-side candidate, but unlock-day impact is not a tested edge.
- Evidence: ARBUSDT turnover_24h_usd 25,818,927 (xs), arb_unlock_usd 27,630,000 (web:itiger.com)
- Invalidation: ARBUSDT, turnover absorbs the unlock without a >1 sigma move around 13:00 UTC
- Rules: R-NEWS-7
- Why this rank: single-token, web-only unlock size, no on-chain confirmation by us.
## No trade / watch
All five ideas are watch/context (direction none); no size or trade proposed by this agent.
## Data gaps
- No NVDA consensus/guidance figures (not in our toolset); only the scheduled time + precedent sizing.
- FOMC minutes time is web-sourced; scheduled_24h in news.json is empty for the next 24h.
- ARB/STRK/ZRO unlock sizes are web-sourced (Tokenomist, iTiger, Kucoin), not on-chain verified.
- FinancialJuice excluded (scraping its calendar breaches its terms); no dedicated rule for daily (vs
  weekly) Deribit expiry behaviour.
## Honesty
direction_claim: false. Base rate: none of these ideas call BTC/ETH direction; R-NEWS-7/R-NEWS-5 apply
directly. Conflicts: R-NEWS-7 (extreme F&G reads like capitulation but has no measured edge), R-VOL-28
(the above-spot max-pain skew looks like upside pin pressure but the tested rule found none), R-PX-12
(NVDA-BTC co-move is same-hour only), R-AVOID-4 (no session tilt). What would change my mind: a forward
re-run of R-VOL-28/R-NEWS-7 flipping holdout skill, or a scheduled event producing a >2 sigma BTC move
that holds past the same-hour window.
