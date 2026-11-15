# The Pre-FOMC Announcement Drift (Lucca & Moench, 2015)

- **Citation:** David O. Lucca, Emanuel Moench (2015). *Journal of Finance* 70(1), 329-371. DOI [10.1111/jofi.12196](https://doi.org/10.1111/jofi.12196)
- **Decay evidence:** Alexander Kurov, Marketa Halova Wolfe, Thomas Gilbert (2021). "The disappearing pre-FOMC announcement drift." *Finance Research Letters* 40, 101781. DOI [10.1016/j.frl.2020.101781](https://doi.org/10.1016/j.frl.2020.101781). Existence verified via Crossref; text not read.
- **Verified:** Crossref and the RePEc abstract (Lucca-Moench), 2026-11-10. Numbers marked (m) come from memory.
- **Topic:** 2. Calendar / event effects automatable on TradFi perps (QQQ, SPX-linked) and possibly crypto.

## Question
Do US equities earn abnormal returns in the hours before scheduled FOMC announcements?

## Data and method
- S&P 500 (SPX futures / index) intraday, scheduled FOMC meetings from 1994 (when the Fed began announcing decisions at a known time) to 2011 (m).
- Event window: the 24 hours before the 14:15 ET announcement (2pm in later years) vs all other 24h windows. Regression of returns on a pre-FOMC dummy, plus international indices, Treasuries and money-market futures, and other macro announcements as placebos.

## Key results
- Large average excess returns on US equities in the 24 hours before scheduled FOMC announcements. They grew over time and account for a large share of annual equity returns: about **49 bp per pre-FOMC window**, and more than 80% of the realised equity premium in their sample (m).
- Present in international equity indices, but **not** in Treasuries or money-market futures. **Other** macro announcements show no such drift (abstract).
- No consensus explanation (risk-based stories struggle).

## Limitations / transfer to Bybit perps
- Kurov, Wolfe & Gilbert (2021) report that the drift **disappeared** after publication (2015 onward; not re-read here). Another published-then-decayed anomaly, like the overnight drift.
- Only 8 scheduled meetings a year. Our TradFi-perp history (~5 months) has 3-4 events, so no inference is possible for years. Crypto history (2019+) gives ~55 events, enough for a weak test.
- The current context (calendar W46: Dec 9 hike probability ~0.77) is unusual; FOMC weeks in 2026 may be dominated by surprise risk, not drift.

## How we use it here
- Deterministic calendar feature `research/calendar/` -> `is_pre_fomc_24h`. Use it as a **vol / risk context** (expect a vol spike at 14:00 ET), not as a directional tilt, unless a registered test supports the tilt.
- Draft hypothesis H-LIT-TF8 (literature report): **no** pre-FOMC drift in BTC / ETH (2019-2026, with the post-cutoff holdout) or the QQQ perp. Prior 0.75 for "no drift", i.e. only ~0.25 that a drift exists, because of documented decay.

## Evidence quality
**Strong** for 1994-2011 (JF). It **decayed** after publication, so the current expected value is low.
