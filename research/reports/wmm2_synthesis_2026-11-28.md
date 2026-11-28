# Weekend market making with selection and news (WMM2): synthesis (2026-11-28)

Program: research/programs/2026-11-27_weekend_mm_news_program.md. Reports: research/reports/wms_2026-11-27.md (SELECT),
wmn_2026-11-27.md (NEWS), wmq_2026-11-27.md (QUOTE). Every hypothesis (H-WMS-1..13, H-WMN-1..15, H-WMQ-1..15) was
registered before its test; results are in research/hypotheses_results.jsonl. FIT = before 2026-07-01, holdout after.

## Answer to the request
The request was to market make a small book of illiquid alts into a calm (Thanksgiving) weekend, picking mean-reverting
names over trenders, watching breaks of structure, the buyer bias and alt bleed, and news-driven adverse selection, with
price or size skew, inventory management and a time-decayed news filter.

1. **No quoting scheme earns money after fees and adverse selection on held weekends** (WMQ, 21 weekends, 12 alts, 504
   symbol-days). The best (symmetric quotes at a calibrated 0.5-1.5 sigma depth) makes +0.0002 USD per symbol-day
   [-0.015, +0.014] at the minimum lot; the live width (B0) lost -0.0125. Near-touch quoting loses (adverse selection
   inside 1 s exceeds the spread), and deep quotes, whose fills do revert (+15 bp at 1 sigma over 5 min), fill too rarely
   to matter. No live pilot is supported.
2. **Price skew (Avellaneda-Stoikov / GLFT) fails as calibrated**: fill intensity is a power law in depth, not exponential;
   the exponential fit puts quotes ~0.1 sigma from the mid and loses -0.29 per symbol-day. **Size skew adds nothing**
   (-0.0008 per lot). Inventory exits and the BTC pull reduce losses of a losing quoter; they do not make one profitable.
3. **Selection** (WMS, 307 weekends, 572 perps incl. 188 delisted): short-horizon mean reversion (5m AR(1), BoS count)
   persists into the weekend; a composite score tilts the pick ~5% toward less trending names but is not a trender
   filter (holdout IC -0.03, t -1.3). Weekend breaks of structure slightly revert. No measurable BTC weekend buyer bias
   (MDE ~0.5%). Alts bleed vs BTC, but just as much midweek. The trenders are few and mostly up-movers, and Friday data
   finds them mainly because they were already volatile. On the quoting replay the selection changes nothing measurable.
4. **News** (WMN): free news explains almost no price jumps (0.1-0.2%, the same as a placebo) and adds no measurable
   adverse selection beyond order flow (bound ~1-1.8 bp). The hazards are flow (same-side 5 min flow costs a maker ~1.6 bp
   at 60 s) and price-detected jumps (~1 bp in the first minute, power-law decay). News categories that move variance:
   cross-exchange listings, Bybit delistings, US macro releases. The frozen news gate's value comes from pulling after
   jumps; its slow news kernels add nothing; on the quoting replay it changes ~nothing.
5. **This weekend was not calm** (P(calm) 0.44 from the frozen model, lifted only by the Thanksgiving flag; BTC 7-day RV at
   the 97th percentile). The first forward day had the case the request feared: WLFI -29% in the 04:00 UTC hour (-57% on
   the day at worst) after a Friday-long bleed, TRUMP -10..-14%, and the USD1 stablecoin depegging to 0.908 on Bybit. WLFI
   was in the selection's top third; the closed calm gate is what would have kept a quoter out.

## What to use and what not to
- Use: the calm-weekend gate as a hard precondition for any weekend quoting; the live jump gate (live/gates.py) for any
  passive order after a jump; depth in sigma units, never fixed bp.
- Do not use: touch-joining or AS/GLFT quotes with an exponential intensity fit; size skew; news headlines as a quoting
  signal (no evidence beyond flow).
- Forward: the weekends from 2026-11-28 are scored as the Bybit archive publishes them (`python -m
  ai_trader.research.wmq_study run fwd`; H-WMN-15 after 2026-12-21). A paper-only shadow of the 1 sigma quoter is the
  most it could earn on current evidence (expect 0 +- 0.015 USD per symbol-day, worst book day ~ -3 USD).

## Unknowns
- Whether deep quotes become worth it at larger size (P&L scales linearly; CI still spans 0 at 10 lots).
- Calm weekends specifically (only 3 held weekends had the gate open: +0.036 [-0.026, +0.115], underpowered).
- Token unlock schedules (no free source) and spread/depth history before 2026-07 (no data).
