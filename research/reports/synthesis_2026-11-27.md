# Specification program: synthesis (2026-11-27)

Ten workstreams (compacted from twenty) reported between 00:14 and 07:50 UTC. Reports: research/reports/ws01..ws16_2026-11-27.md.
Every hypothesis was registered before testing; every result is in research/hypotheses_results.jsonl (835 rows).

## What we now believe

1. **No directional edge survives proper inference.** WS08 ran 196 predictor × asset × horizon cells (1h-30d) with IVX,
   Stambaugh, joint block bootstrap and Romano-Wolf: 2 survive, both at 1h and both net negative after costs (BTC basis
   +1.1 bp gross). Positioning, funding, flows, sentiment, macro, cross-sectional momentum/carry, regime gating (WS07),
   participant types (PART), intraday geometry (GEO), ML meta-labelling (WS12) and the xs networks (WS14: their "relative
   strength" is a volatility-drag units effect) all fail. At 1h/4h these nulls are well powered; at 1d-30d they only rule
   out large effects.
2. **Volatility and risk are forecastable, and our current sigma is the wrong model.** The clock log-HAR on 5m RV with
   a release dummy (loghar_cr) beats the live robust_bc by 35% (1h) and 44% (4h) QLIKE on the holdout (t -8.8 / -4.9) and
   is alone in the 1h Model Confidence Set for every asset (WS04). Adding the high-low range captures most of what small
   deep nets learn (WS12). GARCH variants lose by 25-63%. Today's stops are 30-40% too tight in US hours and ~10% too
   wide at night and on weekends.
3. **Tails:** Student-t with nu ~3.5-3.8 (fatter than the usual default); a 4-sigma 1h move happens 0.31% of hours;
   with a 4-sigma stop the 99% ES of the 1.30 USD unit is 1.27 USD (WS04). Joint crashes are 1.6x fatter than Gaussian
   (t-copula nu~6); DCC is the covariance spec; 2-3 same-side crypto positions are ~1.3 independent bets in every regime,
   co-stopping 27-40% of the time, and the live limits contain it (worst same-side co-stop 3.01 USD) (WS05).
4. **Event risk:** CPI / NFP / FOMC minutes multiply jump intensity ~76x and variance ~14-16x; every forecast band
   (network, ACI, HAR) covers ~45% at 80% in release hours (GEO, WS05, WS14). Liquidation cascades raise jump risk for ~2h,
   not 60 min (R-FLOW-2 is too short; an intensity gate fits better than a longer clock). The field P(jump) is calibrated
   only for its own latent state and is worse than a constant for observable jumps: stop using it as a gate (WS05).
5. **Execution and costs:** pegged-limit entries at the touch cost 3.1-3.8 bp all-in vs 5.5-6.2 bp taker, positive
   saving in every symbol-hour (WS10); stop slippage is small on average (~1 bp, p99 10-18 bp; budget ~20 bp for the
   tail); funding is the missing cost for multi-day trades (zoo_ensemble pays ~0.037 R per trade).
6. **Trend is fragile, not proven.** With the red-team fixes (closed trades only, block >= p90 holding period, prior
   trials counted) and calibrated costs + funding: zoo_ensemble +0.28-0.32 R/trade [~-0.02, +0.70]; 594 of its 687 R come
   from 2023-24 entries; the last 23 months are flat; the closed holdout (+0.08-0.10 R on 131 trades) says nothing.
   38 variants cannot beat plain zoo_ew (SPA p 0.60, PBO 0.42). Forward expectation ~+0.1 R/trade, ~5 trades/month on the
   live engine (~+0.6 USD/month): a test of execution and of whether the edge still exists, not an income stream.
7. **The research process had real defects.** Registry-wide FDR: only 47 of 123 effect "supports" survive BY at 10%;
   15 hypotheses replicate robustly, all about volatility, timing, densities or execution (WS01). verdicts.py judges
   holdouts on sign only; the stress harness used 5-day blocks for week-long trades and marked open positions at the
   data end (this created "zoo_ensemble SURVIVES"). Several A-grade rules rest on underpowered tests (R-LEV-1) or on
   cell effects that vanish with continuous controls (R-MAP-2..7).
8. **Data:** no look-ahead anywhere in the lake (full census). But flow_3s covers ~47% of wall time (PC sleep 58% of
   the loss, Windows Modern Standby 29%, collector stalls 9%), trades are booked late into 3 s candles, and several
   feeds went stale on 2026-11-07..10 (instruments snapshot, OI 1h, index, ls_ratio, stablecoins, Deribit). Gap map:
   data/research/ws02/gaps.parquet.

## Decisions for the user (ranked by impact)

| # | Decision | Recommendation | Touches |
|---|---|---|---|
| D1 | Replace robust_bc with loghar_cr (+ range terms) for sigma_risk at 1h/4h; keep log-HAR at 1d; no weekend shrink in stop sigma | shadow 1-2 weeks next to robust_bc, then switch | live risk input (stops, EV gate) |
| D2 | Release-hour rule: no new entries with stops < 2 sigma from 5 min before to 30 min after CPI / NFP / FOMC; widen bands for the release bar and the next | adopt | live gate |
| D3 | Drop the field P(jump) as a gate; replace R-FLOW-2's fixed 60 min with a Hawkes-intensity gate (~2h decay) | adopt after a shadow | live gate |
| D4 | Paper trend sleeve: replace trend_4h ema_stack with zoo_ew (spec A) + tsmom_30d benchmark (spec B), with the WS16 kill rules | adopt on paper | paper only |
| D5 | Retire shock fade (H-S2) and setup C from live use | adopt | desk playbook |
| D6 | Merge rule changes: R-STRESS / R-GEO / R-WS04 / R-WS05 / R-WS07 / R-WS08 / R-WS10 / R-WS12 / R-WS14 candidates; downgrade R-LEV-1 to C; restate R-MAP-2..7 | review list, then merge | research/rules.yaml |
| D7 | Fix verdicts.py (holdout must pass a test, not just a sign) and stress.py (block length, closed trades) | adopt | research code |
| D8 | Ops: PC never sleeps and Modern Standby off on AC; collector auto-start + heartbeat + candle watermark; restart stale updater jobs; lake cleanups | adopt | machine, collector, updater |
| D9 | Allow the Chronos-Bolt venv install (foundation-model test was refused by the permission system) | optional | install permission |

## What remains unknown
- Whether a small daily-to-monthly directional edge exists (underpowered; needs more history or more assets).
- Whether the 2023-24 trend profits were a regime; whether the short side of trend pays.
- The cross-section without survivorship bias (delisted-perp history is not in the lake).
- Real cross-venue price discovery (needs synchronous data at 1 s or finer).
- Foundation models (not run).
