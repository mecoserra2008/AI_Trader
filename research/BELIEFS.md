# Working beliefs (v1, 2026-11-07)

_What the evidence currently supports, how confident we are, and how each belief is used. Every line traces to [hypotheses.yaml](hypotheses.yaml) + [SCOREBOARD](reports/SCOREBOARD.md). Beliefs change only through registered tests, and holdout data after 2026-07-01 is the only evidence Claude can't have memorised._

Confidence:
- **A**: supported both before and after the cutoff, large effect;
- **B**: supported but small effect, or forward-confirmed once;
- **C**: plausible, forward test running.

## 1. Volatility: what we believe and how we use it

| # | belief | conf. | evidence | how it's used |
|---|---|---|---|---|
| V1 | Weekends (Fri 17:00 → Sun 18:00 ET) are quiet: realised vol ≈ 0.6–0.8× the Friday σ̂ | A | H-W1 | central σ is weekend-scaled (`sigma_rb_next`, m ≈ 0.57 for BTC) |
| V2 | Weekend realised variance is far below what a 7d ATM implied vol charges per calendar hour… | A | H-W7 (t −17 / −7.7) | descriptive truth, but **not harvestable as tested**: Friday 3-day options already price the weekend ~20% cheaper, the Fri 08:00 → Mon 08:00 window includes the loud Friday US session and Sunday reopen, and hedging costs exceed the edge (H-OPT1–3 failed both segments). Only a US-close entry with band hedging remains: H-OPT4, forward-only |
| V3 | …but weekends are jump-prone: shrinking σ for *risk* worsens tail loss | A | H-V2 failed on QLIKE | stops and sizing use `sigma_risk_next` (no weekend shrink) |
| V4 | Jumps decay fast (excess-variance half-life 2–3 h); the plain EWMA over-forecasts post-jump vol by ~27% | A | H-S4 | jump-robust, bias-corrected EWMA (`vol.robust_bc`, H-V1b, QLIKE −0.05 both segments) |
| V5 | Vol clustering is multi-scale: HAR (1d / 7d / 30d) beats the single-λ EWMA at 24h | A | H-VCL1 (QLIKE −0.08 both segments) | daily risk forecast = `vol.har`; per-bar σ = `vol.robust_bc` |
| V6 | Vector candles and Amihud illiquidity spikes do *not* predict extra vol beyond the robust forecast | A (negative) | H-VC4, H-LC2 | no special vol regime after them |
| V7 | Implied vol (DVOL) adds a little to HAR for 7d BTC/ETH variance | B | H-VCL4 (−3% in-sample weak, −19% holdout) | add DVOL as a feature in BTC/ETH risk sizing |
| V8 | No leverage-effect asymmetry, no BTC→alt spillover beyond own HAR, no compression-expansion edge | A (negative) | H-VCL2, H-VCL3, H-VCL5 | HAR stays plain |

## 2. Price behaviour

| # | belief | conf. | evidence | how it's used |
|---|---|---|---|---|
| P1 | **Shocks into calm markets continue; shocks into volatile markets reverse** (vol state at shock time decides) | B / C | H-S2 forward-supported; H-S3 forward test from 2026-11-08 | **first strategy to forward-test: `shock_regime`** |
| P2 | Hourly moves away from trend keep going; fading 2σ hourly deviations loses even before costs | B | H-T1 forward-supported | mean-reversion strategies stay off at intraday horizons |
| P3 | Slow trend has a weak edge (Sharpe ≈ 0.2–0.3 before the cutoff), negative in Jul–Nov 2026; 4h > 1h, 1d not better | B / C | H-X2b, H-X2c forward test | trend is a low-weight baseline sleeve, not a core edge |
| P4 | Structural "levels" (vector-candle opens and closes, DC swing pivots, CME gaps, swept zones) do **not** hold more than a random walk; price goes *through* them slightly more often (break ≈ 0.52–0.55) | A (negative) | H-VZ1–4, H-VC2/3/7, H-GT3, H-W4 | level-chain prior: π_break ≥ 0.5 by default; levels are where to *expect* the move to continue, not to fade it |
| P7 | Simple trend models (21-variant zoo) earned thin edges before the cutoff (best tsmom_14d +0.50, ensemble +0.33; deflated Sharpe of the winner 0.72 < 0.95) and are negative across the board Jul–Nov 2026; walk-forward model selection loses to the ensemble; 'is it trending?' isn't forecastable (H-TR3/H-TR4); zoo is ~1.9 independent bets | A (negative for deployment) | H-TR1–4 | trend stays paper-only (ensemble tracked forward); net-E[R] gate stays at 0.10 |
| P6 | Neither auction balance (value-area overlap) nor the variance ratio says when to trend-follow vs mean-revert; value-area / POC levels behave like random-walk levels | A (negative) | H-AU1–4, H-RG1 | no regime switch on these; 1h MR stays off everywhere |
| P5 | Weekend direction isn't predictable from price/volume; Bybit TradFi perps freeze on weekends and carry no information | A (negative) | H-W2/3/6 | no weekend directional trades; Monday NYSE-open vol is higher after eventful weekends (H-W5b, in-sample) → widen risk there |

## 3. Things that don't work here (don't spend capital on them)

| idea | evidence |
|---|---|
| funding contrarian, OI-price divergence as standalone 4h signals | H-P1, H-P2 |
| market-profile rules (acceptance, 80% rule, POC magnet), variance-ratio regime switch | H-AU1–4, H-RG1 |
| relative value at taker costs: alt residuals vs BTC/ETH, Brent–WTI spread, XAUT–XAU basis (reverts ~28%/24h but sd 9–14 bps vs 30 bps costs) | H-RV1–3 |
| stablecoin growth, VRP as return predictors | H-FU1, H-FU2 |
| Fear & Greed contrarian | prior project (no effect) |
| pivot-geometry trend (STRUCT / PIVOT_BO / FLAG_BO) | H-GT1/2: a noisier copy of slow trend |
| vector-candle continuation, zone revisits, zone S/R | H-VC1–5, H-VZ1–4 |
| close-confirmed stops instead of resting stops | H-X1 (worst −1.9R vs −1.0R) |
| continuous-signal trend at 1h | costs eat the whole gross edge |

## 3b. Carry

- **C1: funding carry is real but regime-dependent and has decayed** (H-C1: net Sharpe 2.8 in-sample, but 57%/yr in 2021 → 0.5% in 2025 → −4.8% in 2026 H1). It's a leverage-mania premium, not a standing one. Forward test: H-C1b, an always-on overlay scaled by trailing 30d funding.

## 4. Execution and risk beliefs

- **Keep resting stops** (Bybit position SL, MarkPrice trigger) (H-X1).
- **Size on `sigma_risk_next`** (robust_bc, no weekend shrink); daily risk budget from HAR.
- **Open design conflict:** structural stops for breakout / structure strategies sit a median **6–8σ** from entry (donchian, FLAG_BO), but `placement.max_stop_distance_sigma = 4`. Either raise the cap (the position just gets smaller; risk per trade stays fixed) or don't run structure-stop strategies. **Decision needed from the user.**
- Order flow: as **entry timing**, not standalone alpha (prior project + H-F5). Untested until ≥ 30 days of `flow_3s`.

## 5. What to exploit first (initial forward-test book)

| priority | sleeve | status | what gets measured forward |
|---|---|---|---|
| 1 | `shock_regime` (P1), robust σ sizing, resting stops | built; forward from 2026-11-08 | H-S3 + strategy P&L vs RANDOM_N on the same shock events |
| 2 | Weekend vol (V2): short-vol study on Bybit BTC/ETH options (weekly, Friday entry, delta-hedged) | to research | premium vs weekend jump losses, net of option spreads |
| 3 | 4h trend baseline (ema_stack / donchian), small weight | built | H-X2c; correlation with sleeve 1 |
| 4 | Order-flow timing for entries | collecting data | H-F1–F5 after ≥ 30 days |

**The LLM's role on top of this book:** read the state cards (regime cell, robust and HAR σ, weekend context, shock context, flow), and express bounded views on these candidates only. A belief changes only through a registered test.
