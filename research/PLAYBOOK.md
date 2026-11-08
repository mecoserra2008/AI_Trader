# Decision playbook (v1, 2026-11-07)

_How Claude decides in the moment, given what the evidence says today ([BELIEFS.md](BELIEFS.md), [SCOREBOARD](reports/SCOREBOARD.md)). State cards: `uv run python scripts/state_now.py BTCUSDT ...`._

## 1. Where the natural alpha is (who is paying, and why they keep paying)

Edge that lasts comes from someone **structurally willing to lose**, not from chart patterns. Every pattern-style idea we tested (levels, zones, vector candles, geometry, gaps) failed against proper nulls. What survived has an identifiable payer:

| rank | source | who pays and why | our evidence | how to harvest | status |
|---|---|---|---|---|---|
| 1 (downgraded 2026-11-07) | **Weekend variance risk premium**: H-OPT1–3 failed; market already discounts weekends in short-dated options; only a US-close entry (H-OPT4) remains to test forward | option buyers pay for calendar time while TradFi is closed; insurance demand doesn't price the quiet | H-W7: weekend realised var = 0.2–0.45× implied (t −17 / −7.7); H-W1 | sell weekend vol (Fri → before Mon US open), delta-hedged, on Bybit options | **strongest. Needs an options cost and tail study** |
| 2 | **Liquidity provision after forced flow** | liquidated / panicking traders sell at any price in volatile regimes; whoever absorbs it is paid | H-S2 (R9 −0.23 t −4.5), H-S3 forward test | `shock_regime` REVERSAL leg (fade shocks in normal/high-vol cells) | forward test from 2026-11-08 |
| 3 | **Under-reaction to news in calm markets** | slow diffusion: a big move into a quiet book reprices and keeps going | H-S2 (R1 +0.14 t 3.8; holdout +0.45 t 2.8) | `shock_regime` CONTINUATION leg | forward test |
| 4 | **Leverage carry** | long speculators pay funding to hold leverage | H-C1 supported but decayed: 57%/yr (2021) → 0.5% (2025) → −4.8% (2026 H1) | delta-neutral short perp / long spot, scaled up only in high-funding regimes | H-C1b forward test |
| 5 | **Slow trend (time-series momentum)** | behavioural herding, hedging flows | weak (Sharpe 0.2–0.3 before the cutoff, negative Jul–Nov) | 4h ema_stack / donchian, small weight | baseline only |
| 6 | **Not paying the adverse-selection tax** | (cost, not alpha) informed flow picks off careless liquidity | H-F5 / H-AS1 / H-AS2 pending flow data | time entries with flow; avoid trading into toxic OFI | collecting data |

Two rules fall out of this:
- **Be the insurer and the liquidity provider, not the chart-pattern trader.**
- **Volatility is far more predictable than direction.** Robust EWMA, HAR, the weekend effect and post-jump decay are all supported; direction signals mostly aren't. So the main edges are **sizing, timing and vol selling**, with direction only where a forced-flow story exists.

## 2. The in-the-moment decision procedure

Every session, for each candidate, Claude reads the state card and walks this tree in order. The first matching scenario sets the stance; the risk engine sizes it.

| # | state (all from the card) | stance | why (belief) | invalidation / kill |
|---|---|---|---|---|
| S1 | a \|z\| ≥ 4 shock in the last bars **and** vol_state was **low** at the shock | **CONTINUATION** in the shock's direction | P1, H-S2 | full retrace of the shock bar (close beyond its open) |
| S2 | a \|z\| ≥ 4 shock **and** vol_state was **normal/high** (esp. R9) | **FADE** toward the shock bar's open | P1, H-S2 | close beyond the shock extreme + 0.5σ |
| S3 | Friday into the TradFi weekend, implied (DVOL / 7d IV) > HAR forecast | **sell weekend vol** (when the options sleeve exists); no new directional trades | V1, V2, P5 | close the short-vol position before Monday 09:30 ET; any weekend \|z\| ≥ 4 shock means flatten |
| S4 | weekend in progress, no shock | **hold, don't initiate**; keep risk σ unshrunk (stops never weekend-tightened) | V3, P5 | n/a |
| S5 | Monday NYSE open after an eventful weekend (\|wk_z\| ≥ 2) | **cut size into 09:00–12:00 ET** (vol ≈ 1.4× normal), no fresh entries in the first hour | H-W5b | n/a |
| S6 | trend cell (R1/R2/R7/R8), 4h trend signal aligned | small trend position; **never fade hourly deviations** | P2, P3 | trend signal flips |
| S7 | range / compression (R4/R5), no shock | **stand aside** (mean reversion has no edge here); a shock in R4 is S1 (calm, so continuation) | P2, P4 | n/a |
| S8 | price approaching a "level" (vector open, swing high, gap) | **don't fade it**: break ≥ 0.5 by default; targets go *beyond* levels, not in front | P4 | n/a |

Sizing, in every scenario:
- `qty = R / (stop distance + costs)`, with the stop from structure and σ from `sigma_risk_next`; daily risk is checked against the HAR forecast;
- the LLM's view can only tilt `p_TP` within ±1 log-odds (ARCHITECTURE §2.4).

## 3. Worked example: Saturday 2026-11-07, 21:00 UTC (from `state_now.py`)

| | BTC | ETH | SOL | XRP |
|---|---|---|---|---|
| cell | R7 grind down, low vol (9th pct) | R7, low vol (10th pct) | R7, low vol | R8 trend down, normal vol |
| weekend so far (24 h) | −0.10σ, realised 0.26× Friday σ̂ | −0.14σ, realised 0.30× | | |
| last shock | −5.1σ Thu 15:00 in R6 (high vol) → S2 fade; trade already closed | −5.4σ Thu 15:00 in R3 (high vol) → S2 fade **still open** (entry 2,251, target 2,342, now 2,231) | none | none |
| 4h trend | −0.41 | −0.36 | −0.66 | −0.62 |
| implied / HAR forecast | 1.19× (DVOL 40 vs 34) | **1.39×** (56 vs 41) | | |
| funding carry | +7%/yr | +4%/yr | +5%/yr | +1%/yr |

**How I would decide tonight:**
1. **Main edge: volatility (S3/S4).** We're 24h into a TradFi weekend that's delivering 0.26–0.30× the Friday vol, while options price ETH at 1.39× and BTC at 1.19× our forecast. That's before the weekend discount, which our multiplier puts at m ≈ 0.57–0.68. This is exactly the H-W7 situation. With an options sleeve: short ETH/BTC vol into Sunday, closed before Monday 09:30 ET. Without one (today), it's a note, not a trade.
2. **No new directional entries (S4).** Weekend direction isn't predictable (P5), and nothing has shocked.
3. **Keep the ETH S2 fade on as specified.** The rule decides; I don't override it because it's underwater. It exits at 2,342 (target) or on invalidation beyond the shock low + 0.5σ. Sizing used the unshrunk risk σ.
4. **Trend sleeve: small short tilt (S6)** across BTC/ETH/SOL/XRP: R7/R8 with negative 4h trend. The R7 forward-path prior skews down (mean −0.14σ at 24h, MAE −4.2 vs MFE +3.3), but the trend edge is weak, so size stays small.
5. **Prepare Monday (S1/S5).** Vol is at its 9–10th percentile, so any weekend or Monday shock of 4σ or more lands in a **calm** state: by S1 that's **continuation**, not a fade. If the weekend stays quiet (\|wk_z\| < 2), Monday's open isn't special.
6. **Carry:** +4–7%/yr is too thin after costs to act on until H-C1 says otherwise.

## 4. What would change my mind

- **S1/S2:** forward results from 2026-11-08 (H-S3: ≥ 30 events per side) against RANDOM_N on the same shock bars. If the sign fails, the shock sleeve goes off.
- **S3:** the options cost study. If bid/ask plus hedging costs exceed about half of the weekend premium, or one weekend jump erases more than 6 months of premium in the backtest, there's no sleeve.
- **S6:** H-X2c forward. A trend Sharpe below 0 over 90 more days means the trend sleeve is retired.
- **Everything:** any registered test, never a single trade outcome.
