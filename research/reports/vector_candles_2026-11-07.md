# Vector candles: H-VC1, H-VC2, H-VC3, H-VC5 (2026-11-07)

**Verdicts recorded** (`research/hypotheses_results.jsonl`, mechanical rules taken from the registry):

| hypothesis | llm_in_sample | clean_holdout | pooled |
|---|---|---|---|
| H-VC1 continuation (signed z at h=24) | **contradicts** | supports (sign only, t = 0.67) | – |
| H-VC2 zone-mid revisit vs reflection null | **contradicts** (+3.0pp, t = 7.0, below the 5pp bar) | supports (+4.2pp, t = 2.9) | – |
| H-VC3 near-edge rejection > 0.5 | **contradicts** (0.484, z = −3.9) | **contradicts** (0.454, z = −3.2) | – |
| H-VC5 aligned-flow minus unaligned | contradicts | contradicts | **contradicts** (diff −0.05, t = −0.15) |

A holdout "supports" means only that the registered holdout condition (correct sign) was met. Neither H-VC1
nor H-VC2 passed its in-sample bar, so neither is validated.

## Definition (exactly as registered; `src/ai_trader/features/candles.py`)
- 1h bars from `market_frame(symbol, '1h', now, flow=True)`. Price-sanity filter and listing burn-in come from the frame.
- `range_sig = ln(high/low) / sigma_prev`, where sigma_prev is the EWMA(0.94) forecast for bar t made before bar t.
- `vol_rel = turnover / median(turnover of the 168 PRIOR bars)`. The current bar is excluded, and the first 168 bars give null.
- `body = |close − open| / (high − low)`.
- **Vector**: `range_sig ≥ 3` AND `vol_rel ≥ 2` AND `body ≥ 0.6`. `d = sign(close − open)`. Zone = body `[min(o,c), max(o,c)]`, and `zone_mid` is its midpoint.
- `thin`: `vol_rel / range_sig ≤` the 20% quantile (linear interpolation) of its own prior-168-bar values.
- Declustering (causal): an event is kept only if it comes more than 24 bars after the previous *kept* event on the same symbol.
- Every feature, event and forward outcome is computed on each symbol's full history. The panel is then split by the event bar's `event_ts` at the knowledge cutoff (2026-07-01).
- Unit tests in `tests/test_candles.py` check that a sweep bar is flagged, that the current bar is excluded from the median, the thin quantile, truncation invariance (causality) and the decluster rule.

**Statistics** (`src/ai_trader/research/vector_study.py`): events with the same `event_ts` are first averaged across symbols, giving one observation per timestamp. The t-stat is then Newey–West over that time-ordered series. Rates against a fixed null use a binomial z on raw events, and the cluster t is reported alongside. For H-VC5, the test is a Welch t on timestamp-clustered values, aligned vs unaligned.

**Order of work:** all in-sample numbers were computed and saved before the holdout was printed. No parameter was changed after that.

## Event counts (declustered)
In-sample: **15,333** events (6,101 distinct timestamps), from 2020-04-15 onward. Holdout: **1,203** events (490 timestamps), up to 2026-11-05.
Up-vector share: 45.9% / 48.3%. Thin share: 5.0% (765) / 5.7% (69).

Counts are given as in-sample / holdout:
AAVE 698/44 · ADA 745/56 · ATOM 606/47 · AVAX 680/57 · BCH 796/54 · BNB 690/52 · BTC 1015/61 · DOGE 701/54 · DOT 724/50 · ENA 307/51 · ETC 707/47 · ETH 857/60 · FIL 683/47 · HYPE 181/50 · LINK 771/47 · LTC 832/49 · NEAR 617/51 · SOL 670/58 · SUI 437/53 · TRX 638/53 · UNI 640/56 · XLM 627/51 · XRP 711/55

## H-VC1: continuation, signed z = d·ln(c_{t+h}/c_t)/(σ_prev·√h)

| segment | h | mean signed z | NW t (clustered) | n events / clusters | P(z24 > 0) |
|---|---|---|---|---|---|
| in-sample | 4 | +0.042 | 1.99 | 15,333 / 6,101 | |
| in-sample | 24 | **−0.005** | −0.27 | 15,333 / 6,101 | 0.475 |
| holdout | 4 | −0.021 | −0.32 | 1,203 / 490 | |
| holdout | 24 | +0.058 | 0.67 | 1,203 / 490 | 0.472 |

There is no continuation at 24 bars. A small 4-bar drift appears in-sample (t ≈ 2), but the holdout does not show it.

## H-VC2: revisit of zone_mid within 72 bars
A revisit means zone_mid lies inside [min low, max high] of bars t+1 … t+72, strictly after the event bar.
The null is `gap_fill_baseline(d, 72)` with `d = |ln(close/zone_mid)|/σ_prev`, and excess is measured per event.

| segment | observed | baseline | excess | t (clustered) | n | median d (σ) |
|---|---|---|---|---|---|---|
| in-sample | 84.9% | 83.2% | **+3.0pp** | 7.0 | 15,333 | 1.58 |
| holdout | 88.7% | 82.9% | +4.2pp | 2.9 | 1,180 | 1.60 |

The excess is real and consistent in both segments, but it is below the registered 5pp threshold, so H-VC2 fails in-sample. The baseline is already about 83%, because the midpoint sits only about 1.6σ from the close. Saying "zones get filled" mostly restates the random walk.

## H-VC3: first touch of the near edge, then a ±1σ race on 1h closes
| segment | reject rate | resolved | binom z | cluster t | unresolved | no touch | touch at bar t+1 |
|---|---|---|---|---|---|---|---|
| in-sample | **0.484** | 15,333 | −3.85 | −4.31 | 0 | 0 | 100% |
| holdout | **0.454** | 1,203 | −3.20 | −2.53 | 0 | 0 | 100% |

**Design caveat.** The near edge is the vector's own close, so the next bar touches it in 100% of events. The registered test therefore reduces to "is the first ±1σ close move from the vector close in the candle's direction?" The answer is no: breaks (moves against the candle) win 52–55% of the time, and this is significant in both segments. The vector's close is not support or resistance. If anything it gets faded slightly, at the 1σ scale. The race starts with the touch bar's own close, and σ = σ_prev at the touch bar. No events were dropped as unresolved.

## H-VC5: taker-flow alignment (BTC/ETH, flow bars about 180 days)
An event is aligned when `sign(buy − sell) = d` and `|imbalance| ≥ 0.2`. The outcome is signed z at h = 24.

| segment | n aligned / unaligned | mean aligned | mean unaligned | diff | Welch t |
|---|---|---|---|---|---|
| in-sample | 12 / 32 | +0.66 | +0.30 | +0.36 | 0.65 |
| holdout | 32 / 89 | +0.01 | +0.23 | −0.22 | −0.57 |
| **pooled** | 44 / 121 | +0.20 | +0.25 | **−0.05** | **−0.15** |

Taker-flow alignment does not separate good vectors from bad ones. Only about 27% of vectors are aligned at all.

## Exploratory splits (descriptive only: NOT tested, NOT recorded)
H-VC1 signed z, NW t clustered by timestamp:

| split | in-sample h4 | in-sample h24 | holdout h4 | holdout h24 |
|---|---|---|---|---|
| vol_state = low (n 7,776 / 552) | +0.095 (t 3.0) | +0.026 (0.9) | −0.036 (−0.4) | +0.083 (0.7) |
| vol_state = normal (4,320 / 398) | +0.022 (0.7) | −0.010 (−0.3) | +0.023 (0.2) | +0.001 (0.0) |
| vol_state = high (2,575 / 253) | +0.004 (0.1) | −0.023 (−0.8) | −0.033 (−0.4) | +0.032 (0.3) |
| thin = False (14,513 / 1,134) | +0.055 (2.6) | +0.002 (0.1) | +0.015 (0.2) | +0.097 (1.1) |
| **thin = True** (765 / 69) | −0.055 (−0.8) | −0.089 (−1.4) | **−0.515 (−2.6)** | **−0.454 (−2.2)** |

There are 662 in-sample events with a null vol_state (labeller warm-up).
- Most vectors (about 51%) fire in the **low** vol state, because `range_sig ≥ 3` is easiest to reach when σ_prev is small. The only in-sample short-horizon continuation is there (h4 t = 3.0), and it is not in the holdout.
- **Thin vectors** (large range on comparatively little turnover, an "air pocket") show negative signed returns, which means reversal. The sign is the same in both segments, and the holdout effect is large but rests on only 69 events. I have now looked at both segments, so this cannot be confirmed on this data.

## Conclusion: what is exploitable
- **No continuation entry rule.** Vector candles on 1h do not continue at 24 bars (H-VC1), and taker-flow alignment does not rescue it (H-VC5). Do not buy or sell a vector close on the expectation of follow-through.
- **The vector close is not a level** (H-VC3). For the level chain (ARCHITECTURE §3.11), the vector's close-side edge should get a break probability π ≈ 0.52–0.55 on the first return, meaning slightly *more* likely to break than a neutral level. It must not be placed as a support or resistance level for an SL.
- **zone_mid as a TP magnet: weak.** Revisit probability within 72 bars is about 3–4pp above the reflection null, out of a base of about 83%. This could enter `pred.barrier` as a small drift-toward-mid adjustment, but it is not large enough to carry a strategy after costs on its own. It failed its registered threshold.
- Net: the vector-candle lore does not hold up as a source of edge on 1h crypto perps. The one interesting lead is the thin-vector reversal below, which needs fresh data.

## Candidate hypotheses (proposed, NOT tested; both segments have been seen)
- **H-VC6 (candidate), thin vectors revert.** Statement: among declustered 1h vector candles with `thin = True`, the mean signed z at h = 24 is < 0. Test: the same event study and clustering, run on events after 2026-11-07 (forward data) and, independently, on the 4h universe with the same definition (4h has not been looked at). Success: mean < 0 with t < −2 on 4h full history, and the same sign on forward 1h events (n ≥ 30).
- **H-VC7 (candidate), the first ±1σ move after a vector fades it.** Statement: from a vector's close, the first ±1σ_prev 1h-close excursion goes against d with probability > 0.5. Test: the H-VC3 race code (as implemented) on 4h bars and on forward 1h data. Success: rate > 0.52 with binomial z > 2. Note that the expected effect is small (2–5pp) and may not survive costs.
