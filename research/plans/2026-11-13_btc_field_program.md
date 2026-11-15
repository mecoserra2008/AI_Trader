# BTC field program (2026-11-13): distributions and targets, a polynomial probability field, polynomial-neuron networks with conformal intervals, news, a Bybit-semantics matching engine with generated books, and a dashboard of the state and latent spaces

## The idea, made precise

The user's request has four parts:
1. Translate the knowledge into distributions and targets.
2. Discover the polynomial relationships of co-moving stochastic variables in a high-dimensional state space, treating a fluid-like field as the ultimate prior and updating it.
3. Use delta (signed flow), news, events and the order book.
4. Make conformal predictions from a small network whose neurons carry polynomial complexity, train partly on generated order books from an exact matching engine, and watch the real and latent spaces in a dashboard with a feedback loop.

**The equation.** The precise version of the "Navier-Stokes field" is the **Fokker-Planck / Kolmogorov-Feller equation**. It transports the probability density of the market state X:

  dp/dt = - div( mu(X) p ) + 1/2 sum_ij d2/dxi dxj ( D_ij(X) p ) + jump term with intensity lambda(X) and jump law nu(X)

It has the same structure as Navier-Stokes: a continuity law, advection by a velocity field mu, and diffusion D. We estimate mu, D and lambda from data as Kramers-Moyal conditional moments of the increments, then fit them as **sparse polynomials** of the state (SINDy-style discovery with stability selection). We propagate the density forward in time and update the coefficients online (the "prior updated accordingly").

**What we can realistically trade.** Every direction test so far failed, so H-FLD2 and H-NEWS2 are registered as nulls. The profit target is the **distribution**:
- volatility and range forecasts;
- jump risk;
- calibrated target and stop probabilities for the existing pipeline;
- quote widths for market making.

Execution and any live use stay paper and forward until the user explicitly says go (memory: research only).

**Hypotheses** (registered before any fit): H-FLD1-3, H-PNN1-3, H-NEWS1-2, H-OBG1-2. Their priors are shaded down because our priors were about 2.3x too optimistic.

## State vector for BTC (point-in-time)

| Variable | Source |
|---|---|
| Log vol level (log-HAR terms, robust sigma) | lake bars |
| Hour-of-week and weekday | calendar |
| Signed flow / **delta** (taker buy minus sell) | Binance klines' taker-buy volume for history (in the lake, check columns); flow_3s and trades_1s for recent days |
| Funding level and z | lake |
| OI change | lake, where it exists |
| Recent jump flag and Hawkes intensity | jump_diffusion |
| Regime cell | v0 |
| News / event / on-chain intensity | agent 3 |

## Agents (5; each <= 2 threads and ~1 GB RAM; the machine has ~8 GB free with other apps running)

| Agent | Builds | Hypotheses |
|---|---|---|
| **F1 field** | `ai_trader/field/`: Kramers-Moyal estimation, sparse polynomial fields (stability selection), density propagation (Monte-Carlo of the SDE with jumps, or a 1-D/2-D Fokker-Planck solver where cheap), online Bayesian coefficient updates, **targets** (quantiles and probabilities of hitting levels at 1h/4h/1d), and the plain-language "distribution card" per state | H-FLD1-3 |
| **F2 polynomial network** | `ai_trader/ml/pnn.py`: a Kolmogorov-Arnold-style network (learnable polynomial/spline edges, <= ~2k params), quantile heads, adaptive conformal intervals, a latent bottleneck with linear latent dynamics fed back, symbolic extraction of the learned edge functions | H-PNN1-3 |
| **F3 news / events / on-chain** | `ai_trader/news/`: point-in-time daily and hourly features from GDELT timelines (bitcoin, crypto, blockchain, exchange / regulation / ETF / hack themes), scheduled events (FRED release dates, FOMC, halving, known ETF / regulatory dates), on-chain (blockchain.com charts API). Then **phase 2: the dashboard** | H-NEWS1-2 |
| **F4 matching engine (Rust)** | new crate `crates/at-match`: an exact Bybit-semantics limit order book per `docs/MATCHING_ENGINE.md`, plus `atx match-sim` | engine tests |
| **F5 order-book generator** | `ai_trader/sim/ob_gen.py`: a queue-reactive order-flow generator conditioned on price action and volume, calibrated on real BTC book and trade aggregates (micro_days book_1s / trades_1s, the 2026-11-10 book_100ms, plus fresh ob200 days downloaded one at a time); runs through at-match; builds the half-real / half-generated training sets | H-OBG1-2 |

Dependencies:
- F5 uses F4's engine once its CLI exists, and calibrates and validates in Python before that.
- F2 consumes F1's state builder and, later, F3's news features. It starts with the existing `ai_trader.ml.features` plus BTC delta.
- The dashboard (F3, phase 2) reads F1's field, F2's latent space and conformal bands.

## Storage and compute discipline (30 GB hard budget; about 6 GB used)

- **Raw data** (archives, ob200 zips) is downloaded one day at a time into `data/raw/<task>/`, processed, verified, then deleted.
- **Manifests** (`data/research/<task>/manifest.json`) record what has been processed, so nothing is ever recomputed or re-downloaded.
- **Allocations:**

| Item | Allowance |
|---|---|
| News | <= 1 GB |
| Generated books (aggregated only, not raw event streams) | <= 3 GB |
| Models | <= 1 GB |
| Transient raw | <= 2 GB at any time |

- **Checkpoints:** every long job checkpoints so a machine sleep or usage limit loses at most one step. This happened twice before.
- **Build, test, pilot, then run in full:** each agent builds, writes unit tests, runs a small pilot, checks the numbers, and only then runs the full job.

## Outputs to the decision layer

1. A per-state **distribution card**: quantiles, P(hit level), jump probability, conformal bands.
2. Rules (`research/rules.yaml`) from the recorded results.
3. **Targets** that plug into the execution pipeline's `p_tp` and `net_er`, replacing flat priors.
4. The dashboard.
