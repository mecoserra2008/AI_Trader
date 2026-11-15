# A Data-driven Market Simulator for Small Data Environments (Buehler, Horvath, Lyons, Perez Arribas & Wood, 2020)

- **Citation:** Hans Buehler, Blanka Horvath, Terry Lyons, Imanol Perez Arribas, Ben Wood (2020). arXiv [2006.14498](https://arxiv.org/abs/2006.14498). Working paper (no journal reference on arXiv; the SSRN version was not checked).
- **Verified:** arXiv API (title, authors, abstract), 2026-11-10. Details marked (m) come from memory.
- **Topic:** 4. Market generators for small samples (signatures plus a VAE). Hypotheses: H-SYN1 / H-SYN2 (generator choice); relevant to the TradFi perps' short history.

## Question
How do we build a generative market simulator that works when training data are scarce (e.g. a few years of monthly paths), and how should we evaluate it?

## Data and method
- Overview of generative approaches (GANs, VAEs) and of evaluation metrics for financial time series, contrasted with classical simulation (abstract).
- Proposed model: represent each path segment (e.g. 1-month windows of an index (m)) by its truncated **path signature** (iterated integrals, a rough-paths feature that summarises the path's shape). Train a small **conditional VAE** in signature space, conditioned on the previous segment's signature or on market state. Invert signatures to paths.
- Evaluation: a signature-based metric (maximum mean discrepancy on expected signatures), which compares the laws of path segments rather than marginal moments.

## Key results
- A parsimonious VAE works reliably with little training data (abstract). The signature MMD is proposed as a suitable performance metric (abstract). The use case is deep hedging (abstract).

## Limitations / transfer to Bybit perps
- The demonstrations are on small examples; no strategy-selection out-of-sample evidence.
- Signature methods need a library (`esig` / `iisignature` / `signatory`) and care with path scaling.

## How we use it here
- **Evaluation idea worth adopting:** compare generated vs real **paths** (weekly segments of 1h returns), not only marginal moments, using an MMD on simple path features. Signatures truncated at level 2-3 equal increments, signed areas (lead-lag) and quadratic variation, which is cheap to compute. It is a richer B5 generator check than kurtosis plus ACF alone.
- Conditional generation on the previous week's state is the same idea as the plan's regime-switching resampling, so the plan already follows its spirit.

## Evidence quality
**Weak-medium** (credible authors, an unrefereed working paper, no OOS selection evidence).
