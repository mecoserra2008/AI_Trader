# WS12 trained models (hand-off to WS14 INTERP)

Written by WS12 (specification program 2026-11-27). Research only, never read by the desk. Code:
`src/ai_trader/research/ws12_deep.py` (data, networks, training), `ws12_eval.py` (scores), `ws12_meta.py` (LightGBM
meta-labelling). Registry: H-WS12-1..7 (research/hypotheses_pending/WS12.yaml).

## Files

`<arch>_<refit>_lr<lr>_s<seed>.pt` = torch `state_dict`; `.json` next to it = everything needed to rebuild the inputs:
arch, refit dates, lr, seed, epochs, training / validation loss curve, parameter count, `standardiser` (mx/sx for the
per-step or mlp inputs, ms/ss for the static inputs), feature names, window length L, width d, horizons.

- arch: `gru` (1-layer GRU, d 32), `tcn` (4 residual blocks of causal dilated convs, receptive field 61 h), `mlp`
  (2 x Linear 32 on the 19 H-NN8 hourly features).
- refit `R1`: trained on rows with label end <= 2025-06-30 (tested 2025-07-01..2026-06-30); `R2`: label end <= 2026-06-30
  (tested 2026-07-01..2026-11-25, the clean holdout). For each (arch, refit): seed 0 at lr 1e-3 and 3e-3 (the lr with the
  lower inner-validation loss is the chosen one; see `../logs/train_<arch>_<refit>.json`), seeds 1 and 2 at the chosen lr.
- `../preds/<name>.npy`: test-row outputs (n, 3 horizons, 4 = mu, scale, df, logF); `../preds/rows_<refit>.npy` = row
  indices into `ws12_deep.panel()`.

## Rebuilding a model and its inputs

```python
from ai_trader.research import ws12_deep as wd
import json, numpy as np, torch
meta = json.load(open("data/research/ws12/models/gru_R2_lr0.001_s0.json"))
net = wd.make_net(meta["arch"], len(wd.STEP_FEATURES), len(wd.STATIC_FEATURES), 19)
net.load_state_dict(torch.load("data/research/ws12/models/gru_R2_lr0.001_s0.pt")); net.eval()
pn = wd.panel()                       # per-asset caches in data/research/ws12/cache/<SYMBOL>.parquet
st = meta["standardiser"]             # x = clip((raw - mx) / sx, -5, 5); s = clip((static - ms) / ss, -5, 5)
```

Inputs (all point-in-time, decision at the open of 1h bar t; see the ws12_deep docstring for exact definitions):
- sequence (bars t-72..t-1, 13 per step): z (vol-standardised return), lu (log z^2), lrange (Parkinson, base units),
  lsig (log next-bar base variance, made relative to the last step inside the batch: `W[:, :, LSIG] -= W[:, -1:, LSIG]`),
  lturn, fund (bp), doi (24 h log OI change), calendar of the next bar (hour / weekday sin-cos, funding-hour flag), lfacn
  (log hour-of-week factor of the next bar).
- static at t (6): lsig_rel1y, har_d, har_w, har_m, lfac_n4, lfac_n24.
- mlp input: the 19 `ml.hourly.HOURLY_FEATURES` at t.

Outputs per horizon h in (1, 4, 24) hours: Student-t (mu, scale, df) for y_h = r_{t..t+h-1} / sqrt(h s2_t), and logF =
log forecast of v_h = sum r^2 / (h s2_t), s2_t = robust_bc base variance known at the open (`s2` column).

## Suggested questions for WS14

Is the GRU/TCN a scaled HAR (distil logF onto lsig_rel1y, har_d/w/m, lfac_n*)? Which steps of the window matter
(gradient x input by lag)? Does the learned df move with the vol state? Does the 24 h head use anything but the
calendar and har_w?
