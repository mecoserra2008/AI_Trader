"""Cross-sectional network v3 (H-XS1..H-XS3, block XS, 2026-11-15, agent X3): a shared per-token encoder, a
permutation-equivariant mixing layer across the tokens present at the same hour, and per-token quantile heads.
Research tool; torch CPU, 2 threads, deterministic per seed. Design fixed in the diary entry 'XS v3 design frozen'.

Panel (build_panel; universe L, Bybit 1h bars, one row per hour t and token i). The decision for hour t is made at
its open (the close of bar t-1, + the lake's 2 s known_ts lag); every input uses bars t-1 and earlier, funding / OI
prints known by the close of t-1, and the calendar of hour t (known in advance).

  own vol terms   ml.hourly.HOURLY_FEATURES (19: lsig2, lfac, funding-hour flags, lu_1..lu_24, lu_m6, lu_m24, jump24,
                  since_jump, har_d/w/m, dow_sin/cos) + lfac4, lfac24 (log mean hour-of-week factor over the next 4 /
                  24 hours, profile as of t: seasonal_multi) + hod_sin, hod_cos
  own returns     ret{h}_z = (log C_{t-1} - log C_{t-1-h}) / (sqrt(h) s_t), h = 1, 4, 24, 168; s_t = sqrt(s2_t)
  relative        rel{h}_z = (R_i - mean_j R_j) / (sqrt(h) s_t) over the same past window, j = tokens with a finite
                  window return (h = 1, 4, 24, 168)
  betas           beta_btc, beta_ew: rolling OLS slope of r_i on r_BTC / on the equal-weight universe bar return,
                  hourly bars t-720..t-1 (>= 480 pairs)
  funding / OI    fund_bp (latest settled rate x 1e4, clipped +/- 30), fund_z90 (z vs the trailing 2160 hourly
                  values, >= 720), oi4, oi24 (log change of the latest known 4h OI print over 4 / 24 h, clipped +/- 1),
                  oi_ok (1 if both are finite; missing OI -> 0 after standardisation)
  sector          config/sectors.yaml overrides (btc, l1, payments, meme, oracle, exchange, infra, defi)
  market token    mkt_r{1,4,24,168}_z (EW past return / (sqrt(h) mean s)), mkt_lu1, mkt_lsig2 (cross-sectional
                  means), mkt_disp24 (log cross-sectional sd of 24h returns / (sqrt(24) mean s)), n_frac (present / 23)
A token is present at hour t iff every feature except the OI ones is finite and s2 > 0; an hour is used iff >= MIN_TOKENS
tokens are present. Absent tokens are masked everywhere (features zeroed, attention keys removed, losses masked).

Targets (forward, hour t .. t+h-1, from closes: R_i(t, h) = log C_{t+h-1} - log C_{t-1}):
  vol  a_h = |R_i(t, h)| / sqrt(h s2_t), h = 1, 4, 24; quantiles TAUS (0.05 .. 0.95)
  rel  rel_i(t, h) = R_i(t, h) - mean_{j in U} R_j(t, h), U = tokens present at t with a finite R_j(t, h) (>= MIN_TOKENS);
       standardised y = rel / (sqrt(h) s_t); quantiles REL_TAUS (0.10, 0.50, 0.90), h = 4, 24
  pairwise  1{rel_i > rel_j} (= 1{R_i > R_j}); scored from the rel heads (pairwise_prob), no separate head

Network (XSNet, kind 'kan' | 'mlp', mix True | False):
  z_i = Enc(x_i) in R^D (kan: pnn.KANLayer(F -> D); mlp: Linear(F, H1) - ReLU - Linear(H1, D), H1 matched)
  h_i = z_i + E[sector_i];  m = Linear(market features) + E[market]
  a_i = MHA(query h_i; keys / values {h_j : j present} + {m}), HEADS heads (mix=True)
        mix=False (H-XS3 ablation): the key set is {h_i} only (same parameters, no information across tokens)
  out_i = Head([z_i, a_i]) (kan: KANLayer(2D -> 27); mlp: Linear(2D, H2) - ReLU - Linear(H2, 27), H2 matched)
  27 raw outputs = 3 vol heads x 7 (non-crossing in log space, exp) + 2 rel heads x 3 (non-crossing)
Loss: sum over the 5 heads of the mean pinball over valid (hour, token) cells (vol-standardised units).
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass

import numpy as np

from ai_trader.ml import pnn

TAUS = pnn.TAUS
REL_TAUS = (0.10, 0.50, 0.90)
VOL_H = (1, 4, 24)
REL_H = (4, 24)
D = 8
HEADS = 2
MIN_TOKENS = 5
N_OUT = len(VOL_H) * len(TAUS) + len(REL_H) * len(REL_TAUS)
TORCH_THREADS = 2
SECTOR_ORDER = ("btc", "l1", "payments", "meme", "oracle", "exchange", "infra", "defi")
MKT_FEATURES = ("mkt_r1_z", "mkt_r4_z", "mkt_r24_z", "mkt_r168_z", "mkt_lu1", "mkt_lsig2", "mkt_disp24", "n_frac")
EXTRA_FEATURES = ("lfac4", "lfac24", "hod_sin", "hod_cos",
                  "ret1_z", "ret4_z", "ret24_z", "ret168_z", "rel1_z", "rel4_z", "rel24_z", "rel168_z",
                  "beta_btc", "beta_ew", "fund_bp", "fund_z90", "oi4", "oi24", "oi_ok")
OPTIONAL = ("oi4", "oi24")  # may be missing for a present token (filled 0 after standardisation; oi_ok flags it)
BETA_W, BETA_MIN = 720, 480
FUND_W, FUND_MIN = 2160, 720


def feature_names() -> tuple[str, ...]:
    from ai_trader.ml.hourly import HOURLY_FEATURES

    return (*HOURLY_FEATURES, *EXTRA_FEATURES)


def _torch():
    import torch

    torch.set_num_threads(TORCH_THREADS)
    return torch


# ------------------------------------------------------------------ panel maths (pure numpy, testable)


def seasonal_multi(how: np.ndarray, u: np.ndarray, hs=(4, 24), *, kappa: float = 20.0, bandwidth: int = 1, cap: float = 50.0,
                   floor: float = 0.1, min_weeks: int = 8) -> dict[int, np.ndarray]:
    """Like features.seasonality.seasonal_factor (out[t] uses u[j], j < t only), but returns for every horizon h the
    mean of the profile known at t over the hours-of-week of t .. t+h-1 (a regular hourly grid is assumed)."""
    from ai_trader.features.seasonality import HOW, _kernel, _shape

    how = np.asarray(how, dtype=np.int64)
    u = np.asarray(u, dtype=float)
    n = len(u)
    out = {h: np.full(n, np.nan) for h in hs}
    kern = _kernel(bandwidth)
    sums, counts = np.zeros(HOW), np.zeros(HOW)
    need, seen, cache, cum = min_weeks * HOW, 0, None, None
    for t in range(n):
        if seen >= need:
            if cache is None:
                cache = _shape(sums, counts, kappa, kern, floor)
                cum = np.r_[0.0, np.cumsum(np.r_[cache, cache])]  # circular prefix sums over two weeks
            k0 = how[t]
            for h in hs:
                out[h][t] = (cum[k0 + h] - cum[k0]) / h
        x = u[t]
        if np.isfinite(x):
            x = min(max(x, 0.0), cap)
            sums[how[t]] += x
            counts[how[t]] += 1
            seen += 1
            cache = None
    return out


def back_ret(logc: np.ndarray, h: int) -> np.ndarray:
    """(T, N) past window log return known at hour t: log C_{t-1} - log C_{t-1-h} (NaN if an endpoint is missing)."""
    T = logc.shape[0]
    out = np.full_like(logc, np.nan)
    if T > h + 1:
        out[h + 1:] = logc[h:-1] - logc[: T - h - 1]
    return out


def fwd_ret(logc: np.ndarray, h: int) -> np.ndarray:
    """(T, N) forward log return over hours t .. t+h-1: log C_{t+h-1} - log C_{t-1}."""
    T = logc.shape[0]
    out = np.full_like(logc, np.nan)
    if T > h:
        out[1: T - h + 1] = logc[h:] - logc[: T - h]
    return out


def relative(R: np.ndarray, member: np.ndarray, min_n: int = MIN_TOKENS) -> np.ndarray:
    """rel_i = R_i - mean_{j in U} R_j with U = member & finite(R), row by row; NaN outside U or if |U| < min_n."""
    ok = member & np.isfinite(R)
    n = ok.sum(axis=1)
    s = np.where(ok, R, 0.0).sum(axis=1)
    with np.errstate(all="ignore"):
        mu = s / n
    out = np.where(ok, R - mu[:, None], np.nan)
    out[n < min_n] = np.nan
    return out


def rolling_beta(y: np.ndarray, x: np.ndarray, w: int = BETA_W, min_n: int = BETA_MIN) -> np.ndarray:
    """Slope of y on x over bars t-w .. t-1 (finite pairs, >= min_n), known at t. y (T, N), x (T,) or (T, N)."""
    x = np.broadcast_to(x[:, None] if x.ndim == 1 else x, y.shape)
    ok = np.isfinite(y) & np.isfinite(x)
    yy, xx = np.where(ok, y, 0.0), np.where(ok, x, 0.0)

    def win(a):
        c = np.cumsum(np.r_[np.zeros((1, a.shape[1])), a], axis=0)  # c[k] = sum a[:k]
        out = np.full(a.shape, np.nan)
        T = a.shape[0]
        idx = np.arange(T)
        lo = np.maximum(idx - w, 0)
        out[:] = c[idx] - c[lo]  # rows t: sum over [t-w, t-1]
        return out

    n = win(ok.astype(float))
    sx, sy, sxy, sxx = win(xx), win(yy), win(xx * yy), win(xx * xx)
    with np.errstate(all="ignore"):
        cov = sxy - sx * sy / n
        var = sxx - sx * sx / n
        b = cov / var
    b[(n < min_n) | ~np.isfinite(b) | (var <= 0)] = np.nan
    return b


def lagged_rolling_z(v: np.ndarray, w: int = FUND_W, min_n: int = FUND_MIN) -> np.ndarray:
    """z of v[t] vs the trailing window v[t-w .. t-1] (v already lagged to be known at t), per column."""
    import polars as pl

    out = np.full(v.shape, np.nan)
    for j in range(v.shape[1]):
        s = pl.Series(v[:, j]).fill_nan(None)
        prev = s.shift(1)
        mu = prev.rolling_mean(w, min_samples=min_n).to_numpy().astype(float)
        sd = prev.rolling_std(w, min_samples=min_n).to_numpy().astype(float)
        with np.errstate(all="ignore"):
            out[:, j] = (v[:, j] - mu) / np.where(sd > 1e-12, sd, np.nan)
    return out


def pairwise_prob(q_rel: np.ndarray) -> np.ndarray:
    """P(token i beats token j) from rel quantiles (..., N, 3) = (q10, q50, q90) in raw return units: normal
    approximation with median m and sd s = (q90 - q10) / (2 z_0.9); P = Phi((m_i - m_j) / sqrt(s_i^2 + s_j^2)).
    Returns (..., N, N); antisymmetric around 1/2 and 1/2 on the diagonal."""
    from scipy.special import ndtr

    z90 = 1.2815515655446004
    m = q_rel[..., 1]
    s = np.maximum((q_rel[..., 2] - q_rel[..., 0]) / (2 * z90), 1e-12)
    dm = m[..., :, None] - m[..., None, :]
    ss = np.sqrt(s[..., :, None] ** 2 + s[..., None, :] ** 2)
    return ndtr(dm / ss)


def spearman_rows(a: np.ndarray, b: np.ndarray, mask: np.ndarray, min_n: int = MIN_TOKENS) -> np.ndarray:
    """Cross-sectional Spearman correlation per row over masked entries (average ranks for ties)."""
    from scipy.stats import rankdata

    out = np.full(a.shape[0], np.nan)
    for t in range(a.shape[0]):
        k = mask[t] & np.isfinite(a[t]) & np.isfinite(b[t])
        if k.sum() < min_n:
            continue
        ra, rb = rankdata(a[t, k]), rankdata(b[t, k])
        if ra.std() == 0 or rb.std() == 0:
            continue
        out[t] = float(np.corrcoef(ra, rb)[0, 1])
    return out


def rank_contrib(a: np.ndarray, b: np.ndarray, mask: np.ndarray, min_n: int = MIN_TOKENS) -> np.ndarray:
    """Per-cell contribution c_ti = za_ti zb_ti (z of ranks, population sd) whose row mean is the Spearman IC."""
    from scipy.stats import rankdata

    out = np.full(a.shape, np.nan)
    for t in range(a.shape[0]):
        k = mask[t] & np.isfinite(a[t]) & np.isfinite(b[t])
        if k.sum() < min_n:
            continue
        ra, rb = rankdata(a[t, k]), rankdata(b[t, k])
        if ra.std() == 0 or rb.std() == 0:
            continue
        out[t, k] = ((ra - ra.mean()) / ra.std()) * ((rb - rb.mean()) / rb.std())
    return out


# ------------------------------------------------------------------ network


def _build():
    torch = _torch()
    nn = torch.nn
    F = torch.nn.functional
    KANLayer = pnn.classes()[0]

    class MLP2(nn.Module):
        def __init__(self, n_in: int, hidden: int, n_out: int) -> None:
            super().__init__()
            self.l1, self.l2 = nn.Linear(n_in, hidden), nn.Linear(hidden, n_out)
            self.bias = self.l2.bias  # head bias (init_bias)

        def forward(self, x):
            return self.l2(torch.relu(self.l1(x)))

    class XSNet(nn.Module):
        def __init__(self, n_in: int, n_mkt: int, n_sector: int, kind: str = "kan", mix: bool = True, d: int = D, heads: int = HEADS,
                     h1: int = 0, h2: int = 0) -> None:
            super().__init__()
            self.cfg = {"n_in": n_in, "n_mkt": n_mkt, "n_sector": n_sector, "kind": kind, "mix": mix, "d": d, "heads": heads, "h1": h1, "h2": h2}
            self.kind, self.mix, self.d, self.heads = kind, mix, d, heads
            if kind == "kan":
                self.enc = KANLayer(n_in, d)
                self.head = KANLayer(2 * d, N_OUT)
            else:
                self.enc = MLP2(n_in, h1, d)
                self.head = MLP2(2 * d, h2, N_OUT)
            self.sector = nn.Embedding(n_sector + 1, d)  # last row = the market token
            with torch.no_grad():
                self.sector.weight.mul_(0.1)
            self.mkt = nn.Linear(n_mkt, d)
            self.qkv = nn.Linear(d, 3 * d)
            self.o = nn.Linear(d, d)

        def encode(self, X, sec):
            B, N, Fn = X.shape
            z = self.enc(X.reshape(B * N, Fn)).reshape(B, N, self.d)
            return z, z + self.sector(sec)[None, :, :]

        def attend(self, h, M, present):
            """h (B, N, d), M (B, n_mkt), present (B, N) bool -> a (B, N, d), w (B, heads, N, N + 1)."""
            B, N, d = h.shape
            nh, dh = self.heads, d // self.heads
            m = self.mkt(M) + self.sector.weight[-1][None, :]
            keys = torch.cat([h, m[:, None, :]], dim=1)  # (B, N+1, d)
            q = self.qkv(h)[..., :d]
            kv = self.qkv(keys)
            k, v = kv[..., d:2 * d], kv[..., 2 * d:]
            q = q.reshape(B, N, nh, dh).transpose(1, 2)
            k = k.reshape(B, N + 1, nh, dh).transpose(1, 2)
            v = v.reshape(B, N + 1, nh, dh).transpose(1, 2)
            s = q @ k.transpose(-1, -2) / math.sqrt(dh)  # (B, nh, N, N+1)
            if self.mix:
                allow = torch.cat([present, torch.ones(B, 1, dtype=torch.bool)], dim=1)[:, None, None, :].expand(B, nh, N, N + 1)
            else:
                eye = torch.eye(N, dtype=torch.bool)
                allow = torch.cat([eye, torch.zeros(N, 1, dtype=torch.bool)], dim=1)[None, None].expand(B, nh, N, N + 1)
            s = s.masked_fill(~allow, float("-inf"))
            w = torch.softmax(s, dim=-1)
            a = (w @ v).transpose(1, 2).reshape(B, N, d)
            return self.o(a), w

        def forward(self, X, M, present, sec, return_attn: bool = False):
            z, h = self.encode(X, sec)
            a, w = self.attend(h, M, present)
            B, N, _ = z.shape
            o = self.head(torch.cat([z, a], dim=-1).reshape(B * N, 2 * self.d)).reshape(B, N, N_OUT)
            vol, rel = split_raw(o)
            return (vol, rel, w) if return_attn else (vol, rel)

    def split_raw(o):
        """o (B, N, 27) -> vol quantiles (B, N, 3, 7) > 0 and rel quantiles (B, N, 2, 3)."""
        sp = F.softplus
        nv = len(VOL_H) * len(TAUS)
        ov = o[..., :nv].reshape(*o.shape[:-1], len(VOL_H), len(TAUS))
        q50 = ov[..., 3]
        q75 = q50 + sp(ov[..., 4])
        q90 = q75 + sp(ov[..., 5])
        q95 = q90 + sp(ov[..., 6])
        q25 = q50 - sp(ov[..., 2])
        q10 = q25 - sp(ov[..., 1])
        q05 = q10 - sp(ov[..., 0])
        vol = torch.exp(torch.stack([q05, q10, q25, q50, q75, q90, q95], dim=-1).clamp(-12, 6))
        orl = o[..., nv:].reshape(*o.shape[:-1], len(REL_H), len(REL_TAUS))
        m = orl[..., 1]
        rel = torch.stack([m - sp(orl[..., 0]), m, m + sp(orl[..., 2])], dim=-1)
        return vol, rel

    return XSNet, split_raw


_C = None


def classes():
    global _C
    if _C is None:
        _C = _build()
    return _C


def mlp_widths(n_in: int, d: int = D) -> tuple[int, int]:
    """Hidden widths whose MLP layers match the KAN encoder / head parameter counts."""
    kan_enc = n_in * d * (pnn.DEGREE + 1) + d
    kan_head = 2 * d * N_OUT * (pnn.DEGREE + 1) + N_OUT
    h1 = min(range(1, 400), key=lambda h: abs((n_in + 1) * h + (h + 1) * d - kan_enc))
    h2 = min(range(1, 400), key=lambda h: abs((2 * d + 1) * h + (h + 1) * N_OUT - kan_head))
    return h1, h2


def make(n_in: int, n_mkt: int, n_sector: int, kind: str = "kan", mix: bool = True):
    h1, h2 = mlp_widths(n_in) if kind == "mlp" else (0, 0)
    return classes()[0](n_in, n_mkt, n_sector, kind=kind, mix=mix, h1=h1, h2=h2)


def n_params(model) -> int:
    return int(sum(p.numel() for p in model.parameters()))


def init_bias(model, a_by_h: list[np.ndarray], y_rel_by_h: list[np.ndarray]) -> None:
    """Head bias = unconditional quantiles (vol: log scale; rel: standardised), so training starts from them."""
    torch = _torch()
    inv = lambda v: float(np.log(np.expm1(max(v, 1e-4))))
    b = np.zeros(N_OUT)
    for k, a in enumerate(a_by_h):
        qs = np.log(np.maximum(np.quantile(a[np.isfinite(a)], TAUS), 1e-4))
        dq = np.maximum(np.diff(qs), 1e-3)
        o = k * len(TAUS)
        b[o + 3] = qs[3]
        b[o + 4], b[o + 5], b[o + 6] = inv(dq[3]), inv(dq[4]), inv(dq[5])
        b[o + 2], b[o + 1], b[o + 0] = inv(dq[2]), inv(dq[1]), inv(dq[0])
    nv = len(VOL_H) * len(TAUS)
    for k, y in enumerate(y_rel_by_h):
        q = np.quantile(y[np.isfinite(y)], REL_TAUS)
        o = nv + k * 3
        b[o + 1] = q[1]
        b[o + 0], b[o + 2] = inv(q[1] - q[0]), inv(q[2] - q[1])
    with torch.no_grad():
        model.head.bias.copy_(torch.as_tensor(b, dtype=model.head.bias.dtype))


def pinball_t(y, q, taus, mask):
    """Mean pinball over valid cells. y (..., ), q (..., K), mask (...) bool."""
    torch = _torch()
    t = torch.as_tensor(np.asarray(taus, dtype=np.float32))
    e = torch.where(mask[..., None], y[..., None] - q, torch.zeros_like(q))
    lo = torch.maximum(t * e, (t - 1) * e).mean(-1)
    n = mask.sum().clamp(min=1)
    return (lo * mask).sum() / n


def loss_fn(vol, rel, Yv, Mv, Yr, Mr):
    """vol (B, N, 3, 7), rel (B, N, 2, 3); Yv (B, N, 3), Yr (B, N, 2) with masks."""
    tot = 0.0
    for k in range(len(VOL_H)):
        tot = tot + pinball_t(Yv[..., k], vol[..., k, :], TAUS, Mv[..., k])
    for k in range(len(REL_H)):
        tot = tot + pinball_t(Yr[..., k], rel[..., k, :], REL_TAUS, Mr[..., k])
    return tot


@dataclass
class Batchable:
    """Tensors for one block of hours: Z (T, N, F) standardised & clipped (absent = 0), M (T, n_mkt), P (T, N) present,
    Yv / Mv (T, N, 3), Yr / Mr (T, N, 2)."""
    Z: np.ndarray
    M: np.ndarray
    P: np.ndarray
    Yv: np.ndarray
    Mv: np.ndarray
    Yr: np.ndarray
    Mr: np.ndarray

    def tensors(self, idx=None):
        torch = _torch()
        s = slice(None) if idx is None else idx
        f = lambda a, dt=torch.float32: torch.as_tensor(np.ascontiguousarray(a[s]), dtype=dt)
        return (f(self.Z), f(self.M), f(self.P, torch.bool), f(np.nan_to_num(self.Yv)), f(self.Mv, torch.bool),
                f(np.nan_to_num(self.Yr)), f(self.Mr, torch.bool))


def train(model, fit: Batchable, val: Batchable, sec: np.ndarray, *, seed: int, lr: float, wd: float, batch: int = 64,
          epoch_hours: int = 8192, max_epochs: int = 60, patience: int = 6, val_max: int = 4000) -> dict:
    """AdamW on the summed pinball; an epoch = `epoch_hours` hours drawn without replacement; early stopping on the
    validation block (evenly thinned to <= val_max hours); best iterate kept. Deterministic given the seed."""
    torch = _torch()
    torch.manual_seed(seed)
    g = np.random.default_rng(seed)
    S = torch.as_tensor(sec, dtype=torch.long)
    vi = np.linspace(0, len(val.P) - 1, min(val_max, len(val.P))).round().astype(int)
    V = val.tensors(vi)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)

    def vloss() -> float:
        model.eval()
        with torch.no_grad():
            tot = 0.0
            for i in range(0, len(vi), 512):
                sl = slice(i, i + 512)
                vol, rel = model(V[0][sl], V[1][sl], V[2][sl], S)
                tot += float(loss_fn(vol, rel, V[3][sl], V[4][sl], V[5][sl], V[6][sl])) * (min(i + 512, len(vi)) - i)
            return tot / len(vi)

    best, best_state, best_ep, bad, hist = vloss(), copy.deepcopy(model.state_dict()), 0, 0, []
    n = len(fit.P)
    for ep in range(1, max_epochs + 1):
        model.train()
        pick = g.choice(n, size=min(epoch_hours, n), replace=False)
        for i in range(0, len(pick), batch):
            b = np.sort(pick[i:i + batch])
            Z, M, P, Yv, Mv, Yr, Mr = fit.tensors(b)
            opt.zero_grad()
            vol, rel = model(Z, M, P, S)
            loss = loss_fn(vol, rel, Yv, Mv, Yr, Mr)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
        v = vloss()
        hist.append(v)
        if v < best - 1e-6:
            best, best_state, best_ep, bad = v, copy.deepcopy(model.state_dict()), ep, 0
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    model.eval()
    return {"best_val": best, "epoch": best_ep, "epochs_run": len(hist), "hist": hist}


def predict(model, B: Batchable, sec: np.ndarray, chunk: int = 512, attn: bool = False):
    """vol (T, N, 3, 7), rel (T, N, 2, 3) in standardised units; optionally the attention weights averaged over heads
    (T, N, N + 1) as float16."""
    torch = _torch()
    S = torch.as_tensor(sec, dtype=torch.long)
    model.eval()
    vs, rs, ws = [], [], []
    with torch.no_grad():
        for i in range(0, len(B.P), chunk):
            sl = slice(i, i + chunk)
            Z, M, P = (torch.as_tensor(np.ascontiguousarray(a[sl]), dtype=dt) for a, dt in
                       ((B.Z, torch.float32), (B.M, torch.float32), (B.P, torch.bool)))
            vol, rel, w = model(Z, M, P, S, return_attn=True)
            vs.append(vol.numpy())
            rs.append(rel.numpy())
            if attn:
                ws.append(w.mean(1).numpy().astype(np.float16))
    out = (np.concatenate(vs).astype(float), np.concatenate(rs).astype(float))
    return (*out, np.concatenate(ws)) if attn else out


__all__ = [
    "BETA_MIN",
    "BETA_W",
    "EXTRA_FEATURES",
    "HEADS",
    "MIN_TOKENS",
    "MKT_FEATURES",
    "N_OUT",
    "OPTIONAL",
    "REL_H",
    "REL_TAUS",
    "SECTOR_ORDER",
    "TAUS",
    "VOL_H",
    "Batchable",
    "D",
    "back_ret",
    "classes",
    "feature_names",
    "fwd_ret",
    "init_bias",
    "lagged_rolling_z",
    "loss_fn",
    "make",
    "mlp_widths",
    "n_params",
    "pairwise_prob",
    "pinball_t",
    "predict",
    "rank_contrib",
    "relative",
    "rolling_beta",
    "seasonal_multi",
    "spearman_rows",
    "train",
]
