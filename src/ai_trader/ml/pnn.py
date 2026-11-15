"""Polynomial-neuron (Kolmogorov-Arnold style) quantile network, an equal-parameter MLP, adaptive conformal inference
and linear latent dynamics (H-PNN1..H-PNN3, BTC field program 2026-11-13). Research tool; CPU, deterministic per seed.

Network (PNN):
  x (standardised, clipped to +/- CLIP) -> KANLayer(n_in -> k)  = latent z (the bottleneck, k = LATENT = 4)
                                        -> KANLayer(k [+ k] -> 7) = raw head o -> non-crossing quantiles
  KANLayer (Liu et al. 2024, Chebyshev variant): every edge i -> j carries its own univariate function
      phi_ji(x) = w_ji silu(x) + sum_{d=1..D} c_jid T_d(tanh(S x)),     S = 0.5, D = DEGREE = 4
  and node j outputs b_j + sum_i phi_ji(x_i). T_d are Chebyshev polynomials of the first kind (recursion
  T_{d+1} = 2u T_d - T_{d-1}); T_0 is absorbed into the node bias, so an edge has D + 1 parameters.
  The residual SiLU is the KAN paper's base function b(x).

Quantile head (both families): o = (o_0..o_6) for taus (0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95):
  q_50 = o_3; q_75 = q_50 + sp(o_4), q_90 = q_75 + sp(o_5), q_95 = q_90 + sp(o_6);
  q_25 = q_50 - sp(o_2), q_10 = q_25 - sp(o_1), q_05 = q_10 - sp(o_0); sp = softplus. Never crosses.
  The head bias starts at the training sample's unconditional quantiles.

Feedback (H-PNN3): z_{t+1} ~ A z_t + B u_t + c (latent standardised with fit-block mean / sd; u_t = calendar of the
next step, a pure function of timestamps), fitted by ridge least squares on the fit block's consecutive pairs; the
head of the feedback model takes [z_t, zhat_{t+1}] (zhat mapped back to the raw latent scale). Extra head edges start
at zero, so feedback and control models start from the same function.

Equal-parameter MLP: x -> Linear(n_in, H) -> ReLU -> Linear(H, k) -> ReLU -> Linear(k, 7) -> the same head, with H
chosen to match the PNN parameter count (mlp_hidden_for).

Training (both families, identical): AdamW, pinball loss averaged over the 7 taus, early stopping on a validation
block (patience PATIENCE, max MAX_EPOCHS), best iterate kept; torch threads TORCH_THREADS.

Conformal: CQR scores s = max(lo - y, y - hi) (Romano et al. 2019); split conformal = the finite-sample
ceil((n+1)(1-a))/n quantile of a calibration block; adaptive conformal inference (Gibbs and Candes 2021):
alpha_{t+1} = alpha_t + gamma (alpha - err_t), interval at t uses the 1 - alpha_t quantile of the last W realised
scores (alpha_t <= 0 -> the window max; >= 1 -> zero-width adjustment).

Symbolic extraction: each learned edge function is evaluated on the empirical distribution of its input and fitted by
least squares with polynomials of degree 1..3 in the (standardised) input; reports coefficients and R^2.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass

import numpy as np

TAUS = (0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95)
NQ = len(TAUS)
DEGREE = 4
LATENT = 4
SCALE = 0.5
CLIP = 6.0
PARAM_CAP = 2000
TORCH_THREADS = 2
MAX_EPOCHS = 200
PATIENCE = 10
DEFAULT_CFG = {"lr": 3e-3, "weight_decay": 1e-4, "batch": 1024, "max_epochs": MAX_EPOCHS, "patience": PATIENCE}
GRID = [{"lr": lr, "weight_decay": wd} for lr in (1e-3, 3e-3) for wd in (1e-4, 1e-2)]


def _torch():
    import torch

    torch.set_num_threads(TORCH_THREADS)
    return torch


# ------------------------------------------------------------------ layers and models


def _build_classes():
    torch = _torch()
    nn = torch.nn
    F = torch.nn.functional

    class KANLayer(nn.Module):
        def __init__(self, n_in: int, n_out: int, degree: int = DEGREE, scale: float = SCALE) -> None:
            super().__init__()
            self.n_in, self.n_out, self.degree, self.scale = n_in, n_out, degree, scale
            std = 0.3 / math.sqrt(max(n_in, 1))
            self.coef = nn.Parameter(torch.randn(n_out, n_in, degree) * std)
            self.w_res = nn.Parameter(torch.randn(n_out, n_in) * std)
            self.bias = nn.Parameter(torch.zeros(n_out))

        def basis(self, x):
            u = torch.tanh(self.scale * x)
            ts = [u]
            if self.degree >= 2:
                ts.append(2 * u * u - 1)
            for _ in range(2, self.degree):
                ts.append(2 * u * ts[-1] - ts[-2])
            return torch.stack(ts, dim=-1)  # (N, n_in, D)

        def edges(self, x):
            """(N, n_out, n_in): phi_ji(x_i) without the node bias."""
            B = self.basis(x)
            return torch.einsum("nid,oid->noi", B, self.coef) + F.silu(x)[:, None, :] * self.w_res[None]

        def forward(self, x):
            return torch.einsum("nid,oid->no", self.basis(x), self.coef) + F.silu(x) @ self.w_res.T + self.bias

    def quantiles_from_raw(o):
        sp = F.softplus
        q50 = o[:, 3]
        q75 = q50 + sp(o[:, 4])
        q90 = q75 + sp(o[:, 5])
        q95 = q90 + sp(o[:, 6])
        q25 = q50 - sp(o[:, 2])
        q10 = q25 - sp(o[:, 1])
        q05 = q10 - sp(o[:, 0])
        return torch.stack([q05, q10, q25, q50, q75, q90, q95], dim=1)

    class PNN(nn.Module):
        def __init__(self, n_in: int, latent: int = LATENT, degree: int = DEGREE, fb: bool = False, u_dim: int = 0) -> None:
            super().__init__()
            self.n_in, self.latent, self.degree, self.fb, self.u_dim = n_in, latent, degree, fb, u_dim
            self.enc = KANLayer(n_in, latent, degree)
            self.head = KANLayer(latent * (2 if fb else 1), NQ, degree)
            if fb:
                self.register_buffer("A", torch.zeros(latent, latent))
                self.register_buffer("B", torch.zeros(latent, max(u_dim, 1)))
                self.register_buffer("c", torch.zeros(latent))
                self.register_buffer("zmu", torch.zeros(latent))
                self.register_buffer("zsd", torch.ones(latent))

        def latent_of(self, x):
            return self.enc(x)

        def zhat(self, z, u):
            zs = (z - self.zmu) / self.zsd
            nxt = zs @ self.A.T + self.c
            if self.u_dim:
                nxt = nxt + u @ self.B[:, : self.u_dim].T
            return nxt * self.zsd + self.zmu

        def head_in(self, x, u=None):
            z = self.enc(x)
            return torch.cat([z, self.zhat(z, u)], dim=1) if self.fb else z

        def forward(self, x, u=None):
            return quantiles_from_raw(self.head(self.head_in(x, u)))

    class QMLP(nn.Module):
        def __init__(self, n_in: int, hidden: int, latent: int = LATENT) -> None:
            super().__init__()
            self.n_in, self.hidden, self.latent = n_in, hidden, latent
            self.l1 = nn.Linear(n_in, hidden)
            self.l2 = nn.Linear(hidden, latent)
            self.head = nn.Linear(latent, NQ)
            with torch.no_grad():  # start near the unconditional quantiles, like the PNN head
                self.head.weight.normal_(0.0, 0.3 / math.sqrt(latent) * 0.3)

        def latent_of(self, x):
            return torch.relu(self.l2(torch.relu(self.l1(x))))

        def forward(self, x, u=None):
            return quantiles_from_raw(self.head(self.latent_of(x)))

    return KANLayer, PNN, QMLP, quantiles_from_raw


_CLASSES = None


def classes():
    global _CLASSES
    if _CLASSES is None:
        _CLASSES = _build_classes()
    return _CLASSES


def make_pnn(n_in: int, latent: int = LATENT, degree: int = DEGREE, fb: bool = False, u_dim: int = 0):
    return classes()[1](n_in, latent, degree, fb, u_dim)


def make_mlp(n_in: int, hidden: int, latent: int = LATENT):
    return classes()[2](n_in, hidden, latent)


def n_params(model) -> int:
    return int(sum(p.numel() for p in model.parameters()))


def pnn_params(n_in: int, latent: int = LATENT, degree: int = DEGREE, fb: bool = False) -> int:
    k_head = latent * (2 if fb else 1)
    return n_in * latent * (degree + 1) + latent + k_head * NQ * (degree + 1) + NQ


def mlp_params(n_in: int, hidden: int, latent: int = LATENT) -> int:
    return (n_in + 1) * hidden + (hidden + 1) * latent + (latent + 1) * NQ


def mlp_hidden_for(n_in: int, target: int, latent: int = LATENT) -> int:
    """Hidden width whose MLP parameter count is closest to `target`."""
    return min(range(1, 400), key=lambda h: abs(mlp_params(n_in, h, latent) - target))


# ------------------------------------------------------------------ losses


def pinball_np(y: np.ndarray, q: np.ndarray, taus=TAUS) -> np.ndarray:
    """Per-row pinball loss averaged over taus. y (N,), q (N, len(taus))."""
    t = np.asarray(taus, dtype=float)[None, :]
    e = np.asarray(y, dtype=float)[:, None] - np.asarray(q, dtype=float)
    return np.mean(np.maximum(t * e, (t - 1) * e), axis=1)


def _pinball_t(y, q, taus_t):
    import torch

    e = y[:, None] - q
    return torch.maximum(taus_t * e, (taus_t - 1) * e).mean()


def init_head_bias(head, y: np.ndarray) -> None:
    """Set the raw head bias so the network starts at the unconditional quantiles of y."""
    torch = _torch()
    qs = np.quantile(np.asarray(y, dtype=float), TAUS)
    d = np.maximum(np.diff(qs), 1e-4)
    inv = lambda v: float(np.log(np.expm1(v)))  # noqa: E731  softplus^-1
    b = [0.0] * NQ
    b[3] = float(qs[3])
    b[4], b[5], b[6] = inv(d[3]), inv(d[4]), inv(d[5])
    b[2], b[1], b[0] = inv(d[2]), inv(d[1]), inv(d[0])
    with torch.no_grad():
        head.bias.copy_(torch.as_tensor(b, dtype=head.bias.dtype))


# ------------------------------------------------------------------ training


@dataclass
class QFit:
    kind: str           # 'pnn' | 'mlp'
    state: dict
    meta: dict          # constructor args
    best_val: float
    epochs: int

    def model(self):
        m = make_pnn(**self.meta) if self.kind == "pnn" else make_mlp(**self.meta)
        m.load_state_dict(self.state)
        m.eval()
        return m

    def predict(self, Z: np.ndarray, U: np.ndarray | None = None) -> np.ndarray:
        torch = _torch()
        m = self.model()
        with torch.no_grad():
            x = torch.as_tensor(np.asarray(Z, dtype=np.float32))
            u = None if U is None else torch.as_tensor(np.asarray(U, dtype=np.float32))
            return m(x, u).numpy().astype(float)

    def latent(self, Z: np.ndarray) -> np.ndarray:
        torch = _torch()
        m = self.model()
        with torch.no_grad():
            return m.latent_of(torch.as_tensor(np.asarray(Z, dtype=np.float32))).numpy().astype(float)


def train(model, Zf, yf, Zv, yv, *, seed: int, cfg: dict | None = None, Uf=None, Uv=None, params=None, init_bias: bool = True):
    """Train `model` (in place) with AdamW on the mean pinball loss; early stopping on (Zv, yv). Returns
    (best_state, best_val, best_epoch). Deterministic given the seed (CPU)."""
    torch = _torch()
    c = {**DEFAULT_CFG, **(cfg or {})}
    torch.manual_seed(seed)
    g = torch.Generator().manual_seed(seed)
    if init_bias:
        init_head_bias(model.head, yf)
    taus_t = torch.as_tensor(np.asarray(TAUS, dtype=np.float32))[None, :]
    Xf = torch.as_tensor(np.asarray(Zf, dtype=np.float32))
    Yf = torch.as_tensor(np.asarray(yf, dtype=np.float32))
    Xv = torch.as_tensor(np.asarray(Zv, dtype=np.float32))
    Yv = torch.as_tensor(np.asarray(yv, dtype=np.float32))
    Tf = None if Uf is None else torch.as_tensor(np.asarray(Uf, dtype=np.float32))
    Tv = None if Uv is None else torch.as_tensor(np.asarray(Uv, dtype=np.float32))
    ps = list(model.parameters()) if params is None else list(params)
    opt = torch.optim.AdamW(ps, lr=c["lr"], weight_decay=c["weight_decay"])

    def vloss() -> float:
        model.eval()
        with torch.no_grad():
            return float(_pinball_t(Yv, model(Xv, Tv), taus_t))

    best, best_state, best_ep, bad = vloss(), copy.deepcopy(model.state_dict()), 0, 0
    n = Xf.shape[0]
    for ep in range(1, c["max_epochs"] + 1):
        model.train()
        perm = torch.randperm(n, generator=g)
        for i in range(0, n, c["batch"]):
            idx = perm[i:i + c["batch"]]
            opt.zero_grad()
            loss = _pinball_t(Yf[idx], model(Xf[idx], None if Tf is None else Tf[idx]), taus_t)
            loss.backward()
            opt.step()
        v = vloss()
        if v < best - 1e-7:
            best, best_state, best_ep, bad = v, copy.deepcopy(model.state_dict()), ep, 0
        else:
            bad += 1
            if bad >= c["patience"]:
                break
    model.load_state_dict(best_state)
    model.eval()
    return best_state, best, best_ep


def fit_pnn(Zf, yf, Zv, yv, *, seed: int, cfg: dict | None = None, latent: int = LATENT, degree: int = DEGREE) -> QFit:
    torch = _torch()
    torch.manual_seed(seed)
    m = make_pnn(Zf.shape[1], latent, degree)
    st, bv, ep = train(m, Zf, yf, Zv, yv, seed=seed, cfg=cfg)
    return QFit("pnn", st, {"n_in": Zf.shape[1], "latent": latent, "degree": degree}, bv, ep)


def fit_mlp(Zf, yf, Zv, yv, *, seed: int, hidden: int, cfg: dict | None = None, latent: int = LATENT) -> QFit:
    torch = _torch()
    torch.manual_seed(seed)
    m = make_mlp(Zf.shape[1], hidden, latent)
    st, bv, ep = train(m, Zf, yf, Zv, yv, seed=seed, cfg=cfg)
    return QFit("mlp", st, {"n_in": Zf.shape[1], "hidden": hidden, "latent": latent}, bv, ep)


def fit_head_variant(base: QFit, Zf, yf, Zv, yv, *, seed: int, cfg: dict | None = None, dyn: LatentDynamics | None = None,
                     Uf=None, Uv=None) -> QFit:
    """Continue training only the head of a fitted PNN with the encoder frozen: control (dyn None) or feedback (dyn
    given: the head also reads zhat_{t+1}; the extra edges start at zero so both variants start from `base`)."""
    torch = _torch()
    torch.manual_seed(seed)
    meta = dict(base.meta)
    fb = dyn is not None
    u_dim = 0 if not fb else int(dyn.B.shape[1])
    m = make_pnn(meta["n_in"], meta["latent"], meta["degree"], fb=fb, u_dim=u_dim)
    src = base.state
    with torch.no_grad():
        m.enc.load_state_dict({k[4:]: v for k, v in src.items() if k.startswith("enc.")})
        k = meta["latent"]
        m.head.bias.copy_(src["head.bias"])
        m.head.coef.zero_()
        m.head.w_res.zero_()
        m.head.coef[:, :k, :] = src["head.coef"]
        m.head.w_res[:, :k] = src["head.w_res"]
        if fb:
            m.A.copy_(torch.as_tensor(dyn.A, dtype=torch.float32))
            m.B[:, :u_dim] = torch.as_tensor(dyn.B, dtype=torch.float32)
            m.c.copy_(torch.as_tensor(dyn.c, dtype=torch.float32))
            m.zmu.copy_(torch.as_tensor(dyn.zmu, dtype=torch.float32))
            m.zsd.copy_(torch.as_tensor(dyn.zsd, dtype=torch.float32))
    for p in m.enc.parameters():
        p.requires_grad_(False)
    st, bv, ep = train(m, Zf, yf, Zv, yv, seed=seed, cfg=cfg, Uf=Uf if fb else None, Uv=Uv if fb else None,
                       params=m.head.parameters(), init_bias=False)
    meta.update({"fb": fb, "u_dim": u_dim})
    return QFit("pnn", st, meta, bv, ep)


def ensemble_quantiles(qs: list[np.ndarray]) -> np.ndarray:
    """Mean of per-member quantiles (a mean of sorted vectors stays sorted)."""
    return np.mean(np.stack(qs, axis=0), axis=0)


# ------------------------------------------------------------------ latent dynamics


@dataclass
class LatentDynamics:
    A: np.ndarray
    B: np.ndarray
    c: np.ndarray
    zmu: np.ndarray
    zsd: np.ndarray

    def predict(self, z: np.ndarray, u: np.ndarray) -> np.ndarray:
        zs = (np.asarray(z, dtype=float) - self.zmu) / self.zsd
        nxt = zs @ self.A.T + np.asarray(u, dtype=float) @ self.B.T + self.c
        return nxt * self.zsd + self.zmu

    def to_dict(self) -> dict:
        return {k: np.asarray(getattr(self, k)).tolist() for k in ("A", "B", "c", "zmu", "zsd")}


def fit_latent_dynamics(z_t: np.ndarray, z_next: np.ndarray, u_t: np.ndarray, ridge: float = 1e-3) -> LatentDynamics:
    """Ridge least squares of standardised z_{t+1} on [z_t, u_t, 1] (intercept unpenalised)."""
    z_t, z_next, u_t = (np.asarray(a, dtype=float) for a in (z_t, z_next, u_t))
    zmu = z_t.mean(axis=0)
    zsd = z_t.std(axis=0)
    zsd = np.where(zsd > 1e-9, zsd, 1.0)
    X = np.c_[(z_t - zmu) / zsd, u_t, np.ones(len(z_t))]
    Y = (z_next - zmu) / zsd
    P = ridge * len(X) * np.eye(X.shape[1])
    P[-1, -1] = 0.0
    W = np.linalg.solve(X.T @ X + P, X.T @ Y)  # (k + m + 1, k)
    k, m = z_t.shape[1], u_t.shape[1]
    return LatentDynamics(W[:k].T.copy(), W[k:k + m].T.copy(), W[-1].copy(), zmu, zsd)


def latent_r2_gain(z_t, z_next, z_pred, zmu, zsd) -> dict:
    """Out-of-sample R^2 of the dynamics vs persistence (latent standardised by training mean / sd, dims weighted
    equally). Also returns per-row gain contributions g = (e_pers^2 - e_model^2) / SST, which average to the gain."""
    zs = lambda a: (np.asarray(a, dtype=float) - zmu) / zsd  # noqa: E731
    a, b, p = zs(z_t), zs(z_next), zs(z_pred)
    sst = np.sum((b - b.mean(axis=0)) ** 2, axis=0)  # per dim, around the test mean (classical R^2)
    sst = np.where(sst > 0, sst, 1.0)
    e_m = (b - p) ** 2
    e_p = (b - a) ** 2
    r2_m = 1 - e_m.sum(0) / sst
    r2_p = 1 - e_p.sum(0) / sst
    g = ((e_p - e_m) / sst[None, :]).mean(axis=1) * len(b)  # mean over dims, scaled so mean(g) = gain
    return {"r2_model": r2_m.tolist(), "r2_persist": r2_p.tolist(), "gain": float(np.mean(r2_m - r2_p)), "g": g}


# ------------------------------------------------------------------ conformal


def cqr_scores(lo: np.ndarray, hi: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.maximum(np.asarray(lo) - np.asarray(y), np.asarray(y) - np.asarray(hi))


def conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    """Finite-sample split-conformal quantile: the ceil((n+1)(1-alpha))-th smallest score (inf if beyond n)."""
    s = np.sort(np.asarray(scores, dtype=float)[np.isfinite(scores)])
    n = len(s)
    if n == 0:
        return float("inf")
    k = int(math.ceil((n + 1) * (1 - alpha)))
    if k > n:
        return float(s[-1]) if alpha <= 0 else float("inf")
    return float(s[max(k, 1) - 1])


def split_conformal(lo, hi, cal_scores, alpha: float) -> tuple[np.ndarray, np.ndarray, float]:
    Q = conformal_quantile(cal_scores, alpha)
    return np.asarray(lo) - Q, np.asarray(hi) + Q, Q


def aci(lo: np.ndarray, hi: np.ndarray, y: np.ndarray, alpha: float, *, gamma: float = 0.005, window: int = 1000,
        warm: np.ndarray | None = None) -> dict:
    """Adaptive conformal inference over a time-ordered stream. At step t the adjustment Q_t is the
    (1 - alpha_t) empirical quantile (finite-sample rank) of the last `window` realised scores (warm-start scores
    first); y_t is revealed after the interval is issued, then alpha_{t+1} = alpha_t + gamma (alpha - err_t)."""
    lo, hi, y = (np.asarray(a, dtype=float) for a in (lo, hi, y))
    n = len(y)
    buf = list(np.asarray(warm, dtype=float)[-window:]) if warm is not None else []
    import bisect
    from collections import deque

    dq = deque(buf)
    srt = sorted(buf)
    at = alpha
    out_lo, out_hi, err, alphas, Qs = (np.empty(n) for _ in range(5))
    for t in range(n):
        m = len(srt)
        if m == 0:
            Q = 0.0
        elif at <= 0:
            Q = srt[-1]
        elif at >= 1:
            Q = -np.inf
        else:
            k = int(math.ceil((m + 1) * (1 - at)))
            Q = srt[min(max(k, 1), m) - 1]
        Qs[t] = Q
        if np.isfinite(Q):
            out_lo[t], out_hi[t] = lo[t] - Q, hi[t] + Q
        else:  # empty set
            out_lo[t] = out_hi[t] = 0.5 * (lo[t] + hi[t])
        e = float(not (out_lo[t] <= y[t] <= out_hi[t])) if np.isfinite(Q) else 1.0
        err[t] = e
        alphas[t] = at
        at = at + gamma * (alpha - e)
        s = max(lo[t] - y[t], y[t] - hi[t])
        if np.isfinite(s):
            dq.append(s)
            bisect.insort(srt, s)
            if len(dq) > window:
                old = dq.popleft()
                srt.pop(bisect.bisect_left(srt, old))
    return {"lo": out_lo, "hi": out_hi, "err": err, "alpha_t": alphas, "Q": Qs}


# ------------------------------------------------------------------ symbolic extraction


def poly_fit(x: np.ndarray, f: np.ndarray, max_deg: int = 3) -> list[dict]:
    """Least-squares polynomial fits of f(x), degree 1..max_deg: [{'deg', 'coef' (c0..cd), 'r2'}]."""
    x = np.asarray(x, dtype=float)
    f = np.asarray(f, dtype=float)
    sst = float(np.sum((f - f.mean()) ** 2))
    out = []
    for d in range(1, max_deg + 1):
        V = np.vander(x, d + 1, increasing=True)
        c, *_ = np.linalg.lstsq(V, f, rcond=None)
        sse = float(np.sum((f - V @ c) ** 2))
        out.append({"deg": d, "coef": c.tolist(), "r2": 1 - sse / sst if sst > 0 else 1.0})
    return out


def poly_str(coef, var: str = "x", digits: int = 3) -> str:
    terms = []
    for i, c in enumerate(coef):
        if abs(c) < 10 ** (-digits):
            continue
        v = "" if i == 0 else (var if i == 1 else f"{var}^{i}")
        s = f"{abs(c):.{digits}f}" + (f" {v}" if v else "")
        terms.append(("-" if c < 0 else "+") + " " + s)
    if not terms:
        return "0"
    s = " ".join(terms)
    return s[2:] if s.startswith("+ ") else "-" + s[2:]


def edge_table(layer, X: np.ndarray, in_names, out_names, *, max_deg: int = 3, r2_ok: float = 0.99, n_max: int = 5000,
               seed: int = 0) -> list[dict]:
    """For every edge of a KANLayer: importance (sd of phi over the sample X of its inputs), polynomial fits and the
    chosen closed form (lowest degree with R^2 >= r2_ok, else max_deg)."""
    torch = _torch()
    X = np.asarray(X, dtype=float)
    if len(X) > n_max:
        X = X[np.random.default_rng(seed).choice(len(X), n_max, replace=False)]
    with torch.no_grad():
        E = layer.edges(torch.as_tensor(X, dtype=torch.float32)).numpy().astype(float)  # (N, out, in)
    rows = []
    for j in range(E.shape[1]):
        for i in range(E.shape[2]):
            f = E[:, j, i]
            fits = poly_fit(X[:, i], f, max_deg)
            pick = next((p for p in fits if p["r2"] >= r2_ok), fits[-1])
            rows.append({"out": out_names[j], "in": in_names[i], "importance": float(np.std(f)), "fits": fits,
                         "deg": pick["deg"], "r2": pick["r2"], "coef": pick["coef"], "formula": poly_str(pick["coef"], in_names[i])})
    return sorted(rows, key=lambda r: -r["importance"])


__all__ = [
    "CLIP", "DEFAULT_CFG", "DEGREE", "GRID", "LATENT", "LatentDynamics", "NQ", "PARAM_CAP", "QFit", "TAUS", "aci", "classes",
    "conformal_quantile", "cqr_scores", "edge_table", "ensemble_quantiles", "fit_head_variant", "fit_latent_dynamics", "fit_mlp",
    "fit_pnn", "latent_r2_gain", "make_mlp", "make_pnn", "mlp_hidden_for", "mlp_params", "n_params", "pinball_np", "pnn_params",
    "poly_fit", "poly_str", "split_conformal", "train",
]
