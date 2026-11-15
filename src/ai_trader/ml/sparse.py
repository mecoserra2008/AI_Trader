"""Sparsity tools for the daily ML panel (H-SP1..H-SP3, program 2026-11-12, agent D): lottery-ticket pruning of the
H-NN1 MLP, randomised-lasso stability selection, and additive-spline (GAM) descriptions of hidden units.

Pruning (Frankle and Carbin 2019, with weight rewinding as Frankle et al. 2020):
  * `train_masked` reproduces models.fit_mlp step for step (same seed, generator, init, AdamW, early stopping on the
    validation block); with no mask it is the H-NN1 member. It also returns the weights, and the batch-order generator
    state, at the end of epoch REWIND_EPOCH (the rewind point).
  * `imp_chain` runs iterative magnitude pruning: train dense, then for each round prune the smallest-|w| surviving
    weights of each layer (HIDDEN_RATE = 20% of the survivors of each hidden Linear layer, HEAD_RATE = 10% of the head,
    as Frankle and Carbin's output layer; at least one weight per layer survives), rewind the survivors to the rewind
    point, and retrain with the mask re-applied after every optimizer step. Biases are never pruned.
  * Masks are deterministic counts, so `sparsity_schedule` gives the sparsity of every round before any fit.

Stability selection (Meinshausen and Buhlmann 2010): `lasso_path_gram` is coordinate descent on the Gram matrix with
per-feature penalty weights; `stability_selection` draws subsamples of half of the DATES (rows of a date stay together),
random penalty weights 1/W_k with W_k ~ U[alpha, 1], a common lambda grid, and returns selection probabilities per
lambda. The region Lambda = lambdas down to the first one at which the mean size of the union of selected sets reaches
q_max; stable set = max over Lambda of the probability >= pi_thr. Per-family error bound E[V] <= q^2 / ((2 pi - 1) p).

GAM: `spline_basis` (cubic truncated-power basis with knots at quantiles of the fitting data; one-hot for inputs with
few distinct values), `gam_search` (best single input and best additive pair by fitting-sample R^2 using Gram blocks;
R^2 reported on a separate evaluation sample).
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field

import numpy as np

from ai_trader.ml import models as md

REWIND_EPOCH = 1
HIDDEN_RATE = 0.20
HEAD_RATE = 0.10
N_ROUNDS = 22  # first round with >= 99% overall sparsity for the (32, 16) vol network (99.04%)


def _torch():
    return md._torch()


# ------------------------------------------------------------------ masks


def weight_keys(n_layers: int) -> list[str]:
    return [*(f"layers.{i}.weight" for i in range(n_layers)), "head.weight"]


def dense_masks(n_in: int, hidden) -> dict[str, np.ndarray]:
    dims = [n_in, *hidden, 1]
    keys = weight_keys(len(hidden))
    return {k: np.ones((b, a), dtype=bool) for k, a, b in zip(keys, dims[:-1], dims[1:], strict=True)}


def _n_prune(n_alive: int, rate: float) -> int:
    return int(min(max(n_alive - 1, 0), round(rate * n_alive)))


def sparsity_schedule(n_in: int, hidden, rounds: int = N_ROUNDS, hidden_rate: float = HIDDEN_RATE,
                      head_rate: float = HEAD_RATE) -> list[dict]:
    """Alive weights per layer and overall sparsity of every round (deterministic: depends only on counts)."""
    m = dense_masks(n_in, hidden)
    keys = list(m)
    alive = {k: int(v.size) for k, v in m.items()}
    total = sum(alive.values())
    out = [{"round": 0, "alive": dict(alive), "n_alive": total, "sparsity": 0.0}]
    for r in range(1, rounds + 1):
        for k in keys:
            alive[k] -= _n_prune(alive[k], head_rate if k == "head.weight" else hidden_rate)
        n = sum(alive.values())
        out.append({"round": r, "alive": dict(alive), "n_alive": n, "sparsity": 1.0 - n / total})
    return out


def round_at(schedule: list[dict], target: float) -> int:
    """First round whose overall sparsity is >= target."""
    for s in schedule:
        if s["sparsity"] >= target - 1e-12:
            return int(s["round"])
    raise ValueError(f"schedule never reaches {target}")


def prune(state: dict, masks: dict[str, np.ndarray], hidden_rate: float = HIDDEN_RATE, head_rate: float = HEAD_RATE) -> dict[str, np.ndarray]:
    """Next masks: drop the smallest-|w| surviving weights of each layer (layer-wise magnitude pruning)."""
    out = {}
    for k, m in masks.items():
        w = np.abs(np.asarray(state[k].detach().cpu().numpy() if hasattr(state[k], "detach") else state[k], dtype=float)).ravel()
        mf = m.ravel().copy()
        alive = np.flatnonzero(mf)
        k_drop = _n_prune(len(alive), head_rate if k == "head.weight" else hidden_rate)
        if k_drop > 0:
            order = alive[np.argsort(w[alive], kind="stable")]
            mf[order[:k_drop]] = False
        out[k] = mf.reshape(m.shape)
    return out


def apply_masks(model, masks: dict[str, np.ndarray] | None) -> None:
    if not masks:
        return
    torch = _torch()
    sd = dict(model.named_parameters())
    with torch.no_grad():
        for k, m in masks.items():
            sd[k].mul_(torch.as_tensor(m, dtype=sd[k].dtype))


# ------------------------------------------------------------------ masked training (mirrors models.fit_mlp)


@dataclass
class Trained:
    fit: md.MLPFit
    masks: dict | None
    rewind_state: dict | None = None
    rewind_gen: object = None


def train_masked(Zf, yf, Zv, yv, task: str, seed: int, cfg: dict | None = None, masks: dict | None = None,
                 rewind: tuple | None = None) -> Trained:
    """models.fit_mlp with an optional weight mask. rewind = (state, generator_state, epoch) restarts from the rewind
    point (weights masked, fresh AdamW, same batch order as the dense run from that epoch on). Without `rewind` the run
    starts from the seed's initialisation and records the rewind point at the end of epoch REWIND_EPOCH."""
    torch = _torch()
    c = {**md.MLP_DEFAULT, **(cfg or {})}
    torch.manual_seed(seed)
    g = torch.Generator().manual_seed(seed)
    yf = np.asarray(yf, dtype=float)
    yv = np.asarray(yv, dtype=float)
    if task == "vol":
        y_mu, y_sd = float(yf.mean()), float(yf.std() or 1.0)
        tf, tv = (yf - y_mu) / y_sd, (yv - y_mu) / y_sd
        lossf = torch.nn.MSELoss()
    else:
        y_mu, y_sd = 0.0, 1.0
        tf, tv = yf, yv
        lossf = torch.nn.BCEWithLogitsLoss()
    Xf = torch.as_tensor(np.asarray(Zf, dtype=np.float32))
    Yf = torch.as_tensor(tf.astype(np.float32))
    Xv = torch.as_tensor(np.asarray(Zv, dtype=np.float32))
    Yv = torch.as_tensor(tv.astype(np.float32))
    m = md.make_mlp(Xf.shape[1], tuple(c["hidden"]))
    if task == "dir":
        with torch.no_grad():
            p = float(np.clip(yf.mean(), 1e-3, 1 - 1e-3))
            m.head.bias.fill_(math.log(p / (1 - p)))
    ep0 = 0
    if rewind is not None:
        st, gst, ep0 = rewind
        m.load_state_dict(copy.deepcopy(st))
        g.set_state(gst.clone())
    apply_masks(m, masks)
    opt = torch.optim.AdamW(m.parameters(), lr=c["lr"], weight_decay=c["weight_decay"])
    with torch.no_grad():
        best = float(lossf(m(Xv), Yv))
    best_state, best_ep, bad = copy.deepcopy(m.state_dict()), ep0, 0
    rw_state, rw_gen = None, None
    n = Xf.shape[0]
    for ep in range(ep0 + 1, c["max_epochs"] + 1):
        m.train()
        perm = torch.randperm(n, generator=g)
        for i in range(0, n, c["batch"]):
            idx = perm[i:i + c["batch"]]
            opt.zero_grad()
            loss = lossf(m(Xf[idx]), Yf[idx])
            loss.backward()
            opt.step()
            apply_masks(m, masks)
        m.eval()
        if rewind is None and ep == REWIND_EPOCH:
            rw_state, rw_gen = copy.deepcopy(m.state_dict()), g.get_state().clone()
        with torch.no_grad():
            v = float(lossf(m(Xv), Yv))
        if v < best - 1e-6:
            best, best_state, best_ep, bad = v, copy.deepcopy(m.state_dict()), ep, 0
        else:
            bad += 1
            if bad >= c["patience"]:
                break
    if rewind is None and rw_state is None:  # stopped before the rewind epoch cannot happen (>= 1 epoch), guard anyway
        rw_state, rw_gen = copy.deepcopy(m.state_dict()), g.get_state().clone()
    fit = md.MLPFit(best_state, tuple(c["hidden"]), Xf.shape[1], y_mu, y_sd, task, best, best_ep)
    return Trained(fit, masks, rw_state, rw_gen)


@dataclass
class Chain:
    fits: list = field(default_factory=list)  # MLPFit per round (round 0 = dense)
    masks: list = field(default_factory=list)  # masks per round (round 0 = all ones)


def imp_chain(Zf, yf, Zv, yv, task: str, seed: int, cfg: dict | None = None, rounds: int = N_ROUNDS) -> Chain:
    """Iterative magnitude pruning with rewinding to the end of epoch REWIND_EPOCH (module docstring)."""
    c = {**md.MLP_DEFAULT, **(cfg or {})}
    dense = train_masked(Zf, yf, Zv, yv, task, seed, c)
    masks = dense_masks(np.asarray(Zf).shape[1], tuple(c["hidden"]))
    ch = Chain([dense.fit], [masks])
    rewind = (dense.rewind_state, dense.rewind_gen, REWIND_EPOCH)
    cur = dense.fit
    for _ in range(rounds):
        masks = prune(cur.state, masks)
        tr = train_masked(Zf, yf, Zv, yv, task, seed, c, masks=masks, rewind=rewind)
        cur = tr.fit
        ch.fits.append(cur)
        ch.masks.append(masks)
    return ch


def check_mask(fit: md.MLPFit, masks: dict[str, np.ndarray]) -> bool:
    """True iff every masked weight of the fitted state is exactly zero."""
    for k, m in masks.items():
        w = fit.state[k].detach().cpu().numpy()
        if np.any(w[~m] != 0):
            return False
    return True


# ------------------------------------------------------------------ numpy forward pass with unit patching


def np_weights(fit: md.MLPFit) -> list[tuple[np.ndarray, np.ndarray]]:
    s = fit.state
    out = []
    for i in range(len(fit.hidden)):
        out.append((s[f"layers.{i}.weight"].detach().cpu().numpy().astype(float), s[f"layers.{i}.bias"].detach().cpu().numpy().astype(float)))
    out.append((s["head.weight"].detach().cpu().numpy().astype(float), s["head.bias"].detach().cpu().numpy().astype(float)))
    return out


def forward_np(fit: md.MLPFit, Z: np.ndarray, patch: dict[tuple[int, int], float] | None = None) -> tuple[np.ndarray, list[np.ndarray]]:
    """Output in target units (log RV for vol, logit for dir) and hidden activations; patch {(layer, unit): value}
    replaces a unit's activation (ablation) before it feeds the next layer."""
    W = np_weights(fit)
    z = np.asarray(Z, dtype=float)
    hs = []
    for li, (w, b) in enumerate(W[:-1]):
        z = np.maximum(z @ w.T + b, 0.0)
        if patch:
            for (l_, u), v in patch.items():
                if l_ == li:
                    z[:, u] = v
        hs.append(z)
    w, b = W[-1]
    out = (z @ w.T + b).ravel()
    if fit.task == "vol":
        out = out * fit.y_sd + fit.y_mu
    return out, hs


def surviving_units(fit: md.MLPFit, masks: dict[str, np.ndarray]) -> list[tuple[int, int]]:
    """(layer, unit) pairs with at least one surviving input weight and a surviving path to the output."""
    L = len(fit.hidden)
    keys = weight_keys(L)
    # backward reachability to the output
    reach = [None] * L
    nxt = masks[keys[-1]].any(axis=0)  # head: (1, h_L)
    reach[L - 1] = nxt
    for li in range(L - 2, -1, -1):
        m = masks[keys[li + 1]]  # (h_{li+1}, h_li)
        reach[li] = (m[reach[li + 1], :]).any(axis=0)
    # forward: reachable from at least one input through surviving weights
    out = []
    fwd = masks[keys[0]].any(axis=1)
    for li in range(L):
        if li > 0:
            fwd = masks[keys[li]][:, fwd].any(axis=1)
        out += [(li, int(u)) for u in np.flatnonzero(fwd & reach[li])]
    return out


def input_paths(masks: dict[str, np.ndarray]) -> np.ndarray:
    """(n_in,) bool: input j still reaches the output through surviving weights."""
    keys = [k for k in masks if k.startswith("layers.")] + ["head.weight"]
    reach = masks[keys[-1]].any(axis=0)
    for k in reversed(keys[1:-1]):
        reach = masks[k][reach, :].any(axis=0)
    return masks[keys[0]][reach, :].any(axis=0)


def unit_attribution(fit: md.MLPFit, Z: np.ndarray) -> dict[tuple[int, int], float]:
    """Mean |activation x d output / d activation| per hidden unit (gradient x activation, output in target units)."""
    torch = _torch()
    m = fit.model()
    x = torch.as_tensor(np.asarray(Z, dtype=np.float32))
    acts = []
    z = x
    for lin in m.layers:
        z = torch.relu(lin(z))
        z.retain_grad()
        acts.append(z)
    out = m.head(z).squeeze(-1) * (fit.y_sd if fit.task == "vol" else 1.0)
    out.sum().backward()
    res = {}
    for li, a in enumerate(acts):
        ga = (a * a.grad).abs().mean(0).detach().numpy().astype(float)
        for u, v in enumerate(ga):
            res[(li, u)] = float(v)
    return res


# ------------------------------------------------------------------ lasso / stability selection


def lasso_path_gram(G: np.ndarray, c: np.ndarray, lams: np.ndarray, w_pen: np.ndarray | None = None, tol: float = 1e-6,
                    max_iter: int = 500, stop_union: int | None = None) -> np.ndarray:
    """Coefficients (len(lams), p) of min (1/2) b'Gb - c'b + lam sum_k w_k |b_k| (G = X'X/n, c = X'y/n, standardised X,
    centred y), warm-started along a decreasing lambda grid. Features with G_kk = 0 stay at zero. With stop_union, the
    path stops once more than stop_union features have ever been active; the remaining rows are NaN."""
    p = len(c)
    w_pen = np.ones(p) if w_pen is None else np.asarray(w_pen, dtype=float)
    d = np.diag(G).copy()
    ok = d > 1e-12
    b = np.zeros(p)
    r = c.copy()  # r = c - G b (partial residual correlation)
    out = np.full((len(lams), p), np.nan)
    ever = np.zeros(p, dtype=bool)
    for li, lam in enumerate(lams):
        if stop_union is not None and ever.sum() > stop_union:
            break
        thr = lam * w_pen
        for _ in range(max_iter):
            # one full sweep, then sweeps over the active set until converged
            delta = _cd_sweep(G, b, r, d, thr, np.flatnonzero(ok))
            if delta < tol:
                break
            act = np.flatnonzero(ok & (b != 0))
            for _ in range(max_iter):
                if _cd_sweep(G, b, r, d, thr, act) < tol:
                    break
        out[li] = b
        ever |= b != 0
    return out


def _cd_sweep(G: np.ndarray, b: np.ndarray, r: np.ndarray, d: np.ndarray, thr: np.ndarray, ks: np.ndarray) -> float:
    delta = 0.0
    for k in ks:
        bk = b[k]
        z = r[k] + d[k] * bk
        nb = math.copysign(max(abs(z) - thr[k], 0.0), z) / d[k]
        if nb != bk:
            diff = nb - bk
            r -= G[:, k] * diff
            b[k] = nb
            delta = max(delta, abs(diff) * math.sqrt(d[k]))
    return delta


def _std_cols(X: np.ndarray) -> np.ndarray:
    mu = X.mean(0)
    sd = X.std(0)
    return np.where(sd[None, :] > 1e-12, (X - mu) / np.where(sd > 1e-12, sd, 1.0), 0.0)


def lambda_grid(X: np.ndarray, y: np.ndarray, n: int = 40, ratio: float = 0.005) -> np.ndarray:
    Xs = _std_cols(np.asarray(X, dtype=float))
    yc = np.asarray(y, dtype=float) - np.mean(y)
    lmax = float(np.max(np.abs(Xs.T @ yc / len(yc))))
    return lmax * np.logspace(0, math.log10(ratio), n)


@dataclass
class StabilityResult:
    probs: np.ndarray  # (n_lambda, p) selection probabilities
    lams: np.ndarray
    union_size: np.ndarray  # mean over subsamples of |union of selected sets for lambda' >= lambda|
    n_sub: int
    alpha: float
    sel: np.ndarray | None = None  # (n_sub, n_lambda, p) selection indicators when keep_sel=True

    def region(self, q_max: float) -> int:
        """Index of the smallest lambda of Lambda (lambdas[0..idx]) with mean union size <= q_max."""
        ok = np.flatnonzero(self.union_size <= q_max + 1e-12)
        return int(ok.max()) if len(ok) else 0

    def max_prob(self, q_max: float) -> np.ndarray:
        return self.probs[: self.region(q_max) + 1].max(axis=0)

    def selected(self, q_max: float, pi_thr: float) -> np.ndarray:
        return np.flatnonzero(self.max_prob(q_max) >= pi_thr)

    def q_hat(self, q_max: float) -> float:
        return float(self.union_size[self.region(q_max)])


def pfer_bound(q: float, p: int, pi_thr: float) -> float:
    """Meinshausen-Buhlmann bound on the expected number of falsely selected variables."""
    return float(q * q / ((2 * pi_thr - 1) * p))


def q_for_bound(p: int, pi_thr: float, ev: float = 1.0) -> float:
    return math.sqrt(ev * (2 * pi_thr - 1) * p)


def stability_selection(X: np.ndarray, y: np.ndarray, groups: np.ndarray, n_sub: int = 100, alpha: float = 0.5,
                        lams: np.ndarray | None = None, seed: int = 0, progress=None, stop_union: int | None = None, keep_sel: bool = False) -> StabilityResult:
    """Randomised lasso over subsamples of half the groups (dates). Each subsample is standardised on itself.
    stop_union (speed): a subsample's path stops once its union of selected sets exceeds stop_union; its later lambdas
    count as union size p (so the region Lambda can only shrink, never grow) and as not selected."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    rng = np.random.default_rng(seed)
    ug, inv = np.unique(groups, return_inverse=True)
    lams = lambda_grid(X, y) if lams is None else np.asarray(lams, dtype=float)
    p = X.shape[1]
    cnt = np.zeros((len(lams), p))
    usz = np.zeros(len(lams))
    keep = np.zeros((n_sub, len(lams), p), dtype=bool) if keep_sel else None
    for s in range(n_sub):
        pick = rng.choice(len(ug), size=len(ug) // 2, replace=False)
        rows = np.isin(inv, pick)
        Xs = _std_cols(X[rows])
        ys = y[rows] - y[rows].mean()
        n = len(ys)
        G = Xs.T @ Xs / n
        c = Xs.T @ ys / n
        W = rng.uniform(alpha, 1.0, size=p)
        B = lasso_path_gram(G, c, lams, w_pen=1.0 / W, stop_union=stop_union)
        done = np.isfinite(B).all(axis=1)
        sel = (B != 0) & done[:, None]
        cnt += sel
        if keep is not None:
            keep[s] = sel
        usz += np.where(done, np.cumsum(sel, axis=0).astype(bool).sum(axis=1), p)
        if progress is not None:
            progress(s)
    return StabilityResult(cnt / n_sub, lams, usz / n_sub, n_sub, alpha, keep)


# ------------------------------------------------------------------ additive splines (GAM)


@dataclass
class Basis:
    kind: str  # "spline" | "onehot" | "const"
    knots: np.ndarray | None = None
    levels: np.ndarray | None = None
    lo: float = 0.0
    sc: float = 1.0

    def __call__(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        if self.kind == "const":
            return np.zeros((len(x), 0))
        if self.kind == "onehot":
            return np.column_stack([(np.isclose(x, v)).astype(float) for v in self.levels[1:]])
        u = (x - self.lo) / self.sc
        cols = [u, u ** 2, u ** 3] + [np.maximum(u - k, 0.0) ** 3 for k in self.knots]
        return np.column_stack(cols)


def spline_basis(x: np.ndarray, n_knots: int = 6, max_levels: int = 8) -> Basis:
    """Cubic truncated-power basis with knots at interior quantiles of x (fitting data), or one-hot for few levels."""
    x = np.asarray(x, dtype=float)
    lv = np.unique(np.round(x, 9))
    if len(lv) <= 1:
        return Basis("const")
    if len(lv) <= max_levels:
        return Basis("onehot", levels=lv)
    lo, hi = np.quantile(x, [0.005, 0.995])
    sc = (hi - lo) or 1.0
    u = (x - lo) / sc
    kn = np.unique(np.quantile(u, np.linspace(0, 1, n_knots + 2)[1:-1]))
    return Basis("spline", knots=kn, lo=float(lo), sc=float(sc))


def _r2(y: np.ndarray, f: np.ndarray) -> float:
    v = np.sum((y - y.mean()) ** 2)
    return float(1 - np.sum((y - f) ** 2) / v) if v > 1e-12 else float("nan")


@dataclass
class GamFit:
    inputs: tuple
    coef: np.ndarray
    bases: list
    mu: np.ndarray
    sd: np.ndarray

    def design(self, X: np.ndarray) -> np.ndarray:
        B = np.column_stack([self.bases[j](X[:, j]) for j in self.inputs]) if self.inputs else np.zeros((len(X), 0))
        return np.c_[np.ones(len(X)), (B - self.mu) / self.sd]

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.design(X) @ self.coef


def gam_search(Xfit: np.ndarray, yfit: np.ndarray, Xev: np.ndarray, yev: np.ndarray, ridge: float = 1e-3, pairs: bool = True,
               top_singles: int | None = None) -> dict:
    """Best 1-input and best additive 2-input spline fits (chosen by fitting-sample R^2), R^2 on the evaluation sample.
    Gram blocks are computed once, so the pair search costs O(p^2 k^3)."""
    Xfit = np.asarray(Xfit, dtype=float)
    Xev = np.asarray(Xev, dtype=float)
    yfit = np.asarray(yfit, dtype=float)
    yev = np.asarray(yev, dtype=float)
    p = Xfit.shape[1]
    bases = [spline_basis(Xfit[:, j]) for j in range(p)]
    Bs = [bases[j](Xfit[:, j]) for j in range(p)]
    idx, start = [], 0
    for B in Bs:
        idx.append(np.arange(start, start + B.shape[1]))
        start += B.shape[1]
    Ball = np.column_stack([B for B in Bs if B.shape[1]]) if start else np.zeros((len(yfit), 0))
    mu = Ball.mean(0)
    sd = Ball.std(0)
    sd = np.where(sd > 1e-12, sd, 1.0)
    Bc = (Ball - mu) / sd
    n = len(yfit)
    G = Bc.T @ Bc / n
    yc = yfit - yfit.mean()
    cvec = Bc.T @ yc / n
    vy = float(yc @ yc / n)

    def fit_r2(cols: np.ndarray) -> float:
        if len(cols) == 0:
            return 0.0
        Gs = G[np.ix_(cols, cols)] + ridge * np.eye(len(cols))
        b = np.linalg.solve(Gs, cvec[cols])
        return float((2 * b @ cvec[cols] - b @ G[np.ix_(cols, cols)] @ b) / vy) if vy > 0 else float("nan")

    single = np.array([fit_r2(idx[j]) for j in range(p)])
    best1 = int(np.nanargmax(single))
    res = {"single_fit_r2": single.tolist(), "best1": best1}
    cand = range(p) if top_singles is None else np.argsort(-np.nan_to_num(single, nan=-1))[:top_singles]
    best2, b2 = (best1, best1), -np.inf
    if pairs:
        cand = list(cand)
        for a_ in range(len(cand)):
            for b_ in range(a_ + 1, len(cand)):
                i, j = cand[a_], cand[b_]
                r = fit_r2(np.r_[idx[i], idx[j]])
                if r > b2:
                    best2, b2 = (int(i), int(j)), r
    res["best2"] = list(best2)

    def make(inputs: tuple) -> GamFit:
        cols = np.concatenate([idx[j] for j in inputs]) if inputs else np.zeros(0, int)
        if len(cols):
            bvec = np.linalg.solve(G[np.ix_(cols, cols)] + ridge * np.eye(len(cols)), cvec[cols])
        else:
            bvec = np.zeros(0)
        return GamFit(tuple(inputs), np.r_[yfit.mean(), bvec], bases, mu[cols], sd[cols])

    g1 = make((best1,))
    res["fit1"] = g1
    res["r2_fit_1"] = float(single[best1])
    res["r2_ev_1"] = _r2(yev, g1.predict(Xev))
    if pairs:
        g2 = make(tuple(best2))
        res["fit2"] = g2
        res["r2_fit_2"] = float(b2)
        res["r2_ev_2"] = _r2(yev, g2.predict(Xev))
    return res


def gam_bivariate(Xfit: np.ndarray, yfit: np.ndarray, Xev: np.ndarray, yev: np.ndarray, i: int, j: int, n_knots: int = 3,
                  ridge: float = 1e-3) -> float:
    """Descriptive: additive pair plus tensor-product interaction of low-dimension bases; R^2 on the evaluation sample."""
    bi, bj = spline_basis(Xfit[:, i], n_knots), spline_basis(Xfit[:, j], n_knots)

    def des(X):
        A, B = bi(X[:, i]), bj(X[:, j])
        T = (A[:, :, None] * B[:, None, :]).reshape(len(X), -1) if A.shape[1] and B.shape[1] else np.zeros((len(X), 0))
        return np.c_[A, B, T]

    D = des(Xfit)
    mu, sd = D.mean(0), D.std(0)
    sd = np.where(sd > 1e-12, sd, 1.0)
    Dc = (D - mu) / sd
    b = np.linalg.solve(Dc.T @ Dc / len(D) + ridge * np.eye(D.shape[1]), Dc.T @ (yfit - yfit.mean()) / len(D))
    f = yfit.mean() + ((des(Xev) - mu) / sd) @ b
    return _r2(np.asarray(yev, dtype=float), f)


__all__ = [
    "HEAD_RATE",
    "HIDDEN_RATE",
    "N_ROUNDS",
    "REWIND_EPOCH",
    "Basis",
    "Chain",
    "GamFit",
    "StabilityResult",
    "Trained",
    "apply_masks",
    "check_mask",
    "dense_masks",
    "forward_np",
    "gam_bivariate",
    "gam_search",
    "imp_chain",
    "input_paths",
    "lambda_grid",
    "lasso_path_gram",
    "pfer_bound",
    "prune",
    "q_for_bound",
    "round_at",
    "sparsity_schedule",
    "spline_basis",
    "stability_selection",
    "surviving_units",
    "train_masked",
    "unit_attribution",
    "weight_keys",
]
