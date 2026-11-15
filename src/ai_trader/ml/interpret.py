"""Interpretability tools for the daily-panel models (H-NN3, H-NN4 and the descriptive comparisons).

TreeSHAP (Lundberg et al. 2018/2020), via xgboost's exact path-dependent implementation:
  shap_main(fit, Z)      (n, F+1) contributions; the last column is the bias; rows sum to the margin prediction
  shap_interactions(...) (n, F+1, F+1) interaction values; row i sums to the main contribution of feature i, the
                         diagonal holds the main effects, off-diagonal (i, j) + (j, i) the pairwise interaction
  interaction_summary    from per-date sums of |interaction values|: the off-diagonal share of total absolute
                         attribution (bias row / column excluded) and the top pairs by mean |phi_ij| + |phi_ji|

Sparse autoencoder (dictionary learning on activations; Bricken et al. 2023 'Towards Monosemanticity', Cunningham et
al. 2023), fit_sae:
  x~ = x / s, s = sqrt(mean ||x||^2 / d)                     (input scale normalisation)
  z  = ReLU(W_e (x~ - b_d) + b_e),  x^ = W_d z + b_d          (untied encoder / decoder, pre-encoder bias b_d)
  loss = mean ||x~ - x^||^2 + l1 * mean sum_i z_i             (decoder columns renormalised to unit norm each step,
                                                               so the L1 penalty cannot be dodged by scaling)
  b_d initialised at the data mean; dead features (no activation over the last `resample_window` steps) are
  resampled at every `resample_every` steps up to half of training: the decoder column is set to the normalised
  residual of an input drawn with probability ~ its squared reconstruction loss, the encoder row to 0.2 x that
  direction x the mean alive encoder norm, its bias to 0 (Adam state reset).
  Feature 'use' = firing frequency on the fitting activations.

Integrated gradients (Sundararajan et al. 2017): IG_i(x) = (x_i - x'_i) * mean_k d f(x' + a_k (x - x')) / d x_i over
a midpoint Riemann grid a_k = (k - 1/2) / K, baseline x' = 0 (= the training mean after standardisation). For an
ensemble the attributions of the averaged output are the average of the members' attributions (IG is linear in f).

Linear probes: ridge regression of a state variable on a layer's activations, fit on one sample, R^2 on another.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ai_trader.ml.models import _torch

# ------------------------------------------------------------------ TreeSHAP


def shap_main(fit, Z: np.ndarray) -> np.ndarray:
    return fit.predict(Z, pred_contribs=True)


def shap_interactions(fit, Z: np.ndarray) -> np.ndarray:
    return fit.predict(Z, pred_interactions=True)


def interaction_summary(abs_sum: np.ndarray, names, top: int = 5) -> dict:
    """abs_sum: (F, F) sum over rows of |phi_ij| (bias excluded). Off-diagonal share and the top pairs."""
    A = np.asarray(abs_sum, dtype=float)
    tot = A.sum()
    diag = np.trace(A)
    share = float((tot - diag) / tot) if tot > 0 else float("nan")
    P = A + A.T
    iu = np.triu_indices(len(A), 1)
    vals = P[iu]
    order = np.argsort(-vals)
    pairs = [(str(names[iu[0][k]]), str(names[iu[1][k]]), float(vals[k] / tot)) for k in order[:top]]
    return {"interaction_share": share, "top_pairs": pairs, "main_share": {str(names[i]): float(A[i, i] / tot) for i in range(len(A))}}


def pair_set(summary: dict) -> set[frozenset]:
    return {frozenset((a, b)) for a, b, _ in summary["top_pairs"]}


# ------------------------------------------------------------------ sparse autoencoder


@dataclass
class SAE:
    W_e: np.ndarray  # (m, d)
    b_e: np.ndarray  # (m,)
    W_d: np.ndarray  # (d, m), unit-norm columns
    b_d: np.ndarray  # (d,)
    scale: float
    l1: float
    history: dict

    @property
    def m(self) -> int:
        return self.W_e.shape[0]

    def encode(self, X: np.ndarray) -> np.ndarray:
        Xs = np.asarray(X, dtype=float) / self.scale
        return np.maximum((Xs - self.b_d) @ self.W_e.T + self.b_e, 0.0)

    def decode(self, Z: np.ndarray) -> np.ndarray:
        """Reconstruction in the ORIGINAL activation units."""
        return (np.asarray(Z, dtype=float) @ self.W_d.T + self.b_d) * self.scale

    def atom(self, i: int) -> np.ndarray:
        """Feature i's contribution per unit activation, in original units."""
        return self.W_d[:, i] * self.scale

    def metrics(self, X: np.ndarray) -> dict:
        X = np.asarray(X, dtype=float)
        Z = self.encode(X)
        R = self.decode(Z)
        fvu = float(np.sum((X - R) ** 2) / np.sum((X - X.mean(0)) ** 2))
        return {"fvu": fvu, "l0": float((Z > 0).sum(1).mean()), "dead_frac": float(np.mean((Z > 0).sum(0) == 0)),
                "freq": (Z > 0).mean(0)}

    def to_dict(self) -> dict:
        return {"W_e": self.W_e.tolist(), "b_e": self.b_e.tolist(), "W_d": self.W_d.tolist(), "b_d": self.b_d.tolist(),
                "scale": self.scale, "l1": self.l1}


def fit_sae(X: np.ndarray, m: int, l1: float = 0.3, steps: int = 6000, batch: int = 256, lr: float = 1e-3, seed: int = 0,
            resample_every: int = 1000, resample_window: int = 500) -> SAE:
    torch = _torch()
    X = np.asarray(X, dtype=np.float64)
    n, d = X.shape
    scale = float(np.sqrt(np.mean(np.sum(X * X, axis=1)) / d)) or 1.0
    Xs = torch.as_tensor((X / scale).astype(np.float32))
    torch.manual_seed(seed)
    g = torch.Generator().manual_seed(seed)
    W_d = torch.randn(d, m, generator=g)
    W_d = W_d / W_d.norm(dim=0, keepdim=True)
    W_e = W_d.T.clone() * 0.5
    params = {"W_e": W_e.requires_grad_(), "b_e": torch.zeros(m, requires_grad=True), "W_d": W_d.requires_grad_(),
              "b_d": Xs.mean(0).clone().requires_grad_()}
    opt = torch.optim.Adam(list(params.values()), lr=lr)
    last_fire = np.zeros(m, dtype=np.int64)
    hist = {"loss": [], "resampled": []}

    def fwd(xb):
        z = torch.relu((xb - params["b_d"]) @ params["W_e"].T + params["b_e"])
        xh = z @ params["W_d"].T + params["b_d"]
        return z, xh

    for step in range(1, steps + 1):
        idx = torch.randint(0, n, (batch,), generator=g)
        xb = Xs[idx]
        z, xh = fwd(xb)
        mse = ((xb - xh) ** 2).sum(1).mean()
        loss = mse + l1 * z.sum(1).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        with torch.no_grad():
            params["W_d"].data /= params["W_d"].data.norm(dim=0, keepdim=True).clamp_min(1e-8)
            fired = (z > 0).any(0).numpy()
            last_fire[fired] = step
        if step % 200 == 0:
            hist["loss"].append(float(mse.detach()))
        if step % resample_every == 0 and step <= steps // 2:
            dead = np.flatnonzero(step - last_fire >= resample_window)
            if len(dead):
                with torch.no_grad():
                    sub = torch.randint(0, n, (min(n, 4096),), generator=g)
                    xs = Xs[sub]
                    _, xh2 = fwd(xs)
                    lo = ((xs - xh2) ** 2).sum(1)
                    pr = (lo * lo) / (lo * lo).sum()
                    pick = torch.multinomial(pr, len(dead), replacement=True, generator=g)
                    alive = np.setdiff1d(np.arange(m), dead)
                    en = float(params["W_e"].data[alive].norm(dim=1).mean()) if len(alive) else 1.0
                    for k, i in enumerate(dead):
                        v = xs[pick[k]] - params["b_d"].data
                        v = v / v.norm().clamp_min(1e-8)
                        params["W_d"].data[:, i] = v
                        params["W_e"].data[i] = v * 0.2 * en
                        params["b_e"].data[i] = 0.0
                    last_fire[dead] = step
                opt = torch.optim.Adam(list(params.values()), lr=lr)
                hist["resampled"].append([step, int(len(dead))])
    p = {k: v.detach().numpy().astype(float) for k, v in params.items()}
    return SAE(p["W_e"], p["b_e"], p["W_d"], p["b_d"], scale, l1, hist)


def atom_recovery(true_dict: np.ndarray, sae: SAE) -> np.ndarray:
    """For each true atom (columns of true_dict), the max cosine similarity with any learned decoder column."""
    D = np.asarray(true_dict, dtype=float)
    D = D / np.linalg.norm(D, axis=0, keepdims=True)
    W = sae.W_d / np.linalg.norm(sae.W_d, axis=0, keepdims=True)
    return (D.T @ W).max(axis=1)


# ------------------------------------------------------------------ integrated gradients


def integrated_gradients(fits, Z: np.ndarray, steps: int = 32) -> np.ndarray:
    """IG of the ensemble's log-RV output (original units) w.r.t. standardised inputs Z, baseline 0. fits: [MLPFit]."""
    torch = _torch()
    Zt = torch.as_tensor(np.asarray(Z, dtype=np.float32))
    a = (torch.arange(steps, dtype=torch.float32) + 0.5) / steps
    out = np.zeros(Z.shape, dtype=float)
    for f in fits:
        m = f.model()
        grads = torch.zeros_like(Zt)
        for ak in a:
            x = (ak * Zt).clone().requires_grad_(True)
            y = m(x).sum()
            (gk,) = torch.autograd.grad(y, x)
            grads += gk
        out += (Zt * grads / steps).numpy().astype(float) * f.y_sd
    return out / len(fits)


def hidden_activations(fit, Z: np.ndarray) -> list[np.ndarray]:
    torch = _torch()
    with torch.no_grad():
        hs = fit.model().hidden_acts(torch.as_tensor(np.asarray(Z, dtype=np.float32)))
    return [h.numpy().astype(float) for h in hs]


def head_from_last(fit, H: np.ndarray) -> np.ndarray:
    """MLP output (log-RV units) from (possibly patched) last-hidden-layer activations H."""
    s = fit.state
    w = s["head.weight"].numpy().astype(float).ravel()
    b = float(s["head.bias"].numpy().ravel()[0])
    return (np.asarray(H, dtype=float) @ w + b) * fit.y_sd + fit.y_mu


# ------------------------------------------------------------------ probes and correlations


def ridge_probe(Htr: np.ndarray, ytr: np.ndarray, Hte: np.ndarray, yte: np.ndarray, ridge: float = 1.0) -> float:
    mu, sd = Htr.mean(0), Htr.std(0)
    sd = np.where(sd > 0, sd, 1.0)
    A = np.c_[np.ones(len(ytr)), (Htr - mu) / sd]
    P = ridge * np.eye(A.shape[1])
    P[0, 0] = 0
    b = np.linalg.solve(A.T @ A + P, A.T @ ytr)
    pred = np.c_[np.ones(len(yte)), (Hte - mu) / sd] @ b
    return float(1 - np.sum((yte - pred) ** 2) / np.sum((yte - yte.mean()) ** 2))


def _rank(x: np.ndarray) -> np.ndarray:
    from scipy.stats import rankdata

    return rankdata(x)


def spearman(a: np.ndarray, b: np.ndarray, min_n: int = 10) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < min_n or np.std(a[ok]) == 0 or np.std(b[ok]) == 0:
        return float("nan")
    return float(np.corrcoef(_rank(a[ok]), _rank(b[ok]))[0, 1])


__all__ = ["SAE", "atom_recovery", "fit_sae", "head_from_last", "hidden_activations", "integrated_gradients", "interaction_summary",
           "pair_set", "ridge_probe", "shap_interactions", "shap_main", "spearman"]
