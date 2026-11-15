"""Nonlinear signal generators for the daily panel: a small MLP ensemble (torch, CPU) and gradient-boosted trees
(xgboost), with the same validation protocol, so the two families get comparable tuning effort (H-NN1, H-NN2).

Protocol inside every training window (fixed before any test result was seen; diary 2026-11-12 "ML design fixed"):
  * the window's training rows are split by decision date: the last VAL_FRAC = 15% of distinct dates are the
    validation block, the rest the fit block; fit rows whose 24h label overlaps the validation block are purged
    (T < first validation date - 1 day). Features are standardised with fit-block statistics (Standardizer);
  * both families stop early on the validation block (MLP: patience 10 epochs, max 200; XGB: 50 rounds, max 2000)
    and keep the best iterate (validation rows are never fitted);
  * the variance bias correction of the log forecasts is estimated on the validation block too:
    F = exp(f + c), c = log mean(RV / exp(f)) (the QLIKE-optimal constant scale; in-sample only).

MLP: input -> Linear -> ReLU -> Linear -> ReLU -> Linear(1); AdamW (lr 1e-3), batch 512, MSE on the standardised
log-RV target (vol) or BCE (direction); ensemble of N_SEEDS = 3 seeds (mean of log forecasts / of probabilities).
XGB: hist trees, eta 0.03, subsample 0.8, colsample 0.8, lambda 1; reg:squarederror on log RV or binary:logistic.

Hyperparameter search (in-sample only, before the first test month, identical effort for both families): 4 configs
each, MLP hidden in {(64, 32), (32, 16)} x weight decay in {1e-4, 1e-2}; XGB max_depth in {3, 6} x min_child_weight in
{20, 200}. Search data = rows with decision dates before the first walk-forward test month (never a test row),
split 80/20 by date (purged); criterion = validation MSE (log RV) or log loss (direction). The chosen config is then
frozen for the whole walk-forward. Defaults (used when no search is run): MLP (64, 32), wd 1e-4; XGB depth 4, mcw 50.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field

import numpy as np

VAL_FRAC = 0.15
N_SEEDS = 3
TORCH_THREADS = 4

MLP_DEFAULT = {"hidden": (64, 32), "lr": 1e-3, "weight_decay": 1e-4, "batch": 512, "max_epochs": 200, "patience": 10}
XGB_DEFAULT = {"eta": 0.03, "max_depth": 4, "min_child_weight": 50, "subsample": 0.8, "colsample_bytree": 0.8,
               "lambda": 1.0, "max_rounds": 2000, "early_stop": 50}
MLP_GRID = [{"hidden": h, "weight_decay": wd} for h in ((64, 32), (32, 16)) for wd in (1e-4, 1e-2)]
XGB_GRID = [{"max_depth": d, "min_child_weight": m} for d in (3, 6) for m in (20, 200)]


def _torch():
    import torch

    torch.set_num_threads(TORCH_THREADS)
    return torch


# ------------------------------------------------------------------ splits


def val_split(T: np.ndarray, frac: float = VAL_FRAC) -> tuple[np.ndarray, np.ndarray]:
    """(fit mask, val mask) over training rows by decision date; fit labels (24h) never reach the validation block."""
    days = np.unique(T.astype("datetime64[D]"))
    k = max(1, int(round(len(days) * frac)))
    v0 = days[-k].astype("datetime64[us]")
    va = T >= v0
    fit = T < v0 - np.timedelta64(86400 * 1_000_000, "us")
    return fit, va


# ------------------------------------------------------------------ MLP


def make_mlp(n_in: int, hidden=(64, 32)):
    torch = _torch()
    nn = torch.nn

    class MLP(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            dims = [n_in, *hidden]
            self.layers = nn.ModuleList([nn.Linear(a, b) for a, b in zip(dims[:-1], dims[1:], strict=True)])
            self.head = nn.Linear(dims[-1], 1)

        def hidden_acts(self, x):
            hs = []
            z = x
            for lin in self.layers:
                z = torch.relu(lin(z))
                hs.append(z)
            return hs

        def forward(self, x):
            z = x
            for lin in self.layers:
                z = torch.relu(lin(z))
            return self.head(z).squeeze(-1)

    return MLP()


@dataclass
class MLPFit:
    state: dict
    hidden: tuple
    n_in: int
    y_mu: float
    y_sd: float
    task: str
    best_val: float
    epochs: int

    def model(self):
        m = make_mlp(self.n_in, self.hidden)
        m.load_state_dict(self.state)
        m.eval()
        return m

    def predict(self, Z: np.ndarray) -> np.ndarray:
        """Log-RV forecast (vol) or probability (dir) from standardised inputs Z."""
        torch = _torch()
        with torch.no_grad():
            out = self.model()(torch.as_tensor(np.asarray(Z, dtype=np.float32))).numpy().astype(float)
        if self.task == "vol":
            return out * self.y_sd + self.y_mu
        return 1.0 / (1.0 + np.exp(-out))


def fit_mlp(Zf: np.ndarray, yf: np.ndarray, Zv: np.ndarray, yv: np.ndarray, task: str, seed: int, cfg: dict | None = None) -> MLPFit:
    """Train one MLP with early stopping on (Zv, yv). Deterministic given the seed (CPU)."""
    torch = _torch()
    c = {**MLP_DEFAULT, **(cfg or {})}
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
    m = make_mlp(Xf.shape[1], tuple(c["hidden"]))
    if task == "dir":  # start at the base rate
        with torch.no_grad():
            p = float(np.clip(yf.mean(), 1e-3, 1 - 1e-3))
            m.head.bias.fill_(math.log(p / (1 - p)))
    opt = torch.optim.AdamW(m.parameters(), lr=c["lr"], weight_decay=c["weight_decay"])
    best, best_state, best_ep, bad = float("inf"), copy.deepcopy(m.state_dict()), 0, 0
    with torch.no_grad():
        best = float(lossf(m(Xv), Yv))
    n = Xf.shape[0]
    ep = 0
    for ep in range(1, c["max_epochs"] + 1):
        m.train()
        perm = torch.randperm(n, generator=g)
        for i in range(0, n, c["batch"]):
            idx = perm[i:i + c["batch"]]
            opt.zero_grad()
            loss = lossf(m(Xf[idx]), Yf[idx])
            loss.backward()
            opt.step()
        m.eval()
        with torch.no_grad():
            v = float(lossf(m(Xv), Yv))
        if v < best - 1e-6:
            best, best_state, best_ep, bad = v, copy.deepcopy(m.state_dict()), ep, 0
        else:
            bad += 1
            if bad >= c["patience"]:
                break
    return MLPFit(best_state, tuple(c["hidden"]), Xf.shape[1], y_mu, y_sd, task, best, best_ep)


@dataclass
class MLPEnsemble:
    members: list = field(default_factory=list)

    def predict(self, Z: np.ndarray) -> np.ndarray:
        return np.mean([m.predict(Z) for m in self.members], axis=0)


def fit_mlp_ensemble(Zf, yf, Zv, yv, task: str, seeds=range(N_SEEDS), cfg: dict | None = None) -> MLPEnsemble:
    return MLPEnsemble([fit_mlp(Zf, yf, Zv, yv, task, s, cfg) for s in seeds])


# ------------------------------------------------------------------ XGBoost


@dataclass
class XGBFit:
    booster: object
    best_iteration: int
    task: str

    def dmatrix(self, Z: np.ndarray, names=None):
        import xgboost as xgb

        return xgb.DMatrix(np.asarray(Z, dtype=np.float32), feature_names=list(names) if names is not None else None)

    def predict(self, Z: np.ndarray, **kw) -> np.ndarray:
        return self.booster.predict(self.dmatrix(Z, self.booster.feature_names), iteration_range=(0, self.best_iteration + 1), **kw)


def fit_xgb(Zf, yf, Zv, yv, task: str, cfg: dict | None = None, seed: int = 0, nthread: int = 4, names=None) -> XGBFit:
    import xgboost as xgb

    c = {**XGB_DEFAULT, **(cfg or {})}
    params = {"eta": c["eta"], "max_depth": c["max_depth"], "min_child_weight": c["min_child_weight"], "subsample": c["subsample"],
              "colsample_bytree": c["colsample_bytree"], "lambda": c["lambda"], "tree_method": "hist", "nthread": nthread, "seed": seed,
              "objective": "reg:squarederror" if task == "vol" else "binary:logistic",
              "eval_metric": "rmse" if task == "vol" else "logloss"}
    if task == "dir":
        params["base_score"] = float(np.clip(np.mean(yf), 1e-3, 1 - 1e-3))
    names = list(names) if names is not None else None
    df = xgb.DMatrix(np.asarray(Zf, dtype=np.float32), label=np.asarray(yf, dtype=np.float32), feature_names=names)
    dv = xgb.DMatrix(np.asarray(Zv, dtype=np.float32), label=np.asarray(yv, dtype=np.float32), feature_names=names)
    b = xgb.train(params, df, num_boost_round=c["max_rounds"], evals=[(dv, "val")], early_stopping_rounds=c["early_stop"], verbose_eval=False)
    return XGBFit(b, int(b.best_iteration), task)


# ------------------------------------------------------------------ linear reference (same inputs)


@dataclass
class LinFit:
    coef: np.ndarray
    task: str

    def predict(self, Z: np.ndarray) -> np.ndarray:
        f = self.coef[0] + np.asarray(Z, dtype=float) @ self.coef[1:]
        return f if self.task == "vol" else 1.0 / (1.0 + np.exp(-np.clip(f, -35, 35)))


def fit_linear(Z: np.ndarray, y: np.ndarray, task: str, ridge: float = 1.0) -> LinFit:
    """Ridge OLS on log RV (vol) or ridge logistic (dir; decision_budget's fit_logit), intercept unpenalised."""
    Z = np.asarray(Z, dtype=float)
    y = np.asarray(y, dtype=float)
    if task == "vol":
        A = np.c_[np.ones(len(y)), Z]
        P = ridge * np.eye(A.shape[1])
        P[0, 0] = 0.0
        return LinFit(np.linalg.solve(A.T @ A + P, A.T @ y), task)
    from ai_trader.research.regime_switch_study import fit_logit

    b, _ = fit_logit(Z, y, ridge=ridge)
    return LinFit(np.asarray(b, dtype=float), task)


# ------------------------------------------------------------------ bias correction and losses


def bias_const(rv: np.ndarray, logf: np.ndarray) -> float:
    """c with mean(RV / exp(f + c)) = 1 (QLIKE-optimal scale of a log forecast), from in-sample rows."""
    rv = np.asarray(rv, dtype=float)
    logf = np.asarray(logf, dtype=float)
    ok = np.isfinite(rv) & np.isfinite(logf) & (rv > 0)
    return float(np.log(np.mean(rv[ok] / np.exp(logf[ok])))) if ok.sum() else 0.0


def qlike(rv: np.ndarray, f: np.ndarray) -> np.ndarray:
    rv = np.asarray(rv, dtype=float)
    f = np.asarray(f, dtype=float)
    with np.errstate(all="ignore"):
        r = rv / f
        out = r - np.log(r) - 1.0
    return np.where(np.isfinite(out) & (f > 0) & (rv > 0), out, np.nan)


def logloss(y: np.ndarray, p: np.ndarray) -> float:
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def search(Z: np.ndarray, y: np.ndarray, T: np.ndarray, rv: np.ndarray | None, task: str, family: str) -> dict:
    """In-sample-only config search (see module docstring). Returns {'best': cfg, 'scores': [...]}."""
    fit, va = val_split(T, 0.2)
    scores = []
    grid = MLP_GRID if family == "mlp" else XGB_GRID
    for cfg in grid:
        if family == "mlp":
            m = fit_mlp_ensemble(Z[fit], y[fit], Z[va], y[va], task, seeds=(0,), cfg=cfg)
            p = m.predict(Z[va])
        else:
            m = fit_xgb(Z[fit], y[fit], Z[va], y[va], task, cfg=cfg)
            p = m.predict(Z[va])
        s = float(np.mean((p - y[va]) ** 2)) if task == "vol" else logloss(y[va], p)
        scores.append({"cfg": {k: list(v) if isinstance(v, tuple) else v for k, v in cfg.items()}, "val_loss": s})
    best = min(range(len(grid)), key=lambda i: scores[i]["val_loss"])
    return {"best": grid[best], "scores": scores}


__all__ = ["MLPEnsemble", "MLPFit", "MLP_DEFAULT", "MLP_GRID", "N_SEEDS", "VAL_FRAC", "XGBFit", "XGB_DEFAULT", "XGB_GRID",
           "LinFit", "bias_const", "fit_linear", "fit_mlp", "fit_mlp_ensemble", "fit_xgb", "logloss", "make_mlp", "qlike", "search", "val_split"]
