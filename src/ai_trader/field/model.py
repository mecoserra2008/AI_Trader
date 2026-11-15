"""The fitted BTC probability field: state construction parameters + three sparse polynomials + the Kou jump law.

One-step (1h) law of the next log return given the state X_t (known at the close of bar t):
    D(X)   = vf_t * exp(P_D(X~))                       diffusion (variance per hour)
    mu(X)  = sqrt(D(X)) * P_mu(X~)                     drift (log return per hour)
    p(X)   = min(exp(P_lam(X~)), 0.5)                  jump probability per hour (lambda dt)
    r      = mu + sqrt(D) * (e + 1[jump] K),  K ~ Kou(p_up, eta_up, eta_down) in sd units (mean size >= 3 sd)
             e ~ unit-variance Student-t(t_dof) off the jump branch (core 't': the KM diffusion integrated over
             unobserved intra-hour vol), N(0,1) on the jump branch; core 'gauss': e ~ N(0,1) throughout
JSON schema (data/models/field/*.json) is documented in `Field.to_dict` and in the report.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import polars as pl

from ai_trader.field.kramers_moyal import (
    EMFit,
    Kou,
    components,
    mixture_cdf,
    mixture_logpdf,
)
from ai_trader.field.poly import SparsePoly, Standardizer, design
from ai_trader.field.state import COORDS, StateConfig

SCHEMA_VERSION = "field.v1"


@dataclass
class Field:
    std: Standardizer
    mu: SparsePoly
    logD: SparsePoly
    loglam: SparsePoly
    kou: Kou
    profile: np.ndarray
    cfg: StateConfig = field(default_factory=StateConfig)
    t_dof: float = float("nan")
    meta: dict = field(default_factory=dict)
    vol_field: dict = field(default_factory=dict)
    core: str = "t"   # non-jump branch: unit-variance Student-t with t_dof ('t') or Gaussian ('gauss')

    def nu(self, core: str | None = None):
        c = core or self.core
        return self.t_dof if (c == "t" and np.isfinite(self.t_dof)) else None

    # ---------------------------------------------------------- evaluation
    def Xs(self, X: np.ndarray) -> np.ndarray:
        return self.std(X)

    def designs(self, Xs: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return design(Xs, self.mu.monos), design(Xs, self.logD.monos), design(Xs, self.loglam.monos)

    def params(self, X: np.ndarray, lnvf: np.ndarray, coefs=None, drift: bool = True):
        """(mu, D, p) per row of raw coordinates X (n, 9) with offsets ln vf."""
        Ts = self.designs(self.Xs(X))
        c = coefs or (self.mu.coef, self.logD.coef, self.loglam.coef)
        mu, D, p = components(Ts, c, lnvf)
        return (mu if drift else np.zeros_like(mu)), D, p

    def logpdf(self, r, X, lnvf, *, drift: bool = True, core: str | None = None, coefs=None) -> np.ndarray:
        mu, D, p = self.params(X, lnvf, coefs, drift)
        return mixture_logpdf(np.asarray(r, float), mu, D, p, self.kou, self.nu(core))

    def cdf(self, r, X, lnvf, *, drift: bool = True, core: str | None = None, coefs=None) -> np.ndarray:
        mu, D, p = self.params(X, lnvf, coefs, drift)
        return mixture_cdf(np.asarray(r, float), mu, D, p, self.kou, self.nu(core))

    # ---------------------------------------------------------- io
    @classmethod
    def from_em(cls, em: EMFit, std: Standardizer, profile, cfg: StateConfig, meta: dict | None = None) -> Field:
        return cls(std, em.mu, em.logD, em.loglam, em.kou, np.asarray(profile, float), cfg, em.t_dof, meta or {}, {}, em.core)

    def to_dict(self) -> dict:
        return {
            "schema": SCHEMA_VERSION,
            "symbol": "BTCUSDT", "venue": "bybit", "freq": "1h",
            "meta": self.meta,
            "state": {"config": self.cfg.to_dict(), "coords": list(COORDS), "standardizer": self.std.to_dict(),
                      "hour_of_week_profile": np.asarray(self.profile).tolist()},
            "law": "r = mu + sqrt(D) (e + 1[jump] K); D = vf exp(P_D); mu = sqrt(D) P_mu; p = min(exp(P_lam), 0.5); K ~ Kou (sd units); "
                   "e ~ unit-variance Student-t(t_dof) off the jump branch if core == 't', N(0,1) on it (and everywhere if core == 'gauss')",
            "core": self.core,
            "components": {"mu": self.mu.to_dict(), "logD": self.logD.to_dict(), "loglam": self.loglam.to_dict()},
            "kou": self.kou.to_dict(),
            "t_dof": self.t_dof,
            "vol_field": self.vol_field,
        }

    @classmethod
    def from_dict(cls, d: dict) -> Field:
        st = d["state"]
        k = d["kou"]
        return cls(Standardizer.from_dict(st["standardizer"]), SparsePoly.from_dict(d["components"]["mu"]),
                   SparsePoly.from_dict(d["components"]["logD"]), SparsePoly.from_dict(d["components"]["loglam"]),
                   Kou(k["p_up"], k["eta_up"], k["eta_down"]), np.asarray(st["hour_of_week_profile"], float), StateConfig(**st["config"]),
                   float(d["t_dof"]) if d.get("t_dof") is not None else float("nan"), d.get("meta", {}), d.get("vol_field", {}),
                   d.get("core", "t"))

    def save(self, path: Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.to_dict(), indent=1, default=float), encoding="utf-8")
        tmp.replace(path)
        return path

    @classmethod
    def load(cls, path: Path) -> Field:
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def coords_matrix(state: pl.DataFrame) -> np.ndarray:
    return np.column_stack([state[c].to_numpy().astype(float) for c in COORDS])


__all__ = ["SCHEMA_VERSION", "Field", "coords_matrix"]
