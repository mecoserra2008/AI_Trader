"""Strategy contract.

A strategy maps a point-in-time market frame (ai_trader.research.frame.market_frame) to a signal
in [-1, 1] per row: +1 = full long conviction, -1 = full short, 0 = flat. The value at row t may
use **only rows <= t** and never label columns (`fwd_ret`). Evaluation applies signal_t to the
next bar's return, so a signal is always traded after it is known.

Parameters are pre-registered in config/strategies.yaml (no parameter search on the full sample).
Every strategy states a falsifiable `hypothesis` and the regime cells where it is expected to
work (`expected_regimes`): this is the **prior** that evaluation then confirms or rejects.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import polars as pl

from ai_trader.core.config import load

FAMILIES = ("trend", "mean_reversion", "volatility", "positioning", "flow", "calendar", "shock")
CELLS = tuple(f"R{i}" for i in range(1, 10))


@dataclass(frozen=True)
class StrategySpec:
    name: str
    version: str
    family: str
    description: str
    hypothesis: str
    requires: tuple[str, ...] = ("bars",)  # column groups: bars | funding | oi | flow | calendar
    expected_regimes: tuple[str, ...] = ()
    params: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.family not in FAMILIES:
            raise ValueError(f"{self.name}: unknown family {self.family!r}")
        bad = set(self.expected_regimes) - set(CELLS)
        if bad:
            raise ValueError(f"{self.name}: unknown regime cells {bad}")


class Strategy(ABC):
    spec: StrategySpec

    def __init__(self, **overrides: Any) -> None:
        cfg = (load().strategies.get("strategies") or {}).get(self.spec.name) or {}
        self.params: dict[str, Any] = {**self.spec.params, **cfg.get("params", {}), **overrides}

    @abstractmethod
    def signal(self, frame: pl.DataFrame) -> pl.Series:
        """Signal in [-1, 1], same length/order as `frame`. Row t uses rows <= t only."""

    def __call__(self, frame: pl.DataFrame) -> pl.Series:
        for col in ("fwd_ret",):
            if col in frame.columns:
                frame = frame.drop(col)  # labels are never visible to strategies
        s = self.signal(frame)
        if len(s) != frame.height:
            raise ValueError(f"{self.spec.name}: signal length {len(s)} != frame height {frame.height}")
        return s.cast(pl.Float64).fill_nan(None).fill_null(0.0).clip(-1.0, 1.0).alias(self.spec.name)


REGISTRY: dict[str, type[Strategy]] = {}


def register(cls: type[Strategy]) -> type[Strategy]:
    name = cls.spec.name
    if name in REGISTRY:
        raise KeyError(f"strategy {name} registered twice")
    REGISTRY[name] = cls
    return cls


def all_strategies(**kwargs: Any) -> list[Strategy]:
    return [cls(**kwargs) for cls in REGISTRY.values()]


def assert_causal(strategy: Strategy, frame: pl.DataFrame, cut_points: int = 5, seed: int = 0) -> None:
    """Signal on a truncated frame must equal the full-frame signal on the same rows.
    Catches centred windows, negative shifts, full-sample normalisation and label use."""
    full = strategy(frame).to_numpy()
    rng = np.random.default_rng(seed)
    n = frame.height
    for k in sorted(rng.integers(max(10, n // 4), n, size=cut_points)):
        part = strategy(frame.head(int(k))).to_numpy()
        if not np.allclose(part, full[: int(k)], equal_nan=True, atol=1e-10):
            i = int(np.argmax(~np.isclose(part, full[: int(k)], equal_nan=True, atol=1e-10)))
            raise AssertionError(f"{strategy.spec.name} is not causal: row {i} differs when data after row {k} is removed")
