"""Deterministic strategy library. Strategies are signal generators: they never size positions
(the risk engine does) and they are fully backtestable because they are deterministic.
Importing this package registers every strategy module."""

from .base import REGISTRY, Strategy, StrategySpec, all_strategies, assert_causal, register  # noqa: F401

# Strategy modules register themselves on import.
from . import calendar, flow, mean_reversion, positioning, shock, structure, trend, volatility  # noqa: F401,E402
