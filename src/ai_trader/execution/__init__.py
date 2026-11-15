"""Strategy execution pipeline, Python side (agent K; contract: docs/EXECUTION_PIPELINE.md).

    strategies (REGISTRY, point-in-time signals)          signals.py
      -> bias gate (playbook rules + vol + calendar + placement)  gate.py
      -> OrderIntent JSONL (never a size)                  intents.py
      -> `atx exec-intents` (Rust risk engine + PaperBroker, agent L) -> RiskDecision JSONL   runner.py

Research / paper only: the runner refuses any mode other than `replay` and `paper` and refuses to start when the
configuration points at mainnet. Nothing in this package talks to an exchange.
"""

from .gate import BiasGate, GateResult, Placement  # noqa: F401
from .intents import build_intent, validate_intent, write_intents  # noqa: F401
from .signals import SignalEvent, StrategyRef, strategy_refs  # noqa: F401
