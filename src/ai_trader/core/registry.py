"""Tool registry. Python implementations register with @tool; config/tools.yaml is the catalogue
(status, cadence, where it is implemented). A registered tool must exist in the catalogue."""

from __future__ import annotations

import inspect
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from functools import wraps

from .config import load
from .schemas import ToolResult
from .time import utc


@dataclass(frozen=True)
class ToolSpec:
    name: str
    version: str
    fn: Callable[..., ToolResult]

    @property
    def catalogue(self) -> dict:
        return load().tools["tools"][self.name]

    @property
    def status(self) -> str:
        return self.catalogue["status"]


_REGISTRY: dict[str, ToolSpec] = {}


def tool(name: str, version: str):
    """Register a tool. Its signature must include a keyword `as_of`; the output is a ToolResult
    whose name/version must match the registration (enforced on every call)."""

    def deco(fn: Callable[..., ToolResult]):
        if "as_of" not in inspect.signature(fn).parameters:
            raise TypeError(f"tool {name} must take an `as_of` argument")
        if name not in load().tools["tools"]:
            raise KeyError(f"tool {name} is not in config/tools.yaml")

        @wraps(fn)
        def wrapper(*args, as_of: datetime, **kwargs) -> ToolResult:
            out = fn(*args, as_of=utc(as_of), **kwargs)
            if not isinstance(out, ToolResult) or (out.tool, out.version) != (name, version):
                raise TypeError(f"tool {name} must return ToolResult(tool={name!r}, version={version!r})")
            return out

        _REGISTRY[name] = ToolSpec(name, version, wrapper)
        return wrapper

    return deco


def get(name: str) -> ToolSpec:
    return _REGISTRY[name]


def registered() -> dict[str, ToolSpec]:
    return dict(_REGISTRY)
