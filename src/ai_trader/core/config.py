"""Config loading shared with the Rust core: same YAML files, same `.env`, same overrides."""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv


def find_root() -> Path:
    """`AI_TRADER_ROOT`, else the first ancestor of the cwd or this file containing config/settings.yaml."""
    if env := os.environ.get("AI_TRADER_ROOT"):
        return Path(env)
    for start in (Path.cwd(), Path(__file__).resolve()):
        for d in (start, *start.parents):
            if (d / "config" / "settings.yaml").is_file():
                return d
    raise FileNotFoundError("config/settings.yaml not found in any parent directory")


def _yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


@dataclass(frozen=True)
class Config:
    root: Path
    settings: dict[str, Any]
    risk: dict[str, Any]
    schema: dict[str, Any]
    datasources: dict[str, Any]
    tools: dict[str, Any]
    strategies: dict[str, Any]
    hypotheses: dict[str, Any]

    @property
    def environment(self) -> str:
        return os.environ.get("AI_TRADER_ENVIRONMENT", self.settings["environment"])

    def path(self, key: str) -> Path:
        p = Path(self.settings["paths"][key])
        return p if p.is_absolute() else self.root / p

    @property
    def lake_dir(self) -> Path:
        return self.path("lake")


@lru_cache(maxsize=4)
def load(root: Path | None = None) -> Config:
    root = Path(root) if root else find_root()
    load_dotenv(root / ".env", override=False)
    c = root / "config"
    return Config(
        root=root,
        settings=_yaml(c / "settings.yaml"),
        risk=_yaml(c / "risk.yaml"),
        schema=_yaml(c / "schema.yaml"),
        datasources=_yaml(c / "datasources.yaml"),
        tools=_yaml(c / "tools.yaml"),
        strategies=_yaml(c / "strategies.yaml") if (c / "strategies.yaml").is_file() else {"strategies": {}},
        hypotheses=_yaml(root / "research" / "hypotheses.yaml") if (root / "research" / "hypotheses.yaml").is_file() else {"hypotheses": {}},
    )


def secret(name: str) -> str | None:
    v = os.environ.get(name, "").strip()
    return v or None
