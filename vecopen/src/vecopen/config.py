"""Frozen configuration loading and hashing (SPEC section 6, overfitting control 1; section 7).

``load_config()`` returns the parsed ``vecopen/config/baseline.yaml`` together with its
sha256. The hash is taken over the file bytes with line endings normalised to LF, so a
Windows checkout (``core.autocrlf=true``) and a Linux checkout of the same committed file
give the same hash. The file itself is read-only for agents (standing rule 1).
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, NamedTuple

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]  # .../vecopen
CONFIG_PATH = PROJECT_ROOT / "config" / "baseline.yaml"

# Top-level sections the baseline must define; a missing one is a malformed config.
REQUIRED_SECTIONS = (
    "version", "seed", "legs", "levels", "labels", "events", "costs", "split",
    "robustness_grid", "stats", "cv", "gate_g3", "assets",
)


class LoadedConfig(NamedTuple):
    """Parsed config, the sha256 of its (LF-normalised) bytes, and the file it came from."""

    data: dict[str, Any]
    sha256: str
    path: Path


def config_sha256(raw: bytes) -> str:
    """sha256 hex digest of config bytes, with CRLF/CR normalised to LF."""
    normalised = raw.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    return hashlib.sha256(normalised).hexdigest()


def load_config(path: str | Path | None = None) -> LoadedConfig:
    """Load and hash the baseline config (default ``vecopen/config/baseline.yaml``).

    The hash and the parsed data come from the same single read, so they cannot disagree.
    Raises ``ValueError`` if the file is not a mapping or lacks a required section.
    """
    p = Path(path) if path is not None else CONFIG_PATH
    raw = p.read_bytes()
    data = yaml.safe_load(raw.decode("utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{p}: config must be a YAML mapping, got {type(data).__name__}")
    missing = [k for k in REQUIRED_SECTIONS if k not in data]
    if missing:
        raise ValueError(f"{p}: missing required config sections {missing}")
    if not isinstance(data["seed"], int):
        raise ValueError(f"{p}: seed must be an integer (standing rule 8)")
    return LoadedConfig(data=data, sha256=config_sha256(raw), path=p)
