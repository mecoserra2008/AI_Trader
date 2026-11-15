"""vecopen test configuration.

Prints the baseline config hash in the pytest header, and guards the real trial registry:
no test may modify ``vecopen/trials/registry.jsonl`` or its head anchor (tests use tmp paths).
"""

from __future__ import annotations

import pytest

from vecopen import load_config, registry


def pytest_report_header(config):
    cfg = load_config()
    return [f"vecopen config {cfg.path.name} sha256: {cfg.sha256}"]


def _snapshot():
    out = {}
    for p in (registry.REGISTRY_PATH, registry.head_path(registry.REGISTRY_PATH)):
        out[p] = p.read_bytes() if p.exists() else None
    return out


@pytest.fixture(scope="session", autouse=True)
def _real_registry_untouched():
    before = _snapshot()
    yield
    assert _snapshot() == before, "a test modified the real trial registry"


@pytest.fixture(scope="session")
def cfg():
    return load_config()
