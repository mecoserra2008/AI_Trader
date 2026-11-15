"""Phase 0 scaffold: package layout of SPEC section 8."""

from __future__ import annotations

import importlib
import shutil
import subprocess

import pytest

from vecopen.config import PROJECT_ROOT

MODULES = ["data", "bars", "legs", "levels", "events", "labels", "placebo", "features", "context",
           "stats", "cv", "models", "backtest", "registry"]


@pytest.mark.parametrize("name", MODULES)
def test_module_importable_with_docstring(name):
    mod = importlib.import_module(f"vecopen.{name}")
    assert mod.__doc__ and len(mod.__doc__.strip()) > 40


def test_layout_directories_exist():
    for d in ("src/vecopen", "tests", "trials", "reports", "data", "config"):
        assert (PROJECT_ROOT / d).is_dir(), d
    assert (PROJECT_ROOT / "SPEC.md").is_file()
    assert (PROJECT_ROOT / "CLAUDE.md").is_file()


@pytest.mark.skipif(shutil.which("git") is None, reason="git not installed")
@pytest.mark.parametrize("rel", ["vecopen/data/research/probe.parquet", "vecopen/data/holdout/probe.parquet",
                                 "vecopen/data/raw/probe.zip"])
def test_data_is_git_ignored(rel):
    # check-ignore only matches the path pattern; it does not read or list data/holdout/.
    out = subprocess.run(["git", "check-ignore", "-q", "--no-index", rel], cwd=PROJECT_ROOT.parent)
    assert out.returncode == 0, f"{rel} is not git-ignored"


def test_thread_limit_variables_set_on_import():
    import os
    for var in ("POLARS_MAX_THREADS", "OMP_NUM_THREADS"):
        assert os.environ.get(var), var  # set to "2" by vecopen/__init__ unless preset by the caller
