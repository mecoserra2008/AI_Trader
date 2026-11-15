"""registry.append() / verify(): append-only, hash-chained trial registry (phase 0)."""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from vecopen import registry
from vecopen.registry import GENESIS, RegistryError, append, head_path, line_hash, verify

T0 = datetime(2026, 11, 14, 12, 0, tzinfo=UTC)


def _fill(path: Path, cfg, n: int, repo_root: Path) -> list[dict]:
    return [append(0, {"tests": i, "x": 0.5 * i}, kind="scaffold", note=f"line {i}", config=cfg,
                   path=path, repo_root=repo_root, now=T0 + timedelta(minutes=i)) for i in range(n)]


@pytest.fixture
def reg(tmp_path, cfg):
    path = tmp_path / "trials" / "registry.jsonl"
    recs = _fill(path, cfg, 4, tmp_path)
    return path, recs


def _lines(path: Path) -> list[bytes]:
    return path.read_bytes().split(b"\n")[:-1]


def _write_lines(path: Path, lines: list[bytes]) -> None:
    path.write_bytes(b"".join(x + b"\n" for x in lines))


# ---------------------------------------------------------------- happy path

def test_empty_registry_verifies(tmp_path):
    res = verify(tmp_path / "none.jsonl")
    assert res.ok and res.n_lines == 0 and res.last_sha256 == GENESIS


def test_append_writes_one_canonical_line_with_required_fields(tmp_path, cfg):
    path = tmp_path / "registry.jsonl"
    rec = append(0, {"tests": 7}, kind="scaffold", config=cfg, path=path, repo_root=tmp_path, now=T0)
    lines = _lines(path)
    assert len(lines) == 1
    stored = json.loads(lines[0])
    assert stored == rec
    for key in ("config_sha256", "git_commit", "utc", "phase", "metrics", "prev_sha256", "line_sha256"):
        assert key in stored
    assert stored["config_sha256"] == cfg.sha256
    assert stored["phase"] == 0 and stored["metrics"] == {"tests": 7} and stored["seq"] == 0
    assert stored["prev_sha256"] == GENESIS
    assert stored["utc"] == "2026-11-14T12:00:00.000000Z"
    assert stored["git_commit"] == "uncommitted" and len(stored["diff_sha256"]) == 64  # tmp dir: no git
    assert line_hash(stored) == stored["line_sha256"]
    assert verify(path).ok


def test_chain_links_each_line_to_the_previous(reg):
    path, recs = reg
    assert [r["seq"] for r in recs] == [0, 1, 2, 3]
    assert recs[0]["prev_sha256"] == GENESIS
    for a, b in zip(recs, recs[1:]):
        assert b["prev_sha256"] == a["line_sha256"]
    res = verify(path)
    assert res.ok, res.errors
    assert res.n_lines == 4 and res.last_sha256 == recs[-1]["line_sha256"]
    head = json.loads(head_path(path).read_text())
    assert head == {"n": 4, "last_sha256": recs[-1]["line_sha256"]}
    assert registry.read(path) == recs


def test_utc_is_normalised_from_other_offsets(tmp_path, cfg):
    t = datetime(2026, 11, 14, 13, 0, tzinfo=timezone(timedelta(hours=1)))
    rec = append(0, {}, config=cfg, path=tmp_path / "r.jsonl", repo_root=tmp_path, now=t)
    assert rec["utc"] == "2026-11-14T12:00:00.000000Z"
    with pytest.raises(ValueError):
        append(0, {}, config=cfg, path=tmp_path / "r2.jsonl", repo_root=tmp_path, now=datetime(2026, 1, 1))


# ---------------------------------------------------------------- tamper detection

def _assert_detected(path: Path, needle: str | None = None) -> list[str]:
    res = verify(path)
    assert not res.ok
    if needle:
        assert any(needle in e for e in res.errors), res.errors
    return res.errors


def test_edit_of_past_value_detected(reg):
    path, _ = reg
    lines = _lines(path)
    lines[1] = lines[1].replace(b'"tests":1', b'"tests":9')
    _write_lines(path, lines)
    _assert_detected(path, "line 2: line_sha256 does not match")


def test_edit_with_recomputed_line_hash_breaks_chain(reg):
    path, _ = reg
    lines = _lines(path)
    rec = json.loads(lines[1])
    rec["metrics"]["tests"] = 99
    rec["line_sha256"] = line_hash(rec)
    lines[1] = registry._canonical(rec).encode()
    _write_lines(path, lines)
    _assert_detected(path, "line 3: prev_sha256 does not match")


def test_reformatting_a_line_detected(reg):
    path, _ = reg
    lines = _lines(path)
    lines[0] = json.dumps(json.loads(lines[0]), indent=None).encode()  # same content, spaces added
    _write_lines(path, lines)
    _assert_detected(path, "canonical")


@pytest.mark.parametrize("idx", [0, 1, 3])
def test_deleting_any_line_detected(reg, idx):
    path, _ = reg
    lines = _lines(path)
    del lines[idx]
    _write_lines(path, lines)
    _assert_detected(path)


def test_deleting_last_line_detected_by_head_anchor(reg):
    path, _ = reg
    _write_lines(path, _lines(path)[:-1])
    errors = _assert_detected(path, "head anchor says 4 lines, file has 3")
    assert all(e.startswith("head anchor") for e in errors)  # the chain itself is intact


def test_swapping_lines_detected(reg):
    path, _ = reg
    lines = _lines(path)
    lines[1], lines[2] = lines[2], lines[1]
    _write_lines(path, lines)
    _assert_detected(path, "seq=")


def test_inserting_a_forged_line_detected(reg):
    path, recs = reg
    lines = _lines(path)
    lines.insert(2, lines[1])
    _write_lines(path, lines)
    _assert_detected(path)


def test_truncation_and_file_deletion_detected(reg):
    path, _ = reg
    raw = path.read_bytes()
    path.write_bytes(raw[:-10])
    _assert_detected(path, "newline")
    path.unlink()
    _assert_detected(path, "missing")


def test_missing_or_corrupt_head_detected(reg):
    path, _ = reg
    hp = head_path(path)
    hp.write_text("not json")
    _assert_detected(path, "unreadable")
    hp.unlink()
    _assert_detected(path, "head anchor registry.head.json is missing")


def test_append_refuses_to_extend_a_tampered_registry(reg, cfg, tmp_path):
    path, _ = reg
    lines = _lines(path)
    lines[0] = lines[0].replace(b'"tests":0', b'"tests":5')
    _write_lines(path, lines)
    before = path.read_bytes(), head_path(path).read_bytes()
    with pytest.raises(RegistryError, match="failed verification"):
        append(0, {"tests": 1}, config=cfg, path=path, repo_root=tmp_path)
    assert (path.read_bytes(), head_path(path).read_bytes()) == before


@pytest.mark.parametrize("metrics, exc", [
    ({"x": float("nan")}, ValueError),
    ({"x": float("inf")}, ValueError),
    ({"x": {"nested": float("nan")}}, ValueError),
    ([("x", 1)], TypeError),
    ({1: 2}, TypeError),
    ({"x": object()}, TypeError),
])
def test_bad_metrics_rejected_and_nothing_written(tmp_path, cfg, metrics, exc):
    path = tmp_path / "r.jsonl"
    with pytest.raises(exc):
        append(0, metrics, config=cfg, path=path, repo_root=tmp_path)
    assert not path.exists() and not head_path(path).exists()


@pytest.mark.parametrize("phase", [-1, 11, 1.0, True, "0"])
def test_bad_phase_rejected(tmp_path, cfg, phase):
    with pytest.raises(ValueError):
        append(phase, {}, config=cfg, path=tmp_path / "r.jsonl", repo_root=tmp_path)


@pytest.fixture(scope="module")
def reg_bytes(tmp_path_factory):
    from vecopen import load_config
    d = tmp_path_factory.mktemp("prop")
    path = d / "registry.jsonl"
    _fill(path, load_config(), 3, d)
    return path.read_bytes(), head_path(path).read_bytes()


@settings(max_examples=150, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(data=st.data())
def test_any_single_byte_change_is_detected(reg_bytes, data):
    raw, head = reg_bytes
    i = data.draw(st.integers(0, len(raw) - 1), label="position")
    b = data.draw(st.integers(0, 255).filter(lambda v: v != raw[i]), label="new byte")
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "registry.jsonl"
        path.write_bytes(raw[:i] + bytes([b]) + raw[i + 1:])
        head_path(path).write_bytes(head)
        assert not verify(path).ok


@settings(max_examples=25, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(metrics=st.lists(st.dictionaries(
    st.text(min_size=1, max_size=8),
    st.one_of(st.integers(-10**12, 10**12), st.floats(allow_nan=False, allow_infinity=False),
              st.text(max_size=12), st.booleans(), st.none()),
    max_size=4), min_size=1, max_size=4))
def test_arbitrary_metrics_roundtrip_and_verify(cfg, metrics):
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "registry.jsonl"
        recs = [append(1, m, config=cfg, path=path, repo_root=td) for m in metrics]
        assert verify(path).ok
        assert [r["metrics"] for r in registry.read(path)] == metrics == [r["metrics"] for r in recs]


# ---------------------------------------------------------------- git identity

needs_git = pytest.mark.skipif(shutil.which("git") is None, reason="git not installed")


def _git(cwd: Path, *args: str) -> str:
    return subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@example.invalid",
                           "-c", "commit.gpgsign=false", *args],
                          cwd=cwd, check=True, capture_output=True, text=True).stdout.strip()


@needs_git
def test_git_state_commit_dirty_and_untracked(tmp_path):
    repo = tmp_path / "repo"
    (repo / "vecopen" / "src").mkdir(parents=True)
    _git(repo, "init", "-q")
    f = repo / "vecopen" / "src" / "m.py"
    f.write_text("x = 1\n")

    s0 = registry.git_state(repo)  # no commit yet
    assert s0["git_commit"] == "uncommitted" and s0["git_head"] is None and len(s0["diff_sha256"]) == 64

    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "init")
    sha = _git(repo, "rev-parse", "HEAD")
    s1 = registry.git_state(repo)
    assert s1 == {"git_commit": sha, "git_head": sha, "git_dirty": False, "diff_sha256": None}

    (repo / "outside.txt").write_text("not a code path\n")  # changes outside code paths don't matter
    assert registry.git_state(repo) == s1

    f.write_text("x = 2\n")
    s2 = registry.git_state(repo)
    assert s2["git_commit"] == "uncommitted" and s2["git_head"] == sha and s2["git_dirty"]
    f.write_text("x = 3\n")
    s3 = registry.git_state(repo)
    assert s3["diff_sha256"] != s2["diff_sha256"]

    f.write_text("x = 1\n")
    (repo / "vecopen" / "src" / "new.py").write_text("y = 1\n")  # untracked file in a code path
    s4 = registry.git_state(repo)
    assert s4["git_commit"] == "uncommitted" and s4["diff_sha256"] is not None


def test_real_registry_verifies():
    """The committed trial registry must never be edited (standing rule 3)."""
    res = verify(registry.REGISTRY_PATH)
    assert res.ok, res.errors
