"""Append-only, hash-chained trial registry (standing rule 3; SPEC section 6, overfitting control 2).

Every run appends one JSON line to ``vecopen/trials/registry.jsonl`` holding the config hash,
the git commit (or ``"uncommitted"`` plus a diff hash), the UTC time, the phase and the metrics.
Later phases take the number of ``kind == "trial"`` lines as the N of the deflated Sharpe ratio.

Tamper evidence
---------------
* Each line is the canonical JSON of its record (sorted keys, no spaces, UTF-8, no NaN).
* ``seq`` is the 0-based line index.
* ``prev_sha256`` is the ``line_sha256`` of the previous line (``GENESIS`` for line 0).
* ``line_sha256`` is the sha256 of the canonical JSON of the record without ``line_sha256``.
* A head anchor ``registry.head.json`` beside the file stores the line count and the last
  ``line_sha256``, so a deleted or edited *last* line, or a truncated file, is also caught.

``verify()`` recomputes all of this and reports every inconsistency. ``append()`` runs
``verify()`` first and refuses to write onto a registry that fails it. There is no API to
edit or delete a line. The chain makes edits evident; the git history of the two files is
the external anchor against someone who rewrites the whole chain and the head together.
The registry assumes a single writer (the orchestrator runs phases sequentially).
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from vecopen.config import PROJECT_ROOT, LoadedConfig, load_config

REGISTRY_PATH = PROJECT_ROOT / "trials" / "registry.jsonl"
GENESIS = "0" * 64
SCHEMA_VERSION = 1

# Paths (relative to the git top level) whose state defines "the code that produced a result".
CODE_PATHS: tuple[str, ...] = (
    "vecopen/src",
    "vecopen/tests",
    "vecopen/config",
    "vecopen/SPEC.md",
    "vecopen/CLAUDE.md",
    "pyproject.toml",
)


class RegistryError(RuntimeError):
    """The registry failed verification, or a record could not be written."""


@dataclass
class VerifyResult:
    n_lines: int = 0
    last_sha256: str = GENESIS
    errors: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors


# ---------------------------------------------------------------- hashing helpers

def _canonical(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def line_hash(record: Mapping[str, Any]) -> str:
    """sha256 of the canonical JSON of ``record`` without its ``line_sha256`` field."""
    body = {k: v for k, v in record.items() if k != "line_sha256"}
    return _sha256(_canonical(body).encode("utf-8"))


def head_path(path: str | Path) -> Path:
    p = Path(path)
    return p.with_name(p.stem + ".head.json")


# ---------------------------------------------------------------- verification

def verify(path: str | Path | None = None) -> VerifyResult:
    """Check the whole chain and the head anchor. Never raises on bad content."""
    p = Path(path) if path is not None else REGISTRY_PATH
    hp = head_path(p)
    res = VerifyResult()

    head: dict[str, Any] | None = None
    if hp.exists():
        try:
            head = json.loads(hp.read_text(encoding="utf-8"))
            if not (isinstance(head, dict) and isinstance(head.get("n"), int)
                    and isinstance(head.get("last_sha256"), str)):
                raise ValueError("bad shape")
        except (ValueError, UnicodeDecodeError) as exc:
            res.errors.append(f"head anchor {hp.name} is unreadable ({exc})")
            head = None

    if not p.exists():
        if hp.exists():
            res.errors.append(f"{p.name} is missing but its head anchor exists (file deleted)")
        return res

    raw = p.read_bytes()
    if raw and not raw.endswith(b"\n"):
        res.errors.append("file does not end with a newline (truncated or partial write)")
    lines = raw.split(b"\n")
    if lines and lines[-1] == b"":
        lines = lines[:-1]

    prev = GENESIS
    for i, bline in enumerate(lines):
        where = f"line {i + 1}"
        try:
            text = bline.decode("utf-8")
        except UnicodeDecodeError:
            res.errors.append(f"{where}: not valid UTF-8")
            prev = "<broken>"
            continue
        try:
            rec = json.loads(text)
        except ValueError as exc:
            res.errors.append(f"{where}: not valid JSON ({exc.__class__.__name__})")
            prev = "<broken>"
            continue
        if not isinstance(rec, dict):
            res.errors.append(f"{where}: not a JSON object")
            prev = "<broken>"
            continue
        try:
            canonical = _canonical(rec)
        except ValueError:
            canonical = None
        if canonical != text:
            res.errors.append(f"{where}: not in canonical form (line was re-serialised or edited)")
        if rec.get("seq") != i:
            res.errors.append(f"{where}: seq={rec.get('seq')!r}, expected {i} (line inserted, deleted or moved)")
        if rec.get("prev_sha256") != prev:
            res.errors.append(f"{where}: prev_sha256 does not match the previous line (chain broken)")
        stored = rec.get("line_sha256")
        try:
            computed = line_hash(rec)
        except ValueError:
            computed = "<unhashable>"
        if stored != computed:
            res.errors.append(f"{where}: line_sha256 does not match its content (line edited)")
        prev = stored if isinstance(stored, str) else "<broken>"

    res.n_lines = len(lines)
    res.last_sha256 = prev if lines else GENESIS

    if res.n_lines and not hp.exists():
        res.errors.append(f"head anchor {hp.name} is missing")
    elif head is not None:
        if head["n"] != res.n_lines:
            res.errors.append(f"head anchor says {head['n']} lines, file has {res.n_lines} (lines deleted or added)")
        if head["last_sha256"] != res.last_sha256:
            res.errors.append("head anchor last_sha256 does not match the last line (last line edited or removed)")
    return res


def verify_or_raise(path: str | Path | None = None) -> VerifyResult:
    res = verify(path)
    if not res.ok:
        raise RegistryError("registry failed verification:\n  " + "\n  ".join(res.errors))
    return res


def read(path: str | Path | None = None) -> list[dict[str, Any]]:
    """All records, after verification (raises ``RegistryError`` if the registry is tampered with)."""
    p = Path(path) if path is not None else REGISTRY_PATH
    verify_or_raise(p)
    if not p.exists():
        return []
    # Split on b"\n" only: str.splitlines() would also split on U+0085/U+2028 inside JSON strings.
    return [json.loads(line) for line in p.read_bytes().split(b"\n") if line]


# ---------------------------------------------------------------- git state

def _git(args: Sequence[str], cwd: Path) -> bytes | None:
    try:
        out = subprocess.run(["git", *args], cwd=cwd, capture_output=True, check=False, timeout=60)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return out.stdout if out.returncode == 0 else None


def _file_digest(path: Path) -> bytes:
    data = path.read_bytes().replace(b"\r\n", b"\n")
    return hashlib.sha256(data).digest()


def git_state(repo_root: str | Path | None = None, code_paths: Sequence[str] = CODE_PATHS) -> dict[str, Any]:
    """Commit identity of the code that produced a result.

    Returns ``git_commit`` (the HEAD sha if the code paths are clean, else ``"uncommitted"``),
    ``git_head`` (HEAD sha or None), ``git_dirty`` and ``diff_sha256`` (None when clean). The
    diff hash covers ``git diff HEAD`` over the code paths plus the contents of untracked,
    non-ignored files there; without any commit it covers the contents of all files there.
    """
    start = Path(repo_root) if repo_root is not None else PROJECT_ROOT
    top_raw = _git(["rev-parse", "--show-toplevel"], start)
    if top_raw is None:  # not a git repo: fingerprint the files on disk
        h = hashlib.sha256(b"nogit\0")
        for rel in code_paths:
            base = start / rel
            if base.is_file():
                files = [base]
            elif base.is_dir():
                files = sorted(q for q in base.rglob("*") if q.is_file())
            else:
                files = []
            for f in files:
                if "__pycache__" in f.parts or f.suffix == ".pyc":
                    continue
                h.update(f.relative_to(start).as_posix().encode() + b"\0" + _file_digest(f))
        return {"git_commit": "uncommitted", "git_head": None, "git_dirty": True, "diff_sha256": h.hexdigest()}

    top = Path(top_raw.decode().strip())
    paths = list(code_paths)
    head_raw = _git(["rev-parse", "--verify", "-q", "HEAD"], top)
    head = head_raw.decode().strip() if head_raw else None

    h = hashlib.sha256()
    if head is None:
        listed = _git(["ls-files", "-z", "--cached", "--others", "--exclude-standard", "--", *paths], top) or b""
        h.update(b"nocommit\0")
        for rel in sorted(set(filter(None, listed.decode().split("\0")))):
            f = top / rel
            if f.is_file():
                h.update(rel.encode() + b"\0" + _file_digest(f))
        return {"git_commit": "uncommitted", "git_head": None, "git_dirty": True, "diff_sha256": h.hexdigest()}

    status = _git(["status", "--porcelain", "--untracked-files=all", "--", *paths], top) or b""
    if not status.strip():
        return {"git_commit": head, "git_head": head, "git_dirty": False, "diff_sha256": None}

    diff = _git(["diff", "HEAD", "--binary", "--no-color", "--no-ext-diff", "--", *paths], top) or b""
    h.update(b"diff\0" + diff)
    untracked = _git(["ls-files", "-z", "--others", "--exclude-standard", "--", *paths], top) or b""
    for rel in sorted(set(filter(None, untracked.decode().split("\0")))):
        f = top / rel
        if f.is_file():
            h.update(b"\0untracked\0" + rel.encode() + b"\0" + _file_digest(f))
    return {"git_commit": "uncommitted", "git_head": head, "git_dirty": True, "diff_sha256": h.hexdigest()}


# ---------------------------------------------------------------- append

def _utc_now_str(now: datetime | None) -> str:
    t = now if now is not None else datetime.now(UTC)
    if t.tzinfo is None or t.utcoffset() is None:
        raise ValueError("now must be timezone-aware")
    return t.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _check_metrics(metrics: Any) -> dict[str, Any]:
    if not isinstance(metrics, Mapping):
        raise TypeError(f"metrics must be a mapping, got {type(metrics).__name__}")
    out = dict(metrics)
    if not all(isinstance(k, str) for k in out):
        raise TypeError("metric names must be strings")
    try:
        _canonical(out)
    except ValueError as exc:  # NaN / inf
        raise ValueError(f"metrics must be finite JSON values ({exc}); use null for a missing value") from exc
    except TypeError as exc:
        raise TypeError(f"metrics must be JSON-serialisable ({exc})") from exc
    return out


def append(
    phase: int,
    metrics: Mapping[str, Any],
    *,
    kind: str = "run",
    note: str | None = None,
    config: LoadedConfig | None = None,
    path: str | Path | None = None,
    repo_root: str | Path | None = None,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Append one record and return it. Refuses (``RegistryError``) if ``verify()`` fails.

    ``kind`` labels the line: ``"trial"`` for a config-level trial counted in the DSR's N,
    ``"scaffold"``/``"run"``/``"gate"`` etc. for other runs.
    """
    if isinstance(phase, bool) or not isinstance(phase, int) or not 0 <= phase <= 10:
        raise ValueError(f"phase must be an int in [0, 10], got {phase!r}")
    if not isinstance(kind, str) or not kind:
        raise ValueError("kind must be a non-empty string")
    if note is not None and not isinstance(note, str):
        raise TypeError("note must be a string or None")
    m = _check_metrics(metrics)
    cfg = config if config is not None else load_config()
    p = Path(path) if path is not None else REGISTRY_PATH

    state = verify_or_raise(p)

    record: dict[str, Any] = {
        "schema": SCHEMA_VERSION,
        "seq": state.n_lines,
        "utc": _utc_now_str(now),
        "phase": phase,
        "kind": kind,
        "config_sha256": cfg.sha256,
        "config_version": cfg.data.get("version"),
        **git_state(repo_root),
        "python": platform.python_version(),
        "metrics": m,
        "note": note,
        "prev_sha256": state.last_sha256,
    }
    record["line_sha256"] = line_hash(record)
    line = (_canonical(record) + "\n").encode("utf-8")

    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "ab") as fh:
        fh.write(line)
        fh.flush()
        os.fsync(fh.fileno())

    hp = head_path(p)
    tmp = hp.with_name(hp.name + ".tmp")
    anchor = _canonical({"n": state.n_lines + 1, "last_sha256": record["line_sha256"]}) + "\n"
    tmp.write_bytes(anchor.encode("utf-8"))  # bytes, so no CRLF translation on Windows
    os.replace(tmp, hp)
    return record
