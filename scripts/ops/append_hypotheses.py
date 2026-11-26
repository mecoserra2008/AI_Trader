"""Append a pending hypotheses block to research/hypotheses.yaml (run under `governor.py lock registry`).

    python scripts/ops/append_hypotheses.py research/hypotheses_pending/<WS>.yaml

The pending file holds YAML in the registry's own format: a mapping of ids (H-<WS>-n) to hypothesis entries, optionally
preceded by a comment header. It is validated (parses, ids new, required fields present) and appended verbatim, indented
under the registry's top-level mapping, so concurrent agents never interleave partial writes.
"""

from __future__ import annotations

import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
REG = ROOT / "research" / "hypotheses.yaml"
REQUIRED = ("title", "statement", "test", "success")


def main(path: str) -> int:
    text = Path(path).read_text(encoding="utf-8")
    new = yaml.safe_load(text) or {}
    if not isinstance(new, dict) or not new:
        print("pending file must be a non-empty mapping of hypothesis ids", file=sys.stderr)
        return 2
    reg = yaml.safe_load(REG.read_text(encoding="utf-8"))
    existing = reg.get("hypotheses", reg)
    dup = [k for k in new if k in existing]
    if dup:
        print(f"duplicate ids already in the registry: {dup}", file=sys.stderr)
        return 3
    bad = [k for k, v in new.items() if not isinstance(v, dict) or any(f not in v for f in REQUIRED)]
    if bad:
        print(f"entries missing {REQUIRED}: {bad}", file=sys.stderr)
        return 4
    indent = "  " if "hypotheses" in reg else ""
    block = "\n" + "\n".join((indent + line) if line.strip() else line for line in text.rstrip().splitlines()) + "\n"
    before = REG.read_text(encoding="utf-8")
    REG.write_text(before.rstrip("\n") + "\n" + block, encoding="utf-8")
    check = yaml.safe_load(REG.read_text(encoding="utf-8"))
    got = check.get("hypotheses", check)
    if any(k not in got for k in new):
        REG.write_text(before, encoding="utf-8")
        print("append broke the registry structure; restored", file=sys.stderr)
        return 5
    print(f"appended {len(new)} hypotheses: {', '.join(new)}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
