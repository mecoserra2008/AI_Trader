"""Command line: ``uv run --extra vecopen python -m vecopen <command>`` from the repo root.

Commands
--------
``hash``     print the sha256 of ``config/baseline.yaml``.
``verify``   verify the trial registry chain; exit 1 if it fails.
``phase0``   run ``vecopen/tests`` with pytest, print the config hash, append one registry line
             (phase 0, kind ``scaffold``, metrics = test counts) and verify the registry.
             ``--dry-run`` skips the append.
``data ...`` phase 1 data steps (``python -m vecopen data -h``): ``crypto-download``, ``cfd-download``,
             ``build-split``, ``gaps``, ``storage``.
``phase1``   the phase 1 reproduce command: crypto download (skips files in the manifest), build and split
             of every staged asset (assets already split are not touched), research-sample gap tables and the
             history table, the vecopen tests, one registry line (kind ``data``) and
             ``reports/phase_1_tables.md``. ``--with-cfd`` also runs the cTrader download (needs the keys).
``phase2``   the phase 2 reproduce command: smoke run of the level registry over one asset's research bars
             (default BTCUSDT), ``reports/phase_2_tables.md``, the vecopen tests and one registry line
             (kind ``build``). ``--dry-run`` skips the append.
``phase3``   the phase 3 reproduce command: touch events, placebo levels (3 families), labels and family-1 matching
             for every crypto asset (one at a time), tables under ``data/research/events/``,
             ``reports/phase_3_tables.md``, the vecopen tests and one registry line (kind ``build``).
             ``--assets A,B`` picks a subset, ``--dry-run`` skips the append.
``phase3-balance``  phase 3 amendment (2026-11-14): matched-sample covariate balance per timeframe group (intraday
             15m-4h, daily_plus 1d-1M; SPEC section 9) from the stored phase 3 events and matches tables, checked
             against the authoritative run's manifest; ``reports/phase_3_group_balance.md``, the vecopen tests and
             one registry line (phase 3, kind ``build``). Covariates only. ``--dry-run``, ``--skip-tests``.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

from vecopen import registry
from vecopen.config import PROJECT_ROOT, load_config


def _cmd_hash(_: argparse.Namespace) -> int:
    cfg = load_config()
    print(f"config {cfg.path.relative_to(PROJECT_ROOT.parent).as_posix()} sha256 {cfg.sha256}")
    return 0


def _cmd_verify(_: argparse.Namespace) -> int:
    res = registry.verify()
    if res.ok:
        print(f"registry OK: {res.n_lines} lines, last line_sha256 {res.last_sha256}")
        return 0
    print("registry FAILED verification:")
    for e in res.errors:
        print(f"  {e}")
    return 1


def run_tests(test_dir: Path) -> dict[str, int]:
    """Run pytest on ``test_dir`` in a subprocess and return counts parsed from its JUnit XML."""
    with tempfile.TemporaryDirectory() as td:
        xml_path = Path(td) / "junit.xml"
        cmd = [sys.executable, "-m", "pytest", str(test_dir), "-q", "-p", "no:cacheprovider",
               f"--junitxml={xml_path}"]
        proc = subprocess.run(cmd, cwd=PROJECT_ROOT.parent, env=os.environ.copy(), check=False)
        if not xml_path.exists():
            return {"tests": 0, "passed": 0, "failed": 0, "errors": 1, "skipped": 0,
                    "pytest_exit_code": proc.returncode}
        root = ET.parse(xml_path).getroot()
        suites = [root] if root.tag == "testsuite" else list(root.iter("testsuite"))
        tot = {k: sum(int(s.get(k, 0)) for s in suites) for k in ("tests", "failures", "errors", "skipped")}
    return {
        "tests": tot["tests"],
        "passed": tot["tests"] - tot["failures"] - tot["errors"] - tot["skipped"],
        "failed": tot["failures"],
        "errors": tot["errors"],
        "skipped": tot["skipped"],
        "pytest_exit_code": proc.returncode,
    }


def _cmd_phase0(args: argparse.Namespace) -> int:
    cfg = load_config()
    counts = run_tests(PROJECT_ROOT / "tests")
    print(f"\nconfig sha256: {cfg.sha256}")
    print(f"tests: {counts}")
    green = counts["pytest_exit_code"] == 0 and counts["failed"] == 0 and counts["errors"] == 0
    if args.dry_run:
        print("dry run: registry not written")
    else:
        rec = registry.append(
            0, counts, kind="scaffold", config=cfg,
            note="phase 0 scaffold: load_config + hash-chained registry; metrics are vecopen/tests counts",
        )
        print(f"registry line {rec['seq'] + 1} appended: line_sha256 {rec['line_sha256']} "
              f"git_commit {rec['git_commit']} diff_sha256 {rec['diff_sha256']}")
    rc = _cmd_verify(args)
    return 0 if green and rc == 0 else 1


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m vecopen", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("hash").set_defaults(fn=_cmd_hash)
    sub.add_parser("verify").set_defaults(fn=_cmd_verify)
    p0 = sub.add_parser("phase0")
    p0.add_argument("--dry-run", action="store_true")
    p0.set_defaults(fn=_cmd_phase0)
    from vecopen import phase1 as _p1
    from vecopen import phase2 as _p2
    from vecopen import phase3 as _p3
    _p1.add_commands(sub)
    _p2.add_commands(sub)
    _p3.add_commands(sub)
    args = ap.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
