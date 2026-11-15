# Phase 0: scaffold

Date: 2026-11-14 (UTC). Builder: phase 0 subagent. Status: **acceptance met.**

## Reproduce (single command, from the repo root)

```
uv run --extra vecopen python -m vecopen phase0
```

This runs `vecopen/tests` with pytest, prints the config sha256, appends one registry line
(phase 0, kind `scaffold`, metrics = test counts) and verifies the whole registry chain.
Every run appends a line (standing rule 3). Use `--dry-run` to check without appending.

## Acceptance (SPEC section 8, phase 0)

| Criterion | Result |
|---|---|
| `pytest` green | 59 passed, 0 failed, 0 errors, 0 skipped (`uv run --extra vecopen pytest vecopen/tests`) |
| Hash printed | `config/baseline.yaml` sha256 `9e78f831b0e95c6824529485628b0c1777733db5374f6e0f6163894f0c2ae14a`. It is printed in the pytest header, by `python -m vecopen hash` and by `phase0`, and it matches `sha256sum` of the file |
| Registry refuses edits to past lines | Chained hash per line plus a head anchor; `verify()` detects edits, re-serialisation, deletion of any line (including the last), reordering, insertion, truncation, file deletion and a missing or corrupt head. A property test flips one random byte anywhere in the file (150 examples) and checks every flip is caught. `append()` refuses to write onto a registry that fails `verify()` |
| Root suite unaffected | `uv run pytest`: 565 passed, and it does not collect `vecopen/tests` |

Registry line 1 (`vecopen/trials/registry.jsonl`): `seq 0`, `phase 0`, `kind scaffold`,
`config_sha256 9e78f831…ae14a`, `git_commit "uncommitted"` (HEAD `a69a940`, and vecopen is not yet committed),
`diff_sha256 cfffdff2cf052725bdbc0bb77b527a569f95bb8b71a0dc4e5d08a1011a56ad70`,
metrics `{tests: 59, passed: 59, failed: 0, errors: 0, skipped: 0}`,
`line_sha256 88722fa29b8209e876ea3a2bb889ebde9a77ccd147a69e45149c770dc30cc01c`.

## What was built

- **Layout (section 8).** `vecopen/src/vecopen/` has the 14 listed modules. `data, bars, legs, levels, events,
  labels, placebo, features, context, stats, cv, models, backtest` are docstring-only stubs naming their phase,
  SPEC section and planned streaming API. `registry` is implemented. The repo also has `tests/`, `trials/` and
  `reports/`, plus `data/`, which is git-ignored by the root `data/` rule and an explicit `vecopen/data/` line.
  `data/research/` and `data/holdout/` are left for the phase 1 split script; nothing under `data/holdout/` was touched.
- **`config.py`** (one module added beyond the section 8 list). `load_config(path=None) -> LoadedConfig(data, sha256, path)`
  is a NamedTuple, so it also unpacks as `(config, sha256, path)`. The sha256 is taken over the file bytes with CRLF
  normalised to LF, so Windows (`core.autocrlf=true`) and Linux checkouts hash the same. Hash and data come from one read.
  It rejects a non-mapping config, a missing top-level section, or a non-integer seed. Re-exported from `vecopen`.
- **`registry.py`**
  - `append(phase, metrics, *, kind, note, config, path, repo_root, now)` writes one line. The line is the canonical
    JSON of the record (sorted keys, no spaces, UTF-8, NaN/inf rejected). It holds `schema, seq, utc, phase, kind,
    config_sha256, config_version, git_commit, git_head, git_dirty, diff_sha256, python, metrics, note,
    prev_sha256, line_sha256`. It is written in binary append mode and fsynced, then the head anchor
    `registry.head.json` is replaced atomically.
  - `verify()` returns `VerifyResult(n_lines, last_sha256, errors)` and never raises on bad content.
  - `verify_or_raise()` and `read()` are also provided. There is no edit or delete API.
  - `git_state()`: if the code paths (`vecopen/src, tests, config, SPEC.md, CLAUDE.md`, root `pyproject.toml`) are
    clean, `git_commit` is the HEAD sha. Otherwise it is `"uncommitted"` with `diff_sha256` over `git diff HEAD` plus
    the untracked files in those paths. With no commit at all, or no git, `diff_sha256` is a hash of the file contents.
  - `kind` separates config-level trials (`"trial"`, the N of the DSR in later phases) from other runs such as this `scaffold` line.
- **`__main__.py`.** `python -m vecopen hash | verify | phase0 [--dry-run]`.
- **`__init__.py`.** Sets `POLARS_MAX_THREADS`, `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS` and `MKL_NUM_THREADS` to 2 unless the caller already set them (resource rule).
- **`vecopen/.gitattributes`.** Marks `trials/registry.jsonl` and `trials/registry.head.json` as `-text`, so git never rewrites their line endings. The chain is byte-exact.
- **Tests** (`vecopen/tests/`, 59)
  - `test_vo_config.py`: hash = sha256 of the file, stable, CRLF-invariant, content-sensitive; the frozen values equal the SPEC section 7 table; the baseline sits at the centre of the robustness grid; session conventions; malformed configs are rejected.
  - `test_vo_registry.py`: required fields, chain links, every tamper case above, refusal to append, metric and phase validation, the hypothesis byte-flip and metrics round-trip properties, git identity on a temporary repo, and a check that the real registry verifies.
  - `test_vo_scaffold.py`: the modules import and have docstrings; the directories exist; `vecopen/data/{research,holdout,raw}` are git-ignored, checked with `git check-ignore --no-index` so nothing under them is read; the thread variables are set.
  - `conftest.py`: prints the config hash in the pytest header, and fails the session if any test modifies the real registry.

## Packaging

- The root `pyproject.toml` has a new optional extra `vecopen`: polars>=1.12, numpy>=2.1, scipy>=1.14 and pyyaml>=6, all with the existing pins, plus scikit-learn>=1.9.1, hmmlearn>=0.3.3, lightgbm>=4.7.0 and hypothesis>=6.170.15.
- The hatch wheel `packages` now includes `vecopen/src/vecopen`, and the editable install adds `vecopen/src` to `sys.path`. `uv.lock` was relocked.
- pytest uses the root config: `testpaths = ["tests"]` still governs plain `uv run pytest`, and `vecopen/tests` runs when it is named explicitly. The vecopen test files are named `test_vo_*` and the folder has no `__init__.py`, so its modules cannot clash with the root `tests` package.

## Findings and deviations

- Hypothesis found a bug during the build. `read()` split lines with `str.splitlines()`, which also splits on U+0085 and U+2028 inside JSON strings. It now splits on `\n` only, like `verify()`. No registry line had been written at that point.
- Python 3.13.7, not 3.11 (deviation already recorded in SPEC.md).
- The head anchor catches deletion of the last line, which a backward chain alone cannot. The external anchor against a full rewrite of both files is the git history, so both files should be committed.
- `SPEC.md`, `config/baseline.yaml`, `.env`, keys, `config/schema.yaml` and `bin/` were not touched.
