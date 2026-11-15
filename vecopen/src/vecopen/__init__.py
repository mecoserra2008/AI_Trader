"""vecopen: Vector-Open level signal generator (see ``vecopen/SPEC.md``).

Phase 0 provides the scaffold, ``load_config()`` (the frozen baseline config plus its
sha256) and the append-only, hash-chained trial registry (``vecopen.registry``).
"""

import os as _os

# Resource rule (vecopen/CLAUDE.md): at most 2 threads per agent. Set before numpy/polars load.
for _var in ("POLARS_MAX_THREADS", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    _os.environ.setdefault(_var, "2")

from vecopen.config import CONFIG_PATH, PROJECT_ROOT, LoadedConfig, config_sha256, load_config  # noqa: E402

__all__ = ["CONFIG_PATH", "PROJECT_ROOT", "LoadedConfig", "config_sha256", "load_config"]
