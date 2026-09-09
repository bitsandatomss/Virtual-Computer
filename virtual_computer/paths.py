"""Path bootstrap for the Virtual-Computer monorepo.

Layout (repo root = directory containing ``context.txt``)::

    Virtual-Computer/
      context.txt
      microarchitecture/{vmicro,vmarch,...}   # microarchitecture layer
      compiler/{virtual_compiler,self_compiler,...}  # compiler layer
      kernel/{virtual_kernel,learned_kernel,...}     # OS-kernel layer
      virtual_computer/...                 # unified facade (this package)
      tests/...  docs/...

Each layer keeps its original top-level package names, so this module
inserts the three layer roots at the front of ``sys.path``. Importing
``virtual_computer`` (or anything under it) therefore makes
``vmicro``, ``vmarch``, ``virtual_compiler``, ``virtual_kernel`` and
``learned_kernel`` importable from the repo root with no install step.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
LAYER_DIRS = (
    REPO_ROOT / "microarchitecture",
    REPO_ROOT / "compiler",
    REPO_ROOT / "kernel",
)


def ensure_layer_paths() -> list[str]:
    added: list[str] = []
    for d in LAYER_DIRS:
        s = str(d)
        if d.is_dir() and s not in sys.path:
            sys.path.insert(0, s)
            added.append(s)
    return added


ensure_layer_paths()
