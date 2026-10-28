"""Make the shared, read-only v2ecoli venv importable from the fresh `.venv-m4`
test venv (v2ecoli + process-bigraph 1.8.3 + bigraph-schema 1.6.0 + wholecell
live there; we never modify that venv). Override the location with
`V2ECOLI_ROOT` if the checkout is elsewhere. A `PYTHONPATH` already covering
these makes this a no-op.
"""

import os
import sys

_ROOT = os.environ.get("V2ECOLI_ROOT", "/Users/eranagmon/code/v2ecoli")
_PATHS = [
    os.path.join(_ROOT, ".venv", "lib", "python3.12", "site-packages"),
    _ROOT,
]
for _p in _PATHS:
    if os.path.isdir(_p) and _p not in sys.path:
        # Front of path: v2ecoli's numpy/pb/bs take precedence over anything the
        # m4 venv happens to carry, so the conformance target stays pinned.
        sys.path.insert(0, _p)
