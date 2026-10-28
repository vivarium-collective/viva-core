"""M3 consolidated quadrant-2 benchmark: runs both component benches.

Requires a release wheel built with `--features glpk`. Run:
  PYTHONPATH=python <venv>/bin/python python/benchmarks/bench_m3.py
"""

import runpy
from pathlib import Path

here = Path(__file__).parent
for name in ("bench_m3_protein_degradation.py", "bench_m3_metabolism.py"):
    runpy.run_path(str(here / name), run_name="__main__")
