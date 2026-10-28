# viva-core Milestone 3 results: native process ports (quadrant 2)

M3 moves processes themselves into Rust (quadrant 2: native engine plus native
processes). Two ports are done and bit-identical to the Python originals. The
whole-model speedup is still unmeasured.

## Key finding: not every process can be ported from its serialized config

ProteinDegradation is **config-serializable**. Its `get_protein_degradation_config`
output is plain data (rates, ids, AA counts, seed), so a pure native port works
and needs no Python at run time.

Metabolism is **not**. Its per-tick LP assembly calls live `sim_data` objects
(the `'node'` / `'method'` entries in its config), which cannot be serialized.
The GLPK solve core is ported and bit-identical, but a fully native metabolism
also needs the LP assembly and its sim_data dependencies ported. Until then the
native metabolism process is fail-loud when enabled (a clear "LP assembly
deferred" error, never a silent divergence).

This splits the remaining processes in two:

- (a) **Config-serializable:** port like ProteinDegradation (config in, native
  update out). Cheap per process; the cost is the RNG distributions and numerics
  each one needs.
- (b) **Live-sim_data-dependent:** need either their assembly and sim_data
  dependencies ported, or a hybrid where Python builds per-tick inputs and Rust
  runs the heavy compute.

For the whole-model path (M4+): the speedup cannot come from porting processes
one by one by config alone. Each remaining process must first be classified
(a) or (b), and the (b) set decides whether we do a hybrid or port sim_data
access. Estimates of whole-model gain should wait on that classification.

## What is ported, bit-identical

| Piece | Result |
|---|---|
| numpy `RandomState` RNG foundation (MT19937, poisson with Knuth below 10 and PTRS from 10 up) | bit-for-bit vs numpy |
| ProteinDegradation, full native process | exact bulk-count equality vs pb and vs Rust+Python, 40 ticks, several seeds; test poisons the Python methods to prove no callback |
| FBA solve core via GLPK FFI | exact `==` on objective and flux vector vs swiglpk |

GLPK is behind the `glpk` cargo feature. The default wheel is GLPK-free.
Native processes are opt-in via `VIVA_NATIVE_PROCESSES` (unset means the pure
Python path).

## Conformance and no-regression gate

- `cargo test --workspace`: green, default and `--features glpk`. fmt and
  clippy (`-D warnings`) clean.
- `python/tests` (quadrant 3/4 parity, native ProteinDegradation, native
  metabolism solve, zero-copy, conformance battery, upstream suite): 25 passed
  with `VIVA_NATIVE_PROCESSES` unset and with it set to `1`.

## Quadrant-2 numbers (release build, glpk feature)

Per-tick cost, setup and warm-up excluded, best of 5, on this laptop.

ProteinDegradation, 4300 proteins x 21 amino acids, 300 ticks
(`python/benchmarks/bench_m3_protein_degradation.py`; final bulk asserted
identical across all three before timing):

| Configuration | ticks/s | ms/tick |
|---|---|---|
| pb engine + Python process | 5698 | 0.175 |
| Rust engine + Python process (quadrant 3) | 2739 | 0.365 |
| Rust engine + native process (quadrant 2) | 8099 | 0.123 |

Native is 2.96x over Python-on-Rust and 1.42x over pb all-Python. The Rust
engine with a Python process is slower than pb for this bulk-heavy process (the
update marshaling of 2x4300 elements and per-tick view building dominate); that
is a real finding, not hidden by the quadrant-2 number.

FBA solve, 400 materials x 2398 flows, 4794 nonzeros
(`python/benchmarks/bench_m3_metabolism.py`; solutions asserted bit-identical):

| Configuration | solves/s | ms/solve |
|---|---|---|
| swiglpk from Python | 64.5 | 15.51 |
| native GLPK FFI | 231.8 | 4.31 |

3.60x (3.55x in the Task 3 run). Both do identical GLPK C work, so this is the
dispatch/FFI win only, on the solve alone.

`python/benchmarks/bench_m3.py` runs both.

## Honesty caveats

- These are micro-composites of one process each. They overstate whole-model
  speedup, where many other processes, the engine and emitters also cost time.
- The FBA number is solve-only. The per-tick LP assembly, where much of
  metabolism's Python time sits, is not ported and not measured.
- The LP is built at metabolism-like scale by the real wrapper, not captured
  from a live ~9k-reaction sim (needs ParCa sim_data).
- Python warm-starts one LP across ticks while the native solve is cold per
  call. A cold-vs-cold comparison is exact; multi-tick warm-start equivalence
  matters once a full drop-in lands (keep problem and basis in Rust).
- Tick-count semantics of Rust `run()` vs pb `until` were checked by exact
  equality, not independently.
- The whole-model number needs the full `partitioned.pbg` plus the remaining
  ports, and has not been measured.

## Deferred

- Full metabolism drop-in (fail-loud today): LP assembly and sim_data access.
- Elongation: numba `polymerize` to native Rust.
- rna_degradation and tf_binding: more distributions (multinomial, binomial,
  choice).
- Complexation (arrowhead.so FFI) and scipy-ODE processes (SUNDIALS FFI, or
  bounded divergence).
- Whole-model benchmark on the full `partitioned.pbg` (M4).
