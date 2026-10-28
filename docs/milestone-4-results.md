# viva-core Milestone 4 results: the whole-model number (honest)

M4's goal was the first **honest whole-model speedup number** — run the real
v2ecoli baseline composite (one generation) on the Rust engine with the ported
native processes swapped in, bit-identical to process-bigraph (pb) 1.8.3, and
measure ticks/s vs pb.

**The honest headline: there is no whole-model engine number yet, and that is the
result.** The whole real baseline cannot run through `run_prebuilt` as-is; two
concrete blockers gate it (below). What *is* measured — on the real baseline, not
a synthetic micro-composite — is (A) the pb whole-model reference the engine must
beat, and (B) the one ported process (ProteinDegradation) at full baseline scale,
run three ways and gated bit-identical. The (B) numbers say plainly why one-by-one
porting is the only path to a whole-model win: **running a Python process on the
Rust engine is slower than pb; the speedup comes only from the native port.**

Scope of everything below: **one generation, baseline (minimal glucose) media,
single seed (0), not run to division.** On-disk ParCa cache (`out/cache/`); no
ParCa run. Release wheel; warm-up excluded; per-tick cost isolated by the
differential method (time at N1 and N2 ticks, `(t(N2)-t(N1))/(N2-N1)`, which
cancels the one-time engine-build / realize cost). Laptop (Apple Silicon).

## What runs bit-identical, and what blocks the rest

Task 1 (`run_prebuilt`) hands pre-built Python process **instances** (carrying
their live `sim_data`) to the Rust engine as instance-bound shims, serializing the
state **stores** to JSON. That works for processes wired as edges over
JSON-serializable stores. The real baseline agent (46 edges / ~55 processes, 20
stores) additionally uses two patterns the JSON state channel cannot carry:

**Blocker 1 — the partition "process-as-data" store.** v2ecoli's partition
architecture stores *live process instances* in a `process` **data store**; the 3
PartitionedProcesses' `Requester`/`Evolver` steps read those instances from state
each tick (6 edges). Live Python objects cannot round-trip through the JSON state
codec. (`test_wholemodel_handoff_blocked_process_store`.)

**Blocker 2 — pint/Unum units in stores.** `listeners` (touched by ~28 of 46
edges) and `environment` (~11) hold pint `Quantity` values. The Rust codec has no
unit type, and the view/update boundary can't re-wrap a bare float as a unit-
carrying `Quantity` for the Python processes that consume it. So nearly the whole
model — not a few processes — is gated on a **unit bridge**.
(`test_wholemodel_handoff_blocked_pint_units`.)

Both are the forward unknowns flagged in the M4 plan ("cross-process pint/Unum
serialization through the view/update boundary"; whole-model scheduling across all
~55 processes). They are recorded as asserting tests so a future fix trips them.

A third, lesser gap — **native-swap empty-config sourcing** (the plan's third
forward unknown) — *was* closed for ported processes: baseline edges carry an
empty JSON `config` (sim_data is baked into the instance), so `build_payload` now
sources a native twin's config from the live instance's `.parameters`. That is why
ProteinDegradation can be native-swapped in the handoff below.

The largest slice that runs bit-identical today is exercised by
`test_wholemodel_conformance.py` and `test_run_prebuilt.py`: real `GlobalClock`,
`ProteinDegradation` (full scale), the gated `ReplicationData` deriver, and the
unique stores (incl. the Task-2 sub-array fields `promoter.bound_TF`,
`chromosome_domain.child_domains`). Processes that read the `process` store or
pint-carrying stores (metabolism, elongation, transcription, the listeners/mass
derivers, …) do not run through the handoff yet.

## The numbers (`python/benchmarks/bench_m4_wholemodel.py`)

### (A) Whole model — pb reference only

| Configuration | ms/tick | ticks/s |
|---|---|---|
| pb 1.8.3, full 46-edge baseline agent | ~57 | ~17.6 |
| `run_prebuilt` (engine), whole model | — | **BLOCKED** (not measurable) |

The engine number is blocked by Blockers 1 and 2 above. ~17.6 ticks/s warm is the
bar the Rust engine must beat once the whole model can run on it.

### (B) ProteinDegradation at full baseline scale (16321-wide real bulk, live sim_data)

Bit-identity (final bulk counts, RNG-exact) asserted across all three before timing.

| Configuration | ms/tick | ticks/s | vs pb |
|---|---|---|---|
| pb 1.8.3 engine + Python process | 0.106 | ~9400 | 1.00x |
| `run_prebuilt` native-swap **OFF** (Rust engine + Python instance) | 0.230 | ~4350 | **0.46x** |
| `run_prebuilt` native-swap **ON** (Rust engine + native Rust twin) | 0.061 | ~16300 | **1.73x** |

Native ON is ~3.75x over Python-on-Rust (OFF). One-time engine-build (full-bulk
JSON parse) ~9-13 ms, excluded from per-tick as it is paid once per run.

## Honest reading

- **Python-on-Rust (OFF) is *slower* than pb (~0.46x).** The PyO3 view/update
  boundary — building each process's input views and applying its update across
  the Rust↔Python boundary — costs more than pb's in-process Python scheduling.
  This reproduces, on the real baseline, the same direction M3 saw on a synthetic
  composite (OFF ~0.48x pb). It is a real finding, not hidden.
- **The speedup is entirely from the native port (ON, ~1.73x vs pb).**
- **Therefore a whole-model run today would be < 1x vs pb.** With 1 of ~55
  processes ported, a whole-model `run_prebuilt` (once unblocked) would be
  dominated by OFF-class Python-on-Rust processes, each slower than pb. The
  whole-model win requires porting the hot processes to native — not merely moving
  the scheduler to Rust.
- **Micro-benchmarks overstate.** A single-process number (even the honest 1.73x
  here, let alone the solve-only FBA numbers from M3) is not a whole-model number.
  The whole-model number remains unmeasured, now with the precise reasons why.

## No-regression gate

- `python/tests` green with `VIVA_NATIVE_PROCESSES` unset and set (M1 quadrant-3/4
  parity, M2 zero-copy incl. sub-array unique, M3 native ProteinDegradation + FBA
  solve, Task-1 `run_prebuilt`, Task-3 whole-model conformance).
- `cargo test --workspace`, `cargo fmt --check`, `cargo clippy -D warnings`: clean.
  (M4 Task 3 adds no Rust; native-swap config sourcing is Python-side in
  `prebuilt.py`.)
- "Bit-for-bit" is same-libm-scoped (the RNG poisson PTRS path), as in M3; cross-
  host / manylinux re-verification is deferred to the cluster milestone.

## Deferred to M5+

- **Unit bridge** (Blocker 2): carry pint/Unum through the view/update boundary so
  listeners/environment-touching processes can run on the engine. This unblocks
  most of the model.
- **Live-store objects** (Blocker 1): let the engine hold opaque Python objects in
  stores (the partition `process` store) and hand them back through views
  unchanged — or restructure the partition pattern.
- **More native ports** — the actual source of whole-model speedup: elongation
  (`polymerize` + the global-numpy-RNG coupling hazard), the full metabolism
  assembly (the M4 hybrid wires the native FBA solve into the Python assembly; port
  the assembly if it dominates), transcription/degradation distributions.
- Functional division; multi-generation / multi-seed / multi-condition; the
  Slurm/manylinux cluster build with cross-host RNG re-verification.
