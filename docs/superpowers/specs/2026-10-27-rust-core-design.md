# rust-core — a general-purpose fast process-bigraph engine in Rust (design)

**Status:** design, pending approval
**Date:** 2026-10-27
**Author session:** rust-core
**Spike evidence:** `scratchpad/rust_spike_report.md` (all 6 stages green)
**Research:** `scratchpad/language_research.md`, `scratchpad/rewrite_surface_recon.md`

---

## 1. Intent & why

Reimplement **bigraph-schema + process-bigraph** as a **general-purpose, high-performance Rust core** ("rust-core") that is byte-compatible with the existing bigraph-schema **JSON contract**, so the entire Python viva-ecosystem (workbench/dashboard, investigations, every `viva-*` package, v2ecoli) plugs into it unchanged. This is not a v2ecoli-specific effort — v2ecoli is the first *consumer* that proves it.

**Why now (evidence, not hope):** a full benchmark of the distilled-ecoli re-architecture showed it delivered *no* speedup (1.29× slower serial; parallelism Amdahl-capped at ~1.2× by p≈0.19) because ~80% of every tick is Python glue. Recon localized that glue precisely: **~300 `plum` multiple-dispatch calls per tick** through bigraph-schema `apply`+`reconcile`. The spike then proved a Rust static-dispatch engine runs that same work **~160× faster** (217k vs 1,358 ticks/s). The bottleneck is the interpreted dynamic-dispatch schema engine, and a compiled core eliminates exactly it.

**The dramatic-speedup thesis, corrected by the spike:** the engine rewrite is *necessary but not sufficient*. Once the engine is fast, Python processes dominate (Rust-engine + Python-processes ≈ 33.5k tps vs fully-native ≈ 208k tps — a further ~6.2×). So the full win requires porting hot processes to Rust, incrementally, after the engine lands.

## 2. Goals

- A Rust crate set reimplementing the bigraph-schema type system + process-bigraph runtime, **byte-compatible with the JSON contract** (pinned to the running versions: **bigraph-schema 1.6.0, process-bigraph 1.8.3**).
- **Full bidirectional Python↔Rust interchangeability** (all 4 quadrants, spike-proven):
  1. Python engine + Python process (today, unchanged).
  2. Rust engine + Rust process (max speed).
  3. **Rust engine runs existing Python processes** via PyO3 callbacks (zero-copy numpy; 0.04 µs/call) — run all-Python v2ecoli on rust-core day one.
  4. **Python engine runs Rust processes** (PyO3 `#[pyclass]` + a ~10-line Python `Process` shell; bit-identical, ~1.26× overhead) — drop-in accelerate one process without switching engines.
- **GIL-free shared-memory parallelism** (`rayon`; 7.36×/8 threads measured) and **distributed MPI** (`rsmpi`→system MPI, Slurm-launched, ensemble-level) — both spike-proven on the local stack.
- **Pip-installable** as a PyO3 wheel (maturin), so the ecosystem consumes it like any package.
- **Bit-identical to the Python reference** per ported component — the existing v2ecoli parity + behavioral suites become the **cross-language conformance suite**.

## 3. Non-goals / anti-goals

- **Not** rewriting ParCa — it's a one-time offline `sim_data` build (~164 MB dill, pint-aware); it stays Python and rust-core consumes its output read-only. (Anti-goal: porting ParCa.)
- **Not** a big-bang greenfield replacement — anti-goal is any step that breaks the running ecosystem. Everything lands behind the JSON contract with the Python path always working.
- **Not** within-run spatial/one-sided-MPI decomposition in v1 — MPI is ensemble-level (seeds/conditions across Slurm ranks). (The one case that would have favored C++; explicitly deferred.)
- **Not** reimplementing numeric kernels that are already compiled — FFI-reuse them (GLPK for FBA, `arrowhead.so` for complexation) rather than rewrite. (Anti-goal: re-deriving the science.)

## 4. Key decisions (made, with rationale)

- **Language: Rust.** Only candidate giving C-level GIL-free speed *with* compile-time data-race safety (`Send`/`Sync`+`rayon`) *and* a clean in-process Python binding (PyO3/maturin — the Polars/ruff/uv pattern) *and* no JIT warmup over millions of short ticks. C++ was #2 (only wins for within-run one-sided-MPI, a non-goal); Julia #3 (embedding + JIT + MPI/threads/GC risks undercut the two hardest requirements). Slurm + ensemble MPI makes `rsmpi`→system-MPI sufficient, reinforcing Rust.
- **Interop is the architecture, not an afterthought.** A uniform process-invocation ABI: the engine dispatches each process via the JSON `address` protocol to either a native Rust process or a PyO3-wrapped Python process. The JSON contract is the single source of truth; engine and process are independently swappable.
- **Zero-copy array boundary.** Bulk/unique arrays are shared across Rust↔Python as numpy views (buffer protocol), so Python-process callbacks pay no per-tick marshaling tax (spike: 0.04 µs/call, GIL held across the loop).
- **Lenient JSON codec.** The real `.pbg` emits bare `Infinity`/`NaN` (Python `allow_nan`); the Rust codec must accept these for byte-compat (strict `serde_json` rejects them).
- **FFI for compiled kernels, port the rest.** Metabolism→GLPK and complexation→`arrowhead` via `extern "C"` FFI (spike-proven for GLPK). Port numba `polymerize` (elongation) and the numpy/scipy processes (equilibrium/TCS ODE, initiation/degradation/tf) to native Rust — hard constraint: **exact RNG parity (MT19937 + stochasticRound)** for bit-identical results.
- **Migration order:** (1) engine first — the full 4-quadrant runtime with Python processes, so v2ecoli runs on rust-core immediately and every consumer keeps working; (2) then port hot processes one at a time (metabolism is already FFI; elongation is the top numba port), each gated bit-identical against its Python twin.

## 5. Constraints

- Byte-compatible with bigraph-schema 1.6.0 JSON (two serializers: schema-paired `methods/serialize.py`+`realize.py`, and schema-agnostic `json_codec.py` tagged-dict for pint/numpy/set/bytes). The hardest port surface: the parsimonious PEG type grammar + ~35 Node dataclasses + dispatch tables for ~23 ops.
- Core size: bigraph-schema ~16.5k LOC (hardest), process-bigraph core ~4.9k, v2ecoli runtime subset ~20–25k (ParCa excluded). Multi-month effort.
- Coordinate with the **optimize** session (owns the Python bigraph-schema/pb mypyc lever on the same engine) — rust-core supersedes that lever; align so effort isn't duplicated.
- Target deployment: the lab's **Slurm HPC** + system MPI; wheels must build cross-platform (macOS dev, Linux cluster).
- Conformance: every ported piece bit-identical to the Python reference via the existing parity/behavioral test ladder.

## 6. What's left to the builder (open, for the plan/approval stage)

- **Repo layout & names:** proposed — a workspace with `bigraph-schema-rs` + `process-bigraph-rs` crates + a `pyo3` binding crate publishing one wheel (e.g. `viva-core`). One repo vs separate. (Builder's call unless the user has a preference.)
- **First milestone scope:** proposed — "engine MVP": the schema core (type grammar + registry + apply/reconcile) + scheduler + the JSON codec + the PyO3 wheel, running a real v2ecoli `.pbg` with **all processes still Python** (quadrants 1–3), bit-identical to the Python engine, with a rayon ensemble + an rsmpi smoke **on this Mac** (cluster/manylinux deferred). This proves the engine + interchangeability end-to-end before any process port.
- **Then:** incremental process-port milestones (elongation first), each a bit-identical gate.
- Exact zero-copy strategy for the unique-molecule stores (structured arrays) vs bulk (flat arrays).
- Whether the Python-shell-over-Rust-process (quadrant 4) is auto-generated or hand-written per process.
- The conformance harness mechanics (reuse `distilled_compare.py`-style per-process bit-identical checks across the language boundary).

## 7. Spike evidence summary (all green)

| Claim | Result | Honest caveat |
|---|---|---|
| Static- vs dynamic-dispatch, isolated | **~160×** (217k vs 1,358 tps) | This is Rust static dispatch vs a *pure-Python element loop* — it sizes the glue, it is **NOT** the engine-swap speedup. |
| **Day-one engine swap** (Rust engine + Python processes) | **~1.2×** vs an idealized numpy Python engine; true value vs the *real* plum engine sits between ~1.2× and ~7.5× | The honest day-one number. Running all-Python v2ecoli on rust-core is a real but **partial** win. |
| **Full native ceiling** (Rust engine + Rust processes) | ~208k tps; **~6.2× over day-one** | Realized only by porting hot processes; the project must budget for this. |
| JSON round-trip (real 15MB .pbg) | **structurally** identical (not byte-identical; floats reformat) | Needs lenient Infinity/NaN codec; conformance is on **decoded state**, not bytes. |
| PyO3 wheel, Python drives Rust | works | synthetic model |
| rayon threads | 7.36× / 8 | macOS |
| rsmpi on system Open MPI 5 | builds + 2-rank scatter/gather | **macOS/Homebrew only — NOT yet on the Slurm Linux cluster** |
| GLPK FFI | LP solved | complexation→arrowhead FFI **not** yet spiked |
| Rust engine × Python process (Q3) | zero-copy ok, 0.04 µs/call | **synthetic** 16-process model, not real v2ecoli processes |
| Python engine × Rust process (Q4) | real pb 1.8.3 drives Rust process, bit-identical, 1.26× | **trivial** process; real numpy-heavy process would favor Rust more |

**Honest status of "spike-proven":** the quadrants + mechanisms are proven **in principle on synthetic models and a trivial process**; the real 15 MB `.pbg` got only a *structural* round-trip, and real v2ecoli processes (sim_data.dill, `local:` importlib registry, pint, **unique structured stores**) were never run. Closing that gap — running a real v2ecoli model end-to-end, bit-identical — is exactly **milestone 1**, not a solved problem.

**Conclusion:** the direction is validated — the mechanisms all work and the engine glue is provably the bottleneck. But the dramatic speedup comes from **porting hot processes** on top of a fast engine; the engine swap alone is a modest day-one win that chiefly unlocks interchangeability + parallelism + the migration path.

---

## 8. Builder-check resolutions (2026-10-27)

A fresh builder-check reviewed §1–§7. Honesty corrections above (§7) applied. Builder's-call resolutions (recorded for user veto):

- **MVP op/node-type scope (A1):** port only the op × Node-type **subset `partitioned.pbg` actually exercises** (measure empirically), not the full ~23 ops × ~35 types up front. YAGNI; grow as more models need more. This is what keeps milestone 1 to weeks, not months.
- **"Bit-identical" definition (A2):** conformance is on **decoded state**, not serialized bytes (serde reformats floats) — integer molecule counts **exact**, floats within the existing parity-ladder epsilon (solver/INT). Reuse the `distilled_compare.py`-style per-component checks across the language boundary. **Open technical risk:** Python bigint vs Rust i64 for unbounded counters — audit whether any count can exceed i64; use i128/checked math where it can.
- **Unique structured stores (A3):** **IN scope for milestone 1** — a real `.pbg` uses them, so the MVP must handle per-molecule unique arrays, not just flat bulk (the spike only did flat bulk → this is net-new MVP work). Zero-copy for structured arrays is a milestone-1 problem to solve, not deferred.
- **RNG parity (B7):** **port numpy's MT19937 stream + the exact stochasticRound to native Rust** (don't FFI numpy's RNG — it would re-couple to Python and defeat the speed). Validate the Rust stream against numpy bit-for-bit in an early plan task. This is the long-pole for every stochastic-process port; the spike did NOT test it.
- **Upstream drift (B8):** **pin bigraph-schema 1.6.0 / pb 1.8.3** as the v1 conformance target; revisit tracking once rust-core is the primary engine. (The `~/code` sources are already divergent at 1.4.3/1.10.0 — the installed versions are the truth.)
- **Q4 shell (B9):** **hand-written** ~10-line Python `Process` shell per Rust process initially; add codegen only if the ported-process count makes it worth it.
- **Emitters/listeners (C11):** fold RAMEmitter/JSONEmitter handling into milestone 1 (they run every tick).
- **ODE processes (C10):** port native Rust first; FFI to SUNDIALS/CVODE only if native ODE accuracy/parity is hard.

**USER decisions (resolved 2026-10-27):**
- **(A4) Target this Mac for milestone 1.** Develop + validate the engine MVP on this macOS machine (Homebrew Open MPI 5 — already spike-proven here: rayon, rsmpi, GLPK). The **Slurm/Linux cluster + manylinux wheels are deferred to a later milestone**, not milestone 1. (Removes the untested-target risk from the first milestone.)
- **(B6) Pause the `optimize`-session mypyc lever**; revisit after milestone 1 lands and the real day-one engine-swap number is known. (rust-core supersedes it; avoid duplicated effort without discarding the fallback.)
- **(B5) One `viva-core` wheel** from a single Cargo workspace: `bigraph-schema-rs` + `process-bigraph-rs` + a PyO3 binding crate → one pip-installable wheel.
