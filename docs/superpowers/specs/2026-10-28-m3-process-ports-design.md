# viva-core Milestone 3 — native Rust process ports (design)

**Status:** design, read-only investigation (no production code written)
**Date:** 2026-10-28
**Branch:** `m3-process-ports` (off the M2 zero-copy tip)
**Pinned conformance target:** bigraph-schema 1.6.0 / process-bigraph 1.8.3 (unchanged from M1/M2)
**Goal:** port v2ecoli processes from Python to **native Rust** so the Rust engine
runs native Rust processes (quadrant 2), not PyO3 callbacks (quadrant 3). M1/M2
proved the engine + zero-copy boundary; the remaining ~6× (spec §7: Rust-engine +
Python-processes ≈ 33.5k tps vs fully-native ≈ 208k tps) is unlocked only by
porting hot process bodies. Every port must stay **bit-identical** to its Python
twin — which, for the stochastic processes, means **exact numpy-RNG parity**.

---

## 0. The three cruxes — verdicts up front

1. **RNG bit-identity — FEASIBLE by reimplementation, not by any off-the-shelf crate.**
   numpy's legacy `RandomState` = a standard **MT19937 bit generator** + numpy's
   **own per-distribution algorithms** (the randomkit/`distributions.c` lineage).
   The *bit-generator* layer is reproducible with a stock Rust MT19937; the
   *distribution* layer is numpy-specific and must be reimplemented. Probe-verified
   below. Decision stands (spec §8-B7): **port the stream to native Rust, do not
   FFI numpy's RNG.**
2. **sim_data → config: fully serializable at build-time.** The first-port
   process reads **nothing live** from sim_data on the hot path — all params are
   already extracted into the process config by `sim_data.py::get_*_config` and
   passed through the edge `config`. Confirmed for protein degradation.
3. **First process: `ProteinDegradation`, not elongation.** Simplest complete
   stochastic Step — one distribution (`poisson`), no numba, no ODE, config fully
   serializable, bulk I/O already proven bit-identical in M1/M2. Elongation
   (numba `polymerize` + multinomial + large arrays) is the second wave.

---

## 1. Crux 1 — RNG bit-identity (the long pole)

### 1.1 What v2ecoli actually draws (enumerated from the real process sources)

`grep` over `v2ecoli/processes/*.py` (distribution calls, excluding ParCa which
stays Python):

| distribution | call count | processes |
|---|---|---|
| `multinomial` | 13 | transcript/polypeptide initiation, rna_degradation, elongation |
| `choice` | 9 | rna_degradation, tf_binding, initiation |
| `rand` / `random_sample` | 7 (direct) + **all `stochasticRound`** | equilibrium, tf_binding, transcript_initiation, metabolism, elongation |
| `binomial` | 4 | (initiation/allocation helpers) |
| `poisson` | 3 | **protein_degradation**, polypeptide_initiation, transcript_initiation |
| `randint` | 2 | complexation |
| `normal` | 0 in target processes | (ParCa / plotting only — out of scope) |

Plus `stochasticRound` (defined `v2ecoli/library/random.py:46`), used by
equilibrium (3), metabolism (4), elongation (2+3), tf_binding (2),
transcript_initiation (1) — and it is **pure `rand`** (see 1.4). So `random_sample`
is the foundation under almost everything.

### 1.2 numpy legacy `RandomState` is NOT interchangeable at the distribution level

numpy's `RandomState` (legacy, MT19937) uses its own algorithm per distribution —
`rk_poisson`, `rk_binomial` (BTPE/inversion), `rk_gauss` (cached polar Box–Muller),
masked-rejection `rk_random`/`rk_interval` for `randint`, `multinomial` as repeated
conditional `binomial`. These differ from `rand`/`rand_distr` and from the new
`Generator`/PCG64 path. **No Rust crate reproduces this distribution layer.** The
port must reimplement each algorithm on top of a matching MT19937 stream.

### 1.3 The bit-generator layer IS reproducible — probe-verified

Probe (`scratchpad/rng_probe.py`, run under `.venv-m1`, numpy 2.4.6):

- **Seeding = plain `init_genrand(seed)`** for the small-int seeds v2ecoli uses.
  `RandomState(42).get_state()` → MT19937 key with `key[0]==42` and
  `key[1]==3_107_752_595`, which is exactly `init_genrand(42)`
  (`mt[0]=seed; mt[i]=1812433253*(mt[i-1]^(mt[i-1]>>30))+i`). v2ecoli always seeds
  with a small int (`np.random.RandomState(seed=self.seed)`, seed from
  `_seedFromName`), so the simple scalar seeding applies — **not** `init_by_array`
  (that path is only for seeds ≥ 2³² or array seeds). Any stock MT19937
  (`rand_mt::Mt19937GenRand32::new(seed)` uses exactly `init_genrand`) matches.
- **`random_sample`/`rand` = `genrand_res53`**: two 32-bit words `a,b` →
  `((a>>5)*67108864.0 + (b>>6)) / 9007199254740992.0`. Verified against the
  canonical reference: `RandomState(42).random_sample(3)` →
  `[0.37454011884736..., 0.95071430640991..., 0.73199394181140...]`.
  `rand(3)` is identical (it is `random_sample`).
- **`poisson`** both regimes verified: `lam<10` → Knuth multiplication
  (`rk_poisson_mult`); `lam>=10` → transformed rejection (`rk_poisson_ptrs`,
  needs a Lanczos `loggam` matching numpy's constants); `lam==0` → 0. Vector `lam`
  is drawn **elementwise in C order**, advancing the shared stream per element
  (probe confirmed `poisson([0, 2.5, 25]) = [0, 2, 26]` consumes left-to-right).

### 1.4 `stochasticRound` semantics (exact, pure `rand`)

```python
def stochasticRound(randomState, value):
    value = np.array(value); valueRavel = np.ravel(value)        # C order
    roundUp = randomState.rand(valueRavel.size) < (valueRavel % 1)  # ONE batched rand draw
    valueRavel[roundUp]  = np.ceil(valueRavel[roundUp])
    valueRavel[~roundUp] = np.floor(valueRavel[~roundUp])
    return valueRavel   # (1-D path; the 2-D unravel branch is unused in practice)
```
To reproduce bit-for-bit: ravel in C order, draw `rand(size)` as **one** batch
(stream order matters), `up = u < frac(v)`, then ceil/floor. Probe-verified.

### 1.5 Per-distribution feasibility verdict

| primitive | feasibility | burden | needed by first port? |
|---|---|---|---|
| MT19937 + `init_genrand` seeding | ✅ bit-for-bit | reuse `rand_mt` (verify) or ~80-line module | yes |
| `random_sample`/`rand` (res53) | ✅ bit-for-bit | trivial | (foundation) |
| `stochasticRound` | ✅ bit-for-bit | trivial (pure `rand`) | no, but needed by ~everything after |
| `poisson` | ✅ bit-for-bit | **moderate** — two algos + `loggam` | **yes (protein_degradation)** |
| `binomial` | ✅ bit-for-bit | moderate-hard (BTPE + inversion branch) | second wave |
| `multinomial` | ✅ (once binomial done — repeated conditional binomial) | moderate | second wave |
| `randint` | ✅ bit-for-bit | moderate (legacy masked rejection, not Lemire) | complexation track |
| `choice` | ✅ bit-for-bit | moderate (several branches: replace/p/permutation) | second wave |
| `normal` (`rk_gauss`) | ✅ bit-for-bit (cached-gaussian state) | low-moderate | not in target set |

**Overall RNG verdict: bit-for-bit parity is feasible for every distribution
v2ecoli's non-ParCa processes use.** The make-or-break is solved — it is bounded
reimplementation work, front-loaded once as a shared `rng` module, not an open
risk. No distribution in the target set is "infeasible / bounded-divergent."

### 1.6 The one genuinely hard class — scipy ODE, flag it separately

`two_component_system` and `equilibrium` integrate with `scipy.integrate.solve_ivp`
(adaptive BDF/LSODA/Radau, step-halving on negativity). Bit-identity against
scipy's **adaptive** solver is effectively infeasible in native Rust. These are
**not** first-wave: either FFI to SUNDIALS/CVODE (spec §8-C10 fallback) or accept
bounded-divergent (parity-ladder epsilon, not bit-identical). Keep out of the
bit-identical track.

### 1.7 How to validate bit-for-bit — an oracle, like M1's `gen_oracle.py`

Reuse the M1 oracle pattern (`crates/bigraph-schema-rs/tests/oracle/gen_oracle.py`
→ hard-coded expected values → `cargo test`). Add
`crates/.../tests/oracle/gen_rng_oracle.py` (run under v2ecoli `.venv` or
`.venv-m1`) that dumps, for the exact seeds v2ecoli uses:
- the first N raw MT19937 words + `get_state()` key[0:3] (seeding check),
- `random_sample(N)` and `rand(N)`,
- `poisson(lam)` for `lam ∈ {0, <10 scalar, ≥10 scalar, mixed vector}`,
- `stochasticRound` on a fractional vector.
`cargo test` asserts the Rust `rng` module is **bit-equal** (ints exact; doubles
exact bit pattern, not epsilon — RNG must be exact). Grow the oracle per
distribution as later ports add `binomial`/`multinomial`/`choice`/`randint`.

---

## 2. Crux 2 — sim_data → config sourcing

**A native Rust process never touches the live `sim_data` dill on the hot path.**
v2ecoli already extracts every param at composite-build time in
`v2ecoli/library/sim_data.py` (`get_<process>_config`), and the engine passes it
through the edge `config` to `registry.create(address, config)` → the process
constructor. The Rust process reads the same JSON/array config in its `#[new]`.

Confirmed for protein degradation — `get_protein_degradation_config`
(`sim_data.py:1312`) returns a **fully serializable** dict:

| config key | type | source | hot-path use |
|---|---|---|---|
| `raw_degradation_rate` | `array[float[1/s]]` | `translation.monomer_data["deg_rate"]` (pint→magnitude) | poisson mean `k·dt·n` |
| `amino_acid_counts` | `array[integer]` (n_prot × n_aa) | `monomer_data["aa_counts"]` | builds S matrix once |
| `protein_ids` / `amino_acid_ids` | `list[string]` | `monomer_data["id"]`, `molecule_groups.amino_acids` | bulk index mapping (t=0) |
| `water_id` | `string` | `molecule_ids.water` | S matrix water row |
| `protein_lengths` | `array[integer[aa]]` | `monomer_data["length"]` | **vestigial** (read, never used) |
| `seed` | `integer` | `_seedFromName("ProteinDegradation")` | `RandomState(seed)` |

The S (stoichiometry) matrix is built **once** in `initialize()` from config — no
per-tick Python. Bulk index resolution (`bulk_name_to_idx`) happens once at t=0
from the live `bulk["id"]` array it already receives through the zero-copy view.

**Plan:** config is extracted once at build-time into the process config
(unchanged from today's Python flow); the Rust constructor deserializes arrays
(Vec<f64>, Vec<Vec<i64>>, Vec<String>) and precomputes the S matrix. No live
Python needed for this process. Later ports (elongation, initiation) need bigger
matrices/rate tables but the same build-time-extraction pattern holds — verify
per-process that no config value is a Python callable or a live sim_data handle
(none observed in `get_*_config` for the first-wave set).

---

## 3. Crux 3 — first process + the port order

### 3.1 Recommended first port: `ProteinDegradation` (`v2ecoli/processes/protein_degradation.py`, 221 lines)

Why it beats "elongation first" (spec §4/§6 said elongation; M3 should go simpler
first, and the task explicitly asks to reassess):

- **One distribution only** (`poisson`) — exercises the RNG long pole on the
  cleanest possible footprint. No `stochasticRound`, no `multinomial`, no `choice`.
- **No numba, no scipy/ODE.** (Elongation = numba `polymerize` + 2× multinomial +
  large arrays; a much bigger bite for a *first* port.)
- **It is a `Step`** (single-pass; no request/allocate/evolve cycle) — minimal
  control flow.
- **Config fully serializable** (§2) — proves the sim_data→config pattern cleanly.
- **I/O already proven bit-identical.** `update()` returns
  `{"bulk": [(idx_array, delta), (protein_idx, -n)]}` — the **vectorized
  `(idx_array, delta_array)` bulk shape M1 Task 8 + M2 already proved bit-identical**
  across the boundary. Nothing new on the apply side.
- **i64-safe** (spec A2 risk): counts and `S @ n_degraded` deltas are molecule
  counts (≪ 2⁶³); no bigint exposure. The only cap is `np.fmin(poisson, counts)` —
  trivial in Rust.

The port therefore isolates exactly one new thing — **native RNG parity under the
real engine** — while reusing the proven bulk path and the proven config flow.

### 3.2 Order (bit-identical track)

0. **RNG foundation module** (shared `rng/`): MT19937 + `init_genrand` seeding +
   `random_sample` (res53) + `stochasticRound` + `poisson`, with the oracle
   (§1.7). This is the long pole; do it as the first sub-task of the first port.
1. **`ProteinDegradation`** — first full native process (poisson only). ← the first task.
2. **`rna_degradation`** (801 LOC; `multinomial`×2 + `choice`) **or** `tf_binding`
   (`stochasticRound` + `choice`) — adds `multinomial`/`choice`/`binomial` to the
   `rng` module; still no numba/ODE. (Pick the one with the smaller config surface.)
3. **Elongation second wave** — `transcript_elongation` then
   `polypeptide_elongation`: port numba `polymerize` to native Rust (the design's
   top numba target) + `multinomial` + the big elongation arrays. Highest value,
   hardest; do it only once RNG + native-Process + config are proven on 1–2.
4. **Initiation** (`transcript_initiation`, `polypeptide_initiation`) — poisson +
   multinomial + choice; reuses everything above.

### 3.3 Parallel track — GLPK / metabolism (no RNG long pole)

`metabolism.py` has **no sampled distribution** — its only randomness is
`stochasticRound` (flux→integer-count conversion at the end), which the foundation
(§3.2-0) already provides. The science is **FBA via the GLPK C solver**, and the
**M1 spike already proved the GLPK `extern "C"` FFI** (spec §7). So metabolism is a
**separate, parallelizable track**: FFI-reuse GLPK (do not re-derive FBA) + the
shared `stochasticRound`. Likewise `complexation` (Gillespie SSA via `arrowhead.so`
FFI + `randint`) is an FFI track, not a native-RNG port — though its `arrowhead`
FFI was **not** spiked in M1 and `randint` parity would be new work. Sequence the
FFI tracks after the first native port proves the registration/conformance
machinery, but they do not depend on the RNG distribution ladder.

---

## 4. Conformance gate (reuse M1/M2 harness)

The gate is the existing **decoded-state bit-identical** ladder (spec A2): integer
counts **exact**, floats within `1e-12`; RNG draws themselves checked **exactly**
(bit pattern) via the oracle.

- **Rust-unit layer:** `cargo test --workspace` + the new `gen_rng_oracle.py`
  oracle (§1.7) — the Rust `rng` module bit-equal to numpy for every seed/dist the
  port uses.
- **Cross-language process layer:** extend `python/viva_core/conformance.py`
  (M1/M2's decoded-state comparator, mirrors v2ecoli `distilled_compare.py`) to run
  the **same** protein-degradation composite twice under the Rust engine — once
  with the process as a **Python `PyProcess`** (quadrant 3) and once with the
  **native Rust factory** registered at the same `local:` address (quadrant 2) —
  for N ticks, asserting the decoded `bulk` state is **bit-identical** each tick.
  Because both share the same seed, parity is **exact**, not statistical. Add the
  run()-level case to `upstream_suite.py` Part 2 (the 6/6 battery → 7/7).
- **Edge cases to force:** `lam==0` rows, the `np.fmin(poisson, counts)` cap,
  vector-`lam` stream-order, and a multi-tick run so RNG stream advancement (not
  just the first draw) is checked.
- **Build discipline (M2 lesson):** benchmark the **release** build (debug inverts
  ratios); report debug too for like-for-like.

---

## 5. First task (concrete, smallest bit-identical native port)

**Title:** Port `ProteinDegradation` to a native Rust `Process`, bit-identical to
its Python twin under the real engine, RNG included, re-benchmarked.

**Steps:**
1. **RNG foundation** — add a `rng` module (in `process-bigraph-rs` or a new
   `viva-rng` crate): MT19937 (`init_genrand` seeding) + `random_sample` (res53) +
   `stochasticRound` + `poisson` (both regimes + `loggam` + vector/C-order + lam=0).
   First verify whether `rand_mt::Mt19937GenRand32` reproduces numpy's raw stream
   (seed→first words) against the oracle; if yes reuse it, else ~80-line MT. Land
   `gen_rng_oracle.py` + `cargo test` bit-equality (§1.7).
2. **`ProteinDegradationCore`** — a native Rust `Process` (impl the `process.rs`
   trait: `inputs`/`outputs`/`update`; it's a Step → default `perform_update`).
   `#[new]` deserializes config (`raw_degradation_rate: Vec<f64>`,
   `amino_acid_counts: Vec<Vec<i64>>`, `protein_ids`/`amino_acid_ids: Vec<String>`,
   `water_id: String`, `seed: i64`), precomputes the S matrix once. `update()`:
   read `bulk` counts via the M2 zero-copy view, `n = min(poisson(k·dt·n), n)`,
   `delta = S @ n`, emit the vectorized `(idx_array, delta_array)` bulk op (the
   proven shape).
3. **Registration** — in `viva-core`'s `build_registry` (lib.rs ~131), register the
   native factory for the protein-degradation address **ahead of** the `PyProcess`
   fallback, behind a flag/allow-list so the *same* `.pbg` runs Python or native.
4. **Conformance** — extend `conformance.py`: run the protein-degradation composite
   Python-process vs native-Rust-process under the Rust engine, N ticks, assert
   decoded `bulk` bit-identical; add to `upstream_suite.py` Part 2. Force the edge
   cases in §4.
5. **Benchmark** — `bench_m1.py`-style, **release** build: whole-composite ticks/s
   with the native process vs the Python process. This is the **first real
   quadrant-2 measurement** (M1 §2 noted the full-native ceiling was not even
   measurable through `run()` in M1 — this task makes it measurable).

**Guardrails:** the throwaway RNG probe stays in `scratchpad/` (out of the repo);
no AI-attribution trailers; worktree already established (`m3-process-ports`).

---

## 6. Evidence index
- Probe: `scratchpad/rng_probe.py` (numpy 2.4.6 under `.venv-m1`) — seeding,
  res53, poisson regimes, stochasticRound, binomial/multinomial/randint samples.
- Source read: `v2ecoli/processes/protein_degradation.py`,
  `v2ecoli/library/random.py`, `v2ecoli/library/sim_data.py:1312` + grep of all
  `processes/*.py` distribution calls.
- Engine: `crates/process-bigraph-rs/src/{engine,process,registry}.rs`,
  `crates/viva-core/src/{rust_process,pyproc,lib}.rs`.
- M1/M2: `docs/milestone-1-results.md`, `docs/milestone-2-results.md`,
  `crates/bigraph-schema-rs/tests/oracle/gen_oracle.py`,
  `python/viva_core/{conformance,upstream_suite}.py`.
