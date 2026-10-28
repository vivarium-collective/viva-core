# viva-core Milestone 4 — the whole-model speedup number (design)

**Status:** design, read-only investigation (no production code written)
**Date:** 2026-10-28
**Branch:** `m4-wholemodel` (off clean main = M1+M2+M3 merged, HEAD 35e895b)
**Pinned conformance target:** bigraph-schema 1.6.0 / process-bigraph 1.8.3 (unchanged)
**Goal:** the first *honest, end-to-end* "is the Rust engine faster than Python pb
on a real (or near-real) v2ecoli whole-cell model" number. M1 proved the engine +
zero-copy boundary (release: bulk ~1.2×, unique ~1.4×, scalar ~6×); M3 proved two
native process ports bit-identical (ProteinDegradation full, metabolism solve only).
None of those is a whole-model number — all are micro-composites of one process.

---

## 0. Verdicts up front

1. **The real wall is NOT import drift — it is the entry surface.** `viva_core.run`
   takes a JSON spec (path or string) and the engine *constructs every process
   itself* from the JSON `cfg` (`lib.rs:193` → `PyProcess::create(py, addr, cfg)`).
   But **11 of the baseline's process types carry live, non-JSON-serializable
   sim_data handles** (`'_type':'node'` / `'method'`) in their config. A JSON `.pbg`
   cannot carry those, so the model **cannot be constructed from a JSON spec at all**.
   This dwarfs the listeners→derivers import drift (a trivial regen fix).
2. **Recommended M4 scope = option (iii):** add a thin **Python→Rust composite
   handoff** so v2ecoli's own live builder (`ecoli_baseline.py`, which already
   injects the sim_data handles) constructs the baseline composite, and the Rust
   engine *schedules the pre-built Python process instances* (quadrant 3) with the
   already-ported native processes swapped in. Measure **one generation** (no
   division). This is the one honest whole-model signal achievable in a milestone.
3. **Metabolism → hybrid (C).** Near-free: the native solve (`glpk::solve`) and
   `flux_to_bulk_delta` are **already ported, bit-identical, and already exposed as
   `viva_core.fba_solve` / `viva_core.flux_to_bulk_delta` pyfunctions** (`lib.rs:285,335`).
   A full assembly port (A/B) is out of scope and high-risk; defer.
4. **Elongation → class (a) for baseline** (its `make_elongation_rates` 'method'
   wraps *static* data + trivial `stochasticRound`), unlike metabolism. Portable,
   but it is the **largest single port** and carries one genuine bit-identity
   hazard: `polymerize`'s rare limit-hit paths draw on the **global** numpy RNG.
   Defer the full port past M4; the handoff is what unblocks the number.

---

## 1. Question 1 — the whole-model-runnable path (the crux)

### 1.1 What the full model actually is
`ecoli_baseline.py` docstring: **55 processes, partitioned**. The partitioned
architecture splits hot processes into **Requester + Evolver** pairs with
**Allocator** layers between them, plus derivers (the former listeners, now
`v2ecoli/steps/derivers/*`), `GlobalClock`, `ExchangeData`, `MediaUpdate`,
`Division`, emitters. Config is assembled by a `CachedConfigLoader` from the ParCa
`sim_data` cache — **already on disk** at `out/cache/sim_data_cache.dill` and
`tests/fixtures/cache/sim_data_cache.dill`, so **ParCa need not be re-run**.

### 1.2 What blocks running a full model today (enumerated)

| # | Blocker | Severity | Fix |
|---|---|---|---|
| 1 | **Live sim_data handles in config** — 11 process types declare `'node'`/`'method'` config entries (counts: metabolism 7, two_component_system 3, equilibrium 3, chromosome_replication 3, transcript_initiation 2, transcript_elongation 2, polypeptide_initiation 2, ppgpp_initiation 1, tf_binding 1, polypeptide_elongation 1, chromosome_structure 1). These are live Python objects/bound methods, **not JSON-serializable**. | **Fatal for JSON path** | Python-side composite handoff (§1.3) |
| 2 | **No Python→Rust composite entry** — `viva_core.run` only takes a JSON spec string/path and *constructs* processes from `cfg`. There is no way to hand it pre-built process instances. | **Fatal** | new `run_prebuilt` pyfunction |
| 3 | **Committed `partitioned.pbg` (15 MB) is stale** — 32 `local:` addresses, incl. `v2ecoli.steps.listeners.*` (reorg'd to `steps.derivers.*`; those modules no longer import) and it predates the 55-process partitioned arch. | Medium (moot once §1.3) | not needed if we build live |
| 4 | **Division = topology mutation** (spawn daughters; vivarium-/`_add`-style), untested on the Rust engine. | Medium | **sidestep — measure one generation** |
| 5 | Partition Requester/Evolver/Allocator + derivers are *extra Steps/Processes*. | Low | handled as ordinary PyProcesses; M1 already runs Steps + gating |
| 6 | Emitters/listeners run every tick. | Low | M1 already handles RAMEmitter; derivers are ordinary Steps |

### 1.3 The three options, judged

- **(i) Regenerate a current `.pbg` + inject sim_data → full 16/55-process model via JSON.**
  **Infeasible as a JSON round-trip.** Blocker #1 is structural: the sim_data
  `'node'`/`'method'` handles can never serialize into the `.pbg`, and the Rust
  engine's only construction path is from that JSON `cfg`. Would require both a
  full assembly/sim_data port *and* a serialization scheme for live handles.
  Reject for M4.

- **(ii) Representative multi-process sub-composite, end-to-end.**
  Achievable and real, but a *sub-model*. Pick the **config-serializable** set
  (no `'node'`/`'method'`): ProteinDegradation (native ✓) + e.g. rna_degradation,
  tf_unbinding, a deriver, GlobalClock, emitter — wire a small `.pbg`, run on both
  engines. Honest but **not the whole-model signal** the user wants. Keep as the
  **conformance anchor / de-risking first step**, not the headline.

- **(iii) Full baseline model, mostly Python-on-Rust + ported-native swapped in.** ✅ **RECOMMENDED.**
  Let v2ecoli's **own live builder** construct the real 55-process baseline (it
  already injects the sim_data handles into each Python process instance), then
  hand the *instances* to the Rust engine via a new `run_prebuilt`. The engine
  schedules them as `PyProcess` shims (quadrant 3), swapping the native twin for
  any address in `native_ctor` (today: ProteinDegradation; metabolism-solve via
  the hybrid). Measure **one generation** (no division event) ticks/s, Rust engine
  vs pb, same composite + master_seed. **This is the real whole-model number**:
  the engine-swap gain on the actual model, plus the partial-port gain, with an
  honest "most processes are still Python-on-Rust" caveat.

**Why (iii) fits one milestone:** no new *process* port is strictly required to
get the first number — it is **plumbing** (one pyfunction + a PyProcess variant
that calls a *given* instance's `update` instead of importing+constructing from
JSON) + reuse of `ecoli_baseline.py` + the existing M1/M2 boundary + the existing
ProteinDegradation native twin.

### 1.4 Minimal steps to a real end-to-end number
1. Add `viva_core.run_prebuilt(processes, initial_state, n_ticks, *, native_allowlist)`:
   accept already-instantiated Python process objects + their address + topology +
   decoded initial state; register each as a `PyProcess` **bound to the instance**
   (new shim path: call `instance.update(...)`, do **not** re-import/re-construct
   from `cfg` — this is what carries the live sim_data handles across). Swap the
   native ctor for allow-listed addresses.
2. In v2ecoli (or a scratch harness), build the baseline composite from the cached
   sim_data via `ecoli_baseline.py`; extract (address, instance, topology) + the
   initial state.
3. Run K ticks on **pb** and on **viva_core.run_prebuilt** within one generation;
   assert decoded-state bit-identical (§5) — the conformance gate.
4. Benchmark ticks/s (release build) on both; report Rust/pb ratio with
   native-swap on vs off. First honest whole-model number.
5. Repeat with metabolism on the **hybrid** solve (§2) to show the per-tick
   metabolism gain inside the whole-model run.

---

## 2. Question 2 — metabolism (b)-class assembly port → **recommend C (hybrid)**

`Metabolism._do_update` (metabolism.py:656+) re-assembles the LP **every tick**
from **live, per-tick-dynamic** inputs funneled through the pint/Unum unit bridge:
- `getBiomassAsConcentrations(doubling_time, rp_ratio=...)`, `getppGppConc(doubling_time)`
  — `'method'` handles; `doubling_time`/`rp_ratio` derived from **live listener mass**
  (`cell_mass`, `dry_mass`, `rna_mass`, `protein_mass`).
- `get_import_constraints(...)` / `exchange_constraints` — read `environment.exchange_data`
  (`unconstrained`/`constrained`) and `environment.media_id` (live).
- Cross-process reads: `polypeptide_elongation.gtp_to_hydrolyze`, `aa_count_diff`,
  `bulk_total` AA counts; `counts_to_molar` from live cell volume.

**(A) serialize sim_data into config — rejected.** The assembly's inputs are
**per-tick dynamic whole-cell state**, not static params. Only the *rate tables /
stoichiometry* are static; the bounds, objective targets and exchange constraints
are recomputed each tick from live mass/environment/cross-process state and the
unit bridge. You cannot precompute them.

**(B) full assembly port to Rust — rejected for M4.** Means porting
`wholecell/utils/modular_fba.py` (~1.6k LOC) + the netflow builder + several
sim_data dataclass methods + the pint/Unum bridge. Multi-milestone, and the unit
bridge + LP-assembly ordering are a high bit-identity risk.

**(C) hybrid — recommended, near-free.** Keep assembly in Python; call the
**native solve + flux→count** from it. Both halves are **already ported,
bit-identical, and already exposed** as `viva_core.fba_solve(lp_dict)` and
`viva_core.flux_to_bulk_delta(...)` (lib.rs:285, 335). Effort = wire those two
calls into `Metabolism._do_update` in place of the swiglpk solve + Python
stochasticRound. **Bit-identity implication:** trivial — the solve is proven
bit-identical cold-vs-cold (M3: exact `==` on objective + flux vector), the
flux→count uses the shared numpy-parity `stochastic_round`, and the assembly is
unchanged Python. Caveat to carry: Python warm-starts one LP across ticks; the
native call is cold per call — verify multi-tick warm-start equivalence or accept
the cold-solve cost (M3 measured solve-only 3.6× even cold).

---

## 3. Question 3 — elongation port assessment

### 3.1 Distributions drawn
- `polymerize` kernel (`library/polymerize.py`, 577 LOC): in the **rare**
  resource-/reaction-limited final-step paths — `np.random.random` (`_sample_array`)
  and `np.random.shuffle` (`_choices`) on the **GLOBAL** numpy RNG; plus
  `self._randomState.choice` ×2 in `_finalize_resource_limited_elongations` (seeded).
- `make_elongation_rates`: `stochasticRound` (pure `rand`) — already in M3 rng.
- `polypeptide_elongation.py`: `stochasticRound` ×2 (tRNA charging),
  `self.random_state.multinomial` ×2 (AA-target adjustment).
- `transcript_elongation.py`: `stochasticRound` (attenuation) + `polymerize`.

### 3.2 Bit-for-bit reproducibility by extending the M3 rng module
- **Have:** MT19937 + `init_genrand` seeding, `random_sample`/`rand`, `stochasticRound`.
- **Add:** `multinomial` (M3 design: once `binomial` lands — repeated conditional
  binomial), `choice`, and the two globals used by polymerize: `random` (trivial,
  = `random_sample`) and **`shuffle`** (numpy legacy Fisher–Yates, reverse
  iteration via `rk_interval`/`randint` — same masked-rejection as `randint`).
  All feasible per the M3 verdict (no distribution in the target set is
  bounded-divergent).
- **The one genuine hazard — global-RNG coupling.** `polymerize` consumes the
  **process-global** `np.random` stream (not the process's seeded `random_state`),
  so bit-identity requires reproducing a matching *global* MT19937 stream **and**
  knowing its exact state at the draw — which depends on every prior global-RNG
  consumer in the tick. These fire only in limit-hit paths (rare, but they *do*
  occur under resource limitation). This, not the kernel math, is why elongation
  is the hard port. Options: mirror a dedicated global stream in Rust seeded +
  advanced identically, or route the rare limit-hit branch back through Python
  (as upstream already does for the shuffle step — polymerize.py:20-21,461).

### 3.3 Size and class
- **Size: largest single port.** `polymerize.py` 577 LOC (3 numba njit kernels —
  `sum_monomers`/`buildSequences`/`computeMassIncrease`: clean integer/array
  arithmetic; 2 elongate loops fixed/variable; the `PolymerizeMatrices`
  orchestration) + the hot slices of the two elongation processes (~2.6k LOC total,
  but most is ports/topology/derivers, not compute). Materially bigger than
  ProteinDegradation.
- **Class: (a) for the baseline config.** Its `'method'` handle
  `make_elongation_rates` wraps **static** data (`n_monomers`,
  `ribosomal_protein_indexes`, `max_elongation_rate`) + trivial
  `stochasticRound(rs, dt·rates)` — effectively config-serializable, unlike
  metabolism. The optional `get_attenuation_stop_probabilities` (tRNA attenuation)
  and trna_charging features read **live** concentrations (charged_tRNA) → those
  features push it toward (b)/hybrid. **Baseline = (a); with attenuation/charging = (b)-leaning.**

**Recommendation:** defer the full elongation port past M4. In the M4 whole-model
run it stays Python-on-Rust (quadrant 3). Make it the headline *next* native port
(M5), starting with the `multinomial`/`choice`/`shuffle` rng additions + the numba
kernels, baseline config (attenuation off) first.

---

## 4. Recommended M4 scope (deliver vs defer — one milestone)

**Deliver:**
1. **Python→Rust composite handoff** (`run_prebuilt` + instance-bound PyProcess shim).
2. **First whole-model number:** real 55-process baseline, one generation, Rust
   engine (quadrant 3 + ProteinDegradation native) vs pb, release build.
3. **Metabolism hybrid** wired into the whole-model run (reuse existing
   `fba_solve`/`flux_to_bulk_delta` pyfunctions) — per-tick metabolism gain in situ.
4. **Conformance:** decoded-state bit-identical gate on the baseline composite
   (§5); upstream-suite battery grows by the whole-model case.

**Defer:** full metabolism assembly port (A/B); full elongation port; division /
multi-generation; manylinux/Linux-cluster builds; the stale `partitioned.pbg` regen
(moot under the handoff). State these explicitly in the results doc.

---

## 5. Bit-identical conformance gate (reuse M1/M2/M3 ladder)
- **Decoded-state, same composite, same master_seed, within one generation:** run
  K ticks on pb and on `run_prebuilt`; integer molecule counts **exact**, floats
  within `1e-12` (spec A2). Compare `bulk`, `unique`, `listeners`/deriver outputs
  each tick, not just the end state.
- **RNG-exact** for any native-swapped process (ProteinDegradation, and
  metabolism's flux→count): same seed → exact, via the existing `gen_rng_oracle`
  (`cargo test --workspace`).
- **Metabolism hybrid:** assert `fba_solve`/`flux_to_bulk_delta` outputs `==` the
  swiglpk+numpy path on the live-assembled LP, each tick.
- **Build discipline (M2 lesson):** benchmark the **release** build; report debug
  for like-for-like. i64 audit still applies to any newly-crossed counters.
- **Honesty caveat to carry forward:** this is "Rust engine + *mostly Python*
  processes + a few native" — the number is the engine-swap gain on the real model
  plus a partial-port gain, **not** the full-native ceiling. Say so.

---

## 6. Concrete FIRST task

**Title:** Add a Python→Rust composite handoff and measure the first whole-model
number on the real baseline composite (one generation), bit-identical to pb.

**Steps:**
1. **`viva_core.run_prebuilt(processes, initial_state, n_ticks, native_allowlist=...)`**
   — a pyfunction that takes a list of `(address, py_instance, topology)` + decoded
   initial state, builds the engine, and for each address registers either the
   `native_ctor` (if allow-listed) or a **new instance-bound `PyProcess`** shim
   that invokes the *given* `py_instance.update(...)` (carrying its live sim_data
   handles) instead of importing+constructing from JSON `cfg`. Reuse M1/M2 zero-copy
   views + the existing scheduler.
2. **Harness (scratch, out of repo):** under the shared `v2ecoli/.venv`
   (read-only), build the baseline composite from `out/cache/sim_data_cache.dill`
   via `ecoli_baseline.py`; extract `(address, instance, topology)` + initial state.
3. **Conformance:** run K (e.g. 8–16) ticks on pb and on `run_prebuilt`; assert
   decoded-state bit-identical (§5). Fail-loud on first divergence with the path.
4. **Benchmark (release):** ticks/s on pb vs `run_prebuilt`, native-swap off vs on
   (ProteinDegradation; then metabolism hybrid). Report the ratio + the honest
   "mostly Python-on-Rust" caveat → `docs/milestone-4-results.md`.

**Guardrails:** throwaway harness/probe stays in `scratchpad/` (out of the repo);
build viva_core wheels into a **fresh local venv** (do not modify the shared
`v2ecoli/.venv`); no AI-attribution trailers; worktree already established
(`m4-wholemodel`).

---

## 7. Evidence index
- Entry surface: `crates/viva-core/src/lib.rs` (`run`/`run_report` JSON-only;
  `build_engine` `PyProcess::create(addr,cfg)` at :193; `fba_solve` :285,
  `flux_to_bulk_delta` :335).
- Native twins: `crates/viva-core/src/processes/{protein_degradation,metabolism}.rs`
  (metabolism `update` fail-loud; solve + flux→count ported).
- Full model: `v2ecoli/composites/ecoli_baseline.py` (55 processes, partitioned),
  `_helpers.py` (PARTITIONED_PROCESSES, ALLOCATOR_LAYERS), `steps/partition.py`
  (Requester/Evolver), `steps/derivers/*` (former listeners).
- Live sim_data handles: grep of `'_type':'node'|'method'` across
  `v2ecoli/processes/*` + `steps/*` (11 process types; metabolism 7).
- Metabolism per-tick deps: `v2ecoli/processes/metabolism.py:656+` (`_do_update`).
- Elongation: `v2ecoli/processes/{polypeptide,transcript}_elongation.py`,
  `v2ecoli/library/polymerize.py` (global-RNG paths :132,142,415,461; choice :534),
  `parca/.../translation.py:312` + `parca/wholecell/utils/random.py:87`
  (`make_elongation_rates` = static data + stochasticRound).
- Stale pbg: `models/partitioned.pbg` (32 `local:` addrs incl. drifted
  `steps.listeners.*`). sim_data cache present: `out/cache/sim_data_cache.dill`.
- Prior milestones: `docs/milestone-{1,2,3}-results.md`,
  `docs/superpowers/specs/2026-10-28-m3-process-ports-design.md`.
