# viva-core Milestone 5 — the whole model RUNS on the Rust engine (plan)

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development. Steps use `- [ ]`.

**Goal:** Make the REAL v2ecoli baseline composite (one generation, baseline media, seed 0) run for N ticks under `viva_core.run_prebuilt`, native-swap OFF, **bit-identical to process-bigraph 1.8.3** — every process running as its pre-built Python instance on the Rust engine, with no store the handoff cannot carry. Enablement/correctness milestone, NOT speed (whole-model stays <1× vs pb until M6+ ports).

**Approach:** Remove the four handoff blockers found in the M5 investigation: (1) a pint **unit bridge** at the PyO3 boundary (Option A — opaque tagged Value in Rust, wrap/unwrap in `convert.rs`); (2) a Python-owned **handle table** for the live `process` store and `allocator_rng` (Option 1 — Rust never serializes them, the view returns the live object); (3) a **realized-`Composite` handoff**; (4) **validate** the whole model end-to-end and close the two known-unknowns (listener `ndarray`→list; non-overwrite nested listener merge).

**Tech:** Rust, pyo3 0.29 + rust-numpy; the M1–M4 conformance harness as the bit-identical gate; maturin; the real baseline via v2ecoli `ecoli_baseline.baseline` + on-disk sim_data.

**Spec:** `docs/superpowers/specs/2026-10-28-m5-wholemodel-bridge-design.md` (+ the companion `2026-10-28-m5-investigation.md`).

## Global Constraints
- **Bit-identical to pb 1.8.3** on the real baseline: decoded state int-exact, floats ≤1e-12, RNG-exact; Quantity leaves equal in BOTH magnitude AND units.
- **No v2ecoli change.** Shared `/Users/eranagmon/code/v2ecoli/.venv` + canonical `~/code/v2ecoli` are READ-ONLY. Wheels into a FRESH local venv. sim_data cache on disk (`out/cache/`); no ParCa run.
- Hot path (`apply`/`reconcile`/scheduler) stays **unit-unaware** — units opaque in Rust state.
- Current v2ecoli main is @20c69d78 (post sim_data-decouple). Re-verify the bridge against it in Task 1 Step 0.
- No AI attribution; `cargo fmt` + `cargo clippy --workspace --all-targets -- -D warnings` clean; GLPK feature-gated (default wheel GLPK-free); M1–M4 conformance stays green.

## Review Focus
- **Scalar `np.float64`→`float` demotion (unit bridge):** the tagged round-trip must preserve the magnitude dtype or `1/Quantity(0.0)` flips from `inf` (np.float64) to `ZeroDivisionError` (float); mass divisors hit this. The bridge test must include a zero-valued mass divisor.
- **The one strict unit read:** `polypeptide_initiation.py:318` does `eff_rate.to(units.aa/units.s)` on `effective_elongation_rate` (written by `polypeptide_elongation.py:743` the prior tick). A bare float or tag-dict raises `AttributeError` — the field must survive write→read across ticks as a real Quantity.
- **Final-state parity, not just reads:** the state the comparator diffs against pb contains Quantities; the write side must preserve them or the diff is type-different even when magnitude-equal.
- **Handle identity / RNG stream:** the `process` and `allocator_rng` handles must be the SAME object pb uses, mutated in place — a copy of `allocator_rng` would diverge the RNG stream. Writes to a handle path must be dropped (no-op overwrite of the same object), and the Requester→Evolver ordering edge preserved.
- **Non-overwrite nested listener merges:** listener output types that aren't `overwrite[...]` hit the `_ → Op::Overwrite` arm; verify that matches pb's merge for nested listener dicts, or record the divergence.

---

### Task 1: Unit bridge (Option A) + bridge pre-flight

**Files:** Modify `crates/viva-core/src/convert.rs` (`py_to_value` Quantity arm; split the `value_to_py` `Tagged` arm so `Tagged(Pint,..)` rebuilds a Quantity; a cached registry handle + unit-string parse cache). Modify `python/viva_core/prebuilt.py` (`_serialize` Quantity case). Modify `crates/viva-core/src/lib.rs`/`convert.rs` `state_to_py` to reuse the Pint rebuild. Test: `python/tests/test_unit_bridge.py`.

**Interfaces:**
- Pint wire shape (from json_codec.py): `{"__pint__": true, "magnitude": <f64|list>, "units": str(q.units)[, "__magnitude_dtype__": "<dtype>"]}`. The scalar branch has no dtype tag upstream — the bridge adds an extension key (e.g. `"__magnitude_dtype__": "float64"` or a viva-core-private key) so `value_to_py` can rebuild `np.float64` vs Python `float`. The extra key must round-trip through the Rust codec (tagged maps preserve unknown keys) and be ignored by `bigraph_json_hook`.
- `value_to_py(Tagged(Pint, m))` → `get_quantity_registry().Quantity(mag, units)` with `mag` as `np.float64(x)` / `np.array(x, dtype)`.

- [ ] Step 0 (pre-flight): build the wheel into a fresh venv; run the existing `python/tests/test_run_prebuilt.py` + `test_subarray_unique.py` against current v2ecoli main (@20c69d78). Confirm green (run_prebuilt is post-resolve; decoupled configs expected harmless). Record the result. If red, STOP and report — the bridge re-verify is a precondition.
- [ ] Step 1: Write failing `test_unit_bridge.py`: (a) round-trip a scalar `np.float64` Quantity (e.g. `femtogram`) and a `1 / second` rate and `amino_acid / second` through `py_to_value`→state→`value_to_py`, asserting type `pint.Quantity`, unit equality, magnitude bit-exact, AND magnitude dtype preserved (`type(q.magnitude) is np.float64`); (b) a zero-valued mass Quantity where `1/q` must give `inf` not raise; (c) `_serialize` on a Quantity emits the tagged dict (json-dumpable).
- [ ] Step 2: Run → FAIL (today: Null / plain dict / recursion).
- [ ] Step 3: Implement the two `convert.rs` arms + registry/unit cache + `_serialize` case + `state_to_py` reuse.
- [ ] Step 4: Run → PASS; M1–M4 conformance green (flag off); `cargo test --workspace` + clippy + fmt.
- [ ] Step 5: Commit (no attribution).

---

### Task 2: Handle table for the live `process` store + `allocator_rng` (Option 1)

**Files:** Modify `crates/viva-core/src/convert.rs` (extend the per-path object table — add `handles: HashMap<Vec<String>, Py<PyAny>>` alongside `DtypeMap.ids`; `storenode_to_py` returns the handle object for a handle path; the update boundary drops writes to a handle path, resolving absolute path via the view path). Modify `crates/viva-core/src/lib.rs`/`pyproc.rs` (thread `handles` into the engine build). Modify `python/viva_core/prebuilt.py` (`build_payload` splits live-object stores → `handles={path: obj}`; state JSON gets a placeholder at those paths; `_serialize` no longer errors on them). Test: `python/tests/test_handle_store.py`.

**Interfaces:**
- `build_payload` → `{processes, topology, state, handles}` where `handles` maps `["process", <name>]` → the live `(instance,)` tuple and `["allocator_rng"]` → the live `RandomState`.
- `storenode_to_py`: if the absolute path is a handle path, return the handle object verbatim (the `(instance,)` tuple so `proc_state[0]` works; the `RandomState` itself) instead of decoding a `Value`.

- [ ] Step 1: Write failing `test_handle_store.py`: a small real slice with the 3 partitioned processes (Requester/Evolver) + the Allocator, run N ticks under pb and under `run_prebuilt`; assert (a) bulk/listener state bit-identical, (b) the handle object handed to a view `is` the one pb uses (identity, incl. `allocator_rng`), (c) a write to a handle path is a no-op (same object), (d) RNG stream unchanged (allocator draws match pb).
- [ ] Step 2: Run → FAIL (today: TypeError at serialize / `[None]` view).
- [ ] Step 3: Implement the handle table + placeholder state + view/update boundary + `build_payload` split.
- [ ] Step 4: Run → PASS; Task 1 + M1–M4 conformance green; cargo/clippy/fmt.
- [ ] Step 5: Commit.

---

### Task 3: Realized-`Composite` handoff

**Files:** Modify `python/viva_core/prebuilt.py` (`build_payload`/the `run_prebuilt` entry so it accepts a realized `process_bigraph.Composite` — or realizes the agent — so every edge carries a live `instance`, e.g. `shape_step`). Test: extend `python/tests/test_run_prebuilt.py` or a new `test_realized_handoff.py`.

**Interfaces:** `run_prebuilt` accepts either a realized `Composite` (preferred) or realizes the passed agent state; `build_payload` no longer raises `ValueError("no realized 'instance'")` on a fully-built baseline agent.

- [ ] Step 1: Write failing test: build the REAL baseline agent, realize it as a `Composite`, hand it to `run_prebuilt`; assert `build_payload` does NOT raise the unrealized-instance `ValueError` and the handoff proceeds (combined with Tasks 1+2, the whole agent is now serializable/handable).
- [ ] Step 2: Run → FAIL (ValueError today).
- [ ] Step 3: Implement realized-composite acceptance.
- [ ] Step 4: Run → PASS; prior tasks + M1–M4 green; cargo/clippy/fmt.
- [ ] Step 5: Commit.

---

### Task 4: Whole-model conformance + close the two known-unknowns + M5 results

**Files:** `python/tests/test_wholemodel_runs.py` (new); `python/benchmarks/bench_m5_wholemodel.py` (new, reuse M4's harness); `docs/milestone-5-results.md` (new).

**Interfaces:** run the REAL baseline whole model (one generation, baseline media, seed 0) under pb `Composite` and `viva_core.run_prebuilt` (native-swap OFF) for N ticks; conformance-gate decoded state bit-identical (int-exact, float ≤1e-12, RNG-exact, Quantity magnitude+units equal). Then diff the two known-unknowns against pb and record.

- [ ] Step 1: Write failing `test_wholemodel_runs`: the whole baseline runs N ticks under both engines bit-identical. ALSO two focused checks: (a) a `listeners` leaf that is an `ndarray` round-trips so a reader doing array math sees the same values (document if it's a list vs array type change and whether any real reader is affected); (b) a non-`overwrite` nested listener output merges identically to pb. If any process still can't run, record EXACTLY which + why (honest — do not fake a full run).
- [ ] Step 2: Run → FAIL (or surface the real residual blocker honestly).
- [ ] Step 3: Implement the harness; get the whole model running end-to-end bit-identical; resolve or honestly record the two known-unknowns.
- [ ] Step 4: Run → record the result: the whole model RUNS bit-identical on the Rust engine (the M5 deliverable), plus the whole-model ticks/s pb vs Rust-engine native-OFF (expected <1× — the honest enablement number; the PyO3 tax remains). Write `docs/milestone-5-results.md` honestly (one generation / baseline media / single seed; the whole model now runs; speed is still <1× pending M6+ ports; any residual).
- [ ] Step 5: Commit.

## Notes
- M6+: native process ports (the speedup program) — elongation (polymerize + RNG), rna_degradation/tf_binding, the metabolism hybrid against the decoupled #883 `build_biomass_concentrations_data` form; then multi-gen/seed/condition; the Slurm/manylinux build + cross-host RNG re-verify.
- Stacked on nothing (off clean main @935efa9 = M1+M2+M3+M4). One PR for M5 vs main; report the whole-model-RUNS result to the user + the `optimize` session.
