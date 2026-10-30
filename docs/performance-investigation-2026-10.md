# v2ecoli performance — investigation, finding, and decision (Oct 2026)

**Question we set out to answer:** how do we make v2ecoli (the whole-cell E. coli
model) *much* faster — and is a Rust/compiled rewrite of the engine the way?

## Bottom-line finding

**The language/engine was not the lever.** A Rust reimplementation of
process-bigraph (built and validated here as `viva-core`, M1–M5, bit-identical to
pb 1.8.3) does **not** make the whole cell faster, because:

1. **Running Python processes on the Rust engine is net-negative** — every process
   pays a PyO3 boundary crossing every tick, and that boundary is ~72% of the Rust
   tick. Measured: pb ≈ 68 ms/tick vs Rust-engine ≈ 295 ms/tick (**0.23×**).
2. **The per-cell compute is already near-native.** pb's own engine overhead is
   only ~7%; the other ~93% is real process compute that is *already* in C / numba /
   scipy (GLPK for FBA, LSODA for the stiff ODEs, numba for polymerize). There is
   little "Python glue" left to remove.
3. **So porting processes to native Rust can't reach the goal.** Full-native caps
   at ~1.0–1.6× (depending on framework cost + per-process gain); the lone measured
   port (ProteinDegradation, 1.7×) is a mirage — it's 0.12 ms in the real model.
4. **No other language helps.** The boundary is an *architecture* cost (any
   compiled engine calling Python processes per tick pays it), and the heavy
   compute is already compiled. Julia would fit the *numerics* better than Rust for
   a from-scratch native model, but that path has the same ~1.3–1.6× ceiling.

**Exact per-cell 3× is unreachable.** Stacking the exact levers (hot-path cleanup,
compiled stiff-ODE solver, FBA setup refactor) reaches only **~1.4×** (band
1.27–1.56). The one big exact lever we'd have bet on — **FBA warm-start — is
already implemented** (GLPK reuses its basis: 5.5 ms / 6 iterations warm vs
1476 ms / 4729 cold = 270× already captured). Getting ≥3× per cell requires
*approximation* (surrogates, larger timesteps) — a scientific-fidelity tradeoff.

**Where the speed actually is: throughput** (run many cells in parallel). And the
real throughput bottleneck was **a Ray over-reservation config, not memory or
compute** — a node that should run ~12 cells was pinned at 3–4.

## Evidence (measured)

All on an M4 Max laptop (16c/128 GB) against the real `ecoli_baseline`, pb 1.8.3 /
1.10.1 as noted; absolute ms are machine-specific, the shares + ratios are the
signal.

| Measurement | Result |
|---|---|
| pb whole tick | ~68 ms (compute 93%, engine/framework **7%**) |
| Rust engine, Python processes (steady state) | ~295 ms = **0.23×** (72% is PyO3 boundary) |
| Per-tick compute split | translation ~39%, metabolism/FBA ~23%, tail ~38% |
| FBA warm-start (already in GLPK) | 5.5 ms/6 iters warm vs 1476 ms/4729 iters cold = **270×** already realized |
| Exact per-cell levers, stacked | **~1.4×** (1.27–1.56); deleting *all* FBA+ODE only gives 1.7× |
| Full-native per-cell projection | ~1.0–1.6× (1.7× anchor is unreliable; PD is 0.12 ms in-model) |
| Cell cycle wall-clock (birth→division, basal, ~2550 s sim) | **~212 s on pb** (Python); Rust-engine ≈2.3× slower |
| **Throughput bottleneck** | Ray reserved 12 GB + `cores//n_seeds` CPUs per single-threaded seed → **3–4 concurrent** on a 12c/69 GB box, cores idle |
| After the config fix (seed-reservation) | concurrency 7→15 on the laptop; **~2.5–3.4×** cells/hour projected on the mini |

Detailed memos (the raw spikes behind the table) were produced during the
investigation; the numbers above are their consolidated conclusions.

## Decision & outcomes (shipped)

- **Throughput, not per-cell.** v2ecoli **#884** (merged): right-sized the Ray seed
  reservation — `num_cpus=1` single-threaded packing, default per-seed memory
  12→6 GB, and the production multigen path scales the reservation by generation
  count (deep lineages stay OOM-safe). **~2.5–3.4× cells/hour, nearly config-only.**
- **viva-core kept as a clean, reusable general-purpose engine.** The E. coli
  native-port experiment was removed from the core (viva-core **#7**, merged) and
  **archived on the branch `ecoli-native-processes`** (restorable). viva-core now
  ships domain-agnostic, with a general (empty) native-process registry any domain
  can plug into. v2ecoli has zero Rust; viva-core has zero E. coli.

## Open lever (not yet pursued)

**Scale-out = 10–50×+.** The config fix fills one node; the large multiplier is
running across the cluster. v2ecoli has **no Slurm sim dispatch** today (production
is GovCloud AWS Batch Ray). Realizing it needs the HPC target + scale (Slurm vs
more GovCloud nodes; node count, cores/RAM each) — that number *is* the multiplier.
The ~1.4× exact per-cell cleanup (if ever done) multiplies on top of throughput.
