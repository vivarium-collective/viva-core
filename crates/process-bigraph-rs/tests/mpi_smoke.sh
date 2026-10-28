#!/usr/bin/env bash
# Builds mpi_smoke and runs it under 2 MPI ranks (Homebrew Open MPI 5).
set -euo pipefail
cd "$(dirname "$0")/../../.."
cargo build -p process-bigraph-rs --bin mpi_smoke --features mpi
out=$(mpirun -n 2 target/debug/mpi_smoke)
echo "$out"
grep -q MPI_SCATTER_GATHER_OK <<<"$out"
