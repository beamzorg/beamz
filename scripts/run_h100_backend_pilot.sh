#!/usr/bin/env bash
# Execute the correctness gate and bounded pilot on an already prepared H100 pod.
set -euo pipefail
cd "$(dirname "$0")/.."
py="${BEAMZ_BENCHMARK_PYTHON:-/workspace/.venvs/beamz-h100/bin/python}"
out="${1:?Provide a fresh output directory}"
mkdir -p "$out"
out="$(cd "$out" && pwd)"
export PYTHONPATH="$PWD" LD_LIBRARY_PATH=""
export XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=.80
export BEAMZ_CUDA_CPML_PSI_PRECISION=fp32 NUMPY_MADVISE_HUGEPAGE=0
export OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=8
export BEAMZ_DISABLE_JAX_PERSISTENT_CACHE=1 BEAMZ_RASTER_CACHE=0
unset CUDA_VISIBLE_DEVICES
nvidia-smi --query-gpu=timestamp,index,name,utilization.gpu,memory.used,clocks.sm,power.draw --format=csv -l 1 > "$out/telemetry.csv" &
telemetry_pid=$!
trap 'kill "$telemetry_pid" 2>/dev/null || true' EXIT
timeout 900 "$py" -m pytest -q tests/hardware/test_cuda_backends.py \
    tests/hardware/test_h100_backend_pilot.py \
    -k 'streamed_cuda_matches_jax_complete_state or sharded_streamed or modal_backend_complete_state' \
    --junitxml="$out/parity.xml" > "$out/parity.log" 2>&1
timeout 7200 "$py" scripts/benchmark_h100_backends.py \
    --counts 1 2 4 --samples 5 --timesteps 256 --timeout 600 \
    --output "$out/sweep" > "$out/sweep.log" 2>&1
