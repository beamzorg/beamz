#!/bin/bash
set -euo pipefail
while [[ ! -f /workspace/evidence/sweep/sweep.done ]]; do
  [[ $(date -u +%s) -lt $(date -u -d '2026-09-26 19:03:00' +%s) ]] || exit 0
  sleep 5
done
exec 9>/workspace/gpu.lock
flock 9
[[ $(date -u +%s) -lt $(date -u -d '2026-09-26 19:03:00' +%s) ]] || exit 0
cd /workspace/beamz-h100
export PATH=/workspace/venv/bin:/usr/local/cuda/bin:$PATH PYTHONPATH=. LD_LIBRARY_PATH='' XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=.95 NCCL_NVLS_ENABLE=0 BEAMZ_CUDA_CPML_PSI_PRECISION=fp32 NUMPY_MADVISE_HUGEPAGE=0
CUDA_VISIBLE_DEVICES=0 timeout 100 python scripts/validate_modal_scaling.py --backend cuda_streamed --devices 1 --shape 80 120 320 --steps 2560 --resolution-nm 64 --output /workspace/evidence/validation/cuda1-64.json > /workspace/evidence/validation/cuda1-64.log 2>&1
python scripts/compare_modal_scaling.py /workspace/evidence/validation/jax-64.json /workspace/evidence/validation/cuda1-64.json --output /workspace/evidence/validation/comparison-jax-cuda1-64.json > /workspace/evidence/validation/comparison-jax-cuda1-64.log 2>&1 || true
python scripts/compare_modal_scaling.py /workspace/evidence/validation/cuda1-64.json /workspace/evidence/validation/cuda-64.json --output /workspace/evidence/validation/comparison-cuda1-cuda8-64.json > /workspace/evidence/validation/comparison-cuda1-cuda8-64.log 2>&1 || true
touch /workspace/evidence/validation/diagnosis.done
