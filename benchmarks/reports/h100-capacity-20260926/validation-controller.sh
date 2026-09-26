#!/bin/bash
set -euo pipefail
cd /workspace/beamz-h100
export PATH=/workspace/venv/bin:/usr/local/cuda/bin:$PATH PYTHONPATH=. LD_LIBRARY_PATH='' XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=.95 NCCL_NVLS_ENABLE=0 BEAMZ_CUDA_CPML_PSI_PRECISION=fp32 NUMPY_MADVISE_HUGEPAGE=0
mkdir -p /workspace/evidence/validation
exec 9>/workspace/gpu.lock
flock 9
# Do not begin expensive work near the independent deletion deadline.
for dx in 80 64; do
 [[ $(date -u +%s) -lt $(date -u -d '2026-09-26 19:05:00' +%s) ]] || exit 0
 if [[ $dx = 80 ]]; then shape='64 96 256'; steps=2048; else shape='80 120 320'; steps=2560; fi
 CUDA_VISIBLE_DEVICES=0 timeout 100 python scripts/validate_modal_scaling.py --backend jax --devices 1 --shape $shape --steps $steps --resolution-nm $dx --output /workspace/evidence/validation/jax-$dx.json > /workspace/evidence/validation/jax-$dx.log 2>&1
 timeout 100 python scripts/validate_modal_scaling.py --backend cuda_streamed --devices 8 --shape $shape --steps $steps --resolution-nm $dx --output /workspace/evidence/validation/cuda-$dx.json > /workspace/evidence/validation/cuda-$dx.log 2>&1
 python scripts/compare_modal_scaling.py /workspace/evidence/validation/jax-$dx.json /workspace/evidence/validation/cuda-$dx.json --output /workspace/evidence/validation/comparison-$dx.json > /workspace/evidence/validation/comparison-$dx.log 2>&1
done
touch /workspace/evidence/validation/done
