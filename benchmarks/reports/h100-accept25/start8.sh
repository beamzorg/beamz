#!/bin/bash
set -euo pipefail
bash /workspace/setup.sh
cp /workspace/_native.abi3.so /workspace/fixed/beamz/design/raster/
cp /workspace/_native.abi3.so /workspace/baseline/beamz/design/raster/
nvidia-smi --query-gpu=timestamp,index,temperature.gpu,clocks.sm,power.draw,utilization.gpu,memory.used --format=csv -l 2 > /workspace/evidence/telemetry.csv &
python /workspace/health8.py > /workspace/evidence/health.log
/workspace/venv/bin/python /workspace/eight.py
