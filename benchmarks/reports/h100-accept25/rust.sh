#!/bin/bash
set -euo pipefail
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs -o /workspace/rustup.sh
sh /workspace/rustup.sh -y --profile minimal
source /root/.cargo/env
source /workspace/venv/bin/activate
uv pip install maturin
cd /workspace/fixed
maturin develop --release
cp beamz/design/raster/_native*.so /workspace/baseline/beamz/design/raster/
python /workspace/run.py
