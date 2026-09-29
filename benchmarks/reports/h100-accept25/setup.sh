#!/bin/bash
set -euo pipefail
mkdir -p /workspace/evidence /workspace/fixed /workspace/baseline
for rev in fixed baseline; do
 tar xzf /workspace/$rev.tar.gz -C /workspace/$rev
 git -C /workspace/$rev init -q
 git -C /workspace/$rev config user.email benchmark@localhost
 git -C /workspace/$rev config user.name Benchmark
 git -C /workspace/$rev add .
 git -C /workspace/$rev commit -qm "Source snapshot $rev; see evidence/manifest.json for original SHA"
done
python3 -m venv /workspace/venv
source /workspace/venv/bin/activate
python -m pip install -q uv
uv pip install 'jax[cuda12]==0.9.0' numpy scipy shapely optax matplotlib scikit-image xarray pytest pytest-cov hypothesis
uv pip install --no-deps /workspace/beamz_cuda_component-0.21.0-cp312-cp312-linux_x86_64.whl
python - <<'PY'
import site,shutil
from pathlib import Path
for p in Path(site.getsitepackages()[0]).glob('beamz/_cuda*.so'):
 for rev in ('fixed','baseline'): shutil.copy2(p,Path('/workspace')/rev/'beamz'/p.name)
PY
export PYTHONPATH=/workspace/fixed LD_LIBRARY_PATH=''
python - <<'PY'
import jax,beamz._cuda,hashlib
from pathlib import Path
print(jax.__version__,jax.devices())
print('NATIVE_SHA256',hashlib.sha256(Path(beamz._cuda.__file__).read_bytes()).hexdigest())
PY
nvidia-smi -q > /workspace/evidence/nvidia-smi.txt
nvidia-smi topo -m > /workspace/evidence/topology.txt
python -m pip freeze > /workspace/evidence/packages.txt
