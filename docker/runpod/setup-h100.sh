#!/usr/bin/env bash
# Run inside a CUDA 12.8 development Pod with Python 3.11 or 3.12.
set -euo pipefail

if [[ "${1:-}" == "--help" ]]; then
    echo "Usage: bash docker/runpod/setup-h100.sh"
    echo "Run from a CUDA 12.8 devel/Jupyter Pod. Creates /workspace/.venvs/beamz-h100."
    echo "Optional: BEAMZ_RUNPOD_VENV, BEAMZ_BUILD_JOBS (default 2)."
    exit 0
fi

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
venv_dir="${BEAMZ_RUNPOD_VENV:-/workspace/.venvs/beamz-h100}"
build_jobs="${BEAMZ_BUILD_JOBS:-2}"
command -v nvcc >/dev/null || {
    echo "nvcc is missing. Choose a CUDA 12.8 DEVEL template, not a runtime image." >&2
    exit 1
}
nvcc --version
nvcc --version | grep -q 'release 12\.8' || {
    echo "This setup targets CUDA 12.8; use a matching development template." >&2
    exit 1
}
nvidia-smi
python3 -c 'import sys; assert (3, 11) <= sys.version_info[:2] <= (3, 12), "Use Python 3.11 or 3.12"'

if [[ "$(id -u)" == 0 ]]; then
    apt-get update
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        build-essential curl ca-certificates git pkg-config python3-venv
fi
for tool in g++ curl git pkg-config; do
    command -v "$tool" >/dev/null || { echo "Missing prerequisite: $tool" >&2; exit 1; }
done
export PATH="${HOME}/.cargo/bin:${PATH}"
if ! command -v cargo >/dev/null; then
    installer="$(mktemp)"
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs -o "$installer"
    sh "$installer" -y --profile minimal --default-toolchain stable
    rm "$installer"
fi
rustc --version

mkdir -p "$(dirname "$venv_dir")"
python3 -m venv "$venv_dir"
py="${venv_dir}/bin/python"
"$py" -m pip install --upgrade pip 'uv==0.9.7'
cd "$repo_dir"
export UV_PROJECT_ENVIRONMENT="$venv_dir"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=.80
# Use JAX's pip CUDA libraries rather than inherited PyTorch/cuDNN search paths.
unset LD_LIBRARY_PATH
mkdir -p "$repo_dir/.cache/runpod"
"${venv_dir}/bin/uv" export --frozen --no-emit-project --no-hashes \
    --extra dev --extra test --extra gds \
    --output-file "$repo_dir/.cache/runpod/constraints.txt" >/dev/null
"${venv_dir}/bin/uv" pip install --python "$py" \
    --requirement "$repo_dir/.cache/runpod/constraints.txt"
"${venv_dir}/bin/uv" pip install --python "$py" --no-deps --editable "$repo_dir"
"${venv_dir}/bin/uv" pip install --python "$py" \
    --constraint "$repo_dir/.cache/runpod/constraints.txt" \
    'jax[cuda12]==0.9.0' jupyterlab ipykernel \
    'scikit-build-core>=0.10' 'nanobind>=2.4' 'cmake>=3.24' ninja
# Use fresh objects: transferred source snapshots can predate cached binaries.
wheel_dir="$(mktemp -d "$repo_dir/.cache/runpod/wheels.XXXXXX")"
export PATH="${venv_dir}/bin:${PATH}"
CMAKE_BUILD_PARALLEL_LEVEL="$build_jobs" "$py" -m pip wheel "$repo_dir/cuda" \
    --no-deps --no-build-isolation --wheel-dir "$wheel_dir" \
    --config-settings=build-dir="$wheel_dir/build" \
    --config-settings=cmake.define.BEAMZ_CUDA_ARCHITECTURES=90 \
    --config-settings=cmake.define.BEAMZ_CUDA_FAST_MATH=OFF \
    --config-settings=cmake.build-type=Release
"$py" -m pip install --no-deps --force-reinstall "$wheel_dir"/*.whl
# Editable BeamZ resolves to the source package. Place the freshly built module
# there atomically; never overwrite the contents of a loaded shared library.
"$py" - "$repo_dir" "$wheel_dir" <<'PY'
import os
import sys
import tempfile
import zipfile
from pathlib import Path

repo, wheels = map(Path, sys.argv[1:])
wheel, = wheels.glob('*.whl')
with zipfile.ZipFile(wheel) as archive:
    module, = [n for n in archive.namelist() if n.startswith('beamz/_cuda.') and n.endswith('.so')]
    target = repo / module
    with tempfile.NamedTemporaryFile(dir=target.parent, delete=False) as handle:
        handle.write(archive.read(module))
        temporary = handle.name
    os.replace(temporary, target)
PY
"$py" -m ipykernel install --user --name beamz-h100 --display-name 'BeamZ H100 (CUDA)'
"$py" - "$repo_dir" <<'PY'
import json
import sys
from pathlib import Path
from jupyter_client.kernelspec import KernelSpecManager

path = Path(KernelSpecManager().get_kernel_spec('beamz-h100').resource_dir) / 'kernel.json'
spec = json.loads(path.read_text())
spec['env'] = dict(PYTHONPATH=sys.argv[1], XLA_PYTHON_CLIENT_PREALLOCATE='false',
                   XLA_PYTHON_CLIENT_MEM_FRACTION='.80', LD_LIBRARY_PATH='')
path.write_text(json.dumps(spec, indent=2) + '\n')
PY
"$py" - <<'PY'
import jax
import jax.numpy as jnp
import beamz._cuda as extension
from beamz.simulation.backend import cuda_backend_status

devices = jax.devices()
assert len(devices) == 1 and 'H100' in devices[0].device_kind, devices
jax.jit(lambda x: x + 1)(jnp.ones(16)).block_until_ready()
status = cuda_backend_status()
print(status.as_dict())
assert status.available, status.reason
print('Native extension:', extension.__file__)
PY
"$py" -m pip freeze > "$repo_dir/.cache/runpod/installed-packages.txt"
echo "Ready. Refresh JupyterLab and select the BeamZ H100 (CUDA) kernel."
echo "Restart existing BeamZ kernels after rebuilding. See docker/runpod/H100.md."
