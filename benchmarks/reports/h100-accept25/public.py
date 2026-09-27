import json
import os
import pathlib
import subprocess
import time

OUT = pathlib.Path("/workspace/evidence")
ROOT = pathlib.Path("/workspace/fixed")
PY = "/workspace/venv/bin/python"
env = dict(
    os.environ,
    PYTHONPATH=str(ROOT),
    CUDA_VISIBLE_DEVICES="0,1",
    LD_LIBRARY_PATH="",
    XLA_PYTHON_CLIENT_PREALLOCATE="true",
    XLA_PYTHON_CLIENT_MEM_FRACTION=".90",
    BEAMZ_CUDA_CPML_PSI_PRECISION="fp32",
    BEAMZ_CUDA_AUTOTUNE="off",
    NUMPY_MADVISE_HUGEPAGE="0",
    OMP_NUM_THREADS="8",
    OPENBLAS_NUM_THREADS="8",
)
records = []
for name, shape, backend, seconds in [
    ("public-two-z-cuda-268m", (512, 512, 1024), "cuda_streamed", 600),
    ("public-two-z-jax-268m", (512, 512, 1024), "jax", 600),
    ("public-two-z-cuda-3750m", (2500, 1250, 1200), "cuda_streamed", 1000),
]:
    command = [
        PY,
        str(ROOT / "scripts/benchmark_compile_capacity.py"),
        "--workload",
        "modal",
        "--shape",
        *map(str, shape),
        "--devices",
        "2",
        "--axis",
        "z",
        "--backend",
        backend,
        "--frequencies",
        "101",
        "--steps",
        "32",
        "--samples",
        "3",
        "--output",
        str(OUT / f"{name}.json"),
    ]
    print("START", name, flush=True)
    start = time.time()
    with (OUT / f"{name}.log").open("w") as log:
        try:
            code = subprocess.run(
                command,
                cwd=ROOT,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                timeout=seconds,
            ).returncode
        except subprocess.TimeoutExpired:
            code = 124
    row = dict(name=name, command=command, seconds=time.time() - start, returncode=code)
    records.append(row)
    (OUT / "public-runs.json").write_text(json.dumps(records, indent=2))
    print("DONE", row, flush=True)
