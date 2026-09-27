import datetime
import json
import os
import pathlib
import subprocess
import time

OUT = pathlib.Path("/workspace/evidence")
ROOT = pathlib.Path("/workspace/fixed")
PY = "/workspace/venv/bin/python"
ENV = dict(
    os.environ,
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
cases = [
    ("one-cuda-capacity", "fixed", "cuda_streamed", 1, "none", (1250, 1250, 1200)),
    ("one-jax-capacity", "fixed", "jax", 1, "none", (1000, 1000, 1000)),
    ("two-z-cuda-moderate", "fixed", "cuda_streamed", 2, "z", (1250, 1250, 1200)),
    ("two-x-cuda-moderate", "fixed", "cuda_streamed", 2, "x", (1250, 1250, 1200)),
    ("two-z-jax-capacity", "fixed", "jax", 2, "z", (2000, 1000, 1000)),
    (
        "two-z-cuda-target-baseline",
        "baseline",
        "cuda_streamed",
        2,
        "z",
        (2500, 1250, 1200),
    ),
    ("two-z-cuda-target-fixed", "fixed", "cuda_streamed", 2, "z", (2500, 1250, 1200)),
    ("two-x-cuda-target-fixed", "fixed", "cuda_streamed", 2, "x", (1250, 1250, 2400)),
]
for name, rev, backend, count, axis, shape in cases:
    env = dict(
        ENV,
        PYTHONPATH=f"/workspace/{rev}",
        CUDA_VISIBLE_DEVICES=",".join(map(str, range(count))),
    )
    command = [
        PY,
        str(ROOT / "scripts/benchmark_compile_capacity.py"),
        "--shape",
        *map(str, shape),
        "--devices",
        str(count),
        "--axis",
        axis,
        "--backend",
        backend,
        "--steps",
        "32",
        "--samples",
        "3",
        "--output",
        str(OUT / f"{name}.json"),
    ]
    print(
        "START",
        name,
        datetime.datetime.now(datetime.timezone.utc).isoformat(),
        flush=True,
    )
    start = time.time()
    with (OUT / f"{name}.log").open("w") as log:
        try:
            code = subprocess.run(
                command,
                cwd=f"/workspace/{rev}",
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                timeout=480,
            ).returncode
        except subprocess.TimeoutExpired:
            code = 124
    row = dict(
        name=name,
        revision=rev,
        command=command,
        returncode=code,
        seconds=time.time() - start,
    )
    records.append(row)
    (OUT / "capacity-runs.json").write_text(json.dumps(records, indent=2))
    print("DONE", row, flush=True)
