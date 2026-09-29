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


def run(name, args, count=8, rev="fixed", timeout=300):
    env = dict(
        ENV,
        PYTHONPATH=f"/workspace/{rev}",
        CUDA_VISIBLE_DEVICES=",".join(map(str, range(count))),
    )

    if (
        time.time() + timeout
        > datetime.datetime(
            2026, 9, 27, 10, 44, 30, tzinfo=datetime.timezone.utc
        ).timestamp()
    ):
        raise SystemExit("Preserving retrieval/deletion budget")
    command = [PY, *map(str, args)]
    start = time.time()
    print(
        "START",
        name,
        datetime.datetime.now(datetime.timezone.utc).isoformat(),
        flush=True,
    )
    with (OUT / f"{name}.log").open("w") as log:
        try:
            code = subprocess.run(
                command,
                cwd=f"/workspace/{rev}",
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                timeout=timeout,
            ).returncode
        except subprocess.TimeoutExpired:
            code = 124
    row = dict(
        name=name,
        command=command,
        revision=rev,
        devices=count,
        seconds=time.time() - start,
        returncode=code,
    )
    records.append(row)
    (OUT / "final-performance-runs.json").write_text(json.dumps(records, indent=2))
    print("DONE", row, flush=True)
    return code == 0


for backend in ["cuda_streamed", "jax"]:
    for rev in ["baseline", "fixed"]:
        name = f"paired-0-1-x-{backend}-{rev}"
        run(
            name,
            [
                ROOT / "scripts/benchmark_modal_stepping.py",
                "--shape",
                256,
                256,
                256,
                "--devices",
                1,
                "--backend",
                backend,
                "--timesteps",
                128,
                "--samples",
                5,
                "--frequencies",
                101,
                "--host-setup",
                "--donate-state",
                "--output",
                OUT / f"{name}.json",
            ],
            1,
            rev,
            timeout=45,
        )
