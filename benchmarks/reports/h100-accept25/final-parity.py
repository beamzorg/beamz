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
    XLA_PYTHON_CLIENT_MEM_FRACTION=".03",
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
        process = subprocess.Popen(
            command,
            cwd=f"/workspace/{rev}",
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
        while process.poll() is None:
            if (
                OUT / "public-eight-z-15b-fixed.json"
            ).exists() or time.time() - start > timeout:
                process.kill()
                process.wait()
                raise SystemExit("Yielding GPUs to public workload or timeout")
            time.sleep(0.05)
        code = process.returncode
    row = dict(
        memory_fraction=".03",
        context="During CPU-only public preparation; killed before public lower/compile if needed",
        name=name,
        command=command,
        revision=rev,
        devices=count,
        seconds=time.time() - start,
        returncode=code,
    )
    records.append(row)
    (OUT / "final-parity-runs.json").write_text(json.dumps(records, indent=2))
    print("DONE", row, flush=True)
    return code == 0


while not (OUT / "diagnostics-complete.json").exists():
    if (OUT / "public-eight-z-15b-fixed.json").exists():
        raise SystemExit("Public workload ready")
    time.sleep(1)
if not run(
    "eight-gpu-parity-established",
    ["-m", "pytest", "tests/hardware/test_multi_gpu_capacity.py", "-q"],
    timeout=240,
):
    raise SystemExit("Established parity criteria failed")
(OUT / "validation-final-complete.json").write_text("{}")
