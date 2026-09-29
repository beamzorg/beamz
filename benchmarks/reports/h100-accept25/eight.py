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
    (OUT / "runs.json").write_text(json.dumps(records, indent=2))
    print("DONE", row, flush=True)
    return code == 0


for rev in ["baseline", "fixed"]:
    name = f"prepared-eight-z-15b-{rev}"
    okay = run(
        name,
        [
            ROOT / "scripts/benchmark_compile_capacity.py",
            "--shape",
            10000,
            1250,
            1200,
            "--devices",
            8,
            "--axis",
            "z",
            "--backend",
            "cuda_streamed",
            "--steps",
            32,
            "--samples",
            3,
            "--output",
            OUT / f"{name}.json",
        ],
        rev=rev,
        timeout=300,
    )
    if rev == "fixed" and not okay:
        raise SystemExit("fixed 15B prepared failed")
name = "public-eight-z-15b-fixed"
run(
    name,
    [
        ROOT / "scripts/benchmark_compile_capacity.py",
        "--workload",
        "modal",
        "--shape",
        10000,
        1250,
        1200,
        "--devices",
        8,
        "--axis",
        "z",
        "--backend",
        "cuda_streamed",
        "--frequencies",
        101,
        "--steps",
        32,
        "--samples",
        3,
        "--output",
        OUT / f"{name}.json",
    ],
    timeout=1950,
)
print("EIGHT_TARGET_COMPLETE", flush=True)
