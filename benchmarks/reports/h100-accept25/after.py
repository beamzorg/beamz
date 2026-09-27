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
    (OUT / "after-runs.json").write_text(json.dumps(records, indent=2))
    print("DONE", row, flush=True)
    return code == 0


if not run(
    "eight-gpu-parity",
    ["-m", "pytest", "tests/hardware/test_multi_gpu_capacity.py", "-q"],
    timeout=240,
):
    raise SystemExit("eight GPU parity failed")
modal = []
for count, axis, backend in [
    (1, "z", "jax"),
    (8, "z", "cuda_streamed"),
    (8, "z", "jax"),
    (8, "x", "cuda_streamed"),
]:
    name = f"propagated-{count}-{axis}-{backend}"
    output = OUT / f"{name}.json"
    modal.append(output)
    if not run(
        name,
        [
            ROOT / "scripts/validate_modal_scaling.py",
            "--shape",
            128,
            96,
            256,
            "--steps",
            2048,
            "--frequencies",
            101,
            "--devices",
            count,
            "--shard-axis",
            axis,
            "--backend",
            backend,
            "--output",
            output,
        ],
        count,
        timeout=60,
    ):
        raise SystemExit("propagation failed")
if not run(
    "propagated-comparison",
    [
        ROOT / "scripts/compare_modal_scaling.py",
        *modal,
        "--output",
        OUT / "propagated-comparison.json",
    ],
    timeout=30,
):
    raise SystemExit("spectra failed")
for count, axis, backend in [
    (1, "x", "cuda_streamed"),
    (1, "x", "jax"),
    (2, "x", "cuda_streamed"),
    (2, "z", "cuda_streamed"),
    (2, "x", "jax"),
    (2, "z", "jax"),
]:
    shape = (256, 256, 256) if count == 1 else (256, 256, 512)
    for repetition in range(3):
        for rev in (
            ["baseline", "fixed"] if repetition % 2 == 0 else ["fixed", "baseline"]
        ):
            name = f"paired-{repetition}-{count}-{axis}-{backend}-{rev}"
            run(
                name,
                [
                    ROOT / "scripts/benchmark_modal_stepping.py",
                    "--shape",
                    *shape,
                    "--devices",
                    count,
                    "--shard-axis",
                    axis,
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
                count,
                rev,
                timeout=90,
            )
