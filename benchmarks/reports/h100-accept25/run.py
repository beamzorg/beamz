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
    NUMPY_MADVISE_HUGEPAGE="0",
    OMP_NUM_THREADS="8",
    OPENBLAS_NUM_THREADS="8",
)
records = []


def run(name, args, count=2, rev="fixed", timeout=600, extra=None):
    env = dict(
        ENV,
        PYTHONPATH=f"/workspace/{rev}",
        CUDA_VISIBLE_DEVICES=",".join(map(str, range(count))),
    )
    if extra:
        env.update(extra)
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


if not run(
    "one-gpu-parity",
    ["-m", "pytest", "tests/hardware/test_cuda_capacity.py", "-q"],
    1,
    timeout=600,
):
    raise SystemExit("one-GPU parity failed")
if not run(
    "two-gpu-parity",
    ["-m", "pytest", "tests/hardware/test_multi_gpu_capacity.py", "-q"],
    2,
    timeout=600,
):
    raise SystemExit("two-GPU parity failed")
modal = []
for count, axis, backend in [
    (1, "x", "jax"),
    (1, "x", "cuda_streamed"),
    (2, "x", "jax"),
    (2, "x", "cuda_streamed"),
    (2, "z", "jax"),
    (2, "z", "cuda_streamed"),
]:
    name = f"propagated-{count}-{axis}-{backend}"
    output = OUT / f"{name}.json"
    modal.append(output)
    if not run(
        name,
        [
            ROOT / "scripts/validate_modal_scaling.py",
            "--shape",
            64,
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
):
    raise SystemExit("spectra failed")
for repetition in range(3):
    for count, axis in [(1, "x"), (2, "x"), (2, "z")]:
        shape = (256, 256, 256) if count == 1 else (256, 256, 512)
        for backend in ["cuda_streamed", "jax"]:
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
                )
print("PAIRED_COMPLETE", flush=True)
