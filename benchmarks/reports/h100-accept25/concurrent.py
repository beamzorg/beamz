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
    (OUT / "concurrent-runs.json").write_text(json.dumps(records, indent=2))
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

(OUT / "concurrent-gates-ok.json").write_text(
    json.dumps(
        dict(
            completed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            memory_fraction=".03",
        )
    )
)
