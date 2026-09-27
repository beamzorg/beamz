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
    XLA_PYTHON_CLIENT_MEM_FRACTION=".04",
    BEAMZ_CUDA_CPML_PSI_PRECISION="fp32",
    BEAMZ_CUDA_AUTOTUNE="off",
    NUMPY_MADVISE_HUGEPAGE="0",
    OMP_NUM_THREADS="8",
    OPENBLAS_NUM_THREADS="8",
)
records = []


def public_gpu_active():
    if (OUT / "public-eight-z-15b-fixed.json").exists():
        return True
    sample = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,used_gpu_memory",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    )
    for line in sample.splitlines():
        columns = line.split(",")
        if (
            len(columns) == 2
            and columns[0].strip() == "9141"
            and columns[1].strip().isdigit()
            and int(columns[1]) > 4096
        ):
            return True
    return False


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
    if public_gpu_active():
        raise SystemExit("Public GPU allocation has begun")
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
            if public_gpu_active() or time.time() - start > timeout:
                process.kill()
                process.wait()
                raise SystemExit("Yielding GPUs to public workload or timeout")
            time.sleep(0.2)
        code = process.returncode
    row = dict(
        memory_fraction=".04",
        context="During CPU-only public preparation; killed before public lower/compile if needed",
        name=name,
        command=command,
        revision=rev,
        devices=count,
        seconds=time.time() - start,
        returncode=code,
    )
    records.append(row)
    (OUT / "concurrent-performance-runs.json").write_text(json.dumps(records, indent=2))
    print("DONE", row, flush=True)
    return code == 0


while not (OUT / "validation-final-complete.json").exists():
    if public_gpu_active():
        raise SystemExit("No isolated GPU window remains")
    time.sleep(2)
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
        pair = []
        try:
            for rev in (
                ["baseline", "fixed"] if repetition % 2 == 0 else ["fixed", "baseline"]
            ):
                name = f"paired-{repetition}-{count}-{axis}-{backend}-{rev}"
                pair.append(name)
                if not run(
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
                ):
                    raise SystemExit("Performance case failed")
            if public_gpu_active():
                raise SystemExit("Pair overlaps public GPU placement")
            fixed = json.loads(
                (
                    OUT / f"paired-{repetition}-{count}-{axis}-{backend}-fixed.json"
                ).read_text()
            )
            if (fixed.get("cuda_schedule") or {}).get("capacity", False):
                raise SystemExit(
                    "Small allocator changed CUDA scheduling; rerun isolated"
                )
        except SystemExit:
            rejected = OUT / "concurrent-interrupted"
            rejected.mkdir(exist_ok=True)
            for name in pair:
                for extension in (".json", ".log"):
                    path = OUT / (name + extension)
                    if path.exists():
                        path.rename(rejected / path.name)
            raise
