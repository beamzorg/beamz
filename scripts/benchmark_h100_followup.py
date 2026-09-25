#!/usr/bin/env python3
"""Bounded large-grid, partition-axis and physical-crossing follow-up trials."""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).resolve().parents[1]
    worker = root / "scripts/benchmark_h100_backends.py"
    jobs = []
    for count in (1, 2, 4):
        for backend in (
            "jax",
            "cuda_streamed",
        ):
            jobs.append(
                (
                    f"large-{backend}-{count}gpu",
                    count,
                    [
                        str(worker),
                        "--worker",
                        "--workload",
                        "realistic_3d",
                        "--shape",
                        "512",
                        "512",
                        "512",
                        "--backend",
                        backend,
                        "--devices",
                        str(count),
                        "--timesteps",
                        "256",
                        "--samples",
                        "5",
                    ],
                )
            )
    for axis in ("y", "z"):
        jobs.append(
            (
                f"axis-{axis}-cuda_streamed-4gpu",
                4,
                [
                    str(worker),
                    "--worker",
                    "--workload",
                    "realistic_3d",
                    "--shape",
                    "128",
                    "256",
                    "384",
                    "--backend",
                    "cuda_streamed",
                    "--devices",
                    "4",
                    "--shard-axis",
                    axis,
                    "--timesteps",
                    "256",
                    "--samples",
                    "5",
                ],
            )
        )
    for resolution, mode in ((36, "throughput"), (25, "throughput"), (36, "full")):
        for backend in ("cuda_streamed", "jax"):
            jobs.append(
                (
                    f"crossing-{resolution}nm-{mode}-{backend}",
                    1,
                    [
                        str(root / "scripts/benchmark_cosine_h100.py"),
                        "--child",
                        "--resolution",
                        str(resolution),
                        "--mode",
                        mode,
                        "--backend",
                        backend,
                        "--samples",
                        "5",
                    ],
                )
            )
    env = {k: v for k, v in os.environ.items() if not k.startswith("BEAMZ_CUDA_")}
    env.update(
        PYTHONPATH=str(root),
        LD_LIBRARY_PATH="",
        XLA_PYTHON_CLIENT_PREALLOCATE="false",
        XLA_PYTHON_CLIENT_MEM_FRACTION=".80",
        BEAMZ_CUDA_CPML_PSI_PRECISION="fp32",
        NUMPY_MADVISE_HUGEPAGE="0",
        OPENBLAS_NUM_THREADS="4",
        OMP_NUM_THREADS="8",
        BEAMZ_DISABLE_JAX_PERSISTENT_CACHE="1",
        BEAMZ_RASTER_CACHE="0",
        MPLBACKEND="Agg",
    )
    records = []
    for name, count, command in jobs:
        output = args.output.resolve() / f"{name}.json"
        command = [sys.executable, *command, "--output", str(output)]
        print(name, flush=True)
        start = time.monotonic()
        with (args.output / f"{name}.log").open("w") as log:
            try:
                result = subprocess.run(
                    command,
                    cwd=root,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    env=dict(
                        env, CUDA_VISIBLE_DEVICES=",".join(map(str, range(count)))
                    ),
                    timeout=600,
                )
                status = (
                    "ok" if result.returncode == 0 and output.exists() else "failed"
                )
            except subprocess.TimeoutExpired:
                status = "timeout"
        records.append(
            {
                "name": name,
                "status": status,
                "wall_s": time.monotonic() - start,
                "command": command,
            }
        )
        (args.output / "summary.json").write_text(json.dumps(records, indent=2))


if __name__ == "__main__":
    main()
