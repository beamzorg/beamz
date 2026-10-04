"""Run sequential local RTX3090 cube trials with a host-memory cgroup limit.

Requires systemd's user service manager. Uses the current Python environment and
the selected checkout's source/native extension. No remote resources are created.
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkout", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sides", nargs="+", type=int, required=True)
    parser.add_argument(
        "--allocator", choices=("bfc", "cuda_async"), default="cuda_async"
    )
    parser.add_argument("--host-memory-gib", type=int, default=14)
    parser.add_argument("--memory-fraction", type=float, default=0.95)
    args = parser.parse_args()
    if not 0 < args.memory_fraction <= 1:
        parser.error("memory-fraction must be in (0, 1]")
    root, output = args.checkout.resolve(), args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    (output / "telemetry-timezone.json").write_text(
        json.dumps({"utc_offset_seconds": time.localtime().tm_gmtoff}) + "\n"
    )
    env = dict(os.environ)
    env.update(
        PYTHONPATH=str(root),
        JAX_PLATFORMS="cuda",
        NUMPY_MADVISE_HUGEPAGE="0",
        XLA_PYTHON_CLIENT_PREALLOCATE="false",
        XLA_PYTHON_CLIENT_MEM_FRACTION=str(args.memory_fraction),
        XLA_PYTHON_CLIENT_ALLOCATOR=args.allocator,
        OPENBLAS_NUM_THREADS="4",
        OMP_NUM_THREADS="4",
        BEAMZ_DISABLE_JAX_PERSISTENT_CACHE="1",
        BEAMZ_RASTER_CACHE="0",
    )
    for side in args.sides:
        result = output / f"cube-{side}.json"
        if result.exists():
            raise FileExistsError(result)
        command = [
            "systemd-run",
            "--user",
            "--quiet",
            "--pipe",
            "--wait",
            "--collect",
            "-p",
            f"MemoryMax={args.host_memory_gib}G",
            "-p",
            "MemorySwapMax=0",
            "-p",
            "RuntimeMaxSec=900",
            f"--working-directory={root}",
        ]
        # Service processes do not inherit the calling shell's environment.
        for key, value in env.items():
            if key in (
                "PATH",
                "LD_LIBRARY_PATH",
                "PYTHONPATH",
                "JAX_PLATFORMS",
                "NUMPY_MADVISE_HUGEPAGE",
                "OPENBLAS_NUM_THREADS",
                "OMP_NUM_THREADS",
            ) or key.startswith(("XLA_PYTHON_CLIENT", "BEAMZ_")):
                command.append(f"--setenv={key}={value}")
        command.extend(
            [
                sys.executable,
                "scripts/profile_cuda_preparation.py",
                "--expected-device",
                "RTX 3090",
                "--samples",
                "3",
                "--side",
                str(side),
                "--output",
                str(result),
            ]
        )
        if side == 64:
            command.extend(["--state-output", str(output / "state.npz")])
        print("START", output.name, side, flush=True)
        with (output / f"cube-{side}-telemetry.csv").open("w") as telemetry:
            monitor = subprocess.Popen(
                [
                    "nvidia-smi",
                    "--query-gpu=timestamp,memory.used,memory.total,utilization.gpu,utilization.memory,clocks.sm,clocks.mem,power.draw,temperature.gpu",
                    "--format=csv",
                    "-lms",
                    "200",
                ],
                stdout=telemetry,
            )
            try:
                with result.with_suffix(".log").open("w") as log:
                    code = subprocess.run(
                        command, stdout=log, stderr=subprocess.STDOUT
                    ).returncode
            finally:
                monitor.terminate()
                monitor.wait()
        (output / f"cube-{side}-exit.json").write_text(
            json.dumps(dict(exit_code=code)) + "\n"
        )
        print("END", output.name, side, code, flush=True)
        if code:
            break


if __name__ == "__main__":
    main()
