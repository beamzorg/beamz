#!/usr/bin/env python3
"""Run isolated eight-H100 capacity probes with a wall-clock deadline.

The worker retains its input state (the established benchmark protocol). NVML
sampling includes setup and compilation; allocator live/peak memory is recorded
separately by the worker. This script never provisions or deletes infrastructure.
"""

import argparse
import datetime as dt
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--deadline", required=True, help="UTC ISO 8601 cutoff")
    parser.add_argument("--case-timeout", type=int, default=1200)
    args = parser.parse_args()
    deadline = dt.datetime.fromisoformat(args.deadline).timestamp()
    args.output.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env.update(
        PYTHONPATH=str(Path.cwd()),
        LD_LIBRARY_PATH="",
        XLA_PYTHON_CLIENT_PREALLOCATE="false",
        XLA_PYTHON_CLIENT_MEM_FRACTION=".95",
        NCCL_NVLS_ENABLE="0",
        BEAMZ_CUDA_CPML_PSI_PRECISION="fp32",
        NUMPY_MADVISE_HUGEPAGE="0",
    )
    cases = [
        (f"balanced-{n}", (n, n, 8 * n), 80.0) for n in (512, 640, 704, 768, 832, 896)
    ]
    cases += [(f"planar-{n}", (n, 8 * n, 64 * n), 80.0) for n in (128, 160, 192, 224)]
    cases += [(f"refined-{n}", (n, n, 8 * n), 80.0 * 512 / n) for n in (640, 768, 896)]
    rows = []
    failed_groups = set()
    for name, shape, resolution in cases:
        group = name.split("-")[0]
        if group in failed_groups:
            continue
        remaining = deadline - time.time()
        if remaining < 360:
            break
        output = args.output / f"{name}.json"
        command = [
            sys.executable,
            "scripts/benchmark_modal_stepping.py",
            "--shape",
            *map(str, shape),
            "--devices",
            "8",
            "--backend",
            "cuda_streamed",
            "--host-setup",
            "--frequencies",
            "101",
            "--timesteps",
            "256",
            "--samples",
            "5",
            "--resolution-nm",
            str(resolution),
            "--output",
            str(output),
        ]
        start = time.time()
        row = dict(
            name=name,
            shape=shape,
            resolution_nm=resolution,
            command=command,
            started_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
        )
        print(json.dumps(dict(stage="start", **row)), flush=True)
        with (
            (args.output / f"{name}.nvml.csv").open("w") as telemetry,
            (args.output / f"{name}.log").open("w") as log,
        ):
            monitor = subprocess.Popen(
                [
                    "nvidia-smi",
                    "--query-gpu=timestamp,index,memory.total,memory.used,utilization.gpu,utilization.memory,power.draw,clocks.sm,clocks.mem",
                    "--format=csv,nounits",
                    "-lms",
                    "500",
                ],
                stdout=telemetry,
                stderr=subprocess.STDOUT,
            )
            process = subprocess.Popen(
                command,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            try:
                code = process.wait(timeout=min(args.case_timeout, remaining - 30))
                row["status"] = "ok" if code == 0 and output.exists() else "failed"
                row["returncode"] = code
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
                row["status"] = "timeout"
            finally:
                monitor.terminate()
                monitor.wait(timeout=15)
        row["wall_s"] = time.time() - start
        if row["status"] != "ok":
            failed_groups.add(group)
        else:
            data = json.loads(output.read_text())
            row["gcups"] = data["kernel_gcups"]
            row["max_peak_live_bytes"] = max(
                d["stats"]["peak_bytes_in_use"] for d in data["device_memory"]
            )
        rows.append(row)
        (args.output / "summary.json").write_text(json.dumps(rows, indent=2) + "\n")
        print(json.dumps(dict(stage="finished", **row)), flush=True)
    manifest = dict(
        deadline=args.deadline,
        environment=env,
        cases=cases,
        runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    )
    # Keep only relevant non-secret environment settings in committed evidence.
    manifest["environment"] = {
        k: v
        for k, v in env.items()
        if k.startswith(("XLA_", "BEAMZ_", "NCCL_", "NUMPY_"))
    }
    (args.output / "protocol.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (args.output / "sweep.done").touch()


if __name__ == "__main__":
    main()
