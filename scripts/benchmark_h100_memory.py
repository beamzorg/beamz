#!/usr/bin/env python3
"""Deadline-bound donating capacity probes; provisioning/teardown is external.

Run after propagated modal parity passes. Keep the original retained-state
throughput harness for comparisons; this run measures evolving-state capacity.
"""

import argparse
import datetime as dt
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
    parser.add_argument("--deadline", required=True, help="Timezone-aware ISO time")
    parser.add_argument(
        "--target-only",
        action="store_true",
        help="Attempt only 15B after an earlier setup/control run",
    )
    args = parser.parse_args()
    deadline = dt.datetime.fromisoformat(args.deadline)
    if deadline.tzinfo is None:
        parser.error("deadline must include timezone")
    cutoff = deadline.timestamp()
    args.output.mkdir(parents=True, exist_ok=True)
    environment = dict(os.environ)
    environment.update(
        PYTHONPATH=str(Path.cwd()),
        LD_LIBRARY_PATH="",
        XLA_PYTHON_CLIENT_PREALLOCATE="false",
        XLA_PYTHON_CLIENT_MEM_FRACTION=".95",
        NCCL_NVLS_ENABLE="0",
        BEAMZ_CUDA_CPML_PSI_PRECISION="fp32",
        NUMPY_MADVISE_HUGEPAGE="0",
    )
    summary_path = args.output / "summary.json"
    rows = (
        json.loads(summary_path.read_text())
        if args.target_only and summary_path.exists()
        else []
    )
    previous = None
    # Reproduce the old x-partition setup failure first. For capacity, z
    # partitions use contiguous native storage without per-phase x rotations.
    # The long dimension moves to z; this is a different physical domain.
    cases = [
        ("5.75B", (896, 896, 7168), "x"),
        ("10B", (10000, 1000, 1000), "z"),
        ("15B", (10000, 1250, 1200), "z"),
    ]
    if args.target_only:
        cases = cases[-1:]
    for name, shape, axis in cases:
        cells = shape[0] * shape[1] * shape[2]
        remaining = cutoff - time.time()
        estimate = (
            previous["wall_s"] * cells / previous["cells"] * 1.2 if previous else 1200
        )
        # Preserve time for the actual target when an intermediate 10B probe
        # plus 15B would exceed the budget. Selection uses wall time, not GCUPS.
        if name == "10B" and previous:
            target_estimate = previous["wall_s"] * 15e9 / previous["cells"] * 1.2
            if remaining < estimate + target_estimate + 180:
                rows.append(
                    dict(
                        name=name,
                        shape=shape,
                        status="skipped_to_reserve_target_time",
                        estimated_s=estimate,
                        remaining_s=remaining,
                    )
                )
                continue
        if remaining < estimate + 90:
            rows.append(
                dict(
                    name=name,
                    shape=shape,
                    status="skipped_deadline",
                    estimated_s=estimate,
                    remaining_s=remaining,
                )
            )
            break
        output = args.output / f"{name}.json"
        if output.exists():
            raise RuntimeError(f"Refusing to overwrite {output}")
        environment["BEAMZ_TRACE_PLACEMENT"] = str(
            args.output / f"{name}.placement.jsonl"
        )
        command = [
            sys.executable,
            "scripts/benchmark_modal_stepping.py",
            "--shape",
            *map(str, shape),
            "--devices",
            "8",
            "--backend",
            "cuda_streamed",
            "--shard-axis",
            axis,
            "--host-setup",
            "--donate-state",
            "--timesteps",
            "256",
            "--samples",
            "5",
            "--frequencies",
            "101",
            "--output",
            str(output),
        ]
        row = dict(
            name=name,
            shape=shape,
            shard_axis=axis,
            cells=cells,
            command=command,
            started_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
            target_only=args.target_only,
        )
        print(json.dumps(dict(stage="start", **row)), flush=True)
        start = time.monotonic()
        with (args.output / f"{name}.log").open("w") as log:
            process = subprocess.Popen(
                command,
                env=environment,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            try:
                code = process.wait(timeout=max(1, cutoff - time.time() - 30))
                row.update(returncode=code, status="ok" if code == 0 else "failed")
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
                row["status"] = "timeout"
        row["wall_s"] = time.monotonic() - start
        if row["status"] == "ok":
            result = json.loads(output.read_text())
            if not (
                result["final_state_finite"]
                and result["donate_state"]
                and result["final_current_step"] == 1536
                and result["monitor_weight_min"] == 1536
            ):
                raise RuntimeError(f"Invalid continuation result: {name}")
            row["gcups"] = result["kernel_gcups"]
            row["peak_bytes_per_cell"] = (
                sum(d["stats"]["peak_bytes_in_use"] for d in result["device_memory"])
                / cells
            )
        rows.append(row)
        (args.output / "summary.json").write_text(json.dumps(rows, indent=2) + "\n")
        print(json.dumps(dict(stage="finished", **row)), flush=True)
        if row["status"] != "ok":
            break
        previous = row
    (args.output / "summary.json").write_text(json.dumps(rows, indent=2) + "\n")


if __name__ == "__main__":
    main()
