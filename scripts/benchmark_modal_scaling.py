#!/usr/bin/env python3
"""Fresh-process modal CPML size sweep with explicit strong/weak scaling.

The weak suite lengthens a coupled waveguide along x at fixed resolution and
cross-section. It is one simulation, not independent replicas. Results count
physical material cells and complete Yee steps. No performance extrapolation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TARGETS = {"cuda_streamed": 150.0, "jax": 75.0}


def cases(counts, local_sizes, cubes, frequencies):
    for frequency in frequencies:
        for count in counts:
            for side in local_sizes:
                yield "weak", (side, side, side * count), count, frequency
            for side in cubes:
                yield "strong", (side, side, side), count, frequency


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--counts", nargs="+", type=int, default=[1, 2, 4, 8])
    parser.add_argument("--local-sizes", nargs="*", type=int, default=[256, 384, 512])
    parser.add_argument("--cubes", nargs="*", type=int, default=[512, 768, 1024])
    parser.add_argument("--frequencies", nargs="+", type=int, default=[3, 101])
    parser.add_argument("--backends", nargs="+", choices=TARGETS, default=list(TARGETS))
    parser.add_argument("--timesteps", type=int, default=256)
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--max-wall-seconds", type=float, default=3600)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--host-setup",
        action="store_true",
        help="Avoid constructing the complete global grid on GPU 0",
    )
    args = parser.parse_args()
    if (
        any(n not in (1, 2, 4, 8) for n in args.counts)
        or any(n <= 24 for n in [*args.local_sizes, *args.cubes])
        or min(args.frequencies) < 1
        or args.timesteps < 32
        or args.samples < 5
        or args.timeout <= 0
        or args.max_wall_seconds <= 0
    ):
        parser.error(
            "need counts 1/2/4/8, dimensions >24, positive frequencies/time limits, >=32 steps and >=5 samples"
        )
    args.output.mkdir(parents=True, exist_ok=False)
    manifest = {
        "targets_gcups_8gpu": TARGETS,
        "commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "physics": {
            "cpml_cells": 12,
            "source": "mode",
            "monitors": 2,
            "monitor_type": "mode",
            "resolution_m": 80e-9,
            "material": "binary_waveguide",
        },
        "args": {
            k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()
        },
        "note": "Warm throughput only; 256 steps may precede pulse arrival at distant monitors. Validate spectra separately with propagation-length-dependent duration.",
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (args.output / "source.patch").write_text(
        subprocess.check_output(["git", "diff"], cwd=ROOT, text=True)
    )
    env = dict(
        os.environ,
        PYTHONPATH=str(ROOT),
        LD_LIBRARY_PATH="",
        XLA_PYTHON_CLIENT_PREALLOCATE="false",
        XLA_PYTHON_CLIENT_MEM_FRACTION=".80",
        BEAMZ_CUDA_CPML_PSI_PRECISION="fp32",
        NUMPY_MADVISE_HUGEPAGE="0",
    )
    started = time.monotonic()
    records = []
    for index, (suite, shape, count, frequency) in enumerate(
        cases(args.counts, args.local_sizes, args.cubes, args.frequencies)
    ):
        for backend in args.backends if index % 2 == 0 else reversed(args.backends):
            remaining = args.max_wall_seconds - (time.monotonic() - started)
            if remaining <= 0:
                print(
                    "Sweep wall-time budget exhausted; pod teardown is the caller's responsibility.",
                    flush=True,
                )
                return
            name = f"{suite}-{'x'.join(map(str, shape))}-{count}gpu-{backend}-{frequency}freq"
            output = args.output / f"{name}.json"
            command = [
                sys.executable,
                str(ROOT / "scripts/benchmark_h100_backends.py"),
                "--worker",
                "--workload",
                "modal_cpml12",
                "--shape",
                *map(str, shape),
                "--devices",
                str(count),
                "--backend",
                backend,
                "--shard-axis",
                "x",
                "--frequencies",
                str(frequency),
                "--timesteps",
                str(args.timesteps),
                "--samples",
                str(args.samples),
                "--output",
                str(output.resolve()),
            ]
            if args.host_setup:
                command.append("--host-setup")
            record = dict(
                name=name,
                suite=suite,
                shape=shape,
                devices=count,
                backend=backend,
                frequencies=frequency,
                command=command,
            )
            print(name, flush=True)
            if args.dry_run:
                record["status"] = "planned"
            else:
                trial_start = time.monotonic()
                trial_env = dict(
                    env, CUDA_VISIBLE_DEVICES=",".join(map(str, range(count)))
                )
                with (args.output / f"{name}.log").open("w") as log:
                    try:
                        result = subprocess.run(
                            command,
                            env=trial_env,
                            cwd=ROOT,
                            stdout=log,
                            stderr=subprocess.STDOUT,
                            timeout=min(args.timeout, remaining),
                        )
                        record["status"] = (
                            "ok"
                            if result.returncode == 0 and output.exists()
                            else "failed"
                        )
                        record["returncode"] = result.returncode
                    except subprocess.TimeoutExpired:
                        record["status"] = "timeout"
                record["wall_s"] = time.monotonic() - trial_start
                if record["status"] == "ok":
                    data = json.loads(output.read_text())
                    record["kernel_gcups"] = data["kernel_gcups"]
                    record["peak_memory_bytes"] = data["peak_memory_bytes"]
            records.append(record)
            (args.output / "summary.json").write_text(
                json.dumps(records, indent=2) + "\n"
            )


if __name__ == "__main__":
    main()
