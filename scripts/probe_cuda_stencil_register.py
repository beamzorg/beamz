#!/usr/bin/env python3
"""Compile/run the generated one-shared-plane temporal stencil experiment.

No production backend or API changes. This synthetic periodic/common-extent
probe compares every result with the independent C++ scalar reference from
probe_cuda_stencil_trace. GPU execution requires --run explicitly.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from datetime import UTC, datetime
from pathlib import Path

try:
    from scripts.cuda_stencil_register import (
        generate_register_probe,
        register_resource_estimate,
    )
except ModuleNotFoundError:
    from cuda_stencil_register import (
        generate_register_probe,
        register_resource_estimate,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--nvcc", default=shutil.which("nvcc"))
    parser.add_argument("--arch", default="sm_86")
    parser.add_argument(
        "--tile", type=int, nargs=3, default=(32, 8, 16), metavar=("Z", "Y", "X")
    )
    parser.add_argument(
        "--shape", type=int, nargs=3, default=(23, 29, 37), metavar=("Z", "Y", "X")
    )
    parser.add_argument("--depths", type=int, nargs="+", default=(1, 2, 4))
    parser.add_argument("--steps", type=int, default=9)
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--e-history", choices=("register", "shared"), default="shared")
    parser.add_argument("--h-history", choices=("register", "shared"), default="shared")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    if any(value < 1 for value in args.shape) or args.steps < 1 or args.samples < 1:
        parser.error("shape, steps and samples must be positive")
    depths = tuple(sorted(set(args.depths)))
    if 1 not in depths or any(depth < 1 for depth in depths):
        parser.error("depths must be positive and include one")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    source = args.output_dir / "register_trace.cu"
    binary = args.output_dir / "register_trace"
    source.write_text(
        generate_register_probe(
            tuple(args.tile), depths, e_history=args.e_history, h_history=args.h_history
        )
    )
    report = {
        "schema": "beamz.experimental.register-trace-probe.v1",
        "scope": "synthetic periodic equal-extent scalar fields, not production FDTD throughput",
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "shape_zyx": args.shape,
        "tile_zyx": args.tile,
        "depths": depths,
        "steps": args.steps,
        "samples": args.samples,
        "e_history": args.e_history,
        "h_history": args.h_history,
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "resources": [
            register_resource_estimate(
                tuple(args.tile),
                depth,
                e_history=args.e_history,
                h_history=args.h_history,
            )
            for depth in depths
        ],
        "depth_zero_control": "same equations in two ordinary global-memory kernels per timestep",
    }
    if args.compile or args.run:
        if not args.nvcc:
            parser.error("nvcc is required for --compile/--run")
        command = [
            args.nvcc,
            "-std=c++17",
            "-O3",
            "--fmad=false",
            "-Xcompiler=-ffp-contract=off",
            f"-arch={args.arch}",
            "-Xptxas=-v",
            str(source),
            "-o",
            str(binary),
        ]
        build = subprocess.run(command, capture_output=True, text=True, timeout=180)
        (args.output_dir / "build.log").write_text(build.stdout + build.stderr)
        build.check_returncode()
        report["compile_command"] = command
        report["binary_sha256"] = hashlib.sha256(binary.read_bytes()).hexdigest()
    if args.run:
        nz, ny, nx = args.shape
        result = subprocess.run(
            [
                str(binary.resolve()),
                str(nx),
                str(ny),
                str(nz),
                str(args.steps),
                str(args.samples),
            ],
            capture_output=True,
            text=True,
            timeout=180,
        )
        (args.output_dir / "run.log").write_text(result.stdout + result.stderr)
        result.check_returncode()
        report["results"] = [json.loads(line) for line in result.stdout.splitlines()]
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
