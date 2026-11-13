#!/usr/bin/env python3
"""Serial, fresh-process CUDA launch-variant trials with frozen binary checks.

Uses the retained CPML12 realistic fixture and full-state reference machinery.
Generated reports live in a caller-selected artifact directory; this does not
change engine defaults or choose a new schedule automatically.
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

TILES = {"16x32", "8x32", "8x16", "16x16"}
SEGMENTS = {"whole", "32", "64", "128", "256"}


def gpu_status():
    return subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=name,uuid,power.limit,power.draw,clocks.sm,clocks.mem,temperature.gpu,utilization.gpu,memory.used",
            "--format=csv",
        ],
        text=True,
    ).strip()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--shape", type=int, nargs=3, required=True)
    parser.add_argument("--steps", type=int, default=256)
    parser.add_argument("--material", choices=("binary", "smooth"), default="binary")
    parser.add_argument("--boundary", choices=("all", "right", "pec"), default="all")
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--warmup-seconds", type=float, default=1)
    parser.add_argument("--expected-binary-sha", required=True)
    parser.add_argument(
        "--variants",
        nargs="+",
        default=["16x32:whole", "16x32:32", "8x16:32", "8x32:64", "16x16:32"],
    )
    args = parser.parse_args()
    if args.samples < 3 or args.warmups < 1 or args.steps < 2:
        parser.error("Need >=3 samples, >=1 warmup, >=2 timesteps")
    variants = []
    for variant in args.variants:
        parts = variant.split(":")
        if len(parts) != 2 or parts[0] not in TILES or parts[1] not in SEGMENTS:
            parser.error(f"Unsupported tile:segment {variant}")
        variants.append(parts)
    if len(set(args.variants)) != len(args.variants):
        parser.error("Duplicate variants would overwrite a trial")
    root = Path(__file__).resolve().parents[1]
    binaries = list((root / "beamz").glob("_cuda*.so"))
    if len(binaries) != 1:
        parser.error("Expected exactly one local native CUDA component")
    binary = binaries[0]
    args.output_dir.mkdir(parents=True, exist_ok=False)
    report = {
        "fixture": "retained_realistic_cpml12",
        "shape_zyx": args.shape,
        "steps": args.steps,
        "trials": [],
        "expected_binary_sha": args.expected_binary_sha,
    }
    for tile, segment in variants:
        digest = hashlib.sha256(binary.read_bytes()).hexdigest()
        if digest != args.expected_binary_sha:
            raise RuntimeError(
                "Native binary changed; refusing mixed-binary trial matrix"
            )
        label = f"{tile}-{segment}"
        output = args.output_dir / f"{label}.json"
        command = [
            sys.executable,
            str(root / "scripts/benchmark_cuda_revision.py"),
            "--root",
            str(root),
            "--shape",
            *map(str, args.shape),
            "--steps",
            str(args.steps),
            "--backend",
            "cuda_mstep",
            "--material",
            args.material,
            "--boundary",
            args.boundary,
            "--samples",
            str(args.samples),
            "--warmups",
            str(args.warmups),
            "--warmup-seconds",
            str(args.warmup_seconds),
            "--reference",
            str(args.reference.resolve()),
            "--output",
            str(output.resolve()),
            "--mstep-tile",
            tile,
            "--mstep-segment",
            segment,
        ]
        environment = dict(os.environ)
        environment["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
        before = gpu_status()
        started = time.perf_counter()
        with (args.output_dir / f"{label}.log").open("w") as log:
            process = subprocess.run(
                command, env=environment, stdout=log, stderr=subprocess.STDOUT
            )
        trial = {
            "tile": tile,
            "segment": segment,
            "command": command,
            "exit_code": process.returncode,
            "wall_s": time.perf_counter() - started,
            "gpu_before": before,
            "gpu_after": gpu_status(),
            "binary_sha": digest,
        }
        if output.exists():
            result = json.loads(output.read_text())
            trial.update(
                gcups=result["median_gcups"],
                samples_s=result["samples_s"],
                exact_state=all(v["exact"] for v in result["final_state_validation"]),
                tolerance_pass=all(
                    v["tolerance_pass"] for v in result["final_state_validation"]
                ),
            )
        report["trials"].append(trial)
        (args.output_dir / "matrix.json").write_text(
            json.dumps(report, indent=2) + "\n"
        )
        print(json.dumps(trial), flush=True)
        if process.returncode or not trial.get("tolerance_pass"):
            raise RuntimeError(f"Trial {label} failed; matrix stopped")


if __name__ == "__main__":
    main()
