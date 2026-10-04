#!/usr/bin/env python3
"""Run a sequential lossless domain-shape study with exactly 12 CPML cells.

Uses fixed-aperture mode monitors to avoid scaling their area with the domain.
The full-field controls deliberately measure the additional aperture workload.
All dimensions are physical material-grid dimensions, ordered z, y, x. No
physical-domain rounding is introduced by this study. Optional storage padding
is excluded from useful-cell counts.
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path


def cases():
    result = []

    def add(name, shape, **options):
        result.append(dict(name=name, shape=shape, **options))

    for shape in (
        (64, 128, 256),
        (128, 128, 256),
        (128, 256, 256),
        (128, 256, 512),
        (128, 256, 768),
        (256, 256, 256),
    ):
        add("volume_" + "_".join(map(str, shape)), shape)
    for shape in itertools.permutations((64, 256, 1024)):
        add("aspect_" + "_".join(map(str, shape)), shape)
    for axis, center in enumerate((128, 256, 512)):
        for delta in (-1, 1):
            shape = [128, 256, 512]
            shape[axis] = center + delta
            add(f"alignment_{'zyx'[axis]}_{center + delta}", shape)
    # Interior extent and material extent have different alignment boundaries.
    for nx in (504, 520, 536, 544):
        add(f"alignment_x_{nx}", (128, 256, nx))
    for shape in ((128, 256, 512), (64, 256, 1024), (1024, 256, 64)):
        add("full_field_" + "_".join(map(str, shape)), shape, monitor_type="field")
    add("long", (128, 256, 512), steps=1024)
    add("developed", (128, 256, 512), presteps=1024)
    add("scale_96_192_384", (96, 192, 384))
    add("scale_144_288_576", (144, 288, 576))
    add("irregular_97_289_593", (97, 289, 593))
    add("irregular_71_479_503", (71, 479, 503))
    add("dispatch_64_256_1024_off", (64, 256, 1024), fusion="0")
    add("dispatch_64_1024_256_on", (64, 1024, 256), fusion="1")
    add("dispatch_1024_64_256_on", (1024, 64, 256), fusion="1")
    add("dispatch_1024_256_64_on", (1024, 256, 64), fusion="1")
    return result


def telemetry():
    raw = subprocess.check_output(
        [
            "nvidia-smi",
            "-i",
            "0",
            "--query-gpu=memory.free,memory.used,utilization.gpu,power.draw,power.limit,temperature.gpu,clocks.sm",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    )
    keys = (
        "free_mib",
        "used_mib",
        "utilization",
        "power_w",
        "limit_w",
        "temperature_c",
        "sm_mhz",
    )
    return dict(zip(keys, (float(x.strip()) for x in raw.split(",")), strict=True))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument(
        "--source-root",
        type=Path,
        help="Optional frozen checkout for baseline comparisons",
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--cases", nargs="*", help="Case names; default: complete study"
    )
    parser.add_argument("--seed", type=int, default=19)
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument(
        "--padding", choices=("none", "32", "64", "32x8", "64x8"), default="none"
    )
    parser.add_argument(
        "--tile", choices=("auto", "32x8x8", "64x4x8", "32x4x8"), default="auto"
    )
    parser.add_argument("--temporal-steps", type=int, choices=(1, 2), default=1)
    parser.add_argument("--fusion", choices=("auto", "0", "1"), default="auto")
    parser.add_argument(
        "--shell-tile", choices=("64x4", "32x8", "32x4"), default="64x4"
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    selected = cases()
    if args.cases:
        unknown = set(args.cases) - {case["name"] for case in selected}
        if unknown:
            parser.error(f"Unknown cases: {sorted(unknown)}")
        selected = [case for case in selected if case["name"] in args.cases]
    random.Random(args.seed).shuffle(selected)
    if args.dry_run:
        print(json.dumps(selected, indent=2))
        return
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "manifest.json").write_text(json.dumps(selected, indent=2) + "\n")
    (args.output / "settings.json").write_text(
        json.dumps(vars(args), default=str, indent=2) + "\n"
    )
    root = Path(__file__).resolve().parents[1]
    env = dict(
        os.environ,
        PYTHONPATH=str(args.source_root.resolve() if args.source_root else root),
        XLA_PYTHON_CLIENT_PREALLOCATE="false",
        XLA_PYTHON_CLIENT_MEM_FRACTION=".45",
        BEAMZ_CUDA_FIELD_PADDING=args.padding,
        BEAMZ_CUDA_CPML_TILE=args.tile,
        BEAMZ_CUDA_TEMPORAL_STEPS=str(args.temporal_steps),
        BEAMZ_CUDA_CPML_SHELL_TILE=args.shell_tile,
    )
    env.pop("BEAMZ_CUDA_CPML_CORE_FUSION", None)
    if args.fusion != "auto":
        env["BEAMZ_CUDA_CPML_CORE_FUSION"] = args.fusion
    for number, case in enumerate(selected, 1):
        name = case["name"]
        output = args.output / f"{name}.json"
        command = [
            args.python,
            str(root / "scripts/benchmark_cuda_realistic.py"),
            "--shape",
            *map(str, case["shape"]),
            "--pml",
            "12",
            "--material",
            "binary",
            "--source",
            "mode",
            "--monitor-type",
            case.get("monitor_type", "mode"),
            "--monitors",
            "2",
            "--frequencies",
            "3",
            "--steps",
            str(case.get("steps", 256)),
            "--presteps",
            str(case.get("presteps", 0)),
            "--samples",
            str(args.samples),
            "--warmups",
            "4",
            "--output",
            str(output.resolve()),
        ]
        print(f"[{number}/{len(selected)}] {name}", flush=True)
        case_env = env.copy()
        if "fusion" in case:
            case_env["BEAMZ_CUDA_CPML_CORE_FUSION"] = case["fusion"]
        readings = []
        started = time.monotonic()
        with (args.output / f"{name}.log").open("w") as log:
            process = subprocess.Popen(
                command, cwd=root, env=case_env, stdout=log, stderr=log
            )
            try:
                while process.poll() is None:
                    reading = telemetry()
                    reading["elapsed_s"] = time.monotonic() - started
                    readings.append(reading)
                    if reading["free_mib"] < 2048:
                        raise RuntimeError(
                            "Stopped benchmark to preserve 2 GiB of GPU headroom"
                        )
                    if reading["temperature_c"] >= 85:
                        raise RuntimeError("Stopped benchmark at 85 C")
                    if reading["elapsed_s"] > 300:
                        raise RuntimeError(
                            "Benchmark exceeded five-minute case timeout"
                        )
                    time.sleep(1)
            finally:
                if process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
                (args.output / f"{name}.telemetry.json").write_text(
                    json.dumps(readings, indent=2) + "\n"
                )
            if process.returncode:
                raise RuntimeError(f"{name} failed; see {name}.log")
        data = json.loads(output.read_text())
        print(f"  {data['median_gcups']:.3f} GCUPS", flush=True)


if __name__ == "__main__":
    main()
