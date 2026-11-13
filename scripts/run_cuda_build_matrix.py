#!/usr/bin/env python3
"""Serial native-build A/B trials with full-state and hardware gates.

Run only while this process owns the GPU. Each candidate has a frozen build
manifest. The installed component is restored even when a candidate is rejected.
No Git operation or source-default promotion occurs here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

FIXTURES = {
    "historical": ((160, 320, 384), "historical-reference"),
    "cube256": ((256, 256, 256), "cube256-reference"),
    "large": ((256, 512, 512), "large-reference"),
}
HARDWARE_GATES = (
    "complete_state_matches_cuda_and_jax or source_phases_and_continuation "
    "or small_domain_and_bounded_long_run or dense_h_coefficients "
    "or verified_dense_conductive_coefficients or unequal_six_face_cpml "
    "or mode_source_and_windowed_interval_dft "
    "or constant_block_headers or material_preparation_cache_order or traced "
    "or bf16_recurrence or without_cpml_sources_or_monitors or publication"
)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def python_source_digest(root):
    value = hashlib.sha256()
    for path in sorted((root / "beamz").rglob("*.py")):
        value.update(str(path.relative_to(root)).encode())
        value.update(path.read_bytes())
    return value.hexdigest()


def gpu_status():
    return subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=name,uuid,power.limit,power.draw,clocks.sm,clocks.mem,temperature.gpu,utilization.gpu,memory.used",
            "--format=csv",
        ],
        text=True,
    ).strip()


def install(source, target):
    temporary = target.with_name(".cuda-research-install.tmp")
    shutil.copy2(source, temporary)
    temporary.replace(target)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--builds-root", type=Path, required=True)
    parser.add_argument("--variants", nargs="+", required=True)
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--fixtures", nargs="+", choices=FIXTURES, default=list(FIXTURES)
    )
    parser.add_argument("--samples", type=int, default=7)
    parser.add_argument("--warmups", type=int, default=4)
    parser.add_argument("--hardware-gates", action="store_true")
    parser.add_argument("--full-hardware", action="store_true")
    parser.add_argument("--mstep-tile", choices=("16x32", "8x32", "8x16", "16x16"))
    parser.add_argument("--mstep-segment", choices=("whole", "32", "64", "128", "256"))
    parser.add_argument("--prepare-backend", choices=("cuda_streamed", "cuda_mstep"))
    args = parser.parse_args()
    if args.samples < 3 or args.warmups < 1:
        parser.error("Need at least three samples and one warmup")
    if args.full_hardware and not args.hardware_gates:
        parser.error("--full-hardware requires --hardware-gates")
    if len(set(args.variants)) != len(args.variants):
        parser.error("Variant names must be unique")
    root = Path(__file__).resolve().parents[1]
    binaries = list((root / "beamz").glob("_cuda*.so"))
    if len(binaries) != 1:
        parser.error("Expected exactly one installed CUDA component")
    active = binaries[0]
    manifests = []
    for name in args.variants:
        manifest = json.loads((args.builds_root / name / "manifest.json").read_text())
        binary = Path(manifest["binary"])
        if (
            manifest.get("status") != "ok"
            or digest(binary) != manifest["binary_sha256"]
        ):
            parser.error(f"Unverified build artifact {name}")
        manifests.append((name, manifest, binary))
    for fixture in args.fixtures:
        reference = args.reference_root / FIXTURES[fixture][1]
        if not reference.is_dir():
            parser.error(f"Missing full-state reference {reference}")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    backup = args.output_dir / "installed-before.so"
    shutil.copy2(active, backup)
    before_sha = digest(backup)
    report = {
        "schema": "beamz.cuda-native-ab.v1",
        "initial_binary_sha": before_sha,
        "fixture_cpml_cells": 12,
        "launch_tile": args.mstep_tile or "16x32",
        "launch_segment": args.mstep_segment or "whole",
        "prepare_backend": args.prepare_backend,
        "host_python_sha_at_start": python_source_digest(root),
        "variants": [],
    }
    environment = dict(os.environ)
    for key in list(environment):
        if key.startswith("BEAMZ_CUDA_"):
            del environment[key]
    environment["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    environment["PYTHONPATH"] = (
        str(root) + os.pathsep + environment.get("PYTHONPATH", "")
    )

    def persist():
        (args.output_dir / "matrix.json").write_text(
            json.dumps(report, indent=2) + "\n"
        )

    try:
        for name, manifest, binary in manifests:
            variant_dir = args.output_dir / name
            variant_dir.mkdir()
            install(binary, active)
            variant = {"name": name, "build_manifest": manifest, "trials": []}
            report["variants"].append(variant)
            print(
                json.dumps(
                    {"stage": "installed", "candidate": name, "sha": digest(active)}
                ),
                flush=True,
            )
            if args.hardware_gates:
                before = gpu_status()
                python_before = python_source_digest(root)
                started = time.perf_counter()
                with (variant_dir / "hardware.log").open("w") as log:
                    gate_command = [
                        sys.executable,
                        "-m",
                        "pytest",
                        "tests/hardware/test_cuda_mstep.py",
                        "tests/hardware/test_cuda_mstep_material_blocks.py",
                        "tests/hardware/test_cuda_traced_dependencies.py",
                        "tests/hardware/test_cuda_mstep_publication.py",
                        "-q",
                    ]
                    if not args.full_hardware:
                        gate_command += ["-k", HARDWARE_GATES]
                    process = subprocess.run(
                        gate_command,
                        cwd=root,
                        env=environment,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        timeout=600,
                    )
                variant["hardware_gate"] = {
                    "exit_code": process.returncode,
                    "wall_s": time.perf_counter() - started,
                }
                persist()
                if process.returncode:
                    variant["status"] = "rejected_hardware_gate"
                    persist()
                    continue
            for fixture in args.fixtures:
                if digest(active) != manifest["binary_sha256"]:
                    raise RuntimeError(
                        "Installed native binary changed during exclusive GPU trial"
                    )
                shape, reference_name = FIXTURES[fixture]
                output = variant_dir / f"{fixture}.json"
                command = [
                    sys.executable,
                    str(root / "scripts/benchmark_cuda_revision.py"),
                    "--root",
                    str(root),
                    "--shape",
                    *map(str, shape),
                    "--steps",
                    "256",
                    "--material",
                    "binary",
                    "--backend",
                    "cuda_mstep",
                    "--warmups",
                    str(args.warmups),
                    "--warmup-seconds",
                    "2",
                    "--samples",
                    str(args.samples),
                    "--reference",
                    str((args.reference_root / reference_name).resolve()),
                    "--output",
                    str(output.resolve()),
                ]
                if args.mstep_tile:
                    command += ["--mstep-tile", args.mstep_tile]
                if args.mstep_segment:
                    command += ["--mstep-segment", args.mstep_segment]
                if args.prepare_backend:
                    command += ["--prepare-backend", args.prepare_backend]
                before = gpu_status()
                python_before = python_source_digest(root)
                started = time.perf_counter()
                with (variant_dir / f"{fixture}.log").open("w") as log:
                    process = subprocess.run(
                        command,
                        cwd=root,
                        env=environment,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        timeout=600,
                    )
                trial = {
                    "fixture": fixture,
                    "exit_code": process.returncode,
                    "wall_s": time.perf_counter() - started,
                    "command": command,
                    "gpu_before": before,
                    "gpu_after": gpu_status(),
                    "host_python_sha_before": python_before,
                    "host_python_sha_after": python_source_digest(root),
                }
                variant["trials"].append(trial)
                if output.exists():
                    result = json.loads(output.read_text())
                    trial.update(
                        gcups=result["median_gcups"],
                        samples_s=result["samples_s"],
                        exact_state=all(
                            v["exact"] for v in result["final_state_validation"]
                        ),
                        tolerance_pass=all(
                            v["tolerance_pass"]
                            for v in result["final_state_validation"]
                        ),
                    )
                persist()
                print(json.dumps({"candidate": name, **trial}), flush=True)
                if trial["host_python_sha_before"] != trial["host_python_sha_after"]:
                    raise RuntimeError("Engine Python source changed during a trial")
                if (
                    process.returncode
                    or not trial.get("tolerance_pass")
                    or not trial.get("exact_state")
                ):
                    variant["status"] = "rejected_full_state_gate"
                    break
            else:
                variant["status"] = "passed"
            persist()
    finally:
        install(backup, active)
        report["restored_binary_sha"] = digest(active)
        persist()
    return int(any(v["status"] != "passed" for v in report["variants"]))


if __name__ == "__main__":
    raise SystemExit(main())
