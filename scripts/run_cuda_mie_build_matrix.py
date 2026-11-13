#!/usr/bin/env python3
"""Serial CPML12 nanoparticle trials against immutable native build manifests.

Requires exclusive GPU ownership. Restores the installed component on exit.
The underlying worker provides cold, solve, public-call and memory measurements.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from run_cuda_build_matrix import digest, gpu_status, install, python_source_digest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--builds-root", type=Path, required=True)
    parser.add_argument("--variants", nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--resolutions-nm", nargs="+", type=float, default=[5, 4, 2.5, 2]
    )
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--public-samples", type=int, default=1)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--include-streamed", action="store_true")
    parser.add_argument("--prepare-backend", choices=("cuda_streamed", "cuda_mstep"))
    parser.add_argument(
        "--launch",
        action="append",
        default=[],
        metavar="BUILD:TILE:SEGMENT",
        help="Per-build mstep settings; absent entries use the engine default",
    )
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    launches = {}
    for setting in args.launch:
        try:
            name, tile, segment = setting.split(":")
        except ValueError:
            parser.error("--launch expects BUILD:TILE:SEGMENT")
        if name not in args.variants or tile not in ("16x32", "8x32", "8x16", "16x16"):
            parser.error("Unknown build or unsupported tile in --launch")
        if segment not in ("whole", "32", "64", "128", "256"):
            parser.error("Unsupported segment in --launch")
        launches[name] = (tile, segment)
    manifests = []
    for name in args.variants:
        manifest = json.loads((args.builds_root / name / "manifest.json").read_text())
        binary = Path(manifest["binary"])
        if (
            manifest.get("status") != "ok"
            or digest(binary) != manifest["binary_sha256"]
        ):
            parser.error(f"Unverified native build {name}")
        manifests.append((name, manifest, binary))
    binaries = list((root / "beamz").glob("_cuda*.so"))
    if len(binaries) != 1:
        parser.error("Expected one installed CUDA component")
    active = binaries[0]
    args.output_dir.mkdir(parents=True, exist_ok=False)
    backup = args.output_dir / "installed-before.so"
    shutil.copy2(active, backup)
    original_sha = digest(backup)
    report = {
        "schema": "beamz.cuda-mie-native-ab.v1",
        "initial_binary_sha": original_sha,
        "fixture_cpml_cells": 12,
        "host_python_sha": python_source_digest(root),
        "variants": [],
    }

    def persist():
        (args.output_dir / "matrix.json").write_text(
            json.dumps(report, indent=2) + "\n"
        )

    try:
        for index, (name, manifest, binary) in enumerate(manifests):
            install(binary, active)
            env = {
                k: v for k, v in os.environ.items() if not k.startswith("BEAMZ_CUDA_")
            }
            env["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
            env["PYTHONPATH"] = str(root) + os.pathsep + env.get("PYTHONPATH", "")
            if name in launches:
                tile, segment = launches[name]
                env["BEAMZ_CUDA_MSTEP_TILE"] = tile
                env["BEAMZ_CUDA_MSTEP_SEGMENT"] = segment
            output = (args.output_dir / f"{name}.json").resolve()
            command = [
                sys.executable,
                str(root / "scripts/benchmark_cuda_mie.py"),
                "--output",
                str(output),
                "--resolutions-nm",
                *map(str, args.resolutions_nm),
                "--schedules",
                *(
                    ["streamed", "mstep"]
                    if args.include_streamed and index == 0
                    else ["mstep"]
                ),
                "--samples",
                str(args.samples),
                "--public-samples",
                str(args.public_samples),
                "--warmups",
                str(args.warmups),
                "--validate-schedules",
            ]
            if args.prepare_backend:
                command += ["--prepare-backend", args.prepare_backend]
            variant = {
                "name": name,
                "build_manifest": manifest,
                "launch": launches.get(name, ("16x32", "whole")),
                "command": command,
                "gpu_before": gpu_status(),
            }
            report["variants"].append(variant)
            persist()
            with (args.output_dir / f"{name}.log").open("w") as log:
                process = subprocess.run(
                    command, cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT
                )
            variant.update(exit_code=process.returncode, gpu_after=gpu_status())
            if output.exists():
                results = json.loads(output.read_text())["results"]
                variant["case_statuses"] = [r["status"] for r in results]
                variant["native_sha_verified"] = all(
                    r.get("native_binary_sha256") == manifest["binary_sha256"]
                    for r in results
                )
                if not variant["native_sha_verified"]:
                    raise RuntimeError("Worker loaded an unexpected native component")
            if digest(active) != manifest["binary_sha256"]:
                raise RuntimeError(
                    "Installed component changed during exclusive GPU trial"
                )
            if python_source_digest(root) != report["host_python_sha"]:
                raise RuntimeError("Engine Python changed during final matrix")
            persist()
            print(
                json.dumps({"candidate": name, "exit_code": process.returncode}),
                flush=True,
            )
    finally:
        install(backup, active)
        report["restored_binary_sha"] = digest(active)
        persist()
        if report["restored_binary_sha"] != original_sha:
            raise RuntimeError("Failed to restore native component")
    return int(any(v.get("exit_code") for v in report["variants"]))


if __name__ == "__main__":
    raise SystemExit(main())
