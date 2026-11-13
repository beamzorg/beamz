#!/usr/bin/env python3
"""Build an immutable, precisely identified CUDA research candidate.

The staged cuda/ source is a build artifact, not a Git checkout. Copying before
compilation prevents concurrent edits from silently changing a candidate.
The existing installed native component is never replaced by this script.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def digest_tree(root):
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*")):
        if path.is_file():
            digest.update(str(path.relative_to(root)).encode())
            digest.update(path.read_bytes())
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--nvcc", type=Path, required=True)
    parser.add_argument("--arch", default="86")
    parser.add_argument("--jobs", type=int, default=2)
    parser.add_argument("--option", action="append", default=[])
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error("output-dir must be new; candidate artifacts are immutable")
    if args.jobs < 1:
        parser.error("jobs must be positive")
    if any(
        not option.startswith("BEAMZ_CUDA_") or "=" not in option
        for option in args.option
    ):
        parser.error("options must be BEAMZ_CUDA_NAME=VALUE")
    output = args.output_dir.resolve()
    source, build = output / "source", output / "build"
    output.mkdir(parents=True)
    shutil.copytree(
        ROOT / "cuda",
        source,
        ignore=shutil.ignore_patterns("build", "dist", "__pycache__", "*.egg-info"),
    )
    # Resolving a venv interpreter symlink loses its pyvenv.cfg and packages.
    python = args.python.absolute()
    cmake = python.parent / "cmake"
    ninja = python.parent / "ninja"
    if not cmake.exists():
        cmake = Path(shutil.which("cmake") or "cmake")
    if not ninja.exists():
        ninja = Path(shutil.which("ninja") or "ninja")
    nanobind = subprocess.check_output(
        [str(python), "-c", "import nanobind; print(nanobind.cmake_dir())"], text=True
    ).strip()
    command = [
        str(cmake),
        "-S",
        str(source),
        "-B",
        str(build),
        "-G",
        "Ninja",
        f"-DCMAKE_MAKE_PROGRAM={ninja}",
        f"-DCMAKE_CUDA_COMPILER={args.nvcc.resolve()}",
        f"-DPython_EXECUTABLE={python}",
        f"-Dnanobind_DIR={nanobind}",
        f"-DBEAMZ_CUDA_ARCHITECTURES={args.arch}",
        "-DCMAKE_BUILD_TYPE=Release",
        "-DBEAMZ_CUDA_FAST_MATH=OFF",
        *["-D" + option for option in args.option],
    ]
    manifest = {
        "schema": "beamz.cuda-research-build.v1",
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "git_head": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "source_sha256": digest_tree(source),
        "configure_command": command,
        "build_command": [str(cmake), "--build", str(build), "-j", str(args.jobs)],
        "status": "building",
    }
    (output / "source.patch").write_bytes(
        subprocess.check_output(["git", "diff", "--binary", "--", "cuda"], cwd=ROOT)
    )
    metadata = output / "manifest.json"
    metadata.write_text(json.dumps(manifest, indent=2) + "\n")
    start = time.monotonic()
    try:
        with (output / "build.log").open("w") as log:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
            subprocess.run(
                manifest["build_command"],
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
            )
        binaries = list(build.glob("_cuda*.so"))
        if len(binaries) != 1:
            raise RuntimeError("Expected exactly one built native component")
        manifest.update(
            status="ok",
            binary=str(binaries[0]),
            binary_sha256=hashlib.sha256(binaries[0].read_bytes()).hexdigest(),
        )
    except (OSError, RuntimeError, subprocess.CalledProcessError) as error:
        manifest.update(status="error", error=str(error))
    manifest["wall_s"] = time.monotonic() - start
    metadata.write_text(json.dumps(manifest, indent=2) + "\n")
    print(str(metadata), flush=True)
    return int(manifest["status"] != "ok")


if __name__ == "__main__":
    raise SystemExit(main())
