"""Sequential, seeded large-shape comparison of main and the current checkout.

Uses the portable revision worker for identical CPML12 mode-source/monitor
workloads and ABBA process order. Does not tune layouts or change GPU settings.
"""

import argparse
import hashlib
import json
import math
import os
import random
import subprocess
import sys
import tarfile
import time
from pathlib import Path

# This runner is also invoked by absolute path from another working directory.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.benchmark_cuda_shapes import telemetry  # noqa: E402


def random_cases(seed):
    rng = random.Random(seed)
    strata = [
        ("short_x", ((700, 1500), (160, 400), (50, 110)), "binary"),
        ("long_y", ((55, 110), (700, 1400), (180, 400)), "smooth"),
        ("flat_z", ((65, 140), (180, 400), (400, 1000)), "binary"),
        ("irregular", ((190, 390), (180, 350), (200, 450)), "smooth"),
    ]
    cases = []
    for name, bounds, material in strata:
        while True:
            shape = tuple(rng.randint(low, high) for low, high in bounds)
            if 16 * 1024**2 <= math.prod(shape) <= 24 * 1024**2:
                break
        cases.append(dict(name=name, shape=shape, material=material, bounds=bounds))
    return cases


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def source_hash(root):
    digest = hashlib.sha256()
    paths = sorted([*root.glob("beamz/**/*.py"), *root.glob("cuda/src/*")])
    for path in paths:
        if path.is_file():
            digest.update(str(path.relative_to(root)).encode())
            digest.update(path.read_bytes())
    return digest.hexdigest(), paths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--main-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=2026091807)
    parser.add_argument(
        "--build-description", default="Not supplied; see native hashes"
    )
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    main_root = args.main_root.resolve()
    worker = ROOT / "scripts/benchmark_cuda_revision.py"
    cases = random_cases(args.seed)
    main_hash, _ = source_hash(main_root)
    branch_hash, branch_paths = source_hash(ROOT)
    write_json(
        out / "provenance.json",
        dict(
            seed=args.seed,
            cases=cases,
            selection="Stratified random integer extents, accepted at 16–24 Mi cells",
            process_order=["main", "branch", "branch", "main"],
            steps=256,
            warmups=4,
            samples=9,
            main_commit=subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=main_root, text=True
            ).strip(),
            branch_head=subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
            ).strip(),
            main_source_sha256=main_hash,
            branch_source_sha256=branch_hash,
            native_build=args.build_description,
            precision="FP32",
            pml=12,
            branch_selection="Default predictive auto, no manual overrides",
        ),
    )
    with tarfile.open(out / "branch-solver-snapshot.tar.gz", "w:gz") as archive:
        for path in branch_paths + [Path(__file__), worker]:
            if path.is_file():
                archive.add(path, arcname=str(path.relative_to(ROOT)))
    records = []
    for case in cases:
        reference_data = None
        counts = {}
        for index, variant in enumerate(("main", "branch", "branch", "main")):
            counts[variant] = counts.get(variant, 0) + 1
            tag = f"{case['name']}-{variant}-{counts[variant]}"
            checkout = main_root if variant == "main" else ROOT
            env = dict(
                os.environ,
                PYTHONPATH=str(checkout),
                XLA_PYTHON_CLIENT_PREALLOCATE="false",
                XLA_PYTHON_CLIENT_MEM_FRACTION=".45",
                PYTHONUNBUFFERED="1",
            )
            cmd = [
                sys.executable,
                str(worker),
                "--root",
                str(checkout),
                "--shape",
                *map(str, case["shape"]),
                "--material",
                case["material"],
                "--output",
                str(out / f"{tag}.json"),
                "--reference",
                str(args.reference_dir.resolve() / case["name"]),
            ]
            if index == 0:
                cmd.append("--save-reference")
            print("START", tag, case["shape"], flush=True)
            readings = []
            start = time.monotonic()
            status = telemetry()
            if status["free_mib"] < 4096 or status["temperature_c"] >= 80:
                raise RuntimeError(status)
            with (out / f"{tag}.log").open("w") as log:
                proc = subprocess.Popen(
                    cmd, cwd=checkout, env=env, stdout=log, stderr=log
                )
                try:
                    while proc.poll() is None:
                        status = dict(telemetry(), elapsed_s=time.monotonic() - start)
                        readings.append(status)
                        if (
                            status["free_mib"] < 2048
                            or status["temperature_c"] >= 85
                            or status["elapsed_s"] > 600
                        ):
                            raise RuntimeError(status)
                        time.sleep(1)
                finally:
                    if proc.poll() is None:
                        proc.terminate()
                        try:
                            proc.wait(timeout=10)
                        except subprocess.TimeoutExpired:
                            proc.kill()
                            proc.wait()
                    write_json(out / f"{tag}.telemetry.json", readings)
                if proc.returncode:
                    raise RuntimeError(f"{tag} failed; see its log")
            data = json.loads((out / f"{tag}.json").read_text())
            if reference_data is None:
                reference_data = data
            for key in ("workload", "initial_state", "coefficients", "sources"):
                assert data[key] == reference_data[key], (tag, key)
            if variant == "branch":
                assert data["tuning"]["selection_method"] == "geometry"
                assert data["tuning"]["calibration_s"] == 0
            records.append(
                dict(
                    case=case["name"],
                    variant=variant,
                    repeat=counts[variant],
                    median_gcups=data["median_gcups"],
                )
            )
            write_json(out / "summary.json", records)
            print("DONE", tag, data["median_gcups"], flush=True)
    assert source_hash(ROOT)[0] == branch_hash, (
        "Branch solver changed during comparison"
    )
    assert source_hash(main_root)[0] == main_hash, (
        "Main solver changed during comparison"
    )
    print("COMPLETE", flush=True)


if __name__ == "__main__":
    main()
