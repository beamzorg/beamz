"""The scheduler must order real RAW/WAW/WAR hazards, not just be acyclic."""

import json
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def test_prepared_native_cohort_hazards_and_reuse_counterexamples(tmp_path):
    compiler = shutil.which("g++")
    if compiler is None:
        pytest.skip("A C++17 compiler is required for the actual native planner")
    binary = tmp_path / "cohort-proof"
    subprocess.run(
        [
            compiler,
            "-std=c++17",
            "-O2",
            "-fsanitize=undefined",
            "-I" + str(ROOT / "scripts"),
            "-I" + str(ROOT / "cuda/src"),
            str(ROOT / "scripts/prove_cuda_native_cohort_dag.cc"),
            str(ROOT / "cuda/src/native_moving.cc"),
            "-o",
            str(binary),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    result = subprocess.run([str(binary)], check=True, capture_output=True, text=True)
    report = json.loads(result.stdout)
    assert report["pass"]
    assert report["case_count"] == 42
    assert report["hazard_reachability_checks"] > 1_000_000
    assert report["topological_schedules"] == 168
    assert report["missing_reuse_edges_rejected"] > 0
    assert report["psi_affine_handoffs"] > 0


def test_actual_host_launcher_captures_dependencies_and_rejoins(tmp_path):
    compiler = shutil.which("g++")
    if compiler is None:
        pytest.skip("A C++17 compiler is required")
    binary = tmp_path / "cohort-launch-proof"
    subprocess.run(
        [
            compiler,
            "-std=c++17",
            "-O2",
            "-fsanitize=undefined",
            "-I" + str(ROOT / "scripts"),
            "-I" + str(ROOT / "cuda/src"),
            str(ROOT / "scripts/prove_cuda_native_cohort_launch.cc"),
            str(ROOT / "cuda/src/native_moving.cc"),
            "-o",
            str(binary),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    result = subprocess.run([str(binary)], check=True, capture_output=True, text=True)
    report = json.loads(result.stdout)
    assert report["pass"]
    assert report["cases"] == 6
    assert report["macros_per_case"] == 3
    assert report["capture_dependency_checks"] > 1000
    assert report["cohort_launches"] > 1000
