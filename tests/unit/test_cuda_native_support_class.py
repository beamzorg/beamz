"""A support certificate must not erase active CPML recurrences."""

import json
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def test_native_boundary_fact_proves_full_support_but_keeps_cpml(tmp_path):
    compiler = shutil.which("g++")
    if compiler is None:
        pytest.skip("Actual native planner needs a C++17 compiler")
    binary = tmp_path / "support-proof"
    subprocess.run(
        [
            compiler,
            "-std=c++17",
            "-O2",
            "-fsanitize=undefined",
            "-I" + str(ROOT / "cuda/src"),
            str(ROOT / "scripts/prove_cuda_native_support_class.cc"),
            str(ROOT / "cuda/src/native_moving.cc"),
            "-o",
            str(binary),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    result = subprocess.run([str(binary)], capture_output=True, text=True, check=True)
    report = json.loads(result.stdout)
    assert report["pass"] and report["cases"] == 18
    assert report["owned_axis_support_checks"] > 1_000_000
    assert report["import_axis_support_checks"] > 1_000_000
    assert report["active_cpml_guards_that_must_remain"] > 0
