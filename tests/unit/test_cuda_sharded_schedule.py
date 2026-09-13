"""Exercise the production launch indexing without a CUDA device."""

import os
import shutil
import subprocess
from pathlib import Path


def test_sharded_launch_covers_elongated_supports(tmp_path):
    compiler = os.environ.get("CXX") or shutil.which("clang++") or shutil.which("g++")
    assert compiler, "The native schedule test requires a C++17 compiler"
    root = Path(__file__).resolve().parents[2]
    executable = tmp_path / "sharded_schedule"
    subprocess.run(
        [
            compiler,
            "-std=c++17",
            "-O1",
            f"-I{root / 'cuda/src'}",
            str(root / "tests/unit/cuda_sharded_schedule.cc"),
            "-o",
            str(executable),
        ],
        check=True,
        timeout=90,
    )
    subprocess.run([str(executable)], check=True, timeout=30)
