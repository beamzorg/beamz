"""Compile and exercise graph-cache identity against the real native builder."""

import shutil
import subprocess
from pathlib import Path

import pytest


@pytest.mark.skipif(shutil.which("nvcc") is None, reason="requires CUDA toolkit")
def test_every_monitor_buffer_address_participates_in_graph_identity(tmp_path):
    root = Path(__file__).resolve().parents[2]
    executable = tmp_path / "graph-key"
    subprocess.run(
        [
            shutil.which("nvcc"),
            "-std=c++17",
            "-I",
            str(root / "cuda/src"),
            str(root / "cuda/src/graph.cu"),
            str(root / "cuda/tests/graph_key.cc"),
            "-o",
            str(executable),
        ],
        check=True,
        timeout=120,
    )
    subprocess.run([str(executable)], check=True, timeout=30)
