"""Actual copied lowering is parent-bound and topology guards fail closed."""

import shutil
from pathlib import Path

import pytest

from scripts.patch_cuda_cyclic_source import ENTRY10, PARENT, PLANE10, transformed
from scripts.validate_cuda_cyclic_source import validate


SOURCE = Path(".cache/cuda-two-hour-2026-11-13-0819/source-plane-axis-cg-v1/source")


def test_wrong_parent_is_rejected_before_any_emission():
    with pytest.raises(ValueError, match="Exact frozen52865"):
        transformed({n: b"wrong" for n in PARENT})


def test_mask10_plane_callback_retains_two_separate_ordered_products_and_adds():
    assert PLANE10.count("__fmul_rn(") == PLANE10.count("__fadd_rn(") == 2
    assert "__fmaf" not in PLANE10
    assert "z==s.starts[0]" in PLANE10
    assert PLANE10.index("s.coefficients+offset") < PLANE10.index("s.coefficients+s.ny*s.nx+offset")
    assert "Policy==3||Policy==4" in ENTRY10


def test_actual_parent_keeps_ordered_cell_graph_and_all_original_entries(tmp_path):
    if not SOURCE.exists():
        pytest.skip("Frozen52865 CUDA source required")
    before = {n: (SOURCE / "src" / n).read_bytes() for n in PARENT}
    after = transformed(before)
    assert after["native_moving.cc"] == before["native_moving.cc"]
    assert after["asymmetric_native_cell_generated.cuh"] == before["asymmetric_native_cell_generated.cuh"]
    old = before["asymmetric_native.cuh"].decode()
    new = after["asymmetric_native.cuh"].decode()
    for name in ("AsymmetricSourcePending20", "AsymmetricSourcePlanePair20"):
        a = old.index("template<int Depth,int Policy>\n__global__ __launch_bounds__(1024,1) void " + name)
        b = old.index("\n}\n", a) + 3
        assert old[a:b] in new
    if not shutil.which("c++"):
        pytest.skip("Host C++ compiler required")
    out = tmp_path / "source"
    shutil.copytree(SOURCE, out)
    for n, value in after.items():
        (out / "src" / n).write_bytes(value)
    proof = validate(out, tmp_path / "proof")
    assert proof["fp_cases"] == 49152
    assert proof["rejected_topology_mutations"] == 24
    assert proof["errors"] == 0
