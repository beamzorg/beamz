"""A source topology class cannot silently erase transverse support guards."""

import shutil
from pathlib import Path

import pytest

from scripts.patch_cuda_v12_source_plane_pair import (
    ENTRY,
    ORIGINAL,
    PARENT_HASHES,
    PLANE,
    patch_copy,
    transformed,
)
from scripts.validate_cuda_source_plane_pair import validate


def test_plane_specialization_retains_two_products_and_ordered_adds():
    plane = PLANE.split("    }else{")[0]
    assert plane.count("__fmul_rn(") == plane.count("__fadd_rn(") == 2
    assert "__fmaf" not in plane
    assert plane.index("s.coefficients+offset") < plane.index("s.coefficients+s.nz*s.ny+offset")
    assert "x==s.starts[2]" in plane and "z-12" in plane and "y-12" in plane
    assert ORIGINAL in PLANE


def test_only_cpml_free_source_policies_instantiate_plane_body():
    assert "Policy==3||Policy==4" in ENTRY
    assert "SourcePendingPairedBody<Depth,1,false,true,Policy,false,1024,20,true>" in ENTRY


@pytest.mark.parametrize("n", [32, 35, 64, 136, 512, 564])
def test_every_cpml_free_transverse_endpoint_is_inside_full_plane(n):
    for coord in (12, n - 13):
        assert 0 <= coord - 12 < n - 24
    # The corresponding PML-adjacent point is outside the topology guarantee.
    assert 11 - 12 < 0
    assert n - 12 - 12 == n - 24


def test_wrong_parent_rejected_before_any_copy_or_kernel_emission():
    with pytest.raises(ValueError, match="Exact v12"):
        transformed({n: b"wrong" for n in PARENT_HASHES})


def test_changed_planner_cannot_inherit_transverse_bounds_fact(tmp_path):
    source, output = tmp_path / "source", tmp_path / "output"
    (source / "src").mkdir(parents=True)
    (source / "src/native_moving.cc").write_bytes(b"different trajectory flags")
    with pytest.raises(ValueError, match="planner trajectory"):
        patch_copy(source, output)
    assert not output.exists()


def test_exact_parent_composition_keeps_storage_schedule_default_and_private_starts(tmp_path):
    source = Path(".cache/cuda-five-hour-2026-11-13/source-pending-paired-on-v12/source/src")
    if not source.exists():
        pytest.skip("Frozen research parent required")
    parent = {n: (source / n).read_bytes() for n in PARENT_HASHES}
    result = transformed(parent)
    c = result["asymmetric_native.cuh"].decode()
    assert "int SourceMask=-1,bool PlanePair=false> struct Environment" in c
    assert "UniformE,SourceMask,PlanePair>env" in c
    assert "SourcePendingPairedBody<Depth,1,false,true,Policy,false,1024,20>(p,ids)" in c
    assert result["asymmetric_native_cell_generated.cuh"] == parent["asymmetric_native_cell_generated.cuh"]
    host = result["native_extension.cc"].decode()
    assert "(i-source_offset_)%3==2" in host and "host_[source_offset_+3*g+2].data" in host
    if not shutil.which("c++"):
        pytest.skip("Host C++ compiler required")
    out = tmp_path / "source"
    shutil.copytree(source.parent, out)
    for n, data in result.items():
        (out / "src" / n).write_bytes(data)
    r = validate(out, tmp_path / "oracle")
    assert r["errors"] == 0 and r["fp_cases"] == 49152
    assert r["rejected_topology_mutations"] == 22
