"""Source count/plane and NoIO axis facts occupy distinct template arguments."""

from pathlib import Path

import pytest

from scripts.patch_cuda_source_plane_axis_bundle import PARENTS, patch_copy, transformed


def test_wrong_parent_cannot_inherit_either_fact_class():
    with pytest.raises(ValueError, match="Exact source-plane"):
        transformed({n: b"wrong" for n in PARENTS})


def test_changed_planner_fails_before_any_source_copy(tmp_path):
    source, output = tmp_path / "source", tmp_path / "output"
    (source / "src").mkdir(parents=True)
    (source / "src/native_moving.cc").write_bytes(b"different trajectory flags")
    with pytest.raises(ValueError, match="planner trajectory"):
        patch_copy(source, output, tmp_path / "reference")
    assert not output.exists()


def test_actual_composition_preserves_plane_tail_and_binds_all_three_fact_arguments():
    source = Path(".cache/cuda-two-hour-2026-11-13-0819/source-plane-pair-v1/source/src")
    if not source.exists():
        pytest.skip("Frozen research parent required")
    parent = {n: (source / n).read_bytes() for n in PARENTS}
    files = transformed(parent)
    c = files["asymmetric_native.cuh"].decode()
    assert "int SourceMask=-1,bool PlanePair=false,int ActiveAxes=-1> struct Environment" in c
    assert "UniformE,-1,false,ActiveAxes>env" in c
    assert "UniformE,SourceMask,PlanePair>env" in c
    marker = "// Private source-only storage schedule; source-bound"
    old = parent["asymmetric_native.cuh"].decode()
    assert c[c.index(marker):] == old[old.index(marker):]
    assert files["native_moving.h"] == parent["native_moving.h"]
    assert files["asymmetric_native_cell_generated.cuh"] == parent["asymmetric_native_cell_generated.cuh"]
