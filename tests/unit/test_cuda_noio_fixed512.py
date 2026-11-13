"""Exact512 geometry changes addressing facts without baking physics values."""

import shutil
from pathlib import Path

import pytest

from scripts.patch_cuda_noio_fixed512 import PARENT, clone_body, derive_strides, helper, transformed
from scripts.patch_cuda_noio_mux import original_body
from scripts.prove_cuda_noio_fixed512 import validate


SOURCE = Path(".cache/cuda-two-hour-2026-11-13-0819/source-plane-axis-cg-v1/source")


def parent():
    if not SOURCE.exists():
        pytest.skip("Frozen selected52865 source required")
    return {n: (SOURCE / "src" / n).read_bytes() for n in PARENT}


def test_real_typed_target_supports_derive_all_twelve_distinct_packed_strides():
    data = derive_strides()
    slots = data["slots"]
    assert len(slots) == 12
    assert slots[1]["packed_shape"] == [24, 512, 513]
    assert slots[2]["packed_shape"] == [24, 513, 512]
    assert slots[1]["row"] != slots[2]["row"]
    assert slots[0]["plane"] != slots[6]["plane"]


def test_only_geometry_is_fixed_and_derived_Load_Store_really_override_addressing():
    original = parent()["asymmetric_native.cuh"].decode()
    text = helper(original)
    assert "p.physical_n" not in text and "psi.row" not in text and "psi.plane" not in text
    assert "float LoadLocal(" in text and "void StoreLocal(" in text
    assert "input[PsiOffset(slot,z,y,x)]" in text
    assert "output[PsiOffset(slot,z,y,x)]=value" in text
    assert "p.phase[phase].inv" in text
    assert "return Base::Coefficient(kind,index,z,y,x)" in text
    assert "psi.a+packed" in text and "psi.b+packed" in text and "psi.inv+packed" in text


def test_original_entries_and_bodies_remain_and_exact_host_shape_gate_selects_two_new_entries():
    before = parent()
    after, proof = transformed(before)
    old, new = before["asymmetric_native.cuh"].decode(), after["asymmetric_native.cuh"].decode()
    assert original_body(new) == original_body(old)
    body = clone_body(old)
    assert body.count("__syncthreads()") == original_body(old).count("__syncthreads()")
    assert "AsymmetricNoIOUniform1024<Depth><<<uniform_count,1024,SharedBytes,stream>>>" in new
    assert "AsymmetricNoIOFixed512<Depth><<<uniform_count,1024,SharedBytes,stream>>>" in new
    assert "return p.physical_n[0]==512&&p.physical_n[1]==512&&p.physical_n[2]==512" in new
    assert proof["expected_moving_kernels"] == 64
    assert proof["exact_source_inverse"]
    for name in PARENT:
        if name != "asymmetric_native.cuh":
            assert after[name] == before[name]


def test_unknown_source_cannot_inherit_fixed_geometry_admission():
    with pytest.raises(ValueError, match="Exact selected52865"):
        transformed({n: b"wrong" for n in PARENT})


def test_actual_CPP_methods_preserve_bounds_strides_profiles_state_and_ordered_FP(tmp_path):
    if not shutil.which("c++"):
        pytest.skip("Host C++ required")
    after, _ = transformed(parent())
    source = tmp_path / "source" / "src"
    source.mkdir(parents=True)
    for name, value in after.items():
        (source / name).write_bytes(value)
    report = validate(source.parent, tmp_path / "oracle")
    counts = report["actual_CXX"]
    assert counts["errors"] == 0
    assert counts["Packed_PsiOffset_checks"] == 105456
    assert counts["independent_profile_bit_checks"] == 182520
    assert counts["ordered_Difference_bit_checks"] == 20032
    assert counts["state_load_store_bit_checks"] == 2880
    assert counts["base_dispatch_metadata_fault_negatives"] > 0
    assert counts["wrong_shape_negatives"] > 0
    assert counts["cross_phase_profile_negatives"] > 0
    assert counts["actual_exact512_selector_checks"] == 8
