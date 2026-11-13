"""Common CPML point addresses retain exact staggered terms and profile bits."""

import shutil
from pathlib import Path

import pytest

from scripts.patch_cuda_cpml_point_context import E_LOOP, H_LOOP, PARENT, derive, helper, transformed
from scripts.validate_cuda_cpml_point_context import validate


SOURCE = Path(".cache/cuda-two-hour-2026-11-13-0819/source-plane-axis-cg-v1/source")


def parent():
    if not SOURCE.exists():
        pytest.skip("Frozen selected52865 source required")
    return {n: (SOURCE / "src" / n).read_bytes() for n in PARENT}


def test_extra_term_classes_are_derived_from_actual_typed_target_supports():
    result = derive()
    assert [(r["phase"], r["axis"], r["extra_term"]) for r in result] == [
        (0, 0, 1), (0, 1, 0), (0, 2, 3), (1, 0, 2), (1, 1, 5), (1, 2, 4)]
    assert all(len(r["terms"]) == 2 and sum(t["extra_delta"] for t in r["terms"]) == 1 for r in result)
    assert "Phase==0?(Axis==0?1:Axis==1?0:3):(Axis==0?2:Axis==1?5:4)" in helper()


def test_invalid_points_do_not_read_memory_during_integer_context_construction():
    context = helper()
    begin = context.index("CubicCpmlPointContext(const Base&")
    end = context.index("bool Eligible(", begin)
    constructor = context[begin:end]
    assert "point_active=coordinate<12||coordinate>=size-12" in constructor
    assert not any(x in constructor for x in ("__ldg", "input[", "output[", "Coefficient(", "LoadLocal("))


def test_unknown_parent_or_changed_native_support_admission_cannot_inherit_proof():
    with pytest.raises(ValueError, match="Exact selected52865"):
        transformed({n: b"wrong" for n in PARENT})


def test_cubic_onehot_selector_preserves_all_original_fallback_loops_and_barriers():
    before = parent()
    after, proof = transformed(before)
    text, old = after["asymmetric_native.cuh"].decode(), before["asymmetric_native.cuh"].decode()
    assert "p.physical_n[0]==p.physical_n[1]&&p.physical_n[1]==p.physical_n[2]" in text
    assert text.count(H_LOOP) == text.count(E_LOOP) == 1
    assert "if constexpr(PointContext)" in text
    assert "CubicCpmlPointContext<0,PointAxis,decltype(env)>point_env(env,z,y,x);" in text
    assert "CubicCpmlPointContext<1,PointAxis,decltype(env)>point_env(env,z,y,x);" in text
    assert text.count("__syncthreads()") == old.count("__syncthreads()")
    assert proof["exact_source_inverse"] and proof["shared_bytes_unchanged"] == 51748
    for name in PARENT:
        if name != "asymmetric_native.cuh":
            assert after[name] == before[name]


def test_actual_context_vs_original_CXX_methods_preserves_offsets_guards_profiles_and_state_bits(tmp_path):
    if not shutil.which("c++"):
        pytest.skip("Host C++ compiler required")
    before = parent()
    after, _ = transformed(before)
    source = tmp_path / "source" / "src"
    source.mkdir(parents=True)
    for name, data in after.items():
        (source / name).write_bytes(data)
    proof = validate(source.parent, tmp_path / "oracle")
    assert proof["errors"] == 0
    assert proof["address_checks"] == 67392
    assert proof["active_guard_checks"] == 404352
    assert proof["profile_bit_checks"] == 717984
    assert proof["state_load_store_checks"] == 18144
    assert proof["signed_zero_checks"] > 0
    assert proof["wrong_extra_term_counterexamples"] > 0
    assert proof["cross_phase_profile_counterexamples"] > 0
