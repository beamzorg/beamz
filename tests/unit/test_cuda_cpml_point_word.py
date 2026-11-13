"""One-word context preserves actual cubic CPML addresses and profile bits."""

import hashlib
import shutil
from pathlib import Path

import pytest

from scripts.patch_cuda_cpml_point_context import H_LOOP, E_LOOP
from scripts.patch_cuda_cpml_point_word import PARENT_CONSUMER, helper, transformed
from scripts.validate_cuda_cpml_point_word import validate


SOURCE = Path(".cache/cuda-cyclic-axis-2026-11-13/point-context-v1/source")


def parent():
    if not SOURCE.exists():
        pytest.skip("Frozen four-field point context required")
    text = (SOURCE / "src/asymmetric_native.cuh").read_text()
    assert hashlib.sha256(text.encode()).hexdigest() == PARENT_CONSUMER
    return text


def test_one_unsigned_member_and_arithmetic_decode_keep_constructor_free_of_loads():
    text = helper()
    assert "uint32_t packed_active;" in text
    assert "static_cast<uint32_t>(packed)<<1" in text
    assert "static_cast<int32_t>(packed_active)>>1" in text
    assert not any(x in text for x in ("point_base", "point_delta", "point_active", "packed_index"))
    constructor = text[text.index("CubicCpmlPointContext(const Base&"):text.index("bool Eligible(")]
    assert not any(x in constructor for x in ("__ldg", "input[", "output[", "LoadLocal(", "Coefficient("))


def test_wrong_parent_cannot_inherit_context_proof():
    with pytest.raises(ValueError, match="Exact reviewed"):
        transformed("wrong")


def test_copy_changes_only_helper_and_cubic_range_guard():
    before = parent()
    after = transformed(before)
    assert "physical_n[0]>0&&p.physical_n[0]<=(1<<20)" in after
    assert after.count(H_LOOP) == after.count(E_LOOP) == 1
    assert after.count("__syncthreads()") == before.count("__syncthreads()")
    assert "Offset(slot,z,y,x)" in after
    assert after.count("CubicCpmlPointContext<0,PointAxis,decltype(env)>point_env(env,z,y,x);") == 1


def test_actual_CXX_methods_preserve_all_six_classes_and_negative_packed_coordinates(tmp_path):
    if not shutil.which("c++"):
        pytest.skip("Host C++ required")
    source = tmp_path / "source" / "src"
    source.mkdir(parents=True)
    (source / "asymmetric_native.cuh").write_text(transformed(parent()))
    proof = validate(source.parent, tmp_path / "oracle")
    assert proof["errors"] == 0
    assert proof["address_checks"] == 67392
    assert proof["active_guard_checks"] == 404352
    assert proof["profile_bit_checks"] == 717984
    assert proof["state_load_store_checks"] == 18144
    assert proof["packing_checks"] == 624
    assert proof["wrong_unsigned_decode_counterexamples"] > 0
    assert proof["wrong_extra_term_counterexamples"] > 0
    assert proof["cross_phase_profile_counterexamples"] > 0
