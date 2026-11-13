"""Independent CPML profile staging preserves bits and existing visibility."""

import shutil
from pathlib import Path

import pytest

from scripts.patch_cuda_shared_profiles import PARENT, constructor_witness, transformed
from scripts.validate_cuda_shared_profiles import validate


SOURCE = Path(".cache/cuda-two-hour-2026-11-13-0819/source-plane-axis-cg-v1/source")


def parent():
    if not SOURCE.exists():
        pytest.skip("Frozen selected52865 source required")
    return {n: (SOURCE / "src" / n).read_bytes() for n in PARENT}


def test_wrong_parent_cannot_inherit_profile_readonly_or_schedule_proof():
    with pytest.raises(ValueError, match="Exact selected52865"):
        transformed({n: b"wrong" for n in PARENT})


def test_constructor_alias_or_unsynchronized_upload_mutation_fails_closed():
    before = parent()
    for old, new in (
        ("auto memory = std::make_unique<DeviceMemory>(host_.back().bytes);", "auto memory = reused_memory();"),
        ('Check(cudaStreamSynchronize(stream_.value), "finish native preparation");', "// Removed synchronization"),
    ):
        modified = dict(before)
        modified["native_extension.cc"] = before["native_extension.cc"].replace(old.encode(), new.encode(), 1)
        with pytest.raises(ValueError, match="allocation/upload/outputbinding"):
            constructor_witness(modified)


def test_staging_does_not_alias_profiles_or_add_barriers_or_change_source_bodies():
    before = parent()
    after, proof = transformed(before)
    original, result = before["asymmetric_native.cuh"].decode(), after["asymmetric_native.cuh"].decode()
    assert proof["exact_source_inverse"] and proof["all_barrier_sites_identical"]
    assert result.count("__syncthreads()") == original.count("__syncthreads()")
    assert "bool CachedProfiles=false" in result
    assert "if constexpr(CachedProfiles)StageIndependentCpmlProfiles(p,profiles);" in result
    assert "Environment<Layout,Fast,IsotropicH,Policy,UniformE,-1,false,ActiveAxes,CachedProfiles>env{p,flags,profiles};" in result
    for axis in (1, 2, 4, -1):
        assert f"UniformBody<Depth,1,false,true,1,true,{axis},true>(p,tiles);" in result
    for name in PARENT:
        if name != "asymmetric_native.cuh":
            assert after[name] == before[name]
    immutable = proof["immutable_profile_admission"]
    assert not set(immutable["profile_ABI_inputs"]) & set(immutable["mutable_ABI_inputs"])
    assert len(immutable["profile_ABI_inputs"]) == 36
    assert proof["shared_bytes_after"] - proof["shared_bytes_before"] == 36 * 24 * 4


def test_actual_cxx_staging_and_read_helpers_preserve_unequal_H_E_profiles_and_signed_zero(tmp_path):
    if not shutil.which("c++"):
        pytest.skip("Host C++ compiler required")
    before = parent()
    after, _ = transformed(before)
    source = tmp_path / "source"
    (source / "src").mkdir(parents=True)
    for name, value in after.items():
        (source / "src" / name).write_bytes(value)
    proof = validate(source, tmp_path / "oracle")
    assert proof["errors"] == 0
    assert proof["profile_bit_checks"] == proof["single_writer_checks"] == 110592
    assert proof["signed_zero_checks"] == 4608
    assert proof["cross_family_alias_counterexamples"] > 0
    assert proof["swapped_kind_counterexamples"] > 0
