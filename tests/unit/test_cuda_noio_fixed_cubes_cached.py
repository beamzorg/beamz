"""Two common-shape AOT entries preserve independently captured profile bits."""

import shutil
from pathlib import Path

import pytest

from scripts.patch_cuda_noio_fixed_cubes_cached import PARENT, clone_body, helper, original_body, transformed
from scripts.prove_cuda_noio_fixed_cubes_cached import validate


SOURCE = Path(".cache/cuda-trace-five-hour-2026-11-13-1059/source-uniform-profiles-v1/source")


def parent():
    if not SOURCE.exists():
        pytest.skip("Exact current source-uniform/profile-cache best required")
    return {n: (SOURCE / "src" / n).read_bytes() for n in PARENT}


def test_unknown_source_cannot_inherit_best_capture_and_profile_cache_binding():
    with pytest.raises(ValueError, match="Exact current"):
        transformed({n: b"wrong" for n in PARENT})


def test_fixed_environment_inherits_actual_profile_pointer_and_reads_independent_cache_indices():
    text = helper(parent()["asymmetric_native.cuh"].decode())
    assert "Environment<1,false,true,1,true,-1,false,ActiveAxes,true>" in text
    assert "using Base::profiles" in text
    assert "Base{physics,facts,captured_profiles}" in text
    assert "ReadIndependentProfileCache(profiles,index,kind,Packed(index,z,y,x))" in text
    assert "N==512||N==564" in text
    assert "p.physical_n" not in text and "psi.row" not in text and "psi.plane" not in text


def test_original_best_kernels_capture_host_and_cached_fallback_stay_identical():
    before = parent()
    after, proof = transformed(before)
    old, new = before["asymmetric_native.cuh"].decode(), after["asymmetric_native.cuh"].decode()
    assert original_body(new) == original_body(old)
    for n in (512, 564):
        assert f"AsymmetricNoIOFixedCube<Depth,{n}><<<uniform_count,1024,FixedCubeSharedBytes,stream>>>" in new
    assert "AsymmetricNoIOUniform1024<Depth><<<uniform_count,1024,ProfileCachedSharedBytes,stream>>>" in new
    assert proof["expected_moving_kernels"] == 66
    assert proof["fixed_dynamic_shared_bytes"] == 55204
    assert proof["private_capture_profile_immutability_binding"]["profile_buffers_never_bound_as_outputs"]
    assert proof["private_capture_profile_immutability_binding"]["constructor_hash"] == PARENT["native_extension.cc"]
    for name in PARENT:
        if name != "asymmetric_native.cuh":
            assert after[name] == before[name]


def test_staging_precedes_existing_barrier_and_has_no_extra_barrier_sites():
    old = parent()["asymmetric_native.cuh"].decode()
    body = clone_body(old)
    assert "UniformE=true,CachedProfiles=true" in body
    stage = body.index("if constexpr(CachedProfiles)StageIndependentCpmlProfiles(p,profiles);")
    barrier = body.index("__syncthreads();", stage)
    first_read = body.index("FixedCubeCachedEnvironment<N,ActiveAxes>env")
    assert stage < barrier < first_read
    assert body.count("__syncthreads()") == original_body(old).count("__syncthreads()")


def test_actual_CPP_both_shapes_keep_profiles_addresses_state_and_physics_bits(tmp_path):
    if not shutil.which("c++"):
        pytest.skip("Host C++ required")
    after, _ = transformed(parent())
    source = tmp_path / "source" / "src"
    source.mkdir(parents=True)
    for name, value in after.items():
        (source / name).write_bytes(value)
    counts = validate(source.parent, tmp_path / "oracle")["actual_CXX"]
    assert counts["errors"] == 0
    assert counts["Packed_PsiOffset_checks"] == 210912
    assert counts["independent_profile_bit_checks"] == 365040
    assert counts["state_load_store_bit_checks"] == 5760
    assert counts["staged_independent_profile_bit_checks"] == 6912
    assert counts["actual_cache_pointer_route_checks"] == 288
    assert counts["actual_shape_selector_checks"] == 18
    assert counts["base_dispatch_metadata_fault_negatives"] > 0
    assert counts["wrong_shape_negatives"] > 0
