"""Exact compactY halo footprint and phase/owner gates for512Hprivate."""

from pathlib import Path

import pytest

from scripts.patch_cuda_traced_lifetime_hybrid_y import transform
from scripts.prove_cuda_traced_lifetime_hybrid_y import NEGATIVES, Hazard, compact_key, prove_footprint, simulate


def parents():
    source = Path(".cache/cuda-two-hour-2026-11-13-0819/source-plane-axis-cg-v1/source/src")
    if not source.is_dir():
        pytest.skip("Exact frozen research source required")
    return tuple((source / name).read_text() for name in ("asymmetric_native.cuh", "native_extension.cc", "native_moving.h"))


def test_exact_Y_shared_reader_and_writer_footprints_include_lowY_ghost():
    result = prove_footprint()
    assert len(result) == 16
    assert all(case["compact_H_words"] == 6510 for case in result)
    assert all(case["lowY_ghost_words"] == 434 and case["highY_owned_words_retained"] == 0 for case in result)


def test_compact_Y_key_excludes_wrong_faces_but_accepts_actual_lowY_providers():
    assert compact_key(0, (1, 0, 1), (0, 0, 0)) == 0
    assert compact_key(2, (7, 14, 31), (0, 0, 0)) == 6509
    for point in ((0, 1, 1), (1, 15, 1), (1, 1, 0)):
        with pytest.raises(Hazard, match="out of bounds"):
            compact_key(0, point, (0, 0, 0))


@pytest.mark.parametrize("first,seed", [(1, 7), (4, 29), (13, 7)])
def test_Hprivate_Z_and_warpX_sharedY_keep_all_actual_field_versions(first, seed):
    result = simulate(first=first, steps=2, seed=seed)
    assert result["actual_field_clock_checks"] > 200000
    assert result["H_Z_private_neighbor_reads"] > 0
    assert result["H_X_warp_neighbor_reads"] > 0
    assert result["H_Y_compact_neighbor_reads"] > 0
    assert result["loop_barriers"] == 4


@pytest.mark.parametrize("negative", NEGATIVES)
def test_corrupt_actual_neighbor_storage_or_phase_order_is_rejected(negative):
    with pytest.raises(Hazard):
        simulate(first=1, steps=2, seed=7, negative=negative)


def test_generated_body_and_launch_match_owner_storage_and_retirement_proofs():
    original, original_host, original_header = parents()
    consumer, host, header, proof = transform(original, original_host, original_header)
    start, finish = consumer.index("void HybridYBody("), consumer.index("void UniformBody(")
    body = consumer[start:finish]
    assert "constexpr int T=512,Slots=8,ZStep=1;" in body
    assert "float h[3][Slots];" in body and "float e[" not in body
    assert "if((c==0||c==2)&&dy<BY)" in body
    assert "if(sy==hy&&zs!=hz&&sx!=hx)" in body
    assert "if(dz)return h[field][(j+7)&7]" in body
    assert body.index("HybridWarpX32(h[1][j]") < body.index("if(zs!=ehz&&sy!=ehy&&sx!=ehx)")
    assert body.index("const float hv=h[c][j]") < body.index("// PF2 publishes")
    assert "AsymmetricNoIOHybridY512<Depth><<<uniform_count,512,HybridYSharedBytes" in consumer
    assert "AsymmetricNoIOUniform1024<Depth><<<uniform_count,1024,SharedBytes" in consumer
    assert "append_bytes(&hybrid_y_selector,sizeof(bool))" in host
    assert "BeamzResearchHybridYEnabled" in header
    assert proof["shared_bytes"] == 77792


def test_original_uniform_body_and_all_source_schedules_stay_byte_identical():
    values = parents()
    original = values[0]
    consumer = transform(*values)[0]
    begin = "template<int Depth,int Layout,bool Fast,bool IsotropicH,int Policy,bool UniformE,int ActiveAxes=-1>\n__device__ __forceinline__ void UniformBody"
    end = "template<int Depth,int Layout,bool Fast,bool IsotropicH,int Policy=(Fast?0:5)>\n__global__"
    assert original[original.index(begin):original.index(end)] == consumer[consumer.index(begin):consumer.index(end)]
    source = "// Private source-only storage schedule; source-bound"
    assert original[original.index(source):original.index("inline bool MakeSource")] == consumer[consumer.index(source):consumer.index("inline bool MakeSource")]


def test_selector_off_is_a_CXX_only_causal_control():
    a = transform(*parents(), enabled=True)
    b = transform(*parents(), enabled=False)
    assert a[0] == b[0] and a[2] == b[2]
    assert a[1].replace("const bool BeamzResearchHybridYEnabled=true;", "const bool BeamzResearchHybridYEnabled=false;", 1) == b[1]
