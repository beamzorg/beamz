"""Actual512x8 owner mapping and cached/fallback launch consistency."""

from pathlib import Path

import pytest

from scripts.patch_cuda_traced_lifetime_private_psi512 import transform
from scripts.prove_cuda_traced_lifetime_private_psi512 import prove_owner_mapping, rebound_oracle


def parents():
    directory = Path(".cache/cuda-two-hour-2026-11-13-0819/source-plane-axis-cg-v1/source/src")
    if not directory.is_dir():
        pytest.skip("Frozen research source required")
    return tuple((directory / name).read_text() for name in ("asymmetric_native.cuh", "native_extension.cc", "native_moving.h"))


def test_512threads_eightZslots_bijectively_cover_cache_and_private_state():
    result = prove_owner_mapping()
    assert result["point_owner_bijection"] == 4096
    assert result["private_history_floats_per_thread"] == 32
    assert all(axis["actual_history_state_owners"] == 16384 for axis in result["history_state_owners_per_axis"])


def test_inheriting1024_Zstep_with512threads_would_duplicate_or_omit_points():
    with pytest.raises(AssertionError, match="inverse|duplicates"):
        prove_owner_mapping(zstep=2)


@pytest.mark.parametrize("first,seed", [(1, 7), (4, 29), (13, 7)])
def test_512field_oracle_preserves_actual_read_versions_under_adversarial_warps(first, seed):
    simulate, _ = rebound_oracle()
    result = simulate(first=first, steps=2, seed=seed)
    assert result["actual_field_and_clock_checks"] > 200000
    assert result["loop_barriers"] == 4


def test_same_phase_512word_hazards_and_future_ghost_retirement():
    _, hazards = rebound_oracle()
    result = hazards()
    assert len(result) == 16
    assert all(case["same_owner_ordered_conflicts"] > 0 for case in result)
    assert all(case["old_publication_future_lowX_H_writer_overlap"] > 0 for case in result)


def test_all_new_cached_entry_branches_use512threads_even_without_history():
    consumer, host, _, proof = transform(*parents())
    begin = consumer.index("void AsymmetricNoIOPrivatePsi512(")
    end = consumer.index("void AsymmetricObserverUniform1024(", begin)
    entry = consumer[begin:end]
    assert "case 2:PermanentFamilyBody<Depth,1,false,true,1,true,1,true>" in entry
    assert "default:PermanentFamilyBody<Depth,1,false,true,1,true,-1,true>" in entry
    assert "constexpr int T=CachePsi?512:" in consumer
    assert "constexpr bool CacheState=CachePsi&&(ActiveAxes==1||ActiveAxes==2||ActiveAxes==4);" in consumer
    assert "AsymmetricNoIOPrivatePsi512<Depth><<<uniform_count,512," in consumer
    assert "AsymmetricNoIOUniform1024<Depth><<<uniform_count,1024," in consumer
    assert "__launch_bounds__(512,1) void AsymmetricNoIOPrivatePsi512(" in consumer
    assert proof["private_psi_floats_per_thread"] == 32
    assert 'result["moving_private_psi_floats_per_thread"]=32;' in host
    assert 'module.attr("moving_private_psi_threads")=512;' in host


def test_512_selector_off_is_host_only_so_causal_device_control_is_possible():
    a = transform(*parents(), enabled=True)
    b = transform(*parents(), enabled=False)
    assert a[0] == b[0] and a[2] == b[2]
    assert a[1].replace("const bool BeamzResearchPrivatePsiEnabled=true;", "const bool BeamzResearchPrivatePsiEnabled=false;", 1) == b[1]
