"""Canonical route, shared-word and resource-contract tests for warp frontiers."""

import hashlib
from pathlib import Path

import pytest

from scripts.cuda_warp_frontier_layout import descriptor, emit_header, widths
from scripts.patch_cuda_warp_frontier_512 import transformed
from scripts.patch_cuda_warp_frontier_phase_points import transformed as point_clocked
from scripts.patch_cuda_warp_frontier_onehot import transformed as onehot
from scripts.patch_cuda_warp_frontier_component_points import transformed as component_points, peers
from scripts.prove_cuda_warp_frontier import Hazard, prove_layout, simulate


@pytest.mark.parametrize("warp_y,words", [(2, 8960), (4, 5888), (8, 5888)])
def test_actual_remote_reads_have_unique_dense_shared_producers(warp_y, words):
    proof = prove_layout(warp_y)
    assert proof["bijection_words"] == words
    assert len(proof["actual_route_cases"]) == 16
    assert all(c["unique_shared_producers"] == words for c in proof["actual_route_cases"])


@pytest.mark.parametrize("warp_y", [2, 4, 8])
@pytest.mark.parametrize("first,seed", [(1, 7), (13, 29)])
def test_all_field_versions_match_with_adversarial_warp_order(warp_y, first, seed):
    proof = simulate(first=first, steps=4, seed=seed, warp_y=warp_y)
    assert proof["actual_checks"] > 400000
    assert all(n > 0 for n in proof["read_routes"].values())


@pytest.mark.parametrize("warp_y", [2, 4, 8])
@pytest.mark.parametrize("mutation", ["alias-family-faces", "early-E-shared-import",
    "wrong-Y-shuffle", "missing-PF1", "missing-PF2", "single-shared-clock"])
def test_unsafe_communication_mutations_are_rejected(warp_y, mutation):
    with pytest.raises(Hazard):
        simulate(first=1, steps=4, seed=7, negative=mutation, warp_y=warp_y)


def frozen_parent():
    parent = Path(".cache/cuda-two-hour-2026-11-13-0819/source-plane-axis-cg-v1/source/src/asymmetric_native.cuh")
    if not parent.exists():
        pytest.skip("Private frozen research parent unavailable")
    return parent.read_text()


def test_default_generator_preserves_original_hardware_candidate_bytes():
    code, header, report = transformed(frozen_parent())
    assert hashlib.sha256(code.encode()).hexdigest() == "e9c93bf3b244586f847608a3cfb1d2bbf91e5b6a663bf903f3cd31f075d27ed8"
    assert hashlib.sha256(header.encode()).hexdigest() == "44bd5639487e25657a29aef11bd832eb1e2d518632dfd8a0c085a9f3fd3eeb0c"
    assert report["paired_source_bodies_byte_identical"]


@pytest.mark.parametrize("warp_y,shared", [(2, 38440), (4, 26152), (8, 26152)])
def test_owner_geometry_and_launch_contract_are_generated_together(warp_y, shared):
    code, header, report = transformed(frozen_parent(), warp_y=warp_y)
    assert report["shared_bytes"] == shared
    assert f"constexpr int WarpFrontierWords={descriptor(warp_y)['shared_words']};" in header
    assert "__launch_bounds__(512,1) void AsymmetricNoIOWarpFrontier512" in code
    assert "<<<uniform_count,512,PermanentFamilySharedBytes,stream>>>" in code


@pytest.mark.parametrize("warp_y", [2, 4, 8])
def test_point_clock_intervals_bind_literal_register_slots(warp_y):
    code, header, report = point_clocked(frozen_parent(), warp_y=warp_y)
    begin = code.index("void PermanentFamilyBody")
    end = code.index("void UniformBody", begin)
    body = code[begin:end]
    assert body.count("const int step=*h_step") == 8
    assert body.count("const int step=*e_step") == 8
    assert all(body.count(f"constexpr int j={j};") == 2 for j in range(8))
    assert report["phase_clock_loads_per_thread_step"] == 16
    assert header == emit_header(warp_y)


def test_unknown_warp_geometry_fails_closed():
    with pytest.raises(ValueError):
        widths(3)


@pytest.mark.parametrize("warp_y", [2, 4, 8])
def test_onehot_branch_retains_canonical_default_and_max_shared_allocation(warp_y):
    code, header, report = onehot(frozen_parent(), warp_y=warp_y)
    assert report["launch_shared_bytes"] == 51748
    assert code.count("default:AliasedFamilyBody<Depth,1,false,true,1,false>(p,tiles);return;") == 1
    assert "<<<uniform_count,512,WarpFrontierLaunchSharedBytes,stream>>>" in code


def test_component_peer_frontier_is_derived_and_only_reads_opposite_family():
    expected = [{"ez_y"}, {"ez_x"}, {"ex_y", "ey_x"},
                {"hz_y"}, {"hz_x"}, {"hx_y", "hy_x"}]
    actual = [peers(phase, c) for phase in (0, 1) for c in range(3)]
    assert actual == expected
    for phase in (0, 1):
        family = "e" if phase == 0 else "h"
        assert all(name.startswith(family) for c in range(3) for name in peers(phase, c))
        assert len(set.union(*(peers(phase, c) for c in range(3)))) == 4


@pytest.mark.parametrize("warp_y", [2, 4, 8])
def test_component_schedule_preserves_fullmask_count_and_scalar_update_order(warp_y):
    code, header, report = component_points(frozen_parent(), warp_y=warp_y)
    begin = code.index("void PermanentFamilyBody")
    end = code.index("void UniformBody", begin)
    body = code[begin:end]
    assert body.count("__shfl_sync(0xffffffffu,") == 64
    assert body.count("__syncwarp") == 0
    assert report["peak_peer_floats_source_level"] == 2
    for phase in (0, 1):
        calls = [body.index(f"AdvanceNativeCell<{phase},{c},false>") for c in range(3)]
        assert calls == sorted(calls)
        assert all(body.count(f"AdvanceNativeCell<{phase},{c},false>") == 8 for c in range(3))
