"""Real schedule-owner checks for unconditional warp-neighbor lowering."""

from pathlib import Path

import pytest

from scripts.patch_cuda_traced_lifetime_warp_x import (
    BODY_BEGIN,
    BODY_END,
    E_NEW,
    E_POINT,
    E_READ,
    E_REPLACEMENT,
    HELPER,
    H_NEW,
    H_POINT,
    H_READ,
    H_REPLACEMENT,
    PARENTS,
    prove_ownership,
    scope_enabled,
    traced_x_reads,
    transform,
)


def parent_files():
    directory = Path(".cache/cuda-two-hour-2026-11-13-0819/source-plane-axis-cg-v1/source/src")
    if not directory.is_dir():
        pytest.skip("Exact frozen research source required")
    return {name: (directory / name).read_bytes() for name in PARENTS}


def test_actual_typed_yee_graph_has_only_admitted_directional_reads():
    assert traced_x_reads() == {
        "H": [("Ey", (0, 0, 1)), ("Ez", (0, 0, 1))],
        "E": [("Hy", (0, 0, -1)), ("Hz", (0, 0, -1))],
    }


def test_all_actual_thread_and_slot_owners_match_shared_word_provenance():
    result = prove_ownership()
    assert result["checked_payload_owners"] == 416640
    assert result["all32_lanes_before_predicates"]


@pytest.mark.parametrize("wrong_delta", [0, 1, -1, 2, 16, 32])
def test_wrong_neighbor_lanes_fail(wrong_delta):
    with pytest.raises(ValueError, match="different field owner"):
        prove_ownership(lane_delta=wrong_delta)


def test_collective_inside_owned_predicate_is_rejected():
    with pytest.raises(ValueError, match="inactive ghost lane"):
        prove_ownership(predicate_before_collective=True)


def test_unknown_thread_mapping_is_rejected():
    with pytest.raises(ValueError, match="thread counts"):
        prove_ownership(threads=(256,))


@pytest.mark.parametrize("mode", ["h", "e", "both"])
def test_source_inverse_binds_unmodified_physics_and_all_stores_barriers(mode):
    parent = parent_files()
    changed = transform(parent, mode)
    original = parent["asymmetric_native.cuh"].decode()
    result = changed["asymmetric_native.cuh"].decode()
    restored = result.replace(HELPER, "", 1)
    for old, new in ((H_POINT, H_NEW), (H_READ, H_REPLACEMENT)) if mode == "h" else (
        ((E_POINT, E_NEW), (E_READ, E_REPLACEMENT)) if mode == "e" else
        ((H_POINT, H_NEW), (H_READ, H_REPLACEMENT), (E_POINT, E_NEW), (E_READ, E_REPLACEMENT))
    ):
        restored = restored.replace(new, old, 1)
    assert restored == original
    for name in PARENTS:
        if name != "asymmetric_native.cuh":
            assert changed[name] == parent[name]
    old_body = original[original.index(BODY_BEGIN):original.index(BODY_END)]
    new_body = result[result.index(BODY_BEGIN):result.index(BODY_END)]
    assert old_body.count("__syncthreads()") == new_body.count("__syncthreads()")
    assert new_body.count("TracedWarpX(") == (4 if mode == "both" else 2)


@pytest.mark.parametrize("file", list(PARENTS))
def test_changed_dependency_fails_closed(file):
    parent = parent_files()
    parent[file] += b"\n"
    with pytest.raises(ValueError, match="selected52865"):
        transform(parent)


def test_unknown_mode_is_rejected():
    with pytest.raises(ValueError, match="Mode"):
        transform({}, "unreviewed")


def test_noio_bulk_scope_only_admits_resource_clean_owner_classes():
    import itertools
    count = 0
    for layout, fast, iso, policy, uniform in itertools.product((1, 2), (False, True), (False, True), range(6), (False, True)):
        actual = scope_enabled("noio-bulk", layout=layout, fast=fast, isotropic_h=iso,
                               policy=policy, uniform_e=uniform)
        expected = layout == 1 and iso and (fast or policy == 1 and uniform)
        assert actual == expected
        if layout == 2 or not iso or not fast and policy == 2:
            assert not actual
        count += 1
    assert count == 96


@pytest.mark.parametrize("mode", ["h", "e", "both"])
def test_scoped_lowering_guards_collective_and_router_with_same_compile_fact(mode):
    parent = parent_files()
    changed = transform(parent, mode, "noio-bulk")
    result = changed["asymmetric_native.cuh"].decode()
    original = parent["asymmetric_native.cuh"].decode()
    assert result.count("constexpr bool TracedWarpXEnabled=") == 1
    phases = 2 if mode == "both" else 1
    assert result.count("if constexpr(TracedWarpXEnabled)") == 2 * phases
    assert result.count("__syncthreads()") == original.count("__syncthreads()")
    assert all(changed[name] == parent[name] for name in parent if name != "asymmetric_native.cuh")
