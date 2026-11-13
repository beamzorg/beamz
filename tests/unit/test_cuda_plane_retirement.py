"""Actual traced Yee field/version lifetimes in the shared-plane schedule."""

import pytest

from scripts.prove_cuda_plane_retirement import (
    LifetimeError,
    dependencies,
    end_fusion_certificate,
    last_uses,
    late_start_certificate,
    prove,
    prove_final_overlap,
    prove_stage_reuse,
    retirement_certificate,
)


def test_actual_operator_has_only_tangential_z_neighbor_components():
    reads = dependencies()
    h_z = {r.field for name, items in reads.items() if name[0] == "H"
           for r in items if r.offset[0]}
    e_z = {r.field for name, items in reads.items() if name[0] == "E"
           for r in items if r.offset[0]}
    assert h_z == {"Ex", "Ey"}
    assert e_z == {"Hx", "Hy"}
    assert all(r.version == 0 for name, items in reads.items() if name[0] == "H" for r in items)
    assert all(r.version == 1 for name, items in reads.items() if name[0] == "E"
               for r in items if r.field[0] == "H")


def test_component_last_use_and_reserved_last_plane_are_distinct():
    uses = last_uses(16)
    assert uses["Hx:3"] == uses["Hy:3"] == [3, 4]
    assert uses["Hz:3"] == [3]
    assert uses["Hx:15"] == uses["Hy:15"] == [0, 15]
    assert uses["Hz:15"] == [15]


def test_early_previous_plane_retirement_has_no_cross_worker_conflict():
    result = retirement_certificate((16, 16, 32))
    assert result["early_previous_write_cross_thread_conflicts"] == 0
    assert result["final_plane_cross_thread_conflicts"] > 0


def test_actual_owned_z15_late_start_requires_only_versioned_low_ghosts():
    result = late_start_certificate()
    assert result["first"] == 12 and result["required_previous_version"] == 11
    assert result["valid_low_points"] > 0
    with pytest.raises(LifetimeError, match="Late startup"):
        late_start_certificate(omit_low=True)


@pytest.mark.parametrize("early", [False, True])
def test_cache16_wraps_and_cross_warp_y_reads_with_nine_short_e_values(early):
    result = prove((16, 3, 32), steps=17, seed=39, early_previous=early)
    assert result["cross_warp_y_reads"] > 0
    assert result["private_H_floats_per_thread"] == 48
    assert result["short_E_floats_per_thread"] == 9


@pytest.mark.parametrize("negative", [
    "missing_old_e_retirement", "missing_h_publication",
    "overwrite_reserved_last", "missing_last_old",
    "missing_final_retirement", "wrong_z_wrap", "old_e_ghost",
])
def test_unsafe_interleavings_and_versions_have_counterexamples(negative):
    with pytest.raises(LifetimeError):
        prove((4, 3, 32), steps=5, seed=0, negative=negative)


def test_cross_warp_low_face_stage_reuse_exhausts_legal_orders():
    assert prove_stage_reuse()["legal_interleavings"] > 1
    with pytest.raises(LifetimeError, match="Cross-warp H-stage"):
        prove_stage_reuse(early=True)


def test_last_two_planes_can_overlap_only_until_final_retirement_barrier():
    assert prove_final_overlap()["legal_interleavings"] > 1
    with pytest.raises(LifetimeError, match="Unretired final overlap reader"):
        prove_final_overlap(early_pending=True)


def test_fused_end_ghost_partitions_and_same_thread_write_order():
    result = end_fusion_certificate()
    assert result["pure_x_packet_points"] == 225
    with pytest.raises(LifetimeError, match="clobbers a restored ghost"):
        end_fusion_certificate(restore_before_final=True)
