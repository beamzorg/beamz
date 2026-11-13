"""Proper cyclic relabeling preserves ordered rounded graphs, not gather order."""

import itertools

import pytest

from scripts.cuda_space_time_ir import IllegalSchedule, yee_operator
from scripts.prove_cuda_cyclic_axis_equivariance import (
    AxisMap,
    SourceBatch,
    cyclic,
    dft_reorder_witness,
    rotate_operator,
    signature,
    source_remaps,
    supports,
)

AXES = [a for size in range(4) for a in itertools.combinations(range(3), size)]


@pytest.mark.parametrize("steps", range(3))
@pytest.mark.parametrize("precision", ["float32", "bfloat16"])
@pytest.mark.parametrize("axes", AXES)
def test_ordered_graph_matches_independent_frontend_retrace(steps, precision, axes):
    rotation = cyclic(steps)
    before = yee_operator(cpml_axes=axes, precision=precision)
    after = rotate_operator(before, rotation)
    expected = yee_operator(cpml_axes=tuple(sorted(rotation.zyx[a] for a in axes)), precision=precision)
    assert signature(after) == signature(expected)


@pytest.mark.parametrize("steps", range(3))
def test_plain_yee_supports_and_high_low_packed_coordinates(steps):
    before = yee_operator()
    assert signature(rotate_operator(before, cyclic(steps))) == signature(before)
    assert supports(cyclic(steps)) > 500


@pytest.mark.parametrize("xyz", [(1, 0, 2), (0, 2, 1), (2, 1, 0)])
def test_odd_permutation_cannot_inherit_signless_cyclic_proof(xyz):
    with pytest.raises(IllegalSchedule, match="orientation-preserving"):
        rotate_operator(yee_operator(cpml_axes=(0, 1, 2)), AxisMap(xyz))


@pytest.mark.parametrize("xyz", [(0, 0, 2), (True, 1, 2), (0, 1), (0, 1, 3)])
def test_invalid_axis_maps_fail_closed(xyz):
    with pytest.raises(IllegalSchedule):
        AxisMap(xyz)


def test_three_rotations_restore_every_field_state_profile_and_ordered_node():
    before = yee_operator(cpml_axes=(0, 1, 2), precision="bfloat16")
    after = before
    for _ in range(3):
        after = rotate_operator(after, cyclic(1))
    assert signature(after) == signature(before)
    for step in range(3):
        assert [cyclic(step).term(t) for t in range(6)] == [(t + 2 * step) % 6 for t in range(6)]


def test_source_plane_worldX_becomes_internalZ_without_changing_clock_or_source_order():
    rotation = cyclic(2)
    before = SourceBatch(5, ((12, 12, 104), (12, 12, 104)), (488, 488, 1))
    after = before.rotated(rotation)
    assert after.group == 4 and after.starts == ((104, 12, 12),) * 2
    assert after.slab_zyx == (1, 488, 488)
    assert rotation.field("Hz") == "Hy" and rotation.field("Ey") == "Ex"
    assert rotation.presence(20) == 10
    proof = source_remaps()
    assert proof["typed_batches"] == 108 and proof["active_source_accesses"] > 0


def test_unknown_coordinate_semantics_must_not_silently_remain_world_labeled():
    with pytest.raises(IllegalSchedule, match="Unknown semantic"):
        cyclic(2).name("source_unmapped_tensor_axis")


def test_reordering_internal_keys_can_change_a_normalized_weighted_gather():
    proof = dft_reorder_witness()
    assert sum(proof["weights"]) == 1
    assert proof["original_fma_gather_result"] == 3.25
    assert proof["reordered_fma_gather_result"] == 0.0
