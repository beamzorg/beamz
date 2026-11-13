"""Both packet families cross warps; publication and next reuse are distinct."""

import pytest

from scripts.prove_cuda_plane_stage_reuse import physical_witness, prove


def test_actual_225_point_stage_contains_a_two_warp_exchange():
    witness = physical_witness()
    assert [(w["producer_warp"], w["consumer_warp"]) for w in witness] == [(0, 1), (1, 0)]


def test_all_barrier_preserving_stage_interleavings_pass():
    assert prove()["legal_interleavings"] == 16


@pytest.mark.parametrize("option", ["omit_h_retirement", "omit_e_publication", "omit_next_retirement"])
def test_each_missing_stage_lifetime_edge_has_a_legal_counterexample(option):
    with pytest.raises(AssertionError, match="Cross-warp packet lifetime failure"):
        prove(**{option: True})
