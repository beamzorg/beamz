import pytest

from scripts.prove_cuda_warp_z_pairs import prove


def test_every_pair_and_half_boundary_routes_the_correct_source_operand():
    result = prove()
    assert result["neighbor_routes_verified"] == 32768
    assert result["paired_Z_routes"] == 16384
    assert result["cross_half_Y_routes"] == 1024


@pytest.mark.parametrize("wrong", ["wrong_parity", "wrong_x_block"])
def test_incorrect_source_parity_or_half_frame_is_detected(wrong):
    with pytest.raises(AssertionError):
        prove(**{wrong: True})
