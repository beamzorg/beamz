"""Fixed-scalar queue routing and participation negative controls."""

import pytest

from scripts.prove_cuda_native_scalar_queue import prove


@pytest.mark.parametrize("depth", (4, 8, 16, 64))
def test_runtime_z_loop_preserves_fixed_private_slots(depth):
    report = prove(depth)
    assert report["verified_private_operands"] == depth * 18600
    assert report["private_floats_per_thread"] == 12


@pytest.mark.parametrize("option", ("skip_excluded_rotation", "wrong_previous", "omit_ghost_shuffle"))
def test_rejects_excluded_rotation_wrong_z_or_missing_lane(option):
    with pytest.raises(AssertionError):
        prove(**{option: True})
