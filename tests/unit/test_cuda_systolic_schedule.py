"""Moving boxes must eliminate dependency cycles without duplicated updates."""

import pytest

from scripts.prove_cuda_systolic_schedule import prove


@pytest.mark.parametrize("cpml", [False, True])
def test_moving_boxes_own_each_update_once(cpml):
    result = prove((3, 4, 5), (2, 3, 4), 4, cpml=cpml)
    assert result["executed_over_useful_updates"] == 1
    assert result["unique_cross_box_values"] > 0
    assert result["physical_wall_reads"] > 0


def test_unskewed_boxes_have_cyclic_dependencies():
    with pytest.raises(AssertionError, match="cyclic box graph"):
        prove((3, 4, 5), (2, 3, 4), 2, shear=False)


def test_missing_face_publication_is_rejected():
    with pytest.raises(AssertionError, match="Missing face-trail"):
        prove((3, 4, 5), (2, 3, 4), 2, drop_boundary=True)
