"""Depth8 ownership/frontier legality uses the actual traced Yee/CPML reads."""

import pytest

from scripts.prove_cuda_systolic_ring import prove_ring
from scripts.prove_cuda_systolic_schedule import prove


@pytest.mark.parametrize("cpml", [False, True])
def test_eight_versions_preserve_unique_updates_and_four_rank_frontier(cpml):
    result = prove((4, 5, 6), (2, 3, 4), steps=8, cpml=cpml)
    assert result["executed_over_useful_updates"] == 1.0
    ring = prove_ring((4, 5, 6), (2, 3, 4), steps=8, cpml=cpml)
    assert ring["epochs"] == 4 and ring["checked_consumer_values"] > 0


@pytest.mark.parametrize("negative", [{"epochs": 3}, {"wrong_version": True}])
def test_depth8_still_rejects_early_epoch_reuse_and_wrong_version(negative):
    with pytest.raises(AssertionError):
        prove_ring(steps=8, cpml=True, **negative)
