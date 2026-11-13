"""Fail closed on vacuous/stale or physically invalid shared-Epsi witnesses."""
from copy import deepcopy

import numpy as np
import pytest

from scripts.validate_cuda_native_shared_epsi import cache_witness


def witness():
    return {
        "moving_shared_epsi_selector_enabled": True,
        "moving_shared_epsi_state_independent": True,
        "moving_shared_epsi_dispatch_active": True,
        "moving_shared_epsi_single_axis_tiles_zyx": (2, 1, 1),
        "moving_shared_epsi_active_tiles": 2,
        "moving_shared_epsi_retained_tiles": 1,
        "moving_shared_epsi_retained_witness": [0, 2, 7, 0, 1, 1, 0, 1],
    }


def fields():
    return [np.empty(shape, np.float32) for shape in
            ((32, 32, 33), (32, 33, 32), (33, 32, 32),
             (33, 33, 32), (33, 32, 33), (32, 33, 33))]


def test_actual_cpml_retained_witness_is_accepted():
    result = cache_witness(witness(), {"moving_macro_count": 2}, True, fields())
    assert result["positive_retained_path"]
    assert result["all_three_single_axes_present"]


@pytest.mark.parametrize("change", (
    "vacuous-before-compaction", "tail-with-prepared-counts", "past-depth",
    "unowned-initial-state", "inactive-cpml-point", "wrong-term-axis",
    "overlapping-state", "disabled-selector",
))
def test_vacuous_stale_and_invalid_witnesses_fail(change):
    metadata, timing = deepcopy(witness()), {"moving_macro_count": 2}
    if change == "vacuous-before-compaction":
        metadata.update(moving_shared_epsi_single_axis_tiles_zyx=(0, 0, 0),
                        moving_shared_epsi_active_tiles=0,
                        moving_shared_epsi_retained_tiles=0,
                        moving_shared_epsi_retained_witness=(-1,) * 8)
    elif change == "tail-with-prepared-counts":
        timing["moving_macro_count"] = 0
    elif change == "past-depth":
        metadata["moving_shared_epsi_retained_witness"][1] = 17
    elif change == "unowned-initial-state":
        metadata["moving_shared_epsi_retained_witness"][7] = 0
    elif change == "inactive-cpml-point":
        metadata["moving_shared_epsi_retained_witness"][3:6] = (16, 16, 16)
    elif change == "wrong-term-axis":
        metadata["moving_shared_epsi_retained_witness"][6] = 1
    elif change == "overlapping-state":
        metadata["moving_shared_epsi_state_independent"] = False
    else:
        metadata["moving_shared_epsi_selector_enabled"] = False
    with pytest.raises(AssertionError, match="nonvacuous"):
        cache_witness(metadata, timing, True, fields())
