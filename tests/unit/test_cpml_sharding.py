"""CPML absorption profiles must survive padding for device sharding."""

from types import SimpleNamespace

import numpy as np
import pytest

from beamz.simulation.model import CpmlPackedSlabSpec, CpmlTerm
from beamz.simulation.sharding import _lower_cpml_term


@pytest.mark.parametrize("axis", range(3))
@pytest.mark.parametrize("compact", [True, False])
def test_cpml_profiles_preserve_absorption_across_physical_slab(axis, compact):
    logical_shape = [5, 6, 7]
    logical_shape[axis] = 3
    padded_shape = [8, 8, 8]
    padded_shape[axis] = 3
    profile_shape = [1, 1, 1] if compact else logical_shape.copy()
    profile_shape[axis] = 3
    profile = np.linspace(0.1, 0.4, np.prod(profile_shape), dtype=np.float32).reshape(
        profile_shape
    )
    slab = CpmlPackedSlabSpec(axis, 2, 1, tuple(logical_shape), (5, 6, 7)[axis])
    term = CpmlTerm("Ex", axis, 1.0, -profile, 1 - profile, 1 / (1 + profile), slab)

    lowered = _lower_cpml_term(term, SimpleNamespace(padded_shapes={"Ex": (8, 8, 8)}))

    assert lowered.slab.shape == tuple(padded_shape)
    assert lowered.slab.logical_stop == slab.logical_stop
    physical_cells = tuple(slice(0, size) for size in logical_shape)
    for name, neutral in (("a", 0.0), ("b", 1.0), ("inv_kappa", 1.0)):
        expected = np.broadcast_to(getattr(term, name), logical_shape)
        actual = np.broadcast_to(np.asarray(getattr(lowered, name)), padded_shape)
        np.testing.assert_array_equal(actual[physical_cells], expected)
        if compact:
            # Keep compact coefficients cheap instead of materializing each slab.
            assert getattr(lowered, name).shape == tuple(profile_shape)
        else:
            storage_only_cells = np.ones(padded_shape, dtype=bool)
            storage_only_cells[physical_cells] = False
            np.testing.assert_array_equal(actual[storage_only_cells], neutral)
