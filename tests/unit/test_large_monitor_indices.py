"""Sparse monitor metadata must stay correct beyond global int32 cell counts."""

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from beamz.devices.monitors.compiler import _sampling_indices
from beamz.lattice import _component_plane_plan_3d, sampling_coordinates
from beamz.simulation.observe import _sample_components


def test_15b_plane_plan_and_jitted_sampling_without_large_field_allocation():
    dx = 80e-9
    shape = (10008, 1251, 1200)
    indices, weights = _component_plane_plan_3d(
        "Ex",
        normal_axis="x",
        plane_position=360.5 * dx,
        coordinates=(np.array([5000 * dx]), np.array([625 * dx])),
        resolution=dx,
        grid_shape=(10000, 1250, 1200),
        field_shape=shape,
    )
    assert indices.dtype == np.int64
    assert indices.min() > np.iinfo(np.int32).max
    expected = np.ravel_multi_index((5000, 625, 360), shape)
    np.testing.assert_array_equal(indices[weights > 0], expected)

    # A virtual analytic field tests the actual sampling path without allocating
    # 60 GB. A wrapped global index would sample a different z coordinate.
    class AnalyticField:
        dtype = jnp.float32

        def __init__(self):
            self.shape = shape

        def __getitem__(self, coordinates):
            z, y, x = coordinates
            return z.astype(jnp.float32) * 10 + y + x / 1000

    with jax.enable_x64(False):
        metadata = _sampling_indices(indices)
        assert isinstance(metadata, np.ndarray)
        assert not metadata.flags.writeable
        coordinates = jax.jit(lambda: sampling_coordinates(metadata, shape))()
        for actual, coordinate in zip(coordinates, (5000, 625, 360), strict=True):
            np.testing.assert_array_equal(actual[weights > 0], coordinate)
            assert actual.dtype == jnp.int32
        field = AnalyticField()
        sampled = jax.jit(
            lambda: _sample_components((field,), (metadata,), (jnp.asarray(weights),))
        )()
        np.testing.assert_allclose(sampled, [[50625.36]], rtol=0, atol=0.004)


def test_native_graph_rejects_wide_global_monitor_indices():
    from beamz.simulation.cuda.runtime import pack_dft_monitors

    monitor = SimpleNamespace(dft_flat_idx=(np.array([[2**31]], dtype=np.int64),))
    with pytest.raises(ValueError, match="indices exceed int32"):
        pack_dft_monitors((monitor,))
