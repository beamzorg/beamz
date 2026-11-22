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

    try:
        from jax import enable_x64
    except ImportError:
        from jax.experimental import enable_x64

    with enable_x64(False):
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


def test_plane_gather_width_is_shared_across_batches():
    dx = 0.125e-6
    z = np.arange(1, 131, dtype=np.float64)
    z[-1] += 0.25
    y = np.arange(1, 139, dtype=np.float64) + 0.5

    def gather(rows):
        return _component_plane_plan_3d(
            "Ex",
            normal_axis="x",
            plane_position=2.5 * dx,
            coordinates=(rows * dx, y * dx),
            resolution=dx,
            grid_shape=(132, 140, 8),
            field_shape=(133, 141, 8),
        )

    indices, weights = gather(z)
    assert indices.shape == weights.shape == (len(z) * len(y), 4)
    assert np.count_nonzero(weights[0]) == 2
    assert np.count_nonzero(weights[-1]) == 4
    # Place the widest row in the first stripe. All indices, including padded
    # zero-weight lanes, must be independent of the stripe in which it occurs.
    for forward, backward in zip((indices, weights), gather(z[::-1]), strict=True):
        np.testing.assert_array_equal(
            forward.reshape(len(z), len(y), 4),
            backward.reshape(len(z), len(y), 4)[::-1],
        )


def test_native_graph_rejects_wide_global_monitor_indices():
    from beamz.simulation.cuda.runtime import pack_dft_monitors

    monitor = SimpleNamespace(dft_flat_idx=(np.array([[2**31]], dtype=np.int64),))
    with pytest.raises(ValueError, match="indices exceed int32"):
        pack_dft_monitors((monitor,))


def test_temporal_shard_remaps_global_64_bit_dft_before_narrowing():
    from beamz.devices.monitors.compiler import CompiledMonitorSpec
    from beamz.simulation.cuda.engine._capabilities import _Unsupported
    from beamz.simulation.cuda.runtime.native_sharding import (
        _FIELD_ORDER,
        _local_monitors,
    )

    global_shape = (8192, 512, 1024)
    coordinate = (6145, 256, 512)
    index = np.array(
        [
            [
                np.ravel_multi_index(coordinate, global_shape),
                np.ravel_multi_index((3072, 256, 512), global_shape),
            ]
        ],
        dtype=np.int64,
    )
    assert index[0, 0] > 2**31
    spec = CompiledMonitorSpec(
        name="seam",
        monitor_index=0,
        record_interval=1,
        accumulate_power=False,
        power_scale=1.0,
        freq_count=1,
        dft_point_count=1,
        dft_component_mask=np.ones(6),
        dft_flat_idx=(index,) * 6,
        dft_weights=(np.ones((1, 2), np.float32),) * 6,
    )
    logical = dict.fromkeys(_FIELD_ORDER, global_shape)
    local_shape = (2048 + 78, 512, 1024)
    local = dict.fromkeys(_FIELD_ORDER, local_shape)
    (part,), _ = _local_monitors(
        (spec,), logical, local, 0, 6144, 8192, origin=6144 - 78
    )
    expected = np.ravel_multi_index((79, 256, 512), local_shape)
    for flat in part.dft_flat_idx:
        assert flat.dtype == np.int32
        np.testing.assert_array_equal(flat, [[expected, -1]])
    (clock,), _ = _local_monitors((spec,), logical, local, 0, 0, 2048)
    for flat, weights in zip(clock.dft_flat_idx, clock.dft_weights, strict=True):
        np.testing.assert_array_equal(flat, [[0, -1]])
        np.testing.assert_array_equal(weights, [[0.0, 0.0]])
    with pytest.raises(_Unsupported, match="local CUDA monitor index"):
        _local_monitors((spec,), logical, logical, 0, 0, 8192)
