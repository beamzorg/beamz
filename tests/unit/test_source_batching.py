"""Regression contracts for shape-safe compiled source batching."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from beamz.devices.sources.compiler import CompiledSourceSpec, batch_slab_specs
from beamz.simulation.execute import _apply_batched_slabs, _apply_specs


def _slab(start, shape, value):
    return CompiledSourceSpec(
        component="Ex",
        timing="e",
        index=tuple(
            slice(low, low + size) for low, size in zip(start, shape, strict=True)
        ),
        coeff=jnp.full(shape, value, dtype=jnp.float32),
        waveform=jnp.ones((1,), dtype=jnp.float32),
        is_slab=True,
        slab_starts=tuple(start),
        slab_sizes=tuple(shape),
    )


def test_unequal_source_slabs_keep_exact_shape_and_boundary_location():
    small = _slab((3, 3, 3), (1, 1, 1), 1.0)
    large = _slab((0, 0, 0), (2, 2, 2), 2.0)

    batch, rest = batch_slab_specs((small, large))

    assert batch is not None
    assert batch.max_sizes == (1, 1, 1)
    assert rest == (large,)
    field = _apply_batched_slabs(
        jnp.zeros((4, 4, 4), dtype=jnp.float32),
        jnp.asarray(0, dtype=jnp.int32),
        batch,
        dense_single_slab=False,
    )
    field = _apply_specs(field, jnp.asarray(0, dtype=jnp.int32), rest)

    expected = np.zeros((4, 4, 4), dtype=np.float32)
    expected[:2, :2, :2] = 2.0
    expected[3, 3, 3] = 1.0
    np.testing.assert_array_equal(field, expected)


def test_source_batch_selects_largest_exact_shape_group_stably():
    first = _slab((0, 0, 0), (1, 2, 2), 1.0)
    second = _slab((1, 1, 1), (2, 1, 2), 2.0)
    third = _slab((2, 0, 0), (1, 2, 2), 3.0)

    batch, rest = batch_slab_specs((first, second, third))

    assert batch is not None
    assert batch.n == 2
    assert batch.max_sizes == (1, 2, 2)
    assert batch.starts_tuple == ((0, 0, 0), (2, 0, 0))
    assert rest == (second,)


@pytest.mark.parametrize("source_count", [1, 2, 3])
def test_source_slabs_preserve_overlap_and_dynamic_slice_clamping(source_count):
    specs = tuple(
        _slab((3, -1, 3), (2, 2, 2), float(index + 1)) for index in range(source_count)
    )
    batch, rest = batch_slab_specs(specs)
    assert not rest
    original = jnp.arange(64, dtype=jnp.float32).reshape((4, 4, 4))
    result = _apply_batched_slabs(
        original, jnp.asarray(0), batch, dense_single_slab=False
    )
    # dynamic_slice normalizes negative indices before clamping each origin.
    expected = np.asarray(original).copy()
    expected[2:4, 2:4, 2:4] += sum(range(1, source_count + 1))
    np.testing.assert_array_equal(result, expected)
