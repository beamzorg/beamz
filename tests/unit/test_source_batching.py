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


@pytest.mark.contract
@pytest.mark.parametrize(
    "index,coeff,slab,starts,sizes",
    [
        ((-1, slice(1, 3)), [2, 3], True, (3, 1), (1, 2)),
        ((slice(None), slice(None, None, 2)), [[2, 3]] * 4, False, None, None),
        ((slice(None), slice(None)), 2, False, None, None),
        ((np.array([0, 2]), 1), [2, 3], False, None, None),
    ],
)
@pytest.mark.parametrize("complex_fields", [False, True])
def test_injection_packing_preserves_coefficients_and_provenance(
    index, coeff, slab, starts, sizes, complex_fields
):
    from beamz.devices.sources.compiler import (
        InjectionPlanEntry,
        TemporalWaveform,
        _compile_injection_plan,
    )

    values = np.asarray(coeff) * (1 + 2j)
    waveform = TemporalWaveform(jnp.array([1.0, 2.0]), jnp.array([3.0, 4.0]))
    entry = InjectionPlanEntry(
        component="Ex",
        timing="e",
        index=index,
        values=values,
        waveform=waveform,
        target_shape=(4, 4),
        launched_power=2.5,
    )
    specs = _compile_injection_plan(
        (entry,), source_index=7, complex_fields=complex_fields
    )
    expected_parts = [values] if complex_fields else [values.real, -values.imag]
    assert len(specs) == len(expected_parts)
    for spec, expected in zip(specs, expected_parts, strict=True):
        assert spec.is_slab is slab
        assert spec.slab_starts == starts
        assert spec.slab_sizes == sizes
        assert spec.source_index == 7
        assert spec.launched_power == 2.5
        assert spec.timing == "e"
        assert spec.component == "Ex"
        np.testing.assert_array_equal(
            spec.coeff, expected.reshape(sizes) if slab else expected
        )
    if complex_fields:
        np.testing.assert_array_equal(
            specs[0].waveform, waveform.values - 1j * waveform.quadrature
        )
    else:
        np.testing.assert_array_equal(specs[0].waveform, waveform.values)
        np.testing.assert_array_equal(specs[1].waveform, waveform.quadrature)
