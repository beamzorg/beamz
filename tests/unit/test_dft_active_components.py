"""DFT specialization must preserve inactive arenas and weighted active samples."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from beamz.devices.monitors.compiler import CompiledMonitorSpec
from beamz.simulation.observe import _accumulate_dft


@pytest.mark.parametrize("active", [(2,), (0, 1, 3, 4), tuple(range(6))])
def test_dft_active_components_preserve_ragged_arena(active):
    nf, points, offset = 17, 3, 7
    indices = tuple(jnp.array([[0, 1], [2, 3], [1, 3]]) for _ in range(6))
    weights = tuple(jnp.full((points, 2), 0.5) for _ in range(6))
    mask = np.array([float(c in active) for c in range(6)], dtype=np.float32)
    frequencies = np.linspace(0, 1, nf, dtype=np.float32)
    monitor = CompiledMonitorSpec(
        name="masked",
        monitor_index=0,
        record_interval=1,
        accumulate_power=False,
        power_scale=1,
        freq_count=nf,
        freq_hz=jnp.asarray(frequencies),
        dft_enabled=True,
        dft_point_count=points,
        dft_value_offset=offset,
        dft_weight_offset=2,
        dft_component_mask=jnp.asarray(mask),
        dft_flat_idx=indices,
        dft_weights=weights,
    )
    fields = tuple(
        jnp.arange(4, dtype=jnp.float32).reshape(2, 2) + c
        if c in active
        else jnp.full((2, 2), jnp.nan)
        for c in range(6)
    )
    size = offset + 6 * nf * points + 5
    initial = (jnp.full(size, 3.0), jnp.full(size, -2.0), jnp.ones(nf + 5))
    result = jax.jit(
        lambda carry: _accumulate_dft(monitor, carry, fields, 0.125, 0.01)
    )(initial)
    expected = [np.asarray(value).copy() for value in initial]
    phase = np.exp(2j * np.pi * frequencies * 0.125)
    for c in active:
        samples = np.array([0.5, 2.5, 2.0]) + c
        delta = phase[:, None] * samples[None, :]
        region = slice(offset + c * nf * points, offset + (c + 1) * nf * points)
        expected[0][region] += delta.real.ravel()
        expected[1][region] += delta.imag.ravel()
    expected[2][2 : 2 + nf] += 1
    for actual, reference in zip(result, expected, strict=True):
        np.testing.assert_allclose(actual, reference, rtol=1e-6, atol=1e-6)
