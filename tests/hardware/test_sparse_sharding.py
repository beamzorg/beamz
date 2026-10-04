"""Sparse monitor/source operations must not gather complete field volumes."""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap


def test_sparse_monitor_and_source_updates_preserve_field_partitioning():
    code = r"""
import jax
import jax.numpy as jnp
import numpy as np

from beamz.devices.sources.compiler import CompiledSourceSpec, batch_slab_specs
from beamz.simulation.execute import _apply_batched_slabs
from beamz.simulation.observe import _sample_components

mesh = jax.sharding.Mesh(np.asarray(jax.devices()), ('shard',))
rng = np.random.default_rng(13)
for shape in ((7, 32), (3, 7, 32)):
    partition = jax.sharding.NamedSharding(
        mesh, jax.sharding.PartitionSpec(*([None] * (len(shape) - 1)), 'shard')
    )
    replicated = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
    fields = tuple(rng.normal(size=shape).astype(np.float32) for _ in range(6))
    # Include both outer boundaries, adjacent shards, and repeated interpolation taps.
    flat = np.asarray([[0, 1, 3, 4], [15, 16, 30, 31],
                       [np.prod(shape) - 1] * 4], dtype=np.int32)
    indices = tuple(jnp.asarray(flat) for _ in fields)
    weights = tuple(jnp.asarray([[0.1, 0.2, 0.3, 0.4]] * 3) for _ in fields)
    placed = tuple(jax.device_put(field, partition) for field in fields)
    sample = jax.jit(
        lambda values: _sample_components(values, indices, weights),
        out_shardings=replicated,
    )
    executable = sample.lower(placed).compile()
    expected = np.stack([
        np.sum(field.reshape(-1)[flat] * np.asarray(weight), axis=-1)
        for field, weight in zip(fields, weights)
    ])
    np.testing.assert_allclose(executable(placed), expected, rtol=2e-6, atol=2e-7)
    assert 'all-gather' not in executable.as_text().lower()

    slab_shape = (1,) * len(shape)
    starts = ((0,) * len(shape), tuple(size - 1 for size in shape))
    specs = tuple(CompiledSourceSpec(
        component='Ex', timing='e',
        index=tuple(slice(start, start + 1) for start in location),
        coeff=jnp.full(slab_shape, index + 1, dtype=jnp.float32),
        waveform=jnp.asarray([0.25, 0.75], dtype=jnp.float32),
        is_slab=True, slab_starts=location, slab_sizes=slab_shape,
    ) for index, location in enumerate(starts))
    group, rest = batch_slab_specs(specs)
    assert not rest
    inject = jax.jit(
        lambda field: _apply_batched_slabs(
            field, jnp.asarray(1), group, dense_single_slab=False
        ), out_shardings=partition,
    )
    executable = inject.lower(placed[0]).compile()
    expected = fields[0].copy()
    expected[starts[0]] += 0.75
    expected[starts[1]] += 1.5
    np.testing.assert_array_equal(executable(placed[0]), expected)
    assert 'all-gather' not in executable.as_text().lower()
"""
    environment = os.environ.copy()
    environment["JAX_PLATFORMS"] = "cpu"
    environment["XLA_FLAGS"] = "--xla_force_host_platform_device_count=8"
    subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code)],
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )
