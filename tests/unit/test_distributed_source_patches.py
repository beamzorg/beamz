"""Local patch injection preserves overlapping and partition-crossing sources."""

import os
import subprocess
import sys
import textwrap


def test_local_source_patches_match_global_updates_on_every_axis():
    code = r"""
from types import SimpleNamespace
import jax
import jax.numpy as jnp
import numpy as np
from beamz.devices.sources.compiler import CompiledSourceSpec, batch_slab_specs
from beamz.simulation.distributed_sources import apply_batched_slabs

shape = (8, 12, 16)
mesh = jax.sharding.Mesh(np.asarray(jax.devices()), ('fdtd',))
original = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
for axis in range(3):
    spec = jax.sharding.PartitionSpec(*('fdtd' if d == axis else None for d in range(3)))
    target = jax.sharding.NamedSharding(mesh, spec)
    plan = SimpleNamespace(mesh=mesh, layout=SimpleNamespace(axis=axis, num_devices=2))
    seam = [2, 3, 4]
    seam[axis] = shape[axis] // 2 - 1
    starts = (tuple(seam), tuple(seam), (100, -1, 100))
    sources = tuple(CompiledSourceSpec(
        component='Ex', timing='e', index=tuple(slice(s, s + 3) for s in start),
        coeff=jnp.full((3, 3, 3), i + 1, dtype=jnp.float32),
        waveform=jnp.asarray([0.25, -0.5], dtype=jnp.float32),
        is_slab=True, slab_starts=start, slab_sizes=(3, 3, 3),
    ) for i, start in enumerate(starts))
    group, rest = batch_slab_specs(sources)
    assert not rest
    inject = jax.jit(lambda field, step: apply_batched_slabs(field, step, group, plan))
    for step in (-1, 0, 1, 9):
        expected = original.copy()
        amplitude = 0.25 if step <= 0 else -0.5
        for i, start in enumerate(starts):
            normalized = [min(max(s if s >= 0 else s + n, 0), n - 3)
                          for s, n in zip(start, shape)]
            expected[tuple(slice(s, s + 3) for s in normalized)] += (i + 1) * amplitude
        result = inject(jax.device_put(original, target), jnp.int32(step))
        np.testing.assert_array_equal(result, expected)
        assert len(result.sharding.device_set) == 2
"""
    subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code)],
        env={
            **os.environ,
            "JAX_PLATFORMS": "cpu",
            "XLA_FLAGS": "--xla_force_host_platform_device_count=2",
        },
        check=True,
        capture_output=True,
        text=True,
    )
