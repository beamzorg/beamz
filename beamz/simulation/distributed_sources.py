"""Inject each slab source only on the ranks intersecting its physical support."""

from functools import partial

import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P


def _clamped_starts(value, group):
    return tuple(
        tuple(
            min(
                max(start if start >= 0 else start + value.shape[d], 0),
                value.shape[d] - group.max_sizes[d],
            )
            for d, start in enumerate(source_start)
        )
        for source_start in group.starts_tuple
    )


def requires_local_injection(value, group, plan):
    """Static, single-owner scatters already lower well; localize crossing slabs.

    Larger source batches otherwise use dynamic global read/modify/write. Both
    those and patches crossing a partition can introduce non-stencil gathers.
    """
    if group.n > 2:
        return True
    axis = plan.layout.axis
    extent = value.shape[axis] // plan.layout.num_devices
    width = group.max_sizes[axis]
    return any(
        start[axis] // extent != (start[axis] + width - 1) // extent
        for start in _clamped_starts(value, group)
    )


def apply_batched_slabs(value, abs_step, group, plan):
    axis, count = plan.layout.axis, plan.layout.num_devices
    spec = P(*("fdtd" if d == axis else None for d in range(value.ndim)))
    extent = value.shape[axis] // count
    starts = _clamped_starts(value, group)
    try:
        shard_map = jax.shard_map
    except AttributeError:
        from jax.experimental.shard_map import shard_map

    @partial(shard_map, mesh=plan.mesh, in_specs=(spec, P(), P(), P()), out_specs=spec)
    def inject(local, step, coefficients, waveforms):
        safe_step = jnp.clip(step, 0, waveforms.shape[1] - 1)
        origin = jax.lax.axis_index("fdtd") * extent
        width = min(group.max_sizes[axis], extent)
        patch_shape = list(group.max_sizes)
        patch_shape[axis] = width
        broadcast = [1] * local.ndim
        broadcast[axis] = width
        for source, start in enumerate(starts):
            local_start = jnp.clip(start[axis] - origin, 0, extent - width)
            positions = origin + local_start + jnp.arange(width) - start[axis]
            valid = (positions >= 0) & (positions < group.max_sizes[axis])
            patch = jnp.take(coefficients[source], positions, axis=axis, mode="clip")
            addition = jnp.where(
                valid.reshape(broadcast), patch * waveforms[source, safe_step], 0
            )
            target = list(start)
            target[axis] = local_start
            target = tuple(jnp.asarray(index, dtype=jnp.int32) for index in target)
            previous = jax.lax.dynamic_slice(local, target, tuple(patch_shape))
            local = jax.lax.dynamic_update_slice(
                local, previous + addition.astype(local.dtype), target
            )
        return local

    return inject(value, abs_step, group.coeffs, group.waveforms)
