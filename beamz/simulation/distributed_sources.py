"""Inject each slab source only on the ranks intersecting its physical support."""

from functools import partial
from typing import cast

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


def sources_are_interior(state, batches, plan):
    """Prove that every possible source write avoids physical boundary cells.

    Use the padded batch extent and the actual clamped start, not just the
    original source bounds. Irregular or boundary-touching sources keep masks.
    """
    for (_, component), (group, rest) in batches.items():
        if rest:
            return False
        if group is None:
            continue
        shape = plan.layout.logical_shapes[component]
        value = getattr(state, component.lower())
        for start in _clamped_starts(value, group):
            if any(
                first < 1 or first + width > size - 1
                for first, width, size in zip(
                    start, group.max_sizes, shape, strict=True
                )
            ):
                return False
    return True


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
            addition = cast(
                jax.Array,
                jnp.where(
                    valid.reshape(broadcast), patch * waveforms[source, safe_step], 0
                ),
            )
            target = list(start)
            target[axis] = local_start
            target = tuple(jnp.asarray(index, dtype=jnp.int32) for index in target)
            if local.size > 2**31 - 1:
                # Keep multidimensional indexing when a flat FP32-mode JAX
                # index would overflow its default signed 32-bit integer.
                previous = jax.lax.dynamic_slice(local, target, tuple(patch_shape))
                local = jax.lax.dynamic_update_slice(
                    local, previous + addition.astype(local.dtype), target
                )
                continue
            # A multidimensional read/modify/write slice can make XLA transpose
            # entire field shards to favor a thin source plane. Native stepping
            # then needs the original layout too. Linear indices keep the patch
            # update in the field's existing row-major storage without those
            # full-volume layout buffers.
            flat_indices = jnp.asarray(0, dtype=jnp.int32)
            stride = 1
            for dim in reversed(range(local.ndim)):
                index_shape = [1] * local.ndim
                index_shape[dim] = patch_shape[dim]
                flat_indices = (
                    flat_indices
                    + (
                        target[dim]
                        + jnp.arange(patch_shape[dim], dtype=jnp.int32).reshape(
                            index_shape
                        )
                    )
                    * stride
                )
                stride *= local.shape[dim]
            local = (
                local.reshape(-1)
                .at[flat_indices.reshape(-1)]
                .add(addition.astype(local.dtype).reshape(-1), unique_indices=True)
                .reshape(local.shape)
            )
        return local

    return inject(value, abs_step, group.coeffs, group.waveforms)
