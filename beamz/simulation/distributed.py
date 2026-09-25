"""Shared halo exchange and scan-local CPML ownership for distributed backends."""

from __future__ import annotations

from typing import cast

import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P

_MESH_AXIS = "fdtd"


def exchange_halos(value, *, axis, num_devices, lower=True, upper=True):
    """Attach adjacent one-cell faces inside a manual ``fdtd`` mesh.

    The outer faces are zero, never periodic. Only source buffers have halos;
    the native phase returns owned cells in the original partition layout.
    """
    low = jax.lax.slice_in_dim(value, 0, 1, axis=axis)
    high = jax.lax.slice_in_dim(
        value, value.shape[axis] - 1, value.shape[axis], axis=axis
    )
    from_low = (
        jax.lax.ppermute(high, _MESH_AXIS, [(i, i + 1) for i in range(num_devices - 1)])
        if lower
        else jnp.zeros_like(high)
    )
    from_high = (
        jax.lax.ppermute(low, _MESH_AXIS, [(i + 1, i) for i in range(num_devices - 1)])
        if upper
        else jnp.zeros_like(low)
    )
    return jnp.concatenate((from_low, value, from_high), axis=axis)


def _owned_psi(value, term, *, axis, origin, extent, logical_shape):
    """Keep only this rank's physical recurrence entries; neutralize padding."""
    mask = jnp.asarray(True)
    for d, size in enumerate(value.shape):
        positions = jnp.arange(size, dtype=jnp.int32)
        if d == term.axis:
            positions = cast(
                jax.Array,
                jnp.where(
                    positions < term.slab.low,
                    positions,
                    positions - term.slab.low + logical_shape[d] - term.slab.high,
                ),
            )
        elif d == axis:
            positions = positions + origin
        valid = (positions >= 0) & (positions < logical_shape[d])
        if d == axis and d == term.axis:
            valid &= (positions >= origin) & (positions < origin + extent)
        shape = [1, 1, 1]
        shape[d] = size
        mask = mask & valid.reshape(shape)
    return jnp.where(mask, value, jnp.zeros_like(value))


def scan_local_cpml(state, program, *, assemble=False):
    """Keep normal CPML recurrences on their owners between scan boundaries.

    A leading mesh axis stores one packed slab per rank. The physical entries
    are disjoint; only the scan output reduces them into public continuation
    arrays. This is internal storage, never a change to SimulationState's API.
    """
    plan = program.sharding
    axis = plan.layout.axis
    try:
        shard_map = jax.shard_map
    except AttributeError:
        from jax.experimental.shard_map import shard_map

    updates = {}
    for phase in ("h", "e"):
        name = f"cpml_psi_{phase}_terms"
        values = []
        for value, term in zip(
            getattr(state, name),
            getattr(program.boundary.cpml, f"{phase}_terms"),
            strict=True,
        ):
            if term.axis != axis:
                values.append(value)
                continue
            extent = (
                plan.layout.padded_shapes[term.component][axis]
                // plan.layout.num_devices
            )
            logical_shape = plan.layout.logical_shapes[term.component]

            def distribute(
                value, term=term, extent=extent, logical_shape=logical_shape
            ):
                owned = _owned_psi(
                    value,
                    term,
                    axis=axis,
                    origin=jax.lax.axis_index(_MESH_AXIS) * extent,
                    extent=extent,
                    logical_shape=logical_shape,
                )
                return owned[None, ...]

            def collect(value):
                return jax.lax.psum(value[0], _MESH_AXIS)

            convert = shard_map(
                collect if assemble else distribute,
                mesh=plan.mesh,
                in_specs=P(_MESH_AXIS) if assemble else P(),
                out_specs=P() if assemble else P(_MESH_AXIS),
            )
            values.append(convert(value))
        updates[name] = tuple(values)
    return state._replace(**updates)
