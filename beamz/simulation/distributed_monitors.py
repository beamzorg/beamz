"""Accumulate linear DFT contributions locally; reduce at the scan boundary."""

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import PartitionSpec as P

_AXIS = "fdtd"


def supported(program):
    return (
        program.config.is_3d
        and program.sharding.layout.enabled
        and bool(program.monitors)
        and all(
            mon.dft_enabled
            and mon.freq_count > 0
            and mon.dft_point_count > 0
            and not mon.accumulate_power
            and not mon.accumulate_frequency
            and mon.recorder_index < 0
            for mon in program.monitors
        )
    )


def _shard_map():
    try:
        return jax.shard_map
    except AttributeError:
        from jax.experimental.shard_map import shard_map

        return shard_map


def scan_local_dft(state, program, *, assemble=False):
    """Keep public one-dimensional arenas unchanged outside compiled scans.

    A continuation's prior integral is placed on rank zero exactly once. New
    contributions belong to the ranks owning their interpolation neighbors.
    Weights are replicated and count time samples, not devices.
    """

    def distribute(value):
        return jnp.where(jax.lax.axis_index(_AXIS) == 0, value, 0)[None, :]

    def collect(value):
        return jax.lax.psum(value[0], _AXIS)

    convert = _shard_map()(
        collect if assemble else distribute,
        mesh=program.sharding.mesh,
        in_specs=P(_AXIS) if assemble else P(),
        out_specs=P() if assemble else P(_AXIS),
    )
    return state._replace(
        dft_vec_re=convert(state.dft_vec_re),
        dft_vec_im=convert(state.dft_vec_im),
    )


def accumulate_dft(program, mon, carry, fields, time, dt):
    from .observe import _accumulate_dft

    axis = program.sharding.layout.axis
    spec = P(*(_AXIS if i == axis else None for i in range(3)))
    global_shapes = tuple(field.shape for field in fields)

    @partial(
        _shard_map(),
        mesh=program.sharding.mesh,
        in_specs=((spec,) * 6, P(_AXIS), P(_AXIS), P(), P(), P()),
        out_specs=(P(_AXIS), P(_AXIS), P()),
    )
    def local(local_fields, re, im, weight, time, dt):
        vectors = []
        for component, (field, shape, indices, weights) in enumerate(
            zip(
                local_fields,
                global_shapes,
                mon.dft_flat_idx,
                mon.dft_weights,
                strict=True,
            )
        ):
            if np.asarray(mon.dft_component_mask)[component] == 0:
                vectors.append(jnp.zeros(indices.shape[:-1], dtype=field.dtype))
                continue
            coordinates = list(jnp.unravel_index(indices, shape))
            local_coordinate = (
                coordinates[axis] - jax.lax.axis_index(_AXIS) * field.shape[axis]
            )
            owned = (local_coordinate >= 0) & (local_coordinate < field.shape[axis])
            coordinates[axis] = jnp.clip(local_coordinate, 0, field.shape[axis] - 1)
            samples = jnp.where(owned, field[tuple(coordinates)], 0)
            vectors.append(jnp.sum(samples * weights, axis=-1))
        re, im, weight = _accumulate_dft(
            mon,
            (re[0], im[0], weight),
            None,
            time,
            dt,
            sampled_vectors=jnp.stack(vectors),
        )
        return re[None, :], im[None, :], weight

    return local(fields, *carry, time, dt)
