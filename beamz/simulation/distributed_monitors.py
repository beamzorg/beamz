"""Accumulate linear DFT contributions locally; reduce at the scan boundary."""

from functools import partial
from typing import cast

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import PartitionSpec as P

from beamz.simulation.distributed import _shard_map
from beamz.simulation.observe import is_vector_dft_monitor

_AXIS = "fdtd"


def supported(program):
    return (
        program.config.is_3d
        and program.sharding.layout.enabled
        and bool(program.monitors)
        and all(is_vector_dft_monitor(mon) for mon in program.monitors)
    )


def scan_local_dft(state, program, *, assemble=False):
    """Keep public one-dimensional arenas unchanged outside compiled scans.

    A continuation's prior integral is placed on rank zero exactly once. New
    contributions belong to the ranks owning their interpolation neighbors.
    Weights are replicated and count time samples, not devices.
    """

    def distribute(value):
        return cast(jax.Array, jnp.where(jax.lax.axis_index(_AXIS) == 0, value, 0))[
            None, :
        ]

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
        vectors = mon.dft_sampling.sample(
            local_fields,
            active_mask=np.asarray(mon.dft_component_mask),
            local_axis=axis,
            shard_index=jax.lax.axis_index(_AXIS),
            shapes=global_shapes,
        )
        re, im, weight = _accumulate_dft(
            mon,
            (re[0], im[0], weight),
            None,
            time,
            dt,
            sampled_vectors=vectors,
        )
        return re[None, :], im[None, :], weight

    return local(fields, *carry, time, dt)
