"""Placement and material interpolation for resident distributed reverse mode."""

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import PartitionSpec as P

from beamz.lattice import component_axis_offsets_3d
from beamz.simulation import distributed_monitors, sharding
from beamz.simulation.distributed import exchange_halos


def validate(program):
    """Reject physics/layouts without a bounded distributed transpose."""
    cfg, grid = program.config, program.grid
    if (
        cfg.backend not in {"jax", "cuda_streamed"}
        or not cfg.is_3d
        or cfg.metric_kind != "isotropic_uniform"
        or program.coefficients.e_inverse_offdiagonal.size
        or program.boundary.periodic_axes
        or jax.config.read("jax_enable_x64")
        or any(
            d.process_index != jax.process_index()
            for d in program.sharding.mesh.devices.flat
        )
        or any(
            np.dtype(getattr(grid, c).dtype) != np.dtype("float32")
            for c in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz")
        )
    ):
        raise ValueError(
            "Sharded autodiff requires single-host FP32 uniform 3D scalar Yee physics with PEC/CPML boundaries"
        )
    if any(
        np.any(np.asarray(getattr(grid, f"sig_{a}")) != 0)
        or np.any(np.asarray(getattr(grid, f"sigma_m_h{a}")) != 0)
        or np.any(np.asarray(getattr(grid, f"mu_h{a}")) != 1)
        for a in "xyz"
    ):
        raise ValueError(
            "Sharded autodiff requires lossless dielectrics and unit permeability"
        )
    if program.monitors and (
        not distributed_monitors.supported(program)
        or any(mon.normal_axis not in (0, 1, 2) for mon in program.monitors)
    ):
        raise ValueError(
            "Sharded autodiff supports planar vector DFT observations only"
        )


def parameter_shape(program):
    shape = list(program.grid.permittivity.shape)
    axis = program.sharding.layout.axis
    shape[axis] = program.sharding.layout.padded_shapes["Ex"][axis]
    if any(
        padded[axis] != shape[axis]
        for padded in program.sharding.layout.padded_shapes.values()
    ):
        raise ValueError(
            "Sharded material interpolation requires aligned Yee partitions"
        )
    return tuple(shape)


def place_parameter(program, value):
    """Pad host cells directly into device shards, before applying grad/jit.

    Spatial differentiation inputs use this padded shape; padding has zero
    sensitivity. Scalars remain replicated compact parameters.
    """
    shape = parameter_shape(program)
    if np.ndim(value) == 0:
        return sharding.place_tree(program, value, shard_arrays=False)
    if tuple(value.shape) not in (program.grid.permittivity.shape, shape):
        raise ValueError(f"Cell parameters must have logical or padded shape {shape}")
    target = sharding._array_sharding(
        program, jax.ShapeDtypeStruct(shape, value.dtype), program.sharding.mesh
    )
    if sharding._is_host_array(value):
        return sharding._place_padded_host_array(value, shape, target)
    return jax.lax.with_sharding_constraint(
        sharding._pad_high_to_shape(value, shape), target
    )


def sample_material(program, value, component):
    """Interpolate owning cells plus one halo, including the exact transpose."""
    plan = program.sharding
    axis, count = plan.layout.axis, plan.layout.num_devices
    spec = P(*("fdtd" if d == axis else None for d in range(3)))
    logical = program.grid.permittivity.shape
    shape = plan.layout.padded_shapes[component]
    offsets = component_axis_offsets_3d(component)
    try:
        shard_map = jax.shard_map
    except AttributeError:
        from jax.experimental.shard_map import shard_map

    @partial(shard_map, mesh=plan.mesh, in_specs=spec, out_specs=spec)
    def sample(local):
        origin = jax.lax.axis_index("fdtd") * local.shape[axis]
        result = exchange_halos(local, axis=axis, num_devices=count, upper=False)
        for d, name in enumerate("zyx"):
            positions = jnp.arange(shape[d] // count if d == axis else shape[d])
            positions = positions + (origin if d == axis else 0)

            def take(shift, positions=positions, d=d, result=result):
                indices = jnp.clip(positions + shift, 0, logical[d] - 1)
                if d == axis:
                    indices = indices - origin + 1
                return jnp.take(result, indices, axis=d, mode="clip")

            result = 0.5 * (take(-1) + take(0)) if offsets[name] == 0 else take(0)
        return result

    if value.ndim == 0:
        value = jnp.broadcast_to(value, parameter_shape(program))
    if value.shape != parameter_shape(program):
        raise ValueError(
            "Place spatial parameters with solver.prepare_permittivity before grad/jit"
        )
    return sample(value)
