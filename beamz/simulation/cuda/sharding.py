"""Halo exchange and packed CPML ownership around native CUDA Yee phases."""

from __future__ import annotations

from dataclasses import replace
from functools import partial
from typing import cast

import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P

from beamz.simulation import _cuda_abi as abi
from beamz.simulation.backend import CudaBackendUnavailable

_MESH_AXIS = "fdtd"


def _validate_devices(mesh):
    if any(device.platform != "gpu" for device in mesh.devices.flat):
        raise CudaBackendUnavailable("CUDA sharding requires a GPU device mesh")


def validate_sharded_config(config, boundary, plan):
    """Validate the streamed phase and its common component partition layout."""
    if config.backend != "cuda_streamed" or not config.is_3d:
        raise CudaBackendUnavailable(
            "CUDA sharding requires cuda_streamed and a 3D grid. "
            "Use backend='jax' for other sharded configurations."
        )
    if plan is None or not plan.layout.enabled or plan.mesh is None:
        raise ValueError("CUDA sharding requires a compiled device layout")
    axis = plan.layout.axis
    if len({shape[axis] for shape in plan.layout.padded_shapes.values()}) != 1:
        raise ValueError("CUDA component partitions must share global interfaces")
    _validate_devices(plan.mesh)


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


def _phase(state, ctx, coeffs, *, phase):
    from . import runtime

    plan = ctx.sharding_plan
    axis, count = plan.layout.axis, plan.layout.num_devices
    spec = P(*(_MESH_AXIS if i == axis else None for i in range(3)))
    prefix = "h" if phase == 0 else "e"
    targets = tuple(getattr(state, prefix + c) for c in "xyz")
    tensor_update = phase == 1 and bool(coeffs.e_inverse_offdiagonal.size)
    sources = tuple(getattr(state, ("e" if phase == 0 else "h") + c) for c in "xyz")
    materials = tuple(
        getattr(coeffs, f"{prefix}_{kind}_{c}")
        for kind in ("decay", "source")
        for c in "xyz"
    )
    material_specs = tuple(P() if value.ndim == 0 else spec for value in materials)
    terms = ctx.boundary.cpml.h_terms if phase == 0 else ctx.boundary.cpml.e_terms
    psi = state.cpml_psi_h_terms if phase == 0 else state.cpml_psi_e_terms
    # A normal CPML slab is only a pair of thin faces, never a full field. It is
    # replicated and reduced over disjoint owners; transverse slabs partition
    # with the fields. This preserves the public packed continuation layout.
    psi_specs = tuple(P() if term.axis == axis else spec for term in terms)
    profiles = tuple(
        value for term in terms for value in (term.a, term.b, term.inv_kappa)
    )
    logical = plan.layout.logical_shapes
    target_names = tuple(prefix.upper() + c for c in "xyz")
    source_names = tuple(("E" if phase == 0 else "H") + c for c in "xyz")
    shapes = jnp.asarray(
        tuple(logical[name] for name in (*target_names, *source_names)), dtype=jnp.int32
    )

    # Keep the compatibility import local: older supported JAX installations
    # expose shard_map through the experimental module.
    try:
        shard_map = jax.shard_map
    except AttributeError:
        from jax.experimental.shard_map import shard_map

    @partial(
        shard_map,
        mesh=plan.mesh,
        in_specs=(
            (spec,) * 3,
            (spec,) * 3,
            material_specs,
            psi_specs,
            (P(),) * 3,
            (P(),) * len(profiles),
        ),
        out_specs=(*(spec,) * 3, *psi_specs),
    )
    def local_update(
        local_targets,
        local_sources,
        local_materials,
        local_psi,
        metrics,
        local_profiles,
    ):
        extent = local_targets[0].shape[axis]
        origin = jax.lax.axis_index(_MESH_AXIS) * extent
        header = jnp.stack((jnp.int32(axis), origin, jnp.int32(tensor_update)))[None, :]
        geometry = jnp.concatenate((header, shapes), axis=0)
        local_terms = tuple(
            replace(
                term,
                a=local_profiles[3 * i],
                b=local_profiles[3 * i + 1],
                inv_kappa=local_profiles[3 * i + 2],
            )
            for i, term in enumerate(terms)
        )
        local_psi = tuple(
            _owned_psi(
                value,
                term,
                axis=axis,
                origin=origin,
                extent=extent,
                logical_shape=logical[term.component],
            )
            for value, term in zip(local_psi, terms, strict=True)
        )
        outputs = runtime._ffi_phase(
            abi.CUDA_SHARDED_TARGET,
            phase,
            local_targets,
            tuple(
                exchange_halos(
                    value,
                    axis=axis,
                    num_devices=count,
                    # H uses forward differences, E backward differences.
                    # A curl never differentiates the normal field component
                    # along the partition axis (array axes are z, y, x).
                    lower=phase == 1 and component != 2 - axis,
                    upper=phase == 0 and component != 2 - axis,
                )
                for component, value in enumerate(local_sources)
            ),
            local_materials,
            local_terms,
            local_psi,
            metrics,
            metric_kind=runtime._metric_kind_code(ctx),
            dt=ctx.dt,
            resolution=ctx.resolution,
            cuda_flags=ctx.config.cuda_flags,
            metallic_edges=ctx.boundary.cpml.metallic_edges,
            shard_geometry=geometry,
        )
        return (
            *outputs[:3],
            *(
                jax.lax.psum(value, _MESH_AXIS) if term.axis == axis else value
                for value, term in zip(outputs[3:], terms, strict=True)
            ),
        )

    outputs = local_update(
        targets, sources, materials, psi, runtime._phase_metrics(ctx, phase), profiles
    )
    fields = outputs[:3]
    if tensor_update:
        from beamz.simulation.kernels import (
            advance_e_centered_tensor,
            fit_array_to_shape,
        )

        # Colocation extrapolates at physical support endpoints. Storage-only
        # zeros must not move that endpoint or halve an edge's coupled curl.
        slices = tuple(
            tuple(slice(0, n) for n in logical[name]) for name in target_names
        )
        node_shape = tuple(
            max(shape[d] for shape in logical.values()) for d in range(3)
        )
        node_slice = tuple(slice(0, n) for n in node_shape)

        fields = advance_e_centered_tensor(
            tuple(value[sl] for value, sl in zip(targets, slices, strict=True)),
            tuple(value[sl] for value, sl in zip(fields, slices, strict=True)),
            tuple(
                getattr(coeffs, f"e_inverse_diagonal_{c}")[sl]
                for c, sl in zip("xyz", slices, strict=True)
            ),
            coeffs.e_inverse_offdiagonal[node_slice],
            ("Ex", "Ey", "Ez"),
            ctx.dt_scalar,
        )
        fields = tuple(
            fit_array_to_shape(value, target.shape)
            for value, target in zip(fields, targets, strict=True)
        )
    return state._replace(
        **dict(zip((prefix + c for c in "xyz"), fields, strict=True)),
        **{f"cpml_psi_{prefix}_terms": outputs[3:]},
    )


def select_sharded_kernel(ctx):
    from beamz.simulation.kernels import StepUpdateKernel

    validate_sharded_config(ctx.config, ctx.boundary, ctx.sharding_plan)
    return StepUpdateKernel(
        "cuda_streamed_sharded",
        partial(_phase, phase=0),
        partial(_phase, phase=1),
    )
