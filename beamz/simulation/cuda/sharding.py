"""Halo exchange and packed CPML ownership around native CUDA Yee phases."""

from __future__ import annotations

from dataclasses import replace
from functools import partial

import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P

from beamz.simulation import _cuda_abi as abi
from beamz.simulation.backend import CudaBackendUnavailable
from beamz.simulation.distributed import _owned_psi, exchange_faces

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


def _phase(state, ctx, coeffs, *, phase):
    from . import runtime

    plan = ctx.sharding_plan
    axis, count = plan.layout.axis, plan.layout.num_devices
    phase_call = runtime._ffi_phase
    if (
        axis == 2
        and ctx.config.metric_kind == "isotropic_uniform"
        and runtime._uniform_cpml_thickness(ctx) > 0
        and not coeffs.e_inverse_offdiagonal.size
        and all(
            value.dtype == jnp.float32
            for value in (*state.cpml_psi_h_terms, *state.cpml_psi_e_terms)
        )
    ):
        from .storage import wrap_sharded_phase

        phase_call = wrap_sharded_phase(phase_call, (2, 0, 1))
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
    owner_local = tuple(value.ndim == 4 for value in psi)
    psi_specs = tuple(
        P(_MESH_AXIS) if owned else P() if term.axis == axis else spec
        for term, owned in zip(terms, owner_local, strict=True)
    )
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
            value[0] if owned else value
            for value, owned in zip(local_psi, owner_local, strict=True)
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
        outputs = phase_call(
            abi.CUDA_SHARDED_TARGET,
            phase,
            local_targets,
            local_sources,
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
            shard_halos=tuple(
                face
                for component, value in enumerate(local_sources)
                for face in exchange_faces(
                    value,
                    axis=axis,
                    num_devices=count,
                    lower=phase == 1 and component != 2 - axis,
                    upper=phase == 0 and component != 2 - axis,
                )
            ),
        )
        return (
            *outputs[:3],
            *(
                value[None, ...]
                if owned
                else jax.lax.psum(value, _MESH_AXIS)
                if term.axis == axis
                else value
                for value, term, owned in zip(
                    outputs[3:], terms, owner_local, strict=True
                )
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
