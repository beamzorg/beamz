"""Explicit one-cell stencil exchange for uniform, diagonal pure-JAX updates."""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P

from beamz.simulation.distributed import exchange_halos

_AXIS = "fdtd"
_CURL = (((2, 1), (1, 0)), ((0, 0), (2, 2)), ((1, 2), (0, 1)))
_EDGES = (("front", "back"), ("bottom", "top"), ("left", "right"))


def supported(program):
    return (
        program.config.backend == "jax"
        and program.config.is_3d
        and program.sharding.layout.enabled
        and program.config.metric_kind == "isotropic_uniform"
        and program.boundary.cpml.enabled
        and not program.coefficients.e_inverse_offdiagonal.size
    )


def _phase(state, ctx, coeffs, *, phase):
    plan = ctx.sharding_plan
    axis, count = plan.layout.axis, plan.layout.num_devices
    spec = P(*(_AXIS if d == axis else None for d in range(3)))
    prefix, other = ("h", "e") if phase == 0 else ("e", "h")
    targets = tuple(getattr(state, prefix + c) for c in "xyz")
    sources = tuple(getattr(state, other + c) for c in "xyz")
    sigmas = tuple(
        getattr(coeffs, f"{prefix}_{'sigma_m' if phase == 0 else 'conductivity'}_{c}")
        for c in "xyz"
    )
    scales = tuple(getattr(coeffs, f"{prefix}_source_{c}") for c in "xyz")
    materials = (*sigmas, *scales)
    terms = ctx.boundary.cpml.h_terms if phase == 0 else ctx.boundary.cpml.e_terms
    psi = getattr(state, f"cpml_psi_{prefix}_terms")
    psi_specs = tuple(P(_AXIS) if term.axis == axis else spec for term in terms)
    profiles = tuple(
        value for term in terms for value in (term.a, term.b, term.inv_kappa)
    )
    logical = plan.layout.logical_shapes
    edges = ctx.boundary.cpml.metallic_edges
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
            tuple(P() if v.ndim == 0 else spec for v in materials),
            psi_specs,
            (P(),) * len(profiles),
        ),
        out_specs=(*(spec,) * 3, *psi_specs),
    )
    def update(local_targets, local_sources, materials, memories, profiles):
        origin = jax.lax.axis_index(_AXIS) * local_targets[0].shape[axis]
        halos = tuple(
            exchange_halos(
                v,
                axis=axis,
                num_devices=count,
                lower=phase == 1 and c != 2 - axis,
                upper=phase == 0 and c != 2 - axis,
            )
            for c, v in enumerate(local_sources)
        )
        fields, next_memories = [], []
        for component, target in enumerate(local_targets):
            shape = target.shape
            physical = logical[prefix.upper() + "xyz"[component]]
            coords = []
            valid = jnp.asarray(True)
            constrained = jnp.asarray(False)
            for d in range(3):
                broadcast = [1, 1, 1]
                broadcast[d] = shape[d]
                coordinate = jnp.arange(shape[d]).reshape(broadcast) + (
                    origin if d == axis else 0
                )
                coords.append(coordinate)
                valid = valid & (coordinate < physical[d])
                if (phase == 0) == (d == 2 - component):
                    if _EDGES[d][0] in edges:
                        constrained = constrained | (coordinate == 0)
                    if _EDGES[d][1] in edges:
                        constrained = constrained | (coordinate == physical[d] - 1)
            curl = jnp.zeros_like(target)
            for which, (source_component, derivative_axis) in enumerate(
                _CURL[component]
            ):
                source = halos[source_component]
                # One-cell zero pads make both stagger directions static slices.
                padding = [(0, 0) if d == axis else (1, 1) for d in range(3)]
                padded = jnp.pad(source, padding)
                starts = [1, 1, 1]
                neighbor = starts.copy()
                neighbor[derivative_axis] += 1 if phase == 0 else -1

                def sample(indices, padded=padded, shape=shape):
                    return padded[
                        tuple(
                            slice(start, start + n)
                            for start, n in zip(indices, shape, strict=True)
                        )
                    ]

                center, adjacent = sample(starts), sample(neighbor)
                derivative = (
                    adjacent - center if phase == 0 else center - adjacent
                ) / ctx.resolution
                coordinate = coords[derivative_axis]
                source_stop = logical[other.upper() + "xyz"[source_component]][
                    derivative_axis
                ]
                if phase == 0:
                    derivative = jnp.where(coordinate + 1 < source_stop, derivative, 0)
                else:
                    allowed = (coordinate > 0) & (coordinate < source_stop)
                    if _EDGES[derivative_axis][0] in edges:
                        allowed = allowed | (coordinate == 0)
                    if _EDGES[derivative_axis][1] in edges:
                        allowed = allowed | (coordinate == source_stop)
                    derivative = jnp.where(allowed, derivative, 0)
                term_index = 2 * component + which
                term = terms[term_index]
                memory = (
                    memories[term_index][0]
                    if term.axis == axis
                    else memories[term_index]
                )
                packed = jnp.arange(term.slab.low + term.slab.high)
                global_positions = jnp.where(
                    packed < term.slab.low,
                    packed,
                    physical[term.axis] - term.slab.high + packed - term.slab.low,
                )
                indices = global_positions - (origin if term.axis == axis else 0)
                owned = (indices >= 0) & (indices < shape[term.axis])
                reshape = [1, 1, 1]
                reshape[term.axis] = len(packed)
                selected = jnp.take(derivative, indices, axis=term.axis, mode="clip")
                a, b, inv_kappa = profiles[3 * term_index : 3 * term_index + 3]
                next_memory = jnp.where(
                    owned.reshape(reshape), b * memory + a * selected, 0
                ).astype(memory.dtype)
                correction = selected * (inv_kappa - 1) + next_memory
                # Positive out-of-range indices are dropped; negative indices must
                # never wrap around and modify the opposite physical boundary.
                indices = jnp.where(owned, indices, shape[term.axis])
                index = [slice(None)] * 3
                index[term.axis] = indices
                derivative = derivative.at[tuple(index)].add(correction, mode="drop")
                curl = curl + term.sign * derivative
                next_memories.append(
                    next_memory[None, ...] if term.axis == axis else next_memory
                )
            signed_curl = -curl if phase == 0 else curl
            next_field = target + materials[3 + component] * (
                signed_curl - materials[component] * target
            )
            fields.append(jnp.where(valid & ~constrained, next_field, 0))
        return (*fields, *next_memories)

    outputs = update(targets, sources, materials, psi, profiles)
    return state._replace(
        **dict(zip((prefix + c for c in "xyz"), outputs[:3], strict=True)),
        **{f"cpml_psi_{prefix}_terms": outputs[3:]},
    )


def select_kernel():
    from beamz.simulation.kernels import StepUpdateKernel

    return StepUpdateKernel(
        "jax_local_cpml", partial(_phase, phase=0), partial(_phase, phase=1)
    )
