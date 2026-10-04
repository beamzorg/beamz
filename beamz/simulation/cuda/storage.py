"""Cyclic storage transforms around the existing native graph.

No simulation is rebuilt and no mode is solved again. Sources and monitor
sample order stay physically identical. Only right-handed cyclic axis orders
are supported. All transposes and coefficient repacking are inside the timed
compiled invocation, not silently excluded from its cost.
"""

import math
from dataclasses import replace

import jax.numpy as jnp
import numpy as np


def wrap_program_call(original, axes):
    axes = tuple(axes)
    if axes not in ((0, 1, 2), (1, 2, 0), (2, 0, 1)):
        raise ValueError("CUDA storage axes must be a right-handed cyclic permutation")
    inverse_axes = tuple(axes.index(a) for a in range(3))
    components = tuple(2 - axes[2 - c] for c in range(3))
    inverse_components = tuple(components.index(c) for c in range(3))
    terms = tuple(2 * c + k for c in components for k in range(2))
    inverse_terms = tuple(terms.index(t) for t in range(6))
    six_components = (*components, *(3 + c for c in components))

    def transpose(value):
        return jnp.transpose(value, axes) if getattr(value, "ndim", 0) == 3 else value

    def permute_fields(state, order, spatial):
        return {
            phase + "xyz"[c]: jnp.transpose(getattr(state, phase + "xyz"[old]), spatial)
            for phase in ("h", "e")
            for c, old in enumerate(order)
        }

    def repack(value, shape):
        # Four unsigned byte IDs per little-endian int32. Keep the lookup table
        # unchanged and remove old tail bytes before transposing logical cells.
        raw = value.astype(jnp.uint32)
        unpacked = (
            (raw[:, None] >> (8 * jnp.arange(4, dtype=jnp.uint32))) & 255
        ).reshape(-1)
        unpacked = (
            unpacked[: math.prod(shape)].reshape(shape).transpose(axes).reshape(-1)
        )
        padded = jnp.pad(unpacked, (0, -unpacked.size % 4)).reshape(-1, 4)
        return (
            padded[:, 0]
            | (padded[:, 1] << 8)
            | (padded[:, 2] << 16)
            | (padded[:, 3] << 24)
        ).astype(jnp.int32)

    def call(state, ctx, coeffs, groups, monitors, nsteps, **clock):
        if axes == (0, 1, 2):
            return original(state, ctx, coeffs, groups, monitors, nsteps)
        if not (
            ctx.is_3d
            and ctx.config.metric_kind == "isotropic_uniform"
            and ctx.boundary.cpml.enabled
            and coeffs.e_inverse_offdiagonal.size == 0
        ):
            raise ValueError(
                "Cyclic CUDA storage requires uniform 3D CPML and diagonal materials"
            )
        # Monitor arenas are ragged in point/frequency count. Preserve their
        # canonical offsets and spatial sample order; only component lanes move.
        counts = (
            np.asarray(monitors[4])
            if monitors is not None
            else np.empty((0, 5), dtype=np.int32)
        )
        dft_order = np.arange(state.dft_vec_re.size)
        for frequencies, points, _, offset, _ in counts:
            count = 6 * int(frequencies) * int(points)
            old = np.arange(int(offset), int(offset) + count).reshape(6, -1)
            dft_order[int(offset) : int(offset) + count] = old[
                np.asarray(six_components)
            ].reshape(-1)
        inverse_dft = np.argsort(dft_order)
        new_state = state._replace(
            **permute_fields(state, components, axes),
            cpml_psi_h_terms=tuple(transpose(state.cpml_psi_h_terms[t]) for t in terms),
            cpml_psi_e_terms=tuple(transpose(state.cpml_psi_e_terms[t]) for t in terms),
            dft_vec_re=state.dft_vec_re[dft_order],
            dft_vec_im=state.dft_vec_im[dft_order],
        )

        def boundary_terms(old_terms):
            result = []
            for new_index, old_index in enumerate(terms):
                term = old_terms[old_index]
                component = term.component[0] + "xyz"[new_index // 2]
                axis = inverse_axes[term.axis]
                if axis != (1, 0, 0, 2, 2, 1)[new_index] or term.sign != (
                    1 if new_index % 2 == 0 else -1
                ):
                    raise ValueError(
                        "Cyclic storage requires canonical 3D CPML curl terms"
                    )
                result.append(
                    replace(
                        term,
                        component=component,
                        axis=axis,
                        a=transpose(term.a),
                        b=transpose(term.b),
                        inv_kappa=transpose(term.inv_kappa),
                        slab=term.slab._replace(
                            axis=axis, shape=tuple(term.slab.shape[a] for a in axes)
                        ),
                    )
                )
            return tuple(result)

        faces = (("front", "back"), ("bottom", "top"), ("left", "right"))
        old_cpml = ctx.boundary.cpml
        new_edges = frozenset(
            faces[new][side]
            for new, old in enumerate(axes)
            for side in range(2)
            if faces[old][side] in old_cpml.metallic_edges
        )
        cpml = replace(
            old_cpml,
            metallic_edges=new_edges,
            h_terms=boundary_terms(old_cpml.h_terms),
            e_terms=boundary_terms(old_cpml.e_terms),
        )
        new_ctx = replace(
            ctx,
            boundary=replace(ctx.boundary, cpml=cpml),
            metrics=ctx.metrics._replace(
                **{
                    f"{kind}_{'xyz'[c]}": getattr(ctx.metrics, f"{kind}_{'xyz'[old]}")
                    for kind in ("e_to_h", "h_to_e")
                    for c, old in enumerate(components)
                }
            ),
        )
        material = {
            name: transpose(
                getattr(coeffs, name[:-1] + "xyz"[components["xyz".index(name[-1])]])
            )
            for name in coeffs._fields
            if name[-2:] in ("_x", "_y", "_z")
        }
        for phase in ("h", "e"):
            for c, old in enumerate(components):
                decay = getattr(coeffs, f"{phase}_decay_{'xyz'[old]}")
                source = getattr(coeffs, f"{phase}_source_{'xyz'[old]}")
                if phase == "e" and source.ndim == 1 and source.dtype == jnp.int32:
                    shape = getattr(state, "e" + "xyz"[old]).shape
                    source = repack(source, shape)
                else:
                    source = transpose(source)
                material[f"{phase}_decay_{'xyz'[c]}"] = transpose(decay)
                material[f"{phase}_source_{'xyz'[c]}"] = source
        new_coeffs = coeffs._replace(**material)
        new_groups = []
        for phase in range(3):
            for old in components:
                group = groups[3 * phase + old]
                new_groups.append(
                    None
                    if group is None
                    else replace(
                        group,
                        coeffs=group.coeffs.transpose((0, *(1 + a for a in axes))),
                        starts=group.starts[:, axes],
                        starts_tuple=tuple(
                            tuple(start[a] for a in axes)
                            for start in group.starts_tuple
                        ),
                        max_sizes=tuple(group.max_sizes[a] for a in axes),
                    )
                )
        new_monitors = None
        if monitors is not None:
            new_indices = []
            for old in six_components:
                name = ("ex", "ey", "ez", "hx", "hy", "hz")[old]
                shape = getattr(state, name).shape
                flat = monitors[0][:, old]
                coord = (
                    flat // (shape[1] * shape[2]),
                    (flat // shape[2]) % shape[1],
                    flat % shape[2],
                )
                remapped = (coord[axes[0]] * shape[axes[1]] + coord[axes[1]]) * shape[
                    axes[2]
                ] + coord[axes[2]]
                new_indices.append(
                    jnp.where((flat >= 0) & (flat < math.prod(shape)), remapped, -1)
                )
            new_monitors = (
                jnp.stack(new_indices, axis=1),
                monitors[1][:, six_components],
                monitors[2],
                monitors[3][:, six_components],
                *monitors[4:],
            )
        result = original(
            new_state,
            new_ctx,
            new_coeffs,
            tuple(new_groups),
            new_monitors,
            nsteps,
            **clock,
        )
        return result._replace(
            **permute_fields(result, inverse_components, inverse_axes),
            cpml_psi_h_terms=tuple(
                jnp.transpose(result.cpml_psi_h_terms[t], inverse_axes)
                for t in inverse_terms
            ),
            cpml_psi_e_terms=tuple(
                jnp.transpose(result.cpml_psi_e_terms[t], inverse_axes)
                for t in inverse_terms
            ),
            dft_vec_re=result.dft_vec_re[inverse_dft],
            dft_vec_im=result.dft_vec_im[inverse_dft],
        )

    return call


def wrap_native_calls(run_steps, run_program_steps, run_source_group_steps, axes):
    """Wrap each native graph entry once, before tracing a compiled scan."""
    program = wrap_program_call(run_program_steps, axes)
    source = wrap_program_call(
        lambda state, ctx, coeffs, groups, _monitors, nsteps: run_source_group_steps(
            state, ctx, coeffs, groups, nsteps
        ),
        axes,
    )
    plain = wrap_program_call(
        lambda state, ctx, coeffs, _groups, _monitors, nsteps: run_steps(
            state, ctx, coeffs, nsteps
        ),
        axes,
    )

    def source_call(state, ctx, coeffs, groups, nsteps):
        return source(state, ctx, coeffs, groups, None, nsteps)

    def plain_call(state, ctx, coeffs, nsteps):
        return plain(state, ctx, coeffs, (None,) * 9, None, nsteps)

    return plain_call, program, source_call


def wrap_sharded_phase(original, axes):
    """Rotate a local phase and its faces without changing physical ownership."""
    axes = tuple(axes)
    if axes not in ((0, 1, 2), (1, 2, 0), (2, 0, 1)):
        raise ValueError("CUDA storage axes must be a right-handed cyclic permutation")
    inverse_axes = tuple(axes.index(a) for a in range(3))
    components = tuple(2 - axes[2 - c] for c in range(3))
    inverse_components = tuple(components.index(c) for c in range(3))

    def transpose(value):
        return jnp.transpose(value, axes) if value.ndim == 3 else value

    def call(target, phase, targets, sources, materials, terms, psi, metrics, **kwargs):
        if axes == (0, 1, 2):
            return original(
                target,
                phase,
                targets,
                sources,
                materials,
                terms,
                psi,
                metrics,
                **kwargs,
            )
        order = tuple(2 * c + k for c in components for k in range(2)) if terms else ()
        inverse_terms = tuple(order.index(t) for t in range(len(terms)))
        rotated_terms = []
        for new_index, old_index in enumerate(order):
            term = terms[old_index]
            axis = inverse_axes[term.axis]
            if axis != (1, 0, 0, 2, 2, 1)[new_index] or term.sign != (
                1 if new_index % 2 == 0 else -1
            ):
                raise ValueError("Cyclic storage requires canonical 3D CPML curl terms")
            rotated_terms.append(
                replace(
                    term,
                    component=term.component[0] + "xyz"[new_index // 2],
                    axis=axis,
                    a=transpose(term.a),
                    b=transpose(term.b),
                    inv_kappa=transpose(term.inv_kappa),
                    slab=term.slab._replace(
                        axis=axis, shape=tuple(term.slab.shape[a] for a in axes)
                    ),
                )
            )
        faces = (("front", "back"), ("bottom", "top"), ("left", "right"))
        old_edges = kwargs["metallic_edges"]
        kwargs["metallic_edges"] = frozenset(
            faces[new][side]
            for new, old in enumerate(axes)
            for side in range(2)
            if faces[old][side] in old_edges
        )
        geometry = kwargs["shard_geometry"]
        # Origin and return-curl mode are unchanged. Axis and component shapes
        # rotate with the fields; normal-first storage makes the new axis zero.
        new_axis = jnp.asarray(inverse_axes, dtype=jnp.int32)[geometry[0, 0]]
        header = geometry[:1].at[0, 0].set(new_axis)
        rows = jnp.asarray((*[1 + c for c in components], *[4 + c for c in components]))
        kwargs["shard_geometry"] = jnp.concatenate(
            (header, geometry[rows][:, jnp.asarray(axes)])
        )
        halos = kwargs["shard_halos"]
        kwargs["shard_halos"] = tuple(
            transpose(halos[2 * c + s]) for c in components for s in (0, 1)
        )
        outputs = original(
            target,
            phase,
            tuple(transpose(targets[c]) for c in components),
            tuple(transpose(sources[c]) for c in components),
            tuple(
                transpose(materials[base + c]) for base in (0, 3) for c in components
            ),
            tuple(rotated_terms),
            tuple(transpose(psi[t]) for t in order),
            tuple(metrics[a] for a in axes),
            **kwargs,
        )
        return (
            *(jnp.transpose(outputs[c], inverse_axes) for c in inverse_components),
            *(jnp.transpose(outputs[3 + t], inverse_axes) for t in inverse_terms),
        )

    return call
