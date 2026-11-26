"""Lossless electric constitutive coefficients used by propagation and ports."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from beamz.const import EPS_0
from beamz.lattice import component_axis_offsets_3d


def inverse_permittivity_components(values):
    """Invert packed (xx, yy, zz, xy, xz, yz) tensors; components are trailing."""
    packed = np.asarray(values)
    if packed.shape[0] != 6:
        raise ValueError("Full electric constitutive support requires six components")
    xx, yy, zz, xy, xz, yz = packed
    determinant = (
        xx * yy * zz + 2.0 * xy * xz * yz - xx * yz * yz - yy * xz * xz - zz * xy * xy
    )
    cofactors = (
        yy * zz - yz * yz,
        xx * zz - xz * xz,
        xx * yy - xy * xy,
        xz * yz - xy * zz,
        xy * yz - xz * yy,
        xy * xz - xx * yz,
    )
    return np.stack(cofactors, axis=-1) / determinant[..., None]


def electric_tensor_coefficients(tensors):
    """Return E-site inverse diagonals and shared-node inverse cross terms."""
    diagonal = tuple(
        inverse_permittivity_components(tensors["eps_" + axis])[..., index]
        for index, axis in enumerate("xyz")
    )
    cross = inverse_permittivity_components(tensors["eps_node"])[..., 3:]
    return diagonal, cross


def expand_packed(values) -> np.ndarray:
    """Expand compact ``(1|3|6, ...)`` symmetric tensors to six components."""

    values = np.asarray(values, dtype=np.float64)
    if values.shape[0] == 6:
        return values
    diagonal = np.broadcast_to(values, (3, *values.shape[1:]))
    return np.concatenate((diagonal, np.zeros((3, *values.shape[1:]))), axis=0)


def fit_array_to_shape(arr, target_shape, *, pad_value=0.0):
    """Crop or high-pad a JAX array to one static component shape."""
    arr = jnp.asarray(arr)
    target_shape = tuple(int(value) for value in target_shape)
    if arr.ndim != len(target_shape):
        raise ValueError(
            f"Cannot fit rank-{arr.ndim} array to rank-{len(target_shape)} shape"
        )
    out = arr[
        tuple(
            slice(0, min(int(size), target))
            for size, target in zip(arr.shape, target_shape, strict=True)
        )
    ]
    padding = tuple(
        (0, max(0, target - int(size)))
        for size, target in zip(out.shape, target_shape, strict=True)
    )
    return (
        jnp.pad(out, padding, constant_values=jnp.asarray(pad_value, dtype=arr.dtype))
        if any(high for _, high in padding)
        else out
    )


def periodic_seams(periodic_axes, periodic_phases, ndim: int) -> tuple:
    """``(public axis, phase)`` seams from storage-axis periodic metadata."""
    axes = ("z", "y", "x")[-ndim:]
    return tuple(
        (axes[int(a)], periodic_phases[int(a)] if periodic_phases else 1.0)
        for a in sorted(periodic_axes)
    )


def _support_offsets(support: str) -> dict[str, float]:
    """Yee offsets of a field component, a primal node, or a cell center."""
    if support == "Node":
        return {"z": 0.0, "y": 0.0, "x": 0.0}
    if support == "Cell":
        return {"z": 0.5, "y": 0.5, "x": 0.5}
    return component_axis_offsets_3d(support)


def collocate_yee_component(
    value, source: str, target: str, target_shape, *, cell_widths=None, seams=()
):
    """Average Yee, node, or cell supports with reciprocal control-volume weights.

    ``cell_widths`` is an optional (x, y, z) tuple of primal widths. Center-to-node
    averaging is the volume-weighted adjoint of node-to-center averaging. This
    preserves constants and symmetry of the coupled constitutive operator in
    the Yee volume inner product, including on graded meshes.

    ``seams`` lists ``(axis, phase)`` pairs for periodic/Bloch axes the arrays
    span completely (``axis`` is a public name ``"x"|"y"|"z"``). There the end
    node averages across the seam, ``(c[-1] / phase, c[0])``, and its duplicate
    is ``phase`` times the first node, matching the stored Bloch supports. On
    other axes the end node copies its single neighbor.
    """

    out = jnp.asarray(value)
    seam_phase = dict(seams)
    axes = ("z", "y", "x")[-out.ndim :]
    source_offsets = _support_offsets(source)
    target_offsets = _support_offsets(target)
    for axis_index, axis in enumerate(axes):
        source_offset = source_offsets[axis]
        target_offset = target_offsets[axis]
        if source_offset == target_offset:
            continue
        if source_offset == 0.0 and target_offset == 0.5:
            low = [slice(None)] * out.ndim
            high = [slice(None)] * out.ndim
            low[axis_index] = slice(0, -1)
            high[axis_index] = slice(1, None)
            out = 0.5 * (out[tuple(low)] + out[tuple(high)])
        elif source_offset == 0.5 and target_offset == 0.0:
            first = jnp.take(out, jnp.asarray([0]), axis=axis_index)
            last = jnp.take(out, jnp.asarray([-1]), axis=axis_index)
            low = [slice(None)] * out.ndim
            high = [slice(None)] * out.ndim
            low[axis_index] = slice(0, -1)
            high[axis_index] = slice(1, None)
            widths = None if cell_widths is None else cell_widths["xyz".index(axis)]
            if widths is None or jnp.ndim(widths) == 0 or jnp.size(widths) == 0:
                middle = 0.5 * (out[tuple(low)] + out[tuple(high)])
            else:
                widths = jnp.asarray(widths, dtype=out.real.dtype)
                if widths.shape != (out.shape[axis_index],):
                    raise ValueError(
                        "Constitutive cell widths must match the electric support"
                    )
                shape = [1] * out.ndim
                shape[axis_index] = widths.size - 1
                weight = (widths[:-1] / (widths[:-1] + widths[1:])).reshape(shape)
                middle = weight * out[tuple(low)] + (1.0 - weight) * out[tuple(high)]
            if axis in seam_phase:
                phase = seam_phase[axis]
                if widths is None or jnp.ndim(widths) == 0 or jnp.size(widths) == 0:
                    edge = 0.5 * (last / phase + first)
                else:
                    edge = (widths[-1] * last / phase + widths[0] * first) / (
                        widths[-1] + widths[0]
                    )
                first, last = edge, phase * edge
            out = jnp.concatenate((first, middle, last), axis=axis_index)
        else:  # pragma: no cover - canonical Yee offsets are binary
            raise ValueError("Unsupported Yee colocation offset.")
    return fit_array_to_shape(out, target_shape)


def advance_e_tensor_dense(
    fields,
    curls,
    inverse_diagonals,
    inverse_offdiagonal,
    components,
    dt,
    *,
    cell_widths=None,
    seams=(),
):
    """Apply the lossless dense Yee tensor update shared by sources and propagation."""
    scale = jnp.asarray(dt, dtype=fields[0].dtype) / jnp.asarray(
        EPS_0, dtype=fields[0].dtype
    )
    node_shape = inverse_offdiagonal.shape[:-1]
    centered_curls = tuple(
        collocate_yee_component(
            curl, source, "Node", node_shape, cell_widths=cell_widths, seams=seams
        )
        for curl, source in zip(curls, components, strict=True)
    )
    cross_component = {(0, 1): 0, (0, 2): 1, (1, 2): 2}
    updated = []
    for row, (field, diagonal, target) in enumerate(
        zip(fields, inverse_diagonals, components, strict=True)
    ):
        coupled = diagonal * curls[row]
        for column, centered_curl in enumerate(centered_curls):
            if row == column:
                continue
            pair = (min(row, column), max(row, column))
            centered = inverse_offdiagonal[..., cross_component[pair]] * centered_curl
            coupled = coupled + collocate_yee_component(
                centered,
                "Node",
                target,
                field.shape,
                cell_widths=cell_widths,
                seams=seams,
            )
        updated.append(field + scale * coupled)
    return tuple(updated)
