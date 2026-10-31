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


def fit_array_to_shape(arr, target_shape, *, pad_value=0.0):
    """Crop or high-pad a JAX array to one static component shape."""
    return _fit_array_to_shape(arr, target_shape, pad_value=pad_value, xp=jnp)


def _fit_array_to_shape(arr, target_shape, *, pad_value, xp):
    """Share static crop/pad geometry while preserving the caller's array backend."""
    arr = xp.asarray(arr)
    target_shape = tuple(int(value) for value in target_shape)
    if arr.shape == target_shape:
        return arr
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
        xp.pad(out, padding, constant_values=pad_value)
        if any(high for _, high in padding)
        else out
    )


def collocate_yee_component(
    value, source: str, target: str, target_shape, *, cell_widths=None
):
    """Average electric/node supports with reciprocal control-volume weights.

    ``cell_widths`` is an optional (x, y, z) tuple of primal widths. Center-to-node
    averaging is the volume-weighted adjoint of node-to-center averaging. This
    preserves constants and symmetry of the coupled constitutive operator in
    the Yee volume inner product, including on graded meshes.
    """

    out = jnp.asarray(value)
    axes = ("z", "y", "x")[-out.ndim :]
    source_offsets = (
        {"z": 0.0, "y": 0.0, "x": 0.0}
        if source == "Node"
        else component_axis_offsets_3d(source)
    )
    target_offsets = (
        {"z": 0.0, "y": 0.0, "x": 0.0}
        if target == "Node"
        else component_axis_offsets_3d(target)
    )
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
):
    """Apply the lossless dense Yee tensor update shared by sources and propagation."""
    scale = jnp.asarray(dt, dtype=fields[0].dtype) / jnp.asarray(
        EPS_0, dtype=fields[0].dtype
    )
    node_shape = inverse_offdiagonal.shape[:-1]
    centered_curls = tuple(
        collocate_yee_component(
            curl, source, "Node", node_shape, cell_widths=cell_widths
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
                centered, "Node", target, field.shape, cell_widths=cell_widths
            )
        updated.append(field + scale * coupled)
    return tuple(updated)
