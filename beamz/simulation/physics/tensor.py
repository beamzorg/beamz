"""Centered full-tensor electric updates."""

from __future__ import annotations

import jax.numpy as jnp

from beamz.const import EPS_0
from beamz.constitutive import advance_e_tensor_dense, collocate_yee_component
from beamz.lattice import component_axis_offsets_3d
from beamz.simulation.material_blocks import material_values
from beamz.simulation.tensor_regions import TensorRegion


def advance_e_centered_tensor(
    fields,
    curls,
    inverse_diagonals,
    inverse_offdiagonal,
    components,
    dt,
    *,
    cell_widths=None,
):
    """Advance E with diagonal terms on E sites and cross terms on dual centers."""

    inverse_diagonals = tuple(material_values(v) for v in inverse_diagonals)
    scale = jnp.asarray(dt, dtype=fields[0].dtype) / jnp.asarray(
        EPS_0, dtype=fields[0].dtype
    )
    if isinstance(inverse_offdiagonal, TensorRegion):
        region = inverse_offdiagonal
        axes = ("z", "y", "x")[-fields[0].ndim :]
        bounds = tuple(
            (max(0, start - 1), min(count, stop + 1))
            for (start, stop), count in zip(
                region.bounds, region.node_shape, strict=True
            )
        )
        values = jnp.pad(
            region.values,
            tuple(
                (start - outer_start, outer_stop - stop)
                for (start, stop), (outer_start, outer_stop) in zip(
                    region.bounds, bounds, strict=True
                )
            )
            + ((0, 0),),
        )
        slices = tuple(
            tuple(
                slice(
                    start, stop - int(component_axis_offsets_3d(component)[axis] == 0.5)
                )
                for axis, (start, stop) in zip(axes, bounds, strict=True)
            )
            for component in components
        )
        widths = cell_widths
        if widths is not None:
            widths = list(widths)
            for axis, (start, stop) in zip(axes, bounds, strict=True):
                index = "xyz".index(axis)
                if jnp.ndim(widths[index]) and jnp.size(widths[index]):
                    widths[index] = widths[index][start : stop - 1]
            widths = tuple(widths)
        local_curls = tuple(
            curl[part] for curl, part in zip(curls, slices, strict=True)
        )
        local = _tensor_cross_curls(local_curls, values, components, widths)
        return tuple(
            field
            + scale
            * (
                diagonal * curl
                + jnp.pad(
                    value,
                    tuple(
                        (part.start, size - part.stop)
                        for part, size in zip(parts, field.shape, strict=True)
                    ),
                )
            )
            for field, diagonal, curl, parts, value in zip(
                fields, inverse_diagonals, curls, slices, local, strict=True
            )
        )
    return advance_e_tensor_dense(
        fields,
        curls,
        inverse_diagonals,
        inverse_offdiagonal,
        components,
        dt,
        cell_widths=cell_widths,
    )


def _tensor_cross_curls(curls, inverse_offdiagonal, components, cell_widths):
    """Sum cross terms at their shared support before interpolating once per E."""
    node_shape = inverse_offdiagonal.shape[:-1]
    centered_curls = tuple(
        collocate_yee_component(
            curl, source, "Node", node_shape, cell_widths=cell_widths
        )
        for curl, source in zip(curls, components, strict=True)
    )
    cross_component = {(0, 1): 0, (0, 2): 1, (1, 2): 2}
    updated = []
    for row, (curl, target) in enumerate(zip(curls, components, strict=True)):
        centered = sum(
            inverse_offdiagonal[..., cross_component[(min(row, col), max(row, col))]]
            * value
            for col, value in enumerate(centered_curls)
            if col != row
        )
        updated.append(
            collocate_yee_component(
                centered, "Node", target, curl.shape, cell_widths=cell_widths
            )
        )
    return tuple(updated)
