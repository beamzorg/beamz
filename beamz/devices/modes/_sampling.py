"""Physical sampling of the mode solver's native Yee components."""

from dataclasses import replace

import numpy as np

from beamz.lattice import component_axis_offsets_3d


def native_coordinates(component, axes, edges):
    """Native solves omit the final boundary node on node-aligned axes."""
    offsets = component_axis_offsets_3d(component)
    return tuple(
        0.5 * (edge[:-1] + edge[1:]) if offsets[axis] == 0.5 else edge[:-1]
        for axis, edge in zip(axes, edges, strict=True)
    )


def collocate_result(result, axes, edges):
    """Interpolate native Yee fields onto common transverse cell centres.

    Missing high nodes have the homogeneous PEC value zero. All node-aligned
    components are either tangential E or normal H at that boundary.
    """
    if (result.solver_info or {}).get("cell_centres"):
        return result
    fields = {}
    for name, data in result.fields.items():
        values = np.asarray(data)
        for axis, edge in zip(axes, edges, strict=True):
            dimension = 2 + result.axes.index(axis)
            if component_axis_offsets_3d(name)[axis] == 0:
                padding = [(0, 0)] * values.ndim
                padding[dimension] = (0, 1)
                extended = np.pad(values, padding)
                lo = np.take(extended, np.arange(len(edge) - 1), axis=dimension)
                hi = np.take(extended, np.arange(1, len(edge)), axis=dimension)
                values = 0.5 * (lo + hi)
        fields[name] = values
    return replace(
        result,
        fields=fields,
        solver_info={**(result.solver_info or {}), "cell_centres": True},
    )


def plane_material_samples(spec, component, values, fallback):
    """Extract propagation's component-specific materials on native solver sites."""
    shape = spec.scalar_permittivity.shape
    if component in values:
        array = np.asarray(values[component])
        dimension = {"z": 0, "y": 1, "x": 2}[spec.axis]
        index = min(spec.plane_index, array.shape[dimension] - 1)
        plane = np.take(array, index, axis=dimension)
        storage_axes = tuple(axis for axis in ("z", "y", "x") if axis != spec.axis)
        plane = np.transpose(
            plane, tuple(storage_axes.index(axis) for axis in spec.transverse_axes)
        )
        return plane[tuple(slice(0, size) for size in shape)]
    return cell_material_samples(fallback, component, spec.transverse_axes)


def cell_material_samples(values, component, axes):
    """Interpolate cell-centred material profiles onto native component sites."""
    plane = np.asarray(values)
    for dimension, axis in enumerate(axes):
        if component_axis_offsets_3d(component)[axis] == 0:
            previous = np.take(
                plane,
                np.maximum(np.arange(plane.shape[dimension]) - 1, 0),
                axis=dimension,
            )
            plane = 0.5 * (previous + plane)
    return plane


def launch_samples(field, component, spec):
    """Sample a cell-centred mode on the existing compact launch Yee support."""
    offsets = component_axis_offsets_3d(component)
    values = np.asarray(field)
    for dimension, axis in enumerate(spec.transverse_axes):
        count = spec.grid_shape[{"z": 0, "y": 1, "x": 2}[axis]]
        edge = (
            np.arange(count + 1) * spec.resolution
            if spec.grid is None
            else np.asarray(spec.grid.axis_edges(axis))
        )
        source = 0.5 * (edge[:-1] + edge[1:])
        target = edge[:-1] if offsets[axis] == 0 else source[:-1]
        values = np.apply_along_axis(
            lambda row, target=target, source=source: np.interp(
                target, source, row, left=0.0, right=0.0
            ),
            dimension,
            values,
        )
    return values
