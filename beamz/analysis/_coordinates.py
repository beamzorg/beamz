"""Shared coordinate reconstruction for detached monitor analysis."""

from __future__ import annotations

import numpy as np

from beamz.devices._placement import SnappedRegion, snap_plane_region_grid
from beamz.lattice import (
    common_grid_shape_3d,
    component_coordinates_2d_um,
    component_coordinates_3d_um,
    component_coordinates_rectilinear,
    coordinates_in_public_frame,
    plane_axes_3d,
    yee_plane_coordinates_3d,
)

# Field arrays are stored as (z, y, x). Public offsets and device tuples are (x, y, z).
_STORAGE_AXIS = {"z": 0, "y": 1, "x": 2}
_XYZ_INDEX = {"x": 0, "y": 1, "z": 2}
_PLANE_AXES = {axis: plane_axes_3d(axis) for axis in "xyz"}


def monitor_plane_coordinates_3d(
    metadata,
    monitor,
    *,
    region: SnappedRegion | None = None,
) -> tuple[SnappedRegion, np.ndarray, np.ndarray]:
    """Reconstruct the exact compiled coordinates for a 3D monitor plane."""
    grid = getattr(metadata, "grid", None)
    if grid is not None:
        region = region or snap_plane_region_grid(
            center=monitor.center,
            size=monitor.size,
            plane_normal=monitor.plane_normal,
            grid=grid,
        )
        coord0, coord1 = yee_plane_coordinates_3d(
            monitor.center,
            monitor.size,
            monitor.plane_normal,
            region,
            grid=grid,
        )
    else:
        if region is None:
            fields = getattr(metadata, "fields", None)
            base_shape = common_grid_shape_3d(fields)
            resolution = float(metadata.resolution)
            region = monitor.get_snapped_region(
                dx=resolution,
                dy=resolution,
                dz=resolution,
                field_shape=base_shape,
            )
            if region is None:
                raise ValueError("3D analysis plane requires a snapped monitor region.")
        coord0, coord1 = yee_plane_coordinates_3d(
            monitor.center,
            monitor.size,
            monitor.plane_normal,
            region,
        )
    return region, np.asarray(coord0, dtype=float), np.asarray(coord1, dtype=float)


def _sample_extent(values, *, fallback_step):
    """Half-step bounds around ordered sample coordinates."""
    coordinates = np.asarray(values, dtype=float)
    if coordinates.size == 0:
        raise ValueError("Recorded field coordinates are empty.")
    step = (
        float(np.nanmedian(np.diff(coordinates)))
        if coordinates.size > 1
        else float(fallback_step)
    )
    return (
        float(coordinates[0] - 0.5 * step),
        float(coordinates[-1] + 0.5 * step),
    )


def _public_component_coordinates(metadata, component, *, is_3d, grid_shape=None):
    """Return one component's Yee coordinates in the public frame, in metres."""
    grid = getattr(metadata, "grid", None)
    shape = metadata.fields.grid_shape if grid_shape is None else grid_shape
    resolution = float(metadata.resolution)
    polarization = getattr(metadata, "polarization_2d", "tm")
    if grid is not None:
        coords = component_coordinates_rectilinear(
            component,
            grid,
            plane=None if is_3d else metadata.plane_2d,
            polarization=polarization,
        )
    else:
        samples_um = (
            component_coordinates_3d_um(component, shape, resolution * 1e6)
            if is_3d
            else component_coordinates_2d_um(
                component,
                shape,
                resolution * 1e6,
                metadata.plane_2d,
                polarization,
            )
        )
        coords = {
            axis: np.asarray(values, dtype=float) * 1e-6
            for axis, values in samples_um.items()
        }
    return coordinates_in_public_frame(
        coords, getattr(metadata, "coordinate_offset", (0.0, 0.0, 0.0))
    )


__all__ = ["monitor_plane_coordinates_3d"]
