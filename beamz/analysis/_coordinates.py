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
    yee_plane_coordinates_3d,
)


def field_coordinates(metadata, component, shape):
    """Return component coordinates in metres in the public frame."""
    grid = metadata.grid
    if grid is not None:
        coordinates = component_coordinates_rectilinear(
            component,
            grid,
            plane=None if len(shape) == 3 else metadata.plane_2d,
            polarization=metadata.polarization_2d,
        )
    else:
        raw_shape = metadata.fields.grid_shape
        if len(shape) == 3:
            coordinates = component_coordinates_3d_um(
                component,
                raw_shape if len(raw_shape) == 3 else shape,
                metadata.resolution * 1e6,
            )
        else:
            coordinates = component_coordinates_2d_um(
                component,
                raw_shape,
                metadata.resolution * 1e6,
                metadata.plane_2d,
                metadata.polarization_2d,
            )
        coordinates = {
            axis: np.asarray(values, dtype=float) * 1e-6
            for axis, values in coordinates.items()
        }
    return coordinates_in_public_frame(coordinates, metadata.coordinate_offset)


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
        grid=grid,
    )
    return region, np.asarray(coord0, dtype=float), np.asarray(coord1, dtype=float)


__all__ = ["field_coordinates", "monitor_plane_coordinates_3d"]
