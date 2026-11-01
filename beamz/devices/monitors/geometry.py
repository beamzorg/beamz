"""Bind acquisition coordinates and restore them for legacy/synthetic results."""

import numpy as np

from beamz.devices.monitors.monitors import FieldMonitor
from beamz.lattice import common_grid_shape_3d, yee_plane_coordinates_3d


def planar_sample_region_2d(monitor, grid, resolution, shape, region=None):
    """Bind the cell-center coordinates used by a planar 2D acquisition."""
    region = region or monitor.get_snapped_region(
        grid=grid,
        dx=resolution,
        dy=resolution,
        dz=resolution,
        field_shape=(1, *shape),
    )
    assert region is not None
    coordinates = []
    for axis, count in zip(("y", "x"), shape, strict=True):
        interval = region.axis_interval(axis)
        assert interval is not None
        centers = (
            grid.centers(axis)
            if grid is not None
            else (np.arange(count) + 0.5) * resolution
        )
        coordinates.append(centers[interval.start : interval.stop])
    return region.with_samples(coordinates)


def retained_sample_region(metadata, monitor, region=None):
    if (
        monitor is None
        or (
            not metadata.is_3d
            and not (isinstance(monitor, FieldMonitor) and monitor.is_3d)
        )
        or getattr(monitor, "region", "slice") == "domain"
        or (region is not None and region.sample_coordinates)
    ):
        return region
    return _restore_sample_region(metadata, monitor, region)


def _restore_sample_region(metadata, monitor, region):
    grid = metadata.grid
    if not metadata.is_3d:
        return planar_sample_region_2d(
            monitor, grid, metadata.resolution, metadata.fields.grid_shape, region
        )
    # Shifted uniform planes also use realized sampling, just like the compiler.
    active = (
        grid
        if grid is not None
        and (grid.metric_kind != "isotropic_uniform" or grid.origin != (0.0, 0.0, 0.0))
        else None
    )
    region = region or monitor.get_snapped_region(
        grid=active,
        dx=metadata.resolution,
        dy=metadata.resolution,
        dz=metadata.resolution,
        field_shape=common_grid_shape_3d(metadata.fields),
    )
    if region is None:
        raise ValueError("A three-dimensional acquisition requires a snapped plane.")
    return region.with_samples(
        yee_plane_coordinates_3d(
            monitor.center, monitor.size, monitor.plane_normal, region, grid=active
        )
    )
