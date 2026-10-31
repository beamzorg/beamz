"""Restore acquisition coordinates only at legacy/synthetic result boundaries."""

from beamz.lattice import common_grid_shape_3d, yee_plane_coordinates_3d


def retained_sample_region(metadata, monitor, region=None):
    if (
        monitor is None
        or not metadata.is_3d
        or getattr(monitor, "region", "slice") == "domain"
        or (region is not None and region.sample_coordinates)
    ):
        return region
    return _restore_sample_region(metadata, monitor, region)


def _restore_sample_region(metadata, monitor, region):
    grid = metadata.grid
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
