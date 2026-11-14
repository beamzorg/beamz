"""Convert density arrays to physical geometry."""

from __future__ import annotations

import numpy as np
from shapely.geometry import Polygon as ShapelyPolygon
from shapely.ops import unary_union

from beamz.optimization.design.geometry.contour import (
    _contour_geometry,
    _iter_shapely_polygons,
)
from beamz.optimization.design.geometry.pixels import _pixel_geometry
from beamz.optimization.design.validation import fixed_density_masks


def _filter_area(geometry, min_area):
    polygons = []
    for poly in _iter_shapely_polygons(geometry):
        if poly.area >= min_area:
            polygons.append(
                ShapelyPolygon(
                    poly.exterior,
                    [
                        ring
                        for ring in poly.interiors
                        if ShapelyPolygon(ring).area >= min_area
                    ],
                )
            )
    return unary_union(polygons)


def density_to_shapely_geometry(
    density,
    *,
    level=0.5,
    x0=0.0,
    y0=0.0,
    dx=1.0,
    dy=None,
    min_area=0.0,
    mode="contour",
    sample_spacing=None,
    region_mask=None,
    fixed_solid=None,
    fixed_void=None,
):
    """Convert cell-centered density in ``(y, x)`` order to physical geometry.

    ``x0, y0`` are the lower cell-edge coordinates, and ``dx, dy`` are cell
    sizes, all in metres. The domain ends at ``x0 + nx*dx, y0 + ny*dy``.

    ``mode="pixel_exact"`` unions cells whose density is strictly above ``level``.
    ``mode="contour"`` (default) extracts linearly interpolated contours with
    edge-repeat extension and clips them to the exact domain. ``sample_spacing``
    optionally sets the maximum spacing in metres for bilinear reconstruction
    before contouring; original samples are retained. This changes geometry,
    not the simulation mesh, and does not guarantee electromagnetic accuracy.

    ``region_mask`` optionally clips to an exact union of cells, while density
    outside it supplies contour context. ``fixed_solid`` and ``fixed_void`` are
    disjoint boolean masks of cells whose exact footprints are preserved after
    contouring. ``min_area`` removes small islands/holes (square metres) before
    enforcing these fixed footprints. No implicit dilation or smoothing is used.
    """
    array = np.asarray(density, dtype=float)
    if array.ndim != 2 or 0 in array.shape:
        raise ValueError("density_to_shapely_geometry expects a nonempty 2D array")
    if not np.isfinite(array).all():
        raise ValueError("density must be finite.")
    dy = float(dx if dy is None else dy)
    dx, x0, y0, level, min_area = map(float, (dx, x0, y0, level, min_area))
    if not np.isfinite([dx, dy, x0, y0, level, min_area]).all():
        raise ValueError("Coordinates, spacings, level and min_area must be finite.")
    if min(dx, dy) <= 0 or min_area < 0:
        raise ValueError("Spacings must be positive and min_area nonnegative.")
    if mode not in ("contour", "pixel_exact"):
        raise ValueError("mode must be 'contour' or 'pixel_exact'.")
    if sample_spacing is not None:
        if not np.isfinite(sample_spacing) or sample_spacing <= 0:
            raise ValueError("sample_spacing must be finite and positive.")
        if mode != "contour":
            raise ValueError("sample_spacing is only supported in contour mode.")
    solid, void = fixed_density_masks(fixed_solid, fixed_void, array.shape)
    if solid is not None:
        array = np.where(solid, 1.0, array)
    if void is not None:
        array = np.where(void, 0.0, array)
    grid = dict(x0=x0, y0=y0, dx=dx, dy=dy)
    if mode == "pixel_exact":
        geometry = _pixel_geometry(array > level, **grid)
    else:
        geometry = _contour_geometry(
            array, level=level, sample_spacing=sample_spacing, **grid
        )
    if min_area:
        geometry = _filter_area(geometry, min_area)
    if solid is not None:
        geometry = geometry.union(_pixel_geometry(solid, **grid))
    if void is not None:
        geometry = geometry.difference(_pixel_geometry(void, **grid))
    if region_mask is not None:
        mask, _ = fixed_density_masks(region_mask, None, array.shape)
        geometry = geometry.intersection(_pixel_geometry(mask, **grid))
    # Intersections can leave zero-area lines at a clipping boundary.
    return unary_union(list(_iter_shapely_polygons(geometry)))
