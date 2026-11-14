"""Extract and traverse contour geometry."""

from __future__ import annotations

import numpy as np
from scipy.ndimage import map_coordinates
from shapely.geometry import GeometryCollection, box
from shapely.geometry import Polygon as ShapelyPolygon


def _iter_shapely_polygons(geometry):
    if geometry.is_empty:
        return
    if geometry.geom_type == "Polygon":
        yield geometry
        return
    if hasattr(geometry, "geoms"):
        for part in geometry.geoms:
            yield from _iter_shapely_polygons(part)


def _contour_geometry(array, *, level, x0, y0, dx, dy, sample_spacing):
    from skimage.measure import find_contours

    # Repeat boundary samples beyond the cell-edge domain. Contours close outside
    # that domain; clipping below puts solid edges exactly on the physical frame.
    extended = np.pad(array, 1, mode="edge")
    fx = fy = 1
    if sample_spacing is not None:
        fx = max(1, int(np.ceil(dx / sample_spacing)))
        fy = max(1, int(np.ceil(dy / sample_spacing)))
        yy, xx = np.meshgrid(
            np.arange((array.shape[0] + 1) * fy + 1) / fy,
            np.arange((array.shape[1] + 1) * fx + 1) / fx,
            indexing="ij",
        )
        extended = map_coordinates(extended, [yy, xx], order=1, mode="nearest")
    padded = np.pad(extended, 1, constant_values=level - 1.0)
    contours = find_contours(
        padded, level=level, fully_connected="low", positive_orientation="low"
    )
    polygons = []
    for contour in contours:
        if contour.shape[0] < 3:
            continue
        coords = [
            (x0 + ((col - 1) / fx - 0.5) * dx, y0 + ((row - 1) / fy - 0.5) * dy)
            for row, col in contour
        ]
        polygons.extend(_iter_shapely_polygons(ShapelyPolygon(coords).buffer(0)))
    if not polygons:
        return GeometryCollection()
    # Contour loops are disjoint or nested. Even/odd fill directly preserves
    # islands, holes, and islands inside holes without a separate containment tree.
    geometry = GeometryCollection()
    for polygon in polygons:
        geometry = geometry.symmetric_difference(polygon)
    ny, nx = array.shape
    return geometry.intersection(box(x0, y0, x0 + nx * dx, y0 + ny * dy))
