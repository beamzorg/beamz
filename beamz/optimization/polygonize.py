from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.ndimage import map_coordinates
from shapely.geometry import GeometryCollection, box
from shapely.geometry import Polygon as ShapelyPolygon
from shapely.ops import unary_union

from beamz.design.structures import Polygon

from ._validation import fixed_density_masks


@dataclass
class _ContourNode:
    polygon: ShapelyPolygon
    children: list[int] = field(default_factory=list)


def _iter_shapely_polygons(geometry):
    if geometry.is_empty:
        return
    if geometry.geom_type == "Polygon":
        yield geometry
        return
    if hasattr(geometry, "geoms"):
        for part in geometry.geoms:
            yield from _iter_shapely_polygons(part)


def _build_parent_map(polygons):
    parents: list[int | None] = [None] * len(polygons)
    for idx, polygon in enumerate(polygons):
        point = polygon.representative_point()
        best_parent = None
        best_area = None
        for parent_idx in range(idx + 1, len(polygons)):
            candidate = polygons[parent_idx]
            if not candidate.covers(point):
                continue
            area = candidate.area
            if best_area is None or area < best_area:
                best_parent = parent_idx
                best_area = area
        parents[idx] = best_parent
    return parents


def _build_occupied_geometry(nodes, node_idx):
    geometry = nodes[node_idx].polygon
    if not nodes[node_idx].children:
        return geometry
    child_geometry = unary_union(
        [
            _build_occupied_geometry(nodes, child_idx)
            for child_idx in nodes[node_idx].children
        ]
    )
    return geometry.difference(child_geometry).buffer(0)


def _pixel_geometry(mask, *, x0, y0, dx, dy):
    """Union horizontal runs instead of allocating a polygon for every cell."""
    rectangles = []
    for row, values in enumerate(mask):
        edges = np.flatnonzero(np.diff(np.pad(values.astype(np.int8), 1)))
        for start, stop in edges.reshape(-1, 2):
            rectangles.append(
                box(x0 + start * dx, y0 + row * dy, x0 + stop * dx, y0 + (row + 1) * dy)
            )
    return unary_union(rectangles)


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
    polygons.sort(key=lambda polygon: polygon.area)
    parents = _build_parent_map(polygons)
    nodes = [_ContourNode(polygon=polygon) for polygon in polygons]
    roots = []
    for idx, parent_idx in enumerate(parents):
        if parent_idx is None:
            roots.append(idx)
        else:
            nodes[parent_idx].children.append(idx)
    geometry = unary_union(
        [_build_occupied_geometry(nodes, root_idx) for root_idx in roots]
    )
    ny, nx = array.shape
    return geometry.intersection(box(x0, y0, x0 + nx * dx, y0 + ny * dy))


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


def shapely_geometry_to_polygons(geometry, *, material, min_area=0.0):
    """Convert Shapely geometry into BEAMZ polygon structures.

    Parameters
    ----------
    geometry : shapely.Geometry
        Polygon, multipolygon, or geometry collection to convert.
    material : Material
        Material assigned to every returned structure.
    min_area : float, default=0
        Minimum retained exterior and hole area in square metres.

    Returns
    -------
    list of Polygon
        Immutable BEAMZ polygons with interior rings represented as holes.
    """
    polygons = []
    min_area = float(max(0.0, min_area))
    for polygon in _iter_shapely_polygons(geometry):
        if polygon.area < min_area:
            continue
        interiors = []
        for ring in polygon.interiors:
            hole = ShapelyPolygon(ring)
            if hole.area >= min_area:
                interiors.append(list(ring.coords[:-1]))
        polygons.append(
            Polygon(
                vertices=list(polygon.exterior.coords[:-1]),
                interiors=interiors,
                material=material,
            )
        )
    return polygons


def density_to_polygons(
    density,
    *,
    material,
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
    """Return BeamZ polygons; see :func:`density_to_shapely_geometry` for options.

    Coordinates and spacings are in metres. Interior rings remain holes.
    """
    geometry = density_to_shapely_geometry(
        density,
        level=level,
        x0=x0,
        y0=y0,
        dx=dx,
        dy=dy,
        min_area=min_area,
        mode=mode,
        sample_spacing=sample_spacing,
        region_mask=region_mask,
        fixed_solid=fixed_solid,
        fixed_void=fixed_void,
    )
    return shapely_geometry_to_polygons(geometry, material=material)
