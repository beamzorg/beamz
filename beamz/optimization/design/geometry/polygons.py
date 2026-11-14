"""Convert density geometry into design polygons."""

from __future__ import annotations

from shapely.geometry import Polygon as ShapelyPolygon

from beamz.design.geometry.primitives import Polygon
from beamz.optimization.design.geometry.contour import _iter_shapely_polygons
from beamz.optimization.design.geometry.density import density_to_shapely_geometry


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


def density_to_polygons(density, *, material, **options):
    """Return BeamZ polygons; see :func:`density_to_shapely_geometry` for options.

    Coordinates and spacings are in metres. Interior rings remain holes.
    All geometry options are forwarded to the authoritative conversion API.
    """
    geometry = density_to_shapely_geometry(density, **options)
    return shapely_geometry_to_polygons(geometry, material=material)
