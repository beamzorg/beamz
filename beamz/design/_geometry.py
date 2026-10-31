"""Design geometry interpretation; native Scene serialization remains a boundary policy."""

import numpy as np


def polygon_rings(structure):
    """Return the exterior and holes in the physical layout plane."""
    return tuple(
        tuple((float(x), float(y)) for x, y, *_ in path)
        for path in (
            getattr(structure, "vertices", ()),
            *getattr(structure, "interiors", ()),
        )
    )


def planar_geometry(structure, *, repair=False):
    from shapely.geometry import Polygon
    from shapely.geometry.polygon import orient

    shell, *holes = polygon_rings(structure)
    if len(shell) < 3:
        return None
    geometry = Polygon(shell, holes)
    if repair and not geometry.is_valid:
        geometry = geometry.buffer(0)
    return orient(geometry, sign=1.0) if not geometry.is_empty else None


def extrusion_bounds(structure, depth):
    """Zero-depth planar specifications inherit their design's extrusion."""
    lower = float(structure.z)
    return lower, lower + float(structure.depth or depth)


def structure_bounds(structure, *, two_dimensional=False):
    from beamz.design.structures import Box, Sphere

    if isinstance(structure, Box):
        return structure.lower, structure.upper
    if isinstance(structure, Sphere):
        center = np.asarray(structure.center)
        return tuple(center - structure.radius), tuple(center + structure.radius)
    shell, *_ = polygon_rings(structure)
    points = np.asarray(shell)
    z0, z1 = (0.0, 1.0) if two_dimensional else extrusion_bounds(structure, 0.0)
    return (*points.min(axis=0), z0), (*points.max(axis=0), z1)
