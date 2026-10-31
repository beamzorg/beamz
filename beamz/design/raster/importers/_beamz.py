from __future__ import annotations

from typing import Any

import numpy as np

from beamz._cache_tokens import cache_token
from beamz.design._geometry import extrusion_bounds, polygon_rings
from beamz.design.structures import Polygon, Shape

from ..schema import Box, Cylinder, Material, Object, Scene


def _material(value: Any) -> Material:
    if isinstance(value, Material):
        return value
    epsilon_r, mu_r, conductivity = value.get_sample(0.0, 0.0, None)
    return Material(epsilon_r=epsilon_r, mu_r=mu_r, conductivity=conductivity)


def _same_coordinate(first, second):
    tolerance = 16.0 * np.finfo(float).eps * np.maximum(abs(first), abs(second))
    return abs(first - second) <= tolerance


def _extend_clipped_polygon(vertices, *, design_size, padded_size):
    """Extend polygon edges lying on a padded high-side design boundary."""
    result = np.asarray(vertices, dtype=float).copy()
    for axis in range(2):
        if padded_size[axis] != design_size[axis]:
            boundary = _same_coordinate(result[:, axis], design_size[axis])
            edges = boundary & (np.roll(boundary, 1) | np.roll(boundary, -1))
            result[edges, axis] = padded_size[axis]
    return tuple(map(tuple, result))


def _padded_upper(upper, physical_size, padded_size):
    return tuple(
        padded if _same_coordinate(value, physical) else value
        for value, physical, padded in zip(
            upper, physical_size, padded_size, strict=True
        )
    )


def _raster_geometry(structure: Shape, design, extrusion_depth, padded_size):
    """Realize domain-dependent extrusion and terminal padding on canonical solids."""
    physical_size = (design.width, design.height, design.depth)
    if structure.shape_kind == "box":
        lower, upper = structure.lower, structure.upper
        if design.depth == 0:
            lower, upper = (*lower[:2], 0.0), (*upper[:2], extrusion_depth)
        elif lower[2] == upper[2]:
            lower, upper = (*lower[:2], 0.0), (*upper[:2], padded_size[2])
        return Box(lower, _padded_upper(upper, physical_size, padded_size))
    if structure.shape_kind == "sphere":
        if design.depth:
            return structure
        squared_radius = structure.radius**2 - structure.center[2] ** 2
        return (
            Cylinder(
                structure.center[:2], np.sqrt(squared_radius), 0.0, extrusion_depth
            )
            if squared_radius > 0
            else None
        )
    lower, upper = (
        (0.0, extrusion_depth)
        if design.depth == 0
        else extrusion_bounds(structure, design.depth)
    )
    if design.depth and _same_coordinate(upper, design.depth):
        upper = padded_size[2]
    if structure.shape_kind == "rectangle" and not structure.sidewall_angle:
        x, y, *_ = structure.vertices[0]
        return Box(
            (x, y, lower),
            _padded_upper(
                (x + structure.width, y + structure.height, upper),
                physical_size,
                padded_size,
            ),
        )
    shell, *holes = polygon_rings(structure)
    if len(shell) < 3:
        raise TypeError(f"Unsupported BeamZ structure {structure.shape_kind}.")
    vertices = _extend_clipped_polygon(
        shell, design_size=physical_size[:2], padded_size=padded_size[:2]
    )
    return Polygon(
        vertices,
        interiors=tuple(holes),
        z=lower,
        depth=upper - lower,
        sidewall_angle=structure.sidewall_angle,
        width_to_z=structure.width_to_z,
    )


def from_beamz(
    design: Any,
    *,
    two_dimensional_depth: float = 1.0,
    padded_size: tuple[float, float, float] | None = None,
) -> Scene:
    """Convert a BeamZ ``Design`` without invoking its existing rasterizer.

    Painter order becomes increasing object priority. A 2D BeamZ design is
    extruded through ``two_dimensional_depth``; that depth cancels from area
    fractions when a single z cell spanning the same interval is used.
    """
    if design.depth == 0 and two_dimensional_depth <= 0:
        raise ValueError("two_dimensional_depth must be positive.")
    padded_size = (
        (design.width, design.height, design.depth)
        if padded_size is None
        else padded_size
    )
    materials = []
    material_ids = {}

    def add_material(value):
        material = _material(value)
        key = cache_token(material.cache_spec())
        if key not in material_ids:
            material_ids[key] = len(materials)
            materials.append(material)
        return material_ids[key]

    background_id = add_material(design.background)
    objects = []
    for index, structure in enumerate(design.structures, 1):
        material_id = add_material(structure.material)
        geometry = _raster_geometry(
            structure, design, two_dimensional_depth, padded_size
        )
        if geometry is not None:
            objects.append(Object(geometry, material_id, index, index))
    return Scene(tuple(materials), tuple(objects), background_id)
