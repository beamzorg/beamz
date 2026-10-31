from __future__ import annotations

from typing import Any

import numpy as np

from beamz._cache_tokens import HashToken, cache_token
from beamz.design._geometry import extrusion_bounds, polygon_rings, structure_bounds

from ..schema import (
    Box,
    Cylinder,
    Material,
    Object,
    Polygon,
    Scene,
    Sphere,
    _extrude_polygon,
)


def _material(value: Any) -> Material:
    if isinstance(value, Material):
        return value
    epsilon_r, mu_r, conductivity = value.get_sample(0.0, 0.0, None)
    return Material(
        epsilon_r=epsilon_r,
        mu_r=mu_r,
        conductivity=conductivity,
    )


def _same_coordinate(first: float, second: float) -> bool:
    tolerance = 16.0 * np.finfo(float).eps * max(abs(first), abs(second))
    return abs(first - second) <= tolerance


def _extend_clipped_polygon(
    vertices: tuple[tuple[float, float], ...],
    *,
    design_size: tuple[float, float],
    padded_size: tuple[float, float],
) -> tuple[tuple[float, float], ...]:
    """Extend polygon edges lying on a padded high-side design boundary."""

    result = [list(point) for point in vertices]
    for axis in range(2):
        if padded_size[axis] == design_size[axis]:
            continue
        on_boundary = [
            _same_coordinate(point[axis], design_size[axis]) for point in vertices
        ]
        for index in range(len(vertices)):
            if on_boundary[index] and (
                on_boundary[index - 1] or on_boundary[(index + 1) % len(vertices)]
            ):
                result[index][axis] = padded_size[axis]
    return tuple((point[0], point[1]) for point in result)


def _z_bounds(structure, design, extrusion_depth, padded_size):
    if design.depth == 0:
        return 0.0, float(extrusion_depth)
    lower, upper = extrusion_bounds(structure, design.depth)
    return lower, padded_size[2] if _same_coordinate(upper, design.depth) else upper


def _padded_upper(
    upper: tuple[float, float, float],
    physical_size: tuple[float, float, float],
    padded_size: tuple[float, float, float],
) -> tuple[float, float, float]:
    def coordinate(axis: int) -> float:
        return (
            padded_size[axis]
            if _same_coordinate(upper[axis], physical_size[axis])
            else upper[axis]
        )

    return coordinate(0), coordinate(1), coordinate(2)


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
    physical_size = (float(design.width), float(design.height), float(design.depth))
    padded_size = physical_size if padded_size is None else padded_size

    materials: list[Material] = []
    material_ids: dict[HashToken, int] = {}

    def add_material(value: Any) -> int:
        material = _material(value)
        key = cache_token(material.cache_spec())
        if key not in material_ids:
            material_ids[key] = len(materials)
            materials.append(material)
        return material_ids[key]

    background_id = add_material(design.background)
    objects = []
    for index, structure in enumerate(design.structures, start=1):
        from beamz.design.structures import Box as BeamZBox
        from beamz.design.structures import Rectangle as BeamZRectangle
        from beamz.design.structures import Sphere as BeamZSphere

        material_id = add_material(structure.material)

        def add(
            geometry: Any,
            _material_id: int = material_id,
            _index: int = index,
        ) -> None:
            objects.append(Object(geometry, _material_id, _index, _index))

        if isinstance(structure, BeamZBox):
            lower_values, upper_values = structure_bounds(structure)
            lower = (lower_values[0], lower_values[1], lower_values[2])
            upper = (upper_values[0], upper_values[1], upper_values[2])
            if design.depth == 0:
                lower = (lower[0], lower[1], 0.0)
                upper = (upper[0], upper[1], float(two_dimensional_depth))
            elif lower[2] == upper[2]:
                lower = (lower[0], lower[1], 0.0)
                upper = (upper[0], upper[1], padded_size[2])
            geometry = Box(lower, _padded_upper(upper, physical_size, padded_size))
            add(geometry)
            continue
        if isinstance(structure, BeamZSphere):
            center = np.asarray(structure.center, dtype=np.float64)
            radius = float(structure.radius)
            if design.depth == 0:
                squared_cross_section = radius**2 - float(center[2]) ** 2
                if squared_cross_section <= 0.0:
                    continue
                cross_section_radius = np.sqrt(squared_cross_section)
                add(
                    Cylinder(
                        center=(float(center[0]), float(center[1])),
                        radius=float(cross_section_radius),
                        z_min=0.0,
                        z_max=float(two_dimensional_depth),
                    )
                )
                continue

            add(
                Sphere(
                    center=(float(center[0]), float(center[1]), float(center[2])),
                    radius=radius,
                )
            )
            continue

        sidewall = float(getattr(structure, "sidewall_angle", 0.0) or 0.0)
        z_min, z_max = _z_bounds(structure, design, two_dimensional_depth, padded_size)
        if isinstance(structure, BeamZRectangle) and sidewall == 0.0:
            x_min, y_min, *_ = structure.vertices[0]
            upper = (
                float(x_min + structure.width),
                float(y_min + structure.height),
                z_max,
            )
            add(
                Box(
                    (float(x_min), float(y_min), z_min),
                    _padded_upper(upper, physical_size, padded_size),
                )
            )
            continue

        shell, *holes = polygon_rings(structure)
        vertices = _extend_clipped_polygon(
            shell,
            design_size=physical_size[:2],
            padded_size=padded_size[:2],
        )
        if len(vertices) < 3:
            raise TypeError(f"Unsupported BeamZ structure {type(structure).__name__}.")
        polygon = Polygon(vertices, tuple(holes))
        geometry = _extrude_polygon(
            polygon,
            z_min,
            z_max,
            sidewall,
            float(getattr(structure, "width_to_z", 0.0) or 0.0),
        )
        add(geometry)
    return Scene(tuple(materials), tuple(objects), background_id)
