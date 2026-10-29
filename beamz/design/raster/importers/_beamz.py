from __future__ import annotations

from typing import Any

import numpy as np

from beamz._cache_tokens import HashToken, cache_token

from ..schema import (
    Box,
    Cylinder,
    ExtrudedPolygon,
    Material,
    Object,
    Polygon,
    Scene,
    Sphere,
    TaperedExtrudedPolygon,
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


def _high_side(value: float, physical: float, padded: float) -> float:
    value = float(value)
    return float(padded) if _same_coordinate(value, physical) else value


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
            lower = tuple(float(value) for value in structure.lower)
            upper = tuple(float(value) for value in structure.upper)
            if design.depth == 0:
                lower = (lower[0], lower[1], 0.0)
                upper = (upper[0], upper[1], float(two_dimensional_depth))
            elif lower[2] == upper[2]:
                lower = (lower[0], lower[1], 0.0)
                upper = (upper[0], upper[1], padded_size[2])
            add(
                Box(
                    lower,
                    tuple(
                        _high_side(upper[axis], physical_size[axis], padded_size[axis])
                        for axis in range(3)
                    ),
                )
            )
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
        if isinstance(structure, BeamZRectangle) and sidewall == 0.0:
            if design.depth == 0:
                z_min, depth = 0.0, float(two_dimensional_depth)
            else:
                z_min = float(structure.z if structure.z is not None else 0.0)
                depth = float(structure.depth or design.depth)
            x_min, y_min, *_rest = structure.vertices[0]
            add(
                Box(
                    (float(x_min), float(y_min), z_min),
                    (
                        _high_side(
                            float(x_min) + float(structure.width),
                            physical_size[0],
                            padded_size[0],
                        ),
                        _high_side(
                            float(y_min) + float(structure.height),
                            physical_size[1],
                            padded_size[1],
                        ),
                        _high_side(z_min + depth, physical_size[2], padded_size[2]),
                    ),
                )
            )
            continue

        vertices = _extend_clipped_polygon(
            tuple((float(x), float(y)) for x, y, *_ in structure.vertices),
            design_size=physical_size[:2],
            padded_size=padded_size[:2],
        )
        holes = tuple(
            tuple((float(x), float(y)) for x, y, *_ in ring)
            for ring in getattr(structure, "interiors", ())
        )
        if len(vertices) < 3:
            raise TypeError(f"Unsupported BeamZ structure {type(structure).__name__}.")
        z_min = float(getattr(structure, "z", 0.0))
        depth = float(getattr(structure, "depth", 0.0))
        if design.depth == 0:
            z_min = 0.0
            depth = float(two_dimensional_depth)
        elif depth == 0:
            depth = float(design.depth)
        z_max = _high_side(z_min + depth, physical_size[2], padded_size[2])
        polygon = Polygon(vertices, holes)
        geometry = (
            TaperedExtrudedPolygon(
                polygon,
                z_min=z_min,
                z_max=z_max,
                sidewall_angle_degrees=sidewall,
                width_to_z=float(getattr(structure, "width_to_z", 0.0) or 0.0),
            )
            if sidewall != 0.0
            else ExtrudedPolygon(
                polygon,
                z_min=z_min,
                z_max=z_max,
            )
        )
        add(geometry)
    return Scene(tuple(materials), tuple(objects), background_id)
