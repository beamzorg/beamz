"""Small, serializable scene model shared by every raster importer."""

from __future__ import annotations

import json
import operator
from dataclasses import dataclass
from typing import Any

import numpy as np

from beamz.design.grid import RectilinearGrid as Grid
from beamz.design.materials import Material
from beamz.design.structures import Polygon as _Polygon
from beamz.design.structures import Shape

from . import _native  # type: ignore[attr-defined]

_CACHE_SCHEMA_VERSION = 14
_ENGINE_VERSION = str(_native.ENGINE_VERSION)


def _integer(value: Any, name: str, *, minimum: int = 0) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be an integer.")
    try:
        result = operator.index(value)
    except TypeError as exc:
        raise TypeError(f"{name} must be an integer.") from exc
    if result < minimum:
        raise ValueError(f"{name} must be at least {minimum}.")
    return result


def Polygon(exterior, holes=()):
    return _Polygon(exterior, interiors=holes)


def Box(minimum, maximum):
    return Shape("box", bounds=(minimum, maximum))


def Sphere(center, radius):
    return Shape("sphere", position=center, radius=radius)


def Cylinder(center, radius, z_min, z_max):
    return Shape(
        "cylinder",
        position=(*center, z_min),
        z=z_min,
        radius=radius,
        depth=z_max - z_min,
    )


def ExtrudedPolygon(polygon, z_min, z_max):
    return _extrude_polygon(polygon, z_min, z_max)


def TaperedExtrudedPolygon(
    polygon, z_min, z_max, sidewall_angle_degrees, width_to_z=0.0
):
    return _extrude_polygon(polygon, z_min, z_max, sidewall_angle_degrees, width_to_z)


def _extrude_polygon(polygon, z_min, z_max, sidewall=0.0, width_to_z=0.0):
    return _Polygon(
        polygon.vertices,
        interiors=polygon.interiors,
        z=z_min,
        depth=z_max - z_min,
        sidewall_angle=sidewall,
        width_to_z=width_to_z,
    )


def _mesh_arrays(vertices: Any, triangles: Any) -> tuple[np.ndarray, np.ndarray]:
    vertices = np.asarray(vertices, dtype=np.float64)
    triangles = np.asarray(triangles)
    if vertices.ndim != 2 or vertices.shape[1:] != (3,) or len(vertices) == 0:
        raise ValueError("Mesh vertices must have shape (N, 3) with N > 0.")
    if triangles.ndim != 2 or triangles.shape[1:] != (3,):
        raise ValueError("Mesh triangles must have shape (M, 3).")
    if not np.isfinite(vertices).all():
        raise ValueError("Mesh vertices must be finite.")
    if np.issubdtype(triangles.dtype, np.floating):
        if not np.isfinite(triangles).all() or np.any(triangles != np.floor(triangles)):
            raise ValueError("Mesh triangle indices must be finite integers.")
    elif not np.issubdtype(triangles.dtype, np.integer):
        raise TypeError("Mesh triangle indices must use an integer dtype.")
    if np.any(triangles < 0) or np.any(triangles > np.iinfo(np.uint32).max):
        raise ValueError("Mesh triangle indices must fit in uint32.")
    return vertices, triangles.astype(np.uint32, copy=False)


@dataclass(frozen=True, slots=True)
class MeshInspection:
    vertices: int
    triangles: int
    connected_components: int
    boundary_edges: int
    nonmanifold_edges: int
    inconsistent_edges: int
    degenerate_triangles: int
    self_intersections: int
    signed_volume: float
    valid_for_rasterization: bool

    @property
    def topology_is_valid(self) -> bool:
        return not any(
            (
                self.boundary_edges,
                self.nonmanifold_edges,
                self.inconsistent_edges,
                self.degenerate_triangles,
                self.self_intersections,
            )
        )


def inspect_mesh(vertices: Any, triangles: Any) -> MeshInspection:
    """Run the exact native checks used when a scene is compiled."""

    vertex_array, triangle_array = _mesh_arrays(vertices, triangles)
    report = _native.inspect_mesh(vertex_array.tolist(), triangle_array.tolist())
    return MeshInspection(**json.loads(report))


@dataclass(frozen=True, slots=True, eq=False)
class Mesh:
    vertices: np.ndarray
    triangles: np.ndarray

    def validated_arrays(self) -> tuple[np.ndarray, np.ndarray]:
        return _mesh_arrays(self.vertices, self.triangles)

    def to_dict(self) -> dict[str, Any]:
        vertices, triangles = self.validated_arrays()
        return {
            "type": "triangle_mesh",
            "vertices": vertices.tolist(),
            "triangles": triangles.tolist(),
            "bounds": {
                "min": vertices.min(axis=0).tolist(),
                "max": vertices.max(axis=0).tolist(),
            },
        }


Geometry = Shape | Mesh


@dataclass(frozen=True, slots=True)
class Object:
    geometry: Geometry
    material_id: int
    priority: int = 0
    id: int = 0


@dataclass(frozen=True, slots=True)
class Scene:
    materials: tuple[Material, ...]
    objects: tuple[Object, ...] = ()
    background_material: int = 0

    def to_dict(self) -> dict[str, Any]:
        used_ids = {obj.id for obj in self.objects if obj.id}
        next_id = 1
        objects = []
        for obj in self.objects:
            object_id = _integer(obj.id, "Object ID")
            if object_id == 0:
                while next_id in used_ids:
                    next_id += 1
                object_id = next_id
                used_ids.add(object_id)
                next_id += 1
            objects.append(
                {
                    "id": object_id,
                    "material_id": _integer(obj.material_id, "Material ID"),
                    "priority": int(obj.priority),
                    "geometry": obj.geometry.to_dict(),
                }
            )
        return {
            "materials": [material.to_dict() for material in self.materials],
            "objects": objects,
            "background_material": _integer(
                self.background_material, "Background material ID"
            ),
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))

    def rasterize(
        self,
        grid: Grid,
        *,
        options: Any = None,
        cache_directory: Any = None,
    ):
        """Rasterize this reusable scene on an explicit Cartesian grid.

        Returns
        -------
        RasterResult
            Immutable cell and Yee-support material tensors.
        """
        from .engine import rasterize

        return rasterize(
            self,
            grid,
            options=options,
            cache_directory=cache_directory,
        )
