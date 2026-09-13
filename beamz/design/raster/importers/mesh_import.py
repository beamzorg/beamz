from __future__ import annotations

from collections import defaultdict
from numbers import Integral
from pathlib import Path
from typing import Any

import numpy as np

from ..schema import Material, Mesh, Object, Scene


def _cell_indices(
    values: Any,
    *,
    width: int,
    point_count: int,
    cell_type: str,
) -> np.ndarray:
    result = np.asarray(values)
    if result.ndim != 2 or result.shape[1] != width:
        raise ValueError(f"{cell_type} cells must have exactly {width} vertex indices.")
    if np.issubdtype(result.dtype, np.floating):
        if not np.isfinite(result).all() or np.any(result != np.floor(result)):
            raise ValueError(f"{cell_type} cell indices must be finite integers.")
    elif not np.issubdtype(result.dtype, np.integer):
        raise TypeError(f"{cell_type} cell indices must use an integer numeric dtype.")
    if np.any(result < 0) or np.any(result >= point_count):
        raise ValueError(f"{cell_type} cell index is out of range.")
    return result.astype(np.uint32, copy=False)


def _cell_tags(values: Any, *, cell_count: int) -> np.ndarray:
    result = np.asarray(values)
    if result.shape != (cell_count,):
        raise ValueError("Gmsh physical tags must match their cell block length.")
    if np.issubdtype(result.dtype, np.floating):
        if not np.isfinite(result).all() or np.any(result != np.floor(result)):
            raise ValueError("Gmsh physical tags must be finite integers.")
    elif not np.issubdtype(result.dtype, np.integer):
        raise TypeError("Gmsh physical tags must use an integer numeric dtype.")
    return result


def from_mesh_arrays(
    vertices: Any,
    triangles: Any,
    *,
    material: Material,
    background: Material | None = None,
) -> Scene:
    background = Material() if background is None else background
    mesh = Mesh(
        np.asarray(vertices, dtype=np.float64),
        np.asarray(triangles),
    )
    return Scene(
        materials=(background, material),
        objects=(Object(mesh, material_id=1, id=1),),
        background_material=0,
    )


def _tetra_faces(points: np.ndarray, tetrahedra: np.ndarray) -> np.ndarray:
    """Orient all faces in bulk using local, scale-normalized determinants."""
    vertices = points[tetrahedra]
    edges = vertices[:, 1:] - vertices[:, :1]
    scale = np.max(np.abs(edges), axis=(1, 2))
    if np.any(scale == 0):
        raise ValueError("Degenerate tetrahedral cells have zero volume.")
    edges = edges / scale[:, None, None]
    determinant = np.einsum("ij,ij->i", edges[:, 0], np.cross(edges[:, 1], edges[:, 2]))
    tolerance = (
        128 * np.finfo(float).eps * np.prod(np.linalg.norm(edges, axis=2), axis=1)
    )
    if np.any(np.abs(determinant) <= tolerance):
        raise ValueError("Degenerate tetrahedral cells have zero or unresolved volume.")
    faces = tetrahedra[:, [[1, 2, 3], [0, 3, 2], [0, 1, 3], [0, 2, 1]]].copy()
    faces[determinant < 0] = faces[determinant < 0, :, ::-1]
    return faces.reshape(-1, 3)


def _tetra_boundary(points: np.ndarray, tetrahedra: np.ndarray) -> np.ndarray:
    # Sorting replaces four Python cross products and dictionary entries per
    # element. Keep one orientation per face and reject invalid cancellation.
    if len(np.unique(np.sort(tetrahedra, axis=1), axis=0)) != len(tetrahedra):
        raise ValueError("Duplicate tetrahedral cells are not supported.")
    faces = _tetra_faces(points, tetrahedra)
    _, first, inverse, counts = np.unique(
        np.sort(faces, axis=1),
        axis=0,
        return_index=True,
        return_inverse=True,
        return_counts=True,
    )
    inversions = (
        (faces[:, 0] > faces[:, 1]).astype(int)
        + (faces[:, 0] > faces[:, 2])
        + (faces[:, 1] > faces[:, 2])
    )
    balance = np.bincount(inverse, weights=1 - 2 * (inversions % 2))
    if np.any(counts > 2) or np.any((counts == 2) & (balance != 0)):
        raise ValueError("Nonmanifold or overlapping tetrahedral cells share a face.")
    return faces[first[counts == 1]]


def from_mesh(
    path: str | Path,
    *,
    material: Material | None = None,
    materials: dict[str | int, Material] | None = None,
    background: Material | None = None,
    unit_scale: float = 1.0,
    coordinate_origin: Any = (0.0, 0.0, 0.0),
    priorities: dict[str | int, int] | None = None,
) -> Scene:
    """Import linear triangles or tetrahedral physical regions via meshio.

    Coordinates become ``(points - coordinate_origin) * unit_scale``; origin
    is in file units. Explicit priorities use physical names or tags, with
    higher values winning overlaps. Unspecified priorities retain tag order.
    Higher-order and unsupported surface/volume elements are rejected.
    """

    background = Material() if background is None else background
    unit_scale = float(unit_scale)
    if not np.isfinite(unit_scale) or unit_scale <= 0.0:
        raise ValueError("unit_scale must be finite and positive.")
    origin = np.asarray(coordinate_origin, dtype=np.float64)
    if origin.shape != (3,) or not np.isfinite(origin).all():
        raise ValueError("coordinate_origin must contain three finite coordinates.")
    priorities = {} if priorities is None else priorities
    if any(
        not isinstance(value, Integral)
        or isinstance(value, bool)
        or not -(2**31) <= value < 2**31
        for value in priorities.values()
    ):
        raise ValueError("Region priorities must be signed 32-bit integers.")
    try:
        import meshio  # type: ignore[import-not-found]
    except ImportError as exc:
        raise ImportError("Install BeamZ with meshio to import mesh files.") from exc

    data = meshio.read(path)
    points = np.asarray(data.points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise ValueError("Mesh points must be a finite array with shape (N, 3).")
    with np.errstate(over="ignore", invalid="ignore"):
        points = (points - origin) * unit_scale
    if not np.isfinite(points).all():
        raise ValueError("Transformed mesh coordinates must be finite.")
    unsupported = sorted(
        {
            block.type
            for block in data.cells
            if len(block.data)
            and block.type not in {"triangle", "tetra", "vertex"}
            and not block.type.startswith("line")
        }
    )
    if unsupported:
        raise ValueError(
            f"Unsupported mesh cell types: {', '.join(unsupported)}. "
            "Export first-order triangles or tetrahedra; curved/high-order "
            "elements must be tessellated before import."
        )
    material_lookup = materials or {}
    field_names = {
        (int(values[0]), int(values[1]) if len(values) > 1 else -1): name
        for name, values in getattr(data, "field_data", {}).items()
        if len(values) >= 1
    }

    surface_regions: dict[int, list[np.ndarray]] = defaultdict(list)
    surface_blocks: list[np.ndarray] = []
    tetra_blocks: list[tuple[np.ndarray, np.ndarray | None]] = []
    physical_data = data.cell_data.get("gmsh:physical", [])
    for index, block in enumerate(data.cells):
        if not len(block.data):
            continue
        if block.type == "triangle":
            triangles = _cell_indices(
                block.data,
                width=3,
                point_count=len(points),
                cell_type="triangle",
            )
            tags = (
                _cell_tags(physical_data[index], cell_count=len(triangles))
                if index < len(physical_data)
                else None
            )
            if tags is None:
                surface_blocks.append(triangles)
            else:
                for tag in np.unique(tags):
                    surface_regions[int(tag)].append(triangles[tags == tag])
        elif block.type == "tetra":
            tetrahedra = _cell_indices(
                block.data,
                width=4,
                point_count=len(points),
                cell_type=block.type,
            )
            tags = (
                _cell_tags(physical_data[index], cell_count=len(tetrahedra))
                if index < len(physical_data)
                else None
            )
            tetra_blocks.append((tetrahedra, tags))

    if tetra_blocks:
        surface = surface_blocks + [
            block for blocks in surface_regions.values() for block in blocks
        ]
        if surface:
            # Surface cells accompanying a volume mesh must annotate its faces.
            faces = np.concatenate(
                [
                    cells[:, [[1, 2, 3], [0, 3, 2], [0, 1, 3], [0, 2, 1]]].reshape(
                        -1, 3
                    )
                    for cells, _ in tetra_blocks
                ]
            )
            keys = np.unique(np.sort(faces, axis=1), axis=0)
            annotations = np.sort(np.concatenate(surface), axis=1)
            if len(np.unique(np.vstack((keys, annotations)), axis=0)) != len(keys):
                raise ValueError(
                    "Surface cells outside the tetrahedral mesh would be lost; import them separately."
                )
        tetra_regions: dict[int, list[np.ndarray]] = defaultdict(list)
        for tetrahedra, tags in tetra_blocks:
            if tags is None:
                tetra_regions[0].append(tetrahedra)
            else:
                for tag in np.unique(tags):
                    tetra_regions[int(tag)].append(tetrahedra[tags == tag])
        regions = {
            tag: [_tetra_boundary(points, np.concatenate(blocks))]
            for tag, blocks in tetra_regions.items()
        }
        physical_dimension = 3
    else:
        regions = surface_regions
        if surface_blocks:
            regions[0].extend(surface_blocks)
        physical_dimension = 2

    if not regions:
        raise ValueError("The mesh contains no triangle or tetrahedral cells.")

    scene_materials = [background]
    objects = []
    known_regions = set(regions)
    for object_id, (tag, blocks) in enumerate(sorted(regions.items()), start=1):
        name = field_names.get(
            (tag, physical_dimension),
            field_names.get((tag, -1), tag),
        )
        known_regions.add(name)
        region_material = material_lookup.get(name, material_lookup.get(tag, material))
        if region_material is None:
            raise ValueError(
                f"No material configured for mesh physical region {name!r}."
            )
        scene_materials.append(region_material)
        triangles = np.concatenate(blocks)
        # Bounds must describe this physical region, not every region in the
        # source file. Interior tetrahedral nodes are no longer needed either.
        used, remapped = np.unique(triangles, return_inverse=True)
        objects.append(
            Object(
                Mesh(points[used], remapped.reshape(triangles.shape).astype(np.uint32)),
                material_id=len(scene_materials) - 1,
                priority=int(priorities.get(name, priorities.get(tag, object_id))),
                id=object_id,
            )
        )
    if unknown := priorities.keys() - known_regions:
        raise ValueError(
            f"Unknown physical regions in priorities: {sorted(unknown, key=str)!r}."
        )
    return Scene(tuple(scene_materials), tuple(objects), 0)
