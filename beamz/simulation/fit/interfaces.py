"""Geometry and polarized dielectric averaging for planar interfaces.

Geometry is processed on the host. Fractions use exact convex clipping of each
cut Cartesian cell, not supersampling. Tensor construction follows the normal
harmonic / tangential arithmetic dielectric averages (Farjadpour et al.).
"""

from dataclasses import dataclass
from itertools import product

import numpy as np
from scipy.spatial import ConvexHull


def _clip_polygon(vertices, normal, offset):
    clipped = []
    for a, b in zip(vertices, np.roll(vertices, -1, axis=0)):
        da, db = a @ normal - offset, b @ normal - offset
        if da <= 0:
            clipped.append(a)
        if (da < 0 < db) or (db < 0 < da):
            clipped.append(a + da / (da - db) * (b - a))
    return np.asarray(clipped)


def _unit_cell_fraction(normal, offset):
    """Fraction of a unit square/cube satisfying normal dot x <= offset."""
    dimension = len(normal)
    vertices = np.asarray(list(product((0.0, 1.0), repeat=dimension)))
    distances = vertices @ normal - offset
    if np.max(distances) <= 0:
        return 1.0
    if np.min(distances) >= 0:
        return 0.0
    if dimension == 2:
        polygon = _clip_polygon(vertices[[0, 2, 3, 1]], normal, offset)
        if len(polygon) < 3:
            return 0.0
        return float(
            abs(
                np.sum(
                    polygon[:, 0] * np.roll(polygon[:, 1], -1)
                    - polygon[:, 1] * np.roll(polygon[:, 0], -1)
                )
            )
            / 2
        )
    points = [vertex for vertex, distance in zip(vertices, distances) if distance <= 0]
    for i, a in enumerate(vertices):
        for j in range(i + 1, len(vertices)):
            b = vertices[j]
            if np.count_nonzero(a != b) != 1:
                continue
            da, db = distances[i], distances[j]
            if (da < 0 < db) or (db < 0 < da):
                points.append(a + da / (da - db) * (b - a))
    points = np.unique(np.asarray(points), axis=0)
    if len(points) < 4 or np.linalg.matrix_rank(points[1:] - points[0]) < 3:
        return 0.0
    return float(np.clip(ConvexHull(points).volume, 0, 1))


@dataclass(frozen=True)
class FITInterfaceMaterial:
    """Two positive isotropic dielectrics with fractions and interface normals.

    ``fraction_minus`` is cell-shaped, ``normals`` is (..., dimension) in xyz
    order or one constant vector. This prepared geometry contract permits later
    polygon/curved-interface builders without changing the time-step operator.
    """

    fraction_minus: object
    normals: object
    permittivity_minus: float
    permittivity_plus: float

    def prepared(self, mesh):
        fraction = np.asarray(self.fraction_minus, dtype=float)
        if (
            fraction.shape != mesh.shape
            or not np.all(np.isfinite(fraction))
            or np.any((fraction < 0) | (fraction > 1))
        ):
            raise ValueError(
                "Interface fractions must be finite, in [0, 1], and cell-shaped"
            )
        eps = np.asarray([self.permittivity_minus, self.permittivity_plus], dtype=float)
        if not np.all(np.isfinite(eps)) or np.any(eps <= 0):
            raise ValueError("Interface permittivities must be finite and positive")
        normals = np.asarray(self.normals, dtype=float)
        if normals.shape not in ((mesh.dimension,), (*mesh.shape, mesh.dimension)):
            raise ValueError(
                "Normals must be xyz vectors or a cell-shaped vector field"
            )
        normals = np.broadcast_to(normals, (*mesh.shape, mesh.dimension)).copy()
        lengths = np.linalg.norm(normals, axis=-1)
        if not np.all(np.isfinite(normals)) or np.any(lengths == 0):
            raise ValueError("Interface normals must be finite and nonzero")
        normals /= lengths[..., None]
        return fraction, normals, eps

    def tensors(self, mesh):
        """Relative effective permittivity in xyz order and scalar parallel ε."""
        fraction, normals, eps = self.prepared(mesh)
        parallel = fraction * eps[0] + (1 - fraction) * eps[1]
        perpendicular = 1 / (fraction / eps[0] + (1 - fraction) / eps[1])
        if mesh.dimension == 2 and mesh.polarization == "tm":
            # Ez is tangential to every in-plane interface.
            return parallel[..., None, None], parallel
        tensor = parallel[..., None, None] * np.eye(mesh.dimension)
        tensor += (
            (perpendicular - parallel)[..., None, None]
            * normals[..., :, None]
            * normals[..., None, :]
        )
        return tensor, parallel


@dataclass(frozen=True)
class PlanarDielectricInterface:
    """Plane normal dot position = offset in SI coordinates (xyz order).

    The minus dielectric occupies normal dot position <= offset. The vector
    need not be normalized; offset scales with it. A 2D normal is (nx, ny).
    """

    normal: tuple[float, ...]
    offset: float
    permittivity_minus: float
    permittivity_plus: float

    def material(self, mesh):
        normal = np.asarray(self.normal, dtype=float)
        if (
            normal.shape != (mesh.dimension,)
            or not np.all(np.isfinite(normal))
            or np.linalg.norm(normal) == 0
            or not np.isfinite(self.offset)
        ):
            raise ValueError("Plane requires a finite nonzero normal and finite offset")
        length = np.linalg.norm(normal)
        normal, offset = normal / length, float(self.offset) / length
        origins = np.stack(
            np.meshgrid(
                *(np.arange(n) * mesh.spacing for n in mesh.shape), indexing="ij"
            ),
            axis=-1,
        )[..., ::-1]
        local_offset = (offset - origins @ normal) / mesh.spacing
        vertices = np.asarray(list(product((0.0, 1.0), repeat=mesh.dimension)))
        low, high = np.min(vertices @ normal), np.max(vertices @ normal)
        fractions = np.where(local_offset >= high, 1.0, 0.0)
        cut = (local_offset > low) & (local_offset < high)
        for index in zip(*np.nonzero(cut)):
            fractions[index] = _unit_cell_fraction(normal, local_offset[index])
        material = FITInterfaceMaterial(
            fractions, normal, self.permittivity_minus, self.permittivity_plus
        )
        material.prepared(mesh)
        return material
