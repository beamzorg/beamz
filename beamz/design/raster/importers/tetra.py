"""Design implementation."""

from __future__ import annotations

import numpy as np


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
