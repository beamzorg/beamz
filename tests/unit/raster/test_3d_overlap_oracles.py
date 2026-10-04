"""True 3D overlap volumes from an independent halfspace/convex-hull oracle."""

from functools import lru_cache
from itertools import combinations

import numpy as np
import pytest
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation

from beamz.design import raster as r

from .test_independent_raster_oracles import assert_support_oracle


def box_halfspaces(lower, upper):
    return np.column_stack(
        (np.vstack((np.eye(3), -np.eye(3))), np.r_[-np.asarray(upper), lower])
    )


def halfspace_volume(planes):
    """Enumerate feasible three-plane intersections; Qhull measures their hull.

    This neither clips polygon faces nor samples material ownership.
    """
    triples = np.array(list(combinations(range(len(planes)), 3)))
    matrices = planes[triples, :3]
    rhs = -planes[triples, 3]
    nonparallel = np.abs(np.linalg.det(matrices)) > 1e-12
    vertices = np.linalg.solve(matrices[nonparallel], rhs[nonparallel, :, None])[..., 0]
    vertices = vertices[
        np.all(vertices @ planes[:, :3].T + planes[:, 3] <= 1e-10, axis=1)
    ]
    vertices = np.unique(vertices, axis=0)
    if (
        len(vertices) < 4
        or np.linalg.matrix_rank(vertices[1:] - vertices[0], tol=1e-11) < 3
    ):
        return 0.0
    return ConvexHull(vertices).volume


def test_halfspace_oracle_matches_cube_and_simplex_formulas():
    cube = box_halfspaces(np.zeros(3), np.ones(3))
    assert halfspace_volume(cube) == pytest.approx(1.0, abs=1e-12)
    assert halfspace_volume(np.vstack((cube, [1.0, 1.0, 1.0, -1.0]))) == pytest.approx(
        1 / 6, abs=1e-12
    )


@pytest.mark.parametrize("seed", range(4))
@pytest.mark.parametrize("scale", [1e-9, 1.0, 1e6])
def test_rotated_3d_tetrahedra_match_halfspace_volume_oracle(seed, scale):
    rng = np.random.default_rng(177 + seed)
    origin = np.array([7.0, -3.0, 11.0]) * scale
    regular = np.array(
        [[1.0, 1.0, 1.0], [1.0, -1.0, -1.0], [-1.0, 1.0, -1.0], [-1.0, -1.0, 1.0]]
    )
    equations, objects = [], []
    for index in range(2):
        points = Rotation.random(random_state=rng).apply(
            regular * rng.uniform(0.32, 0.45)
        ) + rng.uniform(0.4, 0.6, size=3)
        hull = ConvexHull(points)
        equations.append(hull.equations)
        faces = hull.simplices.copy()
        for face, equation in zip(faces, hull.equations, strict=True):
            a, b, c = points[face]
            if np.dot(np.cross(b - a, c - a), equation[:3]) < 0:
                face[:] = face[::-1]
        if seed % 2:
            faces = faces[:, ::-1]
        objects.append(
            r.Object(
                r.Mesh(points * scale + origin, faces),
                index + 1,
                priority=index,
                id=index + 1,
            )
        )
    values = {
        "epsilon": np.array([1.0, 4.0, 9.0]),
        "mu": np.array([1.0, 1.5, 2.0]),
        "conductivity": np.array([0.0, 0.1, 0.7]),
    }
    materials = tuple(
        r.Material(values["epsilon"][i], values["mu"][i], values["conductivity"][i])
        for i in range(3)
    )
    grid = r.Grid.uniform(tuple(origin), tuple(origin + scale), (1, 1, 1))

    @lru_cache(maxsize=32)
    def fractions(lower, upper):
        support = box_halfspaces(lower, upper)
        first, second = (
            halfspace_volume(np.vstack((plane, support))) for plane in equations
        )
        overlap = halfspace_volume(np.vstack((*equations, support)))
        total = np.prod(np.asarray(upper) - lower)
        result = (
            np.array([total - first - second + overlap, first - overlap, second])
            / total
        )
        assert result.min() > -1e-9
        np.testing.assert_allclose(result.sum(), 1.0, atol=1e-12)
        return result

    def oracle(name, lower, upper):
        return np.dot(
            fractions(tuple((lower - origin) / scale), tuple((upper - origin) / scale)),
            values[name],
        )

    result = r.rasterize(
        r.Scene(materials, tuple(objects[::-1])),
        grid,
        options=r.RasterOptions(smoothing="volume", quality="reference"),
    )
    assert_support_oracle(result, grid, oracle)
    assert result.diagnostics["adaptive_samples"] == 0
