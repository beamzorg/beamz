"""Independent spatial oracles for every constitutive support, not route equality."""

import numpy as np
import pytest
from shapely.geometry import Polygon, box

from beamz.design import raster as r
from beamz.design.raster.importers import from_mesh

from .test_geometry_equivalence import extruded_mesh

LOCATIONS = {
    "cell": (False, False, False),
    "ex": (False, True, True),
    "ey": (True, False, True),
    "ez": (True, True, False),
    "hx": (True, False, False),
    "hy": (False, True, False),
    "hz": (False, False, True),
    "node": (True, True, True),
}


def assert_support_oracle(result, grid, oracle, *, rtol=3e-6, atol=3e-7):
    """Integrate an independent reference over every cell and Yee dual volume."""
    for arrays, cell in ((result.tensors, True), (result.yee_tensors, False)):
        for name, array in arrays.items():
            property_name = name if cell else name.rsplit("_", 1)[0]
            stagger = LOCATIONS["cell" if cell else name.rsplit("_", 1)[1]]
            bounds = []
            for edges, on_edge in zip(grid.edges, stagger, strict=True):
                # A dual volume ends at its two neighboring cell centers,
                # clipped to the domain boundary at exterior Yee nodes.
                endpoints = (
                    np.r_[edges[0], (edges[:-1] + edges[1:]) / 2, edges[-1]]
                    if on_edge
                    else edges
                )
                bounds.append(np.column_stack((endpoints[:-1], endpoints[1:])))
            expected = np.empty(array.shape[1:])
            for z, y, x in np.ndindex(expected.shape):
                selected = np.array([bounds[0][x], bounds[1][y], bounds[2][z]])
                expected[z, y, x] = oracle(
                    property_name, selected[:, 0], selected[:, 1]
                )
            assert array.shape[0] == 1
            np.testing.assert_allclose(
                array[0], expected, rtol=rtol, atol=atol, err_msg=name
            )


@pytest.mark.parametrize("seed", range(8))
@pytest.mark.parametrize("scale", [1e-9, 1.0, 1e6])
@pytest.mark.parametrize("representation", ["mesh", "polygon"])
def test_random_overlaps_match_shapely_on_every_yee_support(
    seed, scale, representation
):
    rng = np.random.default_rng(seed)
    rotation = seed % 3 if representation == "mesh" else 0
    origin = np.array([7.0, -3.0, 11.0]) * scale
    local_edges = [
        np.array([0.0, 0.17, 0.48, 0.81, 1.0]),
        np.array([0.0, 0.29, 0.73, 1.0]),
        np.array([0.0, 0.38, 1.0]),
    ]
    grid = r.Grid(
        *(
            local_edges[(axis - rotation) % 3] * scale + origin[axis]
            for axis in range(3)
        )
    )
    values = {
        "epsilon": [1.0, 4.0, 9.0, 2.0],
        "mu": [1.0, 2.0, 1.5, 3.0],
        "conductivity": [0.0, 0.1, 4.0, 0.7],
    }
    materials = tuple(
        r.Material(values["epsilon"][i], values["mu"][i], values["conductivity"][i])
        for i in range(4)
    )
    priorities, identifiers = [1, 1, 0], [40, 10, 80]
    polygons, objects = [], []
    for index in range(3):
        angles = np.arange(3) * 2 * np.pi / 3 + rng.uniform(0, 2 * np.pi)
        center = rng.uniform(0.25, 0.75, size=2)
        radius = rng.uniform(0.45, 0.8)
        vertices = center + radius * np.column_stack((np.cos(angles), np.sin(angles)))
        polygons.append(Polygon(vertices))
        if representation == "mesh":
            points, faces = extruded_mesh(vertices)
            points[:, 2] = 4 * points[:, 2] - 1
            points = np.roll(points, rotation, axis=1) * scale + origin
            if seed % 2:
                faces = faces[:, ::-1]
            faces = faces[rng.permutation(len(faces))]
            geometry = r.Mesh(points, faces)
        else:
            geometry = r.ExtrudedPolygon(
                r.Polygon(tuple(map(tuple, vertices * scale + origin[:2]))),
                origin[2] - scale,
                origin[2] + 3 * scale,
            )
        objects.append(
            r.Object(
                geometry, index + 1, priority=priorities[index], id=identifiers[index]
            )
        )
    # Resolve painter order independently by subtracting the already-owned area.
    remaining = box(0, 0, 1, 1)
    owned = {}
    for index in sorted(
        range(3), key=lambda i: (priorities[i], identifiers[i]), reverse=True
    ):
        owned[index + 1] = remaining.intersection(polygons[index])
        remaining = remaining.difference(polygons[index])
    owned[0] = remaining

    def oracle(name, lower, upper):
        lower, upper = (
            np.roll((lower - origin) / scale, -rotation),
            np.roll((upper - origin) / scale, -rotation),
        )
        support = box(lower[0], lower[1], upper[0], upper[1])
        return (
            sum(
                region.intersection(support).area * values[name][material]
                for material, region in owned.items()
            )
            / support.area
        )

    objects = tuple(objects[index] for index in rng.permutation(3))
    result = r.rasterize(
        r.Scene(materials, objects),
        grid,
        options=r.RasterOptions(smoothing="volume", quality="reference"),
    )
    assert_support_oracle(result, grid, oracle)
    assert result.diagnostics["adaptive_samples"] == 0


def tetrahedral_cube(count):
    """Build a test mesh without depending on checkout-only benchmark scripts."""
    indices = np.indices((count + 1,) * 3).reshape(3, -1).T
    points = indices.astype(float) / count
    origins = np.indices((count,) * 3).reshape(3, -1).T
    corners = np.array(
        [
            [0, 0, 0],
            [1, 0, 0],
            [1, 1, 0],
            [0, 1, 0],
            [0, 0, 1],
            [1, 0, 1],
            [1, 1, 1],
            [0, 1, 1],
        ]
    )
    addresses = origins[:, None, :] + corners
    ids = np.ravel_multi_index(addresses.transpose(2, 0, 1), (count + 1,) * 3)
    pattern = np.array(
        [
            [0, 1, 2, 6],
            [0, 2, 3, 6],
            [0, 3, 7, 6],
            [0, 7, 4, 6],
            [0, 4, 5, 6],
            [0, 5, 1, 6],
        ]
    )
    return points, ids[:, pattern].reshape(-1, 4).astype(np.uint32)


@pytest.mark.parametrize("binary", [False, True])
@pytest.mark.parametrize("scale", [1e-9, 1.0, 1e6])
def test_shuffled_tetrahedral_regions_match_box_integrals(tmp_path, binary, scale):
    meshio = pytest.importorskip("meshio")

    points, cells = tetrahedral_cube(3)
    rng = np.random.default_rng(8701)
    tags = np.repeat(rng.permutation(np.arange(1, 28)), 6)
    cells[::2] = cells[::2, [1, 0, 2, 3]]
    order = rng.permutation(len(cells))
    path = tmp_path / "regions.msh"
    meshio.write(
        path,
        meshio.Mesh(
            points * scale,
            [("tetra", cells[order])],
            cell_data={
                "gmsh:physical": [tags[order]],
                "gmsh:geometrical": [tags[order]],
            },
        ),
        file_format="gmsh22",
        binary=binary,
    )
    values = {
        "epsilon": np.r_[1.0, rng.uniform(1, 12, 27)],
        "mu": np.r_[1.0, rng.uniform(1, 3, 27)],
        "conductivity": np.r_[0.0, rng.uniform(0, 4, 27)],
    }
    materials = {
        tag: r.Material(
            values["epsilon"][tag], values["mu"][tag], values["conductivity"][tag]
        )
        for tag in range(1, 28)
    }
    grid = r.Grid(
        np.array([-0.1, 0.12, 0.51, 0.87, 1.1]) * scale,
        np.array([0.0, 0.3, 0.8, 1.0]) * scale,
        np.array([-0.1, 0.24, 0.68, 1.1]) * scale,
    )
    lower_boxes = np.indices((3, 3, 3)).reshape(3, -1).T / 3
    upper_boxes = lower_boxes + 1 / 3

    def oracle(name, lower, upper):
        lower, upper = lower / scale, upper / scale
        volumes = np.prod(
            np.maximum(
                0, np.minimum(upper_boxes, upper) - np.maximum(lower_boxes, lower)
            ),
            axis=1,
        )
        total = np.prod(upper - lower)
        return (
            values[name][0]
            + np.dot(volumes, values[name][tags[::6]] - values[name][0]) / total
        )

    result = r.rasterize(
        from_mesh(path, materials=materials),
        grid,
        options=r.RasterOptions(smoothing="volume", quality="reference"),
    )
    assert_support_oracle(result, grid, oracle)
    assert result.diagnostics["adaptive_samples"] == 0
