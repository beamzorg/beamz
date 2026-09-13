"""Representation must not change constitutive coefficients on a fixed grid."""

from types import SimpleNamespace

import numpy as np
import pytest

import beamz as bz
import beamz.design.raster as raster
from beamz.design import MaterialGrid
from beamz.design.raster.importers import from_gdsfactory, from_mesh, from_mesh_arrays

from .test_mesh import cube


def assert_same_coefficients(first, second):
    for group in ("tensors", "yee_tensors"):
        expected, actual = getattr(first, group), getattr(second, group)
        assert expected.keys() == actual.keys()
        for name in expected:
            np.testing.assert_allclose(
                actual[name],
                expected[name],
                rtol=2e-6,
                atol=2e-7,
                err_msg=f"{group}.{name}",
            )


@pytest.mark.parametrize("quality", ["fast", "balanced", "reference"])
@pytest.mark.parametrize(
    "smoothing", ["volume", "farjadpour_diagonal", "farjadpour_full"]
)
@pytest.mark.parametrize("scale", [1e-9, 1.0, 1e6])
@pytest.mark.parametrize("axis", [0, 1, 2])
def test_box_polygon_and_mesh_agree_at_cell_boundaries(quality, smoothing, scale, axis):
    vertices, triangles = cube()
    upper = np.full(3, scale)
    upper[axis] *= 0.37
    material = raster.Material(epsilon_r=4, mu_r=2, conductivity=3)
    geometries = (
        raster.Box((0, 0, 0), tuple(upper)),
        raster.ExtrudedPolygon(
            raster.Polygon(
                ((0, 0), (upper[0], 0), (upper[0], upper[1]), (0, upper[1]))
            ),
            z_min=0,
            z_max=upper[2],
        ),
        raster.Mesh(vertices * upper, triangles),
    )
    grid = raster.Grid.uniform((0, 0, 0), (scale, scale, scale), (2, 2, 2))
    results = [
        raster.rasterize(
            raster.Scene((raster.Material(), material), (raster.Object(geometry, 1),)),
            grid,
            options=raster.RasterOptions(quality=quality, smoothing=smoothing),
        )
        for geometry in geometries
    ]
    for result in results[1:]:
        assert_same_coefficients(results[0], result)


@pytest.mark.parametrize("file_format", ["gmsh22", "gmsh"])
def test_gmsh_distant_physical_region_does_not_disable_smoothing(tmp_path, file_format):
    meshio = pytest.importorskip("meshio")
    vertices, triangles = cube()
    first = vertices * [0.5, 2, 2] - [0, 0.5, 0.5]
    second = vertices + [10, 0, 0]
    tetrahedra = np.array(
        [
            [0, 1, 2, 6],
            [0, 2, 3, 6],
            [0, 3, 7, 6],
            [0, 7, 4, 6],
            [0, 4, 5, 6],
            [0, 5, 1, 6],
        ]
    )
    path = tmp_path / "regions.msh"
    meshio.write(
        path,
        meshio.Mesh(
            np.vstack((first, second)),
            [("tetra", tetrahedra), ("tetra", tetrahedra + 8)],
            point_data={"gmsh:dim_tags": np.repeat([[3, 7], [3, 8]], 8, axis=0)},
            cell_data={
                "gmsh:physical": [np.full(6, 7), np.full(6, 8)],
                "gmsh:geometrical": [np.full(6, 7), np.full(6, 8)],
            },
            field_data={"core": np.array([7, 3]), "distant": np.array([8, 3])},
        ),
        file_format=file_format,
    )
    material = raster.Material(4)
    imported = from_mesh(
        path, materials={"core": material, "distant": raster.Material(9)}
    )
    grid = raster.Grid.uniform((0, 0, 0), (1, 1, 1), (1, 1, 1))
    actual = raster.rasterize(imported, grid)
    expected = raster.rasterize(
        from_mesh_arrays(first, triangles, material=material), grid
    )
    assert_same_coefficients(expected, actual)
    assert actual.tensors["epsilon"][0, 0, 0, 0] == pytest.approx(1.6)
    assert actual.diagnostics["fallback_multiple_objects"] == 0
    for obj in imported.objects:
        assert len(np.unique(obj.geometry.triangles)) == len(obj.geometry.vertices)


def test_direct_and_design_rasterization_defaults_agree():
    design = bz.Design(
        width=1,
        height=1,
        depth=1,
        structures=(
            bz.Box(center=(0.25, 0.5, 0.5), size=(0.5, 1, 1), material=bz.Material(4)),
        ),
    )
    expected = design.rasterize(0.5)
    vertices, triangles = cube()
    result = raster.rasterize(
        from_mesh_arrays(
            vertices * [0.5, 1, 1], triangles, material=raster.Material(4)
        ),
        raster.Grid.uniform((0, 0, 0), (1, 1, 1), (2, 2, 2)),
    )
    actual = MaterialGrid.from_raster_result(result, dimensions=3)
    assert actual.smoothing == expected.smoothing
    for name in ("permittivity", "permeability", "conductivity"):
        np.testing.assert_allclose(getattr(actual, name), getattr(expected, name))
    for name in expected.yee_materials:
        np.testing.assert_allclose(
            actual.yee_materials[name], expected.yee_materials[name]
        )


def extruded_mesh(exterior, holes=()):
    """Triangulate a test prism independently of the native polygon engine."""
    from shapely.geometry import Polygon
    from shapely.geometry.polygon import orient
    from shapely.ops import triangulate

    polygon = orient(Polygon(exterior, holes))
    rings = [list(ring.coords)[:-1] for ring in (polygon.exterior, *polygon.interiors)]
    points = [point for ring in rings for point in ring]
    indices = {tuple(point): index for index, point in enumerate(points)}
    n = len(points)
    faces = []
    for triangle in triangulate(polygon):
        if not polygon.covers(triangle):
            continue
        ids = [
            indices[tuple(point)]
            for point in list(orient(triangle).exterior.coords)[:3]
        ]
        faces.extend((ids[::-1], [index + n for index in ids]))
    start = 0
    for ring in rings:
        for index in range(len(ring)):
            a, b = start + index, start + (index + 1) % len(ring)
            faces.extend(([a, b, b + n], [a, b + n, a + n]))
        start += len(ring)
    vertices = np.array([[x, y, z] for z in (0, 1) for x, y in points])
    return vertices, np.asarray(faces, dtype=np.uint32)


@pytest.mark.parametrize(
    "exterior,holes",
    [
        (((0, 0), (2, 0), (2, 1), (1, 1), (1, 2), (0, 2)), ()),
        (
            ((0, 0), (2, 0), (2, 2), (0, 2)),
            (((0.5, 0.5), (0.5, 1.5), (1.5, 1.5), (1.5, 0.5)),),
        ),
        (((0, 0), (1.5, 0.3), (1.8, 1.5), (0.3, 1.2)), ()),
    ],
)
@pytest.mark.parametrize("reverse", [False, True])
def test_nonconvex_holed_and_oblique_prisms_on_translated_nonuniform_grid(
    exterior, holes, reverse
):
    vertices, triangles = extruded_mesh(exterior, holes)
    scale = 1e-6
    origin = np.array([7, -3, 11]) * scale
    vertices = vertices * scale + origin
    if reverse:
        triangles = triangles[::-1, ::-1]

    def ring(points):
        return tuple(tuple(np.asarray(p) * scale + origin[:2]) for p in points)

    polygon = raster.ExtrudedPolygon(
        raster.Polygon(ring(exterior), tuple(ring(hole) for hole in holes)),
        origin[2],
        origin[2] + scale,
    )
    grid = raster.Grid(
        *[
            origin[axis] + np.array(edges) * scale
            for axis, edges in enumerate(
                (
                    [-0.3, 0, 0.2, 0.5, 0.93, 1.5, 2, 2.3],
                    [-0.3, 0, 0.37, 0.5, 1, 1.5, 2, 2.3],
                    [-0.5, 0, 0.2, 0.8, 1, 1.5],
                )
            )
        ]
    )
    options = raster.RasterOptions(smoothing="farjadpour_full", quality="fast")
    expected = raster.rasterize(
        raster.Scene(
            (raster.Material(), raster.Material(4)), (raster.Object(polygon, 1),)
        ),
        grid,
        options=options,
    )
    actual = raster.rasterize(
        from_mesh_arrays(vertices, triangles, material=raster.Material(4)),
        grid,
        options=options,
    )
    assert_same_coefficients(expected, actual)
    assert actual.diagnostics["adaptive_samples"] == 0


def test_gds_file_and_gmsh_file_match_internal_design(tmp_path, monkeypatch):
    gf = pytest.importorskip("gdsfactory")
    meshio = pytest.importorskip("meshio")
    monkeypatch.setattr(gf.pdk, "_ACTIVE_PDK", gf.Pdk(name="raster-equivalence"))
    component = gf.Component()
    component.add_polygon([(0, 0), (0.37, 0), (0.37, 1), (0, 1)], layer=(1, 0))
    gds_path = component.write_gds(tmp_path / "rectangle.gds")
    imported_component = gf.import_gds(gds_path)
    material = raster.Material(4)
    layer_stack = SimpleNamespace(
        layers={
            "core": SimpleNamespace(
                layer=(1, 0),
                zmin=0,
                thickness=1,
                material="core",
            )
        }
    )
    gds_scene = from_gdsfactory(
        imported_component, layer_stack, material_map={"core": material}
    )
    vertices, triangles = cube()
    mesh_path = tmp_path / "rectangle.msh"
    meshio.write(
        mesh_path,
        meshio.Mesh(vertices * [370, 1000, 1000], [("triangle", triangles)]),
        file_format="gmsh22",
    )
    mesh_scene = from_mesh(mesh_path, material=material, unit_scale=1e-9)
    grid = raster.Grid.uniform((0, 0, 0), (1e-6, 1e-6, 1e-6), (4, 4, 4))
    gds_result, mesh_result = [
        raster.rasterize(scene, grid) for scene in (gds_scene, mesh_scene)
    ]
    assert_same_coefficients(gds_result, mesh_result)
    design = bz.Design(
        width=1e-6,
        height=1e-6,
        depth=1e-6,
        structures=(
            bz.Box(
                center=(0.185e-6, 0.5e-6, 0.5e-6),
                size=(0.37e-6, 1e-6, 1e-6),
                material=material,
            ),
        ),
    )
    expected = design.rasterize(grid)
    actual = MaterialGrid.from_raster_result(mesh_result, dimensions=3)
    np.testing.assert_allclose(actual.permittivity, expected.permittivity, rtol=2e-6)
    for name in expected.yee_materials:
        np.testing.assert_allclose(
            actual.yee_materials[name], expected.yee_materials[name], rtol=2e-6
        )


@pytest.mark.parametrize("polarization", ["tm", "te"])
@pytest.mark.parametrize(
    "smoothing", ["volume", "farjadpour_diagonal", "farjadpour_full"]
)
def test_two_dimensional_mesh_and_polygon_match(polarization, smoothing):
    vertices, triangles = extruded_mesh(((0, 0), (1, 0), (1, 0.37), (0, 0.37)))
    vertices *= [1e-6, 1e-6, 1e-9]
    material = raster.Material(4)
    grid = raster.Grid.uniform((0, 0, 0), (1e-6, 1e-6, 1e-9), (3, 3, 1))
    options = raster.RasterOptions(
        smoothing=smoothing, components=f"two_dimensional_{polarization}"
    )
    expected = raster.rasterize(
        raster.Scene(
            (raster.Material(), material),
            (raster.Object(raster.Box((0, 0, 0), (1e-6, 0.37e-6, 1e-9)), 1),),
        ),
        grid,
        options=options,
    )
    actual = raster.rasterize(
        from_mesh_arrays(vertices, triangles, material=material), grid, options=options
    )
    assert_same_coefficients(expected, actual)


@pytest.mark.parametrize("seed", [14, 82, 191])
def test_polyhedral_rasterization_conserves_independently_computed_volume(seed):
    from scipy.spatial import ConvexHull

    rng = np.random.default_rng(seed)
    points = rng.uniform(-1, 1, (30, 3))
    hull = ConvexHull(points)
    triangles = hull.simplices.copy()
    for face, plane in zip(triangles, hull.equations, strict=True):
        a, b, c = points[face]
        if np.dot(np.cross(b - a, c - a), plane[:3]) < 0:
            face[1], face[2] = face[2], face[1]
    scale = 1e-6
    points *= scale
    grid = raster.Grid.uniform((-1.1e-6,) * 3, (1.1e-6,) * 3, (7, 6, 5))
    result = raster.rasterize(
        from_mesh_arrays(points, triangles, material=raster.Material(4)),
        grid,
        options=raster.RasterOptions(quality="fast", smoothing="volume"),
    )
    fractions = (result.tensors["epsilon"][0] - 1) / 3
    voxel_volume = np.prod([np.diff(edges)[0] for edges in grid.edges])
    occupied_volume = fractions.sum(dtype=np.float64) * voxel_volume
    assert occupied_volume == pytest.approx(hull.volume * scale**3, rel=2e-6, abs=0)
    assert result.diagnostics["adaptive_samples"] == 0


def test_retriangulating_planar_faces_does_not_change_the_raster():
    vertices, triangles = cube()
    vertices[:, 0] = 0.37 * vertices[:, 0] + 0.17 * vertices[:, 2]
    scale = 1e-6
    grid = raster.Grid.uniform((0, 0, 0), (scale, scale, scale), (4, 3, 2))
    options = raster.RasterOptions(quality="fast", smoothing="farjadpour_full")

    def integrate(points, faces):
        return raster.rasterize(
            from_mesh_arrays(
                np.asarray(points) * scale, faces, material=raster.Material(4)
            ),
            grid,
            options=options,
        )

    expected = integrate(vertices, triangles)
    points = list(vertices)
    midpoints = {}

    def midpoint(a, b):
        edge = tuple(sorted((int(a), int(b))))
        if edge not in midpoints:
            midpoints[edge] = len(points)
            points.append((points[a] + points[b]) / 2)
        return midpoints[edge]

    for _ in range(2):
        midpoints.clear()
        refined = []
        for a, b, c in triangles:
            ab, bc, ca = midpoint(a, b), midpoint(b, c), midpoint(c, a)
            refined.extend(((a, ab, ca), (ab, b, bc), (ca, bc, c), (ab, bc, ca)))
        triangles = np.asarray(refined, dtype=np.uint32)
    # Also reverse the global winding and reorder all triangles.
    triangles = np.random.default_rng(21).permutation(triangles[:, ::-1])
    assert_same_coefficients(expected, integrate(points, triangles))


@pytest.mark.parametrize("quality", ["fast", "balanced", "reference"])
@pytest.mark.parametrize(
    "smoothing", ["volume", "farjadpour_diagonal", "farjadpour_full"]
)
@pytest.mark.parametrize("split", [0.5, 0.37])
@pytest.mark.parametrize("cladding", [1.0, 9.0])
def test_adjacent_gmsh_volumes_match_implicit_cladding(
    tmp_path, quality, smoothing, split, cladding
):
    meshio = pytest.importorskip("meshio")
    vertices, triangles = cube()
    tetrahedra = np.array(
        [
            [0, 1, 2, 6],
            [0, 2, 3, 6],
            [0, 3, 7, 6],
            [0, 7, 4, 6],
            [0, 4, 5, 6],
            [0, 5, 1, 6],
        ]
    )
    points = np.vstack(
        (vertices * [split, 1, 1], vertices * [1 - split, 1, 1] + [split, 0, 0])
    )
    path = tmp_path / "adjacent.msh"
    meshio.write(
        path,
        meshio.Mesh(
            points,
            [("tetra", np.vstack((tetrahedra, tetrahedra + 8)))],
            cell_data={
                "gmsh:physical": [np.repeat([7, 8], 6)],
                "gmsh:geometrical": [np.repeat([7, 8], 6)],
            },
        ),
        file_format="gmsh22",
    )
    scene = from_mesh(
        path, materials={7: raster.Material(4), 8: raster.Material(cladding)}
    )
    reference = from_mesh_arrays(
        points[:8],
        triangles,
        material=raster.Material(4),
        background=raster.Material(cladding),
    )
    grid = raster.Grid.uniform((0, 0, 0), (1, 1, 1), (2, 2, 2))
    options = raster.RasterOptions(quality=quality, smoothing=smoothing)
    actual = raster.rasterize(scene, grid, options=options)
    expected = raster.rasterize(reference, grid, options=options)
    assert_same_coefficients(expected, actual)
    assert actual.diagnostics["fallback_multiple_objects"] == 0
    assert actual.diagnostics["adaptive_samples"] == 0


@pytest.mark.parametrize("scale", [1e-9, 1.0, 1e6])
@pytest.mark.parametrize("reverse", [False, True])
def test_adjacent_oblique_regions_match_implicit_cladding(scale, reverse):
    left, lt = extruded_mesh(((0, 0), (0.37, 0), (0.62, 1), (0, 1)))
    right, rt = extruded_mesh(((0.37, 0), (1, 0), (1, 1), (0.62, 1)))
    origin = np.array([7, -3, 11]) * scale
    left, right = left * scale + origin, right * scale + origin
    objects = (
        raster.Object(raster.Mesh(left, lt), 1, id=1),
        raster.Object(raster.Mesh(right, rt), 2, id=2),
    )
    if reverse:
        objects = objects[::-1]
    scene = raster.Scene(
        (raster.Material(), raster.Material(4), raster.Material(9)), objects
    )
    reference = from_mesh_arrays(
        left, lt, material=raster.Material(4), background=raster.Material(9)
    )
    grid = raster.Grid.uniform(tuple(origin), tuple(origin + scale), (3, 2, 2))
    options = raster.RasterOptions(quality="fast", smoothing="farjadpour_full")
    actual = raster.rasterize(scene, grid, options=options)
    expected = raster.rasterize(reference, grid, options=options)
    assert_same_coefficients(expected, actual)
    assert actual.diagnostics["adaptive_samples"] == 0


@pytest.mark.parametrize("scale", [1e-9, 1.0, 1e6])
@pytest.mark.parametrize("cut", [0.31, 0.73, 1.0])
@pytest.mark.parametrize(
    "smoothing", ["volume", "farjadpour_diagonal", "farjadpour_full"]
)
def test_imported_oblique_mesh_matches_analytical_tensor(
    tmp_path, scale, cut, smoothing
):
    """Independent triangle area and laminate formulas, not another raster path."""
    meshio = pytest.importorskip("meshio")
    vertices, triangles = extruded_mesh(((0, 0), (cut, 0), (0, cut)))
    # Extend caps outside the support: the only interior interface is x+y=cut.
    vertices[:, 2] = 3 * vertices[:, 2] - 1
    origin = np.array([7, -3, 11]) * scale
    path = tmp_path / "analytical.msh"
    meshio.write(
        path,
        meshio.Mesh(vertices * scale + origin, [("triangle", triangles)]),
        file_format="gmsh22",
    )
    result = raster.rasterize(
        from_mesh(path, material=raster.Material(4), background=raster.Material(9)),
        raster.Grid.uniform(tuple(origin), tuple(origin + scale), (1, 1, 1)),
        options=raster.RasterOptions(smoothing=smoothing, quality="fast"),
    )
    fraction = cut**2 / 2
    parallel = fraction * 4 + (1 - fraction) * 9
    normal = 1 / (fraction / 4 + (1 - fraction) / 9)
    n = np.array([1, 1, 0]) / np.sqrt(2)
    tensor = parallel * np.eye(3)
    if smoothing != "volume":
        tensor += (normal - parallel) * np.outer(n, n)
    expected = np.array([parallel]) if smoothing == "volume" else tensor.diagonal()
    if smoothing == "farjadpour_full":
        expected = np.r_[expected, tensor[0, 1], tensor[0, 2], tensor[1, 2]]
    np.testing.assert_allclose(
        result.tensors["epsilon"][:, 0, 0, 0], expected, rtol=2e-6, atol=2e-7
    )
    assert result.diagnostics["adaptive_samples"] == 0
