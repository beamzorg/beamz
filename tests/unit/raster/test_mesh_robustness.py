"""Importer failure handling and independent geometry accuracy regressions."""

import sys
from types import SimpleNamespace

import numpy as np
import pytest

from beamz.design import raster
from beamz.design.raster.importers import from_mesh, from_mesh_arrays
from beamz.design.raster.importers.mesh_import import _tetra_boundary

from .test_geometry_equivalence import extruded_mesh
from .test_mesh import cube, tetrahedron


def fake_mesh(monkeypatch, points, cells, **kwargs):
    data = SimpleNamespace(
        points=points,
        cells=[SimpleNamespace(type=t, data=np.asarray(c)) for t, c in cells],
        cell_data={},
        field_data={},
    )
    for key, value in kwargs.items():
        setattr(data, key, value)
    monkeypatch.setitem(sys.modules, "meshio", SimpleNamespace(read=lambda _: data))


@pytest.mark.parametrize(
    "kind,width",
    [("tetra10", 10), ("triangle6", 6), ("hexahedron", 8), ("quad", 4), ("wedge", 6)],
)
def test_unsupported_cells_cannot_disappear_from_mixed_mesh(monkeypatch, kind, width):
    points, triangles = cube()
    fake_mesh(
        monkeypatch, points, [("triangle", triangles), (kind, [np.zeros(width, int)])]
    )
    with pytest.raises(ValueError, match=f"Unsupported mesh cell types: {kind}"):
        from_mesh("mixed", material=raster.Material(4))


def test_surface_annotations_allowed_but_separate_surfaces_not_dropped(monkeypatch):
    points, triangles = tetrahedron()
    fake_mesh(
        monkeypatch,
        points,
        [
            ("tetra", [[0, 1, 2, 3]]),
            ("triangle", triangles),
            ("line3", [[0, 1, 2]]),
            ("vertex", [[0]]),
        ],
    )
    assert len(from_mesh("annotated", material=raster.Material(4)).objects) == 1
    fake_mesh(
        monkeypatch,
        np.vstack((points, points + 3)),
        [("tetra", [[0, 1, 2, 3]]), ("triangle", triangles + 4)],
    )
    with pytest.raises(ValueError, match="would be lost"):
        from_mesh("mixed", material=raster.Material(4))


@pytest.mark.parametrize(
    "cells,message",
    [
        ([[0, 1, 2, 3], [3, 2, 1, 0]], "Duplicate"),
        ([[0, 1, 2, 2]], "Degenerate"),
        ([[0, 1, 2, 3], [0, 1, 2, 4]], "overlapping"),
    ],
)
def test_invalid_tetrahedral_topology_rejected(cells, message):
    points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [0.1, 0.1, 0.5]])
    with pytest.raises(ValueError, match=message):
        _tetra_boundary(points, np.asarray(cells))


@pytest.mark.parametrize("scale", [1e-9, 1.0, 1e9])
def test_vectorized_boundary_preserves_volume_and_cancels_shared_faces(scale):
    points = (
        np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, -1.0]]) * scale
    )
    # Deliberately use opposite input windings. Both output shells must face out.
    faces = _tetra_boundary(points, np.array([[0, 1, 2, 3], [0, 1, 2, 4]]))
    assert faces.shape == (6, 3)
    report = raster.inspect_mesh(points, faces)
    assert report.valid_for_rasterization
    assert report.signed_volume == pytest.approx(scale**3 / 3, rel=1e-12)


def test_region_priority_can_override_tag_order(tmp_path):
    meshio = pytest.importorskip("meshio")
    points, faces = cube()
    path = tmp_path / "overlap.msh"
    meshio.write(
        path,
        meshio.Mesh(
            np.vstack((points * [0.6, 1, 1], points * [0.6, 1, 1] + [0.4, 0, 0])),
            [("triangle", np.vstack((faces, faces + 8)))],
            cell_data={
                "gmsh:physical": [np.repeat([7, 8], 12)],
                "gmsh:geometrical": [np.repeat([7, 8], 12)],
            },
            field_data={"core": [7, 2], "clad": [8, 2]},
        ),
        file_format="gmsh22",
    )
    materials = {"core": raster.Material(4), "clad": raster.Material(9)}
    grid = raster.Grid.uniform((0, 0, 0), (1, 1, 1), (1, 1, 1))
    for priorities, expected in (
        ({}, 0.4 * 4 + 0.6 * 9),
        ({"core": 20, 8: 0}, 0.6 * 4 + 0.4 * 9),
    ):
        result = raster.rasterize(
            from_mesh(path, materials=materials, priorities=priorities),
            grid,
            options=raster.RasterOptions(smoothing="volume"),
        )
        assert result.tensors["epsilon"][0, 0, 0, 0] == pytest.approx(expected)
    with pytest.raises(ValueError, match="Unknown physical regions"):
        from_mesh(path, materials=materials, priorities={"typo": 1})


@pytest.mark.parametrize("priority", [1.5, True, 2**31, -(2**31) - 1])
def test_invalid_priority_rejected_before_reading(priority):
    with pytest.raises(ValueError, match="32-bit integers"):
        from_mesh("unused", priorities={7: priority})


@pytest.mark.parametrize("origin", [(0, 0), (0, 0, np.nan)])
def test_invalid_coordinate_origin_rejected(origin):
    with pytest.raises(ValueError, match="three finite"):
        from_mesh("unused", coordinate_origin=origin)


def test_coordinate_origin_and_thin_far_translated_mesh(tmp_path):
    meshio = pytest.importorskip("meshio")
    points, faces = cube()
    width = 2.0**-20
    origin = np.array([2.0**20, -(2.0**20), 2.0**20])
    path = tmp_path / "thin.msh"
    meshio.write(
        path,
        meshio.Mesh(points * [width, 1, 1] + origin, [("triangle", faces)]),
        file_format="gmsh22",
    )
    for offset in (np.zeros(3), origin):
        lower = origin - offset
        result = raster.rasterize(
            from_mesh(path, material=raster.Material(1e6), coordinate_origin=offset),
            raster.Grid.uniform(tuple(lower), tuple(lower + 1), (1, 1, 1)),
            options=raster.RasterOptions(smoothing="volume", quality="fast"),
        )
        assert result.tensors["epsilon"][0, 0, 0, 0] == pytest.approx(
            1 + (1e6 - 1) * width, rel=2e-6
        )
        assert result.diagnostics["adaptive_samples"] == 0


def test_curved_mesh_converges_to_circular_cylinder_volume():
    radius = 0.4
    grid = raster.Grid.uniform((0, 0, 0), (1, 1, 1), (8, 8, 2))
    errors = []
    for count in (12, 24, 48, 96):
        theta = np.arange(count) * (2 * np.pi / count)
        exterior = np.column_stack(
            (0.5 + radius * np.cos(theta), 0.5 + radius * np.sin(theta))
        )
        points, faces = extruded_mesh(exterior)
        result = raster.rasterize(
            from_mesh_arrays(points, faces, material=raster.Material(4)),
            grid,
            options=raster.RasterOptions(smoothing="volume", quality="fast"),
        )
        volume = float(np.mean((result.tensors["epsilon"][0].astype(float) - 1) / 3))
        polygon_volume = count * radius**2 * np.sin(2 * np.pi / count) / 2
        assert volume == pytest.approx(polygon_volume, abs=2e-7)
        errors.append(abs(volume - np.pi * radius**2))
    # Polygon approximation error decreases quadratically with edge length.
    ratios = np.asarray(errors[:-1]) / errors[1:]
    assert np.all((ratios > 3.8) & (ratios < 4.2))
    assert errors[-1] < 4e-4


@pytest.mark.parametrize("quality", ["fast", "balanced", "reference"])
def test_nonparallel_material_junction_has_exact_volume(quality):
    grid = raster.Grid.uniform((0, 0, 0), (1, 1, 1), (1, 1, 1))
    objects = []
    for index, upper in enumerate(([0.37, 1, 1], [1, 0.61, 1]), start=1):
        points, faces = cube()
        objects.append(
            raster.Object(raster.Mesh(points * upper, faces), index, priority=index)
        )
    scene = raster.Scene(
        (raster.Material(), raster.Material(4), raster.Material(9)), tuple(objects)
    )
    result = raster.rasterize(
        scene, grid, options=raster.RasterOptions(quality=quality, smoothing="volume")
    )
    expected = (1 - 0.37) * (1 - 0.61) + 4 * 0.37 * (1 - 0.61) + 9 * 0.61
    assert abs(float(result.tensors["epsilon"][0, 0, 0, 0]) - expected) < 2e-6
    assert result.diagnostics["adaptive_samples"] == 0


@pytest.mark.parametrize("nested", [False, True])
@pytest.mark.parametrize("reverse", [False, True])
def test_multiple_shells_have_orientation_independent_parity_volume(nested, reverse):
    points, faces = cube()
    if nested:
        vertices = np.vstack((points, points * 0.5 + 0.25))
        fraction = 1 - 0.5**3
    else:
        vertices = np.vstack(
            (points * [0.25, 1, 1], points * [0.25, 1, 1] + [0.75, 0, 0])
        )
        fraction = 0.5
    second_faces = faces[:, ::-1] if reverse else faces
    result = raster.rasterize(
        from_mesh_arrays(
            vertices, np.vstack((faces, second_faces + 8)), material=raster.Material(4)
        ),
        raster.Grid.uniform((0, 0, 0), (1, 1, 1), (1, 1, 1)),
        options=raster.RasterOptions(smoothing="volume", quality="fast"),
    )
    assert result.tensors["epsilon"][0, 0, 0, 0] == pytest.approx(
        1 + 3 * fraction, rel=2e-6
    )
    assert result.diagnostics["adaptive_samples"] == 0


@pytest.mark.parametrize("scale", [1e-9, 1.0, 1e6])
def test_oblique_material_junction_matches_independent_triangle_areas(scale):
    objects = []
    for index, exterior in enumerate(
        (((0, 0), (1, 0), (0, 1)), ((0, 0), (1, 1), (0, 1))), start=1
    ):
        points, faces = extruded_mesh(exterior)
        objects.append(
            raster.Object(raster.Mesh(points * scale, faces), index, priority=index)
        )
    result = raster.rasterize(
        raster.Scene(
            (raster.Material(), raster.Material(4), raster.Material(9)), tuple(objects)
        ),
        raster.Grid.uniform((0, 0, 0), (scale, scale, scale), (1, 1, 1)),
        options=raster.RasterOptions(quality="fast"),
    )
    np.testing.assert_allclose(
        result.tensors["epsilon"][:, 0, 0, 0], 0.25 + 0.25 * 4 + 0.5 * 9, rtol=2e-6
    )
    assert result.diagnostics["adaptive_samples"] == 0
    assert result.diagnostics["fallback_multiple_orientations"] > 0


def test_origin_is_subtracted_before_unit_conversion(monkeypatch):
    points, faces = cube()
    source_origin = np.array([1e6, -2e6, 3e6])
    fake_mesh(monkeypatch, points * 1000 + source_origin, [("triangle", faces)])
    scene = from_mesh(
        "nm",
        material=raster.Material(4),
        coordinate_origin=source_origin,
        unit_scale=1e-9,
    )
    np.testing.assert_allclose(
        scene.objects[0].geometry.vertices, points * 1e-6, rtol=1e-15
    )


@pytest.mark.parametrize("points", [np.zeros((4, 2)), np.full((4, 3), np.nan)])
def test_invalid_point_arrays_rejected(monkeypatch, points):
    fake_mesh(monkeypatch, points, [("tetra", [[0, 1, 2, 3]])])
    with pytest.raises(ValueError, match="finite array"):
        from_mesh("invalid", material=raster.Material(4))


def test_coordinate_scaling_overflow_is_rejected(monkeypatch):
    points, faces = tetrahedron(1e200)
    fake_mesh(monkeypatch, points, [("triangle", faces)])
    with pytest.raises(ValueError, match="Transformed mesh coordinates"):
        from_mesh("overflow", material=raster.Material(4), unit_scale=1e200)


def test_extra_connectivity_nodes_are_not_silently_trimmed(monkeypatch):
    points, _ = tetrahedron()
    fake_mesh(monkeypatch, points, [("tetra", [[0, 1, 2, 3, 0]])])
    with pytest.raises(ValueError, match="exactly 4"):
        from_mesh("invalid", material=raster.Material(4))
