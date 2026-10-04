"""Real file readers must preserve boundaries and every constitutive support."""

from types import SimpleNamespace

import numpy as np
import pytest

import beamz as bz
import beamz.design.raster as raster
from beamz.design import import_component, import_gds
from beamz.design.raster.importers import from_gdsfactory, from_mesh, from_mesh_arrays
from beamz.design.raster.importers._beamz import from_beamz

from .test_geometry_equivalence import assert_same_coefficients
from .test_mesh import cube

# meshio probes ASCII STL headers as uint32; NumPy 2 warns on that probe.
pytestmark = pytest.mark.filterwarnings(
    "ignore:overflow encountered in scalar multiply:RuntimeWarning:meshio.stl._stl"
)

# Explicitly tested formats, not a promise about every meshio reader.
FORMATS = [
    ("stl", "stl", {"binary": False}, "triangle"),
    ("stl", "stl", {"binary": True}, "triangle"),
    ("obj", "obj", {}, "triangle"),
    ("off", "off", {}, "triangle"),
    ("ply", "ply", {"binary": False}, "triangle"),
    ("ply", "ply", {"binary": True}, "triangle"),
    ("vtk", "vtk", {"binary": False}, "triangle"),
    ("vtk", "vtk", {"binary": True}, "triangle"),
    ("vtu", "vtu", {}, "triangle"),
    ("msh", "gmsh22", {"binary": False}, "tetra"),
    ("msh", "gmsh22", {"binary": True}, "tetra"),
    ("vtk", "vtk", {"binary": True}, "tetra"),
    ("vtu", "vtu", {}, "tetra"),
    ("mesh", "medit", {}, "tetra"),
    ("inp", "abaqus", {}, "tetra"),
]
TETRA = np.array(
    [[0, 1, 2, 6], [0, 2, 3, 6], [0, 3, 7, 6], [0, 7, 4, 6], [0, 4, 5, 6], [0, 5, 1, 6]]
)


def boundary_measure(vertices, triangles):
    """Translation-stable area and enclosed volume, independent of rasterizer."""
    a, b, c = (vertices[triangles] - vertices.mean(axis=0)).transpose(1, 0, 2)
    return np.array(
        [
            np.linalg.norm(np.cross(b - a, c - a), axis=1).sum() / 2,
            abs(np.einsum("ij,ij->", a, np.cross(b, c))) / 6,
        ]
    )


@pytest.mark.parametrize("extension,file_format,kwargs,cell_type", FORMATS)
@pytest.mark.parametrize("scale", [1e-9, 1.0, 1e6])
@pytest.mark.parametrize("shape", ["box", "sheared", "disconnected"])
def test_file_formats_preserve_boundary_and_all_coefficients(
    tmp_path, extension, file_format, kwargs, cell_type, scale, shape
):
    meshio = pytest.importorskip("meshio")
    vertices, triangles = cube()
    vertices = vertices * [0.5, 0.75, 0.625] + [0.125, 0.125, 0.125]
    cells = triangles if cell_type == "triangle" else TETRA
    if shape == "sheared":
        vertices[:, 0] += vertices[:, 1] * 0.25
    elif shape == "disconnected":
        vertices = np.vstack((vertices * 0.5, vertices * 0.5 + 0.5))
        cells = np.vstack((cells, cells + 8))
        triangles = np.vstack((triangles, triangles + 8))
    # Reorder vertices and cells and reverse all windings; STL also duplicates
    # vertices on disk, so equivalent boundaries need not have identical indices.
    rng = np.random.default_rng(914)
    order = rng.permutation(len(vertices))
    inverse = np.argsort(order)
    encoded = inverse[cells][rng.permutation(len(cells))][:, ::-1]
    path = tmp_path / f"geometry.{extension}"
    meshio.write(
        path,
        meshio.Mesh(vertices[order], [(cell_type, encoded)]),
        file_format=file_format,
        **kwargs,
    )
    material = raster.Material(epsilon_r=4, mu_r=2, conductivity=0.3)
    imported = from_mesh(path, material=material, unit_scale=scale)
    assert len(imported.objects) == 1
    geometry = imported.objects[0].geometry
    decoded = geometry.vertices / scale
    np.testing.assert_allclose(decoded.min(axis=0), vertices.min(axis=0), atol=1e-12)
    np.testing.assert_allclose(decoded.max(axis=0), vertices.max(axis=0), atol=1e-12)
    np.testing.assert_allclose(
        boundary_measure(decoded, geometry.triangles),
        boundary_measure(vertices, triangles),
        rtol=1e-12,
        atol=1e-12,
    )
    direct = from_mesh_arrays(vertices * scale, triangles, material=material)
    grid = raster.Grid(
        x_edges=np.array([0, 0.2, 0.55, 1]) * scale,
        y_edges=np.array([0, 0.3, 0.7, 1]) * scale,
        z_edges=np.array([0, 0.4, 0.65, 1]) * scale,
    )
    for smoothing in ("volume", "farjadpour_diagonal", "farjadpour_full"):
        options = raster.RasterOptions(quality="reference", smoothing=smoothing)
        expected = raster.rasterize(direct, grid, options=options)
        actual = raster.rasterize(imported, grid, options=options)
        assert_same_coefficients(expected, actual)
        if shape == "box":
            analytic = raster.Scene(
                (raster.Material(), material),
                (
                    raster.Object(
                        raster.Box(
                            tuple(vertices.min(0) * scale),
                            tuple(vertices.max(0) * scale),
                        ),
                        1,
                    ),
                ),
            )
            assert_same_coefficients(
                raster.rasterize(analytic, grid, options=options), actual
            )


@pytest.mark.parametrize("offset", [(0, 0), (-3.25, 7.5)])
@pytest.mark.parametrize("unify", [False, True])
@pytest.mark.parametrize("quality", ["fast", "balanced", "reference"])
def test_component_gds_design_and_mesh_share_geometry(
    tmp_path, monkeypatch, offset, unify, quality
):
    gf = pytest.importorskip("gdsfactory")
    meshio = pytest.importorskip("meshio")
    monkeypatch.setattr(gf.pdk, "_ACTIVE_PDK", gf.Pdk(name="cross-format"))
    component = gf.Component()
    polygon = np.array([[0, 0], [0.5, 0], [0.5, 0.75], [0, 0.75]])
    component.add_polygon(polygon + offset, layer=(1, 0))
    path = component.write_gds(tmp_path / "component.gds")
    settings = dict(
        n_core=2,
        n_clad=1,
        core_thickness=0.625e-6,
        clad_below=0,
        clad_above=0,
        xy_padding=0.125e-6,
        unify=unify,
    )
    gf.get_active_pdk().cells["fixture"] = lambda: component
    designs = [
        import_component("fixture", **settings).design,
        import_component(lambda: component, **settings).design,
        import_component(component, **settings).design,
        import_gds(path, **settings).design,
    ]
    vertices, triangles = cube()
    vertices = vertices * [0.5, 0.75, 0.625] + [0.125, 0.125, 0]
    mesh_path = tmp_path / "component.stl"
    meshio.write(
        mesh_path, meshio.Mesh(vertices, [("triangle", triangles)]), binary=True
    )
    scenes = [from_beamz(design) for design in designs]
    scenes.append(from_mesh(mesh_path, material=raster.Material(4), unit_scale=1e-6))
    # The direct layer-stack API retains layout coordinates; explicitly apply
    # the same normalization used by the high-level Design import.
    normalized = gf.Component()
    normalized.add_polygon(polygon + 0.125, layer=(1, 0))
    stack = SimpleNamespace(
        layers={
            "core": SimpleNamespace(
                layer=(1, 0), zmin=0, thickness=0.625, material="core"
            )
        }
    )
    scenes.append(
        from_gdsfactory(normalized, stack, material_map={"core": raster.Material(4)})
    )
    design = bz.Design(
        width=0.75e-6,
        height=1e-6,
        depth=0.625e-6,
        structures=(
            bz.Box(
                center=(0.375e-6, 0.5e-6, 0.3125e-6),
                size=(0.5e-6, 0.75e-6, 0.625e-6),
                material=bz.Material(4),
            ),
        ),
    )
    scenes.append(from_beamz(design))
    grid = raster.Grid.uniform((0, 0, 0), (0.75e-6, 1e-6, 0.625e-6), (3, 4, 2))
    options = raster.RasterOptions(quality=quality)
    expected = raster.rasterize(scenes[-1], grid, options=options)
    for scene in scenes:
        assert_same_coefficients(
            expected, raster.rasterize(scene, grid, options=options)
        )


@pytest.mark.parametrize(
    "file_format,extension", [("gmsh22", "msh"), ("vtk", "vtk"), ("vtu", "vtu")]
)
@pytest.mark.parametrize("reverse_stack", [False, True])
@pytest.mark.parametrize(
    "smoothing", ["volume", "farjadpour_diagonal", "farjadpour_full"]
)
def test_hierarchical_multilayer_gds_and_tagged_mesh(
    tmp_path, monkeypatch, file_format, extension, reverse_stack, smoothing
):
    gf = pytest.importorskip("gdsfactory")
    meshio = pytest.importorskip("meshio")
    monkeypatch.setattr(gf.pdk, "_ACTIVE_PDK", gf.Pdk(name="multilayer-equivalence"))
    child = gf.Component()
    child.add_polygon([(0, 0), (0.5, 0), (0.5, 0.75), (0, 0.75)], layer=(1, 0))
    child.add_polygon([(0.25, 0), (0.75, 0), (0.75, 0.5), (0.25, 0.5)], layer=(2, 0))
    parent = gf.Component()
    reference = parent.add_ref(child)
    reference.drotate(90)
    reference.dmove((1, 0.125))
    path = parent.write_gds(tmp_path / "hierarchy.gds")
    layers = [
        (
            "first",
            SimpleNamespace(
                layer=(1, 0), zmin=0.125, thickness=0.5, material="first", mesh_order=2
            ),
        ),
        (
            "second",
            SimpleNamespace(
                layer=(2, 0), zmin=0.375, thickness=0.5, material="second", mesh_order=1
            ),
        ),
    ]
    stack = SimpleNamespace(layers=dict(layers[::-1] if reverse_stack else layers))
    materials = {
        "first": raster.Material(4, mu_r=2, conductivity=0.3),
        "second": raster.Material(9, mu_r=3, conductivity=0.7),
    }
    scenes = [
        from_gdsfactory(component, stack, material_map=materials)
        for component in (parent, gf.import_gds(path))
    ]
    base, triangles = cube()
    first = base * [0.5, 0.75, 0.5] + [0, 0, 0.125]
    second = base * [0.5, 0.5, 0.5] + [0.25, 0, 0.375]
    points = np.vstack((first, second))
    points[:, :2] = np.column_stack((1 - points[:, 1], points[:, 0] + 0.125))
    mesh_path = tmp_path / f"layers.{extension}"
    meshio.write(
        mesh_path,
        meshio.Mesh(
            points,
            [("triangle", np.vstack((triangles, triangles + 8)))],
            cell_data={"gmsh:physical": [np.repeat([11, 23], len(triangles))]},
        ),
        file_format=file_format,
    )
    scenes.append(
        from_mesh(
            mesh_path,
            materials={11: materials["first"], 23: materials["second"]},
            priorities={11: -2, 23: -1},
            unit_scale=1e-6,
        )
    )
    assert len(scenes[-1].objects) == 2
    expected_scene = raster.Scene(
        (raster.Material(), materials["first"], materials["second"]),
        tuple(
            raster.Object(
                raster.Box(tuple(p.min(0) * 1e-6), tuple(p.max(0) * 1e-6)),
                i + 1,
                priority=i,
            )
            for i, p in enumerate((points[:8], points[8:]))
        ),
    )
    grid = raster.Grid.uniform((0, 0, 0), (1e-6, 1e-6, 1e-6), (3, 3, 3))
    options = raster.RasterOptions(smoothing=smoothing, quality="reference")
    expected = raster.rasterize(expected_scene, grid, options=options)
    for scene in scenes:
        assert_same_coefficients(
            expected, raster.rasterize(scene, grid, options=options)
        )


@pytest.mark.parametrize("binary", [False, True])
def test_stl_decimal_coordinates_respect_file_precision(tmp_path, binary):
    meshio = pytest.importorskip("meshio")
    vertices, triangles = cube()
    vertices = vertices * [0.37, 0.61, 0.83] + [0.13, 0.17, 0.09]
    path = tmp_path / "decimal.stl"
    meshio.write(path, meshio.Mesh(vertices, [("triangle", triangles)]), binary=binary)
    material = raster.Material(4, mu_r=2, conductivity=0.3)
    scene = from_mesh(path, material=material, unit_scale=1e-6)
    # Binary STL stores float32 vertices. Compare against exactly that geometry
    # as well as the requested shape; the format cannot preserve arbitrary f64.
    stored = vertices.astype(np.float32).astype(float) if binary else vertices
    oracle = from_mesh_arrays(stored * 1e-6, triangles, material=material)
    requested = from_mesh_arrays(vertices * 1e-6, triangles, material=material)
    grid = raster.Grid.uniform((0, 0, 0), (1e-6, 1e-6, 1e-6), (3, 3, 3))
    result = raster.rasterize(scene, grid)
    assert_same_coefficients(raster.rasterize(oracle, grid), result)
    assert_same_coefficients(raster.rasterize(requested, grid), result)


@pytest.mark.parametrize(
    "extension,file_format", [("vtu", "vtu"), ("vtk", "vtk"), ("inp", "abaqus")]
)
def test_real_unsupported_volume_file_is_rejected(tmp_path, extension, file_format):
    meshio = pytest.importorskip("meshio")
    vertices, _ = cube()
    path = tmp_path / f"unsupported.{extension}"
    meshio.write(
        path,
        meshio.Mesh(vertices, [("hexahedron", np.arange(8).reshape(1, 8))]),
        file_format=file_format,
    )
    with pytest.raises(ValueError, match="Unsupported mesh cell types: hexahedron"):
        from_mesh(path, material=raster.Material(4))


@pytest.mark.parametrize(
    "extension,file_format", [("obj", "obj"), ("ply", "ply"), ("stl", "stl")]
)
def test_untagged_files_require_explicit_material(tmp_path, extension, file_format):
    meshio = pytest.importorskip("meshio")
    vertices, triangles = cube()
    path = tmp_path / f"untagged.{extension}"
    meshio.write(
        path, meshio.Mesh(vertices, [("triangle", triangles)]), file_format=file_format
    )
    with pytest.raises(ValueError, match="No material configured"):
        from_mesh(path)
