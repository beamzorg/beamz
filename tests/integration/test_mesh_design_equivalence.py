"""Check that imported geometry gives the same time-domain solution."""

from types import SimpleNamespace

import numpy as np
import pytest

import beamz as bz
from beamz.design import MaterialGrid, raster
from beamz.design.raster.importers import from_gdsfactory, from_mesh
from tests.unit.raster.test_mesh import cube


@pytest.mark.parametrize("junction", [False, True])
@pytest.mark.parametrize("kind", ["3d", "tm", "te"])
@pytest.mark.parametrize("explicit_cladding", [False, True])
def test_gmsh_and_design_produce_matching_nonzero_fields(
    tmp_path, kind, explicit_cladding, record_property, monkeypatch, junction
):
    meshio = pytest.importorskip("meshio")
    dimensions = 3 if kind == "3d" else 2
    polarization = "te" if kind == "te" else "tm"
    material = bz.Material(permittivity=4, conductivity=0.1)
    junction_material = bz.Material(permittivity=9, conductivity=0.1)
    vertices, triangles = cube()
    base_triangles = triangles.copy()
    points = vertices * [370, 1000, 1000]
    tags = np.full(len(triangles), 7)
    if explicit_cladding:
        points = np.vstack((points, vertices * [630, 1000, 1000] + [370, 0, 0]))
        triangles = np.vstack((triangles, triangles + 8))
        tags = np.r_[tags, np.full(len(tags), 8)]
    if junction:
        triangles = np.vstack((triangles, base_triangles + len(points)))
        points = np.vstack((points, vertices * [1000, 610, 1000]))
        tags = np.r_[tags, np.full(12, 9)]
    path = tmp_path / "slab.msh"
    meshio.write(
        path,
        meshio.Mesh(
            points,
            [("triangle", triangles)],
            cell_data={"gmsh:physical": [tags], "gmsh:geometrical": [tags]},
        ),
        file_format="gmsh22",
    )
    grid = raster.Grid.uniform(
        (0, 0, 0), (1e-6, 1e-6, 1e-6), (6, 6, 6 if dimensions == 3 else 1)
    )
    mesh_result = raster.rasterize(
        from_mesh(
            path,
            materials={7: material, 8: bz.Material(), 9: junction_material},
            unit_scale=1e-9,
        ),
        grid,
        options=raster.RasterOptions(
            components="all" if dimensions == 3 else f"two_dimensional_{polarization}",
        ),
    )
    imported = MaterialGrid.from_raster_result(
        mesh_result, dimensions=dimensions, polarization=polarization
    )
    structure = (
        bz.Box(
            center=(0.185e-6, 0.5e-6, 0.5e-6),
            size=(0.37e-6, 1e-6, 1e-6),
            material=material,
        )
        if dimensions == 3
        else bz.Rectangle(
            position=(0, 0), width=0.37e-6, height=1e-6, material=material
        )
    )
    structures = [structure]
    if junction:
        structures.append(
            bz.Box(
                center=(0.5e-6, 0.305e-6, 0.5e-6),
                size=(1e-6, 0.61e-6, 1e-6),
                material=junction_material,
            )
            if dimensions == 3
            else bz.Rectangle(
                position=(0, 0), width=1e-6, height=0.61e-6, material=junction_material
            )
        )
    design = bz.Design(
        width=1e-6,
        height=1e-6,
        depth=1e-6 if dimensions == 3 else 0,
        structures=tuple(structures),
    )
    native = design.rasterize(grid, polarization=polarization)
    dt = 0.35 * (1e-6 / 6) / (bz.LIGHT_SPEED * np.sqrt(dimensions))
    # Read an actual GDS file through the public importer as a third route.
    gf = pytest.importorskip("gdsfactory")
    # No generic technology is needed for a polygon on an explicit layer.
    monkeypatch.setattr(gf.pdk, "_ACTIVE_PDK", gf.Pdk(name="raster-proof"))
    component = gf.Component()
    component.add_polygon([(0, 0), (0.37, 0), (0.37, 1), (0, 1)], layer=(1, 0))
    if junction:
        component.add_polygon([(0, 0), (1, 0), (1, 0.61), (0, 0.61)], layer=(2, 0))
    gds_path = component.write_gds(tmp_path / "slab.gds")
    stack = SimpleNamespace(
        layers={
            "core": SimpleNamespace(layer=(1, 0), zmin=0, thickness=1, material="core")
        }
    )
    if junction:
        stack.layers["junction"] = SimpleNamespace(
            layer=(2, 0), zmin=0, thickness=1, material="junction", mesh_order=-1
        )
    gds = MaterialGrid.from_raster_result(
        raster.rasterize(
            from_gdsfactory(
                gf.import_gds(gds_path),
                stack,
                material_map={"core": material, "junction": junction_material},
            ),
            grid,
            options=raster.RasterOptions(
                components="all"
                if dimensions == 3
                else f"two_dimensional_{polarization}"
            ),
        ),
        dimensions=dimensions,
        polarization=polarization,
    )
    # Negative control: the comparison must detect a real permittivity change.
    control = MaterialGrid.from_raster_result(
        raster.rasterize(
            from_mesh(
                path,
                materials={
                    7: bz.Material(permittivity=5, conductivity=0.1),
                    8: bz.Material(),
                    9: junction_material,
                },
                unit_scale=1e-9,
            ),
            grid,
            options=raster.RasterOptions(
                components="all"
                if dimensions == 3
                else f"two_dimensional_{polarization}"
            ),
        ),
        dimensions=dimensions,
        polarization=polarization,
    )
    steps = 512 if junction else 128
    chunk = steps // 4
    signal = np.r_[1.0, np.zeros(steps - 1)]
    trajectories = []
    for materials in (native, imported, gds, control):
        simulation = bz.Simulation(
            material_grid=materials,
            time=np.arange(len(signal)) * dt,
            polarization=polarization,
            boundaries=(bz.PEC(edges="all"),),
            sources=(
                bz.GaussianSource(
                    position=(0.4e-6,) * dimensions,
                    width=0.2e-6,
                    signal=signal,
                ),
            ),
            normalize_source=None,
        )
        state = None
        snapshots = []
        for _ in range(4):
            state = simulation.advance(
                state=state, num_steps=chunk, progress=False, performance=False
            ).state
            assert int(state.current_step) == chunk * (len(snapshots) + 1)
            snapshots.append(
                np.concatenate(
                    [
                        np.asarray(getattr(state, name)).ravel().copy()
                        for name in ("ex", "ey", "ez", "hx", "hy", "hz")
                    ]
                )
            )
        trajectories.append(np.stack(snapshots))
    expected = trajectories[0]
    assert np.isfinite(expected).all()
    assert np.all(np.linalg.norm(expected, axis=1) > 0)
    for route, actual in zip(("gmsh", "gds"), trajectories[1:3], strict=True):
        assert np.isfinite(actual).all()
        relative_l2 = float(
            np.linalg.norm(actual - expected) / np.linalg.norm(expected)
        )
        record_property(f"{route}_relative_l2", relative_l2)
        record_property(
            f"{route}_max_absolute", float(np.max(np.abs(actual - expected)))
        )
        assert relative_l2 < 3e-6
        np.testing.assert_allclose(actual, expected, rtol=3e-6, atol=1e-8)
    assert np.isfinite(trajectories[3]).all()
    control_error = float(
        np.linalg.norm(trajectories[3] - expected) / np.linalg.norm(expected)
    )
    record_property("negative_control_relative_l2", control_error)
    assert control_error > 0.01
