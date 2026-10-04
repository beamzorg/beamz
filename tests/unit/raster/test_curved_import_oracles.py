"""Curved-geometry checks against independently integrated circle cross sections."""

from functools import lru_cache
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.integrate import quad

import beamz as bz
from beamz.design import MaterialGrid
from beamz.design import raster as r
from beamz.design.raster.importers import from_gdsfactory, from_mesh

from .test_geometry_equivalence import assert_same_coefficients, extruded_mesh
from .test_independent_raster_oracles import assert_support_oracle


@lru_cache(maxsize=512)
def circle_rectangle_area(cx, cy, radius, left, bottom, right, top):
    """One-dimensional quadrature of circular chords, independent of raster sampling."""
    left, right = max(left, cx - radius), min(right, cx + radius)
    if right <= left:
        return 0.0
    points = [cx]
    for edge in (bottom, top):
        if abs(edge - cy) < radius:
            dx = np.sqrt(radius**2 - (edge - cy) ** 2)
            points.extend((cx - dx, cx + dx))

    def chord(x):
        dy = np.sqrt(max(0.0, radius**2 - (x - cx) ** 2))
        return max(0.0, min(top, cy + dy) - max(bottom, cy - dy))

    return quad(
        chord,
        left,
        right,
        points=sorted({p for p in points if left < p < right}),
        epsabs=1e-12,
        epsrel=1e-11,
    )[0]


@pytest.mark.parametrize(
    "center,radius", [((0.47, 0.53), 0.36), ((0.35, 0.62), 0.36), ((0.51, 0.48), 0.19)]
)
def test_curved_design_gds_and_mesh_against_circle_integrals(
    tmp_path, monkeypatch, center, radius
):
    meshio = pytest.importorskip("meshio")
    gf = pytest.importorskip("gdsfactory")
    monkeypatch.setattr(gf.pdk, "_ACTIVE_PDK", gf.Pdk(name="curved-oracle"))
    scale = 1e-6
    grid = r.Grid(
        np.array([0.0, 0.2, 0.6, 1.0]) * scale,
        np.array([0.0, 0.3, 0.7, 1.0]) * scale,
        np.array([0.0, 1.0]) * scale,
    )
    material = r.Material(4.0, 1.0, 0.3)
    expected_values = {
        "epsilon": (1.0, 4.0),
        "mu": (1.0, 1.0),
        "conductivity": (0.0, 0.3),
    }

    def oracle(name, lower, upper):
        lower, upper = lower / scale, upper / scale
        fraction = circle_rectangle_area(
            *center, radius, lower[0], lower[1], upper[0], upper[1]
        ) / np.prod((upper - lower)[:2])
        background, core = expected_values[name]
        return background + (core - background) * fraction

    native = r.rasterize(
        r.Scene(
            (r.Material(), material),
            (
                r.Object(
                    r.Cylinder(
                        tuple(np.asarray(center) * scale), radius * scale, 0.0, scale
                    ),
                    1,
                ),
            ),
        ),
        grid,
        options=r.RasterOptions(smoothing="volume", quality="reference"),
    )
    assert_support_oracle(native, grid, oracle, rtol=0, atol=0.025)
    design = bz.Design(
        width=scale,
        height=scale,
        depth=scale,
        structures=(
            bz.Circle(
                position=tuple(np.asarray(center) * scale),
                radius=radius * scale,
                depth=scale,
                points=96,
                material=material,
            ),
        ),
    )
    design_grid = design.rasterize(grid, smoothing="volume", quality="reference")
    # Refine the actual curved surface; don't change grid resolution or quality.
    volume_errors = []
    cell_volumes = (
        np.diff(grid.z_edges)[:, None, None]
        * np.diff(grid.y_edges)[None, :, None]
        * np.diff(grid.x_edges)[None, None, :]
    )
    exact_volume = circle_rectangle_area(*center, radius, 0.0, 0.0, 1.0, 1.0)
    for count in (24, 96):
        angles = np.arange(count) * 2 * np.pi / count
        vertices_2d = np.asarray(center) + radius * np.column_stack(
            (np.cos(angles), np.sin(angles))
        )
        points, faces = extruded_mesh(vertices_2d)
        path = tmp_path / f"circle-{count}.msh"
        meshio.write(
            path,
            meshio.Mesh(points * scale, [("triangle", faces)]),
            file_format="gmsh22",
        )
        result = r.rasterize(
            from_mesh(path, material=material),
            grid,
            options=r.RasterOptions(smoothing="volume", quality="reference"),
        )
        volume = (
            np.sum((result.tensors["epsilon"][0].astype(float) - 1) / 3 * cell_volumes)
            / scale**3
        )
        volume_errors.append(abs(volume - exact_volume))
        if count == 96:
            assert_support_oracle(result, grid, oracle, rtol=0, atol=0.025)
            # Design.Circle is a polygon with an explicit point count, not
            # the analytical cylinder. Compare the same 96-sided geometry.
            mesh_grid = MaterialGrid.from_raster_result(result, dimensions=3)
            for name in ("permittivity", "permeability", "conductivity"):
                np.testing.assert_allclose(
                    getattr(design_grid, name),
                    getattr(mesh_grid, name),
                    rtol=3e-6,
                    atol=3e-7,
                )
            assert design_grid.yee_materials.keys() == mesh_grid.yee_materials.keys()
            for name in design_grid.yee_materials:
                np.testing.assert_allclose(
                    design_grid.yee_materials[name],
                    mesh_grid.yee_materials[name],
                    rtol=3e-6,
                    atol=3e-7,
                )
    assert volume_errors[1] < volume_errors[0] / 8
    assert volume_errors[1] < 4e-4

    # Use the same nm-quantized polygon in both files to isolate file-import effects.
    vertices_2d = np.round(vertices_2d, 3)
    points, faces = extruded_mesh(vertices_2d)
    meshio.write(
        tmp_path / "quantized.msh",
        meshio.Mesh(points * scale, [("triangle", faces)]),
        file_format="gmsh22",
    )
    mesh = r.rasterize(
        from_mesh(tmp_path / "quantized.msh", material=material),
        grid,
        options=r.RasterOptions(smoothing="volume"),
    )
    component = gf.Component()
    component.add_polygon(vertices_2d, layer=(1, 0))
    path = component.write_gds(tmp_path / "circle.gds")
    stack = SimpleNamespace(
        layers={
            "core": SimpleNamespace(layer=(1, 0), zmin=0, thickness=1, material="core")
        }
    )
    gds = r.rasterize(
        from_gdsfactory(gf.import_gds(path), stack, material_map={"core": material}),
        grid,
        options=r.RasterOptions(smoothing="volume"),
    )
    assert_same_coefficients(mesh, gds)
    assert_support_oracle(gds, grid, oracle, rtol=0, atol=0.025)
